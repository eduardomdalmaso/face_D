#!/bin/bash
echo "=========================================================="
echo "    INICIALIZANDO CONTAINERS DE PRODUÇÃO COM PODMAN      "
echo "=========================================================="

# 1. Cria a rede do Podman se ela não existir
if ! podman network inspect face-net >/dev/null 2>&1; then
    echo "[Podman] Criando rede 'face-net'..."
    podman network create face-net
fi

# Cria diretórios locais se não existirem para evitar erros de volume do Podman
mkdir -p data/registered data/deepface data/models

# 2. Inicia o Qdrant se não estiver rodando
if ! podman ps -a --format '{{.Names}}' | grep -Eq "^face-qdrant$"; then
    echo "[Podman] Iniciando banco Qdrant..."
    podman run -d --name face-qdrant --network face-net -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:Z docker.io/qdrant/qdrant:latest
else
    echo "[Podman] Container Qdrant já existe."
    if ! podman ps --format '{{.Names}}' | grep -Eq "^face-qdrant$"; then
        echo "[Podman] Iniciando Qdrant que estava parado..."
        podman start face-qdrant
    fi
fi

# 3. Builda a imagem da API
echo "[Podman] Construindo imagem face-app..."
podman build -t face-app .

# 4. Inicia a API
if ! podman ps -a --format '{{.Names}}' | grep -Eq "^face-api-app$"; then
    echo "[Podman] Iniciando container da API (face-api-app)..."
    # Nota: Para rodar com aceleração GPU NVIDIA, adicione as flags:
    # --device nvidia.com/gpu=all --security-opt label=disable -e CUDA_VISIBLE_DEVICES=0
    podman run -d --name face-api-app --network face-net \
        -p 8000:8000 \
        -e DATABASE_TYPE=qdrant \
        -e QDRANT_HOST=face-qdrant \
        -e QDRANT_PORT=6333 \
        -v "$(pwd)/data/registered:/app/data/registered:Z" \
        -v "$(pwd)/data/deepface:/root/.deepface:Z" \
        -v "$(pwd)/data/models:/app/data/models:Z" \
        face-app
else
    echo "[Podman] Container face-api-app já existe."
    if ! podman ps --format '{{.Names}}' | grep -Eq "^face-api-app$"; then
        echo "[Podman] Iniciando face-api-app que estava parado..."
        podman start face-api-app
    fi
fi

echo "=========================================================="
echo "    SERVIÇOS DE PRODUÇÃO INICIALIZADOS COM SUCESSO        "
echo "    FastAPI URL: http://localhost:8000                    "
echo "    Qdrant Dashboard: http://localhost:6333/dashboard     "
echo "=========================================================="
