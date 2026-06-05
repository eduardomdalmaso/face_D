#!/bin/bash
echo "=========================================================="
echo "    INICIALIZANDO SERVIDOR DE STREAMING (go2rtc) COM PODMAN "
echo "=========================================================="

# 1. Limpa containers anteriores
echo "[Podman] Parando e removendo containers antigos de streaming..."
podman stop face-go2rtc >/dev/null 2>&1
podman rm face-go2rtc >/dev/null 2>&1
podman stop face-mediamtx >/dev/null 2>&1
podman rm face-mediamtx >/dev/null 2>&1

# 2. Testa se o suporte a GPU NVIDIA (CDI) está configurado no Podman
USE_GPU=false
echo "[Podman] Testando suporte a GPU NVIDIA no Podman..."
if podman run --rm --device nvidia.com/gpu=all docker.io/alpine echo "GPU ok" >/dev/null 2>&1; then
    echo "[Podman] Suporte a GPU NVIDIA CDI detectado com sucesso!"
    USE_GPU=true
else
    echo "[Podman] AVISO: Dispositivo 'nvidia.com/gpu=all' não configurado ou indisponível no Podman."
    echo "[Podman] Executando container em modo de fallback (apenas CPU)."
fi

# 3. Inicializa o go2rtc
echo "[Podman] Iniciando container go2rtc..."
if [ "$USE_GPU" = true ]; then
    podman run -d --name face-go2rtc \
        --network host \
        --device nvidia.com/gpu=all \
        --security-opt label=disable \
        -v "$(pwd)/go2rtc.yaml:/config/go2rtc.yaml:Z" \
        docker.io/alexxit/go2rtc:latest-hardware
else
    podman run -d --name face-go2rtc \
        --network host \
        -v "$(pwd)/go2rtc.yaml:/config/go2rtc.yaml:Z" \
        docker.io/alexxit/go2rtc:latest-hardware
fi

# 4. Verifica status
sleep 2
if podman ps --format '{{.Names}}' | grep -Eq "^face-go2rtc$"; then
    echo "=========================================================="
    echo "    SERVIÇO DE STREAMING INICIALIZADO COM SUCESSO!        "
    echo "    RTSP Ingest/Egress: rtsp://localhost:8554/live        "
    echo "    WebRTC/MSE Console: http://localhost:1984/             "
    echo "=========================================================="
else
    echo "=========================================================="
    echo "    ERRO AO INICIALIZAR CONTAINER go2rtc!                 "
    echo "    Verifique os logs usando: podman logs face-go2rtc     "
    echo "=========================================================="
    exit 1
fi
