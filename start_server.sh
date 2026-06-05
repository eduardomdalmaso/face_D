#!/bin/bash
# Configura caminhos das bibliotecas CUDA instaladas pelo pip para GPU NVIDIA
export LD_LIBRARY_PATH=$(echo /home/hades/miniconda3/envs/face/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):$LD_LIBRARY_PATH

echo "=========================================================="
echo "    INICIALIZANDO SERVIDOR WEB FACIAL COM ACELERAÇÃO GPU  "
echo "=========================================================="
echo "[CUDA] Mapeado com sucesso para GPU NVIDIA."

# Ativa e inicia o uvicorn com o caminho absoluto do ambiente conda
/home/hades/miniconda3/envs/face/bin/uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
