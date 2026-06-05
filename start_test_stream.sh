#!/bin/bash
CODEC="h264"
FFMPEG_CODEC="libx264"
INPUT="testsrc=size=640x480:rate=30"
INPUT_FORMAT="-f lavfi"

# Processa argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        h265)
            CODEC="h265"
            FFMPEG_CODEC="libx265"
            shift
            ;;
        h264)
            CODEC="h264"
            FFMPEG_CODEC="libx264"
            shift
            ;;
        *)
            INPUT="$1"
            INPUT_FORMAT=""
            shift
            ;;
    esac
done

echo "=========================================================="
echo "    INICIANDO STREAM DE TESTE RTSP ($CODEC) COM FFMPEG     "
echo "    Alvo: rtsp://localhost:8554/live"
if [ -n "$INPUT_FORMAT" ]; then
    echo "    Fonte: Padrão de Teste (testsrc)"
else
    echo "    Fonte: Arquivo de vídeo '$INPUT'"
fi
echo "=========================================================="
echo "Pressione Ctrl+C para encerrar o stream."

if [ -z "$INPUT_FORMAT" ]; then
    ffmpeg -re -stream_loop -1 -i "$INPUT" -c:v $FFMPEG_CODEC -preset ultrafast -pix_fmt yuv420p -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live
else
    ffmpeg -re $INPUT_FORMAT -i "$INPUT" -c:v $FFMPEG_CODEC -preset ultrafast -pix_fmt yuv420p -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live
fi
