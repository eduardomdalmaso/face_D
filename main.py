import os
import sys
import time
import cv2
import multiprocessing

# Importações dos módulos locais
from src.capture import capture_from_webcam
from src.detectors.deepface_det import DeepFaceDetector
from src.detectors.mediapipe_det import MediaPipeDetector
from src.detectors.opencv_yunet import OpenCVYuNetDetector
from src.detectors.opencv_haar import OpenCVHaarDetector

# Mapeamento de cores para cada detector (BGR) e rótulos
DETECTORS_CONFIG = {
    'retinaface': {
        'name': 'RetinaFace (DeepFace)',
        'color': (0, 255, 0),        # Verde
        'class': lambda: DeepFaceDetector(detector_backend='retinaface')
    },
    'mediapipe': {
        'name': 'MediaPipe Face Detector',
        'color': (255, 0, 0),        # Azul
        'class': lambda: MediaPipeDetector(score_threshold=0.5)
    },
    'yunet': {
        'name': 'OpenCV YuNet',
        'color': (0, 0, 255),        # Vermelho
        'class': lambda: OpenCVYuNetDetector(score_threshold=0.6)
    },
    'opencv_haar': {
        'name': 'OpenCV Haar Cascades',
        'color': (0, 255, 255),      # Amarelo
        'class': lambda: OpenCVHaarDetector()
    }
}

def draw_detections_on_image(img, detections, color, label_prefix):
    """Desenha retângulos de detecção e rótulos na imagem."""
    for det in detections:
        # Bounding box
        cv2.rectangle(
            img, 
            (det.x, det.y), 
            (det.x + det.w, det.y + det.h), 
            color, 
            2
        )
        # Rótulo de texto simplificado
        label = f"{label_prefix}: {det.confidence:.0%}"
        cv2.putText(
            img, 
            label, 
            (det.x, det.y - 8), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            color, 
            1,
            cv2.LINE_AA
        )

def run_isolated_detector(key, image_path, queue):
    """Função executada dentro de um subprocesso isolado para rodar um detector."""
    try:
        config = DETECTORS_CONFIG[key]
        detector_instance = config['class']()
        
        # Medição de tempo
        start_time = time.time()
        detections = detector_instance.detect(image_path)
        latency_ms = (time.time() - start_time) * 1000
        
        # Coloca o resultado serializável na fila
        # DetectionResult é um modelo Pydantic, convertemos para dict para evitar problemas de pickle
        serialized_detections = [det.model_dump() for det in detections]
        queue.put((True, serialized_detections, latency_ms))
    except Exception as e:
        queue.put((False, str(e), 0.0))

def main():
    # Garante suporte a spawn limpo no Windows/macOS/Linux
    multiprocessing.set_start_method('spawn', force=True)

    print("=" * 70)
    print("     BENCHMARK COMPARATIVO DE DETECTORES FACIAIS (WEBCAM)")
    print("=" * 70)
    
    # 1. Configurar caminhos
    image_path = "data/test_images/webcam_capture.jpg"
    comparison_image_path = "data/test_images/comparison_result.jpg"
    
    # 2. Captura da Webcam
    print("\n[Passo 1/3] Capturando imagem da webcam...")
    success = capture_from_webcam(image_path)
    if not success:
        print("[Erro] Não foi possível obter a imagem da webcam. Encerrando.")
        sys.exit(1)
        
    # Carrega a imagem base para o desenho comparativo
    comparison_img = cv2.imread(image_path)
    if comparison_img is None:
        print("[Erro] Erro ao carregar a imagem capturada para o benchmark.")
        sys.exit(1)
        
    # 3. Executar o benchmark para cada detector em processos isolados
    print("\n[Passo 2/3] Executando detecção facial com múltiplos frameworks...")
    
    results = {}
    
    for key, config in DETECTORS_CONFIG.items():
        print(f"\n--- Iniciando detector: {config['name']} ---")
        
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=run_isolated_detector, 
            args=(key, image_path, queue)
        )
        
        p.start()
        
        # Aguarda a finalização com timeout de 30 segundos
        p.join(timeout=30)
        
        if p.is_alive():
            print(f"  -> [Aviso] Timeout de 30 segundos atingido para {config['name']}. Terminando processo...")
            p.terminate()
            p.join()
            results[key] = {
                'name': config['name'],
                'count': 'TIMEOUT',
                'latency': 0.0
            }
        elif p.exitcode != 0:
            # Captura segmentation faults (código -11 no Linux) ou outros crashs fatais
            exit_code = p.exitcode
            error_type = "SEGFAULT (EGL/GPU)" if exit_code == -11 or exit_code == 139 else f"CRASH ({exit_code})"
            print(f"  -> [Falha] O detector morreu com erro fatal: {error_type}")
            results[key] = {
                'name': config['name'],
                'count': error_type,
                'latency': 0.0
            }
        else:
            # Processo terminou normalmente
            try:
                success_flag, payload, latency = queue.get_nowait()
                if success_flag:
                    # Deserializa os dicionários de volta para a estrutura de dados original
                    from src.detectors.base import DetectionResult
                    detections = [DetectionResult(**det) for det in payload]
                    
                    results[key] = {
                        'name': config['name'],
                        'count': len(detections),
                        'latency': latency
                    }
                    print(f"  -> Rostos encontrados: {len(detections)}")
                    print(f"  -> Tempo de inferência: {latency:.2f} ms")
                    
                    # Desenhar na imagem comparativa
                    draw_detections_on_image(
                        comparison_img, 
                        detections, 
                        config['color'], 
                        key.upper()
                    )
                else:
                    print(f"  -> [Erro] Falha interna: {payload}")
                    results[key] = {
                        'name': config['name'],
                        'count': 'ERRO',
                        'latency': 0.0
                    }
            except Exception as e:
                print(f"  -> [Erro] Falha ao recuperar resultados da fila: {e}")
                results[key] = {
                    'name': config['name'],
                    'count': 'FALHA_FILA',
                    'latency': 0.0
                }
        
    # Salvar a imagem final contendo todas as bounding boxes que funcionaram
    cv2.imwrite(comparison_image_path, comparison_img)
    
    # 4. Tabela de Comparação de Resultados
    print("\n" + "=" * 70)
    print("                TABELA COMPARATIVA DE RESULTADOS")
    print("=" * 70)
    print(f"{'Detector':<25} | {'Rostos Detectados':<18} | {'Latência (ms)':<15}")
    print("-" * 70)
    for key, data in results.items():
        count_str = str(data['count'])
        latency_str = f"{data['latency']:.2f}" if isinstance(data['count'], int) else "N/A"
        print(f"{data['name']:<25} | {count_str:<18} | {latency_str:<15}")
    print("=" * 70)
    print(f"\nResultado comparativo visual salvo em: {comparison_image_path}")
    print("Legenda de Cores das Caixas:")
    print("  🟢 VERDE    - RetinaFace")
    print("  🔵 AZUL     - MediaPipe")
    print("  🔴 VERMELHO - YuNet")
    print("  🟡 AMARELO  - OpenCV Haar Cascades")
    print("=" * 70)

if __name__ == "__main__":
    # Suporte a multiprocessamento seguro
    main()
