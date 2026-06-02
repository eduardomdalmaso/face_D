import cv2
import time
import os

def capture_from_webcam(output_path: str = "data/test_images/webcam_capture.jpg") -> bool:
    """
    Acessa a webcam padrão, aguarda o ajuste de exposição automático,
    captura um único frame e o salva no caminho especificado.
    Não abre janela de visualização (ideal para CLI).
    """
    # Garante que a pasta de destino exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("[Webcam] Inicializando a câmera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[Erro] Não foi possível acessar a webcam. Verifique se ela está conectada e se você possui permissões.")
        return False
    
    # Aguarda um momento para a câmera inicializar e ajustar a exposição automática
    print("[Webcam] Ajustando exposição automática (aguarde 1.5 segundos)...")
    time.sleep(1.5)
    
    # Descarta os primeiros frames para garantir que pegamos um frame atualizado e exposto
    for _ in range(10):
        cap.read()
        
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"[Webcam] Imagem salva com sucesso em: {output_path}")
        success = True
    else:
        print("[Erro] Falha ao capturar frame da webcam.")
        success = False
        
    cap.release()
    return success
