from deepface import DeepFace
from src.detectors.base import BaseDetector, DetectionResult

class DeepFaceDetector(BaseDetector):
    def __init__(self, detector_backend='retinaface'):
        self.detector_backend = detector_backend
        # Verifica se o TensorFlow consegue ver a GPU
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"[RetinaFace] GPU(s) detectada(s) pelo TensorFlow: {gpus}. Aceleração habilitada!")
            else:
                print("[RetinaFace] Nenhuma GPU detectada pelo TensorFlow. Rodando em CPU.")
        except Exception as e:
            print(f"[RetinaFace] Erro ao verificar GPU no TensorFlow: {e}")

    def detect(self, image_path: str) -> list[DetectionResult]:
        try:
            # extract_faces retorna uma lista de dicionários
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            results = []
            for face in faces:
                area = face.get("facial_area", {})
                confidence = face.get("confidence", 0.0)
                
                # Se a confiança for muito baixa ou a área for inválida, pulamos
                if not area or confidence is None:
                    continue
                
                # DeepFace às vezes retorna a imagem inteira com confiança baixa se não achar nenhum rosto.
                # Evitamos falsos positivos comparando se a confiança é muito baixa.
                if confidence < 0.1:
                    continue
                
                results.append(
                    DetectionResult(
                        x=int(area.get("x", 0)),
                        y=int(area.get("y", 0)),
                        w=int(area.get("w", 0)),
                        h=int(area.get("h", 0)),
                        confidence=float(confidence)
                    )
                )
            return results
        except Exception as e:
            print(f"Erro ao rodar DeepFace com backend {self.detector_backend}: {e}")
            return []
