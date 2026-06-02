import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.detectors.base import BaseDetector, DetectionResult

class MediaPipeDetector(BaseDetector):
    def __init__(self, score_threshold=0.5):
        self.model_dir = "data/models"
        self.model_path = os.path.join(self.model_dir, "blaze_face_short_range.tflite")
        self.score_threshold = score_threshold
        self._ensure_model_exists()
        
        # Configurando opções do detector
        use_gpu = os.getenv("USE_GPU", "true").lower() in ("true", "1", "yes")
        delegate = python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU
        
        base_options = python.BaseOptions(
            model_asset_path=self.model_path,
            delegate=delegate
        )
        
        if delegate == python.BaseOptions.Delegate.GPU:
            print("[MediaPipe] Configurado para usar a GPU (CUDA/EGL)!")
        else:
            print("[MediaPipe] Configurado para usar a CPU.")
            
        options = vision.FaceDetectorOptions(
            base_options=base_options, 
            min_detection_confidence=score_threshold
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_dir, exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            print(f"[MediaPipe] Fazendo download do modelo BlazeFace de: {url}")
            urllib.request.urlretrieve(url, self.model_path)
            print("[MediaPipe] Download concluído com sucesso!")

    def detect(self, image_path: str) -> list[DetectionResult]:
        try:
            # MediaPipe requer o objeto mp.Image
            image = mp.Image.create_from_file(image_path)
            detection_result = self.detector.detect(image)
            
            results = []
            for detection in detection_result.detections:
                box = detection.bounding_box
                score = detection.categories[0].score if detection.categories else 1.0
                
                # O MediaPipe pode retornar coordenadas negativas se a detecção estiver no limite da imagem
                x = max(0, int(box.origin_x))
                y = max(0, int(box.origin_y))
                
                results.append(
                    DetectionResult(
                        x=x,
                        y=y,
                        w=int(box.width),
                        h=int(box.height),
                        confidence=float(score)
                    )
                )
            return results
        except Exception as e:
            print(f"Erro ao rodar MediaPipe: {e}")
            return []
