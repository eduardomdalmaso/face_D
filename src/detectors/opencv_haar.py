import cv2
import os
from src.detectors.base import BaseDetector, DetectionResult

class OpenCVHaarDetector(BaseDetector):
    def __init__(self):
        xml_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Arquivo XML Haar Cascades não encontrado: {xml_path}")
        self.face_cascade = cv2.CascadeClassifier(xml_path)

    def detect(self, image_path: str) -> list[DetectionResult]:
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detecta faces com parâmetros padrão equilibrados
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            results = []
            for (x, y, w, h) in faces:
                results.append(
                    DetectionResult(
                        x=int(x),
                        y=int(y),
                        w=int(w),
                        h=int(h),
                        confidence=1.0  # Haar cascades não retorna pontuação contínua direta, assumimos 1.0
                    )
                )
            return results
        except Exception as e:
            print(f"Erro ao rodar OpenCV Haar Cascades: {e}")
            return []
