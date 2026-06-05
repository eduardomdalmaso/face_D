import os
import cv2
import urllib.request
import numpy as np
from src.detectors.base import BaseDetector, DetectionResult

class OpenCVYuNetDetector(BaseDetector):
    def __init__(self, score_threshold=0.6, nms_threshold=0.3):
        self.model_dir = "data/models"
        self.model_path = os.path.join(self.model_dir, "face_detection_yunet_2023mar.onnx")
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._ensure_model_exists()
        
        # Criação do detector com OpenCV
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=5000
        )
        
        # Configura CUDA dinamicamente se disponível no OpenCV DNN
        has_cuda = False
        try:
            if hasattr(cv2.dnn, 'getAvailableBackends'):
                backends = cv2.dnn.getAvailableBackends()
                if hasattr(cv2.dnn, 'DNN_BACKEND_CUDA') and cv2.dnn.DNN_BACKEND_CUDA in backends:
                    self.detector.setBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.detector.setTarget(cv2.dnn.DNN_TARGET_CUDA)
                    has_cuda = True
        except Exception as e:
            print(f"[YuNet] Erro ao configurar CUDA no OpenCV DNN: {e}")
            
        if has_cuda:
            print("[YuNet] Configurado para usar a GPU (CUDA) via OpenCV DNN!")
        else:
            print("[YuNet] GPU não disponível no OpenCV. Executando em CPU.")

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_dir, exist_ok=True)
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            print(f"[YuNet] Fazendo download do modelo YuNet de: {url}")
            # User agent mock para evitar erros HTTP 403 se o GitHub rejeitar requisições de script simples
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, self.model_path)
            print("[YuNet] Download concluído com sucesso!")

    def detect(self, image_path_or_arr) -> list[DetectionResult]:
        try:
            if isinstance(image_path_or_arr, str):
                img = cv2.imread(image_path_or_arr)
            else:
                img = image_path_or_arr

            if img is None:
                return []
            
            h, w, _ = img.shape
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(img)
            
            results = []
            if faces is not None:
                for face in faces:
                    # face contém: [x, y, w, h, x_re, y_re, ..., confidence]
                    box = face[0:4].astype(np.int32)
                    confidence = float(face[14])
                    
                    landmarks = {
                        "right_eye": [int(face[4]), int(face[5])],
                        "left_eye": [int(face[6]), int(face[7])],
                        "nose_tip": [int(face[8]), int(face[9])],
                        "mouth_right": [int(face[10]), int(face[11])],
                        "mouth_left": [int(face[12]), int(face[13])]
                    }
                    
                    results.append(
                        DetectionResult(
                            x=int(box[0]),
                            y=int(box[1]),
                            w=int(box[2]),
                            h=int(box[3]),
                            confidence=confidence,
                            landmarks=landmarks
                        )
                    )
            return results
        except Exception as e:
            print(f"Erro ao rodar OpenCV YuNet: {e}")
            return []
