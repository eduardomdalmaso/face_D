import os
import cv2
import urllib.request
import numpy as np
from src.detectors.base import BaseDetector, DetectionResult

class YOLOv8PersonDetector(BaseDetector):
    def __init__(self, conf_threshold=0.4, nms_threshold=0.45):
        self.model_dir = "data/models"
        self.model_path = os.path.join(self.model_dir, "yolov8n.onnx")
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        self._ensure_model_exists()
        
        # Carrega o modelo no OpenCV DNN
        print(f"[YOLOv8] Carregando modelo do caminho: {self.model_path}")
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        
        # Configuração automática de CPU/GPU
        self.has_cuda = False
        try:
            if hasattr(cv2.dnn, 'getAvailableBackends'):
                backends = cv2.dnn.getAvailableBackends()
                if hasattr(cv2.dnn, 'DNN_BACKEND_CUDA') and cv2.dnn.DNN_BACKEND_CUDA in backends:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    self.has_cuda = True
        except Exception as e:
            print(f"[YOLOv8] Erro ao configurar GPU (CUDA): {e}")
            
        if self.has_cuda:
            print("[YOLOv8] Configurado para usar aceleração de GPU (CUDA) via OpenCV DNN!")
        else:
            print("[YOLOv8] Aceleração de GPU não disponível ou não suportada no OpenCV. Rodando em CPU.")

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_dir, exist_ok=True)
            # URL estável no Hugging Face LFS
            url = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx"
            print(f"[YOLOv8] Modelo ONNX não encontrado localmente. Fazendo download de {url}...")
            
            # User agent mock para evitar bloqueios
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            temp_path = self.model_path + ".tmp"
            try:
                urllib.request.urlretrieve(url, temp_path)
                os.rename(temp_path, self.model_path)
                print("[YOLOv8] Download concluído com sucesso!")
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise RuntimeError(f"Falha ao baixar modelo YOLOv8 ONNX: {e}")

    def detect(self, image_path_or_arr) -> list[DetectionResult]:
        """
        Detecta pessoas na imagem e retorna uma lista de DetectionResult contendo as caixas delimitadoras.
        Aceita um caminho de arquivo (str) ou um array numpy (np.ndarray) representando a imagem.
        """
        try:
            if isinstance(image_path_or_arr, str):
                img = cv2.imread(image_path_or_arr)
            else:
                img = image_path_or_arr

            if img is None or img.size == 0:
                return []

            h, w, _ = img.shape
            scale_x = w / 640.0
            scale_y = h / 640.0

            # Pré-processamento da imagem para YOLOv8 (640x640, BGR para RGB via swapRB=True)
            blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
            self.net.setInput(blob)
            
            # Inferência
            outputs = self.net.forward()
            
            # Formato de saída YOLOv8: (1, 84, 8400)
            # Transpomos para (8400, 84) para facilitar o pós-processamento
            rows = outputs[0].T
            
            boxes = []
            confidences = []
            
            for row in rows:
                # row[4] é o score da classe 0 (pessoa)
                person_score = float(row[4])
                if person_score >= self.conf_threshold:
                    x_center, y_center, box_w, box_h = row[0:4]
                    
                    # Converte de coordenadas centralizadas para canto superior esquerdo (x, y)
                    x = int((x_center - box_w / 2.0) * scale_x)
                    y = int((y_center - box_h / 2.0) * scale_y)
                    bw = int(box_w * scale_x)
                    bh = int(box_h * scale_y)
                    
                    # Limita as coordenadas aos limites da imagem original
                    x = max(0, x)
                    y = max(0, y)
                    bw = min(w - x, bw)
                    bh = min(h - y, bh)
                    
                    if bw > 0 and bh > 0:
                        boxes.append([x, y, bw, bh])
                        confidences.append(person_score)
                        
            # Aplica Non-Maximum Suppression (NMS) para eliminar caixas sobrepostas
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
            
            results = []
            if len(indices) > 0:
                # Trata a variação de formato de retorno do NMSBoxes entre diferentes versões do OpenCV
                flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                for idx in flat_indices:
                    box = boxes[idx]
                    results.append(
                        DetectionResult(
                            x=box[0],
                            y=box[1],
                            w=box[2],
                            h=box[3],
                            confidence=confidences[idx]
                        )
                    )
            return results
            
        except Exception as e:
            print(f"[YOLOv8] Erro durante a detecção de pessoas: {e}")
            return []
