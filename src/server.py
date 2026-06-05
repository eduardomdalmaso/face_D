import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import time
import base64
import cv2
import numpy as np
import json
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import threading
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.database import (
    init_db, register_user, get_all_users, 
    add_history_entry, get_history, cosine_similarity, delete_user,
    add_pending_review, get_pending_reviews, delete_pending_review
)
from src.detectors.opencv_yunet import OpenCVYuNetDetector
from src.detectors.deepface_det import DeepFaceDetector
from src.detectors.yolo_det import YOLOv8PersonDetector
from deepface import DeepFace

# Inicializações
app = FastAPI(title="Pipeline de Detecção & Reconhecimento Facial")
init_db()

# Diretórios necessários
REGISTERED_DIR = "data/registered"
PENDING_DIR = "data/pending_review"
HISTORY_DIR = "data/history"
os.makedirs(REGISTERED_DIR, exist_ok=True)
os.makedirs(PENDING_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs("data/test_images", exist_ok=True)

# Habilita CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monta a pasta de fotos cadastradas como estática para servir os snapshots
app.mount("/registered_images", StaticFiles(directory=REGISTERED_DIR), name="registered_images")
app.mount("/pending_images", StaticFiles(directory=PENDING_DIR), name="pending_images")
app.mount("/history_images", StaticFiles(directory=HISTORY_DIR), name="history_images")

# Inicializa os detectores
detector_yunet = OpenCVYuNetDetector(score_threshold=0.6)
detector_retinaface = DeepFaceDetector(detector_backend='retinaface')
detector_yolo = YOLOv8PersonDetector()

# Cache de usuários cadastrados
registered_users = get_all_users()

# Variáveis globais para classificação SVM e controle de spam de revisão
clf_model = None
label_to_name = []
last_review_time = {}

def train_classifier():
    global clf_model, label_to_name
    print("[SVM] Iniciando treinamento do classificador...")
    users = get_all_users()
    
    X = []
    y = []
    label_to_name_temp = []
    
    for label_idx, user in enumerate(users):
        name = user["name"]
        templates = user["templates"]
        
        if templates:
            label_to_name_temp.append(name)
            idx = len(label_to_name_temp) - 1
            for t in templates:
                X.append(t["embedding"])
                y.append(idx)
                
    num_classes = len(label_to_name_temp)
    print(f"[SVM] Encontradas {num_classes} classes com {len(X)} templates no total.")
    
    if num_classes >= 2 and len(X) >= 2:
        try:
            from sklearn.svm import SVC
            svc = SVC(kernel='linear', probability=True, C=1.0, random_state=42)
            svc.fit(np.array(X), np.array(y))
            clf_model = svc
            label_to_name = label_to_name_temp
            print("[SVM] Classificador treinado com sucesso!")
            return True
        except Exception as e:
            print(f"[SVM] Erro ao treinar classificador: {e}")
            clf_model = None
            label_to_name = []
            return False
    else:
        print("[SVM] Dados insuficientes para treinar o classificador (necessário pelo menos 2 usuários com templates). Fallback para distância cosseno.")
        clf_model = None
        label_to_name = []
        return False

# Treina no startup
train_classifier()

def refresh_user_cache():
    global registered_users
    registered_users = get_all_users()
    train_classifier()

def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decodifica imagem em base64 recebida do navegador em um frame OpenCV (BGR)."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

class RecognizeRequest(BaseModel):
    image_base64: str
    detector: str = "auto" # Pode ser "auto", "yunet" ou "retinaface"
    threshold: float = 0.68 # Limiar de distância (0.00 a 1.00)
    extract_attributes: bool = False # Se True, analisa idade, gênero e emoções

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serviço da interface web HTML principal."""
    # Retorna o arquivo templates/index.html
    html_path = "templates/index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>Arquivo templates/index.html não encontrado.</h3>", status_code=404)

@app.post("/register")
async def register_new_face(name: str = Form(...), file: UploadFile = File(...)):
    """Recebe um nome e uma foto, calcula o embedding e salva no SQLite."""
    # Garante nome limpo
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Nome não pode ser vazio.")
        
    file_extension = os.path.splitext(file.filename)[1]
    if not file_extension:
        file_extension = ".jpg"
        
    # Salva a imagem temporariamente para processar no DeepFace
    temp_path = f"data/temp_register_{int(time.time())}{file_extension}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        # Extrai o embedding usando ArcFace (detector_backend RetinaFace para alta precisão no cadastro)
        print(f"[FastAPI] Gerando embedding de cadastro para {name}...")
        representations = DeepFace.represent(
            img_path=temp_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )
        
        if not representations or len(representations) == 0:
            raise ValueError("Nenhum rosto claro foi detectado na imagem de cadastro.")
            
        embedding = representations[0]["embedding"]
        
        # Salva na pasta permanente
        permanent_image_path = os.path.join(REGISTERED_DIR, f"{name}{file_extension}")
        os.rename(temp_path, permanent_image_path)
        
        # Salva no banco de dados SQLite
        success = register_user(name, embedding, permanent_image_path)
        if success:
            refresh_user_cache()
            return {"status": "success", "message": f"Usuário {name} cadastrado com sucesso!"}
        else:
            raise HTTPException(status_code=500, detail="Falha ao salvar no banco de dados.")
            
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=f"Erro no cadastro: {str(e)}")

def process_recognition_frame(img: np.ndarray, detector_choice: str, threshold_val: float, extract_attrs: bool):
    """
    Função auxiliar para processar um único frame de imagem: detecta faces, extrai embeddings,
    calcula distâncias cosseno, realiza predição com SVM híbrido, lida com upgrade de qualidade
    e filas de aprendizado ativo.
    """
    h, w, _ = img.shape
    
    # 1. Detectores em cascata precedidos por detecção de pessoas (YOLOv8)
    people = detector_yolo.detect(img)
    
    detections = []
    detector_used = detector_choice
    
    for person in people:
        # Adiciona margem de 10% para garantir que a cabeça/rosto não sejam cortados
        margin_x = int(person.w * 0.1)
        margin_y = int(person.h * 0.1)
        
        x1 = max(0, person.x - margin_x)
        y1 = max(0, person.y - margin_y)
        x2 = min(w, person.x + person.w + margin_x)
        y2 = min(h, person.y + person.h + margin_y)
        
        person_crop = img[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue
            
        person_detections = []
        if detector_choice == "retinaface":
            person_detections = detector_retinaface.detect(person_crop)
            detector_used = "retinaface"
        elif detector_choice == "yunet":
            person_detections = detector_yunet.detect(person_crop)
            detector_used = "yunet"
        else: # auto
            person_detections = detector_yunet.detect(person_crop)
            detector_used = "yunet"
            if len(person_detections) == 0:
                person_detections = detector_retinaface.detect(person_crop)
                detector_used = "retinaface"
                
        # Remapeia coordenadas de detecção e landmarks da face para o frame original
        for face_det in person_detections:
            face_det.x += x1
            face_det.y += y1
            if face_det.landmarks:
                for lm_name, lm_coord in face_det.landmarks.items():
                    face_det.landmarks[lm_name] = [lm_coord[0] + x1, lm_coord[1] + y1]
            detections.append(face_det)
        
    # 2. Extrair embeddings de todos os rostos detectados
    extracted_faces = []
    for face_idx, det in enumerate(detections):
        margin_x = int(det.w * 0.1)
        margin_y = int(det.h * 0.1)
        
        x1 = max(0, det.x - margin_x)
        y1 = max(0, det.y - margin_y)
        x2 = min(w, det.x + det.w + margin_x)
        y2 = min(h, det.y + det.h + margin_y)
        
        cropped_face = img[y1:y2, x1:x2]
        if cropped_face.size == 0:
            continue
            
        try:
            # Extrai embedding do crop (detector_backend='skip' para velocidade, pois já detectamos)
            rep = DeepFace.represent(
                img_path=cropped_face,
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False
            )
            if rep and len(rep) > 0:
                face_emb = rep[0]["embedding"]
                extracted_faces.append({
                    "face_idx": face_idx,
                    "det": det,
                    "cropped_face": cropped_face,
                    "face_emb": face_emb
                })
        except Exception as e:
            print(f"[Reconhecimento] Erro ao extrair embedding da face #{face_idx}: {e}")

    # 3. Processa cada face extraída para classificação Híbrida (SVM + Cosine Distance)
    results_by_face = {}
    cache_needs_refresh = False
    
    for f_data in extracted_faces:
        face_idx = f_data["face_idx"]
        face_emb = f_data["face_emb"]
        cropped_face = f_data["cropped_face"]
        
        # 3.1. Calcular distância de cosseno mínima contra todos os templates de todos os usuários
        min_dist_any = 999.0
        nearest_user_name = "Desconhecido"
        
        for user in registered_users:
            for template in user["templates"]:
                sim = cosine_similarity(face_emb, template["embedding"])
                dist = 1.0 - sim
                if dist < min_dist_any:
                    min_dist_any = dist
                    nearest_user_name = user["name"]
                    
        # 3.2. Predição usando o classificador SVM se treinado
        svm_name = None
        svm_prob = 0.0
        
        if clf_model is not None:
            try:
                probs = clf_model.predict_proba([face_emb])[0]
                max_idx = np.argmax(probs)
                svm_name = label_to_name[max_idx]
                svm_prob = float(probs[max_idx])
            except Exception as clf_err:
                print(f"[SVM] Erro ao predizer face #{face_idx}: {clf_err}")
                
        # 3.3. Lógica de decisão híbrida
        name = "Desconhecido"
        confidence = 0.0
        
        if min_dist_any <= threshold_val:
            # Reconhecido pelo banco de dados
            if svm_name is not None and svm_prob >= 0.55:
                # SVM possui confiança no limite estipulado
                name = svm_name
                confidence = svm_prob
            else:
                # Caso contrário, fallback para similaridade cosseno mais próxima
                name = nearest_user_name
                confidence = 1.0 - min_dist_any
        else:
            name = "Desconhecido"
            confidence = 0.0
            
        results_by_face[face_idx] = {
            "name": name,
            "confidence": confidence,
            "min_dist_any": min_dist_any,
            "nearest_user_name": nearest_user_name,
            "svm_name": svm_name,
            "svm_prob": svm_prob,
            "face_emb": face_emb,
            "cropped_face": cropped_face
        }
        
    # Processa atualizações de qualidade automática (upgrade) e constrói a resposta final
    identified_faces = []
    detected_names = []
    
    for face_idx, det in enumerate(detections):
        name = "Desconhecido"
        confidence_match = 0.0
        attributes = None
        
        if face_idx in results_by_face:
            res = results_by_face[face_idx]
            name = res["name"]
            confidence_match = res["confidence"]
            min_dist_any = res["min_dist_any"]
            nearest_user_name = res["nearest_user_name"]
            svm_name = res["svm_name"]
            svm_prob = res["svm_prob"]
            face_emb = res["face_emb"]
            cropped_face = res["cropped_face"]
            
            # Análise de atributos extras se solicitado
            if extract_attrs and cropped_face.size > 0:
                try:
                    analysis = DeepFace.analyze(
                        img_path=cropped_face,
                        actions=['age', 'gender', 'emotion'],
                        enforce_detection=False,
                        detector_backend='skip'
                    )
                    if analysis and len(analysis) > 0:
                        item = analysis[0]
                        attributes = {
                            "age": int(item.get("age", 0)),
                            "gender": item.get("dominant_gender", "Unknown"),
                            "emotion": item.get("dominant_emotion", "Unknown")
                        }
                except Exception as attr_err:
                    print(f"[Atributos] Erro ao extrair atributos da face #{face_idx}: {attr_err}")
            
            # 3.4. Auto-aprendizado / Upgrade de Qualidade
            # Se a face é reconhecida e a distância é de excelente qualidade (dist <= 0.50)
            if name != "Desconhecido" and min_dist_any <= 0.50:
                user_entry = next((u for u in registered_users if u["name"] == name), None)
                if user_entry and len(user_entry["templates"]) < 10:
                    # Verifica se o novo embedding adiciona informação de pose (distância para templates atuais > 0.35)
                    min_dist_user = min(1.0 - cosine_similarity(face_emb, t["embedding"]) for t in user_entry["templates"])
                    if min_dist_user > 0.35:
                        print(f"[Qualidade] Adicionando novo template de pose para '{name}' (Mín Dist: {min_dist_user:.4f})")
                        img_filename = f"{name}_{int(time.time() * 1000)}.jpg"
                        img_path = os.path.join(REGISTERED_DIR, img_filename)
                        cv2.imwrite(img_path, cropped_face)
                        register_user(name, face_emb, img_path, quality_score=min_dist_any)
                        cache_needs_refresh = True
                        
            # 3.5. Aprendizado Ativo (Active Learning)
            # Se a similaridade é limítrofe (distância entre 0.50 e 0.70) e a confiança da detecção é alta (>= 0.80) para evitar falsos positivos
            if 0.50 <= min_dist_any <= 0.70 and det.confidence >= 0.80:
                suggested_name = svm_name if svm_name else nearest_user_name
                conf_val = svm_prob if svm_name else (1.0 - min_dist_any)
                
                # Controle de spam: limite de 10 segundos por pessoa sugerida para evitar entupir fila
                now = time.time()
                if now - last_review_time.get(suggested_name, 0) > 10.0:
                    last_review_time[suggested_name] = now
                    print(f"[Active Learning] Rosto suspeito (Dist: {min_dist_any:.4f}) para '{suggested_name}'. Salvando para revisão...")
                    
                    review_filename = f"pending_{uuid.uuid4().hex[:12]}.jpg"
                    review_path = os.path.join(PENDING_DIR, review_filename)
                    cv2.imwrite(review_path, cropped_face)
                    add_pending_review(face_emb, review_path, suggested_name, conf_val)
        
        detected_names.append(name)
        identified_faces.append({
            "box": {"x": det.x, "y": det.y, "w": det.w, "h": det.h},
            "name": name,
            "confidence": confidence_match,
            "landmarks": det.landmarks,
            "attributes": attributes
        })
        
    if cache_needs_refresh:
        refresh_user_cache()
        
    return identified_faces, detected_names, detector_used

@app.post("/recognize")
async def recognize_faces_in_frame(payload: RecognizeRequest):
    """
    Recebe um frame em base64 da webcam do cliente, detecta os rostos via cascata/detector selecionado,
    calcula os embeddings e compara por similaridade cosseno com o SQLite.
    """
    start_time = time.time()
    try:
        img = decode_base64_image(payload.image_base64)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Imagem inválida."})
            
        identified_faces, detected_names, detector_used = process_recognition_frame(
            img=img,
            detector_choice=payload.detector,
            threshold_val=payload.threshold,
            extract_attrs=payload.extract_attributes
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Salva no histórico se houve alguma detecção
        if len(detected_names) > 0:
            annotated_img = img.copy()
            for face in identified_faces:
                box = face["box"]
                name = face["name"]
                conf = face["confidence"]
                x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                is_unknown = (name == "Desconhecido")
                color = (0, 0, 255) if is_unknown else (0, 255, 135)
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
                label = f"{name} ({(conf * 100):.0f}%)" if not is_unknown else "Desconhecido"
                (w_t, h_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_img, (x, y - h_t - 8), (x + w_t, y), color, -1)
                cv2.putText(annotated_img, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
                            (0, 0, 0) if not is_unknown else (255, 255, 255), 1, cv2.LINE_AA)
            
            history_filename = f"hist_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.jpg"
            history_path = os.path.join(HISTORY_DIR, history_filename)
            cv2.imwrite(history_path, annotated_img)
            add_history_entry(len(detected_names), detected_names, latency_ms, history_path)
            
        return {
            "faces": identified_faces,
            "latency_ms": latency_ms,
            "detector_used": detector_used
        }
    except Exception as e:
        print(f"[FastAPI] Erro interno: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint para receber frames em base64 ou dados de imagem sequenciais
    de forma contínua para processamento em tempo real de baixa latência.
    """
    await websocket.accept()
    print("[WebSocket] Cliente conectado para streaming de vídeo.")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                # O payload pode ser opcionalmente enviado como JSON contendo parâmetros de busca
                payload_dict = json.loads(data)
                image_base64 = payload_dict.get("image_base64")
                detector_choice = payload_dict.get("detector", "auto")
                threshold_val = float(payload_dict.get("threshold", 0.68))
                extract_attrs = bool(payload_dict.get("extract_attributes", False))
            except Exception:
                # Se for enviado base64 cru
                image_base64 = data
                detector_choice = "auto"
                threshold_val = 0.68
                extract_attrs = False
                
            if not image_base64:
                await websocket.send_json({"error": "Dados de imagem vazios"})
                continue
                
            start_time = time.time()
            img = decode_base64_image(image_base64)
            if img is None:
                await websocket.send_json({"error": "Falha ao decodificar frame"})
                continue
                
            identified_faces, detected_names, detector_used = process_recognition_frame(
                img=img,
                detector_choice=detector_choice,
                threshold_val=threshold_val,
                extract_attrs=extract_attrs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if len(detected_names) > 0:
                annotated_img = img.copy()
                for face in identified_faces:
                    box = face["box"]
                    name = face["name"]
                    conf = face["confidence"]
                    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                    is_unknown = (name == "Desconhecido")
                    color = (0, 0, 255) if is_unknown else (0, 255, 135)
                    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
                    label = f"{name} ({(conf * 100):.0f}%)" if not is_unknown else "Desconhecido"
                    (w_t, h_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(annotated_img, (x, y - h_t - 8), (x + w_t, y), color, -1)
                    cv2.putText(annotated_img, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
                                (0, 0, 0) if not is_unknown else (255, 255, 255), 1, cv2.LINE_AA)
                
                history_filename = f"hist_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.jpg"
                history_path = os.path.join(HISTORY_DIR, history_filename)
                cv2.imwrite(history_path, annotated_img)
                add_history_entry(len(detected_names), detected_names, latency_ms, history_path)
                
            await websocket.send_json({
                "faces": identified_faces,
                "latency_ms": latency_ms,
                "detector_used": detector_used
            })
            
    except WebSocketDisconnect:
        print("[WebSocket] Cliente desconectado.")
    except Exception as ws_err:
        print(f"[WebSocket] Erro geral: {ws_err}")

@app.get("/history")
async def get_detection_history():
    """Retorna o histórico de detecções registradas."""
    return get_history(limit=30)

@app.get("/users")
async def get_registered_users():
    """Retorna os nomes e a imagem de snapshot dos usuários cadastrados."""
    result = []
    for u in registered_users:
        if u["templates"]:
            primary_template = u["templates"][0]
            created_at = primary_template["created_at"]
            image_path = primary_template["image_path"]
        else:
            created_at = "N/A"
            image_path = ""
            
        result.append({
            "name": u["name"], 
            "created_at": created_at,
            "image_url": f"/registered_images/{os.path.basename(image_path)}?t={int(time.time())}" if image_path else "",
            "templates_count": len(u["templates"])
        })
    return result

@app.delete("/users/{name}")
async def delete_registered_user(name: str):
    """Deleta um usuário cadastrado pelo nome."""
    delete_user(name)
    # Deleta todos os arquivos de imagem associados ao usuário
    if os.path.exists(REGISTERED_DIR):
        for f in os.listdir(REGISTERED_DIR):
            if f.startswith(f"{name}_") or f.startswith(f"{name}."):
                try:
                    os.remove(os.path.join(REGISTERED_DIR, f))
                except Exception as e:
                    print(f"Erro ao remover arquivo {f}: {e}")
    refresh_user_cache()
    return {"status": "success", "message": f"Usuário {name} removido!"}

# --- ROTAS DE APRENDIZADO ATIVO (ACTIVE LEARNING) ---

@app.get("/reviews")
async def get_all_reviews():
    """Retorna todas as revisões pendentes no banco de dados."""
    reviews = get_pending_reviews()
    # Adiciona a URL de imagem para servir no frontend
    for r in reviews:
        r["image_url"] = f"/pending_images/{os.path.basename(r['image_path'])}?t={int(time.time())}"
    return reviews

class ApproveReviewRequest(BaseModel):
    user_name: str

@app.post("/reviews/{review_id}/approve")
async def approve_review(review_id: str, payload: ApproveReviewRequest):
    """Aprova uma revisão pendente associando o rosto a um usuário existente ou novo."""
    reviews = get_pending_reviews()
    target_review = next((r for r in reviews if r["id"] == review_id), None)
    if not target_review:
        raise HTTPException(status_code=404, detail="Revisão pendente não encontrada.")
        
    user_name = payload.user_name.strip()
    if not user_name:
        raise HTTPException(status_code=400, detail="Nome do usuário não pode ser vazio.")
        
    src_path = target_review["image_path"]
    if not os.path.exists(src_path):
        raise HTTPException(status_code=400, detail="Arquivo da imagem de revisão não encontrado.")
        
    file_extension = os.path.splitext(src_path)[1] or ".jpg"
    dest_filename = f"{user_name}_{int(time.time())}{file_extension}"
    dest_path = os.path.join(REGISTERED_DIR, dest_filename)
    
    try:
        shutil.move(src_path, dest_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao mover arquivo de imagem: {e}")
        
    # Registra no banco como novo template do usuário
    success = register_user(user_name, target_review["embedding"], dest_path, quality_score=target_review["confidence"])
    if not success:
        if os.path.exists(dest_path):
            shutil.move(dest_path, src_path)
        raise HTTPException(status_code=500, detail="Falha ao salvar template no banco de dados.")
        
    # Deleta o registro pendente
    delete_pending_review(review_id)
    
    # Atualiza o cache e reconstrói o SVM
    refresh_user_cache()
    
    return {"status": "success", "message": f"Imagem aprovada com sucesso para '{user_name}'!"}

@app.post("/reviews/{review_id}/reject")
async def reject_review(review_id: str):
    """Rejeita uma revisão pendente e apaga sua imagem temporária."""
    reviews = get_pending_reviews()
    target_review = next((r for r in reviews if r["id"] == review_id), None)
    if not target_review:
        raise HTTPException(status_code=404, detail="Revisão pendente não encontrada.")
        
    img_path = target_review["image_path"]
    if os.path.exists(img_path):
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"[Active Learning] Erro ao remover imagem {img_path}: {e}")
            
    delete_pending_review(review_id)
    return {"status": "success", "message": "Revisão deletada com sucesso."}

# --- RTSP STREAMING INTEGRATION ---

class RTSPProcessor:
    def __init__(self):
        self.url = "rtsp://localhost:8554/live"
        self.cap = None
        self.running = False
        self.latest_frame = None
        self.annotated_frame = None
        self.lock = threading.Lock()
        self.reader_thread = None
        self.processor_thread = None
        self.detector = "auto"
        self.threshold = 0.68
        self.run_recognition = True
        self.latency_ms = 0.0
        self.faces_count = 0
        self.active_detector = "-"
        self.detected_names = []

    def start(self, url=None, detector="auto", threshold=0.68, run_recognition=True):
        with self.lock:
            if url:
                self.url = url
            self.detector = detector
            self.threshold = threshold
            self.run_recognition = run_recognition
            
            if self.running:
                # Se já estiver rodando, apenas garante que as flags estejam atualizadas
                return True
                
            self.running = True
            
            # Inicia thread de leitura em segundo plano
            self.reader_thread = threading.Thread(target=self._reader, daemon=True)
            self.reader_thread.start()
            
            # Inicia thread de processamento em segundo plano
            self.processor_thread = threading.Thread(target=self._processor, daemon=True)
            self.processor_thread.start()
            
            return True

    def stop(self):
        cap_to_release = None
        with self.lock:
            self.running = False
            if self.cap:
                cap_to_release = self.cap
                self.cap = None
            self.latest_frame = None
            self.annotated_frame = None
            self.detected_names = []
            self.faces_count = 0
            self.latency_ms = 0.0
            self.active_detector = "-"

        # Libera o VideoCapture fora do lock para evitar deadlocks
        if cap_to_release:
            try:
                print("[RTSP] Parando stream e liberando recursos de rede...")
                cap_to_release.release()
            except Exception as e:
                print(f"[RTSP] Erro ao liberar VideoCapture no stop: {e}")

    def _reader(self):
        print(f"[RTSP Reader] Iniciando loop de captura: {self.url}")
        
        consecutive_failures = 0
        while self.running:
            cap = None
            with self.lock:
                cap = self.cap
                
            if cap is None:
                # Se cap for None, tenta abrir fora do lock para não bloquear a API
                print(f"[RTSP Reader] Abrindo conexão com o stream: {self.url}")
                new_cap = cv2.VideoCapture(self.url)
                new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                with self.lock:
                    if not self.running:
                        new_cap.release()
                        break
                    self.cap = new_cap
                    cap = new_cap
            
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    print(f"[RTSP Reader] Falha de conexão persistente. Fechando VideoCapture para tentar reconexão...")
                    with self.lock:
                        if self.cap == cap:
                            self.cap = None
                    # Libera fora do lock
                    cap.release()
                    time.sleep(2)
                    consecutive_failures = 0
                else:
                    time.sleep(0.05)
                continue
                
            consecutive_failures = 0
            with self.lock:
                self.latest_frame = frame.copy()
                # Se o reconhecimento estiver desativado, o annotated_frame é o próprio frame cru
                if not self.run_recognition:
                    self.annotated_frame = frame.copy()
                    
            time.sleep(0.01)
            
        # Garante liberação fora do lock ao sair do loop
        cap_to_release = None
        with self.lock:
            if self.cap:
                cap_to_release = self.cap
                self.cap = None
        if cap_to_release:
            cap_to_release.release()
        print(f"[RTSP Reader] Loop de leitura encerrado.")

    def _processor(self):
        print(f"[RTSP Processor] Iniciando thread de processamento (Reconhecimento: {self.run_recognition}).")
        last_processed_time = 0.0
        
        while self.running:
            if not self.run_recognition:
                time.sleep(0.1)
                continue
                
            # Limita a taxa de processamento no CPU para no máximo 10 FPS
            now = time.time()
            if now - last_processed_time < 0.1:
                time.sleep(0.02)
                continue
                
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    
            if frame is None:
                time.sleep(0.05)
                continue
                
            last_processed_time = now
            start_time = time.time()
            try:
                # Processa detecção e reconhecimento facial usando o pipeline principal
                identified_faces, detected_names, detector_used = process_recognition_frame(
                    img=frame,
                    detector_choice=self.detector,
                    threshold_val=self.threshold,
                    extract_attrs=False
                )
                
                annotated = frame.copy()
                for face in identified_faces:
                    box = face["box"]
                    name = face["name"]
                    conf = face["confidence"]
                    landmarks = face["landmarks"]
                    
                    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                    is_unknown = (name == "Desconhecido")
                    color = (0, 0, 255) if is_unknown else (0, 255, 135)
                    
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                    if landmarks:
                        for lm in landmarks:
                            cv2.circle(annotated, (int(lm[0]), int(lm[1])), 3, (0, 255, 255), -1)
                            
                    label = f"{name} ({(conf * 100):.0f}%)" if not is_unknown else "Desconhecido"
                    (w_t, h_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(annotated, (x, y - h_t - 8), (x + w_t, y), color, -1)
                    cv2.putText(annotated, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
                                (0, 0, 0) if not is_unknown else (255, 255, 255), 1, cv2.LINE_AA)
                                
                latency = (time.time() - start_time) * 1000
                if len(detected_names) > 0:
                    add_history_entry(len(detected_names), detected_names, latency)
                    
                with self.lock:
                    self.annotated_frame = annotated
                    self.latency_ms = latency
                    self.faces_count = len(identified_faces)
                    self.active_detector = detector_used
                    self.detected_names = detected_names
                    
            except Exception as e:
                print(f"[RTSP Process] Falha ao analisar frame: {e}")
                # Em caso de erro de processamento, apenas joga o frame cru
                with self.lock:
                    self.annotated_frame = frame.copy()
                    
        print(f"[RTSP Processor] Thread de processamento finalizada.")

# --- WEBRTC PROXY ENDPOINTS FOR GO2RTC ---
import requests
import base64

@app.post("/api/webrtc/register")
def register_webrtc_stream(payload: dict):
    """Registra uma câmera RTSP/RTMP no go2rtc dinamicamente."""
    url = payload.get("url")
    name = payload.get("name", "live")
    if not url:
        raise HTTPException(status_code=400, detail="URL do stream não fornecida.")
        
    go2rtc_url = f"http://localhost:1984/api/streams?name={name}&src={url}"
    try:
        response = requests.put(go2rtc_url, timeout=5)
        if response.status_code in [200, 201]:
            return {"status": "success", "message": f"Stream '{name}' registrado no go2rtc com sucesso!"}
        else:
            return JSONResponse(status_code=500, content={"error": f"Erro no go2rtc: {response.text}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Não foi possível conectar ao go2rtc: {str(e)}"})

@app.post("/api/webrtc/negotiate")
def negotiate_webrtc(payload: dict):
    """Realiza a negociação SDP (Offer/Answer) com o go2rtc."""
    sdp = payload.get("sdp")
    name = payload.get("name", "live")
    if not sdp:
        raise HTTPException(status_code=400, detail="SDP Offer não fornecido.")
        
    go2rtc_url = f"http://localhost:1984/api/webrtc?src={name}"
    try:
        response = requests.post(
            go2rtc_url,
            headers={"Content-Type": "application/sdp"},
            data=sdp,
            timeout=5
        )
        if response.status_code in [200, 201]:
            return {"sdp": response.text}
        else:
            return JSONResponse(status_code=500, content={"error": f"Erro na negociação com go2rtc: {response.text}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Falha de rede ao conectar ao go2rtc: {str(e)}"})

@app.get("/api/discover")
async def discover_onvif_cameras():
    """Busca câmeras ONVIF na rede local via multicast UDP (WS-Discovery)."""
    import socket
    import re
    
    message_id = f"uuid:{uuid.uuid4()}"
    probe_msg = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<Envelope xmlns:tds="http://www.onvif.org/ver10/device/wsdl" xmlns="http://www.w3.org/2003/05/soap-envelope" xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
          '<Header>'
            f'<MessageID xmlns="http://schemas.xmlsoap.org/ws/2004/08/addressing">{message_id}</MessageID>'
            '<To xmlns="http://schemas.xmlsoap.org/ws/2004/08/addressing">urn:schemas-xmlsoap-org:ws:2004:08:addressing</To>'
            '<Action xmlns="http://schemas.xmlsoap.org/ws/2004/08/addressing">http://schemas.xmlsoap.org/ws/2004/08/dmt/Probe</Action>'
          '</Header>'
          '<Body>'
            '<Probe xmlns="http://schemas.xmlsoap.org/ws/2004/08/discovery">'
              '<Types>dn:NetworkVideoTransmitter</Types>'
            '</Probe>'
          '</Body>'
        '</Envelope>'
    )

    multicast_group = "239.255.255.250"
    port = 3702

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.settimeout(1.5)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    cameras = []
    seen_ips = set()
    try:
        sock.sendto(probe_msg.encode('utf-8'), (multicast_group, port))
        
        start_time = time.time()
        while time.time() - start_time < 2.0:
            try:
                data, addr = sock.recvfrom(65507)
                ip = addr[0]
                if ip in seen_ips:
                    continue
                seen_ips.add(ip)
                
                response = data.decode('utf-8', errors='ignore')
                xaddrs = re.findall(r'<[^:>]*:?XAddrs>([^<]+)</[^:>]*:?XAddrs>', response)
                scopes = re.findall(r'<[^:>]*:?Scopes>([^<]+)</[^:>]*:?Scopes>', response)
                
                scopes_parsed = {}
                name = "Câmera Genérica"
                hardware = "ONVIF"
                
                if scopes:
                    for scope in scopes[0].split():
                        parts = scope.split('/')
                        if len(parts) > 3:
                            key = parts[-2]
                            val = parts[-1]
                            scopes_parsed[key] = val
                            if key == "name":
                                name = re.sub(r'%20', ' ', val)
                            elif key == "hardware":
                                hardware = re.sub(r'%20', ' ', val)
                
                cameras.append({
                    "ip": ip,
                    "name": name,
                    "hardware": hardware,
                    "xaddrs": xaddrs[0] if xaddrs else "",
                    "scopes": scopes_parsed
                })
            except socket.timeout:
                break
    except Exception as e:
        print(f"[ONVIF Discovery] Erro ao buscar: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        sock.close()
        
    return {"status": "success", "cameras": cameras}
