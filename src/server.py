import os
import time
import base64
import cv2
import numpy as np
import json
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.database import (
    init_db, register_user, get_all_users, 
    add_history_entry, get_history, cosine_similarity, delete_user,
    add_pending_review, get_pending_reviews, delete_pending_review
)
from src.detectors.opencv_yunet import OpenCVYuNetDetector
from src.detectors.deepface_det import DeepFaceDetector
from deepface import DeepFace

# Inicializações
app = FastAPI(title="Pipeline de Detecção & Reconhecimento Facial")
init_db()

# Diretórios necessários
REGISTERED_DIR = "data/registered"
PENDING_DIR = "data/pending_review"
os.makedirs(REGISTERED_DIR, exist_ok=True)
os.makedirs(PENDING_DIR, exist_ok=True)
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

# Inicializa os detectores
detector_yunet = OpenCVYuNetDetector(score_threshold=0.6)
detector_retinaface = DeepFaceDetector(detector_backend='retinaface')

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
    temp_img_path = f"data/temp_recon_proc_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(temp_img_path, img)
    
    # 1. Detectores em cascata
    detector_used = detector_choice
    if detector_choice == "retinaface":
        detections = detector_retinaface.detect(temp_img_path)
    elif detector_choice == "yunet":
        detections = detector_yunet.detect(temp_img_path)
    else: # auto
        detections = detector_yunet.detect(temp_img_path)
        detector_used = "yunet"
        if len(detections) == 0:
            detections = detector_retinaface.detect(temp_img_path)
            detector_used = "retinaface"
            
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
        
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
            # Se a similaridade é limítrofe (distância entre 0.50 e 0.70), enviar para revisão do usuário
            if 0.50 <= min_dist_any <= 0.70:
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
            add_history_entry(len(detected_names), detected_names, latency_ms)
            
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
                add_history_entry(len(detected_names), detected_names, latency_ms)
                
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
