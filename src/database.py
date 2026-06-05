import os
import json
import uuid
import sqlite3
import numpy as np
from datetime import datetime

# Tipo de banco: "sqlite" ou "qdrant" (padrão: "sqlite" para testes locais fáceis)
DB_TYPE = os.getenv("DATABASE_TYPE", "sqlite")
DB_PATH = "data/face_recognition.db"

# Inicializa o cliente do Qdrant se configurado
client = None
if DB_TYPE == "qdrant":
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# --- BACKEND SQLITE HELPERS ---
def get_sqlite_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def execute_migrations(conn):
    """
    Verifica se a base de dados SQLite está no formato antigo (tabela users com coluna embedding)
    e migra os dados automaticamente para as novas tabelas Users e Face_Templates.
    """
    cursor = conn.cursor()
    
    # 1. Verifica se a tabela users existe e tem a coluna embedding
    cursor.execute("PRAGMA table_info(users)")
    columns = [col["name"] for col in cursor.fetchall()]
    
    if "embedding" in columns:
        print("[Database Migração] Detectada tabela 'users' no formato antigo. Iniciando migração...")
        
        # Lê todos os dados antigos
        cursor.execute("SELECT name, embedding, image_path, created_at, quality_score FROM users")
        old_rows = cursor.fetchall()
        
        # Renomeia tabela antiga
        cursor.execute("ALTER TABLE users RENAME TO users_old")
        
        # Cria as novas tabelas normalizadas
        cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        """)
        
        cursor.execute("""
        CREATE TABLE face_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            embedding TEXT NOT NULL,
            image_path TEXT NOT NULL,
            quality_score REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        
        # Migra os registros
        for row in old_rows:
            name = row["name"]
            embedding = row["embedding"]
            image_path = row["image_path"]
            created_at = row["created_at"]
            quality_score = float(row["quality_score"]) if row["quality_score"] is not None else 0.68
            
            try:
                # Insere o usuário
                cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
                user_id = cursor.lastrowid
            except sqlite3.IntegrityError:
                # Caso já tenha inserido o mesmo nome por algum motivo
                cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
                user_id = cursor.fetchone()["id"]
                
            # Insere o embedding correspondente na tabela face_templates
            cursor.execute("""
            INSERT INTO face_templates (user_id, embedding, image_path, quality_score, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, (user_id, embedding, image_path, quality_score, created_at))
            
        # Remove a tabela antiga
        cursor.execute("DROP TABLE users_old")
        conn.commit()
        print(f"[Database Migração] SUCESSO: {len(old_rows)} perfis de usuários migrados para a galeria multi-template!")

def check_and_clean_incompatible_embeddings(expected_dim: int = 512):
    """Detecta embeddings antigos com dimensões erradas e os exclui do SQLite."""
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, embedding FROM face_templates")
        rows = cursor.fetchall()
        to_delete = []
        for row in rows:
            try:
                emb = json.loads(row["embedding"])
                if not isinstance(emb, list) or len(emb) != expected_dim:
                    to_delete.append(row["id"])
            except Exception:
                to_delete.append(row["id"])
        
        if to_delete:
            print(f"[Database SQLite] Autolimpeza: Deletando {len(to_delete)} templates com embeddings incompatíveis (esperado: {expected_dim} dimensões).")
            cursor.executemany("DELETE FROM face_templates WHERE id = ?", [(id_val,) for id_val in to_delete])
            conn.commit()
    except Exception as e:
        print(f"[Database SQLite] Erro ao validar embeddings para autolimpeza: {e}")
    finally:
        conn.close()


# --- INTERFACE UNIFICADA DE BANCO ---

def init_db():
    if DB_TYPE == "qdrant":
        try:
            if not client.collection_exists(collection_name="users"):
                client.create_collection(
                    collection_name="users",
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
                )
                print("[Qdrant] Coleção 'users' criada para embeddings ArcFace (512-dim).")
            if not client.collection_exists(collection_name="history"):
                client.create_collection(
                    collection_name="history",
                    vectors_config=VectorParams(size=1, distance=Distance.COSINE),
                )
                print("[Qdrant] Coleção 'history' criada.")
            if not client.collection_exists(collection_name="pending_reviews"):
                client.create_collection(
                    collection_name="pending_reviews",
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
                )
                print("[Qdrant] Coleção 'pending_reviews' criada.")
        except Exception as e:
            print(f"[Qdrant] Erro ao inicializar coleções: {e}")
    else:
        # SQLite
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        
        # Cria as tabelas normalizadas se não existirem
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            embedding TEXT NOT NULL,
            image_path TEXT NOT NULL,
            quality_score REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            faces_detected INTEGER NOT NULL,
            identified_names TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            image_path TEXT
        )
        """)
        
        # Migração: adiciona a coluna image_path na tabela history se não existir (para compatibilidade)
        cursor.execute("PRAGMA table_info(history)")
        history_cols = [col["name"] for col in cursor.fetchall()]
        if "image_path" not in history_cols:
            cursor.execute("ALTER TABLE history ADD COLUMN image_path TEXT")
            conn.commit()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS pending_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding TEXT NOT NULL,
            image_path TEXT NOT NULL,
            suggested_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
        
        conn.commit()
        
        # Executa migrações caso venha de uma base de dados anterior
        execute_migrations(conn)
        conn.close()
        
        # Valida as dimensões dos embeddings carregados (ArcFace 512-dim)
        check_and_clean_incompatible_embeddings(512)

def register_user(name: str, embedding: list[float], image_path: str, quality_score: float = 0.68):
    """Adiciona um novo template facial associado ao nome fornecido."""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if DB_TYPE == "qdrant":
        try:
            point_id = str(uuid.uuid4())
            client.upsert(
                collection_name="users",
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "name": name,
                            "image_path": image_path,
                            "created_at": created_at,
                            "quality_score": quality_score
                        }
                    )
                ]
            )
            return True
        except Exception as e:
            print(f"[Qdrant] Erro ao cadastrar usuário '{name}': {e}")
            return False
    else:
        # SQLite
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        embedding_str = json.dumps(embedding)
        try:
            # Insere usuário se ele não existir
            cursor.execute("INSERT OR IGNORE INTO users (name) VALUES (?)", (name,))
            # Pega o ID
            cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
            user_id = cursor.fetchone()["id"]
            
            # Insere o novo template facial
            cursor.execute("""
            INSERT INTO face_templates (user_id, embedding, image_path, quality_score, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, (user_id, embedding_str, image_path, quality_score, created_at))
            
            conn.commit()
            success = True
        except Exception as e:
            print(f"[SQLite] Erro ao cadastrar usuário: {e}")
            success = False
        finally:
            conn.close()
        return success

def get_all_users():
    """Retorna todos os usuários estruturados com sua lista de templates."""
    if DB_TYPE == "qdrant":
        try:
            result, _ = client.scroll(
                collection_name="users",
                limit=1000,
                with_payload=True,
                with_vectors=True
            )
            
            # Agrupa por nome do usuário
            user_groups = {}
            for point in result:
                payload = point.payload
                name = payload.get("name")
                if not name:
                    continue
                
                template_data = {
                    "id": point.id,
                    "embedding": point.vector,
                    "image_path": payload.get("image_path"),
                    "created_at": payload.get("created_at"),
                    "quality_score": float(payload.get("quality_score", 0.68))
                }
                
                if name not in user_groups:
                    user_groups[name] = []
                user_groups[name].append(template_data)
                
            users = []
            for name, templates in user_groups.items():
                users.append({
                    "id": templates[0]["id"], # Dummy User ID
                    "name": name,
                    "templates": templates
                })
            return users
        except Exception as e:
            print(f"[Qdrant] Erro ao listar usuários: {e}")
            return []
    else:
        # SQLite
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        
        # Carrega todos os usuários
        cursor.execute("SELECT id, name FROM users")
        user_rows = cursor.fetchall()
        
        users = []
        for u_row in user_rows:
            user_id = u_row["id"]
            name = u_row["name"]
            
            # Busca os templates faciais deste usuário
            cursor.execute("""
            SELECT id, embedding, image_path, quality_score, created_at 
            FROM face_templates WHERE user_id = ?
            """, (user_id,))
            template_rows = cursor.fetchall()
            
            templates = []
            for t_row in template_rows:
                templates.append({
                    "id": t_row["id"],
                    "embedding": json.loads(t_row["embedding"]),
                    "image_path": t_row["image_path"],
                    "quality_score": float(t_row["quality_score"]),
                    "created_at": t_row["created_at"]
                })
                
            users.append({
                "id": user_id,
                "name": name,
                "templates": templates
            })
        conn.close()
        return users

def delete_user(name: str):
    """Deleta o usuário e todos os seus templates associados (ON DELETE CASCADE no SQLite)."""
    if DB_TYPE == "qdrant":
        try:
            client.delete(
                collection_name="users",
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="name",
                            match=MatchValue(value=name)
                        )
                    ]
                )
            )
            return True
        except Exception as e:
            print(f"[Qdrant] Erro ao deletar usuário '{name}': {e}")
            return False
    else:
        # SQLite
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE name = ?", (name,))
        conn.commit()
        conn.close()
        return True

def add_history_entry(faces_detected: int, identified_names: list[str], latency_ms: float, image_path: str = None):
    if DB_TYPE == "qdrant":
        try:
            entry_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            client.upsert(
                collection_name="history",
                points=[
                    PointStruct(
                        id=entry_id,
                        vector=[0.0],
                        payload={
                            "timestamp": timestamp,
                            "faces_detected": faces_detected,
                            "identified_names": identified_names,
                            "latency_ms": latency_ms,
                            "image_path": image_path
                        }
                    )
                ]
            )
            return True
        except Exception as e:
            print(f"[Qdrant] Erro ao salvar histórico: {e}")
            return False
    else:
        # SQLite
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        names_str = json.dumps(identified_names)
        cursor.execute("""
        INSERT INTO history (timestamp, faces_detected, identified_names, latency_ms, image_path)
        VALUES (?, ?, ?, ?, ?)
        """, (timestamp, faces_detected, names_str, latency_ms, image_path))
        conn.commit()
        conn.close()
        return True

def get_history(limit: int = 50):
    if DB_TYPE == "qdrant":
        try:
            result, _ = client.scroll(
                collection_name="history",
                limit=limit,
                with_payload=True
            )
            history_entries = []
            for point in result:
                payload = point.payload
                history_entries.append({
                    "id": point.id,
                    "timestamp": payload.get("timestamp"),
                    "faces_detected": payload.get("faces_detected"),
                    "identified_names": payload.get("identified_names", []),
                    "latency_ms": payload.get("latency_ms", 0.0),
                    "image_path": payload.get("image_path")
                })
            history_entries.sort(key=lambda x: x["timestamp"], reverse=True)
            return history_entries
        except Exception as e:
            print(f"[Qdrant] Erro ao recuperar histórico: {e}")
            return []
    else:
        # SQLite
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, faces_detected, identified_names, latency_ms, image_path FROM history ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        history_entries = []
        for row in rows:
            history_entries.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "faces_detected": row["faces_detected"],
                "identified_names": json.loads(row["identified_names"]),
                "latency_ms": row["latency_ms"],
                "image_path": row["image_path"]
            })
        conn.close()
        return history_entries


# --- ACTIVE LEARNING: REVISÕES PENDENTES ---

def add_pending_review(embedding: list[float], image_path: str, suggested_name: str, confidence: float):
    """Adiciona uma foto com confiança limítrofe para revisão do usuário."""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if DB_TYPE == "qdrant":
        try:
            point_id = str(uuid.uuid4())
            client.upsert(
                collection_name="pending_reviews",
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "image_path": image_path,
                            "suggested_name": suggested_name,
                            "confidence": confidence,
                            "created_at": created_at
                        }
                    )
                ]
            )
            return True
        except Exception as e:
            print(f"[Qdrant] Erro ao registrar revisão pendente: {e}")
            return False
    else:
        # SQLite
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        embedding_str = json.dumps(embedding)
        try:
            cursor.execute("""
            INSERT INTO pending_reviews (embedding, image_path, suggested_name, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, (embedding_str, image_path, suggested_name, confidence, created_at))
            conn.commit()
            return True
        except Exception as e:
            print(f"[SQLite] Erro ao registrar revisão pendente: {e}")
            return False
        finally:
            conn.close()

def get_pending_reviews():
    """Recupera todas as revisões pendentes de identificação."""
    if DB_TYPE == "qdrant":
        try:
            result, _ = client.scroll(
                collection_name="pending_reviews",
                limit=200,
                with_payload=True,
                with_vectors=True
            )
            reviews = []
            for point in result:
                payload = point.payload
                reviews.append({
                    "id": point.id,
                    "embedding": point.vector,
                    "image_path": payload.get("image_path"),
                    "suggested_name": payload.get("suggested_name"),
                    "confidence": float(payload.get("confidence", 0.0)),
                    "created_at": payload.get("created_at")
                })
            return reviews
        except Exception as e:
            print(f"[Qdrant] Erro ao listar revisões: {e}")
            return []
    else:
        # SQLite
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, embedding, image_path, suggested_name, confidence, created_at FROM pending_reviews ORDER BY id DESC")
        rows = cursor.fetchall()
        reviews = []
        for row in rows:
            reviews.append({
                "id": str(row["id"]), # Convertido para string para manter coerência com os UUIDs do Qdrant
                "embedding": json.loads(row["embedding"]),
                "image_path": row["image_path"],
                "suggested_name": row["suggested_name"],
                "confidence": float(row["confidence"]),
                "created_at": row["created_at"]
            })
        conn.close()
        return reviews

def delete_pending_review(review_id: str):
    """Exclui a revisão pendente pelo ID."""
    if DB_TYPE == "qdrant":
        try:
            client.delete(
                collection_name="pending_reviews",
                points_selector=[review_id]
            )
            return True
        except Exception as e:
            print(f"[Qdrant] Erro ao deletar revisão {review_id}: {e}")
            return False
    else:
        # SQLite
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM pending_reviews WHERE id = ?", (int(review_id),))
            conn.commit()
            return True
        except Exception as e:
            print(f"[SQLite] Erro ao deletar revisão {review_id}: {e}")
            return False
        finally:
            conn.close()


def update_template(template_id: str, embedding: list[float], image_path: str, quality_score: float):
    """Atualiza o embedding e o score de qualidade de um template específico."""
    if DB_TYPE == "qdrant":
        try:
            # No Qdrant, buscamos o ponto original para preservar metadados como nome
            point = client.retrieve(collection_name="users", ids=[template_id])[0]
            name = point.payload.get("name")
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            client.upsert(
                collection_name="users",
                points=[
                    PointStruct(
                        id=template_id,
                        vector=embedding,
                        payload={
                            "name": name,
                            "image_path": image_path,
                            "created_at": created_at,
                            "quality_score": quality_score
                        }
                    )
                ]
            )
            return True
        except Exception as e:
            print(f"[Qdrant] Erro ao atualizar template {template_id}: {e}")
            return False
    else:
        # SQLite
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        embedding_str = json.dumps(embedding)
        try:
            cursor.execute("""
            UPDATE face_templates 
            SET embedding = ?, image_path = ?, quality_score = ?
            WHERE id = ?
            """, (embedding_str, image_path, quality_score, int(template_id)))
            conn.commit()
            return True
        except Exception as e:
            print(f"[SQLite] Erro ao atualizar template {template_id}: {e}")
            return False
        finally:
            conn.close()

def cosine_similarity(a: list[float], b: list[float]) -> float:
    vec_a = np.array(a, dtype=np.float32)
    vec_b = np.array(b, dtype=np.float32)
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))
