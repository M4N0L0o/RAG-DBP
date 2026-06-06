from typing import Dict
import json
import asyncio
import re
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Response, Depends
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import chromadb
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Librerías de Seguridad
import jwt
import bcrypt # Reemplazamos passlib por bcrypt nativo

# ==========================================
# CONFIGURACIÓN GENERAL Y SEGURIDAD
# ==========================================
MONGO_URI = "mongodb://localhost:27017/"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8001
OLLAMA_MODEL = "phi3" 
MAX_DISTANCE = 2.0

# Configuración JWT
SECRET_KEY = "clave_secreta_super_segura_para_tesis_dbp"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

security = HTTPBearer()

# Variables globales
rag_chain_with_history = None
vector_store = None
mongo_collection = None
mongo_users = None
mongo_logs = None

store: Dict[str, BaseChatMessageHistory] = {}
GREETING_KEYWORDS = ['hola', 'buenos', 'buenas', 'halo', '¿quién eres', 'cual es tu nombre', 'como te llamas', 'quién eres', 'qué eres', 'ayuda', 'bye', 'adios', 'gracias']

# ==========================================
# MODELOS DE DATOS
# ==========================================
class LoginRequest(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

# ==========================================
# FUNCIONES DE SEGURIDAD (NATIVAS BCRYPT)
# ==========================================
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Token inválido")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="El token ha expirado")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

def log_audit(username: str, action: str, details: str):
    """Inyecta un registro de auditoría en MongoDB"""
    if mongo_logs is not None:
        mongo_logs.insert_one({
            "usuario": username,
            "accion": action,
            "detalles": details,
            "fecha": datetime.utcnow().isoformat()
        })

# ==========================================
# UTILIDADES RAG
# ==========================================
def is_greeting(text: str) -> bool:
    text_lower = text.lower().strip()
    return any(keyword in text_lower for keyword in GREETING_KEYWORDS)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ==========================================
# INICIALIZACIÓN (LIFESPAN)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain_with_history, vector_store, mongo_collection, mongo_users, mongo_logs
    print("Inicializando Arquitectura Enterprise (MongoDB + ChromaDB Nativo)...")

    # 1. Conectar a MongoDB local y crear colecciones
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping') 
        mongo_db = mongo_client["nexus_db"]
        mongo_collection = mongo_db["documents"]
        mongo_users = mongo_db["users"]
        mongo_logs = mongo_db["audit_logs"]
        print("✅ Conectado a MongoDB local con éxito.")

        # Inyectar usuario administrador por defecto si no existe
        if mongo_users.count_documents({}) == 0:
            hashed_pwd = get_password_hash("admin123")
            # Agregamos el correo al usuario por defecto
            mongo_users.insert_one({"username": "admin", "email": "admin@epn.edu.ec", "password": hashed_pwd, "role": "admin"})
            print("🛡️ Colección 'users' creada. Usuario por defecto generado: admin / admin123")

    except ConnectionFailure:
        print("❌ ERROR: No se pudo conectar a MongoDB.")

    # 2. Cargar Embeddings (Forzado en CPU)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 3. Conectar a ChromaDB (Servidor Nativo)
    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        vector_store = Chroma(
            client=chroma_client,
            collection_name="nexus_collection",
            embedding_function=embeddings
        )
        print("✅ Conectado a ChromaDB Server con éxito.")
    except Exception as e:
        print(f"❌ ERROR conectando a ChromaDB: {e}")

    # 4. Configurar ChatOllama
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.5, top_k=10, top_p=0.5)

    system_prompt = """Eres el asistente virtual oficial de la Dirección de Bienestar Politécnico (DBP). Tu único propósito es responder consultas basándote EXCLUSIVAMENTE en el contexto proporcionado.

        REGLAS DE COMPORTAMIENTO:
        1. IDENTIDAD Y CORTESÍA: Si el usuario te saluda, agradece o pregunta quién eres, responde de forma cordial en una sola oración indicando que eres el asistente del DBP.
        2. PRECISIÓN ESTRICTA (CERO ADORNOS): No agregues explicaciones extra, consejos, suposiciones ni alternativas. Cíñete a los hechos del contexto. 
        3. FORMATO DE PROCESOS: Si se consulta sobre un procedimiento, estructúralo obligatoriamente en una lista numerada con oraciones cortas y directas. No inventes pasos.
        4. LÍMITE DE CONOCIMIENTO: Si la respuesta no está en el contexto o este está vacío, responde EXACTAMENTE: "Mis conocimientos se limitan a las políticas, servicios y procedimientos de la Direccion de Bienestar Politécnico."

    Contexto:
    {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{question}")
    ])

    rag_chain = prompt | llm | StrOutputParser()
    rag_chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    
    print("🚀 API levantada y segura. Lista para recibir peticiones.")
    yield

app = FastAPI(title="API RAG Segura", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ENDPOINTS DE AUTENTICACIÓN Y GESTIÓN DE USUARIOS
# ==========================================

@app.post("/api/login")
async def login(req: LoginRequest):
    if mongo_users is None:
        raise HTTPException(status_code=500, detail="Base de datos no conectada")
    
    user = mongo_users.find_one({"username": req.username})
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")
    
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer", "username": user["username"]}

@app.post("/api/users")
async def create_user(user: UserCreate, current_user: str = Depends(get_current_user)):
    """Crea un nuevo usuario asegurando el correo institucional."""
    if not user.email.endswith("@epn.edu.ec"):
        raise HTTPException(status_code=400, detail="El correo debe tener la terminación institucional @epn.edu.ec")
    
    if mongo_users.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="El nombre de usuario ya existe")
        
    if mongo_users.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="El correo ya está registrado")
    
    hashed_pwd = get_password_hash(user.password)
    mongo_users.insert_one({
        "username": user.username,
        "email": user.email,
        "password": hashed_pwd,
        "role": "admin"
    })
    
    log_audit(current_user, "creo_usuario", f"Registró a {user.username} ({user.email})")
    return {"message": "Usuario creado exitosamente"}

@app.get("/api/users")
async def get_users(current_user: str = Depends(get_current_user)):
    """Lista todos los usuarios omitiendo las contraseñas."""
    try:
        users = list(mongo_users.find({}, {"_id": 0, "password": 0}))
        return {"users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs(current_user: str = Depends(get_current_user)):
    """Obtiene los logs de auditoría ordenados por fecha descendente."""
    try:
        logs = list(mongo_logs.find({}, {"_id": 0}).sort("fecha", -1).limit(100))
        return {"logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ENDPOINTS DE ADMINISTRACIÓN (PROTEGIDOS)
# ==========================================

@app.get("/api/documents")
async def list_documents(current_user: str = Depends(get_current_user)):
    if mongo_collection is None:
        raise HTTPException(status_code=500, detail="Base de datos no conectada.")
    try:
        docs = mongo_collection.find({}, {"filename": 1, "_id": 0})
        return {"documents": [doc["filename"] for doc in docs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{filename}")
async def download_document(filename: str, current_user: str = Depends(get_current_user)):
    if mongo_collection is None:
        raise HTTPException(status_code=500, detail="Base de datos no conectada.")
    try:
        doc = mongo_collection.find_one({"filename": filename})
        if not doc:
            raise HTTPException(status_code=404, detail="Archivo no encontrado.")
        return Response(
            content=doc.get("content", ""), 
            media_type="text/plain", 
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .txt")
    
    try:
        content_bytes = await file.read()
        text_content = content_bytes.decode("utf-8")
        
        # Guardar en MongoDB
        mongo_collection.update_one(
            {"filename": file.filename},
            {"$set": {"filename": file.filename, "content": text_content}},
            upsert=True
        )
        
        # Limpiar vectores antiguos y procesar nuevos
        vector_store._collection.delete(where={"source": file.filename})
        
        semantic_chunks = []
        current_section = "0"
        current_title = "General"
        current_content = []
        
        header_pattern = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
        lines = text_content.split('\n')
        
        for line in lines:
            match = header_pattern.match(line.strip())
            if match:
                content_str = "\n".join(current_content).strip()
                if content_str:
                    semantic_chunks.append(Document(
                        page_content=f"[{current_section} {current_title}]\n{content_str}",
                        metadata={"source": file.filename, "section": current_section, "title": current_title}
                    ))
                current_section = match.group(1)
                current_title = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
                
        content_str = "\n".join(current_content).strip()
        if content_str:
            semantic_chunks.append(Document(
                page_content=f"[{current_section} {current_title}]\n{content_str}",
                metadata={"source": file.filename, "section": current_section, "title": current_title}
            ))

        final_chunks = []
        fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        for chunk in semantic_chunks:
            if len(chunk.page_content) > 1000:
                final_chunks.extend(fallback_splitter.split_documents([chunk]))
            else:
                final_chunks.append(chunk)
        
        vector_store.add_documents(final_chunks)
        
        # Log modificado para guardar detalles
        log_audit(current_user, "subio_archivo", f"Documento: {file.filename}")
        
        return {"message": f"Archivo '{file.filename}' procesado correctamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str, current_user: str = Depends(get_current_user)):
    try:
        result = mongo_collection.delete_one({"filename": filename})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Archivo no encontrado.")
        
        vector_store._collection.delete(where={"source": filename})
        
        log_audit(current_user, "elimino_archivo", f"Documento: {filename}")
        
        return {"message": f"Documento '{filename}' eliminado con éxito."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ENDPOINTS DE CHAT (PÚBLICOS)
# ==========================================
class ChatRequest(BaseModel):
    pregunta: str
    session_id: str = "sesion_default"

class ChatResponse(BaseModel):
    respuesta: str
    contexto_utilizado: list[str] = []

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not rag_chain_with_history or not vector_store:
        raise HTTPException(status_code=500, detail="El sistema RAG no se inicializó correctamente.")
    try:
        resultados = vector_store.similarity_search_with_score(request.pregunta, k=3)
        if not resultados or resultados[0][1] > MAX_DISTANCE:
            if not is_greeting(request.pregunta):
                return ChatResponse(respuesta="Mis conocimientos se limitan a las políticas, servicios y procedimientos de la Dirección de Bienestar Politécnico.", contexto_utilizado=[])
            contexto_crudo = ""
            textos_contexto = []
        else:
            textos_contexto = [doc.page_content for doc, score in resultados]
            contexto_crudo = "\n\n".join(textos_contexto)
        
        respuesta = rag_chain_with_history.invoke(
            {"question": request.pregunta, "context": contexto_crudo},
            config={"configurable": {"session_id": request.session_id}}
        )
        return ChatResponse(respuesta=respuesta, contexto_utilizado=textos_contexto)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chat/stream")
async def chat_stream_endpoint(request: Request, chat_req: ChatRequest):
    async def event_generator():
        try:
            resultados = vector_store.similarity_search_with_score(chat_req.pregunta, k=3)
            if not resultados or resultados[0][1] > MAX_DISTANCE:
                if not is_greeting(chat_req.pregunta):
                    yield f"data: {json.dumps({'respuesta': 'Mis conocimientos se limitan a las políticas, servicios y procedimientos de la Dirección de Bienestar Politécnico.'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                contexto_crudo = ""
            else:
                textos_contexto = [doc.page_content for doc, score in resultados]
                contexto_crudo = "\n\n".join(textos_contexto)

            async for chunk in rag_chain_with_history.astream(
                {"question": chat_req.pregunta, "context": contexto_crudo},
                config={"configurable": {"session_id": chat_req.session_id}}
            ):
                if await request.is_disconnected():
                    break
                yield f"data: {json.dumps({'respuesta': chunk})}\n\n"
                await asyncio.sleep(0)
            
            if not await request.is_disconnected():
                yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)