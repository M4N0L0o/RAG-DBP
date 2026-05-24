from typing import Dict
import json
import asyncio
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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

# Configuración de conexiones locales nativas
MONGO_URI = "mongodb://localhost:27017/"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8001
OLLAMA_MODEL = "phi3" 

# Umbral Anti-Alucinaciones (Distancia L2: 0.0 es perfecto, mayor es menos similar)
MAX_DISTANCE = 1.0 

# Variables globales
rag_chain_with_history = None
vector_store = None
mongo_collection = None

# Almacén de memoria en RAM para las sesiones de chat
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain_with_history, vector_store, mongo_collection
    print("Inicializando Arquitectura Enterprise (MongoDB + ChromaDB Nativo)...")

    # 1. Conectar a MongoDB local
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping') 
        mongo_db = mongo_client["nexus_db"]
        mongo_collection = mongo_db["documents"]
        print("✅ Conectado a MongoDB local con éxito.")
    except ConnectionFailure:
        print("❌ ERROR: No se pudo conectar a MongoDB. ¿Está instalado y corriendo el servicio?")

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
        print(f"❌ ERROR conectando a ChromaDB: {e}. ¿Ejecutaste 'chroma run --port 8001'?")

    # 4. Configurar ChatOllama
    llm = ChatOllama(model=OLLAMA_MODEL)

    # ==========================================
    # NUEVO SYSTEM PROMPT: ESTRICTO Y ANTI-ADORNOS
    # ==========================================
    system_prompt = """Eres el asistente virtual oficial de la Dirección de Bienestar Politécnico (DBP) de la Escuela Politécnica Nacional.

    Reglas de comportamiento ESTRICTAS:
    1. Saludos/Charlas: Si el usuario te saluda, agradece o se despide, responde de forma natural, cordial y MUY breve (máximo 1 oración).
    2. Tu Identidad: Si te preguntan quién eres, responde en una sola oración que eres el asistente del DBP.
    3. Consultas: Responde ÚNICAMENTE basándote en la información exacta del "Contexto".
    4. CERO ADORNOS (CRÍTICO): NO agregues explicaciones extra, NO des consejos que no estén en el texto, y NO inventes alternativas, suposiciones ni pasos siguientes. Cíñete a los hechos. Si la respuesta es directa, dala directa y finaliza.
    5. Límite de Conocimiento: Si el "Contexto" está vacío o no contiene la respuesta a una consulta técnica, di EXACTAMENTE: 'Mis conocimientos se limitan a las políticas, servicios y procedimientos del Departamento de Bienestar Politécnico.'

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
    
    print("🚀 API levantada y lista para recibir peticiones.")
    yield

app = FastAPI(title="API RAG para Asistente 3D", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ENDPOINTS DE ADMINISTRACIÓN 
# ==========================================

@app.get("/api/documents")
async def list_documents():
    if mongo_collection is None:
        raise HTTPException(status_code=500, detail="Base de datos no conectada.")
    try:
        docs = mongo_collection.find({}, {"filename": 1, "_id": 0})
        files = [doc["filename"] for doc in docs]
        return {"documents": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NUEVO ENDPOINT: Descargar archivo específico
@app.get("/api/documents/{filename}")
async def download_document(filename: str):
    if mongo_collection is None:
        raise HTTPException(status_code=500, detail="Base de datos no conectada.")
    try:
        doc = mongo_collection.find_one({"filename": filename})
        if not doc:
            raise HTTPException(status_code=404, detail="Archivo no encontrado en la base de datos.")
        
        # Devolvemos el texto plano forzando la descarga con las cabeceras HTTP
        return Response(
            content=doc.get("content", ""), 
            media_type="text/plain", 
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al recuperar el archivo: {str(e)}")

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .txt")
    if vector_store is None or mongo_collection is None:
        raise HTTPException(status_code=500, detail="Las bases de datos no están listas.")

    try:
        content_bytes = await file.read()
        text_content = content_bytes.decode("utf-8")
        
        # Blindaje contra archivos vacíos (Evita el bug 'got [] in upsert')
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="El archivo está vacío o no contiene texto legible.")
        
        mongo_collection.update_one(
            {"filename": file.filename},
            {"$set": {"filename": file.filename, "content": text_content}},
            upsert=True
        )
        
        vector_store._collection.delete(where={"source": file.filename})
        
        # Chunking Semántico
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
                    doc = Document(
                        page_content=f"[{current_section} {current_title}]\n{content_str}",
                        metadata={"source": file.filename, "section": current_section, "title": current_title}
                    )
                    semantic_chunks.append(doc)
                current_section = match.group(1)
                current_title = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
                
        content_str = "\n".join(current_content).strip()
        if content_str:
            doc = Document(
                page_content=f"[{current_section} {current_title}]\n{content_str}",
                metadata={"source": file.filename, "section": current_section, "title": current_title}
            )
            semantic_chunks.append(doc)

        final_chunks = []
        fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)

        for chunk in semantic_chunks:
            if len(chunk.page_content) > 1000:
                sub_chunks = fallback_splitter.split_documents([chunk])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        # Doble blindaje
        if not final_chunks:
             raise HTTPException(status_code=400, detail="No se pudieron generar fragmentos válidos del texto.")
        
        vector_store.add_documents(final_chunks)
        return {"message": f"Archivo '{file.filename}' guardado. {len(final_chunks)} chunks semánticos indexados en ChromaDB."}
    
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    if vector_store is None or mongo_collection is None:
        raise HTTPException(status_code=500, detail="Las bases de datos no están listas.")
    try:
        result = mongo_collection.delete_one({"filename": filename})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Archivo no encontrado en la base de datos.")
        vector_store._collection.delete(where={"source": filename})
        return {"message": f"Documento '{filename}' eliminado de MongoDB y purgado de ChromaDB."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando el documento: {str(e)}")

# ==========================================
# ENDPOINTS DE CHAT
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
        
        # NUEVO: Si no hay similitud, pasamos contexto vacío para permitir saludos
        if not resultados or resultados[0][1] > MAX_DISTANCE:
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
    if not rag_chain_with_history or not vector_store:
        raise HTTPException(status_code=500, detail="El sistema RAG no se inicializó correctamente.")
    
    async def event_generator():
        try:
            resultados = vector_store.similarity_search_with_score(chat_req.pregunta, k=3)
            
            # NUEVO: Puente de contexto vacío para permitir saludos
            if not resultados or resultados[0][1] > MAX_DISTANCE:
                contexto_crudo = ""
            else:
                textos_contexto = [doc.page_content for doc, score in resultados]
                contexto_crudo = "\n\n".join(textos_contexto)

            async for chunk in rag_chain_with_history.astream(
                {"question": chat_req.pregunta, "context": contexto_crudo},
                config={"configurable": {"session_id": chat_req.session_id}}
            ):
                if await request.is_disconnected():
                    print("Cliente desconectado. Abortando generación RAG...")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)