from typing import Dict
import json
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- IMPORTACIONES EMPRESARIALES ---
import chromadb
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Configuración de conexiones locales nativas
MONGO_URI = "mongodb://localhost:27017/"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8001
OLLAMA_MODEL = "phi3" 

# Variables globales
retriever = None
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
    global retriever, rag_chain_with_history, vector_store, mongo_collection
    print("Inicializando Arquitectura Enterprise (MongoDB + ChromaDB Nativo)...")

    # 1. Conectar a MongoDB local
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping') # Verificar conexión
        mongo_db = mongo_client["nexus_db"]
        mongo_collection = mongo_db["documents"]
        print("Conectado a MongoDB local con éxito.")
    except ConnectionFailure:
        print("ERROR: No se pudo conectar a MongoDB. ¿Está instalado y corriendo el servicio?")

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
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        print("Conectado a ChromaDB Server con éxito.")
    except Exception as e:
        print(f" ERROR conectando a ChromaDB: {e}. ¿Ejecutaste 'chroma run --port 8001'?")

    # 4. Configurar ChatOllama
    llm = ChatOllama(model=OLLAMA_MODEL)

    # 5. Crear el Chat Prompt Template
    system_prompt = """Eres un asistente conversacional (centrado en tu contexto) que responde de manera directa y concisa. 

    Tu comportamiento debe seguir estas reglas:
    1. Actua como un chat conversacional dirigido a estuduantes universitarios ante lo que escriba el usuario (saludos, despedidas, datos, etc.), es decir, responde de manera cordial y directa (oraciones CORTAS, y sin adornos) ante cosas que no tienen que ver con el contexto pero SIN VIOLAR LAS RESTRICCIONES. Siempre guia al usuario hacia tu contexto.
    2. Recuerda los datos que el usuario te da durante la conversación (ej. su nombre, datos, hechos) y úsalos para responder preguntas relacionadas con esa información.
    2. Tu UNICA restriccion son las preguntas, si el usuario te pregunta algo que no esta en tu base de datos o pregunta por informacion que no te ha proporcionado durante la conversacion, debes responder exactamente: "Mi base de datos actual está limitada a informacion sobre el Departamento de Bienestar Politecnico."


    Contexto recuperado:
    {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 6. Construir la cadena RAG
    contextualize_q = RunnablePassthrough.assign(
        context=(lambda x: x["question"]) | retriever | format_docs
    )

    rag_chain = (
        contextualize_q
        | prompt
        | llm
        | StrOutputParser()
    )

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
    """Obtiene la lista de documentos directamente desde MongoDB."""
    if mongo_collection is None:
        raise HTTPException(status_code=500, detail="Base de datos no conectada.")
    try:
        docs = mongo_collection.find({}, {"filename": 1, "_id": 0})
        files = [doc["filename"] for doc in docs]
        return {"documents": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    """Guarda el texto en MongoDB y los vectores en ChromaDB Server (En memoria, sin tocar el disco)."""
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .txt")
    if vector_store is None or mongo_collection is None:
        raise HTTPException(status_code=500, detail="Las bases de datos no están listas.")

    try:
        # Leer archivo en memoria RAM
        content_bytes = await file.read()
        text_content = content_bytes.decode("utf-8")
        
        # 1. Guardar en MongoDB
        mongo_collection.update_one(
            {"filename": file.filename},
            {"$set": {"filename": file.filename, "content": text_content}},
            upsert=True
        )
        
        # 2. Eliminar fragmentos antiguos en ChromaDB si existe
        vector_store._collection.delete(where={"source": file.filename})
        
        # 3. Fragmentar el texto en memoria
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.create_documents(
            texts=[text_content], 
            metadatas=[{"source": file.filename}]
        )
        
        # 4. Enviar a ChromaDB por HTTP
        vector_store.add_documents(chunks)
        
        return {"message": f"Archivo '{file.filename}' guardado en Mongo e indexado en ChromaDB."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Elimina el documento de MongoDB y limpia sus vectores de ChromaDB."""
    if vector_store is None or mongo_collection is None:
        raise HTTPException(status_code=500, detail="Las bases de datos no están listas.")
        
    try:
        # 1. Eliminar de MongoDB
        result = mongo_collection.delete_one({"filename": filename})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Archivo no encontrado en la base de datos.")
            
        # 2. Eliminar vectores en ChromaDB
        vector_store._collection.delete(where={"source": filename})
        
        return {"message": f"Documento '{filename}' eliminado de MongoDB y purgado de ChromaDB."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando el documento: {str(e)}")

# ==========================================
# ENDPOINTS DE CHAT (Streaming e Invocación)
# ==========================================

class ChatRequest(BaseModel):
    pregunta: str
    session_id: str = "sesion_default"

class ChatResponse(BaseModel):
    respuesta: str
    contexto_utilizado: list[str] = []

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not rag_chain_with_history:
        raise HTTPException(status_code=500, detail="El sistema RAG no se inicializó correctamente.")
    try:
        documentos_recuperados = retriever.invoke(request.pregunta)
        textos_contexto = [doc.page_content for doc in documentos_recuperados]
        respuesta = rag_chain_with_history.invoke(
            {"question": request.pregunta},
            config={"configurable": {"session_id": request.session_id}}
        )
        return ChatResponse(respuesta=respuesta, contexto_utilizado=textos_contexto)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chat/stream")
async def chat_stream_endpoint(request: Request, chat_req: ChatRequest):
    if not rag_chain_with_history:
        raise HTTPException(status_code=500, detail="El sistema RAG no se inicializó correctamente.")
    
    async def event_generator():
        try:
            async for chunk in rag_chain_with_history.astream(
                {"question": chat_req.pregunta},
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