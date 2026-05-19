from typing import Dict
import json
import asyncio
import os
import shutil
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

# Importaciones para la Ingesta Dinámica (Traídas de ingest.py)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuración inicial
CHROMA_PATH = "./db"
DATA_PATH = "./data"
OLLAMA_MODEL = "phi3" 

# Variables globales para los componentes
retriever = None
rag_chain_with_history = None
vector_store = None # Promovemos la BD a global para poder administrarla

# Almacén de memoria en RAM
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, rag_chain_with_history, vector_store
    print("Inicializando componentes RAG y Memoria...")

    # Crear carpeta de datos si no existe
    os.makedirs(DATA_PATH, exist_ok=True)

    # 1. Cargar Embeddings (Forzado en CPU)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 2. Conectar a ChromaDB local (Ahora es global)
    try:
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"Advertencia: No se pudo cargar ChromaDB. Error: {e}")

    # 3. Configurar ChatOllama
    llm = ChatOllama(model=OLLAMA_MODEL)

    # 4. Crear el Chat Prompt Template
    system_prompt = """Eres un asistente conversacional (centrado en tu contexto) que responde de manera directa y concisa. 

    Tu comportamiento debe seguir estas reglas:
    1. Actua como un chat conversacional dirigido a estuduantes universitarios ante lo que escriba el usuario (saludos, despedidas, datos, etc.), es decir, responde de manera cordial y directa (oraciones CORTAS, y sin adornos) ante cosas que no tienen que ver con el contexto pero SIN VIOLAR LAS RESTRICCIONES. Siempre guia al usuario hacia tu contexto.
    2. Recuerda los datos que el usuario te da durante la conversación (ej. su nombre, datos, hechos) y úsalos para responder preguntas relacionadas con esa información.
    2. Tu UNICA restriccion son las preguntas, si el usuario te pregunta algo que no esta en tu base de datos o pregunta por informacion que no te ha proporcionado durante la conversacion, debes responder exactamente: "Mi base de datos actual está limitada a las instalaciones del Metaverso de InnovaTech."


    Contexto recuperado:
    {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Construir la cadena RAG
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
    
    print("Componentes inicializados. API lista.")
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
# ENDPOINTS DE ADMINISTRACIÓN (NUEVOS)
# ==========================================

@app.get("/api/documents")
async def list_documents():
    """Lista todos los archivos .txt actualmente en la carpeta de datos."""
    try:
        files = [f for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
        return {"documents": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    """Sube un archivo txt y lo procesa instantáneamente en la base de datos vectorial."""
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .txt")
    if not vector_store:
        raise HTTPException(status_code=500, detail="La base de datos vectorial no está lista.")

    file_path = os.path.join(DATA_PATH, file.filename)
    
    # Guardar archivo físico
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        # LÓGICA DE INGESTA (Emulando ingest.py)
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(docs)
        
        # Añadir a ChromaDB
        vector_store.add_documents(chunks)
        
        return {"message": f"Archivo '{file.filename}' procesado y añadido al conocimiento de NEXUS-7."}
    except Exception as e:
        # Rollback en caso de error
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Elimina el archivo físico y purga sus fragmentos de la base vectorial."""
    if not vector_store:
        raise HTTPException(status_code=500, detail="La base de datos vectorial no está lista.")
        
    file_path = os.path.join(DATA_PATH, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
        
    try:
        # 1. Eliminar de ChromaDB usando el metadata 'source' que inyecta Langchain
        vector_store._collection.delete(where={"source": file_path})
        
        # 2. Eliminar archivo físico
        os.remove(file_path)
        
        return {"message": f"Archivo '{filename}' y su conocimiento asociado han sido eliminados."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando el documento: {str(e)}")

# ==========================================
# ENDPOINTS DE CHAT (EXISTENTES)
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