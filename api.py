from typing import Dict
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama # IMPORTANTE: Cambiado a ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

# Configuración inicial
CHROMA_PATH = "./db"
OLLAMA_MODEL = "phi3" 

# Variables globales para los componentes
retriever = None
rag_chain_with_history = None

# Almacén de memoria en RAM (Diccionario que guarda el historial por ID de sesión)
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Obtiene o crea un historial de chat temporal en la RAM para una sesión."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, rag_chain_with_history
    print("Inicializando componentes RAG y Memoria...")

    # 1. Cargar Embeddings (Forzado en CPU)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 2. Conectar a ChromaDB local
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"Advertencia: No se pudo cargar ChromaDB. Error: {e}")

    # 3. Configurar ChatOllama (Especializado en roles de conversación)
    llm = ChatOllama(model=OLLAMA_MODEL)

    # 4. Crear el Chat Prompt Template
    system_prompt = """Eres un asistente conversacional (centrado en tu contexto) que responde de manera natural (directo y conciso, sin mucho texto, pero cordial) como en un chat normal. 

    Tu comportamiento debe seguir estas reglas:
    1. Actua como un chat conversacional ante lo que escriba el usuario (saludos, despedidas, datos, etc.), es decir, responde de manera cordial y directa ante cosas que no tienen que ver con el contexto pero SIN VIOLAR LAS RESTRICCIONES.Siempre guia al usuario hacia tu contexto.
    2. Recuerda los datos que el usuario te da durante la conversación (ej. su nombre, datos, hechos) y úsalos para responder preguntas relacionadas con esa información.
    2. Tu UNICA restriccion son las preguntas, si el usuario te pregunta algo que no esta en tu base de datos o pregunta por informacion que no te ha proporcionado durante la conversacion, debes responder exactamente: "Mi base de datos actual está limitada a las instalaciones del Metaverso de InnovaTech."


    Contexto recuperado:
    {context}"""

    # El prompt ahora acepta mensajes estructurados: Sistema, Historial (memoria) y Humano
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Construir la cadena con asignación de contexto
    contextualize_q = RunnablePassthrough.assign(
        context=(lambda x: x["question"]) | retriever | format_docs
    )

    rag_chain = (
        contextualize_q
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Envolver la cadena con el gestor de historial de LangChain
    rag_chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    
    print("Componentes inicializados. API con Memoria lista.")
    yield

app = FastAPI(title="API RAG para Asistente 3D", version="1.0.0", lifespan=lifespan)

# Configuración de CORS para permitir que el Frontend HTML se conecte
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class ChatRequest(BaseModel):
    pregunta: str
    session_id: str = "sesion_default" # Añadimos session_id para identificar usuarios únicos

class ChatResponse(BaseModel):
    respuesta: str
    contexto_utilizado: list[str] = []

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not rag_chain_with_history:
        raise HTTPException(status_code=500, detail="El sistema RAG no se inicializó correctamente.")
    
    try:
        # Recuperamos los documentos solo para fines de depuración/visualización
        documentos_recuperados = retriever.invoke(request.pregunta)
        textos_contexto = [doc.page_content for doc in documentos_recuperados]
        
        # Ejecutamos la cadena, pasándole la configuración de la sesión (session_id)
        respuesta = rag_chain_with_history.invoke(
            {"question": request.pregunta},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        return ChatResponse(
            respuesta=respuesta,
            contexto_utilizado=textos_contexto
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # ==========================================
# ENDPOINT DE STREAMING (Respuesta en Tiempo Real)
# ==========================================
@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    if not rag_chain_with_history:
        raise HTTPException(status_code=500, detail="El sistema RAG no se inicializó correctamente.")
    
    async def event_generator():
        try:
            # Iteramos sobre la generación de forma asíncrona (token por token)
            async for chunk in rag_chain_with_history.astream(
                {"question": request.pregunta},
                config={"configurable": {"session_id": request.session_id}}
            ):
                # Empaquetamos cada fragmento en el formato estándar Server-Sent Events (SSE)
                yield f"data: {json.dumps({'respuesta': chunk})}\n\n"
            
            # Enviamos una señal para indicar que el stream ha terminado
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)