import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuración de rutas
DATA_PATH = "./data/datos_prueba.txt"
CHROMA_PATH = "./db"

def ingest_data():
    print("1. Cargando documento...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encontró el archivo en {DATA_PATH}")
        return
    
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()

    print("2. Fragmentando el texto...")
    # Ajusta chunk_size y chunk_overlap según la naturaleza de tus datos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Se generaron {len(chunks)} fragmentos.")

    print("3. Configurando Embeddings en CPU...")
    # RESTRICCIÓN CRÍTICA: Forzamos la ejecución en CPU
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print("4. Guardando en ChromaDB local...")
    # ChromaDB creará la base de datos persistente en la carpeta ./db
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    print("¡Ingesta completada con éxito!")

if __name__ == "__main__":
    # Crea la carpeta de datos si no existe y un archivo de prueba
    os.makedirs("./data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write("Este es un documento de prueba para el asistente virtual 3D. El sistema RAG funciona correctamente en CPU.")
    
    ingest_data()