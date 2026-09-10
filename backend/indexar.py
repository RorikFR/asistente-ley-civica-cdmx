import os
import json
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

load_dotenv()

# 1. Configurar Modelos y Base de Datos
embeddings = HuggingFaceEndpointEmbeddings(
    model="intfloat/multilingual-e5-large",
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

vector_store = Chroma(
    collection_name="leyes_hijos",
    embedding_function=embeddings,
    persist_directory="./chroma_data"
)
store = InMemoryStore()

# 2. Configurar Splitters y Recuperador
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200, separators=["\n\nArtículo", "\n\nARTÍCULO", "\n\n", "."])
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 3. Procesar el PDF local
print("Leyendo documento.pdf...")
loader = PyMuPDFLoader("documento.pdf") # Tu archivo fijo en el backend
paginas = loader.load()

texto_completo = ""
for pagina in paginas:
    texto_completo += pagina.page_content + "\n\n"

texto_completo = texto_completo.replace("-\n", "")
texto_completo = texto_completo.replace("\n\n", "||PARRAFO||")
texto_completo = texto_completo.replace("\n", " ")
texto_completo = texto_completo.replace("||PARRAFO||", "\n\n")

doc_unificado = [Document(page_content=texto_completo, metadata={"source": "normativa_local"})]

print("Generando embeddings y fragmentos (esto puede tardar unos segundos)...")
retriever.add_documents(doc_unificado)

# 4. Hacer persistente la memoria RAM (Documentos Padre)
print("Guardando Documentos Padre en disco...")
datos_padres = {k: v.dict() for k, v in store.store.items()}
with open("padres_data.json", "w", encoding="utf-8") as f:
    json.dump(datos_padres, f, ensure_ascii=False, indent=4)

print("¡Proceso finalizado! Base de datos lista para ser consultada.")