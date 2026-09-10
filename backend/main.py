import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

load_dotenv()

app = FastAPI(title="API Navegador de Leyes RAG")

embeddings = HuggingFaceEndpointEmbeddings(
    model="intfloat/multilingual-e5-large",
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b", 
    temperature=0.0
)

# Cargar Base de Datos Vectorial
vector_store = Chroma(
    collection_name="leyes_hijos",
    embedding_function=embeddings,
    persist_directory="./chroma_data"
)

# Reconstruir la memoria de los Documentos Padre desde el disco
store = InMemoryStore()
if os.path.exists("padres_data.json"):
    with open("padres_data.json", "r", encoding="utf-8") as f:
        datos_padres = json.load(f)
        store.store = {k: Document(**v) for k, v in datos_padres.items()}

# Configurar el Recuperador
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 4}
)

class QueryRequest(BaseModel):
    pregunta: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    pregunta_formateada = f"query: {request.pregunta}"
    documentos_recuperados = retriever.invoke(pregunta_formateada)
    
    contexto_str = "\n\n".join([doc.page_content for doc in documentos_recuperados])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente legal informativo.
        Tu tarea es responder a la pregunta del usuario basándote ÚNICAMENTE en el siguiente contexto legal.
        Proporciona una respuesta concisa y estructurada.
        Los subpárrafos de los artículos deben mostrarse en numerales romanos, solamente realiza la conversión, no lo menciones en tu respuesta.
        Prefiere el formato de lista sobre tablas a menos que la información sea muy densa.
        
        REGLA ESTRICTA: Dentro de tu respuesta, DEBES mencionar explícitamente el o los "Artículos" o "Capítulos" de la normativa en los que te estás basando.
        REGLA ESTRICTA: No se deberán convertir los números decimales a romanos si no son subpárrafos de un artículo o expresen cantidades y tiempo.
        
        Si la respuesta definitivamente no se encuentra en el contexto, responde textualmente: 'No pude encontrar esa información en la versión actual del documento (15/06/2022).'
        
        Contexto legal:
        {contexto}"""),
        ("human", "{pregunta}")
    ])
    
    chain = prompt | llm
    respuesta = chain.invoke({"contexto": contexto_str, "pregunta": request.pregunta})
    
    fuentes = [{"extracto": doc.page_content[:300] + "..."} for doc in documentos_recuperados]
    
    return {"respuesta": respuesta.content, "fuentes": fuentes}