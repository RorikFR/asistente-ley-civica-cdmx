import os
import json
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore

st.set_page_config(page_title="RAG Crashcourse", page_icon="⚖️", layout="wide")
st.title("Asistente para la Ley de Cultura Cívica de la CDMX")
st.subheader("Aquí puedes aclarar todas tus dudas sobre cualquier artículo.")
st.divider()

@st.cache_resource
def iniciar_sistema_rag():
    #Streamlit Secrets
    hf_token = st.secrets["HF_TOKEN"]
    groq_key = st.secrets["GROQ_API_KEY"]
    #HuggingFace inference model
    embeddings = HuggingFaceEndpointEmbeddings(
        model="intfloat/multilingual-e5-large",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token
    )
    #Groq model
    llm = ChatGroq(
        api_key=groq_key,
        model="openai/gpt-oss-120b", 
        temperature=0.0
    )
    #Vector database
    vector_store = Chroma(
        collection_name="leyes_hijos",
        embedding_function=embeddings,
        persist_directory="./chroma_data"
    )

    store = InMemoryStore()
    if os.path.exists("padres_data.json"):
        with open("padres_data.json", "r", encoding="utf-8") as f:
            datos_padres = json.load(f)
            store.store = {k: Document(**v) for k, v in datos_padres.items()}

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 4}
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente legal experto en la Ley de Cultura Cívica de la CDMX.
        Tu tarea es responder a la pregunta basándote ÚNICAMENTE en el siguiente contexto.
        DEBES mencionar explícitamente el o los "Artículos" o "Fracciones" de la normativa en los que te basas.
        Si la respuesta no está en el contexto, responde: 'No pude encontrar esa información en la versión actual del documento (15/06/2022).'
        Contexto legal:\n{contexto}"""),
        ("human", "{pregunta}")
    ])
    
    chain = prompt | llm
    return retriever, chain

# Cargar el motor
with st.spinner("Inicializando base de datos..."):
    retriever, chain = iniciar_sistema_rag()

# Logica del chat
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Renderizar mensajes anteriores
for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])
        if "fuentes" in msg:
            with st.expander("Ver artículos de referencia"):
                for i, fuente in enumerate(msg["fuentes"]):
                    st.caption(f"**Fuente {i+1}:** {fuente['extracto']}")

# Entrada de usuario
pregunta = st.chat_input("Ejemplo: ¿Para qué sirve la Ley Cívica de la CDMX?")

if pregunta:
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando artículos, por favor espera..."):
            
            pregunta_formateada = f"query: {pregunta}"
            docs = retriever.invoke(pregunta_formateada)
            contexto_str = "\n\n".join([doc.page_content for doc in docs])
            
            respuesta = chain.invoke({"contexto": contexto_str, "pregunta": pregunta})
            respuesta_texto = respuesta.content
            
            fuentes = [{"extracto": doc.page_content[:300] + "..."} for doc in docs]
            # -------------------------------------------
            
            st.markdown(respuesta_texto)
            
            with st.expander("Fuentes"):
                for i, fuente in enumerate(fuentes):
                    st.caption(f"**Fuente {i+1}:** {fuente['extracto']}")
            
            st.session_state.mensajes.append({
                "rol": "assistant", 
                "contenido": respuesta_texto,
                "fuentes": fuentes
            })