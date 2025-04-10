import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# Estilos personalizados
st.markdown("""
    <style>
    .animated-title {
        color: green;
        font-size: 36px;
        font-weight: bold;
        animation: moveText 3s infinite alternate;
        text-align: center;
        margin-bottom: 20px;
    }

    @keyframes moveText {
        0% { transform: translateX(0); }
        100% { transform: translateX(10px); }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    body {
        background-color: #1f1f2e
    }
    .stApp {
        background-color: #1f1f2e;
    }

    .animated-title {
        color: yellow;
        font-size: 36px;
        font-weight: bold;
        animation: moveText 3s infinite alternate;
        text-align: center;
        margin-bottom: 20px;
    }

    @keyframes moveText {
        0% { transform: translateX(0); }
        100% { transform: translateX(10px); }
    }
    </style>
""", unsafe_allow_html=True)


# Título con animación y color amarillo
st.markdown('<div class="animated-title">Generación Aumentada por Recuperación (RAG) 💬</div>', unsafe_allow_html=True)

# Mostrar versión de Python
st.write("Versión de Python:", platform.python_version())

# Cargar y mostrar imagen
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Información en la barra lateral
with st.sidebar:
    st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado")

# Ingreso de clave API
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Subida de archivo PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Procesamiento del PDF si está cargado
if pdf is not None and ke:
    try:
        # Extraer texto del PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Texto extraído: {len(text)} caracteres")
        
        # Dividir texto en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")
        
        # Crear embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # Pregunta del usuario
        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")
        
        # Procesar pregunta
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Modelo de lenguaje
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            
            # Cadena de preguntas y respuestas
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Ejecutar la cadena
            response = chain.run(input_documents=docs, question=user_question)
            
            # Mostrar respuesta
            st.markdown("### Respuesta:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
