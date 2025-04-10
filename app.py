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

# Estilos personalizados con fondo y componentes llamativos
st.markdown("""
    <style>
    /* Fondo de la app */
    .stApp {
        background-color: #1f1f2e;
        color: #f5f5f5;
    }

    /* Título animado */
    .animated-title {
        color: yellow;
        font-size: 40px;
        font-weight: bold;
        animation: moveText 2s infinite alternate;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 8px #000000;
    }

    @keyframes moveText {
        0% { transform: translateX(0); }
        100% { transform: translateX(15px); }
    }

    /* Campos de entrada */
    .stTextInput > div > div > input,
    .stTextArea > div > textarea {
        background-color: #2c2c3c;
        color: #f5f5f5;
        border: 1px solid #f5f5f5;
        border-radius: 10px;
        padding: 10px;
    }

    /* Botones */
    .stButton > button {
        background-color: #ffcc00;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #ffaa00;
        color: white;
    }

    /* Cuadro de respuesta */
    .respuesta-box {
        background-color: #333;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ffcc00;
        margin-top: 10px;
    }

    </style>
""", unsafe_allow_html=True)

# Título llamativo
st.markdown('<div class="animated-title">Generación Aumentada por Recuperación (RAG) 💬</div>', unsafe_allow_html=True)

# Mostrar versión de Python
st.write("🧠 Versión de Python:", platform.python_version())

# Cargar y mostrar imagen
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"⚠️ No se pudo cargar la imagen: {e}")

# Información en la barra lateral
with st.sidebar:
    st.subheader("📄 Este Agente te ayudará a realizar análisis sobre el PDF cargado")

# Ingreso de clave API
ke = st.text_input('🔐 Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("🔑 Por favor ingresa tu clave de API de OpenAI para continuar")

# Subida de archivo PDF
pdf = st.file_uploader("📁 Carga el archivo PDF", type="pdf")

# Procesamiento del PDF si está cargado
if pdf is not None and ke:
    try:
        # Extraer texto del PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"📃 Texto extraído: {len(text)} caracteres")
        
        # Dividir texto en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"✅ Documento dividido en {len(chunks)} fragmentos")
        
        # Crear embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # Pregunta del usuario
        st.subheader("❓ Escribe qué quieres saber sobre el documento")
        user_question = st.text_area("💬", placeholder="Escribe tu pregunta aquí...")

        # Procesar pregunta
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Modelo de lenguaje
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            
            # Cadena de preguntas y respuestas
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Ejecutar la cadena
            response = chain.run(input_documents=docs, question=user_question)
            
            # Mostrar respuesta con caja bonita
            st.markdown("### 🔎 Respuesta:")
            st.markdown(f'<div class="respuesta-box">{response}</div>', unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"❌ Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("🔐 Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("📥 Por favor carga un archivo PDF para comenzar")
