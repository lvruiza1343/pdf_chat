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
    .stApp {
        background-color: #1f1f2e;
        color: #f5f5f5;
    }

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

    .respuesta-box {
        background-color: #333;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ffcc00;
        margin-top: 10px;
        scroll-margin-top: 80px;
    }

    .scroll-button {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background-color: #ffcc00;
        color: black;
        padding: 12px 18px;
        border-radius: 50px;
        font-weight: bold;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        z-index: 9999;
    }

    .scroll-button:hover {
        background-color: #ffaa00;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Título animado
st.markdown('<div class="animated-title">Generación Aumentada por Recuperación (RAG) 💬</div>', unsafe_allow_html=True)

# Versión Python
st.write("🧠 Versión de Python:", platform.python_version())

# Imagen decorativa
try:
    image = Image.open('piton.jpg')
    st.image(image, width=700)
except Exception as e:
    st.warning(f"⚠️ No se pudo cargar la imagen: {e}")

# Sidebar
with st.sidebar:
    st.subheader("📄 Este Agente te ayudará a realizar análisis sobre el PDF cargado")

# Clave de OpenAI
ke = st.text_input('🔐 Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("🔑 Por favor ingresa tu clave de API de OpenAI para continuar")

# Subida PDF
pdf = st.file_uploader("📁 Carga el archivo PDF", type="pdf")

# Variables de estado
respuesta_final = ""

# Procesamiento si hay archivo y API
if pdf is not None and ke:
    with st.spinner("📚 Cargando y analizando el PDF..."):
        try:
            pdf_reader = PdfReader(pdf)
            text = "".join([page.extract_text() for page in pdf_reader.pages])
            st.info(f"📃 Texto extraído: {len(text)} caracteres")

            text_splitter = CharacterTextSplitter(
                separator="\n", chunk_size=500, chunk_overlap=20, length_function=len
            )
            chunks = text_splitter.split_text(text)
            st.success(f"✅ Documento dividido en {len(chunks)} fragmentos")

            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            st.subheader("❓ Escribe qué quieres saber sobre el documento")
            user_question = st.text_area("💬", placeholder="Escribe tu pregunta aquí...")

            if user_question:
                docs = knowledge_base.similarity_search(user_question)
                llm = OpenAI(temperature=0, model_name="gpt-4o")
                chain = load_qa_chain(llm, chain_type="stuff")

                response = chain.run(input_documents=docs, question=user_question)
                respuesta_final = response

                # Mostrar botón flotante para ir abajo
                st.markdown("""
                    <a href="#respuesta">
                        <button class="scroll-button">⬇ Ver respuesta</button>
                    </a>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error al procesar el PDF: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# Mostrar respuesta al final
if respuesta_final:
    st.markdown('<div id="respuesta"></div>', unsafe_allow_html=True)
    st.markdown("### 🔎 Respuesta:")
    st.markdown(f'<div class="respuesta-box">{respuesta_final}</div>', unsafe_allow_html=True)

elif pdf is not None and not ke:
    st.warning("🔐 Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("📥 Por favor carga un archivo PDF para comenzar")

