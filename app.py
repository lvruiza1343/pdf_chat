import os
import streamlit as st
from PIL import Image
import PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# Use the original langchain imports if you haven't installed the new packages
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import platform

# App title and presentation
st.title('Generación Aumentada por Recuperación (RAG) 💬')
st.write("Versión de Python:", platform.python_version())

# Load and display image
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar information
with st.sidebar:
    st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado")

# Get API key from user
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Default PDF example (if needed)
try:
    pdfFileObj = open('example.pdf', 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    st.success("Archivo de ejemplo cargado correctamente.")
except Exception as e:
    st.info("No se encontró el archivo de ejemplo o no pudo ser cargado.")

# PDF uploader
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        with st.spinner("Procesando el PDF..."):
            # Extract text from PDF
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Show text length information
            st.info(f"Texto extraído: {len(text)} caracteres")
            
            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=500,
                chunk_overlap=20,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            st.success(f"Documento dividido en {len(chunks)} fragmentos")
            
            # Create embeddings and knowledge base
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            
            # User question interface
            st.subheader("Escribe qué quieres saber sobre el documento")
            user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")
            
            # Process question when submitted
            if user_question:
                with st.spinner("Buscando respuesta..."):
                    docs = knowledge_base.similarity_search(user_question)
                    
                    # You can change the model here
                    llm = OpenAI(model_name="gpt-4o-mini", temperature=0)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=user_question)
                        
                        # Display token usage information
                        st.info(f"""
                        Tokens utilizados:
                        - Prompt tokens: {cb.prompt_tokens}
                        - Completion tokens: {cb.completion_tokens}
                        - Total tokens: {cb.total_tokens}
                        - Costo: ${cb.total_cost:.5f}
                        """)
                    
                    # Display the response
                    st.markdown("### Respuesta:")
                    st.markdown(response)
                    
    except Exception as e:
        st.error(f"Error al procesar el PDF: {e}")
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
