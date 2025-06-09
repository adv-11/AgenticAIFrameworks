from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os
import shutil
from .utils import load_config
import streamlit as st
import logging

config = load_config()
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = "./documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

@st.cache_resource
def get_embedding_function():
    try:
        return FastEmbedEmbeddings(
            model_name=config['embedding_model'],
            max_length=512,
            doc_embed_type="passage",
            cache_dir="./models"
        )
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        st.error(f"Error loading embedding model: {str(e)}")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(cache_folder="./models")

@st.cache_resource
def get_vectorstore():
    embeddings = get_embedding_function()
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def process_documents(uploaded_files, rebuild=False):
    if rebuild:
        clear_vectorstore()

    documents = []
    for file in uploaded_files:
        file_path = os.path.join(DOCUMENTS_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} pages from {file.name}")
        for doc in docs:
            doc.metadata['source'] = file.name
        documents.extend(docs)

    if not documents:
        logger.warning("No documents were loaded.")
        return 0

    chunk_size = int(config['chunk_size'])
    chunk_overlap = min(int(config['chunk_overlap']), chunk_size - 1)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(texts)} chunks")

    if not texts:
        logger.warning("No text chunks were created after splitting.")
        return 0

    vectorstore = get_vectorstore()
    try:
        vectorstore.add_documents(texts)
        vectorstore.persist()
        logger.info(f"Added {len(texts)} chunks to the vector store")
    except Exception as e:
        logger.error(f"Error adding documents to vector store: {str(e)}")
        raise

    return len(texts)

