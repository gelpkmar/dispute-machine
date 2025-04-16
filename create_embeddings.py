import ingest
import utils
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
import torch
from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import (
    HuggingFaceInstructEmbeddings,
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings
)

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

# Verify CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

def create_embeddings(agent_name, device_type, source_directory, persist_directory):
    """Create and store embeddings from documents"""
    # Convert to absolute paths
    source_directory = os.path.abspath(source_directory)
    persist_directory = os.path.abspath(persist_directory)
    
    logging.info(f"Loading documents for agent {agent_name} from: {source_directory}")
    
    # Ensure directories exist
    os.makedirs(source_directory, exist_ok=True)
    os.makedirs(persist_directory, exist_ok=True)

    # Verify source directory has files
    if not os.listdir(source_directory):
        logging.error(f"Source directory is empty: {source_directory}")
        return

    # Load and split documents
    documents = ingest.load_documents(source_directory)
    if not documents:
        logging.error(f"No valid documents found in {source_directory}")
        return

    text_docs, python_docs = ingest.split_documents(documents)
    
    # Configure text splitters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=880,
        chunk_overlap=200
    )
    
    # Split documents
    texts = text_splitter.split_documents(text_docs)
    texts.extend(python_splitter.split_documents(python_docs))
    
    logging.info(f"Processed {len(documents)} documents into {len(texts)} chunks")

    # Get embeddings
    embeddings = utils.get_embeddings(device_type)
    logging.info(f"Using embeddings model: {EMBEDDING_MODEL_NAME}")

    # Clear existing collection if it exists
    if os.path.exists(persist_directory):
        logging.info(f"Clearing existing embeddings at {persist_directory}")
        shutil.rmtree(persist_directory)

    # Create new Chroma collection
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory,
        client_settings=CHROMA_SETTINGS,
        collection_metadata={"hnsw:space": "cosine"},
    )
    
    logging.info(f"Successfully created embeddings in {persist_directory}")
    return db

def main():
    """Main function to create embeddings for agents X and Y"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    # Define paths
    portfolio_x = os.path.join(data_dir, "portfolio_X")
    embeddings_x = os.path.join(data_dir, "embeddings_X")
    portfolio_y = os.path.join(data_dir, "portfolio_Y")
    embeddings_y = os.path.join(data_dir, "embeddings_Y")
    
    # Create embeddings for both agents
    create_embeddings("X", "cuda", portfolio_x, embeddings_x)
    create_embeddings("Y", "cuda", portfolio_y, embeddings_y)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO
    )
    main()