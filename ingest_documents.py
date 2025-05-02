import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma

from main import (
    SOURCE_DIRECTORY,
    PERSIST_DIRECTORY_X,
    PERSIST_DIRECTORY_Y,
    CHROMA_SETTINGS,
    get_embeddings,
    DOCUMENT_MAP
)


logging.basicConfig(level=logging.INFO)

def get_all_documents(source_path):
    """Recursively load documents using appropriate loaders"""
    documents = []
    for root, _, files in os.walk(source_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            loader_class = DOCUMENT_MAP.get(ext)
            if loader_class is None:
                logging.warning(f"Unsupported file type: {file}")
                continue
            file_path = os.path.join(root, file)
            try:
                loader = loader_class(file_path)
                docs = loader.load()
                documents.extend(docs)
                logging.info(f"Loaded {len(docs)} docs from {file_path}")
            except Exception as e:
                logging.warning(f"Failed to load {file_path}: {e}")
    return documents

def ingest(source_subdir, persist_dir):
    logging.info(f"🔍 Ingesting documents from: {source_subdir}")
    documents = get_all_documents(source_subdir)
    if not documents:
        logging.warning("⚠️ No documents found. Skipping.")
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    logging.info(f"🧩 Split into {len(split_docs)} chunks")

    embeddings = get_embeddings()
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        client_settings=CHROMA_SETTINGS
    )
    vectordb.persist()
    logging.info(f"✅ Embeddings saved to {persist_dir}")

if __name__ == "__main__":
    ingest(os.path.join(SOURCE_DIRECTORY, "portfolio_X"), PERSIST_DIRECTORY_X)
    ingest(os.path.join(SOURCE_DIRECTORY, "portfolio_Y"), PERSIST_DIRECTORY_Y)
