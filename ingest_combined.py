import os
import logging
import shutil, torch
from concurrent.futures import ThreadPoolExecutor
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import (
    HuggingFaceInstructEmbeddings,
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
)

# Import settings from constants (adjust as needed)
from constants import (
    SOURCE_DIRECTORY,
    PERSIST_DIRECTORY_X,
    PERSIST_DIRECTORY_Y,
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
    level=logging.INFO
)

def get_embeddings(device_type: str = "cuda"):
    """Unified embedding model loader (from create_embeddings.py)."""
    if EMBEDDING_MODEL_NAME == "hkunlp/instructor-large":
        return HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="Represent the query for retrieval: ",
            embed_instruction="Represent the document for retrieval: "
        )
    elif "bge" in EMBEDDING_MODEL_NAME.lower():
        return HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="Represent this sentence for searching relevant passages: "
        )
    else:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            encode_kwargs={"normalize_embeddings": True}
        )

def load_single_document(file_path: str) -> Document:
    """Load a single document using the appropriate loader (from create_embeddings.py)."""
    try:
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            loader = loader_class(file_path)
            document = loader.load()[0]
            return Document(
                page_content=document.page_content,
                metadata=document.metadata
            )
        else:
            logging.warning(f"Unsupported file type: {file_path}")
            return None
    except Exception as ex:
        logging.error(f"Error loading {file_path}: {ex}")
        return None

def load_documents(source_dir: str) -> List[Document]:
    """Load all documents in parallel (optimized version)."""
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            if os.path.splitext(file_name)[1] in DOCUMENT_MAP:
                paths.append(os.path.join(root, file_name))

    docs = []
    with ThreadPoolExecutor(max_workers=INGEST_THREADS) as executor:
        future_to_path = {
            executor.submit(load_single_document, path): path
            for path in paths
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                doc = future.result()
                if doc:
                    docs.append(doc)
                    logging.info(f"Loaded: {path}")
            except Exception as ex:
                logging.error(f"Failed to load {path}: {ex}")

    return docs

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents with specialized splitters for Python/non-Python files."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=880,
        chunk_overlap=200
    )

    texts = []
    for doc in documents:
        if doc.metadata["source"].endswith(".py"):
            texts.extend(python_splitter.split_documents([doc]))
        else:
            texts.extend(text_splitter.split_documents([doc]))
    return texts

def ingest(source_subdir: str, persist_dir: str, device_type: str = "cuda"):
    """Main ingestion pipeline."""
    logging.info(f"🔍 Processing: {source_subdir} → {persist_dir}")

    # Load and split documents
    documents = load_documents(source_subdir)
    if not documents:
        logging.warning("⚠️ No documents found. Skipping.")
        return

    texts = split_documents(documents)
    logging.info(f"🧩 Split into {len(texts)} chunks")

    # Generate embeddings
    embeddings = get_embeddings(device_type)
    
    # Clear existing DB (optional)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    # Create and persist vectorstore
    Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_dir,
        client_settings=CHROMA_SETTINGS,
        collection_metadata={"hnsw:space": "cosine"}  # From create_embeddings.py
    ).persist()

    logging.info(f"✅ Embeddings saved to {persist_dir}")

if __name__ == "__main__":
    # Example usage
    portfolio_x = os.path.join(SOURCE_DIRECTORY, "portfolio_X")
    portfolio_y = os.path.join(SOURCE_DIRECTORY, "portfolio_Y")

    # Create directories if they don't exist
    os.makedirs(portfolio_x, exist_ok=True)
    os.makedirs(portfolio_y, exist_ok=True)

    # Determine device (GPU/CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run ingestion
    ingest(portfolio_x, PERSIST_DIRECTORY_X, device)
    ingest(portfolio_y, PERSIST_DIRECTORY_Y, device)