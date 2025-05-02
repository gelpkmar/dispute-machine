import logging, os, shutil, torch
from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import (
    HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
)
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('tiger')

# Configuration import
from main import SOURCE_DIRECTORY, EMBEDDINGS_DIRECTORY_X, EMBEDDINGS_DIRECTORY_Y, INGEST_THREADS, CHROMA_SETTINGS, DOCUMENT_MAP, EMBEDDING_MODEL_NAME


def get_embeddings(device_type="cuda"):
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


def file_log(logentry):
    with open("file_ingest.log", "a") as f:
        f.write(logentry + "\n")
    print(logentry)
  
def load_single_document(file_path: str) -> Document:
    try:
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            loader = loader_class(file_path)
            document = loader.load()[0]
            # Make sure the content is assigned to the page_content field
            return Document(page_content=document.page_content, metadata=document.metadata)
        else:
            logging.warning(f"{file_path} document type is undefined.")
            return None
    except Exception as ex:
        logging.error(f"{file_path} loading error: \n{ex}")
        return None


def load_document_batch(filepaths):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(filepaths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        return ([future.result() for future in futures], filepaths)


def load_documents(source_dir: str) -> list[Document]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1]
            if file_extension in DOCUMENT_MAP:
                paths.append(os.path.join(root, file_name))

    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = max(1, round(len(paths) / n_workers))
    docs = []

    with ProcessPoolExecutor(n_workers) as executor:
        futures = [
            executor.submit(load_document_batch, paths[i:i+chunksize])
            for i in range(0, len(paths), chunksize)
        ]
        for future in as_completed(futures):
            try:
                contents, _ = future.result()
                docs.extend([doc for doc in contents if doc])
            except Exception as ex:
                file_log(f"Exception: {ex}")
    return docs


def split_documents(documents: list[Document]):
    text_docs, python_docs = [], []
    for doc in documents:
        if doc:
            ext = os.path.splitext(doc.metadata["source"])[1]
            if ext == ".py":
                python_docs.append(doc)
            else:
                text_docs.append(doc)
    return text_docs, python_docs


def create_embeddings(agent_name, device_type, source_directory, persist_directory):
    source_directory = os.path.abspath(source_directory)
    persist_directory = os.path.abspath(persist_directory)

    logging.info(f"Creating embeddings for agent {agent_name}")

    os.makedirs(source_directory, exist_ok=True)
    os.makedirs(persist_directory, exist_ok=True)

    if not os.listdir(source_directory):
        logging.warning(f"Source directory is empty: {source_directory}")
        return

    documents = load_documents(source_directory)
    if not documents:
        logging.warning(f"No valid documents found in {source_directory}")
        return

    text_docs, python_docs = split_documents(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )

    texts = text_splitter.split_documents(text_docs)
    texts.extend(python_splitter.split_documents(python_docs))

    embeddings = get_embeddings(device_type)

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory,
        client_settings=CHROMA_SETTINGS,
        collection_metadata={"hnsw:space": "cosine"},
    )

    logging.info(f"Embeddings for {agent_name} created at {persist_directory}")
    return db


def main():

    os.makedirs(SOURCE_DIRECTORY, exist_ok=True)  # Ensure base data dir exists

    portfolio_x = os.path.join(SOURCE_DIRECTORY, "portfolio_X")
    portfolio_y = os.path.join(SOURCE_DIRECTORY, "portfolio_Y")

    os.makedirs(portfolio_x, exist_ok=True)
    os.makedirs(portfolio_y, exist_ok=True)

    create_embeddings("X", "cuda" if torch.cuda.is_available() else "cpu", portfolio_x, EMBEDDINGS_DIRECTORY_X)
    create_embeddings("Y", "cuda" if torch.cuda.is_available() else "cpu", portfolio_y, EMBEDDINGS_DIRECTORY_Y)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO
    )
    main()
