import logging, os, shutil, click, torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import (
    HuggingFaceInstructEmbeddings,
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings
)
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredFileLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader

# Configuration
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"
MODELS_PATH = "./models"
INGEST_THREADS = os.cpu_count() or 8
CONTEXT_WINDOW_SIZE = 8096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)
N_GPU_LAYERS = 35  # How many LLM layers to offload to GPU
N_BATCH = 512
# SCRIPT_PATH = "/Users/thealteredmg/kDrive_altered/Studium_UZH/c_CURRENT/AIL_25/deliverable/dispute-machine/data/legal_script_txt.txt"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

DOCUMENT_MAP = {
    ".html": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    # ".pdf": PDFMinerLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Default Instructor Model
EMBEDDING_MODEL_NAME = "sentence-transformers/LaBSE"
# EMBEDDING_MODEL_NAME = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" # Uses 2.5 GB of VRAM
# EMBEDDING_MODEL_NAME = "intfloat/e5-mistral-7b-instruct" ####


# https://huggingface.co/TheBloke/DiscoLM_German_7b_v1-GGUF
MODEL_ID = "TheBloke/DiscoLM_German_7b_v1-GGUF"  # Hugging Face repo
MODEL_BASENAME = "discolm_german_7b_v1.Q5_K_M.gguf" # large, very low quality loss - recommended
# MODEL_BASENAME = "discolm_german_7b_v1.Q6_K.gguf" # very large, extremely low quality loss
# MODEL_BASENAME = "discolm_german_7b_v1.Q8_0.gguf" # very large, extremely low quality loss - not recommended

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def get_embeddings(device_type="cuda"):
    if EMBEDDING_MODEL_NAME == "hkunlp/instructor-large":
        # Special handling for instructor model
        model_kwargs = {"device": device_type}
        encode_kwargs = {"normalize_embeddings": True}
        return HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
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
    file1 = open("file_ingest.log", "a")
    file1.write(logentry + "\n")
    file1.close()
    print(logentry + "\n")


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    try:
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            file_log(file_path + " loaded.")
            loader = loader_class(file_path)
        else:
            file_log(file_path + " document type is undefined.")
            raise ValueError("Document type is undefined")
        return loader.load()[0]
    except Exception as ex:
        file_log("%s loading error: \n%s" % (file_path, ex))
        return None


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        # if futures is None:
        #     file_log(name + " failed to submit")
        #     return None
        # else:
        #     data_list = [future.result() for future in futures]
        #     # return data and file paths
        #     return (data_list, filepaths)
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            print("Importing: " + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            try:
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                file_log("executor task failed: %s" % (ex))
                future = None
            if future is not None:
                futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                file_log("Exception: %s" % (ex))

    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        if doc is not None:
            file_extension = os.path.splitext(doc.metadata["source"])[1]
            if file_extension == ".py":
                python_docs.append(doc)
            else:
                text_docs.append(doc)
    return text_docs, python_docs


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(["cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip", "ve", "fpga", "ort", "xla", "lazy", "vulkan", "mps", "meta", "hpu", "mtia"]),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within fun_localGPT.py.
    
    (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """

    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()

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
    documents = load_documents(source_directory)
    if not documents:
        logging.error(f"No valid documents found in {source_directory}")
        return

    text_docs, python_docs = split_documents(documents)
    
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
    embeddings = get_embeddings(device_type)
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