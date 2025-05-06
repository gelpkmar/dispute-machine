import os
from langchain.document_loaders import CSVLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredFileLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader
from langchain.callbacks.manager import CallbackManager
from chromadb.config import Settings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response

# Configuration
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/data"
PERSIST_DIRECTORY_X = f"{ROOT_DIRECTORY}/data/persist_X"
PERSIST_DIRECTORY_Y = f"{ROOT_DIRECTORY}/data/persist_Y"
EMBEDDINGS_DIRECTORY_X = f"{ROOT_DIRECTORY}/data/embeddings_X"
EMBEDDINGS_DIRECTORY_Y = f"{ROOT_DIRECTORY}/data/embeddings_Y"
MODELS_PATH = "./models"
INGEST_THREADS = os.cpu_count() or 8
CONTEXT_WINDOW_SIZE = 8096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)
N_GPU_LAYERS = 35  # How many LLM layers to offload to GPU
N_BATCH = 512
CALLBACK_MANAGER = CallbackManager([StreamingStdOutCallbackHandler()])

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
    allow_reset=True,
    # chroma_db_impl="duckdb+parquet",
    )

DOCUMENT_MAP = {
    ".html": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Default Instructor Model
EMBEDDING_MODEL_NAME = "sentence-transformers/LaBSE"

# https://huggingface.co/TheBloke/DiscoLM_German_7b_v1-GGUF
MODEL_ID = "TheBloke/DiscoLM_German_7b_v1-GGUF"  # Hugging Face repo
#MODEL_BASENAME = "discolm_german_7b_v1.Q5_K_M.gguf" # large, very low quality loss - recommended
MODEL_BASENAME = "discolm_german_7b_v1.Q6_K.gguf" # very large, extremely low quality loss -> According to Julias is the best
# MODEL_BASENAME = "discolm_german_7b_v1.Q8_0.gguf" # very large, extremely low quality loss - not recommended
