import os

from chromadb.config import Settings

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredHTMLLoader


# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

MODELS_PATH = "./models"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# Context Window and Max New Tokens
CONTEXT_WINDOW_SIZE = 8096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

#### If you get a "not enough space in the buffer" error, you should reduce the values below, start with half of the original values and keep halving the value until the error stops appearing

# N_GPU_LAYERS = 100  # Llama-2-70B has 83 layers
N_GPU_LAYERS = 35  # How many LLM layers to offload to GPU
N_BATCH = 512


# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
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

#### SELECT AN OPEN SOURCE LLM (LARGE LANGUAGE MODEL)
# Select the Model ID and model_basename
# load the LLM for generating Natural Language responses

# https://huggingface.co/TheBloke/DiscoLM_German_7b_v1-GGUF
MODEL_ID = "TheBloke/DiscoLM_German_7b_v1-GGUF"  # Hugging Face repo
MODEL_BASENAME = "discolm_german_7b_v1.Q5_K_M.gguf" # large, very low quality loss - recommended
# MODEL_BASENAME = "discolm_german_7b_v1.Q6_K.gguf" # very large, extremely low quality loss
# MODEL_BASENAME = "discolm_german_7b_v1.Q8_0.gguf" # very large, extremely low quality loss - not recommended
