# python crazy.py --save_qa --show_sources
# python run_dispute.py --save_qa --rounds 5 --show_sources
# git clone --branch ilias git@github.com:gelpkmar/dispute-machine.git
# history -c && history -w

import os, torch, csv, json, click, logging, gc 
from datetime import datetime
from chromadb.config import Settings
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from langchain.vectorstores import Chroma
from transformers import (GenerationConfig, pipeline,)

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredFileLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from langchain.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
# print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
# print(f"Current device: {torch.cuda.current_device()}")


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
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
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

# https://huggingface.co/TheBloke/DiscoLM_German_7b_v1-GGUF
MODEL_ID = "TheBloke/DiscoLM_German_7b_v1-GGUF"  # Hugging Face repo
MODEL_BASENAME = "discolm_german_7b_v1.Q5_K_M.gguf" # large, very low quality loss - recommended
# MODEL_BASENAME = "discolm_german_7b_v1.Q6_K.gguf" # very large, extremely low quality loss
# MODEL_BASENAME = "discolm_german_7b_v1.Q8_0.gguf" # very large, extremely low quality loss - not recommended

### Load Models
def load_quantized_model_gguf_ggml(model_id, model_basename, device_type, logging):
    """
    Load a GGUF/GGML quantized model using LlamaCpp.
    """
    try:
        logging.info("Using Llamacpp for GGUF/GGML quantized models")
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=model_basename,
            resume_download=True,
            cache_dir=MODELS_PATH,
        )
        kwargs = {
            "model_path": model_path,
            "n_ctx": CONTEXT_WINDOW_SIZE,
            "max_tokens": MAX_NEW_TOKENS,
            "n_batch": N_BATCH,
        }
        if device_type.lower() == "mps":
            kwargs["n_gpu_layers"] = 1
        if device_type.lower() == "cuda":
            kwargs["n_gpu_layers"] = N_GPU_LAYERS

        return LlamaCpp(**kwargs)
    except TypeError:
        if "ggml" in model_basename:
            logging.info("If you were using GGML model, LLAMA-CPP Dropped Support, Use GGUF Instead")
        return None


# def load_quantized_model_qptq(model_id, model_basename, device_type, logging):
#     """
#     This function is disabled since it requires auto_gptq
#     """
#     logging.warning("GPTQ quantized models are not supported in this version. Please use GGUF models instead.")
#     return None, None


def load_full_model(model_id, model_basename, device_type, logging):
    """
    Load a full model using either LlamaTokenizer or AutoModelForCausalLM.
    """
    if device_type.lower() in ["mps", "cpu"]:
        logging.info("Using AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="./models/"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models/")
    else:
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models/")
        logging.info("Tokenizer loaded")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True  # ✅ Enables CPU offloading
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=MODELS_PATH,
            trust_remote_code=True,
            quantization_config=bnb_config
        )
        model.tie_weights()
    return model, tokenizer


def load_quantized_model_awq(model_id, logging):
    """
    This function is disabled since it requires special dependencies
    """
    logging.warning("AWQ quantized models are not supported in this version.")
    return None, None


### Utilities

def log_to_csv(question, answer):

    log_dir, log_file = "local_chat_history", "qa_log.csv"
    # Ensure log directory exists, create if not
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Construct the full file path
    log_path = os.path.join(log_dir, log_file)

    # Check if file exists, if not create and write headers
    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "question", "answer"])

    # Append the log entry
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, question, answer])

def save_dispute_history_to_json(agent_x_state, agent_y_state, simulation_round):
    data = {
        "simulation_round": simulation_round,
        "agent_X": {
            "name": agent_x_state['name'],
            "history": agent_x_state['dispute_history'],
        },
        "agent_Y": {
            "name": agent_y_state['name'],
            "history": agent_y_state['dispute_history'],
        }
    }

    # Make sure the output folder exists
    os.makedirs("saved_histories", exist_ok=True)
    filename = f"saved_histories/simulation_round_{simulation_round}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"✅ Dispute history saved to {filename}")



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
    



### LocalGPT

system_prompt = """ """


def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None, history=False):
    if promptTemplate_type == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    elif promptTemplate_type == "llama3":

        B_INST, E_INST = "<|start_header_id|>user<|end_header_id|>", "<|eot_id|>"
        B_SYS, E_SYS = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> ", "<|eot_id|>"
        ASSISTANT_INST = "<|start_header_id|>assistant<|end_header_id|>"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = SYSTEM_PROMPT + B_INST + instruction + ASSISTANT_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = SYSTEM_PROMPT + B_INST + instruction + ASSISTANT_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    elif promptTemplate_type == "mistral":
        B_INST, E_INST = "<s>[INST] ", " [/INST]"
        if history:
            prompt_template = (
                B_INST
                + system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                B_INST
                + system_prompt
                + """
            
            Context: {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    else:
        # change this based on the model you have selected.
        if history:
            prompt_template = (
                system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                system_prompt
                + """
            
            Context: {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    print(f"Here is the prompt used: {prompt}")

    return (
        prompt,
        memory,
    )


# In the load_model function, modify the quantization checks:
def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")
    
    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            if llm is not None:
                return llm
            else:
                LOGGING.warning("GGUF/GGML model loading failed, falling back to full model")
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            if model is None or tokenizer is None:
                LOGGING.warning("GGML model loading failed, falling back to full model")
        elif ".awq" in model_basename.lower():
            LOGGING.warning("AWQ models not supported, falling back to full model")
        else:
            LOGGING.warning("GPTQ models not supported, falling back to full model")
    
    # If we get here, either no model_basename was provided or quantization failed
    model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    generation_config = GenerationConfig.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")
    return local_llm



def load_embeddings():
    """Load HuggingFace Embeddings object onto Gaudi or CPU"""
    
    logging.info("Loading embedding model on cpu")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"}
    )
    return embeddings


def calculate_similarity(model, response, expected_answer):
    """Calculate similarity between response and expected answer using the model"""
    response_embedding = model.client.encode(response, convert_to_tensor=True).squeeze()
    expected_embedding = model.client.encode(
        expected_answer, convert_to_tensor=True
    ).squeeze()
    similarity_score = torch.nn.functional.cosine_similarity(
        response_embedding, expected_embedding, dim=0
    )
    return similarity_score.item()

def retrieval_qa_pipline(device_type, use_history, persist_directory, promptTemplate_type="llama"):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within ingest.py.

    (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """
    embeddings = load_embeddings()
    # if device_type == "hpu":
    #     embeddings = load_embeddings()
    # else:
        # embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    # load the vectorstore
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # load the llm pipeline
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa


### AGENT

class Agent:
    def __init__(self, name, embeddings_dir, device_type, use_history, model_type, persist_dir, promptTemplate_type, agent_state):
        self.name = name  
        self.embeddings_dir = embeddings_dir  
        self.persist_dir = persist_dir
        self.promptTemplate_type = promptTemplate_type
        self.agent_state = agent_state
        # self.opening_statement = opening_statement
        self.qa = retrieval_qa_pipline(device_type, use_history, self.persist_dir, promptTemplate_type=model_type)

    def ask(self, query):
        complete_query = query
        res = self.qa(complete_query)
        return res

def prompt(agent_state: dict, round_number: int, total_rounds: int) -> str:
    if round_number == total_rounds-1:
        return f"""
        Du bist {agent_state['name']}, {agent_state['age']} Jahre alt und {agent_state['gender']}. Du befindest dich aktuell in einem Streit mit {agent_state['dispute_partner']}.

        Aktuelle Situation:
        - Deine Rolle: {agent_state['dispute_context']['context']}
        - Rechtliches Thema: {agent_state['legal_issue_involved']}
        - Streitkontext: {agent_state['dispute_context']['facts']}
        - Bevorzugte Lösung: {agent_state['dispute_context']['preferred_resolution']}
        - Streitverlauf: {agent_state['dispute_history'][-1]}

        Dies ist die letzte Runde deines Streits mit {agent_state['dispute_partner']}. Du hast bisher nach einer Einigung gesucht. 
        
        Reflektiere über den gesamten Streit und beantworte folgende Fragen:
        1. Hat {agent_state['dispute_partner']} deine Wünsche berücksichtigt?
        2. Bist du bereit, zu einer Einigung zu kommen oder nicht?
        3. Falls du weiterhin unzufrieden bist, erkläre warum und ob du den Streit vor Gericht bringen möchtest.

        {agent_state['characteristics']}.
        Bitte triff eine endgültige Entscheidung, entweder zu einer Einigung mit {agent_state['name']} zu kommen oder den Streit vor Gericht zu bringen.
        """
    else:
        return f"""
        Du bist {agent_state['name']}, {agent_state['age']} Jahre alt und {agent_state['gender']}. Du befindest dich aktuell in einem Streit mit {agent_state['dispute_partner']}.

        Aktuelle Situation:
        - Deine Rolle: {agent_state['dispute_context']['context']}
        - Rechtliches Thema: {agent_state['legal_issue_involved']}
        - Streitkontext: {agent_state['dispute_context']['facts']}
        - Bevorzugte Lösung: {agent_state['dispute_context']['preferred_resolution']}
        - Streitverlauf: {agent_state['dispute_history'][-1]}

        Basierend auf den obigen Informationen, sollte deine Antwort die folgenden Punkte behandeln:
        1. Drücke Frustration über die verspätete Lieferung aus.
        2. Betone deine finanziellen Verluste und unterstreiche die Ernsthaftigkeit des Problems.
        3. Fordere einen Kompromiss und/oder eine Vertragsänderung.
        Antworte immer auf die untenstehende Aussage von {agent_state['dispute_partner']} unter Berücksichtigung der oben stehenden Daten (inkl. Streitverlauf).
        {agent_state['characteristics']}. 
        """

# Example of updating agent state after each round
agent_state_x = {
    'name': 'Biagio Badel',
    'age': 33,
    'gender': 'Männlich',
    'legal_issue_involved': 'Preisnachlass wegen verspäteter Lieferung',
    'characteristics': 'Deine finanzielle Lage ist stabil, und du hast ein klares Ziel, finanzielle Entschädigung für den Vertragsbruch zu erhalten. Deine Art ist emotional und kooperativ, aber aggressiv, wenn du deine Position verteidigst.',
    'dispute_context': {
        'context': 'Du bist der Verkäufer von Möbeln und hast vereinbarte Waren zu spät geliefert. Nun befindest du dich in einem Streit um Wiedergutmachung, der kurz davor ist, vor Gericht zu gehen und möchtest die entstandenen Schäden so klein wie möglich halten.', 
        'facts': 'Streit über Servicequalität',
        'preferred_resolution': 'Akzeptanz des Preisnachlasses von maximal 10%.'
    },
    'dispute_partner': 'Hilda Sidler',
    'dispute_history': [[]]
}

agent_state_y = {
    'name': 'Hilda Sidler',
    'age': 64,
    'gender': 'Weiblich',
    'legal_issue_involved': 'Preisnachlass wegen verspäteter Lieferung',
    'characteristics': 'Deine finanzielle Lage ist stabil, und du hast ein klares Ziel, finanzielle Entschädigung für den Vertragsbruch zu erhalten. Deine Art ist emotional und kooperativ, aber aggressiv, wenn du deine Position verteidigst.',
    'dispute_context': {
        'context': 'Du bist der Käufer von Möbeln und hast vereinbarte Waren zu spät geliefert bekommen. Nun befindest du dich in einem Streit um Wiedergutmachung, der kurz davor ist, vor Gericht zu gehen und möchtest einen Preisnachlass erreichen.',
        'facts': 'Streit über Servicequalität',
        'preferred_resolution': 'Preisnachlass von 25% oder kostenlose Lieferung zusätzlicher Artikel wie Stehlampen.'
    },
    'dispute_partner': 'Biagio Badel',
    'dispute_history': [[]]
}


@click.command()
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="llama3",
    type=click.Choice(
        ["llama3", "llama", "mistral", "non_llama"],
    ),
    help="model type, llama3, llama, mistral or non_llama",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="whether to save Q&A pairs to a CSV file (Default is False)",
)
@click.option(
    "--rounds", 
    "-r", 
    default=3,  # Default number of rounds
    type=int,   # Ensure it expects an integer value
    help="Number of rounds for the discussion"
)
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(["cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip", "ve", "fpga", "ort", "xla", "lazy", "vulkan", "mps", "meta", "hpu", "mtia"]),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--number_of_simulations",
    "-n", 
    default=1,  # Default number of rounds
    type=int,   # Ensure it expects an integer value
    help="Number of simulations"
)
def main(device_type, show_sources, use_history, model_type, save_qa, rounds, number_of_simulations):
    for simulation_round in range(1, number_of_simulations + 1):
        logging.info(f"Simulation Round {simulation_round}:")
        logging.info(f"Running on: {device_type}")
        logging.info(f"Display Source Documents set to: {show_sources}")
        logging.info(f"Use history set to: {use_history}")

        if not os.path.exists(MODELS_PATH):
            os.mkdir(MODELS_PATH)

        # ⚡ Use no_grad to disable gradient tracking and save memory
        with torch.no_grad():

            agent_X = Agent(
                name="X", 
                embeddings_dir="./data/embeddings_X",
                device_type="cuda",
                use_history=False, 
                model_type=model_type, 
                persist_dir="./data/persist_X", 
                promptTemplate_type=model_type, 
                agent_state=agent_state_x
            )
            agent_Y = Agent(
                name="Y", 
                embeddings_dir="./data/embeddings_Y",
                device_type="cuda",
                use_history=False, 
                model_type=model_type, 
                persist_dir="./data/persist_Y", 
                promptTemplate_type=model_type,
                agent_state=agent_state_y
            )

            current_context = ""
            for round_num in range(1, rounds + 1): 
                logging.info(f"Simulation {simulation_round}, Round {round_num}:")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Agent X speaks
                agent_X_response = agent_X.ask(prompt(agent_state_x, round_num, rounds) + current_context)
                answer_X, docs = agent_X_response["result"], agent_X_response["source_documents"]
                logging.info(f"Die neuste Aussage von {agent_state_x['name']}: {agent_X_response}")

                if save_qa:
                    log_to_csv(f'Simulation {simulation_round}, Round{round_num}: {current_context}', answer_X)

                current_context = f"\nDie neuste Aussage von {agent_state_x['name']}: {answer_X}"
                agent_state_x['dispute_history'].append([current_context, answer_X])

                # Agent Y speaks
                agent_Y_response = agent_Y.ask(prompt(agent_state_y, round_num, rounds) + current_context)
                answer_Y, docs = agent_Y_response["result"], agent_Y_response["source_documents"]
                logging.info(f"Die neuste Aussage von {agent_state_y['name']}: {agent_Y_response}")

                if save_qa:
                    log_to_csv(f'Simulation {simulation_round}, Round{round_num}: {current_context}', answer_Y)

                current_context = f"\nDie neuste Aussage von {agent_state_y['name']}: {answer_Y}"
                agent_state_y['dispute_history'].append([current_context, answer_Y])

            save_dispute_history_to_json(agent_state_x, agent_state_y, simulation_round)

            # ✅ After each simulation round: clean up
            del agent_X
            del agent_Y
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
