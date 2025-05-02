# python main.py --show_sources --save_qa
# python main.py --save_qa --rounds 5 --show_sources
# git clone --branch crazy git@github.com:gelpkmar/dispute-machine.git
# history -c && history -w

import os, torch, csv, json, click, logging, gc, copy
from datetime import datetime
from chromadb.config import Settings
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from langchain.vectorstores import Chroma
from transformers import (GenerationConfig, pipeline,)

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredFileLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceBgeEmbeddings,
    HuggingFaceInstructEmbeddings
)
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")

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
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

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
    # Handle specific case for DiscoLM (or similar model)
    if EMBEDDING_MODEL_NAME == "TheBloke/DiscoLM_German_7b_v1-GGUF":
        # Load the GGUF model, specify model type and device
        # Using HuggingFaceEmbeddings for standard use
        model_kwargs = {"device": device_type}
        encode_kwargs = {"normalize_embeddings": True}

        # If you're working with a specific model handler (e.g., GGUF API), 
        # you might need to use a more specialized method here.
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    
    # Fallback cases for other models (Instructor, BGE, etc.)
    elif EMBEDDING_MODEL_NAME == "hkunlp/instructor-large":
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
        # Default fallback for other models
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            encode_kwargs={"normalize_embeddings": True}
        )
    

### LocalGPT
DEFAULT_SYSTEM_PROMPT = """\
Du bist ein hilfreicher, sachlicher und präziser deutscher Assistent. \
Nutze den bereitgestellten Kontext, um Fragen möglichst vollständig und verständlich zu beantworten. \
Wenn du keine Antwort weißt, gib dies ehrlich zu.
"""

def get_prompt_template(system_prompt=DEFAULT_SYSTEM_PROMPT, promptTemplate_type=None, history=False):
    prompt = None  # Initialize prompt to avoid unbound local variable error

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
        # Default prompt when no template type matches
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
    print(f"Here is the prompt used: {prompt}")  # For debugging, to see the generated prompt
    return prompt, memory


# In the load_model function, modify the quantization checks:
def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    quant_model_loaded = False

    if model_basename:
        model_basename_lower = model_basename.lower()

        if ".gguf" in model_basename_lower or ".ggml" in model_basename_lower:
            model_result = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            if model_result:
                if isinstance(model_result, tuple):
                    model, tokenizer = model_result
                else:
                    return model_result  # Some implementations return a pipeline directly
                quant_model_loaded = True
            else:
                LOGGING.warning(f"{model_basename} model loading failed, falling back to full model.")

        elif ".awq" in model_basename_lower:
            LOGGING.warning("AWQ models not supported. Falling back to full model.")
        elif ".gptq" in model_basename_lower:
            LOGGING.warning("GPTQ models not supported. Falling back to full model.")
        else:
            LOGGING.warning("Unknown quantization format. Falling back to full model.")

    if not quant_model_loaded:
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
    """Load HuggingFace embeddings on the appropriate device with logging."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading embeddings on device: {device}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        # Sanity check
        test_vector = embeddings.embed_query("Embedding test")
        logging.info(f"Loaded embeddings. Dimension: {len(test_vector)}")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to load embeddings: {e}")
        raise RuntimeError("Embedding model failed to initialize.")


def calculate_similarity(model, response, expected_answer):
    """Cosine similarity between response and expected answer embeddings."""
    try:
        response_embedding = model.client.encode(response, convert_to_tensor=True).squeeze()
        expected_embedding = model.client.encode(expected_answer, convert_to_tensor=True).squeeze()
        similarity_score = torch.nn.functional.cosine_similarity(response_embedding, expected_embedding, dim=0)
        return similarity_score.item()
    except Exception as e:
        logging.error(f"Error calculating similarity: {e}")
        return -1.0  # Return a clearly invalid similarity score


def retrieval_qa_pipline(device_type, use_history, persist_directory, promptTemplate_type="llama"):
    """Set up the retrieval QA pipeline with embedded vector store."""
    try:
        embeddings = load_embeddings()

        # Verify embeddings
        test_embedding = embeddings.embed_query("Test embedding")
        logging.info(f"Embedding dimension: {len(test_embedding)}")

        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS
        )

        prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

        llm = load_model(
            device_type=device_type,
            model_id=MODEL_ID,
            model_basename=MODEL_BASENAME,
            LOGGING=logging
        )

        retriever = db.as_retriever(search_kwargs={"k": 3})
        chain_kwargs = {"prompt": prompt}
        if use_history:
            chain_kwargs["memory"] = memory

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            callbacks=callback_manager,
            chain_type_kwargs=chain_kwargs
        )

        logging.info("Retrieval QA pipeline successfully initialized.")
        return qa

    except Exception as e:
        logging.error(f"Failed to create QA pipeline: {e}")
        raise



### AGENT
class Agent:
    def __init__(self, name, embeddings_dir, device_type, use_history, model_type, persist_dir, promptTemplate_type, agent_state):
        self.name = name  
        self.embeddings_dir = embeddings_dir  
        self.persist_dir = persist_dir
        self.promptTemplate_type = promptTemplate_type
        self.agent_state = agent_state

        try:
            self.qa = retrieval_qa_pipline(
                device_type=device_type,
                use_history=use_history,
                persist_directory=self.persist_dir,
                promptTemplate_type=model_type
            )
        except Exception as e:
            logging.error(f"Agent {self.name} failed to initialize QA pipeline: {e}")
            raise

    def ask(self, query):
        try:
            res = self.qa({"query": query})
            logging.debug(f"Agent {self.name} retrieved {len(res.get('source_documents', []))} sources")
            return res
        except Exception as e:
            logging.error(f"Agent {self.name} failed to process query: {e}")
            return {"result": "Fehler bei der Verarbeitung.", "source_documents": []}


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


def create_agent(name, embeddings_dir, device_type, use_history, model_type, persist_dir, promptTemplate_type, agent_state):
    try:
        return Agent(
            name=name,
            embeddings_dir=embeddings_dir,
            device_type=device_type,
            use_history=use_history,
            model_type=model_type,
            persist_dir=persist_dir,
            promptTemplate_type=promptTemplate_type,
            agent_state=agent_state
        )
    except Exception as e:
        logging.error(f"Failed to create agent {name}: {e}")
        raise

def run_single_round(agent_speaker, agent_listener, speaker_state, listener_state, round_num, total_rounds, save_qa, simulation_round):
    context_prompt = prompt(speaker_state, round_num, total_rounds)
    logging.info(f"{speaker_state['name']} speaking in round {round_num}...")
    
    response = agent_speaker.ask(context_prompt)
    result_text = response.get("result", "")
    speaker_state['dispute_history'].append([context_prompt, result_text])

    logging.debug(f"{speaker_state['name']} said: {result_text}")
    
    if save_qa:
        log_to_csv(f"Simulation {simulation_round}, Round {round_num} Prompt", result_text)
    
    return result_text  # To be passed as context to the other agent

def run_simulation(simulation_round, rounds, device_type, model_type, use_history, save_qa):
    logging.info(f"\n=== Starting Simulation Round {simulation_round} ===")

    # Reset state per simulation
    state_x = copy.deepcopy(agent_state_x)
    state_y = copy.deepcopy(agent_state_y)
    context = ""

    # Create both agents
    agent_x = create_agent("X", EMBEDDINGS_DIRECTORY_X, device_type, use_history, model_type, PERSIST_DIRECTORY_X, model_type, state_x)
    agent_y = create_agent("Y", EMBEDDINGS_DIRECTORY_Y, device_type, use_history, model_type, PERSIST_DIRECTORY_Y, model_type, state_y)

    for round_num in range(1, rounds + 1):
        logging.info(f"--- Round {round_num} ---")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Agent X speaks
        response_x = run_single_round(agent_x, agent_y, state_x, state_y, round_num, rounds, save_qa, simulation_round)
        context = f"\nDie neuste Aussage von {state_x['name']}: {response_x}"

        # Agent Y responds
        response_y = run_single_round(agent_y, agent_x, state_y, state_x, round_num, rounds, save_qa, simulation_round)
        context = f"\nDie neuste Aussage von {state_y['name']}: {response_y}"

    # Save result
    save_dispute_history_to_json(state_x, state_y, simulation_round)

    # Clean up
    del agent_x
    del agent_y
    torch.cuda.empty_cache()
    gc.collect()

@click.command()
@click.option("--show_sources", "-s", is_flag=True, help="Show source docs (default: False)")
@click.option("--use_history", "-h", is_flag=True, help="Use history between rounds")
@click.option(
    "--model_type",
    default="llama3",
    type=click.Choice(["llama3", "llama", "mistral", "non_llama"]),
    help="Model type",
)
@click.option("--save_qa", is_flag=True, help="Save Q&A to CSV (default: False)")
@click.option("--rounds", "-r", default=3, type=int, help="Number of rounds per simulation")
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(["cpu", "cuda", "mps", "xpu", "ort", "meta"]),
    help="Execution device",
)
@click.option(
    "--number_of_simulations", "-n",
    default=1,
    type=int,
    help="Number of full dispute simulations to run"
)
def main(device_type, show_sources, use_history, model_type, save_qa, rounds, number_of_simulations):
    logging.info(f"Running with model_type: {model_type} on device: {device_type}")
    logging.info(f"Use history: {use_history} | Save Q&A: {save_qa} | Rounds: {rounds} | Simulations: {number_of_simulations}")

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    with torch.no_grad():
        for sim_round in range(1, number_of_simulations + 1):
            run_simulation(sim_round, rounds, device_type, model_type, use_history, save_qa)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
