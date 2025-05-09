# python main.py -s -qa -h 
# python main.py -s -qa -r 5 -n 10 -h
# git clone --branch crazy git@github.com:gelpkmar/dispute-machine.git
# history -c && history -w

import os, torch, csv, click, logging, gc 
from datetime import datetime
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.vectorstores import Chroma
from transformers import (GenerationConfig, pipeline,)
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from constants import MODELS_PATH, MODEL_ID, MODEL_BASENAME, PERSIST_DIRECTORY_X, PERSIST_DIRECTORY_Y, CONTEXT_WINDOW_SIZE, CHROMA_SETTINGS, CALLBACK_MANAGER, N_GPU_LAYERS, MAX_NEW_TOKENS, N_BATCH, EMBEDDING_MODEL_NAME
# print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
# print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
# print(f"Current device: {torch.cuda.current_device()}")

def load_quantized_model_gguf_ggml(model_id, model_basename, device_type, logging):
    """
    Load a GGUF/GGML quantized model using LlamaCpp with GPU support.
    """
    try:
        logging.info("⚠️ Using LlamaCpp for GGUF/GGML quantized models")
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=model_basename,
            cache_dir=MODELS_PATH,
        )

        kwargs = {
            "model_path": model_path,
            "n_ctx": CONTEXT_WINDOW_SIZE,
            "max_tokens": MAX_NEW_TOKENS,
            "n_batch": N_BATCH,
            "n_threads": os.cpu_count() or 8,  # use all CPU threads
            "f16_kv": True,  # Must be True for GPU
            "verbose": False,
        }

        # GPU settings
        if device_type.lower() == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("❌ CUDA is not available. Check NVIDIA drivers.")
            logging.info("⚡ Using CUDA for inference")
            kwargs["n_gpu_layers"] = N_GPU_LAYERS  # e.g., 35 for full model on GPU
            # kwargs.update({
            #     "n_gpu_layers": -1,  # Offload all layers to GPU
            #     "main_gpu": 0,       # Use primary GPU
            # })

        elif device_type.lower() == "mps":
            logging.info("🍎 Using Apple MPS for inference")
            kwargs["n_gpu_layers"] = 1  # MPS only supports 1 layer

        else:
            logging.info("🧠 Using CPU for inference")
            kwargs["n_gpu_layers"] = 0

        return LlamaCpp(**kwargs)

    except TypeError as e:
        if "ggml" in model_basename:
            logging.info("⚠️ GGML models are deprecated. Use GGUF instead.")
        logging.error(f"Model loading failed: {e}")
        return None


def load_full_model(model_id, model_basename, device_type, logging):
    """
    Load a full model using either LlamaTokenizer or AutoModelForCausalLM.
    """
    if device_type.lower() in ["mps", "cpu"]:
        logging.info("⚠️ Using AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="./models/"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models/")
    else:
        logging.info("⚠️ Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models/")
        logging.info("⚠️ Tokenizer loaded")
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

def get_embeddings(device_type="cuda"):
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
    # print(f"📚 Here is the prompt used: {prompt}")  # For debugging, to see the generated prompt
    return prompt, memory


# In the load_model function, modify the quantization checks:
def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    logging.info(f"⚠️ Loading Model: {model_id}, on: {device_type}")
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
                LOGGING.warning(f"❌ {model_basename} model loading failed, falling back to full model.")

        elif ".awq" in model_basename_lower:
            LOGGING.warning("AWQ models not supported. Falling back to full model.")
        elif ".gptq" in model_basename_lower:
            LOGGING.warning("❌ GPTQ models not supported. Falling back to full model.")
        else:
            LOGGING.warning("❌ Unknown quantization format. Falling back to full model.")

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
    logging.info(f"⚠️ Loading embeddings on device: {device}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        return embeddings
    except Exception as e:
        logging.error(f"❌ Failed to load embeddings: {e}")
        raise RuntimeError("❌ Embedding model failed to initialize.")


def calculate_similarity(model, response, expected_answer):
    """Cosine similarity between response and expected answer embeddings."""
    try:
        response_embedding = model.client.encode(response, convert_to_tensor=True).squeeze()
        expected_embedding = model.client.encode(expected_answer, convert_to_tensor=True).squeeze()
        similarity_score = torch.nn.functional.cosine_similarity(response_embedding, expected_embedding, dim=0)
        return similarity_score.item()
    except Exception as e:
        logging.error(f"❌ Error calculating similarity: {e}")
        return -1.0  # Return a clearly invalid similarity score


def retrieval_qa_pipline(device_type, use_history, persist_directory, promptTemplate_type):

    embeddings = load_embeddings()

    # load the vectorstore
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # load the llm pipeline
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            callbacks=CALLBACK_MANAGER,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            callbacks=CALLBACK_MANAGER,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa


### AGENT
class Agent:
    def __init__(self, name, device_type, use_history, model_type, persist_dir, promptTemplate_type, agent_state):
        self.name = name  
        self.persist_dir = persist_dir
        self.promptTemplate_type = promptTemplate_type
        self.agent_state = agent_state
        self.qa = retrieval_qa_pipline(device_type, use_history, self.persist_dir, promptTemplate_type=model_type)

    def ask(self, query):
        res = self.qa.invoke({"query": query})
        return res

def prompt(agent_state: dict, round_number: int, total_rounds: int) -> str:
    if round_number == total_rounds-1:
        return f"""
        Du bist {agent_state['name']}, {agent_state['age']} Jahre alt und {agent_state['gender']}. Du befindest dich aktuell in einem Streit mit {agent_state['dispute_partner']}.

        Aktuelle Situation:
        - Deine Rolle: {agent_state['dispute_context']['context']}
        - Rechtliches Thema: {agent_state['legal_issue']}
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
        - Rechtliches Thema: {agent_state['legal_issue']}
        - Streitkontext: {agent_state['dispute_context']['facts']}
        - Bevorzugte Lösung: {agent_state['dispute_context']['preferred_resolution']}
        - Streitverlauf: {agent_state['dispute_history'][-1]}

        Basierend auf den obigen Informationen, sollte deine Antwort die folgenden Punkte behandeln:
        1. Drücke Frustration über die aktuelle Situation aus.
        2. Betone deine finanziellen Verluste und unterstreiche die Ernsthaftigkeit des Problems.
        3. Fordere einen Kompromiss und/oder eine Vertragsänderung.
        Antworte immer auf die untenstehende Aussage von {agent_state['dispute_partner']} unter Berücksichtigung der oben stehenden Daten (inkl. Streitverlauf).
        {agent_state['characteristics']}. 
        """

# Example of updating agent state after each round
## Case 1.1 (Furniture):
# agent_state_x = {
#     'name': 'Bob',
#     'age': 33,
#     'gender': 'Männlich',
#     'legal_issue': 'Preisnachlass wegen verspäteter Lieferung',
#     'characteristics': 'Deine finanzielle Lage ist stabil, und du hast ein klares Ziel, finanzielle Entschädigung für den Vertragsbruch zu für den Käufer zu minimieren. Deine Art ist emotional und kooperativ, aber aggressiv, wenn du deine Position verteidigst.',
#     'dispute_context': {
#         'context': 'Du bist der Verkäufer von Möbeln und hast vereinbarte Waren zu spät geliefert. Nun befindest du dich in einem Streit um Wiedergutmachung, der kurz davor ist, vor Gericht zu gehen und möchtest die entstandenen Schäden so klein wie möglich halten.', 
#         'facts': 'Streit über Servicequalität',
#         'preferred_resolution': 'Akzeptanz des Preisnachlasses von maximal 10%.'
#     },
#     'dispute_partner': 'Alice',
#     'dispute_history': [[]]
# }

# agent_state_y = {
#     'name': 'Alice',
#     'age': 64,
#     'gender': 'Weiblich',
#     'legal_issue': 'Preisnachlass wegen verspäteter Lieferung',
#     'characteristics': 'Deine finanzielle Lage ist stabil, und du hast ein klares Ziel, finanzielle Entschädigung für den Vertragsbruch zu erhalten. Deine Art ist emotional und kooperativ, aber aggressiv, wenn du deine Position verteidigst.',
#     'dispute_context': {
#         'context': 'Du bist der Käufer von Möbeln und hast vereinbarte Waren zu spät geliefert bekommen. Nun befindest du dich in einem Streit um Wiedergutmachung, der kurz davor ist, vor Gericht zu gehen und möchtest einen Preisnachlass erreichen.',
#         'facts': 'Streit über Servicequalität',
#         'preferred_resolution': 'Preisnachlass von 25% oder kostenlose Lieferung zusätzlicher Artikel wie Stehlampen.'
#     },
#     'dispute_partner': 'Bob',
#     'dispute_history': [[]]
# }

## Case 3.4 (Oldtimer):
agent_state_x = {
    'name': 'Bob',
    'age': 33,
    'gender': 'Männlich',
    'legal_issue': 'Preisnachlass oder Rückabwicklung Kauf wegen Lieferung eines Oldtimers mit Schäden.',
    'characteristics': 'Deine finanzielle Lage ist stabil, und du hast ein klares Ziel, den Schaden durch die eingegangene Reklamation zu minimieren. Deine Art ist emotional und kooperativ, aber aggressiv, wenn du deine Position verteidigst.',
    'dispute_context': {
        'context': 'Du bist der Verkäufer von Oldtimern und hast angeblich einen Wagen mit massiven Rostschäden am Rahmen rechts verkauft.', 
        'facts': 'Streit über Servicequalität',
        'preferred_resolution': 'Kein Preisnachlass oder Rückabwicklung des Kaufs da der verkaufte Wagen probegefahren wurde.'
    },
    'dispute_partner': 'Alice',
    'dispute_history': [[]]
}

agent_state_y = {
    'name': 'Alice',
    'age': 64,
    'gender': 'Weiblich',
    'legal_issue': 'Preisnachlass wegen wegen Lieferung eines Oldtimers mit Schäden.',
    'characteristics': 'Deine finanzielle Lage ist stabil, und du hast ein klares Ziel, finanzielle Entschädigung oder eine Rückabwicklung des Kaufs für den fehlerhaften Oldtimer zu erhalten. Deine Art ist emotional und kooperativ, aber aggressiv, wenn du deine Position verteidigst.',
    'dispute_context': {
        'context': 'Du bist der Käufer von einem Oltimer und hast nach Übernahme des Fahrzeugs massive Rostschäden festgestellt. Nun befindest du dich in einem Streit um Wiedergutmachung, der kurz davor ist, vor Gericht zu gehen und möchtest einen Preisnachlass oder eine Rückabwicklung des Kaufs erreichen.',
        'facts': 'Streit über Servicequalität',
        'preferred_resolution': 'Massiver Preisnachlass oder eine Rückabwicklung des Kaufs.'
    },
    'dispute_partner': 'Bob',
    'dispute_history': [[]]
}


@click.command()
@click.option(
    "--show_sources",
    "-s",
    default=False, 
    is_flag=True,
    help="🔧 Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    default=False,
    is_flag=True,
    help="🔧 Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="llama3",
    type=click.Choice(
        ["llama3", "llama", "mistral", "non_llama"],
    ),
    help="🔧 model type, llama3, llama, mistral or non_llama",
)
@click.option(
    "--save_qa",
    "-qa",
    default=False,  
    is_flag=True,
    help="🔧 whether to save Q&A pairs to a CSV file (Default is False)",
)
@click.option(
    "--rounds", 
    "-r", 
    default=3,  # Default number of rounds
    type=int,   # Ensure it expects an integer value
    help="🔧 Number of rounds for the discussion (Default is 3)"
)
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(["cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip", "ve", "fpga", "ort", "xla", "lazy", "vulkan", "mps", "meta", "hpu", "mtia"]),
    help="🔧 Device to run on. (Default is cuda)",
)
@click.option(
    "--number_of_simulations",
    "-n", 
    default=1,  # Default number of rounds
    type=int,   # Ensure it expects an integer value
    help="🔧 Number of simulations (Default is 1)"
)
def main(device_type, show_sources, use_history, model_type, save_qa, rounds, number_of_simulations):
    for simulation_round in range(1, number_of_simulations + 1):
        # logging.info(f"Running on: {device_type}")
        # logging.info(f"Display Source Documents set to: {show_sources}")
        # logging.info(f"Use history set to: {use_history}")

        if not os.path.exists(MODELS_PATH):
            os.mkdir(MODELS_PATH)

        # ⚡ Use no_grad to disable gradient tracking and save memory
        with torch.no_grad():

            agent_X = Agent(
                name="Bob",
                persist_dir=PERSIST_DIRECTORY_X,
                device_type=device_type,
                use_history=use_history,
                model_type=model_type,
                promptTemplate_type=model_type,
                agent_state=agent_state_x
            )

            agent_Y = Agent(
                name="Alice",
                persist_dir=PERSIST_DIRECTORY_Y,
                device_type=device_type,
                use_history=use_history,
                model_type=model_type,
                promptTemplate_type=model_type,
                agent_state=agent_state_y
            )

            current_context = ""            

            # ✅ Start the simulation rounds
            for round_num in range(1, rounds + 1): 
                logging.info(f"🔁 Simulation {simulation_round}, Round {round_num}:")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Agent X speaks
                agent_X_response = agent_X.ask(prompt(agent_state_x, round_num, rounds) + current_context)
                answer_X, docs = agent_X_response["result"], agent_X_response["source_documents"]
                logging.info(f"📚 Die neuste Aussage von {agent_state_x['name']}: {agent_X_response}")

                if save_qa:
                    log_to_csv(f'Simulation {simulation_round}, Round{round_num}: {current_context}', answer_X)
                    if show_sources:
                        log_to_csv(f'Simulation {simulation_round}, Round{round_num}: {current_context} \n\nDocuments:{docs}', answer_X)

                current_context = f"Die neuste Aussage von {agent_state_x['name']}: {answer_X}"
                agent_state_x['dispute_history'].append([current_context, answer_X])

                # Agent Y speaks
                agent_Y_response = agent_Y.ask(prompt(agent_state_y, round_num, rounds) + current_context)
                answer_Y, docs = agent_Y_response["result"], agent_Y_response["source_documents"]
                logging.info(f"📚 Die neuste Aussage von {agent_state_y['name']}: {agent_Y_response}")

                if save_qa:
                    log_to_csv(f'Simulation {simulation_round}, Round{round_num}: {current_context}', answer_Y)
                    if show_sources:
                        log_to_csv(f'Simulation {simulation_round}, Round{round_num}: {current_context} \n\nDocuments:{docs}', answer_Y)

                current_context = f"Die neuste Aussage von {agent_state_y['name']}: {answer_Y}"
                agent_state_y['dispute_history'].append([current_context, answer_Y])

            #📚 After each simulation round: clean up
            del agent_X
            del agent_Y
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO)
    main()
