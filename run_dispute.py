# python run_dispute.py --save_qa --rounds 3 --show_sources
# history -c && history -w
import gc  # Garbage Collector interface
import ingest, run_localGPT, utils
from agent import Agent

import os
import logging
import click
import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
from utils import save_dispute_history_to_json
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader, TextLoader

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template
from utils import get_embeddings

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)

from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,    
)

SCRIPT_PATH = "/Users/thealteredmg/kDrive_altered/Studium_UZH/c_CURRENT/AIL_25/deliverable/dispute-machine/data/legal_script_txt.txt"

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
        'context': 'Du bist der Verkäufer von Möbeln und hast vereinbarte Waren zu spät geliefert. Nun befindest du dich in einem Streit um Wiedergutmachung, der kurz davor ist, vor Gericht zu gehen und möchtest die entstandenen Schäden so klein wie möglich halten..', 
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


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
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
    default=3,  # Default number of rounds
    type=int,   # Ensure it expects an integer value
    help="Number of rounds for the discussion"
)


def main(device_type, show_sources, use_history, model_type, save_qa, rounds):
    for simulation_round in range(1, 10):
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
                embeddings_dir="../data/embeddings_X",
                device_type="cuda",
                use_history=False, 
                model_type=model_type, 
                persist_dir="../data/persist_X", 
                promptTemplate_type=model_type, 
                agent_state=agent_state_x
            )
            agent_Y = Agent(
                name="Y", 
                embeddings_dir="../data/embeddings_Y",
                device_type="cuda",
                use_history=False, 
                model_type=model_type, 
                persist_dir="../data/persist_Y", 
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
                    utils.log_to_csv(f'Simulation {simulation_round}, Round{round_num}: {current_context}', answer_X)

                current_context = f"\nDie neuste Aussage von {agent_state_x['name']}: {answer_X}"
                agent_state_x['dispute_history'].append([current_context, answer_X])

                # Agent Y speaks
                agent_Y_response = agent_Y.ask(prompt(agent_state_y, round_num, rounds) + current_context)
                answer_Y, docs = agent_Y_response["result"], agent_Y_response["source_documents"]
                logging.info(f"Die neuste Aussage von {agent_state_y['name']}: {agent_Y_response}")

                if save_qa:
                    utils.log_to_csv(f'Simulation {simulation_round}, Round{round_num}: {current_context}', answer_Y)

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
