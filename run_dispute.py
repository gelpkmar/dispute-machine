# python run_dispute.py --save_qa --rounds 3
# history -c && history -w
import ingest, run_localGPT, utils
from agent import Agent

import os
import logging
import click
import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
import utils
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

def x_prompt(agent_state: dict, round_number: int, total_rounds: int) -> str:
    """
    Generates the prompt for X based on their current state in the dispute.
    
    Args:
    - agent_state (dict): Contains the current context like goals, emotions, dispute progress, etc.
    - y_last_offer (str): The last offer made by Y.
    - round_number (int): The current round of the dispute.
    - total_rounds (int): The total number of rounds in the dispute.

    Returns:
    - str: The prompt for X's next response.
    """
    if round_number == total_rounds-1:
        return f"""
        Du bist X, ein {agent_state['age']} Jahre alter {agent_state['gender']} Verkäufer aus Italien. 

        Dies ist die letzte Runde deines Streits mit Y. Du hast bisher nach einer Akzeptanz deines Preisnachlasses von 10% gesucht. 

        Aktuelle Situation: Du forderst Akzeptanz deines Preisnachlasses gesucht aufgrund der verspäteten Lieferung und deren Auswirkungen auf Y's Geschäft.
        
        Reflektiere über den gesamten Streit und beantworte folgende Fragen:
        1. Hat Y’s Angebot die finanziellen Verluste und Auswirkungen auf dein Geschäft ausreichend berücksichtigt?
        2. Bist du bereit, das Angebot anzunehmen oder nicht?
        3. Falls du weiterhin unzufrieden bist, erkläre warum und ob du den Streit vor Gericht bringen möchtest.

        Bitte triff eine endgültige Entscheidung, entweder das Angebot anzunehmen oder abzulehnen und den Streit vor Gericht zu bringen.
        """
    else:
        return f"""
        Du bist X, ein {agent_state['age']} Jahre alter {agent_state['gender']} Verkäufer aus Italien. Du befindest dich aktuell in einem Streit wegen eines von dir verschuldeten Vertragsbruchs mit deinem Geschäftspartner. Der Vertrag spezifizierte die Lieferung bestimmter Waren, aber die Lieferung von dir war erheblich verspätet, was zu erheblichen Störungen für Y's Unternehmen geführt hat.

        Deine finanzielle Lage ist stabil, und du hast ein klares Ziel, finanzielle Entschädigung für den Vertragsbruch zu erhalten. Deine Art ist emotional und kooperativ, aber aggressiv, wenn du deine Position verteidigst.

        Aktuelle Situation:
        - Rechtliches Thema: {agent_state['legal_issue_involved']}
        - Streitkontext: {agent_state['dispute_context']['facts']}
        - Bevorzugte Lösung: {agent_state['dispute_context']['preferred_resolution']}
        
        In deiner letzten Auseinandersetzung hat dir dein Geschäftspartner ein Gegenangebot von 25% Preisnachlass gemacht. Du möchtest deine Unzufriedenheit ausdrücken und für eine bessere Lösung verhandeln. Du überlegst, folgende Argumente anzubringen:
        - Die verspätete Lieferung war auf deine Nachlässigkeit zurückzuführen.
        - Die Störungen haben zu erheblichen finanziellen Verlusten geführt.
        - Du forderst eine Minderung des angefragten Preisnachlasses auf 10% und eine Vertragsänderung, um zukünftige Probleme zu vermeiden.
        
        Basierend auf den obigen Informationen, sollte deine Antwort die folgenden Punkte behandeln:
        1. Drücke Frustration über die verspätete Lieferung aus.
        2. Betone deine finanziellen Verluste und unterstreiche die Ernsthaftigkeit des Problems.
        3. Fordere einen Kompromiss und/oder eine Vertragsänderung.
        """

def y_prompt(agent_state: dict, round_number: int, total_rounds: int) -> str:
    """
    Generates the prompt for Y based on their current state in the dispute.
    
    Args:
    - agent_state (dict): Contains current context like goals, emotions, dispute progress, etc.
    - x_last_offer (str): The last offer made by X.
    - round_number (int): The current round of the dispute.
    - total_rounds (int): The total number of rounds in the dispute.

    Returns:
    - str: The prompt for Y's next response.
    """
    if round_number == total_rounds-1:
        return f"""
        Du bist Y, eine {agent_state['age']} Jahre alte {agent_state['gender']} Kunde aus der Schweiz.

        Dies ist die letzte Runde deines Streits mit X. Du hast bisher nach einer vollständigen finanziellen Entschädigung für den schlechten Services gesucht.

        Aktuelle Situation: Du forderst mindestens einen 25%-Rabatt oder zusätzliche Entschädigung für den Schaden an deinem Ruf.

        Reflektiere über den gesamten Streit und beantworte folgende Fragen:
        1. Hat X’s Angebot den Schaden an deinem Ruf und die finanziellen Belastungen ausreichend berücksichtigt?
        2. Bist du bereit, das Angebot anzunehmen, oder fühlst du, dass es nicht dem Ausmaß des Schadens entspricht?
        3. Falls du weiterhin unzufrieden bist, erkläre warum und ob du den Streit vor Gericht bringen möchtest.

        Bitte triff eine endgültige Entscheidung, entweder das Angebot anzunehmen oder abzulehnen und den Streit vor Gericht zu bringen.
        """
    else:
        return f"""
        Du bist Y, eine {agent_state['age']} Jahre alte {agent_state['gender']} Kunde aus der Schweiz. Du befindest dich in einem Streit über die Qualität des Services, den der Verkäufer X erbracht hat. Du strebst eine vollständige monetäre Entschädigung an.

        Deine finanzielle Lage umfasst erhebliche Immobilienbestände, aber du bist hoch verschuldet. Du bist streitsüchtig und stur, aber bevorzugst eine lösungsorientierte Einigung.

        Aktuelle Situation:
        - Rechtliches Thema: {agent_state['legal_issue_involved']}
        - Streitkontext: {agent_state['dispute_context']['facts']}
        - Bevorzugte Lösung: {agent_state['dispute_context']['preferred_resolution']}
        
        In deiner letzten Auseinandersetzung wurde dir ein Preisnachlass von 10% als Entschädigung angeboten. Du findest dies unzureichend aufgrund des erheblichen Schadens an deinem Ruf. Du überlegst, folgende Argumente anzubringen:
        - Die erbrachte Servicequalität lag deutlich unter dem vereinbarten Standard.
        - Der Einfluss auf deinen Ruf war erheblich und muss sich in der Entschädigung widerspiegeln.
        - Du forderst mindestens einen 25%-Rabatt oder eine kostenlose Lieferung zusätzlicher Artikel wie Stehlampen.
        
        Basierend auf den obigen Informationen, sollte deine Antwort die folgenden Punkte behandeln:
        1. Drücke deine Enttäuschung über das angebotene Entgegenkommen aus.
        2. Betone den Einfluss auf deinen Ruf und die Notwendigkeit einer höheren Entschädigung.
        """

# Example of updating agent state after each round
agent_state_x = {
    'age': 33,
    'gender': 'Männlich',
    'legal_issue_involved': 'Personenschadenanspruch',
    'dispute_context': {
        'facts': 'Streit über Servicequalität',
        'preferred_resolution': 'Akzeptanz des Preisnachlasses von maximal 10% Vertragsänderung'
    }
}

agent_state_y = {
    'age': 64,
    'gender': 'Weiblich',
    'legal_issue_involved': 'Personenschadenanspruch',
    'dispute_context': {
        'facts': 'Streit über Servicequalität',
        'preferred_resolution': 'PReisnachlass von 25% oder kostenlose Lieferung zusätzlicher Artikel wie Stehlampen'
    }
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
    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    # qa = run_localGPT.retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)
    agent_X = Agent(
        name="X", 
        embeddings_dir="../data/embeddings_X",
        device_type="cuda",
        use_history=False, 
        model_type=model_type, 
        persist_dir="../data/persist_X", 
        promptTemplate_type=model_type, 
        opening_statement=x_prompt(agent_state_x, round_number=1, total_rounds=rounds)
    )
    agent_Y = Agent(
        name="Y", 
        embeddings_dir="../data/embeddings_Y",
        device_type="cuda",
        use_history=False, 
        model_type=model_type, 
        persist_dir="../data/persist_Y", 
        promptTemplate_type=model_type, 
        opening_statement=y_prompt(agent_state_y, round_number=1, total_rounds=rounds)
    )
    
    """
    Start a discussion between two agents.
    """

    current_context = "Please make an opening statement about your demands from Y."
    for round_num in range(1, rounds + 1): 

        logging.info(f"Round {round_num}:")

        # Clear GPU cache before each round
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Agent X speaks
        agent_X_response = agent_X.ask(current_context)
        answer_X, docs = agent_X_response["result"], agent_X_response["source_documents"]
        logging.info(f"Agent X: {agent_X_response}")
        current_context = f"\nAgent X: {answer_X}"

        # Agent 2 speaks
        agent_Y_response = agent_Y.ask(current_context)
        answer_Y, docs = agent_Y_response["result"], agent_Y_response["source_documents"]
        logging.info(f"Agent Y: {agent_Y_response}")
        current_context = f"\nAgent Y: {answer_Y}"

        # Log the Q&A to CSV only if save_qa is True
        if save_qa:
            utils.log_to_csv(current_context, answer_X)

    

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
