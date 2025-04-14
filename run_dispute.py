import ingest, run_localGPT, utils
from agent import Agent

import os
import logging
import click
import torch
import utils
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager

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
    default=5,  # Default number of rounds
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
        device_type="cpu",
        use_history=False, 
        model_type=model_type, 
        persist_dir="../data/persist_X", 
        promptTemplate_type=model_type, 
        opening_statement="""
            You X (client) commissioned Y (supplier) to create a website for your company. 
            After Y has completed the work, X is dissatisfied with the result and refuses 
            to pay the full amount as he feels the design is not as agreed. 
            However, Y claims that the agreed requirements in the description were met.
            You are participating in a tit-for-tat legal dispute.
            The settlement proposal could include a partial amount of the agreed payment or the improvement of certain parts of the website without the full amount being reclaimed.
            Important documents are: Contract or terms of reference, email correspondence about requirements and changes, screenshots of the website before and after the changes.
            It's your choice whether to be cooperative with Y to find an agreement or not.
            Use all the documents and evidence available to you to make your case. 
            Answer to the following statement (either with a question or a demand, highlight whether you are being cooperative or defecting): """
    )
    agent_Y = Agent(
        name="Y", 
        embeddings_dir="../data/embeddings_Y",
        device_type="cpu",
        use_history=False, 
        model_type=model_type, 
        persist_dir="../data/persist_Y", 
        promptTemplate_type=model_type, 
        opening_statement="""
            You Y (supplier) created a website as commissioned by X (client) for his company. 
            After you successfully completed the work, X is dissatisfied with the result and refuses 
            to pay the full amount as he feels the design is not as agreed. 
            However, you claim that the agreed requirements in the description were met.
            You are participating in a tit-for-tat legal dispute. 
            The settlement proposal could include a partial amount of the agreed payment or the improvement of certain parts of the website without the full amount being reclaimed.
            Important documents are: Contract or terms of reference, email correspondence about requirements and changes, screenshots of the website before and after the changes.
            It's your choice whether to be cooperative with X to find an agreement or not.
            Use all the documents and evidence available to you to make your case.
            Answer to the following statement (either with a question or a demand, highlight whether you are being cooperative or defecting): """
    )
    """
    Start a discussion between two agents.
    """

    # context_file = SCRIPT_PATH
    # with open(context_file, 'r') as f:
    #     context = f.read()

    # logging.info(f"\nStarting discussion with context: {context}\n")

    current_context = "Please make an opening statement about your demands from Y."
    for round_num in range(1, rounds + 1):
        logging.info(f"Round {round_num}:")

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
