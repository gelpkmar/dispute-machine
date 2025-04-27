import os
import csv, json
from datetime import datetime
from constants import EMBEDDING_MODEL_NAME
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings


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

from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
import torch

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