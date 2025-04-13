import os
import argparse
from localGPT import LocalGPT
from localGPT import fine_tune_model  # Assuming you have a fine-tune function or script

def fine_tune_agents(agent_1_model, agent_2_model, agent_1_docs, agent_2_docs):
    """
    Fine-tune two agents' models using their respective document folders.
    """
    print(f"Fine-tuning agent 1 with documents from {agent_1_docs}...")
    fine_tune_model(agent_1_model, agent_1_docs)
    print(f"Fine-tuning agent 2 with documents from {agent_2_docs}...")
    fine_tune_model(agent_2_model, agent_2_docs)
    print("Fine-tuning completed for both agents.")

def start_discussion(agent_1, agent_2, context, rounds):
    """
    Start a discussion between two agents using the provided context.
    """
    print(f"\nStarting discussion with context: {context}\n")
    
    current_context = context
    for round_num in range(1, rounds + 1):
        print(f"Round {round_num}:")
        
        # Agent 1 speaks
        print(f"Agent 1: {agent_1.ask(current_context)}")
        current_context += f"\nAgent 1: {agent_1.ask(current_context)}"
        
        # Agent 2 speaks
        print(f"Agent 2: {agent_2.ask(current_context)}")
        current_context += f"\nAgent 2: {agent_2.ask(current_context)}"

def main():
    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Setup localGPT environment and simulate agent discussion.")
    
    # Add arguments for directories and number of rounds
    parser.add_argument("--agent_1_folder", type=str, required=True, help="Folder containing documents for agent 1")
    parser.add_argument("--agent_2_folder", type=str, required=True, help="Folder containing documents for agent 2")
    parser.add_argument("--context_file", type=str, required=True, help="Context file to start the discussion")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds for the discussion")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model for both agents")
    
    args = parser.parse_args()

    # Load the models (both agents will start with the same model)
    print("Loading models...")
    agent_1 = LocalGPT(model_path=args.model_path)
    agent_2 = LocalGPT(model_path=args.model_path)
    
    # Fine-tune the models using the documents provided in the agent folders
    fine_tune_agents(agent_1, agent_2, args.agent_1_folder, args.agent_2_folder)
    
    # Load the context document
    with open(args.context_file, 'r') as f:
        context = f.read()

    # Start the discussion between agents
    start_discussion(agent_1, agent_2, context, args.rounds)

if __name__ == "__main__":
    main()
