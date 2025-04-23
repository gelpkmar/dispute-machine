import re
import os
from datetime import datetime

def process_raw_output(raw_text):
    """Process the raw dispute log and format it into rounds"""
    rounds = re.split(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - run_dispute\.py:173 - Round \d+:', raw_text)
    rounds = [round.strip() for round in rounds if round.strip()]
    
    output_lines = []
    
    for i, round_content in enumerate(rounds, start=1):
        agent_x_match = re.search(r'run_dispute\.py:182 - Agent X: (\{.*?\})', round_content, re.DOTALL)
        agent_y_match = re.search(r'run_dispute\.py:188 - Agent Y: (\{.*?\})', round_content, re.DOTALL)
        
        if not agent_x_match or not agent_y_match:
            continue
            
        agent_x = clean_json_output(agent_x_match.group(1))
        agent_y = clean_json_output(agent_y_match.group(1))
        
        output_lines.append(f"Round {i}:\n")
        output_lines.append(f"Agent X: {agent_x}\n")
        output_lines.append(f"Agent Y: {agent_y}\n")
        output_lines.append("\n")
    
    return "".join(output_lines)

def clean_json_output(json_str):
    """Clean and format the JSON output"""
    json_str = re.sub(r'\\n', '\n', json_str)  # Convert escaped newlines
    json_str = re.sub(r'\\"', '"', json_str)   # Handle escaped quotes
    json_str = json_str.replace('"{', '{').replace('}"', '}')  # Remove outer quotes if present
    return json_str.strip()

def save_simulation(output_content):
    """Save the formatted output to the Simulations folder"""
    # Create Simulations folder if it doesn't exist
    os.makedirs("Simulations", exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Simulations/dispute_simulation_{timestamp}.txt"
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(output_content)
    
    print(f"Simulation successfully saved to: {filename}")
    return filename

def main():
    # Read input from file
    try:
        with open('input.txt', 'r', encoding='utf-8') as file:
            raw_text = file.read()
    except FileNotFoundError:
        print("Error: input.txt file not found in the current directory")
        return
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return
    
    # Process and save the output
    formatted_output = process_raw_output(raw_text)
    save_simulation(formatted_output)

if __name__ == "__main__":
    main()