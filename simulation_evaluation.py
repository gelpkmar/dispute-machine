import json
import os
import sys
import logging
import re
from typing import List, Dict, Any, Optional
from llama_cpp import Llama
from pathlib import Path

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", 
    level=logging.INFO
)

# Configuration
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
MODEL_BASENAME = "Meta-Llama-3-8B-Instruct.Q8_0.gguf"
SIMULATIONS_PATH = os.path.join(ROOT_DIRECTORY, "Simulations")
OUTPUT_FILE = "evaluation_results.json"

# Enhanced evaluation prompt with strict JSON formatting instructions
EVALUATION_PROMPT = """You are a professional negotiation evaluator.

Evaluate this negotiation between Agent X and Agent Y on:

- debate_quality (0-10)
- professionalism (0-10)
- problem_solving (0-10)
- compliance (0-10)

STRUCTURE YOUR OUTPUT AS A SINGLE VALID JSON OBJECT (NO MARKDOWN, NO EXPLANATION), EXACTLY:

{"debate_quality": int, "professionalism": int, "problem_solving": int, "compliance": int, "overall_score": int, "feedback": "short feedback"}

Conversation:
{conversation}
"""




def load_model(model_path: str) -> Optional[Llama]:
    """Load the GGUF model with enhanced settings"""
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=8192,  # Increased context window
            n_threads=6,
            n_gpu_layers=35,
            verbose=False,  # Reduced verbosity
        )
        logging.info("Model loaded successfully")
        return llm
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        return None

def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extracts JSON object from model response"""
    try:
        # Clean up common formatting issues
        text = text.strip().replace("```json", "").replace("```", "").strip()

        # Attempt to extract the first valid {...} block
        match = re.search(r'\{(?:[^{}]|(?R))*\}', text)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.warning(f"JSON decode error: {e}")
    except Exception as e:
        logging.warning(f"Unexpected error extracting JSON: {e}")
    
    return None



def evaluate_conversation(llm: Llama, convo_x: dict, convo_y: dict) -> Dict[str, Any]:
    if not convo_x or not convo_y or 'result' not in convo_x or 'result' not in convo_y:
        return error_response("Invalid conversation format")

    combined = f"AGENT X STATEMENT:\n{convo_x['result']}\n\nAGENT Y RESPONSE:\n{convo_y['result']}"
    
    try:
        prompt = EVALUATION_PROMPT.format(conversation=combined)
        response = llm(prompt, max_tokens=512, temperature=0.3, stop=["\n\n"])
        response_text = response['choices'][0]['text'].strip()
        print("\n=== RAW MODEL RESPONSE ===\n")
        print(response_text)
        print("\n==========================\n")

        evaluation = extract_json_from_response(response_text)
        
        if not evaluation:
            return error_response(f"Invalid JSON response: {response_text[:100]}...")
        
        required_keys = {'debate_quality', 'professionalism', 'problem_solving', 'compliance', 'overall_score', 'feedback'}
        if not all(key in evaluation for key in required_keys):
            return error_response("Missing required evaluation fields")
        
        return {
            "score": evaluation['overall_score'],
            "label": score_to_label(evaluation['overall_score']),
            "feedback": evaluation['feedback'],
            "category_scores": {
                "debate_quality": evaluation['debate_quality'],
                "professionalism": evaluation['professionalism'],
                "problem_solving": evaluation['problem_solving'],
                "compliance": evaluation['compliance']
            }
        }
        
    except Exception as e:
        return error_response(f"Evaluation failed: {str(e)}")


def score_to_label(score: int) -> str:
    """Convert numeric score to qualitative label"""
    if score >= 9: return "Excellent"
    if score >= 7: return "Good"
    if score >= 5: return "Average"
    return "Needs Improvement"

def error_response(message: str) -> Dict[str, Any]:
    """Standardized error response"""
    return {
        "score": None,
        "label": "Error",
        "feedback": message,
        "error": True
    }

def extract_conversation_rounds(text: str) -> List[Dict[str, Any]]:
    """Improved parser for conversation files"""
    rounds = []
    current_round = None
    current_agent = None
    current_content = []
    
    for line in text.split('\n'):
        line = line.strip()
        
        # Detect round start
        if line.startswith('Round '):
            if current_round is not None:
                if current_agent and current_content:
                    current_round[current_agent] = {"result": "\n".join(current_content).strip()}
                rounds.append(current_round)
            current_round = {"AgentX": None, "AgentY": None}
            current_agent = None
            current_content = []
        
        # Detect agent start
        elif line.startswith('Agent X:') or line.startswith('Agent Y:'):
            if current_agent and current_content:
                current_round[current_agent] = {"result": "\n".join(current_content).strip()}
            
            current_agent = "AgentX" if "Agent X:" in line else "AgentY"
            current_content = []
            
            # Try to extract result if it's in the same line
            if "'result':" in line:
                try:
                    result_part = line.split("'result':")[1]
                    result = result_part.split("',")[0].strip(" '\"")
                    current_round[current_agent] = {"result": result}
                    current_agent = None  # Reset since we got the result
                except IndexError:
                    # If parsing fails, just continue collecting lines
                    pass
        
        # Collect content for current agent
        elif current_agent and line:
            current_content.append(line)
    
    # Add the last round if it exists
    if current_round is not None:
        if current_agent and current_content:
            current_round[current_agent] = {"result": "\n".join(current_content).strip()}
        rounds.append(current_round)
    
    return rounds

def process_simulation_files(llm: Llama) -> Dict[str, Any]:
    """Process all simulation files in the Simulations directory"""
    if not os.path.exists(SIMULATIONS_PATH):
        logging.warning(f"Creating simulations directory at {SIMULATIONS_PATH}")
        os.makedirs(SIMULATIONS_PATH, exist_ok=True)
    
    txt_files = sorted([str(f) for f in Path(SIMULATIONS_PATH).glob("*.txt")])
    
    if not txt_files:
        logging.warning(f"No .txt files found in {SIMULATIONS_PATH}")
        return {}
    
    all_results = {}
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            conversation_rounds = extract_conversation_rounds(content)
            file_results = []
            
            for i, round_data in enumerate(conversation_rounds):
                if "AgentX" in round_data and "AgentY" in round_data and round_data["AgentX"] and round_data["AgentY"]:
                    evaluation = evaluate_conversation(llm, round_data["AgentX"], round_data["AgentY"])
                    file_results.append({
                        "round": i+1,
                        "evaluation": evaluation
                    })
                else:
                    logging.warning(f"Skipping incomplete round in {Path(file_path).name}")
            
            if file_results:
                all_results[Path(file_path).name] = file_results
                logging.info(f"Processed {Path(file_path).name} with {len(file_results)} rounds")
            else:
                all_results[Path(file_path).name] = {"error": "No valid rounds found"}
                logging.warning(f"No valid rounds found in {Path(file_path).name}")
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            all_results[Path(file_path).name] = {
                "error": f"Failed to process file: {str(e)}"
            }
    
    return all_results

def save_results(results: Dict[str, Any], output_path: str = OUTPUT_FILE):
    """Save evaluation results to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info(f"Results saved to {output_path}")

def main():
    """Main execution function"""
    logging.info("Starting simulation evaluation")
    
    # Model path - adjust this to your actual model location
    model_path = os.path.join(ROOT_DIRECTORY, "models", MODEL_BASENAME)
    
    # Verify model exists
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        logging.info("Please ensure the model file exists or download it with:")
        logging.info(f"huggingface-cli download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF {MODEL_BASENAME} --local-dir {os.path.join(ROOT_DIRECTORY, 'models')}")
        sys.exit(1)
    
    # Load the language model
    llm = load_model(model_path)
    if not llm:
        sys.exit(1)
    
    # Process all simulation files
    results = process_simulation_files(llm)
    
    # Save the results
    save_results(results)
    
    logging.info("Evaluation complete")

if __name__ == "__main__":
    main()