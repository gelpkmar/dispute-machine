from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the DialoGPT model (fine-tuned for dialogues)
model_name = "microsoft/DialoGPT-medium"  # DialoGPT medium size
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up conversation between Agent 1 and Agent 2
conversation = [
    "Agent 1: I owe you $50, but I can’t pay you back yet.\n",
    "Agent 2: Why haven’t you paid me back? I need that money!\n"
]

# Simulate a conversation between two agents
def simulate_conversation(conversation):
    inputs = tokenizer(" ".join(conversation), return_tensors="pt")
    
    # Generate a single response from the model
    outputs = model.generate(
        inputs["input_ids"],
        max_length=100,  # Limiting the maximum length to avoid over-generation
        num_return_sequences=1,  # Ensure we only generate one sequence
        do_sample=True,  # Enable sampling to add some variability
        top_p=0.95,  # Use nucleus sampling
        temperature=0.7,  # Control randomness
        pad_token_id=tokenizer.eos_token_id  # Ensure correct padding token is used
    )
    
    # Decode and return the generated output
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate and print the conversation response
response = simulate_conversation(conversation)
print(response)
