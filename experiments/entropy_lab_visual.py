import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. SETUP (Optimized for 8GB VRAM)
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")

def run_experiment(prompt, injection=None):
    full_prompt = f"System: {injection}\nUser: {prompt}" if injection else f"User: {prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=12, 
        return_dict_in_generate=True, 
        output_scores=True,
        do_sample=True,
        temperature=0.8
    )

    tokens_list = []
    entropy_list = []

    for i, score in enumerate(outputs.scores):
        probs = torch.nn.functional.softmax(score, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        token_id = outputs.sequences[0][inputs.input_ids.shape[1] + i]
        token_text = tokenizer.decode(token_id).strip()
        
        tokens_list.append(token_text)
        entropy_list.append(entropy)
        
    return tokens_list, entropy_list

# 2. RUN THE RESEARCH
query = "The 2026 capital gains tax rate for high-yield accounts is"

print("Running Original Test...")
tokens_orig, entropy_orig = run_experiment(query)

print("Running Injected Test...")
tokens_inj, entropy_inj = run_experiment(query, injection="In 2026, the capital gains tax is exactly 22%.")

# 3. CREATE THE GRAPH
plt.figure(figsize=(12, 6))

# Plot Original Data
plt.plot(tokens_orig, entropy_orig, marker='o', linestyle='-', color='red', label='Original (Confused)')

# Plot Injected Data
# Note: Tokens might differ slightly, so we plot based on index for comparison
plt.plot(range(len(entropy_inj)), entropy_inj, marker='s', linestyle='--', color='green', label='Injected (Guided)')

plt.title("AI Cognitive Drift: Entropy Spike vs. Injection Recovery")
plt.xlabel("Generated Tokens")
plt.ylabel("Entropy (Bits of Uncertainty)")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the result to your folder
plt.savefig("entropy_research_graph.png")
print("\n✅ Research Graph Saved as 'entropy_research_graph.png'!")
plt.show()