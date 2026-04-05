import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. THE BRAIN: 4-bit Configuration (Optimized for 8GB VRAM)
# This allows the 2.7 Billion parameters to fit into your GPU without crashing.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# 2. THE LAB SETUP: Load Model & Tokenizer
model_id = "microsoft/phi-2"
print(f"Loading {model_id} on your RTX 5050...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=quant_config, 
    device_map="auto"
)

# 3. THE SENTRY & INJECTOR FUNCTION
def run_sentry_test(prompt, injection_hint=None):
    """
    This function processes a prompt and monitors the AI's internal 'Stress' (Entropy).
    If an injection_hint is provided, it acts as a 'Knowledge Injection'.
    """
    # If we have a hint, we wrap the prompt in a "System Rule"
    if injection_hint:
        full_prompt = f"System Rule: {injection_hint}\nUser: {prompt}"
        display_title = "(WITH KNOWLEDGE INJECTION)"
    else:
        full_prompt = f"User: {prompt}"
        display_title = "(ORIGINAL - NO CONTEXT)"
    
    print(f"\n" + "="*50)
    print(f"RUNNING TEST: {display_title}")
    print(f"Prompt: {prompt}")
    print("="*50)

    # Convert words to numbers (Tokens)
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    # Generate 10 words and capture the 'Scoreboard' (Logits)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=10, 
        return_dict_in_generate=True, 
        output_scores=True,
        do_sample=True,
        temperature=0.7
    )

    # Calculate and Print the 'Heartbeat'
    for i, score in enumerate(outputs.scores):
        # 1. Softmax: Turns raw scores into probabilities
        probs = torch.nn.functional.softmax(score, dim=-1)
        
        # 2. Shannon Entropy: Measures the 'Fog' (Uncertainty)
        # H = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        # 3. Decoding: Get the human word
        token_id = outputs.sequences[0][inputs.input_ids.shape[1] + i]
        token_text = tokenizer.decode(token_id)
        
        # 4. Sentry Threshold: If Entropy > 2.0, trigger ALERT
        if entropy > 2.0:
            status = "🚨 ALERT: HIGH CONFUSION"
        else:
            status = "✅ CLEAR: CONFIDENT"
            
        print(f"Token: {token_text:<12} | Entropy: {entropy:.4f} | {status}")

# 4. THE EXPERIMENT
# We use a Wealth Management scenario where the AI doesn't know the specific rule.
wealth_management_query = "The 2026 tax rate for a conservative portfolio is"

# TEST 1: The AI guessing (Expect High Entropy / ALERTs)
run_sentry_test(wealth_management_query)

# TEST 2: The AI with Injection (Expect Low Entropy / CLEAR)
run_sentry_test(
    wealth_management_query, 
    injection_hint="In the 2026 fiscal year, the official tax rate for conservative portfolios is 20%."
)