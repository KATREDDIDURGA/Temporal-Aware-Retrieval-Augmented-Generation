'''
This script is your "Sentry." It loads a model in 4-bit mode to fit your
8GB VRAM and prints the "heartbeat" (Entropy) for every word the AI generates.'''



'''
torch: This is the engine. It handles all the heavy math (tensors) and talks to your NVIDIA GPU.

AutoModelForCausalLM: This is a "factory" class. It automatically finds and loads the right model architecture for generating text.

AutoTokenizer: This translates human words ("Hello") into numbers the AI understands ([15433]).

BitsAndBytesConfig: This is your VRAM Saver. It’s the tool that allows us to shrink the model so it fits on your laptop.'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig



# 1. 4-bit Configuration (Crucial for 8GB GPUs)

'''
load_in_4bit=True: Normally, an AI stores its knowledge in "16-bit" numbers. We are squashing those into "4-bit" numbers. It's like turning a high-definition video into a smaller file that still looks great but uses 4x less space.

torch.float16: When the AI actually "thinks," it will briefly use 16-bit math for accuracy, then go back to 4-bit storage.

nf4: This stands for "NormalFloat 4." It’s a special mathematical distribution designed specifically for AI weights. It ensures the AI stays smart even after being squashed.'''


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# 2. Load Model & Tokenizer
# We use Phi-2 because it is powerful but small enough for local research

'''
model_id: We chose Phi-2 because it's a "Small Language Model." It has 2.7 Billion parameters. On your 8GB GPU, this model is perfect because it’s fast and accurate.

device_map="auto": This tells the code: "Look at the hardware. If there's an NVIDIA GPU, put the model there. If not, use the CPU." (On your Alienware, it will prioritize the GPU).'''


model_id = "microsoft/phi-2"
print(f"Loading {model_id} on your GPU...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=quant_config, 
    device_map="auto"
)

'''
max_new_tokens=10: We only ask for 10 words. This keeps our test quick.

output_scores=True: This is CRITICAL. Usually, AI models only give you the final word. By setting this to True, we are asking the AI to show us its internal "Scoreboard" (the Logits) for every word it considered.

do_sample=True & temperature=0.7: This allows the AI to be a little creative. If we set temperature to 0, it would always pick the #1 word, and we wouldn't see the "Entropy" (the spread of other ideas) as clearly.'''


# 3. Test Cases (Confident vs. Confused)
prompts = [
    "The capital of France is", # Expected: Low Entropy
    "In the year 2026, the red cat that is actually a blue dog says" # Expected: High Entropy
]

print("-" * 30)

for text in prompts:
    print(f"\nTESTING PROMPT: '{text}'")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Generate 10 words and capture the scores (logits)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=10, 
        return_dict_in_generate=True, 
        output_scores=True,
        do_sample=True,
        temperature=0.7 # We use 0.7 to allow for some 'thinking' spread
    )

    # 4. Calculate Shannon Entropy for each generated token

    '''This is where your research happens:
    softmax: The "Scoreboard" is full of weird numbers (like 15.2, -3.4). 
    Softmax turns these into percentages (like 80%, 5%). 
    All percentages will add up to 100%.
    The Formula ($H = -\sum p \log p$):
    If one word is 100%, the result is 0 (No Entropy/Confused).
    If 10 words are 10% each, the result is High (High Entropy/Confusion).
    1e-10: This is a tiny "safety" number. 
    In math, you can't take the log of zero. 
    This ensures the code doesn't crash if a word has 0% probability.
    '''
    for i, score in enumerate(outputs.scores):
        # Softmax converts raw scores to probabilities (0.0 to 1.0)
        probs = torch.nn.functional.softmax(score, dim=-1)
        # Shannon Entropy formula: H = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        token_id = outputs.sequences[0][inputs.input_ids.shape[1] + i]
        token_text = tokenizer.decode(token_id)
        
        print(f"Token: {token_text:<12} | Entropy: {entropy:.4f}")