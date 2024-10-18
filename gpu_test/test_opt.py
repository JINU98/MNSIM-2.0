import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def test_opt_performance(model_name, input_text, num_iterations=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    # Warm-up run
    model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1000)

    # Measure performance
    total_time = 0
    total_tokens = 0

    for _ in range(num_iterations):
        start_time = time.time()
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1000)
        end_time = time.time()

        tokens_generated = output.shape[1] - input_ids.shape[1]
        total_tokens += tokens_generated
        total_time += end_time - start_time

    avg_tokens_per_second = total_tokens / total_time
    return avg_tokens_per_second

# List of OPT model sizes
model_sizes = [
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
]

# Input text for testing
input_text = '''
In recent years, the rapid growth of technology and the increasing reliance on high-performance computing (HPC) have brought energy efficiency to the forefront of technological discussions. As industries and research organizations increasingly depend on massive computational resources to handle tasks such as artificial intelligence (AI) training, scientific simulations, big data processing, and real-time decision-making, the energy cost associated with these operations has risen significantly.
'''

print("Testing OPT performance on available GPUs:")

for model_name in model_sizes:
    try:
        tokens_per_second = test_opt_performance(model_name, input_text)
        print(f"{model_name}: {tokens_per_second:.2f} tokens/second")
    except Exception as e:
        print(f"Error testing {model_name}: {str(e)}")

print("Performance test completed.")
