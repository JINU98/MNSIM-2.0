import torch
from transformers import OPTForCausalLM
import time
import csv
import gc

def profile_opt_model(model_size, seq_lengths):
    print(f"Loading model OPT-{model_size}")
    try:
        model = OPTForCausalLM.from_pretrained(f"facebook/opt-{model_size}", torch_dtype=torch.float16)
        model.eval()
        model.to("cuda")

        results = []
        
        for seq_length in seq_lengths:
            print(f"  Profiling sequence length: {seq_length}")
            input_ids = torch.randint(0, model.config.vocab_size, (1, seq_length)).to("cuda")
            
            # Measure overall latency
            start_time = time.time()
            with torch.no_grad():
                outputs = model(input_ids)
            overall_latency = time.time() - start_time
            
            # Measure attention latency
            start_time = time.time()
            with torch.no_grad():
                _ = model.model.decoder.layers[0].self_attn(
                    hidden_states=model.model.decoder.embed_tokens(input_ids),
                    attention_mask=None,
                )
            attention_latency = time.time() - start_time
            
            results.append({
                "model_size": model_size,
                "seq_length": seq_length,
                "overall_latency": overall_latency,
                "attention_latency": attention_latency
            })

        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
    except Exception as e:
        print(f"Error profiling OPT-{model_size}: {str(e)}")
        return []

# Adjust these lists based on your GPU memory capacity
model_sizes = ["125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b", "66b"]
seq_lengths = [64, 128, 256, 512, 1024, 2048]

all_results = []

for model_size in model_sizes:
    print(f"Profiling OPT-{model_size}")
    results = profile_opt_model(model_size, seq_lengths)
    all_results.extend(results)
    print(f"Finished profiling OPT-{model_size}")
    print("-" * 40)
    
    # Free up CUDA memory after each model
    torch.cuda.empty_cache()
    gc.collect()

# Save results to CSV file
csv_filename = "opt_model_profiling_results.csv"
csv_headers = ["model_size", "seq_length", "overall_latency", "attention_latency"]

with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
    writer.writeheader()
    for result in all_results:
        writer.writerow(result)

print(f"Results saved to {csv_filename}")
