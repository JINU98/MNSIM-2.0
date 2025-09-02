import time
import torch
import csv
from transformers import GPT2Model, GPT2Tokenizer

def measure_latency(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    return time.perf_counter() - start_time, result

model_names = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
sequence_lengths = [64,128, 256, 512, 1024]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('gpt2_model_latencies.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Model', 'Sequence Length', 'Attention Layer Latency (s)', 'Overall Model Latency (s)'])

    for model_name in model_names:
        print(f"\nProfiling {model_name}")
        
        try:
            model = GPT2Model.from_pretrained(model_name).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)

            for seq_length in sequence_lengths:
                print(f"  Sequence length: {seq_length}")
                
                # Generate a random sequence of tokens
                random_tokens = torch.randint(0, tokenizer.vocab_size, (1, seq_length)).to(device)
                
                # Generate hidden states
                with torch.no_grad():
                    hidden_states = model.wte(random_tokens) + model.wpe(torch.arange(seq_length).unsqueeze(0).to(device))

                # Measure latency for attention layers
                attention_layer = model.h[0].attn
                torch.cuda.synchronize()
                attention_latency, _ = measure_latency(
                    attention_layer.forward,
                    hidden_states=hidden_states,
                    layer_past=None,
                    attention_mask=None,
                    head_mask=None,
                    use_cache=False,
                    output_attentions=False,
                )
                torch.cuda.synchronize()

                # Measure latency for overall model
                torch.cuda.synchronize()
                overall_latency, _ = measure_latency(model.forward, random_tokens)
                torch.cuda.synchronize()

                print(f"    Attention Layer Latency: {attention_latency:.6f} seconds")
                print(f"    Overall Model Latency: {overall_latency:.6f} seconds")

                csvwriter.writerow([model_name, seq_length, attention_latency, overall_latency])

        except Exception as e:
            print(f"Error profiling {model_name}: {e}")
            for seq_length in sequence_lengths:
                csvwriter.writerow([model_name, seq_length, "Error", "Error"])
