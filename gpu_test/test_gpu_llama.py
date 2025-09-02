import time
import torch
import csv
from transformers import LlamaForCausalLM, LlamaTokenizer

def measure_latency(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    return time.perf_counter() - start_time, result

def compute_position_embeddings(seq_length, dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(seq_length, device=device).type_as(inv_freq)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

model_names = ["meta-llama/Llama-2-7b-hf", 
               "meta-llama/Llama-2-13b-hf"
               ]
sequence_lengths = [64,127, 128, 256, 512, 1024, 2048, 4096]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('llama2_model_latencies.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Model', 'Sequence Length', 'Attention Layer Latency (s)', 'Overall Model Latency (s)'])

    for model_name in model_names:
        print(f"\nProfiling {model_name}")
        
        try:
            # Load model and tokenizer in half precision
            model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
            tokenizer = LlamaTokenizer.from_pretrained(model_name)

            for seq_length in sequence_lengths:
                print(f"  Sequence length: {seq_length}")
                
                # Generate a random sequence of tokens
                random_tokens = torch.randint(0, tokenizer.vocab_size, (1, seq_length), dtype=torch.long).to(device)

                # Compute position embeddings
                cos, sin = compute_position_embeddings(seq_length, model.config.hidden_size // model.config.num_attention_heads, device)
                position_embeddings = (cos.half().unsqueeze(0), sin.half().unsqueeze(0))

                # Generate hidden states
                with torch.no_grad():
                    hidden_states = model.model.embed_tokens(random_tokens).half()

                # Measure latency for attention layers
                attention_layer = model.model.layers[0].self_attn
                torch.cuda.synchronize()
                attention_latency, _ = measure_latency(
                    attention_layer.forward,
                    hidden_states=hidden_states,
                    attention_mask=None,
                    position_embeddings=position_embeddings,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
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
