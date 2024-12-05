'''
    This script saves the attention weights and results for the self-attention layer.
    So we can compare the results of the self-attention layer in pytorch with the results of our
    C++ implementation.
'''

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time

# Parameters
EMBEDDINGS_DIR = "./wikitext_embeddings"  # Directory where embeddings are saved
SEQUENCE_LENGTHS = [32, 64, 128, 256, 512, 1024]
HIDDEN_SIZE = 768  # GPT-2 hidden size
NUM_HEADS = 1     # Number of attention heads
NUM_SAMPLES = 20   # Number of samples per sequence type
USE_GPU = True    # Set to True to use GPU; False for CPU
OUTPUT_DIR = "./weights"  # Directory to save weights and results

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define a simple self-attention layer
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)

    def forward(self, x):
        # Transpose for MultiheadAttention: (batch, seq_len, hidden) -> (seq_len, batch, hidden)
        x = x.transpose(0, 1)
        output, weights = self.attention(x, x, x)
        return output.transpose(0, 1), weights  # Back to (batch, seq_len, hidden)

# Initialize device
device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize self-attention layer
self_attention = SelfAttention(HIDDEN_SIZE, NUM_HEADS).to(device)

# Save weights of the attention layer
def save_attention_weights(attention_layer, output_dir):
    # Extract weights
    weights = {
        "query_weight": attention_layer.attention.in_proj_weight[:HIDDEN_SIZE].cpu().tolist(),
        "key_weight": attention_layer.attention.in_proj_weight[HIDDEN_SIZE:2*HIDDEN_SIZE].cpu().tolist(),
        "value_weight": attention_layer.attention.in_proj_weight[2*HIDDEN_SIZE:].cpu().tolist(),
        "output_weight": attention_layer.attention.out_proj.weight.cpu().tolist(),
        "query_bias": attention_layer.attention.in_proj_bias[:HIDDEN_SIZE].cpu().tolist(),
        "key_bias": attention_layer.attention.in_proj_bias[HIDDEN_SIZE:2*HIDDEN_SIZE].cpu().tolist(),
        "value_bias": attention_layer.attention.in_proj_bias[2*HIDDEN_SIZE:].cpu().tolist(),
        "output_bias": attention_layer.attention.out_proj.bias.cpu().tolist(),
    }
    # Save weights as JSON
    with open(os.path.join(output_dir, "attention_weights.json"), "w") as f:
        json.dump(weights, f, indent=4)
    print(f"Saved attention weights to {output_dir}/attention_weights.json")

# Save results for all samples of a sequence length in one file
def save_sequence_results(sequence_length, results, output_dir):
    file_path = os.path.join(output_dir, f"results_seq_{sequence_length}.json")
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results for sequence length {sequence_length} to {file_path}")

# Process embeddings and save results
save_attention_weights(self_attention, OUTPUT_DIR)
for sequence_length in SEQUENCE_LENGTHS:
    # Load embeddings
    file_path = os.path.join(EMBEDDINGS_DIR, f"gpt2_embedded_{sequence_length}.npy")
    if not os.path.exists(file_path):
        print(f"ERROR: Embedding file for sequence length {sequence_length} not found.")
        continue

    print(f"Processing sequence length: {sequence_length}")
    embeddings = np.load(file_path)  # Shape: (20, sequence_length, hidden_size)

    results = []
    for sample_index, embedding in enumerate(embeddings):
        embedding = torch.tensor(embedding, dtype=torch.float32).to(device)  # Move to device
        with torch.no_grad():
            output, _ = self_attention(embedding.unsqueeze(0))  # Add batch dimension
        results.append({
            "sample_index": sample_index,
            "output": output.squeeze(0).cpu().tolist()  # Remove batch dimension for saving
        })

    # Save all results for the sequence length in one file
    save_sequence_results(sequence_length, results, OUTPUT_DIR)
