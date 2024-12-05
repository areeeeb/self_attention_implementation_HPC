'''
    This script loads the attention weights, data and expected results and validates the outputs of the self-attention layer.
    While also timing the inference.
'''

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time

# Parameters
EMBEDDINGS_DIR = "./wikitext_embeddings"  # Directory where embeddings are saved
RESULTS_DIR = "./weights"  # Directory where weights and results are saved
SEQUENCE_LENGTHS = [32, 64, 128, 256, 512, 1024]
HIDDEN_SIZE = 768  # GPT-2 hidden size
NUM_HEADS = 12     # Number of attention heads
USE_GPU = False    # Set to True to use GPU; False for CPU

# Initialize device
device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define self-attention layer
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)

    def forward(self, x):
        x = x.transpose(0, 1)  # Transpose for MultiheadAttention
        output, _ = self.attention(x, x, x)
        return output.transpose(0, 1)  # Back to original shape

# Load attention weights
def load_attention_weights(attention_layer, weights_file):
    with open(weights_file, "r") as f:
        weights = json.load(f)

    # Load weights into the attention layer
    attention_layer.attention.in_proj_weight.data[:HIDDEN_SIZE] = torch.tensor(weights["query_weight"])
    attention_layer.attention.in_proj_weight.data[HIDDEN_SIZE:2*HIDDEN_SIZE] = torch.tensor(weights["key_weight"])
    attention_layer.attention.in_proj_weight.data[2*HIDDEN_SIZE:] = torch.tensor(weights["value_weight"])
    attention_layer.attention.out_proj.weight.data = torch.tensor(weights["output_weight"])
    attention_layer.attention.in_proj_bias.data[:HIDDEN_SIZE] = torch.tensor(weights["query_bias"])
    attention_layer.attention.in_proj_bias.data[HIDDEN_SIZE:2*HIDDEN_SIZE] = torch.tensor(weights["key_bias"])
    attention_layer.attention.in_proj_bias.data[2*HIDDEN_SIZE:] = torch.tensor(weights["value_bias"])
    attention_layer.attention.out_proj.bias.data = torch.tensor(weights["output_bias"])
    print(f"Loaded attention weights from {weights_file}")

# Validate results
def validate_results(expected, actual):
    expected = torch.tensor(expected)
    actual = torch.tensor(actual)
    if not torch.allclose(expected, actual, atol=1e-6):
        print("Validation failed: Outputs do not match!")
        return False
    return True

# Main function to load data, run inference, time it, and validate
def run_and_validate():
    # Initialize self-attention layer
    self_attention = SelfAttention(HIDDEN_SIZE, NUM_HEADS).to(device)

    # Load weights
    weights_file = os.path.join(RESULTS_DIR, "attention_weights.json")
    load_attention_weights(self_attention, weights_file)

    # Process each sequence length
    for sequence_length in SEQUENCE_LENGTHS:
        # Load embeddings
        embeddings_file = os.path.join(EMBEDDINGS_DIR, f"gpt2_embedded_{sequence_length}.npy")
        results_file = os.path.join(RESULTS_DIR, f"results_seq_{sequence_length}.json")
        
        if not os.path.exists(embeddings_file) or not os.path.exists(results_file):
            print(f"ERROR: Missing data or results for sequence length {sequence_length}")
            continue

        print(f"Processing sequence length: {sequence_length}")

        embeddings = np.load(embeddings_file)  # Shape: (20, sequence_length, hidden_size)
        with open(results_file, "r") as f:
            results = json.load(f)

        # Measure inference time and validate outputs
        total_time = 0
        for sample in results:
            sample_index = sample["sample_index"]
            expected_output = sample["output"]
            embedding = torch.tensor(embeddings[sample_index], dtype=torch.float32).to(device)

            # Time the inference
            start_time = time.time()
            with torch.no_grad():
                output = self_attention(embedding.unsqueeze(0)).squeeze(0)  # Add batch dimension
            end_time = time.time()

            total_time += (end_time - start_time)

            # Validate results
            if not validate_results(expected_output, output.cpu().tolist()):
                print(f"Validation failed for sequence {sequence_length}, sample {sample_index}")
                return

        avg_time = total_time / len(results)
        print(f"Average inference time for sequence length {sequence_length}: {avg_time:.6f} seconds")
        print(f"Validation passed for sequence length {sequence_length}")

# Run the function
run_and_validate()