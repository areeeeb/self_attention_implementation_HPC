import torch
import torch.nn as nn
import numpy as np
import os
import time

# Parameters
EMBEDDINGS_DIR = "../wikitext_embeddings"  # Directory where embeddings are saved
SEQUENCE_LENGTHS = [32, 64, 128, 256, 512, 1024]
HIDDEN_SIZE = 768  # GPT-2 hidden size
NUM_HEADS = 1     # Number of attention heads
NUM_SAMPLES = 20   # Number of samples per sequence type
USE_GPU = False    # Set to True to use GPU; False for CPU

# Define a simple self-attention layer
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)

    def forward(self, x):
        # Transpose for MultiheadAttention: (batch, seq_len, hidden) -> (seq_len, batch, hidden)
        x = x.transpose(0, 1)
        output, _ = self.attention(x, x, x)
        return output.transpose(0, 1)  # Back to (batch, seq_len, hidden)

# Initialize device
device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize self-attention layer
self_attention = SelfAttention(HIDDEN_SIZE, NUM_HEADS).to(device)

# Timing function
def measure_attention_time(embeddings, attention_layer):
    total_time = 0
    for embedding in embeddings:
        embedding = torch.tensor(embedding, dtype=torch.float32).to(device)  # Move to the chosen device
        start_time = time.time()
        with torch.no_grad():  # Disable gradient calculation for inference
            _ = attention_layer(embedding.unsqueeze(0))  # Add batch dimension
        end_time = time.time()
        total_time += (end_time - start_time)
    return total_time / len(embeddings)  # Average time per sample

# Load embeddings and measure time
for sequence_length in SEQUENCE_LENGTHS:
    # Load embeddings
    file_path = os.path.join(EMBEDDINGS_DIR, f"gpt2_embedded_{sequence_length}.npy")
    if not os.path.exists(file_path):
        print(f"ERROR: Embedding file for sequence length {sequence_length} not found.")
        continue

    print(f"Processing sequence length: {sequence_length}")
    embeddings = np.load(file_path)  # Shape: (20, sequence_length, hidden_size)
    print(f"Loaded embeddings with shape: {embeddings.shape}")

    # Measure average time
    avg_time = measure_attention_time(embeddings, self_attention)
    print(f"Average time per sample for sequence length {sequence_length}: {avg_time:.6f} seconds")
