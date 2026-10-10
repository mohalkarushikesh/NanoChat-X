"""Print NanoChat-X Model Architecture with Causal Self-Attention."""

import torch
from src.config import GPTConfig
from src.model import NanoGPT, CausalSelfAttention


def print_architecture_details():
    """Print detailed model architecture including causal self-attention."""
    
    # Use default config
    config = GPTConfig()
    
    # Create model
    model = NanoGPT(config)
    
    print("=" * 80)
    print("NANOCHAT-X MODEL ARCHITECTURE")
    print("=" * 80)
    print()
    
    print("Configuration:")
    print(f"  Vocabulary Size: {config.vocab_size}")
    print(f"  Block Size (Context Length): {config.block_size}")
    print(f"  Number of Layers: {config.n_layer}")
    print(f"  Number of Attention Heads: {config.n_head}")
    print(f"  Embedding Dimension: {config.n_embd}")
    print(f"  Head Dimension: {config.n_embd // config.n_head}")
    print(f"  Dropout: {config.dropout}")
    print()
    
    # Print model structure
    print("Model Structure:")
    print(model)
    print()
    
    # Causal Self-Attention
    print("=" * 80)
    print("CAUSAL SELF-ATTENTION")
    print("=" * 80)
    print()
    print(CausalSelfAttention(config))
    print()
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    print()


if __name__ == "__main__":
    print_architecture_details()