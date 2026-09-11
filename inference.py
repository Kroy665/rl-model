import torch
from model import GPTModel


def load_model(checkpoint_path='final.pt'):
    """Load the trained model from checkpoint"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']

    # Extract model configuration
    vocab_size, d_model = state_dict['embed.weight'].shape
    n_layers = len([k for k in state_dict.keys() if 'layers.' in k and '.norm.gamma' in k])
    n_heads = 8
    d_ffn, _ = state_dict['layers.0.ffn.gate_proj.weight'].shape

    # Calculate n_kv_heads from qkv weight shape
    qkv_dim, _ = state_dict['layers.0.attn.qkv.weight'].shape
    head_dim = d_model // n_heads
    # qkv_dim = q_dim + 2 * kv_dim = n_heads * head_dim + 2 * n_kv_heads * head_dim
    # 768 = 512 + 2 * n_kv_heads * 64
    # 256 = 2 * n_kv_heads * 64
    # n_kv_heads = 2
    n_kv_heads = (qkv_dim - d_model) // (2 * head_dim)

    print(f"Loading model: vocab_size={vocab_size}, d_model={d_model}, "
          f"n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, d_ffn={d_ffn}")

    # Create model
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ffn=d_ffn,
        n_kv_heads=n_kv_heads
    )

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded successfully from step {checkpoint['step']}")
    return model


def generate_text(model, prompt_tokens, max_new_tokens=100, temperature=0.8, top_k=50):
    """
    Generate text from the model

    Args:
        model: The GPT model
        prompt_tokens: List of token indices to start generation
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (0.0 = greedy, 1.0 = normal, >1.0 = more random)
        top_k: Only sample from top k tokens (None = no filtering)

    Returns:
        List of generated token indices
    """
    # Convert to tensor
    idx = torch.tensor([prompt_tokens], dtype=torch.long)

    # Generate
    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    return output[0].tolist()


def main():
    """Example inference"""
    print("Loading model...")
    model = load_model('final.pt')

    print("\nModel loaded! Ready for inference.")
    print("\nExample: Generating from random tokens...")

    # Since we don't have a tokenizer, generate from random starting tokens
    # In practice, you would use your tokenizer here
    prompt_tokens = [100, 200, 300]  # Example token IDs

    generated_tokens = generate_text(
        model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50
    )

    print(f"\nPrompt tokens: {prompt_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total length: {len(generated_tokens)} tokens")

    print("\n" + "="*60)
    print("NOTE: To use this model properly, you need:")
    print("1. The tokenizer that was used during training")
    print("2. Or implement a custom tokenizer (e.g., BPE, character-level)")
    print("\nUsage:")
    print("  from inference import load_model, generate_text")
    print("  model = load_model('final.pt')")
    print("  tokens = generate_text(model, [1, 2, 3], max_new_tokens=100)")
    print("="*60)


if __name__ == "__main__":
    main()
