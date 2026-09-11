# GPT-Style Transformer Model

A gated transformer language model with Grouped Query Attention (GQA).

## Model Architecture

- **Type**: GPT-style autoregressive transformer
- **Vocabulary Size**: 16,384 tokens
- **Model Dimension**: 512
- **Layers**: 24
- **Attention Heads**: 8 (with 2 KV heads for GQA)
- **FFN Hidden Dimension**: 512
- **Features**:
  - Grouped Query Attention (GQA) for efficient inference
  - SwiGLU activation in FFN
  - RMSNorm for layer normalization
  - Gating mechanisms for conditional computation
  - Weight tying between embeddings and output projection

## Files

- `model.py` - Model architecture definition
- `inference.py` - Inference and text generation script
- `final.pt` - Trained model checkpoint (step 200)

## Quick Start

### 1. Load the Model

```python
from inference import load_model, generate_text

# Load the model
model = load_model('final.pt')
```

### 2. Generate Text

To use the model, you need a tokenizer. Here's an example with token IDs:

```python
# Example with token IDs
prompt_tokens = [100, 200, 300]

generated_tokens = generate_text(
    model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)

print(f"Generated: {generated_tokens}")
```

### 3. With a Custom Tokenizer

You'll need to implement or use a tokenizer that matches the training:

```python
# Example with a hypothetical tokenizer
from your_tokenizer import encode, decode

# Encode text to tokens
prompt = "Hello, world!"
prompt_tokens = encode(prompt)

# Generate
generated_tokens = generate_text(
    model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=100,
    temperature=0.8
)

# Decode back to text
generated_text = decode(generated_tokens)
print(generated_text)
```

## Generation Parameters

- `max_new_tokens`: Number of tokens to generate (default: 100)
- `temperature`: Sampling temperature
  - Lower (0.1-0.7): More focused/deterministic
  - Normal (0.8-1.0): Balanced
  - Higher (1.0+): More creative/random
- `top_k`: Limit sampling to top k tokens (default: 50)
  - Lower: More focused
  - Higher: More diverse
  - `None`: No filtering

## Requirements

```bash
uv add torch numpy
```

## Run Example

```bash
uv run inference.py
```

## Model Training Info

- Training step: 200
- Precision: bfloat16

## Next Steps

To use this model effectively, you need:

1. **Tokenizer**: The original tokenizer used during training, or
2. **Custom Tokenizer**: Implement your own (BPE, character-level, etc.)
3. **Training Data**: Information about what the model was trained on helps with proper prompting

## Example Usage

```python
import torch
from inference import load_model

# Load model
model = load_model('final.pt')

# Direct inference
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Your token IDs
logits = model(input_ids)
print(f"Logits shape: {logits.shape}")  # [1, 5, 16384]

# Generation
output = model.generate(
    input_ids,
    max_new_tokens=20,
    temperature=0.9,
    top_k=40
)
print(f"Generated tokens: {output[0].tolist()}")
```
