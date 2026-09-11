import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS normalization
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.gamma * x_normalized


class GatedAttention(nn.Module):
    """Grouped Query Attention with gating"""
    def __init__(self, d_model, n_heads, n_kv_heads=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = d_model // n_heads

        # Grouped Query Attention: fewer KV heads than Q heads
        self.q_dim = n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        # Combined QKV projection (Q + K + V)
        self.qkv = nn.Linear(d_model, self.q_dim + 2 * self.kv_dim, bias=False)
        # Output projection
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)

        # Split into Q, K, V
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        # Reshape Q
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Reshape K, V
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KV heads to match Q heads (for grouped query attention)
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)

        # Output projection
        out = self.o(out)
        return out


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network"""
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x):
        # SwiGLU: swish(gate) * up
        gate = F.silu(self.gate_proj(x))  # SiLU is same as Swish
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden)


class TransformerLayer(nn.Module):
    """Transformer layer with gating and RMSNorm"""
    def __init__(self, d_model, n_heads, d_ffn, n_kv_heads=None):
        super().__init__()
        # Attention components
        self.norm = RMSNorm(d_model)
        self.attn = GatedAttention(d_model, n_heads, n_kv_heads)
        self.gate = nn.Parameter(torch.zeros(d_model))

        # FFN components
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ffn)
        self.gate_ffn = nn.Parameter(torch.zeros(d_model))

    def forward(self, x, mask=None):
        # Attention with gating and residual
        residual = x
        x = self.norm(x)
        x = self.attn(x, mask)
        x = residual + torch.sigmoid(self.gate) * x

        # FFN with gating and residual
        residual = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = residual + torch.sigmoid(self.gate_ffn) * x

        return x


class GPTModel(nn.Module):
    """GPT-style transformer with gating and GQA"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ffn, n_kv_heads=None, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.embed = nn.Embedding(vocab_size, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ffn, n_kv_heads)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.out_norm = RMSNorm(d_model)

        # Note: No separate output projection - we'll use weight tying with embeddings

    def forward(self, x, mask=None):
        # Embed tokens
        x = self.embed(x)

        # Create causal mask if not provided
        if mask is None:
            seq_len = x.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Apply final layer norm
        x = self.out_norm(x)

        # Project to vocabulary using tied weights (embed.weight transposed)
        logits = F.linear(x, self.embed.weight)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens autoregressively

        Args:
            idx: (batch_size, seq_len) tensor of token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: if set, only sample from top k most likely tokens
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

            # Forward pass
            logits = self(idx_cond)

            # Get logits for last position
            logits = logits[:, -1, :] / temperature

            # Optionally crop to top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
