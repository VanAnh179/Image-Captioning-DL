"""Model B: Transformer decoder that reads the 49 ResNet-50 regions through cross-attention.

Attention, masks, positional encoding and the decoder layer are written here; only nn.Linear, nn.Embedding,
nn.LayerNorm, nn.Dropout and softmax come from PyTorch. Boolean masks use True = "may attend".
"""

import math

import torch
from torch import nn

from src.data.region_features import FEATURE_DIM, NUM_REGIONS
from src.data.vietnamese_caption_batch import PAD


def sinusoidal_positions(max_len, d_model):
    """[max_len, d_model] sin/cos table that tells the decoder where each word sits in the sentence."""
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    rate = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    table = torch.zeros(max_len, d_model)
    table[:, 0::2] = torch.sin(position * rate)
    table[:, 1::2] = torch.cos(position * rate[:d_model // 2])
    return table


def causal_mask(length, device=None):
    """[length, length]: row t may attend to columns <= t, never to future tokens."""
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()


def self_attention_mask(token_ids):
    """[B, 1, T, T] = causal AND the key is not a PAD position."""
    causal = causal_mask(token_ids.shape[1], token_ids.device)
    return (causal.unsqueeze(0) & token_ids.ne(PAD).unsqueeze(1)).unsqueeze(1)


def scaled_dot_product_attention(query, key, value, mask=None):
    """softmax(Q Kᵀ / √d_k) V. Shapes [B, H, Tq, d_k], [B, H, Tk, d_k]; mask broadcasts to [B, H, Tq, Tk]."""
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if mask is not None:
        weights = weights.masked_fill(~mask, 0.0)
    return weights @ value, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        self.num_heads, self.d_head = num_heads, d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.d_head).transpose(1, 2)

    def forward(self, query, memory, mask=None):
        context, _ = scaled_dot_product_attention(
            self._split(self.q_proj(query)), self._split(self.k_proj(memory)), self._split(self.v_proj(memory)), mask)
        batch, _, length, _ = context.shape
        context = context.transpose(1, 2).reshape(batch, length, -1)
        return self.out_proj(self.dropout(context))


class DecoderLayer(nn.Module):
    """Masked self-attention over words, cross-attention over regions (K, V from M), feed-forward; each with residual."""

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, self_mask):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, self_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory))  # all 49 regions are valid: no mask
        return x + self.dropout(self.ffn(self.norm3(x)))


class RegionProjection(nn.Module):
    """[B, 49, 2048] ResNet regions -> M [B, 49, d]: Linear + LayerNorm + learned embedding of the region's grid position."""

    def __init__(self, d_model, dropout, feature_dim=FEATURE_DIM, num_regions=NUM_REGIONS):
        super().__init__()
        self.linear = nn.Linear(feature_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.position = nn.Embedding(num_regions, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, regions):
        index = torch.arange(regions.shape[1], device=regions.device)
        return self.dropout(self.norm(self.linear(regions)) + self.position(index))


class TransformerCaptionDecoder(nn.Module):
    """forward(regions [B,49,2048], decoder_input [B,T]) -> logits [B,T,vocab]; position t predicts word t+1."""

    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=3, d_ff=2048, dropout=0.1, max_len=64):
        super().__init__()
        if min(vocab_size, d_model, num_heads, num_layers, d_ff, max_len) <= 0:
            raise ValueError("Decoder dimensions, layer count and max_len must be positive")
        self.d_model = d_model
        self.regions = RegionProjection(d_model, dropout)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.register_buffer("positions", sinusoidal_positions(max_len, d_model), persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers))
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, regions, decoder_input):
        if decoder_input.ndim != 2 or not 0 < decoder_input.shape[1] <= len(self.positions):
            raise ValueError(f"decoder_input must have shape [B,T] with 1 <= T <= {len(self.positions)}")
        memory = self.regions(regions)
        x = self.embedding(decoder_input) * math.sqrt(self.d_model) + self.positions[:decoder_input.shape[1]]
        x = self.dropout(x)
        mask = self_attention_mask(decoder_input)
        for layer in self.layers:
            x = layer(x, memory, mask)
        return self.output(self.final_norm(x))
