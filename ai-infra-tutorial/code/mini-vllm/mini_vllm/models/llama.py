"""Llama (TinyLlama-1.1B) decoder-only LM. Uses our AttentionBackend
(Plan 1 Torch backend, Plan 2 Triton). HF safetensors weights are loaded
via mini_vllm.models.llama_loader; this file contains only architecture.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_vllm.config import ModelConfig
from mini_vllm.backends.interface import AttentionBackend


# ---------------------------------------------------------------------------
# Rotary positional embedding
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables for RoPE. Indexed by absolute token position
    so the paged KV cache (which keys K/V by absolute position) works correctly.

    Layout matches HF: emb = concat(freqs, freqs, dim=-1) so the rotate-half
    convention applies uniformly. Shapes:
        cos, sin: [max_position, head_dim]
    """
    def __init__(self, head_dim: int, max_position: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_position).float()
        freqs = torch.outer(positions, inv_freq)        # [max_pos, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)          # [max_pos, head_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # positions: [N] (any 1D tensor of absolute positions, possibly mixed across seqs)
        return self.cos_cached[positions], self.sin_cached[positions]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """q: [N, H_q, D], k: [N, H_kv, D], cos/sin: [N, D].
    Broadcasts cos/sin over the head dim. The rotate-half convention matches
    HF Llama (and our test golden).
    """
    cos = cos.unsqueeze(1)   # [N, 1, D]
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot
