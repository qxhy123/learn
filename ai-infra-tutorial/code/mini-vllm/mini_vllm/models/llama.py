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


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class LlamaRMSNorm(nn.Module):
    """Root-mean-square LayerNorm without bias / mean subtraction.
    out = (x / rms(x)) * weight,  rms(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to fp32 for the norm to stay numerically stable, then back.
        input_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return (self.weight * x).to(input_dtype)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class LlamaMLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x)).

    HF stores gate_proj and up_proj as separate matrices. We FUSE them
    into a single `gate_up_proj` (output dim = 2 * intermediate_size) and
    split internally — this is one of the optimizations vLLM uses, and
    keeps the loader honest about the layout we expect.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = self.gate_up_proj(x)
        gate, up = gu.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# ---------------------------------------------------------------------------
# Attention layer
# ---------------------------------------------------------------------------

class LlamaAttention(nn.Module):
    """GQA + RoPE attention. Q/K/V are FUSED into a single qkv_proj.
    The K/V written to the paged cache are the *post-RoPE* values, matching
    HF's behavior.
    """
    def __init__(self, cfg: ModelConfig, backend: AttentionBackend,
                 rotary: RotaryEmbedding):
        super().__init__()
        self.cfg = cfg
        self.backend = backend
        self.rotary = rotary
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5

        q_size = cfg.num_attention_heads * cfg.head_dim
        kv_size = cfg.num_kv_heads * cfg.head_dim
        self.qkv_proj = nn.Linear(cfg.hidden_size, q_size + 2 * kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, cfg.hidden_size, bias=False)

    def forward(self, x, positions, slot_mapping, kv_cache,
                prefill_block_table, prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                decode_block_table, decode_context_lens):
        N = x.shape[0]
        qkv = self.qkv_proj(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        # Apply RoPE BEFORE writing K to cache — paged cache holds rotated K.
        cos, sin = self.rotary(positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        kc, vc = kv_cache
        self.backend.reshape_and_cache(k, v, kc, vc, slot_mapping)

        out_pre = None
        out_dec = None
        if num_prefill_tokens > 0:
            out_pre = self.backend.prefill(
                q[:num_prefill_tokens], kc, vc, prefill_block_table,
                prefill_seq_lens, prefill_query_lens, self.scale)
        if N - num_prefill_tokens > 0:
            qd = q[num_prefill_tokens:]
            out_dec = self.backend.decode(
                qd, kc, vc, decode_block_table, decode_context_lens, self.scale)

        if out_pre is not None and out_dec is not None:
            out = torch.cat([out_pre, out_dec], dim=0)
        else:
            out = out_pre if out_pre is not None else out_dec
        out = out.reshape(N, self.num_heads * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Decoder layer + full model
# ---------------------------------------------------------------------------

class LlamaDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: AttentionBackend,
                 rotary: RotaryEmbedding):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = LlamaAttention(cfg, backend, rotary)
        self.post_attention_layernorm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = LlamaMLP(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x, positions, slot_mapping, kv_cache,
                prefill_block_table, prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                decode_block_table, decode_context_lens):
        h = self.self_attn(self.input_layernorm(x), positions, slot_mapping, kv_cache,
                           prefill_block_table, prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                           decode_block_table, decode_context_lens)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: AttentionBackend):
        super().__init__()
        self.config = cfg
        self.backend = backend
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        # Single shared RoPE instance reused by all layers.
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings,
                                      base=cfg.rope_theta)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(cfg, backend, self.rotary)
            for _ in range(cfg.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if cfg.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids, positions, slot_mapping, kv_caches,
                prefill_block_table, prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                decode_block_table, decode_context_lens,
                sample_indices: torch.Tensor):
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, positions, slot_mapping, kv_caches[i],
                      prefill_block_table, prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                      decode_block_table, decode_context_lens)
        x = self.norm(x)
        x_sample = x[sample_indices]
        if self.lm_head is None:
            logits = x_sample @ self.embed_tokens.weight.T
        else:
            logits = self.lm_head(x_sample)
        return logits

    @staticmethod
    def tinyllama_config() -> ModelConfig:
        """Hardcoded config for TinyLlama-1.1B-Chat-v1.0."""
        return ModelConfig(
            model_type="llama",
            vocab_size=32000,
            hidden_size=2048,
            num_hidden_layers=22,
            num_attention_heads=32,
            num_kv_heads=4,
            head_dim=64,
            max_position_embeddings=2048,
            intermediate_size=5632,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            tie_word_embeddings=False,   # TinyLlama unties them
            dtype="float32",
        )
