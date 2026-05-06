"""Toy GPT-2-style decoder-only LM used for tests, examples, and the lab
chapter's first walkthrough. Random-initialized weights — outputs are
gibberish but the engine plumbing is exercised end-to-end."""
from __future__ import annotations
from typing import List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_vllm.config import ModelConfig
from mini_vllm.backends.interface import AttentionBackend


class ToyAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: AttentionBackend):
        super().__init__()
        self.cfg = cfg
        self.backend = backend
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        # Fused QKV
        self.qkv_proj = nn.Linear(cfg.hidden_size,
            (cfg.num_attention_heads + 2 * cfg.num_kv_heads) * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim,
                                cfg.hidden_size, bias=True)

    def forward(self, x, slot_mapping, kv_cache,
                prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                decode_block_table, decode_context_lens):
        N = x.shape[0]
        qkv = self.qkv_proj(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        # Write all current K/V into the paged cache.
        kc, vc = kv_cache
        self.backend.reshape_and_cache(k, v, kc, vc, slot_mapping)

        out_pre = None
        out_dec = None
        if num_prefill_tokens > 0:
            out_pre = self.backend.prefill(
                q[:num_prefill_tokens], k[:num_prefill_tokens], v[:num_prefill_tokens],
                prefill_seq_lens, prefill_query_lens, self.scale)
        if N - num_prefill_tokens > 0:
            qd = q[num_prefill_tokens:]   # [B_dec, H, D]
            out_dec = self.backend.decode(
                qd, kc, vc, decode_block_table, decode_context_lens, self.scale)

        if out_pre is not None and out_dec is not None:
            out = torch.cat([out_pre, out_dec], dim=0)
        else:
            out = out_pre if out_pre is not None else out_dec
        out = out.reshape(N, self.num_heads * self.head_dim)
        return self.o_proj(out)


class ToyMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class ToyBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: AttentionBackend):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = ToyAttention(cfg, backend)
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = ToyMLP(cfg)

    def forward(self, x, slot_mapping, kv_cache,
                prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                decode_block_table, decode_context_lens):
        h = self.attn(self.ln1(x), slot_mapping, kv_cache,
                      prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                      decode_block_table, decode_context_lens)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class ToyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: AttentionBackend):
        super().__init__()
        self.config = cfg
        self.backend = backend
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.blocks = nn.ModuleList([ToyBlock(cfg, backend) for _ in range(cfg.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(cfg.hidden_size)
        if cfg.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids, positions, slot_mapping, kv_caches,
                prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                decode_block_table, decode_context_lens,
                sample_indices: torch.Tensor):
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        for i, blk in enumerate(self.blocks):
            x = blk(x, slot_mapping, kv_caches[i],
                    prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                    decode_block_table, decode_context_lens)
        x = self.ln_f(x)
        # Only compute logits at the positions we actually need to sample from
        x_sample = x[sample_indices]
        if self.lm_head is None:
            logits = x_sample @ self.tok_emb.weight.T
        else:
            logits = self.lm_head(x_sample)
        return logits

    @classmethod
    def random_init(cls, backend: AttentionBackend,
                    vocab_size: int = 50257,
                    n_layer: int = 6, d_model: int = 384, n_head: int = 6,
                    max_pos: int = 1024, dtype: torch.dtype = torch.float32,
                    device: str = "cpu", seed: int = 0) -> "ToyGPT":
        torch.manual_seed(seed)
        cfg = ModelConfig(
            model_type="toy_gpt", vocab_size=vocab_size, hidden_size=d_model,
            num_hidden_layers=n_layer, num_attention_heads=n_head, num_kv_heads=n_head,
            head_dim=d_model // n_head, max_position_embeddings=max_pos,
            intermediate_size=4 * d_model, dtype=str(dtype).split('.')[-1],
        )
        m = cls(cfg, backend).to(device=device, dtype=dtype)
        return m
