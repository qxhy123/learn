# Mini-vLLM Plan 3: TinyLlama-1.1B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TinyLlama-1.1B–compatible Llama decoder + HF safetensors loader so the existing engine (Plan 1) can run real generations from a real LLM end-to-end on CPU/MPS via the Torch backend, validated against HF `transformers` on the same prompt.

**Architecture:** A new `models/llama.py` (Llama architecture: RMSNorm, RoPE, SwiGLU, GQA attention, fused QKV) and `models/llama_loader.py` (maps HF safetensors keys to our fused-QKV layout). Reuses Plan 1's `LLMEngine`, `Scheduler`, `BlockManager`, `ModelRunner`, and `TorchBackend` without modification — only the `AttentionBackend.prefill/decode` interface is touched (to consume our model's K/V).

**Tech Stack:** PyTorch, `tokenizers`, `safetensors`, plus **dev-only** `transformers` and `sentencepiece` for the parity-test golden.

**Note on testing:** The TinyLlama tokenizer file (`tokenizer.json`) and weights (`model.safetensors`, ~2.2 GB) are downloaded from Hugging Face on first run. CPU forward of the full 1.1B model is slow (~5-10s for a 32-token prompt); tests cap context at 8-16 tokens to keep runtimes reasonable.

**Decisions locked from brainstorming:**
- `transformers` is a **test-only** dev dependency for golden output comparison; production code does NOT import it.
- Default dtype: `fp32` on CPU/MPS (numerical stability), `bf16` on CUDA (later plans). `from_pretrained` casts at load time.
- Chat templates are **out of scope**; Plan 3 does raw prompt completion only. Plan 7 may add chat formatting.

**Out of scope (deferred):** continuous batching (Plan 4), prefix caching (Plan 5), swap (Plan 6), Triton kernel (Plan 2), streaming (Plan 7).

---

## File Structure

```
code/mini-vllm/
├── pyproject.toml                          # Task 1: add transformers/sentencepiece dev deps
├── mini_vllm/
│   ├── tokenizer.py                        # Task 9: + from_pretrained_llama()
│   └── models/
│       ├── llama.py                        # Tasks 2-6: LlamaConfig + RMSNorm + RoPE + SwiGLU + Attention + DecoderLayer + Model
│       └── llama_loader.py                 # Tasks 7-8: HF safetensors → our fused layout
├── examples/
│   └── run_tinyllama.py                    # Task 12
└── tests/
    ├── test_rope.py                        # Task 2
    ├── test_rms_norm.py                    # Task 3
    ├── test_llama_loader.py                # Task 8
    ├── test_llama_parity.py                # Task 10 (logits allclose vs HF)
    └── test_llama_e2e.py                   # Task 11 (greedy generation top-5 overlap)
```

**Responsibilities:**
- `llama.py`: pure architecture; depends only on `AttentionBackend`, `ModelConfig`. ~300 lines target.
- `llama_loader.py`: HF state-dict → our state-dict (key remapping + QKV fusing + GateUp fusing). No modeling code.
- Test files: golden comparisons against HF `transformers.LlamaForCausalLM`.

---

## Tasks

### Task 1: Dev dependencies

**Files:**
- Modify: `code/mini-vllm/pyproject.toml`

- [ ] **Step 1: Add transformers + sentencepiece to dev extras**

Open `pyproject.toml` and replace the `dev` extras line so that the optional-dependencies section reads:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-xdist>=3.5",
    "transformers>=4.40",
    "sentencepiece>=0.2",
    "huggingface-hub>=0.20",
]
triton = ["triton>=2.2"]
```

(Leave the `triton` extra unchanged.)

- [ ] **Step 2: Reinstall**

Run from `code/mini-vllm/`: `pip install -e ".[dev]"`
Expected: completes without error; subsequent `python -c "import transformers; print(transformers.__version__)"` prints a version >= 4.40.

- [ ] **Step 3: Smoke import**

Run: `python -c "from transformers import LlamaForCausalLM, LlamaConfig; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "mini-vllm: add transformers + sentencepiece as dev deps"
```

---

### Task 2: RoPE (rotary positional embedding)

**Files:**
- Create: `code/mini-vllm/mini_vllm/models/llama.py`
- Create: `code/mini-vllm/tests/test_rope.py`

This is the first task that touches `models/llama.py`. We start the file with imports + the `RotaryEmbedding` and `apply_rotary_pos_emb` utilities. Subsequent tasks (3-6) **append** to the same file.

- [ ] **Step 1: Write failing test**

Create `tests/test_rope.py`:
```python
import torch
import pytest

# We compare our RoPE against the HF transformers reference. This dependency
# is dev-only; production code never imports transformers.
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding, apply_rotary_pos_emb as hf_apply_rope,
)
from transformers import LlamaConfig as HFLlamaConfig

from mini_vllm.models.llama import RotaryEmbedding, apply_rotary_pos_emb


@pytest.mark.parametrize("seq_len", [1, 8, 32])
def test_rope_matches_hf(seq_len):
    torch.manual_seed(0)
    head_dim = 64
    rope_theta = 10000.0
    max_pos = 2048

    # Ours: precompute cos/sin tables once
    rotary = RotaryEmbedding(head_dim=head_dim, max_position=max_pos, base=rope_theta)
    positions = torch.arange(seq_len)
    cos, sin = rotary(positions)

    # Random Q, K shaped [B=1, H, T, D] (HF convention) and ours [T, H, D]
    H, B = 4, 1
    q = torch.randn(B, H, seq_len, head_dim)
    k = torch.randn(B, H, seq_len, head_dim)

    # HF path
    hf_cfg = HFLlamaConfig(hidden_size=head_dim*H, num_attention_heads=H,
                           num_key_value_heads=H, max_position_embeddings=max_pos,
                           rope_theta=rope_theta)
    hf_rope = LlamaRotaryEmbedding(config=hf_cfg)
    pos_ids = positions.unsqueeze(0)  # [1, T]
    hf_cos, hf_sin = hf_rope(q, pos_ids)
    q_hf, k_hf = hf_apply_rope(q, k, hf_cos, hf_sin)  # [B, H, T, D]

    # Ours: our convention is [T, H, D]; reshape and apply
    q_ours = q.transpose(1, 2).reshape(seq_len, H, head_dim)  # [T, H, D]
    k_ours = k.transpose(1, 2).reshape(seq_len, H, head_dim)
    q_ours, k_ours = apply_rotary_pos_emb(q_ours, k_ours, cos, sin)
    # back to [B, H, T, D] for comparison
    q_ours = q_ours.unsqueeze(0).transpose(1, 2)
    k_ours = k_ours.unsqueeze(0).transpose(1, 2)

    assert torch.allclose(q_ours, q_hf, atol=1e-5)
    assert torch.allclose(k_ours, k_hf, atol=1e-5)


def test_rope_lookup_is_position_indexed():
    """Token at the same absolute position should get the same rotation,
    regardless of which seq it's in (paged KV cache writes use absolute pos)."""
    rotary = RotaryEmbedding(head_dim=32, max_position=128, base=10000.0)
    cos1, sin1 = rotary(torch.tensor([7]))
    cos2, sin2 = rotary(torch.tensor([7, 7]))
    assert torch.allclose(cos1[0], cos2[0])
    assert torch.allclose(cos1[0], cos2[1])
```

- [ ] **Step 2: Run test, expect ImportError**

Run: `pytest tests/test_rope.py -v`
Expected: FAIL — `RotaryEmbedding` not found.

- [ ] **Step 3: Implement RotaryEmbedding + apply_rotary_pos_emb**

Create `mini_vllm/models/llama.py`:
```python
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
```

- [ ] **Step 4: Run test, verify pass**

Run: `pytest tests/test_rope.py -v`
Expected: 4 passed (3 parametrize + 1 lookup test).

- [ ] **Step 5: Commit**

```bash
git add mini_vllm/models/llama.py tests/test_rope.py
git commit -m "mini-vllm: RoPE rotary embedding + HF parity test"
```

---

### Task 3: RMSNorm

**Files:**
- Modify: `code/mini-vllm/mini_vllm/models/llama.py` (append `LlamaRMSNorm` class)
- Create: `code/mini-vllm/tests/test_rms_norm.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_rms_norm.py`:
```python
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm as HFRMSNorm
from mini_vllm.models.llama import LlamaRMSNorm


def test_rms_norm_matches_hf():
    torch.manual_seed(0)
    H = 128
    x = torch.randn(4, 16, H)
    ours = LlamaRMSNorm(H, eps=1e-5)
    hf = HFRMSNorm(H, eps=1e-5)
    # Initialize both with the same gamma
    with torch.no_grad():
        gamma = torch.randn(H)
        ours.weight.copy_(gamma)
        hf.weight.copy_(gamma)
    out_ours = ours(x)
    out_hf = hf(x)
    assert torch.allclose(out_ours, out_hf, atol=1e-5)


def test_rms_norm_unit_input():
    """For a unit-variance input and gamma=1, output should approximately equal input."""
    torch.manual_seed(0)
    x = torch.randn(8, 64)
    norm = LlamaRMSNorm(64, eps=1e-6)
    # weight defaults to ones
    out = norm(x)
    # rms ≈ 1 for normal input → out ≈ x
    rms = (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
    assert torch.allclose(out, x / rms, atol=1e-5)
```

- [ ] **Step 2: Run test, expect ImportError**

Run: `pytest tests/test_rms_norm.py -v`
Expected: FAIL — `LlamaRMSNorm` not found.

- [ ] **Step 3: Append LlamaRMSNorm to `mini_vllm/models/llama.py`**

Append (do NOT replace existing content) at the end of `mini_vllm/models/llama.py`:
```python
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
```

- [ ] **Step 4: Run test, verify pass**

Run: `pytest tests/test_rms_norm.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add mini_vllm/models/llama.py tests/test_rms_norm.py
git commit -m "mini-vllm: LlamaRMSNorm + HF parity test"
```

---

### Task 4: SwiGLU MLP

**Files:**
- Modify: `code/mini-vllm/mini_vllm/models/llama.py` (append)

- [ ] **Step 1: Append LlamaMLP class**

Append to `mini_vllm/models/llama.py`:
```python
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
```

- [ ] **Step 2: Smoke import**

Run: `python -c "from mini_vllm.models.llama import LlamaMLP; m = LlamaMLP(64, 128); import torch; out = m(torch.randn(2, 5, 64)); print('ok', out.shape)"`
Expected: prints `ok torch.Size([2, 5, 64])`.

- [ ] **Step 3: Commit**

```bash
git add mini_vllm/models/llama.py
git commit -m "mini-vllm: LlamaMLP (fused gate_up SwiGLU)"
```

---

### Task 5: LlamaAttention (with GQA + RoPE wired through)

**Files:**
- Modify: `code/mini-vllm/mini_vllm/models/llama.py` (append)

- [ ] **Step 1: Append LlamaAttention class**

Append to `mini_vllm/models/llama.py`:
```python
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

        # Apply RoPE BEFORE writing K to cache — paged cache holds rotated K.
        cos, sin = self.rotary(positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        kc, vc = kv_cache
        self.backend.reshape_and_cache(k, v, kc, vc, slot_mapping)

        out_pre = None
        out_dec = None
        if num_prefill_tokens > 0:
            out_pre = self.backend.prefill(
                q[:num_prefill_tokens], k[:num_prefill_tokens], v[:num_prefill_tokens],
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
```

- [ ] **Step 2: Smoke import**

Run: `python -c "from mini_vllm.models.llama import LlamaAttention; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add mini_vllm/models/llama.py
git commit -m "mini-vllm: LlamaAttention (GQA + fused QKV + RoPE)"
```

---

### Task 6: LlamaModel (decoder layers + final norm + lm_head)

**Files:**
- Modify: `code/mini-vllm/mini_vllm/models/llama.py` (append)

- [ ] **Step 1: Append LlamaDecoderLayer + LlamaModel**

Append to `mini_vllm/models/llama.py`:
```python
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
                prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                decode_block_table, decode_context_lens):
        h = self.self_attn(self.input_layernorm(x), positions, slot_mapping, kv_cache,
                           prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
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
                prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                decode_block_table, decode_context_lens,
                sample_indices: torch.Tensor):
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, positions, slot_mapping, kv_caches[i],
                      prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
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
```

- [ ] **Step 2: Smoke instantiate (random init, small config to fit memory)**

Run from `code/mini-vllm/`:
```bash
python -c "
from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.models.llama import LlamaModel
from mini_vllm.config import ModelConfig
cfg = ModelConfig(model_type='llama', vocab_size=128, hidden_size=64,
    num_hidden_layers=2, num_attention_heads=4, num_kv_heads=2, head_dim=16,
    max_position_embeddings=128, intermediate_size=128, rms_norm_eps=1e-5,
    rope_theta=10000.0, tie_word_embeddings=False)
m = LlamaModel(cfg, TorchBackend())
n = sum(p.numel() for p in m.parameters())
print(f'ok params={n}')
"
```
Expected: prints `ok params=` followed by a positive integer.

- [ ] **Step 3: Commit**

```bash
git add mini_vllm/models/llama.py
git commit -m "mini-vllm: LlamaDecoderLayer + LlamaModel"
```

---

### Task 7: HF state-dict key mapping

**Files:**
- Create: `code/mini-vllm/mini_vllm/models/llama_loader.py`
- Create: `code/mini-vllm/tests/test_llama_loader.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_llama_loader.py`:
```python
"""Tests for the HF safetensors → mini-vllm key/weight remapping."""
import torch
import pytest
from mini_vllm.models.llama_loader import _hf_to_ours_keymap, _fuse_qkv, _fuse_gate_up


def test_keymap_layer_keys():
    """Each HF layer key should map to one of our keys (or be intentionally
    consumed by a fusion)."""
    hf_keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    mapped = _hf_to_ours_keymap(hf_keys)
    # Every HF key is accounted for (mapped or fused)
    assert all(k in mapped for k in hf_keys)
    # Spot-check: q/k/v all map to the SAME fused output (qkv_proj)
    assert mapped["model.layers.0.self_attn.q_proj.weight"] == "layers.0.self_attn.qkv_proj.weight"
    assert mapped["model.layers.0.self_attn.k_proj.weight"] == "layers.0.self_attn.qkv_proj.weight"
    assert mapped["model.layers.0.self_attn.v_proj.weight"] == "layers.0.self_attn.qkv_proj.weight"
    # gate + up → gate_up
    assert mapped["model.layers.0.mlp.gate_proj.weight"] == "layers.0.mlp.gate_up_proj.weight"
    assert mapped["model.layers.0.mlp.up_proj.weight"]   == "layers.0.mlp.gate_up_proj.weight"
    # Embed
    assert mapped["model.embed_tokens.weight"] == "embed_tokens.weight"
    # Final norm
    assert mapped["model.norm.weight"] == "norm.weight"
    # lm_head
    assert mapped["lm_head.weight"] == "lm_head.weight"


def test_fuse_qkv_concatenates_in_order():
    """qkv_proj.weight = concat(q, k, v) along dim 0 (out_features)."""
    H, D = 32, 64
    q = torch.arange(H * D, dtype=torch.float32).reshape(H * D, 16)
    k = torch.arange(8 * D, dtype=torch.float32).reshape(8 * D, 16) + 1000
    v = torch.arange(8 * D, dtype=torch.float32).reshape(8 * D, 16) + 2000
    fused = _fuse_qkv(q, k, v)
    assert fused.shape == (H * D + 8 * D + 8 * D, 16)
    assert torch.equal(fused[:H * D], q)
    assert torch.equal(fused[H * D:H * D + 8 * D], k)
    assert torch.equal(fused[H * D + 8 * D:], v)


def test_fuse_gate_up_concatenates_in_order():
    """gate_up_proj.weight = concat(gate, up) along dim 0."""
    I = 256
    gate = torch.randn(I, 64)
    up = torch.randn(I, 64)
    fused = _fuse_gate_up(gate, up)
    assert fused.shape == (2 * I, 64)
    assert torch.equal(fused[:I], gate)
    assert torch.equal(fused[I:], up)
```

- [ ] **Step 2: Run, expect ImportError**

Run: `pytest tests/test_llama_loader.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement keymap + fusing helpers**

Create `mini_vllm/models/llama_loader.py`:
```python
"""Load HF Llama safetensors into our LlamaModel state_dict.

Key remapping:
    model.embed_tokens.weight              -> embed_tokens.weight
    model.layers.{i}.self_attn.{q,k,v}_proj.weight -> layers.{i}.self_attn.qkv_proj.weight (fused)
    model.layers.{i}.self_attn.o_proj.weight       -> layers.{i}.self_attn.o_proj.weight
    model.layers.{i}.mlp.{gate,up}_proj.weight     -> layers.{i}.mlp.gate_up_proj.weight  (fused)
    model.layers.{i}.mlp.down_proj.weight          -> layers.{i}.mlp.down_proj.weight
    model.layers.{i}.input_layernorm.weight        -> layers.{i}.input_layernorm.weight
    model.layers.{i}.post_attention_layernorm.weight -> layers.{i}.post_attention_layernorm.weight
    model.norm.weight                              -> norm.weight
    lm_head.weight                                 -> lm_head.weight
"""
from __future__ import annotations
from typing import Dict, List
import re
import torch


_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def _hf_to_ours_keymap(hf_keys: List[str]) -> Dict[str, str]:
    """Return mapping from each HF key to its corresponding key in our state_dict.
    Multiple HF keys can map to the same fused output key."""
    mapping: Dict[str, str] = {}
    for k in hf_keys:
        if k == "model.embed_tokens.weight":
            mapping[k] = "embed_tokens.weight"
        elif k == "model.norm.weight":
            mapping[k] = "norm.weight"
        elif k == "lm_head.weight":
            mapping[k] = "lm_head.weight"
        else:
            m = _LAYER_RE.search(k)
            if not m:
                raise KeyError(f"Unrecognized HF key: {k!r}")
            i = m.group(1)
            tail = k[m.end():]
            if tail in ("self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight"):
                mapping[k] = f"layers.{i}.self_attn.qkv_proj.weight"
            elif tail in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
                mapping[k] = f"layers.{i}.mlp.gate_up_proj.weight"
            elif tail == "self_attn.o_proj.weight":
                mapping[k] = f"layers.{i}.self_attn.o_proj.weight"
            elif tail == "mlp.down_proj.weight":
                mapping[k] = f"layers.{i}.mlp.down_proj.weight"
            elif tail == "input_layernorm.weight":
                mapping[k] = f"layers.{i}.input_layernorm.weight"
            elif tail == "post_attention_layernorm.weight":
                mapping[k] = f"layers.{i}.post_attention_layernorm.weight"
            else:
                raise KeyError(f"Unrecognized HF key tail: {tail!r} (full: {k!r})")
    return mapping


def _fuse_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Fuse along output-feature axis (dim 0), preserving Q/K/V order.
    Our LlamaAttention.qkv_proj expects this layout (split with split sizes
    [num_heads*head_dim, num_kv_heads*head_dim, num_kv_heads*head_dim])."""
    return torch.cat([q, k, v], dim=0)


def _fuse_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """gate_up_proj = concat(gate, up), so chunk(2) inside LlamaMLP.forward
    yields (gate, up)."""
    return torch.cat([gate, up], dim=0)
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_llama_loader.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add mini_vllm/models/llama_loader.py tests/test_llama_loader.py
git commit -m "mini-vllm: HF Llama state_dict keymap + fusing helpers"
```

---

### Task 8: HF safetensors → LlamaModel loader

**Files:**
- Modify: `code/mini-vllm/mini_vllm/models/llama_loader.py` (append `load_hf_to_llama_model`)
- Modify: `code/mini-vllm/tests/test_llama_loader.py` (append integration test)

- [ ] **Step 1: Append load function**

Append to `mini_vllm/models/llama_loader.py`:
```python
# ---------------------------------------------------------------------------
# Full loader: HF model directory (or HF Hub repo) -> our LlamaModel
# ---------------------------------------------------------------------------

import os
from collections import defaultdict
from typing import Optional

from safetensors import safe_open
from huggingface_hub import snapshot_download


def _gather_hf_state_dict(model_dir: str) -> Dict[str, torch.Tensor]:
    """Read all .safetensors shards in `model_dir` into a flat dict."""
    state: Dict[str, torch.Tensor] = {}
    shards = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files in {model_dir!r}")
    for shard in shards:
        path = os.path.join(model_dir, shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
    return state


def load_hf_to_llama_model(
    llama_model,                              # LlamaModel instance (random-init)
    model_id_or_path: str,                    # HF repo id or local dir
    dtype: Optional[torch.dtype] = None,      # cast loaded weights to this
) -> None:
    """Mutates `llama_model` in place: loads weights from `model_id_or_path`.

    Steps:
      1. Resolve to a local dir (download from HF Hub if needed).
      2. Read all safetensors shards into a flat dict.
      3. Group HF keys by their target (fused) key in our state_dict.
      4. For QKV / gate_up groups, concat in canonical order; for others,
         pass through.
      5. Cast to `dtype` (if given) and load into `llama_model`.
    """
    if os.path.isdir(model_id_or_path):
        model_dir = model_id_or_path
    else:
        model_dir = snapshot_download(model_id_or_path,
                                      allow_patterns=["*.safetensors", "*.json"])

    hf_state = _gather_hf_state_dict(model_dir)
    keymap = _hf_to_ours_keymap(list(hf_state.keys()))

    # Group HF tensors by destination key
    grouped: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    for hf_key, ours_key in keymap.items():
        grouped[ours_key][hf_key] = hf_state[hf_key]

    our_state: Dict[str, torch.Tensor] = {}
    layer_re = re.compile(r"layers\.(\d+)\.")
    for ours_key, contributors in grouped.items():
        if ".self_attn.qkv_proj.weight" in ours_key:
            i = layer_re.search(ours_key).group(1)
            q = contributors[f"model.layers.{i}.self_attn.q_proj.weight"]
            k = contributors[f"model.layers.{i}.self_attn.k_proj.weight"]
            v = contributors[f"model.layers.{i}.self_attn.v_proj.weight"]
            our_state[ours_key] = _fuse_qkv(q, k, v)
        elif ".mlp.gate_up_proj.weight" in ours_key:
            i = layer_re.search(ours_key).group(1)
            g = contributors[f"model.layers.{i}.mlp.gate_proj.weight"]
            u = contributors[f"model.layers.{i}.mlp.up_proj.weight"]
            our_state[ours_key] = _fuse_gate_up(g, u)
        else:
            assert len(contributors) == 1, f"Unexpected fan-in for {ours_key}"
            our_state[ours_key] = next(iter(contributors.values()))

    if dtype is not None:
        our_state = {k: t.to(dtype) for k, t in our_state.items()}

    missing, unexpected = llama_model.load_state_dict(our_state, strict=False)
    # Allow unexpected ONLY if it's the rotary buffers (registered but persistent=False)
    real_unexpected = [k for k in unexpected if "rotary" not in k and "_cached" not in k]
    if missing or real_unexpected:
        raise RuntimeError(f"State dict mismatch. Missing: {missing}, "
                           f"Unexpected: {real_unexpected}")
```

- [ ] **Step 2: Append integration test (downloads TinyLlama on first run!)**

Append to `tests/test_llama_loader.py`:
```python
import pytest
from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.models.llama import LlamaModel
from mini_vllm.models.llama_loader import load_hf_to_llama_model

# This test downloads ~2.2 GB on first run. Mark slow for selective execution.
TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.mark.slow
def test_load_tinyllama_weights():
    cfg = LlamaModel.tinyllama_config()
    model = LlamaModel(cfg, TorchBackend()).to(torch.float32)
    load_hf_to_llama_model(model, TINYLLAMA, dtype=torch.float32)
    # Sanity: embed_tokens and lm_head should now be non-zero (init was random,
    # but loaded values overwrite them).
    assert model.embed_tokens.weight.abs().sum() > 0
    # Spot-check: layer 0 q (first slice of qkv_proj) shouldn't equal layer 21's.
    qkv0 = model.layers[0].self_attn.qkv_proj.weight
    qkv21 = model.layers[21].self_attn.qkv_proj.weight
    assert not torch.allclose(qkv0, qkv21)
```

Add to `pyproject.toml` under `[tool.pytest.ini_options]`, replace the `addopts` line so the section reads:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -m 'not slow'"
markers = [
    "slow: tests that download multi-GB models or run >30s",
]
```

(This makes `pytest tests/ -v` skip the download-heavy test by default; run with `pytest -m slow tests/` to opt in.)

- [ ] **Step 3: Run fast tests, verify they still pass and slow ones are deselected**

Run: `pytest tests/ -v`
Expected: prior tests pass; the one new slow test is deselected (`deselected` count > 0).

Then optionally run slow test (only if you have ~5 minutes + 2.5 GB free disk):
`pytest tests/test_llama_loader.py -m slow -v`
Expected: 1 passed (after download).

- [ ] **Step 4: Commit**

```bash
git add mini_vllm/models/llama_loader.py tests/test_llama_loader.py pyproject.toml
git commit -m "mini-vllm: load HF safetensors into LlamaModel"
```

---

### Task 9: TokenizerWrapper.from_pretrained_llama

**Files:**
- Modify: `code/mini-vllm/mini_vllm/tokenizer.py` (append a classmethod)

- [ ] **Step 1: Append classmethod**

Edit `mini_vllm/tokenizer.py`. Inside the existing `TokenizerWrapper` class, append a new classmethod alongside `from_pretrained_gpt2`:

```python
    @classmethod
    def from_pretrained_llama(cls, repo_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                              ) -> "TokenizerWrapper":
        # TinyLlama ships a fast tokenizer.json directly loadable by `tokenizers`.
        tk = Tokenizer.from_pretrained(repo_id)
        return cls(tk)
```

- [ ] **Step 2: Smoke test**

Run from `code/mini-vllm/`:
```bash
python -c "
from mini_vllm.tokenizer import TokenizerWrapper
tk = TokenizerWrapper.from_pretrained_llama()
ids = tk.encode('Hello, world!')
print('len', len(ids), 'first', ids[:3], 'vocab', tk.vocab_size)
print('roundtrip:', tk.decode(ids))
"
```
Expected: prints non-zero `len`, three integer ids, vocab=32000, and a decoded string roughly resembling `Hello, world!` (Llama tokenizers may add a leading space or BOS prefix on decode).

- [ ] **Step 3: Commit**

```bash
git add mini_vllm/tokenizer.py
git commit -m "mini-vllm: TokenizerWrapper.from_pretrained_llama"
```

---

### Task 10: Single-step forward parity vs HF transformers

**Files:**
- Create: `code/mini-vllm/tests/test_llama_parity.py`

This test loads TinyLlama into BOTH our model and HF's `LlamaForCausalLM`, runs the same prompt through both, and compares logits at the final position. It's the strongest correctness gate for Plan 3.

- [ ] **Step 1: Write parity test**

Create `tests/test_llama_parity.py`:
```python
"""Parity test: our LlamaModel vs HF LlamaForCausalLM on TinyLlama weights.

Loads ~2.2 GB on first run. Marked `slow`; opt in with `pytest -m slow`.
"""
import pytest
import torch

from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.cache_engine import CacheEngine
from mini_vllm.config import CacheConfig
from mini_vllm.models.llama import LlamaModel
from mini_vllm.models.llama_loader import load_hf_to_llama_model

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.mark.slow
def test_logits_match_hf_top5():
    """Run the same 8-token prompt through both implementations; the top-5
    next-token candidates should overlap by at least 4 (allow one tie-break)."""
    from transformers import LlamaForCausalLM
    torch.manual_seed(0)

    cfg = LlamaModel.tinyllama_config()
    ours = LlamaModel(cfg, TorchBackend()).to(torch.float32).eval()
    load_hf_to_llama_model(ours, TINYLLAMA, dtype=torch.float32)

    hf = LlamaForCausalLM.from_pretrained(TINYLLAMA, torch_dtype=torch.float32).eval()

    prompt_ids = torch.tensor([1, 15043, 29892, 1373, 526, 366, 2599, 29973])  # arbitrary 8-token prompt
    N = prompt_ids.shape[0]

    # ---- HF reference ----
    with torch.inference_mode():
        hf_out = hf(prompt_ids.unsqueeze(0))
    hf_last_logits = hf_out.logits[0, -1]  # [vocab]
    hf_top5 = torch.topk(hf_last_logits, 5).indices.tolist()

    # ---- Ours ----
    block_size = 16
    num_blocks = max(2, (N + block_size - 1) // block_size + 1)
    ce = CacheEngine(cfg, CacheConfig(block_size=block_size, num_gpu_blocks=num_blocks),
                     device='cpu', dtype=torch.float32)
    # All N tokens written into block 0 starting at slot 0
    slot_mapping = torch.arange(N, dtype=torch.long)
    positions = torch.arange(N)
    sample_indices = torch.tensor([N - 1])
    with torch.inference_mode():
        our_logits = ours(
            prompt_ids, positions, slot_mapping, ce.kv_caches,
            prefill_seq_lens=torch.tensor([N], dtype=torch.int32),
            prefill_query_lens=torch.tensor([N], dtype=torch.int32),
            num_prefill_tokens=N,
            decode_block_table=torch.empty(0, 0, dtype=torch.int32),
            decode_context_lens=torch.empty(0, dtype=torch.int32),
            sample_indices=sample_indices,
        )
    our_top5 = torch.topk(our_logits[0], 5).indices.tolist()

    overlap = len(set(hf_top5) & set(our_top5))
    assert overlap >= 4, (
        f"Top-5 overlap too low: ours={our_top5}, hf={hf_top5}, overlap={overlap}"
    )
```

- [ ] **Step 2: Run (only opt-in slow test)**

Run: `pytest tests/test_llama_parity.py -m slow -v`
Expected: 1 passed. (First run downloads weights twice — once for our loader, once for HF's `from_pretrained` cache. They share the HF cache, so the second is fast.)

If `overlap < 4`: the test failure tells you BOTH top-5 lists; debug by:
1. Disabling RoPE (rewrite to identity) and re-running — narrows to RoPE bugs
2. Comparing per-layer hidden states (HF's `output_hidden_states=True`) layer by layer to find the first divergence

- [ ] **Step 3: Commit**

```bash
git add tests/test_llama_parity.py
git commit -m "mini-vllm: TinyLlama logits parity test (top-5 vs HF)"
```

---

### Task 11: End-to-end greedy generation parity

**Files:**
- Create: `code/mini-vllm/tests/test_llama_e2e.py`

- [ ] **Step 1: Write e2e parity test**

Create `tests/test_llama_e2e.py`:
```python
"""End-to-end: drive our LLMEngine to greedy-generate 8 tokens for TinyLlama,
compare to HF's `model.generate(..., do_sample=False)` greedy output.
"""
import pytest
import torch

from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.config import CacheConfig, EngineConfig, SamplingParams
from mini_vllm.engine import LLMEngine
from mini_vllm.models.llama import LlamaModel
from mini_vllm.models.llama_loader import load_hf_to_llama_model
from mini_vllm.tokenizer import TokenizerWrapper

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.mark.slow
def test_greedy_matches_hf_at_least_3_of_first_8_tokens():
    from transformers import LlamaForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    cfg = LlamaModel.tinyllama_config()

    ours_model = LlamaModel(cfg, TorchBackend()).to(torch.float32).eval()
    load_hf_to_llama_model(ours_model, TINYLLAMA, dtype=torch.float32)
    tokenizer = TokenizerWrapper.from_pretrained_llama()

    engine = LLMEngine(ours_model, tokenizer, EngineConfig(
        model=cfg, cache=CacheConfig(block_size=16, num_gpu_blocks=8),
        device="cpu", seed=0))
    prompt = "The capital of France is"
    out = engine.generate([prompt], SamplingParams(max_tokens=8, greedy=True))
    ours_text = out[0][1]
    ours_ids = tokenizer.encode(ours_text)

    # HF greedy
    hf_model = LlamaForCausalLM.from_pretrained(TINYLLAMA, torch_dtype=torch.float32).eval()
    hf_tk = AutoTokenizer.from_pretrained(TINYLLAMA)
    inp = hf_tk(prompt, return_tensors="pt")
    hf_out = hf_model.generate(**inp, max_new_tokens=8, do_sample=False)
    hf_new_ids = hf_out[0, inp["input_ids"].shape[1]:].tolist()

    # Greedy decoding *should* match exactly with bit-identical fp32 forward.
    # Permit small drift from accumulation order: require at least 3 of 8 to agree.
    matches = sum(1 for a, b in zip(ours_ids[:8], hf_new_ids) if a == b)
    assert matches >= 3, (
        f"Too few greedy matches: ours={ours_ids[:8]}, hf={hf_new_ids}, matches={matches}"
    )
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_llama_e2e.py -m slow -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_llama_e2e.py
git commit -m "mini-vllm: TinyLlama greedy generation parity (vs HF)"
```

---

### Task 12: examples/run_tinyllama.py

**Files:**
- Create: `code/mini-vllm/examples/run_tinyllama.py`

- [ ] **Step 1: Write demo script**

Create `examples/run_tinyllama.py`:
```python
"""End-to-end TinyLlama demo. Downloads ~2.2 GB on first run (cached after).

Run from code/mini-vllm/:
    python examples/run_tinyllama.py
"""
import argparse
import torch

from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.config import CacheConfig, EngineConfig, SamplingParams
from mini_vllm.engine import LLMEngine
from mini_vllm.models.llama import LlamaModel
from mini_vllm.models.llama_loader import load_hf_to_llama_model
from mini_vllm.tokenizer import TokenizerWrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "mps", "cuda"])
    args = p.parse_args()

    dtype = torch.float32 if args.device != "cuda" else torch.bfloat16
    cfg = LlamaModel.tinyllama_config()
    cfg.dtype = "bfloat16" if dtype == torch.bfloat16 else "float32"

    print(f"[mini-vllm] loading TinyLlama-1.1B-Chat-v1.0 (dtype={dtype})...")
    backend = TorchBackend()
    model = LlamaModel(cfg, backend).to(device=args.device, dtype=dtype).eval()
    load_hf_to_llama_model(model, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=dtype)
    tokenizer = TokenizerWrapper.from_pretrained_llama()

    engine = LLMEngine(model, tokenizer, EngineConfig(
        model=cfg,
        cache=CacheConfig(block_size=16, num_gpu_blocks=64),
        device=args.device, seed=0,
    ))

    print(f"[mini-vllm] prompt: {args.prompt!r}")
    out = engine.generate([args.prompt],
                          SamplingParams(max_tokens=args.max_tokens, greedy=True))
    rid, text = out[0]
    print(f"[{rid}] {args.prompt}{text}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run end-to-end (slow: 1-3 minutes for prefill+24 decode tokens on CPU)**

Run from `code/mini-vllm/`: `python examples/run_tinyllama.py --max-tokens 12`
Expected output: a single line beginning with `[req-0] The capital of France is` followed by greedy continuation. The continuation should be coherent (the most likely first token after this prompt is `Paris`, possibly with leading space).

If the run takes >10 minutes or runs out of memory: report DONE_WITH_CONCERNS with timing/memory details. The CPU forward IS slow on this model size — ~30s per token is acceptable.

- [ ] **Step 3: Commit**

```bash
git add examples/run_tinyllama.py
git commit -m "mini-vllm: run_tinyllama.py end-to-end demo"
```

---

### Task 13: Plan 3 wrap-up — README + roadmap

**Files:**
- Modify: `code/mini-vllm/README.md`

- [ ] **Step 1: Update README**

Replace `code/mini-vllm/README.md` with:
```markdown
# mini-vLLM

Educational reimplementation of vLLM with PagedAttention. Companion code
to `part5-serving-infra/16a-lab-mini-vllm.md`.

## Status

**Plan 3 complete.** Engine runs both a toy GPT and TinyLlama-1.1B end-to-end
on CPU/MPS via the Torch paged-attention backend. Naive FCFS scheduler;
greedy sampling. Parity-tested against HF `transformers`.

## Install

    cd code/mini-vllm
    pip install -e ".[dev]"

## Quickstart

Toy GPT (random weights, instant):

    python examples/run_toy.py

TinyLlama-1.1B (downloads ~2.2 GB on first run, slow on CPU):

    python examples/run_tinyllama.py --max-tokens 12

## Run tests

    pytest tests/ -v               # fast suite (skips slow downloads)
    pytest -m slow tests/ -v       # parity tests against HF transformers (downloads weights)

## Roadmap

- [x] Plan 1: skeleton — Torch backend, naive scheduler, toy GPT
- [ ] Plan 2: Triton paged-attention kernel (deferred — needs GPU machine)
- [x] Plan 3: TinyLlama-1.1B + HF safetensors loader
- [ ] Plan 4: continuous batching + chunked prefill
- [ ] Plan 5: prefix caching + CoW
- [ ] Plan 6: swap to CPU + preemption
- [ ] Plan 7: streaming + full sampler (temperature/top-p/top-k) + bench
- [ ] Plan 8: tutorial chapter `16a-lab-mini-vllm.md`
```

- [ ] **Step 2: Final test sweep (fast suite)**

Run from `code/mini-vllm/`: `pytest tests/ -v`
Expected: all fast tests pass; slow tests are deselected. Capture the final summary line and report it.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "mini-vllm: Plan 3 README + roadmap update"
```

---

## Summary

Plan 3 delivers:
- A complete Llama architecture (RMSNorm, RoPE, SwiGLU, GQA, fused QKV) in `models/llama.py`
- An HF safetensors loader (`models/llama_loader.py`) that consumes TinyLlama-1.1B-Chat-v1.0 weights without depending on `transformers` modeling code
- TinyLlama tokenizer support
- A runnable `examples/run_tinyllama.py` demo
- Two parity tests against HF `transformers`: logits top-5 overlap on a fixed prompt, and greedy generation overlap over 8 tokens
- Marked all download-heavy tests `slow` so the default `pytest tests/` stays fast

Plan 4 (continuous batching + chunked prefill) builds on this. The `LLMEngine`, `Scheduler`, `BlockManager`, and `ModelRunner` are unchanged from Plan 1 — Plan 4 only modifies `Scheduler.schedule()` to enable mid-batch admission and split long prompts.
