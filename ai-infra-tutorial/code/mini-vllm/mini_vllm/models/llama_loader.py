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
