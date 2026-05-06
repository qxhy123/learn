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
    """Mutates `llama_model` in place: loads weights from `model_id_or_path`."""
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
