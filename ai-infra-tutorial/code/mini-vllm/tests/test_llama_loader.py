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
    in_dim = 16
    q = torch.arange(H * D * in_dim, dtype=torch.float32).reshape(H * D, in_dim)
    k = torch.arange(8 * D * in_dim, dtype=torch.float32).reshape(8 * D, in_dim) + 1000
    v = torch.arange(8 * D * in_dim, dtype=torch.float32).reshape(8 * D, in_dim) + 2000
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


from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.models.llama import LlamaModel
from mini_vllm.models.llama_loader import load_hf_to_llama_model

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.mark.slow
def test_load_tinyllama_weights():
    cfg = LlamaModel.tinyllama_config()
    model = LlamaModel(cfg, TorchBackend()).to(torch.float32)
    load_hf_to_llama_model(model, TINYLLAMA, dtype=torch.float32)
    # Sanity: embed_tokens should now be non-zero (init was random,
    # but loaded values overwrite them).
    assert model.embed_tokens.weight.abs().sum() > 0
    # Spot-check: layer 0 q (first slice of qkv_proj) shouldn't equal layer 21's.
    qkv0 = model.layers[0].self_attn.qkv_proj.weight
    qkv21 = model.layers[21].self_attn.qkv_proj.weight
    assert not torch.allclose(qkv0, qkv21)
