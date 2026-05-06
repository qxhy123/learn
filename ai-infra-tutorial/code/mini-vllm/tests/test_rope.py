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
