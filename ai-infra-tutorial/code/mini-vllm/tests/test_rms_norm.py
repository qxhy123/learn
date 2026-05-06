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
