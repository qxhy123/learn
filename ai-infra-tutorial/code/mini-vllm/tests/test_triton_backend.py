"""Triton backend correctness tests.

Gated: skipped unless both `triton` is importable AND CUDA is available.
On a CUDA + Triton box, these run the same workloads through both Torch
and Triton backends and require fp16 outputs to match within
`atol=1e-2, rtol=1e-2`.

NOTE: These tests have not been runtime-verified at write time (the dev
machine has no CUDA / Triton). When running on a real GPU box for the
first time, expect 1-2 minor fixes — typically a stride or mask off-by-one.
The reference (`backends/reference.py`) is the source of truth.
"""
import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available; Triton backend tests skipped",
                allow_module_level=True)

from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.backends.triton_backend import TritonBackend


# ---------------------------------------------------------------------------
# reshape_and_cache
# ---------------------------------------------------------------------------

def test_triton_reshape_and_cache_matches_torch():
    torch.manual_seed(0)
    block_size, num_blocks, H_kv, D = 16, 8, 4, 64
    N = 30
    key = torch.randn(N, H_kv, D, device="cuda", dtype=torch.float16)
    val = torch.randn(N, H_kv, D, device="cuda", dtype=torch.float16)
    slot_mapping = torch.randint(0, num_blocks * block_size, (N,),
                                 device="cuda", dtype=torch.long)

    kc_t = torch.zeros(num_blocks, H_kv, D, block_size, device="cuda", dtype=torch.float16)
    vc_t = torch.zeros_like(kc_t)
    TorchBackend().reshape_and_cache(key, val, kc_t, vc_t, slot_mapping)

    kc_tr = torch.zeros_like(kc_t)
    vc_tr = torch.zeros_like(vc_t)
    TritonBackend().reshape_and_cache(key, val, kc_tr, vc_tr, slot_mapping)

    assert torch.allclose(kc_t, kc_tr, atol=1e-3)
    assert torch.allclose(vc_t, vc_tr, atol=1e-3)


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H,H_kv", [(8, 8), (8, 2), (32, 4)])
def test_triton_decode_matches_torch(H, H_kv):
    torch.manual_seed(0)
    block_size = 16
    num_blocks = 32
    D = 64
    B = 4
    max_blocks_per_seq = 4
    kc = torch.randn(num_blocks, H_kv, D, block_size, device="cuda", dtype=torch.float16)
    vc = torch.randn(num_blocks, H_kv, D, block_size, device="cuda", dtype=torch.float16)
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    block_table = torch.randint(0, num_blocks, (B, max_blocks_per_seq),
                                device="cuda", dtype=torch.int32)
    context_lens = torch.tensor([45, 30, 60, 16], device="cuda", dtype=torch.int32)
    scale = D ** -0.5

    out_t = TorchBackend().decode(q, kc, vc, block_table, context_lens, scale)
    out_tr = TritonBackend().decode(q, kc, vc, block_table, context_lens, scale)
    assert torch.allclose(out_t, out_tr, atol=1e-2, rtol=1e-2), (
        f"Triton decode diverged: max diff {(out_t - out_tr).abs().max()}")


# ---------------------------------------------------------------------------
# prefill (one-shot, no cached prefix)
# ---------------------------------------------------------------------------

def test_triton_prefill_one_shot_matches_torch():
    """query_lens == seq_lens: standard causal prefill."""
    torch.manual_seed(0)
    block_size, num_blocks, H, H_kv, D = 16, 16, 8, 2, 64
    seq_lens = torch.tensor([16, 12], device="cuda", dtype=torch.int32)
    query_lens = seq_lens.clone()
    N = int(seq_lens.sum().item())

    # Pre-fill the cache with the seqs' K/V so prefill can read it back.
    k = torch.randn(N, H_kv, D, device="cuda", dtype=torch.float16)
    v = torch.randn(N, H_kv, D, device="cuda", dtype=torch.float16)
    kc = torch.zeros(num_blocks, H_kv, D, block_size, device="cuda", dtype=torch.float16)
    vc = torch.zeros_like(kc)
    block_table = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    # Manually write K/V into cache slots [0..15] (block 0,1) and [0..11] (block 2,3 partial)
    for b, sl in enumerate([16, 12]):
        cursor = sum(seq_lens[:b].tolist())
        for i in range(sl):
            blk = block_table[b, i // block_size].item()
            off = i % block_size
            kc[blk, :, :, off] = k[cursor + i].permute(1, 0).T  # [H_kv, D] -> store as [H_kv, D]
            vc[blk, :, :, off] = v[cursor + i].permute(1, 0).T

    q = torch.randn(N, H, D, device="cuda", dtype=torch.float16)
    scale = D ** -0.5
    out_t = TorchBackend().prefill(q, kc, vc, block_table, seq_lens, query_lens, scale)
    out_tr = TritonBackend().prefill(q, kc, vc, block_table, seq_lens, query_lens, scale)
    assert torch.allclose(out_t, out_tr, atol=1e-2, rtol=1e-2), (
        f"Triton prefill diverged: max diff {(out_t - out_tr).abs().max()}")


def test_triton_prefill_chunked_matches_torch():
    """seq_lens > query_lens: chunked prefill with cached prefix."""
    torch.manual_seed(0)
    block_size, num_blocks, H, H_kv, D = 16, 16, 8, 2, 64
    # Two seqs: seq 0 has 8 tokens cached + 5 new; seq 1 has 0 cached + 6 new
    seq_lens = torch.tensor([13, 6], device="cuda", dtype=torch.int32)
    query_lens = torch.tensor([5, 6], device="cuda", dtype=torch.int32)
    N = 11

    kc = torch.randn(num_blocks, H_kv, D, block_size, device="cuda", dtype=torch.float16)
    vc = torch.randn(num_blocks, H_kv, D, block_size, device="cuda", dtype=torch.float16)
    block_table = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    q = torch.randn(N, H, D, device="cuda", dtype=torch.float16)
    scale = D ** -0.5

    out_t = TorchBackend().prefill(q, kc, vc, block_table, seq_lens, query_lens, scale)
    out_tr = TritonBackend().prefill(q, kc, vc, block_table, seq_lens, query_lens, scale)
    assert torch.allclose(out_t, out_tr, atol=1e-2, rtol=1e-2), (
        f"Triton chunked prefill diverged: max diff {(out_t - out_tr).abs().max()}")
