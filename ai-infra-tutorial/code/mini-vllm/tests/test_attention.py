import torch
from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.backends.reference import reference_decode


def test_reshape_and_cache_writes_correct_slots():
    block_size = 4
    num_blocks = 4
    H_kv = 2
    D = 8
    kc = torch.zeros(num_blocks, H_kv, D, block_size)
    vc = torch.zeros(num_blocks, H_kv, D, block_size)
    # Two tokens: token 0 -> slot 5 (block 1, offset 1); token 1 -> slot 11 (block 2, offset 3)
    key = torch.randn(2, H_kv, D)
    val = torch.randn(2, H_kv, D)
    slot_mapping = torch.tensor([5, 11], dtype=torch.long)
    backend = TorchBackend()
    backend.reshape_and_cache(key, val, kc, vc, slot_mapping)
    # block 1 offset 1
    assert torch.allclose(kc[1, :, :, 1], key[0])
    assert torch.allclose(vc[1, :, :, 1], val[0])
    # block 2 offset 3
    assert torch.allclose(kc[2, :, :, 3], key[1])
    assert torch.allclose(vc[2, :, :, 3], val[1])
    # Other slots untouched
    assert (kc[0] == 0).all()
    assert (kc[3] == 0).all()


def test_torch_decode_matches_reference():
    torch.manual_seed(0)
    B, H, D, H_kv = 3, 8, 16, 2  # GQA: 4 query heads per kv head
    block_size = 4
    num_blocks = 16
    max_blocks_per_seq = 4
    kc = torch.randn(num_blocks, H_kv, D, block_size)
    vc = torch.randn(num_blocks, H_kv, D, block_size)
    q = torch.randn(B, H, D)
    block_table = torch.tensor([
        [0, 1, 2, 3],
        [4, 5, 6, 0],
        [7, 8, 0, 0],
    ], dtype=torch.int32)
    context_lens = torch.tensor([13, 10, 6], dtype=torch.int32)
    scale = D ** -0.5
    backend = TorchBackend()
    out = backend.decode(q, kc, vc, block_table, context_lens, scale)
    ref = reference_decode(q, kc, vc, block_table, context_lens, scale)
    assert torch.allclose(out, ref, atol=1e-5)
