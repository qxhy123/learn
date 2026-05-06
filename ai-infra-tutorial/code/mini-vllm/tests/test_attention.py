import torch
from mini_vllm.backends.torch_backend import TorchBackend


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
