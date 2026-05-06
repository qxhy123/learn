import torch
from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.models.toy_gpt import ToyGPT
from mini_vllm.cache_engine import CacheEngine
from mini_vllm.config import CacheConfig


def test_toy_gpt_prefill_only_forward():
    torch.manual_seed(0)
    backend = TorchBackend()
    model = ToyGPT.random_init(backend, vocab_size=128, n_layer=2,
                               d_model=32, n_head=4, max_pos=64)
    ce = CacheEngine(model.config, CacheConfig(block_size=4, num_gpu_blocks=8),
                     device='cpu', dtype=torch.float32)

    # One sequence, prefill of length 5
    N = 5
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    positions = torch.arange(N)
    # All five tokens go to the first 5 slots of block 0
    slot_mapping = torch.arange(N, dtype=torch.long)
    sample_indices = torch.tensor([N - 1])  # only sample the last position

    # 5 tokens fit in 2 blocks of size 4: block 0 (offsets 0..3), block 1 (offset 0).
    # We allocate blocks 0 and 1 manually for this single-seq test.
    prefill_block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    logits = model(
        input_ids, positions, slot_mapping, ce.kv_caches,
        prefill_block_table=prefill_block_table,
        prefill_seq_lens=torch.tensor([N]),
        prefill_query_lens=torch.tensor([N]),
        num_prefill_tokens=N,
        decode_block_table=torch.empty(0, 0, dtype=torch.int32),
        decode_context_lens=torch.empty(0, dtype=torch.int32),
        sample_indices=sample_indices,
    )
    assert logits.shape == (1, 128)
    # KV cache should be populated at slots 0..4
    assert (ce.kv_caches[0][0][0, :, :, :5] != 0).any()
    assert (ce.kv_caches[0][0][0, :, :, 5:] == 0).all()
