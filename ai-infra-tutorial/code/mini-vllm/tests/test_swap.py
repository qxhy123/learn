"""Plan 6 tests: swap-to-CPU.

Three layers covered:
  1. CacheEngine: tensor data round-trips through CPU pool intact.
  2. BlockManager: swap_out/in correctly move blocks between pool namespaces
     and preserve no-leak invariant.
  3. End-to-end: an engine with deliberately undersized GPU pool but generous
     CPU pool generates correctly via swap.
"""
import torch
from mini_vllm.cache_engine import CacheEngine
from mini_vllm.config import (
    CacheConfig, EngineConfig, ModelConfig, SamplingParams,
)
from mini_vllm.block_manager import BlockManager
from mini_vllm.sequence import Sequence, SequenceStatus
from mini_vllm.engine import LLMEngine
from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.models.toy_gpt import ToyGPT
from mini_vllm.tokenizer import TokenizerWrapper


# ---------------------------------------------------------------------------
# CacheEngine tensor swap round-trip
# ---------------------------------------------------------------------------

def test_cache_engine_swap_roundtrip_preserves_kv():
    mc = ModelConfig(model_type="toy_gpt", vocab_size=32, hidden_size=16,
                     num_hidden_layers=2, num_attention_heads=2, num_kv_heads=2,
                     head_dim=8, max_position_embeddings=32, intermediate_size=32)
    cc = CacheConfig(block_size=4, num_gpu_blocks=4, num_cpu_blocks=4)
    ce = CacheEngine(mc, cc, device="cpu", dtype=torch.float32)

    # Seed GPU block 1 with a distinctive pattern.
    torch.manual_seed(0)
    pattern_k = torch.randn(2, 8, 4)   # [H_kv, D, block_size]
    pattern_v = torch.randn(2, 8, 4)
    for layer in range(ce.num_layers):
        ce.kv_caches[layer][0][1].copy_(pattern_k)
        ce.kv_caches[layer][1][1].copy_(pattern_v)

    # Swap GPU block 1 → CPU block 2.
    ce.swap_out({1: 2})
    # GPU block 1 still has the data (we didn't zero it on the GPU side; the
    # caller is BlockManager, which decides allocation policy). The CPU side
    # must have a faithful copy.
    for layer in range(ce.num_layers):
        assert torch.allclose(ce.cpu_kv_caches[layer][0][2], pattern_k)
        assert torch.allclose(ce.cpu_kv_caches[layer][1][2], pattern_v)

    # Now zero GPU block 1 and swap_in CPU block 2 → GPU block 0 (different id).
    for layer in range(ce.num_layers):
        ce.kv_caches[layer][0][1].zero_()
        ce.kv_caches[layer][1][1].zero_()
        ce.kv_caches[layer][0][0].zero_()
        ce.kv_caches[layer][1][0].zero_()
    ce.swap_in({2: 0})
    for layer in range(ce.num_layers):
        assert torch.allclose(ce.kv_caches[layer][0][0], pattern_k)
        assert torch.allclose(ce.kv_caches[layer][1][0], pattern_v)


# ---------------------------------------------------------------------------
# BlockManager swap mechanics
# ---------------------------------------------------------------------------

def _seq(rid, prompt_len):
    return Sequence(rid, prompt_token_ids=list(range(prompt_len)),
                    sampling_params=SamplingParams(max_tokens=4))


def test_block_manager_swap_out_in_roundtrip():
    bm = BlockManager(num_blocks=4, block_size=4, num_cpu_blocks=4)
    s = _seq("a", 9)        # 3 blocks (8/4 = 2 full, 1 partial)
    bm.allocate(s)
    gpu_ids_before = [pb.block_id for pb in s.block_table.physical_blocks]
    assert all(pb.device == "gpu" for pb in s.block_table.physical_blocks)

    # Swap out
    out_map = bm.swap_out(s)
    assert set(out_map.keys()) == set(gpu_ids_before)
    cpu_ids = list(out_map.values())
    assert all(pb.device == "cpu" for pb in s.block_table.physical_blocks)
    assert bm.num_free_blocks == 4              # all GPU released
    assert bm.num_free_cpu_blocks == 4 - 3      # 3 CPU consumed

    # Swap in (may land on different GPU ids)
    in_map = bm.swap_in(s)
    assert set(in_map.keys()) == set(cpu_ids)
    assert all(pb.device == "gpu" for pb in s.block_table.physical_blocks)
    assert bm.num_free_blocks == 4 - 3
    assert bm.num_free_cpu_blocks == 4          # all CPU released

    # Free wraps up cleanly
    bm.free(s)
    assert bm.num_free_blocks == 4
    assert bm.num_free_cpu_blocks == 4


def test_can_swap_out_rejects_shared_blocks():
    bm = BlockManager(num_blocks=4, block_size=4, num_cpu_blocks=4,
                      enable_prefix_caching=True)
    s1 = _seq("a", 9)
    bm.allocate(s1); bm.register_filled_blocks(s1)
    s2 = _seq("b", 9)        # shares 2 blocks with s1
    bm.allocate(s2)
    # s1's block 0,1 are shared (ref_count=2). swap_out must refuse.
    assert not bm.can_swap_out(s1)
    assert not bm.can_swap_out(s2)


def test_can_swap_out_rejects_when_cpu_pool_full():
    bm = BlockManager(num_blocks=4, block_size=4, num_cpu_blocks=2)
    s = _seq("a", 9)         # needs 3 blocks but CPU only has 2
    bm.allocate(s)
    assert not bm.can_swap_out(s)


# ---------------------------------------------------------------------------
# End-to-end: under-sized GPU pool with CPU swap produces correct output
# ---------------------------------------------------------------------------

def _build_engine(num_gpu_blocks: int, num_cpu_blocks: int, *,
                  enable_swap: bool = True):
    backend = TorchBackend()
    model = ToyGPT.random_init(backend, vocab_size=50257, n_layer=2,
                               d_model=64, n_head=4, max_pos=128, seed=0)
    tokenizer = TokenizerWrapper.from_pretrained_gpt2()
    return LLMEngine(model, tokenizer, EngineConfig(
        model=model.config,
        cache=CacheConfig(block_size=4, num_gpu_blocks=num_gpu_blocks,
                          num_cpu_blocks=num_cpu_blocks),
        device="cpu", seed=0,
        # Disable prefix caching here: we want each seq to use private blocks
        # so swap is actually eligible (Plan 6 rejects swap of shared blocks).
        enable_prefix_caching=False,
        enable_swap=enable_swap,
    ))


def test_e2e_swap_matches_no_swap_baseline():
    """Same engine code with a generous GPU pool (no swap needed) and with a
    cramped GPU pool (swap fires) must produce identical greedy output for
    the same prompts.

    GPU pool size of 6 blocks (24 slots) is enough for one full generation
    cycle but two concurrent prompts of 9 tokens + decoding will eventually
    force a swap. CPU pool of 16 blocks is plenty.
    """
    prompts = [
        "The quick brown fox",
        "Once upon a midnight",
        "Far away in a land",
    ]
    sp = SamplingParams(max_tokens=6, greedy=True)

    # Baseline: large GPU pool, swap unused.
    eng_big = _build_engine(num_gpu_blocks=64, num_cpu_blocks=0,
                            enable_swap=False)
    out_big = eng_big.generate(prompts, sp)

    # Swap path: cramped GPU pool, generous CPU pool.
    eng_small = _build_engine(num_gpu_blocks=8, num_cpu_blocks=32,
                              enable_swap=True)
    out_small = eng_small.generate(prompts, sp)

    assert [t for _, t in out_big] == [t for _, t in out_small], (
        f"swap-mode output diverged from baseline:\n"
        f"baseline={[t for _, t in out_big]}\n"
        f"swapped ={[t for _, t in out_small]}")
    # Both engines should release all blocks at the end.
    assert eng_big.block_manager.num_free_blocks == 64
    assert eng_small.block_manager.num_free_blocks == 8
    assert eng_small.block_manager.num_free_cpu_blocks == 32
