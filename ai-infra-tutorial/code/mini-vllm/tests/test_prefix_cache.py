"""Tests for prefix caching: hash chain, ref-counted block reuse, no leak.

CoW (copy-on-write) plumbing is present in BlockManager.append_slot but
isn't exercised by Plan 5 generation paths (single-completion greedy never
writes into a shared block). A targeted unit test checks the CoW return
value for the pathological case.
"""
import torch
from mini_vllm.block_manager import BlockManager, AllocStatus
from mini_vllm.sequence import Sequence
from mini_vllm.config import SamplingParams, EngineConfig, CacheConfig
from mini_vllm.engine import LLMEngine
from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.models.toy_gpt import ToyGPT
from mini_vllm.tokenizer import TokenizerWrapper


def _seq(rid, prompt_ids):
    return Sequence(rid, prompt_token_ids=list(prompt_ids),
                    sampling_params=SamplingParams(max_tokens=4))


# ---------------------------------------------------------------------------
# BlockManager-level prefix caching mechanics
# ---------------------------------------------------------------------------

def test_prefix_cache_disabled_by_default_no_sharing():
    """With prefix_caching=False, two seqs with same prompt prefix do NOT share."""
    bm = BlockManager(num_blocks=8, block_size=4, enable_prefix_caching=False)
    s1 = _seq("a", list(range(8)))   # 2 full blocks
    s2 = _seq("b", list(range(8)))   # same prompt
    bm.allocate(s1)
    # No registration happens; no sharing on s2's allocation.
    assert s1.num_prefilled == 0
    bm.register_filled_blocks(s1)    # no-op when disabled
    bm.allocate(s2)
    assert s2.num_prefilled == 0     # no cache hit
    s1_blocks = {pb.block_id for pb in s1.block_table.physical_blocks}
    s2_blocks = {pb.block_id for pb in s2.block_table.physical_blocks}
    assert s1_blocks.isdisjoint(s2_blocks)


def test_prefix_cache_shares_full_blocks():
    bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
    # s1's prompt fills 2 blocks exactly. After admit + register, both blocks
    # are cached. Note: `_effective_cached_count` keeps at least one fresh block,
    # so when s1 has 8 tokens both blocks would be cached but we drop the last.
    # We work around by giving s1 a 9-token prompt: 2 full blocks + 1 trailing.
    s1 = _seq("a", list(range(9)))
    bm.allocate(s1)
    assert s1.num_prefilled == 0     # no prior registry
    # Pretend we ran prefill — register filled blocks.
    bm.register_filled_blocks(s1)
    # Now s2 with same 9-token prompt should hit BOTH full blocks.
    s2 = _seq("b", list(range(9)))
    bm.allocate(s2)
    assert s2.num_prefilled == 8     # 2 cached blocks * 4 tokens each
    # Verify shared block ids: s1's block[0,1] == s2's block[0,1]
    assert s1.block_table.physical_blocks[0].block_id == s2.block_table.physical_blocks[0].block_id
    assert s1.block_table.physical_blocks[1].block_id == s2.block_table.physical_blocks[1].block_id
    # Ref counts should be 2 on shared blocks
    assert s1.block_table.physical_blocks[0].ref_count == 2
    assert s1.block_table.physical_blocks[1].ref_count == 2
    # s2's third block is fresh (s1 also has a third, different one)
    assert (s2.block_table.physical_blocks[2].block_id
            != s1.block_table.physical_blocks[2].block_id)


def test_prefix_cache_no_leak_after_free():
    bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
    s1 = _seq("a", list(range(9)))
    bm.allocate(s1); bm.register_filled_blocks(s1)
    s2 = _seq("b", list(range(9)))
    bm.allocate(s2)
    # Free s1: shared blocks' ref_count drops 2→1, NOT freed.
    bm.free(s1)
    # Free s2: now ref drops 1→0, blocks freed.
    bm.free(s2)
    assert bm.num_free_blocks == 16


def test_prefix_cache_leaves_at_least_one_fresh_token():
    """If full prompt is cached (length == block-aligned), the last cached
    block must be dropped so the model has at least 1 token to forward."""
    bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
    s1 = _seq("a", list(range(8)))   # exactly 2 full blocks
    bm.allocate(s1)
    bm.register_filled_blocks(s1)
    # s2 with the same 8-token prompt would hit BOTH blocks; we drop the last.
    s2 = _seq("b", list(range(8)))
    bm.allocate(s2)
    assert s2.num_prefilled == 4     # only 1 cached block kept (4 tokens)
    # Last block of s2 is fresh, ref_count=1
    assert s2.block_table.physical_blocks[1].ref_count == 1


def test_cow_triggered_when_appending_into_shared_block():
    """Synthetic CoW exercise: manually craft a state where append_slot must
    return a CoW pair.

    Plan 5's normal flow never produces this, but the BlockManager should
    behave correctly if it ever arises (e.g. parallel sampling in a future
    plan)."""
    bm = BlockManager(num_blocks=8, block_size=4, enable_prefix_caching=True)
    s = _seq("a", list(range(4)))   # single block prompt
    bm.allocate(s)
    bm.register_filled_blocks(s)
    # Manually share s's last block (simulate a parallel-sampling fork)
    s.block_table.physical_blocks[-1].ref_count += 1
    # Now generate one token: append_slot should detect ref_count > 1 and CoW.
    s.append_token(99)
    cow = bm.append_slot(s)         # seq_len=5 → still in same block 0 (offset 4? no, 4 = next block)
    # Wait: block_size=4, seq_len=5 → needs 2 blocks. So append_slot extends.
    # For a true CoW into a shared block, we need to write within a shared
    # block (seq_len % block_size != 0 AND the LAST block in table is shared).
    # Reset and try again with a shorter sequence.
    bm.free(s)

    s = _seq("a", [1, 2])  # 2 tokens, single block (1 block of size 4, 2 slots used)
    bm.allocate(s)
    # Mark this block shared
    s.block_table.physical_blocks[-1].ref_count = 2
    s.append_token(99)
    cow = bm.append_slot(s)        # seq_len=3 still fits in same block — CoW fires
    assert cow is not None
    src, dst = cow
    assert src != dst
    # s now points at the new (dst) block
    assert s.block_table.physical_blocks[-1].block_id == dst


# ---------------------------------------------------------------------------
# End-to-end: prefix cache reduces num_prefilled across sequential requests
# ---------------------------------------------------------------------------

def test_e2e_prefix_cache_skips_compute_on_repeat_prompt():
    """Run the same prompt twice. The second run should report a non-zero
    `num_prefilled` immediately after admission (cached prefix)."""
    backend = TorchBackend()
    model = ToyGPT.random_init(backend, vocab_size=50257, n_layer=2,
                               d_model=64, n_head=4, max_pos=128, seed=0)
    tokenizer = TokenizerWrapper.from_pretrained_gpt2()
    eng = LLMEngine(model, tokenizer, EngineConfig(
        model=model.config, cache=CacheConfig(block_size=4, num_gpu_blocks=64),
        device="cpu", seed=0,
        enable_prefix_caching=True))
    # Prompt with enough tokens to fill at least 2 blocks (need >= 9 tokens
    # for "leave 1 fresh" rule to leave at least 1 cached block on second run).
    long_prompt = "the quick brown fox jumps over the lazy dog and runs away fast"
    sp = SamplingParams(max_tokens=2, greedy=True)

    # First run: cache cold, no hit.
    eng.generate([long_prompt], sp)
    # Second run: same prompt, same engine — should hit prefix cache.
    rid = eng.add_request(long_prompt, sp)
    sched_out = eng.scheduler.schedule()
    admitted = next(s for s in sched_out.prefill_seqs if s.request_id == rid)
    # Cached prefix should cover at least one full block (block_size=4 tokens).
    assert admitted.num_prefilled >= 4, (
        f"expected prefix cache hit, got num_prefilled={admitted.num_prefilled}")
    # Drain
    while eng.scheduler.has_unfinished():
        eng.step()
    assert eng.block_manager.num_free_blocks == eng.cfg.cache.num_gpu_blocks
