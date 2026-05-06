import torch
from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.config import CacheConfig, EngineConfig, SamplingParams
from mini_vllm.engine import LLMEngine
from mini_vllm.models.toy_gpt import ToyGPT
from mini_vllm.tokenizer import TokenizerWrapper


def _build_engine():
    backend = TorchBackend()
    model = ToyGPT.random_init(backend, vocab_size=50257, n_layer=2,
                               d_model=64, n_head=4, max_pos=128, seed=0)
    tokenizer = TokenizerWrapper.from_pretrained_gpt2()
    return LLMEngine(model, tokenizer, EngineConfig(
        model=model.config,
        cache=CacheConfig(block_size=8, num_gpu_blocks=32),
        device="cpu", seed=0))


def test_e2e_single_request():
    eng = _build_engine()
    out = eng.generate(["Hello"], SamplingParams(max_tokens=4, greedy=True))
    assert len(out) == 1 and len(out[0][1]) > 0
    # No block leak: all blocks back in free pool
    assert eng.block_manager.num_free_blocks == eng.cfg.cache.num_gpu_blocks


def test_e2e_two_sequential_batches():
    eng = _build_engine()
    eng.generate(["Hello", "World"], SamplingParams(max_tokens=4, greedy=True))
    eng.generate(["foo", "bar baz"],  SamplingParams(max_tokens=3, greedy=True))
    assert eng.block_manager.num_free_blocks == eng.cfg.cache.num_gpu_blocks


def test_e2e_determinism():
    eng1 = _build_engine()
    a = eng1.generate(["Hello there"], SamplingParams(max_tokens=8, greedy=True))
    eng2 = _build_engine()
    b = eng2.generate(["Hello there"], SamplingParams(max_tokens=8, greedy=True))
    assert a[0][1] == b[0][1]


def test_e2e_chunked_prefill_matches_unchunked():
    """Chunked prefill must produce identical greedy output to one-shot prefill.

    Use a long prompt and a small `chunked_prefill_size` to force multiple
    chunks. Run the same model twice (chunked vs unchunked); outputs must be
    bit-identical.
    """
    backend = TorchBackend()
    model = ToyGPT.random_init(backend, vocab_size=50257, n_layer=2,
                               d_model=64, n_head=4, max_pos=128, seed=0)
    tokenizer = TokenizerWrapper.from_pretrained_gpt2()
    # Prompt long enough that chunking actually triggers.
    prompts = ["Once upon a time in a faraway kingdom there lived a king who loved cake"]
    sp = SamplingParams(max_tokens=4, greedy=True)

    eng_unchunked = LLMEngine(model, tokenizer, EngineConfig(
        model=model.config, cache=CacheConfig(block_size=8, num_gpu_blocks=32),
        device="cpu", seed=0,
        enable_chunked_prefill=False))
    out_un = eng_unchunked.generate(prompts, sp)

    eng_chunked = LLMEngine(model, tokenizer, EngineConfig(
        model=model.config, cache=CacheConfig(block_size=8, num_gpu_blocks=32),
        device="cpu", seed=0,
        enable_chunked_prefill=True, chunked_prefill_size=4))   # tiny chunks
    out_ch = eng_chunked.generate(prompts, sp)

    assert out_un[0][1] == out_ch[0][1], (out_un[0][1], out_ch[0][1])
    assert eng_chunked.block_manager.num_free_blocks == eng_chunked.cfg.cache.num_gpu_blocks


def test_e2e_continuous_batching_vs_baseline_same_output():
    """Same prompts must produce identical greedy output regardless of whether
    continuous batching is on (Plan 4 default) or off (Plan 1 baseline).
    Continuous batching changes admission TIMING but not generated tokens."""
    backend = TorchBackend()
    model = ToyGPT.random_init(backend, vocab_size=50257, n_layer=2,
                               d_model=64, n_head=4, max_pos=128, seed=0)
    tokenizer = TokenizerWrapper.from_pretrained_gpt2()
    prompts = ["The cat", "The dog jumped over"]
    sp = SamplingParams(max_tokens=6, greedy=True)

    eng_on = LLMEngine(model, tokenizer, EngineConfig(
        model=model.config, cache=CacheConfig(block_size=8, num_gpu_blocks=32),
        device="cpu", seed=0, enable_continuous_batching=True))
    out_on = eng_on.generate(prompts, sp)

    eng_off = LLMEngine(model, tokenizer, EngineConfig(
        model=model.config, cache=CacheConfig(block_size=8, num_gpu_blocks=32),
        device="cpu", seed=0, enable_continuous_batching=False))
    out_off = eng_off.generate(prompts, sp)

    assert [t for _, t in out_on] == [t for _, t in out_off]
    # Both engines should release all blocks at the end.
    assert eng_on.block_manager.num_free_blocks == eng_on.cfg.cache.num_gpu_blocks
    assert eng_off.block_manager.num_free_blocks == eng_off.cfg.cache.num_gpu_blocks
