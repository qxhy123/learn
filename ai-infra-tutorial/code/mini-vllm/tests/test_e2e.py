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
