"""End-to-end smoke run with a randomly-initialized toy GPT.
Output text is gibberish (random weights) but proves the engine plumbing works.
"""
import torch
from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.cache_engine import CacheEngine  # noqa: F401  (used via engine)
from mini_vllm.config import CacheConfig, EngineConfig, ModelConfig, SamplingParams
from mini_vllm.engine import LLMEngine
from mini_vllm.models.toy_gpt import ToyGPT
from mini_vllm.tokenizer import TokenizerWrapper


def main():
    backend = TorchBackend()
    model = ToyGPT.random_init(backend, vocab_size=50257, n_layer=4,
                               d_model=128, n_head=4, max_pos=512, seed=42)
    tokenizer = TokenizerWrapper.from_pretrained_gpt2()
    engine = LLMEngine(
        model, tokenizer,
        EngineConfig(
            model=model.config,
            cache=CacheConfig(block_size=16, num_gpu_blocks=64),
            device="cpu", seed=42,
        ),
    )
    prompts = ["Hello world,", "Once upon a time"]
    sp = SamplingParams(max_tokens=16, greedy=True)
    for rid, text in engine.generate(prompts, sp):
        print(f"[{rid}] {text!r}")


if __name__ == "__main__":
    main()
