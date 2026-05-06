"""End-to-end TinyLlama demo. Downloads ~2.2 GB on first run (cached after).

Run from code/mini-vllm/:
    python examples/run_tinyllama.py
"""
import argparse
import torch

from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.config import CacheConfig, EngineConfig, SamplingParams
from mini_vllm.engine import LLMEngine
from mini_vllm.models.llama import LlamaModel
from mini_vllm.models.llama_loader import load_hf_to_llama_model
from mini_vllm.tokenizer import TokenizerWrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "mps", "cuda"])
    args = p.parse_args()

    dtype = torch.float32 if args.device != "cuda" else torch.bfloat16
    cfg = LlamaModel.tinyllama_config()
    cfg.dtype = "bfloat16" if dtype == torch.bfloat16 else "float32"

    print(f"[mini-vllm] loading TinyLlama-1.1B-Chat-v1.0 (dtype={dtype})...")
    backend = TorchBackend()
    model = LlamaModel(cfg, backend).to(device=args.device, dtype=dtype).eval()
    load_hf_to_llama_model(model, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=dtype)
    tokenizer = TokenizerWrapper.from_pretrained_llama()

    engine = LLMEngine(model, tokenizer, EngineConfig(
        model=cfg,
        cache=CacheConfig(block_size=16, num_gpu_blocks=64),
        device=args.device, seed=0,
    ))

    print(f"[mini-vllm] prompt: {args.prompt!r}")
    out = engine.generate([args.prompt],
                          SamplingParams(max_tokens=args.max_tokens, greedy=True))
    rid, text = out[0]
    print(f"[{rid}] {args.prompt}{text}")


if __name__ == "__main__":
    main()
