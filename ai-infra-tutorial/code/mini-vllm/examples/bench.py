"""Plan 7 benchmark: 5-config throughput comparison on toy GPT.

Configurations are stacked features (each adds to the previous):
  1. naive             — all advanced flags off (Plan 1 baseline)
  2. + continuous batching
  3. + chunked prefill
  4. + prefix caching
  5. + swap (under-sized GPU pool)

Reports per-config: total wall time, throughput (tok/s), TTFT median.

Note: this runs on toy GPT with random weights — output tokens are gibberish.
The metric is the engine's behaviour under flag combinations, not generation
quality. For TinyLlama benchmarks, run `examples/run_tinyllama.py` and time
externally.

Usage:
    python examples/bench.py
    python examples/bench.py --num-prompts 32 --max-tokens 32
"""
from __future__ import annotations
import argparse
import statistics
import time
from dataclasses import dataclass
from typing import List

from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.config import CacheConfig, EngineConfig, SamplingParams
from mini_vllm.engine import LLMEngine
from mini_vllm.models.toy_gpt import ToyGPT
from mini_vllm.tokenizer import TokenizerWrapper


# Five canonical prompt patterns; we cycle through them to fill `num_prompts`.
# A long shared prefix lets prefix-caching shine when the same prompt repeats.
PROMPTS = [
    "Translate to French: The quick brown fox jumps over the lazy dog.",
    "Translate to French: A journey of a thousand miles begins with",
    "Translate to French: To be or not to be, that is the question",
    "Summarize the following text in one sentence: The history of computing",
    "Continue the story: Once upon a time in a faraway kingdom there",
]


@dataclass
class Config:
    name: str
    engine: EngineConfig


def _build_configs(model_cfg) -> List[Config]:
    base = dict(model=model_cfg, device="cpu", seed=0,
                cache=CacheConfig(block_size=8, num_gpu_blocks=64, num_cpu_blocks=0))
    # Each config flips one more flag.
    configs = []
    cfg = EngineConfig(
        **base,
        enable_continuous_batching=False,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        enable_swap=False,
    )
    configs.append(Config("naive (all off)", cfg))

    cfg = EngineConfig(
        **base,
        enable_continuous_batching=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        enable_swap=False,
    )
    configs.append(Config("+ continuous batching", cfg))

    cfg = EngineConfig(
        **base,
        enable_continuous_batching=True,
        enable_chunked_prefill=True, chunked_prefill_size=8,
        enable_prefix_caching=False,
        enable_swap=False,
    )
    configs.append(Config("+ chunked prefill", cfg))

    cfg = EngineConfig(
        **base,
        enable_continuous_batching=True,
        enable_chunked_prefill=True, chunked_prefill_size=8,
        enable_prefix_caching=True,
        enable_swap=False,
    )
    configs.append(Config("+ prefix caching", cfg))

    # For the swap config, shrink the GPU pool and add a CPU pool.
    swap_base = dict(base)
    swap_base["cache"] = CacheConfig(block_size=8, num_gpu_blocks=16, num_cpu_blocks=64)
    cfg = EngineConfig(
        **swap_base,
        enable_continuous_batching=True,
        enable_chunked_prefill=True, chunked_prefill_size=8,
        enable_prefix_caching=True,
        enable_swap=True,
    )
    configs.append(Config("+ swap (cramped GPU)", cfg))
    return configs


def _run_one(cfg: Config, prompts: List[str], max_tokens: int) -> dict:
    """Run prompts through one engine config; return throughput + TTFT."""
    backend = TorchBackend()
    model = ToyGPT.random_init(backend, vocab_size=50257, n_layer=2,
                               d_model=64, n_head=4, max_pos=256, seed=0)
    tokenizer = TokenizerWrapper.from_pretrained_gpt2()
    eng = LLMEngine(model, tokenizer, cfg.engine)
    sp = SamplingParams(max_tokens=max_tokens, greedy=True)

    # Submit everything up front, then drive step() while measuring.
    rids = [eng.add_request(p, sp) for p in prompts]
    first_token_time = {rid: None for rid in rids}
    submit_time = time.perf_counter()
    total_tokens = 0
    while eng.scheduler.has_unfinished():
        for so in eng.step():
            now = time.perf_counter()
            if first_token_time[so.request_id] is None:
                first_token_time[so.request_id] = now - submit_time
            total_tokens += 1
    end = time.perf_counter()
    wall = end - submit_time
    ttfts = sorted(t for t in first_token_time.values() if t is not None)
    return {
        "wall_s": wall,
        "throughput_tok_s": total_tokens / wall if wall > 0 else 0.0,
        "ttft_p50_s": ttfts[len(ttfts) // 2] if ttfts else 0.0,
        "ttft_p99_s": ttfts[max(0, len(ttfts) * 99 // 100 - 1)] if ttfts else 0.0,
        "tokens": total_tokens,
    }


def _print_table(rows):
    headers = ["config", "wall (s)", "tok/s", "TTFT p50", "TTFT p99", "tokens"]
    widths = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep  = "-+-".join("-" * w for w in widths)
    print(line); print(sep)
    for r in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args()

    # Build prompts: cycle PROMPTS to reach num_prompts. Repeats give prefix-cache
    # something to share.
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_prompts)]

    # Sniff a model_config from a throwaway construction so we can build the
    # configs without running the model yet.
    backend = TorchBackend()
    sniff_model = ToyGPT.random_init(backend, vocab_size=50257, n_layer=2,
                                     d_model=64, n_head=4, max_pos=256, seed=0)
    configs = _build_configs(sniff_model.config)

    rows = []
    print(f"\nbench: {args.num_prompts} prompts × {args.max_tokens} tokens, toy GPT (n_layer=2, d=64)")
    print()
    for c in configs:
        r = _run_one(c, prompts, args.max_tokens)
        rows.append([c.name,
                     f"{r['wall_s']:.2f}",
                     f"{r['throughput_tok_s']:.1f}",
                     f"{r['ttft_p50_s']:.3f}",
                     f"{r['ttft_p99_s']:.3f}",
                     r['tokens']])
    _print_table(rows)


if __name__ == "__main__":
    main()
