# mini-vLLM

Educational reimplementation of vLLM with PagedAttention. Companion code
to `part5-serving-infra/16a-lab-mini-vllm.md`.

## Status

**Plan 7 complete.** Engine supports continuous batching, chunked prefill,
prefix caching with CoW, swap-to-CPU, full sampling (temperature / top-p /
top-k / per-request seed), and token-by-token streaming. TinyLlama-1.1B
greedy parity vs HF `transformers` is preserved (8/8 token match).

Each feature is a flag on `EngineConfig`:
  `enable_continuous_batching`, `enable_chunked_prefill`,
  `enable_prefix_caching`, `enable_swap`.
Set any to `False` for benchmark comparison against the prior baseline.
For swap to engage, set `CacheConfig(num_cpu_blocks=...)` to a non-zero size.

## Streaming

    for rid, tok, done, partial in engine.generate_stream(prompt, params):
        print(partial, end="\r")

## Sampling

    SamplingParams(greedy=False, temperature=0.8, top_p=0.95, top_k=50, seed=42)

## Benchmark

    python examples/bench.py --num-prompts 16 --max-tokens 32

Compares 5 stacked configs (naive → +continuous batching → +chunked prefill
→ +prefix caching → +swap), reporting throughput and TTFT per config.

## Install

    cd code/mini-vllm
    pip install -e ".[dev]"

## Quickstart

Toy GPT (random weights, instant):

    python examples/run_toy.py

TinyLlama-1.1B (downloads ~2.2 GB on first run, slow on CPU):

    python examples/run_tinyllama.py --max-tokens 12

## Run tests

    pytest tests/ -v               # fast suite (skips slow downloads)
    pytest -m slow tests/ -v       # parity tests against HF transformers (downloads weights)

## Roadmap

- [x] Plan 1: skeleton — Torch backend, naive scheduler, toy GPT
- [ ] Plan 2: Triton paged-attention kernel (deferred — needs GPU machine)
- [x] Plan 3: TinyLlama-1.1B + HF safetensors loader
- [x] Plan 4: continuous batching (chunked prefill rolled into Plan 5 — shared kernel)
- [x] Plan 5: prefix caching + CoW + chunked prefill
- [x] Plan 6: swap to CPU + LRU preemption
- [x] Plan 7: streaming + full sampler (temperature/top-p/top-k) + bench
- [x] Plan 8: tutorial chapter `part5-serving-infra/16a-lab-mini-vllm.md`
