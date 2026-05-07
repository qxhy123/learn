# mini-vLLM

Educational reimplementation of vLLM with PagedAttention. Companion code
to `part5-serving-infra/16a-lab-mini-vllm.md`.

This is a correctness-first reference for learning scheduler, block table,
prefix cache, swap, streaming, and sampler semantics. It is intentionally not a
realistic performance path and does not represent vLLM's production kernels,
async engine, CUDA graph capture, distributed workers, or memory layout.

## Status

**Plan 7 complete.** Engine supports continuous batching, chunked prefill,
prefix caching with CoW, swap-to-CPU, full sampling (temperature / top-p /
top-k / per-request seed), and token-by-token streaming. TinyLlama-1.1B
HF `transformers` coverage is kept as slow smoke/parity checks: top-k logits
overlap plus a short greedy generation token-match check.

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

## Plan 2 caveat (Triton backend)

`mini_vllm/backends/triton_backend.py` contains three Triton kernels
(reshape_and_cache, paged decode, ragged prefill) implementing the same
`AttentionBackend` interface as the Torch reference. **The kernels were
written without a CUDA + Triton machine to test on**: math is mirrored
from `backends/reference.py`, but expect 1-2 small fixes (typical: stride
bug, mask off-by-one, fp16 numerics) on first real-GPU run. The validation
contract is:

    pytest tests/test_triton_backend.py -v   # gated; skips without CUDA + Triton

Outputs must `allclose` the Torch reference at `atol=1e-2, rtol=1e-2`
(fp16). The `make_backend(device)` factory in `mini_vllm/backends/__init__.py`
defaults to the Torch/reference backend, including on CUDA. Triton remains an
experimental path and is not selected just because CUDA + Triton import
successfully; opt in explicitly after CUDA runtime validation:

    MINI_VLLM_BACKEND=triton python examples/run_toy.py

## Roadmap

- [x] Plan 1: skeleton — Torch backend, naive scheduler, toy GPT
- [⚠️] Plan 2: Triton paged-attention kernel (code written but not runtime-verified — see caveat below)
- [x] Plan 3: TinyLlama-1.1B + HF safetensors loader
- [x] Plan 4: continuous batching (chunked prefill rolled into Plan 5 — shared kernel)
- [x] Plan 5: prefix caching + CoW + chunked prefill
- [x] Plan 6: swap to CPU + LRU preemption
- [x] Plan 7: streaming + full sampler (temperature/top-p/top-k) + bench
- [x] Plan 8: tutorial chapter `part5-serving-infra/16a-lab-mini-vllm.md`
