# mini-vLLM

Educational reimplementation of vLLM with PagedAttention. Companion code
to `part5-serving-infra/16a-lab-mini-vllm.md`.

This is a correctness-first reference for learning scheduler, block table,
prefix cache, swap, streaming, and sampler semantics. The default backend is a
correctness-first reference path, not a realistic performance path and not a
basis for performance extrapolation to vLLM. Triton is an opt-in experimental
path and does not represent vLLM's production kernels, async engine, CUDA graph
capture, distributed workers, or memory layout.

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

## 实验路线

Use these experiments as the recommended reading/running order. The fast tests
exercise semantics on ToyGPT/random tensors; slow HF checks are smoke/parity
tests and download TinyLlama weights.

| 实验 | 对应测试/命令 | 观察点 | 对应正文章节 | 真实 vLLM 省略项 |
|---|---|---|---|---|
| PagedAttention / block table | `pytest tests/test_block_manager.py tests/test_attention.py -v` | `BlockTable` maps logical token positions to physical blocks; `reshape_and_cache` writes `block_id * block_size + offset`; prefill/decode read K/V through block tables. | `part5-serving-infra/16a-lab-mini-vllm.md` §2-§3 | CUDA paged attention tile kernels, online softmax, production KV memory layout, CUDA graph capture, backend dispatch matrix. |
| continuous batching | `pytest tests/test_scheduler.py::test_continuous_batching_admits_during_decode tests/test_scheduler.py::test_admission_blocked_when_continuous_batching_disabled tests/test_e2e.py::test_e2e_continuous_batching_vs_baseline_same_output -v` | With `enable_continuous_batching=True`, waiting requests can enter while existing requests decode; output remains schedule-invariant for greedy tests. | §5 | Async engine loop, priority/fair scheduling, prefix-cache-aware admission, multi-worker scheduling. |
| chunked prefill | `pytest tests/test_attention.py::test_prefill_chunked_matches_unchunked tests/test_e2e.py::test_e2e_chunked_prefill_matches_unchunked -v` | Chunked and unchunked prompt processing produce the same outputs; `query_lens < seq_lens` validates cached-prefix causal masking. | §6 | Prefill/decode disaggregation, optimized ragged prefill kernels, overlap with decode, production token-budget policies. |
| prefix cache | `pytest tests/test_prefix_cache.py -v` | Repeated full blocks reuse cached physical blocks; ref counts and CoW protect shared blocks; repeat prompt skips already-prefilled tokens. | §7 | vLLM V1 prefix-cache hash policy, eviction heuristics, tenant isolation, persistent cache metrics, cache-aware scheduling. |
| swap / preemption | `pytest tests/test_swap.py -v` | GPU blocks can move to CPU and back without changing greedy output; shared prefix-cache blocks are not swap victims; cramped GPU pool triggers preemption. | §8 | Recompute preemption, async DMA/pinned memory, swap pipelining/prefetch, distributed KV movement. |
| streaming | `pytest tests/test_e2e.py::test_e2e_streaming_yields_same_tokens_as_generate -v` and `python examples/run_toy.py` | `generate_stream()` yields token-by-token results matching `generate()` and frees blocks after completion. | §9 / conclusion API notes | OpenAI-compatible HTTP/SSE server, request cancellation, backpressure, metrics, async client lifecycle. |
| HF smoke/parity | `python examples/run_tinyllama.py --max-tokens 12`; `pytest -m slow tests/test_llama_parity.py tests/test_llama_e2e.py -v` | TinyLlama loader, RoPE, fused QKV/gate-up, top-k logits overlap, and short greedy token-match path catch gross model parity regressions. | §4 | Full model zoo, tensor/pipeline/expert parallelism, quantization, LoRA, speculative decoding, production tokenizer/server compatibility. |

For stacked behavior and rough local timing only, run:

    python examples/bench.py --num-prompts 16 --max-tokens 32

The benchmark compares feature toggles in this mini engine. Because the default
backend is the correctness-first reference/Torch path, these numbers are useful
for local regression checks, not for performance inference about real vLLM.

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
defaults to the Torch/reference correctness-first reference backend, including
on CUDA. Triton remains an opt-in experimental path and is not selected just
because CUDA + Triton import successfully; opt in explicitly after CUDA runtime
validation:

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
