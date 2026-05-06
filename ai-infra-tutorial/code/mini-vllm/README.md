# mini-vLLM

Educational reimplementation of vLLM with PagedAttention. Companion code
to `part5-serving-infra/16a-lab-mini-vllm.md`.

## Status

**Plan 3 complete.** Engine runs both a toy GPT and TinyLlama-1.1B end-to-end
on CPU/MPS via the Torch paged-attention backend. Naive FCFS scheduler;
greedy sampling. Parity-tested against HF `transformers` (8/8 token greedy
match on `"The capital of France is"`).

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
- [ ] Plan 4: continuous batching + chunked prefill
- [ ] Plan 5: prefix caching + CoW
- [ ] Plan 6: swap to CPU + preemption
- [ ] Plan 7: streaming + full sampler (temperature/top-p/top-k) + bench
- [ ] Plan 8: tutorial chapter `16a-lab-mini-vllm.md`
