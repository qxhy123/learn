"""Tests for greedy + temperature/top-p/top-k sampler."""
from collections import Counter
import torch
from mini_vllm.sampler import Sampler
from mini_vllm.config import SamplingParams


def test_greedy_argmax():
    sampler = Sampler()
    logits = torch.tensor([[0.1, 5.0, 0.3], [2.0, 0.5, 1.0]])
    params = [SamplingParams(greedy=True), SamplingParams(greedy=True)]
    out = sampler.sample(logits, params)
    assert out == [1, 0]


def test_top_k_one_equals_greedy():
    """top_k=1 with greedy=False should pick the argmax just like greedy."""
    sampler = Sampler()
    logits = torch.tensor([[0.1, 5.0, 0.3]])
    p = SamplingParams(greedy=False, top_k=1, temperature=1.0, top_p=1.0, seed=0)
    assert sampler.sample(logits, [p]) == [1]


def test_temperature_low_concentrates_on_argmax():
    """With near-zero temperature, distribution sharpens onto argmax."""
    sampler = Sampler()
    # Two tokens of similar logits; very low temperature → almost certainly argmax.
    logits = torch.tensor([[1.0, 1.05, 0.0]])
    p = SamplingParams(greedy=False, temperature=0.01, seed=42)
    counts = Counter(sampler.sample(logits, [p])[0] for _ in range(50))
    # Resample with same seed many times — but seed reseeds the generator each
    # call, so this is deterministic per call. Just verify the argmax dominates
    # over many independent (no-seed) samples.
    p_unseeded = SamplingParams(greedy=False, temperature=0.01)
    counts = Counter(sampler.sample(logits, [p_unseeded])[0] for _ in range(200))
    assert counts.most_common(1)[0][0] == 1   # argmax token wins


def test_top_p_excludes_low_probability_tail():
    """With top_p=0.5 over a peaked distribution, only the top tokens
    accounting for >=50% mass are sampleable."""
    sampler = Sampler()
    # Three tokens with probs roughly [0.7, 0.2, 0.1]:  log-prob ≈ logits.
    # Use logits that translate to near these probs.
    logits = torch.tensor([[2.5, 1.1, 0.4]])    # softmax ≈ [0.66, 0.16, 0.08, ...]
    p = SamplingParams(greedy=False, top_p=0.5, temperature=1.0)
    counts = Counter(sampler.sample(logits, [p])[0] for _ in range(200))
    # With top_p=0.5, only the highest-prob token (which alone exceeds 0.5)
    # should ever be chosen.
    assert set(counts) == {0}


def test_per_request_seed_reproducibility():
    sampler = Sampler()
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])      # uniform
    p1 = SamplingParams(greedy=False, temperature=1.0, seed=123)
    p2 = SamplingParams(greedy=False, temperature=1.0, seed=123)
    p3 = SamplingParams(greedy=False, temperature=1.0, seed=999)
    a = sampler.sample(logits, [p1])
    b = sampler.sample(logits, [p2])
    c = sampler.sample(logits, [p3])
    assert a == b
    # Two different seeds CAN coincidentally pick the same token in a 5-way
    # uniform; just sanity-check the seeded path doesn't crash.
    assert isinstance(c[0], int)


def test_batched_mixed_greedy_and_sampling():
    """A single Sampler.sample() call can mix greedy and non-greedy rows."""
    sampler = Sampler()
    logits = torch.tensor([[5.0, 0.0, 0.0],     # row 0: greedy → arg 0
                           [0.0, 0.0, 5.0]])    # row 1: top_k=1 → arg 2
    params = [SamplingParams(greedy=True),
              SamplingParams(greedy=False, top_k=1)]
    assert sampler.sample(logits, params) == [0, 2]
