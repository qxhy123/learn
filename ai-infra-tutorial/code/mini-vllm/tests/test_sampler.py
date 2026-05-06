import torch
from mini_vllm.sampler import Sampler
from mini_vllm.config import SamplingParams


def test_greedy_argmax():
    sampler = Sampler()
    logits = torch.tensor([[0.1, 5.0, 0.3], [2.0, 0.5, 1.0]])
    params = [SamplingParams(greedy=True), SamplingParams(greedy=True)]
    out = sampler.sample(logits, params)
    assert out == [1, 0]
