"""Token sampler. Plan 1 implements greedy only.
Plan 7 will add temperature, top-p, top-k."""
from __future__ import annotations
from typing import List
import torch

from mini_vllm.config import SamplingParams


class Sampler:
    def sample(self, logits: torch.Tensor, params: List[SamplingParams]) -> List[int]:
        """logits: [B, vocab]. Returns one token id per row."""
        assert logits.dim() == 2
        assert all(p.greedy for p in params), "Plan 1 supports greedy only"
        return logits.argmax(dim=-1).tolist()
