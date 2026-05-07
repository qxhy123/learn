"""Token sampler.

Plan 7 adds the non-greedy path: temperature → top-k → top-p (nucleus) →
multinomial. Greedy (the Plan 1 default) bypasses all of this and returns
the argmax. Per-request seed is supported for reproducibility.

Filter order matches vLLM/HF: temperature scales first, then top-k masks
beyond the k-th, then top-p masks beyond the cumulative-probability cutoff.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn.functional as F

from mini_vllm.config import SamplingParams


class Sampler:
    def sample(self, logits: torch.Tensor, params: List[SamplingParams]) -> List[int]:
        """logits: [B, vocab]. Returns one token id per row."""
        assert logits.dim() == 2
        out: List[int] = []
        for i, p in enumerate(params):
            row = logits[i]
            if p.greedy:
                out.append(int(row.argmax().item()))
                continue
            row = self._apply_filters(row, p)
            tok = self._multinomial(row, p)
            out.append(tok)
        return out

    def _apply_filters(self, logits: torch.Tensor, p: SamplingParams) -> torch.Tensor:
        """Apply temperature → top-k → top-p in that order. Returns a NEW
        tensor; the caller's row is not modified."""
        out = logits
        if p.temperature != 1.0:
            t = max(p.temperature, 1e-5)
            out = out / t
        if p.top_k > 0 and p.top_k < out.shape[-1]:
            # Keep the top_k largest; mask the rest to -inf.
            kth = torch.topk(out, p.top_k).values[-1]
            out = torch.where(out < kth,
                              torch.tensor(float('-inf'), dtype=out.dtype, device=out.device),
                              out)
        if p.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(out, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Tokens with cumulative probability > top_p are excluded — except we
            # always keep the first (highest-prob) token, even if its prob alone
            # exceeds top_p, so something is always sampleable.
            remove_sorted = cum_probs > p.top_p
            remove_sorted[0] = False
            # Map back to original indices.
            remove_idx = sorted_idx[remove_sorted]
            out = out.clone()
            out[remove_idx] = float('-inf')
        return out

    def _multinomial(self, logits: torch.Tensor, p: SamplingParams) -> int:
        probs = F.softmax(logits, dim=-1)
        if p.seed is not None:
            g = torch.Generator(device=probs.device).manual_seed(p.seed)
            tok = torch.multinomial(probs, 1, generator=g)
        else:
            tok = torch.multinomial(probs, 1)
        return int(tok.item())
