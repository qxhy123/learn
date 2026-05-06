"""Scheduler — Plan 5: continuous batching + chunked prefill.

Plan 4 added continuous-batching admission. Plan 5 adds chunked prefill:
long prompts can be split across multiple steps, with each chunk attending
to prior chunks' K/V already in cache. The kernel-level mechanism for that
(prefill reads from KV cache via block_table) is shared with prefix caching.

Plan 6 will add swap_in/out and the `swapped` queue.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

from mini_vllm.block_manager import BlockManager, AllocStatus
from mini_vllm.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    prefill_seqs: List[Sequence] = field(default_factory=list)
    decode_seqs: List[Sequence] = field(default_factory=list)
    swap_in: Dict[int, int] = field(default_factory=dict)    # Plan 6
    swap_out: Dict[int, int] = field(default_factory=dict)   # Plan 6
    blocks_to_copy: List[Tuple[int, int]] = field(default_factory=list)  # Plan 5 prefix cache


class Scheduler:
    def __init__(self, block_manager: BlockManager,
                 max_num_batched_tokens: int = 2048,
                 enable_continuous_batching: bool = True,
                 enable_chunked_prefill: bool = False,
                 chunked_prefill_size: int = 512):
        self.bm = block_manager
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_continuous_batching = enable_continuous_batching
        self.enable_chunked_prefill = enable_chunked_prefill
        self.chunked_prefill_size = chunked_prefill_size
        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def mark_prefilled(self, seq: Sequence) -> None:
        """Legacy helper used by older tests. New code lets `Engine.step`
        advance `num_prefilled` by `scheduled_chunk_len` per step."""
        seq.num_prefilled = seq.num_prompt_tokens

    def free_finished(self) -> List[Sequence]:
        """Return finished seqs and remove them from running. Caller frees blocks."""
        still_running, finished = [], []
        for s in self.running:
            (finished if s.is_finished() else still_running).append(s)
        self.running = still_running
        for s in finished:
            self.bm.free(s)
        return finished

    def _chunk_for(self, seq: Sequence, budget: int) -> int:
        """How many tokens to prefill this step for a seq with `budget` left."""
        remaining = seq.num_prompt_tokens - seq.num_prefilled
        chunk = remaining
        if self.enable_chunked_prefill:
            chunk = min(chunk, self.chunked_prefill_size)
        chunk = min(chunk, budget)
        return max(0, chunk)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.max_num_batched_tokens

        # Reset per-step planning fields.
        for seq in self.running:
            seq.scheduled_chunk_len = 0

        # 1. Continue running seqs (priority: finishing them frees blocks).
        for seq in self.running:
            if seq.num_prefilled < seq.num_prompt_tokens:
                chunk = self._chunk_for(seq, budget)
                if chunk == 0:
                    continue   # no budget this step; pick up next step
                seq.scheduled_chunk_len = chunk
                out.prefill_seqs.append(seq)
                budget -= chunk
            else:
                # Decode — single token per step.
                if budget <= 0:
                    continue
                self.bm.append_slot(seq)
                out.decode_seqs.append(seq)
                budget -= 1

        # 2. Admit waiting requests if budget remains and policy allows.
        can_admit = self.enable_continuous_batching or not self.running
        if can_admit:
            while self.waiting and budget > 0:
                seq = self.waiting[0]
                # Decide chunk size before allocating.
                # When chunked prefill is OFF, we require the FULL prompt fits the
                # budget; partial admission is forbidden.
                remaining = seq.num_prompt_tokens
                if self.enable_chunked_prefill:
                    chunk = min(remaining, self.chunked_prefill_size, budget)
                    if chunk <= 0:
                        break
                else:
                    if remaining > budget:
                        break    # FCFS: head blocks the rest until a step has more budget
                    chunk = remaining

                status = self.bm.can_allocate(seq)
                if status == AllocStatus.OK:
                    self.bm.allocate(seq)
                    seq.status = SequenceStatus.RUNNING
                    seq.scheduled_chunk_len = chunk
                    self.running.append(seq)
                    out.prefill_seqs.append(seq)
                    budget -= chunk
                    self.waiting.popleft()
                elif status == AllocStatus.LATER:
                    break
                else:  # NEVER
                    raise RuntimeError(
                        f"Request {seq.request_id} too large for cache "
                        f"({seq.num_prompt_tokens} tokens)")
        return out
