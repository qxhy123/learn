"""Scheduler — Plan 4: continuous batching with token budget.

State machine:
    waiting -> running: admitted in a `schedule()` call when token budget allows
    running -> finished: when seq.is_finished()

Plan 5 will add prefix-cache lookup; Plan 6 adds swap_in/out and the `swapped` queue.

Continuous batching:
    When `enable_continuous_batching=True` (default), each `schedule()` step
    tries to admit waiting requests in addition to continuing running ones.
    Admission is gated by `max_num_batched_tokens` — total tokens (prefill +
    decode) per step is capped to keep step latency bounded.

    When False, the scheduler falls back to Plan 1 behavior: only admit when
    `running` is empty. This is the comparison baseline for benchmarks.
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
    blocks_to_copy: List[Tuple[int, int]] = field(default_factory=list)  # Plan 5


class Scheduler:
    def __init__(self, block_manager: BlockManager,
                 max_num_batched_tokens: int = 2048,
                 enable_continuous_batching: bool = True):
        self.bm = block_manager
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_continuous_batching = enable_continuous_batching
        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def mark_prefilled(self, seq: Sequence) -> None:
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

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        token_budget = self.max_num_batched_tokens

        # 1. Continue running seqs first (they have priority — already paid the
        #    prefill cost, so completing them is the cheapest path to free blocks).
        for seq in self.running:
            if seq.num_prefilled < seq.num_prompt_tokens:
                # First step after admission: still need prefill (full prompt at once
                # in Plan 4; chunked prefill is Plan 5).
                out.prefill_seqs.append(seq)
                token_budget -= seq.num_prompt_tokens
            else:
                # Need to ensure a slot exists for the upcoming token.
                # Plan 4: still no preemption — out-of-blocks raises. Plan 6 adds swap.
                self.bm.append_slot(seq)
                out.decode_seqs.append(seq)
                token_budget -= 1

        # 2. Admit new waiting requests if budget remains.
        # In Plan 1 mode (continuous batching off), we only admit when the running
        # queue is empty — preserves the original "process one batch at a time" behavior.
        can_admit = self.enable_continuous_batching or not self.running
        if can_admit:
            while self.waiting and token_budget > 0:
                seq = self.waiting[0]
                # Skip if this seq alone would blow the remaining budget.
                # (Don't want to partially-admit; Plan 5 chunked prefill handles that.)
                if seq.num_prompt_tokens > token_budget:
                    break
                status = self.bm.can_allocate(seq)
                if status == AllocStatus.OK:
                    self.bm.allocate(seq)
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)
                    out.prefill_seqs.append(seq)
                    token_budget -= seq.num_prompt_tokens
                    self.waiting.popleft()
                elif status == AllocStatus.LATER:
                    break
                else:  # NEVER
                    raise RuntimeError(
                        f"Request {seq.request_id} too large for cache "
                        f"({seq.num_prompt_tokens} tokens)")
        return out
