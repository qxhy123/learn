"""Plan 1 scheduler: FCFS, no continuous batching, no preemption.

State machine:
    waiting -> running: admitted in a `schedule()` call when running is empty
    running -> finished: when seq.is_finished()

Plan 4 will turn `_can_admit_more()` from "running is empty" into
continuous-batching-with-token-budget; Plan 5 adds prefix cache lookup;
Plan 6 adds swap_in/out and the `swapped` queue.
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
    def __init__(self, block_manager: BlockManager):
        self.bm = block_manager
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
        # Decode existing running seqs first.
        for seq in self.running:
            if seq.num_prefilled < seq.num_prompt_tokens:
                # First step after admission: still need prefill.
                out.prefill_seqs.append(seq)
            else:
                # Need to ensure a slot exists for the upcoming token.
                # Plan 1: just append; out-of-blocks raises (no preemption).
                self.bm.append_slot(seq)
                out.decode_seqs.append(seq)

        # Admit new requests only when no running seqs exist (no continuous batching).
        if not self.running:
            while self.waiting:
                seq = self.waiting[0]
                status = self.bm.can_allocate(seq)
                if status == AllocStatus.OK:
                    self.bm.allocate(seq)
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)
                    out.prefill_seqs.append(seq)
                    self.waiting.popleft()
                elif status == AllocStatus.LATER:
                    break
                else:  # NEVER
                    raise RuntimeError(
                        f"Request {seq.request_id} too large for cache "
                        f"({seq.num_prompt_tokens} tokens)")
        return out
