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
                 chunked_prefill_size: int = 512,
                 enable_swap: bool = False):
        self.bm = block_manager
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_continuous_batching = enable_continuous_batching
        self.enable_chunked_prefill = enable_chunked_prefill
        self.chunked_prefill_size = chunked_prefill_size
        self.enable_swap = enable_swap
        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.swapped: List[Sequence] = []   # Plan 6

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting) or bool(self.running) or bool(self.swapped)

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

    # ---- swap helpers (Plan 6) ----

    def _try_swap_in(self, out: SchedulerOutput) -> None:
        """FIFO swap-in from swapped queue while there's GPU room."""
        while self.swapped:
            seq = self.swapped[0]
            if not self.bm.can_swap_in(seq):
                break
            mapping = self.bm.swap_in(seq)
            out.swap_in.update(mapping)
            self.swapped.pop(0)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

    def _ensure_room_for_append(self, decoding_seq: Sequence,
                                 out: SchedulerOutput) -> None:
        """If `decoding_seq` will need a new GPU block this step and the pool
        is empty, evict the most recently admitted OTHER running seq via swap.
        Repeats until either there's room or no eligible victim remains."""
        # How many extra blocks does decoding_seq need this step? Either 0
        # (still room in last block) or 1 (last block full → new block needed).
        bs = self.bm.block_size
        used = decoding_seq.seq_len
        needed = (used + bs) // bs                    # after writing 1 more token
        have = len(decoding_seq.block_table.physical_blocks)
        extra = max(0, needed - have)
        while extra > self.bm.num_free_blocks:
            victim = self._pick_swap_victim(exclude=decoding_seq)
            if victim is None:
                break  # no eligible victim; the append_slot will raise
            mapping = self.bm.swap_out(victim)
            out.swap_out.update(mapping)
            self.running.remove(victim)
            victim.status = SequenceStatus.SWAPPED
            self.swapped.append(victim)

    def _pick_swap_victim(self, exclude: Sequence) -> "Sequence | None":
        """Pick the most-recently-admitted running seq that's eligible to swap.
        Recency = position from end of `self.running` (LIFO of admissions)."""
        for seq in reversed(self.running):
            if seq is exclude:
                continue
            if self.bm.can_swap_out(seq):
                return seq
        return None

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

        # 0. Try swapping in from the swapped queue (FIFO). Only attempt this
        # if it doesn't itself starve currently-running decodes; we leave a
        # safety margin equal to the worst-case running decode demand.
        if self.enable_swap and self.swapped:
            self._try_swap_in(out)

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
                # Check if a new block is needed and we're out of GPU room.
                # If so, preempt LRU-style: swap out the most-recently-admitted
                # running seq (excluding this one) until there's room.
                if self.enable_swap:
                    self._ensure_room_for_append(seq, out)
                self.bm.append_slot(seq)
                out.decode_seqs.append(seq)
                budget -= 1

        # 2. Admit waiting requests if budget remains and policy allows.
        can_admit = self.enable_continuous_batching or not self.running
        if can_admit:
            while self.waiting and budget > 0:
                seq = self.waiting[0]
                # Account for prefix-cache hits (Plan 5 prefix caching): tokens
                # in matched cached blocks need no compute, so chunk_len is
                # measured against the UNCACHED remainder.
                cached_tokens = self.bm.cached_prefix_tokens(seq.prompt_token_ids)
                remaining = seq.num_prompt_tokens - cached_tokens
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
                    self.bm.allocate(seq)   # may bump seq.num_prefilled via prefix cache
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
