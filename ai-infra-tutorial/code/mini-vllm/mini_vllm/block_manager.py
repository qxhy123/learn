"""Physical KV-block bookkeeping. Plan 1: basic alloc/free/append.
Prefix caching (ref_count > 1, hash chain), swap, CoW are added in later plans
but the data model already accommodates them (ref_count, device fields).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mini_vllm.sequence import Sequence


class AllocStatus(Enum):
    OK = "ok"
    LATER = "later"      # not enough blocks now, retry later
    NEVER = "never"      # request larger than total capacity


@dataclass
class PhysicalBlock:
    block_id: int
    ref_count: int = 1
    block_hash: Optional[int] = None     # Plan 5 fills this
    device: str = "gpu"                  # Plan 6 toggles to "cpu"


@dataclass
class BlockTable:
    physical_blocks: List[PhysicalBlock] = field(default_factory=list)


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Free list: stack of available block_ids (LIFO is fine)
        self._free_block_ids: List[int] = list(range(num_blocks))
        # All allocated blocks indexed by id (so we can lookup by id later)
        self._all_blocks: dict[int, PhysicalBlock] = {}

    # ---- query ----
    @property
    def num_free_blocks(self) -> int:
        return len(self._free_block_ids)

    def can_allocate(self, seq: "Sequence") -> AllocStatus:
        needed = self._num_blocks_needed(seq.num_prompt_tokens)
        if needed > self.num_blocks:
            return AllocStatus.NEVER
        if needed > self.num_free_blocks:
            return AllocStatus.LATER
        return AllocStatus.OK

    # ---- mutate ----
    def allocate(self, seq: "Sequence") -> BlockTable:
        needed = self._num_blocks_needed(seq.num_prompt_tokens)
        blocks = [self._take_free_block() for _ in range(needed)]
        seq.block_table = BlockTable(physical_blocks=blocks)
        return seq.block_table

    def append_slot(self, seq: "Sequence") -> None:
        """Called per generated token. Allocates a new block iff the last
        block is full."""
        assert seq.block_table is not None, "sequence not allocated"
        used_slots = seq.seq_len  # prompt + generated tokens already counted
        needed = self._num_blocks_needed(used_slots)
        have = len(seq.block_table.physical_blocks)
        if needed > have:
            assert needed == have + 1, "append_slot extends by exactly one block"
            seq.block_table.physical_blocks.append(self._take_free_block())

    def free(self, seq: "Sequence") -> None:
        if seq.block_table is None:
            return
        for pb in seq.block_table.physical_blocks:
            pb.ref_count -= 1
            if pb.ref_count == 0:
                self._free_block_ids.append(pb.block_id)
                del self._all_blocks[pb.block_id]
        seq.block_table = None

    def get_slot_mapping(self, seq: "Sequence", start: int, end: int) -> List[int]:
        """Return physical slot ids for token positions [start, end) within seq."""
        mapping: List[int] = []
        bs = self.block_size
        for pos in range(start, end):
            block_idx = pos // bs
            offset = pos % bs
            pb = seq.block_table.physical_blocks[block_idx]
            mapping.append(pb.block_id * bs + offset)
        return mapping

    # ---- helpers ----
    def _num_blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def _take_free_block(self) -> PhysicalBlock:
        block_id = self._free_block_ids.pop()
        pb = PhysicalBlock(block_id=block_id, ref_count=1)
        self._all_blocks[block_id] = pb
        return pb
