"""Physical KV-block bookkeeping.

Plan 1: basic alloc/free/append.
Plan 5: prefix caching via hash chain + ref_count > 1 + copy-on-write (CoW).
Plan 6 will add swap (device='cpu' fields).

Prefix caching algorithm:
    Each FILLED block (i.e., block_size tokens written into it) gets a
    deterministic hash derived from (prev_block_hash, tuple(token_ids)).
    The BlockManager maintains a `hash_to_block` registry. When a new
    sequence is admitted, we walk its prompt block-by-block looking up
    each hash; consecutive hits are reused (ref_count += 1) and only
    blocks past the matched prefix are freshly allocated.

    On decode write into a shared block (ref_count > 1), we trigger
    copy-on-write: allocate a fresh block, copy contents across the
    backend, decrement old ref, point this seq's block_table at the new
    block. The CoW source/dest is reported in `SchedulerOutput` so the
    runner can do the actual tensor copy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

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
    block_hash: Optional[int] = None     # set when block becomes a "filled" cached block
    device: str = "gpu"                  # Plan 6 toggles to "cpu"


@dataclass
class BlockTable:
    physical_blocks: List[PhysicalBlock] = field(default_factory=list)


def _hash_block(prev_hash: Optional[int], token_ids: Tuple[int, ...]) -> int:
    """Stable cross-process hash: (prev, ids). Python's `hash()` is salted
    per-process so we can't use it for serialization, but for in-process
    sharing it's fine and fast."""
    return hash((prev_hash, token_ids))


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int,
                 num_cpu_blocks: int = 0,
                 enable_prefix_caching: bool = False):
        self.num_blocks = num_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self._free_block_ids: List[int] = list(range(num_blocks))
        self._all_blocks: Dict[int, PhysicalBlock] = {}
        # Prefix cache: hash → cached block (still indexed even when ref_count=0
        # so a later seq can hit it). Only FILLED blocks are entered here.
        self._hash_to_block: Dict[int, PhysicalBlock] = {}
        # Evictable list: block_ids whose ref_count is 0 but are still cached
        # (kept alive for prefix sharing). FIFO-evicted when out of fresh blocks.
        self._evictable: List[int] = []
        # CPU swap pool (Plan 6). Disjoint id namespace from GPU.
        self._free_cpu_ids: List[int] = list(range(num_cpu_blocks))
        self._cpu_blocks: Dict[int, PhysicalBlock] = {}

    # ---- query ----
    @property
    def num_free_blocks(self) -> int:
        # GPU side: fresh + evictable.
        return len(self._free_block_ids) + len(self._evictable)

    @property
    def num_free_cpu_blocks(self) -> int:
        return len(self._free_cpu_ids)

    def can_allocate(self, seq: "Sequence") -> AllocStatus:
        # Worst-case need: full prompt with no prefix hit.
        needed = self._num_blocks_needed(seq.num_prompt_tokens)
        if needed > self.num_blocks:
            return AllocStatus.NEVER
        # Subtract prefix-cache hits from the requirement.
        cached_count = self._effective_cached_count(seq.prompt_token_ids,
                                                    seq.num_prompt_tokens)
        needed = max(0, needed - cached_count)
        if needed > self.num_free_blocks:
            return AllocStatus.LATER
        return AllocStatus.OK

    def cached_prefix_tokens(self, token_ids: List[int]) -> int:
        """Number of leading prompt tokens that would be served from the
        prefix cache (always a multiple of block_size, < prompt_len)."""
        if not self.enable_prefix_caching:
            return 0
        return self._effective_cached_count(token_ids, len(token_ids)) * self.block_size

    # ---- mutate ----
    def allocate(self, seq: "Sequence") -> BlockTable:
        """Allocate blocks for `seq`'s full prompt.

        With prefix caching enabled, leading blocks that hash-match cached
        entries are reused (ref_count += 1); only the tail is freshly
        allocated. Sets `seq.num_prefilled` to the number of tokens covered
        by the cached prefix so the scheduler/runner skip computing them.
        """
        n_total = self._num_blocks_needed(seq.num_prompt_tokens)
        cached: List[PhysicalBlock] = []
        if self.enable_prefix_caching:
            cached = self._lookup_cached_prefix(seq.prompt_token_ids)
            # Reserve at least 1 fresh token of compute so the model produces
            # next-token logits even when the prompt is fully cached.
            if cached and len(cached) * self.block_size >= seq.num_prompt_tokens:
                cached = cached[:-1]
            for pb in cached:
                self._rescue_from_evictable(pb)
                pb.ref_count += 1
        # Fresh blocks for the remainder.
        n_fresh = n_total - len(cached)
        fresh = [self._take_free_block() for _ in range(n_fresh)]
        seq.block_table = BlockTable(physical_blocks=cached + fresh)
        # Mark how many prompt tokens are already cached (full blocks only).
        seq.num_prefilled = len(cached) * self.block_size
        return seq.block_table

    def append_slot(self, seq: "Sequence") -> Optional[Tuple[int, int]]:
        """Decode-time block extension. Two side effects can occur:
          1. The last block fills up → allocate a new block.
          2. The last block is shared (ref_count > 1) → CoW: alloc fresh,
             return (src_block_id, dst_block_id) so the runner copies the
             K/V tensor contents across.
        Returns the CoW mapping if it triggered, else None.
        """
        assert seq.block_table is not None, "sequence not allocated"
        used_slots = seq.seq_len
        needed = self._num_blocks_needed(used_slots)
        have = len(seq.block_table.physical_blocks)
        if needed > have:
            # Append a new block. The block we extend INTO is fresh, no CoW.
            assert needed == have + 1, "append_slot extends by exactly one block"
            seq.block_table.physical_blocks.append(self._take_free_block())
            # If the previous (now full) block is shared, the next register-fill
            # could overwrite it; but Plan 5 only registers our OWN newly-filled
            # blocks. We don't write into shared blocks here.
            return None

        # Same block; check CoW. The "last" block (where the new token will
        # land) is physical_blocks[-1].
        last = seq.block_table.physical_blocks[-1]
        if last.ref_count > 1:
            # Allocate a fresh block, decrement old, record CoW pair.
            new = self._take_free_block()
            last.ref_count -= 1
            seq.block_table.physical_blocks[-1] = new
            return (last.block_id, new.block_id)
        return None

    def register_filled_blocks(self, seq: "Sequence") -> None:
        """Walk seq.block_table; any block that just became 'filled'
        (block_size tokens) and isn't yet hashed gets registered for sharing.
        Call this after a step writes new K/V into the cache.
        """
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        prev_hash: Optional[int] = None
        # Compute hashes incrementally; only register full blocks.
        n_filled = seq.seq_len // bs   # number of fully-filled blocks
        for i in range(n_filled):
            pb = seq.block_table.physical_blocks[i]
            tokens = tuple(seq.token_ids[i * bs:(i + 1) * bs])
            h = _hash_block(prev_hash, tokens)
            if pb.block_hash is None:
                pb.block_hash = h
                # Don't displace a stronger entry: only register if this hash
                # isn't already mapped to another block.
                if h not in self._hash_to_block:
                    self._hash_to_block[h] = pb
            prev_hash = pb.block_hash

    def free(self, seq: "Sequence") -> None:
        if seq.block_table is None:
            return
        for pb in seq.block_table.physical_blocks:
            pb.ref_count -= 1
            if pb.ref_count == 0:
                if (self.enable_prefix_caching and pb.block_hash is not None
                        and self._hash_to_block.get(pb.block_hash) is pb):
                    # Keep alive in evictable cache for prefix-share reuse.
                    self._evictable.append(pb.block_id)
                else:
                    if pb.block_hash is not None and self._hash_to_block.get(pb.block_hash) is pb:
                        del self._hash_to_block[pb.block_hash]
                    self._free_block_ids.append(pb.block_id)
                    del self._all_blocks[pb.block_id]
        seq.block_table = None

    # ---- swap (Plan 6) ----

    def can_swap_out(self, seq: "Sequence") -> bool:
        """Eligible iff (1) all GPU blocks are private (ref_count==1), and
        (2) the CPU pool has room for them. Shared blocks (e.g. cached prefix
        blocks held by other seqs) are NOT swap-eligible — swapping them
        would invalidate other sequences' state."""
        if seq.block_table is None:
            return False
        gpu_blocks = [pb for pb in seq.block_table.physical_blocks if pb.device == 'gpu']
        if any(pb.ref_count > 1 for pb in gpu_blocks):
            return False
        return self.num_free_cpu_blocks >= len(gpu_blocks)

    def can_swap_in(self, seq: "Sequence") -> bool:
        if seq.block_table is None:
            return False
        cpu_blocks = [pb for pb in seq.block_table.physical_blocks if pb.device == 'cpu']
        return self.num_free_blocks >= len(cpu_blocks)

    def swap_out(self, seq: "Sequence") -> Dict[int, int]:
        """Move all of `seq`'s GPU blocks to the CPU pool. Returns
        {gpu_block_id: cpu_block_id} so the caller (CacheEngine) can copy
        the K/V tensor data."""
        assert self.can_swap_out(seq), "swap_out called on ineligible seq"
        mapping: Dict[int, int] = {}
        new_blocks: List[PhysicalBlock] = []
        for pb in seq.block_table.physical_blocks:
            if pb.device != 'gpu':
                new_blocks.append(pb)
                continue
            cpu_id = self._free_cpu_ids.pop()
            cpu_pb = PhysicalBlock(block_id=cpu_id, ref_count=1, device='cpu')
            self._cpu_blocks[cpu_id] = cpu_pb
            mapping[pb.block_id] = cpu_id
            # Release the GPU block. `pb` had ref_count==1 (asserted above).
            self._release_gpu_block(pb)
            new_blocks.append(cpu_pb)
        seq.block_table.physical_blocks = new_blocks
        return mapping

    def swap_in(self, seq: "Sequence") -> Dict[int, int]:
        """Move all of `seq`'s CPU blocks back to GPU. Returns
        {cpu_block_id: gpu_block_id}."""
        assert self.can_swap_in(seq), "swap_in called when no room"
        mapping: Dict[int, int] = {}
        new_blocks: List[PhysicalBlock] = []
        for pb in seq.block_table.physical_blocks:
            if pb.device != 'cpu':
                new_blocks.append(pb)
                continue
            gpu_pb = self._take_free_block()
            mapping[pb.block_id] = gpu_pb.block_id
            # Release the CPU block.
            del self._cpu_blocks[pb.block_id]
            self._free_cpu_ids.append(pb.block_id)
            new_blocks.append(gpu_pb)
        seq.block_table.physical_blocks = new_blocks
        return mapping

    def _release_gpu_block(self, pb: PhysicalBlock) -> None:
        """Release a GPU block whose ref_count is currently 1 (caller asserted)."""
        # Remove from hash registry if present.
        if pb.block_hash is not None and self._hash_to_block.get(pb.block_hash) is pb:
            del self._hash_to_block[pb.block_hash]
        # If somehow in evictable list, remove (shouldn't happen for ref_count==1).
        if pb.block_id in self._evictable:
            self._evictable.remove(pb.block_id)
        del self._all_blocks[pb.block_id]
        self._free_block_ids.append(pb.block_id)

    def get_slot_mapping(self, seq: "Sequence", start: int, end: int) -> List[int]:
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
        if self._free_block_ids:
            block_id = self._free_block_ids.pop()
        elif self._evictable:
            # Evict the oldest cached block (FIFO).
            block_id = self._evictable.pop(0)
            old = self._all_blocks[block_id]
            if old.block_hash is not None and self._hash_to_block.get(old.block_hash) is old:
                del self._hash_to_block[old.block_hash]
            del self._all_blocks[block_id]
        else:
            raise RuntimeError("BlockManager: no free blocks available")
        pb = PhysicalBlock(block_id=block_id, ref_count=1)
        self._all_blocks[block_id] = pb
        return pb

    def _effective_cached_count(self, token_ids: List[int], prompt_len: int) -> int:
        """How many prefix-cache blocks would actually be reused for a prompt
        of length `prompt_len`, after enforcing the "at least 1 fresh token"
        rule. Returns 0 if prefix caching is disabled."""
        if not self.enable_prefix_caching:
            return 0
        cached = self._lookup_cached_prefix(token_ids)
        if cached and len(cached) * self.block_size >= prompt_len:
            return len(cached) - 1
        return len(cached)

    def _lookup_cached_prefix(self, token_ids: List[int]) -> List[PhysicalBlock]:
        """Walk hash chain; return as many cached blocks as match the prompt."""
        bs = self.block_size
        n_full = len(token_ids) // bs
        out: List[PhysicalBlock] = []
        prev_hash: Optional[int] = None
        for i in range(n_full):
            tokens = tuple(token_ids[i * bs:(i + 1) * bs])
            h = _hash_block(prev_hash, tokens)
            pb = self._hash_to_block.get(h)
            if pb is None:
                break
            out.append(pb)
            prev_hash = h
        return out

    def _rescue_from_evictable(self, pb: PhysicalBlock) -> None:
        """If `pb` is currently in the evictable list, remove it. Called by
        allocate() right before incrementing ref_count on a cached block."""
        if pb.ref_count == 0 and pb.block_id in self._evictable:
            self._evictable.remove(pb.block_id)
