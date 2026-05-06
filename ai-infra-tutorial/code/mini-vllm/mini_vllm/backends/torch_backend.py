"""CPU/MPS reference implementation. Slow but correct; primary use is to
validate the Triton backend (Plan 2) and to serve as the only backend on
machines without CUDA."""
from __future__ import annotations
import torch
import torch.nn.functional as F

from mini_vllm.backends.reference import reference_decode, reference_prefill


class TorchBackend:
    def reshape_and_cache(self, key, value, key_cache, value_cache, slot_mapping):
        """Scatter `key`/`value` (shape [N, H_kv, D]) into the paged cache at
        positions given by `slot_mapping` (shape [N], values = block_id*block_size + offset).
        Cache layout: [num_blocks, H_kv, D, block_size].
        """
        block_size = key_cache.shape[3]
        block_ids = (slot_mapping // block_size).long()
        offsets = (slot_mapping % block_size).long()
        # key: [N, H_kv, D] -> we want to write key[n] into key_cache[block_ids[n], :, :, offsets[n]]
        for n in range(slot_mapping.shape[0]):
            key_cache[block_ids[n], :, :, offsets[n]] = key[n]
            value_cache[block_ids[n], :, :, offsets[n]] = value[n]

    def prefill(self, q, key_cache, value_cache, block_table, seq_lens, query_lens, scale):
        # Plan 5: prefill reads K/V from cache (chunk's K/V already written).
        # Causal mask within the new chunk; cached prefix fully visible.
        return reference_prefill(q, key_cache, value_cache, block_table,
                                 seq_lens, query_lens, scale)

    def decode(self, q, key_cache, value_cache, block_table, context_lens, scale):
        # Plan 1 uses the reference implementation as the production decode kernel.
        # Plan 2 will replace this with a Triton kernel; the Torch backend will keep
        # using `reference_decode` as its (slow) fallback path.
        return reference_decode(q, key_cache, value_cache, block_table, context_lens, scale)
