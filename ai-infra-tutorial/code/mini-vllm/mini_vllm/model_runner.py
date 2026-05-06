"""Builds tensors from scheduler output and runs the model."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch

from mini_vllm.sequence import Sequence
from mini_vllm.block_manager import BlockManager
from mini_vllm.cache_engine import CacheEngine
from mini_vllm.models.base import CausalLM


@dataclass
class ModelInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    prefill_seq_lens: torch.Tensor
    prefill_query_lens: torch.Tensor
    num_prefill_tokens: int
    decode_block_table: torch.Tensor
    decode_context_lens: torch.Tensor
    sample_indices: torch.Tensor


class ModelRunner:
    def __init__(self, model: CausalLM, cache_engine: CacheEngine,
                 block_manager: BlockManager, device: str):
        self.model = model
        self.cache_engine = cache_engine
        self.block_manager = block_manager
        self.device = device

    def build_input(self, prefill_seqs: List[Sequence],
                    decode_seqs: List[Sequence]) -> ModelInput:
        bs = self.block_manager.block_size
        input_ids: List[int] = []
        positions: List[int] = []
        slot_mapping: List[int] = []
        prefill_seq_lens: List[int] = []
        prefill_query_lens: List[int] = []
        sample_indices: List[int] = []
        cursor = 0

        # Prefill region first
        for seq in prefill_seqs:
            # Plan 1: prefill the full prompt in one shot.
            n = seq.num_prompt_tokens
            input_ids.extend(seq.prompt_token_ids)
            positions.extend(range(n))
            slot_mapping.extend(self.block_manager.get_slot_mapping(seq, 0, n))
            prefill_seq_lens.append(n)
            prefill_query_lens.append(n)
            sample_indices.append(cursor + n - 1)  # last position of this seq
            cursor += n
        num_prefill_tokens = cursor

        # Decode region
        max_blocks = max((len(s.block_table.physical_blocks) for s in decode_seqs),
                        default=0)
        decode_block_table_list: List[List[int]] = []
        decode_context_lens: List[int] = []
        for seq in decode_seqs:
            pos = seq.seq_len - 1   # the new (just-written) token's position
            input_ids.append(seq.token_ids[-1])
            positions.append(pos)
            slot_mapping.extend(self.block_manager.get_slot_mapping(seq, pos, pos + 1))
            ids = [pb.block_id for pb in seq.block_table.physical_blocks]
            ids = ids + [0] * (max_blocks - len(ids))   # pad with 0 (won't be read)
            decode_block_table_list.append(ids)
            decode_context_lens.append(seq.seq_len)     # K/V len = full ctx incl. just-written
            sample_indices.append(cursor)
            cursor += 1

        dev = self.device
        return ModelInput(
            input_ids=torch.tensor(input_ids, dtype=torch.long, device=dev),
            positions=torch.tensor(positions, dtype=torch.long, device=dev),
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=dev),
            prefill_seq_lens=torch.tensor(prefill_seq_lens, dtype=torch.int32, device=dev),
            prefill_query_lens=torch.tensor(prefill_query_lens, dtype=torch.int32, device=dev),
            num_prefill_tokens=num_prefill_tokens,
            decode_block_table=torch.tensor(decode_block_table_list, dtype=torch.int32, device=dev)
                if decode_block_table_list else torch.empty(0, 0, dtype=torch.int32, device=dev),
            decode_context_lens=torch.tensor(decode_context_lens, dtype=torch.int32, device=dev),
            sample_indices=torch.tensor(sample_indices, dtype=torch.long, device=dev),
        )

    def execute(self, prefill_seqs: List[Sequence],
                decode_seqs: List[Sequence]) -> torch.Tensor:
        mi = self.build_input(prefill_seqs, decode_seqs)
        with torch.inference_mode():
            logits = self.model(
                mi.input_ids, mi.positions, mi.slot_mapping,
                self.cache_engine.kv_caches,
                mi.prefill_seq_lens, mi.prefill_query_lens, mi.num_prefill_tokens,
                mi.decode_block_table, mi.decode_context_lens,
                sample_indices=mi.sample_indices,
            )
        return logits
