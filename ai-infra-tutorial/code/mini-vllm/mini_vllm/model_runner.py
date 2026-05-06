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
    prefill_block_table: torch.Tensor   # [num_prefill_seqs, max_blocks]
    prefill_seq_lens: torch.Tensor      # [num_prefill_seqs] full ctx after this step
    prefill_query_lens: torch.Tensor    # [num_prefill_seqs] chunk being prefilled
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

        # ---- Prefill region ----
        # Each prefill seq processes a chunk of length `chunk_len` starting at
        # `seq.num_prefilled`. For Plan 4 (no chunked prefill) chunk_len equals
        # the remaining-to-prefill = num_prompt_tokens - num_prefilled = full prompt
        # on the first step. For Plan 5 chunked prefill, the scheduler advances
        # `num_prefilled` per step and may set chunk_len < remaining.
        prefill_chunk_starts: List[int] = []
        prefill_chunk_lens: List[int] = []
        for seq in prefill_seqs:
            chunk_start = seq.num_prefilled
            chunk_len = seq.scheduled_chunk_len  # set by scheduler; defaults to prompt_len - num_prefilled
            prefill_chunk_starts.append(chunk_start)
            prefill_chunk_lens.append(chunk_len)
            chunk_end = chunk_start + chunk_len  # absolute position past last new token
            input_ids.extend(seq.prompt_token_ids[chunk_start:chunk_end])
            positions.extend(range(chunk_start, chunk_end))
            slot_mapping.extend(
                self.block_manager.get_slot_mapping(seq, chunk_start, chunk_end))
            prefill_seq_lens.append(chunk_end)        # full ctx covered AFTER this step
            prefill_query_lens.append(chunk_len)
            # If this step finishes the prompt, we sample from its last token.
            # If chunk doesn't yet finish the prompt (still prefilling), no sample.
            # We mark this seq's sample slot only when fully prefilled (chunk_end == prompt_len).
            if chunk_end == seq.num_prompt_tokens:
                sample_indices.append(cursor + chunk_len - 1)
            cursor += chunk_len
        num_prefill_tokens = cursor

        # Build prefill_block_table: each prefill seq needs blocks covering
        # positions [0, chunk_end) — both cached prefix and just-written chunk.
        max_prefill_blocks = max(
            ((prefill_chunk_starts[i] + prefill_chunk_lens[i] + bs - 1) // bs
             for i in range(len(prefill_seqs))),
            default=0,
        )
        prefill_block_table_list: List[List[int]] = []
        for seq in prefill_seqs:
            ids = [pb.block_id for pb in seq.block_table.physical_blocks]
            # Truncate to needed blocks (seq.block_table may have more pre-allocated)
            ids = ids[:max_prefill_blocks]
            ids = ids + [0] * (max_prefill_blocks - len(ids))
            prefill_block_table_list.append(ids)

        # ---- Decode region ----
        max_decode_blocks = max((len(s.block_table.physical_blocks) for s in decode_seqs),
                                default=0)
        decode_block_table_list: List[List[int]] = []
        decode_context_lens: List[int] = []
        for seq in decode_seqs:
            pos = seq.seq_len - 1   # the new (just-written) token's position
            input_ids.append(seq.token_ids[-1])
            positions.append(pos)
            slot_mapping.extend(self.block_manager.get_slot_mapping(seq, pos, pos + 1))
            ids = [pb.block_id for pb in seq.block_table.physical_blocks]
            ids = ids + [0] * (max_decode_blocks - len(ids))
            decode_block_table_list.append(ids)
            decode_context_lens.append(seq.seq_len)
            sample_indices.append(cursor)
            cursor += 1

        dev = self.device
        return ModelInput(
            input_ids=torch.tensor(input_ids, dtype=torch.long, device=dev),
            positions=torch.tensor(positions, dtype=torch.long, device=dev),
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=dev),
            prefill_block_table=(
                torch.tensor(prefill_block_table_list, dtype=torch.int32, device=dev)
                if prefill_block_table_list else
                torch.empty(0, 0, dtype=torch.int32, device=dev)),
            prefill_seq_lens=torch.tensor(prefill_seq_lens, dtype=torch.int32, device=dev),
            prefill_query_lens=torch.tensor(prefill_query_lens, dtype=torch.int32, device=dev),
            num_prefill_tokens=num_prefill_tokens,
            decode_block_table=(
                torch.tensor(decode_block_table_list, dtype=torch.int32, device=dev)
                if decode_block_table_list else
                torch.empty(0, 0, dtype=torch.int32, device=dev)),
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
                mi.prefill_block_table, mi.prefill_seq_lens, mi.prefill_query_lens,
                mi.num_prefill_tokens,
                mi.decode_block_table, mi.decode_context_lens,
                sample_indices=mi.sample_indices,
            )
        return logits
