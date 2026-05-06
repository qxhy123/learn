"""Model interface. Engine/runner only see this."""
from __future__ import annotations
from typing import Protocol, List, Tuple, runtime_checkable
import torch

from mini_vllm.config import ModelConfig

KVTensorPair = Tuple[torch.Tensor, torch.Tensor]


@runtime_checkable
class CausalLM(Protocol):
    config: ModelConfig
    def forward(
        self,
        input_ids: torch.Tensor,           # [N_total]
        positions: torch.Tensor,           # [N_total]
        slot_mapping: torch.Tensor,        # [N_total]
        kv_caches: List[KVTensorPair],
        # prefill block
        prefill_seq_lens: torch.Tensor,    # [B_pre]
        prefill_query_lens: torch.Tensor,  # [B_pre]
        num_prefill_tokens: int,
        # decode block
        decode_block_table: torch.Tensor,  # [B_dec, max_blocks]
        decode_context_lens: torch.Tensor, # [B_dec]
    ) -> torch.Tensor:                     # [B_pre + B_dec, vocab]   (one logit row per seq sampled position)
        ...
