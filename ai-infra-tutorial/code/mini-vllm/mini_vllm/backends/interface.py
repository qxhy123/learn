"""AttentionBackend protocol. All backends (torch, triton) implement this."""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import torch


@runtime_checkable
class AttentionBackend(Protocol):
    def reshape_and_cache(
        self,
        key: torch.Tensor,         # [num_tokens, num_kv_heads, head_dim]
        value: torch.Tensor,       # same
        key_cache: torch.Tensor,   # [num_blocks, num_kv_heads, head_dim, block_size]
        value_cache: torch.Tensor, # same
        slot_mapping: torch.Tensor # [num_tokens] int64; block_id*block_size + offset
    ) -> None: ...

    def prefill(
        self,
        q: torch.Tensor,           # [num_prefill_tokens, num_heads, head_dim]
        k: torch.Tensor,           # [num_prefill_tokens, num_kv_heads, head_dim]
        v: torch.Tensor,           # same
        seq_lens: torch.Tensor,    # [batch] full ctx len after this prefill
        query_lens: torch.Tensor,  # [batch] tokens being prefilled this step
        scale: float,
    ) -> torch.Tensor:             # [num_prefill_tokens, num_heads, head_dim]
        ...

    def decode(
        self,
        q: torch.Tensor,           # [batch, num_heads, head_dim]
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor, # [batch, max_blocks] int32
        context_lens: torch.Tensor, # [batch] int32 — kv length to attend over
        scale: float,
    ) -> torch.Tensor:             # [batch, num_heads, head_dim]
        ...
