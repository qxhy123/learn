"""Naive 'gold-standard' attention used only by tests. Materializes the full
KV from block_table back into a contiguous tensor and calls SDPA. Slow but
trivially correct. Backends are validated against this."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def reference_decode(
    q: torch.Tensor,           # [B, H, D]
    key_cache: torch.Tensor,   # [num_blocks, H_kv, D, block_size]
    value_cache: torch.Tensor, # same
    block_table: torch.Tensor, # [B, max_blocks] int
    context_lens: torch.Tensor,# [B] int
    scale: float,
) -> torch.Tensor:
    B, H, D = q.shape
    H_kv = key_cache.shape[1]
    block_size = key_cache.shape[3]
    assert H % H_kv == 0
    group = H // H_kv

    out = torch.zeros_like(q)
    for b in range(B):
        ctx = int(context_lens[b].item())
        # Gather K/V for the seq into [ctx, H_kv, D]
        k_list, v_list = [], []
        remaining = ctx
        for blk_idx in range(block_table.shape[1]):
            if remaining <= 0:
                break
            block_id = int(block_table[b, blk_idx].item())
            take = min(block_size, remaining)
            # key_cache[block_id]: [H_kv, D, block_size] -> [block_size, H_kv, D]
            k_blk = key_cache[block_id, :, :, :take].permute(2, 0, 1)
            v_blk = value_cache[block_id, :, :, :take].permute(2, 0, 1)
            k_list.append(k_blk)
            v_list.append(v_blk)
            remaining -= take
        K = torch.cat(k_list, dim=0)  # [ctx, H_kv, D]
        V = torch.cat(v_list, dim=0)
        # Broadcast K/V across query heads in each group
        K = K.repeat_interleave(group, dim=1)  # [ctx, H, D]
        V = V.repeat_interleave(group, dim=1)
        # q[b]: [H, D], K: [ctx, H, D] -> scores [H, ctx]
        scores = torch.einsum("hd,thd->ht", q[b], K) * scale
        attn = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("ht,thd->hd", attn, V)
    return out


def reference_prefill(
    q: torch.Tensor,           # [N, H, D]
    key_cache: torch.Tensor,   # [num_blocks, H_kv, D, block_size]
    value_cache: torch.Tensor, # same
    block_table: torch.Tensor, # [B, max_blocks] int — blocks holding K/V for [0, seq_len)
    seq_lens: torch.Tensor,    # [B] — full ctx len AFTER this step (cached prefix + new chunk)
    query_lens: torch.Tensor,  # [B] — tokens being prefilled in this step (chunk size)
    scale: float,
) -> torch.Tensor:
    """Causal attention reading ALL K/V from cache via block_table.

    The current chunk's K/V are assumed to have already been written into the
    cache (via `reshape_and_cache`) before this call. So `block_table[b]`
    indexes blocks covering positions [0, seq_lens[b]) — both prior cached
    prefix and the just-written chunk.

    Each query position q in the new chunk attends causally over all positions
    [0, n_cached + q] where n_cached = seq_lens[b] - query_lens[b]. When
    n_cached == 0 (one-shot prefill), this is the standard causal triangular.
    When n_cached > 0 (chunked prefill or prefix cache hit), the new chunk
    attends to the entire prior context with no mask.
    """
    H = q.shape[1]
    H_kv = key_cache.shape[1]
    block_size = key_cache.shape[3]
    assert H % H_kv == 0
    group = H // H_kv

    out = torch.zeros_like(q)
    cursor = 0
    for b in range(len(seq_lens)):
        n_q = int(query_lens[b].item())
        n_kv = int(seq_lens[b].item())
        n_cached = n_kv - n_q

        # Gather K/V for positions [0, n_kv) from block_table[b]
        k_list, v_list = [], []
        remaining = n_kv
        for blk_idx in range(block_table.shape[1]):
            if remaining <= 0:
                break
            block_id = int(block_table[b, blk_idx].item())
            take = min(block_size, remaining)
            k_blk = key_cache[block_id, :, :, :take].permute(2, 0, 1)  # [take, H_kv, D]
            v_blk = value_cache[block_id, :, :, :take].permute(2, 0, 1)
            k_list.append(k_blk)
            v_list.append(v_blk)
            remaining -= take
        K = torch.cat(k_list, dim=0)             # [n_kv, H_kv, D]
        V = torch.cat(v_list, dim=0)
        K = K.repeat_interleave(group, dim=1)    # [n_kv, H, D]
        V = V.repeat_interleave(group, dim=1)

        qb = q[cursor:cursor + n_q]              # [n_q, H, D]
        # scores [H, n_q, n_kv]
        scores = torch.einsum("qhd,khd->hqk", qb, K) * scale
        # Causal mask: query position q (0-indexed within chunk, absolute = n_cached + q)
        # attends to kv positions [0, n_cached + q] inclusive.
        idx_q = torch.arange(n_q, device=q.device).unsqueeze(1)        # [n_q, 1]
        idx_kv = torch.arange(n_kv, device=q.device).unsqueeze(0)      # [1, n_kv]
        # Allowed if idx_kv <= n_cached + idx_q  i.e. idx_kv - idx_q <= n_cached
        allowed = idx_kv <= (n_cached + idx_q)                          # [n_q, n_kv]
        mask = torch.where(allowed, torch.zeros_like(scores[0]),
                           torch.full_like(scores[0], float("-inf")))   # [n_q, n_kv]
        scores = scores + mask                                          # broadcast over H
        attn = torch.softmax(scores, dim=-1)
        out[cursor:cursor + n_q] = torch.einsum("hqk,khd->qhd", attn, V)
        cursor += n_q
    return out
