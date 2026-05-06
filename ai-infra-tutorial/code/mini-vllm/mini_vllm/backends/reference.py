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
    k: torch.Tensor,           # [N, H_kv, D]
    v: torch.Tensor,           # [N, H_kv, D]
    seq_lens: torch.Tensor,    # [B]
    query_lens: torch.Tensor,  # [B]   for Plan 1 query_lens == seq_lens
    scale: float,
) -> torch.Tensor:
    """Causal attention within each sequence; sequences are independent."""
    H = q.shape[1]
    H_kv = k.shape[1]
    group = H // H_kv
    out = torch.zeros_like(q)
    cursor = 0
    for b in range(len(seq_lens)):
        n = int(query_lens[b].item())
        qb = q[cursor:cursor + n]                # [n, H, D]
        kb = k[cursor:cursor + n]                # [n, H_kv, D]
        vb = v[cursor:cursor + n]
        kb = kb.repeat_interleave(group, dim=1)  # [n, H, D]
        vb = vb.repeat_interleave(group, dim=1)
        # SDPA wants [B=1, H, T, D]
        ob = F.scaled_dot_product_attention(
            qb.transpose(0, 1).unsqueeze(0),
            kb.transpose(0, 1).unsqueeze(0),
            vb.transpose(0, 1).unsqueeze(0),
            is_causal=True, scale=scale,
        )  # [1, H, n, D]
        out[cursor:cursor + n] = ob.squeeze(0).transpose(0, 1)
        cursor += n
    return out
