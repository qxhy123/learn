"""Triton paged-attention backend (Plan 2).

GPU implementation of the same `AttentionBackend` interface as the Torch
reference. Three Triton kernels:

  1. `_reshape_and_cache_kernel`  — parallel scatter of new K/V into cache
  2. `_paged_decode_kernel`       — canonical PagedAttention decode: each
     program handles one (batch, query_head) pair, walks the seq's
     block_table, and accumulates online softmax in-register
  3. `_paged_prefill_kernel`      — "ragged" prefill: each program handles
     one (query_token, query_head) pair, reads the seq's K/V from cache via
     block_table, applies causal mask aware of cached prefix vs new chunk

Design constraints:
  - GQA: query heads grouped onto kv heads (`num_heads % num_kv_heads == 0`).
    A program for query head q reads kv_head = q // group_size.
  - Cache layout: `[num_blocks, num_kv_heads, head_dim, block_size]` (matches
    TorchBackend / CacheEngine).
  - Causal: prefill kernel masks out kv positions > n_cached + chunk_pos.

**This file has not been runtime-verified.** It mirrors the math of
`backends/reference.py` line-by-line; tests are gated to skip without GPU
+ Triton. When you run it on a CUDA box for the first time, expect 1-2
small fixes (typical: pointer stride bug, mask off-by-one, fp16 numerics).
The validation contract is straightforward — outputs must `allclose` the
Torch reference at `atol=1e-2, rtol=1e-2` for fp16.
"""
from __future__ import annotations
from typing import List
import math
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ============================================================================
# Kernel 1: reshape_and_cache
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _reshape_and_cache_kernel(
        key_ptr,                # [N, H_kv, D]
        value_ptr,
        key_cache_ptr,          # [num_blocks, H_kv, D, block_size]
        value_cache_ptr,
        slot_mapping_ptr,       # [N] int64
        # Strides
        stride_k_n, stride_k_h, stride_k_d,
        stride_kc_block, stride_kc_h, stride_kc_d, stride_kc_b,
        # Constants
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        slot = tl.load(slot_mapping_ptr + token_idx)
        block_id = slot // BLOCK_SIZE
        offset = slot % BLOCK_SIZE

        # Pointers to the [HEAD_DIM] vector to read + write.
        k_in_ptr = key_ptr + token_idx * stride_k_n + head_idx * stride_k_h
        v_in_ptr = value_ptr + token_idx * stride_k_n + head_idx * stride_k_h
        d_offsets = tl.arange(0, HEAD_DIM)
        k_vec = tl.load(k_in_ptr + d_offsets * stride_k_d)
        v_vec = tl.load(v_in_ptr + d_offsets * stride_k_d)

        # Write into cache[block_id, head_idx, :, offset]
        kc_dst = (key_cache_ptr
                  + block_id * stride_kc_block
                  + head_idx * stride_kc_h
                  + d_offsets * stride_kc_d
                  + offset * stride_kc_b)
        vc_dst = (value_cache_ptr
                  + block_id * stride_kc_block
                  + head_idx * stride_kc_h
                  + d_offsets * stride_kc_d
                  + offset * stride_kc_b)
        tl.store(kc_dst, k_vec)
        tl.store(vc_dst, v_vec)


# ============================================================================
# Kernel 2: paged decode
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _paged_decode_kernel(
        q_ptr,                   # [B, H, D]
        kc_ptr, vc_ptr,          # [num_blocks, H_kv, D, block_size]
        block_table_ptr,         # [B, max_blocks] int32
        context_lens_ptr,        # [B] int32
        out_ptr,                 # [B, H, D]
        scale,
        stride_q_b, stride_q_h, stride_q_d,
        stride_kc_block, stride_kc_h, stride_kc_d, stride_kc_blk,
        stride_bt_b, stride_bt_n,
        HEAD_DIM: tl.constexpr,
        NUM_QUERY_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        MAX_NUM_BLOCKS: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        kv_head_idx = head_idx * NUM_KV_HEADS // NUM_QUERY_HEADS  # GQA group lookup

        ctx_len = tl.load(context_lens_ptr + batch_idx)

        # Load Q vector [HEAD_DIM]
        q = tl.load(q_ptr
                    + batch_idx * stride_q_b
                    + head_idx * stride_q_h
                    + tl.arange(0, HEAD_DIM) * stride_q_d).to(tl.float32)
        q = q * scale

        # Online softmax accumulators
        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        d_offsets = tl.arange(0, HEAD_DIM)
        b_offsets = tl.arange(0, BLOCK_SIZE)

        for blk_idx in range(0, MAX_NUM_BLOCKS):
            # Stop if we've covered all of ctx
            block_start = blk_idx * BLOCK_SIZE
            if block_start >= ctx_len:
                break
            # Load block_id from block_table
            block_id = tl.load(block_table_ptr + batch_idx * stride_bt_b + blk_idx * stride_bt_n)

            # Load K block: shape [HEAD_DIM, BLOCK_SIZE]
            kc_base = (kc_ptr
                       + block_id * stride_kc_block
                       + kv_head_idx * stride_kc_h)
            k_block = tl.load(kc_base
                              + d_offsets[:, None] * stride_kc_d
                              + b_offsets[None, :] * stride_kc_blk).to(tl.float32)
            # Same for V
            vc_base = (vc_ptr
                       + block_id * stride_kc_block
                       + kv_head_idx * stride_kc_h)
            v_block = tl.load(vc_base
                              + d_offsets[:, None] * stride_kc_d
                              + b_offsets[None, :] * stride_kc_blk).to(tl.float32)
            # v_block is [HEAD_DIM, BLOCK_SIZE]; we want [BLOCK_SIZE, HEAD_DIM]
            # for the acc update; just transpose conceptually via the reduction axes.

            # Scores: q @ K → [BLOCK_SIZE]
            scores = tl.sum(q[:, None] * k_block, axis=0)

            # Mask invalid positions (block tail past ctx_len)
            block_pos = block_start + b_offsets
            mask = block_pos < ctx_len
            scores = tl.where(mask, scores, float("-inf"))

            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            # If m_new is -inf (all masked), guard.
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            p = tl.where(mask, p, 0.0)
            l_new = alpha * l_i + tl.sum(p, axis=0)

            # acc update: acc = alpha * acc + sum_b p[b] * v_block[:, b]
            acc = alpha * acc + tl.sum(p[None, :] * v_block, axis=1)
            m_i = m_new
            l_i = l_new

        out = acc / l_i
        # Cast back to original dtype if needed; we just store as fp32 here and
        # let the caller's dtype promotion handle it. To match TorchBackend
        # behavior we store in q's dtype.
        tl.store(out_ptr
                 + batch_idx * stride_q_b
                 + head_idx * stride_q_h
                 + d_offsets * stride_q_d,
                 out.to(q_ptr.dtype.element_ty))


# ============================================================================
# Kernel 3: paged prefill (ragged, per-token)
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _paged_prefill_kernel(
        q_ptr,                       # [N_total, H, D]
        kc_ptr, vc_ptr,              # paged cache
        block_table_ptr,             # [B, max_blocks]
        seq_lens_ptr,                # [B]   full ctx after step
        query_lens_ptr,              # [B]   chunk size per seq
        token_to_batch_ptr,          # [N_total] which seq each token belongs to
        token_to_chunk_pos_ptr,      # [N_total] position within chunk (0..query_lens[b]-1)
        out_ptr,
        scale,
        stride_q_n, stride_q_h, stride_q_d,
        stride_kc_block, stride_kc_h, stride_kc_d, stride_kc_blk,
        stride_bt_b, stride_bt_n,
        HEAD_DIM: tl.constexpr,
        NUM_QUERY_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        MAX_NUM_BLOCKS: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        kv_head_idx = head_idx * NUM_KV_HEADS // NUM_QUERY_HEADS

        batch_idx = tl.load(token_to_batch_ptr + token_idx)
        chunk_pos = tl.load(token_to_chunk_pos_ptr + token_idx)
        seq_len = tl.load(seq_lens_ptr + batch_idx)
        query_len = tl.load(query_lens_ptr + batch_idx)
        n_cached = seq_len - query_len
        # Absolute position of this query token: n_cached + chunk_pos.
        abs_pos = n_cached + chunk_pos

        d_offsets = tl.arange(0, HEAD_DIM)
        b_offsets = tl.arange(0, BLOCK_SIZE)

        q = tl.load(q_ptr
                    + token_idx * stride_q_n
                    + head_idx * stride_q_h
                    + d_offsets * stride_q_d).to(tl.float32)
        q = q * scale

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        for blk_idx in range(0, MAX_NUM_BLOCKS):
            block_start = blk_idx * BLOCK_SIZE
            if block_start >= seq_len:
                break
            block_id = tl.load(block_table_ptr + batch_idx * stride_bt_b + blk_idx * stride_bt_n)

            kc_base = (kc_ptr + block_id * stride_kc_block + kv_head_idx * stride_kc_h)
            vc_base = (vc_ptr + block_id * stride_kc_block + kv_head_idx * stride_kc_h)
            k_block = tl.load(kc_base
                              + d_offsets[:, None] * stride_kc_d
                              + b_offsets[None, :] * stride_kc_blk).to(tl.float32)
            v_block = tl.load(vc_base
                              + d_offsets[:, None] * stride_kc_d
                              + b_offsets[None, :] * stride_kc_blk).to(tl.float32)

            scores = tl.sum(q[:, None] * k_block, axis=0)

            block_pos = block_start + b_offsets
            # Causal: this query (abs_pos) attends to kv positions [0, abs_pos].
            # Also bound by seq_len (don't read past valid ctx).
            mask = (block_pos <= abs_pos) & (block_pos < seq_len)
            scores = tl.where(mask, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            p = tl.where(mask, p, 0.0)
            l_new = alpha * l_i + tl.sum(p, axis=0)
            acc = alpha * acc + tl.sum(p[None, :] * v_block, axis=1)
            m_i = m_new
            l_i = l_new

        out = acc / l_i
        tl.store(out_ptr
                 + token_idx * stride_q_n
                 + head_idx * stride_q_h
                 + d_offsets * stride_q_d,
                 out.to(q_ptr.dtype.element_ty))


# ============================================================================
# Backend class
# ============================================================================

class TritonBackend:
    """GPU paged-attention backend. Same interface as TorchBackend.

    Requires:
      * `triton` package installed
      * tensors on a CUDA device
      * head_dim is a power of 2 (Triton kernel constexpr requirement)
      * `num_query_heads % num_kv_heads == 0`
    """

    def __init__(self):
        if not HAS_TRITON:
            raise RuntimeError(
                "Triton not installed. Install with `pip install triton` "
                "and run on a CUDA-capable machine.")

    def reshape_and_cache(self, key, value, key_cache, value_cache, slot_mapping):
        N, H_kv, D = key.shape
        block_size = key_cache.shape[3]
        grid = (N, H_kv)
        _reshape_and_cache_kernel[grid](
            key, value, key_cache, value_cache, slot_mapping,
            key.stride(0), key.stride(1), key.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            NUM_KV_HEADS=H_kv, HEAD_DIM=D, BLOCK_SIZE=block_size,
        )

    def decode(self, q, key_cache, value_cache, block_table, context_lens, scale):
        B, H, D = q.shape
        H_kv = key_cache.shape[1]
        block_size = key_cache.shape[3]
        max_num_blocks = block_table.shape[1]
        out = torch.empty_like(q)
        grid = (B, H)
        _paged_decode_kernel[grid](
            q, key_cache, value_cache, block_table, context_lens, out,
            scale,
            q.stride(0), q.stride(1), q.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            block_table.stride(0), block_table.stride(1),
            HEAD_DIM=D, NUM_QUERY_HEADS=H, NUM_KV_HEADS=H_kv,
            BLOCK_SIZE=block_size, MAX_NUM_BLOCKS=max_num_blocks,
        )
        return out

    def prefill(self, q, key_cache, value_cache, block_table, seq_lens, query_lens, scale):
        N, H, D = q.shape
        H_kv = key_cache.shape[1]
        block_size = key_cache.shape[3]
        max_num_blocks = block_table.shape[1]

        # Build per-token batch_idx and chunk_pos arrays. Cheap CPU op when N is small;
        # could be precomputed in ModelRunner for hot path.
        token_to_batch = torch.empty(N, dtype=torch.int32, device=q.device)
        token_to_chunk_pos = torch.empty(N, dtype=torch.int32, device=q.device)
        cursor = 0
        ql_cpu = query_lens.tolist() if query_lens.is_cuda else query_lens.cpu().tolist()
        for b, ql in enumerate(ql_cpu):
            token_to_batch[cursor:cursor + ql] = b
            token_to_chunk_pos[cursor:cursor + ql] = torch.arange(ql, dtype=torch.int32, device=q.device)
            cursor += ql

        out = torch.empty_like(q)
        grid = (N, H)
        _paged_prefill_kernel[grid](
            q, key_cache, value_cache, block_table,
            seq_lens, query_lens,
            token_to_batch, token_to_chunk_pos,
            out, scale,
            q.stride(0), q.stride(1), q.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            block_table.stride(0), block_table.stride(1),
            HEAD_DIM=D, NUM_QUERY_HEADS=H, NUM_KV_HEADS=H_kv,
            BLOCK_SIZE=block_size, MAX_NUM_BLOCKS=max_num_blocks,
        )
        return out
