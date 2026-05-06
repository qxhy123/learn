import torch
from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.backends.reference import reference_decode, reference_prefill


def test_reshape_and_cache_writes_correct_slots():
    block_size = 4
    num_blocks = 4
    H_kv = 2
    D = 8
    kc = torch.zeros(num_blocks, H_kv, D, block_size)
    vc = torch.zeros(num_blocks, H_kv, D, block_size)
    # Two tokens: token 0 -> slot 5 (block 1, offset 1); token 1 -> slot 11 (block 2, offset 3)
    key = torch.randn(2, H_kv, D)
    val = torch.randn(2, H_kv, D)
    slot_mapping = torch.tensor([5, 11], dtype=torch.long)
    backend = TorchBackend()
    backend.reshape_and_cache(key, val, kc, vc, slot_mapping)
    # block 1 offset 1
    assert torch.allclose(kc[1, :, :, 1], key[0])
    assert torch.allclose(vc[1, :, :, 1], val[0])
    # block 2 offset 3
    assert torch.allclose(kc[2, :, :, 3], key[1])
    assert torch.allclose(vc[2, :, :, 3], val[1])
    # Other slots untouched
    assert (kc[0] == 0).all()
    assert (kc[3] == 0).all()


def test_torch_decode_matches_reference():
    torch.manual_seed(0)
    B, H, D, H_kv = 3, 8, 16, 2  # GQA: 4 query heads per kv head
    block_size = 4
    num_blocks = 16
    max_blocks_per_seq = 4
    kc = torch.randn(num_blocks, H_kv, D, block_size)
    vc = torch.randn(num_blocks, H_kv, D, block_size)
    q = torch.randn(B, H, D)
    block_table = torch.tensor([
        [0, 1, 2, 3],
        [4, 5, 6, 0],
        [7, 8, 0, 0],
    ], dtype=torch.int32)
    context_lens = torch.tensor([13, 10, 6], dtype=torch.int32)
    scale = D ** -0.5
    backend = TorchBackend()
    out = backend.decode(q, kc, vc, block_table, context_lens, scale)
    ref = reference_decode(q, kc, vc, block_table, context_lens, scale)
    assert torch.allclose(out, ref, atol=1e-5)


def _build_kv_cache_from_kv(k, v, seq_lens, block_size, num_blocks):
    """Helper: write [N, H_kv, D] K/V into a paged cache laid out as
    [num_blocks, H_kv, D, block_size]. Returns (kc, vc, block_table)."""
    H_kv, D = k.shape[1], k.shape[2]
    kc = torch.zeros(num_blocks, H_kv, D, block_size)
    vc = torch.zeros(num_blocks, H_kv, D, block_size)
    block_table_rows = []
    cursor = 0
    next_block = 0
    max_blocks = max((int(s.item()) + block_size - 1) // block_size for s in seq_lens)
    for s in seq_lens:
        n = int(s.item())
        seq_blocks = []
        for i in range(n):
            blk = next_block + i // block_size
            off = i % block_size
            kc[blk, :, :, off] = k[cursor + i]
            vc[blk, :, :, off] = v[cursor + i]
        seq_blocks = list(range(next_block, next_block + (n + block_size - 1) // block_size))
        next_block += len(seq_blocks)
        cursor += n
        seq_blocks = seq_blocks + [0] * (max_blocks - len(seq_blocks))
        block_table_rows.append(seq_blocks)
    return kc, vc, torch.tensor(block_table_rows, dtype=torch.int32)


def test_torch_prefill_matches_reference_and_is_causal():
    torch.manual_seed(0)
    H, D, H_kv = 8, 16, 2
    block_size = 4
    seq_lens = torch.tensor([5, 3], dtype=torch.int32)
    query_lens = torch.tensor([5, 3], dtype=torch.int32)
    N = 8
    q = torch.randn(N, H, D)
    k = torch.randn(N, H_kv, D)
    v = torch.randn(N, H_kv, D)
    scale = D ** -0.5
    kc, vc, bt = _build_kv_cache_from_kv(k, v, seq_lens, block_size, num_blocks=16)
    out = TorchBackend().prefill(q, kc, vc, bt, seq_lens, query_lens, scale)
    ref = reference_prefill(q, kc, vc, bt, seq_lens, query_lens, scale)
    assert torch.allclose(out, ref, atol=1e-5)
    # Causal sanity: position 0 of seq 0 attends only to itself.
    # Zero out cache slots for positions 1..4 of seq 0 (block 0 offsets 1..4 — but
    # block_size=4 so positions 1..3 are in block 0 offsets 1..3, position 4 is
    # in block 1 offset 0). Easier: rebuild cache with k/v[1:5] zeroed.
    k2 = k.clone(); v2 = v.clone()
    k2[1:5] = 0; v2[1:5] = 0
    kc2, vc2, bt2 = _build_kv_cache_from_kv(k2, v2, seq_lens, block_size, num_blocks=16)
    out2 = TorchBackend().prefill(q, kc2, vc2, bt2, seq_lens, query_lens, scale)
    assert torch.allclose(out[0], out2[0], atol=1e-5)


def test_prefill_chunked_matches_unchunked():
    """Chunked prefill must produce identical output to one-shot prefill.

    Two passes: (1) one-shot with seq_len=8, query_len=8.
    (2) chunked: first call seq_len=5, query_len=5 (cache positions 0..4),
        second call seq_len=8, query_len=3 (cache positions 5..7, attending
        to all of 0..7).

    Output of pass (2) for positions 5..7 should match pass (1) positions 5..7.
    """
    torch.manual_seed(1)
    H, D, H_kv = 4, 16, 2
    block_size = 4
    num_blocks = 8
    N = 8
    q = torch.randn(N, H, D)
    k = torch.randn(N, H_kv, D)
    v = torch.randn(N, H_kv, D)
    scale = D ** -0.5

    # Pass 1: one-shot
    seq_lens = torch.tensor([N], dtype=torch.int32)
    query_lens = torch.tensor([N], dtype=torch.int32)
    kc, vc, bt = _build_kv_cache_from_kv(k, v, seq_lens, block_size, num_blocks)
    out_full = TorchBackend().prefill(q, kc, vc, bt, seq_lens, query_lens, scale)

    # Pass 2: chunk 1 = first 5 tokens
    chunk1 = 5
    sl1 = torch.tensor([chunk1], dtype=torch.int32)
    ql1 = torch.tensor([chunk1], dtype=torch.int32)
    kc2, vc2, bt2 = _build_kv_cache_from_kv(k[:chunk1], v[:chunk1], sl1, block_size, num_blocks)
    _ = TorchBackend().prefill(q[:chunk1], kc2, vc2, bt2, sl1, ql1, scale)
    # Now write chunk 2 K/V into the same cache (positions 5..7).
    # In real use, reshape_and_cache would do this; we replicate manually.
    block_size_ = block_size
    for i in range(chunk1, N):
        blk = i // block_size_
        off = i % block_size_
        kc2[blk, :, :, off] = k[i]
        vc2[blk, :, :, off] = v[i]
    sl2 = torch.tensor([N], dtype=torch.int32)        # full ctx after chunk 2
    ql2 = torch.tensor([N - chunk1], dtype=torch.int32)
    # block_table now needs blocks for [0, N) = first ceil(8/4)=2 blocks
    bt_full = torch.tensor([[0, 1]], dtype=torch.int32)
    out_chunk2 = TorchBackend().prefill(q[chunk1:], kc2, vc2, bt_full, sl2, ql2, scale)
    # Compare positions 5..7 of one-shot to chunk 2 output.
    assert torch.allclose(out_full[chunk1:], out_chunk2, atol=1e-5)
