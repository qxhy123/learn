import pytest
from mini_vllm.block_manager import BlockManager, AllocStatus
from mini_vllm.sequence import Sequence
from mini_vllm.config import SamplingParams


def make_seq(rid: str, prompt_len: int) -> Sequence:
    return Sequence(request_id=rid, prompt_token_ids=list(range(prompt_len)),
                    sampling_params=SamplingParams(max_tokens=4))


def test_allocate_consumes_blocks():
    bm = BlockManager(num_blocks=4, block_size=8)
    seq = make_seq("r0", prompt_len=10)  # needs ceil(10/8) = 2 blocks
    assert bm.can_allocate(seq) == AllocStatus.OK
    bm.allocate(seq)
    assert seq.block_table is not None
    assert len(seq.block_table.physical_blocks) == 2
    assert bm.num_free_blocks == 2


def test_allocate_when_full():
    bm = BlockManager(num_blocks=2, block_size=8)
    seq = make_seq("r0", prompt_len=10)  # needs 2 blocks → fits exactly
    bm.allocate(seq)
    seq2 = make_seq("r1", prompt_len=4)  # needs 1, none free
    assert bm.can_allocate(seq2) == AllocStatus.LATER


def test_append_slot_within_last_block():
    bm = BlockManager(num_blocks=4, block_size=8)
    seq = make_seq("r0", prompt_len=4)  # 1 block, 4 slots used
    bm.allocate(seq)
    # simulate 3 more tokens — still in same block
    for _ in range(3):
        seq.output_token_ids.append(0)
        bm.append_slot(seq)
    assert len(seq.block_table.physical_blocks) == 1
    assert bm.num_free_blocks == 3


def test_append_slot_extends_blocks():
    bm = BlockManager(num_blocks=4, block_size=8)
    seq = make_seq("r0", prompt_len=8)  # 1 full block
    bm.allocate(seq)
    # next token forces a new block
    seq.output_token_ids.append(0)
    bm.append_slot(seq)
    assert len(seq.block_table.physical_blocks) == 2
    assert bm.num_free_blocks == 2


def test_free_returns_blocks():
    bm = BlockManager(num_blocks=4, block_size=8)
    seq = make_seq("r0", prompt_len=10)
    bm.allocate(seq)
    bm.free(seq)
    assert bm.num_free_blocks == 4
    assert seq.block_table is None


def test_invariant_no_block_leak_after_alloc_free_cycles():
    bm = BlockManager(num_blocks=8, block_size=4)
    for i in range(20):
        seq = make_seq(f"r{i}", prompt_len=5 + (i % 3))
        bm.allocate(seq)
        bm.free(seq)
    assert bm.num_free_blocks == 8


def test_slot_mapping_for_seq():
    bm = BlockManager(num_blocks=4, block_size=8)
    seq = make_seq("r0", prompt_len=10)
    bm.allocate(seq)
    # Slot mapping for the prompt: positions 0..9 → physical slots
    mapping = bm.get_slot_mapping(seq, start=0, end=10)
    assert len(mapping) == 10
    # Positions 0..7 map to block0; 8..9 to block1
    block_ids = [pb.block_id for pb in seq.block_table.physical_blocks]
    assert mapping[0] == block_ids[0] * 8 + 0
    assert mapping[7] == block_ids[0] * 8 + 7
    assert mapping[8] == block_ids[1] * 8 + 0
    assert mapping[9] == block_ids[1] * 8 + 1
