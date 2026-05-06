from mini_vllm.sequence import Sequence, SequenceStatus
from mini_vllm.config import SamplingParams


def test_sequence_initial_state():
    seq = Sequence(request_id="r0", prompt_token_ids=[1, 2, 3, 4],
                   sampling_params=SamplingParams(max_tokens=8))
    assert seq.status == SequenceStatus.WAITING
    assert seq.num_prompt_tokens == 4
    assert seq.num_generated_tokens == 0
    assert seq.token_ids == [1, 2, 3, 4]
    assert not seq.is_finished()


def test_sequence_append_token():
    seq = Sequence(request_id="r0", prompt_token_ids=[1, 2],
                   sampling_params=SamplingParams(max_tokens=3))
    seq.append_token(99)
    assert seq.token_ids == [1, 2, 99]
    assert seq.num_generated_tokens == 1
    assert not seq.is_finished()
    seq.append_token(100)
    seq.append_token(101)
    assert seq.is_finished()  # max_tokens reached


def test_sequence_stop_token():
    seq = Sequence(request_id="r0", prompt_token_ids=[1],
                   sampling_params=SamplingParams(max_tokens=10, stop_token_ids=(7,)))
    seq.append_token(5)
    assert not seq.is_finished()
    seq.append_token(7)
    assert seq.is_finished()
