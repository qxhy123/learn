from mini_vllm.config import SamplingParams
from mini_vllm.sequence import Sequence, SequenceStatus
from mini_vllm.block_manager import BlockManager
from mini_vllm.scheduler import Scheduler


def make_seq(rid, prompt_len, max_tokens=4):
    return Sequence(rid, list(range(prompt_len)), SamplingParams(max_tokens=max_tokens))


def test_initial_admit_runs_prefill():
    bm = BlockManager(num_blocks=8, block_size=4)
    sched = Scheduler(bm)
    s1 = make_seq("a", 5); s2 = make_seq("b", 3)
    sched.add(s1); sched.add(s2)
    out = sched.schedule()
    assert {s.request_id for s in out.prefill_seqs} == {"a", "b"}
    assert out.decode_seqs == []
    assert s1.status == SequenceStatus.RUNNING
    assert s2.status == SequenceStatus.RUNNING


def test_after_prefill_marked_steps_become_decode():
    bm = BlockManager(num_blocks=8, block_size=4)
    sched = Scheduler(bm)
    s = make_seq("a", 4)
    sched.add(s)
    sched.schedule()
    sched.mark_prefilled(s)
    out2 = sched.schedule()
    assert out2.prefill_seqs == []
    assert [x.request_id for x in out2.decode_seqs] == ["a"]


def test_no_admission_until_running_drained():
    bm = BlockManager(num_blocks=8, block_size=4)
    sched = Scheduler(bm)
    s1 = make_seq("a", 4)
    sched.add(s1)
    sched.schedule()
    sched.mark_prefilled(s1)
    # Add s2 while s1 is decoding — should NOT be admitted in Plan 1
    s2 = make_seq("b", 4)
    sched.add(s2)
    out = sched.schedule()
    assert [x.request_id for x in out.prefill_seqs] == []
    assert [x.request_id for x in out.decode_seqs] == ["a"]
    assert s2.status == SequenceStatus.WAITING
