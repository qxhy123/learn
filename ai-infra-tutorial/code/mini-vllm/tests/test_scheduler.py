from mini_vllm.config import SamplingParams
from mini_vllm.sequence import Sequence, SequenceStatus
from mini_vllm.block_manager import BlockManager
from mini_vllm.scheduler import Scheduler


def make_seq(rid, prompt_len, max_tokens=4):
    return Sequence(rid, list(range(prompt_len)), SamplingParams(max_tokens=max_tokens))


# ---------------------------------------------------------------------------
# Behaviors common to both modes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Continuous batching ON (default since Plan 4)
# ---------------------------------------------------------------------------

def test_continuous_batching_admits_during_decode():
    """Plan 4 default: a new request can be admitted on the same step where
    other requests are already decoding."""
    bm = BlockManager(num_blocks=8, block_size=4)
    sched = Scheduler(bm)  # enable_continuous_batching=True by default
    s1 = make_seq("a", 4)
    sched.add(s1)
    sched.schedule()
    sched.mark_prefilled(s1)
    # Add s2 while s1 is in decode — should be admitted same step.
    s2 = make_seq("b", 4)
    sched.add(s2)
    out = sched.schedule()
    assert [x.request_id for x in out.prefill_seqs] == ["b"]
    assert [x.request_id for x in out.decode_seqs] == ["a"]
    assert s2.status == SequenceStatus.RUNNING


def test_token_budget_caps_admission():
    """Total tokens (prefill + decode) per step must not exceed
    max_num_batched_tokens. With budget=10, admitting a 5-token waiting seq
    while 1 decode + 1 prefill seq already use 4+1=5 tokens leaves 5 left
    in budget — fits exactly. A second 5-token waiting seq does NOT fit."""
    bm = BlockManager(num_blocks=16, block_size=4)
    sched = Scheduler(bm, max_num_batched_tokens=10)
    s1 = make_seq("a", 4)
    sched.add(s1)
    sched.schedule(); sched.mark_prefilled(s1)
    # Now: running has s1 (decode, 1 token). Add a 4-prompt s2 + 5-prompt s3 + 5-prompt s4.
    s2 = make_seq("b", 4)
    s3 = make_seq("c", 5)
    s4 = make_seq("d", 5)
    sched.add(s2); sched.add(s3); sched.add(s4)
    out = sched.schedule()
    # Budget = 10 - 1 (s1 decode) - 4 (s2 prefill) = 5 left → s3 fits (5).
    # Then 0 left → s4 stays waiting.
    admitted = {x.request_id for x in out.prefill_seqs}
    assert admitted == {"b", "c"}
    assert s4.status == SequenceStatus.WAITING


def test_admission_skips_oversized_request():
    """A waiting request whose prompt exceeds the remaining budget should be
    held; smaller subsequent requests should NOT jump it (FCFS)."""
    bm = BlockManager(num_blocks=8, block_size=4)
    sched = Scheduler(bm, max_num_batched_tokens=4)
    big = make_seq("big", 5)   # exceeds budget alone
    small = make_seq("sm", 3)
    sched.add(big); sched.add(small)
    out = sched.schedule()
    # `big` blocks the queue head; `small` waits behind it (FCFS).
    assert out.prefill_seqs == []
    assert big.status == SequenceStatus.WAITING
    assert small.status == SequenceStatus.WAITING


# ---------------------------------------------------------------------------
# Continuous batching OFF (Plan 1 baseline; opt-in for benchmarks)
# ---------------------------------------------------------------------------

def test_admission_blocked_when_continuous_batching_disabled():
    """When `enable_continuous_batching=False`, scheduler matches Plan 1
    behavior: no admission while running queue is non-empty."""
    bm = BlockManager(num_blocks=8, block_size=4)
    sched = Scheduler(bm, enable_continuous_batching=False)
    s1 = make_seq("a", 4)
    sched.add(s1)
    sched.schedule(); sched.mark_prefilled(s1)
    s2 = make_seq("b", 4)
    sched.add(s2)
    out = sched.schedule()
    assert [x.request_id for x in out.prefill_seqs] == []
    assert [x.request_id for x in out.decode_seqs] == ["a"]
    assert s2.status == SequenceStatus.WAITING
