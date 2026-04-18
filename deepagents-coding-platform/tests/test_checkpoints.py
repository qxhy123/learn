from deepagents_coding_platform.checkpoints import SessionLedger
from deepagents_coding_platform.events import EventPhase, ProjectionTag, RuntimeEvent


def make_event(event_id: str) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=event_id,
        session_id="session-1",
        run_id="run-1",
        parent_event_id=None,
        actor="primary_agent",
        event_type="tool_call",
        phase=EventPhase.COMPLETED,
        raw_payload={"index": event_id},
        redacted_payload={"index": event_id},
        summary_payload={"index": event_id},
        projection_tags=(ProjectionTag.REPLAY_REQUIRED,),
    )


def test_resume_uses_latest_checkpoint_and_following_events(tmp_path):
    ledger = SessionLedger(tmp_path)

    ledger.append_event(make_event("evt-1"))
    ledger.commit_checkpoint("cp-1", {"cursor": 1})
    ledger.append_event(make_event("evt-2"))

    resumed = ledger.resume()

    assert resumed.checkpoint_name == "cp-1"
    assert resumed.state["cursor"] == 1
    assert [event.event_id for event in resumed.events_after_checkpoint] == ["evt-2"]
