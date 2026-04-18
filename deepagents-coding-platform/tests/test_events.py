from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.events import EventPhase, ProjectionTag, RuntimeEvent


def test_action_request_captures_kind_name_and_payload():
    action = ActionRequest(
        kind=ActionKind.TOOL_CALL,
        name="read_file",
        payload={"path": "README.md"},
        actor="primary_agent",
    )

    assert action.kind is ActionKind.TOOL_CALL
    assert action.payload["path"] == "README.md"


def test_runtime_event_preserves_lineage_and_projection_tags():
    event = RuntimeEvent(
        event_id="evt-2",
        session_id="session-1",
        run_id="run-1",
        parent_event_id="evt-1",
        actor="primary_agent",
        event_type="tool_call",
        phase=EventPhase.PROPOSED,
        raw_payload={"path": "README.md"},
        redacted_payload={"path": "README.md"},
        summary_payload={"path": "README.md"},
        projection_tags=(
            ProjectionTag.USER_VISIBLE,
            ProjectionTag.AUDIT_VISIBLE,
            ProjectionTag.REPLAY_REQUIRED,
        ),
    )

    assert event.parent_event_id == "evt-1"
    assert ProjectionTag.AUDIT_VISIBLE in event.projection_tags
    assert event.phase is EventPhase.PROPOSED
