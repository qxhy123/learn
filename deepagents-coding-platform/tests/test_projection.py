from deepagents_coding_platform.events import EventPhase, ProjectionTag, RuntimeEvent
from deepagents_coding_platform.projection import Audience, VisibilityProjector


def test_projector_uses_payload_tiers_for_each_audience():
    event = RuntimeEvent(
        event_id="evt-1",
        session_id="session-1",
        run_id="run-1",
        parent_event_id=None,
        actor="primary_agent",
        event_type="shell_exec",
        phase=EventPhase.COMPLETED,
        raw_payload={"command": "cat .env", "stdout": "SECRET=1"},
        redacted_payload={"command": "cat .env", "stdout": "[redacted]"},
        summary_payload={"command": "cat .env", "status": "completed"},
        projection_tags=(
            ProjectionTag.USER_VISIBLE,
            ProjectionTag.AUDIT_VISIBLE,
            ProjectionTag.REPLAY_REQUIRED,
        ),
    )

    projections = VisibilityProjector().project(event)

    assert projections[Audience.CLI].payload["stdout"] == "[redacted]"
    assert projections[Audience.CONTROL_PLANE].payload["status"] == "completed"
    assert projections[Audience.LOCAL_DEBUG].payload["stdout"] == "SECRET=1"


def test_parent_projection_is_omitted_without_parent_visible_tag():
    event = RuntimeEvent(
        event_id="evt-2",
        session_id="session-1",
        run_id="run-1",
        parent_event_id=None,
        actor="primary_agent",
        event_type="tool_call",
        phase=EventPhase.COMPLETED,
        raw_payload={"path": "README.md"},
        redacted_payload={"path": "README.md"},
        summary_payload={"path": "README.md"},
        projection_tags=(ProjectionTag.USER_VISIBLE,),
    )

    projections = VisibilityProjector().project(event)

    assert Audience.PARENT not in projections
