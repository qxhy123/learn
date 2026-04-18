from deepagents_coding_platform.events import EventPhase, ProjectionTag, RuntimeEvent
from deepagents_coding_platform.projection import Audience, VisibilityProjector


def make_runtime_event(*, projection_tags: tuple[ProjectionTag, ...]) -> RuntimeEvent:
    return RuntimeEvent(
        event_id="evt-1",
        session_id="session-1",
        run_id="run-1",
        parent_event_id=None,
        actor="primary_agent",
        event_type="tool_call",
        phase=EventPhase.COMPLETED,
        raw_payload={"secret": "token", "path": "README.md"},
        redacted_payload={"path": "README.md"},
        summary_payload={"message": "read README.md"},
        projection_tags=projection_tags,
    )


def test_cli_gets_redacted_payload():
    event = make_runtime_event(projection_tags=(ProjectionTag.USER_VISIBLE,))

    projected = VisibilityProjector().project(event)

    assert projected[Audience.CLI].payload == {"path": "README.md"}


def test_control_plane_gets_summary_payload():
    event = make_runtime_event(projection_tags=(ProjectionTag.AUDIT_VISIBLE,))

    projected = VisibilityProjector().project(event)

    assert projected[Audience.CONTROL_PLANE].payload == {"message": "read README.md"}


def test_local_debug_gets_raw_payload():
    event = make_runtime_event(projection_tags=())

    projected = VisibilityProjector().project(event)

    assert projected[Audience.LOCAL_DEBUG].payload == {
        "secret": "token",
        "path": "README.md",
    }


def test_parent_projection_omitted_when_parent_visible_tag_absent():
    event = make_runtime_event(projection_tags=(ProjectionTag.USER_VISIBLE,))

    projected = VisibilityProjector().project(event)

    assert Audience.PARENT not in projected
