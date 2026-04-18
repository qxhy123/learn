import httpx

from deepagents_coding_platform.control_plane import ControlPlaneEventClient
from deepagents_coding_platform.events import EventPhase, ProjectionTag, RuntimeEvent


def test_control_plane_client_uploads_summary_payload_only():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = request.content.decode("utf-8")
        return httpx.Response(202, json={"accepted": True})

    client = ControlPlaneEventClient(
        base_url="https://control-plane.example",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    event = RuntimeEvent(
        event_id="evt-9",
        session_id="session-1",
        run_id="run-1",
        parent_event_id=None,
        actor="primary_agent",
        event_type="shell_exec",
        phase=EventPhase.COMPLETED,
        raw_payload={"command": "cat .env", "stdout": "SECRET=1"},
        redacted_payload={"command": "cat .env", "stdout": "[redacted]"},
        summary_payload={"command": "cat .env", "status": "completed"},
        projection_tags=(ProjectionTag.AUDIT_VISIBLE,),
    )

    response = client.upload(event)

    assert response.status_code == 202
    assert "SECRET=1" not in captured["json"]
    assert '"status":"completed"' in captured["json"]
