from dataclasses import dataclass

import httpx

from deepagents_coding_platform.events import RuntimeEvent


@dataclass(slots=True)
class ControlPlaneEventClient:
    base_url: str
    http_client: httpx.Client | None = None

    def upload(self, event: RuntimeEvent) -> httpx.Response:
        client = self.http_client or httpx.Client(base_url=self.base_url, timeout=5.0)
        payload = {
            "event_id": event.event_id,
            "session_id": event.session_id,
            "run_id": event.run_id,
            "event_type": event.event_type,
            "phase": event.phase.value,
            "payload": dict(event.summary_payload or event.redacted_payload),
            "projection_tags": [tag.value for tag in event.projection_tags],
        }
        response = client.post(f"{self.base_url}/v1/runtime-events", json=payload)
        response.raise_for_status()
        return response
