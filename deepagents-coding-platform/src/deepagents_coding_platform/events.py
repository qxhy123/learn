from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class EventPhase(StrEnum):
    PROPOSED = "proposed"
    ALLOWED = "allowed"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"
    REQUIRES_APPROVAL = "requires_approval"
    REDACTED = "redacted"


class ProjectionTag(StrEnum):
    USER_VISIBLE = "user_visible"
    PARENT_VISIBLE = "parent_visible"
    AUDIT_VISIBLE = "audit_visible"
    LOCAL_ONLY = "local_only"
    REPLAY_REQUIRED = "replay_required"


@dataclass(slots=True, frozen=True)
class RuntimeEvent:
    event_id: str
    session_id: str
    run_id: str
    parent_event_id: str | None
    actor: str
    event_type: str
    phase: EventPhase
    raw_payload: Mapping[str, Any] = field(default_factory=dict)
    redacted_payload: Mapping[str, Any] = field(default_factory=dict)
    summary_payload: Mapping[str, Any] = field(default_factory=dict)
    projection_tags: tuple[ProjectionTag, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "parent_event_id": self.parent_event_id,
            "actor": self.actor,
            "event_type": self.event_type,
            "phase": self.phase.value,
            "raw_payload": dict(self.raw_payload),
            "redacted_payload": dict(self.redacted_payload),
            "summary_payload": dict(self.summary_payload),
            "projection_tags": [tag.value for tag in self.projection_tags],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeEvent":
        return cls(
            event_id=str(value["event_id"]),
            session_id=str(value["session_id"]),
            run_id=str(value["run_id"]),
            parent_event_id=value.get("parent_event_id"),
            actor=str(value["actor"]),
            event_type=str(value["event_type"]),
            phase=EventPhase(str(value["phase"])),
            raw_payload=dict(value.get("raw_payload", {})),
            redacted_payload=dict(value.get("redacted_payload", {})),
            summary_payload=dict(value.get("summary_payload", {})),
            projection_tags=tuple(
                ProjectionTag(tag) for tag in value.get("projection_tags", [])
            ),
        )
