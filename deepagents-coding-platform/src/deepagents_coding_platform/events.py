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
