from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from deepagents_coding_platform.events import ProjectionTag, RuntimeEvent


class Audience(StrEnum):
    CLI = "cli"
    PARENT = "parent"
    CONTROL_PLANE = "control_plane"
    LOCAL_DEBUG = "local_debug"


@dataclass(slots=True, frozen=True)
class ProjectedEvent:
    audience: Audience
    event_id: str
    phase: str
    payload: Mapping[str, Any]


class VisibilityProjector:
    def project(self, event: RuntimeEvent) -> dict[Audience, ProjectedEvent]:
        projections: dict[Audience, ProjectedEvent] = {
            Audience.LOCAL_DEBUG: ProjectedEvent(
                audience=Audience.LOCAL_DEBUG,
                event_id=event.event_id,
                phase=event.phase.value,
                payload=event.raw_payload,
            )
        }

        if ProjectionTag.USER_VISIBLE in event.projection_tags:
            projections[Audience.CLI] = ProjectedEvent(
                audience=Audience.CLI,
                event_id=event.event_id,
                phase=event.phase.value,
                payload=event.redacted_payload or event.summary_payload,
            )

        if ProjectionTag.PARENT_VISIBLE in event.projection_tags:
            projections[Audience.PARENT] = ProjectedEvent(
                audience=Audience.PARENT,
                event_id=event.event_id,
                phase=event.phase.value,
                payload=event.summary_payload,
            )

        if ProjectionTag.AUDIT_VISIBLE in event.projection_tags:
            projections[Audience.CONTROL_PLANE] = ProjectedEvent(
                audience=Audience.CONTROL_PLANE,
                event_id=event.event_id,
                phase=event.phase.value,
                payload=event.summary_payload or event.redacted_payload,
            )

        return projections
