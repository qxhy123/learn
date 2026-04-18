from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from uuid import uuid4

from deepagents_coding_platform.actions import ActionRequest
from deepagents_coding_platform.checkpoints import SessionLedger
from deepagents_coding_platform.events import EventPhase, ProjectionTag, RuntimeEvent
from deepagents_coding_platform.plugins import PluginRegistry
from deepagents_coding_platform.policy import PolicyDecision, PolicyOutcome
from deepagents_coding_platform.projection import Audience, ProjectedEvent, VisibilityProjector


@dataclass(slots=True, frozen=True)
class RuntimeResult:
    decision: PolicyDecision
    events: list[RuntimeEvent]
    projections: dict[Audience, list[ProjectedEvent]]
    output: Mapping[str, Any] | None


@dataclass(slots=True)
class RuntimeKernel:
    session_id: str
    run_id: str
    plugins: PluginRegistry = field(default_factory=PluginRegistry)
    projector: VisibilityProjector = field(default_factory=VisibilityProjector)

    def handle(
        self,
        action: ActionRequest,
        executor: Callable[[ActionRequest], Mapping[str, Any]],
        ledger: SessionLedger,
    ) -> RuntimeResult:
        events: list[RuntimeEvent] = []
        projections: dict[Audience, list[ProjectedEvent]] = defaultdict(list)

        proposed = self._make_event(
            action=action,
            phase=EventPhase.PROPOSED,
            payload={"action": action.name, **dict(action.payload)},
        )
        self._record(ledger, proposed, events, projections)

        decision = self._evaluate(action)
        decision_event = self._make_event(
            action=action,
            phase={
                PolicyOutcome.ALLOW: EventPhase.ALLOWED,
                PolicyOutcome.DENY: EventPhase.DENIED,
                PolicyOutcome.REQUIRE_APPROVAL: EventPhase.REQUIRES_APPROVAL,
            }[decision.outcome],
            payload={
                "action": action.name,
                "decision": decision.outcome.value,
                "reason": decision.reason,
            },
        )
        self._record(ledger, decision_event, events, projections)

        if decision.outcome is not PolicyOutcome.ALLOW:
            ledger.commit_checkpoint(
                f"{action.kind.value}-{action.name}",
                {"last_action": action.name, "decision": decision.outcome.value},
            )
            return RuntimeResult(
                decision=decision,
                events=events,
                projections=dict(projections),
                output=None,
            )

        output = dict(executor(action))
        completed = self._make_event(
            action=action,
            phase=EventPhase.COMPLETED,
            payload={"action": action.name, **output},
        )
        self._record(ledger, completed, events, projections)
        ledger.commit_checkpoint(
            f"{action.kind.value}-{action.name}",
            {"last_action": action.name, "decision": decision.outcome.value},
        )

        return RuntimeResult(
            decision=decision,
            events=events,
            projections=dict(projections),
            output=output,
        )

    def _evaluate(self, action: ActionRequest) -> PolicyDecision:
        if not self.plugins.policy_evaluators:
            return PolicyDecision(
                outcome=PolicyOutcome.ALLOW,
                reason="no policy evaluators configured",
            )

        final = PolicyDecision(
            outcome=PolicyOutcome.ALLOW,
            reason="all evaluators allowed the action",
        )
        for evaluator in self.plugins.policy_evaluators:
            decision = evaluator.evaluate(action)
            if decision.outcome is not PolicyOutcome.ALLOW:
                return decision
            final = decision
        return final

    def _make_event(
        self,
        action: ActionRequest,
        phase: EventPhase,
        payload: Mapping[str, Any],
    ) -> RuntimeEvent:
        return RuntimeEvent(
            event_id=f"evt-{uuid4()}",
            session_id=self.session_id,
            run_id=self.run_id,
            parent_event_id=None,
            actor=action.actor,
            event_type=action.kind.value,
            phase=phase,
            raw_payload=dict(payload),
            redacted_payload=dict(payload),
            summary_payload={"action": action.name, "phase": phase.value},
            projection_tags=(
                ProjectionTag.USER_VISIBLE,
                ProjectionTag.AUDIT_VISIBLE,
                ProjectionTag.REPLAY_REQUIRED,
            ),
        )

    def _record(
        self,
        ledger: SessionLedger,
        event: RuntimeEvent,
        events: list[RuntimeEvent],
        projections: dict[Audience, list[ProjectedEvent]],
    ) -> None:
        ledger.append_event(event)
        events.append(event)
        for audience, projected in self.projector.project(event).items():
            projections[audience].append(projected)
