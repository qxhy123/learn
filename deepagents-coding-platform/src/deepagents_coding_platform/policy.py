from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from deepagents_coding_platform.actions import ActionKind, ActionRequest


class PolicyOutcome(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(slots=True, frozen=True)
class PolicyDecision:
    outcome: PolicyOutcome
    reason: str


class PolicyEvaluator(Protocol):
    def evaluate(self, action: ActionRequest) -> PolicyDecision: ...


@dataclass(slots=True)
class StaticPolicyEvaluator:
    blocked_shell_prefixes: tuple[str, ...] = ("rm -rf",)
    approval_shell_prefixes: tuple[str, ...] = ("git push",)
    auto_allow_kinds: set[ActionKind] = field(
        default_factory=lambda: {
            ActionKind.LLM_CALL,
            ActionKind.TOOL_CALL,
            ActionKind.FS_READ,
        }
    )

    def evaluate(self, action: ActionRequest) -> PolicyDecision:
        if action.kind is ActionKind.SHELL_EXEC:
            command = str(action.payload.get("command", ""))

            if command.startswith(self.blocked_shell_prefixes):
                return PolicyDecision(
                    outcome=PolicyOutcome.DENY,
                    reason=f"shell command is blocked by prefix rule: {command}",
                )

            if command.startswith(self.approval_shell_prefixes):
                return PolicyDecision(
                    outcome=PolicyOutcome.REQUIRE_APPROVAL,
                    reason=f"shell command requires approval: {command}",
                )

        if action.kind in {ActionKind.FS_WRITE, ActionKind.SUBAGENT_HANDOFF}:
            return PolicyDecision(
                outcome=PolicyOutcome.REQUIRE_APPROVAL,
                reason=f"{action.kind.value} requires approval",
            )

        if action.kind in self.auto_allow_kinds:
            return PolicyDecision(
                outcome=PolicyOutcome.ALLOW,
                reason=f"{action.kind.value} is auto-allowed",
            )

        return PolicyDecision(
            outcome=PolicyOutcome.REQUIRE_APPROVAL,
            reason=f"{action.kind.value} falls back to approval",
        )
