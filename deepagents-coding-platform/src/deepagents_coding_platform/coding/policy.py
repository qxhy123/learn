from dataclasses import dataclass, field
from pathlib import Path

from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.policy import (
    PolicyDecision,
    PolicyOutcome,
    StaticPolicyEvaluator,
)


@dataclass(slots=True)
class WorkspaceCodingPolicy:
    workspace_root: Path
    blocked_shell_prefixes: tuple[str, ...] = (
        "rm -rf",
        "sudo",
        "shutdown",
        "reboot",
        "mkfs",
        "dd ",
        "git push",
    )
    fallback_policy: StaticPolicyEvaluator = field(default_factory=StaticPolicyEvaluator)

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()

    def evaluate(self, action: ActionRequest) -> PolicyDecision:
        if action.kind in {ActionKind.FS_READ, ActionKind.FS_WRITE}:
            raw_path = str(action.payload.get("path", "."))
            resolved = self._resolve_workspace_path(raw_path)
            if not resolved.is_relative_to(self.workspace_root):
                return PolicyDecision(
                    outcome=PolicyOutcome.DENY,
                    reason=f"path escapes workspace: {raw_path}",
                )
            return PolicyDecision(
                outcome=PolicyOutcome.ALLOW,
                reason=f"{action.kind.value} is allowed inside workspace",
            )

        if action.kind is ActionKind.SHELL_EXEC:
            command = str(action.payload.get("command", "")).strip()
            normalized = " ".join(command.lower().split())

            if command.startswith(self.blocked_shell_prefixes):
                return PolicyDecision(
                    outcome=PolicyOutcome.DENY,
                    reason=f"shell command is blocked by prefix rule: {command}",
                )

            if (
                (normalized.startswith("curl ") or normalized.startswith("wget "))
                and ("| sh" in normalized or "| bash" in normalized)
            ):
                return PolicyDecision(
                    outcome=PolicyOutcome.DENY,
                    reason=f"download-and-execute command is blocked: {command}",
                )

            return PolicyDecision(
                outcome=PolicyOutcome.ALLOW,
                reason="workspace-local shell command allowed",
            )

        return self.fallback_policy.evaluate(action)

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        return candidate.resolve()
