from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.checkpoints import SessionLedger
from deepagents_coding_platform.runtime import RuntimeKernel, RuntimeResult


Executor = Callable[[ActionRequest], Mapping[str, object]]


@dataclass(slots=True)
class LocalRunner:
    workspace_root: Path
    ledger_root: Path
    kernel: RuntimeKernel
    executors: dict[ActionKind, Executor] = field(default_factory=dict)
    ledger: SessionLedger = field(init=False)

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root)
        self.ledger_root = Path(self.ledger_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.ledger = SessionLedger(self.ledger_root)

    def run_action(self, action: ActionRequest) -> RuntimeResult:
        executor = self.executors.get(action.kind, self._default_executor)
        return self.kernel.handle(action=action, executor=executor, ledger=self.ledger)

    def _default_executor(self, action: ActionRequest) -> Mapping[str, object]:
        return {
            "action": action.name,
            "kind": action.kind.value,
            "payload": dict(action.payload),
        }
