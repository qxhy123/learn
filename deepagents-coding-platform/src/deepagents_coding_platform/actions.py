from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class ActionKind(StrEnum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    SHELL_EXEC = "shell_exec"
    FS_READ = "fs_read"
    FS_WRITE = "fs_write"
    SUBAGENT_HANDOFF = "subagent_handoff"
    APPROVAL = "approval"


@dataclass(slots=True, frozen=True)
class ActionRequest:
    kind: ActionKind
    name: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    actor: str = "primary_agent"
