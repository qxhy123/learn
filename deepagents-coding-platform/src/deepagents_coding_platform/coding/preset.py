from pathlib import Path

from deepagents_coding_platform.actions import ActionKind
from deepagents_coding_platform.adapters.deepagents_runtime import (
    DeepagentsRuntimeAdapter,
    RuntimeToolSpec,
)
from deepagents_coding_platform.coding.executors import build_coding_named_executors
from deepagents_coding_platform.coding.policy import WorkspaceCodingPolicy
from deepagents_coding_platform.plugins import PluginRegistry
from deepagents_coding_platform.runner import LocalRunner
from deepagents_coding_platform.runtime import RuntimeKernel


CODING_SYSTEM_PROMPT = """
You are a workspace-scoped coding agent.

- Explore before modifying.
- Prefer list_files, grep_search, and read_file before editing.
- Prefer apply_patch for local edits.
- Use write_file for new files and full rewrites.
- Run validation with shell after edits.
- Operate only inside the workspace.
- Do not claim success unless a tool returned success.
- If blocked by policy, explain the block and try a safer alternative.
""".strip()


def build_coding_tool_specs() -> list[RuntimeToolSpec]:
    return [
        RuntimeToolSpec(
            name="read_file",
            description="Read a text file from the workspace.",
            action_kind=ActionKind.FS_READ,
        ),
        RuntimeToolSpec(
            name="write_file",
            description="Write or overwrite a workspace file.",
            action_kind=ActionKind.FS_WRITE,
        ),
        RuntimeToolSpec(
            name="list_files",
            description="List files and directories in the workspace.",
            action_kind=ActionKind.FS_READ,
        ),
        RuntimeToolSpec(
            name="grep_search",
            description="Search for matching text inside the workspace.",
            action_kind=ActionKind.FS_READ,
        ),
        RuntimeToolSpec(
            name="shell",
            description="Run a shell command inside the workspace.",
            action_kind=ActionKind.SHELL_EXEC,
        ),
        RuntimeToolSpec(
            name="apply_patch",
            description="Apply exact-context text replacements to a file.",
            action_kind=ActionKind.FS_WRITE,
        ),
    ]


def build_coding_runner(workspace_root: Path, ledger_root: Path) -> LocalRunner:
    workspace_root = Path(workspace_root)
    ledger_root = Path(ledger_root)
    plugins = PluginRegistry(
        policy_evaluators=[WorkspaceCodingPolicy(workspace_root=workspace_root)]
    )
    kernel = RuntimeKernel(
        session_id="session-1",
        run_id="run-1",
        plugins=plugins,
    )
    return LocalRunner(
        workspace_root=workspace_root,
        ledger_root=ledger_root,
        kernel=kernel,
        named_executors=build_coding_named_executors(workspace_root),
    )


def build_coding_agent(
    *,
    model: str,
    workspace_root: Path,
    ledger_root: Path,
):
    runner = build_coding_runner(
        workspace_root=workspace_root,
        ledger_root=ledger_root,
    )
    adapter = DeepagentsRuntimeAdapter(runner=runner)
    return adapter.create_agent(
        model=model,
        system_prompt=CODING_SYSTEM_PROMPT,
        tool_specs=build_coding_tool_specs(),
    )
