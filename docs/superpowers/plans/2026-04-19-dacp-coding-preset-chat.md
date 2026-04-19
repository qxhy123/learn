# DACP Coding Preset and Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a workspace-scoped coding preset with built-in coding tools and a `dacp chat` REPL on top of the existing runtime kernel.

**Architecture:** Extend the current `LocalRunner -> RuntimeKernel -> DeepagentsRuntimeAdapter` chain instead of replacing it. Add a coding-specific policy, a named executor set for six coding tools, a preset assembly layer, and a REPL that invokes a deepagents agent with in-memory message history.

**Tech Stack:** Python 3.12+, Typer, Rich, deepagents, pytest, subprocess, pathlib

---

## File Structure

- Modify: `deepagents-coding-platform/src/deepagents_coding_platform/runner.py`
  - add named executor dispatch so multiple tool names can share one `ActionKind`
- Modify: `deepagents-coding-platform/src/deepagents_coding_platform/adapters/deepagents_runtime.py`
  - surface blocked runtime decisions to tools as explicit failures
- Modify: `deepagents-coding-platform/src/deepagents_coding_platform/cli.py`
  - add `chat`
- Modify: `deepagents-coding-platform/README.md`
  - document coding preset usage
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/coding/__init__.py`
  - package exports for coding preset helpers
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/coding/policy.py`
  - workspace-safe deterministic coding policy
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/coding/executors.py`
  - built-in coding executors
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/coding/preset.py`
  - preset assembly for runner, tool specs, and deepagents agent
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/chat.py`
  - chat session and REPL loop
- Modify: `deepagents-coding-platform/tests/test_runner.py`
  - add named executor routing coverage
- Modify: `deepagents-coding-platform/tests/test_deepagents_adapter.py`
  - add blocked-tool failure coverage
- Create: `deepagents-coding-platform/tests/test_coding_policy.py`
  - policy tests
- Create: `deepagents-coding-platform/tests/test_coding_executors.py`
  - built-in coding executor tests
- Create: `deepagents-coding-platform/tests/test_coding_preset.py`
  - preset assembly tests
- Create: `deepagents-coding-platform/tests/test_chat.py`
  - REPL loop tests
- Create: `deepagents-coding-platform/tests/test_cli_chat.py`
  - CLI `chat` tests

---

### Task 1: Strengthen Runner Dispatch and Tool Failure Signaling

**Files:**
- Modify: `deepagents-coding-platform/src/deepagents_coding_platform/runner.py`
- Modify: `deepagents-coding-platform/src/deepagents_coding_platform/adapters/deepagents_runtime.py`
- Modify: `deepagents-coding-platform/tests/test_runner.py`
- Modify: `deepagents-coding-platform/tests/test_deepagents_adapter.py`
- Test: `deepagents-coding-platform/tests/test_runner.py`
- Test: `deepagents-coding-platform/tests/test_deepagents_adapter.py`

- [ ] **Step 1: Add failing routing and blocked-tool tests**

```python
# deepagents-coding-platform/tests/test_runner.py
from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.plugins import PluginRegistry
from deepagents_coding_platform.runner import LocalRunner
from deepagents_coding_platform.runtime import RuntimeKernel


def test_local_runner_routes_actions_to_registered_executors(tmp_path):
    kernel = RuntimeKernel(
        session_id="session-1",
        run_id="run-1",
        plugins=PluginRegistry(),
    )
    called = {"command": None}

    def shell_executor(action):
        called["command"] = action.payload["command"]
        return {"stdout": "ok"}

    runner = LocalRunner(
        workspace_root=tmp_path,
        ledger_root=tmp_path / ".ledger",
        kernel=kernel,
        executors={ActionKind.SHELL_EXEC: shell_executor},
    )

    result = runner.run_action(
        ActionRequest(
            kind=ActionKind.SHELL_EXEC,
            name="shell",
            payload={"command": "echo hello"},
        )
    )

    assert called["command"] == "echo hello"
    assert result.output["stdout"] == "ok"


def test_local_runner_prefers_named_executor_before_kind_executor(tmp_path):
    kernel = RuntimeKernel(
        session_id="session-1",
        run_id="run-1",
        plugins=PluginRegistry(),
    )
    called = {"route": []}

    def named_executor(action):
        called["route"].append(f"name:{action.name}")
        return {"source": "named"}

    def kind_executor(action):
        called["route"].append(f"kind:{action.kind.value}")
        return {"source": "kind"}

    runner = LocalRunner(
        workspace_root=tmp_path,
        ledger_root=tmp_path / ".ledger",
        kernel=kernel,
        executors={ActionKind.FS_READ: kind_executor},
        named_executors={"read_file": named_executor},
    )

    result = runner.run_action(
        ActionRequest(
            kind=ActionKind.FS_READ,
            name="read_file",
            payload={"path": "README.md"},
        )
    )

    assert called["route"] == ["name:read_file"]
    assert result.output["source"] == "named"
```

```python
# deepagents-coding-platform/tests/test_deepagents_adapter.py
import pytest

from deepagents_coding_platform.actions import ActionKind
from deepagents_coding_platform.adapters.deepagents_runtime import (
    DeepagentsRuntimeAdapter,
    RuntimeToolSpec,
)
from deepagents_coding_platform.policy import PolicyDecision, PolicyOutcome


def test_adapter_builds_runtime_wrapped_tools(monkeypatch):
    captured = {}

    def fake_create_deep_agent(*, model, tools, system_prompt):
        captured["model"] = model
        captured["tools"] = tools
        captured["system_prompt"] = system_prompt
        return "agent-object"

    monkeypatch.setattr(
        "deepagents_coding_platform.adapters.deepagents_runtime.create_deep_agent",
        fake_create_deep_agent,
    )

    class FakeRunner:
        def run_action(self, action):
            return type("Result", (), {"output": {"echo": action.payload["path"]}})()

    adapter = DeepagentsRuntimeAdapter(runner=FakeRunner())
    agent = adapter.create_agent(
        model="openai:gpt-4.1",
        system_prompt="You are helpful.",
        tool_specs=[
            RuntimeToolSpec(
                name="read_file",
                description="Read a file through the runtime kernel.",
                action_kind=ActionKind.FS_READ,
                static_payload={"path": "README.md"},
            )
        ],
    )

    assert agent == "agent-object"
    assert captured["model"] == "openai:gpt-4.1"
    assert len(captured["tools"]) == 1


def test_adapter_raises_when_runtime_blocks_action():
    class FakeRunner:
        def run_action(self, action):
            return type(
                "Result",
                (),
                {
                    "decision": PolicyDecision(
                        outcome=PolicyOutcome.DENY,
                        reason="path escapes workspace",
                    ),
                    "output": None,
                },
            )()

    adapter = DeepagentsRuntimeAdapter(runner=FakeRunner())
    tool = adapter.build_tool(
        RuntimeToolSpec(
            name="read_file",
            description="Read a file through the runtime kernel.",
            action_kind=ActionKind.FS_READ,
        )
    )

    with pytest.raises(RuntimeError, match="path escapes workspace"):
        tool(path="../secret.txt")
```

- [ ] **Step 2: Run the targeted tests and verify the new coverage fails**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_runner.py tests/test_deepagents_adapter.py -q
```

Expected:

- `test_local_runner_prefers_named_executor_before_kind_executor` fails because `LocalRunner` does not accept `named_executors`
- `test_adapter_raises_when_runtime_blocks_action` fails because the adapter currently returns `{}` instead of raising

- [ ] **Step 3: Implement named executor routing and blocked-tool failure propagation**

```python
# deepagents-coding-platform/src/deepagents_coding_platform/runner.py
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
    named_executors: dict[str, Executor] = field(default_factory=dict)
    ledger: SessionLedger = field(init=False)

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root)
        self.ledger_root = Path(self.ledger_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.ledger = SessionLedger(self.ledger_root)

    def run_action(self, action: ActionRequest) -> RuntimeResult:
        executor = self.named_executors.get(action.name)
        if executor is None:
            executor = self.executors.get(action.kind, self._default_executor)
        return self.kernel.handle(action=action, executor=executor, ledger=self.ledger)

    def _default_executor(self, action: ActionRequest) -> Mapping[str, object]:
        return {
            "action": action.name,
            "kind": action.kind.value,
            "payload": dict(action.payload),
        }
```

```python
# deepagents-coding-platform/src/deepagents_coding_platform/adapters/deepagents_runtime.py
from dataclasses import dataclass, field
from typing import Any, Mapping

from deepagents import create_deep_agent

from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.policy import PolicyOutcome
from deepagents_coding_platform.runner import LocalRunner


@dataclass(slots=True, frozen=True)
class RuntimeToolSpec:
    name: str
    description: str
    action_kind: ActionKind
    static_payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DeepagentsRuntimeAdapter:
    runner: LocalRunner

    def build_tool(self, spec: RuntimeToolSpec):
        def runtime_wrapped_tool(**payload):
            request = ActionRequest(
                kind=spec.action_kind,
                name=spec.name,
                payload={**dict(spec.static_payload), **payload},
            )
            result = self.runner.run_action(request)
            decision = getattr(result, "decision", None)
            if decision is not None and decision.outcome is not PolicyOutcome.ALLOW:
                raise RuntimeError(f"{decision.outcome.value}: {decision.reason}")
            return dict(result.output or {})

        runtime_wrapped_tool.__name__ = spec.name
        runtime_wrapped_tool.__doc__ = spec.description
        return runtime_wrapped_tool

    def create_agent(
        self,
        *,
        model: str,
        system_prompt: str,
        tool_specs: list[RuntimeToolSpec],
    ):
        tools = [self.build_tool(spec) for spec in tool_specs]
        return create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
        )
```

- [ ] **Step 4: Run the targeted tests again**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_runner.py tests/test_deepagents_adapter.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Commit the foundation changes**

Run:

```bash
git add deepagents-coding-platform/src/deepagents_coding_platform/runner.py \
  deepagents-coding-platform/src/deepagents_coding_platform/adapters/deepagents_runtime.py \
  deepagents-coding-platform/tests/test_runner.py \
  deepagents-coding-platform/tests/test_deepagents_adapter.py
git commit -F - <<'EOF'
Strengthen runner dispatch and runtime-tool failure signaling

Allow action-name routing so multiple coding tools can share one
ActionKind, and propagate blocked runtime decisions back to deepagents
tools instead of returning empty success payloads.

Constraint: Coding tools must share existing ActionKind values without widening the action enum
Rejected: Add one ActionKind per coding tool | would overfit the core action model to one preset
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Keep runner dispatch deterministic: name first, kind second, default last
Tested: uv run pytest tests/test_runner.py tests/test_deepagents_adapter.py -q
Not-tested: Live deepagents tool execution inside a real agent turn
EOF
```

Expected:

```text
[branch-name 111aaaa] Strengthen runner dispatch and runtime-tool failure signaling
```

### Task 2: Add the Workspace-Safe Coding Policy

**Files:**
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/coding/__init__.py`
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/coding/policy.py`
- Create: `deepagents-coding-platform/tests/test_coding_policy.py`
- Test: `deepagents-coding-platform/tests/test_coding_policy.py`

- [ ] **Step 1: Write the failing policy tests**

```python
# deepagents-coding-platform/tests/test_coding_policy.py
from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.coding.policy import WorkspaceCodingPolicy
from deepagents_coding_platform.policy import PolicyOutcome


def test_workspace_policy_allows_workspace_read(tmp_path):
    policy = WorkspaceCodingPolicy(workspace_root=tmp_path)

    decision = policy.evaluate(
        ActionRequest(
            kind=ActionKind.FS_READ,
            name="read_file",
            payload={"path": "README.md"},
        )
    )

    assert decision.outcome is PolicyOutcome.ALLOW


def test_workspace_policy_allows_workspace_write(tmp_path):
    policy = WorkspaceCodingPolicy(workspace_root=tmp_path)

    decision = policy.evaluate(
        ActionRequest(
            kind=ActionKind.FS_WRITE,
            name="write_file",
            payload={"path": "src/app.py"},
        )
    )

    assert decision.outcome is PolicyOutcome.ALLOW


def test_workspace_policy_denies_path_escape(tmp_path):
    policy = WorkspaceCodingPolicy(workspace_root=tmp_path)

    decision = policy.evaluate(
        ActionRequest(
            kind=ActionKind.FS_READ,
            name="read_file",
            payload={"path": "../secret.txt"},
        )
    )

    assert decision.outcome is PolicyOutcome.DENY
    assert "workspace" in decision.reason


def test_workspace_policy_denies_dangerous_shell_command(tmp_path):
    policy = WorkspaceCodingPolicy(workspace_root=tmp_path)

    decision = policy.evaluate(
        ActionRequest(
            kind=ActionKind.SHELL_EXEC,
            name="shell",
            payload={"command": "rm -rf /"},
        )
    )

    assert decision.outcome is PolicyOutcome.DENY


def test_workspace_policy_allows_local_test_command(tmp_path):
    policy = WorkspaceCodingPolicy(workspace_root=tmp_path)

    decision = policy.evaluate(
        ActionRequest(
            kind=ActionKind.SHELL_EXEC,
            name="shell",
            payload={"command": "uv run pytest -q"},
        )
    )

    assert decision.outcome is PolicyOutcome.ALLOW
```

- [ ] **Step 2: Run the policy tests and verify the module is missing**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_coding_policy.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'deepagents_coding_platform.coding'
```

- [ ] **Step 3: Implement the workspace-safe coding policy**

```python
# deepagents-coding-platform/src/deepagents_coding_platform/coding/__init__.py
from deepagents_coding_platform.coding.policy import WorkspaceCodingPolicy

__all__ = ["WorkspaceCodingPolicy"]
```

```python
# deepagents-coding-platform/src/deepagents_coding_platform/coding/policy.py
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
```

- [ ] **Step 4: Run the policy tests again**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_coding_policy.py -q
```

Expected:

```text
5 passed
```

- [ ] **Step 5: Commit the coding policy**

Run:

```bash
git add deepagents-coding-platform/src/deepagents_coding_platform/coding/__init__.py \
  deepagents-coding-platform/src/deepagents_coding_platform/coding/policy.py \
  deepagents-coding-platform/tests/test_coding_policy.py
git commit -F - <<'EOF'
Add a workspace-safe deterministic coding policy

Introduce the first coding-specific policy layer so local file work and
developer commands can execute inside the workspace while path escapes
and dangerous shell commands stay blocked.

Constraint: Safety must stay deterministic and rooted in declared action payloads
Rejected: Let the model infer whether a shell command is dangerous | would weaken auditable execution rules
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Keep workspace path checks and dangerous shell rules in the policy layer, not in individual tools
Tested: uv run pytest tests/test_coding_policy.py -q
Not-tested: End-to-end agent behavior against denied actions
EOF
```

Expected:

```text
[branch-name 222bbbb] Add a workspace-safe deterministic coding policy
```

### Task 3: Add the Built-In Coding Executors

**Files:**
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/coding/executors.py`
- Create: `deepagents-coding-platform/tests/test_coding_executors.py`
- Test: `deepagents-coding-platform/tests/test_coding_executors.py`

- [ ] **Step 1: Write the failing executor tests**

```python
# deepagents-coding-platform/tests/test_coding_executors.py
from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.coding.executors import build_coding_named_executors


def test_read_and_write_file_round_trip(tmp_path):
    executors = build_coding_named_executors(tmp_path)

    write_result = executors["write_file"](
        ActionRequest(
            kind=ActionKind.FS_WRITE,
            name="write_file",
            payload={"path": "src/app.py", "content": "print('hi')\n"},
        )
    )
    read_result = executors["read_file"](
        ActionRequest(
            kind=ActionKind.FS_READ,
            name="read_file",
            payload={"path": "src/app.py"},
        )
    )

    assert write_result["path"] == "src/app.py"
    assert read_result["content"] == "print('hi')\n"


def test_list_files_returns_workspace_entries(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    executors = build_coding_named_executors(tmp_path)

    result = executors["list_files"](
        ActionRequest(
            kind=ActionKind.FS_READ,
            name="list_files",
            payload={"path": ".", "recursive": True},
        )
    )

    assert "src/app.py" in result["entries"]


def test_grep_search_returns_matching_lines(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "one.py").write_text("needle = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "two.py").write_text("other = 2\n", encoding="utf-8")
    executors = build_coding_named_executors(tmp_path)

    result = executors["grep_search"](
        ActionRequest(
            kind=ActionKind.FS_READ,
            name="grep_search",
            payload={"path": ".", "pattern": "needle"},
        )
    )

    assert result["matches"][0]["path"] == "pkg/one.py"
    assert result["matches"][0]["line_number"] == 1


def test_apply_patch_replaces_exact_context(tmp_path):
    target = tmp_path / "main.py"
    target.write_text("name = 'old'\n", encoding="utf-8")
    executors = build_coding_named_executors(tmp_path)

    result = executors["apply_patch"](
        ActionRequest(
            kind=ActionKind.FS_WRITE,
            name="apply_patch",
            payload={
                "path": "main.py",
                "edits": [{"old": "name = 'old'\n", "new": "name = 'new'\n"}],
            },
        )
    )

    assert result["applied"] == 1
    assert target.read_text(encoding="utf-8") == "name = 'new'\n"


def test_apply_patch_reports_missing_context(tmp_path):
    target = tmp_path / "main.py"
    target.write_text("name = 'old'\n", encoding="utf-8")
    executors = build_coding_named_executors(tmp_path)

    result = executors["apply_patch"](
        ActionRequest(
            kind=ActionKind.FS_WRITE,
            name="apply_patch",
            payload={
                "path": "main.py",
                "edits": [{"old": "name = 'missing'\n", "new": "name = 'new'\n"}],
            },
        )
    )

    assert "error" in result
    assert "patch context not found" in result["error"]


def test_shell_runs_inside_workspace_and_returns_output(tmp_path):
    executors = build_coding_named_executors(tmp_path)

    result = executors["shell"](
        ActionRequest(
            kind=ActionKind.SHELL_EXEC,
            name="shell",
            payload={"command": "printf hello"},
        )
    )

    assert result["stdout"] == "hello"
    assert result["returncode"] == 0
```

- [ ] **Step 2: Run the executor tests and verify the module is missing**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_coding_executors.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'deepagents_coding_platform.coding.executors'
```

- [ ] **Step 3: Implement the built-in coding executors**

```python
# deepagents-coding-platform/src/deepagents_coding_platform/coding/executors.py
import re
import subprocess
from pathlib import Path

from deepagents_coding_platform.actions import ActionRequest
from deepagents_coding_platform.runner import Executor


def _resolve_workspace_path(workspace_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(workspace_root):
        raise ValueError(f"path escapes workspace: {raw_path}")
    return resolved


def build_coding_named_executors(workspace_root: Path) -> dict[str, Executor]:
    root = Path(workspace_root).resolve()

    def read_file(action: ActionRequest):
        path = _resolve_workspace_path(root, str(action.payload.get("path", ".")))
        text = path.read_text(encoding="utf-8")
        if "start_line" in action.payload or "end_line" in action.payload:
            lines = text.splitlines()
            start_line = int(action.payload.get("start_line", 1))
            end_line = int(action.payload.get("end_line", len(lines)))
            content = "\n".join(lines[start_line - 1 : end_line])
        else:
            content = text
        return {"path": path.relative_to(root).as_posix(), "content": content}

    def write_file(action: ActionRequest):
        path = _resolve_workspace_path(root, str(action.payload.get("path", ".")))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(action.payload.get("content", ""))
        path.write_text(content, encoding="utf-8")
        return {
            "path": path.relative_to(root).as_posix(),
            "bytes_written": len(content.encode("utf-8")),
        }

    def list_files(action: ActionRequest):
        path = _resolve_workspace_path(root, str(action.payload.get("path", ".")))
        recursive = bool(action.payload.get("recursive", True))
        limit = int(action.payload.get("limit", 200))
        pattern = "**/*" if recursive else "*"
        entries: list[str] = []
        for item in path.glob(pattern):
            if item == path:
                continue
            entries.append(item.relative_to(root).as_posix())
            if len(entries) >= limit:
                break
        return {"entries": entries}

    def grep_search(action: ActionRequest):
        path = _resolve_workspace_path(root, str(action.payload.get("path", ".")))
        pattern = re.compile(str(action.payload.get("pattern", "")))
        glob_pattern = str(action.payload.get("glob", "**/*"))
        limit = int(action.payload.get("limit", 200))
        matches: list[dict[str, object]] = []
        for candidate in path.glob(glob_pattern):
            if not candidate.is_file():
                continue
            for line_number, line in enumerate(
                candidate.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if pattern.search(line):
                    matches.append(
                        {
                            "path": candidate.relative_to(root).as_posix(),
                            "line_number": line_number,
                            "line": line,
                        }
                    )
                    if len(matches) >= limit:
                        return {"matches": matches}
        return {"matches": matches}

    def shell(action: ActionRequest):
        command = str(action.payload.get("command", ""))
        timeout_seconds = int(action.payload.get("timeout_seconds", 60))
        try:
            result = subprocess.run(
                command,
                cwd=root,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "returncode": None,
                "error": f"command timed out after {timeout_seconds}s",
            }

    def apply_patch(action: ActionRequest):
        path = _resolve_workspace_path(root, str(action.payload.get("path", ".")))
        original = path.read_text(encoding="utf-8")
        updated = original
        edits = list(action.payload.get("edits", []))

        for edit in edits:
            old = str(edit["old"])
            new = str(edit["new"])
            if old not in updated:
                return {
                    "path": path.relative_to(root).as_posix(),
                    "error": f"patch context not found: {old}",
                }
            updated = updated.replace(old, new, 1)

        path.write_text(updated, encoding="utf-8")
        return {
            "path": path.relative_to(root).as_posix(),
            "applied": len(edits),
        }

    return {
        "read_file": read_file,
        "write_file": write_file,
        "list_files": list_files,
        "grep_search": grep_search,
        "shell": shell,
        "apply_patch": apply_patch,
    }
```

- [ ] **Step 4: Run the executor tests again**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_coding_executors.py -q
```

Expected:

```text
6 passed
```

- [ ] **Step 5: Commit the built-in executors**

Run:

```bash
git add deepagents-coding-platform/src/deepagents_coding_platform/coding/executors.py \
  deepagents-coding-platform/tests/test_coding_executors.py
git commit -F - <<'EOF'
Add built-in coding executors for the workspace preset

Provide the first concrete coding tools for file reads, writes, search,
listing, shell commands, and exact-context patching inside the
workspace.

Constraint: Built-in tools must execute inside workspace boundaries without inventing a new runtime layer
Rejected: Implement full unified-diff patch parsing first | would widen the first coding slice unnecessarily
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep tool semantics narrow and deterministic; use write_file for full rewrites and apply_patch for exact local edits
Tested: uv run pytest tests/test_coding_executors.py -q
Not-tested: Interaction with a live deepagents agent loop
EOF
```

Expected:

```text
[branch-name 333cccc] Add built-in coding executors for the workspace preset
```

### Task 4: Assemble the Coding Preset

**Files:**
- Modify: `deepagents-coding-platform/src/deepagents_coding_platform/coding/__init__.py`
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/coding/preset.py`
- Create: `deepagents-coding-platform/tests/test_coding_preset.py`
- Test: `deepagents-coding-platform/tests/test_coding_preset.py`

- [ ] **Step 1: Write the failing preset tests**

```python
# deepagents-coding-platform/tests/test_coding_preset.py
from deepagents_coding_platform.coding.policy import WorkspaceCodingPolicy
from deepagents_coding_platform.coding.preset import (
    build_coding_agent,
    build_coding_runner,
)


def test_build_coding_runner_registers_policy_and_all_named_tools(tmp_path):
    runner = build_coding_runner(
        workspace_root=tmp_path,
        ledger_root=tmp_path / ".ledger",
    )

    assert isinstance(runner.kernel.plugins.policy_evaluators[0], WorkspaceCodingPolicy)
    assert sorted(runner.named_executors) == [
        "apply_patch",
        "grep_search",
        "list_files",
        "read_file",
        "shell",
        "write_file",
    ]


def test_build_coding_agent_uses_fixed_prompt_and_six_tools(monkeypatch, tmp_path):
    captured = {}

    def fake_create_agent(self, *, model, system_prompt, tool_specs):
        captured["model"] = model
        captured["system_prompt"] = system_prompt
        captured["tool_specs"] = tool_specs
        return "agent-object"

    monkeypatch.setattr(
        "deepagents_coding_platform.coding.preset.DeepagentsRuntimeAdapter.create_agent",
        fake_create_agent,
    )

    agent = build_coding_agent(
        model="openai:gpt-4.1",
        workspace_root=tmp_path,
        ledger_root=tmp_path / ".ledger",
    )

    assert agent == "agent-object"
    assert captured["model"] == "openai:gpt-4.1"
    assert len(captured["tool_specs"]) == 6
    assert "explore before modifying" in captured["system_prompt"].lower()
```

- [ ] **Step 2: Run the preset tests and verify the module is missing**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_coding_preset.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'deepagents_coding_platform.coding.preset'
```

- [ ] **Step 3: Implement the coding preset assembly**

```python
# deepagents-coding-platform/src/deepagents_coding_platform/coding/__init__.py
from deepagents_coding_platform.coding.executors import build_coding_named_executors
from deepagents_coding_platform.coding.policy import WorkspaceCodingPolicy
from deepagents_coding_platform.coding.preset import (
    CODING_SYSTEM_PROMPT,
    build_coding_agent,
    build_coding_runner,
    build_coding_tool_specs,
)

__all__ = [
    "CODING_SYSTEM_PROMPT",
    "WorkspaceCodingPolicy",
    "build_coding_agent",
    "build_coding_named_executors",
    "build_coding_runner",
    "build_coding_tool_specs",
]
```

```python
# deepagents-coding-platform/src/deepagents_coding_platform/coding/preset.py
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
```

- [ ] **Step 4: Run the preset tests again**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_coding_preset.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit the coding preset**

Run:

```bash
git add deepagents-coding-platform/src/deepagents_coding_platform/coding/__init__.py \
  deepagents-coding-platform/src/deepagents_coding_platform/coding/preset.py \
  deepagents-coding-platform/tests/test_coding_preset.py
git commit -F - <<'EOF'
Assemble the first workspace coding preset

Wire the coding policy, built-in executors, fixed prompt, and runtime
tool specs into one preset that can build a runner or a deepagents
agent.

Constraint: The coding preset must reuse the existing runner and kernel path rather than inventing a separate execution stack
Rejected: Construct tools directly in the CLI command | would duplicate assembly logic and blur boundaries
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep model prompts, tool specs, and workspace policy in the preset layer so other surfaces can reuse them
Tested: uv run pytest tests/test_coding_preset.py -q
Not-tested: Real model-backed agent execution over a repository
EOF
```

Expected:

```text
[branch-name 444dddd] Assemble the first workspace coding preset
```

### Task 5: Add the Chat Session and REPL Loop

**Files:**
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/chat.py`
- Create: `deepagents-coding-platform/tests/test_chat.py`
- Test: `deepagents-coding-platform/tests/test_chat.py`

- [ ] **Step 1: Write the failing chat tests**

```python
# deepagents-coding-platform/tests/test_chat.py
from rich.console import Console

from deepagents_coding_platform.chat import ChatSession


def test_chat_session_runs_turn_and_records_history():
    class FakeAgent:
        def __init__(self):
            self.payloads = []

        def invoke(self, payload):
            self.payloads.append(payload)
            return {
                "messages": [
                    *payload["messages"],
                    {"role": "assistant", "content": "Done."},
                ]
            }

    console = Console(record=True, width=120)
    agent = FakeAgent()
    session = ChatSession(agent=agent, console=console)

    answer = session.run_turn("Fix the test suite.")

    assert answer == "Done."
    assert agent.payloads[0]["messages"] == [
        {"role": "user", "content": "Fix the test suite."}
    ]
    assert session.messages[-1] == {"role": "assistant", "content": "Done."}
    assert "Done." in console.export_text()


def test_chat_session_repl_continues_after_agent_failure():
    console = Console(record=True, width=120)
    state = {"calls": 0}

    class FakeAgent:
        def invoke(self, payload):
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("boom")
            return {
                "messages": [
                    *payload["messages"],
                    {"role": "assistant", "content": "Recovered."},
                ]
            }

    inputs = iter(["first turn", "second turn", "exit"])
    session = ChatSession(agent=FakeAgent(), console=console)
    session.repl(read_input=lambda _prompt: next(inputs))

    output = console.export_text()
    assert "boom" in output
    assert "Recovered." in output
```

- [ ] **Step 2: Run the chat tests and verify the module is missing**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_chat.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'deepagents_coding_platform.chat'
```

- [ ] **Step 3: Implement the chat session and REPL loop**

```python
# deepagents-coding-platform/src/deepagents_coding_platform/chat.py
from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Console


def _extract_assistant_text(result: dict[str, Any]) -> str:
    final_message = result["messages"][-1]
    if isinstance(final_message, dict):
        return str(final_message.get("content", ""))
    return str(getattr(final_message, "content", final_message))


@dataclass(slots=True)
class ChatSession:
    agent: Any
    console: Console
    messages: list[dict[str, str]] = field(default_factory=list)

    def run_turn(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        result = self.agent.invoke({"messages": list(self.messages)})
        answer = _extract_assistant_text(result)
        self.messages.append({"role": "assistant", "content": answer})
        self.console.print(answer)
        return answer

    def repl(self, read_input: Callable[[str], str] = input) -> None:
        while True:
            raw = read_input("dacp> ")
            user_text = raw.strip()
            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                self.console.print("Exiting DACP chat.")
                return
            try:
                self.run_turn(user_text)
            except Exception as exc:
                self.console.print(f"[red]error[/red]: {exc}")
```

- [ ] **Step 4: Run the chat tests again**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_chat.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit the chat loop**

Run:

```bash
git add deepagents-coding-platform/src/deepagents_coding_platform/chat.py \
  deepagents-coding-platform/tests/test_chat.py
git commit -F - <<'EOF'
Add the local chat session loop for DACP

Introduce a small REPL/session layer that keeps in-memory message
history, invokes the deepagents agent per turn, and stays alive across
recoverable turn failures.

Constraint: The first chat surface must stay local and process-scoped rather than adding persistent chat sessions
Rejected: Build streaming or approval UX into the first REPL loop | would widen scope before the coding preset is stable
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Keep chat state ephemeral; persistent truth still lives in runtime events and checkpoints
Tested: uv run pytest tests/test_chat.py -q
Not-tested: Real model-backed multi-turn coding sessions
EOF
```

Expected:

```text
[branch-name 555eeee] Add the local chat session loop for DACP
```

### Task 6: Add the `dacp chat` CLI Command

**Files:**
- Modify: `deepagents-coding-platform/src/deepagents_coding_platform/cli.py`
- Create: `deepagents-coding-platform/tests/test_cli_chat.py`
- Test: `deepagents-coding-platform/tests/test_cli_chat.py`

- [ ] **Step 1: Write the failing CLI chat tests**

```python
# deepagents-coding-platform/tests/test_cli_chat.py
from pathlib import Path

from typer.testing import CliRunner

from deepagents_coding_platform.cli import app


def test_chat_requires_model(tmp_path: Path):
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "chat",
            "--workspace",
            str(tmp_path),
            "--ledger-root",
            str(tmp_path / ".ledger"),
        ],
    )

    assert result.exit_code != 0
    assert "--model" in result.stdout


def test_chat_builds_session_and_runs_repl(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    captured = {}

    class FakeSession:
        def repl(self):
            captured["repl_called"] = True

    def fake_build_chat_session(*, model, workspace, ledger_root):
        captured["model"] = model
        captured["workspace"] = workspace
        captured["ledger_root"] = ledger_root
        return FakeSession()

    monkeypatch.setattr(
        "deepagents_coding_platform.cli._build_chat_session",
        fake_build_chat_session,
    )

    result = runner.invoke(
        app,
        [
            "chat",
            "--model",
            "openai:gpt-4.1",
            "--workspace",
            str(tmp_path),
            "--ledger-root",
            str(tmp_path / ".ledger"),
        ],
    )

    assert result.exit_code == 0
    assert captured["model"] == "openai:gpt-4.1"
    assert captured["repl_called"] is True
```

- [ ] **Step 2: Run the CLI chat tests and verify the command is missing**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_cli_chat.py -q
```

Expected:

```text
E   No such command 'chat'
```

- [ ] **Step 3: Implement the `chat` CLI command**

```python
# deepagents-coding-platform/src/deepagents_coding_platform/cli.py
from pathlib import Path

import typer
from rich.console import Console

from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.chat import ChatSession
from deepagents_coding_platform.coding.preset import build_coding_agent
from deepagents_coding_platform.plugins import PluginRegistry
from deepagents_coding_platform.policy import StaticPolicyEvaluator
from deepagents_coding_platform.runner import LocalRunner
from deepagents_coding_platform.runtime import RuntimeKernel

app = typer.Typer(no_args_is_help=True)
console = Console()


def _parse_payload(items: list[str]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for item in items:
        key, value = item.split("=", 1)
        payload[key] = value
    return payload


def _build_runner(workspace: Path, ledger_root: Path) -> LocalRunner:
    plugins = PluginRegistry(policy_evaluators=[StaticPolicyEvaluator()])
    kernel = RuntimeKernel(session_id="session-1", run_id="run-1", plugins=plugins)
    return LocalRunner(
        workspace_root=workspace,
        ledger_root=ledger_root,
        kernel=kernel,
    )


def _build_chat_session(*, model: str, workspace: Path, ledger_root: Path) -> ChatSession:
    agent = build_coding_agent(
        model=model,
        workspace_root=workspace,
        ledger_root=ledger_root,
    )
    return ChatSession(agent=agent, console=console)


@app.command("run-action")
def run_action(
    kind: ActionKind,
    name: str,
    workspace: Path = typer.Option(...),
    ledger_root: Path = typer.Option(...),
    payload: list[str] = typer.Option([]),
) -> None:
    runner = _build_runner(workspace, ledger_root)
    request = ActionRequest(kind=kind, name=name, payload=_parse_payload(payload))
    result = runner.run_action(request)
    console.print(f"decision={result.decision.outcome.value.upper()}")
    console.print(f"reason={result.decision.reason}")
    if request.payload:
        console.print(f"payload={dict(request.payload)}")


@app.command("resume-session")
def resume_session(ledger_root: Path = typer.Option(...)) -> None:
    runner = _build_runner(Path("."), ledger_root)
    resumed = runner.ledger.resume()
    console.print(f"checkpoint={resumed.checkpoint_name}")
    console.print(f"event_count={len(resumed.events_after_checkpoint)}")


@app.command("chat")
def chat(
    model: str = typer.Option(...),
    workspace: Path = typer.Option(...),
    ledger_root: Path = typer.Option(...),
) -> None:
    session = _build_chat_session(
        model=model,
        workspace=workspace,
        ledger_root=ledger_root,
    )
    session.repl()
```

- [ ] **Step 4: Run the CLI chat tests again**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_cli_chat.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit the CLI chat command**

Run:

```bash
git add deepagents-coding-platform/src/deepagents_coding_platform/cli.py \
  deepagents-coding-platform/tests/test_cli_chat.py
git commit -F - <<'EOF'
Add the DACP chat command for the coding preset

Expose the first chat entry point so a user can launch the coding
preset through the CLI with an explicit model and workspace.

Constraint: CLI must remain a thin consumer of preset assembly rather than owning coding behavior directly
Rejected: Hide model or workspace defaults in the chat command | would make execution roots implicit and harder to reason about
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep the CLI command small and push coding behavior into the preset and chat modules
Tested: uv run pytest tests/test_cli_chat.py -q
Not-tested: Interactive human-in-the-loop chat sessions in a real repository
EOF
```

Expected:

```text
[branch-name 666ffff] Add the DACP chat command for the coding preset
```

### Task 7: Document the Preset and Run the Full Verification Sweep

**Files:**
- Modify: `deepagents-coding-platform/README.md`
- Test: `deepagents-coding-platform/tests/test_smoke.py`
- Test: `deepagents-coding-platform/tests/test_events.py`
- Test: `deepagents-coding-platform/tests/test_policy.py`
- Test: `deepagents-coding-platform/tests/test_projection.py`
- Test: `deepagents-coding-platform/tests/test_checkpoints.py`
- Test: `deepagents-coding-platform/tests/test_runtime.py`
- Test: `deepagents-coding-platform/tests/test_runner.py`
- Test: `deepagents-coding-platform/tests/test_deepagents_adapter.py`
- Test: `deepagents-coding-platform/tests/test_coding_policy.py`
- Test: `deepagents-coding-platform/tests/test_coding_executors.py`
- Test: `deepagents-coding-platform/tests/test_coding_preset.py`
- Test: `deepagents-coding-platform/tests/test_chat.py`
- Test: `deepagents-coding-platform/tests/test_cli.py`
- Test: `deepagents-coding-platform/tests/test_cli_chat.py`
- Test: `deepagents-coding-platform/tests/test_control_plane.py`

- [ ] **Step 1: Update the README with coding preset usage**

```markdown
# deepagents-coding-platform/README.md
## Local development

uv sync
uv run pytest -q
uv run python -m compileall src
uv run dacp --help

## What exists in P1

- typed runtime actions and events
- deterministic local policy evaluation
- audience-specific projections
- local session ledger and checkpoint resume
- local runner and `create_deep_agent()` adapter
- minimal CLI commands
- metadata-safe control-plane export hook

## Coding preset chat

Run the workspace-scoped coding agent REPL:

uv run dacp chat \
  --model openai:gpt-4.1 \
  --workspace /path/to/repo \
  --ledger-root /path/to/repo/.dacp-ledger

Built-in coding tools:

- `read_file`
- `write_file`
- `list_files`
- `grep_search`
- `shell`
- `apply_patch`
```

- [ ] **Step 2: Run the compile check**

Run:

```bash
cd deepagents-coding-platform
uv run python -m compileall src
```

Expected:

```text
Listing 'src'...
Compiling 'src/deepagents_coding_platform/chat.py'...
```

- [ ] **Step 3: Run the full test suite**

Run:

```bash
cd deepagents-coding-platform
uv run pytest -q
```

Expected:

```text
35 passed
```

- [ ] **Step 4: Run the CLI help smoke tests**

Run:

```bash
cd deepagents-coding-platform
uv run dacp --help
uv run dacp chat --help
```

Expected:

```text
Usage: dacp [OPTIONS] COMMAND [ARGS]...
Usage: dacp chat [OPTIONS]
```

- [ ] **Step 5: Commit the documented and verified coding slice**

Run:

```bash
git add deepagents-coding-platform/README.md
git commit -F - <<'EOF'
Document and verify the DACP coding preset slice

Close the first usable coding-agent slice with a documented chat entry
point and a full verification sweep across policy, tools, runner,
adapter, REPL, and CLI.

Constraint: This slice must stop at local workspace coding plus deterministic safety boundaries
Rejected: Expand the same plan into streaming, approvals, or hosted orchestration | would break the scoped delivery target
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep future expansion layered on top of this preset instead of backfilling hidden behavior into the CLI
Tested: uv run python -m compileall src
Tested: uv run pytest -q
Tested: uv run dacp --help
Tested: uv run dacp chat --help
Not-tested: Real-user repository sessions or interactive approval workflows
EOF
```

Expected:

```text
[branch-name 777gggg] Document and verify the DACP coding preset slice
```
