# Deepagents Coding Platform P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first executable slice of a deepagents-based enterprise coding-agent platform by implementing the local runtime kernel, local runner, minimal CLI, and metadata-safe control-plane hook.

**Architecture:** Create a new Python project, `deepagents-coding-platform/`, that keeps execution truth in a runner-side runtime kernel. The kernel emits typed runtime events, runs policy evaluation before execution, projects visibility per audience, checkpoints local session state, and exposes a minimal CLI plus a metadata-first control-plane client. The first plan deliberately stops at `P1` plus the thinnest `P2/P3` hooks needed to prove the architecture.

**Tech Stack:** Python 3.12, `uv`, local editable `deepagents`, `typer`, `rich`, `httpx`, `pytest`

---

## File Structure

### Create

- `deepagents-coding-platform/pyproject.toml`
  - Project metadata, runtime dependencies, editable local `deepagents` source wiring, dev dependencies.
- `deepagents-coding-platform/README.md`
  - Minimal local developer instructions for syncing deps and running the CLI/tests.
- `deepagents-coding-platform/src/deepagents_coding_platform/__init__.py`
  - Package export surface.
- `deepagents-coding-platform/src/deepagents_coding_platform/actions.py`
  - Canonical action request types for model/tool/shell/filesystem/subagent execution.
- `deepagents-coding-platform/src/deepagents_coding_platform/events.py`
  - Typed runtime event schema, serialization helpers, event phase and projection enums.
- `deepagents-coding-platform/src/deepagents_coding_platform/policy.py`
  - Policy decision types and the first deterministic local policy evaluator.
- `deepagents-coding-platform/src/deepagents_coding_platform/projection.py`
  - Audience types and visibility projector.
- `deepagents-coding-platform/src/deepagents_coding_platform/checkpoints.py`
  - Local session ledger, checkpoint persistence, and resume snapshot logic.
- `deepagents-coding-platform/src/deepagents_coding_platform/plugins.py`
  - Tool, policy, sink, and subagent-archetype registries.
- `deepagents-coding-platform/src/deepagents_coding_platform/runtime.py`
  - Runtime kernel orchestration: normalize -> policy -> execute -> emit -> project -> checkpoint.
- `deepagents-coding-platform/src/deepagents_coding_platform/runner.py`
  - Local runner host that owns workspace, executor map, and ledger location.
- `deepagents-coding-platform/src/deepagents_coding_platform/control_plane.py`
  - Metadata-safe control-plane event client.
- `deepagents-coding-platform/src/deepagents_coding_platform/cli.py`
  - `typer`-based CLI for action execution and session resume/status.
- `deepagents-coding-platform/src/deepagents_coding_platform/adapters/__init__.py`
  - Adapter package export.
- `deepagents-coding-platform/src/deepagents_coding_platform/adapters/deepagents_runtime.py`
  - Runtime-backed tool wrapper and `create_deep_agent()` factory adapter.
- `deepagents-coding-platform/tests/test_smoke.py`
  - Package-level smoke test.
- `deepagents-coding-platform/tests/test_events.py`
  - Event schema and action schema tests.
- `deepagents-coding-platform/tests/test_policy.py`
  - Policy evaluator tests.
- `deepagents-coding-platform/tests/test_projection.py`
  - Visibility projection tests.
- `deepagents-coding-platform/tests/test_checkpoints.py`
  - Session ledger and resume tests.
- `deepagents-coding-platform/tests/test_runtime.py`
  - Runtime kernel integration tests.
- `deepagents-coding-platform/tests/test_runner.py`
  - Local runner tests.
- `deepagents-coding-platform/tests/test_deepagents_adapter.py`
  - `deepagents` adapter tests.
- `deepagents-coding-platform/tests/test_cli.py`
  - CLI behavior tests.
- `deepagents-coding-platform/tests/test_control_plane.py`
  - Control-plane export tests.

### Do Not Modify

- `deepagents/`
- `langgraph/`
- `langchain/`
- `deepagents-internal-tutorial/`
- `docs/superpowers/specs/2026-04-18-deepagents-codex-like-platform-design.md`

### Verification Surface

- Unit tests: `tests/test_events.py`, `tests/test_policy.py`, `tests/test_projection.py`, `tests/test_checkpoints.py`
- Integration tests: `tests/test_runtime.py`, `tests/test_runner.py`, `tests/test_deepagents_adapter.py`, `tests/test_cli.py`, `tests/test_control_plane.py`
- Formatting and import sanity: `uv run python -m compileall src`
- Final pass: `uv run pytest -q`

## Task 1: Scaffold the Package and Test Harness

**Files:**
- Create: `deepagents-coding-platform/pyproject.toml`
- Create: `deepagents-coding-platform/README.md`
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/__init__.py`
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/plugins.py`
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/runtime.py`
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/cli.py`
- Create: `deepagents-coding-platform/tests/test_smoke.py`
- Test: `deepagents-coding-platform/tests/test_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# deepagents-coding-platform/tests/test_smoke.py
from deepagents_coding_platform.runtime import RuntimeKernel


def test_runtime_kernel_starts_with_empty_plugin_registry():
    kernel = RuntimeKernel(session_id="session-1", run_id="run-1")

    assert kernel.plugins.tools == {}
    assert kernel.plugins.subagent_archetypes == {}
```

- [ ] **Step 2: Run the smoke test to verify the package does not exist yet**

Run:

```bash
cd deepagents-coding-platform
python -m pytest tests/test_smoke.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'deepagents_coding_platform'
```

- [ ] **Step 3: Write the minimal package scaffold**

```toml
# deepagents-coding-platform/pyproject.toml
[project]
name = "deepagents-coding-platform"
version = "0.1.0"
description = "Runtime-first enterprise coding agent platform built on deepagents."
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "deepagents",
  "typer>=0.12.3",
  "rich>=13.7.1",
]

[dependency-groups]
dev = [
  "httpx>=0.27.0",
  "pytest>=8.2.0",
]

[tool.uv.sources]
deepagents = { path = "../deepagents/libs/deepagents", editable = true }

[project.scripts]
dacp = "deepagents_coding_platform.cli:app"
```

```text
# deepagents-coding-platform/README.md

## Local development

uv sync
uv run pytest -q
uv run dacp --help
```

```python
# deepagents-coding-platform/src/deepagents_coding_platform/__init__.py
"""Deepagents coding platform package."""

from deepagents_coding_platform.plugins import PluginRegistry
from deepagents_coding_platform.runtime import RuntimeKernel

__all__ = ["PluginRegistry", "RuntimeKernel"]
```

```python
# deepagents-coding-platform/src/deepagents_coding_platform/plugins.py
from dataclasses import dataclass, field
from typing import Any, Callable


ToolHandler = Callable[..., dict[str, Any]]
EventSink = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class PluginRegistry:
    tools: dict[str, ToolHandler] = field(default_factory=dict)
    policy_evaluators: list[Any] = field(default_factory=list)
    event_sinks: list[EventSink] = field(default_factory=list)
    subagent_archetypes: dict[str, Any] = field(default_factory=dict)
```

```python
# deepagents-coding-platform/src/deepagents_coding_platform/runtime.py
from dataclasses import dataclass, field

from deepagents_coding_platform.plugins import PluginRegistry


@dataclass(slots=True)
class RuntimeKernel:
    session_id: str
    run_id: str
    plugins: PluginRegistry = field(default_factory=PluginRegistry)
```

```python
# deepagents-coding-platform/src/deepagents_coding_platform/cli.py
import typer

app = typer.Typer(no_args_is_help=True)
```

- [ ] **Step 4: Sync dependencies and run the smoke test again**

Run:

```bash
cd deepagents-coding-platform
uv sync
uv run pytest tests/test_smoke.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit the scaffold**

Run:

```bash
git add deepagents-coding-platform
git commit -F - <<'EOF'
Scaffold the deepagents coding platform package

Create the new runtime-platform package with the smallest importable
surface so subsequent tasks can add behavior through tests.

Constraint: The implementation must live in a new project directory and must not modify upstream deepagents sources
Rejected: Start by wiring runtime logic directly into deepagents/ | would blur product code and upstream source ownership
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Keep the package importable and testable after every task
Tested: uv run pytest tests/test_smoke.py -q
Not-tested: Any runtime behavior beyond package import and registry defaults
EOF
```

Expected:

```text
[branch-name abc1234] Scaffold the deepagents coding platform package
```

## Task 2: Define Action Requests and Typed Runtime Events

**Files:**
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/actions.py`
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/events.py`
- Create: `deepagents-coding-platform/tests/test_events.py`
- Modify: `deepagents-coding-platform/src/deepagents_coding_platform/__init__.py`
- Test: `deepagents-coding-platform/tests/test_events.py`

- [ ] **Step 1: Write the failing schema tests**

```python
# deepagents-coding-platform/tests/test_events.py
from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.events import EventPhase, ProjectionTag, RuntimeEvent


def test_action_request_captures_kind_name_and_payload():
    action = ActionRequest(
        kind=ActionKind.TOOL_CALL,
        name="read_file",
        payload={"path": "README.md"},
        actor="primary_agent",
    )

    assert action.kind is ActionKind.TOOL_CALL
    assert action.payload["path"] == "README.md"


def test_runtime_event_preserves_lineage_and_projection_tags():
    event = RuntimeEvent(
        event_id="evt-2",
        session_id="session-1",
        run_id="run-1",
        parent_event_id="evt-1",
        actor="primary_agent",
        event_type="tool_call",
        phase=EventPhase.PROPOSED,
        raw_payload={"path": "README.md"},
        redacted_payload={"path": "README.md"},
        summary_payload={"path": "README.md"},
        projection_tags=(
            ProjectionTag.USER_VISIBLE,
            ProjectionTag.AUDIT_VISIBLE,
            ProjectionTag.REPLAY_REQUIRED,
        ),
    )

    assert event.parent_event_id == "evt-1"
    assert ProjectionTag.AUDIT_VISIBLE in event.projection_tags
    assert event.phase is EventPhase.PROPOSED
```

- [ ] **Step 2: Run the schema tests to verify the modules are missing**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_events.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'deepagents_coding_platform.actions'
```

- [ ] **Step 3: Implement action and event schemas**

```python
# deepagents-coding-platform/src/deepagents_coding_platform/actions.py
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
```

```python
# deepagents-coding-platform/src/deepagents_coding_platform/events.py
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
```

```python
# deepagents-coding-platform/src/deepagents_coding_platform/__init__.py
"""Deepagents coding platform package."""

from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.events import EventPhase, ProjectionTag, RuntimeEvent
from deepagents_coding_platform.plugins import PluginRegistry
from deepagents_coding_platform.runtime import RuntimeKernel

__all__ = [
    "ActionKind",
    "ActionRequest",
    "EventPhase",
    "PluginRegistry",
    "ProjectionTag",
    "RuntimeEvent",
    "RuntimeKernel",
]
```

- [ ] **Step 4: Run the schema tests again**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_events.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit the schemas**

Run:

```bash
git add deepagents-coding-platform/src/deepagents_coding_platform/actions.py \
  deepagents-coding-platform/src/deepagents_coding_platform/events.py \
  deepagents-coding-platform/src/deepagents_coding_platform/__init__.py \
  deepagents-coding-platform/tests/test_events.py
git commit -F - <<'EOF'
Define typed action and runtime event schemas

Introduce the action and event types that every later kernel, policy,
projection, and replay step will share.

Constraint: Visibility, permission, and replay need a common contract before runtime logic is added
Rejected: Let each subsystem define its own event shape | would create semantic split-brain immediately
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Add serialization helpers to RuntimeEvent instead of inventing parallel DTOs in later tasks
Tested: uv run pytest tests/test_events.py -q
Not-tested: Any persistence or policy behavior using the schemas
EOF
```

Expected:

```text
[branch-name def5678] Define typed action and runtime event schemas
```

## Task 3: Add the First Local Policy Evaluator

**Files:**
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/policy.py`
- Create: `deepagents-coding-platform/tests/test_policy.py`
- Test: `deepagents-coding-platform/tests/test_policy.py`

- [ ] **Step 1: Write the failing policy tests**

```python
# deepagents-coding-platform/tests/test_policy.py
from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.policy import PolicyOutcome, StaticPolicyEvaluator


def test_git_push_requires_approval():
    evaluator = StaticPolicyEvaluator()
    action = ActionRequest(
        kind=ActionKind.SHELL_EXEC,
        name="shell",
        payload={"command": "git push origin main"},
    )

    decision = evaluator.evaluate(action)

    assert decision.outcome is PolicyOutcome.REQUIRE_APPROVAL
    assert "git push" in decision.reason


def test_rm_rf_is_denied():
    evaluator = StaticPolicyEvaluator()
    action = ActionRequest(
        kind=ActionKind.SHELL_EXEC,
        name="shell",
        payload={"command": "rm -rf /"},
    )

    decision = evaluator.evaluate(action)

    assert decision.outcome is PolicyOutcome.DENY


def test_read_only_tool_call_is_allowed():
    evaluator = StaticPolicyEvaluator()
    action = ActionRequest(
        kind=ActionKind.FS_READ,
        name="read_file",
        payload={"path": "README.md"},
    )

    decision = evaluator.evaluate(action)

    assert decision.outcome is PolicyOutcome.ALLOW
```

- [ ] **Step 2: Run the policy tests to verify the module is missing**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_policy.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'deepagents_coding_platform.policy'
```

- [ ] **Step 3: Implement the deterministic policy layer**

```python
# deepagents-coding-platform/src/deepagents_coding_platform/policy.py
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
```

- [ ] **Step 4: Run the policy tests again**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_policy.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit the policy layer**

Run:

```bash
git add deepagents-coding-platform/src/deepagents_coding_platform/policy.py \
  deepagents-coding-platform/tests/test_policy.py
git commit -F - <<'EOF'
Add the first deterministic local policy evaluator

Add a small but explicit policy layer so the runtime can make typed
allow, deny, and approval decisions before executing actions.

Constraint: Execution-time policy enforcement must stay next to the runner
Rejected: Push approval decisions into the future control plane client | would weaken local execution truth
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Keep policy outputs typed and auditable; do not let evaluators perform side effects
Tested: uv run pytest tests/test_policy.py -q
Not-tested: Runtime integration or event emission for policy decisions
EOF
```

Expected:

```text
[branch-name ghi9012] Add the first deterministic local policy evaluator
```

## Task 4: Implement Audience-Specific Visibility Projection

**Files:**
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/projection.py`
- Create: `deepagents-coding-platform/tests/test_projection.py`
- Test: `deepagents-coding-platform/tests/test_projection.py`

- [ ] **Step 1: Write the failing projection tests**

```python
# deepagents-coding-platform/tests/test_projection.py
from deepagents_coding_platform.events import EventPhase, ProjectionTag, RuntimeEvent
from deepagents_coding_platform.projection import Audience, VisibilityProjector


def test_projector_uses_payload_tiers_for_each_audience():
    event = RuntimeEvent(
        event_id="evt-1",
        session_id="session-1",
        run_id="run-1",
        parent_event_id=None,
        actor="primary_agent",
        event_type="shell_exec",
        phase=EventPhase.COMPLETED,
        raw_payload={"command": "cat .env", "stdout": "SECRET=1"},
        redacted_payload={"command": "cat .env", "stdout": "[redacted]"},
        summary_payload={"command": "cat .env", "status": "completed"},
        projection_tags=(
            ProjectionTag.USER_VISIBLE,
            ProjectionTag.AUDIT_VISIBLE,
            ProjectionTag.REPLAY_REQUIRED,
        ),
    )

    projections = VisibilityProjector().project(event)

    assert projections[Audience.CLI].payload["stdout"] == "[redacted]"
    assert projections[Audience.CONTROL_PLANE].payload["status"] == "completed"
    assert projections[Audience.LOCAL_DEBUG].payload["stdout"] == "SECRET=1"


def test_parent_projection_is_omitted_without_parent_visible_tag():
    event = RuntimeEvent(
        event_id="evt-2",
        session_id="session-1",
        run_id="run-1",
        parent_event_id=None,
        actor="primary_agent",
        event_type="tool_call",
        phase=EventPhase.COMPLETED,
        raw_payload={"path": "README.md"},
        redacted_payload={"path": "README.md"},
        summary_payload={"path": "README.md"},
        projection_tags=(ProjectionTag.USER_VISIBLE,),
    )

    projections = VisibilityProjector().project(event)

    assert Audience.PARENT not in projections
```

- [ ] **Step 2: Run the projection tests to verify the projector is missing**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_projection.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'deepagents_coding_platform.projection'
```

- [ ] **Step 3: Implement the audience projector**

```python
# deepagents-coding-platform/src/deepagents_coding_platform/projection.py
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
```

- [ ] **Step 4: Run the projection tests again**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_projection.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit the projector**

Run:

```bash
git add deepagents-coding-platform/src/deepagents_coding_platform/projection.py \
  deepagents-coding-platform/tests/test_projection.py
git commit -F - <<'EOF'
Add audience-specific runtime event projection

Introduce the first visibility projector so CLI users, control-plane
consumers, and local replay can read different payload tiers from the
same event stream.

Constraint: Visibility must be explicit projections over shared runtime events
Rejected: Let each consumer decide ad hoc whether to read raw or redacted payloads | would make visibility untestable
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Keep raw payload access local by default and only export summary payloads to hosted surfaces
Tested: uv run pytest tests/test_projection.py -q
Not-tested: Interaction between projections and checkpoint/replay logic
EOF
```

Expected:

```text
[branch-name jkl3456] Add audience-specific runtime event projection
```

## Task 5: Add Local Session Ledger, Checkpoints, and Resume

**Files:**
- Modify: `deepagents-coding-platform/src/deepagents_coding_platform/events.py`
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/checkpoints.py`
- Create: `deepagents-coding-platform/tests/test_checkpoints.py`
- Test: `deepagents-coding-platform/tests/test_checkpoints.py`

- [ ] **Step 1: Write the failing checkpoint tests**

```python
# deepagents-coding-platform/tests/test_checkpoints.py
from deepagents_coding_platform.checkpoints import SessionLedger
from deepagents_coding_platform.events import EventPhase, ProjectionTag, RuntimeEvent


def make_event(event_id: str) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=event_id,
        session_id="session-1",
        run_id="run-1",
        parent_event_id=None,
        actor="primary_agent",
        event_type="tool_call",
        phase=EventPhase.COMPLETED,
        raw_payload={"index": event_id},
        redacted_payload={"index": event_id},
        summary_payload={"index": event_id},
        projection_tags=(ProjectionTag.REPLAY_REQUIRED,),
    )


def test_resume_uses_latest_checkpoint_and_following_events(tmp_path):
    ledger = SessionLedger(tmp_path)

    ledger.append_event(make_event("evt-1"))
    ledger.commit_checkpoint("cp-1", {"cursor": 1})
    ledger.append_event(make_event("evt-2"))

    resumed = ledger.resume()

    assert resumed.checkpoint_name == "cp-1"
    assert resumed.state["cursor"] == 1
    assert [event.event_id for event in resumed.events_after_checkpoint] == ["evt-2"]
```

- [ ] **Step 2: Run the checkpoint tests to verify the ledger is missing**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_checkpoints.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'deepagents_coding_platform.checkpoints'
```

- [ ] **Step 3: Add event serialization and the session ledger**

```python
# deepagents-coding-platform/src/deepagents_coding_platform/events.py
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
```

```python
# deepagents-coding-platform/src/deepagents_coding_platform/checkpoints.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from deepagents_coding_platform.events import RuntimeEvent


@dataclass(slots=True, frozen=True)
class ResumeState:
    checkpoint_name: str | None
    state: Mapping[str, Any]
    events_after_checkpoint: list[RuntimeEvent]


class SessionLedger:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.checkpoints_dir = self.root / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.latest_path = self.root / "latest_checkpoint.json"

    def append_event(self, event: RuntimeEvent) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict()) + "\n")

    def read_events(self) -> list[RuntimeEvent]:
        if not self.events_path.exists():
            return []

        with self.events_path.open(encoding="utf-8") as handle:
            return [
                RuntimeEvent.from_dict(json.loads(line))
                for line in handle
                if line.strip()
            ]

    def commit_checkpoint(self, name: str, state: Mapping[str, Any]) -> Path:
        events = self.read_events()
        payload = {"name": name, "state": dict(state), "event_cursor": len(events)}
        checkpoint_path = self.checkpoints_dir / f"{name}.json"
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return checkpoint_path

    def resume(self) -> ResumeState:
        events = self.read_events()
        if not self.latest_path.exists():
            return ResumeState(
                checkpoint_name=None,
                state={},
                events_after_checkpoint=events,
            )

        checkpoint = json.loads(self.latest_path.read_text(encoding="utf-8"))
        cursor = int(checkpoint["event_cursor"])
        return ResumeState(
            checkpoint_name=str(checkpoint["name"]),
            state=dict(checkpoint["state"]),
            events_after_checkpoint=events[cursor:],
        )
```

- [ ] **Step 4: Run the checkpoint tests again**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_checkpoints.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit checkpoints and resume**

Run:

```bash
git add deepagents-coding-platform/src/deepagents_coding_platform/events.py \
  deepagents-coding-platform/src/deepagents_coding_platform/checkpoints.py \
  deepagents-coding-platform/tests/test_checkpoints.py
git commit -F - <<'EOF'
Add local session ledger and checkpoint resume support

Persist runtime events locally and add the first checkpoint/resume path
so the runner can recover session truth without relying on the hosted
control plane.

Constraint: Local recovery must work even if the hosted plane is unavailable
Rejected: Store recovery state only in SaaS | would break the runner-truth architecture
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Checkpoint after stable state boundaries; do not checkpoint every stream token
Tested: uv run pytest tests/test_checkpoints.py -q
Not-tested: Runtime-driven checkpoint creation during live execution
EOF
```

Expected:

```text
[branch-name mno7890] Add local session ledger and checkpoint resume support
```

## Task 6: Build the Runtime Kernel Orchestration Loop

**Files:**
- Modify: `deepagents-coding-platform/src/deepagents_coding_platform/runtime.py`
- Create: `deepagents-coding-platform/tests/test_runtime.py`
- Test: `deepagents-coding-platform/tests/test_runtime.py`

- [ ] **Step 1: Write the failing runtime tests**

```python
# deepagents-coding-platform/tests/test_runtime.py
from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.checkpoints import SessionLedger
from deepagents_coding_platform.plugins import PluginRegistry
from deepagents_coding_platform.policy import PolicyOutcome, StaticPolicyEvaluator
from deepagents_coding_platform.projection import Audience
from deepagents_coding_platform.runtime import RuntimeKernel


def test_runtime_kernel_emits_policy_and_completion_events(tmp_path):
    ledger = SessionLedger(tmp_path / ".ledger")
    plugins = PluginRegistry(policy_evaluators=[StaticPolicyEvaluator()])
    kernel = RuntimeKernel(
        session_id="session-1",
        run_id="run-1",
        plugins=plugins,
    )

    action = ActionRequest(
        kind=ActionKind.TOOL_CALL,
        name="read_file",
        payload={"path": "README.md"},
    )

    result = kernel.handle(
        action,
        executor=lambda request: {"content": f"read:{request.payload['path']}"},
        ledger=ledger,
    )

    assert result.decision.outcome is PolicyOutcome.ALLOW
    assert [event.phase.value for event in result.events] == [
        "proposed",
        "allowed",
        "completed",
    ]
    assert result.output["content"] == "read:README.md"
    assert result.projections[Audience.CONTROL_PLANE][0].payload["action"] == "read_file"


def test_runtime_kernel_stops_before_execution_when_approval_is_required(tmp_path):
    ledger = SessionLedger(tmp_path / ".ledger")
    plugins = PluginRegistry(policy_evaluators=[StaticPolicyEvaluator()])
    kernel = RuntimeKernel(
        session_id="session-1",
        run_id="run-1",
        plugins=plugins,
    )

    action = ActionRequest(
        kind=ActionKind.SHELL_EXEC,
        name="shell",
        payload={"command": "git push origin main"},
    )

    executed = {"called": False}

    def fake_executor(_request):
        executed["called"] = True
        return {"stdout": "should-not-run"}

    result = kernel.handle(action, executor=fake_executor, ledger=ledger)

    assert result.decision.outcome is PolicyOutcome.REQUIRE_APPROVAL
    assert executed["called"] is False
    assert result.output is None
```

- [ ] **Step 2: Run the runtime tests to verify the kernel is still too thin**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_runtime.py -q
```

Expected:

```text
E   AttributeError: 'RuntimeKernel' object has no attribute 'handle'
```

- [ ] **Step 3: Implement the runtime orchestration loop**

```python
# deepagents-coding-platform/src/deepagents_coding_platform/runtime.py
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
            payload={"action": action.name, "decision": decision.outcome.value, "reason": decision.reason},
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
```

- [ ] **Step 4: Run the runtime tests again**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_runtime.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit the runtime loop**

Run:

```bash
git add deepagents-coding-platform/src/deepagents_coding_platform/runtime.py \
  deepagents-coding-platform/tests/test_runtime.py
git commit -F - <<'EOF'
Implement the runtime kernel orchestration loop

Wire actions through policy evaluation, event emission, visibility
projection, and checkpoint creation so the local runner has one
deterministic execution path.

Constraint: Permission, visibility, and replay must remain projections over the same event stream
Rejected: Separate runtime, policy, and projection pipelines | would create divergent lifecycle ordering
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep lifecycle ordering deterministic: proposed -> decision -> execution/result
Tested: uv run pytest tests/test_runtime.py -q
Not-tested: Real filesystem, shell, or deepagents-backed execution
EOF
```

Expected:

```text
[branch-name pqr1234] Implement the runtime kernel orchestration loop
```

## Task 7: Add the Local Runner and Deep Agents Adapter

**Files:**
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/runner.py`
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/adapters/__init__.py`
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/adapters/deepagents_runtime.py`
- Create: `deepagents-coding-platform/tests/test_runner.py`
- Create: `deepagents-coding-platform/tests/test_deepagents_adapter.py`
- Test: `deepagents-coding-platform/tests/test_runner.py`
- Test: `deepagents-coding-platform/tests/test_deepagents_adapter.py`

- [ ] **Step 1: Write the failing runner and adapter tests**

```python
# deepagents-coding-platform/tests/test_runner.py
from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.plugins import PluginRegistry
from deepagents_coding_platform.policy import StaticPolicyEvaluator
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
```

```python
# deepagents-coding-platform/tests/test_deepagents_adapter.py
from deepagents_coding_platform.actions import ActionKind
from deepagents_coding_platform.adapters.deepagents_runtime import (
    DeepagentsRuntimeAdapter,
    RuntimeToolSpec,
)


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
```

- [ ] **Step 2: Run the tests to verify the runner and adapter modules are missing**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_runner.py tests/test_deepagents_adapter.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'deepagents_coding_platform.runner'
```

- [ ] **Step 3: Implement the local runner and `deepagents` adapter**

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
```

```python
# deepagents-coding-platform/src/deepagents_coding_platform/adapters/__init__.py
from deepagents_coding_platform.adapters.deepagents_runtime import (
    DeepagentsRuntimeAdapter,
    RuntimeToolSpec,
)

__all__ = ["DeepagentsRuntimeAdapter", "RuntimeToolSpec"]
```

```python
# deepagents-coding-platform/src/deepagents_coding_platform/adapters/deepagents_runtime.py
from dataclasses import dataclass, field
from typing import Any, Mapping

from deepagents import create_deep_agent

from deepagents_coding_platform.actions import ActionRequest, ActionKind
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
            return dict(result.output or {})

        runtime_wrapped_tool.__name__ = spec.name
        runtime_wrapped_tool.__doc__ = spec.description
        return runtime_wrapped_tool

    def create_agent(self, *, model: str, system_prompt: str, tool_specs: list[RuntimeToolSpec]):
        tools = [self.build_tool(spec) for spec in tool_specs]
        return create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
        )
```

- [ ] **Step 4: Run the runner and adapter tests again**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_runner.py tests/test_deepagents_adapter.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit the runner slice**

Run:

```bash
git add deepagents-coding-platform/src/deepagents_coding_platform/runner.py \
  deepagents-coding-platform/src/deepagents_coding_platform/adapters/__init__.py \
  deepagents-coding-platform/src/deepagents_coding_platform/adapters/deepagents_runtime.py \
  deepagents-coding-platform/tests/test_runner.py \
  deepagents-coding-platform/tests/test_deepagents_adapter.py
git commit -F - <<'EOF'
Add the local runner and deepagents runtime adapter

Introduce the runner-side execution host and the first adapter that
wraps runtime-controlled tools into create_deep_agent().

Constraint: Product code must extend deepagents through adapters instead of patching upstream sources
Rejected: Bake runtime policy and projection rules directly into tool functions | would hide the execution contract
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep create_deep_agent() integration in an adapter layer so future CLI or control-plane clients can reuse the same kernel
Tested: uv run pytest tests/test_runner.py tests/test_deepagents_adapter.py -q
Not-tested: Live model-backed deepagents agent execution
EOF
```

Expected:

```text
[branch-name stu5678] Add the local runner and deepagents runtime adapter
```

## Task 8: Add the Minimal CLI Surface

**Files:**
- Modify: `deepagents-coding-platform/src/deepagents_coding_platform/cli.py`
- Create: `deepagents-coding-platform/tests/test_cli.py`
- Test: `deepagents-coding-platform/tests/test_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

```python
# deepagents-coding-platform/tests/test_cli.py
from pathlib import Path

from typer.testing import CliRunner

from deepagents_coding_platform.cli import app


def test_run_action_reports_approval_required(tmp_path: Path):
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-action",
            "shell_exec",
            "shell",
            "--workspace",
            str(tmp_path),
            "--ledger-root",
            str(tmp_path / ".ledger"),
            "--payload",
            "command=git push origin main",
        ],
    )

    assert result.exit_code == 0
    assert "REQUIRE_APPROVAL" in result.stdout
    assert "git push origin main" in result.stdout


def test_resume_session_prints_last_checkpoint(tmp_path: Path):
    runner = CliRunner()

    run_result = runner.invoke(
        app,
        [
            "run-action",
            "fs_read",
            "read_file",
            "--workspace",
            str(tmp_path),
            "--ledger-root",
            str(tmp_path / ".ledger"),
            "--payload",
            "path=README.md",
        ],
    )
    assert run_result.exit_code == 0

    resume_result = runner.invoke(
        app,
        ["resume-session", "--ledger-root", str(tmp_path / ".ledger")],
    )

    assert resume_result.exit_code == 0
    assert "checkpoint" in resume_result.stdout.lower()
```

- [ ] **Step 2: Run the CLI tests to verify the CLI is still too thin**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_cli.py -q
```

Expected:

```text
E   No such command 'run-action'
```

- [ ] **Step 3: Implement the minimal runtime-aware CLI**

```python
# deepagents-coding-platform/src/deepagents_coding_platform/cli.py
from pathlib import Path

import typer
from rich.console import Console

from deepagents_coding_platform.actions import ActionKind, ActionRequest
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
```

- [ ] **Step 4: Run the CLI tests again**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_cli.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit the CLI**

Run:

```bash
git add deepagents-coding-platform/src/deepagents_coding_platform/cli.py \
  deepagents-coding-platform/tests/test_cli.py
git commit -F - <<'EOF'
Add the minimal runtime-aware CLI surface

Add the first terminal commands for routing actions through the local
runner and inspecting the local session ledger.

Constraint: CLI is a consumer of runtime truth and must not own policy or replay semantics
Rejected: Put approval and resume state only in ephemeral terminal state | would break recoverability and auditability
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep CLI commands thin and route all execution through the runner/kernel path
Tested: uv run pytest tests/test_cli.py -q
Not-tested: Richer chat UX, patch review, or interactive approval loops
EOF
```

Expected:

```text
[branch-name vwx9012] Add the minimal runtime-aware CLI surface
```

## Task 9: Add the Metadata-Safe Control Plane Export Hook

**Files:**
- Create: `deepagents-coding-platform/src/deepagents_coding_platform/control_plane.py`
- Create: `deepagents-coding-platform/tests/test_control_plane.py`
- Test: `deepagents-coding-platform/tests/test_control_plane.py`

- [ ] **Step 1: Write the failing control-plane tests**

```python
# deepagents-coding-platform/tests/test_control_plane.py
import httpx

from deepagents_coding_platform.control_plane import ControlPlaneEventClient
from deepagents_coding_platform.events import EventPhase, ProjectionTag, RuntimeEvent


def test_control_plane_client_uploads_summary_payload_only():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = request.content.decode("utf-8")
        return httpx.Response(202, json={"accepted": True})

    client = ControlPlaneEventClient(
        base_url="https://control-plane.example",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    event = RuntimeEvent(
        event_id="evt-9",
        session_id="session-1",
        run_id="run-1",
        parent_event_id=None,
        actor="primary_agent",
        event_type="shell_exec",
        phase=EventPhase.COMPLETED,
        raw_payload={"command": "cat .env", "stdout": "SECRET=1"},
        redacted_payload={"command": "cat .env", "stdout": "[redacted]"},
        summary_payload={"command": "cat .env", "status": "completed"},
        projection_tags=(ProjectionTag.AUDIT_VISIBLE,),
    )

    response = client.upload(event)

    assert response.status_code == 202
    assert "SECRET=1" not in captured["json"]
    assert '"status":"completed"' in captured["json"]
```

- [ ] **Step 2: Run the control-plane tests to verify the client is missing**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_control_plane.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'deepagents_coding_platform.control_plane'
```

- [ ] **Step 3: Implement the metadata-safe export client**

```python
# deepagents-coding-platform/src/deepagents_coding_platform/control_plane.py
from dataclasses import dataclass

import httpx

from deepagents_coding_platform.events import RuntimeEvent


@dataclass(slots=True)
class ControlPlaneEventClient:
    base_url: str
    http_client: httpx.Client | None = None

    def upload(self, event: RuntimeEvent) -> httpx.Response:
        client = self.http_client or httpx.Client(base_url=self.base_url, timeout=5.0)
        payload = {
            "event_id": event.event_id,
            "session_id": event.session_id,
            "run_id": event.run_id,
            "event_type": event.event_type,
            "phase": event.phase.value,
            "payload": dict(event.summary_payload or event.redacted_payload),
            "projection_tags": [tag.value for tag in event.projection_tags],
        }
        response = client.post(f"{self.base_url}/v1/runtime-events", json=payload)
        response.raise_for_status()
        return response
```

- [ ] **Step 4: Run the control-plane tests again**

Run:

```bash
cd deepagents-coding-platform
uv run pytest tests/test_control_plane.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit the control-plane hook**

Run:

```bash
git add deepagents-coding-platform/src/deepagents_coding_platform/control_plane.py \
  deepagents-coding-platform/tests/test_control_plane.py
git commit -F - <<'EOF'
Add a metadata-safe control-plane event export client

Export summary-grade runtime events to a hosted endpoint without
promoting the control plane into the system's only source of execution
truth.

Constraint: The hosted plane must remain metadata-first by default
Rejected: Post raw payloads whenever audit-visible is set | would violate the spec's trust-boundary posture
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Only upload summary or redacted payload tiers from this client
Tested: uv run pytest tests/test_control_plane.py -q
Not-tested: End-to-end delivery with a live hosted control-plane service
EOF
```

Expected:

```text
[branch-name yz01234] Add a metadata-safe control-plane event export client
```

## Task 10: Run the Full Verification Sweep and Tighten the Developer Entry Point

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
- Test: `deepagents-coding-platform/tests/test_cli.py`
- Test: `deepagents-coding-platform/tests/test_control_plane.py`

- [ ] **Step 1: Add a concrete README verification section**

```text
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
Compiling 'src/deepagents_coding_platform/cli.py'...
```

- [ ] **Step 3: Run the full test suite**

Run:

```bash
cd deepagents-coding-platform
uv run pytest -q
```

Expected:

```text
16 passed
```

- [ ] **Step 4: Run the CLI help smoke test**

Run:

```bash
cd deepagents-coding-platform
uv run dacp --help
```

Expected:

```text
Usage: dacp [OPTIONS] COMMAND [ARGS]...
```

- [ ] **Step 5: Commit the verified P1 slice**

Run:

```bash
git add deepagents-coding-platform/README.md
git commit -F - <<'EOF'
Document and verify the P1 runtime kernel slice

Close the first implementation slice with a verified developer entry
point and a full test sweep across the runtime kernel, runner, CLI,
adapter, and control-plane hook.

Constraint: P1 must stop at local execution truth plus minimal CLI/control-plane hooks
Rejected: Expand immediately into multi-tenant control-plane workflows | would break the scoped-first delivery strategy
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Do not widen the first implementation plan beyond runner truth, local recovery, and metadata-safe export
Tested: uv run python -m compileall src
Tested: uv run pytest -q
Tested: uv run dacp --help
Not-tested: Real-user repository sessions or organization-level SaaS administration
EOF
```

Expected:

```text
[branch-name abc6789] Document and verify the P1 runtime kernel slice
```
