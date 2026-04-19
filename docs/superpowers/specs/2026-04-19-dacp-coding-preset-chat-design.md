# DACP Coding Preset and Chat Design

**Date:** 2026-04-19
**Project:** `deepagents-coding-platform`
**Status:** Approved for planning

## Goal

Add a first-class coding preset on top of the existing P1 runtime kernel so the project can act as a workspace-scoped coding agent. The preset must provide built-in coding tools, a workspace-safe execution policy, and a `dacp chat` REPL that runs a deepagents agent through the runtime/runner path.

## Why This Exists

The current P1 slice proves the execution kernel pieces independently:

- typed actions and events
- deterministic policy evaluation
- visibility projection
- local ledger and checkpoints
- local runner
- deepagents adapter
- minimal CLI
- control-plane export hook

What it does not yet provide is a usable "write code for me" workflow. This design closes that gap without widening scope into approvals UI, streaming, or hosted orchestration.

## Product Shape

This feature adds a `coding preset` above the existing runtime stack rather than changing the runtime contract itself.

The execution chain remains:

`deepagents agent -> runtime-wrapped tools -> LocalRunner -> RuntimeKernel -> policy/events/checkpoints`

The new preset provides:

- a workspace-aware policy for coding actions
- built-in coding executors
- tool specs for deepagents integration
- a `dacp chat` REPL for iterative coding sessions

## Chosen Approach

Implement a new coding preset layer on top of the current P1 runtime kernel.

This was chosen over:

1. putting coding logic directly into the CLI
   - rejected because it would collapse assembly, execution, policy, and terminal UX into one file
2. broadening the runtime into a general tool platform first
   - rejected because it would widen scope beyond the first usable coding slice

## Architecture

The implementation is split into four responsibilities:

1. `coding policy`
   - decides whether a coding action is allowed inside the workspace
2. `coding executors`
   - perform real file/search/patch/shell actions inside the workspace
3. `coding preset assembly`
   - wires policy, executors, runner, adapter, and tool specs together
4. `chat REPL`
   - runs a local interactive session using a deepagents agent built from the preset

This keeps the existing runtime contract intact. The runtime remains the source of truth for policy decisions, event ordering, projections, and checkpoints. The preset is only a specialized consumer of that runtime.

## Module Layout

### New Files

- `deepagents-coding-platform/src/deepagents_coding_platform/coding/policy.py`
  - workspace-scoped coding policy
- `deepagents-coding-platform/src/deepagents_coding_platform/coding/executors.py`
  - built-in coding executors
- `deepagents-coding-platform/src/deepagents_coding_platform/coding/preset.py`
  - preset assembly helpers
- `deepagents-coding-platform/src/deepagents_coding_platform/chat.py`
  - REPL loop

### Modified Files

- `deepagents-coding-platform/src/deepagents_coding_platform/cli.py`
  - add `chat`
- `deepagents-coding-platform/README.md`
  - document coding preset usage after implementation

## Workspace-Safe Policy

The preset runs in safety mode `A`: workspace-scoped auto execution with dangerous commands blocked.

### Policy Rules

All file paths must resolve inside `workspace_root`.

- paths outside `workspace_root` are `DENY`
- workspace-local reads are `ALLOW`
- workspace-local writes are `ALLOW`
- workspace-local structured patches are `ALLOW`
- workspace-local listing and grep/search actions are `ALLOW`
- shell commands run with `cwd=workspace_root`

### Dangerous Command Handling

The following shell command prefixes are denied:

- `rm -rf`
- `sudo`
- `shutdown`
- `reboot`
- `mkfs`
- `dd `
- `git push`

The following download-and-execute patterns are denied:

- `curl ... | sh`
- `curl ... | bash`
- `wget ... | sh`
- `wget ... | bash`

Typical local development commands inside the workspace are allowed, including:

- test runs
- formatting
- linting
- `git status`
- `git diff`

The policy must stay deterministic. It is not a model-based safety layer and must not inspect free-form intent beyond the declared action and payload.

## Built-In Coding Tools

The preset provides six tools to the deepagents agent.

### `read_file`

Inputs:

- `path`
- optional `start_line`
- optional `end_line`

Behavior:

- reads a text file inside the workspace
- supports line slicing to avoid large context dumps

### `write_file`

Inputs:

- `path`
- `content`

Behavior:

- writes or overwrites a workspace-local file
- creates parent directories inside the workspace when needed

### `list_files`

Inputs:

- optional `path`, default `.`
- optional `recursive`, default `true`
- optional `limit`, default bounded

Behavior:

- lists workspace files/directories for exploration

### `grep_search`

Inputs:

- `pattern`
- optional `path`, default `.`
- optional `glob`
- optional `limit`

Behavior:

- searches text under the workspace
- used for symbol/callsite discovery

### `shell`

Inputs:

- `command`
- optional `timeout_seconds`

Behavior:

- executes in `workspace_root`
- returns structured output with `stdout`, `stderr`, and `returncode`

### `apply_patch`

Inputs:

- `path`
- `edits`

Behavior:

- performs exact context replacements on a workspace-local file
- each edit is `{old, new}`
- replacement only succeeds when `old` matches exactly
- no fuzzy patching
- fails explicitly on missing context

### Why Both `write_file` and `apply_patch` Exist

- `write_file` is for new files and full rewrites
- `apply_patch` is for local edits

This keeps the first editing surface simple while still supporting narrow code changes.

## Coding Preset Assembly

The preset assembly layer is responsible for producing two things:

- a `LocalRunner` configured for coding work
- a deepagents agent built from that runner

The preset owns:

- the workspace-safe coding policy
- the built-in executor map
- the tool specs exposed to the agent
- the coding system prompt

It does not own:

- runtime event ordering
- policy/event storage format
- checkpoint semantics
- generic adapter behavior outside the coding preset

## `dacp chat` REPL

### CLI Shape

The first interactive coding command is:

`dacp chat --model <provider:model> --workspace <path> --ledger-root <path>`

Rules:

- `--model` is required
- `--workspace` is required
- `--ledger-root` is required

This avoids hidden defaults for model or execution root.

### Session Model

The first version is a local REPL.

Per turn:

1. user enters a prompt
2. the prompt is appended to in-memory message history
3. the deepagents agent is invoked with the current history
4. the assistant response is printed
5. the assistant response is appended back to history
6. the REPL waits for the next user message

### State Boundaries

Persistent state:

- runtime ledger
- checkpoints
- runtime events

Ephemeral state:

- chat message history

This means `resume-session` continues to expose runtime recovery state, but the first `dacp chat` version does not resume full conversational context after process restart.

## System Prompt Policy

The coding preset uses a fixed system prompt that enforces these behaviors:

- explore before modifying
- prefer `list_files`, `grep_search`, and `read_file` before editing
- prefer `apply_patch` for local edits
- use `write_file` for new files or full rewrites
- run validation with `shell` after edits
- operate only inside the workspace
- do not claim a command or edit succeeded unless the tool returned success
- when blocked by policy, explain the block and try a safer alternative

This prompt is part of the preset, not a user responsibility.

## Error Handling

### Policy Denial / Approval Required

If the runtime returns `DENY` or `REQUIRE_APPROVAL`:

- the tool layer must surface that to the agent as an explicit failed tool outcome
- the agent must not receive an empty success-shaped payload

The goal is to keep the model aware that execution did not happen.

### Executor Failure

Executors return structured failure information where applicable:

- `error`
- `stderr`
- `returncode`

This allows the agent to recover by changing the command, narrowing the edit, or asking the user for a different path.

### REPL Failure

If a turn fails unexpectedly:

- print a concise error to the terminal
- keep the REPL process alive
- do not discard the session unless startup itself failed

## Streaming and Approval Scope

The first version explicitly does not include:

- token streaming
- runtime event streaming to the terminal
- interactive approval UI

This is intentional. The first milestone is a stable coding preset that can safely execute inside a workspace, not a full hosted or streaming terminal agent.

## Testing Strategy

Implementation must add tests for four areas.

### Coding Policy Tests

- workspace-local reads are allowed
- workspace-local writes are allowed
- out-of-workspace paths are denied
- dangerous shell commands are denied
- normal local development commands are allowed

### Coding Executor Tests

- `read_file`
- `write_file`
- `list_files`
- `grep_search`
- `apply_patch` success
- `apply_patch` exact-match failure
- `shell` structured output in workspace cwd

### Chat Loop Tests

- REPL performs one full input/output turn
- agent invocation uses message history
- loop survives a handled turn failure

### CLI Chat Tests

- `dacp chat` argument validation
- command starts with required args
- one fake end-to-end session succeeds

## Acceptance Criteria

The feature is complete when all of the following are true:

- `dacp chat --model ... --workspace ... --ledger-root ...` starts a local REPL
- the REPL-backed agent can read files, patch files, write files, search, list files, and run local commands inside the workspace
- attempts to access paths outside the workspace are denied
- dangerous commands are denied
- denied or approval-required actions are surfaced clearly to the agent and user
- runtime execution continues to flow through the existing runner/kernel/ledger path
- tests cover the coding preset, built-in executors, REPL loop, and CLI surface

## Explicit Non-Goals

This slice does not include:

- token streaming
- cross-process chat session recovery
- interactive approval prompts
- multi-workspace management
- full unified diff parsing
- hosted orchestration workflows
- AST/symbol-aware editing
- IDE integration

## Planned Follow-On Work

After this slice is stable, the expected follow-on order is:

1. interactive approval flow
2. streaming terminal UX
3. stronger patch protocol
4. persistent chat sessions
5. richer coding intelligence
6. hosted/team workflows

## Scope Check

This spec is intentionally a single implementation slice:

- one preset
- one REPL command
- one built-in tool family
- one workspace safety model

It is small enough for a single implementation plan and does not require decomposition into separate project specs.
