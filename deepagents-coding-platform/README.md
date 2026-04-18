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
