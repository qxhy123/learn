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
    model: str | None = typer.Option(None),
    workspace: Path = typer.Option(...),
    ledger_root: Path = typer.Option(...),
) -> None:
    if model is None:
        console.print("Missing option '--model'.")
        raise typer.Exit(code=2)

    session = _build_chat_session(
        model=model,
        workspace=workspace,
        ledger_root=ledger_root,
    )
    session.repl()
