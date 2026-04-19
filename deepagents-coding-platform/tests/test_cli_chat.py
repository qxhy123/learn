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
    assert "--model" in result.output


def test_chat_help_marks_model_as_required():
    runner = CliRunner()

    result = runner.invoke(app, ["chat", "--help"])

    assert result.exit_code == 0
    model_line = next(
        line for line in result.stdout.splitlines() if "--model" in line
    )
    assert "[required]" in model_line


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
