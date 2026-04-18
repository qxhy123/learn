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
