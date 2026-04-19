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
