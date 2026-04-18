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
