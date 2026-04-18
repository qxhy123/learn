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
