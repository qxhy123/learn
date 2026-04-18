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
