import pytest

from deepagents_coding_platform.actions import ActionKind
from deepagents_coding_platform.adapters.deepagents_runtime import (
    DeepagentsRuntimeAdapter,
    RuntimeToolSpec,
)
from deepagents_coding_platform.policy import PolicyDecision, PolicyOutcome


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


def test_adapter_raises_when_runtime_blocks_action():
    class FakeRunner:
        def run_action(self, action):
            return type(
                "Result",
                (),
                {
                    "decision": PolicyDecision(
                        outcome=PolicyOutcome.DENY,
                        reason="path escapes workspace",
                    ),
                    "output": None,
                },
            )()

    adapter = DeepagentsRuntimeAdapter(runner=FakeRunner())
    tool = adapter.build_tool(
        RuntimeToolSpec(
            name="read_file",
            description="Read a file through the runtime kernel.",
            action_kind=ActionKind.FS_READ,
        )
    )

    with pytest.raises(RuntimeError, match="path escapes workspace"):
        tool(path="../secret.txt")
