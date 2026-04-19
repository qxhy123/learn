from deepagents_coding_platform.coding.policy import WorkspaceCodingPolicy
from deepagents_coding_platform.coding.preset import (
    build_coding_agent,
    build_coding_runner,
)


def test_build_coding_runner_registers_policy_and_all_named_tools(tmp_path):
    runner = build_coding_runner(
        workspace_root=tmp_path,
        ledger_root=tmp_path / ".ledger",
    )

    assert isinstance(runner.kernel.plugins.policy_evaluators[0], WorkspaceCodingPolicy)
    assert sorted(runner.named_executors) == [
        "apply_patch",
        "grep_search",
        "list_files",
        "read_file",
        "shell",
        "write_file",
    ]


def test_build_coding_agent_uses_fixed_prompt_and_six_tools(monkeypatch, tmp_path):
    captured = {}

    def fake_create_agent(self, *, model, system_prompt, tool_specs):
        captured["model"] = model
        captured["system_prompt"] = system_prompt
        captured["tool_specs"] = tool_specs
        return "agent-object"

    monkeypatch.setattr(
        "deepagents_coding_platform.coding.preset.DeepagentsRuntimeAdapter.create_agent",
        fake_create_agent,
    )

    agent = build_coding_agent(
        model="openai:gpt-4.1",
        workspace_root=tmp_path,
        ledger_root=tmp_path / ".ledger",
    )

    assert agent == "agent-object"
    assert captured["model"] == "openai:gpt-4.1"
    assert len(captured["tool_specs"]) == 6
    assert "explore before modifying" in captured["system_prompt"].lower()
