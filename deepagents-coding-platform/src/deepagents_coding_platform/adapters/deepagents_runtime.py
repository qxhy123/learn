from dataclasses import dataclass, field
from typing import Any, Mapping

from deepagents import create_deep_agent

from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.policy import PolicyOutcome
from deepagents_coding_platform.runner import LocalRunner


@dataclass(slots=True, frozen=True)
class RuntimeToolSpec:
    name: str
    description: str
    action_kind: ActionKind
    static_payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DeepagentsRuntimeAdapter:
    runner: LocalRunner

    def build_tool(self, spec: RuntimeToolSpec):
        def runtime_wrapped_tool(**payload):
            request = ActionRequest(
                kind=spec.action_kind,
                name=spec.name,
                payload={**dict(spec.static_payload), **payload},
            )
            result = self.runner.run_action(request)
            decision = getattr(result, "decision", None)
            if decision is not None and decision.outcome is not PolicyOutcome.ALLOW:
                raise RuntimeError(f"{decision.outcome.value}: {decision.reason}")
            return dict(result.output or {})

        runtime_wrapped_tool.__name__ = spec.name
        runtime_wrapped_tool.__doc__ = spec.description
        return runtime_wrapped_tool

    def create_agent(
        self,
        *,
        model: str,
        system_prompt: str,
        tool_specs: list[RuntimeToolSpec],
    ):
        tools = [self.build_tool(spec) for spec in tool_specs]
        return create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
        )
