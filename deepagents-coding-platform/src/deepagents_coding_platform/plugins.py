from dataclasses import dataclass, field
from typing import Any, Callable


ToolHandler = Callable[..., dict[str, Any]]
EventSink = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class PluginRegistry:
    tools: dict[str, ToolHandler] = field(default_factory=dict)
    policy_evaluators: list[Any] = field(default_factory=list)
    event_sinks: list[EventSink] = field(default_factory=list)
    subagent_archetypes: dict[str, Any] = field(default_factory=dict)
