from dataclasses import dataclass, field

from deepagents_coding_platform.plugins import PluginRegistry


@dataclass(slots=True)
class RuntimeKernel:
    session_id: str
    run_id: str
    plugins: PluginRegistry = field(default_factory=PluginRegistry)
