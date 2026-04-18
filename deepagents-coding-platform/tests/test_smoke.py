from deepagents_coding_platform.runtime import RuntimeKernel


def test_runtime_kernel_starts_with_empty_plugin_registry():
    kernel = RuntimeKernel(session_id="session-1", run_id="run-1")

    assert kernel.plugins.tools == {}
    assert kernel.plugins.subagent_archetypes == {}
