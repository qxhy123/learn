from rich.console import Console

from deepagents_coding_platform.chat import ChatSession


def test_chat_session_runs_turn_and_records_history():
    class FakeAgent:
        def __init__(self):
            self.payloads = []

        def invoke(self, payload):
            self.payloads.append(payload)
            return {
                "messages": [
                    *payload["messages"],
                    {"role": "assistant", "content": "Done."},
                ]
            }

    console = Console(record=True, width=120)
    agent = FakeAgent()
    session = ChatSession(agent=agent, console=console)

    answer = session.run_turn("Fix the test suite.")

    assert answer == "Done."
    assert agent.payloads[0]["messages"] == [
        {"role": "user", "content": "Fix the test suite."}
    ]
    assert session.messages[-1] == {"role": "assistant", "content": "Done."}
    assert "Done." in console.export_text()


def test_chat_session_repl_continues_after_agent_failure():
    console = Console(record=True, width=120)
    state = {"calls": 0}

    class FakeAgent:
        def invoke(self, payload):
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("boom")
            return {
                "messages": [
                    *payload["messages"],
                    {"role": "assistant", "content": "Recovered."},
                ]
            }

    inputs = iter(["first turn", "second turn", "exit"])
    session = ChatSession(agent=FakeAgent(), console=console)
    session.repl(read_input=lambda _prompt: next(inputs))

    output = console.export_text()
    assert "boom" in output
    assert "Recovered." in output
