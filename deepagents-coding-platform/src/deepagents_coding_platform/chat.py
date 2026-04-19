from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Console


def _extract_assistant_text(result: dict[str, Any]) -> str:
    final_message = result["messages"][-1]
    if isinstance(final_message, dict):
        return str(final_message.get("content", ""))
    return str(getattr(final_message, "content", final_message))


@dataclass(slots=True)
class ChatSession:
    agent: Any
    console: Console
    messages: list[dict[str, str]] = field(default_factory=list)

    def run_turn(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        result = self.agent.invoke({"messages": list(self.messages)})
        answer = _extract_assistant_text(result)
        self.messages.append({"role": "assistant", "content": answer})
        self.console.print(answer)
        return answer

    def repl(self, read_input: Callable[[str], str] = input) -> None:
        while True:
            raw = read_input("dacp> ")
            user_text = raw.strip()
            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                self.console.print("Exiting DACP chat.")
                return
            try:
                self.run_turn(user_text)
            except Exception as exc:
                self.console.print(f"[red]error[/red]: {exc}")
