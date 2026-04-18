import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from deepagents_coding_platform.events import RuntimeEvent


@dataclass(slots=True, frozen=True)
class ResumeState:
    checkpoint_name: str | None
    state: Mapping[str, Any]
    events_after_checkpoint: list[RuntimeEvent]


class SessionLedger:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.checkpoints_dir = self.root / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.latest_path = self.root / "latest_checkpoint.json"

    def append_event(self, event: RuntimeEvent) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict()) + "\n")

    def read_events(self) -> list[RuntimeEvent]:
        if not self.events_path.exists():
            return []

        with self.events_path.open(encoding="utf-8") as handle:
            return [
                RuntimeEvent.from_dict(json.loads(line))
                for line in handle
                if line.strip()
            ]

    def commit_checkpoint(self, name: str, state: Mapping[str, Any]) -> Path:
        events = self.read_events()
        payload = {"name": name, "state": dict(state), "event_cursor": len(events)}
        checkpoint_path = self.checkpoints_dir / f"{name}.json"
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return checkpoint_path

    def resume(self) -> ResumeState:
        events = self.read_events()
        if not self.latest_path.exists():
            return ResumeState(
                checkpoint_name=None,
                state={},
                events_after_checkpoint=events,
            )

        checkpoint = json.loads(self.latest_path.read_text(encoding="utf-8"))
        cursor = int(checkpoint["event_cursor"])
        return ResumeState(
            checkpoint_name=str(checkpoint["name"]),
            state=dict(checkpoint["state"]),
            events_after_checkpoint=events[cursor:],
        )
