import re
import subprocess
from pathlib import Path

from deepagents_coding_platform.actions import ActionRequest
from deepagents_coding_platform.runner import Executor


def _resolve_workspace_path(workspace_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(workspace_root):
        raise ValueError(f"path escapes workspace: {raw_path}")
    return resolved


def build_coding_named_executors(workspace_root: Path) -> dict[str, Executor]:
    root = Path(workspace_root).resolve()

    def read_file(action: ActionRequest):
        path = _resolve_workspace_path(root, str(action.payload.get("path", ".")))
        text = path.read_text(encoding="utf-8")
        if "start_line" in action.payload or "end_line" in action.payload:
            lines = text.splitlines()
            start_line = int(action.payload.get("start_line", 1))
            end_line = int(action.payload.get("end_line", len(lines)))
            content = "\n".join(lines[start_line - 1 : end_line])
        else:
            content = text
        return {"path": path.relative_to(root).as_posix(), "content": content}

    def write_file(action: ActionRequest):
        path = _resolve_workspace_path(root, str(action.payload.get("path", ".")))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(action.payload.get("content", ""))
        path.write_text(content, encoding="utf-8")
        return {
            "path": path.relative_to(root).as_posix(),
            "bytes_written": len(content.encode("utf-8")),
        }

    def list_files(action: ActionRequest):
        path = _resolve_workspace_path(root, str(action.payload.get("path", ".")))
        recursive = bool(action.payload.get("recursive", True))
        limit = int(action.payload.get("limit", 200))
        pattern = "**/*" if recursive else "*"
        entries = sorted(
            item.relative_to(root).as_posix()
            for item in path.glob(pattern)
            if item != path
        )
        return {"entries": entries[:limit]}

    def grep_search(action: ActionRequest):
        path = _resolve_workspace_path(root, str(action.payload.get("path", ".")))
        pattern = re.compile(str(action.payload.get("pattern", "")))
        glob_pattern = str(action.payload.get("glob", "**/*"))
        limit = int(action.payload.get("limit", 200))
        matches: list[dict[str, object]] = []
        for candidate in sorted(path.glob(glob_pattern)):
            if not candidate.is_file():
                continue
            for line_number, line in enumerate(
                candidate.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if pattern.search(line):
                    matches.append(
                        {
                            "path": candidate.relative_to(root).as_posix(),
                            "line_number": line_number,
                            "line": line,
                        }
                    )
                    if len(matches) >= limit:
                        return {"matches": matches}
        return {"matches": matches}

    def shell(action: ActionRequest):
        command = str(action.payload.get("command", ""))
        timeout_seconds = int(action.payload.get("timeout_seconds", 60))
        try:
            result = subprocess.run(
                command,
                cwd=root,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "returncode": None,
                "error": f"command timed out after {timeout_seconds}s",
            }

    def apply_patch(action: ActionRequest):
        path = _resolve_workspace_path(root, str(action.payload.get("path", ".")))
        original = path.read_text(encoding="utf-8")
        updated = original
        edits = list(action.payload.get("edits", []))

        for edit in edits:
            old = str(edit["old"])
            new = str(edit["new"])
            if old not in updated:
                return {
                    "path": path.relative_to(root).as_posix(),
                    "error": f"patch context not found: {old}",
                }
            updated = updated.replace(old, new, 1)

        path.write_text(updated, encoding="utf-8")
        return {
            "path": path.relative_to(root).as_posix(),
            "applied": len(edits),
        }

    return {
        "read_file": read_file,
        "write_file": write_file,
        "list_files": list_files,
        "grep_search": grep_search,
        "shell": shell,
        "apply_patch": apply_patch,
    }
