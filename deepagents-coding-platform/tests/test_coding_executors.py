from deepagents_coding_platform.actions import ActionKind, ActionRequest
from deepagents_coding_platform.coding.executors import build_coding_named_executors


def test_read_and_write_file_round_trip(tmp_path):
    executors = build_coding_named_executors(tmp_path)

    write_result = executors["write_file"](
        ActionRequest(
            kind=ActionKind.FS_WRITE,
            name="write_file",
            payload={"path": "src/app.py", "content": "print('hi')\n"},
        )
    )
    read_result = executors["read_file"](
        ActionRequest(
            kind=ActionKind.FS_READ,
            name="read_file",
            payload={"path": "src/app.py"},
        )
    )

    assert write_result["path"] == "src/app.py"
    assert read_result["content"] == "print('hi')\n"


def test_list_files_returns_workspace_entries(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    executors = build_coding_named_executors(tmp_path)

    result = executors["list_files"](
        ActionRequest(
            kind=ActionKind.FS_READ,
            name="list_files",
            payload={"path": ".", "recursive": True},
        )
    )

    assert "src/app.py" in result["entries"]


def test_grep_search_returns_matching_lines(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "one.py").write_text("needle = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "two.py").write_text("other = 2\n", encoding="utf-8")
    executors = build_coding_named_executors(tmp_path)

    result = executors["grep_search"](
        ActionRequest(
            kind=ActionKind.FS_READ,
            name="grep_search",
            payload={"path": ".", "pattern": "needle"},
        )
    )

    assert result["matches"][0]["path"] == "pkg/one.py"
    assert result["matches"][0]["line_number"] == 1


def test_apply_patch_replaces_exact_context(tmp_path):
    target = tmp_path / "main.py"
    target.write_text("name = 'old'\n", encoding="utf-8")
    executors = build_coding_named_executors(tmp_path)

    result = executors["apply_patch"](
        ActionRequest(
            kind=ActionKind.FS_WRITE,
            name="apply_patch",
            payload={
                "path": "main.py",
                "edits": [{"old": "name = 'old'\n", "new": "name = 'new'\n"}],
            },
        )
    )

    assert result["applied"] == 1
    assert target.read_text(encoding="utf-8") == "name = 'new'\n"


def test_apply_patch_reports_missing_context(tmp_path):
    target = tmp_path / "main.py"
    target.write_text("name = 'old'\n", encoding="utf-8")
    executors = build_coding_named_executors(tmp_path)

    result = executors["apply_patch"](
        ActionRequest(
            kind=ActionKind.FS_WRITE,
            name="apply_patch",
            payload={
                "path": "main.py",
                "edits": [{"old": "name = 'missing'\n", "new": "name = 'new'\n"}],
            },
        )
    )

    assert "error" in result
    assert "patch context not found" in result["error"]


def test_shell_runs_inside_workspace_and_returns_output(tmp_path):
    executors = build_coding_named_executors(tmp_path)

    result = executors["shell"](
        ActionRequest(
            kind=ActionKind.SHELL_EXEC,
            name="shell",
            payload={"command": "printf hello"},
        )
    )

    assert result["stdout"] == "hello"
    assert result["returncode"] == 0
