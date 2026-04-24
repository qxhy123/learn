from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.validate_metadata import MetadataError, load_metadata, validate_metadata
except ImportError:  # pragma: no cover - supports `python scripts/generate_scaffold.py`
    from validate_metadata import MetadataError, load_metadata, validate_metadata


def write_file(path: Path, content: str, force: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def module_import_path(solution_path: str) -> str:
    return solution_path.removesuffix(".py").replace("/", ".")


def render_doc(problem: dict[str, Any]) -> str:
    patterns = ", ".join(problem["patterns"])
    example_lines: list[str] = []
    for index, example in enumerate(problem["examples"], start=1):
        example_lines.extend(
            [
                f"### Example {index}",
                f"- Input: `{example['input']!r}`",
                f"- Output: `{example['output']!r}`",
                "",
            ]
        )

    return "\n".join(
        [
            f"# {problem['number']}. {problem['title']}",
            "",
            f"- Difficulty: {problem['difficulty']}",
            f"- LeetCode: {problem['leetcode_url']}",
            f"- Official Group: {problem['official_group']}",
            f"- Pattern Group: {problem['pattern_group']}",
            f"- Patterns: {patterns}",
            "",
            "## Core Pattern",
            "TODO",
            "",
            "## When To Use It",
            "TODO",
            "",
            "## Approach",
            "TODO",
            "",
            "## Correctness Sketch",
            "TODO",
            "",
            "## Complexity",
            "TODO",
            "",
            "## Common Pitfalls",
            "TODO",
            "",
            "## Implementation",
            f"See `{problem['solution_path']}`.",
            "",
            "## Tests",
            f"See `{problem['test_path']}`.",
            "",
            "## Examples",
            "",
            *example_lines,
            "## Follow-up Practice",
            "TODO",
            "",
        ]
    )


def render_solution(problem: dict[str, Any]) -> str:
    signature = str(problem["signature"]).rstrip()
    if not signature.endswith(":"):
        signature = f"{signature}:"

    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "",
            "class Solution:",
            f'    """See `{problem["doc_path"]}`."""',
            "",
            f"    {signature}",
            '        raise NotImplementedError("Implement the solution described in the tutorial.")',
            "",
        ]
    )


def render_test(problem: dict[str, Any]) -> str:
    import_path = module_import_path(problem["solution_path"])
    method_name = problem["method_name"]

    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import pytest",
            "",
            f"from {import_path} import Solution",
            "",
            'pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")',
            "",
            f"EXAMPLES = {problem['examples']!r}",
            "",
            "",
            "def test_official_examples() -> None:",
            "    solution = Solution()",
            f'    # Equivalent direct call form: Solution().{method_name}(**example["input"])',
            "    for example in EXAMPLES:",
            f'        result = solution.{method_name}(**example["input"])',
            '        assert result == example["output"]',
            "",
        ]
    )


def render_official_order(problems: list[dict[str, Any]]) -> str:
    groups: dict[str, list[dict[str, Any]]] = {}
    for problem in problems:
        groups.setdefault(problem["official_group"], []).append(problem)

    lines = ["# Top Interview 150 in Official Order", ""]
    for group_name, group_problems in groups.items():
        lines.append(f"## {group_name}")
        lines.append("")
        for problem in group_problems:
            lines.append(f"- [{problem['number']}. {problem['title']}]({problem['doc_path']}) ({problem['difficulty']})")
        lines.append("")
    return "\n".join(lines)


def render_pattern_roadmap(problems: list[dict[str, Any]]) -> str:
    groups: dict[str, list[dict[str, Any]]] = {}
    for problem in problems:
        groups.setdefault(problem["pattern_group"], []).append(problem)

    lines = ["# Pattern Roadmap", ""]
    for group_name in sorted(groups):
        lines.append(f"## {group_name}")
        lines.append("")
        for problem in groups[group_name]:
            tags = ", ".join(problem["patterns"])
            lines.append(f"- [{problem['number']}. {problem['title']}]({problem['doc_path']}) — {tags}")
        lines.append("")
    return "\n".join(lines)


def generate_scaffold(metadata: dict[str, Any], root: Path, force: bool = False) -> list[str]:
    written: list[str] = []
    problems = metadata["problems"]

    for problem in problems:
        problem_files = {
            problem["doc_path"]: render_doc(problem),
            problem["solution_path"]: render_solution(problem),
            problem["test_path"]: render_test(problem),
        }
        for relative_path, content in problem_files.items():
            if write_file(root / relative_path, content, force=force):
                written.append(relative_path)

    generated_docs = {
        "docs/official-order.md": render_official_order(problems),
        "docs/pattern-roadmap.md": render_pattern_roadmap(problems),
    }
    for relative_path, content in generated_docs.items():
        if write_file(root / relative_path, content, force=True):
            written.append(relative_path)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate LeetCode 150 tutorial scaffold files.")
    parser.add_argument("metadata", type=Path, help="Path to top_interview_150.yaml")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Root directory for generated files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing problem files")
    args = parser.parse_args(argv)

    try:
        metadata = load_metadata(args.metadata)
    except MetadataError as error:
        print(error, file=sys.stderr)
        return 2

    errors = validate_metadata(metadata)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    written = generate_scaffold(metadata, args.root, force=args.force)
    print(f"Generated or updated {len(written)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
