#!/usr/bin/env python3
"""Audit public Markdown sources for version and component drift candidates."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PUBLIC_DIRS = {
    "appendix",
    "part0-foundations-of-systems",
    "part1-foundations",
    "part2-systems-stack",
    "part3-training-infra",
    "part4-data-and-storage",
    "part5-serving-infra",
    "part6-platform-and-orchestration",
    "part7-reliability-security",
    "part8-advanced-and-capstone",
}
PUBLIC_ROOT_FILES = {"README.md", "00-preface.md"}
EXTRA_PUBLIC_FILES = {Path("code/mini-vllm/README.md")}
SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}

DRIFT_RE = re.compile(
    r"\b("
    r"v\d+\.\d+"
    r"|CUDA\s+12\.\d+"
    r"|PyTorch\s+2\.\d+"
    r"|Prometheus\s+3\.8"
    r"|H100|H200|B200|GB200"
    r"|NCCL"
    r"|vLLM"
    r"|SGLang"
    r"|TensorRT-LLM"
    r"|Kueue"
    r")\b"
)
SUGGESTION = "建议核对 appendix/version-matrix.md"


@dataclass(frozen=True)
class Match:
    path: Path
    line: int
    term: str


def is_public_markdown(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if rel in EXTRA_PUBLIC_FILES:
        return True
    if len(rel.parts) == 1:
        return rel.name in PUBLIC_ROOT_FILES
    return rel.parts[0] in PUBLIC_DIRS


def iter_markdown_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        base = Path(current_root)
        for filename in filenames:
            path = base / filename
            if path.suffix.lower() == ".md" and is_public_markdown(path, root):
                files.append(path)
    return sorted(files)


def audit_file(path: Path) -> list[Match]:
    matches: list[Match] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for match in DRIFT_RE.finditer(line):
            matches.append(Match(path, line_number, match.group(0)))
    return matches


def audit(root: Path) -> list[Match]:
    matches: list[Match] = []
    for path in iter_markdown_files(root):
        matches.extend(audit_file(path))
    return matches


def self_test() -> int:
    sample = "CUDA 12.4, PyTorch 2.5, Prometheus 3.8, H100, vLLM, SGLang, TensorRT-LLM, Kueue, v1.2"
    found = [match.group(0) for match in DRIFT_RE.finditer(sample)]
    expected = [
        "CUDA 12.4",
        "PyTorch 2.5",
        "Prometheus 3.8",
        "H100",
        "vLLM",
        "SGLang",
        "TensorRT-LLM",
        "Kueue",
        "v1.2",
    ]
    if found != expected:
        print(f"self-test failed: expected {expected}, got {found}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args == ["--self-test"]:
        return self_test()

    root = Path(args[0] if args else ".").resolve()
    for item in audit(root):
        rel = item.path.relative_to(root)
        print(f"{rel}:{item.line}: {item.term}: {SUGGESTION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
