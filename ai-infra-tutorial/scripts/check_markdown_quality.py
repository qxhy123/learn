#!/usr/bin/env python3
"""Check public Markdown sources for tutorial quality issues."""

from __future__ import annotations

import os
import re
import sys
import urllib.parse
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

MARKDOWN_LINK_RE = re.compile(r"(!?)\[[^\]]*\]\(([^)\s]+)(?:\s+['\"][^)]*['\"])?\)")
MARKDOWN_REF_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s+(\S+)", re.MULTILINE)
H1_RE = re.compile(r"^#(?!#)\s+\S", re.MULTILINE)
FENCE_RE = re.compile(r"^\s*(```|~~~)")
BAD_PATH_PATTERNS = (
    "16a-vllm-inference.md",
    "16b-sglang-radix-attention.md",
    "15-batching-scheduling-kv-cache.md",
    "part4-training-systems",
)
DANGEROUS_ASSERTIONS = ("永远", "必然", "完全免费", "默认开启")


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    level: str
    message: str


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


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def markdown_links(text: str) -> list[tuple[str, int]]:
    links: list[tuple[str, int]] = []
    for regex, group_index in ((MARKDOWN_LINK_RE, 2), (MARKDOWN_REF_RE, 1)):
        for match in regex.finditer(text):
            raw = match.group(group_index)
            if raw.startswith("<") and raw.endswith(">"):
                raw = raw[1:-1]
            links.append((raw, line_number(text, match.start())))
    return links


def iter_markdown_content_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    in_fence = False
    for index, line in enumerate(text.splitlines(), start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append((index, line))
    return lines


def is_local_markdown_link(raw: str) -> bool:
    if not raw or raw.startswith("#"):
        return False
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme or raw.startswith("//"):
        return False
    return parsed.path.endswith(".html")


def check_file(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    findings: list[Finding] = []
    content_lines = iter_markdown_content_lines(text)

    h1_lines = [line_number for line_number, line in content_lines if H1_RE.match(line)]
    if len(h1_lines) > 1:
        findings.append(
            Finding(path, h1_lines[1], "ERROR", "more than one H1 heading")
        )

    for raw, line in markdown_links(text):
        if is_local_markdown_link(raw):
            findings.append(Finding(path, line, "ERROR", f"Markdown internal link points to .html: {raw}"))

    for line_number_, line in content_lines:
        for pattern in BAD_PATH_PATTERNS:
            if pattern in line:
                findings.append(Finding(path, line_number_, "ERROR", f"known stale path pattern: {pattern}"))
        for word in DANGEROUS_ASSERTIONS:
            if word in line:
                findings.append(Finding(path, line_number_, "WARNING", f"dangerous assertion term: {word}"))

    return findings


def check(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_markdown_files(root):
        findings.extend(check_file(path))
    return findings


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[0] if argv else ".").resolve()
    findings = check(root)
    for finding in findings:
        rel = finding.path.relative_to(root)
        print(f"{rel}:{finding.line}: {finding.level}: {finding.message}")
    return 1 if any(finding.level == "ERROR" for finding in findings) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
