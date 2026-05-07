#!/usr/bin/env python3
"""Check relative local links in Markdown and HTML files."""

from __future__ import annotations

import argparse
import html.parser
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path


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
PUBLIC_DIRS = {
    "appendix",
    "code",
    "html",
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

HTML_ATTRS = {"href", "src"}
MARKDOWN_LINK_RE = re.compile(r"(!?)\[[^\]]*\]\(([^)\s]+)(?:\s+['\"][^)]*['\"])?\)")
MARKDOWN_REF_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s+(\S+)", re.MULTILINE)


@dataclass(frozen=True)
class Link:
    source: Path
    target: str
    kind: str
    line: int


class HtmlLinkParser(html.parser.HTMLParser):
    def __init__(self, source: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.source = source
        self.links: list[Link] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in HTML_ATTRS and value:
                self.links.append(Link(self.source, value, name, self.getpos()[0]))


def is_in_scope(path: Path, root: Path, include_internal: bool) -> bool:
    if include_internal:
        return True
    rel = path.relative_to(root)
    if len(rel.parts) == 1:
        return rel.name in PUBLIC_ROOT_FILES
    return rel.parts[0] in PUBLIC_DIRS


def iter_files(root: Path, html_only: bool, include_internal: bool) -> list[Path]:
    suffixes = {".html"} if html_only else {".html", ".htm", ".md", ".markdown"}
    files: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        base = Path(current_root)
        for filename in filenames:
            path = base / filename
            if path.suffix.lower() in suffixes and is_in_scope(path, root, include_internal):
                files.append(path)
    return sorted(files)


def html_links(path: Path) -> list[Link]:
    parser = HtmlLinkParser(path)
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.links


def markdown_links(path: Path) -> list[Link]:
    text = path.read_text(encoding="utf-8")
    links: list[Link] = []
    for regex, kind_offset in ((MARKDOWN_LINK_RE, 2), (MARKDOWN_REF_RE, 1)):
        for match in regex.finditer(text):
            raw = match.group(kind_offset)
            if raw.startswith("<") and raw.endswith(">"):
                raw = raw[1:-1]
            line = text.count("\n", 0, match.start()) + 1
            kind = "image" if kind_offset == 2 and match.group(1) else "link"
            links.append(Link(path, raw, kind, line))
    return links


def is_local_reference(raw: str) -> bool:
    if not raw or raw.startswith("#"):
        return False
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme:
        return False
    if raw.startswith("//"):
        return False
    return True


def target_exists(source: Path, raw: str) -> bool:
    path_part = urllib.parse.unquote(urllib.parse.urlsplit(raw).path)
    if not path_part:
        return True
    target = (source.parent / path_part).resolve()
    return target.exists()


def check(root: Path, html_only: bool, include_internal: bool) -> list[Link]:
    broken: list[Link] = []
    for path in iter_files(root, html_only, include_internal):
        links = html_links(path) if path.suffix.lower() in {".html", ".htm"} else markdown_links(path)
        for link in links:
            if is_local_reference(link.target) and not target_exists(link.source, link.target):
                broken.append(link)
    return broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="repository root to scan")
    parser.add_argument("--html-only", action="store_true", help="scan only HTML files")
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help="also scan docs/superpowers and other non-public planning files",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    broken = check(root, args.html_only, args.include_internal)
    for link in broken:
        source = link.source.relative_to(root)
        print(f"{source}:{link.line}: broken {link.kind} -> {link.target}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
