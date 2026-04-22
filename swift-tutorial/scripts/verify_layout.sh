#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

require_file() {
  local rel="$1"
  [[ -f "$ROOT_DIR/$rel" ]] || fail "missing-file: $rel"
}

require_dir() {
  local rel="$1"
  [[ -d "$ROOT_DIR/$rel" ]] || fail "missing-dir: $rel"
}

require_heading() {
  local rel="$1"
  local pattern="$2"
  rg -q "$pattern" "$ROOT_DIR/$rel" || fail "missing-heading: $rel :: $pattern"
}

require_dir "swift-tutorial/part1-app-first-foundations"
require_dir "swift-tutorial/part2-feature-growth-and-ui-organization"
require_dir "swift-tutorial/part3-data-modeling-persistence-and-shared-core"
require_dir "swift-tutorial/part4-engineering-testing-and-modularization"
require_dir "swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish"
require_dir "swift-tutorial/part6-capstone-and-shipping-readiness"
require_dir "swift-tutorial/projects/focuslist"
require_dir "swift-tutorial/labs"
require_dir "swift-tutorial/appendix"
require_dir "swift-tutorial/scripts"

require_file "swift-tutorial/README.md"
require_file "swift-tutorial/00-preface.md"
require_file "swift-tutorial/projects/README.md"
require_file "swift-tutorial/labs/README.md"
require_file "swift-tutorial/appendix/environment-setup.md"
require_file "swift-tutorial/appendix/glossary.md"
require_file "swift-tutorial/appendix/swiftui-cheatsheet.md"
require_file "swift-tutorial/appendix/swift-testing-cheatsheet.md"
require_file "swift-tutorial/appendix/faq.md"
require_file "swift-tutorial/appendix/answers.md"

require_heading "swift-tutorial/README.md" '^# FocusList：从零到高阶的现代 Swift 教程$'
require_heading "swift-tutorial/00-preface.md" '^# 前言：如何使用这套 FocusList Swift 教程$'
require_heading "swift-tutorial/projects/README.md" '^# Projects：FocusList 项目主线$'
require_heading "swift-tutorial/labs/README.md" '^# Labs：分部综合实验$'

printf 'layout-ok\n'
