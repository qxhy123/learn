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
  grep -qE "$pattern" "$ROOT_DIR/$rel" || fail "missing-heading: $rel :: $pattern"
}

require_file "swift-tutorial/README.md"
require_file "swift-tutorial/00-preface.md"

part_dirs=(
  "swift-tutorial/part1-language-foundations"
  "swift-tutorial/part2-type-system-and-modeling"
  "swift-tutorial/part3-packages-testing-and-cli-engineering"
  "swift-tutorial/part4-concurrency-performance-and-reliability"
  "swift-tutorial/part5-swiftui-foundations"
  "swift-tutorial/part6-swiftui-dataflow-and-app-architecture"
  "swift-tutorial/part7-advanced-swift-and-system-design"
  "swift-tutorial/part8-capstone-and-next-steps"
)

for dir in "${part_dirs[@]}"; do
  require_dir "$dir"
done

require_dir "swift-tutorial/projects"
require_dir "swift-tutorial/labs"
require_dir "swift-tutorial/appendix"

require_heading "swift-tutorial/README.md" '^# 从零到高阶的 Swift 教程$'
require_heading "swift-tutorial/00-preface.md" '^# 前言：如何使用本教程$'

printf 'layout-ok\n'
