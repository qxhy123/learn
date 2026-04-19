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

require_dir "swiftui-system-tutorial"
require_dir "swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system"
require_dir "swiftui-system-tutorial/part2-components-navigation-and-state-ownership"
require_dir "swiftui-system-tutorial/projects"
require_dir "swiftui-system-tutorial/projects/boardflow"
require_dir "swiftui-system-tutorial/projects/boardflow/checkpoints"
require_dir "swiftui-system-tutorial/projects/boardflow/starter"
require_dir "swiftui-system-tutorial/appendix"
require_dir "swiftui-system-tutorial/labs"
require_dir "swiftui-system-tutorial/scripts"

require_file "swiftui-system-tutorial/README.md"
require_file "swiftui-system-tutorial/00-orientation.md"
require_file "swiftui-system-tutorial/01-learning-map.md"
require_file "swiftui-system-tutorial/projects/README.md"
require_file "swiftui-system-tutorial/projects/boardflow/README.md"
require_file "swiftui-system-tutorial/projects/boardflow/checkpoints/README.md"
require_file "swiftui-system-tutorial/projects/boardflow/checkpoints/part1-shell/README.md"
require_file "swiftui-system-tutorial/projects/boardflow/checkpoints/part2-v1-workbench/README.md"

require_heading "swiftui-system-tutorial/README.md" '^# SwiftUI 系统教程：从零到 Mac 创作工具工程$'
require_heading "swiftui-system-tutorial/00-orientation.md" '^# 导读：如何使用这套 SwiftUI 系统教程$'
require_heading "swiftui-system-tutorial/01-learning-map.md" '^# 学习地图：BoardFlow 主线与 SwiftUI 能力图谱$'

printf 'layout-ok\n'
