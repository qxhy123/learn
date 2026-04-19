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

require_heading() {
  local rel="$1"
  local pattern="$2"
  grep -qE "$pattern" "$ROOT_DIR/$rel" || fail "missing-heading: $rel :: $pattern"
}

chapters=(
  "swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/01-what-swiftui-app-code-is-actually-expressing.md|^# 第1章：SwiftUI App 到底在写什么$"
  "swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/02-view-composition-and-the-three-core-layout-stacks.md|^# 第2章：View Composition 与三大基础布局$"
  "swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/03-fundamental-interactive-components.md|^# 第3章：最基本的可交互组件$"
  "swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/04-state-driven-ui-fundamentals.md|^# 第4章：状态驱动界面的第一原则$"
  "swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/05-build-boardflow-home-shell.md|^# 第5章：做出 BoardFlow 的最小工作台首页$"
  "swiftui-system-tutorial/part2-components-navigation-and-state-ownership/06-lists-forms-and-input-contracts.md|^# 第6章：列表、表单与输入契约$"
  "swiftui-system-tutorial/part2-components-navigation-and-state-ownership/07-navigationstack-and-navigationsplitview.md|^# 第7章：NavigationStack 与 NavigationSplitView$"
  "swiftui-system-tutorial/part2-components-navigation-and-state-ownership/08-binding-and-state-ownership.md|^# 第8章：Binding 与状态拥有关系$"
  "swiftui-system-tutorial/part2-components-navigation-and-state-ownership/09-observable-models-and-screen-state-coordination.md|^# 第9章：Observable Model 与屏幕级状态协调$"
  "swiftui-system-tutorial/part2-components-navigation-and-state-ownership/10-build-boardflow-v1-desktop-skeleton.md|^# 第10章：做出 BoardFlow v1 桌面应用骨架$"
)

for item in "${chapters[@]}"; do
  IFS='|' read -r rel pattern <<<"$item"
  require_file "$rel"
  require_heading "$rel" "$pattern"
done

grep -q 'part1-swiftui-language-and-basic-view-system' "$ROOT_DIR/swiftui-system-tutorial/README.md" || fail "missing-readme-link: part1"
grep -q 'part2-components-navigation-and-state-ownership' "$ROOT_DIR/swiftui-system-tutorial/README.md" || fail "missing-readme-link: part2"

printf 'parts-ok\n'
