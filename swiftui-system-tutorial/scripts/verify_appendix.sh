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

files=(
  "swiftui-system-tutorial/appendix/component-atlas.md|^# 组件图谱：基础视图、输入、容器与反馈$"
  "swiftui-system-tutorial/appendix/layout-playbook.md|^# 布局手册：Stack、Split、Overlay 与空间组织$"
  "swiftui-system-tutorial/appendix/state-ownership-guide.md|^# 状态所有权指南：State、Binding、Observable 与 Environment$"
  "swiftui-system-tutorial/appendix/navigation-and-workbench-patterns.md|^# 导航与工作台模式：NavigationStack、NavigationSplitView 与多面板组织$"
  "swiftui-system-tutorial/appendix/gesture-playbook.md|^# 手势手册：点击、拖拽、缩放与冲突处理$"
  "swiftui-system-tutorial/appendix/canvas-and-drawing-guide.md|^# 绘制手册：Shape、Path 与 Canvas$"
  "swiftui-system-tutorial/appendix/animation-guide.md|^# 动画手册：过渡、交易与空间过渡$"
  "swiftui-system-tutorial/appendix/mac-interop-guide.md|^# Mac 互操作手册：AppKit 边界与桥接策略$"
  "swiftui-system-tutorial/appendix/performance-and-identity-guide.md|^# 性能与身份手册：Diffing、刷新与大画布判断$"
  "swiftui-system-tutorial/appendix/glossary.md|^# 术语表：SwiftUI 与 BoardFlow 核心概念$"
  "swiftui-system-tutorial/appendix/faq.md|^# FAQ：学习路径、工程选择与常见误区$"
  "swiftui-system-tutorial/appendix/references.md|^# 参考资料：Apple 文档与延伸阅读$"
  "swiftui-system-tutorial/labs/README.md|^# Labs：把 Part 1 和 Part 2 变成手上的代码$"
  "swiftui-system-tutorial/labs/part1.md|^# Lab 1：把 BoardFlow 首页从静态界面写出来$"
  "swiftui-system-tutorial/labs/part2.md|^# Lab 2：把 BoardFlow 变成带 Sidebar 的桌面骨架$"
)

for item in "${files[@]}"; do
  IFS='|' read -r rel pattern <<<"$item"
  require_file "$rel"
  require_heading "$rel" "$pattern"
done

printf 'appendix-ok\n'
