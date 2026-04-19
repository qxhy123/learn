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

require_grep() {
  local rel="$1"
  local pattern="$2"
  grep -qE "$pattern" "$ROOT_DIR/$rel" || fail "missing-pattern: $rel :: $pattern"
}

appendix_files=(
  "swift-tutorial/appendix/glossary.md"
  "swift-tutorial/appendix/answers.md"
  "swift-tutorial/appendix/environment-setup.md"
  "swift-tutorial/appendix/spm-cheatsheet.md"
  "swift-tutorial/appendix/swiftui-cheatsheet.md"
  "swift-tutorial/appendix/faq.md"
  "swift-tutorial/appendix/references.md"
)

labs_files=(
  "swift-tutorial/labs/part1-language-foundations.md"
  "swift-tutorial/labs/part2-type-system-and-modeling.md"
  "swift-tutorial/labs/part3-packages-testing-and-cli-engineering.md"
  "swift-tutorial/labs/part4-concurrency-performance-and-reliability.md"
  "swift-tutorial/labs/part5-swiftui-foundations.md"
  "swift-tutorial/labs/part6-swiftui-dataflow-and-app-architecture.md"
  "swift-tutorial/labs/part7-advanced-swift-and-system-design.md"
  "swift-tutorial/labs/part8-capstone.md"
)

for file in "${appendix_files[@]}" "${labs_files[@]}"; do
  require_file "$file"
done

require_grep "swift-tutorial/appendix/glossary.md" '^# 术语表$'
require_grep "swift-tutorial/appendix/answers.md" '^# 练习与综合实验答案$'
require_grep "swift-tutorial/appendix/environment-setup.md" '^# 环境准备$'
require_grep "swift-tutorial/appendix/spm-cheatsheet.md" '^# Swift Package Manager 速查$'
require_grep "swift-tutorial/appendix/swiftui-cheatsheet.md" '^# SwiftUI 速查$'
require_grep "swift-tutorial/appendix/faq.md" '^# 常见问题$'
require_grep "swift-tutorial/appendix/references.md" '^# 参考资料$'

printf 'appendix-ok\n'
