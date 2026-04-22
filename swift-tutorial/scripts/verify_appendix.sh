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
  rg -q "$pattern" "$ROOT_DIR/$rel" || fail "missing-heading: $rel :: $pattern"
}

for file in \
  swift-tutorial/appendix/environment-setup.md \
  swift-tutorial/appendix/glossary.md \
  swift-tutorial/appendix/swiftui-cheatsheet.md \
  swift-tutorial/appendix/swift-testing-cheatsheet.md \
  swift-tutorial/appendix/faq.md \
  swift-tutorial/appendix/answers.md \
  swift-tutorial/labs/README.md \
  swift-tutorial/labs/part1-app-first-foundations.md \
  swift-tutorial/labs/part2-feature-growth-and-ui-organization.md \
  swift-tutorial/labs/part3-data-modeling-persistence-and-shared-core.md \
  swift-tutorial/labs/part4-engineering-testing-and-modularization.md \
  swift-tutorial/labs/part5-concurrency-reliability-and-cross-platform-polish.md \
  swift-tutorial/labs/part6-capstone-and-shipping-readiness.md
do
  require_file "$file"
done

require_heading "swift-tutorial/appendix/environment-setup.md" '^# 环境准备$'
require_heading "swift-tutorial/appendix/glossary.md" '^# 术语表$'
require_heading "swift-tutorial/appendix/swiftui-cheatsheet.md" '^# SwiftUI 速查$'
require_heading "swift-tutorial/appendix/swift-testing-cheatsheet.md" '^# Swift Testing 速查$'
require_heading "swift-tutorial/appendix/faq.md" '^# 常见问题$'
require_heading "swift-tutorial/appendix/answers.md" '^# 练习与综合实验答案$'

printf 'appendix-ok\n'
