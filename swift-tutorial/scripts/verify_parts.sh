#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

for file in \
  swift-tutorial/part1-app-first-foundations/*.md \
  swift-tutorial/part2-feature-growth-and-ui-organization/*.md \
  swift-tutorial/part3-data-modeling-persistence-and-shared-core/*.md \
  swift-tutorial/part4-engineering-testing-and-modularization/*.md \
  swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish/*.md \
  swift-tutorial/part6-capstone-and-shipping-readiness/*.md
do
  [[ -f "$ROOT_DIR/$file" ]] || fail "missing-file: $file"
  head -n 1 "$ROOT_DIR/$file" | rg -q '^# 第 ' || fail "bad-heading: $file"
done

for label in "Part 1" "Part 2" "Part 3" "Part 4" "Part 5" "Part 6"; do
  rg -q "$label" "$ROOT_DIR/swift-tutorial/README.md" || fail "missing-part-map: $label"
done

printf 'parts-ok\n'
