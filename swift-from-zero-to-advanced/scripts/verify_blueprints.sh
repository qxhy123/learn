#!/usr/bin/env bash
set -euo pipefail

check_file() {
  local path="$1"
  [[ -f "$path" ]] || {
    echo "missing:$path"
    exit 1
  }
}

check_heading() {
  local pattern="$1"
  local path="$2"
  rg -q "$pattern" "$path" || {
    echo "missing-heading:$path:$pattern"
    exit 1
  }
}

check_topic() {
  local pattern="$1"
  local path="$2"
  rg -q "$pattern" "$path" || {
    echo "missing-topic:$path:$pattern"
    exit 1
  }
}

required_files=(
  "swift-from-zero-to-advanced/parts/part-2-swift-core-engineering/README.md"
  "swift-from-zero-to-advanced/parts/part-3-apple-development-track/README.md"
  "swift-from-zero-to-advanced/parts/part-4-advanced-swift-track/README.md"
)

required_headings=(
  "^## Part Goal$"
  "^## Learning Outcomes$"
  "^## Chapter Sequence$"
  "^## Project Evolution$"
  "^## Drill and Checkpoint Model$"
  "^## Dependencies and Handoffs$"
)

for path in "${required_files[@]}"; do
  check_file "$path"
  for heading in "${required_headings[@]}"; do
    check_heading "$heading" "$path"
  done
done

check_topic "protocols" \
  "swift-from-zero-to-advanced/parts/part-2-swift-core-engineering/README.md"
check_topic "SwiftUI" \
  "swift-from-zero-to-advanced/parts/part-3-apple-development-track/README.md"
check_topic "ARC" \
  "swift-from-zero-to-advanced/parts/part-4-advanced-swift-track/README.md"

echo "blueprints-ok"
