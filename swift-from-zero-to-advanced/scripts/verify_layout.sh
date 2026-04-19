#!/usr/bin/env bash
set -euo pipefail

root="swift-from-zero-to-advanced"
required_files=(
  "$root/README.md"
  "$root/projects/README.md"
  "$root/parts/part-2-swift-core-engineering/README.md"
  "$root/parts/part-3-apple-development-track/README.md"
  "$root/parts/part-4-advanced-swift-track/README.md"
)

for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || {
    echo "missing:$path"
    exit 1
  }
done

echo "layout-ok"
