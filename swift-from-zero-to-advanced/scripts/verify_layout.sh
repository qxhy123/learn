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

rg -q "^# Swift From Zero to Advanced$" "$root/README.md" || {
  echo "missing-heading:$root/README.md"
  exit 1
}

rg -q "^# Projects$" "$root/projects/README.md" || {
  echo "missing-heading:$root/projects/README.md"
  exit 1
}

echo "layout-ok"
