#!/usr/bin/env bash
set -euo pipefail

project_root="swift-from-zero-to-advanced/projects/task-cli-lite"
starter_root="$project_root/starter"

required_files=(
  "$project_root/README.md"
  "$starter_root/Package.swift"
  "$starter_root/Sources/TaskCLILite/main.swift"
  "$starter_root/Tests/TaskCLILiteTests/TaskCLILiteTests.swift"
)

for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || {
    echo "missing:$path"
    exit 1
  }
done

rg -q "TaskCLI Lite" "$project_root/README.md" || {
  echo "missing-readme-title"
  exit 1
}

if command -v swift >/dev/null 2>&1; then
  (
    cd "$starter_root"
    swift test >/dev/null
  )
fi

echo "task-cli-lite-ok"
