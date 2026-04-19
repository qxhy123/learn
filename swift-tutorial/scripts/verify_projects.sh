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

require_grep() {
  local rel="$1"
  local pattern="$2"
  grep -qE "$pattern" "$ROOT_DIR/$rel" || fail "missing-pattern: $rel :: $pattern"
}

project_dirs=(
  "swift-tutorial/projects/task-cli-lite"
  "swift-tutorial/projects/taskcore-taskcli"
  "swift-tutorial/projects/taskflow"
)

for dir in "${project_dirs[@]}"; do
  require_dir "$dir"
done

project_files=(
  "swift-tutorial/projects/task-cli-lite/README.md"
  "swift-tutorial/projects/task-cli-lite/starter/Package.swift"
  "swift-tutorial/projects/task-cli-lite/starter/Sources/TaskCLILite/main.swift"
  "swift-tutorial/projects/task-cli-lite/starter/Tests/TaskCLILiteTests/TaskCLILiteTests.swift"
  "swift-tutorial/projects/task-cli-lite/milestones/part1-v1.md"
  "swift-tutorial/projects/task-cli-lite/final/README.md"
  "swift-tutorial/projects/taskcore-taskcli/README.md"
  "swift-tutorial/projects/taskcore-taskcli/starter/Package.swift"
  "swift-tutorial/projects/taskcore-taskcli/starter/Sources/TaskCore/Task.swift"
  "swift-tutorial/projects/taskcore-taskcli/starter/Sources/TaskCore/TaskStore.swift"
  "swift-tutorial/projects/taskcore-taskcli/starter/Sources/TaskCLI/main.swift"
  "swift-tutorial/projects/taskcore-taskcli/starter/Tests/TaskCoreTests/TaskCoreTests.swift"
  "swift-tutorial/projects/taskcore-taskcli/milestones/part3-v1.md"
  "swift-tutorial/projects/taskcore-taskcli/milestones/part4-runtime-upgrade.md"
  "swift-tutorial/projects/taskcore-taskcli/final/README.md"
  "swift-tutorial/projects/taskflow/README.md"
  "swift-tutorial/projects/taskflow/starter/README.md"
  "swift-tutorial/projects/taskflow/milestones/part5-v1.md"
  "swift-tutorial/projects/taskflow/milestones/part6-architecture.md"
  "swift-tutorial/projects/taskflow/final/README.md"
)

for file in "${project_files[@]}"; do
  require_file "$file"
done

require_grep "swift-tutorial/projects/task-cli-lite/README.md" 'TaskCLI Lite'
require_grep "swift-tutorial/projects/taskcore-taskcli/README.md" 'TaskCore \+ TaskCLI'
require_grep "swift-tutorial/projects/taskflow/README.md" 'TaskFlow'

printf 'projects-ok\n'
