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

for file in \
  swift-tutorial/projects/focuslist/README.md \
  swift-tutorial/projects/focuslist/starter/README.md \
  swift-tutorial/projects/focuslist/starter/Package.swift \
  swift-tutorial/projects/focuslist/starter/Sources/FocusCore/FocusTask.swift \
  swift-tutorial/projects/focuslist/starter/Sources/FocusCore/FocusProject.swift \
  swift-tutorial/projects/focuslist/starter/Sources/FocusCore/FocusStore.swift \
  swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/FocusListApp.swift \
  swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Root/FocusListRootView.swift \
  swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Features/Inbox/InboxView.swift \
  swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Features/Projects/ProjectsView.swift \
  swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Features/Settings/SettingsView.swift \
  swift-tutorial/projects/focuslist/starter/Sources/focusctl/main.swift \
  swift-tutorial/projects/focuslist/starter/Tests/FocusCoreTests/FocusCoreTests.swift \
  swift-tutorial/projects/focuslist/checkpoints/README.md \
  swift-tutorial/projects/focuslist/checkpoints/part1-focuslist-v1/README.md \
  swift-tutorial/projects/focuslist/checkpoints/part2-product-shape/README.md \
  swift-tutorial/projects/focuslist/checkpoints/part3-focuscore-split/README.md \
  swift-tutorial/projects/focuslist/checkpoints/part4-engineering-v1/README.md \
  swift-tutorial/projects/focuslist/checkpoints/part5-polish/README.md \
  swift-tutorial/projects/focuslist/final/README.md
do
  require_file "$file"
done

rg -q 'FocusList' "$ROOT_DIR/swift-tutorial/projects/focuslist/README.md" || fail "missing-focuslist-string"
rg -q 'FocusCore' "$ROOT_DIR/swift-tutorial/projects/focuslist/README.md" || fail "missing-focuscore-string"
rg -q 'focusctl' "$ROOT_DIR/swift-tutorial/projects/focuslist/README.md" || fail "missing-focusctl-string"

printf 'projects-ok\n'
