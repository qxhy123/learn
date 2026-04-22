#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR/swift-tutorial/projects/focuslist/starter"
swift test
swift build --product FocusListApp
swift build --product focusctl

printf 'focuslist-starter-ok\n'
