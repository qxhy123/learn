#!/usr/bin/env bash
set -euo pipefail

cd swift-tutorial/projects/task-cli-lite/starter
swift build
swift test
printf 'task-cli-lite-ok\n'
