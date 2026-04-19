#!/usr/bin/env bash
set -euo pipefail

cd swift-tutorial/projects/taskcore-taskcli/starter
swift build
swift test
printf 'taskcore-taskcli-ok\n'
