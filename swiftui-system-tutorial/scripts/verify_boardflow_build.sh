#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../projects/boardflow/starter"
swift build
swift test
printf 'boardflow-build-ok\n'
