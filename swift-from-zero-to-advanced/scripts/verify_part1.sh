#!/usr/bin/env bash
set -euo pipefail

required_files=(
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/overview.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-swift-setup-and-first-program.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-constants-variables-and-types.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-control-flow.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-functions-and-decomposition.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-collections.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-optionals-and-basic-error-handling.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-strings-tuples-and-pattern-matching.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-part-1-project.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/drills/README.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/checkpoint/part-1-checkpoint.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/project/part-1-task-cli-lite.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/references/part-1-glossary.md"
)

for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || {
    echo "missing:$path"
    exit 1
  }
done

rg -q "^# Part 1: Swift Fundamentals$" \
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/overview.md" || {
  echo "missing-heading:overview"
  exit 1
}

rg -q "^# Chapter 01: Swift Setup and First Program$" \
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-swift-setup-and-first-program.md" || {
  echo "missing-heading:chapter-01"
  exit 1
}

rg -q "^# Part 1 Checkpoint$" \
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/checkpoint/part-1-checkpoint.md" || {
  echo "missing-heading:checkpoint"
  exit 1
}

echo "part1-ok"
