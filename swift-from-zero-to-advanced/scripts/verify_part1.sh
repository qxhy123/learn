#!/usr/bin/env bash
set -euo pipefail

required_files=(
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/overview.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-running-swift.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-values-and-types.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-strings-and-program-io.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-control-flow-for-commands.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-functions-and-program-shape.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-structs-and-data-modeling.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-collections-and-task-state.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-optionals-and-safe-parsing.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/09-enums-and-pattern-matching.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/10-build-task-cli-lite-v1.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/drills/README.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/checkpoint/part-1-checkpoint.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/project/part-1-task-cli-lite.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/references/part-1-glossary.md"
)

chapter_files=(
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-running-swift.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-values-and-types.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-strings-and-program-io.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-control-flow-for-commands.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-functions-and-program-shape.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-structs-and-data-modeling.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-collections-and-task-state.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-optionals-and-safe-parsing.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/09-enums-and-pattern-matching.md"
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/10-build-task-cli-lite-v1.md"
)

required_sections=(
  "^## Problem$"
  "^## Running Example$"
  "^## Semantic Deep Dive$"
  "^## Code Evolution$"
  "^## Common Mistakes$"
  "^## Drills$"
  "^## Checkpoint$"
  "^## Glossary$"
  "^## English Recap$"
  "^## Project Bridge$"
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

rg -q "^# Chapter 01: Running Swift$" \
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-running-swift.md" || {
  echo "missing-heading:chapter-01"
  exit 1
}

rg -q "^# Chapter 10: Build TaskCLI Lite v1$" \
  "swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/10-build-task-cli-lite-v1.md" || {
  echo "missing-heading:chapter-10"
  exit 1
}

for chapter in "${chapter_files[@]}"; do
  for heading in "${required_sections[@]}"; do
    rg -q "$heading" "$chapter" || {
      echo "missing-section:$chapter:$heading"
      exit 1
    }
  done
done

echo "part1-ok"
