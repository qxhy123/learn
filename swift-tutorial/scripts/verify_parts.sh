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

require_grep() {
  local rel="$1"
  local pattern="$2"
  grep -qE "$pattern" "$ROOT_DIR/$rel" || fail "missing-pattern: $rel :: $pattern"
}

chapters=(
  "swift-tutorial/part1-language-foundations/01-toolchain-and-first-swift-program.md"
  "swift-tutorial/part1-language-foundations/02-values-types-and-mutability.md"
  "swift-tutorial/part1-language-foundations/03-strings-collections-and-control-flow.md"
  "swift-tutorial/part1-language-foundations/04-functions-optionals-enums-and-structs.md"
  "swift-tutorial/part1-language-foundations/05-build-taskcli-lite-v1.md"
  "swift-tutorial/part2-type-system-and-modeling/06-methods-properties-and-initializers.md"
  "swift-tutorial/part2-type-system-and-modeling/07-classes-vs-structs-and-value-vs-reference.md"
  "swift-tutorial/part2-type-system-and-modeling/08-protocols-protocol-extensions-and-abstraction-boundaries.md"
  "swift-tutorial/part2-type-system-and-modeling/09-generics-associated-types-and-type-driven-api-design.md"
  "swift-tutorial/part2-type-system-and-modeling/10-errors-results-and-modeling-failure.md"
  "swift-tutorial/part3-packages-testing-and-cli-engineering/11-swift-package-manager-and-module-boundaries.md"
  "swift-tutorial/part3-packages-testing-and-cli-engineering/12-testing-with-xctest-and-core-behavior.md"
  "swift-tutorial/part3-packages-testing-and-cli-engineering/13-parsing-rendering-and-storage-seams.md"
  "swift-tutorial/part3-packages-testing-and-cli-engineering/14-command-organization-and-cli-architecture.md"
  "swift-tutorial/part3-packages-testing-and-cli-engineering/15-build-taskcore-taskcli-v1.md"
  "swift-tutorial/part4-concurrency-performance-and-reliability/16-async-await-and-task-basics.md"
  "swift-tutorial/part4-concurrency-performance-and-reliability/17-actors-isolation-and-sendability.md"
  "swift-tutorial/part4-concurrency-performance-and-reliability/18-arc-memory-and-ownership-in-practice.md"
  "swift-tutorial/part4-concurrency-performance-and-reliability/19-performance-copying-and-measurement-mindset.md"
  "swift-tutorial/part4-concurrency-performance-and-reliability/20-reliability-cancellation-and-failure-surfaces.md"
  "swift-tutorial/part5-swiftui-foundations/21-swiftui-mental-model-and-view-composition.md"
  "swift-tutorial/part5-swiftui-foundations/22-state-binding-and-observable-models.md"
  "swift-tutorial/part5-swiftui-foundations/23-lists-forms-and-navigation-basics.md"
  "swift-tutorial/part5-swiftui-foundations/24-build-taskflow-v1.md"
  "swift-tutorial/part6-swiftui-dataflow-and-app-architecture/25-app-state-and-data-flow.md"
  "swift-tutorial/part6-swiftui-dataflow-and-app-architecture/26-persistence-and-model-integration.md"
  "swift-tutorial/part6-swiftui-dataflow-and-app-architecture/27-async-ui-updates-previews-and-testing.md"
  "swift-tutorial/part6-swiftui-dataflow-and-app-architecture/28-taskflow-architecture-and-feature-growth.md"
  "swift-tutorial/part7-advanced-swift-and-system-design/29-advanced-generics-and-protocol-design.md"
  "swift-tutorial/part7-advanced-swift-and-system-design/30-result-builders-macros-and-api-surface-judgment.md"
  "swift-tutorial/part7-advanced-swift-and-system-design/31-interop-system-apis-and-package-boundary-tradeoffs.md"
  "swift-tutorial/part7-advanced-swift-and-system-design/32-shared-abstractions-and-system-redesign.md"
  "swift-tutorial/part8-capstone-and-next-steps/33-capstone-rebuild-plan.md"
  "swift-tutorial/part8-capstone-and-next-steps/34-capstone-cli-and-core-hardening.md"
  "swift-tutorial/part8-capstone-and-next-steps/35-capstone-taskflow-hardening.md"
  "swift-tutorial/part8-capstone-and-next-steps/36-graduation-roadmap-and-next-steps.md"
)

for chapter in "${chapters[@]}"; do
  require_file "$chapter"
  first_line="$(head -n 1 "$ROOT_DIR/$chapter" || true)"
  [[ "$first_line" =~ ^#\ 第 ]] || fail "invalid-heading-start: $chapter"
done

part_markers=(
  "part1-language-foundations"
  "part2-type-system-and-modeling"
  "part3-packages-testing-and-cli-engineering"
  "part4-concurrency-performance-and-reliability"
  "part5-swiftui-foundations"
  "part6-swiftui-dataflow-and-app-architecture"
  "part7-advanced-swift-and-system-design"
  "part8-capstone-and-next-steps"
)

for marker in "${part_markers[@]}"; do
  require_grep "swift-tutorial/README.md" "$marker"
done

printf 'parts-ok\n'
