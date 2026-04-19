# Swift Tutorial Systematic Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current scaffold-level Swift tutorial with a publication-grade tutorial contract, a fully rewritten long-form Part 1, and detailed blueprints for Parts 2 through 4.

**Architecture:** Keep the tutorial rooted in `swift-from-zero-to-advanced/`, but replace the current shallow content standard with verification-backed long-form documentation. The rewrite happens in six independent documentation lanes: shared course contract, Part 1 structure migration, three Part 1 authoring waves, and later-part blueprint upgrades. Each lane ends with explicit verification and a commit so the tutorial can be reviewed incrementally instead of rewritten in one opaque dump.

**Tech Stack:** Markdown, shell verification scripts, `rg`, `bash`, existing Swift Package starter files

---

## File Structure

### Create

- `docs/superpowers/plans/2026-04-19-swift-tutorial-systematic-rewrite.md`
  - This implementation plan.
- `swift-from-zero-to-advanced/scripts/verify_blueprints.sh`
  - Verification for the richer Part 2, Part 3, and Part 4 blueprint readmes.
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-running-swift.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-values-and-types.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-strings-and-program-io.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-control-flow-for-commands.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-functions-and-program-shape.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-structs-and-data-modeling.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-collections-and-task-state.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-optionals-and-safe-parsing.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/09-enums-and-pattern-matching.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/10-build-task-cli-lite-v1.md`

### Modify

- `swift-from-zero-to-advanced/README.md`
  - Rewrite the course entry page so it describes the stronger tutorial standard and the new Part 1 shape.
- `swift-from-zero-to-advanced/projects/README.md`
  - Clarify the two project spines, how Part 1 differs from later parts, and what the CLI starter is for.
- `swift-from-zero-to-advanced/scripts/verify_layout.sh`
  - Keep the root scaffold check aligned with the course entry files.
- `swift-from-zero-to-advanced/scripts/verify_shared_docs.sh`
  - Tighten the shared-document contract around chapter depth, bilingual rules, and learning paths.
- `swift-from-zero-to-advanced/scripts/verify_part1.sh`
  - Replace the old eight-chapter file map with the new ten-chapter Part 1 file map and required sections.
- `swift-from-zero-to-advanced/references/authoring-rules.md`
- `swift-from-zero-to-advanced/references/bilingual-style-guide.md`
- `swift-from-zero-to-advanced/references/chapter-template.md`
- `swift-from-zero-to-advanced/references/learning-paths.md`
- `swift-from-zero-to-advanced/glossary/core-terms.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/overview.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/drills/README.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/checkpoint/part-1-checkpoint.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/project/part-1-task-cli-lite.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/references/part-1-glossary.md`
- `swift-from-zero-to-advanced/parts/part-2-swift-core-engineering/README.md`
- `swift-from-zero-to-advanced/parts/part-3-apple-development-track/README.md`
- `swift-from-zero-to-advanced/parts/part-4-advanced-swift-track/README.md`

### Delete

- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-swift-setup-and-first-program.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-constants-variables-and-types.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-control-flow.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-functions-and-decomposition.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-collections.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-optionals-and-basic-error-handling.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-strings-tuples-and-pattern-matching.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-part-1-project.md`

### Do Not Modify

- `deepagents/`
- `langgraph/`
- `langchain/`
- `deepagents-internal-tutorial/`
- `deepagents-coding-platform/`
- `docs/superpowers/specs/2026-04-19-swift-from-zero-to-advanced-tutorial-design.md`
- `docs/superpowers/specs/2026-04-19-swift-tutorial-systematic-rewrite-design.md`

### Verification Surface

- `bash swift-from-zero-to-advanced/scripts/verify_layout.sh`
- `bash swift-from-zero-to-advanced/scripts/verify_shared_docs.sh`
- `bash swift-from-zero-to-advanced/scripts/verify_part1.sh`
- `bash swift-from-zero-to-advanced/scripts/verify_blueprints.sh`
- `bash swift-from-zero-to-advanced/scripts/verify_task_cli_lite.sh`
- `git diff --check -- swift-from-zero-to-advanced`

---

### Task 1: Rewrite the Shared Course Contract

**Files:**
- Modify: `swift-from-zero-to-advanced/scripts/verify_layout.sh`
- Modify: `swift-from-zero-to-advanced/scripts/verify_shared_docs.sh`
- Modify: `swift-from-zero-to-advanced/README.md`
- Modify: `swift-from-zero-to-advanced/projects/README.md`
- Modify: `swift-from-zero-to-advanced/references/authoring-rules.md`
- Modify: `swift-from-zero-to-advanced/references/bilingual-style-guide.md`
- Modify: `swift-from-zero-to-advanced/references/chapter-template.md`
- Modify: `swift-from-zero-to-advanced/references/learning-paths.md`
- Modify: `swift-from-zero-to-advanced/glossary/core-terms.md`

- [ ] **Step 1: Tighten the layout and shared-doc verifiers**

```bash
# swift-from-zero-to-advanced/scripts/verify_layout.sh
#!/usr/bin/env bash
set -euo pipefail

root="swift-from-zero-to-advanced"
required_files=(
  "$root/README.md"
  "$root/projects/README.md"
  "$root/parts/part-2-swift-core-engineering/README.md"
  "$root/parts/part-3-apple-development-track/README.md"
  "$root/parts/part-4-advanced-swift-track/README.md"
)

for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || {
    echo "missing:$path"
    exit 1
  }
done

rg -q "^# Swift From Zero to Advanced$" "$root/README.md" || {
  echo "missing-heading:$root/README.md"
  exit 1
}

rg -q "^# Projects$" "$root/projects/README.md" || {
  echo "missing-heading:$root/projects/README.md"
  exit 1
}

echo "layout-ok"
```

```bash
# swift-from-zero-to-advanced/scripts/verify_shared_docs.sh
#!/usr/bin/env bash
set -euo pipefail

check_file() {
  local path="$1"
  [[ -f "$path" ]] || {
    echo "missing:$path"
    exit 1
  }
}

check_heading() {
  local pattern="$1"
  local path="$2"
  rg -q "$pattern" "$path" || {
    echo "missing-heading:$path:$pattern"
    exit 1
  }
}

check_file "swift-from-zero-to-advanced/references/authoring-rules.md"
check_file "swift-from-zero-to-advanced/references/bilingual-style-guide.md"
check_file "swift-from-zero-to-advanced/references/chapter-template.md"
check_file "swift-from-zero-to-advanced/references/learning-paths.md"
check_file "swift-from-zero-to-advanced/glossary/core-terms.md"

check_heading "^# Authoring Rules$" "swift-from-zero-to-advanced/references/authoring-rules.md"
check_heading "^## Chapter Quality Bar$" "swift-from-zero-to-advanced/references/authoring-rules.md"
check_heading "^## Project Spine Rules$" "swift-from-zero-to-advanced/references/authoring-rules.md"
check_heading "^## Drill and Checkpoint Contract$" "swift-from-zero-to-advanced/references/authoring-rules.md"

check_heading "^# Bilingual Style Guide$" "swift-from-zero-to-advanced/references/bilingual-style-guide.md"
check_heading "^## First-Use Term Rule$" "swift-from-zero-to-advanced/references/bilingual-style-guide.md"
check_heading "^## English Recap$" "swift-from-zero-to-advanced/references/bilingual-style-guide.md"
check_heading "^## Non-Rules$" "swift-from-zero-to-advanced/references/bilingual-style-guide.md"

check_heading "^# Chapter Template$" "swift-from-zero-to-advanced/references/chapter-template.md"
check_heading "^## Problem$" "swift-from-zero-to-advanced/references/chapter-template.md"
check_heading "^## Running Example$" "swift-from-zero-to-advanced/references/chapter-template.md"
check_heading "^## Semantic Deep Dive$" "swift-from-zero-to-advanced/references/chapter-template.md"
check_heading "^## Code Evolution$" "swift-from-zero-to-advanced/references/chapter-template.md"
check_heading "^## English Recap$" "swift-from-zero-to-advanced/references/chapter-template.md"
check_heading "^## Project Bridge$" "swift-from-zero-to-advanced/references/chapter-template.md"

check_heading "^# Learning Paths$" "swift-from-zero-to-advanced/references/learning-paths.md"
check_heading "^## Default Path$" "swift-from-zero-to-advanced/references/learning-paths.md"
check_heading "^## Language-First Path$" "swift-from-zero-to-advanced/references/learning-paths.md"
check_heading "^## App-First Path$" "swift-from-zero-to-advanced/references/learning-paths.md"
check_heading "^## CLI / Engineering-First Path$" "swift-from-zero-to-advanced/references/learning-paths.md"

check_heading "^# Core Terms Glossary$" "swift-from-zero-to-advanced/glossary/core-terms.md"
check_heading "Value semantics" "swift-from-zero-to-advanced/glossary/core-terms.md"
check_heading "Optional binding" "swift-from-zero-to-advanced/glossary/core-terms.md"
check_heading "Pattern matching" "swift-from-zero-to-advanced/glossary/core-terms.md"

echo "shared-docs-ok"
```

- [ ] **Step 2: Run the verifiers and confirm the current shared docs are too weak**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-from-zero-to-advanced/scripts/verify_layout.sh
bash swift-from-zero-to-advanced/scripts/verify_shared_docs.sh
```

Expected:

```text
layout-ok
missing-heading:swift-from-zero-to-advanced/references/authoring-rules.md:^## Chapter Quality Bar$
```

- [ ] **Step 3: Rewrite the course entry docs and shared authoring rules**

```markdown
# swift-from-zero-to-advanced/README.md
# Swift From Zero to Advanced

## What This Project Is

This repository is a four-part Swift curriculum for readers who already know
another programming language and want to learn Swift systematically.

This is not a note dump and not a reference manual. It is a guided course with
an explicit writing standard, continuous project spines, bilingual terminology,
drills, checkpoints, and part-level outcomes.

## Reader Profile

The default reader:

- can already program in another language
- is new to Swift
- wants both language depth and engineering depth
- is willing to learn through reading, drills, and project work

## Course Shape

The course has four parts:

1. Part 1: Swift Fundamentals
2. Part 2: Swift Core Engineering
3. Part 3: Apple Development Track
4. Part 4: Advanced Swift Track

Part 1 is written as full long-form tutorial prose.
Parts 2 through 4 are maintained as detailed blueprints until their full prose
phases begin.

## Project Spines

The same task-management domain runs through the whole course.

- CLI / Package spine: `TaskCLI Lite` -> `TaskCore + TaskCLI` -> `TaskCLI Pro`
- Apple app spine: `TaskFlow` plus later advanced enhancements

This keeps the domain stable while the Swift depth grows.

## Part 1 Rewrite Standard

Part 1 is not a chapter outline. Each chapter must:

- start with a concrete problem
- show complete runnable code
- explain Swift semantics in depth
- evolve code from a naive version to a better version
- explain common mistakes and what they look like
- end with drills, a checkpoint, glossary terms, and an English recap

## How To Read This Course

1. Start with the Part 1 overview.
2. Read chapters in order unless a specific learning path tells you otherwise.
3. Run and edit the code while reading.
4. Do the drills before skimming the checkpoint.
5. Treat the part project as required integration work.

## Learning Paths

- Default path: complete Part 1 through Part 4 in order
- Language-first path: focus on the language and engineering spine first
- App-first path: finish Part 1 and Part 2, then move into the Apple track
- CLI / engineering-first path: lean harder into the package and tooling spine
```

```markdown
# swift-from-zero-to-advanced/projects/README.md
# Projects

This directory contains the long-running project spines used by the course.

## CLI / Package Spine

- Part 1: `TaskCLI Lite`
- Part 2: `TaskCore + TaskCLI`
- Part 4: `TaskCLI Pro`

The CLI line teaches Swift as a language first, then as a modular engineering
toolchain.

## Apple App Spine

- Part 3: `TaskFlow`
- Part 4: advanced `TaskFlow` hardening

The app line exists only after the reader has enough shared Swift foundation to
focus on UI, state, data flow, and persistence without relearning the domain.

## Why The Domain Stays The Same

The course keeps one task-management domain on purpose.

Readers should spend their energy on Swift semantics, code shape, and design
tradeoffs, not on re-learning a new business problem every part.

## Part 1 Constraint

Part 1 keeps the starter package intentionally small.
It should look real, but it should not import the architecture that belongs to
Part 2.
```

```markdown
# swift-from-zero-to-advanced/references/authoring-rules.md
# Authoring Rules

## Chapter Quality Bar

A chapter is only considered written if it:

- teaches one clear problem
- contains complete runnable code
- explains the Swift semantics behind the code
- shows at least one code-evolution step
- explains common mistakes through cause and effect
- ends with drills, a checkpoint, glossary terms, an English recap, and a
  project bridge

Short outline-like chapter cards are not acceptable final content.

## Long-Form Rhythm

The default chapter rhythm is:

1. problem framing
2. running example
3. semantic deep dive
4. code evolution
5. common mistakes
6. drills
7. checkpoint
8. glossary
9. English recap
10. project bridge

## Project Spine Rules

- every chapter must move the active project or mental model forward
- project work should evolve incrementally instead of restarting each chapter
- Part 1 should stay small and readable
- Part 2 is where stronger architecture and test boundaries begin

## Drill and Checkpoint Contract

Every full chapter should contain three drill types:

- concept check
- code reading
- hands-on extension

Checkpoints are different from drills.
Drills practice the current chapter.
Checkpoints ask the reader to explain or extend the whole cluster of ideas.

## Optional Advanced Material

When optional advanced material appears:

- label it explicitly
- explain why it is optional
- do not let it interrupt the main path
```

```markdown
# swift-from-zero-to-advanced/references/bilingual-style-guide.md
# Bilingual Style Guide

## Core Model

This course is strongly bilingual, but it is not written as full paragraph-by-
paragraph translation.

Chinese carries:

- explanations
- reasoning
- pitfalls
- design tradeoffs
- exercise guidance

English carries:

- code
- API names
- type names
- canonical technical terms

## First-Use Term Rule

Introduce key concepts in bilingual first-use form:

- `Value semantics（值语义）`
- `Optional binding（可选值绑定）`
- `Pattern matching（模式匹配）`

After first use, whichever side is more natural in context may lead.

## English Recap

Every full Part 1 chapter should end with a short `English Recap` section.
It is not a translation of the whole chapter.
It is a compact technical summary of the chapter's rules, vocabulary, and
engineering takeaways.

## Term Stability

- use one English term for one concept
- use one Chinese translation for one concept unless there is a strong reason
  to change it
- update the glossary before inventing a new translation locally

## Non-Rules

The course should not:

- transliterate Chinese into romanization
- duplicate the full body text in two languages
- translate code, API names, or symbol names into Chinese
- interrupt the teaching flow with glossary spam in every paragraph
```

```markdown
# swift-from-zero-to-advanced/references/chapter-template.md
# Chapter Template

## Problem

State the concrete programming or design problem the chapter solves.

## Running Example

Show a minimal but complete Swift example the reader can run.

## Semantic Deep Dive

Explain what the code means in Swift, especially where Swift differs from other
languages.

## Code Evolution

Refactor the first version into a better version and explain the tradeoff.

## Common Mistakes

Explain the error pattern, why it happens, and what the reader would observe.

## Drills

Include:

- one concept check
- one code-reading task
- one hands-on extension

## Checkpoint

State what the reader should now be able to explain or change.

## Glossary

List the chapter-specific bilingual vocabulary that was introduced.

## English Recap

Summarize the technical takeaways in short English prose or bullets.

## Project Bridge

Say exactly what this chapter adds to the active project.
```

```markdown
# swift-from-zero-to-advanced/references/learning-paths.md
# Learning Paths

## Default Path

The default path is full sequential completion:

1. Part 1
2. Part 2
3. Part 3
4. Part 4

This is the recommended path for most readers.

## Language-First Path

Use this if you care most about Swift the language and engineering design:

- complete all of Part 1
- complete all of Part 2
- then choose whether Part 3 or Part 4 matters more

## App-First Path

Use this if your end goal is Apple-platform work:

- complete Part 1
- complete Part 2
- move into Part 3
- return to Part 4 after the app track feels stable

## CLI / Engineering-First Path

Use this if you care most about packages, tools, and program design:

- complete Part 1
- complete Part 2
- keep following the CLI spine into Part 4
- treat Part 3 as a specialization track rather than the core route
```

```markdown
# swift-from-zero-to-advanced/glossary/core-terms.md
# Core Terms Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Value semantics | 值语义 | Changing one value does not mutate a separate copy. |
| Reference semantics | 引用语义 | Multiple names can point at the same mutable object. |
| Type inference | 类型推断 | Swift infers a type from the expression and context. |
| Optional | 可选值 | A value that may contain data or `nil`. |
| Optional binding | 可选值绑定 | A safe way to unwrap an optional. |
| Pattern matching | 模式匹配 | Matching values against structured cases. |
| Struct | 结构体 | A value type used to model grouped data. |
| Enum | 枚举 | A type that represents one of several named cases. |
| Protocol | 协议 | A contract that types can adopt. |
| Conformance | 遵循 | A type satisfying a protocol's requirements. |
| Swift Package | Swift 包 | A package managed by SwiftPM. |
| Target | 目标 | A buildable unit inside a Swift package. |
| Binding | 绑定 | A two-way connection to mutable state in SwiftUI. |
| Actor | Actor | A concurrency type that protects mutable state. |
```

- [ ] **Step 4: Run the shared-doc verification again**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-from-zero-to-advanced/scripts/verify_layout.sh
bash swift-from-zero-to-advanced/scripts/verify_shared_docs.sh
git diff --check -- swift-from-zero-to-advanced
```

Expected:

```text
layout-ok
shared-docs-ok
```

- [ ] **Step 5: Commit the rewritten course contract**

Run:

```bash
git add \
  swift-from-zero-to-advanced/scripts/verify_layout.sh \
  swift-from-zero-to-advanced/scripts/verify_shared_docs.sh \
  swift-from-zero-to-advanced/README.md \
  swift-from-zero-to-advanced/projects/README.md \
  swift-from-zero-to-advanced/references/authoring-rules.md \
  swift-from-zero-to-advanced/references/bilingual-style-guide.md \
  swift-from-zero-to-advanced/references/chapter-template.md \
  swift-from-zero-to-advanced/references/learning-paths.md \
  swift-from-zero-to-advanced/glossary/core-terms.md
git commit -F - <<'EOF'
Raise the Swift tutorial quality contract before rewriting chapters

The existing course scaffold established paths and files, but it still
allowed shallow content to count as progress. This change rewrites the
shared tutorial contract so later chapter work is forced to clear a
much stronger bar around long-form depth, bilingual consistency, and
project continuity.

Constraint: The rewrite must preserve the existing tutorial root while replacing the weak scaffold-first standard
Rejected: Expand Part 1 first and fix shared rules later | would let weak chapter conventions leak into the rewrite
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Treat these shared docs as the course contract and update them before weakening any chapter-level standard
Tested: bash swift-from-zero-to-advanced/scripts/verify_layout.sh
Tested: bash swift-from-zero-to-advanced/scripts/verify_shared_docs.sh
Not-tested: Part 1 chapter prose or later-part blueprint depth
EOF
```

Expected:

```text
[branch-name abc1234] Raise the Swift tutorial quality contract before rewriting chapters
```

### Task 2: Replace the Part 1 Scaffold With the New Ten-Chapter Architecture

**Files:**
- Modify: `swift-from-zero-to-advanced/scripts/verify_part1.sh`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/overview.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-running-swift.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-values-and-types.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-strings-and-program-io.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-control-flow-for-commands.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-functions-and-program-shape.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-structs-and-data-modeling.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-collections-and-task-state.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-optionals-and-safe-parsing.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/09-enums-and-pattern-matching.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/10-build-task-cli-lite-v1.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/drills/README.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/checkpoint/part-1-checkpoint.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/project/part-1-task-cli-lite.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/references/part-1-glossary.md`
- Delete: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-swift-setup-and-first-program.md`
- Delete: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-constants-variables-and-types.md`
- Delete: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-control-flow.md`
- Delete: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-functions-and-decomposition.md`
- Delete: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-collections.md`
- Delete: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-optionals-and-basic-error-handling.md`
- Delete: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-strings-tuples-and-pattern-matching.md`
- Delete: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-part-1-project.md`

- [ ] **Step 1: Replace the Part 1 verifier so it matches the new chapter set**

```bash
# swift-from-zero-to-advanced/scripts/verify_part1.sh
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
```

- [ ] **Step 2: Run the verifier and confirm the new Part 1 architecture is missing**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-from-zero-to-advanced/scripts/verify_part1.sh
```

Expected:

```text
missing:swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-running-swift.md
```

- [ ] **Step 3: Replace the old chapter map with the new ten-chapter skeleton**

```bash
rm \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-swift-setup-and-first-program.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-constants-variables-and-types.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-control-flow.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-functions-and-decomposition.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-collections.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-optionals-and-basic-error-handling.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-strings-tuples-and-pattern-matching.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-part-1-project.md
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/overview.md
# Part 1: Swift Fundamentals

## Why Part 1 Exists

Part 1 teaches Swift as a language through one continuous command-line project.

## What You Will Learn

- how Swift code runs
- how Swift models values and types
- how to move from loose scripts to a small structured program
- how to model domain data with structs, enums, collections, and optionals

## Chapter Sequence

1. Running Swift
2. Values and Types
3. Strings and Program I/O
4. Control Flow for Commands
5. Functions and Program Shape
6. Structs and Data Modeling
7. Collections and Task State
8. Optionals and Safe Parsing
9. Enums and Pattern Matching
10. Build TaskCLI Lite v1

## Project Spine

The whole part builds toward `TaskCLI Lite v1`, a small Swift command-line task
manager that stays intentionally simple while teaching real Swift semantics.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-running-swift.md
# Chapter 01: Running Swift

## Problem

Explain what it means to run Swift at all.

## Running Example

Start with a minimal command-line program.

## Semantic Deep Dive

Explain `swift`, `swiftc`, and the difference between script execution and a
compiled executable.

## Code Evolution

Move from one inline print to a tiny argument-aware program.

## Common Mistakes

List the first toolchain misunderstandings.

## Drills

- concept check
- code reading
- hands-on extension

## Checkpoint

State what the reader should now understand.

## Glossary

List the chapter's bilingual terms.

## English Recap

Summarize the key technical rules in English.

## Project Bridge

Connect the chapter to the first `TaskCLI Lite` execution path.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-values-and-types.md
# Chapter 02: Values and Types

## Problem

Explain how Swift treats values and type information in small programs.

## Running Example

Model a few task-related values with `let`, `var`, and inferred types.

## Semantic Deep Dive

Explain type inference and why explicit annotations still matter.

## Code Evolution

Move from loosely named values to a clearer data slice.

## Common Mistakes

Explain mutability and annotation misuse.

## Drills

- concept check
- code reading
- hands-on extension

## Checkpoint

State the values-and-types outcome.

## Glossary

List the chapter's bilingual terms.

## English Recap

Summarize the key technical rules in English.

## Project Bridge

Say how these values become the first project state.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-strings-and-program-io.md
# Chapter 03: Strings and Program I/O

## Problem

Explain how small Swift programs talk to the outside world.

## Running Example

Read command-line arguments and format human-readable output.

## Semantic Deep Dive

Explain strings, interpolation, and the program boundary.

## Code Evolution

Improve one brittle output path into a clearer CLI surface.

## Common Mistakes

Explain argument and formatting pitfalls.

## Drills

- concept check
- code reading
- hands-on extension

## Checkpoint

State the input-and-output outcome.

## Glossary

List the chapter's bilingual terms.

## English Recap

Summarize the key technical rules in English.

## Project Bridge

Connect the chapter to command-driven program behavior.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-control-flow-for-commands.md
# Chapter 04: Control Flow for Commands

## Problem

Show how a small CLI decides what to do.

## Running Example

Branch on commands and iterate over task data.

## Semantic Deep Dive

Explain `if`, `switch`, loops, and why command routing is a natural fit.

## Code Evolution

Move from ad hoc branching to a more readable command flow.

## Common Mistakes

Explain branch duplication and loop confusion.

## Drills

- concept check
- code reading
- hands-on extension

## Checkpoint

State the control-flow outcome.

## Glossary

List the chapter's bilingual terms.

## English Recap

Summarize the key technical rules in English.

## Project Bridge

Connect this chapter to command dispatch.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-functions-and-program-shape.md
# Chapter 05: Functions and Program Shape

## Problem

Show why one long script stops scaling quickly.

## Running Example

Break one command-line script into named functions.

## Semantic Deep Dive

Explain parameters, return values, scope, and decomposition.

## Code Evolution

Refactor one repeated path into clearer functions.

## Common Mistakes

Explain over-extraction and vague naming.

## Drills

- concept check
- code reading
- hands-on extension

## Checkpoint

State the function-design outcome.

## Glossary

List the chapter's bilingual terms.

## English Recap

Summarize the key technical rules in English.

## Project Bridge

Connect this chapter to a shaped program instead of a single script.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-structs-and-data-modeling.md
# Chapter 06: Structs and Data Modeling

## Problem

Explain when loose values stop being enough.

## Running Example

Introduce a `Task` struct for the CLI domain.

## Semantic Deep Dive

Explain what a struct is and why value-oriented modeling matters in Swift.

## Code Evolution

Replace separate task fields with a small domain model.

## Common Mistakes

Explain why structs are not just dictionaries with syntax.

## Drills

- concept check
- code reading
- hands-on extension

## Checkpoint

State the modeling outcome.

## Glossary

List the chapter's bilingual terms.

## English Recap

Summarize the key technical rules in English.

## Project Bridge

Connect this chapter to `TaskCLI Lite` data design.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-collections-and-task-state.md
# Chapter 07: Collections and Task State

## Problem

Show how a CLI stores more than one task safely.

## Running Example

Use arrays first, then compare when dictionaries or sets help.

## Semantic Deep Dive

Explain ordering, lookup, uniqueness, and mutation costs.

## Code Evolution

Move from one task to a task collection.

## Common Mistakes

Explain collection misuse and accidental complexity.

## Drills

- concept check
- code reading
- hands-on extension

## Checkpoint

State the collection-design outcome.

## Glossary

List the chapter's bilingual terms.

## English Recap

Summarize the key technical rules in English.

## Project Bridge

Connect the chapter to `TaskCLI Lite` task storage.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-optionals-and-safe-parsing.md
# Chapter 08: Optionals and Safe Parsing

## Problem

Show why command input is fragile until absence and failure are modeled.

## Running Example

Parse missing and invalid input safely.

## Semantic Deep Dive

Explain `nil`, optional binding, and safe failure paths.

## Code Evolution

Replace unsafe assumptions with explicit parsing branches.

## Common Mistakes

Explain force unwraps and blurry error states.

## Drills

- concept check
- code reading
- hands-on extension

## Checkpoint

State the safe-parsing outcome.

## Glossary

List the chapter's bilingual terms.

## English Recap

Summarize the key technical rules in English.

## Project Bridge

Connect the chapter to safe CLI command parsing.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/09-enums-and-pattern-matching.md
# Chapter 09: Enums and Pattern Matching

## Problem

Show how a command system becomes clearer when the command space is modeled.

## Running Example

Introduce an enum-backed command parser.

## Semantic Deep Dive

Explain enums, cases, associated meaning, and pattern matching.

## Code Evolution

Move from raw strings to typed command handling.

## Common Mistakes

Explain overusing strings and underusing pattern matching.

## Drills

- concept check
- code reading
- hands-on extension

## Checkpoint

State the enum-and-matching outcome.

## Glossary

List the chapter's bilingual terms.

## English Recap

Summarize the key technical rules in English.

## Project Bridge

Connect the chapter to the command model that feeds the final integration.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/10-build-task-cli-lite-v1.md
# Chapter 10: Build TaskCLI Lite v1

## Problem

Explain how the earlier pieces come together into one coherent small program.

## Running Example

Assemble the first real `TaskCLI Lite v1` flow.

## Semantic Deep Dive

Review how the earlier chapters interact in one codebase.

## Code Evolution

Move from chapter-local pieces to a readable integrated version.

## Common Mistakes

Explain premature architecture and missing integration checks.

## Drills

- concept check
- code reading
- hands-on extension

## Checkpoint

State the integration outcome.

## Glossary

List the chapter's bilingual terms.

## English Recap

Summarize the key technical rules in English.

## Project Bridge

Explain how this chapter closes Part 1 and hands off to Part 2.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/drills/README.md
# Part 1 Drills

Part 1 drills should stay short enough for normal study sessions, but they
should never collapse into one-mode repetition.

Each chapter should contain:

- one concept check
- one code-reading task
- one hands-on extension
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/checkpoint/part-1-checkpoint.md
# Part 1 Checkpoint

## What The Reader Should Be Able To Do

- explain how Swift code runs
- model task data with structs
- use collections and optionals safely
- route commands through readable control flow
- integrate a small typed command-line program
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/project/part-1-task-cli-lite.md
# Part 1 Project: TaskCLI Lite

## Goal

Build a small task-oriented command-line tool that stays simple enough for Part
1 while still feeling like a real program.

## Required Capabilities

- list tasks
- add tasks
- mark tasks as done

## Part 1 Constraint

The goal is clarity and semantic control, not industrial architecture.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/references/part-1-glossary.md
# Part 1 Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Toolchain | 工具链 | The commands used to build and run Swift code. |
| Struct | 结构体 | A value type used to model grouped data. |
| Optional binding | 可选值绑定 | A safe way to unwrap an optional. |
| Pattern matching | 模式匹配 | Matching a value against structured cases. |
```

- [ ] **Step 4: Run the new Part 1 verifier**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-from-zero-to-advanced/scripts/verify_part1.sh
git diff --check -- swift-from-zero-to-advanced
```

Expected:

```text
part1-ok
```

- [ ] **Step 5: Commit the Part 1 architecture migration**

Run:

```bash
git add \
  swift-from-zero-to-advanced/scripts/verify_part1.sh \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals
git commit -F - <<'EOF'
Replace the Part 1 scaffold with the new chapter architecture

The old Part 1 file map locked the tutorial into a weak concept order and
chapter naming scheme. This change swaps in the new ten-chapter
architecture so long-form writing can grow inside the right boundaries.

Constraint: The rewrite needs better chapter boundaries before long-form prose is authored
Rejected: Keep the old eight-chapter file map and only expand the text | would preserve weak sequencing and push structural debt into every later edit
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Treat this new chapter map as the stable Part 1 structure unless the design spec is revised again
Tested: bash swift-from-zero-to-advanced/scripts/verify_part1.sh
Not-tested: Full long-form chapter prose
EOF
```

Expected:

```text
[branch-name def5678] Replace the Part 1 scaffold with the new chapter architecture
```

### Task 3: Write the Opening Long-Form Part 1 Chapters

**Files:**
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/overview.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-running-swift.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-values-and-types.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-strings-and-program-io.md`

- [ ] **Step 1: Write content checks for the overview and the first three chapters**

Run:

```bash
cd /Users/yangyang/ai_projs/math
rg -q "swiftc" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-running-swift.md
rg -q "type inference" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-values-and-types.md
rg -q "CommandLine.arguments" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-strings-and-program-io.md
```

Expected:

```text
The commands should fail because the skeleton chapters do not contain the long-form content yet.
```

- [ ] **Step 2: Confirm the current skeleton is too thin**

Run:

```bash
cd /Users/yangyang/ai_projs/math
rg -n "swiftc|CommandLine.arguments|English Recap" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-running-swift.md
```

Expected:

```text
Only section headings or no matches appear; the chapter does not yet contain long-form technical explanation.
```

- [ ] **Step 3: Replace the opening files with long-form tutorial prose**

````markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/overview.md
# Part 1: Swift Fundamentals

## Why Part 1 Exists

Part 1 is where the reader stops treating Swift as "just another syntax" and
starts seeing what the language is trying to optimize for: explicitness,
safety, readable modeling, and small pieces that compose into a real program.

The whole part is organized around one question:

How do you grow from "I can print a line in Swift" to "I can build and extend a
small command-line tool without losing control of the code"?

## What Part 1 Does Not Try To Do

Part 1 does not teach everything in Swift.
It also does not front-load architecture, protocols, generics, testing, or
concurrency.

Those belong later.
Part 1 is about getting the language and the mental model right.

## What You Will Learn

By the end of Part 1, you should be able to:

- run Swift code as a script and as a compiled executable
- reason about values, mutability, and type information
- read input from the command line and format useful output
- shape a small program with control flow and functions
- model domain data with structs
- store and update task state with collections
- parse unsafe or missing input with optionals
- model commands with enums and pattern matching
- integrate all of that into `TaskCLI Lite v1`

## Chapter Sequence

1. Running Swift
2. Values and Types
3. Strings and Program I/O
4. Control Flow for Commands
5. Functions and Program Shape
6. Structs and Data Modeling
7. Collections and Task State
8. Optionals and Safe Parsing
9. Enums and Pattern Matching
10. Build TaskCLI Lite v1

## Project Spine

The Part 1 project is `TaskCLI Lite v1`.
At the start, it is barely a program.
By the end, it is still intentionally small, but it is real enough to carry the
semantics from the whole part.

## How To Study Part 1

- run the examples, even when they look trivial
- compare the chapter code before and after the code-evolution sections
- do the drills in order: concept, reading, extension
- do not skip the English recap if you want stronger technical vocabulary

## What Changes In Part 2

Part 2 does not replace Part 1.
It assumes Part 1 is already stable and then asks how the same program should be
structured, tested, and extended when the codebase grows.
````

````markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-running-swift.md
# Chapter 01: Running Swift

## Problem

Before Swift can feel like a language, it has to feel like a real toolchain.
Many readers coming from Python, JavaScript, Go, or Java have an immediate
question:

What exactly is the difference between running Swift code with `swift` and
building a program with `swiftc`?

If that question stays fuzzy, later chapters also stay fuzzy.

## Running Example

Create a file called `hello.swift`:

~~~swift
print("Hello from Swift")
print("Arguments:", CommandLine.arguments.dropFirst())
~~~

Run it as a script:

~~~bash
swift hello.swift one two
~~~

Then compile it:

~~~bash
swiftc hello.swift -o hello
./hello one two
~~~

## Semantic Deep Dive

`swift` is the command you use when you want the Swift toolchain to interpret
and execute the source directly as a script-like workflow.
`swiftc` is the compiler driver that turns Swift source into an executable.

For Part 1, the important point is not the internal implementation detail.
The important point is the development model:

- `swift` is quick for experimentation
- `swiftc` makes the "this is a program" boundary explicit

That distinction matters because `TaskCLI Lite` will eventually live in a real
Swift package, not as a permanent throwaway script.

## Code Evolution

The first version prints raw output.
The second version makes the command-line boundary explicit:

~~~swift
let arguments = Array(CommandLine.arguments.dropFirst())

if arguments.isEmpty {
    print("No command provided.")
} else {
    print("Command:", arguments[0])
}
~~~

This is still tiny, but it already does two useful things:

- it treats the outside world as input, not magic
- it makes later command handling possible

## Common Mistakes

- assuming `CommandLine.arguments` contains only user arguments
  It also contains the program path as the first element.
- assuming `swift` and `swiftc` are interchangeable
  They can run the same source, but they express different development intents.
- delaying the executable model too long
  If you stay in "just a script" mode forever, later package structure feels
  unnatural.

## Drills

- concept check: explain the difference between `swift` and `swiftc` in one
  sentence each
- code reading: say what `CommandLine.arguments.dropFirst()` removes and why
- hands-on extension: print the number of user-supplied arguments before
  printing the first command

## Checkpoint

You should now be able to explain how a Swift source file becomes running
program behavior and why command-line arguments are already a useful boundary.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Toolchain | 工具链 | The commands used to build and run Swift code. |
| Executable | 可执行程序 | A compiled program that can be launched directly. |

## English Recap

- `swift` is useful for quick execution and exploration.
- `swiftc` makes the compilation step explicit.
- `CommandLine.arguments` is the first bridge between the program and the user.

## Project Bridge

`TaskCLI Lite` starts as "a file that can run."
That sounds small, but it is the first step toward a command-driven program
instead of a disconnected code snippet.
````

````markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-values-and-types.md
# Chapter 02: Values and Types

## Problem

Small programs feel easy until their values stop being obvious.
Swift makes an early promise: values should be readable, type information should
stay visible, and mutability should be explicit.

If you ignore that promise, your Part 1 code will still run, but it will stop
feeling like Swift.

## Running Example

Start with a tiny slice of task data:

~~~swift
let title = "Buy milk"
var isDone = false
let priority: Int = 2

print(title, isDone, priority)
~~~

## Semantic Deep Dive

Swift uses `let` for constants and `var` for values that may change.
That looks simple, but it changes how you think about state:

- values should be immutable unless mutation is part of the model
- types can often be inferred, but type inference is not a reason to hide
  useful intent

Type inference（类型推断） works well when the expression is already clear.
Explicit annotations are better when the annotation adds information the reader
would otherwise have to reconstruct.

## Code Evolution

Here is a noisier first version:

~~~swift
var taskTitle: String = "Buy milk"
var taskDone: Bool = false
var taskPriority: Int = 2
~~~

Now tighten the model:

~~~swift
let title = "Buy milk"
var isDone = false
let priority: Int = 2
~~~

This version is better for Part 1 because:

- names are shorter but still domain-specific
- only the value that may actually change uses `var`
- the one explicit type annotation shows where type clarity matters

## Common Mistakes

- using `var` everywhere "just in case"
  That weakens the meaning of mutation in the program.
- adding explicit types to every line
  That often makes the code louder without making it clearer.
- treating type inference as if types disappear
  They do not disappear; Swift still has a concrete type system underneath the
  syntax.

## Drills

- concept check: when is `let` better than `var` even if the code would still
  compile with `var`?
- code reading: name the inferred types in the running example
- hands-on extension: add a `notes` value and decide whether it should use type
  inference or an explicit annotation

## Checkpoint

You should now be able to explain why Swift makes mutability explicit and how
type inference can help without replacing deliberate type design.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Type inference | 类型推断 | Swift infers a type from the expression and context. |
| Mutability | 可变性 | Whether a value is allowed to change after declaration. |

## English Recap

- `let` communicates stable intent.
- `var` should mean real mutation, not convenience.
- Type inference is helpful when the expression is already clear.

## Project Bridge

The task CLI cannot stay as raw print statements forever.
It needs stable values, meaningful names, and intentional mutation before the
command logic starts to grow.
````

````markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-strings-and-program-io.md
# Chapter 03: Strings and Program I/O

## Problem

A command-line program lives at a boundary:

- the user gives it text
- the program turns that text into behavior
- the program sends text back

So even before we talk about complex parsing, we need to understand strings and
program I/O as a design surface, not just as syntax.

## Running Example

~~~swift
let arguments = Array(CommandLine.arguments.dropFirst())

if arguments.isEmpty {
    print("Usage: task-cli-lite <command>")
} else {
    let command = arguments[0]
    print("Received command: \(command)")
}
~~~

## Semantic Deep Dive

String interpolation is not just a convenience feature.
It is one of the simplest ways to keep output readable while still composing
dynamic values.

Program I/O also creates a discipline:

- input arrives as strings
- your program has to decide what those strings mean
- output should be designed for the human on the other side

That is why `CommandLine.arguments` matters so early.
It is the narrow entrance through which the whole CLI will later grow.

## Code Evolution

A weaker version hard-codes output:

~~~swift
print("Received something")
~~~

A stronger version makes the CLI behavior visible:

~~~swift
let arguments = Array(CommandLine.arguments.dropFirst())

if arguments.isEmpty {
    print("Usage: task-cli-lite <command>")
} else {
    let command = arguments[0]
    print("Received command: \(command)")
}
~~~

This is still simple, but now the program is expressing a real interface.

## Common Mistakes

- treating user input as if it is already valid structure
  At this stage it is just text.
- building unreadable output strings with too much inline noise
  String interpolation helps, but clarity still depends on how you compose the
  message.
- skipping usage text because the program is "only for learning"
  Clear output is part of learning to design programs well.

## Drills

- concept check: why is command-line input best thought of as a boundary rather
  than a convenience?
- code reading: explain what happens when no arguments are supplied
- hands-on extension: add support for printing the raw argument count alongside
  the command

## Checkpoint

You should now be able to read command-line input, explain how string
interpolation improves output clarity, and describe why CLI text is a design
surface instead of a side detail.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Command-line argument | 命令行参数 | A value provided to the program when it starts. |
| String interpolation | 字符串插值 | Embedding values inside a string literal. |

## English Recap

- `CommandLine.arguments` is the raw input boundary for the CLI.
- String interpolation helps make dynamic output readable.
- Good CLI output is part of program design, not an afterthought.

## Project Bridge

At this point, `TaskCLI Lite` can accept a command-shaped string and respond to
it in a human-readable way.
The next step is deciding how the program should branch on those commands.
````

- [ ] **Step 4: Verify the opening long-form content**

Run:

```bash
cd /Users/yangyang/ai_projs/math
rg -q "swiftc" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-running-swift.md
rg -q "Type inference（类型推断）" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-values-and-types.md
rg -q "CommandLine.arguments" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-strings-and-program-io.md
bash swift-from-zero-to-advanced/scripts/verify_part1.sh
git diff --check -- swift-from-zero-to-advanced
```

Expected:

```text
part1-ok
```

- [ ] **Step 5: Commit the opening long-form chapters**

Run:

```bash
git add \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/overview.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-running-swift.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-values-and-types.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-strings-and-program-io.md
git commit -F - <<'EOF'
Write the opening long-form Swift tutorial chapters

The new Part 1 chapter map is only useful if the early chapters immediately
feel like real tutorial prose instead of a renamed scaffold. This change
establishes the depth, code style, and bilingual tone for the opening
section of the course.

Constraint: Part 1 must serve readers who know another language but are new to Swift itself
Rejected: Keep the opening chapters terse and defer depth until later topics | would preserve the same shallow first impression the rewrite is trying to remove
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Use these chapters as the prose benchmark for the rest of Part 1
Tested: bash swift-from-zero-to-advanced/scripts/verify_part1.sh
Tested: rg -q "swiftc" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-running-swift.md
Tested: rg -q "Type inference（类型推断）" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-values-and-types.md
Tested: rg -q "CommandLine.arguments" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-strings-and-program-io.md
Not-tested: Later Part 1 chapter depth
EOF
```

Expected:

```text
[branch-name ghi9012] Write the opening long-form Swift tutorial chapters
```

### Task 4: Write the Middle Long-Form Part 1 Chapters

**Files:**
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-control-flow-for-commands.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-functions-and-program-shape.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-structs-and-data-modeling.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-collections-and-task-state.md`

- [ ] **Step 1: Write content checks for the middle chapter group**

Run:

```bash
cd /Users/yangyang/ai_projs/math
rg -q "switch" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-control-flow-for-commands.md
rg -q "func" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-functions-and-program-shape.md
rg -q "struct Task" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-structs-and-data-modeling.md
rg -q "\\[Task\\]" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-collections-and-task-state.md
```

Expected:

```text
The commands should fail because the middle chapters still contain only the structural skeleton.
```

- [ ] **Step 2: Confirm the middle chapters are still outline-level**

Run:

```bash
cd /Users/yangyang/ai_projs/math
sed -n '1,120p' swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-structs-and-data-modeling.md
```

Expected:

```text
The file still reads like a chapter shell rather than a long-form tutorial chapter.
```

- [ ] **Step 3: Replace the middle chapter group with long-form prose**

````markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-control-flow-for-commands.md
# Chapter 04: Control Flow for Commands

## Problem

Once a CLI can read text, it needs a rule for deciding what to do with that
text.
That means control flow is no longer an abstract syntax lesson.
It is the mechanism that turns one input surface into multiple program
behaviors.

## Running Example

~~~swift
let arguments = Array(CommandLine.arguments.dropFirst())

if arguments.isEmpty {
    print("Usage: task-cli-lite <command>")
} else {
    let command = arguments[0]

    switch command {
    case "list":
        print("Listing tasks")
    case "add":
        print("Adding a task")
    default:
        print("Unknown command: \(command)")
    }
}
~~~

## Semantic Deep Dive

Swift gives you several control-flow tools, but they are not interchangeable in
meaning.

- `if` is good for binary or narrow conditional checks
- `switch` is stronger when the input space has named branches
- loops are for repeated work over data or repeated checks over state

In a command router, `switch` often communicates intent better than a stack of
`if` statements because the reader can see the command space in one place.

## Code Evolution

A weaker version looks like this:

~~~swift
if command == "list" {
    print("Listing tasks")
}

if command == "add" {
    print("Adding a task")
}
~~~

This version is weaker because it:

- spreads related cases apart
- makes it easier to forget the fallback path
- hides that all branches are answering the same question

Using `switch` makes the control surface visible.

## Common Mistakes

- writing several unrelated `if` blocks when the program is really choosing
  between known cases
- forgetting the unknown-command path
- mixing command routing and data mutation in the same unreadable block

## Drills

- concept check: when is `switch` clearer than `if` in a CLI?
- code reading: trace the behavior for `task-cli-lite list`
- hands-on extension: add a `help` command and decide whether it belongs in the
  same `switch`

## Checkpoint

You should now be able to explain why command routing is a control-flow design
problem and why `switch` is often the best fit for a small CLI command surface.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Branch | 分支 | One path selected by a condition or case. |
| Fallback | 兜底路径 | The path used when no intended command matches. |

## English Recap

- CLI control flow is about mapping one input space to multiple behaviors.
- `switch` often expresses command routing better than repeated `if` blocks.
- Good control flow makes the command space readable in one place.

## Project Bridge

`TaskCLI Lite` now has real command routing.
The next problem is structure: once those branches start growing, how do we keep
the program from collapsing into one giant file?
````

````markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-functions-and-program-shape.md
# Chapter 05: Functions and Program Shape

## Problem

As soon as a CLI has more than one branch, the file starts to feel heavier.
The question becomes:

How do we keep the program readable without pretending we need "architecture"
before we have even learned the language well?

## Running Example

~~~swift
func printUsage() {
    print("Usage: task-cli-lite <list|add>")
}

func handleList() {
    print("Listing tasks")
}

func handleAdd() {
    print("Adding a task")
}
~~~

## Semantic Deep Dive

Functions are not only about reuse.
They are also about shape.

When a program has a visible set of responsibilities, named functions let the
reader see those responsibilities directly:

- print usage
- handle list
- handle add

That is much easier to reason about than one long body that mixes output,
control flow, and data mutation.

## Code Evolution

A weaker version keeps everything inline.
A stronger version extracts named operations:

~~~swift
func handle(command: String) {
    switch command {
    case "list":
        handleList()
    case "add":
        handleAdd()
    default:
        printUsage()
    }
}
~~~

This still is not over-engineered.
It is simply a clearer program shape.

## Common Mistakes

- extracting functions without a clear responsibility
- naming functions after mechanics instead of intent
- pushing too many unrelated values through a function signature

## Drills

- concept check: why are functions about program shape, not only reuse?
- code reading: identify the responsibility boundary of `handle(command:)`
- hands-on extension: split one inline fallback path into a named helper

## Checkpoint

You should now be able to explain why a small CLI benefits from named
responsibilities before it benefits from larger abstractions.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Function signature | 函数签名 | The name, parameters, and return information of a function. |
| Responsibility | 职责 | The single job a function or unit should own. |

## English Recap

- Functions improve the shape of a program, not only its reuse story.
- Good function names expose intent.
- Small, clear helpers are enough for Part 1.

## Project Bridge

`TaskCLI Lite` now has a more readable command path.
The next step is data modeling: the program still talks mostly in loose values,
which is where Swift's `struct` story begins to matter.
````

````markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-structs-and-data-modeling.md
# Chapter 06: Structs and Data Modeling

## Problem

Up to this point, the CLI can still fake its way forward with separate values:

- a title string
- a done flag
- maybe a priority integer

But once those values belong to one conceptual thing, leaving them separate
stops being clarity and starts being friction.

## Running Example

~~~swift
struct Task {
    let title: String
    var isDone: Bool
}

let firstTask = Task(title: "Buy milk", isDone: false)
print(firstTask.title)
~~~

## Semantic Deep Dive

`struct` is one of the most important parts of early Swift thinking.
It gives you a way to model related data as one named value.

Struct（结构体） is not just a prettier dictionary.
It does several things at once:

- gives the data a real name
- makes the fields explicit
- supports value-oriented design

This is where Value semantics（值语义） starts to become more than vocabulary.
If you copy a value type, Swift wants you to reason in terms of independent
values rather than shared mutable identity.

## Code Evolution

A weaker version spreads task data across independent values:

~~~swift
let title = "Buy milk"
var isDone = false
~~~

A stronger version models the domain explicitly:

~~~swift
struct Task {
    let title: String
    var isDone: Bool
}
~~~

This is better because the code is finally speaking in domain units instead of
just in raw fields.

## Common Mistakes

- delaying `struct` because separate values still "work"
- confusing "value type" with "immutable forever"
- using a dictionary when the domain shape is already known

## Drills

- concept check: why is `Task` a better unit than separate `title` and `isDone`
  values?
- code reading: explain what information the struct declaration makes visible
- hands-on extension: add a `priority` field and decide whether it should use a
  default value

## Checkpoint

You should now be able to explain why a struct is a modeling tool, not merely a
syntax feature, and why it moves the CLI closer to a real program.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Struct | 结构体 | A value type used to model grouped data. |
| Value semantics | 值语义 | Treating copied values as independent values rather than shared mutable identity. |

## English Recap

- A `struct` gives the domain a named data model.
- Swift encourages value-oriented modeling early.
- `Task` is the first real domain object in the CLI.

## Project Bridge

`TaskCLI Lite` no longer has to pretend task data is just loose local state.
It can now store and move around real task values, which makes the next topic
natural: collections.
````

````markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-collections-and-task-state.md
# Chapter 07: Collections and Task State

## Problem

One `Task` is a modeling exercise.
Multiple `Task` values are a program.

As soon as the CLI needs to list, add, or update more than one task, the
question becomes:

What collection shape gives us readable, predictable task state?

## Running Example

~~~swift
struct Task {
    let title: String
    var isDone: Bool
}

var tasks: [Task] = [
    Task(title: "Buy milk", isDone: false),
    Task(title: "Read Swift docs", isDone: true),
]
~~~

## Semantic Deep Dive

For Part 1, an array is the right first collection because it preserves order
and keeps the mental model simple.

Swift also gives you dictionaries and sets, but they solve different problems:

- arrays preserve sequence
- dictionaries optimize lookup by key
- sets optimize uniqueness

The best collection is not the one you personally like most.
It is the one that matches the access pattern of the program.

## Code Evolution

A weaker version hard-codes one task.
A stronger version moves to `[Task]`:

~~~swift
for task in tasks {
    print("- \(task.title) [done: \(task.isDone)]")
}
~~~

Now the CLI can iterate over real state instead of one-off examples.

## Common Mistakes

- choosing a set just because uniqueness sounds attractive
- choosing a dictionary before the program really has stable lookup keys
- mutating collection state inline everywhere without a clear update path

## Drills

- concept check: why is `[Task]` the right first collection for Part 1?
- code reading: explain what the loop prints for the sample state
- hands-on extension: append one extra task and print the updated list

## Checkpoint

You should now be able to compare arrays, dictionaries, and sets in terms of
program needs and explain why task state begins with an array in Part 1.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Array | 数组 | An ordered collection of values. |
| State | 状态 | The current data the program is holding and changing. |

## English Recap

- `[Task]` is the right first storage model for Part 1.
- Collection choice should follow access patterns.
- Arrays make the CLI's task state visible and iterable.

## Project Bridge

`TaskCLI Lite` can now hold real task state.
The next problem is safety: user input is still messy, and the program still
needs a better way to represent missing or invalid command data.
````

- [ ] **Step 4: Verify the middle chapter group**

Run:

```bash
cd /Users/yangyang/ai_projs/math
rg -q "switch" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-control-flow-for-commands.md
rg -q "func handle" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-functions-and-program-shape.md
rg -q "struct Task" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-structs-and-data-modeling.md
rg -q "\\[Task\\]" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-collections-and-task-state.md
bash swift-from-zero-to-advanced/scripts/verify_part1.sh
git diff --check -- swift-from-zero-to-advanced
```

Expected:

```text
part1-ok
```

- [ ] **Step 5: Commit the middle long-form chapters**

Run:

```bash
git add \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-control-flow-for-commands.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-functions-and-program-shape.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-structs-and-data-modeling.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-collections-and-task-state.md
git commit -F - <<'EOF'
Deepen the middle Part 1 engineering chapters

Part 1 starts to become a real Swift course in the middle chapters,
because this is where command flow, function boundaries, structs, and
collections turn scattered syntax into program shape. These chapters
need more than topic labels; they need enough prose and code to build
the reader's Swift mental model.

Constraint: The middle of Part 1 must introduce real Swift modeling without importing Part 2 architecture too early
Rejected: Keep the middle chapters light and save modeling depth for Part 2 | would delay Swift's core value-oriented design too long
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep these chapters focused on readability and semantic clarity, not on early framework-building
Tested: bash swift-from-zero-to-advanced/scripts/verify_part1.sh
Tested: rg -q "switch" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-control-flow-for-commands.md
Tested: rg -q "func handle" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-functions-and-program-shape.md
Tested: rg -q "struct Task" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-structs-and-data-modeling.md
Tested: rg -q "\\[Task\\]" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-collections-and-task-state.md
Not-tested: Final integration chapter
EOF
```

Expected:

```text
[branch-name jkl3456] Deepen the middle Part 1 engineering chapters
```

### Task 5: Finish the Long-Form Part 1 Rewrite

**Files:**
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-optionals-and-safe-parsing.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/09-enums-and-pattern-matching.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/10-build-task-cli-lite-v1.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/drills/README.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/checkpoint/part-1-checkpoint.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/project/part-1-task-cli-lite.md`
- Modify: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/references/part-1-glossary.md`

- [ ] **Step 1: Write content checks for the closing Part 1 files**

Run:

```bash
cd /Users/yangyang/ai_projs/math
rg -q "Optional binding（可选值绑定）" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-optionals-and-safe-parsing.md
rg -q "enum Command" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/09-enums-and-pattern-matching.md
rg -q "TaskCLI Lite v1" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/10-build-task-cli-lite-v1.md
```

Expected:

```text
The commands should fail because the closing chapters still contain only the skeleton content.
```

- [ ] **Step 2: Confirm the current closing files are still structural**

Run:

```bash
cd /Users/yangyang/ai_projs/math
sed -n '1,120p' swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-optionals-and-safe-parsing.md
```

Expected:

```text
The file still reads like a placeholder frame rather than a finished tutorial chapter.
```

- [ ] **Step 3: Replace the closing chapter group and the Part 1 support docs**

````markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-optionals-and-safe-parsing.md
# Chapter 08: Optionals and Safe Parsing

## Problem

The command line is hostile in a very ordinary way:

- the user may forget an argument
- the user may provide the wrong shape of argument
- the program may expect data that is not there

Swift refuses to let that ambiguity stay invisible.
That is why optionals matter.

## Running Example

~~~swift
let arguments = Array(CommandLine.arguments.dropFirst())
let maybeTitle = arguments.dropFirst().first

if let title = maybeTitle {
    print("Will add task: \(title)")
} else {
    print("Missing task title.")
}
~~~

## Semantic Deep Dive

Optional（可选值） means "a value may be present, or it may be `nil`."
Swift does not let you pretend otherwise.

Optional binding（可选值绑定） is one of the most important control points in
the language because it forces you to answer a concrete question:

What should the program do if the value is absent?

In a CLI, that is exactly the right question.

## Code Evolution

A weak version assumes the value exists:

~~~swift
let title = arguments[1]
print("Will add task: \(title)")
~~~

That version may crash.

A stronger version handles absence explicitly:

~~~swift
let maybeTitle = arguments.dropFirst().first

if let title = maybeTitle {
    print("Will add task: \(title)")
} else {
    print("Missing task title.")
}
~~~

## Common Mistakes

- force-unwrapping because the input "should" exist
- treating missing input and invalid input as the same state
- using optionals as mystery values instead of explaining what `nil` means in
  the current context

## Drills

- concept check: why is `nil` useful rather than annoying in a CLI parser?
- code reading: say what happens when the user types `task-cli-lite add`
- hands-on extension: add a second validation branch for an empty title string

## Checkpoint

You should now be able to explain why optionals are central to Swift safety and
why parsing code becomes clearer when absence is explicit.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Optional | 可选值 | A value that may contain data or `nil`. |
| Optional binding | 可选值绑定 | A safe way to unwrap an optional. |

## English Recap

- Optionals make absence explicit.
- Optional binding gives the program a safe decision point.
- CLI parsing gets better when missing input is modeled clearly.

## Project Bridge

`TaskCLI Lite` can now stop crashing on routine user mistakes.
The next step is making the command space itself typed instead of leaving it as
raw strings forever.
````

````markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/09-enums-and-pattern-matching.md
# Chapter 09: Enums and Pattern Matching

## Problem

Strings are enough to get a command router working, but they are not a great
long-term model for a known command space.

If the CLI really understands commands, the program should say so in its types.

## Running Example

~~~swift
enum Command: String {
    case list
    case add
    case done
}

let arguments = Array(CommandLine.arguments.dropFirst())

if let first = arguments.first, let command = Command(rawValue: first) {
    print("Parsed command: \(command)")
} else {
    print("Unknown command")
}
~~~

## Semantic Deep Dive

Enum（枚举） gives a name to a fixed space of cases.
That is exactly what a small command system is.

Pattern matching（模式匹配） then lets the program branch on typed cases instead
of raw string guesses:

~~~swift
switch command {
case .list:
    print("Listing tasks")
case .add:
    print("Adding a task")
case .done:
    print("Completing a task")
}
~~~

This is a very Swift-shaped improvement:

- the input still starts as text
- parsing turns it into a typed command
- `switch` operates on the typed model instead of repeating string comparisons

## Code Evolution

A raw-string approach works:

~~~swift
if first == "list" { ... }
~~~

But the enum approach gives the command surface a type boundary:

~~~swift
enum Command: String {
    case list
    case add
    case done
}
~~~

That makes the intent and the allowed command space explicit.

## Common Mistakes

- leaving a known command space as ad hoc strings forever
- thinking enums are only for large complex models
- writing a typed enum and then still routing mostly through raw strings

## Drills

- concept check: why is an enum better than raw strings for a fixed command
  space?
- code reading: describe what happens if the raw value does not match any case
- hands-on extension: add a `help` case and route it through the same parser

## Checkpoint

You should now be able to explain how enums and pattern matching turn a weak
string-based command router into a typed command model.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Enum | 枚举 | A type that represents one of several named cases. |
| Pattern matching | 模式匹配 | Matching a value against structured cases. |

## English Recap

- Enums model a fixed command space cleanly.
- Pattern matching keeps branching aligned with typed cases.
- Parsing text into types is one of Swift's core strengths.

## Project Bridge

`TaskCLI Lite` finally has a typed command model.
The last step is integration: combining values, structs, collections,
optionals, and enums into one readable Part 1 program.
````

````markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/10-build-task-cli-lite-v1.md
# Chapter 10: Build TaskCLI Lite v1

## Problem

A course can teach features one by one and still leave the reader unable to
assemble them.
So the final Part 1 question is:

Can we build one small, readable program that actually uses the ideas from the
whole part together?

## Running Example

~~~swift
import Foundation

struct Task {
    let title: String
    var isDone: Bool
}

enum Command: String {
    case list
    case add
    case done
}

func usage() {
    print("TaskCLI Lite")
    print("  list")
    print("  add <title>")
    print("  done <title>")
}
~~~

## Semantic Deep Dive

By the end of Part 1, the program should show several different kinds of Swift
thinking at once:

- values and mutability choices
- command-line input and output
- control flow for known command paths
- functions for readability
- structs for domain data
- arrays for task state
- optionals for safe parsing
- enums for command modeling

The point is not that the CLI becomes "production ready."
The point is that the code now has a coherent language shape.

## Code Evolution

A final small integration surface can look like this:

~~~swift
func parseCommand(from arguments: [String]) -> Command? {
    guard let first = arguments.first else {
        return nil
    }

    return Command(rawValue: first)
}

func listTasks(_ tasks: [Task]) {
    for task in tasks {
        print("- \(task.title) [done: \(task.isDone)]")
    }
}
~~~

This is still a tiny program.
That is intentional.
Part 1 is finished when the code is readable, typed, and extendable, not when
it resembles a full application architecture.

## Common Mistakes

- trying to import protocols, layers, and abstractions from future parts
- forgetting that a finished Part 1 project can still be small
- focusing on "feature count" instead of code clarity and semantic control

## Drills

- concept check: list the Part 1 ideas that appear in the integrated program
- code reading: trace the `add` command from raw arguments to output
- hands-on extension: sketch how you would add a `help` command without
  redesigning the whole program

## Checkpoint

You should now be able to explain how `TaskCLI Lite v1` uses the core Part 1
ideas together and why that small integration is enough preparation for Part 2.

## Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Integration | 集成 | Combining separate ideas into one coherent program. |
| Starter surface | 起始表面 | The minimal stable codebase future work can build on. |

## English Recap

- Part 1 ends with a coherent small program, not with isolated feature demos.
- `TaskCLI Lite v1` is intentionally small but semantically complete.
- Part 2 will change structure and engineering depth, not replace the Part 1 foundation.

## Project Bridge

This is the handoff point into Part 2.
The reader now has a program small enough to understand and stable enough to
refactor, test, and modularize in the next part of the course.
````

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/drills/README.md
# Part 1 Drills

## Why The Drill System Exists

Part 1 drills are not filler after the prose.
They exist to make the reader do three different kinds of work:

- recall and explain an idea
- read code carefully
- change code directly

## Required Drill Types

Each chapter should include:

- one concept check
- one code-reading task
- one hands-on extension

## Difficulty Rule

The drill should be hard enough to force attention, but short enough to fit
inside a normal study session.
Part 1 is building fluency, not endurance.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/checkpoint/part-1-checkpoint.md
# Part 1 Checkpoint

## Required Demonstrations

By the end of Part 1, the reader should be able to:

- explain how Swift code runs through `swift` and `swiftc`
- choose between `let` and `var` deliberately
- read command-line arguments and print useful output
- route commands through readable control flow
- break a program into named functions
- model domain data with `struct Task`
- store multiple tasks in `[Task]`
- use optionals to handle missing input safely
- model commands with `enum Command`
- explain how those ideas come together in `TaskCLI Lite v1`

## Completion Standard

Passing Part 1 means the reader can read, explain, and extend the small CLI
without starting over from scratch.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/project/part-1-task-cli-lite.md
# Part 1 Project: TaskCLI Lite

## Project Goal

Build a small command-line task manager that demonstrates the Part 1 Swift
foundation without importing heavier engineering structure from later parts.

## Minimum Capabilities

- `list`
- `add <title>`
- `done <title>`

## Why This Project Fits Part 1

The project is small enough that the reader can still see the whole program,
but rich enough to force real design choices around values, types, command
parsing, data modeling, and readable control flow.

## Finish Line

The Part 1 finish line is not "build a feature-rich task app."
It is:

- build a coherent small CLI
- use the language concepts on purpose
- leave the code in a shape that Part 2 can refactor instead of replace
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/references/part-1-glossary.md
# Part 1 Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| Toolchain | 工具链 | The commands used to build and run Swift code. |
| Type inference | 类型推断 | Swift infers a type from expression and context. |
| String interpolation | 字符串插值 | Embedding values inside a string literal. |
| Struct | 结构体 | A value type used to model grouped data. |
| Value semantics | 值语义 | Treating copies as independent values instead of shared mutable identity. |
| Optional binding | 可选值绑定 | A safe way to unwrap an optional. |
| Enum | 枚举 | A type that represents one of several named cases. |
| Pattern matching | 模式匹配 | Matching values against structured cases. |
| Integration | 集成 | Combining separate ideas into one coherent program. |
```

- [ ] **Step 4: Verify the full Part 1 rewrite**

Run:

```bash
cd /Users/yangyang/ai_projs/math
rg -q "Optional binding（可选值绑定）" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-optionals-and-safe-parsing.md
rg -q "enum Command" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/09-enums-and-pattern-matching.md
rg -q "TaskCLI Lite v1" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/10-build-task-cli-lite-v1.md
bash swift-from-zero-to-advanced/scripts/verify_part1.sh
git diff --check -- swift-from-zero-to-advanced
```

Expected:

```text
part1-ok
```

- [ ] **Step 5: Commit the finished Part 1 rewrite**

Run:

```bash
git add \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-optionals-and-safe-parsing.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/09-enums-and-pattern-matching.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/10-build-task-cli-lite-v1.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/drills/README.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/checkpoint/part-1-checkpoint.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/project/part-1-task-cli-lite.md \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/references/part-1-glossary.md
git commit -F - <<'EOF'
Finish the Part 1 long-form Swift rewrite

The closing Part 1 chapters decide whether the tutorial actually teaches
Swift semantics or merely gestures at them. This change completes the
Part 1 rewrite by making optionals, enums, integration, drills, and the
project handoff concrete enough to support the rest of the course.

Constraint: Part 1 must end with a coherent small program rather than with a project brief pretending to be integration
Rejected: Leave the last chapters and support docs short and rely on the reader to assemble the ideas alone | would undercut the whole rewrite
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep Part 1 semantically complete but intentionally small; do not pull Part 2 architecture back into this layer
Tested: bash swift-from-zero-to-advanced/scripts/verify_part1.sh
Tested: rg -q "Optional binding（可选值绑定）" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-optionals-and-safe-parsing.md
Tested: rg -q "enum Command" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/09-enums-and-pattern-matching.md
Tested: rg -q "TaskCLI Lite v1" swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/10-build-task-cli-lite-v1.md
Not-tested: Later-part blueprint quality
EOF
```

Expected:

```text
[branch-name mno7890] Finish the Part 1 long-form Swift rewrite
```

### Task 6: Turn Parts 2 Through 4 Into Detailed Blueprints

**Files:**
- Create: `swift-from-zero-to-advanced/scripts/verify_blueprints.sh`
- Modify: `swift-from-zero-to-advanced/parts/part-2-swift-core-engineering/README.md`
- Modify: `swift-from-zero-to-advanced/parts/part-3-apple-development-track/README.md`
- Modify: `swift-from-zero-to-advanced/parts/part-4-advanced-swift-track/README.md`

- [ ] **Step 1: Write the failing blueprint verifier**

```bash
# swift-from-zero-to-advanced/scripts/verify_blueprints.sh
#!/usr/bin/env bash
set -euo pipefail

parts=(
  "swift-from-zero-to-advanced/parts/part-2-swift-core-engineering/README.md"
  "swift-from-zero-to-advanced/parts/part-3-apple-development-track/README.md"
  "swift-from-zero-to-advanced/parts/part-4-advanced-swift-track/README.md"
)

headings=(
  "^## Part Goal$"
  "^## Learning Outcomes$"
  "^## Chapter Sequence$"
  "^## Project Evolution$"
  "^## Drill and Checkpoint Model$"
  "^## Dependencies and Handoffs$"
)

for part in "${parts[@]}"; do
  [[ -f "$part" ]] || {
    echo "missing:$part"
    exit 1
  }

  for heading in "${headings[@]}"; do
    rg -q "$heading" "$part" || {
      echo "missing-heading:$part:$heading"
      exit 1
    }
  done
done

rg -q "protocols" "swift-from-zero-to-advanced/parts/part-2-swift-core-engineering/README.md" || {
  echo "missing-topic:part2:protocols"
  exit 1
}

rg -q "SwiftUI" "swift-from-zero-to-advanced/parts/part-3-apple-development-track/README.md" || {
  echo "missing-topic:part3:SwiftUI"
  exit 1
}

rg -q "ARC" "swift-from-zero-to-advanced/parts/part-4-advanced-swift-track/README.md" || {
  echo "missing-topic:part4:ARC"
  exit 1
}

echo "blueprints-ok"
```

- [ ] **Step 2: Run the blueprint verifier and confirm the current later parts are still stubs**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-from-zero-to-advanced/scripts/verify_blueprints.sh
```

Expected:

```text
missing-heading:swift-from-zero-to-advanced/parts/part-2-swift-core-engineering/README.md:^## Part Goal$
```

- [ ] **Step 3: Rewrite the later-part readmes as detailed blueprints**

```markdown
# swift-from-zero-to-advanced/parts/part-2-swift-core-engineering/README.md
# Part 2: Swift Core Engineering

## Part Goal

Part 2 turns the small Part 1 CLI into a more modular Swift codebase.
The goal is not to abandon the beginner-sized program, but to reorganize it so
the reader learns how Swift code grows without collapsing.

## Learning Outcomes

- explain value versus reference modeling with more precision
- use enums, methods, and initializers as design tools
- use protocols and generics at the right scale
- structure code inside a Swift package
- write tests for core domain behavior
- understand the first real concurrency boundaries

## Chapter Sequence

1. Structs, classes, and value vs reference semantics
2. Enums, methods, initializers, and access control
3. Protocols and protocol-oriented design
4. Closures and functional patterns
5. Generics and constraints
6. Modules and Swift Package Manager
7. Testing with XCTest
8. Concurrency foundations
9. Part 2 project integration

## Project Evolution

The Part 2 project takes `TaskCLI Lite` and grows it into:

- `TaskCore` for shared domain logic
- `TaskCLI` for the executable boundary

The point is to separate modeling, behavior, and command-line wiring without
losing sight of the original Part 1 program.

## Drill and Checkpoint Model

Part 2 drills should shift from "can you explain one feature?" toward:

- design comparison
- test reading
- refactor-and-verify tasks

The checkpoint should ask the reader to explain why the package split exists and
how the code became easier to reason about.

## Dependencies and Handoffs

This part depends directly on Part 1.
It assumes the reader can already build and explain the small CLI.

It hands off to:

- Part 3 for Apple-platform specialization
- Part 4 for deeper Swift semantics and advanced hardening
```

```markdown
# swift-from-zero-to-advanced/parts/part-3-apple-development-track/README.md
# Part 3: Apple Development Track

## Part Goal

Part 3 applies the shared Swift foundation to app development through SwiftUI.
The domain stays familiar so the new complexity is UI, state, data flow, and
platform behavior rather than a brand-new business problem.

## Learning Outcomes

- build SwiftUI views from composable state-driven pieces
- reason about bindings and observable models
- structure app navigation and local persistence
- move asynchronous data into UI flows safely
- debug and preview app state more effectively

## Chapter Sequence

1. SwiftUI basics and mental model
2. State, binding, and observable models
3. Lists, forms, and navigation
4. Async UI data flow
5. Local persistence
6. App architecture and folder boundaries
7. Previews, debugging, testing, and accessibility
8. Part 3 project integration

## Project Evolution

The Part 3 project introduces `TaskFlow`, a SwiftUI app that reuses the same
task domain while adding:

- view composition
- state propagation
- persistence
- app-level architecture concerns

## Drill and Checkpoint Model

Part 3 drills should mix:

- UI reasoning
- state-tracing exercises
- hands-on view and data-flow changes

The checkpoint should verify that the reader can explain how state moves through
the app instead of only reproducing screens.

## Dependencies and Handoffs

Part 3 depends on Part 1 and Part 2.
It assumes the reader already understands the shared task domain and basic Swift
engineering organization.

It hands off to Part 4, where both the CLI and app spines are hardened through
advanced Swift semantics and performance work.
```

```markdown
# swift-from-zero-to-advanced/parts/part-4-advanced-swift-track/README.md
# Part 4: Advanced Swift Track

## Part Goal

Part 4 moves from productive Swift to deep Swift.
It focuses on the runtime and design tradeoffs behind ARC, concurrency,
performance, advanced protocols and generics, macros, interop, and tool-grade
code hardening.

## Learning Outcomes

- explain ARC and memory semantics clearly
- reason about deeper concurrency behavior and isolation
- identify obvious performance and allocation hot spots
- design stronger protocol and generic boundaries
- understand where macros and interop fit
- harden both project spines without losing clarity

## Chapter Sequence

1. ARC and memory semantics
2. Advanced concurrency
3. Performance and optimization
4. Advanced generics and protocol design
5. Result builders and macros
6. Objective-C interop and system APIs
7. Advanced CLI and tooling architecture
8. Final capstone and hardening

## Project Evolution

Part 4 pushes both project spines forward:

- `TaskCLI Pro` for the CLI / tooling side
- advanced `TaskFlow` enhancements for the app side

The purpose is not feature inflation.
The purpose is to show how advanced Swift choices change runtime behavior,
maintainability, and design tradeoffs.

## Drill and Checkpoint Model

Part 4 drills should emphasize:

- semantic explanation
- profiling or performance interpretation
- design tradeoff analysis
- targeted hardening changes

The checkpoint should ask the reader to defend a design decision, not just
repeat an API.

## Dependencies and Handoffs

Part 4 depends on the whole earlier course.
It assumes the reader can already build, refactor, and explain both the small
CLI line and the app line.

This is the final hardening layer rather than a new beginner path.
```

- [ ] **Step 4: Verify the detailed blueprints**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-from-zero-to-advanced/scripts/verify_blueprints.sh
git diff --check -- swift-from-zero-to-advanced
```

Expected:

```text
blueprints-ok
```

- [ ] **Step 5: Commit the later-part blueprints**

Run:

```bash
git add \
  swift-from-zero-to-advanced/scripts/verify_blueprints.sh \
  swift-from-zero-to-advanced/parts/part-2-swift-core-engineering/README.md \
  swift-from-zero-to-advanced/parts/part-3-apple-development-track/README.md \
  swift-from-zero-to-advanced/parts/part-4-advanced-swift-track/README.md
git commit -F - <<'EOF'
Turn the later Swift tutorial parts into detailed blueprints

The later parts can remain pre-prose for now, but they cannot stay as
tiny stubs if the course is going to feel coherent. This change upgrades
Parts 2 through 4 into usable blueprints that define chapter sequences,
project evolution, and handoff boundaries for future writing.

Constraint: Parts 2 through 4 are not being fully authored yet, but they still need enough detail to prevent another architecture reset later
Rejected: Leave the later parts as one-paragraph stubs until their full prose pass | would keep the overall course underspecified
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Expand future parts from these blueprints instead of replacing them with ad hoc chapter lists
Tested: bash swift-from-zero-to-advanced/scripts/verify_blueprints.sh
Not-tested: Full long-form prose for Parts 2 through 4
EOF
```

Expected:

```text
[branch-name pqr2345] Turn the later Swift tutorial parts into detailed blueprints
```

## Self-Review Notes

### Spec Coverage

- course-quality reset and shared writing contract: covered by Task 1
- Part 1 structural redesign into the new ten-chapter architecture: covered by Task 2
- Part 1 long-form prose, bilingual first-use terms, English recap, drills, and project bridge: covered by Tasks 3, 4, and 5
- detailed Part 2, Part 3, and Part 4 blueprints: covered by Task 6

No major requirement from the rewrite spec is left without a task.

### Placeholder Scan

- no `TODO` or `TBD` markers remain
- later-part blueprints are written as real mini-specs, not as future stubs
- each verification step has a concrete command and expected result

### Type Consistency

- `TaskCLI Lite v1` is used consistently as the Part 1 integrated CLI name
- the new Part 1 chapter set stays consistent across the file-structure section, Task 2 verifier, and later authoring tasks
- the shared bilingual examples use the same English and Chinese terms across the plan
