# Swift From Zero To Advanced Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the first implementation slice of the Swift tutorial by scaffolding the standalone tutorial directory, adding shared writing assets, building the full Part 1 document skeleton, and creating the starter `TaskCLI Lite` project scaffold.

**Architecture:** Build the tutorial as a dedicated top-level directory, `swift-from-zero-to-advanced/`, rather than mixing it into the existing deepagents tutorial tree. The first implementation slice stops at repository scaffold plus Part 1 and shared assets: it creates stable authoring rules, bilingual glossary/style guidance, chapter templates, Part 1 chapter skeletons, and a starter Swift package layout for the first CLI project. Later parts are intentionally excluded from this first implementation plan.

**Tech Stack:** Markdown, shell verification scripts, directory scaffolding, Swift Package starter files

---

## File Structure

### Create

- `swift-from-zero-to-advanced/README.md`
  - Top-level course index, part map, project-spine summary, and author-facing orientation.
- `swift-from-zero-to-advanced/projects/README.md`
  - Explain the two project spines and where each phase's code will live.
- `swift-from-zero-to-advanced/scripts/verify_layout.sh`
  - Repeatable shell check for top-level scaffold files.
- `swift-from-zero-to-advanced/scripts/verify_shared_docs.sh`
  - Repeatable shell check for glossary, style guide, authoring rules, and templates.
- `swift-from-zero-to-advanced/scripts/verify_part1.sh`
  - Repeatable shell check for the Part 1 overview, chapter skeletons, drills, checkpoint, and project brief.
- `swift-from-zero-to-advanced/scripts/verify_task_cli_lite.sh`
  - Repeatable shell check for the starter project scaffold; optionally run `swift test` if the toolchain exists.
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/overview.md`
  - Part 1 goals, chapter sequence, learning outcomes, and project bridge.
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-swift-setup-and-first-program.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-constants-variables-and-types.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-control-flow.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-functions-and-decomposition.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-collections.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-optionals-and-basic-error-handling.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-strings-tuples-and-pattern-matching.md`
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-part-1-project.md`
  - Initial chapter skeletons for all Part 1 material.
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/drills/README.md`
  - Drill taxonomy and per-chapter drill guidance.
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/checkpoint/part-1-checkpoint.md`
  - End-of-part checkpoint brief.
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/project/part-1-task-cli-lite.md`
  - Part 1 project brief and completion criteria.
- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/references/part-1-glossary.md`
  - Part-specific terminology list.
- `swift-from-zero-to-advanced/parts/part-2-swift-core-engineering/README.md`
- `swift-from-zero-to-advanced/parts/part-3-apple-development-track/README.md`
- `swift-from-zero-to-advanced/parts/part-4-advanced-swift-track/README.md`
  - Minimal future-part stubs with explicit scope statements.
- `swift-from-zero-to-advanced/references/authoring-rules.md`
  - Stable course-wide rules for chapter writing.
- `swift-from-zero-to-advanced/references/bilingual-style-guide.md`
  - Course-wide bilingual writing conventions.
- `swift-from-zero-to-advanced/references/chapter-template.md`
  - Canonical chapter template.
- `swift-from-zero-to-advanced/references/learning-paths.md`
  - Language-first, app-first, and CLI-first reading paths.
- `swift-from-zero-to-advanced/glossary/core-terms.md`
  - Shared course glossary for repeated terms.
- `swift-from-zero-to-advanced/projects/task-cli-lite/README.md`
  - Project overview and starter commands.
- `swift-from-zero-to-advanced/projects/task-cli-lite/starter/Package.swift`
  - Starter Swift package manifest.
- `swift-from-zero-to-advanced/projects/task-cli-lite/starter/Sources/TaskCLILite/main.swift`
  - Starter executable entry point.
- `swift-from-zero-to-advanced/projects/task-cli-lite/starter/Tests/TaskCLILiteTests/TaskCLILiteTests.swift`
  - Starter test file for the project.

### Do Not Modify

- `deepagents/`
- `langgraph/`
- `langchain/`
- `deepagents-internal-tutorial/`
- `deepagents-coding-platform/`
- `docs/superpowers/specs/2026-04-19-swift-from-zero-to-advanced-tutorial-design.md`

### Verification Surface

- `bash swift-from-zero-to-advanced/scripts/verify_layout.sh`
- `bash swift-from-zero-to-advanced/scripts/verify_shared_docs.sh`
- `bash swift-from-zero-to-advanced/scripts/verify_part1.sh`
- `bash swift-from-zero-to-advanced/scripts/verify_task_cli_lite.sh`
- `git diff --check -- swift-from-zero-to-advanced`

---

### Task 1: Scaffold the Tutorial Root and Course Navigation

**Files:**
- Create: `swift-from-zero-to-advanced/scripts/verify_layout.sh`
- Create: `swift-from-zero-to-advanced/README.md`
- Create: `swift-from-zero-to-advanced/projects/README.md`
- Create: `swift-from-zero-to-advanced/parts/part-2-swift-core-engineering/README.md`
- Create: `swift-from-zero-to-advanced/parts/part-3-apple-development-track/README.md`
- Create: `swift-from-zero-to-advanced/parts/part-4-advanced-swift-track/README.md`

- [ ] **Step 1: Write the failing layout verification script**

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

echo "layout-ok"
```

- [ ] **Step 2: Run the layout verification and confirm the scaffold is missing**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-from-zero-to-advanced/scripts/verify_layout.sh
```

Expected:

```text
missing:swift-from-zero-to-advanced/README.md
```

- [ ] **Step 3: Create the root tutorial index and later-part stubs**

```markdown
# swift-from-zero-to-advanced/README.md
# Swift From Zero to Advanced

## What This Project Is

This repository contains a four-part Swift curriculum for readers who already
know another programming language and want to learn Swift systematically.

The course combines:

- guided explanations
- drills and checkpoints
- project work
- bilingual terminology support

## Part Map

- Part 1: Swift Fundamentals
- Part 2: Swift Core Engineering
- Part 3: Apple Development Track
- Part 4: Advanced Swift Track

## Project Spines

- CLI / Package spine: `projects/task-cli-lite/`
- Apple app spine: introduced later through the app track

## First Implementation Slice

This first slice establishes:

- tutorial root scaffold
- shared authoring assets
- Part 1 chapter skeletons
- starter `TaskCLI Lite` project files

## How to Work in This Tutorial

1. Read the part overview before reading any chapter.
2. Use the glossary when Swift-specific terms appear.
3. Complete drills before moving to the checkpoint.
4. Treat the part project as the integration step, not optional material.
```

```markdown
# swift-from-zero-to-advanced/projects/README.md
# Projects

This directory contains the two long-running project spines used by the course.

## CLI / Package Spine

- Part 1: `task-cli-lite`
- Part 2: package-oriented refactor and expansion
- Part 4: advanced hardening and tooling upgrades

## Apple App Spine

- Introduced in Part 3
- Expanded in Part 4

The business domain stays consistent across both tracks so the learner spends
energy on Swift, not on learning a new domain every part.
```

```markdown
# swift-from-zero-to-advanced/parts/part-2-swift-core-engineering/README.md
# Part 2: Swift Core Engineering

This part will deepen the language foundation into engineering practice:

- type design
- protocols and generics
- packages
- testing
- concurrency foundations
```

```markdown
# swift-from-zero-to-advanced/parts/part-3-apple-development-track/README.md
# Part 3: Apple Development Track

This part will apply the shared Swift foundation to SwiftUI and app-oriented
development:

- UI composition
- state and data flow
- navigation
- persistence
- app architecture
```

```markdown
# swift-from-zero-to-advanced/parts/part-4-advanced-swift-track/README.md
# Part 4: Advanced Swift Track

This part will cover the advanced end of the curriculum:

- ARC and memory semantics
- deeper concurrency
- performance
- advanced generics and protocols
- macros and interop
```

- [ ] **Step 4: Run the layout verification again**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-from-zero-to-advanced/scripts/verify_layout.sh
```

Expected:

```text
layout-ok
```

- [ ] **Step 5: Commit the scaffold**

Run:

```bash
git add \
  swift-from-zero-to-advanced/scripts/verify_layout.sh \
  swift-from-zero-to-advanced/README.md \
  swift-from-zero-to-advanced/projects/README.md \
  swift-from-zero-to-advanced/parts/part-2-swift-core-engineering/README.md \
  swift-from-zero-to-advanced/parts/part-3-apple-development-track/README.md \
  swift-from-zero-to-advanced/parts/part-4-advanced-swift-track/README.md
git commit -F - <<'EOF'
Scaffold the Swift tutorial root and course navigation

Create the standalone tutorial directory and the smallest navigation
surface needed to anchor the course before Part 1 content is added.

Constraint: The Swift curriculum must live in its own dedicated directory rather than inside existing deepagents tutorial trees
Rejected: Start by writing Part 1 chapters without a stable root layout | would make later parts and shared assets hard to organize
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Keep top-level navigation stable and use it as the entry point for all future tutorial writing
Tested: bash swift-from-zero-to-advanced/scripts/verify_layout.sh
Not-tested: Any Part 1 content or project scaffolding
EOF
```

Expected:

```text
[branch-name abc1234] Scaffold the Swift tutorial root and course navigation
```

### Task 2: Add Shared Authoring Assets and Bilingual Rules

**Files:**
- Create: `swift-from-zero-to-advanced/scripts/verify_shared_docs.sh`
- Create: `swift-from-zero-to-advanced/references/authoring-rules.md`
- Create: `swift-from-zero-to-advanced/references/bilingual-style-guide.md`
- Create: `swift-from-zero-to-advanced/references/chapter-template.md`
- Create: `swift-from-zero-to-advanced/references/learning-paths.md`
- Create: `swift-from-zero-to-advanced/glossary/core-terms.md`

- [ ] **Step 1: Write the failing shared-doc verification script**

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
check_heading "^# Bilingual Style Guide$" "swift-from-zero-to-advanced/references/bilingual-style-guide.md"
check_heading "^# Chapter Template$" "swift-from-zero-to-advanced/references/chapter-template.md"
check_heading "^# Learning Paths$" "swift-from-zero-to-advanced/references/learning-paths.md"
check_heading "^# Core Terms Glossary$" "swift-from-zero-to-advanced/glossary/core-terms.md"

echo "shared-docs-ok"
```

- [ ] **Step 2: Run the shared-doc verification and confirm the files are missing**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-from-zero-to-advanced/scripts/verify_shared_docs.sh
```

Expected:

```text
missing:swift-from-zero-to-advanced/references/authoring-rules.md
```

- [ ] **Step 3: Write the shared authoring documents**

```markdown
# swift-from-zero-to-advanced/references/authoring-rules.md
# Authoring Rules

## Core Rules

- Every chapter must teach one clear central idea.
- Every chapter must include runnable code.
- Every chapter must include drills.
- Every chapter must explain common mistakes, not just happy paths.
- Project work must evolve incrementally from earlier material.
- Advanced material should be explicitly labeled when it is optional.

## Course Rhythm

- Explain the concept.
- Walk through code.
- Assign drills.
- Close with a checkpoint.
- Link the chapter back to the part project.

## Writing Standard

- Prefer precise explanations over motivational filler.
- Use the glossary when a term first appears.
- Keep examples small enough to hold in working memory.
```

```markdown
# swift-from-zero-to-advanced/references/bilingual-style-guide.md
# Bilingual Style Guide

## Chinese Responsibilities

- explain concepts
- explain pitfalls
- explain design reasoning
- explain exercise instructions

## English Responsibilities

- code
- API names
- type names
- canonical technical terms

## Rules

- Introduce important terms in Chinese and English on first use.
- Keep code and code comments primarily in English.
- Avoid full paragraph-by-paragraph translation.
- Use chapter glossaries to keep terminology stable.
```

```markdown
# swift-from-zero-to-advanced/references/chapter-template.md
# Chapter Template

## What You Will Build

One paragraph describing the concrete output of the chapter.

## Core Concepts

List the Swift ideas the learner should understand before moving on.

## Code Walkthrough

Explain the example code in a top-down order.

## Common Mistakes

Call out the likely semantic traps and debugging patterns.

## Drills

Include concept checks, code reading, and hands-on changes.

## Checkpoint

Define what the learner should be able to explain or implement.

## Glossary

List the chapter-specific terms in Chinese and English.

## Further Reading

Point to focused extensions instead of widening the chapter body.
```

```markdown
# swift-from-zero-to-advanced/references/learning-paths.md
# Learning Paths

## Language-First Path

Read all of Part 1, then Part 2, before specializing.

## App-First Path

Finish Part 1 and Part 2, then prioritize Part 3 before returning to Part 4.

## CLI / Engineering-First Path

Finish Part 1 and Part 2, then study the CLI spine deeply through Part 4.
```

```markdown
# swift-from-zero-to-advanced/glossary/core-terms.md
# Core Terms Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| value semantics | 值语义 | Changes produce independent values instead of shared mutable identity. |
| reference semantics | 引用语义 | Multiple names can point at the same mutable object. |
| optional | 可选值 | A type that can hold a value or `nil`. |
| protocol | 协议 | A contract that types can adopt. |
| generic | 泛型 | A reusable abstraction parameterized by type. |
| concurrency | 并发 | Coordinating work that can progress independently. |
| package | 包 | A SwiftPM unit containing targets and dependencies. |
| target | 目标 | A buildable unit inside a Swift package. |
| binding | 绑定 | A two-way connection to mutable state in SwiftUI. |
| actor | Actor | A type that protects mutable state in concurrent code. |
```

- [ ] **Step 4: Run the shared-doc verification again**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-from-zero-to-advanced/scripts/verify_shared_docs.sh
```

Expected:

```text
shared-docs-ok
```

- [ ] **Step 5: Commit the shared authoring layer**

Run:

```bash
git add \
  swift-from-zero-to-advanced/scripts/verify_shared_docs.sh \
  swift-from-zero-to-advanced/references/authoring-rules.md \
  swift-from-zero-to-advanced/references/bilingual-style-guide.md \
  swift-from-zero-to-advanced/references/chapter-template.md \
  swift-from-zero-to-advanced/references/learning-paths.md \
  swift-from-zero-to-advanced/glossary/core-terms.md
git commit -F - <<'EOF'
Add the shared authoring layer for the Swift tutorial

Lock the course-wide writing rules, bilingual conventions, glossary
entries, and chapter template before any part-specific drafting grows.

Constraint: The curriculum must stay bilingual without turning into full duplicate prose in two languages
Rejected: Let each chapter invent its own structure and terminology | would create inconsistency across a long tutorial
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Reuse these shared writing assets whenever new chapters or later parts are added
Tested: bash swift-from-zero-to-advanced/scripts/verify_shared_docs.sh
Not-tested: Any part-specific content or project scaffolds
EOF
```

Expected:

```text
[branch-name def5678] Add the shared authoring layer for the Swift tutorial
```

### Task 3: Create the Full Part 1 Skeleton

**Files:**
- Create: `swift-from-zero-to-advanced/scripts/verify_part1.sh`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/overview.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-swift-setup-and-first-program.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-constants-variables-and-types.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-control-flow.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-functions-and-decomposition.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-collections.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-optionals-and-basic-error-handling.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-strings-tuples-and-pattern-matching.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-part-1-project.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/drills/README.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/checkpoint/part-1-checkpoint.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/project/part-1-task-cli-lite.md`
- Create: `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/references/part-1-glossary.md`

- [ ] **Step 1: Write the failing Part 1 verification script**

```bash
# swift-from-zero-to-advanced/scripts/verify_part1.sh
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
```

- [ ] **Step 2: Run the Part 1 verification and confirm the files are missing**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-from-zero-to-advanced/scripts/verify_part1.sh
```

Expected:

```text
missing:swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/overview.md
```

- [ ] **Step 3: Create the Part 1 overview, chapter skeletons, and support files**

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/overview.md
# Part 1: Swift Fundamentals

## What This Part Covers

Part 1 establishes the language foundation for the whole course.

## Learning Outcomes

- read and write small Swift programs
- understand types, control flow, collections, and optionals
- prepare for the first CLI project

## Chapter Map

1. Swift Setup and First Program
2. Constants, Variables, and Types
3. Control Flow
4. Functions and Decomposition
5. Collections
6. Optionals and Basic Error Handling
7. Strings, Tuples, and Pattern Matching
8. Part 1 Project

## Part Project

The Part 1 project is `TaskCLI Lite`, a small command-line task manager.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/01-swift-setup-and-first-program.md
# Chapter 01: Swift Setup and First Program

## What You Will Build

A tiny Swift command-line program that prints output and reads command-line arguments.

## Core Concepts

- toolchain
- source file
- compilation vs execution

## Drills

- identify the role of `swift`
- run a first program

## Checkpoint

Explain what happens between source code and program output.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/02-constants-variables-and-types.md
# Chapter 02: Constants, Variables, and Types

## What You Will Build

A tiny model of task data using constants, variables, and explicit types.

## Core Concepts

- `let`
- `var`
- type inference
- annotations

## Drills

- convert values between `let` and `var`
- predict inferred types

## Checkpoint

Explain when explicit type annotations improve clarity.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/03-control-flow.md
# Chapter 03: Control Flow

## What You Will Build

Branching and looping logic for task-list command handling.

## Core Concepts

- `if`
- `switch`
- `for`
- `while`

## Drills

- rewrite `if` chains as `switch`
- trace loop execution by hand

## Checkpoint

Choose the right control-flow construct for a command parser.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/04-functions-and-decomposition.md
# Chapter 04: Functions and Decomposition

## What You Will Build

Extract reusable functions from one long script into smaller units.

## Core Concepts

- parameters
- return values
- local scope
- decomposition

## Drills

- extract one repeated code path into a function
- rename functions for clearer intent

## Checkpoint

Explain how functions reduce duplication in a CLI tool.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/05-collections.md
# Chapter 05: Collections

## What You Will Build

Store and update task items using arrays, dictionaries, and sets.

## Core Concepts

- arrays
- dictionaries
- sets
- iteration over collections

## Drills

- choose the right collection for a small scenario
- update and inspect collection contents

## Checkpoint

Compare array ordering with set uniqueness.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/06-optionals-and-basic-error-handling.md
# Chapter 06: Optionals and Basic Error Handling

## What You Will Build

Safely parse missing or invalid command input without crashing the program.

## Core Concepts

- optionals
- `if let`
- `guard let`
- simple error reporting

## Drills

- unwrap optional input safely
- distinguish invalid input from absent input

## Checkpoint

Explain why optionals are central to Swift safety.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/07-strings-tuples-and-pattern-matching.md
# Chapter 07: Strings, Tuples, and Pattern Matching

## What You Will Build

Human-readable task output and lightweight grouped values for the CLI.

## Core Concepts

- string interpolation
- tuples
- basic pattern matching

## Drills

- format task output clearly
- use tuples to return paired values

## Checkpoint

Explain when a tuple is enough and when a custom type is better.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/chapters/08-part-1-project.md
# Chapter 08: Part 1 Project

## What You Will Build

Integrate the Part 1 concepts into `TaskCLI Lite`.

## Core Concepts

- combining previous chapters
- keeping the program readable
- identifying what still feels awkward

## Drills

- complete missing command branches
- improve user-facing output

## Checkpoint

Show how the project uses types, control flow, functions, collections, and optionals together.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/drills/README.md
# Part 1 Drills

Each Part 1 chapter should include:

- one concept check
- one code-reading task
- one hands-on code change

The drills should stay short enough to fit inside a normal study session.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/checkpoint/part-1-checkpoint.md
# Part 1 Checkpoint

## Required Demonstrations

- explain the role of types and inference
- use control flow to route commands
- extract functions to simplify a script
- use collections and optionals safely

## Completion Standard

The learner should be able to explain and extend the `TaskCLI Lite` starter without starting over from scratch.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/project/part-1-task-cli-lite.md
# Part 1 Project: TaskCLI Lite

## Project Goal

Build a small command-line task manager that can:

- list tasks
- add tasks
- mark tasks as done

## Why This Project Fits Part 1

The project is small enough for language learners, but rich enough to practice
types, control flow, functions, collections, and optionals.

## Part 1 Finish Line

The project does not need industrial architecture yet. It needs clarity,
correctness, and visible use of the concepts from Part 1.
```

```markdown
# swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/references/part-1-glossary.md
# Part 1 Glossary

| English | 中文 | Meaning |
| --- | --- | --- |
| constant | 常量 | A value declared with `let`. |
| variable | 变量 | A value declared with `var`. |
| type inference | 类型推断 | Swift deduces the type from context. |
| control flow | 控制流 | The structures that guide execution order. |
| function | 函数 | A reusable unit of named behavior. |
| optional binding | 可选值绑定 | A safe way to unwrap optional values. |
| pattern matching | 模式匹配 | Matching values against structured cases. |
```

- [ ] **Step 4: Run the Part 1 verification again**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-from-zero-to-advanced/scripts/verify_part1.sh
```

Expected:

```text
part1-ok
```

- [ ] **Step 5: Commit the Part 1 skeleton**

Run:

```bash
git add \
  swift-from-zero-to-advanced/scripts/verify_part1.sh \
  swift-from-zero-to-advanced/parts/part-1-swift-fundamentals
git commit -F - <<'EOF'
Create the full Part 1 skeleton for the Swift curriculum

Lay down the complete Part 1 document structure so later drafting can
fill in content without revisiting chapter order, checkpoint scope, or
project positioning.

Constraint: The first implementation plan must stop at Part 1 structure instead of drafting the entire four-part course
Rejected: Create only one or two example chapters | would leave the course shape unstable and force later re-organization
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Treat these chapter files as stable skeletons and expand them incrementally instead of renaming or reordering them casually
Tested: bash swift-from-zero-to-advanced/scripts/verify_part1.sh
Not-tested: Full lesson prose, drill solutions, or executable Swift code
EOF
```

Expected:

```text
[branch-name ghi9012] Create the full Part 1 skeleton for the Swift curriculum
```

### Task 4: Add the TaskCLI Lite Starter Project Scaffold

**Files:**
- Create: `swift-from-zero-to-advanced/scripts/verify_task_cli_lite.sh`
- Create: `swift-from-zero-to-advanced/projects/task-cli-lite/README.md`
- Create: `swift-from-zero-to-advanced/projects/task-cli-lite/starter/Package.swift`
- Create: `swift-from-zero-to-advanced/projects/task-cli-lite/starter/Sources/TaskCLILite/main.swift`
- Create: `swift-from-zero-to-advanced/projects/task-cli-lite/starter/Tests/TaskCLILiteTests/TaskCLILiteTests.swift`

- [ ] **Step 1: Write the failing TaskCLI Lite verification script**

```bash
# swift-from-zero-to-advanced/scripts/verify_task_cli_lite.sh
#!/usr/bin/env bash
set -euo pipefail

project_root="swift-from-zero-to-advanced/projects/task-cli-lite"
starter_root="$project_root/starter"

required_files=(
  "$project_root/README.md"
  "$starter_root/Package.swift"
  "$starter_root/Sources/TaskCLILite/main.swift"
  "$starter_root/Tests/TaskCLILiteTests/TaskCLILiteTests.swift"
)

for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || {
    echo "missing:$path"
    exit 1
  }
done

rg -q "TaskCLI Lite" "$project_root/README.md" || {
  echo "missing-readme-title"
  exit 1
}

if command -v swift >/dev/null 2>&1; then
  (
    cd "$starter_root"
    swift test >/dev/null
  )
fi

echo "task-cli-lite-ok"
```

- [ ] **Step 2: Run the TaskCLI Lite verification and confirm the scaffold is missing**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-from-zero-to-advanced/scripts/verify_task_cli_lite.sh
```

Expected:

```text
missing:swift-from-zero-to-advanced/projects/task-cli-lite/README.md
```

- [ ] **Step 3: Write the starter project files**

```markdown
# swift-from-zero-to-advanced/projects/task-cli-lite/README.md
# TaskCLI Lite

This is the Part 1 starter project for the Swift curriculum.

## Starter Goals

- expose the learner to a real Swift package layout
- keep the executable small enough to understand
- give later chapters a stable place to add commands and behavior

## Starter Commands

- `list`
- `add <title>`
- `done <title>`
```

```swift
// swift-from-zero-to-advanced/projects/task-cli-lite/starter/Package.swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TaskCLILite",
    products: [
        .executable(name: "task-cli-lite", targets: ["TaskCLILite"]),
    ],
    targets: [
        .executableTarget(
            name: "TaskCLILite"
        ),
        .testTarget(
            name: "TaskCLILiteTests",
            dependencies: ["TaskCLILite"]
        ),
    ]
)
```

```swift
// swift-from-zero-to-advanced/projects/task-cli-lite/starter/Sources/TaskCLILite/main.swift
import Foundation

enum Command: String {
    case list
    case add
    case done
}

func usage() -> String {
    """
    TaskCLI Lite
      list
      add <title>
      done <title>
    """
}

let arguments = Array(CommandLine.arguments.dropFirst())

guard let first = arguments.first, let command = Command(rawValue: first) else {
    print(usage())
    exit(0)
}

switch command {
case .list:
    print("No tasks yet.")
case .add:
    let title = arguments.dropFirst().joined(separator: " ")
    print(title.isEmpty ? "Missing task title." : "Added: \(title)")
case .done:
    let title = arguments.dropFirst().joined(separator: " ")
    print(title.isEmpty ? "Missing task title." : "Completed: \(title)")
}
```

```swift
// swift-from-zero-to-advanced/projects/task-cli-lite/starter/Tests/TaskCLILiteTests/TaskCLILiteTests.swift
import XCTest
@testable import TaskCLILite

final class TaskCLILiteTests: XCTestCase {
    func testUsageMentionsSupportedCommands() {
        let text = usage()

        XCTAssertTrue(text.contains("list"))
        XCTAssertTrue(text.contains("add <title>"))
        XCTAssertTrue(text.contains("done <title>"))
    }
}
```

- [ ] **Step 4: Run the TaskCLI Lite verification again**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-from-zero-to-advanced/scripts/verify_task_cli_lite.sh
```

Expected:

```text
task-cli-lite-ok
```

- [ ] **Step 5: Commit the starter project scaffold**

Run:

```bash
git add \
  swift-from-zero-to-advanced/scripts/verify_task_cli_lite.sh \
  swift-from-zero-to-advanced/projects/task-cli-lite
git commit -F - <<'EOF'
Add the Part 1 TaskCLI Lite starter project scaffold

Create the first executable project surface for the curriculum so Part 1
chapters can refer to a stable codebase instead of isolated snippets.

Constraint: The starter project must stay small enough for beginners while still looking like a real Swift package
Rejected: Keep Part 1 code only as inline snippets in markdown | would weaken the project spine and make integration practice harder
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep this starter intentionally simple and evolve it through the course instead of front-loading later architecture here
Tested: bash swift-from-zero-to-advanced/scripts/verify_task_cli_lite.sh
Not-tested: Expanded command behavior beyond the starter stub
EOF
```

Expected:

```text
[branch-name jkl3456] Add the Part 1 TaskCLI Lite starter project scaffold
```

## Self-Review Notes

### Spec Coverage

- tutorial repository scaffold: covered by Task 1
- global writing rules and shared assets: covered by Task 2
- Part 1 structure and initial chapter skeletons: covered by Task 3
- starter CLI project line for Part 1: covered by Task 4

No spec requirement from the intended first implementation slice is left without a task.

### Placeholder Scan

- No `TBD`, `TODO`, or deferred implementation markers remain in tasks.
- All task steps name exact files.
- All write steps include concrete file content.
- All run steps include exact commands and expected outputs.

### Type and Naming Consistency

- Tutorial root stays `swift-from-zero-to-advanced/` throughout.
- Part 1 project naming stays `TaskCLI Lite` / `task-cli-lite` consistently.
- Verification scripts follow the same directory layout they test.
