# Swift Tutorial Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a brand-new `swift-tutorial/` product from scratch as a complete long-form Swift course with 8 parts, continuous project spines, labs, appendix materials, and verification scripts.

**Architecture:** The rebuild is organized as a tutorial product rather than a markdown scaffold. Work proceeds in layers: shared root contract and verifiers first, then projects and Part 1, then the remaining parts in order, then labs/appendix/final verification. The course centers on one stable domain and three linked project surfaces: `TaskCLI Lite`, `TaskCore + TaskCLI`, and `TaskFlow`.

**Tech Stack:** Markdown, shell verification scripts, Swift Package Manager for CLI/core project assets, Swift source files for tutorial project code, `rg`, `bash`

---

## File Structure

### Create

Root/tutorial contract:

- `swift-tutorial/README.md`
- `swift-tutorial/00-preface.md`
- `swift-tutorial/projects/README.md`
- `swift-tutorial/labs/README.md`
- `swift-tutorial/appendix/glossary.md`
- `swift-tutorial/appendix/answers.md`
- `swift-tutorial/appendix/environment-setup.md`
- `swift-tutorial/appendix/spm-cheatsheet.md`
- `swift-tutorial/appendix/swiftui-cheatsheet.md`
- `swift-tutorial/appendix/faq.md`
- `swift-tutorial/appendix/references.md`
- `swift-tutorial/scripts/verify_layout.sh`
- `swift-tutorial/scripts/verify_parts.sh`
- `swift-tutorial/scripts/verify_projects.sh`
- `swift-tutorial/scripts/verify_appendix.sh`
- `swift-tutorial/scripts/verify_task_cli_lite.sh`
- `swift-tutorial/scripts/verify_taskcore_taskcli.sh`

Part 1:

- `swift-tutorial/part1-language-foundations/01-toolchain-and-first-swift-program.md`
- `swift-tutorial/part1-language-foundations/02-values-types-and-mutability.md`
- `swift-tutorial/part1-language-foundations/03-strings-collections-and-control-flow.md`
- `swift-tutorial/part1-language-foundations/04-functions-optionals-enums-and-structs.md`
- `swift-tutorial/part1-language-foundations/05-build-taskcli-lite-v1.md`

Part 2:

- `swift-tutorial/part2-type-system-and-modeling/06-methods-properties-and-initializers.md`
- `swift-tutorial/part2-type-system-and-modeling/07-classes-vs-structs-and-value-vs-reference.md`
- `swift-tutorial/part2-type-system-and-modeling/08-protocols-protocol-extensions-and-abstraction-boundaries.md`
- `swift-tutorial/part2-type-system-and-modeling/09-generics-associated-types-and-type-driven-api-design.md`
- `swift-tutorial/part2-type-system-and-modeling/10-errors-results-and-modeling-failure.md`

Part 3:

- `swift-tutorial/part3-packages-testing-and-cli-engineering/11-swift-package-manager-and-module-boundaries.md`
- `swift-tutorial/part3-packages-testing-and-cli-engineering/12-testing-with-xctest-and-core-behavior.md`
- `swift-tutorial/part3-packages-testing-and-cli-engineering/13-parsing-rendering-and-storage-seams.md`
- `swift-tutorial/part3-packages-testing-and-cli-engineering/14-command-organization-and-cli-architecture.md`
- `swift-tutorial/part3-packages-testing-and-cli-engineering/15-build-taskcore-taskcli-v1.md`

Part 4:

- `swift-tutorial/part4-concurrency-performance-and-reliability/16-async-await-and-task-basics.md`
- `swift-tutorial/part4-concurrency-performance-and-reliability/17-actors-isolation-and-sendability.md`
- `swift-tutorial/part4-concurrency-performance-and-reliability/18-arc-memory-and-ownership-in-practice.md`
- `swift-tutorial/part4-concurrency-performance-and-reliability/19-performance-copying-and-measurement-mindset.md`
- `swift-tutorial/part4-concurrency-performance-and-reliability/20-reliability-cancellation-and-failure-surfaces.md`

Part 5:

- `swift-tutorial/part5-swiftui-foundations/21-swiftui-mental-model-and-view-composition.md`
- `swift-tutorial/part5-swiftui-foundations/22-state-binding-and-observable-models.md`
- `swift-tutorial/part5-swiftui-foundations/23-lists-forms-and-navigation-basics.md`
- `swift-tutorial/part5-swiftui-foundations/24-build-taskflow-v1.md`

Part 6:

- `swift-tutorial/part6-swiftui-dataflow-and-app-architecture/25-app-state-and-data-flow.md`
- `swift-tutorial/part6-swiftui-dataflow-and-app-architecture/26-persistence-and-model-integration.md`
- `swift-tutorial/part6-swiftui-dataflow-and-app-architecture/27-async-ui-updates-previews-and-testing.md`
- `swift-tutorial/part6-swiftui-dataflow-and-app-architecture/28-taskflow-architecture-and-feature-growth.md`

Part 7:

- `swift-tutorial/part7-advanced-swift-and-system-design/29-advanced-generics-and-protocol-design.md`
- `swift-tutorial/part7-advanced-swift-and-system-design/30-result-builders-macros-and-api-surface-judgment.md`
- `swift-tutorial/part7-advanced-swift-and-system-design/31-interop-system-apis-and-package-boundary-tradeoffs.md`
- `swift-tutorial/part7-advanced-swift-and-system-design/32-shared-abstractions-and-system-redesign.md`

Part 8:

- `swift-tutorial/part8-capstone-and-next-steps/33-capstone-rebuild-plan.md`
- `swift-tutorial/part8-capstone-and-next-steps/34-capstone-cli-and-core-hardening.md`
- `swift-tutorial/part8-capstone-and-next-steps/35-capstone-taskflow-hardening.md`
- `swift-tutorial/part8-capstone-and-next-steps/36-graduation-roadmap-and-next-steps.md`

Projects:

- `swift-tutorial/projects/task-cli-lite/README.md`
- `swift-tutorial/projects/task-cli-lite/starter/Package.swift`
- `swift-tutorial/projects/task-cli-lite/starter/Sources/TaskCLILite/main.swift`
- `swift-tutorial/projects/task-cli-lite/starter/Tests/TaskCLILiteTests/TaskCLILiteTests.swift`
- `swift-tutorial/projects/task-cli-lite/milestones/part1-v1.md`
- `swift-tutorial/projects/task-cli-lite/final/README.md`
- `swift-tutorial/projects/taskcore-taskcli/README.md`
- `swift-tutorial/projects/taskcore-taskcli/starter/Package.swift`
- `swift-tutorial/projects/taskcore-taskcli/starter/Sources/TaskCore/Task.swift`
- `swift-tutorial/projects/taskcore-taskcli/starter/Sources/TaskCore/TaskStore.swift`
- `swift-tutorial/projects/taskcore-taskcli/starter/Sources/TaskCLI/main.swift`
- `swift-tutorial/projects/taskcore-taskcli/starter/Tests/TaskCoreTests/TaskCoreTests.swift`
- `swift-tutorial/projects/taskcore-taskcli/milestones/part3-v1.md`
- `swift-tutorial/projects/taskcore-taskcli/milestones/part4-runtime-upgrade.md`
- `swift-tutorial/projects/taskcore-taskcli/final/README.md`
- `swift-tutorial/projects/taskflow/README.md`
- `swift-tutorial/projects/taskflow/starter/README.md`
- `swift-tutorial/projects/taskflow/milestones/part5-v1.md`
- `swift-tutorial/projects/taskflow/milestones/part6-architecture.md`
- `swift-tutorial/projects/taskflow/final/README.md`

Labs:

- `swift-tutorial/labs/part1-language-foundations.md`
- `swift-tutorial/labs/part2-type-system-and-modeling.md`
- `swift-tutorial/labs/part3-packages-testing-and-cli-engineering.md`
- `swift-tutorial/labs/part4-concurrency-performance-and-reliability.md`
- `swift-tutorial/labs/part5-swiftui-foundations.md`
- `swift-tutorial/labs/part6-swiftui-dataflow-and-app-architecture.md`
- `swift-tutorial/labs/part7-advanced-swift-and-system-design.md`
- `swift-tutorial/labs/part8-capstone.md`

### Modify

- `docs/superpowers/plans/2026-04-19-swift-tutorial-rebuild.md`
  - This implementation plan.

### Do Not Modify

- `deepagents/`
- `langgraph/`
- `langchain/`
- `deepagents-internal-tutorial/`
- `deepagents-coding-platform/`
- `swift-from-zero-to-advanced/`
  - This remains deleted; do not restore it as part of the rebuild.
- `docs/superpowers/specs/2026-04-19-swift-tutorial-rebuild-design.md`
  - Approved design spec; implementation follows it.

### Verification Surface

- `bash swift-tutorial/scripts/verify_layout.sh`
- `bash swift-tutorial/scripts/verify_parts.sh`
- `bash swift-tutorial/scripts/verify_projects.sh`
- `bash swift-tutorial/scripts/verify_appendix.sh`
- `bash swift-tutorial/scripts/verify_task_cli_lite.sh`
- `bash swift-tutorial/scripts/verify_taskcore_taskcli.sh`
- `git diff --check -- swift-tutorial docs/superpowers`

Note: `verify_parts.sh`, `verify_projects.sh`, and `verify_appendix.sh` are full-product verifiers. Until the corresponding trees are fully authored, intermediate tasks should use targeted file-existence, heading, and keyword checks for the files created in that task rather than expecting the global verifiers to pass early.

---

### Task 1: Create the Root Contract and Verification Scripts

**Files:**
- Create: `swift-tutorial/README.md`
- Create: `swift-tutorial/00-preface.md`
- Create: `swift-tutorial/projects/README.md`
- Create: `swift-tutorial/labs/README.md`
- Create: `swift-tutorial/scripts/verify_layout.sh`
- Create: `swift-tutorial/scripts/verify_parts.sh`
- Create: `swift-tutorial/scripts/verify_projects.sh`
- Create: `swift-tutorial/scripts/verify_appendix.sh`

- [ ] **Step 1: Write the layout verifier**

Create `swift-tutorial/scripts/verify_layout.sh` with these requirements:

- shebang `#!/usr/bin/env bash`
- `set -euo pipefail`
- require:
  - `swift-tutorial/README.md`
  - `swift-tutorial/00-preface.md`
  - all `part1-...` through `part8-...` directories
  - `swift-tutorial/projects`
  - `swift-tutorial/labs`
  - `swift-tutorial/appendix`
- require heading `^# 从零到高阶的 Swift 教程$` in `README.md`
- require heading `^# 前言：如何使用本教程$` in `00-preface.md`
- print exactly `layout-ok` on success

- [ ] **Step 2: Write the part, project, and appendix verifiers**

Create:

- `swift-tutorial/scripts/verify_parts.sh`
- `swift-tutorial/scripts/verify_projects.sh`
- `swift-tutorial/scripts/verify_appendix.sh`

Verifier expectations:

`verify_parts.sh`
- require all 36 chapter files
- require each file to start with `# 第`
- require `README.md` to mention all 8 parts
- print `parts-ok`

`verify_projects.sh`
- require all three project directories and all listed starter/milestones/final files
- require `TaskCLI Lite`, `TaskCore + TaskCLI`, and `TaskFlow` strings in the matching project READMEs
- print `projects-ok`

`verify_appendix.sh`
- require all appendix files and all 8 lab files
- require headings:
  - `# 术语表`
  - `# 练习与综合实验答案`
  - `# 环境准备`
  - `# Swift Package Manager 速查`
  - `# SwiftUI 速查`
  - `# 常见问题`
  - `# 参考资料`
- print `appendix-ok`

- [ ] **Step 3: Write the root README**

`swift-tutorial/README.md` must include:

- tutorial定位
- 读者画像
- 8 Part map with one paragraph per part
- continuous project spine
- how to use the tutorial
- tutorial特色

Keep the title exactly:

```markdown
# 从零到高阶的 Swift 教程
```

- [ ] **Step 4: Write the preface**

`swift-tutorial/00-preface.md` must include:

- 教程设计理念
- 为什么主读者是“会别的语言但没系统学过 Swift”
- 为什么教程先讲 Swift 语言与工程本体，再讲 Apple 专项
- 强双语写作约定
- 环境与工具要求
- 推荐学习路径
- 项目线说明
- 本教程不追求什么

Keep the title exactly:

```markdown
# 前言：如何使用本教程
```

- [ ] **Step 5: Write the projects and labs root guides**

`swift-tutorial/projects/README.md` must explain:

- the three project surfaces
- what `starter / milestones / final` mean
- how chapters connect to projects

`swift-tutorial/labs/README.md` must explain:

- what labs are
- how they differ from chapter drills
- when to do part-level labs

- [ ] **Step 6: Verify the root contract**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-tutorial/scripts/verify_layout.sh
bash swift-tutorial/scripts/verify_parts.sh || true
bash swift-tutorial/scripts/verify_projects.sh || true
bash swift-tutorial/scripts/verify_appendix.sh || true
git diff --check -- swift-tutorial docs/superpowers
```

Expected:

```text
layout-ok
```

The other verifiers may fail at this point because later files are not written yet.

- [ ] **Step 7: Commit the root contract**

```bash
git add \
  swift-tutorial/README.md \
  swift-tutorial/00-preface.md \
  swift-tutorial/projects/README.md \
  swift-tutorial/labs/README.md \
  swift-tutorial/scripts/verify_layout.sh \
  swift-tutorial/scripts/verify_parts.sh \
  swift-tutorial/scripts/verify_projects.sh \
  swift-tutorial/scripts/verify_appendix.sh
git commit -F - <<'EOF'
Define the Swift tutorial root contract and verifier surface

The rebuilt Swift tutorial needs a stable product skeleton before long-form
content is added. This establishes the root map, preface, project/lab guides,
and the verification scripts that keep the tutorial tree coherent as the rest
of the course is authored.

Constraint: The new tutorial must live under `swift-tutorial/` and behave like a complete repository tutorial product, not a markdown dump
Rejected: Start by writing chapters without locking root structure and verifiers | too easy to drift into inconsistent layout and broken navigation
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Expand the tutorial inside this contract; do not weaken the root verification surface in later tasks
Tested: bash swift-tutorial/scripts/verify_layout.sh
Tested: git diff --check -- swift-tutorial docs/superpowers
Not-tested: Chapter/project/appendix verifiers against the full future tree
EOF
```

---

### Task 2: Build the Project Surfaces and Part 1 Tutorial

**Files:**
- Create: `swift-tutorial/projects/task-cli-lite/README.md`
- Create: `swift-tutorial/projects/task-cli-lite/starter/Package.swift`
- Create: `swift-tutorial/projects/task-cli-lite/starter/Sources/TaskCLILite/main.swift`
- Create: `swift-tutorial/projects/task-cli-lite/starter/Tests/TaskCLILiteTests/TaskCLILiteTests.swift`
- Create: `swift-tutorial/projects/task-cli-lite/milestones/part1-v1.md`
- Create: `swift-tutorial/projects/task-cli-lite/final/README.md`
- Create: `swift-tutorial/scripts/verify_task_cli_lite.sh`
- Create: all 5 files under `swift-tutorial/part1-language-foundations/`

- [ ] **Step 1: Create the TaskCLI Lite starter package**

The starter package must:

- build with Swift Package Manager
- expose an executable target `TaskCLILite`
- support minimal `list`, `add <title>`, and `done <title>` command handling
- include at least one basic XCTest file

`verify_task_cli_lite.sh` must:

- `cd swift-tutorial/projects/task-cli-lite/starter`
- run `swift build`
- run `swift test`
- print `task-cli-lite-ok`

- [ ] **Step 2: Write the TaskCLI Lite project docs**

`projects/task-cli-lite/README.md` must explain:

- what this project is for
- how it evolves during Parts 1-2
- how to run the starter package

`projects/task-cli-lite/milestones/part1-v1.md` must summarize the Part 1 finish state.

`projects/task-cli-lite/final/README.md` must explain what the final Part 1 state should look like and how it connects forward into `TaskCore + TaskCLI`.

- [ ] **Step 3: Write the Part 1 chapters**

Create these files:

- `01-toolchain-and-first-swift-program.md`
- `02-values-types-and-mutability.md`
- `03-strings-collections-and-control-flow.md`
- `04-functions-optionals-enums-and-structs.md`
- `05-build-taskcli-lite-v1.md`

Each chapter must:

- be long-form
- explain why the topic appears now
- show a weaker starting state
- evolve to a stronger state
- include bilingual key terms
- include common mistakes
- include a compact English recap
- end with drills and a project handoff

Part 1 as a whole must clearly land in `TaskCLI Lite v1`.

- [ ] **Step 4: Verify the project and Part 1**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-tutorial/scripts/verify_task_cli_lite.sh
for file in \
  swift-tutorial/projects/task-cli-lite/README.md \
  swift-tutorial/projects/task-cli-lite/milestones/part1-v1.md \
  swift-tutorial/projects/task-cli-lite/final/README.md \
  swift-tutorial/part1-language-foundations/01-toolchain-and-first-swift-program.md \
  swift-tutorial/part1-language-foundations/02-values-types-and-mutability.md \
  swift-tutorial/part1-language-foundations/03-strings-collections-and-control-flow.md \
  swift-tutorial/part1-language-foundations/04-functions-optionals-enums-and-structs.md \
  swift-tutorial/part1-language-foundations/05-build-taskcli-lite-v1.md
do
  test -f "$file"
done
for file in \
  swift-tutorial/part1-language-foundations/01-toolchain-and-first-swift-program.md \
  swift-tutorial/part1-language-foundations/02-values-types-and-mutability.md \
  swift-tutorial/part1-language-foundations/03-strings-collections-and-control-flow.md \
  swift-tutorial/part1-language-foundations/04-functions-optionals-enums-and-structs.md \
  swift-tutorial/part1-language-foundations/05-build-taskcli-lite-v1.md
do
  rg -q '^# 第' "$file"
done
rg -F -q 'TaskCLI Lite' swift-tutorial/projects/task-cli-lite/README.md
rg -F -q 'TaskCLI Lite' swift-tutorial/projects/task-cli-lite/final/README.md
git diff --check -- swift-tutorial docs/superpowers
```

Expected:

```text
task-cli-lite-ok
```

The targeted file/heading checks should exit successfully. Do not expect `verify_projects.sh` or `verify_parts.sh` to pass yet because later project surfaces and parts are not written.

- [ ] **Step 5: Commit Part 1 and TaskCLI Lite**

```bash
git add \
  swift-tutorial/projects/task-cli-lite \
  swift-tutorial/scripts/verify_task_cli_lite.sh \
  swift-tutorial/part1-language-foundations
git commit -F - <<'EOF'
Author the Swift fundamentals path and TaskCLI Lite project

The rebuilt tutorial only becomes real once the reader can move through a full
Part 1 and a runnable starter project. This adds the TaskCLI Lite surface and
the opening tutorial sequence that teaches Swift fundamentals through one stable
CLI domain.

Constraint: Part 1 must remain tutorial-first and project-grounded, not collapse into syntax notes or over-engineered architecture
Rejected: Keep Part 1 lightweight and rely on later parts to make the tutorial feel real | would repeat the earlier Swift tutorial failure mode
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep future Part 2 work continuous with this CLI line instead of replacing it with a new domain or unrelated project
Tested: bash swift-tutorial/scripts/verify_task_cli_lite.sh
Tested: Targeted file, heading, and keyword checks for TaskCLI Lite docs and Part 1 chapters
Tested: git diff --check -- swift-tutorial docs/superpowers
Not-tested: Future Part 2+ project evolution
EOF
```

---

### Task 3: Author Part 2 and the Modeling Upgrade

**Files:**
- Create: all 5 files under `swift-tutorial/part2-type-system-and-modeling/`

- [ ] **Step 1: Write the Part 2 chapters**

Create these files:

- `06-methods-properties-and-initializers.md`
- `07-classes-vs-structs-and-value-vs-reference.md`
- `08-protocols-protocol-extensions-and-abstraction-boundaries.md`
- `09-generics-associated-types-and-type-driven-api-design.md`
- `10-errors-results-and-modeling-failure.md`

These chapters must:

- actively compare Swift modeling decisions with likely prior-language habits
- connect type design to the existing task domain
- avoid server-side or SwiftUI drift
- prepare the reader for package engineering in Part 3

- [ ] **Step 2: Update the part verifier expectations if needed**

Ensure `verify_parts.sh` checks these chapter files by exact path if any path is missing, but keep it as a full-tree verifier rather than weakening it for partial progress.

- [ ] **Step 3: Verify Part 2**

Run:

```bash
cd /Users/yangyang/ai_projs/math
for file in \
  swift-tutorial/part2-type-system-and-modeling/06-methods-properties-and-initializers.md \
  swift-tutorial/part2-type-system-and-modeling/07-classes-vs-structs-and-value-vs-reference.md \
  swift-tutorial/part2-type-system-and-modeling/08-protocols-protocol-extensions-and-abstraction-boundaries.md \
  swift-tutorial/part2-type-system-and-modeling/09-generics-associated-types-and-type-driven-api-design.md \
  swift-tutorial/part2-type-system-and-modeling/10-errors-results-and-modeling-failure.md
do
  test -f "$file"
  rg -q '^# 第' "$file"
done
git diff --check -- swift-tutorial docs/superpowers
```

Expected:

```text
The targeted checks exit successfully
```

- [ ] **Step 4: Commit Part 2**

```bash
git add swift-tutorial/part2-type-system-and-modeling
git commit -F - <<'EOF'
Deepen the Swift tutorial into type-system and modeling work

The course cannot meaningfully call itself "from zero to advanced" if it stops
at surface syntax. Part 2 turns the task domain into a vehicle for real Swift
modeling decisions around initialization, semantics, abstraction boundaries,
generics, and error design.

Constraint: Part 2 must still feel like a tutorial for experienced programmers new to Swift, not a reference section on the language manual
Rejected: Compress advanced modeling topics into short summaries and defer the real discussion to engineering chapters | would weaken the tutorial's technical spine
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Maintain continuity with the task domain and Part 1 code evolution instead of teaching these topics as detached theory
Tested: Targeted file and heading checks for the Part 2 chapters
Tested: git diff --check -- swift-tutorial docs/superpowers
Not-tested: Part 3 package/project integration
EOF
```

---

### Task 4: Build Part 3 and the TaskCore + TaskCLI Engineering Surface

**Files:**
- Create: all 5 files under `swift-tutorial/part3-packages-testing-and-cli-engineering/`
- Create: `swift-tutorial/projects/taskcore-taskcli/README.md`
- Create: `swift-tutorial/projects/taskcore-taskcli/starter/Package.swift`
- Create: `swift-tutorial/projects/taskcore-taskcli/starter/Sources/TaskCore/Task.swift`
- Create: `swift-tutorial/projects/taskcore-taskcli/starter/Sources/TaskCore/TaskStore.swift`
- Create: `swift-tutorial/projects/taskcore-taskcli/starter/Sources/TaskCLI/main.swift`
- Create: `swift-tutorial/projects/taskcore-taskcli/starter/Tests/TaskCoreTests/TaskCoreTests.swift`
- Create: `swift-tutorial/projects/taskcore-taskcli/milestones/part3-v1.md`
- Create: `swift-tutorial/projects/taskcore-taskcli/final/README.md`
- Create: `swift-tutorial/scripts/verify_taskcore_taskcli.sh`

- [ ] **Step 1: Create the TaskCore + TaskCLI starter package**

The starter package must:

- define library target `TaskCore`
- define executable target `TaskCLI`
- build and test successfully with Swift Package Manager
- contain at least one domain model file and one basic store/behavior file

`verify_taskcore_taskcli.sh` must:

- `cd swift-tutorial/projects/taskcore-taskcli/starter`
- run `swift build`
- run `swift test`
- print `taskcore-taskcli-ok`

- [ ] **Step 2: Write the project docs**

`projects/taskcore-taskcli/README.md` must explain:

- why the split exists
- what belongs in `TaskCore`
- what belongs in `TaskCLI`
- how the split connects Part 3 and Part 4

The milestone and final docs must summarize:

- Part 3 package boundary state
- how Part 4 later strengthens runtime behavior

- [ ] **Step 3: Write the Part 3 chapters**

Create:

- `11-swift-package-manager-and-module-boundaries.md`
- `12-testing-with-xctest-and-core-behavior.md`
- `13-parsing-rendering-and-storage-seams.md`
- `14-command-organization-and-cli-architecture.md`
- `15-build-taskcore-taskcli-v1.md`

These chapters must:

- teach package engineering through the real project
- keep tests concrete and non-performative
- explain CLI layering without prematurely inventing complex architecture

- [ ] **Step 4: Verify Part 3 and project engineering**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-tutorial/scripts/verify_taskcore_taskcli.sh
for file in \
  swift-tutorial/projects/taskcore-taskcli/README.md \
  swift-tutorial/projects/taskcore-taskcli/milestones/part3-v1.md \
  swift-tutorial/projects/taskcore-taskcli/final/README.md \
  swift-tutorial/part3-packages-testing-and-cli-engineering/11-swift-package-manager-and-module-boundaries.md \
  swift-tutorial/part3-packages-testing-and-cli-engineering/12-testing-with-xctest-and-core-behavior.md \
  swift-tutorial/part3-packages-testing-and-cli-engineering/13-parsing-rendering-and-storage-seams.md \
  swift-tutorial/part3-packages-testing-and-cli-engineering/14-command-organization-and-cli-architecture.md \
  swift-tutorial/part3-packages-testing-and-cli-engineering/15-build-taskcore-taskcli-v1.md
do
  test -f "$file"
done
for file in \
  swift-tutorial/part3-packages-testing-and-cli-engineering/11-swift-package-manager-and-module-boundaries.md \
  swift-tutorial/part3-packages-testing-and-cli-engineering/12-testing-with-xctest-and-core-behavior.md \
  swift-tutorial/part3-packages-testing-and-cli-engineering/13-parsing-rendering-and-storage-seams.md \
  swift-tutorial/part3-packages-testing-and-cli-engineering/14-command-organization-and-cli-architecture.md \
  swift-tutorial/part3-packages-testing-and-cli-engineering/15-build-taskcore-taskcli-v1.md
do
  rg -q '^# 第' "$file"
done
rg -F -q 'TaskCore + TaskCLI' swift-tutorial/projects/taskcore-taskcli/README.md
git diff --check -- swift-tutorial docs/superpowers
```

Expected:

```text
taskcore-taskcli-ok
```

The targeted file/heading checks should exit successfully. Do not expect `verify_projects.sh` or `verify_parts.sh` to pass yet because the TaskFlow surface and later parts are not written.

- [ ] **Step 5: Commit Part 3 and TaskCore + TaskCLI**

```bash
git add \
  swift-tutorial/part3-packages-testing-and-cli-engineering \
  swift-tutorial/projects/taskcore-taskcli \
  swift-tutorial/scripts/verify_taskcore_taskcli.sh
git commit -F - <<'EOF'
Turn the Swift tutorial into a package-engineering and testing course

Parts 1 and 2 establish the language spine, but the tutorial still needs a
real engineering surface. This adds the TaskCore plus TaskCLI package line and
the chapters that teach modules, testing, and CLI organization through the live
project instead of through detached examples.

Constraint: The engineering path must stay grounded in the shared task domain and remain buildable with SPM
Rejected: Describe package engineering only in prose while keeping project assets thin or non-runnable | would undercut the credibility of the tutorial product
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Future runtime and UI work must reuse this package line rather than fork away from it
Tested: bash swift-tutorial/scripts/verify_taskcore_taskcli.sh
Tested: Targeted file, heading, and keyword checks for TaskCore + TaskCLI docs and Part 3 chapters
Not-tested: SwiftUI project surface
EOF
```

---

### Task 5: Write Part 4 and the Runtime-Strengthening Layer

**Files:**
- Create: all 5 files under `swift-tutorial/part4-concurrency-performance-and-reliability/`
- Create: `swift-tutorial/projects/taskcore-taskcli/milestones/part4-runtime-upgrade.md`

- [ ] **Step 1: Write the Part 4 chapters**

Create:

- `16-async-await-and-task-basics.md`
- `17-actors-isolation-and-sendability.md`
- `18-arc-memory-and-ownership-in-practice.md`
- `19-performance-copying-and-measurement-mindset.md`
- `20-reliability-cancellation-and-failure-surfaces.md`

Each chapter must:

- tie the advanced Swift topic back to the existing `TaskCore + TaskCLI` line
- explain engineering consequences, not only syntax
- avoid drifting into framework tourism

- [ ] **Step 2: Write the Part 4 milestone doc**

`projects/taskcore-taskcli/milestones/part4-runtime-upgrade.md` must summarize:

- what changed from the Part 3 baseline
- how concurrency, reliability, and performance now affect the project

- [ ] **Step 3: Verify Part 4**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-tutorial/scripts/verify_taskcore_taskcli.sh
for file in \
  swift-tutorial/projects/taskcore-taskcli/milestones/part4-runtime-upgrade.md \
  swift-tutorial/part4-concurrency-performance-and-reliability/16-async-await-and-task-basics.md \
  swift-tutorial/part4-concurrency-performance-and-reliability/17-actors-isolation-and-sendability.md \
  swift-tutorial/part4-concurrency-performance-and-reliability/18-arc-memory-and-ownership-in-practice.md \
  swift-tutorial/part4-concurrency-performance-and-reliability/19-performance-copying-and-measurement-mindset.md \
  swift-tutorial/part4-concurrency-performance-and-reliability/20-reliability-cancellation-and-failure-surfaces.md
do
  test -f "$file"
done
for file in \
  swift-tutorial/part4-concurrency-performance-and-reliability/16-async-await-and-task-basics.md \
  swift-tutorial/part4-concurrency-performance-and-reliability/17-actors-isolation-and-sendability.md \
  swift-tutorial/part4-concurrency-performance-and-reliability/18-arc-memory-and-ownership-in-practice.md \
  swift-tutorial/part4-concurrency-performance-and-reliability/19-performance-copying-and-measurement-mindset.md \
  swift-tutorial/part4-concurrency-performance-and-reliability/20-reliability-cancellation-and-failure-surfaces.md
do
  rg -q '^# 第' "$file"
done
git diff --check -- swift-tutorial docs/superpowers
```

Expected:

```text
taskcore-taskcli-ok
```

The targeted file/heading checks should exit successfully. Do not expect `verify_parts.sh` or `verify_projects.sh` to pass yet because later parts and the TaskFlow surface are not written.

- [ ] **Step 4: Commit Part 4**

```bash
git add \
  swift-tutorial/part4-concurrency-performance-and-reliability \
  swift-tutorial/projects/taskcore-taskcli/milestones/part4-runtime-upgrade.md
git commit -F - <<'EOF'
Add the advanced Swift runtime and reliability layer

The tutorial's "high level" claim has to mean more than syntax and packages.
Part 4 upgrades the course into a real modern Swift engineering path by tying
concurrency, ownership, performance, and reliability back to the existing core
and CLI surfaces.

Constraint: Advanced Swift topics must stay project-grounded and understandable to readers who know other languages but are still consolidating Swift instincts
Rejected: Move advanced topics into a detached appendix or short survey chapter | would make the tutorial's upper half feel hollow
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep this part focused on engineering consequences of language/runtime features, not on framework-specific side quests
Tested: bash swift-tutorial/scripts/verify_taskcore_taskcli.sh
Tested: Targeted file and heading checks for Part 4 chapters and the runtime-upgrade milestone
Not-tested: Runtime behavior of hypothetical future code snapshots beyond current project assets
EOF
```

---

### Task 6: Build Parts 5 and 6 and the TaskFlow App Line

**Files:**
- Create: all 4 files under `swift-tutorial/part5-swiftui-foundations/`
- Create: all 4 files under `swift-tutorial/part6-swiftui-dataflow-and-app-architecture/`
- Create: `swift-tutorial/projects/taskflow/README.md`
- Create: `swift-tutorial/projects/taskflow/starter/README.md`
- Create: `swift-tutorial/projects/taskflow/milestones/part5-v1.md`
- Create: `swift-tutorial/projects/taskflow/milestones/part6-architecture.md`
- Create: `swift-tutorial/projects/taskflow/final/README.md`

- [ ] **Step 1: Write the TaskFlow project docs**

The TaskFlow project docs must explain:

- how `TaskFlow` reuses `TaskCore`
- what is starter versus milestone versus final state
- how the SwiftUI app line differs from the CLI line

These docs are descriptive rather than build-verified code assets.
Do not introduce an Xcode project requirement in this task.

- [ ] **Step 2: Write Part 5**

Create:

- `21-swiftui-mental-model-and-view-composition.md`
- `22-state-binding-and-observable-models.md`
- `23-lists-forms-and-navigation-basics.md`
- `24-build-taskflow-v1.md`

These chapters must:

- assume the reader already knows the earlier core line
- introduce SwiftUI through reuse of the shared task domain
- avoid reducing SwiftUI to screenshot-driven prose

- [ ] **Step 3: Write Part 6**

Create:

- `25-app-state-and-data-flow.md`
- `26-persistence-and-model-integration.md`
- `27-async-ui-updates-previews-and-testing.md`
- `28-taskflow-architecture-and-feature-growth.md`

These chapters must:

- show how app architecture grows from the shared core
- teach state/data flow explicitly
- keep SwiftUI writing tutorial-first rather than docs-like

- [ ] **Step 4: Verify Parts 5 and 6**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-tutorial/scripts/verify_projects.sh
for file in \
  swift-tutorial/part5-swiftui-foundations/21-swiftui-mental-model-and-view-composition.md \
  swift-tutorial/part5-swiftui-foundations/22-state-binding-and-observable-models.md \
  swift-tutorial/part5-swiftui-foundations/23-lists-forms-and-navigation-basics.md \
  swift-tutorial/part5-swiftui-foundations/24-build-taskflow-v1.md \
  swift-tutorial/part6-swiftui-dataflow-and-app-architecture/25-app-state-and-data-flow.md \
  swift-tutorial/part6-swiftui-dataflow-and-app-architecture/26-persistence-and-model-integration.md \
  swift-tutorial/part6-swiftui-dataflow-and-app-architecture/27-async-ui-updates-previews-and-testing.md \
  swift-tutorial/part6-swiftui-dataflow-and-app-architecture/28-taskflow-architecture-and-feature-growth.md
do
  test -f "$file"
  rg -q '^# 第' "$file"
done
git diff --check -- swift-tutorial docs/superpowers
```

Expected:

```text
projects-ok
```

The targeted Part 5 and Part 6 checks should exit successfully. Do not expect `verify_parts.sh` to pass yet because Parts 7 and 8 are not written.

- [ ] **Step 5: Commit the SwiftUI specialization path**

```bash
git add \
  swift-tutorial/part5-swiftui-foundations \
  swift-tutorial/part6-swiftui-dataflow-and-app-architecture \
  swift-tutorial/projects/taskflow
git commit -F - <<'EOF'
Add the SwiftUI specialization path and TaskFlow app line

The tutorial now needs to deliver on its second half: Apple development built
on top of the earlier Swift engineering spine. This introduces TaskFlow and the
SwiftUI parts that show how app-side state, UI composition, and architecture
grow from the shared task core.

Constraint: The Apple-development path must feel like a continuation of the Swift language/engineering spine, not a disconnected second course
Rejected: Start SwiftUI with a separate toy app and parallel domain model | would break the continuity that the tutorial relies on
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Preserve the shared-domain continuity between CLI/core and TaskFlow in all later capstone work
Tested: bash swift-tutorial/scripts/verify_projects.sh
Tested: Targeted file and heading checks for the Part 5 and Part 6 chapters
Not-tested: Build execution of a future full TaskFlow app project
EOF
```

---

### Task 7: Author Parts 7 and 8

**Files:**
- Create: all 4 files under `swift-tutorial/part7-advanced-swift-and-system-design/`
- Create: all 4 files under `swift-tutorial/part8-capstone-and-next-steps/`

- [ ] **Step 1: Write Part 7**

Create:

- `29-advanced-generics-and-protocol-design.md`
- `30-result-builders-macros-and-api-surface-judgment.md`
- `31-interop-system-apis-and-package-boundary-tradeoffs.md`
- `32-shared-abstractions-and-system-redesign.md`

These chapters must:

- deepen advanced Swift without breaking tutorial readability
- tie advanced abstractions back to the already-built project lines
- resist "feature parade" writing

- [ ] **Step 2: Write Part 8**

Create:

- `33-capstone-rebuild-plan.md`
- `34-capstone-cli-and-core-hardening.md`
- `35-capstone-taskflow-hardening.md`
- `36-graduation-roadmap-and-next-steps.md`

These chapters must:

- close the course with an actual capstone path
- summarize how the project lines unify
- leave a clear next-steps map rather than a motivational ending

- [ ] **Step 3: Verify Parts 7 and 8**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-tutorial/scripts/verify_parts.sh
git diff --check -- swift-tutorial docs/superpowers
```

Expected:

```text
parts-ok
```

- [ ] **Step 4: Commit Parts 7 and 8**

```bash
git add \
  swift-tutorial/part7-advanced-swift-and-system-design \
  swift-tutorial/part8-capstone-and-next-steps
git commit -F - <<'EOF'
Complete the upper-half Swift tutorial path and capstone finish

The rebuilt tutorial needs a true upper half and a real graduation path. These
parts bring advanced Swift and system design together with a capstone-oriented
ending so the course reads like a complete product instead of stopping at
mid-level engineering.

Constraint: The final parts must still teach through the existing project line instead of wandering into disconnected advanced examples
Rejected: End the tutorial with generic "next steps" prose and no capstone bridge | would make the product feel unfinished
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep the final parts integrated with TaskCore, TaskCLI, and TaskFlow rather than inventing separate advanced demo systems
Tested: bash swift-tutorial/scripts/verify_parts.sh
Tested: git diff --check -- swift-tutorial docs/superpowers
Not-tested: Combined coherence with labs and appendix materials
EOF
```

---

### Task 8: Write Labs, Appendix, and Final Product Checks

**Files:**
- Create: `swift-tutorial/labs/part1-language-foundations.md`
- Create: `swift-tutorial/labs/part2-type-system-and-modeling.md`
- Create: `swift-tutorial/labs/part3-packages-testing-and-cli-engineering.md`
- Create: `swift-tutorial/labs/part4-concurrency-performance-and-reliability.md`
- Create: `swift-tutorial/labs/part5-swiftui-foundations.md`
- Create: `swift-tutorial/labs/part6-swiftui-dataflow-and-app-architecture.md`
- Create: `swift-tutorial/labs/part7-advanced-swift-and-system-design.md`
- Create: `swift-tutorial/labs/part8-capstone.md`
- Create: all 7 files under `swift-tutorial/appendix/`

- [ ] **Step 1: Write all part-level lab files**

Each lab file must include:

- integrated exercises
- debugging tasks
- refactoring/design tasks
- challenge tasks

The lab files must reference the matching part and its project stage by name.

- [ ] **Step 2: Write all appendix files**

Requirements:

`glossary.md`
- bilingual term table
- cover core Swift, SPM, testing, concurrency, and SwiftUI vocabulary

`answers.md`
- provide selected answer guidance
- do not turn every chapter exercise into a full solution dump

`environment-setup.md`
- Swift toolchain/Xcode/SPM setup guidance

`spm-cheatsheet.md`
- everyday SPM commands and target/package patterns

`swiftui-cheatsheet.md`
- common state/view/data-flow reminders for the SwiftUI parts

`faq.md`
- answer migration questions such as:
  - why not start with class
  - why not start with protocol
  - why SwiftUI appears later
  - why the tutorial avoids server-side expansion

`references.md`
- official Swift docs
- Apple docs
- selected external references with short justification

- [ ] **Step 3: Verify the complete product**

Run:

```bash
cd /Users/yangyang/ai_projs/math
bash swift-tutorial/scripts/verify_layout.sh
bash swift-tutorial/scripts/verify_parts.sh
bash swift-tutorial/scripts/verify_projects.sh
bash swift-tutorial/scripts/verify_appendix.sh
bash swift-tutorial/scripts/verify_task_cli_lite.sh
bash swift-tutorial/scripts/verify_taskcore_taskcli.sh
git diff --check -- swift-tutorial docs/superpowers
```

Expected:

```text
layout-ok
parts-ok
projects-ok
appendix-ok
task-cli-lite-ok
taskcore-taskcli-ok
```

- [ ] **Step 4: Commit labs, appendix, and the complete tutorial product**

```bash
git add \
  swift-tutorial/labs \
  swift-tutorial/appendix
git commit -F - <<'EOF'
Finish the Swift tutorial as a complete learning product

The new Swift tutorial is only complete once the mainline chapters are backed
by integrated labs, appendix materials, and final verification. This finishes
the product so it can function as a real repository tutorial rather than a set
of chapter files without support surfaces.

Constraint: The rebuilt tutorial must ship as a full product with labs and appendix materials, not only as the mainline chapters
Rejected: Stop after writing the parts and leave labs/appendix for a later cleanup pass | would undercut the tutorial's completeness claim
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Preserve the relationship between parts, labs, appendix, and project surfaces in future revisions
Tested: bash swift-tutorial/scripts/verify_layout.sh
Tested: bash swift-tutorial/scripts/verify_parts.sh
Tested: bash swift-tutorial/scripts/verify_projects.sh
Tested: bash swift-tutorial/scripts/verify_appendix.sh
Tested: bash swift-tutorial/scripts/verify_task_cli_lite.sh
Tested: bash swift-tutorial/scripts/verify_taskcore_taskcli.sh
Tested: git diff --check -- swift-tutorial docs/superpowers
Not-tested: Human full-course readthrough from beginning to end
EOF
```

## Self-Review Notes

### Spec Coverage

- new product root under `swift-tutorial/`: covered by Task 1
- 8-part architecture: covered by Tasks 2 through 7
- continuous project spine (`TaskCLI Lite -> TaskCore + TaskCLI -> TaskFlow`): covered by Tasks 2, 4, and 6
- labs and appendix: covered by Task 8
- tutorial-first long-form writing: required in Tasks 2 through 7
- strong bilingual mode: required across all chapter-writing tasks and appendix glossary

No major approved design area is left without a task.

### Placeholder Scan

- no `TODO`/`TBD` placeholders are used
- all major created files are named explicitly
- each task includes concrete verification commands and commit boundaries

### Type Consistency

- tutorial root stays `swift-tutorial/` everywhere
- project spine names stay `TaskCLI Lite`, `TaskCore + TaskCLI`, and `TaskFlow`
- the 8 part directory names match the approved design
- labs and appendix naming matches the root contract and verifier assumptions
