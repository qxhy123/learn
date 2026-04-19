# Swift Tutorial Rebuild Design

## Goal

Rebuild the deleted Swift tutorial as a new tutorial product under
`swift-tutorial/`, using the same overall quality bar as the stronger tutorial
projects elsewhere in this repository instead of the earlier blueprint-heavy
Swift draft.

The rebuilt tutorial must be:

- from zero to advanced
- long-form and tutorial-first
- strongly bilingual
- project-driven
- complete as a tutorial product, not only as a scaffold

## Reader Profile

The default reader already knows how to program in another language but has not
learned Swift systematically before.

This means the tutorial should:

- skip "what is a variable" style teaching
- explain Swift-specific semantics carefully
- actively compare Swift instincts with likely prior-language instincts
- teach language features through engineering pressure rather than isolated syntax

The tutorial may still remain usable for readers who know a little Swift but
feel unsystematic, but they are a secondary audience.

## Core Teaching Direction

The tutorial's long-range direction is:

1. teach Swift language and engineering fundamentals first
2. teach Apple development as a later specialization
3. do not expand into server-side Swift as a full third teaching axis

In practical terms, this means:

- the main spine is Swift language + engineering
- SwiftUI and Apple-platform development appear later as a deliberate handoff
- "high level" means types, value/reference reasoning, protocol/generic design,
  package engineering, testing, concurrency, reliability, performance, and then
  app architecture

## Product Shape

The rebuilt tutorial should use a mixed product form:

- tutorial-first long-form prose
- a continuous engineering project spine
- dedicated projects, labs, and appendix materials

This is intentionally closer to the stronger tutorials in this repository such
as `cuda-tutorial`, `git-tutorial`, `python-tutorial`, and
`computer-network-tutorial`, which provide a clear root map, a preface, major
parts, and supporting materials instead of a loose markdown pile.

## Directory Architecture

The tutorial lives at:

- `swift-tutorial/`

Its root structure should be:

- `swift-tutorial/README.md`
- `swift-tutorial/00-preface.md`
- `swift-tutorial/part1-language-foundations/`
- `swift-tutorial/part2-type-system-and-modeling/`
- `swift-tutorial/part3-packages-testing-and-cli-engineering/`
- `swift-tutorial/part4-concurrency-performance-and-reliability/`
- `swift-tutorial/part5-swiftui-foundations/`
- `swift-tutorial/part6-swiftui-dataflow-and-app-architecture/`
- `swift-tutorial/part7-advanced-swift-and-system-design/`
- `swift-tutorial/part8-capstone-and-next-steps/`
- `swift-tutorial/projects/`
- `swift-tutorial/labs/`
- `swift-tutorial/appendix/`

### Root File Responsibilities

`README.md` must provide:

- who the tutorial is for
- what "from zero to advanced" means here
- the 8-part course map
- the project spine
- how to use the tutorial

`00-preface.md` must provide:

- how to study the tutorial
- environment/setup expectations
- terminology and bilingual rules
- how the project line evolves
- what the tutorial deliberately does not try to do

Each `partN-*` directory is long-form mainline content.

`projects/` contains the evolving code/project artifacts.

`labs/` contains part-level integrated exercises, separate from chapter drills.

`appendix/` contains glossary, answers, FAQ, setup references, and cheatsheets.

## 8-Part Course Architecture

### Part 1: Language Foundations

Purpose:

- build the first stable Swift mental model
- move from toolchain basics to a small real CLI

Coverage includes:

- toolchain basics
- `swift` and `swiftc`
- basic values and mutability
- strings and collections
- control flow and functions
- structs, enums, optionals
- the first integrated CLI

Project output:

- `TaskCLI Lite v1`

### Part 2: Type System and Modeling

Purpose:

- upgrade from "can write Swift" to "can design Swift models and APIs"

Coverage includes:

- methods and properties
- initializers
- classes vs structs
- value vs reference semantics
- protocols and protocol extensions
- generics
- associated types at the appropriate beginner-to-intermediate level
- error handling and basic API design

Project output:

- a more deliberately modeled CLI codebase

### Part 3: Packages, Testing, and CLI Engineering

Purpose:

- turn the codebase into a real Swift project

Coverage includes:

- Swift Package Manager
- target/module boundaries
- XCTest
- CLI command organization
- parsing, rendering, and storage seams
- testability and maintainability

Project output:

- `TaskCore + TaskCLI`

### Part 4: Concurrency, Performance, and Reliability

Purpose:

- deepen Swift into modern engineering territory

Coverage includes:

- `async` / `await`
- `Task`
- actors
- sendability/isolation basics
- ARC/memory reasoning from an engineering perspective
- performance hotspots and copying cost
- cancellation, reliability, and failure surfaces

Project output:

- stronger `TaskCore + TaskCLI` runtime behavior and engineering discipline

### Part 5: SwiftUI Foundations

Purpose:

- start the Apple-development specialization without throwing away earlier
  engineering boundaries

Coverage includes:

- SwiftUI mental model
- view composition
- state, binding, observable models
- basic lists/forms/navigation

Project output:

- `TaskFlow v1`

### Part 6: SwiftUI Data Flow and App Architecture

Purpose:

- move from "can write SwiftUI" to "can structure an app"

Coverage includes:

- app state/data flow
- persistence
- navigation architecture
- async UI updates
- previews
- testing and accessibility baseline

Project output:

- a more structured `TaskFlow`

### Part 7: Advanced Swift and System Design

Purpose:

- consolidate advanced Swift topics and system design judgment

Coverage includes:

- advanced generics
- protocol design tradeoffs
- macros/result builders in their proper place
- interop/system APIs
- deeper boundary design across packages and clients

Project output:

- more systemically designed shared abstractions across the project spine

### Part 8: Capstone and Next Steps

Purpose:

- produce an actual graduation layer instead of only a closing summary

Coverage includes:

- capstone refactoring
- integrated labs
- architecture review
- performance/reliability/UI/package unification
- follow-up study roadmap

Project output:

- a capstone-quality multi-surface Swift learning artifact

## Project Spine

The course is organized around one stable domain:

- task management

The continuous project line is:

1. `TaskCLI Lite`
2. `TaskCore + TaskCLI`
3. `TaskFlow`
4. a capstone unification phase

### Phase 1: TaskCLI Lite

Part 1 uses a small CLI to ground fundamentals.
The point is not large functionality.
The point is to give early language features a real landing surface.

The minimum capability set is:

- `list`
- `add <title>`
- `done <title>`

### Phase 2: TaskCore + TaskCLI

Parts 2 through 4 evolve the CLI into a real package-structured codebase.

`TaskCore` owns:

- domain model
- state transformations
- parsing/storage/service seams where appropriate

`TaskCLI` owns:

- command-line entry
- presentation
- orchestration over the core

### Phase 3: TaskFlow

Parts 5 and 6 create a SwiftUI app client that reuses the core instead of
duplicating domain logic.

`TaskFlow` must feel like a real client of `TaskCore`, not a parallel tutorial
project with different names and concepts.

### Phase 4: Capstone Unification

Parts 7 and 8 unify:

- `TaskCore`
- `TaskCLI`
- `TaskFlow`

The capstone must demonstrate:

- shared domain continuity
- reusable abstractions
- stronger system design choices
- coherent runtime and UI reasoning

## Projects Layout

The `projects/` directory should contain:

- `projects/task-cli-lite/`
- `projects/taskcore-taskcli/`
- `projects/taskflow/`

Each project directory should include:

- `README.md`
- `starter/`
- `milestones/`
- `final/`

This is important for pedagogy.
Readers should be able to see:

- the starting point
- the staged evolution
- the final state

The tutorial must not force the reader to reverse-engineer everything from one
final code snapshot.

## Labs Layout

`labs/` should be organized by part:

- `labs/part1-language-foundations.md`
- `labs/part2-type-system-and-modeling.md`
- `labs/part3-packages-testing-and-cli-engineering.md`
- `labs/part4-concurrency-performance-and-reliability.md`
- `labs/part5-swiftui-foundations.md`
- `labs/part6-swiftui-dataflow-and-app-architecture.md`
- `labs/part7-advanced-swift-and-system-design.md`
- `labs/part8-capstone.md`

Each lab should contain:

- integrated exercises
- debugging exercises
- refactoring exercises
- design questions
- optional challenge tasks

Labs are not chapter drills.
They are part-level synthesis checkpoints.

## Appendix Layout

The appendix must at least contain:

- `appendix/glossary.md`
- `appendix/answers.md`
- `appendix/environment-setup.md`
- `appendix/spm-cheatsheet.md`
- `appendix/swiftui-cheatsheet.md`
- `appendix/faq.md`
- `appendix/references.md`

Responsibilities:

- `glossary.md`: bilingual term control
- `answers.md`: selected answers and lab guidance, not full spoon-feeding
- `environment-setup.md`: runtime and tooling setup
- cheatsheets: practical quick reference
- `faq.md`: recurring conceptual objections and migration confusions
- `references.md`: official docs plus selected high-value external references

## Chapter Writing Contract

Every chapter must be a full tutorial unit, not a topic summary.

### Required Chapter Moves

Each chapter must include:

1. why this problem appears now
2. how it connects to the previous chapter
3. what part of the project is currently weak
4. the current weak version or current starting state
5. the stronger version
6. explanation of why the stronger version is stronger
7. concrete pitfalls or bad smells
8. a local chapter summary
9. drills or mini exercises
10. explicit project handoff

### Required Didactic Shape

Each chapter should have the following internal logic:

- opening problem
- runnable or inspectable starting point
- concept explanation
- code evolution
- engineering meaning
- common mistakes
- summary/checkpoint
- drills
- project bridge

The tutorial must not skip straight to the final solution.
The reader must be able to see:

- the previous state
- the pressure that makes change necessary
- the code/design improvement

### Engineering Requirement

Chapters must not teach Swift as detached syntax.
Each important concept must be tied back to:

- file shape
- model quality
- data flow
- maintainability
- testability
- or app/runtime architecture later in the course

### Bilingual Requirement

This tutorial uses a strong bilingual mode:

- the main explanation language is Chinese
- key Swift terms appear with English systematically
- each chapter includes a compact English recap
- appendix glossary terms must stay consistent with chapter wording

### Anti-Slop Rules

The tutorial must avoid:

- chapter-as-outline writing
- empty motivational filler
- over-eager "advanced" abstractions too early
- reference-manual tone replacing tutorial tone
- definitions without tradeoffs or code consequences

## Completion Standard

The rebuilt Swift tutorial is complete only if it functions as a real tutorial
product, not just a structured repository.

This means it must provide:

- a coherent root map
- a proper preface
- all 8 parts in written form
- a continuous project line
- projects, labs, and appendix materials
- enough long-form explanation that the reader can genuinely study from it

The rebuilt product fails the design if it collapses back into:

- stubs
- skeletons
- blueprint-only content
- or root docs that promise a tutorial without delivering one

## Repository Constraints

The rebuild should create a new `swift-tutorial/` tree.

It should not depend on reviving the previously deleted
`swift-from-zero-to-advanced/` tree.

It should follow the quality patterns of stronger tutorial directories already
present in this repository.

## Intended Execution Scope

This is a full-product build, not a partial scaffold pass.

The expected first delivery includes:

- the full directory skeleton
- all 8 parts
- projects
- labs
- appendix

The implementation may use phased authoring internally, but the delivered
result should read as one coherent tutorial product.
