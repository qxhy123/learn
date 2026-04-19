# Part 2: Swift Core Engineering

## Part Goal

Part 2 is where the learner stops treating `TaskCLI Lite` as a small exercise
and starts treating it as a real Swift codebase that needs structure,
boundaries, tests, and an extension strategy.

The goal is not to jump into app frameworks yet.
The goal is to deepen the reader's command of core Swift so that they can
design types, split modules, test behavior, and introduce concurrency without
losing the clarity established in Part 1.

This part should answer a practical question:

How do you evolve a working Swift command-line tool into a maintainable,
testable, package-based system with clear seams for future app and tooling
clients?

## Learning Outcomes

By the end of Part 2, the learner should be able to:

- explain value vs reference semantics and choose between `struct`, `enum`, and
  `class` based on ownership, mutation, and lifetime needs
- design richer domain models with enums, instance methods, initializers, and
  access control that make invalid states harder to represent
- use protocols and protocol-oriented design to separate capabilities, default
  behavior, and client-facing contracts
- use closures intentionally for transformation, callbacks, injected behavior,
  and asynchronous boundaries
- apply generics where they improve reuse and correctness instead of adding
  abstraction noise
- split code into modules with Swift Package Manager and understand how
  executable, library, and test targets relate
- write XCTest coverage for domain logic, parsing, storage seams, and command
  behavior
- understand concurrency foundations including `async`/`await`, tasks, actor
  isolation, and safe state ownership
- integrate these topics into a larger project shape that prepares for both app
  work and advanced tooling later in the course

## Chapter Sequence

1. **From TaskCLI Lite to Engineered Swift**
   Reframe Part 1's code as a baseline and identify the pain points that appear
   once features, files, and collaborators grow.
2. **Value Semantics, Reference Semantics, and Ownership**
   Compare `struct` and `class` through task models, stores, and service layers.
   Make copying, mutation, and shared state concrete.
3. **Enums, Methods, Initializers, and Access Control**
   Tighten domain modeling with richer command enums, validation initializers,
   helper methods, and explicit API boundaries.
4. **Protocols and Protocol-Oriented Design**
   Introduce storage, rendering, and command-execution contracts. Show where
   protocol extensions help and where concrete types should stay concrete.
5. **Closures as Behavior**
   Use closures in collection pipelines, dependency injection points, sorting,
   filtering, and deferred work. Make escaping and capture semantics visible.
6. **Generics Without Losing the Reader**
   Build reusable parsing and repository helpers, then contrast useful generic
   design with generic code that hides intent.
7. **Swift Package Manager and Module Boundaries**
   Split the codebase into packages and targets, establish public/internal
   boundaries, and explain why module seams matter for scale.
8. **Testing With XCTest**
   Add unit tests, fixture strategies, command tests, and seam-oriented design
   that makes behavior easy to verify.
9. **Concurrency Foundations**
   Introduce asynchronous task loading, actor-backed coordination, and the first
   examples of structured concurrency in a non-UI context.
10. **Integrating the Core System**
    Consolidate the part by turning the code into a layered package layout that
    is stable enough for Part 3's app client and Part 4's advanced extensions.

Each chapter should still feel like a tutorial chapter, not a reference dump.
The reader needs progressive code evolution, not a one-shot architectural leap.

## Project Evolution

The project spine for Part 2 is the transition from `TaskCLI Lite` to a two-part
system:

- `TaskCore`
  The reusable library that owns task models, parsing rules, storage protocols,
  validation, and core concurrency-safe behavior.
- `TaskCLI`
  The executable client that turns command-line input into calls into
  `TaskCore`, handles presentation, and remains thin enough to test and replace.

The evolution should happen in phases:

1. **Stabilize the Part 1 domain**
   Extract the task model, command model, and state transitions into types that
   are explicit about invariants.
2. **Separate policy from interface**
   Move parsing, formatting, and storage rules into `TaskCore`, leaving only the
   command-line shell in the executable target.
3. **Add seams for testing and replacement**
   Introduce protocols for persistence and output so the core logic can be
   tested without shell coupling.
4. **Adopt package boundaries**
   Create a package layout where `TaskCore` can later be imported by SwiftUI and
   by advanced command-line tooling without duplication.
5. **Introduce asynchronous workflows**
   Let loading, saving, and selected commands become asynchronous in a way that
   teaches concurrency foundations without turning the project into a framework
   showcase.

The end of Part 2 should leave the reader with a clean library-plus-client
split, not just a larger single-target CLI.

## Drill and Checkpoint Model

Part 2 drills should shift from syntax recall toward engineering judgment.
Each chapter should include drills from at least three buckets:

- **Concept drills**
  Short prompts about value vs reference tradeoffs, protocol use, closure
  captures, generic constraints, or concurrency ownership rules.
- **Code reading drills**
  Present a partially factored system and ask the reader to explain why a type,
  protocol, or access modifier exists.
- **Extension drills**
  Ask the reader to add a command, swap a storage implementation, refactor a
  generic helper, or add a focused test without breaking architecture.

Checkpoint work for this part should verify that the learner can:

- explain why a given type is a value or reference type
- define and adopt a protocol without over-abstracting
- create a package target boundary that reflects real responsibilities
- write XCTest coverage that protects behavior instead of implementation details
- make a basic asynchronous flow safe and understandable

The part-ending checkpoint should require the learner to ship a working
`TaskCore` plus `TaskCLI` slice with tests, not just answer conceptual
questions.

## Dependencies and Handoffs

Part 2 depends directly on Part 1 and should assume the learner already knows:

- basic Swift syntax and program execution
- functions, structs, collections, optionals, and enums
- how `TaskCLI Lite` currently works as a small single-program baseline

Part 2 hands off the following assets to later parts:

- `TaskCore` as the shared foundation for app and tooling clients
- `TaskCLI` as the engineered command-line surface that can grow into
  `TaskCLI Pro`
- testing habits and module boundaries that make future changes safer
- concurrency foundations that Part 4 can deepen without re-teaching from zero

Part 3 should inherit the core domain and storage seams so SwiftUI lessons can
focus on app state, UI data flow, and Apple-platform concerns instead of
re-litigating domain architecture.

Part 4 should inherit both the module structure and the semantic foundations so
advanced topics like ARC, optimization, and advanced protocol design are taught
against a codebase the reader already understands.
