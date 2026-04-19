# Swift Tutorial Systematic Rewrite Design

Date: 2026-04-19

## 1. Purpose

This design replaces the current "scaffold-first" Swift tutorial delivery
standard with a publication-grade tutorial standard.

The immediate problem is not that the current tutorial files are merely short.
The problem is that they were written as structural placeholders: chapter
stubs, project pointers, and authoring rails. That output is acceptable as a
curriculum skeleton, but it is not acceptable as the tutorial itself.

This redesign therefore changes the contract for the next implementation phase:

- Part 1 must become full long-form tutorial content.
- The overall course standard must be rewritten so future parts do not regress
  back into outline-like content.
- Parts 2, 3, and 4 will not be written as full prose yet, but they must be
  upgraded from shallow stubs into detailed expansion blueprints.

## 2. Reader Model

The target reader remains:

- already competent in at least one other programming language
- new to Swift as a language and ecosystem
- willing to learn through projects, exercises, and semantic comparison

This means the tutorial should not spend time teaching basic programming from
scratch. It should instead spend time on:

- Swift-specific semantics
- differences from other mainstream languages
- how Swift design choices affect code shape and engineering decisions

The tutorial is not optimized for absolute beginners with no programming
background. Any zero-programming support, if later desired, should be additive
and clearly secondary to this main path.

## 3. Chosen Rewrite Direction

Three approaches were considered for fixing the current tutorial:

1. expand the existing eight Part 1 chapter stubs in place
2. reset the tutorial quality standard, redesign Part 1, and rewrite it as real
   tutorial prose
3. produce only a few model chapters first and defer the full restructuring

The chosen direction is approach 2.

Why:

- the current weakness is structural as much as it is textual
- expanding weak chapter boundaries would create a larger but still flawed
  tutorial
- the tutorial needs a durable architecture before more long-form writing is
  added

Rejected alternatives:

- in-place expansion of the existing skeleton: fastest path to more words, but
  likely to preserve weak chapter logic and uneven concept ordering
- model chapters only: useful for local quality proof, but insufficient for
  restoring system-level coherence across the course

## 4. New Tutorial Quality Standard

The tutorial must no longer treat chapter files as acceptable merely because
they exist and have section headings.

Each chapter must instead satisfy all of the following:

- teach one clear problem, not just name a topic
- include a continuous explanation from problem framing to working code
- contain complete, connected code examples instead of isolated fragments
- explain semantic reasoning, not only syntax or API names
- include at least one code evolution step, moving from a naive version to a
  better version
- explain common mistakes through cause and effect, not only as bullet-point
  warnings
- end with drills, a checkpoint, and a concrete connection back to the ongoing
  project

The new minimum bar for "chapter written" is:

- a reader can complete the chapter without relying on external material
- the chapter reads like a technical book chapter, not a lecture outline
- the chapter contributes directly to the evolving project spine

The tutorial root documents and shared authoring references must be rewritten to
make this standard explicit and enforceable.

## 5. Teaching Model

The selected teaching model is concept-plus-engineering integration.

That means:

- concepts must be explained clearly before the reader is asked to wield them
- every major concept must then land in code that changes the running project
- the tutorial should move from "what this Swift feature means" to "what this
  changes in the codebase we are building"

The tutorial should avoid two failure modes:

- concept-only exposition that never accumulates into a real program
- project-only coding that assumes the reader already understands Swift's
  semantics

This hybrid model is especially important for the chosen reader profile. A
reader who knows another language can move quickly through mechanics, but still
needs careful explanation of where Swift behaves differently and why.

## 6. Bilingual Delivery Model

The tutorial will be strongly bilingual, but not paragraph-by-paragraph
translation.

The bilingual model is:

- Chinese carries the main explanatory narrative
- English carries code, API names, type names, and canonical technical terms
- key concepts appear in bilingual first-use form, showing the canonical
  English term plus the Chinese translation side by side, without romanization

Required first-use style examples:

- `Value semantics（值语义）`
- `Optional binding（可选值绑定）`
- `Pattern matching（模式匹配）`

Required bilingual rules:

- do not transliterate Chinese terms into romanization
- do not duplicate the full body text in both languages
- keep code, API names, and symbols in English
- use Chinese for explanations, pitfalls, design tradeoffs, and exercise
  guidance
- use a stable glossary so the same concept is not translated differently in
  later chapters
- later use may use whichever side is more natural in context

To avoid future confusion, the authoring files must define this concretely with
real examples rather than prose-only guidance.

Each full Part 1 chapter should also include a short `English Recap` section.
Its purpose is not translation. Its purpose is to provide a concise technical
summary of the chapter's main rules, concepts, and engineering takeaways.

This bilingual model should improve clarity, not double the body length through
parallel prose.

## 7. Part 1 Rewrite Objective

Part 1 must become a complete long-form tutorial, not a set of warmed-over
scaffolds.

It should teach the reader to build a small but real Swift command-line tool
while learning the language's core semantics.

The Part 1 outcome is:

- the reader can read and write small but idiomatic Swift programs
- the reader understands how Swift models values, control flow, optionality, and
  domain data
- the reader can build and extend a small CLI instead of only recognizing terms

The Part 1 project spine remains `TaskCLI Lite`, but it now becomes the central
teaching thread rather than a chapter-end label.

## 8. New Part 1 Chapter Architecture

The current Part 1 chapter structure is too outline-driven and under-emphasizes
Swift's strongest semantic entry points.

The new Part 1 chapter architecture should be:

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

Rationale:

- the tutorial should first ground the reader in how Swift code actually runs
- command-line input and output should appear early so the project feels real
- `struct` must appear in Part 1 because Swift's value-oriented modeling is too
  central to defer
- `enum` and pattern matching should also appear in Part 1 because command
  routing is a natural Swift use case
- `optionals` should be taught inside parsing and safety, not as an isolated
  syntax topic
- the final chapter must be a true integration chapter, not a project brief

The chapter sequence is designed to produce cumulative project evolution:

- Chapters 1-3 establish execution, input, output, and small runnable programs
- Chapters 4-5 turn a script into a shaped program
- Chapters 6-7 turn loose values into domain data and task state
- Chapters 8-9 turn brittle input handling into safe command parsing
- Chapter 10 integrates the full Part 1 result into a coherent `TaskCLI Lite`

## 9. Part 1 Chapter Depth Standard

Part 1 chapters should be long chapters, not short chapter cards.

Each chapter should follow a stable rhythm:

1. frame the concrete problem
2. present a minimal runnable example
3. explain the code and the Swift semantics underneath it
4. improve the design through one or more evolution steps
5. explain common mistakes and why they happen
6. assign three drill types
7. connect the chapter back to `TaskCLI Lite`
8. close with glossary terms and an English recap

Required drill types:

- concept check
- code reading
- hands-on extension

Required structural content in each long-form chapter:

- problem framing
- runnable example
- semantic explanation
- code evolution
- pitfalls and debugging interpretation
- drills
- checkpoint
- glossary
- English recap
- project bridge

The tutorial should especially slow down for the most leverage-heavy Part 1
chapters:

- Structs and Data Modeling
- Optionals and Safe Parsing
- Enums and Pattern Matching
- Build TaskCLI Lite v1

These chapters define whether the learner actually develops a Swift mental model
or only accumulates surface familiarity.

## 10. Project Spine Requirements for Part 1

`TaskCLI Lite` is not optional decoration. It is the pedagogical spine for Part
1.

Its role is to:

- make each language feature do visible work
- prevent Part 1 from degenerating into disconnected syntax lessons
- create a stable codebase that future parts can extend

The project should evolve incrementally:

- first as a tiny runnable program
- then as a small command router
- then as a structured program with functions
- then as a domain model built around `struct Task`
- then as a small command system using enums, collections, and optionals

The starter package remains intentionally small. Part 1 should not front-load
architecture that belongs to Part 2.

## 11. Shared Authoring Asset Rewrite

The shared course assets must be expanded from "basic guardrails" into a real
authoring contract.

This includes:

- the top-level README
- authoring rules
- bilingual style guide
- chapter template
- learning paths
- shared glossary

These files must define:

- what counts as a real chapter
- how code examples should accumulate
- how bilingual terminology is introduced and reused
- how English recap sections should work
- how drills differ from checkpoints
- how projects evolve across the whole tutorial

The tutorial should be hard to degrade accidentally. These files are the main
mechanism for making that possible.

## 12. Parts 2-4 Blueprint Upgrade

Parts 2, 3, and 4 will remain pre-full-prose for now, but they must be far more
than future stubs.

Each part README should become a detailed blueprint that covers:

- the part's purpose and learning outcomes
- the full chapter sequence
- the central problem solved by each chapter
- the project evolution path
- the likely hard points for readers
- the exercise and checkpoint model
- the dependency relationship with earlier parts

Part expectations:

- Part 2: modular Swift engineering, richer type design, testing, package
  structure, and concurrency foundations
- Part 3: SwiftUI and Apple-platform application development built on the same
  task domain
- Part 4: advanced Swift semantics, runtime behavior, performance, advanced
  design, and hardening of both project spines

The blueprints should be detailed enough that future implementation does not
need to redesign the curriculum shape from scratch.

## 13. File and Structure Implications

The root tutorial directory remains:

- `swift-from-zero-to-advanced/`

Part 1 keeps the same part root:

- `swift-from-zero-to-advanced/parts/part-1-swift-fundamentals/`

However, the internal chapter set is allowed to change materially.

This means the implementation may:

- replace the current eight chapter files
- rename chapter files
- add or remove chapter files
- restructure the Part 1 overview, drills, project, checkpoint, and glossary

Preserving low-quality chapter boundaries simply because they already exist is
not a design goal.

## 14. Non-goals

This rewrite does not aim to:

- fully author Parts 2, 3, and 4 as long-form tutorial prose in the same phase
- become a complete Swift reference manual
- add a UIKit-oriented alternate curriculum
- teach programming fundamentals to absolute beginners as the main path
- prematurely expand `TaskCLI Lite` into a Part 2-scale architecture

## 15. Success Criteria

The rewrite is successful if:

- any Part 1 chapter reads like a real technical book chapter
- Part 1 can be studied continuously without feeling like an outline with filler
- `TaskCLI Lite` is a genuine teaching thread rather than a symbolic project
- the shared course assets make the quality bar explicit and repeatable
- Parts 2, 3, and 4 become detailed expansion blueprints rather than vague
  placeholders
- future writing can proceed without re-deciding the course architecture

## 16. Delivery Sequence

The next implementation plan should target three concrete delivery lanes:

1. rewrite the top-level course standard and authoring references
2. rebuild Part 1 as full long-form content under the new chapter architecture
3. upgrade Parts 2, 3, and 4 into detailed blueprints

The implementation should not try to author the entire four-part curriculum in
one pass. The immediate goal is to restore system quality and create one fully
realized part plus three strong blueprints.
