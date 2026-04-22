# Modern Swift App-First Tutorial Design

Date: 2026-04-22
Status: Approved in conversation, written for review
Owner: Codex

## 1. Purpose

Design a brand-new Swift tutorial product that teaches modern Swift through a
cross-platform Apple app from the beginning instead of postponing app work
until after a long language-first lead-in.

The new product should take a reader who already knows another programming
language, but has not learned Swift systematically, and turn that reader into a
modern SwiftUI engineer who can build and maintain a real `iOS + macOS`
product.

## 2. Product Boundary

This tutorial has no continuity with the existing `swift-tutorial/`.

That boundary is strict:

- it does not inherit the old chapter structure
- it does not inherit the old project spine
- it does not inherit the old terminology or pacing model
- it is not a rewrite pass over old material
- it is not a migration that preserves compatibility with the old product

Implementation-wise, the current `swift-tutorial/` will be deleted and replaced
with a new tutorial product that happens to occupy the same repository path.

The design may learn from strong tutorial products elsewhere in the repository,
but it must not treat the current Swift tutorial as source material.

## 3. Reader Profile

The default reader:

- already knows how to program in another language
- has not learned Swift systematically
- wants to build real Apple apps, not just study syntax in isolation
- wants a project-driven path instead of a reference-manual style curriculum

This product is not optimized for:

- complete programming beginners
- UIKit-first or AppKit-first engineers
- server-side Swift as a parallel curriculum
- a mixed "old Apple stack plus modern stack" teaching story

## 4. End State

The target graduate is a modern SwiftUI engineer who can:

- build and evolve a SwiftUI app on both `iOS` and `macOS`
- reason about state ownership, data flow, and feature boundaries
- use modern Swift language features in service of real app engineering
- persist data, handle errors, test behavior, and ship a maintainable product
- introduce modularity and shared core logic when application complexity
  justifies it

The target graduate is not defined as:

- a UIKit/AppKit interoperability specialist
- a server-side Swift engineer
- a language-theory-heavy Swift generalist detached from app delivery

## 5. Product Positioning

This is an app-first Swift tutorial.

Its teaching identity is:

- product type: full Swift tutorial product
- primary platform focus: `iOS + macOS`
- teaching style: project-driven
- narrative style: pure Chinese prose with code/API names kept in English
- technical posture: modern Swift stack first
- project genre: cross-platform productivity app

The product should feel closer to a sustained project tutorial than to a broad
catalog of disconnected Swift topics.

## 6. Design Goals

The tutorial must:

1. bring readers into a working app quickly
2. teach Swift as the language behind product decisions, not as an isolated
   grammar checklist
3. keep one continuous project spine so advanced topics have a natural home
4. treat `iOS + macOS` as the main teaching surface rather than a late bonus
5. introduce shared-core engineering only when application growth makes it
   necessary
6. cover testing, persistence, concurrency, reliability, and shipping
   readiness as first-class engineering work
7. read like a coherent tutorial product, not like a folder of markdown notes

## 7. Non-Goals

The tutorial will not:

- preserve any part of the current `swift-tutorial/`
- optimize for complete beginners with no programming background
- teach UIKit and AppKit as co-equal tracks
- treat server-side Swift as a major branch
- use bilingual parallel prose as the main teaching mode
- front-load CLI engineering as the primary learning surface

## 8. Core Teaching Strategy

The curriculum is built around one continuously evolving app:

- project: `FocusList`

`FocusList` is a modern task and planning app that runs on both `iOS` and
`macOS`.

The key teaching move is app-first sequencing:

- readers start with a real SwiftUI app early
- the app gains features before it gains formal modular architecture
- shared-core engineering appears later, when complexity creates real pressure
- CLI tooling appears only as a supporting engineering surface

This keeps the tutorial aligned with the user's chosen outcome: modern Apple
application engineering, not CLI-first Swift education.

## 9. Product Shape

The tutorial is a complete product, not just mainline chapters.

Its deliverables are:

- `README.md`
- `00-preface.md`
- six part directories of mainline content
- one continuous project line under `projects/focuslist/`
- part-level labs
- appendix materials
- validation scripts

The outer product form stays comparable to strong mature tutorials in this
repository, but the inner narrative is driven by the evolving app.

## 10. Course Architecture

The tutorial uses six parts, with four chapters per part, for a total of
twenty-four mainline chapters.

This gives enough space for a real climb from entry-level Swift to shipping
readiness without turning the project into a fragmented sequence of tiny topic
notes.

### Part 1: App-First Foundations

Purpose:

- get the reader into Swift and SwiftUI through a working app immediately
- establish the smallest stable mental model for Swift, `View`, state, lists,
  forms, and navigation

Project result:

- first usable `FocusList` app

### Part 2: Feature Growth and UI Organization

Purpose:

- grow the app into something that feels product-shaped instead of demo-shaped
- teach component extraction, feature screens, editing flows, grouping, tagging,
  and filtering

Project result:

- stronger `FocusList` feature set and UI organization

### Part 3: Data Modeling, Persistence, and Shared Core

Purpose:

- introduce more deliberate data modeling and persistence
- extract shared app logic into a reusable core only when the product demands it

Project result:

- `FocusCore` introduced as the shared domain layer

### Part 4: Engineering, Testing, and Modularization

Purpose:

- turn the project into an intentional codebase instead of an organically grown
  app folder
- introduce `SwiftPM`, test structure, module boundaries, and a supporting CLI

Project result:

- `FocusList` plus `FocusCore` plus lightweight `focusctl`

### Part 5: Concurrency, Reliability, and Cross-Platform Polish

Purpose:

- teach async behavior, failure handling, bulk operations, search/refresh
  flows, and platform-specific refinements without leaving the modern SwiftUI
  path

Project result:

- more resilient and polished `iOS + macOS` app behavior

### Part 6: Capstone and Shipping Readiness

Purpose:

- consolidate architecture, tests, performance awareness, accessibility,
  previews, and release readiness into a graduation layer

Project result:

- shipping-grade `FocusList` tutorial finale

## 11. Directory Architecture

The new tutorial root will be:

- `swift-tutorial/README.md`
- `swift-tutorial/00-preface.md`
- `swift-tutorial/part1-app-first-foundations/`
- `swift-tutorial/part2-feature-growth-and-ui-organization/`
- `swift-tutorial/part3-data-modeling-persistence-and-shared-core/`
- `swift-tutorial/part4-engineering-testing-and-modularization/`
- `swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish/`
- `swift-tutorial/part6-capstone-and-shipping-readiness/`
- `swift-tutorial/projects/focuslist/`
- `swift-tutorial/labs/`
- `swift-tutorial/appendix/`
- `swift-tutorial/scripts/`

This is not a "preserve old structure where possible" plan. It is the directory
layout of the new product.

## 12. Project-Line Architecture

`projects/focuslist/` holds one continuous app line rather than multiple
unrelated teaching projects.

Recommended structure:

- `starter/`
- `checkpoints/`
- `final/`

This project line exists to show real growth in one application. It should not
degrade into a collection of disconnected samples.

The project spine evolves in three stages:

1. `FocusList` as a mostly single-app target during early feature growth
2. `FocusCore` extracted when modeling, persistence, and testing pressure make
   shared boundaries necessary
3. `focusctl` added as a lightweight supporting surface to demonstrate core
   reuse and engineering clarity, not as the tutorial's main product

## 13. Technical Stack

The tutorial defaults to the modern Swift stack:

- `Swift 6`
- `SwiftUI`
- `Observation`
- `SwiftData`
- `Swift Testing`
- Swift Concurrency (`async/await`, `Task`, actor use where justified)

This is not a "modern stack plus legacy compatibility track" product.

Compatibility notes may exist where necessary, but the main story is firmly
modern.

## 14. Writing Standard

The tutorial uses pure Chinese prose.

English is retained only where it is naturally part of the engineering surface:

- code
- API names
- type names
- protocol names
- command names
- essential system terms

The writing style should be explanation-heavy and engineering-oriented:

- why this capability is needed now
- what it changes in the app
- what mistake patterns the reader should avoid
- what the reader can now build or refactor

The text should not read like an API glossary or a translated reference manual.

## 15. Chapter Template Standard

Each chapter should consistently answer:

1. what problem the app currently has
2. what Swift or SwiftUI capability resolves that pressure
3. how the mechanism works
4. how the project changes because of it
5. what mistakes or over-engineering patterns to avoid
6. what concrete result the reader should have at chapter end

This keeps the tutorial project-driven even while it teaches language and
engineering concepts.

## 16. Detailed Part-Level Capabilities

### Part 1 Capabilities

- toolchain and project setup
- minimal Swift syntax in service of app code
- `View` composition
- state basics
- lists, forms, and navigation
- the first usable `FocusList`

### Part 2 Capabilities

- feature decomposition
- grouping and tagging
- filtering and editing flows
- screen organization
- reusable SwiftUI components
- product-shaped UI structure

### Part 3 Capabilities

- stronger domain modeling
- `SwiftData` persistence
- queries and data access boundaries
- shared-core extraction into `FocusCore`
- reasoning about what belongs in UI versus domain versus storage

### Part 4 Capabilities

- `SwiftPM`
- module boundaries
- `Swift Testing`
- testable design
- supporting CLI surface through `focusctl`
- engineering reuse instead of app-folder sprawl

### Part 5 Capabilities

- async refresh and search
- batch operations
- cancellation and failure surfaces
- reliability and UX feedback during async work
- `iOS` and `macOS` polish differences

### Part 6 Capabilities

- capstone refactoring
- regression hardening
- accessibility
- preview quality
- release readiness checks
- final product consolidation

## 17. Data Flow Strategy

The data-flow strategy is progressive, not over-architected from day one.

Early-stage rule:

- keep the data flow simple enough that the reader can see ownership clearly

That means:

- local UI state stays local
- shared app state is introduced intentionally
- the reader learns where state should live before learning extra layers

In early parts, the dominant flow is:

- user interaction -> view state change -> model update -> UI rerender

In later parts, after the app gains real complexity, the flow becomes more
structured:

- views declare UI and interaction intent
- `FocusCore` owns domain rules and important operations
- the persistence layer owns storage reads, writes, and query boundaries

The tutorial should actively reject both extremes:

- stuffing all logic into views
- inventing deep abstraction stacks before the app needs them

## 18. Error Handling Strategy

Error handling is part of the engineering curriculum, not a late appendix.

The tutorial should explicitly distinguish:

- normal data state
- empty state
- invalid input
- persistence failure
- async loading state
- async failure state

Early chapters should teach local recoverable feedback.

Later chapters should teach how more serious failures are surfaced, where those
errors belong, and how recovery paths differ between UI concerns and system
concerns.

The tutorial should avoid the anti-pattern of flattening all failures into a
generic alert with no ownership model.

## 19. Testing Strategy

Testing evolves with the app:

- early parts test key feature behavior and state changes
- middle parts test `FocusCore`, persistence, and query behavior
- later parts test async paths, error paths, cross-platform regressions, and
  capstone behavior

`Swift Testing` is the default testing framework.

Tests are not decoration. They are part of teaching what stable engineering
looks like as the app gets more ambitious.

## 20. Labs Strategy

`labs/` is part-level integration work, not filler exercises.

Each part-level lab should force the reader to recombine skills from that part
into realistic engineering tasks, such as:

- fixing a broken state flow
- adding a feature across multiple surfaces
- extracting a reusable component
- moving logic into `FocusCore`
- covering a missing regression with tests
- repairing a cross-platform mismatch

Labs validate "can use this in a real project" rather than "can repeat a local
example."

## 21. Appendix Strategy

The appendix should cover supporting materials that help the reader stay in the
project flow without bloating the main chapters.

Recommended appendix contents:

- environment setup
- glossary
- SwiftUI cheatsheet
- `Swift Testing` cheatsheet
- FAQ
- answer index / lab guidance

## 22. Verification Strategy

The tutorial product must be verifiable, not merely written.

Required validation surfaces include:

- directory layout checks
- chapter and appendix completeness checks
- project and checkpoint build checks
- key `FocusList` stage verification
- supporting `FocusCore` / `focusctl` verification where applicable

The finish condition for the product is not "all files exist."

The real finish condition is:

- the curriculum reads coherently
- the project line advances cleanly
- labs reinforce each part
- checkpoints are understandable and buildable
- the product can defend its own quality through scripts and tests

## 23. Success Criteria

The tutorial is successful if it produces a reader experience where:

- Swift is learned through shipping pressure, not detached syntax accumulation
- the app becomes useful early and grows in believable stages
- modularity appears as a justified engineering move, not directory theater
- modern Apple app engineering is the clear center of gravity
- the reader finishes with both product intuition and codebase judgment

At repository level, the product is successful if:

- the old `swift-tutorial/` is fully replaced
- the new `swift-tutorial/` has a clear identity independent from prior Swift
  material
- the product feels comparable in finish quality to the stronger tutorial lines
  elsewhere in the repository

