# SwiftUI System Tutorial Design

Date: 2026-04-19
Status: Approved in conversation, written for review
Owner: Codex

## 1. Purpose

Design a new, independent SwiftUI tutorial product that fixes the current SwiftUI coverage problem: the UI material is useful but too bridge-like, too scattered, and not systematic enough for readers who want to learn SwiftUI as a complete system.

The new tutorial should teach SwiftUI as an engineering medium, not as a component glossary and not as a loose set of chapters appended to the existing Swift language path.

## 2. Problem Statement

The current Swift tutorial at `swift-tutorial/` serves a broad Swift learning path. Its SwiftUI coverage is intentionally limited and project-bridged through `TaskFlow`. That makes it useful for teaching:

- SwiftUI mental model
- state and binding basics
- list/form/navigation basics
- app state and async UI basics

It does not create a sufficiently system-level SwiftUI map for readers who want to build serious UI products. The current gaps are structural:

- common UI components are introduced as part of a bridge, not a full component and scene system
- desktop workbench structure is not the primary design target
- advanced SwiftUI capabilities do not have a natural narrative path
- the tutorial does not grow into a real creation tool where layout, gestures, drawing, animation, documents, and platform interop become first-class

## 3. Product Positioning

The new product is not a rewrite of `swift-tutorial/`.

It is a separate tutorial line:

- product type: independent SwiftUI system tutorial
- depth target: from zero to advanced system-level SwiftUI engineering
- teaching style: fully project-driven
- platform focus: Mac-first
- project genre: creative tool
- anchor project: `BoardFlow`, a whiteboard / card-canvas editor

This product must remain separate from the existing Swift language curriculum so both products keep clear identities:

- `swift-tutorial/`: full Swift learning path, SwiftUI as bridge and app entry
- `swiftui-system-tutorial/`: SwiftUI-first, desktop creative-tool system tutorial

## 4. Design Goals

The tutorial must:

1. teach readers how to build real SwiftUI applications, not just compose components
2. establish a system map of SwiftUI concepts, containers, state models, rendering, interaction, and platform boundaries
3. use one continuously growing project so advanced topics have a natural home
4. cover both ordinary app UI and spatial / creative-tool UI
5. reach advanced topics such as `Canvas`, gestures, custom `Layout`, document architecture, undo/redo, and AppKit interop without feeling bolted on
6. stay readable as a tutorial rather than degenerating into API reference material

## 5. Non-Goals

The tutorial will not:

- replace the existing `swift-tutorial/`
- become a complete Apple API encyclopedia
- optimize for iPhone-first patterns as the primary narrative
- teach UIKit as a co-equal path
- mix the `TaskFlow` and `BoardFlow` project lines into one product

## 6. Core Teaching Strategy

The tutorial will use a single project that evolves through real capability jumps.

Project: `BoardFlow`

Project shape:

- starts as a minimal Mac SwiftUI shell
- grows into a structured desktop workbench
- becomes an interactive card canvas
- gains drawing, animation, and custom spatial layout
- becomes a document-based creative tool
- expands into a hybrid SwiftUI + AppKit architecture where needed
- ends as an extensible, testable, performance-aware system

Every chapter must answer five fixed questions:

1. What capability does this chapter add to `BoardFlow`?
2. What SwiftUI mechanism is being taught?
3. How does that mechanism actually work in practice?
4. How does the mechanism generalize beyond `BoardFlow`?
5. What are the common misuse patterns or engineering traps?

This structure prevents the material from turning into scattered UI notes.

## 7. Architecture of the Tutorial

The tutorial is divided into eight parts.

### Part 1: SwiftUI Language and Basic View System

Objective:

- establish what SwiftUI code is actually expressing
- teach the basic visual language before any canvas complexity

Project checkpoint:

- `BoardFlow` minimal Mac app shell

Main topics:

- `App`, `Scene`, `WindowGroup`
- `View`, `body`, `some View`
- `Text`, `Image`, `Button`, `Label`
- `VStack`, `HStack`, `ZStack`
- modifier meaning
- `@State` basics

### Part 2: Components, Navigation, and State Ownership

Objective:

- teach common UI components systematically through real page structure
- establish state ownership discipline

Project checkpoint:

- desktop app skeleton with sidebar, list, detail, and form entry points

Main topics:

- `TextField`, `Toggle`, `Picker`, `Stepper`
- `List`, `Form`, `Section`
- `NavigationStack`, `NavigationSplitView`
- `@Binding`
- `@Observable`, `ObservableObject`
- single source of truth
- derived state

### Part 3: Mac Workbench Architecture

Objective:

- move from normal app UI into a real creative-tool shell

Project checkpoint:

- three-zone workbench with sidebar, work area, inspector, and toolbar

Main topics:

- `Toolbar`
- `overlay`, `background`, `safeAreaInset`
- panel composition
- selection context
- `@Environment`
- focus and scene-level context
- command entry points and keyboard shortcuts

### Part 4: Canvas Space and Gestures

Objective:

- move from page UI into spatial UI

Project checkpoint:

- interactive card canvas with selection, dragging, zooming, and panning

Main topics:

- spatial position versus document state
- coordinates and viewport state
- `GeometryReader`
- `TapGesture`, `DragGesture`, magnification
- gesture composition
- selection model
- identity and hit testing

### Part 5: Drawing, Animation, and Custom Layout

Objective:

- teach advanced rendering and spatial composition

Project checkpoint:

- visual feedback system with guides, connections, transitions, and arrangement logic

Main topics:

- `Shape`
- `Path`
- `Canvas`
- `withAnimation`
- transitions
- transactions
- `matchedGeometryEffect`
- custom `Layout`
- partial redraw judgment

### Part 6: Documents, Persistence, and Edit History

Objective:

- make the project behave like a real creative tool instead of a demo

Project checkpoint:

- document model, save/open, autosave, preferences, undo/redo

Main topics:

- serializable board model
- `FileDocument`
- `@SceneStorage`
- `@AppStorage`
- undo manager
- async save/load
- error recovery

### Part 7: AppKit Interop and Engineering Boundaries

Objective:

- teach how and when to cross SwiftUI boundaries correctly

Project checkpoint:

- hybrid architecture with explicit platform seams

Main topics:

- criteria for using AppKit
- `NSViewRepresentable`
- `NSViewControllerRepresentable`
- drag and drop
- hover, cursor, focus details
- dependency injection
- feature boundaries
- preview and test seams

### Part 8: Performance, Testing, and Extensible Architecture

Objective:

- turn the tutorial code into a maintainable system

Project checkpoint:

- extensible final version of `BoardFlow`

Main topics:

- identity and diffing
- refresh cost reasoning
- performance investigation mindset
- previews and tests
- inspector extension points
- multi-window / multi-document flow
- final architecture consolidation

## 8. Chapter-Level Outline

The part structure above expands into the following chapter map.

### Part 1

1. What SwiftUI App code is actually expressing
2. View composition and the three core layout stacks
3. Fundamental interactive components
4. State-driven UI fundamentals
5. Build the minimal `BoardFlow` home shell

### Part 2

6. Lists, forms, and input contracts
7. `NavigationStack` and `NavigationSplitView`
8. `Binding` and state ownership
9. Observable models and screen-level state coordination
10. Build `BoardFlow v1` desktop skeleton

### Part 3

11. Toolbar, inspector, and workbench structure
12. Environment, focus, and cross-layer context
13. Overlay, background, safe areas, and desktop layering
14. Commands, shortcuts, and desktop interaction entry points
15. Build `BoardFlow v2` workbench

### Part 4

16. From page coordinates to canvas coordinates
17. `GeometryReader` and measurable layout
18. Gesture system and card interaction
19. Zoom, pan, and viewport state
20. Build `BoardFlow v3` interactive canvas

### Part 5

21. Shape, Path, and visual semantics
22. `Canvas` and high-frequency drawing zones
23. Animation as a state transition interpreter
24. `matchedGeometryEffect` and cross-boundary transitions
25. Custom `Layout` and spatial arrangement
26. Build `BoardFlow v4` rendering and arrangement system

### Part 6

27. Board document model and serializable state
28. `FileDocument` and document-based Mac apps
29. `SceneStorage`, `AppStorage`, and preference boundaries
30. Undo/redo and edit history
31. Async loading, autosave, and failure recovery
32. Build `BoardFlow v5` document system

### Part 7

33. When AppKit should and should not be used
34. `NSViewRepresentable` and platform capability bridges
35. Drag and drop, cursor, focus, and desktop-level interaction details
36. Module boundaries, dependency injection, and testable state layers
37. Preview and testing as engineering tools
38. Build `BoardFlow v6` hybrid architecture

### Part 8

39. Identity, diffing, and redraw cost
40. Measurement, debugging, and performance diagnosis
41. Complex tool systems and inspector extension points
42. Multi-window and multi-document workflows
43. Final architecture consolidation
44. `BoardFlow Final` and graduation roadmap

## 9. Directory Structure

The tutorial should be created as a new top-level product:

```text
swiftui-system-tutorial/
  README.md
  00-orientation.md
  01-learning-map.md

  part1-swiftui-language-and-basic-view-system/
  part2-components-navigation-and-state-ownership/
  part3-mac-workbench-architecture/
  part4-canvas-space-and-gestures/
  part5-drawing-animation-and-custom-layout/
  part6-documents-persistence-and-edit-history/
  part7-appkit-interop-and-engineering-boundaries/
  part8-performance-testing-and-extensible-architecture/

  projects/
    boardflow/
      starter/
      checkpoints/
        part1-shell/
        part2-v1-workbench/
        part3-v2-studio/
        part4-v3-canvas/
        part5-v4-rendering/
        part6-v5-document/
        part7-v6-hybrid/
      final/

  appendix/
    component-atlas.md
    layout-playbook.md
    state-ownership-guide.md
    navigation-and-workbench-patterns.md
    gesture-playbook.md
    canvas-and-drawing-guide.md
    animation-guide.md
    mac-interop-guide.md
    performance-and-identity-guide.md
    glossary.md
    faq.md
    references.md

  labs/
    part1.md
    part2.md
    part3.md
    part4.md
    part5.md
    part6.md
    part7.md
    part8.md

  scripts/
    verify_layout.sh
    verify_parts.sh
    verify_boardflow_build.sh
    verify_appendix.sh
```

## 10. Relationship to the Existing Swift Tutorial

The existing `swift-tutorial/` should remain intact as the general Swift product.

Its SwiftUI chapters should be treated as bridge material, not upgraded into the new system tutorial.

Recommended role split:

- `swift-tutorial/`: general Swift path, with SwiftUI introduction and app-building bridge
- `swiftui-system-tutorial/`: dedicated SwiftUI system tutorial, with `BoardFlow` as the main project

This keeps both products coherent.

## 11. Migration Strategy

The new tutorial should not transplant `TaskFlow` content directly. Instead, it should extract reusable concepts from the current SwiftUI chapters and rewrite them in `BoardFlow` terms.

Recommended treatment of current chapters:

- `part5-swiftui-foundations/21-swiftui-mental-model-and-view-composition.md`
  - keep as bridge material in `swift-tutorial/`
  - migrate its core ideas into new tutorial Part 1 through a full rewrite
- `part5-swiftui-foundations/22-state-binding-and-observable-models.md`
  - keep as bridge material
  - expand and rewrite its concepts into the new tutorial Part 2
- `part5-swiftui-foundations/23-lists-forms-and-navigation-basics.md`
  - keep as standard app UI bridge material
  - rewrite concepts into the new tutorial Part 2 with Mac workbench framing
- `part5-swiftui-foundations/24-build-taskflow-v1.md`
  - keep in the existing product
  - do not reuse as the new project line
- `part6-swiftui-dataflow-and-app-architecture/25-app-state-and-data-flow.md`
  - conceptually map into new tutorial Parts 2 and 3
- `part6-swiftui-dataflow-and-app-architecture/26-persistence-and-model-integration.md`
  - conceptually map into new tutorial Part 6
- `part6-swiftui-dataflow-and-app-architecture/27-async-ui-updates-previews-and-testing.md`
  - split concepts into new tutorial Parts 6 and 7
- `part6-swiftui-dataflow-and-app-architecture/28-taskflow-architecture-and-feature-growth.md`
  - conceptually map into new tutorial Parts 7 and 8

## 12. Appendix Strategy

The current `swiftui-cheatsheet.md` is useful, but too compressed to carry the full UI system by itself.

The new tutorial should split SwiftUI reference support into focused appendix documents:

- component atlas
- layout playbook
- state ownership guide
- navigation and workbench patterns
- gesture playbook
- canvas and drawing guide
- animation guide
- Mac interop guide
- performance and identity guide

This gives the tutorial a real support system instead of one overloaded cheat sheet.

## 13. Quality Bar

The finished tutorial should satisfy all of the following:

- the SwiftUI UI layer is taught as a connected system rather than scattered notes
- component usage is organized by role, scene, and state ownership
- advanced capabilities have natural narrative placement
- the project line shows credible growth from shell to professional creative tool
- chapter ordering makes both conceptual and engineering sense
- the product is clearly separate from the existing Swift language tutorial

## 14. Risks and Controls

### Risk: the project line becomes so strong that concept extraction gets weak

Control:

- every chapter must explicitly state the reusable SwiftUI mechanism beyond `BoardFlow`

### Risk: advanced chapters turn into Apple API dumping

Control:

- every advanced topic must be tied to a concrete project need and an engineering judgment

### Risk: Mac-specific depth makes the tutorial inaccessible to general SwiftUI learners

Control:

- each chapter must include generalization notes that explain where the mechanism also applies outside desktop creative tools

### Risk: overlap with `swift-tutorial/` confuses readers

Control:

- preserve product separation and use bridge links instead of merging the products

## 15. Acceptance Criteria for Planning Handoff

This design is ready to move into implementation planning when:

1. the product name, scope, and directory split are accepted
2. the eight-part structure is accepted
3. the `BoardFlow` project line is accepted
4. the migration rule is accepted: concept migration, not direct transplant
5. the next implementation phase is scoped around creating the new tutorial skeleton and prioritized initial parts

## 16. Recommended Next Planning Slice

The first implementation plan should focus on:

1. scaffolding `swiftui-system-tutorial/`
2. writing orientation and learning-map documents
3. authoring Part 1 and Part 2
4. scaffolding `projects/boardflow/starter/`
5. creating appendix shells for the new support system
6. adding tutorial verification scripts for layout and product integrity

This sequence keeps the first delivery focused while still proving the product direction.
