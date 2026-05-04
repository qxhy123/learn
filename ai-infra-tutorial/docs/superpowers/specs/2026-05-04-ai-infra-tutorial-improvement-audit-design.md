# AI Infra Tutorial Improvement Audit Design

> Date: 2026-05-04
> Scope: current working-tree Markdown source for the AI Infra tutorial
> Primary output: actionable tutorial improvement spec

## 1. Goal

Review the tutorial from the perspective of a senior AI infrastructure engineer and produce an actionable improvement spec. The review should identify unreasonable structure, missing material, shallow treatment, and places where the tutorial says the right words but does not yet teach the engineering judgment behind them.

The final output should emphasize a practical modification blueprint, not a long free-form review report. Per-chapter reading is still required, but its findings should be compressed into prioritized rewrite actions.

## 2. Source Of Truth

Use the current working-tree Markdown source as the review target, including uncommitted changes:

- `README.md`
- `00-preface.md`
- `part0-foundations-of-systems/`
- `part1-foundations/`
- `part2-systems-stack/`
- `part3-training-infra/`
- `part4-data-and-storage/`
- `part5-serving-infra/`
- `part6-platform-and-orchestration/`
- `part7-reliability-security/`
- `part8-advanced-and-capstone/`
- `appendix/`

Exclude generated HTML from the core audit. Use `html/` only as publishing context if a Markdown/HTML mismatch affects the improvement plan. Use existing `docs/superpowers/` specs and plans only as background, not as the review target.

## 3. Review Lens

Each chapter should be evaluated against the standard expected from a production AI infra engineer:

1. **Positioning**: the chapter's role in the full AI infra capability map is explicit.
2. **Mechanism depth**: concepts are explained through constraints, data paths, control paths, state, and failure modes.
3. **Engineering closure**: advice maps to metrics, commands, configs, admission checks, SOPs, rollback rules, or acceptance criteria.
4. **Capacity and cost reasoning**: major design choices include sizing formulas, resource budgets, utilization/goodput trade-offs, or cost implications when relevant.
5. **Reliability and recovery**: long-running training, serving, data, registry, and platform workflows include failure semantics and recovery protocols.
6. **Evidence quality**: worked examples contain numbers, symptoms, observations, diagnosis, decision points, and verification.
7. **Boundaries**: the chapter distinguishes what a technology is, what it is not, when it fails, and what adjacent layer owns the problem.
8. **Teaching quality**: first-principles sections build usable intuition instead of becoming generic motivation.

## 4. Final Spec Structure

The improvement spec should contain these sections.

### 4.1 Overall Diagnosis

Summarize the tutorial's current state:

- strongest parts that should be preserved
- structural weaknesses across the whole tutorial
- repeated shallow patterns, such as tool lists without decision criteria
- missing cross-cutting engineering lines
- places where chapter granularity or ordering hurts comprehension

### 4.2 Improvement Principles

Define what every strengthened chapter must provide:

- clear "what / not what / adjacent boundary" treatment
- at least one real production path: control path, data path, state path, or failure path
- operational evidence: metrics, logs, traces, commands, configs, or events
- one capacity, performance, reliability, or cost model where the topic naturally requires it
- concrete failure modes and recovery or rollback behavior
- worked example with numbers and a verification loop

Also define disallowed weak patterns:

- "should / usually / best practice" without a decision rule
- listing tools without selection criteria
- performance claims without workload shape, hardware context, or benchmark boundary
- architecture diagrams without state ownership or failure behavior
- exercises that ask for definitions only when the chapter claims to teach engineering design

### 4.3 Part-Level Blueprint

For Part 0 through Part 8 and the appendix, state:

- what should be kept
- what should be reinforced
- what should be trimmed or merged
- what capability the reader should have after completing that part
- which cross-part dependencies must be made explicit

### 4.4 Chapter-Level Actions

For each chapter, list only the highest-value actions, normally 3-6 items:

- action title
- reason
- concrete content to add or rewrite
- expected acceptance signal

Actions must be written as implementable rewrite tasks, not vague comments. For example:

- Good: "Add a 70B serving capacity worked example covering weight memory, KV cache, TTFT/TPOT, batch shape, and rollback threshold."
- Bad: "Serving chapter should be deeper."

### 4.5 Cross-Cutting Capability Lines

Identify horizontal threads that should run through many chapters:

- capacity planning
- performance diagnosis
- reliability and recovery
- multi-tenancy, fairness, and cost
- security, supply chain, and governance
- data and artifact lineage
- evaluation and release safety
- first-principles-to-SOP translation

For each line, define which chapters should carry it and what artifact should make it visible: formulas, checklists, runbooks, diagrams, examples, or appendix tables.

### 4.6 Priority And Roadmap

Classify recommendations:

- **P0**: gaps that make the tutorial materially less credible for senior AI infra readers.
- **P1**: improvements that significantly increase engineering usefulness.
- **P2**: polish, consistency, appendix strengthening, and optional deeper references.

The roadmap should be implementable in waves. Each wave should group chapters by shared theme and avoid mixing unrelated rewrites in one task.

### 4.7 Acceptance Criteria

The spec should define how to check that a rewrite is actually stronger:

- chapter has boundary section and failure-mode section
- chapter has at least one numerical model or explicit decision rule when relevant
- worked example includes evidence, diagnosis, action, and verification
- chapter has no unresolved `TODO`, `TBD`, or vague placeholder wording
- cross-references resolve to existing Markdown files
- exercises and appendix answers match rewritten content

## 5. Review Process

The audit should proceed in five passes:

1. **Inventory pass**: map chapter lengths, headings, examples, checklists, and exercises.
2. **Depth pass**: read chapters for mechanism, engineering closure, and weak generic phrasing.
3. **Gap pass**: compare coverage against senior AI infra expectations across training, serving, data, platform, reliability, security, and cost.
4. **Compression pass**: convert raw findings into part-level and chapter-level rewrite actions.
5. **Roadmap pass**: prioritize P0/P1/P2 and group the work into implementation waves.

## 6. Non-Goals

This audit does not rewrite the tutorial body.

This audit does not use generated HTML as the source of truth.

This audit does not aim to produce exhaustive line-by-line copy editing. Language polish should appear only when it affects technical clarity.

This audit does not require current external web research unless a factual claim is time-sensitive or likely stale. The main judgment is architectural and pedagogical.

## 7. Deliverable

After approval, create a new improvement spec under `docs/superpowers/specs/` with a filename based on the audit topic and current date. The spec should be self-contained enough to drive a later implementation plan.

The expected result is a document that a follow-up planning session can turn into concrete rewrite tasks for the Markdown tutorial.
