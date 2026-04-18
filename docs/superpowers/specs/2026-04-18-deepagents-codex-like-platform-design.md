# Deepagents Codex-Like Enterprise Coding Agent Platform Design

Date: 2026-04-18

## 1. Goal

Design a Codex-like coding agent platform built on top of `deepagents`, but shaped for enterprise use rather than pure consumer convenience.

The target system is:

- enterprise-oriented
- CLI-first
- real-time collaboration first
- SaaS control plane plus customer-side runner
- policy-driven autonomy rather than all-or-nothing approvals
- differentiated by a stronger `deepagents-native runtime`

The core design decision is that the platform should not be centered on chat UX, control-plane workflows, or ad hoc tool glue. It should be centered on a unified runtime contract that makes execution, permissioning, visibility, and replay coherent.

## 2. Product Positioning

This is not just “another coding assistant.”

It is a hybrid execution platform for repository-local coding agents where:

- users collaborate through a terminal-first interface
- execution happens near the repo, tools, and secrets
- organization policy is authored centrally but enforced locally
- audit and recovery are first-class from day one
- customers can extend tools, policy packs, sinks, and subagent archetypes without forking runtime truth

The first target customer is an enterprise engineering organization that wants a hosted governance layer without handing raw repository execution over to a fully hosted SaaS runtime.

## 3. Chosen Direction

Three broad directions were considered:

1. Runtime-first platform
2. Product-first shell
3. Control-plane-first orchestrator

The chosen direction is `Runtime-first platform`.

Why:

- it matches the desired moat: stronger `deepagents-native runtime`
- it keeps visibility, permission, and replay as one system rather than three overlapping subsystems
- it allows the CLI, future IDE integrations, and the control plane to become consumers of runtime truth rather than owners of execution semantics

Rejected alternatives:

- Product-first shell: faster to demo, but it buries the runtime advantage and encourages scattered logic
- Control-plane-first orchestrator: stronger governance early, but too heavy for the desired CLI-first real-time experience

## 4. Scope and Decomposition

The overall product is too large for a single implementation plan. It is intentionally decomposed into four sequential sub-projects:

### P1. Execution Kernel

Build the runtime kernel, runner contract, session model, tool/runtime control, visibility model, local recovery, and replay foundations.

### P2. Terminal Product Surface

Build the CLI experience: live collaboration loop, approvals, patch/diff review, session controls, and human-visible streaming.

### P3. Control Plane

Build the SaaS plane for session registry, policy distribution, metadata ingestion, audit summaries, model/provider configuration, and administrative workflows.

### P4. Enterprise Governance

Build multi-tenant policy management, RBAC, budget/quota controls, compliance integrations, remote execution expansion, and organization-wide observability.

This design document describes the whole product architecture, but the first implementation plan must target `P1`, with only the minimum `P2/P3` hooks needed to avoid rework.

## 5. System Architecture

The platform is split into five major components.

### 5.1 CLI Client

The CLI is the main user entry point.

It owns:

- interactive chat/session UX
- approval prompts
- patch and diff review
- local command surface such as resume, abort, inspect, replay, and status
- live rendering of projected runtime events

It does not own:

- execution semantics
- final policy truth
- replay truth
- audit truth

The CLI is a consumer of runtime projections, not the source of runtime meaning.

### 5.2 Runner

The runner is the trusted executor inside the customer boundary.

It owns:

- repository access
- local workspace state
- shell/tool/filesystem execution
- secret access
- model invocation
- local checkpoint and event storage
- real-time runtime evaluation and enforcement

The runner is intentionally heavy in v1 because execution truth must stay close to the code, the shell, the artifacts, and the secrets.

### 5.3 Runtime Kernel SDK

This is the product core and the main differentiator.

It owns:

- the typed runtime event schema
- policy hook ordering
- visibility projection contracts
- checkpoint and replay contracts
- stable extension points for tools, policy evaluators, event sinks, and subagent archetypes

It is shared across runner, CLI integration, and future product surfaces.

### 5.4 Control Plane

The control plane is hosted and metadata-first by default.

It owns:

- org/workspace/session registry
- policy authoring and versioning
- policy package distribution
- model/provider configuration
- metadata ingestion
- audit summaries and dashboards
- organization admin workflows

It should not become the only source of execution truth.

### 5.5 Plugin / SDK Layer

This is the bounded extensibility surface.

It allows customers to extend the system without redefining core runtime semantics.

## 6. Trust Boundary and Deployment Model

The chosen deployment model is:

- SaaS control plane
- customer-side runner
- first runner target: developer laptop or bastion/jump host

The first release is therefore enterprise-oriented but operationally light enough for a pilot.

By default, the following remain inside the customer execution boundary:

- raw repository content
- raw shell stdout/stderr
- raw tool payloads
- secrets
- full prompt/context assembly
- sensitive artifacts

The control plane receives, by default:

- structured runtime events
- summaries
- metrics
- policy decisions
- audit-safe metadata

This is a metadata-first model, not a raw-data-first model.

## 7. Runtime Kernel Modules

The runtime kernel is intentionally explicit. It should not collapse into one “agent loop” blob.

### 7.1 Event Recorder

Transforms important actions into typed runtime events.

Examples:

- user intent
- agent plan output
- model invocation
- tool call
- shell execution
- filesystem read/write
- approval request
- policy decision
- subagent handoff
- checkpoint commit

### 7.2 Policy Evaluator

Evaluates actions before execution or exposure.

Possible outcomes:

- allow
- deny
- require approval
- redact
- downgrade visibility
- require stronger sandboxing

### 7.3 Visibility Projector

Projects the same event stream into distinct views:

- CLI user view
- parent-agent view
- audit/control-plane view
- local-only debug/replay view

This is the mechanism that keeps “seen by user,” “visible to parent,” and “stored for audit” from collapsing into one concept.

### 7.4 Checkpoint + Replay Manager

Provides session resilience and audit reconstruction.

It stores:

- checkpoint boundaries
- event lineage
- replay cursors
- references to local raw artifacts
- enough semantic context to explain why actions happened

## 8. Execution Pipeline

Every meaningful action follows the same high-level pipeline:

1. user intent or agent action enters the runtime
2. runtime normalizes it into a typed action request
3. policy preflight evaluates whether and how it may execute
4. execution happens in the runner context
5. lifecycle events are emitted
6. visibility projections are produced for each audience
7. checkpoint and metadata-safe persistence occur at stable boundaries

This pipeline is the basis for every tool, model call, shell command, filesystem mutation, and subagent handoff.

## 9. Runtime Event Model

The center of gravity of the whole platform is the typed runtime event model.

Each event should contain, at minimum:

- `event_id`
- `session_id`
- `run_id`
- `parent_event_id`
- `actor`
- `event_type`
- `phase`
- payload tiers
- projection tags

### 9.1 Event Families

The recommended family split is:

- Intent
- Execution
- Policy
- Visibility
- State / Replay
- Audit

This taxonomy is explicit enough to support policy, replay, and extension, but still coarse enough to stay teachable.

### 9.2 Payload Tiers

Each event may have multiple payload forms:

- raw payload
- redacted payload
- summary payload

This is necessary because the platform is metadata-first at the control plane but still needs high-fidelity local execution truth.

### 9.3 Projection Tags

Visibility must be explicit.

Examples:

- user-visible
- parent-visible
- audit-visible
- local-only
- replay-required

No runtime action should become visible “by accident.”

## 10. The Unifying Principle

Visibility, permission, and replay are not separate systems.

They are three projections over the same runtime event stream:

- permission asks whether an event may transition into execution
- visibility asks which audiences may observe the event and at what fidelity
- replay asks which event lineage and state boundaries must be retained for later reconstruction

This is the single most important architectural rule in the design.

If these three concerns are built independently, the platform will drift into conflicting truths.

## 11. Failure Handling, Recovery, and Audit

The platform must assume frequent partial failure.

### 11.1 Failure Classes

Failures are grouped into:

- recoverable execution failures
- policy failures
- state failures
- projection failures

### 11.2 Recovery Posture

The system should:

- fail locally
- record structurally
- resume deterministically

Important rules:

- every important action emits a terminal event
- checkpoint boundaries happen after stable state transitions, not after every token
- control-plane availability loss must not make local sessions unusable
- replay should prefer lineage plus artifact references over raw sensitive payload export

### 11.3 Resume Model

Resume is based on:

- latest stable checkpoint
- subsequent event log
- local artifact references

This allows runner restart, CLI disconnect, or transient infrastructure problems to be survivable instead of catastrophic.

### 11.4 Audit Replay

Audit replay is not the same as byte-for-byte re-simulation.

Its primary job is to answer:

- what happened
- why it happened
- which policy decision enabled or denied it
- what each audience could see

## 12. Policy Model

The product uses policy-driven autonomy.

That means:

- some actions run automatically
- some actions require approval
- some actions are denied
- some actions are allowed but with reduced visibility or stronger sandboxing

### 12.1 Authoring vs Enforcement

Policy authoring belongs in the control plane.

Policy enforcement belongs in or next to the runner.

This split is mandatory. If enforcement moves too far away from execution, approval, replay, and visibility will no longer line up with the actual action boundary.

### 12.2 Policy Inputs

Typical policy dimensions include:

- organization
- repository
- branch
- path/class of file
- tool category
- shell command class
- network target class
- model family
- secret scope
- subagent archetype

### 12.3 Policy Outputs

Policy decisions must be typed, auditable outputs, not hidden plugin behavior.

## 13. Plugin / SDK Model

The platform should be extensible, but bounded.

### 13.1 Open Extension Surfaces

Customers may extend:

- tool adapters
- policy evaluators/rule packs
- event sinks
- model adapters
- subagent archetypes

### 13.2 Closed Core Contracts

Customers may not fork:

- core event semantics
- visibility projection semantics
- checkpoint boundary semantics
- approval semantics

Rule:

Plugins may extend behavior, but they may not fork runtime truth.

## 14. Why Deepagents Is the Right Base

`deepagents` is a strong base because it already has the right conceptual seams:

- graph-based orchestration rather than one opaque agent loop
- tools as runtime surfaces
- subagents as first-class execution structure
- configurable visibility and propagation behavior
- filesystem and memory surfaces that can be made explicit
- policy and middleware hooks that can be elevated into a stronger runtime contract

However, the design requires going beyond stock deepagents usage.

The main additions are:

- a formal runtime event schema
- unified visibility/permission/replay projections
- enterprise-grade local runner semantics
- stable plugin contracts for policy/event/subagent extension
- a metadata-first hosted control plane

## 15. First Implementation Target

The first implementation plan should target:

- `P1 Execution Kernel`
- minimal `P2 CLI surface`
- minimal `P3 control-plane hooks`

Specifically, the first deliverable should establish:

- runner session model
- runtime kernel module boundaries
- typed event schema
- policy evaluation loop
- visibility projection loop
- local checkpoint/resume path
- CLI rendering of approvals, patches, and event projections
- metadata-safe export to a minimal control-plane API

It should not try to fully build:

- complete enterprise admin console
- broad remote worker orchestration
- full multi-tenant governance
- every possible plugin type

## 16. Testing Strategy

Testing must match the architecture.

### 16.1 Kernel Tests

Validate:

- event emission and lineage
- policy decision ordering
- visibility projection correctness
- checkpoint and resume correctness

### 16.2 Runner Integration Tests

Validate:

- shell/tool/filesystem execution under policy
- subagent handoff semantics
- failure recovery after runner restart
- secret boundary and artifact boundary behavior

### 16.3 CLI Contract Tests

Validate:

- approval flows
- patch/diff rendering contract
- replay/status commands
- resilience to stream interruption

### 16.4 Control-Plane Contract Tests

Validate:

- policy package distribution
- metadata ingestion
- summary and audit event shape
- loss of control-plane connectivity without breaking local execution

## 17. Non-Goals for the First Plan

The first plan should explicitly avoid:

- browser-first UX
- heavy central worker-pool architecture
- fully programmable customer-defined runtime semantics
- full raw-data export to SaaS by default
- making the control plane the only recovery backbone

## 18. Summary

The design is intentionally opinionated:

- the runner owns execution truth
- the runtime kernel owns semantic truth
- the control plane owns governance and aggregation
- plugins extend behavior without forking runtime truth
- visibility, permission, and replay are projections over one typed event model

That is what makes this a deepagents-native enterprise coding-agent platform rather than a thin Codex clone with extra admin screens.
