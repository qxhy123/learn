# AI Infra Tutorial Improvement Spec

> Date: 2026-05-04
> Source of truth: current working-tree Markdown source
> Output focus: actionable rewrite and strengthening blueprint

## 1. Overall Diagnosis

**Strengths to preserve**

- The tutorial has the right positioning: it tries to teach AI infrastructure as a system of resource, state, reliability, cost, and platform decisions rather than as a catalog of GPU, Kubernetes, vLLM, and vector database terms.
- The strongest chapters already show senior-engineer texture. Examples include `part0-foundations-of-systems/0a8-cpu-worked-example.md`, `part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md`, `part2-systems-stack/06d-profiling-debugging-and-performance-sop.md`, `part3-training-infra/10-memory-checkpointing-and-recovery.md`, `part4-data-and-storage/11d-streaming-and-dataloader-engineering.md`, `part4-data-and-storage/12b-checkpoint-engineering.md`, `part5-serving-infra/15-batching-scheduling-and-kv-cache.md`, and `part6-platform-and-orchestration/20d-capacity-and-troubleshooting-sop.md`; these often contain first-principles framing, metrics, commands, worked examples, checklists, and production failure modes.
- The split into overview chapters plus deep-dive subchapters is directionally correct. It lets the tutorial serve both readers who need a map and readers who need operational depth.
- Several chapters already model the right teaching style: mechanism -> path -> evidence -> decision -> validation. These should become the standard contract for the whole tutorial.

**Structural weaknesses**

- The tutorial lacks a small set of shared cross-cutting contracts. Evidence packages recur in `00-preface.md`, `part0-foundations-of-systems/0a8-cpu-worked-example.md`, and `part7-reliability-security/21-observability-and-capacity.md`; release units recur in `part4-data-and-storage/12a-model-registry.md`, `part4-data-and-storage/12c-release-governance.md`, and `part5-serving-infra/14-online-inference-architecture.md`; capacity ledgers recur in `part3-training-infra/07-single-node-training.md`, `part5-serving-infra/17-multitenancy-and-cost.md`, and `part6-platform-and-orchestration/20d-capacity-and-troubleshooting-sop.md`, but these are not defined once and reused consistently.
- Overview chapters are uneven. `part2-systems-stack/04-gpu-and-accelerators.md`, `part2-systems-stack/05-memory-interconnect-io.md`, `part2-systems-stack/06-cuda-runtime-and-kernels.md`, `part6-platform-and-orchestration/18-containers-and-runtime.md`, `part6-platform-and-orchestration/19-kubernetes-for-ai.md`, and `part6-platform-and-orchestration/20-queues-quotas-and-autoscaling.md` should explicitly state what artifact the reader can produce after each chapter group.
- Some sections repeat "first-principles" motivation, checklist patterns, and generic warnings without turning them into measurable gates, schemas, runbooks, or decision records; this is most visible in `part7-reliability-security/21-observability-and-capacity.md`, `part7-reliability-security/22-evaluation-release-and-incident.md`, and `part7-reliability-security/23-security-isolation-and-governance.md`.
- Part 7 and Part 8 are materially thinner than Part 5/6. For a senior AI Infra tutorial, `part7-reliability-security/21-observability-and-capacity.md`, `part7-reliability-security/22-evaluation-release-and-incident.md`, `part7-reliability-security/23-security-isolation-and-governance.md`, `part8-advanced-and-capstone/24-build-an-ai-platform.md`, and `part8-advanced-and-capstone/25-agent-and-inference-time-compute.md` need production schemas and drills, not just conceptual summaries.
- Markdown source integrity needs attention. Part 0 overview and split chapters still contain generated `.html` links, including `part0-foundations-of-systems/0c-filesystems-and-storage-internals.md`, `part0-foundations-of-systems/0c1-vfs-inode-dentry-and-block-layer.md`, `part0-foundations-of-systems/0c2-local-filesystems-ext4-xfs-zfs.md`, `part0-foundations-of-systems/0c3-storage-semantics-fsync-direct-io-and-checkpoints.md`, `part0-foundations-of-systems/0c4-object-storage-parallel-filesystems-and-dataset-io.md`, `part0-foundations-of-systems/0d-network-stack-fundamentals.md`, `part0-foundations-of-systems/0d1-linux-network-stack-tcp-and-mtu.md`, `part0-foundations-of-systems/0d2-nic-offload-queues-and-service-network-io.md`, `part0-foundations-of-systems/0d3-rdma-roce-infiniband-and-gpudirect.md`, `part0-foundations-of-systems/0d3a-rdma-verbs-memory-registration-and-queues.md`, `part0-foundations-of-systems/0d3b-roce-infiniband-lossless-fabric-and-congestion.md`, `part0-foundations-of-systems/0d3c-gpudirect-rdma-gpu-nic-topology-and-diagnostics.md`, and `part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md`. Part 4's more important source-integrity risk is high-risk numeric and version-sensitive material in vector, checkpoint, registry, and supply-chain chapters that needs assumption/source labels.

**Repeated shallow patterns**

- Advice often says "should", "usually", or "best practice" but lacks a decision rule, owner, threshold, evidence field, rollback condition, or validation command.
- Worked examples often have plausible stories but inconsistent reproducibility metadata: hardware, model, versions, input distributions, sampling window, benchmark command, baseline, counterfactual, and retest criteria are not always present.
- Capacity and cost appear across the tutorial, but formulas and field names are inconsistent. Readers cannot yet carry one `CapacityLedger` from training to serving to quota to cost governance.
- Security, ACL, cache invalidation, release rollback, and lineage appear in multiple chapters but are not carried as invariants across data, model, index, cache, serving, and agent tool use.
- Tool maps and version-sensitive claims are useful but need freshness notes and source/assumption labels to avoid stale guidance.

**Highest-impact gaps**

- Define and reuse shared contracts: `EvidenceBundle`, `CapacityLedger`, `ReleaseUnit`, `StateManifest`, `TenantBudget`, `RestoreLevel`, and `CacheKeyContract`.
- Standardize numeric claims. Every benchmark, hardware spec, cost estimate, or throughput threshold should be labeled as measured, illustrative, assumption-based, vendor-public, or version-sensitive.
- Convert Part 7 into executable production guidance: observability schema, SLO burn-rate math, release state machine, rollback drill, incident taxonomy, threat model, and policy-as-code examples.
- Convert Part 8 into a true capstone: platform API objects, state machines, budget envelope, agent runtime contract, and a grading rubric.
- Repair Markdown source links and stale anchors before deeper rewrites, so future changes do not compound documentation drift.

## 2. Improvement Principles

Every strengthened chapter must provide the following:

1. **Boundary**: what the topic is, what it is not, which adjacent layer owns nearby problems, and when this chapter should hand off to another chapter.
2. **Path**: at least one concrete control path, data path, state path, or failure path.
3. **Evidence**: metrics, logs, traces, commands, configs, events, admission records, or dashboard fields that prove or falsify a hypothesis.
4. **Model**: a capacity, performance, reliability, or cost model when the topic naturally requires one.
5. **Failure behavior**: symptoms, likely root causes, first actions, mitigations, rollback or recovery behavior, and escalation boundary.
6. **Worked example**: numbers, assumptions, baseline, hypothesis, evidence, change, retest, and rollback condition.
7. **Reader artifact**: a concrete thing the reader can produce after the chapter, such as a runbook, sizing sheet, release decision record, checkpoint manifest, placement map, or security policy.

Weak patterns to remove or rewrite:

- Tool lists without selection criteria.
- Performance claims without workload shape, hardware, version, and benchmark boundary.
- Checklists that only ask "whether" without owner, phase, evidence, threshold, and action.
- Architecture diagrams without state ownership and failure behavior.
- Exercises that only ask definitions where the chapter claims to teach engineering design.
- Version-sensitive claims without source date, supported version range, or freshness note.

## 3. Part-Level Blueprint

### 3.1 Part 0: Foundations Of Systems

**Keep**

- Keep the strong mechanism-to-evidence style in CPU, memory, RDMA, GPUDirect, and NCCL chapters.
- Keep the deep dive split: CPU, memory/IO, storage, and network are real prerequisites for AI Infra diagnosis.

**Reinforce**

- Add a Part 0 evidence package contract: CPU, memory/IO, storage, and network should each produce a standard diagnostic bundle.
- Strengthen storage chapters to match the depth of CPU/RDMA chapters: kernel path, block layer evidence, fio methodology, crash drills, object-store request cost, and recovery validation.
- Add hardware and version boundaries for counters, NIC/driver/firmware behavior, perf events, and public hardware specs.

**Trim Or Merge**

- Reduce repeated checklist and SOP wording where a shared Part 0 evidence package can carry the common structure.
- Keep overview chapters as maps, but require them to state reader deliverables and source links using Markdown paths.

**Reader Capability After This Part**

- Reader can build evidence chains for host CPU stalls, memory/TLB/Page Cache issues, NUMA/PCIe/H2D problems, checkpoint storage semantics, object-store or parallel filesystem bottlenecks, TCP/NIC/RDMA/GDRDMA/NCCL failures.

**Required Cross-Part Links**

- Link Part 0 storage semantics to Part 4/12 checkpoint and artifact management.
- Link Part 0 memory/PCIe/GDRDMA to Part 2 IO, Part 3 training, and Part 5 serving.
- Link Part 0 NCCL/fabric diagnostics to Part 3 data parallel, Part 6 scheduling/topology, and Part 7 observability.

### 3.2 Part 1: AI Infra Foundations

**Keep**

- Keep the broad AI Infra framing, resource/data/governance lines, and model-to-production lifecycle.

**Reinforce**

- Make Part 1 a compact contract for the rest of the tutorial: what counts as evidence, what a minimum production loop contains, and how resource/cost/reliability decisions are made.
- Add a minimal model release evidence packet and model release decision record.

**Trim Or Merge**

- Reduce overlap between Chapter 1 and Chapter 3. Chapter 1 should define the field and mental model; Chapter 3 should become an execution-oriented productionization chapter.

**Reader Capability After This Part**

- Reader can diagnose whether an AI project is missing data, training, evaluation, artifacts, serving, observability, cost, or governance capability, and can write a minimum production readiness plan.

**Required Cross-Part Links**

- Chapter 2 should route bottlenecks to Part 2 deep dives.
- Chapter 3 should link to checkpoint/recovery, profiling SOP, release governance, and cost chapters.

### 3.3 Part 2: Hardware And Systems Stack

**Keep**

- Keep the mechanism-heavy chapters on GPU execution, HBM/roofline, interconnect, data residency, H2D/NUMA, RDMA collectives, CUDA runtime, streams, graphs, kernel libraries, and profiling SOP.

**Reinforce**

- Add hands-on validation labs: Tensor Core miss, HBM/KV sizing, H2D overlap, NCCL preflight, launch-bound profile, CUDA Graph hit/fallback, fusion regression, and profiling report.
- Add source/date/shape labels to hardware and performance figures.
- Turn profiling and preflight guidance into standardized reports.

**Trim Or Merge**

- Leave subchapters split. Improve thin overview chapters by adding problem-routing trees and reader capability checks.

**Reader Capability After This Part**

- Reader can map slow training or serving symptoms to hardware, memory, interconnect, IO, runtime, kernel, or profiling layers.

**Required Cross-Part Links**

- Connect GPU/HBM/runtime chapters to Part 3 training and Part 5 serving capacity.
- Connect RDMA/topology chapters to Part 6 scheduling and Part 7 observability.

### 3.4 Part 3: Training Infrastructure

**Keep**

- Keep the current production training direction: single-node baseline, data parallel/FSDP, model/pipeline/context/expert parallel, checkpoint/recovery, post-training, and FTaaS.
- Preserve worked examples and readiness checklists.

**Reinforce**

- Add a Part 3 blueprint defining run manifest, baseline evidence package, admission record, checkpoint manifest, restore report, and release manifest.
- Label throughput and scaling numbers by assumption type and token/step scope.
- Add a shared restore-level taxonomy: true resume, same-shape restore, reshard restore, model-only warm start, and serving conversion.
- Strengthen cost: GPU-hour, lost tokens, storage, queue wait, retry cost, adapter hot-cache cost.

**Trim Or Merge**

- Compress repeated version matrix, preflight, release, and checklist text into shared templates, then keep chapter-specific gates.

**Reader Capability After This Part**

- Reader can design a production readiness review for a training job from single-node baseline through distributed scaling, checkpointing, recovery, post-training, and adapter publication.

**Required Cross-Part Links**

- Link to Part 2 for hardware/runtime evidence, Part 4/12 for artifacts and checkpoints, Part 5 for serving conversion and adapter loading, Part 6 for scheduling, and Part 7 for observability/release.

### 3.5 Part 4: Data And Storage Infrastructure

**Keep**

- Keep the three-line organization: data pipelines, artifacts/checkpoints, and feature/vector/cache systems.
- Preserve strong chapters such as streaming DataLoader, checkpoint engineering, vector DB operations, and RAG engineering.

**Reinforce**

- Add a Part 4 state contract: immutable identity, mutable alias, manifest fields, lineage fields, state enum, cache key dimensions, version invalidation matrix, ACL invariant, RTO/RPO baseline.
- Normalize Part 12 state names and Part 13 state/cache/index version language.
- Add evidence labels to high-risk numeric claims.

**Trim Or Merge**

- Consolidate repeated blue/green index, golden query, cache invalidation, alias rollback, and lineage YAML patterns into shared patterns.

**Reader Capability After This Part**

- Reader can design auditable, recoverable, rollbackable data/model/index/cache systems and can reason about capacity, cost, lineage, ACL, and incident triage.

**Required Cross-Part Links**

- Link to Part 0 storage semantics, Part 3 training/checkpoint, Part 5 serving/KV/cache, Part 7 release/security/incident, and Part 8 platform capstone.

### 3.6 Part 5: Serving Infrastructure

**Keep**

- Keep serving chain, batching/KV cache, quantization/compilation, vLLM/SGLang internals, multi-tenant cost, and capacity material.

**Reinforce**

- Add a serving `CapacityLedger` spanning TTFT, TPOT, ITL, KV GB, active sequences, prefix hit rate, GPU-second, queue wait, cold start, warm pool cost, and tenant budget.
- Standardize benchmark reproducibility for serving and engine comparisons.
- Connect release unit and rollback contracts to Part 7.

**Trim Or Merge**

- Keep Chapter 16 as the decision entry; avoid duplicating detailed engine internals already covered in 16a/16b.

**Reader Capability After This Part**

- Reader can design LLM serving architecture and explain tail latency, throughput, cache, quantization, engine, multi-tenant, and cost trade-offs with evidence.

**Required Cross-Part Links**

- Link to Part 6 autoscaling/queue/GPU pool, Part 7 SLO/release/security, Part 8 agent sessions, and Part 4 RAG/vector/cache.

### 3.7 Part 6: Platform And Orchestration

**Keep**

- Keep the split across images, runtime/device injection, supply chain, runtime troubleshooting, K8s workload, GPU scheduling, CRD/operator, K8s SOP, queues, GPU sharing, autoscaling, and capacity SOP.

**Reinforce**

- Add control-plane object models, admission decision logs, evidence bundle schema, operator state machines, GPU SKU contracts, capacity ledger, and RACI/owner fields in SOPs.
- Treat 18/19/20 as chapter-group contracts or stronger navigation chapters with reader deliverables.

**Trim Or Merge**

- Reduce duplicated GPU/MIG/topology/checklist content between 18b/19b/20b and clarify ownership boundaries.

**Reader Capability After This Part**

- Reader can turn AI workloads into schedulable, observable, recoverable, secure, and governable platform objects.

**Required Cross-Part Links**

- Link to Part 5 serving needs, Part 7 observability/release/security, Part 3 training requirements, and appendix checklists.

### 3.8 Part 7: Reliability, Security, And Governance

**Keep**

- Keep the themes: observability/capacity, evaluation/release/incident, security/isolation/governance.

**Reinforce**

- Rewrite toward executable production mechanisms: observability schemas, SLO burn-rate formulas, release state machine, rollback drill, incident taxonomy, threat model, policy-as-code, artifact ingest policy, cache security invariants.

**Trim Or Merge**

- Replace generic recommendations with runbooks, gates, schemas, and drills.

**Reader Capability After This Part**

- Reader can review and operate AI production risk: SLOs, capacity, release, rollback, incident response, security policy, tenant isolation, and governance evidence.

**Required Cross-Part Links**

- Link directly to Part 5/6 metrics, versions, evidence, capacity, quota, tenant, and cache policies.

### 3.9 Part 8: Advanced And Capstone

**Keep**

- Keep platform blueprint and inference-time compute / agent infrastructure direction.

**Reinforce**

- Convert Chapter 24 into a true capstone with input constraints, API objects, deliverables, scoring rubric, milestones, and end-to-end evidence chain.
- Convert Chapter 25 into an agent runtime contract: session state schema, budget enforcement, tool sandbox, capacity test, step replay, redaction, and security controls.

**Trim Or Merge**

- Avoid ending with conceptual summaries. The final part should require readers to produce platform and runtime specs.

**Reader Capability After This Part**

- Reader can produce a reviewable medium-scale AI platform design and a production agent runtime contract.

**Required Cross-Part Links**

- Chapter 24 should explicitly reuse Chapters 14-23.
- Chapter 25 should link serving, budget, security, tool use, cache, and observability threads.

### 3.10 Appendix

**Keep**

- Keep glossary, tooling map, checklists, and answers.

**Reinforce**

- Add chapter backlinks, freshness policy, version-sensitive labels, rubric, owner/evidence/threshold/action fields, and assumption labels for estimates.

**Trim Or Merge**

- Split `appendix/answers.md` by part or chapter. Keep a top-level index.

**Reader Capability After This Part**

- Reader can use the appendix as operational lookup, self-assessment, and review checklist rather than as a passive glossary.

**Required Cross-Part Links**

- Every tool, term, checklist item, and answer should point back to the relevant source chapter.

## 4. Chapter-Level Actions

### `README.md`

1. **Add Part 0 and cross-cutting deliverables**
   - Reason: navigation is strong, but reader deliverables are not explicit.
   - Add or rewrite: add expected evidence packages for CPU, memory/IO, storage, network, training, serving, platform, and release.
   - Acceptance signal: a reader can tell what artifact they should produce after each major part.
2. **Declare Markdown source policy**
   - Reason: source Markdown and generated HTML links currently drift.
   - Add or rewrite: state that Markdown is source of truth and HTML is generated output.
   - Acceptance signal: future source links avoid generated `.html` targets.
3. **Expose capacity/cost line**
   - Reason: tutorial claims quantitative engineering but the roadmap does not show the formula path.
   - Add or rewrite: list where readers learn GPU-hour, token cost, checkpoint storage, queue wait, and serving cost.
   - Acceptance signal: learning paths mention cost and capacity artifacts, not only chapters.

### `00-preface.md`

1. **Add evidence-first reading rule**
   - Reason: diagnostic mindset is emphasized but evidence quality is not defined.
   - Add or rewrite: add "no counter/log/trace/retest, no diagnosis" as a recurring standard.
   - Acceptance signal: later SOPs can reference this rule.
2. **Add recovery and rollback questions**
   - Reason: reliability is a core AI Infra promise but not in the main reading questions.
   - Add or rewrite: add questions about failure recovery, rollback validation, RPO/RTO, and degradation.
   - Acceptance signal: chapter exercises can map back to these questions.
3. **Frame capacity, cost, and SLO as a triangle**
   - Reason: budget is mentioned but not operationalized.
   - Add or rewrite: add a short model of capacity/cost/SLO trade-offs.
   - Acceptance signal: readers evaluate designs across all three dimensions.

### `part0-foundations-of-systems/0a-cpu-microarchitecture.md`

1. **Add CPU evidence bundle template**
   - Reason: overview is clear but lacks a deliverable.
   - Add or rewrite: `perf stat`, Top-Down, `perf c2c`, CPU affinity, workload window, conclusion, next action.
   - Acceptance signal: 0a readers can produce a CPU diagnosis packet.
2. **Repair stale anchors and source links**
   - Reason: stale anchors and HTML links weaken source integrity.
   - Add or rewrite: convert links to valid Markdown targets and real anchors.
   - Acceptance signal: `rg ".html|#section" part0-foundations-of-systems/0a*` has no problematic source links.
3. **Add hardware generation boundaries**
   - Reason: perf events and thresholds are platform-specific.
   - Add or rewrite: Intel/AMD/ARM event caveats and fallback observations.
   - Acceptance signal: thresholds are not presented as universal constants.

### `part0-foundations-of-systems/0a1-pipeline.md`

1. **Scope IPC thresholds by workload**
   - Reason: IPC health differs for tokenizer, DataLoader, service gateway, and spin loops.
   - Add or rewrite: add scenario-specific baselines, sampling windows, and false positives.
   - Acceptance signal: IPC guidance cannot be applied without workload context.
2. **Map pipeline stalls to Top-Down evidence**
   - Reason: mechanism is strong but counter mapping should be direct.
   - Add or rewrite: table from hazards/stalls to Top-Down categories and next commands.
   - Acceptance signal: reader can choose the next command from a `perf stat` output.
3. **Strengthen retest loop**
   - Reason: worked example has optimization but needs validation protocol.
   - Add or rewrite: baseline, hypothesis, change, retest, rollback columns.
   - Acceptance signal: example reads like a production performance fix report.

### `part0-foundations-of-systems/0a2-out-of-order-execution.md`

1. **Add ROB/MLP capacity estimate**
   - Reason: OoO depth should translate into engineering sizing intuition.
   - Add or rewrite: formula for how many independent misses are needed to hide latency.
   - Acceptance signal: pointer chasing limitations can be estimated.
2. **Add LSQ and aliasing failure evidence**
   - Reason: memory disambiguation is described, but production symptoms need evidence.
   - Add or rewrite: store-forwarding failures, alias conflicts, relevant counters, exclusions.
   - Acceptance signal: reader can distinguish pointer chasing from store/load ordering issues.
3. **Add data-structure migration trade-offs**
   - Reason: array/index refactors have maintenance and concurrency cost.
   - Add or rewrite: memory overhead, ABI risk, concurrency boundary, rollback plan.
   - Acceptance signal: refactor recommendation includes engineering cost.

### `part0-foundations-of-systems/0a3-branch-prediction.md`

1. **Connect branch misses to tail latency**
   - Reason: P99 claims need measurable correlation.
   - Add or rewrite: perf + trace/log time-alignment method for branch miss bursts.
   - Acceptance signal: reader can prove or reject branch prediction as tail cause.
2. **Add "do not optimize" boundary**
   - Reason: branch cleanup can create complex code with little value.
   - Add or rewrite: hotness threshold, miss-rate threshold, Amdahl bound, and rollback condition.
   - Acceptance signal: cold-path work requires evidence.
3. **Cross-link tokenizer and parser decisions**
   - Reason: branch vs SIMD trade-offs recur in AI host-side code.
   - Add or rewrite: links to SIMD, CPU worked example, and tokenization chapters.
   - Acceptance signal: reader can choose between branch cleanup and vectorization.

### `part0-foundations-of-systems/0a4-simd.md`

1. **Add SIMD end-to-end benefit model**
   - Reason: kernel speedup may vanish through Amdahl, downclock, or memory bandwidth.
   - Add or rewrite: formula including hot fraction, frequency drop, and memory bound.
   - Acceptance signal: 3x kernel speedup can be translated into expected service gain.
2. **Add portability and dispatch gate**
   - Reason: intrinsics create production portability debt.
   - Add or rewrite: CPU feature dispatch, fallback paths, CI matrix for AVX2/AVX-512/NEON.
   - Acceptance signal: SIMD code has supported fallback and tests.
3. **Add non-Intel observation path**
   - Reason: perf SIMD events vary by platform.
   - Add or rewrite: Intel/AMD/ARM alternatives and when to rely on timing/profile instead.
   - Acceptance signal: chapter remains usable outside Intel systems.

### `part0-foundations-of-systems/0a5-cache-hierarchy.md`

1. **Add working-set capacity worksheet**
   - Reason: cache capacity material should produce worker sizing decisions.
   - Add or rewrite: worker count, working set, LLC size, NUMA locality, miss budget.
   - Acceptance signal: reader can estimate DataLoader worker upper bound.
2. **Add prefetcher evidence gate**
   - Reason: prefetcher explanations are easy to overfit.
   - Add or rewrite: control/treatment method, counters, and false attribution warnings.
   - Acceptance signal: prefetcher diagnosis requires comparative evidence.
3. **Disambiguate CPU cache and Page Cache**
   - Reason: readers may confuse microarchitectural cache and OS Page Cache.
   - Add or rewrite: short contrast and link to 0b2.
   - Acceptance signal: glossary and chapter use distinct definitions.

### `part0-foundations-of-systems/0a6-mesi-coherence.md`

1. **Repair import links**
   - Reason: stale `#section` anchors are visible source defects.
   - Add or rewrite: replace with valid Markdown anchors or non-anchor links.
   - Acceptance signal: link scan has no stale anchors.
2. **Separate coherence and memory-ordering bugs**
   - Reason: performance HITM issues and correctness ordering bugs require different actions.
   - Add or rewrite: symptom/tool/root-cause/remediation comparison table.
   - Acceptance signal: reader does not use perf c2c to diagnose ordering correctness bugs.
3. **Quantify cross-socket cost**
   - Reason: UPI/IF traffic should map to throughput loss.
   - Add or rewrite: estimate remote HITM and interconnect utilization impact.
   - Acceptance signal: false-sharing examples explain why scaling reverses.

### `part0-foundations-of-systems/0a7-false-sharing.md`

1. **Add fix cost accounting**
   - Reason: padding and thread-local buffering have memory, ABI, and staleness costs.
   - Add or rewrite: compare padding, batching, thread-local reduction, and semantic redesign.
   - Acceptance signal: each fix includes benefit and cost.
2. **Standardize `perf c2c` evidence**
   - Reason: detection should produce a reusable report.
   - Add or rewrite: minimum fields: line address, symbol, HITM %, writer threads, socket locality.
   - Acceptance signal: reader can write a root-cause statement from c2c output.
3. **Reduce MESI duplication**
   - Reason: 0a6 owns coherence mechanics.
   - Add or rewrite: keep 0a7 focused on diagnosis and fixes, with mechanism link to 0a6.
   - Acceptance signal: duplicate state-machine explanations are reduced.

### `part0-foundations-of-systems/0a8-cpu-worked-example.md`

1. **Make this the 0a graduation task**
   - Reason: comprehensive chapter should define completion standard.
   - Add or rewrite: "0a completion report" with baseline, diagnosis, change, retest, rollback.
   - Acceptance signal: readers know what they can do after 0a.
2. **Add constrained-observability paths**
   - Reason: perf may be unavailable in containers or restricted hosts.
   - Add or rewrite: fallback path for missing privileges, unavailable events, debug pods.
   - Acceptance signal: production container readers can still proceed.
3. **Add non-CPU handoff conditions**
   - Reason: CPU symptoms may originate in IO or network.
   - Add or rewrite: when to jump to 0b/0d from CPU evidence.
   - Acceptance signal: cases have explicit "not CPU" exits.

### `part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md`

1. **Add memory/IO evidence bundle**
   - Reason: overview lacks a reader deliverable.
   - Add or rewrite: faults, TLB, RSS/cgroup, NUMA, PCIe, syscall, iostat, H2D evidence.
   - Acceptance signal: 0b readers can produce a memory/IO diagnosis packet.
2. **Repair chapter naming references**
   - Reason: references such as "第 5b 章" may not match README.
   - Add or rewrite: normalize names or point to exact Markdown paths.
   - Acceptance signal: no ambiguous nonexistent chapter references.
3. **Add formula index**
   - Reason: 0b is central to capacity reasoning.
   - Add or rewrite: guide to page-table memory, dirty writeback, pinned memory, queue depth.
   - Acceptance signal: formulas are discoverable from the overview.

### `part0-foundations-of-systems/0b1-virtual-memory-page-tables-and-tlb.md`

1. **Add page-table memory model**
   - Reason: virtual memory needs a concrete capacity cost.
   - Add or rewrite: 4K/2M/1G page table memory estimates.
   - Acceptance signal: reader can estimate page-table overhead for large mmap.
2. **Standardize fault evidence**
   - Reason: mmap cold start can be TLB, major fault, or storage.
   - Add or rewrite: major/minor fault, page walk, readahead, IO evidence table.
   - Acceptance signal: reader can separate page-fault classes.
3. **Clarify mmap versus GPU access**
   - Reason: mmap does not mean GPU zero-copy.
   - Add or rewrite: counterexample path and links to DMA/GDRDMA chapters.
   - Acceptance signal: readers know when to move to 0b3/0d3c.

### `part0-foundations-of-systems/0b2-page-cache-writeback-and-huge-pages.md`

1. **Define boundary with storage semantics**
   - Reason: Page Cache/writeback and fsync/checkpoint protocols overlap.
   - Add or rewrite: 0b2 owns memory/writeback; 0c3 owns persistence protocol.
   - Acceptance signal: chapters cross-link without duplicating full protocol.
2. **Add dirty writeback capacity formula**
   - Reason: dirty ratio examples should generalize.
   - Add or rewrite: dirty bytes, writeback bandwidth, stall window, checkpoint pause estimate.
   - Acceptance signal: reader can estimate checkpoint-induced writeback risk.
3. **Add Huge Page gate**
   - Reason: THP/HugeTLB should not be blindly enabled.
   - Add or rewrite: required evidence, expected benefit, memory reservation cost, rollback.
   - Acceptance signal: huge-page recommendation requires TLB evidence and retest.

### `part0-foundations-of-systems/0b3-numa-pcie-dma-and-pinned-memory.md`

1. **Repair chapter references**
   - Reason: "第 5b" references are likely stale.
   - Add or rewrite: use exact target Markdown paths.
   - Acceptance signal: no stale chapter names remain.
2. **Add PCIe/H2D bandwidth budget**
   - Reason: H2D problems should be estimable.
   - Add or rewrite: batch size, copy bandwidth, copy engine, NUMA hop, queue overlap table.
   - Acceptance signal: reader can decide whether GPU idle is due to H2D feeding.
3. **Add pinned-memory risk boundary**
   - Reason: pinned memory affects reclaim, cgroups, and Page Cache.
   - Add or rewrite: ulimit/cgroup/Page Cache/OOM monitoring and rollback.
   - Acceptance signal: pinning guidance includes upper bounds and monitoring.

### `part0-foundations-of-systems/0b4-syscall-epoll-io-uring-and-service-io.md`

1. **Convert source links to Markdown**
   - Reason: current source references include generated HTML targets.
   - Add or rewrite: replace `.html` links with `.md` source links.
   - Acceptance signal: `rg ".html" part0-foundations-of-systems/0b4-*.md` has no result.
2. **Add io_uring applicability table**
   - Reason: readers may assume io_uring always improves performance.
   - Add or rewrite: syscall-bound, copy-bound, backend-bound, object-store-bound, slow-client cases.
   - Acceptance signal: chapter includes "do not use io_uring first" scenarios.
3. **Add service IO queueing model**
   - Reason: service concurrency needs capacity reasoning.
   - Add or rewrite: fd count, event loop, worker pool, queue length, P99 wait estimate.
   - Acceptance signal: reader can estimate thread/queue risk from QPS and latency.

### `part0-foundations-of-systems/0c-filesystems-and-storage-internals.md`

1. **Convert source links to Markdown**
   - Reason: overview links point to `.html`.
   - Add or rewrite: replace with `.md` links.
   - Acceptance signal: no `.html` remains in source chapter links.
2. **Add storage evidence bundle**
   - Reason: overview is directory-like and lacks engineering output.
   - Add or rewrite: fio, iostat, fs semantics, crash/restart, object-store, recovery evidence.
   - Acceptance signal: reader can produce dataset/checkpoint/storage diagnosis packet.
3. **Strengthen 0b/0c boundary**
   - Reason: Page Cache and filesystem semantics are easily conflated.
   - Add or rewrite: diagram separating memory cache from persistence semantics.
   - Acceptance signal: each subchapter states what it receives from 0b and owns in 0c.

### `part0-foundations-of-systems/0c1-vfs-inode-dentry-and-block-layer.md`

1. **Convert links to Markdown**
   - Reason: source references point to generated output.
   - Add or rewrite: replace `.html` with `.md`.
   - Acceptance signal: no generated links remain.
2. **Deepen VFS-to-block path**
   - Reason: current mechanism can be more kernel-path explicit.
   - Add or rewrite: open/read/write path through inode, page cache, bio, blk-mq, device queue.
   - Acceptance signal: reader can explain cache hit/miss path differences.
3. **Add fio/iostat interpretation table**
   - Reason: commands need evidence interpretation.
   - Add or rewrite: fio parameters, queue depth, await, util, iops, bandwidth, app/device split.
   - Acceptance signal: reader can decide whether application or device is slow.

### `part0-foundations-of-systems/0c2-local-filesystems-ext4-xfs-zfs.md`

1. **Convert links to Markdown**
   - Reason: source links use generated targets.
   - Add or rewrite: replace `.html` with `.md`.
   - Acceptance signal: source link scan passes.
2. **Add workload-specific filesystem cost table**
   - Reason: selection needs more than feature comparison.
   - Add or rewrite: checkpoint, dataset cache, metadata-heavy workloads across ext4/XFS/ZFS.
   - Acceptance signal: each workload compares throughput, recovery, operational cost.
3. **Add crash-recovery drill**
   - Reason: journaling and CoW claims should be verified.
   - Add or rewrite: power-loss/kill/remount simulation steps and expected results.
   - Acceptance signal: rename/fsync/snapshot behavior is tested, not assumed.

### `part0-foundations-of-systems/0c3-storage-semantics-fsync-direct-io-and-checkpoints.md`

1. **Convert links to Markdown**
   - Reason: source links use `.html`.
   - Add or rewrite: replace generated links.
   - Acceptance signal: source link scan passes.
2. **Strengthen atomic checkpoint protocol**
   - Reason: this is a core reliability chapter.
   - Add or rewrite: multi-rank manifest state machine and crash-point table.
   - Acceptance signal: every crash point has a defined recovery behavior.
3. **Quantify Direct IO boundary**
   - Reason: O_DIRECT is often benchmark-driven and misused.
   - Add or rewrite: alignment, cache bypass, CPU copy, queue depth, and dataset-reader harm cases.
   - Acceptance signal: chapter says when Direct IO hurts.

### `part0-foundations-of-systems/0c4-object-storage-parallel-filesystems-and-dataset-io.md`

1. **Add object-store cost model**
   - Reason: request count and object size dominate production cost.
   - Add or rewrite: LIST/GET/multipart request estimates, latency, cost, concurrency.
   - Acceptance signal: reader can estimate small-file migration cost.
2. **Add parallel filesystem hotspot evidence**
   - Reason: MDS/OSS/stripe issues need observable signals.
   - Add or rewrite: per-target, metadata ops, stripe imbalance, client-side evidence.
   - Acceptance signal: checkpoint hotspot example has complete evidence chain.
3. **Add dataset IO acceptance test**
   - Reason: SOP should verify end-to-end throughput.
   - Add or rewrite: cold/warm cache, workers, prefetch, GPU idle, CPU decode, object/cache/backend split.
   - Acceptance signal: reader can localize dataset IO bottlenecks.

### `part0-foundations-of-systems/0d-network-stack-fundamentals.md`

1. **Convert links to Markdown**
   - Reason: overview uses generated `.html` targets.
   - Add or rewrite: replace with `.md`.
   - Acceptance signal: no `.html` links remain.
2. **Add network evidence bundle**
   - Reason: TCP/NIC/RDMA/NCCL chapters need a shared packet.
   - Add or rewrite: commands and fields for TCP, NIC queues/offload, RDMA/fabric, NCCL logs.
   - Acceptance signal: each layer has next-step evidence.
3. **Separate control plane and data plane**
   - Reason: service RPC and training collectives are often conflated.
   - Add or rewrite: path diagram and symptom table.
   - Acceptance signal: reader distinguishes gateway P99 from AllReduce slowdown.

### `part0-foundations-of-systems/0d1-linux-network-stack-tcp-and-mtu.md`

1. **Convert links to Markdown**
   - Reason: source links use `.html`.
   - Add or rewrite: replace generated links.
   - Acceptance signal: link scan passes.
2. **Standardize BDP/socket-buffer model**
   - Reason: single-flow throughput examples should generalize.
   - Add or rewrite: RTT, cwnd, MSS, pacing, socket buffer calculation.
   - Acceptance signal: reader can estimate throughput cap from `ss -tinm`.
3. **Add PMTU rollback drill**
   - Reason: MTU changes can break service and training.
   - Add or rewrite: validation, monitoring, rollback and blast-radius guidance.
   - Acceptance signal: large-packet failure has a safe recovery path.

### `part0-foundations-of-systems/0d2-nic-offload-queues-and-service-network-io.md`

1. **Convert links to Markdown**
   - Reason: source links use `.html`.
   - Add or rewrite: replace generated links.
   - Acceptance signal: link scan passes.
2. **Add queue/RSS capacity model**
   - Reason: NIC queue behavior should map to CPU and P99.
   - Add or rewrite: flow count, queue count, CPU core, softirq budget mapping.
   - Acceptance signal: reader can diagnose single-queue hotspots.
3. **Add offload change gate**
   - Reason: TSO/GRO/LRO changes affect tail latency and packet interpretation.
   - Add or rewrite: baseline metrics, risk table, rollback trigger.
   - Acceptance signal: offload changes require controlled experiment.

### `part0-foundations-of-systems/0d3-rdma-roce-infiniband-and-gpudirect.md`

1. **Convert links to Markdown**
   - Reason: source links use generated targets.
   - Add or rewrite: replace `.html`.
   - Acceptance signal: link scan passes.
2. **Add RDMA responsibility boundary map**
   - Reason: verbs, fabric, GDRDMA, and NCCL ownership must be clear.
   - Add or rewrite: layer diagram and "go to 0d3a/b/c/0d4 when" table.
   - Acceptance signal: reader can choose the correct deep dive.
3. **Add minimum validation path**
   - Reason: new nodes need smoke tests from host to NCCL.
   - Add or rewrite: host, fabric, verbs, GPU buffer, NCCL tests.
   - Acceptance signal: platform can admit a node with this path.

### `part0-foundations-of-systems/0d3a-rdma-verbs-memory-registration-and-queues.md`

1. **Convert links to Markdown**
   - Reason: source references include `.html`.
   - Add or rewrite: replace generated links.
   - Acceptance signal: link scan passes.
2. **Quantify memory registration cost**
   - Reason: registration cache can dominate latency.
   - Add or rewrite: registration frequency, MR cache hit, pinned memory cap, latency estimate.
   - Acceptance signal: reader can separate registration overhead from transport.
3. **Expand WC error runbook**
   - Reason: WC statuses are operationally important.
   - Add or rewrite: status -> likely cause -> evidence -> next command -> fix table.
   - Acceptance signal: common WC errors have deterministic next steps.

### `part0-foundations-of-systems/0d3b-roce-infiniband-lossless-fabric-and-congestion.md`

1. **Add PFC/ECN change gate**
   - Reason: fabric tuning can trigger cluster-wide incidents.
   - Add or rewrite: baseline pause/ECN/CNP, change window, rollback thresholds.
   - Acceptance signal: pause storm can be detected and reverted.
2. **Connect ECMP entropy to rank/channel mapping**
   - Reason: NCCL path imbalance needs fabric and rank evidence together.
   - Add or rewrite: cross-link to NCCL channel/rank/HCA mapping and evidence.
   - Acceptance signal: AllReduce variance can be tied to path/hash/fabric evidence.
3. **Add version matrix fields**
   - Reason: NIC/switch/firmware behavior is version-sensitive.
   - Add or rewrite: NIC, OFED/rdma-core, firmware, switch OS, DCQCN/PFC settings.
   - Acceptance signal: SOP requires version capture.

### `part0-foundations-of-systems/0d3c-gpudirect-rdma-gpu-nic-topology-and-diagnostics.md`

1. **Add rank-to-topology binding template**
   - Reason: GPU/NIC locality must become a placement artifact.
   - Add or rewrite: local rank, GPU UUID, HCA, NUMA, rail binding table.
   - Acceptance signal: 8-GPU multi-HCA nodes can produce a binding plan.
2. **Standardize GDRDMA fallback evidence**
   - Reason: fallback diagnosis is strong but should be reusable.
   - Add or rewrite: peermem, BAR, IOMMU/ACS, NCCL logs, benchmarks, container view.
   - Acceptance signal: fallback can be classified quickly.
3. **Add container boundary matrix**
   - Reason: Kubernetes often breaks devices and libraries.
   - Add or rewrite: device plugin, mounts, capabilities, driver/userspace versions.
   - Acceptance signal: inside/outside-container test mismatch can be explained.

### `part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md`

1. **Standardize NCCL log decoding**
   - Reason: logs are central evidence.
   - Add or rewrite: transport, algo, proto, HCA, GDRDMA, fallback, channel fields.
   - Acceptance signal: reader can identify socket fallback or rail imbalance from logs.
2. **Add collective time model**
   - Reason: collective mechanisms need budget estimates.
   - Add or rewrite: ring/tree, message size, bandwidth, latency, expected time.
   - Acceptance signal: measured slowdown can be compared to physical limit.
3. **Add NCCL hang recovery runbook**
   - Reason: reliability closure needs abort/retry/isolate steps.
   - Add or rewrite: rank stall, timeout, abort, node quarantine, retry decision.
   - Acceptance signal: on-call has a bounded path for hangs.

### `part1-foundations/01-what-is-ai-infra.md`

1. **Separate positioning from production execution**
   - Reason: overlaps with Chapter 3.
   - Add or rewrite: keep definition and mental model; move detailed maturity/production flow to Chapter 3.
   - Acceptance signal: Chapter 1 can be read as a concise field map.
2. **Add minimum production evidence packet**
   - Reason: evidence-chain thinking is currently conceptual.
   - Add or rewrite: data, code, model, eval, latency, cost, rollback, owner.
   - Acceptance signal: reader can judge whether a model is publishable.
3. **Add cost-risk mini model**
   - Reason: cost governance is listed but not quantified.
   - Add or rewrite: token cost, GPU-hour, idle cost, failure rerun, warm pool cost.
   - Acceptance signal: exercise can estimate monthly cost risk.

### `part1-foundations/02-compute-storage-network.md`

1. **Turn step breakdown into optimization priority**
   - Reason: formulas exist but need decision thresholds.
   - Add or rewrite: examples for `t_load`, `t_h2d`, `t_sync`, `t_checkpoint` and first action.
   - Acceptance signal: reader can choose which segment to optimize first.
2. **Add evidence matrix**
   - Reason: symptom lists need operational evidence.
   - Add or rewrite: GPU low utilization, high P99, poor scaling, checkpoint stalls, cold start.
   - Acceptance signal: each symptom maps to metrics and commands.
3. **Add hard links to Part 2**
   - Reason: Chapter 2 should route readers into deeper systems chapters.
   - Add or rewrite: direct links to HBM, H2D, RDMA, CUDA profiling, storage chapters.
   - Acceptance signal: readers can navigate from bottleneck to mechanism.

### `part1-foundations/03-from-model-to-production.md`

1. **Rewrite as production execution manual**
   - Reason: overlaps with Chapter 1's "why production matters".
   - Add or rewrite: release pipeline, gates, rollback, feedback governance.
   - Acceptance signal: chapter backbone becomes "how to validate" rather than "why".
2. **Add Model Release Decision Record**
   - Reason: quality/latency/cost trade-offs need a review artifact.
   - Add or rewrite: template with quality delta, cost delta, latency, risk, rollback target, owner.
   - Acceptance signal: any release can be reviewed with the template.
3. **Add incident loop**
   - Reason: production chain should include postmortem.
   - Add or rewrite: timeline, blast radius, detection, mitigation, root cause, prevention.
   - Acceptance signal: exercise can produce a complete postmortem.

### `part2-systems-stack/04-gpu-and-accelerators.md`

1. **Add GPU problem routing tree**
   - Reason: overview is too close to a directory.
   - Add or rewrite: route OOM, TPOT, MFU, interconnect, procurement, virtualization to subchapters.
   - Acceptance signal: reader can choose 04a-04d from a symptom.
2. **Add Part 4-style capability check**
   - Reason: overview lacks reader output.
   - Add or rewrite: five scenario-based self checks.
   - Acceptance signal: reader explains evidence, not just target chapter.
3. **Add hardware spec boundary rule**
   - Reason: accelerator specs age quickly.
   - Add or rewrite: date, product form, dense/sparse, per-GPU/system labels.
   - Acceptance signal: future spec tables carry source assumptions.

### `part2-systems-stack/04a-gpu-execution-model-and-tensor-cores.md`

1. **Add nsys-to-ncu experiment**
   - Reason: mechanism is strong but should include validation.
   - Add or rewrite: small experiment proving Tensor Core path hit/miss.
   - Acceptance signal: reader can verify Tensor Core usage.
2. **Add low-precision release gate**
   - Reason: FP8/INT8 need quality and safety controls.
   - Add or rewrite: task, length, safety, goodput, rollback gates.
   - Acceptance signal: low precision cannot be approved on throughput alone.
3. **Normalize TFLOPS labels**
   - Reason: dense/sparse/per-GPU/system/accumulation precision can mislead.
   - Add or rewrite: source and口径 table.
   - Acceptance signal: TFLOPS numbers are traceable.

### `part2-systems-stack/04b-hbm-memory-and-roofline.md`

1. **Extract reusable memory worksheet**
   - Reason: HBM/KV/Roofline content is high value.
   - Add or rewrite: training and serving memory budget templates.
   - Acceptance signal: 70B long-context concurrency can be calculated.
2. **Add headroom risk levels**
   - Reason: 10-20% headroom should become a gate.
   - Add or rewrite: green/yellow/red OOM risk rules.
   - Acceptance signal: budget sheet can approve, warn, or reject.
3. **Add hardware-data freshness labels**
   - Reason: HBM and bandwidth specs vary by product.
   - Add or rewrite: source date/product form labels.
   - Acceptance signal: machine balance table is maintainable.

### `part2-systems-stack/04c-gpu-interconnect-and-systems.md`

1. **Add topology admission checklist**
   - Reason: topology needs platform intake gates.
   - Add or rewrite: node admission, preflight, drain, firmware, topology, NCCL tests.
   - Acceptance signal: new HGX/NVL nodes have acceptance criteria.
2. **Add scheduler resource labels**
   - Reason: topology-aware scheduling needs API surface.
   - Add or rewrite: Kubernetes/Slurm labels and pending reason examples.
   - Acceptance signal: "has GPU but pending" can be explained.
3. **Add committed-capacity formula**
   - Reason: power/cooling/maintenance capacity should be quantitative.
   - Add or rewrite: 512-GPU committed capacity after failure, maintenance, fragmentation.
   - Acceptance signal: available capacity is not equal to installed capacity.

### `part2-systems-stack/04d-gpu-selection-virtualization-and-heterogeneous-accelerators.md`

1. **Add TCO model**
   - Reason: accelerator selection cannot rely only on tokens/sec/$.
   - Add or rewrite: purchase, depreciation, power, utilization, staffing, migration risk.
   - Acceptance signal: heterogeneous pilot includes exit cost.
2. **Add workload-card examples**
   - Reason: template is useful but needs filled cases.
   - Add or rewrite: 70B serving, SFT, embedding, dev notebook cards.
   - Acceptance signal: resource-pool planning can derive from workload cards.
3. **Add virtualization SLO matrix**
   - Reason: MIG/MPS/time-slicing risks differ.
   - Add or rewrite: allowed/forbidden matrix by workload and P99 isolation.
   - Acceptance signal: production inference is not placed on risky sharing by default.

### `part2-systems-stack/05-memory-interconnect-io.md`

1. **Add IO routing tree**
   - Reason: overview is too short.
   - Add or rewrite: GPU idle, cold start, NCCL timeout, checkpoint stalls -> 05a-05d.
   - Acceptance signal: reader can route symptoms.
2. **Define byte-path worksheet**
   - Reason: "where is each byte" should become reusable.
   - Add or rewrite: where, next, boundary, avoid, evidence.
   - Acceptance signal: later chapters can reuse the worksheet.
3. **Clarify overlap with 04/06/training**
   - Reason: IO issues span layers.
   - Add or rewrite: boundary table for HBM, H2D, RDMA, checkpoint, runtime.
   - Acceptance signal: not every slowness is called IO.

### `part2-systems-stack/05a-memory-storage-hierarchy-and-data-residency.md`

1. **Add object-store dataset publish protocol**
   - Reason: POSIX/object and manifest content is strong.
   - Add or rewrite: complete publish/manifest/reader protocol.
   - Acceptance signal: readers never consume half-published datasets.
2. **Add NVMe cache capacity algorithm**
   - Reason: hot/cold policy needs concrete sizing.
   - Add or rewrite: capacity, lease, eviction priority, tenant/job isolation.
   - Acceptance signal: old jobs cannot silently fill hot cache.
3. **Add object request cost estimate**
   - Reason: request amplification is often the hidden cost.
   - Add or rewrite: small-object request and egress estimate.
   - Acceptance signal: shard-prewarm choices can be cost compared.

### `part2-systems-stack/05b-host-device-io-pcie-numa-and-overlap.md`

1. **Add H2D/NUMA profiling lab**
   - Reason: this topic benefits from observable A/B experiments.
   - Add or rewrite: pinned/non-blocking/NUMA binding experiment with nsys output.
   - Acceptance signal: reader can prove overlap improvement.
2. **Add model-load metrics schema**
   - Reason: cold start stages need separate metrics.
   - Add or rewrite: download, read, deserialize, convert, H2D, warmup.
   - Acceptance signal: readiness latency can be attributed.
3. **Add pinned-memory observability**
   - Reason: pinning can harm host memory and Page Cache.
   - Add or rewrite: locked memory, RSS, cgroup, page cache, OOM checks.
   - Acceptance signal: pinning advice includes safety checks.

### `part2-systems-stack/05c-rdma-collectives-and-cluster-topology.md`

1. **Turn preflight into platform spec**
   - Reason: this is the chapter's strongest production line.
   - Add or rewrite: node/RDMA/NCCL/smoke-test pass-fail spec.
   - Acceptance signal: cluster admission can automate it.
2. **Add NCCL timeout evidence bundle**
   - Reason: "NCCL timeout" is not a root cause.
   - Add or rewrite: rank logs, topo, HCA, switch counters, placement, timeline.
   - Acceptance signal: incident reports include root evidence.
3. **Add rank placement examples**
   - Reason: locality tables should become physical maps.
   - Add or rewrite: 64/256 GPU placement showing NVSwitch versus IB/RoCE boundaries.
   - Acceptance signal: reader can place communication groups.

### `part2-systems-stack/05d-training-storage-checkpoint-and-io-diagnostics.md`

1. **Add checkpoint RPO/RTO**
   - Reason: recovery needs targets.
   - Add or rewrite: RPO, RTO, retention, restore-drill frequency.
   - Acceptance signal: checkpoint strategy states loss and recovery time.
2. **Add atomic commit example**
   - Reason: manifest protocol needs concrete fields.
   - Add or rewrite: directory structure, manifest, latest pointer, cleanup.
   - Acceptance signal: restore logic only reads committed manifests.
3. **Add checkpoint QoS**
   - Reason: checkpoint storms are multi-tenant incidents.
   - Add or rewrite: quota, jitter, archival worker rate limits.
   - Acceptance signal: one job cannot overload storage for all.

### `part2-systems-stack/06-cuda-runtime-and-kernels.md`

1. **Add runtime diagnosis routing**
   - Reason: overview is too directory-like.
   - Add or rewrite: launch, sync, kernel, profile quadrants.
   - Acceptance signal: nsys symptoms route to 06a-06d.
2. **Add end-to-end lab path**
   - Reason: readers need practice stitching runtime pieces.
   - Add or rewrite: eager/compile/graph/profile mini model lab.
   - Acceptance signal: reader can run a complete runtime diagnosis.
3. **Clarify when to leave CUDA**
   - Reason: slowness may be HBM/H2D/RDMA.
   - Add or rewrite: boundary table for returning to 04/05.
   - Acceptance signal: CUDA is not the default blame.

### `part2-systems-stack/06a-framework-dispatch-runtime-and-kernel-launch.md`

1. **Add launch-bound profile signal**
   - Reason: fixed launch cost needs measurable threshold.
   - Add or rewrite: profiler output and decision threshold.
   - Acceptance signal: reader can distinguish launch-bound from kernel-bound.
2. **Add compile production gate**
   - Reason: `torch.compile` can fail or regress in production.
   - Add or rewrite: warmup, graph break, recompilation, numerical parity, rollback.
   - Acceptance signal: compile optimization has release checks.
3. **Add allocator evidence checklist**
   - Reason: OOM diagnosis needs memory artifacts.
   - Add or rewrite: memory snapshot, allocated/reserved, dynamic shape, fragmentation.
   - Acceptance signal: dynamic-shape OOM is reproducible.

### `part2-systems-stack/06b-streams-synchronization-and-cuda-graphs.md`

1. **Add implicit sync review checklist**
   - Reason: hidden synchronization is a common production issue.
   - Add or rewrite: `.item()`, CPU copy, print/log, assert, synchronize review rules.
   - Acceptance signal: PR review can catch sync risks.
2. **Add CUDA Graph hit/fallback gate**
   - Reason: graph performance depends on traffic shape.
   - Add or rewrite: hit rate, fallback, recapture, bucket memory cost thresholds.
   - Acceptance signal: graph release is not based on fixed-batch benchmark only.
3. **Add overlap experiment**
   - Reason: H2D/NCCL overlap should be visible.
   - Add or rewrite: nsys before/after timeline.
   - Acceptance signal: reader can prove premature waits.

### `part2-systems-stack/06c-kernel-libraries-fusion-and-sm-resource-limits.md`

1. **Add fusion decision record**
   - Reason: fusion can regress.
   - Add or rewrite: saved bytes, registers, spills, occupancy, end-to-end gain, rollback.
   - Acceptance signal: each fusion has a decision record.
2. **Add library upgrade matrix**
   - Reason: CUDA/PyTorch/Triton/FlashAttention upgrades are high risk.
   - Add or rewrite: performance, memory, correctness, rollback test matrix.
   - Acceptance signal: library upgrades have staged validation.
3. **Add kernel baseline suite**
   - Reason: kernel regressions need standard workloads.
   - Add or rewrite: prefill, decode, norm, sampling, MoE, optimizer baselines.
   - Acceptance signal: nightly can detect kernel regressions.

### `part2-systems-stack/06d-profiling-debugging-and-performance-sop.md`

1. **Make 06d the Part 2 troubleshooting hub**
   - Reason: it can connect hardware, IO, runtime, and kernel diagnosis.
   - Add or rewrite: symptom-to-chapter routing at top.
   - Acceptance signal: reader can navigate from slow symptom to layer.
2. **Add profile report template**
   - Reason: profiling should produce a reviewable artifact.
   - Add or rewrite: baseline, timeline, hypothesis, evidence, fix, retest, regression guard.
   - Acceptance signal: performance changes can be reviewed.
3. **Define performance CI layers**
   - Reason: PR/nightly/release tests differ.
   - Add or rewrite: workload, threshold, artifact, owner by CI layer.
   - Acceptance signal: regressions attach profiler artifacts.

### `part3-training-infra/07-single-node-training.md`

1. **Define single-node baseline evidence package**
   - Reason: later training chapters depend on Chapter 7 output.
   - Add or rewrite: non-pad tokens/s, MFU/HFU, HBM P95, data wait, H2D, checkpoint time, profile window, env digest.
   - Acceptance signal: Chapter 8 can consume the same fields.
2. **Label throughput assumptions**
   - Reason: worked example numbers may be mistaken as universal H100 gates.
   - Add or rewrite: raw/non-pad, microstep/optimizer step, aggregate/per-GPU, model/kernel/hardware assumptions.
   - Acceptance signal: every throughput gate carries scope.
3. **Add data-path A/B matrix**
   - Reason: DataLoader diagnosis needs controlled experiments.
   - Add or rewrite: num_workers, prefetch, pin_memory, local cache, packing order and expected evidence.
   - Acceptance signal: reader can isolate IO, CPU, collate, and H2D.
4. **Add cost per token**
   - Reason: baseline without cost is incomplete.
   - Add or rewrite: tokens/s, GPU count, GPU-hour price, checkpoint pause -> cost per 1B non-pad tokens.
   - Acceptance signal: worked example outputs cost.

### `part3-training-infra/08-data-parallel.md`

1. **Add do-not-scale gate**
   - Reason: unresolved single-node problems should block distributed scaling.
   - Add or rewrite: MFU unexplained, data wait high, checkpoint invalid, memory growth unexplained.
   - Acceptance signal: admission can reject multi-node expansion.
2. **Clarify DDP/FSDP/ZeRO boundaries**
   - Reason: FSDP is often misread as pure data parallel or memory magic.
   - Add or rewrite: replication/sharding/communication/checkpoint comparison.
   - Acceptance signal: each strategy has state and recovery shape.
3. **Add placement artifact**
   - Reason: topology-aware training needs a recorded mapping.
   - Add or rewrite: `placement.json` with node, GPU UUID, HCA, rail, rank.
   - Acceptance signal: incident evidence can replay physical mapping.
4. **Add elastic restart semantics**
   - Reason: world-size changes affect sampler, batch, and LR.
   - Add or rewrite: fixed world size, elastic membership, warm start comparison.
   - Acceptance signal: readers know when true resume is impossible.

### `part3-training-infra/09-model-pipeline-parallel.md`

1. **Add parallel-strategy input sheet**
   - Reason: decision tree needs inputs.
   - Add or rewrite: layers, hidden, heads, seq, topology, batch, RPO/RTO, inference target.
   - Acceptance signal: 70B/405B examples derive from the sheet.
2. **Label model assumptions**
   - Reason: dense/MoE/GQA/long-context assumptions change strategy.
   - Add or rewrite: applicability boundaries on worked examples.
   - Acceptance signal: examples cannot be misapplied to MoE/128K context.
3. **Add training-to-serving conversion checklist**
   - Reason: sharded training checkpoints must become inference artifacts.
   - Add or rewrite: merge/re-shard, PP reassembly, tokenizer/config, golden prompts.
   - Acceptance signal: conversion has a dry-run acceptance path.
4. **Quantify stage imbalance**
   - Reason: pipeline bubble math does not cover all imbalance.
   - Add or rewrite: stage_time P50/P95, virtual stage, layer split, embedding/LM head effects.
   - Acceptance signal: profile data drives PP adjustments.

### `part3-training-infra/10-memory-checkpointing-and-recovery.md`

1. **Define restore-level taxonomy**
   - Reason: true resume and warm start are not interchangeable.
   - Add or rewrite: true resume, same-shape restore, reshard restore, model-only warm start, serving conversion.
   - Acceptance signal: every restore scenario maps to one level.
2. **Add checkpoint schema migration**
   - Reason: schema changes are production changes.
   - Add or rewrite: reader-first, dual-write, migration job, quarantine, restore backfill.
   - Acceptance signal: breaking schema has a release path.
3. **Add object-store failure handling**
   - Reason: checkpoint storage has multipart, object count, marker, and prefix risks.
   - Add or rewrite: object budget, prefix sharding, manifest-only listing, 429/5xx retry.
   - Acceptance signal: large checkpoint example includes metadata pressure.
4. **Add restore report**
   - Reason: dry-run restore needs a standard artifact.
   - Add or rewrite: `restore_report.json` with schema, state coverage, shape diff, duration, loss parity, cursor parity.
   - Acceptance signal: restore drills produce comparable reports.

### `part3-training-infra/10b-alignment-and-post-training.md`

1. **Deepen DPO/GRPO engineering**
   - Reason: PPO/RLHF is stronger than DPO/GRPO sections.
   - Add or rewrite: DPO reference cache/beta/length bias; GRPO group size/rollout cost/rule reward trust.
   - Acceptance signal: DPO and GRPO each have capacity model and failure table.
2. **Add RM governance contract**
   - Reason: RM is a production service, not just a training component.
   - Add or rewrite: RM model card, calibration report, score distribution parity, length-bucket audit.
   - Acceptance signal: RM update cannot alone approve release.
3. **Add rollout buffer state machine**
   - Reason: mixed actor/RM versions are dangerous.
   - Add or rewrite: generated, scored, ready, consumed, stale, quarantined states.
   - Acceptance signal: version mismatch cannot enter update.
4. **Compare weight-sync mechanisms**
   - Reason: rollout/training sync can dominate time.
   - Add or rewrite: full copy, delta sync, shared storage, in-memory broadcast costs and failures.
   - Acceptance signal: examples explain `T_sync`.

### `part3-training-infra/10c-finetuning-and-multi-adapter.md`

1. **Add FTaaS SLO decomposition**
   - Reason: time-to-eval/prod needs queue and resource model.
   - Add or rewrite: queue wait, base locality, GPU fragmentation, retry, eval gate.
   - Acceptance signal: p95 target can be traced to pool capacity.
2. **Define adapter lifecycle events**
   - Reason: state names need owners and idempotency.
   - Add or rewrite: registered, approved, released, loaded, canary, production, rolled_back.
   - Acceptance signal: duplicate events do not corrupt routes.
3. **Add serving-engine compatibility matrix**
   - Reason: trainable adapter does not imply loadable adapter.
   - Add or rewrite: vLLM/LoRAX/TensorRT-LLM/TGI support for rank, dtype, modules, merge, hot load, quantization.
   - Acceptance signal: adapter release requires a passed engine profile.
4. **Add quality regression evidence chain**
   - Reason: adapter regressions must be diagnosable.
   - Add or rewrite: golden replay, shadow diff, refusal/TTFT/error/win-rate dashboard.
   - Acceptance signal: canary failure maps to data, template, base, or serving path.

### `part4-data-and-storage/11-data-pipeline.md`

1. **Add Part 4 data contract table**
   - Reason: downstream chapters need shared invariants.
   - Add or rewrite: record, schema, event_time, dataset_version, lineage, ACL, retention.
   - Acceptance signal: 11a-11e/12a/13a/13d can reference the same contract.
2. **Add end-to-end data triage table**
   - Reason: data issues affect training, checkpoint, RAG, and release.
   - Add or rewrite: symptom -> impacted chapters -> metrics -> first action.
   - Acceptance signal: lag, schema rejection, PII miss, bad shard, lineage gap have paths.
3. **Add capacity-cost worksheet**
   - Reason: deep-dive numbers need a shared entry point.
   - Add or rewrite: TB/day, tokens/day, shuffle, storage, egress, rebuild time.
   - Acceptance signal: estimates can be reused in subchapters.

### `part4-data-and-storage/11a-data-ingestion.md`

1. **Define ingestion record contract**
   - Reason: schema, PII, tenancy, idempotency are spread out.
   - Add or rewrite: record_id, event_time, schema_id, tenant_id, pii_labels, lineage_id, idempotency_key.
   - Acceptance signal: every ingest path maps to the contract.
2. **Clarify exactly-once boundaries**
   - Reason: production often uses at-least-once plus idempotency.
   - Add or rewrite: queue/sink/transaction/dedup-window decision table.
   - Acceptance signal: Kafka/CDC/object events have explicit semantics.
3. **Add GDPR deletion boundary**
   - Reason: deletion does not uniformly apply to backups, derived data, and weights.
   - Add or rewrite: raw, derived, backup, feature/index, trained weights categories.
   - Acceptance signal: each category has SLA and exception path.

### `part4-data-and-storage/11b-data-cleaning-dedup-quality.md`

1. **Add quality policy template**
   - Reason: quality metrics need governance.
   - Add or rewrite: threshold, owner, exception, rollback, retention-rate alert.
   - Acceptance signal: filtering rules are reviewable.
2. **Add contamination CI gate**
   - Reason: eval contamination should block pipelines.
   - Add or rewrite: n-gram overlap, embedding near-dup, manual review, blocking threshold.
   - Acceptance signal: golden/eval contamination fails CI.
3. **Label evidence strength**
   - Reason: performance/quality claims need provenance.
   - Add or rewrite: measured, illustrative, source-derived, assumption labels.
   - Acceptance signal: readers know which numbers are reusable.

### `part4-data-and-storage/11c-tokenization-and-dataset-formats.md`

1. **Add tokenizer compatibility contract**
   - Reason: tokenizer changes affect training, registry, serving, and RAG.
   - Add or rewrite: vocab_hash, special_tokens, normalizer, chat_template, model_family.
   - Acceptance signal: release gates can reuse the check.
2. **Define dataset state schema**
   - Reason: resume/shuffle/packing state must align with DataLoader and checkpoint.
   - Add or rewrite: shard_id, offset, epoch, rng, packing_policy, loss_mask_policy.
   - Acceptance signal: DataLoader and checkpoint examples use same fields.
3. **Add packing unit tests**
   - Reason: document boundaries and SFT masks are high-risk.
   - Add or rewrite: minimal examples and assertions for cu_seqlens, loss mask, resume offset.
   - Acceptance signal: packing bugs are testable.

### `part4-data-and-storage/11d-streaming-and-dataloader-engineering.md`

1. **Add production runbook thresholds**
   - Reason: diagnostics are strong but operational action needs thresholds.
   - Add or rewrite: GPU idle, queue depth, H2D, IO wait, worker skew, pinned memory alerts.
   - Acceptance signal: each metric has owner and first command.
2. **Clarify StatefulDataLoader stability**
   - Reason: APIs and ecosystem change.
   - Add or rewrite: version, import path, alternatives, compatibility risks.
   - Acceptance signal: reader knows whether it is production-safe.
3. **Add NUMA/topology tuning**
   - Reason: DataLoader competes with NCCL and NVMe on large nodes.
   - Add or rewrite: worker pinning, local disk affinity, GPU affinity, CPU reserve.
   - Acceptance signal: worked example includes topology checks.

### `part4-data-and-storage/11e-data-versioning-and-lineage.md`

1. **Define mandatory lineage API**
   - Reason: examples need enforceable fields.
   - Add or rewrite: input/output/code/env/params/actor/time/digest/schema_version.
   - Acceptance signal: missing lineage fails training or release.
2. **Unify immutable identity plus alias**
   - Reason: dataset, checkpoint, model, and index versions share the same pattern.
   - Add or rewrite: immutable digest + mutable alias rule and anti-patterns.
   - Acceptance signal: all versioned objects use the same abstraction.
3. **Add metadata system RPO/RTO**
   - Reason: lineage registry itself is critical infrastructure.
   - Add or rewrite: metadata backup, restore drill, corruption recovery.
   - Acceptance signal: quarterly restore has measurable outcome.

### `part4-data-and-storage/12-artifacts-and-checkpoints.md`

1. **Normalize Part 12 states**
   - Reason: state names differ across 12/12a/12b/12c.
   - Add or rewrite: one lifecycle enum or explicit mapping.
   - Acceptance signal: subchapter state diagrams align.
2. **Separate checkpoint, model package, and release**
   - Reason: these are different governance objects.
   - Add or rewrite: owner, immutability, rollback, retention, security gate comparison.
   - Acceptance signal: reader knows which system a change affects.
3. **Add end-to-end trace graph**
   - Reason: dataset-to-serving evidence needs a spine.
   - Add or rewrite: dataset_version -> train_run -> checkpoint -> registry_version -> release_alias -> serving.
   - Acceptance signal: 12a-12d reference the graph.

### `part4-data-and-storage/12a-model-registry.md`

1. **Define alias/stage transaction rules**
   - Reason: concurrent promotion and rollback are production hazards.
   - Add or rewrite: row lock, unique production alias, compare-and-swap, cache invalidation.
   - Acceptance signal: double promotion has deterministic behavior.
2. **Add immutable field validation**
   - Reason: digest/tokenizer/lineage mutation should be rejected.
   - Add or rewrite: required fields, forbidden mutations, 409/422 errors, audit log.
   - Acceptance signal: registry API protects identity.
3. **Expand LoRA compatibility matrix**
   - Reason: hidden size is insufficient.
   - Add or rewrite: base digest, layer count, target modules, rank, tokenizer, RoPE/config.
   - Acceptance signal: incompatible adapters cannot enter staging.

### `part4-data-and-storage/12b-checkpoint-engineering.md`

1. **Correct optimizer-state sizing**
   - Reason: Adam state may include FP32 moments and master weights.
   - Add or rewrite: ZeRO/FSDP, BF16, FP32 master, Adam moments scenarios.
   - Acceptance signal: checkpoint sizes are reproducible from formulas.
2. **Add object-store partial-upload recovery**
   - Reason: incomplete checkpoints must never be restored.
   - Add or rewrite: multipart cleanup, checksum failure, marker rollback SOP.
   - Acceptance signal: restore reads only committed manifests.
3. **Automate true-resume verification**
   - Reason: loading is not sufficient.
   - Add or rewrite: RNG, optimizer, scheduler, dataset state, loss continuity checks.
   - Acceptance signal: resume fails if loss diverges beyond threshold.

### `part4-data-and-storage/12c-release-governance.md`

1. **Align release state machine with registry**
   - Reason: Chapter 12a and 12c states diverge.
   - Add or rewrite: one state machine with transition conditions, approvers, rollback.
   - Acceptance signal: registry API can support release governance.
2. **Add statistical gate framework**
   - Reason: canary/offline pass needs statistical evidence.
   - Add or rewrite: sample size, MDE, confidence interval, segment coverage, guardrails.
   - Acceptance signal: pass/fail has statistical basis.
3. **Add full rollback runbook**
   - Reason: kill switch is not enough.
   - Add or rewrite: alias switch, warmup, cache invalidation, prompt/index rollback, incident link.
   - Acceptance signal: rollback can meet a measured RTO.

### `part4-data-and-storage/12d-supply-chain-and-signing.md`

1. **Separate model artifact admission**
   - Reason: image signing does not cover weights, tokenizer, config, prompt template.
   - Add or rewrite: registry gate and serving initContainer verification of model bundle.
   - Acceptance signal: unsigned weights cannot load in production.
2. **Add freshness labels to ecosystem claims**
   - Reason: HF Hub, Sigstore, cosign features evolve quickly.
   - Add or rewrite: version/source/date or conditional wording.
   - Acceptance signal: version-sensitive claims are identifiable.
3. **Validate CI examples**
   - Reason: workflow snippets can rot.
   - Add or rewrite: check step IDs and digest outputs in GitHub Actions examples.
   - Acceptance signal: examples are statically coherent.

### `part4-data-and-storage/13-feature-vector-and-cache.md`

1. **Recast as derived-state system overview**
   - Reason: features, vector indexes, RAG, and caches share source/derived state.
   - Add or rewrite: source of truth, derived state, rebuild, alias, invalidation table.
   - Acceptance signal: 13a-13e reuse the state model.
2. **Add version invalidation matrix**
   - Reason: embedding/chunk/ACL/index/prompt/reranker changes have different actions.
   - Add or rewrite: rebuild, reindex, re-eval, cache flush rules.
   - Acceptance signal: every change has downstream action.
3. **Add end-to-end retrieval latency budget**
   - Reason: Part 13 must connect to serving performance.
   - Add or rewrite: embedding, vector, rerank, context build, LLM decode P95/P99.
   - Acceptance signal: 13c/13d/13e examples use common budget.

### `part4-data-and-storage/13a-feature-store.md`

1. **Make parity test concrete**
   - Reason: offline/online parity needs CI form.
   - Add or rewrite: sample selection, dtype tolerance, time boundary, late event, slice alert.
   - Acceptance signal: parity failure blocks release.
2. **Add online capacity worksheet**
   - Reason: online store choice needs QPS/P99/cost.
   - Add or rewrite: Redis/Cassandra/DynamoDB estimates.
   - Acceptance signal: worked example derives node count and cost.
3. **Add feature-store incident runbook**
   - Reason: lag/miss/schema/backfill incidents need actions.
   - Add or rewrite: metrics, mitigation, rollback, user impact.
   - Acceptance signal: on-call can handle freshness/P99 alerts.

### `part4-data-and-storage/13b-vector-index-algorithms.md`

1. **Correct HNSW memory example**
   - Reason: 100M x 768 fp32 raw vectors are about 307GB, conflicting with a 38GB example.
   - Add or rewrite: fp32/fp16/SQ/PQ, stored raw vector, graph overhead scenarios.
   - Acceptance signal: memory table matches raw vector math.
2. **Add benchmark protocol**
   - Reason: recall/latency claims need exact baseline and workload.
   - Add or rewrite: golden queries, ground truth, hardware, warmup, filters, read/write mix.
   - Acceptance signal: index benchmarks are reproducible.
3. **Add degradation and rebuild triggers**
   - Reason: ANN indexes degrade with updates/deletes.
   - Add or rewrite: tombstone ratio, recall drop, graph fragmentation, PQ drift thresholds.
   - Acceptance signal: rebuild is metric-triggered.

### `part4-data-and-storage/13c-vector-db-selection-and-operations.md`

1. **Repair serving-related links**
   - Reason: vLLM/SGLang/KV links may point to renamed targets.
   - Add or rewrite: verify all relative Markdown targets.
   - Acceptance signal: source links resolve.
2. **Add RPO/RTO restore drills**
   - Reason: backup commands are not enough.
   - Add or rewrite: snapshot frequency, restore drill, failover, rebuild-from-source targets.
   - Acceptance signal: vector DB recovery has measurable objectives.
3. **Strengthen ACL deletion propagation**
   - Reason: vector filters/cache/RAG citations can leak data.
   - Add or rewrite: permission-change SLO, pre-filter invariant, cache flush, tombstone propagation.
   - Acceptance signal: old permissions cannot be served after ACL update.

### `part4-data-and-storage/13d-rag-engineering.md`

1. **Make RAG eval CI executable**
   - Reason: golden queries need release gate semantics.
   - Add or rewrite: sample size, business slices, recall/faithfulness/citation thresholds, owner.
   - Acceptance signal: index/prompt changes can be blocked by CI.
2. **Add citation verification**
   - Reason: prompt instructions do not guarantee grounded claims.
   - Add or rewrite: evidence-span extraction, claim-support checks, no-evidence refusal.
   - Acceptance signal: key claims map to authorized document spans.
3. **Add RAG incident runbook**
   - Reason: recall drop, P99 spike, ACL leak, stale cache are common incidents.
   - Add or rewrite: monitoring, stop-the-bleeding actions, rollback, postmortem data.
   - Acceptance signal: canary failure has abort flow.

### `part4-data-and-storage/13e-embedding-and-cache-layer.md`

1. **Define cache key contract**
   - Reason: tenant/user/ACL/model/index/prompt/tool dimensions are scattered.
   - Add or rewrite: mandatory dimensions for embedding, semantic, RAG, and prefix caches.
   - Acceptance signal: version or permission changes cannot reuse stale cache.
2. **Add embedding serving capacity model**
   - Reason: model choice needs throughput/cost/latency.
   - Add or rewrite: GPU/CPU break-even, batch size, P95 queue, autoscaling worksheet.
   - Acceptance signal: example derives instance count and unit cost.
3. **Add embedding upgrade rollback**
   - Reason: query model, doc index, and cache must switch together.
   - Add or rewrite: dual write, shadow eval, index alias, cache flush, rollback order.
   - Acceptance signal: embedding space mismatch is prevented.

### `part5-serving-infra/14-online-inference-architecture.md`

1. **Add latency budget table**
   - Reason: serving chain lacks a fillable TTFT/TPOT/P99 budget.
   - Add or rewrite: gateway, router, engine, cache, RAG, downstream segments.
   - Acceptance signal: P99 TTFT can be localized to at least two layers.
2. **Define serving release unit**
   - Reason: model changes are not the only release risk.
   - Add or rewrite: model, tokenizer, prompt, config, router, engine, image.
   - Acceptance signal: any serving change can be routed to canary/rollback policy.
3. **Add route decision log**
   - Reason: routing failures need evidence.
   - Add or rewrite: tenant, model, route reason, cache hint, queue state, target replica.
   - Acceptance signal: bad routing can be explained post-incident.
4. **Differentiate short request, RAG, and agent sessions**
   - Reason: Chapter 25 changes capacity math.
   - Add or rewrite: three chain shapes and scaling implications.
   - Acceptance signal: reader does not autoscale agents by QPS alone.

### `part5-serving-infra/15-batching-scheduling-and-kv-cache.md`

1. **Unify KV capacity fields**
   - Reason: KV math must feed autoscaling and OOM diagnosis.
   - Add or rewrite: `kv_gb_per_seq`, `active_seq`, `prefix_hit_rate`, `eviction_rate`.
   - Acceptance signal: fields can be reused in 20c and 25.
2. **Add scheduler pseudocode**
   - Reason: continuous batching needs control-flow clarity.
   - Add or rewrite: admission, prefill, decode, preemption, eviction.
   - Acceptance signal: reader can map metric changes to scheduler steps.
3. **Add prefill/decode disaggregation gate**
   - Reason: KV handoff can become a bottleneck.
   - Add or rewrite: handoff bandwidth, serialization, locality, failure threshold.
   - Acceptance signal: chunked prefill versus full disaggregation can be decided.
4. **Add benchmark credibility protocol**
   - Reason: scheduling claims are workload-distribution sensitive.
   - Add or rewrite: prompt/output distribution, concurrency, warmup, goodput, cache state.
   - Acceptance signal: "30% faster" claims are reproducible.

### `part5-serving-infra/16-quantization-compilation-and-engines.md`

1. **Add quantization release gate**
   - Reason: accuracy, latency, cost, and rollback must be reviewed together.
   - Add or rewrite: PTQ/QAT/KV quantization gate and fallback path.
   - Acceptance signal: quantization cannot ship on offline score alone.
2. **Define engine artifact schema**
   - Reason: compiled engine files need shape contracts.
   - Add or rewrite: shape, dtype, batch, sequence, hardware, driver, engine version.
   - Acceptance signal: engine compatibility can be validated before loading.
3. **Remove engine-selection duplication**
   - Reason: selection tables appear in 16/16a/16b.
   - Add or rewrite: Chapter 16 is the entry decision; details live in internals chapters.
   - Acceptance signal: one authoritative selection matrix remains.
4. **Add quantization eval-set governance**
   - Reason: calibration/eval data quality determines risk.
   - Add or rewrite: per-tenant/domain eval sets and golden prompts.
   - Acceptance signal: accuracy loss can be localized by slice.

### `part5-serving-infra/16a-vllm-internals.md`

1. **Add version applicability notes**
   - Reason: vLLM internals change quickly.
   - Add or rewrite: version range and verification entry for internal mechanisms.
   - Acceptance signal: readers know what to revalidate after upgrades.
2. **Make tuning case reproducible**
   - Reason: large throughput gains need conditions.
   - Add or rewrite: hardware, model, token distribution, params, commands, before/after.
   - Acceptance signal: case can be rerun or explicitly marked illustrative.
3. **Map metrics to internals**
   - Reason: production debugging needs Prometheus/log field mapping.
   - Add or rewrite: vLLM metric -> Scheduler/BlockManager/KV/Worker table.
   - Acceptance signal: an alert maps to an internal component.
4. **Add Multi-LoRA serving boundaries**
   - Reason: adapter serving affects cost and isolation.
   - Add or rewrite: adapter cache, switch latency, tenant isolation, billing fields.
   - Acceptance signal: Multi-LoRA capacity can be estimated.

### `part5-serving-infra/16b-sglang-internals.md`

1. **Clarify DSL versus runtime**
   - Reason: SGLang can be misread as only a programming model.
   - Add or rewrite: frontend language, runtime scheduler, RadixAttention responsibility boundary.
   - Acceptance signal: reader can localize DSL versus runtime issues.
2. **Add Radix cache ledger**
   - Reason: prefix sharing should be measurable.
   - Add or rewrite: shared-prefix length, tree nodes, eviction, tenant cache key.
   - Acceptance signal: hit-rate drops can be diagnosed.
3. **Add tool-use runtime contract**
   - Reason: tool use is a security boundary.
   - Add or rewrite: tool schema, sandbox, credential scope, audit, approval.
   - Acceptance signal: tool calls are not controlled by prompt alone.
4. **Define fair agent benchmark**
   - Reason: comparing vLLM/SGLang depends on request decomposition.
   - Add or rewrite: agent workload benchmark conditions.
   - Acceptance signal: SGLang advantages are scoped to specific workloads.

### `part5-serving-infra/17-multitenancy-and-cost.md`

1. **Define tenant cost schema**
   - Reason: chargeback needs fields.
   - Add or rewrite: tenant_id, request_id, GPU-second, token, cache, egress, warm pool cost.
   - Acceptance signal: traces can generate showback/chargeback.
2. **Add budget execution state machine**
   - Reason: budget governance needs automated control actions.
   - Add or rewrite: burn, soft landing, rate limit, degrade, freeze.
   - Acceptance signal: monthly budget exhaustion has defined behavior.
3. **Connect fairness to queue/quota**
   - Reason: Ch17 and 20a overlap.
   - Add or rewrite: serving tenant policy -> queue/quota/priority mapping.
   - Acceptance signal: same policy explains online and offline resource control.
4. **Add noisy-neighbor drill**
   - Reason: isolation failure must be actionable.
   - Add or rewrite: cache, queue, GPU contention incident evidence and mitigation.
   - Acceptance signal: reader can distinguish isolation from capacity shortage.

### `part6-platform-and-orchestration/18-containers-and-runtime.md`

1. **Turn overview into chapter-group contract**
   - Reason: chapter is short but can serve as a strong map.
   - Add or rewrite: 18a-18d capability map and failure ownership.
   - Acceptance signal: reader knows which subchapter handles each runtime incident.
2. **Add runtime boundary matrix**
   - Reason: image/runtime/device/debug responsibilities are core.
   - Add or rewrite: symptom -> boundary -> evidence -> target chapter.
   - Acceptance signal: GPU invisible can be assigned to the right layer.
3. **Add cross-links**
   - Reason: container issues extend to K8s and security.
   - Add or rewrite: links to 19b, 19d, 23, appendix checklists.
   - Acceptance signal: overview is a troubleshooting entry point.

### `part6-platform-and-orchestration/18a-ai-images-and-cuda-compatibility.md`

1. **Productize compatibility matrix**
   - Reason: image baseline should be a platform object.
   - Add or rewrite: driver minimum, CUDA, PyTorch, engine, architecture, digest.
   - Acceptance signal: image can be admitted or rejected automatically.
2. **Split cold-start metrics**
   - Reason: pull, extract, weight load, engine init, and warmup differ.
   - Add or rewrite: per-stage metrics and thresholds.
   - Acceptance signal: cold start optimization localizes stage.
3. **Add CVE exception policy**
   - Reason: vulnerability exceptions need lifecycle.
   - Add or rewrite: owner, expiry, compensating control, blast radius.
   - Acceptance signal: high-risk exceptions expire.
4. **Add expected signals to validation commands**
   - Reason: commands alone are not runbooks.
   - Add or rewrite: expected pass/fail signals per command.
   - Acceptance signal: on-call can interpret output.

### `part6-platform-and-orchestration/18b-container-runtime-and-device-injection.md`

1. **Add CDI/hook/injection decision tree**
   - Reason: device injection strategies are easy to confuse.
   - Add or rewrite: CDI, runtime hook, env, mount injection selection table.
   - Acceptance signal: new device integrations choose the right path.
2. **Add minimum privilege baseline**
   - Reason: multi-node AI jobs should not default to privileged.
   - Add or rewrite: PodSecurity, securityContext, RBAC, SELinux/AppArmor baseline.
   - Acceptance signal: privileged is exception, not default.
3. **Standardize GPU/RDMA evidence bundle**
   - Reason: device issues span node and container.
   - Add or rewrite: `nvidia-smi`, `ldconfig`, `ibstat`, `nccl-tests` fields.
   - Acceptance signal: incident handoff uses text evidence, not screenshots.
4. **Add MIG injection validation**
   - Reason: MIG resource naming and isolation are nuanced.
   - Add or rewrite: profile, UUID, topology, tenant isolation checks.
   - Acceptance signal: MIG pod limitations are explicit.

### `part6-platform-and-orchestration/18c-artifact-supply-chain-and-image-governance.md`

1. **Add image promotion state machine**
   - Reason: digest/promotion should be gate-driven.
   - Add or rewrite: build -> scan -> sign -> stage -> prod -> retire.
   - Acceptance signal: production image is never rebuilt in place.
2. **Add SBOM/attestation example**
   - Reason: SLSA/SBOM needs concrete fields.
   - Add or rewrite: minimal attestation and verification command.
   - Acceptance signal: image provenance is answerable.
3. **Add distribution SLO**
   - Reason: image cache and registry health affect cold start.
   - Add or rewrite: pull P95, cache hit, registry errors, prewarm coverage.
   - Acceptance signal: cold start can be attributed to supply chain.
4. **Add risk acceptance lifecycle**
   - Reason: governance cannot be oral approval.
   - Add or rewrite: owner, deadline, blast radius, rollback.
   - Acceptance signal: risk exceptions are bounded.

### `part6-platform-and-orchestration/18d-runtime-troubleshooting.md`

1. **Standardize runtime evidence bundle**
   - Reason: this should become a shared template.
   - Add or rewrite: symptom, scope, node, image, driver, runtime, device, logs, actions.
   - Acceptance signal: 19d/20d can reuse it.
2. **Add one-page decision tree**
   - Reason: high-pressure incidents need an entry point.
   - Add or rewrite: GPU invisible, library failure, NCCL timeout, slow startup routes.
   - Acceptance signal: reader finds direction in three steps.
3. **Add escalation boundaries**
   - Reason: on-call should know when to stop restarting pods.
   - Add or rewrite: escalate to platform/network/security/hardware conditions.
   - Acceptance signal: incident handling has bounded retries.
4. **Clarify boundary with K8s SOP**
   - Reason: 18d and 19d overlap.
   - Add or rewrite: 18d owns node/runtime; 19d owns K8s object/control plane.
   - Acceptance signal: command tables are not duplicated.

### `part6-platform-and-orchestration/19-kubernetes-for-ai.md`

1. **Make this a Kubernetes group map**
   - Reason: chapter is short as standalone tutorial.
   - Add or rewrite: capabilities and navigation for 19a-19d.
   - Acceptance signal: readers treat it as control-plane map.
2. **Add runtime plane vs AI control plane matrix**
   - Reason: this is the chapter's strongest distinction.
   - Add or rewrite: object, owner, failure mode, evidence, SLO table.
   - Acceptance signal: Pod and TrainingJob failures are separated.
3. **Add output-oriented exercises**
   - Reason: current guide needs reader deliverables.
   - Add or rewrite: design questions for workload selection and control-plane boundaries.
   - Acceptance signal: readers produce a K8s AI workload plan.

### `part6-platform-and-orchestration/19a-kubernetes-ai-workloads.md`

1. **Add workload selection flow**
   - Reason: object selection needs execution path.
   - Add or rewrite: training, eval, inference, batch, service flows.
   - Acceptance signal: script migration chooses the right object.
2. **Add YAML production review rubric**
   - Reason: examples need review standards.
   - Add or rewrite: resources, probes, security, volumes, config, rollback checklist.
   - Acceptance signal: YAML examples can be PR-reviewed.
3. **Connect workload status to release gates**
   - Reason: Running/Succeeded is not Deployable.
   - Add or rewrite: workload status and release gate mapping.
   - Acceptance signal: job completion is not mistaken for production readiness.
4. **Add config/secret rollback**
   - Reason: config changes are releases.
   - Add or rewrite: config version, secret rotation, rollback example.
   - Acceptance signal: config changes are auditable.

### `part6-platform-and-orchestration/19b-gpu-scheduling-and-topology.md`

1. **Add topology benchmark gate**
   - Reason: locality decisions need measured proof.
   - Add or rewrite: `nvidia-smi topo`, NCCL, IB benchmark expected signals.
   - Acceptance signal: scheduling policy has performance evidence.
2. **Add GPU operator version plan**
   - Reason: driver/device plugin/operator drift breaks workloads.
   - Add or rewrite: version matrix and upgrade order.
   - Acceptance signal: upgrades protect workloads.
3. **Separate pending from slow placement**
   - Reason: capacity failure and topology-performance failure differ.
   - Add or rewrite: two troubleshooting paths.
   - Acceptance signal: "no card" and "slow card" are not mixed.
4. **Clarify boundary with GPU partitioning**
   - Reason: MIG/topology appears in 20b too.
   - Add or rewrite: 19b owns scheduling mechanics; 20b owns SKU/product policy.
   - Acceptance signal: duplicated content shrinks.

### `part6-platform-and-orchestration/19c-ai-crd-and-operators.md`

1. **Add CRD schema**
   - Reason: operator chapter needs realistic API object.
   - Add or rewrite: spec/status/conditions/events/finalizers YAML.
   - Acceptance signal: reader can implement a minimal operator.
2. **Add reconcile pseudocode**
   - Reason: production operators rely on idempotency and compensation.
   - Add or rewrite: retry, backoff, finalizer, partial failure handling.
   - Acceptance signal: controller failures do not duplicate resources.
3. **Add CRD upgrade contract**
   - Reason: API versions evolve.
   - Add or rewrite: v1alpha1 -> v1beta1 conversion/storage-version example.
   - Acceptance signal: old objects upgrade safely.
4. **Add responsibility matrix**
   - Reason: users, platform, operator, kubelet own different states.
   - Add or rewrite: RACI table.
   - Acceptance signal: incidents find owners.

### `part6-platform-and-orchestration/19d-kubernetes-ai-troubleshooting.md`

1. **Standardize SOP conclusion format**
   - Reason: postmortems need reusable outputs.
   - Add or rewrite: root cause, evidence, mitigation, prevention, owner.
   - Acceptance signal: SOP output can be pasted into incident review.
2. **Add evidence retention policy**
   - Reason: events/logs expire.
   - Add or rewrite: TTL, automatic collection, archive path.
   - Acceptance signal: incidents remain reviewable after 24 hours.
3. **Add cross-layer handoff conditions**
   - Reason: troubleshooting spans 18d, 20d, 21.
   - Add or rewrite: when to jump chapters.
   - Acceptance signal: on-call path does not dead-end.
4. **Add incident taxonomy**
   - Reason: platform needs aggregate reporting.
   - Add or rewrite: category, severity, blast radius, customer impact.
   - Acceptance signal: monthly incident review can group root causes.

### `part6-platform-and-orchestration/20-queues-quotas-and-autoscaling.md`

1. **Add capacity-control object model**
   - Reason: chapter is short but can define 20a-20d unifying problem.
   - Add or rewrite: queue, quota, resource shape, scaler, SOP objects.
   - Acceptance signal: readers see how subchapters compose.
2. **Add closed-loop capacity chain**
   - Reason: observe/admit/schedule/scale/recover are often separate.
   - Add or rewrite: causal chain across 20a-20d.
   - Acceptance signal: one capacity incident maps to all layers.
3. **Replace thin exercises with integrated design**
   - Reason: overview exercises should output a plan.
   - Add or rewrite: multi-tenant capacity governance design task.
   - Acceptance signal: reader produces a resource-governance plan.

### `part6-platform-and-orchestration/20a-queues-quotas-priority-and-fairness.md`

1. **Add numeric DRF example**
   - Reason: fairness should be computable.
   - Add or rewrite: three tenants across GPU/CPU/memory.
   - Acceptance signal: reader can calculate who is constrained.
2. **Add preemption recovery contract**
   - Reason: preemption must respect checkpoint and job semantics.
   - Add or rewrite: preemptible conditions, checkpoint freshness, resume SLA, responsibility.
   - Acceptance signal: preemption is not just kill.
3. **Define admission decision record**
   - Reason: users need explanation for blocked jobs.
   - Add or rewrite: queue, quota, priority, shape, wait reason, estimated start.
   - Acceptance signal: every reject/queue action is auditable.
4. **Link budget to priority**
   - Reason: cost governance and quota should align.
   - Add or rewrite: budget class -> queue priority mapping.
   - Acceptance signal: overspend affects admission consistently.

### `part6-platform-and-orchestration/20b-gpu-partitioning-and-sharing.md`

1. **Define GPU SKU contract**
   - Reason: GPU partitioning must be productized.
   - Add or rewrite: isolation, memory, compute, topology, price, eligible workloads.
   - Acceptance signal: tenants know what they receive.
2. **Add bin-packing example**
   - Reason: fragmentation should be explainable.
   - Add or rewrite: first-fit, best-fit, spread small calculation.
   - Acceptance signal: reader can predict fragmentation.
3. **Add isolation stress test**
   - Reason: MPS/time-slicing risk must be measured.
   - Add or rewrite: noisy-neighbor workload and interference thresholds.
   - Acceptance signal: sharing policy has safety boundary.
4. **Clarify boundary with 19b**
   - Reason: GPU scheduling and SKU policy overlap.
   - Add or rewrite: scheduling versus product/governance split.
   - Acceptance signal: duplicated MIG/topology content is reduced.

### `part6-platform-and-orchestration/20c-inference-autoscaling.md`

1. **Add scaler pseudocode**
   - Reason: SLO-aware scaling needs explicit inputs and outputs.
   - Add or rewrite: metric input, decision, cooldown, drain, min warm pool.
   - Acceptance signal: reader can implement a custom scaler.
2. **Define load profile matrix**
   - Reason: autoscaling must be tested on realistic traffic.
   - Add or rewrite: burst, long prompt, streaming, cache hit/miss, cold start.
   - Acceptance signal: scale tests cover tails and warm/cold cases.
3. **Add anti-oscillation rules**
   - Reason: autoscalers can flap.
   - Add or rewrite: hysteresis, cooldown, scale-down drain, min replicas.
   - Acceptance signal: replicas do not chase noise.
4. **Add degradation actions**
   - Reason: scale failures need user-experience protection.
   - Add or rewrite: rate limit, model downgrade, max token reduction, queue rejection.
   - Acceptance signal: capacity shortage has deterministic fallback.

### `part6-platform-and-orchestration/20d-capacity-and-troubleshooting-sop.md`

1. **Make this the capacity evidence ledger**
   - Reason: it can unify Part 5/6/7 capacity incidents.
   - Add or rewrite: queue, quota, shape, cold start, SLO, cost fields.
   - Acceptance signal: all capacity incidents use same table.
2. **Add SOP RACI**
   - Reason: actions cross teams.
   - Add or rewrite: owner, approver, escalation for each SOP.
   - Acceptance signal: actions have responsible owners.
3. **Add synthetic capacity drills**
   - Reason: SOPs need practice.
   - Add or rewrite: fragmentation, quota blocked, cold start, preemption recovery drills.
   - Acceptance signal: SOPs are rehearsed quarterly.
4. **Link dashboards**
   - Reason: symptoms should start from alerts/panels.
   - Add or rewrite: dashboard panels and alert names per symptom.
   - Acceptance signal: on-call can move from alert to SOP.

### `part7-reliability-security/21-observability-and-capacity.md`

1. **Define observability schema**
   - Reason: metrics/logs/traces need consistent labels.
   - Add or rewrite: tenant/model/version/request/release/node/engine labels and cardinality rules.
   - Acceptance signal: cross-service trace can aggregate by key dimensions.
2. **Add SLO burn-rate math**
   - Reason: error budget guidance needs formulas.
   - Add or rewrite: burn rate, window, threshold, PromQL example.
   - Acceptance signal: reader can configure one SLO alert.
3. **Add serving capacity case**
   - Reason: capacity section is thin.
   - Add or rewrite: 70B QPS/token/KV/warm-pool/cost estimate.
   - Acceptance signal: chapter derives replicas and budget.
4. **Add dashboard acceptance criteria**
   - Reason: dashboards must drive action.
   - Add or rewrite: owner, query, threshold, action per panel.
   - Acceptance signal: panels are not passive charts.

### `part7-reliability-security/22-evaluation-release-and-incident.md`

1. **Add release state machine**
   - Reason: release needs explicit states.
   - Add or rewrite: candidate -> offline pass -> shadow -> canary -> production -> rollback.
   - Acceptance signal: every state has entry/exit conditions.
2. **Layer quality gates**
   - Reason: AI release cannot rely on system health alone.
   - Add or rewrite: system metric, model quality, safety, cost, business guardrails.
   - Acceptance signal: canary pass requires more than 5xx.
3. **Add rollback drill**
   - Reason: rollback must be practiced.
   - Add or rewrite: model, prompt, config, router, engine, image rollback.
   - Acceptance signal: rollback RTO/RPO can be measured.
4. **Add AI incident taxonomy**
   - Reason: AI incidents include quality and cost failures.
   - Add or rewrite: quality regression, safety breach, cost runaway, latency, data leakage.
   - Acceptance signal: each incident type has detection and action.

### `part7-reliability-security/23-security-isolation-and-governance.md`

1. **Add threat model matrix**
   - Reason: security surface is broad.
   - Add or rewrite: data, model, runtime, RAG, tool, cache, supply chain, tenant threats.
   - Acceptance signal: every risk has control and evidence.
2. **Add policy-as-code examples**
   - Reason: governance should be enforceable.
   - Add or rewrite: OPA/Kyverno/RBAC/network policy/secrets examples.
   - Acceptance signal: unsafe configs can be blocked automatically.
3. **Add multi-tenant cache invariant**
   - Reason: RAG/prefix/tool cache can leak across authorization boundaries.
   - Add or rewrite: cache key must include tenant/auth/tool/prompt/model/index dimensions.
   - Acceptance signal: cache cannot bypass permissions.
4. **Add model artifact ingest policy**
   - Reason: pickle/unsafe weights are production risks.
   - Add or rewrite: signed artifact, allowed formats, exception process.
   - Acceptance signal: unsigned/untrusted weights cannot enter production.

### `part8-advanced-and-capstone/24-build-an-ai-platform.md`

1. **Add capstone task specification**
   - Reason: platform blueprint should become final project.
   - Add or rewrite: input constraints, deliverables, scoring rubric.
   - Acceptance signal: reader can submit a reviewable architecture spec.
2. **Define platform API objects**
   - Reason: platform design needs schemas.
   - Add or rewrite: TrainingJob, EvalRun, ModelVersion, Deployment, ServingEndpoint, TenantBudget.
   - Acceptance signal: blueprint can become an API design.
3. **Add end-to-end evidence trace**
   - Reason: this chapter should integrate the whole tutorial.
   - Add or rewrite: training -> eval -> registry -> deployment -> canary -> incident -> rollback.
   - Acceptance signal: every step links to Ch18-23 concepts.
4. **Add roadmap exit criteria**
   - Reason: platform phases need measurable completion.
   - Add or rewrite: milestone, owner, exit criteria, risk by phase.
   - Acceptance signal: platform build can be reviewed quarterly.

### `part8-advanced-and-capstone/25-agent-and-inference-time-compute.md`

1. **Define agent session schema**
   - Reason: agent runtime needs state objects.
   - Add or rewrite: session, step, budget, trace, tool result, resume token.
   - Acceptance signal: reader can implement a minimal runtime.
2. **Add budget enforcement pseudocode**
   - Reason: inference-time compute requires runtime control.
   - Add or rewrite: reservation, streaming deduction, settlement, degrade, abort.
   - Acceptance signal: budget exhaustion behavior is deterministic.
3. **Add tool sandbox contract**
   - Reason: tool use is a security and side-effect boundary.
   - Add or rewrite: allowlist, schema validation, egress, credential scope, approval, audit.
   - Acceptance signal: side-effect tools are controlled.
4. **Add agent capacity benchmark**
   - Reason: sessions differ from simple requests.
   - Add or rewrite: concurrent sessions, model calls, tool wait, reasoning tokens, retry.
   - Acceptance signal: GPU and tool-pool capacity can be estimated.
5. **Add trace replay and redaction**
   - Reason: agent failures are hard to reproduce.
   - Add or rewrite: step trace replay and sensitive-data redaction rules.
   - Acceptance signal: failed sessions can be debugged safely.

### `appendix/glossary.md`

1. **Add chapter backlinks**
   - Reason: glossary should route readers to mechanisms.
   - Add or rewrite: primary chapter and related chapters per term.
   - Acceptance signal: every important term links to source explanation.
2. **Add confused-term table**
   - Reason: several terms are commonly conflated.
   - Add or rewrite: utilization vs MFU, quota vs rate limit, checkpoint vs model package, cache vs Page Cache.
   - Acceptance signal: glossary corrects common misunderstandings.
3. **Mark version-sensitive terms**
   - Reason: implementations evolve.
   - Add or rewrite: version-sensitive/vendor-specific labels.
   - Acceptance signal: readers know what must be revalidated.
4. **Add acronym index**
   - Reason: AI Infra has dense abbreviations.
   - Add or rewrite: English acronym -> Chinese explanation index.
   - Acceptance signal: acronym search is useful.

### `appendix/tooling-map.md`

1. **Add freshness policy**
   - Reason: tool maps decay quickly.
   - Add or rewrite: last reviewed, version line, replacement candidates.
   - Acceptance signal: stale tool entries are detectable.
2. **Add selection dimensions**
   - Reason: current map is list-heavy.
   - Add or rewrite: maturity, ops burden, lock-in, observability, security, cost.
   - Acceptance signal: tool choice is not name-dropping.
3. **Map tools to chapters and tasks**
   - Reason: readers need task-to-tool navigation.
   - Add or rewrite: related chapters and typical tasks per tool.
   - Acceptance signal: troubleshooting can start from task.
4. **Mark production readiness**
   - Reason: lab tools and production tools differ.
   - Add or rewrite: lab-only, staging-ready, production-critical tags.
   - Acceptance signal: demo tools are not mistaken for production tools.

### `appendix/checklists.md`

1. **Make checklist items fielded**
   - Reason: questions alone do not create execution closure.
   - Add or rewrite: owner, phase, evidence, threshold, action fields.
   - Acceptance signal: checklist can enter approval workflow.
2. **Add chapter backlinks**
   - Reason: failed checklist items should lead to explanations.
   - Add or rewrite: related chapter per item.
   - Acceptance signal: review failures are teachable.
3. **Add pass/fail examples**
   - Reason: reviewers need consistent interpretation.
   - Add or rewrite: red/green examples for high-risk items.
   - Acceptance signal: different reviewers reach similar decisions.
4. **Expand incident checklist entries**
   - Reason: high-frequency platform incidents are not all covered.
   - Add or rewrite: GPU invisible, ImagePull, NCCL timeout, quota blocked, quality regression, security incident.
   - Acceptance signal: Part 6/7 incidents have checklist entry points.

### `appendix/answers.md`

1. **Split the answer book**
   - Reason: one very large file is hard to maintain.
   - Add or rewrite: split by part or chapter with a root index.
   - Acceptance signal: changing one chapter's exercises does not require editing a giant file.
2. **Add scoring rubrics**
   - Reason: design answers need self-assessment.
   - Add or rewrite: excellent / acceptable / insufficient signals.
   - Acceptance signal: learners know what quality level their answer reaches.
3. **Add source mapping**
   - Reason: answer and exercise drift is likely.
   - Add or rewrite: each answer records source file and exercise id.
   - Acceptance signal: renamed exercises remain traceable.
4. **Make assumptions explicit**
   - Reason: capacity/cost answers depend on hardware, prices, and versions.
   - Add or rewrite: list assumptions near each estimate or in a shared section.
   - Acceptance signal: readers can substitute parameters and recompute.

## 5. Cross-Cutting Capability Lines

### EvidenceBundle

- Current issue: many chapters provide commands and metrics, but there is no uniform definition of what evidence is required before claiming a root cause.
- Primary carriers:
  - `README.md`
  - `00-preface.md`
  - `part0-foundations-of-systems/0a8-cpu-worked-example.md`
  - `part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md`
  - `part0-foundations-of-systems/0c-filesystems-and-storage-internals.md`
  - `part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md`
  - `part2-systems-stack/06d-profiling-debugging-and-performance-sop.md`
  - `part3-training-infra/07-single-node-training.md`
  - `part3-training-infra/08-data-parallel.md`
  - `part3-training-infra/10-memory-checkpointing-and-recovery.md`
  - `part5-serving-infra/14-online-inference-architecture.md`
  - `part5-serving-infra/15-batching-scheduling-and-kv-cache.md`
  - `part6-platform-and-orchestration/18d-runtime-troubleshooting.md`
  - `part6-platform-and-orchestration/19d-kubernetes-ai-troubleshooting.md`
  - `part6-platform-and-orchestration/20d-capacity-and-troubleshooting-sop.md`
  - `part7-reliability-security/21-observability-and-capacity.md`
  - `appendix/checklists.md`
- Artifact to add: standardized evidence bundle fields: symptom, scope, time window, workload, versions, topology, metrics, logs, command outputs, hypothesis, action, retest, rollback.
- Acceptance signal: worked examples use baseline -> hypothesis -> evidence -> change -> retest -> rollback.

### CapacityLedger

- Current issue: capacity appears as tokens/s, GPU-hour, KV GB, queue wait, checkpoint bandwidth, cache size, and request cost, but fields are not unified.
- Primary carriers:
  - `part1-foundations/02-compute-storage-network.md`
  - `part2-systems-stack/04b-hbm-memory-and-roofline.md`
  - `part2-systems-stack/05a-memory-storage-hierarchy-and-data-residency.md`
  - `part2-systems-stack/05b-host-device-io-pcie-numa-and-overlap.md`
  - `part2-systems-stack/05c-rdma-collectives-and-cluster-topology.md`
  - `part2-systems-stack/05d-training-storage-checkpoint-and-io-diagnostics.md`
  - `part3-training-infra/07-single-node-training.md`
  - `part3-training-infra/08-data-parallel.md`
  - `part3-training-infra/09-model-pipeline-parallel.md`
  - `part3-training-infra/10-memory-checkpointing-and-recovery.md`
  - `part3-training-infra/10b-alignment-and-post-training.md`
  - `part3-training-infra/10c-finetuning-and-multi-adapter.md`
  - `part4-data-and-storage/13e-embedding-and-cache-layer.md`
  - `part5-serving-infra/14-online-inference-architecture.md`
  - `part5-serving-infra/15-batching-scheduling-and-kv-cache.md`
  - `part5-serving-infra/16-quantization-compilation-and-engines.md`
  - `part5-serving-infra/16a-vllm-internals.md`
  - `part5-serving-infra/16b-sglang-internals.md`
  - `part5-serving-infra/17-multitenancy-and-cost.md`
  - `part6-platform-and-orchestration/20c-inference-autoscaling.md`
  - `part6-platform-and-orchestration/20d-capacity-and-troubleshooting-sop.md`
  - `part7-reliability-security/21-observability-and-capacity.md`
  - `part8-advanced-and-capstone/25-agent-and-inference-time-compute.md`
- Artifact to add: worksheet with workload shape, hardware, utilization/goodput, storage, network, cache, queue, cost, and headroom.
- Acceptance signal: training, serving, RAG, and agent examples can reuse the same ledger fields.

### ReleaseUnit

- Current issue: model, tokenizer, prompt, adapter, engine, image, router, index, cache, and eval gate are discussed separately.
- Primary carriers:
  - `part1-foundations/03-from-model-to-production.md`
  - `part3-training-infra/10c-finetuning-and-multi-adapter.md`
  - `part4-data-and-storage/12a-model-registry.md`
  - `part4-data-and-storage/12b-checkpoint-engineering.md`
  - `part4-data-and-storage/12c-release-governance.md`
  - `part4-data-and-storage/12d-supply-chain-and-signing.md`
  - `part4-data-and-storage/13d-rag-engineering.md`
  - `part4-data-and-storage/13e-embedding-and-cache-layer.md`
  - `part5-serving-infra/14-online-inference-architecture.md`
  - `part5-serving-infra/16-quantization-compilation-and-engines.md`
  - `part5-serving-infra/16a-vllm-internals.md`
  - `part5-serving-infra/16b-sglang-internals.md`
  - `part7-reliability-security/22-evaluation-release-and-incident.md`
  - `part8-advanced-and-capstone/24-build-an-ai-platform.md`
- Artifact to add: release unit schema and state machine with immutable artifact ids, mutable aliases, approvals, gates, rollback targets, and audit log.
- Acceptance signal: a production release can identify every versioned component and roll it back.

### StateManifest

- Current issue: dataset, checkpoint, registry, index, feature, cache, and agent session state use similar patterns without one vocabulary.
- Primary carriers:
  - `part4-data-and-storage/11e-data-versioning-and-lineage.md`
  - `part4-data-and-storage/12-artifacts-and-checkpoints.md`
  - `part4-data-and-storage/12a-model-registry.md`
  - `part4-data-and-storage/12b-checkpoint-engineering.md`
  - `part4-data-and-storage/13-feature-vector-and-cache.md`
  - `part4-data-and-storage/13a-feature-store.md`
  - `part4-data-and-storage/13c-vector-db-selection-and-operations.md`
  - `part4-data-and-storage/13e-embedding-and-cache-layer.md`
  - `part6-platform-and-orchestration/19c-ai-crd-and-operators.md`
  - `part8-advanced-and-capstone/24-build-an-ai-platform.md`
  - `part8-advanced-and-capstone/25-agent-and-inference-time-compute.md`
- Artifact to add: manifest fields for immutable identity, alias, lineage, schema version, owner, status, timestamps, and validation results.
- Acceptance signal: every stateful object has identity, ownership, lifecycle, and restore behavior.

### RestoreLevel

- Current issue: resume, restore, warm start, rollback, and serving conversion are mixed.
- Primary carriers:
  - `part2-systems-stack/05d-training-storage-checkpoint-and-io-diagnostics.md`
  - `part3-training-infra/08-data-parallel.md`
  - `part3-training-infra/09-model-pipeline-parallel.md`
  - `part3-training-infra/10-memory-checkpointing-and-recovery.md`
  - `part4-data-and-storage/12b-checkpoint-engineering.md`
  - `part4-data-and-storage/13c-vector-db-selection-and-operations.md`
  - `part6-platform-and-orchestration/20a-queues-quotas-priority-and-fairness.md`
  - `part7-reliability-security/22-evaluation-release-and-incident.md`
- Artifact to add: restore-level taxonomy: true resume, same-shape restore, reshard restore, model-only warm start, serving conversion, rollback.
- Acceptance signal: every recovery path states what state is preserved and what SLA it can claim.

### CacheKeyContract

- Current issue: cache safety appears in RAG, semantic cache, prefix cache, Multi-LoRA, and tool use but lacks one invariant.
- Primary carriers:
  - `part4-data-and-storage/13c-vector-db-selection-and-operations.md`
  - `part4-data-and-storage/13d-rag-engineering.md`
  - `part4-data-and-storage/13e-embedding-and-cache-layer.md`
  - `part5-serving-infra/15-batching-scheduling-and-kv-cache.md`
  - `part5-serving-infra/16a-vllm-internals.md`
  - `part5-serving-infra/16b-sglang-internals.md`
  - `part5-serving-infra/17-multitenancy-and-cost.md`
  - `part7-reliability-security/23-security-isolation-and-governance.md`
  - `part8-advanced-and-capstone/25-agent-and-inference-time-compute.md`
- Artifact to add: mandatory cache key dimensions: tenant, auth/ACL scope, model/version, prompt/template, index, tool schema, adapter/base, quantization/runtime where relevant.
- Acceptance signal: cache reuse cannot bypass permission or version boundaries.

### TenantBudget

- Current issue: multi-tenant cost appears in serving and scheduling, but no single policy object links chargeback, quota, queue, and degradation.
- Primary carriers:
  - `part5-serving-infra/17-multitenancy-and-cost.md`
  - `part6-platform-and-orchestration/20a-queues-quotas-priority-and-fairness.md`
  - `part6-platform-and-orchestration/20b-gpu-partitioning-and-sharing.md`
  - `part6-platform-and-orchestration/20c-inference-autoscaling.md`
  - `part6-platform-and-orchestration/20d-capacity-and-troubleshooting-sop.md`
  - `part7-reliability-security/21-observability-and-capacity.md`
  - `part8-advanced-and-capstone/24-build-an-ai-platform.md`
  - `part8-advanced-and-capstone/25-agent-and-inference-time-compute.md`
- Artifact to add: tenant budget state machine and cost fields: token, GPU-second, cache, warm pool, storage, egress, queue priority, soft landing.
- Acceptance signal: budget exhaustion triggers deterministic actions rather than ad hoc throttling.

### BenchmarkProtocol

- Current issue: performance claims and worked examples vary in reproducibility.
- Primary carriers:
  - `part2-systems-stack/04a-gpu-execution-model-and-tensor-cores.md`
  - `part2-systems-stack/04b-hbm-memory-and-roofline.md`
  - `part2-systems-stack/05b-host-device-io-pcie-numa-and-overlap.md`
  - `part2-systems-stack/05c-rdma-collectives-and-cluster-topology.md`
  - `part2-systems-stack/06d-profiling-debugging-and-performance-sop.md`
  - `part3-training-infra/07-single-node-training.md`
  - `part3-training-infra/08-data-parallel.md`
  - `part3-training-infra/09-model-pipeline-parallel.md`
  - `part3-training-infra/10b-alignment-and-post-training.md`
  - `part3-training-infra/10c-finetuning-and-multi-adapter.md`
  - `part4-data-and-storage/13b-vector-index-algorithms.md`
  - `part4-data-and-storage/13c-vector-db-selection-and-operations.md`
  - `part5-serving-infra/14-online-inference-architecture.md`
  - `part5-serving-infra/15-batching-scheduling-and-kv-cache.md`
  - `part5-serving-infra/16-quantization-compilation-and-engines.md`
  - `part5-serving-infra/16a-vllm-internals.md`
  - `part5-serving-infra/16b-sglang-internals.md`
  - `part6-platform-and-orchestration/20c-inference-autoscaling.md`
  - `part7-reliability-security/21-observability-and-capacity.md`
- Artifact to add: hardware, software version, model, input distribution, warmup, cache state, command, metric definition, confidence window, counterfactual.
- Acceptance signal: a benchmark claim includes enough context to reproduce or reject.

## 6. Priority And Roadmap

### P0

- Repair Markdown source integrity: remove generated `.html` links, stale anchors, and wrong chapter references from source Markdown.
- Define global contracts: `EvidenceBundle`, `CapacityLedger`, `ReleaseUnit`, `StateManifest`, `RestoreLevel`, `CacheKeyContract`, `TenantBudget`.
- Fix high-risk factual/numeric issues: Part 4 HNSW memory sizing, checkpoint optimizer-state sizing, hardware/spec benchmark口径, and version-sensitive tool claims.
- Strengthen Part 7 into production-grade reliability/security/governance guidance.
- Add Part 8 capstone and agent runtime contracts so the tutorial ends with executable design capability.

### P1

- Add hands-on labs and acceptance artifacts for Part 2 profiling, H2D, NCCL, CUDA Graph, fusion, and hardware selection.
- Add Part 3 training blueprint with run manifests, evidence package, restore levels, and cost ledger.
- Add Part 4 state/version/cache/ACL contracts and restore/rollback runbooks.
- Add Part 5 serving benchmark protocol, latency budget, capacity ledger, release unit, and route evidence logs.
- Add Part 6 control-plane object models, admission logs, CRD schemas, GPU SKU contracts, scaler pseudocode, and capacity SOP RACI.
- Field appendix checklists with owner, phase, evidence, threshold, action, and chapter backlinks.

### P2

- Reduce duplicated first-principles prose where chapters repeat the same motivation without adding mechanisms.
- Split `appendix/answers.md` and add rubrics/source mapping.
- Add freshness notes to tools and version-sensitive implementation details.
- Improve cross-part reading paths for serving engineers, training platform engineers, SREs, security reviewers, and capstone learners.

### Wave 1: Source Integrity And Shared Contracts

- Scope: link repair, stale anchor cleanup, shared contract definitions, numeric evidence labels.
- Chapters:
  - `README.md`
  - `00-preface.md`
  - `part0-foundations-of-systems/0c-filesystems-and-storage-internals.md`
  - `part0-foundations-of-systems/0c1-vfs-inode-dentry-and-block-layer.md`
  - `part0-foundations-of-systems/0c2-local-filesystems-ext4-xfs-zfs.md`
  - `part0-foundations-of-systems/0c3-storage-semantics-fsync-direct-io-and-checkpoints.md`
  - `part0-foundations-of-systems/0c4-object-storage-parallel-filesystems-and-dataset-io.md`
  - `part0-foundations-of-systems/0d-network-stack-fundamentals.md`
  - `part0-foundations-of-systems/0d1-linux-network-stack-tcp-and-mtu.md`
  - `part0-foundations-of-systems/0d2-nic-offload-queues-and-service-network-io.md`
  - `part0-foundations-of-systems/0d3-rdma-roce-infiniband-and-gpudirect.md`
  - `part0-foundations-of-systems/0d3a-rdma-verbs-memory-registration-and-queues.md`
  - `part0-foundations-of-systems/0d3b-roce-infiniband-lossless-fabric-and-congestion.md`
  - `part0-foundations-of-systems/0d3c-gpudirect-rdma-gpu-nic-topology-and-diagnostics.md`
  - `part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md`
  - `part4-data-and-storage/12a-model-registry.md`
  - `part4-data-and-storage/12b-checkpoint-engineering.md`
  - `part4-data-and-storage/12d-supply-chain-and-signing.md`
  - `part4-data-and-storage/13b-vector-index-algorithms.md`
  - `part4-data-and-storage/13c-vector-db-selection-and-operations.md`
  - `appendix/glossary.md`
  - `appendix/checklists.md`
- Done when: source links resolve to Markdown, shared contracts exist, high-risk numeric claims are corrected or labeled.

### Wave 2: Systems Evidence And Capacity Depth

- Scope: Part 0/2 mechanism depth, profiling labs, evidence bundles, capacity worksheets.
- Chapters:
  - `part1-foundations/02-compute-storage-network.md`
  - `part0-foundations-of-systems/0a-cpu-microarchitecture.md`
  - `part0-foundations-of-systems/0a1-pipeline.md`
  - `part0-foundations-of-systems/0a2-out-of-order-execution.md`
  - `part0-foundations-of-systems/0a3-branch-prediction.md`
  - `part0-foundations-of-systems/0a4-simd.md`
  - `part0-foundations-of-systems/0a5-cache-hierarchy.md`
  - `part0-foundations-of-systems/0a6-mesi-coherence.md`
  - `part0-foundations-of-systems/0a7-false-sharing.md`
  - `part0-foundations-of-systems/0a8-cpu-worked-example.md`
  - `part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md`
  - `part0-foundations-of-systems/0b1-virtual-memory-page-tables-and-tlb.md`
  - `part0-foundations-of-systems/0b2-page-cache-writeback-and-huge-pages.md`
  - `part0-foundations-of-systems/0b3-numa-pcie-dma-and-pinned-memory.md`
  - `part0-foundations-of-systems/0b4-syscall-epoll-io-uring-and-service-io.md`
  - `part2-systems-stack/04-gpu-and-accelerators.md`
  - `part2-systems-stack/04a-gpu-execution-model-and-tensor-cores.md`
  - `part2-systems-stack/04b-hbm-memory-and-roofline.md`
  - `part2-systems-stack/04c-gpu-interconnect-and-systems.md`
  - `part2-systems-stack/04d-gpu-selection-virtualization-and-heterogeneous-accelerators.md`
  - `part2-systems-stack/05-memory-interconnect-io.md`
  - `part2-systems-stack/05a-memory-storage-hierarchy-and-data-residency.md`
  - `part2-systems-stack/05b-host-device-io-pcie-numa-and-overlap.md`
  - `part2-systems-stack/05c-rdma-collectives-and-cluster-topology.md`
  - `part2-systems-stack/05d-training-storage-checkpoint-and-io-diagnostics.md`
  - `part2-systems-stack/06-cuda-runtime-and-kernels.md`
  - `part2-systems-stack/06a-framework-dispatch-runtime-and-kernel-launch.md`
  - `part2-systems-stack/06b-streams-synchronization-and-cuda-graphs.md`
  - `part2-systems-stack/06c-kernel-libraries-fusion-and-sm-resource-limits.md`
  - `part2-systems-stack/06d-profiling-debugging-and-performance-sop.md`
  - `appendix/tooling-map.md`
- Done when: CPU/memory/storage/network/GPU/runtime troubleshooting examples all include evidence, formulas, and retest criteria.

### Wave 3: Training, Data, Artifact, And Serving Control Planes

- Scope: training manifests, data/artifact state contracts, release units, serving capacity ledger, benchmark protocol.
- Chapters:
  - `part3-training-infra/07-single-node-training.md`
  - `part3-training-infra/08-data-parallel.md`
  - `part3-training-infra/09-model-pipeline-parallel.md`
  - `part3-training-infra/10-memory-checkpointing-and-recovery.md`
  - `part3-training-infra/10b-alignment-and-post-training.md`
  - `part3-training-infra/10c-finetuning-and-multi-adapter.md`
  - `part4-data-and-storage/11-data-pipeline.md`
  - `part4-data-and-storage/11a-data-ingestion.md`
  - `part4-data-and-storage/11b-data-cleaning-dedup-quality.md`
  - `part4-data-and-storage/11c-tokenization-and-dataset-formats.md`
  - `part4-data-and-storage/11d-streaming-and-dataloader-engineering.md`
  - `part4-data-and-storage/11e-data-versioning-and-lineage.md`
  - `part4-data-and-storage/12-artifacts-and-checkpoints.md`
  - `part4-data-and-storage/12a-model-registry.md`
  - `part4-data-and-storage/12b-checkpoint-engineering.md`
  - `part4-data-and-storage/12c-release-governance.md`
  - `part4-data-and-storage/12d-supply-chain-and-signing.md`
  - `part4-data-and-storage/13-feature-vector-and-cache.md`
  - `part4-data-and-storage/13a-feature-store.md`
  - `part4-data-and-storage/13b-vector-index-algorithms.md`
  - `part4-data-and-storage/13c-vector-db-selection-and-operations.md`
  - `part4-data-and-storage/13d-rag-engineering.md`
  - `part4-data-and-storage/13e-embedding-and-cache-layer.md`
  - `part5-serving-infra/14-online-inference-architecture.md`
  - `part5-serving-infra/15-batching-scheduling-and-kv-cache.md`
  - `part5-serving-infra/16-quantization-compilation-and-engines.md`
  - `part5-serving-infra/16a-vllm-internals.md`
  - `part5-serving-infra/16b-sglang-internals.md`
  - `part5-serving-infra/17-multitenancy-and-cost.md`
- Done when: training and serving examples share fields for capacity, evidence, release, restore, and cost.

### Wave 4: Platform Reliability, Security, Governance, And Capstone

- Scope: platform object models, observability schema, release state machine, incident runbooks, threat model, policy-as-code, capstone specification, agent runtime contract.
- Chapters:
  - `part6-platform-and-orchestration/18-containers-and-runtime.md`
  - `part6-platform-and-orchestration/18a-ai-images-and-cuda-compatibility.md`
  - `part6-platform-and-orchestration/18b-container-runtime-and-device-injection.md`
  - `part6-platform-and-orchestration/18c-artifact-supply-chain-and-image-governance.md`
  - `part6-platform-and-orchestration/18d-runtime-troubleshooting.md`
  - `part6-platform-and-orchestration/19-kubernetes-for-ai.md`
  - `part6-platform-and-orchestration/19a-kubernetes-ai-workloads.md`
  - `part6-platform-and-orchestration/19b-gpu-scheduling-and-topology.md`
  - `part6-platform-and-orchestration/19c-ai-crd-and-operators.md`
  - `part6-platform-and-orchestration/19d-kubernetes-ai-troubleshooting.md`
  - `part6-platform-and-orchestration/20-queues-quotas-and-autoscaling.md`
  - `part6-platform-and-orchestration/20a-queues-quotas-priority-and-fairness.md`
  - `part6-platform-and-orchestration/20b-gpu-partitioning-and-sharing.md`
  - `part6-platform-and-orchestration/20c-inference-autoscaling.md`
  - `part6-platform-and-orchestration/20d-capacity-and-troubleshooting-sop.md`
  - `part7-reliability-security/21-observability-and-capacity.md`
  - `part7-reliability-security/22-evaluation-release-and-incident.md`
  - `part7-reliability-security/23-security-isolation-and-governance.md`
  - `part8-advanced-and-capstone/24-build-an-ai-platform.md`
  - `part8-advanced-and-capstone/25-agent-and-inference-time-compute.md`
  - `appendix/checklists.md`
  - `appendix/answers.md`
- Done when: readers can produce a platform design spec with capacity, SLO, cost, release, rollback, security, and incident artifacts.

## 7. Acceptance Criteria

**This spec is complete when:**

- No empty spec sections remain.
- Every source Markdown file in the audit scope has a `### \`path\`` chapter action entry in this spec.
- Every P0 item maps to at least one implementation wave.
- Every chapter-level action includes reason, concrete edit, and acceptance signal.
- Every cross-cutting capability line names exact Markdown paths for its primary carriers.
- Every roadmap wave names exact Markdown paths and has a clear completion condition.

**Future tutorial rewrites driven by this spec are acceptable only when changed chapters pass these gates:**

- Boundary gate: each changed chapter states what the topic is, what it is not, adjacent ownership boundaries, and handoff conditions.
- Path gate: each changed chapter includes at least one concrete control path, data path, state path, or failure path.
- Evidence gate: each diagnostic or operational claim names metrics, logs, traces, commands, configs, events, admission records, or dashboard fields that prove or falsify it.
- Model gate: each chapter that discusses capacity, performance, reliability, or cost includes a numerical model or explicit decision rule.
- Failure gate: each changed chapter includes symptoms, likely root causes, first actions, mitigations, rollback or recovery behavior, and escalation boundary.
- Worked-example gate: each worked example includes assumptions, baseline, hypothesis, evidence, action, retest, and rollback condition.
- Artifact gate: each changed chapter produces a reader artifact such as a runbook, sizing sheet, release decision record, checkpoint manifest, placement map, policy, or rubric.
- Future rewritten chapters should pass `rg -n "T[O]DO|T[B]D|F[I]XME|待[补]|后续[补]|这里不[展]开" <changed-files>` with no unresolved placeholders.
- Future rewritten chapters should pass `git diff --check -- <changed-files>`.
- Markdown source links should point to existing Markdown files, not generated HTML.
- Every benchmark or numeric gate should label source, assumption, hardware/version, input distribution, and metric definition where relevant.
- Appendix answers must be updated when exercises change, and every answer should map back to a source chapter and exercise id.
