# Part 3 Training Infra Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite Part 3 into a senior AI Infra engineering handbook while keeping the existing six chapter IDs.

**Architecture:** Six content subagents rewrite one Markdown chapter each with disjoint file ownership. The main agent performs two-stage review, regenerates `html/part3`, checks navigation/link integrity, and requests targeted revisions if a chapter stays generic.

**Tech Stack:** Markdown source files, Pandoc-generated static HTML, Mermaid diagrams, `html/assets/tutorial-data.js`, local shell validation with `rg`, `wc`, `git diff --check`, and a Node HTML link checker.

---

## File Structure

### Source Markdown

- Modify: `part3-training-infra/07-single-node-training.md`
- Modify: `part3-training-infra/08-data-parallel.md`
- Modify: `part3-training-infra/09-model-pipeline-parallel.md`
- Modify: `part3-training-infra/10-memory-checkpointing-and-recovery.md`
- Modify: `part3-training-infra/10b-alignment-and-post-training.md`
- Modify: `part3-training-infra/10c-finetuning-and-multi-adapter.md`

### Generated HTML

- Regenerate: `html/part3/07-single-node-training.html`
- Regenerate: `html/part3/08-data-parallel.html`
- Regenerate: `html/part3/09-model-pipeline-parallel.html`
- Regenerate: `html/part3/10-memory-checkpointing-and-recovery.html`
- Regenerate: `html/part3/10b-alignment-and-post-training.html`
- Regenerate: `html/part3/10c-finetuning-and-multi-adapter.html`

### Navigation

- Inspect only unless paths are missing: `html/assets/tutorial-data.js`

### Design Reference

- Read before implementation: `docs/superpowers/specs/2026-05-04-part3-training-infra-rewrite-design.md`

## Global Content Contract

Every chapter rewrite must include these sections or equivalent section titles:

- 第一性原理拆解 + 学习大纲
- 概念边界：是什么、不是什么、相邻概念边界
- 架构：控制路径、数据路径、状态路径、故障路径或责任边界
- 原理：从不可化简的问题推导机制
- 框架实现：map to real framework knobs and constraints
- 工程化落地：配置、版本矩阵、准入、preflight、发布、观测、治理
- 容量与效率：at least one formula or numeric model
- 故障排除：症状、证据、根因、动作表
- 方案设计 / Worked Example：real-scale numbers, decisions, trade-offs
- 反模式、Checklist、本章小结、练习题

The writing bar is senior AI Infra engineering. Avoid generic advice. Claims should be backed by metrics, commands, configuration, failure evidence, boundaries, or counterexamples.

## Task 1: Rewrite Chapter 7 Single-Node Training

**Files:**
- Modify: `part3-training-infra/07-single-node-training.md`

- [ ] **Step 1: Read the design and current chapter**

Run:

```bash
sed -n '1,260p' docs/superpowers/specs/2026-05-04-part3-training-infra-rewrite-design.md
sed -n '1,260p' part3-training-infra/07-single-node-training.md
```

Expected: Understand the global contract and the existing chapter shape.

- [ ] **Step 2: Rewrite the chapter**

Replace the chapter with a senior-engineer version covering:

- training step path: dataset read, CPU preprocessing, DataLoader worker, page cache, pinned memory, H2D, forward, loss, backward, optimizer, AMP, logging, checkpoint
- memory model: params, gradients, optimizer state, activations, temporary buffers, allocator fragmentation
- MFU, HFU, GPU utilization, SM occupancy, tokens/s boundaries
- AMP / BF16 / FP8 engineering trade-offs
- profiler chain: `torch.profiler`, Nsight Systems, Nsight Compute, DCGM, `iostat`, `perf`
- LLaMA-7B single-node 8xH100 worked example with memory, microbatch, gradient accumulation, throughput, MFU/HFU, and bottleneck diagnosis

Required concrete artifacts inside the chapter:

```text
Mermaid diagram: one full training-step timeline
Formula: memory budget = params + grads + optimizer states + activations + temp + fragmentation
Config example: PyTorch AMP/DataLoader/training loop or launcher snippet
Troubleshooting table: symptom, evidence, root cause, action
Worked example: LLaMA-7B on 8xH100
Checklist: single-node baseline acceptance checklist
```

- [ ] **Step 3: Self-check Chapter 7**

Run:

```bash
wc -l part3-training-infra/07-single-node-training.md
rg -n "是什么|不是什么|架构|原理|工程化|故障排除|方案设计|Worked Example|MFU|HFU|torch.profiler|Nsight|DCGM|LLaMA-7B" part3-training-infra/07-single-node-training.md
rg -n "TODO|FIXME|TBD|附加推演|补充推演|边界问题" part3-training-infra/07-single-node-training.md
```

Expected:

- line count is materially expanded from the previous 727 lines
- required concepts are found
- final `rg` has no output

## Task 2: Rewrite Chapter 8 Data Parallel

**Files:**
- Modify: `part3-training-infra/08-data-parallel.md`

- [ ] **Step 1: Read the design and current chapter**

Run:

```bash
sed -n '1,260p' docs/superpowers/specs/2026-05-04-part3-training-infra-rewrite-design.md
sed -n '1,260p' part3-training-infra/08-data-parallel.md
```

Expected: Understand the global contract and the existing chapter shape.

- [ ] **Step 2: Rewrite the chapter**

Replace the chapter with a senior-engineer version covering:

- DDP, FSDP, ZeRO boundaries: what is replicated, sharded, communicated, saved
- AllReduce, ReduceScatter, AllGather in the step timeline
- bucket, overlap, gradient accumulation, global batch, loss scale, straggler, data skew, topology
- NCCL ring/tree, rail, NIC, IB/RoCE, env vars, logs, and evidence
- decision boundary for DP vs FSDP/TP/PP/CP/hybrid parallel
- 8-node 64-GPU worked example with step time decomposition and NCCL/data-skew troubleshooting

Required concrete artifacts inside the chapter:

```text
Mermaid diagram: DDP/FSDP communication timeline
Formula: exposed communication = max(comm - overlap, 0)
Config example: torchrun/DDP or FSDP launcher and NCCL env snippet
Troubleshooting table: NCCL timeout, low bus bandwidth, rank straggler, data skew
Worked example: 8 nodes, 64 GPUs
Checklist: data-parallel production readiness
```

- [ ] **Step 3: Self-check Chapter 8**

Run:

```bash
wc -l part3-training-infra/08-data-parallel.md
rg -n "是什么|不是什么|架构|原理|工程化|故障排除|方案设计|Worked Example|DDP|FSDP|ZeRO|AllReduce|ReduceScatter|AllGather|NCCL|straggler" part3-training-infra/08-data-parallel.md
rg -n "TODO|FIXME|TBD|附加推演|补充推演|边界问题" part3-training-infra/08-data-parallel.md
```

Expected:

- line count is materially expanded from the previous 843 lines
- required concepts are found
- final `rg` has no output

## Task 3: Rewrite Chapter 9 Model and Pipeline Parallel

**Files:**
- Modify: `part3-training-infra/09-model-pipeline-parallel.md`

- [ ] **Step 1: Read the design and current chapter**

Run:

```bash
sed -n '1,260p' docs/superpowers/specs/2026-05-04-part3-training-infra-rewrite-design.md
sed -n '1,260p' part3-training-infra/09-model-pipeline-parallel.md
```

Expected: Understand that Chapter 9 is currently the thinnest Part 3 chapter and needs the most structural rewrite.

- [ ] **Step 2: Rewrite the chapter**

Replace the chapter with a senior-engineer version covering:

- TP, PP, SP, CP, EP, FSDP/ZeRO, 3D parallel, interleaved pipeline, zero bubble
- microbatch, pipeline bubble, virtual stage, activation placement, sequence/context partitioning
- strategy selection by model size, sequence length, GPU topology, NVLink/NVSwitch, IB/RoCE, framework support, checkpoint format, recovery
- Megatron-style configuration, DeepSpeed pipeline boundaries, FSDP hybrid sharding
- parallelism impact on checkpoint, optimizer state, failure recovery, inference conversion
- 70B and 405B worked examples with at least two parallel configurations compared

Required concrete artifacts inside the chapter:

```text
Mermaid diagram: 3D parallel placement across nodes and GPUs
Formula: pipeline bubble fraction or effective throughput with microbatches
Config example: Megatron-style TP/PP/CP/DP args
Troubleshooting table: OOM, bubble too high, TP communication bottleneck, bad placement, checkpoint mismatch
Worked example: 70B and 405B parallel strategy design
Checklist: parallel strategy design checklist
```

- [ ] **Step 3: Self-check Chapter 9**

Run:

```bash
wc -l part3-training-infra/09-model-pipeline-parallel.md
rg -n "是什么|不是什么|架构|原理|工程化|故障排除|方案设计|Worked Example|TP|PP|SP|CP|EP|FSDP|ZeRO|microbatch|pipeline bubble|Megatron|70B|405B" part3-training-infra/09-model-pipeline-parallel.md
rg -n "TODO|FIXME|TBD|附加推演|补充推演|边界问题" part3-training-infra/09-model-pipeline-parallel.md
```

Expected:

- line count is materially expanded from the previous 486 lines
- required concepts are found
- final `rg` has no output

## Task 4: Rewrite Chapter 10 Memory, Checkpointing, and Recovery

**Files:**
- Modify: `part3-training-infra/10-memory-checkpointing-and-recovery.md`

- [ ] **Step 1: Read the design and current chapter**

Run:

```bash
sed -n '1,260p' docs/superpowers/specs/2026-05-04-part3-training-infra-rewrite-design.md
sed -n '1,260p' part3-training-infra/10-memory-checkpointing-and-recovery.md
```

Expected: Understand that this chapter should become the reliability control plane for long training runs.

- [ ] **Step 2: Rewrite the chapter**

Replace or deeply restructure the chapter with a senior-engineer version covering:

- activation checkpointing, offload, optimizer state sharding, mixed precision, FP8, allocator fragmentation
- checkpoint as recovery protocol: contents, writer ownership, visibility, validation, cleanup, cross-parallelism restore
- checkpoint schema, sharded checkpoint, async checkpoint, atomic visibility, metadata, retention, RPO/RTO
- TorchElastic, elastic restart, preflight validation, straggler detection, NCCL hang troubleshooting
- consistency of model params, optimizer, scheduler, RNG, dataset cursor, global step, parallel metadata
- thousand-GPU interruption recovery worked example

Required concrete artifacts inside the chapter:

```text
Mermaid diagram: checkpoint write and restore state machine
Formula: checkpoint cost, RPO/RTO, or memory savings from activation checkpointing
Config example: sharded checkpoint or TorchElastic launcher/preflight snippet
Troubleshooting table: NCCL hang, corrupt checkpoint, slow checkpoint, restore mismatch, straggler
Worked example: thousand-GPU interruption recovery
Checklist: checkpoint and recovery production readiness
```

- [ ] **Step 3: Self-check Chapter 10**

Run:

```bash
wc -l part3-training-infra/10-memory-checkpointing-and-recovery.md
rg -n "是什么|不是什么|架构|原理|工程化|故障排除|方案设计|Worked Example|activation checkpoint|offload|FP8|checkpoint schema|RPO|RTO|TorchElastic|NCCL hang|straggler" part3-training-infra/10-memory-checkpointing-and-recovery.md
rg -n "TODO|FIXME|TBD|附加推演|补充推演|边界问题" part3-training-infra/10-memory-checkpointing-and-recovery.md
```

Expected:

- existing depth is preserved while structure becomes more senior-engineer oriented
- required concepts are found
- final `rg` has no output

## Task 5: Rewrite Chapter 10b Alignment and Post-Training Infra

**Files:**
- Modify: `part3-training-infra/10b-alignment-and-post-training.md`

- [ ] **Step 1: Read the design and current chapter**

Run:

```bash
sed -n '1,260p' docs/superpowers/specs/2026-05-04-part3-training-infra-rewrite-design.md
sed -n '1,260p' part3-training-infra/10b-alignment-and-post-training.md
```

Expected: Understand that PPO/RLHF must be written as system architecture, not only as algorithm.

- [ ] **Step 2: Rewrite the chapter**

Replace or deeply restructure the chapter with a senior-engineer version covering:

- pretraining, SFT, RM, PPO, DPO, GRPO system shapes
- actor, reference, reward, critic, rollout engine, training engine, sample generation, reward scoring, replay/buffer
- rollout/training throughput matching, resource split, checkpoint multi-model consistency
- evaluation gate, experiment tracking, data version, prompt/config version, failure recovery
- DPO/GRPO platformization trade-offs versus PPO
- LLaMA-7B or 70B PPO/RLHF worked example

Required concrete artifacts inside the chapter:

```text
Mermaid diagram: RLHF/PPO control and data paths
Formula: rollout throughput versus training consumption or reward scoring bottleneck
Config example: actor/ref/reward/critic resource layout
Troubleshooting table: reward latency, rollout backlog, inconsistent checkpoints, failed evaluation gate
Worked example: PPO/RLHF pipeline with GPU layout and bottleneck diagnosis
Checklist: post-training pipeline readiness
```

- [ ] **Step 3: Self-check Chapter 10b**

Run:

```bash
wc -l part3-training-infra/10b-alignment-and-post-training.md
rg -n "是什么|不是什么|架构|原理|工程化|故障排除|方案设计|Worked Example|SFT|Reward Model|PPO|DPO|GRPO|actor|reference|critic|rollout|checkpoint" part3-training-infra/10b-alignment-and-post-training.md
rg -n "TODO|FIXME|TBD|附加推演|补充推演|边界问题" part3-training-infra/10b-alignment-and-post-training.md
```

Expected:

- line count remains substantial and engineering depth improves
- required concepts are found
- final `rg` has no output

## Task 6: Rewrite Chapter 10c Fine-Tuning and Multi-Adapter Infra

**Files:**
- Modify: `part3-training-infra/10c-finetuning-and-multi-adapter.md`

- [ ] **Step 1: Read the design and current chapter**

Run:

```bash
sed -n '1,260p' docs/superpowers/specs/2026-05-04-part3-training-infra-rewrite-design.md
sed -n '1,260p' part3-training-infra/10c-finetuning-and-multi-adapter.md
```

Expected: Understand that this chapter must connect fine-tuning outputs to production inference and tenant governance.

- [ ] **Step 2: Rewrite the chapter**

Replace or deeply restructure the chapter with a senior-engineer version covering:

- full fine-tune, LoRA, QLoRA, DoRA resource exchange
- FTaaS control plane: data admission, queue, quota, image, base model constraints, adapter registry, approval, artifact release
- adapter/base compatibility: architecture, tokenizer, rank, target modules, quantization, license, safety policy
- merge deployment versus dynamic attach
- multi-LoRA serving, hot load, cache, A/B, rollback, permission, audit
- full path from training artifact to inference service
- multi-tenant LoRA platform worked example

Required concrete artifacts inside the chapter:

```text
Mermaid diagram: FTaaS control plane and artifact path to serving
Formula: adapter memory budget or tenant capacity model
Config example: adapter registry schema or serving hot-load config
Troubleshooting table: adapter incompatible, hot-load failure, fragmentation, quality regression, tenant isolation
Worked example: multi-tenant LoRA platform
Checklist: FTaaS and multi-adapter production readiness
```

- [ ] **Step 3: Self-check Chapter 10c**

Run:

```bash
wc -l part3-training-infra/10c-finetuning-and-multi-adapter.md
rg -n "是什么|不是什么|架构|原理|工程化|故障排除|方案设计|Worked Example|LoRA|QLoRA|DoRA|FTaaS|adapter registry|Multi-LoRA|hot load|A/B|rollback" part3-training-infra/10c-finetuning-and-multi-adapter.md
rg -n "TODO|FIXME|TBD|附加推演|补充推演|边界问题" part3-training-infra/10c-finetuning-and-multi-adapter.md
```

Expected:

- line count remains substantial and engineering depth improves
- required concepts are found
- final `rg` has no output

## Task 7: Main-Agent Stage 1 Review and Revision Loop

**Files:**
- Inspect: `part3-training-infra/*.md`
- Possibly request revisions from the original chapter subagents

- [ ] **Step 1: Run size and structure checks**

Run:

```bash
wc -l part3-training-infra/*.md
rg -n "^# |^## |^### " part3-training-infra/*.md
rg -n "是什么|不是什么|架构|原理|框架实现|工程化|容量|效率|故障排除|方案设计|Worked Example|Checklist" part3-training-infra/*.md
rg -n "TODO|FIXME|TBD|附加推演|补充推演|边界问题" part3-training-infra/*.md
```

Expected:

- each chapter has the global content contract
- no placeholder or filler markers
- `09` is no longer much thinner than the rest

- [ ] **Step 2: Review depth, not just keywords**

Open and read the worked example and troubleshooting sections in each chapter:

```bash
rg -n "Worked Example|故障排除|症状|证据|根因|动作|方案设计" part3-training-infra/*.md
```

Expected:

- examples include numbers, configuration, and trade-offs
- troubleshooting tables are actionable
- chapters do not only list concepts

- [ ] **Step 3: Request targeted revisions if needed**

If any chapter is generic, send the responsible subagent a specific revision request naming missing sections and expected fixes. Do not rewrite another subagent's chapter in the main thread unless the revision is small and local.

Expected:

- all chapter gaps are fixed before HTML regeneration

## Task 8: Regenerate Part 3 HTML

**Files:**
- Modify: `html/part3/07-single-node-training.html`
- Modify: `html/part3/08-data-parallel.html`
- Modify: `html/part3/09-model-pipeline-parallel.html`
- Modify: `html/part3/10-memory-checkpointing-and-recovery.html`
- Modify: `html/part3/10b-alignment-and-post-training.html`
- Modify: `html/part3/10c-finetuning-and-multi-adapter.html`
- Inspect: `html/assets/tutorial-data.js`

- [ ] **Step 1: Confirm Part 3 navigation entries exist**

Run:

```bash
rg -n '"id": "07"|"id": "08"|"id": "09"|"id": "10"|"id": "10b"|"id": "10c"' html/assets/tutorial-data.js
```

Expected: all six chapter IDs are present.

- [ ] **Step 2: Regenerate HTML from Markdown**

Use the repo's existing static HTML shell pattern: remove top Markdown `h1`, convert Mermaid blocks to `<pre class="mermaid">`, convert local Markdown links to HTML links, build a TOC from `h2`, and include:

```html
<script src="../assets/tutorial-data.js"></script>
<script src="../assets/mermaid.min.js"></script>
<script src="../assets/nav.js"></script>
<script>mermaid.initialize({ startOnLoad: true, theme: 'neutral' });</script>
```

Expected:

- all six `html/part3/*.html` pages are regenerated from the updated Markdown
- sidebar `current` IDs match `07`, `08`, `09`, `10`, `10b`, `10c`

- [ ] **Step 3: Inspect one generated page**

Run:

```bash
sed -n '1,120p' html/part3/09-model-pipeline-parallel.html
tail -n 40 html/part3/09-model-pipeline-parallel.html
```

Expected:

- page has the standard shell, hero, TOC, content, bottom nav, footer, and scripts

## Task 9: Main-Agent Stage 2 Validation

**Files:**
- Inspect: `part3-training-infra`
- Inspect: `html/part3`
- Inspect: `html/assets/tutorial-data.js`

- [ ] **Step 1: Run diff hygiene checks**

Run:

```bash
git diff --check -- part3-training-infra html/part3 html/assets/tutorial-data.js
```

Expected: no output.

- [ ] **Step 2: Run content residue checks**

Run:

```bash
rg -n "\\.md([#\"])|<pre class=\"mermaid\"><code|TODO|FIXME|TBD|附加推演|补充推演|边界问题" part3-training-infra html/part3
```

Expected: no output.

- [ ] **Step 3: Run HTML shell checks**

Run:

```bash
node <<'NODE'
const fs = require('fs');
const files = fs.readdirSync('html/part3').filter(f => f.endsWith('.html')).sort();
let bad = [];
for (const f of files) {
  const s = fs.readFileSync(`html/part3/${f}`, 'utf8');
  for (const token of ['<section class=\"toc\">', '<script src=\"../assets/tutorial-data.js\">', '<script src=\"../assets/nav.js\">']) {
    if (!s.includes(token)) bad.push(`${f}: missing ${token}`);
  }
  if (/```mermaid|language-mermaid|sourceCode mermaid/.test(s)) bad.push(`${f}: unconverted mermaid`);
}
console.log(`checked=${files.length}`);
console.log(`issues=${bad.length}`);
if (bad.length) console.log(bad.join('\\n'));
NODE
```

Expected:

```text
checked=6
issues=0
```

- [ ] **Step 4: Run local HTML link checker**

Run:

```bash
node <<'NODE'
const fs = require('fs');
const path = require('path');
const root = path.resolve('html');
const files = fs.readdirSync('html/part3').filter(f => f.endsWith('.html')).map(f => path.resolve('html/part3', f));
let broken = [];
for (const file of files) {
  const html = fs.readFileSync(file, 'utf8');
  const ids = new Set([...html.matchAll(/\\sid=\"([^\"]+)\"/g)].map(m => m[1]));
  for (const m of html.matchAll(/href=\"([^\"]+)\"/g)) {
    const href = m[1];
    if (/^(https?:|mailto:|#|javascript:)/.test(href)) continue;
    const [rawTarget, hash] = href.split('#');
    if (!rawTarget) {
      if (hash && !ids.has(hash)) broken.push(`${path.relative(root, file)} -> #${hash}`);
      continue;
    }
    const target = path.resolve(path.dirname(file), rawTarget);
    if (!target.startsWith(root)) continue;
    if (!fs.existsSync(target)) broken.push(`${path.relative(root, file)} -> ${href}`);
    else if (hash && target === file && !ids.has(hash)) broken.push(`${path.relative(root, file)} -> ${href}`);
  }
}
console.log(`checked=${files.length}`);
console.log(`broken=${broken.length}`);
if (broken.length) console.log(broken.join('\\n'));
NODE
```

Expected:

```text
checked=6
broken=0
```

- [ ] **Step 5: Summarize changed files**

Run:

```bash
git diff --stat -- part3-training-infra html/part3 html/assets/tutorial-data.js
git status --short -- part3-training-infra html/part3 html/assets/tutorial-data.js
```

Expected: only Part 3 Markdown/HTML and possibly `tutorial-data.js` are relevant to this plan.

## Task 10: Final Review Report

**Files:**
- No file edits unless validation found issues

- [ ] **Step 1: Prepare final report**

Summarize:

- six Markdown chapters rewritten
- HTML regenerated
- navigation checked
- validation commands and results
- any residual risks or files not touched

- [ ] **Step 2: Do not leave running agents**

Close or wait on all subagents used for this plan.

Expected: no active subagent sessions remain for Part 3 rewrite.
