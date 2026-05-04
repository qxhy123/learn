# AI Infra Tutorial Wave 1 Source Integrity And Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. User constraint for subagent execution: use `gpt-5.5` with `xhigh` reasoning for all subagents, with at most 5 subagents running concurrently.

**Goal:** Implement Wave 1 from the AI Infra tutorial improvement spec: repair Markdown source integrity, define shared cross-cutting contracts, and label or correct high-risk numeric/version-sensitive claims.

**Architecture:** Treat Wave 1 as a documentation-foundation pass. First add the shared contracts and source policy, then repair generated HTML links in Markdown source, then normalize high-risk numeric claims in checkpoint and vector-index material, and finally tighten appendix lookup/checklist entries so later waves can reuse the same vocabulary.

**Tech Stack:** Markdown, shell, `rg`, `perl`, `git diff --check`.

---

## Source References

- Spec: `docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md`
- Current tutorial source root: `README.md`, `00-preface.md`, `part0-foundations-of-systems/`, `part4-data-and-storage/`, `appendix/`
- Generated HTML is not source of truth. Do not edit `html/` in this wave.

## File Structure

Read:

- `docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md`
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

Modify:

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

Use this exact Wave 1 file list for final verification and any cleanup commit:

```bash
WAVE1_FILES=(
  README.md
  00-preface.md
  part0-foundations-of-systems/0c-filesystems-and-storage-internals.md
  part0-foundations-of-systems/0c1-vfs-inode-dentry-and-block-layer.md
  part0-foundations-of-systems/0c2-local-filesystems-ext4-xfs-zfs.md
  part0-foundations-of-systems/0c3-storage-semantics-fsync-direct-io-and-checkpoints.md
  part0-foundations-of-systems/0c4-object-storage-parallel-filesystems-and-dataset-io.md
  part0-foundations-of-systems/0d-network-stack-fundamentals.md
  part0-foundations-of-systems/0d1-linux-network-stack-tcp-and-mtu.md
  part0-foundations-of-systems/0d2-nic-offload-queues-and-service-network-io.md
  part0-foundations-of-systems/0d3-rdma-roce-infiniband-and-gpudirect.md
  part0-foundations-of-systems/0d3a-rdma-verbs-memory-registration-and-queues.md
  part0-foundations-of-systems/0d3b-roce-infiniband-lossless-fabric-and-congestion.md
  part0-foundations-of-systems/0d3c-gpudirect-rdma-gpu-nic-topology-and-diagnostics.md
  part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md
  part4-data-and-storage/12a-model-registry.md
  part4-data-and-storage/12b-checkpoint-engineering.md
  part4-data-and-storage/12d-supply-chain-and-signing.md
  part4-data-and-storage/13b-vector-index-algorithms.md
  part4-data-and-storage/13c-vector-db-selection-and-operations.md
  appendix/glossary.md
  appendix/checklists.md
)
```

Do not modify:

- `html/`
- `appendix/answers.md`
- Part 1, Part 2, Part 3, Part 5, Part 6, Part 7, or Part 8 chapters in this wave.

## Task 1: Add The Global Source Policy And Contract Vocabulary

**Files:**

- Modify: `README.md`
- Modify: `00-preface.md`
- Modify: `appendix/glossary.md`
- Modify: `appendix/checklists.md`

- [ ] **Step 1: Inspect current policy and contract gaps**

Run:

```bash
rg -n "source of truth|Markdown|HTML|EvidenceBundle|CapacityLedger|ReleaseUnit|StateManifest|RestoreLevel|CacheKeyContract|TenantBudget|BenchmarkProtocol|证据包|容量账本|发布单元|状态清单|恢复等级|缓存键|租户预算|基准协议" README.md 00-preface.md appendix/glossary.md appendix/checklists.md
```

Expected: output may include ordinary Markdown or HTML wording, but should not show a coherent definition block for all eight contracts.

- [ ] **Step 2: Add source-of-truth policy to `README.md`**

Add a short subsection near the project introduction or before the chapter navigation:

```markdown
## 源稿与生成物约定

本仓库以 Markdown 源稿为唯一事实来源。教程正文链接应指向 `.md` 文件；`html/` 目录是发布构建产物，不作为改稿依据。除非章节专门说明浏览器访问方式，正文不应把生成的 `.html` 文件作为内部章节链接。

后续改稿必须先更新 Markdown，再由构建流程生成 HTML。检查内部链接时，以 Markdown 文件是否存在为准。
```

Acceptance signal: `README.md` states Markdown source policy without changing existing chapter navigation semantics.

- [ ] **Step 3: Add evidence-first reading rule to `00-preface.md`**

Add a subsection after "本教程的设计理念" or before "阅读时应始终问自己的几个问题":

```markdown
## 证据优先的阅读规则

本教程后续所有排障、容量和发布判断都遵循同一条规则：

> 没有指标、日志、trace、配置、命令输出和复测结果，就不能把一个判断称为诊断结论。

每次阅读 worked example 或 SOP 时，都应主动补齐以下字段：

| 字段 | 含义 |
|------|------|
| symptom | 观察到的症状和影响范围 |
| scope | 受影响的模型、任务、租户、节点或时间窗口 |
| workload | batch、token、QPS、向量规模、checkpoint 大小等负载形态 |
| version | 模型、数据、镜像、驱动、框架、引擎和配置版本 |
| evidence | 指标、日志、trace、命令输出、manifest 或审计事件 |
| hypothesis | 当前最可能的解释 |
| action | 执行的改动、降级、回滚或隔离动作 |
| retest | 复测命令、对照窗口和通过阈值 |
| rollback | 改动无效或副作用过大时的回退条件 |
```

Acceptance signal: preface gives a reusable `EvidenceBundle` mental model and later chapters can reference it.

- [ ] **Step 4: Add shared contract glossary entries**

Append a new glossary section before "使用建议":

```markdown
## J. 跨章节工程契约

| 术语 | 简要解释 |
|------|----------|
| EvidenceBundle | 一次诊断或发布判断所需的最小证据包，包含 symptom、scope、workload、version、evidence、hypothesis、action、retest 和 rollback。 |
| CapacityLedger | 跨训练、推理、RAG、平台和成本治理复用的容量账本，记录 workload shape、硬件、利用率、goodput、存储、网络、缓存、队列、成本和 headroom。 |
| ReleaseUnit | 一次可审计发布的最小单元，绑定模型、tokenizer、prompt、adapter、engine、image、router、index、cache、eval gate 和 rollback target。 |
| StateManifest | 描述数据集、checkpoint、模型版本、索引、缓存或 agent session 的状态清单，至少包含 immutable id、alias、lineage、schema version、owner、status、timestamp 和 validation result。 |
| RestoreLevel | 描述恢复语义的等级，包括 true resume、same-shape restore、reshard restore、model-only warm start、serving conversion 和 rollback。 |
| CacheKeyContract | 缓存复用必须满足的键空间约束，至少绑定 tenant、ACL/auth scope、model/version、prompt/template、index、tool schema、adapter/base 和 runtime 口径。 |
| TenantBudget | 租户级预算与降级策略对象，记录 token、GPU-second、cache、warm pool、storage、egress、queue priority 和 soft landing 动作。 |
| BenchmarkProtocol | 性能数字的复现协议，记录 hardware、software version、model、input distribution、warmup、cache state、command、metric definition、confidence window 和 counterfactual。 |
```

Acceptance signal: `appendix/glossary.md` defines all eight contracts with exact names used in the spec.

- [ ] **Step 5: Convert appendix checklist items into evidence-bearing gates**

Add a new section near the top of `appendix/checklists.md`, after the title:

```markdown
## 通用证据字段

任何上线、排障、容量、发布或安全检查项都应尽量落到以下字段，而不是只回答"是否完成"：

| 字段 | 必填性 | 示例 |
|------|--------|------|
| owner | 必填 | training-platform、serving-sre、security-reviewer |
| phase | 必填 | preflight、canary、production、incident、rollback |
| evidence | 必填 | dashboard link、manifest id、command output、audit event |
| threshold | 条件必填 | P99 < 200ms、GPU ECC error = 0、Recall@10 >= 0.92 |
| action | 必填 | proceed、block、rollback、degrade、page owner |
| retest | 条件必填 | rerun benchmark、replay golden queries、restart canary window |
```

Then add one bullet to each existing checklist section that asks for the evidence artifact, for example:

```markdown
- 是否记录 owner、phase、evidence、threshold、action 和 retest 字段，便于后续复盘和审计
```

Acceptance signal: every checklist section has at least one evidence-bearing entry and the top-level field table exists.

- [ ] **Step 6: Verify Task 1**

Run:

```bash
rg -n "EvidenceBundle|CapacityLedger|ReleaseUnit|StateManifest|RestoreLevel|CacheKeyContract|TenantBudget|BenchmarkProtocol" README.md 00-preface.md appendix/glossary.md appendix/checklists.md
rg -n "source of truth|Markdown 源稿|\\.html.*内部章节链接|生成物约定" README.md
git diff --check -- README.md 00-preface.md appendix/glossary.md appendix/checklists.md
```

Expected: first command shows all eight contracts in `appendix/glossary.md`; second command shows source policy in `README.md`; third command exits 0.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git add README.md 00-preface.md appendix/glossary.md appendix/checklists.md
git commit -m "Add AI infra shared documentation contracts"
```

Expected: commit includes only the four Task 1 files.

## Task 2: Repair Part 0 Markdown Links

**Files:**

- Modify: `part0-foundations-of-systems/0c-filesystems-and-storage-internals.md`
- Modify: `part0-foundations-of-systems/0c1-vfs-inode-dentry-and-block-layer.md`
- Modify: `part0-foundations-of-systems/0c2-local-filesystems-ext4-xfs-zfs.md`
- Modify: `part0-foundations-of-systems/0c3-storage-semantics-fsync-direct-io-and-checkpoints.md`
- Modify: `part0-foundations-of-systems/0c4-object-storage-parallel-filesystems-and-dataset-io.md`
- Modify: `part0-foundations-of-systems/0d-network-stack-fundamentals.md`
- Modify: `part0-foundations-of-systems/0d1-linux-network-stack-tcp-and-mtu.md`
- Modify: `part0-foundations-of-systems/0d2-nic-offload-queues-and-service-network-io.md`
- Modify: `part0-foundations-of-systems/0d3-rdma-roce-infiniband-and-gpudirect.md`
- Modify: `part0-foundations-of-systems/0d3a-rdma-verbs-memory-registration-and-queues.md`
- Modify: `part0-foundations-of-systems/0d3b-roce-infiniband-lossless-fabric-and-congestion.md`
- Modify: `part0-foundations-of-systems/0d3c-gpudirect-rdma-gpu-nic-topology-and-diagnostics.md`
- Modify: `part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md`

- [ ] **Step 1: Capture current generated HTML links**

Run:

```bash
rg -n "\]\([^)]*\.html[)#]?" \
  part0-foundations-of-systems/0c*.md \
  part0-foundations-of-systems/0d*.md
```

Expected before changes: output includes the 0c/0d files listed in this task. Save the output in the task notes or commit message body if useful.

- [ ] **Step 2: Replace generated HTML links with Markdown links**

For every internal Part 0 chapter link ending in `.html`, replace it with the corresponding `.md` link. Use the exact mapping below:

```text
0b2-page-cache-writeback-and-huge-pages.html -> 0b2-page-cache-writeback-and-huge-pages.md
0b3-numa-pcie-dma-and-pinned-memory.html -> 0b3-numa-pcie-dma-and-pinned-memory.md
0b4-syscall-epoll-io-uring-and-service-io.html -> 0b4-syscall-epoll-io-uring-and-service-io.md
0c-filesystems-and-storage-internals.html -> 0c-filesystems-and-storage-internals.md
0c1-vfs-inode-dentry-and-block-layer.html -> 0c1-vfs-inode-dentry-and-block-layer.md
0c2-local-filesystems-ext4-xfs-zfs.html -> 0c2-local-filesystems-ext4-xfs-zfs.md
0c3-storage-semantics-fsync-direct-io-and-checkpoints.html -> 0c3-storage-semantics-fsync-direct-io-and-checkpoints.md
0c4-object-storage-parallel-filesystems-and-dataset-io.html -> 0c4-object-storage-parallel-filesystems-and-dataset-io.md
0d-network-stack-fundamentals.html -> 0d-network-stack-fundamentals.md
0d1-linux-network-stack-tcp-and-mtu.html -> 0d1-linux-network-stack-tcp-and-mtu.md
0d2-nic-offload-queues-and-service-network-io.html -> 0d2-nic-offload-queues-and-service-network-io.md
0d3-rdma-roce-infiniband-and-gpudirect.html -> 0d3-rdma-roce-infiniband-and-gpudirect.md
0d3a-rdma-verbs-memory-registration-and-queues.html -> 0d3a-rdma-verbs-memory-registration-and-queues.md
0d3b-roce-infiniband-lossless-fabric-and-congestion.html -> 0d3b-roce-infiniband-lossless-fabric-and-congestion.md
0d3c-gpudirect-rdma-gpu-nic-topology-and-diagnostics.html -> 0d3c-gpudirect-rdma-gpu-nic-topology-and-diagnostics.md
0d4-nccl-collectives-and-network-diagnostics.html -> 0d4-nccl-collectives-and-network-diagnostics.md
```

Do not change external links such as `https://example.com/file.html`.

- [ ] **Step 3: Verify linked Markdown files exist**

Run:

```bash
perl -nE 'while (/\]\(([^)#]+\.md)(?:#[^)]+)?\)/g) { say $1 }' \
  part0-foundations-of-systems/0c*.md \
  part0-foundations-of-systems/0d*.md \
  | sort -u \
  | while IFS= read -r p; do
      case "$p" in
        ./*) test -f "part0-foundations-of-systems/${p#./}" || echo "missing $p" ;;
        ../*) test -f "part0-foundations-of-systems/$p" || echo "missing $p" ;;
        *) test -f "part0-foundations-of-systems/$p" || echo "missing $p" ;;
      esac
    done
```

Expected: no `missing` lines.

- [ ] **Step 4: Verify no generated Part 0 HTML links remain**

Run:

```bash
rg -n "\]\([^)]*\.html[)#]?" \
  part0-foundations-of-systems/0c*.md \
  part0-foundations-of-systems/0d*.md
```

Expected: no output.

- [ ] **Step 5: Run whitespace check**

Run:

```bash
git diff --check -- \
  part0-foundations-of-systems/0c-filesystems-and-storage-internals.md \
  part0-foundations-of-systems/0c1-vfs-inode-dentry-and-block-layer.md \
  part0-foundations-of-systems/0c2-local-filesystems-ext4-xfs-zfs.md \
  part0-foundations-of-systems/0c3-storage-semantics-fsync-direct-io-and-checkpoints.md \
  part0-foundations-of-systems/0c4-object-storage-parallel-filesystems-and-dataset-io.md \
  part0-foundations-of-systems/0d-network-stack-fundamentals.md \
  part0-foundations-of-systems/0d1-linux-network-stack-tcp-and-mtu.md \
  part0-foundations-of-systems/0d2-nic-offload-queues-and-service-network-io.md \
  part0-foundations-of-systems/0d3-rdma-roce-infiniband-and-gpudirect.md \
  part0-foundations-of-systems/0d3a-rdma-verbs-memory-registration-and-queues.md \
  part0-foundations-of-systems/0d3b-roce-infiniband-lossless-fabric-and-congestion.md \
  part0-foundations-of-systems/0d3c-gpudirect-rdma-gpu-nic-topology-and-diagnostics.md \
  part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md
```

Expected: exit 0.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add \
  part0-foundations-of-systems/0c-filesystems-and-storage-internals.md \
  part0-foundations-of-systems/0c1-vfs-inode-dentry-and-block-layer.md \
  part0-foundations-of-systems/0c2-local-filesystems-ext4-xfs-zfs.md \
  part0-foundations-of-systems/0c3-storage-semantics-fsync-direct-io-and-checkpoints.md \
  part0-foundations-of-systems/0c4-object-storage-parallel-filesystems-and-dataset-io.md \
  part0-foundations-of-systems/0d-network-stack-fundamentals.md \
  part0-foundations-of-systems/0d1-linux-network-stack-tcp-and-mtu.md \
  part0-foundations-of-systems/0d2-nic-offload-queues-and-service-network-io.md \
  part0-foundations-of-systems/0d3-rdma-roce-infiniband-and-gpudirect.md \
  part0-foundations-of-systems/0d3a-rdma-verbs-memory-registration-and-queues.md \
  part0-foundations-of-systems/0d3b-roce-infiniband-lossless-fabric-and-congestion.md \
  part0-foundations-of-systems/0d3c-gpudirect-rdma-gpu-nic-topology-and-diagnostics.md \
  part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md
git commit -m "Repair Part 0 Markdown source links"
```

Expected: commit includes only the listed Part 0 Markdown files.

## Task 3: Repair Stale Part 4 Internal Paths And Add Freshness Labels

**Files:**

- Modify: `part4-data-and-storage/12a-model-registry.md`
- Modify: `part4-data-and-storage/12d-supply-chain-and-signing.md`
- Modify: `part4-data-and-storage/13c-vector-db-selection-and-operations.md`

- [ ] **Step 1: Locate stale internal paths and version-sensitive claims**

Run:

```bash
rg -n "part6-inference-and-serving|16a-vllm-multi-lora|16a-vllm-inference|16b-sglang|HuggingFace Hub 从 2024|MLflow|SageMaker|Vertex AI|Kubeflow Model Registry|SLSA L[0-4]|cosign|Trivy|Kyverno|OPA Gatekeeper|Rekor" part4-data-and-storage/12a-model-registry.md part4-data-and-storage/12d-supply-chain-and-signing.md part4-data-and-storage/13c-vector-db-selection-and-operations.md
```

Expected: output includes stale references from `part4-data-and-storage/12a-model-registry.md` to the legacy target string `../part6-inference-and-serving/16a-vllm-multi-lora.md`, stale serving references in `part4-data-and-storage/13c-vector-db-selection-and-operations.md`, plus version-sensitive ecosystem/tool claims.

- [ ] **Step 2: Fix stale vLLM chapter links in `part4-data-and-storage/12a-model-registry.md`**

Replace all references to:

```markdown
../part6-inference-and-serving/16a-vllm-multi-lora.md
```

with:

```markdown
../part5-serving-infra/16a-vllm-internals.md
```

When the surrounding text says "第 16a 章", keep that wording. When it says "Multi-LoRA 专章", rewrite it to "第 16a 章的 Multi-LoRA serving 章节" so the link target matches the current chapter.

Acceptance signal: `rg -n "part6-inference-and-serving|16a-vllm-multi-lora" part4-data-and-storage/12a-model-registry.md` has no output.

- [ ] **Step 3: Fix stale serving chapter links in `part4-data-and-storage/13c-vector-db-selection-and-operations.md`**

Replace stale serving links in the chapter header or related references:

```text
../part5-serving-infra/16a-vllm-inference.md -> ../part5-serving-infra/16a-vllm-internals.md
../part5-serving-infra/16b-sglang.md -> ../part5-serving-infra/16b-sglang-internals.md
```

Acceptance signal: `rg -n "16a-vllm-inference|16b-sglang\\.md" part4-data-and-storage/13c-vector-db-selection-and-operations.md` has no output, and both new target files exist.

- [ ] **Step 4: Add source-date labels to registry comparison**

In `part4-data-and-storage/12a-model-registry.md`, add a short note immediately before the "主流 Registry 对比" table:

```markdown
> **版本口径（2026-05）**：下表是工程选型口径，不是长期有效的产品排名。托管服务 API、LoRA 支持、企业审计、on-prem 能力和价格会变化；落地前需要按当前版本重新核对官方文档，并把核对日期写入 `BenchmarkProtocol` 或发布决策记录。
```

Acceptance signal: the registry comparison table has an explicit source-date and freshness caveat.

- [ ] **Step 5: Add source-date labels to supply-chain tooling claims**

In `part4-data-and-storage/12d-supply-chain-and-signing.md`, add a short note before the tooling comparison or SLSA section:

```markdown
> **版本口径（2026-05）**：Sigstore/cosign、Rekor、SLSA、GitHub attestation、Kyverno、OPA Gatekeeper、Trivy 和 HuggingFace Hub 的能力会随版本变化。本文示例用于说明工程机制，生产落地前必须记录工具版本、策略版本、验证命令和失败处理策略。
```

Acceptance signal: version-sensitive supply-chain claims are labeled as mechanism examples with a freshness boundary.

- [ ] **Step 6: Verify Task 3**

Run:

```bash
rg -n "part6-inference-and-serving|16a-vllm-multi-lora|16a-vllm-inference|16b-sglang\\.md" part4-data-and-storage/12a-model-registry.md part4-data-and-storage/12d-supply-chain-and-signing.md part4-data-and-storage/13c-vector-db-selection-and-operations.md
rg -n "版本口径（2026-05）" part4-data-and-storage/12a-model-registry.md part4-data-and-storage/12d-supply-chain-and-signing.md
test -f part5-serving-infra/16a-vllm-internals.md
test -f part5-serving-infra/16b-sglang-internals.md
git diff --check -- part4-data-and-storage/12a-model-registry.md part4-data-and-storage/12d-supply-chain-and-signing.md part4-data-and-storage/13c-vector-db-selection-and-operations.md
```

Expected: first command has no output; second command shows one freshness label in each of `12a` and `12d`; both `test -f` commands exit 0; final command exits 0.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add part4-data-and-storage/12a-model-registry.md part4-data-and-storage/12d-supply-chain-and-signing.md part4-data-and-storage/13c-vector-db-selection-and-operations.md
git commit -m "Add freshness labels to registry and supply chain chapters"
```

Expected: commit includes only the three listed Part 4 files.

## Task 4: Correct And Label Checkpoint Sizing Assumptions

**Files:**

- Modify: `part4-data-and-storage/12b-checkpoint-engineering.md`

- [ ] **Step 1: Locate all 175B checkpoint sizing statements**

Run:

```bash
rg -n "1\\.05 TB|175B|Adam 状态|Adam 一阶|Adam 二阶|optimizer state|FP32 Adam|sha256|per rank|checkpoint 大小|S3 归档大小" part4-data-and-storage/12b-checkpoint-engineering.md
```

Expected: output includes all sections that use the 175B / 1.05 TB example.

- [ ] **Step 2: Add an explicit assumption block before the first 1.05 TB calculation**

Insert this block before the paragraph that starts with `GPU idle 时间 = 序列化时间 + I/O 时间。175B 模型`:

```markdown
> **数值口径（示例，不是通用常数）**：本节 175B 示例按 BF16 参数 2 bytes/param、Adam 一阶矩 FP32 4 bytes/param、Adam 二阶矩 FP32 4 bytes/param 估算，因此 full training checkpoint 的主要张量约为 `175B × (2 + 4 + 4) bytes ≈ 1.75 TB`。如果实现不保存 FP32 master weights，则没有额外 master copy；如果保存 FP32 master weights，还需再加 `175B × 4 bytes ≈ 700 GB`。下文所有时间估算都必须标注使用哪个状态集合。
```

Then update the immediate calculation from:

```markdown
175B 模型 BF16 参数 ~350 GB，Adam 状态 ~700 GB，共 ~1.05 TB（FP32 optimizer state）
```

to:

```markdown
175B 模型 BF16 参数约 350 GB，Adam 一阶矩约 700 GB，Adam 二阶矩约 700 GB，full training checkpoint 主体约 1.75 TB（不含 FP32 master weights、RNG、scheduler、sampler 和 manifest 元数据）
```

Acceptance signal: the first sizing example no longer states `1.05 TB` for BF16 parameters plus both Adam moments.

- [ ] **Step 3: Update dependent per-rank and retention calculations**

Replace the dependent values consistently:

```text
1.05 TB / 512 ≈ 2.1 GB -> 1.75 TB / 512 ≈ 3.4 GB
11 × 1.05 TB ≈ 11.55 TB -> 11 × 1.75 TB ≈ 19.25 TB
Full checkpoint size ~1.05 TB -> ~1.75 TB without FP32 master weights, ~2.45 TB with FP32 master weights
S3 upload 1.05 TB at 50 GB/s -> 1.75 TB at 50 GB/s ≈ 35 s theoretical
S3 upload 1.05 TB at 200 GB/s -> 1.75 TB at 200 GB/s ≈ 8.75 s theoretical
Adam 状态 | 175B × 4 bytes = 700 GB（FP32 一阶+二阶矩） -> Adam 一阶矩 | 175B × 4 bytes = 700 GB and Adam 二阶矩 | 175B × 4 bytes = 700 GB
总 checkpoint 大小 | ~1.05 TB -> 总 checkpoint 大小 | ~1.75 TB（不含 FP32 master weights）或 ~2.45 TB（含 FP32 master weights）
```

For the `Integrity check（sha256）` row, replace `1.05 TB per rank` with:

```markdown
3.4 GB/rank（512 rank 均匀分片）
```

Acceptance signal: every remaining `1.05 TB` either disappears or is explicitly described as an obsolete comparison in a note. Prefer no remaining `1.05 TB`.

- [ ] **Step 4: Update exercises that depend on 1.05 TB**

In the 12b exercises, update the retention question from:

```markdown
每次 checkpoint 1.05 TB，总 Lustre 容量 50 TB
```

to:

```markdown
每次 full training checkpoint 1.75 TB（不含 FP32 master weights），总 Lustre 容量 50 TB
```

Acceptance signal: exercise assumptions match the chapter calculation.

- [ ] **Step 5: Verify Task 4**

Run:

```bash
rg -n "1\\.05 TB|Adam 状态 ~700 GB|FP32 Adam ~1\\.05 TB|1\\.05TB|Adam 状态 \\| 175B × 4 bytes = 700 GB|总 checkpoint 大小 \\| ~1\\.05 TB" part4-data-and-storage/12b-checkpoint-engineering.md
rg -n "1\\.75 TB|2\\.45 TB|数值口径（示例，不是通用常数）|3\\.4 GB/rank|19\\.25 TB|Adam 一阶矩|Adam 二阶矩" part4-data-and-storage/12b-checkpoint-engineering.md
git diff --check -- part4-data-and-storage/12b-checkpoint-engineering.md
```

Expected: first command has no output; second command shows updated sizing labels and derived values; third command exits 0.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add part4-data-and-storage/12b-checkpoint-engineering.md
git commit -m "Correct checkpoint sizing assumptions"
```

Expected: commit includes only `part4-data-and-storage/12b-checkpoint-engineering.md`.

## Task 5: Correct And Label Vector Index Capacity Assumptions

**Files:**

- Modify: `part4-data-and-storage/13b-vector-index-algorithms.md`
- Modify: `part4-data-and-storage/13c-vector-db-selection-and-operations.md`

- [ ] **Step 1: Locate HNSW and vector-memory examples**

Run:

```bash
rg -n "~38 GB|38\\.4|~35 GB|HNSW.*内存|向量原始内存|HNSW 额外内存|2 TB|921 GB|100M|1536|768" part4-data-and-storage/13b-vector-index-algorithms.md part4-data-and-storage/13c-vector-db-selection-and-operations.md
```

Expected: output includes the 100M 768d worked example in 13b and the 100M 1536d capacity example in 13c.

- [ ] **Step 2: Fix the 13b 100M 768d HNSW memory example**

In `part4-data-and-storage/13b-vector-index-algorithms.md`, add this note under "场景设定":

```markdown
> **数值口径（2026-05 示例）**：本例区分 raw vector storage、graph adjacency、metadata 和压缩码。100M × 768d × float32 的原始向量约 307 GB，因此如果 HNSW 在内存中保留 float32 原始向量，单机 256 GB RAM 不可行。只有在向量被量化、内存映射、分片，或仅把压缩码/图邻接放入内存时，内存数字才可能降到几十 GB。
```

Then replace the HNSW row in the comparison table:

```markdown
| **索引内存** | ~38 GB（向量 + 图） | ~3.6 GB（PQ 压缩） | ~3.2 GB（PQ 压缩） |
```

with:

```markdown
| **索引内存** | > 307 GB 原始向量 + 图邻接；单机 256 GB 不可行，除非量化/分片/内存映射 | ~3.6 GB PQ code + centroid/metadata，不含原始向量副本 | ~3.2 GB PQ code/导航结构 + SSD 原始向量 |
```

Then replace the HNSW decision paragraph so it does not recommend uncompressed single-node HNSW on 256 GB RAM. Use this wording:

```markdown
本场景若必须在单台 256 GB RAM 机器上运行，不应选择保留 float32 原始向量的 HNSW，因为 raw vector storage 已经超过内存。推荐 DiskANN 或 IVFPQ；如果业务强依赖 HNSW 的召回和低延迟，需要先做向量量化、分片到多节点，或把原始向量放到 mmap/SSD 并用压缩表示参与检索。
```

Acceptance signal: 13b no longer claims 100M 768d float32 HNSW uses only ~38 GB while also including raw vectors.

- [ ] **Step 3: Fix the earlier 13b index comparison table**

Find the earlier comparison row that currently says HNSW memory is `~35 GB（原始向量 + 图）`. Replace that HNSW cell with:

```markdown
> 307 GB 原始向量 + 图邻接；需分片、量化或 mmap
```

If the table has a column header that implies all values are in-memory index size, add a short note immediately before the table:

```markdown
> **口径提醒**：下表的 HNSW 内存必须区分 raw vector storage 和 graph adjacency。对 100M × 768d float32，raw vectors alone 约 307 GB，因此不能写成几十 GB 的"原始向量 + 图"。
```

Acceptance signal: `rg -n "~35 GB（原始向量 \\+ 图）" part4-data-and-storage/13b-vector-index-algorithms.md` has no output.

- [ ] **Step 4: Update the 13b tuning log**

In the HNSW tuning YAML, replace:

```yaml
memory_gb: 38.4
```

with:

```yaml
memory_model:
  raw_vectors_gb: 307
  graph_and_metadata_gb: "depends on M, level distribution, id width, implementation"
  compression: "required for single-node 256GB RAM"
```

Acceptance signal: tuning log records assumptions instead of one unexplained memory number.

- [ ] **Step 5: Tighten the 13c capacity formula**

In `part4-data-and-storage/13c-vector-db-selection-and-operations.md`, replace the simplified HNSW formula:

```markdown
HNSW 额外内存 ≈ 向量原始内存 × 1.4-2.0 (取决于 M 参数)
```

with:

```markdown
HNSW 内存 = 原始向量存储 + 图邻接 + level metadata + payload/filter index + allocator overhead
  - 原始向量是否在内存中保留，取决于实现、量化和 mmap 策略
  - 图邻接粗估可按 `N × M × id_width × layer_factor` 起步，再用真实构建结果校准
  - 不要把"图额外内存比例"和"总内存比例"混用
```

Then update the 100M 1536d example so it labels the HNSW number as a conservative full-in-memory estimate:

```markdown
向量原始内存 = 1536 × 4 × 100,000,000 = 614 GB
HNSW 图邻接和元数据：需按实现实测；若按 full-in-memory 粗估，总内存通常会超过 1 TB
总内存需求 = raw vectors + graph/metadata + payload/filter index + safety margin

→ 单机 128GB/256GB 内存不适合承载这个口径的 HNSW；应选择分片、量化、DiskANN、IVFPQ，或把原始向量 mmap 到 SSD 并重新压测召回和 P99。
```

Acceptance signal: 13c separates raw vector memory from HNSW graph overhead and removes the misleading `614 × 1.5 = 921 GB` as if it were a universal formula.

- [ ] **Step 6: Fix the 13c capacity flowchart assumption**

In the capacity planning flowchart, replace any HNSW branch text like:

```markdown
原始内存 × 1.5-2.0
```

with:

```markdown
raw vectors + graph/metadata
按实现实测
```

Acceptance signal: `rg -n "原始内存 × 1\\.5-2\\.0|1\\.5-2\\.0" part4-data-and-storage/13c-vector-db-selection-and-operations.md` does not show HNSW capacity formula text.

- [ ] **Step 7: Verify Task 5**

Run:

```bash
rg -n "~38 GB（向量 \\+ 图）|~35 GB（原始向量 \\+ 图）|memory_gb: 38\\.4|HNSW 额外内存 ≈ 向量原始内存 × 1\\.4-2\\.0|614 × 1\\.5 = 921 GB|原始内存 × 1\\.5-2\\.0" part4-data-and-storage/13b-vector-index-algorithms.md part4-data-and-storage/13c-vector-db-selection-and-operations.md
rg -n "数值口径（2026-05 示例）|raw_vectors_gb: 307|不要把\"图额外内存比例\"和\"总内存比例\"混用|full-in-memory" part4-data-and-storage/13b-vector-index-algorithms.md part4-data-and-storage/13c-vector-db-selection-and-operations.md
git diff --check -- part4-data-and-storage/13b-vector-index-algorithms.md part4-data-and-storage/13c-vector-db-selection-and-operations.md
```

Expected: first command has no output; second command shows the new labels and formula warnings; third command exits 0.

- [ ] **Step 8: Commit Task 5**

Run:

```bash
git add part4-data-and-storage/13b-vector-index-algorithms.md part4-data-and-storage/13c-vector-db-selection-and-operations.md
git commit -m "Correct vector index capacity assumptions"
```

Expected: commit includes only the two vector chapters.

## Task 6: Final Wave 1 Verification

**Files:**

- Read: all modified files from Tasks 1-5

- [ ] **Step 1: Run final verification in one shell block**

Run:

```bash
WAVE1_FILES=(
  README.md
  00-preface.md
  part0-foundations-of-systems/0c-filesystems-and-storage-internals.md
  part0-foundations-of-systems/0c1-vfs-inode-dentry-and-block-layer.md
  part0-foundations-of-systems/0c2-local-filesystems-ext4-xfs-zfs.md
  part0-foundations-of-systems/0c3-storage-semantics-fsync-direct-io-and-checkpoints.md
  part0-foundations-of-systems/0c4-object-storage-parallel-filesystems-and-dataset-io.md
  part0-foundations-of-systems/0d-network-stack-fundamentals.md
  part0-foundations-of-systems/0d1-linux-network-stack-tcp-and-mtu.md
  part0-foundations-of-systems/0d2-nic-offload-queues-and-service-network-io.md
  part0-foundations-of-systems/0d3-rdma-roce-infiniband-and-gpudirect.md
  part0-foundations-of-systems/0d3a-rdma-verbs-memory-registration-and-queues.md
  part0-foundations-of-systems/0d3b-roce-infiniband-lossless-fabric-and-congestion.md
  part0-foundations-of-systems/0d3c-gpudirect-rdma-gpu-nic-topology-and-diagnostics.md
  part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md
  part4-data-and-storage/12a-model-registry.md
  part4-data-and-storage/12b-checkpoint-engineering.md
  part4-data-and-storage/12d-supply-chain-and-signing.md
  part4-data-and-storage/13b-vector-index-algorithms.md
  part4-data-and-storage/13c-vector-db-selection-and-operations.md
  appendix/glossary.md
  appendix/checklists.md
)

rg -n "\]\([^)]*\.html[)#]?" "${WAVE1_FILES[@]}" | rg -v "https?://"
rg -n "part6-inference-and-serving|16a-vllm-multi-lora|16a-vllm-inference|16b-sglang\\.md" "${WAVE1_FILES[@]}"
for term in EvidenceBundle CapacityLedger ReleaseUnit StateManifest RestoreLevel CacheKeyContract TenantBudget BenchmarkProtocol; do
  rg -q "$term" appendix/glossary.md || echo "missing $term"
done
rg -n "1\\.05 TB|Adam 状态 ~700 GB|Adam 状态 \\| 175B × 4 bytes = 700 GB|总 checkpoint 大小 \\| ~1\\.05 TB|~38 GB（向量 \\+ 图）|~35 GB（原始向量 \\+ 图）|memory_gb: 38\\.4|614 × 1\\.5 = 921 GB|HNSW 额外内存 ≈ 向量原始内存 × 1\\.4-2\\.0|原始内存 × 1\\.5-2\\.0" part4-data-and-storage/12b-checkpoint-engineering.md part4-data-and-storage/13b-vector-index-algorithms.md part4-data-and-storage/13c-vector-db-selection-and-operations.md
rg -n "T[O]DO|T[B]D|F[I]XME|待[定]|待[补]|后续[补]|这里不[展]开|PLACE[H]OLDER|\\.\\.\\." "${WAVE1_FILES[@]}"
git diff --check -- "${WAVE1_FILES[@]}"
git diff --stat HEAD -- "${WAVE1_FILES[@]}"
```

Expected:

- Generated internal HTML link command produces no output after filtering external HTTP links.
- Stale serving path command produces no output.
- Contract loop prints no `missing` lines.
- High-risk numeric scan produces no output.
- Placeholder scan produces no unresolved placeholder output. If existing tutorial prose intentionally uses an ellipsis-like punctuation in Chinese text, confirm it is not a placeholder before proceeding.
- `git diff --check` exits 0.
- `git diff --stat` shows only exact Wave 1 files.

- [ ] **Step 2: Review staged and unstaged changes**

Run:

```bash
WAVE1_FILES=(
  README.md
  00-preface.md
  part0-foundations-of-systems/0c-filesystems-and-storage-internals.md
  part0-foundations-of-systems/0c1-vfs-inode-dentry-and-block-layer.md
  part0-foundations-of-systems/0c2-local-filesystems-ext4-xfs-zfs.md
  part0-foundations-of-systems/0c3-storage-semantics-fsync-direct-io-and-checkpoints.md
  part0-foundations-of-systems/0c4-object-storage-parallel-filesystems-and-dataset-io.md
  part0-foundations-of-systems/0d-network-stack-fundamentals.md
  part0-foundations-of-systems/0d1-linux-network-stack-tcp-and-mtu.md
  part0-foundations-of-systems/0d2-nic-offload-queues-and-service-network-io.md
  part0-foundations-of-systems/0d3-rdma-roce-infiniband-and-gpudirect.md
  part0-foundations-of-systems/0d3a-rdma-verbs-memory-registration-and-queues.md
  part0-foundations-of-systems/0d3b-roce-infiniband-lossless-fabric-and-congestion.md
  part0-foundations-of-systems/0d3c-gpudirect-rdma-gpu-nic-topology-and-diagnostics.md
  part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md
  part4-data-and-storage/12a-model-registry.md
  part4-data-and-storage/12b-checkpoint-engineering.md
  part4-data-and-storage/12d-supply-chain-and-signing.md
  part4-data-and-storage/13b-vector-index-algorithms.md
  part4-data-and-storage/13c-vector-db-selection-and-operations.md
  appendix/glossary.md
  appendix/checklists.md
)

git status --short
git diff --stat HEAD -- "${WAVE1_FILES[@]}"
```

Expected: modified files are limited to Wave 1 Markdown scope plus any pre-existing unrelated dirty files. Do not stage `html/` files in this wave.

- [ ] **Step 3: Commit final verification note if needed**

If Tasks 1-5 each committed cleanly and Task 6 made no file changes, do not create an empty commit.

If Task 6 required small fixes, commit only those fixes:

```bash
WAVE1_FILES=(
  README.md
  00-preface.md
  part0-foundations-of-systems/0c-filesystems-and-storage-internals.md
  part0-foundations-of-systems/0c1-vfs-inode-dentry-and-block-layer.md
  part0-foundations-of-systems/0c2-local-filesystems-ext4-xfs-zfs.md
  part0-foundations-of-systems/0c3-storage-semantics-fsync-direct-io-and-checkpoints.md
  part0-foundations-of-systems/0c4-object-storage-parallel-filesystems-and-dataset-io.md
  part0-foundations-of-systems/0d-network-stack-fundamentals.md
  part0-foundations-of-systems/0d1-linux-network-stack-tcp-and-mtu.md
  part0-foundations-of-systems/0d2-nic-offload-queues-and-service-network-io.md
  part0-foundations-of-systems/0d3-rdma-roce-infiniband-and-gpudirect.md
  part0-foundations-of-systems/0d3a-rdma-verbs-memory-registration-and-queues.md
  part0-foundations-of-systems/0d3b-roce-infiniband-lossless-fabric-and-congestion.md
  part0-foundations-of-systems/0d3c-gpudirect-rdma-gpu-nic-topology-and-diagnostics.md
  part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md
  part4-data-and-storage/12a-model-registry.md
  part4-data-and-storage/12b-checkpoint-engineering.md
  part4-data-and-storage/12d-supply-chain-and-signing.md
  part4-data-and-storage/13b-vector-index-algorithms.md
  part4-data-and-storage/13c-vector-db-selection-and-operations.md
  appendix/glossary.md
  appendix/checklists.md
)

git add "${WAVE1_FILES[@]}"
git commit -m "Verify Wave 1 AI infra tutorial cleanup"
```

Expected: final commit includes only Wave 1 Markdown files.

## Execution Notes

- The worktree already contains unrelated uncommitted Markdown and HTML changes. Do not revert or stage unrelated changes.
- Prefer one commit per task so review can isolate source-policy, link repair, freshness labels, checkpoint sizing, and vector sizing.
- If a file has pre-existing user edits outside the planned section, work with them and do not rewrite unrelated content.
- If generated HTML needs rebuilding after the Markdown changes, make that a separate follow-up plan or task outside Wave 1.

## Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-04-ai-infra-tutorial-wave-1-source-integrity-and-contracts.md`. Execution options:

1. **Subagent-Driven** - dispatch fresh `gpt-5.5 xhigh` subagents per task, with at most 5 concurrent agents, and review between tasks.
2. **Inline Execution** - execute tasks in this session with checkpoint reviews after each task.
