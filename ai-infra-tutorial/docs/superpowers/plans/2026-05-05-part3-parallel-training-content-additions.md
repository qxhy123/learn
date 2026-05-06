# Part 3 并行训练策略内容补充实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. User constraint: at most 3 subagents running concurrently; use claude-sonnet-4-6 or better for all subagents.

**Goal:** 按 `docs/superpowers/specs/2026-05-05-part3-parallel-training-gap-spec.md` 中的 10 个 Gap，向 Chapter 8 和 Chapter 9 插入缺失的工程公式、现代架构约束、决策推导和操作指导。

**Architecture:** 三批次串行执行——P0（核心公式，Gap-1/2/3）→ P1（现代架构+决策支撑，Gap-4/5/6/7）→ P2（前沿与转换路径，Gap-8/9/10）。每批次完成后独立 commit，主 agent 做交叉数字核验后才放行下一批次。P0 批次内 Gap-2 先于 Gap-1/3 完成（Gap-1 的 PP 段落引用 activation 维度，Gap-3 的组合模型引用 Gap-1 公式）。

**Tech Stack:** Markdown 源文件（`part3-training-infra/*.md`），`rg`（ripgrep）做内容核验，`wc -l` 做行数 sanity check，`git diff --check` 做空白和冲突检查。不涉及 HTML 重新生成（由后续独立 task 处理）。

---

## Source References

- Spec: `docs/superpowers/specs/2026-05-05-part3-parallel-training-gap-spec.md`
- Chapter 8 source: `part3-training-infra/08-data-parallel.md`（基准 1502 行）
- Chapter 9 source: `part3-training-infra/09-model-pipeline-parallel.md`（基准 1239 行）
- Do NOT modify: `html/` 目录、其他 Part 的源文件

## File Structure

### Modify

- `part3-training-infra/08-data-parallel.md`
  - Gap-2 插入：§5.3 WARNING 脚注后增加交叉引用（≤10 行）
  
- `part3-training-infra/09-model-pipeline-parallel.md`
  - Gap-2 插入：§4.1 之后新增 §4.1a（activation 内存估算）
  - Gap-1 插入：§4.2 之后新增 §4.2a（TP 通信量），§4.3 之后新增 §4.3a（PP 通信量），§4.6 之后新增 §4.6a（CP 通信量）
  - Gap-3 插入：§7.2 之后新增 §7.3（3D parallel step time 组合模型）
  - Gap-4 插入：§5.1 配置审查要点之后新增 GQA/MQA 约束小节；§8.3 框架支持表增加 GQA 行
  - Gap-6 插入：§4.2a 之后新增 §4.2b（通信重叠架构）
  - Gap-5 插入：§8.1 决策树之前新增 §8.0（rank mesh 推导流程）
  - Gap-7 插入：§4.3.3 之后新增 §4.3.4（stage 负载均衡）；§11 故障排除表新增"stage 时间不均"行
  - Gap-8 插入：§5.1 新增 FP8 参数说明；§8.3 新增 FP8 行；§13.3 新增 FP8 检查项
  - Gap-9 插入：§4.4 扩写（zero bubble 工程细节）
  - Gap-10 插入：§8.4 之后新增 §8.5（推理 checkpoint 转换）

### Do NOT modify

- `html/`（任何 .html 文件）
- `part3-training-infra/07-single-node-training.md`
- `part3-training-infra/09e-moe-training-infrastructure.md`
- `part3-training-infra/10-memory-checkpointing-and-recovery.md`
- `part3-training-infra/10b-alignment-and-post-training.md`
- `part3-training-infra/10c-finetuning-and-multi-adapter.md`

---

## Batch 1: P0 核心公式（Gap-2 → Gap-1 → Gap-3）

---

### Task 1: Gap-2 — Activation 内存定量模型

**Files:**
- Modify: `part3-training-infra/09-model-pipeline-parallel.md`（§4.1 最小容量账本 之后）
- Modify: `part3-training-infra/08-data-parallel.md`（§5.3 ZeRO WARNING 脚注处）

- [ ] **Step 1: 定位 Ch9 §4.1 的插入点**

运行：

```bash
grep -n "activation_resident\|attention_workspace\|fragmentation_margin\|communication_buffers" \
  part3-training-infra/09-model-pipeline-parallel.md | head -20
```

预期：找到 §4.1 容量账本公式块所在行号（约 285-310 行范围）。记录最后一行行号，新增内容插入其后。

- [ ] **Step 2: 在 Ch9 §4.1 之后插入 §4.1a（Activation 内存估算）**

在 `part3-training-infra/09-model-pipeline-parallel.md` 中，`容量判断必须用真实训练形态` 段落之后插入以下内容（保持 ### 层级与周围章节一致）：

```markdown
### 4.1a Activation 内存估算

activation 是 HBM 账本中变化最大、最容易被低估的项。它不随并行维度切分而自动缩小——TP 切 hidden 后每 rank 的 activation 维度减半，PP 切层数后每 rank 的层数减少，但 ZeRO/FSDP 不切 activation。

**Per Transformer block activation（BF16，标准 MHA，无 AC）**

```text
以 BF16 训练为基准，一个 Transformer block 的 activation 近似占用：

  attn_input       = batch × seq × hidden × 2
  attn_scores      = batch × num_heads × seq × seq × 2   （quadratic！标准 MHA）
  attn_output      = batch × seq × hidden × 2
  mlp_activation   = batch × seq × intermediate × 2       （通常 intermediate = 4 × hidden）
  residual + norms = batch × seq × hidden × 2 × 3

  per_block_bytes ≈ batch × seq × hidden × (1+1+1+4+3) × 2
                  = batch × seq × hidden × 20 bytes        （BF16，无 AC）
```

**FlashAttention 对 activation 的影响**

FlashAttention 不存储 O(seq²) 的 attention score，只存 softmax 分母（log-sum-exp）：

```text
lse_bytes = batch × num_heads × seq × 4 bytes（FP32 LSE）

节省量（vs 标准 MHA）：
  saved = batch × num_heads × seq × seq × 2
```

对 seq=8192、64 heads、batch=1：`saved = 1 × 64 × 8192 × 8192 × 2 ≈ 8.6 GB / layer`。这是 FlashAttention 在长序列下成为必选项的根本原因。

**不同 AC 策略的 per-block activation 对比（BF16）**

| AC 策略 | Per-block activation | 额外 FLOPs | 备注 |
|---|---|---|---|
| 无 AC（全量存储） | `batch × seq × hidden × 20 bytes` | 0 | 短序列、HBM 充足 |
| Full recompute（每层重算） | `batch × seq × hidden × 2 bytes`（只存 block 输入） | +33%（多一次 forward） | HBM 极度紧张 |
| Selective recompute（重算 attention） | `batch × seq × hidden × 12 bytes` | +~15% | 平衡选项 |
| FlashAttention（自动节省 attention score） | `batch × seq × hidden × 12 bytes` | 0（kernel fusion） | 生产首选 |
| Selective + FlashAttention | `batch × seq × hidden × 8 bytes` | 接近 0 | 长序列生产标准 |

**数字示例（70B，hidden=8192，seq=8192，micro_batch=1，BF16，80 层）**

```text
无 AC：                     80 × 20 × 8192 × 8192 × 2 bytes ≈ 214 GB
Full recompute：            80 × 2  × 8192 × 8192 × 2 bytes ≈  21 GB
Selective + FlashAttention：80 × 8  × 8192 × 8192 × 2 bytes ≈  86 GB
```

> [!DANGER]
> **214 GB 超过整个 8×80GB 节点总 HBM。** 生产 70B 训练必须开 FlashAttention 或 Selective AC，无论是 DDP、FSDP 还是 TP/PP 配置。ZeRO 和 FSDP 不切 activation；降低 activation 的唯一系统手段是 AC（含 FlashAttention 隐式 AC）、降 batch/seq、或 SP/CP 切 sequence 维度。

**TP 和 PP 对 activation 的影响**

- TP=8：每 rank hidden/8，`attn_input/output` 等正比降低；但 `attn_scores`（quadratic）只降 `num_heads/TP`，不降 seq²。
- PP=4（70B 80 层）：每 rank 只有 20 层，activation 降到 1/4；但 stage 边界 send/recv buffer 增加（见 §4.1）。
- CP=4：attention 的 KV/scores 从 full seq² 降到 `(seq/CP)²`，对 attention workspace 有二次方级别改善（见 §4.6a）。
```

- [ ] **Step 3: 在 Ch8 §5.3 ZeRO WARNING 后增加交叉引用**

运行：

```bash
grep -n "ZeRO 不切 activation\|不含 activation\|激活随 batch" \
  part3-training-infra/08-data-parallel.md | head -5
```

预期：找到 §5.3 ZeRO WARNING 块中关于 activation 的说明行号。在该段落最后一行后追加：

```markdown
> **Activation 内存的定量估算和不同 AC 策略对比，参见 [第9章 §4.1a](./09-model-pipeline-parallel.md#41a-activation-内存估算)。**
```

- [ ] **Step 4: 验证插入结果**

运行：

```bash
grep -n "Per Transformer block\|FlashAttention 对 activation\|AC 策略\|Selective + FlashAttention\|214 GB" \
  part3-training-infra/09-model-pipeline-parallel.md
grep -n "第9章.*4.1a\|Activation 内存的定量" \
  part3-training-infra/08-data-parallel.md
wc -l part3-training-infra/09-model-pipeline-parallel.md part3-training-infra/08-data-parallel.md
git diff --check -- part3-training-infra/09-model-pipeline-parallel.md part3-training-infra/08-data-parallel.md
```

预期：
- Ch9 包含所有 5 个关键词；行数比基准（1239）增加 100-150 行
- Ch8 包含交叉引用文字
- `git diff --check` 无输出（无空白错误）

- [ ] **Step 5: Commit Task 1**

```bash
git add part3-training-infra/09-model-pipeline-parallel.md part3-training-infra/08-data-parallel.md
git commit -m "Part3 Gap-2: add activation memory model and AC strategy comparison"
```

---

### Task 2: Gap-1 — TP / PP / CP 通信量公式

**Files:**
- Modify: `part3-training-infra/09-model-pipeline-parallel.md`（§4.2 后、§4.3 后、§4.6 后各插一节）

- [ ] **Step 1: 定位 §4.2、§4.3、§4.6 的末行**

运行：

```bash
grep -n "^### 4\." part3-training-infra/09-model-pipeline-parallel.md
```

预期：打印所有 §4.x 小节标题和行号，从中找到 §4.2、§4.3（PP 调度节）、§4.6（SP 和 CP）的末行（即下一个 `###` 之前的最后一行）。

- [ ] **Step 2: 在 §4.2（TP 通信边界）之后插入 §4.2a（TP 通信量估算）**

插入位置：`### 4.3 PP、microbatch 和 pipeline bubble` 之前。内容：

```markdown
### 4.2a TP 通信量估算

第 8 章给出了 DP AllReduce 的心算公式。TP 需要等价的估算基础。

**Megatron Column-then-Row TP 的 per-layer 通信**

以标准 decoder-only Transformer 的 Megatron TP 实现为基准（Column parallel + Row parallel，每路各触发一次 AllReduce）：

```text
每个 Transformer block 的 AllReduce 次数：
  Attention (Q,K,V projection → Column) + (output projection → Row)：forward 1次，backward 1次
  MLP (FC1 → Column) + (FC2 → Row)：forward 1次，backward 1次
  合计：每 block forward 2次，backward 2次 AllReduce

每次 AllReduce 的数据量（Ring AllReduce，含 ReduceScatter + AllGather 两趟）：
  message_size = micro_batch × seq_len × hidden_size × dtype_bytes
  bytes_per_rank = 2 × (TP-1)/TP × message_size

每 block forward TP 通信量（per rank）：
  tp_fwd_bytes = 2 × 2 × (TP-1)/TP × micro_batch × seq × hidden × dtype_bytes
```

**数字示例（70B，BF16，seq=8192，micro_batch=1，TP=8，节点内 NVSwitch）**

```text
message_size = 1 × 8192 × 8192 × 2 = 134 MB
bytes_per_rank per call = 2 × (7/8) × 134 MB ≈ 235 MB
每 block forward 2次 AllReduce → per block ≈ 470 MB

70B 有 80 层：total forward TP traffic per rank ≈ 80 × 470 MB ≈ 37 GB

NVSwitch 节点内 AllReduce 有效 bus bandwidth ≈ 600 GB/s（8×H100 实测）：
  single call time ≈ 235 MB / 600 GB/s ≈ 0.4 ms
  80层 × 2次 = 160次 AllReduce，理论总时间约 64 ms（完全串行上界）
  实际：大部分 AllReduce 被下层 GEMM 掩盖，exposed tail 通常 5-15 ms（见 §4.2b）
```

**SP（Sequence Parallel）模式的差异**

SP 把 TP AllReduce 改写为 ReduceScatter（scatter 到各 rank 保留 sequence 分片）+ AllGather（在需要全量时聚合），总通信量近似相同，但 activation 常驻量降低：每 rank 保存 `seq/TP` 的 sequence 分片而不是完整 seq。

> [!NOTE]
> 如果 TP collective 在节点内 NVSwitch 完全被 GEMM 掩盖（见 §4.2b），TP 通信量对 step time 几乎无影响。跨节点 TP 时需用节点间带宽（如 400G IB ≈ 50 GB/s bus bw）重算：0.4 ms × (600/50) ≈ 4.8 ms per call，160 次串行约 768 ms——远超 GEMM 时间，成为严重瓶颈。这是"TP 优先节点内"的定量依据。
```

- [ ] **Step 3: 在 §4.3（PP 调度）之后插入 §4.3a（PP 通信量估算）**

插入位置：`### 4.4 virtual stage` 之前。内容：

```markdown
### 4.3a PP 通信量估算

PP activation send/recv 是跨节点通信，带宽上限由 IB/RoCE 决定。

**Per stage boundary，per microbatch**

```text
forward activation send/recv（每个 stage 边界，每个 microbatch）：
  pp_boundary_bytes = micro_batch × seq_len × hidden_size × dtype_bytes

backward gradient send/recv（同一边界）：
  pp_boundary_bwd_bytes = micro_batch × seq_len × hidden_size × dtype_bytes

单 microbatch 走完一次完整 pipeline（PP stage，PP-1 个中间边界）：
  pp_per_microbatch_total = 2 × (PP-1) × pp_boundary_bytes
```

**数字示例（70B，BF16，seq=8192，micro_batch=1，PP=4）**

```text
pp_boundary_bytes = 1 × 8192 × 8192 × 2 = 134 MB per boundary
PP=4 有 3 个中间边界
单 microbatch forward+backward PP 通信量 = 2 × 3 × 134 MB = 804 MB

400G IB（单向 50 GB/s）：
  单边界 forward send ≈ 134 MB / 50 GB/s ≈ 2.7 ms
  1F1B 稳态下 send/recv 与 compute 部分重叠（见 §4.2b）
  exposed PP 通常 1-5 ms per boundary（取决于 overlap 效果）
```

**PP 通信量对 TP 的敏感性**

PP 传输的是 full sequence activation（TP 切 hidden 后维度减半）：

```text
使用 TP=8（hidden 切 1/8）：
  pp_boundary_bytes = 1 × 8192 × (8192/8) × 2 = 16.8 MB per boundary

比 TP=1 降低 8×，大幅减少跨节点 PP 流量。这是 TP 与 PP 配合的隐性收益之一。
```
```

- [ ] **Step 4: 在 §4.6（CP 本质区别）之后插入 §4.6a（CP 通信量估算）**

插入位置：`### 4.7 EP 的位置` 之前。内容：

```markdown
### 4.6a CP 通信量估算

CP 通过 ring 传递 KV（Ring FlashAttention 实现）或 All-to-All（Ulysses 实现），通信量随 context 长度和 CP size 决定。

**Ring FlashAttention CP 模式（每个 attention layer，每轮 ring）**

```text
每个 rank 持有 seq/CP 个 token 的 Q/K/V 分片：
  kv_per_step = micro_batch × (seq/CP) × num_kv_heads × head_dim × 2（K+V）× dtype_bytes

一个 attention layer 完整 CP ring 通信量（CP-1 轮 ring，单向 send）：
  cp_total_per_layer = (CP-1) × kv_per_step
```

**数字示例（70B GQA：8 KV heads，head_dim=128，seq=65536，CP=4，BF16）**

```text
kv_per_step = 1 × (65536/4) × 8 × 128 × 2 × 2 = 536 MB
3 ring steps per layer → 每层 CP 通信量 ≈ 1.6 GB（forward；backward 相似量级）
70B 80 层：单 step CP 通信量 ≈ 80 × 1.6 GB × 2（fwd+bwd）≈ 256 GB
```

> [!DANGER]
> **256 GB 每 step，400G IB（50 GB/s）下纯通信时间 ≈ 5.1 s。** 65K context 的 CP 必须依赖 Ring FA 的 overlap（KV exchange 与本地 attention 同时进行，见 §4.2b）才能将 exposed communication 压缩到 1 s 以内，并且需要 800G 或更高带宽 IB 才能支撑大规模 CP。这是 CP 对网络最敏感的原因。

**Ulysses（All-to-All CP）模式的通信量差异**

Ulysses 把 Q/K/V 在 sequence 维度重排，通信方式是 All-to-All 而不是 ring send/recv：

```text
per All-to-All（每个 attention layer，forward）：
  ulysses_bytes = micro_batch × seq × num_heads × head_dim × dtype_bytes × 2（QKV）

All-to-All 每 rank 发送量 ≈ total / CP，接收量相同，通常效率高于 ring 但对拓扑敏感。
```
```

- [ ] **Step 5: 验证三节插入结果**

运行：

```bash
grep -n "4.2a\|4.3a\|4.6a\|TP 通信量估算\|PP 通信量估算\|CP 通信量估算\|bytes_per_rank per call\|pp_boundary_bytes\|kv_per_step" \
  part3-training-infra/09-model-pipeline-parallel.md
wc -l part3-training-infra/09-model-pipeline-parallel.md
git diff --check -- part3-training-infra/09-model-pipeline-parallel.md
```

预期：找到全部 9 个关键词；行数比 Task 1 后的基准再增加 130-180 行；diff check 无输出。

- [ ] **Step 6: Commit Task 2**

```bash
git add part3-training-infra/09-model-pipeline-parallel.md
git commit -m "Part3 Gap-1: add TP/PP/CP communication volume formulas"
```

---

### Task 3: Gap-3 — 3D Parallel Step Time 组合模型

**Files:**
- Modify: `part3-training-infra/09-model-pipeline-parallel.md`（§7.2 之后插入 §7.3）

- [ ] **Step 1: 定位 §7.2 末行**

运行：

```bash
grep -n "^### 7\." part3-training-infra/09-model-pipeline-parallel.md
```

预期：找到 §7.2（PP bubble 与有效吞吐）和 §7.3（70B 状态粗算）的行号。将新 §7.3 插入旧 §7.3（70B 状态粗算）之前，旧 §7.3/§7.4 编号顺延（更新为 §7.4/§7.5）。

- [ ] **Step 2: 插入 §7.3（3D parallel step time 组合模型）**

插入位置：旧 `### 7.3 70B 状态粗算` 之前。同步将旧 §7.3 和 §7.4 标题更新为 §7.4 和 §7.5。插入内容：

```markdown
### 7.3 3D Parallel Step Time 组合模型

第 8 章的 DP step time 模型（`step_time = compute + exposed_comm + optimizer`）在混合并行下需要扩展。3D parallel 的 step time 来源于四类通信的叠加，每类都有独立的 overlap 条件。

**组合模型**

```text
step_time_3d ≈ max_dp_replica(
    PP_warmup                                           # (PP-1) × avg_stage_time，无法掩盖
  + m × max_stage(
        max_rank_in_stage(
            microbatch_compute                          # CUDA kernel 时间，profiler 直接测量
          + max(tp_collective - tp_overlap, 0)          # TP exposed（节点内通常 5-15 ms）
          + max(pp_send_recv - pp_overlap, 0)           # PP exposed（跨节点通常 1-5 ms per boundary）
        )
    )
  + PP_drain                                            # (PP-1) × avg_stage_time，无法掩盖
  + max(dp_sync - dp_overlap, 0)                        # DP exposed（梯度同步，见第 8 章公式）
  + max(fsdp_allgather - fsdp_overlap, 0)              # FSDP exposed（如启用 hybrid sharding）
  + max(cp_exchange - cp_overlap, 0)                    # CP exposed（KV exchange，长上下文敏感）
  + optimizer
)

bubble_latency = PP_warmup + PP_drain ≈ 2 × (PP-1) × avg_stage_time
bubble_fraction = bubble_latency / pipeline_step_time
```

**各组件数量级参考（70B，TP=8, PP=4, DP=4，seq=8192，节点内 NVSwitch，跨节点 400G IB）**

| 组件 | 典型 P50 | 典型 P95 | 主要 overlap 来源 |
|---|---|---|---|
| microbatch_compute（per stage） | 350-500 ms | 380-540 ms | 无，是基准 |
| tp_collective（per layer，节点内） | 0.4 ms/call，160 calls ≈ 0.6-1 ms exposed | 1-2 ms exposed | 下一层 GEMM（CUDA stream） |
| pp_send_recv（per boundary） | 2.7 ms，3 boundaries ≈ 5-8 ms exposed | 8-15 ms exposed | 下一个 microbatch 某层 compute |
| dp_sync（梯度 AllReduce，DP=4） | 按第 8 章公式，通常 20-60 ms total | 40-80 ms total | backward 最后阶段（DDP bucket） |
| optimizer | 100-140 ms | 130-160 ms | 无（Adam 顺序更新） |
| PP bubble（PP=4，m=32，1F1B） | (4-1)/32 ≈ 9.4% × pipeline_step | 同 | 无 |

**诊断路径**

```text
step_time 高 → 拆分 profiler：

1. PP bubble 明显（stage idle 约占 10% 以上）
   → 增加 microbatch_count（m），或开 interleaved pipeline；先不换 PP size

2. TP collective 暴露（每层 GEMM 间有明显 idle gap）
   → 验证 TP group 在 NVSwitch 内；检查 CUDA_DEVICE_MAX_CONNECTIONS=1；
     运行 nccl-tests 节点内 all_reduce_perf

3. PP send/recv 暴露（stage 结束后空闲等待 recv）
   → 检查 m 是否 ≥ PP（1F1B 稳态需要 m ≥ p）；检查跨节点 IB/RoCE 带宽

4. DP sync 暴露（与第 8 章 §9.2 对齐诊断）
   → 检查 bucket_cap_mb、overlap 配置、DP group 带宽

5. FSDP AllGather 暴露
   → 检查 wrap policy、limit_all_gathers、backward_prefetch

6. CP exchange 暴露（长上下文）
   → 检查 Ring FA 实现是否支持 KV overlap；检查 CP 网络带宽（见 §4.6a）

7. compute 主导（所有通信被完全掩盖）
   → 系统工作正常；优化 kernel（FlashAttention、FP8）或增加 GPU 数量
```

> [!NOTE]
> 上述模型假设 microbatch 内 stage 时间均匀。真实 stage 不均衡（见 §4.3.4）会让 `max_stage()` 明显大于 `avg_stage()`，使 bubble 估算低估实际 stage idle 时间。排查 step time 时应同时检查 stage time 的 P95/P50 比值。
```

- [ ] **Step 3: 同步更新旧 §7.3 和 §7.4 的编号**

将 `### 7.3 70B 状态粗算` 改为 `### 7.4 70B 状态粗算`，将 `### 7.4 405B 状态粗算` 改为 `### 7.5 405B 状态粗算`。

- [ ] **Step 4: 验证插入结果**

运行：

```bash
grep -n "7\.3\|7\.4\|7\.5\|3D Parallel Step Time\|microbatch_compute\|tp_collective\|pp_send_recv\|dp_sync\|cp_exchange\|诊断路径" \
  part3-training-infra/09-model-pipeline-parallel.md
wc -l part3-training-infra/09-model-pipeline-parallel.md
git diff --check -- part3-training-infra/09-model-pipeline-parallel.md
```

预期：找到新 §7.3 标题和 7 个诊断步骤关键词；旧 §7.3/7.4 已更新为 §7.4/7.5；行数再增加 100-130 行；diff check 无输出。

- [ ] **Step 5: Batch 1 交叉验证**

P0 三个 Gap 数字内部一致性检查：

```bash
# 验证 Gap-1 和 Gap-3 使用相同的 70B 参数
grep -A3 "数字示例.*70B" part3-training-infra/09-model-pipeline-parallel.md | \
  grep -E "hidden=|seq=|micro_batch|TP=" | sort | uniq -c
```

预期：hidden=8192、seq=8192、micro_batch=1 在多处出现且一致；TP=8 在 TP 相关段落出现。

- [ ] **Step 6: Commit Task 3**

```bash
git add part3-training-infra/09-model-pipeline-parallel.md
git commit -m "Part3 Gap-3: add 3D parallel step time composition model and diagnostic path"
```

---

## Batch 2: P1 主线（Gap-4 / Gap-5 / Gap-6 / Gap-7）

> Batch 2 可在 Batch 1 commit 后开始；Gap-4 和 Gap-7 互相独立可并行；Gap-6 依赖 Gap-1 已插入（引用 §4.2a 节标题）；Gap-5 依赖 Gap-1/2/3 已插入（引用相关公式）。

---

### Task 4: Gap-4 — GQA / MQA TP 可整除约束

**Files:**
- Modify: `part3-training-infra/09-model-pipeline-parallel.md`（§5.1 配置审查要点后；§8.3 框架支持表）

- [ ] **Step 1: 在 §5.1 配置审查要点之后插入 GQA 约束小节**

定位：

```bash
grep -n "配置审查要点\|KV heads 必须能被 TP 整除\|context-parallel-size" \
  part3-training-infra/09-model-pipeline-parallel.md | head -10
```

在配置审查要点列表的最后一项之后，插入：

```markdown
**GQA / MQA TP 约束专项**

现代 LLM 普遍使用 Grouped Query Attention（GQA），KV heads 远少于 Q heads。**TP size 必须能整除 KV heads**，而不仅仅是 Q heads。

```text
合法性规则（同时满足）：
  num_q_heads   % TP == 0
  num_kv_heads  % TP == 0        ← GQA 额外约束，通常是瓶颈
  hidden_size   % TP == 0
  ffn_size      % TP == 0
  vocab_size    % TP == 0（vocab parallel 模式下为硬约束）
```

**主流 GQA 模型的合法 TP 值**

| 模型 | Q heads | KV heads | 合法 TP 值 | 常见生产 TP |
|---|---|---|---|---|
| Llama-3 8B | 32 | 8 | 1, 2, 4, **8** | 4 或 8 |
| Llama-3 70B | 64 | 8 | 1, 2, 4, **8** | **8** |
| Llama-3 405B | 128 | 8 | 1, 2, 4, **8** | **8**（节点内）|
| Mistral 7B | 32 | 8 | 1, 2, 4, **8** | 4 或 8 |
| Qwen2.5 72B | 64 | 8 | 1, 2, 4, **8** | **8** |
| Qwen2.5 7B | 28 | 4 | 1, 2, **4** | **4** |
| Gemma2 27B | 32 | 16 | 1,2,4,8,**16** | 8 |
| 标准 MHA（如 GPT-3 175B） | 96 | 96 | 1~96（因子集） | 8 或 16 |

> [!DANGER]
> **TP=16 在 8 KV head 模型上无效。** 大多数 GQA 模型的 KV heads 为 8，TP 上限为 8，与单节点 8-GPU NVSwitch 对齐——这是架构约束而非工程选择。尝试 TP=16 时，Megatron-LM 会报错退出，部分 DeepSpeed 版本会产生 silent shape mismatch。

**框架检查行为**

```text
Megatron-LM：assert kv_heads % tp_size == 0，报错退出
DeepSpeed：部分版本不检查，可能 silent mismatch
PyTorch FSDP：不处理 attention head 约束，用户自行保证
Transformer Engine GQA：内置 assertion
```
```

- [ ] **Step 2: 在 §8.3 框架支持表中补充 GQA 约束列**

定位框架支持表（包含 TP/PP/SP/CP 等行的表格），在"主要约束"列为 TP 行追加 `GQA 模型需 KV heads 整除 TP`。

- [ ] **Step 3: 验证**

```bash
grep -n "GQA\|MQA\|KV heads\|num_kv_heads.*TP\|合法 TP 值\|Llama-3 70B.*8.*8" \
  part3-training-infra/09-model-pipeline-parallel.md
git diff --check -- part3-training-infra/09-model-pipeline-parallel.md
```

预期：找到 6 个关键词；diff check 无输出。

- [ ] **Step 4: Commit Task 4**

```bash
git add part3-training-infra/09-model-pipeline-parallel.md
git commit -m "Part3 Gap-4: add GQA/MQA TP divisibility constraints with model table"
```

---

### Task 5: Gap-6 — 3D Parallel 通信重叠架构

**Files:**
- Modify: `part3-training-infra/09-model-pipeline-parallel.md`（§4.2a 之后插入 §4.2b）

- [ ] **Step 1: 定位 §4.2a 末行**

```bash
grep -n "4\.2a\|4\.2b\|4\.3 PP" part3-training-infra/09-model-pipeline-parallel.md | head -10
```

预期：找到 §4.2a（Task 2 中新增）的末行和 §4.3a 的开始行。新 §4.2b 插入两者之间。

- [ ] **Step 2: 插入 §4.2b（通信重叠机制）**

```markdown
### 4.2b 3D Parallel 通信重叠机制

"通信量小"不等于"对 step time 无影响"。每种通信能否被 compute 掩盖，取决于实现条件。

**各类通信的重叠条件**

| 通信类型 | 能与什么重叠 | 实现条件 | 破坏条件 |
|---|---|---|---|
| TP AllReduce（Row parallel 输出） | 下一层的 Column GEMM（前向）、下一层的 backward GEMM | `CUDA_DEVICE_MAX_CONNECTIONS=1` + 独立 CUDA stream（Megatron 默认）| 与 GEMM 在同一 stream；显式 synchronize；未开 async collective |
| PP send/recv（forward activation） | 同 stage 下一个 microbatch 的某些 compute（1F1B 稳态）| 使用异步 `isend`/`irecv`；m ≥ PP（1F1B 稳态前提） | m < PP（warmup 区间 overlap 差）；同步 send/recv API |
| DP gradient AllReduce / ReduceScatter | optimizer step 或下一 step 的 forward | DDP bucket 就绪即启动异步 AllReduce；与 backward 后续计算重叠 | accumulation 中途未用 `no_sync()`；bucket 太大导致 tail 暴露（见第 8 章 §4.5）|
| FSDP AllGather（backward 参数预取） | 前一个 FSDP unit 的 backward compute | `backward_prefetch=BACKWARD_PRE`；FSDP 内部异步 AllGather stream | `limit_all_gathers=True` 过于保守；wrap 粒度太粗 |
| CP KV ring exchange | 本地 context 块的 attention compute（Ring FlashAttention）| Ring FlashAttention 实现（Megatron-Core CP / `ring_flash_attn`）；计算时间 ≥ KV 传输时间 | 未使用 Ring FA（标准 FA 无法重叠 ring exchange）；带宽严重不足时计算来不及覆盖 |

**Profiler 中的健康 vs 不健康形态**

```text
健康（Nsight Systems）：
  NCCL kernel 与 cuBLAS/cuDNN kernel 在时间轴交织；GPU SM active 无大段空洞。

TP AllReduce 暴露（不健康）：
  每层 GEMM 之后有明显 idle gap，AllReduce 结束后才出现下一层 GEMM。
  → 检查 CUDA_DEVICE_MAX_CONNECTIONS 和 TP placement（节点内？）

PP send/recv 暴露（不健康）：
  stage 完成 compute 后，有明显等待 recv 的 idle 段，recv 完成后才恢复 compute。
  → 检查 m vs PP：若 m < PP，进入 warmup 区间，overlap 天然差
  → 检查跨节点带宽：send/recv 时间 > stage compute 时间时无法掩盖

FSDP AllGather 暴露（不健康）：
  backward 中频繁出现 AllGather + idle gap 序列（参数聚合完成前计算停等）。
  → 调整 wrap policy（粒度适中）；确认 backward_prefetch 已开启

CP ring exchange 暴露（不健康）：
  Ring FlashAttention 阶段出现 send/recv 先于 attention compute 结束。
  → 检查带宽（见 §4.6a 估算）；尝试降 CP size 或升级 IB 带宽
```

**调整顺序建议**

```text
出现通信 exposed tail 时，按以下顺序调整：

1. 先确认 placement 正确（TP 在 NVLink/NVSwitch 内，PP stage order 固定）
2. 再看 profiler 是哪类通信暴露（不要凭直觉先调 NCCL env）
3. TP 问题：检查 CUDA_DEVICE_MAX_CONNECTIONS=1，运行节点内 all_reduce_perf
4. PP 问题：检查 m vs PP 关系，确认使用异步 send/recv API
5. DP 问题：对齐第 8 章 §4.5 bucket + overlap 诊断
6. FSDP 问题：调 wrap policy 和 prefetch，不要先调 limit_all_gathers
7. CP 问题：确认 Ring FA 实现，估算 §4.6a 数字后再决定是否降 CP
```
```

- [ ] **Step 3: 验证**

```bash
grep -n "4\.2b\|通信重叠机制\|TP AllReduce.*Row parallel\|PP send/recv.*1F1B\|Ring FlashAttention\|Profiler.*健康\|调整顺序建议" \
  part3-training-infra/09-model-pipeline-parallel.md
git diff --check -- part3-training-infra/09-model-pipeline-parallel.md
```

预期：找到 7 个关键词；diff check 无输出。

- [ ] **Step 4: Commit Task 5**

```bash
git add part3-training-infra/09-model-pipeline-parallel.md
git commit -m "Part3 Gap-6: add 3D parallel communication overlap mechanism and profiler guide"
```

---

### Task 6: Gap-5 — Rank Mesh 推导流程

**Files:**
- Modify: `part3-training-infra/09-model-pipeline-parallel.md`（§8.1 决策树之前插入 §8.0）

- [ ] **Step 1: 定位 §8.1 行号**

```bash
grep -n "^### 8\.1\|^## 8\." part3-training-infra/09-model-pipeline-parallel.md | head -5
```

预期：找到 `## 8. 策略选择` 章节和 `### 8.1 决策树` 的行号。新 §8.0 插入 `## 8.` 章节标题之后、`### 8.1` 之前。

- [ ] **Step 2: 插入 §8.0（Rank Mesh 推导流程）**

```markdown
### 8.0 Rank Mesh 推导：从约束到配置的五步流程

第 9/10 节的 70B/405B worked example 直接给出配置结论。本节补充推导过程，使同样的方法可迁移到其他模型和集群。

**五步推导框架**

```text
Step 1：计算可用 HBM 预算
  available_hbm = gpu_hbm × (1 - fragmentation_ratio - comm_buffer_ratio)
  fragmentation_ratio ≈ 0.10-0.15（经验值；实测用 memory snapshot 校准）
  comm_buffer_ratio ≈ 0.05-0.08（NCCL、FSDP AllGather buffer、PP buffer）

Step 2：确定最小 TP（解决单层 GEMM 峰值）
  单层 activation peak（BF16，无 AC）≈ batch × seq × hidden × 20 bytes（见 §4.1a）
  除以 TP 后 ≤ available_hbm × 0.25（留给其他项）→ 确定 min_tp

  GQA 约束（见 §5.1 GQA 专项）：TP 必须整除 kv_heads
  → 最终 TP = max(min_tp，满足 GQA 的最小合法值) 且 ≤ 节点 GPU 数

Step 3：确定最小 PP（解决整网层数和状态）
  per_rank_params = total_params × 2 bytes / TP      （BF16，不含 optimizer）
  per_rank_optim  = total_params × 12 bytes / (TP × ZeRO_degree × PP)（AdamW，ZeRO 切分后）
  per_rank_activ  = num_layers/PP × batch × seq × hidden × AC_factor × 2（见 §4.1a）

  若 per_rank_params + per_rank_optim + per_rank_activ > available_hbm：
    增大 PP，直到满足预算；min_pp = ceil(合计 / available_hbm) 取上界

Step 4：计算 DP
  dp = world_size / (TP × PP)
  验证 dp ≥ 1；dp 太小（如 dp=1）意味着没有样本并行，可考虑减少 PP 或 TP

Step 5：验证 bubble 和带宽
  microbatch_count m = global_batch / (micro_batch_size × dp)
  必须保证 m ≥ PP（1F1B 稳态条件，否则 bubble 恶化，见 §4.3.1）

  bubble（1F1B） = (PP-1) / m
  若 bubble > 20%：优先增加 m（增大 global_batch 或降 micro_batch_size）；
                   次选 interleaved pipeline；最后再考虑减少 PP

  TP 带宽（节点内 NVSwitch）：用 §4.2a 公式估算 per-call 时间，与 GEMM 时间对比
  PP 带宽（跨节点 IB）：用 §4.3a 公式估算 per-boundary 时间，与 stage compute 对比
```

**推导示例：180B dense 模型，256 GPU，400G IB**

```text
输入：
  180B dense，hidden=12288，layers=96，Q heads=96，KV heads=8，ffn=4×hidden
  集群：32 nodes × 8 H100 80GB（256 GPU），节点内 NVSwitch，跨节点 400G IB
  目标：seq=8192，BF16，AdamW，global_batch=2048

Step 1：available_hbm = 80 GB × (1 - 0.12 - 0.06) = 66 GB

Step 2：KV heads=8 → TP ∈ {1,2,4,8}；min_tp 检查：
  single-layer activation（无 AC，TP=1）= 1 × 8192 × 12288 × 20 = 2.4 GB
  → TP=1 时单层 activation 2.4 GB < 66 GB × 0.25 = 16.5 GB，层级上 TP=1 可行
  但整网状态（见 Step 3）需要 TP≥4；选 TP=8（最大合法值，节点内最优）

Step 3：per_rank_params（TP=8，PP=1）= 180B × 2 / 8 = 45 GB
  per_rank_optim（ZeRO-1，PP=1，256 DP）= 180B × 12 / 256 = 8.4 GB（optimizer shard）
  per_rank_activ（无 PP，AC=selective+FA，factor≈8）= 96 × 1 × 8192 × 12288 × 8 × 2 = 126 GB
  合计 = 45 + 8.4 + 126 = 179.4 GB >> 66 GB → 需要 PP

  尝试 PP=8：
    per_rank_params = 45 × (12/96) = 5.6 GB（每 rank 12 层）
    per_rank_activ  = 12 × 8 × 12288 × 8192 × 2 ≈ 15.8 GB
    合计 ≈ 5.6 + 8.4 + 15.8 + comm_buffer(5 GB) = 34.8 GB ✓（< 66 GB）

Step 4：DP = 256 / (8 × 8) = 4

Step 5：m = 2048 / (1 × 4) = 512；bubble = (8-1)/512 = 1.4% ✓（m >> PP）

  TP 通信估算（per call）= 2 × (7/8) × 8192 × 12288 × 2 = 281 MB
    NVSwitch 600 GB/s → 0.47 ms per call，96 层 × 2 = 192 calls ≈ 90 ms 理论上界
    实际 GEMM 时间远超 0.47 ms/call，几乎完全被掩盖 ✓

  PP 边界通信（TP=8 时 hidden/8 = 1536）：
    pp_boundary = 1 × 8192 × 1536 × 2 = 25 MB per boundary
    400G IB（50 GB/s）→ 0.5 ms per boundary，7 个 boundary ≈ 3.5 ms exposed（可接受）✓

结论：TP=8, PP=8, DP=4 是可行起点。下一步：HBM dry-run（preflight）+ nccl-tests。
```
```

- [ ] **Step 3: 验证**

```bash
grep -n "8\.0\|Rank Mesh 推导\|五步推导框架\|Step 1.*HBM\|Step 2.*TP\|Step 3.*PP\|Step 4.*DP\|Step 5.*bubble\|180B.*推导示例" \
  part3-training-infra/09-model-pipeline-parallel.md
wc -l part3-training-infra/09-model-pipeline-parallel.md
git diff --check -- part3-training-infra/09-model-pipeline-parallel.md
```

预期：找到 9 个关键词；行数再增加 120-160 行；diff check 无输出。

- [ ] **Step 4: Commit Task 6**

```bash
git add part3-training-infra/09-model-pipeline-parallel.md
git commit -m "Part3 Gap-5: add rank mesh derivation 5-step framework with 180B example"
```

---

### Task 7: Gap-7 — Stage 负载均衡操作指导

**Files:**
- Modify: `part3-training-infra/09-model-pipeline-parallel.md`（§4.3.3 之后插入 §4.3.4；§11 故障排除表新增一行）

- [ ] **Step 1: 在 §4.3.3（工程含义）之后插入 §4.3.4**

定位：

```bash
grep -n "4\.3\.3\|4\.3\.4\|工程含义\|生产几乎都用" \
  part3-training-infra/09-model-pipeline-parallel.md | head -10
```

插入内容：

```markdown
#### 4.3.4 Stage 负载均衡

bubble 公式假设所有 stage 计算时间相同。真实集群中首尾 stage 通常更慢，导致实测吞吐低于公式预测。

**量化不均衡**

```text
stage_imbalance_ratio = max(stage_time_p50_per_stage) / mean(stage_time_p50_per_stage) - 1

门限（参考值）：
  < 5%：可接受，公式误差范围内
  5-15%：显著，应用 layer redistribution 改善
  > 15%：强制修复，interleaved pipeline 也无法弥补
```

观测方式：从 training metric 中按 `pp_stage` 维度聚合 `microbatch_compute_time`；或在 Nsight Systems 中按 stage 分组比较 compute kernel 时长。

**常见不均衡来源与修复**

| 来源 | 识别方式 | 修复方式 |
|---|---|---|
| Stage 0 含 embedding | Stage 0 compute 慢 10-30% | 将 embedding 单独作为 stage 0（0 层 Transformer block） |
| Stage N-1 含 LM head + loss | 最后 stage 显著更慢 | Vocab parallel 切分 LM head；或让 LM head 独占最后 stage |
| 序列长度倾斜（data skew） | 与第 8 章 data skew 症状一致（rank token skew P95 高） | 按第 8 章 §10.6 处理，不是 stage 负载均衡问题 |

**Megatron 非均匀切分**

Megatron 默认按均匀层数切分 stage。调整时：

```bash
# 通过 --num-layers-per-virtual-pipeline-stage 配合 interleaving 改变 virtual stage 粒度
# 自定义不均匀切分需修改 megatron/core/pipeline_parallel/schedules.py 中的 partition 函数

# DeepSpeed PipelineModule 提供 partition_method="parameters"（按参数量切分）
# 对 embedding/LM head 不均有一定改善，但不精确
model = PipelineModule(
    layers=layers,
    num_stages=8,
    partition_method="parameters",  # 而非默认 "uniform"
)
```

> [!WARNING]
> 在修复 stage 不均衡之前，先确认 stage 时间差来自 compute（layer 负载），而不是来自 data skew（某些 rank 样本更长）。两种症状都表现为"某个 stage 慢"，但修复方式完全不同。
```

- [ ] **Step 2: 在 §11 故障排除表新增 stage 不均衡行**

定位 §11 故障排除表（含"症状 / 证据 / 可能根因 / 处理动作"四列的表格），在 `pipeline bubble 太高` 行之后增加：

```markdown
| stage 时间不均 | `microbatch_compute_time{pp_stage=X}` P50 中某 stage 比其他 stage 慢 10%+；stage P95/P50 比值高；bubble 公式估算与实测差距 > 5% | embedding/LM head 在普通 stage；vocab parallel 未启用；data skew 误判为 stage 不均 | 按 `pp_stage` 聚合 compute time；将 embedding/LM head 单独处理；排查 data skew（见第 8 章 §10.6）；调整 partition method |
```

- [ ] **Step 3: 验证**

```bash
grep -n "4\.3\.4\|Stage 负载均衡\|stage_imbalance_ratio\|embedding.*stage\|LM head.*stage\|partition_method.*parameters\|stage 时间不均" \
  part3-training-infra/09-model-pipeline-parallel.md
git diff --check -- part3-training-infra/09-model-pipeline-parallel.md
```

预期：找到 7 个关键词；diff check 无输出。

- [ ] **Step 4: Commit Task 7**

```bash
git add part3-training-infra/09-model-pipeline-parallel.md
git commit -m "Part3 Gap-7: add pipeline stage load balancing measurement and fix guidance"
```

---

## Batch 3: P2 前沿与转换路径（Gap-8 / Gap-9 / Gap-10）

> Batch 3 可在 Batch 2 commit 后开始；三个 Gap 互相独立，可并行执行。

---

### Task 8: Gap-8 — FP8 / Transformer Engine 与并行策略交互

**Files:**
- Modify: `part3-training-infra/09-model-pipeline-parallel.md`（§5.1 新增 FP8 参数说明；§8.3 补充 FP8 行；§13.3 框架检查项补充）

- [ ] **Step 1: 在 §5.1 Megatron 配置示例之后增加 FP8 参数块**

定位：

```bash
grep -n "use-distributed-optimizer\|sequence-parallel\|Transformer Engine\|--fp8\|transformer_engine" \
  part3-training-infra/09-model-pipeline-parallel.md | head -10
```

在 §5.1 配置示例（bash 代码块）之后、配置审查要点之前，插入：

```markdown
**FP8 / Transformer Engine 配置（H100/H800/Blackwell）**

H100 及后续架构支持 FP8 训练，通过 NVIDIA Transformer Engine（TE）实现。与并行策略的交互点：

```bash
# Megatron + Transformer Engine FP8 关键参数
torchrun ... pretrain_gpt.py \
  ...
  --fp8-format hybrid \            # E4M3 forward，E5M2 backward
  --fp8-amax-compute-algo max \    # amax 计算方式
  --fp8-amax-history-len 16 \      # 滑动窗口长度
  --transformer-impl transformer_engine  # 启用 TE kernel
```

**FP8 与 TP 的交互：scaling factor 同步**

```text
FP8 per-tensor scaling factor 在 TP 内所有 rank 必须相同：
  - TP 把一个逻辑 tensor 切成多份，scaling 必须对应同一逻辑 amax
  - Transformer Engine 通过 TP group 内 allreduce amax 自动同步
  - 生产配置必须确认 TE 版本支持目标 TP size 的 amax allreduce

FP8 checkpoint 额外状态：
  - 每层有 amax_history（默认 16 步滑动窗口）和 scale factor
  - FSDP/ZeRO checkpoint 需包含这些 metadata，否则 FP8 scale 冷启动
  - 冷启动不影响正确性，但导致训练初期 loss 不稳定（scale 收敛过程）
```

**FP8 与 PP 的交互：stage 边界 activation dtype**

```text
PP send/recv 的 activation 可用 BF16 或 FP8：
  - BF16（默认）：精度安全，占用 2× 带宽
  - FP8（显式开启）：带宽减半，但引入量化误差积累风险

生产建议：PP 边界 activation 默认 BF16；仅在带宽严重不足且 FP8 量化误差经过验证后才使用 FP8 send/recv。
```
```

- [ ] **Step 2: §8.3 框架支持表新增 FP8/TE 行**

在 §8.3 框架支持表（Strategy × Framework 矩阵）的最后增加：

```markdown
| FP8（via TE） | 强（TE 原生）| 视 TE 集成版本 | 有限（需外部 TE wrapper）| TP amax 同步、PP 边界 dtype、checkpoint amax history |
```

- [ ] **Step 3: §13.3 框架检查项新增 FP8 专项**

在 §13.3 的 checklist 中，`FP8/BF16/Transformer Engine 与并行策略兼容` 行之后扩写：

```markdown
- [ ] FP8 训练时确认 TE 版本支持 TP size 的 amax allreduce。
- [ ] FP8 checkpoint 包含 amax_history 和 scale factor；恢复后验证 scale 不重置。
- [ ] PP stage 边界 activation dtype 已明确（默认 BF16）。
```

- [ ] **Step 4: 验证**

```bash
grep -n "FP8\|Transformer Engine\|amax.*history\|scaling factor.*TP\|stage 边界.*dtype\|fp8-format\|transformer_impl" \
  part3-training-infra/09-model-pipeline-parallel.md
git diff --check -- part3-training-infra/09-model-pipeline-parallel.md
```

预期：找到 7 个关键词；diff check 无输出。

- [ ] **Step 5: Commit Task 8**

```bash
git add part3-training-infra/09-model-pipeline-parallel.md
git commit -m "Part3 Gap-8: add FP8/Transformer Engine parallel interaction notes"
```

---

### Task 9: Gap-9 — Zero Bubble Pipeline 工程细节

**Files:**
- Modify: `part3-training-infra/09-model-pipeline-parallel.md`（§4.4 virtual stage 节扩写）

- [ ] **Step 1: 定位 §4.4 末尾**

```bash
grep -n "^### 4\.4\|^### 4\.5\|zero bubble.*调度\|zero_bubble\|W-pass\|platform 侧不需要" \
  part3-training-infra/09-model-pipeline-parallel.md | head -10
```

在 §4.4 当前内容（`platform 侧不需要自己实现算法，但必须知道它改变证据形态` 段落）之后追加以下内容：

```markdown
**Zero Bubble Pipeline 工程细节**

Zero Bubble（ZB1P）的关键创新是把传统 backward 拆分为两个可独立调度的计算单元：

```text
B-pass（backward input gradient）：
  计算 dL/dX（输入梯度），用于向前一个 stage 传播梯度。
  必须在前一 stage 的 B-pass 开始前完成（依赖链不变）。

W-pass（backward weight gradient）：
  计算 dL/dW（权重梯度），累积到 gradient buffer。
  不需要立即完成——只要在 optimizer step 前完成即可。
  → 可以推迟到 pipeline warmup 的空槽中执行，填充原本空闲的 stage。
```

**工程代价**

| 代价项 | 说明 |
|---|---|
| Activation 保留时间延长 | W-pass 推迟时，对应 forward 的 activation 必须保留到 W-pass 执行完毕。推迟越多，activation 常驻量越高，可能超过标准 1F1B |
| Optimizer step 时序 | 调度器必须跟踪每个 microbatch 的 W-pass 状态，确保全部完成才触发 optimizer step |
| Profiler 证据形态变化 | Nsight 中不再是整齐的 F/B/W 块；checkpoint 和 profiler tag 必须能识别 B-pass 和 W-pass 两类 micro-op |

**框架支持现状（2026-05）**

```text
Megatron-Core：实验分支，可用 --pipeline-schedule ZB1P 触发
  → 需要验证 checkpoint metadata 是否记录 W-pass 状态

DeepSeek-V3 训练：自研实现，1024 GPU 规模下验证
  → 报告 bubble 从 ~10% 降到 ~1%；对 PP=16、m=64 场景收益最大

PyTorch：暂无原生支持，需要外部 pipeline schedule 库
  → 使用前必须做 20 step loss continuity 对比（B/W 拆分后梯度累积正确性）
```

**何时值得引入**

```text
Zero Bubble 收益最大的场景：
  PP 阶段数 ≥ 8，microbatch 数 m 受限（无法通过增大 m 降低 bubble）。

不值得引入的场景：
  m >> PP（bubble 已 < 5%，增加实现复杂度不划算）；
  框架未有原生支持（需要自行实现 B/W 拆分，引入正确性风险）；
  activation 内存已经紧张（W-pass 推迟会进一步增加 activation 驻留）。
```
```

- [ ] **Step 2: 验证**

```bash
grep -n "B-pass\|W-pass\|ZB1P\|pipeline-schedule ZB1P\|Activation 保留时间延长\|Optimizer step 时序\|DeepSeek.*bubble" \
  part3-training-infra/09-model-pipeline-parallel.md
git diff --check -- part3-training-infra/09-model-pipeline-parallel.md
```

预期：找到 7 个关键词；diff check 无输出。

- [ ] **Step 3: Commit Task 9**

```bash
git add part3-training-infra/09-model-pipeline-parallel.md
git commit -m "Part3 Gap-9: expand zero bubble pipeline with engineering details and framework status"
```

---

### Task 10: Gap-10 — 推理侧 Checkpoint 转换机制

**Files:**
- Modify: `part3-training-infra/09-model-pipeline-parallel.md`（§8.4 之后新增 §8.5）

- [ ] **Step 1: 定位 §8.4 末行**

```bash
grep -n "^### 8\.4\|^### 8\.5\|^## 9\.\|推理转换要提前设计\|训练 EP" \
  part3-training-infra/09-model-pipeline-parallel.md | head -10
```

在 `训练 EP：推理 router、expert placement` 段落之后，插入 §8.5。

- [ ] **Step 2: 插入 §8.5（推理 Checkpoint 转换机制）**

```markdown
### 8.5 推理侧 Checkpoint 转换机制

训练 checkpoint 到推理 runtime 的转换是生产最后一公里，也是最常踩坑的环节。

**TP 转换：Column parallel 和 Row parallel 的拼接方向不同**

```text
Column parallel（Q/K/V projection，FC1 output dim 切分）：
  merge：torch.cat(shards, dim=0)      ← 沿 output dim 拼接
  reshard（训练 TP=8 → 推理 TP=4）：torch.chunk(merged, 4, dim=0)

Row parallel（attention output proj，FC2 input dim 切分）：
  merge：torch.cat(shards, dim=1)      ← 沿 input dim 拼接
  reshard：torch.chunk(merged, 4, dim=1)

常见错误：对 Row parallel 层用 dim=0 拼接 → 权重 shape 正确但含义错误 → loss 正常但 logit 质量下降。
验证方法：对 10 个 token 比较训练推理 logit 相对差，阈值 < 1e-3（BF16）。
```

**PP 转换（flatten）：必须知道 layer-to-stage 映射**

```text
训练 checkpoint 目录结构（PP=4，每 stage 20 层）：
  ckpt/stage0/model.pt  → layers 0-19
  ckpt/stage1/model.pt  → layers 20-39
  ckpt/stage2/model.pt  → layers 40-59
  ckpt/stage3/model.pt  → layers 60-79

flatten 步骤：
  1. 读取 parallel_metadata.json，确认 layer-to-stage 映射
  2. 按 layer_id 顺序合并：full_model = [stage0_layers, stage1_layers, ...]
  3. 重命名 key（Megatron layer.{i} → HuggingFace model.layers.{i}）

tied embedding（首尾 stage 共享）：
  - 训练时 embedding 在 stage 0，LM head 在 stage N-1
  - merge 时只取 stage N-1 的 LM head weight（两者 gradient 更新历史可能略有差异）
  - 或显式校验两者 cosine similarity > 0.9999 后取均值
```

**常用工具链**

| 工具 | 用途 | 适用场景 | 注意事项 |
|---|---|---|---|
| Megatron `tools/checkpoint/convert_checkpoint.py` | TP/PP reshape，导出 HuggingFace 格式 | Megatron 训练 checkpoint | tied embedding 需要单独验证 |
| DeepSpeed `zero_to_fp32.py` | ZeRO-3 checkpoint 聚合为完整 FP32 权重 | DeepSpeed ZeRO-3 | 不处理 TP/PP，需先聚合再做 TP/PP 转换 |
| HuggingFace `from_pretrained` + `save_pretrained` | HuggingFace 格式互转 | 标准 HuggingFace 模型 | 需要对应的 `modeling_xxx.py` 支持并行格式 |
| vLLM `convert_megatron_checkpoint.py`（社区） | Megatron → vLLM 可读格式 | vLLM 部署 | 社区维护，稳定性不一，需要版本锁定 |

**推理转换 preflight 流程**

```text
1. 保存训练 checkpoint（包含 parallel_metadata.json）
2. 运行 convert_checkpoint.py（或等价工具），指定目标 TP/PP
3. 用推理 runtime 加载转换后权重，对 10-100 个 prompt 做 logit 对比（vs 训练推理 forward）
4. 确认 logit 最大绝对差 < 1e-2（BF16 容忍范围）
5. 如有 tied embedding，显式验证 embed_tokens.weight == lm_head.weight（或 cosine > 0.9999）
6. 在 preflight gate 中记录转换工具版本、权重 hash 和 logit diff 最大值
```

> [!WARNING]
> 推理转换不要等到"需要部署时"才演练。训练开始后每次并行 shape 变更（TP/PP 调整）都会改变 checkpoint 格式，最晚应在 preflight 阶段完成第一次转换 dry-run，确认转换工具与当前 checkpoint schema 兼容。
```

- [ ] **Step 3: 验证**

```bash
grep -n "8\.5\|推理侧 Checkpoint 转换\|Column parallel.*dim=0\|Row parallel.*dim=1\|layer-to-stage\|tied embedding\|zero_to_fp32\|convert_checkpoint\|logit.*对比" \
  part3-training-infra/09-model-pipeline-parallel.md
wc -l part3-training-infra/09-model-pipeline-parallel.md
git diff --check -- part3-training-infra/09-model-pipeline-parallel.md
```

预期：找到 9 个关键词；行数比基准（Task 1 前 1239 行）增加至少 650 行（P0+P1+P2 全部完成）；diff check 无输出。

- [ ] **Step 4: Batch 3 完成验证**

```bash
# 确认所有 10 个 Gap 的核心关键词均存在
rg -n "4\.1a|4\.2a|4\.2b|4\.3a|4\.6a|7\.3 3D Parallel|8\.0 Rank Mesh|4\.3\.4|8\.5 推理侧|GQA.*合法 TP|FP8.*TP|B-pass|W-pass" \
  part3-training-infra/09-model-pipeline-parallel.md | wc -l
# 预期：至少 12 行匹配（每个 Gap 至少 1 个核心标记）

# 确认 Ch8 交叉引用
grep -n "第9章.*4.1a\|Activation 内存" part3-training-infra/08-data-parallel.md

# 确认无残留 TODO/TBD
rg -n "TODO|TBD|FIXME|待补|后续补" \
  part3-training-infra/08-data-parallel.md \
  part3-training-infra/09-model-pipeline-parallel.md

git diff --check -- part3-training-infra/08-data-parallel.md part3-training-infra/09-model-pipeline-parallel.md
git status --short -- part3-training-infra/
```

预期：Gap 核心标记 ≥ 12 行；Ch8 有交叉引用；无 TODO/TBD；diff check 无输出；status 干净（已 commit）。

- [ ] **Step 5: Commit Task 10**

```bash
git add part3-training-infra/09-model-pipeline-parallel.md
git commit -m "Part3 Gap-10: add inference checkpoint conversion mechanics and tool chain"
```

---

## Execution Notes

- **数字一致性**：三个 P0 Gap 使用同一套 70B 参数（hidden=8192, layers=80, seq=8192）；Gap-5 使用 180B 示例，与 worked example 不冲突。执行时检查新增公式数字与已有 §9 worked example 的数字是否内部一致（尤其是 TP AllReduce bytes 和 activation GB 数字）。
- **Gap-1 数值修正**：spec 文档中 Gap-1 的 `bytes_per_rank per call` 计算遗漏了 Ring AllReduce 的 `×2` 系数（ReduceScatter + AllGather 两趟）。Task 2 中已按正确公式（含 `×2`）给出：`235 MB per call`，与 Chapter 8 的 Ring AllReduce 公式一致。
- **节编号顺延**：Task 3 中将旧 §7.3/§7.4 顺延为 §7.4/§7.5；Task 6 中新增 §8.0。执行时同步检查 `## 15. 练习题` 等引用 `§7.3`、`§8.1` 的位置是否需要更新。
- **HTML 不在本计划范围内**：所有改动仅涉及 `.md` 源文件。HTML 重新生成应作为独立后续任务处理。
- **并发约束**：Batch 2 内 Gap-4（Task 4）和 Gap-7（Task 7）可并行；Batch 3 内三个 Gap（Task 8/9/10）互相独立可并行。不超过 3 个并发 subagent。
