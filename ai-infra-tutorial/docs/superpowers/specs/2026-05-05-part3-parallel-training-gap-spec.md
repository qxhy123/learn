# Part 3 并行训练策略补充规格

> 日期：2026-05-05
> 范围：`part3-training-infra/08-data-parallel.md`、`part3-training-infra/09-model-pipeline-parallel.md`
> 分析视角：资深 AI Infra Engineer（负责 LLM 训练平台、容量评估、效率优化、故障排查）
> 输出形态：缺口分析 + 内容补充规格，不含源文件修改，按 P0/P1/P2 三级优先级组织

---

## 1. 背景与分析方法

### 1.1 分析范围

本次精读覆盖：

- `08-data-parallel.md`：1502 行，约 1.0 万字有效内容
- `09-model-pipeline-parallel.md`：1239 行，约 0.9 万字有效内容

参照系：一位资深 AI Infra Engineer 在以下场景中使用本教程：

1. 为新模型规模（如 70B → 200B）设计并行策略，需要从约束出发推导 TP/PP/DP 配置；
2. 在生产集群上做 preflight 带宽准入，需要估算 TP/PP/CP 通信量；
3. 调试混合并行 step time 退化，需要定位哪个维度的通信成为瓶颈；
4. 引入 GQA 架构（Llama-3/Qwen2）或 FP8 训练时，判断现有并行配置是否需要调整；
5. 将训练 checkpoint 转换到推理 runtime。

### 1.2 评分标准

| 维度 | 描述 |
|---|---|
| 公式完整性 | 给出的估算公式是否支持真实 admission 决策，而不仅是数量级感知 |
| 推导可迁移性 | Worked example 能否教会读者用同样方法处理其他模型，而不只是给出结论 |
| 现代架构覆盖 | 是否覆盖 GQA/MQA、FP8、Zero Bubble 等当前主流生产方案 |
| 故障分析支撑 | 给出的 step time 模型和通信重叠框架是否足以作为 debug 起点 |
| 执行指导精度 | 是否能从章节直接导出可执行的配置决策和工具命令 |

### 1.3 总体评价

两章整体质量在技术教程中属于较高水平：架构路径、状态路径、故障路径的分层结构清晰；第 8 章的 ZeRO 显存 reduction 数字和 bubble 公式覆盖较扎实。

但从 AI Infra Engineer 的实操角度，仍有 **10 个实质性缺口**，其中 3 个 P0 缺口会导致工程师在关键决策点拿不到所需信息，只能凭经验或查其他资料。

---

## 2. P0 缺口：核心工程公式缺失

> P0 缺口影响准入决策和性能预算，不补充则章节的"容量与效率"体系不完整。

---

### Gap-1：TP / PP / CP 通信量公式缺失

**现状**

第 8 章（数据并行）给出了 Ring AllReduce 的心算公式：

```text
bytes_per_rank ~= 2 * (N-1)/N * gradient_bytes
comm_time ~= bytes_per_rank / effective_bus_bandwidth + collective_latency
```

并用 7B 模型数字（26.4 GB per rank，220 ms）做了 worked example 演示。

第 9 章在 4.2 节（TP 通信边界）和 7.2 节（PP bubble 效率）分别定性描述通信，但从未给出等价的通信量公式。章节唯一一处接近定量的表述是：

> `TP_comm_exposed = max(TP_collective_time - overlap_with_compute, 0)`

这个表达式的左边（`TP_collective_time`）没有估算方法，工程师仍然不知道 TP collective 在数值上有多大。

**问题**

没有通信量公式，工程师无法：

- 在 preflight 前判断集群带宽是否支撑目标 TP/PP/CP 配置；
- 在 step time 退化时区分"通信太重"还是"overlap 太差"；
- 在跨节点 TP 和节点内 TP 之间做定量对比，而不是只靠"跨节点 TP 不好"的经验判断。

**需要补充的内容**

**TP 通信量估算（每层 Transformer block）**

以 Megatron 标准 Column-then-Row 两段式 TP 实现为例：

```text
每个 Transformer block 的 TP AllReduce 次数（forward + backward）：
  - MLP：forward 1次 AllReduce（Row parallel output），backward 1次
  - Attention：forward 1次 AllReduce（Row parallel output），backward 1次
  - 合计：每 block forward 约 2次，backward 约 2次 AllReduce

每次 AllReduce 数据量（以 Ring AllReduce 计）：
  tokens_per_rank = micro_batch_size × seq_len     （TP 内每 rank 看相同 sequence，DP 切 batch）
  allreduce_bytes_per_call = tokens_per_rank × hidden_size × bytes_per_element × 2 × (TP-1)/TP

单个 Transformer block 一次 forward 的 TP AllReduce 总量：
  tp_comm_bytes_per_block_fwd = 2 × tokens_per_rank × hidden_size × dtype_bytes × 2 × (TP-1)/TP
```

数字示例（70B，BF16，seq=8192，micro_batch=1，TP=8）：

```text
tokens_per_rank = 1 × 8192 = 8192
hidden_size = 8192
allreduce_bytes_per_call = 8192 × 8192 × 2 × (7/8) = 117,964,800 bytes ≈ 118 MB
每 block forward 2次 AllReduce ≈ 236 MB
70B 有 80 层，整 model forward TP AllReduce 总量 ≈ 80 × 236 MB ≈ 18.9 GB

节点内 NVSwitch 带宽约 600 GB/s（AllReduce 有效带宽），
单次 AllReduce 约 118 MB / 600 GB/s ≈ 0.2 ms per call
80 层 × 2次 = 160次，理论通信时间约 32 ms（不算 overlap）
```

> 说明：实际 TP collective 在 Megatron 中是 AllReduce，但也可以用 ReduceScatter + AllGather（SP 模式下）。SP 模式的通信量近似相同，区别在于降低了 sequence 维度的 activation 常驻量。

**PP 通信量估算（每个 microbatch，每个 stage 边界）**

```text
activation send/recv per stage boundary per microbatch（forward）：
  pp_fwd_bytes = micro_batch_size × seq_len × hidden_size × dtype_bytes

activation send/recv per stage boundary（backward）：
  pp_bwd_bytes = micro_batch_size × seq_len × hidden_size × dtype_bytes

单条 pipeline 一个 microbatch 完整走一遍的 PP 通信总量：
  pp_total_bytes = 2 × (PP - 1) × pp_fwd_bytes   （每个中间边界 forward+backward 各一次）
```

数字示例（70B，BF16，seq=8192，micro_batch=1，PP=4）：

```text
pp_fwd_bytes = 1 × 8192 × 8192 × 2 = 134 MB per boundary
PP=4 有 3 个中间边界
单 microbatch forward+backward PP 通信量 ≈ 2 × 3 × 134 MB = 804 MB

跨节点 IB/RoCE 400 Gbps = 50 GB/s，单边界 134 MB 约 2.7 ms
3 个边界串行约 8 ms（但 1F1B 下 send/recv 与 compute 部分重叠）
```

**CP 通信量估算（Ring KV 交换，每个 attention layer，每个 CP ring step）**

```text
ring KV exchange per attention layer per ring step（CP 个 rank 的 ring）：
  kv_bytes_per_step = micro_batch_size × seq_len/CP × num_kv_heads × head_dim × 2(K+V) × dtype_bytes

一个 attention layer 完整 CP ring 通信量（需要 CP-1 次 ring step）：
  cp_total_per_layer = (CP - 1) × kv_bytes_per_step
```

数字示例（70B GQA：8 KV heads，head_dim=128，seq=65536，CP=4，BF16）：

```text
kv_bytes_per_step = 1 × (65536/4) × 8 × 128 × 2 × 2 = 536 MB
3 ring steps per layer，每层 CP 通信量 ≈ 1.6 GB
80 层 × 1.6 GB ≈ 128 GB per step（forward+backward 约 ×2）
```

> 这解释了为什么 CP 要求高速互联：65K 上下文 CP=4 时，每 step KV 通信量已达百 GB 量级，100GbE 必然成为严重瓶颈。

**建议插入位置**：第 9 章 §4.2（TP 的通信边界）后新增 §4.2a，§4.3 后新增 §4.3a，§4.6 后新增 §4.6a，分别承载 TP/PP/CP 的公式段。或统一新增 §4.8（各维度通信量估算汇总表）与第 8 章 §4.8 形成对称结构。

**预估篇幅**：约 120-160 行 Markdown（含公式、示例、说明）。

---

### Gap-2：Activation 内存定量模型缺失

**现状**

第 9 章 §4.1（最小容量账本）列出了 HBM 的组成项：

```text
HBM_rank = parameter_shards + gradient_shards + optimizer_shards
         + activation_resident + attention_workspace
         + communication_buffers + fragmentation_margin
```

但 `activation_resident` 和 `attention_workspace` 始终以"取决于 batch、sequence、layer、AC 策略大幅变化"带过，从未给出公式。第 8 章 §5.3 的 ZeRO 说明同样有一处 WARNING：

> "以上数字不含 activation。激活随 batch、sequence length、layer、AC 策略大幅变化"

两章都指出 activation 重要，但都没有量化它。

**问题**

activation 往往是决定 microbatch 上限、是否需要 AC、AC 粒度的核心约束。没有公式，工程师在做 HBM admission 时只能凭经验或试错，导致：

- 低估 activation 导致 OOM；
- 高估 activation 导致不必要开启 full recompute，浪费算力；
- 无法对比 selective recompute vs full recompute 的 memory/compute 权衡。

**需要补充的内容**

**Activation 内存 per Transformer block（无 AC）**

对标准 decoder-only Transformer（无 GQA 简化，BF16）：

```text
per_block_activation_bytes ≈
  # QKV projection input
  + batch × seq × hidden × 2                              # input to attn
  # Attention scores (quadratic！)
  + batch × num_heads × seq × seq × 2                    # softmax scores（标准 MHA）
  # Attention output
  + batch × seq × hidden × 2
  # MLP activation
  + batch × seq × intermediate_size × 2                  # 通常 intermediate = 4 × hidden
  # Residual / layer norm inputs（约 2-3处）
  + batch × seq × hidden × 2 × 3

简化估算（标准 MHA，intermediate=4×hidden）：
  ≈ batch × seq × hidden × (1 + 1 + 1 + 4 + 3) × 2 bytes
  ≈ batch × seq × hidden × 20 bytes（BF16，无 AC）
```

FlashAttention 隐式 recompute 对 activation 的影响：

```text
FlashAttention 不存储 O(seq²) 的 attention score，只存 softmax LSE（log-sum-exp）：
  lse_bytes = batch × num_heads × seq × 4 bytes（FP32）

使用 FlashAttention 后，attention 部分 activation 减少：
  节省 ≈ batch × num_heads × seq × seq × 2 bytes

对 seq=8192，64 heads，batch=1：
  节省 = 1 × 64 × 8192 × 8192 × 2 = 8.6 GB per layer
```

> 这解释了为什么 FlashAttention 在长序列下是必选项，而不仅是"速度更快"。

**不同 AC 策略的 per-block activation 对比表**

| AC 策略 | Activation 常驻量 | 额外 FLOPs | 典型场景 |
|---|---|---|---|
| 无 AC（全量存储） | ~20× batch×seq×hidden bytes | 0 | 短序列、HBM 充足 |
| Full recompute（每层重算） | ~2× batch×seq×hidden bytes（只存 block 输入） | +33% FLOPs（多一次 forward） | HBM 极度紧张 |
| Selective recompute（只重算 attention） | ~12× batch×seq×hidden bytes（存 MLP，重算 attn） | +~15% FLOPs | 平衡选项 |
| FlashAttention（无额外 AC） | ~12× batch×seq×hidden bytes（自动节省 attn score） | 0（kernelfusion） | 生产首选 |
| Selective + FlashAttention | ~8× batch×seq×hidden bytes | 接近 0 | 长序列生产标准 |

数字示例（70B，hidden=8192，seq=8192，micro_batch=1，BF16，80层）：

```text
无 AC：80 × 20 × 8192 × 8192 × 2 bytes ≈ 214 GB
Full recompute：80 × 2 × 8192 × 8192 × 2 bytes ≈ 21.5 GB
Selective + FlashAttention：80 × 8 × 8192 × 8192 × 2 bytes ≈ 85.9 GB
```

> 这三行数字说明了为什么生产 70B 训练必须开 FlashAttention + Selective AC：无 AC 的 214 GB 直接超过整个 8×80GB 节点的总 HBM。

**建议插入位置**：第 9 章 §4.1（最小容量账本）之后，新增 §4.1a（Activation 内存估算）；或在 §7.3（70B 状态粗算）后新增对 activation 的 per-block 估算，使容量账本完整。同时在第 8 章 §5.3 的 WARNING 脚注处增加交叉引用。

**预估篇幅**：约 100-140 行 Markdown（含公式、表格、示例）。

---

### Gap-3：3D Parallel Step Time 组合模型缺失

**现状**

第 8 章 §4.7 和 §8.4 有一套完整的 DP step time 模型：

```text
step_time = max_rank(data_visible + compute + exposed_communication + optimizer + misc)
exposed_communication = max(comm - overlap, 0)
```

并在 §10（Worked Example）里用这套模型拆解了 64 GPU 的 step time 数字。

第 9 章完全没有等价的 3D parallel step time 分解模型。§6.5（观测指标）列出了要采集哪些指标，但没有说这些指标如何组合成一个整体的 step time 方程。

**问题**

在 3D parallel 下，`exposed_communication` 不再是一个标量，而是 TP collective、PP send/recv、DP sync、FSDP AllGather 四类通信的函数。没有组合模型，工程师在调试混合并行 step time 退化时：

- 无法判断是 TP 通信、PP bubble、DP sync 还是 FSDP AllGather 是主要开销；
- 无法估算引入 CP 后 step time 的变化；
- 无法解释为什么"理论 bubble 14%"但实测吞吐只有理想的 70%。

**需要补充的内容**

**3D parallel step time 分解模型**

```text
step_time_3d ≈ max_dp_replica(
    pipeline_step_time
)

pipeline_step_time ≈
    PP_warmup_latency                                   # (PP-1) 个 microbatch 填充 pipeline 的时间
  + m × max_stage(
        max_rank_in_stage(
            microbatch_compute_time
          + max(tp_collective_time - tp_overlap, 0)    # TP exposed
          + pp_send_recv_time                          # 通常与 compute 部分重叠
        )
    )
  + PP_drain_latency                                   # pipeline 排空时间
  + max(dp_sync_time - dp_overlap, 0)                  # DP exposed（梯度同步）
  + max(fsdp_allgather_time - fsdp_overlap, 0)        # FSDP exposed（参数聚合，如启用）
  + max(cp_exchange_time - cp_overlap, 0)              # CP exposed（KV exchange，如启用）
  + optimizer_time
  + checkpoint_overhead（不含在稳态 step time 内）
```

各项拆解：

| 组件 | 估算方式 | 与什么重叠 | 典型占比 |
|---|---|---|---|
| `microbatch_compute_time` | profiler CUDA timeline，forward+backward per stage | 无，是计算基线 | 50-70% |
| `tp_collective_time` | nccl-tests 节点内 AllReduce，按 Gap-1 公式估算 | 与下一层的 compute（CUDA stream 并发） | 5-15% |
| `pp_send_recv_time` | 按 Gap-1 公式，除以跨节点带宽 | 与下一个 microbatch 的某层 compute（1F1B 调度） | 3-8% |
| `dp_sync_time` | 按第 8 章 AllReduce 公式，DP group 梯度量 | 与 optimizer 或最后 microbatch compute | 5-20% |
| `fsdp_allgather_time` | 参数量 / DP shard 数 × 2 bytes / 带宽 × prefetch factor | 与上一个 FSDP unit backward | 5-15% |
| `cp_exchange_time` | 按 Gap-1 KV 公式 × (CP-1) ring steps | 与本地 attention compute（ring FA） | 5-30%（长上下文） |
| `PP_warmup + drain` | `(PP-1) × avg_stage_time` | 无法重叠（bubble 的来源） | 取决于 m 与 PP 比值 |

**实际意义的诊断路径**

```text
step_time 高，先看 profiler：

1. PP bubble 明显 → 增加 m 或开 interleaved pipeline
2. TP collective 暴露 → 检查 placement（节点内 NVSwitch？），调 CUDA_DEVICE_MAX_CONNECTIONS
3. DP sync 暴露 → 检查 bucket 和 overlap，与第 8 章对齐
4. FSDP AllGather 暴露 → 调 wrap policy、prefetch、limit_all_gathers
5. CP exchange 暴露 → 检查 ring FA 实现是否支持 overlap，检查带宽
6. compute 主导（所有通信都被掩盖）→ 系统工作正常，优化 kernel 或换更多 GPU
```

**建议插入位置**：第 9 章 §7（容量与效率）中，在 §7.2（PP bubble 与有效吞吐）之后新增 §7.3（3D parallel step time 分解模型），对齐第 8 章 §8.4 的风格。

**预估篇幅**：约 100-120 行 Markdown（含模型、表格、诊断路径）。

---

## 3. P1 缺口：现代架构适配 + 关键工程决策支撑

> P1 缺口在生产落地时频繁触发，但不影响章节基础框架的完整性。

---

### Gap-4：GQA / MQA 的 TP 可整除约束缺失

**现状**

第 9 章 §5.1（Megatron-style 配置示例）中有一行：

> "hidden-size、num-attention-heads、KV heads 必须能被 TP 整除"

这是正确但不充分的表述。章节没有展开 GQA（Grouped Query Attention）下"KV heads 被 TP 整除"与"Q heads 被 TP 整除"的不同约束，也没有给出主流模型的合法 TP 值。

**问题**

GQA 是当前所有主流开源 LLM（Llama-3、Mistral、Qwen2、Gemma2、Falcon）的标准配置。KV heads 通常远少于 Q heads（如 64Q/8KV，32Q/8KV）。当 TP size 大于 KV heads 数时，配置直接无效（每个 TP rank 分不到一个完整的 KV head）或需要特殊的 GQA-aware TP 实现（部分框架支持 KV head 复制，但会增加通信或内存）。

不知道这个约束的工程师会尝试 TP=16 配置一个 8 KV head 的模型，然后在框架层拿到非直觉的错误或 silent corruption。

**需要补充的内容**

**GQA 下的 TP 约束规则**

```text
标准 TP 约束（MHA/GQA 通用）：
  - Q heads 必须能被 TP 整除：num_q_heads % TP == 0
  - KV heads 必须能被 TP 整除：num_kv_heads % TP == 0
  - hidden_size 必须能被 TP 整除：hidden_size % TP == 0
  - intermediate_size（MLP）必须能被 TP 整除：ffn_size % TP == 0
  - vocab_size 建议能被 TP 整除（vocab parallel 下是硬约束）

GQA 附加约束：
  当 num_kv_heads < num_q_heads 时（GQA），num_kv_heads 的因子集合更小，
  直接限制了 TP 的可选值域。
```

**主流模型合法 TP 值**

| 模型 | Q heads | KV heads | 合法 TP 值（整除 KV heads） | 常见部署 TP |
|---|---|---|---|---|
| Llama-3 8B | 32 | 8 | 1, 2, 4, 8 | 4 或 8 |
| Llama-3 70B | 64 | 8 | 1, 2, 4, 8 | 8 |
| Llama-3 405B | 128 | 8 | 1, 2, 4, 8 | 8（节点内） |
| Mistral 7B | 32 | 8 | 1, 2, 4, 8 | 4 或 8 |
| Qwen2.5 72B | 64 | 8 | 1, 2, 4, 8 | 8 |
| Qwen2.5 7B | 28 | 4 | 1, 2, 4 | 4 |
| Gemma2 27B | 32 | 16 | 1, 2, 4, 8, 16 | 8 |
| 标准 MHA（如 GPT-3 175B） | 96 | 96 | 1,2,3,4,6,8,12,16,24,32,48,96 | 8 或 16 |

> **关键结论**：使用 GQA 的模型在 TP=16 时通常无效（除非 KV heads ≥ 16）。TP=8 是绝大多数 GQA 模型的实际上限，与 8-GPU NVSwitch 节点刚好对齐，这不是巧合，而是现代 LLM 架构设计时就考虑的约束。

**框架行为说明**

```text
Megatron-LM：强制检查 kv_heads % tp_size == 0，否则报错退出
DeepSpeed：部分版本不检查，可能产生 silent shape mismatch
PyTorch FSDP：不原生处理 attention head 约束，需用户自行保证
Transformer Engine：在 GQA GroupedQueryAttention module 中有 assertion
```

**建议插入位置**：第 9 章 §5.1（Megatron-style 配置）中"配置审查要点"之后，新增"GQA/MQA TP 约束专项"小节；并在 §8.3（框架支持）中的"主要约束"列补充 GQA 条目。

**预估篇幅**：约 60-80 行 Markdown（含规则、表格、框架说明）。

---

### Gap-5：rank mesh 推导过程缺失

**现状**

第 9 章 §9（70B Worked Example）直接给出配置 A（TP=8, PP=4, DP=4）和配置 B（TP=4, PP=8, DP=4），对每种配置说明优缺点，但省略了从约束出发推导这些数字的过程。

读者看完只能知道"对于 70B 应该选 TP=8, PP=4"，不知道如何将同样的推理过程应用到 180B 模型或不同的集群配置。

**问题**

rank mesh 的推导是 AI Infra Engineer 日常工作中最高频的任务之一，往往在选型会议中要实时完成。没有推导框架，工程师：

- 无法在讨论新模型时快速给出合理的起始配置；
- 可能会选出 bubble 极高（m < p）或显存不足的配置；
- 在面试或评审中无法系统展示决策逻辑。

**需要补充的内容**

**rank mesh 推导的标准五步流程**

```text
Step 1：计算 per-GPU 可用 HBM 预算
  available_hbm = total_hbm - fragmentation_margin - comm_buffer_overhead
  fragmentation_margin ≈ 8-15% of total_hbm（经验值）

Step 2：计算最小 TP（解决单层 GEMM 峰值）
  peak_per_layer ≈ hidden_size × max(4*hidden_size, intermediate_size) × dtype_bytes × factor
  若 peak_per_layer < available_hbm × 0.3（留给其他项），则 TP=1 可行；否则求最小 TP 使其可行

  GQA 约束：TP 必须整除 kv_heads（见 Gap-4）

Step 3：计算最小 PP（解决整网层数和状态容量）
  per_rank_param_bytes_no_pp = total_params × dtype_bytes / TP_chosen
  per_rank_optimizer_bytes ≈ total_params × 12 / (TP_chosen × ZeRO_degree)
  per_rank_activation ≈ 见 Gap-2 公式 / TP_chosen（TP 切 hidden）

  若 per_rank_total > available_hbm：
    min_pp = ceil(per_rank_total / available_hbm)

Step 4：计算 DP
  dp = world_size / (TP_chosen × PP_chosen)

Step 5：验证 bubble 和带宽
  bubble = (PP - 1) / microbatch_count   （1F1B 公式，需 m ≥ PP）
  microbatch_count = global_batch / (micro_batch_size × DP)
  若 bubble > 20%：增加 microbatch，或考虑 interleaved pipeline

  带宽验证：
  TP comm（节点内）：按 Gap-1 公式，与节点内 NVSwitch 带宽对比
  PP comm（跨节点）：按 Gap-1 公式，与 IB/RoCE 带宽对比
  DP sync：按第 8 章公式
```

**以 180B 模型为例的推导演示**

```text
输入：
  dense 180B，hidden=12288，layers=96，Q heads=96，KV heads=8，intermediate=4×hidden
  集群：32 nodes × 8 H100 80GB = 256 GPU，节点内 NVSwitch，跨节点 400G IB
  目标：seq=8192，BF16，AdamW

Step 1：available_hbm ≈ 80 GB × 0.88 ≈ 70 GB

Step 2：KV heads=8 → TP ∈ {1,2,4,8}，选 TP=8（单节点）
  per-layer peak（TP=8）：12288×49152×2 / 8 ≈ 1.5 GB → 在 70 GB 范围内

Step 3：per-rank 参数（TP=8，无 PP）：180B×2/8 = 45 GB
  optimizer state（ZeRO-1 or dist opt）：180B×12/256 ≈ 8.4 GB（全 256 rank 分）
  activation（selective+FlashAttention，seq=8192，micro_batch=1）：
    ≈ 96 × 8 × 12288 × 8192 × 2 bytes ≈ 126 GB（仍然太高）
  → 需要 PP 降低每 rank 层数

  若 PP=8：每 rank 12 层，activation ≈ 126/8 = 15.75 GB
  参数：45/1 = 45 GB（TP 切了，PP 不切参数量只切层数）= 45×12/96 = 5.6 GB/rank
  总估算：45/8×(12/96) + 8.4 + 15.75 + comm_buffer ≈ 5.6 + 8.4 + 15.75 + 5 = 34.75 GB ✓

Step 4：DP = 256 / (8 × 8) = 4

Step 5：microbatch=1，global_batch=2048，m = 2048/(1×4) = 512
  bubble（1F1B） = (8-1)/512 = 1.4% ✓ （m >> p，bubble 极低）

  TP 通信（节点内，NVSwitch ≈ 600 GB/s 有效 AllReduce）：
    per-call ≈ 12288×8192×2×(7/8) = 175 MB，约 0.29 ms per call，可被 compute 覆盖 ✓

结论：TP=8, PP=8, DP=4 是可行起点。下一步做 HBM dry-run 和 nccl-tests 验证。
```

**建议插入位置**：第 9 章 §8.1（决策树）之前，新增 §8.0（rank mesh 推导流程），或在 §9.2（70B 配置 A）之前增加"推导过程"小节。

**预估篇幅**：约 120-160 行 Markdown（含推导框架、公式、示例）。

---

### Gap-6：3D parallel 通信重叠架构缺失

**现状**

第 8 章 §4.5 详解了 DDP bucket overlap 的工作原理（backward 中 bucket 就绪即启动 AllReduce，后续 backward 掩盖通信）。这是第 8 章的亮点之一。

第 9 章在 §4.2 中仅写了：

> `TP_comm_exposed = max(TP_collective_time - overlap_with_compute, 0)`

但没有说明 `overlap_with_compute` 在 TP/PP/CP 下分别是什么、如何实现、有什么前提条件。

**问题**

通信重叠是影响实际 MFU 最重要的系统工程细节。工程师如果不知道每种重叠的实现条件，会：

- 以为"通信量小 = 问题不大"，忽视 exposure 问题；
- 错误调整 `CUDA_DEVICE_MAX_CONNECTIONS` 或 NCCL 参数，反而破坏原本工作的重叠；
- 在 profiler 中看到通信 kernel 和计算 kernel 交织时无法判断是正常还是异常。

**需要补充的内容**

**3D parallel 各类通信的重叠机制**

| 通信类型 | 能与什么重叠 | 实现条件 | 破坏条件 |
|---|---|---|---|
| TP AllReduce（列并行输出） | 下一层的 compute（Column Linear forward） | `CUDA_DEVICE_MAX_CONNECTIONS=1` + 独立 CUDA stream | 同 stream 执行、显式 synchronize、未开启 async collective |
| PP send/recv（forward activation） | 同 stage 下一个 microbatch 的某些 compute（1F1B） | 1F1B 调度 + `torch.distributed.isend/irecv` 异步接口 | microbatch 数量不足（m < p）、使用同步 send/recv |
| DP gradient AllReduce（或 FSDP ReduceScatter） | optimizer step 或下一个 step 的 forward（流水线 DP） | DDP bucket ready 触发异步 AllReduce，与 backward 继续执行重叠 | `no_sync()` 未使用（accumulation 中途同步）、bucket 太大导致 tail exposure |
| FSDP AllGather（参数预取） | 前一个 FSDP unit 的 backward（backward_prefetch） | `backward_prefetch=BACKWARD_PRE` + FSDP 内部 stream | `limit_all_gathers=True` 过于保守、wrap 粒度太粗导致 AllGather 覆盖太多 compute |
| CP KV ring exchange | 本地 context 块的 attention compute（Ring FlashAttention） | Ring FlashAttention 实现（如 Megatron-Core CP、FA ring_flash_attn） | 未使用 Ring FA（使用标准 FA 则无法重叠 KV exchange）、带宽严重不足时计算来不及覆盖 |

**重叠质量的 profiler 判断**

```text
健康的重叠（Nsight Systems 视图）：
  NCCL kernel 与 cuBLAS/cuDNN kernel 在时间轴上交织，GPU SM 无大段空闲。

TP AllReduce 暴露（不良）：
  每层 GEMM 后有明显 gap，随后才是 AllReduce，AllReduce 结束后才有下一层 GEMM。

PP send/recv 暴露（不良）：
  stage 完成 compute 后，空闲等待 recv，recv 完成后才开始下一层 compute。
  征兆：stage P95/P50 step time 比值高，且 microbatch 数 m 接近 PP size。

FSDP AllGather 暴露（不良）：
  backward 中频繁出现 AllGather + idle gap（等待参数聚合完成）。
  征兆：FSDP timeline 中 AllGather 和 backward 计算不交织，而是串行。
```

**建议插入位置**：第 9 章 §4.2（TP 通信边界）之后，新增 §4.2b（3D parallel 通信重叠机制），与第 8 章 §4.5 形成对称的跨章节参考结构。

**预估篇幅**：约 80-100 行 Markdown（含表格、profiler 判断说明）。

---

### Gap-7：Pipeline Stage 负载均衡操作指导缺失

**现状**

第 9 章在 §9.2（70B 配置 A 风险）中提到：

> "stage 0 embedding 和 stage 3 LM head 可能更重，需要按 profile 调整 layer split"

§4.3.3 中有一处 NOTE：

> "真实集群 stage 计算时间可能不均匀（某些 stage 算 attention，某些算 MLP），需要 profile 后用 layer placement 调平 stage time"

但没有说明如何量化不均衡、如何调整，以及不均衡的代价是多少。

**问题**

stage 不均衡是 PP 训练中最常见的隐性效率损失。在实践中：

- embedding + position encoding 的 stage 比纯 Transformer block stage 慢 10-30%；
- vocab projection（output embedding）在 vocab parallel 下也不同于普通 block；
- 仅 10% 的 stage time 不均衡就会让整个 pipeline 的有效吞吐降低约 10%（木桶原理）。

不懂如何操作的工程师往往用均匀层数切分，然后用 interleaved pipeline 或增加 microbatch 遮盖问题，而不是从根本上修复 stage 不均衡。

**需要补充的内容**

**Stage 不均衡量化**

```text
stage_imbalance_ratio = max(stage_time_p50) / mean(stage_time_p50) - 1

一般门限：
  imbalance_ratio < 5%：可接受
  imbalance_ratio 5-15%：需要关注，可用 layer redistribution 改善
  imbalance_ratio > 15%：强制修复，否则 interleaving 也无法弥补

观测方式：Nsight Systems 中按 stage（pp_stage 维度）聚合 compute kernel 时间；
         或在训练 metric 中采集 microbatch_compute_time{pp_stage=X}
```

**常见不均衡来源和修复方式**

| 不均衡来源 | 识别方式 | 修复方式 |
|---|---|---|
| Stage 0 含 embedding | Stage 0 compute 慢 5-20% | 将 embedding 单独作为 stage 0（0 层 Transformer block），纯 embedding pass |
| Stage N-1 含 LM head + loss | 最后 stage 显著更慢 | vocab parallel 切 LM head；或让 LM head 独占最后一个 stage |
| Transformer block 计算不均 | 按层统计 compute（罕见，常见于 MoE） | 按层计算量重新分 stage，Megatron `--num-layers-per-pipeline-rank` |
| 序列长度不均（data skew） | 与第 8 章 data skew 诊断对齐 | 按第 8 章 §10.6 处理，不是 stage 负载均衡问题 |

**Megatron 非均匀 layer 切分示例**

```bash
# Megatron 支持为首尾 stage 单独指定 layer 数
torchrun ... pretrain_gpt.py \
  --num-layers 80 \
  --pipeline-model-parallel-size 4 \
  # 可以通过 --num-layers-per-virtual-pipeline-stage 控制 virtual stage
  # 或通过自定义 partition 函数在 megatron/core/pipeline_parallel/schedules.py 中实现
```

> 注意：Megatron 默认按均匀层数切分，不均匀切分需要代码层面配置或 fork；DeepSpeed PipelineModule 的 `partition_method="parameters"` 可按参数量而非层数切分，对 embedding/head 不均有一定改善。

**建议插入位置**：第 9 章 §4.3.3（工程含义）后新增 §4.3.4（stage 负载均衡），或在 §11（故障排除）的表格中补充"stage 时间不均"这一行。

**预估篇幅**：约 60-80 行 Markdown（含量化方法、来源表、工具示例）。

---

## 4. P2 缺口：重要但可延迟的补充

> P2 缺口补充后会显著提升章节的前沿性和实用性，但不影响当前主线逻辑完整性。

---

### Gap-8：FP8 / Transformer Engine 与并行策略的交互

**现状**：无任何相关内容。

**需要补充的内容**

FP8 训练（H100 / H800 / Blackwell）通过 NVIDIA Transformer Engine 实现时，与并行策略有以下交互点：

1. **TP 分片后 FP8 scaling 一致性**：FP8 per-tensor scaling factor 在 TP 内所有 rank 必须相同（因为 TP 把一个逻辑 tensor 切成多份，scaling 必须对应同一逻辑范围）。Transformer Engine 通过 `FP8GlobalState` 管理跨 TP rank 的 amax（最大绝对值）同步，生产配置时需要确认 TP group 内 allreduce amax。
2. **FP8 checkpoint 额外状态**：每个 FP8 layer 有 amax history（默认 16 步滑动窗口）和 scale factor。FSDP/ZeRO checkpoint 需要包含这些 metadata，否则恢复后 FP8 scaling 从头冷启动，可能导致训练初期不稳定。
3. **PP stage 边界的 activation dtype**：FP8 训练中 PP send/recv 的 activation 可以是 BF16（更安全）或 FP8（节省带宽），Megatron 和 Transformer Engine 的 config 需要对齐。

**建议插入位置**：第 9 章 §5.1（Megatron 配置）中增加 FP8 相关参数说明；§8.3（框架支持）中增加"FP8/Transformer Engine"行；§13.3（框架检查项）中增加 FP8 专项。

**预估篇幅**：约 50-70 行。

---

### Gap-9：Zero Bubble Pipeline 工程实现细节

**现状**

第 9 章 §4.3.1 给出了 Zero Bubble 的 bubble fraction（"接近 0，理想"）和框架支持状态（"DeepSeek、Megatron-Core 实验路径"），但没有说明其工程代价和实现约束。

**需要补充的内容**

Zero Bubble（ZB1P）的核心思想是把 backward 拆成两个阶段：

```text
B-pass（backward input gradient）：计算 dL/dX，用于向前一个 stage 传播梯度，可以早于 W-pass 完成
W-pass（backward weight gradient）：计算 dL/dW，累积到 gradient buffer，无需立即完成
```

工程代价：

1. **deferred W-pass 的 activation 保留**：W-pass 推迟时，对应的 forward activation 必须保留到 W-pass 执行完毕。在极端配置下（大量推迟），activation 内存 peak 高于标准 1F1B。
2. **optimizer step 时序**：W-pass 必须在 optimizer step 前全部完成，调度器需要跟踪每个 microbatch 的 W-pass 状态。
3. **框架实现现状**：
   - Megatron-Core：`zero_bubble` 分支，需要显式配置 `--pipeline-schedule ZB1P`
   - DeepSeek-V3 训练：自研实现，验证了 1024 GPU 下 bubble 从 10% 降到约 1%
   - PyTorch：暂无原生支持，需要外部 pipeline schedule 库

**建议插入位置**：第 9 章 §4.4（virtual stage 和 zero bubble）扩写，增加工程细节和框架状态说明。

**预估篇幅**：约 60-80 行。

---

### Gap-10：推理侧 Checkpoint 转换机制

**现状**

第 9 章 §8.4 提到：

> "训练 TP=8，推理 TP=4：需要合并再重切 tensor shard"
> "训练 PP=8，推理通常不使用训练 PP：需要按 layer id 重组完整模型"

但没有说明具体机制、工具和常见踩坑。

**需要补充的内容**

**TP 转换：Column parallel 和 Row parallel 的拼接方向不同**

```text
Column parallel（Q/K/V projection、FC1 output dim 切分）：
  merge: torch.cat(shards, dim=0)  → 按 output dim 拼接
  reshard: torch.chunk(merged, new_tp, dim=0)

Row parallel（attention output proj、FC2 input dim 切分）：
  merge: torch.cat(shards, dim=1)  → 按 input dim 拼接
  reshard: torch.chunk(merged, new_tp, dim=1)
```

**PP flatten：必须知道 layer-to-stage 映射**

```text
训练 checkpoint 结构（PP=4，每 stage 20 层）：
  stage0/model_optim_rng.pt  → layers 0-19
  stage1/model_optim_rng.pt  → layers 20-39
  stage2/model_optim_rng.pt  → layers 40-59
  stage3/model_optim_rng.pt  → layers 60-79

恢复时需要按 layer id 顺序重组：
  full_model_layers = stage0_layers + stage1_layers + stage2_layers + stage3_layers
```

**Tied embedding 的跨 stage 问题**

```text
Tied embedding（embedding weight = LM head weight）在 PP 中：
  - embedding 在 stage 0
  - LM head 在 stage N-1
  - 二者共享同一个参数，但训练中分属两个 stage

merge 时必须只取其中一份（通常取 stage N-1 的 LM head），
否则 embedding 和 LM head 的 gradient 更新历史可能有差异（视实现而定）。
```

**工具链**

| 工具 | 用途 | 限制 |
|---|---|---|
| Megatron `tools/checkpoint/convert_checkpoint.py` | TP/PP reshape，支持转 HuggingFace | 只支持 Megatron 格式，tied embedding 需要注意 |
| DeepSpeed `zero_to_fp32.py` | ZeRO-3 checkpoint 聚合成完整 FP32 | 不处理 TP/PP，需要先聚合再转换 |
| HuggingFace `from_pretrained` + `save_pretrained` | HuggingFace 格式互转 | 需要对应 modeling 代码支持 |
| vLLM `convert_megatron.py` 等社区脚本 | Megatron → vLLM 可读格式 | 社区维护，稳定性不一 |

**建议插入位置**：第 9 章 §8.4 之后，新增 §8.5（推理 checkpoint 转换）作为独立小节，或在现有 §8.4 的"推理转换要提前设计"段落后展开。

**预估篇幅**：约 80-100 行。

---

## 5. 汇总：缺口优先级与补充规模

| Gap | 优先级 | 补充位置 | 预估行数 | 核心价值 |
|---|---|---|---|---|
| Gap-1：TP/PP/CP 通信量公式 | **P0** | Ch9 §4.2a/4.3a/4.6a 或新 §4.8 | 120-160 | 带宽 admission 的数学基础 |
| Gap-2：Activation 内存定量模型 | **P0** | Ch9 §4.1a，Ch8 §5.3 脚注 | 100-140 | HBM 账本的缺失项 |
| Gap-3：3D parallel step time 模型 | **P0** | Ch9 §7.3 | 100-120 | 混合并行调试的分析框架 |
| Gap-4：GQA/MQA TP 约束 | P1 | Ch9 §5.1，§8.3 | 60-80 | 现代模型配置正确性 |
| Gap-5：rank mesh 推导过程 | P1 | Ch9 §8.0 或 §9 前置 | 120-160 | worked example 的可迁移性 |
| Gap-6：通信重叠架构 | P1 | Ch9 §4.2b | 80-100 | overlap 工程实现理解 |
| Gap-7：Stage 负载均衡 | P1 | Ch9 §4.3.4，§11 | 60-80 | PP 效率的隐性损失来源 |
| Gap-8：FP8 / Transformer Engine 交互 | P2 | Ch9 §5.1，§8.3，§13.3 | 50-70 | H100 生产训练前沿 |
| Gap-9：Zero Bubble 工程细节 | P2 | Ch9 §4.4 扩写 | 60-80 | ZB pipeline 生产可用性 |
| Gap-10：推理 checkpoint 转换 | P2 | Ch9 §8.5 | 80-100 | 训练到推理的最后一公里 |

**P0 合计新增**：约 320-420 行（对应 Ch9 当前 1239 行，增量约 26-34%）
**P0+P1 合计**：约 640-860 行
**P0+P1+P2 全部**：约 830-1110 行

---

## 6. 实施建议

### 6.1 执行顺序

推荐按以下顺序执行，每批次独立评审后再继续：

**批次 1（P0，独立可合并）**
- Gap-2（activation 公式）先于 Gap-1 完成，因为 Gap-1 的 PP 通信量公式引用 activation 维度
- Gap-3（step time 模型）依赖 Gap-1 公式完成后在汇总表中交叉引用

**批次 2（P1 主线）**
- Gap-5（rank mesh 推导）完成后，在 worked example 中引用 Gap-1/2/3 的公式
- Gap-6（重叠架构）与 Gap-1 通信量形成互补（量与质）

**批次 3（P1 配置 + P2）**
- Gap-4、Gap-7 独立，可并行
- Gap-8、Gap-9、Gap-10 各自独立

### 6.2 写作规范

每个 Gap 新增内容应遵循现有章节的写作风格：

- 使用第一性原理句式引出公式，而不是直接给表格
- 数字示例使用与 worked example 一致的模型参数（70B：hidden=8192，layers=80，seq=8192）
- Warning/Note/Danger callout 格式与现有使用方式一致
- 新增的故障诊断内容应对应更新 §11 故障排除表，而不是只在新节里孤立出现

### 6.3 执行 agent 指引

执行时建议每个 Gap 由独立 subagent 处理，产出 Markdown 补丁段落（而非直接改原文件），经主 agent 评审插入位置后再合并。理由：

- Gap-1/2/3 的公式数字需要交叉验证（同一模型不同公式的估算结果应内部一致）
- Gap-5 的 180B 推导示例需要引用 Gap-1/2 的公式，必须后于 P0 批次完成

---

## 7. 与现有计划的关系

- Wave 2 计划（`2026-05-05-ai-infra-tutorial-wave-2-systems-evidence-and-capacity-depth.md`）范围为 Part 0/2，明确排除 Part 3，本规格不与 Wave 2 冲突。
- Part 3 重写设计规格（`2026-05-04-part3-training-infra-rewrite-design.md`）已完成初次重写，本规格是 post-rewrite 的 gap 分析，属于 Wave 3 或单独 Part 3 深化轮次的输入。
- 本规格不要求新建章节文件，所有内容在现有 `08-data-parallel.md` 和 `09-model-pipeline-parallel.md` 内嵌入，保持"不拆新章"的历史设计决策。
