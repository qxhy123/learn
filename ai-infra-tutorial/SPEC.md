# AI Infra 教程改进规格说明书 v3 — 第一性原理评审

> 日期：2026-04-24
>
> 本文档基于对全部 25 章 + 4 附录的逐章深度评审，从第一性原理出发识别教程的系统性缺陷。
> 每个改进项设计为独立工作单元（Work Unit），可由独立 agent 并行执行，无跨单元依赖。

---

## 一、第一性原理评审

### 教程的根本目的

一个优秀的 AI Infra 教程应当让读者获得三种不可替代的能力：

1. **诊断能力**：面对系统异常，能快速定位到正确的层级和资源类型
2. **决策能力**：面对多个技术选项，能基于约束条件做出工程判断
3. **设计能力**：面对业务需求，能从资源、链路、平台、治理四个维度组织方案

### 现状评估

| 能力维度 | 核心章节（1-9, 14-23） | 新增章节（10b, 10c, 25） |
|----------|----------------------|------------------------|
| 诊断能力 | 强 — "问题先行"模式一致 | 弱 — 缺少故障场景和排查路径 |
| 决策能力 | 中 — 有概念对比，缺决策树 | 弱 — 只列选项不给判断框架 |
| 设计能力 | 中 — 有架构图，缺实操推演 | 弱 — 内容过薄无法支撑设计 |

### 五个系统性缺陷

从第一性原理出发，当前教程存在五个根本性问题：

**缺陷 1：新增章节深度严重不足**

Ch 10b/10c/25 是 v2 spec 新增的章节，但实现质量远低于核心章节。具体表现：
- 核心章节平均 12-16 道练习题，新章节仅 5-8 道
- 核心章节每节 500-1000 字，新章节部分小节仅 2-3 句话
- 新章节的"问题先行"模式执行不一致，部分段落直接罗列概念

**缺陷 2：缺乏决策工程（Decision Engineering）**

教程擅长解释"X 是什么"和"X 为什么存在"，但普遍缺少"在约束条件 C 下，应该选 X 还是 Y"的决策框架。典型缺失：
- Ch 9：解释了 TP/PP/DP/ZeRO，但没有给出 "给定模型大小和 GPU 数量，如何选择并行策略" 的决策树
- Ch 16：列出了量化方案和推理引擎，但没有 "给定精度要求和硬件，如何选型" 的流程
- Ch 13：介绍了向量数据库，但没有 "给定数据规模和查询模式，如何选型" 的判断框架

**缺陷 3：定量直觉训练不足**

教程给出了很多公式（如 AllReduce 通信量、显存预算），但很少用真实数字做完整推演。读者记住了"12 bytes/param for Adam"，却无法独立完成一次真实训练的资源规划。缺少的是：
- 从模型参数量出发，推算显存 → 卡数 → 通信量 → step time 的完整链条
- 从 QPS 和 SLA 出发，推算副本数 → GPU 成本 → KV Cache 显存的完整链条
- 真实数字的 worked example，而非仅有公式

**缺陷 4：章节间集成叙事薄弱**

教程自称"总装图"，但章节间的因果传导不够显式。例如：
- Checkpoint 策略（Ch 10）如何受并行策略（Ch 9）影响？
- 量化选择（Ch 16）如何影响多租户成本（Ch 17）？
- 数据管道设计（Ch 11）如何约束单机训练效率（Ch 7）？

读者看完每章都理解了，但串不成端到端的系统思维。

**缺陷 5：操作性现实感不足**

教程在概念层很强，但在"生产中真正会遇到什么"上偏弱。具体缺失：
- Ch 10 讲了 NCCL hang 的存在，但排查手段不成体系
- Ch 22 讲了灰度发布，但没有灰度期间的质量采样策略
- 没有一个章节系统地讲"AI 系统的 on-call 看什么指标、怎么分层响应"

---

## 二、改进原则

所有改动必须遵守：

1. **先讲问题再讲方案** — 每个新增段落以"为什么需要"开头
2. **保持平台工程师视角** — 不展开算法推导，聚焦工程判断和权衡
3. **表格优于长文** — 能用表格对比的不写散文
4. **给出工程边界** — 说明"什么时候有效、什么时候会失败"
5. **中文为主、术语保留英文原文** — 如"模型并行（Model Parallelism）"
6. **每节 200-600 字** — 与现有核心章节篇幅保持一致
7. **练习题与核心章节对齐** — 每章至少 12 道练习题（基础 6 + 进阶 4 + 设计 2）

---

## 三、独立工作单元（Work Units）

> **并行执行说明**：以下每个 WU 是完全独立的工作单元，可以分配给不同 agent 同时执行。
> 每个 WU 只修改指定的目标文件，不依赖其他 WU 的输出。
> Agent 执行时应先完整阅读目标文件，再进行修改。

---

### WU-01：Ch 10b 深度补完 — 对齐训练与后训练基础设施

**目标文件**：`part3-training-infra/10b-alignment-and-post-training.md`

**问题**：本章内容深度仅为核心章节的 40-50%。多个小节只有 2-3 句话，缺少 worked example，练习题仅 5 道（核心章节 12-16 道）。PPO 的四模型协调是本章核心卖点，但资源推演不够具体。

**具体改动**：

1. **§10b.2 PPO 基础设施形态 — 补充完整 worked example**
   - 以 LLaMA-7B 为 policy model，推演 PPO 训练在 8×H100 上的完整显存预算：policy 参数 + 梯度 + optimizer（~84GB）、reference model 推理权重（~14GB BF16）、reward model 推理权重（~14GB）、critic model 参数 + 梯度 + optimizer（~84GB）、rollout 阶段的 KV Cache
   - 给出总显存需求估算表，对比同模型 SFT 训练的显存需求
   - 说明为什么 PPO 峰值显存远大于稳态（rollout 和 training 阶段交替，显存复用策略）

2. **§10b.3 DPO/GRPO — 扩写至与 PPO 段落平衡**
   - 当前 DPO 段落过短。补充：DPO 的数据要求（偏好对数据格式、数据量级）、训练形态（和 SFT 几乎相同，但 loss 函数不同带来的数值稳定性差异）、GRPO 去掉 critic 后的显存简化
   - 增加决策框架表：什么情况选 PPO vs DPO vs GRPO

3. **§10b.4 Reward Model 推理子系统 — 补充部署架构选型**
   - 当前只说"可以独立部署也可以共享节点"，缺少判断依据
   - 补充：独立部署 vs 共享部署的决策条件（模型大小、吞吐要求、显存余量）
   - 补充：RM 的批处理策略（rollout batch 全量打分 vs 流式打分）

4. **§10b.5 实验管理 — 补充 checkpoint 和评估的多模型状态管理**
   - 当前只提到"需要保存 policy + critic + optimizer"，但没有说如何组织
   - 补充：checkpoint 应包含哪些状态、如何确保 policy 和 critic 版本一致性
   - 补充：评估 loop 中如何用 LLM-as-judge 做自动化评测

5. **练习题扩充至 12 道**
   - 增加 7 道练习题：PPO 显存预算计算、DPO vs PPO 选型判断、RM 部署方案设计、多模型 checkpoint 恢复策略、KL 系数调优的系统影响、rollout 长度对吞吐的影响、评测 pipeline 设计

**验收标准**：
- 所有小节不低于 400 字
- 至少包含 2 张新增表格（显存预算表、PPO vs DPO vs GRPO 对比表）
- 练习题总数 >= 12
- 保持"问题先行"模式

---

### WU-02：Ch 10c 深度补完 — Fine-Tuning 基础设施与多 Adapter 服务

**目标文件**：`part3-training-infra/10c-finetuning-and-multi-adapter.md`

**问题**：§10c.4.4（Multi-LoRA 显存争用）仅 2 段话，adapter 版本管理和 base model 升级兼容性完全缺失。全章读起来像两篇独立文章（Fine-tuning 和 Multi-Adapter）拼在一起，缺少集成叙事。

**具体改动**：

1. **§10c.4.4 Multi-LoRA 显存管理 — 从 2 段扩写至完整小节**
   - 分析：base model 权重共享 + N 个 adapter 的增量参数 + 请求级 KV Cache 的显存构成
   - 给出公式：总显存 ≈ base_model_size + N × adapter_size + concurrent_requests × kv_cache_per_request
   - 讨论：adapter 数量上限由什么决定（显存 vs 切换延迟 vs 路由复杂度）
   - 热加载/热卸载机制：LRU 驱逐策略、预加载策略

2. **新增 §10c.5a Adapter 与 Base Model 版本兼容性**
   - 这是生产中最痛的问题之一：base model 升级后所有 adapter 是否需要重新训练？
   - 答案是"通常需要"，因为 LoRA 矩阵与 base model 的权重空间绑定
   - 补充：版本绑定策略（adapter metadata 必须记录 base model hash）
   - 补充：灰度升级策略（新旧 base model 并存期间的流量分割）

3. **§10c.3 FTaaS 控制面 — 补充端到端 pipeline**
   - 当前在"发布策略"处戛然而止
   - 补充：从训练完成 → 自动评测 → 注册到 adapter registry → 触发 Multi-LoRA 服务热加载的完整流程
   - 补充：失败处理（训练失败重试、评测不通过回退、加载失败告警）

4. **集成叙事：在章首增加桥接段落**
   - 说明 Fine-tuning 和 Multi-Adapter Serving 不是两个独立话题，而是一条完整的链路：训练 adapter → 管理 adapter → 服务 adapter
   - 这条链路的每个环节都有平台化的需求

5. **练习题扩充至 13 道**
   - 增加 5 道：Multi-LoRA 显存预算计算、adapter 版本升级策略设计、FTaaS 任务状态机设计、adapter A/B 测试方案、adapter 清理策略（何时删除不再使用的 adapter）

**验收标准**：
- §10c.4.4 扩写至 500+ 字
- 新增版本兼容性小节 400+ 字
- 练习题总数 >= 13
- 全章读起来是一个连贯故事，不是两篇拼接

---

### WU-03：Ch 25 深度补完 — AI Agent 与推理时计算基础设施

**目标文件**：`part8-advanced-and-capstone/25-agent-and-inference-time-compute.md`

**问题**：本章是 v2 新增章节，但内容与 Ch 1/3 有重复（Agent 定义），"thinking tokens"部分只以 OpenAI o1 为例未泛化，推理预算管理缺乏工程深度，与前序章节（14-17 推理基础设施）的集成不足。练习题仅 5 道。

**具体改动**：

1. **§25.1 去重 + 聚焦**
   - 删除与 Ch 1/3 重复的 Agent 概念介绍
   - 开头直接从基础设施视角切入："如果你已经理解了 Ch 14-17 的推理系统设计，那么 Agent 和 Thinking Model 会从三个方向打破你的假设：会话持续时间、token 消耗可预测性、并发模型"

2. **§25.3 推理时计算 — 泛化 thinking tokens**
   - 不只讲 o1，泛化为一类模式：任何在推理阶段消耗额外计算的技术
   - 包括：Chain-of-Thought（显式思维链）、Best-of-N sampling、Tree search / beam search、Verifier-guided generation
   - 每种模式的资源特征表：额外 token 量级、GPU 时间波动范围、对调度器的影响

3. **§25.4 成本模型 — 补充推理预算管理的工程实现**
   - 当前只是概念性讨论（"thinking tokens 算谁的？"）
   - 补充工程实现：max_thinking_tokens 参数、per-request GPU-second budget、budget exhaustion 时的降级策略（截断 thinking、切换到更小模型、返回 partial result）
   - 补充：计费模型选择（按 input+output token、按 GPU-second、按 session）

4. **新增 §25.x Agent 基础设施与推理服务的集成**
   - 这是本章最大的缺口：Agent 不是独立系统，它跑在推理基础设施之上
   - 补充：Agent session 如何映射到 vLLM/TRT-LLM 的请求
   - 补充：Tool calling 的执行环境（沙箱、超时、权限控制、结果回注）
   - 补充：与 Ch 15 KV Cache 的关系（Agent 的长 context 需要 prefix caching、KV Cache 生命周期管理）

5. **练习题扩充至 12 道**
   - 增加 7 道练习题

**验收标准**：
- 与 Ch 1/3 的内容重复消除
- thinking tokens 泛化为 3+ 种模式
- 推理预算管理有具体工程实现方案
- 练习题总数 >= 12

---

### WU-04：Ch 9 决策工程 — 并行策略选型决策树

**目标文件**：`part3-training-infra/09-model-pipeline-parallel.md`

**问题**：本章解释了 TP/PP/DP/ZeRO/SP/CP 各自的原理，但读者读完后仍然不知道"给定我的模型和集群，应该怎么配"。这是典型的"决策工程"缺失。

**具体改动**：

1. **新增 §9.x 并行策略选型决策树**
   - 以模型参数量和可用 GPU 数量为输入，给出分层决策
   - 用 ASCII 决策树呈现，配合文字解释每个分支的判断依据

2. **新增 §9.x 典型配置实例表**（至少 4 行真实规模配置）

3. **补充 Interleaved Pipeline 和 Zero Bubble 解释**（各至少 200 字）

**验收标准**：
- 决策树可独立理解
- 实例表至少 4 行
- Interleaved Pipeline 和 Zero Bubble 各至少 200 字

---

### WU-05：Ch 16 决策工程 — 量化选型与推理引擎选型

**目标文件**：`part5-serving-infra/16-quantization-compilation-and-engines.md`

**问题**：列出了方案但缺少"在给定约束下如何选择"的决策框架。

**具体改动**：

1. **新增 §16.x 量化方案选型决策流程**（含决策树）
2. **新增 §16.x 推理引擎选型决策流程**（含决策树）
3. **新增 §16.x 校准（Calibration）过程详解**（300+ 字）

**验收标准**：
- 量化和推理引擎各有清晰决策树
- 校准过程 300+ 字

---

### WU-06：Ch 7 定量直觉 — 训练资源规划 Worked Example

**目标文件**：`part3-training-infra/07-single-node-training.md`

**问题**：有公式但没有用真实模型做完整推演。MFU 完全缺失。

**具体改动**：

1. **新增 §7.x Worked Example：LLaMA-7B 单机训练资源规划**（完整推演每步计算）
2. **新增 §7.x MFU（Model FLOPs Utilization）**（定义、公式、参考值、vs GPU Utilization）
3. **新增 §7.x Mixed Precision（AMP）**（300+ 字）

**验收标准**：
- Worked example 每步计算清晰可复现
- MFU 有公式和参考值表

---

### WU-07：Ch 15 定量直觉 — 推理容量规划 Worked Example

**目标文件**：`part5-serving-infra/15-batching-scheduling-and-kv-cache.md`

**问题**：缺少完整推理容量规划推演。Prefill-Decode 分离架构缺失。

**具体改动**：

1. **新增 §15.x Worked Example：LLaMA-70B 推理容量规划**
2. **新增 §15.x Prefill-Decode 分离架构**
3. **新增 §15.x Speculative Decoding**（200+ 字）

**验收标准**：
- Worked example 每步计算清晰
- Prefill-Decode 分离有对比表

---

### WU-08：Ch 13 决策工程 — 向量系统选型与 RAG 工程

**目标文件**：`part4-data-and-storage/13-feature-vector-and-cache.md`

**问题**：全教程最薄弱章节。向量数据库、ANN 算法、Chunking 策略均缺失。

**具体改动**：

1. **新增 §13.x 向量数据库选型决策框架**（6+ 行选型表）
2. **新增 §13.x ANN 搜索算法与距离度量**
3. **新增 §13.x RAG Chunking 策略**
4. **新增 §13.x 增量更新 vs 全量重建**
5. **新增 §13.x Prefix Caching 在 RAG 中的应用**

**验收标准**：
- 向量数据库选型表至少 6 行
- Chunking 有 3 种模式对比

---

### WU-09：Ch 9 补充 — Sequence Parallelism 与 Context Parallelism

**目标文件**：`part3-training-infra/09-model-pipeline-parallel.md`

**问题**：缺少 SP 和 CP，128K+ 训练无解。

**具体改动**：

1. **新增 §9.x Sequence Parallelism**（300+ 字）
2. **新增 §9.x Context Parallelism**（300+ 字）
3. **新增 TP/SP/CP 对比表**

**验收标准**：
- SP 和 CP 各 300+ 字
- 对比表清晰区分三者

---

### WU-10：Ch 10 补充 — 大规模训练故障处理与弹性训练

**目标文件**：`part3-training-infra/10-memory-checkpointing-and-recovery.md`

**问题**：NCCL hang、慢节点、弹性训练要么一笔带过要么缺失。

**具体改动**：

1. **扩写 NCCL Hang 排障**（排查流程 + 常见根因表）
2. **新增 §10.x Straggler Detection**
3. **新增 §10.x Elastic Training**（300+ 字）
4. **新增 §10.x Pre-flight Validation**

**验收标准**：
- NCCL Hang 有完整排查流程
- Elastic Training 300+ 字
- Pre-flight 有具体检查项列表

---

### WU-11：Ch 5 补充 — 集群网络拓扑

**目标文件**：`part2-systems-stack/05-memory-interconnect-io.md`

**问题**：完全没有讲网络拓扑设计。

**具体改动**：

1. **新增 §5.x 集群网络拓扑**（Fat-tree / Rail-optimized / DragonFly+）
2. **新增拓扑对比表**（3+ 行）
3. **Job Placement 策略**

**验收标准**：
- 拓扑对比表至少 3 行
- 说明拓扑如何影响训练效率

---

### WU-12：Ch 17 补充 — 真实成本工程

**目标文件**：`part5-serving-infra/17-multitenancy-and-cost.md`

**问题**：有成本归因概念但缺真正的成本工程。

**具体改动**：

1. **新增 §17.x Cloud vs On-Prem TCO 分析框架**
2. **新增 §17.x Spot / Preemptible 实例策略**
3. **新增 §17.x GPU 利用率的真实含义**（MFU vs Utilization）
4. **新增 §17.x Chargeback 实践**

**验收标准**：
- TCO 有对比表
- Spot 策略区分训练/推理场景

---

### WU-13：Ch 20 补充 — GPU 虚拟化与资源碎片化

**目标文件**：`part6-platform-and-orchestration/20-queues-quotas-and-autoscaling.md`

**问题**：GPU 被视为不可分割资源，碎片化未涉及。

**具体改动**：

1. **新增 §20.x GPU 虚拟化：MIG / MPS / Time-Slicing**（含对比表）
2. **新增 §20.x GPU 资源碎片化**
3. **新增 §20.x 公平调度算法简述**（DRF）

**验收标准**：
- MIG/MPS/Time-Slicing 对比表完整

---

### WU-14：Ch 22 补充 — 灰度发布质量采样与 A/B 测试

**目标文件**：`part7-reliability-security/22-evaluation-release-and-incident.md`

**问题**：灰度期间质量采样和 A/B 测试区分缺失。

**具体改动**：

1. **新增 §22.x A/B 测试 vs 灰度发布**
2. **新增 §22.x 灰度期间的质量采样策略**
3. **新增 §22.x Prompt/配置变更管理**

**验收标准**：
- A/B vs 灰度区别清晰
- 质量采样有具体操作策略

---

### WU-15：Ch 23 补充 — Secrets 管理与供应链安全

**目标文件**：`part7-reliability-security/23-security-isolation-and-governance.md`

**问题**：Secrets 管理和供应链安全完全缺失。

**具体改动**：

1. **新增 §23.x Secrets 管理**
2. **新增 §23.x 模型安全威胁**（pickle 攻击、SafeTensors）
3. **新增 §23.x 供应链安全**（cosign、Trivy、SLSA）

**验收标准**：
- Secrets 有具体注入方式和常见事故
- pickle 攻击和 SafeTensors 解释清晰

---

### WU-16：附录全面更新

**目标文件**：`appendix/glossary.md`、`appendix/tooling-map.md`、`appendix/checklists.md`、`appendix/answers.md`

**问题**：附录未跟上正文新增内容。

**具体改动**：

1. **Glossary 新增约 20 个术语**
2. **Tooling Map 新增 3+ 个类别**
3. **Checklists 新增 2+ 个清单**
4. **Answers 补充所有新增练习题解答**

**验收标准**：
- 术语表新增 >= 18 个
- 新增 >= 2 个检查清单
- 新增练习题答案覆盖率 100%

**注意**：本 WU 应在其他 WU 全部完成后执行。

---

### WU-17：README 导航与学习路径更新

**目标文件**：`README.md`

**问题**：学习路径未包含新增内容。

**具体改动**：

1. **更新学习路径建议**（四条路径都更新）
2. **增加"决策工程"作为组织方式**
3. **增加"能估算"作为学习目标**

**验收标准**：
- 四条学习路径都更新
- 所有章节链接正确

**注意**：本 WU 应在其他 WU 完成后执行。

---

## 四、执行顺序与并行策略

### 完全独立（可同时执行）

```
并行批次 1（全部可同时启动）：
├── WU-01  Ch 10b 深度补完
├── WU-02  Ch 10c 深度补完
├── WU-03  Ch 25 深度补完
├── WU-04  Ch 9 决策工程
├── WU-05  Ch 16 决策工程
├── WU-06  Ch 7 定量直觉
├── WU-07  Ch 15 定量直觉
├── WU-08  Ch 13 决策工程
├── WU-09  Ch 9 补充（SP/CP）
├── WU-10  Ch 10 故障处理
├── WU-11  Ch 5 网络拓扑
├── WU-12  Ch 17 成本工程
├── WU-13  Ch 20 GPU 虚拟化
├── WU-14  Ch 22 灰度质量
└── WU-15  Ch 23 安全
```

### 需等待前序完成

```
并行批次 2（前序全部完成后）：
├── WU-16  附录更新（依赖 WU-01~15 的新增练习题和术语）
└── WU-17  README 更新（依赖所有 WU 的内容变更）
```

---

## 五、不做的事（Scope Exclusion）

| 排除项 | 原因 |
|--------|------|
| 重写现有核心章节结构 | 核心章节（1-9, 14-23）质量已经很好 |
| 添加代码级实战项目 | 教程定位为认知框架，不是 hands-on lab |
| 添加英文版本 | 超出当前范围 |
| 逐工具的配置教程 | 保持系统思维定位 |
| 添加图片/图表资源 | 当前以 ASCII 图和表格为主 |

---

## 六、工作量估算

| 批次 | WU 数量 | 新增/改动文字量 | 新增表格数 |
|------|---------|----------------|-----------|
| 批次 1（并行） | 15 | ~12,000-16,000 字 | ~25-30 张 |
| 批次 2（收尾） | 2 | ~2,000-3,000 字 | ~3-5 张 |
| **总计** | **17** | **~14,000-19,000 字** | **~28-35 张** |
