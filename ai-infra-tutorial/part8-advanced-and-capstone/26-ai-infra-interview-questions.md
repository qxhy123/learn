# 第26章：AI Infra 面试题、自测与面试官题库

> AI Infra 面试不主要考"你能不能背组件名"，而考"你能不能在真实约束下，把资源、链路、故障和治理串起来推理"。

> **关联章节**：本章是全书前 25 章的综合自测与面试评估手册。它不替代各章末尾的练习，而是从面试官视角对全书内容做交叉验证。系统设计案例需要回到 [第14章 在线推理](../part5-serving-infra/14-online-inference-architecture.md)、[第15章 KV Cache](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md)、[第18章 容器与运行时](../part6-platform-and-orchestration/18-containers-and-runtime.md)、[第24章 平台构建](24-build-an-ai-platform.md)、[第25章 Agent 推理时计算](25-agent-and-inference-time-compute.md) 调取证据。

## 1. 第一性原理拆解 + 学习大纲

### 拆 — 不可化简的问题

把"算法 / 训练 / 推理 / Kubernetes / vLLM / NCCL"这些名字都拿掉，AI Infra 面试想筛选的不可化简能力只有一个：**面对一个有限资源、动态请求、可能失败、需要治理的真实工作负载，候选人能不能用系统语言把"资源 → 链路 → 故障 → 治理"四件事串起来推理。** 能流利说出 `paged_attention`、`zero3`、`pdb`、`tcp keepalive` 是必要不充分；能在 30 分钟内把"为什么夜里 3 点 p99 翻倍"、"为什么换卡之后 throughput 反而降"、"为什么 70B 训练 3 个小时挂在 step 412"诊断到根因，才是真正的信号。

### 推 — 这个能力为什么必须用面试题来筛

工作中真正考察 AI Infra 工程师的场景，是**线上故障复盘 / 平台架构评审 / 资源容量规划 / 跨团队 trade-off 谈判**。在面试 60 分钟里复刻这些场景，唯一可行的形式是结构化提问：先抛一个具体约束（"32 张 H100、3 周交付、目标 8K context 70B 推理 SLA p99 < 500 ms"），让候选人主动收敛资源、链路、故障、治理；再用追问压测他在哪里能讲清楚、哪里会发慌。这就是为什么本章每道题都强制带 `考察点 / 回答框架 / 追问 / 评分要点` 四件套——它们对应的是面试官实际打分时的判断点。

### 绘 — 本章地图

```
26.1 基础认知 + 系统分层(10)         ← 最先考察的"能不能从系统出发"
26.2 硬件 / GPU / 内存 / 网络 / 存储(12)  ← 资源层根因能力
26.3 分布式训练(12)                    ← 训练岗核心
26.4 数据 / 制品 / Checkpoint / Registry(10)  ← 训练-推理的中间链路
26.5 推理服务 / KV Cache / Batching / 引擎(14)  ← 推理岗核心 + 行业最热
26.6 K8s / 调度 / 队列 / 配额 / 平台化(12)  ← 平台岗核心
26.7 可观测 / 发布 / 安全 / 成本 / 多租户(12)  ← 治理与运营
26.8 综合系统设计 + 故障 case(8)         ← Tech lead / 高 P 必考
模拟面试套餐 ×5                        ← 端到端串训
```

## 2. 学习目标

读完并完成本章，应该能做到：

1. 用一致的"资源 → 链路 → 故障 → 治理"框架回答任何 AI Infra 面试题，而不是逐题靠记忆。
2. 区分清楚训练岗 / 推理岗 / 平台岗 / 可靠性岗的考察侧重，并能把同一个故障从不同岗位视角讲出来。
3. 作为候选人，能在 5 分钟内组织出一道高频题的完整回答骨架（结论 → 分类 → 取舍 → 案例）。
4. 作为自测者，能用评分要点反向定位自己最薄弱的章节，回到正文补课。
5. 作为面试官，能用追问从"会背"压到"会用"，并对照评分要点给出可解释的 hire / no-hire 决策。

## 3. 使用方式

本章服务三类读者，使用方式不同：

- **候选人模式**：从你目标岗位对应的小节开始（推理岗优先 26.5/26.6/26.8，训练岗 26.3/26.4/26.8，平台岗 26.6/26.7/26.8）。每题先合上书自己讲一遍，再对照"回答框架"补缺漏，最后用"追问"自我加压。一周写 10-15 题足够。
- **自测模式**：从 26.1 顺序刷到 26.8，每题给自己打"及格 / 良好 / 优秀 / 不会"四档。统计哪一节"不会"最多——那就是你回到正文要重读的部分。Mock pack 留到最后，按 60 分钟严格计时做。
- **面试官模式**：每场面试从"考察点"挑 1-2 条匹配岗位 JD，用"问题"开局，"追问"压深。用"评分要点"做事后校准。Mock pack 可以直接用作 onsite 题面，把节奏交给 pack 即可。

## 4. 题目格式约定

每道题统一 5 段结构。这个结构本身就是教学：候选人答题应该照这个结构组织语言，面试官也照这个结构打分。

- **问题**：题面，刻意带具体约束（数字、场景、限制），不要"什么是 X"那种万能题。
- **考察点**：这道题想压的能力维度，通常 3 条。面试官选题时按这个挑。
- **回答框架**：3-5 条要点，是"良好"档候选人应该自然展开的骨架，**不是标准答案**。展开成完整答案是候选人自己的工作。
- **追问**：2-3 条，挖深度用。面试官按候选人初答的薄弱处选追问；候选人自测时把它当二次压力测试。
- **评分要点**：及格 / 良好 / 优秀 三档。这是面试官事后写反馈、候选人对照差距的依据。

---

## 26.1 AI Infra 基础认知与系统分层

### 26.1.1 AI Infra 面试到底在考什么

**问题**

你来面试 AI Infra 工程师。请用 5 分钟说明：AI Infra 面试和"传统后端 / 大数据 / SRE"面试相比，**核心考察点的差异在哪里**？要求不要只列组件名，要把考察维度按"资源 / 链路 / 故障 / 治理"四个层面区分。

**考察点**

- 是否能从"系统视角"出发，而不是从"工具名词"出发
- 是否能把 GPU / 显存 / 通信这些 AI 特有的资源约束讲清楚
- 是否能把诊断、设计、治理三件事串起来表达

**回答框架**

- 先给定义：AI Infra 面试考的是"在 GPU 受限 + 模型重 + 请求动态 + 治理强约束"的工程系统判断力
- 资源维度：与传统后端的差异是 GPU / HBM / 互联 / 网络拓扑成为一等公民
- 链路维度：训练链路（数据 → 训练 → checkpoint → registry → 评测 → 发布）和推理链路（gateway → batching → KV cache → engine → telemetry）必须能各自画出
- 故障维度：能讲一个真实故障的诊断路径（从指标抖动定位到根因）
- 治理维度：成本归因、多租户隔离、模型合规、模型供应链是 AI Infra 区别于传统平台的特殊命题

**追问**

- "会用 vLLM"和"会做推理平台"差在哪里？
- 训练岗和推理岗的回答侧重点应该有什么不同？
- 你做过的项目里，哪部分最体现"治理"维度？

**评分要点**

- **及格**：能讲出 GPU / 显存 / 互联是 AI Infra 的特殊资源；能说推理和训练是两条不同链路
- **良好**：能给出一条完整的诊断路径，能区分训练/推理/平台三类岗位的侧重
- **优秀**：能结合真实故障或项目展开"资源 → 链路 → 故障 → 治理"四个层面，并主动指出自己最强的一面

---

### 26.1.2 AI Infra 与传统后端基础设施的核心差异

**问题**

一个资深后端架构师转岗 AI Infra。他自信"我懂 K8s、负载均衡、消息队列、存储、可观测，足够了"。请你用 3 分钟说服他：**AI Infra 还有哪些他必须重新学的基础概念**，并按"必须立刻补 / 可以边做边补 / 影响很小"分级。

**考察点**

- 是否能识别 AI Infra 的资源 / 调度 / 数值 / 治理特殊性
- 是否能给出有优先级的学习路径，而不是事无巨细堆名词
- 是否能站在迁移者角度而不是新人视角讲

**回答框架**

- 必须立刻补：GPU 显存预算 / KV Cache / 通信拓扑（NVLink/IB/RoCE）/ 数据并行与张量并行 / 推理引擎 batching 模型
- 可以边做边补：模型 registry / checkpoint sharding / fp16/bf16/fp8 数值差异 / RDMA 工具链 / NCCL 调优
- 影响很小：传统后端的语言 / RPC / 缓存 / 队列经验大多直接复用，但要警惕把"无状态服务"思维硬套到推理服务上

**追问**

- KV Cache 为什么不能用 Redis 当外部缓存解决？
- 训练故障与传统后端故障在排错上有什么本质区别？

**评分要点**

- **及格**：能列出 5+ 个 AI 特有概念，明白它们和后端经验不重叠
- **良好**：能分级，且能解释为什么 KV Cache、通信拓扑必须早学
- **优秀**：能从 trade-off 角度讨论"哪些后端经验在 AI Infra 里反而是包袱"（例如对延迟的直觉、对水平扩容的依赖）

---

### 26.1.3 用一张图给新人讲 AI Infra 系统分层

**问题**

一个刚入职 AI Infra 团队的新人，让你用一张图加 5 分钟讲明白整个系统是怎么分层的。请说出**你的图分几层、每层放哪些组件、为什么这么分**，并解释为什么这种分层比"按训练 / 推理"分更适合做平台决策。

**考察点**

- 是否有清晰的分层心智模型（资源层 / 调度运行层 / 工作流层 / 服务化层 / 治理层）
- 是否能解释分层依据（变化速度、责任边界、SLO 类别）
- 是否能用一个例子串起多层

**回答框架**

- 资源层：GPU / 网络 / 存储 / 节点（变化最慢，最贵，硬件相关）
- 调度运行层：K8s + GPU operator + 调度器 + 容器运行时（管"谁能在哪跑"）
- 工作流层：训练 Job / 数据流水线 / 评测 / 发布编排（管"任务怎么走完"）
- 服务化层：推理网关 / 引擎 / KV cache / batching（管"在线请求怎么处理"）
- 治理层：observability / cost / quota / security / registry（管"看得见、算得清、控得住"）
- 横切线：每一个跨层故障都能在这五层间画出诊断路径

**追问**

- 为什么不直接按"训练 / 推理"两大类分层？
- 把 model registry 放在工作流层 vs 治理层，争论点是什么？

**评分要点**

- **及格**：能给出至少四层及代表组件
- **良好**：能讲清楚分层依据（变化速度、责任边界）
- **优秀**：能用一个跨层故障（如"显卡掉线 → 训练失败 → 自动重启 → 队列堆积 → 推理 SLO 超阈"）把整张图盘活

---

### 26.1.4 训练 / 推理 / Agent 三类负载的资源画像差异

**问题**

同一台 8×H100 节点，分别跑 70B 训练、70B 推理、复杂 Agent 任务。请用三段话分别说明：**这三类负载在显存、计算、通信、I/O、生命周期上的资源画像**为什么完全不同，平台为什么不能一套调度策略全用。

**考察点**

- 是否真的清楚三类负载的物理形态差异
- 是否能用资源画像解释调度策略
- 是否能识别关键 trade-off（吞吐 vs 延迟 vs 稳定性）

**回答框架**

- 训练：显存吃满（model + optimizer + activation + checkpoint），计算密集，通信 all-reduce 主导，I/O 集中在 dataset / checkpoint，生命周期天/周
- 推理：显存吃 KV Cache 而非权重副本，计算 prefill/decode 不对称，通信视 TP 而定，I/O 在 prefix cache / log / metric，生命周期分钟/秒
- Agent：单 session 多步 + 工具等待 + 状态保存，GPU 利用率断断续续，更怕长尾，计费维度从 token 升级到 session
- 调度策略：训练用 gang scheduling + topology-aware；推理用 priority + preemption + ABR；Agent 还需要 budget envelope + tool concurrency 限流

**追问**

- 为什么训练 Job 比推理服务更怕"被打断"？
- 一个混部集群里，怎么避免训练 Job 打满 NIC 让推理 SLO 崩？

**评分要点**

- **及格**：能区分三类负载的显存 / 计算 / 通信特征
- **良好**：能解释为什么调度策略必须分类
- **优秀**：能从"组织优先级 / 资源效率 / 风险控制"角度讨论混部时的优先级与隔离策略

---

### 26.1.5 控制面 / 数据面 / 监控面的边界

**问题**

AI 平台经常出现"控制面 / 数据面 / 监控面"这三个词。请定义这三个面在 AI Infra 上下文里**分别承担什么职责**，并用一个推理服务 + 训练 Job 共存的场景说明：哪些操作天然属于控制面，哪些必须留在数据面，哪些只能由监控面发现。

**考察点**

- 概念边界是否清晰，不会把指标采集塞进数据面或把请求路由放到控制面
- 是否能用具体例子区分三面
- 是否理解"三面解耦"对可用性 / 安全 / 升级的意义

**回答框架**

- 控制面：状态变更（提交训练 Job、发布新模型、修改配额、扩缩副本），慢路径，强一致优先
- 数据面：实际请求 / 训练 step / KV 读写 / GPU 计算，热路径，吞吐 + 延迟优先
- 监控面：observability + metric + log + trace + audit，旁路，最终一致即可
- 例子：发版的"切流"是控制面动作，但"路由表生效后流量真的去新副本"是数据面，"是否真的没有用户报错"由监控面回答
- 解耦价值：控制面挂了不应该让数据面停服；数据面慢了不应该让监控面盲；监控面延迟不应该让控制面误决策

**追问**

- 一个 K8s API Server 挂了，你的推理服务是否会立刻挂？为什么？
- 有人想把模型评分逻辑放进 gateway 数据面，你怎么 push back？

**评分要点**

- **及格**：能定义三面，举一个能区分三面的例子
- **良好**：能讨论三面解耦对可用性的影响
- **优秀**：能讨论"数据面里嵌了多少业务逻辑"是组织成熟度的指标，并能给出何时打破解耦的判断

---

### 26.1.6 在线 vs 离线工作负载的资源策略差异

**问题**

平台同时承接在线推理（毫秒级 SLO）、批量推理（小时级）、训练（天级）、夜间数据处理（弹性）。请说明：**为什么不能用同一种 GPU 资源策略统一管理**，并给出一个能在 8×H100 集群上一周运行良好的混部资源策略草案。

**考察点**

- 是否理解四类工作负载在 SLO / 优先级 / 抢占 / 弹性上的不同
- 是否能给出可执行的混部策略
- 是否考虑夜间空闲与抢占恢复

**回答框架**

- 在线推理：高优先级、不可抢占、独占副本、HPA 基于 latency
- 批量推理：中优先级、可抢占、夜间扩容、按租户限额
- 训练：长 Job、gang scheduling、容忍排队、不能跨 SLA 占用在线池
- 夜间数据处理：低优先级 spot、随时被抢、有重试预算
- 资源策略：在线池保留至少 N 张卡常驻；批量与夜间共享 spot 池；训练独占池避免影响在线；用 preemption + priorityClass 形成层级；用 quota 防止单租户挤爆
- 风险：训练池闲置 + 在线池吃紧时怎么"借调"，需要预先约定回收 SLA

**追问**

- 你怎么决定"在线池"应该保留多少张卡常驻？
- 训练突然抢占成功率为零，你怀疑哪些点？

**评分要点**

- **及格**：能区分四类负载的 SLO 和抢占属性
- **良好**：能给出基于 priorityClass + quota 的可执行策略
- **优秀**：能讨论"借调"机制以及失败回滚预案，主动指出 spot 抢占率与训练 checkpoint 间隔的耦合

---

### 26.1.7 一次告警驱动的故障诊断思路

**问题**

凌晨 3 点你被 oncall 叫醒，告警是"在线推理 p99 latency 5 倍上涨，错误率从 0.1% 涨到 4%"。监控平台只给了你一张 latency 图。请说明你**接下来 10 分钟会按什么顺序看哪些数据，每一步排除什么假设**。

**考察点**

- 排错路径是否有结构（流量 / 服务 / 资源 / 依赖）
- 是否会被表象（latency 图本身）牵着走
- 是否能在限定时间内收敛假设

**回答框架**

- 第 1 分钟：确认告警范围（全局 vs 单租户 vs 单实例 vs 某区域）
- 第 2-3 分钟：流量层（QPS、体积、token 长度分布、prompt 是否突变）
- 第 4-5 分钟：服务层（队列长度、batching 命中、KV evict 率、prefill/decode 比例）
- 第 6-7 分钟：资源层（GPU util、HBM、PCIe、节点温度、邻居训练 Job 是否抢占）
- 第 8-9 分钟：依赖层（gateway → engine → registry → 数据库 / 向量库 / external API）
- 第 10 分钟：得出"是流量打飞 / 实例打挂 / 资源被抢 / 依赖故障"中的哪一类，然后才采取动作（限流 / 扩容 / 回滚 / 隔离）

**追问**

- 如果只看 latency 图，最容易做出哪种错误判断？
- 你怎么判断这是"代码 bug"还是"环境异常"？

**评分要点**

- **及格**：有分层排错思路（不只是猜）
- **良好**：能给出明确时间盒和每步排除的假设
- **优秀**：能主动提到"应急动作 vs 根因定位"分阶段处理，避免在 10 分钟内既想止血又想定位

---

### 26.1.8 平台工程师的"最小可运营单元"

**问题**

你接手一个研究院从无到有的 AI 平台，第一个版本只允许做一件事。请说明**你会选择哪个最小可运营闭环**作为 V1（例如"训练 Job 提交 → checkpoint 落盘 → registry 注册 → 评测"），并解释为什么不是别的闭环。

**考察点**

- 能否从组织约束 / 复用价值出发选闭环
- 能否拒绝"先把所有组件都装上再说"
- 能否说清这个闭环的下一步演进

**回答框架**

- 选定一条："训练 Job 提交 → checkpoint → registry → 评测 → staging → 灰度 → 回滚"
- 理由：这条链路覆盖了平台真正不可化简的能力——可重复训练 + 可解释发布 + 可回滚
- 暂不做：feature store、复杂工作流引擎、多租户配额（前期可硬编码）、复杂监控（用现成 stack）
- V2 演进：先扩"评测门禁 + 灰度策略"，再扩"多租户隔离与成本归因"

**追问**

- 这个 V1 没有 model registry，会变成什么样？
- 老板坚持要"先有 OpenAI-style API gateway"，你怎么拒绝或妥协？

**评分要点**

- **及格**：能给出一个具体闭环
- **良好**：能解释取舍并给出 V2 演进顺序
- **优秀**：能讨论"组织成熟度 vs 平台版本"的关系，主动指出最小闭环要承担的"教育成本"

---

### 26.1.9 描述你做过的项目时如何对齐 AI Infra 视角

**问题**

请用 90 秒描述你做过的一个项目，**让面试官能立刻判断**它落在 AI Infra 的哪一层（资源 / 调度 / 运行 / 服务化 / 治理）、对应哪类岗位（训练 / 推理 / 平台 / 可靠性），以及它解决的"不可化简的问题"是什么。

**考察点**

- 自我介绍是否结构化、能直接对位面试 JD
- 是否能用平台层语言而不是组件名描述工作
- 是否能把项目的"为什么必须做"讲清楚

**回答框架**

- 一句话定位：在哪一层 / 哪类负载 / 解决什么不可化简问题
- 一段话约束：当时的资源 / 团队 / SLA / 时限是什么
- 一段话决策：你的取舍点是什么，为什么这么选
- 一段话证据：用什么数字证明它成功（latency、throughput、成本、故障率、上线模型数）

**追问**

- 这个项目里你最遗憾的取舍是什么？
- 让你重做一次会改哪一点？

**评分要点**

- **及格**：能让面试官知道项目大致做了什么
- **良好**：能用平台分层语言定位，给出 1-2 个数字
- **优秀**：能主动反思取舍并把"如果再做一次"的改动讲清楚——这是高阶候选人的最强信号

---

### 26.1.10 AI Infra 工程师 vs 算法 / SRE / DevOps 的边界

**问题**

公司同时招"AI 算法工程师"、"AI Infra 工程师"、"AI SRE"、"DevOps"。HR 让你写一份内部说明：**这四个岗位面对同一个推理性能问题时，各自该负责什么、不该负责什么、交接点在哪里**。

**考察点**

- 是否能从"职责 / 输出 / 故障 ownership"讲清边界
- 是否能避免把 Infra 工程师当成"会跑模型的 SRE"
- 是否能讲清交接点而不是死守边界

**回答框架**

- 算法：负责模型本身（架构、训练数据、效果），输出训练好的 checkpoint 和评测报告
- AI Infra：负责模型如何在平台上跑（训练效率、推理引擎、KV cache、容量），输出可复用的运行时组件 + SLA
- AI SRE：负责生产稳定性（oncall、告警、容量演练、事故复盘），输出 SLO + runbook
- DevOps：负责通用工程能力（CI/CD、镜像、机密管理、网络），输出可复用基础设施
- 交接点：算法→Infra（checkpoint + 评测元数据）；Infra→SRE（SLO + 运行手册）；DevOps→所有（CI / 镜像 / 网络）

**追问**

- "推理 latency 高"这个问题，四个岗位各应该看什么？
- 一个组织如何避免把所有非算法工作都堆给 SRE？

**评分要点**

- **及格**：能区分四岗职责
- **良好**：能讲清交接点，举一个具体协作例子
- **优秀**：能讨论"小团队该如何合并岗位"以及"合并带来的风险"——体现组织成熟度判断

---

## 26.2 硬件、GPU、内存、网络与存储基础

### 26.2.1 同价位 H100 vs A100 vs L40S 选哪个

**问题**

预算 200 万人民币，要新增一批卡用于"70B 模型推理 + 偶尔 13B 微调"。供应商可选 H100 80GB SXM、A100 80GB SXM、L40S 48GB PCIe。请说明你**选哪一种、为什么**，并指出选错时一年内会先在哪个指标上痛。

**考察点**

- 是否真的理解三张卡的显存 / 算力 / 互联差异
- 是否会用"工作负载特征"反推选型，而不是看 Spec 排名
- 是否能预测错误选型的"先痛点"

**回答框架**

- 70B 推理首要约束：HBM ≥ 70 × dtype × 副本因子 + KV，bf16 大约 140 GB → 单卡 80 GB 必须 TP=2 或 FP8 量化
- H100 强在 FP8 / TP2 NVLink 内带宽 / Hopper TMA，做 70B 推理性价比最高
- A100 适合保守路线，但 FP8 缺失，TP2 必走 NVLink，13B 训练 OK，推理吞吐落后
- L40S 显存 48GB 单卡跑 70B 力不从心，多卡 PCIe 互联通信成本高，更适合 13B 推理 / 中小模型并发服务
- 选错的"先痛点"：选 L40S 半年内会因为 70B KV cache 容量被频繁 evict、prefix cache 命中率低、p99 抖动大；选 A100 一年内会被 FP8 推理性价比拉开

**追问**

- 如果加上"未来 6 个月内出现 200B 模型推理"概率 30%，你的选择会改吗？
- "买 H100 但用 A100 镜像"会立刻丢哪些性能？

**评分要点**

- **及格**：能正确算 70B 显存、知道单卡放不下
- **良好**：能从 FP8 / NVLink / TP 角度比较三卡
- **优秀**：能从 TCO + 二手残值 + 未来工作负载角度给出风险加权选项

---

### 26.2.2 显存预算：训练 70B 时显存都花到哪里去了

**问题**

70B 模型用 bf16 + ZeRO-3 + activation checkpointing 训练。请用一张拆解表说明**显存占用的主要来源**（权重 / 优化器 / 梯度 / 激活 / 通信 buffer / cuDNN workspace 等），并指出哪一项最容易"看起来用不多但其实是 OOM 元凶"。

**考察点**

- 是否真的能列全各组成部分
- 是否懂 ZeRO 各 stage 切分了什么
- 是否能讲出 activation 与 micro-batch / sequence length 的关系

**回答框架**

- 权重：70 × 2 = 140 GB（bf16），ZeRO-3 后单 rank ~140/N
- 优化器（Adam, fp32 states）：70 × 8 = 560 GB，ZeRO-3 切到 N
- 梯度：70 × 2 = 140 GB（bf16，部分 impl 仍用 fp32 累积），ZeRO 切分
- 激活：与 batch × seq² × hidden 有关，是 OOM 第一元凶；activation checkpointing 用 1.3-1.4× 计算换显存
- 通信 buffer：NCCL all-reduce/all-gather 临时缓冲区，常被忽略
- cuDNN / FlashAttention workspace：每层固定开销
- 隐藏元凶：peak activation 在 backward 重新算时短暂双倍；fragmentation 让"还有 5GB 但 OOM"

**追问**

- ZeRO-2 vs ZeRO-3 在通信成本上差什么？
- 如何在线判断 OOM 是 activation 还是 fragmentation？

**评分要点**

- **及格**：能给权重 / 优化器 / 梯度 / 激活四项数字
- **良好**：能加上 buffer + workspace + ZeRO 切分逻辑
- **优秀**：能识别 fragmentation / peak / 短暂双倍这种"隐性"显存压力，并给出排查工具（torch.cuda.memory_summary、nvidia-smi、py-spy）

---

### 26.2.3 NVLink / PCIe / IB / RoCE 各管什么

**问题**

新人问你："为什么我们机器内 GPU 通信用 NVLink，跨机用 InfiniBand？PCIe 不能跨机吗？" 请用一段 3 分钟的回答把**互联拓扑**讲清楚，明确每条链路的带宽量级、延迟量级、典型用途，并解释 RoCE 与 IB 的真实差异。

**考察点**

- 是否清楚带宽 / 延迟 / 拓扑分层（intra-GPU / intra-node / inter-node）
- 是否懂 NCCL 在不同链路下走的协议差异
- 是否能把 IB 和 RoCE 的工程差异讲准（不只是"IB 贵 RoCE 便宜"）

**回答框架**

- NVLink/NVSwitch：~900 GB/s（H100 NVL 4），同节点 GPU 间，All-reduce / TP 走它
- PCIe Gen5 x16：~64 GB/s，CPU↔GPU、GPU↔NIC，跨节点必须经过它出 NIC
- InfiniBand HDR/NDR：~200/400 Gbps 单 NIC，RDMA + 硬件流控，跨节点首选
- RoCEv2：以太网上的 RDMA，便宜易部署，但要 PFC + ECN 治理拥塞，否则丢包毁灭性
- NCCL 选路：单机 NVLink → P2P → SHM；跨机 IB/RoCE → ring/tree all-reduce
- 拓扑感知：训练 Job 必须 topology-aware 调度，否则 8 卡 ring 跨机就退化

**追问**

- RoCE 出现 PFC storm 的根因是什么？
- 没有 NVSwitch 时 8 卡 H100 的 all-reduce 走什么拓扑？

**评分要点**

- **及格**：能给四类链路带宽量级
- **良好**：能解释 NCCL 在不同拓扑下的算法差异
- **优秀**：能讲 RoCE 部署陷阱（PFC、buffer、leaf-spine 拥塞）以及 IB 的"贵在哪里值"

---

### 26.2.4 HBM vs DDR vs SSD：内存层次的真实代价

**问题**

KV Cache 越来越大，有人提出"用 SSD 当 GPU 显存的 swap 池"。请用一段话**否定或限定这个方案**，说明 HBM / DDR (CPU memory) / NVMe SSD 在带宽 / 延迟 / 颗粒度上的真实差距，并解释为什么 vLLM 的 swap 是 GPU↔CPU memory 而不是 GPU↔SSD。

**考察点**

- 是否清楚带宽数量级差距（HBM 2-3TB/s, DDR 60-100GB/s, NVMe 5-10GB/s）
- 是否清楚延迟数量级差距（HBM ns, DDR 100s ns, NVMe 50-100us）
- 是否能识别"颗粒度"问题——KV swap 的最小单位与设备 IO 大小不匹配

**回答框架**

- HBM：~3 TB/s，~100 ns，与计算同设备
- CPU DDR：~80 GB/s，~100 ns 但要走 PCIe → 实测 GPU↔CPU 带宽 ~32 GB/s（PCIe Gen5 x16）
- NVMe SSD：~7 GB/s 顺序，~80 us 随机延迟，块单位 4KB-128KB
- vLLM swap：每 block 几 KB，需要细粒度搬运，CPU memory 是性价比甜点
- SSD 方案的问题：单次 swap 80us 对 50ms 的 token-by-token decode 是灾难；NVMe 写擦放大缩短寿命；多请求并发 IO 抖动会让 p99 长尾爆炸
- 例外：批量 prefix cache 的"温层"可以放在 NVMe（按 minute/hour 复用，不在热路径）

**追问**

- 那"CXL memory" 能不能改变这个判断？
- 把 KV cache 全放 CPU 而不 swap 到 GPU，会怎样？

**评分要点**

- **及格**：能给三层带宽 / 延迟数量级
- **良好**：能从颗粒度否定 SSD swap
- **优秀**：能讨论"温分层 cache" 与 CXL 等未来变化的影响

---

### 26.2.5 GPU 利用率 90% 可能是个谎言

**问题**

监控显示 GPU 利用率 92%，但训练 step time 比同模型公开 benchmark 慢 30%。请说明 `nvidia-smi` 的 GPU-Util 指标到底**测的是什么、为什么会高估真实算力利用**，并给出更靠谱的"GPU 真在干活"的判断方法。

**考察点**

- 是否知道 GPU-Util 实质是"过去采样窗口里有 kernel 在跑的时间比"
- 是否懂 SM occupancy / Tensor Core utilization / memory bandwidth 才是真实 metric
- 是否能给出可执行排查路径

**回答框架**

- GPU-Util 是采样比例，只要有 kernel 在跑就算 100%（哪怕一个空 memcpy）
- 真实"算力利用"：SM occupancy（Nsight Compute / DCGM 看 sm__cycles_active）、Tensor Core util、HBM 带宽利用、kernel 时间分布
- 30% 慢的常见原因：CPU 数据加载 bottleneck（GPU idle 但有少量 kernel 拉满采样）、kernel launch overhead、Python / nccl 同步空隙、显存碎片导致 fragmenting alloc
- 排查路径：`dcgmi dmon` 看 SM occupancy；`torch.profiler` / Nsight Systems 看 kernel timeline；DataLoader profile

**追问**

- DataLoader 拉满 CPU 但 GPU 空，监控会显示什么？
- 单机 8 卡训练，1 张卡 GPU-Util 60% 其它 95%，怎么排查？

**评分要点**

- **及格**：能识别 GPU-Util 不等于算力利用
- **良好**：能给出 SM occupancy / Tensor Core util 等真指标
- **优秀**：能给出完整的排查工具链 + 具体 case 路径

---

### 26.2.6 NUMA 与 PCIe 拓扑对 AI 工作负载的影响

**问题**

8 GPU 单节点服务器，跑同一个训练任务，**绑定 NUMA 后**性能比不绑 NUMA 高 15%。请解释 NUMA 是什么、PCIe topology 怎么和它互动、为什么会影响训练性能，并给出生产上**可复用的绑核策略**。

**考察点**

- 是否懂 NUMA 节点、本地内存 vs 跨 socket 远程内存
- 是否懂 GPU 与哪个 NUMA 节点 PCIe 直连
- 是否懂 NCCL / DataLoader / pin memory 与 NUMA 的耦合

**回答框架**

- NUMA：多 socket CPU，每个 socket 有本地内存，访问对方 socket 内存慢 2-3x
- GPU 通过 PCIe 接到某一个 socket（root complex），"GPU0-3 接 socket0，GPU4-7 接 socket1"是常见拓扑
- 不绑 NUMA：DataLoader 进程可能在 socket1 跑但喂数据给 socket0 的 GPU0，每 batch 都跨 socket 拷贝
- 绑核策略：`numactl --cpunodebind={socket} --membind={socket}` 把 worker / DataLoader / NCCL helper 都绑到对应 GPU 所在 socket
- 验证：`nvidia-smi topo -m` 看 GPU↔CPU 关系，再用 `numastat` 观测 cross-node memory 访问

**追问**

- 推理服务什么时候反而不需要严格绑 NUMA？
- 一台机器 NIC 接在 socket0，GPU 在 socket1，跨机 RDMA 怎么不踩坑？

**评分要点**

- **及格**：能讲清 NUMA + PCIe 拓扑
- **良好**：能给出 numactl + topo 验证流程
- **优秀**：能讨论"NIC 跨 socket"的进一步陷阱以及容器环境下的绑核难点

---

### 26.2.7 网络延迟 50us vs 200us 对训练的真实影响

**问题**

集群有两套 RoCE 配置：A 套 50us 延迟、200 Gbps 带宽；B 套 200us 延迟、400 Gbps 带宽。同样跑 175B 模型 ZeRO-3 训练，**哪一套快**？为什么？请给出基于 all-reduce / all-gather 量级的估算。

**考察点**

- 是否能用通信量 + 延迟 + 带宽算 step 时间
- 是否知道 ZeRO-3 的通信特性是"很多次中等通信"
- 是否能避免"带宽更高就一定快"的误区

**回答框架**

- ZeRO-3 通信量：每 step 大约 2× 模型参数（all-gather + reduce-scatter），175B bf16 = 350 GB
- 延迟主导场景：参数切片很多 → 每片小消息 → 200us 延迟 × N 次 = 显著增加
- 带宽主导场景：长消息聚合后，400 Gbps 优势大
- 估算：如果 ZeRO-3 切成 1024 片，每片~340MB，A 套 50us+340MB/25GB/s=13.6ms，B 套 200us+340MB/50GB/s=6.8ms → B 仍可能更快
- 但若 GPU 数 N 大（>256），延迟项放大 → A 反超
- 结论：取决于"切片粒度 × GPU 数 × 拓扑"，NCCL 算法（ring vs tree）也会改变

**追问**

- A 套延迟 50us 但偶尔丢包 0.001%，对训练有什么影响？
- 如果改成 175B 推理 TP=8，结论会变吗？

**评分要点**

- **及格**：能算 all-reduce 通信量
- **良好**：能讨论延迟 vs 带宽的 trade-off 取决于消息大小
- **优秀**：能引入 NCCL 算法 / 拓扑 / GPU 规模做加权判断

---

### 26.2.8 NIC 选型：单端口 200G vs 双端口 100G

**问题**

新建 GPU 集群，每节点 8 张 H100，NIC 预算只够装一种：A. 单端口 ConnectX-7 NDR 400G，B. 双端口 100G ×2，C. 四端口 25G ×4。请按"训练 / 推理 / 通用"三个用途分别**给推荐**并解释。

**考察点**

- 是否懂"NIC 数量与 GPU-NIC 拓扑亲和"对 NCCL 的影响
- 是否懂 GPUDirect RDMA 要求 NIC 与 GPU 在同 NUMA / PCIe switch
- 是否能避免"看起来 4×25G = 100G 等价"误区

**回答框架**

- 训练：A 优先（NDR 400G + GPUDirect RDMA 简化拓扑，所有 GPU 共享一根高速管道，all-reduce 性能最佳）
- 推理：C 也能用（推理跨机通信少，多端口给副本隔离 / 多租户网络隔离方便）
- 通用 / 混部：B 比较平衡，双端口可分别接两套 leaf 做冗余
- 误区：4×25G 总带宽 100G 看似等于双 100G，但 NCCL 单流走单端口、GPUDirect 不支持端口聚合，实际 effective bandwidth 远低
- 拓扑约束：NIC 应与对应 GPU 在同 PCIe switch / 同 NUMA，否则跨 socket 慢路径

**追问**

- 推理服务什么场景反而需要高带宽 NIC？
- 双端口接两套 leaf 做冗余，怎么避免 ECMP hash 不均？

**评分要点**

- **及格**：能区分训练 / 推理对带宽的不同需求
- **良好**：能讲 GPUDirect RDMA 的拓扑要求
- **优秀**：能讨论多端口聚合的真实带宽利用 / hash 均衡 / 故障切换

---

### 26.2.9 存储分层：训练数据 / checkpoint / 推理日志 应该怎么放

**问题**

为一个 200 人 AI 研究院做存储规划：训练数据集 PB 级、checkpoint 每天 TB 级、在线推理日志每天 GB 级、模型 registry 全量 TB 级。**只能用三种存储介质**：本地 NVMe / 共享并行文件系统（Lustre/CephFS）/ 对象存储（S3 兼容）。请规划放置方案。

**考察点**

- 是否清楚四类数据的访问模式（顺序大块 / 随机小块 / 写多读少 / 写少读多）
- 是否懂训练 IO 瓶颈在哪
- 是否懂 checkpoint 的 lifecycle 与存储成本

**回答框架**

- 训练数据：对象存储（成本最低）+ 本地 NVMe 缓存层（每 epoch 预热）；并行 FS 仅当模型实在大且数据访问随机时用
- Checkpoint：写在并行 FS 或对象存储 multipart upload；本地 NVMe 做 staging buffer 防写阻塞
- 推理日志：对象存储（按租户分桶 + 生命周期策略）；热数据可短暂 ES/ClickHouse
- Registry：对象存储 + 元数据 DB（PostgreSQL / etcd），不要把"模型文件 + 元数据"放同处
- 关键陷阱：训练 DataLoader 直接打 S3 网络风暴；checkpoint 同步阻塞训练 step；推理日志写本地满磁盘

**追问**

- 一个 200GB checkpoint 写 30 秒会怎样？怎么用 async + zero-copy 优化？
- 训练数据 PB 级但每 epoch 只用 10% 子集，怎么设计 cache 命中？

**评分要点**

- **及格**：能给四类数据各选一种存储
- **良好**：能讨论本地 NVMe 缓存策略 + checkpoint async
- **优秀**：能讨论"S3 网络风暴 / 多租户配额 / 数据生命周期"等运营层细节

---

### 26.2.10 一台 8×H100 节点的功耗 / 散热预算

**问题**

新建 100 台 8×H100 GPU 节点的训练集群。请说明：**单节点峰值功耗 / 平均功耗、散热方案、机柜密度** 大约多少，以及电力 / 冷量在采购时常被忽略的"卡脖子"点是什么。

**考察点**

- 是否对 GPU + CPU + NIC 整机功耗有量级感
- 是否懂风冷 / 液冷的边界
- 是否能识别"机柜空间够 GPU 但电力 / 冷量不够"

**回答框架**

- H100 SXM5 单卡 TDP ~700W，8 卡 5.6kW；CPU + 内存 + NIC + PSU 损耗加约 1-2 kW，整机峰值 6.5-8 kW
- 风冷极限：标准 19" 机柜 12-15 kW，能塞 1-2 台 8×H100 节点；高密风冷可达 20-30 kW
- 液冷：直触液冷 / 后门换热 50-80 kW/cabinet，能上 6-8 台节点
- 机柜密度由"功率 + 冷量"主导，不是"U 数"
- 卡脖子：很多机房 PDU 单路 3.6 kVA，1 台 8×H100 就要 2 路；冷却塔冷水温度 > 18°C 时高密风冷直接 throttle

**追问**

- 100 台节点跑训练，PUE 1.3 vs 1.5 一年差多少电费？
- 你会同意"为了密度上液冷"还是"先用风冷过渡"？什么决策点？

**评分要点**

- **及格**：能给单节点功耗量级
- **良好**：能区分风冷 vs 液冷的密度边界
- **优秀**：能讨论 PUE / 电费 / PDU 路数 / 冷却塔水温这种运营约束

---

### 26.2.11 同代 GPU 的不同变体（SXM / PCIe / NVL）该怎么选

**问题**

H100 有 SXM5、PCIe、NVL（双卡 188GB）三个变体；A100 有 SXM4、PCIe、80GB / 40GB。请说明：**这些变体的真实工程差异**（不仅是 spec 差），各自最适合什么工作负载。

**考察点**

- 是否懂 SXM 必须配特定 baseboard、不能像 PCIe 一样灵活
- 是否懂 NVL 对长 context / 大模型的优势
- 是否懂二手市场 / 供应链对选型的影响

**回答框架**

- SXM：高带宽 NVLink 全互联、统一 baseboard 8/4 卡、TDP 高、必须供应商整机
- PCIe：可插任意 PCIe 服务器、跨卡只有 PCIe 带宽、TDP 低 50-100W、灵活但通信弱
- NVL（H100 NVL 188GB）：双卡通过 NVLink Bridge 共享 188GB，针对长 context 推理 / 大模型推理特别好
- 选型：训练优先 SXM（NVLink 全互联）；推理 70B+ 长 context 选 NVL；中小模型推理 / 多租户 / 灵活上下电选 PCIe；冷启动多变量 / 二手市场选 PCIe
- 工程差异：SXM baseboard 维修必须送供应商；PCIe 失败可独立替换

**追问**

- "买 SXM 但只用 4 卡"会浪费什么？
- NVL 与 TP=2 是同一个东西吗？

**评分要点**

- **及格**：能区分三种变体
- **良好**：能讲 NVL 的 188GB 共显存优势
- **优秀**：能从供应链 / 维修 / 二手残值角度给出工程取舍

---

### 26.2.12 PCIe Gen5 / CXL / NVLink C2C 哪个先值得跟进

**问题**

老板给你一份"未来 18 个月新硬件评估"任务，候选包括 PCIe Gen5 普及、CXL 1.1/2.0 落地、NVLink C2C（Grace-Hopper 这类 GPU-CPU 一致性互联）。请按**"对你当前业务（推理为主 + 中等训练）的边际收益"** 排序并说明为什么。

**考察点**

- 是否对未来 18 个月新硬件路线有判断
- 是否能区分"概念性新东西"和"立刻能产生收益"
- 是否能用业务负载反推优先级

**回答框架**

- PCIe Gen5：已普及（H100/B100），新平台标配，主要影响 GPU↔NIC、CPU↔GPU 带宽，立刻有收益（GPUDirect RDMA + KV swap 都受益）
- NVLink C2C / Grace Hopper：CPU-GPU 一致性内存大幅简化大模型推理 KV swap、Prefix cache 管理；对长 context 推理特别有价值，但要绑定 NVIDIA 整机方案
- CXL：理念好（内存池化、跨节点共享），但 1.1 仅本机扩展，2.0 / 3.0 真正用起来需要 CPU + 设备 + OS 协同，未来 18 个月还不够成熟
- 排序：PCIe Gen5（已落地，选型直接拿）> NVLink C2C（强相关业务方向，值得 POC）> CXL（关注但暂不投入）
- 风险：把 CXL 当救命稻草，会比"早一年用上 NVL/Grace Hopper"亏更多

**追问**

- 如果业务是"超大 context（128K+）推理"，结论会变吗？
- CXL memory pool 真正落地时，AI Infra 的哪些组件最先受益？

**评分要点**

- **及格**：能区分三种技术
- **良好**：能给出 18 个月内的边际收益排序
- **优秀**：能讨论"对当前组织规模 / 业务方向"的具体影响并主动调整排序

---

## 26.3 训练基础设施与分布式训练

### 26.3.1 单机 8 卡训练 step time 不稳，怎么定位

**问题**

8×H100 单机训练 13B 模型，每 step 时间从 1.2s 到 2.8s 抖动。GPU-Util 平均 90%，没有任何错误日志。请给出**你的诊断顺序**，至少要排除 4 类可能根因。

**考察点**

- 是否能用结构化方法排查训练性能不稳
- 是否懂训练 step 的时间组成（forward / backward / all-reduce / data load）
- 是否会用 profiler 工具

**回答框架**

- 第一步：profile 单 step 时间组成（torch.profiler / Nsight Systems），确定瓶颈是 compute / comm / data / memory
- 排除 1：DataLoader 抖（CPU 抢占、磁盘 IO、num_workers 不够、prefetch 不够）→ 看 dataloader idle time
- 排除 2：NCCL all-reduce 抖（PCIe 共享、温控降频、邻居进程）→ 看 all-reduce time variance
- 排除 3：显存 fragmentation 触发 alloc / free 慢 → 看 cudaMalloc 调用次数
- 排除 4：GPU 自身降频（功耗 / 散热）→ `nvidia-smi -q -d CLOCK,POWER,TEMP`
- 排除 5：跨节点干扰（共享文件系统抖、邻居 Job）

**追问**

- DataLoader 和 GPU 计算 overlap 不足时，监控会出现什么特征？
- 8 卡里偶尔 1 卡明显慢，最可能是什么？

**评分要点**

- **及格**：能列 3 类根因
- **良好**：能给 profile 工具 + 时间组成拆解
- **优秀**：能加上 GPU 降频 / 邻居 Job / 共享存储等"非显然"维度

---

### 26.3.2 ZeRO 1 / 2 / 3 与 FSDP 的关键区别

**问题**

新人问"为什么我们不直接全用 ZeRO-3 / FSDP？" 请说明 **ZeRO-1 / 2 / 3 与 PyTorch FSDP** 各自切了什么、通信代价多少倍、什么场景下用哪个最划算。

**考察点**

- 是否真的能数清楚每 stage 切了什么、通信多少
- 是否能从模型规模 / 集群规模反推选择
- 是否懂 FSDP 与 ZeRO-3 的工程差异（API、checkpoint、internal mesh）

**回答框架**

- ZeRO-1：切 optimizer states，all-reduce 通信不变（仍 ≈ 2× params/step），显存省最多
- ZeRO-2：切 optimizer + gradient，通信仍 ≈ 2× params（reduce-scatter + all-gather）
- ZeRO-3：切 optimizer + gradient + parameters，每 step 通信 ≈ 3× params（forward all-gather + backward all-gather + reduce-scatter）
- FSDP：PyTorch 原生实现，与 ZeRO-3 等价，但 API 更"PyTorch-y"，支持 mixed precision / activation checkpointing 一体配置；HSDP 还能 hybrid sharding（机内全切，机间 DP）
- 选择：13B 单机能放下 → ZeRO-1/2 即可；70B+ 必走 ZeRO-3/FSDP；多节点训练用 HSDP / 2D mesh 减少跨机 all-gather
- 场景反例：通信带宽不够时 ZeRO-3 反而比 ZeRO-2 + activation checkpointing 慢

**追问**

- HSDP 的 hybrid 是怎么"机内全切机间 DP"的？
- ZeRO-3 在 backward 时为什么还要再 all-gather 一次参数？

**评分要点**

- **及格**：能区分各 stage 切了什么
- **良好**：能给出每 stage 的通信量量级
- **优秀**：能讨论 HSDP / 2D mesh / 通信带宽与 stage 选择的耦合

---

### 26.3.3 张量并行（TP）什么时候比纯 DP 划算

**问题**

70B 模型训练，集群有 64 张 H100。请说明：**TP=2 / TP=4 / TP=8 与纯 DP 的取舍**，并解释为什么 TP 通常不跨机器。

**考察点**

- 是否懂 TP 的通信特性（每层 forward+backward 各一次 all-reduce）
- 是否懂 TP 必须 NVLink 内、跨机会爆炸
- 是否能算 TP × DP 的 mesh 怎么构造

**回答框架**

- TP 切的是单层矩阵：每层 2 次 all-reduce，频率高，对延迟极敏感
- TP 跨机：跨机 200us 延迟 × 每层 N 次 = 不可接受；必须限制在 NVLink 内
- 70B / 64 GPU：TP=4（节点内）+ DP=16（跨节点 ZeRO）较常见；若 NVSwitch 全互联 TP=8 可行
- 与纯 DP 对比：纯 DP 显存放不下 70B；TP+DP 把"装下 + 高效跨机"两件事分开
- 取舍：TP 越大单层吞吐越高但通信越多；超过 NVLink 域立即崩
- 工具：3D parallelism（TP × PP × DP）才是 100B+ 的标准配置

**追问**

- TP=8 但 NVLink 出现 1 张卡掉线，会发生什么？
- TP 切了 attention head 后，head 数不能整除 TP 怎么办？

**评分要点**

- **及格**：能讲 TP 的通信特性
- **良好**：能解释"为什么 TP 不跨机"
- **优秀**：能讨论 3D parallelism mesh 构造和 head 数整除限制

---

### 26.3.4 流水并行（PP）的 bubble 与微批划分

**问题**

8 stage 流水并行训练 175B 模型，micro-batch 数 = 8 时 bubble overhead ~50%，micro-batch 数 = 32 时 bubble ~12%。请解释：**bubble 是什么、为什么微批数能减少它、为什么不能无限增大微批数**。

**考察点**

- 是否懂 PP 调度（GPipe / 1F1B / Interleaved）
- 是否能算 bubble 比例 = (P-1) / (P + M - 1)（GPipe）
- 是否懂微批数受显存 / 数值稳定性限制

**回答框架**

- Bubble：流水线启动 / 排空阶段，部分 stage 空闲
- GPipe bubble：P stage、M micro-batch，bubble 比例 ≈ (P-1)/(P+M-1)；M=8, P=8 → 7/15 ≈ 47%；M=32 → 7/39 ≈ 18%
- 1F1B / Interleaved：稳态期间 forward / backward 交错，bubble 减半
- 微批数上限：每 stage 显存 = micro_batch × activation；越多越占显存
- 数值稳定性：grad accumulation 等价大 batch，但 batch norm / loss scale 要小心
- 实战：DeepSpeed PP / Megatron PP 都用 1F1B，配合 activation checkpointing

**追问**

- Interleaved 1F1B 是怎么把 bubble 从 (P-1)/(P+M-1) 降到 (P-1)/(K(P+M-1)) 的（K 是 chunks）？
- PP 出现某个 stage 显著慢，会引发什么？

**评分要点**

- **及格**：能解释 bubble 概念
- **良好**：能给 GPipe bubble 公式 + 1F1B 改进
- **优秀**：能讨论 Interleaved + chunks 公式 + stage 慢导致的整体崩

---

### 26.3.5 NCCL 调优：超时 / 死锁 / 性能差的常见原因

**问题**

训练突然卡住，NCCL 日志只有 `NCCL WARN Connect to ... failed`。请说明 **NCCL 常见故障类型**（超时 / 死锁 / 性能差）的诊断和处理思路，以及 `NCCL_DEBUG=INFO` 之外你会用的环境变量。

**考察点**

- 是否懂 NCCL 错误的常见模式
- 是否会用 NCCL 调试 / 调优环境变量
- 是否能区分"配置问题"和"硬件问题"

**回答框架**

- 超时：通常是 PCIe / IB 链路抖动 / NCCL_TIMEOUT 阈值低 / 一卡 hang 拖死全部 → `NCCL_TIMEOUT` 调大、`NCCL_BLOCKING_WAIT=1` 看哪卡先卡
- 死锁：rank 间 collective 顺序不一致（控制流分歧 / hang 在不同 op）→ 检查代码所有 collective 是否对齐
- 性能差：拓扑不对（`NCCL_TOPO_FILE` / `NCCL_ALGO`）、走错 NIC / 协议（`NCCL_IB_HCA`、`NCCL_NET_GDR_LEVEL`）、PXN 没开
- 调试变量：`NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS=ALL`、`NCCL_ASYNC_ERROR_HANDLING=1`、`TORCH_NCCL_TRACE_BUFFER_SIZE`
- 硬件 vs 配置：相同代码换一台机器能跑 → 硬件 / 网络；换一台机器还挂 → 代码 / 配置

**追问**

- `NCCL_NET_GDR_LEVEL` 改大改小各有什么影响？
- 一个 rank hang 导致全部 timeout，怎么定位是哪个 rank 先 hang？

**评分要点**

- **及格**：能讲三类故障
- **良好**：能给环境变量 + 排查路径
- **优秀**：能区分硬件 vs 配置 + 给出"先 hang"定位方法（NCCL flight recorder / py-spy）

---

### 26.3.6 梯度同步、梯度累积与 batch size 的耦合

**问题**

有人说"梯度累积 8 步 = 等价 batch size 放大 8 倍"，但实际 loss 曲线和 LR scheduler 会偏离。请解释**梯度累积对 batch size / LR / BN / dropout / log step 的真实影响**，并给出"梯度累积 + LR warmup + batch size scaling rule"如何同时正确。

**考察点**

- 是否懂梯度累积的语义
- 是否懂 batch norm / dropout 与梯度累积的不同
- 是否懂 linear scaling rule / sqrt scaling rule

**回答框架**

- 累积语义：N 个 micro-batch 各 forward+backward，梯度累加，第 N 步才 optimizer.step()
- 等价：等价于 batch_size × N（仅对 grad / loss 而言）
- BN 不等价：每个 micro-batch BN 独立 running stats，与真正大 batch BN 行为不同；用 SyncBN 或 GroupNorm 才等价
- Dropout 不等价：每 micro-batch 独立 dropout mask，与大 batch 不同（影响小但要知道）
- LR scaling：linear rule 用 batch_size_eff = batch_size × N × DP，warmup steps 也要按 N 缩放
- Log step：步数变成原来 1/N，metric / scheduler 要按"effective step"计

**追问**

- 你怎么验证"梯度累积 N 步"和"实际 batch×N"的 loss 曲线一致？
- 累积过程中显存 peak 比 batch×N 高还是低？

**评分要点**

- **及格**：知道累积等价 batch 放大
- **良好**：能讲 BN / dropout 的差异
- **优秀**：能讨论 scaling rule + step 计数 + peak memory 等多维细节

---

### 26.3.7 Activation checkpointing 何时是亏的

**问题**

Activation checkpointing 用 1.3-1.4× 计算换显存。请说明**什么场景下它反而是亏的**，并给出一个判断标准来决定每一层是否值得 checkpoint。

**考察点**

- 是否真的清楚 activation checkpointing 的计算 / 显存 trade-off
- 是否能识别哪些层重算特别贵
- 是否能用工具量化"显存收益 / 计算损失"

**回答框架**

- 划算：层 activation 占显存大、重算便宜（普通 Linear / LayerNorm）
- 亏：注意力层 (FlashAttention 已经融合)、rematerialize 反而触发 recompute kernel 而不是融合 → 实际比"用更激进 ZeRO-3 + 多卡"更慢
- Selective checkpointing：只 checkpoint 大头层（attention 输出、FFN 中间），不要无脑全切
- 量化标准：activation_size_saved × seq_len × batch / extra_compute_time，得到"每秒省多少显存"
- 工具：torch.utils.checkpoint + activation_size profile；与 ZeRO-3 + offload 对比真实 step time

**追问**

- FlashAttention 启用后还该 checkpoint attention 吗？
- selective checkpoint 的"哪些层选"有自动化方法吗？

**评分要点**

- **及格**：知道 1.3× 计算开销
- **良好**：能讲 selective checkpoint
- **优秀**：能给量化标准 + 工具链

---

### 26.3.8 训练故障恢复：checkpoint 频率 / 容错 / 自动重启

**问题**

70B 训练预计跑 20 天，节点 MTBF 约 10 天（150 卡集群每天有故障概率）。请设计 **checkpoint 策略 + 自动恢复机制**，目标是单次故障损失不超过 30 分钟。

**考察点**

- 是否会算"checkpoint 间隔 vs 失败损失 vs IO 开销"
- 是否懂分布式 checkpoint（async / shard）
- 是否懂 K8s / scheduler 层 auto-restart

**回答框架**

- 频率：30 分钟为目标，则 checkpoint 间隔 ≤ 30 min 但不能太频繁（写 IO 阻塞训练）
- 异步 checkpoint：参数同步切片到 host memory，后台写盘，训练继续
- Sharded checkpoint：FSDP / DCP 各 rank 写自己的分片，避免单 rank 聚合瓶颈
- 自动重启：scheduler（K8s + Volcano / Kubeflow）检测节点失败，gang restart 整个 Job 从最近 checkpoint
- 健康检查：NCCL hang detector + heartbeat + topology check
- 数据：DataLoader resume from sample index（不是从 epoch 头）

**追问**

- 节点失败但 NCCL 没报错（silent hang），怎么发现？
- 30 min checkpoint 间隔，IO 占 step time 多少能接受？

**评分要点**

- **及格**：能给频率 + 异步 + 自动重启
- **良好**：能讲 sharded checkpoint + DataLoader resume
- **优秀**：能讨论 silent hang 检测 + 时间预算分配（IO / failure / recovery）

---

### 26.3.9 学习率 / Optimizer 状态在多卡下的一致性陷阱

**问题**

ZeRO-3 训练时，team 偶尔反馈"resume 后 loss 跳"。请列出**多卡训练时 LR / optimizer state / RNG state 的一致性陷阱**，以及如何在 checkpoint 中正确保存恢复。

**考察点**

- 是否懂 LR scheduler / optimizer 与 step 计数的耦合
- 是否懂 RNG state（数据 shuffle / dropout）在分布式下的处理
- 是否懂 ZeRO 切片 optimizer state 的恢复

**回答框架**

- LR scheduler：必须用 step 而非 epoch 计数；resume 后 scheduler.last_epoch 要正确
- Optimizer state：ZeRO 切片后每 rank 只存自己的；恢复时按 rank 加载，DCP / FSDP 自动处理
- RNG state：每 rank 保存自己的 torch / numpy / random / cuda RNG；DataLoader 的 sampler 也要保
- AMP loss scaler：fp16 训练的 GradScaler 状态要保
- 数据 sampler：DistributedSampler 在 epoch 内的 sample index 要保（用 stateful sampler）
- 跳 loss 的常见根因：scheduler last_epoch 没保 / RNG 没保 / DataLoader 从头开始走了相同样本

**追问**

- 怎么验证"resume 后第一个 step 的输入和原来挂掉前的下一个 step 是同样数据"？
- AMP scaler 不保会怎样？

**评分要点**

- **及格**：能列 LR / optimizer / RNG 三类
- **良好**：能讲 scheduler last_epoch + sampler index
- **优秀**：能给"resume 一致性"验证方法 + AMP 影响

---

### 26.3.10 训练资源排队：FCFS / 优先级 / Backfill 怎么选

**问题**

研究院 1000 张卡，训练任务从小（8 卡 / 1 天）到大（256 卡 / 30 天）都有，还要插急（VIP 项目 64 卡 / 立即）。请设计调度策略：**FCFS / Priority / Backfill / Gang scheduling 怎么组合**。

**考察点**

- 是否懂大型 HPC / AI 集群的调度模式
- 是否能用 backfill 提升大 Job 等待期间的小 Job 利用率
- 是否懂 gang scheduling 必要性（部分启动 = 死锁）

**回答框架**

- 基础：所有 AI 训练 Job 必须 gang scheduling（要么所有 worker 全启动要么全不启动）
- 队列分层：高优先级（VIP / 在线服务）、中优先级（生产训练）、低优先级（探索 / spot）
- FCFS within priority：同优先级先到先服务
- Backfill：大 Job 排队时，放小 Job 进去填空，但不能让大 Job starvation（设最大延迟 SLA）
- 抢占：高优先级到达时可抢低优先级，但要 checkpoint friendly（提前 5 min 通知）
- 工具：Volcano、Kueue、Slurm、Ray Cluster

**追问**

- 大 Job 排队 7 天，一个研究员把它拆成 32 个小 Job 蒙混过关，平台怎么办？
- 怎么测算"backfill 利用率提升"是否真有效？

**评分要点**

- **及格**：知道 gang + priority
- **良好**：能讲 backfill / starvation / 抢占
- **优秀**：能讨论拆任务作弊 + 利用率测算 + 工具选型

---

### 26.3.11 训练数据流水线的常见瓶颈

**问题**

70B 训练，dataset 12TB。GPU-Util 平均只有 65%，profile 显示有 30% 时间 GPU 在等数据。请说明**数据流水线的常见瓶颈**和系统级优化思路。

**考察点**

- 是否懂训练数据 pipeline 各阶段（read → decode → preprocess → tokenize → batch → host→device）
- 是否能识别瓶颈层
- 是否懂大数据集的 prefetch / shuffle / caching 策略

**回答框架**

- 阶段：source（S3/CephFS）→ DataLoader worker → CPU preprocess → tokenize → pin memory → host→device transfer
- 常见瓶颈：S3 网络（每 epoch 重新拉）、tokenizer 慢（HF python tokenizer 单线程）、num_workers 太少
- 系统级优化：预 tokenize 后存 parquet/webdataset；本地 NVMe 缓存层；num_workers = num_cpu_cores / num_gpus_per_node × 2；prefetch_factor 增加
- 高级：streaming dataset（webdataset / mosaic-streaming）边下边训；多机间 data sharding 避免重复读
- 工程陷阱：num_workers 过多反而 fork 慢、内存爆、CPU 抢占；pin_memory 没开导致 host→device 慢

**追问**

- 12TB 数据 epoch 用了 30 分钟读取，怎么估算"加 NVMe 缓存"是否值得？
- 每 epoch 不同 shuffle，怎么保证 resume 数据顺序一致？

**评分要点**

- **及格**：能列 3+ 阶段 + 瓶颈
- **良好**：能给 num_workers / prefetch / 预 tokenize 优化
- **优秀**：能讨论 streaming dataset + cache 命中估算 + resume 一致性

---

### 26.3.12 LoRA / 全量 / continue pretraining：训练形态的资源差异

**问题**

公司业务现在做这三件事：base model 预训练（从零）、SFT 全量微调、LoRA 微调。请说明这三种形态在**显存 / 计算 / 数据 / checkpoint 体积 / 部署形态**上的差异，并解释为什么 LoRA 在生产里被广泛采用。

**考察点**

- 是否能区分三种训练形态的资源画像
- 是否懂 LoRA 的"低秩"机制 + 推理时合并
- 是否懂 LoRA 在 multi-tenant 推理下的优势

**回答框架**

- 预训练：100B+ token，万卡级别集群，数据 / 通信 / 容错全部拉满
- SFT 全量：千万 token，几十-几百卡，所有参数都更新，checkpoint 全模型大小
- LoRA：万-千万 token，单机或几卡，只更新低秩 adapter（几 MB-几百 MB），checkpoint 小
- 推理部署：LoRA 可"base model + 多个 adapter 共存"，按请求切换 adapter；vLLM / Punica 支持 multi-LoRA serving
- 优势：显存 + 训练成本低 1-2 个数量级；多任务共享 base；adapter 易分发 / 版本管理
- 限制：能力上限受 LoRA rank；与 base 距离过远的领域适配性差

**追问**

- LoRA rank 选 8 / 16 / 64 各影响什么？
- multi-LoRA serving 时 prefix cache 怎么处理？

**评分要点**

- **及格**：能区分三形态资源画像
- **良好**：能讲 LoRA 推理部署优势
- **优秀**：能讨论 multi-LoRA + prefix cache + rank 选择 + 局限性

---

## 26.4 数据、制品、Checkpoint 与 Registry

### 26.4.1 训练数据集 / Tokenized 数据 / 模型 checkpoint 各应该怎么版本化

**问题**

平台的训练流水线里有三类持久化制品：原始数据集、tokenized 数据、模型 checkpoint。请说明**各自合适的版本化方案**（git-lfs / DVC / S3 + manifest / OCI artifact 等），并解释为什么不能用同一种统一管理。

**考察点**

- 是否懂大文件版本化的工程边界
- 是否懂"内容哈希 vs 语义版本"的差异
- 是否能讲清楚为什么 git-lfs 不适合 PB 级数据集

**回答框架**

- 原始数据集：S3 + manifest 文件（按内容 hash 引用），用 DVC / lakeFS 管理"哪一版数据用哪些样本"
- Tokenized 数据：webdataset / parquet shard + 版本目录；与 tokenizer hash 强绑定，tokenizer 变了 ID 空间就废
- Checkpoint：注册到 model registry（MLflow / Vertex / Bento）；存对象存储；带 metadata（dataset version、code commit、metric、训练超参）
- 不能统一原因：粒度差距（PB vs TB vs GB）、访问模式不同（数据顺序读、checkpoint 整体取）、生命周期不同
- 共同要求：可追溯（任何一个 checkpoint 能回到精确的代码 + 数据 + 超参）

**追问**

- 数据脱敏前后如何区分两个版本？
- 如果有人用错了 tokenizer，怎么从 checkpoint 反查这个错误？

**评分要点**

- **及格**：能给三类合适的工具
- **良好**：能讲 hash + manifest + 元数据
- **优秀**：能讨论 tokenizer 绑定、生命周期、可追溯性

---

### 26.4.2 Checkpoint sharded vs full：什么时候选哪个

**问题**

70B 模型 ZeRO-3 训练，写 checkpoint 时有两种策略：sharded（每 rank 写自己分片）和 full（rank 0 聚合写完整文件）。请说明**两种策略在写延迟 / 恢复方便 / 跨集群迁移**上的差异，以及生产实践应该混合使用。

**考察点**

- 是否懂 sharded checkpoint 的工程优势 / 缺点
- 是否懂"训练 checkpoint vs 发布 checkpoint"是不同需求
- 是否会用 PyTorch DCP / FSDP / DeepSpeed 的 checkpoint 接口

**回答框架**

- Sharded：每 rank 并行写自己的分片，写延迟低；恢复必须用相同 mesh shape；跨集群迁移困难
- Full：rank 0 聚合所有分片，写延迟高（all-gather + 单节点 IO 瓶颈）；恢复方便；可直接发布
- 训练用 sharded：高频 checkpoint 不阻塞训练
- 发布用 full：模型仓库存的是完整 checkpoint，便于跨配置加载
- 工具：DCP（PyTorch Distributed Checkpoint）支持 sharded ↔ full 转换；FSDP `state_dict_type` 可切换
- 生产实践：训练时 sharded + 定期 export full（每 N hours / 评估通过）

**追问**

- 集群从 32 卡 ZeRO-3 切到 64 卡 ZeRO-3，sharded checkpoint 怎么用？
- full checkpoint 写 30 分钟阻塞训练，怎么避免？

**评分要点**

- **及格**：能讲 sharded vs full 主要差异
- **良好**：能讨论训练 / 发布两种用途
- **优秀**：能讲 DCP 的转换 + 跨 mesh 迁移 + async export 实践

---

### 26.4.3 Model registry 应该存什么、不存什么

**问题**

新人想"把所有模型相关的东西都丢到 registry"——checkpoint、训练数据、评测结果、推理配置、prompt 模板。请说明**registry 的合理边界**：哪些必须在 registry 里，哪些必须在外部，哪些可以引用。

**考察点**

- 是否懂 registry 的核心职责（版本 / 谱系 / 发布）
- 是否能区分"模型本体"和"使用模型的上下文"
- 是否懂不同类型制品的存储成本

**回答框架**

- 必须在 registry：模型权重（reference 到对象存储 URL）、元数据（架构、size、quantization、license）、谱系（dataset version、code commit、parent model）、评测结果摘要、发布状态
- 必须在外部：原始训练数据（太大、合规）、详细评测样本（追加多）、prompt 模板（迭代快）、推理配置（部署相关）
- 可引用：dataset URL、tokenizer URL、训练 Job ID、A/B 实验结果
- 边界原则：registry 是"模型这个对象的真相"，不是"所有 ML 工件的总仓库"

**追问**

- 一个客户问"这个模型在哪里训的、用什么数据"，registry 应该多快能答上？
- 同一个 checkpoint 不同量化版本（FP16 / FP8 / INT4）怎么管？

**评分要点**

- **及格**：能讲 registry 存模型 + 元数据
- **良好**：能讨论谱系 / 发布状态 / 引用 vs 内嵌
- **优秀**：能讨论量化变体管理 + 合规追溯

---

### 26.4.4 Checkpoint 体积太大怎么办：压缩 / 量化 / 切片

**问题**

70B 模型 fp16 checkpoint 140 GB。请说明在**存储成本 / 加载时间 / 跨地域同步**三个维度上，可用的优化路径有哪些（量化、压缩、稀疏化、tensor parallel split），各自代价是什么。

**考察点**

- 是否懂量化级别（fp8 / int8 / int4 / GPTQ / AWQ）对体积和精度的影响
- 是否懂通用压缩（zstd / lz4）在浮点 checkpoint 上效果有限
- 是否能讲跨地域同步策略（multipart upload / cross-region replication / CDN）

**回答框架**

- 量化：fp16 → fp8（×0.5）→ int8（×0.25）→ int4（×0.125），精度损失递增；后训练量化 vs 量化感知训练
- 通用压缩：浮点参数熵接近最大，zstd 通常只能省 5-15%；不要指望
- 稀疏化：剪枝 + 稀疏存储，特定模型 50% 稀疏可节省 30-40%
- TP split：每 rank 存自己分片，单文件小，但加载时必须按对应 rank 数
- 跨地域：multipart upload + CRR + CDN 缓存；模型注册时 region affinity；按需懒加载
- 取舍：体积优化必须配合"目标用途"——训练 checkpoint 不能量化，推理 checkpoint 才能

**追问**

- 量化后再训练（fine-tune）会怎样？
- 一个东京的推理服务要拉一个旧金山的 checkpoint，最快的方案？

**评分要点**

- **及格**：知道量化是主要手段
- **良好**：能区分量化级别 + 加载时机
- **优秀**：能讨论 region affinity / multipart / 量化感知训练 trade-off

---

### 26.4.5 模型供应链与签名：怎么防"供应链投毒"

**问题**

你的平台允许第三方 partner 上传自己的 fine-tuned LoRA adapter 给客户使用。请说明**模型供应链的攻击面**（恶意权重 / 隐藏后门 / 注入恶意 prompt 行为），以及你会用什么机制（签名 / scan / sandbox / runtime guard）防御。

**考察点**

- 是否能识别模型供应链的攻击面
- 是否懂 sigstore / OCI artifact signing
- 是否能讲"上传时静态扫描 + 运行时 guard"组合

**回答框架**

- 攻击面：权重里嵌 trigger backdoor、推理时反复输出特定串、消耗预算无意义循环、隐式数据外泄
- 静态扫描：weight diff（与公开 base model 比较）、known backdoor pattern、metadata 检查
- 签名：每个上传制品 sigstore 签名，引用必须验签；OCI artifact + cosign
- Sandbox：上传后先在隔离环境跑评测集（包括安全集），通过才发布
- Runtime guard：output filter、abnormal token sequence detector、budget hard cap
- 治理：partner 信任分级、违规通报机制、强制版本撤回

**追问**

- 你怎么定义"安全评测集"？
- 如果一个 LoRA 在评测集上看起来好，但生产 1 个月后才触发 backdoor，怎么发现？

**评分要点**

- **及格**：能列攻击面 + 签名概念
- **良好**：能讲静态 + 运行时双层
- **优秀**：能讨论评测集设计 + 长期监控 + 撤回机制

---

### 26.4.6 数据合规：训练数据 / 用户输入 / 模型输出 各自怎么处理

**问题**

公司在欧盟和美国都有客户。请说明你的平台如何处理 GDPR / CCPA 等合规要求，**针对训练数据 / 用户在线输入 / 模型输出三类数据**分别给出策略。

**考察点**

- 是否懂训练数据 vs 在线数据合规要求差异
- 是否懂"模型输出"也是合规对象
- 是否能讲数据居留 / 删除权 / 审计

**回答框架**

- 训练数据：来源 license / consent 审查；脱敏（PII 去标识）；保留训练数据→模型版本追溯（合规要求"删除某用户数据"时能定位影响哪些模型）
- 用户在线输入：按租户隔离 + 不持久化（除非租户同意）；EU 数据不出境（地域路由）；prompt 中的 PII 需要识别 + redact
- 模型输出：可能反向推断训练数据（membership inference）→ 输出过滤；客户要求"忘记我"时同时清推理 cache + log + trace
- 治理：数据流图（DPIA）；DPO 角色；定期合规审计；事件响应预案

**追问**

- 模型本身可能"记住"训练数据中的 PII，怎么办？
- 一个 EU 用户要求删除他的所有数据，你的平台 30 天内能做到吗？

**评分要点**

- **及格**：能区分三类数据合规
- **良好**：能讲地域路由 + 删除权
- **优秀**：能讨论 membership inference + 模型遗忘 + DPIA 工程化

---

### 26.4.7 评测制品（Eval results）的版本化与回归

**问题**

每次发模型必须跑评测集。但评测集本身也在演进（新增样本、修复标注错误、增加安全集）。请设计**评测制品的版本化 + 回归对照机制**，让历史模型对比的数字可信。

**考察点**

- 是否懂"评测集变了 → 历史数字不可比"的根本问题
- 是否能给评测集 / 模型 / 评测代码 三方版本化方案
- 是否懂"快慢评测分层"

**回答框架**

- 三方版本化：模型 vN / 评测集 vM / 评测代码 vK；每次评测记录三元组
- 不变量：发布报告必须明确写 (M, K)；同 (M, K) 对比的不同 N 才可信
- 重跑机制：评测集升级到 vM+1 后，对最近 K 个模型重跑一次，让趋势图重新连续
- 分层：快评测（每次提交，~10min）+ 慢评测（每次发布，几小时）+ 安全 / 红队评测（季度）
- 数据：评测结果存为 parquet + manifest，可被 dashboard / A/B 工具消费
- 漂移：定期重抽样评测集，识别 metric 漂移与样本漂移

**追问**

- 一个评测集发现标注错误，怎么处理已经发布的报告？
- "新模型在 v2 评测集上分数高 5%，但 v1 评测集上低 3%"——你怎么解释给业务方？

**评分要点**

- **及格**：能讲版本化 + 重跑
- **良好**：能讲三方版本化 + 分层
- **优秀**：能讨论标注修复溯源 + 漂移识别

---

### 26.4.8 Feature store / Embedding store 与训练-推理一致性

**问题**

推荐系统场景，特征在训练和推理都用，但训练用历史数据离线计算，推理用实时数据在线计算。请说明**training-serving skew** 的来源和**feature store** 怎么保证一致性。

**考察点**

- 是否懂 train-serve skew 的根本原因
- 是否懂 feature store 的双写 / 单源策略
- 是否能讲离线 / 在线特征定义同源

**回答框架**

- Skew 来源：定义不同（训练用 SQL 离线计算，推理用代码在线计算）、时间不同（训练用 t 时刻特征，推理用 t-Δ）、数据不同（训练用 sampled，推理用 raw）
- Feature store 解：单一定义（DSL / 代码生成 SQL + 在线服务）；双写到 offline store（训练读）+ online store（推理读）
- 时间一致：point-in-time correctness——训练时取每条样本对应时间点的特征快照
- 工具：Feast / Tecton / Hopsworks
- 监控：定期比对 offline vs online 特征分布，发现 skew 报警

**追问**

- 一个新加的特征，怎么决定回填多久的历史 offline data？
- Embedding 如何 train-serve 一致？

**评分要点**

- **及格**：能解释 skew 来源
- **良好**：能讲 feature store 双写 + point-in-time
- **优秀**：能讨论回填策略 + embedding 同步 + 漂移监控

---

### 26.4.9 制品仓库：OCI artifact 还是自建对象存储

**问题**

公司原有 docker registry 存镜像。新需求要存 model checkpoint / dataset / LoRA adapter。请说明**复用 OCI registry（artifact）vs 自建对象存储 + 元数据 DB** 各自的取舍。

**考察点**

- 是否懂 OCI artifact 概念（registry 不只是镜像）
- 是否懂大对象在 OCI registry 上的工程局限
- 是否能给出"什么场景该混用"

**回答框架**

- OCI artifact：复用现有 registry 基础设施（auth、rbac、scan、push/pull）；缺点是大对象（>10GB）push/pull 流式不友好、不便部分下载
- 自建对象存储 + 元数据：S3 + PostgreSQL；适合 PB 级、partial download、跨 region replication；缺点是从零搭 auth / scan / lifecycle
- 混用：小对象（< 1GB）走 OCI（LoRA / 量化模型）；大对象（>10GB）走 S3（base model checkpoint），由 registry 元数据 reference
- 现实：OCI artifact 标准（ORAS）正在演进，但工具链对大对象支持仍弱

**追问**

- LoRA adapter 用 OCI artifact 比直接 S3 有什么实际好处？
- 一个 200GB 的 base model 通过 OCI registry 拉取，会踩什么坑？

**评分要点**

- **及格**：能区分两种方案
- **良好**：能讨论混用策略
- **优秀**：能讨论 ORAS 工具链局限 + auth/scan/replication 对比

---

### 26.4.10 数据 / 制品的生命周期治理

**问题**

平台运行 2 年后，对象存储里堆了 50 PB 数据：训练数据集若干代、checkpoint 数千个、评测结果、log/trace。请设计**生命周期治理策略**：保留多久、按什么规则归档、怎么避免"删错了恢复不了"。

**考察点**

- 是否能给出基于"最后访问 / 业务用途"的多级保留策略
- 是否懂归档（S3 Glacier）vs 删除
- 是否懂软删 + 审计 + 恢复路径

**回答框架**

- 分层：hot（< 30 天，标准存储）/ warm（30-180 天，IA 存储）/ cold（>180 天，Glacier）/ delete
- 规则：训练数据集 keep 5 年（法律/合规）；checkpoint 按"是否发布过 / 是否 referenced"分级保留；log/trace 30-90 天压缩归档；评测结果永久（很小）
- 软删：先标记 deleted，30 天 grace period 才物理删；期间可恢复
- 审计：所有删除动作必须签名 + 记录 + 通知 owner
- 自动化：tag 驱动的 lifecycle policy（owner / project / lastAccessed）；定期 report"长期未访问但占地"的 top owners
- 灾难预案：跨 region replication 高优先级数据；定期演练恢复

**追问**

- 误删一个发布过的 checkpoint，30 天内能恢复吗？31 天后呢？
- 怎么估算"再保留 1 年"对成本的影响？

**评分要点**

- **及格**：能给分层 + 软删
- **良好**：能讨论 tag 驱动 + 审计
- **优秀**：能讨论恢复演练 + 成本测算 + 跨 region replication 策略
