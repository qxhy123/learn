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

---

## 26.5 推理服务、KV Cache、Batching 与推理引擎

### 26.5.1 Prefill 与 decode 为什么要分开看

**问题**

请用一段话说清楚：**prefill 和 decode 在计算密集度、延迟约束、batch 友好度、KV 写入模式上的本质差异**，并解释为什么生产推理引擎都把它们分开调度。

**考察点**

- 是否懂 prefill 计算密集（compute-bound）/ decode memory bound
- 是否能讲 batch 友好度差异
- 是否能讲分开调度的工程必要性

**回答框架**

- Prefill：Q × K × V 全长 attention，compute-bound（GPU SM 跑满）；高吞吐但单步慢；batch 友好（多请求合并大 GEMM 利用率高）
- Decode：Q 长度恒为 1，K/V 散落 cache，memory-bound（HBM 带宽决定）；低延迟但 GPU 利用率低；batch 友好但收益递减
- KV 写入：prefill 一次写 N 个 slot；decode 每步只写 1 个 slot（导致 fragmentation 问题）
- 调度差异：prefill 长 → 一旦插入 batch 会拖慢 decode；chunked prefill 切片混入；continuous batching 让 decode 持续转
- 引擎设计：vLLM 的 prefill / decode 阶段在同一个 step 但 budget 分开；分离部署（prefill cluster + decode cluster）也是高级形态

**追问**

- 为什么 chunked prefill 能改善 decode TTFT？
- "prefill / decode disaggregation" 真实生产场景有什么收益？

**评分要点**

- **及格**：能讲两阶段计算特性差异
- **良好**：能讲 batch + KV + chunked prefill
- **优秀**：能讨论 disaggregation + 各自调度 budget

---

### 26.5.2 KV Cache 大小如何估算 + 显存预算

**问题**

70B 模型，bf16，num_layers=80，num_kv_heads=8，head_dim=128。请算一个**单请求 max_seq_len=4096 时 KV Cache 占多少 GB**，并说明在 80 GB H100 上 8 卡 TP=8 部署时，KV cache 池能给多少个并发请求用。

**考察点**

- 能否正确算 KV cache 公式：2 × layers × kv_heads × head_dim × seq_len × dtype_bytes
- 能否算 TP 切分后单卡 KV 量
- 能否预估剩余显存给 KV pool

**回答框架**

- 单请求 KV：2(K+V) × 80 × 8 × 128 × 4096 × 2(bf16) = 1.34 GB / request
- TP=8 切 num_kv_heads（如果 kv_heads ≥ 8）：单卡 KV = 1.34 / 8 = 168 MB / request
- 模型权重 TP=8 切：70B × 2 / 8 = 17.5 GB / 卡
- 单卡剩余给 KV：80 - 17.5 - 5（activation/buffer）≈ 57 GB
- 并发请求数：57 / 0.168 ≈ 339 请求 × 4096 token KV
- 实际：还要扣除 prefix cache 的常驻占用、安全 margin、最长请求的尾部
- 推论：max_seq_len 越长每请求显存涨线性，并发数掉线性

**追问**

- num_kv_heads=8 但 TP=16 时怎么切？
- FP8 KV cache 能让并发数翻倍吗？

**评分要点**

- **及格**：能算单请求 KV
- **良好**：能算 TP 切分后并发数
- **优秀**：能讨论 head 数与 TP 整除约束 + FP8 + 实际 margin

---

### 26.5.3 PagedAttention 的核心机制和工程价值

**问题**

PagedAttention 论文宣称能把 KV Cache 显存利用率从 ~20% 提到 ~96%。请用 OS 虚拟内存的类比解释**它的核心机制**，并说明为什么这个看似只是"换个数据结构"的改动能带来这么大收益。

**考察点**

- 是否懂 block-based KV cache 的核心思想
- 是否能用 OS 类比讲清楚（block table = page table）
- 是否能识别"碎片化是真痛点"

**回答框架**

- 核心：把 KV cache 切成固定大小 block（默认 16 token），每请求维护 block table 间接寻址
- OS 类比：block = 物理页，block table = 页表，block_size = 页大小，prefix sharing = page sharing with COW
- 为什么大幅提升：传统按 max_seq_len 静态分配 → 99% 请求只用 5-10% 实际长度 → 巨大碎片；PagedAttention 按需分配 block，无内部碎片
- 二次价值：block 是共享单位，prefix cache hit 时直接复用 block；多请求相同 system prompt → 显存复用
- 工程含义：BlockManager 类比内存分配器，attention kernel 改造支持 block_table 寻址

**追问**

- 为什么 block_size 默认 16 而不是 1 或 256？
- vLLM 的 paged attention CUDA kernel 比标准 attention kernel 多了什么？

**评分要点**

- **及格**：能讲 block + table 寻址
- **良好**：能用 OS 类比 + 解释碎片
- **优秀**：能讨论 block_size 选型 + 内核改动

---

### 26.5.4 Continuous batching 的最大价值和工程坑

**问题**

vLLM / TGI / TRT-LLM 都有 continuous batching。请说明**它相对静态 batching 的最大收益**（不是"提升 throughput X%"这种笼统说法），以及生产里**最容易踩的几个坑**。

**考察点**

- 是否懂 continuous batching 真正解的是什么问题（短请求等长请求）
- 是否能列出生产坑（admission policy、长尾 prompt、token budget 配错）
- 是否能给出一个具体场景说明

**回答框架**

- 最大价值：长请求 + 短请求混合时，传统静态 batch 必须等最长那个跑完才能解 batch；continuous 每完成一个就接新的，GPU 利用率持续高位
- 反例：所有请求长度一样时收益小
- 生产坑：admission policy 不对（队尾大请求一直进不来 / 高 priority 请求排长队）；token budget 算错（max_num_batched_tokens 太小→吞吐低；太大→TTFT 抖）
- 长尾 prompt：8K 长 prompt 一进 batch，prefill 阶段独占 budget，已 decode 中的请求 latency 跳；用 chunked prefill 解
- 与 prefix cache 互动：continuous + prefix sharing 时新请求 admit 看的是"扣除 cached prefix 后的实际 token"

**追问**

- 在线 chat scenario，continuous batching 和 chunked prefill 必须一起开吗？
- max_num_batched_tokens 怎么定？

**评分要点**

- **及格**：能讲核心价值
- **良好**：能列 2-3 个生产坑
- **优秀**：能讨论与 chunked prefill / prefix cache 的耦合 + budget 决策方法

---

### 26.5.5 Prefix Cache 的命中条件与失效场景

**问题**

Prefix cache 听起来很美：相同 system prompt 复用 KV。但生产里常见"按理说该命中却没命中"。请列出**最容易让 prefix cache 命中失败的 5 个场景**，以及如何在监控里发现。

**考察点**

- 是否懂 prefix cache 的命中条件（token id 完全一致 + tokenizer 一致 + LoRA id 一致）
- 是否能识别"模板差一个空格 / 时间戳"导致每次失败
- 是否懂 cache eviction 模式（容量满 / TTL）

**回答框架**

- 失效 1：模板里嵌了时间戳 / 用户 ID / 会话 ID → 每次 token 序列不同
- 失效 2：tokenizer 升级或不同 tokenizer 共用一个 cache → ID 空间不一致
- 失效 3：LoRA adapter 切换 → KV 计算路径不同（有些引擎按 (base, adapter) 二元组分别 cache）
- 失效 4：cache 容量小或 LRU 频繁 evict
- 失效 5：随机 sampling 时第 1 个 token 不同导致后续无法复用（这个是"无法复用"不是"失效"）
- 监控：prefix_cache_hit_rate（按租户、按模板）；cache eviction rate；不同模板版本的命中率分布

**追问**

- 怎么帮业务方"重新设计 system prompt 让命中率从 30% 升到 80%"？
- 多个 LoRA adapter 共享 prefix cache 在工程上可行吗？

**评分要点**

- **及格**：能列 3 个失效场景
- **良好**：能讲 tokenizer / LoRA / 模板设计
- **优秀**：能给出可执行的命中率提升方法 + multi-LoRA 工程性讨论

---

### 26.5.6 Chunked prefill 解决什么真实问题

**问题**

Chunked prefill 把长 prompt 拆成多个 chunk 与 decode 混合 batch。请说明：**它对 TTFT 和 inter-token latency 的真实影响**（不是"减少 latency"这种泛泛说），以及什么场景反而不应该开。

**考察点**

- 是否真的懂 chunked prefill 改善的是哪些请求的延迟
- 是否能讲"长 prompt 自身 TTFT 可能变长"
- 是否能识别不该开的场景

**回答框架**

- 改善：已 decode 中的请求 inter-token latency 不被新长 prompt 干扰（短请求体感稳）
- 不改善：长 prompt 自己的 TTFT 反而可能略长（被切片 + 共享 budget）
- 不该开：如果工作负载全是短 prompt（API 调用、短问答），开 chunked 反而引入额外 overhead
- 与 prefix cache 协同：chunked prefill 第 N 块需要前 N-1 块的 KV → 必须在 cache 里 → 共用同一个 kernel 路径
- 工程：vLLM `enable_chunked_prefill=True`，`chunked_prefill_size` 默认 512 但要根据负载调
- 监控：先看长尾请求 TTFT 分布是否改善 + 短请求 ITL 抖动是否降

**追问**

- chunked prefill_size 太大有什么影响？太小呢？
- 如果一个长 prompt 一直在被 chunk 但又来了更多新请求，会被"插队"卡住吗？

**评分要点**

- **及格**：能讲 ITL 改善 + TTFT trade-off
- **良好**：能讨论 size 调优 + 短请求场景
- **优秀**：能讨论与 prefix cache kernel 共用 + 长 prompt 插队风险

---

### 26.5.7 推理引擎选型：vLLM / TGI / TRT-LLM / SGLang 怎么挑

**问题**

业务要 deploy Llama-3-70B chat 推理服务，要求 p99 < 1s 单 turn、支持 multi-LoRA、有量化、长 context（32K）。请给一份**引擎选型建议**：vLLM / TGI / TRT-LLM / SGLang 各自适合什么、不适合什么、最坑的差异在哪。

**考察点**

- 对四个主流引擎的真实差异有判断
- 能从 LoRA / 量化 / context / 部署形态匹配业务需求
- 能讲生产工程坑（兼容性 / 模型支持 / 调试难度）

**回答框架**

- vLLM：开源活跃、模型支持广、PagedAttention + prefix cache 强、multi-LoRA 用 Punica；新引擎 V1 重构
- TGI：HuggingFace 系，model loading 方便，运营简单，性能稍弱；continuous batching + medusa
- TRT-LLM：NVIDIA 优化最深，吞吐和延迟最佳；但 model 支持滞后、build 慢、调试难
- SGLang：RadixAttention + structured output 强，prefix cache 设计精妙；模型支持还在追赶
- 建议（70B + multi-LoRA + 32K context）：vLLM 是稳妥首选；如果延迟极致可考虑 TRT-LLM 但接受工程负担
- 选型坑：模型支持名单（最新模型可能没 day-1 支持）；量化兼容（AWQ/GPTQ/FP8 各引擎进度不同）；K8s 集成（liveness、graceful drain）

**追问**

- 切换引擎时，prefix cache 命中率会重新预热多久？
- TRT-LLM 的 build 流程为什么慢？

**评分要点**

- **及格**：能区分四引擎主要特点
- **良好**：能根据业务需求给出建议
- **优秀**：能讨论选型坑 + 切换成本

---

### 26.5.8 Speculative decoding 真实收益与 acceptance rate

**问题**

Speculative decoding（draft / Medusa / EAGLE）在 paper 里宣传 2-3× 加速。请说明：**真实生产里的收益期望、关键变量（acceptance rate）、何时不值得开**。

**考察点**

- 是否懂 spec decode 数学模型（acceptance rate × N draft tokens 决定收益）
- 是否懂 draft 模型 / 多头 / EAGLE 的差异
- 是否能识别"高并发场景反而亏"

**回答框架**

- 数学：每 verify step 期望接受 N_acc 个 token，平摊每 token 时间 = T_target/(1+N_acc)；acceptance 50% 时 N=4 期望提速 2x
- Draft 路径：独立 draft 模型 / 多头（Medusa）/ tree-based（EAGLE）；draft 模型选型决定 acceptance
- 生产收益：低并发场景（GPU 空闲）显著加速；高并发场景 batch 已经填满 GPU，spec 反而把单步变长 → 总吞吐下降
- 不值得开：高吞吐 batch、acceptance < 30%、draft 模型本身重（draft cost > 节省）
- 工程：vLLM / TRT-LLM 都支持，但 acceptance 监控必须做

**追问**

- "draft 用 base model 量化版"和"独立小 draft"哪个更好？
- 多 LoRA 服务时 spec decode 怎么处理？

**评分要点**

- **及格**：能讲 acceptance rate 决定收益
- **良好**：能讲不值得开的场景
- **优秀**：能讨论 draft 选型 + multi-LoRA + 监控

---

### 26.5.9 长 context（128K+）推理的真实瓶颈

**问题**

新需求：支持 128K context 推理。请说明从 4K 到 128K，**计算 / 显存 / KV cache / 数据访问模式 / 用户感知**各自如何变化，以及生产怎么扛。

**考察点**

- 是否懂 attention 复杂度 O(n²) 在长 context 上的爆炸
- 是否懂 KV cache 显存压力急剧上升
- 是否懂长 context 实际是"prefill 慢 + KV 显存大 + 命中率低"复合问题

**回答框架**

- 计算：attention O(n²) → 4K→128K 计算量 1024× ；FlashAttention 仍要扫全部，但 IO 改善 → 实测 prefill 时间 50-100×
- 显存：KV cache 线性增长，128K 单请求 KV 比 4K 大 32×（比例随 layer/head/dtype）
- KV cache：长 context 下 prefix cache 命中率往往低（用户内容差异大）
- 数据访问：prefill 时一次性巨大 attention，对 HBM 带宽要求极高
- 用户感知：TTFT 从 100ms 到 5-10 秒（长 prompt 的硬伤）
- 生产策略：FP8 KV cache 减半显存；分层 KV（最近 N token full precision，远端量化）；chunked prefill 必开；硬件升级到 H100/H200/B200
- 算法：sliding window / Mamba-like alternative / context distillation

**追问**

- 用户给的 128K context 里只有最后 4K 是真问题，怎么优化？
- 长 context 和 multi-LoRA 同时存在，哪个先牺牲？

**评分要点**

- **及格**：能讲 O(n²) + 显存线性
- **良好**：能讲 FlashAttention + FP8 KV
- **优秀**：能讨论分层 KV + sliding window + 硬件路线 + 业务级降级

---

### 26.5.10 Streaming token / SSE / WebSocket 的工程取舍

**问题**

OpenAI API 风格的 streaming 输出，前端可以用 SSE 或 WebSocket。请说明**两者在 AI Infra 场景的取舍**，以及 streaming 在网关 / 引擎 / 前端 三层各自的关键设计点。

**考察点**

- 是否懂 SSE vs WebSocket 在生产的差异
- 是否懂 streaming 对 gateway / engine / 前端的不同要求
- 是否能讲 token-level 流式的 backpressure 与超时

**回答框架**

- SSE：HTTP/1.1 单向流，简单、易代理、CDN 友好；浏览器原生支持
- WebSocket：双向、可发中断信号、支持二进制；但需要 LB 配合、proxy 兼容性差
- 选择：纯 token 输出走 SSE 即可；需要客户端中断 / 上传中流式数据用 WebSocket
- Gateway：必须支持 chunked transfer / 不缓冲；HTTP/2 long-lived；超时配置（keep-alive）
- Engine：generate_stream 接口；token 产生后立刻 push（不等到一行）
- 前端：背压控制（前端慢时不要堆 buffer）；超时和中断处理（用户关闭 tab）
- 中断：客户端断开 → engine 应该快速停止 generate（释放 GPU）

**追问**

- HTTPs/2 vs HTTPs/3 对 streaming 有差异吗？
- 用户中断后 engine 能在多少 ms 内真正停？

**评分要点**

- **及格**：能讲 SSE vs WebSocket 选择
- **良好**：能讲三层设计要点
- **优秀**：能讨论中断响应延迟 + backpressure + 协议演进

---

### 26.5.11 推理服务的 Auto-scaling 怎么设计

**问题**

在线推理服务 QPS 早 8 点 100 高峰、深夜 5。GPU 资源贵。请设计**autoscaling 策略**：基于什么指标、扩缩容速度、怎么避免冷启动 latency 高、怎么处理 prefix cache 在扩缩容时的失效。

**考察点**

- 是否懂 GPU 推理服务的 autoscaling 不能像无状态服务那样简单
- 是否懂 prefix cache 是"状态"，扩缩容会重置
- 是否能讲冷启动模型加载 / cache 预热

**回答框架**

- 指标：QPS 不够（GPU 利用率不直接反映负载），用 token throughput / queue depth / p99 latency 综合
- 扩容：detect → schedule → pull image → load model（10s-1min）→ warm cache → serve；冷启动可达分钟级
- 缩容：先 drain 流量 + 等 in-flight 请求结束（graceful），不要直接 SIGKILL
- 预热：pre-warm replica（启动后跑一段 dummy 请求），prefix cache 必要时 replicate
- 反尖刺：max scale-up rate（每分钟最多 +N 副本）；最低保留副本数防雪崩
- 反过敏：scale-down 滞后（连续 5 min 低负载才缩）

**追问**

- 模型加载 30 秒能否提前？（hot standby 池？lazy load？）
- prefix cache 在副本间能否共享？

**评分要点**

- **及格**：能讲基本扩缩容策略
- **良好**：能讲冷启动 + drain
- **优秀**：能讨论 prefix cache 跨副本 + hot standby + max rate 防尖刺

---

### 26.5.12 推理 SLO 设计：TTFT / TPOT / E2E latency 怎么定

**问题**

业务说"我要 p99 latency < 1s"。请说明**这句话在 LLM 推理上为什么是有歧义的**，并给出你建议的 SLO 拆分（TTFT / TPOT / E2E）以及监控方式。

**考察点**

- 是否懂 LLM 推理 latency 不是单一数字
- 是否能区分 TTFT / TPOT / E2E
- 是否能讲不同业务场景对哪个 SLO 敏感

**回答框架**

- TTFT（Time To First Token）：用户感知"开始响应"的时间
- TPOT（Time Per Output Token）/ ITL（Inter-Token Latency）：稳态吐字速度
- E2E latency：完整请求总耗时，依赖 output token 数（无法单独 SLO）
- 业务映射：chat 关心 TTFT + TPOT；批量摘要关心 E2E；长 reasoning 关心 TPOT 的稳定性
- "p99 < 1s"歧义：1s TTFT？1s 总？1s 100 token？需要拆
- SLO 草案：p99 TTFT < 500ms + p99 TPOT < 50ms（即 20 tok/s）+ p99 E2E < (TTFT + N×TPOT)
- 监控：分位数监控 + 按租户 / 按 prompt length 分桶，避免长尾被平均

**追问**

- 上游网关也增加 latency，谁来 own"端到端 p99"？
- streaming 场景"用户已经开始读"和"全部读完"哪个 SLO 重要？

**评分要点**

- **及格**：能讲 TTFT + TPOT
- **良好**：能讨论业务映射 + 歧义
- **优秀**：能讨论分桶监控 + 上下游 SLO 责任划分

---

### 26.5.13 量化推理（FP8 / INT8 / INT4 / AWQ / GPTQ）取舍

**问题**

70B 模型 fp16 单卡放不下、TP=2 又太贵。同事建议量化。请说明 **FP8 / INT8 / INT4、PTQ vs QAT、AWQ vs GPTQ** 的工程差异，以及生产里你会怎么选。

**考察点**

- 是否能区分量化级别 / 算法
- 是否懂量化的精度 - 速度 - 显存权衡
- 是否懂 KV cache 量化与 weight 量化的不同

**回答框架**

- 级别：FP16(2B) → FP8(1B) → INT8(1B) → INT4(0.5B) → INT2/3 实验性
- PTQ（post-training quantization）：训练后量化，无需重训，最常用；AWQ / GPTQ 都是 PTQ
- QAT（quantization-aware training）：训练时模拟量化，精度好但要重训，不常用
- AWQ（activation-aware）：基于 activation 分布选 salient channel 保留高精度，对 LLM 友好
- GPTQ：逐层最优量化（OBS 算法），精度好但 build 慢
- KV cache 量化：FP8 KV 在 H100 原生支持；INT8 KV 需要专用 kernel
- 选择：H100 + 70B → FP8 weight + FP8 KV cache 是甜点；A100 → AWQ INT4 weight + INT8 KV
- 测：必须在业务评测集上跑量化模型与 FP16 对比，丢点 < 1% 才上线

**追问**

- 为什么 H100 的 FP8 比 A100 的 INT8 性能好？
- AWQ 量化模型能再 fine-tune 吗？

**评分要点**

- **及格**：能讲 PTQ vs QAT + 量化级别
- **良好**：能讲 AWQ vs GPTQ + KV 量化
- **优秀**：能讨论硬件 + 业务测试 + fine-tune 影响

---

### 26.5.14 推理服务故障排查："吞吐忽高忽低"

**问题**

线上 vLLM 服务 QPS 平均 50，但偶尔掉到 10 又恢复，30 分钟一次。监控只看到 GPU-Util 平均 80%。请给出**你的排查路径**。

**考察点**

- 是否能用结构化方法排查推理性能波动
- 是否能想到 prefix cache eviction / 长 prompt prefill / GC pause / 上游 burst
- 是否会查 engine 内部 metrics

**回答框架**

- 第一步：确认是"QPS 实际下降"还是"QPS 显示数据问题"（看上游网关 outbound）
- 第二步：vLLM metrics（prometheus）
  - `running_seqs` / `waiting_seqs`：waiting 突增 = 上游 burst
  - `swapped_seqs`：突增 = 显存压力
  - `prefix_cache_hit_rate`：突降 = 模板换了或 cache evict
  - `prefill_tokens` vs `decode_tokens`：长 prompt 涌入会让 prefill 占满
- 第三步：节点层（GPU 降频 / 邻居 Job / 网络抖）
- 第四步：客户端（prompt 长度分布 / 用户请求 burst）
- 30 min 周期性：考虑定时任务（健康检查 / 监控采集 / batch job）

**追问**

- prefix cache hit rate 从 90% 掉到 30%，最可能什么原因？
- swapped_seqs > 0 说明什么？

**评分要点**

- **及格**：能讲分层排查
- **良好**：能查 vLLM 内部 metrics
- **优秀**：能讨论周期性 root cause + cache 失效具体原因

---

## 26.6 Kubernetes、调度、队列、配额与平台化

### 26.6.1 GPU 在 K8s 上的资源模型

**问题**

K8s 标准 resource 是 CPU + memory，GPU 怎么纳入？请说明 **NVIDIA device plugin / GPU operator / MIG / time-slicing** 各自做什么、什么场景用，以及"一个 Pod 占 1 块 GPU"和"两个 Pod 共享一块 GPU"在工程上要解决哪些问题。

**考察点**

- 是否懂 K8s extended resource 概念
- 是否懂 device plugin 的工作机制
- 是否懂 GPU 共享的几种方式（time-slicing / MPS / MIG）

**回答框架**

- Device plugin：把"GPU"注册成 extended resource（如 `nvidia.com/gpu: 1`），Pod request 时 kube-scheduler 按节点容量调度
- GPU operator：自动部署 driver + container toolkit + device plugin + DCGM exporter 等组件
- MIG（Multi-Instance GPU）：A100/H100 硬件级切片（最多 7 份），强隔离，每片当独立 GPU 用
- Time-slicing：纯软件复用（device plugin 把 1 块 GPU 当 N 块上报），无隔离，互相影响
- MPS（Multi-Process Service）：进程间共享 SM，效率比 time-slicing 高，但需协作
- 共享要解决：显存隔离（time-slicing 没有 → 一个 Pod 写爆 OOM 全挂）、计算配额、优先级、failure 分离
- 推荐：开发 / 推理低负载用 MIG；推理高负载独占；time-slicing 仅限测试

**追问**

- A100 MIG 切到 1g.10gb 时还能跑 70B 吗？
- 一个 GPU operator 升级失败导致所有 GPU 节点掉线，怎么避免？

**评分要点**

- **及格**：能讲 device plugin + 共享 4 种方式
- **良好**：能讨论隔离差异
- **优秀**：能讨论 operator 升级风险 + 业务场景匹配

---

### 26.6.2 GPU 调度的 Topology-aware 与 Bin-packing

**问题**

集群有 100 个节点，每节点 8 张 H100。一个新 Job 要 32 卡 + NVLink 内全互联。请说明 **K8s 默认调度器为什么不够**，以及如何通过 topology-aware / gang scheduling / bin-packing 满足需求。

**考察点**

- 是否懂 K8s 默认调度器对 GPU 拓扑无感
- 是否懂 gang + topology + bin-pack 的协同
- 是否会用 Volcano / Kueue / scheduler plugin 实现

**回答框架**

- 默认 kube-scheduler：按节点资源容量调度，不知道 GPU 之间 NVLink/IB 拓扑
- 拓扑感知：需要 node label（topology zone / island）+ Pod affinity + 自定义 scheduler plugin
- Gang scheduling：32 卡要么全调度到 4 个相邻节点，要么 0 个；Volcano 的 PodGroup
- Bin-packing：避免把 32 卡 Job 分散到 100 节点 → 用 NodeSelector + 优先选填得最满的节点
- 实践：Volcano / Kueue + 自定义 scheduler-plugins / Topology-aware scheduling
- 验证：训练 Job 启动后跑 NCCL test 看 ring/tree 算法选择是否最优

**追问**

- 一个 Job 已经 gang scheduled 但开始训练后发现一张卡降频，怎么处理？
- 训练 Job 比推理 Job 优先吗？为什么不一定？

**评分要点**

- **及格**：能讲 gang + topology
- **良好**：能讨论 bin-pack + scheduler plugin
- **优秀**：能讨论运行时拓扑验证 + 优先级动态调整

---

### 26.6.3 K8s 上 stateful 推理服务怎么 deploy

**问题**

vLLM 推理服务有"状态"（KV cache、prefix cache）。请说明在 K8s 上 deploy 它**为什么不能简单用 Deployment + Service**，并给出推荐的 Workload 类型 + 流量管理方式。

**考察点**

- 是否懂 stateless vs stateful 工作负载在 K8s 上的差异
- 是否懂 readiness / liveness / startup probe 配置
- 是否懂 graceful shutdown 对推理服务的关键性

**回答框架**

- Deployment + Service 的问题：滚动更新会粗暴删 Pod，KV cache 立刻丢；新 Pod 冷启动慢
- 推荐：StatefulSet 或 Deployment + 控制好 surge / unavailable + PreStop hook 实现 graceful drain
- 探针：startup probe（模型加载阶段不杀）；readiness（drain 时摘流）；liveness（hang 时 kill）
- Graceful drain：PreStop hook → 摘流 → 等 in-flight 请求结束 → 退出
- 流量：Service + 多副本负载均衡；session affinity（按 user / conversation）能让 prefix cache 命中更稳
- 高级：KServe / Knative serverless 推理，但 GPU 冷启动让 serverless 不那么直接

**追问**

- session affinity 和负载均衡如何平衡？
- 模型加载 60 秒，K8s 怎么避免 startup probe 失败？

**评分要点**

- **及格**：能讲三种探针 + drain
- **良好**：能讨论 session affinity + cache
- **优秀**：能讨论 KServe / serverless 在 GPU 上的局限性

---

### 26.6.4 K8s 配额与多租户隔离

**问题**

公司 4 个业务团队共享一个 GPU 集群。请说明用 **Namespace + ResourceQuota + LimitRange + NetworkPolicy + PriorityClass** 怎么组合实现"配额 + 优先级 + 软隔离 + 硬隔离"。

**考察点**

- 是否能用 K8s 原生机制做多租户
- 是否懂 ResourceQuota 与 GPU 的搭配限制
- 是否懂软硬隔离差异

**回答框架**

- Namespace：每个团队一个，作为配额 + RBAC + 网络隔离的边界
- ResourceQuota：限制 namespace 总 GPU / CPU / memory（注意 ResourceQuota 不能限制"哪类 GPU"，需要扩展）
- LimitRange：单 Pod 资源 default + max，防止一个 Pod 把 quota 吃光
- PriorityClass：高 / 中 / 低优先级 + 可抢占，结合 preemption 实现重要 Job 优先
- NetworkPolicy：egress / ingress 规则，软隔离（K8s 原生）；硬隔离需要 SR-IOV / 节点池物理切分
- RBAC：team A 不能 list team B 的 Pod
- 监控：成本归因按 namespace + label

**追问**

- 4 团队 quota 总和 = 集群容量的 120%，超卖怎么管？
- 一个 namespace 里 100 个 Pod 把 etcd 压崩了，怎么办？

**评分要点**

- **及格**：能用 namespace + quota
- **良好**：能讨论 priority + 网络
- **优秀**：能讨论超卖 + etcd 容量 + 软硬隔离边界

---

### 26.6.5 GPU 节点怎么 maintain（升级 / 替换 / 故障）

**问题**

一台 8 GPU 节点的 NIC driver 要升级，节点要重启。集群上正在跑训练 + 推理。请说明**完整的维护流程**：怎么 cordon / drain / 重启 / 验证 / 上线，以及 stateful 工作负载（训练 / 推理 prefix cache）怎么处理。

**考察点**

- 是否懂 K8s 节点维护标准流程
- 是否懂训练 / 推理在节点维护时的不同处理
- 是否会做"先验证再上线"

**回答框架**

- 准备：通知业务（训练 Job owner / 推理服务 oncall）；选低峰窗口
- Cordon：节点不再接新 Pod
- Drain：drive 现有 Pod
  - 推理 Pod：通过 PreStop graceful drain，新流量切到其他副本
  - 训练 Pod：触发 checkpoint，scheduler 重新调度（gang restart 整个 Job 到其他节点）
- 重启 / 升级：driver / firmware / NIC / OS
- 验证：跑 GPU 自检（dcgmi diag）+ NCCL test + 内部 ML benchmark
- Uncordon：标记可调度，先小流量验证再放量
- 自动化：Cluster API + node update operator + automated validation

**追问**

- 一个训练 Job gang scheduled 在 32 卡上，要 drain 1 个节点怎么处理？
- 升级后跑 NCCL test 失败，怎么 rollback？

**评分要点**

- **及格**：能讲 cordon / drain / 重启
- **良好**：能讨论训练 vs 推理的不同处理
- **优秀**：能讨论自动化 + rollback + 验证级别

---

### 26.6.6 队列调度：Volcano / Kueue / Slurm / 自研

**问题**

新建训练平台需要 Job 队列。请对比 **Volcano、Kueue、Slurm、自研** 的取舍，给出在"K8s 原生 + 中等团队规模 + 需要 backfill / preemption / fairshare"约束下的选型建议。

**考察点**

- 是否对几个调度器的真实差异有判断
- 是否懂 K8s native vs HPC native（Slurm）的边界
- 是否能识别"自研"的真实成本

**回答框架**

- Volcano：CNCF 项目，K8s native，gang + queue + fairshare + preemption 都有，K8s AI 平台主流
- Kueue：K8s SIG 官方项目，更简洁，资源借用 + 多集群，但 backfill / preemption 仍在演进
- Slurm：HPC 标准，调度强大，但与 K8s 生态融合需要桥接（slurm-operator）；适合纯训练 HPC 集群
- 自研：除非有非常特殊需求（合规 / 内部工具栈），否则不建议
- 建议：K8s native + 中等团队 → Volcano 是甜点；如果未来要扩 multi-cluster 资源借用 → Kueue
- 风险：调度器升级影响所有 Job；调度策略调参需要持续运营

**追问**

- 训练 + 推理混部，调度器需要支持什么？
- Volcano gang + Kueue resource borrowing 能组合吗？

**评分要点**

- **及格**：能区分四者
- **良好**：能给出选型建议 + 理由
- **优秀**：能讨论混部 + 多集群 + 升级风险

---

### 26.6.7 Autoscaling for AI workloads：HPA / VPA / Cluster autoscaler 的局限

**问题**

K8s 标准 HPA 基于 CPU / memory 触发扩缩容。请说明**在 AI 推理场景为什么 HPA 不够**，以及实战中常用的扩缩容方案（KEDA / 自定义指标 / queue depth based / token throughput based）。

**考察点**

- 是否懂 HPA 在 AI 推理上的局限（GPU util 不直接反映负载）
- 是否懂 KEDA / 自定义 metrics 路径
- 是否懂 cluster autoscaler 与 GPU 节点的搭配（GPU 节点冷启动慢）

**回答框架**

- HPA 局限：CPU/memory 不能反映 GPU 推理压力；GPU util 也具有欺骗性（80% 不等于满载）
- 自定义指标：通过 Prometheus adapter 暴露 token throughput / queue depth / running seqs / p99 latency 让 HPA 用
- KEDA：基于事件 / 队列长度扩缩容；适合 batch / async 场景
- VPA：垂直扩缩（改 Pod requests），但 GPU 资源不能动态改
- Cluster autoscaler：触发节点增加，但 GPU 节点拉镜像 + 启动 + driver init 5-10 min；用 over-provisioning Pod 提前撑容量
- 实战组合：Pod 级 KEDA 扩副本 + 节点级 cluster-autoscaler 扩节点 + 关键服务保留 hot standby

**追问**

- 推理服务希望 60s 内扩容，但 GPU 节点冷启动 5 min，怎么办？
- queue depth 突涨能用 HPA 应对吗？

**评分要点**

- **及格**：能讲 HPA 的指标问题
- **良好**：能讲 KEDA + custom metrics
- **优秀**：能讨论冷启动 + over-provisioning + hot standby

---

### 26.6.8 GPU 碎片化：怎么避免 / 怎么治理

**问题**

集群跑 6 个月后，明明有 100 张空闲 GPU，但新提的 32 卡训练 Job 却起不来。请说明**GPU 碎片化在 K8s 上的根因**和**治理手段**（defrag / bin-packing / 抢占 / quota 限制小 Job）。

**考察点**

- 是否懂资源碎片化的工程模式
- 是否能识别"卡数够但拓扑不连"的真实问题
- 是否能给治理路径

**回答框架**

- 根因：小 Job 散落在多节点；每节点剩 1-2 卡；32 卡需要连续节点 / NVLink 内
- 检测：定期统计"max contiguous GPU available" vs total free
- 治理 1（短期）：bin-packing 调度 + node-fitting 算法（找最匹配的节点）
- 治理 2（中期）：抢占小 Job 让大 Job 通过（preemption 必须配合 checkpoint）
- 治理 3（长期）：quota 限制小 Job 数 + 周期性 defrag（migrate 小 Job 整理碎片）
- 治理 4：节点池物理切分（小 Job 池 / 大 Job 池），避免互相干扰
- 监控：每天 report"碎片度"指标（free GPU / max contiguous GPU）

**追问**

- 一个长跑训练 Job 占着 32 卡但只用 8 卡，怎么治理？
- defrag 主动迁移会引发什么副作用？

**评分要点**

- **及格**：能讲根因 + bin-pack
- **良好**：能讨论抢占 + quota
- **优秀**：能讨论 defrag + 节点池切分 + 监控指标

---

### 26.6.9 K8s 网络：Service mesh / 直连 / SR-IOV

**问题**

推理服务从 gateway 流量进 → engine 副本 → 模型 registry / vector DB。请说明**这条链路在 K8s 上常见的网络方案**（kube-proxy + iptables / IPVS / Cilium / service mesh / SR-IOV）以及对推理 latency 的实际影响。

**考察点**

- 是否懂 K8s service 网络栈对延迟的影响
- 是否懂 service mesh sidecar 的 trade-off
- 是否懂 GPU 节点 SR-IOV / 直连方案

**回答框架**

- Kube-proxy + iptables：默认，规则多时（>1000 services）连接建立慢
- IPVS：内核态 LB，规模大时优势明显
- Cilium / eBPF：bypass kube-proxy，连接建立 + 转发都更快；提供 L7 policy
- Service mesh（Istio / Linkerd）：sidecar 注入 mTLS + observability + traffic split；但每跳 +2-5ms latency；推理服务慎用
- SR-IOV：物理网卡虚拟化，给 Pod 直接用 VF；训练 NCCL 大流量场景有用
- 推理建议：gateway → engine 直连（不进 mesh）；mesh 仅在控制面；GPU 训练 NCCL 走 SR-IOV / GPUDirect RDMA

**追问**

- Service mesh sidecar 的 +5ms latency，在 inference 场景能接受吗？
- SR-IOV 的 VF 数量是怎么决定 Pod 密度的？

**评分要点**

- **及格**：能讲 kube-proxy + service mesh
- **良好**：能讨论 mesh 的延迟代价
- **优秀**：能讨论 SR-IOV + GPUDirect + 直连决策

---

### 26.6.10 镜像 / 模型分发的工程问题

**问题**

一个 70B 模型容器镜像 200GB（base + model weights），新副本启动要 5-10 min 拉镜像。请说明 K8s 上**镜像分发的常见优化**（多阶段构建 / 分层 / lazy load / pre-pull / OCI artifact 解耦）。

**考察点**

- 是否懂大镜像的真实痛点
- 是否懂 image pull 优化方案
- 是否懂 model 与 image 解耦的设计

**回答框架**

- 拆开：base image（runtime + python + libs）vs model weights → 镜像 5GB + 模型挂载（PVC / sidecar download）
- Lazy load：stargz / nydus 让镜像按需加载，启动时只拉 metadata
- Pre-pull：用 DaemonSet 在节点启动时预拉 base image
- 节点本地缓存：image pull 后保留，hot 节点再起 Pod 就秒级
- Model weights as artifact：S3 / OCI artifact 单独管理，启动时 init container 拉到 emptyDir / hostPath
- 跨 region：image registry 跨 region replication；模型用 CDN / edge cache
- 进阶：模型 mmap from PVC，启动跳过加载（fork shared memory）

**追问**

- 模型 mmap from PVC 在 OOM 时会怎样？
- pre-pull 占节点磁盘怎么治理？

**评分要点**

- **及格**：能讲拆 base / weights
- **良好**：能讨论 lazy load + pre-pull
- **优秀**：能讨论 mmap + 跨 region + 磁盘治理

---

### 26.6.11 平台化：从"能用 K8s"到"AI 工程师友好"

**问题**

新人 ML 工程师不该被 K8s 复杂度劝退。请说明你会**如何在 K8s 之上抽象出"训练 Job / 推理 Service / 评测 Run"等任务级 API**，以及为什么不能直接让用户写 YAML。

**考察点**

- 是否懂"用户体验 API"和"K8s 原生 API"的差异
- 是否能讲 CRD + operator 抽象路径
- 是否懂"灵活性 vs 易用性"取舍

**回答框架**

- 问题：用户写 K8s YAML 容易出错 + 不知道资源 / 调度 / 配额怎么填
- 解：CRD（CustomResourceDefinition）封装"训练 Job"为高级对象，operator 翻译成底层 PodGroup / Pod / Service / PVC
- 例子：`kind: TrainingJob` 字段是 model / dataset / GPU / NIC / checkpoint，operator 自动生成 200 行 YAML
- 用户接口：CLI / SDK / Web UI；每种都基于同一套 CRD
- 灵活性逃生：高级用户允许传 `extraSpec` 覆盖底层（escape hatch）
- 治理：CRD 强制必填字段（owner / project / cost-center），平台自动注入 quota / priority / network policy

**追问**

- 这个抽象 6 个月后用户开始抱怨"不够灵活"，怎么平衡？
- 你的 CRD schema 升级时怎么不破坏现有 Job？

**评分要点**

- **及格**：能讲 CRD + operator
- **良好**：能讨论用户接口分层 + escape hatch
- **优秀**：能讨论 schema 演进 + 用户抱怨平衡

---

### 26.6.12 K8s 调度新人最常踩的坑

**问题**

请列出**新人在 K8s 上跑 AI 工作负载最容易踩的 5 个坑**，以及给出"老手会怎么提前防"。

**考察点**

- 是否对 K8s + AI 工程坑有一手经验
- 是否能讲解决方案而非只列问题
- 是否能讲"为什么这个坑特别坑"

**回答框架**

- 坑 1：忘配 ImagePullPolicy → 用旧镜像；老手用 `IfNotPresent` + 镜像 tag 严格管理
- 坑 2：requests = limits 没设好 → 资源争抢 / OOMKilled；老手用 LimitRange + 实测
- 坑 3：忘 nodeSelector 限 GPU 节点 → Pod 调度到普通节点 GPU 资源不存在；老手 affinity 双重保险
- 坑 4：训练 Pod 没设 restartPolicy: OnFailure → 失败后再不起；老手用 Job 而非 Pod
- 坑 5：忘 PreStop hook → 推理服务被 SIGKILL；老手默认 30s grace + drain endpoint
- 坑 6（bonus）：configmap / secret 改了但 Pod 没重启；老手用 reloader / hash annotation

**追问**

- 选一个坑详细讲一次故障复盘？
- 怎么把这些"坑"做成 platform 默认值，让新人无法踩？

**评分要点**

- **及格**：能列 3 个坑
- **良好**：能给老手防御方案
- **优秀**：能讨论"如何把坑产品化为默认值"

---

## 26.7 可观测性、发布、安全、成本与多租户治理

### 26.7.1 AI 平台的"四个金信号"是什么

**问题**

Google SRE 提的 four golden signals（latency / traffic / errors / saturation）是给通用服务的。请说明 **AI 推理服务的"四金信号"应该是什么**，以及与传统服务有何不同。

**考察点**

- 能否把 AI 推理特有的指标抽象出来
- 能否区分"业务可见信号"和"系统健康信号"
- 能否讲分位数 / 分桶的重要性

**回答框架**

- TTFT / TPOT 分布（latency 不是单一数字）
- Throughput（tokens/s 而非 QPS，因为请求长短差异大）
- Errors（HTTP errors + content errors，模型输出无效 / safety filter 触发也是 error）
- Saturation（GPU memory util / KV cache util / queue depth；CPU/mem 不是关键）
- 与传统不同：
  - latency 必须分 TTFT+TPOT
  - traffic 必须按 token 而非 request
  - errors 必须含语义层
  - saturation 在 GPU 维度
- 还应有：cost/token、prefix cache hit rate、抢占/swap rate

**追问**

- 用 P50 还是 P99？
- 如何采样 trace 才能既覆盖长尾又不爆 storage？

**评分要点**

- **及格**：能给 AI 版四金信号
- **良好**：能讲与传统差异
- **优秀**：能讨论 trace sampling + 语义错误

---

### 26.7.2 训练任务的可观测：metric / log / trace 各管什么

**问题**

研究员提交的 70B 训练 Job 报"loss 不收敛"。请说明你的平台应该提供哪些**可观测能力**让研究员自己定位（不是次次找 oncall），覆盖 metric / log / trace / profile 四类。

**考察点**

- 是否懂训练任务可观测的差异（vs 推理）
- 是否能讲 step-level metric / loss / gradient 暴露
- 是否懂 profile（torch profiler / Nsight）的接入方式

**回答框架**

- Metric：每 N step 上报 loss / gradient norm / learning rate / step_time / GPU util / NCCL time / data_load_time
- Log：每个 rank 的 stdout/stderr 集中收集（loki / fluent-bit）；按 Job 检索；rank 0 不够，要 all-rank
- Trace：训练过程的事件（start, checkpoint, NaN, OOM, restart），时间线可看
- Profile：torch.profiler 的输出文件存到对象存储 + UI（TensorBoard / Perfetto）；用户能 self-serve
- 平台职责：把这四类标准化、零配置接入；研究员自己写代码不该额外集成监控
- "loss 不收敛"自助路径：先看 loss 曲线 / gradient norm → 数据 / 模型 / LR / numerical → trace 找 NaN / step

**追问**

- 上千张卡训练，每 rank 都收 metric 太多怎么办？
- 一个静态训练几小时后 hang，你期望平台能自动捕获什么？

**评分要点**

- **及格**：能讲 metric / log / trace
- **良好**：能讲 profile + 自助
- **优秀**：能讨论 metric 聚合 + hang 自动诊断

---

### 26.7.3 推理服务的发布：金丝雀 / 蓝绿 / 影子流量

**问题**

新版本模型要上线，可能有 regression。请说明 **金丝雀 / 蓝绿 / 影子流量** 三种发布模式在 AI 推理上的工程差异，以及为什么 AI 模型发布比一般服务更难做"自动判定"。

**考察点**

- 是否懂发布模式的语义
- 是否懂模型发布的特殊性（输出无标准答案）
- 是否能讲"质量门禁"如何自动化

**回答框架**

- 蓝绿：新旧版本各占一组副本，整批切换；快速 rollback；浪费资源
- 金丝雀：先 1% → 10% → 100% 渐进切；可观测期间发现问题 rollback；标准做法
- 影子流量：旧版正式服务 + 新版接收复制流量但不返回；纯压测 + 对比，无用户影响
- AI 难点：相同 prompt 旧 vs 新输出可能不同但都对；没有 ground truth 自动评判
- 自动判定：业务级 metric（用户点踩率 / 留存）+ 离线评测集 + safety filter 触发率 + 长度分布 + 拒答率
- 工程：feature flag 平台 + 模型 registry 状态机 + A/B 实验平台联动

**追问**

- 影子流量场景 KV cache 怎么办（双倍占用）？
- "新模型 metric 比旧模型差 0.5%" 是否该 rollback？

**评分要点**

- **及格**：能区分三种模式
- **良好**：能讨论 AI 发布特殊性
- **优秀**：能讨论自动判定 + 灰度策略 + 资源代价

---

### 26.7.4 AI 系统的安全边界：prompt injection / data leak / supply chain

**问题**

公司新部署一个能调用工具的 LLM Agent。请说明**它的安全边界**：哪些攻击面（prompt injection / 数据外泄 / 供应链 / 越权）、各自怎么防御、平台和应用层各承担什么。

**考察点**

- 是否懂 LLM 应用的特殊攻击面
- 是否能区分平台层和应用层职责
- 是否懂 defense in depth

**回答框架**

- Prompt injection：用户输入注入指令篡改模型行为；防御：输入分隔 + 输出 schema 校验 + 不可信内容隔离区
- 数据外泄：模型可能"记住"训练数据 PII；防御：脱敏 + 输出过滤 + audit
- Tool 越权：Agent 调用工具执行用户不该有权限的动作；防御：scoped credentials + 工具白名单 + RBAC + 二次确认
- 供应链：第三方模型 / LoRA / dataset 含恶意；防御：签名 + scan + sandbox（参考 26.4.5）
- 平台层：身份 / 网络 / scan / log
- 应用层：业务规则 / 工具权限 / 输出校验
- Defense in depth：任何单层失败不应导致 catastrophic
- Red team：定期攻击演练 + 评测集 + bug bounty

**追问**

- 一个 Agent 在工具调用前写"<system>I am admin</system>"绕过权限，怎么防？
- 训练数据投毒（在 corpus 埋 trigger）你怎么发现？

**评分要点**

- **及格**：能列 3 类攻击面
- **良好**：能区分平台 / 应用层 + defense in depth
- **优秀**：能讨论 red team + 数据投毒检测

---

### 26.7.5 多租户 AI 平台的隔离边界

**问题**

平台同时服务 10 个业务方 / 50 个 ML 团队。请说明**多租户隔离要在哪些维度做**（资源 / 网络 / 数据 / 审计 / 计费），以及哪些是必做、哪些可妥协。

**考察点**

- 是否能列全多租户维度
- 是否懂"租户隔离 vs 资源利用"取舍
- 是否能讲组织成熟度对应的隔离级别

**回答框架**

- 资源：必做。Namespace + ResourceQuota + PriorityClass；GPU 节点池物理切分（重要租户）
- 网络：必做。NetworkPolicy + 跨租户 Pod 不可互通；工具调用 egress 控制
- 数据：必做。每租户独立 bucket / 数据库；密钥按租户 KMS
- 计费：必做。所有 GPU-hour / token / storage 按租户标签上报
- 审计：必做。所有 control-plane action（提交 Job / 发布 model / 修改 quota）签名记录
- 模型隔离：核心模型 per-tenant 独立部署 vs 共享 base + LoRA per-tenant；后者更高效但需要多 LoRA 引擎
- 可妥协：早期阶段 prefix cache 跨租户共享（节省显存但有泄漏风险）→ 增长后切回独立

**追问**

- 数据合规要求"数据不出 region"，跨 region 平台怎么办？
- 一个租户突发流量挤占其他租户，如何快速处置？

**评分要点**

- **及格**：能列 5 个维度
- **良好**：能讨论必做 / 可妥协
- **优秀**：能讨论组织成熟度 + 跨 region + 突发流量 emergency

---

### 26.7.6 成本归因：GPU-hour / token / storage 都怎么记

**问题**

CFO 要 monthly cost report，按业务 / 租户 / 项目摊分。请设计**成本归因系统**：捕获哪些计量点、归因到什么粒度、怎么处理共享资源（base model / prefix cache / 平台基础设施）。

**考察点**

- 是否懂 AI 平台的多种计量维度
- 是否懂共享资源的摊分难题
- 是否能讲业务可解释性

**回答框架**

- 计量点：训练 GPU-hour（按 Pod 实际占用）；推理 input/output tokens（按 request 累加）；storage GB-month；网络出口；评测 GPU-hour
- 归因：每个 Pod / 每个 request 都带 owner label（team / project / cost center）；自动汇总
- 共享资源：base model storage 按使用租户数平均分；prefix cache 显存按 hit 比例分（细，复杂）；平台 control-plane 按总用量比例分
- 报表：按 owner 维度月度成本 + token / GPU-hour 占比
- 业务可解释：每个 owner 能下钻到具体 Job / request；成本异常报警
- 工具：基于 Prometheus + 数据仓库 + BI

**追问**

- 一个 Job 失败 5 次重试，用户愿意承担成本吗？
- prefix cache 共享让某团队"白嫖"另一团队，怎么处理？

**评分要点**

- **及格**：能列计量点 + 标签
- **良好**：能讨论共享资源摊分
- **优秀**：能讨论失败重试 + 跨团队公平 + 异常报警

---

### 26.7.7 模型发布的回滚机制

**问题**

新模型上线后 30 分钟，发现 safety filter 触发率从 0.5% 涨到 5%，可能是新模型对边界 case 处理变差。请说明**完整 rollback 流程**：从决策到流量切回 / 显存回收 / 通知 / 复盘。

**考察点**

- 是否懂 rollback 不只是"切流量"
- 是否懂 stateful 服务 rollback 的复杂度
- 是否能讲事后复盘机制

**回答框架**

- 决策：oncall 用 dashboard + 自动告警 + runbook 决定 rollback；优先级"业务影响 > 工程完美"
- 切流量：feature flag 切到旧版本副本（蓝绿则秒级，金丝雀则按比例反向）
- 显存：新版本副本不立刻删（保留 5-10 min 防再 rollback），观察期过后再 scale down
- 通知：业务方 / 同 oncall / 用户（status page）；事故等级标记
- 数据：保留新版本期间的 trace / log（30 min）便于复盘
- 复盘：5-Why；模型出现问题是数据 / 训练 / 评测 / 发布哪一环漏了
- 修复后：评测集补充该 case → 不让相同问题再发
- 自动化：把 safety filter 触发率作为发布门禁的硬指标

**追问**

- "新版本性能 +20% 但 safety 略差" 该 rollback 吗？
- 数据库 schema 跟着模型变了，rollback 怎么办？

**评分要点**

- **及格**：能讲切流量 + 通知
- **良好**：能讨论 stateful + 复盘
- **优秀**：能讨论性能 / safety 取舍 + schema migration

---

### 26.7.8 SLO / Error budget 在 AI 推理的应用

**问题**

平台给业务方承诺 SLA 99.9% availability + p99 latency < 1s。请说明 **error budget** 在 AI 推理上怎么定义、怎么消耗、用什么决策（要不要 rollback / 要不要冻结发布）。

**考察点**

- 是否懂 SLO / SLI / error budget 的概念
- 是否能把它应用到 AI 推理特殊性
- 是否能讲"用 budget 推动决策"

**回答框架**

- SLI：可用率 + latency（TTFT / TPOT）+ 内容质量（safety filter / refuse rate）
- SLO：99.9% availability + p99 TTFT < 500ms + safety < 1%
- Error budget：每月 0.1% 不可用 = 43 min 容忍；超过则发布冻结
- 消耗：每次告警 / 故障 / 失败请求扣减
- 决策：budget 充裕 → 可激进发布；budget 不足 → 冻结风险大的变更
- 工程：实时 budget dashboard + 自动冻结 / 解冻
- AI 特殊：SLI 不只是 HTTP success，还包括"safety/quality"维度，需要独立预算

**追问**

- 一个新模型把 latency 改善 20% 但 quality 下降，如何用 budget 决定上线？
- error budget 消耗速度突增，怎么 alert？

**评分要点**

- **及格**：能讲 SLO / budget
- **良好**：能讨论 AI 特殊 SLI
- **优秀**：能讨论自动化决策 + 多维 budget

---

### 26.7.9 配额超卖与紧急降级

**问题**

平台总 GPU 1000，分配给 5 个业务方 quota 总和 1200（超卖 20%）。某天峰值真的来了 1100 GPU 需求。请说明**紧急降级策略**：怎么决定谁让步、怎么执行、怎么通知。

**考察点**

- 是否懂超卖的工程逻辑（峰均比 < 1）
- 是否能讲紧急降级的优先级 / 自动化
- 是否能讲业务沟通

**回答框架**

- 超卖前提：业务峰值不重叠 + 历史峰均比 < 1
- 紧急降级触发：监控发现 utilization > 95% + 仍有 pending Jobs
- 优先级：在线推理 > 生产训练 > 探索 > spot；low-priority 立刻被 preempted
- 执行：自动 preempt 低优 Pod（提前 5 min 通知 + checkpoint friendly）
- 通知：业务 owner 自动消息（slack / 邮件） + 状态页
- 业务沟通：事先签 SLA 写明"紧急时被 preempt"；事后 review 看是否需要扩容 / 调 quota
- 防再发：监控 quota usage 趋势，提前预警
- 极端：业务大 owner 临时申请扩容，平台留紧急 buffer 池

**追问**

- 业务方说"我从来没被 preempt 过，所以 SLA 没写明"——怎么办？
- 超卖比例怎么定才安全？

**评分要点**

- **及格**：能讲优先级 + preempt
- **良好**：能讨论自动化 + 通知
- **优秀**：能讨论事先 SLA + buffer 池 + 比例决策

---

### 26.7.10 Trace / Log 的存储成本与采样

**问题**

每天 1 亿次推理请求，全量 trace 存 30 天 = PB 级。请设计**采样策略**：哪些必采、哪些采样、怎么平衡可观测能力与存储成本。

**考察点**

- 是否懂 trace 采样常见策略
- 是否能识别"长尾 / 错误"场景必须采全
- 是否懂 log 压缩 / 归档

**回答框架**

- 全量必采：errors / latency > p99 / safety filter 触发 / 用户反馈 thumbs-down → 100%
- 采样：正常请求 1% 随机；按租户分层（VIP 10%，普通 1%）
- 进阶：head-based sampling（trace 入口决定）+ tail-based sampling（结束后看延迟决定）
- Log：原始日志只保 7 天 hot；30 天 warm（gzip 压缩）；3-12 month cold（Glacier）
- Metric：分钟级保 30 天；小时级保 1 年；天级保多年
- 工程：OpenTelemetry collector 做采样；ELK / ClickHouse 存 log；Prometheus + Thanos 存 metric
- 节省：按 token 计采样而非按 request；同一 user 多次请求同 trace ID

**追问**

- 1% 采样下，怎么定位"某用户某时刻问题"？
- tail-based sampling 实现的瓶颈是什么？

**评分要点**

- **及格**：能讲 head sampling
- **良好**：能讨论 tail sampling + 必采集
- **优秀**：能讨论存储分层 + OTel collector + 工程瓶颈

---

### 26.7.11 灾难恢复（DR）：训练 / 推理 / 数据 各自策略

**问题**

主 region 的整个 AZ 失效，你的平台要在多久内恢复？请分别给出**训练 / 推理 / 数据存储**的 DR 策略和 RTO / RPO 目标。

**考察点**

- 是否懂 DR 概念（RTO / RPO）
- 是否能针对三种工作负载分别给方案
- 是否懂跨 region 复制的成本

**回答框架**

- 推理：跨 region 多活（active-active）；DNS / GSLB 故障切换；RTO 5min RPO 0
- 训练：单 region 主跑 + 跨 region checkpoint 备份；故障时另 region 重启 from latest checkpoint；RTO 1h RPO 30min（看 checkpoint 频率）
- 数据：S3 cross-region replication（自动）；模型 registry 多 region 同步；RTO 实时 RPO 几分钟
- 控制面：K8s control plane 跨 AZ HA；prometheus / grafana 高可用
- 演练：每季度 DR 演练（关一个 AZ）；发现问题修复
- 成本：active-active 推理双倍；training cross-region replication 网络费

**追问**

- 推理跨 region 多活时，prefix cache 怎么处理？
- 一个 70B checkpoint 跨 region 同步要多久？

**评分要点**

- **及格**：能给三类 RTO/RPO
- **良好**：能讨论 multi-active + replication
- **优秀**：能讨论演练 + cache 多 region + 成本

---

### 26.7.12 安全合规审计：哪些动作必须有审计日志

**问题**

合规审计要求"所有重要动作可追溯"。请列出 AI 平台**必须有审计日志的动作类别**，以及审计日志本身应该如何防篡改。

**考察点**

- 是否能列全审计对象（控制面 / 数据面 / 模型 / 数据）
- 是否懂审计日志防篡改（append-only / WORM / signing）
- 是否懂合规对应（SOC2 / ISO27001 / GDPR）

**回答框架**

- 控制面：用户登录 / 提交 Job / 发布 model / 修改 quota / RBAC 变更 / 删除资源
- 数据面：训练数据 / 用户输入 / 模型输出（按合规要求）
- 模型：上传 / 下载 / 部署 / 撤回 / 量化
- Secret：访问 / 修改 / 创建 token
- 防篡改：append-only log（不能改写）；WORM 存储（S3 Object Lock）；定期签名 hash chain
- 谁能查：分级 RBAC，敏感日志只有合规审计员能查
- 合规对应：SOC2 要求审计 + retention；GDPR 要求"删除我"操作可证明
- 工程：所有动作通过统一 control-plane 网关，自动写审计

**追问**

- 一个工程师为了调试线上 bug 临时给自己提了 admin 权限，审计应该 catch 到什么？
- 审计日志保留多久合理？

**评分要点**

- **及格**：能列审计对象
- **良好**：能讨论防篡改 + 分级
- **优秀**：能讨论合规对应 + 临时权限 + retention
