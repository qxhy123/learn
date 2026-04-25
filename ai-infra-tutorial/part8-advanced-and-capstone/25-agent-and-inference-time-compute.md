# 第25章：AI Agent 与推理时计算基础设施

> 模型进入 agent 阶段后，平台要管理的就不只是“一次推理”，而是一段带预算、带工具、带回退条件的计算过程。

> **关联章节**：本章把 [第13章](../part4-data-and-storage/13-feature-vector-and-cache.md) 的检索与缓存、[第14章](../part5-serving-infra/14-online-inference-architecture.md) 的在线链路、[第17章](../part5-serving-infra/17-multitenancy-and-cost.md) 的成本治理收束到同一个问题上：什么时候值得在推理时多花算力，怎样把这笔算力花得可控。

## 学习目标

完成本章学习后，你将能够：

1. 区分 agent、tool use 和 inference-time compute 的边界
2. 理解为什么“多想几步”会直接改变 serving 架构与成本模型
3. 设计一个最小可运营的 agent loop
4. 为 agent 系统设置预算、停止条件和回退策略
5. 判断什么时候值得把更多算力放到推理时，而不是训练时

---

## 正文内容

### 25.1 从基础设施视角看 Agent 和 Thinking Model

如果你已经理解了 Ch 14-17 的推理系统设计，那么 Agent 和 Thinking Model 会从三个方向打破你的假设：会话持续时间、token 消耗可预测性、并发模型。

| 被打破的假设 | 传统在线推理的近似 | Agent / Thinking Model 下的新现实 | 基础设施含义 |
|--------------|--------------------|-----------------------------------|--------------|
| 会话持续时间 | 请求通常在一次生成内结束 | 一个任务可能持续数十秒到数分钟，中间多次挂起、恢复、调用工具 | 需要 session state、step trace、显式关闭和空闲回收 |
| token 消耗可预测性 | 输出长度大致可由 `max_tokens` 控制 | thinking、候选采样、验证和上下文回注会让 token 与 GPU 时间同时波动 | 需要推理预算、动态降级和按步骤记账 |
| 并发模型 | 容量主要看 QPS、batch size、单请求延迟 | 容量取决于并发 session、每个 session 的推理调用次数和工具等待时间 | 调度器要同时管理 GPU 队列、工具池和长上下文缓存 |

因此，本章不重复介绍 Agent 的概念，而是关注它进入服务化系统后带来的运行时治理问题：怎样把一段多步、不确定、会调用外部环境的计算过程，映射回可以调度、计费、隔离和观测的基础设施对象。

### 25.2 从 QPS 到 Session 的并发模型

传统推理系统常把压力近似成：

```text
QPS x request latency
```

但 agent 更接近：

```text
concurrent sessions x session duration x model calls per session
```

因为一个用户任务可能持续数十秒到数分钟，并在中途多次挂起、恢复、调用工具和继续思考。平台如果还只按短请求 QPS 估容量，就会低估会话态、KV 生命周期和工具执行池的占用。

### 25.3 推理时计算不只是 thinking tokens

训练时算力是为了把能力写进参数；推理时算力是为了在具体问题上多做搜索、采样、验证、复核或外部动作。`thinking tokens` 只是其中一种表现形式，更通用的定义是：任何在推理阶段为了提高任务成功率而主动消耗额外计算的技术。

一个粗略表达可以写成：

$$
t_{\text{answer}} \approx \sum_{i=1}^{N_{\text{steps}}} \left(t_{\text{model}, i} + t_{\text{search}, i} + t_{\text{tool}, i} + t_{\text{verify}, i}\right)
$$

对应的成本也会随步骤数一起上升：

$$
\text{cost per answer} \approx \sum_{i=1}^{N_{\text{steps}}} \left(\text{token unit cost}_i \cdot (\text{input tokens}_i + \text{reasoning tokens}_i + \text{output tokens}_i) + \text{gpu-second cost}_i + \text{tool cost}_i\right)
$$

这意味着 agent 系统的单位成本，通常不再与“一次生成多少 token”线性对应，而与“走了多少步、扩展了多少候选、调用了多少工具、做了几次复核”一起决定。更稳妥的做法，是把 token、GPU-second、工具和 session 占用分开记账，再在控制面汇总成最终货币成本。

| 推理时计算模式 | 额外 token 量级 | GPU 时间波动范围 | 对调度器的影响 |
|----------------|-----------------|------------------|----------------|
| Chain-of-Thought / thinking tokens | 常见为最终输出的 1-10 倍，难题可能更高 | 通常随生成长度线性增加，长尾来自无法提前预测的思考长度 | 需要 `max_thinking_tokens`、流式预算扣减和超预算截断 |
| Best-of-N sampling / rerank | 约为单候选的 `N` 倍，再加 rerank prompt | 近似 `N` 倍，若并行采样会造成瞬时 GPU 峰值 | 调度器要限制候选数、并发分支和单租户 burst |
| Self-consistency | 多条 reasoning path，常见 3-20 条 | 可并行也可串行，尾延迟受最慢 path 或投票策略影响 | 需要把一次用户请求拆成多个可取消的子请求 |
| Tree search / beam search | 随分支因子和深度增长，可能指数级膨胀 | 波动最大，受剪枝、终止条件和 verifier 频率影响 | 需要 step-level budget、优先级队列和中间状态回收 |
| Verifier-guided generation | 生成候选 + verifier prompt / 小模型评分 | 每轮增加一次或多次评分，失败重试会放大尾延迟 | 需要把生成队列和验证队列分开限流，避免 verifier 反压主模型 |
| Tool-augmented reasoning | 模型 token 不一定最多，但会产生结果回注和继续生成 | GPU 时间被工具等待打断，恢复时可能重新 prefill 长上下文 | 调度器要处理挂起 session、工具超时和恢复后的 KV 复用 |

这些方法的共同点是：都在把“更多推理阶段计算”换成“更高任务成功率”。差别在于，有些主要消耗 token，有些主要消耗 GPU 时间，有些主要制造调度不可预测性。

### 25.4 成本模型与推理预算管理

agent 系统的成本模型不能只看平均 tokens。一个可运营的成本面通常要同时支持三类计费口径：

| 计费模型 | 适用场景 | 优点 | 风险 |
|----------|----------|------|------|
| input + output token | API 型模型服务、简单 chat、可解释账单 | 易理解，和传统 serving 兼容 | 难表达 hidden thinking、搜索分支和 GPU 空转 |
| GPU-second | 自建推理集群、高成本 thinking / search 任务 | 贴近真实资源消耗，适合容量治理 | 需要准确归因到请求、租户和 session |
| session | 长任务 agent、企业套餐、异步工作流 | 对用户更稳定，方便设置任务级 SLA | 如果内部预算缺失，平台可能承担长尾成本 |

工程上应把预算管理放在控制面，而不是只写在 prompt 里。一次请求进入 agent runtime 时，控制面先生成预算 envelope，并在每一步扣减：

| 预算字段 | 工程实现 | 超预算时的降级策略 |
|----------|----------|--------------------|
| `max_thinking_tokens` | 解码器或网关按 hidden / reasoning token 流式计数，接近阈值时触发 stop sequence 或强制 final answer | 截断 thinking，要求模型基于当前中间状态给出最短可用答案 |
| `per_request_gpu_second_budget` | scheduler 记录 prefill、decode、verifier、rerank 的 GPU 时间，按 request / session / tenant 归因 | 停止扩展新分支，切到小模型 summarizer 或返回 partial result |
| `max_model_calls_per_session` | agent runtime 对 planner、executor、verifier 的模型调用做 step counter | 跳过下一轮反思，直接进入 finalizer 或人工接管 |
| `max_tool_wall_time` | tool runner 对每次调用设置 timeout、取消令牌和幂等 retry key | 返回工具不可用的可解释失败，禁止无限重试 |
| `tenant_budget_remaining` | billing / quota 服务在每步前做 reservation，完成后 settle | 降级到低预算策略、切小模型、拒绝非关键工具调用 |

预算耗尽不应该表现成随机断流。更好的顺序是：先停止新增搜索分支，再截断 thinking，然后切小模型压缩上下文，最后返回 partial result 或进入人工接管。这样用户得到的是可解释的降级结果，而不是一次不可复现的超时。

### 25.5 Agent 基础设施与推理服务的集成

Agent session 不是 vLLM / TRT-LLM 里的一个超长请求。更常见的映射是：

```text
agent session
  -> model call 1: plan
  -> tool call / retrieval
  -> model call 2: observe + continue
  -> verifier call
  -> model call 3: final answer
```

也就是说，长 session = 多次推理调用 + context 管理。推理服务仍然处理一批批 prefill / decode 请求，但 agent runtime 要负责把 session state、工具结果、摘要记忆和预算状态拼回下一次模型调用。

| 集成点 | Agent runtime 负责 | vLLM / TRT-LLM 负责 | 关键风险 |
|--------|--------------------|---------------------|----------|
| 请求拆分 | 把一次任务拆成 planner / executor / verifier / finalizer 多次调用 | 对每次调用做 batching、prefill、decode、streaming | 拆分过细会增加 prefill 成本和调度开销 |
| Context 管理 | 决定保留原文、摘要、工具结果还是结构化状态 | 承载本次请求的 prompt 和 KV | 回注过多会让上下文线性膨胀 |
| Prefix caching | 标记可复用的 system prompt、工具说明、历史前缀 | 复用相同前缀的 KV / prefix cache | 前缀失配或 TTL 过短会让 cache 命中率下降 |
| KV Cache 生命周期 | 按 session close、idle timeout、priority 管理缓存保留 | 分配、驱逐和复用 KV block | 长 session 占住 KV 会挤压短请求吞吐 |
| Tool calling 执行环境 | 在沙箱中执行工具，设置超时、权限、网络和文件边界，并把结果回注给下一轮模型 | 不直接执行工具，只接收回注后的 prompt | 工具结果未经过滤会扩大提示注入和数据泄露风险 |

这也是它与 Ch 15 KV Cache 的直接关系：长 context 不只是 prompt 变长，还会让 KV Cache 变成 session 级资源。平台需要 prefix caching 降低重复 prefill，需要 KV Cache 生命周期管理避免长会话挤占共享池，还需要在工具结果回注前做大小限制和安全过滤。

### 25.6 长会话状态、上下文压缩与流式中间结果

agent 系统里，context window 不再只是“一段越来越长的聊天记录”，而是一份要持续被治理的运行时状态。

| 问题 | 平台要回答什么 | 常见做法 |
|------|----------------|----------|
| 会话状态保留多久 | KV / memory 是秒级、分钟级还是任务级 | 分层 TTL、显式 session close、空闲回收 |
| 上下文何时截断 | 哪些历史必须保留，哪些可以丢弃 | 基于窗口上限做 truncation |
| 上下文何时压缩 | 历史太长时是否转摘要或结构化记忆 | summarize / distill / state extraction |
| 中间结果如何返回 | 是只回最终答案，还是流式返回步骤进度 | streaming token + step event + final bundle |

如果没有这层治理，agent 很容易同时出现三类问题：上下文无限变长、KV 生命周期失控、前端和监控只能看到最后一句答案却看不到中间失败。

### 25.7 一个最小 agent loop

一个最小但可运营的 agent loop，通常至少包含：

```text
user request
  -> planner
  -> tool / retrieval / executor
  -> verifier
  -> stop or continue
  -> final answer
```

这里最关键的不是组件名，而是每一步都要可解释：

| 环节 | 主要职责 | 平台关注点 |
|------|----------|------------|
| Planner | 决定先做什么 | 是否有最大步数与任务边界 |
| Executor | 调工具、跑检索、写中间结果 | 超时、权限、幂等性 |
| Verifier | 判断结果是否可接受 | 是否会无限循环、是否能给出失败理由 |
| Finalizer | 组织最终输出 | 是否保留审计轨迹与引用来源 |

没有 verifier 的 agent，常常会演变成“会调用工具的无限循环”。

### 25.8 Planner、Executor、Verifier 为什么要分开

把三者混在一个 prompt 里当然可以跑，但平台很难治理。

| 角色 | 如果职责不清会怎样 | 分开后的平台收益 |
|------|--------------------|------------------|
| Planner | 一边规划一边执行，步骤不可审计 | 易限制最大步数与工具白名单 |
| Executor | 工具调用和答案生成混在一起 | 易做超时、重试、幂等与权限控制 |
| Verifier | 失败时继续盲试 | 易设置 stop condition 与人工接管 |

这并不意味着一定要三模型三服务。重点是：运行时语义上要能区分这三类动作。

### 25.9 Tool use 与 retrieval 会怎样改写 serving

一旦引入工具和检索，在线推理链路会从“单模型调用”变成“多跳依赖图”。

| 新增对象 | 平台含义 | 对应章节 |
|----------|----------|----------|
| Retrieval | 召回、缓存、权限过滤进入主链路 | [第13章](../part4-data-and-storage/13-feature-vector-and-cache.md) |
| Tool API | 外部服务变成推理步骤的一部分 | [第14章](../part5-serving-infra/14-online-inference-architecture.md) |
| Budget policy | 预算和租户规则直接影响能走几步 | [第17章](../part5-serving-infra/17-multitenancy-and-cost.md) |
| Trace / audit | 需要记录每一步用了什么信息和工具 | [第21章](../part7-reliability-security/21-observability-and-capacity.md) |

所以 agent 不是给模型多接几个 API，而是把“工作流执行”放进了推理面。

### 25.10 Tool use 的信任边界必须先于优化存在

很多团队做 agent 时，先补预算、trace 和 retry，却把真正危险的边界留到最后。这个顺序是反的。对工具调用型 agent 来说，最低限度的控制至少包括：

| 控制项 | 平台要回答什么 | 失效后会怎样 |
|--------|----------------|--------------|
| 工具白名单 | 哪类任务可调用哪些工具 | 模型可能调用高风险或无关工具 |
| 隔离执行环境 | 代码、命令、浏览器是否跑在沙箱内 | 单次任务故障会污染宿主或其他会话 |
| Scoped credentials | 每步拿到的 token 是否只够当前动作使用 | 工具泄露会扩大成租户级事故 |
| Egress 限制 | 能访问哪些网络、文件、设备 | 过度联网或读写会绕过业务边界 |
| Approval gate | 改状态动作是否需要显式批准 | 模型可能在无确认下执行破坏性操作 |

所以更稳妥的设计顺序应是：先定义 trust boundary，再谈多步推理优化。一个“会算预算的 agent”如果权限仍是全开的，本质上仍然不可运营。

### 25.11 预算、停止条件与回退策略

agent 系统最容易失控的地方，不是模型不会思考，而是它会一直思考。

| 控制项 | 常见上限 | 为什么重要 |
|--------|----------|------------|
| 最大步骤数 | 例如 4-8 步 | 防止无限循环和尾延迟失控 |
| 最大工具调用数 | 例如 2-5 次 | 防止外部依赖账单失控 |
| 最大 token 预算 | 输入 + 推理 + 输出合并计 | 防止长上下文任务吞掉共享池 |
| 最大 wall-clock 时间 | 例如 5-20 秒 | 防止单请求拖垮高优流量 |
| 回退路径 | 超预算后切单次回答或人工接管 | 把失败做成可预期行为 |

平台更稳妥的做法，是把这些预算写进控制面，而不是埋在 prompt 文本里。

### 25.12 什么时候 inference-time compute 真值得

不是所有任务都值得把更多算力放在推理时。

| 任务类型 | 更可能值得 | 原因 |
|----------|------------|------|
| 数学、代码、规划 | 是 | 额外搜索和验证能显著提升成功率 |
| 实时问答 / 检索增强 | 视情况 | tool use 通常有收益，但步数不宜太多 |
| 低价值、高 QPS 文本生成 | 否 | 成本放大快于质量收益 |
| 强 SLA 在线客服 | 通常谨慎 | 多步链路容易放大尾延迟 |

一个实用判断是：如果多花 2-3 倍推理成本，不能带来明显更高的任务成功率或更少的人类返工，就不值得。

### 25.13 成本与 SLA 怎样一起治理

agent 系统的治理重点不是“平均成本”，而是把高价值请求和低价值请求区分开。

| 治理动作 | 目标 | 常见做法 |
|----------|------|----------|
| 分级服务 | 把高价值任务允许更多步骤 | 关键租户走高预算策略，普通租户走快路径 |
| 两段式回答 | 先给快速草答，再决定是否继续搜索 | 先满足交互感知，再异步补强 |
| Step-level timeout | 控制单步卡死风险 | 每次工具调用与 verifier 都有独立超时 |
| Budget-aware routing | 把复杂任务送到更贵但更强的策略 | 结合租户、任务类型、剩余预算决定 |

这和传统 serving 的差别在于：治理对象从“请求”扩展成了“请求中的一串步骤”。

### 25.14 工程建议

- 先定义任务成功率，再决定是否引入更多 inference-time compute；不要只因为“模型能想更久”就默认开启
- Agent loop 必须有最大步数、最大预算和明确回退路径
- Tool use、retrieval 和 verifier 都要进入 trace，不要只记录最终答案
- 有副作用的工具必须放在白名单、沙箱、scoped credential 和审批门之后
- 对高价值任务和高 QPS 任务使用不同策略，不要让所有请求都走最重路径
- 把 agent 成本拆成模型 cost、工具 cost、人工接管 cost 三部分，才能真正做经营决策

#### 本章涉及的常见工具

| 概念 | 常见工具 / 命令 | 备注 |
|------|-----------------|------|
| Agent 编排 | LangGraph、OpenAI Agents SDK、AutoGen | 适合组织 planner / executor / verifier 流程 |
| 检索与工具接入 | LangChain、LlamaIndex | 适合把 retrieval 与工具调用接入 agent loop |
| 在线观测 | OpenTelemetry、Langfuse、Arize Phoenix | 适合记录 step trace、工具调用和失败原因 |
| 压测与评测 | GenAI-Perf、自定义 task eval harness | 要同时看成功率、步骤数、成本与延迟 |

---

## 本章小结

| 主题 | 关键点 |
|------|--------|
| Agent 本质 | 一段带规划、执行、验证和停止条件的推理流程 |
| Inference-time compute | 用更多推理时算力换更高任务成功率 |
| 平台难点 | 多步状态、预算控制、工具依赖、成本放大 |
| 是否值得 | 取决于任务价值、成功率提升和 SLA 约束 |
| 治理重点 | 把预算、回退和 trace 做成控制面能力 |

---

## 练习题

1. 从基础设施视角看，Agent 和 Thinking Model 分别会怎样改变会话持续时间、token 消耗可预测性和并发模型？
2. 给定 1,000 个并发 agent session、平均每个 session 6 次模型调用、每次调用平均 1.5 秒 GPU 时间，估算总 GPU-second 需求，并说明还缺哪些容量假设。
3. 为什么 agent 容量规划不能只用 `QPS x request latency`？请和传统在线推理调度做对比。
4. Chain-of-Thought、Best-of-N sampling、Tree search / beam search、Verifier-guided generation 分别会把成本放大到哪里？
5. 如果一个请求允许 `max_thinking_tokens=8,000`，但租户剩余预算只够 4,000 个 reasoning tokens，你会如何设计截断和降级策略？
6. 请为一个“数学求解 agent”设计最大步数、最大工具调用数、`per_request_gpu_second_budget` 和回退策略。
7. 为什么 verifier-guided generation 需要把生成队列和验证队列分开限流？
8. Agent session 如何映射到 vLLM / TRT-LLM 请求？为什么长 session 不应该直接等同于一个超长推理请求？
9. Tool calling 沙箱至少应该限制哪些资源？请说明超时、权限、网络和文件边界各自防止什么风险。
10. 多 Agent 并发运行时，怎样做租户级资源隔离，避免一个高预算 session 挤占其他短请求？
11. 长 context 与 Ch 15 的 KV Cache 有什么关系？prefix caching 和 KV 生命周期管理分别解决什么问题？
12. 如果一个 agent 系统成功率更高，但单位成本翻了 4 倍，你会如何判断它是否应该上线？
13. 设计一个 Agent 成本预测面板，至少包含 token、GPU-second、工具调用、session 时长和失败重试成本。
14. Agent 执行失败后，哪些步骤适合自动重试，哪些步骤应该直接返回 partial result 或转人工？为什么？
