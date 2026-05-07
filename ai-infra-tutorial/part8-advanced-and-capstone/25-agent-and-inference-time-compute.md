# 第25章：AI Agent 与推理时计算基础设施

> 模型进入 agent 阶段后，平台要管理的就不只是“一次推理”，而是一段带预算、带工具、带回退条件的计算过程。

> **关联章节**：本章把 [第13章](../part4-data-and-storage/13-feature-vector-and-cache.md) 的检索与缓存、[第14章](../part5-serving-infra/14-online-inference-architecture.md) 的在线链路、[第17章](../part5-serving-infra/17-multitenancy-and-cost.md) 的成本治理收束到同一个问题上：什么时候值得在推理时多花算力。

## 1. 第一性原理拆解 + 学习大纲

### 拆 — 不可化简的问题

把 Agent、tool use、thinking model、inference-time compute 这些名字都拿掉，本章只剩一个不可化简的问题：**一次用户任务不再等于一次模型前向，而是一段会持续消耗资源、改变外部状态、并在不确定路径上寻找可接受结果的计算过程；平台怎样让这段过程可调度、可计费、可隔离、可观测、可停止？**

传统在线推理的基础抽象是 request。request 有输入 token、输出 token、队列等待、prefill、decode、streaming、结束。它的容量估算可以粗略落在 `QPS x latency` 或 `concurrency x tokens/s` 上，成本也可以近似按 input/output token 计价。但 agent 任务的真实单位更接近 session：用户提出一个目标，系统先规划，再调用检索、工具或代码执行环境，拿到观察结果后继续推理，必要时生成多个候选、做验证、重试、压缩上下文，最后才给出答案。这个过程可能只用 2 次模型调用，也可能走到 8 步、20 次工具请求、几十秒 wall-clock 时间。基础设施的难点不是“模型能不能调用工具”，而是每一步都可能占用 GPU、KV Cache、工具池、网络连接、credential、trace 存储和租户预算。

因此，本章的第一性问题不是“如何写一个 agent prompt”，而是“如何把开放式计算过程重新变成工程系统能治理的对象”。如果没有预算，agent 会把一次难题扩展成无上限搜索；如果没有停止条件，失败会表现成循环、超时或账单失控；如果没有信任边界，工具调用会把提示注入、越权访问和副作用带进生产链路；如果没有 step trace，平台只能看到最后失败，却无法知道预算花在 planning、retrieval、verifier 还是重试上。Agent 基础设施的核心，是把“多想几步”翻译成明确的资源合同：最多多少 reasoning tokens、多少 GPU-second、多少模型调用、多少工具 wall time、哪些工具可用、何时降级、如何结算。

### 推 — 从这个问题如何推导出每个机制

从“开放式计算必须可治理”出发，几个机制几乎是被迫出现的。首先，request 抽象不够用，所以要引入 session、step 和 state。session 表示一次用户目标的生命周期，step 表示可计费、可观测、可取消的最小运行单元，state 表示上下文、工具结果、摘要、KV 复用线索和预算余额。没有这三层，平台无法回答“这次任务现在卡在哪里”“还能不能继续”“继续会花多少钱”。

其次，推理时计算不能只理解为 thinking tokens。额外算力可能花在模型内部的隐式推理，也可能花在多候选采样、树搜索、verifier 复核、retrieval、代码解释器、浏览器和外部 API。它们共同目标都是用更多 inference-time compute 换更高任务成功率，但资源形态不同：token 型压力主要放大 decode；分支型压力放大并发与峰值 GPU；工具型压力放大 wall-clock、恢复 prefill 和外部依赖；验证型压力则引入额外队列和反压。所以平台必须把推理时计算泛化成 4 类工程模式，而不是只暴露一个 `max_thinking_tokens`。

第三，成本模型从 token 计费扩展到预算 envelope。一次 agent session 进入运行时前，控制面要根据租户、任务类型、SLA、风险等级生成预算：总 token、reasoning token、GPU-second、模型调用次数、工具调用次数、工具 wall time、session TTL、最大步骤数。每一步开始前做 reservation，执行中流式扣减，结束后 settle，失败时记录原因。预算耗尽也不能随机断流，而要按“停止扩展分支 -> 降低候选数 -> 截断隐式推理 -> 压缩上下文或切小模型 -> 返回 partial result / 人工接管”的顺序降级。

第四，Agent runtime 和推理服务必须分工。vLLM、TRT-LLM、SGLang 这类 serving engine 负责单次模型调用的 batching、prefill、decode、KV block、prefix cache；Agent runtime 负责任务拆分、工具执行、状态管理、预算扣减、安全边界和 step trace。把整个 session 塞进一个超长请求会破坏调度效率，也会让工具等待期间 GPU 无法高效复用。更合理的映射是：长 session 被拆成多次短模型调用，中间由 runtime 挂起、恢复、压缩、回注观察结果，并把可复用前缀交给推理服务命中 cache。

第五，工具调用要求先定义信任边界，再谈优化。检索、浏览器、代码执行、数据库、工单系统和写操作 API 都不只是“模型能力增强”，它们会把外部世界接入推理循环。平台必须明确工具白名单、参数 schema、沙箱、scoped credentials、egress 限制、审批门、幂等 retry key 和审计日志。否则，一个预算控制完美但权限全开的 agent，仍然不可运营。

### 绘 — 因果链路

```mermaid
mindmap
  root((Agent 与推理时计算))
    不可化简的问题
      用户任务不是一次前向
      路径不确定
      资源必须可治理
      工具会改变外部状态
    运行时抽象
      Session
        生命周期
        TTL
        状态恢复
      Step
        可计费
        可取消
        可观测
      State
        上下文
        工具结果
        摘要记忆
        预算余额
    推理时计算四模式
      隐式推理预算
      多候选与搜索
      验证与反思
      工具增强执行
    基础设施机制
      预算 envelope
      Step trace
      Prefix cache
      KV 生命周期
      Tool sandbox
      Queue isolation
    工程结果
      成本可控
      SLA 可解释
      失败可回放
      租户可隔离
```

### 导 — 读完本章你应该能回答

1. 为什么 agent 的容量规划不能只用 `QPS x request latency`，而要转向 session、step、模型调用次数和工具等待时间？
2. `thinking tokens`、多候选采样、verifier、tool use 都属于 inference-time compute，但它们分别把压力施加到哪些资源上？
3. 一个生产级 agent runtime 至少需要哪些预算字段？这些预算应该在 prompt、网关、scheduler 还是 billing/quota 服务里执行？
4. 为什么长 session 不应该直接映射成 serving engine 里的一个超长请求？Agent runtime 与 vLLM / TRT-LLM / SGLang 的边界应如何划分？
5. 工具调用型 agent 的 trust boundary 应该先定义哪些内容？为什么 scoped credential、sandbox 和 approval gate 比 prompt 约束更可靠？
6. 当预算耗尽或 verifier 连续失败时，怎样设计有序降级，而不是让用户看到随机超时、断流或无限循环？
7. 如何判断一个任务是否值得多花 2-5 倍推理时计算：看 token 成本、GPU-second、人工返工率、成功率还是 SLA？

## 2. 学习目标

完成本章学习后，你将能够：

1. 区分 agent、tool use 和 inference-time compute 的边界
2. 理解为什么“多想几步”会直接改变 serving 架构与成本模型
3. 设计一个最小可运营的 agent loop
4. 为 agent 系统设置预算、停止条件和回退策略
5. 判断什么时候值得把更多算力放到推理时，而不是训练时

---

## 3. 正文内容

### 25.1 从基础设施视角看 Agent 和 Thinking Model

如果你已经理解了 Ch 14-17 的推理系统设计，那么 Agent 和 Thinking Model 会从三个方向打破你的假设：会话持续时间、token 消耗可预测性、并发模型。

| 被打破的假设 | 传统在线推理的近似 | Agent / Thinking Model 下的新现实 | 基础设施含义 |
|--------------|--------------------|-----------------------------------|--------------|
| 会话持续时间 | 请求通常在一次生成内结束 | 一个任务可能持续数十秒到数分钟，中间多次挂起、恢复、调用工具 | 需要 session state、step trace、显式关闭和空闲回收 |
| token 消耗可预测性 | 输出长度大致可由 `max_tokens` 控制 | 隐式推理、候选采样、验证和上下文回注会让 token 与 GPU 时间同时波动 | 需要预算 envelope、动态降级和按步骤记账 |
| 并发模型 | 容量主要看 QPS、batch size、单请求延迟 | 容量取决于并发 session、每个 session 的模型调用次数和工具等待时间 | 调度器要同时管理 GPU 队列、工具池和长上下文缓存 |

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

一个更贴近容量规划的拆法是：

```text
active_gpu_work = concurrent_sessions
                x avg_model_calls_per_session
                x avg_gpu_seconds_per_call
                / avg_session_wall_time

tool_pool_concurrency = concurrent_sessions
                      x avg_tool_calls_per_session
                      x avg_tool_wall_time
                      / avg_session_wall_time
```

这个公式故意把 GPU 工作和工具工作拆开。工具等待期间 GPU 不应被 session 独占；但 state、预算、trace 和权限 token 仍然存在。因此 runtime 必须支持挂起和恢复。

**工程边界**：session 并发不是 GPU 并发。一个 10,000 并发 session 的系统，如果每个 session 平均只有 15% 时间在模型 decode，其 GPU 活跃并发可能远低于 10,000；但 metadata store、trace pipeline、tool worker 和 KV/prefix cache 的压力可能先到瓶颈。

### 25.3 推理时计算的 4 种工程模式

训练时算力是为了把能力写进参数；推理时算力是为了在具体问题上多做搜索、采样、验证、复核或外部动作。`thinking tokens` 只是其中一种表现形式。更通用的定义是：任何在推理阶段为了提高任务成功率而主动消耗额外计算的技术。

一个粗略表达可以写成：

$$
t_{\text{answer}} \approx \sum_{i=1}^{N_{\text{steps}}} \left(t_{\text{model}, i} + t_{\text{search}, i} + t_{\text{tool}, i} + t_{\text{verify}, i}\right)
$$

agent 的单位成本通常不再与“一次生成多少 token”线性对应，而与“走了多少步、扩展多少候选、调用多少工具、复核几次”一起决定。工程上可归为 4 种模式：

| 模式 | 典型技术 | 主要收益 | 主要成本 | 调度影响 | 预算控制 |
|------|----------|----------|----------|----------|----------|
| 隐式推理预算 | Chain-of-Thought、thinking tokens、deliberation | 难题上提高单路径质量 | reasoning tokens、decode 时间、长尾输出 | 单请求 decode 变长，batch 内尾部拖慢 | `max_reasoning_tokens`、按 token 流式扣减、接近阈值触发 finalizer |
| 多候选与搜索 | Best-of-N、self-consistency、beam/tree search | 用多路径提高命中率，适合数学、代码、规划 | 候选数近似 `N` 倍，树搜索可能随深度膨胀 | 并行分支制造 GPU burst，串行分支制造尾延迟 | `max_candidates`、`max_branch_depth`、分支优先级、早停 |
| 验证与反思 | verifier-guided generation、critic、rerank、unit test | 把“生成答案”改成“生成 + 判断” | verifier 模型调用、测试执行、失败重试 | 生成队列和验证队列互相反压 | `max_verify_rounds`、独立 verifier 队列、失败阈值 |
| 工具增强执行 | Retrieval、browser、code interpreter、DB/API call | 接入外部事实、执行动作、降低模型记忆压力 | tool wall time、回注 token、权限与审计成本 | GPU 工作被工具等待打断，恢复时可能重新 prefill | `max_tool_calls`、`max_tool_wall_time`、工具白名单、结果大小上限 |

4 种模式可以组合。例如代码修复 agent 可能先规划，再采样 3 个 patch，运行测试作为 verifier，最后调用版本控制工具。但组合越多，成本乘法越明显。平台应把每种额外计算的预算独立化。

**工程边界**：thinking tokens 可由模型或 API 暴露，也可能是隐藏计费项；Best-of-N 能并行加速但会抬高瞬时 GPU 峰值；verifier 如果和主模型共用队列，可能在高失败率时反压所有请求；工具增强执行的最大风险往往不是 token，而是权限、超时和不可幂等副作用。

### 25.4 推理预算工程实现

agent 成本模型不能只看平均 tokens，通常要同时支持三类计费口径：

| 计费模型 | 适用场景 | 优点 | 风险 |
|----------|----------|------|------|
| input + output + reasoning token | API 型模型服务、简单 chat、thinking model | 易理解，和传统 serving 兼容 | 难表达工具等待、搜索分支和 GPU 空转 |
| GPU-second | 自建推理集群、高成本 search / verifier 任务 | 贴近真实资源消耗，适合容量治理 | 需要准确归因到 request、tenant 和 session |
| session / task | 长任务 agent、企业套餐、异步工作流 | 对用户更稳定，方便设置任务级 SLA | 如果内部预算缺失，平台可能承担长尾成本 |

工程上应把预算管理放在控制面，而不是只写在 prompt 里。一次请求进入 agent runtime 时，控制面先生成预算 envelope，并在每一步扣减：

```text
1. classify: 识别 tenant、任务类型、风险等级、SLA tier
2. quote: 生成预算 envelope，返回预估成本和可用策略
3. reserve: 每个 step 开始前预占 token / GPU-second / tool quota
4. execute: runtime 调用模型或工具，流式上报消耗
5. settle: step 结束后按真实消耗结算，多退少补
6. decide: 根据剩余预算、verifier 结果和 stop condition 决定继续或结束
7. degrade: 预算不足时按策略降级，记录可解释原因
```

| 预算字段 | 执行位置 | 工程实现 | 超预算时的降级策略 |
|----------|----------|----------|--------------------|
| `max_reasoning_tokens` | 模型网关 / decoder | hidden 或 visible reasoning token 流式计数，接近阈值触发 stop sequence 或 finalizer | 截断隐式推理，要求基于当前中间状态给出最短可用答案 |
| `max_output_tokens` | 模型网关 | 与传统 serving 一致，限制最终可见输出 | 改用摘要式输出，附上“不完整”状态 |
| `per_step_gpu_second_budget` | scheduler | 记录 prefill、decode、verifier、rerank 的 GPU 时间，按 step 归因 | 停止扩展新分支，切小模型 summarizer |
| `max_model_calls_per_session` | agent runtime | 对 planner、executor、verifier、finalizer 调用做 step counter | 跳过下一轮反思，直接 final answer 或人工接管 |
| `max_tool_wall_time` | tool runner | 每次调用设置 timeout、取消令牌和幂等 retry key | 返回工具不可用的可解释失败，禁止无限重试 |
| `max_context_tokens` | agent runtime / gateway | 控制历史、工具结果、摘要和检索片段总量 | 触发 summarization、state extraction 或丢弃低价值观察 |
| `tenant_budget_remaining` | billing / quota 服务 | 每步前 reservation，完成后 settle，失败要释放未用额度 | 降级到低预算策略、切小模型、拒绝非关键工具调用 |

预算耗尽不应表现成随机断流。推荐降级顺序是：停止新增搜索分支 -> 降低候选数和 verifier 轮数 -> 截断隐式推理 -> 压缩上下文或切小模型 -> partial result / 人工接管。

**工程边界**：预算系统不能依赖模型“自觉遵守”。prompt 可以提示模型节省步骤，但真正的上限必须由 gateway、scheduler、tool runner 和 quota 服务强制执行。跨服务 reservation 要处理并发扣减和失败释放，否则高并发 agent 会把租户预算打成负数或制造虚假拒绝。

### 25.5 Agent / 推理服务集成

Agent session 不是 vLLM / TRT-LLM / SGLang 里的一个超长请求。更常见的映射是：

```text
agent session
  -> model call 1: plan
  -> tool call / retrieval
  -> model call 2: observe + continue
  -> verifier call
  -> model call 3: final answer
```

也就是说，长 session = 多次推理调用 + context 管理。推理服务仍然处理一批批 prefill / decode 请求，但 agent runtime 要负责把 session state、工具结果、摘要记忆和预算状态拼回下一次模型调用。

| 集成点 | Agent runtime 负责 | 推理服务负责 | 关键风险 |
|--------|--------------------|--------------|----------|
| 请求拆分 | 把一次任务拆成 planner / executor / verifier / finalizer 多次调用 | 对每次调用做 batching、prefill、decode、streaming | 拆分过细会增加 prefill 成本和调度开销 |
| Context 管理 | 决定保留原文、摘要、工具结果还是结构化状态 | 承载本次请求的 prompt 和 KV | 回注过多会让上下文线性膨胀 |
| Prefix caching | 标记可复用的 system prompt、工具说明、策略前缀 | 复用相同前缀的 KV / prefix cache | 前缀失配会让 cache 命中率下降 |
| KV Cache 生命周期 | 按 session close、idle timeout、priority 管理缓存保留意图 | 分配、驱逐和复用 KV block | 长 session 占住 KV 会挤压短请求吞吐 |
| Queue isolation | 给 planner、finalizer、verifier、summarizer 标注优先级 | 在不同队列或优先级中调度 decode | verifier 风暴可能拖慢高优用户请求 |
| Tool calling 执行环境 | 在沙箱中执行工具，设置超时、权限、网络和文件边界，并把结果回注给下一轮模型 | 不直接执行工具，只接收回注后的 prompt | 工具结果未过滤会扩大提示注入和数据泄露风险 |

这也是它与 [第15章](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md) KV Cache 的直接关系：长 context 不只是 prompt 变长，还会让 KV Cache 变成 session 级资源。平台需要 prefix caching 降低重复 prefill，需要 KV 生命周期管理避免长会话挤占共享池，还需要在工具结果回注前做大小限制和安全过滤。

**工程边界**：不要让推理服务理解所有业务工具，也不要让 agent runtime 绕过推理服务直接占 GPU。前者会让 serving engine 被业务语义污染，后者会绕开 batching、KV 管理和配额。清晰边界是：runtime 管流程和状态，serving engine 管模型执行，quota/billing 管预算，tool runner 管外部动作。

### 25.6 长会话状态、上下文压缩与流式中间结果

agent 系统里，context window 不再只是“一段越来越长的聊天记录”，而是一份要持续被治理的运行时状态。

| 问题 | 平台要回答什么 | 常见做法 |
|------|----------------|----------|
| 会话状态保留多久 | KV / memory 是秒级、分钟级还是任务级 | 分层 TTL、显式 session close、空闲回收 |
| 上下文何时截断 | 哪些历史必须保留，哪些可以丢弃 | 基于窗口上限做 truncation |
| 上下文何时压缩 | 历史太长时是否转摘要或结构化记忆 | summarize / distill / state extraction |
| 中间结果如何返回 | 是只回最终答案，还是流式返回步骤进度 | streaming token + step event + final bundle |

如果没有这层治理，agent 很容易同时出现三类问题：上下文无限变长、KV 生命周期失控、前端和监控只能看到最后一句答案却看不到中间失败。

**工程边界**：摘要不是无损压缩。审计、复现、计费依赖原始 step trace；下一轮模型调用可以只用摘要或结构化状态。

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

每一步都要可解释：

| 环节 | 主要职责 | 平台关注点 |
|------|----------|------------|
| Planner | 决定先做什么 | 是否有最大步数与任务边界 |
| Executor | 调工具、跑检索、写中间结果 | 超时、权限、幂等性 |
| Verifier | 判断结果是否可接受 | 是否会无限循环、是否能给出失败理由 |
| Finalizer | 组织最终输出 | 是否保留审计轨迹与引用来源 |

### 25.8 Planner、Executor、Verifier 为什么要分开

把三者混在一个 prompt 里当然可以跑，但平台很难治理。

| 角色 | 如果职责不清会怎样 | 分开后的平台收益 |
|------|--------------------|------------------|
| Planner | 一边规划一边执行，步骤不可审计 | 易限制最大步数与工具白名单 |
| Executor | 工具调用和答案生成混在一起 | 易做超时、重试、幂等与权限控制 |
| Verifier | 失败时继续盲试 | 易设置 stop condition 与人工接管 |

这不意味着一定要三模型三服务。重点是运行时语义上要能区分三类动作。

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

所以更稳妥的设计顺序应是：先定义 trust boundary，再谈多步推理优化。

**工程边界**：只读工具和写工具必须分级。所有有副作用的工具都应该具备审批、幂等 key、回滚方案和审计事件。

### 25.11 预算、停止条件与回退策略

agent 系统最容易失控的地方，不是模型不会思考，而是它会一直思考。

| 控制项 | 常见上限 | 为什么重要 |
|--------|----------|------------|
| 最大步骤数 | 例如 4-8 步 | 防止无限循环和尾延迟失控 |
| 最大工具调用数 | 例如 2-5 次 | 防止外部依赖账单失控 |
| 最大 token 预算 | 输入 + reasoning + 输出合并计 | 防止长上下文任务吞掉共享池 |
| 最大 wall-clock 时间 | 例如 5-20 秒 | 防止单请求拖垮高优流量 |
| 回退路径 | 超预算后切单次回答或人工接管 | 把失败做成可预期行为 |

这些上限应由控制面执行；reservation、settlement 与降级流程见 §25.4。

### 25.12 什么时候 inference-time compute 真值得

不是所有任务都值得把更多算力放在推理时。

| 任务类型 | 更可能值得 | 原因 |
|----------|------------|------|
| 数学、代码、规划 | 是 | 额外搜索和验证能显著提升成功率 |
| 实时问答 / 检索增强 | 视情况 | tool use 通常有收益，但步数不宜太多 |
| 低价值、高 QPS 文本生成 | 否 | 成本放大快于质量收益 |
| 强 SLA 在线客服 | 通常谨慎 | 多步链路容易放大尾延迟 |

实用判断：如果多花 2-3 倍推理成本，不能带来明显更高成功率或更少人工返工，就不值得。

### 25.13 成本与 SLA 怎样一起治理

agent 系统的治理重点不是“平均成本”，而是把高价值请求和低价值请求区分开。

| 治理动作 | 目标 | 常见做法 |
|----------|------|----------|
| 分级服务 | 把高价值任务允许更多步骤 | 关键租户走高预算策略，普通租户走快路径 |
| 两段式回答 | 先给快速草答，再决定是否继续搜索 | 先满足交互感知，再异步补强 |
| Step-level timeout | 控制单步卡死风险 | 每次工具调用与 verifier 都有独立超时 |
| Budget-aware routing | 把复杂任务送到更贵但更强的策略 | 结合租户、任务类型、剩余预算决定 |

这和传统 serving 的差别在于：治理对象从“请求”扩展成了“请求中的一串步骤”。

**工程边界**：SLA 不应只写最终延迟。agent 还需要定义 first token latency、step event 间隔、最终完成时间和异步补偿时间。

### 25.14 工程建议

- 先定义任务成功率，再决定是否引入更多 inference-time compute
- Agent loop 必须有最大步数、最大预算和明确回退路径
- Tool use、retrieval 和 verifier 都要进入 trace，不要只记录最终答案
- 有副作用的工具必须放在白名单、沙箱、scoped credential 和审批门之后
- 对高价值任务和高 QPS 任务使用不同策略，不要让所有请求都走最重路径
- 把 agent 成本拆成模型 cost、工具 cost、人工接管 cost 三部分，才能真正做经营决策

### 25.15 Agent 架构模式对比

Agent 架构不是"prompt 写法"的差异，而是任务分解方式、控制流位置和状态传递策略的根本差异。把架构选错，后续无论怎么调预算或 trace 都难以弥补。工程上有 4 种主流模式，各有适用边界。

#### ReAct：Thought → Action → Observation 循环

ReAct（Reason + Act）把推理和行动交织在单模型单上下文中：每一步先生成 Thought（分析当前状态），再生成 Action（调用工具或进行计算），最后把工具返回值作为 Observation 追加到 context，下一轮继续。这种架构实现最简单，适合步骤间依赖性强、工具调用数量少（通常 3-8 步）且上下文可以线性积累的任务。

**工程缺点**：全部状态都在同一个 context 里滚动，context 越来越长；工具失败时没有结构化回退路径，只能依赖模型"下一步自己意识到失败"；多 step 错误在 context 中累积后模型容易迷失方向。适合对话式问答、简单的 retrieval + reasoning 场景。

#### Plan-Execute：先规划后执行，含 Reflexion 自我批评

Plan-Execute 将任务分两阶段：先让模型一次性输出完整计划（步骤列表或 DAG），再按计划逐步执行。Reflexion 在此基础上引入第三个环节：执行结束后，让模型对结果写 self-critique，把批评结果写入长期记忆，作为下一轮 plan 的输入。

**工程优点**：计划是结构化的，平台可以在执行前做安全审查（工具白名单、步骤上限、副作用检查）；执行步骤可以并行化；失败时可以回退到 plan 而不是重跑所有 step。**工程缺点**：初始 plan 质量决定上限，规划失败的回代价很高；Reflexion 引入额外模型调用，成本较 ReAct 高 30-60%。适合代码修复、研究报告、工单处理等有明确阶段划分的任务。

#### Multi-Agent：hierarchical / debate / swarm

Multi-Agent 不是"更多 AI"，而是通过分工降低单个 agent 上下文复杂度。有三种形态：

- **Hierarchical**：Orchestrator agent 接任务，分发给多个 sub-agent，每个 sub-agent 只关注自己的子问题，结果回汇给 orchestrator 做综合。适合任务可分解且子任务间耦合低的场景（如批量文档分析）。
- **Debate**：多个 agent 对同一问题各自产出答案，通过多轮辩论或投票收敛到更可信结果。适合高不确定性或需要多视角核查的任务（如法律条款解读、医疗辅助诊断）。
- **Swarm**：去中心化多 agent，每个 agent 只感知局部状态，通过规则或 emergent behavior 协作。适合探索性任务，实现最复杂，生产落地最谨慎。

**工程挑战**：multi-agent 系统的状态管理、消息路由、资源隔离和 trace 复杂度均成倍上升；单个 sub-agent 的失败会通过消息传播影响其他 agent；成本审计需要跨 agent 归因。

#### 框架映射

| 框架 | 主要模式 | 核心机制 | 适合场景 |
|------|----------|----------|----------|
| LangGraph | Plan-Execute + ReAct 混合 | 有状态图（节点 = step，边 = 条件跳转），支持持久化检查点 | 复杂多步、需要人工审批或中断恢复的 agent |
| AutoGen | Multi-Agent（hierarchical / debate） | GroupChat + ConversableAgent，消息驱动异步协作 | 多角色协作、代码生成与验证 |
| CrewAI | Multi-Agent（hierarchical，role-based） | Crew + Agent + Task，内置角色和工具注册 | 内容生产、营销自动化、结构化工作流 |
| OpenAI Swarm | Multi-Agent（去中心化 handoff） | 轻量 handoff 机制，agent 间直接传递 context | 教学/实验场景，生产需二次加固 |

#### 决策矩阵

| 场景 | 推荐架构 | 原因 |
|------|----------|------|
| 用户自然语言问答 + 工具调用 | ReAct | 步骤少、可线性积累，实现成本低 |
| 代码生成 + 测试验证 | Plan-Execute + Reflexion | 计划可审查，失败有回退，自我批评改善迭代质量 |
| 批量文档摘要 | Hierarchical Multi-Agent | 文档间独立，sub-agent 并行处理后聚合 |
| 法律或医疗高风险决策 | Debate Multi-Agent | 多视角降低单模型幻觉风险 |
| 长时研究任务（> 10 分钟） | Plan-Execute + LangGraph 检查点 | 支持中断恢复、人工审批门 |
| 高 QPS 简单任务（< 3 步） | 单模型 + function calling | 架构越重成本越高，简单任务不需要 orchestrator |

```mermaid
flowchart LR
  subgraph ReAct
    RT[Thought] --> RA[Action] --> RO[Observation] --> RT
  end
  subgraph PlanExecute["Plan-Execute + Reflexion"]
    PE_P[Plan] --> PE_E[Execute] --> PE_V[Verify] --> PE_R[Reflect]
    PE_R -->|写入长期记忆| PE_P
  end
  subgraph Hierarchical
    HO[Orchestrator] --> HS1[Sub-Agent 1]
    HO --> HS2[Sub-Agent 2]
    HO --> HS3[Sub-Agent 3]
    HS1 --> HR[聚合结果]
    HS2 --> HR
    HS3 --> HR
  end
  subgraph Debate
    DA[Agent A] --> DC{投票/评分}
    DB[Agent B] --> DC
    DC -->|收敛| DF[最终答案]
  end
```

> **工程建议**：先用 ReAct 跑通原型，再根据失败模式判断是否需要更复杂架构。架构升级的代价是成本、延迟和调试复杂度全部上升，升级决策应建立在实测失败率数据上，而不是预期。

> **工程边界**：Multi-Agent 系统的 trace 链路会横跨多个 agent，确保每个 agent_id 都被记入同一个 root trace_id，否则后续成本归因和失败定位将极为困难。

---

### 25.16 Agent Observability 与 Trace Schema

没有 trace 的 agent 是黑盒。一次 agent session 失败后，如果只能看到"最终答案是错的"，平台无法区分是规划阶段出错、工具返回脏数据、verifier 判断有误还是预算不足被截断。本节定义一套生产可用的 trace schema 并讨论 observability 平台选型。

#### Trace Schema 字段规范

每个 step 应产生一条 trace 记录，字段如下：

| 字段 | 类型 | 说明 |
|------|------|------|
| `trace_id` | string (UUID) | 整个 session 的唯一标识，从 root request 传递到所有子步骤 |
| `step_id` | string | 本步骤唯一 ID，格式建议 `{trace_id}_{step_index}` |
| `parent_id` | string \| null | 父步骤 ID，顶层 step 为 null；构成树形结构 |
| `agent_id` | string | 执行本步骤的 agent 标识（multi-agent 场景中标识不同角色） |
| `step_type` | enum | plan / execute / tool_call / verify / reflect / finalize |
| `tool_name` | string \| null | 被调用工具名，非工具步骤为 null |
| `tool_args` | object \| null | 工具入参，需做 PII 脱敏再存储 |
| `tool_result` | object \| null | 工具返回值，大型结果应截断并存摘要 |
| `tokens_in` | int | 本步骤模型输入 token 数 |
| `tokens_out` | int | 本步骤模型输出 token 数 |
| `reasoning_tokens` | int | thinking/reasoning 阶段消耗的 token（如有） |
| `latency_ms` | int | 本步骤 wall-clock 时间（毫秒） |
| `cost_usd` | float | 本步骤估算成本（美元），便于 session 级汇总 |
| `outcome` | enum | success / tool_error / timeout / budget_exceeded / halted |
| `error_class` | string \| null | 失败类型，如 `ToolTimeoutError` / `SqlSyntaxError` / `PermissionDenied` |
| `model_id` | string | 调用的模型版本，便于多模型路由场景归因 |
| `tenant_id` | string | 租户标识，用于成本归因和访问控制 |
| `timestamp` | ISO8601 | 步骤开始时间 |

> **PII 脱敏**：`tool_args` 和 `tool_result` 在写入 trace store 前，必须过一遍 PII 检测管道，对手机号、邮件、身份证号、密码字段做掩码或 hash。不要把原始 SQL 查询结果（可能含用户数据行）直接存入 trace。

> **Trace 大小控制**：工具返回值超过 10 KB 时，存储截断版本（前 500 字符 + size_bytes 元信息），完整数据只写 primary 存储（如 S3）并记录引用 URL。

#### OpenTelemetry 与 gen_ai.* namespace

OpenTelemetry Semantic Conventions 定义了 `gen_ai.*` 命名空间用于 LLM 可观测性：

- `gen_ai.system`：模型提供商（openai / anthropic / bedrock）
- `gen_ai.request.model`：请求模型 ID
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`：token 消耗
- `gen_ai.agent.id`：agent 标识（社区扩展提案）
- `gen_ai.tool.name` / `gen_ai.tool.call.id`：工具调用标识

Agent span 的父子关系应映射到 OTel span parent：同一 session 的所有 step span 挂在同一个 root trace 下，跨 agent 调用通过 `traceparent` HTTP header 传递 trace context。

#### 主流可观测性平台

| 平台 | 定位 | 优势 | 局限 |
|------|------|------|------|
| Langfuse | 开源 LLM 可观测性 | 原生支持 trace / span / observation 三层、cost 归因、eval 评分 | 社区版功能完整，需自托管 |
| LangSmith | LangChain 官方平台 | 深度集成 LangChain / LangGraph，trace 结构无缝 | 商业付费，与 LangChain 强耦合 |
| Weights & Biases Traces | MLOps 平台扩展 | 与训练实验、模型版本一体化管理 | trace 深度略逊于专业 LLM 平台 |
| Helicone | 轻量 proxy + 可观测性 | 无代码接入，proxy 层自动截获所有 OpenAI 调用 | 深度 agent trace 需额外 SDK |
| Arize Phoenix | 开源 ML 可观测性 | 强大的 embedding drift 和 eval 框架，支持 span 查询 | agent trace 支持相对较新 |

#### Agent Eval Framework

| Benchmark | 测试重点 | 评分方式 | 适用性 |
|-----------|----------|----------|--------|
| GAIA | 通用 agent 任务（搜索、计算、多跳推理） | 人工标注答案精确匹配 | 衡量通用 agent 能力上限 |
| SWE-bench（含 Verified subset） | GitHub issue 修复（真实代码库） | 单元测试通过率 | 衡量代码 agent 实用性，Verified subset 减少 false positive |
| AgentBench | 8 类 agent 任务（OS / DB / web / game） | 任务完成率 | 多维度评估 agent 适应性 |
| τ-bench | 工具调用可靠性（真实 API 场景） | 工具调用成功率 + 参数正确率 | 聚焦工具使用质量 |
| WebArena | 浏览器 web 操作 | 任务完成率 | 衡量 web agent 实用性 |
| OSWorld | 桌面 OS 操作 | 截图验证完成状态 | 最接近真实用户操作场景 |

```mermaid
flowchart TD
  R[Root Span: session_id] --> S1[Step 1: plan<br/>tokens_in=512 latency=340ms]
  R --> S2[Step 2: tool_call=text_to_sql<br/>tokens_in=1024 latency=89ms]
  R --> S3[Step 3: tool_call=execute_sql<br/>latency=1200ms outcome=success]
  R --> S4[Step 4: finalize<br/>tokens_out=280 cost=$0.0042]
  S2 -. PII脱敏 .-> TS[(Trace Store)]
  S3 -. 截断result .-> TS
  TS --> OPS[Langfuse / Arize Phoenix]
  OPS --> EVAL[Eval / Cost Dashboard]
```

> **Eval 闭环**：生产 trace 是最好的 eval 数据集。把真实失败 trace（outcome=tool_error 或 budget_exceeded）自动采样入 eval 集，用 judge model 对结果打分，形成"生产 → trace → eval → 改进"的闭环，比单纯跑 benchmark 更能反映业务实际质量。

---

### 25.17 Tool Sandbox 工程实现

工具调用是 agent 触碰真实世界的地方，也是最高风险的地方。"在沙箱里执行"不是一个开关，而是一组需要逐层设计的隔离机制。

#### 隔离方案矩阵

| 方案 | 隔离强度 | 启动延迟 | 适用场景 | 主要风险 |
|------|----------|----------|----------|----------|
| subprocess（直接） | 无 | <5ms | 开发调试 | 完全不隔离，不可用于生产 |
| Docker container | 中 | 100-500ms（已有镜像） | 通用代码执行 | 容器逃逸风险，需配合 seccomp |
| gVisor（runsc） | 高 | 150-600ms | 安全要求高的代码执行 | 部分系统调用不兼容，性能约 30% 损失 |
| Firecracker microVM | 极高 | 100-200ms（提前预热） | 金融、医疗等高合规场景 | 需 KVM，配置复杂度高 |
| Wasmtime / WASM | 高（沙箱语言级） | <10ms | 轻量函数执行、插件系统 | 语言支持有限，不能跑任意 Python |
| E2B（云服务） | 高（托管） | 300-1000ms | 快速落地不想自建 | 外部依赖，成本随并发线性增长 |
| Modal（云服务） | 高（托管） | 200-800ms | GPU 密集型工具执行 | 同上 |

#### seccomp Profile

生产环境的代码执行容器应限制以下系统调用（Docker `--security-opt seccomp=policy.json`）：

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "syscalls": [
    {
      "names": ["read","write","open","close","stat","fstat","lstat","poll",
                "mmap","mprotect","munmap","brk","rt_sigaction","rt_sigprocmask",
                "ioctl","pread64","pwrite64","readv","writev","access","pipe",
                "select","sched_yield","mremap","msync","dup","dup2","getpid",
                "socket","connect","accept","sendto","recvfrom","sendmsg","recvmsg",
                "shutdown","getsockname","getpeername","socketpair","setsockopt",
                "getsockopt","clone","fork","vfork","execve","exit","wait4",
                "getcwd","chdir","rename","mkdir","rmdir","unlink","readlink",
                "chmod","chown","getuid","getgid","getgroups","setuid","setgid",
                "utime","futex","nanosleep","clock_gettime","exit_group","epoll_ctl",
                "epoll_wait","openat","newfstatat","readlinkat"],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

**明确禁止**：`ptrace`（进程追踪）、`kexec_load`（内核替换）、`create_module`（内核模块）、`mount`（文件系统挂载）、`pivot_root`（根切换）、`syslog`（内核日志）。

#### Network Isolation

- **禁止所有外网**：适合纯计算类工具（数学、数据处理），Docker `--network none`
- **白名单出站**：只允许访问预定义的内部服务（数据库、向量库），通过 iptables/ebpf 规则实现
- **完全隔离 + 代理**：所有出站请求必须经过 egress proxy，proxy 做域名白名单、流量审计和速率限制

#### File System Overlay

```text
/
├── base/        # 只读基础镜像层（包含 Python 运行时、依赖库）
├── workspace/   # tmpfs，执行期间的工作目录，容器退出后自动清理
│   └── code/    # 用户代码写入这里
└── output/      # tmpfs，执行结果输出（大小上限 50MB）
```

用户代码不能读写 `base/` 以外的路径。通过 Linux bind mount + overlayfs 实现：base 层只读，workspace 层是 tmpfs（内存），执行完后整个 tmpfs 被丢弃，不留磁盘痕迹。

#### 超时信号链

```text
SIGTERM  →  等待 30s  →  SIGKILL  →  等待 5s  →  强制清理容器/VM
```

实现要点：发送 SIGTERM 后等待进程优雅退出（给正在写磁盘的进程机会 flush）；如果 30s 内未退出，发送 SIGKILL；SIGKILL 5s 后还未退出则强制销毁容器，清理 tmpfs 和网络 namespace。整个超时链必须有独立的看门狗进程，不能让被执行代码自己管理超时。

#### 完整 Dockerfile 示例

```dockerfile
FROM python:3.12-slim AS base

# 安装基础依赖，不装 curl/wget/git 等网络工具
RUN pip install --no-cache-dir numpy pandas scipy scikit-learn \
    && rm -rf /root/.cache

# 创建非特权用户
RUN useradd -m -u 1000 -s /bin/bash sandbox

FROM base AS runtime
USER sandbox
WORKDIR /workspace

# 只读绑定 base 层依赖
# workspace 在运行时通过 tmpfs 挂载
ENTRYPOINT ["python", "-u", "/runner/execute.py"]
```

运行命令（配合 seccomp + 网络隔离）：

```bash
docker run \
  --rm \
  --user 1000 \
  --network none \
  --memory 512m \
  --cpus 1 \
  --read-only \
  --tmpfs /workspace:size=100m,mode=1777 \
  --tmpfs /tmp:size=50m \
  --security-opt seccomp=/etc/docker/seccomp/code-exec.json \
  --security-opt no-new-privileges \
  --pids-limit 64 \
  code-sandbox:latest
```

> **工程建议**：E2B 和 Modal 等托管沙箱服务可以显著降低自建复杂度，适合早期产品。当执行量超过 10 万次/天或合规要求不允许数据出境时，再考虑自建 Firecracker 或 gVisor 方案。

> **工程边界**：沙箱超时不能只靠 Python 的 `signal.alarm`，被执行代码可以捕获 SIGALRM 或调用 `signal.signal(signal.SIGALRM, signal.SIG_IGN)` 绕过。超时必须由容器外层的 orchestrator 发送 SIGKILL 来强制终止。

```mermaid
flowchart LR
  AR[Agent Runtime] -->|提交代码 + 超时| TR[Tool Runner]
  TR -->|docker run --network none --memory 512m| SB[Sandbox Container]
  SB --> CE[Code Execution<br/>非特权用户 1000]
  CE --> OF[Output tmpfs]
  OF -->|读取结果| TR
  TR -->|SIGTERM 30s → SIGKILL 5s| SB
  TR --> QS[Quota: 记录 wall_time + exit_code]
  TR -->|结果 + 元信息| AR
  SB -.禁止.-> NET[外网]
  SB -.禁止.-> FS[宿主文件系统]
```

---

### 25.18 Long-term Memory 与状态持久化

Agent 的记忆不是一个选项，而是决定它能解决哪类任务的基础能力。没有记忆，每次对话都从零开始；记忆设计错误，会导致状态泄露、成本失控或恢复失败。本节按时间维度分层讨论。

#### 三层记忆架构

**短期记忆（秒级到分钟级）**：就是 context window 内的内容——对话历史、工具调用结果、中间推理步骤。主要受 `max_context_tokens` 约束，超出后触发 truncation 或 summarization。这一层完全在内存里，KV cache 是其在 GPU 侧的物化形式。不需要持久化，session 结束即释放。

**中期记忆（分钟到天级）**：跨 step 的 session 状态，存于 Redis 或 SQLite：

| 存储 | TTL 建议 | 适用内容 |
|------|----------|----------|
| Redis（内存） | 30min - 2h | 活跃 session 的 pending tool calls、partial output、预算余额 |
| Redis（持久化） | 1 - 7 day | 用户短期偏好、最近使用工具、跨 session 上下文摘要 |
| SQLite / PostgreSQL | 7 - 90 day | 审计日志、成本记录、任务结果存档 |

**长期记忆（永久或大 TTL）**：语义可检索的知识，分三类：

- **向量库**：对历史 session 摘要、工具调用结果、用户偏好做 embedding，语义相似度检索。代表方案：Qdrant、Weaviate、Pinecone。适合"类似的问题过去怎么解决的"查询。
- **知识图谱**：结构化关系存储，支持多跳推理（如"用户 A 负责项目 B，项目 B 使用数据库 C"）。代表方案：Neo4j、FalkorDB。适合有明确实体和关系的领域。
- **Episodic Memory**：按时间序列存储完整 episode（一次 agent 运行的完整 trace），支持"上次我问过类似问题，当时 agent 是怎么做的"的精确回放。

#### 主流 Memory 方案

| 方案 | 定位 | 核心机制 | 适用规模 |
|------|------|----------|----------|
| MemGPT / Letta | 自管理记忆的 agent OS | 把 LLM 上下文当 RAM，外部存储当磁盘，模型自决定何时 page in/out | 研究和复杂长任务 agent |
| mem0 | 轻量 memory API | 自动提取 fact/preference 写入向量库，检索时按语义召回 | 快速接入，生产推荐 |
| Zep | 对话 memory 平台 | 结构化 session memory + 图实体提取 + 时间衰减 | 对话助手、CRM 类场景 |
| Cognee | 知识图谱 + 向量混合 | 把文档、对话解析为图关系，查询时图遍历 + 向量混合召回 | 知识密集型 agent |

#### 状态持久化协议

**必须持久化的内容**：

- `pending_tool_calls`：agent 已规划但尚未执行的工具调用列表（防止 crash 后重复规划）
- `partial_output`：已生成的部分答案（用于 resume 时继续输出，而非重新生成）
- `reasoning_state`：当前推理步骤编号、预算余额、已执行 tool 列表
- `session_metadata`：tenant_id、task_description、start_time、budget_envelope

**不应持久化的内容**：

- **KV Cache**：KV Cache 是 GPU 内存的物化形式，成本高、TTL 短，持久化 KV 没有意义，应通过 prefix caching 自动命中，或在 resume 时重新 prefill（prefill 成本远低于 KV 持久化的存储和恢复成本）
- **Raw tool results > 10 KB**：大型工具结果应截断后存 summary，完整数据写对象存储（S3/GCS）并记录引用
- **中间 logits / hidden states**：计算成本极高且通常没有复用价值

#### 失败恢复：异步 Agent 崩溃后 Resume

```text
1. Crash 检测：看门狗发现 heartbeat 超时（通常 30s）
2. 状态读取：从 Redis 读取 reasoning_state + pending_tool_calls
3. Dedup 检查：对 pending_tool_calls 查幂等 key，避免重复执行有副作用工具
4. Context 重建：用 session_metadata + partial_output 重建上下文（不需要重跑所有 step）
5. Resume 点：从 last_completed_step + 1 继续，而不是从头开始
6. 预算校验：检查剩余预算是否仍足够完成剩余步骤，不足则返回 partial result
```

> **工程建议**：对有副作用的工具调用（如写数据库、发送邮件），在执行前先把 `{tool_name, idempotency_key, args_hash}` 写入持久存储，执行成功后写 `completed` 标志。Resume 时先检查该标志，避免因崩溃重试导致重复副作用。

> **工程边界**：向量库的语义检索是模糊的，不能用它做 idempotency 判断，只能用于"找相关记忆"。精确的状态持久化和 dedup 必须用 key-value 存储。

```mermaid
flowchart LR
  CW[Context Window<br/>短期 秒级] --> SUM[Summarizer]
  SUM --> RS[Redis Session Store<br/>中期 30min-7day]
  RS --> VDB[(向量库<br/>语义长期记忆)]
  RS --> KG[(知识图谱<br/>结构化关系)]
  RS --> EP[(Episodic Store<br/>完整 episode)]
  VDB --> RET[检索 top-K]
  KG --> RET
  EP --> RET
  RET --> CW
  RS --> DR[失败恢复<br/>resume from checkpoint]
```

---

### 25.19 端到端 Worked Example：SQL 查询 Agent

本节用一个完整的 SQL 查询 agent 把本章所有机制串联起来：从用户提问到 SQL 执行结果，经过完整的 ReAct 循环、工具沙箱、trace 记录、KV prefix caching 和失败处理。

#### 业务场景

用户用自然语言向公司内部数据仓库提问，如"上个月各区域销售额 Top 5 是哪些？"。系统需要把自然语言转成 SQL、验证 SQL 合法性、执行 SQL、格式化结果并返回。不能让用户直接写 SQL，也不能给 agent 直接写权限——必须通过受控工具链。

#### 完整架构

```mermaid
flowchart TD
  U[用户请求] --> API[FastAPI Gateway<br/>鉴权 限流 trace_id 注入]
  API --> LG[LangGraph Orchestrator<br/>ReAct 状态机]
  LG --> M[模型调用<br/>system prompt + schema + history]
  M -->|Thought + Action| LG
  LG --> T1[text_to_sql<br/>LLM 翻译工具]
  LG --> T2[validate_sql<br/>语法检查 + 权限验证]
  LG --> T3[execute_sql<br/>只读 sandbox DB 连接]
  LG --> T4[format_result<br/>Markdown 表格 + 摘要]
  T1 --> SB[SQL Sandbox<br/>只读连接 + 超时 5s]
  T2 --> SB
  T3 --> SB
  T4 -->|最终结果| U
  LG -->|每步| TR[Trace Pipeline<br/>step_id cost latency]
  M -. prefix cache命中 .-> KV[(KV Cache<br/>system prompt前缀)]
```

#### Tool Spec

```python
tools = [
    {
        "name": "text_to_sql",
        "description": "将自然语言问题转换为 SQL 查询语句。只生成 SELECT 语句。",
        "parameters": {
            "question": "string",  # 原始用户问题
            "schema_hint": "string"  # 相关表和字段的 schema 描述
        }
    },
    {
        "name": "validate_sql",
        "description": "验证 SQL 语法正确性并检查是否只包含只读操作。",
        "parameters": {
            "sql": "string"
        }
    },
    {
        "name": "execute_sql",
        "description": "执行经过验证的只读 SQL 查询，返回最多 100 行结果。",
        "parameters": {
            "sql": "string",
            "timeout_s": "int"  # 最大 5
        }
    },
    {
        "name": "format_result",
        "description": "将 SQL 查询结果格式化为用户友好的 Markdown 表格和摘要。",
        "parameters": {
            "rows": "list",
            "columns": "list",
            "question": "string"
        }
    }
]
```

#### 一次完整 Trace

| step_id | step_type | tool_name | tokens_in | tokens_out | reasoning_tokens | latency_ms | cost_usd | outcome |
|---------|-----------|-----------|-----------|------------|-----------------|------------|----------|---------|
| s1 | plan | — | 1024 | 180 | 320 | 1340 | $0.0021 | success |
| s2 | tool_call | text_to_sql | 512 | 95 | 0 | 680 | $0.0009 | success |
| s3 | tool_call | validate_sql | 200 | 30 | 0 | 45 | $0.0001 | success |
| s4 | tool_call | execute_sql | 0 | 0 | 0 | 1250 | $0.0000 | success |
| s5 | finalize | format_result | 640 | 320 | 0 | 890 | $0.0014 | success |
| **合计** | | | **2376** | **625** | **320** | **4205ms** | **$0.0045** | |

#### KV Prefix Caching 策略

system prompt 包含数据库 schema 描述（约 800 token），每次查询都相同。配合 vLLM 的 hash-based prefix caching，所有以相同 system prompt 开头的请求都命中缓存，只需计算 user question 部分的 prefill。

实测数据（LLaMA-3 70B，A100 × 2）：
- 冷启动（无 cache）：TTFT 1800ms
- 热路径（cache 命中）：TTFT 340ms（减少 81%）
- cache 命中率：生产环境平均 73%（同一租户反复问相关问题时命中率更高）

> **工程建议**：把 system prompt + 工具描述设计为稳定前缀（不要把时间戳或 session_id 塞进 system prompt），最大化 prefix cache 命中率。动态内容放在 user turn 或 assistant turn，不要污染 system prompt。

#### 失败模式与处理

| 失败类型 | 触发条件 | 处理策略 | error_class |
|----------|----------|----------|-------------|
| SQL 语法错 | validate_sql 返回错误 | 把错误信息回注 context，让模型重写 SQL，最多重试 2 次 | `SqlSyntaxError` |
| 查询超时 | execute_sql > 5s | 返回"查询超时，建议缩小时间范围"，不重试 | `QueryTimeoutError` |
| 权限不足 | validate_sql 检测到写操作 | 拒绝执行，返回"只允许只读查询" | `PermissionDeniedError` |
| 空结果 | 查询成功但返回 0 行 | 触发 format_result 生成"未找到数据"提示，不报错 | — |
| 预算耗尽 | reasoning_tokens 超过 max | 截断推理，要求基于已有信息给出最简答案 | `BudgetExceededError` |
| 幻觉表名 | validate_sql 检测表不存在 | 把可用表列表回注，让模型重选，最多重试 1 次 | `TableNotFoundError` |

#### Eval 设计

**100-prompt benchmark 构建**：从生产日志采样 80 条真实问题（涵盖时间范围查询、聚合、多表 join、条件过滤等类型），加 20 条手工构造的边界用例（超长时间范围、不存在的表名、包含写操作的恶意输入）。

**Judge Model 评分**：对每条问题，比较 agent 输出和标准 SQL 答案（人工标注）。用 GPT-4o 作为 judge，prompt 要求它判断：(1) 结果是否正确；(2) SQL 是否等价；(3) 格式是否友好。

**实测性能数据**（100-prompt benchmark，生产环境）：

| 指标 | 数值 |
|------|------|
| 平均端到端 latency | 4.2s |
| P99 latency | 11.8s（含 DB 慢查询） |
| cost per query（平均） | $0.0045 |
| tool call success rate | 94.2% |
| SQL 正确率（judge 评分） | 87% |
| 首次 SQL 命中率（不需重试） | 78% |
| 空结果率（不是错误） | 8% |

> **工程建议**：P99 latency 的主要来源是 DB 慢查询（execute_sql 超时），而不是模型推理。对 P99 做优化应先看工具侧超时分布，再看模型调用次数，最后才考虑模型参数调整。

> **工程边界**：Judge model 评分不是 ground truth，87% 的"正确率"中可能有 3-5% 是 judge 本身的误判。重要决策（如上线新版本）应配合人工抽查，不能完全依赖 LLM judge。

---

## 本章小结

| 主题 | 关键点 |
|------|--------|
| Agent 本质 | 一段带规划、执行、验证和停止条件的推理流程 |
| Inference-time compute | 用更多推理时算力换更高任务成功率，工程上可拆为隐式推理、多候选搜索、验证反思、工具增强执行 4 种模式 |
| 平台难点 | 多步状态、预算控制、工具依赖、成本放大、权限边界 |
| 是否值得 | 取决于任务价值、成功率提升、人工返工减少和 SLA 约束 |
| 治理重点 | 把预算、回退、trace、tool policy 和 quota 做成控制面能力 |

---

## 练习题

1. 从基础设施视角看，Agent 和 Thinking Model 分别会怎样改变会话持续时间、token 消耗可预测性和并发模型？
2. 给定 1,000 个并发 agent session、平均每个 session 6 次模型调用、每次调用平均 1.5 秒 GPU 时间，估算总 GPU-second 需求，并说明还缺哪些容量假设。
3. 为什么 agent 容量规划不能只用 `QPS x request latency`？请和传统在线推理调度做对比。
4. 隐式推理预算、多候选与搜索、验证与反思、工具增强执行 4 种 inference-time compute 模式分别会把成本放大到哪里？
5. 如果一个请求允许 `max_reasoning_tokens=8,000`，但租户剩余预算只够 4,000 个 reasoning tokens，你会如何设计截断和降级策略？
6. 请为一个“数学求解 agent”设计最大步数、最大工具调用数、`per_request_gpu_second_budget` 和回退策略。
7. 为什么 verifier-guided generation 需要把生成队列和验证队列分开限流？
8. Agent session 如何映射到 vLLM / TRT-LLM / SGLang 请求？为什么长 session 不应该直接等同于一个超长推理请求？
9. Tool calling 沙箱至少应该限制哪些资源？请说明超时、权限、网络和文件边界各自防止什么风险。
10. 多 Agent 并发运行时，怎样做租户级资源隔离，避免一个高预算 session 挤占其他短请求？
11. 长 context 与 Ch 15 的 KV Cache 有什么关系？prefix caching 和 KV 生命周期管理分别解决什么问题？
12. 如果一个 agent 系统成功率更高，但单位成本翻了 4 倍，你会如何判断它是否应该上线？
13. 设计一个 Agent 成本预测面板，至少包含 token、GPU-second、工具调用、session 时长和失败重试成本。
14. Agent 执行失败后，哪些步骤适合自动重试，哪些步骤应该直接返回 partial result 或转人工？为什么？
15. 新增：为一个代码修复 agent 设计预算 envelope，至少包含 `max_candidates`、`max_verify_rounds`、`max_tool_wall_time`、`max_context_tokens` 和 `tenant_budget_remaining`。
16. 新增：如果 Best-of-N 从 1 提高到 8，平均成功率从 62% 提高到 78%，但 P95 延迟从 3 秒变成 11 秒，你会怎样决定是否对所有租户开启？
17. 新增：请画出一个 agent runtime 与模型网关、serving engine、quota 服务、tool runner、trace pipeline 的调用链，并标注每一步的预算扣减点。
18. 新增：某 agent 每次失败都会重新调用同一个写操作工具，导致重复创建工单。请设计幂等 key、审批门和 retry 策略。
19. 新增：当 verifier 队列积压导致主模型 GPU 利用率下降时，你会先调整队列隔离、模型路由、验证轮数还是候选数？说明顺序。
20. 新增：给定一个 30 分钟异步研究 agent，哪些状态应该保存到 durable store，哪些只应保留在短 TTL cache 或 KV cache？
21. 新增：请设计一组 dashboard 指标，区分“模型推理成本上涨”“工具依赖变慢”“预算策略过宽”和“上下文压缩失败”四类问题。
