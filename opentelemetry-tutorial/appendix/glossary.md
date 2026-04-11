# 附录A：术语表

本附录用于快速查阅 OpenTelemetry 学习过程中最常出现的术语。中文为主，首次给出常见英文写法，并尽量用工程语境解释，而不是只给字面定义。

## 一、核心对象

| 术语 | 英文 | 简要解释 | 常见误区 |
|------|------|----------|----------|
| 可观测性 | Observability | 通过系统输出的信号来理解系统内部状态与行为的能力 | 不等于“有监控面板” |
| 遥测数据 | Telemetry | 系统发出的观测数据总称，通常包括 traces、metrics、logs | 不只指 tracing |
| 信号 | Signal | telemetry 的类别，如 trace、metric、log | 三种信号不是替代关系，而是互补关系 |
| 观测栈 | Observability Stack | 从信号生产、处理、存储、查询到告警和排障闭环的整套系统 | 不等于某一个 backend 产品 |
| OpenTelemetry | OpenTelemetry / OTel | 一套统一的观测数据模型、API、SDK、Collector 与生态规范 | 不等于某个单一 tracing 产品 |
| 后端 | Backend | 存储、查询、可视化 telemetry 的系统 | 不一定只有一种，也不必须与 OTel 绑定 |

## 二、Trace 与上下文相关术语

| 术语 | 英文 | 简要解释 | 你应该关注什么 |
|------|------|----------|------------------|
| 链路追踪 | Distributed Tracing | 记录请求穿过多个服务或组件时的路径与耗时 | 用于回答“请求经过了哪里、慢在哪、错在哪” |
| Trace | Trace | 一次端到端请求或事务的完整链路集合 | 一条 trace 由多个 spans 构成 |
| Span | Span | trace 中的单个操作单元，表示一个阶段、边界或调用 | 不要把所有函数都建成 span |
| 根 Span | Root Span | 一条 trace 的起点 span | 通常对应入口请求或任务起点 |
| 父子关系 | Parent-Child | span 之间的调用层级关系 | 同步调用更常见，异步不能机械套用 |
| Span Kind | Span Kind | span 角色，如 server、client、producer、consumer | 有助于理解服务边界和调用方向 |
| Span Event | Span Event | span 生命周期内的关键瞬时事件 | 适合记录重试、异常、状态切换等瞬时信息 |
| Span Status | Span Status | span 成功或失败的状态标记 | 不等同于业务是否成功 |
| Trace ID | Trace ID | 标识一条完整 trace 的唯一 ID | 跨服务关联日志时非常重要 |
| Span ID | Span ID | 标识单个 span 的唯一 ID | 用于精确定位某个链路节点 |
| 上下文 | Context | 在进程内传递当前 trace/span 等信息的执行上下文 | Node.js、异步环境尤其容易出问题 |
| 传播 | Propagation | 在进程间传递 trace 上下文的过程 | 断链路通常先查传播 |
| Baggage | Baggage | 随上下文传播的轻量键值对 | 不要把它当作任意数据包 |
| Link | Link | span 与其他 span context 的关联关系 | 适合异步、批处理、多输入场景 |
| 头采样 | Head Sampling | 在请求开始时决定是否保留 trace | 成本低，但可能错过后续异常 |
| 尾采样 | Tail Sampling | 在 Collector 看到更多 trace 信息后再决定是否保留 | 更智能，但更耗状态和资源 |

## 三、Metrics 相关术语

| 术语 | 英文 | 简要解释 | 你应该关注什么 |
|------|------|----------|------------------|
| 指标 | Metrics | 用于反映系统趋势、统计与聚合的时序数据 | 更擅长看整体变化，不擅长还原单次请求 |
| 数据点 | Datapoint | 某个时间点上的一次指标样本 | 不等同于时间序列 |
| 时间序列 | Time Series | 指标名加上一组属性后的连续观测序列 | 成本经常主要由 series 数量决定 |
| Counter | Counter | 只增不减的计数器 | 适合请求数、错误数、消费数 |
| Gauge | Gauge | 某时刻的当前值 | 适合队列长度、并发数、温度等 |
| Histogram | Histogram | 用于记录值的分布，常用于延迟 | 很适合 SLI、P95/P99 等延迟分析 |
| 同步 Instrument | Synchronous Instrument | 在业务执行路径中主动记录数值 | 适合请求内事件 |
| 异步 Instrument | Asynchronous Instrument | 由回调在采集时读取当前状态 | 适合当前内存占用、队列长度 |
| 标签 / 属性维度 | Labels / Attributes | 用于给指标增加维度的键值对 | 高基数字段会直接增加 series 数量 |
| SLI | Service Level Indicator | 服务质量指标，如错误率、延迟、可用性 | 应与业务承诺或运行目标直接相关 |
| SLO | Service Level Objective | 基于 SLI 设定的目标值 | 用于运维承诺和告警设计 |

## 四、Logs 与关联相关术语

| 术语 | 英文 | 简要解释 | 你应该关注什么 |
|------|------|----------|------------------|
| 日志 | Logs | 记录离散事件、错误细节、业务证据的文本或结构化数据 | 适合回答“这次到底发生了什么” |
| 结构化日志 | Structured Logging | 用固定字段输出日志，便于机器检索和聚合 | 比自由文本更适合大规模系统 |
| 日志关联 | Log Correlation | 让日志带上 trace_id、span_id 等关联信息 | 是排障闭环的关键一步 |
| 日志记录 | Log Record | 一条独立日志事件 | 不等于 span event |
| 审计日志 | Audit Log | 面向合规、审计或重要操作留痕的日志 | 不应和普通调试日志混为一谈 |
| 噪声日志 | Noisy Logs | 量大、重复、价值低的日志 | 会带来高成本和检索干扰 |

## 五、Resource、Attributes 与语义约定

| 术语 | 英文 | 简要解释 | 你应该关注什么 |
|------|------|----------|------------------|
| 资源 | Resource | telemetry 来源实体的身份信息 | 典型字段有 service.name、environment |
| 资源属性 | Resource Attributes | 附着在资源上的键值对 | 应更稳定、更偏身份层 |
| 属性 | Attributes | 附着在 span、metric、log 上的键值对 | 要区分稳定维度与高基数实例值 |
| 语义约定 | Semantic Conventions | 对常见领域字段名和含义的标准化约定 | 目的是统一语义，不是限制业务表达 |
| 服务名 | `service.name` | 服务的主身份字段 | 应保持长期稳定、组织内可识别 |
| 环境 | `deployment.environment.name` | 运行环境字段，如 production、staging、development | 应统一枚举值，避免 prod/prod01 混乱 |
| 版本 | `service.version` | 服务版本标识 | 对灰度、回滚、问题回溯很重要 |

## 六、Collector 与 Pipeline 相关术语

| 术语 | 英文 | 简要解释 | 你应该关注什么 |
|------|------|----------|------------------|
| Collector | OpenTelemetry Collector | 独立的 telemetry pipeline 组件 | 不只是转发器，更是治理层 |
| Receiver | Receiver | Collector 的输入组件 | 决定数据从哪里进入 |
| Processor | Processor | Collector 的中间处理组件 | 负责过滤、批处理、变换、采样等 |
| Exporter | Exporter | Collector 的输出组件 | 决定数据发往哪里 |
| Pipeline | Pipeline | receiver、processor、exporter 组成的流水线 | traces、metrics、logs 通常分别配置 |
| Agent 模式 | Agent Mode | Collector 靠近应用部署的模式 | 更像接入层 |
| Gateway 模式 | Gateway Mode | Collector 集中部署的模式 | 更像治理层 |
| OTLP | OpenTelemetry Protocol | OTel 常用传输协议 | 可用 gRPC 或 HTTP 承载 |
| 批处理 | Batch | 合并小批量数据后再导出 | 提高效率，降低发送开销 |
| 内存限制器 | Memory Limiter | 保护 Collector 内存的 processor | 生产环境常见基础配置 |
| 过滤 | Filter | 丢弃不需要的数据 | 用于去噪、降成本、做治理 |
| 变换 | Transform | 修改字段、删除字段或重写结构 | 常用于标准化与脱敏 |
| 路由 | Routing | 根据条件把数据发往不同出口 | 常用于多租户、多后端、多环境 |

## 七、工程治理与成本相关术语

| 术语 | 英文 | 简要解释 | 你应该关注什么 |
|------|------|----------|------------------|
| 基数 | Cardinality | 某个字段可能取值的数量规模 | 高基数会影响成本、性能和查询体验 |
| 高基数 | High Cardinality | 字段唯一值很多，如 user.id、session.id | 通常不适合作为高频 metrics 维度 |
| 采样 | Sampling | 只保留部分 telemetry 的策略 | 主要用于控制 traces 成本 |
| 保留期 | Retention | 数据在 backend 中保留的时间 | 不同信号适合不同保留时长 |
| 查询延迟 | Query Latency | 查询 telemetry 数据所需时间 | 与 schema、索引、基数都有关 |
| 数据治理 | Telemetry Governance | 对命名、字段、保留、采样、访问等进行统一管理 | 没有治理，观测系统会快速失控 |
| 厂商中立 | Vendor-Neutral | 不把应用和观测模型强耦合到某家私有实现 | 有助于迁移、多后端和平台统一治理 |
| rollout | Rollout | 分阶段启用、灰度和迁移的过程 | 适用于 OTel 接入、Collector 变更和后端迁移 |

## 八、速记：初学者最容易混淆的几个概念

| 容易混淆的概念 | 正确区分 |
|----------------|----------|
| OTel vs backend | OTel 负责标准化接入与模型，backend 负责存储、查询与呈现 |
| Resource vs Attributes | Resource 更偏身份层；Attributes 更偏某次 span、metric、log 的上下文 |
| Trace vs Span | Trace 是整条链路；Span 是链路中的单个节点 |
| Metrics vs Logs | metrics 负责趋势和聚合；logs 负责细节与证据 |
| Span Event vs Log | event 是 span 内部的关键瞬时注记；log 是独立日志记录 |
| Agent vs Gateway | agent 靠近应用负责接入；gateway 集中负责治理 |
| 高基数字段 vs 有价值字段 | 有价值不等于适合聚合；很多字段只适合 trace 或 log，不适合 metric |

## 九、学习建议：如何使用本术语表

建议你把术语表当作下面三种场景的快速参考：

1. 阅读章节时，遇到概念一时记不清，先回来看简明解释。
2. 设计埋点和 Collector pipeline 时，用它检查概念是否放对层次。
3. 做团队评审时，用统一词汇减少“同词异义”或“同义异名”问题。

如果你发现自己经常混淆某几个术语，通常意味着你还没有完全建立“信号直觉”“管道直觉”或“生产直觉”。这很正常，建议配合第 2、9、13、16、22、24 章反复对照阅读。
