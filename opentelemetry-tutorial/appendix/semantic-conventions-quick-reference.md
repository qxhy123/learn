# 附录C：Semantic Conventions 速查

本附录用于快速查阅 OpenTelemetry 中最常见的语义约定字段。它不是完整规范翻译，而是面向学习与工程设计的“高频字段参考”。重点是帮助你判断：

- 哪些字段更适合放在 Resource
- 哪些字段常见于 span / metric / log attributes
- 哪些字段更适合聚合，哪些字段只适合排障

## 一、先记住三条总原则

### 原则 1：先区分 Resource 和普通 Attributes

| 类型 | 更适合承载什么 |
|------|----------------|
| Resource | 服务、环境、区域、实例等“身份层”信息 |
| Span / Metric / Log Attributes | 某次操作、某类事件、某条指标样本的上下文信息 |

### 原则 2：优先使用稳定、可解释、可迁移的字段

好的字段通常满足：

- 语义清楚
- 不依赖某家私有后端解释
- 适合长期复用
- 不会轻易制造高基数爆炸

### 原则 3：高价值不等于高频聚合友好

例如：

- `user.id` 对某次排障可能很有价值
- 但通常不适合作为高频 metrics 的维度

所以设计时要同时问：

1. 这个字段有没有诊断价值？
2. 它适合放在哪种信号里？
3. 它是否适合长期聚合和索引？

## 二、最重要的 Resource 字段

这些字段通常用于表达 telemetry 来源身份，适合长期保留和聚合。

| 字段 | 常见含义 | 建议用途 | 注意事项 |
|------|----------|----------|----------|
| `service.name` | 服务主身份 | 查询、聚合、告警分组的核心字段 | 应稳定、清晰、组织内唯一 |
| `service.namespace` | 服务所属域或命名空间 | 降低重名冲突，表达组织边界 | 适合中大型组织 |
| `service.version` | 当前服务版本 | 灰度、回滚、版本对比 | 不要和镜像 tag 混用得过于随意 |
| `deployment.environment.name` | 环境，如 production、staging、development | 环境隔离与过滤 | 枚举值应统一 |
| `service.instance.id` | 单实例身份 | 精确排障 | 不适合作为核心高频聚合维度 |
| `host.name` | 主机名 | 主机排障 | 云与容器环境中可能不稳定 |
| `cloud.region` | 区域 | 多区域分析 | 命名应与平台环境对齐 |
| `k8s.cluster.name` | 集群名 | 多集群定位 | 平台侧统一注入更稳 |
| `k8s.namespace.name` | K8s namespace | 平台分组与隔离 | 常与 service.name 一起使用 |
| `k8s.pod.name` | Pod 名 | 单实例排障 | 高变化，谨慎做长期聚合 |

### 资源字段速记

- `service.*` 更偏“服务身份”
- `deployment.*` 更偏“部署环境”
- `host.*`、`k8s.*`、`cloud.*` 更偏“运行位置”

## 三、HTTP 相关常见字段

HTTP 是最常见的 OTel 语义场景之一，通常同时影响 traces、metrics 和 logs。

### 1. 常见 HTTP 字段速查

| 字段 | 常见含义 | 更适合的用途 | 设计提醒 |
|------|----------|--------------|----------|
| `http.request.method` | 请求方法，如 GET、POST | traces、metrics、logs | 低基数，适合聚合 |
| `http.response.status_code` | 响应状态码 | traces、metrics、logs | 适合错误率与分类分析 |
| `url.scheme` | 协议，如 http、https | traces | 通常用于协议上下文 |
| `url.path` | 原始路径 | traces、logs | 原始路径可能含高基数片段 |
| `http.route` | 模板路由，如 `/orders/:id` | traces、metrics、logs | 比原始路径更适合聚合 |
| `server.address` | 服务端地址 | traces | 用于网络定位 |
| `server.port` | 服务端端口 | traces | 辅助上下文 |
| `client.address` | 客户端地址 | traces、logs | 需注意隐私与高基数 |
| `network.protocol.version` | 协议版本，如 HTTP/1.1 | traces | 适合协议层诊断 |
| `user_agent.original` | User-Agent 原始值 | traces、logs | 高变化，谨慎聚合 |

### 2. HTTP 字段设计建议

| 适合长期聚合的字段 | 不宜直接做高频聚合的字段 |
|-------------------|------------------------------|
| `http.request.method` | `url.path` 原始路径 |
| `http.route` | `user_agent.original` |
| `http.response.status_code` | `client.address` |
| `deployment.environment.name` | 带查询参数的完整 URL、`url.full` 等原始地址字段 |

### 3. HTTP 设计提醒

- 优先用 `http.route` 这类模板化字段做指标维度
- 谨慎记录完整 URL，尤其是查询参数中可能含敏感信息时
- 技术状态码与业务结果最好区分，不要全部混成一个字段

## 四、RPC 与远程调用相关字段

适用于 gRPC、Thrift、内部 RPC 等场景。

| 字段 | 含义 | 常见用途 | 备注 |
|------|------|----------|------|
| `rpc.system` | RPC 体系，如 grpc | traces、metrics | 低基数，适合聚合 |
| `rpc.service` | RPC 服务名 | traces、metrics | 表示目标服务接口 |
| `rpc.method` | RPC 方法名 | traces、metrics | 适合延迟和错误率分析 |
| `server.address` | 目标地址 | traces | 更偏排障 |
| `server.port` | 目标端口 | traces | 辅助上下文 |

设计提醒：

- `rpc.service`、`rpc.method` 很适合做调用级指标维度
- 原始请求对象、动态参数不适合直接做属性输出

## 五、数据库相关字段

数据库 span 是微服务 trace 中最常见的依赖之一。

| 字段 | 含义 | 更适合的用途 | 注意事项 |
|------|------|--------------|----------|
| `db.system` | 数据库类型，如 mysql、postgresql、redis | traces、metrics | 适合依赖分类 |
| `db.namespace` | 数据库、schema、keyspace 等命名空间 | traces | 适合定位数据库域 |
| `db.operation.name` | 操作名，如 SELECT、INSERT | traces、metrics | 很适合做依赖层聚合 |
| `db.collection.name` | 集合、表等逻辑对象名 | traces | 谨慎评估数量规模 |
| `server.address` | 数据库地址 | traces | 用于定位依赖实例 |
| `server.port` | 数据库端口 | traces | 辅助上下文 |

数据库场景的关键提醒：

- 数据库类型、操作类型通常适合聚合
- 原始 SQL 文本、动态参数、主键值一般不适合高频输出或索引
- 如涉及敏感数据，日志与 trace 中都要优先考虑脱敏

## 六、消息队列与异步系统相关字段

异步系统是很多链路断裂和 schema 失控的来源，因此相关语义字段要特别谨慎。

| 字段 | 含义 | 常见用途 | 设计提醒 |
|------|------|----------|----------|
| `messaging.system` | 消息系统，如 kafka、rabbitmq | traces、metrics | 低基数，适合聚合 |
| `messaging.operation.type` | 操作类型，如 publish、receive、process | traces、metrics | 有助于理解角色 |
| `messaging.destination.name` | topic / queue / exchange 名称 | traces、metrics | 适合依赖层分析，但要控制命名规模 |
| `messaging.client_id` | 客户端标识 | traces | 常用于排障，不宜滥用聚合 |
| `messaging.batch.message_count` | 批次消息数 | traces、metrics | 批处理场景很有价值 |
| `messaging.consumer.group.name` | 消费组名 | traces、metrics | 适合消费延迟与积压分析 |

异步场景提醒：

- topic、queue 等逻辑名称通常适合聚合
- 单条消息 ID、业务主键通常更适合 trace/log，而不是 metric 维度
- 异步链路的上下游关系不一定是 parent-child，字段设计要配合 trace 模型

## 七、异常与错误相关字段

错误相关字段经常是排障效率高低的关键。

| 字段 | 含义 | 常见用途 | 设计提醒 |
|------|------|----------|----------|
| `error.type` | 错误类别，如 timeout、validation_error | traces、metrics、logs | 建议做成有限枚举 |
| `exception.type` | 异常类型 | traces、logs | 更偏语言或运行时层 |
| `exception.message` | 异常信息 | traces、logs | 可能包含动态内容，不适合高频聚合 |
| `exception.stacktrace` | 调用堆栈 | traces、logs | 成本高，需谨慎保留 |

设计建议：

- 尽量区分技术错误类型与业务拒绝类型
- 用 `error.type` 这类稳定字段支撑聚合与告警
- 动态 message 和 stacktrace 更适合作为排障细节，而不是核心索引维度

## 八、部署、运行时与基础设施字段

这些字段常见于平台侧统一注入或自动探测。

| 字段 | 含义 | 常见用途 | 注意事项 |
|------|------|----------|----------|
| `deployment.environment.name` | 环境标识 | 多环境隔离、仪表盘过滤、告警分组 | 应统一命名 |
| `process.pid` | 进程 ID | 进程级排障 | 高变化，不适合长期聚合 |
| `process.executable.name` | 进程可执行名 | 排障、运行时识别 | 辅助字段 |
| `os.type` | 操作系统类型 | 平台分析 | 通常低基数 |
| `container.id` | 容器 ID | 单实例排障 | 高变化，谨慎聚合 |
| `container.name` | 容器名 | K8s / 容器环境定位 | 需要和部署模型区分 |

## 九、日志中最值得统一的字段

虽然日志字段不一定都来自语义规范，但以下字段通常值得统一：

| 字段 | 用途 | 建议 |
|------|------|------|
| `timestamp` | 时间排序与检索 | 统一格式与时区策略 |
| `severity` / `level` | 日志级别 | 枚举统一 |
| `message` | 人类可读摘要 | 保持简洁，避免堆整包内容 |
| `service.name` | 服务身份 | 与 Resource 对齐 |
| `deployment.environment.name` | 环境隔离 | 与 Resource 对齐 |
| `trace_id` | trace 关联 | 强烈建议保留 |
| `span_id` | span 关联 | 推荐保留 |
| `error.type` | 错误分类 | 便于聚合与过滤 |
| `request.id` | 非 OTel 专属的本地请求标识 | 视系统需要保留，但不要替代 trace_id |

## 十、哪些字段更适合聚合，哪些字段更适合排障

| 字段类别 | 更适合聚合 | 更适合排障 |
|----------|------------|------------|
| 服务身份 | `service.name`、`deployment.environment.name` | `service.instance.id` |
| HTTP | `http.route`、`http.request.method`、`http.response.status_code` | `url.path` 原始值、`client.address` |
| DB | `db.system`、`db.operation.name` | 原始 SQL、动态参数 |
| Messaging | `messaging.system`、`messaging.destination.name`、`messaging.consumer.group.name` | message ID、业务主键 |
| Error | `error.type` | `exception.message`、`exception.stacktrace` |
| 业务上下文 | `user.tier`、`tenant.plan` | `user.id`、`order.id`、`session.id` |

一个实用判断句是：

> 能帮助长期看趋势的，更可能适合聚合；只能帮助定位单次请求的，更可能适合 trace 或 log 细节。

## 十一、速记：设计字段时的五个检查问题

每当你准备新增一个字段时，可以快速问自己：

1. 它属于 Resource，还是属于 span / metric / log attributes？
2. 它是稳定类别，还是高基数实例值？
3. 它的主要用途是聚合、关联还是单次排障？
4. 它是否可能包含敏感信息？
5. 它是否和已有语义字段重叠、冲突或命名不一致？

如果这五个问题回答不清，通常说明字段设计还不成熟。

## 十二、学习建议：如何使用本附录

你可以这样使用这份速查：

- 写埋点时，用它检查字段是否放到了正确层次
- 评审 metrics 设计时，用它判断哪些字段不该做高频维度
- 设计日志 schema 时，用它对照 trace/log 关联字段是否齐全
- 迁移旧体系字段时，用它建立旧字段到统一语义的映射

如果你希望把这份速查真正用好，建议结合第 9、12、16、20、22、23、24 章一起看。附录负责帮助你“快速命名和放对位置”，正文负责帮助你理解“为什么这样命名和这样放”。
