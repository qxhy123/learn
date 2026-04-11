# 第24章：搭建一个观测栈

## 学习目标

- 从系统视角理解一个完整 observability stack 需要哪些层次，以及它们各自负责什么
- 学会为应用、Collector、backend 设计端到端的 traces、metrics、logs 数据路径与治理边界
- 理解从采集、处理、存储、查询到排障闭环的关键设计取舍，而不把观测栈误解成某个单一产品
- 能根据规模、团队分工与可靠性需求，规划 agent、gateway、多后端和多环境的部署蓝图
- 建立“观测栈本身也必须被观测”的工程意识，形成完整运行闭环

## 1. 观测栈不是一个产品，而是一条从应用到决策的链路

很多人第一次接触 observability stack，会直接想到某个 tracing 后端、某个指标系统或某个日志平台。但如果从工程设计角度看，一个完整的观测栈至少要回答六个问题：

1. 数据从哪里产生
2. 数据如何被接收与处理
3. 数据如何被可靠地传输与路由
4. 数据落到哪里保存和查询
5. 人如何从一个异常信号跳转到其他信号
6. 这套观测栈自己是否稳定可靠

因此，观测栈不是某个单点组件，而是下面这条链路：

```text
应用 / 平台信号源
    -> OpenTelemetry SDK / Agent / Exporter
    -> Collector 接入层
    -> Collector 治理层
    -> traces / metrics / logs backend
    -> 仪表盘 / 告警 / 查询界面 / 排障流程
    -> 运维决策与工程反馈
```

如果只关注其中某一个点，例如“后端能不能搜到 trace”，就容易忽略真正影响效果的其他环节：

- 应用侧 schema 是否稳定
- Collector 是否能承受流量和做统一治理
- logs 是否与 trace 真正关联
- metrics 是否能支撑告警和 SLO
- 值班流程是否能从一个告警快速跳到相关 trace 和日志

### 1.1 一个观测栈至少包含三层职责

| 层次 | 主要职责 |
|------|----------|
| 生产层 | 应用、运行时、宿主机、Kubernetes、消息系统产生 telemetry |
| 管道层 | Collector 接收、处理、过滤、采样、路由、导出 |
| 消费层 | backend、仪表盘、告警、检索、排障与治理报表 |

成熟的观测栈设计，关键不是让每一层都变强，而是让每一层的边界清晰。

## 2. 应用层蓝图：从服务、运行时到日志关联

一个端到端观测栈的起点，始终是信号生产层。应用层至少要解决三件事：

- 生成正确的 traces、metrics、logs
- 使用统一的 Resource 和语义约定
- 把数据稳定送到近端或统一入口，而不是直接耦合多个后端

### 2.1 应用侧的最小蓝图

对于一个典型微服务系统，可以把应用层抽象为：

```text
HTTP / RPC 服务
异步 worker
批处理任务
数据库 / 缓存 / MQ 客户端
结构化日志输出
```

这些组件通过 OTel API / SDK 与 auto-instrumentation、手动埋点相结合，生产出三种信号：

- traces：请求链路、依赖调用、业务阶段、异步消费
- metrics：请求量、错误率、延迟、队列积压、worker 并发
- logs：结构化事件、错误细节、审计或补偿证据

### 2.2 应用层要尽量保持 vendor-neutral

应用侧更适合负责：

- Resource 身份
- 关键业务 span
- 业务指标与日志字段
- OTLP 输出到本地或近端 Collector

而不适合负责：

- 多后端并发导出逻辑
- 复杂脱敏、路由、尾采样规则
- 后端专有 query 或专有 agent 绑定逻辑

这让应用代码更容易长期维护，也便于未来迁移 backend。

### 2.3 应用层蓝图示意

下面给出一个偏教学用途的蓝图：

```text
[user request]
    -> gateway service
    -> order service
    -> payment service
    -> message broker
    -> inventory worker

每个服务内部：
- OTel SDK 负责 traces / metrics
- structured logger 注入 trace_id / span_id
- OTLP exporter 指向本地或节点级 Collector
```

这套设计的关键点是：**服务负责生产语义，Collector 负责管道治理。**

## 3. Collector 层蓝图：接入层、治理层与多信号管道

第 13 到 15 章已经讨论了 Collector 的基础、processor 和部署方式。到了完整观测栈层面，要把 Collector 看成两个逻辑层次。

### 3.1 接入层 Collector：靠近应用，负责统一入口

接入层 Collector 常见部署形式包括：

- 本机 agent
- Kubernetes DaemonSet
- sidecar

它们更适合承担：

- 接收应用 OTLP 数据
- 吸收局部网络抖动
- 补充主机或节点侧上下文
- 做轻量批处理和内存保护

一个简化接入层示意：

```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 256
  batch:
    timeout: 2s

exporters:
  otlp/gateway:
    endpoint: otel-gateway.observability.svc.cluster.local:4317
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlp/gateway]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlp/gateway]
    logs:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlp/gateway]
```

这层的原则是“轻处理”，不要把所有复杂逻辑都分散到每个节点。

### 3.2 治理层 Collector：集中做策略、路由与出口控制

治理层 Collector 更适合承担：

- 统一资源字段修正
- 脱敏与过滤
- 多租户或多环境路由
- traces 的 tail sampling
- 多后端导出与失败重试策略

例如：

```yaml
receivers:
  otlp:
    protocols:
      grpc:

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 1024
  resource:
    attributes:
      - key: telemetry.pipeline
        value: gateway
        action: upsert
  batch:
    send_batch_size: 2048
    timeout: 5s

exporters:
  otlp/traces-backend:
    endpoint: traces-backend.internal:4317
    tls:
      insecure: true
  otlp/metrics-backend:
    endpoint: metrics-backend.internal:4317
    tls:
      insecure: true
  otlp/logs-backend:
    endpoint: logs-backend.internal:4317
    tls:
      insecure: true
  debug:
    verbosity: basic

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, resource, batch]
      exporters: [debug, otlp/traces-backend]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, resource, batch]
      exporters: [otlp/metrics-backend]
    logs:
      receivers: [otlp]
      processors: [memory_limiter, resource, batch]
      exporters: [otlp/logs-backend]
```

上面并不假设 traces、metrics、logs 必须落到同一个后端。相反，它强调的是：**统一入口、统一治理，不等于统一存储实现。**

### 3.3 三种信号的 Collector 策略不必完全相同

一个成熟的观测栈通常不会把三种信号完全等价处理。因为它们的目标和成本结构不同：

| 信号 | Collector 常见重点 |
|------|------------------|
| traces | 采样、链路完整性、错误优先保留、多后端导出 |
| metrics | 稳定聚合、低高基数、长期趋势、远端写入可靠性 |
| logs | 过滤噪声、脱敏、字段标准化、与 trace 关联 |

## 4. Backend 与消费层蓝图：不是只存数据，而是形成排障闭环

一个真正可用的观测栈，不会停在“数据进 backend 了”。更重要的是，人能否基于这些数据完成监控、分析、定位与反馈。

### 4.1 backend 层通常至少包含三类能力

| 能力 | 作用 |
|------|------|
| 存储与检索 | 按信号类型保存并支持查询 |
| 可视化与聚合 | 仪表盘、列表、查询视图、时间序列分析 |
| 告警与通知 | 基于指标、日志或 trace 衍生信号触发告警 |

不同团队可以采用：

- 单一后端承载多信号
- 不同后端分别承载 traces、metrics、logs
- 一体化查询界面叠加多种底层存储

关键不在于选哪种，而在于是否保持了以下能力：

- 信号之间能够关联
- 查询路径与 runbook 清晰
- schema 不因后端不同而彻底分裂

### 4.2 从告警到定位的闭环设计

一个成熟的排障闭环通常是这样的：

```text
指标告警
  -> 定位异常服务 / 路由 / 依赖
  -> 打开相关 trace
  -> 确认慢点或错误点
  -> 跳转到关联日志
  -> 找到输入条件、错误细节、补偿记录
  -> 回到发布记录、版本、变更单或 runbook
```

这要求三类信号不是孤立存在，而是能互相缩小搜索空间。

### 4.3 第 24 章要求的端到端蓝图

下面给出一个完整的 vendor-neutral 端到端蓝图：

```text
[客户端 / 上游系统]
        |
        v
[gateway service] -----------------------------.
        |                                      |
        v                                      |
[order service] ----> [payment service]        |
        |                 |                    |
        |                 v                    |
        |            [payment DB]              |
        v                                      |
[message broker] ----> [inventory worker] -----'

每个服务 / worker：
- OTel SDK 生成 traces、metrics
- structured logs 注入 trace_id / span_id
- Resource 标识 service.name / version / environment
- OTLP 导出到节点级或本地 Collector

节点 / 本地 Collector（接入层）：
- 接收 OTLP traces / metrics / logs
- memory_limiter + batch
- 转发到集中 gateway Collector

Gateway Collector（治理层）：
- resource 修正与统一标签
- filter / transform / 脱敏
- sampling / routing
- traces -> trace backend
- metrics -> metrics backend
- logs -> log backend

消费层：
- metrics backend 负责趋势、SLO、告警
- trace backend 负责链路分析、错误定位
- log backend 负责检索细节、审计与证据
- 统一仪表盘和 runbook 串联排障流程
```

这个蓝图体现了 3 个关键点：

1. 应用只依赖统一入口，不直接绑多个后端
2. Collector 把接入层与治理层职责分开
3. traces / metrics / logs 共同服务于排障与运维闭环

### 4.4 观测栈本身也要有消费视角

除了排障闭环，还应有治理视角的消费面：

- 哪些服务没有 trace 关联日志
- 哪些指标 series 数量异常膨胀
- 哪些 Collector exporter 失败率上升
- 哪些环境还在使用旧字段或旧路径

这让观测栈不只支持“查业务问题”，也支持“治理观测系统本身”。

## 5. 多环境、多团队与可靠性设计：观测栈如何规模化

当观测栈从单团队扩展到多团队、多环境时，问题就不再只是“功能够不够”，而是“边界能否维持住”。

### 5.1 多环境设计

最少要明确：

- development、staging、production 是否使用同一 Collector 架构
- 不同环境是否共用 backend，如何隔离
- 非生产环境是否降低采样和保留时长
- 演练环境和临时环境是否允许缩减信号范围

一个常见做法是：

| 环境 | 典型策略 |
|------|----------|
| development | 保留最小闭环，便于开发验证 |
| staging | 尽量接近生产 schema 与 pipeline |
| production | 完整治理、采样、告警与保留策略 |

### 5.2 多团队设计

多团队下，最好分清：

- 平台统一定义的字段与 pipeline
- 业务团队可扩展的业务字段
- 哪些公共告警由平台负责，哪些由服务团队负责

否则就容易出现：

- 平台团队过度限制，业务方无法表达关键语义
- 业务团队无限制扩展，schema 快速失控

### 5.3 可靠性设计：观测系统自己也会失败

观测栈本身要考虑：

- Collector 的 HA 与扩容
- backend 的吞吐、保留和查询压力
- 导出失败重试是否会引发积压
- 网络分区时数据丢失与降级行为
- 关键告警链路在部分信号失效时是否仍可工作

一个实用原则是：

- 不要把所有关键能力都建立在单一后端或单一 Collector 实例上
- 对关键链路要有退化设计，例如优先保住核心 metrics 告警能力
- 允许某些低优先级信号在高压时被丢弃，但必须知道丢弃了什么

## 6. 观测栈的运行闭环：让系统持续演进，而不是上线即结束

观测栈搭建完成，不代表工作结束。真正成熟的系统还需要运行闭环。

### 6.1 先观测观测栈本身

至少应持续关注：

| 组件 | 应监控内容 |
|------|------------|
| 应用 SDK | 导出失败、队列积压、flush 异常 |
| 接入层 Collector | CPU、内存、接收吞吐、丢弃量 |
| 治理层 Collector | processor 延迟、导出失败、队列积压 |
| backend | 写入失败、查询延迟、索引压力、存储增长 |
| 告警链路 | 告警触发率、通知失败、静默配置异常 |

### 6.2 建立变更评审与成本回顾

观测栈一旦规模化，以下变更都应进入评审：

- 新增高频指标与维度
- 修改 `service.name` 或关键资源字段
- 新增日志索引字段
- 增加 Collector processor、路由或导出目标
- 调整采样与保留策略

同时，应定期回顾：

- 哪些信号几乎没人使用
- 哪些字段成本高但价值低
- 哪些告警噪音大、无实际处置价值
- 哪些服务的 trace / logs 关联仍然断裂

### 6.3 用 runbook 把技术栈与操作流程接起来

如果观测栈没有与 runbook、值班手册、发布流程关联，最终很可能只是“系统里多了一堆面板”。

更好的做法是把关键问题串成标准动作，例如：

1. 收到 `order-api` 高错误率告警
2. 先看对应 SLO 仪表盘
3. 再打开 trace 查询，筛选最近错误样本
4. 跳转到关联日志，确认错误类型和下游响应模式
5. 对照最近版本变更和依赖变更
6. 记录根因、缺失字段与后续埋点改进

这一步，才真正把 observability stack 从“技术组件集合”变成“工程操作系统的一部分”。

## 本章小结

| 主题 | 核心结论 |
|------|----------|
| 观测栈定义 | 不是单一产品，而是从信号生产到运维决策的完整链路 |
| 应用层职责 | 生产正确语义并输出到统一入口，尽量保持 vendor-neutral |
| Collector 层职责 | 分为接入层与治理层，负责接收、保护、过滤、路由和导出 |
| backend 层职责 | 存储、查询、可视化与告警，重点是形成跨信号排障闭环 |
| 端到端蓝图 | 应兼顾 traces、metrics、logs 与从告警到定位的完整路径 |
| 持续运行 | 观测栈本身也必须被观测、评审并持续优化 |

## OTel实验

### 实验目标

画出并评审一份适合自己系统的 observability stack 蓝图，要求同时覆盖应用、Collector、backend 和排障闭环。

### 实验步骤

1. 列出你的系统中至少 4 类信号源，例如 API 服务、异步 worker、Kubernetes 节点、文件日志来源。
2. 为这些信号源规划应用侧接入方式，区分自动 instrumentation、手动埋点和结构化日志注入。
3. 设计一层接入型 Collector 和一层治理型 Collector，明确每层的 receiver、processor、exporter 职责。
4. 决定 traces、metrics、logs 是否落到同一个 backend，或分别落到不同 backend，并写出原因。
5. 设计一条排障闭环路径：从某个服务高错误率告警开始，如何逐步跳到 trace、日志和发布信息。
6. 再补一张“观测栈观测项”表，列出你会如何监控 SDK、Collector、backend 和告警链路本身。
7. 最后检查整个蓝图是否满足 vendor-neutral、可迁移、可治理三项要求。

### 预期收获

- 你会更清楚一个完整观测栈其实是多层系统，而不是一个后端产品。
- 你会看到接入层和治理层分离后，应用与平台职责会更清晰。
- 你会意识到 traces、metrics、logs 的价值不在于各自存在，而在于能否共同形成排障闭环。

## 练习题

1. 为什么说 observability stack 不是某个具体后端，而是一条从信号生产到运维决策的完整链路？
2. 在一个端到端蓝图中，应用层、接入层 Collector、治理层 Collector、backend 层各自更适合负责什么？
3. 为什么很多生产系统会把 Collector 分成靠近应用的接入层和集中治理的 gateway 层？这种分层有什么好处？
4. 请描述一个从指标告警开始，经过 trace 和日志，最终完成问题定位的排障闭环。
5. 为什么“观测栈本身也要被观测”？如果忽略这一点，最容易出现哪些盲区？
