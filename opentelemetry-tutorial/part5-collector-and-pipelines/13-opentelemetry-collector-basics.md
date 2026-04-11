# 第13章：OpenTelemetry Collector 基础

## 学习目标

- 理解 OpenTelemetry Collector 在整个可观测性体系中的职责与定位
- 掌握 receiver、processor、exporter、pipeline 这四个核心概念
- 说清楚 agent 与 gateway 两种 Collector 模式分别适合什么场景
- 能读懂一个最小 Collector YAML 配置，并知道数据如何流动
- 理解为什么 Collector 是治理层而不是“只是个转发器”

## 1. Collector 在系统里到底负责什么

如果说应用中的 API 与 SDK 负责“产生 telemetry”，那么 Collector 更像一个独立的 **telemetry pipeline 运行时**。它位于应用与后端之间，负责接收、处理、变换、过滤、采样、路由并导出信号。

因此，Collector 的价值并不是“少写两行 exporter 配置”这么简单，而是把很多原本散落在应用里的运行时决策收敛到一个更适合统一治理的位置。

Collector 常见可以承担的工作包括：

- 接收 OTLP、Prometheus scrape 或其他输入协议
- 对 traces、metrics、logs 做批处理和内存保护
- 删除敏感字段、补充资源属性、规范化数据结构
- 进行 tail sampling、过滤无价值流量、路由到不同后端
- 统一出口，减少应用直接依赖多个观测系统

这也是为什么在生产环境中，团队往往更推荐：

- 应用尽量只负责生成 telemetry 并发送到近端 Collector
- Collector 负责更复杂的数据管道控制

## 2. Collector 的四个核心概念

### 2.1 receiver：数据从哪里来

receiver 负责接收数据输入。例如：

- `otlp`：接收来自应用 SDK 的 traces、metrics、logs
- `prometheus`：主动抓取暴露的指标端点
- `filelog`：从文件中读取日志
- `hostmetrics`：采集主机层指标

一个常见的 OTLP receiver 配置如下：

```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:
```

这表示 Collector 可以同时监听 OTLP/gRPC 与 OTLP/HTTP。

### 2.2 processor：数据在中间怎么处理

processor 位于 pipeline 中间，负责处理进入 Collector 的 telemetry。典型用途包括：

- `batch`：把小批量数据合并后再导出，提高效率
- `memory_limiter`：防止 Collector 内存失控
- `resource`：统一添加或修改资源属性
- `filter`：过滤不需要的数据
- `transform`：变换属性与字段
- `tail_sampling`：按完整 trace 结果做采样决策

### 2.3 exporter：数据往哪里去

exporter 负责把 telemetry 送到下游目标。例如：

- 下一个 Collector
- OTLP 兼容后端
- 调试输出
- 其他协议接收端

教学阶段经常先用 `debug` exporter，因为它能让你直观看到 Collector 到底收到了什么。

### 2.4 pipeline：把三者串起来

pipeline 是 Collector 的核心组织方式。它把某类信号的 receiver、processor、exporter 串成一条流水线。

一个最小示例如下：

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

exporters:
  debug:

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [debug]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [debug]
```

这段配置表达的是：

- 应用通过 OTLP 发来 traces 和 metrics
- Collector 先做内存保护，再做批处理
- 最后打印到 debug 输出

## 3. Agent 模式与 Gateway 模式

Collector 最重要的部署抽象之一，是 **agent** 与 **gateway** 两种模式。理解这两个模式，基本就理解了很多生产部署选择题。

### 3.1 Agent Collector

agent 模式指的是：Collector 部署得尽量靠近应用实例本身，例如：

- 与应用在同一个节点上
- 作为 DaemonSet 跑在每个 Kubernetes 节点上
- 作为 sidecar 附近部署在每个 Pod 内
- 作为本机进程运行在宿主机上

agent 的优势：

- 离应用近，网络路径短
- 有助于吸收瞬时抖动和本地缓冲
- 应用配置通常更统一，只需发到本地地址
- 可以更方便地采集节点、本机、容器邻近信息

agent 的代价：

- 实例数量多，运维面更大
- 配置升级与版本管理更分散
- 如果每个 agent 都做重处理，整体资源成本会提高

### 3.2 Gateway Collector

gateway 模式指的是：Collector 作为一个共享的集中入口，接收多个应用或多个 agent 发来的 telemetry，再统一处理并转发到后端。

gateway 的优势：

- 更适合做集中治理，如统一 routing、sampling、脱敏、出口认证
- 便于控制配置版本和出口依赖
- 更容易实现多后端分发和租户级策略

gateway 的代价：

- 它是更显著的集中组件，需要考虑高可用与扩容
- 距离应用更远，网络抖动影响更直接
- 如果设计不当，容易成为瓶颈或单点压力中心

### 3.3 什么时候用 agent，什么时候用 gateway

最常见的实践不是二选一，而是两层组合：

- 应用先发给近端 agent
- agent 做轻处理后转发到 gateway
- gateway 做集中治理后再导出到后端

这样做的原因是：

- 近端 agent 负责吸收本地连接与部署差异
- 中央 gateway 负责统一管道治理

你也可以简单地理解为：

- agent 更接近“接入层”
- gateway 更接近“治理层”

## 4. 一个更完整的 Collector 示例

下面这个 YAML 比最小示例更接近实际教学中的基础模板：

```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 512
  resource:
    attributes:
      - key: telemetry.pipeline
        value: gateway
        action: upsert
  batch:
    send_batch_size: 1024
    timeout: 5s

exporters:
  otlp:
    endpoint: backend.example.local:4317
    tls:
      insecure: true
  debug:
    verbosity: basic

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, resource, batch]
      exporters: [debug, otlp]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, resource, batch]
      exporters: [debug, otlp]
```

这里新增了几个实践点：

- 用 `resource` processor 为进入 gateway 的数据统一补充标识
- 同时导出到 `debug` 与 `otlp`，便于教学和排查
- `batch` 控制导出粒度，提高吞吐效率

注意，这些配置虽然常见，但并不意味着“所有处理都应该丢给 Collector”。例如业务阶段 span 仍然应该由应用负责，而不是指望 Collector 凭空生成正确语义。

## 5. Collector 为什么不是“只是一个转发器”

很多团队在早期只把 Collector 当成 exporter 中转层，这种理解太窄。Collector 真正重要的地方在于，它把大量运行时治理能力从应用中抽离出来，使应用能更专注于：

- 生成正确语义
- 保持最小 vendor-neutral 接入
- 把数据送到近端统一入口

而把下列事情集中交给 Collector：

- 批处理与内存保护
- 字段过滤与脱敏
- 数据整形与路由
- 采样与多后端导出
- 运维级的统一配置管理

这种分层的好处是非常实际的。否则每个应用都得自己处理：

- 导出重试
- 路由多后端
- 条件过滤
- 不同环境不同 endpoint
- 安全策略与证书管理

结果就是应用代码越来越重，也越来越难统一治理。

## 6. 使用 Collector 时的几个边界意识

### 6.1 Collector 不替代应用里的语义建模

Collector 可以改字段、删字段、加字段，但它通常不该替代应用去决定真正的业务 span 结构。

### 6.2 Collector 不自动解决错误 schema

如果应用里已经把高基数属性设计错了，Collector 顶多做止损，不代表上游设计没问题。

### 6.3 Collector 本身也是要观测的系统组件

在生产中，Collector 自己的 CPU、内存、队列积压、导出失败、丢弃量，都应该被持续观测。否则你只是在系统中间又加了一个黑箱。

### 6.4 先做最小闭环，再做复杂 pipeline

很多初学者一上来就配置十几个 processor，结果连最基本的数据流都没验证清楚。更好的顺序是：

1. 先让应用把 OTLP 数据成功发到 Collector
2. 再确认 Collector 能正确导出到 debug 或后端
3. 最后逐步加入 batch、resource、filter、sampling 等治理能力

## 本章小结

| 主题 | 结论 |
|------|------|
| Collector 的定位 | 独立的 telemetry pipeline 运行时，而不只是转发器 |
| receiver | 负责接收数据输入 |
| processor | 负责过滤、变换、批处理、采样和保护 |
| exporter | 负责把数据送往下游目标 |
| agent 模式 | 靠近应用，适合接入层与近端缓冲 |
| gateway 模式 | 集中治理，适合统一路由、采样和出口控制 |
| 推荐理解 | agent 更偏接入，gateway 更偏治理 |

## OTel实验

### 实验目标

通过最小 YAML 配置直观看懂 Collector 的基础数据流，并比较 agent 与 gateway 的角色差异。

### 实验步骤

1. 配置一个最小 Collector，只启用 `otlp` receiver、`batch` processor 和 `debug` exporter。
2. 让本地 Node.js 应用把 traces 和 metrics 发到该 Collector。
3. 观察 debug 输出，确认 receiver 与 pipeline 工作正常。
4. 再新增一个第二层 Collector，模拟 gateway，让第一层 Collector 把数据转发过去。
5. 在第二层 Collector 上用 `resource` processor 给所有数据补 `telemetry.pipeline=gateway`。
6. 对比单层与双层模式下的数据流理解差异。

### 预期现象

- 单层模式更容易帮助你理解基本 pipeline 结构
- 双层模式更容易理解 agent 与 gateway 的职责分工
- 在 gateway 层增加统一属性后，更容易看出集中治理的价值

## 练习题

1. 请分别解释 receiver、processor、exporter、pipeline 的职责，并各举一个例子。
2. 为什么说 Collector 是治理层而不只是转发器？请列出至少三种典型治理能力。
3. agent 与 gateway 各自更适合解决什么问题？为什么很多生产系统会同时使用二者？
4. 如果一个团队把所有复杂处理都塞进每个节点本地 agent，会带来哪些可能的问题？
5. 为什么在学习 Collector 时，应该先做最小闭环，再逐步增加 processor？
