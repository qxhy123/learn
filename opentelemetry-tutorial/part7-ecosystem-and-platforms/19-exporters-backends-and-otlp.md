# 第19章：Exporter、Backend 与 OTLP 策略

## 学习目标

- 理解 OTLP 为什么是 OpenTelemetry 体系里的首选出口协议，以及它解决了什么工程问题
- 区分 exporter、backend、Collector 三者的职责边界，避免把它们混成一个概念
- 掌握应用直连 backend 与经 Collector 中转两种接入路径的优缺点
- 学会围绕厂商中立、可迁移性和生产治理设计导出策略
- 建立“先统一协议，再选择后端”的可观测性平台直觉

## 1. 先分清 exporter、backend、Collector 各自是什么

很多人第一次接触 OpenTelemetry 时，容易把 exporter、Collector 和 backend 混成同一层。实际上，它们在系统中负责的是不同问题。

- **exporter**：负责把 telemetry 发到某个下游目标
- **Collector**：负责接收、处理、变换、采样、路由并导出 telemetry
- **backend**：负责存储、索引、查询、可视化、告警与长期消费

如果只记一句话，可以记成：

- exporter 解决“怎么发出去”
- Collector 解决“怎么治理这条管道”
- backend 解决“数据到了之后如何被消费”

这三个角色经常一起出现，但边界不能混淆。比如：

- backend 不负责替应用做埋点设计
- Collector 不负责替 backend 提供查询语言和可视化
- exporter 不等于完整观测系统，它只是出口组件

把边界想清楚之后，很多架构选择题就没那么混乱了。

## 2. 为什么 OTLP 应该成为默认优先选择

OpenTelemetry 最大的工程价值之一，是尽量让应用与具体后端解耦。而 OTLP 正是这层解耦里最核心的协议之一。

### 2.1 OTLP 的价值不只是“官方协议”

OTLP 的实际意义在于，它让 traces、metrics、logs 可以通过一套统一的 OpenTelemetry 语义和传输方式被发送出去。

这带来几个重要好处：

- 应用更容易保持 vendor-neutral
- Collector 与 backend 之间更容易形成统一的数据通道
- traces、metrics、logs 三类信号可以共享一致的出口心智模型
- 团队迁移 backend 时，不必从应用代码层重写一整套私有接入

这也是为什么教程里一直强调：**尽量让应用先学会说 OTLP，再考虑它最终会被谁消费。**

### 2.2 OTLP 优先，不代表只能有一种后端

选择 OTLP，并不是说全世界只能有一个 backend，而是说：

- 上游采集尽量先统一到 OpenTelemetry 标准协议
- 下游可以根据组织需要选择不同后端
- 是否做二次转换，尽量放在 Collector 或平台层，而不是放在每个业务服务里

这和 API 设计有点像：你希望系统内部先说一种稳定语言，再决定边界外怎么映射。

### 2.3 一个 Node.js 指向 OTLP Collector 的最小示意

```ts
import { NodeSDK } from '@opentelemetry/sdk-node'
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http'

const sdk = new NodeSDK({
  traceExporter: new OTLPTraceExporter({
    url: 'http://localhost:4318/v1/traces',
  }),
})

sdk.start()
```

这个例子的重点不是具体用 HTTP 还是 gRPC，而是：应用只知道 OTLP 地址，而不是把某个专有后端的接入细节硬编码到服务里。

## 3. Backend 真正负责什么：不要把“看到图表”误当成 OTel 全部

backend 是观测系统里最接近用户消费的一层。它通常负责：

- 存储 traces、metrics、logs
- 为常用字段建立索引或聚合视图
- 提供查询、检索、仪表盘和告警
- 支持长期趋势分析、排障、容量分析或审计

但 backend 的强弱，并不改变 OpenTelemetry 本身的职责边界。

### 3.1 backend 的角色更偏“消费层”

不管 backend 是自建还是托管，它最终解决的是：

- 数据如何被检索和分析
- 哪些字段可以高效查询
- 不同信号怎样在界面中关联
- 保留多久、如何分层存储、如何做权限控制

也正因此，不同 backend 在体验上可能差异很大：

- 有的 trace 查询强
- 有的 metrics 聚合强
- 有的 logs 检索强
- 有的在跨信号关联、告警和租户治理上更成熟

但无论选择哪种 backend，都不应该反过来推翻上游的 OpenTelemetry 基础设计。否则一旦后端变化，你的应用会被产品特性深度绑死。

### 3.2 后端很强，也不能替代 schema 与 pipeline 设计

一些团队会产生错觉：

- “后端自己会做很多智能归一，所以应用怎么上报都行”
- “后端有自己的 agent，所以 Collector 可有可无”
- “后端查询很方便，所以字段名不统一也没关系”

这些想法短期看似省事，长期往往会导致：

- 应用接入变得不一致
- backend 迁移成本变高
- 告警和 dashboard 依赖私有字段
- 一旦要做统一治理，平台侧很难收口

## 4. 应用直连 backend，还是经 Collector 中转

这是 OpenTelemetry 架构里最常见的路径选择题之一。

### 4.1 直连 backend：简单，但边界更薄

“直连”指的是应用 SDK 直接把数据发送到某个 backend 或其接收端。

它的优点很明显：

- 路径最短，少一层中间组件
- 小规模场景中部署更简单
- 早期试验或开发环境上手快

适合的情况通常包括：

- 个人实验环境
- 单服务或小规模系统
- 暂时不需要集中治理的 PoC 阶段

但直连的问题也很实际：

- 每个应用都要知道 backend 地址、协议、认证方式
- 多后端导出会把复杂度压回应用
- 统一脱敏、routing、sampling 更难集中实施
- 一旦切换后端，变更面会扩散到大量服务

### 4.2 经 Collector 中转：多一层，但治理能力大幅增强

经 Collector 中转的典型路径是：

- 应用 -> OTLP -> Collector -> OTLP 或其他 exporter -> backend

它的好处是：

- 应用统一只依赖 OTLP 和近端 Collector
- 平台可以集中做 batch、filter、transform、sampling、routing
- 更容易做多后端 fan-out、双写、灰度切换和安全出口控制
- 有利于平台侧统一观测 schema 和接入规范

一个简单示意配置如下：

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
  batch:
    timeout: 5s

exporters:
  otlp/backend:
    endpoint: backend.example:4317
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlp/backend]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlp/backend]
```

这个模型的关键价值，不是“多了一跳”，而是“多了一层可治理的中间面”。

### 4.3 两种模式的核心权衡

| 维度 | 应用直连 backend | 经 Collector 中转 |
|------|------------------|-------------------|
| 接入简单度 | 小规模场景更简单 | 初期多一层组件 |
| 应用耦合 | 更容易耦合具体后端 | 更容易保持 OTLP 边界 |
| 统一治理 | 较弱 | 较强 |
| 多后端/迁移 | 应用改动面大 | 更容易平台侧处理 |
| 采样/路由/脱敏 | 分散在应用中，难统一 | 更适合集中实施 |

因此，一个很实用的经验是：

- 学习和实验时可以接受直连
- 一旦进入团队协作和生产治理，就更值得把 Collector 纳入架构

## 5. Exporter 策略：先统一出口语言，再讨论多后端和迁移

当系统逐渐复杂后，真正关键的问题不再是“有没有 exporter”，而是“出口策略是什么”。

### 5.1 先把应用出口统一成 OTLP

最健康的起点通常是：

- 应用只说 OTLP
- Collector 负责下游映射与出口治理

这样做的最大好处是：应用层的改动最少，平台层的演进空间最大。

### 5.2 多 backend 不等于让应用装多个 exporter

如果一个团队同时需要：

- 一个主观测后端
- 一个归档或审计后端
- 一个迁移期目标后端

更推荐的做法通常不是在每个应用里配置多个 exporter，而是在 Collector 做 fan-out 或阶段性双写。

示意：

```yaml
exporters:
  otlp/main:
    endpoint: main-backend.example:4317
    tls:
      insecure: true
  otlp/migration:
    endpoint: migration-backend.example:4317
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/main, otlp/migration]
```

这样迁移时更容易：

- 控制双写范围
- 灰度切换
- 比较新老后端结果
- 在不改应用代码的情况下完成出口切换

### 5.3 让 backend 成为“可替换组件”，而不是“架构中心”

真正成熟的平台思路，不是让所有服务围绕某个 backend 私有能力组织，而是：

- 以 OpenTelemetry 数据模型为基础
- 以 OTLP 为主出口协议
- 让 backend 成为可替换的消费层

这样即使未来要：

- 切换产品
- 双写迁移
- 调整保留策略
- 分离不同信号的存储后端

上游应用和大部分 Collector 接入逻辑也不需要被整体重写。

## 6. 设计 OTLP 优先策略时的几个工程直觉

### 6.1 先问“谁最容易被绑死”

通常最不应该被绑死的是业务应用。因为：

- 数量最多
- 变更成本最高
- 团队分布最广

所以把厂商差异尽量留在 Collector 和平台层，长期看更稳。

### 6.2 不要为了短期方便，牺牲长期一致性

很多系统一开始看似简单：

- A 服务直连后端 X
- B 服务直连后端 Y
- C 服务通过某个专有 agent 送日志

短期都能跑，长期就会出现：

- 字段不一致
- dashboard 难统一
- 迁移成本高
- 平台侧很难形成统一接入规范

OTLP 优先的真正意义，就是先把这条路径收束起来。

### 6.3 “厂商中立”不是反对 backend，而是避免上游过早耦合

vendor-neutral 并不意味着不能使用强大的 backend，也不意味着要回避任何产品特性。它真正强调的是：

- 应用和基础埋点尽量不依赖私有协议
- 组织级治理尽量基于开放模型和标准出口
- backend 的增强能力应建立在稳定的 OpenTelemetry 基础之上

## 本章小结

| 主题 | 核心结论 |
|------|----------|
| exporter 的职责 | 把 telemetry 发往下游目标 |
| Collector 的职责 | 负责接收、处理、采样、路由和统一导出 |
| backend 的职责 | 负责存储、查询、可视化、告警和长期消费 |
| OTLP 优先 | 有助于保持 vendor-neutral 和上游解耦 |
| 应用直连 backend | 上手简单，但更容易把后端差异压回应用 |
| 经 Collector 中转 | 多一层组件，但显著增强治理和迁移能力 |
| 推荐策略 | 应用优先说 OTLP，平台层再决定下游 backend |

## OTel实验

### 实验目标

对比“应用直连 backend 风格路径”和“应用 -> Collector -> backend 路径”，观察两者在配置、迁移和治理上的差异。

### 实验步骤

1. 准备一个本地 Node.js 服务，先让它直接通过 OTLP exporter 发往一个测试接收端。
2. 记录应用侧必须显式知道哪些信息，例如地址、协议、认证、环境差异。
3. 然后在中间加入一个最小 Collector，让应用改为只发到本地或近端 Collector。
4. 在 Collector 中加入 `batch`，再尝试增加第二个 exporter，模拟双写或迁移。
5. 比较两种路径下：
   - 应用配置复杂度
   - 切换下游目标时的改动范围
   - 是否容易集中做脱敏、routing 和采样

### 预期现象

- 直连路径更容易快速跑通，但后端变化会更直接影响应用
- 经 Collector 中转后，应用配置更稳定，平台更容易做统一治理
- 当需要双写、迁移或改出口时，Collector 路径的变更面更集中

## 练习题

1. 为什么说 exporter、Collector、backend 三者虽然相邻，但职责边界完全不同？
2. OTLP 为什么适合作为默认优先出口协议？它最重要的工程价值是什么？
3. 在什么情况下应用直连 backend 是可以接受的？一旦进入生产，它最容易暴露哪些问题？
4. 为什么多 backend 导出通常更适合在 Collector 做，而不是让每个应用自己配置多个 exporter？
5. 请解释“让 backend 成为可替换组件，而不是架构中心”这句话的含义。
