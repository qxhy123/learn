# 前言：如何使用本教程

## 教程设计理念

OpenTelemetry 不是“会把 SDK 装上并把数据发出去”就结束的技能。真正掌握 OpenTelemetry，需要同时建立三层理解：

1. **信号直觉**：traces、metrics、logs 分别解决什么问题，为什么必须协同
2. **管道直觉**：应用、SDK、Collector、backend 各自负责什么，为什么数据会在这些层之间流动
3. **生产直觉**：什么时候该自动注入，什么时候该手动埋点，什么时候该控制 cardinality、采样和成本

本教程围绕这三层能力展开，遵循以下设计原则：

1. **从零开始，但不绕远路**：默认你没有 OpenTelemetry 经验，但不会把教程写成一整套后端基础课
2. **先建立结构，再进入产品实现**：先讲 traces、metrics、logs、propagation、Collector 的边界，再讲具体接入方式
3. **厂商中立**：优先解释 OpenTelemetry 自身的概念、规范和工程边界，而不是某个观测产品的私有能力
4. **实验驱动**：重要概念尽量通过配置片段、TypeScript / Node.js 示例和 Collector pipeline 小实验说明
5. **贴近真实工程**：不仅讲 instrumentation，还讲 schema 设计、采样、成本、安全、治理、迁移和排障
6. **中文优先**：尽量把术语说清，把设计因果讲透，把常见误区点明

## 为什么很多人“用过 OTel”，却还没真正理解它？

很多人接触 OpenTelemetry 的方式是：

- 安装一个自动注入包
- 看到后端里出现了几条 trace
- 再配一个 Collector 把数据转发出去
- 然后觉得自己“已经接上了 OTel”

结果通常是：

- trace 有时候断、有时候连得起来，但说不清为什么
- metrics 发出去了，却不知道 instrument 选型是否合理
- logs 和 traces 无法关联，排障时仍然要手工翻日志
- attributes 越加越多，最后 cardinality 和成本一起失控
- 看到 receiver、processor、exporter 配置，但不清楚哪一层该负责什么
- 知道 OTLP、Semantic Conventions、Baggage、sampling 这些词，却不知道它们在系统里真正解决什么问题

本教程的目标不是让你“见过很多名词”，而是让你在面对一个新的系统时，能系统地问出并回答下面这些问题：

1. 这个系统真正需要哪些观测信号？
2. 哪些信息应该放在 span、metric、log、resource 或 baggage 中？
3. 应该优先自动注入、手动埋点，还是两者混合？
4. Collector 应该部署在 agent、gateway、sidecar 还是 daemonset 位置？
5. 哪些属性能帮助诊断，哪些属性会把成本炸掉？
6. 当前问题出在应用埋点、传播断裂、Collector pipeline，还是 backend 查询层？

## 章节结构

每章采用相对统一的结构，便于系统学习：

### 1. 学习目标
每章开头列出 5 个学习目标，帮助你在开始前明确重点。

### 2. 正文内容
正文一般包含 4-6 个部分，通常覆盖：
- 概念与直觉
- 规范与数据模型
- 配置或代码片段
- 常见错误与注意事项
- 工程实践建议

### 3. 本章小结
以表格或要点形式回顾本章的核心概念与常见结论。

### 4. OTel实验
每章都会设计至少一个实验，重点不是“把一段配置跑通”，而是让你观察：
- trace 是如何被创建、传播和导出的
- metric 的含义和聚合方式是否真的正确
- logs 是否能和 trace 关联
- Collector pipeline 改动后，数据会发生什么变化
- 采样、属性设计和部署方式为什么会影响成本与诊断能力

### 5. 练习题
每章练习题分为三类：
- **基础题**：验证概念是否真正理解
- **实现题**：要求补全配置、设计埋点或写伪代码
- **思考题**：要求你解释设计权衡、生产边界或排障路径

### 6. 答案提示
完整答案汇总在附录中，鼓励你先独立思考，再回头看答案要点。

## 示例约定

本教程中的代码和配置主要使用以下两类形式：

### TypeScript / Node.js 示例
用于说明：
- HTTP 服务接入
- middleware 与 context propagation
- 手动 span / metrics / logs 设计
- 应用与 SDK 的关系

```ts
import { trace } from '@opentelemetry/api'

const tracer = trace.getTracer('demo-service')

export async function handleRequest() {
  return tracer.startActiveSpan('handle_request', async (span) => {
    try {
      span.setAttribute('app.feature', 'demo')
      return { ok: true }
    } finally {
      span.end()
    }
  })
}
```

### Collector YAML 示例
用于说明：
- receiver / processor / exporter 关系
- pipeline 结构
- 过滤、批处理、采样、路由等配置模式

```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:

processors:
  batch:

exporters:
  debug:

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
```

约定说明：

- 配置片段优先强调结构和意图，不追求把所有环境变量和后端细节一次讲完
- 首次出现的重要字段，会解释其职责与边界
- 示例会尽量保持最小闭环，避免把一章写成完整工程脚手架

## 输出格式说明

代码输出示例统一采用如下形式：

```ts
console.log('trace exported')
// 输出: trace exported
```

命令行输出示例统一采用如下形式：

```bash
docker compose up collector
node app.js
# 输出: OTLP exporter connected
```

## 学习建议

### 对完全没有 OpenTelemetry 经验的学习者

1. 严格按顺序学习第 1-9 章，不要跳过 signals、propagation 和 semantic conventions
2. 每看到一个 span / metric / resource 字段，都先问自己“它为什么放在这里，而不是别处”
3. 一旦 trace 断了，优先怀疑传播边界、上下文作用域和 Collector 配置，而不是“后端看起来没数据”
4. 在开始做高级 pipeline 和采样前，先确保最小链路数据正确且可解释

### 对后端工程师

1. 不要只停留在“自动注入能出 trace”这一层
2. 重点关注第 5-15 章，这些章节决定你是否真正能把 OTel 接进服务并长期维护
3. 学第 16-24 章时，把 telemetry schema、成本、Collector 拓扑和你的现有系统联系起来

### 对平台与 SRE / 可观测性工程师

1. 每章都尽量做对照理解：应用侧责任 vs Collector 责任 vs backend 责任
2. 记录属性设计、采样策略、processor 使用和部署拓扑之间的关系
3. 不要只看“数据进没进去”，要看“这套数据结构是否能长期支持排障与治理”

## 与本仓库其他教程的关系

本教程在仓库中的定位更偏“观测系统底层与工程治理”。你可以把它和其他系列一起使用：

- **TypeScript 教程**：帮助你理解 Node.js 后端、middleware 和完整项目结构，从而更容易看懂 instrumentation 示例
- **AI Infra 教程**：帮助你理解 traces、metrics、logs 在平台可靠性、容量规划和治理中的更高层位置
- **计算机网络教程**：帮助你理解请求跨代理、TLS、网关和异步边界时，传播为什么容易断

如果你已经在使用日志平台、监控平台或 tracing backend，但对 OpenTelemetry 为什么能把这些能力串起来仍然感觉像黑箱，那么本教程正好负责打开这个黑箱。

## 常见误区

### 误区一：只要接上自动注入，就算学会了 OpenTelemetry
不对。自动注入只是入口，真正难的是 schema、propagation、pipeline 和生产治理。

### 误区二：OpenTelemetry 只等于 tracing
不对。它同时覆盖 traces、metrics、logs，以及它们之间的关联能力。

### 误区三：Collector 只是一个转发器
不对。它还是处理、过滤、采样、变换和路由的 telemetry pipeline 组件。

### 误区四：属性越多越好
不对。高 cardinality 会直接影响存储成本、查询性能和系统稳定性。

### 误区五：厂商后端已经很强，就不需要理解 OTel 规范
不对。只有理解规范边界，才能避免被产品实现细节牵着走。

---

*准备好了吗？接下来我们先回答一个最根本的问题：为什么今天的分布式系统越来越倾向于用 OpenTelemetry 作为统一观测入口？*

[下一章：为什么需要 OpenTelemetry](./part1-foundations/01-why-opentelemetry.md)
