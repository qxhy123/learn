# 第3章：第一个端到端链路

## 学习目标

- 理解一个最小 OpenTelemetry 闭环需要哪些组件
- 明确应用/SDK -> Collector -> backend 或 debug exporter 的基本数据流
- 能读懂一个最小 Node.js + Collector YAML 示例在做什么
- 识别“数据没出来”时最常见的几个断点位置
- 建立先验证最小闭环、再讨论高级治理的工程习惯

## 1. 什么叫最小闭环

初学 OpenTelemetry 时，最容易犯的错误之一，是一上来就研究很多高级 processor、采样策略或平台拓扑，却还没有真正看懂一条最基础的数据流。

所谓最小闭环，指的是至少把下面这条链路走通一次：

```text
应用代码 / 自动 instrumentation
  -> OTel SDK
  -> OTLP exporter
  -> Collector
  -> backend 或 debug exporter
```

这条链路有两个关键目标：

1. 证明应用确实创建了 telemetry
2. 证明 telemetry 确实通过管道到达了下游

这里的“下游”既可以是真实 backend，也可以只是 Collector 的 `debug` exporter。对教学和排障来说，`debug` exporter 往往更直观，因为你不必先理解某个具体后端的查询界面，就能看到 Collector 实际收到了什么。

因此，本章的重点不是“搭一个完整观测平台”，而是先把最小数据流讲清楚。

## 2. 最小闭环里每一层分别做什么

如果你只记住一个图，请记住这一版：

```text
Node.js 应用
  -> NodeSDK
  -> OTLP/HTTP 或 OTLP/gRPC
  -> Collector receiver
  -> Collector processor
  -> Collector exporter
  -> backend / debug 输出
```

### 2.1 应用与 SDK：负责产生数据

应用负责：

- 创建 span、记录 metric、输出可关联日志
- 配置 service identity
- 初始化 SDK
- 把数据导出到 Collector 暴露的 OTLP 入口

这一步如果没有成功，后面所有配置都不会有数据可处理。

### 2.2 Collector：负责接收与处理中转

Collector 负责：

- 接收应用发来的 OTLP 数据
- 进行批处理、保护、过滤或变换
- 再导出到 debug exporter 或真实 backend

在最小闭环阶段，Collector 通常先做两件事就够了：

- `otlp` receiver
- `debug` exporter 或 `otlp` exporter

### 2.3 backend 或 debug exporter：负责证明数据到达了哪里

- 如果使用 `debug` exporter，你可以直接在 Collector 输出中看到 span 和 metric 的结构。
- 如果使用真实 backend，你可以继续验证资源属性、span 树和查询结果是否符合预期。

在教学顺序上，通常建议先看 `debug` exporter，再看 backend。这样可以把“Collector 有没有收到数据”和“后端有没有正确展示数据”这两个问题拆开。

## 3. 用 Node.js 建一个最小应用侧示例

下面给出一个偏教学用途的最小 TypeScript / Node.js 示例。重点不是把所有包名与工程细节讲全，而是帮助你看懂应用侧闭环最小需要什么。

### 3.1 telemetry.ts

```ts
import { NodeSDK } from '@opentelemetry/sdk-node'
import { resourceFromAttributes } from '@opentelemetry/resources'
import { ATTR_SERVICE_NAME } from '@opentelemetry/semantic-conventions'
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http'
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http'
import { PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics'
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node'

export function startTelemetry() {
  const sdk = new NodeSDK({
    resource: resourceFromAttributes({
      [ATTR_SERVICE_NAME]: 'checkout-service',
      'deployment.environment.name': 'development',
    }),
    traceExporter: new OTLPTraceExporter({
      url: 'http://localhost:4318/v1/traces',
    }),
    metricReader: new PeriodicExportingMetricReader({
      exporter: new OTLPMetricExporter({
        url: 'http://localhost:4318/v1/metrics',
      }),
      exportIntervalMillis: 10000,
    }),
    instrumentations: [getNodeAutoInstrumentations()],
  })

  sdk.start()
  return sdk
}
```

这段代码表达了几个最小前提：

- 这是一个应用，而不是通用库，所以它负责初始化 SDK
- 它给出了明确的 `service.name`
- traces 与 metrics 都通过 OTLP/HTTP 发往本地 Collector
- 开启自动 instrumentation，帮助先得到基础 HTTP span

### 3.2 server.ts

如果你使用的是 Node.js ESM，想确保自动 instrumentation 尽量早于框架模块加载，更稳妥的教学写法是把 telemetry 初始化放到独立入口文件最前面，再动态导入真正的服务器模块。

```ts
import { startTelemetry } from './telemetry'

startTelemetry()
await import('./server-app')
```

而真正承载 Express 逻辑的文件可以保持普通写法：

```ts
import express from 'express'
import { trace } from '@opentelemetry/api'

const tracer = trace.getTracer('checkout-service', '1.0.0')
const app = express()
app.use(express.json())

app.post('/checkout', async (_req, res) => {
  await tracer.startActiveSpan('checkout.submit_order', async (span) => {
    try {
      span.setAttribute('checkout.channel', 'web')
      await new Promise((resolve) => setTimeout(resolve, 50))
      res.status(200).json({ ok: true })
    } finally {
      span.end()
    }
  })
})

app.listen(3000, () => {
  console.log('server listening on :3000')
})
```

这里最值得注意的不是业务逻辑，而是两个工程点：

1. 在需要依赖模块加载时机的场景里，telemetry 初始化应尽量早于框架模块导入
2. 除了自动 instrumentation 产生的 HTTP 基础 span，又手动加了一个业务 span

这就构成了一个足够小、但已经有解释力的应用侧起点。

## 4. 用一个最小 Collector 让数据真正流起来

应用侧准备好之后，还需要一个 Collector 把数据接住。最小配置如下：

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
    verbosity: basic

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
```

这段 YAML 的意思非常直接：

- Collector 接受应用通过 OTLP 发来的 traces 和 metrics
- 中间只做最基础的 `batch` 处理
- 最后输出到 `debug` exporter

### 4.1 为什么这里先不用真实 backend

因为在学习最小闭环时，你最想先确认的是：

- 应用有没有发出来
- Collector 有没有收到
- Collector 收到的字段长什么样

`debug` exporter 有助于你把这些问题变得非常直接。等你确认数据已经正确到达 Collector，再把 exporter 改成真实 backend，会更容易定位问题。

### 4.2 如果要接真实 backend，数据流只多一步

如果你换成真实 backend，数据流通常会变成：

```text
应用/SDK
  -> OTLP exporter
  -> Collector
  -> OTLP exporter 或其他 exporter
  -> backend
```

本质上只是把“Collector 的输出目标”从 debug 改成了 backend，应用侧的 vendor-neutral 入口并没有改变。这就是 OTel 架构非常重要的一点：**应用尽量少感知后端差异，把后端耦合更多压到 Collector 与 exporter 层。**

## 5. 如何验证这个闭环真的成立

最小闭环不是“我写了配置”就算完成，而是你能按顺序证明每一层都正常。

### 5.1 第一步：确认应用确实在产生数据

你至少要确认：

- SDK 启动没有报错
- 请求 `/checkout` 后，确实执行到了手动 span 代码
- 自动 instrumentation 没有因为初始化时机过晚而失效

### 5.2 第二步：确认 Collector 真的接收到了 OTLP 数据

如果 `debug` exporter 中能看到：

- `service.name = checkout-service`
- HTTP 基础 span
- `checkout.submit_order` 业务 span

那么至少说明应用 -> SDK -> OTLP -> Collector 这段已经打通。

### 5.3 第三步：如果接真实 backend，再确认展示与查询层

这一步要重点检查：

- backend 中是否出现了正确的服务名
- span 树是否合理
- metrics 是否能按预期维度被查询
- 查询时间范围与字段名是否匹配

这样做的好处是，你不会把“应用没发出来”“Collector 没收进来”“后端没展示出来”混成一个模糊问题。

### 5.4 最小闭环阶段最常见的几个断点

- OTLP endpoint 配错
- Collector 没开对应协议或端口
- 应用先导入框架再初始化 SDK，导致自动 instrumentation 不完整
- `service.name` 没设，导致在后端里难以识别服务身份
- 进程退出太快，没有完成 flush 或 shutdown

这些问题都很常见，而且都属于“最小闭环没验证清楚”带来的后果。

## 6. 先建立最小闭环，再逐步进入复杂系统

很多人学习 OTel 时会被大量名词吓到，例如：

- resource detector
- tail sampling
- transform processor
- routing processor
- gateway topology
- 多后端导出

这些内容都重要，但它们的前提是你已经有了一个稳定、可验证的基础链路。

一个健康的学习和工程顺序通常是：

1. 先让应用产生最小 trace 和 metrics
2. 再让 Collector 接住并输出到 `debug`
3. 再切到真实 backend
4. 最后再逐步加入 resource、filter、sampling、routing 等治理能力

这个顺序看似慢，实际上更快。因为它把复杂系统拆成了可验证的小层次，而不是让你一次面对所有问题。

## 本章小结

| 主题 | 核心结论 |
|------|----------|
| 最小闭环定义 | 应用/SDK -> Collector -> backend 或 debug exporter 的完整数据流 |
| 应用侧职责 | 产生 telemetry、初始化 SDK、配置 service identity、发送 OTLP |
| Collector 侧职责 | 接收、批处理、导出，先保证最基础的数据流通 |
| debug exporter 价值 | 帮助直接验证 Collector 是否收到正确数据 |
| 验证顺序 | 先应用，再 Collector，再 backend |
| 学习建议 | 先跑通最小闭环，再进入复杂 pipeline 与生产治理 |

## OTel实验

### 实验目标

构建并验证一个最小端到端链路，亲眼看到应用产生的 telemetry 如何经过 Collector 流到下游。

### 实验步骤

1. 准备一个最小 Node.js 服务，包含一个 `POST /checkout` 接口。
2. 在应用启动层初始化 `NodeSDK`，把 traces 和 metrics 通过 OTLP 发往 `http://localhost:4318`。
3. 在业务代码中创建一个 `checkout.submit_order` span。
4. 启动一个只包含 `otlp` receiver、`batch` processor、`debug` exporter 的 Collector。
5. 发送几次请求，观察 Collector 输出中是否出现：
   - `service.name`
   - HTTP 自动 span
   - 业务 span
6. 再把 Collector 的 exporter 从 `debug` 改为真实 backend，比较“看 Collector 输出”和“看 backend 界面”两种验证路径的区别。

### 预期现象

- 你可以看到最小数据流确实是应用/SDK -> Collector -> 下游目标。
- 自动 instrumentation 与手动 span 会同时出现在同一条链路中。
- 当你把问题分层验证后，排障会明显比“直接看后端没数据”更清楚。

## 练习题

1. 为什么第 3 章强调最小闭环，而不是一开始就讨论复杂 Collector 拓扑？
2. 请用自己的话说明：应用/SDK -> Collector -> backend 这条数据流里，每一层最核心的职责是什么。
3. `debug` exporter 在教学和排障里为什么很有价值？
4. 如果应用能跑通，但 Collector 看不到数据，你会优先检查哪些位置？
5. 为什么说把后端耦合尽量放在 Collector 和 exporter 层，有利于保持 vendor-neutral？
