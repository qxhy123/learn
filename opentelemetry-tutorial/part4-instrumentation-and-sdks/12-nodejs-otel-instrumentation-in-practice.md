# 第12章：用 Node.js / TypeScript 接入 OTel

## 学习目标

- 在 Node.js / TypeScript 服务中建立一个最小可运行的 OpenTelemetry 接入方案
- 理解启动顺序为什么会影响自动 instrumentation 是否生效
- 学会同时接入 traces、metrics 与结构化日志关联的基本做法
- 知道如何为 HTTP 服务、出站请求和数据库访问补充关键业务语义
- 识别本地开发与生产环境中常见的 Node.js OTel 接入坑点

## 1. 一个最小可运行方案需要哪些组件

在 Node.js 中，最常见的最小闭环不是“把所有包都装上”，而是把几层责任接通：

1. 应用启动前初始化 SDK
2. 启用自动 instrumentation 覆盖通用组件
3. 在业务代码里手动补充关键 span
4. 通过 OTLP 把 telemetry 发给 Collector
5. 由 Collector 再转发到后端或直接先用 debug exporter 观察输出

一个典型的目录结构可能如下：

```text
src/
  telemetry.ts
  server.ts
  routes/
    orders.ts
```

其中：

- `telemetry.ts` 负责启动 SDK
- `server.ts` 负责最先加载 telemetry 初始化逻辑
- `routes/orders.ts` 负责业务处理与手动 span

在真实项目里，最容易踩的第一个坑就是：**telemetry 初始化得太晚**。如果你先 `import express from 'express'`，再去启动 SDK，部分自动 patch 可能就错过了加载时机。

## 2. 启动层初始化 SDK

下面先给出一个偏教学用途、但结构清晰的 `telemetry.ts` 示例。

```ts
import { NodeSDK } from '@opentelemetry/sdk-node'
import { resourceFromAttributes } from '@opentelemetry/resources'
import { ATTR_SERVICE_NAME, ATTR_SERVICE_VERSION } from '@opentelemetry/semantic-conventions'
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http'
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http'
import { PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics'
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node'

export function startTelemetry() {
  const sdk = new NodeSDK({
    resource: resourceFromAttributes({
      [ATTR_SERVICE_NAME]: 'checkout-service',
      [ATTR_SERVICE_VERSION]: '1.0.0',
      'deployment.environment.name': process.env.NODE_ENV ?? 'development',
    }),
    traceExporter: new OTLPTraceExporter({
      url: process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT ?? 'http://localhost:4318/v1/traces',
    }),
    metricReader: new PeriodicExportingMetricReader({
      exporter: new OTLPMetricExporter({
        url: process.env.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT ?? 'http://localhost:4318/v1/metrics',
      }),
      exportIntervalMillis: 10000,
    }),
    instrumentations: [
      getNodeAutoInstrumentations({
        '@opentelemetry/instrumentation-http': {
          ignoreIncomingRequestHook(req) {
            return req.url === '/healthz'
          },
        },
      }),
    ],
  })

  sdk.start()

  process.on('SIGTERM', async () => {
    await sdk.shutdown()
  })

  return sdk
}
```

这个启动文件体现了几个关键点：

- Resource 用来声明服务身份，而不是把服务名写到每个 span 属性里
- traces 与 metrics 可以分别指定 OTLP endpoint
- 健康检查路径通常应排除，避免制造噪音
- 在优雅退出时调用 `shutdown()`，减少进程结束前数据丢失

## 3. 启动顺序：为什么要先初始化 telemetry

Node.js 项目中，自动 instrumentation 很依赖模块加载顺序。一个实用原则是：**在应用导入框架、数据库驱动、HTTP 客户端之前，先完成 SDK 初始化**。

例如，在 ESM 场景下，一个更稳妥的入口写法是：

```ts
import { startTelemetry } from './telemetry'

startTelemetry()
await import('./app-bootstrap')
```

而把真正导入框架与路由的逻辑放在被延迟加载的模块中：

```ts
import express from 'express'
import { ordersRouter } from './routes/orders'

const app = express()
app.use(express.json())
app.use('/orders', ordersRouter)

app.get('/healthz', (_req, res) => {
  res.json({ ok: true })
})

app.listen(3000, () => {
  console.log('server listening on :3000')
})
```

如果顺序写反，可能会出现以下现象：

- Express 请求进来了，但自动生成的 server span 不完整
- `http` 或数据库客户端没有被成功 patch
- 同样的代码在本地有 span，在生产某些打包方式下却没有

因此，Node.js 的 OTel 接入不是“只看配置”，还要看**模块初始化时机**。

## 4. 在业务代码里补充关键 span 与 metrics

自动 instrumentation 能覆盖很多基础路径，但你仍然应该在业务层补充真正重要的流程语义。

下面是一个简化的订单路由示例：

```ts
import express from 'express'
import { metrics, trace, SpanStatusCode } from '@opentelemetry/api'

const router = express.Router()
const tracer = trace.getTracer('checkout-service', '1.0.0')
const meter = metrics.getMeter('checkout-service', '1.0.0')

const orderSubmitCounter = meter.createCounter('orders_submitted_total', {
  description: 'Total number of submitted orders',
})

const orderLatency = meter.createHistogram('checkout_submit_duration_ms', {
  description: 'Duration of order submission',
  unit: 'ms',
})

router.post('/', async (req, res) => {
  const start = Date.now()

  await tracer.startActiveSpan('checkout.submit_order', async (span) => {
    try {
      const { orderId, channel } = req.body as {
        orderId: string
        channel: 'web' | 'mobile'
      }

      span.setAttribute('sales.channel', channel)
      span.setAttribute('checkout.has_order_id', Boolean(orderId))

      await validateOrder(orderId)
      await reserveInventory(orderId)
      await chargePayment(orderId)

      orderSubmitCounter.add(1, { sales_channel: channel, result: 'success' })
      res.status(201).json({ ok: true })
    } catch (error) {
      span.recordException(error as Error)
      span.setStatus({ code: SpanStatusCode.ERROR, message: 'submit order failed' })
      orderSubmitCounter.add(1, { result: 'error' })
      res.status(500).json({ ok: false })
    } finally {
      orderLatency.record(Date.now() - start)
      span.end()
    }
  })
})

export const ordersRouter = router

async function validateOrder(orderId: string) {
  return orderId
}

async function reserveInventory(orderId: string) {
  return orderId
}

async function chargePayment(orderId: string) {
  return orderId
}
```

这个示例说明了三件事：

1. **span 用于解释流程和错误上下文**
2. **metric 用于稳定统计趋势与 SLA/SLO 观测**
3. **属性要尽量控制基数**，例如这里没有直接把 `orderId` 作为 metric attribute

这并不是说 `orderId` 永远不能出现，而是要区分：

- 某些高基数值偶尔适合出现在 trace span 中
- 但通常不适合作为 metric 标签长期聚合

## 5. 出站请求、数据库与日志关联

### 5.1 出站 HTTP 请求

如果你启用了常见的 Node.js 自动 instrumentation，`fetch`、`http`、`https` 或部分 HTTP 客户端通常会自动生成出站 span。但你仍然要注意：

- 某些封装层是否绕过了默认客户端
- 某些重试逻辑是否让 span 结构变得难理解
- 是否需要在业务 span 上补充“为什么发这个请求”的语义

例如，与其只看到一个出站 `/payments/charge` 请求，不如在上层业务 span 里明确它属于 `checkout.charge_payment` 阶段。

### 5.2 数据库访问

数据库自动 instrumentation 通常会给你带来：

- 查询时延
- 连接目标
- 操作类型
- 某些受控的语义属性

但要特别小心：

- 不要随意记录完整 SQL 文本，尤其是包含敏感数据时
- 不要把动态参数直接做成高基数字段
- 不要把数据库 span 当成唯一的业务解释层

### 5.3 日志与 trace 关联

本章不展开日志体系，但在 Node.js 里你至少应该理解一个原则：**日志要尽量带上当前 trace 上下文，方便排障时从日志跳回 trace，或从 trace 查到日志。**

一个简化示意如下：

```ts
import { context, trace } from '@opentelemetry/api'

export function logInfo(message: string, extra: Record<string, unknown> = {}) {
  const span = trace.getSpan(context.active())
  const spanContext = span?.spanContext()

  console.log(JSON.stringify({
    level: 'info',
    message,
    trace_id: spanContext?.traceId,
    span_id: spanContext?.spanId,
    ...extra,
  }))
}
```

这不是完整日志方案，但它强调了一个重要思想：日志系统与 trace 不是彼此独立的两个世界。

## 6. 本地开发、容器与生产中的常见坑

### 6.1 本地能跑，生产没数据

常见原因包括：

- OTLP endpoint 配错
- 容器网络地址与本机地址不一致
- Collector 只开了 gRPC，你却用 HTTP exporter
- 启动顺序导致自动 instrumentation 没生效
- 进程退出太快，没有完成 flush/shutdown

### 6.2 所有 span 都有，但服务身份混乱

这通常是 Resource 没配好，或者多个服务都用了同一个 `service.name`。结果就是后端里链路混在一起，难以区分服务边界。

### 6.3 指标数量爆炸

往往是把高基数值放进 metric attributes，例如：

- `user_id`
- `order_id`
- 原始 URL 查询参数
- 动态错误文本

### 6.4 开了自动 instrumentation 但 trace 很吵

常见原因是没有过滤健康检查、管理接口、静态资源、内部探活流量。基础覆盖并不等于什么都要采。

### 6.5 只会抄代码，不知道如何验证

一个健康的验证路径通常是：

1. 先看应用日志，确认 SDK 启动成功
2. 再看 Collector debug exporter，确认 OTLP 数据已到达
3. 再看后端里 `service.name`、span 树、属性、错误状态是否符合预期
4. 最后再讨论采样、性能和成本优化

## 本章小结

| 主题 | 结论 |
|------|------|
| 最小接入闭环 | 启动 SDK、开启自动 instrumentation、补关键业务 span、经 OTLP 导出 |
| Node.js 关键坑点 | 启动顺序会直接影响自动 instrumentation 是否生效 |
| 业务层职责 | 用手动 span 和 metrics 补充真正关键的业务语义 |
| Resource 的意义 | 声明服务身份，避免把服务信息重复塞进每个 span |
| 日志关联原则 | 尽量把 trace_id 与 span_id 写入结构化日志 |
| 验证顺序 | 先看应用，再看 Collector，再看后端呈现 |

## OTel实验

### 实验目标

在本地做一个 Node.js 最小接入闭环，并验证自动 instrumentation 与手动埋点如何同时工作。

### 实验步骤

1. 创建一个 Express 服务，包含 `/healthz` 与 `POST /orders` 两个接口。
2. 在 `telemetry.ts` 中初始化 `NodeSDK`，开启自动 instrumentation，并把 traces、metrics 发往本地 Collector。
3. 在 `POST /orders` 里手动创建 `checkout.submit_order` span 和两个 metrics。
4. 配一个最小 Collector，把 OTLP 数据输出到 `debug` exporter。
5. 发送几次正常请求和一次故意失败请求，观察输出。
6. 刻意把 telemetry 初始化顺序调晚，再观察自动 instrumentation 是否失效。

### 预期现象

- 正常顺序下，可以同时看到 HTTP 基础 span 与业务 span
- 失败请求会在业务 span 上出现异常与错误状态
- 把初始化顺序调晚后，部分自动 span 会缺失或异常

## 练习题

1. 在 Node.js 中，为什么“先初始化 telemetry，再导入框架和驱动”很重要？
2. Resource 与 span attributes 分别更适合承载哪类信息？请各举两个例子。
3. 为什么 `order_id` 往往不适合作为 metric attribute，却可能在某些 trace 场景里仍然有价值？
4. 如果你发现本地有数据、容器环境没数据，你会按什么顺序排查？
5. 请说明自动 instrumentation、业务 span、结构化日志三者在排障中的分工关系。
