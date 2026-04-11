# 第10章：API、SDK 与 Library Guidelines

## 学习目标

- 理解 OpenTelemetry API、SDK、Collector 与 backend 之间的职责边界
- 说清楚为什么应用通常负责初始化 SDK，而库通常只依赖 API
- 区分业务库、自动 instrumentation 库与应用代码三类角色
- 能在 TypeScript / Node.js 中写出厂商中立、可复用的 telemetry 接入方式
- 识别重复初始化、写死 exporter、属性设计失控等常见反模式

## 1. 为什么 OpenTelemetry 要把 API 与 SDK 分开

很多初学者第一次接触 OpenTelemetry 时，会把“写 span”“导出 trace”“设置采样率”“配置 OTLP 地址”看成一件事。实际上，OTel 故意把这些能力拆成了不同层级，其中最重要的分界就是 **API** 与 **SDK**。

把它们分开的根本原因有三个：

1. **让库代码不依赖具体后端**：一个通用库不应该因为使用了某个 tracing 产品，就把所有使用者一起绑到该产品上。
2. **让应用保留最终控制权**：采样、导出、资源属性、processor、采集地址，本质上都属于应用或平台的运行时决策，而不是库的内部实现细节。
3. **让未接入 telemetry 的场景也能正常运行**：只依赖 API 的代码在没有安装 SDK 时通常会退化为 no-op，不应影响业务功能。

这意味着：

- API 更像一组“约定好的接口”和“全局入口”
- SDK 更像“真正把数据采集、处理、导出出去的实现”

如果把两者混在一起，最常见的结果就是：库偷偷初始化全局 provider、写死 exporter 地址、强行决定采样率，最后导致应用层根本无法统一治理。

## 2. API 负责什么，SDK 负责什么

可以先建立一个最重要的判断标准：**API 负责“怎么调用”，SDK 负责“怎么实现与导出”**。

| 层次 | 主要职责 | 典型内容 | 不该负责什么 |
|------|----------|----------|--------------|
| API | 定义访问入口与数据模型接口 | `trace.getTracer()`、`metrics.getMeter()`、context、propagation 接口 | exporter、采样器、批处理、真实导出 |
| SDK | 提供具体实现 | tracer provider、meter provider、span processor、metric reader、sampler、exporter | 业务代码里的语义建模 |
| Collector | 处理与转发遥测管道 | receiver、processor、exporter、routing、sampling | 替代应用里的业务语义埋点 |
| Backend | 存储、查询、展示、告警 | trace 查询、指标聚合、日志检索、看板 | 定义 OTel 规范本身 |

在 Node.js 里，这种分界通常表现为：

- 应用代码或库代码使用 `@opentelemetry/api`
- 启动阶段由应用装配 `@opentelemetry/sdk-node` 和具体 exporter
- Collector 再负责运行时处理、过滤、路由和转发

一个非常关键的事实是：**只有 API 没有 SDK 时，代码也应该尽量能继续运行**。这就是为什么很多库可以安全地调用 `trace.getTracer()`，即使使用者没有真正部署可观测性系统。

## 3. 应用、业务库与 instrumentation 库分别该做什么

在工程里，最容易混淆的不是 API 和 SDK 本身，而是“谁来接它们”。至少要区分下面三种角色：

### 3.1 应用代码

应用是最终运行单元，通常应该负责：

- 初始化 SDK
- 设置 `service.name` 等 Resource 信息
- 决定 exporter、采样、批处理与导出地址
- 控制自动 instrumentation 是否开启
- 决定是否把数据先发给本地 agent Collector，再发往 gateway Collector

### 3.2 业务库

业务库是被多个应用复用的代码模块，例如支付 SDK、库存库、鉴权库、数据库访问封装层。它更适合：

- 依赖 `@opentelemetry/api`
- 暴露稳定的 span 名称、事件与属性
- 记录与该库职责密切相关的领域语义
- 在没有 SDK 时优雅退化

它不适合：

- 初始化全局 SDK
- 配置 OTLP 地址
- 决定采样率
- 假设所有调用方都使用同一个 backend

### 3.3 instrumentation 库

instrumentation 库和业务库又不同。它的职责往往是：

- 自动 patch HTTP 客户端、Web 框架、数据库驱动、消息队列客户端
- 生成通用的基础 span 与基础属性
- 尽量少侵入业务代码

例如对 `http`、`express`、`pg`、`mysql`、`redis` 的自动埋点，本质上属于 instrumentation 库的工作，不是应用业务代码自己去手写所有基础 span。

## 4. Library Guidelines：为什么库通常只依赖 API

OpenTelemetry 的一个核心设计原则是：**库负责产出语义，应用负责接入实现**。因此，通用库最稳妥的做法通常是只依赖 API。

下面是一个更符合 Guidelines 的 TypeScript 示例：

```ts
import { metrics, trace, SpanStatusCode } from '@opentelemetry/api'

const tracer = trace.getTracer('inventory-lib', '1.0.0')
const meter = metrics.getMeter('inventory-lib', '1.0.0')

const reservationCounter = meter.createCounter('inventory_reservations_total', {
  description: 'Number of inventory reservation attempts',
})

export async function reserveInventory(
  quantity: number,
  channel: 'web' | 'mobile',
) {
  return tracer.startActiveSpan('inventory.reserve', async (span) => {
    reservationCounter.add(1, { sales_channel: channel })
    span.setAttribute('inventory.quantity', quantity)
    span.setAttribute('sales.channel', channel)

    try {
      // 省略真实库存逻辑
      return { reserved: true }
    } catch (error) {
      span.recordException(error as Error)
      span.setStatus({ code: SpanStatusCode.ERROR, message: 'reserve failed' })
      throw error
    } finally {
      span.end()
    }
  })
}
```

这个例子体现了几条重要原则：

1. `getTracer()` 与 `getMeter()` 使用的是库自己的名字和版本
2. 库只表达自己的语义，不决定如何导出
3. 属性尽量稳定、低基数、可复用
4. 发生异常时记录异常，但不吞掉错误

### 4.1 库里不应该做的事

下面这种写法通常是不推荐的：

```ts
import { NodeSDK } from '@opentelemetry/sdk-node'
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http'

const sdk = new NodeSDK({
  traceExporter: new OTLPTraceExporter({
    url: 'http://telemetry.example/v1/traces',
  }),
})

sdk.start()
```

问题不在于这段代码“不能运行”，而在于它把本应由应用决定的事情偷偷提前决定了：

- 你把库绑定到了某种 SDK 装配方式
- 你把 exporter 与地址写死了
- 你可能覆盖或干扰应用自己的全局 provider
- 你让一个通用库承担了运行时治理职责

这会直接破坏厂商中立和应用层统一配置能力。

## 5. 应用如何接住库产生的 telemetry

只要库依赖 API，应用就可以在启动时统一安装 SDK，把库里产生的 span 与指标真正接出来。

下面是一个应用启动层的典型例子：

```ts
import { NodeSDK } from '@opentelemetry/sdk-node'
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http'
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http'
import { PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics'
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node'

const sdk = new NodeSDK({
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
```

应用这样做有几个好处：

- 可以统一设置 `OTEL_SERVICE_NAME`、`OTEL_RESOURCE_ATTRIBUTES`
- 可以统一选择直连 backend 或先发给 Collector
- 可以统一决定是只开 traces，还是 traces + metrics 一起开
- 可以在不同环境使用不同采样率与导出目标

换句话说，**库里的 API 调用负责“产生语义”，应用里的 SDK 装配负责“让这些语义进入管道”**。

## 6. 常见错误与工程边界

### 6.1 错误一：业务库直接依赖 SDK

这样做通常会让复用代码失去中立性，也更容易出现多个 provider 重复初始化的问题。

### 6.2 错误二：把 exporter 当成库的一部分

exporter、endpoint、认证方式、TLS 策略都属于部署与运行时问题，不应由通用库偷偷决定。

### 6.3 错误三：把高基数业务标识直接塞进属性

例如用户 ID、订单号、原始 URL、完整 SQL、动态缓存 key 等，往往会让存储成本和查询性能快速恶化。库层尤其要克制，因为库一旦把坏 schema 扩散出去，所有上层应用都会受影响。

### 6.4 错误四：span 名称和属性名没有稳定命名规则

如果今天叫 `inventory.reserve`，明天改成 `reserve_inventory_action`，后天又加一套私有缩写，后续查询、告警和仪表盘都会越来越难维护。

### 6.5 错误五：认为“库只依赖 API”就完全不用思考语义设计

恰恰相反。库虽然不负责 SDK 与 exporter，但非常负责 span 名称、事件、状态码和属性 schema 的长期稳定性。**不绑后端，不等于不做设计。**

## 本章小结

| 主题 | 结论 |
|------|------|
| API 与 SDK 的区别 | API 定义调用接口，SDK 提供采集与导出实现 |
| 谁初始化 SDK | 通常应由应用或平台启动层负责 |
| 业务库怎么接 OTel | 优先依赖 API，只表达自身语义 |
| instrumentation 库的职责 | 自动 patch 通用框架与客户端，产出基础 telemetry |
| 为什么不能在库里写死 exporter | 会破坏应用统一治理、厂商中立和运行时控制 |
| 设计重点 | 稳定命名、低基数属性、清晰错误语义、优雅退化 |

## OTel实验

### 实验目标

验证“库只依赖 API，而应用决定是否接 SDK”这一设计是否真的成立。

### 实验步骤

1. 编写一个只依赖 `@opentelemetry/api` 的小型 TypeScript 库，在函数里创建一个自定义 span。
2. 在应用中先**不初始化 SDK**，直接调用这个库函数，观察业务功能是否依旧正常。
3. 再在应用启动层加入 `NodeSDK` 和 OTLP exporter，把数据发往本地 Collector 或 debug exporter。
4. 对比两次运行结果：第一次业务能跑但几乎没有可观测数据，第二次业务不改代码却能看到完整 telemetry。

### 预期现象

- 没有 SDK 时，API 调用通常退化为 no-op，不应影响业务返回值
- 一旦应用初始化 SDK，库里原有的 span 与指标就会被真实采集与导出
- 把配置放在应用层后，你可以不改库代码就切换不同环境与不同后端

## 练习题

1. 为什么一个通用支付库通常不应该在内部直接创建 `NodeSDK`？请从复用性、运行时治理和厂商中立三个角度回答。
2. API 与 SDK 的边界分别是什么？请分别举出两个“应该放在 API 层”和两个“应该放在 SDK 层”的能力。
3. 如果你维护一个库存业务库，你会如何命名 tracer、meter、span，以及哪些属性应该避免直接写入？
4. 什么情况下适合写业务库埋点，什么情况下更适合交给自动 instrumentation 库处理？
5. 设想一个团队要求所有库都写死 OTLP 地址，理由是“这样接入更省事”。请指出这种做法的三个长期问题。
