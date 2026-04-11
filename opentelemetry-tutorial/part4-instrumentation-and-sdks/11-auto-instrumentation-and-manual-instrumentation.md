# 第11章：自动注入、手动埋点与混合策略

## 学习目标

- 理解 zero-code、auto-instrumentation、manual instrumentation 三种接入方式的差异
- 知道自动注入擅长解决什么问题，又在哪些地方天然不够
- 能根据服务类型、团队成熟度和排障目标选择合适的混合策略
- 学会判断哪些 span、metrics 和 logs 必须由业务代码手动补充
- 识别重复埋点、上下文断裂、语义失真等常见接入问题

## 1. 什么是 zero-code、自动注入与手动埋点

在 OTel 语境里，很多人会把“自动注入”与“zero-code”混用，但从工程角度最好把它们区分开。

### 1.1 zero-code

zero-code 通常指的是：**尽量不改业务代码，通过启动参数、环境变量、注入 agent、预加载模块或运行时 patch 的方式获得基础 telemetry**。

它的目标是降低接入门槛，帮助团队先把 HTTP 请求、数据库调用、消息客户端等通用路径观测起来。

### 1.2 自动 instrumentation

自动 instrumentation 是 zero-code 的核心技术手段之一。它一般通过：

- patch Node.js 核心模块，如 `http`、`https`
- patch Web 框架，如 `express`、`koa`、`fastify`
- patch 数据库驱动，如 `pg`、`mysql2`、`mongodb`
- patch 消息中间件客户端，如 `amqplib`、Kafka 客户端等

自动 instrumentation 的价值在于：

- 统一生成基础 span
- 自动传播 context
- 降低漏埋点概率
- 帮助你快速拿到“请求穿过哪些基础组件”的链路图

### 1.3 手动 instrumentation

手动埋点指的是在业务代码里显式创建 span、记录事件、添加属性、更新 metrics、把 trace 上下文写入结构化日志等。

它更适合表达：

- 业务阶段边界
- 领域错误语义
- 关键判定分支
- 业务 KPI 或领域指标
- 自动 instrumentation 无法感知的内部处理过程

一句话总结：

- zero-code 解决“先看见”
- 自动 instrumentation 解决“通用路径自动覆盖”
- 手动 instrumentation 解决“业务语义真正可解释”

## 2. 自动注入能帮你解决什么，不能帮你解决什么

自动 instrumentation 的强项很明显，但边界也同样明显。

### 2.1 自动 instrumentation 的强项

#### 能快速建立最小闭环

对一个尚未系统接入可观测性的团队来说，最重要的第一步往往不是把所有业务语义都埋对，而是先拿到稳定的基础链路。自动 instrumentation 正适合完成这个目标。

#### 能减少重复劳动

如果每个服务都要手动为 HTTP server、HTTP client、数据库客户端写 span，团队会很快陷入样板代码和不一致命名中。自动库可以把这些基础工作标准化。

#### 能帮助发现传播问题

因为自动 instrumentation 通常会自动处理 context 传播，所以它非常适合暴露“为什么这个 trace 到这里断了”的问题。

### 2.2 自动 instrumentation 的天然边界

#### 它不理解你的业务阶段

自动库能知道你发起了一个 HTTP 请求，但不知道“风控校验”“库存预占”“优惠券核销”“支付确认”这些领域阶段意味着什么。

#### 它不应该决定高层业务语义

自动 instrumentation 生成的 span 名称通常偏通用，比如 `GET /orders/:id`、`postgres.query`。这对基础观测很有帮助，但无法单独支撑复杂排障与业务分析。

#### 它无法替代业务指标设计

自动库可以生成运行时指标和请求时延指标，但无法替你决定“支付成功率”“库存锁定冲突率”“结算重试次数”应该如何建模。

#### 它可能存在覆盖盲区

对于自定义队列封装、内部任务调度、跨线程或跨异步边界的特殊上下文管理，自动 patch 往往无法完全覆盖。

## 3. 手动埋点的核心价值：让 trace 变得可解释

如果说自动 instrumentation 给你的是“系统骨架”，那么手动埋点补上的就是“业务肌肉与神经”。

下面这个订单提交流程的示例，自动 instrumentation 只能看到 HTTP 请求和几个数据库调用，但手动埋点可以把真正重要的业务阶段显式建模出来。

```ts
import { trace, SpanStatusCode } from '@opentelemetry/api'

const tracer = trace.getTracer('checkout-service', '1.0.0')

export async function submitOrder(input: {
  orderId: string
  amount: number
  channel: 'web' | 'mobile'
}) {
  return tracer.startActiveSpan('checkout.submit_order', async (span) => {
    span.setAttribute('order.channel', input.channel)
    span.setAttribute('checkout.amount', input.amount)

    try {
      await validateOrder(input)

      await tracer.startActiveSpan('checkout.reserve_inventory', async (childSpan) => {
        try {
          await reserveInventory(input.orderId)
        } finally {
          childSpan.end()
        }
      })

      await tracer.startActiveSpan('checkout.charge_payment', async (childSpan) => {
        try {
          await chargePayment(input.orderId, input.amount)
        } finally {
          childSpan.end()
        }
      })

      return { success: true }
    } catch (error) {
      span.recordException(error as Error)
      span.setStatus({ code: SpanStatusCode.ERROR, message: 'submit order failed' })
      throw error
    } finally {
      span.end()
    }
  })
}
```

这个例子最重要的价值不在于“多写了几个 span”，而在于：

- 把订单提交拆成清晰的业务阶段
- 让错误定位可以直接落到业务步骤
- 让后续按阶段统计耗时成为可能
- 让链路图更接近真实处理语义，而不是只有基础 I/O 调用

## 4. 什么时候该优先自动注入，什么时候该优先手动埋点

### 4.1 更适合优先自动注入的场景

- 刚开始接入 OTel，需要快速拿到可见性
- 服务类型比较标准，以 HTTP + DB + cache 为主
- 团队还没有统一的 span 命名与属性规范
- 当前主要目标是找传播断裂、看依赖拓扑、看基础延迟

### 4.2 更适合优先手动埋点的场景

- 业务流程长，且关键阶段不能只靠 HTTP/DB span 表达
- 存在大量异步任务、状态机、批处理、工作流引擎
- 团队需要基于领域阶段做 SLI/SLO 或性能分析
- 自动 instrumentation 已经接上，但仍然“看得到请求，看不懂问题”

### 4.3 最常见也最推荐的方案：混合策略

在大多数真实系统中，最佳策略并不是二选一，而是：

1. 用自动 instrumentation 覆盖通用组件
2. 用手动埋点补齐业务阶段与领域指标
3. 用 Collector 做统一处理、采样和导出

这是因为二者解决的是不同层级的问题：

- 自动 instrumentation 保证基础覆盖率
- 手动 instrumentation 保证业务解释力

## 5. Node.js 中的自动注入与手动埋点如何配合

在 Node.js / TypeScript 项目里，一个常见做法是先在启动层加载自动 instrumentation，然后在业务代码里继续使用 API 手动补充关键 span。

### 5.1 启动层加载自动 instrumentation

```ts
import { NodeSDK } from '@opentelemetry/sdk-node'
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http'
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node'

const sdk = new NodeSDK({
  traceExporter: new OTLPTraceExporter({
    url: 'http://localhost:4318/v1/traces',
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
```

这段配置体现了两个现实问题：

- 自动 instrumentation 不是非黑即白，你通常仍需做过滤与细化
- 健康检查、metrics 拉取、内部探活等低价值请求，往往应该排除，避免噪音与成本浪费

### 5.2 业务代码里补充关键 span

```ts
import { trace } from '@opentelemetry/api'

const tracer = trace.getTracer('pricing-service', '1.0.0')

export async function calculatePrice(productId: string, region: string) {
  return tracer.startActiveSpan('pricing.calculate_final_price', async (span) => {
    span.setAttribute('product.region', region)

    try {
      const basePrice = await queryBasePrice(productId)
      const discount = await loadDiscountPolicy(region)
      return basePrice - discount
    } finally {
      span.end()
    }
  })
}
```

这样做后，自动 instrumentation 仍然会提供数据库或 HTTP 调用的底层 span，而你的业务 span 则把这些操作组织成一个更容易理解的流程。

## 6. 常见误区与实践建议

### 6.1 误区一：有了自动注入，就不需要手动埋点

这通常会导致“链路有很多 span，但没有一个真正代表业务动作”。系统能看，但难排障，更难支持业务层性能分析。

### 6.2 误区二：所有业务步骤都要强行加 span

span 不是越多越好。如果每个函数、每个 if 分支都开 span，反而会让链路图失真、成本升高、阅读体验恶化。应优先给：

- 关键业务阶段
- 远程调用或等待边界
- 容易失败或性能敏感的环节
- 需要单独统计的领域步骤

### 6.3 误区三：自动与手动会天然冲突

二者冲突通常来自错误的命名、重复包裹或不清楚职责边界，而不是模式本身。只要你清楚：基础组件交给自动库，业务阶段交给手动埋点，反而会形成更完整的层次结构。

### 6.4 误区四：zero-code 等于零配置、零理解

实际上，zero-code 只是“尽量少改业务代码”，并不意味着你可以不理解资源属性、采样、Collector 路由、属性基数和数据治理。

### 6.5 实践建议

- 先用自动 instrumentation 建立基础可见性
- 再从最关键的业务路径开始补手动 span
- 优先补“排障时最常问的问题”，而不是追求全覆盖
- 对健康检查、静态资源、噪音路径做过滤
- 给业务 span 设计稳定命名规则，避免随版本频繁变化

## 本章小结

| 主题 | 结论 |
|------|------|
| zero-code 的意义 | 尽量少改代码，快速建立基础可见性 |
| 自动 instrumentation 的优势 | 覆盖通用框架与客户端，减少样板埋点 |
| 自动 instrumentation 的边界 | 不理解业务阶段，无法替代领域语义建模 |
| 手动埋点的价值 | 让链路更可解释，补齐关键业务阶段和领域指标 |
| 推荐方案 | 自动覆盖基础组件，手动补充核心业务流程 |
| 关键注意事项 | 防止重复埋点、噪音路径、上下文断裂和高基数属性 |

## OTel实验

### 实验目标

对比“只开自动 instrumentation”和“自动 + 手动混合埋点”在排障可解释性上的差异。

### 实验步骤

1. 启动一个 Node.js HTTP 服务，只开启 `http` 和数据库客户端的自动 instrumentation。
2. 发起一次订单提交请求，观察链路中主要只有 HTTP server/client 与数据库查询 span。
3. 在业务代码中手动补充 `checkout.submit_order`、`checkout.reserve_inventory`、`checkout.charge_payment` 三个 span。
4. 再次发起同样请求，对比 trace 树形结构与可读性。
5. 尝试让支付步骤故意报错，观察错误究竟更容易落在哪一层被解释出来。

### 预期现象

- 只靠自动埋点时，可以看到依赖调用，但很难直接识别业务阶段
- 加入手动 span 后，trace 更像业务流程图，而不是底层 I/O 列表
- 错误定位会更容易聚焦到具体业务步骤

## 练习题

1. 请分别用一句话定义 zero-code、自动 instrumentation 和手动 instrumentation。
2. 为什么自动 instrumentation 通常无法替代业务指标设计？请举两个例子说明。
3. 一个订单系统已经自动采集了 HTTP 与数据库 span，但研发仍然觉得 trace 不好用。最可能缺少哪一类埋点？为什么？
4. 在 Node.js 服务里，哪些请求通常适合从自动 instrumentation 中排除？请说明原因。
5. 如果一个团队把每个函数都手动创建 span，会带来哪些问题？请从链路可读性、成本和治理三个角度回答。
