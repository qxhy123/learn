# 第20章：消息队列、批处理与异步系统

## 学习目标

- 理解异步系统中的 trace 建模为什么比同步 HTTP 链路更容易出错
- 掌握 producer/consumer 场景下 parent-child 与 link 两种关系的适用边界
- 学会为消息队列、后台 job、批处理任务和 cron 设计更可解释的 telemetry
- 理解异步系统中的重试、延迟、堆积和批量处理如何影响三种信号
- 建立“异步系统不追求强行串成一条长链，而追求解释清楚因果关系”的工程直觉

## 1. 为什么异步系统总让 trace 变得难懂

在同步 HTTP 调用里，链路通常比较直观：

- 请求进入服务 A
- A 调用 B
- B 再调用 C
- 最后响应返回

这时 parent-child 往往很好理解，因为控制流和时间流基本一致。

但在异步系统里，事情会迅速复杂起来：

- producer 发出消息后，真正处理可能在几秒甚至几小时后发生
- 一个消息可能被多个消费者独立消费
- 一个批处理任务可能由多个上游事件共同触发
- 一个 cron job 可能没有明确的人类请求入口
- 重试、重放、死信队列会让“同一业务事件”出现多个处理尝试

这意味着一个关键事实：**异步系统里的可观测性，重点不是强行画出一条看似连续的长 trace，而是表达清楚因果关系和系统状态。**

如果忽略这一点，常见后果是：

- trace 树看似“连起来了”，但层级非常奇怪
- 一个父 span 生命周期长得不合理，挂着大量离散消费者任务
- 重试与重放混成一团，根本看不出哪次处理真正成功
- dashboard 只看 HTTP 延迟，却完全看不到队列堆积和消费延迟

## 2. Producer / Consumer：先区分“产生消息”和“处理消息”

消息系统中最基础的边界，是 producer 与 consumer 的职责不同。

- **producer** 负责把一条消息或任务提交到异步系统
- **consumer** 负责在后续某个时间点取出并处理它

它们属于同一个业务过程，但并不一定属于同一个“同步执行上下文”。

### 2.1 producer 侧更像“交付动作”

producer span 通常关心的是：

- 消息是否被成功发布
- 发布到哪个 topic、queue 或 exchange
- 发布时附带了哪些稳定、可解释的业务属性
- 发布延迟或失败原因是什么

例如：

```ts
import { trace } from '@opentelemetry/api'

const tracer = trace.getTracer('order-service')

export async function publishOrderCreated(orderType: string) {
  return tracer.startActiveSpan('publish order.created', async (span) => {
    try {
      span.setAttributes({
        'messaging.system': 'kafka',
        'messaging.destination.name': 'order.created',
        'app.order.type': orderType,
      })

      // await producer.send(...)
      return { ok: true }
    } catch (error) {
      span.recordException(error as Error)
      throw error
    } finally {
      span.end()
    }
  })
}
```

这里 producer span 表达的是“发布动作本身”，而不是“整个消费链路已经完成”。

### 2.2 consumer 侧更像“独立处理单元”

consumer span 则更关心：

- 这条消息什么时候被取出
- 由哪个 consumer group 或 worker 处理
- 当前处理是否成功
- 处理延迟、重试次数、批量大小是多少

也就是说，producer 和 consumer 虽然有关联，但不一定要被硬塞成严格的同步父子关系。

## 3. Parent-child 还是 link：关键不在语法，而在因果模型

这是异步系统里最重要的建模选择之一。

### 3.1 什么时候适合 parent-child

当异步边界仍然可以被理解为“单一上游工作单元明确触发单一下游工作单元”，并且你希望保留较强的层级关系时，parent-child 仍然可以成立。

例如：

- 一个请求提交一个后台任务
- 一个任务只会被一个 worker 处理一次
- 延迟虽有存在，但逻辑上仍然是一对一的延续

在这类情况下，consumer span 作为 producer 或调度 span 的子节点，往往还能保持可解释性。

### 3.2 什么时候更适合 link

当下列情况出现时，link 往往更合适：

- 一个消息会被多个消费者独立处理
- 一个批处理任务由多个上游事件汇聚而来
- 一个重放任务重新处理历史数据，不应伪装成原始请求的直接子调用
- 一个 cron 或补偿任务需要引用历史上下文，但不属于原调用链的直接继续

link 的价值在于：**表达“相关”或“因果引用”，而不是强行表达“层级父子”。**

这对异步系统尤其重要，因为很多真正的系统关系是“关联”而不是“同步嵌套”。

### 3.3 一个用 link 建模消费者的示意

```ts
import { context, trace, SpanKind } from '@opentelemetry/api'

const tracer = trace.getTracer('payment-worker')

export async function handleMessage(messageTraceContext: { traceId: string; spanId: string }) {
  const linkedSpanContext = {
    traceId: messageTraceContext.traceId,
    spanId: messageTraceContext.spanId,
    traceFlags: 1,
    isRemote: true,
  }

  const span = tracer.startSpan('consume payment.requested', {
    kind: SpanKind.CONSUMER,
    links: [{ context: linkedSpanContext }],
    attributes: {
      'messaging.system': 'kafka',
      'messaging.operation': 'process',
      'messaging.destination.name': 'payment.requested',
    },
  })

  try {
    // 处理消息
  } finally {
    span.end()
  }
}
```

这个例子的重点不是某个 API 细节，而是模型选择：消费者工作单元引用上游消息来源，但不一定把自己挂成那个 span 的直接子节点。

### 3.4 一个很实用的判断方法

问自己两个问题：

1. 下游任务是不是上游调用链的“自然继续”，并且基本是一对一？
2. 如果把它们做成父子关系，最终 trace 树还是否直观、可解释？

如果答案是否定的，通常就该认真考虑 link。

## 4. 批处理、队列堆积和消费延迟：异步系统真正该关注的信号

异步系统不是只有 trace。很多关键问题其实更依赖 metrics 和 logs。

### 4.1 只看 trace，往往看不见队列状态

在消息系统里，真正重要的问题常常包括：

- topic 或 queue 是否堆积
- 消费延迟是否持续上升
- 死信队列是否增长
- 重试率是否增加
- 每批处理大小是否异常

这些信息更适合通过指标表达，例如：

- 当前 backlog 大小
- 消费延迟 histogram
- 每秒成功/失败消费数
- 每批消息数量
- 重试次数和死信数

### 4.2 一个批处理 worker 的指标示意

```ts
import { metrics } from '@opentelemetry/api'

const meter = metrics.getMeter('batch-worker')

const batchSize = meter.createHistogram('batch.job.size', {
  description: '每次批处理包含的消息数',
})

const jobLatency = meter.createHistogram('batch.job.duration.ms', {
  description: '批处理任务耗时',
})

export function recordBatchRun(size: number, durationMs: number, jobType: string) {
  batchSize.record(size, {
    'job.type': jobType,
  })

  jobLatency.record(durationMs, {
    'job.type': jobType,
  })
}
```

这里要注意的依旧是维度控制。像 `job.type` 这种稳定类别字段通常是合理的，而每个 `job.id`、每个 `message.id` 则不适合作为高频指标维度。

### 4.3 队列延迟常常比处理耗时更能解释系统问题

举例来说：

- consumer 真正处理消息只花了 50ms
- 但消息在队列里等待了 10 分钟

如果你只埋点“处理耗时”，会误以为系统很快；而用户体验上，任务已经严重延迟。因此异步系统里经常要区分：

- **processing latency**：真正执行处理所花时间
- **queue latency / lag**：从消息产生到开始处理之间的等待时间

这两者回答的是不同问题。

## 5. Job、Cron 与调度系统：它们不是 HTTP 请求的延长线

很多后台任务和定时任务没有自然的 HTTP 入口，这时如果仍然硬套“请求进入 -> 下游调用”的模型，就会很别扭。

### 5.1 Job span 应表达“这次执行”

对 job 或 worker 来说，更合理的 span 边界通常是：

- 一次 job 执行
- 一次批量处理窗口
- 一次 cron 调度触发
- 一次重试尝试

而不是人为把它们挂到某个很久之前的入口请求之下。

例如：

```ts
import { trace } from '@opentelemetry/api'

const tracer = trace.getTracer('report-scheduler')

export async function runDailyReport(reportType: string) {
  return tracer.startActiveSpan('cron daily_report', async (span) => {
    try {
      span.setAttributes({
        'job.type': 'daily_report',
        'job.report.type': reportType,
        'job.trigger': 'cron',
      })

      // 执行任务
    } finally {
      span.end()
    }
  })
}
```

### 5.2 Cron 更像“系统触发源”

cron 任务本身往往没有上游用户请求，因此它更像一种系统触发源。你应该更关心的是：

- 调度是否按时触发
- 是否有任务漏跑
- 单次执行耗时是否变长
- 成功率和失败率如何
- 是否造成下游队列积压

这意味着对 cron 场景，除了 trace 之外，往往还需要重点设计：

- 调度成功/失败 counter
- 执行耗时 histogram
- 最后一次成功时间 gauge 或可推导指标
- 失败日志与告警

### 5.3 重试要被表达成“新的尝试”

异步系统里一个常见误区是把所有重试都混在一个 span 里。这样虽然看起来“简单”，但会失去很多诊断价值。

更有解释力的方式通常是：

- 每次尝试各有自己的 span 或事件
- 记录 `retry.count` 或 attempt 序号
- 在日志中清楚区分第一次失败和第 N 次重试成功

否则你最后只能看到“这个任务总共花了很久”，却不知道是单次处理慢，还是反复失败后才成功。

## 6. 异步系统里的工程边界与常见误区

### 6.1 不要为了“链路完整”牺牲可解释性

异步系统里最容易犯的错误，是为了让 UI 上看起来像一条完整长链，就把所有工作单元都硬挂到一起。结果往往是：

- trace 树极深、极长
- 父 span 生命周期不合理
- 批处理、多消费者、重试语义全被模糊掉

正确目标不是“图更长”，而是“因果更清楚”。

### 6.2 不要只埋消费耗时，不埋堆积与等待

异步系统的很多真实故障，不是处理慢，而是：

- 消费者跟不上
- 调度间隔不合理
- 某个分区热点
- 死信和重试堆积

因此指标设计往往比单条 trace 更关键。

### 6.3 消息 ID 适合定位，不适合高频聚合

`message.id`、`job.id`、`request.id` 这类字段通常很适合：

- span attributes
- 结构化日志
- 排障检索

但通常不适合：

- 作为高频 histogram 维度
- 作为长期 dashboard 分组键
- 作为 Collector routing 的条件

### 6.4 消费者语义要区分“取到消息”和“业务处理完成”

有些系统把“收到消息”与“业务落库完成”混为一个事件，这会让排障很困难。更清晰的方式是把它们分成不同阶段：

- 收到消息
- 开始处理
- 处理成功 / 失败
- 提交 offset / ack
- 如有需要，再记录下游副作用

## 本章小结

| 主题 | 核心结论 |
|------|----------|
| 异步系统难点 | 控制流、时间流和因果关系不再天然一致 |
| producer / consumer | 属于同一业务过程，但通常是不同执行单元 |
| parent-child | 更适合一对一、可解释的延续关系 |
| link | 更适合多来源、重放、多消费者、批处理汇聚等场景 |
| 异步系统关键指标 | backlog、消费延迟、批大小、失败率、重试率 |
| job / cron 建模 | 更应表达“这次执行”而不是硬接到某个旧请求之下 |
| 常见误区 | 为了图看起来连续而牺牲语义清晰度 |

## OTel实验

### 实验目标

为一个包含 producer、consumer 和定时 job 的异步系统设计更可解释的 traces 与 metrics，对比 parent-child 与 link 的效果差异。

### 实验步骤

1. 准备一个简单示例：HTTP 服务接到创建订单请求后发布消息，由后台 worker 处理支付，再由定时任务补偿失败订单。
2. 第一轮实验，把 producer 到 consumer 强行建成 parent-child，观察 trace 树是否在延迟、重试和多消费者场景下变得难理解。
3. 第二轮实验，把 consumer 改为使用 link 指向消息来源，比较 trace 结构的可解释性。
4. 为 worker 增加指标：
   - 批处理大小
   - 处理耗时
   - 重试次数
   - 队列等待时间
5. 为 cron 补偿任务增加独立 span 和独立指标，确认它不再伪装成原始用户请求的直接子调用。

### 预期现象

- 在一对一、短延迟场景下，parent-child 仍然可能可读
- 在多消费者、批处理和重试场景下，link 往往比强行父子关系更清楚
- 仅靠 traces 很难完整解释异步系统问题，必须结合 backlog、消费延迟和失败率等 metrics

## 练习题

1. 为什么异步系统里的目标不是“强行串成一条超长 trace”，而是“表达清楚因果关系”？
2. 什么情况下 producer 和 consumer 更适合 parent-child？什么情况下更适合 link？
3. 为什么队列等待时间常常比单次处理耗时更能解释用户感知到的问题？
4. 设计一个 cron 任务的 telemetry 时，你最希望有哪些 span 属性和指标？
5. 为什么 `message.id` 和 `job.id` 适合排障定位，却通常不适合作为高频指标维度？
