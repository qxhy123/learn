# 第7章：Metrics 模型与常见 Instrument

## 学习目标

- 理解 metrics 为什么更适合表达时间窗口内的聚合趋势，而不是单次请求细节
- 掌握 Counter、Gauge、Histogram 等常见 instrument 分别适合什么场景
- 理解同步 instrument 与异步 instrument 的基本区别
- 学会根据问题类型选择合适的指标模型，而不是“看到数字就记成 metric”
- 建立 metrics 设计与属性控制、成本治理之间的联系

## 1. Metrics 想解决什么问题

如果说 trace 擅长回答“这一次请求到底发生了什么”，那么 metric 更擅长回答：

- 最近 10 分钟服务整体是不是变慢了
- 错误率是否持续升高
- 当前队列积压量是多少
- CPU、内存、连接数等资源状态是否异常

这说明 metric 的核心价值不在于单条数据本身，而在于：

- 可以持续上报
- 可以长期聚合
- 可以支持趋势分析和告警

因此，metric 的设计思路和 trace 很不一样。trace 关注单次执行路径，metric 关注某类行为在时间窗口内的统计形态。

### 1.1 为什么 metric 不能替代 trace

你用 metric 可以知道：

- `/checkout` 的 p95 变高了
- `payment` 依赖的错误率升高了

但你通常不能仅靠 metric 知道：

- 哪一次请求具体为什么失败
- 是不是某个特殊租户触发的问题
- 某次慢请求在库存、支付还是风控阶段卡住了

这就是 metric 的边界。它特别适合发现趋势和触发告警，但不打算替代逐请求解释。

### 1.2 为什么 metric 又非常关键

因为真实生产系统里，很多问题的入口都不是 trace，而是：

- 告警系统触发了
- Dashboard 上某个时延曲线抬高了
- 错误率看板突然变红了

也就是说，metric 往往是“告诉你哪里值得深入看”的第一层信号。

## 2. OpenTelemetry metrics 的基本模型

OpenTelemetry 里的 metrics 不只是“记个数字”，而是围绕几件事组织起来的：

- instrument 类型
- measurement（一次记录）
- attributes（记录时附带的维度）
- aggregation（如何聚合）
- export interval（多久导出一次）

你可以把它理解成：

> 应用在运行过程中不断产出测量值，SDK 再按照某种聚合规则把这些测量整理成对外可导出的指标数据。

### 2.1 instrument 不是指标名的同义词

instrument 更像“记录方式的声明”。例如：

- 这是一个只能递增的计数器
- 这是一个分布型的耗时记录器
- 这是一个随时间上下波动的当前值观察器

同样是“订单相关”，不同问题就需要不同 instrument。

例如：

- 统计订单提交总次数，适合 Counter
- 统计订单处理耗时分布，适合 Histogram
- 观察当前待处理订单数，适合 Gauge 类 instrument

### 2.2 attributes 决定指标怎么被切分

metric 可以带 attributes，例如：

- `http.route=/checkout`
- `deployment.environment.name=production`
- `user.tier=premium`

这些维度让你能按不同类别观察同一指标。但也正因为如此，metric 的属性设计必须更克制，因为每增加一个高基数维度，都可能让时间序列数量膨胀。

## 3. 常见 instrument：Counter、Gauge、Histogram

### 3.1 Counter：适合只增不减的累计事件

Counter 最适合记录“发生了多少次”这类问题，例如：

- 请求总数
- 错误总数
- 消息消费总数
- 订单提交次数

一个简化示例：

```ts
import { metrics } from '@opentelemetry/api'

const meter = metrics.getMeter('checkout-service', '1.0.0')

const ordersSubmitted = meter.createCounter('orders_submitted_total', {
  description: 'Total number of submitted orders',
})

export function recordOrderSubmitted(channel: 'web' | 'mobile') {
  ordersSubmitted.add(1, {
    checkout_channel: channel,
  })
}
```

Counter 的关键语义是：**只增不减**。如果你的数据会回退、上下波动，那它通常就不是 Counter。

### 3.2 Gauge：适合表达某个时刻的当前值

Gauge 更适合表示“当前是多少”，例如：

- 当前队列长度
- 当前活动连接数
- 当前内存占用
- 当前 worker 数量

Gauge 类指标关注的是某一时刻的状态，而不是累计过程。

初学阶段要特别注意：

- 如果你记录的是“累计发生次数”，不要用 Gauge
- 如果你需要的是“当前系统状态”，Gauge 往往更合适

### 3.3 Histogram：适合记录分布而不只是总量

Histogram 非常适合记录：

- 请求耗时
- 数据库查询耗时
- 消息处理耗时
- 响应体大小

因为它的重点不只是“平均值”，而是分布信息。生产里大家经常关心的 p50、p95、p99，本质上都依赖分布型指标。

例如：

```ts
import { metrics } from '@opentelemetry/api'

const meter = metrics.getMeter('checkout-service', '1.0.0')

const checkoutLatency = meter.createHistogram('checkout_submit_duration_ms', {
  description: 'Duration of checkout submission',
  unit: 'ms',
})

export function recordCheckoutLatency(durationMs: number, route: string) {
  checkoutLatency.record(durationMs, {
    'http.route': route,
  })
}
```

Histogram 的优势是能表达延迟和大小的分布；它的代价是相对更复杂，也更需要控制 attributes，避免系列爆炸。

## 4. 同步 instrument 与异步 instrument

除了 instrument 类型，还要区分一个维度：是“事件发生时主动记录”，还是“系统来采时再回调提供当前值”。

### 4.1 同步 instrument：在事件发生时记录

典型场景包括：

- 每来一个请求，Counter 加一
- 每完成一次请求，Histogram 记录耗时
- 每发生一次失败，错误计数器加一

它的特点是：**业务事件发生时，应用主动提交测量值。**

### 4.2 异步 instrument：在观测周期内报告当前值

典型场景包括：

- 当前队列深度
- 当前活动连接数
- 当前内存占用
- 当前缓存条目数

它更像“被问到时上报当前状态”，而不是每次状态变化都显式记录。

### 4.3 什么时候容易选错

初学阶段最容易出错的情况包括：

- 用 Gauge 记录请求总数
- 用 Counter 表示队列当前长度
- 用 Histogram 记录一个不会形成分布意义的值
- 用异步 instrument 去表达“每次事件发生”的计数

一个简单判断法是：

- 如果是事件次数，优先想 Counter
- 如果是当前状态，优先想 Gauge
- 如果是耗时/大小分布，优先想 Histogram

## 5. 指标选型要围绕问题，而不是围绕字段

很多指标设计失控，不是因为 API 不会用，而是因为设计问题的出发点错了。

### 5.1 先问你想回答什么问题

例如：

- “最近错误多不多” -> Counter
- “当前连接数多少” -> Gauge
- “请求延迟分布如何” -> Histogram

比起先想“我这里有个 `durationMs` 变量”，更重要的是先想“我要用这个指标长期回答什么问题”。

### 5.2 指标要适合长期聚合，而不是一次性调试

某些字段很适合日志或 trace，却不适合 metric。例如：

- `user.id`
- `order.id`
- `session.id`
- 原始 URL 参数

这些信息更适合用于单次定位，而不是长期高频聚合。

### 5.3 一个简化的选型表

| 你想观察的问题 | 更适合的 instrument | 原因 |
|----------------|--------------------|------|
| 总共发生了多少次支付失败 | Counter | 只增不减的累计事件 |
| 当前队列里还有多少任务 | Gauge | 关注当前状态 |
| 请求延迟分布如何 | Histogram | 需要观察分布与分位数 |
| 当前活跃 worker 数量 | Gauge | 这是时刻状态，不是累计量 |
| 每小时处理消息的总量 | Counter | 这是累计吞吐 |

## 6. Metrics 设计中的常见误区

### 6.1 误区一：什么都想做成 metric

metric 擅长的是稳定、可聚合、可长期观察的问题。单次异常上下文、完整业务原文、复杂错误细节，通常不适合只靠 metric 表达。

### 6.2 误区二：attributes 越多越方便

多一个 attribute，往往就多一层切分方式。高基数字段一旦进入 metric，会直接影响时间序列规模、存储成本和查询性能。

### 6.3 误区三：只看 API，不看语义

就算你能正确调用 `createCounter()`，如果这个指标本身回答不了长期问题，或者命名、属性、单位都混乱，那它依旧不是一个好指标。

### 6.4 误区四：把调试需求直接变成聚合指标

很多人为了“以后方便筛选”，把主键类字段塞进 histogram attributes。这样短期看起来方便，长期几乎必然带来成本与治理问题。

## 本章小结

| 主题 | 核心结论 |
|------|----------|
| metrics 价值 | 擅长表达时间窗口内的聚合趋势、告警与长期观察 |
| Counter | 适合只增不减的累计事件 |
| Gauge | 适合表达某一时刻的当前状态 |
| Histogram | 适合记录时延、大小等分布型数据 |
| 同步 vs 异步 | 同步在事件发生时记录，异步在观测周期内提供当前值 |
| 设计原则 | 先想问题，再选 instrument，谨慎控制 attributes |

## OTel实验

### 实验目标

围绕一个简单的 HTTP 服务，分别设计 Counter、Histogram 和 Gauge，体验不同 instrument 的语义差异。

### 实验步骤

1. 准备一个 Node.js 服务，包含 `POST /checkout` 和一个简单任务队列。
2. 为以下问题分别设计指标：
   - 订单提交总次数
   - 订单提交耗时分布
   - 当前待处理任务数
3. 使用 Counter 记录订单次数，Histogram 记录请求耗时，Gauge 观测当前队列深度。
4. 连续发送多次请求，并刻意制造几次慢请求。
5. 观察三类指标在 Collector 或 backend 中的呈现差异。
6. 再尝试错误地把 `user.id` 加入 Histogram attributes，思考这会带来什么后果。

### 预期现象

- Counter 更适合看吞吐与错误总量。
- Histogram 更适合看延迟分布，而不是只看平均值。
- Gauge 更适合看系统当前状态。
- 错误的属性设计会让原本合理的指标迅速变得昂贵且难治理。

## 练习题

1. 为什么 metric 更适合回答趋势问题，而不是单次请求问题？
2. Counter、Gauge、Histogram 三者分别更适合哪些典型场景？
3. 什么是同步 instrument，什么是异步 instrument？请各举一个例子。
4. 为什么 `user.id`、`order.id` 这类字段通常不适合作为高频 metric 的 attributes？
5. 请为“支付成功率”“当前消费积压”“请求延迟分布”分别选择合适的 instrument，并说明理由。
