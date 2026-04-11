# 第14章：Processor、Sampling 与 Routing

## 学习目标

- 理解 processor 在 Collector pipeline 中承担的治理职责，以及它与应用埋点的边界
- 掌握 filter、transform、batch 三类常见 processor 分别解决什么问题
- 说清 head sampling 与 tail sampling 的差异，并理解 tail sampling 更适合放在什么位置
- 区分 routing 与 fan-out：一个是按条件分流，一个是把同一份数据复制到多个下游
- 学会围绕稳定、低基数的属性设计 pipeline，而不是靠临时字段堆复杂规则

## 1. Processor 是 Collector 真正的“中间层能力”

在上一章中，我们已经知道 Collector 由 receiver、processor、exporter 和 pipeline 组成。真正让 Collector 从“中转器”变成“治理层”的，主要就是 processor。

如果 receiver 负责把数据接进来，exporter 负责把数据送出去，那么 processor 负责回答的是另外一个问题：**这些 telemetry 在离开当前系统之前，应该被怎样处理。**

这类处理通常包括：

- 删除明显无价值或高噪声的数据
- 统一字段名、补充资源属性、做轻量整形
- 对 traces 做采样决策
- 对导出做批处理与节流保护
- 按环境、租户、服务类别把数据送往不同下游

但是要注意一个重要边界：**processor 擅长治理，不擅长替代业务建模。**

也就是说，下面这些事情更应该由应用负责：

- 哪些业务操作值得建 span
- span 名称、层级和错误语义应该怎么设计
- 哪些指标真正代表系统状态
- 哪些日志字段是排障所必需的

而下面这些事情则更适合交给 Collector：

- 删除 `/health`、`/ready` 这类高频低价值请求
- 去掉不该保留的敏感字段或高基数字段
- 对 traces 做 tail sampling
- 把一部分流量送到归档后端，把另一部分送到主后端
- 用统一策略控制不同团队的出口行为

你可以把它理解成：

- 应用负责“产生正确语义”
- Collector 负责“让这些语义以可治理、可承受、可演进的方式流动起来”

## 2. filter 与 transform：删噪声、做归一，但不要假装它们能修好一切

### 2.1 filter：把明显不需要的数据挡在中间层

filter 的核心价值，是在 Collector 中间层统一去掉不想保留的 telemetry。最常见的场景包括：

- 丢弃健康检查、就绪探针、静态资源请求
- 丢弃测试环境、临时服务或压测流量
- 去掉没有排障价值的低级别日志
- 阻止某些明显错误的字段继续扩散

一个示意配置如下：

```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:

processors:
  filter/drop_noise:
    error_mode: ignore
    traces:
      span:
        - attributes["http.route"] == "/health"
        - attributes["http.route"] == "/ready"
    logs:
      log_record:
        - severity_text == "DEBUG" and resource.attributes["deployment.environment.name"] == "production"

exporters:
  debug:
    verbosity: basic

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [filter/drop_noise]
      exporters: [debug]
    logs:
      receivers: [otlp]
      processors: [filter/drop_noise]
      exporters: [debug]
```

这段配置体现的不是“某个语法技巧”，而是一个工程思路：**先把噪声挡在管道中间，而不是让所有下游都为噪声付费。**

不过 filter 也有边界：

- 如果你依赖 filter 去“修正”上游混乱 schema，往往只能止损，不能根治
- 如果条件写得过细、过多，Collector 自己会变成很难维护的规则系统
- 如果用高基数字段做过滤条件，治理成本本身也会升高

### 2.2 transform：做字段归一与轻量修补

transform 更适合做字段整形和归一，而不是在 Collector 里凭空创造业务语义。

例如：

- 删除不该保留的属性
- 补充一些统一的辅助属性
- 对旧字段做兼容性改名或迁移
- 把上游偶发不规范字段转换成统一结构

示意配置如下：

```yaml
processors:
  transform/normalize:
    error_mode: ignore
    trace_statements:
      - context: span
        statements:
          - delete_key(attributes, "app.raw_query")
          - delete_key(attributes, "enduser.email")
          - set(attributes["app.traffic.class"], "internal") where resource.attributes["service.namespace"] == "platform"
    log_statements:
      - context: log
        statements:
          - delete_key(attributes, "request.body")
```

这类 transform 的价值在于统一治理，但它不应该被滥用成“业务逻辑补丁层”。例如：

- 不要指望 Collector 自动把糟糕的 span 结构变成优秀的 span 结构
- 不要把复杂业务判断写进 transform 规则，最后比应用代码还难读
- 不要把所有 schema 演进都拖到 Collector，导致应用和平台互相掩盖问题

一个更健康的做法是：**Collector 做轻量纠偏，应用负责根因修复。**

## 3. batch：它解决的是吞吐与稳定性，而不是概念上的“完整性”

batch 是最常见、也最容易被低估的 processor 之一。它的作用不是改变语义，而是改善导出效率。

如果没有 batch，应用或 Collector 可能会频繁发送很多小请求，带来：

- 更高的网络开销
- 更高的序列化和 TLS 成本
- 更频繁的 exporter 调用
- 更差的吞吐表现

一个常见配置如下：

```yaml
processors:
  batch:
    send_batch_size: 1024
    timeout: 5s
```

它表达的直觉很简单：

- 数据积累到一定数量就发送
- 即使数量不够，超过一定时间也发送

为什么几乎所有生产 pipeline 都会用到 batch？因为绝大多数观测系统面对的不是“单条请求是否能发出去”，而是“持续、高频、波动流量下，整条管道是否还稳定”。

### 3.1 batch 通常放在什么位置

在大多数场景下，batch 更适合放在多数处理逻辑之后、exporter 之前。原因是：

- 先 filter，避免给即将被丢弃的数据做批处理
- 先 transform，避免对尚未标准化的数据过早打包
- 对 traces 而言，tail sampling 之后再 batch 更符合直觉，因为未保留的 trace 不值得进入最终批次

一个常见顺序是：

1. `memory_limiter`
2. `filter` / `transform` / `resource`
3. `tail_sampling`（如果使用）
4. `batch`
5. `exporter`

这不是绝对铁律，但它反映了一个很实用的原则：**先决定哪些数据应该留下，再决定如何高效地把它们送出去。**

### 3.2 batch 的代价是什么

batch 不是没有代价。它会带来：

- 一定的内存占用
- 一定的可见延迟
- 在极端故障时更依赖队列和重试策略

所以不要把它理解成“数值越大越好”。真正合理的批量大小，要结合：

- 你的吞吐量
- 可接受的延迟
- Collector 资源上限
- 下游 exporter 和 backend 的承载方式

## 4. Sampling：先区分 head sampling 和 tail sampling

采样是 OpenTelemetry 中最常被误解的话题之一。很多团队会把“采样”理解成一个统一开关，但实际上至少要区分两种典型位置。

### 4.1 head sampling：在请求开始时做决定

head sampling 通常发生在 SDK 侧，也就是请求刚进入应用、span 刚准备被创建时就决定是否保留。

它的优点是：

- 成本低
- 对应用和 Collector 的状态压力较小
- 适合超大流量场景做基础限流

它的缺点是：

- 决策发生得太早
- 还没看到错误、延迟、重试结果，就已经决定要不要保留
- 容易错过那些“少见但重要”的坏请求

### 4.2 tail sampling：在看到整条 trace 后再做决定

tail sampling 更常见于 Collector，尤其是 gateway 层。它的思路是：等一条 trace 的关键 span 基本到齐，再根据结果决定是否保留。

这让它很适合表达真正的诊断意图，例如：

- 错误 trace 全保留
- 超过阈值的慢请求保留
- 某些关键服务或关键租户的 trace 保留
- 其他普通成功请求按较低比例保留

示意配置如下：

```yaml
processors:
  tail_sampling:
    decision_wait: 10s
    num_traces: 50000
    policies:
      - name: keep-errors
        type: status_code
        status_code:
          status_codes: [ERROR]
      - name: keep-slow-traces
        type: latency
        latency:
          threshold_ms: 1000
      - name: keep-checkout
        type: string_attribute
        string_attribute:
          key: service.name
          values: [checkout]
  batch:
    timeout: 5s

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [tail_sampling, batch]
      exporters: [debug]
```

### 4.3 为什么 tail sampling 更适合 gateway 而不是每个近端 agent

因为 tail sampling 需要看到一条 trace 的更多上下文。若同一条 trace 的 span 分散在不同 Collector 实例，而这些实例彼此不共享完整视图，那么每个实例都只能基于“局部片段”做决定。

这会带来两个问题：

- 采样结果不稳定
- 你以为保留了“错误 trace”，实际只保留了错误 trace 的一部分

因此，一个非常常见的生产直觉是：

- 轻处理放在近端 agent
- tail sampling 这类需要全局视图的处理放在更中心的 gateway

### 4.4 采样解决不了什么

采样能有效控制 trace 体量，但它解决不了这些问题：

- 指标高基数
- 日志正文过大
- 错误 schema 设计
- 不合理的业务 span 切分

也就是说，**采样是 trace 侧的控量手段，不是全栈可观测性治理的万能钥匙。**

## 5. Routing 与 fan-out：分流和复制不是一回事

很多团队第一次做多后端导出时，会把 routing 和 fan-out 混为一谈。实际上，这两者回答的是完全不同的问题。

- **fan-out**：同一份 telemetry 复制给多个下游
- **routing**：不同 telemetry 走不同下游

### 5.1 fan-out：一份数据，多份出口

最直观的 fan-out，就是一个 pipeline 同时挂多个 exporter：

```yaml
exporters:
  otlp/main:
    endpoint: backend-main.example:4317
    tls:
      insecure: true
  otlp/archive:
    endpoint: archive-gateway.example:4317
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/main, otlp/archive]
```

这种模式适合：

- 主后端 + 归档后端
- 主观测后端 + 调试出口
- 迁移期双写

但要记住，fan-out 意味着：

- 出口带宽更高
- 下游成本可能乘法增长
- 一个 exporter 卡顿时，可能影响整条 pipeline 的行为

因此 fan-out 不应成为默认习惯，而应是明确、有成本意识的选择。

### 5.2 routing：不同数据，走不同出口

routing 的重点不在“多发一份”，而在“把不同类别的数据送到不同地方”。

虽然不同版本和组件形态下，Collector 里可以用不同方式实现 routing，但从工程思维上看，最重要的是两件事：

1. 路由条件必须稳定、可解释
2. 路由结果必须和组织边界对应

一个版本无关、容易理解的示意思路，是使用多个 pipeline 配合条件过滤完成“分流”：

```yaml
processors:
  filter/keep_gold:
    error_mode: ignore
    traces:
      span:
        - resource.attributes["tenant.tier"] != "gold"
  filter/drop_gold:
    error_mode: ignore
    traces:
      span:
        - resource.attributes["tenant.tier"] == "gold"
  batch:
    timeout: 5s

exporters:
  otlp/default:
    endpoint: default-backend.example:4317
    tls:
      insecure: true
  otlp/high_priority:
    endpoint: premium-backend.example:4317
    tls:
      insecure: true

service:
  pipelines:
    traces/default:
      receivers: [otlp]
      processors: [filter/drop_gold, batch]
      exporters: [otlp/default]
    traces/high_priority:
      receivers: [otlp]
      processors: [filter/keep_gold, batch]
      exporters: [otlp/high_priority]
```

这里表达的核心思想是：

- `tenant.tier=gold` 的数据走高优先级出口
- 其他流量走默认出口
- 路由依赖的是稳定、低基数、业务上可解释的字段，而不是临时主键

这里要特别注意一个边界：`tenant.tier` 更像请求或租户上下文，而不是服务身份本身。生产里如果要按它做 routing，更稳妥的做法通常是：

- 在应用侧或接入层先把它整理成 Collector 可稳定使用的低基数字段
- 再在治理层按同一层次的字段做过滤或分流
- 避免一边把它当 Resource 身份字段，一边又只在 span attributes 里设置，导致规则与数据对不上

### 5.3 什么样的字段适合做 routing

适合 routing 的字段通常具备三个特征：

- 值集合有限
- 语义长期稳定
- 能对应清晰的组织或治理边界

例如：

- `deployment.environment.name`
- `service.namespace`
- `service.name`
- `tenant.tier`
- `telemetry.class`

而不适合 routing 的字段包括：

- `request.id`
- `user.id`
- `session.id`
- 原始 URL
- 任何会无限增长的主键

### 5.4 应用如何为 routing 提供稳定信号

如果你确实需要做分流，应用更应该做的是提供**稳定、低基数、业务上可解释**的属性，而不是把所有细节都塞给 Collector。

例如：

```ts
import { trace } from '@opentelemetry/api'

const tracer = trace.getTracer('checkout-service')

export async function submitOrder(tenantTier: 'gold' | 'standard', trafficClass: 'internal' | 'external') {
  return tracer.startActiveSpan('submit_order', async (span) => {
    try {
      span.setAttributes({
        'tenant.tier': tenantTier,
        'app.traffic.class': trafficClass,
      })

      return { ok: true }
    } finally {
      span.end()
    }
  })
}
```

上面这个例子表达的是“应用如何产生稳定信号”，而不是要求你直接把 span attributes 当成 Resource 来用。更准确地说：

- 如果 routing 规则基于 Resource，就应该在进入该 pipeline 前把字段整理到 Resource 层
- 如果 routing 规则基于 span 或 datapoint attributes，就应该在 Collector 中按对应层次写条件
- 不要混用两种层次，否则最容易出现“代码里明明有字段，routing 却不生效”的困惑

这类属性适合参与 routing；而像 `order.id`、`user.id` 这类字段，更适合排障定位，不适合做管道分流依据。

## 6. 设计 Processor 链时的工程直觉

当 pipeline 开始变复杂时，真正困难的往往不是“语法怎么写”，而是“这条链是否长期可维护”。下面是一些比具体配置更重要的直觉。

### 6.1 先把重处理和轻处理分层

通常更推荐：

- 近端 agent 做轻处理，如 `memory_limiter`、少量 `batch`、必要的资源补充
- 中央 gateway 做重处理，如 `tail_sampling`、集中 routing、多后端 fan-out、统一脱敏

这样可以减少每个节点上的重复开销，也让治理策略更集中。

### 6.2 先做最小 pipeline，再逐步加规则

一个健康的演进顺序通常是：

1. `otlp -> batch -> debug`
2. 再加 `memory_limiter`
3. 再加少量 `filter` / `transform`
4. 最后再引入 `tail_sampling`、routing、fan-out

如果一上来就堆满规则，很容易在数据还没跑通之前就把系统变成黑箱。

### 6.3 顺序会改变结果

processor 的顺序不是装饰，而是行为的一部分。举例来说：

- 先 filter 再 batch，和先 batch 再 filter，资源消耗不同
- 先 transform 再 routing，和先 routing 再 transform，分流依据可能不同
- 先 tail sampling 再 fan-out，和先 fan-out 再 tail sampling，下游看到的数据规模也不同

所以不要只问“有没有这个 processor”，还要问“为什么它放在这里”。

### 6.4 少用“救火型配置”，多做长期稳定 schema

Collector 非常适合做止损，但真正成熟的团队不会让 Collector 成为永久补锅层。更好的目标是：

- 应用输出的 schema 尽量稳定
- Collector 只承担必要治理
- backend 查询与告警依赖长期可预测的字段

## 本章小结

| 主题 | 核心结论 |
|------|----------|
| processor 的价值 | 让 Collector 具备过滤、变换、采样、路由和批处理等治理能力 |
| filter | 适合去掉噪声和明显不需要的数据 |
| transform | 适合做轻量归一和字段修补，不适合替代业务建模 |
| batch | 主要改善吞吐和稳定性，通常放在大多数处理之后 |
| head sampling | 决策早、成本低，但可能错过重要坏请求 |
| tail sampling | 决策晚、诊断价值高，更适合集中式 gateway 层 |
| fan-out | 一份数据导出到多个下游，适合双写和归档 |
| routing | 不同数据走不同出口，应依赖稳定、低基数字段 |

## OTel实验

### 实验目标

把一条基础 traces pipeline 逐步演化为带 filter、transform、tail sampling 和多出口的治理型 pipeline，观察数据量与可解释性如何变化。

### 实验步骤

1. 先准备一个最小 Collector：`otlp -> batch -> debug`。
2. 让本地 Node.js 服务持续产生以下三类请求：
   - 正常业务请求
   - `/health` 健康检查
   - 少量故意制造的慢请求或错误请求
3. 在 Collector 中增加 `filter`，丢弃 `/health` 请求，比较 debug 输出数量变化。
4. 增加 `transform`，删除一个不应保留的字段，例如 `app.raw_query` 或 `enduser.email`，确认输出中字段已被清理。
5. 再增加 `tail_sampling`，设置“错误 trace 全保留、慢请求保留、其他普通请求低比例保留”的策略，观察 trace 总量变化。
6. 最后把 exporter 改为两个出口：一个 `debug`，一个 `otlp`，理解 fan-out 的行为；或者用两条 pipeline + 条件 filter 模拟 routing，比较不同租户流量如何进入不同出口。

### 预期现象

- filter 会直接减少噪声流量，而不会改变剩余数据的基本语义
- transform 更像字段整形，而不是业务逻辑重写
- tail sampling 会显著降低普通 trace 数量，但错误和慢请求更容易被保留
- fan-out 会让同一份数据在多个出口都可见；routing 则会让不同类别流量进入不同下游

## 练习题

1. 为什么说 processor 是 Collector 的治理层能力，而不是应用语义建模的替代品？
2. `filter` 和 `transform` 的核心区别是什么？各举两个适合它们处理的问题。
3. 为什么 `tail_sampling` 一般比 `head sampling` 更适合保留错误和慢请求？它又为什么通常更适合部署在 gateway 层？
4. 请解释 routing 和 fan-out 的区别，并分别举出一个适合使用它们的生产场景。
5. 如果一个团队想按 `request.id` 做 routing，把不同请求导向不同后端，你认为这会带来哪些问题？
