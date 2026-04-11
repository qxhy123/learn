# 第16章：Cardinality、成本与性能

## 学习目标

- 理解 cardinality、数据点数量、时间序列数量之间的区别
- 分别识别 traces、metrics、logs 三类信号的主要成本来源
- 学会判断哪些 attributes 适合长期保留，哪些会引发高基数问题
- 掌握在应用、SDK、Collector、backend 四个层次控制成本与性能的方法
- 建立“先保证可解释，再控制规模”的生产观测优化思路

## 1. 为什么可观测性成本首先是数据模型问题

很多团队第一次遇到观测成本失控，不是因为后端太贵，而是因为最前面的数据模型设计就已经失控了。OpenTelemetry 本身并不会强迫你把每个请求都变成昂贵的数据，但它也不会替你自动阻止坏的 schema 进入生产环境。

理解成本，先要区分三件事：

1. **单条数据有多大**：一个 span、一个 log record、一个 metric datapoint 的体积
2. **单位时间产生多少条**：请求量、错误量、批处理规模、采集频率
3. **这些数据会分裂成多少组**：也就是 cardinality，高基数意味着更多索引、更多时间序列、更多聚合状态

三类信号的主要成本结构并不相同：

- **Trace 成本**通常近似于：采样后的请求数 × 每个请求的 span 数 × 每个 span 的属性/事件大小
- **Metric 成本**通常近似于：时间序列数 × 上报频率 × 保留时长
- **Log 成本**通常近似于：日志条数 × 平均体积 × 索引字段数量 × 保留时长

这也是为什么很多团队会误判问题：

- 看到 trace 太多，就试图只做 trace 采样，但真正爆炸的是 metrics label cardinality
- 看到日志成本高，就一刀切地减少日志级别，结果关键排障信息一起消失
- 看到 Collector CPU 高，就怪 Collector，实际上是应用层把 `user.id`、`session.id`、`cart.id` 全部做成了指标标签

一个更实用的判断方式是：**成本不是“观测系统的后果”，而是“观测建模的结果”**。

## 2. Cardinality 从哪里来

很多人把 cardinality 只理解成“Prometheus 标签太多”，这其实过窄。OpenTelemetry 里高基数可能出现在多个位置。

### 2.1 Resource 层的高基数

Resource 描述的是数据来源实体，例如：

- `service.name`
- `service.namespace`
- `service.version`
- `deployment.environment.name`
- `k8s.namespace.name`
- `k8s.pod.name`
- `host.name`

其中有些字段天然会变化，比如 `k8s.pod.name`。这不代表它们不能存在，而是你要清楚它们的用途：

- 用于定位单个 Pod 故障是有价值的
- 用于作为长期聚合维度就可能很昂贵

因此，资源属性要区分：

- **稳定身份字段**：适合长期聚合，例如 `service.name`、`deployment.environment.name`
- **瞬时实例字段**：适合排障定位，例如 `k8s.pod.name`、`container.id`

### 2.2 Span / Log Attributes 的高基数

以下字段经常有诊断价值，但不适合作为普遍聚合键：

- `user.id`
- `session.id`
- `order.id`
- `request.id`
- 完整 URL 查询参数
- 原始 SQL 语句
- 文件路径、对象键、租户内业务主键

这些字段适合出现在：

- 某些 span attribute 中，用于精确定位单次请求
- 某些结构化日志中，用于按主键检索

但它们通常不适合：

- 进入 histogram label
- 作为 counter 的常驻维度
- 进入长期高频索引

### 2.3 Metrics Label 的高基数

指标的高基数最危险，因为它会直接影响时间序列数量。一个常见错误是把“方便筛选”的字段直接塞到指标标签里。

错误示例：

```ts
requestDuration.record(durationMs, {
  'user.id': userId,
  'session.id': sessionId,
  'http.route': req.path,
})
```

上面的问题有三个：

1. `user.id` 和 `session.id` 近似无限增长
2. `req.path` 如果包含真实路径参数，就会制造大量唯一值
3. 这些值对长期 SLA 聚合帮助很有限，却会显著增加系列数

更合理的方式是：

```ts
requestDuration.record(durationMs, {
  'http.route': '/orders/:id',
  'user.tier': userTier,
  'deployment.environment.name': process.env.NODE_ENV === 'production' ? 'production' : 'development',
})
```

一个简化判断表如下：

| 属性 | 是否适合做高频 metric 维度 | 原因 |
|------|-----------------------------|------|
| `service.name` | 适合 | 稳定、聚合价值高 |
| `deployment.environment.name` | 适合 | 生产、预发、测试边界清晰 |
| `http.route` | 通常适合 | 需保证是模板路由而非原始路径 |
| `user.tier` | 视情况适合 | 类别有限且业务解释明确 |
| `user.id` | 不适合 | 唯一值过多 |
| `session.id` | 不适合 | 近似无限增长 |
| `order.id` | 不适合 | 主键级字段不适合做时序维度 |
| `k8s.pod.name` | 谨慎使用 | 对排障有用，但会快速膨胀 |

## 3. 三类信号各自怎样影响性能

除了后端存储成本，OpenTelemetry 还会影响运行时性能。性能问题通常来自以下几个方向。

### 3.1 应用内开销

应用侧开销主要包括：

- 创建 span、attribute、event 的对象分配成本
- 上下文传播与异步上下文维护成本
- histogram 聚合状态维护成本
- 日志序列化与 I/O 成本
- exporter 队列与批处理线程开销

大多数情况下，**少量高价值埋点的收益远高于其开销**。真正危险的是：

- 在热路径上创建大量短生命周期 span
- 在高频循环里记录高维度 metrics
- 在每次请求里输出大体积 debug logs

### 3.2 网络与序列化开销

OTLP 导出会产生：

- protobuf 编码成本
- 网络发送成本
- 批处理缓存占用
- 重试与排队带来的尾部延迟风险

如果应用直接导出到远端 backend，那么抖动、重试、TLS 握手、出口带宽等问题都会更直接地作用在业务进程上。这也是为什么生产环境常常更偏向“应用到本地或近端 Collector，再由 Collector 对外发送”。

### 3.3 Collector 开销

Collector 的成本主要由这些因素驱动：

- 接收的信号种类与吞吐量
- processor 链长度
- tail sampling、transform、filter 等处理逻辑复杂度
- 队列大小、批处理大小、重试策略
- 多路 exporter 并发发送

Collector 不是免费的，但它的价值在于把复杂处理从业务进程中拿出来集中治理。

## 4. 成本控制要分层做，而不是只靠一个开关

控制成本最有效的方法不是等数据已经堆到后端再处理，而是在不同层次分别做约束。

### 4.1 应用层：先决定什么值得被观测

应用层应该控制：

- 只为真正有边界意义的操作建 span
- 指标只保留稳定、有限、解释清楚的维度
- 避免把原始主键、完整请求体、敏感字段塞入日志和 attributes
- 优先记录业务类别，而不是业务实例主键

一个很重要的经验是：**如果某个字段主要用于单次定位，它更可能适合 span 或 log；如果它主要用于长期聚合，它才适合 metric 维度。**

### 4.2 SDK 层：控制批量、队列和上限

SDK 层常见控制项包括：

- span attribute 数量上限
- span event 数量上限
- 批量导出大小与超时
- metric 导出间隔
- 日志批量与队列策略

这些配置不会修复坏的 schema，但可以防止坏配置把进程直接拖垮。

### 4.3 Collector 层：做统一过滤、变换与保护

Collector 很适合做组织级控制，例如：

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
    timeout: 2s
    send_batch_size: 2048
  transform/sanitize_metrics:
    metric_statements:
      - context: datapoint
        statements:
          - delete_key(attributes, "user.id")
          - delete_key(attributes, "session.id")

exporters:
  debug:
    verbosity: basic

service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, transform/sanitize_metrics, batch]
      exporters: [debug]
```

这个例子表达的重点不是某个具体产品配置，而是三个治理原则：

1. 先保护内存
2. 再清理明显错误的高基数字段
3. 最后再批量输出

### 4.4 Backend 层：保留与索引策略

后端一般负责：

- 不同信号的保留时长分层
- 热数据与冷数据分层
- 是否为某些字段建立索引
- 是否对高成本查询加限制

但需要注意：**backend 能优化的是“已经被采进来的数据”，修不好“本不该进来的数据”。**

## 5. 采样、聚合与保留：三种最常用的控成本手段

### 5.1 采样不是万能药

采样对 traces 最有效，因为 trace 天然是“离散事件集合”。

- **head sampling**：在请求开始时决定是否采样，成本低，但可能错过后续错误上下文
- **tail sampling**：在 Collector 聚合后基于结果决定保留哪些 trace，更适合保留错误、慢请求和特定租户流量，但需要额外状态和内存

但是采样不能解决所有问题：

- 它不能降低已经被创建出来的 metric label cardinality
- 它不能修复日志中无节制的原始数据输出
- 它不能替代 schema 设计

### 5.2 聚合要围绕问题而不是围绕字段

指标的价值来自回答问题，例如：

- 服务是否变慢了
- 某个依赖是否错误率升高
- 某个消费者组是否积压

因此应优先聚合能帮助回答这些问题的维度，而不是“所有可能有用的字段”。例如：

- 看整体延迟时，用 `service.name`、`http.route`、`http.response.status_code`
- 看业务分层时，用 `user.tier`、`payment.method_type`
- 不要为了“以后可能方便筛选”而加 `user.id`、`order.id`

### 5.3 保留策略要根据信号特性区分

一个常见做法是：

- 指标保留更长时间，用于趋势分析和容量规划
- 关键 trace 保留中等时长，用于排障和回溯
- 原始日志保留最短，但对审计类日志单独延长

生产实践里，真正成熟的团队不会只问“保留多久”，而会同时问：

- 这些数据由谁消费
- 多久之后价值迅速下降
- 哪些字段必须保留，哪些只在短期排障时有意义

## 6. 先观测你的观测系统，再谈优化

如果你想控制观测成本，就必须先把观测管道本身也纳入观测范围。至少要关注：

- Collector CPU、内存、队列长度
- exporter 失败率、重试次数、发送延迟
- dropped spans / logs / metric datapoints
- 应用侧 telemetry 队列长度
- OTLP 发送错误与超时

一个成熟的优化顺序通常是：

1. 先确认数据是否正确、可解释
2. 再确认问题出在应用、Collector 还是 backend
3. 然后清理高基数字段和噪声日志
4. 最后再用采样、保留、分层存储做精细调优

不要一开始就把所有东西都采样掉。那样你往往只是更快地丢失了诊断能力，而不是更聪明地降低了成本。

## 本章小结

| 主题 | 核心结论 |
|------|----------|
| cardinality 本质 | 它是数据分组规模问题，不只是“标签多一点” |
| traces 成本 | 与采样率、每条请求 span 数、span 体积强相关 |
| metrics 成本 | 与时间序列数量最相关，高基数标签最危险 |
| logs 成本 | 与条数、体积、索引策略、保留时长强相关 |
| 最佳控制点 | 优先在应用和 Collector 层做 schema 与过滤治理 |
| 常见误区 | 只靠采样、只靠 backend、只盯单一信号都不够 |

## OTel实验

### 实验目标

比较低基数与高基数指标设计对数据量和诊断价值的影响。

### 实验步骤

1. 为同一个请求延迟 histogram 设计两版 attributes：
   - 版本 A：`service.name`、`http.route`、`deployment.environment.name`
   - 版本 B：额外加入 `user.id`、`session.id`
2. 用脚本或压测工具生成 1000 次请求，并让每次请求带不同的 `user.id`
3. 观察本地 Collector 或 backend 中：
   - 系列数是否显著增加
   - 查询延迟是否变慢
   - Collector CPU / 内存是否上升
4. 删除 `user.id` 和 `session.id`，改为 `user.tier` 这类有限类别字段，再次比较
5. 对 traces 打开 10% head sampling，验证这能否显著改善 metrics 侧问题

### 预期现象

- 版本 B 会制造大量唯一时间序列
- trace 采样率下降后，trace 成本下降明显，但 metrics 的系列膨胀不会同步消失
- 用分类字段替代主键字段后，指标仍然保留业务可解释性，但成本显著更可控

## 练习题

1. 为什么降低 trace 的 head sampling 比例，通常不能解决 metrics 的高基数问题？
2. 下列字段中，哪些更适合作为高频 histogram 的 attributes：`service.name`、`user.id`、`http.route`、`session.id`、`deployment.environment.name`、`user.tier`？请说明理由。
3. 假设你在一个电商系统中负责治理观测成本，请分别从应用层、Collector 层、backend 层给出至少两条控制措施。
4. 为什么 `order.id` 适合出现在某些 span 或日志中，但通常不适合作为 metrics 标签？
