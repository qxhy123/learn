# 附录B：Collector Pipeline 速查表

本附录用于快速回顾 OpenTelemetry Collector 中最常见的 pipeline 结构、部署模式与配置组合。内容以“模式速查”为主，帮助你在设计或排障时快速定位思路。

## 一、Collector pipeline 最小心智模型

可以先把 Collector 理解成下面的流水线：

```text
receiver -> processor -> exporter
```

按信号类型拆开后，通常是：

```text
traces pipeline
metrics pipeline
logs pipeline
```

每条 pipeline 回答三个问题：

1. 数据从哪里来
2. 中途要做什么处理
3. 最后发到哪里去

## 二、最常见组件职责速查

### 1. Receiver 速查

| 组件 | 常见用途 | 适合场景 | 备注 |
|------|----------|----------|------|
| `otlp` | 接收应用通过 OTLP 发来的 traces / metrics / logs | 应用 SDK、近端 agent、上游 Collector | 最常见统一入口 |
| `prometheus` | 抓取指标端点 | Prometheus 风格指标接入 | 更偏 metrics 场景 |
| `filelog` | 从文件读取日志 | 宿主机日志、容器文件日志 | 常用于日志采集 |
| `hostmetrics` | 采集主机层指标 | 节点、VM、宿主机 | 更偏平台与基础设施观测 |

### 2. Processor 速查

| 组件 | 主要作用 | 常见位置 | 备注 |
|------|----------|----------|------|
| `memory_limiter` | 控制内存占用，避免 Collector 被打爆 | 接近 pipeline 前部 | 生产环境常见基础组件 |
| `batch` | 合并小批量数据，提高导出效率 | pipeline 末段常见 | 几乎所有 pipeline 都常用 |
| `resource` | 添加、修改、删除资源属性 | gateway 或统一治理层 | 适合统一 service / environment 相关字段 |
| `filter` | 按条件丢弃数据 | 治理层 | 常用于去噪、降成本 |
| `transform` | 改写属性、标准化字段、脱敏 | 治理层 | 功能强，需防止过度复杂 |
| `tail_sampling` | 基于完整 trace 结果采样 | traces pipeline 的治理层 | 更智能，但更耗资源 |

### 3. Exporter 速查

| 组件 | 主要作用 | 适合场景 | 备注 |
|------|----------|----------|------|
| `debug` | 打印 Collector 收到的数据 | 本地验证、教学、排障 | 最适合确认“数据到底进没进来” |
| `otlp` | 向下游 Collector 或兼容 backend 发数据 | 通用出口 | vendor-neutral 常见选择 |
| 其他协议 exporter | 发往特定协议或系统 | 特殊兼容场景 | 需评估是否引入额外耦合 |

## 三、最小闭环模板

### 模板 1：本地验证最小 traces pipeline

适合场景：

- 刚开始学习 Collector
- 想确认应用是否真的把 trace 发到了 Collector
- 先验证数据结构，再谈后端

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

重点：

- 先让链路通
- 不要一上来加很多 processor
- `debug` 是最直接的真相来源

### 模板 2：最小 traces + metrics 双信号闭环

适合场景：

- 验证 Node.js、Java、Go 等应用同时发 traces 与 metrics
- 想看 Collector 的多 pipeline 结构

```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 256
  batch:

exporters:
  debug:

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [debug]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [debug]
```

重点：

- `memory_limiter` 常作为生产前的第一道保护
- traces 与 metrics 可以共用 receiver，但 pipeline 仍是分开的

## 四、常见部署模式模板

### 模式 1：单层 Collector

```text
app -> collector -> backend
```

适合：

- 学习阶段
- 小规模系统
- 单团队最小闭环

优点：

- 结构最简单
- 容易排查
- 改动面小

注意点：

- 如果 Collector 同时承担接入与治理，后续规模化时可能变重
- 不要让应用直接耦合过多后端出口细节

### 模式 2：Agent + Gateway 双层模式

```text
app -> agent collector -> gateway collector -> backend
```

适合：

- Kubernetes 集群
- 多团队共享治理能力
- 希望统一采样、脱敏、路由、多后端导出

分层理解：

| 层次 | 更适合做什么 |
|------|--------------|
| agent | 接入、近端缓冲、轻量批处理、节点侧信号汇聚 |
| gateway | 统一治理、采样、变换、路由、出口控制 |

注意点：

- 不要把复杂治理逻辑分散到每个 agent
- gateway 需要重点考虑 HA、扩容和监控

### 模式 3：多环境隔离模式

```text
development apps -> development collector -> development backend
staging apps -> staging collector -> staging backend
production apps -> production collector -> production backend
```

适合：

- 环境隔离要求强
- 不同环境保留期、采样率不同
- 避免开发流量污染生产观测数据

注意点：

- 环境名要统一，例如统一使用 `deployment.environment.name`
- staging 尽量接近生产 schema，避免“预发验证通过，生产全走样”

## 五、常见治理组合模板

### 模板 3：统一补充资源字段

适合场景：

- 各服务接入方式不同，需要 gateway 层补统一字段
- 想为所有流量标记来源 pipeline、区域或组织信息

```yaml
processors:
  resource:
    attributes:
      - key: telemetry.pipeline
        value: gateway
        action: upsert
      - key: deployment.environment.name
        value: production
        action: upsert
```

常用动作理解：

| 动作 | 含义 |
|------|------|
| `upsert` | 有则更新，无则新增 |
| `insert` | 仅在不存在时插入 |
| `delete` | 删除字段 |

### 模板 4：日志或指标字段清理

适合场景：

- 需要止损高基数字段
- 需要删除明显敏感字段
- 希望在进入 backend 前统一清理

```yaml
processors:
  transform/sanitize:
    metric_statements:
      - context: datapoint
        statements:
          - delete_key(attributes, "user.id")
          - delete_key(attributes, "session.id")
```

使用提醒：

- 这是治理止损，不是替代上游 schema 设计
- 删除前要确认不会破坏关键消费面

### 模板 5：按信号分别导出到不同 backend

适合场景：

- traces、metrics、logs 使用不同后端
- 希望保持统一入口但分开存储与消费

```yaml
exporters:
  otlp/traces:
    endpoint: traces-backend.internal:4317
    tls:
      insecure: true
  otlp/metrics:
    endpoint: metrics-backend.internal:4317
    tls:
      insecure: true
  otlp/logs:
    endpoint: logs-backend.internal:4317
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/traces]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/metrics]
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/logs]
```

重点：

- 统一入口不等于统一后端
- 三种信号可根据成本和消费方式分别优化

### 模板 6：保留 debug 出口做排障旁路

适合场景：

- rollout 初期
- 排查某条 pipeline 是否真的收到数据
- 对比变换前后效果

```yaml
exporters:
  debug:
    verbosity: basic
  otlp:
    endpoint: backend.internal:4317
    tls:
      insecure: true
```

使用提醒：

- 生产环境中应谨慎长期保留高量 debug 输出
- 更适合阶段性验证或受控排障窗口

## 六、不同信号的 pipeline 设计重点

| 信号 | 更常见的 pipeline 重点 | 常见误区 |
|------|------------------------|----------|
| traces | 批处理、链路完整性、采样、错误优先保留 | 只看到 trace 能出图，就忽略链路断裂和采样偏差 |
| metrics | 稳定维度、低高基数、远端写入可靠性 | 把主键字段当作标签 |
| logs | 结构化、脱敏、噪声过滤、trace 关联 | 把所有原始内容都发出去 |

## 七、配置顺序经验法则

一个常见、比较稳的 processor 顺序思路是：

```text
memory_limiter -> resource / transform / filter -> batch -> exporter
```

可理解为：

1. 先保护自己
2. 再做标准化和清理
3. 最后再批量导出

注意：

- 并不是所有 pipeline 都必须完全一样
- 某些复杂 processor 的顺序会影响结果，要专门验证
- 学习阶段先求“通”和“可解释”，再谈“复杂”和“优雅”

## 八、排障速查：看到异常时先查哪里

| 现象 | 优先检查点 |
|------|------------|
| 应用本地看似正常，Collector 没数据 | OTLP endpoint、协议是否匹配、网络连通性 |
| Collector 有 traces，没有 metrics | 应用侧是否启用 metric reader，metrics pipeline 是否存在 |
| debug 有数据，backend 没数据 | exporter endpoint、认证、后端兼容性、出口失败日志 |
| 数据有了，但服务名混乱 | Resource 配置、平台统一注入、gateway resource processor |
| 指标 series 突然暴涨 | 新增 attributes、transform 是否失效、应用 schema 变更 |
| trace 断裂 | propagation、异步边界、消息链路模型、跨语言 header 兼容 |

## 九、设计建议：什么时候该把逻辑放到 Collector

### 更适合放到 Collector 的事

- 统一入口与出口
- 批处理、内存保护
- 公共脱敏和字段规范化
- 多后端路由
- 组织级采样策略

### 更适合留在应用里的事

- 关键业务 span 设计
- 业务指标定义
- 结构化日志业务字段
- 与业务语义强相关的错误分类

一句话总结：

> Collector 更像治理与运输层，应用更像语义生产层。

## 十、使用本速查表的建议

你可以在三种场景下快速使用本附录：

1. 写新 Collector YAML 前，先从模板选择最接近的骨架。
2. 排障时，根据“现象 -> 优先检查点”快速缩小范围。
3. 评审 pipeline 方案时，用“哪些逻辑该放 Collector、哪些不该放”检查边界是否合理。

如果你准备进入生产设计，建议与第 13、14、15、16、23、24 章一起对照阅读。速查表负责帮助你快速回忆模式，而章节正文负责解释为什么这样设计。
