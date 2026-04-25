# 第21章：可观测性与容量规划

> 看不见的系统，迟早会变成靠运气维护的系统；而看得见但解释不了的系统，依然不算真正可观测。

> **关联章节**：本章提供灰度、回滚和事故响应所依赖的证据面。如果没有这些信号，[第22章](./22-evaluation-release-and-incident.md) 的发布门禁和事故决策就会失去依据。

## 学习目标

完成本章学习后，你将能够：

1. 区分 metrics、logs、traces 在 AI 系统中的不同职责
2. 设计覆盖资源、任务、模型质量和业务目标的观测体系
3. 用容量模型把流量、SLA、模型大小和 GPU 数量联系起来
4. 理解为什么“监控很多”不等于“定位很快”
5. 识别 AI 系统中最常见的容量规划误区

---

## 正文内容

### 21.1 可观测性不只是监控面板

一个成熟的可观测系统至少能回答三类问题：

1. **发生了什么**：哪个指标、哪个租户、哪个模型异常
2. **为什么发生**：是上游流量变化、资源饱和、调度问题还是模型版本变更
3. **接下来怎么办**：扩容、回滚、限流、降级还是继续观察

因此，可观测性不是“把指标都采上来”，而是让系统证据足够支持诊断。

### 21.2 三种主要观测信号

#### 21.2.1 Metrics

适合回答：

- 当前系统是否在偏离正常范围
- 趋势是上升还是下降
- 是否应该触发告警或扩容

典型指标包括：

- GPU 利用率
- 显存占用
- queue wait time
- tokens/s
- P95 / P99 latency
- 请求错误率

#### 21.2.2 Logs

适合回答：

- 这次失败到底是哪一步出错
- 哪个模型版本在服务这个请求
- 某次作业为什么退出

日志的关键不是多，而是可关联：

- request_id
- trace_id
- model_version
- tenant_id
- job_id

#### 21.2.3 Traces

在 AI 系统中，trace 特别适合多段链路：

```text
网关 -> 鉴权 -> embedding -> 向量检索 -> rerank -> LLM -> 安全过滤
```

如果没有 trace，很多尾延迟问题会被粗暴归到“模型太慢”，而真正慢的可能是检索或上游回源。

> 延伸阅读：本章更关注 AI 系统里的可观测性职责与容量规划。如果你想系统理解 traces、metrics、logs、Collector、上下文传播以及生产治理，可继续阅读 [OpenTelemetry 教程](../../opentelemetry-tutorial/README.md)，尤其是 [三种信号与 OTel 架构](../../opentelemetry-tutorial/part1-foundations/02-signals-and-architecture.md)、[Logs 与跨信号关联](../../opentelemetry-tutorial/part3-metrics-logs-and-semantics/08-logs-and-cross-signal-correlation.md) 和 [Cardinality、成本与性能](../../opentelemetry-tutorial/part6-production-operations/16-cardinality-cost-and-performance.md)。

#### 21.2.4 日志与 Trace 采样策略

为什么这件事要提前设计？因为 AI 链路常常又长、又贵、又高基数。全量留痕当然最理想，但在高吞吐线上系统里通常不可持续。

| 策略 | 更适合什么场景 | 代价 / 风险 |
|------|----------------|-------------|
| Head-based sampling | 请求刚进入系统时就决定是否采样；实现简单、成本稳定 | 容易错过真正慢的尾请求 |
| Tail-based sampling | 先看完整链路结果，再保留慢请求或错误请求 | 成本更高，需要更强的 Collector / 存储能力 |
| 日志分级 | 把 `INFO / WARN / ERROR` 与租户、模型版本结合 | 如果字段设计差，后续仍然难关联 |

平台实践里更常见的是：平时用较低比例 head-based sampling 控成本，事故窗口或高风险租户临时切高采样，并保留结构化错误日志。

### 21.3 AI 系统应该看哪些指标

#### 21.3.1 资源面

- GPU utilization
- GPU memory used
- CPU / memory
- network tx/rx
- disk / object storage throughput

#### 21.3.2 任务面

- step time
- dataloader time
- all-reduce time
- queue length
- pending jobs
- checkpoint duration

#### 21.3.3 服务面

- requests/s
- tokens/s
- active sequences
- prefill latency
- decode throughput
- cache hit ratio

#### 21.3.4 业务 / 质量面

- 离线评测指标
- 线上反馈分数
- 召回 / 重排质量
- 人工审核通过率
- 成本 / 请求

这最后一层很关键。AI 服务即使资源和延迟都正常，也可能因为模型行为退化而业务失败。

#### 21.3.5 GPU 监控指标详解

GPU 监控最容易被误用的地方，是把“GPU 忙”直接等同于“系统健康”。不同指标回答的是不同问题。

| 指标 | 它回答什么 | 常见误读 | 更适合从哪里拿 |
|------|------------|----------|----------------|
| GPU Utilization | SM 是否在忙、设备是否有持续计算 | 高利用率就一定代表吞吐高 | `nvidia-smi` 快速排障，DCGM Exporter 长期采集 |
| GPU Memory Utilization | 显存是否接近容量上限 | 显存高就一定表示计算充分 | `nvidia-smi`、DCGM |
| SM Occupancy | 单个 kernel 对 SM 资源的填充效率 | 低 occupancy 就一定是坏事 | Profiler、Nsight，更适合性能分析 |

`nvidia-smi` 适合登录节点后快速判断“设备现在发生了什么”；DCGM Exporter 更适合把 GPU 指标长期送进 Prometheus / Grafana，让第22章的灰度和回滚规则有稳定证据源（详见 [第22章](./22-evaluation-release-and-incident.md) §22.3）。

### 21.4 一个最小 SLI / SLO 框架

例如，一个问答服务可以定义：

```yaml
slis:
  availability_ratio: successful_requests / total_requests
  p95_latency_ms: observed end-to-end p95 latency
  answer_grounded_rate: grounded_answers / sampled_answers
  cost_per_1k_tokens_usd: total_cost_usd / (processed_tokens / 1000)
slo:
  availability_ratio: ">= 99.9%"
  p95_latency_ms: "<= 2500"
  answer_grounded_rate: ">= 0.90"
  cost_per_1k_tokens_usd: "<= 0.35"
```

这个例子体现出 AI 系统和普通服务的差异：

- 不仅有可用性和延迟
- 还有质量与成本目标

如果只有系统 SLO 没有质量 SLO，平台很容易把“快速返回错误答案”误当成成功。

### 21.5 容量规划：把流量和资源联系起来

容量规划的核心问题是：

> 给定目标流量、目标延迟和模型特性，需要多少资源以及多少余量？

### 一个常见近似方法

假设某模型单 GPU 的稳定吞吐约为 $R$ tokens/s，请求平均 token 总量为 $T_{avg}$，目标 QPS 为 $Q$，则所需 GPU 数量近似为：

$$
\text{gpus required} \approx \frac{Q \times T_{avg}}{R \times \text{target utilization}}
$$

这个近似只适合先做数量级判断。对 LLM serving，尤其是长 prompt、RAG、多峰长度分布和 agent 场景，更稳妥的做法是把 prefill 与 decode 分开建模，因为两者受算力、显存和并发约束的方式并不相同。

例如：

- 平均每请求 1200 token
- 目标 20 QPS
- 单 GPU 稳定输出 12000 tokens/s
- 目标利用率不超过 70%

则：

$$
\frac{20 \times 1200}{12000 \times 0.7} \approx 2.86
$$

即至少需要 3 张 GPU，而且这还没算冗余和峰值抖动。

### 为什么要留余量

如果你把容量设计到刚好够平均流量，一旦出现：

- 上下文长度上升
- 冷门大请求集中到来
- 某个副本故障
- 上游检索抖动造成排队

P95 / P99 很快就会失控。

> **参考数量级（仅供建立直觉，实际值因模型、SLA 和流量分布差异较大）**
>
> | 场景 | 典型值 | 说明 |
> |------|--------|------|
> | 交互式 LLM 服务稳态 GPU 利用率目标 | 55%-75% | 留出尾延迟和突发余量，避免把系统压到极限 |
> | 交互式服务稳态显存水位 | 70%-85% | 过高会让长上下文或 batch 波动更容易触发 OOM |
> | 默认 Trace 采样比例 | 1%-10% | 常见于高吞吐线上路径，事故期通常临时拉高 |
> | 关键发布窗口额外容量余量 | 20%-30% | 给灰度、单节点故障和临时回滚留空间 |

### 21.6 常见可观测性误区

**误区一：GPU 利用率高就说明系统健康。**  
不对。可能队列很长，用户已经在排队等待。

**误区二：指标很多就说明可观测。**  
不对。没有 request_id / tenant_id / model_version 关联，很多指标没有解释力。

**误区三：容量规划只看平均流量。**  
不对。AI 业务对长度分布、峰值突发、故障余量都很敏感。

### 21.7 建议的最小 dashboard

一个 LLM / RAG 服务最小可用面板至少可以包含：

1. 请求量与 token 量
2. P50 / P95 / P99 延迟
3. queue wait time
4. GPU 利用率与显存占用
5. active sequences / batch tokens
6. cache hit ratio
7. 成本 / 请求或成本 / 1k tokens
8. 关键质量指标

如果这 8 类数据能按 tenant、model_version、region 维度切开，诊断效率会高很多。

## 本章涉及的常见工具

| 概念 | 常见工具 / 命令 | 备注 |
|------|-----------------|------|
| GPU 指标采集 | DCGM Exporter、`nvidia-smi` | 前者适合长期监控，后者适合现场排障 |
| 统一埋点与 Trace | OpenTelemetry、OTel Collector | 适合把 metrics / logs / traces 串成统一上下文 |
| 面板与告警 | Prometheus、Grafana、Alertmanager | 用于 SLI / SLO、容量和灰度告警 |
| 链路追踪 | Jaeger、Tempo | 适合定位多段链路尾延迟 |

---

## 本章小结

| 主题 | 关键点 |
|------|--------|
| 可观测性 | 不只是“看到异常”，还要能解释与决策 |
| 指标体系 | 要覆盖资源、任务、服务、质量、成本五层 |
| SLO | AI 服务不能只有延迟和可用性，还应包含质量目标 |
| 容量规划 | 必须把流量、token 分布、GPU 吞吐和冗余一起考虑 |

## 练习题

1. 为什么 GPU 利用率高不一定代表用户体验好？
2. 设计一个推理服务的最小指标面板。
3. 容量规划时为什么要考虑故障冗余？
4. 如果平均流量稳定，但 P99 延迟突然恶化，你会优先看哪些指标关联？
