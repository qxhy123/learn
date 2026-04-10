# 第21章：可观测性与容量规划

> 看不见的系统，迟早会变成靠运气维护的系统；而看得见但解释不了的系统，依然不算真正可观测。

## 学习目标

完成本章学习后，你将能够：

1. 区分 metrics、logs、traces 在 AI 系统中的不同职责
2. 设计覆盖资源、任务、模型质量和业务目标的观测体系
3. 用容量模型把流量、SLA、模型大小和 GPU 数量联系起来
4. 理解为什么“监控很多”不等于“定位很快”
5. 识别 AI 系统中最常见的容量规划误区

---

## 正文内容

## 21.1 可观测性不只是监控面板

一个成熟的可观测系统至少能回答三类问题：

1. **发生了什么**：哪个指标、哪个租户、哪个模型异常
2. **为什么发生**：是上游流量变化、资源饱和、调度问题还是模型版本变更
3. **接下来怎么办**：扩容、回滚、限流、降级还是继续观察

因此，可观测性不是“把指标都采上来”，而是让系统证据足够支持诊断。

## 21.2 三种主要观测信号

### 21.2.1 Metrics

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

### 21.2.2 Logs

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

### 21.2.3 Traces

在 AI 系统中，trace 特别适合多段链路：

```text
网关 -> 鉴权 -> embedding -> 向量检索 -> rerank -> LLM -> 安全过滤
```

如果没有 trace，很多尾延迟问题会被粗暴归到“模型太慢”，而真正慢的可能是检索或上游回源。

## 21.3 AI 系统应该看哪些指标

### 21.3.1 资源面

- GPU utilization
- GPU memory used
- CPU / memory
- network tx/rx
- disk / object storage throughput

### 21.3.2 任务面

- step time
- dataloader time
- all-reduce time
- queue length
- pending jobs
- checkpoint duration

### 21.3.3 服务面

- requests/s
- tokens/s
- active sequences
- prefill latency
- decode throughput
- cache hit ratio

### 21.3.4 业务 / 质量面

- 离线评测指标
- 线上反馈分数
- 召回 / 重排质量
- 人工审核通过率
- 成本 / 请求

这最后一层很关键。AI 服务即使资源和延迟都正常，也可能因为模型行为退化而业务失败。

## 21.4 一个最小 SLI / SLO 框架

例如，一个问答服务可以定义：

```yaml
slis:
  availability: successful_requests / total_requests
  p95_latency_ms: 2500
  answer_grounded_rate: 0.92
  cost_per_1k_tokens: 0.35
slo:
  availability: ">= 99.9%"
  p95_latency_ms: "<= 2500"
  answer_grounded_rate: ">= 0.90"
```

这个例子体现出 AI 系统和普通服务的差异：

- 不仅有可用性和延迟
- 还有质量与成本目标

如果只有系统 SLO 没有质量 SLO，平台很容易把“快速返回错误答案”误当成成功。

## 21.5 容量规划：把流量和资源联系起来

容量规划的核心问题是：

> 给定目标流量、目标延迟和模型特性，需要多少资源以及多少余量？

### 一个常见近似方法

假设某模型单 GPU 的稳定吞吐约为 $R$ tokens/s，请求平均 token 总量为 $T_{avg}$，目标 QPS 为 $Q$，则所需 GPU 数量近似为：

$$
\text{gpus required} \approx \frac{Q \times T_{avg}}{R \times \text{target utilization}}
$$

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

## 21.6 常见可观测性误区

**误区一：GPU 利用率高就说明系统健康。**  
不对。可能队列很长，用户已经在排队等待。

**误区二：指标很多就说明可观测。**  
不对。没有 request_id / tenant_id / model_version 关联，很多指标没有解释力。

**误区三：容量规划只看平均流量。**  
不对。AI 业务对长度分布、峰值突发、故障余量都很敏感。

## 21.7 建议的最小 dashboard

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

