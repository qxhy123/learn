# 第23章：监控与可观测性

> 生产环境中，你看不到的东西迟早会出问题。监控是从"跑起来就行"到"稳定运行"的关键一步。

---

## 学习目标

学完本章，你将能够：

1. 理解 vLLM 暴露的 Prometheus 指标及其含义
2. 搭建 Prometheus + Grafana 监控面板
3. 设置关键告警规则
4. 通过日志诊断运行时问题
5. 建立 LLM 推理服务的监控最佳实践

---

## 23.1 Prometheus 指标

### 获取指标

```bash
# vLLM 默认在 /metrics 端点暴露 Prometheus 指标
curl http://localhost:8000/metrics
```

### 核心指标

| 指标 | 类型 | 含义 |
|------|------|------|
| `vllm:num_requests_running` | Gauge | 当前运行中的请求数 |
| `vllm:num_requests_waiting` | Gauge | 当前等待中的请求数 |
| `vllm:kv_cache_usage_perc` | Gauge | KV Cache block 使用率 |
| `vllm:prefix_cache_queries` | Counter | 前缀缓存查询次数 |
| `vllm:prefix_cache_hits` | Counter | 前缀缓存命中次数 |
| `vllm:num_preemptions_total` | Counter | 累计抢占次数 |
| `vllm:request_success_total` | Counter | 成功完成的请求数 |
| `vllm:avg_prompt_throughput_toks_per_s` | Gauge | 输入 token 吞吐 |
| `vllm:avg_generation_throughput_toks_per_s` | Gauge | 生成 token 吞吐 |

### 延迟指标

| 指标 | 含义 |
|------|------|
| `vllm:e2e_request_latency_seconds` | 端到端请求延迟分布 |
| `vllm:time_to_first_token_seconds` | TTFT 分布 |
| `vllm:time_per_output_token_seconds` | TPOT 分布 |

---

## 23.2 搭建监控面板

### Prometheus 配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm-server:8000']
    metrics_path: '/metrics'
```

### Docker Compose 一键部署

```yaml
# docker-compose.yml
version: '3'
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    ports:
      - "8000:8000"
    command: --model Qwen/Qwen2.5-7B-Instruct

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### 关键 Grafana 面板

建议创建以下面板：

1. **请求状态**：running / waiting 请求数（V1 不再有 swapped 状态）
2. **KV Cache 使用率**：`vllm:kv_cache_usage_perc`
3. **Prefix Cache 命中率**：`vllm:prefix_cache_hits` / `vllm:prefix_cache_queries`
4. **吞吐量**：输入和生成 token 吞吐
5. **延迟分布**：TTFT 和 TPOT 的 P50/P95/P99
6. **抢占率**：每分钟抢占次数

---

## 23.3 告警规则

### 推荐告警

```yaml
# Prometheus alerting rules
groups:
  - name: vllm_alerts
    rules:
      - alert: HighKVCacheUsage
        expr: vllm:kv_cache_usage_perc > 0.95
        for: 5m
        annotations:
          summary: "KV Cache usage > 95%, preemption likely"

      - alert: HighWaitingRequests
        expr: vllm:num_requests_waiting > 50
        for: 2m
        annotations:
          summary: "Many requests waiting, consider scaling"

      - alert: HighPreemptionRate
        expr: rate(vllm:num_preemptions_total[5m]) > 1
        for: 5m
        annotations:
          summary: "Frequent preemptions, reduce concurrency or add GPUs"

      - alert: HighTTFT
        expr: histogram_quantile(0.95, vllm:time_to_first_token_seconds_bucket) > 5
        for: 5m
        annotations:
          summary: "P95 TTFT > 5s"
```

### V1 特有的关键指标

当前 V1 引擎暴露了一些旧教程没有覆盖的重要指标：

| 指标 | 含义 | 关注点 |
|------|------|--------|
| `vllm:prefix_cache_queries` | 前缀缓存查询总次数 | 配合 hits 算命中率 |
| `vllm:prefix_cache_hits` | 前缀缓存命中总次数 | 命中率下降说明 prompt 结构变了 |
| `vllm:spec_decode_num_accepted_tokens` | 投机解码接受 token 数 | 低接受率意味着 draft 质量差 |
| `vllm:spec_decode_num_draft_tokens` | 投机解码 draft token 数 | 配合 accepted 算接受率 |

注意：V1 已经移除了 `swapped` 状态和 CPU swap 相关指标。如果你在旧配置中引用了 `gpu_cache_usage_perc` 或 `cpu_cache_usage_perc`，需要更新为 `kv_cache_usage_perc`。

---

## 23.4 日志分析

### 日志级别

```bash
# 调整日志级别
VLLM_LOGGING_LEVEL=DEBUG vllm serve model    # 详细调试
VLLM_LOGGING_LEVEL=INFO vllm serve model     # 标准 (默认)
VLLM_LOGGING_LEVEL=WARNING vllm serve model  # 只看警告
```

### 关键日志模式

```
# 正常启动
INFO: Model loaded successfully
INFO: Initializing a V1 LLM engine ...
INFO: init engine (profile, create kv cache, warmup model) took X.XX seconds

# 性能警告
WARNING: Preempting 3 sequences
WARNING: KV cache usage: 98%

# 错误
ERROR: CUDA out of memory
ERROR: Request timeout
```

---

## 23.5 监控最佳实践

### 日常监控 Dashboard

```
必看指标:
  1. GPU KV Cache 使用率 → 超过 90% 需要注意
  2. waiting 请求数 → 持续 > 0 说明到达率超过处理能力
  3. 抢占次数 → > 0 说明显存压力大
  4. P95 TTFT 和 TPOT → 超过 SLA 需要告警
  5. token 吞吐 → 下降说明系统退化
```

### 容量规划

```
根据监控数据回答:
  1. 当前系统能承受多少 QPS？(看吞吐上限)
  2. 还有多少余量？(看 KV Cache 使用率)
  3. 什么时候需要扩容？(看 waiting 请求趋势)
```

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 指标获取 | `/metrics` 端点，Prometheus 格式 |
| 核心指标 | KV Cache 使用率、请求状态、吞吐、延迟 |
| 告警 | KV Cache > 95%、waiting > 50、频繁抢占 |
| 日志 | VLLM_LOGGING_LEVEL 控制级别 |

---

## 练习题

### 实践题

1. 搭建 Prometheus + Grafana 监控 vLLM 服务。
2. 压测时观察 KV Cache 使用率和抢占次数的变化。

### 思考题

3. 哪些指标能帮你判断"需要扩容"？
4. 如何区分"模型推理慢"和"排队等待长"？
