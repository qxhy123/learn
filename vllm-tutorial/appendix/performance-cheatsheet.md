# 附录B：性能调优速查表

## 快速诊断

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| TTFT 高 | Prompt 长、Prefill 慢 | 启用前缀缓存、chunked prefill |
| TPOT 高 | Batch 太大、模型太大 | 减少 max_num_seqs、量化、TP |
| TPOT 抖动 | 长 prefill 阻塞 decode | 启用 chunked prefill |
| GPU 利用率低 | Batch 太小 | 增加 max_num_seqs |
| GPU 显存满 | KV Cache 不足 | 减小 max_model_len、量化 |
| 频繁抢占 | 并发太高、序列太长 | 减少 max_num_seqs、增加 swap |
| 排队严重 | 到达率超过处理能力 | 多副本、多 GPU |
| 模型加载慢 | 模型大、网络慢 | 本地存储、safetensors 格式 |
| OOM | 显存不足 | 量化、限制 max_model_len、TP |

---

## 场景配置模板

### 低延迟实时聊天

```bash
vllm serve model \
    --max-num-seqs 32 \
    --max-model-len 2048 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048
```

### 高吞吐离线处理

```bash
vllm serve model \
    --max-num-seqs 256 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95
```

### 长文本场景

```bash
vllm serve model \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.95
```

### 显存受限（16GB GPU）

```bash
vllm serve model-awq \
    --quantization awq \
    --max-model-len 2048 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.9
```

### 多 GPU 大模型

```bash
vllm serve large-model \
    --tensor-parallel-size 4 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9
```

---

## 参数影响速查

### max_num_seqs

```
↑ 增大 → 吞吐 ↑, GPU 利用率 ↑, 延迟可能 ↑
↓ 减小 → 延迟 ↓, 吞吐 ↓
```

### max_model_len

```
↑ 增大 → 支持更长输入, 并发 ↓ (KV Cache 空间减少)
↓ 减小 → 并发 ↑, 但限制输入长度
```

### gpu_memory_utilization

```
↑ 增大 → 更多 KV Cache → 更高并发/吞吐
↓ 减小 → 更安全, 但浪费显存
推荐: 0.85-0.95
```

### block_size

```
默认 16, 通常不需要修改
↑ 增大 → 管理开销低, 内部碎片大
↓ 减小 → 碎片小, 管理开销高
```

---

## 模型显存估算公式

### 模型权重

```
FP16:  参数量 × 2 bytes
INT8:  参数量 × 1 byte
INT4:  参数量 × 0.5 bytes

例: 7B FP16 = 7×10⁹ × 2 = 14 GB
```

### KV Cache（单请求）

```
KV Cache = 2 × 层数 × KV头数 × head_dim × 序列长度 × dtype_bytes

例: Llama-3.1-8B, FP16, seq=4096
    = 2 × 32 × 8 × 128 × 4096 × 2 = 1.07 GB
```

### 最大并发估算

```
最大并发 ≈ (GPU显存 × 利用率 - 模型权重 - 开销) / 单请求KV Cache
```

---

## Prometheus 关键指标

| 指标 | 健康范围 | 告警阈值 |
|------|---------|---------|
| `gpu_cache_usage_perc` | < 0.85 | > 0.95 |
| `num_requests_waiting` | 0 | > 50 持续 2min |
| `num_preemptions_total` (rate) | 0 | > 1/min |
| TTFT P95 | < 2s | > 5s |
| TPOT P95 | < 50ms | > 100ms |

---

## 调优优先级

```
1. 确认模型和量化方案是否合适
2. 设置合理的 max_model_len（不要用默认的超大值）
3. 调整 max_num_seqs（平衡延迟和吞吐）
4. 启用前缀缓存（如果有共享前缀）
5. 启用 chunked prefill（如果 TPOT 抖动）
6. 考虑 KV Cache FP8（显存紧张时）
7. 考虑多 GPU / 多副本（性能仍不足时）
```
