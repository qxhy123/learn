# 第22章：性能调优

> 性能调优不是到处改参数碰运气，而是一个系统化的过程：先确定瓶颈在哪，再针对性地优化。

---

## 学习目标

学完本章，你将能够：

1. 系统化地定位 vLLM 的性能瓶颈
2. 掌握关键引擎参数的调优方法
3. 使用 benchmark 工具进行性能测试
4. 理解不同工作负载的调优策略
5. 建立性能调优的思维框架

---

## 22.1 性能调优框架

### 四步法

```
1. 定义目标 → 你优化的是 TTFT、TPOT、吞吐还是并发？
2. 建立基线 → 用 benchmark 工具测量当前性能
3. 定位瓶颈 → 是 GPU 计算、内存带宽、KV Cache、还是调度？
4. 针对优化 → 调整对应参数，重新测量
```

### 瓶颈判断清单

| 指标 | 可能的瓶颈 | 调优方向 |
|------|-----------|---------|
| GPU 利用率低 (< 50%) | batch 太小 | 增加 max_num_seqs |
| GPU 显存满 | KV Cache 空间不足 | 量化、减小 max_model_len |
| TTFT 高 | prefill 计算量大 | chunked prefill、前缀缓存 |
| TPOT 高 | batch 太大或模型太大 | 减少并发、量化、TP |
| 频繁抢占 | 显存不足 | 减少并发、启用 prefix caching、量化 |
| 排队长 | 到达率超过处理能力 | 多副本、多 GPU |

---

## 22.2 关键参数调优

### GPU 显存利用率

```bash
# 平衡安全性和性能
# 低: 安全但浪费显存
# 高: 最大化吞吐但可能 OOM
vllm serve model --gpu-memory-utilization 0.9  # 默认值，通常合适
```

### 最大序列长度

```bash
# 根据实际需求设置，不要盲目用模型默认值
# 对话场景: 2048-4096 通常够用
# 长文本: 8192-32768
vllm serve model --max-model-len 4096
```

### 最大并发序列数

```bash
# 吞吐优先
vllm serve model --max-num-seqs 256

# 延迟优先
vllm serve model --max-num-seqs 32
```

### 前缀缓存

```bash
# 相同系统 prompt 多的场景，开启可显著降低 TTFT
vllm serve model --enable-prefix-caching
```

### Chunked Prefill

```bash
# 长 prompt 多的场景，开启可减少 TPOT 抖动
vllm serve model --enable-chunked-prefill --max-num-batched-tokens 2048
```

### V1 调度专属参数

当前 V1 引入了一组更精细的调度参数，很多旧教程没有覆盖：

```bash
# 控制长 prompt 如何被切分
vllm serve model \
    --max-num-partial-prefills 2 \         # 最多同时有几个请求在做 partial prefill
    --max-long-partial-prefills 1 \        # 其中长 prompt 的 partial prefill 最多几个
    --long-prefill-token-threshold 1024    # 超过多少 token 算"长 prompt"

# 异步调度（减少 GPU 空等 CPU 调度的时间）
vllm serve model --async-scheduling

# 调度策略
vllm serve model --scheduling-policy priority  # 支持 fcfs（默认）和 priority
```

这些参数直接对应 `vllm/vllm/config/scheduler.py` 中的 `SchedulerConfig`。

### 前缀缓存 hash 算法

如果你的场景对 prefix cache 命中率很敏感，可以调整 hash 算法：

```bash
vllm serve model \
    --enable-prefix-caching \
    --prefix-caching-hash-algo xxhash   # 更快，也可选 sha256（更稳妥）
```

---

## 22.3 Benchmark 工具

### vLLM 内置 benchmark

```bash
# 安装 benchmark 工具
pip install vllm[benchmark]

# 吞吐测试
python -m vllm.entrypoints.openai.api_server &

python -m vllm.benchmark.benchmark_serving \
    --backend vllm \
    --model model-name \
    --num-prompts 1000 \
    --request-rate 10 \
    --dataset-name sharegpt
```

### 自定义 benchmark

```python
import asyncio
import time
from openai import AsyncOpenAI

async def benchmark(num_requests, concurrency):
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="x")
    semaphore = asyncio.Semaphore(concurrency)

    async def single_request(i):
        async with semaphore:
            start = time.time()
            resp = await client.chat.completions.create(
                model="model-name",
                messages=[{"role": "user", "content": f"Write a short story #{i}"}],
                max_tokens=200,
            )
            return time.time() - start, resp.usage.completion_tokens

    start = time.time()
    results = await asyncio.gather(*[single_request(i) for i in range(num_requests)])
    total = time.time() - start

    latencies = [r[0] for r in results]
    tokens = sum(r[1] for r in results)

    print(f"总时间: {total:.1f}s")
    print(f"请求吞吐: {num_requests/total:.1f} req/s")
    print(f"Token 吞吐: {tokens/total:.0f} tok/s")
    print(f"P50 延迟: {sorted(latencies)[len(latencies)//2]:.2f}s")
    print(f"P99 延迟: {sorted(latencies)[int(len(latencies)*0.99)]:.2f}s")

asyncio.run(benchmark(500, concurrency=50))
```

---

## 22.4 场景化调优

### 低延迟场景（实时聊天）

```bash
vllm serve model \
    --max-num-seqs 32 \
    --max-model-len 2048 \
    --enable-prefix-caching \
    --enable-chunked-prefill
```

### 高吞吐场景（离线批处理）

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
    --max-num-seqs 16 \
    --enable-prefix-caching \
    --enable-chunked-prefill
```

---

## 本章小结

| 原则 | 要点 |
|------|------|
| 先测后优 | 用 benchmark 建立基线，不要凭直觉 |
| 定位瓶颈 | GPU 利用率、显存占用、抢占频率 |
| 关键参数 | max_num_seqs、max_model_len、gpu_memory_utilization |
| 场景导向 | 低延迟 vs 高吞吐，调优方向不同 |

---

## 练习题

### 实践题

1. 在你的 GPU 上运行 benchmark，记录基线性能。
2. 逐个调整关键参数，观察性能变化。

### 思考题

3. 吞吐和延迟能同时优化到最优吗？为什么？
4. 什么指标能告诉你"系统已经到达性能上限"？
