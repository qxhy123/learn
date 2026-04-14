# 第19章：张量并行

> 当模型大到一张 GPU 放不下时，你需要把模型"劈开"放到多张 GPU 上。张量并行是最常用的方式——它把每一层的权重矩阵水平切分，让多个 GPU 同时计算同一层。

---

## 学习目标

学完本章，你将能够：

1. 理解张量并行的切分策略和通信模式
2. 在 vLLM 中配置和使用张量并行
3. 分析张量并行的通信开销和扩展效率
4. 选择合适的 tensor_parallel_size
5. 理解张量并行对 KV Cache 的影响

---

## 19.1 为什么需要张量并行？

### 单卡显存限制

```
模型大小 vs GPU 显存:

模型        FP16权重    单卡A100(80GB)    需要几卡?
7B          14 GB       ✓ (余 66GB)       1
13B         26 GB       ✓ (余 54GB)       1
34B         68 GB       ✓ (余 12GB)       1 (勉强)
70B        140 GB       ✗                  2+
405B       810 GB       ✗                  11+
```

即使模型能放下，KV Cache 也需要显存——大模型 + 高并发往往需要多卡。

---

## 19.2 张量并行原理

### 切分策略

张量并行将每一层的权重矩阵按列或按行切分到不同 GPU：

```
原始矩阵 W [4096, 4096]
张量并行度 = 4

GPU 0: W₀ [4096, 1024]  ← 列切分
GPU 1: W₁ [4096, 1024]
GPU 2: W₂ [4096, 1024]
GPU 3: W₃ [4096, 1024]

计算:
  GPU 0: y₀ = W₀ × x   (局部计算)
  GPU 1: y₁ = W₁ × x
  GPU 2: y₂ = W₂ × x
  GPU 3: y₃ = W₃ × x
  
  全局结果: y = [y₀, y₁, y₂, y₃]  (需要 AllReduce 通信)
```

### Attention 层的切分

```
Multi-Head Attention (32 heads, TP=4):
  GPU 0: Head  0- 7  (8 heads)
  GPU 1: Head  8-15  (8 heads)
  GPU 2: Head 16-23  (8 heads)
  GPU 3: Head 24-31  (8 heads)

每个 GPU 独立计算自己负责的 attention heads
最后通过 AllReduce 合并结果
```

### 通信模式

```
每层需要 2 次 AllReduce:
  1. Attention 输出后: AllReduce
  2. FFN 输出后: AllReduce

AllReduce 的数据量 = hidden_size × batch_tokens × dtype_size
对于 hidden_size=4096, batch=32, FP16:
  每次 AllReduce ≈ 256 KB
  
每层 2 次 × 32 层 = 64 次 AllReduce/iteration
```

---

## 19.3 在 vLLM 中使用

### 基本配置

```python
from vllm import LLM, SamplingParams

# 使用 4 卡张量并行
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
)
```

### 命令行启动

```bash
# 需要先设置可见 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4
```

### 选择并行度

```
tensor_parallel_size 的选择:
  
  TP=1: 默认，单卡
  TP=2: 最常用，2 卡
  TP=4: 大模型常用
  TP=8: 超大模型（70B+ FP16）

原则:
  1. 必须能整除注意力头数
  2. 越少越好（通信开销越小）
  3. 优先使用 NVLink 连接的 GPU
```

---

## 19.4 性能分析

### 扩展效率

```
理想情况: TP=4 → 4× 吞吐
实际情况: TP=4 → 3.2-3.6× 吞吐 (80-90% 效率)

效率损失来自:
  1. AllReduce 通信开销
  2. GPU 间同步等待
  3. batch 较小时 GPU 利用率不足
```

### NVLink vs PCIe

```
NVLink (GPU 间直连):
  带宽: 600+ GB/s
  延迟: 低
  → TP 扩展效率 85-95%

PCIe (通过 CPU 桥接):
  带宽: 32 GB/s
  延迟: 较高
  → TP 扩展效率 60-80%

结论: 张量并行强烈推荐使用 NVLink 连接的 GPU
```

### KV Cache 的切分

```
KV Cache 也会被切分到各 GPU:

TP=4 时，每个 GPU 的 KV Cache:
  num_kv_heads_per_gpu = num_kv_heads / TP

对于 Llama-3.1-70B (8 KV heads, TP=4):
  每个 GPU 只有 2 个 KV head
  KV Cache 显存 = 原始的 1/4

好处: 每个 GPU 的 KV Cache 压力大幅降低
```

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 切分方式 | 列切分/行切分权重矩阵 |
| 通信 | 每层 2 次 AllReduce |
| NVLink | 对 TP 性能至关重要 |
| KV Cache | 也被切分，每卡压力降低 |
| 扩展效率 | NVLink 下 85-95%，PCIe 下 60-80% |

---

## 动手实验

### 实验 1：TP 性能对比

如果有多张 GPU，对比 TP=1 和 TP=2 的吞吐和延迟。

### 实验 2：大模型部署

使用 TP=4 部署一个 70B 模型，测量端到端性能。

---

## 练习题

### 基础题

1. 张量并行的通信模式是什么？每层需要几次通信？
2. 为什么 NVLink 对张量并行很重要？
3. `tensor_parallel_size` 对 KV Cache 有什么影响？

### 思考题

4. TP=2 和 TP=4 时，通信开销如何变化？是线性增长吗？
5. 如果有 8 张 GPU，模型需要 2 张 GPU 的显存，你会怎么选择 TP 和副本数？
