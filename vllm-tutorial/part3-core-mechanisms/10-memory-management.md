# 第10章：内存管理与块引擎

> 如果说 PagedAttention 是理论创新，那么 BlockSpaceManager 就是让这个理论真正运转起来的工程实现。它就像操作系统的内存管理子系统——看不见，但一旦出问题，整个系统就崩了。

---

## 学习目标

学完本章，你将能够：

1. 理解 BlockSpaceManager 的职责和工作方式
2. 掌握物理块的分配、释放和引用计数机制
3. 理解 GPU 块和 CPU 块（swap space）的管理
4. 跟踪一个请求完整生命周期中的块变化
5. 理解块管理对系统容量和性能的影响

---

## 10.1 BlockSpaceManager 概览

### 职责

BlockSpaceManager 是 vLLM 内存管理的核心组件，负责：

1. 管理 GPU 上的物理块池（分配和回收）
2. 管理 CPU 上的 swap 块池
3. 为每个请求维护块表（逻辑块到物理块的映射）
4. 处理块的共享和 Copy-on-Write
5. 计算可用块数，辅助调度器决策

### 初始化

```python
# BlockSpaceManager 在 vLLM 启动时初始化
# 它根据 GPU 显存和模型大小计算可用的物理块数量

total_gpu_memory = get_gpu_memory()        # 例如 80 GB
model_weight_memory = get_model_memory()   # 例如 16 GB
kv_cache_memory = total_gpu_memory * gpu_memory_utilization - model_weight_memory

# 每个块的大小
block_memory = 2 * num_layers * num_kv_heads * head_dim * block_size * dtype_size

# 可用物理块数
num_gpu_blocks = kv_cache_memory // block_memory
num_cpu_blocks = swap_space // block_memory
```

### 块池结构

```
GPU 块池:
┌──────┬──────┬──────┬──────┬──────┬──────┬───────┐
│ 块 0 │ 块 1 │ 块 2 │ 块 3 │ 块 4 │ ...  │ 块 N  │
│(used)│(free)│(used)│(free)│(used)│      │(free) │
└──────┴──────┴──────┴──────┴──────┴──────┴───────┘

CPU 块池 (swap):
┌──────┬──────┬──────┬──────┬───────┐
│ 块 0 │ 块 1 │ 块 2 │ ...  │ 块 M  │
│(free)│(free)│(free)│      │(free) │
└──────┴──────┴──────┴──────┴───────┘

空闲块列表: [1, 3, ...]
```

---

## 10.2 块的生命周期

### 分配

当一个新请求被接纳时，BlockSpaceManager 分配初始块：

```
请求 A 开始 (prompt = 50 tokens, block_size = 16):
  需要 ceil(50/16) = 4 个块

分配前空闲块: [0, 1, 2, 3, 4, 5, 6, 7, ...]

分配: 块 0, 1, 2, 3 分配给请求 A
块表 A: [逻辑0→物理0, 逻辑1→物理1, 逻辑2→物理2, 逻辑3→物理3]

分配后空闲块: [4, 5, 6, 7, ...]

块 3 的填充情况: [tok₄₉, tok₅₀, _, _, _, _, _, _, _, _, _, _, _, _, _, _]
                                 ↑ 还有 14 个 slot 可用
```

### 追加（生成新 token）

每生成一个新 token，将其 KV 向量写入当前最后一个块的下一个 slot：

```
生成第 51 个 token:
  块 3 还有空间 → 直接写入 slot 3
  块 3: [tok₄₉, tok₅₀, tok₅₁, _, _, ..., _]

生成第 64 个 token:
  块 3: [tok₄₉, tok₅₀, ..., tok₆₄]  ← 块 3 已满

生成第 65 个 token:
  需要新块 → 从空闲池分配块 4
  块表 A: [..., 逻辑4→物理4]
  块 4: [tok₆₅, _, _, ..., _]
```

### 释放

请求完成后，释放所有块回空闲池：

```
请求 A 完成:
  释放块 0, 1, 2, 3, 4
  空闲块: [0, 1, 2, 3, 4, 5, 6, 7, ...]
  块表 A: 删除
```

### 引用计数

当块被共享时（如并行采样），使用引用计数管理生命周期：

```
请求 A 使用 n=2 生成两个结果:

Prefill 后:
  序列 A₁ → 块表: [块0(ref=2), 块1(ref=2), 块2(ref=2)]
  序列 A₂ → 块表: [块0(ref=2), 块1(ref=2), 块2(ref=2)]

Decode 阶段（分叉）:
  序列 A₁ 生成新 token → 新块 3 (ref=1)
  序列 A₂ 生成新 token → 新块 4 (ref=1)

  序列 A₁: [块0(ref=2), 块1(ref=2), 块2(ref=2), 块3(ref=1)]
  序列 A₂: [块0(ref=2), 块1(ref=2), 块2(ref=2), 块4(ref=1)]

序列 A₁ 完成:
  释放块 3 (ref=1→0, 回收)
  块 0,1,2 的 ref: 2→1 (不回收，A₂ 还在用)

序列 A₂ 完成:
  释放块 4 (ref=1→0, 回收)
  块 0,1,2 的 ref: 1→0 (全部回收)
```

---

## 10.3 Swap 操作

### GPU → CPU（换出）

当调度器决定抢占一个请求时，BlockSpaceManager 执行 swap out：

```python
def swap_out(self, seq):
    """将请求的 KV Cache 从 GPU 换出到 CPU"""
    gpu_to_cpu_mapping = {}
    for logical_block in seq.block_table:
        gpu_block = logical_block.physical_block
        cpu_block = self.cpu_allocator.allocate()
        gpu_to_cpu_mapping[gpu_block] = cpu_block

    # 异步复制 GPU → CPU
    # 实际通过 CUDA memcpy async 实现
    return gpu_to_cpu_mapping
```

### CPU → GPU（换入）

当被换出的请求重新被调度时，执行 swap in：

```python
def swap_in(self, seq):
    """将请求的 KV Cache 从 CPU 换回 GPU"""
    cpu_to_gpu_mapping = {}
    for cpu_block in seq.swapped_blocks:
        gpu_block = self.gpu_allocator.allocate()
        cpu_to_gpu_mapping[cpu_block] = gpu_block

    # 异步复制 CPU → GPU
    return cpu_to_gpu_mapping
```

### Swap 的性能开销

```
PCIe Gen4 x16 带宽: ~32 GB/s
NVLink 带宽:         ~600 GB/s (GPU 间)

假设换出一个请求的 KV Cache = 0.5 GB:
  PCIe 传输时间 ≈ 0.5 / 32 ≈ 16ms
  
这 16ms 相当于多少个 decode iteration？
  如果 TPOT ≈ 20ms，那就是 ~1 个 iteration 的时间

结论：swap 的开销不可忽略，但对于长 prompt 请求来说，
     比重新 prefill 还是快得多。
```

---

## 10.4 请求完整生命周期中的块变化

### 全流程跟踪

```
时刻 T0: 请求到达 (prompt=100 tokens)
  状态: Waiting
  块: 无

时刻 T1: 被调度器接纳，开始 Prefill
  状态: Waiting → Running
  分配: ceil(100/16) = 7 个块
  块表: [块5, 块12, 块3, 块8, 块21, 块9, 块15]
  块 15 使用: 4/16 slots

时刻 T2-T50: Decode 阶段
  每生成 1 个 token，填充当前块
  块 15 满后，分配新块 22
  块 22 满后，分配新块 7
  ...

时刻 T60: 显存不足，被抢占 (swap)
  状态: Running → Swapped
  GPU 块释放，KV Cache 复制到 CPU 块
  CPU 块表: [cpu_0, cpu_1, ..., cpu_10]

时刻 T80: 显存充足，恢复
  状态: Swapped → Running
  CPU 块复制回 GPU 新块
  GPU 块表: [块2, 块14, 块6, ...]

时刻 T120: 生成完毕 (总计 200 tokens)
  状态: Running → 完成
  所有 GPU 块释放回空闲池
```

---

## 10.5 容量规划

### 计算最大并发数

```python
def estimate_max_concurrent(
    gpu_memory_gb: float,
    model_memory_gb: float,
    gpu_memory_utilization: float,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    avg_seq_len: int,
    dtype_bytes: int = 2,
) -> int:
    """估算最大并发请求数"""
    # 可用于 KV Cache 的显存
    available = gpu_memory_gb * gpu_memory_utilization - model_memory_gb

    # 每个块的大小 (GB)
    block_gb = (
        2 * num_layers * num_kv_heads * head_dim
        * block_size * dtype_bytes
    ) / (1024**3)

    # 总块数
    total_blocks = int(available / block_gb)

    # 每个请求需要的块数
    blocks_per_request = (avg_seq_len + block_size - 1) // block_size

    # 最大并发数
    return total_blocks // blocks_per_request

# 示例：A100 80GB, Llama-3.1-8B
max_concurrent = estimate_max_concurrent(
    gpu_memory_gb=80,
    model_memory_gb=16,
    gpu_memory_utilization=0.9,
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
    block_size=16,
    avg_seq_len=2048,
)
print(f"最大并发: {max_concurrent}")
```

### 显存分布

```
典型的 vLLM 显存分布 (A100 80GB, 7B 模型):

模型权重:     16 GB  (20%)
KV Cache:    56 GB  (70%)
激活值/开销:   8 GB  (10%)
─────────────────────────
总计:         80 GB  (100%)

KV Cache 部分是显存的主要使用者！
```

---

## 本章小结

| 概念 | 要点 |
|------|------|
| BlockSpaceManager | 管理物理块池、块表、引用计数 |
| 块生命周期 | 分配 → 填充 → 共享/CoW → 释放 |
| 引用计数 | 支持块共享，ref=0 时回收 |
| Swap | GPU ↔ CPU 块传输，开销 ~16ms/0.5GB |
| 容量规划 | 可用显存 / 每请求 KV Cache = 最大并发 |

---

## 动手实验

### 实验 1：估算你的 GPU 容量

用上面的公式计算你的 GPU 在不同模型和序列长度下的最大并发数。

### 实验 2：观察 KV Cache 使用率

```bash
# 启动 vLLM 后查看 KV Cache 信息
curl http://localhost:8000/metrics | grep gpu_cache
```

---

## 练习题

### 基础题

1. BlockSpaceManager 管理哪两种块池？
2. 块的引用计数在什么情况下会大于 1？
3. Swap out 操作的性能开销主要来自哪里？

### 实践题

4. 计算 A100 80GB 上运行 Llama-3.1-70B（4 卡张量并行）时，每张卡的最大并发数。

### 思考题

5. 如果 swap 空间（CPU 内存）也不够了，系统应该怎么处理？
6. 块大小为什么不能设得太小（如 1）？管理开销包括哪些？
