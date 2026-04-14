# 第8章：连续批处理

> 静态批处理像公交车——必须等所有乘客到齐才发车。连续批处理像出租车调度——随到随走，有人下车就接新客。

---

## 学习目标

学完本章，你将能够：

1. 理解静态批处理的局限性以及它为什么浪费 GPU 资源
2. 掌握连续批处理（Continuous Batching）的工作原理
3. 理解 iteration-level 调度如何实现请求的动态加入和退出
4. 分析连续批处理对吞吐和延迟的影响
5. 理解 prefill 和 decode 请求混合调度的挑战

---

## 8.1 静态批处理的问题

### 什么是静态批处理？

静态批处理（Static Batching）是最朴素的批处理方式：

1. 收集一批请求
2. 将所有请求 padding 到相同长度
3. 同时进行前向计算
4. 等所有请求都生成完毕
5. 返回结果，开始下一批

### 浪费在哪里？

```
时间 →
请求 A: [████████████████████░░░░░░░░░░░░░░░]  生成 20 个 token 后结束
请求 B: [████████████████████████████████████]  生成 35 个 token 后结束
请求 C: [████████████░░░░░░░░░░░░░░░░░░░░░░░]  生成 12 个 token 后结束

█ = 有效计算    ░ = 等待其他请求完成（GPU 资源浪费）
```

问题 1：**请求 A 和 C 完成后，GPU 仍然在为它们分配计算资源**，直到最慢的请求 B 完成。

问题 2：**新请求必须等当前 batch 全部完成**，才能被处理。

问题 3：**不同请求的输入长度不同**，需要 padding，padding 部分的计算也是浪费。

### 量化浪费

假设一个 batch 有 32 个请求，生成长度分布在 50-500 token：

- 最长请求生成 500 token
- 平均长度 200 token
- 浪费比例：1 - 200/500 = 60%

也就是说，**GPU 有 60% 的时间在做无用功**。

---

## 8.2 连续批处理的原理

### 核心思想

连续批处理在**每个 iteration（前向计算步骤）** 级别进行调度：

- 当一个请求在某次 iteration 后生成了 EOS，**立即从 batch 中移除**
- 如果有等待中的请求，**立即加入当前 batch**
- 不需要等其他请求完成

### 工作流程

```
Iteration 1: [A₁, B₁, C₁]           → 3 个请求同时做第 1 次 decode
Iteration 2: [A₂, B₂, C₂]           → 3 个请求同时做第 2 次 decode
...
Iteration 12: [A₁₂, B₁₂, C₁₂]      → C 生成了 EOS
Iteration 13: [A₁₃, B₁₃, D₁]       → C 移除，D 加入（prefill）
...
Iteration 20: [A₂₀, B₂₀, D₈]       → A 生成了 EOS
Iteration 21: [E₁, B₂₁, D₉]        → A 移除，E 加入
...
```

### 视觉对比

```
静态批处理：

  [A████████████░░░░░░░░]
  [B████████████████████]
  [C████░░░░░░░░░░░░░░░]
  ──────────────────────→ 时间
                         ↑ 整个 batch 完成后才能处理新请求

连续批处理：

  [A████████████]
  [B████████████████████]
  [C████][D████████████]
  [     ][E█████████]
  ──────────────────────→ 时间
      ↑ C结束后D立即加入
            ↑ A结束后E立即加入
```

---

## 8.3 Iteration-Level 调度

### 每次 Iteration 的流程

在 vLLM 中，每次 iteration 包含以下步骤：

```
1. 调度器决策
   ├── 检查哪些正在运行的请求已完成 → 移除
   ├── 检查等待队列中是否有新请求 → 可以加入
   ├── 检查显存是否足够 → 决定是否接纳新请求
   └── 输出本次 iteration 的执行计划

2. 准备输入
   ├── 对正在运行的 decode 请求：准备 1 个新 token 的输入
   └── 对新加入的 prefill 请求：准备整个 prompt 的输入

3. 执行前向计算
   ├── 所有请求的 Q 在同一个 batch 中计算
   └── 各请求通过各自的块表访问自己的 KV Cache

4. 处理输出
   ├── 采样得到每个请求的下一个 token
   ├── 判断是否到达停止条件
   └── 对流式请求返回新 token
```

### 混合 Prefill 和 Decode

一个有趣的问题：**新请求的 prefill 和旧请求的 decode 可以在同一个 iteration 中执行吗？**

答案是可以，但需要处理两个挑战：

1. **计算量不对称**：prefill 请求有大量 token，decode 请求只有 1 个 token
2. **注意力计算的差异**：prefill 是自注意力，decode 是增量注意力

vLLM 的解决方案：

```
同一次 Iteration 中：

Decode 请求:  [tok₁]  [tok₁]  [tok₁]    ← 每个只有 1 个 token
Prefill 请求: [tok₁, tok₂, ..., tok₅₀₀]  ← 有 500 个 token

↓ 合并到一个 batch 中计算 ↓

需要处理：
- 不同请求有不同数量的 token
- 注意力掩码的形状不同
- 使用 Flash Attention 的 varlen 接口处理可变长度
```

### Chunked Prefill

当 prefill 请求的 prompt 很长时，一次性处理会导致：
- 这次 iteration 的计算量激增
- 正在 decode 的请求等待时间变长（TPOT 抖动）

**Chunked Prefill** 将长 prompt 分成多个 chunk，分散到多个 iteration 中处理：

```
没有 Chunked Prefill：
Iter 1: [D₁, D₂, D₃, P(500 tokens)]  ← 这次 iteration 非常慢

有 Chunked Prefill（chunk_size=128）：
Iter 1: [D₁, D₂, D₃, P_chunk1(128)]
Iter 2: [D₁, D₂, D₃, P_chunk2(128)]
Iter 3: [D₁, D₂, D₃, P_chunk3(128)]
Iter 4: [D₁, D₂, D₃, P_chunk4(116)]
Iter 5: [D₁, D₂, D₃, D₄_new]         ← P 的 prefill 完成，开始 decode
```

---

## 8.4 吞吐分析

### 为什么连续批处理能提升吞吐？

关键原因是 **GPU 利用率的提升**：

1. **消除等待浪费**：请求完成后立即释放资源给新请求
2. **保持 batch 饱满**：始终尽量让更多请求同时执行
3. **减少排队等待**：新请求不需要等当前 batch 完成

### 吞吐提升的定量分析

假设请求到达率为 λ，平均生成长度为 L：

**静态批处理**：
- 每个 batch 的处理时间 = max(各请求生成长度) × TPOT
- 有效 token 数 = sum(各请求实际长度)
- GPU 利用率 ≈ 平均长度 / 最大长度

**连续批处理**：
- 每次 iteration 的 batch 几乎总是满的（受显存限制）
- GPU 利用率 ≈ 接近 100%（忽略显存限制外的开销）

### 实际提升幅度

根据 vLLM 论文和实际测试，连续批处理相比静态批处理的吞吐提升：

| 场景 | 提升倍数 |
|------|---------|
| 短对话（输出 50-100 token） | 2-3× |
| 长文本生成（输出 500+ token） | 3-5× |
| 高变异场景（输出长度差异大） | 5-10× |

输出长度差异越大，连续批处理的优势越明显。

---

## 8.5 延迟的影响

### TTFT（首 Token 延迟）

连续批处理对 TTFT 的影响是双面的：

- **好处**：新请求不需要等当前 batch 完成，可以更快被调度
- **代价**：如果当前 batch 很大，prefill 可能被延迟（需要等显存释放）

### TPOT（Token 间延迟）

连续批处理可能增加 TPOT：

- 更大的 batch 意味着每次 iteration 的计算量更大
- 但也意味着 GPU 利用率更高
- 在合理范围内，TPOT 增加很小

### 延迟-吞吐权衡

```python
# vLLM 通过 max_num_seqs 控制最大 batch 大小
# 较小的值 → 较低延迟，较低吞吐
# 较大的值 → 较高吞吐，可能增加延迟

# 低延迟优先
llm = LLM(model="...", max_num_seqs=16)

# 高吞吐优先
llm = LLM(model="...", max_num_seqs=256)
```

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 静态批处理 | 等最慢的请求完成，大量 GPU 资源浪费 |
| 连续批处理 | iteration 级别动态添加/移除请求 |
| 核心优势 | 保持 batch 饱满，GPU 利用率接近 100% |
| 吞吐提升 | 2-10×，取决于输出长度的变异程度 |
| Chunked Prefill | 将长 prompt 分块处理，减少 TPOT 抖动 |
| 延迟权衡 | 更大 batch = 更高吞吐但可能增加单请求延迟 |

---

## 动手实验

### 实验 1：模拟对比

```python
import random
import time

def simulate_static_batching(requests, batch_size):
    """模拟静态批处理"""
    total_iterations = 0
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i+batch_size]
        # 必须等最长的请求完成
        total_iterations += max(batch)
    return total_iterations

def simulate_continuous_batching(requests, max_batch_size):
    """模拟连续批处理"""
    total_iterations = 0
    running = []  # (remaining_tokens,)
    queue = list(requests)

    while running or queue:
        # 添加新请求到 batch
        while queue and len(running) < max_batch_size:
            running.append(queue.pop(0))

        # 一次 iteration
        total_iterations += 1
        running = [r - 1 for r in running]
        running = [r for r in running if r > 0]

    return total_iterations

# 生成 100 个请求，长度在 50-500 之间
requests = [random.randint(50, 500) for _ in range(100)]

static = simulate_static_batching(requests, batch_size=32)
continuous = simulate_continuous_batching(requests, max_batch_size=32)

print(f"静态批处理:   {static} iterations")
print(f"连续批处理:   {continuous} iterations")
print(f"效率提升:     {static / continuous:.1f}×")
```

### 实验 2：vLLM 吞吐测试

使用 vLLM 对比不同 max_num_seqs 设置下的吞吐和延迟。

---

## 练习题

### 基础题

1. 静态批处理中，如果一个 batch 的 4 个请求分别生成 10、50、200、500 个 token，GPU 利用率大约是多少？
2. 连续批处理为什么能提升 GPU 利用率？
3. Chunked Prefill 解决的是什么问题？

### 实践题

4. 运行模拟代码，改变请求长度分布（如均匀 vs 极端差异），观察连续批处理的优势变化。

### 思考题

5. 在什么场景下，连续批处理相比静态批处理的优势最大？最小？
6. 如果所有请求都生成完全相同数量的 token，连续批处理还有优势吗？
