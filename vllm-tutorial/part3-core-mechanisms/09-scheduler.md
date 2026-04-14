# 第9章：调度器与请求管理

> 调度器是 vLLM 的大脑。它在每次 iteration 前做出关键决策：谁先执行？显存不够怎么办？新请求什么时候能被接纳？

---

## 学习目标

学完本章，你将能够：

1. 理解 vLLM 调度器的整体设计和决策流程
2. 掌握请求的三种状态（waiting、running、swapped）及其转换
3. 理解抢占（preemption）机制及其两种策略
4. 理解调度策略对延迟和吞吐的影响
5. 掌握关键调度参数的调优方法

---

## 9.1 调度器的角色

### 调度器需要回答的问题

在每次 iteration 之前，调度器需要做出以下决策：

1. **哪些正在运行的请求可以继续？**（检查显存是否足够为它们分配新块）
2. **哪些等待中的请求可以被接纳？**（检查是否有足够的显存启动新请求）
3. **如果显存不足怎么办？**（是否需要抢占某些请求？）
4. **如何在 prefill 和 decode 之间分配资源？**

### 请求的三种状态

```
                ┌─── 显存充足 ───→ Running ←── 恢复 ──┐
                │                    │                  │
Waiting ────────┘                    │ 显存不足          │
  ↑                                  ↓                  │
  │                               Swapped ──────────────┘
  │                                  │
  └────────── 请求完成/取消 ──────────┘
```

**Waiting（等待）**：请求已到达但尚未开始执行（尚未做 prefill）。

**Running（运行中）**：请求正在执行中，已有 KV Cache 分配在 GPU 上。

**Swapped（换出）**：请求被暂停，KV Cache 从 GPU 换出到 CPU 内存。

---

## 9.2 调度决策流程

### 每次 Iteration 的调度逻辑

```python
def schedule(self):
    # 阶段 1：处理 running 队列
    # 为每个运行中的请求尝试分配新块（用于下一个 token）
    running_scheduled = []
    preempted = []

    for seq in self.running:
        if can_allocate_new_block(seq):
            running_scheduled.append(seq)
        else:
            # 显存不足，需要抢占
            preempted.append(seq)

    # 阶段 2：处理 swapped 队列
    # 尝试恢复之前被换出的请求
    swapped_in = []
    for seq in self.swapped:
        if can_restore(seq):
            swapped_in.append(seq)

    # 阶段 3：处理 waiting 队列
    # 尝试接纳新请求
    newly_admitted = []
    for seq in self.waiting:
        if can_admit(seq):
            newly_admitted.append(seq)

    # 返回调度结果
    return SchedulerOutput(
        scheduled_running=running_scheduled,
        scheduled_swapped=swapped_in,
        scheduled_new=newly_admitted,
        preempted=preempted,
    )
```

### 优先级原则

vLLM 调度器遵循以下优先级（从高到低）：

1. **Running 请求继续执行**：已经在运行的请求优先，避免浪费已计算的 KV Cache
2. **Swapped 请求恢复**：已有部分 KV Cache 的请求优先于全新请求
3. **Waiting 请求接纳**：排队最久的新请求

这个优先级的逻辑是：已经投入资源的请求应该优先完成，否则之前的计算就白费了。

---

## 9.3 抢占机制

### 什么时候会触发抢占？

当 GPU 显存不足以为所有 running 请求分配新块时，调度器需要抢占一些请求来释放显存。

典型场景：
- 大量请求同时运行，KV Cache 增长到显存上限
- 某些请求的输出特别长，占用越来越多的块

### 抢占的两种策略

**策略 1：Recompute（重计算）**

- 将被抢占请求的 KV Cache 直接丢弃
- 请求回到 waiting 队列
- 再次被调度时需要重新做 prefill

```
优点：实现简单，释放显存快
缺点：已完成的 KV Cache 计算被浪费
适用：prefill 成本低的场景（短 prompt）
```

**策略 2：Swap（换出到 CPU）**

- 将 KV Cache 从 GPU 复制到 CPU 内存
- 请求进入 swapped 队列
- 恢复时将 KV Cache 从 CPU 复制回 GPU

```
优点：不浪费已计算的 KV Cache
缺点：GPU-CPU 之间的数据传输有开销
适用：prefill 成本高的场景（长 prompt）
```

### 抢占的选择策略

vLLM 默认选择抢占哪个请求的策略：

- **FCFS（先来先服务的反向）**：最后到达的请求最先被抢占
- 直觉：先到的请求已经投入更多计算，应该被保护

```
Running 请求（按到达时间）:
  请求 A (到达最早) → 最后被抢占
  请求 B
  请求 C
  请求 D (到达最晚) → 最先被抢占
```

---

## 9.4 调度策略

### FCFS 调度（默认）

先来先服务，最简单也最公平：

- 等待队列按到达时间排序
- 先到的请求先被处理
- 保证每个请求最终都能被执行

### 优先级调度

vLLM 支持请求级别的优先级设置：

```python
# 通过 API 设置优先级
response = client.chat.completions.create(
    model="model-name",
    messages=[...],
    extra_body={"priority": 1},  # 更低的数字 = 更高优先级
)
```

### Prefill 与 Decode 的调度

一个重要的调度决策是如何平衡 prefill 和 decode：

**Decode 优先**：
- 优先为 running 请求做 decode
- 新请求的 prefill 可能被延迟
- 效果：TPOT 更稳定，但 TTFT 可能变长

**Prefill 优先**：
- 优先处理新请求的 prefill
- 可能暂时影响 decode 请求的 TPOT
- 效果：TTFT 更短，但 TPOT 可能抖动

vLLM 的默认行为倾向于 decode 优先，因为保护已在运行的请求能避免资源浪费。

---

## 9.5 调度相关参数

### 关键参数

```bash
vllm serve model \
    --max-num-seqs 256 \            # 最大并发序列数
    --max-num-batched-tokens 8192 \  # 每次 iteration 最大 token 数
    --swap-space 4 \                 # CPU swap 空间 (GB)
    --preemption-mode recompute \    # 抢占策略: recompute 或 swap
    --scheduler-delay-factor 0.0     # 调度延迟因子
```

### max-num-seqs 的影响

```
max-num-seqs 较小 (如 16):
  + 每个请求延迟更低
  + TPOT 更稳定
  - 吞吐较低
  - GPU 可能利用率不足

max-num-seqs 较大 (如 256):
  + 吞吐更高
  + GPU 利用率高
  - 单请求延迟可能增加
  - 显存压力大，可能触发抢占
```

### max-num-batched-tokens 的影响

这个参数限制每次 iteration 中处理的 token 总数，主要影响 prefill：

```
较小值 (如 2048):
  + prefill 不会阻塞 decode 太久
  + TPOT 更稳定
  - 长 prompt 需要更多 iteration 来完成 prefill
  - TTFT 可能增加

较大值 (如 32768):
  + 长 prompt 可以更快完成 prefill
  + TTFT 更短
  - 可能导致 TPOT 抖动
```

### swap-space 的影响

```
swap-space=0:
  - 只能用 recompute 抢占
  - 适合短 prompt 场景

swap-space=4 (4 GB):
  - 可以将 KV Cache 换出到 CPU 内存
  - 适合长 prompt 或高并发场景
  - 消耗 CPU 内存
```

---

## 9.6 调度可视化

### 理解调度行为

通过日志可以观察 vLLM 的调度决策：

```bash
# 启动时设置详细日志
VLLM_LOGGING_LEVEL=DEBUG vllm serve model

# 日志中会看到：
# Scheduling: running=32, waiting=5, swapped=0
# Preempting 2 sequences (swap)
# Admitting 3 new sequences
```

### 调度指标

vLLM 通过 Prometheus 暴露调度相关指标：

| 指标 | 含义 |
|------|------|
| `vllm:num_requests_running` | 当前运行中的请求数 |
| `vllm:num_requests_waiting` | 当前等待中的请求数 |
| `vllm:num_requests_swapped` | 当前被换出的请求数 |
| `vllm:num_preemptions_total` | 累计抢占次数 |
| `vllm:gpu_cache_usage_perc` | GPU KV Cache 使用率 |

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 调度器角色 | 每次 iteration 前决定执行哪些请求 |
| 三种状态 | Waiting → Running ↔ Swapped |
| 优先级 | Running > Swapped > Waiting |
| 抢占策略 | Recompute（丢弃重算）或 Swap（换出到 CPU） |
| 关键参数 | max-num-seqs、max-num-batched-tokens、swap-space |
| 核心权衡 | 并发越高 → 吞吐越高 → 但延迟可能增加 |

---

## 动手实验

### 实验 1：观察调度行为

启动 vLLM 服务器并发送大量请求，观察 running/waiting/swapped 状态的变化。

### 实验 2：触发抢占

发送大量长输出请求，尝试触发 OOM 和抢占，观察日志中的抢占信息。

### 实验 3：参数调优

对比 `max-num-seqs=16` 和 `max-num-seqs=256` 的吞吐和延迟差异。

---

## 练习题

### 基础题

1. vLLM 中请求的三种状态是什么？它们之间如何转换？
2. 抢占的两种策略分别是什么？各自的优缺点？
3. 为什么 vLLM 优先保护 running 请求而非 waiting 请求？

### 实践题

4. 通过 Prometheus 指标，观察高并发下的 `num_requests_running` 和 `gpu_cache_usage_perc` 变化。

### 思考题

5. 在什么场景下，swap 策略比 recompute 更好？反过来呢？
6. 如果一个应用同时有实时聊天（低延迟）和离线翻译（高吞吐）两种请求，你会如何配置调度参数？
