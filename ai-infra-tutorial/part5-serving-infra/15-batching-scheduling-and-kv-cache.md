# 第15章：批处理、调度与 KV Cache

> 现代 LLM 推理系统的核心竞争力，很多时候不体现在模型本身，而体现在调度器如何组织请求、缓存状态和显存。

## 学习目标

完成本章学习后，你将能够：

1. 理解批处理与调度在推理系统中的价值
2. 区分 Prefill / Decode 两个阶段的资源特征
3. 认识 KV Cache 与 PagedAttention 的核心思想
4. 学会从吞吐、延迟与显存三者之间做权衡
5. 读懂一个简化版 LLM 调度器在解决什么问题

---

## 正文内容

### 15.1 为什么批处理是推理系统的第一性问题

GPU 适合高吞吐批量计算。如果请求一个个独立执行，GPU 很可能无法高效利用。

因此，批处理的目标是：

- 合并时间上接近的请求
- 提高矩阵计算规模
- 摊薄单次 launch 和调度成本

但批处理不是免费午餐，因为它会引入等待时间。  
你可以把总延迟粗略写成：

$$
t_{\text{request}} = t_{\text{queue}} + t_{\text{batch\_compute}}
$$

批处理越激进，`t_batch_compute` 可能更经济，但 `t_queue` 往往会上升。  
这就是吞吐和延迟的第一层矛盾。

### 15.2 Prefill 与 Decode 的本质区别

LLM 推理通常分成两个阶段：

#### Prefill

- 处理用户输入 prompt
- 对整段上下文做 attention
- 更接近“大块计算”

#### Decode

- 一次生成一个 token
- 更强调调度粒度和缓存状态
- 请求长度差异更明显

这两个阶段的资源特征并不相同：

| 阶段 | 更像什么 | 常见瓶颈 |
|------|----------|----------|
| Prefill | 批量矩阵计算 | 算力、带宽 |
| Decode | 小步增量生成 | 调度、KV Cache、显存 |

好的推理系统不会把这两个阶段完全当成同一类工作负载处理。

### 15.3 KV Cache 为什么重要

如果每生成一个新 token 都重新计算全部历史上下文，复杂度会迅速变大。  
KV Cache 的核心思想是：

> 把历史 token 的 key / value 保留在显存里，后续 decode 只计算新增 token。

一个非常粗略的 KV Cache 显存估算式可以写成：

$$
M_{\text{kv}} \approx 2 \times L \times H \times T \times B \times \text{dtype\_bytes}
$$

其中：

- `L`：层数
- `H`：每层隐藏表示规模（或等价 head 维度总量）
- `T`：上下文长度
- `B`：并发请求数

这条式子揭示了两个现实：

1. 长上下文会直接把显存吃满
2. 高并发与长输出叠加时，KV Cache 可能比权重本身更难管理

### 15.4 为什么固定 batch 不够

普通批处理适合形状相近、耗时相近的任务。  
但 LLM 请求往往非常不规则：

- 输入长度不同
- 输出长度不同
- 有的请求几步就结束，有的请求持续很久

如果用固定 batch，会遇到：

- 短请求等长请求
- 已完成请求占着 batch 槽位
- 资源利用率不稳定

这就是为什么现代 LLM Serving 更强调 **continuous batching**。

### 15.5 一个简化版 continuous batching 调度器

可以用下面的伪代码理解它的目标：

```text
while service_is_running:
    collect newly arrived requests
    move ready requests into prefill queue
    run as much prefill as memory allows

    move prefilled requests into decode queue
    build current decode batch from active requests
    run one decode step

    remove finished requests
    recycle freed KV blocks
```

调度器真正解决的问题不是“如何把请求放进队列”这么简单，而是：

- 哪些请求先进
- 哪些请求可被延后
- 显存还够不够
- KV block 怎么回收
- 是否允许长请求饿死短请求

### 15.6 PagedAttention 在解决什么

如果 KV Cache 需要按连续显存预分配，那么会造成显著浪费：

- 请求长度未知
- 已完成请求释放后留下碎片
- 高并发下显存利用率下降

PagedAttention 的思路类似操作系统分页：

- 把 KV Cache 切成固定大小 block
- 逻辑上连续，物理上可不连续
- 通过映射表管理分配和回收

这意味着推理系统可以：

- 更高效地复用显存
- 降低碎片
- 更灵活地容纳不同长度请求

### 15.7 一个简单的调度权衡表

| 设计选择 | 收益 | 代价 |
|----------|------|------|
| 更大 batch | 更高吞吐 | 更长排队时间 |
| 更激进 KV Cache | 更快 decode | 更大显存压力 |
| 更强公平调度 | 降低饥饿 | 可能损失吞吐 |
| 更保守 admission | 更稳 | 峰值吞吐下降 |

所以调度器的本质不是“尽量快”，而是：

> 在显存、吞吐、延迟和公平性之间做持续的实时权衡。

### 15.8 常见误区

#### 误区一：LLM 推理优化就是换一个更快算子

不对。很多收益来自请求组织和缓存管理，而不是单个 kernel。

#### 误区二：KV Cache 只有收益，没有代价

不对。它在降低计算量的同时，把显存管理难度推到了前台。

#### 误区三：批处理越大越好

不对。在线服务还必须承担用户等待时间和尾延迟。

---

## 本章小结

| 技术 | 主要收益 | 主要代价 |
|------|----------|----------|
| Dynamic / Continuous Batching | 提升吞吐 | 增加排队与调度复杂度 |
| KV Cache | 降低重复计算 | 占用显存、带来回收问题 |
| PagedAttention | 提高显存利用率 | 映射与调度实现复杂 |

---

## 练习题

1. 为什么批处理策略会直接影响用户延迟？
2. Prefill 和 Decode 为什么应该区别对待？
3. KV Cache 为什么既是性能利器，也是显存压力来源？
4. Continuous batching 相比固定 batch 在工程上解决了什么问题？

