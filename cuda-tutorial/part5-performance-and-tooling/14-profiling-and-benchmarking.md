# 第14章：Profiling 与 Benchmarking

> CUDA 优化最危险的习惯，不是“不会调优”，而是还没测量就开始自信地下结论。

---

## 学习目标

学完本章，你将能够：

1. 理解为什么 CUDA 性能分析必须建立在正确测量之上
2. 知道 wall-clock、CUDA events、系统级 tracing 分别适合回答什么问题
3. 理解 warmup、重复测量和同步在 benchmark 中的重要性
4. 对 Nsight Systems 与 Nsight Compute 的角色分工建立清晰认知
5. 能避免若干最常见的 CUDA 测量误区

---

## 14.1 为什么“看起来快了”完全不够？

CUDA 程序有很多异步行为，例如：

- kernel launch 往往不是立即阻塞 host
- 某些拷贝和执行可以异步进行
- 多个 stream 之间还可能重叠

这意味着：

- 你用普通 CPU 计时方式测到的，并不一定是实际 GPU 执行时间
- 你以为自己在测 kernel，可能实际只测到了 launch 开销
- 你以为一个优化有效，可能只是缓存、warmup 或输入规模造成的假象

所以 CUDA 性能分析的第一原则是：

- **先把测量做对，再讨论优化。**

---

## 14.2 你到底在测什么？

在测量之前，先明确你要回答的问题。

### 问题一：整个程序运行多久？

这更接近 wall-clock 时间，适合看端到端效果。

### 问题二：某个 kernel 本身执行多久？

这通常更适合用 CUDA events 或 profiler。

### 问题三：GPU、CPU、拷贝、kernel 之间的时间线关系是什么？

这类问题更适合系统级 tracing 工具，如 Nsight Systems。

### 问题四：这个 kernel 的瓶颈究竟在算术、访存还是发射效率？

这更适合用 kernel 级分析工具，如 Nsight Compute。

一旦问题不同，适合的工具就不同。

---

## 14.3 最基础的错误：没同步就开始计时

假设你这样写：

```cpp
auto start = std::chrono::high_resolution_clock::now();
my_kernel<<<blocks, threads>>>(...);
auto end = std::chrono::high_resolution_clock::now();
```

这通常并不能准确表示 kernel 执行时间，因为：

- kernel launch 可能是异步的
- host 线程很快就继续往下跑了

### 更严谨的最小修正

```cpp
auto start = std::chrono::high_resolution_clock::now();
my_kernel<<<blocks, threads>>>(...);
cudaDeviceSynchronize();
auto end = std::chrono::high_resolution_clock::now();
```

这样至少能确保：

- 在记录结束时间前，GPU 已经执行完这个 kernel

虽然它还不是最细粒度、最推荐的 kernel 计时方式，但比“完全不同步”可靠得多。

---

## 14.4 用 CUDA events 测 kernel 时间

对于单个 kernel 或一段 GPU 工作，CUDA events 是很常见的基础计时方法。

### 基本思路

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
my_kernel<<<blocks, threads>>>(...);
cudaEventRecord(stop);

cudaEventSynchronize(stop);

float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
```

### 为什么它更适合 GPU 计时？

因为它测量的是 GPU 时间线中的事件间隔，而不是单纯依赖 host 侧时钟去猜测。

### 它适合什么？

- 测单个 kernel 时间
- 测一组 GPU 操作总耗时
- 做基础 benchmark

---

## 14.5 为什么要做 warmup？

很多 CUDA 程序第一次运行时，并不能代表稳定状态。

原因可能包括：

- 上下文初始化
- 内核首次加载
- cache 尚未热起来
- JIT 编译或惰性初始化

因此，benchmark 时通常不应直接拿第一次结果做结论。

### 一个常见做法

- 先运行若干次 warmup
- 再进行正式计时
- 取平均值、最小值或中位数等统计量

这能减少偶然因素影响。

---

## 14.6 为什么要重复测量？

单次结果很容易受这些因素影响：

- 系统抖动
- 温度与频率变化
- 其他进程干扰
- 输入数据差异

因此更合理的 benchmark 方式通常是：

1. 固定输入规模
2. warmup 若干次
3. 正式重复执行多轮
4. 记录统计结果

常见统计方式包括：

- 平均值
- 中位数
- 最小值
- 标准差（如果需要更严格分析）

初学阶段至少要做到：

- 不用单次结果下结论

---

## 14.7 Nsight Systems 与 Nsight Compute 分别看什么？

这是非常重要的工具分工。

### Nsight Systems

它更像是“全局时间线观察器”。

适合回答：

- CPU 和 GPU 是否在重叠工作？
- 哪些 memcpy 和 kernel 按什么顺序发生？
- stream 是否真正实现了并发和 overlap？
- 程序时间花在大块的哪里？

如果你的问题是：

- “为什么 pipeline 没有重叠？”
- “为什么 GPU 有空闲段？”
- “为什么 host 在等？”

通常更先看 Nsight Systems。

### Nsight Compute

它更像是“单个 kernel 的体检报告”。

适合回答：

- kernel 的 memory throughput 如何？
- occupancy 怎样？
- 是否存在明显的 memory bottleneck？
- warp 执行效率如何？
- 指令、访存、缓存等细节指标怎样？

如果你的问题是：

- “这个 kernel 为什么慢？”
- “瓶颈在内存还是算术？”
- “coalescing 是否有问题？”

通常更先看 Nsight Compute。

---

## 14.8 benchmark 里最常见的错误清单

### 14.8.1 把数据初始化时间和 kernel 时间混在一起

如果你的目标是测 kernel，就不要把：

- host 数据构造
- 文件读取
- 内存分配
- 随机数初始化

混进同一段计时里，除非你明确要测端到端性能。

### 14.8.2 每次测量都重新分配显存

如果你想比较 kernel 本身，频繁 `cudaMalloc` / `cudaFree` 会污染结果。

### 14.8.3 输入规模太小

太小的问题规模会让：

- launch overhead
- 测量误差
- 调度抖动

相对变得过大，结果不稳定也不具代表性。

### 14.8.4 忘记验证结果正确

一个“更快”的 kernel 如果结果错了，就没有任何意义。

### 14.8.5 只看时间，不看吞吐和规模

相同的毫秒数，在不同输入规模下意义完全不同。适当记录：

- 元素数量
- 字节数
- GFLOPS / GB/s（如果适用）

会更有解释力。

---

## 14.9 一个基础 benchmark 骨架

```cpp
// warmup
for (int i = 0; i < 10; ++i) {
    my_kernel<<<blocks, threads>>>(...);
}
cudaDeviceSynchronize();

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
for (int i = 0; i < 100; ++i) {
    my_kernel<<<blocks, threads>>>(...);
}
cudaEventRecord(stop);

cudaEventSynchronize(stop);

float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
std::cout << "avg kernel time = " << ms / 100.0f << " ms" << std::endl;
```

这段骨架的意义在于：

- 先 warmup
- 多次运行取平均
- 使用 event 测 GPU 执行区间

它并不覆盖所有严谨 benchmark 细节，但已经比“单次 launch + host 直接计时”可靠很多。

---

## 14.10 怎么把 profiling 结果用于真正优化？

一个合理流程通常是：

1. 先确认结果正确
2. 用基础 benchmark 建立基线
3. 用 Nsight Systems 看全局时间线和重叠关系
4. 用 Nsight Compute 看重点 kernel 的瓶颈指标
5. 每次只改一个变量
6. 重新测量并记录结果

这比“想当然地连改十个点再看有没有变快”更有效，也更容易解释原因。

---

## 本章小结

| 核心概念 | 要点 |
|----------|------|
| 测量前提 | 先明确你是在测端到端、单个 kernel，还是系统时间线 |
| 同步 | 没同步就计时，结果通常不可靠 |
| CUDA events | 基础 kernel 计时常用方法 |
| warmup | 首次运行常不代表稳定性能 |
| 重复测量 | 避免用单次结果下结论 |
| 工具分工 | Nsight Systems 看全局时间线，Nsight Compute 看单 kernel 细节 |

---

## CUDA实验

### 实验 1：修正一个错误的 benchmark

给定一段代码：

```cpp
auto start = now();
my_kernel<<<blocks, threads>>>(...);
auto end = now();
```

请指出：

1. 为什么它不可靠
2. 至少应该如何改进
3. 为什么 `cudaDeviceSynchronize()` 或 CUDA events 能提高可信度

### 实验 2：设计一个最小 benchmark 流程

为一个向量加法 kernel 设计实验流程，要求包含：

- warmup
- 正式重复执行
- 结果正确性验证
- 平均时间统计

你可以只写伪代码和步骤，不必立刻完整实现。

---

## 练习题

1. **基础题**：为什么很多 CUDA kernel 不能直接用 host 侧开始/结束时间简单相减来精确计时？
2. **基础题**：CUDA events 和 wall-clock 计时各更适合回答什么问题？
3. **实现题**：写出一段用 CUDA events 测量 kernel 平均时间的伪代码。
4. **思考题**：为什么 benchmark 时需要 warmup 和多次重复执行？
5. **思考题**：Nsight Systems 和 Nsight Compute 的核心区别是什么？
