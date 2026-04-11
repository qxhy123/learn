# 第13章：访存合并与 Occupancy

> GPU 优化里最容易被误解的两个词，往往就是 coalescing 和 occupancy：一个关心线程怎样把内存请求组织得更顺，另一个关心硬件是否有足够多的活跃线程去掩盖延迟。

---

## 学习目标

学完本章，你将能够：

1. 理解 memory coalescing 的基本直觉
2. 知道为什么相邻线程访问相邻地址通常更有利于性能
3. 理解 occupancy 的含义以及它与吞吐的关系
4. 知道寄存器和 shared memory 使用量为什么会影响 occupancy
5. 建立“高 occupancy 不等于一定最快”的工程判断

---

## 13.1 为什么这一章重要？

到目前为止，你已经见过很多 CUDA 基础概念：

- 线程组织
- shared memory
- 同步
- 并行模式

但真正进入性能优化时，两个问题会反复出现：

1. 线程访问 global memory 的方式是否规整？
2. GPU 上是否有足够多的活跃线程去隐藏访存和执行延迟？

第一个问题常对应 **memory coalescing**，第二个问题常对应 **occupancy**。

这两个概念经常同时出现，但它们解决的不是同一件事。

---

## 13.2 什么是访存合并（coalescing）？

先建立最朴素的直觉：

- 一个 warp 中的线程如果访问一串连续、对齐良好的地址
- 硬件通常更容易把这些访问组织成更高效的内存事务

这就是所谓的 **memory coalescing**。

### 一个友好的访问模式

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
out[idx] = in[idx];
```

如果一个 warp 中线程的 `idx` 是连续的，那么它们通常也会访问连续地址。

### 一个不友好的访问模式

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
out[idx] = in[idx * stride];
```

如果 `stride` 很大，那么相邻线程会跳着访问数据，内存请求就可能更分散。

### 初学阶段先抓住一句话

- **相邻线程访问相邻数据，通常更容易获得好的 global memory 吞吐。**

---

## 13.3 为什么“跨步访问”常常更慢？

假设一个 warp 中线程访问：

```text
in[0], in[1], in[2], in[3], ...
```

这通常很规整。

但如果访问变成：

```text
in[0], in[32], in[64], in[96], ...
```

那么同一个 warp 中线程访问的位置就分散得多。

这往往会导致：

- 更多内存事务
- 更差的带宽利用
- cache 效果变差

所以很多 GPU 优化本质上都在努力让：

- 线程映射
- 数据布局
- 访问顺序

三者尽量对齐。

---

## 13.4 数据布局为什么会直接影响 coalescing？

还是以二维矩阵为例。

如果矩阵按 row-major 存储，而你的线程布局让相邻线程访问相邻列：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
val = A[row * width + col];
```

那么对同一行来说，`threadIdx.x` 连续增长通常对应连续地址。

但如果你让相邻线程访问跨行元素，例如：

```cpp
val = A[col * width + row];
```

那么访问可能变得更分散。

这也是为什么：

- 矩阵转置
- 按列访问 row-major 数据
- 不规则 gather/scatter

往往更容易遇到访存效率问题。

---

## 13.5 什么是 occupancy？

occupancy 通常可以粗略理解为：

- 一个 SM 上实际活跃线程（或 warp）数量
- 相对于该 SM 理论最大活跃线程（或 warp）数量的比例

### 它关心的核心问题是什么？

不是“单个线程快不快”，而是：

- 当一部分线程在等待内存或其他延迟时
- 是否还有足够多的其他 warp 可以被调度执行

换句话说，occupancy 的一个重要作用是：

- **帮助 GPU 隐藏延迟**

### 为什么 GPU 需要这件事？

因为 global memory 访问很慢，相对寄存器和算术执行来说延迟很高。

如果一个 SM 上只有很少活跃 warp，那么一旦这些 warp 都在等内存，执行单元就可能闲下来。

---

## 13.6 什么因素会限制 occupancy？

一个 block 能不能更多地并发驻留在某个 SM 上，不只取决于 block 大小，还取决于资源使用。

### 常见限制因素

#### 1. 每线程寄存器使用量

如果一个 kernel 每线程使用很多寄存器，那么同一个 SM 上能容纳的活跃线程数会下降。

#### 2. 每 block shared memory 使用量

如果一个 block 占用了很多 shared memory，那么同一个 SM 上可同时驻留的 block 数会减少。

#### 3. block 大小本身

过大或过小的 block 配置，都可能影响活跃 warp 数量与调度灵活性。

#### 4. 硬件本身的上限

例如：

- 每 SM 最多支持多少线程
- 最多支持多少 block
- 最多支持多少 warp

这些都是架构给定的上限。

---

## 13.7 高 occupancy 一定更快吗？

不一定。这是非常重要的判断。

### 为什么不一定？

因为 occupancy 只是“有多少活跃线程”的指标之一，它不是对性能的直接保证。

举几个典型情况：

#### 情况一：访存模式很差

即使 occupancy 很高，如果每个 warp 的 global memory 访问都非常零碎，性能仍然可能很差。

#### 情况二：算术或数据复用已经很高

有些 kernel 即使 occupancy 不是很高，也仍然可以很快，因为：

- 每个线程做了很多有效计算
- shared memory / cache 复用很好
- 不太需要靠大量 warp 去隐藏延迟

#### 情况三：为了追求 occupancy 过度压缩资源使用

例如为了让更多 block 驻留，你牺牲了：

- 数据复用
- 算法结构
- 中间缓存

最终未必更快。

### 更准确的理解

- **occupancy 是重要参考，不是唯一目标。**

---

## 13.8 一个典型权衡：shared memory vs occupancy

shared memory 常常能减少 global memory 访问，但它也会占用片上资源。

这带来一个经典权衡：

- 用更多 shared memory，可能提高数据复用
- 但同一 SM 上可驻留的 block 数量可能下降
- occupancy 因此降低

### 这时该怎么判断？

不要凭感觉下结论，应结合：

- profiling
- 实际 kernel 时间
- 内存吞吐和计算吞吐指标

有时：

- 稍低 occupancy + 更好数据复用

会优于：

- 很高 occupancy + 大量重复 global memory 访问

---

## 13.9 一个典型权衡：寄存器使用 vs 并发度

类似地，更多寄存器可能意味着：

- 更少 spill 到 local memory
- 更好的中间值保存

但也可能意味着：

- 每个 SM 上可同时活跃的线程更少

所以“寄存器越多越好”或者“越少越好”都不对。

真正要看的，是资源分配是否让这个 kernel 的整体吞吐更优。

---

## 13.10 如何开始分析一个内存受限 kernel？

如果你怀疑某个 kernel 是访存瓶颈，可以先问：

1. 相邻线程是否在访问相邻地址？
2. 是否存在大量跨步访问、随机访问或转置式访问？
3. 是否有重复 global memory 读取，能否用 shared memory 缓存？
4. 当前 block 大小和资源使用会不会导致 occupancy 太低？
5. profiling 工具显示瓶颈更偏内存，还是更偏计算？

这比一上来盲目调 block 大小更有效。

---

## 13.11 常见误区

### 误区一：只要把 occupancy 拉满，性能就会最好

不对。occupancy 高只是说明活跃线程多，不等于访存、算术和同步都高效。

### 误区二：coalescing 只是底层小细节，可以忽略

不对。对于很多 global memory 密集 kernel，它几乎是最直接的性能决定因素之一。

### 误区三：只调 block 大小就算做优化了

block 大小重要，但它只是资源与调度的一部分，访存模式、数据布局和算法结构同样关键。

### 误区四：shared memory 导致 occupancy 下降，所以一定不该用

不对。shared memory 带来的数据复用收益可能远大于 occupancy 的下降成本。

---

## 本章小结

| 核心概念 | 要点 |
|----------|------|
| coalescing | 相邻线程访问相邻地址通常更有利于 global memory 吞吐 |
| 跨步访问 | 常常导致内存请求更分散、效率更差 |
| occupancy | 活跃线程 / warp 相对理论上限的比例，主要用于隐藏延迟 |
| 限制因素 | 寄存器、shared memory、block 大小和硬件上限都会影响 occupancy |
| 性能判断 | 高 occupancy 不等于一定更快，需结合访存模式和 profiling |
| 优化思路 | 先修正访存模式，再结合资源使用调并发度 |

---

## CUDA实验

### 实验 1：比较连续访问与跨步访问

设计两个 kernel：

1. 连续访问：`out[idx] = in[idx]`
2. 跨步访问：`out[idx] = in[idx * stride]`

要求：

- 尝试不同 `stride`
- 记录运行时间变化趋势
- 思考为什么 `stride` 变大后性能可能下降

### 实验 2：观察 block 大小变化对资源和性能的影响

固定一个简单 kernel，尝试：

- 64
- 128
- 256
- 512

分析：

- 结果是否都正确
- 哪些配置更适合作为基线
- 为什么“更多线程”不必然等于“更高性能”

---

## 练习题

1. **基础题**：什么是 memory coalescing？
2. **基础题**：occupancy 的核心作用是什么？
3. **实现题**：写出一个跨步访问数组的 kernel 伪代码。
4. **思考题**：为什么高 occupancy 并不保证最优性能？
5. **思考题**：为什么 shared memory 的收益与 occupancy 的下降之间常常需要做权衡？
