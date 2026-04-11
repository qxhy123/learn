# 第16章：Streams、Events 与计算-传输重叠

> GPU 性能优化走到一定阶段后，瓶颈往往不再只是“某个 kernel 快不快”，而是“数据传输、kernel 执行、多个任务流之间能否真正并行推进”。

---

## 学习目标

学完本章，你将能够：

1. 理解 CUDA stream 的基本含义和默认 stream 的直觉
2. 知道为什么异步执行不等于自动重叠
3. 理解 CUDA event 在计时与依赖管理中的作用
4. 建立“拷贝与计算重叠”的 pipeline 思维
5. 知道实现 overlap 通常还需要哪些前提条件

---

## 16.1 为什么要从“单个 kernel”走向“任务流”？

在前面的章节里，我们经常把注意力放在一个 kernel 上：

- 是否正确
- 是否访存高效
- 是否用了 shared memory

但真实工程里，一个完整工作流往往包含：

1. host 准备数据
2. 数据拷到 GPU
3. 执行一个或多个 kernel
4. 把结果拷回 host
5. 再继续下一批数据

如果这些步骤完全串行，那么 GPU 和数据通路中会出现很多空闲段。

所以更高阶的问题变成：

- 能不能让不同步骤重叠？
- 能不能让下一批数据的拷贝与上一批数据的计算同时进行？

这就是 stream 和 event 登场的地方。

---

## 16.2 什么是 stream？

你可以先把 stream 理解为：

- 一条提交给 GPU 的工作队列
- 队列内任务按顺序执行
- 不同 stream 之间的工作在满足条件时可以并发或重叠

### 一个很重要的直觉

同一个 stream 内：

- 任务通常按提交顺序推进

不同 stream 之间：

- 不一定互相等待
- 有机会并行推进

这为我们表达更复杂执行关系提供了基础。

---

## 16.3 默认 stream 与显式 stream

如果你什么都不写，很多 CUDA 操作会落到默认 stream 中。

例如：

```cpp
my_kernel<<<blocks, threads>>>(...);
```

这通常意味着：

- 使用默认 stream

如果你想显式创建自己的 stream，可以写：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

my_kernel<<<blocks, threads, 0, stream>>>(...);
```

这里 launch 语法中的第四个参数就是 stream。

### 为什么显式 stream 有意义？

因为只有当你开始区分“这是哪一条任务流中的工作”时，才更容易构建：

- 多批次流水
- 传输与计算重叠
- 多阶段异步调度

---

## 16.4 什么是异步？为什么异步不等于自动变快？

很多 CUDA API 都有异步版本或异步执行特性。例如：

- kernel launch 常常具有异步性质
- `cudaMemcpyAsync`
- stream 级排队执行

但这里要特别警惕一个误解：

- **异步不等于自动重叠，更不等于自动更快。**

### 为什么？

因为想要真正重叠，通常还要满足：

- 使用合适的 stream
- 操作之间不存在强依赖
- 设备支持相关并发能力
- host / device 侧资源准备方式正确
- 数据传输路径也允许异步推进

所以“把 API 改成 Async”只是第一步，不是终点。

---

## 16.5 `cudaMemcpyAsync` 的基本角色

相比同步拷贝：

```cpp
cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
```

异步拷贝通常写作：

```cpp
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
```

这表示：

- 把拷贝操作提交到某个 stream 中
- host 不必在调用点立即阻塞到传输完全结束

### 为什么这很关键？

因为一旦拷贝也进入了 stream 语义，你就可以更自然地安排：

- 某批数据的 H2D
- 紧接着该批数据的 kernel
- 同时另一批数据的准备或回传

从而构建流水。

---

## 16.6 Event 用来做什么？

event 常见有两个核心用途：

### 用途一：计时

例如：

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
my_kernel<<<blocks, threads, 0, stream>>>(...);
cudaEventRecord(stop, stream);
```

然后可以测量某段 GPU 工作在该时间线中的耗时。

### 用途二：表示“某个时间点 / 某段工作已完成”

这使得 event 不只是计时工具，也可以作为依赖管理工具。

例如某个 stream 可以等待另一个 stream 中某个 event 完成后再继续。

### 第一层直觉

你可以把 event 理解为：

- GPU 时间线上的一个标记点
- 可用于测量，也可用于协调不同 stream 的先后关系

---

## 16.7 一个最小的双缓冲 pipeline 直觉

假设你有很多批次数据，每批都要经历：

1. H2D 拷贝
2. kernel 计算
3. D2H 拷贝

如果完全串行，时间线可能像这样：

```text
批1: H2D -> compute -> D2H
批2: H2D -> compute -> D2H
批3: H2D -> compute -> D2H
```

而 pipeline 思维会尝试把它改成：

```text
批1: H2D -> compute -> D2H
批2:      H2D -> compute -> D2H
批3:           H2D -> compute -> D2H
```

理想情况下：

- 批 2 的 H2D 可以与批 1 的 compute 重叠
- 批 3 的准备也能与前面的阶段交叠

这就是“计算-传输重叠”的核心图景。

---

## 16.8 实现 overlap 的常见前提

想真正实现比较好的 overlap，通常要留意这些条件：

### 1. 操作需要在不同 stream 中组织

如果所有操作都放在一个 stream 里，它们通常仍然按顺序排队。

### 2. 数据应分批处理

如果任务只有一整块单次大输入，流水空间有限。

### 3. host 内存准备方式要合适

很多高效异步传输场景更偏好 pinned memory。

### 4. 工作之间不能有不必要的全局同步

如果你每一步都 `cudaDeviceSynchronize()`，很多潜在重叠都会被你自己打断。

### 5. 硬件和运行时要支持相应并发能力

并不是所有设备和所有场景都能获得相同程度的重叠。

---

## 16.9 一个最小异步执行骨架

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

cudaMemcpyAsync(d_a1, h_a1, bytes, cudaMemcpyHostToDevice, stream1);
my_kernel<<<blocks, threads, 0, stream1>>>(d_a1, d_out1);

cudaMemcpyAsync(d_a2, h_a2, bytes, cudaMemcpyHostToDevice, stream2);
my_kernel<<<blocks, threads, 0, stream2>>>(d_a2, d_out2);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

这段代码不保证一定获得理想重叠，但它已经展示了：

- 不同 stream 可承载不同批次工作
- 同步可以局限在 stream 级，而不是整个 device 级

---

## 16.10 调试异步程序时为什么更要谨慎？

一旦引入 stream 和 overlap，程序正确性和时序关系会更复杂。

常见风险包括：

- 某批数据还没拷完就开始被错误使用
- 某个输出缓冲区在回传前被下一批覆盖
- 全局同步太少导致隐藏 bug
- 全局同步太多又把 overlap 全部抹掉

### 一个实用习惯

在刚搭 pipeline 时：

1. 先做一个正确的串行版本
2. 再改成分 stream 版本
3. 每加一层异步都重新验证结果
4. 最后再看 Nsight Systems 的时间线是否真的出现重叠

---

## 本章小结

| 核心概念 | 要点 |
|----------|------|
| stream | GPU 工作队列，同一 stream 内通常顺序执行 |
| 异步执行 | 不等于自动更快，重叠需要额外条件 |
| `cudaMemcpyAsync` | 让传输进入 stream 语义，便于构建 pipeline |
| event | 既可计时，也可作为时间线上的依赖标记 |
| overlap | 常见目标是让传输与计算、不同批次工作互相重叠 |
| 工程方法 | 先验证正确性，再用时间线工具确认是否真的重叠 |

---

## CUDA实验

### 实验 1：把同步拷贝改写成异步版本

给定一个流程：

1. H2D 拷贝
2. kernel 执行
3. D2H 拷贝

请把它从默认同步写法改成：

- 使用显式 stream
- 使用 `cudaMemcpyAsync`
- 使用 stream 同步收尾

并解释每一处修改的意义。

### 实验 2：画一个两批数据的重叠时间线

假设有批 1 和批 2，每批都要：

- H2D
- compute
- D2H

请画出：

1. 完全串行版本的时间线
2. 理想 overlap 版本的时间线

说明：

- 哪些阶段有机会重叠
- 为什么单 stream 往往不容易形成这种结构

---

## 练习题

1. **基础题**：stream 的基本作用是什么？
2. **基础题**：为什么说异步 API 不等于自动获得性能提升？
3. **实现题**：写一段最小伪代码，使用两个 stream 分别提交两批 kernel 工作。
4. **思考题**：为什么 `cudaDeviceSynchronize()` 在异步 pipeline 中既有调试价值，又可能破坏重叠？
5. **思考题**：为什么 Nsight Systems 比单纯计时更适合用来判断是否真的实现了 overlap？
