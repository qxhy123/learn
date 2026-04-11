# 第17章：Unified Memory 与异步编程

> Unified Memory 的价值，不在于“帮你省掉所有思考”，而在于它把一部分显式数据管理交给了运行时；但你仍然需要理解数据会在什么时候迁移、为什么有时会变慢。

---

## 学习目标

学完本章，你将能够：

1. 理解 Unified Memory 的基本模型以及 `cudaMallocManaged` 的作用
2. 知道为什么“统一地址空间”不等于“没有数据迁移成本”
3. 理解 prefetch 的作用以及何时值得使用
4. 理解异步 API 在更复杂数据流中的意义
5. 建立“方便性、可维护性与性能边界”的权衡意识

---

## 17.1 为什么会有 Unified Memory？

前面我们学习的典型 CUDA 编程方式是：

1. host 分配数据
2. device 分配显存
3. 显式 `cudaMemcpy`
4. kernel 执行
5. 再显式拷回

这种方式很清晰，也有利于性能控制，但编程上会比较繁琐。

尤其在这些场景下：

- 数据结构复杂
- 指针层次多
- 原型开发阶段更重视可读性和迭代速度
- 希望减少显式拷贝样板代码

Unified Memory 就是为此提供的一种更统一的内存使用方式。

---

## 17.2 `cudaMallocManaged` 做了什么？

最常见的 Unified Memory 分配方式是：

```cpp
float* data = nullptr;
cudaMallocManaged(&data, n * sizeof(float));
```

你可以先把它理解为：

- 返回一个统一地址空间中的指针
- host 和 device 都可以使用这个地址

这和你前面见过的 `cudaMalloc` 很不同，因为 `cudaMalloc` 分配出的指针主要是 device 侧使用，而 managed memory 的目标是：

- 尽量减少显式 host/device 双份指针与手动拷贝管理

### 一个最小示例

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void add_one(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

int main() {
    int n = 1024;
    float* data = nullptr;

    cudaMallocManaged(&data, n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i);
    }

    add_one<<<(n + 255) / 256, 256>>>(data, n);
    cudaDeviceSynchronize();

    std::cout << data[0] << std::endl;

    cudaFree(data);
    return 0;
}
```

这里没有显式 `cudaMemcpy`，但不代表数据移动完全不存在。

---

## 17.3 统一地址空间不等于没有迁移成本

这是理解 Unified Memory 最重要的一点。

虽然你看到的是同一个指针 `data`，但底层仍然要解决：

- 当前这段数据应该驻留在哪
- 当 CPU 和 GPU 分别访问它时，页面如何迁移
- 访问发生时是否会触发缺页式迁移或其他运行时处理

所以更准确的理解是：

- Unified Memory 简化了编程模型
- 但没有消灭物理数据移动和访问成本

### 常见误解

错误理解：

- “既然 host 和 device 都能直接用同一个指针，那就完全没有传输了。”

更合理的理解：

- “显式拷贝代码可以消失，但数据迁移本身和其性能影响仍然存在。”

---

## 17.4 Prefetch 为什么重要？

如果 managed memory 数据在 CPU 上初始化后，接下来 GPU 要大量使用它，那么一个常见优化思路是：

- 在真正启动 kernel 前，主动把数据预取到 GPU

常见接口是：

```cpp
cudaMemPrefetchAsync(data, bytes, device_id, stream);
```

### 它的作用直觉

你可以把 prefetch 理解为：

- 不等第一次访问时再被动迁移
- 而是提前告诉运行时：这段数据接下来主要会被这个 device 使用

这通常有助于：

- 减少运行时按页迁移带来的抖动
- 让访问行为更可预测

### 同样地，prefetch 也不是魔法

如果你的访问模式本身就频繁在 CPU 和 GPU 之间来回切换，那么 Unified Memory 仍然可能表现不佳。

---

## 17.5 Unified Memory 适合什么场景？

### 适合的场景

#### 原型开发和教学

它能显著减少样板代码，让你更快聚焦在算法本身。

#### 数据结构复杂

例如：

- 多层指针结构
- 图结构
- 树结构
- 复杂对象图

在这类问题中，显式管理每一层数据拷贝会很麻烦。

#### 先求正确性和开发效率

当你还在快速迭代阶段，Unified Memory 往往能让代码更容易组织。

### 不一定理想的场景

#### 对性能控制非常敏感

如果你需要极细粒度地控制：

- 数据什么时候传
- 传到哪
- 与哪些 kernel 重叠

显式内存管理通常更透明。

#### CPU/GPU 频繁交替访问同一数据

这种来回切换可能导致频繁迁移，性能不稳定。

---

## 17.6 Unified Memory 与异步编程如何结合？

虽然 Unified Memory 简化了显式拷贝，但你仍然可以利用异步能力改善行为，例如：

- `cudaMemPrefetchAsync`
- stream 中的预取与后续 kernel 组织
- event 控制前后依赖

这说明：

- Unified Memory 并不是“告别异步编程”
- 反而在复杂工作流中，仍然可以和 stream / event 配合

### 一个思维升级

从显式拷贝模型转到 Unified Memory 后，你的优化重点会更多变成：

- 数据何时会被哪一端访问
- 是否需要预取
- 是否需要避免来回访问抖动

而不只是“哪里写一条 `cudaMemcpy`”。

---

## 17.7 一个基本 prefetch 骨架

```cpp
int device = 0;
size_t bytes = n * sizeof(float);

cudaMallocManaged(&data, bytes);

for (int i = 0; i < n; ++i) {
    data[i] = static_cast<float>(i);
}

cudaMemPrefetchAsync(data, bytes, device, stream);
my_kernel<<<blocks, threads, 0, stream>>>(data, n);
```

这个骨架表达的是：

- CPU 初始化完数据后
- 提前把 managed memory 预取到将要执行 kernel 的 GPU
- 然后再执行 kernel

这比完全依赖首次访问时的被动迁移，通常更可控。

---

## 17.8 常见误区

### 误区一：用了 Unified Memory，就完全不用关心数据移动

不对。你只是少写了显式拷贝代码，不是消除了物理迁移。

### 误区二：Unified Memory 一定更慢

也不对。在很多原型、教学或复杂数据结构场景中，它的开发收益很大；某些场景下配合 prefetch 也可以获得不错表现。

### 误区三：Unified Memory 可以替代所有显式内存管理

不对。对于一些极度关注吞吐、传输重叠和内存位置控制的场景，显式管理仍然更合适。

### 误区四：异步 API 只和 `cudaMemcpyAsync` 有关

不对。Unified Memory 的 prefetch、本章讨论的 stream 组织，也都属于异步编程的一部分。

---

## 17.9 一个实用判断框架

当你考虑是否用 Unified Memory 时，可以先问：

1. 现在更需要开发效率，还是更需要极致性能控制？
2. 数据结构是否复杂到显式拷贝会显著增加代码复杂度？
3. 数据主要由 GPU 使用，还是会在 CPU/GPU 间频繁切换？
4. 是否可以通过 prefetch 让主要访问位置更稳定？
5. 如果性能不理想，我是否愿意再退回显式管理模型？

这个框架能帮助你把 Unified Memory 放在合适的位置，而不是神化它，也不是妖魔化它。

---

## 本章小结

| 核心概念 | 要点 |
|----------|------|
| Unified Memory | 用统一地址空间简化 host/device 数据管理 |
| `cudaMallocManaged` | 常见 managed memory 分配接口 |
| 本质边界 | 统一指针不代表没有数据迁移成本 |
| prefetch | 提前把数据迁移到预期访问设备，提高可预测性 |
| 适用场景 | 原型、教学、复杂数据结构、强调开发效率的场景 |
| 工程判断 | 性能敏感场景仍需权衡是否改回显式内存管理 |

---

## CUDA实验

### 实验 1：把显式 `cudaMemcpy` 示例改写成 Unified Memory 版本

请把一个向量加法示例改写为：

- 使用 `cudaMallocManaged`
- 不再手写 `cudaMemcpy`
- 在 kernel 执行后同步并读取结果

分析：

- 代码变简单了哪些地方
- 哪些数据移动成本只是“被隐藏了”，而不是消失了

### 实验 2：给 managed memory 增加 prefetch

设计一个实验流程：

1. CPU 初始化 managed memory
2. 使用 `cudaMemPrefetchAsync` 预取到 GPU
3. 启动 kernel
4. 比较“有 prefetch”和“无 prefetch”两种版本的执行行为

先不要求严格得出性能结论，但要明确你想观察什么。

---

## 练习题

1. **基础题**：Unified Memory 和显式 `cudaMemcpy` 模型的主要区别是什么？
2. **基础题**：为什么说统一地址空间不等于没有迁移成本？
3. **实现题**：写一个使用 `cudaMallocManaged` 的最小 kernel 示例伪代码。
4. **思考题**：为什么 prefetch 能帮助 Unified Memory 更可控？
5. **思考题**：在哪些场景下，显式内存管理仍然比 Unified Memory 更合适？
