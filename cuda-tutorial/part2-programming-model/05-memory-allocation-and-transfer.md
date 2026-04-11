# 第5章：内存分配与数据传输

> 对 CPU 程序来说，“数据在哪里”常常不是第一反应；而对 CUDA 程序来说，这往往正是性能和正确性的起点。

---

## 学习目标

学完本章，你将能够：

1. 理解 host 内存与 device 内存的基本区别
2. 正确使用 `cudaMalloc`、`cudaFree`、`cudaMemcpy`
3. 理解常见拷贝方向 `HostToDevice`、`DeviceToHost`、`DeviceToDevice`
4. 知道为什么数据传输可能成为 CUDA 程序的性能瓶颈
5. 对 pinned memory 建立第一层工程直觉

---

## 5.1 为什么 CUDA 需要显式内存管理？

在很多基础 CUDA 程序里，CPU 内存和 GPU 内存默认是分开的。

这意味着：

- 你在 host 上创建的数组，GPU 不能直接当作自己的全局内存来用
- 你在 device 上分配的显存，CPU 也不能像普通数组一样直接访问

因此一个典型流程是：

1. host 分配和初始化输入数据
2. device 分配对应显存
3. 把输入从 host 拷到 device
4. GPU 执行计算
5. 把结果从 device 拷回 host

这也是为什么很多 CUDA 程序比纯 CPU 程序多出了一层显式数据管理。

---

## 5.2 `cudaMalloc` 和 `cudaFree`

### 基本分配方式

```cpp
float* d_x = nullptr;
size_t bytes = n * sizeof(float);
cudaMalloc(&d_x, bytes);
```

你可以把 `cudaMalloc` 理解为“在 device 全局内存上分配一段空间”。

### 释放

```cpp
cudaFree(d_x);
```

和普通堆内存一样，分配后最终要释放，否则会造成显存泄漏。

### 常见习惯

```cpp
float* d_x = nullptr;
float* d_y = nullptr;
float* d_out = nullptr;
```

使用 `d_` 前缀有助于你在阅读和调试时迅速区分：

- 哪些指针属于 device
- 哪些数据不能在 host 上直接解引用

---

## 5.3 `cudaMemcpy` 在拷什么？

最常用的 CUDA 数据传输接口是：

```cpp
cudaMemcpy(dst, src, bytes, kind);
```

四个参数分别表示：

1. 目标地址
2. 源地址
3. 拷贝字节数
4. 拷贝方向

### 常见拷贝方向

#### Host 到 Device

```cpp
cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);
```

这表示把 CPU 内存中的数据复制到 GPU 显存。

#### Device 到 Host

```cpp
cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
```

这表示把 GPU 的计算结果复制回 CPU。

#### Device 到 Device

```cpp
cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice);
```

这表示在 GPU 内部进行显存到显存拷贝。

---

## 5.4 一个完整的数据搬运骨架

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main() {
    int n = 1024;
    size_t bytes = n * sizeof(float);

    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(n, 0.0f);

    float* d_x = nullptr;
    float* d_y = nullptr;

    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, d_x, bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost);

    std::cout << "h_y[0] = " << h_y[0] << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
```

这个程序即使没有 kernel，也能帮助你熟悉：

- host 数据准备
- device 分配
- 不同方向的数据搬运
- 回传验证

---

## 5.5 为什么数据传输常常是性能瓶颈？

GPU 的计算吞吐很高，但 host 和 device 之间的数据传输通常没有那么快。

因此，如果程序流程变成这样：

1. 拷过去
2. 算一点点
3. 马上拷回来
4. 再拷过去
5. 再算一点点

那么传输开销就可能吞掉大部分收益。

### 一个重要原则

如果要让 GPU 真正发挥价值，通常希望：

- 一次把数据搬上去后做更多计算
- 尽量减少频繁往返
- 让多个 kernel 在 GPU 上连续处理同一批数据

这也是后面为什么会讨论：

- kernel 融合
- 异步拷贝
- stream 重叠
- Unified Memory

---

## 5.6 Host 普通内存与 Pinned Memory

### 什么是 Pinned Memory？

Pinned memory 常译为“页锁定内存”或“固定页内存”。

它是 host 侧的一种特殊内存，通常通过：

```cpp
float* h_x = nullptr;
cudaMallocHost(&h_x, bytes);
```

来分配。

释放则使用：

```cpp
cudaFreeHost(h_x);
```

### 为什么它重要？

因为很多情况下，host 和 device 的高速传输更偏好 pinned memory。

你可以先建立这样的直觉：

- 普通 host 内存：使用方便，通用性强
- pinned memory：更适合高性能传输路径，但资源更“重”

### 初学阶段该怎么用？

初学阶段不需要一上来就把所有数据都改成 pinned memory。更合理的顺序是：

1. 先用普通 host 内存建立正确的数据流
2. 明白 host/device 拷贝的基本路径
3. 在需要优化传输时，再引入 pinned memory

---

## 5.7 常见错误：拷贝方向、字节数、指针混淆

### 5.7.1 拷贝方向写反

例如：

```cpp
cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyDeviceToHost);
```

这在语义上就是错误的。因为：

- 目标是 device
- 源是 host
- 却写成了 `DeviceToHost`

### 5.7.2 字节数算错

例如有人写：

```cpp
cudaMemcpy(d_x, h_x.data(), n, cudaMemcpyHostToDevice);
```

如果 `n` 表示元素个数，而不是字节数，就会导致拷贝不完整。

正确做法通常是：

```cpp
size_t bytes = n * sizeof(float);
```

### 5.7.3 把 device 指针当 host 指针访问

例如：

```cpp
std::cout << d_x[0] << std::endl;
```

这不是正确的 host 侧使用方式，因为 `d_x` 指向的是 device 内存。

正确思路通常是：

- 先拷回 host
- 再在 host 上读取和打印

---

## 5.8 正确性优先的最小检查习惯

每次写完带 `cudaMemcpy` 的程序，建议都检查下面几件事：

1. `bytes` 是否真的是字节数
2. 源指针和目标指针是否都在正确内存空间
3. 拷贝方向是否正确
4. 回传后是否验证了数据
5. 分配后的资源是否都被释放

这类检查看起来基础，但能提前避免大量“程序能跑却结果不对”的问题。

---

## 本章小结

| 核心概念 | 要点 |
|----------|------|
| host vs device | 两者默认是分离的内存空间 |
| `cudaMalloc` / `cudaFree` | 用于分配和释放 device 显存 |
| `cudaMemcpy` | 用于在 host/device 或 device/device 之间搬运数据 |
| 性能瓶颈 | 频繁 host-device 往返会严重影响整体性能 |
| pinned memory | 更适合高性能传输，但不必在入门阶段滥用 |
| 常见错误 | 拷贝方向写反、字节数错误、混淆 host/device 指针 |

---

## CUDA实验

### 实验 1：只做数据搬运，不写 kernel

请写一个程序完成以下步骤：

1. 在 host 上创建长度为 `n` 的数组并初始化
2. 用 `cudaMalloc` 在 device 上分配空间
3. 执行一次 `HostToDevice` 拷贝
4. 再执行一次 `DeviceToDevice` 拷贝
5. 最后执行一次 `DeviceToHost` 拷贝并验证结果

目标：先熟悉数据流，不引入计算逻辑。

### 实验 2：比较“少算多拷”与“多算少拷”的思维差异

假设有两种方案：

- 方案 A：每处理一步都把数据拷回 CPU
- 方案 B：把数据一次放到 GPU，连续做 5 个 kernel，再统一拷回

请你从性能直觉上分析：

- 哪种更可能更快
- 为什么
- 在什么情况下这种判断可能不成立

---

## 练习题

1. **基础题**：`cudaMalloc` 和普通 `new` / `malloc` 的核心区别是什么？
2. **基础题**：`cudaMemcpyHostToDevice` 和 `cudaMemcpyDeviceToHost` 各自表示什么？
3. **实现题**：写出一段最小代码，把一个 `float` 数组从 host 拷到 device 再拷回 host。
4. **思考题**：为什么频繁的数据往返会让 GPU 加速效果变差？
5. **思考题**：为什么 pinned memory 更适合高性能传输，却不应在入门时到处乱用？
