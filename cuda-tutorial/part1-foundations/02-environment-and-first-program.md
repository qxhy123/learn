# 第2章：环境配置与第一个 CUDA 程序

> 在 CUDA 学习的最开始，环境问题不是“杂事”，而是决定你能否建立稳定实验闭环的第一道门槛。

---

## 学习目标

学完本章，你将能够：

1. 理解驱动、CUDA Toolkit、`nvcc`、GPU 硬件之间的基本关系
2. 知道在 Linux / Windows / 远程服务器上分别应关注什么环境问题
3. 使用 `nvidia-smi`、`nvcc --version` 等命令检查 CUDA 开发环境
4. 看懂一个最小的 CUDA C++ 程序由哪些部分组成
5. 编译并理解一个向量加法示例的完整执行流程

---

## 2.1 驱动、Toolkit、编译器分别是什么？

很多初学者第一次装 CUDA 时，会把下面几件事混在一起：

- 显卡驱动
- CUDA Toolkit
- `nvcc`
- C++ 编译器
- 实际的 NVIDIA GPU

它们分别扮演不同角色。

### NVIDIA 驱动

驱动负责让操作系统能够识别和管理 GPU，并为 CUDA 运行时提供与硬件交互的基础能力。

如果没有正确安装驱动，通常会出现这些现象：

- `nvidia-smi` 无法运行
- 系统看不到 NVIDIA GPU
- CUDA 程序启动失败

### CUDA Toolkit

CUDA Toolkit 是开发工具包，通常包含：

- `nvcc` 编译器驱动
- 头文件，如 `cuda_runtime.h`
- 运行时库
- 一些示例和工具
- profiling / debugging 工具

简单说：

- **驱动**负责“让 GPU 能工作”
- **Toolkit**负责“让你能开发 CUDA 程序”

### `nvcc` 是什么？

`nvcc` 不是“真正包打天下的唯一编译器”，更准确地说，它是 CUDA 的编译驱动程序。

它会处理：

- 哪些代码是 host 侧 C/C++ 代码
- 哪些代码是 device 侧 CUDA 代码
- 如何调用底层主机编译器
- 如何把 device 代码编译为目标 GPU 可执行形式

### 主机侧 C++ 编译器

CUDA C++ 程序并不只有 GPU 代码，也有很多 CPU 侧代码。比如：

- 分配 host 内存
- 调用 `cudaMemcpy`
- 启动 kernel
- 检查结果
- 打印输出

因此你还需要一个主机侧编译器，例如：

- Linux: `g++`
- Windows: MSVC
- 某些环境也可用 `clang`

---

## 2.2 平台选择与现实建议

### Linux：最常见、最稳定

如果你希望长期学习 CUDA，Linux 往往是最自然的环境，因为：

- 文档和示例多数优先覆盖 Linux
- 驱动、编译链和性能工具更常见
- 云 GPU 与服务器环境通常也是 Linux

### Windows：可用，但更要注意工具链匹配

Windows 也可以进行 CUDA 开发，但你需要额外留意：

- Visual Studio 版本是否受支持
- 环境变量是否正确
- 命令行和 IDE 的配置是否一致

### macOS：近年的设备不适合本地 CUDA 开发

如果你使用近年的 Mac，尤其是 Apple Silicon 设备，那么需要明确一点：

- 它们不能作为本地 CUDA 开发平台

这不影响你阅读教程，但实际实验应放在：

- 远程 Linux 服务器
- 云 GPU 环境
- Windows / Linux 工作站

---

## 2.3 用哪些命令检查环境？

进入 CUDA 学习前，至少应熟悉下面三个命令。

### 2.3.1 查看 GPU 与驱动信息

```bash
nvidia-smi
```

你通常会看到：

- GPU 型号
- 驱动版本
- 显存使用情况
- 当前运行中的 GPU 进程

如果这个命令都失败了，那么先不要急着写 CUDA 代码，应先解决驱动或机器环境问题。

### 2.3.2 查看 CUDA 编译器版本

```bash
nvcc --version
```

这个命令主要用于确认：

- `nvcc` 是否已经安装
- 当前 Toolkit 版本是多少

示意输出：

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 12.x, V12.x.x
```

### 2.3.3 查看主机 C++ 编译器

```bash
g++ --version
```

或在 Windows 上检查 MSVC 工具链。

这一步的意义是确认：

- 你的主机编译器存在
- 版本与 CUDA 工具链能正常配合

---

## 2.4 第一个 CUDA 程序由哪些部分组成？

一个最小 CUDA 程序通常包含下面几个阶段：

1. 在 host 上准备输入数据
2. 在 device 上分配显存
3. 把输入数据从 host 拷贝到 device
4. 启动 kernel
5. 把结果从 device 拷回 host
6. 在 host 上验证结果
7. 释放资源

这个结构非常重要，因为后面大量 CUDA 程序都只是对这个骨架做扩展。

### 最小向量加法示例

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    std::cout << "h_c[0] = " << h_c[0] << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
```

---

## 2.5 如何编译并运行这个程序？

假设文件名为 `vector_add.cu`，最基本的编译命令通常是：

```bash
nvcc vector_add.cu -o vector_add
```

运行：

```bash
./vector_add
# 输出: h_c[0] = 3
```

### 为什么文件扩展名是 `.cu`？

因为其中既包含普通 C++ 代码，也包含 CUDA 扩展语法，例如：

- `__global__`
- `<<<blocks, threads>>>`
- CUDA Runtime API

### 为什么这里没有立即看到错误？

这是 CUDA 新手常见误区之一。因为很多 CUDA API 和 kernel 执行都带有异步特征，所以：

- 程序“看起来跑了”
- 不代表就一定完全正确

在后面的调试章节里，你会学到更严谨的错误检查方式。本章先建立最小执行闭环，不急着把所有工程细节一次讲完。

---

## 2.6 逐行理解这个最小示例

### 2.6.1 Kernel 定义

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

这里最关键的是两点：

1. `__global__` 表示这是一个可以从 host 启动、在 device 上执行的 kernel
2. 每个线程通过自己的全局索引 `idx` 处理一个元素

### 2.6.2 Host 数据准备

```cpp
std::vector<float> h_a(n, 1.0f);
std::vector<float> h_b(n, 2.0f);
std::vector<float> h_c(n, 0.0f);
```

这里的 `h_` 前缀表示这些数据位于 host 内存。

### 2.6.3 Device 内存分配

```cpp
cudaMalloc(&d_a, bytes);
cudaMalloc(&d_b, bytes);
cudaMalloc(&d_c, bytes);
```

这里的 `d_` 前缀表示这些指针指向 device 内存。

### 2.6.4 数据拷贝

```cpp
cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
```

这一步把 CPU 侧数据拷到 GPU。

### 2.6.5 Kernel 启动配置

```cpp
int threads_per_block = 256;
int blocks = (n + threads_per_block - 1) / threads_per_block;
```

这个写法的含义是：

- 每个 block 放 256 个线程
- block 数量足够覆盖全部 `n` 个元素

### 2.6.6 Kernel 启动

```cpp
vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
```

这里的三尖括号语法是 CUDA 中最具辨识度的一部分。你可以先把它理解为：

- 按给定的 grid / block 规模
- 在 GPU 上启动这段并行函数

---

## 2.7 初学者最常见的环境和入门问题

### 2.7.1 `nvidia-smi` 能跑，但 `nvcc` 找不到

这通常意味着：

- 驱动装好了
- 但 Toolkit 没装好，或者 `PATH` 没配置好

### 2.7.2 `nvcc` 存在，但编译失败

常见原因包括：

- 主机编译器版本不兼容
- 头文件或库路径问题
- Windows 下 Visual Studio 工具链未正确配置

### 2.7.3 程序能编译，但运行时报错

常见原因包括：

- GPU 不支持当前目标架构
- 驱动与 Toolkit 版本组合不合适
- 代码存在越界、空指针、错误拷贝方向等问题

### 2.7.4 程序运行了，但结果不对

这时候优先怀疑：

- 索引计算错误
- 少了边界判断
- kernel 结果没有正确拷回
- host 侧验证逻辑写错

---

## 本章小结

| 核心概念 | 要点 |
|----------|------|
| 驱动 vs Toolkit | 驱动让 GPU 能被系统管理；Toolkit 让你能开发 CUDA 程序 |
| `nvcc` | CUDA 的编译驱动，负责组织 host/device 编译流程 |
| 最小程序骨架 | 分配、拷贝、launch、回传、验证、释放 |
| 环境检查命令 | `nvidia-smi`、`nvcc --version`、主机编译器版本检查 |
| 入门目标 | 先建立可编译、可运行、可验证的最小闭环 |

---

## CUDA实验

### 实验 1：检查你的环境

在你的机器或远程服务器上运行以下命令：

```bash
nvidia-smi
nvcc --version
g++ --version
```

请记录：

- GPU 型号
- 驱动版本
- CUDA Toolkit 版本
- 主机编译器版本

思考：

- 哪些命令说明“GPU 可用”？
- 哪些命令说明“开发工具链可用”？

### 实验 2：手动拆解向量加法程序

请把本章的向量加法程序按下面 7 个步骤重新标注：

1. host 数据准备
2. device 内存分配
3. host 到 device 拷贝
4. kernel 启动参数计算
5. kernel 启动
6. device 到 host 拷贝
7. 资源释放

要求你不仅能运行，还能解释每一步的职责。

---

## 练习题

1. **基础题**：驱动、CUDA Toolkit、`nvcc` 三者分别负责什么？
2. **基础题**：为什么 CUDA 程序通常既需要 GPU，也需要主机侧 C++ 编译器？
3. **实现题**：把向量加法示例中的输入改成 `h_a[i] = i`、`h_b[i] = 2 * i`，并预测 `h_c[10]` 的结果。
4. **思考题**：为什么说 `nvidia-smi` 能运行，不代表你已经具备了完整的 CUDA 开发环境？
5. **思考题**：为什么一个 CUDA 程序通常必须显式进行 host 和 device 间的数据传输？
