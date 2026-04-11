# 第4章：Kernel 编写与启动配置

> CUDA 编程最容易犯错的地方，不是“不会写并行”，而是把一个本来很简单的计算映射错了线程、配置错了 launch，最后结果悄悄地错掉。

---

## 学习目标

学完本章，你将能够：

1. 理解 `__global__`、`__device__`、`__host__` 这些函数修饰符的基本区别
2. 独立写出一个最小 kernel，并从 host 正确启动它
3. 理解 `<<<grid, block>>>` 启动配置的含义
4. 掌握一维 kernel 中最常见的边界处理模式
5. 知道 block 大小选择的基本经验，以及为什么它和性能相关

---

## 4.1 Kernel 到底是什么？

在 CUDA 里，kernel 指的是：

- 由 host 发起启动
- 在 GPU 上并行执行
- 通常由大量线程共同运行的函数

最常见的定义方式是：

```cpp
__global__ void saxpy(const float* x, float* y, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}
```

这里的 `saxpy` 表示常见的线性代数操作：

$$y_i = a x_i + y_i$$

它是一个非常经典的 CUDA 入门例子，因为：

- 数据并行很清晰
- 每个元素独立
- 索引映射简单
- 易于验证结果

---

## 4.2 `__global__`、`__device__`、`__host__` 分别是什么意思？

### `__global__`

表示这是一个 kernel：

- 从 host 启动
- 在 device 上执行

```cpp
__global__ void my_kernel(...) {
}
```

### `__device__`

表示这是一个只能在 device 上调用的函数，通常被 kernel 或其他 device 函数调用。

```cpp
__device__ float square(float x) {
    return x * x;
}
```

### `__host__`

表示函数在 host 上执行。普通 C/C++ 函数默认就可以理解为 host 函数。

```cpp
__host__ void prepare_data() {
}
```

初学阶段最重要的是分清：

- `__global__`：host 发起，device 执行
- `__device__`：device 内部调用
- 普通函数 / `__host__`：CPU 上执行

---

## 4.3 一个完整的 SAXPY 示例

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void saxpy(const float* x, float* y, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

int main() {
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    float a = 2.0f;

    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(n, 3.0f);

    float* d_x = nullptr;
    float* d_y = nullptr;

    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    saxpy<<<blocks, threads_per_block>>>(d_x, d_y, a, n);

    cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost);

    std::cout << "h_y[0] = " << h_y[0] << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
```

如果 `x[i] = 1`、`y[i] = 3`、`a = 2`，那么：

$$y_i = 2 \cdot 1 + 3 = 5$$

所以输出应为：

```bash
# 输出: h_y[0] = 5
```

---

## 4.4 `<<<grid, block>>>` 到底在配置什么？

CUDA kernel 的启动语法通常写成：

```cpp
kernel<<<num_blocks, threads_per_block>>>(...);
```

你可以把它理解为两件事：

1. 启动多少个 block
2. 每个 block 里放多少个线程

例如：

```cpp
saxpy<<<4096, 256>>>(...);
```

含义是：

- grid 中有 4096 个 block
- 每个 block 有 256 个线程

总线程数是：

```text
4096 × 256
```

但要注意，总线程数只是“提供出来的线程槽位”，并不意味着每个线程都一定处理合法数据。因此仍然需要边界判断。

### 2D / 3D 配置

除了整数形式，`grid` 和 `block` 也可以是 `dim3`：

```cpp
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y);

kernel<<<grid, block>>>(...);
```

这对矩阵和图像问题特别常见。

---

## 4.5 为什么 launch 配置总是和数据规模绑在一起？

因为你需要让线程总数能够覆盖数据。

最常见的一维写法是：

```cpp
int threads_per_block = 256;
int blocks = (n + threads_per_block - 1) / threads_per_block;
```

这个公式本质是在做向上取整。

### 例子

如果：

- `n = 1000`
- `threads_per_block = 256`

那么：

```cpp
blocks = (1000 + 255) / 256 = 4;
```

4 个 block 就能覆盖 1000 个元素。

这是几乎所有一维数据并行 kernel 的基础模板。

---

## 4.6 初学者最常见的 kernel 编写错误

### 4.6.1 忘记边界判断

错误示意：

```cpp
__global__ void bad_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in[idx] * 2.0f;
}
```

如果线程总数大于 `n`，这段代码就可能越界。

正确写法通常是：

```cpp
if (idx < n) {
    out[idx] = in[idx] * 2.0f;
}
```

### 4.6.2 用错 host/device 指针

例如把 host 指针传给 kernel，或者把 device 指针直接当 host 数据访问，都是初学者常见问题。

要建立很清晰的约定：

- `h_` 前缀表示 host 内存
- `d_` 前缀表示 device 内存

### 4.6.3 把 launch 配置写反

例如把 `blocks` 和 `threads_per_block` 概念混淆。

虽然语法上可能不报错，但结果会非常不同。

### 4.6.4 忘记结果验证

很多人只看程序“跑完了”，却没有检查输出。对于并行程序，这非常危险。

至少应验证：

- 一个已知位置的值
- 全量误差是否在可接受范围内

---

## 4.7 block 大小该怎么起步？

初学阶段可以先采用经验值：

- 128
- 256
- 512

其中 256 常被当作通用起点。

### 为什么不是越大越好？

因为 block 大小会影响：

- 每个 SM 同时驻留多少个 block
- 寄存器使用压力
- shared memory 资源分配
- warp 调度灵活性

### 为什么也不是越小越好？

因为 block 太小会导致：

- 并行度利用不足
- 调度开销相对更高
- 很难充分隐藏内存延迟

所以正确理解应该是：

- **block 大小是性能参数，不是数学常数**
- **先用常见值跑通，再通过 profiling 调优**

---

## 4.8 从“写出 kernel”到“写对 kernel”

初学阶段最重要的习惯不是写花哨代码，而是每次都检查这几件事：

1. 每个线程处理哪个元素？
2. 是否有边界判断？
3. launch 配置是否足够覆盖全部数据？
4. host 和 device 指针是否分清？
5. 是否验证了结果？

如果这 5 件事没有做清楚，后面谈 shared memory、streams、graphs 都会失去基础。

---

## 本章小结

| 核心概念 | 要点 |
|----------|------|
| kernel | 从 host 启动、在 device 上并行执行的函数 |
| 函数修饰符 | `__global__` 用于 kernel，`__device__` 用于 device 内部函数 |
| launch 配置 | `<<<grid, block>>>` 指定 block 数和每个 block 的线程数 |
| 常见模板 | 一维数据并行通常用 `idx = blockIdx.x * blockDim.x + threadIdx.x` |
| 边界处理 | `if (idx < n)` 是最基本的正确性保障 |
| block 大小 | 是性能调节参数，常从 256 开始尝试 |

---

## CUDA实验

### 实验 1：实现 SAXPY

请你基于本章示例实现：

$$y_i = a x_i + y_i$$

要求：

- 使用一维 grid 和一维 block
- 自己写出 `idx` 计算
- 加入边界判断
- 用 host 侧代码验证 3 个位置的输出是否正确

### 实验 2：比较不同 block 大小

把 `threads_per_block` 改成：

- 64
- 128
- 256
- 512

记录：

- 程序是否都能得到正确结果
- 哪些值更适合作为起点
- 为什么“能跑”不等于“最好”

先不要求严格性能结论，但要开始建立“launch 配置会影响效果”的意识。

---

## 练习题

1. **基础题**：`__global__` 和 `__device__` 的区别是什么？
2. **基础题**：为什么 kernel 启动时需要 `<<<grid, block>>>` 这样的配置？
3. **实现题**：写一个 kernel，使每个线程执行 `out[idx] = 3 * in[idx] - 1`。
4. **思考题**：为什么 CUDA 示例几乎总会写边界判断，即使看起来线程数量已经算好了？
5. **思考题**：为什么说 block 大小的选择既影响性能，又不应在入门阶段过度神化？
