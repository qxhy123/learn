# 第8章：共享内存与分块优化

> 共享内存真正强大的地方，不是“它更快”这四个字，而是它允许你把原本重复、分散、昂贵的全局访存，改造成有组织的局部复用。

---

## 学习目标

学完本章，你将能够：

1. 理解 shared memory 在 block 级优化中的核心作用
2. 知道什么是 tiling，以及它为什么能减少重复 global memory 访问
3. 看懂一个最小的 tiled 矩阵乘法骨架
4. 对 bank conflict 建立第一层直觉
5. 理解 shared memory 优化的收益与代价都来自哪里

---

## 8.1 为什么“分块”是 CUDA 优化中的高频词？

很多 GPU 优化本质上都在做一件事：

- 把大问题拆成适合 block 局部处理的小块
- 让线程协作把这小块数据搬到 shared memory
- 在 block 内重复利用这些数据

这个思想通常就叫 **tiling**，中文常说“分块”或“分块优化”。

### 为什么它能提升性能？

因为很多问题里，数据并不是只用一次。

如果同一批数据会被多个线程多次访问，那么直接每次都去 global memory 读，会非常浪费。更高效的方式通常是：

1. 从 global memory 把一小块数据装进 shared memory
2. block 内线程反复使用
3. 减少对 global memory 的重复访问次数

---

## 8.2 先从一个简单重复读取例子理解

假设一个 block 中 256 个线程，都需要读取同一段长度为 256 的输入块，并基于它做多次计算。

### 不用 shared memory 的做法

- 每个线程直接从 global memory 读取自己需要的数据
- 如果后续又要重复使用，就继续从 global memory 读

结果可能是：

- 同一数据被重复取很多次
- global memory 带宽压力大

### 用 shared memory 的做法

- 每个线程先协作把这 256 个元素搬到 shared memory
- 同步一次
- 后续计算都从 shared memory 取数

结果通常是：

- global memory 读取次数显著减少
- block 内数据复用变得便宜

这就是 shared memory 优化最朴素、也最重要的直觉。

---

## 8.3 一个最小的 shared memory tile 示例

```cpp
__global__ void scale_tile(const float* in, float* out, int n) {
    __shared__ float tile[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) {
        tile[tid] = in[idx];
    }
    __syncthreads();

    if (idx < n) {
        float x = tile[tid];
        out[idx] = x * 2.0f + 1.0f;
    }
}
```

这个例子还没有体现“复用带来的巨大收益”，但它已经展示了 shared memory 的基本模式：

1. 声明 `__shared__` 数组
2. 每个线程装载一部分数据
3. `__syncthreads()` 保证所有线程都看见完整 tile
4. 在 block 内使用这份局部缓存

---

## 8.4 为什么矩阵乘法特别适合用 tiling 理解？

以矩阵乘法为例：

$$C = A \times B$$

对于输出元素：

$$C_{ij} = \sum_k A_{ik} B_{kj}$$

如果直接从 global memory 读取，多个线程可能反复访问：

- `A` 的同一行片段
- `B` 的同一列片段

这就给 shared memory 分块提供了天然空间。

### 分块直觉

你可以把大矩阵想成由很多小 tile 组成。每个 block：

- 负责输出矩阵中的一个小 tile
- 协作加载 `A` 和 `B` 中对应的输入 tile
- 在 shared memory 中完成局部乘加累积

这样做的核心收益是：

- 同一输入 tile 可被 block 中多个线程重复利用
- 减少 global memory 重复访问

---

## 8.5 一个最小 tiled matmul 骨架

下面先看一个教学骨架，不追求所有边界情况和最佳性能，只强调结构：

```cpp
#define TILE 16

__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        if (row < M && a_col < K) {
            tile_a[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            tile_b[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

这段代码最重要的不是背下来，而是看懂它在做三件事：

1. block 决定当前负责输出矩阵的哪个 tile
2. 每轮循环加载 `A` 和 `B` 的一个输入 tile 到 shared memory
3. 在 shared memory 上做局部乘加累积

---

## 8.6 `__syncthreads()` 在这里为什么必不可少？

shared memory 是 block 内共享的，所以线程之间存在协作关系。

例如在 tiled matmul 中：

- 某些线程负责加载 `tile_a` 的一部分
- 某些线程负责加载 `tile_b` 的一部分
- 但后续每个线程都要使用整个 tile 的数据

因此在计算前必须保证：

- 所有需要的数据都已经装载完成

这就是第一处 `__syncthreads()` 的意义。

第二处 `__syncthreads()` 则是为了保证：

- 当前这一轮 tile 的计算全部完成
- shared memory 中的数据不会被下一轮装载提前覆盖

如果缺少这些同步，结果可能不稳定甚至完全错误。

---

## 8.7 什么是 bank conflict？

shared memory 虽然快，但它不是“无限并行、完全零代价”的神奇空间。

其中一个经典问题是 **bank conflict**。

### 第一层直觉

你可以先把 shared memory 想成被划分成多个 bank。理想情况下：

- 一个 warp 中的多个线程访问不同 bank
- 这些访问可以更顺畅地并行进行

但如果很多线程同时访问同一个 bank 上不同位置，就可能发生 bank conflict，导致访问被串行化或部分串行化。

### 初学阶段不必过度沉迷细节

你现在只需要记住：

- shared memory 不一定天然完美高效
- 数据布局和访问模式仍然重要
- 某些二维 tile 的行列访问方式可能触发 bank conflict

这在矩阵转置和某些 tiled 算法中特别常见。

---

## 8.8 shared memory 优化的代价是什么？

很多人刚学到 shared memory 时容易形成一个误解：

> 既然它快，那是不是所有 kernel 都先搬进去再说？

不对，因为 shared memory 的使用有成本：

1. 需要显式装载代码
2. 需要同步
3. 占用片上资源
4. 可能限制每个 SM 同时驻留的 block 数量
5. 访问方式不好还可能产生 bank conflict

因此只有在下面情况较明显时，它才通常更划算：

- 数据会被重复使用
- global memory 重复访问很多
- block 内协作自然存在

如果只是“读一次、算一次、写回去”，shared memory 不一定值得引入。

---

## 8.9 如何判断某个问题适不适合 tiling？

可以先问自己 3 个问题：

### 问题 1：同一批输入数据会被 block 内多个线程复用吗？

如果会，tiling 很可能有价值。

### 问题 2：global memory 是否存在明显重复读取？

如果大量重复读取同一小块数据，shared memory 值得考虑。

### 问题 3：数据块能自然映射到 block 吗？

比如：

- 矩阵小块
- 图像 tile
- stencil 邻域块

如果问题本身就有局部块结构，那么 tiling 往往更自然。

---

## 本章小结

| 核心概念 | 要点 |
|----------|------|
| shared memory | 适合 block 内共享和数据复用 |
| tiling | 把大问题切成 block 可处理的小块 |
| 优化本质 | 减少重复 global memory 访问 |
| tiled matmul | 是理解分块优化的经典范式 |
| `__syncthreads()` | 用来保证装载和复用阶段的正确协作 |
| bank conflict | shared memory 访问模式不佳时也会损失性能 |

---

## CUDA实验

### 实验 1：写一个最小 tile 缓存示例

实现一个 kernel：

- 每个 block 把一段输入数据搬到 shared memory
- 同步后再对数据做一次简单变换
- 写回 global memory

重点观察：

- shared memory 的声明方式
- 为什么需要同步
- `threadIdx.x` 如何自然充当 tile 内索引

### 实验 2：阅读并解释 tiled matmul 骨架

请不要急着改代码，先逐段解释本章的 `matmul_tiled`：

1. `tile_a` 和 `tile_b` 各自缓存什么
2. `for (int t = 0; ...)` 为什么存在
3. 为什么每轮都需要两次 `__syncthreads()`
4. `sum` 代表输出矩阵中的哪个元素

---

## 练习题

1. **基础题**：为什么 shared memory 常常能帮助提升性能？
2. **基础题**：tiling 的核心思想是什么？
3. **实现题**：写一个简单 kernel，把一段输入数据先搬到 shared memory 再写回输出。
4. **思考题**：为什么 shared memory 并不是“用了就一定更快”？
5. **思考题**：矩阵乘法为什么是理解 shared memory 与 tiling 的经典例子？
