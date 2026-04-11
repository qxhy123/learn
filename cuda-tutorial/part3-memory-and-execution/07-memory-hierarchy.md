# 第7章：CUDA 内存层次

> 写 CUDA 程序时，真正决定性能上限的，往往不是“做了多少运算”，而是“数据从哪里来、要走多远、会被谁复用”。

---

## 学习目标

学完本章，你将能够：

1. 理解 CUDA 中 register、local、shared、global、constant memory 的基本区别
2. 建立“不同内存层次延迟和容量差异很大”的直觉
3. 知道线程私有数据、block 共享数据、全局输入输出分别通常放在哪里
4. 理解为什么访存模式会成为 GPU 程序性能瓶颈
5. 对寄存器溢出、local memory、缓存行为建立第一层工程认识

---

## 7.1 为什么 CUDA 要讲“内存层次”？

在 CPU 入门里，很多时候我们先关心算法，再逐步深入缓存和内存。

但在 CUDA 中，内存层次几乎从一开始就是核心问题，因为：

- GPU 有大量线程同时访问数据
- 不同层次内存的延迟和带宽差异明显
- 许多 kernel 的瓶颈不在算术，而在访存
- 正确的数据放置方式会直接影响吞吐

因此你需要的不只是“会分配显存”，还要知道：

- 哪类数据适合线程私有
- 哪类数据适合 block 内共享
- 哪类数据只能放在全局内存
- 哪类只读常量可以用广播式访问

---

## 7.2 一张总览图先建立直觉

你可以先把 CUDA 常见内存层次粗略理解为：

```text
线程私有
├── register
└── local memory（逻辑上私有，物理上通常不在寄存器里）

block 共享
└── shared memory

grid / 全局可见
├── global memory
└── constant memory（只读、小容量、适合广播）
```

需要先明确一点：

- 这不是完整硬件细节图
- 但足够支持初学阶段的大多数判断

---

## 7.3 Register：最快，但最有限

### 什么是 register？

寄存器通常用于存放：

- 当前线程正在使用的局部变量
- 中间计算结果
- 短生命周期的标量

例如：

```cpp
__global__ void add_bias(const float* in, float* out, float bias, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        float y = x + bias;
        out[idx] = y;
    }
}
```

这里的 `x`、`y` 很可能被编译器放进寄存器。

### 为什么说它最快？

因为寄存器离执行单元最近，线程访问自己的寄存器代价最低。

但它有两个重要限制：

1. **容量有限**
2. **线程私有，不能共享**

### 寄存器用太多会怎样？

如果一个 kernel 每线程使用太多寄存器，可能会导致：

- 每个 SM 上可同时驻留的线程 / block 数下降
- occupancy 降低
- 某些变量被迫溢出到 local memory

这就是常说的 **register pressure**。

---

## 7.4 Local Memory：名字容易误导

### 它为什么叫 local？

因为从编程模型上看，它是“线程私有”的。

但要特别注意：

- **local memory 并不等于“很近、很快”**

它往往用于承载：

- 无法放进寄存器的大数组
- 编译器溢出的局部变量
- 某些按索引访问、不适合放寄存器的局部数据

### 为什么初学者容易误解？

因为“local”这个词听起来像“局部缓存”，但在 CUDA 语境里，它更多是逻辑上的线程私有地址空间，不等于高性能存储层。

因此当你看到：

- 大的局部数组
- 复杂函数导致寄存器压力很高

就要警惕 local memory 访问可能拖慢性能。

---

## 7.5 Shared Memory：block 内协作的核心工具

### 它的作用是什么？

shared memory 是同一个 block 内线程共享的一块高速片上存储。

它最常见的用途有：

- 缓存会被 block 内多个线程重复使用的数据
- 做 block 内协作计算
- 降低对 global memory 的重复访问
- 为 tiled 算法提供数据暂存区

### 基本写法

```cpp
__global__ void copy_tile(const float* in, float* out, int n) {
    __shared__ float tile[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) {
        tile[tid] = in[idx];
    }
    __syncthreads();

    if (idx < n) {
        out[idx] = tile[tid];
    }
}
```

这里 `tile` 对 block 内所有线程都可见。

### shared memory 的关键限制

- 只在同一个 block 内共享
- 容量远小于 global memory
- 需要你显式管理装载和同步

这意味着：

- 用得好，可以显著加速
- 用得不好，也可能引入同步开销和 bank conflict

---

## 7.6 Global Memory：容量大，但访问代价高

全局内存通常就是你通过 `cudaMalloc` 分配出来的 device 显存。

例如：

```cpp
float* d_x = nullptr;
cudaMalloc(&d_x, bytes);
```

它的特点是：

- 容量大
- grid 中不同 block 的线程都可以访问
- 是大多数输入输出张量/数组的默认落点
- 访问延迟和带宽代价相对更高

### 为什么它仍然最重要？

因为：

- 你的大部分真实数据通常放不进 shared memory
- 输入、输出、中间结果最终大多要落在 global memory
- 很多优化的本质，就是减少或重排对 global memory 的访问

后面会反复出现的关键词：

- coalescing
- data reuse
- tiling
- cache

本质上都和如何更高效地使用 global memory 有关。

---

## 7.7 Constant Memory：适合只读且被大量线程共同读取的小数据

constant memory 的典型适用场景是：

- 数据量小
- 只读
- 很多线程会读取同一位置

比如：

- 小型卷积核系数
- 固定参数表
- 某些广播式常量

你可以先建立这样的直觉：

- 如果许多线程经常读取同一个只读常量，constant memory 可能比普通 global memory 更合适

但它不适合：

- 大规模数据集
- 高频随机写入
- 大量不规则索引访问

---

## 7.8 缓存与“为什么同样是 global memory，效果也会不同”

初学阶段不需要把所有缓存层次细节背下来，但需要知道：

- 某些 global memory 访问可能会受缓存帮助
- 访问是否连续、是否有空间局部性，会影响实际效果
- 相同的数据量，不同访问模式，速度可能差很多

例如：

- 连续访问相邻元素
- 随机跳跃访问元素

这两者在实际执行时通常不会有相同表现。

所以当你看到“global memory 很慢”这句话时，不要把它理解成一个绝对口号，而应理解为：

- global memory 往往比寄存器和 shared memory 更贵
- 但其实际成本仍然取决于访问模式是否规整

---

## 7.9 常见的“数据该放哪”判断思路

### 线程私有的少量中间变量

通常优先让编译器放到寄存器。

### block 内重复使用的小块数据

通常考虑 shared memory。

### 大规模输入 / 输出数组

通常落在 global memory。

### 很小、只读、广播式访问的数据

可以考虑 constant memory。

### 很大的线程私有局部数组

要警惕它们可能落入 local memory，进而拖慢性能。

---

## 7.10 一个简单例子：为什么 shared memory 能减少 global 读取？

假设一个 block 中多个线程都需要访问同一小段邻域数据。

如果直接从 global memory 读，可能出现：

- 同一块数据被重复读取多次

如果先由 block 协作把这段数据搬到 shared memory：

- 每个元素只需从 global 读一次或少量次
- 后续多次复用都在 shared memory 内完成

这就是后面 tiled matrix multiplication、stencil、卷积优化的核心直觉。

---

## 7.11 常见误区

### 误区一：shared memory 一定比 global memory 快，所以能用就全用

不对。shared memory 需要：

- 显式装载
- 同步
- 占用片上资源

如果没有数据复用，只是简单搬进去再搬出来，未必划算。

### 误区二：local memory 很“本地”，所以一定快

不对。它只是逻辑上线程私有，不代表物理上接近执行单元。

### 误区三：性能优化就是尽量把所有变量塞进寄存器

不对。寄存器过多会影响 occupancy，甚至引发 spill。

### 误区四：global memory 慢，所以所有瓶颈都靠算术优化解决

很多 CUDA 程序首先是**内存受限**，不是算术受限。

---

## 本章小结

| 核心概念 | 要点 |
|----------|------|
| register | 线程私有、最快、容量有限 |
| local memory | 线程私有的逻辑空间，但不应被误解为高速存储 |
| shared memory | block 内共享，适合数据复用与协作 |
| global memory | 容量大、全局可见，但访问代价较高 |
| constant memory | 适合小型只读广播数据 |
| 优化思路 | 减少昂贵访存、提高数据复用、匹配合适内存层次 |

---

## CUDA实验

### 实验 1：判断数据应该放在哪一层

请对下面几类数据分别判断更适合哪一层内存，并说明原因：

1. 每个线程自己的临时标量
2. block 内所有线程都会重复访问的一小块矩阵 tile
3. 一个长度为一千万的输入向量
4. 一个小型只读卷积核系数表
5. 每个线程私有、长度很大的局部数组

### 实验 2：观察“重复全局读取”与“共享复用”的区别

设想一个 block 中每个线程都需要读取某段相邻数据 4 次。

请分别画出两种方案：

- 方案 A：每次都直接从 global memory 读取
- 方案 B：先搬到 shared memory，再重复使用

分析：

- 哪种方案可能更省 global memory 访问
- 为何第二种方案需要同步

---

## 练习题

1. **基础题**：register、shared memory、global memory 三者的基本区别是什么？
2. **基础题**：为什么说 local memory 这个名字容易让初学者误解？
3. **实现题**：写一个最小示例，使用 `__shared__` 数组在 block 内暂存一份输入数据。
4. **思考题**：为什么 shared memory 不是“用了就一定快”？
5. **思考题**：为什么很多 CUDA kernel 的瓶颈并不在算术，而在内存访问？
