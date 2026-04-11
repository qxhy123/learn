# 第6章：索引计算与数据布局

> 很多 CUDA 错误看起来像“算法问题”，本质上却只是索引算错了一位、行列理解反了，或者线程映射和数据布局彼此不匹配。

---

## 学习目标

学完本章，你将能够：

1. 熟练写出 1D、2D、3D 数据的基本索引公式
2. 理解 row-major 布局在 CUDA 中的常见写法
3. 知道二维矩阵与线性内存之间如何互相映射
4. 理解转置、邻域访问这类问题为什么更容易出索引 bug
5. 对 pitch 和对齐建立第一层直觉

---

## 6.1 为什么索引计算是 CUDA 基础中的基础？

在 CUDA 中，很多 kernel 的核心其实并不复杂，真正难的是：

- 把线程位置映射到正确的数据位置
- 保证不越界
- 保证读写的是你以为的那个元素
- 保证访问模式尽量规整

这也是为什么很多初学者会遇到这样的情况：

- 程序能跑
- 没有直接崩溃
- 结果却总有一点不对

这类问题里，索引错误的概率非常高。

---

## 6.2 一维数据：最基础的线性映射

对于一维数组，最常见写法是：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
    out[idx] = in[idx] * 2.0f;
}
```

这说明：

- 线程的全局编号 `idx`
- 直接对应数组中的线性位置 `idx`

对于向量、线性缓冲区、展平后的数据，这种映射最常见。

---

## 6.3 二维数据：从 `(row, col)` 到线性地址

很多实际问题天然是二维的：

- 图像
- 矩阵
- 网格
- stencil 邻域

假设矩阵有：

- `height` 行
- `width` 列

如果采用 row-major 布局，那么元素 `(row, col)` 的线性地址通常是：

```cpp
int idx = row * width + col;
```

### 对应的线程映射

假设使用二维 block 和二维 grid：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

if (row < height && col < width) {
    int idx = row * width + col;
    out[idx] = in[idx];
}
```

这里非常关键的一点是：

- `x` 常对应列 `col`
- `y` 常对应行 `row`

这不是硬性规定，但这是最常见、也最自然的约定。

---

## 6.4 row-major 到底意味着什么？

在 C/C++ 语境里，最常见的是 row-major 存储。

你可以把它理解为：

- 先按第 0 行连续存放所有列
- 再存第 1 行
- 再存第 2 行
- 以此类推

例如一个 `3 × 4` 矩阵：

```text
[ a00 a01 a02 a03
  a10 a11 a12 a13
  a20 a21 a22 a23 ]
```

在线性内存中通常排成：

```text
a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23
```

因此：

```cpp
idx = row * width + col;
```

并不是死记公式，而是在表达 row-major 的存储次序。

---

## 6.5 二维转置为什么特别容易出错？

矩阵转置是 CUDA 初学和优化中都非常经典的话题。

原矩阵访问：

```cpp
in[row * width + col]
```

转置后写入：

```cpp
out[col * height + row]
```

这里最容易错的地方有：

- 把 `width` 和 `height` 混淆
- 写输出地址时还沿用输入公式
- grid 覆盖范围不正确
- 行列越界判断漏掉一边

一个最小转置骨架可以写成：

```cpp
__global__ void transpose(const float* in, float* out, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        out[col * height + row] = in[row * width + col];
    }
}
```

这里可以清楚看到：

- 输入矩阵的宽是 `width`
- 输出矩阵的“宽”已经变成原来的 `height`

---

## 6.6 三维数据的基本思路

对于体数据、三维网格、某些科学计算问题，还会遇到 3D 索引。

常见线程坐标：

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

如果数据按 `(z, y, x)` 展平，线性地址可能写成：

```cpp
int idx = z * (height * width) + y * width + x;
```

初学阶段不必急着记很多三维公式，关键是掌握统一模式：

- 外层维度乘以内层总大小
- 中间维度乘以最内层大小
- 最内层维度直接加偏移

---

## 6.7 什么是 pitch？

前面我们默认二维矩阵按紧密连续方式存储，即每一行长度恰好是：

```text
width * sizeof(T)
```

但在某些情况下，GPU 为了更好的对齐和访存组织，会使用带有行间填充的二维内存布局。此时每一行实际步长不一定等于逻辑宽度乘元素大小，这个“行步长”通常就和 pitch 有关。

例如分配二维 pitched 内存时，逻辑上你仍然认为自己处理的是：

- `width` 列
- `height` 行

但真正定位某一行时，需要用“实际行跨度”而不是简单的 `width`。

### 初学阶段应该怎么理解？

你先记住：

- **普通线性二维数组**：常用 `row * width + col`
- **pitched memory**：每一行实际跨度可能更大，需要按 pitch 计算地址

在真正开始写二维高性能代码前，先把普通 row-major 索引弄透，比过早引入 pitch 更重要。

---

## 6.8 数据布局为什么会影响性能？

索引计算不仅影响正确性，也影响访存模式。

例如，假设一个 warp 的线程按连续 `col` 访问：

```cpp
in[row * width + col]
```

这通常对应更连续、更规整的内存访问。

但如果线程采用跨步访问，例如：

```cpp
in[col * height + row]
```

或者每个线程跳着访问，那么可能导致：

- 访存不连续
- cache 利用差
- memory coalescing 变差

所以“索引怎么算”不仅决定程序对不对，也决定程序快不快。

---

## 6.9 常见索引错误清单

### 6.9.1 行列反了

把：

```cpp
row * width + col
```

写成：

```cpp
col * width + row
```

很多情况下都不会立刻崩溃，但结果会悄悄出错。

### 6.9.2 宽高混淆

尤其在转置或非方阵场景下，`width` 和 `height` 一旦混用，输出就很容易错位。

### 6.9.3 只检查一维越界

例如二维 kernel 只写：

```cpp
if (col < width)
```

却忘了检查 `row < height`。

### 6.9.4 输入输出布局不一致

输入按 row-major 读取，输出却按另一种布局写入，但代码里没有同步调整公式。

---

## 本章小结

| 核心概念 | 要点 |
|----------|------|
| 一维索引 | 全局线程索引通常直接映射线性数组位置 |
| 二维索引 | 常用 `(row, col)` 与 `row * width + col` |
| row-major | 按行连续存放，是 C/C++ 中最常见布局 |
| 转置 | 输入输出的线性地址公式不同，容易混淆宽高 |
| 3D 映射 | 维度展开遵循“外层乘以内层总大小” |
| pitch | 二维内存真实行跨度可能大于逻辑宽度 |

---

## CUDA实验

### 实验 1：写一个二维复制 kernel

假设有一个 `height × width` 的矩阵存放在一维数组中。

请写一个二维 kernel，实现：

```cpp
out[row, col] = in[row, col]
```

要求：

- 使用二维 block 和二维 grid
- 正确计算 `row` 与 `col`
- 正确写出线性地址
- 做完整边界判断

### 实验 2：手写转置索引

假设输入矩阵大小为 `2 × 3`：

```text
1 2 3
4 5 6
```

请你先不要运行程序，而是手写：

- 输入数组的线性存储顺序
- 转置后 `3 × 2` 矩阵的线性存储顺序
- `out[col * height + row] = in[row * width + col]` 中每个变量的具体含义

---

## 练习题

1. **基础题**：为什么二维矩阵在一维线性内存中仍然可以用一个公式定位元素？
2. **基础题**：row-major 布局是什么意思？
3. **实现题**：写出一个二维 kernel 的伪代码，使每个线程将矩阵元素加 1。
4. **思考题**：为什么矩阵转置比二维复制更容易写错？
5. **思考题**：为什么说数据布局不只是正确性问题，也会影响性能？
