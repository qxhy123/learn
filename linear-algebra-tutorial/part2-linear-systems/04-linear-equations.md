# 第4章：线性方程组

> 线性方程组是线性代数的核心问题——几乎所有重要理论都可以追溯到"方程组是否有解"这一根本追问。

---

## 学习目标

完成本章学习后，你将能够：

- 理解线性方程组的矩阵表示，用 $A\mathbf{x} = \mathbf{b}$ 统一描述方程组
- 掌握增广矩阵的概念，并能熟练写出方程组对应的增广矩阵
- 判断方程组解的存在性与唯一性，区分无解、唯一解、无穷多解三种情形
- 理解齐次线性方程组的结构，区分平凡解与非平凡解，掌握解空间的概念

---

## 4.1 线性方程组的概念

### 线性方程

一个关于未知量 $x_1, x_2, \ldots, x_n$ 的**线性方程**具有如下形式：

$$a_1 x_1 + a_2 x_2 + \cdots + a_n x_n = b$$

其中 $a_1, a_2, \ldots, a_n$ 称为**系数**，$b$ 称为**常数项**（右端项）。

"线性"的含义是：方程中每个未知量都以一次幂出现，不存在 $x_i^2$、$x_i x_j$、$\sin x_i$ 等非线性项。

**线性方程的例子：**

$$2x_1 - 3x_2 + x_3 = 7$$

**不是线性方程的例子：**

$$x_1^2 + x_2 = 5 \quad \text{（含二次项）}$$
$$x_1 x_2 = 3 \quad \text{（含乘积项）}$$

### 线性方程组

由 $m$ 个线性方程、$n$ 个未知量构成的**线性方程组**（简称方程组）如下：

$$\begin{cases} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\ \quad \vdots \\ a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m \end{cases}$$

其中 $a_{ij}$ 表示第 $i$ 个方程中 $x_j$ 的系数，$b_i$ 是第 $i$ 个方程的常数项。

**具体示例：** 3个方程2个未知量的方程组

$$\begin{cases} x_1 + 2x_2 = 5 \\ 2x_1 - x_2 = 0 \\ x_1 + x_2 = 3 \end{cases}$$

### 解的概念

若一组数 $(s_1, s_2, \ldots, s_n)$ 代入方程组后使**每个方程都成立**，则称其为方程组的一个**解**。

方程组的所有解构成的集合称为**解集**。

**对上面的示例验证：** 将 $(x_1, x_2) = (1, 2)$ 代入：
- $1 + 2 \cdot 2 = 5$ $\checkmark$
- $2 \cdot 1 - 2 = 0$ $\checkmark$
- $1 + 2 = 3$ $\checkmark$

所以 $(1, 2)$ 是该方程组的一个解。

---

## 4.2 矩阵表示

手写展开式繁琐且容易出错。矩阵语言能将方程组压缩为一行，并开辟出强大的理论工具。

### 系数矩阵

将方程组中所有系数按原位置排列，得到 $m \times n$ 的**系数矩阵** $A$：

$$A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}$$

### 增广矩阵

将系数矩阵 $A$ 与常数列向量 $\mathbf{b}$ 并排，得到 $m \times (n+1)$ 的**增广矩阵**，记为 $[A \mid \mathbf{b}]$：

$$[A \mid \mathbf{b}] = \left(\begin{array}{cccc|c} a_{11} & a_{12} & \cdots & a_{1n} & b_1 \\ a_{21} & a_{22} & \cdots & a_{2n} & b_2 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} & b_m \end{array}\right)$$

竖线将系数部分与常数部分分隔开，使增广矩阵的结构一目了然。

**示例：** 对方程组

$$\begin{cases} x_1 + 2x_2 = 5 \\ 2x_1 - x_2 = 0 \\ x_1 + x_2 = 3 \end{cases}$$

系数矩阵与增广矩阵分别为：

$$A = \begin{pmatrix} 1 & 2 \\ 2 & -1 \\ 1 & 1 \end{pmatrix}, \quad [A \mid \mathbf{b}] = \left(\begin{array}{cc|c} 1 & 2 & 5 \\ 2 & -1 & 0 \\ 1 & 1 & 3 \end{array}\right)$$

### 矩阵形式 $A\mathbf{x} = \mathbf{b}$

将未知量排成列向量 $\mathbf{x} = (x_1, x_2, \ldots, x_n)^T$，常数项排成列向量 $\mathbf{b} = (b_1, b_2, \ldots, b_m)^T$，则整个方程组等价于一个矩阵方程：

$$A\mathbf{x} = \mathbf{b}$$

展开验证：$A\mathbf{x}$ 的第 $i$ 行等于 $\sum_{j=1}^n a_{ij} x_j$，恰好就是原方程组第 $i$ 个方程的左侧。

**直观理解：** $A\mathbf{x} = \mathbf{b}$ 可以从**列的视角**来看——若记 $A$ 的第 $j$ 列为 $\mathbf{a}_j$，则

$$A\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n = \mathbf{b}$$

求解方程组等价于：**能否将 $\mathbf{b}$ 表示为 $A$ 各列的线性组合？** 未知量 $x_j$ 即为对应的组合系数。

> **这个视角极为重要**，它将"求解方程组"与"线性组合/列空间"直接联系起来，是后续章节（行阶梯形、列空间）的基础。

---

## 4.3 解的分类

线性方程组 $A\mathbf{x} = \mathbf{b}$ 的解集只有三种可能：无解、唯一解、无穷多解。没有"恰好两个解"的情况——这是线性结构的基本性质。

### 唯一解

方程组**恰好有一个解**。

**示例：**

$$\begin{cases} 2x_1 + x_2 = 5 \\ x_1 - x_2 = 1 \end{cases}$$

由第二个方程 $x_1 = x_2 + 1$，代入第一个方程：$2(x_2+1) + x_2 = 5$，解得 $x_2 = 1$，进而 $x_1 = 2$。

解唯一：$(x_1, x_2) = (2, 1)$。

### 无解

方程组**没有任何解**，此时称方程组**不相容**（inconsistent）。

**示例：**

$$\begin{cases} x_1 + x_2 = 2 \\ x_1 + x_2 = 5 \end{cases}$$

两个方程相矛盾（同样的左侧不可能同时等于 2 和 5），故无解。

在增广矩阵中，无解的特征是出现形如

$$\left(\begin{array}{cc|c} \cdots & \cdots & \cdots \\ 0 & 0 & \neq 0 \end{array}\right)$$

的行——系数全为零但常数项非零，即 $0 = c$（$c \ne 0$），这是矛盾式。

### 无穷多解

方程组有**无穷多个解**，可以用含自由参数的表达式描述。

**示例：**

$$\begin{cases} x_1 + 2x_2 = 4 \\ 2x_1 + 4x_2 = 8 \end{cases}$$

第二个方程是第一个的 2 倍，实际上只有一个独立约束：$x_1 = 4 - 2x_2$。令 $x_2 = t$（$t$ 为任意实数），则

$$\mathbf{x} = \begin{pmatrix} 4 - 2t \\ t \end{pmatrix} = \begin{pmatrix} 4 \\ 0 \end{pmatrix} + t\begin{pmatrix} -2 \\ 1 \end{pmatrix}, \quad t \in \mathbb{R}$$

解集是一条直线（过点 $(4, 0)$，方向为 $(-2, 1)$），有无穷多个。

### 几何解释

以二元一次方程组（每个方程对应平面上的一条直线）为例：

| 解的情形 | 几何含义 |
|----------|----------|
| 唯一解 | 两条直线**相交于一点** |
| 无解 | 两条直线**平行**（不相交） |
| 无穷多解 | 两条直线**重合** |

三元一次方程组（每个方程对应空间中的一个平面）同理：三个平面可能交于一点（唯一解）、没有公共点（无解）或交于一条线乃至一个平面（无穷多解）。

**关键定理（相容性判断）：**

方程组 $A\mathbf{x} = \mathbf{b}$ 有解（相容）的充要条件是：

$$\text{rank}(A) = \text{rank}([A \mid \mathbf{b}])$$

即，增广矩阵与系数矩阵的秩相同——增加常数列不增加新的约束。若 $\text{rank}(A) < \text{rank}([A \mid \mathbf{b}])$，则无解。

> 秩（rank）的严格定义见第6章，此处直觉理解为"矩阵中独立行（或列）的数目"即可。

当方程组有解时：
- 若 $\text{rank}(A) = n$（未知量个数），则解**唯一**
- 若 $\text{rank}(A) < n$，则有**无穷多解**，自由变量个数为 $n - \text{rank}(A)$

---

## 4.4 齐次线性方程组

### 定义

当方程组右端项全为零时，即 $\mathbf{b} = \mathbf{0}$，方程组

$$A\mathbf{x} = \mathbf{0}$$

称为**齐次线性方程组**。

对应的增广矩阵为 $[A \mid \mathbf{0}]$，最后一列全为零。

**示例：**

$$\begin{cases} x_1 + 2x_2 - x_3 = 0 \\ 2x_1 - x_2 + x_3 = 0 \end{cases}$$

### 平凡解与非平凡解

齐次方程组**永远有解**——零向量 $\mathbf{x} = \mathbf{0}$ 总是满足 $A\mathbf{0} = \mathbf{0}$，称为**平凡解**（trivial solution）。

若存在非零向量 $\mathbf{x} \ne \mathbf{0}$ 也满足 $A\mathbf{x} = \mathbf{0}$，则称之为**非平凡解**（nontrivial solution）。

**非平凡解存在的条件：** 当 $\text{rank}(A) < n$（方程数 $m$ 小于未知量数 $n$，或方程存在线性相关时），齐次方程组必有非平凡解。

**直觉：** 若变量多于独立约束，方程系统就"过于自由"——在满足所有约束的同时，还有额外的自由度，导致非平凡解的存在。

**示例验证：** 对上面的齐次方程组，可以验证 $\mathbf{x} = (1, 1, 3)^T$ 是一个非平凡解：

$$1 + 2 \cdot 1 - 3 = 0 \checkmark, \quad 2 \cdot 1 - 1 + 3 = 4 \ne 0$$

让我们重新找一个正确的非平凡解。用消元法：

$$\begin{pmatrix} 1 & 2 & -1 \\ 2 & -1 & 1 \end{pmatrix} \xrightarrow{R_2 \to R_2 - 2R_1} \begin{pmatrix} 1 & 2 & -1 \\ 0 & -5 & 3 \end{pmatrix}$$

由第二行：$x_2 = \frac{3}{5}x_3$；代入第一行：$x_1 = x_3 - 2x_2 = x_3 - \frac{6}{5}x_3 = -\frac{1}{5}x_3$。

令 $x_3 = 5$，得非平凡解 $\mathbf{x} = (-1, 3, 5)^T$（可验证代入两方程均为0）。

### 解空间

齐次方程组 $A\mathbf{x} = \mathbf{0}$ 的所有解构成的集合称为 $A$ 的**零空间**（null space）或**核**（kernel），记为 $\ker(A)$ 或 $\mathcal{N}(A)$。

**零空间是向量空间：** 若 $\mathbf{u}$ 和 $\mathbf{v}$ 均为 $A\mathbf{x} = \mathbf{0}$ 的解，则对任意标量 $c, d$：

$$A(c\mathbf{u} + d\mathbf{v}) = cA\mathbf{u} + dA\mathbf{v} = c\mathbf{0} + d\mathbf{0} = \mathbf{0}$$

所以 $c\mathbf{u} + d\mathbf{v}$ 也是解——解集对线性组合封闭，构成向量空间。

> **对比非齐次方程组：** $A\mathbf{x} = \mathbf{b}$（$\mathbf{b} \ne \mathbf{0}$）的解集**不是**向量空间，因为 $\mathbf{0}$ 不是其解。非齐次方程组的完整解可以表示为：一个**特解** $\mathbf{x}_p$（满足 $A\mathbf{x}_p = \mathbf{b}$）加上对应齐次方程组的通解：
>
> $$\mathbf{x} = \mathbf{x}_p + \mathbf{x}_h, \quad A\mathbf{x}_h = \mathbf{0}$$

**零空间的维数**（称为矩阵的**零化度**，nullity）等于 $n - \text{rank}(A)$，这是秩-零化度定理的结论，将在后续章节严格证明。

---

## 本章小结

| 概念 | 描述 |
|------|------|
| 线性方程 | $a_1x_1 + \cdots + a_nx_n = b$，各未知量均为一次 |
| 系数矩阵 $A$ | 由所有系数 $a_{ij}$ 构成的 $m \times n$ 矩阵 |
| 增广矩阵 $[A \mid \mathbf{b}]$ | 系数矩阵右侧拼接常数列，形状为 $m \times (n+1)$ |
| 矩阵方程 | $A\mathbf{x} = \mathbf{b}$ 完整表达整个方程组 |
| 唯一解 | $\text{rank}(A) = \text{rank}([A \mid \mathbf{b}]) = n$ |
| 无穷多解 | $\text{rank}(A) = \text{rank}([A \mid \mathbf{b}]) < n$ |
| 无解 | $\text{rank}(A) < \text{rank}([A \mid \mathbf{b}])$ |
| 齐次方程组 | $A\mathbf{x} = \mathbf{0}$，必有平凡解 $\mathbf{x} = \mathbf{0}$ |
| 零空间 $\ker(A)$ | 齐次方程组全体解，是一个向量空间 |

**核心要点回顾：**

1. 线性方程组可以用矩阵方程 $A\mathbf{x} = \mathbf{b}$ 简洁表达，增广矩阵 $[A \mid \mathbf{b}]$ 是分析解的核心工具。
2. 解集只有三种情形：无解、唯一解、无穷多解，不存在有限多个解的情况。
3. 解的存在性由秩判断：$\text{rank}(A) = \text{rank}([A \mid \mathbf{b}])$ 是有解的充要条件。
4. 齐次方程组必有零解，其解集（零空间）是一个向量空间；非齐次方程组的通解 = 特解 + 齐次通解。

---

## 深度学习应用

### 神经网络参数求解

训练神经网络的本质是求解一个参数优化问题。在最简单的情形——**线性回归**——中，这等价于求解一个线性方程组。

给定 $m$ 个训练样本 $(\mathbf{x}_i, y_i)$，寻找参数向量 $\mathbf{w}$ 使得 $\mathbf{x}_i^T \mathbf{w} \approx y_i$，写成矩阵形式就是：

$$X\mathbf{w} = \mathbf{y}$$

其中 $X \in \mathbb{R}^{m \times n}$（数据矩阵，每行是一个样本），$\mathbf{w} \in \mathbb{R}^n$（参数向量），$\mathbf{y} \in \mathbb{R}^m$（标签）。

- 若 $m = n$ 且 $X$ 可逆：存在唯一解 $\mathbf{w} = X^{-1}\mathbf{y}$
- 若 $m > n$（样本多于参数，通常情形）：方程组过定，一般无精确解，需改用最小二乘法求近似解 $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$
- 若 $m < n$（参数多于样本，过参数化模型）：方程组欠定，有无穷多解，需要正则化来选择"最好的"解

### 线性层的方程视角

神经网络中的全连接层（Linear Layer）执行的变换：

$$\mathbf{y} = W\mathbf{x} + \mathbf{b}$$

从方程组的视角看，这是对输入向量 $\mathbf{x}$ 施加一个线性方程系统。权重矩阵 $W$ 是系数矩阵，偏置 $\mathbf{b}$ 是常数项，输出 $\mathbf{y}$ 是右端项。

多层网络叠加线性层时（中间无激活函数），整体变换仍是线性的：

$$\mathbf{y} = W_L (W_{L-1}(\cdots W_1 \mathbf{x} \cdots)) = (W_L W_{L-1} \cdots W_1)\mathbf{x}$$

这说明**纯线性网络的表达能力等价于单层线性网络**，这正是激活函数（ReLU、Sigmoid 等非线性变换）不可或缺的原因。

### 约束优化中的线性约束

在强化学习、物理仿真等场景中，优化问题常带有线性等式约束：

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to} \quad A\mathbf{x} = \mathbf{b}$$

这里 $A\mathbf{x} = \mathbf{b}$ 正是非齐次线性方程组——它定义了可行解的集合。利用本章的理论，可以把问题投影到零空间方向（齐次解部分）上来降维，将有约束优化转化为无约束优化。

### 代码示例（PyTorch）

```python
import torch
import numpy as np

# ── 示例1：用 torch.linalg.solve 求解方程组 ─────────────────
# 解方程组 Ax = b
#   2x1 +  x2 = 5
#    x1 - x2  = 1

A = torch.tensor([[2.0, 1.0],
                  [1.0, -1.0]])
b = torch.tensor([5.0, 1.0])

x = torch.linalg.solve(A, b)
print(f"唯一解: x = {x}")          # tensor([2., 1.])
print(f"验证 Ax = {A @ x}")         # tensor([5., 1.]) ✓

# ── 示例2：增广矩阵与解的判断 ────────────────────────────────
def check_system(A, b):
    """判断线性方程组 Ax=b 解的类型"""
    A_aug = torch.cat([A, b.unsqueeze(1)], dim=1)   # 构造增广矩阵

    rank_A = torch.linalg.matrix_rank(A)
    rank_aug = torch.linalg.matrix_rank(A_aug)
    n = A.shape[1]   # 未知量个数

    print(f"系数矩阵形状: {A.shape}, 秩: {rank_A.item()}")
    print(f"增广矩阵形状: {A_aug.shape}, 秩: {rank_aug.item()}")
    print(f"未知量个数 n = {n}")

    if rank_A < rank_aug:
        print("-> 无解（不相容）")
    elif rank_A == rank_aug == n:
        print("-> 唯一解")
    else:
        print(f"-> 无穷多解（自由变量个数: {n - rank_A.item()}）")

# 唯一解
print("=== 情形1：唯一解 ===")
A1 = torch.tensor([[2.0, 1.0], [1.0, -1.0]])
b1 = torch.tensor([5.0, 1.0])
check_system(A1, b1)

# 无解
print("\n=== 情形2：无解 ===")
A2 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
b2 = torch.tensor([2.0, 5.0])
check_system(A2, b2)

# 无穷多解
print("\n=== 情形3：无穷多解 ===")
A3 = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
b3 = torch.tensor([4.0, 8.0])
check_system(A3, b3)

# ── 示例3：线性回归的最小二乘解 ──────────────────────────────
# 超定方程组 (m > n)：用伪逆求最小二乘解
torch.manual_seed(42)
m, n = 100, 3   # 100个样本，3个特征

X = torch.randn(m, n)
w_true = torch.tensor([1.0, -2.0, 3.0])
y = X @ w_true + 0.1 * torch.randn(m)   # 带噪声的标签

# 最小二乘解：w = (X^T X)^{-1} X^T y
# 等价于 torch.linalg.lstsq
result = torch.linalg.lstsq(X, y)
w_est = result.solution
print(f"\n线性回归最小二乘解:")
print(f"真实参数:   {w_true.tolist()}")
print(f"估计参数:   {[round(v, 4) for v in w_est.tolist()]}")

# ── 示例4：齐次方程组的零空间 ────────────────────────────────
# 求矩阵 A 的零空间（核），即 Ax = 0 的所有解
A4 = torch.tensor([[1.0, 2.0, -1.0],
                   [2.0, -1.0, 1.0]])

# 用 SVD 求零空间：A = U Sigma V^T，零空间由 V 中对应零奇异值的列组成
U, S, Vh = torch.linalg.svd(A4, full_matrices=True)
print(f"\n矩阵 A 的形状: {A4.shape}")
print(f"奇异值: {S}")

# 找近似为零的奇异值对应的右奇异向量
tol = 1e-6
null_mask = S < tol   # 哪些奇异值约为零
# 对于 m < n 的矩阵，Vh 最后 (n - rank) 行构成零空间基
rank = (S > tol).sum().item()
null_space = Vh[rank:].T   # 零空间的基向量（列）
print(f"矩阵秩: {rank}")
if null_space.shape[1] > 0:
    v = null_space[:, 0]
    print(f"零空间基向量: {v.tolist()}")
    print(f"验证 A @ v = {(A4 @ v).tolist()}")   # 应接近 [0, 0]
```

---

## 练习题

**练习 4.1（写出矩阵表示）**

将下列方程组写成矩阵方程 $A\mathbf{x} = \mathbf{b}$ 的形式，并给出系数矩阵 $A$、未知向量 $\mathbf{x}$、常数向量 $\mathbf{b}$ 以及增广矩阵 $[A \mid \mathbf{b}]$：

$$\begin{cases} 3x_1 - x_2 + 2x_3 = 1 \\ x_1 + 4x_2 = -3 \\ 2x_1 - x_2 + x_3 = 0 \end{cases}$$

---

**练习 4.2（解的分类）**

判断下列各方程组解的类型（无解、唯一解或无穷多解），并给出理由。

(a) $\begin{cases} x + 2y = 4 \\ 3x + 6y = 12 \end{cases}$

(b) $\begin{cases} x + y = 3 \\ x - y = 1 \\ 2x + y = 5 \end{cases}$

(c) $\begin{cases} x_1 + x_2 + x_3 = 6 \\ x_1 - x_2 + 2x_3 = 5 \\ 2x_1 + x_3 = 7 \end{cases}$

---

**练习 4.3（齐次方程组）**

对齐次方程组

$$\begin{cases} x_1 + x_2 - 2x_3 = 0 \\ 2x_1 - x_2 + x_3 = 0 \end{cases}$$

(a) 用消元法求通解（含自由参数的表达式）。

(b) 验证通解中任意两个解的线性组合仍是解，以此说明解集是向量空间。

---

**练习 4.4（矩阵方程的列视角）**

设 $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$，$\mathbf{b} = \begin{pmatrix} 5 \\ 11 \\ 17 \end{pmatrix}$。

(a) 将 $A\mathbf{x} = \mathbf{b}$ 用列视角写成"$\mathbf{b}$ 是 $A$ 各列的线性组合"的形式。

(b) 观察 $\mathbf{b}$ 与 $A$ 的各列之间的规律，猜测一组解 $(x_1, x_2)$，并验证。

---

**练习 4.5（应用：欠定与过定系统）**

在机器学习中，$n$ 个参数、$m$ 个训练样本对应矩阵方程 $X\mathbf{w} = \mathbf{y}$，其中 $X \in \mathbb{R}^{m \times n}$。

(a) 当 $m < n$ 时（参数多于样本，即**过参数化**），方程组欠定，解有无穷多个。为什么神经网络在这种情况下仍能工作良好？（提示：考虑正则化和梯度下降的隐式偏置。）

(b) 当 $m > n$ 时（样本多于参数），方程组通常无精确解。写出最小二乘问题 $\min_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|^2$ 的正规方程（Normal Equations），并说明它与原方程组的关系。

(c) 当 $m = n$ 且 $X$ 可逆时，最小二乘解退化为什么？

---

## 练习答案

<details>
<summary>练习 4.1 答案</summary>

$$A = \begin{pmatrix} 3 & -1 & 2 \\ 1 & 4 & 0 \\ 2 & -1 & 1 \end{pmatrix}, \quad \mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} 1 \\ -3 \\ 0 \end{pmatrix}$$

矩阵方程：$\begin{pmatrix} 3 & -1 & 2 \\ 1 & 4 & 0 \\ 2 & -1 & 1 \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} = \begin{pmatrix} 1 \\ -3 \\ 0 \end{pmatrix}$

增广矩阵：

$$[A \mid \mathbf{b}] = \left(\begin{array}{ccc|c} 3 & -1 & 2 & 1 \\ 1 & 4 & 0 & -3 \\ 2 & -1 & 1 & 0 \end{array}\right)$$

注意第二个方程中 $x_3$ 的系数为 0，在矩阵中该位置填 0，不可省略。

</details>

<details>
<summary>练习 4.2 答案</summary>

**(a)** 第二个方程是第一个方程的 3 倍（$3(x + 2y) = 3 \cdot 4 \Rightarrow 3x + 6y = 12$），两方程等价，只有一个独立约束，未知量有 2 个。因此有**无穷多解**。

通解：令 $y = t$，则 $x = 4 - 2t$，即 $\mathbf{x} = (4, 0)^T + t(-2, 1)^T$，$t \in \mathbb{R}$。

**(b)** 对增广矩阵做行变换：

$$\left(\begin{array}{cc|c} 1 & 1 & 3 \\ 1 & -1 & 1 \\ 2 & 1 & 5 \end{array}\right) \xrightarrow{R_2-R_1,\ R_3-2R_1} \left(\begin{array}{cc|c} 1 & 1 & 3 \\ 0 & -2 & -2 \\ 0 & -1 & -1 \end{array}\right) \xrightarrow{R_3-\frac{1}{2}R_2} \left(\begin{array}{cc|c} 1 & 1 & 3 \\ 0 & -2 & -2 \\ 0 & 0 & 0 \end{array}\right)$$

由第二行：$y = 1$；代入第一行：$x = 2$。三个方程相容，解**唯一**：$(x, y) = (2, 1)$。

**(c)** 对增广矩阵化简：

$$\left(\begin{array}{ccc|c} 1 & 1 & 1 & 6 \\ 1 & -1 & 2 & 5 \\ 2 & 0 & 1 & 7 \end{array}\right) \xrightarrow{R_2-R_1,\ R_3-2R_1} \left(\begin{array}{ccc|c} 1 & 1 & 1 & 6 \\ 0 & -2 & 1 & -1 \\ 0 & -2 & -1 & -5 \end{array}\right) \xrightarrow{R_3-R_2} \left(\begin{array}{ccc|c} 1 & 1 & 1 & 6 \\ 0 & -2 & 1 & -1 \\ 0 & 0 & -2 & -4 \end{array}\right)$$

由第三行：$x_3 = 2$；第二行：$x_2 = (1 + x_3)/2 = \frac{3}{2}$... 让我们直接代回：$-2x_2 + 2 = -1 \Rightarrow x_2 = 3/2$；第一行：$x_1 = 6 - x_2 - x_3 = 6 - 3/2 - 2 = 5/2$。解**唯一**：$(x_1, x_2, x_3) = (5/2, 3/2, 2)$。

</details>

<details>
<summary>练习 4.3 答案</summary>

**(a)** 对增广矩阵（常数列全为 0）做行变换：

$$\left(\begin{array}{ccc|c} 1 & 1 & -2 & 0 \\ 2 & -1 & 1 & 0 \end{array}\right) \xrightarrow{R_2 - 2R_1} \left(\begin{array}{ccc|c} 1 & 1 & -2 & 0 \\ 0 & -3 & 5 & 0 \end{array}\right)$$

由第二行：$x_2 = \frac{5}{3}x_3$；代入第一行：$x_1 = 2x_3 - x_2 = 2x_3 - \frac{5}{3}x_3 = \frac{1}{3}x_3$。

令 $x_3 = 3t$（取 3 的倍数以消去分母），则通解为：

$$\mathbf{x} = t\begin{pmatrix} 1 \\ 5 \\ 3 \end{pmatrix}, \quad t \in \mathbb{R}$$

可验证：$1 + 5 - 2 \cdot 3 = 0$ $\checkmark$，$2 \cdot 1 - 5 + 3 = 0$ $\checkmark$。

**(b)** 取两个解 $\mathbf{u} = (1, 5, 3)^T$（$t=1$），$\mathbf{v} = (2, 10, 6)^T$（$t=2$）。

设 $\mathbf{w} = c_1 \mathbf{u} + c_2 \mathbf{v} = (c_1 + 2c_2)(1, 5, 3)^T$，仍是通解的形式（令 $t = c_1 + 2c_2$），所以 $\mathbf{w}$ 也满足方程组。

更一般地，若 $A\mathbf{u} = \mathbf{0}$ 且 $A\mathbf{v} = \mathbf{0}$，则 $A(c_1\mathbf{u} + c_2\mathbf{v}) = c_1 A\mathbf{u} + c_2 A\mathbf{v} = \mathbf{0}$，线性组合仍是解，解集对线性组合封闭，因此是向量空间（零空间）。

</details>

<details>
<summary>练习 4.4 答案</summary>

**(a)** 列视角写法：

$$x_1 \begin{pmatrix} 1 \\ 3 \\ 5 \end{pmatrix} + x_2 \begin{pmatrix} 2 \\ 4 \\ 6 \end{pmatrix} = \begin{pmatrix} 5 \\ 11 \\ 17 \end{pmatrix}$$

即：$\mathbf{b}$ 是 $A$ 第一列与第二列的线性组合，系数分别为 $x_1$ 和 $x_2$。

**(b)** 观察规律：

$$\begin{pmatrix} 5 \\ 11 \\ 17 \end{pmatrix} = 1 \cdot \begin{pmatrix} 1 \\ 3 \\ 5 \end{pmatrix} + 2 \cdot \begin{pmatrix} 2 \\ 4 \\ 6 \end{pmatrix}$$

验证：$1 \cdot 1 + 2 \cdot 2 = 5$ $\checkmark$，$1 \cdot 3 + 2 \cdot 4 = 11$ $\checkmark$，$1 \cdot 5 + 2 \cdot 6 = 17$ $\checkmark$。

因此解为 $(x_1, x_2) = (1, 2)$。

</details>

<details>
<summary>练习 4.5 答案</summary>

**(a)** 当 $m < n$ 时，方程组欠定，有无穷多解。神经网络在此情形仍表现良好，原因在于：

- **隐式正则化：** 梯度下降（尤其是随机梯度下降）在无穷多解中有偏好地找到"简单"解（范数小的解），这提供了隐式的正则化效果。
- **显式正则化：** L2 正则化（权重衰减）将问题变为 $\min \|X\mathbf{w} - \mathbf{y}\|^2 + \lambda\|\mathbf{w}\|^2$，在无穷多精确解中选择范数最小的那个（岭回归解）。
- **泛化能力：** 选到范数小或结构简单的解往往具有更好的泛化能力（奥卡姆剃刀原则的体现）。

**(b)** 最小二乘问题 $\min_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|^2$ 对 $\mathbf{w}$ 求梯度并令其为零：

$$\nabla_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|^2 = 2X^T(X\mathbf{w} - \mathbf{y}) = \mathbf{0}$$

得到**正规方程**（Normal Equations）：

$$X^T X \mathbf{w} = X^T \mathbf{y}$$

这是一个 $n \times n$ 的线性方程组（$n$ 个未知量，$n$ 个方程）。当 $X^T X$ 可逆时，唯一解为 $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$。

与原方程组 $X\mathbf{w} = \mathbf{y}$ 的关系：两边同乘 $X^T$ 就得到正规方程——相当于将每个方程"投影"到各特征方向上，把过定系统转化为可解的方形系统。

**(c)** 当 $m = n$ 且 $X$ 可逆时，$X^T X$ 也可逆，正规方程的解为：

$$\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y} = X^{-1}(X^T)^{-1} X^T \mathbf{y} = X^{-1}\mathbf{y}$$

退化为原方程组的精确解，与直接求解 $X\mathbf{w} = \mathbf{y}$ 等价。

</details>
