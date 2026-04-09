# 第2章：矩阵

> 矩阵是线性代数的核心对象，也是现代机器学习与深度学习的基础数据结构。理解矩阵，就是理解如何系统地组织和变换数据。

---

## 学习目标

学完本章后，你应该能够：

- 理解矩阵的严格代数定义和规范表示方法
- 掌握矩阵的基本概念：行、列、元素、维度（形状）
- 识别并构造各种特殊矩阵：零矩阵、单位矩阵、对角矩阵、三角矩阵、对称矩阵
- 理解矩阵相等的准确含义
- 建立"矩阵 = 结构化数据容器"的直觉，为后续矩阵运算打下基础

---

## 2.1 矩阵的定义

### 2.1.1 代数定义

**矩阵**（Matrix）是将若干数按照矩形阵列排列所形成的数学对象。具体而言，一个 $m \times n$ 矩阵由 $m$ 行、$n$ 列共 $mn$ 个数（称为**元素**或**分量**）组成。

正式定义如下：

$$A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}$$

其中每个 $a_{ij}$ 是一个实数（或复数），称为矩阵 $A$ 的**第 $i$ 行第 $j$ 列元素**。

### 2.1.2 记号约定

线性代数中有几种常见的书写约定：

| 记号 | 含义 |
|------|------|
| $A$、$B$、$C$ | 用大写粗体或大写斜体字母表示矩阵 |
| $a_{ij}$ 或 $A_{ij}$ | 矩阵 $A$ 的第 $i$ 行第 $j$ 列元素 |
| $A \in \mathbb{R}^{m \times n}$ | $A$ 是元素为实数的 $m \times n$ 矩阵 |
| $(A)_{ij}$ | 取矩阵 $A$ 的 $(i,j)$ 元素 |

**维度（形状）**用 $m \times n$（读作"$m$ 乘 $n$"）表示，其中 $m$ 是行数，$n$ 是列数。例如：

$$B = \begin{pmatrix} 1 & 0 & -3 \\ 2 & 5 & 7 \end{pmatrix} \in \mathbb{R}^{2 \times 3}$$

$B$ 有 2 行 3 列，$b_{12} = 0$，$b_{23} = 7$。

### 2.1.3 行、列与元素

对于矩阵 $A \in \mathbb{R}^{m \times n}$：

- **第 $i$ 行**（row $i$）：$(a_{i1},\ a_{i2},\ \ldots,\ a_{in})$，共 $n$ 个元素
- **第 $j$ 列**（column $j$）：$(a_{1j},\ a_{2j},\ \ldots,\ a_{mj})^T$，共 $m$ 个元素
- **主对角线**（main diagonal）：元素 $a_{11},\ a_{22},\ \ldots,\ a_{\min(m,n)\,\min(m,n)}$

> **索引顺序提示**：下标 $a_{ij}$ 遵循"先行后列"的规则——$i$ 是行号，$j$ 是列号。这与坐标系的 $(x, y)$ 顺序相反，初学时需特别注意。

---

## 2.2 矩阵的类型

掌握各类特殊矩阵的名称与性质，是读懂教材、论文和代码的必要基础。

### 2.2.1 方阵（Square Matrix）

行数等于列数（$m = n$）的矩阵称为 **$n$ 阶方阵**。

$$S = \begin{pmatrix} 3 & 1 \\ -2 & 4 \end{pmatrix} \in \mathbb{R}^{2 \times 2}$$

方阵是线性代数中最重要的一类矩阵，行列式、特征值等概念均只针对方阵定义。

### 2.2.2 行矩阵与列矩阵

- **行矩阵**（Row Matrix / Row Vector）：只有 1 行，形如 $1 \times n$。

$$\mathbf{r} = \begin{pmatrix} 1 & -3 & 0 & 5 \end{pmatrix} \in \mathbb{R}^{1 \times 4}$$

- **列矩阵**（Column Matrix / Column Vector）：只有 1 列，形如 $m \times 1$。

$$\mathbf{c} = \begin{pmatrix} 2 \\ -1 \\ 7 \end{pmatrix} \in \mathbb{R}^{3 \times 1}$$

行矩阵和列矩阵本质上就是**向量**，第1章介绍的向量可以看作列矩阵的特殊情形。

### 2.2.3 零矩阵（Zero Matrix）

所有元素均为 $0$ 的矩阵，记作 $O$ 或 $\mathbf{0}_{m \times n}$。

$$O_{2 \times 3} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

零矩阵在矩阵加法中扮演"零元素"的角色，类似实数中的 $0$。

### 2.2.4 单位矩阵（Identity Matrix）

**$n$ 阶单位矩阵** $I_n$（有时记作 $E_n$）是主对角线全为 1、其余元素全为 0 的方阵：

$$I_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

用 Kronecker delta 符号可以简洁地表示其元素：

$$(I_n)_{ij} = \delta_{ij} = \begin{cases} 1, & i = j \\ 0, & i \neq j \end{cases}$$

单位矩阵是矩阵乘法的"乘法单位元"，满足 $AI_n = I_m A = A$（对任意 $A \in \mathbb{R}^{m \times n}$）。

### 2.2.5 对角矩阵（Diagonal Matrix）

非主对角线元素全为 0 的方阵。若主对角线元素为 $d_1, d_2, \ldots, d_n$，常记作：

$$D = \mathrm{diag}(d_1, d_2, \ldots, d_n) = \begin{pmatrix} d_1 & & \\ & d_2 & \\ & & \ddots & \\ & & & d_n \end{pmatrix}$$

单位矩阵 $I_n = \mathrm{diag}(1, 1, \ldots, 1)$ 是对角矩阵的特例。对角矩阵计算极为方便：两个同阶对角矩阵的乘积仍是对角矩阵，其对角元素分别相乘。

### 2.2.6 上三角矩阵与下三角矩阵

- **上三角矩阵**（Upper Triangular Matrix）：主对角线以下的元素全为 0，即 $i > j$ 时 $a_{ij} = 0$。

$$U = \begin{pmatrix} 2 & 3 & 1 \\ 0 & -1 & 4 \\ 0 & 0 & 5 \end{pmatrix}$$

- **下三角矩阵**（Lower Triangular Matrix）：主对角线以上的元素全为 0，即 $i < j$ 时 $a_{ij} = 0$。

$$L = \begin{pmatrix} 1 & 0 & 0 \\ 2 & 3 & 0 \\ -1 & 4 & 7 \end{pmatrix}$$

三角矩阵在数值线性代数中极为重要。LU 分解（第6章）将一般方阵分解为一个下三角矩阵与一个上三角矩阵的乘积，是求解线性方程组的高效算法基础。

### 2.2.7 对称矩阵与反对称矩阵

设 $A \in \mathbb{R}^{n \times n}$ 为方阵，$A^T$ 表示 $A$ 的转置（第3章详细介绍）。

- **对称矩阵**（Symmetric Matrix）：满足 $A^T = A$，即 $a_{ij} = a_{ji}$。

$$S = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 5 & -1 \\ 3 & -1 & 4 \end{pmatrix}$$

对称矩阵沿主对角线"镜像对称"。协方差矩阵、Gram 矩阵均是对称矩阵。

- **反对称矩阵**（Skew-Symmetric / Antisymmetric Matrix）：满足 $A^T = -A$，即 $a_{ij} = -a_{ji}$。由此可知反对称矩阵的对角线元素必须全为 0。

$$K = \begin{pmatrix} 0 & -2 & 1 \\ 2 & 0 & -3 \\ -1 & 3 & 0 \end{pmatrix}$$

> **分解定理**：任意方阵 $A$ 均可唯一分解为一个对称矩阵与一个反对称矩阵之和：
> $$A = \underbrace{\frac{A + A^T}{2}}_{\text{对称部分}} + \underbrace{\frac{A - A^T}{2}}_{\text{反对称部分}}$$

---

## 2.3 矩阵的相等

### 2.3.1 相等的定义

两个矩阵 $A$ 和 $B$ **相等**（记作 $A = B$）当且仅当：

1. 它们具有相同的维度：$A \in \mathbb{R}^{m \times n}$ 且 $B \in \mathbb{R}^{m \times n}$
2. 对应位置的每个元素均相等：$a_{ij} = b_{ij}$，对所有 $1 \le i \le m$，$1 \le j \le n$ 成立

**两个条件缺一不可。** 维度不同的矩阵，即便包含相同的数字，也不相等。

### 2.3.2 示例

$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad B = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad C = \begin{pmatrix} 1 & 2 & 0 \\ 3 & 4 & 0 \end{pmatrix}$$

- $A = B$：维度相同（$2 \times 2$），所有对应元素相等。
- $A \neq C$：维度不同（$2 \times 2$ vs $2 \times 3$），无法比较。

### 2.3.3 用相等求解未知量

矩阵相等的条件可用于建立方程组。例如，若

$$\begin{pmatrix} x + y & 2 \\ 3 & x - y \end{pmatrix} = \begin{pmatrix} 5 & 2 \\ 3 & 1 \end{pmatrix}$$

则由元素对应相等得：$x + y = 5$ 且 $x - y = 1$，解得 $x = 3,\ y = 2$。

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 矩阵 $A \in \mathbb{R}^{m \times n}$ | $m$ 行 $n$ 列的数表；元素 $a_{ij}$ 位于第 $i$ 行第 $j$ 列 |
| 方阵 | $m = n$；行列式、特征值等概念的前提 |
| 零矩阵 $O$ | 所有元素为 0；加法单位元 |
| 单位矩阵 $I_n$ | 对角线为 1，其余为 0；乘法单位元 |
| 对角矩阵 | 非对角元素全为 0；$\mathrm{diag}(d_1,\ldots,d_n)$ |
| 上/下三角矩阵 | 对角线一侧全为 0；LU 分解的基础 |
| 对称矩阵 | $A^T = A$；协方差矩阵的典型形式 |
| 反对称矩阵 | $A^T = -A$；对角线元素必为 0 |
| 矩阵相等 | 维度相同 + 所有对应元素相等 |

**下一章预告**：掌握矩阵的定义后，我们将学习矩阵的加法、数乘和矩阵乘法——这些运算构成了神经网络前向传播的数学核心。

---

## 深度学习应用

### 概念回顾

矩阵是深度学习框架（PyTorch、TensorFlow、JAX）的核心数据结构。在代码层面，矩阵对应**二维张量（2D Tensor）**。理解矩阵的数学结构，有助于正确理解模型的参数形状、数据流动以及各种操作的含义。

### 在深度学习中的应用

**1. 权重矩阵（Weight Matrix）**

全连接层（Linear Layer）的参数存储为矩阵 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$。输入向量 $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$，输出为 $\mathbf{y} = W\mathbf{x} + \mathbf{b}$，其中 $\mathbf{b}$ 是偏置向量。一个隐藏层大小为 512、输出大小为 10 的层，其权重矩阵形状为 $(10, 512)$，包含 $5120$ 个可训练参数。

**2. 数据批处理（Mini-batch）**

训练时不逐样本计算，而是将一批 $N$ 个样本组织为矩阵 $X \in \mathbb{R}^{N \times d}$（$N$ 为批大小，$d$ 为特征维度）。矩阵运算天然支持批并行，GPU 可以高效处理大批次。

**3. 图像表示**

- 灰度图像：$H \times W$ 矩阵，每个元素是像素强度值（通常 0–255）
- 彩色图像（RGB）：$H \times W \times 3$ 三维张量（严格说是3阶张量，不是矩阵）
- 一批彩色图像：$N \times C \times H \times W$ 四维张量（PyTorch 的标准格式）

**4. 注意力分数矩阵（Attention Score Matrix）**

Transformer 中，自注意力机制计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中 $QK^T \in \mathbb{R}^{L \times L}$（$L$ 为序列长度）是一个方阵，表示序列中每个位置对其他位置的"关注程度"。

### 代码示例（Python / PyTorch）

```python
import torch
import torch.nn as nn

# ── 1. 创建矩阵 ──────────────────────────────────────────────
# 2×3 矩阵（手动指定元素）
A = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(f"A 的形状: {A.shape}")        # torch.Size([2, 3])
print(f"A[1][2] = {A[1, 2]}")       # 6.0  （第2行第3列，0-indexed）

# ── 2. 特殊矩阵 ──────────────────────────────────────────────
zeros = torch.zeros(3, 4)           # 3×4 零矩阵
ones  = torch.ones(2, 2)            # 2×2 全1矩阵
I3    = torch.eye(3)                # 3阶单位矩阵
D     = torch.diag(torch.tensor([2.0, -1.0, 3.0]))  # 对角矩阵
print(f"\n单位矩阵 I3:\n{I3}")
print(f"\n对角矩阵 D:\n{D}")

# ── 3. 全连接层的权重矩阵 ─────────────────────────────────────
d_in, d_out = 512, 10
linear = nn.Linear(d_in, d_out, bias=True)
print(f"\n权重矩阵形状: {linear.weight.shape}")  # torch.Size([10, 512])
print(f"偏置向量形状: {linear.bias.shape}")      # torch.Size([10])

# ── 4. 批数据矩阵 ─────────────────────────────────────────────
batch_size, num_features = 32, 512
X = torch.randn(batch_size, num_features)  # 模拟一批数据
print(f"\n数据矩阵 X 形状: {X.shape}")           # torch.Size([32, 512])

# ── 5. 检验对称性 ─────────────────────────────────────────────
# 协方差矩阵 X^T X 是对称矩阵
cov = X.T @ X                             # shape: [512, 512]
is_sym = torch.allclose(cov, cov.T, atol=1e-5)
print(f"\nX^T X 是对称矩阵: {is_sym}")    # True

# ── 6. 矩阵相等判断 ───────────────────────────────────────────
B = A.clone()
print(f"\nA == B (元素级): {torch.all(A == B).item()}")   # True
```

> **运行环境**：需安装 `torch`（`pip install torch`）。代码在 PyTorch 2.x 下验证通过。

### 延伸阅读

- **《深度学习》（Goodfellow 等）** 第2章：线性代数——系统介绍深度学习所需的矩阵知识
- **3Blue1Brown《线性代数的本质》** YouTube 系列——可视化理解矩阵的几何意义
- **PyTorch 官方文档**：[torch.Tensor](https://pytorch.org/docs/stable/tensors.html)——了解张量（矩阵）的完整 API
- **《矩阵计算》（Golub & Van Loan）** ——数值线性代数经典教材，适合进阶学习

---

## 练习题

**题1（基础）** 写出矩阵 $A = \begin{pmatrix} 5 & -2 & 0 \\ 1 & 3 & 7 \\ -4 & 6 & 2 \end{pmatrix}$ 的维度，并分别给出元素 $a_{13}$、$a_{31}$、$a_{22}$ 的值。

---

**题2（特殊矩阵识别）** 判断下列每个矩阵属于哪种特殊类型（可能不止一种），并说明理由：

$$P = \begin{pmatrix} 3 & 0 \\ 0 & 3 \end{pmatrix}, \quad Q = \begin{pmatrix} 0 & -1 & 2 \\ 1 & 0 & -3 \\ -2 & 3 & 0 \end{pmatrix}, \quad R = \begin{pmatrix} 1 & 4 & 7 \\ 0 & 2 & 5 \\ 0 & 0 & 3 \end{pmatrix}$$

---

**题3（矩阵相等求解）** 已知下列两个矩阵相等：

$$\begin{pmatrix} 2x - y & x + z \\ 3 & x + y \end{pmatrix} = \begin{pmatrix} 1 & 4 \\ 3 & 5 \end{pmatrix}$$

求 $x$、$y$、$z$ 的值。

---

**题4（对称分解）** 将矩阵

$$M = \begin{pmatrix} 2 & 5 & -1 \\ 1 & 3 & 4 \\ -3 & 2 & 0 \end{pmatrix}$$

分解为一个对称矩阵与一个反对称矩阵之和，即写出 $M = S + K$，其中 $S^T = S$，$K^T = -K$。

---

**题5（深度学习应用）** 一个两层全连接神经网络的结构如下：

- 输入层：784 个节点（对应 $28 \times 28$ 像素的 MNIST 手写数字图像展平后）
- 隐藏层：256 个节点
- 输出层：10 个节点（对应 10 个数字类别）

(a) 写出两个权重矩阵 $W_1$、$W_2$ 的维度。
(b) 设批大小为 $N = 64$，写出数据在每一层的矩阵形状（忽略偏置）。
(c) 该网络的权重参数总数是多少？

---

## 练习答案

<details>
<summary>点击展开答案</summary>

### 题1 答案

矩阵 $A$ 的维度为 $3 \times 3$（3行3列，是一个方阵）。

- $a_{13}$：第1行第3列，$a_{13} = 0$
- $a_{31}$：第3行第1列，$a_{31} = -4$
- $a_{22}$：第2行第2列，$a_{22} = 3$

---

### 题2 答案

**矩阵 $P$：**

$$P = \begin{pmatrix} 3 & 0 \\ 0 & 3 \end{pmatrix} = 3I_2$$

- 方阵（$2 \times 2$）
- 对角矩阵（$p_{12} = p_{21} = 0$）
- 对称矩阵（$P^T = P$）
- 上三角矩阵且同时是下三角矩阵（对角矩阵是两者的特例）
- 数量矩阵（对角元素相同）

**矩阵 $Q$：**

$$Q = \begin{pmatrix} 0 & -1 & 2 \\ 1 & 0 & -3 \\ -2 & 3 & 0 \end{pmatrix}$$

验证：$q_{12} = -1,\ q_{21} = 1 = -q_{12}$；$q_{13} = 2,\ q_{31} = -2 = -q_{13}$；对角线全为 0。

- 方阵（$3 \times 3$）
- **反对称矩阵**（$Q^T = -Q$）

**矩阵 $R$：**

$$R = \begin{pmatrix} 1 & 4 & 7 \\ 0 & 2 & 5 \\ 0 & 0 & 3 \end{pmatrix}$$

主对角线以下元素全为 0（$r_{21} = r_{31} = r_{32} = 0$）。

- 方阵（$3 \times 3$）
- **上三角矩阵**

---

### 题3 答案

由矩阵相等，对应元素分别相等：

$$\begin{cases} 2x - y = 1 & \text{（位置 (1,1)）} \\ x + z = 4 & \text{（位置 (1,2)）} \\ x + y = 5 & \text{（位置 (2,2)）} \end{cases}$$

由方程 (1) 和 (3) 相加：$(2x - y) + (x + y) = 1 + 5 \Rightarrow 3x = 6 \Rightarrow x = 2$

代入方程 (3)：$y = 5 - x = 3$

代入方程 (2)：$z = 4 - x = 2$

$$\boxed{x = 2,\quad y = 3,\quad z = 2}$$

验证：$2(2)-3=1\ \checkmark$，$2+2=4\ \checkmark$，$2+3=5\ \checkmark$

---

### 题4 答案

利用公式 $S = \dfrac{M + M^T}{2}$，$K = \dfrac{M - M^T}{2}$。

先计算 $M^T$：

$$M^T = \begin{pmatrix} 2 & 1 & -3 \\ 5 & 3 & 2 \\ -1 & 4 & 0 \end{pmatrix}$$

**对称部分 $S$：**

$$S = \frac{1}{2}\begin{pmatrix} 4 & 6 & -4 \\ 6 & 6 & 6 \\ -4 & 6 & 0 \end{pmatrix} = \begin{pmatrix} 2 & 3 & -2 \\ 3 & 3 & 3 \\ -2 & 3 & 0 \end{pmatrix}$$

验证：$S^T = S\ \checkmark$

**反对称部分 $K$：**

$$K = \frac{1}{2}\begin{pmatrix} 0 & 4 & 2 \\ -4 & 0 & 2 \\ -2 & -2 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 2 & 1 \\ -2 & 0 & 1 \\ -1 & -1 & 0 \end{pmatrix}$$

验证：$K^T = -K\ \checkmark$，对角线全为 0 $\checkmark$

**验证 $S + K = M$：**

$$\begin{pmatrix} 2 & 3 & -2 \\ 3 & 3 & 3 \\ -2 & 3 & 0 \end{pmatrix} + \begin{pmatrix} 0 & 2 & 1 \\ -2 & 0 & 1 \\ -1 & -1 & 0 \end{pmatrix} = \begin{pmatrix} 2 & 5 & -1 \\ 1 & 3 & 4 \\ -3 & 2 & 0 \end{pmatrix} = M\ \checkmark$$

---

### 题5 答案

**(a) 权重矩阵维度**

约定：层运算为 $\mathbf{y} = W\mathbf{x}$（输出维度 × 输入维度）。

- $W_1 \in \mathbb{R}^{256 \times 784}$：将 784 维输入映射到 256 维隐藏层
- $W_2 \in \mathbb{R}^{10 \times 256}$：将 256 维隐藏层映射到 10 维输出

**(b) 批数据形状（$N = 64$）**

| 位置 | 矩阵形状 | 说明 |
|------|----------|------|
| 输入 $X$ | $(64, 784)$ | 64个样本，每样本784维 |
| 隐藏层激活 $H = XW_1^T$ | $(64, 256)$ | 64个样本，每样本256维特征 |
| 输出 $Y = HW_2^T$ | $(64, 10)$ | 64个样本，每样本10个类别得分 |

**(c) 权重参数总数**

$$|W_1| + |W_2| = 256 \times 784 + 10 \times 256 = 200{,}704 + 2{,}560 = \boxed{203{,}264}$$

若加上偏置：$256 + 10 = 266$ 个偏置参数，总计 $203{,}530$ 个参数。

</details>
