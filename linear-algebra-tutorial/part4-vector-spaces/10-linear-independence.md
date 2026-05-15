# 第10章：线性相关与线性无关（融合版）

> **难度**：★★☆☆☆
> **前置知识**：第9章（向量空间）、第2章（矩阵与行化简）
> **本文件**：融合"原版严格推导 + 速记 / 套路 / 自测"。保留原版完整正文（学习目标 / 10.1–10.4 / 深度学习应用 / 练习题）+ 在最前置一例速记 / 思维路径还原 + 最后追加方法总结与自测。

> **一例速记**：
> **线性无关**：$c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$ 的唯一解是全零解 $\Leftrightarrow$ 无冗余方向 $\Leftrightarrow$ 矩阵 $[\mathbf{v}_1\mid\cdots\mid\mathbf{v}_k]$ 列满秩。
> **线性相关**：存在不全为零的系数使上式成立 $\Leftrightarrow$ 至少有一个向量能被其余向量线性表示 $\Leftrightarrow$ 行化简后存在自由列。
> **判定工具**：行化简（最通用）；行列式（仅限方阵，非零则无关）；$\text{rank}(A) = k$ 则无关。
> **张成（Span）**：所有线性组合的集合，是包含这组向量的最小子空间。
> **AI 关联**：特征冗余 = 特征矩阵列线性相关；低秩分解（LoRA）= 用线性无关的低维子空间近似高维权重矩阵。

---

## 引入：特征矩阵的冗余检测

> **题目**：某数据集有三个特征 $f_1, f_2, f_3$，对应 4 条样本的特征矩阵为：
>
> $$X = \begin{pmatrix} 1 & 2 & 4 \\ 1 & 0 & 2 \\ 0 & 1 & 1 \\ 2 & 1 & 5 \end{pmatrix}$$
>
> 其中每列代表一个特征。$f_3$ 是否"冗余"（即能被 $f_1, f_2$ 线性表示）？

请先停下来想一想：如果 $f_3 = 2f_1 + f_2$，那么 $f_3$ 提供的信息完全可以由 $f_1$ 和 $f_2$ 推算出来——这正是**线性相关**的含义。下面还原完整解题思路。

---

## 思维路径还原（解题者的内心独白）

> "题目问三列是否线性相关。我直接构造矩阵 $A = [f_1 \mid f_2 \mid f_3]$，对 $A$ 做行化简，看秩。
>
> **行化简过程**：
> $$A = \begin{pmatrix}1&2&4\\1&0&2\\0&1&1\\2&1&5\end{pmatrix}$$
> $R_2 \leftarrow R_2 - R_1$：$\begin{pmatrix}1&2&4\\0&-2&-2\\0&1&1\\2&1&5\end{pmatrix}$
>
> $R_4 \leftarrow R_4 - 2R_1$：$\begin{pmatrix}1&2&4\\0&-2&-2\\0&1&1\\0&-3&-3\end{pmatrix}$
>
> $R_2 \leftarrow -\frac{1}{2}R_2$，得第二行 $(0,1,1)$；$R_3$ 与第二行相同，$R_3 \leftarrow R_3 - R_2 = 0$；$R_4 \leftarrow R_4 + 3R_2 = 0$。
>
> 最终 REF：$\begin{pmatrix}1&2&4\\0&1&1\\0&0&0\\0&0&0\end{pmatrix}$。只有 2 个主元，第 3 列是自由列。$\text{rank}(A) = 2 < 3$（列数），故三列**线性相关**。
>
> **找具体关系**：从 REF 读出 $f_3 = 2f_1 + f_2$（即用 RREF 回代：自由变量 $c_3 = 1$ 时，$c_1 = -2,\, c_2 = -1$，即 $-2f_1 - f_2 + f_3 = \mathbf{0}$，变形得 $f_3 = 2f_1 + f_2$）。
>
> **AI 含义**：$f_3$ 完全冗余——它携带的信息已经包含在 $f_1$ 和 $f_2$ 中。保留三个特征会造成多重共线性，应删去 $f_3$（或等价地，只保留 2 个线性无关的特征）。"

---

## 学习目标

学完本章后，你将能够：

- 理解**线性组合**的概念，并能计算给定向量的线性组合
- 掌握**线性相关**和**线性无关**的严格定义，理解二者的本质区别
- 运用行化简、行列式等方法判断向量组的线性相关性
- 理解**张成（Span）**的概念，能描述一组向量所能生成的空间
- 将线性相关性与线性方程组的解的存在性联系起来

---

## 10.1 线性组合

### 定义

设 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ 是 $\mathbb{R}^n$ 中的 $k$ 个向量，$c_1, c_2, \ldots, c_k$ 是 $k$ 个实数（称为**标量**或**系数**），则向量

$$\mathbf{w} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$$

称为向量 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ 的一个**线性组合（Linear Combination）**。

**例1** 设

$$\mathbf{v}_1 = \begin{pmatrix}1\\2\end{pmatrix}, \quad \mathbf{v}_2 = \begin{pmatrix}3\\1\end{pmatrix}$$

取 $c_1=2, c_2=-1$，则

$$\mathbf{w} = 2\begin{pmatrix}1\\2\end{pmatrix} + (-1)\begin{pmatrix}3\\1\end{pmatrix} = \begin{pmatrix}2\\4\end{pmatrix} + \begin{pmatrix}-3\\-1\end{pmatrix} = \begin{pmatrix}-1\\3\end{pmatrix}$$

### 几何意义

在二维空间中，$c_1\mathbf{v}_1 + c_2\mathbf{v}_2$ 表示：先沿 $\mathbf{v}_1$ 方向缩放 $c_1$ 倍，再沿 $\mathbf{v}_2$ 方向缩放 $c_2$ 倍，两段位移首尾相接后到达的终点。

直观地说：
- 如果 $\mathbf{v}_1$ 和 $\mathbf{v}_2$ 不平行（不共线），通过调整 $c_1, c_2$，可以到达平面上的**任意一点**。
- 如果 $\mathbf{v}_1$ 和 $\mathbf{v}_2$ 平行（共线），无论如何调整系数，只能在这条直线上移动，无法覆盖整个平面。

这个几何直觉正是引出线性相关性概念的出发点。

---

## 10.2 线性相关与线性无关

### 定义

设 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k \in \mathbb{R}^n$。考虑齐次方程

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \tag{*}$$

- 若方程 $(*)$ **只有零解**（即 $c_1 = c_2 = \cdots = c_k = 0$ 是唯一解），则称这组向量**线性无关（Linearly Independent）**。
- 若方程 $(*)$ **存在非零解**（即存在不全为零的 $c_1, c_2, \ldots, c_k$ 使等式成立），则称这组向量**线性相关（Linearly Dependent）**。

> **关键视角**：线性相关意味着"其中至少有一个向量可以被其余向量的线性组合表示"，即存在"冗余"向量。线性无关则意味着每个向量都提供了独立的方向信息。

**例2** 判断以下向量组是否线性相关：

$$\mathbf{v}_1 = \begin{pmatrix}1\\0\end{pmatrix}, \quad \mathbf{v}_2 = \begin{pmatrix}0\\1\end{pmatrix}, \quad \mathbf{v}_3 = \begin{pmatrix}2\\3\end{pmatrix}$$

设 $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + c_3\mathbf{v}_3 = \mathbf{0}$，即

$$c_1 + 2c_3 = 0, \quad c_2 + 3c_3 = 0$$

取 $c_3 = 1$，则 $c_1 = -2, c_2 = -3$，非零解存在，因此这三个向量**线性相关**。

验证：$\mathbf{v}_3 = 2\mathbf{v}_1 + 3\mathbf{v}_2$，$\mathbf{v}_3$ 确实可以用 $\mathbf{v}_1, \mathbf{v}_2$ 线性表示。

### 几何解释

- **二维空间**：两个向量线性相关当且仅当它们共线（方向相同或相反）。
- **三维空间**：三个向量线性相关当且仅当它们共面（处于同一个二维平面内）。
- **通用情形**：$k$ 个 $n$ 维向量线性相关，意味着它们共同"生成"的空间维数小于 $k$，即存在"多余"的向量没有贡献新的方向。

特别地，若向量组中**包含零向量**，则该向量组必定线性相关（因为零向量的系数可取任意非零值而不影响等式成立）。

若向量组中**存在两个相同（或成比例）的向量**，该向量组也必定线性相关。

### 判定方法

**方法一：行化简法（最通用）**

将向量 $\mathbf{v}_1, \ldots, \mathbf{v}_k$ 排列为矩阵 $A$ 的列，对 $A$ 作初等行变换化为行阶梯形（REF）或简化行阶梯形（RREF）：

- 若每列都有一个**主元**（pivot），则向量组**线性无关**。
- 若存在**自由列**（无主元的列），则向量组**线性相关**。

等价地，若矩阵的秩 $\text{rank}(A) = k$（列满秩），则线性无关；若 $\text{rank}(A) < k$，则线性相关。

**方法二：行列式法（仅适用于方阵）**

当 $k = n$（向量个数等于向量维数）时，构成方阵 $A = [\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_n]$：

- $\det(A) \neq 0 \Rightarrow$ 向量组**线性无关**
- $\det(A) = 0 \Rightarrow$ 向量组**线性相关**

**例3** 判断以下三个三维向量是否线性无关：

$$\mathbf{v}_1 = \begin{pmatrix}1\\2\\3\end{pmatrix}, \quad \mathbf{v}_2 = \begin{pmatrix}0\\1\\4\end{pmatrix}, \quad \mathbf{v}_3 = \begin{pmatrix}5\\6\\0\end{pmatrix}$$

计算行列式：

$$\det\begin{pmatrix}1&0&5\\2&1&6\\3&4&0\end{pmatrix} = 1\cdot(1\cdot0 - 6\cdot4) - 0 + 5\cdot(2\cdot4 - 1\cdot3) = 1\cdot(-24) + 5\cdot5 = -24+25 = 1 \neq 0$$

因此这三个向量**线性无关**。

---

## 10.3 张成（Span）

### 定义

设 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k \in \mathbb{R}^n$，这些向量的所有线性组合构成的集合称为它们的**张成（Span）**，记作

$$\text{Span}\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\} = \{c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k \mid c_1, c_2, \ldots, c_k \in \mathbb{R}\}$$

张成是一个**子空间**（满足向量空间的封闭性），也称为由这组向量**生成**的子空间。

### 张成空间的几何形态

| 向量组情况 | 张成空间 |
|---|---|
| 仅含零向量 | $\{\mathbf{0}\}$（原点） |
| 一个非零向量 $\mathbf{v}$ | 过原点、沿 $\mathbf{v}$ 方向的直线 |
| 两个线性无关向量 | 过原点的平面 |
| $n$ 个线性无关的 $n$ 维向量 | 整个 $\mathbb{R}^n$ |

**关键事实**：向向量组中添加已经在张成空间内的向量，不会扩大张成空间。只有添加"不在当前张成中"的向量，才能增加张成空间的维度。

**例4** 设 $\mathbf{v}_1 = (1,0,0)^T$，$\mathbf{v}_2 = (0,1,0)^T$，$\mathbf{v}_3 = (2,3,0)^T$。

由于 $\mathbf{v}_3 = 2\mathbf{v}_1 + 3\mathbf{v}_2 \in \text{Span}\{\mathbf{v}_1, \mathbf{v}_2\}$，所以

$$\text{Span}\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\} = \text{Span}\{\mathbf{v}_1, \mathbf{v}_2\}$$

即 $xy$ 平面（$\mathbb{R}^3$ 中 $z=0$ 的子空间）。

### 张成与线性相关的联系

向量组 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ 线性相关，等价于其中某个向量在其余向量的张成中：

$$\exists\, i: \mathbf{v}_i \in \text{Span}\{\mathbf{v}_1, \ldots, \mathbf{v}_{i-1}, \mathbf{v}_{i+1}, \ldots, \mathbf{v}_k\}$$

反之，线性无关的向量组中，每个向量都不在其余向量的张成内——每个向量都在提供"新的方向"。

---

## 10.4 线性相关性的性质

### 基本性质

**性质1（包含零向量）** 若向量组中含有零向量，则该向量组必线性相关。

**性质2（部分组与整体）** 若向量组的**某个部分组**线性相关，则整体也线性相关；若整体线性无关，则任意部分组也线性无关。

**性质3（维数界）** $\mathbb{R}^n$ 中任意 $n+1$ 个或更多向量必线性相关。（$n$ 维空间中最多有 $n$ 个线性无关向量。）

**性质4（延伸性）** 若向量组 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ 线性无关，则在每个向量后面添加相同的分量所得到的 $(n+m)$ 维向量组也线性无关。（线性无关性不因延伸分量而破坏。）

**性质5（缩短性的逆命题）** 若 $k$ 维截断后的向量组线性相关，则原向量组也线性相关。

### 与线性方程组的联系

线性相关性本质上是**齐次方程组** $A\mathbf{x} = \mathbf{0}$ 是否有非零解的问题。设矩阵 $A = [\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_k]$，则：

| 线性相关性 | 等价的方程组表述 |
|---|---|
| 线性相关 | $A\mathbf{x}=\mathbf{0}$ 有非零解（自由变量存在） |
| 线性无关 | $A\mathbf{x}=\mathbf{0}$ 只有零解（列满秩） |

这一联系非常重要：**判断线性相关性，就是判断齐次方程组解的唯一性**。

进一步，向量 $\mathbf{b}$ 是否在 $\text{Span}\{\mathbf{v}_1,\ldots,\mathbf{v}_k\}$ 中，等价于**非齐次方程组** $A\mathbf{x}=\mathbf{b}$ 是否有解：

- 有解 $\Rightarrow$ $\mathbf{b} \in \text{Span}\{\mathbf{v}_1,\ldots,\mathbf{v}_k\}$
- 无解 $\Rightarrow$ $\mathbf{b} \notin \text{Span}\{\mathbf{v}_1,\ldots,\mathbf{v}_k\}$

**综合示例** 判断 $\mathbf{b} = (1, 2, 3)^T$ 是否在以下两个向量的张成中：

$$\mathbf{v}_1 = \begin{pmatrix}1\\1\\0\end{pmatrix}, \quad \mathbf{v}_2 = \begin{pmatrix}0\\1\\2\end{pmatrix}$$

对增广矩阵 $[A|\mathbf{b}]$ 行化简：

$$\begin{pmatrix}1&0&1\\1&1&2\\0&2&3\end{pmatrix} \rightarrow \begin{pmatrix}1&0&1\\0&1&1\\0&2&3\end{pmatrix} \rightarrow \begin{pmatrix}1&0&1\\0&1&1\\0&0&1\end{pmatrix}$$

最后一行 $0=1$，方程组无解，故 $\mathbf{b} \notin \text{Span}\{\mathbf{v}_1, \mathbf{v}_2\}$。

---

## 本章小结

本章从线性组合出发，建立了线性代数中最核心的概念之一——线性相关性。

**核心概念回顾：**

1. **线性组合**：$c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k$，是向量按比例缩放后相加的结果。

2. **线性无关**：$c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$ 只有零解，每个向量提供独立的方向。

3. **线性相关**：存在非零系数使上式成立，至少有一个向量是"冗余的"，可被其余向量线性表示。

4. **张成（Span）**：所有线性组合构成的集合，是一个子空间，描述了向量组能"覆盖"的空间范围。

5. **判定方法**：
   - 行化简：看列是否全有主元
   - 行列式（方阵）：$\det \neq 0$ 则线性无关
   - 齐次方程组：只有零解则线性无关

**直觉总结：** 线性无关的向量组是"高效的"——没有冗余信息，每个向量都在扩大张成空间。线性相关的向量组是"冗余的"——可以去掉若干向量而不缩小张成空间。

---

## 深度学习应用

线性相关性的思想在深度学习中无处不在，以下三个方向尤为重要。

### 过参数化网络中的冗余

现代深度神经网络（如大型 Transformer 模型）往往拥有远超数据复杂度所需的参数量，称为**过参数化（Overparameterization）**。

从线性代数角度理解：网络某一层的权重矩阵 $W \in \mathbb{R}^{m \times n}$，如果其列向量（即各个神经元的权重向量）线性相关，则该层的有效表示能力（秩）低于 $\min(m,n)$。这意味着：

- 若 $\text{rank}(W) = r \ll \min(m,n)$，大量神经元是线性相关的，实际上只有 $r$ 个"独立神经元"在工作。
- **低秩分解（Low-Rank Decomposition）** 技术（如 LoRA）正是利用这一点，将 $W$ 近似为 $W \approx AB$（$A \in \mathbb{R}^{m \times r}$，$B \in \mathbb{R}^{r \times n}$），大幅减少参数量。

### 特征冗余检测

在特征工程中，若输入特征矩阵的列（各个特征）线性相关，会导致：

- 模型参数估计不稳定（共线性问题）
- 训练效率下降（梯度包含冗余信息）
- 模型难以区分各特征的独立贡献

检测特征冗余的方法包括：计算特征矩阵的**条件数**（Condition Number）和**方差膨胀因子（VIF）**，它们本质上都在衡量特征矩阵的列接近线性相关的程度。

### 正则化与特征选择

**L1 正则化（Lasso）** 会促使权重向量稀疏，相当于自动完成特征选择——把"冗余特征"（那些可以被其他特征线性表示的特征）的权重压缩为零，保留线性无关的特征子集。

从 Span 的角度理解：Lasso 在寻找一个尽可能小的线性无关特征子集，使其张成空间足以近似描述目标变量。

### 代码示例

以下 Python 代码演示如何通过矩阵秩和 SVD 检测特征矩阵的线性相关性：

```python
import numpy as np

# 构造特征矩阵：前两个特征线性无关，第三个是前两个的线性组合
np.random.seed(42)
v1 = np.array([1.0, 2.0, 3.0, 4.0])
v2 = np.array([2.0, 1.0, 0.0, -1.0])
v3 = 2 * v1 - v2  # 线性相关！

X = np.column_stack([v1, v2, v3])
print("特征矩阵 X:\n", X)

# 方法1：计算矩阵的秩
rank = np.linalg.matrix_rank(X)
print(f"\n矩阵的秩: {rank}")
print(f"特征数量: {X.shape[1]}")
print(f"结论: {'存在线性相关' if rank < X.shape[1] else '线性无关'}")

# 方法2：奇异值分解（SVD），近零奇异值揭示线性相关
U, s, Vt = np.linalg.svd(X)
print(f"\n奇异值: {s}")
print(f"接近零的奇异值数量（阈值1e-10）: {np.sum(s < 1e-10)}")

# 方法3：检测可以用其他特征表示的特征（相关系数矩阵）
corr_matrix = np.corrcoef(X.T)
print(f"\n特征相关系数矩阵:\n{np.round(corr_matrix, 3)}")

# 方法4：计算条件数——数值越大，越接近线性相关
cond_number = np.linalg.cond(X)
print(f"\n矩阵条件数: {cond_number:.2e}")
print("（条件数 > 1e10 通常表示存在严重的近似线性相关）")

# 方法5：用 PCA 去除冗余，保留线性无关的主成分
from numpy.linalg import eig
cov = np.cov(X.T)
eigenvalues, eigenvectors = eig(cov)
print(f"\n协方差矩阵特征值: {np.round(eigenvalues, 6)}")
print("（接近零的特征值对应冗余方向）")
```

**输出分析**：
- 秩为 2（不是 3），直接说明存在线性相关
- 第三个奇异值接近机器精度零
- 条件数极大，表明矩阵接近奇异

---

## 练习题

**练习1**（基础）

判断以下向量是否线性相关，并说明理由：

$$\mathbf{u} = \begin{pmatrix}2\\4\\-2\end{pmatrix}, \quad \mathbf{v} = \begin{pmatrix}-1\\-2\\1\end{pmatrix}$$

**练习2**（计算）

用行化简法判断以下向量组的线性相关性：

$$\mathbf{v}_1 = \begin{pmatrix}1\\-2\\3\end{pmatrix}, \quad \mathbf{v}_2 = \begin{pmatrix}2\\1\\-1\end{pmatrix}, \quad \mathbf{v}_3 = \begin{pmatrix}4\\-3\\5\end{pmatrix}$$

**练习3**（理解）

设 $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3 \in \mathbb{R}^4$ 线性无关。问向量组 $\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3, \mathbf{v}_1 + \mathbf{v}_2 - \mathbf{v}_3\}$ 是否线性相关？请证明你的结论。

**练习4**（张成）

设

$$\mathbf{a}_1 = \begin{pmatrix}1\\0\\-1\end{pmatrix}, \quad \mathbf{a}_2 = \begin{pmatrix}2\\1\\0\end{pmatrix}$$

向量 $\mathbf{b} = \begin{pmatrix}3\\1\\-1\end{pmatrix}$ 是否在 $\text{Span}\{\mathbf{a}_1, \mathbf{a}_2\}$ 中？若是，写出具体的线性组合表达式。

**练习5**（深度学习关联）

某神经网络一层的权重矩阵为

$$W = \begin{pmatrix}1&2&3\\2&4&6\\1&2&3\end{pmatrix}$$

（1）计算 $\text{rank}(W)$。
（2）这意味着该层神经元存在什么问题？
（3）若将 $W$ 视为特征变换矩阵 $\mathbf{y} = W\mathbf{x}$，输出向量 $\mathbf{y}$ 实际上处于 $\mathbb{R}^3$ 的哪个子空间？

---

## 练习答案

<details>
<summary>点击展开 练习 1 答案</summary>

观察到 $\mathbf{u} = -2\mathbf{v}$（验证：$-2 \times (-1,-2,1)^T = (2,4,-2)^T = \mathbf{u}$），即两向量成比例，因此**线性相关**。

等价地，设 $c_1\mathbf{u} + c_2\mathbf{v} = \mathbf{0}$，取 $c_1=1, c_2=2$ 即得非零解：$\mathbf{u} + 2\mathbf{v} = (2-2, 4-4, -2+2)^T = \mathbf{0}$。

</details>

<details>
<summary>点击展开 练习 2 答案</summary>

构造矩阵 $A = [\mathbf{v}_1 | \mathbf{v}_2 | \mathbf{v}_3]$ 并行化简：

$$\begin{pmatrix}1&2&4\\-2&1&-3\\3&-1&5\end{pmatrix}$$

$R_2 \leftarrow R_2 + 2R_1$，$R_3 \leftarrow R_3 - 3R_1$：

$$\begin{pmatrix}1&2&4\\0&5&5\\0&-7&-7\end{pmatrix}$$

$R_2 \leftarrow \frac{1}{5}R_2$，$R_3 \leftarrow R_3 + 7R_2$（经化简后 $R_3$）：

$$R_3 \leftarrow R_3 + \frac{7}{5}R_2: \quad \begin{pmatrix}1&2&4\\0&5&5\\0&0&0\end{pmatrix}$$

第三列没有主元（是自由列），因此向量组**线性相关**。

验证：$\mathbf{v}_3 = 2\mathbf{v}_1 + \mathbf{v}_2$（可验算：$2(1,-2,3)^T + (2,1,-1)^T = (4,-3,5)^T = \mathbf{v}_3$，正确）。

</details>

<details>
<summary>点击展开 练习 3 答案</summary>

令第四个向量为 $\mathbf{v}_4 = \mathbf{v}_1 + \mathbf{v}_2 - \mathbf{v}_3$。

考虑方程 $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + c_3\mathbf{v}_3 + c_4\mathbf{v}_4 = \mathbf{0}$，代入 $\mathbf{v}_4$ 的表达式：

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + c_3\mathbf{v}_3 + c_4(\mathbf{v}_1 + \mathbf{v}_2 - \mathbf{v}_3) = \mathbf{0}$$

$$(c_1+c_4)\mathbf{v}_1 + (c_2+c_4)\mathbf{v}_2 + (c_3-c_4)\mathbf{v}_3 = \mathbf{0}$$

由于 $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$ 线性无关，各系数必须为零：

$$c_1 + c_4 = 0, \quad c_2 + c_4 = 0, \quad c_3 - c_4 = 0$$

取 $c_4 = 1$，得 $c_1 = -1, c_2 = -1, c_3 = 1$，这是一个非零解。

因此 $\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3, \mathbf{v}_4\}$ **线性相关**。（直观原因：$\mathbf{v}_4$ 已经在 $\text{Span}\{\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3\}$ 中。）

</details>

<details>
<summary>点击展开 练习 4 答案</summary>

设 $\mathbf{b} = x_1\mathbf{a}_1 + x_2\mathbf{a}_2$，即求解方程组：

$$x_1 + 2x_2 = 3, \quad x_2 = 1, \quad -x_1 = -1$$

由第三个方程得 $x_1 = 1$，第二个方程得 $x_2 = 1$，代入第一个方程：$1 + 2 = 3$，成立。

因此 $\mathbf{b} \in \text{Span}\{\mathbf{a}_1, \mathbf{a}_2\}$，具体表达式为：

$$\mathbf{b} = 1 \cdot \mathbf{a}_1 + 1 \cdot \mathbf{a}_2 = \begin{pmatrix}1\\0\\-1\end{pmatrix} + \begin{pmatrix}2\\1\\0\end{pmatrix} = \begin{pmatrix}3\\1\\-1\end{pmatrix} \checkmark$$

</details>

<details>
<summary>点击展开 练习 5 答案</summary>

**(1)** 观察到矩阵 $W$ 的三行完全相同（行1=行3，行2=2×行1），故

$$\text{rank}(W) = 1$$

（矩阵只有一个线性无关的行/列方向。）

**(2)** 该层所有神经元的权重向量线性相关（实际上只有一个独立方向）。这意味着无论输入 $\mathbf{x}$ 是什么，三个输出神经元的激活值总满足 $y_1 = y_2/2 = y_3$（成比例），即存在严重冗余——三个神经元实际上在做同一件事，浪费了 $2/3$ 的参数和计算资源。

**(3)** 输出 $\mathbf{y} = W\mathbf{x}$ 的形式为：

$$\mathbf{y} = \begin{pmatrix}x_1+2x_2+3x_3\\2(x_1+2x_2+3x_3)\\x_1+2x_2+3x_3\end{pmatrix} = (x_1+2x_2+3x_3)\begin{pmatrix}1\\2\\1\end{pmatrix}$$

因此 $\mathbf{y}$ 总是向量 $(1,2,1)^T$ 的标量倍，处于由 $(1,2,1)^T$ 张成的**一维子空间（直线）**中，而非整个 $\mathbb{R}^3$。这一层的表示能力极度受限。

</details>

---

## 抽象成方法（套路总结）

### 核心等价表述速查

| 结论 | 等价条件 |
|---|---|
| $\mathbf{v}_1,\ldots,\mathbf{v}_k$ **线性无关** | $c_1\mathbf{v}_1+\cdots+c_k\mathbf{v}_k=\mathbf{0}$ 只有零解 |
| 线性无关 | 矩阵 $A=[\mathbf{v}_1\vert\cdots\vert\mathbf{v}_k]$ 的 $\text{rank}(A)=k$（列满秩） |
| 线性无关 | $A\mathbf{x}=\mathbf{0}$ 只有零解（无自由变量） |
| 线性无关（方阵） | $\det(A)\neq 0$ |
| $\mathbf{v}_1,\ldots,\mathbf{v}_k$ **线性相关** | 存在不全为零的系数使组合为零 |
| 线性相关 | 至少一个向量在其余向量的张成中 |
| 线性相关 | $\text{rank}(A) < k$（存在自由列） |

### 判定线性相关性：标准 3 步

1. **构造矩阵** $A = [\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_k]$（向量排列为列）
2. **行化简**得到 REF；**数主元数量** = $\text{rank}(A)$
3. **判断**：$\text{rank}(A) = k$ → 线性无关；$\text{rank}(A) < k$ → 线性相关

**特殊情形快速判定**：
- 含零向量 → 必线性相关
- 两向量成比例 → 必线性相关
- $k > n$（向量个数 $>$ 维数）→ 必线性相关
- 方阵时 $\det \neq 0$ → 线性无关（最快判断）

### 判断 $\mathbf{b} \in \text{Span}\{\mathbf{v}_1,\ldots,\mathbf{v}_k\}$：2 步

1. 对增广矩阵 $[A \mid \mathbf{b}]$ 行化简
2. 若无矛盾行（如 $0 = c,\, c \neq 0$）→ 有解 → $\mathbf{b}$ 在 Span 中；否则不在

---

## 方法变形

### 变形 1：$k > n$ 时的自动线性相关

$\mathbb{R}^n$ 中任意 $n+1$ 个或更多向量必线性相关。不需要行化简，直接给出结论。

### 变形 2：从线性相关关系找具体表示

RREF 中自由列对应的变量置为 1，主元列回代，即可得到各向量之间的具体线性组合关系。

### 变形 3：判断能否扩张成基

若 $\{\mathbf{v}_1,\ldots,\mathbf{v}_k\}$ 线性无关且 $k < n$，则可以找到额外的向量（通常从标准基中选）扩充为 $\mathbb{R}^n$ 的一个基。

### 变形 4：Wronskian 行列式（函数线性相关性）

对函数 $f_1(x), f_2(x), \ldots, f_n(x)$，其 Wronskian 行列式为：

$$W(x) = \det\begin{pmatrix}f_1 & f_2 & \cdots & f_n \\ f_1' & f_2' & \cdots & f_n' \\ \vdots & & & \vdots \\ f_1^{(n-1)} & \cdots & & f_n^{(n-1)}\end{pmatrix}$$

若存在某点 $x_0$ 使 $W(x_0) \neq 0$，则这些函数线性无关。例如 $\{1, \sin x, \cos x\}$ 在函数空间中线性无关。

---

## 思考路标（条件反射）

1. 看到"判断线性相关" → 构造矩阵 → 行化简 → 看秩是否等于列数
2. 看到"含零向量" → 立刻说"线性相关"，无需计算
3. 看到"两向量平行（成比例）" → 立刻说"线性相关"
4. 看到"$k > n$" → 立刻说"线性相关"（$\mathbb{R}^n$ 维数界）
5. 看到"方阵判断" → 算行列式，非零则无关
6. 看到"$\mathbf{b}$ 是否在 Span 中" → 增广矩阵行化简，看有无矛盾行
7. 看到"$A\mathbf{x}=\mathbf{0}$ 只有零解" → 列线性无关 $\Leftrightarrow$ 列满秩
8. 看到"特征冗余" → 特征矩阵某列是其余列的线性组合 → 该特征可删去
9. 看到"权重矩阵低秩" → 神经元存在线性相关，存在冗余
10. 看到"$\text{rank}(A) = k$" → 向量组线性无关；$< k$ 则相关

---

## 易错点

1. **判断"只有零解"时漏写验证**：零解一定存在（$c_i = 0$ 全部成立），判断线性无关要证明**没有非零解**，即行化简后无自由列。

2. **线性相关 $\neq$ 所有向量都能被其余表示**：线性相关只要求**至少一个**向量能被其余表示，不一定是每一个。例如 $\{(1,0)^T, (2,0)^T, (0,1)^T\}$ 线性相关，但 $(0,1)^T$ 不能被前两个向量线性表示（前两个张成 $x$ 轴）。

3. **行化简后看列的主元**：行化简后**主元所在的列**（而非行！）决定线性独立方向。第 $j$ 列是自由列意味着 $\mathbf{v}_j$ 可被之前的主元列表示。

4. **$\text{Span}$ 的维数 $\neq$ 向量个数**：$\text{Span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ 的维数等于 $\text{rank}([\mathbf{v}_1\vert\cdots\vert\mathbf{v}_k])$，而非 $k$（当向量线性相关时两者不同）。

5. **函数线性相关与向量线性相关混淆**：函数空间中的线性相关意味着某函数对所有 $x$ 都等于其余函数的线性组合，而非某点处相等。Wronskian 行列式在某点非零则线性无关，但为零不一定相关（反例存在）。

---

## 典型应用例题

### 例 1：行化简判断线性相关

> **题目**：判断 $\mathbf{v}_1=(1,2,1)^T$，$\mathbf{v}_2=(2,5,3)^T$，$\mathbf{v}_3=(0,1,1)^T$ 的线性相关性。

【思路】构造矩阵，行化简，数主元。

【解】

$$A = \begin{pmatrix}1&2&0\\2&5&1\\1&3&1\end{pmatrix}$$

$R_2 \leftarrow R_2 - 2R_1$，$R_3 \leftarrow R_3 - R_1$：

$$\begin{pmatrix}1&2&0\\0&1&1\\0&1&1\end{pmatrix}$$

$R_3 \leftarrow R_3 - R_2$：

$$\begin{pmatrix}1&2&0\\0&1&1\\0&0&0\end{pmatrix}$$

$\text{rank}(A) = 2 < 3$（列数），故**线性相关**。

自由列为第 3 列：令 $c_3 = 1$，回代得 $c_2 = -1$，$c_1 = 2$，即 $2\mathbf{v}_1 - \mathbf{v}_2 + \mathbf{v}_3 = \mathbf{0}$，即 $\mathbf{v}_3 = \mathbf{v}_2 - 2\mathbf{v}_1$。

【答案】$\boxed{\text{线性相关}；\mathbf{v}_3 = \mathbf{v}_2 - 2\mathbf{v}_1}$

### 例 2：行列式快速判断

> **题目**：判断 $\mathbf{a}_1=(1,0,2)^T$，$\mathbf{a}_2=(0,1,-1)^T$，$\mathbf{a}_3=(3,2,4)^T$ 是否线性无关。

【思路】三维三向量，算行列式。

【解】

$$\det\begin{pmatrix}1&0&3\\0&1&2\\2&-1&4\end{pmatrix} = 1\cdot(4-(-2)) - 0 + 3\cdot(0-2) = 6 - 6 = 0$$

行列式为 0，故**线性相关**。

【注】$\mathbf{a}_3 = 3\mathbf{a}_1 + 2\mathbf{a}_2$（可验证：$(3,0,6)^T + (0,2,-2)^T = (3,2,4)^T$）。

【答案】$\boxed{\text{线性相关}；行列式为 0}$

### 例 3：判断向量是否在张成空间中

> **题目**：$\mathbf{u}_1=(1,1,0)^T$，$\mathbf{u}_2=(2,0,1)^T$。向量 $\mathbf{b}=(5,1,2)^T$ 是否在 $\text{Span}\{\mathbf{u}_1,\mathbf{u}_2\}$ 中？若是，写出线性组合。

【思路】增广矩阵行化简。

【解】

$$[A\mid\mathbf{b}] = \begin{pmatrix}1&2&5\\1&0&1\\0&1&2\end{pmatrix}$$

$R_2 \leftarrow R_2 - R_1$：

$$\begin{pmatrix}1&2&5\\0&-2&-4\\0&1&2\end{pmatrix}$$

$R_2 \leftarrow -\frac{1}{2}R_2$，$R_3 \leftarrow R_3 - R_2$：

$$\begin{pmatrix}1&2&5\\0&1&2\\0&0&0\end{pmatrix}$$

无矛盾行，方程组有解。回代：$c_2 = 2$，$c_1 = 5 - 2\cdot 2 = 1$。

验证：$1\cdot(1,1,0)^T + 2\cdot(2,0,1)^T = (1,1,0)^T + (4,0,2)^T = (5,1,2)^T = \mathbf{b}$。

【答案】$\mathbf{b} \in \text{Span}\{\mathbf{u}_1,\mathbf{u}_2\}$；$\mathbf{b} = \mathbf{u}_1 + 2\mathbf{u}_2$。

---

## 自测题

**自测 1**　判断 $\{(1,2)^T, (-2,-4)^T\}$ 的线性相关性。若线性相关，写出具体关系。

> 提示：第二向量是第一向量的 $-2$ 倍，成比例，线性相关。$1\cdot(1,2)^T + \frac{1}{2}\cdot(-2,-4)^T = \mathbf{0}$，或 $(-2,-4)^T = -2(1,2)^T$。

**自测 2**　$\mathbb{R}^3$ 中有 4 个非零向量 $\mathbf{w}_1,\mathbf{w}_2,\mathbf{w}_3,\mathbf{w}_4$，无需计算直接判断其线性相关性，说明理由。

> 提示：向量个数 $4 > 3 = \dim(\mathbb{R}^3)$，由"维数界"（性质3）直接结论：必线性相关，无需任何计算。

**自测 3**　设 $\mathbf{a}=(1,0,0)^T$，$\mathbf{b}=(0,1,0)^T$，$\mathbf{c}=(1,1,0)^T$。问 $\mathbf{d}=(0,0,1)^T$ 是否在 $\text{Span}\{\mathbf{a},\mathbf{b},\mathbf{c}\}$ 中？

> 提示：$\text{Span}\{\mathbf{a},\mathbf{b},\mathbf{c}\} = \text{Span}\{\mathbf{a},\mathbf{b}\}$（$\mathbf{c}=\mathbf{a}+\mathbf{b}$），是 $xy$ 平面。$\mathbf{d}$ 有 $z$ 分量，不在 $xy$ 平面中，故 $\mathbf{d}\notin\text{Span}$。

**自测 4**　某神经网络权重矩阵行化简后有 3 个主元（共 5 列）。该层有效独立方向几个？冗余维度几个？

> 提示：有效独立方向 = $\text{rank} = 3$；冗余维度 = $5 - 3 = 2$（零空间维数，由秩-零化度定理）。3 个线性独立的方向"真正在工作"，另外 2 个方向是线性相关的冗余。

**自测 5**　证明：若 $\mathbf{v}_1,\mathbf{v}_2$ 线性无关，则 $\mathbf{v}_1+\mathbf{v}_2$ 和 $\mathbf{v}_1-\mathbf{v}_2$ 也线性无关。

> 提示：设 $a(\mathbf{v}_1+\mathbf{v}_2) + b(\mathbf{v}_1-\mathbf{v}_2) = \mathbf{0}$，整理得 $(a+b)\mathbf{v}_1 + (a-b)\mathbf{v}_2 = \mathbf{0}$。由 $\mathbf{v}_1,\mathbf{v}_2$ 线性无关，得 $a+b=0$ 且 $a-b=0$，解得 $a=b=0$，故 $\mathbf{v}_1\pm\mathbf{v}_2$ 线性无关。

---

**回头看一眼"一例速记"**：

> 线性无关 = 齐次方程只有零解 = 列满秩；线性相关 = 存在自由列 = 某向量在其余向量张成中。
> 判定工具：行化简数主元；方阵用行列式；$k>n$ 直接相关。
> $\mathbf{b}\in\text{Span}$：增广矩阵行化简，无矛盾行则在。

如果现在不看笔记，能独立完成例 1 + 自测 5——本章，你拿下了。

---

## 融合版说明

| 段落 | 来源 | 价值 |
|---|---|---|
| 一例速记 + 引入 + 思维路径还原 | 重写版（前置） | 建立直觉 / 条件反射 |
| 学习目标 + 10.1–10.4 严格正文 | 原版 | 完整定义与推导 |
| 本章小结 | 原版 | 核心概念速查 |
| 深度学习应用 + 代码 | 原版 | 工业实战关联 |
| 练习题 + 详解 | 原版 | 系统巩固 |
| 抽象成方法 + 方法变形 | 重写版（后置） | 套路固化 |
| 思考路标 + 易错点 | 融合两版 | 条件反射 + 避雷 |
| 典型应用例题 3 例 | 重写版 | 演练精讲 |
| 自测题 5 题 | 重写版 | 额外验收 |
