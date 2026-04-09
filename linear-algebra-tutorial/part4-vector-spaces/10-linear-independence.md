# 第10章：线性相关与线性无关

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
