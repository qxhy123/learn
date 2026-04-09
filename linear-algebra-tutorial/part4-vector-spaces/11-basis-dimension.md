# 第11章：基与维数

> 基是向量空间的"坐标系骨架"，维数是空间"自由度"的精确度量。掌握这两个概念，是理解线性代数深层结构的关键一步。

---

## 学习目标

完成本章学习后，你将能够：

- 理解基（Basis）的严格定义：线性无关性与张成性的统一
- 掌握如何验证一组向量是否构成基，以及如何求向量空间的基
- 理解维数（Dimension）的概念及其与基的关系
- 计算向量在给定基下的坐标，理解坐标变换的本质
- 了解基与维数在主成分分析（PCA）中的核心作用

---

## 11.1 基的定义

### 11.1.1 什么是基

直觉上，"基"就是描述一个向量空间所需的最小信息集合。形式化地说：

**定义（基）：** 设 $V$ 是域 $F$ 上的向量空间，$V$ 的一个**基**（Basis）是满足以下两个条件的向量组 $\mathcal{B} = \{v_1, v_2, \ldots, v_n\}$：

1. **线性无关**：$v_1, v_2, \ldots, v_n$ 线性无关，即方程
   $$c_1 v_1 + c_2 v_2 + \cdots + c_n v_n = 0$$
   只有零解 $c_1 = c_2 = \cdots = c_n = 0$。

2. **张成空间**：$V = \text{span}\{v_1, v_2, \ldots, v_n\}$，即 $V$ 中每个向量都可以写成 $v_1, \ldots, v_n$ 的线性组合。

两个条件缺一不可：只有张成而无线性无关，意味着存在"冗余"向量；只有线性无关而不张成，意味着描述不够完整。

### 11.1.2 标准基

最常用的基是**标准基**（Standard Basis），也称**自然基**。

**$\mathbb{R}^n$ 的标准基：**

$$e_1 = \begin{pmatrix}1\\0\\\vdots\\0\end{pmatrix}, \quad e_2 = \begin{pmatrix}0\\1\\\vdots\\0\end{pmatrix}, \quad \ldots, \quad e_n = \begin{pmatrix}0\\0\\\vdots\\1\end{pmatrix}$$

**验证：**

- 线性无关：若 $c_1 e_1 + c_2 e_2 + \cdots + c_n e_n = 0$，则逐分量比较得 $c_1 = c_2 = \cdots = c_n = 0$。
- 张成：任意 $x = (x_1, x_2, \ldots, x_n)^T = x_1 e_1 + x_2 e_2 + \cdots + x_n e_n$。

**其他空间的标准基：**

- $\mathbb{R}^{2 \times 2}$（$2\times2$ 矩阵空间）的标准基：
  $$E_{11} = \begin{pmatrix}1&0\\0&0\end{pmatrix}, \quad E_{12} = \begin{pmatrix}0&1\\0&0\end{pmatrix}, \quad E_{21} = \begin{pmatrix}0&0\\1&0\end{pmatrix}, \quad E_{22} = \begin{pmatrix}0&0\\0&1\end{pmatrix}$$

- $P_n$（次数不超过 $n$ 的多项式空间）的标准基：$\{1, x, x^2, \ldots, x^n\}$

### 11.1.3 基的非唯一性

一个向量空间的基并不唯一。以 $\mathbb{R}^2$ 为例，以下都是合法的基：

$$\mathcal{B}_1 = \left\{\begin{pmatrix}1\\0\end{pmatrix}, \begin{pmatrix}0\\1\end{pmatrix}\right\}, \quad \mathcal{B}_2 = \left\{\begin{pmatrix}1\\1\end{pmatrix}, \begin{pmatrix}1\\-1\end{pmatrix}\right\}, \quad \mathcal{B}_3 = \left\{\begin{pmatrix}2\\0\end{pmatrix}, \begin{pmatrix}0\\3\end{pmatrix}\right\}$$

**关键事实**：虽然基不唯一，但同一有限维向量空间的所有基包含**相同数量**的向量。这个数量就是空间的维数。

---

## 11.2 维数

### 11.2.1 维数的定义

**定义（维数）：** 向量空间 $V$ 的**维数**（Dimension），记作 $\dim(V)$，定义为 $V$ 的任意一个基中向量的个数。

若 $V$ 只含零向量，则 $\dim(V) = 0$（空集是零空间的基）。

若 $V$ 的基包含无穷多个向量，则称 $V$ 是**无穷维**空间（如函数空间）。

本章只讨论有限维情形。

### 11.2.2 维数定理与基本性质

**定理（基的等势性）：** 有限维向量空间 $V$ 的任意两个基包含相同数量的向量。

这保证了维数定义是良好的。

**推论与常用性质：**

设 $\dim(V) = n$，则：

1. $V$ 中任意 $n+1$ 个向量必线性相关。
2. $V$ 中任意 $n$ 个线性无关的向量构成 $V$ 的一个基。
3. $V$ 中任意张成 $V$ 的 $n$ 个向量构成 $V$ 的一个基。
4. $V$ 的任意线性无关子集都可以**扩充**为 $V$ 的一个基。

**子空间的维数不等式：** 若 $W$ 是 $V$ 的子空间，则 $\dim(W) \leq \dim(V)$，等号成立当且仅当 $W = V$。

### 11.2.3 常见空间的维数

| 向量空间 | 维数 |
|---|---|
| $\mathbb{R}^n$ | $n$ |
| $\mathbb{C}^n$（作为实向量空间） | $2n$ |
| $\mathbb{R}^{m \times n}$（$m \times n$ 实矩阵） | $mn$ |
| $P_n$（次数 $\leq n$ 的实多项式） | $n+1$ |
| $\{0\}$（零空间） | $0$ |

**秩-零化度定理（Rank-Nullity Theorem）：** 设 $A$ 是 $m \times n$ 矩阵，$T: \mathbb{R}^n \to \mathbb{R}^m$ 是对应的线性变换，则：

$$\dim(\text{Col}(A)) + \dim(\text{Null}(A)) = n$$

即**秩 + 零化度 = 列数**。这是线性代数中最重要的维数关系之一。

---

## 11.3 求基的方法

### 11.3.1 通过行化简求基

**方法：将矩阵行化简为行阶梯形（REF）或行最简形（RREF），主元列对应原矩阵的列构成列空间的基。**

**例 11.1：** 求矩阵 $A$ 的列空间的基：

$$A = \begin{pmatrix}1 & 2 & 3 & 0\\2 & 4 & 7 & -1\\3 & 6 & 10 & -1\end{pmatrix}$$

**解：** 对 $A$ 行化简：

$$A \xrightarrow{R_2 - 2R_1} \begin{pmatrix}1 & 2 & 3 & 0\\0 & 0 & 1 & -1\\3 & 6 & 10 & -1\end{pmatrix} \xrightarrow{R_3 - 3R_1} \begin{pmatrix}1 & 2 & 3 & 0\\0 & 0 & 1 & -1\\0 & 0 & 1 & -1\end{pmatrix} \xrightarrow{R_3 - R_2} \begin{pmatrix}1 & 2 & 3 & 0\\0 & 0 & 1 & -1\\0 & 0 & 0 & 0\end{pmatrix}$$

主元在第 1、3 列，故 $A$ 的列空间的基为原矩阵 $A$ 的第 1、3 列：

$$\mathcal{B} = \left\{\begin{pmatrix}1\\2\\3\end{pmatrix}, \begin{pmatrix}3\\7\\10\end{pmatrix}\right\}$$

$\text{rank}(A) = 2$，$\dim(\text{Col}(A)) = 2$。

**求行空间的基：** 行化简后，非零行构成行空间的基。

**求零空间（核）的基：** 解方程 $Ax = 0$，自由变量各取 1 其余取 0，得到零空间的一组基向量。

### 11.3.2 扩充为基

**问题：** 给定 $V$ 中的线性无关集 $S$，将 $S$ 扩充为 $V$ 的一个基。

**方法：**

1. 从 $V$ 的标准基出发，将 $S$ 中的向量与标准基向量合并。
2. 用行化简删去线性相关的向量，保留主元位置对应的向量。

**例 11.2：** 在 $\mathbb{R}^3$ 中，将 $\{v_1\} = \left\{\begin{pmatrix}1\\1\\0\end{pmatrix}\right\}$ 扩充为一个基。

**解：** 构造矩阵（列为 $v_1, e_1, e_2, e_3$）并行化简，选出 3 个主元列：

$$\begin{pmatrix}1&1&0&0\\1&0&1&0\\0&0&0&1\end{pmatrix} \xrightarrow{\text{行化简}} \begin{pmatrix}1&1&0&0\\0&-1&1&0\\0&0&0&1\end{pmatrix}$$

主元在第 1、2、4 列，对应向量 $v_1, e_1, e_3$。

扩充后的基：

$$\mathcal{B} = \left\{\begin{pmatrix}1\\1\\0\end{pmatrix}, \begin{pmatrix}1\\0\\0\end{pmatrix}, \begin{pmatrix}0\\0\\1\end{pmatrix}\right\}$$

---

## 11.4 坐标

### 11.4.1 在给定基下的坐标

**定理（坐标唯一性）：** 设 $\mathcal{B} = \{b_1, b_2, \ldots, b_n\}$ 是向量空间 $V$ 的一个基，则 $V$ 中每个向量 $x$ 都能**唯一**地表示为：

$$x = c_1 b_1 + c_2 b_2 + \cdots + c_n b_n$$

**定义（坐标）：** 上述表示中的系数 $c_1, c_2, \ldots, c_n$ 称为 $x$ 在基 $\mathcal{B}$ 下的**坐标**（Coordinates），列向量 $[x]_\mathcal{B} = (c_1, c_2, \ldots, c_n)^T$ 称为 $x$ 的**坐标向量**。

### 11.4.2 计算坐标

**方法：** 求解线性方程组 $[b_1 \mid b_2 \mid \cdots \mid b_n] \cdot c = x$，解 $c$ 即为坐标向量。

**例 11.3：** 设基 $\mathcal{B} = \left\{b_1 = \begin{pmatrix}1\\2\end{pmatrix}, b_2 = \begin{pmatrix}3\\5\end{pmatrix}\right\}$，求 $x = \begin{pmatrix}7\\11\end{pmatrix}$ 在 $\mathcal{B}$ 下的坐标。

**解：** 解方程组：

$$\begin{pmatrix}1&3\\2&5\end{pmatrix}\begin{pmatrix}c_1\\c_2\end{pmatrix} = \begin{pmatrix}7\\11\end{pmatrix}$$

增广矩阵行化简：

$$\begin{pmatrix}1&3&7\\2&5&11\end{pmatrix} \xrightarrow{R_2-2R_1} \begin{pmatrix}1&3&7\\0&-1&-3\end{pmatrix} \xrightarrow{-R_2} \begin{pmatrix}1&3&7\\0&1&3\end{pmatrix} \xrightarrow{R_1-3R_2} \begin{pmatrix}1&0&-2\\0&1&3\end{pmatrix}$$

故 $c_1 = -2, c_2 = 3$，即 $[x]_\mathcal{B} = \begin{pmatrix}-2\\3\end{pmatrix}$。

**验证：** $-2 \begin{pmatrix}1\\2\end{pmatrix} + 3\begin{pmatrix}3\\5\end{pmatrix} = \begin{pmatrix}-2+9\\-4+15\end{pmatrix} = \begin{pmatrix}7\\11\end{pmatrix}$ ✓

### 11.4.3 坐标变换的几何意义

不同的基给同一个向量赋予不同的坐标，但向量本身不变——**坐标是描述工具，空间是客观存在**。

选择合适的基，可以使向量的坐标表示更简洁、计算更高效。这正是 PCA 等降维方法的核心思想。

---

## 本章小结

| 概念 | 定义 | 关键性质 |
|---|---|---|
| **基** | 线性无关且张成空间的向量组 | 非唯一，但基的大小（元素个数）唯一 |
| **维数** | 基中向量的个数 | $\dim(V) = n$ 意味着 $V \cong \mathbb{R}^n$ |
| **坐标** | 向量在给定基下的线性组合系数 | 给定基，坐标唯一确定 |
| **秩-零化度定理** | $\text{rank}(A) + \text{nullity}(A) = n$ | 连接列空间与零空间 |

**核心逻辑链：**

$$\text{线性无关} + \text{张成} \Rightarrow \text{基} \Rightarrow \text{维数（唯一整数）} \Rightarrow \text{坐标（唯一向量）}$$

**实用要点：**

- 求列空间的基：行化简，取主元列对应的**原矩阵列**
- 求行空间的基：行化简后取**非零行**
- 求零空间的基：解 $Ax = 0$，参数化后取基向量
- 验证 $n$ 个向量是 $\mathbb{R}^n$ 的基：行列式非零，或秩为 $n$

---

## 深度学习应用：PCA 降维的数学基础

### 背景：高维数据的"维数灾难"

在机器学习中，数据往往是高维的（如 $1000$ 维的词向量，$784$ 维的 MNIST 手写数字图像）。高维数据带来计算开销大、过拟合等问题。

**主成分分析（PCA, Principal Component Analysis）** 的核心任务，正是**寻找一组新的基**，使数据在这组基的前几个方向上的投影能最大程度地保留原始信息，从而实现降维。

### 数学原理

**步骤 1：中心化数据**

设数据矩阵 $X \in \mathbb{R}^{m \times n}$（$m$ 个样本，$n$ 个特征），先将每个特征减去均值，得到中心化矩阵 $\tilde{X}$。

**步骤 2：计算协方差矩阵**

$$C = \frac{1}{m-1} \tilde{X}^T \tilde{X} \in \mathbb{R}^{n \times n}$$

$C$ 是对称半正定矩阵，描述了各特征之间的相关性。

**步骤 3：特征分解**

对 $C$ 进行特征分解：

$$C = Q \Lambda Q^T$$

其中 $Q$ 的列（特征向量）是 $\mathbb{R}^n$ 的一个**正交基**，$\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ 按特征值从大到小排列。

**步骤 4：选取主成分**

选前 $k$ 个特征向量（对应最大的 $k$ 个特征值）构成新基 $P_k = [q_1, q_2, \ldots, q_k] \in \mathbb{R}^{n \times k}$。

**步骤 5：投影降维**

$$Z = \tilde{X} P_k \in \mathbb{R}^{m \times k}$$

$Z$ 是数据在新 $k$ 维子空间中的坐标表示。

**信息保留率：**

$$\text{explained variance ratio} = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^n \lambda_i}$$

### 线性代数视角

PCA 本质上是：**在 $\mathbb{R}^n$ 的所有 $k$ 维子空间中，找到使数据投影方差最大的那个子空间，并以其正交基作为新坐标系。**

- 特征值 $\lambda_i$ = 数据在 $q_i$ 方向上的方差
- 选前 $k$ 个特征向量 = 选最能"解释"数据变化的 $k$ 个基向量
- 坐标变换 $Z = \tilde{X} P_k$ = 将数据表示为新基下的坐标

### 代码示例：PCA 实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ---- 1. 加载数据（4 维 -> 2 维）----
data = load_iris()
X = data.data          # shape: (150, 4)
y = data.target

# ---- 2. 中心化 ----
X_mean = X.mean(axis=0)
X_centered = X - X_mean   # shape: (150, 4)

# ---- 3. 协方差矩阵 ----
C = np.cov(X_centered.T)   # shape: (4, 4)
print("协方差矩阵:\n", np.round(C, 3))

# ---- 4. 特征分解（特征值降序排列）----
eigenvalues, eigenvectors = np.linalg.eigh(C)
# eigh 保证实对称矩阵输出实数，升序排列，需要反转
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]   # 每列是一个特征向量（主成分方向）

print("\n特征值（方差贡献）:", np.round(eigenvalues, 4))
print("各主成分解释方差比:")
explained = eigenvalues / eigenvalues.sum()
for i, ev in enumerate(explained):
    print(f"  PC{i+1}: {ev*100:.2f}%")

# ---- 5. 选取前 2 个主成分并降维 ----
k = 2
P_k = eigenvectors[:, :k]   # shape: (4, 2) — 新基（列向量）
Z = X_centered @ P_k         # shape: (150, 2) — 新坐标

print(f"\n原始维数: {X.shape[1]}  ->  降维后: {Z.shape[1]}")
print(f"保留信息: {explained[:k].sum()*100:.2f}%")

# ---- 6. 可视化 ----
colors = ['navy', 'turquoise', 'darkorange']
labels = data.target_names
plt.figure(figsize=(7, 5))
for c, label, target in zip(colors, labels, [0, 1, 2]):
    mask = y == target
    plt.scatter(Z[mask, 0], Z[mask, 1], c=c, label=label, alpha=0.8)
plt.xlabel(f'PC1 ({explained[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({explained[1]*100:.1f}%)')
plt.title('PCA: Iris 数据集 4D -> 2D')
plt.legend()
plt.tight_layout()
plt.savefig('pca_iris.png', dpi=150)
plt.show()
```

**代码解读：**

- `np.linalg.eigh(C)`：对实对称矩阵求特征分解，得到正交特征向量矩阵（即新基）
- `eigenvectors[:, :k]`：取前 $k$ 列构成新基矩阵 $P_k$
- `X_centered @ P_k`：矩阵乘法实现坐标变换，每行是一个样本在新基下的坐标

在 Iris 数据集上，仅用 2 个主成分即可保留约 97.8% 的信息，同时实现 4→2 维的降维。

---

## 练习题

**练习 1（验证基）**

判断下列向量组是否构成 $\mathbb{R}^3$ 的基，并说明理由：

$$v_1 = \begin{pmatrix}1\\0\\-1\end{pmatrix}, \quad v_2 = \begin{pmatrix}2\\1\\0\end{pmatrix}, \quad v_3 = \begin{pmatrix}0\\1\\2\end{pmatrix}$$

**练习 2（求基与维数）**

设 $W = \left\{(x_1, x_2, x_3, x_4) \in \mathbb{R}^4 \mid x_1 - 2x_2 + x_3 = 0, \quad x_2 - x_4 = 0\right\}$，求 $W$ 的一个基和维数。

**练习 3（坐标计算）**

设 $\mathbb{R}^2$ 中的基 $\mathcal{B} = \left\{b_1 = \begin{pmatrix}2\\-1\end{pmatrix}, b_2 = \begin{pmatrix}-3\\4\end{pmatrix}\right\}$，求向量 $x = \begin{pmatrix}4\\-5\end{pmatrix}$ 在基 $\mathcal{B}$ 下的坐标向量 $[x]_\mathcal{B}$。

**练习 4（秩-零化度定理）**

设矩阵 $A = \begin{pmatrix}1&2&0&-1\\0&1&3&2\\1&3&3&1\end{pmatrix}$，利用秩-零化度定理，确定 $A$ 的零空间的维数，并求零空间的一个基。

**练习 5（思考题）**

设 $V$ 和 $W$ 都是 $\mathbb{R}^5$ 的子空间，且 $\dim(V) = 3$，$\dim(W) = 3$。

(a) $V \cap W$ 的维数可能是哪些值？

(b) $V + W$（即 $\{v + w \mid v \in V, w \in W\}$）的维数是多少？请利用维数公式 $\dim(V + W) = \dim(V) + \dim(W) - \dim(V \cap W)$ 给出答案范围。

---

## 练习答案

<details>
<summary>点击展开 练习 1 答案</summary>

构造矩阵 $A = [v_1 \mid v_2 \mid v_3]$，计算行列式：

$$\det(A) = \det\begin{pmatrix}1&2&0\\0&1&1\\-1&0&2\end{pmatrix}$$

按第一列展开：

$$= 1 \cdot \det\begin{pmatrix}1&1\\0&2\end{pmatrix} - 0 + (-1) \cdot (-1) \cdot \det\begin{pmatrix}2&0\\1&1\end{pmatrix}$$

$$= 1 \cdot (2 - 0) + 1 \cdot (2 - 0) = 2 + 2 = 4 \neq 0$$

行列式非零，故 $v_1, v_2, v_3$ 线性无关。又因为它们是 $\mathbb{R}^3$（维数为 3）中 3 个线性无关向量，由性质 3 知它们构成 $\mathbb{R}^3$ 的一个基。

</details>

<details>
<summary>点击展开 练习 2 答案</summary>

将约束条件整理：令自由变量为 $x_2, x_4$。

由 $x_2 = x_4$（第二个方程），代入第一个方程：$x_1 = 2x_2 - x_3$。

令 $x_2 = s, x_3 = t, x_4 = s$（$s, t$ 为自由参数），则 $x_1 = 2s - t$：

$$\begin{pmatrix}x_1\\x_2\\x_3\\x_4\end{pmatrix} = s\begin{pmatrix}2\\1\\0\\1\end{pmatrix} + t\begin{pmatrix}-1\\0\\1\\0\end{pmatrix}$$

故 $W$ 的一个基为：

$$\mathcal{B}_W = \left\{\begin{pmatrix}2\\1\\0\\1\end{pmatrix}, \begin{pmatrix}-1\\0\\1\\0\end{pmatrix}\right\}$$

$\dim(W) = 2$。

</details>

<details>
<summary>点击展开 练习 3 答案</summary>

解方程组 $c_1 b_1 + c_2 b_2 = x$：

$$\begin{pmatrix}2&-3\\-1&4\end{pmatrix}\begin{pmatrix}c_1\\c_2\end{pmatrix} = \begin{pmatrix}4\\-5\end{pmatrix}$$

计算系数矩阵的逆（行列式 $= 8-3 = 5$）：

$$\begin{pmatrix}2&-3\\-1&4\end{pmatrix}^{-1} = \frac{1}{5}\begin{pmatrix}4&3\\1&2\end{pmatrix}$$

$$\begin{pmatrix}c_1\\c_2\end{pmatrix} = \frac{1}{5}\begin{pmatrix}4&3\\1&2\end{pmatrix}\begin{pmatrix}4\\-5\end{pmatrix} = \frac{1}{5}\begin{pmatrix}16-15\\4-10\end{pmatrix} = \frac{1}{5}\begin{pmatrix}1\\-6\end{pmatrix} = \begin{pmatrix}1/5\\-6/5\end{pmatrix}$$

故 $[x]_\mathcal{B} = \begin{pmatrix}1/5\\-6/5\end{pmatrix}$。

**验证：** $\frac{1}{5}\begin{pmatrix}2\\-1\end{pmatrix} + \left(-\frac{6}{5}\right)\begin{pmatrix}-3\\4\end{pmatrix} = \begin{pmatrix}2/5+18/5\\-1/5-24/5\end{pmatrix} = \begin{pmatrix}4\\-5\end{pmatrix}$ ✓

</details>

<details>
<summary>点击展开 练习 4 答案</summary>

对 $A$ 进行行化简：

$$\begin{pmatrix}1&2&0&-1\\0&1&3&2\\1&3&3&1\end{pmatrix} \xrightarrow{R_3-R_1} \begin{pmatrix}1&2&0&-1\\0&1&3&2\\0&1&3&2\end{pmatrix} \xrightarrow{R_3-R_2} \begin{pmatrix}1&2&0&-1\\0&1&3&2\\0&0&0&0\end{pmatrix}$$

$\text{rank}(A) = 2$，$A$ 有 $n=4$ 列，由秩-零化度定理：

$$\dim(\text{Null}(A)) = n - \text{rank}(A) = 4 - 2 = 2$$

继续行化简至 RREF：

$$\xrightarrow{R_1-2R_2} \begin{pmatrix}1&0&-6&-5\\0&1&3&2\\0&0&0&0\end{pmatrix}$$

自由变量为 $x_3 = s, x_4 = t$。则 $x_2 = -3s - 2t$，$x_1 = 6s + 5t$：

$$x = s\begin{pmatrix}6\\-3\\1\\0\end{pmatrix} + t\begin{pmatrix}5\\-2\\0\\1\end{pmatrix}$$

零空间的基为 $\left\{\begin{pmatrix}6\\-3\\1\\0\end{pmatrix}, \begin{pmatrix}5\\-2\\0\\1\end{pmatrix}\right\}$。

</details>

<details>
<summary>点击展开 练习 5 答案</summary>

**(a)** 由维数公式 $\dim(V \cap W) = \dim(V) + \dim(W) - \dim(V+W)$，以及 $V + W \subseteq \mathbb{R}^5$，可知 $\dim(V+W) \leq 5$。

又 $\dim(V+W) \geq \max(\dim(V), \dim(W)) = 3$，故 $3 \leq \dim(V+W) \leq 5$。

因此 $\dim(V \cap W) = 3 + 3 - \dim(V+W) \in \{1, 2, 3\}$。

这三种情况均可实现：
- $\dim(V \cap W) = 3$：$V = W$（两空间完全重合）
- $\dim(V \cap W) = 2$：$V$ 和 $W$ 有一个公共二维子空间
- $\dim(V \cap W) = 1$：$V + W = \mathbb{R}^5$，两空间只共享一维子空间

**(b)** 由维数公式：

$$\dim(V+W) = \dim(V) + \dim(W) - \dim(V \cap W) = 6 - \dim(V \cap W)$$

由 (a) 知 $\dim(V \cap W) \in \{1, 2, 3\}$，故 $\dim(V+W) \in \{3, 4, 5\}$。

</details>
