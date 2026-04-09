# 第19章：特征值与特征向量

> 有些向量在矩阵作用下不会"转向"——它们只是被拉伸或压缩。这些特殊方向就是矩阵的**特征方向**，对应的缩放比例就是**特征值**。理解特征值与特征向量，就是读懂矩阵"最本质的行为模式"。

---

## 学习目标

完成本章学习后，你将能够：

- 理解特征值与特征向量的几何含义，从"不改变方向的变换"角度直觉地把握这一核心概念
- 掌握特征多项式的推导方式，利用 $\det(A - \lambda I) = 0$ 求解特征值
- 计算 $2 \times 2$ 和 $3 \times 3$ 矩阵的特征值与特征向量，理解代数重数与几何重数的区别
- 理解特征空间的结构：特征空间是 $(A - \lambda I)$ 的零空间
- 运用迹与行列式公式（迹 = 特征值之和，行列式 = 特征值之积）快速验算结果

---

## 19.1 特征值与特征向量的定义

### 19.1.1 定义

设 $A$ 是 $n \times n$ 的方阵。若存在**非零向量** $\mathbf{v} \in \mathbb{R}^n$ 和标量 $\lambda \in \mathbb{R}$（或 $\mathbb{C}$），使得：

$$\boxed{A\mathbf{v} = \lambda \mathbf{v}}$$

则称 $\lambda$ 是矩阵 $A$ 的一个**特征值**（eigenvalue），$\mathbf{v}$ 是对应于 $\lambda$ 的一个**特征向量**（eigenvector）。

**注意事项：**
- 特征向量必须是**非零**向量（若 $\mathbf{v} = \mathbf{0}$，则 $A\mathbf{0} = \lambda\mathbf{0}$ 对任意 $\lambda$ 成立，无意义）
- 若 $\mathbf{v}$ 是特征向量，则 $c\mathbf{v}$（$c \neq 0$）也是同一特征值对应的特征向量
- 特征值可以是复数，即使 $A$ 的元素全为实数

### 19.1.2 几何直觉

线性变换 $T(\mathbf{x}) = A\mathbf{x}$ 通常会旋转并拉伸向量——既改变方向，又改变长度。但特征向量是"方向不变"的例外：

$$A\mathbf{v} = \lambda \mathbf{v}$$

这意味着 $A$ 作用在 $\mathbf{v}$ 上，结果仍与 $\mathbf{v}$ 共线（同向或反向）。$|\lambda|$ 是拉伸（$>1$）或压缩（$<1$）的比例：

- $\lambda > 1$：同方向拉伸
- $0 < \lambda < 1$：同方向压缩
- $\lambda = 1$：方向和长度都不变
- $\lambda = 0$：$A\mathbf{v} = \mathbf{0}$，$\mathbf{v}$ 被映射到零向量（说明 $A$ 奇异）
- $\lambda < 0$：方向反转，再乘以 $|\lambda|$ 的比例缩放

```
几何示意（二维）：

    ↑ v₂                 ↑ A·v₂ = λ₂·v₂
    |                    | （沿 v₂ 方向缩放 λ₂ 倍）
    |    v₁              |
    +--------→           +--------→ A·v₁ = λ₁·v₁
                                    （沿 v₁ 方向缩放 λ₁ 倍）

特征向量 v₁, v₂ 在 A 的作用下只改变长度，不改变方向。
```

**例子（直观验证）：** 设 $A = \begin{pmatrix}2 & 0 \\ 0 & 3\end{pmatrix}$（对角矩阵），$\mathbf{v}_1 = (1, 0)^T$，$\mathbf{v}_2 = (0, 1)^T$。

$$A\mathbf{v}_1 = \begin{pmatrix}2\\0\end{pmatrix} = 2\mathbf{v}_1, \quad A\mathbf{v}_2 = \begin{pmatrix}0\\3\end{pmatrix} = 3\mathbf{v}_2$$

$\mathbf{v}_1$ 是特征值 $\lambda_1 = 2$ 的特征向量，$\mathbf{v}_2$ 是特征值 $\lambda_2 = 3$ 的特征向量。对角矩阵的特征值就是对角元素，特征向量就是标准基向量。

---

## 19.2 特征多项式

### 19.2.1 特征方程的推导

要求解 $A\mathbf{v} = \lambda\mathbf{v}$（$\mathbf{v} \neq \mathbf{0}$），改写为：

$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

这是一个关于 $\mathbf{v}$ 的齐次线性方程组。它有**非零解**当且仅当系数矩阵 $(A - \lambda I)$ **奇异**（不可逆），即：

$$\boxed{\det(A - \lambda I) = 0}$$

这就是**特征方程**（characteristic equation）。

### 19.2.2 特征多项式

展开 $\det(A - \lambda I)$ 后，得到关于 $\lambda$ 的多项式，称为**特征多项式**（characteristic polynomial）：

$$p(\lambda) = \det(A - \lambda I)$$

对 $n \times n$ 矩阵，$p(\lambda)$ 是 $n$ 次多项式：

$$p(\lambda) = (-1)^n \lambda^n + c_{n-1}\lambda^{n-1} + \cdots + c_1\lambda + c_0$$

**重要系数：**
- 最高次项系数：$(-1)^n$（$\lambda^n$ 项）
- $\lambda^{n-1}$ 系数：$(-1)^{n-1} \text{tr}(A)$，其中 $\text{tr}(A) = \sum_i a_{ii}$ 是矩阵的**迹**（trace）
- 常数项：$p(0) = \det(A - 0 \cdot I) = \det(A)$

由代数基本定理，$p(\lambda)$ 在复数域上有 $n$ 个根（含重数），即矩阵恰好有 $n$ 个特征值（含重数计算）。

### 19.2.3 二阶矩阵的特征多项式

设 $A = \begin{pmatrix}a & b \\ c & d\end{pmatrix}$，则：

$$A - \lambda I = \begin{pmatrix}a-\lambda & b \\ c & d-\lambda\end{pmatrix}$$

$$p(\lambda) = \det(A - \lambda I) = (a-\lambda)(d-\lambda) - bc = \lambda^2 - (a+d)\lambda + (ad - bc)$$

即：

$$\boxed{p(\lambda) = \lambda^2 - \text{tr}(A)\,\lambda + \det(A)}$$

特征值满足二次方程 $\lambda^2 - \text{tr}(A)\,\lambda + \det(A) = 0$，由韦达定理立得：

$$\lambda_1 + \lambda_2 = \text{tr}(A), \qquad \lambda_1 \cdot \lambda_2 = \det(A)$$

---

## 19.3 特征值的计算

### 19.3.1 二阶矩阵的计算例子

**例 19.1：** 设 $A = \begin{pmatrix}4 & 1 \\ 2 & 3\end{pmatrix}$，求其特征值与特征向量。

**第一步：求特征多项式**

$$p(\lambda) = \det\begin{pmatrix}4-\lambda & 1 \\ 2 & 3-\lambda\end{pmatrix} = (4-\lambda)(3-\lambda) - 2$$

$$= \lambda^2 - 7\lambda + 12 - 2 = \lambda^2 - 7\lambda + 10 = (\lambda - 2)(\lambda - 5)$$

**第二步：求特征值**

令 $p(\lambda) = 0$，得 $\lambda_1 = 2$，$\lambda_2 = 5$。

**验证：** $\lambda_1 + \lambda_2 = 7 = \text{tr}(A) = 4 + 3$，$\lambda_1 \lambda_2 = 10 = \det(A) = 12 - 2$。✓

**第三步：求 $\lambda_1 = 2$ 对应的特征向量**

求 $(A - 2I)\mathbf{v} = \mathbf{0}$ 的非零解：

$$A - 2I = \begin{pmatrix}2 & 1 \\ 2 & 1\end{pmatrix} \xrightarrow{\text{行化简}} \begin{pmatrix}2 & 1 \\ 0 & 0\end{pmatrix}$$

方程 $2v_1 + v_2 = 0$，取 $v_1 = 1$，则 $v_2 = -2$。

$$\mathbf{v}_1 = \begin{pmatrix}1\\-2\end{pmatrix}$$

**验证：** $A\mathbf{v}_1 = \begin{pmatrix}4 \cdot 1 + 1 \cdot (-2) \\ 2 \cdot 1 + 3 \cdot (-2)\end{pmatrix} = \begin{pmatrix}2\\-4\end{pmatrix} = 2\begin{pmatrix}1\\-2\end{pmatrix}$。✓

**第四步：求 $\lambda_2 = 5$ 对应的特征向量**

$$A - 5I = \begin{pmatrix}-1 & 1 \\ 2 & -2\end{pmatrix} \xrightarrow{\text{行化简}} \begin{pmatrix}-1 & 1 \\ 0 & 0\end{pmatrix}$$

方程 $-v_1 + v_2 = 0$，取 $v_1 = 1$，则 $v_2 = 1$。

$$\mathbf{v}_2 = \begin{pmatrix}1\\1\end{pmatrix}$$

**验证：** $A\mathbf{v}_2 = \begin{pmatrix}5\\5\end{pmatrix} = 5\begin{pmatrix}1\\1\end{pmatrix}$。✓

---

### 19.3.2 三阶矩阵的计算例子

**例 19.2：** 设 $A = \begin{pmatrix}1 & 0 & 0 \\ 1 & 2 & 0 \\ 1 & 1 & 3\end{pmatrix}$（下三角矩阵），求特征值与特征向量。

**第一步：求特征多项式**

由于 $A - \lambda I$ 是下三角矩阵，其行列式等于对角元之积：

$$p(\lambda) = \det(A - \lambda I) = (1-\lambda)(2-\lambda)(3-\lambda)$$

特征值为 $\lambda_1 = 1$，$\lambda_2 = 2$，$\lambda_3 = 3$。

**规律：三角矩阵（上三角或下三角）的特征值就是对角元素。**

**第二步：求 $\lambda_1 = 1$ 的特征向量**

$$A - I = \begin{pmatrix}0 & 0 & 0 \\ 1 & 1 & 0 \\ 1 & 1 & 2\end{pmatrix} \xrightarrow{R_3 \leftarrow R_3 - R_2} \begin{pmatrix}0 & 0 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 2\end{pmatrix}$$

由第三行：$2v_3 = 0 \Rightarrow v_3 = 0$；由第二行：$v_1 + v_2 = 0 \Rightarrow v_2 = -v_1$；取 $v_1 = 1$：

$$\mathbf{v}_1 = \begin{pmatrix}1\\-1\\0\end{pmatrix}$$

**第三步：求 $\lambda_2 = 2$ 的特征向量**

$$A - 2I = \begin{pmatrix}-1 & 0 & 0 \\ 1 & 0 & 0 \\ 1 & 1 & 1\end{pmatrix} \xrightarrow{\text{行化简}} \begin{pmatrix}1 & 0 & 0 \\ 0 & 1 & 1 \\ 0 & 0 & 0\end{pmatrix}$$

$v_1 = 0$，$v_2 + v_3 = 0$，取 $v_3 = 1$，$v_2 = -1$：

$$\mathbf{v}_2 = \begin{pmatrix}0\\-1\\1\end{pmatrix}$$

**第四步：求 $\lambda_3 = 3$ 的特征向量**

$$A - 3I = \begin{pmatrix}-2 & 0 & 0 \\ 1 & -1 & 0 \\ 1 & 1 & 0\end{pmatrix} \xrightarrow{\text{行化简}} \begin{pmatrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0\end{pmatrix}$$

$v_1 = 0$，$v_2 = 0$，$v_3$ 自由，取 $v_3 = 1$：

$$\mathbf{v}_3 = \begin{pmatrix}0\\0\\1\end{pmatrix}$$

---

### 19.3.3 代数重数与几何重数

**定义：** 设 $\lambda_0$ 是特征多项式 $p(\lambda)$ 的根。

- **代数重数**（algebraic multiplicity）：$\lambda_0$ 作为 $p(\lambda)$ 根的重数，记为 $m_a(\lambda_0)$
- **几何重数**（geometric multiplicity）：特征空间 $\ker(A - \lambda_0 I)$ 的维数，记为 $m_g(\lambda_0)$

**定理：** 对任意特征值 $\lambda_0$，有

$$1 \leq m_g(\lambda_0) \leq m_a(\lambda_0)$$

当所有特征值的 $m_g = m_a$ 时，矩阵**可对角化**（第20章内容）。

**例子（代数重数 $>$ 几何重数）：** 设 $A = \begin{pmatrix}2 & 1 \\ 0 & 2\end{pmatrix}$（Jordan块）。

$$p(\lambda) = (2-\lambda)^2 \implies \lambda = 2 \text{ 的代数重数为 } 2$$

$$A - 2I = \begin{pmatrix}0 & 1 \\ 0 & 0\end{pmatrix} \implies \ker(A - 2I) = \text{span}\left\{\begin{pmatrix}1\\0\end{pmatrix}\right\}$$

几何重数为 1，但代数重数为 2。$m_g < m_a$，此矩阵**不可对角化**。

---

## 19.4 特征空间

### 19.4.1 特征空间的定义

设 $\lambda_0$ 是矩阵 $A$ 的特征值，定义对应的**特征空间**（eigenspace）为：

$$\boxed{E_{\lambda_0} = \ker(A - \lambda_0 I) = \{\mathbf{v} \in \mathbb{R}^n \mid (A - \lambda_0 I)\mathbf{v} = \mathbf{0}\}}$$

即 $(A - \lambda_0 I)$ 的**零空间**（null space）。

特征空间 $E_{\lambda_0}$ 是一个子空间，它包含：
- 零向量（平凡地满足方程）
- 所有对应于 $\lambda_0$ 的特征向量（非零元素）

**维数：** $\dim(E_{\lambda_0}) = m_g(\lambda_0)$（即 $\lambda_0$ 的几何重数）。

### 19.4.2 特征空间的几何含义

特征空间 $E_{\lambda_0}$ 是矩阵 $A$ 作用下"被缩放 $\lambda_0$ 倍"的所有方向构成的子空间。在这个子空间中，矩阵 $A$ 的行为极为简单：它仅仅是标量乘法 $\mathbf{v} \mapsto \lambda_0 \mathbf{v}$。

**示例：** 反射矩阵 $A = \begin{pmatrix}1 & 0 \\ 0 & -1\end{pmatrix}$（关于 $x$ 轴的反射）。

$$p(\lambda) = (1-\lambda)(-1-\lambda) = \lambda^2 - 1 = 0 \implies \lambda_1 = 1, \lambda_2 = -1$$

特征空间：
- $E_1 = \ker(A - I) = \ker\begin{pmatrix}0&0\\0&-2\end{pmatrix} = \text{span}\left\{\begin{pmatrix}1\\0\end{pmatrix}\right\}$（$x$ 轴方向，不被反射影响）
- $E_{-1} = \ker(A + I) = \ker\begin{pmatrix}2&0\\0&0\end{pmatrix} = \text{span}\left\{\begin{pmatrix}0\\1\end{pmatrix}\right\}$（$y$ 轴方向，被反转）

几何含义清晰：$x$ 轴上的向量经反射不动（特征值1），$y$ 轴上的向量被翻转（特征值-1）。

### 19.4.3 不同特征值的特征向量线性无关

**定理：** 对应于**不同特征值**的特征向量线性无关。

**证明思路（$n=2$ 的情形）：** 设 $\lambda_1 \neq \lambda_2$，$A\mathbf{v}_1 = \lambda_1\mathbf{v}_1$，$A\mathbf{v}_2 = \lambda_2\mathbf{v}_2$。假设 $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 = \mathbf{0}$，对两边作用 $A$ 并利用特征方程：

$$c_1\lambda_1\mathbf{v}_1 + c_2\lambda_2\mathbf{v}_2 = \mathbf{0}$$

与原方程组联立，由 $\lambda_1 \neq \lambda_2$ 可得 $c_1 = c_2 = 0$，故线性无关。$\square$

**推论：** $n \times n$ 矩阵若有 $n$ 个不同特征值，则其特征向量构成 $\mathbb{R}^n$ 的一组基（第20章：可对角化的充分条件）。

---

## 19.5 特征值的性质

### 19.5.1 迹与特征值之和

**定理：** 矩阵 $A$ 的**迹**（trace）等于其所有特征值的和（含重数）：

$$\text{tr}(A) = \sum_{i=1}^{n} a_{ii} = \sum_{i=1}^{n} \lambda_i$$

**推导：** 特征多项式 $p(\lambda) = \det(A - \lambda I) = (-1)^n(\lambda - \lambda_1)(\lambda - \lambda_2)\cdots(\lambda - \lambda_n)$。展开 $\lambda^{n-1}$ 系数，两边比较即得。

**例子：** $A = \begin{pmatrix}4 & 1 \\ 2 & 3\end{pmatrix}$，$\text{tr}(A) = 7 = 2 + 5$。✓

### 19.5.2 行列式与特征值之积

**定理：** 矩阵 $A$ 的**行列式**等于其所有特征值的乘积（含重数）：

$$\det(A) = \prod_{i=1}^{n} \lambda_i$$

**推导：** 令 $\lambda = 0$ 代入 $p(\lambda) = (-1)^n(\lambda - \lambda_1)\cdots(\lambda - \lambda_n)$：

$$p(0) = \det(A) = (-1)^n(-\lambda_1)\cdots(-\lambda_n) = \lambda_1\lambda_2\cdots\lambda_n$$

**推论：** $A$ 可逆 $\iff$ $\det(A) \neq 0$ $\iff$ $A$ 的所有特征值均非零。

**例子：** $A = \begin{pmatrix}4 & 1 \\ 2 & 3\end{pmatrix}$，$\det(A) = 10 = 2 \times 5$。✓

### 19.5.3 常用特征值性质汇总

| 性质 | 描述 |
|:---|:---|
| $A\mathbf{v} = \lambda\mathbf{v}$ | 则 $A^k\mathbf{v} = \lambda^k\mathbf{v}$（矩阵幂） |
| $A$ 可逆且 $\lambda \neq 0$ | 则 $A^{-1}\mathbf{v} = \dfrac{1}{\lambda}\mathbf{v}$（逆矩阵） |
| $A\mathbf{v} = \lambda\mathbf{v}$ | 则 $(A - cI)\mathbf{v} = (\lambda - c)\mathbf{v}$（平移） |
| $A$ 实对称矩阵 | 特征值全为实数，不同特征值的特征向量两两正交 |
| $A$ 正交矩阵 | 特征值满足 $|\lambda| = 1$ |
| $A$ 正定矩阵 | 特征值全为正数 |

**矩阵幂的计算（重要应用）：** 若 $A\mathbf{v} = \lambda\mathbf{v}$，则：

$$A^2\mathbf{v} = A(A\mathbf{v}) = A(\lambda\mathbf{v}) = \lambda(A\mathbf{v}) = \lambda^2\mathbf{v}$$

$$A^k\mathbf{v} = \lambda^k\mathbf{v}$$

这使得利用特征分解计算矩阵高次幂变得极为高效（将在第20章详细讨论）。

---

## 本章小结

- **特征方程** $A\mathbf{v} = \lambda\mathbf{v}$：特征向量 $\mathbf{v}$ 在矩阵作用下只改变幅度，方向不变（或反向）；特征值 $\lambda$ 是缩放比例。

- **特征多项式** $p(\lambda) = \det(A - \lambda I)$：$n$ 次多项式，其根即为特征值；求特征向量需进一步求解 $(A - \lambda I)\mathbf{v} = \mathbf{0}$ 的非零解。

- **特征空间** $E_\lambda = \ker(A - \lambda I)$：是一个子空间，包含所有 $\lambda$-特征向量（加上零向量）；维数称为 $\lambda$ 的几何重数。

- **代数重数 $\geq$ 几何重数**：若二者相等，矩阵可对角化；若某特征值的代数重数大于几何重数（如Jordan块），矩阵不可对角化。

- **迹与行列式公式**：$\text{tr}(A) = \sum \lambda_i$，$\det(A) = \prod \lambda_i$，是快速验算特征值计算结果的工具。

---

## 深度学习应用：PCA 与谱聚类

### 背景

特征值与特征向量是数据科学和机器学习中最基础的工具之一。**主成分分析（PCA）** 和**谱聚类（Spectral Clustering）** 都依赖于对特定矩阵做特征分解，分别用于降维和图数据的聚类。

---

### 19.6.1 PCA：协方差矩阵的特征向量

**问题设定：** 给定数据集 $X \in \mathbb{R}^{n \times d}$（$n$ 个样本，每个样本 $d$ 维），希望找到 $k \ll d$ 维的低维表示，使信息损失最小。

**协方差矩阵：** 设数据已中心化（即 $\sum_i \mathbf{x}_i = \mathbf{0}$），则样本协方差矩阵为：

$$\Sigma = \frac{1}{n-1} X^T X \in \mathbb{R}^{d \times d}$$

$\Sigma$ 是实对称半正定矩阵，因此：
- 特征值全为非负实数：$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$
- 不同特征值对应的特征向量两两正交

**PCA的核心定理：** 设 $\Sigma\mathbf{q}_i = \lambda_i\mathbf{q}_i$，其中 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$，$\{\mathbf{q}_1, \ldots, \mathbf{q}_d\}$ 是标准正交特征向量（主成分方向）。

将数据投影到前 $k$ 个主成分方向上：

$$\tilde{X} = X Q_k, \quad Q_k = [\mathbf{q}_1 \mid \mathbf{q}_2 \mid \cdots \mid \mathbf{q}_k] \in \mathbb{R}^{d \times k}$$

**为什么是最优的？** 投影到 $\mathbf{q}_1$ 方向上，样本方差为：

$$\text{Var} = \frac{1}{n-1}\sum_{i=1}^n (\mathbf{q}_1^T \mathbf{x}_i)^2 = \mathbf{q}_1^T \Sigma \mathbf{q}_1$$

在 $\|\mathbf{q}_1\| = 1$ 的约束下，Rayleigh商 $\dfrac{\mathbf{q}_1^T \Sigma \mathbf{q}_1}{\mathbf{q}_1^T \mathbf{q}_1}$ 的最大值恰好是 $\Sigma$ 的最大特征值 $\lambda_1$，最大化点为对应特征向量 $\mathbf{q}_1$。

**方差解释率：** 前 $k$ 个主成分捕获的总方差比例为：

$$\text{explained variance ratio} = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i}$$

这是选择 $k$ 的量化依据（通常取 $\geq 90\%$ 或 $\geq 95\%$）。

---

### 19.6.2 谱聚类简介：图拉普拉斯矩阵的特征向量

**问题设定：** 给定 $n$ 个数据点，它们之间有相似度 $w_{ij} \geq 0$（构成相似度图），希望将数据分成 $k$ 个簇，使得簇内相似度高、簇间相似度低。

**图拉普拉斯矩阵（Graph Laplacian）：**

设相似度矩阵 $W \in \mathbb{R}^{n \times n}$（$W_{ij} = w_{ij}$），度矩阵 $D = \text{diag}(d_1, \ldots, d_n)$（其中 $d_i = \sum_j w_{ij}$），则图拉普拉斯矩阵为：

$$L = D - W$$

$L$ 是实对称半正定矩阵，特征值 $0 = \mu_1 \leq \mu_2 \leq \cdots \leq \mu_n$。

**谱聚类的关键性质：**

- $L$ 的特征值 $\mu_1 = 0$，对应特征向量为全1向量 $\mathbf{1}$
- **$\mu_2 = 0$ 的代数重数 = 图的连通分量数**
- 若图有 $k$ 个连通分量，前 $k$ 个特征值均为0，其余 $> 0$

**谱聚类算法（标准流程）：**

1. 构建相似度矩阵 $W$（如高斯核 $w_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$）
2. 计算图拉普拉斯 $L = D - W$（或其归一化版本）
3. 计算 $L$ 的前 $k$ 个最小特征值对应的特征向量 $[\mathbf{u}_1, \ldots, \mathbf{u}_k]$
4. 以每个数据点在特征向量空间中的坐标 $(\mathbf{u}_1[i], \ldots, \mathbf{u}_k[i])$ 作为新特征
5. 在低维特征空间中运行 $k$-means 聚类

**几何直觉：** 图拉普拉斯的特征向量将数据嵌入到 $k$ 维空间，使得图上"相连"的节点在嵌入空间中距离近，"不相连"的节点距离远。这个嵌入使得 $k$-means 等线性聚类方法能处理原空间中非线性可分的数据。

---

### 19.6.3 PyTorch 代码：从零实现 PCA

```python
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 1. 生成示例数据（二维高斯，带相关性）─────────────────────────
torch.manual_seed(42)
n = 200
# 原始数据：沿 (1, 2) 方向方差大，垂直方向方差小
mean = torch.zeros(2)
# 协方差结构：主方向为 (1/√5, 2/√5)
L_factor = torch.tensor([[2.0, 0.5],
                          [0.5, 0.5]])  # Cholesky 因子
data = torch.randn(n, 2) @ L_factor.T  # shape: (n, 2)

# ── 2. 从零实现 PCA ─────────────────────────────────────────────
def pca_from_scratch(X: torch.Tensor, k: int):
    """
    输入:
        X: (n, d) 数据矩阵
        k: 保留的主成分数
    输出:
        components: (d, k) 主成分方向（列向量）
        explained_var: (d,) 各方向的方差（特征值）
        X_reduced: (n, k) 降维后的数据
    """
    n_samples = X.shape[0]

    # 步骤1：中心化
    mean_vec = X.mean(dim=0)           # (d,)
    X_centered = X - mean_vec          # (n, d)

    # 步骤2：计算协方差矩阵
    cov = X_centered.T @ X_centered / (n_samples - 1)   # (d, d)

    # 步骤3：对协方差矩阵做特征分解
    # torch.linalg.eigh 专用于实对称矩阵，返回升序特征值
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # 升序排列

    # 步骤4：按特征值降序排列
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]         # (d,)  降序特征值
    eigenvectors = eigenvectors[:, idx]    # (d, d) 对应特征向量（列）

    # 步骤5：取前 k 个主成分
    components = eigenvectors[:, :k]       # (d, k)

    # 步骤6：投影数据
    X_reduced = X_centered @ components   # (n, k)

    return components, eigenvalues, X_reduced, mean_vec


# ── 3. 执行 PCA ─────────────────────────────────────────────────
components, eigenvalues, data_2d, mean_vec = pca_from_scratch(data, k=2)

print("=" * 50)
print("PCA 结果分析")
print("=" * 50)

total_var = eigenvalues.sum().item()
for i, (val, ratio) in enumerate(
        zip(eigenvalues.tolist(),
            (eigenvalues / total_var).tolist())):
    print(f"  PC{i+1}: 特征值 = {val:.4f}, 方差解释率 = {ratio*100:.1f}%")

print(f"\n前1个主成分方向: {components[:, 0].tolist()}")
print(f"（理论方向约为 (1/√5, 2/√5) ≈ {[0.447, 0.894]}）")

# ── 4. 验证：与 torch.linalg.svd 比较 ──────────────────────────
X_centered = data - mean_vec
U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
svd_eigenvalues = (S ** 2) / (data.shape[0] - 1)
print(f"\n通过 SVD 验证特征值: {svd_eigenvalues.tolist()}")
print(f"通过协方差矩阵得特征值: {eigenvalues.tolist()}")
print("（两者应相同）")

# ── 5. 一维降维示例：重构误差 ───────────────────────────────────
k_list = [1, 2]
for k in k_list:
    comp_k = components[:, :k]                    # (2, k)
    proj = (X_centered @ comp_k) @ comp_k.T       # (n, 2)，投影到 k 维子空间后还原
    recon_error = ((X_centered - proj) ** 2).mean().item()
    var_captured = eigenvalues[:k].sum().item() / total_var * 100
    print(f"\nk={k}: 方差捕获率={var_captured:.1f}%, 重构均方误差={recon_error:.4f}")

# ── 6. 可视化 ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：原始数据 + 主成分方向
ax = axes[0]
ax.scatter(data[:, 0].numpy(), data[:, 1].numpy(),
           alpha=0.5, s=20, label="数据点")
scale = 2.0
for i in range(2):
    direction = components[:, i].numpy()
    ax.annotate("", xy=mean_vec.numpy() + scale * direction,
                xytext=mean_vec.numpy(),
                arrowprops=dict(arrowstyle="->", color=f"C{i+1}", lw=2))
    ax.text(*(mean_vec.numpy() + (scale + 0.2) * direction),
            f"PC{i+1} (λ={eigenvalues[i]:.2f})", fontsize=9, color=f"C{i+1}")
ax.set_title("原始数据与主成分方向")
ax.set_aspect("equal")
ax.legend()
ax.grid(True, alpha=0.3)

# 右图：方差解释率曲线（碎石图）
ax = axes[1]
ratios = (eigenvalues / total_var * 100).tolist()
cumulative = [sum(ratios[:i+1]) for i in range(len(ratios))]
ax.bar(range(1, len(ratios)+1), ratios, alpha=0.7, label="各成分方差解释率")
ax.step(range(1, len(ratios)+1), cumulative, where="mid",
        color="red", linewidth=2, label="累积方差解释率")
ax.axhline(y=95, color="gray", linestyle="--", alpha=0.7, label="95%阈值")
ax.set_xlabel("主成分编号")
ax.set_ylabel("方差解释率 (%)")
ax.set_title("碎石图（Scree Plot）")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pca_demo.png", dpi=100, bbox_inches="tight")
print("\n图表已保存至 pca_demo.png")
```

**代码解读：**

- `pca_from_scratch` 完整实现了 PCA 的数学步骤：中心化 → 协方差矩阵 → 特征分解 → 按特征值排序 → 投影
- `torch.linalg.eigh` 专用于实对称矩阵（Hermitian），比通用的 `torch.linalg.eig` 更快且数值更稳定，且保证返回实数特征值
- SVD 验证部分说明了 PCA 和 SVD 的等价性：协方差矩阵的特征值等于数据矩阵奇异值的平方除以 $(n-1)$
- **碎石图（Scree Plot）** 是实际中选择 $k$ 的可视化工具，常在"肘部"（曲线斜率急剧下降处）截断

**PCA 与特征值的联系总结：**

| 概念 | 数学含义 |
|:---|:---|
| 主成分方向 | 协方差矩阵 $\Sigma$ 的特征向量 $\mathbf{q}_i$ |
| 各方向方差 | 对应特征值 $\lambda_i$ |
| 降维后的坐标 | $\tilde{\mathbf{x}}_i = Q_k^T (\mathbf{x}_i - \bar{\mathbf{x}})$ |
| 方差解释率 | $\lambda_i / \sum_j \lambda_j$ |
| 最优性保证 | Rayleigh 商的最大化点 = 最大特征向量 |

---

## 练习题

**练习 19.1（基础）** 设矩阵 $A = \begin{pmatrix}3 & 0 \\ 8 & -1\end{pmatrix}$。

(a) 计算 $A$ 的特征多项式 $p(\lambda) = \det(A - \lambda I)$。
(b) 求 $A$ 的两个特征值 $\lambda_1, \lambda_2$，并用迹和行列式公式验证。
(c) 分别求对应两个特征值的特征向量。

---

**练习 19.2（中等）** 设矩阵 $B = \begin{pmatrix}0 & -1 \\ 1 & 0\end{pmatrix}$（旋转 $90°$ 的矩阵）。

(a) 求 $B$ 的特征多项式，说明 $B$ 在实数范围内没有特征值。
(b) 在复数范围内，求 $B$ 的特征值（$\lambda \in \mathbb{C}$）并解释几何含义。
(c) 若矩阵代表一个旋转，特征值在几何上意味着什么？为何实数旋转矩阵通常没有实数特征值（除了旋转 $0°$ 和 $180°$）？

---

**练习 19.3（中等）** 对矩阵 $C = \begin{pmatrix}2 & 2 & 1 \\ 0 & 2 & 1 \\ 0 & 0 & 3\end{pmatrix}$ 求所有特征值与特征向量，并计算每个特征值的代数重数与几何重数。

---

**练习 19.4（较难）** 设 $A$ 是 $n \times n$ 实对称矩阵，$\mathbf{v}_1, \mathbf{v}_2$ 分别是对应特征值 $\lambda_1 \neq \lambda_2$ 的特征向量。

(a) 证明 $\mathbf{v}_1 \perp \mathbf{v}_2$（即 $\mathbf{v}_1^T \mathbf{v}_2 = 0$）。（提示：利用 $A^T = A$，考虑 $\mathbf{v}_2^T A \mathbf{v}_1$ 的两种展开方式。）
(b) 设 $A = \begin{pmatrix}5 & 2 \\ 2 & 2\end{pmatrix}$，求其特征值和特征向量，验证两个特征向量正交。

---

**练习 19.5（挑战）** 设数据矩阵 $X = \begin{pmatrix}2 & 0 \\ 0 & 1 \\ -2 & -1\end{pmatrix}$（已中心化，3 个样本，2 维特征）。

(a) 计算协方差矩阵 $\Sigma = \dfrac{1}{2} X^T X$（注意分母为 $n-1=2$）。
(b) 求 $\Sigma$ 的特征值和特征向量（主成分方向）。
(c) 将数据投影到第一主成分方向上，计算一维降维后的坐标。
(d) 计算第一主成分的方差解释率，以及降维引入的重构误差 $\|X - X Q_1 Q_1^T\|_F^2$（其中 $Q_1$ 是第一主成分方向的列向量）。

---

## 练习答案

<details>
<summary>练习 19.1 答案</summary>

**(a) 特征多项式：**

$$A - \lambda I = \begin{pmatrix}3-\lambda & 0 \\ 8 & -1-\lambda\end{pmatrix}$$

$$p(\lambda) = (3-\lambda)(-1-\lambda) - 0 \cdot 8 = (3-\lambda)(-1-\lambda) = -\lambda^2 + 2\lambda + 3$$

或等价地写成 $p(\lambda) = \lambda^2 - 2\lambda - 3 = (\lambda - 3)(\lambda + 1)$（按首一多项式约定取正号）。

**(b) 特征值：**

令 $(\lambda-3)(\lambda+1) = 0$，得 $\lambda_1 = 3$，$\lambda_2 = -1$。

**验证：**
- $\lambda_1 + \lambda_2 = 3 + (-1) = 2 = \text{tr}(A) = 3 + (-1)$。✓
- $\lambda_1 \cdot \lambda_2 = 3 \times (-1) = -3 = \det(A) = 3 \times (-1) - 0 \times 8 = -3$。✓

**(c) 特征向量：**

**对 $\lambda_1 = 3$：**

$$A - 3I = \begin{pmatrix}0 & 0 \\ 8 & -4\end{pmatrix} \xrightarrow{\text{行化简}} \begin{pmatrix}2 & -1 \\ 0 & 0\end{pmatrix}$$

方程 $2v_1 - v_2 = 0$，取 $v_1 = 1$，$v_2 = 2$，故 $\mathbf{v}_1 = \begin{pmatrix}1\\2\end{pmatrix}$。

验证：$A\mathbf{v}_1 = \begin{pmatrix}3\\8-2\end{pmatrix} = \begin{pmatrix}3\\6\end{pmatrix} = 3\begin{pmatrix}1\\2\end{pmatrix}$。✓

**对 $\lambda_2 = -1$：**

$$A + I = \begin{pmatrix}4 & 0 \\ 8 & 0\end{pmatrix} \xrightarrow{\text{行化简}} \begin{pmatrix}1 & 0 \\ 0 & 0\end{pmatrix}$$

方程 $v_1 = 0$，$v_2$ 自由，取 $v_2 = 1$，故 $\mathbf{v}_2 = \begin{pmatrix}0\\1\end{pmatrix}$。

验证：$A\mathbf{v}_2 = \begin{pmatrix}0\\-1\end{pmatrix} = (-1)\begin{pmatrix}0\\1\end{pmatrix}$。✓

</details>

---

<details>
<summary>练习 19.2 答案</summary>

**(a) 特征多项式：**

$$p(\lambda) = \det(B - \lambda I) = \det\begin{pmatrix}-\lambda & -1 \\ 1 & -\lambda\end{pmatrix} = \lambda^2 + 1$$

令 $\lambda^2 + 1 = 0$，在实数范围内无解（$\lambda^2 = -1$ 无实数根），故 $B$ 在实数范围内没有特征值。

**(b) 复数特征值：**

$\lambda^2 = -1 \implies \lambda = \pm i$（虚数单位）。

特征值为 $\lambda_1 = i$，$\lambda_2 = -i$（互为共轭）。

**几何含义：** 旋转 $90°$ 矩阵没有实数意义上"方向不变"的向量——所有非零向量都被旋转了方向。复数特征值 $e^{\pm i\theta}$（模为1）对应旋转操作，$\theta = 90°$ 时恰好是 $\pm i$。

**(c) 关于旋转矩阵的特征值：**

旋转角 $\theta$ 的矩阵特征值为 $e^{\pm i\theta} = \cos\theta \pm i\sin\theta$。

- $\theta = 0°$：特征值为 $e^{0} = 1$（两个相同的实数特征值），矩阵为单位矩阵，每个方向都不变
- $\theta = 180°$：特征值为 $e^{\pm i\pi} = -1$（两个相同的实数特征值），矩阵为 $-I$，每个方向都反转
- 其他角度：特征值为复数 $e^{\pm i\theta}$，模为1但不是实数，物理上意味着不存在被纯粹缩放而不旋转的方向

</details>

---

<details>
<summary>练习 19.3 答案</summary>

$C$ 是上三角矩阵，故特征值为对角元素：$\lambda_1 = 2$（对角出现2次），$\lambda_2 = 3$。

**代数重数：** $p(\lambda) = (2-\lambda)^2(3-\lambda)$，故 $m_a(2) = 2$，$m_a(3) = 1$。

**求 $\lambda_1 = 2$ 的特征向量（$m_a = 2$）：**

$$C - 2I = \begin{pmatrix}0 & 2 & 1 \\ 0 & 0 & 1 \\ 0 & 0 & 1\end{pmatrix} \xrightarrow{\text{行化简}} \begin{pmatrix}0 & 2 & 1 \\ 0 & 0 & 1 \\ 0 & 0 & 0\end{pmatrix}$$

由第二行：$v_3 = 0$；由第一行：$2v_2 + v_3 = 0 \Rightarrow v_2 = 0$；$v_1$ 自由。

$$E_2 = \text{span}\left\{\begin{pmatrix}1\\0\\0\end{pmatrix}\right\}, \quad m_g(2) = 1$$

**结论：** $m_a(2) = 2 > m_g(2) = 1$，特征值 $\lambda = 2$ 对应一个缺陷（矩阵**不可对角化**）。

**求 $\lambda_2 = 3$ 的特征向量（$m_a = 1$）：**

$$C - 3I = \begin{pmatrix}-1 & 2 & 1 \\ 0 & -1 & 1 \\ 0 & 0 & 0\end{pmatrix} \xrightarrow{\text{行化简}} \begin{pmatrix}1 & 0 & -3 \\ 0 & 1 & -1 \\ 0 & 0 & 0\end{pmatrix}$$

$v_1 = 3v_3$，$v_2 = v_3$，取 $v_3 = 1$：

$$E_3 = \text{span}\left\{\begin{pmatrix}3\\1\\1\end{pmatrix}\right\}, \quad m_g(3) = 1 = m_a(3)$$

</details>

---

<details>
<summary>练习 19.4 答案</summary>

**(a) 证明实对称矩阵不同特征值的特征向量正交：**

设 $A\mathbf{v}_1 = \lambda_1\mathbf{v}_1$，$A\mathbf{v}_2 = \lambda_2\mathbf{v}_2$，$\lambda_1 \neq \lambda_2$，$A^T = A$。

考虑 $\mathbf{v}_2^T A \mathbf{v}_1$ 的两种展开方式：

**方式一（先用 $A\mathbf{v}_1 = \lambda_1\mathbf{v}_1$）：**

$$\mathbf{v}_2^T A \mathbf{v}_1 = \mathbf{v}_2^T (\lambda_1 \mathbf{v}_1) = \lambda_1 (\mathbf{v}_2^T \mathbf{v}_1)$$

**方式二（先转置，利用 $A^T = A$）：**

$$\mathbf{v}_2^T A \mathbf{v}_1 = (A^T \mathbf{v}_2)^T \mathbf{v}_1 = (A\mathbf{v}_2)^T \mathbf{v}_1 = (\lambda_2 \mathbf{v}_2)^T \mathbf{v}_1 = \lambda_2 (\mathbf{v}_2^T \mathbf{v}_1)$$

两种方式相减：

$$(\lambda_1 - \lambda_2)(\mathbf{v}_2^T \mathbf{v}_1) = 0$$

因 $\lambda_1 \neq \lambda_2$，故 $\mathbf{v}_1^T \mathbf{v}_2 = \mathbf{v}_2^T \mathbf{v}_1 = 0$，即 $\mathbf{v}_1 \perp \mathbf{v}_2$。$\square$

**(b) 计算 $A = \begin{pmatrix}5&2\\2&2\end{pmatrix}$ 的特征值和特征向量：**

$$p(\lambda) = (5-\lambda)(2-\lambda) - 4 = \lambda^2 - 7\lambda + 6 = (\lambda - 1)(\lambda - 6)$$

特征值：$\lambda_1 = 1$，$\lambda_2 = 6$。

**$\lambda_1 = 1$ 的特征向量：**

$$A - I = \begin{pmatrix}4 & 2 \\ 2 & 1\end{pmatrix} \to \begin{pmatrix}2 & 1 \\ 0 & 0\end{pmatrix}$$

$2v_1 + v_2 = 0$，取 $\mathbf{v}_1 = \begin{pmatrix}1\\-2\end{pmatrix}$。

**$\lambda_2 = 6$ 的特征向量：**

$$A - 6I = \begin{pmatrix}-1 & 2 \\ 2 & -4\end{pmatrix} \to \begin{pmatrix}1 & -2 \\ 0 & 0\end{pmatrix}$$

$v_1 = 2v_2$，取 $\mathbf{v}_2 = \begin{pmatrix}2\\1\end{pmatrix}$。

**验证正交性：** $\mathbf{v}_1 \cdot \mathbf{v}_2 = 1 \times 2 + (-2) \times 1 = 0$。✓

</details>

---

<details>
<summary>练习 19.5 答案</summary>

**(a) 协方差矩阵：**

$$\Sigma = \frac{1}{2} X^T X = \frac{1}{2}\begin{pmatrix}2&0&-2\\0&1&-1\end{pmatrix}\begin{pmatrix}2&0\\0&1\\-2&-1\end{pmatrix}$$

$$= \frac{1}{2}\begin{pmatrix}4+0+4 & 0+0+2 \\ 0+0+2 & 0+1+1\end{pmatrix} = \frac{1}{2}\begin{pmatrix}8 & 2 \\ 2 & 2\end{pmatrix} = \begin{pmatrix}4 & 1 \\ 1 & 1\end{pmatrix}$$

**(b) 特征值和特征向量：**

$$p(\lambda) = (4-\lambda)(1-\lambda) - 1 = \lambda^2 - 5\lambda + 3$$

$$\lambda = \frac{5 \pm \sqrt{25 - 12}}{2} = \frac{5 \pm \sqrt{13}}{2}$$

$\lambda_1 = \dfrac{5 + \sqrt{13}}{2} \approx 4.303$，$\lambda_2 = \dfrac{5 - \sqrt{13}}{2} \approx 0.697$。

**第一主成分方向（$\lambda_1$）：**

$$\Sigma - \lambda_1 I = \begin{pmatrix}4-\lambda_1 & 1 \\ 1 & 1-\lambda_1\end{pmatrix}$$

由第一行：$(4 - \lambda_1)v_1 + v_2 = 0$，即 $v_2 = (\lambda_1 - 4)v_1 = \dfrac{\sqrt{13}-3}{2} v_1$。

归一化后得主成分方向 $\mathbf{q}_1$（分量比为 $1 : \dfrac{\sqrt{13}-3}{2}$）。

**(c) 投影到第一主成分：**

设 $\mathbf{q}_1$ 为归一化后的第一主成分方向，各数据点的一维坐标为：

$$z_i = \mathbf{q}_1^T \mathbf{x}_i, \quad i = 1, 2, 3$$

**(d) 方差解释率与重构误差：**

$$\text{方差解释率} = \frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{(5+\sqrt{13})/2}{5} = \frac{5+\sqrt{13}}{10} \approx \frac{4.303}{5} \approx 86.1\%$$

重构误差（Frobenius 范数平方）等于被丢弃的特征值之和乘以 $(n-1)$（此处为丢弃 $\lambda_2$ 对应方向后的总误差）：

$$\|X - X Q_1 Q_1^T\|_F^2 = (n-1)\lambda_2 = 2 \times \frac{5-\sqrt{13}}{2} = 5 - \sqrt{13} \approx 1.394$$

（其中用到：投影误差 = 丢弃的协方差方向上的总方差 = $(n-1) \times$ 被丢弃的特征值。）

</details>
