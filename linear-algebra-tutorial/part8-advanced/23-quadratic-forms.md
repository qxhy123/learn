# 第23章：二次型

> **前置知识**：第19章（特征值与特征向量）、第20章（对角化）、第21章（对称矩阵与谱定理）
>
> **本章难度**：★★★★☆
>
> **预计学习时间**：5-6 小时

---

## 学习目标

学完本章后，你将能够：

- 将任意二次多项式 $Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ 用实对称矩阵 $A$ 表示，并在矩阵表示与展开形式之间自由转换
- 通过正交变换将二次型化为标准形，消去所有交叉项，理解坐标变换的几何含义
- 熟练运用惯性定理（Sylvester 定理），利用正惯性指数与负惯性指数对二次型进行分类，判断两个二次型是否合同
- 掌握正定、负定、不定等各类二次型的等价判别条件（特征值、顺序主子式、配方法），并能在具体问题中快速判断
- 将单位球面上的约束二次型优化与特征值联系起来，理解最大/最小曲率方向的变分意义，并应用于损失函数的曲率分析

---

## 23.1 二次型的定义

### 什么是二次型

在一元微积分中，$f(x) = ax^2$ 是最简单的二次函数。当变量扩展到 $n$ 维时，**二次型（quadratic form）**是所有变量的二次齐次多项式：

$$Q(x_1, x_2, \ldots, x_n) = \sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij} x_i x_j$$

**两变量的例子**：

$$Q(x_1, x_2) = 3x_1^2 + 4x_1 x_2 - 2x_2^2$$

注意：只有"纯二次项"（$x_i^2$）和"交叉项"（$x_i x_j, i \neq j$），没有一次项或常数项——这是二次型的本质特征。

### 矩阵表示

任意二次型都可以用一个**实对称矩阵** $A$ 简洁表示：

$$\boxed{Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}}$$

其中 $\mathbf{x} = (x_1, x_2, \ldots, x_n)^T$，矩阵元素为：

$$a_{ii} = \text{（} x_i^2 \text{ 的系数）}, \quad a_{ij} = a_{ji} = \frac{1}{2} \times \text{（} x_i x_j \text{ 的系数，} i \neq j \text{）}$$

**为何要对称化？** 对同一个二次型，存在无数个矩阵表示（因为 $x_i x_j = x_j x_i$，交叉项系数可以任意分配给 $a_{ij}$ 和 $a_{ji}$）。规定 $A = A^T$ 保证了表示的唯一性，同时让谱定理可以直接应用。

**例 23.1**：将 $Q(x_1, x_2) = 3x_1^2 + 4x_1 x_2 - 2x_2^2$ 写成矩阵形式。

$$A = \begin{pmatrix} 3 & 2 \\ 2 & -2 \end{pmatrix}$$

（交叉项系数 $4$ 平均分配：$a_{12} = a_{21} = 2$。）

验证：

$$\mathbf{x}^T A \mathbf{x} = \begin{pmatrix} x_1 & x_2 \end{pmatrix} \begin{pmatrix} 3 & 2 \\ 2 & -2 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} x_1 & x_2 \end{pmatrix} \begin{pmatrix} 3x_1 + 2x_2 \\ 2x_1 - 2x_2 \end{pmatrix} = 3x_1^2 + 2x_1 x_2 + 2x_1 x_2 - 2x_2^2 = 3x_1^2 + 4x_1 x_2 - 2x_2^2 \checkmark$$

**例 23.2**（三元二次型）：

$$Q(x_1, x_2, x_3) = x_1^2 + 2x_2^2 + 3x_3^2 + 2x_1 x_2 - 4x_1 x_3 + 6x_2 x_3$$

对应矩阵：

$$A = \begin{pmatrix} 1 & 1 & -2 \\ 1 & 2 & 3 \\ -2 & 3 & 3 \end{pmatrix}$$

### 几何直觉

二次型 $Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ 定义了 $\mathbb{R}^n$ 上的一个"高度函数"：

- 当 $n = 2$ 时，$Q$ 的等高线 $\{Q(\mathbf{x}) = c\}$ 是以原点为中心的**圆锥曲线**（椭圆、双曲线，取决于 $A$ 的符号特征）；
- 当 $A$ 正定时，等高线是同心椭圆，原点是全局最低点（碗形曲面）；
- 当 $A$ 不定时，等高线是双曲线，原点是鞍点（马鞍面）。

交叉项 $x_i x_j$（$i \neq j$）的存在使等高线的主轴不与坐标轴对齐——**标准化**的目标正是旋转坐标轴，使主轴与坐标轴重合，从而消去交叉项。

---

## 23.2 标准化

### 什么是标准形

**标准形（standard form / canonical form）**是没有交叉项的二次型：

$$Q = \lambda_1 y_1^2 + \lambda_2 y_2^2 + \cdots + \lambda_n y_n^2$$

在新坐标 $\mathbf{y} = (y_1, \ldots, y_n)^T$ 下，矩阵是对角的，各个坐标方向相互独立。

### 正交变换法

由第21章的**谱定理**，实对称矩阵 $A$ 可以被正交对角化：

$$A = Q \Lambda Q^T, \quad Q^T Q = I$$

令 $\mathbf{x} = Q\mathbf{y}$（即 $\mathbf{y} = Q^T \mathbf{x}$），代入二次型：

$$\mathbf{x}^T A \mathbf{x} = (Q\mathbf{y})^T A (Q\mathbf{y}) = \mathbf{y}^T Q^T A Q \mathbf{y} = \mathbf{y}^T \Lambda \mathbf{y} = \sum_{i=1}^{n} \lambda_i y_i^2$$

**结论**：通过正交变换 $\mathbf{x} = Q\mathbf{y}$（其中 $Q$ 的列是 $A$ 的标准正交特征向量），任意二次型可以化为标准形，系数恰好是 $A$ 的特征值。

**几何意义**：正交变换是旋转（和/或反射），它将坐标轴旋转到与 $A$ 的主轴（特征向量方向）重合的位置。在新坐标系中，二次型的等高线主轴恰好沿坐标轴排列。

### 具体例子

**例 23.3**：将 $Q(x_1, x_2) = 3x_1^2 + 4x_1 x_2 - x_2^2$ 化为标准形。

**第一步：写出矩阵。**

$$A = \begin{pmatrix} 3 & 2 \\ 2 & -1 \end{pmatrix}$$

**第二步：求特征值。**

$$\det(A - \lambda I) = (3-\lambda)(-1-\lambda) - 4 = \lambda^2 - 2\lambda - 7 = 0$$

$$\lambda = 1 \pm 2\sqrt{2}$$

即 $\lambda_1 = 1 - 2\sqrt{2} \approx -1.83$，$\lambda_2 = 1 + 2\sqrt{2} \approx 3.83$。

**第三步：求特征向量（略去归一化细节）**，设正交矩阵为 $Q = [\mathbf{q}_1 \mid \mathbf{q}_2]$。

**第四步：标准形。** 令 $\mathbf{x} = Q\mathbf{y}$，得：

$$Q(x_1, x_2) = (1 - 2\sqrt{2})\, y_1^2 + (1 + 2\sqrt{2})\, y_2^2$$

交叉项已全部消去。$\lambda_1 < 0 < \lambda_2$ 说明该二次型是**不定的**（等高线为双曲线族）。

### 配方法（另一种标准化方式）

正交变换保持几何形状（旋转不改变距离），但有时我们不需要正交变换，只需消去交叉项。**配方法（completing the square）**通过非正交的线性变换也能达到标准形，且系数变为 $+1$、$-1$、$0$：

**例 23.4**：$Q = x_1^2 + 2x_1 x_2 + 3x_2^2$

$$Q = (x_1 + x_2)^2 - x_2^2 + 3x_2^2 = (x_1 + x_2)^2 + 2x_2^2$$

令 $y_1 = x_1 + x_2$，$y_2 = x_2$，则 $Q = y_1^2 + 2y_2^2$（标准形，两个正系数）。

> **注意**：配方法给出的标准形系数可以任意正实数（不一定是特征值），但由惯性定理（下节），正负系数的**个数**是唯一确定的。

---

## 23.3 惯性定理

### Sylvester 惯性定理

二次型的标准形不唯一——不同的线性变换可能给出不同的系数。然而，有一个深刻的不变量：

**定理（Sylvester 惯性定理，1852）**：对实二次型 $Q = \mathbf{x}^T A \mathbf{x}$，无论通过何种非退化线性变换化为标准形

$$Q = d_1 z_1^2 + d_2 z_2^2 + \cdots + d_n z_n^2 \quad (d_i \in \mathbb{R})$$

标准形中**正系数的个数** $p$（正惯性指数）和**负系数的个数** $q$（负惯性指数）是不变的（与所选变换无关）。

**惯性指数的定义**：

- **正惯性指数** $p$：标准形中正系数（$d_i > 0$）的个数
- **负惯性指数** $q$：标准形中负系数（$d_i < 0$）的个数
- **零惯性指数** $r - p - q$（$r = \text{rank}(A)$）：系数为零的个数

**符号（signature）**：有序对 $(p, q)$ 称为二次型的**符号差**，完整刻画了二次型的"类型"。

**与特征值的关系**：由正交变换得到的标准形系数恰好是 $A$ 的特征值，因此：

$$p = \text{正特征值的个数}, \quad q = \text{负特征值的个数}$$

### 合同变换与合同等价

**定义**：若存在非退化矩阵 $C$，使得 $B = C^T A C$，则称 $A$ 与 $B$ **合同（congruent）**，记作 $A \simeq B$。

合同变换 $\mathbf{x} = C\mathbf{y}$ 将 $\mathbf{x}^T A \mathbf{x}$ 变为 $\mathbf{y}^T B \mathbf{y}$（$B = C^T A C$）。

**惯性定理的等价表述**：两个实对称矩阵合同，当且仅当它们有相同的秩和相同的正惯性指数（即相同的符号差 $(p, q)$）。

### 二次型的完整分类

| 符号差 $(p, q)$ | 条件 | 类型 | 典型等高线 |
|:---:|:---|:---|:---|
| $(n, 0)$ | $p = n, q = 0$ | **正定（positive definite）** | 椭球 |
| $(n-r, 0)$ | $p < n, q = 0$（$r < n$）| **正半定（positive semidefinite）** | 退化椭球 |
| $(0, n)$ | $p = 0, q = n$ | **负定（negative definite）** | 倒椭球 |
| $(0, n-r)$ | $p = 0, q < n$（$r < n$）| **负半定（negative semidefinite）** | 退化倒椭球 |
| $(p, q)$，$p,q > 0$ | 正负均有 | **不定（indefinite）** | 双曲面/马鞍面 |

**例 23.5**：判断 $A = \begin{pmatrix}2&1&0\\1&3&1\\0&1&2\end{pmatrix}$ 的类型。

特征值：$\lambda_1 \approx 1.27$，$\lambda_2 \approx 2$，$\lambda_3 \approx 3.73$（均为正）。

符号差 $(3, 0)$，**正定**。

**例 23.6**：$B = \begin{pmatrix}1&2\\2&1\end{pmatrix}$ 的特征值为 $\lambda = 3$ 和 $\lambda = -1$，符号差 $(1, 1)$，**不定**。

---

## 23.4 正定二次型

### 正定的几何直觉

正定二次型 $Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} > 0$（对 $\mathbf{x} \neq \mathbf{0}$）在几何上是一个"多维碗形曲面"：

- 原点是唯一的全局最低点（函数值为 $0$）；
- 向任何方向移动，函数值都严格增加；
- 等高面 $\{Q(\mathbf{x}) = c\}$（$c > 0$）是以原点为中心的**椭球面**，半轴长度为 $1/\sqrt{\lambda_i}$（$\lambda_i$ 是 $A$ 的特征值）。

**标准椭球**：正定矩阵 $A$ 定义的单位"能量球" $\{\mathbf{x} : \mathbf{x}^T A \mathbf{x} = 1\}$ 是一个椭球，其主轴方向是 $A$ 的特征向量，主轴长度是对应特征值平方根的倒数。

### 等价判定条件

**定理**：对实对称矩阵 $A$，以下条件等价：

**(1) 二次型正定**：$\mathbf{x}^T A \mathbf{x} > 0$ 对所有 $\mathbf{x} \neq \mathbf{0}$

**(2) 特征值全正**：$\lambda_1, \ldots, \lambda_n > 0$

**(3) 顺序主子式全正（Sylvester 准则）**：

$$\Delta_1 = a_{11} > 0, \quad \Delta_2 = \det\begin{pmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{pmatrix} > 0, \quad \ldots, \quad \Delta_n = \det(A) > 0$$

**(4) Cholesky 分解存在**：$A = LL^T$，其中 $L$ 是对角元全正的下三角矩阵

**(5) 作为 Gram 矩阵**：存在列满秩矩阵 $B$ 使得 $A = B^T B$

**各类型的顺序主子式特征**：

| 类型 | 特征值条件 | 顺序主子式 |
|:---|:---|:---|
| 正定 | 全 $> 0$ | 全 $> 0$ |
| 负定 | 全 $< 0$ | 交错变号（$\Delta_1 < 0, \Delta_2 > 0, \ldots$）|
| 半正定 | 全 $\geq 0$，有零 | 全 $\geq 0$，且 $\det(A) = 0$ |
| 不定 | 有正有负 | 不满足上述任一模式 |

### 例题：完整判断流程

**例 23.7**：判断 $Q = 2x_1^2 + x_2^2 + 3x_3^2 - 2x_1 x_2 + 2x_1 x_3$ 的正定性。

矩阵表示：

$$A = \begin{pmatrix} 2 & -1 & 1 \\ -1 & 1 & 0 \\ 1 & 0 & 3 \end{pmatrix}$$

顺序主子式：

$$\Delta_1 = 2 > 0$$

$$\Delta_2 = \det\begin{pmatrix}2 & -1 \\ -1 & 1\end{pmatrix} = 2 - 1 = 1 > 0$$

$$\Delta_3 = \det(A) = 2(3-0) - (-1)(-3-0) + 1(0-1) = 6 - 3 - 1 = 2 > 0$$

三个顺序主子式全正，$A \succ 0$，$Q$ **正定**。

### 正定矩阵的运算性质

- 若 $A \succ 0$，则 $A^{-1} \succ 0$（逆正定）
- 若 $A \succ 0$ 且 $B \succ 0$，则 $A + B \succ 0$（和正定）
- 若 $A \succ 0$ 且 $C$ 列满秩，则 $C^T A C \succ 0$（合同变换保持正定）
- 正定矩阵的子矩阵（主子矩阵）也正定

---

## 23.5 约束优化

### 单位球面上的二次型

考虑约束优化问题：

$$\max_{\|\mathbf{x}\| = 1} Q(\mathbf{x}) = \max_{\|\mathbf{x}\| = 1} \mathbf{x}^T A \mathbf{x}$$

这是在 $n$ 维单位球面上寻找二次型的最大值点——直接求导并不直接（有约束），如何处理？

**解答**：令 $\mathbf{y} = Q^T \mathbf{x}$（$Q$ 是 $A$ 的正交特征向量矩阵），由 $Q^T Q = I$ 知 $\|\mathbf{y}\| = \|\mathbf{x}\| = 1$，且：

$$\mathbf{x}^T A \mathbf{x} = \mathbf{y}^T \Lambda \mathbf{y} = \sum_{i=1}^{n} \lambda_i y_i^2$$

在约束 $\sum_i y_i^2 = 1$ 下，利用凸组合不等式：

$$\lambda_{\min} = \lambda_{\min} \sum_i y_i^2 \leq \sum_i \lambda_i y_i^2 \leq \lambda_{\max} \sum_i y_i^2 = \lambda_{\max}$$

**定理（Rayleigh 商极值定理）**：

$$\boxed{\min_{\|\mathbf{x}\|=1} \mathbf{x}^T A \mathbf{x} = \lambda_{\min}(A), \quad \text{在 } \mathbf{x} = \mathbf{q}_{\min} \text{ 时取到}}$$

$$\boxed{\max_{\|\mathbf{x}\|=1} \mathbf{x}^T A \mathbf{x} = \lambda_{\max}(A), \quad \text{在 } \mathbf{x} = \mathbf{q}_{\max} \text{ 时取到}}$$

其中 $\mathbf{q}_{\min}$，$\mathbf{q}_{\max}$ 分别是 $A$ 的最小和最大特征值对应的标准化特征向量。

### 更一般的约束

**约束于椭球面**：问题 $\min_{\mathbf{x}^T B \mathbf{x} = 1} \mathbf{x}^T A \mathbf{x}$（$B \succ 0$）通过变量替换 $\mathbf{z} = B^{1/2} \mathbf{x}$ 化为标准形式，等价于求矩阵笔形 $(A, B)$ 的广义特征值问题：

$$A \mathbf{x} = \lambda B \mathbf{x}$$

**二次型之比**：Rayleigh 商 $R_A(\mathbf{x}) = \dfrac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}$ 与约束形式等价，其驻点方程 $\nabla R_A = \mathbf{0}$ 正是特征方程 $A\mathbf{x} = \lambda \mathbf{x}$。

### Courant-Fischer 极大极小定理

第 $k$ 小特征值的变分刻画（$k = 1, 2, \ldots, n$）：

$$\lambda_k = \min_{\substack{S \subseteq \mathbb{R}^n \\ \dim(S) = k}} \max_{\mathbf{x} \in S,\, \|\mathbf{x}\|=1} \mathbf{x}^T A \mathbf{x}$$

直觉：在所有 $k$ 维子空间中，先找每个子空间上的最大值，再对所有子空间取最小——第 $k$ 个特征值是这个"极大极小"操作的结果。

### 几何应用：椭圆的主轴

设二次型 $Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = 1$（$A \succ 0$）定义了一个椭圆（$n=2$）。则：

- 椭圆的**主轴方向**是 $A$ 的特征向量 $\mathbf{q}_1, \mathbf{q}_2$；
- 主轴**半径**（轴长）分别为 $1/\sqrt{\lambda_1}$ 和 $1/\sqrt{\lambda_2}$；
- 椭圆面积 $= \pi / \sqrt{\lambda_1 \lambda_2} = \pi / \sqrt{\det A}$。

**例 23.8**：$Q(x_1, x_2) = 5x_1^2 + 4x_1 x_2 + 2x_2^2 = 1$，求椭圆的主轴。

矩阵 $A = \begin{pmatrix}5&2\\2&2\end{pmatrix}$，特征值 $\lambda = \dfrac{7 \pm \sqrt{17}}{2}$（约 $1.44$ 和 $5.56$）。

主轴半径分别约为 $1/\sqrt{1.44} \approx 0.83$ 和 $1/\sqrt{5.56} \approx 0.42$，主轴方向为对应特征向量。

---

## 本章小结

- **二次型** $Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ 用实对称矩阵 $A$ 唯一表示：对角元为纯二次项系数，非对角元 $a_{ij} = a_{ji}$ 等于交叉项系数的一半。

- **正交标准化**：通过正交变换 $\mathbf{x} = Q\mathbf{y}$（$Q$ 为 $A$ 的特征向量矩阵），二次型化为标准形 $\sum \lambda_i y_i^2$，系数为 $A$ 的特征值，交叉项全消。

- **惯性定理（Sylvester）**：无论用何种非退化线性变换化标准形，正系数个数 $p$（正惯性指数）和负系数个数 $q$（负惯性指数）不变。符号差 $(p, q)$ 是二次型的本质不变量，合同等价类由 $(p, q)$ 完全刻画。

- **正定性分类**：
  - $A \succ 0$（正定）$\Leftrightarrow$ 所有特征值 $> 0$ $\Leftrightarrow$ 所有顺序主子式 $> 0$ $\Leftrightarrow$ 存在 Cholesky 分解；
  - $A \succeq 0$（半正定）$\Leftrightarrow$ 所有特征值 $\geq 0$ $\Leftrightarrow$ 存在 $B$ 使 $A = B^T B$；
  - 正负特征值共存 $\Rightarrow$ 不定（有鞍点方向）。

- **约束优化与特征值**：在单位球面上，$\mathbf{x}^T A \mathbf{x}$ 的最大值（最小值）是 $A$ 的最大（最小）特征值，最优方向是对应特征向量。这是 Rayleigh 商理论的核心，也是理解损失函数曲率的几何基础。

**下一章预告**：奇异值分解（SVD）——将谱定理推广到任意形状矩阵，揭示数据矩阵的"几何骨架"，是现代数据科学最核心的工具。

---

## 深度学习应用：损失函数的曲率分析

### 损失曲面的二次近似

设神经网络损失函数 $\mathcal{L}(\boldsymbol{\theta})$，在临界点 $\boldsymbol{\theta}^*$（$\nabla \mathcal{L}(\boldsymbol{\theta}^*) = \mathbf{0}$）附近做 Taylor 展开：

$$\mathcal{L}(\boldsymbol{\theta}^* + \Delta\boldsymbol{\theta}) \approx \mathcal{L}(\boldsymbol{\theta}^*) + \frac{1}{2} \Delta\boldsymbol{\theta}^T H \Delta\boldsymbol{\theta}$$

其中 **Hessian 矩阵** $H_{ij} = \dfrac{\partial^2 \mathcal{L}}{\partial \theta_i \partial \theta_j}$ 是实对称矩阵（由 Schwarz 定理），本章的二次型理论完全适用。

$\Delta\boldsymbol{\theta}^T H \Delta\boldsymbol{\theta}$ 正是以 $H$ 为矩阵的二次型，它决定了离开临界点时损失的变化方向和速率。

### Hessian 的符号差与临界点类型

利用谱定理，$H = Q\Lambda Q^T$，令 $\mathbf{z} = Q^T \Delta\boldsymbol{\theta}$：

$$\Delta\boldsymbol{\theta}^T H \Delta\boldsymbol{\theta} = \mathbf{z}^T \Lambda \mathbf{z} = \sum_{i=1}^{p} \lambda_i z_i^2$$

Hessian 的符号差 $(p, q)$ 直接决定临界点性质：

| Hessian 符号差 $(p, q)$ | 几何意义 | 临界点类型 |
|:---:|:---|:---|
| $(n, 0)$，$H \succ 0$ | 各方向均上弯，碗形 | **局部极小值** |
| $(0, n)$，$H \prec 0$ | 各方向均下弯，倒碗形 | **局部极大值** |
| $(p, q)$，$p,q > 0$ | 部分方向上弯、部分下弯 | **鞍点（saddle point）** |
| 某 $\lambda_i = 0$ | 有平坦方向 | **退化临界点** |

**高维空间中鞍点占主导**：对于参数量为 $p$ 的神经网络，使所有 $p$ 个特征值均为正（局部极小值）的概率随 $p$ 指数减小。实验和理论均表明，大型神经网络的"坏临界点"几乎都是鞍点，而非局部极大值。SGD 的随机噪声天然帮助模型逃离鞍点，沿负曲率方向下降。

### 曲率与学习率的精确联系

沿单位方向 $\hat{\mathbf{v}}$ 的**方向曲率**就是该方向上的 Rayleigh 商：

$$\kappa_{\hat{\mathbf{v}}} = \hat{\mathbf{v}}^T H \hat{\mathbf{v}} = R_H(\hat{\mathbf{v}})$$

- 曲率最大方向 $\hat{\mathbf{v}} = \mathbf{q}_{\max}$，曲率值 $= \lambda_{\max}$：学习率必须满足 $\eta < 2/\lambda_{\max}$，否则沿该方向发散；
- 曲率最小正方向 $\hat{\mathbf{v}} = \mathbf{q}_{\min}$，曲率值 $= \lambda_{\min}$：该方向收敛最慢，是梯度下降的"瓶颈"；
- 条件数 $\kappa(H) = \lambda_{\max}/\lambda_{\min}$ 量化了损失面的"椭圆度"，$\kappa$ 越大椭圆越扁，梯度下降越曲折。

### PyTorch 代码：可视化损失曲面与 Hessian 曲率

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ── 1. 定义二次型损失曲面 ────────────────────────────────────────
# L(x, y) = x^T A x，A 的条件数可调
# 用正定矩阵 A = [[a, b], [b, c]] 控制曲面形状

def make_loss_fn(eigenvalues: list, angle: float = 0.3):
    """
    构造一个二维二次型损失函数。
    eigenvalues: [lambda1, lambda2]（正定，决定两个主轴方向的曲率）
    angle: 旋转角度（弧度），使主轴不与坐标轴重合，产生交叉项
    """
    lam1, lam2 = eigenvalues
    # 旋转矩阵 Q
    c, s = np.cos(angle), np.sin(angle)
    Q = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    lam = torch.tensor([[lam1, 0], [0, lam2]], dtype=torch.float64)
    # A = Q Λ Q^T
    A = Q @ lam @ Q.T

    def loss_fn(params: torch.Tensor) -> torch.Tensor:
        return 0.5 * (params @ A @ params)

    return loss_fn, A

# ── 2. 计算 Hessian（对小维度精确计算）────────────────────────────
def exact_hessian(func, params: torch.Tensor) -> torch.Tensor:
    n = params.numel()
    H = torch.zeros(n, n, dtype=params.dtype)
    p = params.clone().requires_grad_(True)
    grad = torch.autograd.grad(func(p), p, create_graph=True)[0]
    for i in range(n):
        row = torch.autograd.grad(grad[i], p, retain_graph=True)[0]
        H[i] = row.detach()
    return H

# ── 3. 可视化：损失曲面 + 等高线 + 特征向量主轴 ─────────────────
def visualize_loss_landscape(eigenvalues, angle=0.3, title="损失曲面"):
    loss_fn, A = make_loss_fn(eigenvalues, angle)

    # 计算 Hessian 及其谱分解
    theta0 = torch.zeros(2, dtype=torch.float64)
    H = exact_hessian(loss_fn, theta0)
    eigvals, eigvecs = torch.linalg.eigh(H)  # 专用于实对称矩阵

    print(f"\n{title}")
    print(f"  Hessian 特征值: λ₁={eigvals[0].item():.3f}, λ₂={eigvals[1].item():.3f}")
    print(f"  条件数 κ = {eigvals[1].item()/eigvals[0].item():.2f}")
    print(f"  特征向量 q₁ = [{eigvecs[0,0].item():.3f}, {eigvecs[1,0].item():.3f}]")
    print(f"  特征向量 q₂ = [{eigvecs[0,1].item():.3f}, {eigvecs[1,1].item():.3f}]")

    # 生成网格
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            pt = torch.tensor([X[j, i], Y[j, i]], dtype=torch.float64)
            Z[j, i] = loss_fn(pt).item()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：3D 损失曲面
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.85, linewidth=0)
    ax3d.set_xlabel("x₁"); ax3d.set_ylabel("x₂"); ax3d.set_zlabel("L")
    ax3d.set_title(f"{title}（3D）")

    # 右图：等高线 + Hessian 主轴
    ax2d = axes[1]
    levels = np.linspace(0, Z.max() * 0.8, 15)
    cs = ax2d.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
    ax2d.contour(X, Y, Z, levels=levels, colors='white', linewidths=0.5, alpha=0.5)
    plt.colorbar(cs, ax=ax2d, label='损失值')

    # 绘制特征向量（主轴方向），缩放到椭圆半径 1/√λ
    for k in range(2):
        lam_k = eigvals[k].item()
        vec = eigvecs[:, k].numpy()
        scale = 1.0 / np.sqrt(lam_k) * 2  # 放大 2 倍便于可视化
        ax2d.annotate("", xy=(scale * vec[0], scale * vec[1]),
                      xytext=(0, 0),
                      arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax2d.annotate("", xy=(-scale * vec[0], -scale * vec[1]),
                      xytext=(0, 0),
                      arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax2d.text(scale * vec[0] * 1.1, scale * vec[1] * 1.1,
                  f"q₍{k+1}₎\nλ={lam_k:.2f}", color='red', fontsize=9)

    ax2d.set_xlim(-3, 3); ax2d.set_ylim(-3, 3)
    ax2d.set_aspect('equal')
    ax2d.set_xlabel("x₁"); ax2d.set_ylabel("x₂")
    ax2d.set_title(f"{title}（等高线）\nκ = {eigvals[1].item()/eigvals[0].item():.1f}")

    plt.tight_layout()
    plt.savefig(f"loss_landscape_{title}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  图像已保存：loss_landscape_{title}.png")

# 运行：对比低条件数（接近球形）和高条件数（扁椭圆）两种情形
visualize_loss_landscape([1.0, 1.2], angle=0.4, title="低条件数(κ≈1.2)")
visualize_loss_landscape([0.5, 50.0], angle=0.3, title="高条件数(κ=100)")

# ── 4. 鞍点演示：不定 Hessian ──────────────────────────────────────
def saddle_point_demo():
    """
    L(x, y) = x² - y²（标准鞍点，Hessian = diag(2, -2)，不定）
    展示不定二次型的马鞍面
    """
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2  # 不定二次型

    fig = plt.figure(figsize=(12, 5))

    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.plot_surface(X, Y, Z, cmap='RdYlBu', alpha=0.85)
    ax3d.set_xlabel("x₁"); ax3d.set_ylabel("x₂"); ax3d.set_zlabel("L")
    ax3d.set_title("鞍点曲面：L = x₁² - x₂²\nHessian 符号差 (1,1)，不定")

    ax2d = fig.add_subplot(122)
    levels = np.linspace(-3, 3, 20)
    cs = ax2d.contourf(X, Y, Z, levels=levels, cmap='RdYlBu')
    ax2d.contour(X, Y, Z, levels=[0], colors='black', linewidths=2,
                 linestyles='--')  # 零等高线（分水岭）
    plt.colorbar(cs, ax=ax2d, label='损失值')
    ax2d.annotate("鞍点\n(原点)", xy=(0, 0), xytext=(0.5, 0.5),
                  arrowprops=dict(arrowstyle='->', color='black'),
                  fontsize=10)
    ax2d.set_xlabel("x₁"); ax2d.set_ylabel("x₂")
    ax2d.set_title("等高线（黑色虚线为零面）\n沿 x₁ 上弯（λ=2），沿 x₂ 下弯（λ=-2）")
    plt.tight_layout()
    plt.savefig("saddle_point.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("鞍点图像已保存：saddle_point.png")

saddle_point_demo()

# ── 5. 梯度下降在高条件数损失面上的收敛轨迹 ───────────────────────
def gradient_descent_trajectory(kappa: float, n_steps: int = 100,
                                 lr: float = None):
    """
    在椭圆损失面 L = 0.5*(x² + kappa*y²) 上跑梯度下降，
    绘制参数轨迹，直观展示"锯齿"现象
    """
    if lr is None:
        lr = 2.0 / (1.0 + kappa)  # 最优学习率

    trajectory = [(2.0, 2.0)]  # 初始点
    x, y = 2.0, 2.0
    losses = []
    for _ in range(n_steps):
        L = 0.5 * (x**2 + kappa * y**2)
        losses.append(L)
        gx, gy = x, kappa * y  # 精确梯度
        x -= lr * gx
        y -= lr * gy
        trajectory.append((x, y))

    traj = np.array(trajectory)

    # 绘制等高线 + 轨迹
    xg = np.linspace(-2.5, 2.5, 200)
    yg = np.linspace(-2.5, 2.5, 200)
    Xg, Yg = np.meshgrid(xg, yg)
    Zg = 0.5 * (Xg**2 + kappa * Yg**2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    cs = ax.contourf(Xg, Yg, Zg, levels=20, cmap='viridis', alpha=0.7)
    ax.contour(Xg, Yg, Zg, levels=20, colors='white', linewidths=0.4, alpha=0.4)
    ax.plot(traj[:, 0], traj[:, 1], 'ro-', markersize=3, linewidth=1.5,
            label=f"GD 轨迹（η={lr:.4f}）")
    ax.plot(0, 0, 'w*', markersize=15, label="全局极小值")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.set_title(f"条件数 κ={kappa}\n梯度下降轨迹（{n_steps} 步）")
    ax.legend(fontsize=8)

    axes[1].semilogy(losses, 'b-', linewidth=1.5)
    axes[1].set_xlabel("迭代步数"); axes[1].set_ylabel("损失（对数尺度）")
    axes[1].set_title(f"损失收敛曲线\n理论收敛率 ρ*={(kappa-1)/(kappa+1):.4f}")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"gd_trajectory_kappa{int(kappa)}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"梯度下降轨迹图（κ={kappa}）已保存")

gradient_descent_trajectory(kappa=2.0,   n_steps=30)   # 低条件数：快速收敛
gradient_descent_trajectory(kappa=50.0,  n_steps=200)  # 高条件数：锯齿轨迹
```

**代码解读**：

- **第 1-3 部分（损失曲面可视化）**：构造了两种典型二次型损失面——低条件数的"近球形碗"和高条件数的"极扁椭圆碗"。用 `torch.linalg.eigh` 对 Hessian 做谱分解，将主轴（特征向量方向）和半径（$1/\sqrt{\lambda_i}$）叠加在等高线图上，直观展示正定二次型的椭球等高线。

- **第 4 部分（鞍点演示）**：$L = x_1^2 - x_2^2$ 是不定二次型（Hessian 符号差 $(1,1)$），马鞍面在 $x_1$ 方向上弯（正特征值 $\lambda_1 = 2$）、在 $x_2$ 方向下弯（负特征值 $\lambda_2 = -2$）。零等高线（黑色虚线）将鞍点的"上弯"和"下弯"区域分隔开，清晰展现不定二次型的几何结构。

- **第 5 部分（收敛轨迹）**：在不同条件数的椭圆损失面上运行梯度下降，$\kappa = 2$ 时轨迹接近直线，快速收敛；$\kappa = 50$ 时出现明显"锯齿"现象——这是因为在扁椭圆中，梯度方向与最优下降方向夹角较大，需要来回迂回才能到达极值点，印证了条件数对收敛速度的决定性影响。

| 二次型概念 | 深度学习对应物 | 实际影响 |
|:---|:---|:---|
| 正定二次型（$H \succ 0$）| 局部极小值 | 稳定最优点，梯度下降收敛 |
| 不定二次型（$H$ 有正负特征值）| 鞍点 | 需要逃脱，SGD 噪声起作用 |
| 正惯性指数 $p$ | 上弯方向个数 | 逃脱方向 $= n - p$（负特征值数 $q$）|
| 最大特征值 $\lambda_{\max}$ | 最大曲率 | 学习率上界 $\eta < 2/\lambda_{\max}$ |
| 条件数 $\kappa = \lambda_{\max}/\lambda_{\min}$ | 椭圆扁度 | 决定梯度下降收敛速度 |
| Rayleigh 商 $R_H(\mathbf{v})$ | 方向曲率 | 各方向步长的理论依据 |

---

## 练习题

**练习 1**（基础——矩阵表示与标准形）

给定二次型 $Q(x_1, x_2, x_3) = x_1^2 + 4x_2^2 - 2x_3^2 + 4x_1 x_2 - 6x_1 x_3 + 2x_2 x_3$。

（a）写出对应的实对称矩阵 $A$。

（b）计算 $A$ 的特征值，确定正惯性指数 $p$ 和负惯性指数 $q$。

（c）写出二次型经正交变换后的标准形，说明该二次型属于哪类（正定、负定、不定等）。

---

**练习 2**（基础——正定性判断）

判断下列二次型的正定性，给出完整判断依据：

（a）$Q_1 = 2x_1^2 + x_2^2 + 3x_3^2 + 2x_1 x_2$

（b）$Q_2 = x_1^2 + 4x_1 x_2 + x_2^2$

（c）$Q_3 = x_1^2 + 2x_2^2 + 3x_3^2 - 2x_1 x_2 + 2x_1 x_3 - 4x_2 x_3$

---

**练习 3**（中等——惯性定理与合同）

（a）证明：$A = \begin{pmatrix}1&2\\2&1\end{pmatrix}$ 与 $B = \begin{pmatrix}1&0\\0&-3\end{pmatrix}$ 合同，并找出合同变换矩阵 $C$（满足 $B = C^T A C$）。

（b）矩阵 $M = \begin{pmatrix}3&0&0\\0&-1&0\\0&0&2\end{pmatrix}$ 与 $N = \begin{pmatrix}1&1&0\\1&2&1\\0&1&-1\end{pmatrix}$ 是否合同？说明理由。

---

**练习 4**（中等——约束优化）

设 $A = \begin{pmatrix}4&2\\2&3\end{pmatrix}$，考虑约束优化问题：

$$\max_{\|\mathbf{x}\| = 1} \mathbf{x}^T A \mathbf{x}, \quad \min_{\|\mathbf{x}\| = 1} \mathbf{x}^T A \mathbf{x}$$

（a）求 $A$ 的特征值与标准正交特征向量，给出最大值和最小值。

（b）在约束 $\mathbf{x}^T \mathbf{x} = 1$ 下，用 Lagrange 乘数法求极值，验证与 (a) 的结论一致。

（c）椭圆 $\{(x_1, x_2) : \mathbf{x}^T A \mathbf{x} = 1\}$ 的两条主轴方向和半径各是多少？

---

**练习 5**（进阶——综合：二次型、Hessian、优化）

考虑二维损失函数 $\mathcal{L}(x_1, x_2) = 3x_1^2 + 2x_1 x_2 + 2x_2^2 - 4x_1 - 6x_2 + 5$。

（a）将 $\mathcal{L}$ 写成 $\mathcal{L}(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} + \mathbf{b}^T \mathbf{x} + c$ 的形式，给出 $A$、$\mathbf{b}$、$c$。

（b）求所有临界点（令 $\nabla \mathcal{L} = \mathbf{0}$），判断临界点类型。

（c）计算 Hessian（即 $2A$），求其特征值，确定符号差 $(p, q)$。

（d）用"配方法"将 $\mathcal{L}$ 改写为关于 $(\mathbf{x} - \mathbf{x}^*)$ 的纯二次型（其中 $\mathbf{x}^*$ 是极小值点），与正交标准形进行比较。

（e）若在极小值附近用梯度下降，最优学习率 $\eta^*$、理论收敛率 $\rho^*$ 和条件数 $\kappa$ 各是多少？预估收敛到 $0.1\%$ 精度所需步数。

---

## 练习答案

<details>
<summary>点击展开 练习1 答案</summary>

**（a）实对称矩阵**

对角元直接读取系数，非对角元为交叉项系数的一半：

$$A = \begin{pmatrix} 1 & 2 & -3 \\ 2 & 4 & 1 \\ -3 & 1 & -2 \end{pmatrix}$$

验证：$a_{12} = a_{21} = \dfrac{4}{2} = 2$，$a_{13} = a_{31} = \dfrac{-6}{2} = -3$，$a_{23} = a_{32} = \dfrac{2}{2} = 1$。

**（b）特征值与惯性指数**

计算特征多项式 $\det(A - \lambda I) = 0$：

展开（利用行化简或软件辅助）：

$$\det(A - \lambda I) = -\lambda^3 + 3\lambda^2 + 21\lambda - 14$$

三个实根约为 $\lambda_1 \approx -3.96$，$\lambda_2 \approx 0.61$，$\lambda_3 \approx 5.79$。（实际数值可由数值方法或 Numpy 求得。）

正惯性指数 $p = 2$（正特征值个数），负惯性指数 $q = 1$（负特征值个数）。

**（c）标准形与分类**

经正交变换 $\mathbf{x} = Q\mathbf{y}$ 后，标准形为：

$$Q \approx -3.96 \, y_1^2 + 0.61 \, y_2^2 + 5.79 \, y_3^2$$

符号差 $(p, q) = (2, 1)$，有正有负特征值，该二次型为**不定型**（等高面为双叶双曲面族）。

</details>

<details>
<summary>点击展开 练习2 答案</summary>

**（a）$Q_1 = 2x_1^2 + x_2^2 + 3x_3^2 + 2x_1 x_2$**

矩阵 $A_1 = \begin{pmatrix}2&1&0\\1&1&0\\0&0&3\end{pmatrix}$

顺序主子式：

$$\Delta_1 = 2 > 0, \quad \Delta_2 = \det\begin{pmatrix}2&1\\1&1\end{pmatrix} = 2 - 1 = 1 > 0, \quad \Delta_3 = 3 \cdot \Delta_2 = 3 > 0$$

（$A_1$ 是块对角矩阵，$\det(A_1) = \det\begin{pmatrix}2&1\\1&1\end{pmatrix} \cdot 3 = 1 \cdot 3 = 3$。）

三个顺序主子式全正，$Q_1$ **正定**。

---

**（b）$Q_2 = x_1^2 + 4x_1 x_2 + x_2^2$**

矩阵 $A_2 = \begin{pmatrix}1&2\\2&1\end{pmatrix}$

$\det(A_2) = 1 - 4 = -3 < 0$，顺序主子式不全正。

特征值：$\lambda_1 = -1 < 0$，$\lambda_2 = 3 > 0$，有正有负。

$Q_2$ **不定**（取 $\mathbf{x} = (1,1)^T$：$Q_2 = 1 + 4 + 1 = 6 > 0$；取 $\mathbf{x} = (1,-1)^T$：$Q_2 = 1 - 4 + 1 = -2 < 0$，验证不定性）。

---

**（c）$Q_3 = x_1^2 + 2x_2^2 + 3x_3^2 - 2x_1 x_2 + 2x_1 x_3 - 4x_2 x_3$**

矩阵 $A_3 = \begin{pmatrix}1&-1&1\\-1&2&-2\\1&-2&3\end{pmatrix}$

顺序主子式：

$$\Delta_1 = 1 > 0$$

$$\Delta_2 = \det\begin{pmatrix}1&-1\\-1&2\end{pmatrix} = 2 - 1 = 1 > 0$$

$$\Delta_3 = \det(A_3)$$

沿第一行展开：

$$= 1 \cdot (6-4) - (-1)(-3+2) + 1 \cdot (2-2) = 2 - 1 + 0 = 1 > 0$$

三个顺序主子式全正，$Q_3$ **正定**。

</details>

<details>
<summary>点击展开 练习3 答案</summary>

**（a）$A$ 与 $B$ 的合同关系**

$A = \begin{pmatrix}1&2\\2&1\end{pmatrix}$ 的特征值：$\lambda = 1 \pm 2$，即 $\lambda_1 = -1$，$\lambda_2 = 3$。

$B = \begin{pmatrix}1&0\\0&-3\end{pmatrix}$ 的特征值：$1$ 和 $-3$。

两者均有一正一负特征值，符号差均为 $(1, 1)$，秩均为 $2$，故合同。

寻找合同变换 $C$（使 $C^T A C = B$）：

$A$ 的谱分解：$A = Q \Lambda Q^T$，其中 $\Lambda = \begin{pmatrix}-1&0\\0&3\end{pmatrix}$，$Q$ 为正交特征向量矩阵。

构造 $C = Q \begin{pmatrix}1&0\\0&1/\sqrt{3}\end{pmatrix}$（将特征值 $\lambda_2 = 3$ 缩放为 $1$），则：

$$C^T A C = \begin{pmatrix}1/\sqrt{3}&0\\0&1\end{pmatrix} Q^T \cdot Q\Lambda Q^T \cdot Q \begin{pmatrix}1/\sqrt{3}&0\\0&1\end{pmatrix} = \begin{pmatrix}1/\sqrt{3}&0\\0&1\end{pmatrix}\begin{pmatrix}-1&0\\0&3\end{pmatrix}\begin{pmatrix}1/\sqrt{3}&0\\0&1\end{pmatrix} = \begin{pmatrix}-1/3&0\\0&3\end{pmatrix}$$

这不是 $B$，需调整缩放。更直接地，令 $\mathbf{c}_1 = \mathbf{q}_1$（$\lambda_1 = -1$ 的特征向量，已归一化），$\mathbf{c}_2 = \mathbf{q}_2 / \sqrt{3}$（缩放使 $\mathbf{c}_2^T A \mathbf{c}_2 = 3/3 = 1$），则 $C = [\mathbf{c}_1 \mid \mathbf{c}_2]$ 满足：

$$C^T A C = \begin{pmatrix}\mathbf{c}_1^T A \mathbf{c}_1 & 0 \\ 0 & \mathbf{c}_2^T A \mathbf{c}_2\end{pmatrix} = \begin{pmatrix}-1 & 0 \\ 0 & 1\end{pmatrix}$$

这与 $B' = \begin{pmatrix}-1&0\\0&1\end{pmatrix}$ 合同（符号差相同），但与 $B = \begin{pmatrix}1&0\\0&-3\end{pmatrix}$ 的主对角元交换位置后仍合同（列排列是合同变换的一种）。

**结论**：$A \simeq B$，合同关系成立（两者符号差均为 $(1,1)$，由惯性定理即可断言）。

---

**（b）$M$ 与 $N$ 的合同性**

$M = \text{diag}(3, -1, 2)$：特征值 $3, -1, 2$，符号差 $(p, q) = (2, 1)$，$\text{rank} = 3$。

$N = \begin{pmatrix}1&1&0\\1&2&1\\0&1&-1\end{pmatrix}$：

$\det(N) = 1(2\cdot(-1)-1\cdot 1) - 1(1\cdot(-1)-0) + 0 = 1(-3) - 1(-1) = -3 + 1 = -2 \neq 0$，$\text{rank}(N) = 3$。

$N$ 的顺序主子式：$\Delta_1 = 1 > 0$，$\Delta_2 = 2 - 1 = 1 > 0$，$\Delta_3 = \det(N) = -2 < 0$。

$N$ 的符号差：因 $\Delta_3 < 0$ 且 $\Delta_2 > 0$，$N$ 有负特征值，$q \geq 1$；因 $\Delta_1 > 0$，$N$ 有正特征值，$p \geq 2$；$p + q = 3$，故 $(p,q) = (2, 1)$。

$M$ 与 $N$ 的符号差均为 $(2, 1)$，秩均为 $3$，由惯性定理，**$M$ 与 $N$ 合同**。

</details>

<details>
<summary>点击展开 练习4 答案</summary>

**（a）特征值、特征向量与极值**

$A = \begin{pmatrix}4&2\\2&3\end{pmatrix}$

$$\det(A - \lambda I) = (4-\lambda)(3-\lambda) - 4 = \lambda^2 - 7\lambda + 8 = 0$$

$$\lambda = \frac{7 \pm \sqrt{49-32}}{2} = \frac{7 \pm \sqrt{17}}{2}$$

$\lambda_1 = \dfrac{7-\sqrt{17}}{2} \approx 1.44$（最小特征值），$\lambda_2 = \dfrac{7+\sqrt{17}}{2} \approx 5.56$（最大特征值）。

最大值 $= \lambda_2 = \dfrac{7+\sqrt{17}}{2}$，在 $\mathbf{x} = \mathbf{q}_2$（$\lambda_2$ 对应特征向量）时取到。

最小值 $= \lambda_1 = \dfrac{7-\sqrt{17}}{2}$，在 $\mathbf{x} = \mathbf{q}_1$ 时取到。

**（b）Lagrange 乘数法验证**

目标：$\max \mathbf{x}^T A \mathbf{x}$ 约束 $\|\mathbf{x}\|^2 = 1$。

Lagrangian：$\mathcal{L} = \mathbf{x}^T A \mathbf{x} - \lambda(\mathbf{x}^T \mathbf{x} - 1)$

KKT 条件：$\nabla_\mathbf{x} \mathcal{L} = 2A\mathbf{x} - 2\lambda \mathbf{x} = 0$，即 $A\mathbf{x} = \lambda \mathbf{x}$

这正是 $A$ 的特征方程！乘子 $\lambda$ 就是特征值，最优点 $\mathbf{x}^*$ 就是对应特征向量。约束 $\|\mathbf{x}\|=1$ 要求特征向量归一化。

与 (a) 完全一致。$\square$

**（c）椭圆主轴**

椭圆 $\mathbf{x}^T A \mathbf{x} = 1$ 的：

- 主轴方向：$A$ 的特征向量 $\mathbf{q}_1$（对应 $\lambda_1$）和 $\mathbf{q}_2$（对应 $\lambda_2$）
- 主轴半径：$r_1 = 1/\sqrt{\lambda_1} = \sqrt{2/(7-\sqrt{17})} \approx 0.83$，$r_2 = 1/\sqrt{\lambda_2} \approx 0.42$
- 椭圆面积：$\pi r_1 r_2 = \pi/\sqrt{\lambda_1 \lambda_2} = \pi/\sqrt{\det A} = \pi/\sqrt{8} \approx 1.11$

</details>

<details>
<summary>点击展开 练习5 答案</summary>

**（a）矩阵形式分解**

$$\mathcal{L}(x_1, x_2) = 3x_1^2 + 2x_1 x_2 + 2x_2^2 - 4x_1 - 6x_2 + 5$$

$$A = \begin{pmatrix}3 & 1 \\ 1 & 2\end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix}-4 \\ -6\end{pmatrix}, \quad c = 5$$

验证：$\mathbf{x}^T A \mathbf{x} = 3x_1^2 + x_1 x_2 + x_1 x_2 + 2x_2^2 = 3x_1^2 + 2x_1 x_2 + 2x_2^2$ ✓

**（b）临界点**

$$\nabla \mathcal{L} = 2A\mathbf{x} + \mathbf{b} = \mathbf{0} \Rightarrow 2A\mathbf{x} = -\mathbf{b} \Rightarrow A\mathbf{x} = \begin{pmatrix}2\\3\end{pmatrix}$$

$$\det(A) = 6 - 1 = 5, \quad A^{-1} = \frac{1}{5}\begin{pmatrix}2&-1\\-1&3\end{pmatrix}$$

$$\mathbf{x}^* = A^{-1}\begin{pmatrix}2\\3\end{pmatrix} = \frac{1}{5}\begin{pmatrix}4-3\\-2+9\end{pmatrix} = \frac{1}{5}\begin{pmatrix}1\\7\end{pmatrix} = \begin{pmatrix}0.2\\1.4\end{pmatrix}$$

唯一临界点 $\mathbf{x}^* = (0.2, 1.4)^T$。

**（c）Hessian 与符号差**

Hessian $H = 2A = \begin{pmatrix}6&2\\2&4\end{pmatrix}$（常数矩阵，因 $\mathcal{L}$ 是二次函数）。

特征值：$(6-\lambda)(4-\lambda) - 4 = \lambda^2 - 10\lambda + 20 = 0$

$$\lambda = 5 \pm \sqrt{5}, \quad \lambda_1 = 5 - \sqrt{5} \approx 2.76, \quad \lambda_2 = 5 + \sqrt{5} \approx 7.24$$

两个特征值均为正，符号差 $(p, q) = (2, 0)$，$H \succ 0$。

临界点 $\mathbf{x}^*$ 是**全局极小值点**（$\mathcal{L}$ 是严格凸二次函数）。

**（d）配方法与标准形**

$$\mathcal{L} = \mathbf{x}^T A \mathbf{x} + \mathbf{b}^T \mathbf{x} + c = (\mathbf{x} - \mathbf{x}^*)^T A (\mathbf{x} - \mathbf{x}^*) + \mathcal{L}(\mathbf{x}^*)$$

（利用 $2A\mathbf{x}^* = -\mathbf{b}$，可验证上式展开后与原式一致。）

令 $\Delta \mathbf{x} = \mathbf{x} - \mathbf{x}^*$，则：

$$\mathcal{L}(\mathbf{x}) = \Delta\mathbf{x}^T A \Delta\mathbf{x} + \mathcal{L}(\mathbf{x}^*)$$

其中 $\mathcal{L}(\mathbf{x}^*) = 3(0.04) + 2(0.28) + 2(1.96) - 0.8 - 8.4 + 5 = 0.12 + 0.56 + 3.92 - 0.8 - 8.4 + 5 = 0.4$

（注意：$\Delta\mathbf{x}^T A \Delta\mathbf{x}$ 是以 $A$ 为矩阵的纯二次型，而 $H = 2A$，故等价于 $\dfrac{1}{2}\Delta\mathbf{x}^T H \Delta\mathbf{x}$。）

通过正交变换 $\Delta\mathbf{x} = Q\mathbf{y}$ 进一步得到正交标准形：

$$\mathcal{L} = \frac{\lambda_1}{2} y_1^2 + \frac{\lambda_2}{2} y_2^2 + \mathcal{L}(\mathbf{x}^*) \approx 1.38 y_1^2 + 3.62 y_2^2 + 0.4$$

**（e）梯度下降参数**

Hessian $H = 2A$，特征值 $\lambda_1 = 5-\sqrt{5}$，$\lambda_2 = 5+\sqrt{5}$。

条件数：

$$\kappa = \frac{\lambda_2}{\lambda_1} = \frac{5+\sqrt{5}}{5-\sqrt{5}} = \frac{(5+\sqrt{5})^2}{20} = \frac{30+10\sqrt{5}}{20} = \frac{3+\sqrt{5}}{2} \approx 2.618$$

（有趣：$\kappa = \phi^2$，其中 $\phi = (1+\sqrt{5})/2$ 是黄金比例！）

最优学习率：

$$\eta^* = \frac{2}{\lambda_1 + \lambda_2} = \frac{2}{10} = 0.2$$

理论最坏收敛率：

$$\rho^* = \frac{\kappa - 1}{\kappa + 1} = \frac{\sqrt{5} - 1}{\sqrt{5} + 1} = \frac{(\sqrt{5}-1)^2}{4} = \frac{6 - 2\sqrt{5}}{4} = \frac{3 - \sqrt{5}}{2} \approx 0.382$$

（同样是黄金比例的倒数：$\rho^* = 1/\phi^2$。）

收敛到 $0.1\%$（误差缩小到初始的 $1/1000$）所需步数：

$$(\rho^*)^N = 0.001 \Rightarrow N = \frac{\ln 0.001}{\ln \rho^*} = \frac{-6.908}{\ln(0.382)} = \frac{-6.908}{-0.962} \approx 7.2$$

约 **8 步**（因条件数 $\approx 2.6$，本例收敛极快，这正是接近"球形"损失面时梯度下降的高效表现）。

</details>
