# 第7章：行列式的定义（融合版）

> 行列式是矩阵的一个标量"指纹"——它在一个数字中浓缩了矩阵的全部可逆性信息，也记录了线性变换对空间的拉伸或翻转程度。
>
> **本文件**：融合"原版严格推导 + 速记 / 套路 / 自测"。保留原版完整正文 + 在最前置一例速记 / 思维路径 + 最后追加方法总结与自测。

> **一例速记**：
> **2×2 行列式**：$\det\!\begin{pmatrix}a&b\\c&d\end{pmatrix} = ad - bc$（主对角乘积减副对角乘积）。
> **几何意义**：$|\det(A)|$ = 线性变换对有向面积（体积）的缩放比；$\det(A)=0$ 矩阵奇异；$\det(A)<0$ 翻转方向。
> **代数余子式**：$C_{ij}=(-1)^{i+j}M_{ij}$，棋盘格符号，左上角为 $+$。
> **Laplace 展开**：按任意行 $\sum_j a_{ij}C_{ij}$ 或任意列 $\sum_i a_{ij}C_{ij}$；优先选零最多的行/列。
> **三角矩阵**：$\det$ = 主对角线元素之积。
> **AI 关联**：Jacobian 行列式 $|\det J_f|$ 是概率密度变换的"体积补偿因子"，是 Normalizing Flow 的核心。

---

## 引入：一道"行列式为零"的识别题

> **题目**：不展开计算，判断以下矩阵的行列式是否为零，并给出几何解释：
>
> $$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$$

请先停下来想一想：你不必逐一计算 6 个三乘三展开项——如果能**看穿矩阵的结构**，答案立刻出来。

---

## 思维路径还原（解题者的内心独白）

> "看到这个矩阵，我先不急着用 Sarrus 法则，先观察结构。
>
> **关键发现**：每一行都是等差数列，且第三行 = 第一行 + 第二行 − (第一行 − 第二行) = ……等等，直接看：$r_3 = 2r_2 - r_1$（$7=2\times4-1$，$8=2\times5-2$，$9=2\times6-3$）。
>
> 三行线性相关 → 矩阵奇异 → $\det(A) = 0$。
>
> **几何解释**：三个行向量 $(1,2,3),(4,5,6),(7,8,9)$ 共面（都在同一个二维子空间里），以它们为棱构成的"平行六面体"体积退化为零。
>
> **验证**：如果非要算，Sarrus：$(1\cdot5\cdot9+2\cdot6\cdot7+3\cdot4\cdot8)-(3\cdot5\cdot7+1\cdot6\cdot8+2\cdot4\cdot9) = (45+84+96)-(105+48+72)=225-225=0$。✓
>
> **延伸思考**：如果将 $a_{33}$ 从 $9$ 改为 $10$，三行不再线性相关，行列式就不再为零——改一个元素，行列式从 $0$ 跳到非零，可逆性发生突变。这正是行列式"脆弱性"的来源，也是数值稳定性问题的根源。"

---

## 学习目标

完成本章学习后，你将能够：

- 理解行列式的定义及其几何意义（面积、体积的有向缩放比）
- 熟练计算 $2 \times 2$ 和 $3 \times 3$ 矩阵的行列式
- 理解余子式与代数余子式的定义及符号规则
- 掌握 Laplace 展开定理，并能按任意行或列展开计算行列式
- 了解行列式在深度学习中的核心应用——Jacobian 行列式与概率密度变换

---

## 7.1 行列式的引入

### 从二元线性方程组出发

考虑二元一次方程组：

$$\begin{cases} a_{11}x_1 + a_{12}x_2 = b_1 \\ a_{21}x_1 + a_{22}x_2 = b_2 \end{cases}$$

用消元法：第一个方程乘以 $a_{22}$，第二个方程乘以 $a_{12}$，相减得：

$$(a_{11}a_{22} - a_{12}a_{21})x_1 = b_1 a_{22} - b_2 a_{12}$$

类似地可以解出 $x_2$。关键在于，若系数

$$D = a_{11}a_{22} - a_{12}a_{21} \ne 0$$

则方程组有唯一解；若 $D = 0$，则方程组要么无解，要么有无穷多解。

这个量 $D$ 就是系数矩阵

$$A = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}$$

的**行列式**（determinant），记作 $\det(A)$ 或 $|A|$。行列式"检测"了矩阵是否可逆——这是它最根本的代数意义。

### 几何意义：面积与体积

行列式的几何意义是线性变换对**有向体积**的缩放比。

**二维情形：** 设 $A$ 是 $2 \times 2$ 矩阵，其列向量为 $\mathbf{a}_1, \mathbf{a}_2 \in \mathbb{R}^2$。以这两个向量为邻边构成的平行四边形，其**有向面积**恰好等于 $\det(A)$。

- $|\det(A)| > 1$：变换将面积放大；
- $0 < |\det(A)| < 1$：变换将面积缩小；
- $\det(A) = 0$：变换将平面压缩到一条线（面积为零），矩阵奇异；
- $\det(A) < 0$：变换在缩放的同时翻转了方向（"手性"改变）。

**三维情形：** 设 $A$ 是 $3 \times 3$ 矩阵，列向量为 $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3 \in \mathbb{R}^3$。以这三个向量为棱构成的平行六面体，其**有向体积**等于 $\det(A)$。

> **核心直觉：** $\det(A)$ 告诉我们，经过矩阵 $A$ 对应的线性变换后，单位超立方体变成了多大的平行多面体。绝对值 $|\det(A)|$ 是体积的缩放倍数，符号描述方向是否被翻转。

---

## 7.2 行列式的定义

### 2×2 行列式

对于 $2 \times 2$ 矩阵

$$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$$

其行列式定义为：

$$\det(A) = \begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc$$

**计算示例：**

$$\det\begin{pmatrix} 3 & 2 \\ 1 & 4 \end{pmatrix} = 3 \cdot 4 - 2 \cdot 1 = 12 - 2 = 10$$

$$\det\begin{pmatrix} 2 & 6 \\ 1 & 3 \end{pmatrix} = 2 \cdot 3 - 6 \cdot 1 = 6 - 6 = 0 \quad \text{（奇异矩阵）}$$

### 3×3 行列式（Sarrus 法则）

对于 $3 \times 3$ 矩阵

$$A = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix}$$

行列式的展开式为：

$$\det(A) = a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32} - a_{13}a_{22}a_{31} - a_{12}a_{21}a_{33} - a_{11}a_{23}a_{32}$$

**Sarrus 法则**（对角线记忆法）提供了一种直观的记忆方式：将矩阵的前两列重复写在右侧，然后沿六条对角线方向分别取乘积，右斜方向三条取正号，左斜方向三条取负号：

$$\det(A) = \underbrace{(a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32})}_{\text{三条右斜对角线，取正}} - \underbrace{(a_{13}a_{22}a_{31} + a_{11}a_{23}a_{32} + a_{12}a_{21}a_{33})}_{\text{三条左斜对角线，取负}}$$

> **警告：** Sarrus 法则仅适用于 $3 \times 3$ 矩阵，不能推广到 $4 \times 4$ 及更高阶。

**计算示例：**

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$$

$$\det(A) = (1 \cdot 5 \cdot 9 + 2 \cdot 6 \cdot 7 + 3 \cdot 4 \cdot 8) - (3 \cdot 5 \cdot 7 + 1 \cdot 6 \cdot 8 + 2 \cdot 4 \cdot 9)$$

$$= (45 + 84 + 96) - (105 + 48 + 72) = 225 - 225 = 0$$

行列式为零，说明该矩阵奇异——观察可知第三行 $= 2 \times$ 第二行 $-$ 第一行，行向量线性相关。

再来一个非零的例子：

$$B = \begin{pmatrix} 1 & 0 & 2 \\ 3 & 1 & 0 \\ 0 & 2 & 1 \end{pmatrix}$$

$$\det(B) = (1 \cdot 1 \cdot 1 + 0 \cdot 0 \cdot 0 + 2 \cdot 3 \cdot 2) - (2 \cdot 1 \cdot 0 + 1 \cdot 0 \cdot 1 + 0 \cdot 3 \cdot 1)$$

$$= (1 + 0 + 12) - (0 + 0 + 0) = 13$$

### $n$ 阶行列式的递归定义

对于一般的 $n \times n$ 矩阵，行列式通过**递归（Laplace 展开）**定义。

**基础情形：** $\det([a]) = a$（$1 \times 1$ 矩阵的行列式就是其唯一元素）。

**递归步骤：** $n \times n$ 行列式通过沿第一行展开定义为：

$$\det(A) = \sum_{j=1}^{n} (-1)^{1+j} \, a_{1j} \, M_{1j}$$

其中 $M_{1j}$ 是删去第 $1$ 行、第 $j$ 列后得到的 $(n-1) \times (n-1)$ 子矩阵的行列式（称为**余子式**，详见 7.3 节）。

这一定义将 $n$ 阶行列式归结为 $n$ 个 $(n-1)$ 阶行列式，层层递归直到 $1 \times 1$ 情形。

---

## 7.3 余子式与代数余子式

### 余子式的定义

设 $A$ 是 $n \times n$ 矩阵。**删去第 $i$ 行和第 $j$ 列**后，得到一个 $(n-1) \times (n-1)$ 子矩阵，其行列式称为元素 $a_{ij}$ 的**余子式**（minor），记作 $M_{ij}$。

**示例：** 对于

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$$

元素 $a_{12} = 2$ 的余子式是删去第1行第2列后的子矩阵的行列式：

$$M_{12} = \begin{vmatrix} 4 & 6 \\ 7 & 9 \end{vmatrix} = 4 \cdot 9 - 6 \cdot 7 = 36 - 42 = -6$$

元素 $a_{23} = 6$ 的余子式是删去第2行第3列后的子矩阵的行列式：

$$M_{23} = \begin{vmatrix} 1 & 2 \\ 7 & 8 \end{vmatrix} = 1 \cdot 8 - 2 \cdot 7 = 8 - 14 = -6$$

### 代数余子式的定义

元素 $a_{ij}$ 的**代数余子式**（cofactor）$C_{ij}$ 是在余子式 $M_{ij}$ 前面加上一个符号因子：

$$C_{ij} = (-1)^{i+j} M_{ij}$$

### 符号规则

符号因子 $(-1)^{i+j}$ 由位置 $(i,j)$ 决定。当 $i+j$ 为偶数时取正号，为奇数时取负号，形成棋盘格状的符号阵：

$$\begin{pmatrix} + & - & + & - & \cdots \\ - & + & - & + & \cdots \\ + & - & + & - & \cdots \\ - & + & - & + & \cdots \\ \vdots & \vdots & \vdots & \vdots & \ddots \end{pmatrix}$$

对于 $3 \times 3$ 矩阵，符号阵为：

$$\begin{pmatrix} + & - & + \\ - & + & - \\ + & - & + \end{pmatrix}$$

**示例（续）：** 对上面的矩阵 $A$，

$$C_{12} = (-1)^{1+2} M_{12} = (-1)^3 \cdot (-6) = (+6)$$

$$C_{23} = (-1)^{2+3} M_{23} = (-1)^5 \cdot (-6) = (+6)$$

> **记忆技巧：** 位置 $(1,1)$（左上角）始终取正号。沿行或列每移动一格，符号交替改变一次。

---

## 7.4 按行（列）展开

### Laplace 展开定理

**定理（Laplace 展开）：** $n \times n$ 矩阵 $A$ 的行列式可以按**任意**一行（或任意一列）展开：

**按第 $i$ 行展开：**

$$\det(A) = \sum_{j=1}^{n} a_{ij} C_{ij} = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} M_{ij}$$

**按第 $j$ 列展开：**

$$\det(A) = \sum_{i=1}^{n} a_{ij} C_{ij} = \sum_{i=1}^{n} (-1)^{i+j} a_{ij} M_{ij}$$

这一定理的强大之处在于**选择权**：实际计算时，应优先选择含有最多零元素的行或列展开，从而减少计算量。

**异行（列）展开恒为零：** 若将第 $i$ 行的元素与第 $k$ 行（$k \ne i$）的代数余子式配对求和，结果为零：

$$\sum_{j=1}^{n} a_{ij} C_{kj} = 0 \quad (i \ne k)$$

这一性质在理论推导中经常用到。

### 按行展开计算示例

**示例1：** 利用按第一行展开计算

$$A = \begin{pmatrix} 2 & -1 & 3 \\ 0 & 4 & 1 \\ 5 & 2 & -2 \end{pmatrix}$$

按第一行展开：

$$\det(A) = a_{11}C_{11} + a_{12}C_{12} + a_{13}C_{13}$$

$$= 2 \cdot (+1) \begin{vmatrix} 4 & 1 \\ 2 & -2 \end{vmatrix} + (-1) \cdot (-1) \begin{vmatrix} 0 & 1 \\ 5 & -2 \end{vmatrix} + 3 \cdot (+1) \begin{vmatrix} 0 & 4 \\ 5 & 2 \end{vmatrix}$$

分别计算三个 $2 \times 2$ 行列式：

$$\begin{vmatrix} 4 & 1 \\ 2 & -2 \end{vmatrix} = 4 \cdot (-2) - 1 \cdot 2 = -8 - 2 = -10$$

$$\begin{vmatrix} 0 & 1 \\ 5 & -2 \end{vmatrix} = 0 \cdot (-2) - 1 \cdot 5 = 0 - 5 = -5$$

$$\begin{vmatrix} 0 & 4 \\ 5 & 2 \end{vmatrix} = 0 \cdot 2 - 4 \cdot 5 = 0 - 20 = -20$$

代入：

$$\det(A) = 2 \cdot (-10) + (-1) \cdot (-1) \cdot (-5) + 3 \cdot (-20)$$

$$= -20 + (-5) + (-60) = -85$$

**示例2：** 选择含零元素最多的行/列展开

$$B = \begin{pmatrix} 3 & 0 & 0 & 2 \\ 1 & 5 & 0 & 0 \\ 0 & 2 & 4 & 0 \\ 1 & 0 & 3 & 6 \end{pmatrix}$$

第一行有两个零，按第一行展开只需计算两个 $3 \times 3$ 余子式：

$$\det(B) = 3 \cdot C_{11} + 0 + 0 + 2 \cdot C_{14}$$

$$= 3 \cdot (+1)\begin{vmatrix} 5 & 0 & 0 \\ 2 & 4 & 0 \\ 0 & 3 & 6 \end{vmatrix} + 2 \cdot (-1)^{1+4}\begin{vmatrix} 1 & 5 & 0 \\ 0 & 2 & 4 \\ 1 & 0 & 3 \end{vmatrix}$$

对第一个行列式（上三角形），继续按第一行展开（或直接用三角矩阵规律）：

$$\begin{vmatrix} 5 & 0 & 0 \\ 2 & 4 & 0 \\ 0 & 3 & 6 \end{vmatrix} = 5 \cdot 4 \cdot 6 = 120 \quad \text{（下三角矩阵，对角线之积）}$$

对第二个行列式按第一行展开：

$$\begin{vmatrix} 1 & 5 & 0 \\ 0 & 2 & 4 \\ 1 & 0 & 3 \end{vmatrix} = 1 \cdot \begin{vmatrix} 2 & 4 \\ 0 & 3 \end{vmatrix} - 5 \cdot \begin{vmatrix} 0 & 4 \\ 1 & 3 \end{vmatrix} + 0$$

$$= 1 \cdot (6 - 0) - 5 \cdot (0 - 4) = 6 + 20 = 26$$

因此：

$$\det(B) = 3 \cdot 120 + 2 \cdot (-1) \cdot 26 = 360 - 52 = 308$$

### 三角矩阵的行列式

**定理：** 上三角矩阵、下三角矩阵和对角矩阵的行列式均等于**主对角线元素之积**：

$$\det\begin{pmatrix} d_1 & * & \cdots & * \\ 0 & d_2 & \cdots & * \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & d_n \end{pmatrix} = d_1 d_2 \cdots d_n$$

这一规律是高斯消元计算行列式的理论基础，也解释了为何 $2 \times 2$ 情形的公式 $ad - bc$ 中，上三角情形（$c=0$）直接得到 $ad$。

---

## 本章小结

| 概念 | 核心内容 |
|------|---------|
| 行列式的引入 | 方程组有唯一解的判别量；$D = a_{11}a_{22} - a_{12}a_{21}$ |
| 几何意义 | 线性变换对有向面积（体积）的缩放比；$\det = 0$ 时矩阵奇异 |
| $2\times 2$ 行列式 | $\det(A) = ad - bc$ |
| $3\times 3$ 行列式 | Sarrus 法则：三条正斜线之积减三条负斜线之积（仅限 $3\times 3$）|
| 余子式 $M_{ij}$ | 删去第 $i$ 行第 $j$ 列后 $(n-1)\times(n-1)$ 子矩阵的行列式 |
| 代数余子式 $C_{ij}$ | $C_{ij} = (-1)^{i+j}M_{ij}$，棋盘格符号 |
| Laplace 展开 | 可按任意行或列展开；优先选零元素多的行/列 |
| 三角矩阵 | $\det$ 等于主对角线元素之积 |

**核心要点回顾：**

1. 行列式是矩阵可逆性的"一个数检验"：$\det(A) \ne 0$ 当且仅当 $A$ 可逆。
2. 几何上，$|\det(A)|$ 是线性变换对有向体积的缩放倍数。
3. 代数余子式 $C_{ij} = (-1)^{i+j}M_{ij}$ 在符号上遵循棋盘格规律。
4. Laplace 展开的自由选择性是计算技巧的核心：选零多的行/列，大幅降低计算量。
5. 三角矩阵的行列式等于主对角线之积，是最高效的特殊情形。

---

## 深度学习应用

### Jacobian 行列式的概念

在深度学习中，行列式最重要的应用来自 **Jacobian 矩阵的行列式**（Jacobian determinant）。

设 $f: \mathbb{R}^n \to \mathbb{R}^n$ 是一个可微映射，其 **Jacobian 矩阵**定义为所有一阶偏导数构成的矩阵：

$$J_f(\mathbf{x}) = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_n}{\partial x_1} & \frac{\partial f_n}{\partial x_2} & \cdots & \frac{\partial f_n}{\partial x_n} \end{pmatrix}$$

Jacobian 矩阵在局部将非线性变换 $f$ 近似为线性变换，而 $\det(J_f(\mathbf{x}))$ 就是这个局部线性变换对体积的缩放比。

### 概率密度变换中的作用

设随机变量 $\mathbf{z}$ 服从分布 $p_Z(\mathbf{z})$，令 $\mathbf{x} = f(\mathbf{z})$ 是一个可逆变换（$f$ 可微且 Jacobian 处处非奇异）。则 $\mathbf{x}$ 的概率密度为：

$$p_X(\mathbf{x}) = p_Z(f^{-1}(\mathbf{x})) \cdot \left|\det\left(J_{f^{-1}}(\mathbf{x})\right)\right|$$

或等价地（利用反函数定理 $\det(J_{f^{-1}}) = 1/\det(J_f)$）：

$$p_X(\mathbf{x}) = p_Z(\mathbf{z}) \cdot \left|\det\left(J_f(\mathbf{z})\right)\right|^{-1}$$

**直觉：** 变换 $f$ 在某个区域将空间"压缩"（$|\det J_f| < 1$），则该区域的概率密度必须相应"拉伸"才能保证总概率为1。Jacobian 行列式精确地量化了这种压缩/拉伸的程度。

这一公式是**变量替换公式**在多维情形下的精确表达，是现代生成模型的数学核心。

### Normalizing Flows 基础

**Normalizing Flows**（规范化流）是一类生成模型，通过学习一系列可逆变换，将简单分布（如标准正态分布）逐步变换为复杂的目标分布。

设变换链为 $\mathbf{x} = f_K \circ f_{K-1} \circ \cdots \circ f_1(\mathbf{z})$，利用链式法则，总 Jacobian 行列式为各层之积：

$$\log p_X(\mathbf{x}) = \log p_Z(\mathbf{z}) - \sum_{k=1}^{K} \log \left|\det\left(J_{f_k}(\mathbf{z}_{k-1})\right)\right|$$

其中 $\mathbf{z}_0 = \mathbf{z}$，$\mathbf{z}_k = f_k(\mathbf{z}_{k-1})$。

训练目标是最大化数据的对数似然，这直接依赖于对每层 Jacobian 行列式的高效计算。

**关键工程挑战：** 对 $n$ 维向量，一般 Jacobian 矩阵是 $n \times n$ 的，计算其行列式需要 $O(n^3)$。对高维数据（如图像，$n$ 可达数万），这完全不可行。因此 Normalizing Flows 的核心设计原则是**构造使 Jacobian 行列式易于计算的变换结构**，例如：

- **耦合层（Coupling Layers，RealNVP）：** 将变换设计成下三角结构，行列式退化为对角线元素之积，从 $O(n^3)$ 降至 $O(n)$；
- **自回归流（Autoregressive Flows，MAF/IAF）：** Jacobian 矩阵为三角矩阵，行列式同样为 $O(n)$；
- **1×1 卷积（Glow）：** 对有限维的混合矩阵精确计算行列式，同时利用 LU 分解将复杂度从 $O(n^3)$ 降至 $O(n)$。

### 代码示例

```python
import numpy as np
import torch
import torch.nn as nn

# ── 示例1：行列式的基本计算 ─────────────────────────────────────
A_2x2 = np.array([[3, 2],
                  [1, 4]], dtype=float)
print(f"2×2 行列式: {np.linalg.det(A_2x2):.4f}")  # 10.0

A_3x3 = np.array([[1, 0, 2],
                  [3, 1, 0],
                  [0, 2, 1]], dtype=float)
print(f"3×3 行列式: {np.linalg.det(A_3x3):.4f}")  # 13.0

# ── 示例2：Jacobian 行列式——变量变换 ────────────────────────────
# 将二维标准正态变量 z 通过线性变换 x = Az 得到相关正态分布
# 变换后分布的密度需要除以 |det(A)|

def standard_normal_log_prob(z):
    """标准正态分布的对数概率密度"""
    return -0.5 * np.sum(z**2) - 0.5 * len(z) * np.log(2 * np.pi)

def transformed_log_prob(x, A):
    """经过线性变换 x = Az 后的对数概率密度"""
    A_inv = np.linalg.inv(A)
    z = A_inv @ x                          # 求逆变换
    log_pz = standard_normal_log_prob(z)
    log_det_J = np.log(np.abs(np.linalg.det(A)))  # Jacobian 的对数
    return log_pz - log_det_J              # 变量替换公式

A_transform = np.array([[2.0, 0.5],
                         [0.0, 1.5]])
x_sample = np.array([1.0, 0.5])

log_p = transformed_log_prob(x_sample, A_transform)
print(f"变换后的对数概率密度: {log_p:.4f}")
print(f"Jacobian 行列式 |det(A)|: {np.abs(np.linalg.det(A_transform)):.4f}")  # 3.0

# ── 示例3：Normalizing Flow 的耦合层（RealNVP 风格）─────────────
class AffineCouplingLayer(nn.Module):
    """
    RealNVP 耦合层：设计使 Jacobian 行列式为对角线元素之积，
    从而将 O(n^3) 的行列式计算降至 O(n)。

    变换：
      x1 = z1                          （恒等）
      x2 = z2 * exp(s(z1)) + t(z1)    （仿射变换，由 z1 参数化）

    Jacobian 矩阵为下三角，行列式 = 1 * prod(exp(s(z1)))
    """
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        # s 和 t 网络：以 z1 为输入，输出缩放和平移参数
        self.scale_net = nn.Sequential(
            nn.Linear(half, 64), nn.Tanh(),
            nn.Linear(64, dim - half)
        )
        self.translate_net = nn.Sequential(
            nn.Linear(half, 64), nn.Tanh(),
            nn.Linear(64, dim - half)
        )

    def forward(self, z):
        """前向（z -> x）：返回变换后的 x 和对数 Jacobian 行列式"""
        z1, z2 = z.chunk(2, dim=-1)
        s = self.scale_net(z1)
        t = self.translate_net(z1)
        x1 = z1
        x2 = z2 * torch.exp(s) + t
        # 对数 Jacobian 行列式 = sum(s)，O(n) 而非 O(n^3)
        log_det_J = s.sum(dim=-1)
        return torch.cat([x1, x2], dim=-1), log_det_J

    def inverse(self, x):
        """逆变换（x -> z）"""
        x1, x2 = x.chunk(2, dim=-1)
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        z1 = x1
        z2 = (x2 - t) * torch.exp(-s)
        return torch.cat([z1, z2], dim=-1)


# 验证耦合层
torch.manual_seed(42)
layer = AffineCouplingLayer(dim=4)
z = torch.randn(3, 4)          # batch_size=3, dim=4

x, log_det = layer.forward(z)
z_recovered = layer.inverse(x)

print(f"\n耦合层验证：")
print(f"  输入 z 与逆变换恢复的 z 差异: {(z - z_recovered).abs().max().item():.2e}")  # ~0
print(f"  对数 Jacobian 行列式 shape: {log_det.shape}")   # (3,)
print(f"  对数 Jacobian 行列式 (第一个样本): {log_det[0].item():.4f}")

# ── 示例4：LU 分解计算行列式（高效方式）────────────────────────
# NumPy 内部即使用 LU 分解来计算行列式，slogdet 给出符号和对数绝对值
A_large = np.random.randn(100, 100)
sign, logabsdet = np.linalg.slogdet(A_large)
print(f"\n100×100 矩阵的对数行列式: sign={sign:.0f}, log|det|={logabsdet:.2f}")
# 直接计算 det 可能溢出，slogdet 更数值稳定
```

### 延伸阅读

- **《Deep Learning》**（Goodfellow et al.）第3章：概率与信息论，变量变换公式
- **《Normalizing Flows for Generative Modeling》**（Kobyzev et al., 2020）：Normalizing Flows 综述
- **RealNVP 论文**（Dinh et al., 2017）：耦合层结构的开创性工作
- **Glow 论文**（Kingma & Dhariwal, 2018）：基于可逆 $1 \times 1$ 卷积的生成模型

---

## 抽象成方法（套路总结）

### 核心公式速查

| 名称 | 公式 | 备注 |
|---|---|---|
| **2×2 行列式** | $ad - bc$ | 主对角 − 副对角 |
| **3×3 Sarrus** | 三正斜线之积 − 三负斜线之积 | 仅限 $3\times3$ |
| **代数余子式** | $C_{ij}=(-1)^{i+j}M_{ij}$ | 棋盘格符号 |
| **Laplace 展开（按行 $i$）** | $\sum_{j=1}^n a_{ij}C_{ij}$ | 可按任意行/列 |
| **三角矩阵** | $\det = d_1 d_2 \cdots d_n$ | 主对角线之积 |
| **可逆判据** | $\det(A)\neq 0 \Leftrightarrow A$ 可逆 | 最核心的单数检验 |

### Laplace 展开 3 步选行/列策略

1. **数零**：统计每行每列的零元素个数
2. **选最多**：选零元素最多的那行（列）展开
3. **展开**：只需计算非零 $a_{ij}$ 对应的 $C_{ij}$（零项跳过）

### Jacobian 行列式 2 步快算（线性变换）

1. **写 Jacobian 矩阵** $J_f$（偏导数矩阵）
2. **算 $|\det J_f|$**：体积缩放因子；对数概率密度变换公式中取对数 $\log|\det J_f|$

---

## 方法变形

### 变形 1：含字母参数的行列式

先展开成关于参数的表达式，再令其等于零或非零判断可逆性。例如，$\det\begin{pmatrix}a&1\\1&a\end{pmatrix}=a^2-1$，当 $a=\pm1$ 时奇异。

### 变形 2：行（列）整体含公因子

先提出公因子再展开，降低中间计算量。例如：

$$\det\begin{pmatrix}2&4\\3&6\end{pmatrix} = 2\cdot3\cdot\det\begin{pmatrix}1&2\\1&2\end{pmatrix}$$

（两行有公因子 2 和 3，提出后发现行相同，行列式为零。）

### 变形 3：识别"线性相关行/列" → 立刻为零

若某行 = 其他行的线性组合（或某列 = 其他列的线性组合），无需计算，直接 $\det = 0$。

### 变形 4：$4\times4$ 及更高阶

Sarrus 失效。标准路线：选含零多的行/列做 Laplace 展开，降至 $3\times3$ 再用 Sarrus 或继续降阶。

---

## 思考路标（条件反射）

1. 看到"行列式" → 先看阶数：$2\times2$ 直接公式；$3\times3$ Sarrus 或展开；$n\times n$ Laplace 展开
2. 看到行列式中有"整行/列全为零" → 直接 $\det = 0$
3. 看到"两行（列）相同或成比例" → 直接 $\det = 0$
4. 看到"三角矩阵" → $\det$ = 主对角线之积，不需展开
5. 看到"行列式 = 0" → 矩阵奇异、不可逆、行（列）向量线性相关
6. 看到"高阶行列式" → 数零、选零多的行/列展开
7. 看到 $C_{ij}$ → 检查符号：位置 $(i,j)$，$i+j$ 偶数取 $+$，奇数取 $-$
8. 看到"概率密度变换" → Jacobian 行列式乘到分子（逆方向除到分母）
9. 看到 $\log p(x)$ 公式含 $\log\vert\det J\vert$ → 这是 Normalizing Flow 的对数似然修正项
10. 看到"旋转矩阵" → $\det = 1$（不改变体积，不翻转）；"反射矩阵" → $\det = -1$

---

## 易错点

1. **Sarrus 法则只适用于 $3\times3$**：许多人习惯性地把它推广到 $4\times4$，结果完全错误。$4$ 阶及以上必须用 Laplace 展开或高斯消元。

2. **代数余子式的下标顺序**：$M_{ij}$ 是删去**第 $i$ 行第 $j$ 列**后的子矩阵行列式，不是第 $j$ 行第 $i$ 列。伴随矩阵 $\text{adj}(A)$ 的 $(i,j)$ 元素是 $C_{ji}$（有转置），常见混淆。

3. **符号棋盘格从 $(1,1)$ 开始是 $+$**：角落不一定是正号——$(2,3)$ 位置是 $(-1)^{2+3}=(-1)^5=-1$，奇数位（行+列为奇）取负。记住"左上角为正，交错变号"。

4. **按行展开只乘当行元素**：$\det(A)=\sum_j a_{ij}C_{ij}$，其中 $C_{ij}$ 是**同一矩阵 $A$** 的代数余子式，不是修改后矩阵的余子式。

5. **$|\det(A)|\neq\det(\vert A\vert)$**：行列式的绝对值 vs 元素取绝对值后再算行列式——这是两件完全不同的事，不要混淆。

6. **Jacobian 行列式的方向**：密度变换公式中，$p_X(\mathbf{x})=p_Z(\mathbf{z})\cdot|\det J_f|^{-1}$，分母是正向变换的 Jacobian；若用逆变换写则分母变分子，两者方向相反，勿弄反。

---

## 典型应用例题

### 例 1：选"零最多列"展开 $3\times3$ 行列式

> **题目**：计算
> $$\det\begin{pmatrix}2&0&3\\0&5&0\\1&0&4\end{pmatrix}$$

【思路】第二列 $(0,5,0)^T$ 只有一个非零元，选第二列展开，只需算一个 $2\times2$ 余子式。

【解】按第二列展开：

$$\det = 0\cdot C_{12} + 5\cdot C_{22} + 0\cdot C_{32} = 5\cdot C_{22}$$

$$C_{22} = (-1)^{2+2}M_{22} = +\begin{vmatrix}2&3\\1&4\end{vmatrix} = 8-3=5$$

$$\det = 5\times 5 = 25$$

【答案】$\boxed{25}$。

【注】若选第一行或第一列，需计算两个 $2\times2$ 行列式；选第二列只需一个，效率翻倍。

### 例 2：含参数的行列式 + 可逆判断

> **题目**：$A = \begin{pmatrix}1&k\\k&4\end{pmatrix}$。问 $k$ 取何值时 $A$ 奇异？

【思路】$\det(A) = 4 - k^2$，令其等于零。

【解】$4-k^2=0 \Rightarrow k=\pm 2$。

当 $k=2$ 时 $A=\begin{pmatrix}1&2\\2&4\end{pmatrix}$（第二行 = 2 倍第一行，线性相关）；$k=-2$ 类似。

【答案】$\boxed{k=\pm 2}$ 时奇异；$k\neq\pm 2$ 时可逆。

### 例 3：Jacobian 行列式 + 概率密度变换

> **题目**：$z\sim\mathcal{N}(0,1)$，令 $x=3z+2$。用 Jacobian 行列式求 $p_X(x)$，并说明 $x$ 服从何分布。

【思路】一维情形 Jacobian = $|dx/dz| = 3$；密度变换公式 $p_X(x)=p_Z(z)/|dx/dz|$，代入 $z=(x-2)/3$。

【解】

$$p_X(x)=\frac{1}{\sqrt{2\pi}}e^{-z^2/2}\cdot\frac{1}{3}\bigg|_{z=(x-2)/3} = \frac{1}{3\sqrt{2\pi}}\exp\!\left(-\frac{(x-2)^2}{18}\right)$$

这正是 $\mathcal{N}(2,9)$（均值 $2$，方差 $9=3^2$）的密度。

【答案】$\boxed{x\sim\mathcal{N}(2,9)}$，Jacobian $=3$ 起到"拉伸密度"的补偿作用。

---

## 自测题

**自测 1**　计算 $\det\begin{pmatrix}3&-1\\2&5\end{pmatrix}$，并判断该矩阵是否可逆。

> 💡 提示：$3\times5-(-1)\times2=15+2=17\neq0$，可逆。

**自测 2**　$A=\begin{pmatrix}0&1&2\\0&3&4\\5&6&7\end{pmatrix}$。不展开，先选最优行/列，再计算 $\det(A)$。

> 💡 提示：第一列 $(0,0,5)^T$ 只有一个非零元。按第一列展开：$\det=5\cdot C_{31}=5\cdot(+1)\cdot\det\begin{pmatrix}1&2\\3&4\end{pmatrix}=5\cdot(4-6)=5\cdot(-2)=-10$。

**自测 3**　矩阵 $B$ 的第三行是第一行加第二行的两倍。不计算，直接给出 $\det(B)$ 的值并说明原因。

> 💡 提示：$r_3=r_1+2r_2$，三行线性相关（第三行 = 其余行的线性组合）$\Rightarrow\det(B)=0$。

**自测 4**　$C=\begin{pmatrix}2&0&0\\0&3&0\\0&0&5\end{pmatrix}$。直接写出 $\det(C)$，并解释几何含义。

> 💡 提示：对角矩阵，$\det=2\times3\times5=30$。几何含义：该变换将三维单位立方体体积放大 30 倍（沿三个轴分别拉伸 2、3、5 倍）。

**自测 5**　Normalizing Flow 某层变换 $f:\mathbb{R}^2\to\mathbb{R}^2$，其 Jacobian 矩阵为 $J=\begin{pmatrix}e^s&0\\t'&e^s\end{pmatrix}$（下三角）。求 $\log|\det J|$。

> 💡 提示：下三角矩阵，$\det J=e^s\cdot e^s=e^{2s}$，$\log|\det J|=2s$。这正是仿射耦合层的 $O(n)$ 对数行列式计算。

---

**回头看一眼"一例速记"**：

> $2\times2$：$ad-bc$；$3\times3$：Sarrus（三正斜 − 三负斜）；三角矩阵：对角线之积。
> $C_{ij}=(-1)^{i+j}M_{ij}$，棋盘格；Laplace 展开选零多的行/列。
> $\det(A)=0\Leftrightarrow$ 奇异；$|\det J_f|$ = 概率密度变换的体积补偿因子。

如果现在不看笔记，能独立完成例 1 + 例 3 + 自测 2 + 自测 5——本章，你拿下了。

---

## 融合版说明

本版 = **原版（严格推导 + 深度学习应用 + 练习）** + **重写版（速记 / 套路 / 例题 / 自测）** 融合：

| 段落 | 来源 | 价值 |
|---|---|---|
| 一例速记 + 引入 + 思维路径还原 | 重写版（前置） | 建立直觉 / 条件反射 |
| 学习目标 + 7.1–7.4 严格正文 | 原版 | 完整定义与推导 |
| 本章小结 | 原版 | 公式速查 |
| 深度学习应用 + 代码 | 原版 | Jacobian / Normalizing Flow 实战 |
| 练习题 + 详解 | 原版 | 系统巩固 |
| 抽象成方法 + 方法变形 | 重写版（后置） | 套路固化 |
| 思考路标 + 易错点 | 融合两版 | 条件反射 + 避雷 |
| 典型应用例题 3 例 | 重写版 | 演练精讲 |
| 自测题 5 题 | 重写版 | 额外验收 |

**适用**：一站式学习——先速记建立直觉，看严格推导，做套路总结，看代码实战，做习题巩固，自测验收。

---

## 练习题

**练习 7.1（2×2 行列式计算）**

计算以下矩阵的行列式，并说明各矩阵是否可逆：

$$A = \begin{pmatrix} 5 & 3 \\ 2 & 1 \end{pmatrix}, \quad B = \begin{pmatrix} -2 & 4 \\ 3 & -6 \end{pmatrix}, \quad C = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

对 $C$，解释其行列式值的几何意义。

---

**练习 7.2（Sarrus 法则与代数余子式）**

设

$$A = \begin{pmatrix} 2 & 1 & -1 \\ 0 & 3 & 2 \\ 1 & 0 & 4 \end{pmatrix}$$

(a) 用 Sarrus 法则计算 $\det(A)$。

(b) 计算元素 $a_{21} = 0$、$a_{22} = 3$、$a_{23} = 2$ 的代数余子式 $C_{21}$、$C_{22}$、$C_{23}$。

(c) 验证按第二行展开的结果与 (a) 相同，即 $\det(A) = a_{21}C_{21} + a_{22}C_{22} + a_{23}C_{23}$。

---

**练习 7.3（选择最优展开行/列）**

计算以下矩阵的行列式，要求选择含零最多的行或列展开，并说明选择理由：

$$D = \begin{pmatrix} 0 & 2 & 0 & 1 \\ 3 & 0 & 0 & -1 \\ 0 & 1 & 5 & 0 \\ 2 & 0 & 0 & 4 \end{pmatrix}$$

---

**练习 7.4（三角矩阵与递归）**

(a) 不展开计算，直接写出以下矩阵的行列式：

$$E = \begin{pmatrix} 3 & 7 & -2 & 5 \\ 0 & -1 & 4 & 0 \\ 0 & 0 & 2 & 9 \\ 0 & 0 & 0 & 6 \end{pmatrix}$$

(b) 证明：$n$ 阶上三角矩阵的行列式等于主对角线元素之积（利用按第一列展开做数学归纳法）。

---

**练习 7.5（Jacobian 行列式与密度变换）**

设 $z \sim \mathcal{N}(0, 1)$ 是标准正态分布的随机变量，令 $x = f(z) = 2z + 1$（仿射变换）。

(a) 计算变换 $f$ 的 Jacobian（导数），并写出 Jacobian 行列式（一维情形即 $|f'(z)|$）。

(b) 利用一维变量变换公式 $p_X(x) = p_Z(z) \cdot |dx/dz|^{-1}$，推导 $x$ 的概率密度函数，并验证 $x \sim \mathcal{N}(1, 4)$（即均值为1，方差为4的正态分布）。

(c) 将 (a)(b) 推广到二维情形：设 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I_2)$，令 $\mathbf{x} = A\mathbf{z} + \boldsymbol{\mu}$，其中 $A = \begin{pmatrix}2 & 0\\1 & 3\end{pmatrix}$，$\boldsymbol{\mu} = \begin{pmatrix}1\\0\end{pmatrix}$。写出 $\mathbf{x}$ 的协方差矩阵和 Jacobian 行列式，并说明 $\mathbf{x}$ 服从什么分布。

---

## 练习答案

<details>
<summary>练习 7.1 答案</summary>

**矩阵 $A$：**

$$\det(A) = 5 \cdot 1 - 3 \cdot 2 = 5 - 6 = -1 \ne 0$$

$A$ 可逆。

**矩阵 $B$：**

$$\det(B) = (-2)(-6) - 4 \cdot 3 = 12 - 12 = 0$$

$B$ 不可逆（奇异矩阵）。观察可知 $B$ 的第二列是第一列的 $-2$ 倍，列向量线性相关。

**矩阵 $C$（旋转矩阵）：**

$$\det(C) = \cos\theta \cdot \cos\theta - (-\sin\theta)\cdot\sin\theta = \cos^2\theta + \sin^2\theta = 1$$

**几何意义：** $C$ 是将 $\mathbb{R}^2$ 旋转角度 $\theta$ 的矩阵。旋转不改变面积，因此 $\det(C) = 1$；旋转可逆（逆变换是旋转 $-\theta$），因此行列式非零。$\det(C) = 1 > 0$ 还说明旋转不改变方向（手性），与直觉一致。

</details>

<details>
<summary>练习 7.2 答案</summary>

**(a) Sarrus 法则：**

$$\det(A) = (2 \cdot 3 \cdot 4 + 1 \cdot 2 \cdot 1 + (-1) \cdot 0 \cdot 0) - ((-1) \cdot 3 \cdot 1 + 2 \cdot 2 \cdot 0 + 1 \cdot 0 \cdot 4)$$

$$= (24 + 2 + 0) - (-3 + 0 + 0) = 26 - (-3) = 29$$

**(b) 代数余子式：**

$$C_{21} = (-1)^{2+1} M_{21} = -\begin{vmatrix} 1 & -1 \\ 0 & 4 \end{vmatrix} = -(1 \cdot 4 - (-1) \cdot 0) = -4$$

$$C_{22} = (-1)^{2+2} M_{22} = +\begin{vmatrix} 2 & -1 \\ 1 & 4 \end{vmatrix} = +(2 \cdot 4 - (-1) \cdot 1) = +(8 + 1) = 9$$

$$C_{23} = (-1)^{2+3} M_{23} = -\begin{vmatrix} 2 & 1 \\ 1 & 0 \end{vmatrix} = -(2 \cdot 0 - 1 \cdot 1) = -(-1) = 1$$

**(c) 验证按第二行展开：**

$$\det(A) = a_{21}C_{21} + a_{22}C_{22} + a_{23}C_{23} = 0 \cdot (-4) + 3 \cdot 9 + 2 \cdot 1 = 0 + 27 + 2 = 29 \checkmark$$

与 Sarrus 法则结果一致。注意 $a_{21} = 0$ 使得第一项直接为零，选择有零元素的行可减少计算步骤。

</details>

<details>
<summary>练习 7.3 答案</summary>

观察各行/列的零元素个数：

- 第1行：$(0, 2, 0, 1)$，含2个零；
- 第2行：$(3, 0, 0, -1)$，含2个零；
- 第3行：$(0, 1, 5, 0)$，含2个零；
- 第4行：$(2, 0, 0, 4)$，含2个零；
- 第3列：$(0, 0, 5, 0)^T$，含3个零——**最优选择**。

按第3列展开（只有 $d_{33} = 5$ 非零）：

$$\det(D) = 5 \cdot C_{33}$$

$$C_{33} = (-1)^{3+3} M_{33} = +\begin{vmatrix} 0 & 2 & 1 \\ 3 & 0 & -1 \\ 2 & 0 & 4 \end{vmatrix}$$

对这个 $3 \times 3$ 行列式，第3列 $(1, -1, 4)^T$ 无零，选第2列 $(2, 0, 0)^T$（含两个零）按第2列展开：

$$\begin{vmatrix} 0 & 2 & 1 \\ 3 & 0 & -1 \\ 2 & 0 & 4 \end{vmatrix} = 2 \cdot (-1)^{1+2}\begin{vmatrix} 3 & -1 \\ 2 & 4 \end{vmatrix} = -2 \cdot (12 + 2) = -2 \cdot 14 = -28$$

因此：

$$\det(D) = 5 \cdot (-28) = -140$$

</details>

<details>
<summary>练习 7.4 答案</summary>

**(a)** $E$ 是上三角矩阵，行列式等于主对角线元素之积：

$$\det(E) = 3 \cdot (-1) \cdot 2 \cdot 6 = -36$$

**(b) 归纳证明：**

**基础情形（$n=1$）：** $\det([d_1]) = d_1$，等于主对角线元素之积，成立。

**归纳假设：** 设对所有 $(n-1) \times (n-1)$ 上三角矩阵，行列式等于主对角线元素之积。

**归纳步骤：** 设 $A$ 是 $n \times n$ 上三角矩阵，主对角线为 $d_1, d_2, \ldots, d_n$。按第一列展开：

第一列形如 $(d_1, 0, 0, \ldots, 0)^T$，只有 $a_{11} = d_1$ 非零，故：

$$\det(A) = d_1 \cdot C_{11} = d_1 \cdot (-1)^{1+1} M_{11} = d_1 \cdot M_{11}$$

其中 $M_{11}$ 是删去第1行第1列后的 $(n-1)\times(n-1)$ 子矩阵的行列式。由于 $A$ 是上三角矩阵，$M_{11}$ 的子矩阵仍是上三角矩阵，主对角线为 $d_2, d_3, \ldots, d_n$。由归纳假设：

$$M_{11} = d_2 d_3 \cdots d_n$$

因此 $\det(A) = d_1 d_2 \cdots d_n$，归纳完成。$\square$

</details>

<details>
<summary>练习 7.5 答案</summary>

**(a) Jacobian：**

$f(z) = 2z + 1$，故 $f'(z) = \frac{dx}{dz} = 2$。

Jacobian 行列式（一维）为 $|f'(z)| = |2| = 2$。

**(b) 密度变换：**

$z = f^{-1}(x) = \frac{x-1}{2}$，$\frac{dz}{dx} = \frac{1}{2}$。

利用变量变换公式：

$$p_X(x) = p_Z(z) \cdot \left|\frac{dz}{dx}\right| = \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \cdot \frac{1}{2}$$

代入 $z = \frac{x-1}{2}$：

$$p_X(x) = \frac{1}{2\sqrt{2\pi}} \exp\!\left(-\frac{(x-1)^2}{8}\right) = \frac{1}{\sqrt{2\pi \cdot 4}} \exp\!\left(-\frac{(x-1)^2}{2 \cdot 4}\right)$$

这正是 $\mathcal{N}(1, 4)$（均值为1，方差为 $\sigma^2 = 4$，即标准差为2）的概率密度函数。$\checkmark$

直觉上，$f(z) = 2z + 1$ 将分布的中心从0平移到1（均值 $+1$），并将分布的尺度放大了2倍（标准差 $\times 2$，方差 $\times 4$）。

**(c) 二维推广：**

$\mathbf{x} = A\mathbf{z} + \boldsymbol{\mu}$，其中 $A = \begin{pmatrix}2 & 0\\1 & 3\end{pmatrix}$，$\boldsymbol{\mu} = \begin{pmatrix}1\\0\end{pmatrix}$。

**协方差矩阵：**

$$\Sigma = A \cdot \text{Cov}(\mathbf{z}) \cdot A^T = A I_2 A^T = AA^T = \begin{pmatrix}2 & 0\\1 & 3\end{pmatrix}\begin{pmatrix}2 & 1\\0 & 3\end{pmatrix} = \begin{pmatrix}4 & 2\\2 & 10\end{pmatrix}$$

**Jacobian 行列式：**

变换 $\mathbf{z} \mapsto \mathbf{x} = A\mathbf{z} + \boldsymbol{\mu}$ 的 Jacobian 矩阵就是 $A$，因此：

$$|\det(J_f)| = |\det(A)| = |2 \cdot 3 - 0 \cdot 1| = 6$$

**分布：** 由于线性变换将正态分布映射为正态分布，$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$，即：

$$\mathbf{x} \sim \mathcal{N}\!\left(\begin{pmatrix}1\\0\end{pmatrix},\ \begin{pmatrix}4 & 2\\2 & 10\end{pmatrix}\right)$$

密度函数中出现的归一化常数包含因子 $\frac{1}{\det(A)} = \frac{1}{6}$，与一维情形的 $\frac{1}{|f'|} = \frac{1}{2}$ 完全类比。

</details>
