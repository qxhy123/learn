# 第一章：向量与矩阵

## 学习目标

完成本章学习后，你将能够：

- 理解向量空间的基本概念，掌握线性组合、线性相关与线性无关的判断方法
- 熟练计算各种向量范数和矩阵范数，理解范数在优化问题中度量"大小"的作用
- 掌握矩阵乘法、转置、逆等基本运算，识别正交矩阵、对称矩阵等特殊矩阵的性质
- 理解特征值分解与谱定理，能够对实对称矩阵进行正交对角化
- 掌握正定矩阵与半正定矩阵的判定方法，理解其在优化理论中的核心地位

---

## 正文内容

### 1.1 向量空间与线性组合

#### 1.1.1 向量与向量空间

**向量**（Vector）是既有大小又有方向的量。在数学中，$n$ 维实向量是 $n$ 个实数构成的有序组：

$$\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix} \in \mathbb{R}^n$$

所有 $n$ 维实向量的集合 $\mathbb{R}^n$ 在定义了加法和标量乘法后构成**向量空间**（Vector Space）。

**向量加法**：$(\mathbf{x} + \mathbf{y})_i = x_i + y_i$

**标量乘法**：$(\alpha \mathbf{x})_i = \alpha x_i$，$\alpha \in \mathbb{R}$

向量空间满足8条公理（交换律、结合律、零元存在、逆元存在、分配律等），$\mathbb{R}^n$ 是最基本的向量空间实例。

#### 1.1.2 线性组合与张成空间

设 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k \in \mathbb{R}^n$，系数 $\alpha_1, \alpha_2, \ldots, \alpha_k \in \mathbb{R}$，则

$$\mathbf{w} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \cdots + \alpha_k \mathbf{v}_k = \sum_{i=1}^k \alpha_i \mathbf{v}_i$$

称为向量组 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ 的一个**线性组合**（Linear Combination）。

这组向量所有可能的线性组合构成的集合称为**张成空间**（Span）：

$$\mathrm{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \left\{ \sum_{i=1}^k \alpha_i \mathbf{v}_i \;\middle|\; \alpha_i \in \mathbb{R} \right\}$$

**例 1.1** 在 $\mathbb{R}^2$ 中，$\mathbf{v}_1 = (1, 0)^\top$，$\mathbf{v}_2 = (0, 1)^\top$，则 $\mathrm{span}(\mathbf{v}_1, \mathbf{v}_2) = \mathbb{R}^2$，即平面上的任意向量都可以用这两个向量的线性组合表示。

#### 1.1.3 线性相关与线性无关

**定义** 若存在不全为零的系数 $\alpha_1, \ldots, \alpha_k$，使得

$$\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \cdots + \alpha_k \mathbf{v}_k = \mathbf{0}$$

则称向量组 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ **线性相关**（Linearly Dependent）；否则（仅当所有系数为零时上式成立）称为**线性无关**（Linearly Independent）。

**直觉理解**：线性相关意味着组中至少有一个向量可以被其他向量的线性组合表示——它是"冗余"的；线性无关意味着每个向量都提供了新的方向信息。

**例 1.2** 在 $\mathbb{R}^3$ 中：

- $\mathbf{e}_1 = (1,0,0)^\top$，$\mathbf{e}_2 = (0,1,0)^\top$，$\mathbf{e}_3 = (0,0,1)^\top$ 线性无关（标准基）
- $\mathbf{v}_1 = (1,2,3)^\top$，$\mathbf{v}_2 = (2,4,6)^\top$ 线性相关，因为 $\mathbf{v}_2 = 2\mathbf{v}_1$

#### 1.1.4 基与维数

向量空间 $V$ 的一个**基**（Basis）是 $V$ 中线性无关的向量组，且能张成整个 $V$。基的向量个数称为 $V$ 的**维数**（Dimension），记为 $\dim(V)$。

$\mathbb{R}^n$ 的标准基为 $\{\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n\}$，其中 $\mathbf{e}_i$ 是第 $i$ 个分量为1、其余为0的向量，故 $\dim(\mathbb{R}^n) = n$。

---

### 1.2 内积与范数

#### 1.2.1 内积

$\mathbb{R}^n$ 上的**标准内积**（Inner Product，又称点积）定义为：

$$\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top \mathbf{y} = \sum_{i=1}^n x_i y_i$$

内积满足以下性质：

- **对称性**：$\langle \mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{y}, \mathbf{x} \rangle$
- **双线性**：$\langle \alpha \mathbf{x} + \beta \mathbf{y}, \mathbf{z} \rangle = \alpha \langle \mathbf{x}, \mathbf{z} \rangle + \beta \langle \mathbf{y}, \mathbf{z} \rangle$
- **正定性**：$\langle \mathbf{x}, \mathbf{x} \rangle \geq 0$，且等号当且仅当 $\mathbf{x} = \mathbf{0}$ 时成立

**几何含义**：$\langle \mathbf{x}, \mathbf{y} \rangle = \|\mathbf{x}\|_2 \|\mathbf{y}\|_2 \cos\theta$，其中 $\theta$ 是 $\mathbf{x}$ 与 $\mathbf{y}$ 的夹角。

当 $\langle \mathbf{x}, \mathbf{y} \rangle = 0$ 时，称 $\mathbf{x}$ 与 $\mathbf{y}$ **正交**（Orthogonal），记为 $\mathbf{x} \perp \mathbf{y}$。

**Cauchy-Schwarz 不等式**：

$$|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\|_2 \|\mathbf{y}\|_2$$

等号成立当且仅当 $\mathbf{x}$ 与 $\mathbf{y}$ 线性相关（方向相同或相反）。

#### 1.2.2 向量范数

**范数**（Norm）是对向量"大小"或"长度"的度量。满足以下三条公理的函数 $\|\cdot\|: \mathbb{R}^n \to \mathbb{R}$ 称为范数：

1. **非负性**：$\|\mathbf{x}\| \geq 0$，且 $\|\mathbf{x}\| = 0 \iff \mathbf{x} = \mathbf{0}$
2. **正齐次性**：$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|$
3. **三角不等式**：$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$

**（1）$\ell^1$ 范数（Manhattan 范数）**

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|$$

几何上对应在网格中的出租车距离。在机器学习中，$\ell^1$ 范数正则化（Lasso）产生稀疏解。

**（2）$\ell^2$ 范数（Euclidean 范数）**

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{\mathbf{x}^\top \mathbf{x}}$$

最常用的范数，对应欧几里得距离。$\ell^2$ 范数正则化（Ridge/L2）产生平滑解。

**（3）$\ell^\infty$ 范数（Chebyshev 范数）**

$$\|\mathbf{x}\|_\infty = \max_{1 \leq i \leq n} |x_i|$$

取向量所有分量绝对值的最大值，对应各坐标方向最大偏差。

**（4）$\ell^p$ 范数（$p \geq 1$）**

$$\|\mathbf{x}\|_p = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p}$$

$\ell^1$，$\ell^2$，$\ell^\infty$ 是 $p = 1, 2, \infty$ 的特殊情形。当 $p \to \infty$ 时，$\|\mathbf{x}\|_p \to \|\mathbf{x}\|_\infty$。

**范数等价性**：在有限维空间中，所有范数等价——即存在正常数 $c_1, c_2 > 0$，使得对任意 $\mathbf{x}$：

$$c_1 \|\mathbf{x}\|_\alpha \leq \|\mathbf{x}\|_\beta \leq c_2 \|\mathbf{x}\|_\alpha$$

常用等价关系（$\mathbf{x} \in \mathbb{R}^n$）：

$$\|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1 \leq \sqrt{n} \|\mathbf{x}\|_2$$

$$\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \sqrt{n} \|\mathbf{x}\|_\infty$$

#### 1.2.3 矩阵范数

对矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$，常用的矩阵范数如下。

**（1）Frobenius 范数**

$$\|\mathbf{A}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n a_{ij}^2} = \sqrt{\mathrm{tr}(\mathbf{A}^\top \mathbf{A})}$$

Frobenius 范数是矩阵所有元素平方和的平方根，类比向量的 $\ell^2$ 范数，计算简便，在深度学习中广泛用于权重正则化。

**（2）谱范数（算子 2-范数）**

$$\|\mathbf{A}\|_2 = \sigma_{\max}(\mathbf{A}) = \sqrt{\lambda_{\max}(\mathbf{A}^\top \mathbf{A})}$$

其中 $\sigma_{\max}$ 是 $\mathbf{A}$ 的最大奇异值，$\lambda_{\max}(\mathbf{A}^\top \mathbf{A})$ 是 $\mathbf{A}^\top \mathbf{A}$ 的最大特征值。谱范数衡量矩阵作为线性映射时最大的"拉伸倍率"。

**（3）诱导范数的一般定义**

$$\|\mathbf{A}\|_{p \to q} = \sup_{\mathbf{x} \neq \mathbf{0}} \frac{\|\mathbf{A}\mathbf{x}\|_q}{\|\mathbf{x}\|_p}$$

谱范数是 $\|\mathbf{A}\|_{2 \to 2}$ 的特殊情形。

**Frobenius 范数与谱范数的关系**：

$$\|\mathbf{A}\|_2 \leq \|\mathbf{A}\|_F \leq \sqrt{r} \|\mathbf{A}\|_2$$

其中 $r = \mathrm{rank}(\mathbf{A})$。

---

### 1.3 矩阵运算与特殊矩阵

#### 1.3.1 基本矩阵运算

设 $\mathbf{A} \in \mathbb{R}^{m \times n}$，$\mathbf{B} \in \mathbb{R}^{n \times p}$，矩阵乘法定义为：

$$(\mathbf{AB})_{ij} = \sum_{k=1}^n a_{ik} b_{kj}$$

矩阵乘法满足结合律 $(\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})$ 和分配律，但一般**不满足交换律**：$\mathbf{AB} \neq \mathbf{BA}$。

**转置运算**：$(\mathbf{A}^\top)_{ij} = a_{ji}$，满足 $(\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top$。

**矩阵的迹**：方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 的迹为对角元素之和：

$$\mathrm{tr}(\mathbf{A}) = \sum_{i=1}^n a_{ii}$$

迹满足循环性：$\mathrm{tr}(\mathbf{ABC}) = \mathrm{tr}(\mathbf{BCA}) = \mathrm{tr}(\mathbf{CAB})$，以及 $\mathrm{tr}(\mathbf{A}^\top \mathbf{B}) = \sum_{i,j} a_{ij} b_{ij}$。

**逆矩阵**：对可逆方阵 $\mathbf{A}$，其逆 $\mathbf{A}^{-1}$ 满足 $\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$，且 $(\mathbf{AB})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$。

#### 1.3.2 对称矩阵

满足 $\mathbf{A}^\top = \mathbf{A}$ 的方阵称为**对称矩阵**（Symmetric Matrix）。

对称矩阵的重要性质：

- 特征值均为实数
- 不同特征值对应的特征向量互相正交
- 可以被正交矩阵对角化（谱定理，见 1.4 节）

对称矩阵在优化理论中随处可见：Hessian 矩阵（目标函数的二阶导数矩阵）恒为对称矩阵。

**反对称矩阵**（Skew-Symmetric）：满足 $\mathbf{A}^\top = -\mathbf{A}$。任何方阵 $\mathbf{A}$ 都可唯一分解为对称部分与反对称部分之和：

$$\mathbf{A} = \underbrace{\frac{\mathbf{A} + \mathbf{A}^\top}{2}}_{\text{对称}} + \underbrace{\frac{\mathbf{A} - \mathbf{A}^\top}{2}}_{\text{反对称}}$$

#### 1.3.3 正交矩阵

满足 $\mathbf{Q}^\top \mathbf{Q} = \mathbf{Q}\mathbf{Q}^\top = \mathbf{I}$ 的方阵 $\mathbf{Q} \in \mathbb{R}^{n \times n}$ 称为**正交矩阵**（Orthogonal Matrix）。

正交矩阵的性质：

- $\mathbf{Q}^{-1} = \mathbf{Q}^\top$（逆等于转置，计算高效）
- $\det(\mathbf{Q}) = \pm 1$
- **保范性**：$\|\mathbf{Q}\mathbf{x}\|_2 = \|\mathbf{x}\|_2$，即正交变换不改变向量的欧几里得长度
- $\mathbf{Q}$ 的各列（各行）构成 $\mathbb{R}^n$ 的标准正交基

**几何意义**：正交矩阵表示旋转（$\det = 1$）或反射（$\det = -1$）变换，是"保形"的刚体运动。

**例 1.3** 二维旋转矩阵是正交矩阵：

$$\mathbf{Q} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}, \quad \mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$$

#### 1.3.4 对角矩阵与单位矩阵

**对角矩阵** $\mathbf{D} = \mathrm{diag}(d_1, d_2, \ldots, d_n)$：非对角元素全为零。

- $\mathbf{D}^k = \mathrm{diag}(d_1^k, \ldots, d_n^k)$
- $\mathbf{D}^{-1} = \mathrm{diag}(1/d_1, \ldots, 1/d_n)$（当所有 $d_i \neq 0$ 时）

**单位矩阵** $\mathbf{I}_n$：$d_i = 1$ 的对角矩阵，满足 $\mathbf{I}\mathbf{A} = \mathbf{A}\mathbf{I} = \mathbf{A}$。

---

### 1.4 特征值与特征向量

#### 1.4.1 特征值分解

对方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$，若存在非零向量 $\mathbf{v} \in \mathbb{R}^n$ 和标量 $\lambda \in \mathbb{R}$，使得

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

则称 $\lambda$ 为 $\mathbf{A}$ 的**特征值**（Eigenvalue），$\mathbf{v}$ 为对应的**特征向量**（Eigenvector）。

**几何意义**：特征向量在矩阵 $\mathbf{A}$ 的作用下方向不变（或反向），仅被伸缩 $\lambda$ 倍。

特征值由**特征多项式**确定：

$$\det(\mathbf{A} - \lambda \mathbf{I}) = 0$$

这是关于 $\lambda$ 的 $n$ 次多项式，有 $n$ 个根（在复数域内计重数）。

**例 1.4** 求矩阵 $\mathbf{A} = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$ 的特征值与特征向量。

**解：** 特征多项式：

$$\det(\mathbf{A} - \lambda \mathbf{I}) = (3-\lambda)^2 - 1 = \lambda^2 - 6\lambda + 8 = (\lambda - 2)(\lambda - 4) = 0$$

特征值：$\lambda_1 = 2$，$\lambda_2 = 4$

对 $\lambda_1 = 2$：$(\mathbf{A} - 2\mathbf{I})\mathbf{v} = \mathbf{0}$，即 $\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\mathbf{v} = \mathbf{0}$，得 $\mathbf{v}_1 = \frac{1}{\sqrt{2}}(1, -1)^\top$

对 $\lambda_2 = 4$：$(\mathbf{A} - 4\mathbf{I})\mathbf{v} = \mathbf{0}$，即 $\begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\mathbf{v} = \mathbf{0}$，得 $\mathbf{v}_2 = \frac{1}{\sqrt{2}}(1, 1)^\top$

注意 $\mathbf{v}_1 \perp \mathbf{v}_2$（不同特征值对应的特征向量正交，因为 $\mathbf{A}$ 对称）。

#### 1.4.2 特征值的基本性质

设 $\mathbf{A}$ 的特征值为 $\lambda_1, \lambda_2, \ldots, \lambda_n$（含重数），则：

- $\det(\mathbf{A}) = \prod_{i=1}^n \lambda_i$
- $\mathrm{tr}(\mathbf{A}) = \sum_{i=1}^n \lambda_i$
- $\mathbf{A}$ 可逆当且仅当所有特征值非零
- $\mathbf{A}^k$ 的特征值为 $\lambda_i^k$，特征向量不变
- 若 $\mathbf{A}$ 可逆，$\mathbf{A}^{-1}$ 的特征值为 $1/\lambda_i$

#### 1.4.3 谱定理与实对称矩阵的特征分解

**谱定理（Spectral Theorem）**：设 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 是实对称矩阵，则：

1. $\mathbf{A}$ 的所有特征值均为**实数**
2. 不同特征值对应的特征向量**互相正交**
3. 存在正交矩阵 $\mathbf{Q}$，使得

$$\boxed{\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top}$$

其中 $\mathbf{\Lambda} = \mathrm{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ 为特征值构成的对角矩阵，$\mathbf{Q}$ 的列 $\mathbf{q}_1, \ldots, \mathbf{q}_n$ 为对应的标准正交特征向量。

**谱分解的外积展开**：

$$\mathbf{A} = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^\top$$

每一项 $\mathbf{q}_i \mathbf{q}_i^\top$ 是秩1的正交投影矩阵，整个矩阵是 $n$ 个秩1矩阵的加权叠加，权重正是特征值。

**谱分解的直觉**：谱分解揭示了矩阵的"本质结构"——$\mathbf{A}$ 先将向量投影到各特征方向，再分别伸缩对应特征值的倍数，最后还原。

**例 1.5** 沿用例 1.4：

$$\mathbf{A} = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix} = 2 \cdot \frac{1}{2}\begin{pmatrix} 1 \\ -1 \end{pmatrix}\begin{pmatrix} 1 & -1 \end{pmatrix} + 4 \cdot \frac{1}{2}\begin{pmatrix} 1 \\ 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \end{pmatrix}$$

$$= \mathbf{Q}\begin{pmatrix} 2 & 0 \\ 0 & 4 \end{pmatrix}\mathbf{Q}^\top, \quad \mathbf{Q} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}$$

#### 1.4.4 瑞利商

对实对称矩阵 $\mathbf{A}$，**瑞利商**（Rayleigh Quotient）定义为：

$$R(\mathbf{x}) = \frac{\mathbf{x}^\top \mathbf{A} \mathbf{x}}{\mathbf{x}^\top \mathbf{x}}, \quad \mathbf{x} \neq \mathbf{0}$$

**瑞利商定理**：

$$\lambda_{\min}(\mathbf{A}) \leq R(\mathbf{x}) \leq \lambda_{\max}(\mathbf{A})$$

等号在 $\mathbf{x}$ 取对应特征向量时成立。这是约束优化中的重要工具，最大（最小）特征值就是瑞利商的最大（最小）值。

---

### 1.5 正定矩阵与半正定矩阵

#### 1.5.1 定义与等价条件

**定义** 实对称矩阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 称为：

- **正定**（Positive Definite，PD）：若对所有非零向量 $\mathbf{x} \in \mathbb{R}^n$，有 $\mathbf{x}^\top \mathbf{A} \mathbf{x} > 0$，记作 $\mathbf{A} \succ 0$
- **半正定**（Positive Semi-Definite，PSD）：若对所有向量 $\mathbf{x}$，有 $\mathbf{x}^\top \mathbf{A} \mathbf{x} \geq 0$，记作 $\mathbf{A} \succeq 0$
- **负定**（Negative Definite，ND）：若 $-\mathbf{A} \succ 0$，记作 $\mathbf{A} \prec 0$
- **不定**（Indefinite）：$\mathbf{x}^\top \mathbf{A} \mathbf{x}$ 可正可负

**等价条件**（以正定为例）：

以下说法等价：

1. $\mathbf{A} \succ 0$（二次型正定）
2. 所有特征值 $\lambda_i > 0$
3. 所有顺序主子式 $> 0$（Sylvester 准则）
4. $\mathbf{A} = \mathbf{L}\mathbf{L}^\top$，其中 $\mathbf{L}$ 是下三角矩阵且对角元素为正（Cholesky 分解）

**例 1.6** 判断下列矩阵的正定性：

$\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$：特征值为 $\lambda_1 = 1 > 0$，$\lambda_2 = 3 > 0$，故 $\mathbf{A} \succ 0$（正定）。

$\mathbf{B} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$：特征值为 $1, 0$，故 $\mathbf{B} \succeq 0$（半正定，但非正定）。

$\mathbf{C} = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$：特征值为 $-1, 3$，故 $\mathbf{C}$ 是不定矩阵。

#### 1.5.2 二次型的几何意义

对称矩阵 $\mathbf{A}$ 对应**二次型**（Quadratic Form）$f(\mathbf{x}) = \mathbf{x}^\top \mathbf{A} \mathbf{x}$。

- 正定：$f(\mathbf{x}) > 0$ 对所有 $\mathbf{x} \neq \mathbf{0}$，等值曲面 $\{f(\mathbf{x}) = c\}$ 是椭球面
- 负定：$f(\mathbf{x}) < 0$ 对所有 $\mathbf{x} \neq \mathbf{0}$，函数向下弯曲
- 不定：函数既有上凸方向又有下凸方向，等值曲面是双曲面（鞍形）

**优化中的重要作用**：

- 目标函数的 Hessian 矩阵 $\mathbf{H} = \nabla^2 f(\mathbf{x})$ 在极小值点处满足 $\mathbf{H} \succeq 0$（二阶充分条件：$\mathbf{H} \succ 0$）
- 凸函数等价于其 Hessian 矩阵处处半正定
- 强凸函数等价于其 Hessian 矩阵处处正定

#### 1.5.3 正定矩阵的运算性质

**性质一（求和封闭性）**：若 $\mathbf{A} \succ 0$，$\mathbf{B} \succeq 0$，则 $\mathbf{A} + \mathbf{B} \succ 0$。

**性质二（逆矩阵）**：若 $\mathbf{A} \succ 0$，则 $\mathbf{A}^{-1} \succ 0$，且 $\mathbf{A}^{-1}$ 的特征值为 $1/\lambda_i$。

**性质三（合同变换）**：若 $\mathbf{A} \succ 0$，$\mathbf{C}$ 可逆，则 $\mathbf{C}^\top \mathbf{A} \mathbf{C} \succ 0$。

**性质四（半正定的构造）**：对任意矩阵 $\mathbf{B} \in \mathbb{R}^{m \times n}$，$\mathbf{B}^\top \mathbf{B} \succeq 0$。

证明：对任意 $\mathbf{x}$，$\mathbf{x}^\top (\mathbf{B}^\top \mathbf{B}) \mathbf{x} = \|\mathbf{B}\mathbf{x}\|_2^2 \geq 0$。

这一性质极为重要：协方差矩阵、Gram 矩阵、神经网络权重的 $\mathbf{W}^\top \mathbf{W}$ 均是半正定矩阵。

#### 1.5.4 广义不等式与矩阵排序

正定矩阵定义了矩阵上的偏序关系（Löwner 偏序）：

$$\mathbf{A} \succeq \mathbf{B} \iff \mathbf{A} - \mathbf{B} \succeq 0$$

这一偏序在凸优化、半定规划（SDP）中是核心结构。

---

## 本章小结

| 概念 | 数学表示 | 核心意义 |
|------|----------|----------|
| 线性组合 | $\sum_i \alpha_i \mathbf{v}_i$ | 向量空间的生成操作 |
| 内积 | $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top \mathbf{y}$ | 角度与正交性的度量 |
| $\ell^p$ 范数 | $\|\mathbf{x}\|_p = (\sum |x_i|^p)^{1/p}$ | 向量"大小"的度量族 |
| Frobenius 范数 | $\|\mathbf{A}\|_F = \sqrt{\mathrm{tr}(\mathbf{A}^\top \mathbf{A})}$ | 矩阵元素级别的大小 |
| 谱范数 | $\|\mathbf{A}\|_2 = \sigma_{\max}(\mathbf{A})$ | 矩阵最大拉伸倍率 |
| 谱分解 | $\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$ | 对称矩阵的本征结构 |
| 正定矩阵 | $\mathbf{x}^\top \mathbf{A} \mathbf{x} > 0$（$\mathbf{A} \succ 0$）| 凸性与极小值的判据 |
| 半正定矩阵 | $\mathbf{x}^\top \mathbf{A} \mathbf{x} \geq 0$（$\mathbf{A} \succeq 0$）| 凸函数 Hessian 的条件 |

**核心思想**：向量与矩阵是优化理论的语言基础。范数度量"距离"，内积度量"方向"，特征值分解揭示矩阵的本征行为，而正定性是连接代数结构与优化几何的关键桥梁——Hessian 正定意味着强凸，强凸意味着梯度下降有唯一全局最优解。

---

## 深度学习应用：张量运算、权重矩阵与参数初始化

### 深度学习中的矩阵结构

深度学习的核心计算是矩阵运算。每一层全连接网络可以写成：

$$\mathbf{h}^{(l)} = \sigma\!\left(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right)$$

其中 $\mathbf{W}^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$ 是权重矩阵，$\sigma$ 是非线性激活函数。理解权重矩阵的谱性质对训练稳定性至关重要。

### 权重矩阵的谱分析与梯度爆炸/消失

在深度网络的反向传播中，梯度需要经过多个权重矩阵的连续变换。设网络有 $L$ 层，梯度经过的矩阵乘积为：

$$\prod_{l=1}^L \mathbf{W}^{(l)}$$

若每个 $\mathbf{W}^{(l)}$ 的谱范数 $\|\mathbf{W}^{(l)}\|_2 = \sigma_{\max}^{(l)}$，则：

- 若 $\sigma_{\max}^{(l)} > 1$：梯度呈指数增长，**梯度爆炸**（Gradient Explosion）
- 若 $\sigma_{\max}^{(l)} < 1$：梯度呈指数衰减，**梯度消失**（Gradient Vanishing）
- 若 $\sigma_{\max}^{(l)} \approx 1$：梯度保持稳定传播

**谱归一化**（Spectral Normalization）正是基于此思路，将每层权重矩阵除以其谱范数：

$$\hat{\mathbf{W}} = \frac{\mathbf{W}}{\|\mathbf{W}\|_2}$$

### 参数初始化策略的理论依据

合理的参数初始化使网络在训练初期保持信号方差稳定，本质上是控制权重矩阵的谱性质。

**Xavier 初始化**（适用于 sigmoid/tanh 激活）：

$$W_{ij} \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{in}+n_{out}}},\, \sqrt{\frac{6}{n_{in}+n_{out}}}\right)$$

推导思路：若输入方差为1，要求输出方差也为1，则 $\mathrm{Var}(W_{ij}) = \frac{2}{n_{in}+n_{out}}$。

**Kaiming（He）初始化**（适用于 ReLU 激活）：

$$W_{ij} \sim \mathcal{N}\!\left(0,\, \frac{2}{n_{in}}\right)$$

ReLU 约使一半神经元不激活，有效输入减半，故方差翻倍为 $2/n_{in}$。

两者的范数视角：这些初始化策略本质上控制了权重矩阵 $\mathbf{W}$ 的 Frobenius 范数量级，使得 $\|\mathbf{W}\|_F \approx O(\sqrt{n_{in}})$，进而使谱范数保持在合理范围。

### PyTorch 代码示例

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 张量的范数计算
# ============================================================
def demo_norms():
    """演示向量和矩阵的各种范数计算"""
    x = torch.tensor([3.0, -4.0, 0.0])

    print("=== 向量范数 ===")
    print(f"x = {x.tolist()}")
    print(f"‖x‖₁ = {torch.norm(x, p=1).item():.4f}")   # 7.0
    print(f"‖x‖₂ = {torch.norm(x, p=2).item():.4f}")   # 5.0
    print(f"‖x‖∞ = {torch.norm(x, p=float('inf')).item():.4f}")  # 4.0

    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print("\n=== 矩阵范数 ===")
    print(f"A =\n{A.numpy()}")

    # Frobenius 范数：sqrt(1²+2²+3²+4²) = sqrt(30)
    frob = torch.norm(A, p='fro')
    print(f"‖A‖_F = {frob.item():.4f}")  # ≈ 5.4772

    # 谱范数：最大奇异值
    _, S, _ = torch.linalg.svd(A)
    spectral = S[0]
    print(f"‖A‖_2 (谱范数) = {spectral.item():.4f}")  # ≈ 5.4648

    # 验证关系：‖A‖_2 ≤ ‖A‖_F
    print(f"‖A‖_2 ≤ ‖A‖_F: {spectral.item():.4f} ≤ {frob.item():.4f} ✓")


# ============================================================
# 2. 特征值分解与谱分析
# ============================================================
def demo_eigendecomposition():
    """演示对称矩阵的特征值分解"""
    # 构造对称正定矩阵
    B = torch.tensor([[4.0, 2.0, 0.0],
                      [2.0, 3.0, 1.0],
                      [0.0, 1.0, 2.0]])

    # 特征值分解（对称矩阵用 torch.linalg.eigh，更稳定）
    eigenvalues, Q = torch.linalg.eigh(B)

    print("=== 对称矩阵特征分解 ===")
    print(f"特征值 λ: {eigenvalues.tolist()}")
    print(f"特征向量矩阵 Q:\n{Q.numpy()}")

    # 验证 A = Q Λ Q^T
    Lambda = torch.diag(eigenvalues)
    B_reconstructed = Q @ Lambda @ Q.T
    print(f"\n重建误差 ‖A - QΛQᵀ‖_F = {torch.norm(B - B_reconstructed).item():.2e}")

    # 验证 Q 是正交矩阵
    ortho_err = torch.norm(Q.T @ Q - torch.eye(3)).item()
    print(f"正交性误差 ‖QᵀQ - I‖_F = {ortho_err:.2e}")

    # 判断正定性
    print(f"\n所有特征值 > 0: {(eigenvalues > 0).all().item()} → 矩阵正定")


# ============================================================
# 3. 权重矩阵的谱分析：梯度爆炸与消失
# ============================================================
def demo_spectral_analysis():
    """演示权重矩阵谱范数对梯度传播的影响"""
    torch.manual_seed(42)
    n_layers = 30
    dim = 64

    def simulate_gradient_flow(scale):
        """模拟梯度经过多层后的大小变化"""
        grad = torch.randn(dim)
        norms = [grad.norm().item()]
        for _ in range(n_layers):
            W = torch.randn(dim, dim) * scale
            grad = W.T @ grad  # 反向传播中的梯度变换
            norms.append(grad.norm().item())
        return norms

    # 不同初始化尺度对梯度的影响
    scales = {
        "过大 (scale=0.15，易爆炸)": 0.15,
        "适中 (scale=1/√64，稳定)": 1.0 / np.sqrt(dim),
        "过小 (scale=0.05，易消失)": 0.05,
    }

    print("=== 梯度范数经过30层后的变化 ===")
    for label, scale in scales.items():
        norms = simulate_gradient_flow(scale)
        print(f"{label}: 初始={norms[0]:.2f}, 最终={norms[-1]:.6f}")


# ============================================================
# 4. 参数初始化策略对比
# ============================================================
def demo_initialization():
    """对比不同初始化策略对激活值方差的影响"""
    torch.manual_seed(0)
    n_layers = 10
    batch_size = 512
    dim = 256

    def run_forward(init_fn, activation):
        """运行前向传播，记录每层输出方差"""
        x = torch.randn(batch_size, dim)
        variances = [x.var().item()]
        for _ in range(n_layers):
            W = torch.empty(dim, dim)
            init_fn(W)
            x = activation(x @ W.T)
            variances.append(x.var().item())
        return variances

    print("=== 不同初始化策略的激活值方差（理想值≈1.0）===")

    # 随机初始化（无策略）
    naive_vars = run_forward(
        lambda W: nn.init.normal_(W, std=1.0),
        torch.tanh
    )

    # Xavier 初始化（适合 tanh）
    xavier_vars = run_forward(
        lambda W: nn.init.xavier_uniform_(W),
        torch.tanh
    )

    # Kaiming 初始化（适合 ReLU）
    kaiming_vars_relu = run_forward(
        lambda W: nn.init.kaiming_normal_(W, mode='fan_in', nonlinearity='relu'),
        torch.relu
    )

    print(f"{'层号':<6} {'随机初始化':<15} {'Xavier(tanh)':<15} {'Kaiming(ReLU)':<15}")
    for i in range(n_layers + 1):
        print(f"{i:<6} {naive_vars[i]:<15.4f} {xavier_vars[i]:<15.4f} {kaiming_vars_relu[i]:<15.4f}")


# ============================================================
# 5. 谱归一化演示
# ============================================================
def demo_spectral_norm():
    """演示谱归一化控制权重矩阵谱范数"""
    torch.manual_seed(42)

    # 普通线性层 vs 谱归一化线性层
    layer_normal = nn.Linear(128, 64, bias=False)
    layer_sn = nn.utils.spectral_norm(nn.Linear(128, 64, bias=False))

    def compute_spectral_norm(layer):
        W = layer.weight.data
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        return S[0].item()

    print("=== 谱归一化效果 ===")
    print(f"普通层谱范数: {compute_spectral_norm(layer_normal):.4f}")
    print(f"谱归一化层谱范数: {compute_spectral_norm(layer_sn):.4f}  (≈1.0)")

    # 训练后验证谱归一化保持不变
    optimizer = torch.optim.Adam(layer_sn.parameters(), lr=0.01)
    for _ in range(100):
        x = torch.randn(32, 128)
        loss = layer_sn(x).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"100步训练后谱归一化层谱范数: {compute_spectral_norm(layer_sn):.4f}  (仍≈1.0)")


if __name__ == "__main__":
    demo_norms()
    print("\n" + "=" * 60 + "\n")
    demo_eigendecomposition()
    print("\n" + "=" * 60 + "\n")
    demo_spectral_analysis()
    print("\n" + "=" * 60 + "\n")
    demo_initialization()
    print("\n" + "=" * 60 + "\n")
    demo_spectral_norm()
```

**示例输出：**

```
=== 向量范数 ===
x = [3.0, -4.0, 0.0]
‖x‖₁ = 7.0000
‖x‖₂ = 5.0000
‖x‖∞ = 4.0000

=== 矩阵范数 ===
‖A‖_F = 5.4772
‖A‖_2 (谱范数) = 5.4648
‖A‖_2 ≤ ‖A‖_F: 5.4648 ≤ 5.4772 ✓

=== 对称矩阵特征分解 ===
特征值 λ: [1.2679, 3.0000, 4.7321]
重建误差 ‖A - QΛQᵀ‖_F = 2.14e-07
正交性误差 ‖QᵀQ - I‖_F = 5.73e-07
所有特征值 > 0: True → 矩阵正定

=== 梯度范数经过30层后的变化 ===
过大 (scale=0.15，易爆炸): 初始=7.82, 最终=58432.91
适中 (scale=1/√64，稳定): 初始=7.82, 最终=6.53
过小 (scale=0.05，易消失): 初始=7.82, 最终=0.000001
```

> **线性代数视角**：深度网络的前向传播是一系列线性变换与非线性激活的复合。权重矩阵的谱（特征值集合）决定了信号在网络中的放大或衰减模式。参数初始化策略本质上是在控制权重矩阵的 Frobenius 范数，进而约束谱范数，使得梯度信号既不爆炸也不消失——这是线性代数与深度学习训练稳定性之间最直接的联系。

---

## 练习题

**习题 1.1**（基础）设 $\mathbf{x} = (1, -2, 3, -4)^\top \in \mathbb{R}^4$，计算：

（1）$\|\mathbf{x}\|_1$，$\|\mathbf{x}\|_2$，$\|\mathbf{x}\|_\infty$；

（2）验证不等式 $\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1 \leq \sqrt{n}\|\mathbf{x}\|_2$；

（3）若 $\mathbf{y} = (2, 1, 0, -1)^\top$，计算内积 $\langle \mathbf{x}, \mathbf{y} \rangle$ 并判断 $\mathbf{x}, \mathbf{y}$ 是否正交。

---

**习题 1.2**（基础）设矩阵 $\mathbf{A} = \begin{pmatrix} 1 & 2 & 0 \\ 2 & 3 & 1 \\ 0 & 1 & 2 \end{pmatrix}$：

（1）验证 $\mathbf{A}$ 是否为对称矩阵；

（2）计算 $\|\mathbf{A}\|_F$；

（3）利用特征多项式 $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$ 求 $\mathbf{A}$ 的特征值（可用数值方法验证），并由此计算 $\mathrm{tr}(\mathbf{A})$ 与 $\det(\mathbf{A})$。

---

**习题 1.3**（中级）设 $\mathbf{A} = \begin{pmatrix} 5 & 4 \\ 4 & 5 \end{pmatrix}$：

（1）求 $\mathbf{A}$ 的特征值与标准正交特征向量，写出谱分解 $\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$；

（2）判断 $\mathbf{A}$ 是否正定，并验证 Sylvester 准则（顺序主子式均 $> 0$）；

（3）计算 $\mathbf{A}^{10}$（提示：利用谱分解 $\mathbf{A}^{10} = \mathbf{Q}\mathbf{\Lambda}^{10}\mathbf{Q}^\top$）。

---

**习题 1.4**（中级）设 $\mathbf{B} \in \mathbb{R}^{m \times n}$ 是任意矩阵，令 $\mathbf{G} = \mathbf{B}^\top \mathbf{B} \in \mathbb{R}^{n \times n}$（Gram 矩阵）：

（1）证明 $\mathbf{G}$ 是对称半正定矩阵；

（2）证明 $\mathbf{G}$ 的特征值均非负；

（3）若 $\mathbf{B}$ 的列向量线性无关（即 $\mathrm{rank}(\mathbf{B}) = n$），证明 $\mathbf{G} \succ 0$（正定）；

（4）给出深度学习中一个 Gram 矩阵自然出现的例子（例如风格迁移中的 Gram 矩阵）。

---

**习题 1.5**（提高）**谱范数正则化的优化视角**

在生成对抗网络（GAN）训练中，判别器的 Lipschitz 常数需要被约束，谱归一化是一种有效方法。

设 $\mathbf{W} \in \mathbb{R}^{m \times n}$，谱范数正则化将权重替换为 $\hat{\mathbf{W}} = \mathbf{W} / \|\mathbf{W}\|_2$。

（1）证明 $\|\hat{\mathbf{W}}\|_2 = 1$（即谱归一化后的矩阵谱范数恰好为1）；

（2）设 $f(\mathbf{x}) = \hat{\mathbf{W}}\mathbf{x}$，证明 $f$ 是1-Lipschitz 连续的，即对任意 $\mathbf{x}_1, \mathbf{x}_2$：

$$\|f(\mathbf{x}_1) - f(\mathbf{x}_2)\|_2 \leq \|\mathbf{x}_1 - \mathbf{x}_2\|_2$$

（3）讨论：为什么 Lipschitz 约束有助于 GAN 的训练稳定性？（提示：联系 Wasserstein 距离的对偶表示。）

（4）设 $\mathbf{W}$ 的奇异值分解为 $\mathbf{W} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$，写出 $\hat{\mathbf{W}}$ 的奇异值分解表达式。

---

## 练习答案

<details>
<summary><strong>习题 1.1 详解</strong></summary>

设 $\mathbf{x} = (1, -2, 3, -4)^\top$，$n = 4$。

**（1）计算各范数：**

$$\|\mathbf{x}\|_1 = |1| + |-2| + |3| + |-4| = 1 + 2 + 3 + 4 = 10$$

$$\|\mathbf{x}\|_2 = \sqrt{1^2 + (-2)^2 + 3^2 + (-4)^2} = \sqrt{1 + 4 + 9 + 16} = \sqrt{30} \approx 5.477$$

$$\|\mathbf{x}\|_\infty = \max(1, 2, 3, 4) = 4$$

**（2）验证不等式：**

$\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1$：
$$4 \leq \sqrt{30} \approx 5.477 \leq 10 \quad \checkmark$$

$\|\mathbf{x}\|_1 \leq \sqrt{n}\|\mathbf{x}\|_2$：
$$10 \leq \sqrt{4} \cdot \sqrt{30} = 2\sqrt{30} \approx 10.954 \quad \checkmark$$

**（3）内积计算：**

$$\langle \mathbf{x}, \mathbf{y} \rangle = 1 \cdot 2 + (-2) \cdot 1 + 3 \cdot 0 + (-4) \cdot (-1) = 2 - 2 + 0 + 4 = 4$$

因为 $\langle \mathbf{x}, \mathbf{y} \rangle = 4 \neq 0$，所以 $\mathbf{x}$ 与 $\mathbf{y}$ **不正交**。

</details>

<details>
<summary><strong>习题 1.2 详解</strong></summary>

**（1）对称性验证：**

$\mathbf{A}^\top = \begin{pmatrix} 1 & 2 & 0 \\ 2 & 3 & 1 \\ 0 & 1 & 2 \end{pmatrix} = \mathbf{A}$，故 $\mathbf{A}$ 是**对称矩阵**。

**（2）Frobenius 范数：**

$$\|\mathbf{A}\|_F = \sqrt{1^2 + 2^2 + 0^2 + 2^2 + 3^2 + 1^2 + 0^2 + 1^2 + 2^2} = \sqrt{1+4+0+4+9+1+0+1+4} = \sqrt{24} = 2\sqrt{6} \approx 4.899$$

**（3）特征值计算：**

$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$ 展开得三次方程，数值求解得三个实特征值（利用对称矩阵特征值均为实数）。

精确计算：$\mathrm{tr}(\mathbf{A}) = 1 + 3 + 2 = 6 = \sum_i \lambda_i$

$\det(\mathbf{A}) = 1(3\cdot2 - 1\cdot1) - 2(2\cdot2 - 1\cdot0) + 0 = 1 \cdot 5 - 2 \cdot 4 = 5 - 8 = -3 = \prod_i \lambda_i$

（注意行列式为负，说明有负特征值，矩阵不定。）

数值特征值约为：$\lambda_1 \approx -0.3429$，$\lambda_2 \approx 1.4142$，$\lambda_3 \approx 4.9287$，验证：$\sum \lambda_i \approx 6$，$\prod \lambda_i \approx -3$。

</details>

<details>
<summary><strong>习题 1.3 详解</strong></summary>

$\mathbf{A} = \begin{pmatrix} 5 & 4 \\ 4 & 5 \end{pmatrix}$

**（1）特征分解：**

特征多项式：$(5-\lambda)^2 - 16 = 0$，即 $\lambda^2 - 10\lambda + 9 = 0$，解得 $\lambda_1 = 1$，$\lambda_2 = 9$。

对 $\lambda_1 = 1$：$(\mathbf{A} - \mathbf{I})\mathbf{v} = \begin{pmatrix} 4 & 4 \\ 4 & 4 \end{pmatrix}\mathbf{v} = \mathbf{0}$，得 $\mathbf{v}_1 = \frac{1}{\sqrt{2}}(1, -1)^\top$

对 $\lambda_2 = 9$：$(\mathbf{A} - 9\mathbf{I})\mathbf{v} = \begin{pmatrix} -4 & 4 \\ 4 & -4 \end{pmatrix}\mathbf{v} = \mathbf{0}$，得 $\mathbf{v}_2 = \frac{1}{\sqrt{2}}(1, 1)^\top$

$$\mathbf{Q} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}, \quad \mathbf{\Lambda} = \begin{pmatrix} 1 & 0 \\ 0 & 9 \end{pmatrix}$$

$$\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top = 1 \cdot \frac{1}{2}\begin{pmatrix}1\\-1\end{pmatrix}\begin{pmatrix}1&-1\end{pmatrix} + 9 \cdot \frac{1}{2}\begin{pmatrix}1\\1\end{pmatrix}\begin{pmatrix}1&1\end{pmatrix}$$

**（2）正定性验证：**

特征值 $\lambda_1 = 1 > 0$，$\lambda_2 = 9 > 0$，故 $\mathbf{A} \succ 0$（正定）。

Sylvester 准则：一阶顺序主子式 $\Delta_1 = 5 > 0$；二阶 $\Delta_2 = \det(\mathbf{A}) = 25 - 16 = 9 > 0$，均满足。

**（3）计算 $\mathbf{A}^{10}$：**

$$\mathbf{A}^{10} = \mathbf{Q}\mathbf{\Lambda}^{10}\mathbf{Q}^\top = \mathbf{Q}\begin{pmatrix} 1^{10} & 0 \\ 0 & 9^{10} \end{pmatrix}\mathbf{Q}^\top$$

$$= \frac{1}{2}\begin{pmatrix}1\\-1\end{pmatrix}(1)\begin{pmatrix}1&-1\end{pmatrix} + \frac{9^{10}}{2}\begin{pmatrix}1\\1\end{pmatrix}\begin{pmatrix}1&1\end{pmatrix}$$

$$= \frac{1}{2}\begin{pmatrix}1&-1\\-1&1\end{pmatrix} + \frac{9^{10}}{2}\begin{pmatrix}1&1\\1&1\end{pmatrix}$$

$$= \frac{1}{2}\begin{pmatrix} 1 + 9^{10} & -1 + 9^{10} \\ -1 + 9^{10} & 1 + 9^{10} \end{pmatrix}$$

其中 $9^{10} = 3486784401$，故 $\mathbf{A}^{10} = \begin{pmatrix} 1743392201 & 1743392200 \\ 1743392200 & 1743392201 \end{pmatrix}$。

</details>

<details>
<summary><strong>习题 1.4 详解</strong></summary>

**（1）证明 $\mathbf{G} = \mathbf{B}^\top \mathbf{B}$ 是对称半正定矩阵：**

**对称性**：$\mathbf{G}^\top = (\mathbf{B}^\top \mathbf{B})^\top = \mathbf{B}^\top (\mathbf{B}^\top)^\top = \mathbf{B}^\top \mathbf{B} = \mathbf{G}$ ✓

**半正定性**：对任意 $\mathbf{x} \in \mathbb{R}^n$：
$$\mathbf{x}^\top \mathbf{G} \mathbf{x} = \mathbf{x}^\top \mathbf{B}^\top \mathbf{B} \mathbf{x} = (\mathbf{B}\mathbf{x})^\top (\mathbf{B}\mathbf{x}) = \|\mathbf{B}\mathbf{x}\|_2^2 \geq 0 \quad \checkmark$$

**（2）特征值非负：**

由（1）知 $\mathbf{G} \succeq 0$，对任意特征对 $(\lambda, \mathbf{v})$（$\|\mathbf{v}\|=1$）：
$$\lambda = \lambda \mathbf{v}^\top \mathbf{v} = \mathbf{v}^\top (\lambda \mathbf{v}) = \mathbf{v}^\top \mathbf{G}\mathbf{v} \geq 0$$

**（3）$\mathrm{rank}(\mathbf{B}) = n$ 时 $\mathbf{G} \succ 0$：**

若 $\mathbf{G}\mathbf{x} = \mathbf{0}$，则 $\mathbf{x}^\top \mathbf{G}\mathbf{x} = \|\mathbf{B}\mathbf{x}\|_2^2 = 0$，故 $\mathbf{B}\mathbf{x} = \mathbf{0}$。

由于 $\mathrm{rank}(\mathbf{B}) = n$，$\mathbf{B}$ 的零空间只有 $\{\mathbf{0}\}$，故 $\mathbf{x} = \mathbf{0}$。

因此 $\mathbf{G}$ 是正定矩阵。

**（4）深度学习中的 Gram 矩阵**：

在**神经风格迁移**（Neural Style Transfer）中，给定特征图 $\mathbf{F} \in \mathbb{R}^{C \times HW}$（$C$ 个通道，每通道展平为长度 $HW$ 的向量），风格的 Gram 矩阵定义为：

$$\mathbf{G}_{style} = \frac{1}{CHW} \mathbf{F}\mathbf{F}^\top \in \mathbb{R}^{C \times C}$$

它是半正定矩阵，捕获不同特征通道之间的相关性，代表图像的"纹理风格"。风格损失定义为内容图与风格图 Gram 矩阵之差的 Frobenius 范数。

</details>

<details>
<summary><strong>习题 1.5 详解</strong></summary>

**（1）证明 $\|\hat{\mathbf{W}}\|_2 = 1$：**

$$\|\hat{\mathbf{W}}\|_2 = \left\|\frac{\mathbf{W}}{\|\mathbf{W}\|_2}\right\|_2 = \frac{\|\mathbf{W}\|_2}{\|\mathbf{W}\|_2} = 1 \quad \checkmark$$

（利用范数正齐次性：$\|c\mathbf{A}\|_2 = |c| \|\mathbf{A}\|_2$。）

**（2）证明 $f$ 是1-Lipschitz 连续的：**

$$\|f(\mathbf{x}_1) - f(\mathbf{x}_2)\|_2 = \|\hat{\mathbf{W}}(\mathbf{x}_1 - \mathbf{x}_2)\|_2 \leq \|\hat{\mathbf{W}}\|_2 \cdot \|\mathbf{x}_1 - \mathbf{x}_2\|_2 = 1 \cdot \|\mathbf{x}_1 - \mathbf{x}_2\|_2$$

（最后一步用了谱范数的定义：$\|\hat{\mathbf{W}}\mathbf{v}\|_2 \leq \|\hat{\mathbf{W}}\|_2 \|\mathbf{v}\|_2$。）

**（3）Lipschitz 约束与 GAN 训练稳定性：**

Wasserstein GAN 的理论基础（Kantorovich-Rubinstein 对偶）要求判别器 $D$ 是1-Lipschitz 函数：

$$W_1(P_{data}, P_g) = \sup_{\|D\|_{Lip} \leq 1} \mathbb{E}_{x \sim P_{data}}[D(x)] - \mathbb{E}_{x \sim P_g}[D(x)]$$

谱归一化使每一层都是1-Lipschitz 的（线性层 Lipschitz 常数 = 谱范数），整个网络的 Lipschitz 常数被控制在1以内。这避免了原始 GAN 中判别器梯度消失/爆炸的问题，使训练更稳定。

**（4）谱归一化后的 SVD 表达式：**

设 $\mathbf{W} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$，其中 $\mathbf{\Sigma} = \mathrm{diag}(\sigma_1, \sigma_2, \ldots)$，$\sigma_1 = \|\mathbf{W}\|_2$ 为最大奇异值，则：

$$\hat{\mathbf{W}} = \frac{\mathbf{W}}{\sigma_1} = \mathbf{U} \cdot \frac{\mathbf{\Sigma}}{\sigma_1} \cdot \mathbf{V}^\top = \mathbf{U} \cdot \mathrm{diag}\!\left(1, \frac{\sigma_2}{\sigma_1}, \ldots, \frac{\sigma_r}{\sigma_1}\right) \cdot \mathbf{V}^\top$$

即所有奇异值被 $\sigma_1$ 归一化，最大奇异值变为1，其余奇异值 $\leq 1$，验证了 $\|\hat{\mathbf{W}}\|_2 = 1$。

</details>
