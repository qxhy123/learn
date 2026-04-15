# 第16章：内积与正交性

> **前置知识**：第9章（向量空间）、第10章（线性相关与线性无关）、第11章（基与维数）、第13章（线性映射）
>
> **本章难度**：★★★★☆
>
> **预计学习时间**：4-5 小时

---

## 学习目标

学完本章后，你将能够：

- 理解内积的严格公理化定义（正定性、对称性、线性性），并在具体向量空间中验证内积
- 掌握由内积诱导的范数与距离函数，理解它们的几何意义
- 判断向量是否正交，构造正交集与标准正交集，理解其在坐标计算中的便利性
- 理解正交补的定义与性质，建立子空间的直交分解
- 证明并应用 Cauchy-Schwarz 不等式与三角不等式，解释其几何含义

---

## 16.1 内积的定义

### 为什么需要内积？

向量空间的定义只涉及加法和标量乘法——它告诉我们如何"移动"向量，却没有告诉我们如何"度量"向量。要谈论长度、角度、垂直，就需要引入额外的结构。**内积**正是为此而生：它在向量空间上定义了一种"乘法"，使我们能够度量长度与方向。

### 内积的公理化定义

**定义**：设 $V$ 是实数域 $\mathbb{R}$ 上的向量空间。**内积（inner product）**是一个函数 $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$，满足以下四条公理（对所有 $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ 和 $c \in \mathbb{R}$）：

1. **正性（Positivity）**：$\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$

2. **正定性（Definiteness）**：$\langle \mathbf{v}, \mathbf{v} \rangle = 0 \iff \mathbf{v} = \mathbf{0}$

3. **对称性（Symmetry）**：$\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$

4. **第一变元的线性性（Linearity in the first argument）**：
$$\langle \mathbf{u} + \mathbf{w}, \mathbf{v} \rangle = \langle \mathbf{u}, \mathbf{v} \rangle + \langle \mathbf{w}, \mathbf{v} \rangle$$
$$\langle c\mathbf{u}, \mathbf{v} \rangle = c\langle \mathbf{u}, \mathbf{v} \rangle$$

具备内积结构的向量空间称为**内积空间（inner product space）**。

> **注意**：由对称性和第一变元的线性性，可以推导出第二变元同样满足线性性（双线性性）：$\langle \mathbf{u}, c\mathbf{v} + \mathbf{w} \rangle = c\langle \mathbf{u}, \mathbf{v} \rangle + \langle \mathbf{u}, \mathbf{w} \rangle$。

### 标准内积（点积）

在 $\mathbb{R}^n$ 中，最常用的内积是**标准内积**，即向量的**点积（dot product）**：

$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = \mathbf{u}^T \mathbf{v}$$

**验证四条公理**（以 $\mathbb{R}^2$ 为例，$\mathbf{u} = (u_1, u_2)^T$，$\mathbf{v} = (v_1, v_2)^T$）：

1. 正性：$\langle \mathbf{v}, \mathbf{v} \rangle = v_1^2 + v_2^2 \geq 0$ ✓
2. 正定性：$v_1^2 + v_2^2 = 0 \iff v_1 = v_2 = 0 \iff \mathbf{v} = \mathbf{0}$ ✓
3. 对称性：$\sum u_i v_i = \sum v_i u_i$ ✓
4. 线性性：$\langle \mathbf{u} + \mathbf{w}, \mathbf{v} \rangle = \sum(u_i + w_i)v_i = \sum u_i v_i + \sum w_i v_i = \langle \mathbf{u}, \mathbf{v} \rangle + \langle \mathbf{w}, \mathbf{v} \rangle$ ✓

### 加权内积

标准内积并非唯一选择。给定正实数 $w_1, w_2, \ldots, w_n > 0$，可定义 $\mathbb{R}^n$ 上的**加权内积**：

$$\langle \mathbf{u}, \mathbf{v} \rangle_W = \sum_{i=1}^n w_i u_i v_i$$

权重 $w_i$ 体现了各分量的"重要程度"——在统计学中，这对应于不同维度的方差归一化。

### 函数内积

内积不局限于有限维向量。在连续函数空间 $C([a,b])$ 上，定义：

$$\langle f, g \rangle = \int_a^b f(x) g(x)\, dx$$

可验证这满足所有内积公理。Fourier 级数的正交性正是基于此内积。

### 矩阵内积（Frobenius 内积）

在 $m \times n$ 实矩阵空间 $\mathbb{R}^{m \times n}$ 上，**Frobenius 内积**定义为：

$$\langle A, B \rangle_F = \text{tr}(A^T B) = \sum_{i,j} a_{ij} b_{ij}$$

即将矩阵"展平"成向量后的点积。

---

## 16.2 范数与距离

### 由内积诱导的范数

有了内积，自然可以定义向量的"长度"。

**定义**：设 $V$ 是内积空间，向量 $\mathbf{v} \in V$ 的**范数（norm）**（也称**模长**）定义为：

$$\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}$$

在 $\mathbb{R}^n$ 的标准内积下，这给出熟悉的**欧几里得范数**：

$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

### 范数的基本性质

**命题**：由内积诱导的范数满足：

1. **正性**：$\|\mathbf{v}\| \geq 0$，且 $\|\mathbf{v}\| = 0 \iff \mathbf{v} = \mathbf{0}$
2. **齐次性**：$\|c\mathbf{v}\| = |c|\,\|\mathbf{v}\|$
3. **三角不等式**：$\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$（将在 16.5 节证明）

**证明（齐次性）**：

$$\|c\mathbf{v}\| = \sqrt{\langle c\mathbf{v}, c\mathbf{v} \rangle} = \sqrt{c^2 \langle \mathbf{v}, \mathbf{v} \rangle} = |c|\sqrt{\langle \mathbf{v}, \mathbf{v} \rangle} = |c|\,\|\mathbf{v}\| \quad \square$$

### 单位向量与归一化

**定义**：范数为 1 的向量称为**单位向量（unit vector）**。

将任意非零向量除以其范数，得到同方向的单位向量，称为**归一化（normalization）**：

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

### 距离函数

**定义**：内积空间中，两向量 $\mathbf{u}$ 和 $\mathbf{v}$ 之间的**距离（distance）**定义为：

$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\| = \sqrt{\langle \mathbf{u} - \mathbf{v},\, \mathbf{u} - \mathbf{v} \rangle}$$

**距离的性质**（度量公理）：

1. **非负性**：$d(\mathbf{u}, \mathbf{v}) \geq 0$，且 $d(\mathbf{u}, \mathbf{v}) = 0 \iff \mathbf{u} = \mathbf{v}$
2. **对称性**：$d(\mathbf{u}, \mathbf{v}) = d(\mathbf{v}, \mathbf{u})$
3. **三角不等式**：$d(\mathbf{u}, \mathbf{w}) \leq d(\mathbf{u}, \mathbf{v}) + d(\mathbf{v}, \mathbf{w})$

**几何直觉**：范数是从原点出发的"距离"，距离函数则是任意两点之间的"直线距离"。距离函数使得每个内积空间自然成为一个**度量空间（metric space）**。

### 夹角公式

内积还刻画了向量之间的**夹角**。由 16.5 节将要证明的 Cauchy-Schwarz 不等式，可以保证：

$$-1 \leq \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\|\,\|\mathbf{v}\|} \leq 1$$

因此可以定义两非零向量之间的**夹角** $\theta \in [0, \pi]$：

$$\boxed{\cos\theta = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\|\,\|\mathbf{v}\|}}$$

这与高中几何中"向量点积等于模长乘积乘以夹角余弦"完全一致。

---

## 16.3 正交性

### 正交向量

**定义**：若 $\langle \mathbf{u}, \mathbf{v} \rangle = 0$，则称 $\mathbf{u}$ 与 $\mathbf{v}$ **正交（orthogonal）**，记作 $\mathbf{u} \perp \mathbf{v}$。

由夹角公式，$\langle \mathbf{u}, \mathbf{v} \rangle = 0$ 意味着 $\cos\theta = 0$，即 $\theta = 90°$——这正是几何意义上的"垂直"。

**特别规定**：零向量与任何向量正交（$\langle \mathbf{0}, \mathbf{v} \rangle = 0$ 对所有 $\mathbf{v}$ 成立）。

### 勾股定理（Pythagorean Theorem）

**定理**：若 $\mathbf{u} \perp \mathbf{v}$，则

$$\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$$

**证明**：

$$\|\mathbf{u} + \mathbf{v}\|^2 = \langle \mathbf{u} + \mathbf{v}, \mathbf{u} + \mathbf{v} \rangle = \langle \mathbf{u}, \mathbf{u} \rangle + 2\langle \mathbf{u}, \mathbf{v} \rangle + \langle \mathbf{v}, \mathbf{v} \rangle = \|\mathbf{u}\|^2 + 0 + \|\mathbf{v}\|^2 \quad \square$$

这正是平面几何中勾股定理的内积版本。

### 正交集

**定义**：向量集合 $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ 称为**正交集（orthogonal set）**，如果其中任意两个不同向量都正交：

$$\langle \mathbf{v}_i, \mathbf{v}_j \rangle = 0, \quad \forall i \neq j$$

**命题**：正交集中的非零向量线性无关。

**证明**：设 $c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_k \mathbf{v}_k = \mathbf{0}$，对两边取与 $\mathbf{v}_i$ 的内积：

$$\left\langle c_1 \mathbf{v}_1 + \cdots + c_k \mathbf{v}_k,\ \mathbf{v}_i \right\rangle = 0$$

$$c_1 \langle \mathbf{v}_1, \mathbf{v}_i \rangle + \cdots + c_k \langle \mathbf{v}_k, \mathbf{v}_i \rangle = 0$$

由正交性，当 $j \neq i$ 时 $\langle \mathbf{v}_j, \mathbf{v}_i \rangle = 0$，故只剩 $c_i \langle \mathbf{v}_i, \mathbf{v}_i \rangle = 0$。

由于 $\mathbf{v}_i \neq \mathbf{0}$，有 $\langle \mathbf{v}_i, \mathbf{v}_i \rangle > 0$，所以 $c_i = 0$。对所有 $i$ 成立，故线性无关。$\square$

**几何直觉**：正交意味着"互相独立的方向"。正交集中的向量从不同的"垂直方向"张成空间，彼此之间没有"重叠"，因而线性无关。

### 标准正交集

**定义**：若正交集中每个向量都是单位向量，则称之为**标准正交集（orthonormal set）**：

$$\langle \mathbf{v}_i, \mathbf{v}_j \rangle = \delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

其中 $\delta_{ij}$ 是 **Kronecker delta**。

**标准正交基（orthonormal basis）**：若标准正交集同时是空间的基，则称之为**标准正交基**。

**标准正交基的优越性**：设 $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$ 是 $V$ 的标准正交基，任意向量 $\mathbf{v} \in V$ 的坐标可以直接用内积计算：

$$\mathbf{v} = \sum_{i=1}^n \langle \mathbf{v}, \mathbf{e}_i \rangle\, \mathbf{e}_i$$

这比一般基的坐标计算（需解线性方程组）简单得多。

**例**：$\mathbb{R}^3$ 的标准基 $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\}$ 是标准正交基，$\langle \mathbf{e}_i, \mathbf{e}_j \rangle = \delta_{ij}$。

**例**：$\mathbb{R}^2$ 中，$\left\{\dfrac{1}{\sqrt{2}}\begin{pmatrix}1\\1\end{pmatrix},\ \dfrac{1}{\sqrt{2}}\begin{pmatrix}1\\-1\end{pmatrix}\right\}$ 也是标准正交基（旋转 $45°$ 的坐标轴）。

### 正交投影

设 $\hat{\mathbf{u}}$ 是单位向量，向量 $\mathbf{v}$ 在 $\hat{\mathbf{u}}$ 方向上的**正交投影**为：

$$\text{proj}_{\hat{\mathbf{u}}}(\mathbf{v}) = \langle \mathbf{v}, \hat{\mathbf{u}} \rangle\, \hat{\mathbf{u}}$$

对非单位向量 $\mathbf{u}$，投影公式为：

$$\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\langle \mathbf{v}, \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle}\, \mathbf{u} = \frac{\langle \mathbf{v}, \mathbf{u} \rangle}{\|\mathbf{u}\|^2}\, \mathbf{u}$$

**几何意义**：$\text{proj}_{\mathbf{u}}(\mathbf{v})$ 是 $\mathbf{v}$ 在 $\mathbf{u}$ 方向上的"影子"。残差向量 $\mathbf{v} - \text{proj}_{\mathbf{u}}(\mathbf{v})$ 与 $\mathbf{u}$ 正交（可验证）。

---

## 16.4 正交补

### 正交补的定义

**定义**：设 $W$ 是内积空间 $V$ 的子空间，$W$ 的**正交补（orthogonal complement）**定义为：

$$W^\perp = \{\mathbf{v} \in V \mid \langle \mathbf{v}, \mathbf{w} \rangle = 0 \text{ 对所有 } \mathbf{w} \in W\}$$

即 $W^\perp$ 是 $V$ 中所有与 $W$ 的每个向量都正交的向量的集合。

**命题**：$W^\perp$ 是 $V$ 的子空间。

**证明**：
- 零向量：$\langle \mathbf{0}, \mathbf{w} \rangle = 0$ 对所有 $\mathbf{w}$，故 $\mathbf{0} \in W^\perp$ ✓
- 加法封闭：若 $\mathbf{v}_1, \mathbf{v}_2 \in W^\perp$，则 $\langle \mathbf{v}_1 + \mathbf{v}_2, \mathbf{w} \rangle = \langle \mathbf{v}_1, \mathbf{w} \rangle + \langle \mathbf{v}_2, \mathbf{w} \rangle = 0 + 0 = 0$ ✓
- 标量封闭：若 $\mathbf{v} \in W^\perp$，则 $\langle c\mathbf{v}, \mathbf{w} \rangle = c\langle \mathbf{v}, \mathbf{w} \rangle = 0$ ✓ $\square$

### 正交补的核心性质

设 $V = \mathbb{R}^n$，$W$ 是 $V$ 的子空间，则：

**性质 1（直和分解）**：$V = W \oplus W^\perp$

即：$V$ 中每个向量 $\mathbf{v}$ 可唯一分解为

$$\mathbf{v} = \mathbf{w} + \mathbf{w}^\perp, \quad \mathbf{w} \in W,\; \mathbf{w}^\perp \in W^\perp$$

**性质 2（维数公式）**：$\dim(W) + \dim(W^\perp) = \dim(V)$

**性质 3（双正交补）**：$(W^\perp)^\perp = W$

**性质 4（零交）**：$W \cap W^\perp = \{\mathbf{0}\}$

**几何直觉（以 $\mathbb{R}^3$ 为例）**：

- 若 $W$ 是 $\mathbb{R}^3$ 中的一个平面（过原点），则 $W^\perp$ 是垂直于该平面的直线（过原点）。
- 若 $W$ 是一条直线（过原点），则 $W^\perp$ 是垂直于该直线的平面（过原点）。
- 每个 $\mathbb{R}^3$ 中的向量都唯一地分解为"平面内分量"与"法向量分量"之和。

**示例**：设 $W = \text{span}\!\left\{\begin{pmatrix}1\\0\\0\end{pmatrix}, \begin{pmatrix}0\\1\\0\end{pmatrix}\right\}$（$xy$ 平面），则

$$W^\perp = \left\{\begin{pmatrix}0\\0\\z\end{pmatrix} \mid z \in \mathbb{R}\right\} = \text{span}\!\left\{\begin{pmatrix}0\\0\\1\end{pmatrix}\right\}$$

这正是 $z$ 轴，与 $xy$ 平面垂直。

### 与矩阵子空间的联系

对矩阵 $A \in \mathbb{R}^{m \times n}$，有以下重要关系：

$$(\text{Col}(A))^\perp = \text{Null}(A^T), \qquad (\text{Row}(A))^\perp = \text{Null}(A)$$

这揭示了矩阵四个基本子空间之间的正交关系：行空间与零空间正交互补，列空间与左零空间正交互补。

---

## 16.5 Cauchy-Schwarz 不等式与三角不等式

### Cauchy-Schwarz 不等式

**定理（Cauchy-Schwarz 不等式）**：设 $V$ 是实内积空间，对所有 $\mathbf{u}, \mathbf{v} \in V$，有：

$$\boxed{|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\|\,\|\mathbf{v}\|}$$

等号成立当且仅当 $\mathbf{u}$ 与 $\mathbf{v}$ 线性相关（即一个是另一个的标量倍）。

**证明**：

若 $\mathbf{u} = \mathbf{0}$，不等式两边均为 $0$，显然成立。

设 $\mathbf{u} \neq \mathbf{0}$。对任意实数 $t$，考虑向量 $\mathbf{u} - t\mathbf{v}$，由正性：

$$0 \leq \|\mathbf{u} - t\mathbf{v}\|^2 = \langle \mathbf{u} - t\mathbf{v}, \mathbf{u} - t\mathbf{v} \rangle$$

$$= \|\mathbf{u}\|^2 - 2t\langle \mathbf{u}, \mathbf{v} \rangle + t^2\|\mathbf{v}\|^2$$

令 $f(t) = \|\mathbf{v}\|^2 t^2 - 2\langle \mathbf{u}, \mathbf{v} \rangle t + \|\mathbf{u}\|^2 \geq 0$。

这是关于 $t$ 的二次函数（首项系数 $\|\mathbf{v}\|^2 \geq 0$），恒非负，故其判别式非正：

$$\Delta = 4\langle \mathbf{u}, \mathbf{v} \rangle^2 - 4\|\mathbf{v}\|^2\|\mathbf{u}\|^2 \leq 0$$

$$\Rightarrow \langle \mathbf{u}, \mathbf{v} \rangle^2 \leq \|\mathbf{u}\|^2\,\|\mathbf{v}\|^2$$

取平方根得 $|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\|\,\|\mathbf{v}\|$。$\square$

**等号条件**：$\Delta = 0$ 时，$f(t)$ 恰有一个实根 $t_0 = \dfrac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{v}\|^2}$，使得 $\mathbf{u} - t_0 \mathbf{v} = \mathbf{0}$，即 $\mathbf{u} = t_0 \mathbf{v}$——两向量线性相关。

### Cauchy-Schwarz 不等式的具体形式

**在 $\mathbb{R}^n$ 中**（Schwarz 不等式）：

$$\left(\sum_{i=1}^n u_i v_i\right)^2 \leq \left(\sum_{i=1}^n u_i^2\right)\!\left(\sum_{i=1}^n v_i^2\right)$$

**在 $C([a,b])$ 中**（积分形式）：

$$\left(\int_a^b f(x)g(x)\,dx\right)^2 \leq \left(\int_a^b f(x)^2\,dx\right)\!\left(\int_a^b g(x)^2\,dx\right)$$

### 几何意义

Cauchy-Schwarz 不等式保证了夹角公式的合法性：

$$\cos\theta = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\|\,\|\mathbf{v}\|} \in [-1, 1]$$

由 Cauchy-Schwarz，$|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\|\,\|\mathbf{v}\|$ 保证了上式的绝对值不超过 1，从而 $\theta = \arccos(\cdots)$ 是有意义的。

**直觉**：两向量的内积衡量它们的"对齐程度"，最大对齐（同向）时 $\langle \mathbf{u}, \mathbf{v} \rangle = \|\mathbf{u}\|\,\|\mathbf{v}\|$，完全垂直时 $\langle \mathbf{u}, \mathbf{v} \rangle = 0$，完全反向时 $\langle \mathbf{u}, \mathbf{v} \rangle = -\|\mathbf{u}\|\,\|\mathbf{v}\|$。

### 三角不等式

**定理（三角不等式）**：设 $V$ 是实内积空间，对所有 $\mathbf{u}, \mathbf{v} \in V$，有：

$$\boxed{\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|}$$

**证明**：

$$\|\mathbf{u} + \mathbf{v}\|^2 = \langle \mathbf{u} + \mathbf{v}, \mathbf{u} + \mathbf{v} \rangle = \|\mathbf{u}\|^2 + 2\langle \mathbf{u}, \mathbf{v} \rangle + \|\mathbf{v}\|^2$$

$$\leq \|\mathbf{u}\|^2 + 2|\langle \mathbf{u}, \mathbf{v} \rangle| + \|\mathbf{v}\|^2 \leq \|\mathbf{u}\|^2 + 2\|\mathbf{u}\|\,\|\mathbf{v}\| + \|\mathbf{v}\|^2 = (\|\mathbf{u}\| + \|\mathbf{v}\|)^2$$

两边取平方根得 $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$。$\square$

**几何意义**：三角形任意一边的长度不超过另外两边之和——这正是初等几何中"两点之间直线最短"的精确表述。

---

## 本章小结

- **内积**是向量空间上满足正性、正定性、对称性、线性性四条公理的双线性函数，使空间具备了度量结构。

- **范数** $\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}$ 由内积诱导，衡量向量的"长度"；**距离** $d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|$ 衡量两向量之间的"远近"。

- **正交**（$\langle \mathbf{u}, \mathbf{v} \rangle = 0$）是垂直的推广；非零向量的正交集线性无关；标准正交基使坐标计算化为内积计算。

- **正交补** $W^\perp$ 是与子空间 $W$ 垂直的所有向量的集合，构成子空间；$V = W \oplus W^\perp$ 给出了空间的正交直和分解。

- **Cauchy-Schwarz 不等式** $|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\|\,\|\mathbf{v}\|$ 是内积理论的核心不等式，保证了夹角定义的合法性，并蕴含三角不等式。

**下一章预告**：Gram-Schmidt 正交化过程——如何从任意基出发，系统地构造标准正交基。

---

## 16.6 正交矩阵

### 定义

**定义** 实方阵 $Q$ 称为**正交矩阵**（Orthogonal Matrix），若：

$$Q^T Q = QQ^T = I \quad \Longleftrightarrow \quad Q^{-1} = Q^T$$

等价地，$Q$ 的列向量构成 $\mathbb{R}^n$ 的一组标准正交基。

### 正交矩阵的性质

| 性质 | 说明 |
|------|------|
| **保持内积** | $\langle Q\mathbf{u}, Q\mathbf{v} \rangle = \mathbf{u}^T Q^T Q \mathbf{v} = \langle \mathbf{u}, \mathbf{v} \rangle$ |
| **保持范数（等距）** | $\|Q\mathbf{v}\| = \|\mathbf{v}\|$ |
| **保持夹角** | 向量间的夹角在正交变换下不变 |
| **行列式** | $\det(Q) = \pm 1$（$+1$为旋转，$-1$为反射） |
| **特征值** | $|\lambda| = 1$（在复数域中） |
| **乘积封闭** | 正交矩阵的乘积仍是正交矩阵 |
| **逆也正交** | $Q^{-1} = Q^T$ 也是正交矩阵 |

### 典型例子

- **旋转矩阵**：$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$，$\det(R) = 1$
- **反射矩阵**：$H = I - 2\mathbf{u}\mathbf{u}^T$（$\|\mathbf{u}\| = 1$），$\det(H) = -1$
- **置换矩阵**：每行每列恰有一个 1 的 0-1 矩阵

### 正交矩阵的重要性

正交矩阵在数值计算中极为重要：它们的**条件数** $\kappa(Q) = 1$（最优），意味着正交变换**不放大误差**。QR分解（第17章）、SVD（第22章）的核心都涉及正交矩阵。

---

## 16.7 矩阵范数

### 从向量范数到矩阵范数

向量有范数衡量"大小"，矩阵同样需要。常用的矩阵范数有：

### Frobenius 范数

$$\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\text{tr}(A^T A)} = \sqrt{\sum_{i=1}^{r} \sigma_i^2}$$

**直觉**：将矩阵"拉平"成向量后取 $L^2$ 范数。

### 谱范数（算子 2-范数）

$$\|A\|_2 = \max_{\|\mathbf{x}\|=1} \|A\mathbf{x}\| = \sigma_{\max}(A)$$

即最大奇异值，衡量矩阵作为线性变换的最大"拉伸"倍数。

### 核范数

$$\|A\|_* = \sum_{i=1}^{r} \sigma_i$$

所有奇异值之和，在低秩矩阵恢复（矩阵补全）中作为秩的凸松弛。

### 条件数

矩阵 $A$ 的**条件数**定义为：

$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}$$

- $\kappa(A) \approx 1$：**良态**矩阵，数值计算稳定
- $\kappa(A) \gg 1$：**病态**矩阵，微小扰动导致巨大误差
- 正交矩阵 $\kappa(Q) = 1$，对角矩阵 $\kappa(D) = |d_{\max}/d_{\min}|$

条件数在深度学习中影响**优化收敛速度**：Hessian的条件数越大，梯度下降收敛越慢。

---

## 深度学习应用：余弦相似度与正交初始化

### 余弦相似度：归一化的内积

在机器学习与自然语言处理中，我们经常需要衡量两个向量"有多相似"。原始内积 $\langle \mathbf{u}, \mathbf{v} \rangle$ 受向量长度影响，长向量的内积值天然偏大，无法公平比较。

**余弦相似度（Cosine Similarity）**通过归一化消除了长度因素：

$$\text{CosSim}(\mathbf{u}, \mathbf{v}) = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\|\,\|\mathbf{v}\|} = \cos\theta$$

由 Cauchy-Schwarz 不等式，$\text{CosSim} \in [-1, 1]$，其中：
- $\text{CosSim} = 1$：同方向，完全相似
- $\text{CosSim} = 0$：正交，完全不相关
- $\text{CosSim} = -1$：反方向，完全相反

**本质**：余弦相似度就是两向量归一化后的内积——先将每个向量投影到单位球面上，再计算内积。

### 推荐系统中的应用

在协同过滤推荐系统中，用户兴趣和物品特征均表示为**嵌入向量（embedding vectors）**。两个用户或两件物品的相似度用余弦相似度衡量：

$$\text{相似度}(\text{用户}_A, \text{用户}_B) = \frac{\mathbf{e}_A \cdot \mathbf{e}_B}{\|\mathbf{e}_A\|\,\|\mathbf{e}_B\|}$$

向量方向相近 → 兴趣相似 → 互相推荐对方喜欢的内容。

### NLP 中的词向量相似度

在 Word2Vec、GloVe 等词嵌入模型中，每个词被映射为一个向量，语义相近的词在向量空间中方向相近：

$$\text{CosSim}(\text{"king"}, \text{"queen"}) \approx 0.85$$
$$\text{CosSim}(\text{"king"}, \text{"apple"}) \approx 0.05$$

著名的词向量类比：$\mathbf{e}_{\text{king}} - \mathbf{e}_{\text{man}} + \mathbf{e}_{\text{woman}} \approx \mathbf{e}_{\text{queen}}$，本质上是向量加法在语义空间中的正交分解。

### 正交权重初始化

神经网络的权重初始化对训练稳定性至关重要。**正交初始化（Orthogonal Initialization）**让权重矩阵 $W$ 满足：

$$W^T W = I \quad \text{（列正交）}$$

**为何有效**：

1. **保持梯度范数**：若 $W$ 是正交矩阵，则 $\|W\mathbf{x}\| = \|\mathbf{x}\|$——前向传播时信号不发散也不消失。

2. **梯度传播稳定**：反向传播时，梯度乘以 $W^T$；若 $W$ 正交，$W^T W = I$，梯度范数保持不变，避免梯度爆炸/消失。

3. **信息保留**：正交映射是**等距变换（isometry）**，不压缩也不拉伸空间，最大程度保留输入信息。

**数学原理**：若 $W$ 的奇异值全为 1（即正交矩阵），则条件数 $\kappa(W) = 1$——这是最"优良"的矩阵，数值上最稳定。

### 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# 1. 余弦相似度的计算与应用
# ============================================================
torch.manual_seed(42)

# 模拟词嵌入：每个词是一个 4 维向量
word_embeddings = {
    "king":   torch.tensor([0.8, 0.3, 0.2, 0.7]),
    "queen":  torch.tensor([0.7, 0.4, 0.3, 0.8]),
    "apple":  torch.tensor([0.1, 0.9, 0.8, 0.1]),
    "fruit":  torch.tensor([0.2, 0.8, 0.7, 0.2]),
}

def cosine_similarity(u: torch.Tensor, v: torch.Tensor) -> float:
    """余弦相似度 = 归一化内积"""
    dot_product = torch.dot(u, v)              # 内积 <u, v>
    norm_u = torch.norm(u)                     # ||u||
    norm_v = torch.norm(v)                     # ||v||
    return (dot_product / (norm_u * norm_v)).item()

# 计算词对之间的相似度
pairs = [("king", "queen"), ("king", "apple"), ("apple", "fruit")]
print("词向量余弦相似度:")
for w1, w2 in pairs:
    sim = cosine_similarity(word_embeddings[w1], word_embeddings[w2])
    print(f"  CosSim({w1:6s}, {w2:6s}) = {sim:.4f}")

# PyTorch 内置余弦相似度（批量计算）
u_batch = torch.stack(list(word_embeddings.values()))  # (4, 4)
cos_sim_matrix = F.cosine_similarity(
    u_batch.unsqueeze(0),   # (1, 4, 4)
    u_batch.unsqueeze(1),   # (4, 1, 4)
    dim=2
)
print("\n余弦相似度矩阵（行列对应 king/queen/apple/fruit）:")
print(cos_sim_matrix.numpy().round(3))

# ============================================================
# 2. 验证余弦相似度 = 归一化内积
# ============================================================
u = torch.randn(8)
v = torch.randn(8)

# 方法1：手动计算
u_norm = u / torch.norm(u)   # 归一化到单位球
v_norm = v / torch.norm(v)
sim_manual = torch.dot(u_norm, v_norm).item()

# 方法2：PyTorch F.cosine_similarity
sim_pytorch = F.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0)).item()

print(f"\n手动归一化内积 = {sim_manual:.6f}")
print(f"F.cosine_similarity = {sim_pytorch:.6f}")
print(f"两者差异: {abs(sim_manual - sim_pytorch):.2e}")  # ≈ 0

# ============================================================
# 3. 正交权重初始化
# ============================================================
# 标准随机初始化 vs 正交初始化：前向传播信号范数比较

def signal_norm_through_layers(init_type: str, n_layers: int = 20,
                                dim: int = 64, seed: int = 0) -> list:
    """追踪信号范数在多层线性网络中的变化"""
    torch.manual_seed(seed)
    x = torch.randn(dim)
    norms = [torch.norm(x).item()]

    for _ in range(n_layers):
        # 创建线性层并按指定方式初始化
        layer = nn.Linear(dim, dim, bias=False)

        if init_type == "random":
            # 默认 Kaiming 均匀初始化
            nn.init.uniform_(layer.weight, -0.1, 0.1)
        elif init_type == "orthogonal":
            # 正交初始化：权重矩阵是正交矩阵
            nn.init.orthogonal_(layer.weight)

        with torch.no_grad():
            x = layer(x)
        norms.append(torch.norm(x).item())

    return norms

norms_random = signal_norm_through_layers("random", n_layers=20)
norms_ortho  = signal_norm_through_layers("orthogonal", n_layers=20)

print("\n前向传播信号范数（20 层线性网络）:")
print(f"{'层数':>4}  {'随机初始化':>12}  {'正交初始化':>12}")
for i in [0, 4, 9, 14, 19]:
    print(f"  {i:>2}    {norms_random[i]:>12.4f}    {norms_ortho[i]:>12.4f}")

print(f"\n随机初始化 - 最终范数: {norms_random[-1]:.6f}")
print(f"正交初始化 - 最终范数: {norms_ortho[-1]:.6f}  ← 几乎不变！")

# ============================================================
# 4. 验证正交矩阵保持范数（等距变换）
# ============================================================
torch.manual_seed(7)
n = 5
# 生成一个正交矩阵（QR 分解的 Q）
A = torch.randn(n, n)
Q, _ = torch.linalg.qr(A)

x = torch.randn(n)
y = Q @ x

print(f"\n正交矩阵等距性验证:")
print(f"  ||x||    = {torch.norm(x).item():.6f}")
print(f"  ||Qx||   = {torch.norm(y).item():.6f}")
print(f"  差异     = {abs(torch.norm(x) - torch.norm(y)).item():.2e}")  # ≈ 0

# 验证 Q^T Q = I
QTQ = Q.T @ Q
identity_error = torch.norm(QTQ - torch.eye(n)).item()
print(f"  ||Q^T Q - I|| = {identity_error:.2e}")  # ≈ 0（正交性验证）

# ============================================================
# 5. 推荐系统：基于余弦相似度的 Top-K 推荐
# ============================================================
torch.manual_seed(42)
n_users, n_items, embed_dim = 5, 8, 6

# 随机初始化用户和物品嵌入
user_emb = torch.randn(n_users, embed_dim)
item_emb = torch.randn(n_items, embed_dim)

# 归一化（投影到单位球面）
user_norm = F.normalize(user_emb, p=2, dim=1)  # (n_users, embed_dim)
item_norm = F.normalize(item_emb, p=2, dim=1)  # (n_items, embed_dim)

# 计算所有用户-物品余弦相似度矩阵（矩阵乘法 = 批量内积）
sim_matrix = user_norm @ item_norm.T  # (n_users, n_items)

# 为用户 0 推荐 Top-3 物品
user_id = 0
topk = torch.topk(sim_matrix[user_id], k=3)
print(f"\n用户 {user_id} 的 Top-3 推荐物品:")
for rank, (score, item) in enumerate(zip(topk.values, topk.indices)):
    print(f"  排名 {rank+1}: 物品 {item.item()}, 余弦相似度 = {score.item():.4f}")
```

**代码解读**：

- **第 1-2 部分**：展示词向量余弦相似度的计算，验证"余弦相似度 = 归一化内积"，误差约为机器精度。
- **第 3 部分**：对比随机初始化与正交初始化在 20 层线性网络中信号范数的变化——随机初始化时范数迅速衰减或爆炸，正交初始化时范数几乎保持不变，体现了正交变换的等距性。
- **第 4 部分**：数值验证 $Q^T Q = I$ 和 $\|Q\mathbf{x}\| = \|\mathbf{x}\|$。
- **第 5 部分**：展示归一化嵌入向量通过矩阵乘法（批量内积）实现高效的 Top-K 推荐，这正是现代推荐系统的核心计算。

| 内积概念 | 深度学习对应物 | 实际应用 |
|:---|:---|:---|
| 内积 $\langle \mathbf{u}, \mathbf{v} \rangle$ | 向量点积/注意力分数 | Transformer 自注意力机制 |
| 余弦相似度 | 归一化内积 | 语义相似度、推荐系统 |
| 正交向量 | 不相关特征 | 特征解耦、独立分量分析 |
| 正交矩阵 | 正交初始化权重 | 稳定梯度传播 |
| 正交补 | 残差连接 | 跳跃连接保留原始信息 |

---

## 练习题

**练习 1**（基础——验证内积公理）

设 $\mathbb{R}^2$ 上定义函数 $\langle \mathbf{u}, \mathbf{v} \rangle = 2u_1 v_1 + 3u_2 v_2$。

（a）验证这是一个合法的内积（逐条验证四条公理）。

（b）计算向量 $\mathbf{u} = (1, 2)^T$ 和 $\mathbf{v} = (3, -1)^T$ 在此内积下的范数和夹角。

（c）与标准内积相比，此内积对哪个坐标方向"更重视"？为什么？

---

**练习 2**（基础——正交性与投影）

设 $\mathbf{u} = (2, 1, -1)^T$ 和 $\mathbf{v} = (1, 0, 2)^T$（使用标准内积）。

（a）计算 $\langle \mathbf{u}, \mathbf{v} \rangle$ 和两向量之间的夹角 $\theta$。

（b）计算 $\mathbf{u}$ 在 $\mathbf{v}$ 方向上的正交投影 $\text{proj}_{\mathbf{v}}(\mathbf{u})$。

（c）验证残差向量 $\mathbf{u} - \text{proj}_{\mathbf{v}}(\mathbf{u})$ 与 $\mathbf{v}$ 正交。

---

**练习 3**（中等——正交补与直和分解）

设 $W = \text{span}\!\left\{\begin{pmatrix}1\\1\\0\end{pmatrix}, \begin{pmatrix}0\\1\\1\end{pmatrix}\right\} \subseteq \mathbb{R}^3$。

（a）求 $W^\perp$ 的一组基。

（b）验证 $\dim(W) + \dim(W^\perp) = 3$。

（c）将向量 $\mathbf{b} = (1, 2, 3)^T$ 分解为 $\mathbf{b} = \mathbf{w} + \mathbf{w}^\perp$，其中 $\mathbf{w} \in W$，$\mathbf{w}^\perp \in W^\perp$。

---

**练习 4**（中等——Cauchy-Schwarz 不等式的应用）

利用 Cauchy-Schwarz 不等式证明以下各项：

（a）对任意实数 $a_1, \ldots, a_n$ 和 $b_1, \ldots, b_n$，有
$$\left(\sum_{i=1}^n a_i b_i\right)^2 \leq \left(\sum_{i=1}^n a_i^2\right)\!\!\left(\sum_{i=1}^n b_i^2\right)$$

（b）对 $n$ 个正实数 $x_1, \ldots, x_n$，证明算术-调和平均不等式的特例：
$$\frac{x_1 + x_2 + \cdots + x_n}{n} \cdot \frac{n}{\frac{1}{x_1} + \cdots + \frac{1}{x_n}} \leq 1$$

（提示：令 $a_i = \sqrt{x_i}$，$b_i = \dfrac{1}{\sqrt{x_i}}$，应用 Cauchy-Schwarz。）

---

**练习 5**（进阶——正交初始化与梯度传播）

考虑一个 $L$ 层线性网络（无偏置、无激活函数），每层的权重矩阵 $W_l \in \mathbb{R}^{n \times n}$，$l = 1, \ldots, L$。

（a）若每层权重矩阵是正交矩阵（$W_l^T W_l = I$），证明前向传播输出的范数等于输入范数：$\|W_L \cdots W_1 \mathbf{x}\| = \|\mathbf{x}\|$。

（b）设损失函数 $\mathcal{L}$ 关于第 $l$ 层输出的梯度为 $\nabla_l$，则关于第 $l-1$ 层输出的梯度为 $\nabla_{l-1} = W_l^T \nabla_l$。若 $W_l$ 是正交矩阵，证明 $\|\nabla_{l-1}\| = \|\nabla_l\|$（梯度范数在反向传播中保持不变）。

（c）若使用一般随机初始化，设每个权重 $w_{ij} \sim \mathcal{N}(0, \sigma^2)$，对于 $n \times n$ 矩阵，要使信号范数不发散，$\sigma$ 应满足什么条件？（提示：考虑 $\mathbb{E}[\|W\mathbf{x}\|^2]$，设 $\|\mathbf{x}\| = 1$。）

---

## 练习答案

<details>
<summary>点击展开 练习1 答案</summary>

**（a）验证内积公理**（$\langle \mathbf{u}, \mathbf{v} \rangle = 2u_1v_1 + 3u_2v_2$）：

1. **正性**：$\langle \mathbf{v}, \mathbf{v} \rangle = 2v_1^2 + 3v_2^2 \geq 0$（因为 $2, 3 > 0$）✓

2. **正定性**：$2v_1^2 + 3v_2^2 = 0 \iff v_1 = v_2 = 0 \iff \mathbf{v} = \mathbf{0}$ ✓

3. **对称性**：$\langle \mathbf{u}, \mathbf{v} \rangle = 2u_1v_1 + 3u_2v_2 = 2v_1u_1 + 3v_2u_2 = \langle \mathbf{v}, \mathbf{u} \rangle$ ✓

4. **线性性**：
$$\langle \mathbf{u} + \mathbf{w}, \mathbf{v} \rangle = 2(u_1+w_1)v_1 + 3(u_2+w_2)v_2 = (2u_1v_1 + 3u_2v_2) + (2w_1v_1 + 3w_2v_2) = \langle \mathbf{u}, \mathbf{v} \rangle + \langle \mathbf{w}, \mathbf{v} \rangle \checkmark$$
$$\langle c\mathbf{u}, \mathbf{v} \rangle = 2(cu_1)v_1 + 3(cu_2)v_2 = c(2u_1v_1 + 3u_2v_2) = c\langle \mathbf{u}, \mathbf{v} \rangle \checkmark$$

四条公理全部验证通过，故这是合法的内积。

**（b）计算范数与夹角**：

$$\langle \mathbf{u}, \mathbf{v} \rangle = 2(1)(3) + 3(2)(-1) = 6 - 6 = 0$$

$$\|\mathbf{u}\| = \sqrt{2 \cdot 1^2 + 3 \cdot 2^2} = \sqrt{2 + 12} = \sqrt{14}$$

$$\|\mathbf{v}\| = \sqrt{2 \cdot 3^2 + 3 \cdot (-1)^2} = \sqrt{18 + 3} = \sqrt{21}$$

$$\cos\theta = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\|\,\|\mathbf{v}\|} = \frac{0}{\sqrt{14}\cdot\sqrt{21}} = 0 \implies \theta = 90°$$

在此加权内积下，$\mathbf{u}$ 与 $\mathbf{v}$ 正交（注意在标准内积下 $\mathbf{u} \cdot \mathbf{v} = 3 - 2 + 1 = 1 \neq 0$，不正交）。

**（c）对哪个坐标更重视**：

权重系数 $w_2 = 3 > w_1 = 2$，第二个坐标方向的"权重"更大。在范数计算中，第二分量的贡献被放大了 $3/2$ 倍。这意味着此内积"认为"第二个方向比第一个方向更重要——类似于统计学中对方差较小的变量给予更高权重。

</details>

<details>
<summary>点击展开 练习2 答案</summary>

**（a）内积与夹角**：

$$\langle \mathbf{u}, \mathbf{v} \rangle = (2)(1) + (1)(0) + (-1)(2) = 2 + 0 - 2 = 0$$

$$\|\mathbf{u}\| = \sqrt{4 + 1 + 1} = \sqrt{6}, \quad \|\mathbf{v}\| = \sqrt{1 + 0 + 4} = \sqrt{5}$$

$$\cos\theta = \frac{0}{\sqrt{6}\cdot\sqrt{5}} = 0 \implies \theta = 90°$$

$\mathbf{u}$ 与 $\mathbf{v}$ 正交！

**（b）正交投影**：

$$\text{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{v}\|^2}\,\mathbf{v} = \frac{0}{5}\begin{pmatrix}1\\0\\2\end{pmatrix} = \begin{pmatrix}0\\0\\0\end{pmatrix}$$

由于 $\mathbf{u} \perp \mathbf{v}$，$\mathbf{u}$ 在 $\mathbf{v}$ 方向上没有分量，投影为零向量。

**（c）验证残差正交**：

$$\mathbf{u} - \text{proj}_{\mathbf{v}}(\mathbf{u}) = \begin{pmatrix}2\\1\\-1\end{pmatrix} - \begin{pmatrix}0\\0\\0\end{pmatrix} = \begin{pmatrix}2\\1\\-1\end{pmatrix} = \mathbf{u}$$

$$\left\langle \mathbf{u} - \text{proj}_{\mathbf{v}}(\mathbf{u}),\, \mathbf{v} \right\rangle = \langle \mathbf{u}, \mathbf{v} \rangle = 0 \checkmark$$

正交，符合预期（因为 $\mathbf{u}$ 本来就与 $\mathbf{v}$ 正交，投影为零）。

</details>

<details>
<summary>点击展开 练习3 答案</summary>

**（a）求 $W^\perp$**：

$\mathbf{x} \in W^\perp$ 当且仅当 $\mathbf{x}$ 与 $W$ 的两个生成向量都正交：

$$\begin{cases} x_1 + x_2 = 0 \\ x_2 + x_3 = 0 \end{cases}$$

由第一个方程 $x_2 = -x_1$，代入第二个方程 $-x_1 + x_3 = 0$，即 $x_3 = x_1$。

令 $x_1 = t$，得 $W^\perp = \text{span}\!\left\{\begin{pmatrix}1\\-1\\1\end{pmatrix}\right\}$，$\dim(W^\perp) = 1$。

**（b）维数验证**：

$\dim(W) = 2$（两个生成向量线性无关），$\dim(W^\perp) = 1$：

$$\dim(W) + \dim(W^\perp) = 2 + 1 = 3 = \dim(\mathbb{R}^3) \checkmark$$

**（c）分解 $\mathbf{b} = (1, 2, 3)^T$**：

设 $\mathbf{w} = \alpha \begin{pmatrix}1\\1\\0\end{pmatrix} + \beta \begin{pmatrix}0\\1\\1\end{pmatrix} \in W$，$\mathbf{w}^\perp = \gamma \begin{pmatrix}1\\-1\\1\end{pmatrix} \in W^\perp$，则：

$$\begin{pmatrix}1\\2\\3\end{pmatrix} = \alpha\begin{pmatrix}1\\1\\0\end{pmatrix} + \beta\begin{pmatrix}0\\1\\1\end{pmatrix} + \gamma\begin{pmatrix}1\\-1\\1\end{pmatrix}$$

解方程组：
$$\alpha + \gamma = 1, \quad \alpha + \beta - \gamma = 2, \quad \beta + \gamma = 3$$

由第三个方程 $\beta = 3 - \gamma$；代入第二个方程 $\alpha + (3-\gamma) - \gamma = 2$，即 $\alpha = 2\gamma - 1$；代入第一个方程 $(2\gamma-1) + \gamma = 1$，解得 $\gamma = \dfrac{2}{3}$，$\alpha = \dfrac{1}{3}$，$\beta = \dfrac{7}{3}$。

$$\mathbf{w} = \frac{1}{3}\begin{pmatrix}1\\1\\0\end{pmatrix} + \frac{7}{3}\begin{pmatrix}0\\1\\1\end{pmatrix} = \begin{pmatrix}1/3\\8/3\\7/3\end{pmatrix}, \quad \mathbf{w}^\perp = \frac{2}{3}\begin{pmatrix}1\\-1\\1\end{pmatrix} = \begin{pmatrix}2/3\\-2/3\\2/3\end{pmatrix}$$

验证：$\mathbf{w} + \mathbf{w}^\perp = \left(\dfrac{1}{3}+\dfrac{2}{3},\, \dfrac{8}{3}-\dfrac{2}{3},\, \dfrac{7}{3}+\dfrac{2}{3}\right)^T = (1, 2, 3)^T = \mathbf{b}$ ✓

</details>

<details>
<summary>点击展开 练习4 答案</summary>

**（a）直接应用 Cauchy-Schwarz**：

取 $\mathbf{u} = (a_1, \ldots, a_n)^T$，$\mathbf{v} = (b_1, \ldots, b_n)^T$，使用 $\mathbb{R}^n$ 标准内积：

$$\langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^n a_i b_i, \quad \|\mathbf{u}\| = \sqrt{\sum a_i^2}, \quad \|\mathbf{v}\| = \sqrt{\sum b_i^2}$$

由 Cauchy-Schwarz 不等式 $|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\|\,\|\mathbf{v}\|$，平方即得：

$$\left(\sum_{i=1}^n a_i b_i\right)^2 \leq \left(\sum_{i=1}^n a_i^2\right)\!\!\left(\sum_{i=1}^n b_i^2\right) \quad \square$$

**（b）算术-调和平均不等式**：

令 $a_i = \sqrt{x_i}$，$b_i = \dfrac{1}{\sqrt{x_i}}$（其中 $x_i > 0$），代入（a）的结论：

$$\left(\sum_{i=1}^n \sqrt{x_i} \cdot \frac{1}{\sqrt{x_i}}\right)^2 \leq \left(\sum_{i=1}^n x_i\right)\!\!\left(\sum_{i=1}^n \frac{1}{x_i}\right)$$

左边 $= \left(\sum_{i=1}^n 1\right)^2 = n^2$，故：

$$n^2 \leq \left(\sum_{i=1}^n x_i\right)\!\!\left(\sum_{i=1}^n \frac{1}{x_i}\right)$$

$$\Rightarrow \frac{\sum x_i}{n} \cdot \frac{n}{\sum \frac{1}{x_i}} \leq 1 \quad \square$$

等号成立当且仅当所有 $\dfrac{\sqrt{x_i}}{1/\sqrt{x_i}} = x_i$ 相等，即 $x_1 = x_2 = \cdots = x_n$。

</details>

<details>
<summary>点击展开 练习5 答案</summary>

**（a）正交矩阵保持范数**：

若每层 $W_l$ 是正交矩阵（$W_l^T W_l = I$），则：

$$\|W_l \mathbf{x}\|^2 = (W_l \mathbf{x})^T (W_l \mathbf{x}) = \mathbf{x}^T W_l^T W_l \mathbf{x} = \mathbf{x}^T I \mathbf{x} = \|\mathbf{x}\|^2$$

对所有层逐层应用，设 $\mathbf{x}^{(0)} = \mathbf{x}$，$\mathbf{x}^{(l)} = W_l \mathbf{x}^{(l-1)}$，则：

$$\|\mathbf{x}^{(l)}\| = \|W_l \mathbf{x}^{(l-1)}\| = \|\mathbf{x}^{(l-1)}\| = \cdots = \|\mathbf{x}^{(0)}\| = \|\mathbf{x}\|$$

故 $\|W_L \cdots W_1 \mathbf{x}\| = \|\mathbf{x}\|$。$\square$

**（b）梯度范数保持不变**：

$$\|\nabla_{l-1}\| = \|W_l^T \nabla_l\|$$

由（a）的推导，若 $W_l$ 是正交矩阵，则 $W_l^T$ 也是正交矩阵（正交矩阵的转置也是正交矩阵，因为 $(W_l^T)^T W_l^T = W_l W_l^T = I$ 对方正交矩阵成立）。

因此：$\|W_l^T \nabla_l\| = \|\nabla_l\|$，即 $\|\nabla_{l-1}\| = \|\nabla_l\|$。

梯度范数在每层反向传播中保持不变，不存在梯度爆炸或消失。$\square$

**（c）随机初始化的 $\sigma$ 条件**：

设 $\mathbf{x}$ 满足 $\|\mathbf{x}\| = 1$，$W$ 的每个元素独立同分布 $w_{ij} \sim \mathcal{N}(0, \sigma^2)$。

计算 $\mathbb{E}[\|W\mathbf{x}\|^2]$：

$$\mathbb{E}[\|W\mathbf{x}\|^2] = \mathbb{E}\left[\sum_{i=1}^n \left(\sum_{j=1}^n w_{ij} x_j\right)^2\right] = \sum_{i=1}^n \sum_{j=1}^n \sigma^2 x_j^2 = n\sigma^2 \|\mathbf{x}\|^2 = n\sigma^2$$

为使 $\mathbb{E}[\|W\mathbf{x}\|^2] = \|\mathbf{x}\|^2 = 1$，需要：

$$n\sigma^2 = 1 \implies \sigma = \frac{1}{\sqrt{n}}$$

这正是 **LeCun 初始化**的结论：$\sigma = \dfrac{1}{\sqrt{n}}$（$n$ 为输入维度）。在每层满足此条件时，期望意义下信号范数不发散。正交初始化从根本上保证了所有奇异值等于 1，比 LeCun 初始化更强。

</details>
