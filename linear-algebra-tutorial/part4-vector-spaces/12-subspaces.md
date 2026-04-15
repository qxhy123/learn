# 第12章：子空间

> **前置知识**：第9章（向量空间）、第10章（线性相关与线性无关）、第11章（基与维数）
>
> **本章难度**：★★★☆☆
>
> **预计学习时间**：3-4 小时

---

## 学习目标

学完本章后，你将能够：

- 理解子空间的严格定义，并能验证一个集合是否构成子空间
- 掌握矩阵的四个基本子空间（列空间、行空间、零空间、左零空间）的定义与几何意义
- 理解并运用秩-零化度定理，分析线性方程组解的结构
- 理解子空间的交与和，以及直和的概念
- 了解低秩近似与 LoRA（Low-Rank Adaptation）在大模型微调中的数学原理

---

## 12.1 子空间的定义

### 什么是子空间？

在研究向量空间时，我们常常关注其中具有特殊结构的"子集"。一个随机挑选的子集并不一定保持向量空间的性质，但如果该子集在加法和标量乘法下"封闭"，它就构成一个**子空间（subspace）**。

**定义**：设 $V$ 是域 $\mathbb{F}$ 上的向量空间，$W \subseteq V$ 是 $V$ 的一个非空子集。若 $W$ 在 $V$ 的加法和标量乘法下满足以下三个条件，则称 $W$ 是 $V$ 的一个**子空间**：

1. **包含零向量**：$\mathbf{0} \in W$
2. **对加法封闭**：若 $\mathbf{u}, \mathbf{v} \in W$，则 $\mathbf{u} + \mathbf{v} \in W$
3. **对标量乘法封闭**：若 $\mathbf{v} \in W$，$c \in \mathbb{F}$，则 $c\mathbf{v} \in W$

条件 2 和条件 3 合在一起，等价于**对线性组合封闭**：若 $\mathbf{u}, \mathbf{v} \in W$，$a, b \in \mathbb{F}$，则 $a\mathbf{u} + b\mathbf{v} \in W$。

**直觉**：子空间是向量空间"内部"的一个平整结构——它不会在某个方向上突然"边界截断"，而是向两端无限延伸。

### 两个平凡的子空间

对任意向量空间 $V$，以下两个子空间始终存在，称为**平凡子空间**：

- **零子空间** $\{\mathbf{0}\}$：只包含零向量，是最小的子空间
- **全空间** $V$ 自身：是最大的子空间

所有真正"有意义"的子空间都介于这两者之间。

### 验证子空间：三步法

**例 1**：验证 $\mathbb{R}^3$ 中的集合 $W = \{(x, y, 0)^T \mid x, y \in \mathbb{R}\}$（$xy$ 平面）是否为子空间。

**第一步（零向量）**：取 $x = y = 0$，得 $(0, 0, 0)^T \in W$。✓

**第二步（加法封闭）**：设 $\mathbf{u} = (x_1, y_1, 0)^T \in W$，$\mathbf{v} = (x_2, y_2, 0)^T \in W$，则：

$$\mathbf{u} + \mathbf{v} = (x_1 + x_2,\ y_1 + y_2,\ 0)^T \in W \quad \checkmark$$

**第三步（标量乘法封闭）**：设 $\mathbf{v} = (x, y, 0)^T \in W$，$c \in \mathbb{R}$，则：

$$c\mathbf{v} = (cx, cy, 0)^T \in W \quad \checkmark$$

三个条件均满足，故 $W$ 是 $\mathbb{R}^3$ 的子空间（几何上就是 $xy$ 平面）。

**例 2（反例）**：集合 $S = \{(x, y)^T \mid x \geq 0, y \geq 0\}$（第一象限，含坐标轴）是否为子空间？

取 $\mathbf{v} = (1, 1)^T \in S$，令 $c = -1$，则 $(-1)\mathbf{v} = (-1, -1)^T \notin S$（第三象限）。

**标量乘法不封闭**，故 $S$ 不是 $\mathbb{R}^2$ 的子空间。直觉上，子空间必须在两个方向上都无限延伸，第一象限的"边界"破坏了这一性质。

### 张成子空间

给定向量组 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k \in V$，它们的所有线性组合构成的集合：

$$\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \{c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k \mid c_1, \ldots, c_k \in \mathbb{F}\}$$

称为这些向量**张成（span）**的子空间。可以验证，张成集合一定是子空间，且是包含 $\mathbf{v}_1, \ldots, \mathbf{v}_k$ 的**最小子空间**。

---

## 12.2 矩阵的四个基本子空间

对于一个 $m \times n$ 矩阵 $A$，存在四个与之自然关联的子空间，它们从不同角度刻画了矩阵的结构，被称为**矩阵的四个基本子空间**。

设 $A$ 是 $m \times n$ 矩阵，将其分块表示为列向量形式：

$$A = \begin{pmatrix} \mathbf{a}_1 & \mathbf{a}_2 & \cdots & \mathbf{a}_n \end{pmatrix}$$

其中每个 $\mathbf{a}_j \in \mathbb{R}^m$。

### 列空间 Col(A)

**定义**：矩阵 $A$ 的所有列向量的线性组合构成的集合，称为 $A$ 的**列空间**，记作 $\text{Col}(A)$（或 $\mathcal{C}(A)$、$\text{Im}(A)$、$\text{Range}(A)$）：

$$\text{Col}(A) = \text{span}(\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_n) = \{A\mathbf{x} \mid \mathbf{x} \in \mathbb{R}^n\}$$

**所在空间**：$\text{Col}(A) \subseteq \mathbb{R}^m$（$m$ 维空间中的子空间）

**几何意义**：$A\mathbf{x} = \mathbf{b}$ 有解，当且仅当 $\mathbf{b} \in \text{Col}(A)$。列空间描述了矩阵 $A$ 作为线性映射时的**像（image）**——所有可能的输出。

**例**：设

$$A = \begin{pmatrix} 1 & 2 \\ 3 & 6 \\ 2 & 4 \end{pmatrix}$$

注意到第二列是第一列的 2 倍，因此 $\text{Col}(A) = \text{span}\!\left(\begin{pmatrix}1\\3\\2\end{pmatrix}\right)$，是 $\mathbb{R}^3$ 中过原点的一条直线（一维子空间）。

### 行空间 Row(A)

**定义**：矩阵 $A$ 的所有行向量的线性组合构成的集合，称为 $A$ 的**行空间**，记作 $\text{Row}(A)$（或 $\mathcal{C}(A^T)$）：

$$\text{Row}(A) = \text{Col}(A^T) = \text{span}(\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_m)$$

其中 $\mathbf{r}_i$ 是 $A$ 的第 $i$ 行（视为列向量）。

**所在空间**：$\text{Row}(A) \subseteq \mathbb{R}^n$（$n$ 维空间中的子空间）

**重要性质**：行变换（高斯消元）不改变行空间。行最简形的非零行就是行空间的一组基。

### 零空间 Null(A)

**定义**：使得 $A\mathbf{x} = \mathbf{0}$ 成立的所有向量 $\mathbf{x}$ 组成的集合，称为 $A$ 的**零空间**（也称**核**），记作 $\text{Null}(A)$（或 $\ker(A)$）：

$$\text{Null}(A) = \{\mathbf{x} \in \mathbb{R}^n \mid A\mathbf{x} = \mathbf{0}\}$$

**所在空间**：$\text{Null}(A) \subseteq \mathbb{R}^n$（$n$ 维输入空间中的子空间）

**验证它是子空间**：
- 零向量：$A\mathbf{0} = \mathbf{0}$，所以 $\mathbf{0} \in \text{Null}(A)$ ✓
- 加法封闭：若 $A\mathbf{u} = \mathbf{0}$，$A\mathbf{v} = \mathbf{0}$，则 $A(\mathbf{u}+\mathbf{v}) = A\mathbf{u} + A\mathbf{v} = \mathbf{0}$ ✓
- 标量封闭：若 $A\mathbf{v} = \mathbf{0}$，则 $A(c\mathbf{v}) = cA\mathbf{v} = \mathbf{0}$ ✓

**几何意义**：零空间描述了映射 $A$ 的"核"——所有被"压缩"成零向量的输入方向。$\text{Null}(A) = \{\mathbf{0}\}$ 当且仅当 $A\mathbf{x} = \mathbf{0}$ 仅有零解，即 $A$ 的列线性无关。

**关键关系**：线性方程组 $A\mathbf{x} = \mathbf{b}$ 的通解 = 一个特解 + 零空间中的任意向量：

$$\mathbf{x} = \mathbf{x}_p + \mathbf{x}_h, \quad \mathbf{x}_h \in \text{Null}(A)$$

### 左零空间 Null(A^T)

**定义**：$A^T$ 的零空间，称为 $A$ 的**左零空间**，记作 $\text{Null}(A^T)$：

$$\text{Null}(A^T) = \{\mathbf{y} \in \mathbb{R}^m \mid A^T\mathbf{y} = \mathbf{0}\} = \{\mathbf{y} \in \mathbb{R}^m \mid \mathbf{y}^T A = \mathbf{0}^T\}$$

**所在空间**：$\text{Null}(A^T) \subseteq \mathbb{R}^m$（$m$ 维输出空间中的子空间）

之所以叫"左零空间"，是因为条件写成 $\mathbf{y}^T A = \mathbf{0}^T$，向量 $\mathbf{y}^T$ 从左边乘以 $A$。

### 四个子空间的汇总

| 子空间 | 记号 | 所在空间 | 维数 | 几何含义 |
|:---|:---|:---:|:---:|:---|
| 列空间 | $\text{Col}(A)$ | $\mathbb{R}^m$ | $r$ | $A\mathbf{x}=\mathbf{b}$ 有解的充要条件 |
| 行空间 | $\text{Row}(A)$ | $\mathbb{R}^n$ | $r$ | 与零空间互补的正交补 |
| 零空间 | $\text{Null}(A)$ | $\mathbb{R}^n$ | $n-r$ | 被映射为零的输入方向 |
| 左零空间 | $\text{Null}(A^T)$ | $\mathbb{R}^m$ | $m-r$ | 与列空间互补的正交补 |

其中 $r = \text{rank}(A)$ 为矩阵的秩（见下节）。四个子空间以两对正交关系相互配对：

$$\text{Row}(A) \perp \text{Null}(A) \quad \text{（均在 } \mathbb{R}^n \text{ 中）}$$

$$\text{Col}(A) \perp \text{Null}(A^T) \quad \text{（均在 } \mathbb{R}^m \text{ 中）}$$

这两对正交关系构成了线性代数最深刻的结论之一，揭示了矩阵结构的完整图景。

---

## 12.3 秩与秩-零化度定理

### 秩的定义

矩阵 $A$ 的**秩（rank）**定义为其列空间（等价地，行空间）的维数：

$$\text{rank}(A) = \dim\bigl(\text{Col}(A)\bigr) = \dim\bigl(\text{Row}(A)\bigr)$$

**重要性质**：

1. **行秩 = 列秩**：对任意矩阵，行空间的维数等于列空间的维数。（这是一个非平凡的结论。）
2. **最大秩**：$\text{rank}(A) \leq \min(m, n)$
3. **行变换保秩**：初等行变换不改变矩阵的秩（因为行空间不变）
4. **转置保秩**：$\text{rank}(A^T) = \text{rank}(A)$

**满秩矩阵**：若 $\text{rank}(A) = \min(m, n)$，称 $A$ 为**满秩矩阵**。
- 对 $n \times n$ 方阵，满秩等价于可逆，此时 $\text{rank}(A) = n$。
- 对 $m \times n$ 矩阵（$m > n$），列满秩（$\text{rank} = n$）意味着 $A\mathbf{x}=\mathbf{0}$ 仅有零解。

**计算秩**：对 $A$ 做行化简，秩等于行阶梯形中非零行（主元行）的数目。

### 零化度（nullity）

矩阵 $A$ 的**零化度（nullity）**定义为零空间的维数：

$$\text{nullity}(A) = \dim\bigl(\text{Null}(A)\bigr)$$

### 秩-零化度定理

**定理（秩-零化度定理，Rank-Nullity Theorem）**：对任意 $m \times n$ 矩阵 $A$，有：

$$\boxed{\text{rank}(A) + \text{nullity}(A) = n}$$

即：**列空间维数 + 零空间维数 = 列数**。

**证明思路**：设 $r = \text{rank}(A)$，对 $A$ 做行化简得到行阶梯形 $R$，其中有 $r$ 个主元列和 $n - r$ 个自由列。每个自由变量对应零空间的一个基向量，故 $\text{nullity}(A) = n - r$，即 $r + (n - r) = n$。$\square$

### 几何解释

秩-零化度定理有深刻的几何含义：$n$ 维输入空间被矩阵 $A$ 的线性映射"分割"为两个互补的部分——

- **行空间**（$r$ 维）：这个方向上的输入被 $A$ "有效地映射"到输出空间中，携带信息。
- **零空间**（$n-r$ 维）：这个方向上的输入被 $A$ "压缩"为零，信息丢失。

两部分维数之和恰好等于输入空间的维数 $n$。

| 矩阵类型 | $r$ | $\text{nullity}$ | 解的情况 |
|:---|:---:|:---:|:---|
| 列满秩（$r = n$） | $n$ | $0$ | $A\mathbf{x}=\mathbf{0}$ 仅有零解 |
| 行满秩（$r = m$） | $m$ | $n-m$ | $A\mathbf{x}=\mathbf{b}$ 对所有 $\mathbf{b}$ 有解 |
| 满秩方阵（$r = m = n$） | $n$ | $0$ | $A\mathbf{x}=\mathbf{b}$ 有唯一解 |
| 秩亏缺（$r < \min(m,n)$） | $r$ | $>0$ | 无穷多解或无解 |

### 计算示例

设

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 1 & 2 \end{pmatrix}$$

行化简：

$$\xrightarrow{R_2 \leftarrow R_2 - 2R_1} \begin{pmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 1 & 1 & 2 \end{pmatrix} \xrightarrow{R_3 \leftarrow R_3 - R_1} \begin{pmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 0 & -1 & -1 \end{pmatrix} \xrightarrow{R_2 \leftrightarrow R_3} \begin{pmatrix} 1 & 2 & 3 \\ 0 & -1 & -1 \\ 0 & 0 & 0 \end{pmatrix}$$

主元有 2 个（第1、2列），故 $\text{rank}(A) = 2$，$\text{nullity}(A) = 3 - 2 = 1$。

零空间：自由变量为 $x_3$，令 $x_3 = t$，回代得 $x_2 = -t$，$x_1 = -t$，即：

$$\text{Null}(A) = \text{span}\!\left(\begin{pmatrix} -1 \\ -1 \\ 1 \end{pmatrix}\right)$$

这与定理一致：零空间是一维的（$\mathbb{R}^3$ 中的一条直线）。

### 秩的重要不等式

以下是秩的常用不等式，在证明和计算中非常有用。设 $A$ 为 $m \times n$ 矩阵，$B$ 为 $n \times p$ 矩阵。

**1. 乘积的秩不等式**：

$$\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$$

**直觉**：线性映射的复合不会增加像的维数。

**2. Sylvester 秩不等式**（下界）：

$$\boxed{\text{rank}(A) + \text{rank}(B) - n \leq \text{rank}(AB)}$$

**证明思路**：由零化度不等式 $\text{nullity}(AB) \leq \text{nullity}(A) + \text{nullity}(B)$ 及秩-零化度定理推出。

**3. 秩一更新**：若 $\mathbf{u}, \mathbf{v}$ 为非零向量，则 $|\text{rank}(A + \mathbf{u}\mathbf{v}^T) - \text{rank}(A)| \leq 1$。

**4. 其他常用不等式**：

| 不等式 | 条件 |
|--------|------|
| $\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$ | 秩的次可加性 |
| $\text{rank}(A^T A) = \text{rank}(A)$ | 恒成立 |
| $\text{rank}(A) = \text{rank}(A^T)$ | 行秩 = 列秩 |
| $\text{rank}(PAQ) = \text{rank}(A)$ | $P, Q$ 可逆 |

**例题** 设 $A$ 为 $3 \times 5$ 矩阵，$B$ 为 $5 \times 4$ 矩阵，$\text{rank}(A) = 3$，$\text{rank}(B) = 4$。求 $\text{rank}(AB)$ 的范围。

**解**：上界 $\text{rank}(AB) \leq \min(3, 4) = 3$。Sylvester 下界 $\text{rank}(AB) \geq 3 + 4 - 5 = 2$。故 $2 \leq \text{rank}(AB) \leq 3$。

---

## 12.4 子空间的交与和

### 子空间的交

设 $U$ 和 $W$ 是向量空间 $V$ 的两个子空间，它们的**交（intersection）**定义为：

$$U \cap W = \{\mathbf{v} \in V \mid \mathbf{v} \in U \text{ 且 } \mathbf{v} \in W\}$$

**命题**：两个子空间的交仍然是子空间。

**证明**：
- $\mathbf{0} \in U$ 且 $\mathbf{0} \in W$，所以 $\mathbf{0} \in U \cap W$ ✓
- 若 $\mathbf{u}, \mathbf{v} \in U \cap W$，则 $\mathbf{u} + \mathbf{v}$ 既在 $U$ 中又在 $W$ 中，故 $\mathbf{u} + \mathbf{v} \in U \cap W$ ✓
- 若 $\mathbf{v} \in U \cap W$，则 $c\mathbf{v}$ 既在 $U$ 中又在 $W$ 中，故 $c\mathbf{v} \in U \cap W$ ✓ $\square$

**几何直觉**：三维空间中两个过原点的平面（二维子空间）的交，是一条过原点的直线（一维子空间）——除非两平面重合（交为二维）或只有零向量（但两平面过原点时至少有零向量，若不平行则恰为直线）。

**注意**：两个子空间的**并** $U \cup W$ 一般不是子空间（取 $U$ 和 $W$ 各一个向量相加，结果未必在二者的并中）。

### 子空间的和

设 $U$ 和 $W$ 是向量空间 $V$ 的两个子空间，它们的**和（sum）**定义为：

$$U + W = \{\mathbf{u} + \mathbf{w} \mid \mathbf{u} \in U,\ \mathbf{w} \in W\}$$

**命题**：$U + W$ 是子空间，且是同时包含 $U$ 和 $W$ 的**最小子空间**。

**维数公式**：两个有限维子空间满足类似容斥原理的维数公式：

$$\boxed{\dim(U + W) = \dim(U) + \dim(W) - \dim(U \cap W)}$$

这是向量空间版本的"容斥原理"，揭示了子空间的重叠程度对和空间维数的影响。

### 直和

当 $U + W$ 中每个向量的分解方式唯一时，称之为**直和（direct sum）**，记作 $U \oplus W$。

**定义**：若 $U \cap W = \{\mathbf{0}\}$，则 $U + W$ 是直和，即对 $V$ 中任意向量 $\mathbf{v} \in U + W$，存在**唯一**的 $\mathbf{u} \in U$ 和 $\mathbf{w} \in W$，使得 $\mathbf{v} = \mathbf{u} + \mathbf{w}$。

**等价条件**：$U + W = U \oplus W \iff U \cap W = \{\mathbf{0}\}$

**直和的维数**：

$$\dim(U \oplus W) = \dim(U) + \dim(W)$$

**重要例子——正交补分解**：行空间与零空间构成直和：

$$\mathbb{R}^n = \text{Row}(A) \oplus \text{Null}(A)$$

类似地，$\mathbb{R}^m = \text{Col}(A) \oplus \text{Null}(A^T)$。

这是秩-零化度定理的几何核心：$\mathbb{R}^n$ 被分解为两个互补的正交子空间，维数分别为 $r$ 和 $n-r$，合计为 $n$。

**直和可以推广**：若 $V = W_1 \oplus W_2 \oplus \cdots \oplus W_k$（两两交只含零向量），则：

$$\dim(V) = \dim(W_1) + \dim(W_2) + \cdots + \dim(W_k)$$

---

## 本章小结

| 概念 | 定义 | 关键性质 |
|:---|:---|:---|
| 子空间 | 向量空间的非空子集，对加法和标量乘法封闭 | 必含零向量；张成集是最小子空间 |
| 列空间 $\text{Col}(A)$ | $A$ 的列向量的线性组合 | $\subseteq \mathbb{R}^m$，维数为 $r$ |
| 行空间 $\text{Row}(A)$ | $A$ 的行向量的线性组合 | $\subseteq \mathbb{R}^n$，维数为 $r$ |
| 零空间 $\text{Null}(A)$ | $A\mathbf{x}=\mathbf{0}$ 的解集 | $\subseteq \mathbb{R}^n$，维数为 $n-r$ |
| 左零空间 $\text{Null}(A^T)$ | $A^T\mathbf{y}=\mathbf{0}$ 的解集 | $\subseteq \mathbb{R}^m$，维数为 $m-r$ |
| 秩 $\text{rank}(A)$ | 列空间（=行空间）的维数 | 行秩=列秩；满秩方阵可逆 |
| 秩-零化度定理 | $\text{rank}(A) + \text{nullity}(A) = n$ | 揭示输入空间的完整分解 |
| 子空间的交 | $U \cap W$ | 仍是子空间 |
| 子空间的和 | $U + W$ | $\dim(U+W) = \dim U + \dim W - \dim(U \cap W)$ |
| 直和 $U \oplus W$ | $U \cap W = \{\mathbf{0}\}$ 时的和 | $\mathbb{R}^n = \text{Row}(A) \oplus \text{Null}(A)$ |

**核心思路**：矩阵的四个基本子空间提供了理解线性映射的完整框架——输入空间被分解为"有效"（行空间）和"无效"（零空间）两部分，输出空间被分解为"可达"（列空间）和"不可达"（左零空间）两部分，每对子空间正交互补。秩-零化度定理是这一分解的定量描述，也是分析线性方程组解的结构性工具。

**下一章预告**：线性映射——从代数运算到几何变换的统一语言。

---

## 深度学习应用

### 低秩近似与信息压缩

现代深度学习中的权重矩阵往往维度极高，但实践中发现许多权重矩阵是**近似低秩**的——即其有效信息可以用远低于矩阵维数的若干方向来刻画。

设权重矩阵 $W \in \mathbb{R}^{m \times n}$，若 $\text{rank}(W) = r \ll \min(m, n)$，则 $W$ 的列空间只有 $r$ 维，大量参数是冗余的。

**奇异值分解（SVD）**给出了最优的低秩近似（详见第22章）：

$$W \approx W_r = U_r \Sigma_r V_r^T$$

其中 $U_r \in \mathbb{R}^{m \times r}$，$\Sigma_r \in \mathbb{R}^{r \times r}$，$V_r \in \mathbb{R}^{n \times r}$。存储参数从 $mn$ 降至 $r(m + n)$，当 $r$ 很小时压缩比极大。

**Eckart-Young 定理**保证，在 Frobenius 范数意义下，$W_r$ 是所有秩不超过 $r$ 的矩阵中距离 $W$ 最近的矩阵：

$$W_r = \arg\min_{\text{rank}(B) \leq r} \|W - B\|_F$$

### LoRA：Low-Rank Adaptation

**背景**：将 GPT-3（1750 亿参数）等预训练大模型微调到下游任务，若全量更新所有参数，计算和存储成本极高。

**核心假设**（Hu et al., 2021）：微调时权重的**更新量** $\Delta W$ 是低秩的。即使原始权重 $W$ 是满秩的，任务迁移所需的"增量信息"可以被一个低维子空间刻画。

**LoRA 方法**：冻结预训练权重 $W_0 \in \mathbb{R}^{d \times d}$，将更新量参数化为两个小矩阵的乘积：

$$W = W_0 + \Delta W = W_0 + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times d}$，秩 $r \ll d$（典型取值 $r = 4, 8, 16$）。

**子空间视角**：$\Delta W = BA$ 的列空间维数最多为 $r$（秩-零化度定理的直接推论），训练时模型只在 $r$ 维子空间内搜索最优更新方向。

**参数效率**：原参数量 $d^2$，LoRA 参数量 $2dr$，压缩比约为 $d / (2r)$。取 $d = 4096$，$r = 8$，压缩比达 256 倍。

**初始化**：$A$ 用随机高斯初始化，$B$ 初始化为零矩阵，保证训练开始时 $\Delta W = BA = 0$，不破坏预训练模型的初始状态。

### 矩阵分解在大模型微调中的应用

LoRA 之后涌现出一系列基于低秩思想的方法：

| 方法 | 核心思想 | 关键改进 |
|:---|:---|:---|
| LoRA | 用 $BA$ 参数化 $\Delta W$ | 奠基工作 |
| AdaLoRA | 自适应分配秩 $r$ 给不同层 | 非均匀秩分配 |
| QLoRA | 4-bit 量化 + LoRA | 在消费级 GPU 上微调 65B 模型 |
| DoRA | 将权重分解为方向+幅度，仅对方向用 LoRA | 接近全量微调性能 |

这些方法的数学基础，都是本章讲述的子空间理论：大型矩阵的变换可以被低维子空间高效刻画，而四个基本子空间和秩-零化度定理提供了分析这一压缩的理论工具。

### 代码示例（LoRA 示例）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. 验证四个基本子空间的维数（秩-零化度定理）
# ============================================================
A = torch.tensor([
    [1., 2., 3.],
    [2., 4., 6.],
    [1., 1., 2.]
], dtype=torch.float64)

# 用 SVD 计算秩（奇异值非零的个数）
U, S, Vh = torch.linalg.svd(A)
tolerance = 1e-9
rank = (S > tolerance).sum().item()
n_cols = A.shape[1]
nullity = n_cols - rank

print(f"矩阵 A 的形状: {A.shape}")        # torch.Size([3, 3])
print(f"rank(A)    = {rank}")              # 2
print(f"nullity(A) = {nullity}")           # 1
print(f"秩 + 零化度 = {rank + nullity} = n = {n_cols}")  # 3

# ============================================================
# 2. 计算零空间基向量（零空间 = Null(A)）
# ============================================================
# Vh 的最后 nullity 行即为零空间的标准正交基
null_basis = Vh[-nullity:, :]   # shape: (nullity, n)
print(f"\n零空间基向量:\n{null_basis}")
print(f"验证 A @ null_basis.T ≈ 0:\n{A @ null_basis.T}")

# ============================================================
# 3. LoRA 层实现
# ============================================================
class LoRALinear(nn.Module):
    """
    将标准线性层 y = Wx + b 替换为带 LoRA 旁路的版本：
        y = (W_0 + BA) x + b
    其中 W_0 冻结，B 和 A 是可训练的低秩矩阵。
    """
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank  = rank
        self.alpha = alpha   # 缩放因子，控制 LoRA 更新的幅度

        # 冻结的预训练权重（模拟预训练后的状态）
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02,
            requires_grad=False   # 冻结：不参与梯度更新
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # 可训练的低秩矩阵 A（r×d）和 B（d×r）
        # 初始化：A ~ N(0,1)，B = 0 → 训练开始时 ΔW = BA = 0
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 缩放比例 = alpha / r，防止 r 变化时输出幅度剧烈变化
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始前向计算：W_0 x + b
        base_output = F.linear(x, self.weight, self.bias)
        # LoRA 增量：(B A) x × scaling
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_output + lora_output

    def lora_param_count(self) -> int:
        return self.rank * (self.in_features + self.out_features)

    def base_param_count(self) -> int:
        return self.in_features * self.out_features

# ============================================================
# 4. 演示 LoRA 的参数效率
# ============================================================
d_model = 4096   # Transformer 隐层维度（如 LLaMA-7B）
r       = 8      # LoRA 秩

lora_layer = LoRALinear(d_model, d_model, rank=r, alpha=16.0)

base_params = lora_layer.base_param_count()
lora_params = lora_layer.lora_param_count()
compression = base_params / lora_params

print(f"\n--- LoRA 参数效率分析 ---")
print(f"原始参数量: {base_params:,}   ({d_model}×{d_model})")
print(f"LoRA 参数量: {lora_params:,}   ({r}×{d_model} + {d_model}×{r})")
print(f"压缩比: {compression:.1f}×")
# 输出: 压缩比: 256.0×

# ============================================================
# 5. 验证：Delta_W = B @ A 是低秩矩阵
# ============================================================
with torch.no_grad():
    delta_W = lora_layer.lora_B @ lora_layer.lora_A  # shape: (d, d)
    _, S_delta, _ = torch.linalg.svd(delta_W.float())
    effective_rank = (S_delta > 1e-5).sum().item()
    print(f"\nΔW = B @ A 的有效秩: {effective_rank}")
    print(f"设计秩 r = {r}，理论上 rank(ΔW) ≤ r：{'✓' if effective_rank <= r else '✗'}")

# ============================================================
# 6. 前向传播示例
# ============================================================
batch_size, seq_len = 2, 10
x = torch.randn(batch_size, seq_len, d_model)

with torch.no_grad():
    y = lora_layer(x)
    print(f"\n输入形状:  {x.shape}")    # torch.Size([2, 10, 4096])
    print(f"输出形状:  {y.shape}")    # torch.Size([2, 10, 4096])
```

**运行说明**：执行 `pip install torch` 后即可运行。代码演示了用 SVD 验证秩-零化度定理、计算零空间基向量，以及 LoRA 层的完整实现——包含冻结权重、低秩旁路、参数效率分析，以及对 $\Delta W = BA$ 低秩性质的数值验证。

### 延伸阅读

- **LoRA 原始论文**：Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685)（2021）——大模型高效微调的奠基工作
- **QLoRA 论文**：Dettmers et al., ["QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314)（2023）——4-bit 量化结合 LoRA，消费级 GPU 可微调 65B 模型
- **Gilbert Strang 线性代数**：*Introduction to Linear Algebra*（第5版）——四个基本子空间的经典讲解，图示清晰，强烈推荐
- **3Blue1Brown**：[线性变换与矩阵](https://www.youtube.com/watch?v=kYB8IZa5AuE)——核与像的几何动画演示

---

## 练习题

**练习 1**（基础——验证子空间）

判断下列集合是否为 $\mathbb{R}^3$ 的子空间，并给出完整理由：

（a）$W_1 = \{(x, y, z)^T \mid 2x - y + z = 0\}$（过原点的平面）

（b）$W_2 = \{(x, y, z)^T \mid x + y + z = 1\}$（不过原点的平面）

（c）$W_3 = \{(x, y, z)^T \mid x = y = z\}$（主对角线方向直线）

---

**练习 2**（基础——四个基本子空间）

设

$$A = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \\ 2 & 1 & 3 \end{pmatrix}$$

（a）求 $\text{rank}(A)$

（b）分别求 $\text{Col}(A)$、$\text{Row}(A)$、$\text{Null}(A)$、$\text{Null}(A^T)$ 的一组基及其维数

（c）验证秩-零化度定理

---

**练习 3**（中等——秩-零化度定理的应用）

设 $A$ 是 $5 \times 7$ 矩阵，已知 $\text{rank}(A) = 3$。

（a）$\text{Null}(A)$ 的维数是多少？

（b）$\text{Col}(A)$ 是 $\mathbb{R}^m$ 中 $m$ 等于多少的子空间，维数是多少？

（c）$\text{Null}(A^T)$ 的维数是多少？

（d）线性方程组 $A\mathbf{x} = \mathbf{b}$ 对所有 $\mathbf{b} \in \mathbb{R}^5$ 都有解吗？为什么？

---

**练习 4**（中等——直和与维数公式）

设 $U = \text{span}\{(1, 0, 1)^T, (0, 1, 0)^T\}$，$W = \text{span}\{(1, 1, 1)^T, (0, 1, -1)^T\}$ 是 $\mathbb{R}^3$ 的两个子空间。

（a）分别求 $\dim(U)$ 和 $\dim(W)$

（b）求 $U \cap W$（给出具体描述或基向量）

（c）利用维数公式 $\dim(U+W) = \dim U + \dim W - \dim(U \cap W)$，计算 $\dim(U+W)$

（d）$U + W$ 是否等于整个 $\mathbb{R}^3$？给出理由

---

**练习 5**（进阶——LoRA 的线性代数原理）

设预训练权重 $W_0 \in \mathbb{R}^{512 \times 512}$，LoRA 更新量 $\Delta W = BA$，其中 $B \in \mathbb{R}^{512 \times 4}$，$A \in \mathbb{R}^{4 \times 512}$。

（a）利用秩-零化度定理，分析矩阵 $A$（作为 $\mathbb{R}^{512} \to \mathbb{R}^4$ 的线性映射）的零空间维数，说明 LoRA 微调时"忽略"了输入空间的哪个维度大小的方向。

（b）$\Delta W$ 的秩最大是多少？

（c）参数量分析：原始全量微调需要更新 $W_0$ 的多少个参数？LoRA 微调需要更新多少个参数？压缩比是多少？

（d）若将 $W = W_0 + BA$ 视为整体，分析 $\text{Col}(W)$ 和 $\text{Col}(W_0)$ 的关系：$W$ 的列空间相比 $W_0$ 的列空间可能发生什么变化？

---

## 练习答案

<details>
<summary>点击展开 练习1 答案</summary>

**（a）$W_1 = \{(x, y, z)^T \mid 2x - y + z = 0\}$：是子空间**

- 零向量：$2(0) - 0 + 0 = 0$ ✓
- 加法封闭：若 $2x_1 - y_1 + z_1 = 0$ 且 $2x_2 - y_2 + z_2 = 0$，则 $2(x_1+x_2) - (y_1+y_2) + (z_1+z_2) = 0$ ✓
- 标量封闭：若 $2x - y + z = 0$，则 $2(cx) - (cy) + (cz) = c(2x - y + z) = 0$ ✓

$W_1$ 是 $\mathbb{R}^3$ 中过原点的一个平面（二维子空间）。

**（b）$W_2 = \{(x, y, z)^T \mid x + y + z = 1\}$：不是子空间**

零向量：$0 + 0 + 0 = 0 \neq 1$，故 $\mathbf{0} \notin W_2$，第一条件不满足。

$W_2$ 是不过原点的平面，不含零向量，不是子空间。

**（c）$W_3 = \{(x, y, z)^T \mid x = y = z\}$：是子空间**

- 零向量：$0 = 0 = 0$ ✓
- 加法封闭：若 $x_1 = y_1 = z_1$ 且 $x_2 = y_2 = z_2$，则 $x_1+x_2 = y_1+y_2 = z_1+z_2$ ✓
- 标量封闭：若 $x = y = z$，则 $cx = cy = cz$ ✓

$W_3 = \text{span}((1,1,1)^T)$，是过原点的一条直线（一维子空间）。

</details>

<details>
<summary>点击展开 练习2 答案</summary>

对 $A$ 做行化简：

$$A = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \\ 2 & 1 & 3 \end{pmatrix} \xrightarrow{R_3 \leftarrow R_3 - 2R_1} \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \\ 0 & 1 & -1 \end{pmatrix} \xrightarrow{R_3 \leftarrow R_3 - R_2} \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \\ 0 & 0 & 0 \end{pmatrix}$$

**(a)** 主元有 2 个（第 1、2 列），故 $\text{rank}(A) = 2$。

**(b)** 各子空间：

**列空间** $\text{Col}(A)$（取主元列，即 $A$ 的第 1、2 列）：

$$\text{Col}(A) = \text{span}\!\left\{\begin{pmatrix}1\\0\\2\end{pmatrix},\ \begin{pmatrix}0\\1\\1\end{pmatrix}\right\}, \quad \dim = 2$$

**行空间** $\text{Row}(A)$（取行阶梯形的非零行，视为列向量）：

$$\text{Row}(A) = \text{span}\!\left\{\begin{pmatrix}1\\0\\2\end{pmatrix},\ \begin{pmatrix}0\\1\\-1\end{pmatrix}\right\}, \quad \dim = 2$$

**零空间** $\text{Null}(A)$：自由变量 $x_3 = t$，回代得 $x_2 = t$，$x_1 = -2t$：

$$\text{Null}(A) = \text{span}\!\left\{\begin{pmatrix}-2\\1\\1\end{pmatrix}\right\}, \quad \dim = 1$$

**左零空间** $\text{Null}(A^T)$：对 $A^T$ 做行化简，$\text{rank}(A^T) = 2$，故 $\dim(\text{Null}(A^T)) = 3 - 2 = 1$。解 $A^T \mathbf{y} = \mathbf{0}$，得基向量 $(2, -1, 1)^T$（验算略）：

$$\text{Null}(A^T) = \text{span}\!\left\{\begin{pmatrix}2\\-1\\1\end{pmatrix}\right\}, \quad \dim = 1$$

**(c)** 验证秩-零化度定理（$n = 3$ 列）：

$$\text{rank}(A) + \text{nullity}(A) = 2 + 1 = 3 = n \quad \checkmark$$

</details>

<details>
<summary>点击展开 练习3 答案</summary>

$A$ 是 $5 \times 7$ 矩阵（$m = 5$，$n = 7$），$\text{rank}(A) = 3$。

**(a)** 由秩-零化度定理：

$$\text{nullity}(A) = n - \text{rank}(A) = 7 - 3 = 4$$

$\text{Null}(A)$ 是 $\mathbb{R}^7$ 中的 4 维子空间。

**(b)** 列空间 $\text{Col}(A) \subseteq \mathbb{R}^m = \mathbb{R}^5$，$\dim(\text{Col}(A)) = \text{rank}(A) = 3$。

**(c)** 左零空间 $\text{Null}(A^T) \subseteq \mathbb{R}^m = \mathbb{R}^5$，由对 $A^T$ 应用秩-零化度定理：

$$\dim(\text{Null}(A^T)) = m - \text{rank}(A^T) = 5 - 3 = 2$$

**(d)** $A\mathbf{x} = \mathbf{b}$ 对所有 $\mathbf{b} \in \mathbb{R}^5$ 有解，当且仅当 $\text{Col}(A) = \mathbb{R}^5$，即 $\text{rank}(A) = m = 5$。但 $\text{rank}(A) = 3 < 5$，故**不是**对所有 $\mathbf{b}$ 都有解——只有 $\mathbf{b}$ 恰好落在 $\text{Col}(A)$ 这个三维子空间中时，方程组才有解。

</details>

<details>
<summary>点击展开 练习4 答案</summary>

**(a)** 判断线性无关性：

$U$ 的生成向量 $(1,0,1)^T$ 和 $(0,1,0)^T$ 显然线性无关，$\dim(U) = 2$。

$W$ 的生成向量 $(1,1,1)^T$ 和 $(0,1,-1)^T$：前者不是后者的标量倍，线性无关，$\dim(W) = 2$。

**(b)** 求 $U \cap W$：$\mathbf{v} \in U \cap W$ 意味着：

$$\mathbf{v} = a(1,0,1)^T + b(0,1,0)^T = c(1,1,1)^T + d(0,1,-1)^T$$

即 $(a, b, a)^T = (c, c+d, c-d)^T$，联立得 $a = c$，$b = c + d$，$a = c - d$，从前两个方程得 $d = 0$，$c = a$，$b = a$，即 $\mathbf{v} = a(1,1,1)^T$。

$$U \cap W = \text{span}\!\left\{(1,1,1)^T\right\}, \quad \dim(U \cap W) = 1$$

**(c)** 由维数公式：

$$\dim(U + W) = \dim(U) + \dim(W) - \dim(U \cap W) = 2 + 2 - 1 = 3$$

**(d)** 由于 $\dim(U + W) = 3 = \dim(\mathbb{R}^3)$，且 $U + W \subseteq \mathbb{R}^3$，故 $U + W = \mathbb{R}^3$。即两个子空间的和覆盖了整个三维空间。

</details>

<details>
<summary>点击展开 练习5 答案</summary>

**(a)** 矩阵 $A \in \mathbb{R}^{4 \times 512}$（$m=4$，$n=512$）作为线性映射 $\mathbb{R}^{512} \to \mathbb{R}^4$，由秩-零化度定理：

$$\text{nullity}(A) = n - \text{rank}(A) = 512 - \text{rank}(A)$$

由于 $A$ 最多 4 行，$\text{rank}(A) \leq 4$，故 $\text{nullity}(A) \geq 508$。

LoRA 微调实际上"忽略"了输入空间中至少 $508$ 个维度的方向——只有落在 $\text{Row}(A)$（最多 4 维）的输入分量才能通过 $A$ 产生有效输出；零空间（至少 508 维）中的输入成分被 $A$ 直接映射为零，对 LoRA 更新不做任何贡献。这正是 LoRA 低秩假设的核心：任务迁移只需要少数几个"方向"。

**(b)** $\Delta W = BA$，其中 $B \in \mathbb{R}^{512 \times 4}$，$A \in \mathbb{R}^{4 \times 512}$。由秩的乘积不等式：

$$\text{rank}(\Delta W) = \text{rank}(BA) \leq \min(\text{rank}(B), \text{rank}(A)) \leq 4$$

故 $\Delta W$ 的秩最大为 **4**（等于 $r$）。

**(c)** 参数量分析：

- 全量微调：更新 $W_0 \in \mathbb{R}^{512 \times 512}$，参数量 $= 512 \times 512 = 262144$
- LoRA 微调：更新 $B \in \mathbb{R}^{512 \times 4}$ 和 $A \in \mathbb{R}^{4 \times 512}$，参数量 $= 512 \times 4 + 4 \times 512 = 4096$
- 压缩比 $= 262144 / 4096 = \mathbf{64}$ 倍

**(d)** $W = W_0 + BA$ 的列空间分析：

$$\text{Col}(W) = \text{Col}(W_0 + BA)$$

由于 $\text{Col}(BA) \subseteq \text{Col}(B)$，而 $\text{Col}(B)$ 是 $\mathbb{R}^{512}$ 中至多 4 维的子空间，$\text{Col}(W)$ 是在 $\text{Col}(W_0)$ 的基础上，叠加了一个来自 $BA$ 的至多 4 维"扰动"。若 $W_0$ 是满秩的（$\text{rank}(W_0) = 512$），则 $\text{Col}(W_0) = \mathbb{R}^{512}$，加上低秩扰动后列空间不变，仍为 $\mathbb{R}^{512}$。若 $W_0$ 是低秩的，则 $\text{Col}(W)$ 可能比 $\text{Col}(W_0)$ 最多扩大 4 个维度。LoRA 的本质是：在一个低维子空间中精细调整映射方向，而不改变整体的大尺度结构。

</details>
