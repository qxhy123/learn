# 第13章：线性映射

> **前置知识**：第9章（向量空间）、第10章（线性相关与线性无关）、第11章（基与维数）、第12章（子空间）
>
> **本章难度**：★★★★☆
>
> **预计学习时间**：4-5 小时

---

## 学习目标

学完本章后，你将能够：

- 理解线性映射的严格定义（可加性与齐次性），并验证一个映射是否为线性映射
- 掌握旋转、投影、缩放、微分、积分等典型线性映射的具体形式
- 理解核（零空间）与像（值域）的定义，以及它们与线性映射性质的联系
- 区分单射、满射与同构，并运用秩-零化度定理分析线性映射的结构
- 理解线性映射的复合，以及神经网络全连接层作为线性变换的数学本质

---

## 13.1 线性映射的定义

### 什么是线性映射？

在数学中，映射是从一个集合到另一个集合的"规则"。当两个集合都是向量空间时，我们自然希望映射能够"尊重"向量空间的代数结构——即与加法和标量乘法相容。满足这一要求的映射，称为**线性映射（linear map）**，也叫**线性变换（linear transformation）**。

**定义**：设 $V$ 和 $W$ 是域 $\mathbb{F}$ 上的两个向量空间。映射 $T: V \to W$ 称为**线性映射**，如果它满足以下两条性质：

1. **可加性（Additivity）**：对所有 $\mathbf{u}, \mathbf{v} \in V$，有

$$T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$$

2. **齐次性（Homogeneity）**：对所有 $\mathbf{v} \in V$，$c \in \mathbb{F}$，有

$$T(c\mathbf{v}) = cT(\mathbf{v})$$

两条性质合并为一条等价条件——**对线性组合的保持性**：对所有 $\mathbf{u}, \mathbf{v} \in V$，$a, b \in \mathbb{F}$，有

$$\boxed{T(a\mathbf{u} + b\mathbf{v}) = aT(\mathbf{u}) + bT(\mathbf{v})}$$

更一般地，线性映射保持任意有限线性组合：

$$T\!\left(\sum_{i=1}^k c_i \mathbf{v}_i\right) = \sum_{i=1}^k c_i T(\mathbf{v}_i)$$

### 线性映射的基本性质

**命题**：若 $T: V \to W$ 是线性映射，则：

1. $T(\mathbf{0}_V) = \mathbf{0}_W$（零向量映射到零向量）
2. $T(-\mathbf{v}) = -T(\mathbf{v})$（负向量映射到负向量）

**证明（性质 1）**：令 $a = 0$，$b = 0$，由齐次性：$T(\mathbf{0}) = T(0 \cdot \mathbf{0}) = 0 \cdot T(\mathbf{0}) = \mathbf{0}$。$\square$

**几何直觉**：线性映射是"保持直线结构"的映射——原空间中的直线（或更一般的平坦子结构）在映射后仍然是直线（或平坦结构）；原点始终映射到原点。

### 线性映射的矩阵表示

当 $V = \mathbb{R}^n$，$W = \mathbb{R}^m$ 时，每个线性映射 $T: \mathbb{R}^n \to \mathbb{R}^m$ 都对应唯一一个 $m \times n$ 矩阵 $A$，使得：

$$T(\mathbf{x}) = A\mathbf{x}, \quad \forall \mathbf{x} \in \mathbb{R}^n$$

矩阵 $A$ 的第 $j$ 列恰好是 $T(\mathbf{e}_j)$——即标准基向量 $\mathbf{e}_j$ 在 $T$ 下的像：

$$A = \begin{pmatrix} T(\mathbf{e}_1) & T(\mathbf{e}_2) & \cdots & T(\mathbf{e}_n) \end{pmatrix}$$

这说明：**一个线性映射完全由它在基向量上的作用决定。** 知道了每个基向量的像，就唯一确定了整个映射。

### 验证线性映射

**例 1（是线性映射）**：$T: \mathbb{R}^2 \to \mathbb{R}^2$，$T\!\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}2x + y\\x - 3y\end{pmatrix}$。

验证：$T(a\mathbf{u} + b\mathbf{v}) = \begin{pmatrix}2(au_1+bv_1)+(au_2+bv_2)\\(au_1+bv_1)-3(au_2+bv_2)\end{pmatrix} = aT(\mathbf{u}) + bT(\mathbf{v})$。$\checkmark$

**例 2（不是线性映射）**：$f: \mathbb{R} \to \mathbb{R}$，$f(x) = x + 1$。

反例：$f(0) = 1 \neq 0$，违反了性质"零向量映射到零向量"。$f$ 是仿射映射，不是线性映射。

**例 3（不是线性映射）**：$g: \mathbb{R}^2 \to \mathbb{R}$，$g\!\begin{pmatrix}x\\y\end{pmatrix} = xy$（两分量之积）。

反例：$g\!\left(2 \begin{pmatrix}1\\1\end{pmatrix}\right) = g\!\begin{pmatrix}2\\2\end{pmatrix} = 4$，但 $2g\!\begin{pmatrix}1\\1\end{pmatrix} = 2 \cdot 1 = 2$。违反齐次性。

---

## 13.2 线性映射的例子

### 旋转

$\mathbb{R}^2$ 中将向量逆时针旋转角度 $\theta$ 的映射 $R_\theta: \mathbb{R}^2 \to \mathbb{R}^2$：

$$R_\theta\!\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}x\cos\theta - y\sin\theta\\x\sin\theta + y\cos\theta\end{pmatrix}$$

对应矩阵：

$$A_{R_\theta} = \begin{pmatrix}\cos\theta & -\sin\theta\\\sin\theta & \cos\theta\end{pmatrix}$$

**验证线性性**：旋转保持向量加法（平行四边形不变）和标量乘法（方向不变，长度等比缩放），因此是线性映射。

**几何意义**：旋转把整个平面绕原点转动，所有向量的长度不变，彼此之间的夹角不变。

### 投影

**正交投影**：将 $\mathbb{R}^2$ 中的向量投影到 $x$ 轴的映射 $P: \mathbb{R}^2 \to \mathbb{R}^2$：

$$P\!\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}x\\0\end{pmatrix}, \quad A_P = \begin{pmatrix}1&0\\0&0\end{pmatrix}$$

更一般地，在 $\mathbb{R}^n$ 中沿单位向量 $\hat{\mathbf{u}}$ 方向的正交投影：

$$P_{\hat{\mathbf{u}}}(\mathbf{v}) = (\mathbf{v} \cdot \hat{\mathbf{u}})\hat{\mathbf{u}}, \quad A_{P_{\hat{\mathbf{u}}}} = \hat{\mathbf{u}}\hat{\mathbf{u}}^T$$

**几何意义**：投影"压缩"了空间——将高维向量降维到低维子空间上。投影的像落在子空间内，核是该子空间的正交补。

### 缩放（伸缩）

将 $\mathbb{R}^n$ 中的每个向量乘以标量 $c$ 的映射：

$$S_c(\mathbf{v}) = c\mathbf{v}, \quad A_{S_c} = cI_n$$

其中 $I_n$ 是 $n$ 阶单位矩阵。特殊情况：

- $c = 1$：恒等映射 $\text{id}$
- $c = -1$：关于原点的反射（中心对称）
- $0 < c < 1$：收缩，$c > 1$：膨胀

各坐标方向独立缩放的**对角映射**：

$$D(\mathbf{x}) = \begin{pmatrix}\lambda_1 & & \\ & \ddots & \\ & & \lambda_n\end{pmatrix}\mathbf{x}$$

### 微分算子

设 $V = P_n$ 为次数不超过 $n$ 的实多项式空间，$W = P_{n-1}$。微分算子 $D: V \to W$ 定义为：

$$D(p) = p'$$

**验证线性性**：

- 可加性：$(p + q)' = p' + q'$ ✓
- 齐次性：$(cp)' = cp'$ ✓

以 $n = 3$ 为例，$V = P_3$ 的标准基为 $\{1, x, x^2, x^3\}$，映射矩阵为：

$$A_D = \begin{pmatrix}0 & 1 & 0 & 0\\0 & 0 & 2 & 0\\0 & 0 & 0 & 3\end{pmatrix}$$

**几何意义**：微分算子把多项式"降阶"——$n$ 次多项式的导数是 $n-1$ 次多项式。常数函数（"高度"不变的函数）的导数为零，构成微分算子的核。

### 积分算子

设 $V = C([0,1])$ 为 $[0,1]$ 上的连续函数空间，积分算子 $I: V \to \mathbb{R}$ 定义为：

$$I(f) = \int_0^1 f(x)\,dx$$

**验证线性性**：由积分的线性性直接得到：$I(af + bg) = a I(f) + b I(g)$。$\checkmark$

积分算子是从函数空间到实数的线性映射（泛函），在泛函分析中有深刻应用。

### 汇总

| 映射 | 定义域 $\to$ 值域 | 矩阵（若有限维） | 几何效果 |
|:---|:---|:---|:---|
| 旋转 $R_\theta$ | $\mathbb{R}^2 \to \mathbb{R}^2$ | $\begin{pmatrix}\cos\theta & -\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}$ | 保角保长旋转 |
| 投影 $P$ | $\mathbb{R}^n \to \mathbb{R}^n$ | $\hat{\mathbf{u}}\hat{\mathbf{u}}^T$ | 压缩到子空间 |
| 缩放 $S_c$ | $\mathbb{R}^n \to \mathbb{R}^n$ | $cI$ | 均匀伸缩 |
| 微分 $D$ | $P_n \to P_{n-1}$ | 上移矩阵 | 降阶 |
| 积分 $I$ | $C([0,1]) \to \mathbb{R}$ | —（无穷维） | 压缩为数 |

---

## 13.3 核与像

### 核（零空间）

**定义**：线性映射 $T: V \to W$ 的**核（kernel）**，也称**零空间（null space）**，是所有被映射到 $W$ 的零向量的输入向量组成的集合：

$$\ker(T) = \{\mathbf{v} \in V \mid T(\mathbf{v}) = \mathbf{0}_W\}$$

**命题**：$\ker(T)$ 是 $V$ 的子空间。

**证明**：
- 零向量：$T(\mathbf{0}_V) = \mathbf{0}_W$，故 $\mathbf{0}_V \in \ker(T)$ ✓
- 加法封闭：若 $T(\mathbf{u}) = T(\mathbf{v}) = \mathbf{0}$，则 $T(\mathbf{u}+\mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v}) = \mathbf{0}$ ✓
- 标量封闭：若 $T(\mathbf{v}) = \mathbf{0}$，则 $T(c\mathbf{v}) = cT(\mathbf{v}) = \mathbf{0}$ ✓ $\square$

**几何意义**：核是 $T$ "压缩成零"的所有方向——核的元素是线性映射"看不见"的输入。核越大，映射丢失的信息越多。

**与矩阵的关系**：若 $T(\mathbf{x}) = A\mathbf{x}$，则 $\ker(T) = \text{Null}(A) = \{\mathbf{x} \mid A\mathbf{x} = \mathbf{0}\}$。

**例**：设 $T: \mathbb{R}^3 \to \mathbb{R}^2$，$T\!\begin{pmatrix}x\\y\\z\end{pmatrix} = \begin{pmatrix}x + y\\y + z\end{pmatrix}$，对应矩阵 $A = \begin{pmatrix}1&1&0\\0&1&1\end{pmatrix}$。

解 $A\mathbf{x} = \mathbf{0}$：行化简得自由变量 $z = t$，回代得 $y = -t$，$x = t$。

$$\ker(T) = \text{span}\!\left\{\begin{pmatrix}1\\-1\\1\end{pmatrix}\right\}$$

核是 $\mathbb{R}^3$ 中的一条直线（一维子空间）。

### 像（值域）

**定义**：线性映射 $T: V \to W$ 的**像（image）**，也称**值域（range）**，是所有可能输出的集合：

$$\text{Im}(T) = \{T(\mathbf{v}) \mid \mathbf{v} \in V\} = T(V)$$

**命题**：$\text{Im}(T)$ 是 $W$ 的子空间。

**证明**：
- 零向量：$T(\mathbf{0}_V) = \mathbf{0}_W \in \text{Im}(T)$ ✓
- 加法封闭：若 $\mathbf{w}_1 = T(\mathbf{v}_1)$，$\mathbf{w}_2 = T(\mathbf{v}_2)$，则 $\mathbf{w}_1 + \mathbf{w}_2 = T(\mathbf{v}_1 + \mathbf{v}_2) \in \text{Im}(T)$ ✓
- 标量封闭：若 $\mathbf{w} = T(\mathbf{v})$，则 $c\mathbf{w} = T(c\mathbf{v}) \in \text{Im}(T)$ ✓ $\square$

**与矩阵的关系**：若 $T(\mathbf{x}) = A\mathbf{x}$，则 $\text{Im}(T) = \text{Col}(A)$——即矩阵 $A$ 的列空间。

**例（承上）**：$\text{Im}(T) = \text{Col}(A) = \text{span}\!\left\{\begin{pmatrix}1\\0\end{pmatrix}, \begin{pmatrix}1\\1\end{pmatrix}\right\}$。

由于这两列线性无关，$\text{Im}(T) = \mathbb{R}^2$（整个目标空间）。

### 核与像的维数关系：秩-零化度定理

**定理（秩-零化度定理，Rank-Nullity Theorem）**：设 $T: V \to W$ 是有限维向量空间 $V$ 上的线性映射，则：

$$\boxed{\dim(\ker(T)) + \dim(\text{Im}(T)) = \dim(V)}$$

即：**核的维数（零化度）+ 像的维数（秩）= 定义域维数**。

**直觉**：$V$ 的维数代表"自由度总量"。这些自由度分配给两类：一类是"被压缩成零"的方向（核），一类是"有效传递到输出"的方向（像），两者加起来恰好用完所有自由度。

**例（承上）**：$\dim(\ker(T)) = 1$，$\dim(\text{Im}(T)) = 2$，$\dim(\mathbb{R}^3) = 3$：$1 + 2 = 3$ ✓

---

## 13.4 单射、满射与同构

### 单射（Injective）

**定义**：若 $T(\mathbf{u}) = T(\mathbf{v}) \Rightarrow \mathbf{u} = \mathbf{v}$，则称 $T$ 是**单射（injective）**，也称**一一映射（one-to-one）**。

**等价条件**：$T$ 是单射，当且仅当 $\ker(T) = \{\mathbf{0}\}$。

**证明（充分性）**：若 $\ker(T) = \{\mathbf{0}\}$，设 $T(\mathbf{u}) = T(\mathbf{v})$，则 $T(\mathbf{u} - \mathbf{v}) = \mathbf{0}$，故 $\mathbf{u} - \mathbf{v} \in \ker(T) = \{\mathbf{0}\}$，即 $\mathbf{u} = \mathbf{v}$。$\square$

**几何意义**：单射意味着不同的输入有不同的输出——映射不"压缩"，不丢失信息。对应矩阵满足 $A\mathbf{x} = \mathbf{0}$ 只有零解（列满秩）。

**维数条件**：若 $T: V \to W$ 单射，则 $\dim(V) \leq \dim(W)$。（单射把低维空间"嵌入"高维空间。）

### 满射（Surjective）

**定义**：若 $\text{Im}(T) = W$（像等于整个目标空间），则称 $T$ 是**满射（surjective）**，也称**映上（onto）**。

**等价条件**：$T$ 是满射，当且仅当对每个 $\mathbf{w} \in W$，方程 $T(\mathbf{v}) = \mathbf{w}$ 有解。对应矩阵 $A$ 是行满秩（$\text{rank}(A) = m$）。

**几何意义**：满射意味着输出空间的每个点都"被覆盖"——映射能够到达 $W$ 中的任何位置。

**维数条件**：若 $T: V \to W$ 满射，则 $\dim(V) \geq \dim(W)$。（满射意味着定义域"足够大"以覆盖整个目标空间。）

### 同构（Isomorphism）

**定义**：既是单射又是满射的线性映射称为**同构（isomorphism）**，也称**双射线性映射**。此时称 $V$ 与 $W$ **同构**，记作 $V \cong W$。

**等价条件**：$T: V \to W$ 是同构，当且仅当：
- $\ker(T) = \{\mathbf{0}\}$（单射）
- $\text{Im}(T) = W$（满射）

对应矩阵 $A$ 是可逆方阵（满秩）。

**重要定理**：两个有限维向量空间同构，当且仅当它们的**维数相同**：

$$V \cong W \iff \dim(V) = \dim(W)$$

**推论**：任何 $n$ 维实向量空间都与 $\mathbb{R}^n$ 同构。这说明所有 $n$ 维实向量空间在代数结构上"本质相同"，$\mathbb{R}^n$ 是它们的代表。

### 四种类型的汇总

| 类型 | 条件 | 等价矩阵条件 | 几何含义 |
|:---|:---|:---|:---|
| 一般线性映射 | — | — | 保持线性结构 |
| 单射 | $\ker(T) = \{\mathbf{0}\}$ | 列满秩 | 不压缩，信息无损 |
| 满射 | $\text{Im}(T) = W$ | 行满秩 | 覆盖全目标空间 |
| 同构 | 单射 + 满射 | 方阵且满秩（可逆） | 两空间结构等价 |

---

## 13.5 线性映射的复合

### 复合的定义

设 $T_1: U \to V$ 和 $T_2: V \to W$ 都是线性映射，它们的**复合（composition）** $T_2 \circ T_1: U \to W$ 定义为：

$$(T_2 \circ T_1)(\mathbf{u}) = T_2(T_1(\mathbf{u})), \quad \forall \mathbf{u} \in U$$

**命题**：两个线性映射的复合仍然是线性映射。

**证明**：
$$(T_2 \circ T_1)(a\mathbf{u} + b\mathbf{v}) = T_2(T_1(a\mathbf{u} + b\mathbf{v})) = T_2(aT_1(\mathbf{u}) + bT_1(\mathbf{v}))$$
$$= aT_2(T_1(\mathbf{u})) + bT_2(T_1(\mathbf{v})) = a(T_2 \circ T_1)(\mathbf{u}) + b(T_2 \circ T_1)(\mathbf{v}) \quad \square$$

### 复合与矩阵乘法

当 $T_1$ 对应矩阵 $A_1 \in \mathbb{R}^{p \times n}$，$T_2$ 对应矩阵 $A_2 \in \mathbb{R}^{m \times p}$ 时，复合 $T_2 \circ T_1$ 对应矩阵 $A_2 A_1 \in \mathbb{R}^{m \times n}$：

$$(T_2 \circ T_1)(\mathbf{x}) = T_2(A_1\mathbf{x}) = A_2(A_1\mathbf{x}) = (A_2 A_1)\mathbf{x}$$

**这正是矩阵乘法的几何含义**：矩阵乘法 $A_2 A_1$ 表示先施加变换 $A_1$，再施加变换 $A_2$。

### 复合的代数性质

设所有映射的维数相容，则：

1. **结合律**：$(T_3 \circ T_2) \circ T_1 = T_3 \circ (T_2 \circ T_1)$
2. **一般不满足交换律**：$T_2 \circ T_1 \neq T_1 \circ T_2$（即使两者都有意义）
3. **与恒等映射复合**：$T \circ \text{id}_V = \text{id}_W \circ T = T$

**例（旋转的复合）**：$\mathbb{R}^2$ 中，先旋转 $\alpha$ 再旋转 $\beta$，等于旋转 $\alpha + \beta$：

$$R_\beta \circ R_\alpha = R_{\alpha+\beta}$$

用矩阵验证：

$$\begin{pmatrix}\cos\beta&-\sin\beta\\\sin\beta&\cos\beta\end{pmatrix}\begin{pmatrix}\cos\alpha&-\sin\alpha\\\sin\alpha&\cos\alpha\end{pmatrix} = \begin{pmatrix}\cos(\alpha+\beta)&-\sin(\alpha+\beta)\\\sin(\alpha+\beta)&\cos(\alpha+\beta)\end{pmatrix}$$

这同时也是三角函数加法公式的矩阵证明。

### 逆映射

若 $T: V \to W$ 是同构（双射线性映射），则存在唯一的**逆映射** $T^{-1}: W \to V$，满足：

$$T^{-1} \circ T = \text{id}_V, \quad T \circ T^{-1} = \text{id}_W$$

$T^{-1}$ 也是线性映射，对应矩阵为 $A^{-1}$（若 $T$ 由 $A$ 表示）。

---

## 本章小结

| 概念 | 定义 | 关键性质 |
|:---|:---|:---|
| 线性映射 $T: V \to W$ | 满足可加性和齐次性 | 由基上的作用唯一确定；保零向量 |
| 核 $\ker(T)$ | $\{\mathbf{v} \mid T(\mathbf{v}) = \mathbf{0}\}$ | $V$ 的子空间；$= \text{Null}(A)$ |
| 像 $\text{Im}(T)$ | $\{T(\mathbf{v}) \mid \mathbf{v} \in V\}$ | $W$ 的子空间；$= \text{Col}(A)$ |
| 秩-零化度定理 | $\dim(\ker T) + \dim(\text{Im}\, T) = \dim V$ | 核与像分配了定义域的全部维度 |
| 单射 | $\ker(T) = \{\mathbf{0}\}$ | 列满秩；信息无损 |
| 满射 | $\text{Im}(T) = W$ | 行满秩；覆盖目标空间 |
| 同构 | 单射 + 满射 | $\dim V = \dim W$；可逆方阵 |
| 复合 $T_2 \circ T_1$ | 先 $T_1$ 后 $T_2$ | 对应矩阵乘法 $A_2 A_1$ |

**核心逻辑**：线性映射是向量空间之间的"结构保持映射"，矩阵是其有限维表示。核刻画了"信息丢失"的维度，像刻画了"可达输出"的维度，两者维数之和等于输入空间的维数。单射意味着无信息丢失，满射意味着全面覆盖，同构意味着两个空间在结构上完全等价。

**下一章预告**：线性映射的矩阵表示——如何在不同基下描述同一个线性映射，以及基变换矩阵的推导。

---

## 深度学习应用：神经网络层作为线性变换

### 全连接层的数学本质

神经网络的基本构件是**层（layer）**，其中最基础的是**全连接层（fully connected layer）**，也称**线性层（linear layer）**。

设输入向量 $\mathbf{x} \in \mathbb{R}^n$，全连接层（不含激活函数）的计算为：

$$\mathbf{y} = W\mathbf{x} + \mathbf{b}$$

其中 $W \in \mathbb{R}^{m \times n}$ 是**权重矩阵**，$\mathbf{b} \in \mathbb{R}^m$ 是**偏置向量**，$\mathbf{y} \in \mathbb{R}^m$ 是输出向量。

**线性映射视角**：当 $\mathbf{b} = \mathbf{0}$ 时，$\mathbf{y} = W\mathbf{x}$ 是从 $\mathbb{R}^n$ 到 $\mathbb{R}^m$ 的线性映射，完全符合本章的定义。当 $\mathbf{b} \neq \mathbf{0}$ 时，$\mathbf{y} = W\mathbf{x} + \mathbf{b}$ 是**仿射映射（affine map）**——线性映射加上一个平移。

$$\underbrace{\mathbf{y} = W\mathbf{x} + \mathbf{b}}_{\text{仿射映射}} = \underbrace{W\mathbf{x}}_{\text{线性部分}} + \underbrace{\mathbf{b}}_{\text{平移}}$$

仿射映射不再把原点映射到原点（当 $\mathbf{b} \neq \mathbf{0}$），但若将其齐次化（扩充维度），仍可用矩阵乘法表示：

$$\begin{pmatrix}\mathbf{y}\\1\end{pmatrix} = \begin{pmatrix}W & \mathbf{b}\\\mathbf{0}^T & 1\end{pmatrix}\begin{pmatrix}\mathbf{x}\\1\end{pmatrix}$$

### 激活函数打破线性性

**关键问题**：若神经网络所有层都是线性映射，整个网络的复合仍然是线性映射！

$$T_k \circ \cdots \circ T_2 \circ T_1 = \text{线性映射}$$

无论堆叠多少层，效果等同于单个矩阵 $W_k \cdots W_2 W_1$。这样的网络无法学习非线性关系。

**激活函数的作用**：在每层之间插入非线性函数 $\sigma$（如 ReLU、Sigmoid），破坏整体的线性性：

$$\mathbf{h}^{(1)} = \sigma(W_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{h}^{(2)} = \sigma(W_2 \mathbf{h}^{(1)} + \mathbf{b}_2), \quad \ldots$$

这样，网络整体成为强大的非线性函数近似器。**每个线性层负责"旋转/缩放/投影"特征空间，激活函数负责引入非线性。**

### 核（零空间）的信息论解释

设全连接层 $T(\mathbf{x}) = W\mathbf{x}$，其核为 $\ker(T) = \text{Null}(W)$。

由秩-零化度定理：

$$\dim(\ker T) = n - \text{rank}(W)$$

**信息论视角**：核的维数代表**被这一层"丢弃"的输入信息的维度数**。

- 若 $\text{rank}(W) = n$（列满秩，$m \geq n$）：核只含零向量，不丢失信息——单射。
- 若 $\text{rank}(W) = m < n$（行满秩，$m < n$）：核是 $n-m$ 维子空间，将输入压缩到 $m$ 维——**降维**。
- 若 $\text{rank}(W) < \min(m, n)$（秩亏缺）：核更大，信息丢失更严重。

**实际含义**：当网络层将 $n$ 维输入压缩到 $m < n$ 维时，输入空间中 $\ker(W)$ 方向的所有信息都会被彻底丢弃，无法从输出恢复。这正是自编码器（Autoencoder）瓶颈层的工作原理——强迫网络只保留最关键的 $m$ 维特征。

### 可视化理解

```
输入空间 R^n
    │
    ├──── 行空间（rank(W) 维）────────→ 输出空间 R^m
    │     （"有效"方向，信息保留）          （像 Im(W)）
    │
    └──── 核 Null(W)（n - rank(W) 维）──→ 0
          （"无效"方向，信息丢失）
```

### 代码示例

```python
import torch
import torch.nn as nn

# ============================================================
# 1. 全连接层即线性映射：检验可加性与齐次性
# ============================================================
torch.manual_seed(42)

# 定义一个线性层（无偏置）：R^4 -> R^3
linear = nn.Linear(in_features=4, out_features=3, bias=False)
W = linear.weight   # shape: (3, 4)，即 W ∈ R^{3×4}

print("权重矩阵 W (3×4):")
print(W.data)

# 随机输入向量
u = torch.randn(4)
v = torch.randn(4)
a, b = 2.0, -1.5

with torch.no_grad():
    # 验证可加性：T(u + v) == T(u) + T(v)
    lhs_add = linear(u + v)
    rhs_add = linear(u) + linear(v)
    print(f"\n可加性误差: {(lhs_add - rhs_add).abs().max().item():.2e}")  # ≈ 0

    # 验证齐次性：T(a*u) == a*T(u)
    lhs_hom = linear(a * u)
    rhs_hom = a * linear(u)
    print(f"齐次性误差: {(lhs_hom - rhs_hom).abs().max().item():.2e}")    # ≈ 0

# ============================================================
# 2. 核（零空间）的计算：哪些输入被"压缩"成零？
# ============================================================
W_np = W.data.numpy()

# 用 SVD 计算零空间基（Vh 最后 nullity 行）
import numpy as np
U_svd, S, Vh = np.linalg.svd(W_np)

tol = 1e-6
rank = int((S > tol).sum())
nullity = 4 - rank

print(f"\nrank(W)   = {rank}")      # 最多 3（行数）
print(f"nullity(W) = {nullity}")    # = 4 - rank
print(f"秩 + 零化度 = {rank + nullity} = n = 4")

# 零空间的标准正交基
null_basis = Vh[rank:, :]           # shape: (nullity, 4)
print(f"\n零空间基向量（共 {nullity} 个）:")
print(null_basis)

# 验证：W @ null_basis.T ≈ 0
null_vecs = torch.tensor(null_basis, dtype=torch.float32)
with torch.no_grad():
    output = linear(null_vecs[0])   # 第一个零空间基向量的像
    print(f"\n零空间向量的像（应接近零向量）:")
    print(f"{output.numpy()}  (max abs = {output.abs().max().item():.2e})")

# ============================================================
# 3. 多层线性网络 = 单个线性映射（无激活函数时）
# ============================================================
W1 = torch.randn(5, 4)   # R^4 -> R^5
W2 = torch.randn(3, 5)   # R^5 -> R^3

# 两层线性网络
def two_layer_linear(x):
    return (W2 @ W1) @ x   # 复合 = 矩阵乘法

# 等价的单层网络
W_composed = W2 @ W1      # shape: (3, 4)，直接合并
x = torch.randn(4)

with torch.no_grad():
    y1 = W2 @ (W1 @ x)        # 两层顺序计算
    y2 = W_composed @ x        # 单层等价计算
    print(f"\n两层线性网络 vs 等价单矩阵，差异: {(y1 - y2).abs().max().item():.2e}")
    # ≈ 0，证明多层线性网络本质上是单层

# ============================================================
# 4. 激活函数打破线性性
# ============================================================
relu = nn.ReLU()

with torch.no_grad():
    # 验证 ReLU 不满足齐次性（当 a < 0 时）
    x_pos = torch.tensor([1.0, 2.0, 3.0, 4.0])
    a_neg = -1.0

    lhs = relu(a_neg * x_pos)           # relu(-x)，所有元素变为 0
    rhs = a_neg * relu(x_pos)           # -relu(x)，所有元素为负
    print(f"\nReLU 齐次性误差（a=-1）: {(lhs - rhs).abs().max().item():.4f}")
    # 非零，说明 ReLU 破坏了线性性

# ============================================================
# 5. 带激活函数的深层网络：真正的非线性映射
# ============================================================
class MLP(nn.Module):
    """两层 MLP，激活函数使网络具有非线性表达能力"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)    # 线性映射 R^4 -> R^8
        self.relu = nn.ReLU()          # 非线性激活
        self.fc2 = nn.Linear(8, 3)    # 线性映射 R^8 -> R^3

    def forward(self, x):
        # 结构：线性 -> 非线性 -> 线性
        h = self.relu(self.fc1(x))     # 隐藏层
        return self.fc2(h)             # 输出层

mlp = MLP()
x1, x2 = torch.randn(4), torch.randn(4)

with torch.no_grad():
    # 验证 MLP 不满足可加性（因为 ReLU 破坏了线性性）
    out_sum = mlp(x1 + x2)
    out_add = mlp(x1) + mlp(x2)
    diff = (out_sum - out_add).abs().max().item()
    print(f"\nMLP 可加性误差（有 ReLU）: {diff:.4f}")
    # 通常非零，说明 MLP 整体是非线性的

print("\n结论：激活函数是神经网络非线性表达能力的来源，")
print("      而线性层提供了旋转、缩放、投影等几何变换能力。")
```

**代码解读**：

- **第1部分**：直接验证 `nn.Linear`（无偏置）满足线性映射的两条性质，误差约为机器精度 $10^{-7}$，确认其线性性。
- **第2部分**：用 SVD 计算权重矩阵的零空间基，验证秩-零化度定理，并直观展示"零空间向量被压缩成零"。
- **第3部分**：证明多层纯线性网络等价于单个矩阵乘法——这是深度学习必须使用激活函数的根本原因。
- **第4部分**：展示 ReLU 对负数输入不满足齐次性，从而打破线性性。
- **第5部分**：完整的两层 MLP 验证了引入 ReLU 后整体映射的非线性性。

### 延伸思考

| 线性代数概念 | 神经网络对应物 | 实际意义 |
|:---|:---|:---|
| 线性映射 $T(\mathbf{x}) = W\mathbf{x}$ | 全连接层（无偏置） | 旋转、缩放、投影特征 |
| 仿射映射 $W\mathbf{x} + \mathbf{b}$ | 全连接层（含偏置） | 平移决策边界 |
| 核 $\ker(W)$ | 被丢弃的特征方向 | 瓶颈层的信息压缩 |
| 像 $\text{Im}(W)$ | 输出特征子空间 | 模型的"表达空间" |
| 复合 $W_2 W_1$ | 多层线性网络 | 等价于单层（须加激活） |
| 同构（可逆映射） | 可逆网络（如 Flow 模型） | 精确重建输入 |

---

## 练习题

**练习 1**（基础——验证线性映射）

判断下列映射是否为线性映射，并给出完整证明或反例：

（a）$T: \mathbb{R}^2 \to \mathbb{R}^2$，$T\!\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}3x - y\\2x + 5y\end{pmatrix}$

（b）$f: \mathbb{R}^2 \to \mathbb{R}$，$f\!\begin{pmatrix}x\\y\end{pmatrix} = x^2 + y^2$（平方和）

（c）$g: \mathbb{R}^3 \to \mathbb{R}^2$，$g\!\begin{pmatrix}x\\y\\z\end{pmatrix} = \begin{pmatrix}x + z\\2y - x\end{pmatrix}$

---

**练习 2**（基础——核与像）

设 $T: \mathbb{R}^3 \to \mathbb{R}^2$ 的对应矩阵为：

$$A = \begin{pmatrix}1 & -1 & 2\\3 & 0 & 1\end{pmatrix}$$

（a）求 $\ker(T)$（给出基向量）

（b）求 $\text{Im}(T)$（给出基向量）

（c）验证秩-零化度定理

---

**练习 3**（中等——单射与满射）

设 $T: \mathbb{R}^4 \to \mathbb{R}^3$ 是线性映射，对应矩阵为：

$$A = \begin{pmatrix}1 & 0 & -1 & 2\\0 & 1 & 3 & -1\\2 & -1 & -5 & 5\end{pmatrix}$$

（a）$T$ 是单射吗？说明理由。

（b）$T$ 是满射吗？说明理由。

（c）$\ker(T)$ 的维数是多少？$\text{Im}(T)$ 的维数是多少？

---

**练习 4**（中等——旋转的复合）

设 $R_\theta$ 是 $\mathbb{R}^2$ 中逆时针旋转角度 $\theta$ 的线性映射。

（a）写出 $R_{90°}$ 和 $R_{180°}$ 的矩阵。

（b）计算 $R_{90°} \circ R_{90°}$（即连续旋转两次 $90°$），验证结果等于 $R_{180°}$。

（c）证明旋转映射 $R_\theta$ 是 $\mathbb{R}^2$ 上的同构。（提示：找出 $R_\theta^{-1}$。）

---

**练习 5**（进阶——神经网络的线性代数分析）

设神经网络第一层权重矩阵 $W_1 \in \mathbb{R}^{3 \times 5}$（将 $\mathbb{R}^5$ 映射到 $\mathbb{R}^3$），第二层权重矩阵 $W_2 \in \mathbb{R}^{2 \times 3}$（将 $\mathbb{R}^3$ 映射到 $\mathbb{R}^2$），均不含激活函数。

（a）用秩-零化度定理，说明 $T_1(\mathbf{x}) = W_1 \mathbf{x}$ 的核的维数下界。

（b）设 $\text{rank}(W_1) = 3$，$\text{rank}(W_2) = 2$，计算复合映射 $T_2 \circ T_1: \mathbb{R}^5 \to \mathbb{R}^2$ 对应矩阵 $W_2 W_1$ 的秩（给出上界和可能的取值）。

（c）若在两层之间加入 ReLU 激活函数，复合映射是否还能等价为单个矩阵乘法？为什么？

（d）从信息论角度解释：这个两层无激活网络（从 $\mathbb{R}^5$ 到 $\mathbb{R}^2$），最多能保留输入的多少维度的信息？这与核的维数有何关系？

---

## 练习答案

<details>
<summary>点击展开 练习1 答案</summary>

**（a）$T\!\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}3x-y\\2x+5y\end{pmatrix}$：是线性映射**

设 $\mathbf{u} = (u_1, u_2)^T$，$\mathbf{v} = (v_1, v_2)^T$，$a, b \in \mathbb{R}$：

$$T(a\mathbf{u} + b\mathbf{v}) = T\!\begin{pmatrix}au_1+bv_1\\au_2+bv_2\end{pmatrix} = \begin{pmatrix}3(au_1+bv_1)-(au_2+bv_2)\\2(au_1+bv_1)+5(au_2+bv_2)\end{pmatrix}$$

$$= a\begin{pmatrix}3u_1-u_2\\2u_1+5u_2\end{pmatrix} + b\begin{pmatrix}3v_1-v_2\\2v_1+5v_2\end{pmatrix} = aT(\mathbf{u}) + bT(\mathbf{v}) \quad \checkmark$$

对应矩阵：$A = \begin{pmatrix}3&-1\\2&5\end{pmatrix}$。

**（b）$f\!\begin{pmatrix}x\\y\end{pmatrix} = x^2 + y^2$：不是线性映射**

反例（违反齐次性）：令 $\mathbf{v} = (1, 0)^T$，$c = 2$：

$$f(2\mathbf{v}) = f\!\begin{pmatrix}2\\0\end{pmatrix} = 4, \quad 2f(\mathbf{v}) = 2f\!\begin{pmatrix}1\\0\end{pmatrix} = 2 \cdot 1 = 2$$

$4 \neq 2$，齐次性不成立。$f$ 是二次映射，不是线性映射。

**（c）$g\!\begin{pmatrix}x\\y\\z\end{pmatrix} = \begin{pmatrix}x+z\\2y-x\end{pmatrix}$：是线性映射**

对应矩阵 $A = \begin{pmatrix}1&0&1\\-1&2&0\end{pmatrix}$，映射形如 $g(\mathbf{v}) = A\mathbf{v}$，矩阵乘法保证线性性。$\checkmark$

</details>

<details>
<summary>点击展开 练习2 答案</summary>

对矩阵 $A = \begin{pmatrix}1&-1&2\\3&0&1\end{pmatrix}$ 进行行化简：

$$\begin{pmatrix}1&-1&2\\3&0&1\end{pmatrix} \xrightarrow{R_2 \leftarrow R_2 - 3R_1} \begin{pmatrix}1&-1&2\\0&3&-5\end{pmatrix}$$

**（a）$\ker(T)$**：解 $A\mathbf{x} = \mathbf{0}$，自由变量为 $x_3 = t$。

由第二行：$3x_2 = 5t \Rightarrow x_2 = \tfrac{5}{3}t$；由第一行：$x_1 = x_2 - 2t = \tfrac{5}{3}t - 2t = -\tfrac{1}{3}t$。

取 $t = 3$，得基向量 $(-1, 5, 3)^T$：

$$\ker(T) = \text{span}\!\left\{\begin{pmatrix}-1\\5\\3\end{pmatrix}\right\}, \quad \dim(\ker T) = 1$$

**（b）$\text{Im}(T)$**：行化简后有 2 个主元（第 1、2 列），取原矩阵对应列：

$$\text{Im}(T) = \text{span}\!\left\{\begin{pmatrix}1\\3\end{pmatrix},\ \begin{pmatrix}-1\\0\end{pmatrix}\right\}, \quad \dim(\text{Im}\, T) = 2$$

（注意 $\text{Im}(T) \subseteq \mathbb{R}^2$，两列线性无关，故 $\text{Im}(T) = \mathbb{R}^2$。）

**（c）验证秩-零化度定理**（$\dim V = \dim\mathbb{R}^3 = 3$）：

$$\dim(\ker T) + \dim(\text{Im}\, T) = 1 + 2 = 3 = \dim(\mathbb{R}^3) \quad \checkmark$$

</details>

<details>
<summary>点击展开 练习3 答案</summary>

对矩阵 $A$ 进行行化简：

$$\begin{pmatrix}1&0&-1&2\\0&1&3&-1\\2&-1&-5&5\end{pmatrix} \xrightarrow{R_3 \leftarrow R_3 - 2R_1} \begin{pmatrix}1&0&-1&2\\0&1&3&-1\\0&-1&-3&1\end{pmatrix} \xrightarrow{R_3 \leftarrow R_3 + R_2} \begin{pmatrix}1&0&-1&2\\0&1&3&-1\\0&0&0&0\end{pmatrix}$$

$\text{rank}(A) = 2$（两个主元列），自由列为第 3、4 列。

**（a）$T$ 不是单射**。因为 $\ker(T)$ 不只含零向量：$A\mathbf{x} = \mathbf{0}$ 有非零解（自由变量 $x_3, x_4$ 可任取）。等价地，$\text{rank}(A) = 2 < 4 = n$，不是列满秩。

**（b）$T$ 是满射**。$\text{Im}(T) = \text{Col}(A)$，$\dim(\text{Im}\, T) = 2 = m$（目标空间 $\mathbb{R}^3$ 的……

等等，$m = 3$（$A$ 有 3 行），$\dim(\text{Im}\, T) = 2 \neq 3$，故**不是满射**。$\text{Im}(T)$ 是 $\mathbb{R}^3$ 中的二维子空间（平面），无法覆盖整个 $\mathbb{R}^3$。

**（c）由秩-零化度定理**（$n = 4$，$\text{rank}(A) = 2$）：

$$\dim(\ker T) = n - \text{rank}(A) = 4 - 2 = 2$$
$$\dim(\text{Im}\, T) = \text{rank}(A) = 2$$

</details>

<details>
<summary>点击展开 练习4 答案</summary>

**（a）旋转矩阵**：

$$R_{90°} = \begin{pmatrix}\cos 90° & -\sin 90°\\\sin 90° & \cos 90°\end{pmatrix} = \begin{pmatrix}0&-1\\1&0\end{pmatrix}$$

$$R_{180°} = \begin{pmatrix}\cos 180° & -\sin 180°\\\sin 180° & \cos 180°\end{pmatrix} = \begin{pmatrix}-1&0\\0&-1\end{pmatrix}$$

**（b）复合验证**：

$$R_{90°} \circ R_{90°} \leftrightarrow R_{90°}^2 = \begin{pmatrix}0&-1\\1&0\end{pmatrix}^2 = \begin{pmatrix}0\cdot0+(-1)\cdot1 & 0\cdot(-1)+(-1)\cdot0\\1\cdot0+0\cdot1 & 1\cdot(-1)+0\cdot0\end{pmatrix} = \begin{pmatrix}-1&0\\0&-1\end{pmatrix} = R_{180°} \quad \checkmark$$

**（c）$R_\theta$ 是同构**：

旋转矩阵 $A_{R_\theta} = \begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}$ 的行列式为 $\cos^2\theta + \sin^2\theta = 1 \neq 0$，故 $A_{R_\theta}$ 可逆，$R_\theta$ 是 $\mathbb{R}^2$ 上的同构。

逆映射：$R_\theta^{-1} = R_{-\theta}$（逆时针旋转 $\theta$ 的逆是顺时针旋转 $\theta$），对应矩阵 $A_{R_\theta}^{-1} = \begin{pmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{pmatrix} = A_{R_{-\theta}}$。$\square$

</details>

<details>
<summary>点击展开 练习5 答案</summary>

**（a）$T_1$ 的核的维数下界**：

$T_1: \mathbb{R}^5 \to \mathbb{R}^3$，$W_1 \in \mathbb{R}^{3 \times 5}$。

$\text{rank}(W_1) \leq \min(3, 5) = 3$，由秩-零化度定理：

$$\dim(\ker T_1) = 5 - \text{rank}(W_1) \geq 5 - 3 = 2$$

无论 $W_1$ 如何设置，核至少是二维的——从 $\mathbb{R}^5$ 映射到 $\mathbb{R}^3$，至少有 $5 - 3 = 2$ 个维度的信息被丢弃。

**（b）复合映射的秩**：

$W_2 W_1 \in \mathbb{R}^{2 \times 5}$，由秩的乘积不等式：

$$\text{rank}(W_2 W_1) \leq \min(\text{rank}(W_2), \text{rank}(W_1)) = \min(2, 3) = 2$$

实际上，$\text{rank}(W_2 W_1) \leq 2$（不超过目标空间维数 $m = 2$），可能取值为 $0, 1, 2$。若 $W_1$ 列满秩（$\text{rank} = 3$），$W_2$ 满秩（$\text{rank} = 2$），则通常 $\text{rank}(W_2 W_1) = 2$。

**（c）加入 ReLU 后的非线性性**：

不能。ReLU 定义为 $\sigma(x) = \max(0, x)$，对负数置零，对正数不变。这不满足齐次性（$\sigma(-x) \neq -\sigma(x)$），因此：

$$T_2(\sigma(T_1(\mathbf{x}))) \neq (W_2 W_1)\mathbf{x}$$

整体复合是分段线性函数，不再是线性映射，无法用单个矩阵乘法表示。

**（d）信息保留维度**：

最终输出在 $\mathbb{R}^2$ 中，$\dim(\text{Im}(T_2 \circ T_1)) \leq 2$，即最多保留 2 维信息。

由秩-零化度定理，$\ker(W_2 W_1)$ 的维数 $= 5 - \text{rank}(W_2 W_1) \geq 5 - 2 = 3$。

**核的维数反映了丢失信息的维度**：至少 3 维的输入信息被"压缩成零"，无法从输出恢复。这是从 $\mathbb{R}^5$ 压缩到 $\mathbb{R}^2$ 的必然代价——自编码器、PCA 等降维方法的瓶颈层利用的正是这一机制，有意识地丢弃"低价值"维度。

</details>
