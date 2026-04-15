# 第1章：向量

> **前置知识**：高中数学（基本代数运算，平面几何）
>
> **本章难度**：★☆☆☆☆
>
> **预计学习时间**：2-3 小时

---

## 学习目标

学完本章后，你将能够：

- 理解向量的定义和两种表示方法（几何与代数）
- 掌握向量的加法、标量乘法、减法等基本运算，以及它们的运算性质
- 理解向量的几何意义：长度（范数）、方向、两向量之间的夹角
- 掌握向量内积（点积）的定义、性质，以及内积与夹角的关系
- 理解正交向量的概念及其重要性
- 了解向量在高维空间中的推广，以及在深度学习中的核心应用

---

## 1.1 向量的定义与表示

### 从标量到向量：为什么需要方向？

在日常生活中，我们经常处理两类不同的量。

**第一类**：只需要一个数字就能完整描述的量，例如温度 25°C、体重 70 kg、银行账户余额 3000 元。这类量称为**标量（scalar）**。

**第二类**：仅有数字不够，还必须知道方向才能完整描述的量。例如"向正北方向行驶 60 公里"和"向正南方向行驶 60 公里"是完全不同的两件事。又例如，风速 10 m/s 向东和风速 10 m/s 向西对船的影响截然相反。这类既有**大小**又有**方向**的量，就是**向量（vector）**。

向量的英文 vector 来源于拉丁语，原意是"携带者"——向量"携带"着方向信息，将你从一个位置"带到"另一个位置。

### 几何定义：有向线段

在几何上，向量被表示为一条**有向线段**，用带箭头的线段表示：

- 箭头的**起点**称为始点（tail）
- 箭头的**终点**称为终点（head）
- 线段的**长度**代表向量的大小（也称"模"）
- 箭头的**指向**代表向量的方向

一个关键约定：**位置不影响向量的身份**。也就是说，两条平行、等长、同向的有向线段，无论起点在哪里，它们代表的是同一个向量。向量关心的只是"位移了多少、往哪个方向"，而不在乎"从哪里出发"。

向量通常用以下符号表示：
- 粗体小写字母：$\mathbf{v}$、$\mathbf{u}$、$\mathbf{w}$（印刷体最常用）
- 带箭头的字母：$\vec{v}$（手写常用）
- 从 $A$ 到 $B$ 的有向线段：$\overrightarrow{AB}$

### 代数表示：坐标形式

几何表示直观，但不方便计算。建立坐标系后，我们可以用一组有序数字来精确表示向量。

在**二维平面**中，以原点为始点，到点 $(3, 4)$ 的向量写作：

$$\mathbf{v} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$$

含义是：沿 $x$ 轴方向移动 3 个单位，沿 $y$ 轴方向移动 4 个单位。

在**三维空间**中：

$$\mathbf{w} = \begin{pmatrix} 1 \\ -2 \\ 5 \end{pmatrix}$$

表示沿 $x$、$y$、$z$ 轴分别移动 $1$、$-2$、$5$ 个单位（负数表示沿负轴方向）。

这些分量有时也写成括号加逗号的形式：$\mathbf{v} = (3, 4)^T$，上标 $T$ 表示转置（见下文）。

### 行向量与列向量

同样一组数字，可以写成两种排列方式：

**列向量**（最常用的默认形式）：

$$\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$

**行向量**（列向量的转置）：

$$\mathbf{v}^T = \begin{pmatrix} v_1 & v_2 & \cdots & v_n \end{pmatrix}$$

两者之间通过**转置（transpose）**操作相互转换，记号为右上角的 $T$：将列向量"躺下"变成行向量，或将行向量"立起来"变成列向量。

**约定**：本教程中，除非特别说明，向量均指**列向量**。在深度学习框架（如 PyTorch、NumPy）中，一个形状为 `(n,)` 的一维数组可以视为行向量，而形状 `(n, 1)` 对应列向量。

### n 维向量与高维空间

向量最强大的地方在于可以推广到任意维度。**$n$ 维向量**是 $n$ 个有序实数组成的数组：

$$\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix} \in \mathbb{R}^n$$

记号 $\mathbb{R}^n$（读作"R-n"）表示**所有 $n$ 维实数向量组成的空间**，也叫 $n$ 维欧几里得空间。

- $\mathbb{R}^1$：数轴，每个点就是一个实数
- $\mathbb{R}^2$：平面，每个点由两个坐标确定
- $\mathbb{R}^3$：三维空间，日常生活所在的世界
- $\mathbb{R}^{784}$：一张 $28 \times 28$ 的灰度图像可以展平为 784 维向量
- $\mathbb{R}^{768}$：BERT 等语言模型的词向量维度

对于 $n > 3$ 的情况，我们无法直接"看到"这个空间，但所有代数运算规则与低维完全相同。这是线性代数的魅力所在：用统一的语言描述任意维度的空间。

---

## 1.2 向量的基本运算

向量运算是深度学习所有计算的基础。掌握这三种运算（加法、标量乘法、减法）及其几何意义，是理解后续内容的关键。

### 向量加法

**定义**：两个维度相同的向量 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$，它们的和定义为**对应分量相加**：

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 \\ u_2 \\ \vdots \\ u_n \end{pmatrix} + \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{pmatrix}$$

**注意**：只有维度相同的向量才能相加。$\mathbb{R}^2$ 中的向量不能和 $\mathbb{R}^3$ 中的向量相加，就像苹果不能和橘子相比。

**几何意义（首尾相接法则）**：将 $\mathbf{v}$ 的起点平移到 $\mathbf{u}$ 的终点处，则从 $\mathbf{u}$ 的起点到 $\mathbf{v}$ 的（新）终点的向量，就是 $\mathbf{u} + \mathbf{v}$。这等价于**平行四边形法则**：以 $\mathbf{u}$ 和 $\mathbf{v}$ 为两邻边构成平行四边形，对角线就是它们的和。

**具体例子**：

$$\begin{pmatrix} 1 \\ 2 \end{pmatrix} + \begin{pmatrix} 3 \\ -1 \end{pmatrix} = \begin{pmatrix} 4 \\ 1 \end{pmatrix}$$

直觉：先向右走 1 步、向上走 2 步，再向右走 3 步、向下走 1 步，最终等效于向右走 4 步、向上走 1 步。

### 标量乘法

**定义**：标量 $c \in \mathbb{R}$（一个普通数字）与向量 $\mathbf{v} \in \mathbb{R}^n$ 的乘积，是将向量每个分量都乘以 $c$：

$$c\mathbf{v} = c \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix} = \begin{pmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{pmatrix}$$

**几何意义（缩放与翻转）**：

| 标量 $c$ 的范围 | 效果 |
|:---|:---|
| $c > 1$ | 方向不变，向量被**拉长**（放大） |
| $0 < c < 1$ | 方向不变，向量被**缩短**（缩小） |
| $c = 1$ | 向量不变 |
| $c = 0$ | 向量变为**零向量** $\mathbf{0}$ |
| $c = -1$ | 方向**翻转**，长度不变 |
| $c < 0$ | 方向翻转，同时长度按 $|c|$ 缩放 |

**例**：$3 \times \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 3 \\ 6 \end{pmatrix}$，方向不变，长度变为原来的 3 倍。

### 向量减法

$\mathbf{u} - \mathbf{v}$ 等价于 $\mathbf{u} + (-1)\mathbf{v}$，即先将 $\mathbf{v}$ 取反再相加：

$$\mathbf{u} - \mathbf{v} = \begin{pmatrix} u_1 - v_1 \\ u_2 - v_2 \\ \vdots \\ u_n - v_n \end{pmatrix}$$

**几何意义**：当 $\mathbf{u}$ 和 $\mathbf{v}$ 共起点时，$\mathbf{u} - \mathbf{v}$ 是从 $\mathbf{v}$ 的终点指向 $\mathbf{u}$ 的终点的向量。

向量差在深度学习中频繁出现，例如损失函数中的**残差**（预测值与真实值的差）：$\mathbf{e} = \hat{\mathbf{y}} - \mathbf{y}$，其范数 $\|\mathbf{e}\|$ 反映了预测的误差大小。

### 运算性质

设 $\mathbf{u}$、$\mathbf{v}$、$\mathbf{w} \in \mathbb{R}^n$，$a, b \in \mathbb{R}$，向量运算满足以下性质：

| 性质名称 | 公式 |
|:---|:---|
| 加法交换律 | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ |
| 加法结合律 | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ |
| 零向量（加法单位元） | $\mathbf{v} + \mathbf{0} = \mathbf{v}$ |
| 加法逆元 | $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ |
| 标量乘法对向量加法的分配律 | $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$ |
| 标量乘法对标量加法的分配律 | $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$ |
| 标量乘法的结合律 | $(ab)\mathbf{v} = a(b\mathbf{v})$ |
| 单位标量 | $1 \cdot \mathbf{v} = \mathbf{v}$ |

这些性质看似显然，但它们构成了**向量空间（vector space）**的公理——任何满足这 8 条性质的集合，都可以用线性代数的整套工具来研究，无论那个集合的元素是数字、函数还是矩阵。

---

## 1.3 向量的几何意义

### 向量的长度：范数

向量 $\mathbf{v} = (v_1, v_2, \ldots, v_n)^T$ 的**长度**，在数学上称为**范数（norm）**。最常用的是**欧几里得范数**（也叫 $\ell_2$ 范数，或 2-范数），记作 $\|\mathbf{v}\|$ 或 $\|\mathbf{v}\|_2$：

$$\|\mathbf{v}\| = \|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\sum_{i=1}^n v_i^2}$$

这是勾股定理在高维空间中的直接推广。在二维平面中，向量 $(v_1, v_2)^T$ 对应一个直角三角形，两直角边长分别为 $|v_1|$ 和 $|v_2|$，斜边即为向量的长度 $\sqrt{v_1^2 + v_2^2}$。

**计算例子**：

- $\|(3, 4)^T\| = \sqrt{9 + 16} = \sqrt{25} = 5$（经典的勾股数）
- $\|(1, 1, 1)^T\| = \sqrt{1 + 1 + 1} = \sqrt{3} \approx 1.732$
- $\|(0, 0, \ldots, 0)^T\| = 0$（只有零向量的长度为 0）

**范数的基本性质**：

1. **非负性**：$\|\mathbf{v}\| \geq 0$，当且仅当 $\mathbf{v} = \mathbf{0}$ 时等号成立
2. **齐次性**：$\|c\mathbf{v}\| = |c| \cdot \|\mathbf{v}\|$（缩放向量，长度按相同比例缩放）
3. **三角不等式**：$\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$（两边之和不小于第三边）

除了 $\ell_2$ 范数，还有其他常用范数，例如 $\ell_1$ 范数（各分量绝对值之和）用于正则化，但本教程以 $\ell_2$ 为主。

### 单位向量与归一化

**单位向量（unit vector）**是长度恰好为 1 的向量。

将任意非零向量 $\mathbf{v}$ 除以其自身的长度，即可得到与它**同方向**的单位向量：

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

这个操作叫做**归一化（normalization）**。归一化后，向量的长度（幅度）信息被丢弃，只保留**方向**信息。

**验证**：$\|\hat{\mathbf{v}}\| = \left\|\frac{\mathbf{v}}{\|\mathbf{v}\|}\right\| = \frac{\|\mathbf{v}\|}{\|\mathbf{v}\|} = 1$ ✓

**例**：将 $\mathbf{v} = (3, 4)^T$ 归一化：

$$\hat{\mathbf{v}} = \frac{1}{5}\begin{pmatrix} 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 0.6 \\ 0.8 \end{pmatrix}$$

验证：$\|\hat{\mathbf{v}}\| = \sqrt{0.36 + 0.64} = \sqrt{1} = 1$ ✓

归一化在深度学习中极为常见，例如：计算余弦相似度之前需要归一化，Batch Normalization 层也涉及类似操作。

### 向量的方向

在二维空间中，向量 $(v_1, v_2)^T$ 的方向可以用它与 $x$ 轴正方向的夹角 $\theta$ 来刻画：

$$\cos\theta = \frac{v_1}{\|\mathbf{v}\|}, \quad \sin\theta = \frac{v_2}{\|\mathbf{v}\|}$$

即归一化后的两个分量分别是该角度的余弦和正弦。

在高维空间（$n \geq 3$）中，单一角度无法完整描述方向，但我们可以通过**两个向量之间的夹角**来比较方向的相似性，这由内积给出（见 1.4 节）。

---

## 1.4 内积（点积）

内积是向量运算中最核心、最实用的操作之一。它将两个向量"压缩"成一个标量，同时编码了两向量之间的几何关系。

### 内积的定义

两个相同维度的向量 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ 的**内积**（也称**点积**，dot product），记作 $\mathbf{u} \cdot \mathbf{v}$、$\langle \mathbf{u}, \mathbf{v} \rangle$ 或 $\mathbf{u}^T\mathbf{v}$，定义为**对应分量乘积之和**：

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n$$

**重点**：内积的结果是一个**标量**（数字），不是向量。

用矩阵记号表示，若 $\mathbf{u}$ 和 $\mathbf{v}$ 均为列向量，则：

$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v}$$

（将 $\mathbf{u}$ 转置为行向量后，与列向量 $\mathbf{v}$ 相乘，得到 $1 \times 1$ 的矩阵，即一个标量。）

**计算例子**：

$$\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix} \cdot \begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix} = 1 \times 4 + 2 \times 5 + 3 \times 6 = 4 + 10 + 18 = 32$$

再来一个：

$$\begin{pmatrix} 1 \\ 0 \end{pmatrix} \cdot \begin{pmatrix} 0 \\ 1 \end{pmatrix} = 1 \times 0 + 0 \times 1 = 0$$

这两个向量的内积为 0——我们很快会看到这意味着什么。

### 内积的性质

设 $\mathbf{u}$、$\mathbf{v}$、$\mathbf{w} \in \mathbb{R}^n$，$c \in \mathbb{R}$：

1. **交换律**：$\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$
2. **对加法的分配律**：$\mathbf{u} \cdot (\mathbf{v} + \mathbf{w}) = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \cdot \mathbf{w}$
3. **标量提取**：$(c\mathbf{u}) \cdot \mathbf{v} = c(\mathbf{u} \cdot \mathbf{v})$
4. **正定性**：$\mathbf{v} \cdot \mathbf{v} \geq 0$，等号当且仅当 $\mathbf{v} = \mathbf{0}$ 时成立

由性质 4，我们得到范数与内积的重要联系：

$$\mathbf{v} \cdot \mathbf{v} = \|\mathbf{v}\|^2 \quad \Longleftrightarrow \quad \|\mathbf{v}\| = \sqrt{\mathbf{v} \cdot \mathbf{v}}$$

这意味着**范数可以由内积来定义**——内积是更基本的概念。

### 内积与夹角的关系

内积最优美的特性是它与几何角度的联系：

$$\boxed{\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \cdot \|\mathbf{v}\| \cdot \cos\theta}$$

其中 $\theta \in [0, \pi]$ 是 $\mathbf{u}$ 和 $\mathbf{v}$ 之间的夹角。

由此可以导出**计算夹角的公式**：

$$\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \cdot \|\mathbf{v}\|}$$

这个公式在深度学习中化身为**余弦相似度（cosine similarity）**，用于衡量两个向量（两个物品、两段文本、两个用户）之间的相似程度。

**几何直觉**：内积衡量的是"两个向量在多大程度上指向同一方向"。

| 夹角 $\theta$ | $\cos\theta$ 的值 | 内积的符号 | 几何含义 |
|:---:|:---:|:---:|:---|
| $0°$ | $1$ | 正，最大 | 完全同向 |
| $0° < \theta < 90°$ | $(0, 1)$ | 正 | 大体同向 |
| $90°$ | $0$ | $0$ | 垂直（正交） |
| $90° < \theta < 180°$ | $(-1, 0)$ | 负 | 大体反向 |
| $180°$ | $-1$ | 负，最小 | 完全反向 |

**计算例子**：求 $\mathbf{u} = (1, 0)^T$ 和 $\mathbf{v} = (1, 1)^T$ 之间的夹角：

$$\cos\theta = \frac{1 \times 1 + 0 \times 1}{\sqrt{1} \times \sqrt{2}} = \frac{1}{\sqrt{2}} = \frac{\sqrt{2}}{2}$$

所以 $\theta = 45°$，与直觉一致（$(1,1)^T$ 恰好是 45° 方向）。

### 正交向量

若两向量 $\mathbf{u}$ 和 $\mathbf{v}$ 的内积为 0：

$$\mathbf{u} \cdot \mathbf{v} = 0$$

则称它们**正交（orthogonal）**，几何上即互相垂直（$\theta = 90°$）。正交用符号 $\mathbf{u} \perp \mathbf{v}$ 表示。

**为什么正交性重要？**

1. **信息独立性**：正交向量之间没有"重叠"的信息成分，彼此独立。就像 $x$ 轴和 $y$ 轴方向相互独立，沿 $x$ 轴的运动不影响 $y$ 坐标。

2. **坐标系基础**：标准基向量 $\mathbf{e}_1 = (1, 0, \ldots, 0)^T$，$\mathbf{e}_2 = (0, 1, \ldots, 0)^T$ 等，两两正交。正交基的好处是坐标计算极为简洁。

3. **深度学习应用**：PCA（主成分分析）要求主成分方向两两正交，确保各成分捕获独立的方差方向。

**例子**：在三维空间中，$(1, 0, 0)^T$、$(0, 1, 0)^T$、$(0, 0, 1)^T$ 两两正交，构成三维空间的标准正交基。

**注意**：零向量 $\mathbf{0}$ 与任何向量的内积都为 0，所以零向量与所有向量正交，但这是平凡情形，通常在讨论正交时默认向量非零。

---

## 1.5 叉积（向量积）

内积的结果是**标量**，叉积（Cross Product）的结果是**向量**，且仅在 $\mathbb{R}^3$ 中定义。

### 叉积的定义

设 $\mathbf{u} = (u_1, u_2, u_3)^T$，$\mathbf{v} = (v_1, v_2, v_3)^T$，则叉积定义为：

$$\mathbf{u} \times \mathbf{v} = \begin{vmatrix} \mathbf{e}_1 & \mathbf{e}_2 & \mathbf{e}_3 \\ u_1 & u_2 & u_3 \\ v_1 & v_2 & v_3 \end{vmatrix} = \begin{pmatrix} u_2 v_3 - u_3 v_2 \\ u_3 v_1 - u_1 v_3 \\ u_1 v_2 - u_2 v_1 \end{pmatrix}$$

### 叉积的几何意义

1. **方向**：$\mathbf{u} \times \mathbf{v}$ 同时垂直于 $\mathbf{u}$ 和 $\mathbf{v}$（右手法则确定方向）
2. **大小**：$\|\mathbf{u} \times \mathbf{v}\| = \|\mathbf{u}\| \|\mathbf{v}\| \sin\theta$，等于以 $\mathbf{u}, \mathbf{v}$ 为边的**平行四边形面积**

### 叉积的性质

| 性质 | 公式 |
|------|------|
| **反交换律** | $\mathbf{u} \times \mathbf{v} = -(\mathbf{v} \times \mathbf{u})$ |
| 分配律 | $\mathbf{u} \times (\mathbf{v} + \mathbf{w}) = \mathbf{u} \times \mathbf{v} + \mathbf{u} \times \mathbf{w}$ |
| 标量结合 | $(c\mathbf{u}) \times \mathbf{v} = c(\mathbf{u} \times \mathbf{v})$ |
| 自叉为零 | $\mathbf{u} \times \mathbf{u} = \mathbf{0}$ |

**注意**：叉积**不满足结合律**：$(\mathbf{u} \times \mathbf{v}) \times \mathbf{w} \neq \mathbf{u} \times (\mathbf{v} \times \mathbf{w})$。

### 混合积与体积

**标量三重积**：$\mathbf{u} \cdot (\mathbf{v} \times \mathbf{w})$ 的绝对值等于以 $\mathbf{u}, \mathbf{v}, \mathbf{w}$ 为棱的**平行六面体体积**：

$$\mathbf{u} \cdot (\mathbf{v} \times \mathbf{w}) = \begin{vmatrix} u_1 & u_2 & u_3 \\ v_1 & v_2 & v_3 \\ w_1 & w_2 & w_3 \end{vmatrix}$$

该值为零当且仅当三个向量共面（线性相关）。

**例 1.8** 求 $\mathbf{u} = (1, 2, 3)^T$ 与 $\mathbf{v} = (4, 5, 6)^T$ 的叉积。

$$\mathbf{u} \times \mathbf{v} = \begin{pmatrix} 2 \cdot 6 - 3 \cdot 5 \\ 3 \cdot 4 - 1 \cdot 6 \\ 1 \cdot 5 - 2 \cdot 4 \end{pmatrix} = \begin{pmatrix} -3 \\ 6 \\ -3 \end{pmatrix}$$

验证正交性：$\mathbf{u} \cdot (\mathbf{u} \times \mathbf{v}) = 1(-3) + 2(6) + 3(-3) = 0$ ✓

---

## 本章小结

本章从几何直觉出发，建立了向量的代数体系，并引入了内积这一核心工具。

| 概念 | 定义 | 关键公式 |
|:---|:---|:---|
| 向量 | 有序数组，兼具大小和方向 | $\mathbf{v} \in \mathbb{R}^n$ |
| 向量加法 | 对应分量相加 | $(u_i + v_i)_{i=1}^n$ |
| 标量乘法 | 每个分量乘以标量 | $(cv_i)_{i=1}^n$ |
| 欧几里得范数 | 各分量平方和的算术平方根 | $\|\mathbf{v}\| = \sqrt{\sum_i v_i^2}$ |
| 归一化 | 除以自身长度得到单位向量 | $\hat{\mathbf{v}} = \mathbf{v}/\|\mathbf{v}\|$ |
| 内积（点积） | 对应分量乘积之和，结果为标量 | $\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i$ |
| 夹角公式 | 由内积和范数联合确定 | $\cos\theta = \dfrac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ |
| 正交 | 内积为 0，方向垂直 | $\mathbf{u} \cdot \mathbf{v} = 0$ |

**核心思路**：向量将方向和大小统一编码为一组有序数字。内积是连接**代数**（数字运算）与**几何**（角度、长度）的桥梁，是线性代数中最重要的工具之一。掌握这一桥梁，你将能在高维空间中"看到"数据的形状与结构。

**下一章预告**：矩阵——向量的集合，以及线性变换的语言。

---

## 深度学习应用

### 概念回顾

向量是线性代数的基本对象，在深度学习中无处不在。每一条训练样本、每一个模型参数、每一层的输出激活，本质上都是向量（或向量的批次——矩阵/张量）。理解向量运算，是理解神经网络计算的第一步。

### 在深度学习中的应用

**1. 词向量（Word Embeddings）**

Word2Vec、GloVe 等模型将词汇表中的每个单词映射为一个稠密的实数向量（如 300 维）。相似语义的词在向量空间中距离更近（内积更大，夹角更小）。最著名的例子：

$$\text{vec}(\text{"king"}) - \text{vec}(\text{"man"}) + \text{vec}(\text{"woman"}) \approx \text{vec}(\text{"queen"})$$

这说明词向量不仅捕捉了词义，还编码了词语之间的**关系**（如"性别"这一维度）。词向量之间的相似度通常用**余弦相似度**度量，这正是内积与夹角公式的直接应用：

$$\text{cosine\_similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \cdot \|\mathbf{v}\|} = \cos\theta$$

**2. 特征向量表示**

几乎所有机器学习模型的输入都是向量：

- 图像：$224 \times 224 \times 3$ 的 RGB 图像展平为 $150528$ 维向量（或保留其张量结构）
- 文本：经过分词和编码后表示为向量序列
- 用户画像：年龄、消费水平、历史行为等特征拼接成特征向量
- 推荐系统：用户和商品都被映射为同一向量空间中的向量，内积衡量匹配程度

**3. Embedding 层与向量查找**

`nn.Embedding` 层本质上是一个**查找表（lookup table）**：给定离散 ID（单词 ID、用户 ID、商品 ID），返回对应的实数向量。这些向量随模型训练而更新，使得语义相关的 ID 对应相近的向量。

**4. 全连接层的计算**

神经网络中全连接层的核心计算是：

$$\mathbf{y} = W\mathbf{x} + \mathbf{b}$$

其中 $W$ 是权重矩阵，$\mathbf{x}$ 是输入向量，$\mathbf{b}$ 是偏置向量。$W$ 的每一行都是一个权重向量，与输入 $\mathbf{x}$ 做内积，得到输出的一个分量。整个全连接层的计算，本质上是**一批内积运算**。

### 代码示例（Python/PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. 基本向量运算
# ============================================================
u = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([4.0, 5.0, 6.0])

# 向量加法：对应分量相加
print("u + v =", u + v)            # tensor([5., 7., 9.])

# 标量乘法：每个分量乘以标量
print("2 * u =", 2 * u)            # tensor([2., 4., 6.])

# 向量减法
print("u - v =", u - v)            # tensor([-3., -3., -3.])

# 内积（点积）
dot = torch.dot(u, v)
print("u · v =", dot.item())       # 32.0

# 欧几里得范数（L2 范数）
norm_u = torch.norm(u)             # 等价于 torch.linalg.norm(u)
print("||u|| =", norm_u.item())    # 3.7417...

# ============================================================
# 2. 归一化与余弦相似度
# ============================================================
# 手动归一化
u_hat = u / torch.norm(u)
print("归一化 u:", u_hat)
print("||u_hat|| =", torch.norm(u_hat).item())  # 1.0

# 使用 F.normalize（支持批次操作，更常用）
u_norm = F.normalize(u.unsqueeze(0), p=2, dim=1).squeeze(0)

# 余弦相似度：两种等价写法
cos_sim_manual = torch.dot(u, v) / (torch.norm(u) * torch.norm(v))
cos_sim_builtin = F.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0))
print("余弦相似度（手动）:", cos_sim_manual.item())   # 0.9746...
print("余弦相似度（内置）:", cos_sim_builtin.item())  # 0.9746...

# ============================================================
# 3. 验证正交性
# ============================================================
e1 = torch.tensor([1.0, 0.0, 0.0])
e2 = torch.tensor([0.0, 1.0, 0.0])
e3 = torch.tensor([0.0, 0.0, 1.0])

print("e1 · e2 =", torch.dot(e1, e2).item())  # 0.0（正交）
print("e1 · e3 =", torch.dot(e1, e3).item())  # 0.0（正交）
print("e1 · e1 =", torch.dot(e1, e1).item())  # 1.0（单位向量）

# ============================================================
# 4. Embedding 层：将离散 ID 映射为向量
# ============================================================
vocab_size = 10000   # 词汇表大小（例如 10000 个词）
embed_dim  = 128     # 向量维度

# 创建可学习的 Embedding 层
embedding = nn.Embedding(vocab_size, embed_dim)

# 将单词 ID 序列转换为向量
word_ids  = torch.tensor([42, 100, 256])   # 三个单词的 ID
word_vecs = embedding(word_ids)
print("词向量形状:", word_vecs.shape)       # torch.Size([3, 128])

# 计算 "42" 和 "100" 两个词的余弦相似度
v1, v2 = word_vecs[0], word_vecs[1]
similarity = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
print("词语相似度:", similarity.item())     # 初始随机，训练后有意义

# ============================================================
# 5. 全连接层中的内积：理解神经元的计算
# ============================================================
# 单个线性神经元：y = w · x + b
w = torch.tensor([1.0, 2.0, -1.0])   # 权重向量
b = 0.5                               # 偏置（标量）
x = torch.tensor([1.0, 1.0, 1.0])    # 输入向量

y = torch.dot(w, x) + b
print("神经元输出:", y.item())         # 1*1 + 2*1 + (-1)*1 + 0.5 = 2.5
```

**运行说明**：执行 `pip install torch` 安装 PyTorch 后即可运行。代码展示了向量基本运算、归一化、余弦相似度、正交验证、Embedding 层用法，以及神经元的内积计算。

### 延伸阅读

- **Word2Vec 原始论文**：Mikolov et al., ["Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/abs/1301.3781)（2013）——词向量的开创性工作
- **GloVe 论文**：Pennington et al., ["GloVe: Global Vectors for Word Representation"](https://nlp.stanford.edu/projects/glove/)（2014）——基于全局共现统计的词向量
- **3Blue1Brown 线性代数系列**：[Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)——通过精美动画直观理解向量和线性变换，强烈推荐
- **PyTorch 文档**：[torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)——Embedding 层的官方文档与使用示例

---

## 练习题

**练习 1**（基础——向量运算）

设 $\mathbf{u} = (2, -1, 3)^T$，$\mathbf{v} = (-1, 4, 2)^T$，计算：

（a）$\mathbf{u} + \mathbf{v}$

（b）$3\mathbf{u} - 2\mathbf{v}$

（c）$\|\mathbf{u}\|$

---

**练习 2**（基础——夹角计算）

计算向量 $\mathbf{a} = (1, 1)^T$ 和 $\mathbf{b} = (1, 0)^T$ 之间的夹角（以度为单位）。

---

**练习 3**（中等——代数恒等式）

证明以下等式对任意向量 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ 均成立：

$$\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + 2(\mathbf{u} \cdot \mathbf{v}) + \|\mathbf{v}\|^2$$

并说明：当 $\mathbf{u} \perp \mathbf{v}$（正交）时，该等式退化为哪个熟知的定理？

---

**练习 4**（中等——正交单位向量）

在二维空间中，找到所有与向量 $\mathbf{v} = (3, 4)^T$ 正交的单位向量，并给出完整推导过程。

---

**练习 5**（进阶——神经元与内积）

设 $\mathbf{x} \in \mathbb{R}^n$，$\mathbf{w} \in \mathbb{R}^n$，$b \in \mathbb{R}$。神经网络中一个最简单的线性神经元计算：

$$y = \mathbf{w} \cdot \mathbf{x} + b$$

其中 $\mathbf{w}$ 是权重向量，$b$ 是偏置，$\mathbf{x}$ 是输入特征向量。

（a）基于内积与夹角的关系，解释当 $\|\mathbf{x}\|$ 固定时，输入 $\mathbf{x}$ 与权重 $\mathbf{w}$ 方向越接近，输出 $y$ 如何变化？这说明权重向量 $\mathbf{w}$ 代表什么含义？

（b）设 $\mathbf{w} = (1, 2, -1)^T$，$b = 0.5$，分别计算输入 $\mathbf{x}_1 = (1, 1, 1)^T$ 和 $\mathbf{x}_2 = (-1, 0, 2)^T$ 对应的输出 $y_1$ 和 $y_2$。

（c）若在 $y$ 后接 sigmoid 激活函数 $\sigma(y) = \frac{1}{1+e^{-y}}$，分别计算 $\sigma(y_1)$ 和 $\sigma(y_2)$（保留 4 位小数）。$\sigma(y)$ 的值域是什么？

---

## 练习答案

<details>
<summary>点击展开 练习1 答案</summary>

**（a）向量加法，对应分量相加：**

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} 2+(-1) \\ -1+4 \\ 3+2 \end{pmatrix} = \begin{pmatrix} 1 \\ 3 \\ 5 \end{pmatrix}$$

**（b）先做标量乘法，再做减法：**

$$3\mathbf{u} = \begin{pmatrix} 6 \\ -3 \\ 9 \end{pmatrix}, \quad 2\mathbf{v} = \begin{pmatrix} -2 \\ 8 \\ 4 \end{pmatrix}$$

$$3\mathbf{u} - 2\mathbf{v} = \begin{pmatrix} 6-(-2) \\ -3-8 \\ 9-4 \end{pmatrix} = \begin{pmatrix} 8 \\ -11 \\ 5 \end{pmatrix}$$

**（c）欧几里得范数：**

$$\|\mathbf{u}\| = \sqrt{2^2 + (-1)^2 + 3^2} = \sqrt{4 + 1 + 9} = \sqrt{14} \approx 3.742$$

</details>

<details>
<summary>点击展开 练习2 答案</summary>

**第一步：计算内积**

$$\mathbf{a} \cdot \mathbf{b} = 1 \times 1 + 1 \times 0 = 1$$

**第二步：计算范数**

$$\|\mathbf{a}\| = \sqrt{1^2 + 1^2} = \sqrt{2}, \quad \|\mathbf{b}\| = \sqrt{1^2 + 0^2} = 1$$

**第三步：代入夹角公式**

$$\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|} = \frac{1}{\sqrt{2} \times 1} = \frac{\sqrt{2}}{2}$$

因此：

$$\theta = \arccos\!\left(\frac{\sqrt{2}}{2}\right) = 45°$$

**几何验证**：向量 $(1, 1)^T$ 位于 $x$ 轴和 $y$ 轴的正角平分线上，与 $x$ 轴（即 $(1, 0)^T$ 方向）的夹角确实是 45°。

</details>

<details>
<summary>点击展开 练习3 答案</summary>

**证明过程：**

利用 $\|\mathbf{x}\|^2 = \mathbf{x} \cdot \mathbf{x}$，将左边展开：

$$\|\mathbf{u} + \mathbf{v}\|^2 = (\mathbf{u} + \mathbf{v}) \cdot (\mathbf{u} + \mathbf{v})$$

利用内积对加法的分配律：

$$= \mathbf{u} \cdot (\mathbf{u} + \mathbf{v}) + \mathbf{v} \cdot (\mathbf{u} + \mathbf{v})$$

$$= \mathbf{u} \cdot \mathbf{u} + \mathbf{u} \cdot \mathbf{v} + \mathbf{v} \cdot \mathbf{u} + \mathbf{v} \cdot \mathbf{v}$$

利用内积的交换律 $\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$，以及 $\mathbf{x} \cdot \mathbf{x} = \|\mathbf{x}\|^2$：

$$= \|\mathbf{u}\|^2 + 2(\mathbf{u} \cdot \mathbf{v}) + \|\mathbf{v}\|^2 \quad \square$$

**当 $\mathbf{u} \perp \mathbf{v}$ 时**，$\mathbf{u} \cdot \mathbf{v} = 0$，等式退化为：

$$\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$$

这正是**勾股定理（Pythagorean Theorem）**的向量形式！$\mathbf{u}$ 和 $\mathbf{v}$ 是两条互相垂直的直角边，$\mathbf{u} + \mathbf{v}$ 是斜边，斜边长度的平方等于两直角边长度平方之和。

</details>

<details>
<summary>点击展开 练习4 答案</summary>

设所求向量为 $\mathbf{u} = (a, b)^T$，需同时满足两个条件：

**条件一（正交）**：$\mathbf{v} \cdot \mathbf{u} = 0$

$$3a + 4b = 0 \implies a = -\frac{4b}{3}$$

**条件二（单位向量）**：$\|\mathbf{u}\| = 1$

$$a^2 + b^2 = 1$$

将条件一代入条件二：

$$\left(-\frac{4b}{3}\right)^2 + b^2 = 1 \implies \frac{16b^2}{9} + b^2 = 1 \implies \frac{25b^2}{9} = 1 \implies b = \pm\frac{3}{5}$$

对应地：
- $b = \dfrac{3}{5}$：$a = -\dfrac{4}{5}$，得 $\mathbf{u}_1 = \left(-\dfrac{4}{5},\ \dfrac{3}{5}\right)^T$
- $b = -\dfrac{3}{5}$：$a = \dfrac{4}{5}$，得 $\mathbf{u}_2 = \left(\dfrac{4}{5},\ -\dfrac{3}{5}\right)^T$

**验证** $\mathbf{u}_1$：

$$\mathbf{v} \cdot \mathbf{u}_1 = 3 \times \left(-\frac{4}{5}\right) + 4 \times \frac{3}{5} = -\frac{12}{5} + \frac{12}{5} = 0 \checkmark$$

$$\|\mathbf{u}_1\| = \sqrt{\frac{16}{25} + \frac{9}{25}} = \sqrt{1} = 1 \checkmark$$

注意 $\mathbf{u}_2 = -\mathbf{u}_1$，两者方向相反。在二维空间中，与给定向量正交的单位向量**恰好有两个**，它们互为相反向量。

</details>

<details>
<summary>点击展开 练习5 答案</summary>

**（a）几何意义**

由内积与夹角的关系：

$$\mathbf{w} \cdot \mathbf{x} = \|\mathbf{w}\| \cdot \|\mathbf{x}\| \cdot \cos\theta$$

当 $\|\mathbf{x}\|$ 固定时，$\mathbf{w} \cdot \mathbf{x}$ 的大小取决于 $\cos\theta$：$\mathbf{x}$ 与 $\mathbf{w}$ 方向越接近（$\theta$ 越小），$\cos\theta$ 越接近 1，内积越大，输出 $y$ 越大。

权重向量 $\mathbf{w}$ 定义了该神经元"最敏感"的方向：当输入**完全沿 $\mathbf{w}$ 方向**时，激活最强；当输入**垂直于 $\mathbf{w}$** 时，内积为 0，神经元几乎不响应。本质上，$\mathbf{w}$ 代表了该神经元"在寻找什么特征方向"。

**（b）计算线性输出**

$y_1$（输入 $\mathbf{x}_1 = (1, 1, 1)^T$）：

$$y_1 = \mathbf{w} \cdot \mathbf{x}_1 + b = (1 \times 1 + 2 \times 1 + (-1) \times 1) + 0.5 = 2 + 0.5 = 2.5$$

$y_2$（输入 $\mathbf{x}_2 = (-1, 0, 2)^T$）：

$$y_2 = \mathbf{w} \cdot \mathbf{x}_2 + b = (1 \times (-1) + 2 \times 0 + (-1) \times 2) + 0.5 = -3 + 0.5 = -2.5$$

**（c）接 sigmoid 激活**

sigmoid 函数：$\sigma(y) = \dfrac{1}{1 + e^{-y}}$，值域为 $(0, 1)$。

$$\sigma(y_1) = \frac{1}{1 + e^{-2.5}} = \frac{1}{1 + 0.0821} \approx \frac{1}{1.0821} \approx 0.9241$$

$$\sigma(y_2) = \frac{1}{1 + e^{2.5}} = \frac{1}{1 + 12.182} \approx \frac{1}{13.182} \approx 0.0759$$

$\sigma(y_1) \approx 0.924$，接近 1，表示该神经元对输入 $\mathbf{x}_1$ 强烈"激活"（输出高置信度）。

$\sigma(y_2) \approx 0.076$，接近 0，表示该神经元对输入 $\mathbf{x}_2$ 几乎"不激活"（输出低置信度）。

在二分类任务中，这对应着 $\mathbf{x}_1$ 和 $\mathbf{x}_2$ 被分配到不同类别的结论。sigmoid 将任意实数压缩到 $(0, 1)$，可以解释为"概率"。

</details>
