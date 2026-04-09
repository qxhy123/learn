# 第9章：向量空间

## 学习目标

学完本章后，你应当能够：

- 理解向量空间的抽象定义，以及为何需要这种抽象
- 掌握向量空间必须满足的八条公理，并理解每条公理的含义
- 能够逐步验证一个给定集合在给定运算下是否构成向量空间
- 认识 $\mathbb{R}^n$、矩阵空间、多项式空间、函数空间等常见向量空间
- 理解向量空间概念在深度学习中的核心应用，特别是潜在空间（Latent Space）

---

## 9.1 向量空间的定义

### 9.1.1 动机：从具体到抽象

在前面的章节中，我们已经深入学习了 $\mathbb{R}^n$ 中的向量——有大小、有方向的箭头。我们学会了向量加法、数乘，以及由此衍生出来的线性组合、线性相关等概念。

但仔细想想，这些运算的本质是什么？向量加法满足交换律、结合律；数乘满足分配律；存在零向量作为加法单位元；每个向量都有加法逆元。这些性质其实并不依赖于"箭头"这个具体形象。

考虑以下三个看似不同的对象：

1. **$\mathbb{R}^2$ 中的向量**：$\begin{pmatrix} 1 \\ 2 \end{pmatrix} + \begin{pmatrix} 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 4 \\ 6 \end{pmatrix}$

2. **实数多项式**：$(1 + 2x) + (3 + 4x) = 4 + 6x$

3. **连续函数**：$f(x) = \sin x$，$g(x) = \cos x$，$(f + g)(x) = \sin x + \cos x$

这三类对象的"加法"和"数乘"运算满足完全相同的结构性质。数学家意识到：与其对每类对象分别证明定理，不如抽象出共同结构，一次性证明所有结论。

**向量空间**就是这种抽象的产物——它不问"向量是什么形状"，只问"它们的运算满足什么规则"。

### 9.1.2 向量空间的八条公理

**定义**：设 $V$ 是一个非空集合，$\mathbb{F}$ 是一个数域（通常取 $\mathbb{R}$ 或 $\mathbb{C}$）。若在 $V$ 上定义了两种运算：

- **向量加法**：$V \times V \to V$，将 $\mathbf{u}, \mathbf{v} \in V$ 映射到 $\mathbf{u} + \mathbf{v} \in V$
- **标量乘法**：$\mathbb{F} \times V \to V$，将 $c \in \mathbb{F}, \mathbf{v} \in V$ 映射到 $c\mathbf{v} \in V$

并且这两种运算满足以下**八条公理**，则称 $(V, +, \cdot)$ 为 $\mathbb{F}$ 上的**向量空间**（vector space），$V$ 的元素称为**向量**。

**加法公理（4条）**

| 编号 | 名称 | 内容 |
|------|------|------|
| A1 | 加法封闭性 | 对所有 $\mathbf{u}, \mathbf{v} \in V$，有 $\mathbf{u} + \mathbf{v} \in V$ |
| A2 | 加法交换律 | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ |
| A3 | 加法结合律 | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ |
| A4 | 零向量存在 | 存在 $\mathbf{0} \in V$，使得 $\mathbf{v} + \mathbf{0} = \mathbf{v}$ 对所有 $\mathbf{v} \in V$ 成立 |
| A5 | 加法逆元存在 | 对每个 $\mathbf{v} \in V$，存在 $-\mathbf{v} \in V$，使得 $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ |

**标量乘法公理（4条）**

| 编号 | 名称 | 内容 |
|------|------|------|
| S1 | 数乘封闭性 | 对所有 $c \in \mathbb{F}, \mathbf{v} \in V$，有 $c\mathbf{v} \in V$ |
| S2 | 数乘结合律 | $(cd)\mathbf{v} = c(d\mathbf{v})$，其中 $c, d \in \mathbb{F}$ |
| S3 | 单位元 | $1 \cdot \mathbf{v} = \mathbf{v}$，其中 $1$ 是数域中的乘法单位元 |
| S4a | 向量分配律 | $c(\mathbf{u} + \mathbf{v}) = c\mathbf{u} + c\mathbf{v}$ |
| S4b | 标量分配律 | $(c + d)\mathbf{v} = c\mathbf{v} + d\mathbf{v}$ |

> **注意**：有些教材将 A1 和 S1（封闭性）隐含在"运算定义"中，只列出六条公理。本书遵循八条公理的完整表述，以便验证时不遗漏封闭性检查。

### 9.1.3 定义的本质

八条公理看起来很多，但可以从三个层次理解其本质：

1. **封闭性（A1, S1）**：运算不会"逃出"集合。这是基础保障。

2. **加法的群结构（A2–A5）**：$V$ 关于加法构成一个**交换群**（阿贝尔群）。这保证了加法运算的代数一致性。

3. **数乘的相容性（S2–S4）**：标量乘法与加法、以及数域内的乘法相容。这将代数结构从集合层面提升到线性结构层面。

理解了这三个层次，验证向量空间时就有了清晰的思路。

---

## 9.2 向量空间的例子

### 9.2.1 $\mathbb{R}^n$ 空间

最经典的向量空间。$\mathbb{R}^n$ 是所有 $n$ 维实数列向量的集合：

$$\mathbb{R}^n = \left\{ \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix} \,\middle|\, x_i \in \mathbb{R} \right\}$$

加法和数乘按分量定义：

$$\begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix} + \begin{pmatrix} y_1 \\ \vdots \\ y_n \end{pmatrix} = \begin{pmatrix} x_1 + y_1 \\ \vdots \\ x_n + y_n \end{pmatrix}, \qquad c\begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix} = \begin{pmatrix} cx_1 \\ \vdots \\ cx_n \end{pmatrix}$$

零向量是 $\mathbf{0} = (0, 0, \ldots, 0)^T$。所有八条公理均可由实数的性质直接验证。

### 9.2.2 矩阵空间

所有 $m \times n$ 实矩阵的集合 $\mathbb{R}^{m \times n}$，在矩阵加法和数乘矩阵运算下构成向量空间。

$$A + B = [a_{ij} + b_{ij}], \qquad cA = [ca_{ij}]$$

零向量是全零矩阵 $O$。每个矩阵 $A$ 的加法逆元是 $-A = [-a_{ij}]$。

这个例子告诉我们：**向量空间中的"向量"不必是箭头**，矩阵也可以是"向量"。

### 9.2.3 多项式空间

次数不超过 $n$ 的实系数多项式的集合，记作 $\mathcal{P}_n$：

$$\mathcal{P}_n = \{a_0 + a_1 x + a_2 x^2 + \cdots + a_n x^n \mid a_i \in \mathbb{R}\}$$

按通常的多项式加法和数乘定义运算：

$$(a_0 + a_1 x + \cdots) + (b_0 + b_1 x + \cdots) = (a_0 + b_0) + (a_1 + b_1)x + \cdots$$

零向量是零多项式 $p(x) = 0$（所有系数为零）。可验证 $\mathcal{P}_n$ 满足所有八条公理。

类似地，所有实系数多项式（不限次数）的集合 $\mathcal{P}$ 也构成向量空间。

### 9.2.4 函数空间

设 $C[a, b]$ 是闭区间 $[a, b]$ 上所有连续函数的集合。定义逐点加法和数乘：

$$(f + g)(x) = f(x) + g(x), \qquad (cf)(x) = c \cdot f(x)$$

零向量是零函数 $\mathbf{0}(x) = 0$。每个连续函数之和仍连续（由连续函数的性质保证），每个连续函数的常数倍仍连续，因此运算封闭。其他公理由实数的性质逐点验证。$C[a, b]$ 是一个**无穷维**向量空间。

函数空间在分析学、信号处理和机器学习中极为重要——神经网络本质上是在近似某个函数空间中的元素。

---

## 9.3 验证向量空间

### 9.3.1 验证步骤

验证一个集合 $V$ 在给定运算下是否为向量空间，需要**系统地检查所有八条公理**。推荐的步骤如下：

**步骤 1：明确定义。** 写清楚集合 $V$ 是什么，加法和数乘如何定义。

**步骤 2：检查封闭性（A1, S1）。** 取任意 $\mathbf{u}, \mathbf{v} \in V$ 和 $c \in \mathbb{F}$，验证 $\mathbf{u} + \mathbf{v} \in V$ 和 $c\mathbf{v} \in V$。**这是最容易遗漏的步骤。**

**步骤 3：验证加法群公理（A2–A5）。** 特别注意找出零向量的具体形式，以及每个元素的加法逆元。

**步骤 4：验证数乘公理（S2–S4）。** 通常这些由数域自身的性质保证，但仍需逐一说明。

**示例**：验证 $W = \{(x, y) \in \mathbb{R}^2 \mid y = 2x\}$ 在通常的向量加法和数乘下是向量空间。

- **封闭性**：设 $\mathbf{u} = (a, 2a)$，$\mathbf{v} = (b, 2b)$，则 $\mathbf{u} + \mathbf{v} = (a+b, 2a+2b) = (a+b, 2(a+b)) \in W$。设 $c \in \mathbb{R}$，则 $c\mathbf{u} = (ca, 2ca) \in W$。封闭。
- **零向量**：$(0, 0) = (0, 2 \cdot 0) \in W$。
- **加法逆元**：若 $(a, 2a) \in W$，则 $(-a, -2a) = (-a, 2(-a)) \in W$。
- **其余公理**：继承自 $\mathbb{R}^2$，直接满足。

结论：$W$ 是向量空间（实际上是 $\mathbb{R}^2$ 的一个子空间）。

### 9.3.2 非向量空间的反例

验证不是向量空间时，只需**找到一条公理不满足**即可。

**反例 1：正实数 $\mathbb{R}^+$，加法为通常乘法**

定义 $u \oplus v = uv$（乘法作为"加法"），$c \odot v = v^c$（幂运算作为"数乘"）。

验证：零向量应满足 $v \oplus \mathbf{0} = v$，即 $v \cdot \mathbf{0} = v$，故零向量为 $1$。加法逆元：$v \oplus (-v) = 1$，即 $v \cdot (-v) = 1$，故 $-v = 1/v \in \mathbb{R}^+$。单位元：$1 \odot v = v^1 = v$。分配律也可验证。事实上，$(\mathbb{R}^+, \oplus, \odot)$ **是**一个向量空间（通过对数同构于 $(\mathbb{R}, +, \cdot)$）。

**反例 2：$\mathbb{R}^2$ 中的单位圆**

$S = \{(x, y) \mid x^2 + y^2 = 1\}$，使用通常的向量加法。

取 $(1, 0), (0, 1) \in S$，则 $(1, 0) + (0, 1) = (1, 1)$，但 $1^2 + 1^2 = 2 \neq 1$，故 $(1, 1) \notin S$。

**加法封闭性（A1）不满足**，$S$ 不是向量空间。

**反例 3：$\mathbb{R}^2$ 中第一象限**

$Q = \{(x, y) \mid x \geq 0, y \geq 0\}$，使用通常加法和数乘。

取 $(1, 1) \in Q$ 和 $c = -1$，则 $(-1)(1, 1) = (-1, -1) \notin Q$。

**数乘封闭性（S1）不满足**，$Q$ 不是向量空间。

---

## 本章小结

本章从具体的 $\mathbb{R}^n$ 出发，抽象出向量空间的一般定义。核心要点如下：

1. **向量空间是一种抽象结构**，由集合 $V$、数域 $\mathbb{F}$、加法运算和数乘运算共同组成，必须满足八条公理。

2. **八条公理分两组**：加法构成交换群（5条），数乘与加法相容（4条，其中封闭性与加法封闭性并列列出）。实践中，A1 和 S1（封闭性）是最常需要单独验证的。

3. **向量空间实例多样**：$\mathbb{R}^n$、矩阵空间、多项式空间、函数空间——它们在形式上截然不同，却共享同一代数结构，使得线性代数的定理对所有这些对象一律适用。

4. **验证策略**：逐条检查八条公理；若发现某条不满足，立即给出反例，无需继续验证。

5. **子空间**（下一章的主题）是向量空间中的向量空间，理解向量空间的定义是学习子空间的先决条件。

---

## 深度学习应用

### 特征空间的概念

在机器学习中，数据往往被表示为高维向量。例如，一张 $28 \times 28$ 的灰度图像可以展开为 $\mathbb{R}^{784}$ 中的一个向量。所有可能的输入数据构成一个**特征空间**——从向量空间的视角看，这就是输入数据所在的向量空间（通常取其一个子集）。

神经网络的每一层都在对输入向量做线性变换（矩阵乘法）加非线性激活，本质上是在不同向量空间之间"流动"数据表示。线性变换保留了向量空间结构，而深度网络通过组合多次这样的变换，逐层构建出对任务有用的表示。

### 潜在空间（Latent Space）

**潜在空间**是深度学习中最重要的向量空间概念之一。自编码器（Autoencoder）将高维输入 $\mathbf{x} \in \mathbb{R}^n$ 编码为低维表示 $\mathbf{z} \in \mathbb{R}^k$（$k \ll n$），$\mathbf{z}$ 所在的 $\mathbb{R}^k$ 就是潜在空间。

潜在空间的关键特性：数据的语义结构被映射为几何结构。例如，在人脸生成模型中，潜在空间中"微笑方向"的向量可以被找到，沿该方向移动 $\mathbf{z}$ 就能连续改变生成人脸的笑容程度。这种"语义算术"正是向量空间线性结构的体现。

### VAE 和 GAN 中的潜在空间

**变分自编码器（VAE）** 将潜在空间赋予概率结构：编码器输出潜在向量的均值 $\boldsymbol{\mu}$ 和方差 $\boldsymbol{\sigma}^2$，解码器从该分布中采样 $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ 并重构输入。VAE 的损失函数包含一项 KL 散度正则化，鼓励潜在空间接近标准正态分布 $\mathcal{N}(\mathbf{0}, I)$，从而使潜在空间具有良好的插值性质。

**生成对抗网络（GAN）** 的生成器从随机噪声 $\mathbf{z} \in \mathbb{R}^k$ 生成数据。这个随机噪声向量所在的空间就是 GAN 的潜在空间。GAN 的训练目标是让生成器将标准正态分布映射到数据分布，而这个映射的研究依赖于对潜在空间几何结构的深入理解。

### 代码示例

下面的 Python 代码展示了 VAE 潜在空间的核心结构，以及如何在潜在空间中进行线性插值（体现向量空间的加法和数乘）：

```python
import numpy as np
import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    """将输入映射到潜在空间的编码器"""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """重参数化技巧：从 N(mu, sigma^2) 中采样"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)          # eps ~ N(0, I)
    return mu + eps * std                # 潜在向量 z = mu + eps * sigma


def latent_interpolation(
    z1: np.ndarray,
    z2: np.ndarray,
    steps: int = 10
) -> list[np.ndarray]:
    """
    在潜在空间中对两个向量进行线性插值。
    体现向量空间的数乘和加法：
        z(t) = (1 - t) * z1 + t * z2,  t in [0, 1]
    """
    interpolated = []
    for t in np.linspace(0, 1, steps):
        z_t = (1 - t) * z1 + t * z2    # 向量空间的线性组合
        interpolated.append(z_t)
    return interpolated


# 演示：在潜在空间中做"语义算术"
# 类比 Word2Vec 中著名的：king - man + woman ≈ queen
def semantic_arithmetic(
    z_king: np.ndarray,
    z_man: np.ndarray,
    z_woman: np.ndarray
) -> np.ndarray:
    """潜在空间中的向量加减法（向量空间线性结构的直接应用）"""
    return z_king - z_man + z_woman      # 向量加法和数乘（乘以 -1）


# 验证插值路径
if __name__ == "__main__":
    latent_dim = 8
    z1 = np.random.randn(latent_dim)    # 潜在空间中的"点A"
    z2 = np.random.randn(latent_dim)    # 潜在空间中的"点B"

    path = latent_interpolation(z1, z2, steps=5)
    print("潜在空间线性插值路径（5步）：")
    for i, z in enumerate(path):
        t = i / 4
        print(f"  t={t:.2f}: z = {np.round(z, 3)}")

    # 验证端点
    assert np.allclose(path[0], z1), "t=0 时应等于 z1"
    assert np.allclose(path[-1], z2), "t=1 时应等于 z2"
    print("\n插值端点验证通过。")
```

代码中 `latent_interpolation` 函数的核心操作 `(1 - t) * z1 + t * z2` 正是向量空间中的线性组合：标量 $(1-t)$ 和 $t$ 分别与向量 `z1`、`z2` 做数乘，再做向量加法。这在数学上成立，正是因为潜在空间 $\mathbb{R}^k$ 是一个向量空间。

---

## 练习题

**练习 9.1**（公理验证）

设 $V = \{(x, y) \in \mathbb{R}^2 \mid 3x - y = 0\}$，使用 $\mathbb{R}^2$ 中通常的向量加法和数乘。请验证 $V$ 是否构成 $\mathbb{R}$ 上的向量空间。

---

**练习 9.2**（识别反例）

对以下两个集合，判断在通常的加法和数乘下是否构成向量空间。若不是，指出违反了哪条公理并给出具体反例。

(a) $A = \{(x, y) \in \mathbb{R}^2 \mid x^2 + y^2 \leq 1\}$（单位闭圆盘）

(b) $B = \{(x, y) \in \mathbb{R}^2 \mid x + y = 1\}$

---

**练习 9.3**（多项式空间）

设 $\mathcal{P}_2 = \{a + bx + cx^2 \mid a, b, c \in \mathbb{R}\}$ 是次数不超过 2 的实系数多项式集合，定义通常的多项式加法和数乘。

(a) 写出 $\mathcal{P}_2$ 中的零向量。

(b) 多项式 $p(x) = 2 - x + 3x^2$ 的加法逆元是什么？

(c) 验证加法封闭性：若 $p, q \in \mathcal{P}_2$，证明 $p + q \in \mathcal{P}_2$。

---

**练习 9.4**（自定义运算）

在 $\mathbb{R}^2$ 上定义非标准运算：
$$
(x_1, y_1) \oplus (x_2, y_2) = (x_1 + x_2,\ y_1 + y_2 + 1)
$$
$$
c \odot (x, y) = (cx,\ cy + c - 1)
$$

请判断 $(\mathbb{R}^2, \oplus, \odot)$ 是否构成向量空间。（提示：先找零向量。）

---

**练习 9.5**（深度学习联系）

在 VAE 的潜在空间 $\mathbb{R}^k$ 中，设 $\mathbf{z}_A$ 和 $\mathbf{z}_B$ 分别是两张图像对应的潜在向量。

(a) 写出从 $\mathbf{z}_A$ 到 $\mathbf{z}_B$ 的线性插值公式，参数为 $t \in [0, 1]$。

(b) 解释为什么这个插值操作在向量空间中是合法的（引用具体公理）。

(c) 若 $\mathbf{z}_A = (1, 2, -1)^T$，$\mathbf{z}_B = (3, 0, 1)^T$，计算 $t = 0.25$ 和 $t = 0.75$ 时的插值向量。

---

## 练习答案

<details>
<summary>点击展开 练习 9.1 答案</summary>

$V = \{(x, y) \mid y = 3x\}$，即满足 $3x - y = 0$ 的点集。

**封闭性**：取 $\mathbf{u} = (a, 3a)$，$\mathbf{v} = (b, 3b) \in V$。
- $\mathbf{u} + \mathbf{v} = (a+b, 3a+3b) = (a+b, 3(a+b)) \in V$。加法封闭。
- 对 $c \in \mathbb{R}$：$c\mathbf{u} = (ca, 3ca) = (ca, 3(ca)) \in V$。数乘封闭。

**零向量**：$(0, 0) = (0, 3 \cdot 0) \in V$，满足 A4。

**加法逆元**：$(a, 3a)$ 的逆元为 $(-a, -3a) = (-a, 3(-a)) \in V$，满足 A5。

**其余公理（A2, A3, S2, S3, S4a, S4b）**：这些运算继承自 $\mathbb{R}^2$，对 $V$ 中的元素自然满足。

**结论**：$V$ 是 $\mathbb{R}$ 上的向量空间。

</details>

<details>
<summary>点击展开 练习 9.2 答案</summary>

**(a) $A$（单位闭圆盘）**：不是向量空间。

**违反 S1（数乘封闭性）**：取 $(1, 0) \in A$（因为 $1^2 + 0^2 = 1 \leq 1$），令 $c = 2$，则 $2 \cdot (1, 0) = (2, 0)$，但 $2^2 + 0^2 = 4 > 1$，故 $(2, 0) \notin A$。

**(b) $B$（$x + y = 1$ 的直线）**：不是向量空间。

**违反 A1（加法封闭性）**：取 $(1, 0), (0, 1) \in B$，则 $(1, 0) + (0, 1) = (1, 1)$，但 $1 + 1 = 2 \neq 1$，故 $(1, 1) \notin B$。

（也可指出 A4 不满足：零向量 $(0, 0)$ 不在 $B$ 中，因为 $0 + 0 = 0 \neq 1$。）

</details>

<details>
<summary>点击展开 练习 9.3 答案</summary>

**(a)** 零向量是零多项式 $\mathbf{0} = 0 + 0 \cdot x + 0 \cdot x^2$，即所有系数为零的多项式。

**(b)** $p(x) = 2 - x + 3x^2$ 的加法逆元是 $-p(x) = -2 + x - 3x^2$。

验证：$p(x) + (-p(x)) = (2-2) + (-1+1)x + (3-3)x^2 = 0$。

**(c)** 设 $p = a_0 + a_1 x + a_2 x^2$，$q = b_0 + b_1 x + b_2 x^2$，其中 $a_i, b_i \in \mathbb{R}$。

$$p + q = (a_0 + b_0) + (a_1 + b_1)x + (a_2 + b_2)x^2$$

因为 $a_i + b_i \in \mathbb{R}$，且 $p + q$ 仍是次数不超过 2 的实系数多项式，故 $p + q \in \mathcal{P}_2$。加法封闭。

</details>

<details>
<summary>点击展开 练习 9.4 答案</summary>

**寻找零向量**：设零向量为 $(e_1, e_2)$，需满足对所有 $(x, y)$：

$$(x, y) \oplus (e_1, e_2) = (x, y)$$
$$(x + e_1, y + e_2 + 1) = (x, y)$$

由此得 $e_1 = 0$，$e_2 + 1 = 0$，即 $e_2 = -1$。**零向量为 $(0, -1)$**。

**验证单位元公理（S3）**：需要 $1 \odot (x, y) = (x, y)$。

计算：$1 \odot (x, y) = (1 \cdot x, 1 \cdot y + 1 - 1) = (x, y)$。S3 成立。

**验证标量分配律（S4b）**：需要 $(c + d) \odot (x, y) = c \odot (x, y) \oplus d \odot (x, y)$。

左边：$(c+d) \odot (x, y) = ((c+d)x, (c+d)y + (c+d) - 1)$

右边：$c \odot (x, y) \oplus d \odot (x, y) = (cx, cy + c - 1) \oplus (dx, dy + d - 1)$
$= (cx + dx, (cy + c - 1) + (dy + d - 1) + 1) = ((c+d)x, (c+d)y + c + d - 1)$

左边 $= ((c+d)x, (c+d)y + c + d - 1)$ = 右边。S4b 成立。

类似可验证其余公理均成立（加法交换律、结合律、加法逆元、数乘结合律、向量分配律）。

**结论**：$(\mathbb{R}^2, \oplus, \odot)$ **是**向量空间。（它同构于通常的 $\mathbb{R}^2$，通过映射 $(x, y) \mapsto (x, y+1)$ 建立同构。）

</details>

<details>
<summary>点击展开 练习 9.5 答案</summary>

**(a)** 线性插值公式：

$$\mathbf{z}(t) = (1 - t)\,\mathbf{z}_A + t\,\mathbf{z}_B, \quad t \in [0, 1]$$

**(b)** 合法性基于以下公理：

- **S1（数乘封闭）**：$(1-t)\mathbf{z}_A \in \mathbb{R}^k$ 和 $t\mathbf{z}_B \in \mathbb{R}^k$，因为 $\mathbb{R}^k$ 对数乘封闭。
- **A1（加法封闭）**：两个属于 $\mathbb{R}^k$ 的向量之和仍在 $\mathbb{R}^k$ 中，因为加法封闭。

因此 $\mathbf{z}(t) \in \mathbb{R}^k$ 对所有 $t$ 成立。此外，线性插值是一个**线性组合**，而向量空间对线性组合封闭（由 A1 和 S1 共同保证）。

**(c)** 已知 $\mathbf{z}_A = (1, 2, -1)^T$，$\mathbf{z}_B = (3, 0, 1)^T$。

$$\mathbf{z}(t) = (1-t)(1, 2, -1)^T + t(3, 0, 1)^T = (1 + 2t,\ 2 - 2t,\ -1 + 2t)^T$$

当 $t = 0.25$：
$$\mathbf{z}(0.25) = (1 + 0.5,\ 2 - 0.5,\ -1 + 0.5)^T = (1.5,\ 1.5,\ -0.5)^T$$

当 $t = 0.75$：
$$\mathbf{z}(0.75) = (1 + 1.5,\ 2 - 1.5,\ -1 + 1.5)^T = (2.5,\ 0.5,\ 0.5)^T$$

</details>
