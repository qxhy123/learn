# 第6章 齐次、线性、仿射与非线性

## 学习目标

通过本章学习，你将能够：

- 准确区分**齐次**、**线性**、**仿射**与**非线性**四个常被混用的概念，并理解它们之间的严格包含关系
- 从函数视角和方程视角分别理解齐次性与线性性：可加性、齐次性、叠加原理
- 区分**线性映射**与**线性函数**（中学意义的“一次函数”），理解为什么 $y=kx+b$（$b\ne0$）是仿射而非线性
- 理解齐次方程与非齐次方程的关系：通解 = 齐次通解 + 一个特解
- 掌握判断一个映射或方程是否线性、是否齐次、是否仿射的标准流程
- 建立这些概念与后续微分方程、线性代数、神经网络层（全连接、激活函数、归一化）之间的联系

---

## 6.1 为什么微积分需要严格区分这些概念

在中学和工程语境中，“线性”经常被宽泛使用：$y=2x+3$ 常被称作“线性函数”，$\frac{dy}{dx}+y=\sin x$ 也常被称作“一阶线性微分方程”。这里的“线性”含义并不一致：

- 在线性代数中，**线性映射**必须满足 $f(x+y)=f(x)+f(y)$ 与 $f(\alpha x)=\alpha f(x)$，所以 $y=2x+3$ 严格说不是线性映射；
- 在微分方程中，“线性”指方程关于未知函数及其导数是一次的，**允许**有非齐次项；
- 在机器学习中，神经网络的“线性层”实际上是 $\mathbf y=W\mathbf x+\mathbf b$，按线性代数定义是**仿射变换**。

如果不区分清楚，后面学习矩阵微积分、ODE、神经网络结构时就会反复混淆。因此本章要建立三对清晰的对比：

1. **齐次 vs 非齐次**：方程或函数中有没有“常数项 / 强迫项”；
2. **线性 vs 仿射**：映射是否过原点，是否满足严格的可加性与齐次性；
3. **线性 vs 非线性**：映射是否破坏叠加原理。

掌握这些区分后，许多看似复杂的现象（神经网络的非线性来自哪里？为什么解 ODE 时先求齐次通解？）就会变得自然。

> **资料参考**：Strang *Introduction to Linear Algebra* 系统区分线性映射与仿射映射；Boyd & Vandenberghe *Convex Optimization* 强调仿射与线性的区别；OpenStax Calculus Volume 2 中关于线性 ODE 的章节给出齐次/非齐次的标准定义。

---

## 6.2 齐次性

### 6.2.1 函数的齐次性

**定义（$k$ 次齐次函数）**：设 $f:\mathbb R^n\to\mathbb R^m$。若存在实数 $k$，使得对任意 $\mathbf x$ 和任意 $\alpha>0$：

$$
f(\alpha\mathbf x)=\alpha^k f(\mathbf x),
$$

则称 $f$ 为 **$k$ 次齐次函数**。$k=1$ 时称为**一次齐次**或简称**齐次**。

直观上，齐次函数对自变量的“整体缩放”有一致的响应：放大输入若干倍，输出按某个幂次同步放大。

| 例子 | 齐次性 |
|:---:|:---|
| $f(x,y)=3x+2y$ | $1$ 次齐次：$f(\alpha x,\alpha y)=\alpha(3x+2y)=\alpha f(x,y)$ |
| $f(x,y)=x^2+xy+y^2$ | $2$ 次齐次：$f(\alpha x,\alpha y)=\alpha^2 f(x,y)$ |
| $f(x,y)=\frac{x^2}{y}$ | $1$ 次齐次 |
| $f(x,y)=x+y+1$ | **非齐次**：常数项 $1$ 破坏齐次性 |
| $f(x,y)=\sin(x+y)$ | 非齐次 |

**关键观察**：齐次函数必须满足 $f(\mathbf 0)=0$（取 $\alpha\to 0$ 即可），所以**任何含非零常数项的函数都不齐次**。

### 6.2.2 方程的齐次性

**定义（齐次方程）**：方程

$$
L[y]=g(x)
$$

中，若 $g(x)\equiv 0$，称其为**齐次方程**；否则称为**非齐次方程**，$g(x)$ 称为**非齐次项**或**强迫项**。

例如：

| 方程 | 类型 |
|:---|:---|
| $y''+3y'+2y=0$ | 齐次线性 ODE |
| $y''+3y'+2y=\sin x$ | 非齐次线性 ODE |
| $A\mathbf x=\mathbf 0$ | 齐次线性方程组 |
| $A\mathbf x=\mathbf b\ (\mathbf b\ne\mathbf 0)$ | 非齐次线性方程组 |

齐次方程总有零解 $y\equiv 0$（或 $\mathbf x=\mathbf 0$），非齐次方程一般没有零解。这是判断齐次性的最快办法。

### 6.2.3 齐次的几何意义

对线性方程组 $A\mathbf x=\mathbf 0$，解集是过原点的子空间。对非齐次方程组 $A\mathbf x=\mathbf b$，解集是该子空间**平移**后的仿射子空间——通过任意一个特解 $\mathbf x_p$ 平移：

$$
\{\mathbf x_p+\mathbf v:A\mathbf v=\mathbf 0\}.
$$

这是“**通解 = 齐次通解 + 特解**”的几何来源。

---

## 6.3 线性性

### 6.3.1 线性映射的定义

**定义（线性映射）**：设 $V,W$ 是同一个数域上的向量空间。映射 $T:V\to W$ 称为**线性映射**，如果对所有 $\mathbf x,\mathbf y\in V$ 和所有标量 $\alpha,\beta$：

$$
T(\alpha\mathbf x+\beta\mathbf y)=\alpha T(\mathbf x)+\beta T(\mathbf y).
$$

这条等式等价于以下两条之合：

1. **可加性**：$T(\mathbf x+\mathbf y)=T(\mathbf x)+T(\mathbf y)$；
2. **齐次性**：$T(\alpha\mathbf x)=\alpha T(\mathbf x)$。

合并后的形式即**叠加原理**：输入的线性组合映到输出的相应线性组合。

### 6.3.2 立即推论

由定义直接得到：

- $T(\mathbf 0)=\mathbf 0$（取 $\alpha=\beta=0$）。**任何不过原点的映射都不是线性映射**；
- 线性映射保持原点、保持过原点的直线、保持加法与标量乘；
- $T$ 在选定基底下可以由一个矩阵唯一表示：$T(\mathbf x)=A\mathbf x$。

### 6.3.3 “线性函数”的两种含义

| 语境 | 含义 |
|:---|:---|
| 中学 / 工程 | “一次函数” $y=kx+b$，图像是直线 |
| 线性代数 / 微积分 | 严格线性映射，必须满足 $T(\mathbf 0)=\mathbf 0$ |

注意 $y=kx+b$（$b\ne0$）满足“图像为直线”，但不是线性映射——因为 $f(0)=b\ne 0$。它属于下一节要讲的**仿射函数**。

### 6.3.4 线性方程与线性微分方程

**线性方程**：方程称为线性，如果未知量（包括它在不同位置出现）以一次形式出现，并且不与未知量的非线性函数相乘。

例如：

- $3x+2y=5$ 关于 $x,y$ 线性；
- $\sin(x)+y=0$ 关于 $y$ 线性，关于 $x$ 非线性；
- $xy=1$ 既不关于 $x$ 线性，也不关于 $y$ 线性。

**线性微分方程**：形如

$$
a_n(x)y^{(n)}+\cdots+a_1(x)y'+a_0(x)y=g(x).
$$

未知函数 $y$ 及其各阶导数都以一次形式出现，且不出现 $y\cdot y'$、$\sin y$、$y^2$ 等非线性项。注意：**系数 $a_i(x)$ 可以是 $x$ 的任意函数**——它们对“线性”不构成破坏，因为我们只要求方程关于 $y$ 及其导数线性。

> **例题 6.1** 下列方程哪些是线性的？哪些是齐次的？
> 1. $y''+x^2 y=0$　　2. $y''+y^2=0$　　3. $y'+(\sin x)y=e^x$　　4. $yy'=1$

**解**：

1. 关于 $y,y'$ 一次，无非齐次项，是**齐次线性**ODE；
2. 含 $y^2$，**非线性**；
3. 关于 $y,y'$ 一次，含非齐次项 $e^x$，是**非齐次线性**ODE；
4. 含 $y\cdot y'$，**非线性**。

---

## 6.4 仿射性

### 6.4.1 仿射映射的定义

**定义（仿射映射）**：映射 $f:V\to W$ 称为**仿射映射**，如果存在线性映射 $L$ 和常向量 $\mathbf b\in W$，使得

$$
f(\mathbf x)=L(\mathbf x)+\mathbf b.
$$

等价地，$f$ 仿射 $\iff f(\mathbf x)-f(\mathbf 0)$ 是线性映射。

### 6.4.2 仿射的等价刻画

仿射映射保持**仿射组合**：对系数 $\sum\alpha_i=1$ 的任意组合，

$$
f\!\left(\sum_i\alpha_i\mathbf x_i\right)=\sum_i\alpha_i f(\mathbf x_i),\quad\text{当 }\sum_i\alpha_i=1.
$$

注意与线性的区别：线性要求**所有**线性组合都被保持；仿射只要求**系数和为 $1$** 的组合被保持。系数和为 $1$ 包含两种重要的特例：

- 凸组合（$\alpha_i\ge 0$ 且 $\sum\alpha_i=1$）；
- 直线参数化 $(1-t)\mathbf x_0+t\mathbf x_1$。

因此仿射映射把直线映为直线，把凸集映为凸集，但不要求过原点。

### 6.4.3 线性 ⊂ 仿射

线性映射是 $\mathbf b=\mathbf 0$ 的特殊仿射映射。反之未必。判定流程：

1. 若 $f(\mathbf 0)=\mathbf 0$ 且 $f$ 满足 $f(\alpha\mathbf x+\beta\mathbf y)=\alpha f(\mathbf x)+\beta f(\mathbf y)$，则**线性**；
2. 若 $f(\mathbf 0)\ne\mathbf 0$，但 $f(\mathbf x)-f(\mathbf 0)$ 线性，则**仿射**但非线性；
3. 否则**非线性**。

### 6.4.4 常见仿射映射例子

| 映射 | 是否线性 | 是否仿射 |
|:---|:---:|:---:|
| $f(x)=3x$ | ✅ | ✅ |
| $f(x)=3x+5$ | ❌ | ✅ |
| $f(\mathbf x)=A\mathbf x$ | ✅ | ✅ |
| $f(\mathbf x)=A\mathbf x+\mathbf b\ (\mathbf b\ne\mathbf 0)$ | ❌ | ✅ |
| 旋转 + 平移（刚体变换） | 仅当平移为 $0$ 时线性 | ✅ |
| $f(x)=x^2$ | ❌ | ❌ |

---

## 6.5 非线性

### 6.5.1 非线性的定义

**非线性**就是“不是线性”，即至少破坏一条：可加性或齐次性。常见来源：

- 出现 $\mathbf x$ 的高次项：$x^2,xy,\|x\|^2$；
- 出现非线性函数：$\sin x,\ e^x,\ \log x,\ \max(0,x)$；
- 出现未知量相乘：$y\cdot y'$、$y\cdot u$；
- 出现绝对值、分段函数、阈值。

注意：**仿射不是线性，但通常不被称为非线性**。在工程语境中，“非线性”特指偏离仿射形式（即不能写成 $A\mathbf x+\mathbf b$）。本章采用这一惯例。

### 6.5.2 非线性的代价与价值

非线性带来三个性质：

1. **失去叠加**：不能把复杂输入拆成简单输入分别求解再相加；
2. **可能多解或无解**：解集结构变复杂；
3. **表达能力强**：可以近似任意函数（万能近似定理）。

神经网络的强大表达力来自激活函数引入的非线性。如果所有层都仿射，则整体仍是仿射，无论多深都不会增加表达能力。

### 6.5.3 神经网络层的视角

一个标准的全连接层 + 激活：

$$
\mathbf z=W\mathbf x+\mathbf b,\qquad \mathbf a=\sigma(\mathbf z).
$$

- $\mathbf z=W\mathbf x+\mathbf b$ 是**仿射变换**（$\mathbf b\ne\mathbf 0$ 时不是线性变换）；
- $\sigma$（如 ReLU、Sigmoid、Tanh、GELU）是**逐元素非线性函数**；
- 因此“线性层 + 激活”整体是**非线性**映射。

去掉激活函数后，多层堆叠

$$
\mathbf y=W_L\cdots W_2(W_1\mathbf x+\mathbf b_1)+\mathbf b_2\cdots+\mathbf b_L
$$

仍是关于 $\mathbf x$ 的仿射函数——可化为单层 $W'\mathbf x+\mathbf b'$。所以激活函数是网络能逼近非线性现象的关键。

> **例题 6.2** 判定下列映射的类型：线性、仿射但非线性、非线性。
> 1. $f(x,y)=2x-3y$　　2. $f(x,y)=2x-3y+1$　　3. $f(x,y)=xy$　　4. $f(\mathbf x)=\max(0,W\mathbf x+\mathbf b)$（ReLU 层）

**解**：

1. 满足两条公理，**线性**；
2. $f(0,0)=1\ne 0$，但 $f-1$ 线性，**仿射但非线性**；
3. $f(\alpha x,\alpha y)=\alpha^2 xy\ne \alpha f$，**非线性**（且不仿射）；
4. ReLU 是分段线性，整体不仿射，**非线性**。

---

## 6.6 齐次方程与非齐次方程的解结构

线性方程（含 ODE）的核心定理把齐次与非齐次连接起来。

### 6.6.1 叠加原理

设 $L$ 是线性算子（例如 $L[y]=y''+py'+qy$ 或 $L[\mathbf x]=A\mathbf x$）。

**齐次叠加**：若 $L[y_1]=0,\ L[y_2]=0$，则对任意常数 $c_1,c_2$，

$$
L[c_1y_1+c_2y_2]=0.
$$

即齐次方程的解集是一个**向量空间**。

**非齐次叠加**：若 $L[y_p]=g$ 且 $L[y_h]=0$，则

$$
L[y_h+y_p]=g.
$$

### 6.6.2 通解结构定理

线性方程

$$
L[y]=g
$$

的通解可写为

$$
y=y_h+y_p,
$$

其中 $y_h$ 取遍齐次方程 $L[y]=0$ 的所有解，$y_p$ 是非齐次方程的任一**特解**。

应用这一结构的标准流程：

1. 先解齐次方程 $L[y]=0$，得到 $y_h$；
2. 用特定方法（待定系数、参数变易、Laplace 变换等）求一个特解 $y_p$；
3. 写出通解 $y=y_h+y_p$；
4. 若给定初值或边值，代入确定 $y_h$ 中的待定常数。

### 6.6.3 线性方程组的对应

对 $A\mathbf x=\mathbf b$，若 $A\mathbf x_p=\mathbf b$，则全部解为

$$
\mathbf x=\mathbf x_p+\mathbf v,\qquad \mathbf v\in\ker A.
$$

这就是矩阵论中“**特解 + 零空间**”的形式，本质与 ODE 通解结构是同一个定理。

> **例题 6.3** 求 $y'+y=e^{x}$ 的通解。

**解**：

1. 齐次方程 $y'+y=0$ 的通解为 $y_h=Ce^{-x}$；
2. 设特解 $y_p=Ae^{x}$。代入：$Ae^x+Ae^x=e^x$，得 $A=\tfrac12$，所以 $y_p=\tfrac12 e^x$；
3. 通解：
$$
y=Ce^{-x}+\frac12 e^x.
$$

---

## 6.7 判断流程汇总

下面是一个实用的判定流程图（文字描述形式），可用于任何映射 $f$ 或方程 $L[y]=g$。

**对映射 $f:V\to W$**：

1. 计算 $f(\mathbf 0)$。
   - 若 $f(\mathbf 0)\ne\mathbf 0$：肯定不是线性，转入第 3 步；
   - 若 $f(\mathbf 0)=\mathbf 0$：可能线性，进入第 2 步；
2. 验证 $f(\alpha\mathbf x+\beta\mathbf y)\stackrel{?}{=}\alpha f(\mathbf x)+\beta f(\mathbf y)$。
   - 成立：**线性**；
   - 不成立：**非线性**（但不一定仿射）；
3. 计算 $g(\mathbf x):=f(\mathbf x)-f(\mathbf 0)$，对 $g$ 重新执行第 2 步。
   - $g$ 线性：$f$ **仿射但非线性**；
   - $g$ 非线性：$f$ **非线性且非仿射**。

**对方程 $L[y]=g(x)$**：

1. 看 $L$ 是否是线性算子（关于 $y$ 与其导数线性）。
   - 不是：**非线性方程**；
   - 是：进入第 2 步；
2. 看 $g(x)$。
   - $g\equiv 0$：**齐次线性方程**；
   - $g\not\equiv 0$：**非齐次线性方程**。

### 概念关系总览

```
                ┌─────────────────────────────────┐
                │           所有映射 / 方程        │
                │                                 │
                │   ┌──────────────────────────┐  │
                │   │        仿射映射          │  │
                │   │                          │  │
                │   │   ┌──────────────────┐   │  │
                │   │   │   线性映射       │   │  │
                │   │   │ （齐次 + 可加）  │   │  │
                │   │   └──────────────────┘   │  │
                │   │  （线性 + 非零常数项）   │  │
                │   └──────────────────────────┘  │
                │     非线性（既非线性也非仿射）  │
                └─────────────────────────────────┘
```

- **线性 ⊂ 仿射 ⊂ 所有映射**；
- **齐次性** 是另一根轴：线性必齐次，仿射不一定齐次；
- **非线性** 是补集，凡不能写成 $A\mathbf x+\mathbf b$ 的都属于这一类。

---

## 6.8 深度学习应用

齐次、线性、仿射、非线性的区分在深度学习中无处不在。

### 6.8.1 全连接层 = 仿射变换

一层全连接

$$
\mathbf z=W\mathbf x+\mathbf b
$$

是仿射变换。若 $\mathbf b=\mathbf 0$（无偏置）则退化为线性变换。

去掉 $\mathbf b$ 在某些场景（如紧跟 BatchNorm 的卷积层）反而更合适，因为 BN 自己引入了可学习的平移参数，再加 $\mathbf b$ 是冗余。这就是 ResNet 等结构中 `Conv-BN-ReLU` 块里卷积层常常 `bias=False` 的原因。

### 6.8.2 激活函数引入非线性

ReLU、Sigmoid、Tanh、GELU 等激活函数都是**非线性**逐元素函数：

$$
\mathrm{ReLU}(x)=\max(0,x),\qquad
\mathrm{GELU}(x)=x\cdot\Phi(x).
$$

激活函数是网络脱离“仿射变换合成仍是仿射”这一陷阱的唯一来源。万能近似定理保证：单隐层 + 非常数有界连续激活，就足以在紧集上一致逼近任意连续函数。

### 6.8.3 LayerNorm 的仿射参数

LayerNorm 把激活归一化后引入可学习的**仿射变换**：

$$
\mathrm{LN}(\mathbf x)=\gamma\odot\frac{\mathbf x-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta.
$$

- 归一化部分是非线性的（依赖 $\mathbf x$ 的统计量）；
- 最后的 $\gamma\odot\hat{\mathbf x}+\beta$ 是仿射的，让模型能恢复任意均值与方差。

### 6.8.4 损失函数中的齐次性

很多正则项是齐次的：

- L2 正则 $\|\theta\|^2$ 是 $2$ 次齐次；
- L1 正则 $\|\theta\|_1$ 是 $1$ 次齐次。

这一齐次性导致**权重缩放与学习率缩放等价**：把 $\theta$ 缩放 $s$ 倍，正则项缩放 $s^k$ 倍，等价于改变正则强度。这一性质在分析尺度不变性、隐式正则化时很有用。

### 代码示例：验证仿射 vs 线性

```python
import torch
import torch.nn as nn


def is_linear(layer: nn.Module, input_dim: int, n_samples: int = 5) -> bool:
    """检查 layer(0) == 0，即满足线性的必要条件。"""
    zero_in = torch.zeros(1, input_dim)
    zero_out = layer(zero_in)
    return torch.allclose(zero_out, torch.zeros_like(zero_out), atol=1e-6)


linear_no_bias = nn.Linear(4, 3, bias=False)
linear_with_bias = nn.Linear(4, 3, bias=True)

print("no bias  -> linear:", is_linear(linear_no_bias, 4))
print("with bias-> linear:", is_linear(linear_with_bias, 4))
# 典型输出：
# no bias  -> linear: True
# with bias-> linear: False  （它是仿射，不是线性）
```

`nn.Linear(in, out)` 默认是仿射变换；只有 `bias=False` 时才是严格的线性映射。

---

## 本章小结

1. **齐次** 要求 $f(\alpha\mathbf x)=\alpha^k f(\mathbf x)$；齐次方程是右端为 $0$ 的方程。
2. **线性** 要求可加性与齐次性同时成立；线性映射必过原点，并可由矩阵表示。
3. **仿射** = 线性 + 平移：$f(\mathbf x)=A\mathbf x+\mathbf b$。$y=kx+b$（$b\ne0$）是仿射而非线性。
4. **非线性** 是补集：凡不能写成仿射形式的映射都是非线性。
5. **解结构定理**：线性方程通解 = 齐次通解 + 非齐次特解；对线性方程组对应于“特解 + 零空间”。
6. **判定流程**：先看 $f(\mathbf 0)$，再验证可加性/齐次性；对方程先看是否关于未知量线性，再看右端是否为零。
7. **深度学习联系**：全连接层 = 仿射，激活函数 = 非线性，LayerNorm 末端 = 仿射；正则项的齐次性带来尺度不变性。

---

## 资料与延伸阅读

- Strang G. *Introduction to Linear Algebra*, 5th ed., Chapter 8。重点参考线性映射与仿射映射的对比、解结构定理。
- Boyd S., Vandenberghe L. *Convex Optimization*, Chapter 2。重点参考仿射集、仿射函数与凸集的关系。
- [OpenStax Calculus Volume 2, Chapter 4: Introduction to Differential Equations](https://openstax.org/books/calculus-volume-2/pages/4-introduction)。重点参考齐次/非齐次线性 ODE 的标准定义。
- [Paul's Online Math Notes, Differential Equations: Linear Equations](https://tutorial.math.lamar.edu/Classes/DE/Linear.aspx)。重点参考一阶线性 ODE 的解题流程。
- Goodfellow, Bengio, Courville. *Deep Learning*, Chapter 6。重点参考前馈网络中仿射变换 + 非线性激活的组合结构。

---

## 练习题

**1.** ⭐ 判断下列函数是否齐次，若是请指出齐次次数：
   (a) $f(x,y)=4x-y$　　(b) $f(x,y)=x^2+y^2$　　(c) $f(x,y)=\frac{x^3+y^3}{x+y}$　　(d) $f(x,y)=x+y+1$

**2.** ⭐ 判断下列映射是线性、仿射但非线性、还是非线性：
   (a) $f(x)=5x$　　(b) $f(x)=5x-2$　　(c) $f(x)=x^2$　　(d) $f(\mathbf x)=A\mathbf x+\mathbf b\ (\mathbf b\ne 0)$

**3.** ⭐ 判断下列方程的类型（线性/非线性，齐次/非齐次）：
   (a) $y''+4y=0$　　(b) $y''+4y=\cos x$　　(c) $y'+y^2=0$　　(d) $(\sin x)y'+y=e^x$

**4.** ⭐⭐ 证明：若 $f:\mathbb R^n\to\mathbb R$ 是一次齐次函数，则它必满足 $f(\mathbf 0)=0$。给出该结论的反例不成立的简要说明（即非齐次函数不一定满足 $f(\mathbf 0)=0$）。

**5.** ⭐⭐ 设 $f(\mathbf x)=A\mathbf x+\mathbf b$，$A$ 为给定矩阵，$\mathbf b\ne\mathbf 0$。证明 $f$ 是仿射但不是线性，并验证 $f$ 保持仿射组合 $\sum\alpha_i\mathbf x_i$（$\sum\alpha_i=1$）但**不**保持一般线性组合。

**6.** ⭐⭐ 求一阶线性 ODE $y'+y=e^{x}$ 的通解，并验证你的解满足结构定理 $y=y_h+y_p$。

**7.** ⭐⭐⭐ 考虑两层无激活的“线性”网络

$$
\mathbf y=W_2(W_1\mathbf x+\mathbf b_1)+\mathbf b_2.
$$

证明它整体上等价于一个单层仿射变换 $\mathbf y=W'\mathbf x+\mathbf b'$，并写出 $W',\mathbf b'$ 的表达式。再说明这为什么意味着“没有激活函数的深度网络与浅层网络等价”。

**8.** ⭐⭐⭐ 在卷积神经网络中，紧跟 BatchNorm 的卷积层常常设置 `bias=False`。从仿射变换的角度解释这种设计为什么不会损失表达能力。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.**
(a) $f(\alpha x,\alpha y)=\alpha(4x-y)=\alpha f$，**$1$ 次齐次**。

(b) $f(\alpha x,\alpha y)=\alpha^2(x^2+y^2)=\alpha^2 f$，**$2$ 次齐次**。

(c) 分子 $3$ 次、分母 $1$ 次，整体 $2$ 次：$f(\alpha x,\alpha y)=\alpha^2 f$，**$2$ 次齐次**。

(d) $f(0,0)=1\ne 0$，**非齐次**。

---

**2.**
(a) $f(0)=0$ 且可加齐次，**线性**。

(b) $f(0)=-2\ne 0$，但 $f(x)+2=5x$ 线性，**仿射但非线性**。

(c) $f(\alpha x)=\alpha^2 x^2\ne\alpha f$，**非线性**。

(d) $f(\mathbf 0)=\mathbf b\ne\mathbf 0$，但 $f-\mathbf b=A\mathbf x$ 线性，**仿射但非线性**。

---

**3.**
(a) 关于 $y,y''$ 一次，右端 $0$，**齐次线性**。

(b) 关于 $y,y''$ 一次，右端 $\cos x\ne 0$，**非齐次线性**。

(c) 含 $y^2$，**非线性**。

(d) 关于 $y,y'$ 一次（系数 $\sin x$ 是 $x$ 的函数，不破坏线性），右端 $e^x$，**非齐次线性**。

---

**4.** 由齐次定义，对任意 $\alpha>0$ 有 $f(\alpha\mathbf 0)=\alpha f(\mathbf 0)$。而 $\alpha\mathbf 0=\mathbf 0$，所以 $f(\mathbf 0)=\alpha f(\mathbf 0)$。取 $\alpha\ne 1$ 即得 $f(\mathbf 0)=0$。

反例方向：$f(x)=x+1$ 满足 $f(0)=1\ne 0$，因此不齐次，但它在 $x\ne 0$ 处仍取实数值——非齐次只是意味着不满足 $f(\alpha\mathbf x)=\alpha^k f(\mathbf x)$ 的形式，并不要求 $f$ 在原点必有特殊行为。

---

**5.** $f(\mathbf 0)=\mathbf b\ne\mathbf 0$，故 $f$ 不是线性。但 $f(\mathbf x)-f(\mathbf 0)=A\mathbf x$ 是线性，因此 $f$ 是仿射。

验证仿射组合：设 $\sum\alpha_i=1$，

$$
f\!\left(\sum_i\alpha_i\mathbf x_i\right)
=A\sum_i\alpha_i\mathbf x_i+\mathbf b
=\sum_i\alpha_i A\mathbf x_i+\left(\sum_i\alpha_i\right)\mathbf b
=\sum_i\alpha_i(A\mathbf x_i+\mathbf b)
=\sum_i\alpha_i f(\mathbf x_i).
$$

而对一般线性组合（$\sum\alpha_i\ne 1$），多出的项 $\bigl(\sum_i\alpha_i-1\bigr)\mathbf b\ne\mathbf 0$，等号不成立。

---

**6.** 齐次方程 $y'+y=0$ 的解：$y_h=Ce^{-x}$。

设特解 $y_p=Ae^x$，代入：$Ae^x+Ae^x=e^x\Rightarrow A=\tfrac12$。

通解：

$$
y=Ce^{-x}+\frac12 e^x.
$$

验证结构定理：$y_h=Ce^{-x}$ 满足齐次方程；$y_p=\tfrac12 e^x$ 满足非齐次方程；二者之和满足非齐次方程。 $\square$

---

**7.** 展开：

$$
\mathbf y=W_2 W_1\mathbf x+W_2\mathbf b_1+\mathbf b_2.
$$

所以 $W'=W_2 W_1$，$\mathbf b'=W_2\mathbf b_1+\mathbf b_2$，整体仍为仿射变换。

这意味着没有激活的多层网络等价于单层仿射变换：无论堆多深，函数族都是 $\{\mathbf x\mapsto W'\mathbf x+\mathbf b':W',\mathbf b'\}$，表达能力与单层完全一致。要获得非线性表达能力，必须引入非线性激活函数。

---

**8.** 卷积层加偏置后输出为 $W*\mathbf x+\mathbf b$（仿射）。紧跟 BatchNorm：

$$
\mathrm{BN}(z)=\gamma\cdot\frac{z-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta.
$$

BN 先减去均值 $\mu$。由于 $\mathbf b$ 在所有样本上是相同常数，它会被合并进 $\mu$ 而被减掉——也就是说，卷积层的 $\mathbf b$ 对 BN 后的输出没有影响，等价于该参数恒为零。

同时，BN 末端的可学习参数 $\beta$ 已经提供了一个独立的平移自由度，完全可以扮演 $\mathbf b$ 的角色。因此卷积层去掉 $\mathbf b$（变成纯线性变换）不会损失表达能力，反而减少冗余参数与计算。

</details>

---

## 思考路标（条件反射）

- 看到 $f(x+y) = f(x) + f(y)$ + $f(\lambda x) = \lambda f(x)$ → **线性**
- 看到 $f(\lambda x) = \lambda f(x)$ 仅满足 → **齐次（不一定线性）**
- 看到 $f(x) = Ax + b$（$b \neq 0$）→ **仿射（不是线性）**
- 看到神经网络层 $y = Wx + b$ → **仿射变换**（不是纯线性，除非 $b = 0$）
- 看到 BatchNorm 后的 Conv → $b$ 被均值减去 → 可省略 $b$（卷积层 bias=False）
- 看到 ReLU / sigmoid / tanh → **非线性激活**（破坏线性）
- 看到"叠加性"或"superposition" → 线性的关键性质
- 看到"齐次解 + 特解" → ODE 解结构定理（线性 ODE 通解）

## 易错点

1. **齐次 vs 线性 vs 仿射**：齐次只是 $f(\lambda x) = \lambda f(x)$；线性还要求加性；仿射是 $Ax + b$。
2. **$f(x) = x^2$ 是齐次的吗**？$f(\lambda x) = \lambda^2 x^2$ 不是 $\lambda f(x)$，所以**不齐次**（除非定义二次齐次）。
3. **仿射 ≠ 线性**：$f(0) = b \neq 0$ 不满足线性的 $f(0) = 0$。
4. **神经网络的"线性层"实际是仿射层**：俗称"linear"，严格是 affine。
5. **激活函数破坏线性后才能拟合复杂函数**：纯堆叠线性层 = 一个线性层（无意义）。
