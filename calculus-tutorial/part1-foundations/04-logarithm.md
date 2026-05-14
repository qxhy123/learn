# 第4章 对数与指数函数

## 学习目标

通过本章学习，你将能够：

- 理解指数函数与对数函数互为反函数，并掌握自然底数 $e$ 为什么是微积分中的自然底数
- 熟练进行指数与对数的换底、化简和方程求解，掌握常用底（$e$、$10$、$2$）之间的转换
- 从定义域、值域、单调性、奇偶性、渐近行为和图像变换角度分析指数函数与对数函数
- 熟练使用对数恒等式进行化简、证明、解方程，并理解每条恒等式背后的指数运算规律
- 理解对数为什么在概率、信息论与数值计算中无处不在：把乘法变加法、把指数变线性、把跨数量级的量压缩到可比较的尺度
- 建立对数、指数与后续极限、导数、积分、概率分布、损失函数和深度学习数值稳定性的联系

---

## 4.1 为什么微积分需要对数与指数

指数函数描述**按比例增长或衰减**：人口、复利、放射性衰变、RC 电路、神经网络中的概率分布都是指数形式。对数函数是指数函数的反函数，它把乘法变成加法、把指数变成线性，让跨数量级的量可以比较和处理。

在微积分中，自然指数 $e^x$ 是唯一一个**导数等于自身**的函数；自然对数 $\ln x$ 是唯一一个**导数等于 $\frac1x$** 的函数。这两个性质让 $e$ 和 $\ln$ 成为所有微分、积分公式中最简洁的底。

例如：

- 概率论中的高斯分布、指数分布、Softmax 都建立在 $e^x$ 之上；
- 信息论中的熵 $H=-\sum p\log p$ 与交叉熵损失直接来自对数；
- 数值计算中常用 log-sum-exp、log-likelihood 来避免上溢和下溢；
- 物理与工程中的 dB（分贝）、pH、地震震级都是对数标度。

因此本章不仅要记公式，更要掌握三个核心思想：

1. **对数把乘法变加法、把幂变乘法**；
2. **自然底数 $e$ 是微积分中导数与积分最简洁的底**；
3. **对数把数量级差距巨大的量压成可比的线性尺度**。

> **资料参考**：OpenStax Calculus Volume 1 用极限定义 $e$ 并系统介绍指对数；OpenStax Precalculus 涵盖指数对数函数、方程和应用；Paul's Online Math Notes 与 Khan Academy 提供大量化简、求值、解方程练习。

---

## 4.2 指数函数与对数函数的定义

### 4.2.1 指数函数

固定底数 $a>0$ 且 $a\ne1$，定义**指数函数**

$$
y=a^x,\qquad x\in\mathbb R.
$$

整数次幂由乘法重复定义；有理数次幂由 $a^{p/q}=\sqrt[q]{a^p}$ 定义；无理数次幂通过有理数的极限定义，并要求 $a^x$ 关于 $x$ 连续。

最重要的指数运算律：

$$
a^x\cdot a^y=a^{x+y},
\qquad
\frac{a^x}{a^y}=a^{x-y},
\qquad
(a^x)^y=a^{xy},
\qquad
(ab)^x=a^x b^x.
$$

特别地，

$$
a^0=1,\qquad a^{-x}=\frac1{a^x}.
$$

### 4.2.2 自然底数 $e$

自然底数 $e$ 是一个无理数，约等于 $2.71828\ldots$。它有多种等价定义，常见的有：

$$
e=\lim_{n\to\infty}\left(1+\frac1n\right)^n,
$$

$$
e=\sum_{n=0}^\infty\frac1{n!}=1+1+\frac1{2!}+\frac1{3!}+\cdots.
$$

$e$ 在微积分中的重要性来自一个关键事实：

$$
\frac{d}{dx}e^x=e^x.
$$

即 $e^x$ 是唯一一个（在常数倍意义下）导数等于自身的函数。

### 4.2.3 对数函数

固定底数 $a>0$ 且 $a\ne1$，**对数函数**定义为指数函数的反函数：

$$
y=\log_a x
\quad\Longleftrightarrow\quad
a^y=x,\qquad x>0.
$$

也就是说，$\log_a x$ 回答的问题是“$a$ 的多少次方等于 $x$”。

常用记号：

- $\ln x=\log_e x$，**自然对数**；
- $\lg x=\log_{10} x$，**常用对数**（部分教材记作 $\log x$）；
- $\log_2 x$，在计算机科学与信息论中常用。

由反函数关系：

$$
a^{\log_a x}=x\quad(x>0),
\qquad
\log_a(a^x)=x\quad(x\in\mathbb R).
$$

特别地，

$$
\log_a 1=0,\qquad \log_a a=1.
$$

### 4.2.4 换底公式

任意底数之间的对数可通过下式换算：

$$
\log_a x=\frac{\log_b x}{\log_b a},\qquad a,b>0,\ a,b\ne1,\ x>0.
$$

特别地，

$$
\log_a x=\frac{\ln x}{\ln a}=\frac{\lg x}{\lg a}.
$$

这说明：**不同底的对数只差一个常数因子**，因此微积分中只需对 $\ln x$ 求导，其他底的对数自动获得。

> **例题 4.1** 不用计算器，求 $\log_2 100$ 的近似值（取 $\lg 2\approx 0.301$）。

**解**：由换底公式，

$$
\log_2 100=\frac{\lg 100}{\lg 2}=\frac{2}{0.301}\approx 6.64.
$$

---

## 4.3 指数与对数函数的基本性质

### 4.3.1 定义域与值域

| 函数 | 定义域 | 值域 | 关键点 |
|:---:|:---:|:---:|:---:|
| $a^x\ (a>1)$ | $\mathbb R$ | $(0,+\infty)$ | $(0,1),(1,a)$ |
| $a^x\ (0<a<1)$ | $\mathbb R$ | $(0,+\infty)$ | $(0,1),(1,a)$ |
| $\log_a x\ (a>1)$ | $(0,+\infty)$ | $\mathbb R$ | $(1,0),(a,1)$ |
| $\log_a x\ (0<a<1)$ | $(0,+\infty)$ | $\mathbb R$ | $(1,0),(a,1)$ |

注意：**指数函数恒正**，故 $a^x>0$ 对所有 $x$ 成立；**对数函数只对正数有定义**。

### 4.3.2 单调性

- 当 $a>1$：$a^x$ 严格单调递增，$\log_a x$ 严格单调递增；
- 当 $0<a<1$：$a^x$ 严格单调递减，$\log_a x$ 严格单调递减。

这一性质让我们可以用对数把不等式两边同时取对数而保号（底大于 $1$ 时）。

### 4.3.3 渐近行为

当 $a>1$ 时：

$$
\lim_{x\to+\infty}a^x=+\infty,\qquad
\lim_{x\to-\infty}a^x=0^+,
$$

$$
\lim_{x\to+\infty}\log_a x=+\infty,\qquad
\lim_{x\to 0^+}\log_a x=-\infty.
$$

即 $a^x$ 以 $x$ 轴为水平渐近线，$\log_a x$ 以 $y$ 轴为竖直渐近线。

进一步，对任意 $\alpha>0$ 与 $a>1$：

$$
\lim_{x\to+\infty}\frac{x^\alpha}{a^x}=0,
\qquad
\lim_{x\to+\infty}\frac{\log_a x}{x^\alpha}=0.
$$

这就是“**指数压倒幂函数、幂函数压倒对数**”，记作

$$
\log x\ll x^\alpha\ll a^x\qquad(x\to+\infty).
$$

这一组比较是后续洛必达法则与渐近分析的核心。

### 4.3.4 奇偶性与图像对称

指数函数与对数函数**都不是奇函数也不是偶函数**。

但有两个重要的对称关系：

- $y=a^x$ 与 $y=\log_a x$ 的图像关于直线 $y=x$ 对称（因为互为反函数）；
- $y=a^x$ 与 $y=\left(\frac1a\right)^x=a^{-x}$ 的图像关于 $y$ 轴对称。

### 4.3.5 图像与参数变换

函数

$$
y=A\,a^{B(x-h)}+k
$$

可由 $y=a^x$ 经过以下变换得到：

| 参数 | 作用 |
|:---:|:---|
| $\|A\|$ | 竖直伸缩；$A<0$ 还会上下翻转 |
| $\|B\|$ | 横向伸缩；$B<0$ 还会左右翻转 |
| $h$ | 向右平移 $h$ |
| $k$ | 向上平移 $k$，水平渐近线为 $y=k$ |

> **例题 4.2** 分析函数 $y=3\cdot2^{x-1}-4$ 的渐近线、单调性以及由 $y=2^x$ 得到的变换。

**解**：

- 由 $y=2^x$ 先向右平移 $1$（得到 $2^{x-1}$）；
- 再竖直拉伸为原来的 $3$ 倍（得到 $3\cdot2^{x-1}$）；
- 最后向下平移 $4$。

由于 $3>0$ 且 $2>1$，函数严格单调递增；水平渐近线为 $y=-4$。

---

## 4.4 对数恒等式

对数恒等式不是孤立公式，本质上是**指数运算律在反函数侧的对应表述**。学习时建议先掌握三条核心，再由它们推出其他。

### 4.4.1 三条核心恒等式

对 $x,y>0$，$a>0,\ a\ne1$，$r\in\mathbb R$：

$$
\log_a(xy)=\log_a x+\log_a y,
$$

$$
\log_a\frac{x}{y}=\log_a x-\log_a y,
$$

$$
\log_a(x^r)=r\log_a x.
$$

证明都来自指数运算律。例如令 $u=\log_a x,\ v=\log_a y$，则 $x=a^u,\ y=a^v$，于是 $xy=a^{u+v}$，所以 $\log_a(xy)=u+v=\log_a x+\log_a y$。

### 4.4.2 派生公式

由核心三条可推出：

$$
\log_a\sqrt[n]{x}=\frac1n\log_a x,
$$

$$
\log_{a^n}x=\frac1n\log_a x,
$$

$$
\log_a x\cdot\log_x a=1\quad(x\ne1).
$$

最后一个公式说明 $\log_a x$ 与 $\log_x a$ 互为倒数，这在换底时非常方便。

### 4.4.3 与指数互相消去

$$
a^{\log_a x}=x\quad(x>0),
\qquad
\log_a(a^x)=x\quad(x\in\mathbb R).
$$

更一般地：

$$
b^x=e^{x\ln b},
\qquad
x^r=e^{r\ln x}\quad(x>0).
$$

后一个等式把任意幂统一写成 $e$ 的指数，是微分中处理 $x^r$、$x^x$、$f(x)^{g(x)}$ 等表达式的标准技巧（**对数求导法**）。

### 4.4.4 恒等式证明策略

证明含对数的恒等式时，常用策略：

1. **先写明定义域**：所有出现 $\log_a u$ 的位置都需要 $u>0$；
2. **化为同底**：用换底公式把所有对数写成同一底（通常是 $\ln$）；
3. **把乘除幂转成加减乘**：核心三条公式正反两个方向都要熟练；
4. **令 $u=\log_a x$ 引入变量替换**：把对数问题化为指数问题。

> **例题 4.3** 证明
> $$
> \log_a x\cdot\log_b y=\log_a y\cdot\log_b x,
> $$
> 其中 $a,b,x,y>0$ 且 $a,b\ne1$。

**解**：用换底公式，

$$
\log_a x\cdot\log_b y
=\frac{\ln x}{\ln a}\cdot\frac{\ln y}{\ln b}
=\frac{\ln x\ln y}{\ln a\ln b}.
$$

同理

$$
\log_a y\cdot\log_b x
=\frac{\ln y}{\ln a}\cdot\frac{\ln x}{\ln b}
=\frac{\ln x\ln y}{\ln a\ln b}.
$$

两式相等，证毕。 $\square$

---

## 4.5 指数与对数方程入门

### 4.5.1 基本方程

**指数方程** $a^x=b$（$a>0,\ a\ne1,\ b>0$）的解为

$$
x=\log_a b=\frac{\ln b}{\ln a}.
$$

若 $b\le0$，则方程无解（因为 $a^x>0$）。

**对数方程** $\log_a x=b$（$a>0,\ a\ne1$）的解为

$$
x=a^b.
$$

### 4.5.2 解题策略

1. **化为同底**：例如 $4^x=2^{x+1}$，两边写成 $2$ 的幂；
2. **取对数降幂**：两边取 $\ln$ 把指数移下来；
3. **引入辅助变量**：例如令 $t=a^x$，把 $a^{2x}+a^x-6=0$ 化为二次方程；
4. **使用对数恒等式合并**：例如 $\log_a x+\log_a(x-1)=\log_a 6$ 化为 $\log_a[x(x-1)]=\log_a 6$；
5. **检验定义域**：解出的 $x$ 必须使所有对数的真数为正。

> **例题 4.4** 解方程 $\log_2(x+1)+\log_2(x-1)=3$。

**解**：定义域要求 $x+1>0$ 且 $x-1>0$，即 $x>1$。

由对数恒等式，

$$
\log_2[(x+1)(x-1)]=3,
$$

即 $(x+1)(x-1)=2^3=8$，所以 $x^2=9$，$x=\pm3$。

由定义域 $x>1$，舍去 $x=-3$，得 $x=3$。

### 4.5.3 简单不等式

由单调性：当 $a>1$ 时，

$$
a^x>a^y\Longleftrightarrow x>y,
\qquad
\log_a x>\log_a y\Longleftrightarrow x>y>0.
$$

当 $0<a<1$ 时，不等号方向反转：

$$
a^x>a^y\Longleftrightarrow x<y,
\qquad
\log_a x>\log_a y\Longleftrightarrow 0<x<y.
$$

> **例题 4.5** 解不等式 $\left(\frac12\right)^x>\frac14$。

**解**：写成 $\left(\frac12\right)^x>\left(\frac12\right)^2$。因为底 $\frac12<1$，函数递减，所以

$$
x<2.
$$

---

## 4.6 自然指数 $e^x$ 与自然对数 $\ln x$

虽然任意底 $a>0,\ a\ne1$ 都能定义指数与对数，但微积分里**自然底 $e$** 是首选，因为它的导数与积分公式最简洁。

### 4.6.1 自然指数函数

$y=e^x$ 是定义在 $\mathbb R$ 上、严格递增、值域为 $(0,+\infty)$ 的函数。

幂级数展开：

$$
e^x=\sum_{n=0}^\infty\frac{x^n}{n!}=1+x+\frac{x^2}{2}+\frac{x^3}{6}+\cdots.
$$

这一展开式对所有 $x\in\mathbb R$ 收敛，是后续 Taylor 级数与 Euler 公式 $e^{i\theta}=\cos\theta+i\sin\theta$ 的基础。

### 4.6.2 自然对数函数

$y=\ln x$ 是 $y=e^x$ 在 $\mathbb R$ 上的反函数，定义域为 $(0,+\infty)$，值域为 $\mathbb R$。

基本关系：

$$
e^{\ln x}=x\quad(x>0),
\qquad
\ln(e^x)=x\quad(x\in\mathbb R).
$$

自然对数有一个等价的积分定义：

$$
\ln x=\int_1^x\frac{1}{t}\,dt,\qquad x>0.
$$

这个定义把 $\ln x$ 直接刻画成 $\frac1x$ 的原函数，因此

$$
(\ln x)'=\frac1x.
$$

### 4.6.3 常用底之间的换算

$$
\log_a x=\frac{\ln x}{\ln a},
\qquad
a^x=e^{x\ln a}.
$$

实际工程中经常用到的换算常数：

$$
\ln 2\approx 0.6931,
\qquad
\ln 10\approx 2.3026,
\qquad
\log_2 e\approx 1.4427.
$$

> **例题 4.6** 求 $\lim_{n\to\infty}\left(1+\frac{2}{n}\right)^{3n}$。

**解**：将式子改写为

$$
\left(1+\frac{2}{n}\right)^{3n}
=\left[\left(1+\frac{2}{n}\right)^{n/2}\right]^{6}.
$$

由 $e$ 的定义，

$$
\left(1+\frac{2}{n}\right)^{n/2}\to e^1=e\quad(n\to\infty).
$$

所以原极限等于 $e^6$。

---

## 4.7 与微积分的连接

本章是后续微积分中许多重要结论的前置基础。

### 4.7.1 两个重要极限

后面学习极限时会证明：

$$
\lim_{x\to0}\frac{e^x-1}{x}=1,
\qquad
\lim_{x\to0}\frac{\ln(1+x)}{x}=1.
$$

这两个极限说明在 $x=0$ 附近，

$$
e^x\approx 1+x,
\qquad
\ln(1+x)\approx x.
$$

它们是导数公式 $(e^x)'=e^x$ 和 $(\ln x)'=\frac1x$ 的极限定义形式。

### 4.7.2 导数与积分

基本导数公式：

$$
(e^x)'=e^x,
\qquad
(a^x)'=a^x\ln a,
$$

$$
(\ln x)'=\frac1x\quad(x>0),
\qquad
(\log_a x)'=\frac1{x\ln a}.
$$

相应地，不定积分中会出现：

$$
\int e^x\,dx=e^x+C,
\qquad
\int a^x\,dx=\frac{a^x}{\ln a}+C,
$$

$$
\int\frac1x\,dx=\ln|x|+C.
$$

最后一个公式中的绝对值要特别注意：它使公式同时对 $x>0$ 和 $x<0$ 成立。

### 4.7.3 对数求导法

形如 $y=f(x)^{g(x)}$ 的函数无法直接套用幂法则或指数法则。标准做法是：

$$
\ln y=g(x)\ln f(x),
$$

两边求导后再乘以 $y$。这就是**对数求导法**。

例如对 $y=x^x$（$x>0$）：

$$
\ln y=x\ln x
\ \Rightarrow\
\frac{y'}{y}=\ln x+1
\ \Rightarrow\
y'=x^x(\ln x+1).
$$

### 4.7.4 微分方程中的指数

最简单的微分方程

$$
\frac{dy}{dx}=ky
$$

的通解是 $y=Ce^{kx}$。这说明**“变化率正比于自身”的现象必然导致指数函数**，这是后续 ODE、放射性衰变、复利、生物种群、神经网络中各类指数衰减/增长的统一来源。

---

## 4.8 深度学习应用

对数与指数在现代深度学习中是核心工具，以下介绍三个最常见的场景。

### 4.8.1 Softmax 与交叉熵

分类模型最后一层常用 **Softmax** 把任意实向量 $\mathbf z\in\mathbb R^K$ 映射为概率分布：

$$
\mathrm{softmax}(\mathbf z)_i=\frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}.
$$

对应的损失函数是 **交叉熵**：

$$
\mathcal L=-\sum_{i=1}^K y_i\log\hat p_i.
$$

其中 $y_i$ 是真实分布（通常是 one-hot），$\hat p_i$ 是预测概率。
对 Softmax + 交叉熵的组合，梯度有非常简洁的形式

$$
\frac{\partial \mathcal L}{\partial z_i}=\hat p_i-y_i,
$$

这是对数与指数互为反函数带来的天然简化，使训练在数值上稳定且高效。

### 4.8.2 Log-Sum-Exp 与数值稳定性

直接计算 $\log\sum_j e^{z_j}$ 在 $z_j$ 很大时会溢出，在 $z_j$ 很小时所有 $e^{z_j}$ 都几乎为零导致下溢。标准技巧 **Log-Sum-Exp** 是：

$$
\log\sum_{j=1}^K e^{z_j}=m+\log\sum_{j=1}^K e^{z_j-m},
$$

其中 $m=\max_j z_j$。这使指数内的最大值变为 $0$，既不溢出也不下溢。

这一恒等式直接来自对数运算律：

$$
\log\sum_j e^{z_j}
=\log\left(e^m\sum_j e^{z_j-m}\right)
=m+\log\sum_j e^{z_j-m}.
$$

PyTorch 中的 `torch.logsumexp`、`F.log_softmax` 等函数都是这一思想的实现。

### 4.8.3 对数似然与负对数似然损失

最大似然估计选取参数 $\theta$ 使观测数据出现的概率最大：

$$
\hat\theta=\arg\max_\theta\prod_{i=1}^N p(x_i;\theta).
$$

直接对乘积求导很困难，但取对数后变为求和：

$$
\hat\theta=\arg\max_\theta\sum_{i=1}^N\log p(x_i;\theta).
$$

这就是 **对数似然**。深度学习中通常最小化 **负对数似然损失**：

$$
\mathcal L(\theta)=-\frac1N\sum_{i=1}^N\log p(x_i;\theta).
$$

对数把概率乘积变成求和，让梯度计算可拆分到每个样本；同时把 $[0,1]$ 上的概率映射到 $(-\infty,0]$ 的对数尺度上，避免数值下溢。

### 代码示例：数值稳定的 log-softmax

```python
import torch


def log_softmax_stable(z: torch.Tensor) -> torch.Tensor:
    """数值稳定的 log-softmax 实现。"""
    m = z.max(dim=-1, keepdim=True).values
    z_shift = z - m
    log_sum_exp = torch.log(torch.exp(z_shift).sum(dim=-1, keepdim=True))
    return z_shift - log_sum_exp


z = torch.tensor([1000.0, 1001.0, 1002.0])
print(log_softmax_stable(z))
# tensor([-2.4076, -1.4076, -0.4076])

# 对比直接实现（会溢出为 -inf 或 nan）
# print(torch.log(torch.exp(z) / torch.exp(z).sum()))
```

`m = z.max(...)` 把指数的最大值移到 $0$，是 Log-Sum-Exp 技巧的核心。

---

## 本章小结

1. **指数与对数互为反函数**，对数把指数运算律“乘变加、幂变乘”的等价形式写在反函数侧。
2. **自然底 $e$** 是微积分中最自然的底：$(e^x)'=e^x$，$(\ln x)'=\frac1x$，其他底通过换底公式自动获得。
3. **基本性质**包括定义域（指数为 $\mathbb R$、对数为 $(0,+\infty)$）、单调性、渐近行为以及“指数压倒幂、幂压倒对数”的大小关系。
4. **三条核心恒等式**：$\log(xy)=\log x+\log y$、$\log\frac{x}{y}=\log x-\log y$、$\log x^r=r\log x$；所有其他对数公式都可由它们推出。
5. **方程与不等式**的关键是利用单调性、化同底、引入辅助变量，并始终检验定义域。
6. **应用连接**包括两个重要极限、对数求导法、指数型 ODE，以及深度学习中的 Softmax、交叉熵、Log-Sum-Exp 与负对数似然。

---

## 资料与延伸阅读

- [OpenStax Calculus Volume 1, Section 1.5: Exponential and Logarithmic Functions](https://openstax.org/books/calculus-volume-1/pages/1-5-exponential-and-logarithmic-functions)。重点参考自然底 $e$、对数定义与基本性质。
- [OpenStax Precalculus 2e, Chapter 4: Exponential and Logarithmic Functions](https://openstax.org/books/precalculus-2e/pages/4-introduction-to-exponential-and-logarithmic-functions)。重点参考图像、运算律、方程与应用。
- [Paul's Online Math Notes, Algebra: Exponential and Logarithm Functions](https://tutorial.math.lamar.edu/Classes/Alg/ExpAndLogFcns.aspx)。重点参考化简、解方程与常见误区。
- [Khan Academy: Logarithms](https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:logs)。重点参考对数定义、运算律和换底公式的交互式练习。
- Goodfellow, Bengio, Courville. *Deep Learning*, Chapter 3 & 6。重点参考 Softmax、交叉熵以及 Log-Sum-Exp 的数值稳定性讨论。

---

## 练习题

**1.** ⭐ 化简下列表达式：
   (a) $\log_2 32 + \log_2\frac18$　　(b) $\lg 25 + \lg 4$　　(c) $\log_3 18-\log_3 2$　　(d) $\ln e^5 - e^{\ln 3}$

**2.** ⭐ 用换底公式求下列对数的精确表达：
   (a) $\log_4 8$　　(b) $\log_9 27$　　(c) $\log_2 5\cdot\log_5 2$

**3.** ⭐ 解下列方程：
   (a) $3^{2x-1}=27$　　(b) $\log_5(x-2)=2$　　(c) $4^x-2^{x+1}-8=0$

**4.** ⭐⭐ 证明恒等式 $\log_a x\cdot\log_b y=\log_a y\cdot\log_b x$（$a,b,x,y>0$，$a,b\ne1$），并说明所需要的定义域条件。

**5.** ⭐⭐ 解方程 $\log_2(x+1)+\log_2(x-1)=3$，并验证解的合法性。

**6.** ⭐⭐ 已知 $y=3\cdot 2^{x-1}-4$。求它的水平渐近线、单调性，并说明其图像由 $y=2^x$ 经过哪些变换得到。

**7.** ⭐⭐⭐ 求极限
$$
\lim_{n\to\infty}\left(1+\frac{2}{n}\right)^{3n}.
$$

**8.** ⭐⭐⭐ 设 $\mathbf z=(z_1,\ldots,z_K)\in\mathbb R^K$ 且 $m=\max_i z_i$。证明
$$
\log\sum_{i=1}^K e^{z_i}=m+\log\sum_{i=1}^K e^{z_i-m},
$$
并说明为什么这种形式在数值上比直接计算 $\log\sum_i e^{z_i}$ 更稳定。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.**
(a) $\log_2 32+\log_2\frac18=\log_2\left(32\cdot\frac18\right)=\log_2 4=2$。

(b) $\lg 25+\lg 4=\lg(25\cdot 4)=\lg 100=2$。

(c) $\log_3 18-\log_3 2=\log_3\frac{18}{2}=\log_3 9=2$。

(d) $\ln e^5-e^{\ln 3}=5-3=2$。

---

**2.**
(a) $\log_4 8=\frac{\log_2 8}{\log_2 4}=\frac{3}{2}$。

(b) $\log_9 27=\frac{\log_3 27}{\log_3 9}=\frac{3}{2}$。

(c) $\log_2 5\cdot\log_5 2=\frac{\ln 5}{\ln 2}\cdot\frac{\ln 2}{\ln 5}=1$。

---

**3.**
(a) $27=3^3$，所以 $3^{2x-1}=3^3$，$2x-1=3$，$x=2$。

(b) $\log_5(x-2)=2$ 即 $x-2=5^2=25$，所以 $x=27$（满足 $x-2>0$）。

(c) 令 $t=2^x>0$。由 $4^x=t^2$、$2^{x+1}=2t$，方程化为
$$
t^2-2t-8=0\Rightarrow (t-4)(t+2)=0.
$$
因为 $t>0$，舍去 $t=-2$，得 $t=4$，所以 $2^x=4$，$x=2$。

---

**4.** 定义域条件：$a,b,x,y>0$ 且 $a,b\ne1$。

由换底公式，
$$
\log_a x\cdot\log_b y
=\frac{\ln x}{\ln a}\cdot\frac{\ln y}{\ln b}
=\frac{\ln x\ln y}{\ln a\ln b},
$$
$$
\log_a y\cdot\log_b x
=\frac{\ln y}{\ln a}\cdot\frac{\ln x}{\ln b}
=\frac{\ln x\ln y}{\ln a\ln b}.
$$
两式相等，证毕。 $\square$

---

**5.** 定义域要求 $x+1>0$ 且 $x-1>0$，即 $x>1$。

合并对数：
$$
\log_2[(x+1)(x-1)]=3
\Rightarrow x^2-1=8
\Rightarrow x^2=9
\Rightarrow x=\pm 3.
$$

由 $x>1$ 舍去 $x=-3$，得 $x=3$。

代入原式：$\log_2 4+\log_2 2=2+1=3$，验证通过。

---

**6.** 改写为
$$
y=3\cdot 2^{x-1}-4.
$$

- **水平渐近线**：当 $x\to-\infty$ 时 $2^{x-1}\to 0$，所以 $y\to-4$，水平渐近线为 $y=-4$。
- **单调性**：$3>0$ 且底 $2>1$，所以 $y$ 在 $\mathbb R$ 上严格递增。
- **图像变换**：由 $y=2^x$ 先向右平移 $1$ 得 $2^{x-1}$，再竖直拉伸 $3$ 倍得 $3\cdot 2^{x-1}$，最后向下平移 $4$。

---

**7.** 将式子改写为
$$
\left(1+\frac{2}{n}\right)^{3n}
=\left[\left(1+\frac{2}{n}\right)^{n/2}\right]^6.
$$

由自然底数定义，
$$
\left(1+\frac{2}{n}\right)^{n/2}
=\left[\left(1+\frac{2}{n}\right)^{n/2}\right]\to e\quad(n\to\infty),
$$
所以原极限等于 $e^6$。

---

**8.** 由对数运算律，
$$
\log\sum_{i=1}^K e^{z_i}
=\log\left(e^m\sum_{i=1}^K e^{z_i-m}\right)
=\log e^m+\log\sum_{i=1}^K e^{z_i-m}
=m+\log\sum_{i=1}^K e^{z_i-m}.
$$

数值稳定性的原因：移位后所有 $z_i-m\le 0$，故 $e^{z_i-m}\in(0,1]$，不会上溢；同时至少有一项（对应最大值的那项）等于 $e^0=1$，使求和至少为 $1$，因此对数有定义且不会下溢成 $-\infty$。

直接计算 $\log\sum_i e^{z_i}$ 时，若某个 $z_i$ 很大（如 $1000$），$e^{z_i}$ 会溢出为 $+\infty$；若所有 $z_i$ 都很小（如 $-1000$），$e^{z_i}$ 全部下溢为 $0$，对数变为 $-\infty$。Log-Sum-Exp 通过减去最大值同时避免这两种情形。

</details>

---

## 思考路标（条件反射）

- 看到 $\log_a b = N$ → 等价 $a^N = b$（指数对数互逆）
- 看到 $\log(MN)$ → $\log M + \log N$（积变和）
- 看到 $\log(M/N)$ → $\log M - \log N$
- 看到 $\log_a M^k$ → $k \log_a M$
- 看到换底 → $\log_a b = \log_c b / \log_c a$
- 看到 $\ln$ → 默认底为 $e$
- 看到 $\log e^x$ 或 $\ln e^x$ → 直接化简为 $x$
- 看到 $\log\sum e^{z_i}$ → 想 Log-Sum-Exp 减最大值的数值稳定技巧

## 易错点

1. **$\log_a M^k = k\log_a M$ 仅当 $M > 0$**；$\log(-1)^2 \neq 2\log(-1)$（左边 $=0$，右边无定义）。
2. **$\log_a (M + N) \neq \log_a M + \log_a N$**：对数只对乘积分配，加法不分配。
3. **$\log_a b$ 的换底 $\log_c b / \log_c a$**：不要写成 $\log_c (b/a)$（学生常错）。
4. **Log-Sum-Exp 数值稳定**：直接 $\log\sum e^{z_i}$ 若 $z_i$ 大易上溢；正确：$m = \max z_i$，再算 $m + \log\sum e^{z_i - m}$。
5. **底数限制**：$a > 0, a \neq 1$。$\log_1 b$ 无定义（因 $1^x = 1$ 不能取到其它值）。
