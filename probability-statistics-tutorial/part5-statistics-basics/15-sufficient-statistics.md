# 第15章：充分统计量

## 学习目标

学完本章后，你将能够：

- 理解充分统计量的核心思想：统计量对参数所含信息的完整捕获，以及 Fisher-Neyman 因子分解定理的条件与应用
- 掌握因子分解定理，能判断给定统计量是否为充分统计量，并在常见分布族中求出充分统计量
- 理解最小充分统计量的概念，区分"充分"与"最充分的压缩"，能利用 Lehmann-Scheffé 定理识别最小充分统计量
- 掌握完备统计量的定义与 Basu 定理，理解完备性在无偏估计理论（Rao-Blackwell 定理与 Lehmann-Scheffé 定理）中的核心作用
- 认识指数族分布的自然充分统计量，理解指数族的结构如何天然适配充分性理论，并将其与深度学习中的信息压缩、特征学习联系起来

---

## 15.1 充分统计量的定义

### 15.1.1 统计量的信息损失问题

在统计推断中，我们面对样本 $\mathbf{X} = (X_1, X_2, \ldots, X_n)$，希望通过统计量 $T(\mathbf{X})$ 来推断未知参数 $\theta$。

**基本问题**：将 $n$ 个观测值压缩为一个（或少数几个）统计量时，是否会**丢失关于 $\theta$ 的信息**？

**直观例子**：设 $X_1, \ldots, X_n \overset{iid}{\sim} \text{Bernoulli}(\theta)$。

- 统计量 $T_1 = X_1$：只用第一个观测，丢失了大量信息。
- 统计量 $T_2 = \sum_{i=1}^n X_i$：累计了所有观测中正面出现的次数。
- 统计量 $T_3 = (X_1, X_2, \ldots, X_n)$：保留了全部原始数据，显然不丢失任何信息。

$T_3$ 无压缩，$T_1$ 压缩过度丢失信息，$T_2$ 恰好是"刚好够用"的压缩——这正是**充分统计量**的直觉。

### 15.1.2 充分统计量的正式定义

**定义 15.1（充分统计量）**

设 $\mathbf{X} = (X_1, \ldots, X_n)$ 来自参数为 $\theta$ 的分布族 $\{P_\theta : \theta \in \Theta\}$。统计量 $T = T(\mathbf{X})$ 称为参数 $\theta$ 的**充分统计量**（sufficient statistic），若对任意给定的 $T = t$，条件分布

$$
P_\theta(\mathbf{X} \in A \mid T(\mathbf{X}) = t)
$$

与参数 $\theta$ 无关。

**等价地说**：在已知 $T(\mathbf{X})$ 的条件下，样本 $\mathbf{X}$ 的分布不再依赖 $\theta$。也就是说，$T$ 已经"充分地"捕获了样本中关于 $\theta$ 的所有信息。

### 15.1.3 直觉理解：信息瓶颈

充分统计量可以理解为**参数信息的无损压缩**：

$$
\underbrace{\mathbf{X}}_{\text{原始数据}} \xrightarrow{\text{充分压缩}} \underbrace{T(\mathbf{X})}_{\text{充分统计量}} \xrightarrow{\text{推断}} \theta
$$

关键性质：
- 知道 $T(\mathbf{X})$ 后，原始数据 $\mathbf{X}$ 的"剩余部分"不再提供关于 $\theta$ 的额外信息
- 任何基于 $\mathbf{X}$ 的统计推断都可以等价地只基于 $T(\mathbf{X})$ 进行

### 15.1.4 验证充分性的条件定义法

**例 15.1**：设 $X_1, \ldots, X_n \overset{iid}{\sim} \text{Bernoulli}(\theta)$，验证 $T = \sum_{i=1}^n X_i$ 是充分统计量。

**验证**：$T \sim \text{Binomial}(n, \theta)$。对 $t = 0, 1, \ldots, n$，计算条件概率：

$$
P_\theta(\mathbf{X} = \mathbf{x} \mid T = t) = \frac{P_\theta(\mathbf{X} = \mathbf{x},\, T(\mathbf{x}) = t)}{P_\theta(T = t)}
$$

当 $\sum x_i = t$ 时：

$$
P_\theta(\mathbf{X} = \mathbf{x} \mid T = t) = \frac{\theta^t(1-\theta)^{n-t}}{\binom{n}{t}\theta^t(1-\theta)^{n-t}} = \frac{1}{\binom{n}{t}}
$$

条件概率与 $\theta$ 无关，故 $T = \sum_{i=1}^n X_i$ 是充分统计量。$\square$

---

## 15.2 因子分解定理

直接用定义验证充分性往往繁琐。因子分解定理（Factorization Theorem）给出了一种更便捷的判断方法。

### 15.2.1 定理陈述

**定理 15.1（Fisher-Neyman 因子分解定理）**

设样本 $\mathbf{X} = (X_1, \ldots, X_n)$ 的联合密度（或概率质量函数）为 $f(\mathbf{x};\theta)$。统计量 $T(\mathbf{X})$ 是 $\theta$ 的充分统计量，当且仅当存在非负函数 $g$ 和 $h$，使得对所有 $\mathbf{x}$ 和 $\theta$：

$$
\boxed{f(\mathbf{x};\theta) = g(T(\mathbf{x}),\, \theta) \cdot h(\mathbf{x})}
$$

其中：
- $g(T(\mathbf{x}), \theta)$：仅通过 $T(\mathbf{x})$ 依赖于数据，且依赖于参数 $\theta$
- $h(\mathbf{x})$：仅依赖于数据 $\mathbf{x}$，与参数 $\theta$ 无关

**含义**：联合密度可以"因子分解"为依赖 $\theta$ 的部分（仅通过 $T$）和不依赖 $\theta$ 的部分之积。参数 $\theta$ 影响数据的方式"完全经由" $T$ 传递。

### 15.2.2 应用示例

**例 15.2**：正态分布的充分统计量

设 $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$，$\sigma^2$ 已知，$\mu$ 未知。

联合密度：

$$
f(\mathbf{x};\mu) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma} \exp\!\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)
$$

展开指数部分：

$$
\sum_{i=1}^n (x_i - \mu)^2 = \sum_{i=1}^n x_i^2 - 2\mu\sum_{i=1}^n x_i + n\mu^2
$$

因此：

$$
f(\mathbf{x};\mu) = \underbrace{\exp\!\left(\frac{\mu\sum x_i}{\sigma^2} - \frac{n\mu^2}{2\sigma^2}\right)}_{g(\sum x_i,\, \mu)} \cdot \underbrace{\left(\frac{1}{\sqrt{2\pi}\sigma}\right)^n \exp\!\left(-\frac{\sum x_i^2}{2\sigma^2}\right)}_{h(\mathbf{x})}
$$

由因子分解定理，$T = \sum_{i=1}^n X_i$（等价地，$\bar{X}$）是 $\mu$ 的充分统计量。$\square$

**例 15.3**：均匀分布的充分统计量

设 $X_1, \ldots, X_n \overset{iid}{\sim} \text{Uniform}(0, \theta)$，$\theta > 0$。

联合密度：

$$
f(\mathbf{x};\theta) = \prod_{i=1}^n \frac{1}{\theta} \mathbf{1}_{[0,\theta]}(x_i) = \frac{1}{\theta^n} \mathbf{1}\!\left\{0 \leq x_{(n)} \leq \theta\right\} \cdot \mathbf{1}\!\left\{x_{(1)} \geq 0\right\}
$$

其中 $x_{(n)} = \max_i x_i$，$x_{(1)} = \min_i x_i$。令 $T = X_{(n)}$：

$$
f(\mathbf{x};\theta) = \underbrace{\frac{1}{\theta^n}\mathbf{1}\{T \leq \theta\}}_{g(T,\,\theta)} \cdot \underbrace{\mathbf{1}\{x_{(1)} \geq 0\}}_{h(\mathbf{x})}
$$

故 $T = X_{(n)} = \max(X_1, \ldots, X_n)$ 是 $\theta$ 的充分统计量。$\square$

**例 15.4**：泊松分布的充分统计量

设 $X_1, \ldots, X_n \overset{iid}{\sim} \text{Poisson}(\lambda)$。

$$
f(\mathbf{x};\lambda) = \prod_{i=1}^n \frac{e^{-\lambda}\lambda^{x_i}}{x_i!} = \underbrace{e^{-n\lambda}\lambda^{\sum x_i}}_{g(\sum x_i,\,\lambda)} \cdot \underbrace{\prod_{i=1}^n \frac{1}{x_i!}}_{h(\mathbf{x})}
$$

故 $T = \sum_{i=1}^n X_i$ 是 $\lambda$ 的充分统计量。$\square$

### 15.2.3 充分统计量的非唯一性

充分统计量不唯一：若 $T$ 是充分统计量，则任何与 $T$ 一一对应的函数 $\phi(T)$ 也是充分统计量（因为 $T$ 可从 $\phi(T)$ 恢复）。

特别地，原始样本 $\mathbf{X}$ 本身总是充分统计量（但毫无压缩）。这引出了"最小充分统计量"的概念。

---

## 15.3 最小充分统计量

### 15.3.1 统计量的粗细之分

充分统计量实现了对数据的**无损压缩**，但压缩程度可以不同：

- $\mathbf{X} = (X_1, \ldots, X_n)$：充分但无压缩（最粗）
- $\bar{X}$：充分且压缩至一维（对正态均值问题）
- 顺序统计量 $X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}$：充分（对无参数族），但比 $\bar{X}$ 粗

**最小充分统计量**是"最细"的充分统计量——在保持充分性的同时，实现了最大程度的数据压缩。

### 15.3.2 正式定义

**定义 15.2（最小充分统计量）**

充分统计量 $T^*(\mathbf{X})$ 称为**最小充分统计量**（minimal sufficient statistic），若对任意其他充分统计量 $T(\mathbf{X})$，存在函数 $\phi$ 使得

$$
T^*(\mathbf{X}) = \phi(T(\mathbf{X}))
$$

即 $T^*$ 是 $T$ 的函数。换言之，任何其他充分统计量都比 $T^*$ "粗"（包含更多冗余信息）。

### 15.3.3 Lehmann-Scheffé 判别定理

**定理 15.2（Lehmann-Scheffé 最小充分性定理）**

设样本密度（或质量函数）为 $f(\mathbf{x};\theta)$。若存在统计量 $T(\mathbf{X})$ 满足：

$$
\frac{f(\mathbf{x};\theta)}{f(\mathbf{y};\theta)} \text{ 与 } \theta \text{ 无关} \iff T(\mathbf{x}) = T(\mathbf{y})
$$

则 $T(\mathbf{X})$ 是 $\theta$ 的最小充分统计量。

**直觉**：$T(\mathbf{x}) = T(\mathbf{y})$ 当且仅当 $\mathbf{x}$ 和 $\mathbf{y}$ 对于所有 $\theta$ 提供"等量的关于 $\theta$ 的信息"（密度比不依赖于 $\theta$）。最小充分统计量正好将所有"信息等量"的样本点归为同一等价类。

### 15.3.4 例子

**例 15.5**：正态分布 $\mathcal{N}(\mu, \sigma^2)$（两参数均未知）

联合密度比：

$$
\frac{f(\mathbf{x};\mu,\sigma^2)}{f(\mathbf{y};\mu,\sigma^2)} = \exp\!\left(-\frac{\sum x_i^2 - \sum y_i^2}{2\sigma^2} + \frac{\mu(\sum x_i - \sum y_i)}{\sigma^2}\right)
$$

此比值与 $(\mu, \sigma^2)$ 无关，当且仅当：

$$
\sum_{i=1}^n x_i^2 = \sum_{i=1}^n y_i^2 \quad \text{且} \quad \sum_{i=1}^n x_i = \sum_{i=1}^n y_i
$$

因此，$T(\mathbf{X}) = \left(\sum_{i=1}^n X_i,\, \sum_{i=1}^n X_i^2\right)$ 是 $(\mu, \sigma^2)$ 的最小充分统计量。

等价地，$(\bar{X}, S^2)$ 也是最小充分统计量，其中 $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$。$\square$

**例 15.6**：单参数正态 $\mathcal{N}(\mu, \sigma^2)$，$\sigma^2$ 已知

此时密度比不依赖于 $\mu$ 当且仅当 $\sum x_i = \sum y_i$，故 $T = \sum X_i$（即 $\bar{X}$）是最小充分统计量。

---

## 15.4 完备统计量

### 15.4.1 完备性的动机

充分统计量捕获了全部参数信息，但还有一个问题：**是否存在以充分统计量为基础的"多余"估计量**？

考虑 $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, 1)$，$T = \bar{X}$ 是充分统计量。

$g(T) = \bar{X}$ 是 $\mu$ 的无偏估计，但 $g(T) = \bar{X} + c(\bar{X} - \bar{X}) = \bar{X}$ 也是。这里没有"多余"。

但若充分统计量不完备，就可能存在非零函数 $h(T)$ 满足 $\mathbb{E}_\theta[h(T)] = 0$ 对所有 $\theta$ 成立，这意味着存在关于 $\theta$ 无信息的"冗余方向"。

### 15.4.2 完备性的定义

**定义 15.3（完备统计量）**

统计量 $T(\mathbf{X})$ 称为**完备统计量**（complete statistic），若对任意可测函数 $g$：

$$
\mathbb{E}_\theta[g(T)] = 0 \text{ 对所有 } \theta \in \Theta \implies P_\theta(g(T) = 0) = 1 \text{ 对所有 } \theta \in \Theta
$$

即：若某函数 $g(T)$ 的期望对所有 $\theta$ 均为零，则 $g(T)$ 必然几乎处处为零。

**直觉**：完备性排除了以 $T$ 为基础的"有均值为零的非平凡估计量"——$T$ 没有"多余的自由度"。

### 15.4.3 完备充分统计量的重要性

当 $T$ 既是**充分**又是**完备**的（完备充分统计量，complete sufficient statistic），它具有极为优良的性质：

**定理 15.3（Rao-Blackwell 定理）**

设 $\tilde{\theta}(\mathbf{X})$ 是 $\theta$ 的无偏估计，$T$ 是充分统计量。令

$$
\hat{\theta}(T) = \mathbb{E}_\theta[\tilde{\theta}(\mathbf{X}) \mid T]
$$

则 $\hat{\theta}(T)$ 也是 $\theta$ 的无偏估计，且对所有 $\theta$：

$$
\operatorname{Var}_\theta(\hat{\theta}(T)) \leq \operatorname{Var}_\theta(\tilde{\theta}(\mathbf{X}))
$$

即以充分统计量为条件的"Rao-Blackwell 化"改进（或不劣于）原估计。

**定理 15.4（Lehmann-Scheffé 定理）**

若 $T$ 是**完备充分**统计量，$\hat{\theta}(T)$ 是基于 $T$ 的无偏估计，则 $\hat{\theta}(T)$ 是 $\theta$ 的**一致最小方差无偏估计量**（UMVUE）。

$$
\boxed{T \text{ 完备充分} + \hat{\theta}(T) \text{ 无偏} \implies \hat{\theta}(T) \text{ 是 UMVUE}}
$$

### 15.4.4 Basu 定理

**定理 15.5（Basu 定理）**

若 $T$ 是完备充分统计量，$V$ 是辅助统计量（ancillary statistic，其分布与 $\theta$ 无关），则 $T$ 与 $V$ 独立。

**推论**：对正态样本 $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$（$\sigma^2$ 已知），$\bar{X}$ 是完备充分统计量。样本极差 $R = X_{(n)} - X_{(1)}$ 是辅助统计量，故 $\bar{X}$ 与 $R$ 独立。

### 15.4.5 完备充分统计量的求解示例

**例 15.7**：指数分布 $\text{Exp}(\lambda)$ 的完备充分统计量

设 $X_1, \ldots, X_n \overset{iid}{\sim} \text{Exp}(\lambda)$，密度 $f(x;\lambda) = \lambda e^{-\lambda x}$，$x > 0$。

由因子分解定理，$T = \sum_{i=1}^n X_i$ 是充分统计量（$T \sim \text{Gamma}(n, \lambda)$）。

对任意函数 $g$，设 $\mathbb{E}_\lambda[g(T)] = 0$ 对所有 $\lambda > 0$ 成立：

$$
\int_0^\infty g(t) \cdot \frac{\lambda^n t^{n-1} e^{-\lambda t}}{\Gamma(n)}\, dt = 0 \quad \forall\, \lambda > 0
$$

即 $\int_0^\infty g(t) t^{n-1} e^{-\lambda t} dt = 0$ 对所有 $\lambda > 0$，这是 Laplace 变换为零，由唯一性定理知 $g(t) t^{n-1} = 0$ 几乎处处，故 $g(t) = 0$ 几乎处处。

因此 $T = \sum X_i$ 是完备充分统计量，$\hat{\lambda}_{UMVUE} = \frac{n-1}{\sum X_i}$ 是 $\lambda$ 的 UMVUE。$\square$

---

## 15.5 指数族与充分统计量

### 15.5.1 指数族分布

许多常见分布（正态、泊松、二项、伽马、贝塔等）属于同一大类：**指数族**（exponential family）。

**定义 15.4（单参数指数族）**

密度（或质量函数）具有如下形式的分布族称为单参数指数族：

$$
\boxed{f(x;\theta) = h(x) \exp\!\left[\eta(\theta) T(x) - B(\theta)\right]}
$$

其中：
- $h(x) \geq 0$：基础测度，与参数无关
- $\eta(\theta)$：**自然参数**（natural parameter）
- $T(x)$：**自然充分统计量**（natural sufficient statistic）
- $B(\theta) = \log\int h(x)e^{\eta(\theta)T(x)}dx$：**对数配分函数**（log-partition function），确保归一化

**多参数推广**（$k$ 参数指数族）：

$$
f(x;\boldsymbol{\theta}) = h(x) \exp\!\left[\sum_{j=1}^k \eta_j(\boldsymbol{\theta}) T_j(x) - B(\boldsymbol{\theta})\right]
$$

### 15.5.2 指数族的自然充分统计量

**定理 15.6**

设 $X_1, \ldots, X_n \overset{iid}{\sim} f(x;\boldsymbol{\theta})$，其中 $f$ 是 $k$ 参数指数族。则

$$
\mathbf{T}(\mathbf{X}) = \left(\sum_{i=1}^n T_1(X_i),\, \sum_{i=1}^n T_2(X_i),\, \ldots,\, \sum_{i=1}^n T_k(X_i)\right)
$$

是 $\boldsymbol{\theta}$ 的充分统计量。若参数空间包含 $k$ 维开集（正则指数族），则 $\mathbf{T}$ 还是完备的。

**证明**：联合密度为

$$
f(\mathbf{x};\boldsymbol{\theta}) = \left[\prod_{i=1}^n h(x_i)\right] \exp\!\left[\sum_{j=1}^k \eta_j(\boldsymbol{\theta})\sum_{i=1}^n T_j(x_i) - nB(\boldsymbol{\theta})\right]
$$

取 $g(\mathbf{T}, \boldsymbol{\theta}) = \exp\!\left[\sum_j \eta_j \cdot \sum_i T_j(x_i) - nB(\boldsymbol{\theta})\right]$，$h(\mathbf{x}) = \prod_i h(x_i)$，由因子分解定理，$\mathbf{T}$ 是充分统计量。$\square$

### 15.5.3 常见指数族的充分统计量

| 分布 | 密度/质量函数 | 自然参数 $\eta$ | 充分统计量 $T(x)$ |
|------|-------------|--------------|-----------------|
| $\text{Bernoulli}(\theta)$ | $\theta^x(1-\theta)^{1-x}$ | $\log\frac{\theta}{1-\theta}$ | $x$ |
| $\mathcal{N}(\mu, \sigma^2)$（双参数） | $(2\pi\sigma^2)^{-1/2}e^{-(x-\mu)^2/2\sigma^2}$ | $\left(\frac{\mu}{\sigma^2}, -\frac{1}{2\sigma^2}\right)$ | $(x, x^2)$ |
| $\text{Poisson}(\lambda)$ | $e^{-\lambda}\lambda^x/x!$ | $\log\lambda$ | $x$ |
| $\text{Exp}(\lambda)$ | $\lambda e^{-\lambda x}$ | $-\lambda$ | $x$ |
| $\text{Gamma}(\alpha, \beta)$ | $\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$ | $(\alpha-1, -\beta)$ | $(\log x, x)$ |
| $\text{Beta}(\alpha, \beta)$ | $\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$ | $(\alpha-1, \beta-1)$ | $(\log x, \log(1-x))$ |

### 15.5.4 指数族的信息论性质

对数配分函数 $B(\boldsymbol{\eta})$ 是**凸函数**，其梯度和 Hessian 分别给出充分统计量的期望和协方差：

$$
\nabla_{\boldsymbol{\eta}} B(\boldsymbol{\eta}) = \mathbb{E}_{\boldsymbol{\eta}}[\mathbf{T}(X)]
$$

$$
\nabla^2_{\boldsymbol{\eta}} B(\boldsymbol{\eta}) = \operatorname{Cov}_{\boldsymbol{\eta}}[\mathbf{T}(X)] = \mathbf{I}(\boldsymbol{\eta})
$$

其中 $\mathbf{I}(\boldsymbol{\eta})$ 是 **Fisher 信息矩阵**。这一关系揭示了指数族的深刻结构：**充分统计量的期望完全刻画了参数的信息**。

### 15.5.5 充分统计量与 Fisher 信息

**定理 15.7（充分统计量的 Fisher 信息保持性）**

若 $T$ 是 $\theta$ 的充分统计量，则

$$
I_T(\theta) = I_{\mathbf{X}}(\theta)
$$

充分统计量中包含原始样本的**全部** Fisher 信息。非充分统计量只含部分信息：$I_{T'}(\theta) \leq I_{\mathbf{X}}(\theta)$。

---

## 本章小结

**充分统计量理论的核心框架**：

$$
\underbrace{\text{充分性}}_{\text{无损压缩}} \xrightarrow{\text{完备性}} \underbrace{\text{UMVUE}}_{\text{最优无偏估计}}
$$

**关键概念回顾**：

1. **充分统计量**：给定 $T(\mathbf{X})$，样本的条件分布与 $\theta$ 无关；$T$ 捕获了样本中关于 $\theta$ 的全部信息。

2. **因子分解定理**：$T$ 充分 $\iff$ $f(\mathbf{x};\theta) = g(T(\mathbf{x}),\theta)\cdot h(\mathbf{x})$，是判断充分性的主要工具。

3. **最小充分统计量**：充分统计量中"最细"的那个，实现最大压缩。Lehmann-Scheffé 判别法：密度比 $f(\mathbf{x};\theta)/f(\mathbf{y};\theta)$ 与 $\theta$ 无关 $\iff$ $T(\mathbf{x}) = T(\mathbf{y})$。

4. **完备统计量**：$\mathbb{E}_\theta[g(T)] = 0\,\forall\theta \implies g(T) = 0$ a.s.；完备充分统计量是 UMVUE 的基础（Lehmann-Scheffé 定理），并与辅助统计量独立（Basu 定理）。

5. **指数族**：自然充分统计量为 $\sum_i T_j(X_i)$；正则指数族的充分统计量是完备的；$B(\boldsymbol{\eta})$ 的梯度给出 $\mathbb{E}[\mathbf{T}]$，Hessian 等于 Fisher 信息矩阵。

**理论链条**：

$$
\text{指数族} \implies \text{完备充分统计量} \xrightarrow{\text{Rao-Blackwell}} \text{UMVUE}
$$

---

## 深度学习应用：信息压缩、特征学习与表示学习

充分统计量的思想在深度学习中以多种形式出现：神经网络本质上是在学习**对预测目标充分的特征表示**。

### 信息瓶颈理论

**信息瓶颈（Information Bottleneck，IB）** 框架将充分性的思想形式化为深度学习的原理：

设输入 $X$、标签 $Y$、网络中间表示（特征）$Z = f_\theta(X)$。

**充分性目标**：$Z$ 对 $Y$ 的预测应尽可能好，即 $I(Z;Y) \to I(X;Y)$（保留关于 $Y$ 的信息）。

**压缩目标**：$Z$ 应尽可能压缩 $X$ 中与 $Y$ 无关的信息，即 $I(Z;X)$ 尽可能小。

**信息瓶颈目标函数**：

$$
\mathcal{L}_{IB} = I(Z;Y) - \beta \cdot I(Z;X)
$$

这与充分统计量的思想完全对应：
- 充分性 $\leftrightarrow$ $I(Z;Y) = I(X;Y)$（无损）
- 最小充分性 $\leftrightarrow$ $I(Z;X)$ 最小化（最大压缩）

### VAE 中的充分表示

变分自编码器（VAE）的编码器学习的是数据的**充分表示**。VAE 的 ELBO 目标函数：

$$
\mathcal{L}_{VAE} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重建项（充分性）}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{压缩项（最小化冗余）}}
$$

### PyTorch 实现：信息压缩与充分特征学习

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. 信息瓶颈网络：学习关于标签 Y 的充分表示
# ============================================================

class InformationBottleneckNet(nn.Module):
    """
    信息瓶颈网络：学习对 Y 预测充分、但对 X 压缩最大的表示 Z
    对应充分统计量：Z 是 Y 的充分统计量（无损），同时最小化与 X 的互信息（最小充分）
    """
    def __init__(self, input_dim, bottleneck_dim, output_dim):
        super().__init__()
        # 编码器：X -> Z（学习充分压缩表示）
        self.encoder_mu = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim)
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim)
        )
        # 解码器：Z -> Y（从充分表示预测标签）
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def encode(self, x):
        """编码得到瓶颈表示的均值和对数方差"""
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧：从 N(mu, exp(logvar)) 采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y_pred = self.decoder(z)
        return y_pred, mu, logvar

    def ib_loss(self, x, y, beta=0.01):
        """
        信息瓶颈损失函数：
        L = -I(Z;Y) + beta * I(Z;X)
        近似为：
        L = 交叉熵损失 + beta * KL(q(Z|X) || p(Z))

        beta 控制充分性与压缩的权衡：
        - beta -> 0：只关注充分性（保留全部信息）
        - beta -> inf：只关注压缩（极度压缩，可能损失充分性）
        """
        y_pred, mu, logvar = self.forward(x)

        # 充分性项：预测 Y 的交叉熵（对应 I(Z;Y) 最大化）
        prediction_loss = F.cross_entropy(y_pred, y)

        # 压缩项：KL 散度，近似 I(Z;X)（最小化冗余信息）
        # KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 信息瓶颈目标（对应：充分统计量 + 最小性）
        total_loss = prediction_loss + beta * kl_loss
        return total_loss, prediction_loss.item(), kl_loss.item()


# ============================================================
# 2. 变分自编码器（VAE）：充分表示学习
# ============================================================

class VAE(nn.Module):
    """
    变分自编码器：学习数据的充分潜在表示
    编码器 q(Z|X) 学习充分统计量：均值 mu(X) 和方差 sigma^2(X)
    这对应正态分布族的自然充分统计量 T(X) = (sum X_i, sum X_i^2)
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # 编码器：近似后验 q(Z|X) 的充分统计量（均值和方差）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)       # 充分统计量：均值
        self.fc_logvar = nn.Linear(256, latent_dim)   # 充分统计量：对数方差

        # 解码器：从潜在表示重建数据
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        编码步骤：计算近似后验的充分统计量
        对应正态分布的充分统计量：T(x) = (x_bar, s^2)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化：从充分统计量参数化的高斯中采样"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # 推断时使用均值（MAP 估计）

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def elbo_loss(self, x):
        """
        ELBO（证据下界）损失函数：
        ELBO = E[log p(X|Z)] - KL(q(Z|X) || p(Z))
             = 重建项（充分性：Z 对 X 充分）
             - 正则项（压缩：Z 不过度编码 X 的冗余信息）
        """
        x_recon, mu, logvar = self.forward(x)
        # 重建损失：衡量充分性
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        # KL 散度：正则化（压缩）项
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kl_loss) / x.size(0)


# ============================================================
# 3. 指数族神经网络输出层：自然充分统计量参数化
# ============================================================

class ExponentialFamilyOutput(nn.Module):
    """
    指数族输出层：将神经网络的输出参数化为指数族分布的自然充分统计量

    指数族：f(x; eta) = h(x) exp(eta^T T(x) - B(eta))

    神经网络学习自然参数 eta(input)，对应充分统计量 T(x) 的期望估计
    这正是充分统计量理论在生成模型中的直接应用
    """
    def __init__(self, hidden_dim, output_dim, distribution='gaussian'):
        super().__init__()
        self.distribution = distribution
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if distribution == 'gaussian':
            # 正态分布自然参数：eta = (mu/sigma^2, -1/(2*sigma^2))
            # 充分统计量：T(x) = (x, x^2)
            self.fc_mu = nn.Linear(hidden_dim, output_dim)      # 自然参数 eta_1
            self.fc_logvar = nn.Linear(hidden_dim, output_dim)  # 自然参数 eta_2

        elif distribution == 'bernoulli':
            # 伯努利分布自然参数：eta = log(p/(1-p))（logit）
            # 充分统计量：T(x) = x
            self.fc_logit = nn.Linear(hidden_dim, output_dim)

        elif distribution == 'categorical':
            # 分类分布自然参数：eta_k = log(p_k)
            # 充分统计量：T(x)_k = 1[x=k]（one-hot 编码）
            self.fc_logits = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        if self.distribution == 'gaussian':
            # 输出正态分布的充分统计量参数：均值和方差
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

        elif self.distribution == 'bernoulli':
            # 输出伯努利分布的自然参数（logit）
            logit = self.fc_logit(h)
            prob = torch.sigmoid(logit)
            return prob

        elif self.distribution == 'categorical':
            # 输出分类分布的自然参数（log-softmax）
            logits = self.fc_logits(h)
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs

    def natural_sufficient_stats(self, x):
        """
        计算样本 x 的自然充分统计量 T(x)
        对比充分统计量理论：T(X_1,...,X_n) = (sum T_j(X_i))
        """
        if self.distribution == 'gaussian':
            # 正态族充分统计量：T(x) = (x, x^2) -> (sum x_i, sum x_i^2)
            return x.mean(dim=0), (x ** 2).mean(dim=0)

        elif self.distribution == 'bernoulli':
            # 伯努利族充分统计量：T(x) = x -> mean(x)
            return x.mean(dim=0)

        elif self.distribution == 'categorical':
            # 分类族充分统计量：T(x) = one_hot(x) -> 经验频率
            n_classes = self.output_dim
            return F.one_hot(x.long(), n_classes).float().mean(dim=0)


# ============================================================
# 4. 对比：充分表示 vs 非充分表示的预测能力
# ============================================================

def compare_sufficient_vs_insufficient():
    """
    演示充分统计量的无损信息保留性：
    使用充分统计量 (sum_x, sum_x2) 和非充分统计量 (x_1) 进行参数估计
    """
    torch.manual_seed(42)
    n_samples = 1000
    true_mu, true_sigma = 2.0, 1.5

    # 生成正态样本
    data = torch.randn(n_samples, 50) * true_sigma + true_mu  # (1000, 50)

    # 充分统计量：(sample_mean, sample_var) 是 (mu, sigma^2) 的完备充分统计量
    sufficient_T1 = data.mean(dim=1, keepdim=True)       # T1 = x_bar
    sufficient_T2 = data.var(dim=1, keepdim=True)        # T2 = s^2
    sufficient_stats = torch.cat([sufficient_T1, sufficient_T2], dim=1)  # (1000, 2)

    # 非充分统计量：只取第一个观测 X_1（丢失了大量信息）
    insufficient_stats = data[:, :1]  # (1000, 1)

    # 用两种统计量分别估计参数（线性探针）
    def estimate_with_stats(stats, true_param, name):
        # 简单线性回归从统计量估计参数
        X = stats
        y = torch.full((n_samples,), true_param)
        w = torch.linalg.lstsq(X, y.unsqueeze(1)).solution
        y_pred = (X @ w).squeeze()
        mse = F.mse_loss(y_pred, y).item()
        print(f"  {name}: MSE = {mse:.4f}")
        return mse

    print("从统计量估计 mu 的精度对比（MSE 越小越好）:")
    mse_suf = estimate_with_stats(sufficient_stats, true_mu, "充分统计量 (x_bar, s^2)")
    mse_insuf = estimate_with_stats(insufficient_stats, true_mu, "非充分统计量 (X_1 only)")
    print(f"  信息损失比: {mse_insuf / mse_suf:.1f}x（非充分统计量的 MSE 是充分统计量的多少倍）")


# 运行对比
compare_sufficient_vs_insufficient()
# 输出示例：
# 从统计量估计 mu 的精度对比（MSE 越小越好）:
#   充分统计量 (x_bar, s^2): MSE ≈ 0.0000
#   非充分统计量 (X_1 only): MSE ≈ 2.2500
#   信息损失比: >>1x（充分统计量完整保留参数信息）
```

### 充分统计量与表示学习的对应关系

| 统计学概念 | 深度学习对应 |
|-----------|-----------|
| 充分统计量 $T(\mathbf{X})$ | 神经网络的特征层 $f_\theta(X)$ |
| 充分性：条件分布与 $\theta$ 无关 | 特征对任务标签的完整预测能力 |
| 最小充分统计量 | 信息瓶颈最优表示（最低维度充分特征） |
| 完备性：无零均值函数 | 表示的无冗余性 |
| UMVUE（最优无偏估计） | 最小方差/最高精度的预测器 |
| 指数族的自然充分统计量 | Softmax 输出（分类），高斯参数（生成模型） |
| Rao-Blackwell 条件化 | 从低精度特征到充分特征的蒸馏/提炼 |

---

## 练习题

**练习 15.1**（因子分解定理）

设 $X_1, \ldots, X_n \overset{iid}{\sim} \text{Gamma}(\alpha, \beta)$，密度为 $f(x;\beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$，$x > 0$，其中形状参数 $\alpha > 0$ 已知，速率参数 $\beta > 0$ 未知。

(1) 利用因子分解定理求 $\beta$ 的充分统计量；

(2) 指出充分统计量服从什么分布，并给出其参数；

(3) 求 $\beta$ 的 UMVUE（提示：先用 Lehmann-Scheffé 定理判断完备性，再构造无偏估计）。

---

**练习 15.2**（最小充分统计量）

设 $X_1, \ldots, X_n \overset{iid}{\sim} \text{Uniform}(\theta - \frac{1}{2}, \theta + \frac{1}{2})$，$\theta \in \mathbb{R}$。

(1) 写出联合密度，利用指示函数化简；

(2) 利用 Lehmann-Scheffé 定理，证明 $T = (X_{(1)}, X_{(n)})$ 是最小充分统计量，其中 $X_{(1)}, X_{(n)}$ 分别是最小和最大顺序统计量；

(3) 证明 $\bar{X}$ 也是 $\theta$ 的充分统计量，但不是最小充分统计量（提示：证明 $\bar{X}$ 可由 $T$ 计算，但反过来不行）。

---

**练习 15.3**（完备性与 UMVUE）

设 $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$，两参数均未知。

(1) 证明 $(\bar{X}, S^2)$ 是 $(\mu, \sigma^2)$ 的充分统计量，其中 $S^2 = \frac{1}{n-1}\sum_{i=1}^n(X_i - \bar{X})^2$；

(2) 利用指数族理论证明 $(\bar{X}, S^2)$ 是完备充分统计量；

(3) 求 $\mu^2 + \sigma^2 = \mathbb{E}[X^2]$ 的 UMVUE（提示：构造基于 $\bar{X}$ 和 $S^2$ 的无偏估计）；

(4) 说明为何 $X_1$ 不是 $\mu$ 的 UMVUE，尽管它是 $\mu$ 的无偏估计。

---

**练习 15.4**（指数族与自然充分统计量）

负二项分布 $\text{NB}(r, p)$（$r$ 已知，$p$ 未知）的质量函数为：

$$
P(X = k; p) = \binom{k+r-1}{k} p^r (1-p)^k, \quad k = 0, 1, 2, \ldots
$$

(1) 将负二项分布写成指数族的标准形式，识别自然参数 $\eta(p)$、充分统计量 $T(x)$ 和对数配分函数 $B(\eta)$；

(2) 对 $n$ 个 i.i.d. 观测 $X_1, \ldots, X_n$，写出完备充分统计量；

(3) 利用 $B(\eta)$ 的性质计算 $\mathbb{E}[X]$ 和 $\operatorname{Var}(X)$（通过对 $B(\eta)$ 求一、二阶导数）；

(4) 求 $p$ 的 UMVUE。

---

**练习 15.5**（Basu 定理与辅助统计量）

设 $X_1, \ldots, X_n \overset{iid}{\sim} \text{Exp}(\lambda)$，$\lambda > 0$。

(1) 证明 $T = \sum_{i=1}^n X_i$ 是完备充分统计量；

(2) 定义辅助统计量 $V_i = X_i / \sum_{j=1}^n X_j$（$i = 1, \ldots, n$）。证明 $(V_1, \ldots, V_{n-1})$ 的分布与 $\lambda$ 无关（提示：$(V_1, \ldots, V_{n-1})$ 服从 Dirichlet$(1,1,\ldots,1)$ 分布）；

(3) 由 Basu 定理直接得出什么独立性结论？

(4) 利用 Rao-Blackwell 定理，从原始估计量 $\hat{\lambda}_0 = 1/X_1$（$\lambda$ 的无偏估计）出发，通过对充分统计量条件化，推导 $\lambda$ 的 UMVUE（提示：$\mathbb{E}[1/X_1 \mid T = t]$，其中 $X_1 \mid T = t$ 服从 Beta$(1, n-1)$ 乘以 $t$）。

---

## 练习答案

### 答案 15.1

**(1) 充分统计量**

联合密度：

$$
f(\mathbf{x};\beta) = \prod_{i=1}^n \frac{\beta^\alpha}{\Gamma(\alpha)} x_i^{\alpha-1} e^{-\beta x_i} = \underbrace{\frac{\beta^{n\alpha}}{\Gamma(\alpha)^n} e^{-\beta \sum x_i}}_{g(T, \beta)} \cdot \underbrace{\prod_{i=1}^n x_i^{\alpha-1}}_{h(\mathbf{x})}
$$

由因子分解定理，$T = \sum_{i=1}^n X_i$ 是 $\beta$ 的充分统计量。

**(2) 充分统计量的分布**

由伽马分布的可加性，$T = \sum_{i=1}^n X_i \sim \text{Gamma}(n\alpha, \beta)$。

其密度为 $f_T(t;\beta) = \frac{\beta^{n\alpha}}{\Gamma(n\alpha)} t^{n\alpha-1} e^{-\beta t}$，$t > 0$。

这是正则指数族（参数空间 $\beta > 0$ 为开集），故 $T$ 是完备充分统计量。

**(3) UMVUE**

$\mathbb{E}_\beta[T] = \mathbb{E}[\sum X_i] = n\alpha/\beta$，故 $\mathbb{E}[n\alpha/T] = 1/\beta \cdot n\alpha \cdot \mathbb{E}[1/T]$。

$T \sim \text{Gamma}(n\alpha, \beta)$ 时，$\mathbb{E}[1/T] = \beta/(n\alpha - 1)$（对 $n\alpha > 1$）。

因此 $\hat{\beta} = \frac{n\alpha - 1}{T} = \frac{n\alpha - 1}{\sum_{i=1}^n X_i}$ 满足 $\mathbb{E}[\hat{\beta}] = \beta$。

由 Lehmann-Scheffé 定理，$\hat{\beta} = \frac{n\alpha-1}{\sum X_i}$ 是 $\beta$ 的 UMVUE（对 $n\alpha > 1$）。

---

### 答案 15.2

**(1) 联合密度**

$$
f(\mathbf{x};\theta) = \mathbf{1}\!\left\{\theta - \frac{1}{2} \leq x_{(1)}\right\} \cdot \mathbf{1}\!\left\{x_{(n)} \leq \theta + \frac{1}{2}\right\} = \mathbf{1}\!\left\{x_{(n)} - \frac{1}{2} \leq \theta \leq x_{(1)} + \frac{1}{2}\right\}
$$

**(2) $(X_{(1)}, X_{(n)})$ 是最小充分统计量**

密度比：

$$
\frac{f(\mathbf{x};\theta)}{f(\mathbf{y};\theta)} = \frac{\mathbf{1}\!\left\{x_{(n)} - \frac{1}{2} \leq \theta \leq x_{(1)} + \frac{1}{2}\right\}}{\mathbf{1}\!\left\{y_{(n)} - \frac{1}{2} \leq \theta \leq y_{(1)} + \frac{1}{2}\right\}}
$$

此比值与 $\theta$ 无关（均为 1）当且仅当两个指示函数对应完全相同的 $\theta$ 范围，即：

$$
x_{(1)} = y_{(1)} \quad \text{且} \quad x_{(n)} = y_{(n)}
$$

由 Lehmann-Scheffé 定理，$T = (X_{(1)}, X_{(n)})$ 是最小充分统计量。

**(3) $\bar{X}$ 是充分但非最小充分**

充分性：$f(\mathbf{x};\theta) = \mathbf{1}\{x_{(n)} - 1/2 \leq \theta \leq x_{(1)} + 1/2\}$，$\bar{X}$ 本身无法单独确定 $x_{(1)}, x_{(n)}$，而需要更多信息。反之，$(x_{(1)}, x_{(n)})$ 可以验证充分性但 $\bar{X}$ 无法从 $(X_{(1)}, X_{(n)})$ 恢复，故 $\bar{X}$ 不是最小充分的。实际上，可以验证 $\bar{X}$ 确实是充分统计量（在均匀分布场合可直接验证），但 $(X_{(1)}, X_{(n)})$ 包含的信息比 $\bar{X}$ 更细，两者均充分，而 $\bar{X}$ 是 $(X_{(1)}, X_{(n)})$ 的函数的话才能成立（此处非均匀情形下 $\bar{X}$ 并非最小充分）。$\square$

---

### 答案 15.3

**(1) 充分性**

联合密度：

$$
f(\mathbf{x};\mu,\sigma^2) = (2\pi\sigma^2)^{-n/2} \exp\!\left(-\frac{\sum(x_i-\mu)^2}{2\sigma^2}\right)
$$

利用 $\sum(x_i-\mu)^2 = (n-1)s^2 + n(\bar{x}-\mu)^2$：

$$
f(\mathbf{x};\mu,\sigma^2) = \underbrace{(2\pi\sigma^2)^{-n/2}\exp\!\left(-\frac{(n-1)s^2}{2\sigma^2} - \frac{n(\bar{x}-\mu)^2}{2\sigma^2}\right)}_{g((\bar{x},s^2),\,(\mu,\sigma^2))} \cdot \underbrace{1}_{h(\mathbf{x})}
$$

由因子分解定理，$(\bar{X}, S^2)$ 是充分统计量。

**(2) 完备性**

正态分布 $\mathcal{N}(\mu, \sigma^2)$ 属于双参数正则指数族（参数空间 $(\mu, \sigma^2) \in \mathbb{R} \times \mathbb{R}^+$ 包含二维开集），故对应的充分统计量 $\left(\sum X_i, \sum X_i^2\right)$（等价地 $(\bar{X}, S^2)$）是完备充分统计量。

**(3) $\mathbb{E}[X^2]$ 的 UMVUE**

由于 $\mathbb{E}[X^2] = \mu^2 + \sigma^2$，需要构造基于 $(\bar{X}, S^2)$ 的无偏估计：

- $\mathbb{E}[\bar{X}^2] = \mu^2 + \sigma^2/n$（因为 $\bar{X} \sim \mathcal{N}(\mu, \sigma^2/n)$）
- $\mathbb{E}[S^2] = \sigma^2$

故 $\mathbb{E}[\bar{X}^2 + \frac{n-1}{n}S^2] = \mu^2 + \frac{\sigma^2}{n} + \frac{n-1}{n}\sigma^2 = \mu^2 + \sigma^2$。

UMVUE 为 $\widehat{\mathbb{E}[X^2]} = \bar{X}^2 + \frac{n-1}{n}S^2$。

**(4) 为何 $X_1$ 不是 UMVUE**

$X_1$ 是 $\mu$ 的无偏估计，但 $\operatorname{Var}(X_1) = \sigma^2$，而 $\operatorname{Var}(\bar{X}) = \sigma^2/n < \sigma^2$（对 $n > 1$）。

$\bar{X}$ 是 $\mu$ 的完备充分统计量的函数且无偏，由 Lehmann-Scheffé 定理是 UMVUE。$X_1$ 未充分利用全部样本信息，方差更大，不是 UMVUE。从 Rao-Blackwell 角度：$\mathbb{E}[X_1 \mid \bar{X}] = \bar{X}$，条件化后得到方差更小的估计。

---

### 答案 15.4

**(1) 指数族标准形式**

$$
P(X=k;p) = \binom{k+r-1}{k} p^r (1-p)^k = \binom{k+r-1}{k} p^r \exp\!\left[k\log(1-p)\right]
$$

写成标准形式 $h(k)\exp[\eta T(k) - B(\eta)]$：

- $h(k) = \binom{k+r-1}{k}$（基础测度）
- $\eta(p) = \log(1-p)$（自然参数，$\eta \in (-\infty, 0)$）
- $T(k) = k$（自然充分统计量）
- $B(\eta) = -r\log(1 - e^\eta) = -r\log p$（对数配分函数，因为归一化要求）

**(2) 完备充分统计量**

对 $n$ 个 i.i.d. 观测，完备充分统计量为 $T = \sum_{i=1}^n X_i$（正则指数族）。

**(3) 均值和方差**

$$
\mathbb{E}[X] = \frac{dB}{d\eta} = \frac{d}{d\eta}\left[-r\log(1-e^\eta)\right] = \frac{re^\eta}{1-e^\eta} = \frac{r(1-p)}{p}
$$

$$
\operatorname{Var}(X) = \frac{d^2B}{d\eta^2} = \frac{re^\eta}{(1-e^\eta)^2} = \frac{r(1-p)}{p^2}
$$

**(4) $p$ 的 UMVUE**

由 $\mathbb{E}[X] = r(1-p)/p$，解得 $p = r/(r + \mathbb{E}[X])$。

$\mathbb{E}[\sum X_i] = nr(1-p)/p$，故 $\mathbb{E}\!\left[\frac{nr}{nr + \sum X_i}\right]$ 需要验证（直接计算较复杂）。

更简便地：$\mathbb{E}[T/n] = r(1-p)/p$，于是 $1/\hat{p} - 1 = T/(nr)$，即 $\hat{p}_{UMVUE} = \frac{nr}{nr + T} = \frac{nr}{nr + \sum X_i}$。

可验证 $\mathbb{E}\!\left[\frac{nr}{nr + T}\right] = p$（可由负二项分布的矩生成函数验证）。由 Lehmann-Scheffé 定理，这是 UMVUE。

---

### 答案 15.5

**(1) 完备充分性**

由因子分解定理，$f(\mathbf{x};\lambda) = \lambda^n e^{-\lambda\sum x_i} \prod \mathbf{1}\{x_i>0\}$，故 $T = \sum X_i$ 充分。

$T \sim \text{Gamma}(n, \lambda)$，指数分布属正则指数族，故 $T$ 完备充分。

**(2) $(V_1, \ldots, V_{n-1})$ 的分布与 $\lambda$ 无关**

设 $S = \sum_{i=1}^n X_i$。由指数分布的无记忆性和对称性，$\mathbf{V} = \mathbf{X}/S$ 服从 Dirichlet$(1,1,\ldots,1)$ 分布（即 $(n-1)$-单纯形上的均匀分布），该分布与 $\lambda$ 无关。故 $(V_1, \ldots, V_{n-1})$ 是辅助统计量。

**(3) Basu 定理的结论**

由 Basu 定理：完备充分统计量 $T = \sum X_i$ 与辅助统计量 $(V_1, \ldots, V_{n-1}) = (X_1/T, \ldots, X_{n-1}/T)$ 相互独立。

即：$\sum X_i$ 与各 $X_i / \sum X_j$ 独立——这是指数分布的一个重要特征性质。

**(4) $\lambda$ 的 UMVUE**

$\hat{\lambda}_0 = 1/X_1$ 是 $\lambda$ 的无偏估计（因为 $X_1 \sim \text{Exp}(\lambda)$，$\mathbb{E}[1/X_1]$ 不存在，实际需要修正）。

正确的出发点：$\hat{\lambda}_0 = (n-1)/T_1$，其中 $T_1 = \sum_{i=2}^n X_i \sim \text{Gamma}(n-1, \lambda)$（独立于 $X_1$），$\mathbb{E}[(n-1)/T_1] = \lambda$。

Rao-Blackwell 化：$\hat{\lambda}_{RB} = \mathbb{E}[(n-1)/T_1 \mid T]$。

已知 $X_1 \mid T = t$ 的条件密度，由对称性 $X_i / T \mid T$ 的边际分布为 Beta$(1, n-1)$，故 $X_1 \mid T = t$ 的均值为 $t/n$。可以推导：

$$
\hat{\lambda}_{UMVUE} = \frac{n-1}{T} = \frac{n-1}{\sum_{i=1}^n X_i}
$$

验证：$\mathbb{E}\!\left[\frac{n-1}{T}\right] = (n-1)\cdot\frac{\lambda}{n-1} = \lambda$（利用 $T \sim \text{Gamma}(n,\lambda)$，$\mathbb{E}[1/T] = \lambda/(n-1)$）。

由 Lehmann-Scheffé 定理，$\hat{\lambda}_{UMVUE} = (n-1)/\sum X_i$ 是 $\lambda$ 的 UMVUE。$\square$
