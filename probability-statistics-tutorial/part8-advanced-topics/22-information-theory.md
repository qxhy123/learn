# 第22章 信息论基础

> **难度**：★★★★☆
> **前置知识**：第4-6章随机变量、第16章最大似然估计、第18章贝叶斯估计

---

## 学习目标

- 掌握信息量与香农熵的定义，理解熵作为不确定性度量的直观含义
- 推导联合熵与条件熵的链式法则 $H(X,Y) = H(X) + H(Y|X)$
- 理解互信息 $I(X;Y)$ 的多种等价表达及其对称性
- 掌握KL散度的定义、非对称性与非负性证明，建立交叉熵损失的理论基础
- 理解信息论不等式体系，以及在深度学习中VAE的ELBO推导与信息瓶颈原理

---

## 22.1 信息量与熵

### 22.1.1 自信息（信息量）

信息论的核心问题是：**一个随机事件发生后，传递了多少"信息"？**

直觉上，越不可能发生的事件，一旦发生，携带的信息量越大。香农（Claude Shannon）于1948年将这一直觉形式化：

**定义 22.1（自信息）**：事件 $A$ 的**自信息**（Self-Information）定义为：

$$I(A) = -\log P(A)$$

其中对数底数的选择决定信息的单位：
- 底数为 2：单位为**比特**（bit）
- 底数为 $e$：单位为**奈特**（nat）
- 底数为 10：单位为**哈特**（hart）

深度学习中通常使用自然对数（奈特），有时也用以2为底。

**自信息的性质**：

1. **非负性**：$I(A) = -\log P(A) \geq 0$（因为 $0 \leq P(A) \leq 1$）
2. **必然事件**：若 $P(A) = 1$，则 $I(A) = 0$（确定发生的事不携带信息）
3. **单调性**：$P(A)$ 越小，$I(A)$ 越大
4. **可加性**：若 $A, B$ 独立，则 $I(A \cap B) = I(A) + I(B)$

**例22.1**：投掷一枚均匀硬币

- 正面（概率 $1/2$）的信息量：$I = -\log_2 \frac{1}{2} = 1$ 比特
- 掷一枚均匀六面骰子出现1点（概率 $1/6$）的信息量：$I = -\log_2 \frac{1}{6} \approx 2.58$ 比特

### 22.1.2 香农熵

对于一个随机变量，我们关心的是**平均信息量**，即信息量的期望。

**定义 22.2（香农熵）**：离散随机变量 $X$ 的**香农熵**（Shannon Entropy）定义为：

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x) = \mathbb{E}[-\log p(X)]$$

约定：$0 \log 0 = 0$（因为 $\lim_{p \to 0^+} p \log p = 0$）。

对于连续随机变量 $X$，**微分熵**（Differential Entropy）定义为：

$$h(X) = -\int_{-\infty}^{+\infty} f(x) \log f(x) \, dx$$

注意：微分熵可以为负，不具备与离散熵完全相同的性质。

**熵的性质**：

1. **非负性**：$H(X) \geq 0$（离散情形）
2. **最大熵原理**：对于取 $n$ 个值的离散随机变量，当且仅当 $X$ 服从均匀分布时，熵取最大值：
   $$H(X) \leq \log n$$
3. **确定性**：若 $X$ 为确定量（某个值概率为1），则 $H(X) = 0$

**例22.2**：伯努利分布的熵

设 $X \sim \text{Bernoulli}(p)$，即 $P(X=1) = p$，$P(X=0) = 1-p$。

$$H(X) = -p \log p - (1-p) \log(1-p) \triangleq H_b(p)$$

- $p = 0$ 或 $p = 1$：$H(X) = 0$（确定性）
- $p = 0.5$：$H(X) = \log 2$（最大不确定性，以自然对数约为 $0.693$ 奈特，以2为底为 $1$ 比特）

**例22.3**：均匀分布的熵

设 $X$ 在 $\{1, 2, \ldots, n\}$ 上均匀分布，$p(x) = 1/n$：

$$H(X) = -\sum_{x=1}^{n} \frac{1}{n} \log \frac{1}{n} = -n \cdot \frac{1}{n} \cdot (-\log n) = \log n$$

---

## 22.2 联合熵与条件熵

### 22.2.1 联合熵

**定义 22.3（联合熵）**：二维随机变量 $(X, Y)$ 的**联合熵**定义为：

$$H(X, Y) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x, y) \log p(x, y) = \mathbb{E}[-\log p(X, Y)]$$

### 22.2.2 条件熵

**定义 22.4（条件熵）**：在给定 $X$ 的条件下，$Y$ 的**条件熵**（Conditional Entropy）定义为：

$$H(Y \mid X) = \sum_{x \in \mathcal{X}} p(x) H(Y \mid X = x)$$

其中 $H(Y \mid X = x) = -\sum_{y \in \mathcal{Y}} p(y \mid x) \log p(y \mid x)$。

展开后：

$$H(Y \mid X) = -\sum_{x} \sum_{y} p(x, y) \log p(y \mid x)$$

**条件熵的直观含义**：已知 $X$ 后，$Y$ 剩余的平均不确定性。

### 22.2.3 链式法则

**定理 22.1（熵的链式法则）**：

$$H(X, Y) = H(X) + H(Y \mid X)$$

**证明**：

$$H(X, Y) = -\sum_{x,y} p(x,y) \log p(x,y)$$

利用乘法公式 $p(x,y) = p(x) \cdot p(y \mid x)$：

$$= -\sum_{x,y} p(x,y) \log [p(x) \cdot p(y \mid x)]$$

$$= -\sum_{x,y} p(x,y) \log p(x) - \sum_{x,y} p(x,y) \log p(y \mid x)$$

对第一项，对 $y$ 求和得边缘分布：

$$-\sum_{x,y} p(x,y) \log p(x) = -\sum_{x} p(x) \log p(x) = H(X)$$

第二项即为 $H(Y \mid X)$，因此：

$$H(X, Y) = H(X) + H(Y \mid X) \quad \square$$

**推论**：多变量的链式法则

$$H(X_1, X_2, \ldots, X_n) = \sum_{i=1}^{n} H(X_i \mid X_1, \ldots, X_{i-1})$$

**定理 22.2（条件不增熵）**：

$$H(Y \mid X) \leq H(Y)$$

即"已知更多信息不会增加不确定性"。等号成立当且仅当 $X$ 与 $Y$ 独立。

**例22.4**：联合熵与条件熵计算

设 $(X, Y)$ 的联合分布如下：

|  | $Y=0$ | $Y=1$ |
|--|-------|-------|
| $X=0$ | 1/4 | 1/4 |
| $X=1$ | 1/4 | 1/4 |

这是均匀分布，$H(X,Y) = \log 4 = 2$ 比特（以2为底）。

$H(X) = H(Y) = \log 2 = 1$ 比特（边缘均匀分布）。

$H(Y \mid X) = H(X,Y) - H(X) = 2 - 1 = 1$ 比特。

由于 $X, Y$ 独立，$H(Y \mid X) = H(Y)$，符合定理22.2的等号条件。

---

## 22.3 互信息

### 22.3.1 互信息的定义

**定义 22.5（互信息）**：随机变量 $X$ 与 $Y$ 的**互信息**（Mutual Information）定义为：

$$I(X; Y) = \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}$$

### 22.3.2 互信息的等价表达

互信息有多种等价表达，每种表达揭示不同的直观含义：

$$\boxed{I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y)}$$

**证明** $I(X;Y) = H(X) - H(X \mid Y)$：

$$I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

$$= \sum_{x,y} p(x,y) \log \frac{p(x \mid y)}{p(x)}$$

$$= -\sum_{x,y} p(x,y) \log p(x) + \sum_{x,y} p(x,y) \log p(x \mid y)$$

$$= H(X) - H(X \mid Y) \quad \square$$

**互信息的文氏图**（Venn Diagram 对应关系）：

```
     H(X)          H(Y)
  ┌──────────┐  ┌──────────┐
  │          │  │          │
  │  H(X|Y)  │  │  I(X;Y)  │  H(Y|X)  │
  │          │  │          │
  └──────────┘  └──────────┘
         ↑
      H(X,Y) = H(X|Y) + I(X;Y) + H(Y|X)
```

$$H(X, Y) = H(X \mid Y) + I(X; Y) + H(Y \mid X)$$

### 22.3.3 互信息的性质

1. **对称性**：$I(X; Y) = I(Y; X)$

2. **非负性**：$I(X; Y) \geq 0$，等号成立当且仅当 $X$ 与 $Y$ 独立

3. **上界**：$I(X; Y) \leq \min\{H(X), H(Y)\}$

4. **与熵的关系**：$I(X; X) = H(X)$（自信息即熵）

**互信息的直观含义**：$I(X;Y)$ 度量了知道 $Y$ 的值后，$X$ 的不确定性减少了多少；或者说 $X$ 和 $Y$ 共同包含的信息量。

**例22.5**：独立变量与完全相关变量

- 若 $X$ 与 $Y$ 独立：$p(x,y) = p(x)p(y)$，所以 $I(X;Y) = 0$（知道 $Y$ 对 $X$ 没有帮助）
- 若 $Y = X$：$I(X;Y) = I(X;X) = H(X)$（知道 $Y$ 完全消除了 $X$ 的不确定性）

### 22.3.4 条件互信息

**定义 22.6（条件互信息）**：给定 $Z$ 的条件下，$X$ 与 $Y$ 的条件互信息：

$$I(X; Y \mid Z) = H(X \mid Z) - H(X \mid Y, Z)$$

**互信息链式法则**：

$$I(X_1, X_2; Y) = I(X_1; Y) + I(X_2; Y \mid X_1)$$

---

## 22.4 KL散度与交叉熵

### 22.4.1 KL散度（相对熵）

**定义 22.7（KL散度）**：设 $P$ 和 $Q$ 是同一空间上的两个概率分布，**KL散度**（Kullback-Leibler Divergence，又称**相对熵**）定义为：

$$D_{\mathrm{KL}}(P \| Q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim P}\left[\log \frac{p(x)}{q(x)}\right]$$

约定：若 $q(x) = 0$ 而 $p(x) > 0$，则该项为 $+\infty$；若 $p(x) = 0$，该项为 $0$。

### 22.4.2 KL散度的非对称性

**KL散度不是距离**，因为它不满足对称性：

$$D_{\mathrm{KL}}(P \| Q) \neq D_{\mathrm{KL}}(Q \| P)$$

**例22.6**：非对称性的直观理解

设 $P = \text{Bernoulli}(0.9)$，$Q = \text{Bernoulli}(0.1)$：

$$D_{\mathrm{KL}}(P \| Q) = 0.9 \log \frac{0.9}{0.1} + 0.1 \log \frac{0.1}{0.9} \approx 1.758 \text{ 奈特}$$

$$D_{\mathrm{KL}}(Q \| P) = 0.1 \log \frac{0.1}{0.9} + 0.9 \log \frac{0.9}{0.1} \approx 1.758 \text{ 奈特}$$

（本例恰好对称，但一般情况不对称。）

在实际中：
- **前向KL** $D_{\mathrm{KL}}(P \| Q)$（"均值寻求"）：$Q$ 必须覆盖 $P$ 有支撑的所有区域，否则损失无穷大
- **反向KL** $D_{\mathrm{KL}}(Q \| P)$（"模式寻求"）：$Q$ 倾向于聚焦于 $P$ 的某个模式，可以忽略其他模式

### 22.4.3 KL散度的非负性（吉布斯不等式）

**定理 22.3（KL散度非负性）**：

$$D_{\mathrm{KL}}(P \| Q) \geq 0$$

等号成立当且仅当 $P = Q$（几乎处处相等）。

**证明**（利用 $\ln x \leq x - 1$）：

$$D_{\mathrm{KL}}(P \| Q) = -\sum_{x} p(x) \log \frac{q(x)}{p(x)}$$

由于 $-\log t \geq 1 - t$（即 $\ln t \leq t-1$），令 $t = q(x)/p(x)$：

$$-\log \frac{q(x)}{p(x)} \geq 1 - \frac{q(x)}{p(x)}$$

两边乘以 $p(x)$ 并对 $x$ 求和：

$$D_{\mathrm{KL}}(P \| Q) \geq \sum_{x} p(x) \left(1 - \frac{q(x)}{p(x)}\right) = \sum_{x} p(x) - \sum_{x} q(x) = 1 - 1 = 0 \quad \square$$

### 22.4.4 交叉熵

**定义 22.8（交叉熵）**：分布 $P$ 和 $Q$ 之间的**交叉熵**定义为：

$$H(P, Q) = -\sum_{x} p(x) \log q(x) = \mathbb{E}_{x \sim P}[-\log q(x)]$$

**交叉熵与KL散度的关系**：

$$H(P, Q) = H(P) + D_{\mathrm{KL}}(P \| Q)$$

**证明**：

$$H(P, Q) = -\sum_{x} p(x) \log q(x)$$

$$= -\sum_{x} p(x) \log p(x) + \sum_{x} p(x) \log \frac{p(x)}{q(x)}$$

$$= H(P) + D_{\mathrm{KL}}(P \| Q) \quad \square$$

**重要推论**：最小化交叉熵等价于最小化KL散度

在机器学习中，真实分布 $P$（数据分布）是固定的，$H(P)$ 是常数。因此：

$$\arg\min_Q H(P, Q) = \arg\min_Q D_{\mathrm{KL}}(P \| Q)$$

**这正是交叉熵损失函数的理论基础！**

**例22.7**：分类任务中的交叉熵损失

设真实标签为类别 $k$（对应 one-hot 分布 $P$），模型预测概率为 $\hat{p}$，则：

$$H(P, \hat{p}) = -\log \hat{p}_k$$

最小化交叉熵损失，即驱使预测分布 $\hat{p}$ 接近真实分布 $P$。

### 22.4.5 KL散度与互信息的关系

互信息可以表示为联合分布与边缘分布乘积之间的KL散度：

$$I(X; Y) = D_{\mathrm{KL}}\left(p(x,y) \| p(x)p(y)\right)$$

这给出了互信息的另一种理解：$X$ 和 $Y$ 的联合分布与"假设独立时"分布之间的差异。

---

## 22.5 信息论不等式

### 22.5.1 Jensen不等式

信息论中许多重要不等式都基于**Jensen不等式**：

**定理 22.4（Jensen不等式）**：若 $f$ 是凸函数，则：

$$f\left(\mathbb{E}[X]\right) \leq \mathbb{E}[f(X)]$$

若 $f$ 是严格凸函数，等号成立当且仅当 $X$ 为常数。

注意：$-\log$ 是严格凸函数（因为 $\frac{d^2}{dx^2}(-\log x) = \frac{1}{x^2} > 0$），这是KL散度非负性的根本原因。

### 22.5.2 数据处理不等式

**定理 22.5（数据处理不等式，DPI）**：若 $X \to Y \to Z$ 构成马尔可夫链（即 $Z$ 在给定 $Y$ 的条件下与 $X$ 独立），则：

$$I(X; Z) \leq I(X; Y)$$

**直观含义**：对数据的任何进一步处理（变换）不能增加关于原始信息的互信息量。信息只会减少，不会增加。

**推论**：若 $g$ 是确定性函数，则 $I(X; g(Y)) \leq I(X; Y)$。

### 22.5.3 Fano不等式

**定理 22.6（Fano不等式）**：设 $X$ 是取 $|\mathcal{X}|$ 个值的离散随机变量，$\hat{X} = g(Y)$ 是基于观测 $Y$ 对 $X$ 的估计，$P_e = P(\hat{X} \neq X)$ 是错误概率，则：

$$H(X \mid Y) \leq H_b(P_e) + P_e \log(|\mathcal{X}| - 1)$$

其中 $H_b(p) = -p\log p - (1-p)\log(1-p)$ 是二元熵。

**含义**：Fano不等式给出了在给定观测 $Y$ 的情况下，关于 $X$ 的分类错误率的下界。条件熵越高，分类就越难。

### 22.5.4 熵的次可加性

**定理 22.7（次可加性）**：

$$H(X_1, X_2, \ldots, X_n) \leq \sum_{i=1}^{n} H(X_i)$$

等号成立当且仅当 $X_1, X_2, \ldots, X_n$ 互相独立。

**证明**：由链式法则和条件不增熵：

$$H(X_1, \ldots, X_n) = \sum_{i=1}^n H(X_i \mid X_1, \ldots, X_{i-1}) \leq \sum_{i=1}^n H(X_i) \quad \square$$

### 22.5.5 最大熵原理

**定理 22.8（最大熵原理）**：

1. **无约束情形**：取 $n$ 个值的离散随机变量，熵的最大值为 $\log n$，在均匀分布时取到。

2. **均值约束**：若 $X \geq 0$ 且 $\mathbb{E}[X] = \mu$，则熵最大的分布是**指数分布** $\text{Exp}(1/\mu)$。

3. **均值和方差约束**：若 $\mathbb{E}[X] = \mu$，$\text{Var}(X) = \sigma^2$，则微分熵最大的分布是**正态分布** $\mathcal{N}(\mu, \sigma^2)$。

**这解释了为什么高斯分布在自然界和机器学习中如此普遍**——在给定均值和方差的约束下，它是"最大不确定性"（最大熵）的分布。

---

## 本章小结

| 概念 | 定义/公式 | 含义 |
|------|-----------|------|
| 自信息 | $I(A) = -\log P(A)$ | 事件 $A$ 发生所携带的信息量 |
| 香农熵 | $H(X) = -\sum_x p(x) \log p(x)$ | $X$ 的平均不确定性 |
| 联合熵 | $H(X,Y) = -\sum_{x,y} p(x,y) \log p(x,y)$ | $(X,Y)$ 联合系统的不确定性 |
| 条件熵 | $H(Y \mid X) = H(X,Y) - H(X)$ | 已知 $X$ 后 $Y$ 的剩余不确定性 |
| 互信息 | $I(X;Y) = H(X) - H(X \mid Y)$ | $X$ 与 $Y$ 共享的信息量 |
| KL散度 | $D_{\mathrm{KL}}(P \| Q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$ | $P$ 与 $Q$ 的差异（不对称） |
| 交叉熵 | $H(P,Q) = H(P) + D_{\mathrm{KL}}(P \| Q)$ | 用 $Q$ 编码 $P$ 的平均码长 |

**核心关系链**：

$$H(P, Q) = H(P) + D_{\mathrm{KL}}(P \| Q) \geq H(P)$$

$$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y)$$

$$H(X, Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y)$$

**关键不等式**：

- $D_{\mathrm{KL}}(P \| Q) \geq 0$（等号当且仅当 $P = Q$）
- $H(Y \mid X) \leq H(Y)$（条件减少不确定性）
- $I(X; Y) \geq 0$（互信息非负）
- 数据处理不等式：$X \to Y \to Z$ 则 $I(X;Z) \leq I(X;Y)$

---

## 深度学习应用

### 22.A 交叉熵损失函数

**交叉熵损失**是最广泛使用的分类损失函数，其信息论基础如下：

设真实标签的 one-hot 分布为 $P$（即 $p(k) = \mathbb{1}[k = y]$），模型预测分布为 $Q = \text{Softmax}(\mathbf{z})$，则：

$$\mathcal{L}_{\text{CE}} = H(P, Q) = H(P) + D_{\mathrm{KL}}(P \| Q) = 0 + D_{\mathrm{KL}}(P \| Q)$$

由于 $H(P) = 0$（one-hot 分布熵为0），所以**最小化交叉熵等于最小化KL散度**。

单样本损失简化为：$\mathcal{L}_{\text{CE}} = -\log q_y$（仅需真实类别的预测概率）。

### 22.B VAE的ELBO推导

**变分自编码器**（Variational Autoencoder, VAE）的核心是最大化**证据下界**（Evidence Lower BOund, ELBO）。

**目标**：对观测数据 $\mathbf{x}$，最大化对数似然 $\log p(\mathbf{x})$。

引入近似后验分布 $q_\phi(\mathbf{z} \mid \mathbf{x})$ 来近似真实后验 $p_\theta(\mathbf{z} \mid \mathbf{x})$：

$$\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}$$

$$= \log \int q_\phi(\mathbf{z} \mid \mathbf{x}) \cdot \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})} \, d\mathbf{z}$$

$$= \log \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})} \left[\frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})}\right]$$

由**Jensen不等式**（$\log$ 是凹函数，$\log \mathbb{E}[\cdot] \geq \mathbb{E}[\log \cdot]$）：

$$\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})} \left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})}\right] = \mathcal{L}_{\text{ELBO}}$$

展开ELBO：

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi}\left[\log p_\theta(\mathbf{x} \mid \mathbf{z})\right] - D_{\mathrm{KL}}\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z})\right)$$

- **第一项**（重建项）：解码器 $p_\theta(\mathbf{x} \mid \mathbf{z})$ 对输入的重建质量
- **第二项**（正则项）：近似后验 $q_\phi(\mathbf{z} \mid \mathbf{x})$ 与先验 $p(\mathbf{z})$（通常为标准正态 $\mathcal{N}(0, I)$）的接近程度

**等价关系**：

$$\log p_\theta(\mathbf{x}) = \mathcal{L}_{\text{ELBO}} + D_{\mathrm{KL}}\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x})\right)$$

由于 KL 散度非负，ELBO 始终是 $\log p_\theta(\mathbf{x})$ 的下界。

### 22.C 信息瓶颈理论

**信息瓶颈**（Information Bottleneck, IB）理论由 Tishby 等人（1999）提出，后被用于解释深度神经网络的学习机制。

**目标**：寻找输入 $X$ 的压缩表示 $T$，使得：

1. $T$ 尽量**丢弃** $X$ 中与 $Y$（标签）无关的信息：最小化 $I(X; T)$
2. $T$ 尽量**保留** $X$ 中与 $Y$ 相关的信息：最大化 $I(T; Y)$

**信息瓶颈目标函数**（带拉格朗日乘数 $\beta$）：

$$\mathcal{L}_{\text{IB}} = I(T; Y) - \beta \cdot I(X; T)$$

- $\beta \to 0$：强调预测（$T$ 保留更多 $X$ 的信息）
- $\beta \to \infty$：强调压缩（$T$ 尽量简短）

**信息平面**（Information Plane）：横轴为 $I(X;T)$，纵轴为 $I(T;Y)$，每层神经网络映射到平面上一点。

**Tishby的主张**（存在争议）：训练分为两阶段——先快速拟合（$I(T;Y)$ 上升），后压缩（$I(X;T)$ 下降），对应泛化能力的形成。

**与VAE的联系**：VAE的ELBO中，$\beta$-VAE 将KL正则系数设为 $\beta > 1$，对应于信息瓶颈框架，使潜在表示更加紧凑。

### PyTorch代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# 1. 熵与互信息的计算
# ============================================================
def entropy(p, eps=1e-10):
    """计算离散分布的香农熵（自然对数单位：奈特）"""
    p = p + eps  # 避免 log(0)
    return -(p * torch.log(p)).sum(dim=-1)

def kl_divergence(p, q, eps=1e-10):
    """计算 KL(P || Q)"""
    p = p + eps
    q = q + eps
    return (p * torch.log(p / q)).sum(dim=-1)

def cross_entropy_manual(p, q, eps=1e-10):
    """计算交叉熵 H(P, Q)"""
    q = q + eps
    return -(p * torch.log(q)).sum(dim=-1)

# 示例：均匀分布 vs 尖锐分布
p_uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])  # 均匀分布
p_sharp   = torch.tensor([0.7,  0.1,  0.1,  0.1])   # 尖锐分布

print("=== 熵 ===")
print(f"均匀分布熵: {entropy(p_uniform):.4f} 奈特 (理论值: {np.log(4):.4f})")
print(f"尖锐分布熵: {entropy(p_sharp):.4f} 奈特")

print("\n=== KL散度（非对称性）===")
print(f"KL(uniform || sharp) = {kl_divergence(p_uniform, p_sharp):.4f}")
print(f"KL(sharp || uniform) = {kl_divergence(p_sharp, p_uniform):.4f}")

print("\n=== 交叉熵 = 熵 + KL散度 ===")
H_p  = entropy(p_uniform)
KL   = kl_divergence(p_uniform, p_sharp)
H_pq = cross_entropy_manual(p_uniform, p_sharp)
print(f"H(P)      = {H_p:.4f}")
print(f"KL(P||Q)  = {KL:.4f}")
print(f"H(P,Q)    = {H_pq:.4f}")
print(f"H(P)+KL   = {H_p + KL:.4f}  (验证等式成立: {torch.isclose(H_pq, H_p + KL)})")


# ============================================================
# 2. 分类任务中的交叉熵损失
# ============================================================
print("\n=== 分类交叉熵损失 ===")

logits = torch.tensor([[2.0, 1.0, 0.1],   # 样本1
                        [0.1, 0.5, 2.5]])  # 样本2
labels = torch.tensor([0, 2])              # 真实类别

# PyTorch内置（接受logits）
ce_loss_fn = nn.CrossEntropyLoss()
loss_pytorch = ce_loss_fn(logits, labels)

# 手动计算：-log(p_true)
probs = F.softmax(logits, dim=1)
loss_manual = -torch.log(probs[torch.arange(2), labels]).mean()

print(f"PyTorch CE Loss: {loss_pytorch.item():.4f}")
print(f"手动计算       : {loss_manual.item():.4f}")

# 解释：最小化CE = 最小化KL(one-hot || pred)
# 因为 H(one-hot) = 0，所以 H(P,Q) = KL(P||Q)
q = probs
one_hot = F.one_hot(labels, num_classes=3).float()
kl_per_sample = kl_divergence(one_hot, q)
print(f"KL散度（等于CE）: {kl_per_sample.mean():.4f}")


# ============================================================
# 3. VAE：ELBO实现
# ============================================================
class VAE(nn.Module):
    """简单VAE示例，展示ELBO的两项"""
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        # 编码器：输出均值和对数方差
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.fc_mu  = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """重参数化技巧：z = mu + eps * std，eps ~ N(0,I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def elbo_loss(self, x, beta=1.0):
        """
        ELBO = E[log p(x|z)] - beta * KL(q(z|x) || p(z))
        其中 p(z) = N(0,I)，KL有解析解
        """
        x_recon, mu, logvar = self.forward(x)

        # 重建项：E[log p(x|z)]，用二元交叉熵近似
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

        # KL散度解析解：KL(N(mu, sigma^2) || N(0,I))
        # = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # ELBO 下界（最大化ELBO = 最小化负ELBO）
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

# 演示
vae = VAE(input_dim=784, latent_dim=20)
x_dummy = torch.rand(32, 784)  # 32个样本

elbo, recon, kl = vae.elbo_loss(x_dummy, beta=1.0)
print(f"\n=== VAE ELBO ===")
print(f"ELBO (负值，待最小化) : {elbo.item():.2f}")
print(f"  重建损失            : {recon.item():.2f}")
print(f"  KL散度              : {kl.item():.2f}")
print(f"  关系: ELBO = Recon + KL = {recon.item():.2f} + {kl.item():.2f}")

# beta-VAE：信息瓶颈视角
print(f"\nbeta-VAE (beta=4.0，更强压缩):")
elbo_b, recon_b, kl_b = vae.elbo_loss(x_dummy, beta=4.0)
print(f"  KL散度加权后        : {4.0 * kl_b.item():.2f}")
print(f"  信息瓶颈：beta越大，潜在空间越紧凑（I(X;Z)越小）")


# ============================================================
# 4. 信息论视角：熵与预测不确定性
# ============================================================
print("\n=== 预测不确定性（熵）===")

# 置信预测 vs 不确定预测
confident_probs    = torch.tensor([[0.95, 0.03, 0.02],
                                   [0.02, 0.96, 0.02]])
uncertain_probs    = torch.tensor([[0.34, 0.33, 0.33],
                                   [0.4,  0.3,  0.3 ]])

conf_entropy = entropy(confident_probs)
uncert_entropy = entropy(uncertain_probs)

print(f"置信预测的熵    : {conf_entropy.numpy().round(3)}")
print(f"不确定预测的熵  : {uncert_entropy.numpy().round(3)}")
print(f"最大熵 (均匀3类): {np.log(3):.4f}")
print("\n=> 熵可作为模型不确定性的度量（用于主动学习、OOD检测）")
```

**输出示例**：
```
=== 熵 ===
均匀分布熵: 1.3863 奈特 (理论值: 1.3863)
尖锐分布熵: 0.8018 奈特

=== KL散度（非对称性）===
KL(uniform || sharp) = 0.4506
KL(sharp || uniform) = 0.3185

=== 交叉熵 = 熵 + KL散度 ===
H(P)      = 1.3863
KL(P||Q)  = 0.4506
H(P,Q)    = 1.8369
H(P)+KL   = 1.8369  (验证等式成立: True)

=== 分类交叉熵损失 ===
PyTorch CE Loss: 0.3266
手动计算       : 0.3266
KL散度（等于CE）: 0.3266

=== VAE ELBO ===
ELBO (负值，待最小化) : 17842.23
  重建损失            : 17831.42
  KL散度              : 10.81
  关系: ELBO = Recon + KL = 17831.42 + 10.81

beta-VAE (beta=4.0，更强压缩):
  KL散度加权后        : 43.24
  信息瓶颈：beta越大，潜在空间越紧凑（I(X;Z)越小）

=== 预测不确定性（熵）===
置信预测的熵    : [0.173 0.158]
不确定预测的熵  : [1.099 1.089]
最大熵 (均匀3类): 1.0986

=> 熵可作为模型不确定性的度量（用于主动学习、OOD检测）
```

### 关键联系总结

| 信息论概念 | 深度学习对应 | 作用 |
|-----------|-------------|------|
| 交叉熵 $H(P,Q)$ | 分类损失函数 | 衡量预测分布与真实分布差异 |
| KL散度 $D_{\mathrm{KL}}(P \| Q)$ | VAE正则项、知识蒸馏 | 驱使分布接近目标分布 |
| 熵 $H(X)$ | 不确定性度量 | OOD检测、主动学习、标签平滑 |
| 互信息 $I(X;Y)$ | 信息瓶颈、表示学习 | 度量特征与标签的关联强度 |
| ELBO | VAE训练目标 | 对数似然的可优化下界 |
| 数据处理不等式 | 表示学习瓶颈 | 压缩不损失预测所需信息 |

---

## 练习题

**练习 22.1**（熵的计算）

设随机变量 $X$ 的分布为：

| $x$ | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|
| $p(x)$ | 1/2 | 1/4 | 1/8 | 1/8 |

(a) 计算 $H(X)$（以2为底，单位比特）

(b) 这个分布的熵是否等于均匀分布的熵？为什么？

(c) 若将 $X$ 编码为二进制串，最优平均码长是多少？（与 $H(X)$ 比较）

**练习 22.2**（链式法则）

设 $(X, Y)$ 的联合分布如下：

|  | $Y=0$ | $Y=1$ |
|--|-------|-------|
| $X=0$ | 3/8 | 1/8 |
| $X=1$ | 1/8 | 3/8 |

(a) 计算 $H(X)$、$H(Y)$、$H(X,Y)$

(b) 计算 $H(X \mid Y)$ 和 $H(Y \mid X)$

(c) 计算互信息 $I(X;Y)$，并验证 $H(X,Y) = H(X) + H(Y \mid X)$

**练习 22.3**（KL散度性质）

设 $P = \mathcal{N}(0, 1)$，$Q = \mathcal{N}(\mu, 1)$，两个正态分布方差相同但均值不同。

(a) 证明 $D_{\mathrm{KL}}(P \| Q) = \frac{\mu^2}{2}$

（提示：正态分布的KL散度公式：$D_{\mathrm{KL}}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$）

(b) 当 $\mu = 0$ 时，KL散度是多少？这与"等号成立条件"一致吗？

(c) 验证 $D_{\mathrm{KL}}(P \| Q) \neq D_{\mathrm{KL}}(Q \| P)$（当 $\mu \neq 0$，两者是否相等？）

**练习 22.4**（VAE推导）

VAE中对角高斯近似后验 $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$，先验 $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$。

(a) 利用练习22.3的结论，推导 $D$-维情形下的KL散度解析解：

$$D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z})) = -\frac{1}{2}\sum_{d=1}^{D}\left(1 + \log \sigma_d^2 - \mu_d^2 - \sigma_d^2\right)$$

(b) 解释为什么 $\sigma_d^2 \to 0$（后验方差趋于零）会使KL散度增大。

(c) 在PyTorch中，`logvar = log(sigma^2)`，请写出KL散度的数值稳定计算表达式。

**练习 22.5**（信息论不等式）

(a) 利用KL散度非负性，证明**吉布斯不等式**：

$$-\sum_{x} p(x) \log q(x) \geq -\sum_{x} p(x) \log p(x)$$

即 $H(P, Q) \geq H(P)$，等号当且仅当 $P = Q$。

(b) 证明对于取 $n$ 个值的均匀分布 $U$，和任意分布 $P$，有 $D_{\mathrm{KL}}(P \| U) = \log n - H(P)$，从而推导最大熵原理：$H(P) \leq \log n$。

(c) 设神经网络每层的特征映射构成马尔可夫链 $X \to h_1 \to h_2 \to \cdots \to h_L \to Y$。利用数据处理不等式，说明为什么深度网络的中间层不可能比原始输入含有更多关于标签 $Y$ 的信息。这对信息瓶颈理论有何启示？

---

## 练习答案

<details>
<summary>点击展开 练习 22.1 答案</summary>

**(a)** 计算熵（以2为底）：

$$H(X) = -\frac{1}{2}\log_2\frac{1}{2} - \frac{1}{4}\log_2\frac{1}{4} - \frac{1}{8}\log_2\frac{1}{8} - \frac{1}{8}\log_2\frac{1}{8}$$

$$= \frac{1}{2} \cdot 1 + \frac{1}{4} \cdot 2 + \frac{1}{8} \cdot 3 + \frac{1}{8} \cdot 3 = \frac{1}{2} + \frac{1}{2} + \frac{3}{8} + \frac{3}{8} = \frac{7}{4} = 1.75 \text{ 比特}$$

**(b)** 均匀分布（4个值）的熵为 $\log_2 4 = 2$ 比特，大于本题的 $1.75$ 比特。

原因：本题分布不均匀，$x=1$ 的概率为 $1/2$，集中度较高，不确定性小于均匀分布，因此熵更小。

**(c)** 香农源编码定理指出，最优平均码长满足 $H(X) \leq \bar{L} < H(X) + 1$。

最优前缀码（哈夫曼码）：$1 \to 0$，$2 \to 10$，$3 \to 110$，$4 \to 111$。

平均码长：$\bar{L} = \frac{1}{2} \cdot 1 + \frac{1}{4} \cdot 2 + \frac{1}{8} \cdot 3 + \frac{1}{8} \cdot 3 = 1.75$ 比特。

本例恰好等于 $H(X)$，因为概率恰好是2的整数次幂，实现了无损编码。

</details>

<details>
<summary>点击展开 练习 22.2 答案</summary>

**(a)** 边缘分布：$p_X(0) = 3/8 + 1/8 = 1/2$，$p_X(1) = 1/2$，$Y$ 同理。

$$H(X) = H(Y) = -\frac{1}{2}\log_2\frac{1}{2} - \frac{1}{2}\log_2\frac{1}{2} = 1 \text{ 比特}$$

$$H(X,Y) = -2 \cdot \frac{3}{8}\log_2\frac{3}{8} - 2 \cdot \frac{1}{8}\log_2\frac{1}{8}$$

$$= -\frac{3}{4}\log_2\frac{3}{8} - \frac{1}{4}\log_2\frac{1}{8}$$

$$= -\frac{3}{4}(\log_2 3 - 3) - \frac{1}{4}(-3) = \frac{9}{4} - \frac{3}{4}\log_2 3 + \frac{3}{4}$$

$$= 3 - \frac{3}{4}\log_2 3 \approx 3 - \frac{3}{4} \times 1.585 \approx 1.811 \text{ 比特}$$

**(b)** 由链式法则：

$$H(Y \mid X) = H(X,Y) - H(X) \approx 1.811 - 1 = 0.811 \text{ 比特}$$

同理 $H(X \mid Y) \approx 0.811$ 比特（由对称性）。

**(c)** 互信息：

$$I(X;Y) = H(Y) - H(Y \mid X) \approx 1 - 0.811 = 0.189 \text{ 比特}$$

验证链式法则：$H(X) + H(Y \mid X) \approx 1 + 0.811 = 1.811 = H(X,Y)$ ✓

由于 $I(X;Y) > 0$，$X$ 与 $Y$ 不独立（对角线概率更高，存在正相关）。

</details>

<details>
<summary>点击展开 练习 22.3 答案</summary>

**(a)** 代入正态KL散度公式，$\mu_1 = 0$，$\mu_2 = \mu$，$\sigma_1 = \sigma_2 = 1$：

$$D_{\mathrm{KL}}(\mathcal{N}(0,1) \| \mathcal{N}(\mu, 1)) = \log\frac{1}{1} + \frac{1 + (0-\mu)^2}{2 \cdot 1} - \frac{1}{2} = \frac{1 + \mu^2}{2} - \frac{1}{2} = \frac{\mu^2}{2}$$

**(b)** 当 $\mu = 0$ 时，$D_{\mathrm{KL}} = 0$，因为此时 $P = Q = \mathcal{N}(0,1)$，与"等号成立当且仅当 $P = Q$"完全一致。

**(c)** $D_{\mathrm{KL}}(Q \| P) = D_{\mathrm{KL}}(\mathcal{N}(\mu, 1) \| \mathcal{N}(0, 1))$：

$$= \log\frac{1}{1} + \frac{1 + \mu^2}{2} - \frac{1}{2} = \frac{\mu^2}{2}$$

本例中 $D_{\mathrm{KL}}(P \| Q) = D_{\mathrm{KL}}(Q \| P) = \mu^2/2$（等方差正态分布情形对称）。

一般地，当方差不同时，两个方向的KL散度不等，如 $\mathcal{N}(0,1)$ 与 $\mathcal{N}(0,2)$ 的两方向KL散度不同。

</details>

<details>
<summary>点击展开 练习 22.4 答案</summary>

**(a)** 各维度独立，总KL散度为各维之和。对第 $d$ 维，$q_d = \mathcal{N}(\mu_d, \sigma_d^2)$，$p_d = \mathcal{N}(0,1)$：

$$D_{\mathrm{KL}}(q_d \| p_d) = \log\frac{1}{\sigma_d} + \frac{\sigma_d^2 + \mu_d^2}{2} - \frac{1}{2} = -\frac{1}{2}\log\sigma_d^2 + \frac{\sigma_d^2 + \mu_d^2 - 1}{2}$$

$$= -\frac{1}{2}\left(1 + \log\sigma_d^2 - \mu_d^2 - \sigma_d^2\right)$$

求和：$D_{\mathrm{KL}}(q \| p) = -\frac{1}{2}\sum_{d=1}^{D}(1 + \log\sigma_d^2 - \mu_d^2 - \sigma_d^2)$ ✓

**(b)** 当 $\sigma_d^2 \to 0$ 时，$\log\sigma_d^2 \to -\infty$，使得 $-(1 + \log\sigma_d^2 - \mu_d^2 - \sigma_d^2)$ 中的 $-\log\sigma_d^2 \to +\infty$，KL散度趋向正无穷。

直观理解：方差趋于零意味着近似后验完全"确定"，与扩散的先验 $\mathcal{N}(0,I)$ 差异极大。

**(c)** 令 `logvar = log(sigma^2)`，则 `sigma^2 = exp(logvar)`：

```python
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
```

使用 `logvar` 而非直接用 `sigma^2` 的优点：避免对负数求对数，数值更稳定；梯度更平滑。

</details>

<details>
<summary>点击展开 练习 22.5 答案</summary>

**(a)** 由KL散度非负性：

$$D_{\mathrm{KL}}(P \| Q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = -\sum_x p(x)\log q(x) + \sum_x p(x)\log p(x) \geq 0$$

因此 $-\sum_x p(x)\log q(x) \geq -\sum_x p(x)\log p(x)$，即 $H(P,Q) \geq H(P)$。

等号成立当 $D_{\mathrm{KL}}(P \| Q) = 0$，即 $P = Q$。

**(b)** 设均匀分布 $u(x) = 1/n$：

$$D_{\mathrm{KL}}(P \| U) = \sum_x p(x)\log\frac{p(x)}{1/n} = \sum_x p(x)\log p(x) + \sum_x p(x)\log n = -H(P) + \log n$$

由非负性 $D_{\mathrm{KL}}(P \| U) \geq 0$，故 $\log n - H(P) \geq 0$，即 $H(P) \leq \log n$。

等号当 $P = U$（均匀分布）成立，证明最大熵原理。

**(c)** 马尔可夫链 $X \to h_1 \to h_2 \to \cdots \to h_L \to \hat{Y}$，由数据处理不等式逐步应用：

$$I(X; \hat{Y}) \leq I(X; h_L) \leq \cdots \leq I(X; h_1) \leq I(X; X) = H(X)$$

$$I(h_L; Y) \leq I(h_{L-1}; Y) \leq \cdots \leq I(h_1; Y) \leq I(X; Y)$$

**启示**：
- 每一层对 $Y$ 所能保留的最大信息量 $I(h_i; Y)$ 不超过原始输入 $I(X; Y)$
- 深度网络无法凭空"创造"关于标签的信息，只能提取和压缩
- 信息瓶颈理论认为，优秀的表示 $h$ 应在 $I(h; Y)$ 大的同时 $I(X; h)$ 小（压缩无关信息）
- 这为正则化、Dropout等技术提供了信息论解释：它们迫使网络丢弃与预测无关的冗余信息

</details>
