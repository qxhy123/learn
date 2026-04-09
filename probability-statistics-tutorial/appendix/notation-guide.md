# 符号说明

本文件汇总了全书使用的数学符号及其含义，方便读者随时查阅。

---

## 集合与数系

| 符号 | 含义 | 示例 |
|------|------|------|
| $\mathbb{N}$ | 自然数集 | $\{0, 1, 2, 3, \ldots\}$ |
| $\mathbb{Z}$ | 整数集 | $\{\ldots, -2, -1, 0, 1, 2, \ldots\}$ |
| $\mathbb{Z}^+$ | 正整数集 | $\{1, 2, 3, \ldots\}$ |
| $\mathbb{R}$ | 实数集 | 所有实数 |
| $\mathbb{R}^+$ | 正实数集 | $(0, +\infty)$ |
| $\mathbb{R}^n$ | $n$ 维实向量空间 | $\mathbb{R}^3$ 是三维空间 |
| $[a, b]$ | 闭区间 | $a \leq x \leq b$ |
| $(a, b)$ | 开区间 | $a < x < b$ |
| $\in$ | 属于 | $x \in \mathbb{R}$ |
| $\subset$ | 真子集 | $A \subset B$ |
| $\subseteq$ | 子集 | $A \subseteq B$ |
| $\cup$ | 并集 | $A \cup B$ |
| $\cap$ | 交集 | $A \cap B$ |
| $A^c$ 或 $\bar{A}$ | 补集 | $A$ 的补集 |
| $\emptyset$ | 空集 | 无元素的集合 |

---

## 概率论基础

| 符号 | 含义 | 说明 |
|------|------|------|
| $\Omega$ | 样本空间 | 所有可能结果的集合 |
| $\omega$ | 样本点 | 样本空间中的元素 |
| $A, B, C$ | 事件 | 样本空间的子集 |
| $P(A)$ | 事件 $A$ 的概率 | $0 \leq P(A) \leq 1$ |
| $P(A \mid B)$ | 条件概率 | 在 $B$ 发生条件下 $A$ 的概率 |
| $A \perp B$ | 独立 | $A$ 与 $B$ 相互独立 |
| $A \perp B \mid C$ | 条件独立 | 给定 $C$ 时 $A$ 与 $B$ 条件独立 |

---

## 随机变量

| 符号 | 含义 | 说明 |
|------|------|------|
| $X, Y, Z$ | 随机变量 | 大写字母表示随机变量 |
| $x, y, z$ | 随机变量的取值 | 小写字母表示具体取值 |
| $\mathbf{X}$ | 随机向量 | 粗体表示向量 |
| $P(X = x)$ | 概率质量 | 离散随机变量取值 $x$ 的概率 |
| $p(x)$ 或 $p_X(x)$ | 概率质量函数 (PMF) | 离散分布 |
| $f(x)$ 或 $f_X(x)$ | 概率密度函数 (PDF) | 连续分布 |
| $F(x)$ 或 $F_X(x)$ | 累积分布函数 (CDF) | $F(x) = P(X \leq x)$ |
| $X \sim D$ | 分布记号 | $X$ 服从分布 $D$ |
| $X \overset{d}{=} Y$ | 同分布 | $X$ 与 $Y$ 有相同分布 |
| $X_1, \ldots, X_n \overset{iid}{\sim} D$ | 独立同分布 | i.i.d. 样本 |

---

## 期望与方差

| 符号 | 含义 | 说明 |
|------|------|------|
| $E[X]$ 或 $\mathbb{E}[X]$ | 期望 | 随机变量的均值 |
| $\mu$ 或 $\mu_X$ | 期望 | $E[X]$ 的另一种记法 |
| $\text{Var}(X)$ | 方差 | $E[(X - \mu)^2]$ |
| $\sigma^2$ 或 $\sigma_X^2$ | 方差 | $\text{Var}(X)$ 的另一种记法 |
| $\sigma$ 或 $\sigma_X$ | 标准差 | $\sqrt{\text{Var}(X)}$ |
| $\text{Cov}(X, Y)$ | 协方差 | $E[(X - \mu_X)(Y - \mu_Y)]$ |
| $\rho_{XY}$ 或 $\text{Corr}(X, Y)$ | 相关系数 | $\text{Cov}(X,Y)/(\sigma_X \sigma_Y)$ |
| $E[X \mid Y]$ | 条件期望 | 给定 $Y$ 时 $X$ 的期望 |

---

## 常见分布

### 离散分布

| 符号 | 分布名称 | 参数 |
|------|----------|------|
| $\text{Bernoulli}(p)$ | 伯努利分布 | 成功概率 $p$ |
| $\text{Binomial}(n, p)$ 或 $B(n, p)$ | 二项分布 | 试验次数 $n$，成功概率 $p$ |
| $\text{Poisson}(\lambda)$ | 泊松分布 | 速率参数 $\lambda$ |
| $\text{Geometric}(p)$ | 几何分布 | 成功概率 $p$ |
| $\text{NB}(r, p)$ | 负二项分布 | 成功次数 $r$，成功概率 $p$ |
| $\text{Multinomial}(n, \mathbf{p})$ | 多项分布 | 试验次数 $n$，概率向量 $\mathbf{p}$ |

### 连续分布

| 符号 | 分布名称 | 参数 |
|------|----------|------|
| $\text{Uniform}(a, b)$ 或 $U(a, b)$ | 均匀分布 | 区间端点 $a, b$ |
| $\text{Exp}(\lambda)$ | 指数分布 | 速率参数 $\lambda$ |
| $\mathcal{N}(\mu, \sigma^2)$ | 正态分布 | 均值 $\mu$，方差 $\sigma^2$ |
| $\text{Gamma}(\alpha, \beta)$ | Gamma分布 | 形状 $\alpha$，速率 $\beta$ |
| $\text{Beta}(\alpha, \beta)$ | Beta分布 | 参数 $\alpha, \beta$ |
| $\chi^2(n)$ | 卡方分布 | 自由度 $n$ |
| $t(n)$ | t分布 | 自由度 $n$ |
| $F(m, n)$ | F分布 | 自由度 $m, n$ |
| $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | 多元正态分布 | 均值向量，协方差矩阵 |
| $\text{Dirichlet}(\boldsymbol{\alpha})$ | Dirichlet分布 | 浓度参数向量 $\boldsymbol{\alpha}$ |

---

## 收敛性

| 符号 | 含义 | 说明 |
|------|------|------|
| $X_n \xrightarrow{P} X$ | 依概率收敛 | Convergence in probability |
| $X_n \xrightarrow{d} X$ | 依分布收敛 | Convergence in distribution |
| $X_n \xrightarrow{a.s.} X$ | 几乎必然收敛 | Almost sure convergence |
| $X_n \xrightarrow{L^p} X$ | $L^p$ 收敛 | Mean convergence |
| $X_n \xrightarrow{m.s.} X$ | 均方收敛 | Mean square convergence |

---

## 统计学

| 符号 | 含义 | 说明 |
|------|------|------|
| $\theta$ | 参数 | 待估计的未知量 |
| $\hat{\theta}$ | 估计量 | $\theta$ 的估计 |
| $\hat{\theta}_{MLE}$ | 最大似然估计 | Maximum Likelihood Estimator |
| $\hat{\theta}_{MAP}$ | 最大后验估计 | Maximum A Posteriori |
| $L(\theta)$ 或 $\mathcal{L}(\theta)$ | 似然函数 | $\prod_i f(x_i; \theta)$ |
| $\ell(\theta)$ | 对数似然 | $\sum_i \log f(x_i; \theta)$ |
| $\bar{X}$ | 样本均值 | $\frac{1}{n}\sum_{i=1}^n X_i$ |
| $S^2$ | 样本方差 | $\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$ |
| $S$ | 样本标准差 | $\sqrt{S^2}$ |
| $T$ | 检验统计量 | Test statistic |
| $H_0$ | 原假设 | Null hypothesis |
| $H_1$ 或 $H_a$ | 备择假设 | Alternative hypothesis |
| $\alpha$ | 显著性水平 | Significance level |
| $p$-value | p值 | 在 $H_0$ 下观测到更极端值的概率 |

---

## 信息论

| 符号 | 含义 | 说明 |
|------|------|------|
| $H(X)$ | 熵 | $-\sum_x p(x) \log p(x)$ |
| $H(X, Y)$ | 联合熵 | Joint entropy |
| $H(X \mid Y)$ | 条件熵 | Conditional entropy |
| $I(X; Y)$ | 互信息 | Mutual information |
| $D_{KL}(p \| q)$ | KL散度 | Kullback-Leibler divergence |
| $H(p, q)$ | 交叉熵 | Cross entropy |

---

## 其他常用符号

| 符号 | 含义 |
|------|------|
| $:=$ 或 $\triangleq$ | 定义为 |
| $\forall$ | 对于所有 |
| $\exists$ | 存在 |
| $\Rightarrow$ | 蕴含 |
| $\Leftrightarrow$ | 当且仅当 |
| $\approx$ | 约等于 |
| $\propto$ | 正比于 |
| $\sum$ | 求和 |
| $\prod$ | 求积 |
| $\int$ | 积分 |
| $\arg\max$ | 使目标函数最大的参数 |
| $\arg\min$ | 使目标函数最小的参数 |
| $\log$ | 自然对数（默认以 $e$ 为底） |
| $\log_2$ | 以 2 为底的对数 |
| $\exp(x)$ 或 $e^x$ | 指数函数 |
| $\mathbf{1}_{A}$ 或 $I_A$ | 指示函数 | 事件 $A$ 发生时为 1，否则为 0 |
| $\lfloor x \rfloor$ | 下取整 | 不超过 $x$ 的最大整数 |
| $\lceil x \rceil$ | 上取整 | 不小于 $x$ 的最小整数 |

---

## 深度学习相关符号

| 符号 | 含义 | 说明 |
|------|------|------|
| $\mathbf{x}$ | 输入向量 | 网络的输入 |
| $\mathbf{y}$ | 标签/输出向量 | 真实标签或网络输出 |
| $\hat{\mathbf{y}}$ | 预测值 | 网络的预测输出 |
| $\mathbf{W}$ | 权重矩阵 | 神经网络层的参数 |
| $\mathbf{b}$ | 偏置向量 | 神经网络层的参数 |
| $\theta$ | 模型参数 | 所有可学习参数的集合 |
| $\mathcal{L}$ | 损失函数 | Loss function |
| $\nabla_\theta \mathcal{L}$ | 梯度 | 损失对参数的导数 |
| $p(y \mid \mathbf{x}; \theta)$ | 条件概率模型 | 给定输入和参数的输出分布 |
| $q(\mathbf{z} \mid \mathbf{x})$ | 变分后验 | Variational posterior |
| $p(\mathbf{z})$ | 先验分布 | Prior distribution |
| $\text{ELBO}$ | 证据下界 | Evidence Lower Bound |
| $\text{KL}[q \| p]$ | KL散度 | 变分推断中的正则项 |
