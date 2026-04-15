# 公式速查表

本文件汇总了概率论与数理统计中的常用公式，方便读者随时查阅。

---

## 一、概率基础公式

### 1.1 概率的基本性质

| 公式 | 说明 |
|------|------|
| $0 \leq P(A) \leq 1$ | 概率的范围 |
| $P(\Omega) = 1$ | 必然事件的概率 |
| $P(\emptyset) = 0$ | 不可能事件的概率 |
| $P(A^c) = 1 - P(A)$ | 补事件概率 |
| $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ | 加法公式 |
| $P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(B \cap C) - P(A \cap C) + P(A \cap B \cap C)$ | 容斥原理 |

### 1.2 条件概率与独立性

| 公式 | 说明 |
|------|------|
| $P(A \mid B) = \dfrac{P(A \cap B)}{P(B)}$ | 条件概率定义 |
| $P(A \cap B) = P(A \mid B) P(B) = P(B \mid A) P(A)$ | 乘法公式 |
| $P(A) = \sum_{i=1}^{n} P(A \mid B_i) P(B_i)$ | 全概率公式 |
| $P(B_j \mid A) = \dfrac{P(A \mid B_j) P(B_j)}{\sum_{i=1}^{n} P(A \mid B_i) P(B_i)}$ | 贝叶斯公式 |
| $P(A \cap B) = P(A) P(B)$ | 独立事件 |

### 1.3 组合公式

| 公式 | 说明 |
|------|------|
| $n! = n \times (n-1) \times \cdots \times 1$ | 阶乘 |
| $P_n^k = \dfrac{n!}{(n-k)!}$ | 排列数 |
| $C_n^k = \binom{n}{k} = \dfrac{n!}{k!(n-k)!}$ | 组合数 |
| $(a+b)^n = \sum_{k=0}^{n} \binom{n}{k} a^k b^{n-k}$ | 二项式定理 |
| $\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$ | 帕斯卡恒等式 |

---

## 二、随机变量

### 2.1 离散随机变量

| 公式 | 说明 |
|------|------|
| $P(X = x_i) = p_i$ | 概率质量函数 |
| $\sum_{i} p_i = 1$ | 归一化条件 |
| $E[X] = \sum_{i} x_i p_i$ | 期望 |
| $E[g(X)] = \sum_{i} g(x_i) p_i$ | 函数的期望 |
| $\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$ | 方差 |

### 2.2 连续随机变量

| 公式 | 说明 |
|------|------|
| $P(a \leq X \leq b) = \int_a^b f(x) \, dx$ | 区间概率 |
| $\int_{-\infty}^{\infty} f(x) \, dx = 1$ | 归一化条件 |
| $F(x) = P(X \leq x) = \int_{-\infty}^{x} f(t) \, dt$ | 累积分布函数 |
| $f(x) = F'(x)$ | PDF与CDF关系 |
| $E[X] = \int_{-\infty}^{\infty} x f(x) \, dx$ | 期望 |
| $\text{Var}(X) = \int_{-\infty}^{\infty} (x - \mu)^2 f(x) \, dx$ | 方差 |

### 2.3 期望与方差的性质

| 公式 | 说明 |
|------|------|
| $E[aX + b] = aE[X] + b$ | 期望的线性性 |
| $E[X + Y] = E[X] + E[Y]$ | 期望的可加性 |
| $E[XY] = E[X]E[Y]$ (若独立) | 独立变量乘积期望 |
| $\text{Var}(aX + b) = a^2 \text{Var}(X)$ | 方差的线性变换 |
| $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$ | 方差的可加性 |
| $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ (若独立) | 独立变量方差 |

### 2.4 协方差与相关系数

| 公式 | 说明 |
|------|------|
| $\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]$ | 协方差定义 |
| $\text{Cov}(X, Y) = E[XY] - E[X]E[Y]$ | 协方差计算公式 |
| $\rho_{XY} = \dfrac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$ | 相关系数 |
| $-1 \leq \rho_{XY} \leq 1$ | 相关系数范围 |

---

## 三、常见分布

### 3.1 离散分布

| 分布 | PMF | 期望 | 方差 |
|------|-----|------|------|
| 伯努利 $\text{Bernoulli}(p)$ | $P(X=1) = p, P(X=0) = 1-p$ | $p$ | $p(1-p)$ |
| 二项 $\text{Binomial}(n,p)$ | $\binom{n}{k} p^k (1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| 泊松 $\text{Poisson}(\lambda)$ | $\dfrac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ |
| 几何 $\text{Geometric}(p)$ | $(1-p)^{k-1} p$ | $\dfrac{1}{p}$ | $\dfrac{1-p}{p^2}$ |
| 负二项 $\text{NB}(r,p)$ | $\binom{k-1}{r-1} p^r (1-p)^{k-r}$ | $\dfrac{r}{p}$ | $\dfrac{r(1-p)}{p^2}$ |

### 3.2 连续分布

| 分布 | PDF | 期望 | 方差 |
|------|-----|------|------|
| 均匀 $\text{Uniform}(a,b)$ | $\dfrac{1}{b-a}$, $x \in [a,b]$ | $\dfrac{a+b}{2}$ | $\dfrac{(b-a)^2}{12}$ |
| 指数 $\text{Exp}(\lambda)$ | $\lambda e^{-\lambda x}$, $x \geq 0$ | $\dfrac{1}{\lambda}$ | $\dfrac{1}{\lambda^2}$ |
| 正态 $\mathcal{N}(\mu,\sigma^2)$ | $\dfrac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |
| Gamma $\text{Gamma}(\alpha,\beta)$ | $\dfrac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$ | $\dfrac{\alpha}{\beta}$ | $\dfrac{\alpha}{\beta^2}$ |
| Beta $\text{Beta}(\alpha,\beta)$ | $\dfrac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$ | $\dfrac{\alpha}{\alpha+\beta}$ | $\dfrac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |

### 3.3 多维分布

| 分布 | 公式 |
|------|------|
| 多项分布 | $P(X_1=k_1,\ldots,X_m=k_m) = \dfrac{n!}{k_1!\cdots k_m!} p_1^{k_1} \cdots p_m^{k_m}$ |
| 多元正态 | $f(\mathbf{x}) = \dfrac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\dfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$ |
| Dirichlet | $f(\mathbf{x}) = \dfrac{\Gamma(\sum_i \alpha_i)}{\prod_i \Gamma(\alpha_i)} \prod_{i=1}^{K} x_i^{\alpha_i - 1}$ |

### 3.4 矩母函数（MGF）

| 分布 | $M_X(t) = E[e^{tX}]$ | 存在条件 |
|------|------------------------|----------|
| Bernoulli$(p)$ | $(1-p) + pe^t$ | 所有 $t$ |
| Binomial$(n,p)$ | $[(1-p) + pe^t]^n$ | 所有 $t$ |
| Poisson$(\lambda)$ | $e^{\lambda(e^t - 1)}$ | 所有 $t$ |
| Geometric$(p)$ | $\dfrac{pe^t}{1-(1-p)e^t}$ | $t < -\ln(1-p)$ |
| Uniform$(a,b)$ | $\dfrac{e^{tb} - e^{ta}}{t(b-a)}$ | 所有 $t$ |
| Exp$(\lambda)$ | $\dfrac{\lambda}{\lambda - t}$ | $t < \lambda$ |
| $\mathcal{N}(\mu, \sigma^2)$ | $\exp(\mu t + \sigma^2 t^2/2)$ | 所有 $t$ |
| Gamma$(\alpha, \beta)$ | $\left(\dfrac{\beta}{\beta - t}\right)^\alpha$ | $t < \beta$ |
| $\chi^2(n)$ | $(1 - 2t)^{-n/2}$ | $t < 1/2$ |

**核心性质**：$M_X^{(n)}(0) = E[X^n]$；独立变量之和 $M_{X+Y}(t) = M_X(t) M_Y(t)$

### 3.5 常见共轭先验对

| 似然（数据模型） | 先验分布 | 后验分布 | 后验参数更新 |
|-----------------|----------|----------|-------------|
| Bernoulli / Binomial | Beta$(\alpha, \beta)$ | Beta$(\alpha+k, \beta+n-k)$ | $k$: 成功次数 |
| Poisson$(\lambda)$ | Gamma$(\alpha, \beta)$ | Gamma$(\alpha+\sum x_i, \beta+n)$ | |
| Normal (均值, $\sigma^2$ 已知) | Normal$(\mu_0, \tau^2)$ | Normal$(\tilde{\mu}, \tilde{\tau}^2)$ | 精度加权平均 |
| Normal (方差, $\mu$ 已知) | Inv-Gamma$(\alpha, \beta)$ | Inv-Gamma$(\alpha+n/2, \beta+\frac{1}{2}\sum(x_i-\mu)^2)$ | |
| Multinomial | Dirichlet$(\boldsymbol{\alpha})$ | Dirichlet$(\boldsymbol{\alpha} + \mathbf{x})$ | |
| Exponential$(\lambda)$ | Gamma$(\alpha, \beta)$ | Gamma$(\alpha+n, \beta+\sum x_i)$ | |

---

## 四、极限定理

### 4.1 不等式

| 公式 | 说明 |
|------|------|
| $P(\lvert X \rvert \geq a) \leq \dfrac{E[\lvert X \rvert]}{a}$ | 马尔可夫不等式 |
| $P(\lvert X - \mu \rvert \geq k\sigma) \leq \dfrac{1}{k^2}$ | 切比雪夫不等式 |
| $P(\lvert X - \mu \rvert \geq t) \leq 2\exp\left(-\dfrac{2t^2}{(b-a)^2}\right)$ | Hoeffding不等式 (有界) |

### 4.2 大数定律

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i \xrightarrow{P} \mu \quad \text{(弱大数定律)}$$

$$\bar{X}_n \xrightarrow{a.s.} \mu \quad \text{(强大数定律)}$$

### 4.3 中心极限定理

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

$$\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

---

## 五、抽样分布

### 5.1 三大抽样分布

| 分布 | 定义 | 期望 | 方差 |
|------|------|------|------|
| 卡方分布 $\chi^2(n)$ | $\sum_{i=1}^{n} Z_i^2$, $Z_i \sim \mathcal{N}(0,1)$ | $n$ | $2n$ |
| t分布 $t(n)$ | $\dfrac{Z}{\sqrt{V/n}}$, $Z \sim \mathcal{N}(0,1)$, $V \sim \chi^2(n)$ | $0$ (n>1) | $\dfrac{n}{n-2}$ (n>2) |
| F分布 $F(m,n)$ | $\dfrac{V_1/m}{V_2/n}$, $V_1 \sim \chi^2(m)$, $V_2 \sim \chi^2(n)$ | $\dfrac{n}{n-2}$ (n>2) | 复杂 |

### 5.2 样本统计量分布

设 $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$：

| 统计量 | 分布 |
|--------|------|
| $\bar{X} = \dfrac{1}{n}\sum X_i$ | $\mathcal{N}\left(\mu, \dfrac{\sigma^2}{n}\right)$ |
| $\dfrac{(n-1)S^2}{\sigma^2}$ | $\chi^2(n-1)$ |
| $\dfrac{\bar{X} - \mu}{S/\sqrt{n}}$ | $t(n-1)$ |

---

## 六、参数估计

### 6.1 点估计

| 方法 | 公式 |
|------|------|
| 矩估计 | 令样本矩等于总体矩：$\bar{X} = E[X]$, $\overline{X^2} = E[X^2]$, ... |
| 最大似然估计 | $\hat{\theta}_{MLE} = \arg\max_\theta \prod_{i=1}^{n} f(x_i; \theta)$ |
| 对数似然 | $\hat{\theta}_{MLE} = \arg\max_\theta \sum_{i=1}^{n} \log f(x_i; \theta)$ |

### 6.2 估计量的评价

| 性质 | 定义 |
|------|------|
| 无偏性 | $E[\hat{\theta}] = \theta$ |
| 有效性 | $\text{Var}(\hat{\theta}_1) \leq \text{Var}(\hat{\theta}_2)$ |
| 一致性 | $\hat{\theta}_n \xrightarrow{P} \theta$ |
| MSE | $\text{MSE}(\hat{\theta}) = \text{Var}(\hat{\theta}) + \text{Bias}^2(\hat{\theta})$ |

### 6.3 置信区间

| 参数 | 条件 | 置信区间 |
|------|------|----------|
| $\mu$ ($\sigma^2$ 已知) | 正态总体 | $\bar{X} \pm z_{\alpha/2} \dfrac{\sigma}{\sqrt{n}}$ |
| $\mu$ ($\sigma^2$ 未知) | 正态总体 | $\bar{X} \pm t_{\alpha/2}(n-1) \dfrac{S}{\sqrt{n}}$ |
| $\sigma^2$ | 正态总体 | $\left[\dfrac{(n-1)S^2}{\chi^2_{\alpha/2}(n-1)}, \dfrac{(n-1)S^2}{\chi^2_{1-\alpha/2}(n-1)}\right]$ |

### 6.4 贝叶斯估计

$$p(\theta \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \theta) p(\theta)}{p(\mathbf{x})} \propto p(\mathbf{x} \mid \theta) p(\theta)$$

| 估计 | 公式 |
|------|------|
| MAP | $\hat{\theta}_{MAP} = \arg\max_\theta p(\theta \mid \mathbf{x})$ |
| 后验均值 | $\hat{\theta}_{Bayes} = E[\theta \mid \mathbf{x}]$ |

---

## 七、假设检验

### 7.1 检验统计量

| 检验 | 条件 | 统计量 |
|------|------|--------|
| z检验 | $\sigma^2$ 已知 | $z = \dfrac{\bar{X} - \mu_0}{\sigma/\sqrt{n}}$ |
| t检验 | $\sigma^2$ 未知 | $t = \dfrac{\bar{X} - \mu_0}{S/\sqrt{n}}$ |
| $\chi^2$ 检验 | 方差检验 | $\chi^2 = \dfrac{(n-1)S^2}{\sigma_0^2}$ |
| F检验 | 两总体方差比 | $F = \dfrac{S_1^2}{S_2^2}$ |

### 7.2 两类错误

| 错误类型 | 定义 | 概率 |
|----------|------|------|
| 第一类错误 | 拒绝真 $H_0$ | $\alpha$ (显著性水平) |
| 第二类错误 | 接受假 $H_0$ | $\beta$ |
| 检验功效 | 拒绝假 $H_0$ | $1 - \beta$ |

---

## 八、信息论

| 公式 | 说明 |
|------|------|
| $H(X) = -\sum_x p(x) \log p(x)$ | 离散熵 |
| $H(X) = -\int f(x) \log f(x) \, dx$ | 微分熵 |
| $H(X,Y) = H(X) + H(Y \mid X)$ | 链式法则 |
| $I(X;Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X)$ | 互信息 |
| $D_{KL}(p \| q) = \sum_x p(x) \log \dfrac{p(x)}{q(x)}$ | KL散度 |
| $H(p, q) = -\sum_x p(x) \log q(x)$ | 交叉熵 |
| $H(p, q) = H(p) + D_{KL}(p \| q)$ | 交叉熵分解 |

---

## 九、常用积分与特殊函数

| 公式 | 说明 |
|------|------|
| $\Gamma(n) = (n-1)!$ | Gamma函数 (整数) |
| $\Gamma(\alpha) = \int_0^\infty x^{\alpha-1} e^{-x} \, dx$ | Gamma函数 (一般) |
| $B(\alpha, \beta) = \dfrac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ | Beta函数 |
| $\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}$ | 高斯积分 |
| $\int_{-\infty}^{\infty} e^{-\frac{x^2}{2}} \, dx = \sqrt{2\pi}$ | 标准正态积分 |
