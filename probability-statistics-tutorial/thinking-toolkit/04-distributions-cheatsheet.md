# 常见分布速查表

> **一例速记**：公交车平均每 10 分钟一班，你刚错过一班，等待时间 $T$ 服从什么分布？
> 答：指数分布 $T \sim \text{Exp}(1/10)$，$E(T) = 10$ 分钟。
> 关键识别符：**无记忆性**——已经等了 5 分钟，下一班到达的期望等待时间还是 10 分钟。
> $P(T > s + t \mid T > s) = P(T > t)$，等待时间"不记得"你已经等了多久。

---

## 一、分布的分类框架

随机变量按取值类型分为：

- **离散**：取可数个值（有限或可数无穷）。用 PMF $P(X = k) = p_k$ 描述。
- **连续**：取不可数个值（区间）。用 PDF $f(x)$ 描述，$P(a < X \leq b) = \int_a^b f(x)\,dx$。

选择分布时的思维路径：

| 场景特征 | 候选分布 |
|---|---|
| 单次成功/失败 | Bernoulli$(p)$ |
| $n$ 次独立试验的成功次数 | 二项 $B(n,p)$ |
| 单位时间/空间内的稀有事件计数 | Poisson$(\lambda)$ |
| 首次成功前的试验次数 | 几何 $\text{Geom}(p)$ |
| 有限总体中不放回抽样 | 超几何 |
| 等待第 $r$ 次成功的试验次数 | 负二项 $\text{NB}(r,p)$ |
| 区间上均匀随机 | 均匀 $U(a,b)$ |
| 等待时间、寿命（无记忆性） | 指数 $\text{Exp}(\lambda)$ |
| 自然界大量独立微小影响叠加 | 正态 $N(\mu,\sigma^2)$ |
| 非负、右偏、等待 $k$ 次事件 | Gamma$(\alpha,\beta)$ |
| 取值 $[0,1]$ 的比例 | Beta$(\alpha,\beta)$ |
| 正态样本方差的分布 | $\chi^2_n$ |
| 正态总体均值的 $t$ 检验统计量 | $t_n$ |
| 两个样本方差之比 | $F_{m,n}$ |

---

## 二、离散分布速查表

### 2.1 Bernoulli 分布

$$X \sim B(1, p), \quad P(X=1) = p, \quad P(X=0) = 1-p = q.$$

| 项目 | 值 |
|---|---|
| PMF | $P(X=k) = p^k q^{1-k}$，$k \in \{0,1\}$ |
| $E(X)$ | $p$ |
| $\text{Var}(X)$ | $p(1-p) = pq$ |
| MGF | $q + pe^t$ |
| 典型场景 | 单次抛硬币、产品是否合格 |

### 2.2 二项分布

$$X \sim B(n, p), \quad P(X=k) = \binom{n}{k} p^k q^{n-k}, \quad k = 0, 1, \ldots, n.$$

| 项目 | 值 |
|---|---|
| PMF | $\binom{n}{k} p^k q^{n-k}$ |
| $E(X)$ | $np$ |
| $\text{Var}(X)$ | $npq$ |
| MGF | $(q + pe^t)^n$ |
| 典型场景 | $n$ 次独立成功/失败试验的总成功次数；抽检批次合格率 |

**与 Poisson 的关系**：$n$ 很大、$p$ 很小、$\lambda = np$ 保持适中时，$B(n,p) \approx \text{Poisson}(\lambda)$。

### 2.3 Poisson 分布

$$X \sim \text{Poisson}(\lambda), \quad P(X=k) = \frac{e^{-\lambda}\lambda^k}{k!}, \quad k = 0, 1, 2, \ldots$$

| 项目 | 值 |
|---|---|
| PMF | $e^{-\lambda}\lambda^k / k!$ |
| $E(X)$ | $\lambda$ |
| $\text{Var}(X)$ | $\lambda$（期望 = 方差） |
| MGF | $e^{\lambda(e^t - 1)}$ |
| 典型场景 | 单位时间呼叫次数、每平方米缺陷数、放射性粒子计数 |

**可加性**：若 $X_1 \sim \text{Poisson}(\lambda_1)$，$X_2 \sim \text{Poisson}(\lambda_2)$ 独立，则 $X_1 + X_2 \sim \text{Poisson}(\lambda_1 + \lambda_2)$。

### 2.4 几何分布

首次成功前**试验次数**（第一次成功发生在第 $k$ 次）：

$$X \sim \text{Geom}(p), \quad P(X=k) = (1-p)^{k-1} p, \quad k = 1, 2, \ldots$$

| 项目 | 值 |
|---|---|
| PMF | $(1-p)^{k-1} p$ |
| $E(X)$ | $1/p$ |
| $\text{Var}(X)$ | $(1-p)/p^2 = q/p^2$ |
| MGF | $pe^t / (1 - qe^t)$，$t < -\ln q$ |
| 典型场景 | 首次成功前的试验次数；离散无记忆性等待 |

**无记忆性（离散版）**：$P(X > m+n \mid X > m) = P(X > n)$，几何分布是唯一的离散无记忆分布。

### 2.5 超几何分布

$N$ 件物品中有 $K$ 件"特殊"，不放回取 $n$ 件，其中特殊品数 $X$：

$$P(X=k) = \frac{\dbinom{K}{k}\dbinom{N-K}{n-k}}{\dbinom{N}{n}}, \quad \max(0, n+K-N) \leq k \leq \min(n, K).$$

| 项目 | 值 |
|---|---|
| $E(X)$ | $n K/N$ |
| $\text{Var}(X)$ | $n \frac{K}{N} \frac{N-K}{N} \frac{N-n}{N-1}$ |
| 典型场景 | 不放回抽样中的成功次数；有限总体质量检验 |

**与二项的关系**：$N \to \infty$ 且 $K/N \to p$ 时，超几何 $\to B(n,p)$（有放回近似无放回）。

### 2.6 负二项分布

等待**第 $r$ 次成功**时，已经历的**试验总次数** $X$：

$$P(X=k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}, \quad k = r, r+1, \ldots$$

| 项目 | 值 |
|---|---|
| $E(X)$ | $r/p$ |
| $\text{Var}(X)$ | $rq/p^2$ |
| MGF | $\left(\dfrac{pe^t}{1-qe^t}\right)^r$，$t < -\ln q$ |
| 典型场景 | 等待 $r$ 次成功的总试验次数；质量控制中第 $r$ 件次品出现前的产品数 |

**注意**：$r=1$ 时退化为几何分布；也有以"失败次数"为 $X$ 的等价定义，PMF 形式有差异。

---

## 三、连续分布速查表

### 3.1 均匀分布

$$X \sim U(a, b), \quad f(x) = \frac{1}{b-a}, \quad a < x < b.$$

| 项目 | 值 |
|---|---|
| PDF | $1/(b-a)$ |
| $E(X)$ | $(a+b)/2$ |
| $\text{Var}(X)$ | $(b-a)^2/12$ |
| MGF | $(e^{tb} - e^{ta}) / [t(b-a)]$，$t \neq 0$ |
| 典型场景 | 区间上的等可能随机点；几何概型的"均匀"假设 |

### 3.2 指数分布

$$X \sim \text{Exp}(\lambda), \quad f(x) = \lambda e^{-\lambda x}, \quad x \geq 0.$$

| 项目 | 值 |
|---|---|
| PDF | $\lambda e^{-\lambda x}$ |
| CDF | $1 - e^{-\lambda x}$ |
| $E(X)$ | $1/\lambda$ |
| $\text{Var}(X)$ | $1/\lambda^2$ |
| MGF | $\lambda/(\lambda - t)$，$t < \lambda$ |
| 典型场景 | 等待时间（Poisson 过程相邻事件间隔）；无老化元件寿命 |

**无记忆性（连续版）**：$P(X > s+t \mid X > s) = P(X > t)$，$\forall s, t \geq 0$。指数分布是唯一的连续无记忆分布。

**与 Poisson 过程的联系**：若事件按速率 $\lambda$ 的 Poisson 过程到达，则相邻事件间隔 $\sim \text{Exp}(\lambda)$。

### 3.3 正态分布

$$X \sim N(\mu, \sigma^2), \quad f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$

| 项目 | 值 |
|---|---|
| PDF | $\frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$ |
| $E(X)$ | $\mu$ |
| $\text{Var}(X)$ | $\sigma^2$ |
| MGF | $e^{\mu t + \sigma^2 t^2/2}$ |
| 典型场景 | CLT 极限分布；测量误差；自然界中大量微小随机影响的叠加 |

**标准化**：$Z = (X - \mu)/\sigma \sim N(0,1)$，查标准正态表。

**线性组合**：若 $X_i \sim N(\mu_i, \sigma_i^2)$ 独立，则 $\sum a_i X_i \sim N(\sum a_i \mu_i, \sum a_i^2 \sigma_i^2)$。

**经验法则（68-95-99.7）**：$P(\mu - \sigma < X < \mu + \sigma) \approx 68\%$，$P(\mu - 2\sigma < X < \mu + 2\sigma) \approx 95\%$，$P(\mu - 3\sigma < X < \mu + 3\sigma) \approx 99.7\%$。

### 3.4 Gamma 分布

$$X \sim \Gamma(\alpha, \beta), \quad f(x) = \frac{x^{\alpha-1} e^{-x/\beta}}{\Gamma(\alpha)\beta^\alpha}, \quad x > 0.$$

（$\alpha > 0$ 为形状参数，$\beta > 0$ 为尺度参数；也有用率参数 $\theta = 1/\beta$ 的参数化。）

| 项目 | 值 |
|---|---|
| $E(X)$ | $\alpha\beta$ |
| $\text{Var}(X)$ | $\alpha\beta^2$ |
| MGF | $(1 - \beta t)^{-\alpha}$，$t < 1/\beta$ |
| 典型场景 | 等待 $\alpha$ 次 Poisson 事件的总时间；生存分析；贝叶斯 Poisson 参数的共轭先验 |

**特殊情形**：$\Gamma(1, \beta) = \text{Exp}(1/\beta)$；$\Gamma(n/2, 2) = \chi^2_n$（见下）。

**可加性**：若 $X_i \sim \Gamma(\alpha_i, \beta)$ 独立（同尺度参数），则 $\sum X_i \sim \Gamma(\sum \alpha_i, \beta)$。

### 3.5 Beta 分布

$$X \sim \text{Beta}(\alpha, \beta), \quad f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}, \quad 0 < x < 1.$$

（$B(\alpha,\beta) = \Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)$ 为 Beta 函数。）

| 项目 | 值 |
|---|---|
| $E(X)$ | $\alpha/(\alpha+\beta)$ |
| $\text{Var}(X)$ | $\alpha\beta / [(\alpha+\beta)^2(\alpha+\beta+1)]$ |
| 典型场景 | $[0,1]$ 上的比例；贝叶斯二项分布参数 $p$ 的共轭先验 |

**特殊情形**：$\text{Beta}(1,1) = U(0,1)$；$\alpha = \beta$ 时关于 $1/2$ 对称。

### 3.6 $\chi^2$ 分布

设 $Z_1, \ldots, Z_n$ i.i.d. $\sim N(0,1)$，则

$$\chi^2_n = \sum_{i=1}^n Z_i^2 \sim \Gamma(n/2, 2).$$

| 项目 | 值 |
|---|---|
| $E(\chi^2_n)$ | $n$ |
| $\text{Var}(\chi^2_n)$ | $2n$ |
| 典型场景 | 正态总体样本方差：$(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$；拟合优度检验 |

**可加性**：$\chi^2_m + \chi^2_n = \chi^2_{m+n}$（独立时）。

### 3.7 $t$ 分布

设 $Z \sim N(0,1)$，$V \sim \chi^2_n$，两者独立，则

$$t_n = \frac{Z}{\sqrt{V/n}} \sim t_n.$$

| 项目 | 值 |
|---|---|
| $E(t_n)$ | $0$（$n > 1$） |
| $\text{Var}(t_n)$ | $n/(n-2)$（$n > 2$） |
| 典型场景 | 正态总体均值检验（方差未知）：$T = (\bar{X}-\mu_0)/(S/\sqrt{n}) \sim t_{n-1}$ |

**与正态的关系**：$t_n \to N(0,1)$（$n \to \infty$）；$n$ 小时尾部更重。$t_1$ = Cauchy 分布（无期望）。

### 3.8 $F$ 分布

设 $U \sim \chi^2_m$，$V \sim \chi^2_n$，两者独立，则

$$F_{m,n} = \frac{U/m}{V/n} \sim F(m, n).$$

| 项目 | 值 |
|---|---|
| $E(F_{m,n})$ | $n/(n-2)$（$n > 2$） |
| 典型场景 | 两正态总体方差比检验：$F = S_1^2/S_2^2$ 在方差相等时 $\sim F(n_1-1, n_2-1)$；方差分析（ANOVA） |

**倒数关系**：若 $F \sim F(m,n)$，则 $1/F \sim F(n,m)$。

---

## 四、分布间的关系图

以下关系是推导统计量分布的基础，建议熟记：

$$\text{Bernoulli}(p) \xrightarrow{\text{求和 }n \text{ 个独立}} B(n,p) \xrightarrow{n\to\infty,\, np=\lambda} \text{Poisson}(\lambda)$$

$$\text{Geom}(p) \xrightarrow{\text{等待第 } r \text{ 次}} \text{NB}(r,p)$$

$$\text{Exp}(\lambda) = \Gamma(1, 1/\lambda) \xrightarrow{\text{求和 } n \text{ 个独立}} \Gamma(n, 1/\lambda)$$

$$N(0,1)^2 \xrightarrow{\text{一个}} \chi^2_1 \xrightarrow{\text{求和 } n \text{ 个独立}} \chi^2_n = \Gamma(n/2, 2)$$

$$\frac{N(0,1)}{\sqrt{\chi^2_n/n}} = t_n \xrightarrow{n\to\infty} N(0,1)$$

$$\frac{\chi^2_m/m}{\chi^2_n/n} = F(m,n), \qquad t_n^2 = F(1,n)$$

$$\text{Beta}(1,1) = U(0,1), \qquad \text{Beta}\!\left(\frac{n}{2}, \frac{n}{2}\right) \xrightarrow{n\to\infty} \text{对称} \to N(0,1) \text{（标准化后）}$$

---

## 五、演示题：识别"等公交"= 指数分布（无记忆性）

### 题目

公交车平均每 10 分钟一班，按 Poisson 过程到达（即相邻两班到达时间间隔独立且同分布）。设你刚错过一班，等待时间为 $T$。
1. 写出 $T$ 的分布，求 $E(T)$ 和 $\text{Var}(T)$；
2. 已知你已经等了 5 分钟，问再等 8 分钟以上（即总等待超过 13 分钟）的概率；
3. 对比：如果公交不是 Poisson 过程，而是严格按 10 分钟班次运行（你不知道上次是何时），$T$ 服从什么分布？

> 分析步骤如下：
>
> **识别分布**：Poisson 过程相邻事件间隔服从指数分布，参数 $\lambda = 1/10$（每分钟平均 0.1 班）。
>
> $$T \sim \text{Exp}(1/10), \quad f(t) = \frac{1}{10}e^{-t/10}, \quad t \geq 0.$$
>
> **第 1 问**：
> $$E(T) = \frac{1}{\lambda} = 10 \text{ 分钟}, \quad \text{Var}(T) = \frac{1}{\lambda^2} = 100 \text{ 分钟}^2.$$
>
> **第 2 问（无记忆性）**：
> $$P(T > 13 \mid T > 5) = P(T > 8) = e^{-8/10} = e^{-0.8} \approx 0.449.$$
>
> 关键：已经等了 5 分钟的信息毫无价值——"你已经等了多久"不影响"还要再等多久"的分布。这就是指数分布的无记忆性：
> $$P(T > s+t \mid T > s) = \frac{P(T > s+t)}{P(T > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(T > t).$$
>
> **第 3 问（严格班次）**：若公交严格每 10 分钟一班，你到达时刻在 $[0, 10)$ 上均匀分布（不知道上次班次时刻），等待时间 $T \sim U(0, 10)$。此时 $E(T) = 5$ 分钟（比 Poisson 过程少等）且不满足无记忆性。
>
> **对比小结**：
> - Poisson 过程（无记忆）→ 指数分布，$E(T) = 10$ 分钟；
> - 严格班次（均匀到达）→ 均匀分布，$E(T) = 5$ 分钟；
> - 随机性越大，等待时间越长。

---

## 六、思考路标

1. **识别分布的第一步** → 看"样本空间 + 试验结构"：有限次独立试验 → 二项；稀有事件计数 → Poisson；首次成功 → 几何；等待时间/寿命 → 指数；归一化平方和 → $\chi^2$；比值 → $F$；均值标准化 → $t$。

2. **无记忆性是指数分布的充要特征** → 若等待时间满足无记忆性（过去等待时间不影响未来），则该连续分布必为指数分布。见"已经等了 $s$ 时间，还要等多久"的条件概率问题，立刻用无记忆性化简。

3. **Poisson 分布的识别标志** → 期望 = 方差（$E(X) = \text{Var}(X) = \lambda$）。若数据显示均值远小于方差，可能是过离散（Negative Binomial），应换模型。

4. **$\chi^2, t, F$ 的推导链** → 先写出统计量的构成（哪些标准正态的平方，哪些独立），再套关系图。$t = N(0,1)/\sqrt{\chi^2_n/n}$，$F = \chi^2_m/m \div \chi^2_n/n$，$t_n^2 = F(1,n)$。

5. **贝叶斯共轭先验识别** → Beta 是二项似然的共轭先验；Gamma 是 Poisson 似然的共轭先验；正态是正态似然（已知方差）的共轭先验。选共轭先验，后验与先验同族，计算简单。

6. **Gamma 分布的两种参数化** → 形状-尺度（$\alpha, \beta$，$E = \alpha\beta$）和形状-率（$\alpha, \theta = 1/\beta$，$E = \alpha/\theta$）。见 Gamma 分布先确认用的是哪种参数化，以免期望和方差算错。

7. **正态分布的线性封闭性** → 正态变量的任意线性组合仍是正态（独立时）。这是它在统计推断中无处不在的根本原因——样本均值是正态，$Z$ 统计量是正态，CLT 的极限是正态。

8. **尾部重轻的直觉** → 正态 $\ll$ $t_n$（$n$ 小）$\ll$ Cauchy（= $t_1$，无期望）。做区间估计时，样本量小用 $t$ 分布（尾部更重，置信区间更宽），大样本用正态近似。

---

## 七、典型应用 3 例

### 例 1：Poisson 近似二项

**题目**：某工厂每天生产 10000 件产品，次品率 $p = 0.0003$。用 Poisson 近似计算一天内次品超过 5 件的概率。

**思路**：

$n = 10000$，$p = 0.0003$，$\lambda = np = 3$。$X \sim B(10000, 0.0003) \approx \text{Poisson}(3)$。

$$P(X > 5) = 1 - P(X \leq 5) = 1 - \sum_{k=0}^5 \frac{e^{-3} 3^k}{k!}.$$

计算：$P(X \leq 5) = e^{-3}(1 + 3 + 4.5 + 4.5 + 3.375 + 2.025) \approx e^{-3} \times 18.4 \approx 0.9161$。

$$P(X > 5) \approx 1 - 0.9161 = 0.0839.$$

**条件验证**：$n = 10000$ 很大，$p = 0.0003$ 很小，$\lambda = 3$ 适中，满足 Poisson 近似条件。

---

### 例 2：正态分布 + 标准化

**题目**：$X \sim N(50, 100)$（均值 50，方差 100，即 $\sigma = 10$），求 $P(40 < X < 65)$。

**思路**：

标准化：令 $Z = (X - 50)/10 \sim N(0,1)$。

$$P(40 < X < 65) = P\!\left(\frac{40-50}{10} < Z < \frac{65-50}{10}\right) = P(-1 < Z < 1.5).$$

查标准正态表：$\Phi(1.5) \approx 0.9332$，$\Phi(-1) = 1 - \Phi(1) \approx 1 - 0.8413 = 0.1587$。

$$P(-1 < Z < 1.5) = 0.9332 - 0.1587 = 0.7745.$$

---

### 例 3：用 $\chi^2$ 和 $t$ 分布推导置信区间

**题目**：$X_1, \ldots, X_n$ 独立同分布 $\sim N(\mu, \sigma^2)$（$\sigma^2$ 未知），构造 $\mu$ 的 $95\%$ 置信区间。

**思路**：

已知 $\bar{X} \sim N(\mu, \sigma^2/n)$ 且 $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$，两者独立。

统计量：
$$T = \frac{\bar{X} - \mu}{S/\sqrt{n}} = \frac{(\bar{X}-\mu)/(\sigma/\sqrt{n})}{\sqrt{(n-1)S^2/[\sigma^2(n-1)]}} = \frac{N(0,1)}{\sqrt{\chi^2_{n-1}/(n-1)}} \sim t_{n-1}.$$

由 $t_{n-1}$ 分布的分位数 $t_{\alpha/2, n-1}$（$\alpha = 0.05$）：

$$P\!\left(-t_{0.025, n-1} \leq T \leq t_{0.025, n-1}\right) = 0.95.$$

解出 $\mu$：置信区间为 $\bar{X} \pm t_{0.025, n-1} \cdot S/\sqrt{n}$。

**关系链**：正态假设 → 样本均值 $N$ → 样本方差 $\chi^2$ → $T$ 统计量服从 $t$ 分布 → 区间端点用 $t$ 分位数。

---

## 八、自测题

**第 1 题**：某超市每分钟平均有 $\lambda = 2$ 名顾客到达（Poisson 过程），求 5 分钟内到达顾客数为 0 的概率，以及到达数 $\leq 8$ 的概率。

提示：5 分钟内 $X \sim \text{Poisson}(10)$；$P(X=0) = e^{-10} \approx 0.0000454$；$P(X \leq 8) = \sum_{k=0}^8 e^{-10}10^k/k!$（查 Poisson 表或计算约 $0.333$）。

---

**第 2 题**：$X \sim \text{Exp}(2)$（率参数 $\lambda=2$），求 $E(X)$、$\text{Var}(X)$、$P(X > 1)$，以及已知 $X > 0.5$ 条件下 $P(X > 1.5)$。

提示：$E(X) = 1/2$，$\text{Var}(X) = 1/4$；$P(X > 1) = e^{-2}$；由无记忆性 $P(X > 1.5 \mid X > 0.5) = P(X > 1) = e^{-2} \approx 0.135$。

---

**第 3 题**：设 $Z_1, Z_2, Z_3$ i.i.d. $\sim N(0,1)$，$V = Z_1^2 + Z_2^2 + Z_3^2$，$T = Z_1 / \sqrt{V/3}$，说明 $V$ 和 $T$ 各服从什么分布，并求 $E(V)$ 和 $\text{Var}(V)$。

提示：$V \sim \chi^2_3$，$E(V) = 3$，$\text{Var}(V) = 6$；$T \sim t_3$（$Z_1 \perp Z_2^2+Z_3^2$ 需要验证，或注意 $V = Z_1^2 + (Z_2^2+Z_3^2)$ 分解方式）。

---

**第 4 题**：从含 12 件合格品和 3 件次品（共 15 件）的批次中，不放回抽取 5 件。设次品数为 $X$，写出 $X$ 的分布类型和 $E(X)$。

提示：$X \sim$ 超几何分布（$N=15, K=3, n=5$）；$E(X) = nK/N = 5 \times 3/15 = 1$。

---

**第 5 题**：$X \sim B(n, p)$。若 $E(X) = 6$，$\text{Var}(X) = 4.2$，求 $n$ 和 $p$。

提示：$np = 6$，$np(1-p) = 4.2$，联立得 $1-p = 4.2/6 = 0.7$，即 $p = 0.3$；$n = 6/0.3 = 20$。
