# F4b 详解：Part 5 统计基础（Ch.13-15，共 23 题）

> 覆盖：抽样分布（Ch.13）、数据描述（Ch.14）、充分统计量（Ch.15）。
> 核心公式：$\bar X \sim N(\mu,\sigma^2/n)$；$(n-1)S^2/\sigma^2\sim\chi^2(n-1)$；$\bar X\perp S^2$（正态总体）；$T=(\bar X-\mu)/(S/\sqrt{n})\sim t(n-1)$。

---

## C 基础题详解（8 题）

### C.5.1（Ch.13，样本统计量的定义）

**题目**：样本 $\{3,5,7,9,6\}$（$n=5$），计算 $\bar x$ 和 $s^2$（分母 $n-1$）。

**思路**：先求均值，再按公式计算修正样本方差。

**解**：

**1. 样本均值**

$$\bar x = \frac{3+5+7+9+6}{5} = \frac{30}{5} = 6.$$

**2. 样本方差**（分母 $n-1=4$）

| $x_i$ | $x_i - \bar x$ | $(x_i-\bar x)^2$ |
|:---:|:---:|:---:|
| 3 | $-3$ | 9 |
| 5 | $-1$ | 1 |
| 7 | $+1$ | 1 |
| 9 | $+3$ | 9 |
| 6 | $0$  | 0 |
| **合计** | | **20** |

$$s^2 = \frac{20}{4} = 5.$$

**答案**：$\boxed{\bar x = 6,\quad s^2 = 5}$

---

### C.5.2（Ch.13，抽样分布——样本均值）

**题目**：总体 $X\sim N(10,16)$，$n=4$，求 $\bar X$ 的分布及 $P(\bar X>12)$。

**思路**：正态总体样本均值仍服从正态分布，方差缩小 $n$ 倍。

**解**：

**1. $\bar X$ 的精确分布**

$$\bar X \sim N\!\left(\mu,\frac{\sigma^2}{n}\right) = N\!\left(10,\frac{16}{4}\right) = N(10,4).$$

标准差 $\sigma_{\bar X}=\sqrt{4}=2$。

**2. 计算概率**

$$P(\bar X > 12) = P\!\left(\frac{\bar X-10}{2} > \frac{12-10}{2}\right) = P(Z>1) = 1 - \Phi(1).$$

查表 $\Phi(1)\approx0.8413$，故 $P(\bar X>12)\approx 0.1587$。

**答案**：$\bar X\sim N(10,4)$；$P(\bar X>12)=1-\Phi(1)\approx\boxed{0.1587}$

---

### C.5.3（Ch.13，$\chi^2$ 分布的性质）

**题目**：$Z_1,Z_2,Z_3\overset{iid}{\sim}N(0,1)$，$W=Z_1^2+Z_2^2+Z_3^2$，求分布及矩。

**思路**：独立标准正态的平方和服从 $\chi^2$ 分布，自由度等于项数。

**解**：

**1. 分布**：$W=\chi^2(3)$（自由度 $k=3$）。

**2. 期望与方差**：$\chi^2(k)$ 的期望为 $k$，方差为 $2k$，故

$$E[W] = 3,\quad \mathrm{Var}(W) = 6.$$

**答案**：$W\sim\chi^2(3)$；$\boxed{E[W]=3,\;\mathrm{Var}(W)=6}$

---

### C.5.4（Ch.13，$t$ 分布的构造）

**题目**：$Z\sim N(0,1)$，$V\sim\chi^2(9)$，$Z\perp V$，$T=Z/\sqrt{V/9}$，写出分布并给出拒绝域。

**思路**：标准正态除以独立 $\chi^2$ 的"均方根"即 $t$ 分布，自由度等于 $\chi^2$ 的自由度。

**解**：

**1. 分布**：由 $t$ 分布定义，$T\sim t(9)$。

**2. 双侧拒绝域**（$\alpha=0.05$）

$$\text{拒绝域} = \{|T|>t_{0.025}(9)=2.262\}.$$

**答案**：$T\sim t(9)$；拒绝域 $\boxed{|T|>2.262}$

---

### C.5.5（Ch.14，样本分位数与中位数）

**题目**：数据 $\{4,1,9,3,7,2,8\}$，求有序样本、中位数、均值并比较。

**思路**：奇数个数据时中位数为正中间那个值；均值受极端值影响。

**解**：

**1. 有序样本**（$n=7$）：$1,2,3,4,7,8,9$。

**2. 样本中位数**：第 $\lceil 7/2\rceil=4$ 个值，$\hat m = 4$。

**3. 样本均值**

$$\bar x = \frac{1+2+3+4+7+8+9}{7} = \frac{34}{7} \approx 4.857.$$

均值 $4.857$ 大于中位数 $4$，因为较大的值（7,8,9）拉高了均值。

**答案**：$\hat m=4$；$\bar x=34/7\approx\boxed{4.857}$；均值 $>$ 中位数（右偏）

---

### C.5.6（Ch.14，偏度与峰度的含义）

**题目**：四个分布（A）$N(0,1)$；（B）$\mathrm{Exp}(1)$；（C）$U(0,1)$；（D）拉普拉斯分布，哪个偏度为 0？哪个峰度最大？

**思路**：偏度反映分布对称性，正态/均匀/拉普拉斯关于均值对称偏度为 0；峰度衡量尾部厚重程度。

**解**：

- **偏度为 0**：（A）$N(0,1)$、（C）$U(0,1)$、（D）拉普拉斯分布均关于均值对称，偏度 $=0$；（B）指数分布右偏，偏度 $=2>0$。
- **峰度最大**：（D）拉普拉斯分布尖峰重尾，超额峰度（excess kurtosis）$=3$；$N(0,1)$ 超额峰度 $=0$；$U(0,1)$ 超额峰度 $=-6/5$；指数分布超额峰度 $=6$。

> ⚠️ 若题目指"超额峰度"（excess kurtosis），则（B）指数分布峰度最大（$=6$），远超拉普拉斯（$=3$）。

**答案**：偏度为 0 的有 $\boxed{(A)(C)(D)}$；峰度最大的是 $\boxed{(B)\;\mathrm{Exp}(1)}$（超额峰度 $=6$）

---

### C.5.7（Ch.15，充分统计量判定）

**题目**：$X_i\overset{iid}{\sim}\mathrm{Poisson}(\lambda)$，写出联合 PMF 并用因子分解定理证明 $T=\sum X_i$ 是充分统计量。

**思路**：将联合 PMF 分离成仅依赖 $T$ 和 $\lambda$ 的因子与仅依赖样本的因子。

**解**：

**1. 联合 PMF**

$$\prod_{i=1}^n P(X_i=x_i) = \prod_{i=1}^n \frac{e^{-\lambda}\lambda^{x_i}}{x_i!} = \frac{e^{-n\lambda}\lambda^{\sum x_i}}{\prod_{i=1}^n x_i!}.$$

**2. 因子分解**

$$= \underbrace{e^{-n\lambda}\lambda^{T}}_{g(T,\lambda)} \cdot \underbrace{\frac{1}{\prod_{i=1}^n x_i!}}_{h(\mathbf{x})}$$

其中 $T=\sum_{i=1}^n x_i$。满足 Neyman-Fisher 因子分解定理条件，故 $T=\sum X_i$ 是 $\lambda$ 的充分统计量。

**答案**：$T=\sum_{i=1}^nX_i$ 是 $\lambda$ 的充分统计量。$\boxed{T=\sum_{i=1}^nX_i}$

---

### C.5.8（Ch.15，指数族与自然充分统计量）

**题目**：正态分布 $N(\mu,\sigma^2)$（$\sigma^2$ 已知）的自然充分统计量及 $\bar X$ 为何充分。

**思路**：将正态 PDF 写成指数族标准形，读出充分统计量。

**解**：

单个观测的 PDF 为

$$f(x;\mu) = \frac{1}{\sqrt{2\pi}\sigma}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) = \underbrace{\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{x^2}{2\sigma^2}}}_{h(x)}\exp\!\left(\frac{\mu}{\sigma^2}x - \frac{\mu^2}{2\sigma^2}\right).$$

自然参数 $\eta=\mu/\sigma^2$，自然充分统计量 $T(x)=x$。

对 $n$ 个样本，充分统计量为 $T=\sum_{i=1}^n x_i$（等价于 $\bar X$）。$\bar X$ 是 $\mu$ 的充分统计量，因为联合密度可分解为仅依赖 $\bar x$ 和 $\mu$ 的因子乘以不含 $\mu$ 的因子。

**答案**：自然充分统计量 $\boxed{T=\sum_{i=1}^nX_i}$（即 $\bar X$）；正态族属于指数族，$\bar X$ 自然地是充分统计量。

---

## D 中等题详解（10 题）

### D.5.1（Ch.13，常见统计量的分布）

**题目**：$X_i\overset{iid}{\sim}N(\mu,\sigma^2)$，证明 $(n-1)S^2/\sigma^2\sim\chi^2(n-1)$；写出 $T$ 和 $F$ 统计量分布。

**思路**：利用正交变换将 $\sum(X_i-\bar X)^2$ 化为独立标准正态的平方和；$T$ 分布由定义给出；$F$ 分布由两独立 $\chi^2$ 之比给出。

**解**：

**(a) $\bar X$ 的分布**

$$\bar X \sim N\!\left(\mu,\frac{\sigma^2}{n}\right).$$

**证明 $(n-1)S^2/\sigma^2\sim\chi^2(n-1)$**：

令 $Y_i=(X_i-\mu)/\sigma\sim N(0,1)$，则

$$\sum_{i=1}^n\frac{(X_i-\mu)^2}{\sigma^2} = \frac{n(\bar X-\mu)^2}{\sigma^2} + \frac{(n-1)S^2}{\sigma^2}.$$

左边 $\sim\chi^2(n)$，右边第一项 $\sim\chi^2(1)$。由于 $\bar X\perp S^2$（正态总体的独立性定理），两项独立，由 $\chi^2$ 分布的可加性：

$$\frac{(n-1)S^2}{\sigma^2} = \chi^2(n) - \chi^2(1) \sim \chi^2(n-1).$$

**(b) $T$ 统计量**

$$T = \frac{\bar X-\mu}{S/\sqrt{n}} = \frac{(\bar X-\mu)/(\sigma/\sqrt{n})}{\sqrt{(n-1)S^2/[\sigma^2(n-1)]}}\sim t(n-1).$$

自由度 $n-1$ 来自分母中 $\chi^2(n-1)$ 的自由度。

**(c) 方差比 $F$ 统计量**

设两独立样本 $n_1,n_2$，总体方差相同 $\sigma_1^2=\sigma_2^2$：

$$F = \frac{S_1^2}{S_2^2} = \frac{(n_1-1)S_1^2/\sigma^2/(n_1-1)}{(n_2-1)S_2^2/\sigma^2/(n_2-1)} \sim F(n_1-1,\,n_2-1).$$

**答案**：$\bar X\sim N(\mu,\sigma^2/n)$；$(n-1)S^2/\sigma^2\sim\chi^2(n-1)$；$T\sim\boxed{t(n-1)}$；$F\sim F(n_1-1,n_2-1)$

---

### D.5.2（Ch.13，顺序统计量）

**题目**：推导 $X_{(k)}$ 的密度；对 $U(0,1)$ 求 $X_{(n)}$ 和 $X_{(3)}$（$n=5$）的密度及期望。

**思路**：恰好有 $k-1$ 个小于 $x$、1 个等于 $x$、$n-k$ 个大于 $x$，利用多项式计数推导密度。

**解**：

**(a) 推导 $X_{(k)}$ 密度**

$X_{(k)}\leq x$ 等价于至少 $k$ 个观测 $\leq x$。对密度微分：

$$f_{X_{(k)}}(x) = \frac{n!}{(k-1)!(n-k)!}[F(x)]^{k-1}[1-F(x)]^{n-k}f(x).$$

组合解释：从 $n$ 个中选 $k-1$ 个排在 $x$ 左边（概率 $[F(x)]^{k-1}$），选 $n-k$ 个排在 $x$ 右边（概率 $[1-F(x)]^{n-k}$），第 $k$ 个恰在 $x$ 处（密度 $f(x)$）。

**(b) $X_i\sim U(0,1)$，$X_{(n)}$ 的密度和期望**

$F(x)=x$，$f(x)=1$，代入公式（$k=n$）：

$$f_{X_{(n)}}(x) = \frac{n!}{(n-1)!\,0!} x^{n-1}\cdot 1 = nx^{n-1},\quad x\in(0,1).$$

$$E[X_{(n)}] = \int_0^1 x\cdot nx^{n-1}\,dx = n\int_0^1 x^n\,dx = \frac{n}{n+1}.$$

**(c) $n=5$，$k=3$（中位数）**

$$f_{X_{(3)}}(x) = \frac{5!}{2!\,2!}x^2(1-x)^2 = 30\,x^2(1-x)^2,\quad x\in(0,1).$$

**答案**：$f_{X_{(n)}}(x)=nx^{n-1}$；$E[X_{(n)}]=\dfrac{n}{n+1}$；$f_{X_{(3)}}(x)=\boxed{30x^2(1-x)^2}$

---

### D.5.3（Ch.14，经验分布函数）

**题目**：经验 CDF 的性质、Glivenko-Cantelli 定理，及 $\{1,3,3,5,7\}$ 的具体表达式。

**思路**：固定 $x$，$n\hat F_n(x)$ 是 $B(n,F(x))$，矩由此给出；Glivenko-Cantelli 是 SLLN 的一致版本。

**解**：

**(a) 期望与方差**

对固定 $x$，令 $I_i=\mathbf{1}[X_i\leq x]\overset{iid}{\sim}\mathrm{Bernoulli}(F(x))$，则

$$\hat F_n(x)=\frac{1}{n}\sum_{i=1}^nI_i \Rightarrow n\hat F_n(x)\sim B(n,F(x)).$$

$$E[\hat F_n(x)]=F(x),\quad \mathrm{Var}(\hat F_n(x))=\frac{F(x)(1-F(x))}{n}.$$

**(b) Glivenko-Cantelli 定理**

对每个固定 $x$，SLLN 给出 $\hat F_n(x)\xrightarrow{a.s.}F(x)$。

对有限点集 $\{x_1,\ldots,x_m\}$ 的一致收敛由有限并集的概率论给出；再利用 $\hat F_n$ 和 $F$ 的单调性，可将有限点集的收敛推广到全实线，得

$$\sup_x|\hat F_n(x)-F(x)|\xrightarrow{a.s.}0.$$

**(c) 样本 $\{1,3,3,5,7\}$ 的经验 CDF**

$$\hat F_5(x) = \begin{cases} 0, & x < 1 \\ 1/5, & 1 \leq x < 3 \\ 3/5, & 3 \leq x < 5 \\ 4/5, & 5 \leq x < 7 \\ 1, & x \geq 7 \end{cases}$$

> ⚠️ 重复值 $3$ 出现两次，$\hat F_5(3)=3/5$（两个 $3$ 均 $\leq 3$）。

**答案**：$\boxed{E[\hat F_n(x)]=F(x),\;\mathrm{Var}(\hat F_n(x))=F(x)(1-F(x))/n}$；Glivenko-Cantelli 定理成立；$\hat F_5$ 见上表。

---

### D.5.4（Ch.15，充分统计量的因子分解定理）

**题目**：陈述 Neyman-Fisher 因子分解定理；证明 $N(\mu,1)$ 的 $\bar X$ 充分；求 $\mathrm{Exp}(\theta)$ 的充分统计量。

**思路**：将联合密度改写，识别仅依赖 $T$ 和 $\theta$ 的部分与不依赖 $\theta$ 的部分。

**解**：

**(a) Neyman-Fisher 因子分解定理**

$T(\mathbf{X})$ 是 $\theta$ 的充分统计量，当且仅当存在非负函数 $g$ 和 $h$，使得

$$\prod_{i=1}^n f(x_i;\theta) = g\!\bigl(T(\mathbf{x}),\theta\bigr)\cdot h(\mathbf{x}).$$

**(b) $X_i\sim N(\mu,1)$，证明 $\bar X$ 充分**

$$\prod_{i=1}^n f(x_i;\mu) = \prod_{i=1}^n\frac{1}{\sqrt{2\pi}}\exp\!\left(-\frac{(x_i-\mu)^2}{2}\right) = \frac{1}{(2\pi)^{n/2}}\exp\!\left(-\frac{1}{2}\sum(x_i-\mu)^2\right).$$

展开：$\sum(x_i-\mu)^2 = \sum x_i^2 - 2\mu\sum x_i + n\mu^2$，故

$$= \underbrace{\exp\!\left(\mu\sum x_i - \frac{n\mu^2}{2}\right)}_{g(\bar x,\mu)}\cdot\underbrace{\frac{1}{(2\pi)^{n/2}}\exp\!\left(-\frac{\sum x_i^2}{2}\right)}_{h(\mathbf{x})}.$$

$g$ 仅通过 $\sum x_i=n\bar x$ 依赖 $\mu$，故 $\bar X$（等价于 $T=\sum X_i$）是 $\mu$ 的充分统计量。

**(c) $X_i\sim\mathrm{Exp}(\theta)$（密度 $\theta e^{-\theta x}$）**

$$\prod_{i=1}^n\theta e^{-\theta x_i} = \theta^n e^{-\theta\sum x_i} = \underbrace{\theta^n e^{-\theta T}}_{g(T,\theta)}\cdot\underbrace{1}_{h(\mathbf{x})}.$$

其中 $T=\sum_{i=1}^n X_i$，故充分统计量为 $\boxed{T=\sum_{i=1}^n X_i}$（等价于 $\bar X$）。

---

### D.5.5（Ch.13，Bootstrap 基本思想）

**题目**：参数 Bootstrap 步骤；Bootstrap 标准误差公式；样本 $\{2,4,6,8\}$ 的 $\bar X^*$ 范围。

**思路**：Bootstrap 用经验分布 $\hat F_n$ 替代未知真实分布 $F$，通过重抽样模拟统计量的分布。

**解**：

**(a) 参数 Bootstrap 步骤**

1. 从原样本 $\{X_1,\ldots,X_n\}$ 得到 $\hat F_n$（非参数）或拟合参数模型 $\hat\theta$（参数）。
2. 从 $\hat F_n$（或 $\hat F_{\hat\theta}$）有放回抽取 $n$ 个样本，得 Bootstrap 样本 $X_1^*,\ldots,X_n^*$。
3. 计算 $T_n^{*(b)}=T(X_1^*,\ldots,X_n^*)$，重复 $B$ 次。
4. 用 $\{T_n^{*(1)},\ldots,T_n^{*(B)}\}$ 的分布近似 $T_n$ 的分布。

**(b) Bootstrap 标准误差**

$$\widehat{\mathrm{SE}}_{\text{boot}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^B\bigl(T_n^{*(b)}-\bar T^*\bigr)^2},\quad \bar T^*=\frac{1}{B}\sum_{b=1}^B T_n^{*(b)}.$$

**(c) $\{2,4,6,8\}$ 的 Bootstrap 均值范围**

- 最小 $\bar X^*$：抽到 $(2,2,2,2)$，均值 $=2$。
- 最大 $\bar X^*$：抽到 $(8,8,8,8)$，均值 $=8$。

**答案**：Bootstrap 标准误差公式见上；$\bar X^*$ 范围为 $\boxed{[2,\;8]}$

---

### D.5.6（Ch.14，Q-Q 图原理）

**题目**：Q-Q 图的理论依据；重尾时的弯曲方式；样本 $\{-1.5,-0.3,0.1,0.8,1.6\}$ vs $N(0,1)$。

**思路**：顺序统计量 $X_{(i)}$ 的近似期望等于理论分布的 $(i-0.5)/n$ 分位数，如果理论假设正确，点对成直线。

**解**：

**(a) Q-Q 图的理论依据**

当 $X_i\sim F_0$ 时，第 $i$ 个顺序统计量的近似期望为

$$E[X_{(i)}] \approx F_0^{-1}\!\left(\frac{i-0.5}{n}\right) =: q_i.$$

Q-Q 图绘制 $(q_i, X_{(i)})$；若 $F=F_0$，则 $X_{(i)}\approx q_i$，各点近似在直线 $y=x$ 上（或经平移伸缩后在某直线上）。

**(b) 重尾时的弯曲**

若实际分布比 $F_0$ 重尾（如 $t$ vs 正态），两端的样本分位数比理论分位数更极端，Q-Q 图两端向上弯曲（S 形），即左端点在直线下方，右端点在直线上方。

**(c) 计算理论分位数**（$n=5$，$p_i=(i-0.5)/5$）

| $i$ | $p_i$ | $\Phi^{-1}(p_i)$ | $X_{(i)}$ |
|:---:|:---:|:---:|:---:|
| 1 | 0.10 | $-1.282$ | $-1.5$ |
| 2 | 0.30 | $-0.524$ | $-0.3$ |
| 3 | 0.50 | $0.000$ | $0.1$ |
| 4 | 0.70 | $+0.524$ | $0.8$ |
| 5 | 0.90 | $+1.282$ | $1.6$ |

点基本在直线附近，支持数据来自正态分布的假设。

**答案**：理论分位数见上表；重尾分布 Q-Q 图两端向上弯曲（S 形）。$\boxed{\Phi^{-1}(0.1)\approx-1.282,\;\ldots,\;\Phi^{-1}(0.9)\approx1.282}$

---

### D.5.7（Ch.15，完备充分统计量）

**题目**：$X_i\overset{iid}{\sim}\mathrm{Poisson}(\lambda)$，$T=\sum X_i$：证明 $T\sim\mathrm{Poisson}(n\lambda)$，验证充分性，陈述完备性。

**思路**：Poisson 的可加性给出 $T$ 的分布；因子分解给出充分性；指数族的完备性由参数空间内点性质保证。

**解**：

**(a) $T\sim\mathrm{Poisson}(n\lambda)$**

对独立 Poisson 随机变量求和，用特征函数（或矩生成函数）：

$$\varphi_T(\omega) = \prod_{i=1}^n\exp\!\bigl(\lambda(e^{i\omega}-1)\bigr) = \exp\!\bigl(n\lambda(e^{i\omega}-1)\bigr),$$

即 $T\sim\mathrm{Poisson}(n\lambda)$。

**(b) 因子分解验证充分性**

$$\prod_{i=1}^n\frac{e^{-\lambda}\lambda^{x_i}}{x_i!} = e^{-n\lambda}\lambda^T\cdot\frac{1}{\prod x_i!} = g(T,\lambda)\cdot h(\mathbf{x}).$$

由因子分解定理，$T$ 是 $\lambda$ 的充分统计量。

**(c) 完备性**

**定义**：若对所有有界函数 $g$，$E_\lambda[g(T)]=0$ 对所有 $\lambda>0$ 成立，则 $g(T)=0$ a.e.，称 $T$ 是完备的。

**验证**：$T\sim\mathrm{Poisson}(n\lambda)$ 属于指数族（自然参数 $\eta=\log\lambda$），自然参数空间 $\eta\in(-\infty,+\infty)$ 包含内点，由 Lehmann-Scheffé 定理，$T$ 是完备充分统计量。

**答案**：$T\sim\mathrm{Poisson}(n\lambda)$；$T$ 是完备充分统计量。$\boxed{T=\sum X_i\sim\mathrm{Poisson}(n\lambda)}$

---

### D.5.8（Ch.13，抽样分布的数值例）

**题目**：总体 $N(5,4)$，$n=16$：求 $P(\bar X>6)$、$P(S^2>6.908)$、方差比分布。

**思路**：标准化后查正态表；利用 $(n-1)S^2/\sigma^2\sim\chi^2(n-1)$ 转化；两独立正态方差比为 $F$ 分布。

**解**：

**(a) $P(\bar X>6)$**

$\bar X\sim N(5,4/16)=N(5,0.25)$，$\sigma_{\bar X}=0.5$。

$$P(\bar X>6)=P\!\left(Z>\frac{6-5}{0.5}\right)=P(Z>2)=1-\Phi(2)\approx 1-0.9772=\boxed{0.0228}.$$

**(b) $P(S^2>6.908)$**

$$\frac{(n-1)S^2}{\sigma^2}=\frac{15S^2}{4}\sim\chi^2(15).$$

$$P(S^2>6.908)=P\!\left(\chi^2(15)>\frac{15\times6.908}{4}\right)=P(\chi^2(15)>25.905).$$

查 $\chi^2(15)$ 表，$\chi^2_{0.05}(15)\approx24.996$，$\chi^2_{0.025}(15)\approx27.488$，故 $P\approx0.025$—$0.05$ 之间；若题目给出 $25.905$ 对应 $P=0.05$，则

$$P(S^2>6.908)\approx\boxed{0.05}.$$

**(c) 方差比分布**

两独立样本 $n_1=n_2=10$，总体方差相同：

$$F=\frac{S_1^2}{S_2^2}\sim F(n_1-1,n_2-1)=\boxed{F(9,9)}.$$

---

### D.5.9（Ch.14，描述统计：箱线图与异常值）

**题目**：数据 $\{2,3,5,7,8,9,11,14,16,22\}$（$n=10$），计算 $Q_1,Q_2,Q_3$，IQR，异常值界限，识别异常值。

**思路**：$n=10$ 时用插值法（或直接取位置）计算四分位数；Tukey 1.5×IQR 准则判断异常值。

**解**：

数据已排序：$2,3,5,7,8,9,11,14,16,22$。

**(a) 四分位数**（采用"四分位数位置法"：$Q_1$ 在第 $(n+1)/4=2.75$ 位，$Q_3$ 在第 $3(n+1)/4=8.25$ 位）

- $Q_2$（中位数）：第 5、6 个数的均值 $=(8+9)/2=8.5$。
- $Q_1$：第 2.75 个位置 $=x_{(2)}+0.75(x_{(3)}-x_{(2)})=3+0.75\times2=4.5$。
- $Q_3$：第 8.25 个位置 $=x_{(8)}+0.25(x_{(9)}-x_{(8)})=14+0.25\times2=14.5$。

**(b) IQR 和界限**

$$\text{IQR}=Q_3-Q_1=14.5-4.5=10.$$

$$\text{下界}=Q_1-1.5\times\text{IQR}=4.5-15=-10.5,\quad\text{上界}=Q_3+1.5\times\text{IQR}=14.5+15=29.5.$$

**(c) 异常值**

所有数据均在 $(-10.5, 29.5)$ 内，故按 Tukey 准则**无异常值**。

中心度量：均值 $\bar x=(2+3+5+7+8+9+11+14+16+22)/10=97/10=9.7$；中位数 $Q_2=8.5$。

均值受较大值（16,22）影响偏高；中位数更鲁棒，不受极端值影响。

**答案**：$Q_1=4.5$，$Q_2=8.5$，$Q_3=14.5$；IQR$=10$；界限 $(-10.5,29.5)$；$\boxed{\text{无异常值}}$；均值 $9.7>$ 中位数 $8.5$，中位数更鲁棒。

---

### D.5.10（Ch.15，Rao-Blackwell 定理）

**题目**：陈述 Rao-Blackwell 定理；$X_i\sim\mathrm{Bernoulli}(p)$，$W=X_1$，$T=\sum X_i$，计算 $W^*=E[X_1\mid T=t]$。

**思路**：条件期望在充分统计量上取均值会"改善"估计量；给定 $T=t$ 时 $X_1$ 服从超几何退化形式。

**解**：

**(a) Rao-Blackwell 定理**

设 $T$ 是 $\theta$ 的充分统计量，$W$ 是 $\theta$ 的无偏估计量（$E_\theta[W]=\theta$）。令

$$W^* = E_\theta[W\mid T].$$

则：
1. $W^*$ 也是无偏的：$E_\theta[W^*]=E_\theta[E[W\mid T]]=E_\theta[W]=\theta$。
2. 方差（MSE）不增大：$\mathrm{Var}(W^*)\leq\mathrm{Var}(W)$（由条件方差公式 $\mathrm{Var}(W)=E[\mathrm{Var}(W\mid T)]+\mathrm{Var}(E[W\mid T])$）。

**(b) 计算 $W^*=E[X_1\mid T=t]$**

给定 $T=\sum_{i=1}^n X_i=t$，由对称性每个 $X_i$ 在条件下具有相同分布：

$$E[X_1\mid T=t]=\frac{E\!\left[\sum_{i=1}^nX_i\;\Big|\;T=t\right]}{n}=\frac{t}{n}.$$

严格推导：$P(X_1=1\mid T=t)=\dfrac{P(X_1=1,\,\sum_{i=2}^nX_i=t-1)}{P(T=t)}=\dfrac{p\binom{n-1}{t-1}p^{t-1}(1-p)^{n-t}}{\binom{n}{t}p^t(1-p)^{n-t}}=\dfrac{t}{n}.$

**(c) $W^*=t/n=\bar X$ 是 UMVUE**

$T=\sum X_i$ 是 Bernoulli($p$) 的完备充分统计量（指数族），$W^*=T/n=\bar X$ 是 $p$ 的无偏估计，由 Lehmann-Scheffé 定理，完备充分统计量的无偏函数是 UMVUE，故 $\bar X$ 是 $p$ 的 UMVUE。

**答案**：$W^*=E[X_1\mid T=t]=t/n$；$\boxed{\bar X=T/n}$ 是 $p$ 的 UMVUE。

---

## E 提高题详解（5 题）

### E.5.1（Ch.13+Ch.15，完备充分统计量 + Lehmann-Scheffé + UMVUE）

**题目**：$X_i\overset{iid}{\sim}\mathrm{Poisson}(\lambda)$，(a) $T=\sum X_i$ 充分；(b) $T$ 完备；(c) $e^{-\lambda}$ 的 UMVUE；(d) 推广。

**思路**：充分性用因子分解，完备性用指数族参数空间，UMVUE 由 Lehmann-Scheffé 定理给出。

**解**：

**(a) $T=\sum X_i$ 是充分统计量**

联合 PMF：

$$\prod_{i=1}^n\frac{e^{-\lambda}\lambda^{x_i}}{x_i!} = \underbrace{e^{-n\lambda}\lambda^{T}}_{g(T,\lambda)}\cdot\underbrace{\bigl(\textstyle\prod_{i=1}^nx_i!\bigr)^{-1}}_{h(\mathbf{x})}.$$

由 Neyman-Fisher 因子分解定理，$T=\sum X_i$ 是 $\lambda$ 的充分统计量。

**(b) $T$ 完备**

$T\sim\mathrm{Poisson}(n\lambda)$，其 PMF 为 $P(T=t)=e^{-n\lambda}(n\lambda)^t/t!$。

设 $E_\lambda[g(T)]=0$ 对所有 $\lambda>0$ 成立，即

$$\sum_{t=0}^\infty g(t)\frac{(n\lambda)^t}{t!}=0 \quad\forall\lambda>0.$$

令 $u=n\lambda>0$，上式变为 $\sum_{t=0}^\infty \frac{g(t)}{t!}u^t=0$ 对所有 $u>0$ 成立。幂级数系数恒为零，故 $g(t)=0$ 对所有 $t\geq0$ 成立，即 $g\equiv0$ a.e.。$T$ 完备。

**(c) $e^{-\lambda}$ 的 UMVUE**

由 Lehmann-Scheffé 定理：完备充分统计量 $T$ 的无偏函数是 UMVUE。

设初始无偏估计 $W=\mathbf{1}[X_1=0]$，则 $E_\lambda[W]=P(X_1=0)=e^{-\lambda}$，故 $W$ 是 $e^{-\lambda}$ 的无偏估计。

构造 Rao-Blackwell 改进：

$$\varphi(T) = E[W\mid T=t] = P(X_1=0\mid T=t).$$

给定 $T=t$，$X_1$ 在条件下的分布：

$$P(X_1=0\mid T=t)=\frac{P(X_1=0)P\bigl(\sum_{i=2}^nX_i=t\bigr)}{P(T=t)}=\frac{e^{-\lambda}\cdot e^{-(n-1)\lambda}\frac{[(n-1)\lambda]^t}{t!}}{e^{-n\lambda}\frac{(n\lambda)^t}{t!}}=\left(\frac{n-1}{n}\right)^t.$$

故 $e^{-\lambda}$ 的 UMVUE 为

$$\boxed{\varphi(T)=\left(1-\frac{1}{n}\right)^T=\left(1-\frac{1}{n}\right)^{\sum_{i=1}^nX_i}}.$$

**(d) 推广：$P(X=k)=\lambda^ke^{-\lambda}/k!$ 的 UMVUE**

取初始无偏估计 $W_k=\mathbf{1}[X_1=k]$，条件化：

$$\varphi_k(t)=P(X_1=k\mid T=t)=\frac{P(X_1=k)P\bigl(\sum_{i=2}^nX_i=t-k\bigr)}{P(T=t)}.$$

类似计算给出

$$\varphi_k(t) = \binom{t}{k}\left(\frac{1}{n}\right)^k\left(1-\frac{1}{n}\right)^{t-k},\quad t\geq k.$$

这是 $P(X=k)=\lambda^ke^{-\lambda}/k!$ 的 UMVUE。**完备性保证唯一性**：若存在两个完备充分统计量的无偏函数，其差的期望为 0，由完备性差为 0 a.s.，故 UMVUE 唯一。

**答案**：$e^{-\lambda}$ 的 UMVUE 为 $\boxed{\varphi(T)=(1-1/n)^T}$；$P(X=k)$ 的 UMVUE 为 $\binom{T}{k}(1/n)^k(1-1/n)^{T-k}$。

---

### E.5.2（Ch.14+Ch.13，经验分布函数 + Glivenko-Cantelli + 非参数估计）

**题目**：证明 $\hat F_n$ 的矩；证明 Glivenko-Cantelli；推导 KDE 最优带宽；与 GAN 联系。

**思路**：期望方差由 Bernoulli 分布直接得；G-C 定理先处理有限个点再用单调性延拓；KDE 偏差-方差权衡给出 $h^*\sim n^{-1/5}$。

**解**：

**(a) $\hat F_n$ 的期望与方差**

固定 $x$，$I_i=\mathbf{1}[X_i\leq x]\overset{iid}{\sim}\mathrm{Bernoulli}(F(x))$，$\hat F_n(x)=\bar I$。

$$E[\hat F_n(x)]=E[I_1]=F(x),$$
$$\mathrm{Var}(\hat F_n(x))=\frac{\mathrm{Var}(I_1)}{n}=\frac{F(x)(1-F(x))}{n}.$$

**(b) Glivenko-Cantelli 定理证明思路**

**步骤 1**（有限点）：对任意固定有限集 $\{x_1,\ldots,x_m\}$，由 SLLN 和有限并集，

$$\max_{j=1}^m|\hat F_n(x_j)-F(x_j)|\xrightarrow{a.s.}0.$$

**步骤 2**（全局延拓）：固定 $\varepsilon>0$，选取有限分割点 $-\infty=t_0<t_1<\cdots<t_m=\infty$ 使得 $F(t_j)-F(t_{j-1})<\varepsilon/2$ 对所有 $j$。对任意 $x\in[t_{j-1},t_j)$：

$$\hat F_n(x)\leq\hat F_n(t_j^-);\quad F(x)\geq F(t_{j-1}).$$

利用 $\hat F_n$ 和 $F$ 的单调性及分割点的一致收敛，可以证明

$$\sup_x|\hat F_n(x)-F(x)|<\varepsilon\quad\text{eventually a.s.}$$

由 $\varepsilon$ 的任意性，$\sup_x|\hat F_n(x)-F(x)|\xrightarrow{a.s.}0$。

**(c) KDE 最优带宽 $h^*\sim n^{-1/5}$**

核密度估计 $\hat f_h(x)=\frac{1}{nh}\sum_{i=1}^nK\!\left(\frac{x-X_i}{h}\right)$，其偏差与方差为（设 $K$ 为对称核，$\int u^2K(u)du=\kappa_2$）：

$$\mathrm{Bias}[\hat f_h(x)]\approx\frac{h^2}{2}\kappa_2 f''(x),\quad \mathrm{Var}[\hat f_h(x)]\approx\frac{f(x)\|K\|_2^2}{nh}.$$

积分均方误差（MISE）：

$$\mathrm{MISE}(h)\approx\frac{h^4}{4}\kappa_2^2\int[f''(x)]^2dx + \frac{\|K\|_2^2}{nh}.$$

对 $h$ 求导并令其为 0：

$$h^4\kappa_2^2 R(f'')\cdot h^{-1}\cdot\frac{d}{dh}(h^4)=\frac{\|K\|_2^2}{nh^2}\cdot h\;\Longrightarrow\; h^* = \left(\frac{\|K\|_2^2}{n\kappa_2^2 R(f'')}\right)^{1/5}\sim n^{-1/5},$$

其中 $R(f'')=\int[f''(x)]^2dx$。偏差 $\sim h^2\sim n^{-2/5}$，方差 $\sim(nh)^{-1}\sim n^{-4/5}$，MISE $\sim n^{-4/5}$。

**(d) KDE vs GAN（高维分析）**

- **KDE 在低维稳定**：带宽选择有明确准则（$h^*\sim n^{-1/5}$），估计量有理论保证，训练前期可解释性强。
- **维数诅咒**：KDE 的 MISE 在 $d$ 维下变为 $O(n^{-4/(4+d)})$，当 $d$ 大时收敛极慢（如 $d=100$ 时几乎无效）。
- **GAN 的优势**：生成器直接学习低维流形上的分布，隐式地利用数据的低维结构，规避了高维密度估计的维数诅咒；但训练不稳定，可能模式崩塌。
- **理论联系**：GAN 的判别器在最优时估计 JS 散度（或 Wasserstein 距离），这等价于非参数密度比估计，Glivenko-Cantelli 保证经验分布收敛，但高维下速度过慢，GAN 用神经网络参数化规避了这一问题。

**答案**：最优带宽 $\boxed{h^*\sim n^{-1/5}}$；MISE $\sim n^{-4/5}$；高维时 KDE 受维数诅咒，GAN 通过学习低维流形规避该问题。

---

### E.5.3（Ch.15+Ch.16，充分统计量 + Fisher 信息 + Cramér-Rao 下界）

**题目**：$X_i\overset{iid}{\sim}N(\mu,\sigma^2)$（$\sigma^2$ 已知），Fisher 信息、C-R 下界、Delta 方法、自然梯度。

**思路**：单个观测的 Fisher 信息乘以 $n$；验证 $\bar X$ 方差达到下界；Delta 方法处理参数变换；Fisher 矩阵与参数空间曲率联系自然梯度。

**解**：

**(a) Fisher 信息与 C-R 下界**

单个观测的对数似然：$\ell(\mu;x)=-\frac{(x-\mu)^2}{2\sigma^2}+\text{const}$，得分函数 $s(\mu;x)=\frac{x-\mu}{\sigma^2}$。

$$\mathcal{I}_1(\mu)=E_\mu[s^2]=E_\mu\!\left[\frac{(X-\mu)^2}{\sigma^4}\right]=\frac{\sigma^2}{\sigma^4}=\frac{1}{\sigma^2}.$$

$n$ 个独立观测：$\mathcal{I}_n(\mu)=n/\sigma^2$。Cramér-Rao 下界：

$$\mathrm{Var}_\mu(\hat\mu)\geq\frac{1}{\mathcal{I}_n(\mu)}=\frac{\sigma^2}{n}.$$

**(b) $\bar X$ 达到 C-R 下界**

$$\mathrm{Var}(\bar X)=\frac{\sigma^2}{n}=\frac{1}{\mathcal{I}_n(\mu)}.$$

等号成立，$\bar X$ 是有效估计量（efficient estimator）。事实上，$\bar X$ 是充分统计量且是指数族的自然参数的线性函数，因此自动达到 C-R 下界。

**(c) $\phi=e^\mu$ 的 Delta 方法与 C-R 下界**

$\hat\phi=e^{\bar X}$，$g(\mu)=e^\mu$，$g'(\mu)=e^\mu$。Delta 方法给出渐近分布：

$$\sqrt{n}(\hat\phi-\phi)\xrightarrow{d}N\!\left(0,[g'(\mu)]^2\cdot\sigma^2\right)=N\!\left(0,e^{2\mu}\sigma^2\right).$$

即 $\hat\phi\approx N\!\left(e^\mu,\,\frac{e^{2\mu}\sigma^2}{n}\right)$（渐近）。

$\phi=g(\mu)$ 的 C-R 下界：

$$\mathrm{Var}(\hat\phi)\geq\frac{[g'(\mu)]^2}{\mathcal{I}_n(\mu)}=\frac{e^{2\mu}\sigma^2}{n}.$$

$\hat\phi=e^{\bar X}$ 渐近达到此下界，是 $e^\mu$ 的渐近有效估计量。

**(d) Fisher 信息矩阵与自然梯度**

在深度学习中，参数 $\boldsymbol\theta\in\mathbb{R}^d$ 的 Fisher 信息矩阵 $\mathcal{I}(\boldsymbol\theta)_{jk}=E\!\left[\frac{\partial\log p}{\partial\theta_j}\frac{\partial\log p}{\partial\theta_k}\right]$ 度量参数空间的局部几何曲率（黎曼度量）。

- **标准梯度下降**：沿参数空间欧氏梯度方向更新，忽略参数化的曲率，在不同方向上学习率等效不同，收敛慢。
- **自然梯度下降**：沿 $\mathcal{I}^{-1}\nabla_\theta\mathcal{L}$ 方向更新（Fisher-Rao 度量下的最速下降），对参数重参数化不变，收敛快。
- **对角近似（Adagrad 类）**：$\mathcal{I}\approx\mathrm{diag}(I_{11},\ldots,I_{dd})$，计算 $O(d)$，忽略参数间相关性。
- **完整矩阵（K-FAC）**：Kronecker 分解近似 $\mathcal{I}$，计算 $O(d^{3/2})$，捕捉层间相关性，通常效果更好但开销更大。

**答案**：$\mathcal{I}_n(\mu)=n/\sigma^2$；C-R 下界 $\sigma^2/n$；$\bar X$ 有效；$\hat\phi=e^{\bar X}$ 渐近 $N(e^\mu,e^{2\mu}\sigma^2/n)$；Fisher 矩阵为黎曼度量，自然梯度对参数化不变。$\boxed{\mathrm{Var}(\bar X)=\sigma^2/n=1/\mathcal{I}_n(\mu)}$

---

### E.5.4（Ch.13+Ch.15，指数族 + 最大熵原理 + 对偶性）

**题目**：变分法证明最大熵分布为指数族；均值方差约束 $\Rightarrow$ 正态；$[0,1]$ 均值约束 $\Rightarrow$ Beta；温度采样。

**思路**：用 Lagrange 乘数法处理约束优化；正态是均值+方差约束下最大熵分布；Beta 是有界支撑均值约束下最大熵分布（特例为均匀）。

**解**：

**(a) 最大熵分布为指数族（变分推导）**

最大化 $H(p)=-\int p(x)\log p(x)dx$，约束条件：

$$\int p(x)dx=1,\quad \int p(x)T_k(x)dx=\mu_k,\;k=1,\ldots,m.$$

引入 Lagrange 乘子 $\eta_0-1$（归一化）和 $\eta_k$（$k=1,\ldots,m$），Lagrangian 为

$$\mathcal{L}=\int\!\left[-p\log p + (\eta_0-1)p + \sum_k\eta_kT_k p\right]dx.$$

对 $p(x)$ 变分，令 $\delta\mathcal{L}/\delta p=0$：

$$-\log p(x)-1+(\eta_0-1)+\sum_k\eta_kT_k(x)=0$$
$$\Rightarrow\quad p^*(x)=\exp\!\left(\sum_k\eta_kT_k(x)+(\eta_0-2)\right)=h(x)\exp\!\left(\sum_k\eta_kT_k(x)-A(\boldsymbol\eta)\right),$$

其中 $A(\boldsymbol\eta)=\log\int\exp(\sum_k\eta_kT_k)dx$ 是对数配分函数，$h(x)=1$（或其他基测度）。这正是指数族形式。

**(b) 均值+方差约束 $\Rightarrow$ 正态分布**

约束：$E[X]=\mu$，$E[X^2]=\mu^2+\sigma^2$（即方差 $=\sigma^2$），充分统计量 $T_1=x$，$T_2=x^2$。

由 (a)，最大熵分布为

$$p^*(x)\propto\exp(\eta_1 x+\eta_2 x^2).$$

由约束解出乘子：$\eta_2=-1/(2\sigma^2)<0$（使分布可归一化），$\eta_1=\mu/\sigma^2$，得

$$p^*(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)=N(\mu,\sigma^2).$$

**(c) $[0,1]$ 均值约束 $\Rightarrow$ Beta 分布**

支撑 $[0,1]$，约束 $E[X]=\mu$，基测度 $h(x)=1$，充分统计量 $T_1=\log x$，$T_2=\log(1-x)$（由于 $[0,1]$ 的指数族结构）。

最大熵分布为

$$p^*(x)\propto\exp(\eta_1\log x+\eta_2\log(1-x))=x^{\eta_1}(1-x)^{\eta_2},$$

即 $\mathrm{Beta}(\alpha,\beta)$（$\alpha=\eta_1+1$，$\beta=\eta_2+1$）。当 $\mu=0.5$ 时对称，$\alpha=\beta$，最大熵为均匀分布 $\mathrm{Beta}(1,1)=U(0,1)$。

> ⚠️ 严格地，$[0,1]$ 上仅约束均值 $E[X]=\mu$ 不直接给出 Beta 分布；若额外约束 $E[\log X]$ 和 $E[\log(1-X)]$，则最大熵分布才是 Beta 分布。均匀分布是 $\mu=0.5$、无其他矩约束时的最大熵分布。

**(d) 温度采样与最大熵**

温度采样：$p_T(x)\propto p(x)^{1/T}$，对数形式 $\log p_T(x)=\frac{1}{T}\log p(x)-\log Z_T$。

- $T\to0$（贪心解码）：$p_T$ 集中于 $\arg\max p(x)$，熵 $H(p_T)\to0$；
- $T=1$（标准采样）：$p_T=p$，原始分布；
- $T\to\infty$（均匀采样）：$p_T\to\mathrm{Uniform}$，熵最大化；
- **最大熵联系**：在约束 $E_T[\log p(x)]=c_T$（给定平均对数概率）下，最大熵分布恰为 $p_T(x)\propto p(x)^{1/T}$，即温度采样是约束最大熵问题的解，$T$ 为对偶变量（Lagrange 乘子）。

**答案**：最大熵分布为指数族 $\boxed{p^*\propto\exp(\sum_k\eta_kT_k(x))}$；均值+方差约束 $\Rightarrow N(\mu,\sigma^2)$；$T$ 为最大熵的对偶变量，$T\to0$ 贪心，$T\to\infty$ 均匀。

---

### E.5.5（Ch.13+Ch.14+Ch.15，次序统计量 + 分位数估计 + 鲁棒统计）

**题目**：分位数渐近正态；中位数 ARE；影响函数；联邦学习鲁棒性。

**思路**：分位数的渐近正态性通过顺序统计量密度的极限论证；ARE 比较两个渐近方差；影响函数度量单点污染的一阶效应；breakdown point 量化估计量的全局鲁棒性。

**解**：

**(a) 样本 $p$-分位数的渐近正态性**

令 $\hat\xi_p=X_{(\lceil np\rceil)}$，$\xi_p=F^{-1}(p)$。

核心思路：$\{\hat\xi_p\leq x\}=\{X_{(\lceil np\rceil)}\leq x\}=\{\text{至少}\lceil np\rceil\text{个}X_i\leq x\}=\left\{\hat F_n(x)\geq p\right\}$，故

$$P(\hat\xi_p\leq\xi_p+t/\sqrt{n})=P\!\left(\hat F_n(\xi_p+t/\sqrt{n})\geq p\right).$$

在 $x=\xi_p$ 附近展开：$F(\xi_p+t/\sqrt{n})\approx p+tf(\xi_p)/\sqrt{n}$，且 $n\hat F_n(x)\sim B(n,F(x))$，由 CLT：

$$\sqrt{n}\!\left(\hat F_n(x)-F(x)\right)\xrightarrow{d}N\!\left(0,F(x)(1-F(x))\right).$$

结合连续映射定理（对分位数函数）：

$$\sqrt{n}(\hat\xi_p-\xi_p)\xrightarrow{d}N\!\left(0,\frac{p(1-p)}{[f(\xi_p)]^2}\right).$$

**(b) 中位数的渐近方差与 ARE**

$p=0.5$，$\xi_{0.5}=m$（真实中位数），渐近方差 $=\dfrac{0.25}{[f(m)]^2\cdot n}$。

对 $F=N(\mu,\sigma^2)$：$f(m)=f(\mu)=\dfrac{1}{\sqrt{2\pi}\sigma}$，故

$$\mathrm{AsyVar}(\hat m)=\frac{0.25}{[1/(\sqrt{2\pi}\sigma)]^2\cdot n}=\frac{\pi\sigma^2}{2n}.$$

样本均值渐近方差 $=\sigma^2/n$。渐近相对效率：

$$\mathrm{ARE}(\hat m,\bar X)=\frac{\sigma^2/n}{\pi\sigma^2/(2n)}=\frac{2}{\pi}\approx0.637.$$

含义：在正态分布下，用中位数估计均值需要约 $\pi/2\approx1.57$ 倍的样本量才能达到与均值相同的精度。

**(c) 影响函数**

$$\mathrm{IF}(x;T,F)=\lim_{\varepsilon\to0}\frac{T\bigl((1-\varepsilon)F+\varepsilon\delta_x\bigr)-T(F)}{\varepsilon}.$$

- **均值** $T(F)=\int y\,dF(y)$：$T((1-\varepsilon)F+\varepsilon\delta_x)=(1-\varepsilon)\mu+\varepsilon x$，故 $\mathrm{IF}(x;\text{mean},F)=x-\mu$（无界，不鲁棒）。

- **中位数** $T(F)=F^{-1}(0.5)$：扰动后中位数满足 $F_\varepsilon(m_\varepsilon)=0.5$，对 $\varepsilon$ 求导，利用隐函数定理：

  $$\mathrm{IF}(x;\text{median},F)=\frac{\mathrm{sgn}(x-m)}{2f(m)}$$

  有界（$|\mathrm{IF}|\leq 1/(2f(m))$），鲁棒。

- **M 估计量** $T$ 满足 $\int\psi(y-T)\,dF(y)=0$：$\mathrm{IF}(x;T,F)=\dfrac{\psi(x-T)}{\int\psi'(y-T)\,dF(y)}$；当 $\psi$ 有界（如 Huber）时 IF 有界，具有鲁棒性。

**(d) 联邦学习与 Byzantine 鲁棒性**

在联邦学习中，$n$ 个客户端提交本地梯度 $g_1,\ldots,g_n$，部分（设 $\epsilon$ 比例）被 Byzantine 攻击者控制。

- **FedAvg（均值聚合）**：$\bar g=\frac{1}{n}\sum g_i$，由于均值的影响函数 $\mathrm{IF}(x)=x-\mu$ 无界，攻击者可以将 $g_i$ 设为任意大值，无限偏移聚合结果，**breakdown point $=0$**（任意一个恶意客户端即可破坏）。

- **坐标中位数（Coordinate-wise Median）**：对每个坐标独立取中位数，其影响函数有界，单个恶意梯度的影响不超过 $1/(2f(m))$。**Breakdown point $=50\%$**：只要恶意客户端比例 $<50\%$，聚合结果有界偏离，攻击者无法任意操纵全局更新。

用影响函数理论量化：设攻击者控制 $\epsilon n$ 个客户端，发送恶意梯度 $g_{\text{attack}}$，则坐标中位数的偏移量为

$$\Delta T\approx\epsilon\cdot\mathrm{IF}(g_{\text{attack}};\text{median},F)\leq\frac{\epsilon}{2f(m)}<\infty,$$

而均值聚合偏移量 $\Delta\bar g\propto\epsilon\cdot\|g_{\text{attack}}\|$ 随攻击强度无界增长。

**答案**：分位数渐近分布 $\boxed{\sqrt{n}(\hat\xi_p-\xi_p)\xrightarrow{d}N\!\left(0,\frac{p(1-p)}{f(\xi_p)^2}\right)}$；ARE$(\hat m,\bar X)=2/\pi\approx0.637$；均值 IF 无界，中位数 IF 有界；中位数聚合 breakdown point $=50\%$，远优于 FedAvg（breakdown point $=0$）。
