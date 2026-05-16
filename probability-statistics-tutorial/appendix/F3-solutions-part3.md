# F3 详解：Part 3 常见分布（Ch.7-9，共 35 题）

## C 基础题详解（12 题）

### C.3.1（Ch.7，伯努利分布）

**题目**：$p=0.8$ 的单次射击，$X\sim\mathrm{Bernoulli}(0.8)$，求 PMF、$E[X]$、$\mathrm{Var}(X)$。

**思路**：伯努利分布是最简单的离散分布，直接套公式。

**解**：

1. **PMF**：

$$P(X=k) = \begin{cases} 0.2 & k=0 \\ 0.8 & k=1 \end{cases} \quad \text{即 } P(X=k)=p^k(1-p)^{1-k},\ k\in\{0,1\}$$

2. **期望与方差**：

$$E[X] = 0\cdot(1-p)+1\cdot p = p = 0.8$$

$$\mathrm{Var}(X) = E[X^2]-(E[X])^2 = p - p^2 = p(1-p) = 0.8\times0.2 = 0.16$$

**答案**：$\boxed{E[X]=0.8,\quad \mathrm{Var}(X)=0.16}$

---

### C.3.2（Ch.7，二项分布）

**题目**：均匀硬币抛 8 次，$X\sim B(8,0.5)$，求 PMF、$P(X=4)$、$E[X]$、$\mathrm{Var}(X)$。

**思路**：二项分布 $B(n,p)$ 的 PMF 为组合数乘以概率，直接代入。

**解**：

1. **PMF**：

$$P(X=k) = \binom{8}{k}\left(\frac{1}{2}\right)^k\left(\frac{1}{2}\right)^{8-k} = \binom{8}{k}\frac{1}{256},\quad k=0,1,\ldots,8$$

2. **$P(X=4)$**：

$$P(X=4) = \binom{8}{4}\frac{1}{256} = \frac{70}{256} = \frac{35}{128} \approx 0.2734$$

3. **期望与方差**：

$$E[X] = np = 8\times0.5 = 4$$

$$\mathrm{Var}(X) = np(1-p) = 8\times0.5\times0.5 = 2$$

**答案**：$\boxed{P(X=4)=\dfrac{35}{128},\quad E[X]=4,\quad \mathrm{Var}(X)=2}$

---

### C.3.3（Ch.7，泊松分布）

**题目**：$\lambda=3$ 的泊松呼叫流，求 PMF、$P(X=0)$、$E[X]$、$\mathrm{Var}(X)$。

**思路**：泊松分布 $\mathrm{Poisson}(\lambda)$ 的均值方差均为 $\lambda$。

**解**：

1. **PMF**：

$$P(X=k) = \frac{e^{-\lambda}\lambda^k}{k!} = \frac{e^{-3}\cdot 3^k}{k!},\quad k=0,1,2,\ldots$$

2. **$P(X=0)$**：

$$P(X=0) = e^{-3} \approx 0.0498$$

3. **期望与方差**：

$$E[X] = \lambda = 3,\qquad \mathrm{Var}(X) = \lambda = 3$$

**答案**：$\boxed{P(X=0)=e^{-3},\quad E[X]=\mathrm{Var}(X)=3}$

> ⚠️ 泊松分布的均值等于方差，这是其与二项分布的重要区别。

---

### C.3.4（Ch.7，几何分布）

**题目**：每次成功概率 $p=0.4$，$X$ 为首次成功所需试验次数，求 PMF、$P(X=3)$、$E[X]$。

**思路**：几何分布描述"第一次成功前需多少次试验"，前 $k-1$ 次失败、第 $k$ 次成功。

**解**：

1. **PMF**（首次成功型，支撑 $k=1,2,\ldots$）：

$$P(X=k) = (1-p)^{k-1}p = (0.6)^{k-1}\times0.4$$

2. **$P(X=3)$**：

$$P(X=3) = (0.6)^2\times0.4 = 0.36\times0.4 = 0.144$$

3. **期望**：

$$E[X] = \frac{1}{p} = \frac{1}{0.4} = 2.5$$

**答案**：$\boxed{P(X=3)=0.144,\quad E[X]=2.5}$

---

### C.3.5（Ch.7，泊松近似二项）

**题目**：次品率 $p=0.01$，抽取 $n=200$ 件，用泊松近似求 $\lambda$ 及 $P(X=0)$。

**思路**：$n$ 大、$p$ 小时 $B(n,p)\approx\mathrm{Poisson}(\lambda)$，$\lambda=np$。

**解**：

1. **近似参数**：

$$\lambda = np = 200\times0.01 = 2$$

2. **泊松近似 $P(X=0)$**：

$$P(X=0) \approx e^{-\lambda} = e^{-2} \approx 0.1353$$

（精确值：$(0.99)^{200}\approx0.1340$，误差约 0.1%。）

**答案**：$\boxed{\lambda=2,\quad P(X=0)\approx e^{-2}\approx0.1353}$

---

### C.3.6（Ch.8，均匀分布）

**题目**：$X\sim U(2,8)$，求 PDF、CDF、$P(3\le X\le6)$、$E[X]$、$\mathrm{Var}(X)$。

**思路**：均匀分布在区间上等概率，直接用公式 $b-a=6$。

**解**：

1. **PDF 与 CDF**：

$$f(x) = \frac{1}{b-a} = \frac{1}{6},\quad 2\le x\le8$$

$$F(x) = \begin{cases} 0 & x<2 \\ \dfrac{x-2}{6} & 2\le x\le8 \\ 1 & x>8 \end{cases}$$

2. **$P(3\le X\le6)$**：

$$P(3\le X\le6) = \frac{6-3}{6} = \frac{3}{6} = \frac{1}{2}$$

3. **期望与方差**：

$$E[X] = \frac{a+b}{2} = \frac{2+8}{2} = 5$$

$$\mathrm{Var}(X) = \frac{(b-a)^2}{12} = \frac{36}{12} = 3$$

**答案**：$\boxed{P(3\le X\le6)=\dfrac{1}{2},\quad E[X]=5,\quad \mathrm{Var}(X)=3}$

---

### C.3.7（Ch.8，指数分布与无记忆性）

**题目**：$T\sim\mathrm{Exp}(0.5)$，求 PDF、CDF、$P(T>2)$；利用无记忆性求 $P(T>4\mid T>2)$。

**思路**：指数分布的无记忆性：$P(T>s+t\mid T>s)=P(T>t)$，条件概率等同于重新开始。

**解**：

1. **PDF 与 CDF**（$\lambda=0.5$）：

$$f(t) = \lambda e^{-\lambda t} = 0.5\,e^{-0.5t},\quad t>0$$

$$F(t) = 1 - e^{-\lambda t} = 1 - e^{-0.5t},\quad t>0$$

2. **$P(T>2)$**：

$$P(T>2) = e^{-\lambda\cdot2} = e^{-1} \approx 0.3679$$

3. **无记忆性应用**：

$$P(T>4\mid T>2) = P(T>2) = e^{-1} \approx 0.3679$$

原因：已知 $T>2$，剩余寿命仍服从同参数指数分布，再运行 2 小时以上概率仍为 $e^{-1}$。

**答案**：$\boxed{P(T>2)=e^{-1}\approx0.3679,\quad P(T>4\mid T>2)=e^{-1}}$

> ⚠️ 无记忆性是指数分布（连续情形）和几何分布（离散情形）独有的性质。

---

### C.3.8（Ch.8，正态分布性质）

**题目**：$X\sim N(5,9)$（$\sigma=3$），求 $P(2\le X\le8)$；若 $Y=2X+1$，求 $Y$ 的分布。

**思路**：正态分布标准化后查 $\Phi$ 表；线性变换后仍为正态。

**解**：

1. **$P(2\le X\le8)$**：

标准化：$Z=\dfrac{X-5}{3}\sim N(0,1)$，

$$P(2\le X\le8) = P\!\left(\frac{2-5}{3}\le Z\le\frac{8-5}{3}\right) = P(-1\le Z\le1)$$

$$= \Phi(1)-\Phi(-1) = 2\Phi(1)-1 \approx 2\times0.8413-1 = 0.6826$$

2. **$Y=2X+1$ 的分布**：

$$E[Y] = 2E[X]+1 = 2\times5+1 = 11$$

$$\mathrm{Var}(Y) = 4\,\mathrm{Var}(X) = 4\times9 = 36$$

$$Y \sim N(11,\,36)$$

**答案**：$\boxed{P(2\le X\le8)=2\Phi(1)-1\approx0.6826,\quad Y\sim N(11,36)}$

---

### C.3.9（Ch.8，Gamma 分布）

**题目**：$X\sim\mathrm{Gamma}(3,2)$（$\alpha=3,\beta=2$，速率参数化），求 $E[X]$、$\mathrm{Var}(X)$；三个 $\mathrm{Exp}(2)$ 之和的分布。

**思路**：Gamma 分布的可加性：同速率参数的独立 Gamma 相加，形状参数叠加。

**解**：

1. **期望与方差**（速率参数 $\beta$）：

$$E[X] = \frac{\alpha}{\beta} = \frac{3}{2} = 1.5$$

$$\mathrm{Var}(X) = \frac{\alpha}{\beta^2} = \frac{3}{4} = 0.75$$

2. **三个 $\mathrm{Exp}(2)$ 之和**：

$\mathrm{Exp}(2)=\mathrm{Gamma}(1,2)$，三个独立相加：

$$X_1+X_2+X_3 \sim \mathrm{Gamma}(1+1+1,\,2) = \mathrm{Gamma}(3,2)$$

**答案**：$\boxed{E[X]=1.5,\quad\mathrm{Var}(X)=0.75,\quad X_1+X_2+X_3\sim\mathrm{Gamma}(3,2)}$

---

### C.3.10（Ch.8，卡方分布的构造）

**题目**：$Z_1,\ldots,Z_5\overset{\text{i.i.d.}}{\sim}N(0,1)$，$W=\sum_{i=1}^5 Z_i^2$，求分布、期望、方差。

**思路**：$n$ 个独立标准正态的平方和服从自由度为 $n$ 的卡方分布。

**解**：

1. **$W$ 的分布**：

$$W = \sum_{i=1}^5 Z_i^2 \sim \chi^2(5)$$

（$\chi^2(n)$ 等价于 $\mathrm{Gamma}(n/2,\,1/2)$）

2. **期望与方差**：

$$E[W] = n = 5$$

$$\mathrm{Var}(W) = 2n = 10$$

**答案**：$\boxed{W\sim\chi^2(5),\quad E[W]=5,\quad\mathrm{Var}(W)=10}$

---

### C.3.11（Ch.9，多项分布）

**题目**：均匀六面骰子掷 12 次，$X_k$ 为点数 $k$ 出现次数，求联合分布参数及 $E[X_1]$、$\mathrm{Var}(X_1)$。

**思路**：多项分布是二项分布的推广；每个分量 $X_k\sim B(n,p_k)$。

**解**：

1. **联合分布参数**：

$$(X_1,\ldots,X_6)\sim\mathrm{Multinomial}\!\left(12;\,\frac{1}{6},\frac{1}{6},\frac{1}{6},\frac{1}{6},\frac{1}{6},\frac{1}{6}\right)$$

联合 PMF：

$$P(X_1=k_1,\ldots,X_6=k_6) = \frac{12!}{k_1!\cdots k_6!}\left(\frac{1}{6}\right)^{12},\quad \sum_{j=1}^6 k_j=12$$

2. **$E[X_1]$ 与 $\mathrm{Var}(X_1)$**：

$X_1\sim B(12,1/6)$，故

$$E[X_1] = np_1 = 12\times\frac{1}{6} = 2$$

$$\mathrm{Var}(X_1) = np_1(1-p_1) = 12\times\frac{1}{6}\times\frac{5}{6} = \frac{10}{6} = \frac{5}{3}\approx1.667$$

**答案**：$\boxed{E[X_1]=2,\quad\mathrm{Var}(X_1)=\dfrac{5}{3}}$

---

### C.3.12（Ch.9，多元正态的边缘分布）

**题目**：$(X,Y)\sim N(0,1,4,9,0.5)$，求 $X$、$Y$ 的边缘分布及 $\mathrm{Cov}(X,Y)$。

**思路**：二维正态的边缘仍为正态；协方差由相关系数和标准差给出。

**解**：

1. **$X$ 的边缘分布**：

$$X\sim N(\mu_X,\,\sigma_X^2) = N(0,\,4)$$

2. **$Y$ 的边缘分布**：

$$Y\sim N(\mu_Y,\,\sigma_Y^2) = N(1,\,9)$$

3. **协方差**：

$$\mathrm{Cov}(X,Y) = \rho\,\sigma_X\sigma_Y = 0.5\times2\times3 = 3$$

**答案**：$\boxed{X\sim N(0,4),\quad Y\sim N(1,9),\quad\mathrm{Cov}(X,Y)=3}$

---

## D 中等题详解（15 题）

### D.3.1（Ch.7，泊松分布的可加性）

**题目**：$X\sim\mathrm{Poisson}(\lambda_1)$，$Y\sim\mathrm{Poisson}(\lambda_2)$，$X\perp Y$。证明可加性；求条件分布 $P(X=k\mid X+Y=n)$。

**思路**：用概率母函数（PGF）证明可加性最简洁；条件分布通过贝叶斯公式化简为二项分布。

**解**：

**(a) 可加性证明（矩母函数法）**：

泊松分布 $\mathrm{Poisson}(\lambda)$ 的 MGF：

$$M_X(t) = E[e^{tX}] = e^{\lambda(e^t-1)}$$

由独立性：

$$M_{X+Y}(t) = M_X(t)\cdot M_Y(t) = e^{\lambda_1(e^t-1)}\cdot e^{\lambda_2(e^t-1)} = e^{(\lambda_1+\lambda_2)(e^t-1)}$$

此为 $\mathrm{Poisson}(\lambda_1+\lambda_2)$ 的 MGF，故

$$X+Y\sim\mathrm{Poisson}(\lambda_1+\lambda_2)\qquad\square$$

**(b) 条件分布 $P(X=k\mid X+Y=n)$**：

令 $S=X+Y\sim\mathrm{Poisson}(\lambda_1+\lambda_2)$，

$$P(X=k\mid S=n) = \frac{P(X=k,Y=n-k)}{P(S=n)}$$

分子：$\dfrac{e^{-\lambda_1}\lambda_1^k}{k!}\cdot\dfrac{e^{-\lambda_2}\lambda_2^{n-k}}{(n-k)!}$

分母：$\dfrac{e^{-(\lambda_1+\lambda_2)}(\lambda_1+\lambda_2)^n}{n!}$

相除：

$$P(X=k\mid S=n) = \binom{n}{k}\left(\frac{\lambda_1}{\lambda_1+\lambda_2}\right)^k\left(\frac{\lambda_2}{\lambda_1+\lambda_2}\right)^{n-k}$$

这正是 $B\!\left(n,\,\dfrac{\lambda_1}{\lambda_1+\lambda_2}\right)$ 的 PMF。

**(c) $\lambda_1=\lambda_2$ 时**：

$$p=\frac{\lambda_1}{\lambda_1+\lambda_2}=\frac{1}{2}$$，条件分布为 $B(n,1/2)$。

**答案**：$\boxed{X+Y\sim\mathrm{Poisson}(\lambda_1+\lambda_2);\quad (X\mid X+Y=n)\sim B\!\left(n,\dfrac{\lambda_1}{\lambda_1+\lambda_2}\right)}$

---

### D.3.2（Ch.7，几何分布的矩）

**题目**：$X\sim\mathrm{Geom}(p)$，证明 $E[X]=1/p$、$\mathrm{Var}(X)=(1-p)/p^2$；证明无记忆性；证明唯一性。

**思路**：用 MGF 求矩；无记忆性直接计算条件概率；唯一性用函数方程。

**解**：

**(a) 矩母函数求矩**：

令 $q=1-p$，

$$M_X(t) = \sum_{k=1}^\infty e^{tk}q^{k-1}p = \frac{pe^t}{1-qe^t},\quad t<-\ln q$$

$$M_X'(t) = \frac{pe^t}{(1-qe^t)^2}$$

$$E[X] = M_X'(0) = \frac{p}{(1-q)^2} = \frac{p}{p^2} = \frac{1}{p}$$

$$M_X''(t) = \frac{pe^t(1+qe^t)}{(1-qe^t)^3}$$

$$E[X^2] = M_X''(0) = \frac{p(1+q)}{p^3} = \frac{1+q}{p^2} = \frac{2-p}{p^2}$$

$$\mathrm{Var}(X) = E[X^2]-(E[X])^2 = \frac{2-p}{p^2}-\frac{1}{p^2} = \frac{1-p}{p^2} = \frac{q}{p^2}\qquad\square$$

**(b) 无记忆性证明**：

$$P(X>m) = \sum_{k=m+1}^\infty q^{k-1}p = q^m$$

$$P(X>m+n\mid X>m) = \frac{P(X>m+n)}{P(X>m)} = \frac{q^{m+n}}{q^m} = q^n = P(X>n)\qquad\square$$

**(c) 唯一性（无记忆性 $\Rightarrow$ 几何）**：

设 $X$ 取正整数值，$P(X>k)=g(k)$，无记忆性要求 $g(m+n)=g(m)g(n)$（函数方程）。

由 $g(0)=1$ 及正整数值，$g(k)=g(1)^k$。令 $q=g(1)\in(0,1)$，则

$$P(X=k)=P(X>k-1)-P(X>k)=q^{k-1}-q^k=q^{k-1}(1-q)$$

这正是参数 $p=1-q$ 的几何分布。$\square$

**答案**：$\boxed{E[X]=\dfrac{1}{p},\quad\mathrm{Var}(X)=\dfrac{1-p}{p^2}}$；几何分布是离散无记忆分布的唯一类型。

---

### D.3.3（Ch.8，正态分布线性组合）

**题目**：$X\sim N(\mu_1,\sigma_1^2)$，$Y\sim N(\mu_2,\sigma_2^2)$，$X\perp Y$，证明线性组合仍为正态；求 $Z=X-Y$ 的分布；当 $X,Y\sim N(0,1)$ 时求 $P(X>Y)$。

**思路**：用 MGF 证明线性组合的正态性；$P(X>Y)=P(X-Y>0)$，利用 $Z\sim N(0,2)$。

**解**：

**(a) 线性组合 $aX+bY+c$**：

MGF：

$$M_{aX+bY}(t) = M_X(at)\cdot M_Y(bt) = e^{a\mu_1 t+\frac{1}{2}a^2\sigma_1^2 t^2}\cdot e^{b\mu_2 t+\frac{1}{2}b^2\sigma_2^2 t^2}$$

$$= e^{(a\mu_1+b\mu_2)t+\frac{1}{2}(a^2\sigma_1^2+b^2\sigma_2^2)t^2}$$

这是均值 $a\mu_1+b\mu_2+c$、方差 $a^2\sigma_1^2+b^2\sigma_2^2$ 的正态分布 MGF，故

$$aX+bY+c\sim N(a\mu_1+b\mu_2+c,\;a^2\sigma_1^2+b^2\sigma_2^2)\qquad\square$$

**(b) $Z=X-Y$ 的分布**（取 $a=1,b=-1,c=0$）：

$$Z\sim N(\mu_1-\mu_2,\;\sigma_1^2+\sigma_2^2)$$

**(c) $P(X>Y)$（$\mu_1=\mu_2=0$，$\sigma_1^2=\sigma_2^2=1$）**：

$$Z=X-Y\sim N(0,2)$$

$$P(X>Y) = P(Z>0) = P\!\left(\frac{Z}{\sqrt{2}}>0\right) = 1-\Phi(0) = 0.5$$

**答案**：$\boxed{Z=X-Y\sim N(\mu_1-\mu_2,\,\sigma_1^2+\sigma_2^2);\quad P(X>Y)=0.5}$

> ⚠️ 对称性直接给出 $P(X>Y)=1/2$，无需查表；但不对称情形必须标准化。

---

### D.3.4（Ch.8，伽马分布的性质）

**题目**：$X\sim\mathrm{Gamma}(\alpha,\beta)$，密度 $f(x)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$，证明矩公式、退化为指数、可加性。

**思路**：利用 $\Gamma(\alpha)$ 递推公式 $\Gamma(\alpha+1)=\alpha\Gamma(\alpha)$ 计算矩；MGF 证明可加性。

**解**：

**(a) $E[X]$ 和 $\mathrm{Var}(X)$**：

$$E[X] = \int_0^\infty x\cdot\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}\,dx = \frac{\beta^\alpha}{\Gamma(\alpha)}\int_0^\infty x^\alpha e^{-\beta x}\,dx$$

令 $u=\beta x$，得 $\displaystyle\int_0^\infty x^\alpha e^{-\beta x}dx = \frac{\Gamma(\alpha+1)}{\beta^{\alpha+1}}$，故

$$E[X] = \frac{\beta^\alpha}{\Gamma(\alpha)}\cdot\frac{\Gamma(\alpha+1)}{\beta^{\alpha+1}} = \frac{\alpha\Gamma(\alpha)}{\Gamma(\alpha)\cdot\beta} = \frac{\alpha}{\beta}$$

类似地，

$$E[X^2] = \frac{\beta^\alpha}{\Gamma(\alpha)}\cdot\frac{\Gamma(\alpha+2)}{\beta^{\alpha+2}} = \frac{\alpha(\alpha+1)}{\beta^2}$$

$$\mathrm{Var}(X) = \frac{\alpha(\alpha+1)}{\beta^2}-\frac{\alpha^2}{\beta^2} = \frac{\alpha}{\beta^2}\qquad\square$$

**(b) $\alpha=1$ 时退化为指数分布**：

$$f(x) = \frac{\beta^1}{\Gamma(1)}x^{0}e^{-\beta x} = \beta e^{-\beta x}$$

这正是 $\mathrm{Exp}(\beta)$ 的密度。$\square$

**(c) 可加性（MGF法）**：

$\mathrm{Gamma}(\alpha,\beta)$ 的 MGF（$t<\beta$）：

$$M_X(t) = \left(\frac{\beta}{\beta-t}\right)^\alpha$$

设 $X_i\overset{\text{i.i.d.}}{\sim}\mathrm{Exp}(\beta)=\mathrm{Gamma}(1,\beta)$，则

$$M_{X_1+\cdots+X_n}(t) = \prod_{i=1}^n\left(\frac{\beta}{\beta-t}\right)^1 = \left(\frac{\beta}{\beta-t}\right)^n$$

此为 $\mathrm{Gamma}(n,\beta)$ 的 MGF，故 $X_1+\cdots+X_n\sim\mathrm{Gamma}(n,\beta)$。$\square$

**答案**：$\boxed{E[X]=\dfrac{\alpha}{\beta},\quad\mathrm{Var}(X)=\dfrac{\alpha}{\beta^2};\quad \sum_{i=1}^n X_i\sim\mathrm{Gamma}(n,\beta)}$

---

### D.3.5（Ch.7，超几何分布与二项近似）

**题目**：$N=100$，$K=20$ 红球，无放回取 $n=10$，求超几何 PMF、$E[X]$、$\mathrm{Var}(X)$；与二项近似比较。

**思路**：超几何分布用组合数公式；当 $n/N$ 小时二项近似良好（有限总体修正因子接近 1）。

**解**：

**(a) 超几何 PMF**：

$$P(X=k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}} = \frac{\binom{20}{k}\binom{80}{10-k}}{\binom{100}{10}},\quad k=0,1,\ldots,10$$

**(b) $E[X]$ 与 $\mathrm{Var}(X)$**：

$$E[X] = n\cdot\frac{K}{N} = 10\times\frac{20}{100} = 2$$

$$\mathrm{Var}(X) = n\cdot\frac{K}{N}\cdot\frac{N-K}{N}\cdot\frac{N-n}{N-1} = 10\times0.2\times0.8\times\frac{90}{99} = \frac{144}{99} \approx 1.455$$

有限总体修正因子（FPC）：$\dfrac{N-n}{N-1}=\dfrac{90}{99}\approx0.909$。

**(c) 二项近似 $B(10,0.2)$，$k=2$**：

**精确值（超几何）**：

$$P(X=2) = \frac{\binom{20}{2}\binom{80}{8}}{\binom{100}{10}} = \frac{190\times\binom{80}{8}}{\binom{100}{10}}$$

（数值约为 $0.3182$）

**近似值（二项）**：

$$P(X=2) = \binom{10}{2}(0.2)^2(0.8)^8 = 45\times0.04\times0.1678 \approx 0.3020$$

误差约 $\dfrac{0.3182-0.3020}{0.3182}\approx5\%$，因 $n/N=10\%$ 较小，近似合理。

**答案**：$\boxed{E[X]=2,\quad\mathrm{Var}(X)\approx1.455}$；二项近似误差约 5%。

---

### D.3.6（Ch.8，卡方分布与正态的联系）

**题目**：$Z_1,\ldots,Z_n\overset{\text{i.i.d.}}{\sim}N(0,1)$，$V=\sum Z_i^2$，用 MGF 证明 $V\sim\chi^2(n)$；求均值方差；陈述 $\bar{Z}$ 与 $\sum(Z_i-\bar{Z})^2$ 的独立性。

**思路**：$\chi^2(n)=\mathrm{Gamma}(n/2,1/2)$；独立性来自 Cochran 定理。

**解**：

**(a) MGF 证明 $V\sim\chi^2(n)$**：

单个 $Z^2$ 的 MGF（$t<1/2$）：

$$M_{Z^2}(t) = E[e^{tZ^2}] = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^\infty e^{tz^2}e^{-z^2/2}\,dz = \frac{1}{\sqrt{2\pi}}\int e^{-\frac{1-2t}{2}z^2}\,dz = (1-2t)^{-1/2}$$

由独立性：

$$M_V(t) = \prod_{i=1}^n M_{Z_i^2}(t) = (1-2t)^{-n/2}$$

这正是 $\mathrm{Gamma}(n/2,1/2)$ 的 MGF，即 $\chi^2(n)$。$\square$

**(b) 期望与方差**：

$\chi^2(n)=\mathrm{Gamma}(n/2,1/2)$，故

$$E[V] = \frac{n/2}{1/2} = n,\qquad\mathrm{Var}(V) = \frac{n/2}{(1/2)^2} = 2n$$

**(c) $\bar{Z}$ 与 $V'=\sum(Z_i-\bar{Z})^2$ 的独立性**：

依据 **Cochran 定理**：设 $Z_1,\ldots,Z_n\overset{\text{i.i.d.}}{\sim}N(0,1)$，$\bar{Z}$ 是样本均值，则 $\bar{Z}\sim N(0,1/n)$ 与 $V'=\sum(Z_i-\bar{Z})^2\sim\chi^2(n-1)$ 相互独立。

直觉：$\bar{Z}$ 是 $n$ 维向量 $(Z_1,\ldots,Z_n)$ 在方向 $(1/\sqrt{n},\ldots,1/\sqrt{n})$ 上的投影，$V'$ 依赖正交补空间，正态分布保证两者独立。

**答案**：$\boxed{V\sim\chi^2(n),\quad E[V]=n,\quad\mathrm{Var}(V)=2n}$；$\bar{Z}\perp\sum(Z_i-\bar{Z})^2$ 由 Cochran 定理保证。

---

### D.3.7（Ch.9，二维正态分布）

**题目**：$(X,Y)\sim N(\mu_1,\mu_2,\sigma_1^2,\sigma_2^2,\rho)$，推导边际分布；证明 $\rho=0\Leftrightarrow X\perp Y$；写出条件分布公式。

**思路**：边际化通过对联合密度积分消去一维；正态情形不相关等价独立（特殊性质）。

**解**：

**(a) 边际分布 $X\sim N(\mu_1,\sigma_1^2)$**：

二维正态联合密度（设 $\rho\in(-1,1)$）：

$$f(x,y) = \frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}}\exp\!\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_1)^2}{\sigma_1^2}-2\rho\frac{(x-\mu_1)(y-\mu_2)}{\sigma_1\sigma_2}+\frac{(y-\mu_2)^2}{\sigma_2^2}\right]\right)$$

配方：将指数中 $y$ 的部分写为

$$-\frac{1}{2\sigma_2^2(1-\rho^2)}\left(y-\mu_2-\rho\frac{\sigma_2}{\sigma_1}(x-\mu_1)\right)^2-\frac{(x-\mu_1)^2}{2\sigma_1^2}$$

对 $y$ 积分（高斯积分），结果为

$$f_X(x) = \frac{1}{\sqrt{2\pi}\sigma_1}\exp\!\left(-\frac{(x-\mu_1)^2}{2\sigma_1^2}\right)$$

即 $X\sim N(\mu_1,\sigma_1^2)$。$\square$

**(b) $\rho=0\Leftrightarrow X\perp Y$**：

- 若 $\rho=0$，联合密度分解为

$$f(x,y) = \frac{1}{\sqrt{2\pi}\sigma_1}e^{-\frac{(x-\mu_1)^2}{2\sigma_1^2}}\cdot\frac{1}{\sqrt{2\pi}\sigma_2}e^{-\frac{(y-\mu_2)^2}{2\sigma_2^2}} = f_X(x)\cdot f_Y(y)$$

联合密度可分离，故 $X\perp Y$。

- 若 $X\perp Y$，则 $\mathrm{Cov}(X,Y)=0$，从而 $\rho=0$。$\square$

> ⚠️ **仅对正态分布**，不相关 $\Rightarrow$ 独立。一般分布此结论不成立。

**(c) 条件分布 $Y\mid X=x$**：

$$Y\mid X=x\;\sim\;N\!\left(\mu_2+\rho\frac{\sigma_2}{\sigma_1}(x-\mu_1),\;\sigma_2^2(1-\rho^2)\right)$$

条件均值为 $x$ 的线性函数，条件方差 $\sigma_2^2(1-\rho^2)$ 与 $x$ 无关。

**答案**：$\boxed{Y\mid X=x\sim N\!\left(\mu_2+\rho\dfrac{\sigma_2}{\sigma_1}(x-\mu_1),\;\sigma_2^2(1-\rho^2)\right)}$

---

### D.3.8（Ch.8，对数正态分布）

**题目**：$Y=e^X$，$X\sim N(\mu,\sigma^2)$，求 $Y$ 的密度；利用正态 MGF 求 $E[Y]$、$\mathrm{Var}(Y)$；证明 $\ln Y\sim N(\mu,\sigma^2)$。

**思路**：变量变换法求密度；$E[e^{tX}]$ 即正态 MGF，令 $t=1$ 得 $E[Y]$。

**解**：

**(a) 对数正态密度**：

$X=\ln Y$，由变量变换：

$$f_Y(y) = f_X(\ln y)\cdot\frac{1}{y} = \frac{1}{\sqrt{2\pi}\sigma}\exp\!\left(-\frac{(\ln y-\mu)^2}{2\sigma^2}\right)\cdot\frac{1}{y},\quad y>0$$

**(b) $E[Y]$ 与 $\mathrm{Var}(Y)$**：

利用正态 MGF $M_X(t)=e^{\mu t+\sigma^2t^2/2}$：

$$E[Y] = E[e^X] = M_X(1) = e^{\mu+\sigma^2/2}$$

$$E[Y^2] = E[e^{2X}] = M_X(2) = e^{2\mu+2\sigma^2}$$

$$\mathrm{Var}(Y) = e^{2\mu+2\sigma^2} - e^{2\mu+\sigma^2} = e^{2\mu+\sigma^2}(e^{\sigma^2}-1)$$

**(c) $\ln Y\sim N(\mu,\sigma^2)$**：

由定义 $Y=e^X$，$X\sim N(\mu,\sigma^2)$，取对数 $\ln Y = X\sim N(\mu,\sigma^2)$。$\square$

**答案**：$\boxed{E[Y]=e^{\mu+\sigma^2/2},\quad\mathrm{Var}(Y)=e^{2\mu+\sigma^2}(e^{\sigma^2}-1)}$

---

### D.3.9（Ch.7，二项分布的泊松极限）

**题目**：$X_n\sim B(n,p_n)$，$np_n\to\lambda$，证明 $P(X_n=k)\to\frac{\lambda^k e^{-\lambda}}{k!}$；讨论近似条件；数值对比。

**思路**：直接对二项 PMF 取极限，逐步分析各因子。

**解**：

**(a) 极限证明**：

固定 $k$，令 $\lambda_n=np_n\to\lambda$，$p_n=\lambda_n/n$，

$$P(X_n=k) = \binom{n}{k}p_n^k(1-p_n)^{n-k} = \frac{n(n-1)\cdots(n-k+1)}{k!}\cdot\frac{\lambda_n^k}{n^k}\cdot\left(1-\frac{\lambda_n}{n}\right)^{n-k}$$

分析三个因子：

- $\dfrac{n(n-1)\cdots(n-k+1)}{n^k} = \prod_{j=0}^{k-1}\left(1-\dfrac{j}{n}\right)\to 1$

- $\lambda_n^k\to\lambda^k$

- $\left(1-\dfrac{\lambda_n}{n}\right)^{n-k}\to e^{-\lambda}$（经典极限）

故

$$P(X_n=k)\to\frac{\lambda^k}{k!}\cdot e^{-\lambda}\qquad\square$$

**(b) 近似有效条件**：

$n$ 大（$n\ge20$）、$p$ 小（$p\le0.05$），$\lambda=np$ 适中（$\lambda\le5$）时近似效果好。

**(c) $n=100$，$p=0.02$，$k=0$**：

$$\lambda = 100\times0.02 = 2$$

泊松近似：$P(X=0)\approx e^{-2}\approx0.1353$

精确值：$P(X=0)=(0.98)^{100}$

$$\ln(0.98^{100})=100\ln0.98\approx100\times(-0.0202)=-2.020$$

精确值 $\approx e^{-2.020}\approx0.1326$

误差约 $\dfrac{0.1353-0.1326}{0.1353}\approx2\%$，近似良好。

**答案**：$\boxed{P(X_n=k)\to\dfrac{\lambda^k e^{-\lambda}}{k!}}$；数值对比误差约 2%。

---

### D.3.10（Ch.8，$t$ 分布的推导）

**题目**：$Z\sim N(0,1)$，$V\sim\chi^2(n)$，$Z\perp V$，$T=Z/\sqrt{V/n}$，写出密度；讨论 $n\to\infty$ 的极限；说明应用。

**思路**：$t$ 分布由正态和卡方构造，尾部比正态重；大自由度趋近标准正态。

**解**：

**(a) $T\sim t(n)$ 的密度**：

$$f_T(t) = \frac{\Gamma\!\left(\dfrac{n+1}{2}\right)}{\sqrt{n\pi}\,\Gamma\!\left(\dfrac{n}{2}\right)}\left(1+\frac{t^2}{n}\right)^{-\frac{n+1}{2}},\quad t\in\mathbb{R}$$

$t(n)$ 分布关于 $0$ 对称，自由度 $n>2$ 时方差为 $n/(n-2)$，$n>1$ 时均值为 $0$。

**(b) $n\to\infty$ 趋向标准正态**：

$$\left(1+\frac{t^2}{n}\right)^{-\frac{n+1}{2}}\to e^{-t^2/2}$$

且 $\Gamma$-比值趋向 $1/\sqrt{2\pi}$（Stirling 近似），故 $f_T(t)\to\frac{1}{\sqrt{2\pi}}e^{-t^2/2}$，即 $T\xrightarrow{d}N(0,1)$。

物理直觉：$V/n\xrightarrow{P}1$（大数定律），$T=Z/\sqrt{V/n}\approx Z\sim N(0,1)$。

**(c) $t$ 分布在小样本推断中的作用**：

当总体方差 $\sigma^2$ 未知、样本量 $n$ 小时，用样本方差 $S^2$ 替代 $\sigma^2$，统计量

$$T = \frac{\bar{X}-\mu}{S/\sqrt{n}}\sim t(n-1)$$

**应用场景**：单样本均值检验（$n<30$），置信区间的构造（比正态区间更宽，反映方差估计的不确定性）。

**答案**：$T\sim t(n)$，$\boxed{f_T(t)=\dfrac{\Gamma\!\left(\frac{n+1}{2}\right)}{\sqrt{n\pi}\,\Gamma\!\left(\frac{n}{2}\right)}\left(1+\dfrac{t^2}{n}\right)^{-\frac{n+1}{2}}}$；$n\to\infty$ 时趋近 $N(0,1)$。

---

### D.3.11（Ch.9，多项分布）

**题目**：$n=12$ 球分到 3 盒，$p_1=1/2,p_2=1/3,p_3=1/6$，求多项联合分布、$E[X_i]$、$\mathrm{Var}(X_i)$、$\mathrm{Cov}(X_1,X_2)$。

**思路**：多项分布的边际为二项；协方差 $\mathrm{Cov}(X_i,X_j)=-np_ip_j$（因竞争关系为负）。

**解**：

**(a) 联合 PMF**：

$$P(X_1=k_1,X_2=k_2,X_3=k_3) = \frac{12!}{k_1!\,k_2!\,k_3!}\left(\frac{1}{2}\right)^{k_1}\left(\frac{1}{3}\right)^{k_2}\left(\frac{1}{6}\right)^{k_3}$$

其中 $k_1+k_2+k_3=12$，$k_i\ge0$。

**(b) 边际均值与方差**（$X_i\sim B(12,p_i)$）：

| $i$ | $p_i$ | $E[X_i]=12p_i$ | $\mathrm{Var}(X_i)=12p_i(1-p_i)$ |
|-----|--------|-----------------|----------------------------------|
| 1 | $1/2$ | $6$ | $3$ |
| 2 | $1/3$ | $4$ | $8/3\approx2.667$ |
| 3 | $1/6$ | $2$ | $5/3\approx1.667$ |

**(c) $\mathrm{Cov}(X_1,X_2)$**：

公式：$\mathrm{Cov}(X_i,X_j)=-np_ip_j$（$i\neq j$）

验证思路：$X_1+X_2+X_3=12$（常数），故 $\mathrm{Var}(X_1+X_2+X_3)=0$，展开可得协方差为负。

$$\mathrm{Cov}(X_1,X_2) = -12\times\frac{1}{2}\times\frac{1}{3} = -2$$

**答案**：$\boxed{\mathrm{Cov}(X_1,X_2)=-2,\quad E[X_1]=6,\ E[X_2]=4,\ E[X_3]=2}$

---

### D.3.12（Ch.8，$F$ 分布的性质）

**题目**：$U\sim\chi^2(m)$，$V\sim\chi^2(n)$，$U\perp V$，$F=(U/m)/(V/n)$，讨论性质与应用。

**思路**：$F$ 分布是两个归一化卡方之比；倒数的分布自由度互换；用于方差齐性检验。

**解**：

**(a) $F\sim F(m,n)$ 的定义**：

$F=(U/m)/(V/n)$ 即两个独立 $\chi^2$ 除以各自自由度之比，定义即为 $F(m,n)$ 分布。

密度：

$$f_F(x) = \frac{\Gamma\!\left(\frac{m+n}{2}\right)}{\Gamma\!\left(\frac{m}{2}\right)\Gamma\!\left(\frac{n}{2}\right)}\left(\frac{m}{n}\right)^{m/2}x^{m/2-1}\left(1+\frac{m}{n}x\right)^{-(m+n)/2},\quad x>0$$

**(b) $1/F\sim F(n,m)$**：

$$\frac{1}{F} = \frac{V/n}{U/m}$$

分子 $V\sim\chi^2(n)$，分母 $U\sim\chi^2(m)$，$V\perp U$，故 $1/F\sim F(n,m)$。$\square$

**(c) $F$ 分布与方差齐性检验**：

设两独立正态总体 $N(\mu_1,\sigma_1^2)$、$N(\mu_2,\sigma_2^2)$，样本量 $m,n$，样本方差 $S_1^2,S_2^2$。

在 $H_0:\sigma_1^2=\sigma_2^2$ 下，

$$F = \frac{S_1^2}{S_2^2}\sim F(m-1,\,n-1)$$

单侧（$H_1:\sigma_1^2>\sigma_2^2$）拒绝域：$F>F_\alpha(m-1,n-1)$（查 $F$ 分布临界值表）。

**答案**：$\boxed{1/F\sim F(n,m)}$；$F$ 检验检验两总体方差是否相等。

---

### D.3.13（Ch.7，复合泊松分布）

**题目**：$N\sim\mathrm{Poisson}(\lambda)$，每次损失 $X_i\sim\mathrm{Exp}(\mu)$，总损失 $S=\sum_{i=1}^N X_i$，用重期望和条件方差公式求 $E[S]$、$\mathrm{Var}(S)$。

**思路**：复合分布：先对 $N$ 条件化，用全期望公式和全方差公式。

**解**：

**(a) $E[S]$（重期望公式）**：

$$E[S] = E[E[S\mid N]] = E\!\left[N\cdot E[X_1]\right] = E[N]\cdot E[X_1]$$

其中 $E[X_1]=1/\mu$（指数分布），$E[N]=\lambda$，故

$$E[S] = \lambda\cdot\frac{1}{\mu} = \frac{\lambda}{\mu}$$

**(b) $\mathrm{Var}(S)$（条件方差公式）**：

$$\mathrm{Var}(S) = E[\mathrm{Var}(S\mid N)] + \mathrm{Var}(E[S\mid N])$$

- $E[S\mid N=n]=n/\mu$，故 $\mathrm{Var}(E[S\mid N])=\mathrm{Var}(N/\mu)=\lambda/\mu^2$

- $\mathrm{Var}(S\mid N=n)=n\cdot\mathrm{Var}(X_1)=n/\mu^2$，故 $E[\mathrm{Var}(S\mid N)]=E[N]/\mu^2=\lambda/\mu^2$

$$\mathrm{Var}(S) = \frac{\lambda}{\mu^2}+\frac{\lambda}{\mu^2} = \frac{2\lambda}{\mu^2}$$

**(c) $\lambda=2$，$\mu=1$**：

$$E[S] = \frac{2}{1} = 2,\qquad\mathrm{Var}(S) = \frac{2\times2}{1^2} = 4$$

**答案**：$\boxed{E[S]=\dfrac{\lambda}{\mu}=2,\quad\mathrm{Var}(S)=\dfrac{2\lambda}{\mu^2}=4}$

> ⚠️ 复合泊松方差公式一般形式：$\mathrm{Var}(S)=\lambda E[X_1^2]=\lambda(\mathrm{Var}(X_1)+(E[X_1])^2)$，指数分布时 $E[X^2]=2/\mu^2$，故 $\mathrm{Var}(S)=\lambda\cdot2/\mu^2=2\lambda/\mu^2$，与上式一致。

---

### D.3.14（Ch.8，混合正态分布）

**题目**：$X\mid\Theta=\theta\sim N(\theta,1)$，$\Theta\sim N(0,\tau^2)$，求 $X$ 的边际分布、$E[X]$、$\mathrm{Var}(X)$、$\mathrm{Cov}(X,\Theta)$。

**思路**：正态-正态混合仍为正态（高斯积分封闭性）；用全期望/全方差公式求矩。

**解**：

**(a) $X$ 的边际分布**：

$$X = \Theta + \varepsilon,\quad \varepsilon\sim N(0,1)\perp\Theta\sim N(0,\tau^2)$$

两独立正态之和仍为正态：

$$X\sim N(0,\;\tau^2+1)$$

**(b) $E[X]$ 与 $\mathrm{Var}(X)$**：

$$E[X] = E[E[X\mid\Theta]] = E[\Theta] = 0$$

$$\mathrm{Var}(X) = E[\mathrm{Var}(X\mid\Theta)]+\mathrm{Var}(E[X\mid\Theta]) = E[1]+\mathrm{Var}(\Theta) = 1+\tau^2$$

**(c) $\mathrm{Cov}(X,\Theta)$**：

$$\mathrm{Cov}(X,\Theta) = E[X\Theta]-E[X]E[\Theta] = E[X\Theta]$$

$$E[X\Theta] = E[E[X\Theta\mid\Theta]] = E[\Theta\cdot E[X\mid\Theta]] = E[\Theta\cdot\Theta] = E[\Theta^2] = \mathrm{Var}(\Theta) = \tau^2$$

$$\mathrm{Cov}(X,\Theta) = \tau^2$$

验证：$\rho_{X,\Theta}=\dfrac{\tau^2}{\sqrt{\tau^2+1}\cdot\tau}=\dfrac{\tau}{\sqrt{\tau^2+1}}\in(0,1)$，合理。

**答案**：$\boxed{X\sim N(0,\tau^2+1),\quad E[X]=0,\quad\mathrm{Var}(X)=1+\tau^2,\quad\mathrm{Cov}(X,\Theta)=\tau^2}$

---

### D.3.15（Ch.9，Dirichlet 分布简介）

**题目**：$(X_1,X_2,X_3)\sim\mathrm{Dirichlet}(1,1,1)$，求联合密度、边际分布、$E[X_i]$、$\mathrm{Var}(X_i)$。

**思路**：$\mathrm{Dir}(1,1,1)$ 是单纯形上的均匀分布；边际为 $\mathrm{Beta}(\alpha_i,\alpha_0-\alpha_i)$。

**解**：

**(a) 联合密度**：

$\alpha_0=1+1+1=3$，$B(\boldsymbol{\alpha})=\dfrac{\prod\Gamma(\alpha_i)}{\Gamma(\alpha_0)}=\dfrac{1!\cdot1!\cdot1!}{2!}=\dfrac{1}{2}$，

$$f(x_1,x_2,x_3) = \frac{1}{B(\boldsymbol{\alpha})}\prod_{i=1}^3 x_i^{\alpha_i-1} = 2\cdot x_1^0 x_2^0 x_3^0 = 2$$

在单纯形 $\{x_1+x_2+x_3=1,\,x_i>0\}$ 上为常数 $2$（单纯形面积为 $1/2$，故归一化）。

**(b) 边际分布**：

$$X_i\sim\mathrm{Beta}(\alpha_i,\,\alpha_0-\alpha_i) = \mathrm{Beta}(1,\,2)$$

$\mathrm{Beta}(1,2)$ 的密度：

$$f_{X_i}(x) = \frac{\Gamma(3)}{\Gamma(1)\Gamma(2)}x^0(1-x)^1 = 2(1-x),\quad 0<x<1$$

**(c) $E[X_i]$ 与 $\mathrm{Var}(X_i)$**：

Dirichlet 公式：$E[X_i]=\alpha_i/\alpha_0$，$\mathrm{Var}(X_i)=\dfrac{\alpha_i(\alpha_0-\alpha_i)}{\alpha_0^2(\alpha_0+1)}$

$$E[X_i] = \frac{1}{3},\qquad\mathrm{Var}(X_i) = \frac{1\times2}{9\times4} = \frac{2}{36} = \frac{1}{18}$$

Beta(1,2) 直接验证：$E[X]=\dfrac{1}{1+2}=\dfrac{1}{3}$，$\mathrm{Var}(X)=\dfrac{1\times2}{(1+2)^2(1+2+1)}=\dfrac{2}{36}=\dfrac{1}{18}$。$\checkmark$

**答案**：$\boxed{X_i\sim\mathrm{Beta}(1,2),\quad E[X_i]=\dfrac{1}{3},\quad\mathrm{Var}(X_i)=\dfrac{1}{18}}$

---

## E 提高题详解（8 题）

### E.3.1（Ch.7+Ch.8，指数族 + 充分统计量 + 自然参数）

**题目**：证明泊松、正态、伽马属于指数族；证明 $\nabla A=E[\mathbf{T}]$、$\nabla^2 A=\mathrm{Cov}[\mathbf{T}]$；推导 Fisher 信息；讨论自然梯度。

**思路**：将分布密度/PMF 化为指数族标准形式，读取自然参数和充分统计量；对 $A(\boldsymbol{\eta})$ 求导。

**解**：

**(a) 三类分布属于指数族**

**泊松分布 $\mathrm{Poisson}(\lambda)$**：

$$P(X=x) = \frac{e^{-\lambda}\lambda^x}{x!} = \frac{1}{x!}\exp\!\bigl(x\ln\lambda - \lambda\bigr)$$

令 $\eta=\ln\lambda$，$T(x)=x$，$A(\eta)=e^\eta=\lambda$，$h(x)=1/x!$：

$$P(X=x) = h(x)\exp(\eta T(x)-A(\eta))$$

自然参数 $\eta=\ln\lambda$，充分统计量 $T(X)=X$。

**正态分布 $N(\mu,\sigma^2)$**：

$$f(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\!\left(-\frac{x^2}{2\sigma^2}+\frac{\mu}{\sigma^2}x-\frac{\mu^2}{2\sigma^2}-\ln\sigma\right)$$

令 $\boldsymbol{\eta}=\left(\dfrac{\mu}{\sigma^2},\,-\dfrac{1}{2\sigma^2}\right)$，$\mathbf{T}(x)=(x,x^2)$，

$$A(\boldsymbol{\eta}) = -\frac{\eta_1^2}{4\eta_2}+\frac{1}{2}\ln\!\left(-\frac{\pi}{\eta_2}\right)$$

自然参数 $\boldsymbol{\eta}=(\mu/\sigma^2,-1/2\sigma^2)$，充分统计量 $\mathbf{T}=(X,X^2)$。

**伽马分布 $\mathrm{Gamma}(\alpha,\beta)$**（速率参数化）：

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x} = x^{\alpha-1}e^{-1}\cdot\exp\!\bigl((\alpha-1)\ln x-\beta x-\ln\Gamma(\alpha)+\alpha\ln\beta\bigr)$$

令 $\boldsymbol{\eta}=(\alpha-1,-\beta)$，$\mathbf{T}(x)=(\ln x,x)$，

$$A(\boldsymbol{\eta}) = \ln\Gamma(\eta_1+1)-(\eta_1+1)\ln(-\eta_2)$$

自然参数 $\boldsymbol{\eta}=(\alpha-1,-\beta)$，充分统计量 $\mathbf{T}=(\ln X,X)$。

**(b) $\nabla A=E[\mathbf{T}]$，$\nabla^2 A=\mathrm{Cov}[\mathbf{T}]$**

归一化条件：$\int h(x)\exp(\boldsymbol{\eta}^\top\mathbf{T}(x)-A(\boldsymbol{\eta}))dx=1$

对 $\boldsymbol{\eta}$ 求导（在积分号下对参数微分，合法性由正则性保证）：

$$\int h(x)\exp(\cdots)\left(\mathbf{T}(x)-\nabla A\right)dx = 0$$

$$\Rightarrow\quad E[\mathbf{T}(X)] = \nabla A(\boldsymbol{\eta})\qquad\square$$

再求导（二阶）：

$$\nabla^2 A(\boldsymbol{\eta}) = \int h(x)e^{\boldsymbol{\eta}^\top\mathbf{T}-A}\left(\mathbf{T}-\nabla A\right)\left(\mathbf{T}-\nabla A\right)^\top dx = \mathrm{Cov}[\mathbf{T}(X)]\qquad\square$$

**(c) Fisher 信息矩阵 $\mathcal{I}(\boldsymbol{\eta})=\nabla^2 A(\boldsymbol{\eta})$**

对数似然：$\ell(\boldsymbol{\eta};x)=\boldsymbol{\eta}^\top\mathbf{T}(x)-A(\boldsymbol{\eta})+\ln h(x)$

得分函数：$\nabla_\eta\ell = \mathbf{T}(x)-\nabla A(\boldsymbol{\eta})$

Fisher 信息：

$$\mathcal{I}(\boldsymbol{\eta}) = E\!\left[\nabla\ell\,(\nabla\ell)^\top\right] = E\!\left[(\mathbf{T}-E[\mathbf{T}])(\mathbf{T}-E[\mathbf{T}])^\top\right] = \mathrm{Cov}[\mathbf{T}] = \nabla^2 A(\boldsymbol{\eta})\qquad\square$$

**(d) 自然梯度与 K-FAC**

**普通梯度**的更新方向 $\nabla_\theta\mathcal{L}$ 在参数空间中依赖坐标系的度量，对参数重参数化不稳定。

**自然梯度**利用 Fisher 信息矩阵作为黎曼度量：

$$\tilde{\nabla}\mathcal{L} = \mathcal{I}(\boldsymbol{\theta})^{-1}\nabla_\theta\mathcal{L}$$

这等价于在以 KL 散度度量的分布空间中做最速下降，对参数化方式不变（协变性）。

**K-FAC**（Kronecker-factored Approximate Curvature）：对深度网络，Fisher 矩阵维度巨大（$O(d^2)$）。K-FAC 利用各层激活和梯度的 Kronecker 积近似：

$$\mathcal{I}_\ell \approx A_\ell\otimes G_\ell$$

其中 $A_\ell=E[\mathbf{a}_\ell\mathbf{a}_\ell^\top]$（激活协方差），$G_\ell=E[\mathbf{g}_\ell\mathbf{g}_\ell^\top]$（梯度协方差），求逆代价降为 $O(d_\ell^3)$ 而非 $O(d_\ell^6)$，大幅降低计算成本。

**答案**：$\boxed{\nabla A(\boldsymbol{\eta})=E[\mathbf{T}(X)],\quad\nabla^2 A(\boldsymbol{\eta})=\mathrm{Cov}[\mathbf{T}(X)]=\mathcal{I}(\boldsymbol{\eta})}$

---

### E.3.2（Ch.8+Ch.9，多元正态 + 条件分布 + 高斯过程）

**题目**：分块多元正态，证明边际分布；推导条件分布（Schur 补）；联系 GP 回归；分析 $O(n^3)$ 瓶颈与稀疏近似。

**思路**：边际由 MGF 或特征函数读出；条件分布通过"配方 + 线性变换"得到；GP 回归是无限维多元正态的条件分布特例。

**解**：

**(a) 边际分布 $\mathbf{X}_1\sim N(\boldsymbol{\mu}_1,\boldsymbol{\Sigma}_{11})$**

多元正态 $\mathbf{X}$ 的特征函数：

$$\varphi_{\mathbf{X}}(\mathbf{t}) = \exp\!\left(i\mathbf{t}^\top\boldsymbol{\mu}-\tfrac{1}{2}\mathbf{t}^\top\boldsymbol{\Sigma}\mathbf{t}\right)$$

令 $\mathbf{t}=(\mathbf{s},\mathbf{0})$（仅 $\mathbf{X}_1$ 方向非零）：

$$\varphi_{\mathbf{X}_1}(\mathbf{s}) = \exp\!\left(i\mathbf{s}^\top\boldsymbol{\mu}_1-\tfrac{1}{2}\mathbf{s}^\top\boldsymbol{\Sigma}_{11}\mathbf{s}\right)$$

这正是 $N(\boldsymbol{\mu}_1,\boldsymbol{\Sigma}_{11})$ 的特征函数。$\square$

**(b) 条件分布 $\mathbf{X}_1\mid\mathbf{X}_2=\mathbf{x}_2$**

构造辅助变量 $\mathbf{Y}=\mathbf{X}_1-\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\mathbf{X}_2$，则：

$$\mathrm{Cov}(\mathbf{Y},\mathbf{X}_2) = \boldsymbol{\Sigma}_{12}-\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{22} = \mathbf{0}$$

故 $\mathbf{Y}\perp\mathbf{X}_2$（多元正态下不相关即独立），$\mathrm{Var}(\mathbf{Y})=\boldsymbol{\Sigma}_{11}-\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$（Schur 补）。

给定 $\mathbf{X}_2=\mathbf{x}_2$：

$$\mathbf{X}_1 = \mathbf{Y}+\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\mathbf{x}_2$$

$$\boldsymbol{\mu}_{1\mid2} = \boldsymbol{\mu}_1+\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2-\boldsymbol{\mu}_2)$$

$$\boldsymbol{\Sigma}_{1\mid2} = \boldsymbol{\Sigma}_{11}-\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}\qquad\square$$

**(c) GP 回归与条件分布的联系**

高斯过程 $f\sim\mathcal{GP}(m(\cdot),k(\cdot,\cdot))$，训练集 $\mathbf{X}_{\text{tr}}$ 处观测 $\mathbf{y}=f(\mathbf{X}_{\text{tr}})+\boldsymbol{\varepsilon}$（$\boldsymbol{\varepsilon}\sim N(\mathbf{0},\sigma_n^2\mathbf{I})$）。

将 $(f(\mathbf{X}_*),\mathbf{y})$ 视为联合高斯：

$$\begin{pmatrix}f_*\\\mathbf{y}\end{pmatrix}\sim N\!\left(\mathbf{0},\,\begin{pmatrix}K_{**} & K_{*n}\\K_{n*} & K_{nn}+\sigma_n^2\mathbf{I}\end{pmatrix}\right)$$

由 (b) 得后验：

$$f_*\mid\mathbf{y}\sim N\!\left(K_{*n}(K_{nn}+\sigma_n^2\mathbf{I})^{-1}\mathbf{y},\; K_{**}-K_{*n}(K_{nn}+\sigma_n^2\mathbf{I})^{-1}K_{n*}\right)$$

这正是 (b) 中条件均值与 Schur 补协方差的直接应用。

**(d) $O(n^3)$ 瓶颈与稀疏近似**

瓶颈：$(K_{nn}+\sigma_n^2\mathbf{I})^{-1}$ 的 Cholesky 分解需 $O(n^3)$，存储 $O(n^2)$，大样本不可行。

**诱导点法（Sparse GP / FITC）**：引入 $m\ll n$ 个诱导点 $\mathbf{Z}$，用

$$K_{nn}\approx K_{nz}K_{zz}^{-1}K_{zn}$$

将求逆代价降为 $O(nm^2)$，$m$ 可选 $100\sim1000$。

**随机特征近似（Bochner 定理）**：核函数 $k(x,x')=E_{\boldsymbol{\omega}}[\phi(\boldsymbol{\omega},x)\phi(\boldsymbol{\omega},x')]$，用 $D$ 个随机 Fourier 特征逼近核矩阵，代价 $O(nD)$，但近似精度受 $D$ 控制。

**答案**：条件分布由 Schur 补给出；GP 回归是多元正态条件化的无限维推广；$\boxed{\boldsymbol{\mu}_{1\mid2}=\boldsymbol{\mu}_1+\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2-\boldsymbol{\mu}_2)}$。

---

### E.3.3（Ch.7，负二项分布 + 过度离散 + GLM）

**题目**：$Y\sim\mathrm{NegBin}(r,p)$，证明均值方差；证明泊松-伽马混合表示；解释过度离散；讨论 GLM 实现。

**思路**：PMF 由负二项组合数定义；混合分布通过对 $\Lambda$ 积分化简为负二项；过度离散根源是参数的随机性。

**解**：

**(a) PMF 与矩**

$Y$ 为第 $r$ 次成功前的失败次数，$P(Y=y)=\binom{y+r-1}{y}(1-p)^y p^r$，$y=0,1,2,\ldots$

**期望**（利用 PGF 或负二项求和）：

$$E[Y] = \frac{r(1-p)}{p}$$

**方差**：

$$\mathrm{Var}(Y) = \frac{r(1-p)}{p^2}$$

验证：$\mathrm{Var}(Y)/E[Y]=1/p>1$（过度离散，方差超过均值）。$\square$

**(b) 泊松-伽马混合**

设 $Y\mid\Lambda\sim\mathrm{Poisson}(\Lambda)$，$\Lambda\sim\mathrm{Gamma}(r,\beta)$（$\beta=p/(1-p)$，速率参数），

$$P(Y=y) = \int_0^\infty \frac{e^{-\lambda}\lambda^y}{y!}\cdot\frac{\beta^r}{\Gamma(r)}\lambda^{r-1}e^{-\beta\lambda}\,d\lambda = \frac{\beta^r}{y!\,\Gamma(r)}\int_0^\infty\lambda^{y+r-1}e^{-(1+\beta)\lambda}\,d\lambda$$

$$= \frac{\beta^r}{y!\,\Gamma(r)}\cdot\frac{\Gamma(y+r)}{(1+\beta)^{y+r}} = \frac{\Gamma(y+r)}{y!\,\Gamma(r)}\left(\frac{\beta}{1+\beta}\right)^r\left(\frac{1}{1+\beta}\right)^y$$

令 $p=\beta/(1+\beta)$，得 $P(Y=y)=\binom{y+r-1}{y}p^r(1-p)^y$，即 $\mathrm{NegBin}(r,p)$。$\square$

**(c) 过度离散的根源**

全方差公式：

$$\mathrm{Var}(Y) = E[\mathrm{Var}(Y\mid\Lambda)]+\mathrm{Var}(E[Y\mid\Lambda]) = E[\Lambda]+\mathrm{Var}(\Lambda)$$

（因为 $\mathrm{Var}(Y\mid\Lambda)=E[Y\mid\Lambda]=\Lambda$）。

$$\mathrm{Var}(Y) = \underbrace{E[\Lambda]}_{\text{泊松内在波动}}+\underbrace{\mathrm{Var}(\Lambda)}_{\text{速率本身的随机性}}>E[\Lambda]=E[Y]$$

**结论**：过度离散来源于"速率参数本身不确定（服从随机分布）"，泊松模型假设速率固定是错误的。

**(d) GLM 中的负二项回归**

对比：泊松回归 $\log\mu_i=\mathbf{x}_i^\top\boldsymbol{\beta}$；负二项回归在此基础上引入额外离散参数 $r$。

负对数似然（固定 $r$）：

$$-\ell = -\sum_{i=1}^n\left[\ln\Gamma(y_i+r)-\ln\Gamma(r)-\ln(y_i!)+r\ln p_i+y_i\ln(1-p_i)\right]$$

其中 $p_i=r/(r+\mu_i)$，$\mu_i=e^{\mathbf{x}_i^\top\boldsymbol{\beta}}$。

PyTorch 注意事项：
- 使用 `torch.distributions.NegativeBinomial(total_count=r, probs=p)` 或 `logits` 参数化；
- $r$ 可作为可学习参数（需保证正值，常用 `F.softplus`）；
- 数值稳定性：对数伽马函数用 `torch.lgamma`。

**答案**：$\boxed{E[Y]=\dfrac{r(1-p)}{p},\quad\mathrm{Var}(Y)=\dfrac{r(1-p)}{p^2}=E[Y]+\dfrac{(E[Y])^2}{r}}$；过度离散 $=$ 泊松内在波动 $+$ 速率随机性。

---

### E.3.4（Ch.8+Ch.9，$\chi^2$ 分布 + 卡方检验 + 独立性检验）

**题目**：从伽马推导 $\chi^2$ 密度；证明马氏距离分布；推导 Pearson 统计量的渐近分布；讨论效应量与正确使用。

**思路**：$\chi^2(k)=\mathrm{Gamma}(k/2,1/2)$；马氏距离通过 Cholesky 分解转化为标准正态的平方和。

**解**：

**(a) $\chi^2(k)$ 的密度、均值、方差**

$\chi^2(k)=\mathrm{Gamma}(k/2,\,1/2)$，代入 Gamma 密度：

$$f(x) = \frac{(1/2)^{k/2}}{\Gamma(k/2)}x^{k/2-1}e^{-x/2} = \frac{1}{2^{k/2}\Gamma(k/2)}x^{k/2-1}e^{-x/2},\quad x>0$$

由 Gamma 分布公式（$\alpha=k/2$，$\beta=1/2$）：

$$E[X] = \frac{\alpha}{\beta} = k,\qquad\mathrm{Var}(X) = \frac{\alpha}{\beta^2} = 2k$$

**(b) 马氏距离 $\sim\chi^2(d)$**

设 $\boldsymbol{\Sigma}$ 正定，Cholesky 分解 $\boldsymbol{\Sigma}=\mathbf{L}\mathbf{L}^\top$，令 $\mathbf{Z}=\mathbf{L}^{-1}(\mathbf{X}-\boldsymbol{\mu})\sim N_d(\mathbf{0},\mathbf{I}_d)$，则

$$(\mathbf{X}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{X}-\boldsymbol{\mu}) = \mathbf{Z}^\top\mathbf{Z} = \sum_{i=1}^d Z_i^2 \sim\chi^2(d)\qquad\square$$

**(c) Pearson 统计量渐近分布推导框架**

列联表 $(r\times c)$，$H_0$：行列独立，期望 $E_{ij}=n\hat{p}_{i\cdot}\hat{p}_{\cdot j}$。

向量化：令 $\mathbf{n}=(n_{11},\ldots,n_{rc})^\top$，观测计数 $\sim\mathrm{Multinomial}(n,\{p_{ij}\})$。

由多项分布 CLT，$(\mathbf{n}-n\mathbf{p})/\sqrt{n}\xrightarrow{d}N(\mathbf{0},\boldsymbol{\Sigma})$（$\boldsymbol{\Sigma}=\mathrm{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top$）。

Pearson 统计量是对这个多元正态进行二次型的连续映射，渐近分布为 $\chi^2$ 。自由度 $(r-1)(c-1)$：共 $rc-1$ 个自由参数（归一化约束），$H_0$ 下估计了 $(r-1)+(c-1)$ 个参数，故剩余 $\chi^2$ 自由度为

$$(rc-1)-[(r-1)+(c-1)] = (r-1)(c-1)$$

**(d) 效应量与实践显著性**

**Cramér's $V$**：

$$V = \sqrt{\frac{\chi^2/n}{\min(r-1,c-1)}}$$

$V\in[0,1]$，$V\approx0.1$（弱），$V\approx0.3$（中），$V\approx0.5$（强）。

**大样本问题**：$n$ 大时 $\chi^2\approx nV^2\min(\cdot)$，即使 $V=0.01$（实践上无意义），$n=10^6$ 时也会 $p<0.001$。

**ML 特征选择中的正确用法**：
- 不应仅凭 $p$ 值选特征（大数据下几乎所有特征都显著）；
- 应结合 $V$ 或互信息（MI）设置阈值（如 $V>0.05$）；
- 对高维特征用 FDR 控制（Benjamini-Hochberg）而非 Bonferroni 修正。

**答案**：$\boxed{\chi^2(k)\ 均值=k,\ 方差=2k;\quad \chi^2_{\text{Pearson}}\xrightarrow{d}\chi^2((r-1)(c-1))}$；实践中需同时报告 Cramér's $V$。

---

### E.3.5（Ch.8，Beta 分布 + 共轭先验 + 汤普森采样）

**题目**：证明 Beta-Bernoulli 共轭；证明 Beta 分布的矩；证明汤普森采样的贝叶斯最优性；推广到高斯奖励。

**思路**：共轭性通过核识别；矩由 Beta 函数比值给出；汤普森采样以概率最优化后验期望。

**解**：

**(a) Beta-Bernoulli 共轭**

先验 $\theta\sim\mathrm{Beta}(\alpha,\beta)$，观测 $s$ 次成功、$f$ 次失败，似然 $\theta^s(1-\theta)^f$。

后验 $\propto$ 先验 $\times$ 似然：

$$\pi(\theta\mid s,f) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}\cdot\theta^s(1-\theta)^f = \theta^{\alpha+s-1}(1-\theta)^{\beta+f-1}$$

这是 $\mathrm{Beta}(\alpha+s,\beta+f)$ 的核，故 $\theta\mid s,f\sim\mathrm{Beta}(\alpha+s,\beta+f)$。$\square$

**(b) Beta 分布的均值、众数、方差**

$\mathrm{Beta}(\alpha,\beta)$ 的密度 $\propto\theta^{\alpha-1}(1-\theta)^{\beta-1}$，$B(\alpha,\beta)=\Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)$。

**均值**：

$$E[\theta] = \int_0^1\theta\cdot\frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)}\,d\theta = \frac{B(\alpha+1,\beta)}{B(\alpha,\beta)} = \frac{\Gamma(\alpha+1)\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\alpha+\beta+1)} = \frac{\alpha}{\alpha+\beta}$$

**众数**（$\alpha,\beta>1$，令密度导数为零）：

$$\frac{d}{d\theta}\left[\theta^{\alpha-1}(1-\theta)^{\beta-1}\right] = 0\;\Rightarrow\;(\alpha-1)(1-\theta) = (\beta-1)\theta\;\Rightarrow\;\theta^*=\frac{\alpha-1}{\alpha+\beta-2}$$

**方差**：

$$E[\theta^2] = \frac{B(\alpha+2,\beta)}{B(\alpha,\beta)} = \frac{\alpha(\alpha+1)}{(\alpha+\beta)(\alpha+\beta+1)}$$

$$\mathrm{Var}(\theta) = \frac{\alpha(\alpha+1)}{(\alpha+\beta)(\alpha+\beta+1)}-\frac{\alpha^2}{(\alpha+\beta)^2} = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}\qquad\square$$

**(c) 汤普森采样的贝叶斯最优性**

设 $K$ 个臂，第 $k$ 臂后验 $\theta_k\sim\mathrm{Beta}(\alpha_k,\beta_k)$。汤普森采样：从每个后验采样 $\hat\theta_k$，选 $k^*=\arg\max_k\hat\theta_k$。

**贝叶斯最优性**：选臂 $k$ 的概率为

$$P(k\text{ 被选}) = P(\hat\theta_k>\hat\theta_j,\;\forall j\neq k) = P(\theta_k=\max_j\theta_j)$$

（因为采样后比较等价于后验最优概率）。这正是贝叶斯后验下臂 $k$ 为最优臂的概率，汤普森采样以后验最优概率选臂，是后验感知的最优策略。$\square$

**(d) 高斯奖励的共轭对与 UCB 对比**

高斯奖励 $r\sim N(\mu_k,\sigma^2)$（$\sigma^2$ 已知），共轭先验 $\mu_k\sim N(m_0,v_0^2)$，后验

$$\mu_k\mid r_1,\ldots,r_n\sim N\!\left(\frac{v_0^{-2}m_0+n\sigma^{-2}\bar{r}}{v_0^{-2}+n\sigma^{-2}},\;\frac{1}{v_0^{-2}+n\sigma^{-2}}\right)$$

**与 UCB 对比**：

| 维度 | UCB（UCB1）| 汤普森采样 |
|------|-----------|------------|
| 探索依据 | 置信上界（确定性）| 后验采样（随机性）|
| 理论保证 | $O(\sqrt{T\log T})$ 遗憾 | Bayes 遗憾最优 |
| 计算 | 简单，无需采样 | 需要后验采样 |
| 实践效果 | 对非平稳适应快 | 经验上更优，高维可扩展 |

**答案**：$\boxed{E[\theta]=\dfrac{\alpha}{\alpha+\beta},\quad\mathrm{Var}(\theta)=\dfrac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}}$；汤普森采样以后验最优概率选臂，贝叶斯最优。

---

### E.3.6（Ch.9，Dirichlet 分布 + 多项式后验 + LDA）

**题目**：Dirichlet 密度与矩；证明聚集性；写出 LDA 联合分布与 ELBO；分析 $\alpha_k$ 对稀疏性的影响。

**思路**：Dirichlet 是多项分布的共轭先验；聚集性通过积分边际化；LDA 用变分 EM 推断。

**解**：

**(a) 密度函数与均值、方差**

$\boldsymbol{\alpha}=(\alpha_1,\ldots,\alpha_K)$，$\alpha_0=\sum_k\alpha_k$，

$$f(\mathbf{p}) = \frac{\Gamma(\alpha_0)}{\prod_{k=1}^K\Gamma(\alpha_k)}\prod_{k=1}^K p_k^{\alpha_k-1},\quad \mathbf{p}\in\Delta^{K-1}$$

**均值**：$E[p_k]=\alpha_k/\alpha_0$

**方差**：$\mathrm{Var}(p_k)=\dfrac{\alpha_k(\alpha_0-\alpha_k)}{\alpha_0^2(\alpha_0+1)}$

**推导**（均值）：用 $\mathrm{Beta}$ 边际：$p_k\sim\mathrm{Beta}(\alpha_k,\alpha_0-\alpha_k)$，故 $E[p_k]=\alpha_k/\alpha_0$。

**(b) 聚集性（Aggregation Property）**

设将类 $K-1$ 和 $K$ 合并为新类 $K'$，令 $q=p_{K-1}+p_K$。

$$f(p_1,\ldots,p_{K-2},q) = \int_0^q f(p_1,\ldots,p_{K-2},p_{K-1},q-p_{K-1})\,dp_{K-1}$$

由 Beta 积分，$\int_0^q p_{K-1}^{\alpha_{K-1}-1}(q-p_{K-1})^{\alpha_K-1}dp_{K-1} = q^{\alpha_{K-1}+\alpha_K-1}B(\alpha_{K-1},\alpha_K)$，

积分后密度核变为 $\prod_{k=1}^{K-2}p_k^{\alpha_k-1}\cdot q^{(\alpha_{K-1}+\alpha_K)-1}$，

这是 $\mathrm{Dir}(\alpha_1,\ldots,\alpha_{K-2},\alpha_{K-1}+\alpha_K)$ 的核。$\square$

**(c) LDA 联合分布与变分 ELBO**

LDA 生成过程：

$$\boldsymbol{\theta}_d\sim\mathrm{Dir}(\boldsymbol{\alpha}),\quad \boldsymbol{\phi}_k\sim\mathrm{Dir}(\boldsymbol{\beta}),\quad z_{dn}\sim\mathrm{Cat}(\boldsymbol{\theta}_d),\quad w_{dn}\sim\mathrm{Cat}(\boldsymbol{\phi}_{z_{dn}})$$

**联合分布**：

$$p(\mathbf{w},\mathbf{z},\boldsymbol{\theta},\boldsymbol{\phi}) = \prod_k p(\boldsymbol{\phi}_k\mid\boldsymbol{\beta})\prod_d\left[p(\boldsymbol{\theta}_d\mid\boldsymbol{\alpha})\prod_n p(z_{dn}\mid\boldsymbol{\theta}_d)p(w_{dn}\mid\boldsymbol{\phi}_{z_{dn}})\right]$$

**变分 ELBO**（均值场近似 $q(\mathbf{z},\boldsymbol{\theta},\boldsymbol{\phi})=\prod_d q(\boldsymbol{\theta}_d)\prod_{dn}q(z_{dn})\prod_k q(\boldsymbol{\phi}_k)$）：

$$\mathrm{ELBO} = E_q[\log p(\mathbf{w},\mathbf{z},\boldsymbol{\theta},\boldsymbol{\phi})]-E_q[\log q(\mathbf{z},\boldsymbol{\theta},\boldsymbol{\phi})]$$

$$= E_q[\log p(\boldsymbol{\theta}\mid\boldsymbol{\alpha})]+E_q[\log p(\mathbf{z}\mid\boldsymbol{\theta})]+E_q[\log p(\mathbf{w}\mid\mathbf{z},\boldsymbol{\phi})]+E_q[\log p(\boldsymbol{\phi}\mid\boldsymbol{\beta})]-E_q[\log q]$$

每项均有解析形式（Dirichlet 和多项的期望对数均可用 $\psi$ 函数表达）。

**(d) $\alpha_k$ 对分布形状的影响**

- **$\alpha_k<1$（如 $\alpha_k=0.1$）**：密度在单纯形顶点附近集中，生成的 $\mathbf{p}$ 稀疏（大多数分量接近 0，仅少数分量显著）。类比 LLM：token 分布集中于少数高概率词（尖锐分布，低温采样效果）。

- **$\alpha_k>1$（如 $\alpha_k=5$）**：密度在单纯形中心附近集中，生成的 $\mathbf{p}$ 均匀（所有分量接近 $1/K$）。类比 LLM：token 分布均匀（高温采样，高多样性）。

- **$\alpha_k=1$**：单纯形上的均匀分布（如 D.3.15）。

**答案**：$\boxed{E[p_k]=\dfrac{\alpha_k}{\alpha_0}}$；聚集性保证边际仍为 Dirichlet；$\alpha_k<1$ 产生稀疏分布，$\alpha_k>1$ 产生均匀分布。

---

### E.3.7（Ch.8+Ch.9，Wishart 分布 + 多元样本协方差 + 矩阵分布）

**题目**：证明 $(n-1)\mathbf{S}\sim W_p(n-1,\boldsymbol{\Sigma})$；求 $E[\mathbf{S}]$ 与 $E[\mathbf{S}^{-1}]$；Marchenko-Pastur 定律；高维协方差估计。

**思路**：Wishart 由多元正态样本的外积定义；无偏性用期望的线性性；Marchenko-Pastur 描述高维随机矩阵特征值的极限分布。

**解**：

**(a) $(n-1)\mathbf{S}\sim W_p(n-1,\boldsymbol{\Sigma})$**

令 $\mathbf{Y}_i=\mathbf{X}_i-\bar{\mathbf{X}}$，则 $(n-1)\mathbf{S}=\sum_{i=1}^n\mathbf{Y}_i\mathbf{Y}_i^\top$。

利用正交投影：$(n-1)\mathbf{S}=(\mathbf{X}-\mathbf{1}\bar{\mathbf{X}}^\top)^\top(\mathbf{X}-\mathbf{1}\bar{\mathbf{X}}^\top)$，其等价于将 $n$ 个 $N_p(\mathbf{0},\boldsymbol{\Sigma})$ 向量（通过正交变换去均值得到 $n-1$ 个独立向量）的外积之和。

由 Wishart 分布定义，$m$ 个独立 $N_p(\mathbf{0},\boldsymbol{\Sigma})$ 向量外积之和为 $W_p(m,\boldsymbol{\Sigma})$，故

$$(n-1)\mathbf{S}\sim W_p(n-1,\boldsymbol{\Sigma})\qquad\square$$

**(b) $E[\mathbf{S}]=\boldsymbol{\Sigma}$（无偏性）**

$W_p(m,\boldsymbol{\Sigma})$ 的期望为 $m\boldsymbol{\Sigma}$，故

$$E[(n-1)\mathbf{S}] = (n-1)\boldsymbol{\Sigma}\;\Rightarrow\; E[\mathbf{S}]=\boldsymbol{\Sigma}\qquad\square$$

**$E[\mathbf{S}^{-1}]$（逆的期望）**：

Wishart 矩阵的逆服从逆 Wishart 分布 $W_p^{-1}(m,\boldsymbol{\Sigma}^{-1})$（$m>p+1$），其期望为

$$E\!\left[(n-1)\mathbf{S})^{-1}\right] = \frac{\boldsymbol{\Sigma}^{-1}}{n-1-p-1} = \frac{\boldsymbol{\Sigma}^{-1}}{n-p-2}$$

故 $E[\mathbf{S}^{-1}]=\dfrac{\boldsymbol{\Sigma}^{-1}}{n-p-2}$（要求 $n>p+2$）。$\square$

**(c) Marchenko-Pastur 定律**

当 $p,n\to\infty$，$p/n\to\gamma\in(0,1)$ 时，样本协方差（$\boldsymbol{\Sigma}=\mathbf{I}$）的特征值 $\lambda$ 的经验分布趋向：

$$f(\lambda) = \frac{1}{2\pi\gamma\lambda}\sqrt{(\lambda_+-\lambda)(\lambda-\lambda_-)},\quad \lambda\in[\lambda_-,\lambda_+]$$

其中 $\lambda_\pm=(1\pm\sqrt{\gamma})^2$。

**意义**：即使真实协方差 $=\mathbf{I}$（所有特征值为 1），高维样本协方差的特征值分布在 $[(1-\sqrt{\gamma})^2,(1+\sqrt{\gamma})^2]$ 范围内，产生严重膨胀（最大特征值可达 $(1+\sqrt{\gamma})^2\gg1$），导致样本协方差高估真实协方差的"波动性"。

**(d) 高维协方差估计的解决方案**

| 方法 | 原理 | 代价 |
|------|------|------|
| 正则化（Ledoit-Wolf）| $\hat{\boldsymbol{\Sigma}}=\alpha\mathbf{S}+(1-\alpha)\mathbf{I}$，缩减特征值 | $O(p^2)$ |
| 对角近似（BatchNorm）| $\hat{\boldsymbol{\Sigma}}\approx\mathrm{diag}(\mathbf{S})$，忽略协方差 | $O(p)$ |
| 低秩近似（PCA）| $\hat{\boldsymbol{\Sigma}}\approx\mathbf{U}_r\boldsymbol{\Lambda}_r\mathbf{U}_r^\top+\sigma^2\mathbf{I}$ | $O(pr)$ |

当 $B\ll d$ 时，$\text{rank}(\mathbf{S})\le B-1<d$，矩阵奇异，必须正则化。BatchNorm 的对角近似在实践中效果好，因为批内特征相关性弱。

**答案**：$\boxed{(n-1)\mathbf{S}\sim W_p(n-1,\boldsymbol{\Sigma}),\quad E[\mathbf{S}]=\boldsymbol{\Sigma},\quad E[\mathbf{S}^{-1}]=\boldsymbol{\Sigma}^{-1}/(n-p-2)}$；高维时需正则化克服 Marchenko-Pastur 膨胀。

---

### E.3.8（Ch.7+Ch.8，混合分布 + EM 算法 + 模式坍塌）

**题目**：GMM 完整数据对数似然；推导 EM 的 E 步和 M 步；证明 EM 单调不降；与 GAN 模式坍塌类比及 WGAN 解决方案。

**思路**：EM 通过最大化 ELBO（Q 函数）隐式提升边际似然；单调性由 Jensen 不等式保证；GAN 的模式坍塌类比于某分量退化。

**解**：

**(a) 完整数据对数似然**

隐变量 $Z_i\in\{1,\ldots,K\}$ 表示第 $i$ 个样本属于第 $k$ 个分量。

$$\log p(\mathbf{X},\mathbf{Z}\mid\boldsymbol{\theta}) = \sum_{i=1}^n\sum_{k=1}^K\mathbf{1}[Z_i=k]\left[\log\pi_k+\log N(\mathbf{x}_i;\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)\right]$$

$$= \sum_{i=1}^n\sum_{k=1}^K z_{ik}\left[\log\pi_k-\tfrac{1}{2}\log\vert\boldsymbol{\Sigma}_k\vert-\tfrac{1}{2}(\mathbf{x}_i-\boldsymbol{\mu}_k)^\top\boldsymbol{\Sigma}_k^{-1}(\mathbf{x}_i-\boldsymbol{\mu}_k)+\text{const}\right]$$

**(b) E 步与 M 步**

**E 步**（计算后验责任）：

$$r_{ik} = P(Z_i=k\mid\mathbf{x}_i,\boldsymbol{\theta}^{\text{old}}) = \frac{\pi_k^{\text{old}}N(\mathbf{x}_i;\boldsymbol{\mu}_k^{\text{old}},\boldsymbol{\Sigma}_k^{\text{old}})}{\sum_{j=1}^K\pi_j^{\text{old}}N(\mathbf{x}_i;\boldsymbol{\mu}_j^{\text{old}},\boldsymbol{\Sigma}_j^{\text{old}})}$$

**M 步**（最大化 $Q$ 函数 $=E_{Z\mid\mathbf{X},\boldsymbol{\theta}^{\text{old}}}[\log p(\mathbf{X},\mathbf{Z}\mid\boldsymbol{\theta})]$）：

令 $N_k=\sum_{i=1}^n r_{ik}$（有效样本数），

$$\pi_k^{\text{new}} = \frac{N_k}{n}$$

$$\boldsymbol{\mu}_k^{\text{new}} = \frac{\sum_i r_{ik}\mathbf{x}_i}{N_k}$$

$$\boldsymbol{\Sigma}_k^{\text{new}} = \frac{\sum_i r_{ik}(\mathbf{x}_i-\boldsymbol{\mu}_k^{\text{new}})(\mathbf{x}_i-\boldsymbol{\mu}_k^{\text{new}})^\top}{N_k}$$

**(c) EM 单调不降**

定义 $Q(\boldsymbol{\theta}\mid\boldsymbol{\theta}^{\text{old}})=E_{\mathbf{Z}\mid\mathbf{X},\boldsymbol{\theta}^{\text{old}}}[\log p(\mathbf{X},\mathbf{Z}\mid\boldsymbol{\theta})]$，

$$\log p(\mathbf{X}\mid\boldsymbol{\theta}) = Q(\boldsymbol{\theta}\mid\boldsymbol{\theta}^{\text{old}})-\underbrace{\sum_{\mathbf{z}}q(\mathbf{z})\log\frac{q(\mathbf{z})}{p(\mathbf{z}\mid\mathbf{X},\boldsymbol{\theta})}}_{=\,\mathrm{KL}(q\,\|\,p(\cdot\mid\mathbf{X},\boldsymbol{\theta}))\,\ge\,0} + \text{const}$$

E 步令 $q(\mathbf{z})=p(\mathbf{z}\mid\mathbf{X},\boldsymbol{\theta}^{\text{old}})$（KL $=0$），

M 步令 $\boldsymbol{\theta}^{\text{new}}=\arg\max_{\boldsymbol{\theta}}Q(\boldsymbol{\theta}\mid\boldsymbol{\theta}^{\text{old}})$（$Q$ 不减），

故 $\log p(\mathbf{X}\mid\boldsymbol{\theta}^{\text{new}})\ge\log p(\mathbf{X}\mid\boldsymbol{\theta}^{\text{old}})$。$\square$

**(d) 模式坍塌与 GAN/WGAN**

**EM 退化**：若某分量 $\pi_k\to0$，$N_k\to0$，导致 $\boldsymbol{\Sigma}_k^{-1}$ 不稳定（奇异）。实践上需要对 $\boldsymbol{\Sigma}_k$ 加正则项（对角加 $\epsilon\mathbf{I}$）或设置 $\pi_k$ 的下界。

**GAN 模式坍塌类比**：GAN 生成器 $G$ 学习将噪声映射到数据分布，若训练不均衡，$G$ 可能仅覆盖真实数据的部分模式（某些区域 $\pi_k\to0$），判别器无法有效区分剩余模式，导致生成器只输出少数样本类型（模式坍塌）。

**WGAN 的改进**：传统 GAN 用 JS 散度，当生成分布与真实分布不重叠时梯度为 0（不可微）。WGAN 用 Wasserstein 距离（Earth Mover 距离）：

$$W_1(p_r,p_g) = \sup_{\|f\|_L\le1}E_{x\sim p_r}[f(x)]-E_{x\sim p_g}[f(x)]$$

Wasserstein 距离在分布不重叠时仍提供有意义的梯度信号，避免梯度消失，从而缓解模式坍塌（生成器对所有模式保持梯度）。

**答案**：$\boxed{r_{ik}=\dfrac{\pi_k N(\mathbf{x}_i;\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}{\sum_j\pi_j N(\mathbf{x}_i;\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)}}$；EM 单调不降由 Jensen 不等式保证；WGAN 用 Wasserstein 距离缓解模式坍塌。
