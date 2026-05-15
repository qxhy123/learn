# 公式速查表（A 扩展版）

> 覆盖全 24 章核心公式，按 Part 1–8 组织。每节用 markdown 表格汇总公式 + 简要说明。

---

## Part 1 概率基础（Ch.1–3）

### 1.1 概率公理 & 事件运算

| 公式 | 说明 |
|---|---|
| $0 \leq P(A) \leq 1$ | 概率范围 |
| $P(\Omega) = 1$ | 必然事件 |
| $P(\emptyset) = 0$ | 不可能事件 |
| $P(A^c) = 1 - P(A)$ | 补事件 |
| $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ | 加法公式 |
| $P(A \cup B \cup C) = P(A)+P(B)+P(C)-P(A\cap B)-P(B\cap C)-P(A\cap C)+P(A\cap B\cap C)$ | 容斥原理三项 |
| $P\!\left(\bigcup_{i=1}^n A_i\right) = \sum_i P(A_i) - \sum_{i<j} P(A_i\cap A_j)+\cdots+(-1)^{n+1}P(A_1\cap\cdots\cap A_n)$ | 一般容斥原理 |
| $A \subseteq B \Rightarrow P(A) \leq P(B)$ | 单调性 |
| $P(B \setminus A) = P(B) - P(A \cap B)$ | 差事件 |
| 若 $A_i$ 两两互斥，$P\!\left(\bigcup A_i\right)=\sum P(A_i)$ | 可列可加性 |

### 1.2 条件概率 & 贝叶斯

| 公式 | 说明 |
|---|---|
| $P(A \mid B) = \dfrac{P(A \cap B)}{P(B)},\; P(B)>0$ | 条件概率定义 |
| $P(A \cap B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)$ | 乘法公式 |
| $P(A_1 \cap \cdots \cap A_n) = P(A_1)\,P(A_2\mid A_1)\cdots P(A_n\mid A_1\cdots A_{n-1})$ | 链式乘法 |
| $P(A) = \sum_{i=1}^n P(A \mid B_i)\,P(B_i)$，$\{B_i\}$ 完备 | 全概率公式 |
| $P(B_j \mid A) = \dfrac{P(A \mid B_j)\,P(B_j)}{\sum_{i=1}^n P(A \mid B_i)\,P(B_i)}$ | 贝叶斯公式 |
| $P(A \cap B) = P(A)\,P(B)$ | 事件独立 |
| $A,B,C$ 独立需六条等式同时成立 | 多事件独立条件 |

### 1.3 古典概型 & 几何概型 & 排列组合

| 公式 | 说明 |
|---|---|
| $P(A) = \dfrac{\text{有利基本事件数}}{\text{总基本事件数}}$ | 古典概型 |
| $P(A) = \dfrac{A \text{ 所占测度}}{\Omega \text{ 总测度}}$ | 几何概型 |
| $n! = n(n-1)\cdots 1,\quad 0!=1$ | 阶乘 |
| $A_n^k = P_n^k = \dfrac{n!}{(n-k)!}$ | 排列数 |
| $\dbinom{n}{k} = C_n^k = \dfrac{n!}{k!(n-k)!}$ | 组合数 |
| $(a+b)^n = \sum_{k=0}^n \dbinom{n}{k} a^k b^{n-k}$ | 二项式定理 |
| $\dbinom{n}{k} = \dbinom{n-1}{k-1} + \dbinom{n-1}{k}$ | 帕斯卡恒等式 |
| $n$ 元集合的子集总数 $= 2^n$ | 子集计数 |
| 有放回取 $k$ 个：$n^k$ 种；不放回取 $k$ 个：$A_n^k$ 种 | 有序采样 |

---

## Part 2 随机变量（Ch.4–6）

### 2.1 离散随机变量 PMF / CDF / 期望 / 方差

| 公式 | 说明 |
|---|---|
| $P(X = x_i) = p_i \geq 0$，$\sum_i p_i = 1$ | PMF 定义 |
| $F(x) = P(X \leq x) = \sum_{x_i \leq x} p_i$ | CDF（右连续阶梯函数） |
| $E[X] = \sum_i x_i p_i$ | 期望 |
| $E[g(X)] = \sum_i g(x_i)\,p_i$ | 函数期望（LOTUS） |
| $\operatorname{Var}(X) = E[(X-\mu)^2] = E[X^2] - (E[X])^2$ | 方差 |
| $\sigma = \sqrt{\operatorname{Var}(X)}$ | 标准差 |

### 2.2 连续随机变量 PDF / CDF / 期望 / 方差

| 公式 | 说明 |
|---|---|
| $f(x) \geq 0$，$\int_{-\infty}^{+\infty} f(x)\,dx = 1$ | PDF 性质 |
| $P(a \leq X \leq b) = \int_a^b f(x)\,dx$ | 区间概率 |
| $P(X = a) = 0$（连续型） | 单点概率为零 |
| $F(x) = \int_{-\infty}^x f(t)\,dt$ | PDF → CDF |
| $f(x) = F'(x)$（可微处） | CDF → PDF |
| $F(-\infty)=0$，$F(+\infty)=1$，$F$ 单调不减右连续 | CDF 性质 |
| $E[X] = \int_{-\infty}^{+\infty} x\,f(x)\,dx$ | 连续型期望 |
| $E[g(X)] = \int_{-\infty}^{+\infty} g(x)\,f(x)\,dx$ | LOTUS（连续） |
| $\operatorname{Var}(X) = \int_{-\infty}^{+\infty}(x-\mu)^2 f(x)\,dx = E[X^2]-\mu^2$ | 连续型方差 |

### 2.3 期望 & 方差的性质

| 公式 | 说明 |
|---|---|
| $E[aX+b] = aE[X]+b$ | 期望线性性 |
| $E[X+Y] = E[X]+E[Y]$（无需独立） | 可加性 |
| $E[XY] = E[X]E[Y]$（$X,Y$ 独立） | 独立乘积期望 |
| $\operatorname{Var}(aX+b) = a^2\operatorname{Var}(X)$ | 方差平移不变 |
| $\operatorname{Var}(X+Y) = \operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y)$ | 方差可加 |
| $\operatorname{Var}(X+Y) = \operatorname{Var}(X)+\operatorname{Var}(Y)$（独立时） | 独立方差可加 |
| $E[X] = \sum_{k=1}^\infty P(X \geq k)$（非负整数型） | 期望尾求和 |
| $E[X] = \int_0^\infty P(X > t)\,dt$（非负连续型） | 期望尾积分 |

### 2.4 协方差 & 相关系数

| 公式 | 说明 |
|---|---|
| $\operatorname{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)]$ | 协方差定义 |
| $\operatorname{Cov}(X,Y) = E[XY] - E[X]E[Y]$ | 计算公式 |
| $\operatorname{Cov}(X,X) = \operatorname{Var}(X)$ | 自协方差 |
| $\operatorname{Cov}(aX+b,\,cY+d) = ac\,\operatorname{Cov}(X,Y)$ | 线性变换 |
| $\operatorname{Cov}(X+Y,Z) = \operatorname{Cov}(X,Z)+\operatorname{Cov}(Y,Z)$ | 双线性 |
| $\rho_{XY} = \dfrac{\operatorname{Cov}(X,Y)}{\sigma_X\,\sigma_Y}$ | 相关系数 |
| $-1 \leq \rho_{XY} \leq 1$ | 柯西-施瓦茨 |
| $X,Y$ 独立 $\Rightarrow \rho=0$；反之不成立 | 独立 vs 不相关 |

> ⚠️ 提示：不相关（$\rho=0$）不等于独立，仅当联合正态时等价。

### 2.5 多维联合 / 边缘 / 条件分布

| 公式 | 说明 |
|---|---|
| $f_{X,Y}(x,y)$：$f\geq0$，$\iint f\,dx\,dy=1$ | 联合 PDF |
| $f_X(x) = \int_{-\infty}^{+\infty} f_{X,Y}(x,y)\,dy$ | 边缘 PDF（对 $y$ 积分） |
| $f_{Y\mid X}(y\mid x) = \dfrac{f_{X,Y}(x,y)}{f_X(x)}$ | 条件 PDF |
| $X \perp\!\!\!\perp Y \Leftrightarrow f_{X,Y}(x,y) = f_X(x)\,f_Y(y)$ | 独立等价条件 |
| $f_{Y\mid X}(y\mid x) = f_Y(y)$（独立时） | 独立条件分布等于边缘 |
| $E[Y\mid X=x] = \int y\,f_{Y\mid X}(y\mid x)\,dy$ | 条件期望 |
| $E[Y] = E[E[Y\mid X]]$ | 全期望公式（迭代期望） |
| $\operatorname{Var}(Y) = E[\operatorname{Var}(Y\mid X)] + \operatorname{Var}(E[Y\mid X])$ | 全方差公式 |

### 2.6 变量变换

| 情景 | 公式 |
|---|---|
| $Y=g(X)$，$g$ 严格单调可微 | $f_Y(y) = f_X(g^{-1}(y))\cdot\vert (g^{-1})'(y)\vert$ |
| $Y=g(X)$，$g$ 非单调，各段求和 | $f_Y(y) = \sum_k f_X(x_k)\cdot\vert (g^{-1})'(y)\vert$，$x_k=g_k^{-1}(y)$ |
| $Z=X+Y$（独立） | $f_Z(z) = \int f_X(x)\,f_Y(z-x)\,dx$（卷积） |
| $W=X/Y$（独立，$Y>0$） | $f_W(w)=\int_0^\infty y\,f_X(wy)\,f_Y(y)\,dy$ |
| 二维变换 $(U,V) = g(X,Y)$ | $f_{U,V}(u,v) = f_{X,Y}(x,y)\cdot\vert J\vert^{-1}$，$J = \partial(x,y)/\partial(u,v)$ |

### 2.7 特征函数（CF）

| 公式 | 说明 |
|---|---|
| $\varphi_X(t)=E[e^{itX}]$，$t\in\mathbb{R}$ | 特征函数定义 |
| $\vert\varphi_X(t)\vert\leq1$，$\varphi_X(0)=1$ | 有界性 |
| $X,Y$ 独立：$\varphi_{X+Y}(t)=\varphi_X(t)\,\varphi_Y(t)$ | 独立求和 |
| $E[X^k]=i^{-k}\varphi_X^{(k)}(0)$ | 矩提取 |
| 唯一性：CF 唯一确定分布 | 存在性不需要限制区间 |
| 连续性定理：$\varphi_{X_n}(t)\to\varphi_X(t)$ $\Leftrightarrow$ $X_n\xrightarrow{d}X$ | 依分布收敛等价于 CF 逐点收敛 |
| 标准正态：$\varphi(t)=e^{-t^2/2}$ | 正态 CF |
| Poisson$(\lambda)$：$\varphi(t)=e^{\lambda(e^{it}-1)}$ | Poisson CF |

---

## Part 3 分布大全（Ch.7–9）

### 3.1 常见离散分布

| 分布 | PMF | 期望 | 方差 | MGF |
|---|---|---|---|---|
| Bernoulli$(p)$ | $P(X=1)=p$ | $p$ | $p(1-p)$ | $1-p+pe^t$ |
| Binomial$(n,p)$ | $\dbinom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | $(1-p+pe^t)^n$ |
| Geometric$(p)$ | $(1-p)^{k-1}p$，$k\geq1$ | $\dfrac{1}{p}$ | $\dfrac{1-p}{p^2}$ | $\dfrac{pe^t}{1-(1-p)e^t}$ |
| NegBinom$(r,p)$ | $\dbinom{k-1}{r-1}p^r(1-p)^{k-r}$ | $\dfrac{r}{p}$ | $\dfrac{r(1-p)}{p^2}$ | $\left(\dfrac{pe^t}{1-(1-p)e^t}\right)^r$ |
| Poisson$(\lambda)$ | $\dfrac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | $e^{\lambda(e^t-1)}$ |
| HyperGeom$(N,K,n)$ | $\dfrac{\dbinom{K}{k}\dbinom{N-K}{n-k}}{\dbinom{N}{n}}$ | $\dfrac{nK}{N}$ | $\dfrac{nK(N-K)(N-n)}{N^2(N-1)}$ | — |

> **定理**：二项$(n,p)$：$n$ 大 $p$ 小时 $\approx$ Poisson$(np)$；Poisson 之和仍为 Poisson：$X_1+X_2\sim\text{Pois}(\lambda_1+\lambda_2)$（独立）。几何$(p)$：$\min(X_1,X_2)\sim\text{Geom}(1-(1-p_1)(1-p_2))$（独立）。

> ⚠️ 提示：几何分布有两种参数化：$P(X=k)=(1-p)^{k-1}p$（$k\geq1$，首次成功试验次数）或 $P(X=k)=(1-p)^k p$（$k\geq0$，失败次数）。两者期望相差 1，注意区分。

### 3.2 常见连续分布

| 分布 | PDF（支撑域） | 期望 | 方差 | MGF |
|---|---|---|---|---|
| Uniform$(a,b)$ | $\dfrac{1}{b-a}$，$x\in[a,b]$ | $\dfrac{a+b}{2}$ | $\dfrac{(b-a)^2}{12}$ | $\dfrac{e^{tb}-e^{ta}}{t(b-a)}$ |
| Normal$(\mu,\sigma^2)$ | $\dfrac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | $e^{\mu t+\sigma^2 t^2/2}$ |
| Exp$(\lambda)$ | $\lambda e^{-\lambda x}$，$x\geq0$ | $\dfrac{1}{\lambda}$ | $\dfrac{1}{\lambda^2}$ | $\dfrac{\lambda}{\lambda-t}$，$t<\lambda$ |
| Gamma$(\alpha,\beta)$ | $\dfrac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$，$x>0$ | $\dfrac{\alpha}{\beta}$ | $\dfrac{\alpha}{\beta^2}$ | $\left(\dfrac{\beta}{\beta-t}\right)^\alpha$ |
| Beta$(\alpha,\beta)$ | $\dfrac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$，$x\in(0,1)$ | $\dfrac{\alpha}{\alpha+\beta}$ | $\dfrac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ | — |
| $\chi^2(n)$ | Gamma$(n/2,1/2)$ | $n$ | $2n$ | $(1-2t)^{-n/2}$ |
| $t(n)$ | $\dfrac{\Gamma(\frac{n+1}{2})}{\sqrt{n\pi}\,\Gamma(\frac{n}{2})}\!\left(1+\dfrac{x^2}{n}\right)^{-\frac{n+1}{2}}$ | $0$（$n>1$） | $\dfrac{n}{n-2}$（$n>2$） | — |
| $F(m,n)$ | $\dfrac{V_1/m}{V_2/n}$，$V_i\sim\chi^2$ | $\dfrac{n}{n-2}$（$n>2$） | 复杂 | — |

> **定理（再生性）**：$X_i\sim\mathcal{N}(\mu_i,\sigma_i^2)$ 独立，$\sum a_i X_i \sim \mathcal{N}(\sum a_i\mu_i,\sum a_i^2\sigma_i^2)$。Gamma$(\alpha_1,\beta)$、Gamma$(\alpha_2,\beta)$ 独立之和：Gamma$(\alpha_1+\alpha_2,\beta)$。指数$(λ)$具有无记忆性：$P(X>s+t\mid X>s)=P(X>t)$。

正态分布标准化：若 $X\sim\mathcal{N}(\mu,\sigma^2)$，则 $Z=(X-\mu)/\sigma\sim\mathcal{N}(0,1)$，$P(a<X<b)=\Phi\!\left(\dfrac{b-\mu}{\sigma}\right)-\Phi\!\left(\dfrac{a-\mu}{\sigma}\right)$。

### 3.2b 对数正态与 Pareto

| 分布 | 参数 / PDF | 期望 | 方差 |
|---|---|---|---|
| 对数正态 $\mathrm{LogN}(\mu,\sigma^2)$ | $\ln X\sim\mathcal{N}(\mu,\sigma^2)$ | $e^{\mu+\sigma^2/2}$ | $(e^{\sigma^2}-1)e^{2\mu+\sigma^2}$ |
| Pareto$(x_m,\alpha)$ | $f(x)=\dfrac{\alpha x_m^\alpha}{x^{\alpha+1}}$，$x\geq x_m$ | $\dfrac{\alpha x_m}{\alpha-1}$（$\alpha>1$） | $\dfrac{\alpha x_m^2}{(\alpha-1)^2(\alpha-2)}$（$\alpha>2$） |
| Weibull$(\lambda,k)$ | $f(x)=\dfrac{k}{\lambda}\!\left(\dfrac{x}{\lambda}\right)^{k-1}e^{-(x/\lambda)^k}$ | $\lambda\,\Gamma(1+1/k)$ | $\lambda^2\!\left[\Gamma(1+2/k)-\Gamma(1+1/k)^2\right]$ |

### 3.3 三大抽样分布（正态总体）

| 统计量 | 分布 | 条件 |
|---|---|---|
| $Z_i\overset{iid}{\sim}\mathcal{N}(0,1)$，$\chi^2=\sum_{i=1}^n Z_i^2$ | $\chi^2(n)$ | 独立标准正态平方和 |
| $t = Z / \sqrt{V/n}$，$Z\sim\mathcal{N}(0,1)$，$V\sim\chi^2(n)$，独立 | $t(n)$ | Z 与卡方独立 |
| $F = (V_1/m)/(V_2/n)$，$V_i\sim\chi^2$ 独立 | $F(m,n)$ | 两独立卡方之比 |
| $F(1,n) = [t(n)]^2$ | 关系 | — |
| $1/F(m,n) \sim F(n,m)$ | 倒数关系 | — |

设 $X_1,\ldots,X_n\overset{iid}{\sim}\mathcal{N}(\mu,\sigma^2)$，$\bar X = \frac{1}{n}\sum X_i$，$S^2 = \frac{1}{n-1}\sum(X_i-\bar X)^2$：

| 统计量 | 精确分布 |
|---|---|
| $\bar X$ | $\mathcal{N}\!\left(\mu,\dfrac{\sigma^2}{n}\right)$ |
| $\dfrac{(n-1)S^2}{\sigma^2}$ | $\chi^2(n-1)$ |
| $\dfrac{\bar X - \mu}{S/\sqrt{n}}$ | $t(n-1)$ |
| $\bar X$ 与 $S^2$ | 相互独立 |

### 3.4 矩母函数（MGF）性质

| 性质 | 公式 |
|---|---|
| 定义 | $M_X(t) = E[e^{tX}]$ |
| 矩提取 | $M_X^{(k)}(0) = E[X^k]$ |
| 独立求和 | $M_{X+Y}(t) = M_X(t)\,M_Y(t)$（$X,Y$ 独立） |
| 线性变换 | $M_{aX+b}(t) = e^{bt}M_X(at)$ |
| 唯一性定理 | MGF 在某邻域存在则唯一确定分布 |
| 正态 MGF 证明 CLT | $M_{Z_n}(t)=\left[1+\frac{t^2}{2n}+o(1/n)\right]^n\to e^{t^2/2}$ |

### 3.4b 分位数与百分位数

| 公式 | 说明 |
|---|---|
| $p$ 分位数：$F(x_p)=p$，即 $x_p=F^{-1}(p)$ | 分位数定义 |
| 中位数：$x_{0.5}$，$P(X\leq x_{0.5})\geq0.5$ | 中位数 |
| $\mathcal{N}(\mu,\sigma^2)$：$x_p=\mu+\sigma\,z_p$ | 正态分位数 |
| $\text{Exp}(\lambda)$：$x_p=-\ln(1-p)/\lambda$ | 指数分位数 |
| $\chi^2(n)$，$t(n)$，$F(m,n)$ | 查统计表或软件计算 |

### 3.5 特殊函数

| 公式 | 说明 |
|---|---|
| $\Gamma(\alpha) = \int_0^\infty x^{\alpha-1}e^{-x}\,dx$ | Gamma 函数定义 |
| $\Gamma(n) = (n-1)!$（正整数） | 阶乘联系 |
| $\Gamma(1/2) = \sqrt{\pi}$ | 半整数值 |
| $\Gamma(\alpha+1) = \alpha\,\Gamma(\alpha)$ | 递推关系 |
| $B(\alpha,\beta) = \dfrac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ | Beta 函数 |
| $\int_{-\infty}^{+\infty}e^{-x^2}\,dx = \sqrt{\pi}$ | 高斯积分 |
| $\Phi(x)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^x e^{-t^2/2}\,dt$，$\Phi(-x)=1-\Phi(x)$ | 标准正态 CDF |

### 3.6 多元分布

| 分布 | 公式 |
|---|---|
| 多项分布 $\text{Multi}(n;\mathbf{p})$ | $P(X_1=k_1,\ldots,X_m=k_m)=\dfrac{n!}{k_1!\cdots k_m!}p_1^{k_1}\cdots p_m^{k_m}$ |
| 多元正态 $\mathcal{N}(\boldsymbol\mu,\boldsymbol\Sigma)$ | $f(\mathbf{x})=\dfrac{1}{(2\pi)^{n/2}\vert\boldsymbol\Sigma\vert^{1/2}}\exp\!\left(-\dfrac{1}{2}(\mathbf{x}-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\right)$ |
| Dirichlet$(\boldsymbol\alpha)$ | $f(\mathbf{x})=\dfrac{\Gamma(\sum_i\alpha_i)}{\prod_i\Gamma(\alpha_i)}\prod_i x_i^{\alpha_i-1}$ |

---

## Part 4 极限定理（Ch.10–12）

### 4.1 概率不等式

| 名称 | 公式 | 条件 |
|---|---|---|
| 马尔可夫不等式 | $P(\vert X\vert \geq a) \leq \dfrac{E[\vert X\vert]}{a}$ | $a>0$，$E[\vert X\vert]<\infty$ |
| 切比雪夫不等式 | $P(\vert X-\mu\vert \geq k\sigma) \leq \dfrac{1}{k^2}$ | 方差存在 |
| 切比雪夫（$t$ 形式） | $P(\vert X-\mu\vert \geq t) \leq \dfrac{\sigma^2}{t^2}$ | — |
| Hoeffding 单变量 | $P(X-E[X]\geq t)\leq e^{-2t^2/(b-a)^2}$ | $X\in[a,b]$ a.s. |
| Hoeffding 求和 | $P\!\left(\bar X_n - \mu \geq t\right)\leq\exp\!\left(-\dfrac{2n^2t^2}{\sum_i(b_i-a_i)^2}\right)$ | $X_i\in[a_i,b_i]$ |
| Jensen 不等式 | $f(E[X])\leq E[f(X)]$（$f$ 凸） | — |
| Cauchy-Schwarz | $(E[XY])^2 \leq E[X^2]\,E[Y^2]$ | — |

> **定理（马尔可夫的推广）**：对任意 $r>0$，$P(\vert X\vert \geq a)\leq E[\vert X\vert^r]/a^r$（矩不等式）。

### 4.2 大数定律

| 名称 | 结论 | 条件 |
|---|---|---|
| 弱大数定律（WLLN，Chebyshev） | $\bar X_n \xrightarrow{P} \mu$ | $E[X_i]=\mu$，$\operatorname{Var}(X_i)\leq C$ |
| 弱大数定律（Khintchine） | $\bar X_n \xrightarrow{P} \mu$ | i.i.d.，$E[\vert X\vert]<\infty$ |
| 强大数定律（SLLN，Kolmogorov） | $\bar X_n \xrightarrow{a.s.} \mu$ | i.i.d.，$E[\vert X\vert]<\infty$ |

$$\bar X_n = \frac{1}{n}\sum_{i=1}^n X_i$$

> ⚠️ 提示：a.s. 收敛 $\Rightarrow$ 依概率收敛，反之不成立。

### 4.3 中心极限定理

| 版本 | 条件 | 结论 |
|---|---|---|
| Lindeberg-Lévy（经典 CLT） | i.i.d.，$\mu,\sigma^2<\infty$ | $\dfrac{\bar X_n-\mu}{\sigma/\sqrt{n}}\xrightarrow{d}\mathcal{N}(0,1)$ |
| De Moivre-Laplace | $X\sim B(n,p)$ | $\dfrac{X-np}{\sqrt{np(1-p)}}\xrightarrow{d}\mathcal{N}(0,1)$ |
| Lindeberg（一般 CLT） | 独立非同分布，Lindeberg 条件 | $\dfrac{S_n-ES_n}{\sqrt{\operatorname{Var}(S_n)}}\xrightarrow{d}\mathcal{N}(0,1)$ |
| Lyapunov 条件 | $\exists\delta>0$：$L_n(\delta)\to0$ | 蕴含 Lindeberg 条件 |
| Berry-Esseen 误差上界 | i.i.d.，$E[\vert X\vert^3]<\infty$ | $\sup_x\vert F_n(x)-\Phi(x)\vert \leq \dfrac{C\,E[\vert X-\mu\vert^3]}{\sigma^3\sqrt{n}}$ |

等价写法：$\sqrt{n}(\bar X_n-\mu)\xrightarrow{d}\mathcal{N}(0,\sigma^2)$；连续性修正：$B(n,p)\approx\mathcal{N}(np,np(1-p))$，整数 $k$ 对应区间 $[k-0.5,k+0.5]$。

CLT 应用示意：

| 应用场景 | 近似公式 |
|---|---|
| 样本均值置信区间（$\sigma$ 已知） | $\bar X\pm z_{\alpha/2}\sigma/\sqrt{n}$ |
| 样本均值置信区间（$\sigma$ 未知，大样本） | $\bar X\pm z_{\alpha/2}S/\sqrt{n}$（$S$ 代替 $\sigma$） |
| 比例估计 $\hat p=X/n$，$X\sim B(n,p)$ | $(\hat p-p)/\sqrt{p(1-p)/n}\xrightarrow{d}\mathcal{N}(0,1)$ |
| 泊松近似正态：$X\sim\text{Pois}(\lambda)$，$\lambda$ 大 | $(X-\lambda)/\sqrt\lambda\xrightarrow{d}\mathcal{N}(0,1)$ |

### 4.4 四种收敛类型

| 类型 | 符号 | 定义 |
|---|---|---|
| 几乎必然收敛 | $X_n\xrightarrow{a.s.}X$ | $P(\lim_{n\to\infty}X_n=X)=1$ |
| 依概率收敛 | $X_n\xrightarrow{P}X$ | $\forall\varepsilon>0$：$P(\vert X_n-X\vert>\varepsilon)\to0$ |
| $L^p$ 收敛 | $X_n\xrightarrow{L^p}X$ | $E[\vert X_n-X\vert^p]\to0$ |
| 依分布收敛 | $X_n\xrightarrow{d}X$ | $F_n(x)\to F(x)$ 在 $F$ 连续点处 |

收敛强度：a.s. $\Rightarrow$ $P$ $\Rightarrow$ $d$；$L^p$ $\Rightarrow$ $L^q$（$p>q$）$\Rightarrow$ $P$ $\Rightarrow$ $d$。

> ⚠️ 提示：依分布收敛到常数 $c$（即 $P(X=c)=1$ 的退化分布）等价于依概率收敛到 $c$。这是 Slutsky 定理成立的关键原因。

### 4.4b 随机变量序列的其他性质

| 性质 | 说明 |
|---|---|
| Borel-Cantelli 引理 I | $\sum_n P(A_n)<\infty$ $\Rightarrow$ $P(\limsup A_n)=0$（a.s. 有限次发生） |
| Borel-Cantelli 引理 II | $A_n$ 独立，$\sum_n P(A_n)=\infty$ $\Rightarrow$ $P(\limsup A_n)=1$（a.s. 无限次发生） |
| 一致可积（UI） | $\sup_n E[\vert X_n\vert\,\mathbf{1}(\vert X_n\vert>M)]\to0$（$M\to\infty$） |
| UI + 依分布收敛 $\Rightarrow$ $L^1$ 收敛 | 连接分布收敛与期望收敛 |

### 4.5 Slutsky 定理 & 连续映射定理 & Delta 方法

| 定理 | 结论 |
|---|---|
| **Slutsky**：$X_n\xrightarrow{d}X$，$Y_n\xrightarrow{P}c$（常数） | $X_n+Y_n\xrightarrow{d}X+c$；$X_n Y_n\xrightarrow{d}cX$ |
| **CMT（连续映射定理）**：$g$ 连续 | $X_n\xrightarrow{d}X\Rightarrow g(X_n)\xrightarrow{d}g(X)$；$P$ / a.s. 类似 |
| **Delta 方法**：$\sqrt{n}(\hat\theta_n-\theta)\xrightarrow{d}\mathcal{N}(0,\sigma^2)$，$g'(\theta)\neq0$ | $\sqrt{n}(g(\hat\theta_n)-g(\theta))\xrightarrow{d}\mathcal{N}(0,\sigma^2[g'(\theta)]^2)$ |

---

## Part 5 统计基础（Ch.13–15）

### 5.0 参数空间与统计模型

| 概念 | 定义 |
|---|---|
| 统计模型 | $\mathcal{P}=\{P_\theta:\theta\in\Theta\}$，参数集 $\Theta$ |
| 可识别性 | $\theta_1\neq\theta_2\Rightarrow P_{\theta_1}\neq P_{\theta_2}$ |
| 参数统计 vs 非参数统计 | $\Theta\subset\mathbb{R}^k$ 有限维 vs $\mathcal{P}$ 为所有连续分布族 |
| 指数族 | $f(x;\theta)=h(x)\exp[\eta(\theta)^\top T(x)-A(\theta)]$，$A(\theta)=\log Z(\theta)$ |
| 自然参数 | $\eta=\eta(\theta)$（可以是 $\theta$ 本身，即自然参数化） |
| 充分统计量（指数族） | $T(x)$ 是自然充分统计量 |
| 均值参数 | $\mu=E_\theta[T(X)]=\nabla A(\eta)$ |

### 5.1 抽样统计量 & 抽样分布

| 统计量 | 定义 / 公式 |
|---|---|
| 样本均值 | $\bar X = \dfrac{1}{n}\sum_{i=1}^n X_i$ |
| 样本方差 | $S^2 = \dfrac{1}{n-1}\sum_{i=1}^n(X_i-\bar X)^2$（无偏） |
| 样本标准差 | $S = \sqrt{S^2}$ |
| $k$ 阶样本矩 | $A_k = \dfrac{1}{n}\sum_{i=1}^n X_i^k$ |
| $k$ 阶中心样本矩 | $B_k = \dfrac{1}{n}\sum_{i=1}^n(X_i-\bar X)^k$ |
| 顺序统计量 | $X_{(1)}\leq X_{(2)}\leq\cdots\leq X_{(n)}$ |
| 最小值 PDF | $f_{X_{(1)}}(x)=n\,[1-F(x)]^{n-1}f(x)$ |
| 最大值 PDF | $f_{X_{(n)}}(x)=n\,[F(x)]^{n-1}f(x)$ |
| 经验分布 | $\hat F_n(x)=\dfrac{1}{n}\sum_{i=1}^n\mathbf{1}(X_i\leq x)$ |

### 5.1b 两正态总体参数区间估计

设 $X_i\overset{iid}{\sim}\mathcal{N}(\mu_1,\sigma_1^2)$，$Y_j\overset{iid}{\sim}\mathcal{N}(\mu_2,\sigma_2^2)$，独立：

| 参数 | 条件 | 枢轴统计量 | 置信区间 |
|---|---|---|---|
| $\mu_1-\mu_2$，$\sigma_1^2,\sigma_2^2$ 已知 | — | $z=\dfrac{(\bar X-\bar Y)-(\mu_1-\mu_2)}{\sqrt{\sigma_1^2/n_1+\sigma_2^2/n_2}}$ | $\bar X-\bar Y\pm z_{\alpha/2}\sqrt{\sigma_1^2/n_1+\sigma_2^2/n_2}$ |
| $\mu_1-\mu_2$，$\sigma_1^2=\sigma_2^2$ 未知 | 方差齐 | $t\sim t(n_1+n_2-2)$ | $\bar X-\bar Y\pm t_{\alpha/2}S_p\sqrt{1/n_1+1/n_2}$ |
| $\sigma_1^2/\sigma_2^2$ | — | $F=S_1^2/S_2^2\sim F(n_1-1,n_2-1)$ | $\left[S_1^2/(S_2^2 F_{1-\alpha/2}),\,S_1^2/(S_2^2 F_{\alpha/2})\right]$ |

### 5.2 Bootstrap

| 公式 | 说明 |
|---|---|
| 重采样：有放回从 $\{X_1,\ldots,X_n\}$ 抽 $n$ 个 | Bootstrap 样本 |
| $\hat\sigma_B^2 = \dfrac{1}{B-1}\sum_{b=1}^B(\hat\theta^*_b-\bar\theta^*)^2$ | Bootstrap 方差估计 |
| Bootstrap 置信区间（百分位法）：$[\hat\theta^*_{(\alpha/2)},\hat\theta^*_{(1-\alpha/2)}]$ | 基于 Bootstrap 分位数 |

### 5.3 描述统计

| 度量 | 公式 |
|---|---|
| 样本中位数 | $X_{(n/2)}$（$n$ 偶数取均值） |
| 四分位距 | $\text{IQR} = Q_3 - Q_1$ |
| 偏度 | $\text{Skew} = \dfrac{E[(X-\mu)^3]}{\sigma^3}$ |
| 峰度 | $\text{Kurt} = \dfrac{E[(X-\mu)^4]}{\sigma^4} - 3$（超额峰度） |
| 样本相关系数 | $r = \dfrac{\sum(X_i-\bar X)(Y_i-\bar Y)}{\sqrt{\sum(X_i-\bar X)^2\sum(Y_i-\bar Y)^2}}$ |

### 5.4 充分统计量 & 因子分解定理

| 概念 | 公式 / 说明 |
|---|---|
| 充分统计量定义 | $T(X)$ 充分 $\Leftrightarrow$ $p(\mathbf{x}\mid\theta)$ 中 $\theta$ 只通过 $T(\mathbf{x})$ 出现 |
| **因子分解定理**（Neyman-Fisher） | $T$ 充分 $\Leftrightarrow$ $f(\mathbf{x};\theta)=g(T(\mathbf{x}),\theta)\cdot h(\mathbf{x})$ |
| 完备统计量 | $E_\theta[g(T)]=0$，$\forall\theta$ $\Rightarrow$ $g(T)=0$ a.s. |
| 最小充分统计量 | 包含于所有充分统计量中（信息量最少的充分统计量） |
| 指数族形式 | $f(x;\theta)=h(x)\exp[\eta(\theta)T(x)-A(\theta)]$，$T(x)$ 是自然充分统计量 |

> **定理（Rao-Blackwell）**：若 $\tilde\theta$ 是 $\theta$ 的无偏估计，$T$ 是充分统计量，则 $\hat\theta=E[\tilde\theta\mid T]$ 满足 $\operatorname{Var}(\hat\theta)\leq\operatorname{Var}(\tilde\theta)$，且等号成立当且仅当 $\tilde\theta=\hat\theta$ a.s.。

> **定理（Lehmann-Scheffé）**：若 $T$ 完备充分，且 $h(T)$ 无偏，则 $h(T)$ 是唯一的 UMVUE（一致最小方差无偏估计量）。

常见指数族与充分统计量：

| 分布 | 自然充分统计量 $T(\mathbf{x})$ |
|---|---|
| 正态$(\mu,\sigma^2)$ | $(\sum x_i, \sum x_i^2)$ |
| 指数$(\lambda)$ | $\sum x_i$ |
| Bernoulli$(p)$ | $\sum x_i$（成功次数） |
| Poisson$(\lambda)$ | $\sum x_i$ |
| Gamma$(\alpha,\beta)$（$\alpha$ 已知） | $\sum x_i$ |

---

## Part 6 参数估计（Ch.16–18）

### 6.1 矩估计

| 公式 | 说明 |
|---|---|
| 令 $\bar X = E_\theta[X]$ | 一阶矩方程 |
| 令 $\dfrac{1}{n}\sum X_i^2 = E_\theta[X^2]$ | 二阶矩方程 |
| $p$ 个参数列 $p$ 个方程，联立求解 | 矩估计一般步骤 |
| 正态：$\hat\mu=\bar X$，$\hat\sigma^2=\dfrac{1}{n}\sum(X_i-\bar X)^2$ | 矩估计结果（$\sigma^2$ 有偏） |

### 6.1b 矩估计步骤示例

| 分布 | 矩方程 | 矩估计量 |
|---|---|---|
| 正态 $\mathcal{N}(\mu,\sigma^2)$ | $\mu=\bar X$，$\sigma^2+\mu^2=\overline{X^2}$ | $\hat\mu=\bar X$，$\hat\sigma^2=\overline{X^2}-\bar X^2$ |
| Gamma$(\alpha,\beta)$ | $\alpha/\beta=\bar X$，$\alpha/\beta^2=S^2/n\cdot(n-1)/n$ | $\hat\beta=\bar X/S^2$，$\hat\alpha=\bar X^2/S^2$ |
| Beta$(\alpha,\beta)$ | $\mu=\bar X$，$\sigma^2=\bar X(1-\bar X)/(m+1)$ | 解 $\hat\alpha,\hat\beta$ |
| Uniform$(a,b)$ | $(a+b)/2=\bar X$，$(b-a)^2/12=S^2$ | $\hat a=\bar X-\sqrt{3}S$，$\hat b=\bar X+\sqrt{3}S$ |

### 6.2 最大似然估计（MLE）

| 公式 | 说明 |
|---|---|
| $L(\theta)=\prod_{i=1}^n f(x_i;\theta)$ | 似然函数 |
| $\ell(\theta)=\sum_{i=1}^n\log f(x_i;\theta)$ | 对数似然 |
| $\hat\theta_{MLE}=\arg\max_\theta\ell(\theta)$ | MLE 定义 |
| 似然方程：$\dfrac{\partial\ell}{\partial\theta}=0$ | 求解条件（内点解） |
| **不变性**：$g(\hat\theta_{MLE})$ 是 $g(\theta)$ 的 MLE | MLE 不变性原理 |
| 渐近正态：$\sqrt{n}(\hat\theta_{MLE}-\theta)\xrightarrow{d}\mathcal{N}\!\left(0,\dfrac{1}{I(\theta)}\right)$ | 大样本性质 |

常见 MLE：

| 分布 | MLE |
|---|---|
| 正态$(\mu,\sigma^2)$ | $\hat\mu=\bar X$，$\hat\sigma^2=\dfrac{1}{n}\sum(X_i-\bar X)^2$ |
| 指数$(\lambda)$ | $\hat\lambda=1/\bar X$ |
| Bernoulli$(p)$ | $\hat p=\bar X$ |
| Poisson$(\lambda)$ | $\hat\lambda=\bar X$ |
| Uniform$(0,\theta)$ | $\hat\theta=X_{(n)}$（最大值） |

### 6.3 Fisher 信息量 & Cramér-Rao 下界

| 公式 | 说明 |
|---|---|
| $I(\theta)=E_\theta\!\left[\left(\dfrac{\partial\log f(X;\theta)}{\partial\theta}\right)^2\right]$ | Fisher 信息量（score 方差） |
| $I(\theta)=-E_\theta\!\left[\dfrac{\partial^2\log f(X;\theta)}{\partial\theta^2}\right]$ | 等价计算（正则条件下） |
| $I_n(\theta)=n\,I(\theta)$（i.i.d.样本） | 样本 Fisher 信息 |
| $\operatorname{Var}(\hat\theta)\geq\dfrac{1}{n\,I(\theta)}$（无偏） | **Cramér-Rao 下界** |
| $\operatorname{Var}(\hat\theta)\geq\dfrac{[g'(\theta)]^2}{n\,I(\theta)}$（估计 $g(\theta)$） | 一般形式 |

> ⚠️ 提示：CR 下界只对无偏估计成立；有偏估计可低于 CR 下界。若 MLE 的渐近方差达到 $1/(nI(\theta))$，称该 MLE 是渐近有效的（正则条件下恒成立）。

### 6.3b Fisher 信息矩阵（多参数）

| 公式 | 说明 |
|---|---|
| $\mathcal{I}(\boldsymbol\theta)_{jk}=E\!\left[\dfrac{\partial\log f}{\partial\theta_j}\dfrac{\partial\log f}{\partial\theta_k}\right]$ | Fisher 信息矩阵 |
| $\mathcal{I}(\boldsymbol\theta)_{jk}=-E\!\left[\dfrac{\partial^2\log f}{\partial\theta_j\partial\theta_k}\right]$ | 等价形式（正则条件） |
| CR 下界（多参数） | $\operatorname{Cov}(\hat{\boldsymbol\theta})\succeq [n\mathcal{I}(\boldsymbol\theta)]^{-1}$（矩阵不等号） |
| 参数变换 | $I_{g(\theta)}=(g'(\theta))^2/I(\theta)$ |

### 6.4 估计量评价标准

| 标准 | 定义 |
|---|---|
| 无偏性 | $E_\theta[\hat\theta]=\theta$ |
| 偏差 | $\text{Bias}(\hat\theta)=E[\hat\theta]-\theta$ |
| 均方误差 | $\text{MSE}(\hat\theta)=\operatorname{Var}(\hat\theta)+[\text{Bias}(\hat\theta)]^2$ |
| 有效性 | $\hat\theta_1$ 比 $\hat\theta_2$ 有效：$\operatorname{Var}(\hat\theta_1)\leq\operatorname{Var}(\hat\theta_2)$（同为无偏） |
| 一致性 | $\hat\theta_n\xrightarrow{P}\theta$（$n\to\infty$） |
| UMVUE | 最小方差无偏估计量 |

### 6.5 区间估计

| 参数 | 条件 | $100(1-\alpha)\%$ 置信区间 |
|---|---|---|
| $\mu$，$\sigma^2$ 已知 | 正态总体 | $\bar X\pm z_{\alpha/2}\dfrac{\sigma}{\sqrt{n}}$ |
| $\mu$，$\sigma^2$ 未知 | 正态总体 | $\bar X\pm t_{\alpha/2}(n-1)\dfrac{S}{\sqrt{n}}$ |
| $\sigma^2$ | 正态总体 | $\left[\dfrac{(n-1)S^2}{\chi^2_{\alpha/2}(n-1)},\dfrac{(n-1)S^2}{\chi^2_{1-\alpha/2}(n-1)}\right]$ |
| 比例 $p$（大样本） | $n\hat p\geq5$，$n(1-\hat p)\geq5$ | $\hat p\pm z_{\alpha/2}\sqrt{\dfrac{\hat p(1-\hat p)}{n}}$ |
| 比例 $p$（Wilson 区间） | 小样本亦适用 | $\dfrac{\hat p+z^2/(2n)\pm z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}}{1+z^2/n}$，$z=z_{\alpha/2}$ |

### 6.6 贝叶斯估计

| 公式 | 说明 |
|---|---|
| $p(\theta\mid\mathbf{x})\propto p(\mathbf{x}\mid\theta)\,p(\theta)$ | 后验正比于似然×先验 |
| **MAP**：$\hat\theta_{MAP}=\arg\max_\theta p(\theta\mid\mathbf{x})$ | 后验众数 |
| **MMSE**（后验均值）：$\hat\theta_{MMSE}=E[\theta\mid\mathbf{x}]$ | 最小均方误差 |
| **MAD**（后验中位数）：$\hat\theta_{MAD}=\text{median}(\theta\mid\mathbf{x})$ | 最小绝对偏差 |

### 6.7 常见共轭先验

| 似然 | 共轭先验 | 后验 | 更新规则 |
|---|---|---|---|
| Bernoulli/Binomial | Beta$(\alpha,\beta)$ | Beta$(\alpha+k,\,\beta+n-k)$ | $k$：成功次数 |
| Poisson$(\lambda)$ | Gamma$(\alpha,\beta)$ | Gamma$(\alpha+\sum x_i,\,\beta+n)$ | — |
| Normal（$\sigma^2$ 已知） | Normal$(\mu_0,\tau^2)$ | Normal$\!\left(\tilde\mu,\tilde\tau^2\right)$ | 精度加权均值 |
| Normal（$\mu$ 已知） | Inv-Gamma$(\alpha,\beta)$ | Inv-Gamma$\!\left(\alpha+n/2,\,\beta+\tfrac{1}{2}\sum(x_i-\mu)^2\right)$ | — |
| Multinomial | Dirichlet$(\boldsymbol\alpha)$ | Dirichlet$(\boldsymbol\alpha+\mathbf{x})$ | — |
| Exponential$(\lambda)$ | Gamma$(\alpha,\beta)$ | Gamma$(\alpha+n,\,\beta+\sum x_i)$ | — |

正态均值后验参数（$\sigma^2$ 已知）：

$$\tilde\tau^2 = \frac{1}{1/\tau^2+n/\sigma^2}, \qquad \tilde\mu = \tilde\tau^2\!\left(\frac{\mu_0}{\tau^2}+\frac{n\bar x}{\sigma^2}\right)$$

---

## Part 7 假设检验（Ch.19–21）

### 7.1 基本框架

| 概念 | 公式 / 说明 |
|---|---|
| 第一类错误（$\alpha$） | 拒绝真 $H_0$，显著性水平 |
| 第二类错误（$\beta$） | 接受假 $H_0$ |
| 检验功效 | $\text{Power}=1-\beta=P(\text{拒绝}H_0\mid H_1\text{真})$ |
| $p$ 值 | 假设 $H_0$ 成立时，观测统计量或更极端的概率 |
| 拒绝域准则 | $p\text{值}<\alpha\Rightarrow$拒绝 $H_0$ |
| 单侧检验（右）：$H_1:\theta>\theta_0$ | 拒绝域 $T>c_\alpha$ |
| 单侧检验（左）：$H_1:\theta<\theta_0$ | 拒绝域 $T<-c_\alpha$ |
| 双侧检验：$H_1:\theta\neq\theta_0$ | 拒绝域 $\vert T\vert>c_{\alpha/2}$ |
| 功效函数 | $\beta(\theta)=P_\theta(\text{拒绝}H_0)$ |

> **定理（Neyman-Pearson 引理）**：最优势检验（UMP）的拒绝域形式为似然比 $\Lambda=L(\theta_1)/L(\theta_0)\geq k$。

### 7.2 广义似然比检验（GLRT）

| 公式 | 说明 |
|---|---|
| $\Lambda(\mathbf{x})=\dfrac{\sup_{\theta\in\Theta}L(\theta)}{\sup_{\theta\in\Theta_0}L(\theta)}$ | 广义似然比 |
| $-2\log\Lambda(\mathbf{x})\xrightarrow{d}\chi^2(d)$（大样本） | Wilks 定理，$d=\dim\Theta-\dim\Theta_0$ |

### 7.3 常用参数检验

| 检验 | 条件 | 统计量 | 零分布 |
|---|---|---|---|
| $z$ 检验（均值） | $\sigma^2$ 已知，正态/大样本 | $z=\dfrac{\bar X-\mu_0}{\sigma/\sqrt{n}}$ | $\mathcal{N}(0,1)$ |
| 单样本 $t$ 检验 | $\sigma^2$ 未知，正态 | $t=\dfrac{\bar X-\mu_0}{S/\sqrt{n}}$ | $t(n-1)$ |
| 配对 $t$ 检验 | 差值 $D_i=X_i-Y_i$，正态 | $t=\dfrac{\bar D-0}{S_D/\sqrt{n}}$ | $t(n-1)$ |
| 两样本 $t$ 检验（方差齐） | $\sigma_1^2=\sigma_2^2$ | $t=\dfrac{\bar X_1-\bar X_2}{S_p\sqrt{1/n_1+1/n_2}}$，$S_p^2=\dfrac{(n_1-1)S_1^2+(n_2-1)S_2^2}{n_1+n_2-2}$ | $t(n_1+n_2-2)$ |
| $\chi^2$ 检验（方差） | 正态 | $\chi^2=\dfrac{(n-1)S^2}{\sigma_0^2}$ | $\chi^2(n-1)$ |
| $F$ 检验（方差比） | 两正态总体 | $F=S_1^2/S_2^2$ | $F(n_1-1,n_2-1)$ |

### 7.4 方差分析（ANOVA）

单因素 ANOVA，$k$ 组，组 $i$ 有 $n_i$ 个观测，总 $N=\sum n_i$：

| 量 | 公式 |
|---|---|
| 组间 SS | $SS_A=\sum_i n_i(\bar X_i-\bar X)^2$ |
| 组内 SS | $SS_E=\sum_i\sum_j(X_{ij}-\bar X_i)^2$ |
| 总 SS | $SS_T=SS_A+SS_E$ |
| 组间 MS | $MS_A=SS_A/(k-1)$ |
| 组内 MS | $MS_E=SS_E/(N-k)$ |
| $F$ 统计量 | $F=MS_A/MS_E\sim F(k-1,N-k)$（$H_0$ 下） |

### 7.5 非参数检验

| 检验 | 统计量 | 零分布（大样本） |
|---|---|---|
| 符号检验 | $B^+=\#\{D_i>0\}\sim\text{Bin}(n',1/2)$ | — |
| Wilcoxon 符号秩检验 | $W^+=\sum_{D_i>0}R_i$；$E_0=n'(n'+1)/4$，$\operatorname{Var}_0=n'(n'+1)(2n'+1)/24$ | 近似 $\mathcal{N}$ |
| Mann-Whitney U 检验 | $U=\sum_{i,j}\mathbf{1}(X_i>Y_j)$；$E_0=mn/2$，$\operatorname{Var}_0=mn(m+n+1)/12$ | 近似 $\mathcal{N}$ |
| Kruskal-Wallis 检验 | $H=\dfrac{12}{N(N+1)}\sum_i\dfrac{R_i^2}{n_i}-3(N+1)$ | $\chi^2(k-1)$ |
| KS 检验 | $D_n=\sup_x\vert\hat F_n(x)-F_0(x)\vert$ | KS 分布 |
| 卡方拟合优度 | $\chi^2=\sum_i\dfrac{(O_i-E_i)^2}{E_i}$ | $\chi^2(k-1-p)$，$p$ 为估计参数数 |
| 列联表独立性 | $\chi^2=\sum_{ij}\dfrac{(O_{ij}-E_{ij})^2}{E_{ij}}$，$E_{ij}=R_i C_j/n$ | $\chi^2\!((r-1)(c-1))$ |

### 7.6 多重检验校正

| 方法 | 控制量 | 公式 |
|---|---|---|
| Bonferroni | FWER | 拒绝阈值 $\alpha/m$（$m$ 次检验） |
| Holm-Bonferroni | FWER（较强功效） | 步降法，第 $i$ 小 $p$ 值与 $\alpha/(m-i+1)$ 比较 |
| Benjamini-Hochberg | FDR | 排序 $p_{(1)}\leq\cdots\leq p_{(m)}$，拒绝 $p_{(k)}\leq k\alpha/m$ |

> ⚠️ 提示：Bonferroni 校正保守（FWER 控制可能过紧）；BH 方法在独立或正相关检验下控制 FDR，功效更高。

### 7.7 样本量计算

| 情景 | 公式 | 说明 |
|---|---|---|
| 均值检验（已知 $\sigma$） | $n=\left(\dfrac{(z_{\alpha/2}+z_\beta)\sigma}{\delta}\right)^2$ | $\delta$：最小检验差；$z_\beta$：功效对应分位数 |
| 比例检验 | $n=\dfrac{(z_{\alpha/2}+z_\beta)^2 p_0(1-p_0)}{\delta^2}$ | $\delta=p_1-p_0$ |
| 置信区间宽度控制 | $n=\left(\dfrac{z_{\alpha/2}\sigma}{d/2}\right)^2$ | $d$：期望区间宽度 |

---

## Part 8 高级主题（Ch.22–24）

### 8.1 信息论

| 公式 | 说明 |
|---|---|
| $I(A)=-\log P(A)$ | 自信息（比特/奈特） |
| $H(X)=-\sum_x p(x)\log p(x)=E[-\log p(X)]$ | 离散熵 |
| $h(X)=-\int f(x)\log f(x)\,dx$ | 微分熵（连续） |
| $H(X)\leq\log\vert\mathcal{X}\vert$，等号当且仅当均匀 | 熵的最大值 |
| $H(X)\geq0$（离散）；微分熵可为负 | 熵的非负性 |
| $H(X,Y)=H(X)+H(Y\mid X)$ | 链式法则 |
| $H(Y\mid X)=\sum_x p(x)H(Y\mid X=x)$ | 条件熵 |
| $H(Y\mid X)\leq H(Y)$，等号当且仅当 $X,Y$ 独立 | 条件减熵 |
| $I(X;Y)=H(X)-H(X\mid Y)=H(Y)-H(Y\mid X)$ | 互信息 |
| $I(X;Y)=H(X)+H(Y)-H(X,Y)$ | 互信息等价 |
| $I(X;Y)=D_{\mathrm{KL}}(p(x,y)\,\|\,p(x)p(y))\geq0$ | KL 形式 |
| $D_{\mathrm{KL}}(P\,\|\,Q)=\sum_x p(x)\log\dfrac{p(x)}{q(x)}\geq0$ | KL 散度（非对称） |
| $D_{\mathrm{KL}}(P\,\|\,Q)=0\Leftrightarrow P=Q$ a.e. | 零条件 |
| $H(P,Q)=-\sum_x p(x)\log q(x)$ | 交叉熵 |
| $H(P,Q)=H(P)+D_{\mathrm{KL}}(P\,\|\,Q)$ | 交叉熵分解 |
| $f$ 凸 $\Rightarrow$ $f(E[X])\leq E[f(X)]$ | Jensen 不等式 |
| $C=\max_{p(x)}I(X;Y)$（比特/次） | 信道容量 |
| $C=\frac{1}{2}\log(1+\text{SNR})$（高斯信道） | Shannon-Hartley 定理 |

> **定理（数据处理不等式）**：$X\to Y\to Z$ 马尔可夫链，$I(X;Z)\leq I(X;Y)$。处理不增加信息。

### 8.1b 信息论不等式体系

| 不等式 | 公式 | 条件 |
|---|---|---|
| 非负熵 | $H(X)\geq0$ | 离散 |
| 熵的次可加性 | $H(X_1,\ldots,X_n)\leq\sum_i H(X_i)$ | 等号当且仅当独立 |
| 条件减熵 | $H(X\mid Y,Z)\leq H(X\mid Y)$ | 更多条件不增加不确定性 |
| KL 非对称性 | $D_{\mathrm{KL}}(P\,\|\,Q)\neq D_{\mathrm{KL}}(Q\,\|\,P)$ 一般成立 | — |
| 对称 KL | $D_{\mathrm{KL}}(P\,\|\,Q)+D_{\mathrm{KL}}(Q\,\|\,P)\geq0$ | — |
| Pinsker 不等式 | $\vert\vert P-Q\vert\vert_1\leq\sqrt{2D_{\mathrm{KL}}(P\,\|\,Q)}$ | 全变差与 KL 散度联系 |
| Fano 不等式 | $H(X\mid\hat X)\leq h_b(P_e)+P_e\log(\vert\mathcal{X}\vert-1)$ | $P_e=P(X\neq\hat X)$ |

### 8.2 Monte Carlo 方法

| 公式 | 说明 |
|---|---|
| $I=\int_a^b f(x)\,dx=(b-a)\,E_{U(a,b)}[f(X)]$ | 改写为期望 |
| $\hat I_n=\dfrac{b-a}{n}\sum_{i=1}^n f(X_i)$，$X_i\overset{iid}{\sim}U(a,b)$ | MC 估计量 |
| $E[\hat I_n]=I$（无偏）；$\operatorname{Var}(\hat I_n)=\dfrac{(b-a)^2\sigma_f^2}{n}$ | 方差 |
| $\text{SE}=\sigma_f/\sqrt{n}$，与维度 $d$ 无关 | 维度无关收敛 |
| $\hat I^{\mathrm{IS}}=\dfrac{1}{n}\sum_{i=1}^n f(X_i)\,w(X_i)$，$w(x)=p(x)/q(x)$ | 重要性采样 |
| 最优提议分布：$q^*(x)\propto\vert f(x)\vert p(x)$ | 方差最小（理论值） |
| $A(x'\mid x)=\min\!\left(1,\dfrac{\tilde p(x')\,q(x\mid x')}{\tilde p(x)\,q(x'\mid x)}\right)$ | Metropolis-Hastings 接受率 |
| 细致平衡：$\pi(x)T(x'\mid x)=\pi(x')T(x\mid x')$ | 平稳分布充分条件 |
| Gibbs：轮流从 $p(x_i\mid\mathbf{x}_{-i})$ 采样，接受率恒为 1 | Gibbs 采样 |
| $H(\mathbf{x},\boldsymbol\rho)=-\log p(\mathbf{x})+\dfrac{1}{2}\boldsymbol\rho^\top M^{-1}\boldsymbol\rho$ | HMC 哈密顿量 |
| Leapfrog 步：$\boldsymbol\rho_{t+\epsilon/2}=\boldsymbol\rho_t-\frac{\epsilon}{2}\nabla_\mathbf{x}\log p(\mathbf{x}_t)$，等 | HMC 积分步 |
| 自归一化重要性采样：$\hat\mu^{\mathrm{SIS}}=\dfrac{\sum_i w_i f(X_i)}{\sum_i w_i}$ | 无需知道归一化常数 |
| 有效样本量（ESS）：$\text{ESS}=\dfrac{(\sum_i w_i)^2}{\sum_i w_i^2}$ | 衡量重要性权重的均匀程度 |

> ⚠️ 提示：重要性采样要求 $q$ 的支撑包含 $p$ 的支撑；$q$ 尾部若比 $p$ 轻则权重方差可爆炸。MCMC 中需丢弃 burn-in 样本；链的自相关使有效样本量远小于总样本量。

### 8.2b 拒绝采样

| 公式 | 说明 |
|---|---|
| 找包络：$M\,q(x)\geq \tilde p(x)$，$\forall x$ | $\tilde p$ 为目标（未归一化），$q$ 为提议分布 |
| 生成 $X\sim q$，$U\sim U(0,1)$ | — |
| 接受条件：$U\leq\dfrac{\tilde p(X)}{M\,q(X)}$ | 接受概率 $=1/M$（最优时） |
| 接受率 $=1/M$，越小越低效 | 高维下通常不可行 |

### 8.3 概率图模型

#### DAG（贝叶斯网络）

| 公式 | 说明 |
|---|---|
| $P(X_1,\ldots,X_n)=\prod_{i=1}^n P(X_i\mid\text{Pa}(X_i))$ | DAG 联合因子分解 |
| $X\perp\!\!\!\perp Y\mid Z\Leftrightarrow P(X,Y\mid Z)=P(X\mid Z)P(Y\mid Z)$ | 条件独立定义 |

d-分离规则（路径 $X-m-Y$，观测集 $\mathbf{Z}$）：

| 结构 | $m\in\mathbf{Z}$ | $m\notin\mathbf{Z}$ |
|---|---|---|
| 链 $X\to m\to Y$ | 阻断 | 畅通 |
| 分叉 $X\leftarrow m\to Y$ | 阻断 | 畅通 |
| 碰撞 $X\to m\leftarrow Y$ | **畅通** | 阻断 |

> ⚠️ 提示：碰撞节点的方向与链/分叉相反——观测了碰撞节点反而打通路径（解释消去效应）。

#### MRF（马尔可夫随机场）

| 公式 | 说明 |
|---|---|
| $P(\mathbf{X})=\dfrac{1}{Z}\prod_{c\in\mathcal{C}}\psi_c(\mathbf{X}_c)$ | 吉布斯分布，势函数乘积 |
| $Z=\sum_\mathbf{x}\prod_c\psi_c(\mathbf{x}_c)$ | 配分函数（归一化常数） |
| Hammersley-Clifford：MRF $\Leftrightarrow$ Gibbs 分布 | 团势分解等价于马尔可夫性 |

#### 因子图 & 信念传播

| 公式 | 说明 |
|---|---|
| 变量→因子消息：$\mu_{x\to f}(x)=\prod_{h\in\text{ne}(x)\setminus f}\mu_{h\to x}(x)$ | 变量节点消息 |
| 因子→变量消息：$\mu_{f\to x}(x)=\sum_{\sim x}f(\mathbf{X}_s)\prod_{y\in\text{ne}(f)\setminus x}\mu_{y\to f}(y)$ | 因子节点消息 |
| 边缘分布：$p(x)\propto\prod_{f\in\text{ne}(x)}\mu_{f\to x}(x)$ | 信念（树形图精确） |
| 循环信念传播（Loopy BP） | 有环图上迭代运行 BP，近似推断，无收敛保证 |

#### 变分推断（VI）

| 公式 | 说明 |
|---|---|
| 目标：$\min_{q\in\mathcal{Q}}D_{\mathrm{KL}}(q(\mathbf{Z})\,\|\,p(\mathbf{Z}\mid\mathbf{X}))$ | 用变分族 $\mathcal{Q}$ 近似后验 |
| ELBO：$\mathcal{L}(q)=E_q[\log p(\mathbf{X},\mathbf{Z})]-E_q[\log q(\mathbf{Z})]$ | 最大化 ELBO $\Leftrightarrow$ 最小化 KL |
| $\log p(\mathbf{X})=\mathcal{L}(q)+D_{\mathrm{KL}}(q\,\|\,p(\cdot\mid\mathbf{X}))\geq\mathcal{L}(q)$ | ELBO 是证据下界 |
| 均场假设：$q(\mathbf{Z})=\prod_i q_i(Z_i)$ | 各变量独立的变分族 |
| 均场更新：$\log q_j^*(Z_j)=E_{-j}[\log p(\mathbf{X},\mathbf{Z})]+\text{const}$ | 坐标上升 VI（CAVI） |

#### HMM 前向-后向算法

| 量 | 定义 / 递推 |
|---|---|
| 前向变量 | $\alpha_t(i)=P(O_1,\ldots,O_t,q_t=s_i)$ |
| $\alpha_1(i)=\pi_i b_i(O_1)$ | 初始化 |
| $\alpha_{t+1}(j)=b_j(O_{t+1})\sum_i\alpha_t(i)a_{ij}$ | 递推 |
| 后向变量 | $\beta_t(i)=P(O_{t+1},\ldots,O_T\mid q_t=s_i)$ |
| $\beta_T(i)=1$ | 初始化 |
| $\beta_t(i)=\sum_j a_{ij}b_j(O_{t+1})\beta_{t+1}(j)$ | 递推 |
| $P(\mathbf{O}\mid\lambda)=\sum_i\alpha_T(i)$ | 观测序列概率 |
| 状态后验：$\gamma_t(i)=\dfrac{\alpha_t(i)\beta_t(i)}{\sum_j\alpha_t(j)\beta_t(j)}$ | 前向×后向 |

#### EM 算法

| 步骤 | 公式 |
|---|---|
| E 步 | $Q(\theta\mid\theta^{(t)})=E_{\mathbf{Z}\mid\mathbf{X},\theta^{(t)}}[\log p(\mathbf{X},\mathbf{Z}\mid\theta)]$ |
| M 步 | $\theta^{(t+1)}=\arg\max_\theta Q(\theta\mid\theta^{(t)})$ |
| 单调性 | $\log p(\mathbf{X}\mid\theta^{(t+1)})\geq\log p(\mathbf{X}\mid\theta^{(t)})$ |
| ELBO 分解 | $\log p(\mathbf{X})=\mathcal{L}(q,\theta)+D_{\mathrm{KL}}(q(\mathbf{Z})\,\|\,p(\mathbf{Z}\mid\mathbf{X},\theta))$ |

### 8.4 马尔可夫链

| 概念 | 公式 / 说明 |
|---|---|
| 转移矩阵 | $P=(p_{ij})$，$p_{ij}=P(X_{n+1}=j\mid X_n=i)\geq0$，$\sum_j p_{ij}=1$ |
| $n$ 步转移 | $P^{(n)}=P^n$（矩阵 $n$ 次幂） |
| 平稳分布 | $\pi=\pi P$（行向量形式）；$\sum_j\pi_j=1$，$\pi_j\geq0$ |
| 细致平衡 | $\pi_i p_{ij}=\pi_j p_{ji}$，$\forall i,j$（可逆马尔可夫链充要条件） |
| 遍历定理 | 正常返、遍历链：$\frac{1}{n}\sum_{t=0}^{n-1}f(X_t)\xrightarrow{a.s.}E_\pi[f(X)]$ |
| 混合时间 | $t_{\mathrm{mix}}=\min\{t:\max_x\vert\vert P^t(x,\cdot)-\pi\vert\vert_{\mathrm{TV}}<1/4\}$ |
| 谱隙 | $\gamma=1-\lambda_2$（第二大特征值），混合时间 $\sim O(1/\gamma)$ |

### 8.5 高维统计与正则化（扩展）

| 方法 | 目标函数 | 特性 |
|---|---|---|
| Ridge（$L^2$） | $\vert\vert\mathbf{y}-X\boldsymbol\beta\vert\vert_2^2+\lambda\vert\vert\boldsymbol\beta\vert\vert_2^2$ | 系数收缩，无变量选择 |
| Lasso（$L^1$） | $\vert\vert\mathbf{y}-X\boldsymbol\beta\vert\vert_2^2+\lambda\vert\vert\boldsymbol\beta\vert\vert_1$ | 稀疏解，自动变量选择 |
| Elastic Net | $\vert\vert\mathbf{y}-X\boldsymbol\beta\vert\vert_2^2+\lambda_1\vert\vert\boldsymbol\beta\vert\vert_1+\lambda_2\vert\vert\boldsymbol\beta\vert\vert_2^2$ | 兼顾分组与稀疏 |
| MAP 角度 | Ridge $\leftrightarrow$ 正态先验；Lasso $\leftrightarrow$ Laplace 先验 | 贝叶斯解释 |

---

## 附录：常用特殊值速查

| 量 | 值 |
|---|---|
| $z_{0.025}=1.96$，$z_{0.05}=1.645$ | 正态分位数（双/单侧 5%） |
| $z_{0.005}=2.576$ | 正态分位数（双侧 1%） |
| $\Phi(1)=0.8413$，$\Phi(2)=0.9772$，$\Phi(3)=0.9987$ | 标准正态 CDF |
| $\Gamma(1/2)=\sqrt{\pi}$，$\Gamma(1)=1$，$\Gamma(3/2)=\sqrt{\pi}/2$ | Gamma 特殊值 |
| $e\approx2.718$，$\ln2\approx0.693$，$\log_2 e\approx1.443$ | 常用常数 |
| $\pi\approx3.14159$，$\sqrt{2}\approx1.414$，$\sqrt{\pi}\approx1.772$ | 几何常数 |
| $n=1$：$t_{0.025}(1)=12.706$；$n=9$：$t_{0.025}(9)=2.262$；$n\to\infty$：$\to 1.96$ | t 分位数趋近正态 |
| Gamma$(1,1)=\text{Exp}(1)$；Gamma$(n/2,1/2)=\chi^2(n)$；Beta$(1,1)=U(0,1)$ | 分布特殊情形 |
| $\chi^2(1)=Z^2$，$Z\sim\mathcal{N}(0,1)$；$\chi^2(2)=\text{Exp}(1/2)$ | 卡方特殊情形 |
| 二项式系数求和：$\sum_{k=0}^n\binom{n}{k}=2^n$；$\sum_{k=0}^n(-1)^k\binom{n}{k}=0$ | 恒等式 |

## 附录 B：公式记忆口诀

| 公式 | 口诀 |
|---|---|
| 贝叶斯公式 | "后验∝似然×先验"，归一化后得后验 |
| 全概率 vs 贝叶斯 | 全概率"由因到果"，贝叶斯"由果推因" |
| 切比雪夫 | "距均值 $k$ 倍标准差以外，概率不超过 $1/k^2$" |
| CLT | "i.i.d.，方差有限，标准化后趋向标准正态" |
| CR 下界 | "无偏估计的方差不低于 Fisher 信息量的倒数" |
| Rao-Blackwell | "任意无偏估计条件于充分统计量，MSE 不增" |
| d-分离碰撞结构 | "未观测碰撞阻断，观测碰撞打通"（与链/分叉相反） |
| EM 单调性 | "每步 ELBO 不减，似然单调不减" |

## 附录 C：易混淆公式对比

| 易混点 | 公式 A | 公式 B | 关键区别 |
|---|---|---|---|
| 方差 vs 无偏方差 | $\hat\sigma^2_{MLE}=\frac{1}{n}\sum(X_i-\bar X)^2$ | $S^2=\frac{1}{n-1}\sum(X_i-\bar X)^2$ | 分母 $n$ vs $n-1$ |
| 概率 vs 概率密度 | $P(X=x)=p(x)$ | $f(x)\Delta x\approx P(x<X<x+\Delta x)$ | 离散 vs 连续 |
| KL 散度 | $D_{\mathrm{KL}}(P\,\|\,Q)$ | $D_{\mathrm{KL}}(Q\,\|\,P)$ | 非对称，正向/反向 KL 行为不同 |
| MAP vs MMSE | $\arg\max p(\theta\mid x)$ | $E[\theta\mid x]$ | 后验众数 vs 后验均值 |
| 依概率收敛 vs a.s. 收敛 | $P(\vert X_n-X\vert>\varepsilon)\to0$ | $P(\lim X_n=X)=1$ | 弱/强大数定律对应 |
| 条件期望 vs 边缘期望 | $E[Y\mid X=x]$（函数）| $E[Y]$（数） | 全期望公式联系二者 |
| 充分统计量 vs 完备统计量 | $f(\mathbf{x};\theta)=g(T,\theta)h(\mathbf{x})$ | $E[g(T)]=0\Rightarrow g=0$ a.s. | Rao-Blackwell vs Lehmann-Scheffé |
