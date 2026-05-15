# 中等题库 D（100 题）

> 难度：★★★☆☆ — 多步推导 / 套路组合 / 含基本证明。每题 5-15 分钟。
> 编号约定：D.{Part}.{序}。**不附答案**。

**使用说明**：本题库共 100 题，覆盖概率论与数理统计全部 8 个 Part（Ch.1-24）。每题含 (a)(b)(c) 三小问，分别对应逐层推进的计算、证明或应用。题目类型分布：多步计算（~40%）、简单证明（~25%）、应用题（~25%）、反例构造（~10%）。建议与正文各章节配套练习，难度高于基础题库 C 一档，低于综合题库 E。

**章节对照**：Ch.1 随机事件 · Ch.2 条件概率 · Ch.3 组合数学 · Ch.4 离散随机变量 · Ch.5 连续随机变量 · Ch.6 多维随机变量 · Ch.7 离散分布族 · Ch.8 连续分布族 · Ch.9 多维分布 · Ch.10 大数定律 · Ch.11 中心极限定理 · Ch.12 收敛性理论 · Ch.13 统计量与抽样分布 · Ch.14 数据描述 · Ch.15 充分统计量 · Ch.16 点估计 · Ch.17 区间估计 · Ch.18 贝叶斯估计 · Ch.19 假设检验基础 · Ch.20 参数检验 · Ch.21 非参数检验 · Ch.22 信息论 · Ch.23 蒙特卡洛方法 · Ch.24 概率图模型。

---

## Part 1 概率基础（Ch.1-3，共 12 题）

> 覆盖随机事件（Ch.1）、条件概率与独立性（Ch.2）、组合计数（Ch.3）。
> 题型：多步全概率/贝叶斯 → 容斥 → 几何概型 → 独立性辨析 → 计数组合。
> 关键公式：全概率 $P(A)=\sum_i P(A\mid B_i)P(B_i)$；贝叶斯 $P(B_j\mid A)=P(A\mid B_j)P(B_j)/P(A)$；容斥 $P(A\cup B\cup C)=\sum P - \sum P(\cap) + P(\cap\cap)$。

**D.1.1**（Ch.2，全概率 + 贝叶斯反推）
某工厂三台机器生产同种产品，产量比为 3:2:5，次品率分别为 0.01、0.02、0.03。

(a) 求随机抽取一件产品为次品的概率。
(b) 已知取到次品，求它来自第 1 台机器的概率。
(c) 已知取到次品，求它来自第 3 台机器的概率，并与直觉比较。

---

**D.1.2**（Ch.1，容斥原理三事件）
调查 100 名学生，喜欢数学的 60 人，喜欢物理的 45 人，喜欢化学的 30 人；数学与物理都喜欢的 20 人，数学与化学都喜欢的 15 人，物理与化学都喜欢的 10 人；三门都喜欢的 5 人。

(a) 求至少喜欢一门的人数。
(b) 求恰好只喜欢一门的人数。
(c) 求恰好喜欢两门的人数。

---

**D.1.3**（Ch.3，几何概型）
在区间 $[0, 3]$ 上随机取两点 $X$ 和 $Y$（独立均匀）。

(a) 求 $|X - Y| \leq 1$ 的概率（几何概型，计算有利面积）。
(b) 求 $X + Y \leq 3$ 且 $X \leq Y$ 同时成立的概率。
(c) 求 $\max(X, Y) \leq 2$ 的概率。

---

**D.1.4**（Ch.2，独立性与条件概率辨析）
设 $A$、$B$ 是两个事件，$P(A) = 0.4$，$P(B) = 0.3$，$P(A \cup B) = 0.58$。

(a) 求 $P(A \cap B)$，判断 $A$ 与 $B$ 是否独立。
(b) 求 $P(A \mid B)$ 和 $P(B \mid A)$。
(c) 若再知 $P(A \mid B^c) = 0.5$，重新验证 $A$、$B$ 是否独立。

---

**D.1.5**（Ch.3，排列组合计数）
将 5 封不同的信随机投入 3 个不同的邮箱。

(a) 求每个邮箱至少有 1 封信的概率（容斥）。
(b) 求恰好有 2 个邮箱为空的概率。
(c) 求指定邮箱恰好有 2 封信的概率。

---

**D.1.6**（Ch.2，条件概率链）
某诊断测试对阳性患者的正确率为 95%，对阴性（健康）人的错误率为 3%。设人群患病率为 1%。

(a) 求测试结果为阳性的概率。
(b) 求测试结果为阳性时实际患病的概率（阳性预测值）。
(c) 求测试结果为阴性时实际患病的概率（假阴性率）。

---

**D.1.7**（Ch.1，概率公理与简单证明）
设 $P$ 满足概率公理，$A \subset B$。

(a) 证明：$P(B \setminus A) = P(B) - P(A)$（用可加性）。
(b) 证明：$P(A) \leq P(B)$（单调性）。
(c) 设 $A_1, A_2, \ldots$ 两两互斥，证明 $P\!\left(\bigcup_{i=1}^n A_i\right) = \sum_{i=1}^n P(A_i)$（有限可加性）。

---

**D.1.8**（Ch.3，二项系数与概率）
某人连续独立射击 10 次，每次命中率 0.6。

(a) 求恰好命中 6 次的概率（精确表达式）。
(b) 求命中次数不超过 4 次的概率（累积二项概率，列表达式即可）。
(c) 求至少命中 7 次的概率。

---

**D.1.9**（Ch.2，全概率 + 多阶段）
袋中有 3 个红球和 2 个白球。每次随机取 1 球，观察后放回，共取 3 次。

(a) 求 3 次全取红球的概率。
(b) 求恰好取到 2 个红球的概率。
(c) 已知第 3 次取到红球，求第 1 次也取到红球的概率。

---

**D.1.10**（Ch.1，事件运算与德摩根律）
设 $A$、$B$、$C$ 是三个事件。

(a) 用集合运算表达"$A$ 发生但 $B$ 和 $C$ 均不发生"。
(b) 证明 $\overline{A \cup B \cup C} = \overline{A} \cap \overline{B} \cap \overline{C}$（德摩根律）。
(c) 若三事件两两独立且满足 $P(A) = P(B) = P(C) = 0.5$，求 $P(A \cap B \cap C)$ 的可能取值范围（提示：两两独立不蕴含相互独立）。

---

**D.1.11**（Ch.2，Borel-Cantelli 预备）
设独立事件列 $\{A_n\}$ 满足 $P(A_n) = p$ 对所有 $n$ 成立，$0 < p < 1$。

(a) 求前 $n$ 次中 $A_n$ 一次也不发生的概率。
(b) 求 $n \to \infty$ 时上述概率的极限。
(c) 直觉上说明：为何当 $p > 0$ 时"$A_n$ 无穷多次发生"的概率为 1。

---

**D.1.12**（Ch.3，有限集合上的均匀模型）
从 $\{1, 2, \ldots, 52\}$ 中有放回地随机取 3 次。

(a) 求三次取出的数均不相同的概率。
(b) 求恰好出现 2 个相同数的概率（考虑哪两次相同）。
(c) 若改为**无放回**取 3 次，重新计算 (a)。

---

## Part 2 随机变量（Ch.4-6，共 15 题）

> 覆盖离散随机变量（Ch.4）、连续随机变量（Ch.5）、多维随机变量（Ch.6）。
> 题型：函数变换、矩计算、条件期望、联合/边际/条件分布、协方差与相关系数。
> 核心工具：变量变换公式 $f_Y(y)=f_X(g^{-1}(y))\cdot|dg^{-1}/dy|$；重期望 $E[X]=E[E[X\mid Y]]$；重方差 $\mathrm{Var}(X)=E[\mathrm{Var}(X\mid Y)]+\mathrm{Var}(E[X\mid Y])$。

**D.2.1**（Ch.5，连续随机变量函数变换）
设 $X \sim U(0, 2)$，令 $Y = X^2$。

(a) 求 $Y$ 的概率密度函数 $f_Y(y)$（用变量变换法）。
(b) 求 $E[Y]$（两种方法：直接用 $f_Y$ 或 LOTUS 公式）。
(c) 求 $\mathrm{Var}(Y)$。

---

**D.2.2**（Ch.4，离散随机变量的矩）
某随机变量 $X$ 的分布律为：

| $x$ | $-1$ | $0$ | $1$ | $2$ |
|-----|------|-----|-----|-----|
| $p$ | $a$ | $0.3$ | $0.2$ | $b$ |

且 $E[X] = 0.4$。

(a) 确定 $a$ 和 $b$。
(b) 求 $E[X^2]$ 和 $\mathrm{Var}(X)$。
(c) 求 $E[3X^2 - 2X + 1]$。

---

**D.2.3**（Ch.5，正态分布分位数与标准化）
设 $X \sim N(2, 9)$（均值 2，方差 9）。设 $\Phi$ 为标准正态 CDF。

(a) 用 $\Phi$ 表达 $P(1 \leq X \leq 5)$。
(b) 求 $P(|X - 2| > 3)$。
(c) 已知 $P(X > c) = 0.05$，求 $c$（用 $\Phi^{-1}$ 或 $z_{0.05} \approx 1.645$ 表达）。

---

**D.2.4**（Ch.4，泊松过程初步）
某网站每分钟访问量服从均值为 3 的泊松分布。

(a) 求某分钟内无访问的概率。
(b) 求某分钟内访问量超过 5 次的概率（用累积泊松 CDF 表达）。
(c) 若每次访问产生 0.1 元收益，求每分钟收益的期望和方差。

---

**D.2.5**（Ch.5，指数分布无记忆性）
设元件寿命 $X \sim \mathrm{Exp}(\lambda)$，$\lambda = 0.1$（单位：小时）。

(a) 求 $P(X > 20)$。
(b) 证明无记忆性：$P(X > s + t \mid X > s) = P(X > t)$，对任意 $s, t > 0$。
(c) 已知元件已工作 10 小时，求还能再工作 15 小时的概率。

---

**D.2.6**（Ch.4，期望与方差的线性性）
设 $X$ 和 $Y$ 相互独立，$E[X] = 1$，$\mathrm{Var}(X) = 2$，$E[Y] = -1$，$\mathrm{Var}(Y) = 3$。

(a) 求 $E[2X - 3Y + 4]$。
(b) 求 $\mathrm{Var}(2X - 3Y + 4)$。
(c) 求 $E[XY]$ 和 $E[X^2 Y^2]$（利用独立性）。

---

**D.2.7**（Ch.5，均匀分布的顺序统计量）
设 $X_1, X_2, X_3 \overset{\text{i.i.d.}}{\sim} U(0,1)$，令 $M = \max(X_1, X_2, X_3)$。

(a) 求 $M$ 的 CDF：$F_M(m) = P(M \leq m)$。
(b) 求 $M$ 的密度函数 $f_M(m)$。
(c) 求 $E[M]$。

---

**D.2.8**（Ch.4，条件期望）
掷一枚骰子，点数为 $N$；再掷 $N$ 枚硬币，设正面朝上次数为 $X$。

(a) 求 $E[X \mid N = n]$。
(b) 用重期望公式求 $E[X]$。
(c) 用条件方差公式求 $\mathrm{Var}(X)$（提示：$\mathrm{Var}(X) = E[\mathrm{Var}(X \mid N)] + \mathrm{Var}(E[X \mid N])$）。

---

**D.2.9**（Ch.5，混合分布）
设随机变量 $X$ 满足：以概率 $p$ 服从 $\mathrm{Exp}(1)$，以概率 $1-p$ 恒等于 0（即取值 0 的质量为 $1-p$）。

(a) 写出 $X$ 的 CDF（注意混合型分布的表达）。
(b) 求 $E[X]$ 和 $E[X^2]$。
(c) 求 $\mathrm{Var}(X)$，化简后用 $p$ 表达。

---

**D.2.10**（Ch.4，矩母函数入门）
设 $X \sim \mathrm{Bernoulli}(p)$，矩母函数 $M_X(t) = E[e^{tX}]$。

(a) 计算 $M_X(t)$。
(b) 利用 $M_X'(0)$ 求 $E[X]$，利用 $M_X''(0)$ 求 $E[X^2]$。
(c) 设 $Y = X_1 + X_2 + \cdots + X_n$（独立同分布 $\mathrm{Bernoulli}(p)$），写出 $M_Y(t)$，并指出 $Y$ 服从何分布。

---

**D.2.11**（Ch.5，Beta 分布积分）
设 $X \sim \mathrm{Beta}(2, 3)$，密度函数 $f(x) = 12x(1-x)^2$，$0 < x < 1$。

(a) 验证 $\int_0^1 12x(1-x)^2 \, dx = 1$。
(b) 求 $E[X]$ 和 $E[X^2]$（直接积分）。
(c) 求 $\mathrm{Var}(X)$，与 $\mathrm{Beta}(\alpha,\beta)$ 方差公式 $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ 对比。

---

**D.2.12**（Ch.4，负二项分布）
独立重复伯努利试验，每次成功概率为 $p$。设第 $r$ 次成功所需的总试验次数为 $X$（负二项分布）。

(a) 写出 $P(X = k)$，$k = r, r+1, \ldots$。
(b) 当 $r = 1$ 时，验证退化为几何分布。
(c) 利用 $E[X] = r/p$ 解释：平均需要多少次试验才能连续成功 $r = 3$ 次（$p = 0.5$）。

---

**D.2.13**（Ch.5，截断分布）
设 $X \sim \mathrm{Exp}(1)$，观测到 $X \leq 2$（截断）。设条件随机变量 $Y = (X \mid X \leq 2)$。

(a) 求 $Y$ 的 CDF 和密度函数。
(b) 求 $E[Y]$（积分后化简）。
(c) 与 $E[X] = 1$ 比较，说明截断后均值的变化方向及原因。

---

**D.2.14**（Ch.6，二维联合分布协方差）
设 $(X, Y)$ 的联合分布律为：

| | $Y = 0$ | $Y = 1$ |
|---|---------|---------|
| $X = 0$ | $0.1$ | $0.2$ |
| $X = 1$ | $0.3$ | $0.4$ |

(a) 求边际分布 $P(X = x)$ 和 $P(Y = y)$。
(b) 判断 $X$ 与 $Y$ 是否独立。
(c) 求 $\mathrm{Cov}(X, Y)$ 和相关系数 $\rho(X, Y)$。

---

**D.2.15**（Ch.6，二维连续分布的条件密度）
设 $(X, Y)$ 的联合密度为 $f(x,y) = 2$，$0 < x < y < 1$。

(a) 求边际密度 $f_X(x)$ 和 $f_Y(y)$。
(b) 求条件密度 $f_{Y|X}(y \mid x)$。
(c) 求 $E[Y \mid X = x]$，并验证 $E[E[Y \mid X]] = E[Y]$。

---

## Part 3 分布（Ch.7-9，共 15 题）

> 覆盖离散分布族（Ch.7）、连续分布族（Ch.8）、多维分布（Ch.9）。
> 题型：可加性证明、矩推导、极限近似、条件分布、多元正态及关联分布（$\chi^2$、$t$、$F$）。
> 重点分布：泊松（$\lambda$）、几何（$p$）、超几何、$\mathrm{Gamma}(\alpha,\beta)$、对数正态、多项、Dirichlet；关联：$Z^2\sim\chi^2(1)$，$Z/\sqrt{V/n}\sim t(n)$，$(U/m)/(V/n)\sim F(m,n)$。

**D.3.1**（Ch.7，泊松分布的可加性）
设 $X \sim \mathrm{Poisson}(\lambda_1)$，$Y \sim \mathrm{Poisson}(\lambda_2)$，且 $X \perp Y$。

(a) 利用概率母函数或矩母函数证明 $X + Y \sim \mathrm{Poisson}(\lambda_1 + \lambda_2)$。
(b) 求 $P(X = k \mid X + Y = n)$，说明其为二项分布（超几何结构）。
(c) 在 (b) 中，当 $\lambda_1 = \lambda_2$ 时，条件分布的参数是多少？

---

**D.3.2**（Ch.7，几何分布的矩）
设 $X \sim \mathrm{Geom}(p)$，即 $P(X = k) = (1-p)^{k-1}p$，$k = 1, 2, \ldots$。

(a) 利用矩母函数或生成函数，证明 $E[X] = 1/p$，$\mathrm{Var}(X) = (1-p)/p^2$。
(b) 证明无记忆性：$P(X > m + n \mid X > m) = P(X > n)$。
(c) 证明几何分布在离散非负整值分布中是**唯一**具有无记忆性的分布（反向：若无记忆则为几何）。

---

**D.3.3**（Ch.8，正态分布线性组合）
设 $X \sim N(\mu_1, \sigma_1^2)$，$Y \sim N(\mu_2, \sigma_2^2)$，$X \perp Y$。

(a) 证明 $aX + bY + c \sim N(a\mu_1 + b\mu_2 + c,\; a^2\sigma_1^2 + b^2\sigma_2^2)$。
(b) 令 $Z = X - Y$，求 $Z$ 的分布参数。
(c) 若 $X, Y \sim N(0, 1)$，求 $P(X > Y)$（利用 (b) 的结果）。

---

**D.3.4**（Ch.8，伽马分布的性质）
设 $X \sim \mathrm{Gamma}(\alpha, \beta)$，密度 $f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$，$x > 0$。

(a) 利用 Gamma 函数，证明 $E[X] = \alpha/\beta$，$E[X^2] = \alpha(\alpha+1)/\beta^2$，$\mathrm{Var}(X) = \alpha/\beta^2$。
(b) 验证：当 $\alpha = 1$ 时退化为指数分布 $\mathrm{Exp}(\beta)$。
(c) 设 $X_1, \ldots, X_n \overset{\text{i.i.d.}}{\sim} \mathrm{Exp}(\beta)$，证明 $X_1 + \cdots + X_n \sim \mathrm{Gamma}(n, \beta)$（用矩母函数）。

---

**D.3.5**（Ch.7，超几何分布与二项近似）
箱中有 $N = 100$ 个球，其中 $K = 20$ 个红球，无放回取 $n = 10$ 个。

(a) 写出取到恰好 $k$ 个红球的超几何分布公式。
(b) 求 $E[X]$ 和 $\mathrm{Var}(X)$（用超几何分布均值方差公式）。
(c) 用二项分布 $B(10, 0.2)$ 近似，计算 $P(X = 2)$ 的精确值与近似值，比较误差。

---

**D.3.6**（Ch.8，卡方分布与正态的联系）
设 $Z_1, Z_2, \ldots, Z_n \overset{\text{i.i.d.}}{\sim} N(0,1)$，$V = \sum_{i=1}^n Z_i^2$。

(a) 利用矩母函数证明 $V \sim \chi^2(n)$（即 $\mathrm{Gamma}(n/2, 1/2)$）。
(b) 求 $E[V]$ 和 $\mathrm{Var}(V)$。
(c) 设 $\bar{Z} = \frac{1}{n}\sum Z_i$，证明 $\bar{Z}$ 与 $V' = \sum(Z_i - \bar{Z})^2$ 相互独立（仅需陈述结论并说明依赖的定理）。

---

**D.3.7**（Ch.9，二维正态分布）
设 $(X, Y) \sim N(\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho)$。

(a) 写出边际分布 $X \sim N(\mu_1, \sigma_1^2)$ 的推导要点（积分消去 $y$）。
(b) 证明：在二维正态分布下，$\rho = 0 \Leftrightarrow X \perp Y$（利用联合密度可分离）。
(c) 求条件分布 $Y \mid X = x$ 的均值和方差（写出公式，不推导）。

---

**D.3.8**（Ch.8，对数正态分布）
设 $Y = e^X$，其中 $X \sim N(\mu, \sigma^2)$（对数正态分布）。

(a) 求 $Y$ 的密度函数（对数正态密度）。
(b) 求 $E[Y]$ 和 $\mathrm{Var}(Y)$（利用正态矩母函数 $E[e^{tX}] = e^{\mu t + \sigma^2 t^2/2}$）。
(c) 证明：$\ln Y \sim N(\mu, \sigma^2)$，即 $Y$ 的对数是正态的。

---

**D.3.9**（Ch.7，二项分布的泊松极限）
设 $X_n \sim B(n, p_n)$，其中 $n p_n \to \lambda > 0$ 当 $n \to \infty$。

(a) 证明 $P(X_n = k) \to \frac{\lambda^k e^{-\lambda}}{k!}$（利用 Stirling 近似或直接极限）。
(b) 举例说明何时该近似误差较小（$n$ 大、$p$ 小的条件）。
(c) 当 $n = 100$，$p = 0.02$ 时，用泊松近似计算 $P(X = 0)$ 与精确值，比较两者差异。

---

**D.3.10**（Ch.8，$t$ 分布的推导）
设 $Z \sim N(0,1)$，$V \sim \chi^2(n)$，且 $Z \perp V$，令 $T = Z / \sqrt{V/n}$。

(a) 写出 $T$ 的密度函数（$t(n)$ 分布，可查表引用，不要求推导完整）。
(b) 当 $n \to \infty$ 时，说明 $T$ 趋向标准正态（陈述理由）。
(c) 说明 $t$ 分布在小样本推断中的核心作用，举一应用场景。

---

**D.3.11**（Ch.9，多项分布）
将 $n = 12$ 个球随机分配到 3 个盒子，各盒概率分别为 $p_1 = 1/2$，$p_2 = 1/3$，$p_3 = 1/6$。

(a) 写出 $(X_1, X_2, X_3)$ 的多项分布公式。
(b) 求 $E[X_i]$ 和 $\mathrm{Var}(X_i)$（$i = 1, 2, 3$）。
(c) 求 $\mathrm{Cov}(X_1, X_2)$ 并验证 $\mathrm{Cov}(X_i, X_j) = -n p_i p_j$（$i \neq j$）。

---

**D.3.12**（Ch.8，$F$ 分布的性质）
设 $U \sim \chi^2(m)$，$V \sim \chi^2(n)$，$U \perp V$，令 $F = (U/m)/(V/n)$。

(a) 说明 $F \sim F(m, n)$（$F$ 分布的定义）。
(b) 证明：若 $F \sim F(m, n)$，则 $1/F \sim F(n, m)$。
(c) 说明 $F$ 分布与方差齐性检验的联系，描述单侧检验的拒绝域。

---

**D.3.13**（Ch.7，复合泊松分布）
设车辆事故次数 $N \sim \mathrm{Poisson}(\lambda)$，每次事故损失 $X_i \overset{\text{i.i.d.}}{\sim} \mathrm{Exp}(\mu)$（独立于 $N$），总损失 $S = \sum_{i=1}^N X_i$（约定 $N=0$ 时 $S=0$）。

(a) 求 $E[S]$（用重期望）。
(b) 求 $\mathrm{Var}(S)$（用条件方差公式）。
(c) 当 $\lambda = 2$，$\mu = 1$ 时，求 $E[S]$ 和 $\mathrm{Var}(S)$ 的数值。

---

**D.3.14**（Ch.8，混合正态分布）
设 $X \mid \Theta = \theta \sim N(\theta, 1)$，而 $\Theta \sim N(0, \tau^2)$（随机效应）。

(a) 求 $X$ 的边际分布（用重期望和全方差公式）。
(b) 求 $E[X]$ 和 $\mathrm{Var}(X)$。
(c) 求 $\mathrm{Cov}(X, \Theta)$（利用 $\mathrm{Cov}(X,\Theta) = E[X\Theta] - E[X]E[\Theta]$）。

---

**D.3.15**（Ch.9，Dirichlet 分布简介）
设 $(X_1, X_2, X_3)$ 服从 $\mathrm{Dirichlet}(1, 1, 1)$ 分布（即 $[0,1]^2$ 上的均匀单纯形）。

(a) 写出联合密度（在单纯形 $x_1 + x_2 + x_3 = 1$，$x_i > 0$ 上）。
(b) 求每个边际分布（$X_i \sim \mathrm{Beta}(1, 2)$），写出密度。
(c) 求 $E[X_i]$ 和 $\mathrm{Var}(X_i)$，验证与 $\mathrm{Dirichlet}$ 均值方差公式一致。

---

## Part 4 极限定理（Ch.10-12，共 10 题）

> 覆盖大数定律（Ch.10）、中心极限定理（Ch.11）、收敛性理论（Ch.12）。
> 题型：WLLN/SLLN 条件验证、CLT 近似与 Berry-Esseen、各类收敛的证明与反例、特征函数。
> 收敛关系（强弱顺序）：$L^2$ 收敛 $\Rightarrow$ 依概率收敛；几乎处处收敛 $\Rightarrow$ 依概率收敛；依分布收敛最弱。逆向均不成立（需反例区分）。

**D.4.1**（Ch.10，大数定律验证条件）
设 $X_1, X_2, \ldots$ 独立，$P(X_n = \pm n^\alpha) = 1/2$（$\alpha > 0$）。

(a) 计算 $E[X_n]$ 和 $\mathrm{Var}(X_n)$。
(b) 确定 $\alpha$ 的范围使得弱大数定律成立（Chebyshev 条件：$\sum \mathrm{Var}(X_n)/n^2 < \infty$）。
(c) 当 $\alpha = 1$ 时，$n^{-1}\sum_{i=1}^n X_i \xrightarrow{P} 0$ 是否成立？给出理由。

---

**D.4.2**（Ch.11，CLT 的应用）
某保险公司承保 1000 个独立客户，每个客户年赔付额均值为 500 元，标准差为 2000 元。设总赔付额为 $S$。

(a) 用 CLT 近似 $P(S > 550000)$（写出标准化步骤）。
(b) 求使 $P(S \leq c) \geq 0.99$ 的最小 $c$（用 $z_{0.01} \approx 2.326$）。
(c) 若各客户赔付额实际服从均值相同的指数分布（方差更大），CLT 近似的准确性如何？

---

**D.4.3**（Ch.10，Markov 不等式与 Chebyshev 不等式）
设 $X \geq 0$，$E[X] = \mu$，$\mathrm{Var}(X) = \sigma^2$。

(a) 证明 Markov 不等式：$P(X \geq a) \leq \mu/a$（$a > 0$）。
(b) 由 Markov 不等式推导 Chebyshev 不等式：$P(|X - \mu| \geq k\sigma) \leq 1/k^2$。
(c) 对 $X \sim U(0,1)$，$k = 2$：计算 Chebyshev 上界，再精确计算 $P(|X - 0.5| \geq 0.5 \cdot 2 / \sqrt{12})$，比较上界的松紧程度。

---

**D.4.4**（Ch.12，依概率收敛的反例）
构造随机变量序列 $\{X_n\}$，使得 $X_n \xrightarrow{P} 0$，但 $E[X_n] \not\to 0$。

(a) 令 $P(X_n = n) = 1/n$，$P(X_n = 0) = 1 - 1/n$。计算 $E[X_n]$。
(b) 证明 $X_n \xrightarrow{P} 0$（定义验证）。
(c) 说明这个例子的含义：依概率收敛不保证均值的收敛。

---

**D.4.5**（Ch.11，CLT 的 Berry-Esseen 定理应用）
设 $X_i \overset{\text{i.i.d.}}{\sim}$，$E[X_i] = 0$，$E[X_i^2] = 1$，$E[\vert X_i \vert^3] = \rho < \infty$。Berry-Esseen 定理：$\sup_x |P(S_n/\sqrt{n} \leq x) - \Phi(x)| \leq C\rho/\sqrt{n}$（$C \leq 0.4785$）。

(a) 对 $X_i \sim \mathrm{Bernoulli}(1/2) - 1/2$（中心化），求 $\rho = E[|X_i|^3]$。
(b) 当 $n = 100$ 时，Berry-Esseen 上界为多少？
(c) 解释 Berry-Esseen 定理的实用意义：$n$ 多大才能使正态近似误差小于 0.01？

---

**D.4.6**（Ch.12，各种收敛关系）
证明或举反例：

(a) 几乎处处收敛 $\Rightarrow$ 依概率收敛（证明）。
(b) 依概率收敛 $\not\Rightarrow$ 几乎处处收敛（构造反例：在 $[0,1]$ 上的"走马灯"序列）。
(c) $L^2$ 收敛 $\Rightarrow$ 依概率收敛（证明，利用 Markov 不等式）。

---

**D.4.7**（Ch.10，强大数定律）
设 $X_1, X_2, \ldots \overset{\text{i.i.d.}}{\sim}$，$E[|X_1|] < \infty$，$E[X_1] = \mu$。

(a) 陈述 Kolmogorov 强大数定律（SLLN）的条件和结论。
(b) 用 SLLN 说明：若 $X_i \sim \mathrm{Bernoulli}(p)$，则样本频率 $\bar{X}_n \xrightarrow{\text{a.s.}} p$。
(c) SLLN 与 WLLN 的结论强弱比较：哪个结论更强？给出逻辑关系。

---

**D.4.8**（Ch.11，多维 CLT）
设 $\mathbf{X}_1, \ldots, \mathbf{X}_n \overset{\text{i.i.d.}}{\sim}$（$\mathbb{R}^d$ 值），$E[\mathbf{X}_i] = \boldsymbol{\mu}$，$\mathrm{Cov}(\mathbf{X}_i) = \boldsymbol{\Sigma}$。

(a) 陈述多维 CLT：$\sqrt{n}(\bar{\mathbf{X}} - \boldsymbol{\mu}) \xrightarrow{d} N(\mathbf{0}, \boldsymbol{\Sigma})$。
(b) 用 delta 方法：若 $g: \mathbb{R}^d \to \mathbb{R}$ 可微，写出 $\sqrt{n}(g(\bar{\mathbf{X}}) - g(\boldsymbol{\mu}))$ 的渐近分布。
(c) 对 $d = 1$，$g(x) = x^2$，具体写出渐近方差（$\mu, \sigma^2$ 表达）。

---

**D.4.9**（Ch.12，特征函数与弱收敛）
设 $X_n \xrightarrow{d} X$。利用特征函数（CF）工具：

(a) 陈述 Lévy 连续性定理：$X_n \xrightarrow{d} X \Leftrightarrow \varphi_{X_n}(t) \to \varphi_X(t)$（逐点）。
(b) 利用 CF 证明 CLT 对 $X_i \sim N(0,1)$ 是平凡的（说明 $\varphi_{\bar{X}\sqrt{n}}$ 的计算）。
(c) 说明为何 CF 方法比直接 CDF 方法更强大（例：可处理非对称或重尾分布）。

---

**D.4.10**（Ch.11，样本均值的大样本置信区间）
设 $X_1, \ldots, X_n \overset{\text{i.i.d.}}{\sim}$，$E[X_i] = \mu$，$\mathrm{Var}(X_i) = \sigma^2$（未知）。

(a) 用 CLT 构造 $\mu$ 的近似 $95\%$ 置信区间（用样本标准差 $S$ 替代 $\sigma$）。
(b) 若样本为：$2, 5, 3, 7, 4$（$n=5$），计算 $\bar{X}$ 和 $S$，写出 95% CI 的具体数值。
(c) 解释：为何 $n = 5$ 时用 CLT 近似而非 $t$ 分布是不严格的？

---

## Part 5 统计基础（Ch.13-15，共 10 题）

> 覆盖统计量与抽样分布（Ch.13）、数据描述（Ch.14）、充分统计量（Ch.15）。
> 题型：抽样分布推导、顺序统计量、经验 CDF、因子分解定理、Rao-Blackwell、Bootstrap。
> 核心结论：$\bar{X} \sim N(\mu, \sigma^2/n)$；$(n-1)S^2/\sigma^2 \sim \chi^2(n-1)$；$\bar{X} \perp S^2$（正态总体）；$T = (\bar{X}-\mu)/(S/\sqrt{n}) \sim t(n-1)$。

**D.5.1**（Ch.13，常见统计量的分布）
设 $X_1, \ldots, X_n \overset{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$，$\bar{X}$ 为样本均值，$S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$。

(a) 写出 $\bar{X}$ 的分布，并证明 $(n-1)S^2/\sigma^2 \sim \chi^2(n-1)$（陈述结论与关键步骤）。
(b) 写出 $T = (\bar{X} - \mu)/(S/\sqrt{n})$ 的分布（$t$ 分布，说明为何自由度 $n-1$）。
(c) 写出两独立正态总体方差比的 $F$ 统计量及其分布。

---

**D.5.2**（Ch.13，顺序统计量）
设 $X_1, \ldots, X_n \overset{\text{i.i.d.}}{\sim} F(x)$，密度为 $f(x)$。设 $X_{(1)} \leq \cdots \leq X_{(n)}$ 为顺序统计量。

(a) 推导第 $k$ 个顺序统计量 $X_{(k)}$ 的密度函数：$f_{X_{(k)}}(x) = \frac{n!}{(k-1)!(n-k)!} [F(x)]^{k-1}[1-F(x)]^{n-k} f(x)$。
(b) 对 $X_i \sim U(0,1)$，求 $X_{(n)}$（最大值）的密度和期望。
(c) 对 $n = 5$，$k = 3$（中位数），写出 $X_{(3)}$ 的密度（$U(0,1)$ 情形）。

---

**D.5.3**（Ch.14，经验分布函数）
设 $x_1, \ldots, x_n$ 为样本，经验 CDF 为 $\hat{F}_n(x) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}[x_i \leq x]$。

(a) 证明：对固定 $x$，$n\hat{F}_n(x) \sim B(n, F(x))$，因此 $E[\hat{F}_n(x)] = F(x)$，$\mathrm{Var}(\hat{F}_n(x)) = F(x)(1-F(x))/n$。
(b) 由 SLLN，$\hat{F}_n(x) \xrightarrow{\text{a.s.}} F(x)$（陈述 Glivenko-Cantelli 定理的结论）。
(c) 对样本 $\{1, 3, 3, 5, 7\}$（$n=5$），写出 $\hat{F}_5(x)$ 的表达式（分段函数）。

---

**D.5.4**（Ch.15，充分统计量的因子分解定理）
设总体密度为 $f(x;\theta)$，样本 $X_1, \ldots, X_n$ 的联合密度为 $\prod_{i=1}^n f(x_i;\theta)$。

(a) 陈述 Neyman-Fisher 因子分解定理：$T(\mathbf{X})$ 是 $\theta$ 的充分统计量 $\Leftrightarrow \prod f(x_i;\theta) = g(T(\mathbf{x}), \theta) h(\mathbf{x})$。
(b) 对 $X_i \sim N(\mu, 1)$，证明 $\bar{X}$ 是 $\mu$ 的充分统计量（写出联合密度分解）。
(c) 对 $X_i \sim \mathrm{Exp}(\theta)$（密度 $\theta e^{-\theta x}$），求 $\theta$ 的充分统计量。

---

**D.5.5**（Ch.13，Bootstrap 基本思想）
设样本 $X_1, \ldots, X_n \overset{\text{i.i.d.}}{\sim} F$，统计量 $T_n = T(X_1,\ldots,X_n)$。

(a) 描述参数 Bootstrap 的步骤：用 $\hat{F}$ 替代 $F$，从 $\hat{F}$ 有放回抽样 $B$ 次，得到 $T_n^{*(1)}, \ldots, T_n^{*(B)}$。
(b) 用 Bootstrap 估计 $T_n$ 的标准误差（写出公式）。
(c) 对样本 $\{2, 4, 6, 8\}$（$n=4$），若 $T_n = \bar{X}$，列出所有可能的 Bootstrap 样本中，$T^* = \bar{X}^*$ 的最小值和最大值。

---

**D.5.6**（Ch.14，Q-Q 图原理）
设理论分布为 $F_0$，样本 $X_1 \leq \cdots \leq X_n$ 为顺序统计量。Q-Q 图横轴为理论分位数，纵轴为样本分位数。

(a) 若 $X_i \sim F_0$，则 Q-Q 图大致为直线，说明其理论依据（用顺序统计量的近似期望）。
(b) 若实际分布比 $F_0$ 有更重的尾（如 $t$ 分布 vs 正态），Q-Q 图会出现何种弯曲？
(c) 对样本 $\{-1.5, -0.3, 0.1, 0.8, 1.6\}$，与 $N(0,1)$ 做 Q-Q 图（列出理论分位数 $\Phi^{-1}((i-0.5)/5)$ 对应 $i=1,\ldots,5$）。

---

**D.5.7**（Ch.15，完备充分统计量）
设 $X_1, \ldots, X_n \overset{\text{i.i.d.}}{\sim} \mathrm{Poisson}(\lambda)$，$T = \sum X_i$。

(a) 证明 $T \sim \mathrm{Poisson}(n\lambda)$（可加性）。
(b) 验证 $T$ 是 $\lambda$ 的充分统计量（因子分解定理）。
(c) 陈述完备性定义，并说明 $T$ 是否完备（提示：指数族的自然参数统计量通常是完备充分的）。

---

**D.5.8**（Ch.13，抽样分布的数值例）
设总体 $N(5, 4)$（$\sigma^2 = 4$），取样本量 $n = 16$。

(a) 求 $P(\bar{X} > 6)$（标准化后查正态表）。
(b) 求 $P(S^2 > 6.908)$（利用 $15S^2/4 \sim \chi^2(15)$，查表或写出 $\chi^2$ 分位数）。
(c) 若两个独立样本各 $n_1 = n_2 = 10$，总体方差相同，写出方差比 $S_1^2/S_2^2$ 的分布。

---

**D.5.9**（Ch.14，描述统计：箱线图与异常值）
一组数据：$\{2, 3, 5, 7, 8, 9, 11, 14, 16, 22\}$（$n=10$）。

(a) 计算 $Q_1$（第 25 百分位）、$Q_2$（中位数）、$Q_3$（第 75 百分位）。
(b) 计算 IQR，并确定异常值的判别界限（$Q_1 - 1.5\cdot\text{IQR}$ 和 $Q_3 + 1.5\cdot\text{IQR}$）。
(c) 指出数据中的异常值，并用均值和中位数分别衡量中心，比较两者对异常值的鲁棒性。

---

**D.5.10**（Ch.15，Rao-Blackwell 定理）
设 $T$ 是参数 $\theta$ 的充分统计量，$W$ 是 $\theta$ 的任意无偏估计量（$E_\theta[W] = \theta$）。

(a) 陈述 Rao-Blackwell 定理：令 $W^* = E[W \mid T]$，则 $W^*$ 也是无偏的且 $\mathrm{MSE}(W^*) \leq \mathrm{MSE}(W)$。
(b) 对 $X_i \sim \mathrm{Bernoulli}(p)$，$W = X_1$（只用第一个观测），$T = \sum X_i$。计算 $W^* = E[X_1 \mid T = t]$。
(c) 验证 $W^* = T/n = \bar{X}$ 是 $p$ 的 UMVUE（均匀最小方差无偏估计量）。

---

## Part 6 估计（Ch.16-18，共 14 题）

> 覆盖点估计（Ch.16）、区间估计（Ch.17）、贝叶斯估计（Ch.18）。
> 题型：矩估计、MLE 与不变性、CRB 与有效性、CI 构造与宽度分析、共轭先验、MAP 与正则化联系、EM 算法思想。
> 常用共轭对：Beta-Binomial，Gamma-Poisson，Normal-Normal（均值未知），Dirichlet-Multinomial；CRB：$\mathrm{Var}(\hat{\theta}) \geq 1/I(\theta)$，其中 $I(\theta) = -E[\partial^2 \log f/\partial\theta^2]$。

**D.6.1**（Ch.16，矩估计法）
设 $X_1, \ldots, X_n \overset{\text{i.i.d.}}{\sim} \mathrm{Gamma}(\alpha, \beta)$，均值 $\alpha/\beta$，方差 $\alpha/\beta^2$。

(a) 用矩估计法，令样本一阶矩 = $\alpha/\beta$，样本二阶矩 = $\alpha(\alpha+1)/\beta^2$，解出 $\hat{\alpha}$ 和 $\hat{\beta}$（用 $\bar{X}$ 和 $\overline{X^2}$ 表达）。
(b) 另一种矩估计：令 $\bar{X} = \alpha/\beta$，$S^2 = \alpha/\beta^2$，解出 $\hat{\alpha}$ 和 $\hat{\beta}$（更简洁的形式）。
(c) 比较两种矩估计的表达，说明选取哪些总体矩对应哪些样本矩会影响结果（矩估计的非唯一性）。

---

**D.6.2**（Ch.16，最大似然估计）
设 $X_1, \ldots, X_n \overset{\text{i.i.d.}}{\sim} \mathrm{Uniform}(0, \theta)$，$\theta > 0$ 未知。

(a) 写出似然函数 $L(\theta) = \prod_{i=1}^n f(x_i; \theta)$（注意定义域约束 $\theta \geq x_{(n)}$）。
(b) 求 $\theta$ 的 MLE：$\hat{\theta} = X_{(n)}$（最大顺序统计量）。
(c) 证明 $\hat{\theta}$ 是有偏的，求偏差，并构造一个无偏修正估计量 $\hat{\theta}^*$。

---

**D.6.3**（Ch.16，MLE 的不变性）
设 $X_1, \ldots, X_n \overset{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$。

(a) 求 $\mu$ 和 $\sigma^2$ 的 MLE（$\hat{\mu} = \bar{X}$，$\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$）。
(b) 利用 MLE 的不变性，求 $P(X \leq c) = \Phi\!\left(\frac{c-\mu}{\sigma}\right)$ 的 MLE。
(c) 求 $\sigma$ 的 MLE 的渐近分布（用 Fisher 信息量 $I(\sigma^2)$ 表达 CRB）。

---

**D.6.4**（Ch.16，Cramér-Rao 下界）
设 $X_1, \ldots, X_n \overset{\text{i.i.d.}}{\sim} \mathrm{Poisson}(\lambda)$，Fisher 信息量 $I(\lambda) = n/\lambda$。

(a) 计算单个样本 $X_1$ 的 Fisher 信息量 $I_1(\lambda)$（验证为 $1/\lambda$）。
(b) 写出 $\lambda$ 的无偏估计量方差的 Cramér-Rao 下界（CRB）。
(c) 验证 $\bar{X}$ 的方差 $\lambda/n$ 达到 CRB，因此 $\bar{X}$ 是有效估计量。

---

**D.6.5**（Ch.17，单正态均值的区间估计）
设 $X_1, \ldots, X_{25} \overset{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$，$\bar{X} = 12.3$，$S = 2.4$，$\sigma$ 未知。

(a) 构造 $\mu$ 的 95% 置信区间（用 $t(24)$ 分布，$t_{0.025}(24) \approx 2.064$）。
(b) 若已知 $\sigma = 2.5$，改用正态分布（$z_{0.025} = 1.96$）构造 95% CI，比较区间宽度。
(c) 解释：样本量 $n$ 增加时，置信区间宽度如何变化？若要使宽度减半需增加几倍样本量？

---

**D.6.6**（Ch.17，方差的区间估计）
设样本 $X_1, \ldots, X_{10} \sim N(\mu, \sigma^2)$（均值未知），$S^2 = 4.5$。

(a) 写出 $\sigma^2$ 的 95% 置信区间（用 $\chi^2(9)$ 的上下分位数 $\chi^2_{0.025}(9) \approx 19.02$，$\chi^2_{0.975}(9) \approx 2.70$）。
(b) 写出 $\sigma$（标准差）的 95% CI（对 (a) 两端开根号）。
(c) 解释为何方差的 CI 是不对称的，与均值 CI 的对称性形成对比。

---

**D.6.7**（Ch.17，两正态均值差的区间估计）
两组独立样本：$n_1 = 10$，$\bar{X}_1 = 15$，$S_1^2 = 4$；$n_2 = 12$，$\bar{X}_2 = 13$，$S_2^2 = 5$。假设方差相等（等方差）。

(a) 计算合并方差 $S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}$。
(b) 构造 $\mu_1 - \mu_2$ 的 95% CI（自由度 $n_1 + n_2 - 2 = 20$，$t_{0.025}(20) \approx 2.086$）。
(c) 如果不假设方差相等，改用 Welch-Satterthwaite 近似自由度（写出公式，不计算具体值）。

---

**D.6.8**（Ch.18，先验与后验）
设硬币正面概率 $\theta \in [0,1]$，取先验 $\theta \sim \mathrm{Beta}(2, 2)$。观测 $n = 10$ 次，$k = 7$ 次正面。

(a) 写出似然 $L(\theta) \propto \theta^7 (1-\theta)^3$。
(b) 写出后验 $\theta \mid k \sim \mathrm{Beta}(9, 5)$（用 Beta-Binomial 共轭性）。
(c) 求后验均值、后验众数（MAP）、95% 后验可信区间的端点（Beta 分位数，写出表达式即可）。

---

**D.6.9**（Ch.18，共轭先验族）
设 $X_1, \ldots, X_n \overset{\text{i.i.d.}}{\sim} \mathrm{Poisson}(\lambda)$，取先验 $\lambda \sim \mathrm{Gamma}(\alpha, \beta)$（率参数化：密度 $\propto \lambda^{\alpha-1}e^{-\beta\lambda}$）。

(a) 写出似然 $L(\lambda \mid \mathbf{x}) \propto \lambda^{\sum x_i} e^{-n\lambda}$。
(b) 推导后验 $\lambda \mid \mathbf{x} \sim \mathrm{Gamma}(\alpha + \sum x_i, \; \beta + n)$（共轭性）。
(c) 写出后验均值，解释它是先验均值 $\alpha/\beta$ 与 MLE $\bar{X}$ 的加权平均。

---

**D.6.10**（Ch.16，正则化与 MAP 的联系）
在线性回归中，设参数 $\boldsymbol{\beta}$ 的先验为 $N(\mathbf{0}, \tau^2 \mathbf{I})$，噪声方差 $\sigma^2$ 已知。

(a) 写出 MAP 估计等价于最小化 $\|\mathbf{y} - X\boldsymbol{\beta}\|^2 + \frac{\sigma^2}{\tau^2}\|\boldsymbol{\beta}\|^2$（推导等价关系）。
(b) 说明正则化参数 $\lambda = \sigma^2/\tau^2$ 的贝叶斯解释：先验方差 $\tau^2 \to \infty$ 对应 $\lambda \to 0$（无正则化）。
(c) 比较 Ridge 回归（L2 正则化）与 Lasso（L1 正则化）对应的先验分布族（Laplace 先验）。

---

**D.6.11**（Ch.17，比例的区间估计）
调查 $n = 400$ 人，$\hat{p} = 0.6$ 支持某政策。

(a) 构造 $p$ 的近似 95% CI（正态近似，$z_{0.025} = 1.96$）。
(b) 确定样本量 $n^*$，使 95% CI 的宽度不超过 0.04（最保守估计：$p = 0.5$）。
(c) 若用 Wilson 区间代替 Wald 区间，说明 Wilson 区间在 $\hat{p}$ 接近 0 或 1 时的优势。

---

**D.6.12**（Ch.18，贝叶斯预测分布）
设 $X \mid \theta \sim N(\theta, 1)$，先验 $\theta \sim N(0, \tau^2)$。观测 $X = x_0$。

(a) 求后验 $\theta \mid X = x_0$（正态-正态共轭，写出后验均值和方差）。
(b) 求预测分布 $\tilde{X} \mid X = x_0$ 的分布，其中 $\tilde{X} \mid \theta \sim N(\theta, 1)$（用重期望）。
(c) 比较预测方差与后验方差，解释预测方差更大的原因（额外的采样不确定性）。

---

**D.6.13**（Ch.16，EM 算法思想）
设混合高斯分布：以概率 $\pi$ 来自 $N(\mu_1, 1)$，以概率 $1-\pi$ 来自 $N(\mu_2, 1)$。观测样本 $x_1, \ldots, x_n$。

(a) 引入潜变量 $Z_i \in \{1, 2\}$，写出完整数据对数似然 $\ell_c(\theta)$。
(b) E 步：计算后验 $r_{ik} = P(Z_i = k \mid x_i, \theta^{(t)})$（软分配责任）。
(c) M 步：最大化 $Q(\theta \mid \theta^{(t)}) = E[\ell_c \mid \mathbf{x}, \theta^{(t)}]$，写出 $\hat{\mu}_k$ 的更新公式（加权均值）。

---

**D.6.14**（Ch.18，Jeffreys 先验）
对参数 $\theta$ 的模型 $f(x;\theta)$，Jeffreys 先验 $\pi_J(\theta) \propto \sqrt{I(\theta)}$（$I(\theta)$ 为 Fisher 信息量）。

(a) 对 $X \sim \mathrm{Bernoulli}(\theta)$，计算 $I(\theta) = 1/(\theta(1-\theta))$，推导 Jeffreys 先验。
(b) 说明 Jeffreys 先验是 $\mathrm{Beta}(1/2, 1/2)$ 分布，与均匀先验 $\mathrm{Beta}(1,1)$ 比较。
(c) Jeffreys 先验的重参数不变性：若 $\phi = g(\theta)$ 是单调变换，说明 Jeffreys 先验在变换下保持同等性质（简述原理）。

---

## Part 7 假设检验（Ch.19-21，共 12 题）

> 覆盖假设检验基础（Ch.19）、参数检验（Ch.20）、非参数检验（Ch.21）。
> 题型：$z/t/\chi^2/F$ 检验步骤、功效计算、N-P 引理、Wilcoxon 符号秩、ANOVA、独立性 $\chi^2$、多重检验校正。

**D.7.1**（Ch.19，单均值 $z$ 检验）
某厂声称产品均值 $\mu_0 = 50$，$\sigma = 8$ 已知。抽取 $n = 64$，$\bar{X} = 48.5$。

(a) 建立双侧检验的假设 $H_0: \mu = 50$ vs $H_1: \mu \neq 50$。
(b) 计算检验统计量 $z$，在 $\alpha = 0.05$ 下做出决策（$z_{0.025} = 1.96$）。
(c) 计算 $p$ 值，并解释 $p$ 值的含义。

---

**D.7.2**（Ch.19，单均值 $t$ 检验）
某班成绩假设均值 $\mu_0 = 75$，抽取 $n = 16$，$\bar{X} = 78$，$S = 8$。

(a) 建立单侧检验 $H_0: \mu \leq 75$ vs $H_1: \mu > 75$。
(b) 计算 $T = (\bar{X} - \mu_0)/(S/\sqrt{n})$，在 $\alpha = 0.05$ 下做决策（$t_{0.05}(15) \approx 1.753$）。
(c) 若 $\mu$ 的真实值为 80，求此次检验的功效 $\beta$（计算非中心参数 $\delta = (\mu_1 - \mu_0)/(S/\sqrt{n})$，查非中心 $t$ 表或近似）。

---

**D.7.3**（Ch.19，两类错误与检验的功效）
设 $H_0: \mu = 0$ vs $H_1: \mu = 1$，$X \sim N(\mu, 1)$，拒绝域为 $X > c$。

(a) 写出第一类错误 $\alpha(c) = P(X > c \mid \mu = 0)$ 的表达式。
(b) 写出第二类错误 $\beta(c) = P(X \leq c \mid \mu = 1)$ 的表达式。
(c) 画出（或描述）$\alpha + \beta$ 随 $c$ 变化的趋势，说明两类错误不能同时最小化（但增大 $n$ 可同时减小两者）。

---

**D.7.4**（Ch.20，方差的 $\chi^2$ 检验）
某产品标准差 $\sigma_0 = 2$，抽取 $n = 25$，$S^2 = 5.8$。检验 $H_0: \sigma^2 = 4$ vs $H_1: \sigma^2 > 4$，$\alpha = 0.05$。

(a) 写出检验统计量 $\chi^2 = (n-1)S^2/\sigma_0^2$ 并计算。
(b) 查 $\chi^2(24)$ 分布的 0.95 分位数（$\chi^2_{0.05}(24) \approx 36.42$），做出决策。
(c) 若改为双侧检验，写出拒绝域（需两端分位数 $\chi^2_{0.025}(24)$ 和 $\chi^2_{0.975}(24)$）。

---

**D.7.5**（Ch.20，两均值 $t$ 检验，等方差）
两组数据：$n_1 = 8$，$\bar{X}_1 = 20.5$，$S_1^2 = 6$；$n_2 = 10$，$\bar{X}_2 = 18.0$，$S_2^2 = 7$。假设等方差，检验 $H_0: \mu_1 = \mu_2$ vs $H_1: \mu_1 \neq \mu_2$，$\alpha = 0.05$。

(a) 计算合并方差 $S_p^2$。
(b) 计算 $t$ 统计量，自由度 $df = n_1 + n_2 - 2 = 16$，$t_{0.025}(16) \approx 2.120$，做出决策。
(c) 在做双均值检验前，应先检验等方差假设，描述 $F$ 检验的步骤。

---

**D.7.6**（Ch.20，配对 $t$ 检验）
10 名受试者治疗前后血压（mmHg）：

| 受试者 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|--------|---|---|---|---|---|---|---|---|---|---|
| 前 | 120 | 135 | 128 | 142 | 118 | 130 | 125 | 138 | 122 | 131 |
| 后 | 115 | 129 | 124 | 137 | 120 | 128 | 122 | 134 | 120 | 127 |

(a) 计算差值 $D_i = $ 前 $-$ 后，求 $\bar{D}$ 和 $S_D$（列出计算步骤）。
(b) 检验 $H_0: \mu_D = 0$ vs $H_1: \mu_D > 0$，$\alpha = 0.05$，$t_{0.05}(9) \approx 1.833$。
(c) 说明为何配对设计比独立双样本设计更有效率（减少个体间变异）。

---

**D.7.7**（Ch.19，Neyman-Pearson 引理）
设 $X \sim N(\mu, 1)$，检验 $H_0: \mu = 0$ vs $H_1: \mu = 1$，单次观测。

(a) 写出似然比 $\Lambda(x) = f(x;1)/f(x;0)$，化简后说明拒绝域等价于 $X > c$。
(b) 由 Neyman-Pearson 引理，这是大小为 $\alpha$ 的最优势检验，说明"最优势"的含义。
(c) 若检验为复合 $H_1: \mu > 0$，N-P 引理是否直接适用？简述一致最优势检验（UMP）的存在条件。

---

**D.7.8**（Ch.21，Wilcoxon 符号秩检验）
$n = 8$ 个差值（配对前后）：$+3, -1, +5, +2, -4, +1, +6, +2$。检验 $H_0$：中位数为 0（双侧，$\alpha = 0.05$）。

(a) 对 $|D_i|$ 排秩，然后按原差值正负分配正负秩。
(b) 计算正秩和 $T^+$ 和负秩和 $T^-$。
(c) 查 Wilcoxon 符号秩表（$n=8$，$\alpha=0.05$ 双侧，临界值 $W = 4$），做出决策，并与参数 $t$ 检验比较。

---

**D.7.9**（Ch.20，单因素方差分析 ANOVA）
三组数据（$n_i = 5$）：

| 组 1 | 组 2 | 组 3 |
|------|------|------|
| 12, 14, 11, 13, 10 | 20, 22, 19, 21, 18 | 15, 17, 16, 14, 18 |

(a) 计算各组均值 $\bar{X}_1, \bar{X}_2, \bar{X}_3$ 和总均值 $\bar{X}$。
(b) 计算组间 $SS_B = \sum_i n_i(\bar{X}_i - \bar{X})^2$ 和组内 $SS_W = \sum_{i,j}(X_{ij}-\bar{X}_i)^2$，以及对应均方。
(c) 计算 $F = MS_B/MS_W$，查 $F(2, 12)$ 分布的 0.05 分位数（$F_{0.05}(2,12) \approx 3.89$），做出决策。

---

**D.7.10**（Ch.21，卡方拟合优度检验）
掷骰子 120 次，各面出现次数：18, 22, 17, 25, 19, 19。检验骰子是否均匀（$\alpha = 0.05$）。

(a) 写出理论频数（每面期望 20 次）和检验统计量 $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$。
(b) 计算 $\chi^2$ 统计量，自由度 $df = 6 - 1 = 5$，$\chi^2_{0.05}(5) \approx 11.07$，做出决策。
(c) 解释 $p$ 值的含义，并讨论若某面期望次数小于 5 时需如何处理（合并格子）。

---

**D.7.11**（Ch.21，独立性卡方检验）
调查 200 人，按性别与是否吸烟列联表如下：

| | 吸烟 | 不吸烟 | 合计 |
|---|------|--------|------|
| 男 | 60 | 40 | 100 |
| 女 | 30 | 70 | 100 |
| 合计 | 90 | 110 | 200 |

(a) 计算各单元格的期望频数 $E_{ij} = R_i C_j / n$。
(b) 计算 $\chi^2$ 统计量，$df = (2-1)(2-1) = 1$，$\chi^2_{0.05}(1) \approx 3.84$，做出决策。
(c) 计算 Phi 系数 $\phi = \sqrt{\chi^2/n}$，解释其作为效应量的含义。

---

**D.7.12**（Ch.19，多重检验问题）
同时检验 $m = 20$ 个独立假设，均为真（$H_0$ 全真），每个用 $\alpha = 0.05$。

(a) 计算至少出现一次第一类错误（假阳性）的概率（家族误差率 FWER）。
(b) Bonferroni 校正：将每个检验的水平调为 $\alpha^* = 0.05/20$，求 FWER 的上界。
(c) 若 20 个假设中有 5 个为假，描述 Benjamini-Hochberg 程序的思路（控制 FDR），并说明比 Bonferroni 更宽松的原因。

---

## Part 8 高级主题（Ch.22-24，共 12 题）

> 覆盖信息论基础（Ch.22）、蒙特卡洛方法（Ch.23）、概率图模型（Ch.24）。
> 题型：熵与 KL 散度证明、互信息与信道容量、MC 积分误差、重要性采样、M-H 算法、贝叶斯网络 d-分离、马尔可夫链平稳分布、GP 回归、变分推断 ELBO。

**D.8.1**（Ch.22，信息熵的基本性质）
设离散分布 $p = (p_1, \ldots, p_n)$，Shannon 熵 $H(p) = -\sum_i p_i \log p_i$。

(a) 证明 $H(p) \geq 0$（利用 $p_i \log p_i \leq 0$）。
(b) 证明均匀分布最大化熵：$H(p) \leq \log n$（用 Jensen 不等式，$\log$ 是凹函数）。
(c) 计算 $\mathrm{Bernoulli}(p)$ 的熵，求使熵最大的 $p$，并验证等于 $\log 2$。

---

**D.8.2**（Ch.22，KL 散度）
设两个分布 $P$ 和 $Q$，KL 散度 $D_{KL}(P \| Q) = \sum_i p_i \log(p_i/q_i)$。

(a) 证明 $D_{KL}(P \| Q) \geq 0$（用 Jensen 不等式，$-\log$ 是凸函数）。
(b) 验证 $D_{KL}(P \| Q) = 0 \Leftrightarrow P = Q$。
(c) 对 $P = \mathrm{Bernoulli}(0.7)$，$Q = \mathrm{Bernoulli}(0.5)$，计算 $D_{KL}(P \| Q)$ 和 $D_{KL}(Q \| P)$，验证不对称性。

---

**D.8.3**（Ch.22，互信息与独立性）
设 $(X, Y)$ 的联合分布为 $p(x,y)$，互信息 $I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$。

(a) 证明 $I(X;Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X)$。
(b) 证明 $I(X;Y) \geq 0$，等号成立 $\Leftrightarrow$ $X \perp Y$（用 KL 散度非负性）。
(c) 对二元对称信道（$P(Y=0\mid X=0) = P(Y=1\mid X=1) = 1-\varepsilon$，输入均匀），计算 $I(X;Y)$（信道容量）。

---

**D.8.4**（Ch.23，Monte Carlo 积分误差）
用 Monte Carlo 方法估计 $I = \int_0^1 g(x) \, dx$：$\hat{I}_n = \frac{1}{n}\sum_{i=1}^n g(U_i)$，$U_i \overset{\text{i.i.d.}}{\sim} U(0,1)$。

(a) 证明 $\hat{I}_n$ 是无偏的，$\mathrm{Var}(\hat{I}_n) = \mathrm{Var}(g(U))/n$。
(b) 由 CLT，$\hat{I}_n$ 的近似 95% CI 为 $\hat{I}_n \pm 1.96 \hat{\sigma}/\sqrt{n}$，其中 $\hat{\sigma}^2 = \frac{1}{n-1}\sum(g(U_i)-\hat{I}_n)^2$。对 $g(x) = e^x$（精确值 $e - 1 \approx 1.718$），要求 CI 宽度 $\leq 0.01$，估计需要的样本量。
(c) 对比数值积分（如 Simpson 法则）与 MC 积分在高维下的精度衰减，说明 MC 的维度优势。

---

**D.8.5**（Ch.23，重要性采样）
设目标函数 $g(x)$，目标分布 $p(x)$，用提议分布 $q(x)$ 采样，重要性采样估计 $E_p[g(X)] = E_q\!\left[g(X)\frac{p(X)}{q(X)}\right]$。

(a) 证明重要性采样估计量 $\hat{I}_{IS} = \frac{1}{n}\sum_{i=1}^n g(X_i) w(X_i)$ 是无偏的，其中权重 $w(x) = p(x)/q(x)$，$X_i \sim q$。
(b) 写出 $\mathrm{Var}(\hat{I}_{IS})$ 的表达式，说明当 $q \propto |g| p$ 时方差最小（最优提议分布）。
(c) 描述自归一化重要性采样（SNIS）的步骤，说明在 $p$ 不可归一化时的优势。

---

**D.8.6**（Ch.23，Metropolis-Hastings 算法）
目标分布 $\pi(x) \propto e^{-x^2/2}$（标准正态），提议分布 $q(x'\mid x) = \mathrm{Uniform}(x - \delta, x + \delta)$。

(a) 写出 M-H 接受率 $\alpha(x, x') = \min\!\left(1, \frac{\pi(x')q(x\mid x')}{\pi(x)q(x'\mid x)}\right)$，化简（由于 $q$ 对称，接受率简化为 $\min(1, \pi(x')/\pi(x))$）。
(b) 验证 M-H 满足细致平衡条件 $\pi(x)\alpha(x,x')q(x'\mid x) = \pi(x')\alpha(x',x)q(x\mid x')$。
(c) 讨论步长 $\delta$ 对混合速度的影响：$\delta$ 过大或过小各有什么问题？最优 $\delta$ 使接受率约为多少？

---

**D.8.7**（Ch.24，贝叶斯网络的条件独立性）
设贝叶斯网络：$A \to C \leftarrow B$，$C \to D$（V 形结构加链）。

(a) 写出联合分布的分解：$P(A, B, C, D) = P(A)P(B)P(C\mid A,B)P(D\mid C)$。
(b) 在不观测 $C$ 时，证明 $A \perp B$（$C$ 是 collider，未观测时阻断）。
(c) 若观测 $C = c$，说明 $A$ 和 $B$ 不再独立（explaining away 效应），举例说明。

---

**D.8.8**（Ch.24，马尔可夫链基本性质）
设有限状态马尔可夫链，状态空间 $\{1, 2, 3\}$，转移矩阵 $P = \begin{pmatrix} 0.7 & 0.2 & 0.1 \\ 0.3 & 0.5 & 0.2 \\ 0.1 & 0.3 & 0.6 \end{pmatrix}$。

(a) 验证每行之和为 1（随机矩阵验证）。
(b) 求平稳分布 $\boldsymbol{\pi}$：解线性方程组 $\boldsymbol{\pi} P = \boldsymbol{\pi}$，$\boldsymbol{\pi} \mathbf{1} = 1$。
(c) 说明此马尔可夫链是否遍历（不可约且非周期），并说明遍历性保证 $P^n \to \mathbf{1}\boldsymbol{\pi}^\top$（收敛到平稳分布）。

---

**D.8.9**（Ch.22，最大熵原理）
设离散随机变量 $X \in \{1, 2, \ldots, n\}$，约束为 $E[X] = \mu$（给定均值）。

(a) 用 Lagrange 乘数法最大化 $H(p) = -\sum p_i \log p_i$，约束 $\sum p_i = 1$ 和 $\sum i \cdot p_i = \mu$，得到最大熵分布 $p_i \propto e^{\lambda i}$（几何分布族）。
(b) 当约束为 $E[X] = \mu$ 且 $\mathrm{Var}(X) = \sigma^2$（连续情形），最大熵分布为正态分布（陈述结论，不推导）。
(c) 解释最大熵原理的统计物理含义（信息量最少假设的先验选择）。

---

**D.8.10**（Ch.23，变分推断基本思想）
目标：近似后验 $p(\mathbf{z} \mid \mathbf{x})$，用参数族 $q_\phi(\mathbf{z})$ 近似，最小化 $D_{KL}(q_\phi \| p(\mathbf{z} \mid \mathbf{x}))$。

(a) 推导证据下界（ELBO）：$\mathcal{L}(\phi) = E_q[\log p(\mathbf{x}, \mathbf{z})] - E_q[\log q_\phi(\mathbf{z})]$，使得 $\log p(\mathbf{x}) \geq \mathcal{L}(\phi)$。
(b) 说明最大化 ELBO 等价于最小化 $D_{KL}(q_\phi \| p(\mathbf{z} \mid \mathbf{x}))$（推导等价关系）。
(c) 对均场近似 $q(\mathbf{z}) = \prod_i q_i(z_i)$，写出坐标上升 VI 的更新公式 $\log q_j^*(z_j) = E_{-j}[\log p(\mathbf{x}, \mathbf{z})] + \text{const}$。

---

**D.8.11**（Ch.24，隐马尔可夫模型前向算法）
设 HMM 有隐状态 $\{S_1, S_2\}$，初始分布 $\pi$，转移矩阵 $A$，发射矩阵 $B$，观测序列 $O = (o_1, o_2, o_3)$。

(a) 定义前向变量 $\alpha_t(i) = P(o_1, \ldots, o_t, S_t = i \mid \lambda)$，写出初始化和递推公式。
(b) 用前向变量表达似然 $P(O \mid \lambda) = \sum_i \alpha_T(i)$（求和消去最终隐状态）。
(c) 与暴力枚举所有路径（$2^T$ 条路径）相比，前向算法的计算复杂度是多少（$O(N^2 T)$），解释其动态规划本质。

---

**D.8.12**（Ch.24，高斯过程回归简介）
设高斯过程 $f \sim \mathcal{GP}(m(\cdot), k(\cdot, \cdot))$，观测 $\mathbf{y} = f(\mathbf{X}) + \boldsymbol{\varepsilon}$，$\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma_n^2 \mathbf{I})$。

(a) 写出观测向量 $\mathbf{y}$ 的边际分布（多元正态，均值和协方差矩阵）。
(b) 写出后验预测分布 $f^* \mid \mathbf{X}, \mathbf{y}, \mathbf{x}^*$ 的均值和方差公式（GP 回归的闭式后验）。
(c) 说明 GP 回归与核岭回归（kernel ridge regression）的等价性（在贝叶斯视角下，预测均值与 KRR 解一致）。

---

---

## 汇总

> **难度定位**：D 级（中等）介于 C 级（基础，单步套公式）与 E 级（综合，多知识点融合/研究生考题）之间。本题库各 Part 均匀分布多步计算、简单证明、应用题与反例构造四类题型，适合系统复习和课后专项训练。

| Part | 章节 | 题数 |
|------|------|------|
| Part 1 概率基础 | Ch.1-3 | 12 |
| Part 2 随机变量 | Ch.4-6 | 15 |
| Part 3 分布 | Ch.7-9 | 15 |
| Part 4 极限定理 | Ch.10-12 | 10 |
| Part 5 统计基础 | Ch.13-15 | 10 |
| Part 6 估计 | Ch.16-18 | 14 |
| Part 7 假设检验 | Ch.19-21 | 12 |
| Part 8 高级主题 | Ch.22-24 | 12 |
| **合计** | | **100** |
