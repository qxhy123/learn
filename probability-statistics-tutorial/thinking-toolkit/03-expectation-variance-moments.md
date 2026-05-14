# 期望 / 方差 / 矩

> **一例速记**：$X \sim B(n, p)$，用"分解为 Bernoulli 之和"的技巧，期望 $E(X) = np$，方差 $\text{Var}(X) = np(1-p)$。
> 若直接用定义展开 $E(X) = \sum_{k=0}^n k\binom{n}{k} p^k(1-p)^{n-k}$，需要用到组合恒等式；
> 若先拆 $X = X_1 + \cdots + X_n$（$X_i \sim B(1,p)$），则 $E(X) = nE(X_1) = np$，方差（独立时可加）$\text{Var}(X) = n\text{Var}(X_1) = np(1-p)$，一行搞定。
> **拆分 + 线性性是期望计算的核武器。**

---

## 一、为什么需要期望和矩

随机变量 $X$ 的概率分布（PMF 或 PDF）包含了它的"全部信息"，但这个信息往往过于详细——我们有时只需要若干关键数字来刻画分布的"重心"、"散布程度"和"形态"。

**矩**（moments）正是把分布压缩为几个数字的标准工具：
- 一阶矩（期望）$E(X)$：分布的"重心"，衡量平均水平；
- 二阶矩 $E(X^2)$ 和方差 $\text{Var}(X)$：衡量散布；
- 三阶中心矩：衡量偏斜（skewness）；
- 四阶中心矩：衡量尖峰程度（kurtosis）。

矩母函数（Moment Generating Function, MGF）则把所有矩都打包进一个函数，是强大的分析工具。

本篇系统整理期望、方差、协方差、相关系数和 MGF，并用二项分布作演示题对比"定义法"和"MGF 法"。

---

## 二、期望

### 2.1 定义

**离散随机变量**：设 $X$ 的 PMF 为 $P(X = x_i) = p_i$（$i = 1, 2, \ldots$），若 $\sum_i |x_i| p_i < \infty$，定义期望

$$E(X) = \sum_i x_i p_i.$$

**连续随机变量**：设 $X$ 的 PDF 为 $f(x)$，若 $\int_{-\infty}^{+\infty} |x| f(x)\,dx < \infty$，定义期望

$$E(X) = \int_{-\infty}^{+\infty} x f(x)\,dx.$$

**注意绝对可积条件**：若 $\sum_i |x_i| p_i = +\infty$（如 Cauchy 分布），期望不存在。

### 2.2 期望的线性性

这是期望最重要的性质，**对任意随机变量**（不需要独立）：

$$E(aX + bY + c) = a E(X) + b E(Y) + c \quad (a, b, c \in \mathbb{R}).$$

推广：对 $n$ 个随机变量（无论是否独立）：

$$E\!\left(\sum_{i=1}^n a_i X_i + c\right) = \sum_{i=1}^n a_i E(X_i) + c.$$

**非线性函数的期望**：一般 $E[g(X)] \neq g(E[X])$。例如 $E(X^2) \neq (E[X])^2$（二者之差正是方差）。正确写法：

$$E[g(X)] = \sum_i g(x_i) p_i \quad \text{（离散）}, \qquad E[g(X)] = \int_{-\infty}^{+\infty} g(x) f(x)\,dx \quad \text{（连续）}.$$

这被称为**无意识统计学家法则**（Law of the Unconscious Statistician, LOTUS）。

### 2.3 乘积的期望（独立时）

若 $X$ 与 $Y$ **独立**，则：
$$E(XY) = E(X) \cdot E(Y).$$

反向不成立：$E(XY) = E(X)E(Y)$ 不蕴含独立（只蕴含"不相关"，弱于独立）。

---

## 三、方差

### 3.1 定义与计算公式

**定义**：
$$\text{Var}(X) = E\!\left[(X - E[X])^2\right].$$

**展开计算公式**（最常用）：
$$\text{Var}(X) = E(X^2) - (E[X])^2.$$

**推导**：
$$E[(X - \mu)^2] = E[X^2 - 2\mu X + \mu^2] = E(X^2) - 2\mu E(X) + \mu^2 = E(X^2) - \mu^2.$$

其中 $\mu = E(X)$。

**标准差**：$\text{SD}(X) = \sqrt{\text{Var}(X)}$，与 $X$ 同单位，常用 $\sigma$ 表示。

### 3.2 方差的性质

| 性质 | 公式 | 条件 |
|---|---|---|
| 常数的方差 | $\text{Var}(c) = 0$ | $c$ 为常数 |
| 线性变换 | $\text{Var}(aX + b) = a^2 \text{Var}(X)$ | 加常数不影响散布；乘系数平方放缩 |
| 加性（独立） | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ | $X, Y$ 独立 |
| 一般加性 | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$ | 一般情形 |

注意：**方差不具有线性性**（$\text{Var}(X+Y) \neq \text{Var}(X) + \text{Var}(Y)$ 一般情形），这与期望不同。

---

## 四、协方差与相关系数

### 4.1 协方差

**定义**：
$$\text{Cov}(X, Y) = E\!\left[(X - E[X])(Y - E[Y])\right] = E(XY) - E(X) E(Y).$$

**性质**：

| 性质 | 公式 |
|---|---|
| 对称性 | $\text{Cov}(X, Y) = \text{Cov}(Y, X)$ |
| 自协方差 | $\text{Cov}(X, X) = \text{Var}(X)$ |
| 双线性 | $\text{Cov}(aX + bY, Z) = a\,\text{Cov}(X, Z) + b\,\text{Cov}(Y, Z)$ |
| 独立则不相关 | $X \perp Y \Rightarrow \text{Cov}(X, Y) = 0$ |

**注意**：$\text{Cov}(X, Y) = 0$（不相关）**不蕴含**独立（正态分布的例外：不相关 $\Rightarrow$ 独立）。

### 4.2 相关系数

**定义**：

$$\rho_{XY} = \text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)} \cdot \sqrt{\text{Var}(Y)}} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}.$$

**性质**：

1. $-1 \leq \rho \leq 1$（Cauchy-Schwarz 不等式保证）；
2. $\rho = 1$：完全正线性相关（$Y = aX + b$，$a > 0$）；
3. $\rho = -1$：完全负线性相关（$Y = aX + b$，$a < 0$）；
4. $\rho = 0$：线性不相关（但可能存在非线性关系）。

**相关系数衡量线性相关程度**，不反映非线性关系。

---

## 五、矩母函数（MGF）

### 5.1 定义

**定义**：若期望 $E(e^{tX})$ 在 $t = 0$ 的某邻域内有限，称

$$M_X(t) = E(e^{tX})$$

为随机变量 $X$ 的**矩母函数**（MGF）。

**离散情形**：$M_X(t) = \sum_i e^{t x_i} p_i$；  
**连续情形**：$M_X(t) = \int_{-\infty}^{+\infty} e^{tx} f(x)\,dx$。

### 5.2 为什么叫"矩母函数"

将 $e^{tX}$ 展开为 Taylor 级数（在 $t = 0$ 处）：

$$e^{tX} = \sum_{n=0}^\infty \frac{(tX)^n}{n!} = 1 + tX + \frac{t^2 X^2}{2!} + \cdots$$

取期望：

$$M_X(t) = E(e^{tX}) = \sum_{n=0}^\infty \frac{t^n}{n!} E(X^n).$$

对 $t$ 求 $n$ 次导，令 $t = 0$，得**第 $n$ 阶矩**：

$$M_X^{(n)}(0) = E(X^n).$$

特别地，$M_X'(0) = E(X)$，$M_X''(0) = E(X^2)$，从而 $\text{Var}(X) = M_X''(0) - [M_X'(0)]^2$。

### 5.3 MGF 的重要性质

| 性质 | 公式 | 应用 |
|---|---|---|
| 矩的提取 | $E(X^n) = M_X^{(n)}(0)$ | 快速计算各阶矩 |
| 线性变换 | $M_{aX+b}(t) = e^{bt} M_X(at)$ | 标准化等变换 |
| 独立和 | $M_{X+Y}(t) = M_X(t) M_Y(t)$（$X, Y$ 独立） | 和分布的推导 |
| 唯一性 | MGF（若存在）唯一确定分布 | 识别分布类型 |

---

## 六、演示题：二项分布 $B(n,p)$ 的期望与方差——两种方法

### 题目

设 $X \sim B(n, p)$，PMF 为 $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$（$k = 0, 1, \ldots, n$）。用两种方法求 $E(X)$ 和 $\text{Var}(X)$。

### 方法一：分解法（拆为 Bernoulli 之和）

> 直接对 $\sum_{k=0}^n k\binom{n}{k} p^k q^{n-k}$ 求和需要用到组合恒等式。有更聪明的做法：先观察 $B(n,p)$ 的"起源"。
>
> $X \sim B(n, p)$ 可以理解为 $n$ 次独立 Bernoulli 试验的成功次数，即 $X = X_1 + X_2 + \cdots + X_n$，其中 $X_i \sim B(1, p)$（每次试验是否成功，成功为 1，失败为 0），$X_i$ 相互独立。
>
> **计算 $E(X_i)$**：$E(X_i) = 1 \cdot p + 0 \cdot (1-p) = p$。
>
> **计算 $E(X)$**：由期望的线性性（不需独立）：
> $$E(X) = E(X_1 + \cdots + X_n) = \sum_{i=1}^n E(X_i) = np.$$
>
> **计算 $\text{Var}(X_i)$**：$E(X_i^2) = 1^2 \cdot p + 0^2 \cdot (1-p) = p$，所以
> $$\text{Var}(X_i) = E(X_i^2) - [E(X_i)]^2 = p - p^2 = p(1-p).$$
>
> **计算 $\text{Var}(X)$**：$X_i$ 相互独立，方差可加：
> $$\text{Var}(X) = \sum_{i=1}^n \text{Var}(X_i) = n \cdot p(1-p).$$
>
> 整个推导只用了"期望线性性"和"独立时方差可加"，没有繁琐的组合运算。

### 方法二：MGF 法

> 先求 $X_i \sim B(1, p)$ 的 MGF：
> $$M_{X_i}(t) = E(e^{tX_i}) = e^{t \cdot 0}(1-p) + e^{t \cdot 1} p = (1-p) + pe^t = q + pe^t,$$
> 其中 $q = 1-p$。
>
> 因 $X = X_1 + \cdots + X_n$ 且各 $X_i$ 独立，MGF 相乘：
> $$M_X(t) = \prod_{i=1}^n M_{X_i}(t) = (q + pe^t)^n.$$
>
> **提取期望**：对 $t$ 求一阶导：
> $$M_X'(t) = n(q + pe^t)^{n-1} \cdot pe^t.$$
> 令 $t = 0$：
> $$E(X) = M_X'(0) = n(q + p)^{n-1} \cdot p = n \cdot 1 \cdot p = np. \quad \checkmark$$
>
> **提取二阶矩**：对 $t$ 求二阶导：
> $$M_X''(t) = n(n-1)(q+pe^t)^{n-2}(pe^t)^2 + n(q+pe^t)^{n-1} pe^t.$$
> 令 $t = 0$：
> $$E(X^2) = M_X''(0) = n(n-1)p^2 + np.$$
>
> **计算方差**：
> $$\text{Var}(X) = E(X^2) - [E(X)]^2 = n(n-1)p^2 + np - n^2p^2$$
> $$= n^2p^2 - np^2 + np - n^2p^2 = np - np^2 = np(1-p). \quad \checkmark$$
>
> 两种方法结果一致：$E(X) = np$，$\text{Var}(X) = np(1-p)$。
>
> **方法比较**：分解法更直观简洁；MGF 法更具通用性（对不能轻易分解的分布也适用），也可以一次性提取任意阶矩。

---

## 七、思考路标

1. **计算期望的首选策略** → 先看能否用线性性拆分（$X = X_1 + \cdots + X_n$），再看有无 MGF 可用，最后才直接用定义硬算。

2. **$E[g(X)]$ 的计算** → 不要"先求分布再求期望"，用 LOTUS 直接 $E[g(X)] = \int g(x)f(x)\,dx$（连续）或 $\sum g(x_i)p_i$（离散）。

3. **方差与期望的关系** → 计算方差优先用 $\text{Var}(X) = E(X^2) - (EX)^2$，比直接展开 $(X - \mu)^2$ 取期望更简洁。关键是先求 $E(X)$ 和 $E(X^2)$，两步分开处理。

4. **方差不具线性性** → 若 $X, Y$ 不独立，$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$。若题目没说独立，不能直接相加。

5. **协方差 = 0 不代表独立** → 不相关只是弱于独立的条件。若分布是多元正态，不相关才等价于独立。其他情形需要独立性的完整验证。

6. **MGF 不存在的情形** → 重尾分布（如 Cauchy、$t_1$）的 MGF 不存在，这时用特征函数 $\phi(t) = E(e^{itX})$（复值，$i$ 虚数单位），总是存在。

7. **MGF 唯一确定分布** → 若两个分布有相同 MGF，则它们是同一分布。这常用于：已知分布的 MGF 形式，将待求分布的 MGF 化成已知形式，从而识别分布类型。

8. **标准化随机变量** → 令 $Z = (X - \mu)/\sigma$，则 $E(Z) = 0$，$\text{Var}(Z) = 1$。标准化不改变分布形状，但把均值移到 0，把标准差化为 1，便于比较不同量纲的随机变量。

---

## 八、典型应用 3 例

### 例 1：几何分布的期望（无穷级数法）

**题目**：$X \sim \text{Geom}(p)$，PMF 为 $P(X = k) = (1-p)^{k-1} p$（$k = 1, 2, \ldots$）。求 $E(X)$。

**思路**：

$$E(X) = \sum_{k=1}^\infty k (1-p)^{k-1} p = p \sum_{k=1}^\infty k q^{k-1}, \quad q = 1-p.$$

利用公式 $\sum_{k=1}^\infty k q^{k-1} = \dfrac{1}{(1-q)^2} = \dfrac{1}{p^2}$（对几何级数 $\sum q^k$ 求导）：

$$E(X) = p \cdot \frac{1}{p^2} = \frac{1}{p}.$$

**直觉**：平均需要 $1/p$ 次试验才能成功一次（成功概率 $p$），符合直觉。

---

### 例 2：Poisson 分布的方差

**题目**：$X \sim \text{Poisson}(\lambda)$，PMF 为 $P(X = k) = e^{-\lambda}\lambda^k/k!$（$k = 0, 1, \ldots$）。求 $E(X)$ 和 $\text{Var}(X)$。

**思路**（MGF 法）：

$$M_X(t) = E(e^{tX}) = \sum_{k=0}^\infty e^{tk} \cdot \frac{e^{-\lambda}\lambda^k}{k!} = e^{-\lambda} \sum_{k=0}^\infty \frac{(\lambda e^t)^k}{k!} = e^{-\lambda} e^{\lambda e^t} = e^{\lambda(e^t - 1)}.$$

一阶导：$M_X'(t) = \lambda e^t \cdot e^{\lambda(e^t-1)}$，令 $t = 0$：$E(X) = \lambda$.

二阶导：$M_X''(t) = (\lambda e^t)^2 e^{\lambda(e^t-1)} + \lambda e^t e^{\lambda(e^t-1)}$，令 $t = 0$：$E(X^2) = \lambda^2 + \lambda$.

$$\text{Var}(X) = E(X^2) - (EX)^2 = (\lambda^2 + \lambda) - \lambda^2 = \lambda.$$

**结论**：Poisson 分布的期望和方差相等，都等于 $\lambda$——这是识别 Poisson 分布的特征性质。

---

### 例 3：线性组合的方差——用协方差

**题目**：已知 $E(X) = 2$，$E(Y) = 3$，$\text{Var}(X) = 4$，$\text{Var}(Y) = 9$，$\text{Cov}(X, Y) = -2$。求 $\text{Var}(2X - Y + 1)$。

**思路**：

$$\text{Var}(2X - Y + 1) = \text{Var}(2X - Y) = 4\text{Var}(X) + \text{Var}(Y) - 2 \cdot 2 \cdot \text{Cov}(X, Y)$$
$$= 4 \times 4 + 9 - 4 \times (-2) = 16 + 9 + 8 = 33.$$

注意：加常数 $+1$ 不影响方差；$\text{Var}(aX + bY) = a^2\text{Var}(X) + b^2\text{Var}(Y) + 2ab\,\text{Cov}(X,Y)$，系数 $a=2, b=-1$。

---

## 九、自测题

**第 1 题**：设 $X$ 取 $\{1, 2, 3\}$，概率分别为 $1/2, 1/3, 1/6$，求 $E(X)$ 和 $\text{Var}(X)$。

提示：$E(X) = 1 \cdot \frac{1}{2} + 2 \cdot \frac{1}{3} + 3 \cdot \frac{1}{6} = \frac{1}{2} + \frac{2}{3} + \frac{1}{2} = \frac{5}{3}$；$E(X^2) = 1 \cdot \frac{1}{2} + 4 \cdot \frac{1}{3} + 9 \cdot \frac{1}{6} = \frac{1}{2} + \frac{4}{3} + \frac{3}{2} = \frac{10}{3}$；$\text{Var}(X) = \frac{10}{3} - \left(\frac{5}{3}\right)^2 = \frac{10}{3} - \frac{25}{9} = \frac{5}{9}$。

---

**第 2 题**：$X \sim U(0, 1)$（均匀分布），$f(x) = 1$（$0 < x < 1$）。求 $E(X)$，$E(X^2)$，$\text{Var}(X)$。

提示：$E(X) = \int_0^1 x\,dx = 1/2$；$E(X^2) = \int_0^1 x^2\,dx = 1/3$；$\text{Var}(X) = 1/3 - 1/4 = 1/12$。

---

**第 3 题**：设 $X_1, \ldots, X_{100}$ 独立同分布，$E(X_i) = 5$，$\text{Var}(X_i) = 9$，令 $\bar{X} = \frac{1}{100}\sum_{i=1}^{100} X_i$，求 $E(\bar{X})$ 和 $\text{Var}(\bar{X})$。

提示：$E(\bar{X}) = 5$（线性性）；$\text{Var}(\bar{X}) = \frac{1}{100^2} \sum \text{Var}(X_i) = \frac{100 \times 9}{100^2} = \frac{9}{100} = 0.09$。

---

**第 4 题**：利用 MGF 证明：若 $X \sim N(\mu, \sigma^2)$，其 MGF 为 $M_X(t) = e^{\mu t + \sigma^2 t^2/2}$，则 $E(X) = \mu$，$\text{Var}(X) = \sigma^2$。

提示：$M_X'(t) = (\mu + \sigma^2 t) e^{\mu t + \sigma^2 t^2 / 2}$，$t=0$ 时 $= \mu$；$M_X''(t)$ 在 $t=0$ 时等于 $\mu^2 + \sigma^2$；$\text{Var} = (\mu^2+\sigma^2) - \mu^2 = \sigma^2$。

---

**第 5 题**：设 $X, Y$ 相关系数 $\rho = -0.8$，$\text{Var}(X) = 4$，$\text{Var}(Y) = 9$，求 $\text{Cov}(X, Y)$ 和 $\text{Var}(X + Y)$。

提示：$\text{Cov}(X,Y) = \rho \sigma_X \sigma_Y = (-0.8)(2)(3) = -4.8$；$\text{Var}(X+Y) = 4 + 9 + 2(-4.8) = 13 - 9.6 = 3.4$。
