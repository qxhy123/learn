# 微积分中的不等式技巧

> **一例速记**：证明 $e^x \geq 1 + x$（对所有实数 $x$）。
> 构造 $h(x) = e^x - 1 - x$，$h'(x) = e^x - 1$：$h' < 0$（$x < 0$），$h'(0) = 0$，$h' > 0$（$x > 0$）。所以 $x = 0$ 是 $h$ 的全局最小值点，$h(0) = 0$，故 $h(x) \geq 0$，即 $e^x \geq 1 + x$，等号仅当 $x = 0$ 成立。
> **"构造辅助函数 + 导数判最小值"是微积分证不等式的第一套路。**

---

## 一、为什么不等式是微积分的基石

不等式在微积分中无处不在：极限的 $\varepsilon$-$\delta$ 定义依赖绝对值不等式，积分估计依赖被积函数的上下界，收敛性判别依赖级数大小比较，优化理论依赖凸性不等式。可以说，**微积分的精确性就是用一套严密的不等式体系来保证的**。

学会证不等式的几类核心技巧，不仅能应对考试，更能在读书推导、理解证明时"看懂每一步为什么成立"。

本篇系统介绍微积分中四大不等式技巧，并附上三巨头（Hardy / Hölder / Minkowski）的简介，作为进一步学习的入口。

---

## 二、四大技巧

### 技巧 1：单调性证不等式

**核心思路**：要证 $f(x) \geq g(x)$（对 $x \in I$），构造辅助函数 $h(x) = f(x) - g(x)$，通过求导证明 $h$ 在 $I$ 上单调（或证明 $h$ 在某点取最小值且最小值 $\geq 0$）。

**标准流程**：

**Step 1** — 构造 $h(x) = f(x) - g(x)$，目标是证 $h(x) \geq 0$（或 $> 0$）。

**Step 2** — 求 $h'(x)$，判断符号：
- 若 $h'(x) \geq 0$ 在整个区间 $I$ 上成立：$h$ 单调递增，取左端点值 $h(a) \geq 0$ 即可（若 $h(a) = 0$ 则结论严格大于在内部成立）。
- 若 $h$ 先减后增（有极小值点 $x_0$）：证明 $h(x_0) \geq 0$ 即可（极小值 $\geq 0 \Rightarrow$ 全局 $\geq 0$）。

**Step 3** — 验证端点或极值点处的等号条件，写出等号成立的充要条件。

**关键判断**：用这个技巧时，需要 $h'(x)$ 比 $h(x)$ 本身更容易分析符号——如果 $h'$ 还是很复杂，考虑再求一次导（即分析 $h''$），甚至更高阶。

**经典不等式（单调性证法）**：

| 不等式 | 辅助函数 $h$ | 分析 |
|---|---|---|
| $e^x \geq 1 + x$ | $e^x - 1 - x$ | $h' = e^x - 1$，$h'(0)=0$，极小值 $h(0)=0$ |
| $\ln x \leq x - 1$（$x > 0$） | $\ln x - x + 1$ | $h' = 1/x - 1$，极大值 $h(1) = 0$ |
| $\sin x < x$（$x > 0$） | $x - \sin x$ | $h' = 1 - \cos x \geq 0$，$h(0)=0$，故 $h(x) \geq 0$ |
| $\tan x > x$（$0 < x < \pi/2$） | $\tan x - x$ | $h' = \sec^2 x - 1 = \tan^2 x > 0$，$h(0)=0$ |

---

### 技巧 2：凸性 + Jensen 不等式

**核心思路**：若函数 $f$ 是凸函数（$f'' \geq 0$），则其满足 Jensen 不等式：对任意 $\lambda_i \geq 0$，$\sum \lambda_i = 1$：

$$f\!\left(\sum_{i=1}^n \lambda_i x_i\right) \leq \sum_{i=1}^n \lambda_i f(x_i)$$

取 $\lambda_i = 1/n$ 就是常用的离散均值版本：

$$f\!\left(\frac{x_1 + \cdots + x_n}{n}\right) \leq \frac{f(x_1) + \cdots + f(x_n)}{n}$$

**常见凸函数**：$e^x$，$x^2$，$x\ln x$（$x > 0$），$-\ln x$（$x > 0$），$|x|^p$（$p \geq 1$）；

**常见凹函数**：$\ln x$（$x > 0$），$\sqrt{x}$（$x \geq 0$），$\sin x$（$0 \leq x \leq \pi$）。

**应用推导 AM-GM 不等式**：取 $f(x) = -\ln x$（凸函数），Jensen 给出：

$$-\ln\!\left(\frac{a_1 + \cdots + a_n}{n}\right) \leq \frac{-\ln a_1 - \cdots - \ln a_n}{n} = -\frac{1}{n}\ln(a_1 \cdots a_n)$$

即 $\ln\!\left(\dfrac{\sum a_i}{n}\right) \geq \ln\sqrt[n]{a_1 \cdots a_n}$，取指数得 **AM-GM 不等式**：$\dfrac{\sum a_i}{n} \geq \sqrt[n]{\prod a_i}$。

---

### 技巧 3：切线放缩（"切线砖头"）

**核心思路**：凸函数的切线在图像下方（见 Toolkit 10，路标 2），因此可以用切线提供**全局有效的简单上界/下界**：

$$f(x) \geq f(a) + f'(a)(x - a) \quad \text{（}f \text{ 是凸函数）}$$

**选取切点 $a$** 的原则：选让右端最简单的点（如 $a = 0$ 或让等号成立的点）。

**最重要的切线放缩不等式**（都由凸/凹函数在 $x=0$ 处的切线得到）：

| 不等式 | 来源 | 等号成立 |
|---|---|---|
| $e^x \geq 1 + x$ | $e^x$ 是凸函数，在 $x=0$ 处切线 $y = 1+x$ | $x = 0$ |
| $\ln(1+x) \leq x$（$x > -1$） | $-\ln(1+x)$ 是凸函数，等价于 $-\ln(1+x) \geq -x$，即切线在图像下方 | $x = 0$ |
| $\ln x \leq x - 1$（$x > 0$） | $-\ln x$ 是凸函数，在 $x=1$ 处切线 $y = x - 1$ | $x = 1$ |
| $1 + x \leq e^x$ | 同第一条 | $x = 0$ |
| $\sin x \leq x$（$x \geq 0$） | $-\sin x$ 是凸函数（$0 \leq x \leq \pi$），在 $x=0$ 处切线 $y = x$ | $x = 0$ |

**切线放缩的威力**在于：把复杂的非线性函数替换为线性函数，大幅简化后续计算，特别适用于累积乘积不等式（取对数后变成求和，再用 $\ln(1+x) \leq x$）。

**例**：证明 $\prod_{k=1}^n\!\left(1 + \dfrac{1}{k^2}\right) < e$。

取对数：$\sum_{k=1}^n \ln\!\left(1 + \dfrac{1}{k^2}\right) < \sum_{k=1}^n \dfrac{1}{k^2} < \dfrac{\pi^2}{6} < 2 < \infty$（只需估计上界）；由于 $\ln(1+t) \leq t$，得 $\sum \ln(1+1/k^2) \leq \sum 1/k^2 = \pi^2/6 \approx 1.645 < 2$，故乘积 $< e^2$（实际上可进一步缩紧）。

---

### 技巧 4：积分不等式

**核心思路**：将代数不等式"积分化"，利用积分的单调性（被积函数更大则积分更大）或特定积分恒等式。

**积分单调性**：若 $f(x) \leq g(x)$ 在 $[a, b]$ 上成立，则 $\displaystyle\int_a^b f(x)\,dx \leq \int_a^b g(x)\,dx$。

**Cauchy-Schwarz 积分不等式**：

$$\left(\int_a^b f(x)g(x)\,dx\right)^2 \leq \int_a^b [f(x)]^2\,dx \cdot \int_a^b [g(x)]^2\,dx$$

等号成立当且仅当 $f$ 与 $g$ 成比例（$f = cg$，$c$ 为常数）。

**证明**：对任意实数 $t$，$\int_a^b [f + tg]^2\,dx \geq 0$，展开得 $t^2\int g^2 + 2t\int fg + \int f^2 \geq 0$，这是关于 $t$ 的二次函数，其判别式 $\leq 0$：

$$4\left(\int fg\right)^2 - 4\left(\int f^2\right)\!\left(\int g^2\right) \leq 0$$

即 Cauchy-Schwarz 积分不等式。

**积分均值不等式**：若 $f$ 在 $[a, b]$ 上连续，$m \leq f(x) \leq M$，则：

$$m(b-a) \leq \int_a^b f(x)\,dx \leq M(b-a)$$

**积分中值定理**的不等式形式：$\displaystyle\int_a^b f(x)\,dx = f(\xi)(b-a)$ 对某 $\xi \in (a, b)$ 成立（要求 $f$ 连续）。

---

## 三、三巨头简介（Hardy / Hölder / Minkowski）

这三个不等式是分析数学中最重要的工具，在泛函分析、偏微分方程和调和分析中无处不在。

### 3.1 Hölder 不等式

**离散版**：设 $p, q > 1$，$\dfrac{1}{p} + \dfrac{1}{q} = 1$（共轭指数），$a_i, b_i \geq 0$，则：

$$\sum_{i=1}^n a_i b_i \leq \left(\sum_{i=1}^n a_i^p\right)^{1/p}\!\left(\sum_{i=1}^n b_i^q\right)^{1/q}$$

**积分版**：若 $f \in L^p[a,b]$，$g \in L^q[a,b]$（$1/p + 1/q = 1$），则：

$$\int_a^b |f(x)g(x)|\,dx \leq \left(\int_a^b |f|^p\,dx\right)^{1/p}\!\left(\int_a^b |g|^q\,dx\right)^{1/q}$$

Cauchy-Schwarz 是 $p = q = 2$ 的特例。

**证明要点**：用 Young 不等式 $ab \leq \dfrac{a^p}{p} + \dfrac{b^q}{q}$（对 $a, b \geq 0$，$1/p + 1/q = 1$，由 $\ln$ 的凹性或 $e^x$ 的凸性得到）。

---

### 3.2 Minkowski 不等式（三角不等式的推广）

对 $p \geq 1$，$f, g \in L^p[a, b]$：

$$\left(\int_a^b |f + g|^p\,dx\right)^{1/p} \leq \left(\int_a^b |f|^p\,dx\right)^{1/p} + \left(\int_a^b |g|^p\,dx\right)^{1/p}$$

即 $L^p$ 范数满足三角不等式。$p = 2$ 时退化为欧氏距离的三角不等式。

---

### 3.3 Hardy 不等式

对 $p > 1$，$f \geq 0$，$f \in L^p(0, +\infty)$，定义平均函数 $F(x) = \dfrac{1}{x}\int_0^x f(t)\,dt$，则：

$$\int_0^\infty [F(x)]^p\,dx \leq \left(\frac{p}{p-1}\right)^p \int_0^\infty [f(x)]^p\,dx$$

常数 $\left(\dfrac{p}{p-1}\right)^p$ 是最优的。$p = 2$ 时常数为 4，这是控制均值的平方积分的经典结果。

**实际意义**：Hardy 不等式说明"对 $f$ 做平均"这个操作不会使 $L^p$ 范数增大太多（至多增大常数倍）。它是估计偏微分方程解的重要工具。

---

## 四、不等式四大技巧对比表

| 技巧 | 适用场景 | 关键操作 | 典型例子 |
|---|---|---|---|
| **单调性** | 证 $f(x) \geq g(x)$ | 构造 $h = f - g$，分析 $h'$ 的符号 | $e^x \geq 1 + x$；$\ln x \leq x - 1$ |
| **凸性 + Jensen** | $n$ 个值的加权平均不等式 | 确认 $f'' \geq 0$，直接套 Jensen | AM-GM；$f(\bar{x}) \leq \overline{f(x)}$ |
| **切线放缩** | 用简单函数替代复杂函数上界/下界 | 找凸函数 + 选切点 | $\ln(1+x) \leq x$；乘积化为求和估计 |
| **积分不等式** | $L^2$ / $L^p$ 空间，被积函数比较 | Cauchy-Schwarz、Hölder、积分单调性 | $(\int fg)^2 \leq \int f^2 \cdot \int g^2$ |

---

## 五、演示题

**题目**：证明 $e^x \geq 1 + x$ 对所有实数 $x$ 成立，并找出等号成立的充要条件。

> **第一步：识别技巧。**
>
> 我们要证 $e^x - (1 + x) \geq 0$。右端是 $e^x$（指数函数），左端是其在 $x=0$ 处的切线 $1 + x$。这正是**切线放缩**的标准形式：$e^x$ 是凸函数，其在任意点的切线都在图像下方。
>
> 但我们用**单调性法**来完整证明，这样的逻辑链更严密。

> **第二步：构造辅助函数。**
>
> 令 $h(x) = e^x - 1 - x$，目标是证 $h(x) \geq 0$ 对所有 $x \in \mathbb{R}$ 成立。

> **第三步：求导分析。**
>
> $$h'(x) = e^x - 1$$
>
> 分析 $h'$ 的符号：
> - 当 $x < 0$ 时：$e^x < e^0 = 1$，故 $h'(x) = e^x - 1 < 0$，$h$ 严格递减；
> - 当 $x = 0$ 时：$h'(0) = 0$；
> - 当 $x > 0$ 时：$e^x > 1$，故 $h'(x) > 0$，$h$ 严格递增。
>
> 所以 $h$ 在 $(-\infty, 0]$ 上严格递减，在 $[0, +\infty)$ 上严格递增，$x = 0$ 是**全局最小值点**。

> **第四步：计算最小值。**
>
> $$h(0) = e^0 - 1 - 0 = 1 - 1 - 0 = 0$$
>
> 由于 $h$ 的全局最小值为 $0$，因此 $h(x) \geq 0$ 对所有 $x$ 成立，即 $e^x \geq 1 + x$。

> **第五步：等号条件。**
>
> 等号 $e^x = 1 + x$ 成立当且仅当 $h(x) = 0$，即 $x = 0$（唯一最小值点）。

> **第六步：双重验证（用凸性观点）。**
>
> $h''(x) = e^x > 0$ 对所有 $x$ 成立，故 $h$ 是**严格凸函数**，而 $h$ 的驻点 $x = 0$（$h'(0) = 0$）是严格凸函数的唯一极小值，必是全局最小值——与单调性论证一致 ✓。
>
> 从切线角度：$e^x$ 是凸函数，在 $x = 0$ 处的切线是 $y = 1 + x$，凸函数图像在任何切线的上方，故 $e^x \geq 1 + x$ ✓。

---

## 六、思考路标

**路标 1**：见到"证明不等式 $f(x) \geq g(x)$"，**第一反应是构造 $h = f - g$ 并求导**，这是微积分证不等式最通用的工具。不要直接做代数变形；先问："$h$ 在哪里取最小值？最小值是多少？"

**路标 2**：用单调性法时，**最容易遗漏的是"极值点不唯一"的情形**。若 $h'(x) = 0$ 有多个根（多个驻点），需分析 $h$ 在每个驻点处的值，取最小值，确认其 $\geq 0$。若有多个极小值，必须逐一验证。

**路标 3**：切线放缩的关键是**选对切点**。通常选让等号成立的那个点（不等式中两边相等的 $x$ 值），因为那里切线恰好"贴"着函数图像。例如 $\ln(1+x) \leq x$ 中，等号在 $x=0$ 成立，所以在 $x=0$ 处取切线（$\ln$ 的切线斜率 $= 1$，切线方程 $y = x$）。

**路标 4**：Jensen 不等式只在**凸函数**方向成立。用 Jensen 之前，**必须先验证函数的凸性**（$f'' \geq 0$）。若 $f$ 是凹函数，Jensen 不等式方向反转（$\leq$ 变 $\geq$）。混淆方向是最常见的错误。

**路标 5**：Cauchy-Schwarz 不等式的**等号条件**是 $f$ 与 $g$ 成比例，即 $f(x) = cg(x)$（$c$ 为常数）。在最优化问题中，Cauchy-Schwarz 等号给出了达到最大值的条件，是确定最优方案的关键。

**路标 6**：证明不等式时，**结论的方向（$\geq$ 还是 $\leq$）决定了需要用凸性还是凹性**。想要 $f(\bar{x}) \leq \overline{f(x)}$（均值的函数 $\leq$ 函数的均值），用 $f$ 凸；想要 $f(\bar{x}) \geq \overline{f(x)}$，用 $f$ 凹。对数函数（凹）给出 AM-GM，指数函数（凸）给出 GM $\leq$ AM（通过 $e$ 的 Jensen）。

**路标 7**：积分不等式 Cauchy-Schwarz 的**应用判断**：当需要估计 $\int fg$ 而 $\int f$ 和 $\int g$ 单独更好处理时，立刻想到 Cauchy-Schwarz。它把一个乘积的积分转化为各自平方的积分之积，经常大幅简化问题。

**路标 8**：Hardy / Hölder / Minkowski 是进阶工具，标准微积分课程中可以只了解结论和 $p = 2$ 的特殊情形（即 Cauchy-Schwarz），不需要背全部证明。但遇到 $L^p$ 空间的估计问题，这三个名字是检索方向的入口。

---

## 七、典型应用例题

### 例 1：单调性法

**题目**：证明 $\ln(1 + x) \leq x$ 对所有 $x > -1$ 成立。

**分析**：这是切线放缩不等式之一，但用单调性法完整证明。

**证明**：令 $h(x) = x - \ln(1+x)$（$x > -1$），目标是证 $h(x) \geq 0$。

$$h'(x) = 1 - \frac{1}{1+x} = \frac{x}{1+x}$$

对 $x > -1$：当 $x > 0$ 时 $h'(x) > 0$（递增）；当 $-1 < x < 0$ 时 $h'(x) < 0$（递减）；$x = 0$ 时 $h'(0) = 0$。

故 $x = 0$ 是 $h$ 的全局最小值点，$h(0) = 0 - \ln 1 = 0$。

因此 $h(x) \geq 0$，即 $\ln(1+x) \leq x$，等号当且仅当 $x = 0$ 时成立。$\blacksquare$

---

### 例 2：切线放缩 + Jensen

**题目**：设 $a_1, a_2, \ldots, a_n > 0$，证明 $\dfrac{a_1 + a_2 + \cdots + a_n}{n} \geq \sqrt[n]{a_1 a_2 \cdots a_n}$（AM-GM 不等式）。

**分析**：用 Jensen 不等式，利用 $\ln$ 函数的凹性。

**证明**：$\ln x$ 是凸吗？$(\ln x)'' = -1/x^2 < 0$（$x > 0$），故 $\ln x$ 是**凹函数**。

对凹函数，Jensen 不等式反向：$f\!\left(\dfrac{1}{n}\sum a_i\right) \geq \dfrac{1}{n}\sum f(a_i)$，即：

$$\ln\!\left(\frac{a_1 + \cdots + a_n}{n}\right) \geq \frac{1}{n}\sum_{i=1}^n \ln a_i = \frac{1}{n}\ln(a_1 \cdots a_n) = \ln\sqrt[n]{a_1 \cdots a_n}$$

由 $\ln$ 的单调性，两边取指数得：

$$\frac{a_1 + \cdots + a_n}{n} \geq \sqrt[n]{a_1 \cdots a_n}$$

等号成立当且仅当 $a_1 = a_2 = \cdots = a_n$（凹函数 Jensen 等号条件）。$\blacksquare$

---

### 例 3：Cauchy-Schwarz 积分不等式应用

**题目**：设 $f$ 在 $[0, 1]$ 上连续，证明 $\left(\displaystyle\int_0^1 f(x)\,dx\right)^2 \leq \int_0^1 [f(x)]^2\,dx$。

**分析**：取 $g(x) \equiv 1$（常数函数），直接用 Cauchy-Schwarz。

**证明**：由 Cauchy-Schwarz 积分不等式，取 $g(x) = 1$：

$$\left(\int_0^1 f(x) \cdot 1\,dx\right)^2 \leq \int_0^1 [f(x)]^2\,dx \cdot \int_0^1 1^2\,dx = \int_0^1 [f(x)]^2\,dx \cdot 1$$

即 $\left(\displaystyle\int_0^1 f(x)\,dx\right)^2 \leq \int_0^1 [f(x)]^2\,dx$。$\blacksquare$

**注**：等号成立当且仅当 $f(x) \equiv c$（常数）。直觉上：方差为零（$f$ 是常数）时均值的平方等于平方的均值；方差大于零时，平方的均值严格大于均值的平方（这正是方差非负的含义：$\text{Var}(f) = \mathbb{E}f^2 - (\mathbb{E}f)^2 \geq 0$）。

---

## 八、自测题

**第 1 题**：用单调性法证明 $\sin x < x$（对所有 $x > 0$）。

> 提示：令 $h(x) = x - \sin x$，$h'(x) = 1 - \cos x \geq 0$（等号仅在 $x = 2k\pi$ 时成立），所以 $h$ 在 $[0, +\infty)$ 上单调不减。$h(0) = 0$，且 $h$ 在任意 $\epsilon > 0$ 处已经开始增大（因为 $1 - \cos\epsilon > 0$ 对小 $\epsilon > 0$ 成立），故对所有 $x > 0$ 有 $h(x) > 0$，即 $\sin x < x$。

**第 2 题**：设 $0 < p < 1$，函数 $f(x) = x^p$（$x > 0$），用 Jensen 不等式证明：对 $a, b > 0$，$\left(\dfrac{a+b}{2}\right)^p \geq \dfrac{a^p + b^p}{2}$（说明 $x^p$ 是凹函数时的 Jensen 方向）。

> 提示：$f''(x) = p(p-1)x^{p-2}$，当 $0 < p < 1$ 时 $p(p-1) < 0$，所以 $f$ 是凹函数。凹函数的 Jensen：$f\!\left(\dfrac{a+b}{2}\right) \geq \dfrac{f(a)+f(b)}{2}$，即 $\left(\dfrac{a+b}{2}\right)^p \geq \dfrac{a^p + b^p}{2}$。

**第 3 题**：用 Cauchy-Schwarz 积分不等式证明 $\left(\displaystyle\int_0^{\pi/2} \sqrt{\sin x}\,dx\right)^2 \leq \dfrac{\pi}{2}$。

> 提示：取 $f(x) = (\sin x)^{1/2}$，$g(x) = 1$，Cauchy-Schwarz 给出 $\left(\displaystyle\int_0^{\pi/2} \sqrt{\sin x}\,dx\right)^2 \leq \int_0^{\pi/2} \sin x\,dx \cdot \int_0^{\pi/2} 1\,dx = 1 \cdot \dfrac{\pi}{2} = \dfrac{\pi}{2}$。

**第 4 题**：证明 $\dfrac{a^2}{b} + \dfrac{b^2}{c} + \dfrac{c^2}{a} \geq a + b + c$（$a, b, c > 0$），提示：用 Cauchy-Schwarz 的代数版。

> 提示：由 Cauchy-Schwarz（Titu 引理 / Engel 形式）：$\dfrac{x_1^2}{y_1} + \dfrac{x_2^2}{y_2} + \dfrac{x_3^2}{y_3} \geq \dfrac{(x_1+x_2+x_3)^2}{y_1+y_2+y_3}$。取 $x_1 = a, x_2 = b, x_3 = c$，$y_1 = b, y_2 = c, y_3 = a$，得 $\dfrac{a^2}{b} + \dfrac{b^2}{c} + \dfrac{c^2}{a} \geq \dfrac{(a+b+c)^2}{a+b+c} = a+b+c$。

**第 5 题**：利用切线放缩 $e^x \geq 1 + x$，证明对任意 $n$ 个正数 $a_1, \ldots, a_n$ 满足 $\sum a_i = 1$，有 $\prod_{i=1}^n a_i^{a_i} \geq e^{-1}$（即带权幂积的下界）。

> 提示：取对数，需证 $\sum a_i \ln a_i \geq -1$（注意 $\sum a_i = 1$）。对每个 $i$：由 $\ln a_i \leq a_i - 1$（即 $e^x \geq 1 + x$ 的对数版本），有 $a_i \ln a_i \leq a_i(a_i - 1) = a_i^2 - a_i$，求和得 $\sum a_i \ln a_i \leq \sum a_i^2 - \sum a_i = \sum a_i^2 - 1$，方向是给上界不是下界！要改用：对 $a_i \in (0, 1]$，$\ln a_i \geq 1 - 1/a_i$ 不成立……正确路线：直接用 $x\ln x \geq -1/e$（每项 $a_i \ln a_i \geq -1/e$，$n$ 项求和 $\geq -n/e$，但需精确到 $-1$）。更好的方法：注意 $\sum a_i \ln a_i$ 是负熵，直接用 Jensen（$f(x) = x\ln x$ 是凸函数）：$\sum a_i (a_i \ln a_i) \geq \left(\sum a_i^2\right)\ln\left(\sum a_i^2\right)$——此路较复杂；或直接引用 Gibbs 不等式（熵最大在均匀分布）给出下界 $\sum a_i \ln a_i \geq -\ln n > -\infty$，但 $-1$ 的精确界来自 $xe^{x-1} \geq x$ 的变形。这是一道有深度的练习题，完整证明需要 Gibbs/熵相关引理。
