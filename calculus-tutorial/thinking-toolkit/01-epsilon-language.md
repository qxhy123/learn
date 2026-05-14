# 极限的 ε 语言

> **一例速记**：用 $\varepsilon$-$N$ 语言证 $\lim_{n\to\infty}\dfrac{1}{n} = 0$。
> 对任意 $\varepsilon > 0$，要使 $\left|\dfrac{1}{n} - 0\right| = \dfrac{1}{n} < \varepsilon$，只需 $n > \dfrac{1}{\varepsilon}$。
> 取 $N = \left\lfloor\dfrac{1}{\varepsilon}\right\rfloor$（$\varepsilon$ 的倒数取整），则 $n > N$ 时 $\dfrac{1}{n} \leq \dfrac{1}{N+1} < \varepsilon$。
> **三步：设 $\varepsilon$ → 从不等式反解 $n$ 的下界 → 验证。**

---

## 一、为什么需要 ε 语言

直觉上，"$a_n$ 趋近于 $A$"意思是 $n$ 越来越大时，$a_n$ 与 $A$ 之差越来越小。但"越来越小"本身还是模糊的——它到底允不允许反复震荡？能不能永远不到达 $A$？能不能离 $A$ 的距离先变小再变大？

$\varepsilon$-$N$ 语言把这种模糊直觉精确化：**无论你给定的误差容限 $\varepsilon$ 有多小（哪怕是 $10^{-100}$），从某个足够大的 $N$ 开始，$a_n$ 与 $A$ 的误差就永远被压在 $\varepsilon$ 之内。**

这个精确化的过程在数学上叫做"$\varepsilon$-$N$ 定义"（Cauchy 于 19 世纪初给出），它是整个分析学的基石。理解它，微积分就有了坚实的地基；不理解它，求极限只是在"猜答案"。

本篇从数列极限的 $\varepsilon$-$N$ 语言讲起，再迁移到函数极限的 $\varepsilon$-$\delta$ 语言，最后系统整理三步证明范式和反向用法。

---

## 二、数列极限的 ε-N 定义

### 2.1 正式定义

**定义（数列极限）**：设 $\{a_n\}$ 是实数列，$A \in \mathbb{R}$。若

$$\forall \varepsilon > 0,\; \exists N \in \mathbb{N},\; \forall n > N:\; |a_n - A| < \varepsilon,$$

则称数列 $\{a_n\}$ **收敛**于 $A$，记作 $\lim_{n\to\infty} a_n = A$（或 $a_n \to A$，$n \to \infty$）。

### 2.2 逐句拆解

| 符号 | 含义 | 直观翻译 |
|---|---|---|
| $\forall \varepsilon > 0$ | 对任意正数 $\varepsilon$ | 你随意指定误差容限，无论多小 |
| $\exists N \in \mathbb{N}$ | 存在一个正整数 $N$ | 我能找到一个临界下标 |
| $\forall n > N$ | 对所有超过 $N$ 的下标 | 从 $N+1$ 项往后 |
| $|a_n - A| < \varepsilon$ | $a_n$ 与 $A$ 之差的绝对值小于 $\varepsilon$ | 误差被压在容限之内 |

关键点：
- $N$ 是可以**依赖** $\varepsilon$ 的（$\varepsilon$ 越小，$N$ 通常越大）。
- 不要求 $a_n = A$，只要求误差**小于** $\varepsilon$，即 $a_n$ 在以 $A$ 为中心、半径 $\varepsilon$ 的区间 $(A-\varepsilon, A+\varepsilon)$ 内。
- "$\forall\varepsilon > 0$"排在最前，是量词的嵌套结构：$\varepsilon$ 是"挑战者"先提出的，$N$ 是"应答者"后给出的。顺序不能颠倒。

### 2.3 三步证明范式

证明 $\lim_{n\to\infty} a_n = A$ 的标准步骤：

**第一步**：**设任意 $\varepsilon > 0$**（表明对所有正数容限都能应对）。

**第二步**：**找 $N$**——这是最关键的一步。分析"$|a_n - A| < \varepsilon$ 对 $n$ 有什么要求"，即**从这个不等式反解 $n$ 的下界**，得到形如 $n > f(\varepsilon)$ 的条件。然后令 $N$ 为 $f(\varepsilon)$ 取整（或任何使不等式满足的整数）。

**第三步**：**验证**——取任意 $n > N$，代入估计 $|a_n - A|$，严格证明它 $< \varepsilon$。

### 2.4 "找 N"的技巧

找 $N$ 的本质是做"**草稿估计**"：

1. 假装 $|a_n - A| < \varepsilon$ 已经是你的目标，把它当作不等式求 $n$ 的范围。
2. 有时估计不是等价变形，而是**放大**（用更大的式子去比较），使分析更简单，但要保证放大后的式子仍然 $< \varepsilon$。
3. 找到"$n > $ 某个依赖 $\varepsilon$ 的表达式"即可取 $N$。

---

## 三、函数极限的 ε-δ 定义

### 3.1 正式定义

**定义（函数极限）**：设 $f$ 在 $x_0$ 的某去心邻域有定义，$L \in \mathbb{R}$。若

$$\forall \varepsilon > 0,\; \exists \delta > 0,\; \forall x:\; 0 < |x - x_0| < \delta \Rightarrow |f(x) - L| < \varepsilon,$$

则称 $f(x)$ 在 $x \to x_0$ 时的极限为 $L$，记作 $\lim_{x\to x_0} f(x) = L$。

### 3.2 与 ε-N 的对比

| 对象 | 挑战者给的小量 | 应答者找的量 | 条件触发 |
|---|---|---|---|
| 数列 $\{a_n\}$ | $\varepsilon > 0$ | $N \in \mathbb{N}$ | $n > N$ |
| 函数 $f(x)$ | $\varepsilon > 0$ | $\delta > 0$ | $0 < |x-x_0| < \delta$ |

函数极限中：
- $0 < |x - x_0| < \delta$ 表示"**去心**"——$x \neq x_0$（$x$ 接近但不等于 $x_0$），极限不关心 $f(x_0)$ 的值甚至是否有意义。
- $\delta$ 依赖 $\varepsilon$（$\varepsilon$ 越小，$\delta$ 通常越小）。

### 3.3 单侧极限

- **右极限**：$0 < x - x_0 < \delta$，即 $x$ 从右侧趋近 $x_0$，记 $\lim_{x\to x_0^+} f(x) = L^+$。
- **左极限**：$-\delta < x - x_0 < 0$，即从左侧，记 $\lim_{x\to x_0^-} f(x) = L^-$。
- **结论**：$\lim_{x\to x_0} f(x) = L$ 当且仅当 $L^+ = L^- = L$。

### 3.4 无穷大处的极限

$$\lim_{x\to +\infty} f(x) = L \iff \forall \varepsilon > 0,\; \exists X > 0,\; x > X \Rightarrow |f(x) - L| < \varepsilon.$$

这里用 $X$（实数）代替了 $N$（整数），其余结构完全一样。

---

## 四、四类证明情形

| 情形 | 典型例子 | 找 $N$ 或 $\delta$ 的技巧 |
|---|---|---|
| 多项式型 $\dfrac{P(n)}{Q(n)}$ | $\dfrac{3n+1}{n+2} \to 3$ | 分子分母同除 $n$，化为 $\dfrac{3}{1+2/n}$；估计 $|a_n - 3|$，上界含 $\dfrac{1}{n}$，取 $N > \dfrac{C}{\varepsilon}$ |
| 根号型 $\sqrt{n+1} - \sqrt{n}$ | $\to 0$ | 有理化：$\dfrac{1}{\sqrt{n+1}+\sqrt{n}} < \dfrac{1}{\sqrt{n}} < \varepsilon$ 要求 $n > \dfrac{1}{\varepsilon^2}$ |
| 指数 / 等比型 | $q^n \to 0$（$|q|<1$） | 取对数：$|q^n - 0| = |q|^n < \varepsilon$ 要求 $n > \dfrac{\ln\varepsilon}{\ln|q|}$（负数因 $|q|<1$）|
| 夹逼型（三明治） | $0 \leq a_n \leq b_n \to 0$ | 只需证 $b_n < \varepsilon$（$b_n$ 更好估计），$0 \leq a_n \leq b_n < \varepsilon$ 则 $|a_n - 0| < \varepsilon$ |

---

## 五、演示题：用 ε-N 证 $\lim_{n\to\infty}\dfrac{1}{n} = 0$

> 拿到这道证明题，目标是向 $\varepsilon$-$N$ 语言的"挑战者-应答者"框架靠拢。
>
> **我先在草稿纸上想"怎么找 $N$"：**
>
> 想要 $|a_n - 0| = \dfrac{1}{n} < \varepsilon$，直接解这个不等式对 $n$：
> $$\frac{1}{n} < \varepsilon \iff n > \frac{1}{\varepsilon}.$$
>
> 所以只要 $n > \dfrac{1}{\varepsilon}$，误差就被压住了。$N$ 取"$\dfrac{1}{\varepsilon}$ 的取整"就行——或者为了严格，取 $N = \left\lfloor \dfrac{1}{\varepsilon} \right\rfloor$（这确保 $N$ 是正整数，且 $N \geq \dfrac{1}{\varepsilon} - 1 > \dfrac{1}{\varepsilon} - 1$）。
>
> 其实，只要 $N$ 是满足 $N \geq \dfrac{1}{\varepsilon}$ 的某个正整数即可，取哪个都行，定义只要求"存在"。
>
> **现在写正式证明（用第一人称展示书写格式）：**
>
> **证明：** 设任意 $\varepsilon > 0$。取 $N = \left\lfloor \dfrac{1}{\varepsilon} \right\rfloor$（即不超过 $\dfrac{1}{\varepsilon}$ 的最大正整数，或取任意整数 $> \dfrac{1}{\varepsilon}$）。
>
> 则对任意 $n > N$，有 $n > N \geq \dfrac{1}{\varepsilon}$，从而 $\dfrac{1}{\varepsilon} < n$，即 $\dfrac{1}{n} < \varepsilon$。
>
> 因此，
> $$\left| \frac{1}{n} - 0 \right| = \frac{1}{n} < \varepsilon.$$
>
> 由 $\varepsilon$ 的任意性，$\lim_{n\to\infty} \dfrac{1}{n} = 0$。$\square$
>
> **回顾**：草稿做的事是"反解"——先假设不等式成立，解出 $n$ 的下界；正式证明做的事是"顺推"——取好 $N$，从 $n > N$ 一步步推到误差 $< \varepsilon$。草稿和正式证明是同一个逻辑的两个方向。

---

## 六、思考路标

1. **见"证明 $\lim a_n = A$"** → 立刻启动三步框架：设 $\varepsilon > 0$ → 草稿反解 $|a_n - A| < \varepsilon$ 得 $n$ 的范围 → 取 $N$ 等于该范围的下界的取整值 → 正式顺推验证。

2. **找 $N$ 的草稿可以用"放大法"** → 把 $|a_n - A|$ 放大为一个更简单的表达式（但仍能 $< \varepsilon$），不必是精确等价变形。放大后的结果 $< \varepsilon$ 就能给出 $n$ 的范围。

3. **见 $\varepsilon$-$\delta$ 证明** → 类比 $\varepsilon$-$N$：草稿上先从 $|f(x) - L| < \varepsilon$ 反解 $|x - x_0|$ 的范围，得到"只要 $|x - x_0| < $ 某表达式 $(\varepsilon)$"，取 $\delta$ 等于该表达式（或其简化上界）。

4. **$N$（或 $\delta$）不唯一** → 任何满足条件的 $N$（或 $\delta$）都合法，不需要取"最优"的。如果一个 $\delta$ 能用，任何更小的 $\delta' \leq \delta$ 也能用。

5. **见单侧极限问题** → 先分别用 $\varepsilon$-$\delta$ 验证左极限和右极限，再判断两者是否相等。不等 → 极限不存在。

6. **反向用法：已知极限求参数** → 极限表达式本身含参数 $a$（如 $\lim_{x\to 0}\dfrac{\sin(ax)}{x}$），先用极限运算法则（或等价无穷小）把极限"算出来"（含 $a$），再令结果等于题目给定值，解 $a$。

7. **数列 $\{a_n\}$ 收敛的必要条件：有界** → 若 $a_n \to A$，则 $\{a_n\}$ 有界（$|a_n| \leq M$ 对某个 $M$）。若 $\{a_n\}$ 无界，极限必不存在。此条件常用来否定极限的存在性。

8. **单调有界数列必收敛** → 单调递增有上界、或单调递减有下界的数列收敛。证收敛时若能证单调 + 有界，不必给出极限值，只需说明"极限存在"（然后再用递推方程求极限值）。

---

## 七、典型应用 3 例

### 例 1：根号型极限的 ε-N 证明

**题目**：证明 $\lim_{n\to\infty} (\sqrt{n+1} - \sqrt{n}) = 0$。

**思路**：

令 $a_n = \sqrt{n+1} - \sqrt{n}$。有理化：

$$a_n = \sqrt{n+1} - \sqrt{n} = \frac{(n+1) - n}{\sqrt{n+1} + \sqrt{n}} = \frac{1}{\sqrt{n+1} + \sqrt{n}}.$$

因为 $\sqrt{n+1} + \sqrt{n} > \sqrt{n}$，所以 $a_n < \dfrac{1}{\sqrt{n}}$。

对任意 $\varepsilon > 0$，要使 $a_n < \varepsilon$，只需 $\dfrac{1}{\sqrt{n}} < \varepsilon$，即 $n > \dfrac{1}{\varepsilon^2}$。

取 $N = \left\lfloor \dfrac{1}{\varepsilon^2} \right\rfloor + 1$。则 $n > N$ 时，$n > \dfrac{1}{\varepsilon^2}$，从而

$$0 < a_n < \frac{1}{\sqrt{n}} < \varepsilon.$$

故 $|a_n - 0| = a_n < \varepsilon$，证明完毕。$\square$

**关键技巧**：有理化 + 放大（用 $\dfrac{1}{\sqrt{n}}$ 代替精确表达式 $\dfrac{1}{\sqrt{n+1}+\sqrt{n}}$）。

---

### 例 2：含参极限求参数值

**题目**：已知 $\lim_{x\to 0}\dfrac{x^2 + ax}{x} = 2$，求常数 $a$。

**思路**：

当 $x \neq 0$ 时，化简：

$$\frac{x^2 + ax}{x} = \frac{x(x + a)}{x} = x + a.$$

因此

$$\lim_{x\to 0} \frac{x^2 + ax}{x} = \lim_{x\to 0} (x + a) = 0 + a = a.$$

题目给定极限为 $2$，故 $a = 2$。

**反向用法的要点**：先化简极限表达式（得到含 $a$ 的显式形式），再令其等于已知极限值，解出参数。

---

### 例 3：单调有界证数列收敛

**题目**：数列 $\{a_n\}$ 定义为 $a_1 = 1$，$a_{n+1} = \sqrt{2 + a_n}$（$n \geq 1$）。证明 $\{a_n\}$ 收敛，并求其极限。

**思路（分两步）**：

**第一步：单调有界，故收敛。**

先用数学归纳法证 $a_n \leq 2$（有上界）：
- 基础：$a_1 = 1 \leq 2$ ✓。
- 归纳：若 $a_n \leq 2$，则 $a_{n+1} = \sqrt{2 + a_n} \leq \sqrt{2 + 2} = 2$ ✓。

再证 $\{a_n\}$ 单调递增：$a_{n+1} - a_n = \sqrt{2 + a_n} - a_n$。令 $f(t) = \sqrt{2+t} - t$，$f(1) = \sqrt{3} - 1 > 0$，$f'(t) = \dfrac{1}{2\sqrt{2+t}} - 1 < 0$（$t \geq 0$），$f(2) = 0$。

故对 $1 \leq a_n \leq 2$，$f(a_n) = a_{n+1} - a_n \geq 0$（可更简洁地验证：$a_n^2 \leq 2 + a_n \iff (a_n-2)(a_n+1) \leq 0 \iff a_n \leq 2$），即 $a_{n+1} \geq a_n$。

由**单调有界收敛定理**，$\lim_{n\to\infty} a_n$ 存在，设为 $L$。

**第二步：用递推方程求 $L$。**

对 $a_{n+1} = \sqrt{2 + a_n}$ 两边取极限：

$$L = \sqrt{2 + L} \implies L^2 = 2 + L \implies L^2 - L - 2 = 0 \implies (L-2)(L+1) = 0.$$

故 $L = 2$ 或 $L = -1$。因为 $a_n \geq 1 > 0$，所以 $L \geq 0$，舍去 $L = -1$，得 $L = 2$。

**结论**：$\lim_{n\to\infty} a_n = 2$。

---

## 八、自测题

**第 1 题**：用 $\varepsilon$-$N$ 语言证明 $\lim_{n\to\infty} \dfrac{2n-1}{n+3} = 2$。

提示：计算 $\left|\dfrac{2n-1}{n+3} - 2\right| = \left|\dfrac{-7}{n+3}\right| = \dfrac{7}{n+3} < \dfrac{7}{n}$。要使 $\dfrac{7}{n} < \varepsilon$，需 $n > \dfrac{7}{\varepsilon}$，取 $N = \left\lfloor\dfrac{7}{\varepsilon}\right\rfloor + 1$。

---

**第 2 题**：用 $\varepsilon$-$\delta$ 语言证明 $\lim_{x\to 2}(3x - 1) = 5$。

提示：$|f(x) - 5| = |3x - 1 - 5| = |3x - 6| = 3|x - 2|$。要使 $3|x-2| < \varepsilon$，需 $|x-2| < \dfrac{\varepsilon}{3}$，取 $\delta = \dfrac{\varepsilon}{3}$。

---

**第 3 题**：已知 $\lim_{n\to\infty}(a_n - 2n) = 5$，求 $\lim_{n\to\infty}\dfrac{a_n}{n}$。

提示：设 $b_n = a_n - 2n \to 5$，则 $a_n = 2n + b_n$，$\dfrac{a_n}{n} = 2 + \dfrac{b_n}{n}$。因 $b_n \to 5$，故 $\dfrac{b_n}{n} \to 0$（有界 ÷ $n$），所以 $\dfrac{a_n}{n} \to 2$。

---

**第 4 题**：数列 $\{a_n\}$ 满足 $a_1 = 2$，$a_{n+1} = \dfrac{a_n}{2} + 1$。证明 $\{a_n\}$ 收敛并求极限。

提示：先证有上界 $a_n \leq 2$（归纳：$a_{n+1} = \dfrac{a_n}{2}+1 \leq \dfrac{2}{2}+1 = 2$）；再证单调递减（$a_2 = 2$，$a_{n+1} - a_n = 1 - \dfrac{a_n}{2}$，若 $a_n \geq 2$，则 $a_{n+1} \geq a_n$；实际证明 $a_n = 2$ 恒成立）。设极限 $L$，$L = \dfrac{L}{2} + 1$，解得 $L = 2$。

---

**第 5 题**：用 $\varepsilon$-$N$ 语言证明：若 $|q| < 1$，则 $\lim_{n\to\infty} q^n = 0$。

提示：若 $q = 0$，结论显然。若 $0 < |q| < 1$，取对数：$|q^n - 0| = |q|^n < \varepsilon$ 等价于 $n\ln|q| < \ln\varepsilon$（注意 $\ln|q| < 0$），即 $n > \dfrac{\ln\varepsilon}{\ln|q|}$。取 $N = \left\lfloor\dfrac{\ln\varepsilon}{\ln|q|}\right\rfloor + 1$（注意 $\dfrac{\ln\varepsilon}{\ln|q|}$ 可能是负数，但取整后仍为合法下界）。
