# 第14章 广义积分

## 学习目标

通过本章学习，你将能够：

- 理解无穷区间上积分的定义，判断其收敛性与发散性
- 掌握无界函数积分（瑕积分）的定义与计算方法
- 熟练运用比较判别法和极限比较判别法判断广义积分的收敛性
- 理解绝对收敛与条件收敛的概念及其关系
- 掌握Gamma函数的定义、性质和常用公式

---

## 14.1 无穷区间上的积分

在定积分的定义中，积分区间 $[a, b]$ 是有限的。但在许多实际问题中，我们需要在无穷区间上进行积分。这类积分称为**无穷限广义积分**或**第一类广义积分**。

### 14.1.1 基本定义

**定义**（无穷上限积分）：设函数 $f(x)$ 在 $[a, +\infty)$ 上连续，若极限
$$\lim_{b \to +\infty} \int_a^b f(x) \, dx$$
存在，则称此极限为 $f(x)$ 在 $[a, +\infty)$ 上的**广义积分**，记作
$$\int_a^{+\infty} f(x) \, dx = \lim_{b \to +\infty} \int_a^b f(x) \, dx$$
此时称该广义积分**收敛**。若极限不存在，则称该广义积分**发散**。

**定义**（无穷下限积分）：类似地，定义
$$\int_{-\infty}^b f(x) \, dx = \lim_{a \to -\infty} \int_a^b f(x) \, dx$$

**定义**（双无穷积分）：对于 $(-\infty, +\infty)$ 上的积分，定义
$$\int_{-\infty}^{+\infty} f(x) \, dx = \int_{-\infty}^c f(x) \, dx + \int_c^{+\infty} f(x) \, dx$$
其中 $c$ 为任意实数。当且仅当右边两个积分都收敛时，左边的积分才收敛。

### 14.1.2 例题详解

> **例题 14.1** 计算 $\int_1^{+\infty} \dfrac{1}{x^2} \, dx$。

**解**：按定义计算：
$$\int_1^{+\infty} \frac{1}{x^2} \, dx = \lim_{b \to +\infty} \int_1^b \frac{1}{x^2} \, dx = \lim_{b \to +\infty} \left[-\frac{1}{x}\right]_1^b = \lim_{b \to +\infty} \left(-\frac{1}{b} + 1\right) = 1$$

因此该广义积分收敛，其值为 $1$。$\square$

> **例题 14.2** 判断 $\int_1^{+\infty} \dfrac{1}{x} \, dx$ 的敛散性。

**解**：
$$\int_1^{+\infty} \frac{1}{x} \, dx = \lim_{b \to +\infty} \int_1^b \frac{1}{x} \, dx = \lim_{b \to +\infty} [\ln x]_1^b = \lim_{b \to +\infty} \ln b = +\infty$$

因此该广义积分发散。$\square$

> **例题 14.3**（p-积分）讨论 $\int_1^{+\infty} \dfrac{1}{x^p} \, dx$ 的敛散性（$p > 0$）。

**解**：当 $p \neq 1$ 时：
$$\int_1^{+\infty} \frac{1}{x^p} \, dx = \lim_{b \to +\infty} \left[\frac{x^{1-p}}{1-p}\right]_1^b = \lim_{b \to +\infty} \frac{1}{1-p}\left(b^{1-p} - 1\right)$$

- 若 $p > 1$，则 $1 - p < 0$，$b^{1-p} \to 0$（$b \to +\infty$），积分收敛于 $\dfrac{1}{p-1}$
- 若 $0 < p < 1$，则 $1 - p > 0$，$b^{1-p} \to +\infty$，积分发散
- 若 $p = 1$，由例题14.2知积分发散

**结论**：$\int_1^{+\infty} \dfrac{1}{x^p} \, dx$ 当 $p > 1$ 时收敛，当 $p \leq 1$ 时发散。$\square$

> **例题 14.4** 计算 $\int_0^{+\infty} e^{-x} \, dx$。

**解**：
$$\int_0^{+\infty} e^{-x} \, dx = \lim_{b \to +\infty} \int_0^b e^{-x} \, dx = \lim_{b \to +\infty} \left[-e^{-x}\right]_0^b = \lim_{b \to +\infty} (1 - e^{-b}) = 1$$

$\square$

> **例题 14.5** 计算 $\int_{-\infty}^{+\infty} \dfrac{1}{1 + x^2} \, dx$。

**解**：取 $c = 0$，分成两部分：
$$\int_{-\infty}^{0} \frac{1}{1 + x^2} \, dx = \lim_{a \to -\infty} [\arctan x]_a^0 = 0 - \left(-\frac{\pi}{2}\right) = \frac{\pi}{2}$$
$$\int_{0}^{+\infty} \frac{1}{1 + x^2} \, dx = \lim_{b \to +\infty} [\arctan x]_0^b = \frac{\pi}{2} - 0 = \frac{\pi}{2}$$

因此 $\int_{-\infty}^{+\infty} \dfrac{1}{1 + x^2} \, dx = \dfrac{\pi}{2} + \dfrac{\pi}{2} = \pi$。$\square$

### 14.1.3 无穷限积分的性质

**性质1**（线性性）：若 $\int_a^{+\infty} f(x) \, dx$ 和 $\int_a^{+\infty} g(x) \, dx$ 都收敛，则
$$\int_a^{+\infty} [\alpha f(x) + \beta g(x)] \, dx = \alpha \int_a^{+\infty} f(x) \, dx + \beta \int_a^{+\infty} g(x) \, dx$$

**性质2**（区间可加性）：$\int_a^{+\infty} f(x) \, dx = \int_a^c f(x) \, dx + \int_c^{+\infty} f(x) \, dx$，且两边同时收敛或发散。

**性质3**（比较原则）：设 $0 \leq f(x) \leq g(x)$（$x \geq a$），则
- 若 $\int_a^{+\infty} g(x) \, dx$ 收敛，则 $\int_a^{+\infty} f(x) \, dx$ 也收敛
- 若 $\int_a^{+\infty} f(x) \, dx$ 发散，则 $\int_a^{+\infty} g(x) \, dx$ 也发散

---

## 14.2 无界函数的积分（瑕积分）

当被积函数在积分区间的某点无界时，普通的定积分定义不再适用。这类积分称为**无界函数的广义积分**或**瑕积分**，也称**第二类广义积分**。

### 14.2.1 瑕点的概念

**定义**（瑕点）：若 $f(x)$ 在点 $x_0$ 的任意邻域内无界，则称 $x_0$ 为 $f(x)$ 的**瑕点**（或**奇点**）。

### 14.2.2 瑕点在端点的情况

**定义**：设 $f(x)$ 在 $(a, b]$ 上连续，$x = a$ 为瑕点，则定义
$$\int_a^b f(x) \, dx = \lim_{\varepsilon \to 0^+} \int_{a+\varepsilon}^b f(x) \, dx$$
若极限存在，称瑕积分收敛；否则称发散。

类似地，若 $x = b$ 为瑕点：
$$\int_a^b f(x) \, dx = \lim_{\varepsilon \to 0^+} \int_a^{b-\varepsilon} f(x) \, dx$$

> **例题 14.6** 计算 $\int_0^1 \dfrac{1}{\sqrt{x}} \, dx$。

**解**：$x = 0$ 是瑕点。
$$\int_0^1 \frac{1}{\sqrt{x}} \, dx = \lim_{\varepsilon \to 0^+} \int_\varepsilon^1 x^{-1/2} \, dx = \lim_{\varepsilon \to 0^+} \left[2\sqrt{x}\right]_\varepsilon^1 = \lim_{\varepsilon \to 0^+} (2 - 2\sqrt{\varepsilon}) = 2$$

$\square$

> **例题 14.7**（q-积分）讨论 $\int_0^1 \dfrac{1}{x^q} \, dx$ 的敛散性（$q > 0$）。

**解**：$x = 0$ 是瑕点。当 $q \neq 1$ 时：
$$\int_0^1 \frac{1}{x^q} \, dx = \lim_{\varepsilon \to 0^+} \left[\frac{x^{1-q}}{1-q}\right]_\varepsilon^1 = \lim_{\varepsilon \to 0^+} \frac{1}{1-q}\left(1 - \varepsilon^{1-q}\right)$$

- 若 $0 < q < 1$，则 $1 - q > 0$，$\varepsilon^{1-q} \to 0$（$\varepsilon \to 0^+$），积分收敛于 $\dfrac{1}{1-q}$
- 若 $q > 1$，则 $1 - q < 0$，$\varepsilon^{1-q} \to +\infty$，积分发散
- 若 $q = 1$，$\int_0^1 \dfrac{1}{x} \, dx = \lim\limits_{\varepsilon \to 0^+} [-\ln \varepsilon] = +\infty$，发散

**结论**：$\int_0^1 \dfrac{1}{x^q} \, dx$ 当 $0 < q < 1$ 时收敛，当 $q \geq 1$ 时发散。$\square$

> **例题 14.8** 计算 $\int_0^1 \ln x \, dx$。

**解**：$x = 0$ 是瑕点（$\lim\limits_{x \to 0^+} \ln x = -\infty$）。
$$\int_0^1 \ln x \, dx = \lim_{\varepsilon \to 0^+} \int_\varepsilon^1 \ln x \, dx = \lim_{\varepsilon \to 0^+} [x\ln x - x]_\varepsilon^1$$
$$= \lim_{\varepsilon \to 0^+} [(0 - 1) - (\varepsilon \ln \varepsilon - \varepsilon)] = -1 - \lim_{\varepsilon \to 0^+} \varepsilon \ln \varepsilon$$

利用洛必达法则：$\lim\limits_{\varepsilon \to 0^+} \varepsilon \ln \varepsilon = \lim\limits_{\varepsilon \to 0^+} \dfrac{\ln \varepsilon}{1/\varepsilon} = \lim\limits_{\varepsilon \to 0^+} \dfrac{1/\varepsilon}{-1/\varepsilon^2} = \lim\limits_{\varepsilon \to 0^+} (-\varepsilon) = 0$

因此 $\int_0^1 \ln x \, dx = -1$。$\square$

### 14.2.3 瑕点在区间内部

**定义**：设 $f(x)$ 在 $[a, b]$ 上除点 $c$（$a < c < b$）外连续，$x = c$ 为瑕点，则定义
$$\int_a^b f(x) \, dx = \int_a^c f(x) \, dx + \int_c^b f(x) \, dx$$
当且仅当右边两个瑕积分都收敛时，左边的积分才收敛。

> **例题 14.9** 计算 $\int_{-1}^{1} \dfrac{1}{x^2} \, dx$。

**解**：$x = 0$ 是瑕点，位于区间内部。

$$\int_0^{1} \frac{1}{x^2} \, dx = \lim_{\varepsilon \to 0^+} \int_\varepsilon^1 \frac{1}{x^2} \, dx = \lim_{\varepsilon \to 0^+} \left[-\frac{1}{x}\right]_\varepsilon^1 = \lim_{\varepsilon \to 0^+} \left(-1 + \frac{1}{\varepsilon}\right) = +\infty$$

因此 $\int_0^{1} \dfrac{1}{x^2} \, dx$ 发散，从而 $\int_{-1}^{1} \dfrac{1}{x^2} \, dx$ 发散。

**注意**：不能直接计算 $\left[-\dfrac{1}{x}\right]_{-1}^{1} = -1 - 1 = -2$，这是错误的！$\square$

> **例题 14.10** 计算 $\int_0^2 \dfrac{1}{\sqrt{|x-1|}} \, dx$。

**解**：$x = 1$ 是瑕点。
$$\int_0^2 \frac{1}{\sqrt{|x-1|}} \, dx = \int_0^1 \frac{1}{\sqrt{1-x}} \, dx + \int_1^2 \frac{1}{\sqrt{x-1}} \, dx$$

对第一个积分，设 $u = 1 - x$：
$$\int_0^1 \frac{1}{\sqrt{1-x}} \, dx = \lim_{\varepsilon \to 0^+} \left[-2\sqrt{1-x}\right]_0^{1-\varepsilon} = \lim_{\varepsilon \to 0^+} (-2\sqrt{\varepsilon} + 2) = 2$$

对第二个积分：
$$\int_1^2 \frac{1}{\sqrt{x-1}} \, dx = \lim_{\varepsilon \to 0^+} \left[2\sqrt{x-1}\right]_{1+\varepsilon}^2 = 2 - 0 = 2$$

因此 $\int_0^2 \dfrac{1}{\sqrt{|x-1|}} \, dx = 2 + 2 = 4$。$\square$

---

## 14.3 广义积分的收敛判别

### 14.3.1 比较判别法

**定理**（比较判别法）：设 $f(x)$、$g(x)$ 在 $[a, +\infty)$ 上连续，且 $0 \leq f(x) \leq g(x)$，则：

1. 若 $\int_a^{+\infty} g(x) \, dx$ 收敛，则 $\int_a^{+\infty} f(x) \, dx$ 也收敛
2. 若 $\int_a^{+\infty} f(x) \, dx$ 发散，则 $\int_a^{+\infty} g(x) \, dx$ 也发散

> **例题 14.11** 判断 $\int_1^{+\infty} e^{-x^2} \, dx$ 的敛散性。

**解**：当 $x \geq 1$ 时，$x^2 \geq x$，故 $e^{-x^2} \leq e^{-x}$。

由于 $\int_1^{+\infty} e^{-x} \, dx = e^{-1}$ 收敛，由比较判别法，$\int_1^{+\infty} e^{-x^2} \, dx$ 也收敛。$\square$

> **例题 14.12** 判断 $\int_2^{+\infty} \dfrac{1}{x \ln x} \, dx$ 的敛散性。

**解**：设 $u = \ln x$，则 $du = \dfrac{1}{x} dx$。当 $x = 2$ 时 $u = \ln 2$，当 $x \to +\infty$ 时 $u \to +\infty$。

$$\int_2^{+\infty} \frac{1}{x \ln x} \, dx = \int_{\ln 2}^{+\infty} \frac{1}{u} \, du$$

由 p-积分（$p = 1$）知此积分发散。$\square$

### 14.3.2 极限比较判别法

**定理**（极限比较判别法）：设 $f(x)$、$g(x)$ 在 $[a, +\infty)$ 上非负连续，且
$$\lim_{x \to +\infty} \frac{f(x)}{g(x)} = l$$

1. 若 $0 < l < +\infty$，则 $\int_a^{+\infty} f(x) \, dx$ 与 $\int_a^{+\infty} g(x) \, dx$ 同敛散
2. 若 $l = 0$ 且 $\int_a^{+\infty} g(x) \, dx$ 收敛，则 $\int_a^{+\infty} f(x) \, dx$ 也收敛
3. 若 $l = +\infty$ 且 $\int_a^{+\infty} g(x) \, dx$ 发散，则 $\int_a^{+\infty} f(x) \, dx$ 也发散

**实用技巧**：通常取 $g(x) = \dfrac{1}{x^p}$，利用 p-积分的结论。

> **例题 14.13** 判断 $\int_1^{+\infty} \dfrac{x}{x^3 + 1} \, dx$ 的敛散性。

**解**：当 $x \to +\infty$ 时，$\dfrac{x}{x^3 + 1} \sim \dfrac{x}{x^3} = \dfrac{1}{x^2}$。

取 $g(x) = \dfrac{1}{x^2}$，则 $\lim\limits_{x \to +\infty} \dfrac{f(x)}{g(x)} = \lim\limits_{x \to +\infty} \dfrac{x^3}{x^3 + 1} = 1$。

由于 $\int_1^{+\infty} \dfrac{1}{x^2} \, dx$ 收敛（$p = 2 > 1$），故原积分收敛。$\square$

> **例题 14.14** 判断 $\int_0^1 \dfrac{1}{\sqrt{x(1-x)}} \, dx$ 的敛散性。

**解**：$x = 0$ 和 $x = 1$ 都是瑕点。分成两部分讨论：

在 $x = 0$ 附近：$\dfrac{1}{\sqrt{x(1-x)}} \sim \dfrac{1}{\sqrt{x}}$，由 q-积分（$q = \dfrac{1}{2} < 1$）知收敛。

在 $x = 1$ 附近：设 $u = 1 - x$，则 $\dfrac{1}{\sqrt{x(1-x)}} \sim \dfrac{1}{\sqrt{u}}$，同样收敛。

因此原积分收敛。$\square$

### 14.3.3 绝对收敛与条件收敛

**定义**：设广义积分 $\int_a^{+\infty} f(x) \, dx$ 收敛。

- 若 $\int_a^{+\infty} |f(x)| \, dx$ 也收敛，则称原积分**绝对收敛**
- 若 $\int_a^{+\infty} |f(x)| \, dx$ 发散，则称原积分**条件收敛**

**定理**：绝对收敛的广义积分必定收敛。

> **例题 14.15** 判断 $\int_1^{+\infty} \dfrac{\sin x}{x^2} \, dx$ 的敛散性。

**解**：由于 $\left|\dfrac{\sin x}{x^2}\right| \leq \dfrac{1}{x^2}$，而 $\int_1^{+\infty} \dfrac{1}{x^2} \, dx$ 收敛，

由比较判别法，$\int_1^{+\infty} \left|\dfrac{\sin x}{x^2}\right| \, dx$ 收敛，故原积分绝对收敛。$\square$

> **例题 14.16** 积分 $\int_1^{+\infty} \dfrac{\sin x}{x} \, dx$ 条件收敛（狄利克雷积分）。

**说明**：可以证明此积分收敛于 $\dfrac{\pi}{2} - \text{Si}(1)$，其中 $\text{Si}(x) = \int_0^x \dfrac{\sin t}{t} \, dt$ 是正弦积分函数。

但 $\int_1^{+\infty} \left|\dfrac{\sin x}{x}\right| \, dx$ 发散（因为 $\left|\dfrac{\sin x}{x}\right| \geq \dfrac{\sin^2 x}{x} = \dfrac{1 - \cos 2x}{2x}$）。

因此该积分是条件收敛的。$\square$

---

## 14.4 Gamma函数

### 14.4.1 定义

**定义**（Gamma函数）：对于 $s > 0$，定义
$$\Gamma(s) = \int_0^{+\infty} x^{s-1} e^{-x} \, dx$$

这是一个含参变量的广义积分，它在 $s > 0$ 时收敛。

**收敛性分析**：

将积分分成两部分：$\int_0^{+\infty} = \int_0^1 + \int_1^{+\infty}$

- 对于 $\int_0^1 x^{s-1} e^{-x} \, dx$：当 $x \to 0^+$ 时，$x^{s-1} e^{-x} \sim x^{s-1}$，由 q-积分，当 $s > 0$ 时收敛
- 对于 $\int_1^{+\infty} x^{s-1} e^{-x} \, dx$：由于 $e^{-x}$ 衰减比任何 $x^n$ 都快，该积分收敛

### 14.4.2 递推公式

**定理**（递推公式）：对于 $s > 0$，有
$$\Gamma(s+1) = s \cdot \Gamma(s)$$

**证明**：利用分部积分。
$$\Gamma(s+1) = \int_0^{+\infty} x^s e^{-x} \, dx$$

设 $u = x^s$，$dv = e^{-x} dx$，则 $du = sx^{s-1} dx$，$v = -e^{-x}$。

$$\Gamma(s+1) = \left[-x^s e^{-x}\right]_0^{+\infty} + s\int_0^{+\infty} x^{s-1} e^{-x} \, dx$$

由于 $\lim\limits_{x \to +\infty} x^s e^{-x} = 0$ 且 $\lim\limits_{x \to 0^+} x^s e^{-x} = 0$（当 $s > 0$），故
$$\Gamma(s+1) = s \cdot \Gamma(s)$$

$\square$

### 14.4.3 与阶乘的关系

**定理**：$\Gamma(1) = 1$，且对于正整数 $n$，有
$$\Gamma(n+1) = n!$$

**证明**：
$$\Gamma(1) = \int_0^{+\infty} e^{-x} \, dx = 1$$

由递推公式反复应用：
$$\Gamma(n+1) = n \cdot \Gamma(n) = n(n-1) \cdot \Gamma(n-1) = \cdots = n(n-1)\cdots 2 \cdot 1 \cdot \Gamma(1) = n!$$

$\square$

因此，Gamma函数是阶乘在正实数上的推广。

### 14.4.4 常用值

| $s$ | $\Gamma(s)$ |
|:---:|:---:|
| $1$ | $1$ |
| $2$ | $1$ |
| $3$ | $2$ |
| $n+1$（正整数） | $n!$ |
| $\dfrac{1}{2}$ | $\sqrt{\pi}$ |
| $\dfrac{3}{2}$ | $\dfrac{\sqrt{\pi}}{2}$ |
| $\dfrac{5}{2}$ | $\dfrac{3\sqrt{\pi}}{4}$ |

**重要公式**：
$$\Gamma\left(\frac{1}{2}\right) = \sqrt{\pi}$$

**证明**：
$$\Gamma\left(\frac{1}{2}\right) = \int_0^{+\infty} x^{-1/2} e^{-x} \, dx$$

设 $x = t^2$，则 $dx = 2t \, dt$：
$$\Gamma\left(\frac{1}{2}\right) = \int_0^{+\infty} t^{-1} e^{-t^2} \cdot 2t \, dt = 2\int_0^{+\infty} e^{-t^2} \, dt$$

利用高斯积分 $\int_0^{+\infty} e^{-t^2} \, dt = \dfrac{\sqrt{\pi}}{2}$（证明见概率论），得
$$\Gamma\left(\frac{1}{2}\right) = 2 \cdot \frac{\sqrt{\pi}}{2} = \sqrt{\pi}$$

$\square$

> **例题 14.17** 计算 $\Gamma\left(\dfrac{5}{2}\right)$。

**解**：由递推公式：
$$\Gamma\left(\frac{5}{2}\right) = \frac{3}{2} \cdot \Gamma\left(\frac{3}{2}\right) = \frac{3}{2} \cdot \frac{1}{2} \cdot \Gamma\left(\frac{1}{2}\right) = \frac{3}{4}\sqrt{\pi}$$

$\square$

> **例题 14.18** 计算 $\int_0^{+\infty} x^3 e^{-2x} \, dx$。

**解**：设 $t = 2x$，则 $x = \dfrac{t}{2}$，$dx = \dfrac{1}{2} dt$。

$$\int_0^{+\infty} x^3 e^{-2x} \, dx = \int_0^{+\infty} \frac{t^3}{8} e^{-t} \cdot \frac{1}{2} \, dt = \frac{1}{16}\int_0^{+\infty} t^3 e^{-t} \, dt = \frac{1}{16}\Gamma(4) = \frac{3!}{16} = \frac{6}{16} = \frac{3}{8}$$

$\square$

> **例题 14.19** 计算 $\int_0^{+\infty} \sqrt{x} \, e^{-x} \, dx$。

**解**：
$$\int_0^{+\infty} \sqrt{x} \, e^{-x} \, dx = \int_0^{+\infty} x^{1/2} e^{-x} \, dx = \Gamma\left(\frac{3}{2}\right) = \frac{1}{2}\Gamma\left(\frac{1}{2}\right) = \frac{\sqrt{\pi}}{2}$$

$\square$

---

## 14.5 Beta函数

### 14.5.1 定义

**定义**（Beta函数）：对于 $p > 0$，$q > 0$，定义

$$B(p,q) = \int_0^1 x^{p-1}(1-x)^{q-1}\,dx$$

**收敛性分析**：被积函数在 $x = 0$ 附近的行为由 $x^{p-1}$ 决定，在 $x = 1$ 附近由 $(1-x)^{q-1}$ 决定。当 $p > 0$ 时 $x = 0$ 处的瑕积分收敛，当 $q > 0$ 时 $x = 1$ 处的瑕积分收敛。

### 14.5.2 基本性质

**对称性**：

$$B(p,q) = B(q,p)$$

**证明**：在 $B(p,q) = \int_0^1 x^{p-1}(1-x)^{q-1}\,dx$ 中令 $t = 1 - x$，则 $dx = -dt$，

$$B(p,q) = \int_1^0 (1-t)^{p-1}t^{q-1}(-dt) = \int_0^1 t^{q-1}(1-t)^{p-1}\,dt = B(q,p) \quad \square$$

**与Gamma函数的关系**：

$$\boxed{B(p,q) = \frac{\Gamma(p)\,\Gamma(q)}{\Gamma(p+q)}}$$

这是Beta函数最重要的性质，将Beta函数完全归结为Gamma函数。证明需要用到二重积分的技巧（此处从略）。

### 14.5.3 常用特殊值

| $B(p,q)$ | 值 |
|:---:|:---:|
| $B(1,1)$ | $1$ |
| $B\!\left(\dfrac{1}{2},\dfrac{1}{2}\right)$ | $\pi$ |
| $B(m,n)$（$m,n$ 为正整数） | $\dfrac{(m-1)!(n-1)!}{(m+n-1)!}$ |

其中 $B\!\left(\dfrac{1}{2},\dfrac{1}{2}\right) = \dfrac{\Gamma(1/2)\,\Gamma(1/2)}{\Gamma(1)} = \dfrac{\sqrt{\pi}\cdot\sqrt{\pi}}{1} = \pi$。

### 14.5.4 例题

> **例题 14.20** 计算 $\int_0^1 x^3(1-x)^4\,dx$。

**解**：由Beta函数定义，$\int_0^1 x^3(1-x)^4\,dx = B(4,5)$。

利用与Gamma函数的关系：

$$B(4,5) = \frac{\Gamma(4)\,\Gamma(5)}{\Gamma(9)} = \frac{3!\cdot 4!}{8!} = \frac{6 \times 24}{40320} = \frac{144}{40320} = \frac{1}{280}$$

$\square$

> **例题 14.21** 计算 $\int_0^1 \dfrac{1}{\sqrt{x(1-x)}}\,dx$。

**解**：被积函数可写为 $x^{-1/2}(1-x)^{-1/2}$，故

$$\int_0^1 \frac{1}{\sqrt{x(1-x)}}\,dx = B\!\left(\frac{1}{2},\frac{1}{2}\right) = \frac{\Gamma(1/2)\,\Gamma(1/2)}{\Gamma(1)} = \frac{\sqrt{\pi}\cdot\sqrt{\pi}}{1} = \pi$$

$\square$

---

## 本章小结

1. **无穷区间上的积分**：
   - 定义：$\int_a^{+\infty} f(x) \, dx = \lim\limits_{b \to +\infty} \int_a^b f(x) \, dx$
   - p-积分：$\int_1^{+\infty} \dfrac{1}{x^p} \, dx$ 当 $p > 1$ 时收敛，当 $p \leq 1$ 时发散

2. **无界函数的积分（瑕积分）**：
   - 瑕点在端点：通过极限定义
   - 瑕点在内部：分成两个积分
   - q-积分：$\int_0^1 \dfrac{1}{x^q} \, dx$ 当 $0 < q < 1$ 时收敛，当 $q \geq 1$ 时发散

3. **收敛判别法**：
   - 比较判别法：与已知敛散性的积分比较
   - 极限比较判别法：计算 $\lim \dfrac{f(x)}{g(x)}$，利用 p-积分或 q-积分
   - 绝对收敛必收敛

4. **Gamma函数**：
   - 定义：$\Gamma(s) = \int_0^{+\infty} x^{s-1} e^{-x} \, dx$（$s > 0$）
   - 递推公式：$\Gamma(s+1) = s\Gamma(s)$
   - 与阶乘关系：$\Gamma(n+1) = n!$
   - 重要值：$\Gamma(1) = 1$，$\Gamma\left(\dfrac{1}{2}\right) = \sqrt{\pi}$

5. **Beta函数**：
   - 定义：$B(p,q) = \int_0^1 x^{p-1}(1-x)^{q-1}\,dx$（$p,q > 0$）
   - 与Gamma函数的关系：$B(p,q) = \dfrac{\Gamma(p)\,\Gamma(q)}{\Gamma(p+q)}$
   - 对称性：$B(p,q) = B(q,p)$

---

## 深度学习应用

广义积分不仅是纯数学工具，在深度学习中也有深刻的应用。本节介绍几个核心场景。

### 14.5.1 Gamma函数与激活函数

**GELU（高斯误差线性单元）** 是 Transformer 架构中广泛使用的激活函数，其定义直接依赖误差函数（erf）：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$$

其中误差函数本身是广义积分：

$$\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \, dt$$

当 $x \to +\infty$ 时，$\text{erf}(x) \to 1$，对应 $\int_0^{+\infty} e^{-t^2} \, dt = \dfrac{\sqrt{\pi}}{2}$（即 $\Gamma\!\left(\dfrac{1}{2}\right)/2 = \dfrac{\sqrt{\pi}}{2}$）。

**Swish/SiLU** 激活函数 $\text{Swish}(x) = x \cdot \sigma(x) = \dfrac{x}{1+e^{-x}}$ 的积分性质：

$$\int_{-\infty}^{+\infty} \text{Swish}(x) \, e^{-x^2/2} \, dx$$

此积分收敛，体现了激活函数在 Gaussian 先验下的期望值，是 GELU 与 Swish 等价近似的理论基础。

### 14.5.2 正则化与广义积分

**$L_p$ 正则化** 的数学基础是幂函数的广义积分。对权重 $w \in \mathbb{R}$ 的单变量情形：

$$R_p(w) = \int |w|^p \, dw = \frac{|w|^{p+1}}{p+1} \cdot \text{sgn}(w) + C \quad (p > 0, \, p \neq -1)$$

- $p = 1$（Lasso）：对应拉普拉斯先验 $p(w) \propto e^{-\lambda |w|}$，归一化常数 $Z = \int_{-\infty}^{+\infty} e^{-\lambda|w|} \, dw = \dfrac{2}{\lambda}$
- $p = 2$（Ridge）：对应高斯先验，归一化常数 $Z = \sqrt{\dfrac{2\pi}{\lambda}}$

**贝叶斯先验的归一化常数** 本质上是广义积分，其收敛性决定了先验是否是合法概率分布：

$$\int_{-\infty}^{+\infty} e^{-\lambda|w|^p} \, dw = \frac{2}{p} \cdot \lambda^{-1/p} \cdot \Gamma\!\left(\frac{1}{p}\right)$$

此结论将 Gamma 函数与正则化理论统一起来。

### 14.5.3 概率分布的尾部行为

深度学习中的梯度、损失值常呈现**重尾分布**。广义积分的收敛性分析直接刻画了分布的尾部行为。

**$t$ 分布 vs 正态分布**：

- 正态分布密度 $f(x) = \dfrac{1}{\sqrt{2\pi}} e^{-x^2/2}$，其各阶矩 $\int_{-\infty}^{+\infty} |x|^k f(x) \, dx$ 对所有 $k \geq 0$ 均收敛。

- 自由度为 $\nu$ 的 $t$ 分布密度 $f_\nu(x) \propto \left(1 + \dfrac{x^2}{\nu}\right)^{-(\nu+1)/2}$，其 $k$ 阶矩的收敛性由广义积分判别：

$$\int_1^{+\infty} x^k \cdot x^{-(\nu+1)} \, dx = \int_1^{+\infty} x^{k - \nu - 1} \, dx$$

由 p-积分，当 $k - \nu - 1 < -1$，即 $k < \nu$ 时收敛。因此 $t$ 分布的 $k$ 阶矩仅在 $k < \nu$ 时存在，这正是重尾分布的数学特征。

**实际意义**：梯度噪声的重尾性（$t$ 分布类型）解释了为何 Adam 优化器比 SGD 更鲁棒——$t$ 分布尾部的积分发散意味着极大梯度更频繁出现，自适应学习率能有效应对。

### 14.5.4 Beta函数与注意力机制

**Beta函数** 是广义积分：

$$B(\alpha, \beta) = \int_0^1 t^{\alpha-1}(1-t)^{\beta-1} \, dt \quad (\alpha > 0, \, \beta > 0)$$

与 Gamma 函数的关系：

$$B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}$$

**Dirichlet 分布** 是 Beta 分布的多维推广，其密度函数为：

$$p(\mathbf{x}; \boldsymbol{\alpha}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{i=1}^K x_i^{\alpha_i - 1}, \quad \mathbf{x} \in \Delta^{K-1}$$

其中归一化常数 $B(\boldsymbol{\alpha}) = \dfrac{\prod_{i=1}^K \Gamma(\alpha_i)}{\Gamma\!\left(\sum_{i=1}^K \alpha_i\right)}$ 由 Gamma 函数给出。

**在注意力机制中**：Transformer 中的 softmax 注意力权重 $\mathbf{a} = \text{softmax}(\mathbf{q}^\top \mathbf{K}/\sqrt{d})$ 满足 $\sum_i a_i = 1, \, a_i > 0$，即落在单纯形 $\Delta^{K-1}$ 上。Dirichlet 分布是此单纯形上的自然先验，$\alpha_i$ 控制注意力的集中程度——$\alpha_i \to 0$ 时趋向稀疏注意力，$\alpha_i \to \infty$ 时趋向均匀注意力。

### 14.5.5 代码示例

```python
import torch
import torch.nn.functional as F
import math

# GELU 与误差函数
def gelu_exact(x):
    """GELU(x) = x * Φ(x)，其中 Φ 是标准正态 CDF"""
    return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))

x = torch.linspace(-3, 3, 100)
gelu_pytorch = F.gelu(x)
gelu_manual = gelu_exact(x)
print(f"GELU 实现误差: {(gelu_pytorch - gelu_manual).abs().max():.8f}")

# Gamma 函数验证
def gamma_numerical(s, n_terms=10000):
    """Γ(s) = ∫_0^∞ t^{s-1} e^{-t} dt 的数值近似"""
    t = torch.linspace(0.001, 50, n_terms)
    dt = t[1] - t[0]
    integrand = t**(s-1) * torch.exp(-t)
    return (integrand * dt).sum()

# 验证 Γ(5) = 4! = 24
gamma_5 = gamma_numerical(5.0)
print(f"Γ(5) 数值计算: {gamma_5.item():.2f}, 4! = 24")

# 正则化中的广义积分
def lp_regularization(weights, p=2):
    """L_p 正则化 = Σ|w|^p"""
    return weights.abs().pow(p).sum()
```

**运行说明**：
- `gelu_exact` 直接实现了 $\text{GELU}(x) = x \cdot \frac{1}{2}[1 + \text{erf}(x/\sqrt{2})]$，与 PyTorch 内置实现误差应接近机器精度（$\approx 10^{-7}$）
- `gamma_numerical` 用黎曼和近似 $\Gamma(s)$，验证了 $\Gamma(5) = 4! = 24$
- `lp_regularization` 展示了 $L_p$ 正则化如何对应权重的广义积分结构

---

## 练习题

**1.** 计算广义积分 $\int_0^{+\infty} xe^{-x} \, dx$。

**2.** 判断广义积分 $\int_1^{+\infty} \dfrac{1}{\sqrt{x^3 + 1}} \, dx$ 的敛散性。

**3.** 计算瑕积分 $\int_0^4 \dfrac{1}{\sqrt{4-x}} \, dx$。

**4.** 判断广义积分 $\int_0^1 \dfrac{\ln x}{\sqrt{x}} \, dx$ 的敛散性，若收敛，求其值。

**5.** 利用Gamma函数计算 $\int_0^{+\infty} x^4 e^{-x^2} \, dx$。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.** 使用分部积分或直接利用Gamma函数。

方法一（分部积分）：
$$\int_0^{+\infty} xe^{-x} \, dx = \lim_{b \to +\infty} \int_0^b xe^{-x} \, dx$$

设 $u = x$，$dv = e^{-x}dx$，则 $du = dx$，$v = -e^{-x}$：
$$\int_0^b xe^{-x} \, dx = \left[-xe^{-x}\right]_0^b + \int_0^b e^{-x} \, dx = -be^{-b} + \left[-e^{-x}\right]_0^b = -be^{-b} - e^{-b} + 1$$

当 $b \to +\infty$ 时，$be^{-b} \to 0$，$e^{-b} \to 0$，故原积分 $= 1$。

方法二（Gamma函数）：
$$\int_0^{+\infty} xe^{-x} \, dx = \Gamma(2) = 1! = 1$$

---

**2.** 当 $x \to +\infty$ 时，$\dfrac{1}{\sqrt{x^3 + 1}} \sim \dfrac{1}{x^{3/2}}$。

取 $g(x) = \dfrac{1}{x^{3/2}}$，则 $\lim\limits_{x \to +\infty} \dfrac{f(x)}{g(x)} = \lim\limits_{x \to +\infty} \dfrac{x^{3/2}}{\sqrt{x^3+1}} = 1$。

由于 $\int_1^{+\infty} \dfrac{1}{x^{3/2}} \, dx$ 收敛（$p = \dfrac{3}{2} > 1$），故原积分**收敛**。

---

**3.** $x = 4$ 是瑕点。
$$\int_0^4 \frac{1}{\sqrt{4-x}} \, dx = \lim_{\varepsilon \to 0^+} \int_0^{4-\varepsilon} (4-x)^{-1/2} \, dx$$

设 $u = 4 - x$，$du = -dx$：
$$= \lim_{\varepsilon \to 0^+} \int_4^\varepsilon u^{-1/2} \cdot (-du) = \lim_{\varepsilon \to 0^+} \int_\varepsilon^4 u^{-1/2} \, du = \lim_{\varepsilon \to 0^+} \left[2\sqrt{u}\right]_\varepsilon^4 = 4 - 0 = 4$$

---

**4.** $x = 0$ 是瑕点。

当 $x \to 0^+$ 时，$\dfrac{\ln x}{\sqrt{x}} = \dfrac{\ln x}{x^{1/2}}$。设 $f(x) = \dfrac{|\ln x|}{\sqrt{x}}$。

利用洛必达：$\lim\limits_{x \to 0^+} x^{1/4} \cdot \dfrac{|\ln x|}{x^{1/2}} = \lim\limits_{x \to 0^+} \dfrac{|\ln x|}{x^{1/4}} = \lim\limits_{x \to 0^+} \dfrac{1/x}{-\frac{1}{4}x^{-5/4}} = \lim\limits_{x \to 0^+} (-4x^{1/4}) = 0$

这说明 $\dfrac{|\ln x|}{\sqrt{x}} = o\left(\dfrac{1}{x^{3/4}}\right)$，由于 $\int_0^1 \dfrac{1}{x^{3/4}} dx$ 收敛（$q = \dfrac{3}{4} < 1$），原积分收敛。

计算：设 $t = \sqrt{x}$，则 $x = t^2$，$dx = 2t \, dt$：
$$\int_0^1 \frac{\ln x}{\sqrt{x}} \, dx = \int_0^1 \frac{2\ln t}{t} \cdot 2t \, dt = 4\int_0^1 \ln t \, dt = 4[t\ln t - t]_0^1 = 4(0 - 1 - 0) = -4$$

---

**5.** 设 $t = x^2$，则 $x = \sqrt{t}$，$dx = \dfrac{1}{2\sqrt{t}} dt$。

$$\int_0^{+\infty} x^4 e^{-x^2} \, dx = \int_0^{+\infty} t^2 e^{-t} \cdot \frac{1}{2\sqrt{t}} \, dt = \frac{1}{2}\int_0^{+\infty} t^{3/2} e^{-t} \, dt = \frac{1}{2}\Gamma\left(\frac{5}{2}\right)$$

由 $\Gamma\left(\dfrac{5}{2}\right) = \dfrac{3}{2} \cdot \dfrac{1}{2} \cdot \Gamma\left(\dfrac{1}{2}\right) = \dfrac{3}{4}\sqrt{\pi}$：

$$\int_0^{+\infty} x^4 e^{-x^2} \, dx = \frac{1}{2} \cdot \frac{3\sqrt{\pi}}{4} = \frac{3\sqrt{\pi}}{8}$$

</details>
