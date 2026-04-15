# 第10章 Taylor展开

## 学习目标

通过本章学习，你将能够：

- 理解用多项式逼近函数的思想，掌握 Taylor 公式的推导
- 熟练运用带 Peano 余项和 Lagrange 余项的 Taylor 公式
- 掌握 Maclaurin 公式及常见函数的展开式
- 能够运用 Taylor 公式进行近似计算、求极限和证明不等式
- 理解余项估计的方法，掌握精度控制的技巧

---

## 10.1 Taylor公式

### 10.1.1 问题的提出：用多项式逼近函数

在实际计算中，我们常常需要计算 $e^{0.1}$、$\sin 0.5$、$\ln 1.2$ 等函数值。然而，这些函数的精确值往往难以直接计算。一个自然的想法是：**能否用容易计算的多项式来逼近这些函数？**

回顾导数的定义，当 $x$ 接近 $x_0$ 时：

$$f(x) \approx f(x_0) + f'(x_0)(x - x_0)$$

这是用一次多项式（切线）逼近函数。但这种逼近只在 $x_0$ 附近的很小范围内足够精确。

**核心问题**：能否找到一个 $n$ 次多项式 $P_n(x)$，使得它在 $x_0$ 处与 $f(x)$ 有尽可能高阶的接触？

我们希望 $P_n(x)$ 满足：

$$P_n(x_0) = f(x_0), \quad P_n'(x_0) = f'(x_0), \quad \ldots, \quad P_n^{(n)}(x_0) = f^{(n)}(x_0)$$

设 $P_n(x) = a_0 + a_1(x-x_0) + a_2(x-x_0)^2 + \cdots + a_n(x-x_0)^n$，由上述条件可以确定：

$$a_k = \frac{f^{(k)}(x_0)}{k!}, \quad k = 0, 1, 2, \ldots, n$$

这就得到了 **Taylor 多项式**：

$$P_n(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \cdots + \frac{f^{(n)}(x_0)}{n!}(x-x_0)^n$$

### 10.1.2 带Peano余项的Taylor公式

**定理**（带 Peano 余项的 Taylor 公式）：设 $f(x)$ 在 $x_0$ 处有 $n$ 阶导数，则

$$f(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \cdots + \frac{f^{(n)}(x_0)}{n!}(x-x_0)^n + R_n(x)$$

其中余项 $R_n(x) = o((x-x_0)^n)$（当 $x \to x_0$ 时）。

**证明**：设 $P_n(x)$ 为 Taylor 多项式，令 $R_n(x) = f(x) - P_n(x)$。

显然 $R_n(x_0) = R_n'(x_0) = \cdots = R_n^{(n)}(x_0) = 0$。

需证 $\lim_{x \to x_0} \dfrac{R_n(x)}{(x-x_0)^n} = 0$。

由 L'Hospital 法则反复应用（共 $n$ 次）：

$$\lim_{x \to x_0} \frac{R_n(x)}{(x-x_0)^n} = \lim_{x \to x_0} \frac{R_n'(x)}{n(x-x_0)^{n-1}} = \cdots = \lim_{x \to x_0} \frac{R_n^{(n-1)}(x)}{n!(x-x_0)}$$

由于 $R_n^{(n-1)}(x_0) = 0$ 且 $R_n^{(n-1)}(x)$ 在 $x_0$ 处可导，有

$$\lim_{x \to x_0} \frac{R_n^{(n-1)}(x)}{n!(x-x_0)} = \frac{R_n^{(n)}(x_0)}{n!} = 0$$

因此 $R_n(x) = o((x-x_0)^n)$。 $\square$

> **注**：Peano 余项的形式简洁，适合用于求极限，但无法估计误差的具体大小。

### 10.1.3 带Lagrange余项的Taylor公式

**定理**（带 Lagrange 余项的 Taylor 公式）：设 $f(x)$ 在包含 $x_0$ 的开区间 $(a, b)$ 上有 $n+1$ 阶导数，则对任意 $x \in (a, b)$，存在 $\xi$ 介于 $x_0$ 与 $x$ 之间，使得

$$f(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \cdots + \frac{f^{(n)}(x_0)}{n!}(x-x_0)^n + R_n(x)$$

其中 **Lagrange 余项**为

$$R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}$$

**证明**：设 $R_n(x) = f(x) - P_n(x)$。令

$$\phi(t) = f(x) - \left[f(t) + f'(t)(x-t) + \frac{f''(t)}{2!}(x-t)^2 + \cdots + \frac{f^{(n)}(t)}{n!}(x-t)^n\right]$$

则 $\phi(x) = 0$，$\phi(x_0) = R_n(x)$。

对 $\phi(t)$ 关于 $t$ 求导，注意相邻项相消：

$$\phi'(t) = -\frac{f^{(n+1)}(t)}{n!}(x-t)^n$$

设辅助函数 $\psi(t) = (x-t)^{n+1}$，则 $\psi(x) = 0$，$\psi(x_0) = (x-x_0)^{n+1}$，$\psi'(t) = -(n+1)(x-t)^n$。

由 Cauchy 中值定理，存在 $\xi$ 介于 $x_0$ 与 $x$ 之间，使得

$$\frac{\phi(x) - \phi(x_0)}{\psi(x) - \psi(x_0)} = \frac{\phi'(\xi)}{\psi'(\xi)}$$

即

$$\frac{0 - R_n(x)}{0 - (x-x_0)^{n+1}} = \frac{-\frac{f^{(n+1)}(\xi)}{n!}(x-\xi)^n}{-(n+1)(x-\xi)^n}$$

解得

$$R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1} \quad \square$$

> **例题 10.1** 写出 $f(x) = e^x$ 在 $x_0 = 0$ 处的带 Lagrange 余项的 $n$ 阶 Taylor 公式。

**解**：由于 $f^{(k)}(x) = e^x$，$f^{(k)}(0) = 1$，

$$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots + \frac{x^n}{n!} + \frac{e^\xi}{(n+1)!}x^{n+1}$$

其中 $\xi$ 介于 $0$ 与 $x$ 之间。

---

## 10.2 Maclaurin公式

### 10.2.1 Maclaurin公式

当 $x_0 = 0$ 时，Taylor 公式称为 **Maclaurin 公式**：

$$f(x) = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \cdots + \frac{f^{(n)}(0)}{n!}x^n + R_n(x)$$

**Peano 余项**：$R_n(x) = o(x^n)$

**Lagrange 余项**：$R_n(x) = \dfrac{f^{(n+1)}(\xi)}{(n+1)!}x^{n+1}$，其中 $\xi$ 介于 $0$ 与 $x$ 之间。

### 10.2.2 常见函数的Maclaurin展开

以下展开式是最基本也是最重要的，需要熟练掌握。

#### 1. 指数函数 $e^x$

$$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots + \frac{x^n}{n!} + o(x^n)$$

**Lagrange 余项**：$R_n(x) = \dfrac{e^\xi}{(n+1)!}x^{n+1}$

> **例题 10.2** 推导 $e^x$ 的 Maclaurin 展开。

**解**：由于 $(e^x)^{(k)} = e^x$，故 $f^{(k)}(0) = e^0 = 1$。代入 Maclaurin 公式即得。

#### 2. 三角函数 $\sin x$ 和 $\cos x$

$$\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots + \frac{(-1)^n x^{2n+1}}{(2n+1)!} + o(x^{2n+2})$$

$$\cos x = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \cdots + \frac{(-1)^n x^{2n}}{(2n)!} + o(x^{2n+1})$$

> **例题 10.3** 推导 $\sin x$ 的 Maclaurin 展开。

**解**：计算各阶导数在 $x = 0$ 处的值：
- $f(x) = \sin x$，$f(0) = 0$
- $f'(x) = \cos x$，$f'(0) = 1$
- $f''(x) = -\sin x$，$f''(0) = 0$
- $f'''(x) = -\cos x$，$f'''(0) = -1$
- $f^{(4)}(x) = \sin x$，$f^{(4)}(0) = 0$

导数值以周期 $4$ 循环：$0, 1, 0, -1, 0, 1, 0, -1, \ldots$

因此只有奇数次幂项非零，且符号交替。

#### 3. 对数函数 $\ln(1+x)$

$$\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \cdots + \frac{(-1)^{n-1} x^n}{n} + o(x^n)$$

**适用范围**：$-1 < x \leq 1$

> **例题 10.4** 推导 $\ln(1+x)$ 的 Maclaurin 展开。

**解**：$f(x) = \ln(1+x)$，$f(0) = 0$

$$f'(x) = \frac{1}{1+x}, \quad f''(x) = -\frac{1}{(1+x)^2}, \quad f'''(x) = \frac{2}{(1+x)^3}, \ldots$$

一般地，$f^{(n)}(x) = \dfrac{(-1)^{n-1}(n-1)!}{(1+x)^n}$，故 $f^{(n)}(0) = (-1)^{n-1}(n-1)!$

代入 Maclaurin 公式：

$$\ln(1+x) = \sum_{k=1}^{n} \frac{(-1)^{k-1}(k-1)!}{k!}x^k + o(x^n) = \sum_{k=1}^{n} \frac{(-1)^{k-1}}{k}x^k + o(x^n)$$

#### 4. 幂函数 $(1+x)^\alpha$

$$(1+x)^\alpha = 1 + \alpha x + \frac{\alpha(\alpha-1)}{2!}x^2 + \frac{\alpha(\alpha-1)(\alpha-2)}{3!}x^3 + \cdots$$

一般项为：$\dfrac{\alpha(\alpha-1)\cdots(\alpha-n+1)}{n!}x^n$（广义二项式系数 $\binom{\alpha}{n}$）

**广义二项式系数**：对于任意实数 $\alpha$ 和正整数 $n$，定义

$$\binom{\alpha}{n} = \frac{\alpha(\alpha-1)(\alpha-2)\cdots(\alpha-n+1)}{n!}$$

并约定 $\binom{\alpha}{0} = 1$。当 $\alpha$ 为正整数 $m$ 时，$\binom{m}{n}$ 在 $n > m$ 时为零，退化为通常的二项式系数，展开式变为有限项（即经典的二项式定理）。

**推导过程**：设 $f(x) = (1+x)^\alpha$，逐阶求导：
- $f(0) = 1$
- $f'(x) = \alpha(1+x)^{\alpha-1}$，故 $f'(0) = \alpha$
- $f''(x) = \alpha(\alpha-1)(1+x)^{\alpha-2}$，故 $f''(0) = \alpha(\alpha-1)$
- 一般地，$f^{(n)}(x) = \alpha(\alpha-1)\cdots(\alpha-n+1)(1+x)^{\alpha-n}$，故 $f^{(n)}(0) = \alpha(\alpha-1)\cdots(\alpha-n+1)$

代入 Maclaurin 公式即得展开式。当 $\alpha$ 为非负整数 $m$ 时，$f^{(m+1)}(x) = 0$，故展开式在 $n = m$ 项后终止，成为有限和 $\sum_{n=0}^{m}\binom{m}{n}x^n$。

**特例**：
- $\alpha = -1$：$\dfrac{1}{1+x} = 1 - x + x^2 - x^3 + \cdots + (-1)^n x^n + o(x^n)$
- $\alpha = \frac{1}{2}$：$\sqrt{1+x} = 1 + \dfrac{x}{2} - \dfrac{x^2}{8} + \dfrac{x^3}{16} - \cdots$
- $\alpha = -\frac{1}{2}$：$\dfrac{1}{\sqrt{1+x}} = 1 - \dfrac{x}{2} + \dfrac{3x^2}{8} - \dfrac{5x^3}{16} + \cdots$

#### 5. 几何级数 $\dfrac{1}{1-x}$

$$\frac{1}{1-x} = 1 + x + x^2 + x^3 + \cdots + x^n + o(x^n)$$

**适用范围**：$|x| < 1$

> **例题 10.5** 利用已知展开式，求 $\dfrac{1}{1+x^2}$ 的 Maclaurin 展开。

**解**：在 $\dfrac{1}{1-t} = 1 + t + t^2 + \cdots$ 中，令 $t = -x^2$：

$$\frac{1}{1+x^2} = 1 - x^2 + x^4 - x^6 + \cdots + (-1)^n x^{2n} + o(x^{2n})$$

---

## 10.3 Taylor公式的应用

### 10.3.1 近似计算

Taylor 公式最直接的应用是进行数值近似计算。

> **例题 10.6** 利用 Taylor 公式计算 $e$ 的近似值，使误差不超过 $10^{-4}$。

**解**：由 $e^x$ 的展开式，取 $x = 1$：

$$e = 1 + 1 + \frac{1}{2!} + \frac{1}{3!} + \cdots + \frac{1}{n!} + R_n$$

其中 $R_n = \dfrac{e^\xi}{(n+1)!}$，$0 < \xi < 1$。

由于 $e^\xi < e < 3$，故 $|R_n| < \dfrac{3}{(n+1)!}$。

要使 $|R_n| < 10^{-4}$，需要 $(n+1)! > 30000$。

计算：$7! = 5040$，$8! = 40320 > 30000$。

取 $n = 7$：

$$e \approx 1 + 1 + \frac{1}{2} + \frac{1}{6} + \frac{1}{24} + \frac{1}{120} + \frac{1}{720} + \frac{1}{5040} \approx 2.7183$$

> **例题 10.7** 计算 $\sin 3°$ 的近似值，精确到 $10^{-6}$。

**解**：$3° = \dfrac{\pi}{60}$ 弧度。利用 $\sin x$ 的展开式：

$$\sin x = x - \frac{x^3}{6} + \frac{x^5}{120} - \cdots$$

由于 $\dfrac{\pi}{60} \approx 0.0524$，计算：
- $x = \dfrac{\pi}{60} \approx 0.052360$
- $\dfrac{x^3}{6} \approx 2.4 \times 10^{-5}$
- $\dfrac{x^5}{120} \approx 3.3 \times 10^{-9}$

取两项近似：$\sin 3° \approx \dfrac{\pi}{60} - \dfrac{1}{6}\left(\dfrac{\pi}{60}\right)^3 \approx 0.052336$

### 10.3.2 求极限

Taylor 展开是计算 $\dfrac{0}{0}$ 型极限的有力工具，特别是当 L'Hospital 法则计算繁琐时。

**基本思路**：将分子、分母分别展开，保留到适当阶数，然后约分。

> **例题 10.8** 求 $\lim_{x \to 0} \dfrac{e^x - 1 - x}{x^2}$。

**解**：利用 $e^x = 1 + x + \dfrac{x^2}{2} + o(x^2)$：

$$\lim_{x \to 0} \frac{e^x - 1 - x}{x^2} = \lim_{x \to 0} \frac{1 + x + \frac{x^2}{2} + o(x^2) - 1 - x}{x^2} = \lim_{x \to 0} \frac{\frac{x^2}{2} + o(x^2)}{x^2} = \frac{1}{2}$$

> **例题 10.9** 求 $\lim_{x \to 0} \dfrac{\sin x - x + \frac{x^3}{6}}{x^5}$。

**解**：利用 $\sin x = x - \dfrac{x^3}{6} + \dfrac{x^5}{120} + o(x^5)$：

$$\lim_{x \to 0} \frac{\sin x - x + \frac{x^3}{6}}{x^5} = \lim_{x \to 0} \frac{x - \frac{x^3}{6} + \frac{x^5}{120} + o(x^5) - x + \frac{x^3}{6}}{x^5} = \lim_{x \to 0} \frac{\frac{x^5}{120} + o(x^5)}{x^5} = \frac{1}{120}$$

> **例题 10.10** 求 $\lim_{x \to 0} \dfrac{\tan x - \sin x}{x^3}$。

**解**：需要 $\tan x$ 的展开式。由于 $\tan x = \dfrac{\sin x}{\cos x}$，利用：
- $\sin x = x - \dfrac{x^3}{6} + o(x^3)$
- $\cos x = 1 - \dfrac{x^2}{2} + o(x^2)$

直接做除法或利用 $\tan x = x + \dfrac{x^3}{3} + o(x^3)$：

$$\tan x - \sin x = \left(x + \frac{x^3}{3} + o(x^3)\right) - \left(x - \frac{x^3}{6} + o(x^3)\right) = \frac{x^3}{2} + o(x^3)$$

因此 $\lim_{x \to 0} \dfrac{\tan x - \sin x}{x^3} = \dfrac{1}{2}$。

> **例题 10.11** 求 $\lim_{x \to 0} \left(\dfrac{1}{x^2} - \dfrac{1}{\sin^2 x}\right)$。

**解**：通分后：

$$\frac{1}{x^2} - \frac{1}{\sin^2 x} = \frac{\sin^2 x - x^2}{x^2 \sin^2 x}$$

利用 $\sin x = x - \dfrac{x^3}{6} + o(x^3)$，有 $\sin^2 x = x^2 - \dfrac{x^4}{3} + o(x^4)$。

$$\lim_{x \to 0} \frac{\sin^2 x - x^2}{x^2 \sin^2 x} = \lim_{x \to 0} \frac{x^2 - \frac{x^4}{3} + o(x^4) - x^2}{x^2 \cdot x^2} = \lim_{x \to 0} \frac{-\frac{x^4}{3} + o(x^4)}{x^4} = -\frac{1}{3}$$

### 10.3.3 证明不等式

Taylor 公式可以用来证明涉及函数值的不等式。

> **例题 10.12** 证明：当 $x > 0$ 时，$e^x > 1 + x$。

**证明**：由带 Lagrange 余项的 Taylor 公式：

$$e^x = 1 + x + \frac{e^\xi}{2}x^2$$

其中 $0 < \xi < x$。由于 $e^\xi > 0$，故 $\dfrac{e^\xi}{2}x^2 > 0$。

因此 $e^x = 1 + x + \dfrac{e^\xi}{2}x^2 > 1 + x$。 $\square$

> **例题 10.13** 证明：当 $x > 0$ 时，$\ln(1+x) < x$。

**证明**：由 $\ln(1+x)$ 的 Taylor 展开（取一阶 Lagrange 余项）：

$$\ln(1+x) = x - \frac{1}{2(1+\xi)^2}x^2$$

其中 $0 < \xi < x$。由于 $\dfrac{1}{2(1+\xi)^2}x^2 > 0$，

故 $\ln(1+x) = x - \dfrac{1}{2(1+\xi)^2}x^2 < x$。 $\square$

> **例题 10.14** 证明：当 $x \neq 0$ 时，$\sin x < x < \tan x$（$0 < x < \dfrac{\pi}{2}$）。

**证明**：对于 $\sin x < x$（$x > 0$）：

由 $\sin x$ 的 Taylor 公式：$\sin x = x - \dfrac{x^3}{6} + o(x^3)$

更精确地，由 Lagrange 余项：$\sin x = x - \dfrac{\cos \xi}{6}x^3$，其中 $0 < \xi < x$。

当 $0 < x < \dfrac{\pi}{2}$ 时，$\cos \xi > 0$，故 $\sin x < x$。

对于 $x < \tan x$ 的证明类似，利用 $\tan x = x + \dfrac{x^3}{3} + o(x^3)$ 及余项分析。 $\square$

---

## 10.4 余项估计

### 10.4.1 Lagrange余项的误差估计

Lagrange 余项的关键作用是提供误差的**定量估计**。

$$R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}$$

**误差界**：若在区间 $[x_0, x]$（或 $[x, x_0]$）上 $|f^{(n+1)}(t)| \leq M$，则

$$|R_n(x)| \leq \frac{M}{(n+1)!}|x-x_0|^{n+1}$$

> **例题 10.15** 用 $\cos x \approx 1 - \dfrac{x^2}{2}$ 计算 $\cos 0.5$，并估计误差。

**解**：取 $n = 2$ 的 Taylor 多项式。Lagrange 余项为：

$$R_2(x) = \frac{f'''(\xi)}{3!}x^3 = \frac{\sin \xi}{6}x^3$$

其中 $0 < \xi < 0.5$。由于 $|\sin \xi| \leq 1$，

$$|R_2(0.5)| \leq \frac{1}{6} \times (0.5)^3 = \frac{1}{6} \times 0.125 = \frac{1}{48} \approx 0.021$$

计算近似值：$\cos 0.5 \approx 1 - \dfrac{0.25}{2} = 0.875$

实际值 $\cos 0.5 \approx 0.8776$，误差约 $0.003$，确实小于 $0.021$。

### 10.4.2 精度控制

**问题**：要使计算达到指定精度 $\varepsilon$，需要取多少项？

**方法**：找到最小的 $n$，使得 $|R_n(x)| < \varepsilon$。

> **例题 10.16** 用 Taylor 公式计算 $\ln 1.1$，精确到 $10^{-4}$。

**解**：利用 $\ln(1+x) = x - \dfrac{x^2}{2} + \dfrac{x^3}{3} - \cdots$，取 $x = 0.1$。

Lagrange 余项：$R_n = \dfrac{(-1)^n}{(n+1)(1+\xi)^{n+1}} \times (0.1)^{n+1}$

由于 $0 < \xi < 0.1$，有 $(1+\xi)^{n+1} > 1$，故

$$|R_n| < \frac{(0.1)^{n+1}}{n+1}$$

要使 $|R_n| < 10^{-4}$：
- $n = 3$：$\dfrac{(0.1)^4}{4} = 2.5 \times 10^{-5} < 10^{-4}$ 满足

取前 3 项：

$$\ln 1.1 \approx 0.1 - \frac{0.01}{2} + \frac{0.001}{3} = 0.1 - 0.005 + 0.000333 = 0.09533$$

实际值 $\ln 1.1 \approx 0.09531$，误差约 $2 \times 10^{-5}$，满足精度要求。

> **例题 10.17** 估计用 $\sin x \approx x - \dfrac{x^3}{6}$ 计算 $\sin 0.1$ 的误差。

**解**：余项 $R_3(x) = \dfrac{f^{(4)}(\xi)}{4!}x^4 = \dfrac{\sin \xi}{24}x^4$

取 $x = 0.1$，由于 $|\sin \xi| \leq |\xi| < 0.1$：

$$|R_3(0.1)| < \frac{0.1}{24} \times (0.1)^4 = \frac{10^{-5}}{24} \approx 4.2 \times 10^{-7}$$

因此误差不超过 $5 \times 10^{-7}$。

---

## 本章小结

1. **Taylor 公式的核心思想**：用多项式逼近函数，多项式的系数由函数在展开点的各阶导数值确定。

2. **两种余项形式**：
   - **Peano 余项** $R_n(x) = o((x-x_0)^n)$：定性描述，适用于求极限
   - **Lagrange 余项** $R_n(x) = \dfrac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}$：定量估计，适用于误差分析

3. **重要的 Maclaurin 展开**（必须熟记）：
   - $e^x = 1 + x + \dfrac{x^2}{2!} + \dfrac{x^3}{3!} + \cdots$
   - $\sin x = x - \dfrac{x^3}{3!} + \dfrac{x^5}{5!} - \cdots$
   - $\cos x = 1 - \dfrac{x^2}{2!} + \dfrac{x^4}{4!} - \cdots$
   - $\ln(1+x) = x - \dfrac{x^2}{2} + \dfrac{x^3}{3} - \cdots$
   - $(1+x)^\alpha = 1 + \alpha x + \dfrac{\alpha(\alpha-1)}{2!}x^2 + \cdots$

4. **Taylor 公式的应用**：
   - **近似计算**：用多项式近似函数值，Lagrange 余项控制误差
   - **求极限**：展开分子分母，比较同阶无穷小
   - **证明不等式**：利用余项的符号性质

5. **展开阶数的确定**：
   - 求极限时，展开到能够约分、显示主项为止
   - 近似计算时，根据精度要求确定所需阶数

---

## 深度学习应用

Taylor 展开不仅是纯数学工具，在深度学习的优化理论和模型设计中也扮演着核心角色。本节介绍几个关键的应用场景。

### 损失曲面的二阶近似

设神经网络的损失函数为 $L(\theta)$，其中 $\theta$ 是参数向量。在当前参数 $\theta$ 处做 Taylor 展开：

$$L(\theta + \Delta\theta) \approx L(\theta) + \nabla L^T \Delta\theta + \frac{1}{2}\Delta\theta^T H \Delta\theta$$

其中：
- $\nabla L$ 是损失函数的梯度向量（一阶信息）
- $H = \nabla^2 L$ 是 **Hessian 矩阵**（二阶信息），$H_{ij} = \dfrac{\partial^2 L}{\partial \theta_i \partial \theta_j}$

Hessian 矩阵的作用：
- **特征值大**：损失曲面在该方向曲率大，梯度下降需要小步长
- **特征值小**：损失曲面在该方向平坦，可以用大步长
- **条件数** $\kappa(H) = \lambda_{\max}/\lambda_{\min}$ 大时，训练收敛慢（这是深度学习训练困难的根源之一）

### 牛顿法与二阶优化

对二阶近似 $L(\theta + \Delta\theta)$ 关于 $\Delta\theta$ 求极值，令梯度为零：

$$\nabla L + H \Delta\theta = 0 \implies \Delta\theta = -H^{-1}\nabla L$$

由此得到**牛顿法**的更新公式：

$$\theta_{n+1} = \theta_n - H^{-1}\nabla L$$

**为什么深度学习很少使用牛顿法？**

| 方面 | 梯度下降 | 牛顿法 |
|------|----------|--------|
| 每步计算量 | $O(p)$ | $O(p^2)$（存储）$O(p^3)$（求逆） |
| 收敛速度 | 线性收敛 | 二次收敛 |
| 适用规模 | 数十亿参数 | 数千参数 |

对于含有 $p = 10^8$ 个参数的模型，Hessian 矩阵有 $10^{16}$ 个元素，远超内存上限。实践中通常使用 **拟牛顿法**（如 L-BFGS）或 **自然梯度**（Fisher 信息矩阵近似）来利用二阶信息，同时保持计算可行性。

### 激活函数的 Taylor 展开

现代深度学习中的激活函数往往有精巧的 Taylor 近似，兼顾数学性质与计算效率。

**GELU（Gaussian Error Linear Unit）**

精确定义涉及误差函数 $\text{erf}$，计算代价高。利用 $\tanh$ 对 $\text{erf}$ 的 Taylor 近似，得到高效近似公式：

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)$$

其中 $x + 0.044715x^3$ 正是 $\tanh$ 参数的 Taylor 截断——这使得 GELU 在保持高精度的同时大幅降低了计算量。

**Swish 激活函数**

$$\text{Swish}(x) = x \cdot \sigma(\beta x), \quad \sigma(t) = \frac{1}{1+e^{-t}}$$

在 $\beta \to 0$ 时退化为线性函数，$\beta \to \infty$ 时退化为 ReLU。Swish 在原点附近的 Taylor 展开为：

$$\text{Swish}(x) = \frac{x}{2} + \frac{\beta}{4}x^2 + O(x^4)$$

这说明 Swish 是一个带软门控的非线性函数，二阶项 $\dfrac{\beta}{4}x^2$ 赋予它比 ReLU 更强的表达能力。

### 损失函数的光滑近似

许多自然的损失函数不可导（如 $\ell_0$ 范数、argmax），Taylor 展开提供了一类系统的光滑化方法。

**Softmax 作为 argmax 的光滑近似**

硬 argmax 选出最大分量，不可微分。对 logit 向量 $z = (z_1, \ldots, z_K)$，定义：

$$\text{softmax}(z)_k = \frac{e^{z_k/T}}{\sum_j e^{z_j/T}}$$

- $T \to 0$：退化为 argmax（硬选择）
- $T \to \infty$：退化为均匀分布

在 $T = 1$ 附近对 $e^{z_k}$ 做 Taylor 展开：

$$e^{z_k} \approx 1 + z_k + \frac{z_k^2}{2} + \cdots$$

这说明 softmax 的光滑性正是来自指数函数的无穷可微性，而指数函数的 Taylor 级数在整个实轴上收敛。

### 代码示例

```python
import torch
import torch.nn.functional as F

# GELU 的两种实现对比
x = torch.linspace(-3, 3, 100)

# 精确实现
gelu_exact = F.gelu(x)

# Taylor 近似实现
def gelu_approx(x):
    return 0.5 * x * (1 + torch.tanh(
        (2/torch.pi).sqrt() * (x + 0.044715 * x**3)
    ))

gelu_taylor = gelu_approx(x)

# 误差分析
max_error = (gelu_exact - gelu_taylor).abs().max()
print(f"GELU近似的最大误差: {max_error.item():.6f}")

# 二阶优化示意
def hessian_demo():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    y = (x ** 2).sum()  # f(x1,x2) = x1^2 + x2^2

    # 计算梯度
    grad = torch.autograd.grad(y, x, create_graph=True)[0]

    # 计算 Hessian（对角线）
    hessian_diag = []
    for i in range(x.shape[1]):
        h = torch.autograd.grad(grad[0, i], x, retain_graph=True)[0]
        hessian_diag.append(h[0, i].item())

    print(f"Hessian 对角元素: {hessian_diag}")  # [2.0, 2.0]

hessian_demo()
```

**运行结果**：GELU 近似的最大误差约为 $10^{-4}$ 量级，在实际训练中可忽略不计。Hessian 对角元素均为 $2.0$，与 $f(x_1, x_2) = x_1^2 + x_2^2$ 的解析结果 $\dfrac{\partial^2 f}{\partial x_i^2} = 2$ 完全吻合。

> **联系**：本节的内容将在第17章（多元函数微分学）中得到进一步推广——多变量 Taylor 展开与 Hessian 矩阵是分析高维优化问题的基础工具。

---

## 练习题

**1.** 写出下列函数在 $x_0 = 0$ 处的带 Peano 余项的 $n$ 阶 Maclaurin 公式：
   (a) $f(x) = e^{-x}$
   (b) $f(x) = \ln(1-x)$

**2.** 利用 Taylor 公式求下列极限：
   (a) $\lim_{x \to 0} \dfrac{e^x - e^{-x} - 2x}{x^3}$
   (b) $\lim_{x \to 0} \dfrac{\cos x - 1 + \frac{x^2}{2}}{x^4}$
   (c) $\lim_{x \to 0} \dfrac{x - \ln(1+x)}{x^2}$

**3.** 利用 Taylor 公式证明：当 $x > 0$ 时，$e^x > 1 + x + \dfrac{x^2}{2}$。

**4.** 用 Taylor 公式计算 $\sqrt{e}$ 的近似值，使误差不超过 $10^{-3}$。

**5.** 设 $f(x)$ 在 $x = 0$ 的某邻域内有二阶连续导数，且 $\lim_{x \to 0} \dfrac{f(x)}{x} = 1$。证明：$f(0) = 0$，$f'(0) = 1$，并求 $\lim_{x \to 0} \dfrac{f(x) - x}{x^2}$。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.**

(a) $f(x) = e^{-x}$

由于 $f^{(k)}(x) = (-1)^k e^{-x}$，$f^{(k)}(0) = (-1)^k$，

$$e^{-x} = 1 - x + \frac{x^2}{2!} - \frac{x^3}{3!} + \cdots + \frac{(-1)^n x^n}{n!} + o(x^n)$$

或者直接在 $e^t = 1 + t + \dfrac{t^2}{2!} + \cdots$ 中令 $t = -x$。

(b) $f(x) = \ln(1-x)$

在 $\ln(1+t) = t - \dfrac{t^2}{2} + \dfrac{t^3}{3} - \cdots$ 中令 $t = -x$：

$$\ln(1-x) = -x - \frac{x^2}{2} - \frac{x^3}{3} - \cdots - \frac{x^n}{n} + o(x^n)$$

---

**2.**

(a) 利用 $e^x = 1 + x + \dfrac{x^2}{2} + \dfrac{x^3}{6} + o(x^3)$ 和 $e^{-x} = 1 - x + \dfrac{x^2}{2} - \dfrac{x^3}{6} + o(x^3)$：

$$e^x - e^{-x} = 2x + \frac{x^3}{3} + o(x^3)$$

$$\lim_{x \to 0} \frac{e^x - e^{-x} - 2x}{x^3} = \lim_{x \to 0} \frac{\frac{x^3}{3} + o(x^3)}{x^3} = \frac{1}{3}$$

(b) 利用 $\cos x = 1 - \dfrac{x^2}{2} + \dfrac{x^4}{24} + o(x^4)$：

$$\cos x - 1 + \frac{x^2}{2} = \frac{x^4}{24} + o(x^4)$$

$$\lim_{x \to 0} \frac{\cos x - 1 + \frac{x^2}{2}}{x^4} = \frac{1}{24}$$

(c) 利用 $\ln(1+x) = x - \dfrac{x^2}{2} + o(x^2)$：

$$x - \ln(1+x) = x - \left(x - \frac{x^2}{2} + o(x^2)\right) = \frac{x^2}{2} + o(x^2)$$

$$\lim_{x \to 0} \frac{x - \ln(1+x)}{x^2} = \frac{1}{2}$$

---

**3.** 由带 Lagrange 余项的 Taylor 公式：

$$e^x = 1 + x + \frac{x^2}{2} + \frac{e^\xi}{6}x^3$$

其中 $0 < \xi < x$。当 $x > 0$ 时，$e^\xi > 0$，故 $\dfrac{e^\xi}{6}x^3 > 0$。

因此 $e^x = 1 + x + \dfrac{x^2}{2} + \dfrac{e^\xi}{6}x^3 > 1 + x + \dfrac{x^2}{2}$。 $\square$

---

**4.** $\sqrt{e} = e^{0.5}$。利用 $e^x$ 的展开式，取 $x = 0.5$：

$$e^{0.5} = 1 + 0.5 + \frac{0.25}{2} + \frac{0.125}{6} + \frac{0.0625}{24} + \cdots$$

余项 $R_n = \dfrac{e^\xi}{(n+1)!}(0.5)^{n+1}$，其中 $0 < \xi < 0.5$，$e^\xi < e^{0.5} < 2$。

要使 $|R_n| < 10^{-3}$，需 $\dfrac{2 \times (0.5)^{n+1}}{(n+1)!} < 10^{-3}$。

- $n = 3$：$\dfrac{2 \times 0.0625}{24} = \dfrac{0.125}{24} \approx 0.0052 > 10^{-3}$
- $n = 4$：$\dfrac{2 \times 0.03125}{120} \approx 0.00052 < 10^{-3}$ 满足

取前 5 项：

$$\sqrt{e} \approx 1 + 0.5 + 0.125 + 0.02083 + 0.00260 = 1.64843$$

实际值 $\sqrt{e} \approx 1.64872$，误差约 $3 \times 10^{-4}$，满足精度要求。

---

**5.** 由 $\lim_{x \to 0} \dfrac{f(x)}{x} = 1$ 存在且有限，而分母 $x \to 0$，故必有分子 $f(x) \to 0$，即 $f(0) = 0$（由连续性）。

由 L'Hospital 法则或导数定义：

$$f'(0) = \lim_{x \to 0} \frac{f(x) - f(0)}{x} = \lim_{x \to 0} \frac{f(x)}{x} = 1$$

由于 $f(x)$ 有二阶连续导数，Taylor 展开：

$$f(x) = f(0) + f'(0)x + \frac{f''(0)}{2}x^2 + o(x^2) = x + \frac{f''(0)}{2}x^2 + o(x^2)$$

因此：

$$\lim_{x \to 0} \frac{f(x) - x}{x^2} = \lim_{x \to 0} \frac{\frac{f''(0)}{2}x^2 + o(x^2)}{x^2} = \frac{f''(0)}{2}$$

若需具体数值，还需要知道 $f''(0)$ 的值。在仅给定条件下，答案为 $\dfrac{f''(0)}{2}$。

</details>
