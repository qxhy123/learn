# 第8章 求导法则

## 学习目标

通过本章学习，你将能够：

- 掌握导数的四则运算法则：和差法则、积的法则、商的法则
- 理解并熟练运用链式法则求复合函数的导数
- 掌握反函数求导法则，能推导反三角函数的导数
- 掌握隐函数求导法和对数求导法
- 学会参数方程的求导方法，包括二阶导数的计算
- 理解高阶导数的概念，掌握莱布尼茨公式和常见函数的 $n$ 阶导数

---

## 8.1 导数的四则运算

### 8.1.1 和差法则

**定理**（和差法则）：若 $f(x)$ 和 $g(x)$ 在点 $x$ 处可导，则 $f(x) \pm g(x)$ 也在 $x$ 处可导，且

$$(f \pm g)'(x) = f'(x) \pm g'(x)$$

**证明**：由导数定义，

$$\lim_{h \to 0} \frac{[f(x+h) \pm g(x+h)] - [f(x) \pm g(x)]}{h} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} \pm \lim_{h \to 0} \frac{g(x+h) - g(x)}{h} = f'(x) \pm g'(x)$$

$\square$

**推广**：对于有限个可导函数的和差，有

$$(f_1 \pm f_2 \pm \cdots \pm f_n)' = f_1' \pm f_2' \pm \cdots \pm f_n'$$

> **例题 8.1** 求 $y = x^3 + 2x^2 - 5x + 1$ 的导数。

**解**：

$$y' = (x^3)' + (2x^2)' - (5x)' + (1)' = 3x^2 + 4x - 5$$

### 8.1.2 积的法则

**定理**（乘积法则/Leibniz 法则）：若 $f(x)$ 和 $g(x)$ 在点 $x$ 处可导，则 $f(x) \cdot g(x)$ 也在 $x$ 处可导，且

$$(f \cdot g)'(x) = f'(x) g(x) + f(x) g'(x)$$

**证明**：

$$\begin{aligned}
(fg)'(x) &= \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x)g(x)}{h} \\
&= \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x)g(x+h) + f(x)g(x+h) - f(x)g(x)}{h} \\
&= \lim_{h \to 0} \frac{[f(x+h) - f(x)]g(x+h)}{h} + \lim_{h \to 0} \frac{f(x)[g(x+h) - g(x)]}{h} \\
&= f'(x) \cdot \lim_{h \to 0} g(x+h) + f(x) \cdot g'(x) \\
&= f'(x) g(x) + f(x) g'(x)
\end{aligned}$$

其中用到了可导必连续：$\lim_{h \to 0} g(x+h) = g(x)$。 $\square$

**推论**：$(cf)' = cf'$，其中 $c$ 是常数。

> **例题 8.2** 求 $y = x^2 e^x$ 的导数。

**解**：设 $f(x) = x^2$，$g(x) = e^x$，则

$$y' = (x^2)' e^x + x^2 (e^x)' = 2x e^x + x^2 e^x = (x^2 + 2x)e^x$$

**推广**：对于三个函数的乘积，

$$(fgh)' = f'gh + fg'h + fgh'$$

### 8.1.3 商的法则

**定理**（商的法则）：若 $f(x)$ 和 $g(x)$ 在点 $x$ 处可导，且 $g(x) \neq 0$，则 $\dfrac{f(x)}{g(x)}$ 也在 $x$ 处可导，且

$$\left(\frac{f}{g}\right)'(x) = \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}$$

**证明**：

$$\begin{aligned}
\left(\frac{f}{g}\right)'(x) &= \lim_{h \to 0} \frac{\frac{f(x+h)}{g(x+h)} - \frac{f(x)}{g(x)}}{h} = \lim_{h \to 0} \frac{f(x+h)g(x) - f(x)g(x+h)}{h \cdot g(x+h)g(x)} \\
&= \lim_{h \to 0} \frac{[f(x+h) - f(x)]g(x) - f(x)[g(x+h) - g(x)]}{h \cdot g(x+h)g(x)} \\
&= \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}
\end{aligned}$$

$\square$

**特例**：$\left(\dfrac{1}{g}\right)' = -\dfrac{g'}{g^2}$

> **例题 8.3** 求 $y = \tan x$ 的导数。

**解**：

$$(\tan x)' = \left(\frac{\sin x}{\cos x}\right)' = \frac{(\sin x)' \cos x - \sin x (\cos x)'}{\cos^2 x} = \frac{\cos^2 x + \sin^2 x}{\cos^2 x} = \frac{1}{\cos^2 x} = \sec^2 x$$

---

## 8.2 链式法则

### 8.2.1 复合函数的导数

**定理**（链式法则）：设 $y = f(u)$，$u = g(x)$。若 $g(x)$ 在点 $x$ 可导，$f(u)$ 在点 $u = g(x)$ 可导，则复合函数 $y = f(g(x))$ 在点 $x$ 可导，且

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = f'(g(x)) \cdot g'(x)$$

**证明**：令 $\Delta u = g(x + \Delta x) - g(x)$。由于 $g$ 可导（从而连续），当 $\Delta x \to 0$ 时，$\Delta u \to 0$。

当 $\Delta u \neq 0$ 时，

$$\frac{\Delta y}{\Delta x} = \frac{\Delta y}{\Delta u} \cdot \frac{\Delta u}{\Delta x}$$

当 $\Delta x \to 0$ 时，$\Delta u \to 0$，从而

$$\frac{dy}{dx} = \lim_{\Delta x \to 0} \frac{\Delta y}{\Delta x} = \lim_{\Delta u \to 0} \frac{\Delta y}{\Delta u} \cdot \lim_{\Delta x \to 0} \frac{\Delta u}{\Delta x} = f'(u) \cdot g'(x)$$

$\square$

### 8.2.2 链式法则的直观理解

链式法则可以这样理解：如果 $u$ 关于 $x$ 的变化率是 $g'(x)$，而 $y$ 关于 $u$ 的变化率是 $f'(u)$，那么 $y$ 关于 $x$ 的变化率就是这两个变化率的乘积。

> **例题 8.4** 求 $y = \sin(x^2)$ 的导数。

**解**：令 $u = x^2$，则 $y = \sin u$。

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = \cos u \cdot 2x = 2x \cos(x^2)$$

> **例题 8.5** 求 $y = e^{\sin x}$ 的导数。

**解**：令 $u = \sin x$，则 $y = e^u$。

$$y' = e^u \cdot (\sin x)' = e^{\sin x} \cdot \cos x$$

### 8.2.3 多重复合

对于多重复合函数，链式法则可以逐层应用。

设 $y = f(u)$，$u = g(v)$，$v = h(x)$，则

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dv} \cdot \frac{dv}{dx} = f'(u) \cdot g'(v) \cdot h'(x)$$

> **例题 8.6** 求 $y = \ln(\cos(e^x))$ 的导数。

**解**：设 $u = \cos(e^x)$，$v = e^x$，则

$$y' = \frac{1}{u} \cdot (-\sin v) \cdot e^x = \frac{-\sin(e^x) \cdot e^x}{\cos(e^x)} = -e^x \tan(e^x)$$

---

## 8.3 反函数求导法则

### 8.3.1 反函数求导定理

**定理 8.1**（反函数求导法则）：设函数 $y = f(x)$ 在区间 $I$ 上严格单调且可导，且 $f'(x) \neq 0$，则其反函数 $x = f^{-1}(y)$ 在对应区间上也可导，且

$$[f^{-1}]'(y) = \frac{1}{f'(x)}$$

或等价地写成

$$\frac{dx}{dy} = \frac{1}{\dfrac{dy}{dx}}$$

**证明**：由于 $f(x)$ 严格单调且连续（可导蕴含连续），由反函数存在性定理，反函数 $x = f^{-1}(y)$ 存在且连续。

设 $y$ 有增量 $\Delta y \neq 0$，对应 $x$ 有增量 $\Delta x = f^{-1}(y + \Delta y) - f^{-1}(y)$。由 $f$ 的严格单调性知 $\Delta x \neq 0$，且当 $\Delta y \to 0$ 时，由 $f^{-1}$ 的连续性知 $\Delta x \to 0$。

于是

$$[f^{-1}]'(y) = \lim_{\Delta y \to 0} \frac{\Delta x}{\Delta y} = \lim_{\Delta x \to 0} \frac{1}{\dfrac{\Delta y}{\Delta x}} = \frac{1}{\lim_{\Delta x \to 0} \dfrac{\Delta y}{\Delta x}} = \frac{1}{f'(x)}$$

$\square$

**直观理解**：若 $y$ 关于 $x$ 的变化率为 $f'(x)$，那么 $x$ 关于 $y$ 的变化率自然是其倒数 $\dfrac{1}{f'(x)}$。条件 $f'(x) \neq 0$ 保证了倒数有意义。

### 8.3.2 应用：反三角函数的导数

利用反函数求导法则，可以系统地推导反三角函数的导数。

**（1）$(\arcsin x)' = \dfrac{1}{\sqrt{1-x^2}}$（$|x| < 1$）**

设 $y = \arcsin x$，则 $x = \sin y$，其中 $y \in \left(-\dfrac{\pi}{2}, \dfrac{\pi}{2}\right)$。

由反函数求导法则：

$$(\arcsin x)' = \frac{1}{(\sin y)'} = \frac{1}{\cos y}$$

在 $y \in \left(-\dfrac{\pi}{2}, \dfrac{\pi}{2}\right)$ 上，$\cos y > 0$，故

$$\cos y = \sqrt{1 - \sin^2 y} = \sqrt{1 - x^2}$$

因此

$$(\arcsin x)' = \frac{1}{\sqrt{1 - x^2}}$$

**（2）$(\arccos x)' = -\dfrac{1}{\sqrt{1-x^2}}$（$|x| < 1$）**

设 $y = \arccos x$，则 $x = \cos y$，其中 $y \in (0, \pi)$。

$$(\arccos x)' = \frac{1}{(\cos y)'} = \frac{1}{-\sin y}$$

在 $y \in (0, \pi)$ 上，$\sin y > 0$，故

$$\sin y = \sqrt{1 - \cos^2 y} = \sqrt{1 - x^2}$$

因此

$$(\arccos x)' = -\frac{1}{\sqrt{1 - x^2}}$$

**注**：$(\arcsin x)' + (\arccos x)' = 0$，这与恒等式 $\arcsin x + \arccos x = \dfrac{\pi}{2}$ 一致。

**（3）$(\arctan x)' = \dfrac{1}{1+x^2}$**

设 $y = \arctan x$，则 $x = \tan y$，其中 $y \in \left(-\dfrac{\pi}{2}, \dfrac{\pi}{2}\right)$。

$$(\arctan x)' = \frac{1}{(\tan y)'} = \frac{1}{\sec^2 y} = \frac{1}{1 + \tan^2 y} = \frac{1}{1 + x^2}$$

> **例题 8.7** 求 $y = \arcsin(2x - 1)$ 的导数。

**解**：利用链式法则和 $\arcsin$ 的导数公式：

$$y' = \frac{1}{\sqrt{1 - (2x-1)^2}} \cdot (2x - 1)' = \frac{2}{\sqrt{1 - (2x - 1)^2}}$$

化简根号内部：$1 - (2x-1)^2 = 1 - 4x^2 + 4x - 1 = 4x - 4x^2 = 4x(1-x)$，故

$$y' = \frac{2}{\sqrt{4x(1-x)}} = \frac{2}{2\sqrt{x(1-x)}} = \frac{1}{\sqrt{x(1-x)}} \quad (0 < x < 1)$$

> **例题 8.8** 求 $y = \arctan\dfrac{1}{x}$（$x \neq 0$）的导数。

**解**：

$$y' = \frac{1}{1 + \left(\dfrac{1}{x}\right)^2} \cdot \left(-\frac{1}{x^2}\right) = \frac{1}{\dfrac{x^2 + 1}{x^2}} \cdot \left(-\frac{1}{x^2}\right) = \frac{x^2}{x^2 + 1} \cdot \left(-\frac{1}{x^2}\right) = -\frac{1}{x^2 + 1}$$

**注**：当 $x > 0$ 时，$\arctan x + \arctan\dfrac{1}{x} = \dfrac{\pi}{2}$，两边求导即得上述结果。

---

## 8.4 隐函数求导

### 8.4.1 隐函数的概念

**显函数**：$y = f(x)$ 的形式，$y$ 明确地表示为 $x$ 的函数。

**隐函数**：由方程 $F(x, y) = 0$ 确定的函数关系，$y$ 没有明确表示为 $x$ 的函数。

例如，方程 $x^2 + y^2 = 1$ 确定了隐函数 $y = y(x)$（在上半圆为 $y = \sqrt{1-x^2}$，在下半圆为 $y = -\sqrt{1-x^2}$）。

### 8.4.2 隐函数求导法

**方法**：将方程 $F(x, y) = 0$ 两边对 $x$ 求导，把 $y$ 看作 $x$ 的函数，利用链式法则，然后解出 $\dfrac{dy}{dx}$。

> **例题 8.9** 设 $x^2 + y^2 = 1$，求 $\dfrac{dy}{dx}$。

**解**：方程两边对 $x$ 求导：

$$2x + 2y \cdot \frac{dy}{dx} = 0$$

解得：

$$\frac{dy}{dx} = -\frac{x}{y} \quad (y \neq 0)$$

> **例题 8.10** 设 $e^y + xy - e = 0$，求 $y'(0)$。

**解**：首先，将 $x = 0$ 代入方程：$e^y - e = 0$，得 $y = 1$。

方程两边对 $x$ 求导：

$$e^y \cdot y' + y + x \cdot y' = 0$$

将 $x = 0$，$y = 1$ 代入：

$$e \cdot y'(0) + 1 + 0 = 0$$

解得：$y'(0) = -\dfrac{1}{e}$

### 8.4.3 对数求导法

对于形如 $y = f(x)^{g(x)}$ 或多个因式乘除的函数，可以先取对数再求导。

**步骤**：
1. 两边取对数：$\ln y = g(x) \ln f(x)$
2. 两边对 $x$ 求导：$\dfrac{y'}{y} = \ldots$
3. 解出 $y'$

> **例题 8.11** 求 $y = x^x$（$x > 0$）的导数。

**解**：两边取对数：

$$\ln y = x \ln x$$

两边对 $x$ 求导：

$$\frac{y'}{y} = \ln x + x \cdot \frac{1}{x} = \ln x + 1$$

因此：

$$y' = y(\ln x + 1) = x^x(\ln x + 1)$$

> **例题 8.12** 求 $y = \dfrac{x \sqrt{1-x^2}}{(1+x^2)^2}$（$|x| < 1$）的导数。

**解**：取对数：

$$\ln|y| = \ln|x| + \frac{1}{2}\ln(1-x^2) - 2\ln(1+x^2)$$

两边求导：

$$\frac{y'}{y} = \frac{1}{x} + \frac{-2x}{2(1-x^2)} - \frac{4x}{1+x^2} = \frac{1}{x} - \frac{x}{1-x^2} - \frac{4x}{1+x^2}$$

通分化简：

$$\frac{y'}{y} = \frac{(1-x^2)(1+x^2) - x^2(1+x^2) - 4x^2(1-x^2)}{x(1-x^2)(1+x^2)} = \frac{1 - 6x^2 + x^4}{x(1-x^4)}$$

因此 $y' = y \cdot \dfrac{1 - 6x^2 + x^4}{x(1-x^4)}$。

---

## 8.5 参数方程求导

### 8.5.1 参数方程的导数

设曲线由参数方程给出：

$$\begin{cases} x = \varphi(t) \\ y = \psi(t) \end{cases}$$

若 $\varphi(t)$ 和 $\psi(t)$ 可导，且 $\varphi'(t) \neq 0$，则

$$\frac{dy}{dx} = \frac{dy/dt}{dx/dt} = \frac{\psi'(t)}{\varphi'(t)}$$

> **例题 8.13** 椭圆的参数方程为 $x = a\cos t$，$y = b\sin t$，求 $\dfrac{dy}{dx}$。

**解**：

$$\frac{dy}{dx} = \frac{(b\sin t)'}{(a\cos t)'} = \frac{b\cos t}{-a\sin t} = -\frac{b}{a}\cot t$$

> **例题 8.14** 摆线的参数方程为 $x = a(t - \sin t)$，$y = a(1 - \cos t)$，求 $\dfrac{dy}{dx}$。

**解**：

$$\frac{dx}{dt} = a(1 - \cos t), \quad \frac{dy}{dt} = a\sin t$$

$$\frac{dy}{dx} = \frac{a\sin t}{a(1 - \cos t)} = \frac{\sin t}{1 - \cos t} = \frac{2\sin\frac{t}{2}\cos\frac{t}{2}}{2\sin^2\frac{t}{2}} = \cot\frac{t}{2}$$

### 8.5.2 参数方程的二阶导数

$$\frac{d^2y}{dx^2} = \frac{d}{dx}\left(\frac{dy}{dx}\right) = \frac{\frac{d}{dt}\left(\frac{dy}{dx}\right)}{\frac{dx}{dt}}$$

设 $\dfrac{dy}{dx} = \dfrac{\psi'(t)}{\varphi'(t)}$，则

$$\frac{d^2y}{dx^2} = \frac{\psi''(t)\varphi'(t) - \psi'(t)\varphi''(t)}{[\varphi'(t)]^3}$$

> **例题 8.15** 对于摆线 $x = a(t - \sin t)$，$y = a(1 - \cos t)$，求 $\dfrac{d^2y}{dx^2}$。

**解**：由例题 8.14，$\dfrac{dy}{dx} = \cot\dfrac{t}{2}$。

$$\frac{d}{dt}\left(\frac{dy}{dx}\right) = -\frac{1}{2}\csc^2\frac{t}{2}$$

$$\frac{d^2y}{dx^2} = \frac{-\frac{1}{2}\csc^2\frac{t}{2}}{a(1 - \cos t)} = \frac{-\frac{1}{2}\csc^2\frac{t}{2}}{2a\sin^2\frac{t}{2}} = -\frac{1}{4a\sin^4\frac{t}{2}}$$

---

## 8.6 高阶导数

### 8.6.1 高阶导数的定义

**定义**：若 $f'(x)$ 可导，则称 $(f'(x))'$ 为 $f(x)$ 的**二阶导数**，记为

$$f''(x), \quad y'', \quad \frac{d^2y}{dx^2}, \quad \frac{d^2f}{dx^2}$$

一般地，$f(x)$ 的 $n$ 阶导数定义为 $(n-1)$ 阶导数的导数：

$$f^{(n)}(x) = \left(f^{(n-1)}(x)\right)'$$

记号：$f^{(n)}(x)$，$y^{(n)}$，$\dfrac{d^ny}{dx^n}$

> **例题 8.16** 求 $y = e^x$ 的 $n$ 阶导数。

**解**：$y' = e^x$，$y'' = e^x$，...，归纳得

$$(e^x)^{(n)} = e^x$$

> **例题 8.17** 求 $y = \sin x$ 的 $n$ 阶导数。

**解**：
- $y' = \cos x = \sin(x + \frac{\pi}{2})$
- $y'' = -\sin x = \sin(x + \pi) = \sin(x + \frac{2\pi}{2})$
- $y''' = -\cos x = \sin(x + \frac{3\pi}{2})$
- $y^{(4)} = \sin x = \sin(x + 2\pi) = \sin(x + \frac{4\pi}{2})$

归纳得：

$$(\sin x)^{(n)} = \sin\left(x + \frac{n\pi}{2}\right)$$

类似地：

$$(\cos x)^{(n)} = \cos\left(x + \frac{n\pi}{2}\right)$$

### 8.6.2 莱布尼茨公式

**定理**（Leibniz 公式）：若 $f(x)$ 和 $g(x)$ 都有 $n$ 阶导数，则

$$(fg)^{(n)} = \sum_{k=0}^{n} \binom{n}{k} f^{(k)}(x) g^{(n-k)}(x)$$

其中 $\binom{n}{k} = \dfrac{n!}{k!(n-k)!}$ 是二项式系数，$f^{(0)} = f$。

展开形式：

$$(fg)^{(n)} = f^{(n)}g + nf^{(n-1)}g' + \frac{n(n-1)}{2!}f^{(n-2)}g'' + \cdots + fg^{(n)}$$

> **例题 8.18** 求 $y = x^2 e^x$ 的 $n$ 阶导数（$n \geq 2$）。

**解**：设 $f(x) = e^x$，$g(x) = x^2$。

$g' = 2x$，$g'' = 2$，$g^{(k)} = 0$（$k \geq 3$）

由莱布尼茨公式：

$$y^{(n)} = e^x \cdot x^2 + n \cdot e^x \cdot 2x + \frac{n(n-1)}{2} \cdot e^x \cdot 2 = e^x(x^2 + 2nx + n^2 - n)$$

### 8.6.3 常见函数的 $n$ 阶导数

| 函数 | $n$ 阶导数 |
|:---:|:---:|
| $e^{ax}$ | $a^n e^{ax}$ |
| $a^x$ | $a^x (\ln a)^n$ |
| $\sin(ax+b)$ | $a^n \sin(ax + b + \frac{n\pi}{2})$ |
| $\cos(ax+b)$ | $a^n \cos(ax + b + \frac{n\pi}{2})$ |
| $\ln(ax+b)$ | $\dfrac{(-1)^{n-1}(n-1)! \cdot a^n}{(ax+b)^n}$ |
| $(ax+b)^\alpha$ | $\alpha(\alpha-1)\cdots(\alpha-n+1) \cdot a^n \cdot (ax+b)^{\alpha-n}$ |
| $\dfrac{1}{ax+b}$ | $\dfrac{(-1)^n n! \cdot a^n}{(ax+b)^{n+1}}$ |

---

## 本章小结

1. **四则运算法则**：
   - 和差法则：$(f \pm g)' = f' \pm g'$
   - 积的法则：$(fg)' = f'g + fg'$
   - 商的法则：$\left(\dfrac{f}{g}\right)' = \dfrac{f'g - fg'}{g^2}$

2. **链式法则**：$(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$，即 $\dfrac{dy}{dx} = \dfrac{dy}{du} \cdot \dfrac{du}{dx}$

3. **反函数求导法则**：$[f^{-1}]'(y) = \dfrac{1}{f'(x)}$，由此推导反三角函数的导数

4. **隐函数求导**：方程两边对 $x$ 求导，$y$ 视为 $x$ 的函数，解出 $y'$

5. **对数求导法**：先取对数，再求导，适用于幂指函数和复杂乘除式

6. **参数方程求导**：$\dfrac{dy}{dx} = \dfrac{dy/dt}{dx/dt}$，二阶导数需要再除以 $\dfrac{dx}{dt}$

7. **高阶导数**：逐次求导，常用莱布尼茨公式处理乘积的高阶导数

---

## 深度学习应用

### 8.7.1 链式法则与反向传播

深度学习中最核心的训练算法——反向传播（Backpropagation）——其数学本质正是多重复合函数的链式法则。

对于一个深度神经网络，损失函数 $L$ 是关于各层输出的复合函数。设

$$L = f(g(h(x)))$$

则损失函数关于输入 $x$ 的梯度为：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

反向传播算法从输出层开始，逐层向后计算每个参数对损失的偏导数，这正是链式法则的逐层应用。每一层只需知道来自上一层的梯度，再乘以本层的局部导数，即可得到本层的梯度。

**反向传播的计算流程**：

1. **前向传播**：依次计算 $h = h(x)$，$g = g(h)$，$f = f(g)$，得到 $L$
2. **反向传播**：依次计算 $\dfrac{\partial L}{\partial f}$，$\dfrac{\partial L}{\partial g} = \dfrac{\partial L}{\partial f} \cdot \dfrac{\partial f}{\partial g}$，$\dfrac{\partial L}{\partial h} = \dfrac{\partial L}{\partial g} \cdot \dfrac{\partial g}{\partial h}$，$\dfrac{\partial L}{\partial x} = \dfrac{\partial L}{\partial h} \cdot \dfrac{\partial h}{\partial x}$

### 8.7.2 自动微分（AutoDiff）

手动推导梯度公式既繁琐又容易出错，自动微分技术通过程序化地追踪计算过程来自动求导。

**前向模式 vs 反向模式**：

- **前向模式**（Forward Mode）：与函数求值同步，逐步计算 $\dfrac{\partial \text{输出}}{\partial \text{某个输入}}$。适合输入维度远小于输出维度的情况。
- **反向模式**（Reverse Mode）：先完成前向计算，再从输出反向追踪，一次性计算损失关于所有参数的梯度。深度学习中几乎都使用反向模式，因为参数数量（百万级以上）远大于损失的维度（通常为标量）。

**PyTorch 的 autograd 机制**：

PyTorch 通过构建**计算图**（Computational Graph）来实现反向模式自动微分。每次进行张量运算时，PyTorch 记录该运算及其输入，形成一张有向无环图（DAG）。调用 `.backward()` 时，沿计算图反向遍历，依链式法则累积梯度。

关键概念：
- `requires_grad=True`：标记需要追踪梯度的张量
- `.backward()`：触发反向传播，计算所有叶节点的梯度
- `.grad`：存储累积梯度
- `create_graph=True`：保留计算图以支持高阶导数

### 8.7.3 高阶导数在深度学习中的应用

**Hessian 矩阵与二阶优化**：

函数 $f(\mathbf{x})$ 的 Hessian 矩阵 $\mathbf{H}$ 由所有二阶偏导数构成：

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

二阶优化方法（如牛顿法）利用 Hessian 矩阵描述损失曲面的**曲率**，从而自适应调整步长：

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \mathbf{H}^{-1} \nabla f(\mathbf{x}_t)$$

在曲率大的方向（梯度变化快）步长小，在曲率小的方向步长大，比纯梯度下降更高效。

**Fisher 信息矩阵**：

Fisher 信息矩阵 $\mathbf{F}$ 是对数似然函数的期望 Hessian 矩阵的负值，衡量模型参数对分布的敏感程度：

$$\mathbf{F} = \mathbb{E}\left[\nabla \log p(x|\theta) \cdot \nabla \log p(x|\theta)^\top\right]$$

自然梯度法（Natural Gradient）使用 Fisher 信息矩阵代替 Hessian，在参数空间的黎曼几何意义下进行最优化，在强化学习（如 TRPO、PPO）和变分推断中有重要应用。

### 8.7.4 代码示例：自动微分演示

```python
import torch

# 自动微分：链式法则的自动应用
x = torch.tensor([2.0], requires_grad=True)

# 复合函数 f(g(h(x))) = sin(exp(x^2))
h = x ** 2        # h(x) = x^2
g = torch.exp(h)  # g(h) = e^h
f = torch.sin(g)  # f(g) = sin(g)

# 反向传播：自动应用链式法则
f.backward()

# 手动验证：df/dx = cos(e^(x^2)) * e^(x^2) * 2x
manual_grad = torch.cos(torch.exp(x**2)) * torch.exp(x**2) * 2 * x
print(f"自动微分: {x.grad.item():.6f}")
print(f"手动计算: {manual_grad.item():.6f}")

# 计算二阶导数（Hessian）
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
grad2 = torch.autograd.grad(grad1, x)[0]
print(f"二阶导数 d²(x³)/dx² = 6x = {grad2.item():.1f}")
```

> **注**：上述代码在 $x = 2$ 处计算 $\sin(e^{x^2})$ 的导数。由链式法则，结果为 $\cos(e^4) \cdot e^4 \cdot 4$，自动微分与手动计算应完全一致。对于 $y = x^3$，二阶导数 $\dfrac{d^2y}{dx^2} = 6x$，在 $x=2$ 处值为 $12$。

---

## 练习题

**1.** 求下列函数的导数：
   (a) $y = x^3 \ln x$
   (b) $y = \dfrac{e^x}{1 + x^2}$
   (c) $y = \sin^3(2x)$

**2.** 设 $x^3 + y^3 = 3xy$，求 $\dfrac{dy}{dx}$。

**3.** 求 $y = (\sin x)^x$（$0 < x < \pi$）的导数。

**4.** 设 $x = \ln(1 + t^2)$，$y = t - \arctan t$，求 $\dfrac{dy}{dx}$ 和 $\dfrac{d^2y}{dx^2}$。

**5.** 求 $y = x^2 \sin x$ 的 $n$ 阶导数（$n \geq 2$）。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.**

(a) $y = x^3 \ln x$

$$y' = 3x^2 \ln x + x^3 \cdot \frac{1}{x} = 3x^2 \ln x + x^2 = x^2(3\ln x + 1)$$

(b) $y = \dfrac{e^x}{1 + x^2}$

$$y' = \frac{e^x(1 + x^2) - e^x \cdot 2x}{(1 + x^2)^2} = \frac{e^x(1 + x^2 - 2x)}{(1 + x^2)^2} = \frac{e^x(1 - x)^2}{(1 + x^2)^2}$$

(c) $y = \sin^3(2x)$

设 $u = \sin(2x)$，则 $y = u^3$。

$$y' = 3u^2 \cdot u' = 3\sin^2(2x) \cdot \cos(2x) \cdot 2 = 6\sin^2(2x)\cos(2x)$$

或用二倍角公式：$y' = 3\sin(4x)\sin(2x)$

---

**2.** 方程 $x^3 + y^3 = 3xy$ 两边对 $x$ 求导：

$$3x^2 + 3y^2 \cdot y' = 3y + 3x \cdot y'$$

整理：

$$3y^2 y' - 3xy' = 3y - 3x^2$$

$$y'(y^2 - x) = y - x^2$$

$$\frac{dy}{dx} = \frac{y - x^2}{y^2 - x} \quad (y^2 \neq x)$$

---

**3.** $y = (\sin x)^x$

取对数：$\ln y = x \ln(\sin x)$

两边求导：

$$\frac{y'}{y} = \ln(\sin x) + x \cdot \frac{\cos x}{\sin x} = \ln(\sin x) + x\cot x$$

因此：

$$y' = (\sin x)^x \left[\ln(\sin x) + x\cot x\right]$$

---

**4.** $x = \ln(1 + t^2)$，$y = t - \arctan t$

$$\frac{dx}{dt} = \frac{2t}{1 + t^2}, \quad \frac{dy}{dt} = 1 - \frac{1}{1 + t^2} = \frac{t^2}{1 + t^2}$$

$$\frac{dy}{dx} = \frac{dy/dt}{dx/dt} = \frac{\frac{t^2}{1 + t^2}}{\frac{2t}{1 + t^2}} = \frac{t^2}{2t} = \frac{t}{2} \quad (t \neq 0)$$

对 $\dfrac{dy}{dx} = \dfrac{t}{2}$ 关于 $t$ 求导，再除以 $\dfrac{dx}{dt}$：

$$\frac{d^2y}{dx^2} = \frac{\frac{d}{dt}\left(\frac{t}{2}\right)}{\frac{dx}{dt}} = \frac{\frac{1}{2}}{\frac{2t}{1+t^2}} = \frac{1+t^2}{4t}$$

---

**5.** $y = x^2 \sin x$

设 $f(x) = \sin x$，$g(x) = x^2$。

$f^{(k)}(x) = \sin\left(x + \dfrac{k\pi}{2}\right)$

$g(x) = x^2$，$g'(x) = 2x$，$g''(x) = 2$，$g^{(k)} = 0$（$k \geq 3$）

由莱布尼茨公式（$n \geq 2$）：

$$y^{(n)} = \sin\left(x + \frac{n\pi}{2}\right) \cdot x^2 + n \sin\left(x + \frac{(n-1)\pi}{2}\right) \cdot 2x + \frac{n(n-1)}{2} \sin\left(x + \frac{(n-2)\pi}{2}\right) \cdot 2$$

化简：

$$y^{(n)} = x^2 \sin\left(x + \frac{n\pi}{2}\right) + 2nx \sin\left(x + \frac{(n-1)\pi}{2}\right) + n(n-1) \sin\left(x + \frac{(n-2)\pi}{2}\right)$$

利用 $\sin\left(x + \dfrac{(n-1)\pi}{2}\right) = \cos\left(x + \dfrac{(n-2)\pi}{2}\right)$ 和 $\sin\left(x + \dfrac{(n-2)\pi}{2}\right) = -\cos\left(x + \dfrac{(n-1)\pi}{2}\right)$，可进一步化简为：

$$y^{(n)} = (x^2 - n^2 + n) \sin\left(x + \frac{n\pi}{2}\right) + 2nx \cos\left(x + \frac{n\pi}{2}\right)$$

</details>
