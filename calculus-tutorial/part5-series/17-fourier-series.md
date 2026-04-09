# 第17章 Fourier级数

## 学习目标

通过本章学习，你将能够：

- 理解三角函数系的正交性，掌握正交性的积分表达式
- 掌握Fourier系数的计算公式及其推导过程
- 熟练计算周期为 $2\pi$ 和周期为 $2l$ 的函数的Fourier展开
- 理解Dirichlet收敛定理，掌握Fourier级数在连续点和间断点的收敛行为
- 熟练运用奇延拓和偶延拓将函数展开为正弦级数或余弦级数
- 能够运用Fourier级数求某些数项级数的和

---

## 17.1 三角级数与正交性

### 17.1.1 三角函数系

**三角函数系**是指由以下函数组成的函数集合：

$$1, \cos x, \sin x, \cos 2x, \sin 2x, \ldots, \cos nx, \sin nx, \ldots$$

这些函数都是以 $2\pi$ 为周期的周期函数。

**三角级数**：由三角函数系构成的级数

$$\frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos nx + b_n \sin nx)$$

称为**三角级数**，其中 $a_0, a_1, b_1, a_2, b_2, \ldots$ 为常数。

> **注**：常数项写成 $\dfrac{a_0}{2}$ 的形式是为了使后面的Fourier系数公式统一。

### 17.1.2 三角函数系的正交性

**正交性定义**：设 $f(x)$ 和 $g(x)$ 在区间 $[a, b]$ 上可积，若

$$\int_a^b f(x) g(x) \, dx = 0$$

则称 $f(x)$ 与 $g(x)$ 在 $[a, b]$ 上**正交**。

**定理（三角函数系的正交性）**：三角函数系 $\{1, \cos nx, \sin nx\}_{n=1}^{\infty}$ 在区间 $[-\pi, \pi]$ 上两两正交，即对任意非负整数 $m, n$：

$$\int_{-\pi}^{\pi} \cos mx \cos nx \, dx = \begin{cases} 0, & m \neq n \\ \pi, & m = n \neq 0 \\ 2\pi, & m = n = 0 \end{cases}$$

$$\int_{-\pi}^{\pi} \sin mx \sin nx \, dx = \begin{cases} 0, & m \neq n \\ \pi, & m = n \neq 0 \end{cases}$$

$$\int_{-\pi}^{\pi} \cos mx \sin nx \, dx = 0 \quad (\text{对所有 } m, n)$$

**证明**（选证部分）：利用积化和差公式。

当 $m \neq n$ 时：

$$\cos mx \cos nx = \frac{1}{2}[\cos(m-n)x + \cos(m+n)x]$$

$$\int_{-\pi}^{\pi} \cos mx \cos nx \, dx = \frac{1}{2}\left[\frac{\sin(m-n)x}{m-n} + \frac{\sin(m+n)x}{m+n}\right]_{-\pi}^{\pi} = 0$$

当 $m = n \neq 0$ 时：

$$\int_{-\pi}^{\pi} \cos^2 nx \, dx = \int_{-\pi}^{\pi} \frac{1 + \cos 2nx}{2} \, dx = \frac{1}{2}\left[x + \frac{\sin 2nx}{2n}\right]_{-\pi}^{\pi} = \pi$$

对于余弦与正弦的乘积，由于 $\cos mx \sin nx$ 是奇函数，在对称区间上积分为零。$\square$

### 17.1.3 周期函数的三角级数展开

设 $f(x)$ 是周期为 $2\pi$ 的周期函数，如果能将其展开为三角级数：

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos nx + b_n \sin nx)$$

则利用三角函数系的正交性，可以确定系数 $a_n$ 和 $b_n$。这就是Fourier级数的核心思想。

---

## 17.2 Fourier系数

### 17.2.1 Fourier系数公式

**定理**：设 $f(x)$ 是周期为 $2\pi$ 的可积函数，若能展开为三角级数

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos nx + b_n \sin nx)$$

则系数由以下公式确定：

$$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos nx \, dx \quad (n = 0, 1, 2, \ldots)$$

$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin nx \, dx \quad (n = 1, 2, 3, \ldots)$$

这些系数 $a_n, b_n$ 称为 $f(x)$ 的**Fourier系数**。

### 17.2.2 推导过程

**求 $a_0$**：将展开式两边在 $[-\pi, \pi]$ 上积分：

$$\int_{-\pi}^{\pi} f(x) \, dx = \frac{a_0}{2} \cdot 2\pi + \sum_{n=1}^{\infty} \left(a_n \int_{-\pi}^{\pi} \cos nx \, dx + b_n \int_{-\pi}^{\pi} \sin nx \, dx\right)$$

由于 $\int_{-\pi}^{\pi} \cos nx \, dx = \int_{-\pi}^{\pi} \sin nx \, dx = 0$（$n \geq 1$），得

$$\int_{-\pi}^{\pi} f(x) \, dx = \pi a_0 \quad \Rightarrow \quad a_0 = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \, dx$$

**求 $a_m$**（$m \geq 1$）：将展开式两边乘以 $\cos mx$，再在 $[-\pi, \pi]$ 上积分：

$$\int_{-\pi}^{\pi} f(x) \cos mx \, dx = \frac{a_0}{2} \int_{-\pi}^{\pi} \cos mx \, dx + \sum_{n=1}^{\infty} a_n \int_{-\pi}^{\pi} \cos nx \cos mx \, dx + \sum_{n=1}^{\infty} b_n \int_{-\pi}^{\pi} \sin nx \cos mx \, dx$$

由正交性，只有 $n = m$ 时的余弦项积分非零：

$$\int_{-\pi}^{\pi} f(x) \cos mx \, dx = a_m \cdot \pi \quad \Rightarrow \quad a_m = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos mx \, dx$$

**求 $b_m$**：类似地，将展开式两边乘以 $\sin mx$ 并积分，由正交性得

$$b_m = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin mx \, dx$$

### 17.2.3 周期为 $2\pi$ 的情况

给定周期为 $2\pi$ 的函数 $f(x)$，其**Fourier级数**定义为

$$f(x) \sim \frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos nx + b_n \sin nx)$$

其中 Fourier 系数由上述公式给出。符号 "$\sim$" 表示形式对应，是否等号成立需要讨论收敛性。

> **例题 17.1** 将函数 $f(x) = x$（$-\pi < x \leq \pi$），以 $2\pi$ 为周期延拓，求其Fourier级数。

**解**：$f(x) = x$ 是奇函数，故 $f(x) \cos nx$ 是奇函数，$f(x) \sin nx$ 是偶函数。

$$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} x \cos nx \, dx = 0 \quad (n = 0, 1, 2, \ldots)$$

$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} x \sin nx \, dx = \frac{2}{\pi} \int_0^{\pi} x \sin nx \, dx$$

利用分部积分：

$$\int_0^{\pi} x \sin nx \, dx = \left[-\frac{x \cos nx}{n}\right]_0^{\pi} + \frac{1}{n} \int_0^{\pi} \cos nx \, dx = -\frac{\pi \cos n\pi}{n} = \frac{(-1)^{n+1} \pi}{n}$$

因此

$$b_n = \frac{2}{\pi} \cdot \frac{(-1)^{n+1} \pi}{n} = \frac{2(-1)^{n+1}}{n}$$

Fourier级数为

$$f(x) \sim 2\left(\sin x - \frac{\sin 2x}{2} + \frac{\sin 3x}{3} - \cdots\right) = 2\sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n} \sin nx$$

> **例题 17.2** 将函数 $f(x) = |x|$（$-\pi \leq x \leq \pi$），以 $2\pi$ 为周期延拓，求其Fourier级数。

**解**：$f(x) = |x|$ 是偶函数，故 $b_n = 0$。

$$a_0 = \frac{1}{\pi} \int_{-\pi}^{\pi} |x| \, dx = \frac{2}{\pi} \int_0^{\pi} x \, dx = \frac{2}{\pi} \cdot \frac{\pi^2}{2} = \pi$$

$$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} |x| \cos nx \, dx = \frac{2}{\pi} \int_0^{\pi} x \cos nx \, dx \quad (n \geq 1)$$

分部积分：

$$\int_0^{\pi} x \cos nx \, dx = \left[\frac{x \sin nx}{n}\right]_0^{\pi} - \frac{1}{n} \int_0^{\pi} \sin nx \, dx = 0 + \frac{1}{n} \left[\frac{\cos nx}{n}\right]_0^{\pi} = \frac{\cos n\pi - 1}{n^2}$$

当 $n$ 为偶数时，$\cos n\pi = 1$，$a_n = 0$。

当 $n$ 为奇数时，$\cos n\pi = -1$，$a_n = \dfrac{2}{\pi} \cdot \dfrac{-2}{n^2} = -\dfrac{4}{\pi n^2}$。

因此

$$|x| \sim \frac{\pi}{2} - \frac{4}{\pi}\left(\cos x + \frac{\cos 3x}{9} + \frac{\cos 5x}{25} + \cdots\right) = \frac{\pi}{2} - \frac{4}{\pi} \sum_{k=0}^{\infty} \frac{\cos(2k+1)x}{(2k+1)^2}$$

### 17.2.4 周期为 $2l$ 的情况

设 $f(x)$ 是周期为 $2l$ 的函数。令 $t = \dfrac{\pi x}{l}$，则 $g(t) = f\left(\dfrac{lt}{\pi}\right)$ 是周期为 $2\pi$ 的函数。

将 $g(t)$ 展开为Fourier级数后，换回变量 $x$，得到周期为 $2l$ 的Fourier级数：

$$f(x) \sim \frac{a_0}{2} + \sum_{n=1}^{\infty} \left(a_n \cos \frac{n\pi x}{l} + b_n \sin \frac{n\pi x}{l}\right)$$

其中Fourier系数为

$$a_n = \frac{1}{l} \int_{-l}^{l} f(x) \cos \frac{n\pi x}{l} \, dx \quad (n = 0, 1, 2, \ldots)$$

$$b_n = \frac{1}{l} \int_{-l}^{l} f(x) \sin \frac{n\pi x}{l} \, dx \quad (n = 1, 2, 3, \ldots)$$

> **例题 17.3** 将函数 $f(x) = x$（$-1 < x \leq 1$），以 $2$ 为周期延拓，求其Fourier级数。

**解**：这里 $l = 1$。由于 $f(x) = x$ 是奇函数，$a_n = 0$。

$$b_n = \frac{1}{1} \int_{-1}^{1} x \sin n\pi x \, dx = 2 \int_0^{1} x \sin n\pi x \, dx$$

分部积分：

$$\int_0^{1} x \sin n\pi x \, dx = \left[-\frac{x \cos n\pi x}{n\pi}\right]_0^{1} + \frac{1}{n\pi} \int_0^{1} \cos n\pi x \, dx = -\frac{\cos n\pi}{n\pi} = \frac{(-1)^{n+1}}{n\pi}$$

因此 $b_n = \dfrac{2(-1)^{n+1}}{n\pi}$，

$$x \sim \frac{2}{\pi} \sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n} \sin n\pi x \quad (-1 < x < 1)$$

---

## 17.3 Fourier级数的收敛性

### 17.3.1 Dirichlet收敛定理

**定理（Dirichlet收敛定理）**：设 $f(x)$ 是周期为 $2\pi$ 的函数，若 $f(x)$ 在 $[-\pi, \pi]$ 上满足**Dirichlet条件**：

1. $f(x)$ 在 $[-\pi, \pi]$ 上连续或只有有限个第一类间断点
2. $f(x)$ 在 $[-\pi, \pi]$ 上只有有限个极值点（即分段单调）

则 $f(x)$ 的Fourier级数在每一点都收敛，且

$$\frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos nx + b_n \sin nx) = \frac{f(x^-) + f(x^+)}{2}$$

其中 $f(x^-)$ 和 $f(x^+)$ 分别表示 $f$ 在 $x$ 处的左极限和右极限。

### 17.3.2 收敛情况分析

**在连续点**：若 $f(x)$ 在 $x_0$ 处连续，则 $f(x_0^-) = f(x_0^+) = f(x_0)$，故

$$\frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos nx_0 + b_n \sin nx_0) = f(x_0)$$

**在间断点**：若 $f(x)$ 在 $x_0$ 处有第一类间断点，则Fourier级数收敛于左右极限的平均值：

$$\frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos nx_0 + b_n \sin nx_0) = \frac{f(x_0^-) + f(x_0^+)}{2}$$

### 17.3.3 在间断点的行为

> **例题 17.4** 讨论例题17.1中 $f(x) = x$（$-\pi < x \leq \pi$）的Fourier级数在各点的收敛情况。

**解**：由Dirichlet定理，在 $(-\pi, \pi)$ 内的每一点，级数收敛于 $f(x) = x$。

在 $x = \pi$ 处，$f(\pi^-) = \pi$，$f(\pi^+) = f(-\pi^+) = -\pi$（由周期性），故级数收敛于

$$\frac{\pi + (-\pi)}{2} = 0$$

因此

$$2\sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n} \sin nx = \begin{cases} x, & -\pi < x < \pi \\ 0, & x = \pm\pi \end{cases}$$

> **例题 17.5** 利用例题17.2的结果，求级数 $\sum_{n=0}^{\infty} \dfrac{1}{(2n+1)^2}$ 的和。

**解**：由例题17.2，

$$|x| = \frac{\pi}{2} - \frac{4}{\pi} \sum_{k=0}^{\infty} \frac{\cos(2k+1)x}{(2k+1)^2} \quad (-\pi \leq x \leq \pi)$$

令 $x = 0$，得

$$0 = \frac{\pi}{2} - \frac{4}{\pi} \sum_{k=0}^{\infty} \frac{1}{(2k+1)^2}$$

因此

$$\sum_{k=0}^{\infty} \frac{1}{(2k+1)^2} = 1 + \frac{1}{9} + \frac{1}{25} + \frac{1}{49} + \cdots = \frac{\pi^2}{8}$$

---

## 17.4 正弦级数与余弦级数

### 17.4.1 奇函数与偶函数的Fourier展开

**偶函数**：若 $f(x)$ 是周期为 $2\pi$ 的偶函数，则

$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin nx \, dx = 0$$

$$a_n = \frac{2}{\pi} \int_0^{\pi} f(x) \cos nx \, dx$$

Fourier级数只含余弦项，称为**余弦级数**：

$$f(x) \sim \frac{a_0}{2} + \sum_{n=1}^{\infty} a_n \cos nx$$

**奇函数**：若 $f(x)$ 是周期为 $2\pi$ 的奇函数，则

$$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos nx \, dx = 0$$

$$b_n = \frac{2}{\pi} \int_0^{\pi} f(x) \sin nx \, dx$$

Fourier级数只含正弦项，称为**正弦级数**：

$$f(x) \sim \sum_{n=1}^{\infty} b_n \sin nx$$

### 17.4.2 奇延拓与正弦级数

设 $f(x)$ 只在 $[0, l]$ 上有定义，要将其展开为正弦级数，可进行**奇延拓**：

$$F(x) = \begin{cases} f(x), & 0 < x \leq l \\ 0, & x = 0 \\ -f(-x), & -l \leq x < 0 \end{cases}$$

然后将 $F(x)$ 以 $2l$ 为周期延拓，得到奇函数，其Fourier级数只含正弦项：

$$f(x) \sim \sum_{n=1}^{\infty} b_n \sin \frac{n\pi x}{l} \quad (0 < x < l)$$

其中

$$b_n = \frac{2}{l} \int_0^{l} f(x) \sin \frac{n\pi x}{l} \, dx$$

### 17.4.3 偶延拓与余弦级数

设 $f(x)$ 只在 $[0, l]$ 上有定义，要将其展开为余弦级数，可进行**偶延拓**：

$$F(x) = \begin{cases} f(x), & 0 \leq x \leq l \\ f(-x), & -l \leq x < 0 \end{cases}$$

然后将 $F(x)$ 以 $2l$ 为周期延拓，得到偶函数，其Fourier级数只含余弦项：

$$f(x) \sim \frac{a_0}{2} + \sum_{n=1}^{\infty} a_n \cos \frac{n\pi x}{l} \quad (0 \leq x \leq l)$$

其中

$$a_n = \frac{2}{l} \int_0^{l} f(x) \cos \frac{n\pi x}{l} \, dx$$

> **例题 17.6** 将 $f(x) = x$（$0 < x < \pi$）分别展开为正弦级数和余弦级数。

**解**：

**正弦级数（奇延拓）**：

$$b_n = \frac{2}{\pi} \int_0^{\pi} x \sin nx \, dx = \frac{2}{\pi} \cdot \frac{(-1)^{n+1} \pi}{n} = \frac{2(-1)^{n+1}}{n}$$

$$x = 2\sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n} \sin nx \quad (0 < x < \pi)$$

**余弦级数（偶延拓）**：

$$a_0 = \frac{2}{\pi} \int_0^{\pi} x \, dx = \frac{2}{\pi} \cdot \frac{\pi^2}{2} = \pi$$

$$a_n = \frac{2}{\pi} \int_0^{\pi} x \cos nx \, dx = \frac{2}{\pi} \cdot \frac{\cos n\pi - 1}{n^2} = \frac{2[(-1)^n - 1]}{\pi n^2}$$

当 $n$ 为偶数时，$a_n = 0$；当 $n$ 为奇数时，$a_n = -\dfrac{4}{\pi n^2}$。

$$x = \frac{\pi}{2} - \frac{4}{\pi}\left(\cos x + \frac{\cos 3x}{9} + \frac{\cos 5x}{25} + \cdots\right) \quad (0 \leq x \leq \pi)$$

> **例题 17.7** 将 $f(x) = 1$（$0 < x < l$）展开为正弦级数。

**解**：进行奇延拓，

$$b_n = \frac{2}{l} \int_0^{l} 1 \cdot \sin \frac{n\pi x}{l} \, dx = \frac{2}{l} \left[-\frac{l}{n\pi} \cos \frac{n\pi x}{l}\right]_0^{l} = \frac{2}{n\pi}(1 - \cos n\pi) = \frac{2}{n\pi}[1 - (-1)^n]$$

当 $n$ 为偶数时，$b_n = 0$；当 $n$ 为奇数时，$b_n = \dfrac{4}{n\pi}$。

$$1 = \frac{4}{\pi}\left(\sin \frac{\pi x}{l} + \frac{1}{3}\sin \frac{3\pi x}{l} + \frac{1}{5}\sin \frac{5\pi x}{l} + \cdots\right) \quad (0 < x < l)$$

---

## 17.5 Fourier级数的应用

### 17.5.1 求和公式

利用Fourier级数在特定点的收敛值，可以求某些数项级数的和。

> **例题 17.8** 求级数 $\sum_{n=1}^{\infty} \dfrac{1}{n^2}$ 的和。

**解**：由例题17.2，对于 $f(x) = |x|$ 的Fourier展开，在 $x = \pi$ 处：

$$\pi = \frac{\pi}{2} - \frac{4}{\pi} \sum_{k=0}^{\infty} \frac{\cos(2k+1)\pi}{(2k+1)^2} = \frac{\pi}{2} + \frac{4}{\pi} \sum_{k=0}^{\infty} \frac{1}{(2k+1)^2}$$

由例题17.5，$\sum_{k=0}^{\infty} \dfrac{1}{(2k+1)^2} = \dfrac{\pi^2}{8}$。

注意到

$$\sum_{n=1}^{\infty} \frac{1}{n^2} = \sum_{k=0}^{\infty} \frac{1}{(2k+1)^2} + \sum_{k=1}^{\infty} \frac{1}{(2k)^2} = \sum_{k=0}^{\infty} \frac{1}{(2k+1)^2} + \frac{1}{4}\sum_{n=1}^{\infty} \frac{1}{n^2}$$

设 $S = \sum_{n=1}^{\infty} \dfrac{1}{n^2}$，则

$$S = \frac{\pi^2}{8} + \frac{S}{4} \quad \Rightarrow \quad \frac{3S}{4} = \frac{\pi^2}{8} \quad \Rightarrow \quad S = \frac{\pi^2}{6}$$

因此

$$\sum_{n=1}^{\infty} \frac{1}{n^2} = 1 + \frac{1}{4} + \frac{1}{9} + \frac{1}{16} + \cdots = \frac{\pi^2}{6}$$

这就是著名的**Basel问题**的解答。

> **例题 17.9** 利用Fourier级数证明 $\sum_{n=1}^{\infty} \dfrac{(-1)^{n+1}}{n} = \ln 2$。

**解**：考虑 $f(x) = x$（$-\pi < x < \pi$）的Fourier级数（例题17.1）：

$$x = 2\sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n} \sin nx$$

这不能直接代入求和。改用另一方法：由 $\ln(1+x)$ 的Taylor展开，在 $x = 1$ 处：

$$\ln 2 = 1 - \frac{1}{2} + \frac{1}{3} - \frac{1}{4} + \cdots = \sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n}$$

### 17.5.2 信号分析简介

Fourier级数在信号处理中有重要应用。任何周期信号都可以分解为不同频率的正弦波（谐波）的叠加。

**基波与谐波**：在Fourier级数

$$f(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos n\omega t + b_n \sin n\omega t)$$

中：
- $\dfrac{a_0}{2}$ 是**直流分量**
- $n = 1$ 的项称为**基波**（频率为 $\omega$）
- $n \geq 2$ 的项称为**谐波**（频率为 $n\omega$）

**振幅-相位形式**：每个谐波可写成

$$a_n \cos n\omega t + b_n \sin n\omega t = A_n \cos(n\omega t - \varphi_n)$$

其中**振幅** $A_n = \sqrt{a_n^2 + b_n^2}$，**相位** $\varphi_n = \arctan\dfrac{b_n}{a_n}$。

---

## 本章小结

1. **三角函数系的正交性**是Fourier分析的基础。三角函数系 $\{1, \cos nx, \sin nx\}$ 在 $[-\pi, \pi]$ 上两两正交。

2. **Fourier系数**的计算公式：
   - 周期 $2\pi$：$a_n = \dfrac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos nx \, dx$，$b_n = \dfrac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin nx \, dx$
   - 周期 $2l$：$a_n = \dfrac{1}{l} \int_{-l}^{l} f(x) \cos \dfrac{n\pi x}{l} \, dx$，$b_n = \dfrac{1}{l} \int_{-l}^{l} f(x) \sin \dfrac{n\pi x}{l} \, dx$

3. **Dirichlet收敛定理**：满足Dirichlet条件的函数，其Fourier级数在连续点收敛于函数值，在间断点收敛于左右极限的平均值。

4. **正弦级数与余弦级数**：
   - **奇延拓**得到正弦级数：$b_n = \dfrac{2}{l} \int_0^{l} f(x) \sin \dfrac{n\pi x}{l} \, dx$
   - **偶延拓**得到余弦级数：$a_n = \dfrac{2}{l} \int_0^{l} f(x) \cos \dfrac{n\pi x}{l} \, dx$

5. **应用**：Fourier级数可用于求数项级数的和（如 $\sum \dfrac{1}{n^2} = \dfrac{\pi^2}{6}$），以及信号的频谱分析。

---

## 深度学习应用

Fourier 分析不仅是经典数学工具，也是现代深度学习的理论基础之一。本节介绍其在神经网络中的四个核心应用场景。

### 17.6.1 频域分析与 CNN

**卷积定理**指出，时域的卷积运算等价于频域的逐点乘法：

$$\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}$$

其中 $\mathcal{F}$ 表示 Fourier 变换，$*$ 表示卷积。这个定理给出了计算卷积的高效途径：

1. 将信号变换到频域（FFT，复杂度 $O(n \log n)$）
2. 在频域做逐点乘法（$O(n)$）
3. 逆变换回时域（iFFT，复杂度 $O(n \log n)$）

相比直接在时域计算的 $O(n^2)$ 复杂度，频域方法对大核卷积有显著加速。

**CNN 的频域解释**：卷积神经网络的每个滤波器本质上是一个**频率选择器**。低频滤波器捕捉图像中的平滑结构和整体形状，高频滤波器检测边缘、纹理等细节。网络通过训练学习到不同频段的特征表示。

### 17.6.2 谱归一化（Spectral Normalization）

在生成对抗网络（GAN）训练中，判别器的 Lipschitz 常数决定了训练稳定性。**谱归一化**通过限制权重矩阵的**谱范数**（最大奇异值 $\sigma_1$）来控制 Lipschitz 常数：

$$\bar{W} = \frac{W}{\sigma_1(W)}$$

其中谱范数定义为

$$\|W\|_2 = \sigma_1(W) = \max_{\|x\|=1} \|Wx\|$$

归一化后的权重矩阵满足 $\|\bar{W}\|_2 = 1$，从而使判别器成为 1-Lipschitz 函数，有效防止梯度爆炸并稳定 GAN 训练。

实际计算中，精确的奇异值分解开销较大，通常用**幂迭代法**高效估计最大奇异值：

$$\tilde{v} \leftarrow \frac{W^\top \hat{u}}{\|W^\top \hat{u}\|}, \quad \tilde{u} \leftarrow \frac{W\tilde{v}}{\|W\tilde{v}\|}, \quad \sigma_1 \approx \hat{u}^\top W \tilde{v}$$

### 17.6.3 傅里叶特征编码

神经网络在拟合高频信号时存在"谱偏差"（spectral bias）——网络倾向于先学习低频分量。**傅里叶特征编码**通过显式引入高频基函数来克服这一问题。

**随机傅里叶特征**：将输入 $\mathbf{x} \in \mathbb{R}^d$ 映射为

$$\gamma(\mathbf{x}) = [\cos(2\pi \mathbf{b}_1^\top \mathbf{x}),\ \sin(2\pi \mathbf{b}_1^\top \mathbf{x}),\ \ldots,\ \cos(2\pi \mathbf{b}_m^\top \mathbf{x}),\ \sin(2\pi \mathbf{b}_m^\top \mathbf{x})]$$

其中频率向量 $\mathbf{b}_i$ 从某分布中采样，将输入提升为 $2m$ 维特征。

**NeRF 中的位置编码**：Neural Radiance Fields 使用确定性的多尺度编码：

$$\gamma(p) = [\sin(2^0 \pi p),\ \cos(2^0 \pi p),\ \sin(2^1 \pi p),\ \cos(2^1 \pi p),\ \ldots,\ \sin(2^{L-1} \pi p),\ \cos(2^{L-1} \pi p)]$$

频率以 $2$ 的幂次递增，覆盖从粗到细的多个尺度，使网络能够重建细节丰富的三维场景。

### 17.6.4 图神经网络的谱方法

对于图 $\mathcal{G} = (V, E)$，定义**图拉普拉斯矩阵** $L = D - A$，其中 $D$ 是度矩阵，$A$ 是邻接矩阵。$L$ 是半正定矩阵，可做特征分解 $L = U \Lambda U^\top$，其中特征向量矩阵 $U$ 构成图上的"Fourier 基"。

图上信号 $\mathbf{x}$ 的**图 Fourier 变换**定义为

$$\hat{\mathbf{x}} = U^\top \mathbf{x}$$

逆变换为 $\mathbf{x} = U\hat{\mathbf{x}}$，与经典 Fourier 变换完全类比。

**谱图卷积**（Spectral Graph Convolution）在频域定义图卷积：

$$\mathbf{x} *_{\mathcal{G}} g = U \left( (U^\top \mathbf{x}) \odot (U^\top \mathbf{g}) \right)$$

GCN（Chebyshev 近似版本）通过截断 Chebyshev 多项式展开，将谱方法转化为局部空域操作，避免了完整特征分解的 $O(n^3)$ 计算开销，成为现代图神经网络的理论基础。

### 17.6.5 代码示例

```python
import torch
import torch.nn as nn
import torch.fft as fft
import math

# 频域卷积演示
def freq_domain_conv(x, kernel):
    """时域卷积 = 频域乘法"""
    # 零填充到相同大小
    n = x.shape[-1] + kernel.shape[-1] - 1
    X = fft.fft(x, n=n)
    K = fft.fft(kernel, n=n)
    # 频域乘法
    Y = X * K
    # 逆变换
    return fft.ifft(Y).real

# 谱归一化
class SpectralNorm(nn.Module):
    def __init__(self, module, n_power_iterations=1):
        super().__init__()
        self.module = module
        self.n_power_iterations = n_power_iterations

    def forward(self, x):
        # 使用幂迭代估计最大奇异值
        w = self.module.weight
        u = torch.randn(w.shape[0], 1, device=w.device)

        for _ in range(self.n_power_iterations):
            v = w.t() @ u
            v = v / v.norm()
            u = w @ v
            u = u / u.norm()

        sigma = (u.t() @ w @ v).squeeze()
        return nn.functional.linear(x, w / sigma, self.module.bias)

# 傅里叶特征编码 (NeRF style)
class FourierFeatures(nn.Module):
    def __init__(self, input_dim, n_frequencies=10):
        super().__init__()
        # 频率 2^0, 2^1, ..., 2^(L-1)
        freqs = 2.0 ** torch.arange(n_frequencies)
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        # [sin(2πf·x), cos(2πf·x)] for each frequency
        x_proj = x.unsqueeze(-1) * self.freqs * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).flatten(-2)
```

**验证频域卷积的等价性**：

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
k = torch.tensor([1.0, -1.0])

# 时域卷积（使用 torch.nn.functional）
import torch.nn.functional as F
y_time = F.conv1d(x.view(1, 1, -1), k.view(1, 1, -1), padding=1).squeeze()

# 频域卷积
y_freq = freq_domain_conv(x, k)

print("时域结果:", y_time)
print("频域结果:", y_freq[:len(x)])  # 截取有效部分
```

---

## 练习题

**1.** 将函数 $f(x) = x^2$（$-\pi \leq x \leq \pi$），以 $2\pi$ 为周期延拓，求其Fourier级数。

**2.** 将函数 $f(x) = e^x$（$-\pi < x < \pi$），以 $2\pi$ 为周期延拓，求其Fourier级数。

**3.** 将 $f(x) = \pi - x$（$0 < x < \pi$）展开为正弦级数。

**4.** 利用第1题的结果，求 $\sum_{n=1}^{\infty} \dfrac{1}{n^4}$ 的值。

**5.** 设 $f(x) = \begin{cases} 0, & -\pi \leq x < 0 \\ 1, & 0 \leq x \leq \pi \end{cases}$，以 $2\pi$ 为周期延拓，求其Fourier级数，并求 $\sum_{n=0}^{\infty} \dfrac{(-1)^n}{2n+1}$ 的值。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.** $f(x) = x^2$ 是偶函数，故 $b_n = 0$。

$$a_0 = \frac{1}{\pi} \int_{-\pi}^{\pi} x^2 \, dx = \frac{2}{\pi} \int_0^{\pi} x^2 \, dx = \frac{2}{\pi} \cdot \frac{\pi^3}{3} = \frac{2\pi^2}{3}$$

$$a_n = \frac{2}{\pi} \int_0^{\pi} x^2 \cos nx \, dx \quad (n \geq 1)$$

分部积分两次：

$$\int_0^{\pi} x^2 \cos nx \, dx = \left[\frac{x^2 \sin nx}{n}\right]_0^{\pi} - \frac{2}{n} \int_0^{\pi} x \sin nx \, dx = -\frac{2}{n}\left[-\frac{x \cos nx}{n}\Big|_0^{\pi} + \frac{1}{n}\int_0^{\pi} \cos nx \, dx\right]$$

$$= -\frac{2}{n}\left[-\frac{\pi \cos n\pi}{n}\right] = \frac{2\pi \cos n\pi}{n^2} = \frac{2\pi (-1)^n}{n^2}$$

因此 $a_n = \dfrac{2}{\pi} \cdot \dfrac{2\pi (-1)^n}{n^2} = \dfrac{4(-1)^n}{n^2}$。

$$x^2 = \frac{\pi^2}{3} + 4\sum_{n=1}^{\infty} \frac{(-1)^n}{n^2} \cos nx = \frac{\pi^2}{3} - 4\cos x + \cos 2x - \frac{4\cos 3x}{9} + \cdots$$

---

**2.** $f(x) = e^x$ 既非奇函数也非偶函数。

$$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} e^x \cos nx \, dx, \quad b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} e^x \sin nx \, dx$$

利用公式 $\int e^x \cos nx \, dx = \dfrac{e^x (cos nx + n \sin nx)}{1 + n^2}$：

$$a_n = \frac{1}{\pi} \cdot \frac{e^x(\cos nx + n\sin nx)}{1+n^2}\Big|_{-\pi}^{\pi} = \frac{1}{\pi(1+n^2)}[(e^\pi - e^{-\pi})\cos n\pi] = \frac{2(-1)^n \sinh\pi}{\pi(1+n^2)}$$

类似地，$b_n = \dfrac{-2n(-1)^n \sinh\pi}{\pi(1+n^2)}$。

$$e^x = \frac{\sinh\pi}{\pi} + \frac{2\sinh\pi}{\pi}\sum_{n=1}^{\infty} \frac{(-1)^n}{1+n^2}(\cos nx - n\sin nx)$$

---

**3.** 奇延拓后：

$$b_n = \frac{2}{\pi} \int_0^{\pi} (\pi - x) \sin nx \, dx = \frac{2}{\pi}\left[\pi \cdot \frac{1-\cos n\pi}{n} - \frac{(-1)^{n+1}\pi}{n}\right] = \frac{2}{n}$$

$$\pi - x = 2\sum_{n=1}^{\infty} \frac{\sin nx}{n} \quad (0 < x < \pi)$$

---

**4.** 由第1题，在 $x = \pi$ 处：

$$\pi^2 = \frac{\pi^2}{3} + 4\sum_{n=1}^{\infty} \frac{(-1)^n \cos n\pi}{n^2} = \frac{\pi^2}{3} + 4\sum_{n=1}^{\infty} \frac{1}{n^2}$$

因此 $\sum_{n=1}^{\infty} \dfrac{1}{n^2} = \dfrac{\pi^2}{6}$。

利用Parseval等式（或另一方法）：将 $x^2$ 的Fourier展开在 $[-\pi, \pi]$ 上积分的平方关系，可得

$$\sum_{n=1}^{\infty} \frac{1}{n^4} = \frac{\pi^4}{90}$$

---

**5.** 计算Fourier系数：

$$a_0 = \frac{1}{\pi} \int_0^{\pi} 1 \, dx = 1$$

$$a_n = \frac{1}{\pi} \int_0^{\pi} \cos nx \, dx = \frac{1}{\pi} \cdot \frac{\sin nx}{n}\Big|_0^{\pi} = 0$$

$$b_n = \frac{1}{\pi} \int_0^{\pi} \sin nx \, dx = \frac{1}{\pi} \cdot \frac{1-\cos n\pi}{n} = \frac{1-(-1)^n}{n\pi}$$

当 $n$ 为偶数时，$b_n = 0$；当 $n = 2k+1$ 为奇数时，$b_n = \dfrac{2}{(2k+1)\pi}$。

$$f(x) = \frac{1}{2} + \frac{2}{\pi}\sum_{k=0}^{\infty} \frac{\sin(2k+1)x}{2k+1}$$

在 $x = \dfrac{\pi}{2}$ 处，$f\left(\dfrac{\pi}{2}\right) = 1$，且 $\sin\dfrac{(2k+1)\pi}{2} = (-1)^k$。

$$1 = \frac{1}{2} + \frac{2}{\pi}\sum_{k=0}^{\infty} \frac{(-1)^k}{2k+1}$$

因此

$$\sum_{k=0}^{\infty} \frac{(-1)^k}{2k+1} = 1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \cdots = \frac{\pi}{4}$$

这就是著名的**Leibniz公式**。

</details>
