# 第12章 定积分

## 学习目标

通过本章学习，你将能够：

- 理解定积分的几何背景，掌握Riemann和与定积分的定义
- 掌握定积分的基本性质，包括线性性、区间可加性、保号性和积分中值定理
- 深刻理解微积分基本定理，熟练运用Newton-Leibniz公式计算定积分
- 掌握定积分的换元法和分部积分法，会利用对称性简化计算
- 能够应用定积分求平面图形的面积、旋转体的体积和曲线的弧长

---

## 12.1 定积分的概念

### 12.1.1 从面积问题引入

如何计算由曲线 $y = f(x)$（$f(x) \geq 0$）、直线 $x = a$、$x = b$ 及 $x$ 轴所围成的曲边梯形的面积？

**基本思想**："分割、近似、求和、取极限"

1. **分割**：将区间 $[a, b]$ 分成 $n$ 个小区间 $[x_{i-1}, x_i]$，其中 $a = x_0 < x_1 < \cdots < x_n = b$
2. **近似**：在每个小区间上取一点 $\xi_i \in [x_{i-1}, x_i]$，用小矩形面积 $f(\xi_i) \Delta x_i$ 近似曲边梯形的面积
3. **求和**：总面积近似为 $\sum_{i=1}^{n} f(\xi_i) \Delta x_i$
4. **取极限**：令分割越来越细，即 $\lambda = \max\{\Delta x_i\} \to 0$，得到精确面积

### 12.1.2 Riemann和的定义

**定义**（Riemann和）：设函数 $f(x)$ 在区间 $[a, b]$ 上有定义，对 $[a, b]$ 作分割 $P$：
$$a = x_0 < x_1 < x_2 < \cdots < x_n = b$$

记 $\Delta x_i = x_i - x_{i-1}$，$\lambda = \max_{1 \leq i \leq n}\{\Delta x_i\}$。在每个小区间 $[x_{i-1}, x_i]$ 上任取一点 $\xi_i$，则和式
$$S_n = \sum_{i=1}^{n} f(\xi_i) \Delta x_i$$
称为函数 $f(x)$ 在区间 $[a, b]$ 上的一个 **Riemann和**（或积分和）。

### 12.1.3 定积分的定义

**定义**（定积分）：设函数 $f(x)$ 在区间 $[a, b]$ 上有定义。若存在常数 $I$，对于任意给定的 $\varepsilon > 0$，总存在 $\delta > 0$，使得对 $[a, b]$ 的任意分割 $P$（只要 $\lambda < \delta$）以及任意选取的点 $\xi_i \in [x_{i-1}, x_i]$，都有
$$\left| \sum_{i=1}^{n} f(\xi_i) \Delta x_i - I \right| < \varepsilon$$
则称函数 $f(x)$ 在区间 $[a, b]$ 上**可积**，$I$ 称为 $f(x)$ 在 $[a, b]$ 上的**定积分**，记作
$$I = \int_a^b f(x) \, dx$$

其中：$a$ 称为**积分下限**，$b$ 称为**积分上限**，$[a, b]$ 称为**积分区间**。

**可积的充分条件**：
- 若 $f(x)$ 在 $[a, b]$ 上连续，则 $f(x)$ 在 $[a, b]$ 上可积
- 若 $f(x)$ 在 $[a, b]$ 上有界且只有有限个间断点，则 $f(x)$ 在 $[a, b]$ 上可积

> **例题 12.1** 利用定义计算 $\int_0^1 x^2 \, dx$。

**解**：将 $[0, 1]$ 等分为 $n$ 份，取 $x_i = \dfrac{i}{n}$，$\Delta x_i = \dfrac{1}{n}$，$\xi_i = x_i = \dfrac{i}{n}$。

Riemann和为：
$$S_n = \sum_{i=1}^{n} \left(\frac{i}{n}\right)^2 \cdot \frac{1}{n} = \frac{1}{n^3} \sum_{i=1}^{n} i^2 = \frac{1}{n^3} \cdot \frac{n(n+1)(2n+1)}{6}$$

$$= \frac{(n+1)(2n+1)}{6n^2} = \frac{2n^2 + 3n + 1}{6n^2}$$

取极限：
$$\int_0^1 x^2 \, dx = \lim_{n \to \infty} S_n = \lim_{n \to \infty} \frac{2n^2 + 3n + 1}{6n^2} = \frac{1}{3}$$

$\square$

**约定**：
- 当 $a = b$ 时，$\int_a^a f(x) \, dx = 0$
- 当 $a > b$ 时，$\int_a^b f(x) \, dx = -\int_b^a f(x) \, dx$

---

## 12.2 定积分的性质

### 12.2.1 线性性质

**性质1**（线性性）：设 $f(x)$、$g(x)$ 在 $[a, b]$ 上可积，$k_1$、$k_2$ 为常数，则
$$\int_a^b [k_1 f(x) + k_2 g(x)] \, dx = k_1 \int_a^b f(x) \, dx + k_2 \int_a^b g(x) \, dx$$

### 12.2.2 区间可加性

**性质2**（区间可加性）：设 $f(x)$ 在包含 $a$、$b$、$c$ 的区间上可积，则无论 $a$、$b$、$c$ 的相对位置如何，都有
$$\int_a^b f(x) \, dx = \int_a^c f(x) \, dx + \int_c^b f(x) \, dx$$

### 12.2.3 保号性与估值定理

**性质3**（保号性）：若 $f(x) \geq 0$ 在 $[a, b]$ 上成立，则
$$\int_a^b f(x) \, dx \geq 0$$

**推论**：若 $f(x) \geq g(x)$ 在 $[a, b]$ 上成立，则
$$\int_a^b f(x) \, dx \geq \int_a^b g(x) \, dx$$

**性质4**（估值定理）：设 $f(x)$ 在 $[a, b]$ 上可积，且 $m \leq f(x) \leq M$，则
$$m(b-a) \leq \int_a^b f(x) \, dx \leq M(b-a)$$

**性质5**（绝对值不等式）：若 $f(x)$ 在 $[a, b]$ 上可积，则 $|f(x)|$ 也可积，且
$$\left| \int_a^b f(x) \, dx \right| \leq \int_a^b |f(x)| \, dx$$

### 12.2.4 积分中值定理

**定理**（积分中值定理）：设 $f(x)$ 在 $[a, b]$ 上连续，则至少存在一点 $\xi \in [a, b]$，使得
$$\int_a^b f(x) \, dx = f(\xi)(b - a)$$

**几何意义**：曲边梯形的面积等于以 $[a, b]$ 为底、$f(\xi)$ 为高的矩形面积。

**证明**：由估值定理，设 $m$、$M$ 分别为 $f(x)$ 在 $[a, b]$ 上的最小值和最大值，则
$$m \leq \frac{1}{b-a} \int_a^b f(x) \, dx \leq M$$

由连续函数的介值定理，存在 $\xi \in [a, b]$，使得
$$f(\xi) = \frac{1}{b-a} \int_a^b f(x) \, dx$$

即 $\int_a^b f(x) \, dx = f(\xi)(b - a)$。$\square$

> **例题 12.2** 估计积分 $\int_0^1 e^{-x^2} \, dx$ 的值。

**解**：在 $[0, 1]$ 上，$0 \leq x^2 \leq 1$，故 $e^{-1} \leq e^{-x^2} \leq 1$。

由估值定理：
$$e^{-1} \cdot (1 - 0) \leq \int_0^1 e^{-x^2} \, dx \leq 1 \cdot (1 - 0)$$

即 $\dfrac{1}{e} \leq \int_0^1 e^{-x^2} \, dx \leq 1$，约为 $0.368 \leq I \leq 1$。$\square$

---

## 12.3 微积分基本定理

### 12.3.1 变上限积分函数

**定义**（变上限积分）：设 $f(x)$ 在 $[a, b]$ 上可积，定义函数
$$\Phi(x) = \int_a^x f(t) \, dt, \quad x \in [a, b]$$
称为 $f(x)$ 的**变上限积分函数**（或积分上限函数）。

注意：积分变量 $t$ 是哑变量，$\Phi(x)$ 是关于上限 $x$ 的函数。

### 12.3.2 微积分第一基本定理

**定理**（微积分第一基本定理）：设 $f(x)$ 在 $[a, b]$ 上连续，则变上限积分函数
$$\Phi(x) = \int_a^x f(t) \, dt$$
在 $[a, b]$ 上可导，且
$$\Phi'(x) = \frac{d}{dx} \int_a^x f(t) \, dt = f(x)$$

**证明**：对任意 $x \in [a, b)$，考虑增量
$$\Phi(x + \Delta x) - \Phi(x) = \int_a^{x+\Delta x} f(t) \, dt - \int_a^x f(t) \, dt = \int_x^{x+\Delta x} f(t) \, dt$$

由积分中值定理，存在 $\xi$ 介于 $x$ 与 $x + \Delta x$ 之间，使得
$$\int_x^{x+\Delta x} f(t) \, dt = f(\xi) \cdot \Delta x$$

因此
$$\frac{\Phi(x + \Delta x) - \Phi(x)}{\Delta x} = f(\xi)$$

当 $\Delta x \to 0$ 时，$\xi \to x$，由 $f$ 的连续性，$f(\xi) \to f(x)$。故
$$\Phi'(x) = \lim_{\Delta x \to 0} \frac{\Phi(x + \Delta x) - \Phi(x)}{\Delta x} = f(x)$$

$\square$

**推论**：若 $f(x)$ 连续，则 $\int_a^x f(t) \, dt$ 是 $f(x)$ 的一个原函数。

> **例题 12.3** 求 $\dfrac{d}{dx} \int_0^{x^2} \sin t \, dt$。

**解**：设 $u = x^2$，则
$$\frac{d}{dx} \int_0^{x^2} \sin t \, dt = \frac{d}{du} \int_0^u \sin t \, dt \cdot \frac{du}{dx} = \sin u \cdot 2x = 2x \sin x^2$$

$\square$

> **例题 12.4** 求 $\lim_{x \to 0} \dfrac{\int_0^x t e^{t^2} \, dt}{x^2}$。

**解**：这是 $\dfrac{0}{0}$ 型极限，用L'Hospital法则：
$$\lim_{x \to 0} \frac{\int_0^x t e^{t^2} \, dt}{x^2} = \lim_{x \to 0} \frac{x e^{x^2}}{2x} = \lim_{x \to 0} \frac{e^{x^2}}{2} = \frac{1}{2}$$

$\square$

### 12.3.3 微积分第二基本定理（Newton-Leibniz公式）

**定理**（Newton-Leibniz公式）：设 $f(x)$ 在 $[a, b]$ 上连续，$F(x)$ 是 $f(x)$ 的任意一个原函数，则
$$\int_a^b f(x) \, dx = F(b) - F(a) \triangleq F(x) \Big|_a^b$$

**证明**：由微积分第一基本定理，$\Phi(x) = \int_a^x f(t) \, dt$ 是 $f(x)$ 的一个原函数。

由原函数的结构定理，$F(x) = \Phi(x) + C$，其中 $C$ 为某常数。

因此：
$$F(b) - F(a) = [\Phi(b) + C] - [\Phi(a) + C] = \Phi(b) - \Phi(a)$$
$$= \int_a^b f(t) \, dt - \int_a^a f(t) \, dt = \int_a^b f(x) \, dx$$

$\square$

**Newton-Leibniz公式的意义**：它将定积分的计算转化为求原函数的问题，使得定积分的计算变得简便。这是微积分中最重要的公式之一。

> **例题 12.5** 计算 $\int_0^{\pi/2} \cos x \, dx$。

**解**：$\cos x$ 的一个原函数是 $\sin x$，由Newton-Leibniz公式：
$$\int_0^{\pi/2} \cos x \, dx = \sin x \Big|_0^{\pi/2} = \sin\frac{\pi}{2} - \sin 0 = 1 - 0 = 1$$

$\square$

> **例题 12.6** 计算 $\int_1^e \dfrac{1}{x} \, dx$。

**解**：$\dfrac{1}{x}$ 的一个原函数是 $\ln x$（$x > 0$）：
$$\int_1^e \frac{1}{x} \, dx = \ln x \Big|_1^e = \ln e - \ln 1 = 1 - 0 = 1$$

$\square$

---

## 12.4 定积分的计算

### 12.4.1 换元法

**定理**（定积分的换元法）：设 $f(x)$ 在 $[a, b]$ 上连续，若函数 $x = \varphi(t)$ 满足：
1. $\varphi(\alpha) = a$，$\varphi(\beta) = b$
2. $\varphi(t)$ 在 $[\alpha, \beta]$（或 $[\beta, \alpha]$）上有连续导数，且值域包含于 $[a, b]$

则
$$\int_a^b f(x) \, dx = \int_\alpha^\beta f[\varphi(t)] \varphi'(t) \, dt$$

**注意**：换元后，积分限也要相应改变；计算完毕后无需换回原变量。

> **例题 12.7** 计算 $\int_0^4 \sqrt{x}(1 + \sqrt{x}) \, dx$。

**解**：设 $t = \sqrt{x}$，则 $x = t^2$，$dx = 2t \, dt$。当 $x = 0$ 时 $t = 0$，当 $x = 4$ 时 $t = 2$。
$$\int_0^4 \sqrt{x}(1 + \sqrt{x}) \, dx = \int_0^2 t(1 + t) \cdot 2t \, dt = 2\int_0^2 (t^2 + t^3) \, dt$$
$$= 2\left[\frac{t^3}{3} + \frac{t^4}{4}\right]_0^2 = 2\left(\frac{8}{3} + 4\right) = 2 \cdot \frac{20}{3} = \frac{40}{3}$$

$\square$

> **例题 12.8** 计算 $\int_0^1 \sqrt{1 - x^2} \, dx$。

**解**：设 $x = \sin t$，则 $dx = \cos t \, dt$，$\sqrt{1 - x^2} = \cos t$。当 $x = 0$ 时 $t = 0$，当 $x = 1$ 时 $t = \dfrac{\pi}{2}$。
$$\int_0^1 \sqrt{1 - x^2} \, dx = \int_0^{\pi/2} \cos t \cdot \cos t \, dt = \int_0^{\pi/2} \cos^2 t \, dt$$
$$= \int_0^{\pi/2} \frac{1 + \cos 2t}{2} \, dt = \frac{1}{2}\left[t + \frac{\sin 2t}{2}\right]_0^{\pi/2} = \frac{1}{2} \cdot \frac{\pi}{2} = \frac{\pi}{4}$$

几何意义：这正是单位圆在第一象限部分的面积。$\square$

### 12.4.2 分部积分法

**定理**（定积分的分部积分法）：设 $u(x)$、$v(x)$ 在 $[a, b]$ 上有连续导数，则
$$\int_a^b u \, dv = uv \Big|_a^b - \int_a^b v \, du$$

> **例题 12.9** 计算 $\int_0^1 x e^x \, dx$。

**解**：取 $u = x$，$dv = e^x \, dx$，则 $du = dx$，$v = e^x$。
$$\int_0^1 x e^x \, dx = x e^x \Big|_0^1 - \int_0^1 e^x \, dx = e - (e^x \Big|_0^1) = e - (e - 1) = 1$$

$\square$

> **例题 12.10** 计算 $\int_0^{\pi/2} e^x \sin x \, dx$。

**解**：设 $I = \int_0^{\pi/2} e^x \sin x \, dx$。分部积分两次：

第一次：$u = \sin x$，$dv = e^x \, dx$
$$I = e^x \sin x \Big|_0^{\pi/2} - \int_0^{\pi/2} e^x \cos x \, dx = e^{\pi/2} - \int_0^{\pi/2} e^x \cos x \, dx$$

第二次：$u = \cos x$，$dv = e^x \, dx$
$$\int_0^{\pi/2} e^x \cos x \, dx = e^x \cos x \Big|_0^{\pi/2} + \int_0^{\pi/2} e^x \sin x \, dx = -1 + I$$

代入：$I = e^{\pi/2} - (-1 + I) = e^{\pi/2} + 1 - I$

解得：$I = \dfrac{e^{\pi/2} + 1}{2}$ $\square$

### 12.4.3 对称性的利用

**定理**（奇偶函数的定积分）：设 $f(x)$ 在 $[-a, a]$ 上连续，则：

1. 若 $f(x)$ 为**偶函数**，则 $\int_{-a}^a f(x) \, dx = 2\int_0^a f(x) \, dx$
2. 若 $f(x)$ 为**奇函数**，则 $\int_{-a}^a f(x) \, dx = 0$

**证明**：由区间可加性，$\int_{-a}^a f(x) \, dx = \int_{-a}^0 f(x) \, dx + \int_0^a f(x) \, dx$

对 $\int_{-a}^0 f(x) \, dx$，设 $x = -t$，则
$$\int_{-a}^0 f(x) \, dx = -\int_a^0 f(-t) \, dt = \int_0^a f(-t) \, dt$$

若 $f$ 为偶函数，$f(-t) = f(t)$，则 $\int_{-a}^0 f(x) \, dx = \int_0^a f(t) \, dt$，故 $\int_{-a}^a f(x) \, dx = 2\int_0^a f(x) \, dx$。

若 $f$ 为奇函数，$f(-t) = -f(t)$，则 $\int_{-a}^0 f(x) \, dx = -\int_0^a f(t) \, dt$，故 $\int_{-a}^a f(x) \, dx = 0$。$\square$

> **例题 12.11** 计算 $\int_{-1}^1 (x^3 + x^4) \, dx$。

**解**：$x^3$ 是奇函数，$x^4$ 是偶函数。
$$\int_{-1}^1 (x^3 + x^4) \, dx = \int_{-1}^1 x^3 \, dx + \int_{-1}^1 x^4 \, dx = 0 + 2\int_0^1 x^4 \, dx = 2 \cdot \frac{x^5}{5}\Big|_0^1 = \frac{2}{5}$$

$\square$

---

## 12.5 定积分的应用（几何）

### 12.5.1 平面图形的面积

**情形1**：由曲线 $y = f(x) \geq 0$、$x = a$、$x = b$ 及 $x$ 轴围成的面积：
$$S = \int_a^b f(x) \, dx$$

**情形2**：由曲线 $y = f(x)$ 与 $y = g(x)$（$f(x) \geq g(x)$）及 $x = a$、$x = b$ 围成的面积：
$$S = \int_a^b [f(x) - g(x)] \, dx$$

**情形3**：由参数方程 $x = x(t)$，$y = y(t)$（$\alpha \leq t \leq \beta$）所围成的面积：
$$S = \int_\alpha^\beta |y(t) x'(t)| \, dt$$

> **例题 12.12** 求由抛物线 $y = x^2$ 与直线 $y = x$ 所围成的面积。

**解**：先求交点：$x^2 = x$，得 $x = 0$ 或 $x = 1$。在 $[0, 1]$ 上，$x \geq x^2$。
$$S = \int_0^1 (x - x^2) \, dx = \left[\frac{x^2}{2} - \frac{x^3}{3}\right]_0^1 = \frac{1}{2} - \frac{1}{3} = \frac{1}{6}$$

$\square$

### 12.5.2 旋转体的体积

**绕 $x$ 轴旋转**：由曲线 $y = f(x)$、$x = a$、$x = b$ 及 $x$ 轴围成的图形绕 $x$ 轴旋转所得旋转体的体积：
$$V_x = \pi \int_a^b [f(x)]^2 \, dx$$

**绕 $y$ 轴旋转**（圆柱壳法）：
$$V_y = 2\pi \int_a^b x |f(x)| \, dx$$

> **例题 12.13** 求由 $y = \sqrt{x}$、$x = 1$ 及 $x$ 轴围成的图形绕 $x$ 轴旋转所得旋转体的体积。

**解**：
$$V_x = \pi \int_0^1 (\sqrt{x})^2 \, dx = \pi \int_0^1 x \, dx = \pi \cdot \frac{x^2}{2}\Big|_0^1 = \frac{\pi}{2}$$

$\square$

> **例题 12.14** 求由 $y = \sin x$（$0 \leq x \leq \pi$）与 $x$ 轴围成的图形绕 $y$ 轴旋转所得旋转体的体积。

**解**：用圆柱壳法：
$$V_y = 2\pi \int_0^\pi x \sin x \, dx$$

分部积分：取 $u = x$，$dv = \sin x \, dx$，则 $du = dx$，$v = -\cos x$。
$$\int_0^\pi x \sin x \, dx = -x \cos x \Big|_0^\pi + \int_0^\pi \cos x \, dx = \pi + \sin x \Big|_0^\pi = \pi$$

故 $V_y = 2\pi \cdot \pi = 2\pi^2$。$\square$

### 12.5.3 曲线的弧长

**直角坐标形式**：曲线 $y = f(x)$（$a \leq x \leq b$）的弧长为：
$$L = \int_a^b \sqrt{1 + [f'(x)]^2} \, dx$$

**参数形式**：曲线 $x = x(t)$，$y = y(t)$（$\alpha \leq t \leq \beta$）的弧长为：
$$L = \int_\alpha^\beta \sqrt{[x'(t)]^2 + [y'(t)]^2} \, dt$$

> **例题 12.15** 求曲线 $y = \dfrac{2}{3}x^{3/2}$ 从 $x = 0$ 到 $x = 1$ 的弧长。

**解**：$y' = x^{1/2}$，$1 + (y')^2 = 1 + x$。
$$L = \int_0^1 \sqrt{1 + x} \, dx = \frac{2}{3}(1 + x)^{3/2}\Big|_0^1 = \frac{2}{3}(2\sqrt{2} - 1)$$

$\square$

---

## 本章小结

1. **定积分的定义**：定积分是Riemann和的极限，即 $\int_a^b f(x) \, dx = \lim_{\lambda \to 0} \sum_{i=1}^n f(\xi_i) \Delta x_i$。它源于面积问题，体现了"分割、近似、求和、取极限"的思想。

2. **定积分的性质**：
   - 线性性：$\int_a^b [k_1 f + k_2 g] \, dx = k_1 \int_a^b f \, dx + k_2 \int_a^b g \, dx$
   - 区间可加性：$\int_a^b f \, dx = \int_a^c f \, dx + \int_c^b f \, dx$
   - 积分中值定理：存在 $\xi \in [a, b]$ 使 $\int_a^b f(x) \, dx = f(\xi)(b - a)$

3. **微积分基本定理**：
   - 第一基本定理：$\dfrac{d}{dx} \int_a^x f(t) \, dt = f(x)$（连接微分与积分）
   - 第二基本定理（Newton-Leibniz公式）：$\int_a^b f(x) \, dx = F(b) - F(a)$

4. **计算方法**：
   - 换元法：换元后积分限相应改变，结果无需换回
   - 分部积分法：$\int_a^b u \, dv = uv \Big|_a^b - \int_a^b v \, du$
   - 利用对称性：奇函数在对称区间上积分为零，偶函数可简化为两倍

5. **几何应用**：
   - 面积：$S = \int_a^b |f(x) - g(x)| \, dx$
   - 旋转体体积：$V_x = \pi \int_a^b [f(x)]^2 \, dx$
   - 弧长：$L = \int_a^b \sqrt{1 + [f'(x)]^2} \, dx$

---

## 12.6 深度学习应用

定积分不仅是几何工具，在深度学习和机器学习中也有重要的理论与实践意义。

### 12.6.1 损失函数的积分形式

在有监督学习中，损失函数衡量模型预测与真实值之间的差距。

**经验风险**（基于有限样本）：
$$\hat{R}(f) = \frac{1}{n}\sum_{i=1}^n L(f(x_i), y_i)$$

这是对有限样本点的求和，本质上是对真实期望的估计。

**期望风险**（基于数据的真实分布）：
$$R(f) = \int L(f(x), y)\, p(x,y)\, dx\, dy$$

其中 $p(x, y)$ 是输入-输出对 $(x, y)$ 的联合概率密度函数。

**关系**：经验风险是期望风险的 Monte Carlo 估计，当 $n \to \infty$ 时，
$$\hat{R}(f) \xrightarrow{a.s.} R(f)$$

由大数定律保证收敛，这正是 Riemann 和收敛到定积分的随机版本。

### 12.6.2 ROC 曲线下面积（AUC）

ROC（Receiver Operating Characteristic）曲线描述二分类器在不同阈值下的性能，横轴为假正率（FPR），纵轴为真正率（TPR）。

**AUC 的积分定义**：

$$\text{AUC} = \int_0^1 \text{TPR}\!\left(\text{FPR}^{-1}(t)\right) dt$$

即以 FPR 为积分变量，对 TPR 求定积分。AUC = 1 表示完美分类，AUC = 0.5 对应随机猜测。

**梯形法则近似**：实践中，AUC 通过 $n$ 个离散阈值下的 (FPR, TPR) 点对，用梯形法则（数值积分）计算：
$$\text{AUC} \approx \sum_{i=1}^{n-1} \frac{(\text{FPR}_{i+1} - \text{FPR}_i)(\text{TPR}_i + \text{TPR}_{i+1})}{2}$$

这正是定积分数值计算的直接应用。

### 12.6.3 积分在正则化中的应用

**权重衰减的路径积分解释**

L2 正则化（权重衰减）在贝叶斯框架下等价于对参数施加高斯先验 $p(\theta) \propto e^{-\lambda \|\theta\|^2}$。

后验分布满足：
$$p(\theta | \mathcal{D}) \propto p(\mathcal{D} | \theta)\, p(\theta)$$

对 $\theta$ 的边际化需要计算积分：
$$p(\mathcal{D}) = \int p(\mathcal{D} | \theta)\, p(\theta)\, d\theta$$

**函数空间正则化（Sobolev 正则化）**

另一类正则化直接限制函数的"光滑程度"，通过限制其导数的积分来实现：
$$\Omega(f) = \int \left[f''(x)\right]^2 dx$$

这是一个以导数的平方为被积函数的定积分，鼓励模型选择曲率较小的函数。

### 12.6.4 Newton-Leibniz 公式与自动微分

**变上限积分与自动微分**

微积分第一基本定理揭示了积分与微分的互逆关系：
$$\frac{d}{dx} \int_a^x f(t)\, dt = f(x)$$

在深度学习的自动微分（Autograd）框架中，这一原理被直接利用：若将数值积分 $F(x) = \int_a^x f(t)\, dt$ 视为一个计算节点，则其反向传播梯度即为被积函数在上限处的值 $f(x)$。

**神经 ODE（Neural ODE）**

Neural ODE 将神经网络的前向传播建模为常微分方程的解：
$$\mathbf{h}(T) = \mathbf{h}(0) + \int_0^T f_\theta(\mathbf{h}(t), t)\, dt$$

其中 $f_\theta$ 是参数化的神经网络。训练时，梯度通过**伴随方法**（adjoint method）高效计算，本质上是对 ODE 做反向积分，避免了存储中间状态的开销。

### 12.6.5 代码示例

```python
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

# AUC 的积分计算
def compute_auc_manual(y_true, y_score):
    """手动计算 AUC = ∫TPR d(FPR)"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # 梯形法则积分
    auc = np.trapz(tpr, fpr)
    return auc

# 示例
y_true = np.array([0, 0, 1, 1, 1])
y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.9])

auc_sklearn = roc_auc_score(y_true, y_score)
auc_manual = compute_auc_manual(y_true, y_score)
print(f"sklearn AUC: {auc_sklearn:.4f}")
print(f"手动积分 AUC: {auc_manual:.4f}")

# 变上限积分函数的求导（利用 PyTorch 自动微分）
# F(t) = ∫_0^t e^{-s^2} ds 的数值近似及其导数
def F(t, n=1000):
    """数值计算变上限积分 F(t) = ∫_0^t e^{-s^2} ds"""
    s = torch.linspace(0, t, n)
    return torch.trapz(torch.exp(-s**2), s)

# 在 t=1 处验证 F'(1) ≈ e^{-1}
t = torch.tensor(1.0, requires_grad=True)
val = F(t)
val.backward()
print(f"F'(1) 数值结果: {t.grad.item():.6f}")
print(f"e^{{-1}} 理论值: {np.exp(-1):.6f}")
```

**关键联系**：
- `np.trapz` 实现的梯形法则对应定积分的数值近似
- `torch.trapz` + `backward()` 利用微积分第一基本定理自动计算变上限积分的导数
- AUC 的计算本质是 ROC 曲线下方面积的定积分

---

## 练习题

**1.** 利用定积分的性质，证明：$\int_0^{\pi/2} \sin^n x \, dx = \int_0^{\pi/2} \cos^n x \, dx$。

**2.** 计算定积分：$\int_0^2 x\sqrt{4 - x^2} \, dx$。

**3.** 计算定积分：$\int_0^1 x^2 e^x \, dx$。

**4.** 求由曲线 $y = e^x$、$y = e^{-x}$ 与直线 $x = 1$ 所围成图形的面积。

**5.** 求由曲线 $y = x^2$ 与 $y = \sqrt{x}$ 围成的图形绕 $x$ 轴旋转所得旋转体的体积。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.** 设 $x = \dfrac{\pi}{2} - t$，则 $dx = -dt$。当 $x = 0$ 时 $t = \dfrac{\pi}{2}$，当 $x = \dfrac{\pi}{2}$ 时 $t = 0$。

$$\int_0^{\pi/2} \sin^n x \, dx = \int_{\pi/2}^0 \sin^n\left(\frac{\pi}{2} - t\right) \cdot (-dt) = \int_0^{\pi/2} \cos^n t \, dt$$

故 $\int_0^{\pi/2} \sin^n x \, dx = \int_0^{\pi/2} \cos^n x \, dx$。$\square$

---

**2.** 设 $u = 4 - x^2$，则 $du = -2x \, dx$，即 $x \, dx = -\dfrac{1}{2} du$。当 $x = 0$ 时 $u = 4$，当 $x = 2$ 时 $u = 0$。

$$\int_0^2 x\sqrt{4 - x^2} \, dx = -\frac{1}{2}\int_4^0 \sqrt{u} \, du = \frac{1}{2}\int_0^4 u^{1/2} \, du$$
$$= \frac{1}{2} \cdot \frac{2}{3} u^{3/2}\Big|_0^4 = \frac{1}{3} \cdot 8 = \frac{8}{3}$$

---

**3.** 两次分部积分。取 $u = x^2$，$dv = e^x \, dx$：

$$\int_0^1 x^2 e^x \, dx = x^2 e^x \Big|_0^1 - 2\int_0^1 x e^x \, dx = e - 2\int_0^1 x e^x \, dx$$

对 $\int_0^1 x e^x \, dx$，取 $u = x$，$dv = e^x \, dx$：

$$\int_0^1 x e^x \, dx = x e^x \Big|_0^1 - \int_0^1 e^x \, dx = e - (e - 1) = 1$$

代入：$\int_0^1 x^2 e^x \, dx = e - 2 \cdot 1 = e - 2$

---

**4.** 在 $x \in [0, 1]$ 上，$e^x \geq e^{-x}$。两曲线在 $x = 0$ 处交于 $(0, 1)$。

$$S = \int_0^1 (e^x - e^{-x}) \, dx = (e^x + e^{-x})\Big|_0^1 = (e + e^{-1}) - (1 + 1) = e + \frac{1}{e} - 2$$

---

**5.** 两曲线的交点：$x^2 = \sqrt{x}$ 得 $x^4 = x$，即 $x(x^3 - 1) = 0$，所以 $x = 0$ 或 $x = 1$。

在 $[0, 1]$ 上，$\sqrt{x} \geq x^2$。

$$V = \pi \int_0^1 \left[(\sqrt{x})^2 - (x^2)^2\right] dx = \pi \int_0^1 (x - x^4) \, dx$$
$$= \pi \left[\frac{x^2}{2} - \frac{x^5}{5}\right]_0^1 = \pi \left(\frac{1}{2} - \frac{1}{5}\right) = \frac{3\pi}{10}$$

</details>
