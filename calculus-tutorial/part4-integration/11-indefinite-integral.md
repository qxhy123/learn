# 第11章 不定积分

## 学习目标

通过本章学习，你将能够：

- 理解原函数与不定积分的概念，掌握不定积分的几何意义
- 熟记基本积分公式表，能够直接应用公式求简单不定积分
- 掌握第一类换元法（凑微分法），熟练运用常见凑微分技巧
- 掌握第二类换元法，包括三角代换、根式代换和倒代换
- 掌握分部积分法，能够运用LIATE法则选择合适的函数分解

---

## 11.1 原函数与不定积分

### 11.1.1 原函数的定义

在学习导数时，我们研究的是"已知函数，求其导数"。现在我们反过来思考：已知一个函数的导数，能否找到原来的函数？

**定义**（原函数）：设 $f(x)$ 是定义在区间 $I$ 上的函数。若存在函数 $F(x)$，对 $I$ 上的每一点都有
$$F'(x) = f(x)$$
则称 $F(x)$ 是 $f(x)$ 在区间 $I$ 上的一个**原函数**。

> **例题 11.1** 验证 $F(x) = x^3$ 是 $f(x) = 3x^2$ 的一个原函数。

**解**：计算 $F'(x) = (x^3)' = 3x^2 = f(x)$。因此 $F(x) = x^3$ 确实是 $f(x) = 3x^2$ 的原函数。 $\square$

**原函数的非唯一性**：注意到 $(x^3 + 1)' = 3x^2$，$(x^3 - 5)' = 3x^2$。事实上，$x^3 + C$（$C$ 为任意常数）都是 $3x^2$ 的原函数。

**定理**（原函数的结构）：若 $F(x)$ 是 $f(x)$ 的一个原函数，则 $f(x)$ 的全部原函数为 $F(x) + C$，其中 $C$ 是任意常数。

**证明**：设 $G(x)$ 也是 $f(x)$ 的原函数，则 $G'(x) = f(x) = F'(x)$，故 $(G(x) - F(x))' = 0$。由拉格朗日中值定理，$G(x) - F(x)$ 在区间 $I$ 上为常数。 $\square$

### 11.1.2 不定积分的定义

**定义**（不定积分）：函数 $f(x)$ 的全体原函数称为 $f(x)$ 的**不定积分**，记作
$$\int f(x) \, dx = F(x) + C$$
其中：
- $\int$ 称为**积分号**
- $f(x)$ 称为**被积函数**
- $f(x) \, dx$ 称为**被积表达式**
- $x$ 称为**积分变量**
- $C$ 称为**积分常数**

### 11.1.3 不定积分的几何意义

不定积分 $\int f(x) \, dx = F(x) + C$ 表示一族曲线，称为**积分曲线族**。这些曲线具有相同的形状，只是在 $y$ 方向上平移了不同的距离。

几何上，$f(x)$ 在点 $x_0$ 处的值 $f(x_0)$ 就是过点 $(x_0, F(x_0))$ 的切线斜率。因此，积分曲线族中的每一条曲线在横坐标相同的点处具有相同的切线斜率。

### 11.1.4 基本性质

**性质1**（求导与积分互逆）：
$$\frac{d}{dx}\left[\int f(x) \, dx\right] = f(x) \quad \text{或} \quad \left[\int f(x) \, dx\right]' = f(x)$$

**性质2**（微分与积分互逆）：
$$\int F'(x) \, dx = F(x) + C \quad \text{或} \quad \int dF(x) = F(x) + C$$

**性质3**（线性性质）：
$$\int [af(x) + bg(x)] \, dx = a\int f(x) \, dx + b\int g(x) \, dx$$
其中 $a, b$ 为常数。

---

## 11.2 基本积分公式

由导数公式可以直接得到相应的积分公式。以下是常用的基本积分表：

### 11.2.1 幂函数与指数函数

| 序号 | 积分公式 | 对应的导数公式 |
|:---:|:---|:---|
| 1 | $\int k \, dx = kx + C$ | $(kx)' = k$ |
| 2 | $\int x^n \, dx = \dfrac{x^{n+1}}{n+1} + C \quad (n \neq -1)$ | $(x^{n+1})' = (n+1)x^n$ |
| 3 | $\int \dfrac{1}{x} \, dx = \ln|x| + C$ | $(\ln|x|)' = \dfrac{1}{x}$ |
| 4 | $\int e^x \, dx = e^x + C$ | $(e^x)' = e^x$ |
| 5 | $\int a^x \, dx = \dfrac{a^x}{\ln a} + C \quad (a > 0, a \neq 1)$ | $(a^x)' = a^x \ln a$ |

### 11.2.2 三角函数

| 序号 | 积分公式 | 对应的导数公式 |
|:---:|:---|:---|
| 6 | $\int \cos x \, dx = \sin x + C$ | $(\sin x)' = \cos x$ |
| 7 | $\int \sin x \, dx = -\cos x + C$ | $(\cos x)' = -\sin x$ |
| 8 | $\int \sec^2 x \, dx = \tan x + C$ | $(\tan x)' = \sec^2 x$ |
| 9 | $\int \csc^2 x \, dx = -\cot x + C$ | $(\cot x)' = -\csc^2 x$ |
| 10 | $\int \sec x \tan x \, dx = \sec x + C$ | $(\sec x)' = \sec x \tan x$ |
| 11 | $\int \csc x \cot x \, dx = -\csc x + C$ | $(\csc x)' = -\csc x \cot x$ |

### 11.2.3 反三角函数相关

| 序号 | 积分公式 |
|:---:|:---|
| 12 | $\int \dfrac{1}{\sqrt{1-x^2}} \, dx = \arcsin x + C$ |
| 13 | $\int \dfrac{1}{1+x^2} \, dx = \arctan x + C$ |
| 14 | $\int \dfrac{1}{\sqrt{x^2 \pm a^2}} \, dx = \ln|x + \sqrt{x^2 \pm a^2}| + C$ |
| 15 | $\int \dfrac{1}{x^2 - a^2} \, dx = \dfrac{1}{2a}\ln\left|\dfrac{x-a}{x+a}\right| + C$ |

> **例题 11.2** 求 $\int (3x^2 - 2\sin x + \dfrac{1}{x}) \, dx$。

**解**：利用线性性质和基本积分公式：
$$\int \left(3x^2 - 2\sin x + \frac{1}{x}\right) dx = 3 \cdot \frac{x^3}{3} - 2(-\cos x) + \ln|x| + C = x^3 + 2\cos x + \ln|x| + C$$

---

## 11.3 第一类换元法（凑微分法）

### 11.3.1 方法原理

**定理 11.1**（第一类换元法）：设 $f(u)$ 具有原函数 $F(u)$（即 $\int f(u) \, du = F(u) + C$），$u = \varphi(x)$ 在所考虑的区间上**连续可导**，则
$$\int f[\varphi(x)] \cdot \varphi'(x) \, dx = \int f[\varphi(x)] \, d\varphi(x) = F[\varphi(x)] + C$$

> **注**：条件 "$\varphi(x)$ 连续可导"不可省略。连续性保证 $\varphi(x)$ 的值域落在 $f(u)$ 有原函数的区间内，可导性保证 $\varphi'(x)$ 存在且 $d\varphi(x) = \varphi'(x)\,dx$ 有意义。若 $\varphi(x)$ 不可导，凑微分 $d\varphi(x)$ 这一步骤本身就无法进行。

**核心思想**：将 $g(x) \, dx$ 凑成 $d\varphi(x)$ 的形式，从而把复杂的积分转化为简单的积分。

### 11.3.2 常见凑微分技巧

以下是最常用的凑微分公式：

1. $x^n \, dx = \dfrac{1}{n+1} d(x^{n+1})$
2. $\dfrac{1}{x} \, dx = d(\ln|x|)$
3. $e^x \, dx = d(e^x)$
4. $\cos x \, dx = d(\sin x)$，$\sin x \, dx = -d(\cos x)$
5. $\sec^2 x \, dx = d(\tan x)$
6. $\dfrac{1}{\sqrt{1-x^2}} \, dx = d(\arcsin x)$
7. $\dfrac{1}{1+x^2} \, dx = d(\arctan x)$

### 11.3.3 例题详解

> **例题 11.3** 求 $\int \cos 2x \, dx$。

**解**：注意到 $d(2x) = 2 \, dx$，因此：
$$\int \cos 2x \, dx = \frac{1}{2} \int \cos 2x \cdot 2 \, dx = \frac{1}{2} \int \cos 2x \, d(2x) = \frac{1}{2} \sin 2x + C$$

> **例题 11.4** 求 $\int \dfrac{x}{1+x^2} \, dx$。

**解**：注意到 $d(1+x^2) = 2x \, dx$，因此：
$$\int \frac{x}{1+x^2} \, dx = \frac{1}{2} \int \frac{1}{1+x^2} \cdot 2x \, dx = \frac{1}{2} \int \frac{d(1+x^2)}{1+x^2} = \frac{1}{2} \ln(1+x^2) + C$$

> **例题 11.5** 求 $\int \tan x \, dx$。

**解**：将 $\tan x = \dfrac{\sin x}{\cos x}$，注意到 $d(\cos x) = -\sin x \, dx$：
$$\int \tan x \, dx = \int \frac{\sin x}{\cos x} \, dx = -\int \frac{d(\cos x)}{\cos x} = -\ln|\cos x| + C = \ln|\sec x| + C$$

> **例题 11.6** 求 $\int e^x \sin e^x \, dx$。

**解**：设 $u = e^x$，则 $du = e^x \, dx$：
$$\int e^x \sin e^x \, dx = \int \sin e^x \, d(e^x) = -\cos e^x + C$$

---

## 11.4 第二类换元法

当被积函数含有根式或某些特殊结构时，第一类换元法往往不适用。此时需要引入新变量来简化积分。

**定理 11.2**（第二类换元法）：设 $x = \varphi(t)$ 在区间 $I$ 上**严格单调、连续可导**，且 $\varphi'(t) \neq 0$。若

$$\int f[\varphi(t)] \cdot \varphi'(t) \, dt = G(t) + C$$

则

$$\int f(x) \, dx = G[\varphi^{-1}(x)] + C$$

其中 $\varphi^{-1}(x)$ 是 $\varphi(t)$ 的反函数。

> **注**：第二类换元法对 $\varphi(t)$ 的要求比第一类更强。**严格单调性**保证反函数 $\varphi^{-1}(x)$ 存在，使得最终能将变量 $t$ 回代为 $x$；**连续可导**保证微分替换 $dx = \varphi'(t)\,dt$ 合法；**$\varphi'(t) \neq 0$** 保证该替换是可逆的，不会丢失信息。若违反这些条件——例如代换函数不单调——则回代时可能产生歧义或得到错误结果。

### 11.4.1 三角代换

**适用情形**：被积函数含有 $\sqrt{a^2 - x^2}$、$\sqrt{x^2 + a^2}$ 或 $\sqrt{x^2 - a^2}$。

| 根式类型 | 代换方法 | 简化结果 |
|:---:|:---:|:---:|
| $\sqrt{a^2 - x^2}$ | $x = a\sin t$ | $a\cos t$ |
| $\sqrt{x^2 + a^2}$ | $x = a\tan t$ | $a\sec t$ |
| $\sqrt{x^2 - a^2}$ | $x = a\sec t$ | $a\tan t$ |

> **例题 11.7** 求 $\int \dfrac{1}{\sqrt{1-x^2}} \, dx$（用三角代换法验证）。

**解**：设 $x = \sin t$，$t \in (-\frac{\pi}{2}, \frac{\pi}{2})$，则 $dx = \cos t \, dt$，且 $\sqrt{1-x^2} = \cos t$。
$$\int \frac{1}{\sqrt{1-x^2}} \, dx = \int \frac{\cos t}{\cos t} \, dt = \int dt = t + C = \arcsin x + C$$

> **例题 11.8** 求 $\int \sqrt{a^2 - x^2} \, dx$（$a > 0$）。

**解**：设 $x = a\sin t$，$t \in [-\frac{\pi}{2}, \frac{\pi}{2}]$，则 $dx = a\cos t \, dt$，$\sqrt{a^2 - x^2} = a\cos t$。
$$\int \sqrt{a^2 - x^2} \, dx = \int a\cos t \cdot a\cos t \, dt = a^2 \int \cos^2 t \, dt$$

利用 $\cos^2 t = \dfrac{1 + \cos 2t}{2}$：
$$= a^2 \int \frac{1 + \cos 2t}{2} \, dt = \frac{a^2}{2}\left(t + \frac{\sin 2t}{2}\right) + C = \frac{a^2}{2}(t + \sin t \cos t) + C$$

将 $t = \arcsin\dfrac{x}{a}$，$\sin t = \dfrac{x}{a}$，$\cos t = \dfrac{\sqrt{a^2-x^2}}{a}$ 代回：
$$= \frac{a^2}{2} \arcsin\frac{x}{a} + \frac{x\sqrt{a^2-x^2}}{2} + C$$

### 11.4.2 根式代换

**适用情形**：被积函数含有 $\sqrt[n]{ax+b}$ 或 $\sqrt[n]{\dfrac{ax+b}{cx+d}}$ 等根式。

**方法**：设 $t = \sqrt[n]{ax+b}$，则 $x = \dfrac{t^n - b}{a}$，$dx = \dfrac{nt^{n-1}}{a} \, dt$。

> **例题 11.9** 求 $\int \dfrac{1}{1+\sqrt{x}} \, dx$。

**解**：设 $t = \sqrt{x}$，则 $x = t^2$，$dx = 2t \, dt$。
$$\int \frac{1}{1+\sqrt{x}} \, dx = \int \frac{2t}{1+t} \, dt = 2\int \frac{t+1-1}{1+t} \, dt = 2\int \left(1 - \frac{1}{1+t}\right) dt$$
$$= 2(t - \ln|1+t|) + C = 2\sqrt{x} - 2\ln(1+\sqrt{x}) + C$$

### 11.4.3 倒代换

**适用情形**：被积函数的分母次数较高，或分子次数比分母低较多。

**方法**：设 $x = \dfrac{1}{t}$，则 $dx = -\dfrac{1}{t^2} \, dt$。

> **例题 11.10** 求 $\int \dfrac{1}{x^2\sqrt{x^2+1}} \, dx$。

**解**：设 $x = \dfrac{1}{t}$（$t > 0$），则 $dx = -\dfrac{1}{t^2} \, dt$，$\sqrt{x^2+1} = \sqrt{\dfrac{1}{t^2}+1} = \dfrac{\sqrt{1+t^2}}{t}$。
$$\int \frac{1}{x^2\sqrt{x^2+1}} \, dx = \int \frac{t^2}{\frac{\sqrt{1+t^2}}{t}} \cdot \left(-\frac{1}{t^2}\right) dt = -\int \frac{t}{\sqrt{1+t^2}} \, dt$$
$$= -\sqrt{1+t^2} + C = -\sqrt{1+\frac{1}{x^2}} + C = -\frac{\sqrt{x^2+1}}{x} + C$$

---

## 11.5 分部积分法

### 11.5.1 分部积分公式

由乘积的微分公式 $(uv)' = u'v + uv'$，可得
$$uv' = (uv)' - u'v$$

两边积分：
$$\int u \, dv = uv - \int v \, du$$

这就是**分部积分公式**。

### 11.5.2 LIATE法则

在应用分部积分时，需要选择哪部分作为 $u$，哪部分作为 $dv$。一般原则是：选择 $u$ 使得 $u'$ 更简单，选择 $dv$ 使得 $v$ 容易求出。

**LIATE法则**提供了选择 $u$ 的优先顺序（从高到低）：

- **L**：对数函数（Logarithmic），如 $\ln x$
- **I**：反三角函数（Inverse trigonometric），如 $\arctan x$、$\arcsin x$
- **A**：代数函数（Algebraic），如 $x^n$、多项式
- **T**：三角函数（Trigonometric），如 $\sin x$、$\cos x$
- **E**：指数函数（Exponential），如 $e^x$

排在前面的优先作为 $u$。

> **例题 11.11** 求 $\int x e^x \, dx$。

**解**：按LIATE法则，$x$（代数）在 $e^x$（指数）之前，故取 $u = x$，$dv = e^x \, dx$。

则 $du = dx$，$v = e^x$。

$$\int x e^x \, dx = x e^x - \int e^x \, dx = x e^x - e^x + C = (x-1)e^x + C$$

> **例题 11.12** 求 $\int x^2 \cos x \, dx$。

**解**：取 $u = x^2$，$dv = \cos x \, dx$，则 $du = 2x \, dx$，$v = \sin x$。

$$\int x^2 \cos x \, dx = x^2 \sin x - 2\int x \sin x \, dx$$

对 $\int x \sin x \, dx$ 再次分部积分：取 $u = x$，$dv = \sin x \, dx$。

$$\int x \sin x \, dx = -x\cos x + \int \cos x \, dx = -x\cos x + \sin x + C_1$$

代入原式：
$$\int x^2 \cos x \, dx = x^2 \sin x - 2(-x\cos x + \sin x) + C = x^2 \sin x + 2x\cos x - 2\sin x + C$$

> **例题 11.13** 求 $\int \ln x \, dx$。

**解**：取 $u = \ln x$，$dv = dx$，则 $du = \dfrac{1}{x} \, dx$，$v = x$。

$$\int \ln x \, dx = x\ln x - \int x \cdot \frac{1}{x} \, dx = x\ln x - x + C = x(\ln x - 1) + C$$

### 11.5.3 循环积分

有时分部积分后会出现原积分，这时可以通过解方程求得结果。

> **例题 11.14** 求 $\int e^x \cos x \, dx$。

**解**：设 $I = \int e^x \cos x \, dx$。取 $u = \cos x$，$dv = e^x \, dx$：

$$I = e^x \cos x - \int e^x (-\sin x) \, dx = e^x \cos x + \int e^x \sin x \, dx$$

对 $\int e^x \sin x \, dx$，取 $u = \sin x$，$dv = e^x \, dx$：

$$\int e^x \sin x \, dx = e^x \sin x - \int e^x \cos x \, dx = e^x \sin x - I$$

代入原式：
$$I = e^x \cos x + e^x \sin x - I$$

解得：
$$2I = e^x(\cos x + \sin x)$$
$$I = \frac{e^x(\cos x + \sin x)}{2} + C$$

---

## 本章小结

1. **原函数与不定积分**：若 $F'(x) = f(x)$，则 $F(x)$ 是 $f(x)$ 的原函数。$f(x)$ 的全部原函数构成不定积分 $\int f(x) \, dx = F(x) + C$。

2. **基本积分公式**：熟记常用积分公式是求不定积分的基础，这些公式与导数公式一一对应。

3. **第一类换元法**（凑微分法）：利用 $\int f[\varphi(x)] \cdot \varphi'(x) \, dx = F[\varphi(x)] + C$，通过恰当的凑微分将复杂积分转化为简单积分。

4. **第二类换元法**：
   - 三角代换：适用于含 $\sqrt{a^2 - x^2}$、$\sqrt{x^2 + a^2}$、$\sqrt{x^2 - a^2}$ 的积分
   - 根式代换：适用于含 $\sqrt[n]{ax+b}$ 的积分
   - 倒代换：适用于分母次数较高的积分

5. **分部积分法**：利用公式 $\int u \, dv = uv - \int v \, du$，结合LIATE法则选择合适的 $u$ 和 $dv$。对于循环积分，可通过解方程求解。

---

## 深度学习应用

不定积分不只是抽象的数学工具——在深度学习中，积分是概率论、信息论和变分推断的核心语言。本节展示积分如何出现在现代机器学习的关键概念中。

### 11.6.1 概率密度函数与积分

概率密度函数 $p(x)$ 描述连续随机变量的分布，其核心约束是**归一化条件**：
$$\int_{-\infty}^{\infty} p(x) \, dx = 1$$

以标准正态分布为例：
$$p(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$$

归一化常数 $\dfrac{1}{\sqrt{2\pi}}$ 正是通过计算高斯积分 $\int_{-\infty}^{\infty} e^{-x^2/2} \, dx = \sqrt{2\pi}$ 得到的。一般正态分布 $\mathcal{N}(\mu, \sigma^2)$ 的密度为：
$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

其中 $\sigma$ 正是保证 $\int_{-\infty}^{\infty} p(x) \, dx = 1$ 成立的归一化因子。

### 11.6.2 期望的积分形式

随机变量函数的**期望**定义为：
$$\mathbb{E}[f(X)] = \int_{-\infty}^{\infty} f(x) \, p(x) \, dx$$

在深度学习中，损失函数通常以期望形式表达。例如，均方误差损失为：
$$\mathcal{L}(\theta) = \mathbb{E}_{(x,y) \sim p_{\text{data}}}\!\left[\|f_\theta(x) - y\|^2\right] = \int \|f_\theta(x) - y\|^2 \, p_{\text{data}}(x, y) \, dx \, dy$$

其中 $p_{\text{data}}$ 是数据的真实分布。由于我们只能用有限样本近似，训练时将积分替换为样本均值（蒙特卡洛估计）：
$$\mathcal{L}(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \|f_\theta(x_i) - y_i\|^2$$

### 11.6.3 KL散度与交叉熵

**KL散度**（Kullback-Leibler 散度）衡量分布 $q$ 与分布 $p$ 之间的差异：
$$D_{\mathrm{KL}}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx$$

KL散度具有非负性 $D_{\mathrm{KL}}(p \| q) \geq 0$，且当且仅当 $p = q$ 时等号成立（可用积分的 Jensen 不等式证明）。

在**变分推断**中，目标是找到近似后验分布 $q_\phi(z|x)$ 使其尽量接近真实后验 $p(z|x)$。优化目标（ELBO）包含 KL 散度的积分：
$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(z|x)}[\log p(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))$$

**交叉熵**与 KL 散度密切相关：
$$H(p, q) = -\int p(x) \log q(x) \, dx = H(p) + D_{\mathrm{KL}}(p \| q)$$

分类任务的交叉熵损失正是对真实分布与模型预测分布之间交叉熵的蒙特卡洛估计。

### 11.6.4 重参数化技巧

在变分自编码器（VAE）中，需要对 $z \sim q_\phi(z|x)$ 求期望的梯度：
$$\nabla_\phi \mathbb{E}_{z \sim q_\phi(z|x)}[f(z)] = \nabla_\phi \int f(z) \, q_\phi(z|x) \, dz$$

直接对积分求梯度很困难，因为积分域依赖于参数 $\phi$。**重参数化技巧**通过变量替换解决这一问题：

设 $q_\phi(z|x) = \mathcal{N}(\mu_\phi, \sigma_\phi^2)$，引入辅助变量 $\epsilon \sim \mathcal{N}(0, 1)$，令
$$z = \mu_\phi + \sigma_\phi \cdot \epsilon$$

则积分变量从 $z$ 换为 $\epsilon$（积分域不再依赖 $\phi$）：
$$\mathbb{E}_{z \sim \mathcal{N}(\mu_\phi, \sigma_\phi^2)}[f(z)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1)}[f(\mu_\phi + \sigma_\phi \cdot \epsilon)]$$

此时梯度可以移入期望内部，允许通过反向传播训练编码器参数。

### 11.6.5 代码示例

```python
import torch
import torch.distributions as dist

# 概率密度的归一化验证
normal = dist.Normal(0, 1)
x = torch.linspace(-5, 5, 1000)
dx = x[1] - x[0]

# 数值积分验证 ∫p(x)dx = 1
pdf = torch.exp(normal.log_prob(x))
integral = (pdf * dx).sum()
print(f"正态分布积分: {integral.item():.4f}")  # ≈ 1.0

# 期望的数值计算 E[X^2] = ∫x^2 p(x)dx
expectation = ((x**2) * pdf * dx).sum()
print(f"E[X^2] = {expectation.item():.4f}")  # ≈ 1.0 (方差)

# 重参数化技巧 (VAE)
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps  # z = μ + σ * ε
```

---

## 练习题

**1.** 求不定积分：$\int \dfrac{x^3 + 1}{x^2} \, dx$。

**2.** 用凑微分法求：$\int \dfrac{e^{\sqrt{x}}}{\sqrt{x}} \, dx$。

**3.** 用三角代换求：$\int \dfrac{x^2}{\sqrt{4-x^2}} \, dx$。

**4.** 用分部积分法求：$\int x^2 e^{-x} \, dx$。

**5.** 求不定积分：$\int e^{2x} \sin 3x \, dx$。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.** 先化简被积函数：
$$\int \frac{x^3 + 1}{x^2} \, dx = \int \left(x + \frac{1}{x^2}\right) dx = \int x \, dx + \int x^{-2} \, dx$$
$$= \frac{x^2}{2} + \frac{x^{-1}}{-1} + C = \frac{x^2}{2} - \frac{1}{x} + C$$

---

**2.** 设 $u = \sqrt{x}$，则 $du = \dfrac{1}{2\sqrt{x}} \, dx$，即 $\dfrac{dx}{\sqrt{x}} = 2 \, du$。
$$\int \frac{e^{\sqrt{x}}}{\sqrt{x}} \, dx = 2\int e^u \, du = 2e^u + C = 2e^{\sqrt{x}} + C$$

---

**3.** 设 $x = 2\sin t$，$t \in (-\frac{\pi}{2}, \frac{\pi}{2})$，则 $dx = 2\cos t \, dt$，$\sqrt{4-x^2} = 2\cos t$。
$$\int \frac{x^2}{\sqrt{4-x^2}} \, dx = \int \frac{4\sin^2 t}{2\cos t} \cdot 2\cos t \, dt = 4\int \sin^2 t \, dt$$

利用 $\sin^2 t = \dfrac{1 - \cos 2t}{2}$：
$$= 4 \cdot \frac{1}{2}\left(t - \frac{\sin 2t}{2}\right) + C = 2t - \sin 2t + C = 2t - 2\sin t \cos t + C$$

将 $t = \arcsin\dfrac{x}{2}$，$\sin t = \dfrac{x}{2}$，$\cos t = \dfrac{\sqrt{4-x^2}}{2}$ 代回：
$$= 2\arcsin\frac{x}{2} - \frac{x\sqrt{4-x^2}}{2} + C$$

---

**4.** 取 $u = x^2$，$dv = e^{-x} \, dx$，则 $du = 2x \, dx$，$v = -e^{-x}$。
$$\int x^2 e^{-x} \, dx = -x^2 e^{-x} + 2\int x e^{-x} \, dx$$

对 $\int x e^{-x} \, dx$，取 $u = x$，$dv = e^{-x} \, dx$：
$$\int x e^{-x} \, dx = -x e^{-x} + \int e^{-x} \, dx = -x e^{-x} - e^{-x}$$

代入：
$$\int x^2 e^{-x} \, dx = -x^2 e^{-x} + 2(-x e^{-x} - e^{-x}) + C = -e^{-x}(x^2 + 2x + 2) + C$$

---

**5.** 设 $I = \int e^{2x} \sin 3x \, dx$。取 $u = \sin 3x$，$dv = e^{2x} \, dx$：
$$I = \frac{1}{2}e^{2x}\sin 3x - \frac{3}{2}\int e^{2x}\cos 3x \, dx$$

对 $\int e^{2x}\cos 3x \, dx$，取 $u = \cos 3x$，$dv = e^{2x} \, dx$：
$$\int e^{2x}\cos 3x \, dx = \frac{1}{2}e^{2x}\cos 3x + \frac{3}{2}\int e^{2x}\sin 3x \, dx = \frac{1}{2}e^{2x}\cos 3x + \frac{3}{2}I$$

代入：
$$I = \frac{1}{2}e^{2x}\sin 3x - \frac{3}{2}\left(\frac{1}{2}e^{2x}\cos 3x + \frac{3}{2}I\right)$$
$$I = \frac{1}{2}e^{2x}\sin 3x - \frac{3}{4}e^{2x}\cos 3x - \frac{9}{4}I$$
$$\frac{13}{4}I = \frac{1}{2}e^{2x}\sin 3x - \frac{3}{4}e^{2x}\cos 3x$$
$$I = \frac{e^{2x}(2\sin 3x - 3\cos 3x)}{13} + C$$

</details>
