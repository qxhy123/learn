# 公式速查表

## 1. 基本导数公式

| 函数 $f(x)$ | 导数 $f'(x)$ |
|:---:|:---:|
| $c$（常数） | $0$ |
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $a^x$ | $a^x \ln a$ |
| $\ln x$ | $\dfrac{1}{x}$ |
| $\log_a x$ | $\dfrac{1}{x \ln a}$ |

### 三角函数

| 函数 | 导数 |
|:---:|:---:|
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $\tan x$ | $\sec^2 x$ |
| $\cot x$ | $-\csc^2 x$ |
| $\sec x$ | $\sec x \tan x$ |
| $\csc x$ | $-\csc x \cot x$ |

### 反三角函数

| 函数 | 导数 |
|:---:|:---:|
| $\arcsin x$ | $\dfrac{1}{\sqrt{1-x^2}}$ |
| $\arccos x$ | $-\dfrac{1}{\sqrt{1-x^2}}$ |
| $\arctan x$ | $\dfrac{1}{1+x^2}$ |
| $\text{arccot}\, x$ | $-\dfrac{1}{1+x^2}$ |

### 双曲函数

| 函数 | 导数 |
|:---:|:---:|
| $\sinh x$ | $\cosh x$ |
| $\cosh x$ | $\sinh x$ |
| $\tanh x$ | $\text{sech}^2 x$ |
| $\text{coth}\, x$ | $-\text{csch}^2 x$ |

---

## 2. 基本积分公式

| 被积函数 | 积分结果 |
|:---:|:---:|
| $x^n \; (n \neq -1)$ | $\dfrac{x^{n+1}}{n+1} + C$ |
| $\dfrac{1}{x}$ | $\ln \|x\| + C$ |
| $e^x$ | $e^x + C$ |
| $a^x$ | $\dfrac{a^x}{\ln a} + C$ |
| $\sin x$ | $-\cos x + C$ |
| $\cos x$ | $\sin x + C$ |
| $\sec^2 x$ | $\tan x + C$ |
| $\csc^2 x$ | $-\cot x + C$ |
| $\sec x \tan x$ | $\sec x + C$ |
| $\csc x \cot x$ | $-\csc x + C$ |
| $\dfrac{1}{\sqrt{1-x^2}}$ | $\arcsin x + C$ |
| $\dfrac{1}{1+x^2}$ | $\arctan x + C$ |
| $\dfrac{1}{\sqrt{x^2 \pm a^2}}$ | $\ln \|x + \sqrt{x^2 \pm a^2}\| + C$ |
| $\dfrac{1}{a^2 + x^2}$ | $\dfrac{1}{a}\arctan\dfrac{x}{a} + C$ |
| $\dfrac{1}{a^2 - x^2}$ | $\dfrac{1}{2a}\ln\left\|\dfrac{a+x}{a-x}\right\| + C$ |

---

## 3. 求导法则

### 四则运算

| 法则 | 公式 |
|:---|:---:|
| 和差 | $(u \pm v)' = u' \pm v'$ |
| 常数倍 | $(cu)' = cu'$ |
| 乘积 | $(uv)' = u'v + uv'$ |
| 商 | $\left(\dfrac{u}{v}\right)' = \dfrac{u'v - uv'}{v^2}$ |

### 链式法则

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

或写作：$[f(g(x))]' = f'(g(x)) \cdot g'(x)$

### 隐函数求导

设 $F(x, y) = 0$ 确定 $y = y(x)$，则：
$$\frac{dy}{dx} = -\frac{F_x}{F_y}$$

### 参数方程求导

设 $x = x(t)$，$y = y(t)$，则：
$$\frac{dy}{dx} = \frac{y'(t)}{x'(t)}, \quad \frac{d^2y}{dx^2} = \frac{y''(t)x'(t) - y'(t)x''(t)}{[x'(t)]^3}$$

---

## 4. 积分技巧

### 换元法

**第一类（凑微分）**：$\displaystyle\int f(g(x))g'(x)\,dx = \int f(u)\,du$，其中 $u = g(x)$

**第二类**：设 $x = \varphi(t)$，则 $\displaystyle\int f(x)\,dx = \int f(\varphi(t))\varphi'(t)\,dt$

### 分部积分

$$\int u\,dv = uv - \int v\,du$$

口诀：**反对幂指三**（按此顺序选 $u$）

### 常用三角代换

| 被积式含有 | 代换 | 范围 |
|:---:|:---:|:---:|
| $\sqrt{a^2 - x^2}$ | $x = a\sin t$ | $t \in [-\frac{\pi}{2}, \frac{\pi}{2}]$ |
| $\sqrt{a^2 + x^2}$ | $x = a\tan t$ | $t \in (-\frac{\pi}{2}, \frac{\pi}{2})$ |
| $\sqrt{x^2 - a^2}$ | $x = a\sec t$ | $t \in [0, \frac{\pi}{2}) \cup (\frac{\pi}{2}, \pi]$ |

---

## 5. Taylor/Maclaurin 展开

| 函数 | Maclaurin 展开 | 收敛域 |
|:---:|:---|:---:|
| $e^x$ | $\displaystyle\sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$ | $(-\infty, +\infty)$ |
| $\sin x$ | $\displaystyle\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!} = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$ | $(-\infty, +\infty)$ |
| $\cos x$ | $\displaystyle\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!} = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \cdots$ | $(-\infty, +\infty)$ |
| $\ln(1+x)$ | $\displaystyle\sum_{n=1}^{\infty} \frac{(-1)^{n-1} x^n}{n} = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots$ | $(-1, 1]$ |
| $(1+x)^\alpha$ | $\displaystyle\sum_{n=0}^{\infty} \binom{\alpha}{n} x^n = 1 + \alpha x + \frac{\alpha(\alpha-1)}{2!}x^2 + \cdots$ | $(-1, 1)^*$ |
| $\arctan x$ | $\displaystyle\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{2n+1} = x - \frac{x^3}{3} + \frac{x^5}{5} - \cdots$ | $[-1, 1]$ |

> $^*$ 当 $\alpha > 0$ 时在 $x = -1$ 收敛；当 $\alpha > -1$ 时在 $x = 1$ 收敛

---

## 6. 级数收敛判别法

设 $\sum a_n$ 为正项级数。

| 判别法 | 方法 | 结论 |
|:---|:---|:---|
| **比值判别法** | $\displaystyle\lim_{n\to\infty}\frac{a_{n+1}}{a_n} = \rho$ | $\rho < 1$ 收敛，$\rho > 1$ 发散，$\rho = 1$ 不确定 |
| **根值判别法** | $\displaystyle\lim_{n\to\infty}\sqrt[n]{a_n} = \rho$ | $\rho < 1$ 收敛，$\rho > 1$ 发散，$\rho = 1$ 不确定 |
| **比较判别法** | 与已知级数 $\sum b_n$ 比较 | $a_n \leq b_n$ 且 $\sum b_n$ 收敛 $\Rightarrow \sum a_n$ 收敛 |
| **极限比较** | $\displaystyle\lim_{n\to\infty}\frac{a_n}{b_n} = L \in (0, +\infty)$ | 同敛散 |
| **积分判别法** | $a_n = f(n)$，$f$ 单调递减 | $\sum a_n$ 与 $\int_1^\infty f(x)\,dx$ 同敛散 |

### 交错级数（Leibniz 判别法）

若 $a_n > 0$，$a_n$ 单调递减，且 $\lim a_n = 0$，则 $\sum (-1)^n a_n$ 收敛。

---

## 7. 多元微积分

### 偏导数与梯度

$$\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}\right) = f_x\,\mathbf{i} + f_y\,\mathbf{j} + f_z\,\mathbf{k}$$

### 方向导数

$$\frac{\partial f}{\partial \mathbf{l}} = \nabla f \cdot \mathbf{e}_l = |\nabla f| \cos\theta$$

### 散度与旋度

设 $\mathbf{F} = (P, Q, R)$

| 算子 | 定义 |
|:---:|:---:|
| 散度 | $\nabla \cdot \mathbf{F} = \dfrac{\partial P}{\partial x} + \dfrac{\partial Q}{\partial y} + \dfrac{\partial R}{\partial z}$ |
| 旋度 | $\nabla \times \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ P & Q & R \end{vmatrix}$ |
| Laplace | $\nabla^2 f = \dfrac{\partial^2 f}{\partial x^2} + \dfrac{\partial^2 f}{\partial y^2} + \dfrac{\partial^2 f}{\partial z^2}$ |

### 坐标变换

| 坐标系 | 变换公式 | Jacobi 行列式 |
|:---|:---|:---:|
| **极坐标** | $x = r\cos\theta$，$y = r\sin\theta$ | $r$ |
| **柱坐标** | $x = r\cos\theta$，$y = r\sin\theta$，$z = z$ | $r$ |
| **球坐标** | $x = \rho\sin\varphi\cos\theta$，$y = \rho\sin\varphi\sin\theta$，$z = \rho\cos\varphi$ | $\rho^2\sin\varphi$ |

---

## 8. 向量分析定理

### Green 公式（平面）

$$\oint_L P\,dx + Q\,dy = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA$$

> $L$ 为 $D$ 的正向边界（逆时针）

### Gauss 公式（散度定理）

$$\oiint_S \mathbf{F} \cdot d\mathbf{S} = \iiint_V (\nabla \cdot \mathbf{F})\,dV$$

或写作：
$$\oiint_S P\,dydz + Q\,dzdx + R\,dxdy = \iiint_V \left(\frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}\right) dV$$

### Stokes 公式（旋度定理）

$$\oint_L \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S}$$

或写作：
$$\oint_L P\,dx + Q\,dy + R\,dz = \iint_S \begin{vmatrix} dydz & dzdx & dxdy \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ P & Q & R \end{vmatrix}$$

---

## 9. 常微分方程

### 一阶线性方程

$$y' + P(x)y = Q(x)$$

**通解**：
$$y = e^{-\int P\,dx}\left[\int Q e^{\int P\,dx}\,dx + C\right]$$

### 可分离变量

$$\frac{dy}{dx} = f(x)g(y) \quad\Rightarrow\quad \int\frac{dy}{g(y)} = \int f(x)\,dx$$

### 齐次方程

$$\frac{dy}{dx} = \varphi\left(\frac{y}{x}\right)$$

令 $u = \dfrac{y}{x}$，则 $y = ux$，$y' = u + xu'$

### 二阶常系数线性齐次方程

$$y'' + py' + qy = 0$$

**特征方程**：$r^2 + pr + q = 0$

| 特征根 | 通解 |
|:---|:---|
| $r_1 \neq r_2$（实根） | $y = C_1 e^{r_1 x} + C_2 e^{r_2 x}$ |
| $r_1 = r_2 = r$（重根） | $y = (C_1 + C_2 x)e^{rx}$ |
| $r = \alpha \pm \beta i$（共轭复根） | $y = e^{\alpha x}(C_1\cos\beta x + C_2\sin\beta x)$ |

### 二阶常系数非齐次方程

$$y'' + py' + qy = f(x)$$

**通解** = 齐次通解 + 特解

| $f(x)$ 的形式 | 特解设法 |
|:---|:---|
| $P_m(x)e^{\lambda x}$ | $y^* = x^k Q_m(x)e^{\lambda x}$，$k$ = $\lambda$ 作为特征根的重数 |
| $e^{\lambda x}[P_l(x)\cos\omega x + P_n(x)\sin\omega x]$ | $y^* = x^k e^{\lambda x}[R_m(x)\cos\omega x + S_m(x)\sin\omega x]$，$m = \max(l,n)$ |

---

## 10. 常用恒等式

### 三角恒等式

$$\sin^2 x + \cos^2 x = 1, \quad 1 + \tan^2 x = \sec^2 x, \quad 1 + \cot^2 x = \csc^2 x$$

$$\sin 2x = 2\sin x\cos x, \quad \cos 2x = \cos^2 x - \sin^2 x$$

$$\cos^2 x = \frac{1 + \cos 2x}{2}, \quad \sin^2 x = \frac{1 - \cos 2x}{2}$$

### 双曲恒等式

$$\cosh^2 x - \sinh^2 x = 1$$

$$\sinh x = \frac{e^x - e^{-x}}{2}, \quad \cosh x = \frac{e^x + e^{-x}}{2}$$

---

## 11. 矩阵微积分常用公式（A.7）

| 公式 | 说明 |
|:---|:---|
| $\dfrac{\partial (Ax)}{\partial x} = A$ | 线性层对输入的 Jacobian |
| $\dfrac{\partial (x^\top a)}{\partial x} = a$ | 向量内积求导 |
| $\dfrac{\partial (x^\top A x)}{\partial x} = (A + A^\top)x$ | 对称矩阵时化为 $2Ax$ |
| $\dfrac{\partial \operatorname{tr}(AB)}{\partial A} = B^\top$ | 迹技巧基础公式 |
| $\dfrac{\partial \operatorname{tr}(ABA^\top)}{\partial A} = A(B + B^\top)$ | 常见二次型矩阵求导 |
| $\dfrac{\partial \ln\|A\|}{\partial A} = A^{-\top}$ | 对数行列式求导 |
| $\dfrac{\partial A^{-1}}{\partial t} = -A^{-1}\left(\dfrac{\partial A}{\partial t}\right)A^{-1}$ | 逆矩阵微分公式 |

---

## 12. 概率分布积分公式（A.8）

| 公式 | 说明 |
|:---|:---|
| $\displaystyle \int_{-\infty}^{+\infty} e^{-ax^2}\,dx = \sqrt{\frac{\pi}{a}} \quad (a>0)$ | 高斯积分 |
| $\displaystyle \Gamma(n) = \int_0^\infty t^{n-1}e^{-t}\,dt = (n-1)! \quad (n\in\mathbb{N}^+)$ | Gamma 函数 |
| $\displaystyle B(\alpha,\beta) = \int_0^1 t^{\alpha-1}(1-t)^{\beta-1}\,dt = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ | Beta 函数 |
| $\displaystyle \mathrm{KL}\!\left(\mathcal N(\mu_1,\sigma_1^2)\|\mathcal N(\mu_2,\sigma_2^2)\right) = \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12$ | 一维高斯 KL 散度 |

---

## 13. 常用不等式（A.9）

| 不等式 | 形式 |
|:---|:---|
| AM-GM | $\displaystyle \frac{a+b}{2} \ge \sqrt{ab}$ |
| Cauchy-Schwarz | $\displaystyle |\langle x,y\rangle| \le \|x\|\,\|y\|$ |
| Jensen | $\displaystyle f(\mathbb E[X]) \le \mathbb E[f(X)]$（$f$ 为凸函数） |
| Young | $\displaystyle ab \le \frac{a^p}{p} + \frac{b^q}{q}, \quad \frac{1}{p}+\frac{1}{q}=1$ |
| Hölder | $\displaystyle \|fg\|_1 \le \|f\|_p\,\|g\|_q, \quad \frac{1}{p}+\frac{1}{q}=1$ |

---

> 本速查表涵盖微积分核心公式，供复习和快速查阅使用。

---

## 14. Toolkit 12 篇核心要点速查

> 以下为 `thinking-toolkit/` 12 篇的精华摘录，每篇 5–8 行，配合正文 toolkit 使用效果最佳。

### TK-01 极限的 ε 语言（epsilon-language）

- **ε-N 定义**：$\forall\varepsilon>0,\;\exists N\in\mathbb{N},\;\forall n>N:\;|a_n-A|<\varepsilon$，称 $\{a_n\}$ 收敛于 $A$。
- **ε-δ 定义**：$\forall\varepsilon>0,\;\exists\delta>0,\;\forall 0<|x-x_0|<\delta:\;|f(x)-L|<\varepsilon$，称 $\lim_{x\to x_0}f(x)=L$。
- **三步证法**：① 设任意 $\varepsilon>0$；② 从不等式 $|a_n-A|<\varepsilon$ 反解 $n$ 的下界 $N(\varepsilon)$；③ 写"当 $n>N$ 时 $\cdots<\varepsilon$"验证。
- **常见陷阱**：$N$ 必须只依赖 $\varepsilon$，不得依赖 $n$；$\delta$ 取 $\min\{\cdots\}$ 时各分支须验证。
- **否命题**：$\{a_n\}$ 不收敛于 $A$ $\Leftrightarrow$ $\exists\varepsilon_0>0,\;\forall N,\;\exists n>N:\;|a_n-A|\geq\varepsilon_0$。
- **两个重要极限**：$\lim_{x\to0}\dfrac{\sin x}{x}=1$；$\lim_{x\to\infty}\!\left(1+\dfrac{1}{x}\right)^x=e$（或等价地 $\lim_{x\to0}(1+x)^{1/x}=e$）。

### TK-02 等价无穷小与小 o 记号（equivalent-infinitesimals）

- **等价定义**：$\alpha\sim\beta \Leftrightarrow \lim\dfrac{\alpha}{\beta}=1 \Leftrightarrow \alpha=\beta+o(\beta)$（$x\to x_0$）。
- **乘除安全，加减危险**：$\lim\dfrac{\alpha\cdot f}{\beta\cdot g}=\lim\dfrac{f}{g}$（合法）；加减中直接替换可能出错（用 Taylor！）。
- **常用等价表**（$x\to 0$）：$\sin x\sim x$，$\tan x\sim x$，$1-\cos x\sim\dfrac{x^2}{2}$，$e^x-1\sim x$，$\ln(1+x)\sim x$，$(1+x)^\alpha-1\sim\alpha x$，$\arcsin x\sim x$，$\arctan x\sim x$。
- **加减陷阱示例**：$\lim_{x\to0}\dfrac{\sin x-\tan x}{x^3}$，不能各自替换为 $x$；需 Taylor 展开：$\sin x-\tan x=-\dfrac{x^3}{2}+o(x^3)$，极限为 $-\dfrac{1}{2}$。
- **小 o 运算**：$o(x^m)+o(x^n)=o(x^{\min(m,n)})$；$o(x^m)\cdot x^n=o(x^{m+n})$；常数 $\times o(x^n)=o(x^n)$。

### TK-03 求导套路系统化（differentiation-rules）

- **6 大规则决策树**：看到 $f+g$ → 和法则；$f\cdot g$ → 乘积法则；$f/g$ → 商法则；$f(g(x))$ → 链式；$f^g$（均含 $x$）→ 对数求导；$F(x,y)=0$ → 隐函数。
- **对数求导法**：$y=f^g \Rightarrow \ln y=g\ln f \Rightarrow \dfrac{y'}{y}=g'\ln f+g\cdot\dfrac{f'}{f} \Rightarrow y'=y\!\left(g'\ln f+g\dfrac{f'}{f}\right)$。
- **隐函数公式**：$F(x,y)=0 \Rightarrow y'=-\dfrac{F_x}{F_y}$（$F_y\neq0$）；高阶隐函数导数：再对 $x$ 微分一次即可。
- **参数方程**：$x=x(t),y=y(t) \Rightarrow \dfrac{dy}{dx}=\dfrac{y'(t)}{x'(t)}$，$\dfrac{d^2y}{dx^2}=\dfrac{\left(y'/x'\right)'_t}{x'(t)}$（分母再乘 $x'(t)$）。
- **Leibniz 公式**（高阶导数）：$(uv)^{(n)}=\sum_{k=0}^n\binom{n}{k}u^{(k)}v^{(n-k)}$；常用：$(\sin x)^{(n)}=\sin(x+n\pi/2)$，$(\cos x)^{(n)}=\cos(x+n\pi/2)$。

### TK-04 积分技巧：LIATE 与换元（integration-techniques）

- **分部积分公式**：$\int u\,dv=uv-\int v\,du$；选 $u$ 按 LIATE 优先级（**L**og > **I**nverse-trig > **A**lgebra > **T**rig > **E**xp）。
- **循环积分处理**：对 $\int e^x\sin x\,dx$ 型，分部两次后设 $I=$ 原积分，移项解方程，得 $I=e^x(\sin x-\cos x)/2+C$。
- **5 大换元**：① 凑微分；② 三角换元（三种情形见第 4 节）；③ 万能换元（$t=\tan\frac{x}{2}$）；④ 倒代换（$x=1/t$）；⑤ 部分分式（有理函数）。
- **定积分换元**：换元同时更换积分上下限，回代后直接代入，无需换回原变量。
- **对称区间化简**：$\int_{-a}^a f\,dx = 2\int_0^a f\,dx$（$f$ 偶）或 $0$（$f$ 奇）；华里士公式 $\int_0^{\pi/2}\sin^n x\,dx$ 的递推记法。

### TK-05 级数判敛决策树（series-convergence）

- **步骤 0（必要条件）**：$a_n\not\to0$ $\Rightarrow$ 立刻发散；$a_n\to0$ 只是必要条件（调和级数 $\sum 1/n$ 反例）。
- **步骤 1（识别类型）**：正项级数 → 步骤 2；交错级数 → Leibniz；任意项 → 先检验绝对收敛。
- **步骤 2（比值法）**：$\rho=\lim\left|\dfrac{a_{n+1}}{a_n}\right|$；含 $n!$、$a^n$、$n^n$ 时首选；$\rho=1$ 失效。
- **步骤 3（根值法）**：$\rho=\lim\sqrt[n]{|a_n|}$；通项含 $(\cdot)^n$ 时首选；$\rho=1$ 失效。
- **步骤 4（比较法）**：$p$-级数（$p>1$ 收敛）、等比、调和（发散）为三大参照；极限比较 $L\in(0,+\infty)$ 则同敛散；$f(n)$ 单调递减正值时用积分判别。
- **Leibniz 判别**：$b_n>0$，$b_n$ 单调递减，$b_n\to0$ $\Rightarrow$ $\sum(-1)^n b_n$ 条件收敛；误差 $|S-S_n|\leq b_{n+1}$。

### TK-06 Taylor 展开与误差（taylor-and-error）

- **6 大 Maclaurin 展开**：$e^x$，$\sin x$，$\cos x$，$\ln(1+x)$，$(1+x)^\alpha$，$\arctan x$（详见第 15 节）。
- **Peano 余项**：$f(x)=P_n(x)+o\!\left((x-x_0)^n\right)$，用于求极限（误差可被忽略）。
- **Lagrange 余项**：$R_n(x)=\dfrac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}$，$\xi$ 介于 $x_0$ 和 $x$ 之间，用于截断误差估计。
- **展开阶的选择**：极限题中展开到分子分母最低幂次能约消为止；误差估计中展开到所要求精度。
- **复合展开**：先展内层再代入外层，如 $e^{x^2}$ 直接替换 $x\to x^2$；注意收敛域的变化。
- **Taylor vs L'Hôpital**：含多重复合函数、或需要展开多项时，Taylor 更高效；简单"零比零"时两者效率相当。

### TK-07 多元链式与梯度（multivar-chain）

- **基本形式**（$z=f(u,v)$，$u,v$ 均依赖 $x,y$）：$\partial z/\partial x=f_u\cdot u_x+f_v\cdot v_x$；每条路径上的偏导连乘，所有路径相加。
- **树形依赖图**：画变量依赖树，从 $z$ 到目标变量的每条路径乘积，再求和。
- **梯度向量**：$\nabla f=\left(f_{x_1},\ldots,f_{x_n}\right)^T$；方向导数 $\partial f/\partial\mathbf{l}=\nabla f\cdot\mathbf{e}_l$；梯度方向是函数值增速最大的方向。
- **Jacobian 矩阵**：$(J_f)_{ij}=\partial f_i/\partial x_j$；多元链式 $=$ Jacobian 矩阵乘法 $J_{z\circ u}=J_z\cdot J_u$。
- **全微分**：$dz=z_x\,dx+z_y\,dy$；误差传播：$|\Delta z|\approx|z_x||\Delta x|+|z_y||\Delta y|$。
- **反向传播实质**：对参数 $W_k$ 的梯度，沿计算图由后往前，每层连乘 Jacobian（标量情形即逐元素链式）。

### TK-08 多元积分变换（multivar-integration）

- **坐标选择口诀**：矩形区域 → 直角坐标；圆盘/扇形 → 极坐标（$dA=r\,dr\,d\theta$）；绕 $z$ 轴旋转体 → 柱坐标（$dV=r\,dr\,d\theta\,dz$）；球体/球壳 → 球坐标（$dV=\rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$）。
- **换元关键**：新变量的 Jacobian 行列式的绝对值 $|J|$ 不能省略；$dA\to|J|\,du\,dv$。
- **积分次序交换**：先画积分区域 $D$，重新确定积分上下限（X 型或 Y 型区域）。
- **对称性化简**：$D$ 关于 $x$ 轴对称且被积函数关于 $y$ 为奇函数 → 积分为零；偶函数 → 乘 $2$；各向同性区域 + 关于 $y=x$ 对称可交换变量。
- **三重积分投影法**：先对 $z$ 积分（由上下曲面决定 $z$ 的范围），再对投影区域 $D_{xy}$ 做二重积分。
- **Fubini 定理**：在被积函数连续且积分区域有界的条件下，累次积分顺序可交换。

### TK-09 常微分方程类型识别（ode-classification）

- **一阶 ODE 5 类识别树**：先看能否分离变量 → 检验 $y/x$ 型（齐次）→ 检验线性（$y'+p y=q$）→ 检验 Bernoulli（$y'+py=qy^n$）→ 验证恰当（$P_y=Q_x$）。
- **一阶线性通解**：$y=e^{-\int p\,dx}\!\left[\int q\,e^{\int p\,dx}\,dx+C\right]$（积分因子法）。
- **3 种二阶特征根**：两不同实根 → $C_1e^{r_1x}+C_2e^{r_2x}$；重根 $r$ → $(C_1+C_2x)e^{rx}$；复根 $\alpha\pm\beta i$ → $e^{\alpha x}(C_1\cos\beta x+C_2\sin\beta x)$。
- **非齐次特解设法**：$f(x)=P_m e^{\lambda x}$ 时，$y^*=x^kQ_me^{\lambda x}$（$k=\lambda$ 的重数 0/1/2）。
- **初值问题**：由 $y(x_0)=y_0$ 确定 $C$；二阶 ODE 需 $y(x_0)=y_0$ 和 $y'(x_0)=y_0'$ 两个初始条件。
- **ODE 与 AI**：常见激活函数 $\sigma$ 满足的 ODE（如 $\sigma'=\sigma(1-\sigma)$）和 Neural ODE 的 $\dot z=f(z,t;\theta)$。

### TK-10 凸性、单调与极值（convexity-extrema）

- **凸函数定义**：$f(tx_1+(1-t)x_2)\leq tf(x_1)+(1-t)f(x_2)$（$t\in[0,1]$）；充分条件 $f''\geq0$（单变量）。
- **多元凸性**：Hessian $H=\nabla^2f\succcurlyeq0$（半正定）$\Leftrightarrow$ $f$ 凸；$H\succ0$（正定）$\Rightarrow$ $f$ 严格凸。
- **极值判定**：必要 $\nabla f=\mathbf{0}$（驻点）；充分：$H$ 正定 → 极小；负定 → 极大；不定 → 鞍点。
- **KKT 简表**：等式约束 $h_i=0$ + 不等式约束 $g_j\leq0$：$\nabla f+\sum\lambda_i\nabla h_i+\sum\mu_j\nabla g_j=\mathbf{0}$，互补松弛 $\mu_j g_j=0$，$\mu_j\geq0$。
- **凸问题的特权**：局部最小 = 全局最小；KKT 条件既必要又充分；梯度下降从任意初值出发均可收敛。
- **Jensen 不等式**：$f$ 凸 $\Rightarrow$ $f(\mathbb{E}X)\leq\mathbb{E}f(X)$；KL 散度非负性的基础。

### TK-11 微积分中的不等式技巧（inequality-techniques）

- **技巧 1（单调性）**：构造 $h=f-g$，通过 $h'$ 的符号分析证 $h\geq0$；有时需再求一次导（证 $h'$ 单调）。
- **技巧 2（Taylor 放缩）**：凸函数切线在图像下方 $f(x)\geq f(a)+f'(a)(x-a)$；Taylor 余项正负可用于单向放缩。
- **技巧 3（积分估计）**：$m(b-a)\leq\int_a^b f\,dx\leq M(b-a)$；三角不等式 $\left|\int f\right|\leq\int|f|$；换元后比较被积函数大小。
- **技巧 4（均值不等式族）**：AM-GM → Young 不等式 → Hölder 不等式 → Cauchy-Schwarz（链式依赖）。
- **微积分常用放缩**：$\sin x<x<\tan x$（$x\in(0,\pi/2)$）；$x>\ln(1+x)>x-x^2/2$（$x>0$）；$e^x\geq1+x$（恒成立）。
- **证明路标**：先看等号何时成立 → 等号成立点往往就是辅助函数的极值点 → 在该点验证最小值为零。

### TK-12 微积分中的 AI 思维（calculus-for-ai）

- **梯度下降**：$\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)$；负梯度方向是函数值减小最快的方向（一阶 Taylor 展开的直接推论）。
- **收敛保证**：Lipschitz 梯度常数 $\beta$ 下，$\eta\leq1/\beta$ 时每步保证 $L(\theta_{t+1})\leq L(\theta_t)-\frac{\eta}{2}\|\nabla L\|^2$。
- **反向传播 = 多元链式**：损失对参数 $W_k$ 的梯度，沿计算图由后往前连乘 Jacobian（或逐元素链式）。
- **Hessian 的角色**：$H=\nabla^2L$；正定 → 严格极小；$\lambda_{\max}$ 决定最优步长 $\eta^*=1/\lambda_{\max}$；条件数 $\kappa(H)$ 影响收敛速度。
- **KL 散度**：$\mathrm{KL}(p\|q)=\mathbb{E}_p\ln\frac{p}{q}\geq0$（Jensen 不等式保证）；最小化交叉熵等价于最小化对真实分布的 KL 散度。
- **Adam vs GD**：Adam 用二阶矩估计近似 Hessian 对角线，实现自适应步长；在梯度稀疏时远优于标准 GD。

---

## 15. 6 大 Maclaurin 展开整理表

> 来自 toolkit/06，无条件记忆的"基本弹药"。

| 函数 | Maclaurin 展开（级数形式） | 前几项 | 收敛域 |
|:---:|:---|:---|:---:|
| $e^x$ | $\displaystyle\sum_{n=0}^{\infty}\dfrac{x^n}{n!}$ | $1+x+\dfrac{x^2}{2!}+\dfrac{x^3}{3!}+\dfrac{x^4}{4!}+\cdots$ | $(-\infty,+\infty)$ |
| $\sin x$ | $\displaystyle\sum_{n=0}^{\infty}\dfrac{(-1)^n x^{2n+1}}{(2n+1)!}$ | $x-\dfrac{x^3}{6}+\dfrac{x^5}{120}-\dfrac{x^7}{5040}+\cdots$ | $(-\infty,+\infty)$ |
| $\cos x$ | $\displaystyle\sum_{n=0}^{\infty}\dfrac{(-1)^n x^{2n}}{(2n)!}$ | $1-\dfrac{x^2}{2}+\dfrac{x^4}{24}-\dfrac{x^6}{720}+\cdots$ | $(-\infty,+\infty)$ |
| $\ln(1+x)$ | $\displaystyle\sum_{n=1}^{\infty}\dfrac{(-1)^{n-1}x^n}{n}$ | $x-\dfrac{x^2}{2}+\dfrac{x^3}{3}-\dfrac{x^4}{4}+\cdots$ | $(-1,1]$ |
| $(1+x)^\alpha$ | $\displaystyle\sum_{n=0}^{\infty}\binom{\alpha}{n}x^n$ | $1+\alpha x+\dfrac{\alpha(\alpha-1)}{2!}x^2+\dfrac{\alpha(\alpha-1)(\alpha-2)}{3!}x^3+\cdots$ | $(-1,1)^*$ |
| $\arctan x$ | $\displaystyle\sum_{n=0}^{\infty}\dfrac{(-1)^n x^{2n+1}}{2n+1}$ | $x-\dfrac{x^3}{3}+\dfrac{x^5}{5}-\dfrac{x^7}{7}+\cdots$ | $[-1,1]$ |

> $^*$ $(1+x)^\alpha$：$\alpha>0$ 时 $x=-1$ 收敛；$\alpha>-1$ 时 $x=1$ 收敛；$-1<\alpha<0$ 时仅 $(-1,1)$ 内收敛。

### 常用特例（从 $(1+x)^\alpha$ 推导）

| 特例 | 展开（前三项） | 适用范围 |
|:---:|:---|:---:|
| $\dfrac{1}{1-x}$ | $1+x+x^2+x^3+\cdots$ | $|x|<1$ |
| $\dfrac{1}{1+x}$ | $1-x+x^2-x^3+\cdots$ | $|x|<1$ |
| $\sqrt{1+x}$ | $1+\dfrac{x}{2}-\dfrac{x^2}{8}+\dfrac{x^3}{16}-\cdots$ | $|x|<1$ |
| $\dfrac{1}{\sqrt{1+x}}$ | $1-\dfrac{x}{2}+\dfrac{3x^2}{8}-\dfrac{5x^3}{16}+\cdots$ | $|x|<1$ |

### 记忆规律速查

| 函数 | 幂次规律 | 符号规律 |
|:---:|:---:|:---:|
| $e^x$ | 全部 $0,1,2,3,\ldots$ | 全部为正 |
| $\sin x$ | 奇数幂次 $1,3,5,\ldots$ | 交替 $+,-,+,-$ |
| $\cos x$ | 偶数幂次 $0,2,4,\ldots$ | 交替 $+,-,+,-$ |
| $\ln(1+x)$ | 全部 $1,2,3,\ldots$ | 交替 $+,-,+,-$ |
| $\arctan x$ | 奇数幂次 $1,3,5,\ldots$ | 交替 $+,-,+,-$ |

**关系检验**：$\cos x=(\sin x)'$，对 $\sin x$ 的展开逐项求导即得 $\cos x$ 的展开；两者可互相验证。

---

## 16. LIATE 优先级 + 5 大换元

> 来自 toolkit/04。

### LIATE 优先级表

选 $u$ 的顺序（越靠前越优先，$dv$ 选另一个）：

| 字母 | 类型 | 典型函数 | 选为 $u$ 的理由 |
|:---:|:---:|:---|:---|
| **L** | 对数型 | $\ln x$，$\log_a x$ | 求导后消去对数，化为代数式 |
| **I** | 反三角型 | $\arcsin x$，$\arctan x$ | 求导后化为代数，直接积分困难 |
| **A** | 代数/多项式型 | $x^n$，$\sqrt{x}$，$x^{-1}$ | 求导后降次，最终归零 |
| **T** | 三角型 | $\sin x$，$\cos x$ | 求导后仍为三角，可构成循环 |
| **E** | 指数型 | $e^x$，$a^x$ | 求导后不变，放 $dv$ 最稳定 |

**循环积分口诀**：$\int e^x\sin x\,dx$ 型分部两次后原积分再现，设 $I=$ 原积分，移项解方程。

### 常见配对

| 被积函数 | 选 $u$ | 选 $dv$ |
|:---:|:---:|:---:|
| $x^n e^x$ | $u=x^n$（A > E）| $dv=e^x\,dx$ |
| $x^n\sin x$ | $u=x^n$（A > T）| $dv=\sin x\,dx$ |
| $(\ln x)\cdot x^n$ | $u=\ln x$（L > A）| $dv=x^n\,dx$ |
| $(\arctan x)\cdot x^n$ | $u=\arctan x$（I > A）| $dv=x^n\,dx$ |
| $e^x\sin x$ | $u=\sin x$ 或 $e^x$（均可循环）| 另一个 |

### 5 大换元类型

| 类型 | 触发信号 | 换元操作 | 注意事项 |
|:---:|:---|:---|:---|
| **① 凑微分** | 含 $f(g(x))g'(x)$ 结构 | $u=g(x)$，$du=g'(x)\,dx$ | 识别分子是分母的导数（或其倍数）|
| **② 三角换元** | $\sqrt{a^2-x^2}$ | $x=a\sin t$，$t\in[-\pi/2,\pi/2]$ | 回代用 $\sin t=x/a$，$\cos t=\sqrt{1-(x/a)^2}$ |
| | $\sqrt{a^2+x^2}$ | $x=a\tan t$，$t\in(-\pi/2,\pi/2)$ | $\sqrt{a^2+x^2}=a\sec t$，$\sec t>0$ |
| | $\sqrt{x^2-a^2}$ | $x=a\sec t$，$t\in[0,\pi/2)\cup(\pi/2,\pi]$ | 分 $x>a$ 和 $x<-a$ 两区间讨论 |
| **③ 万能换元** | 有理三角式 $R(\sin x,\cos x)$ | $t=\tan\dfrac{x}{2}$，$\sin x=\dfrac{2t}{1+t^2}$，$\cos x=\dfrac{1-t^2}{1+t^2}$，$dx=\dfrac{2}{1+t^2}dt$ | 计算量大，其他换元失败时使用 |
| **④ 倒代换** | 含 $1/x^n$ 或无穷端点处的衰减 | $t=1/x$，$dt=-dx/x^2$ | 积分上下限需同步变换 |
| **⑤ 部分分式** | 有理函数 $P(x)/Q(x)$，$\deg P<\deg Q$ | 将 $Q(x)$ 分解为不可约因子的乘积，对每个因子设待定系数 | 复杂因子 $(x^2+px+q)^k$ 对应 $\dfrac{Ax+B}{x^2+px+q}$ 型 |

---

## 17. 级数判敛决策树文字版

> 来自 toolkit/05。沿决策树走一遍，30 秒锁定正确判别法。

```
【级数判敛决策树】

Step 0：必要条件检验
  a_n → 0 吗？
  否 → 立刻发散（结束）
  是 → 继续（仅是必要条件，不充分）

Step 1：识别类型
  ├── 正项级数（a_n ≥ 0）→ Step 2
  ├── 交错级数（(-1)^n b_n，b_n > 0）→ Step 5（Leibniz）
  └── 任意项级数（符号无规律）
        → 先检验绝对收敛：对 |a_n| 用 Step 2–4
        → 绝对收敛 ⟹ 收敛（但反之不成立）

Step 2：比值法（达朗贝尔判别法）
  触发信号：通项含 n!、a^n、n^n 的乘积
  ρ = lim_{n→∞} |a_{n+1}/a_n|
  ρ < 1 → 绝对收敛   ρ > 1 → 发散   ρ = 1 → 失效，换 Step 3

Step 3：根值法（柯西判别法）
  触发信号：通项形如 (f(n))^n
  ρ = lim_{n→∞} n√|a_n|
  ρ < 1 → 绝对收敛   ρ > 1 → 发散   ρ = 1 → 失效，换 Step 4

Step 4：比较法
  三大参照标准：
    ① p-级数 Σ n^{-p}：p > 1 收敛，p ≤ 1 发散
    ② 等比级数 Σ q^n：|q| < 1 收敛，|q| ≥ 1 发散
    ③ 调和级数 Σ 1/n：发散
  极限比较：lim(a_n/b_n) = L ∈ (0,+∞) → 与 Σb_n 同敛散
  积分判别：若 a_n = f(n)，f 在 [1,∞) 上连续单调递减正值
              → Σa_n 与 ∫_1^∞ f(x)dx 同敛散

Step 5：Leibniz 判别（交错级数）
  条件：b_n > 0，b_n 单调递减，lim b_n = 0
  结论：Σ(-1)^n b_n 条件收敛
  误差估计：|S - S_n| ≤ b_{n+1}
```

### 常见标准级数速查

| 级数类型 | 收敛性 | 说明 |
|:---:|:---:|:---|
| $\sum n^{-p}$（$p$-级数） | $p>1$ 收敛，$p\leq1$ 发散 | $p=1$ 为调和级数（发散）|
| $\sum q^n$（等比） | $|q|<1$ 收敛，$|q|\geq1$ 发散 | — |
| $\sum \dfrac{1}{n\ln n}$ | 发散 | 积分判别：$\int_2^\infty\frac{dx}{x\ln x}=+\infty$ |
| $\sum \dfrac{1}{n(\ln n)^2}$ | 收敛 | 积分判别：$\int_2^\infty\frac{dx}{x(\ln x)^2}<+\infty$ |
| $\sum \dfrac{n!}{n^n}$ | 收敛 | 比值法 $\to 1/e<1$ |
| $\sum \dfrac{(-1)^n}{\sqrt{n}}$ | 条件收敛 | Leibniz 判别；$\sum 1/\sqrt{n}$ 发散（非绝对收敛）|

---

## 18. 多元链式法则 + 多元积分变换公式表

> 来自 toolkit/07–08。

### 18.1 多元链式法则

**一般形式**（$z=f(u_1,\ldots,u_m)$，每个 $u_i=u_i(x_1,\ldots,x_n)$）：

$$\frac{\partial z}{\partial x_j}=\sum_{i=1}^{m}\frac{\partial z}{\partial u_i}\cdot\frac{\partial u_i}{\partial x_j}, \quad j=1,\ldots,n$$

**矩阵形式**（Jacobian 连乘）：$J_{z,\mathbf{x}}=J_{z,\mathbf{u}}\cdot J_{\mathbf{u},\mathbf{x}}$，其中 $(J_{f})_{ij}=\partial f_i/\partial x_j$。

**常见特殊情形**

| 依赖结构 | 链式结果 |
|:---|:---|
| $z=f(u)$，$u=g(x,y)$ | $z_x=f'(u)\cdot g_x$，$z_y=f'(u)\cdot g_y$ |
| $z=f(u,v)$，$u=u(x)$，$v=v(x)$ | $\dfrac{dz}{dx}=f_u u'+f_v v'$ |
| $z=f(x,y)$，$y=g(x)$ | $\dfrac{dz}{dx}=f_x+f_y\cdot g'(x)$（全导数）|
| 隐函数 $F(x,y)=0$ | $\dfrac{dy}{dx}=-\dfrac{F_x}{F_y}$（$F_y\neq0$）|
| 隐函数 $F(x,y,z)=0$ | $z_x=-\dfrac{F_x}{F_z}$，$z_y=-\dfrac{F_y}{F_z}$ |

### 18.2 多元积分变换公式表

**一般换元公式（二重积分）**：

$$\iint_D f(x,y)\,dA = \iint_{D'} f\!\left(x(u,v),\,y(u,v)\right)\left|\frac{\partial(x,y)}{\partial(u,v)}\right|du\,dv$$

**常用坐标变换汇总**

| 变换 | 公式 | Jacobian $|J|$ | 面积/体积元 |
|:---:|:---|:---:|:---:|
| **极坐标**（$\mathbb{R}^2$） | $x=r\cos\theta$，$y=r\sin\theta$ | $r$ | $dA=r\,dr\,d\theta$ |
| **柱坐标**（$\mathbb{R}^3$） | $x=r\cos\theta$，$y=r\sin\theta$，$z=z$ | $r$ | $dV=r\,dr\,d\theta\,dz$ |
| **球坐标**（$\mathbb{R}^3$） | $x=\rho\sin\varphi\cos\theta$，$y=\rho\sin\varphi\sin\theta$，$z=\rho\cos\varphi$ | $\rho^2\sin\varphi$ | $dV=\rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$ |

**积分限约定**（球坐标）：$\rho\geq0$，$\varphi\in[0,\pi]$（与 $z$ 轴的夹角），$\theta\in[0,2\pi)$（方位角）。

**Fubini 定理**（积分次序交换）：在被积函数连续、积分区域有界的条件下：

$$\iint_D f(x,y)\,dA=\int_a^b\!\int_{g_1(x)}^{g_2(x)}f(x,y)\,dy\,dx=\int_c^d\!\int_{h_1(y)}^{h_2(y)}f(x,y)\,dx\,dy$$

---

## 19. 一阶 / 二阶 ODE 速查表

> 来自 toolkit/09。

### 19.1 一阶 ODE 5 类识别与求解

| 类型 | 识别特征 | 标准解法 | 通解形式 |
|:---:|:---|:---|:---|
| **可分离变量** | $y'=f(x)g(y)$ | $\dfrac{dy}{g(y)}=f(x)\,dx$，两边积分 | $G(y)=F(x)+C$ |
| **齐次方程** | $y'=\varphi(y/x)$ | 令 $u=y/x$，$y'=u+xu'$，化为可分离 | 解 $u$ 后还原 $y=ux$ |
| **一阶线性** | $y'+p(x)y=q(x)$ | 积分因子 $\mu=e^{\int p\,dx}$，$(\mu y)'=\mu q$ | $y=e^{-\int p\,dx}\!\left[\int q\,e^{\int p\,dx}\,dx+C\right]$ |
| **Bernoulli** | $y'+p(x)y=q(x)y^n$，$n\neq0,1$ | 令 $v=y^{1-n}$，$v'+(1-n)pv=(1-n)q$ | 以 $v$ 解线性方程，最后换回 $y$ |
| **恰当方程** | $P\,dx+Q\,dy=0$，$P_y=Q_x$ | 求势函数 $u$：$u_x=P$，$u_y=Q$ | $u(x,y)=C$ |

**注意**：每类 ODE 解出时，检查常数解（$g(y)=0$ 或 $y=0$ 等）是否已包含在通解中，避免遗漏奇解。

### 19.2 二阶常系数线性 ODE：特征根 3 种情形

**齐次方程** $y''+py'+qy=0$，特征方程 $r^2+pr+q=0$，$\Delta=p^2-4q$：

| 特征根类型 | 判别条件 | 通解 |
|:---:|:---:|:---|
| **两不同实根** | $\Delta>0$，$r_1\neq r_2$ | $y=C_1e^{r_1x}+C_2e^{r_2x}$ |
| **重实根** | $\Delta=0$，$r_1=r_2=r$ | $y=(C_1+C_2x)e^{rx}$ |
| **共轭复根** | $\Delta<0$，$r=\alpha\pm\beta i$ | $y=e^{\alpha x}(C_1\cos\beta x+C_2\sin\beta x)$ |

**非齐次方程** $y''+py'+qy=f(x)$ 的特解设法（待定系数法）：

| $f(x)$ 形式 | 特解形式 $y^*$ | $k$ 值 |
|:---|:---|:---:|
| $P_m(x)e^{\lambda x}$ | $x^k Q_m(x)e^{\lambda x}$（$Q_m$ 为 $m$ 次多项式）| $\lambda$ 作为特征根的重数（0/1/2）|
| $e^{\lambda x}[A_l\cos\omega x+B_n\sin\omega x]$ | $x^k e^{\lambda x}[R_m\cos\omega x+S_m\sin\omega x]$，$m=\max(l,n)$ | $\lambda+\omega i$ 是否为特征根（0/1）|

**全解** = 齐次通解 $y_H$ + 非齐次特解 $y^*$：$y=y_H+y^*$。

---

## 20. 凸性判定 + KKT 简表

> 来自 toolkit/10。

### 20.1 凸函数判定速查

**单变量**：

| 判据 | 条件 | 结论 |
|:---:|:---|:---:|
| 二阶导（充分条件）| $f''(x)>0$ 在区间 $(a,b)$ 上 | $f$ 在 $(a,b)$ 上严格凸 |
| 二阶导（充分条件）| $f''(x)\geq0$ 在 $(a,b)$ 上 | $f$ 在 $(a,b)$ 上凸（允许"平段"）|
| 定义（充要条件）| $f(tx_1+(1-t)x_2)\leq tf(x_1)+(1-t)f(x_2)$，$t\in[0,1]$ | $f$ 是凸函数 |

**多变量**（$f:\mathbb{R}^n\to\mathbb{R}$，$f\in C^2$）：

| Hessian 特性 | 条件 | 结论 |
|:---:|:---:|:---:|
| 半正定 | $\nabla^2f(\mathbf{x})\succcurlyeq0$（对所有 $\mathbf{x}$）| $f$ 凸 |
| 正定 | $\nabla^2f(\mathbf{x})\succ0$（对所有 $\mathbf{x}$）| $f$ 严格凸 |
| 半负定 | $\nabla^2f(\mathbf{x})\preccurlyeq0$ | $f$ 凹 |
| 不定 | 特征值有正有负 | 非凸非凹（可能有鞍点）|

**凸函数常见例子**：$e^x$，$x^2$，$-\ln x$（$x>0$），$\|\mathbf{x}\|^p$（$p\geq1$），$\mathbf{x}^TA\mathbf{x}$（$A\succcurlyeq0$）。

**切线放缩**：$f$ 凸 $\Rightarrow$ $f(x)\geq f(a)+f'(a)(x-a)$（切线在图像下方）；这是 Jensen 不等式的单点形式。

### 20.2 Lagrange 乘子法与 KKT 简表

**无约束极值**（必要条件）：$\nabla f(\mathbf{x}^*)=\mathbf{0}$；充分判别：Hessian 正定 → 极小；负定 → 极大；不定 → 鞍点。

**等式约束极值**（$h_i(\mathbf{x})=0$，$i=1,\ldots,p$）：

$$\nabla f(\mathbf{x}^*)=\sum_{i=1}^p\lambda_i\nabla h_i(\mathbf{x}^*), \quad h_i(\mathbf{x}^*)=0$$

**KKT 条件**（等式约束 $h_i=0$ + 不等式约束 $g_j\leq0$，$j=1,\ldots,m$）：

| KKT 条件 | 公式 |
|:---|:---|
| **驻点条件** | $\nabla f(\mathbf{x}^*)+\displaystyle\sum_{i}\lambda_i\nabla h_i(\mathbf{x}^*)+\sum_{j}\mu_j\nabla g_j(\mathbf{x}^*)=\mathbf{0}$ |
| **原可行性** | $h_i(\mathbf{x}^*)=0$，$g_j(\mathbf{x}^*)\leq0$ |
| **对偶可行性** | $\mu_j\geq0$ |
| **互补松弛** | $\mu_j g_j(\mathbf{x}^*)=0$（$g_j<0$ 则 $\mu_j=0$；$\mu_j>0$ 则 $g_j=0$）|

**凸问题的 KKT**：当 $f,g_j$ 均为凸函数、$h_i$ 为仿射时，KKT 条件既是必要条件也是充分条件。

---

## 21. AI 微积分关键公式表

> 来自 toolkit/12，对接深度学习 / AI 工程应用场景。

### 21.1 梯度下降族

| 方法 | 更新规则 | 说明 |
|:---:|:---|:---|
| **梯度下降（GD）** | $\theta\leftarrow\theta-\eta\nabla L(\theta)$ | $\eta$：学习率；沿负梯度方向（最速下降）|
| **SGD** | $\theta\leftarrow\theta-\eta\nabla L_i(\theta)$ | $i$ 为随机选取的单样本/小批量 |
| **动量法** | $v\leftarrow\beta v-\eta\nabla L$；$\theta\leftarrow\theta+v$ | $\beta\in[0,1)$：历史梯度的指数衰减 |
| **Adam** | $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$；$v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$；$\theta\leftarrow\theta-\eta\dfrac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$ | 自适应步长；$\hat m_t=m_t/(1-\beta_1^t)$，$\hat v_t=v_t/(1-\beta_2^t)$ 为偏差修正 |

**梯度下降收敛保证**（凸 $L$，梯度 $\beta$-Lipschitz，$\eta\leq1/\beta$）：

$$L(\theta_{t+1})\leq L(\theta_t)-\frac{\eta}{2}\|\nabla L(\theta_t)\|^2$$

### 21.2 反向传播关键公式

设第 $k$ 层：$\mathbf{a}_k=W_k\mathbf{z}_{k-1}+\mathbf{b}_k$，$\mathbf{z}_k=\sigma(\mathbf{a}_k)$（激活函数 $\sigma$ 逐元素作用）：

$$\frac{\partial L}{\partial W_k}=\delta_k\cdot\mathbf{z}_{k-1}^T, \quad \frac{\partial L}{\partial\mathbf{b}_k}=\delta_k$$

$$\delta_k\triangleq\frac{\partial L}{\partial\mathbf{a}_k}, \quad \delta_{k-1}=\left(W_k^T\delta_k\right)\odot\sigma'(\mathbf{a}_{k-1})$$

其中 $\odot$ 为逐元素乘法（Hadamard 积）；$\delta_k$ 从最后一层反向传播到第一层。

### 21.3 Hessian 矩阵速查

$$H=\nabla^2 L\in\mathbb{R}^{n\times n}, \quad H_{ij}=\frac{\partial^2 L}{\partial\theta_i\partial\theta_j}$$

| 应用 | 公式 / 结论 |
|:---|:---|
| **极值判别** | $H\succ0$ → 极小；$H\prec0$ → 极大；$H$ 不定 → 鞍点 |
| **最优步长上界** | $\eta^*=1/\lambda_{\max}(H)$（最大特征值的倒数）|
| **Newton 法** | $\theta\leftarrow\theta-H^{-1}\nabla L$（局部二次收敛速率）|
| **收敛速度** | 条件数 $\kappa(H)=\lambda_{\max}/\lambda_{\min}$ 越大，GD 收敛越慢（"病态"）|
| **Adam 近似** | $v_t\approx\text{diag}(H)^2$，通过对角 Hessian 近似实现自适应步长 |

### 21.4 KL 散度公式速查

$$\mathrm{KL}(p\|q)=\int p(x)\ln\frac{p(x)}{q(x)}\,dx\geq0$$

非负性由 Jensen 不等式保证：$-\ln$ 是严格凸函数，$\mathrm{KL}(p\|q)=\mathbb{E}_p[-\ln(q/p)]\geq-\ln\mathbb{E}_p[q/p]=0$。

| 分布对 | KL 散度闭合公式 |
|:---:|:---|
| **一维高斯** $\mathcal{N}(\mu_1,\sigma_1^2)\|\mathcal{N}(\mu_2,\sigma_2^2)$ | $\ln\dfrac{\sigma_2}{\sigma_1}+\dfrac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\dfrac{1}{2}$ |
| **多元高斯** $\mathcal{N}(\mu_1,\Sigma_1)\|\mathcal{N}(\mu_2,\Sigma_2)$ | $\dfrac{1}{2}\!\left[\mathrm{tr}(\Sigma_2^{-1}\Sigma_1)+(\mu_1-\mu_2)^T\Sigma_2^{-1}(\mu_1-\mu_2)-n+\ln\dfrac{|\Sigma_2|}{|\Sigma_1|}\right]$ |
| **离散分布** | $\displaystyle\sum_x p(x)\ln\dfrac{p(x)}{q(x)}$（约定 $0\ln 0=0$）|

**KL 散度 vs 交叉熵**：$H(p,q)=H(p)+\mathrm{KL}(p\|q)$，其中 $H(p)=-\mathbb{E}_p[\ln p]$ 为熵。训练时真实分布 $p$ 固定，最小化交叉熵等价于最小化 $\mathrm{KL}(p\|q)$。
