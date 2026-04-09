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
| $\dfrac{1}{x}$ | $\ln |x| + C$ |
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
| $\dfrac{1}{\sqrt{x^2 \pm a^2}}$ | $\ln |x + \sqrt{x^2 \pm a^2}| + C$ |
| $\dfrac{1}{a^2 + x^2}$ | $\dfrac{1}{a}\arctan\dfrac{x}{a} + C$ |
| $\dfrac{1}{a^2 - x^2}$ | $\dfrac{1}{2a}\ln\left|\dfrac{a+x}{a-x}\right| + C$ |

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
| $\dfrac{1}{1-x}$ | $\displaystyle\sum_{n=0}^{\infty} x^n = 1 + x + x^2 + x^3 + \cdots$ | $(-1, 1)$ |

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

> 本速查表涵盖微积分核心公式，供复习和快速查阅使用。
