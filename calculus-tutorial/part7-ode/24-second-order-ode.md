# 第24章 二阶线性微分方程

## 学习目标

通过本章学习，你将能够：

- 理解二阶线性微分方程的结构，区分齐次与非齐次方程
- 掌握解的叠加原理和通解的结构定理
- 熟练运用特征方程法求解常系数齐次方程
- 掌握待定系数法求解常系数非齐次方程的特解
- 能够将弹簧振动和RLC电路问题转化为二阶微分方程求解

---

## 24.1 二阶线性方程的结构

### 24.1.1 齐次方程与非齐次方程

**定义**：二阶线性微分方程的一般形式为

$$y'' + P(x)y' + Q(x)y = f(x)$$

- 当 $f(x) \equiv 0$ 时，称为**二阶齐次线性方程**：$y'' + P(x)y' + Q(x)y = 0$
- 当 $f(x) \not\equiv 0$ 时，称为**二阶非齐次线性方程**

**术语**：与齐次方程 $y'' + P(x)y' + Q(x)y = 0$ 对应的非齐次方程称为其**对应的非齐次方程**，反之亦然。

### 24.1.2 解的叠加原理

**定理**（齐次方程解的叠加原理）：若 $y_1(x)$ 和 $y_2(x)$ 都是齐次方程 $y'' + P(x)y' + Q(x)y = 0$ 的解，则对任意常数 $C_1, C_2$，

$$y = C_1 y_1(x) + C_2 y_2(x)$$

也是该齐次方程的解。

**证明**：将 $y = C_1 y_1 + C_2 y_2$ 代入方程左边：

$$y'' + Py' + Qy = (C_1 y_1'' + C_2 y_2'') + P(C_1 y_1' + C_2 y_2') + Q(C_1 y_1 + C_2 y_2)$$

$$= C_1(y_1'' + Py_1' + Qy_1) + C_2(y_2'' + Py_2' + Qy_2) = C_1 \cdot 0 + C_2 \cdot 0 = 0$$

故 $y = C_1 y_1 + C_2 y_2$ 是方程的解。 $\square$

### 24.1.3 线性无关与Wronskian行列式

**定义**（线性无关）：两个函数 $y_1(x)$ 和 $y_2(x)$ 在区间 $I$ 上**线性无关**，如果 $\dfrac{y_1(x)}{y_2(x)} \neq$ 常数（在 $y_2 \neq 0$ 时）。

等价地，$y_1$ 和 $y_2$ 线性无关当且仅当：若 $C_1 y_1 + C_2 y_2 \equiv 0$，则必有 $C_1 = C_2 = 0$。

**定义**（Wronskian行列式）：对于两个可微函数 $y_1(x)$ 和 $y_2(x)$，定义其 **Wronskian行列式**为

$$W(y_1, y_2) = \begin{vmatrix} y_1 & y_2 \\ y_1' & y_2' \end{vmatrix} = y_1 y_2' - y_2 y_1'$$

**定理**：设 $y_1(x)$ 和 $y_2(x)$ 是齐次方程 $y'' + P(x)y' + Q(x)y = 0$ 的两个解，则：

1. 若 $W(y_1, y_2) \neq 0$（在某一点，从而在整个区间），则 $y_1, y_2$ 线性无关
2. 若 $y_1, y_2$ 线性无关，则 $W(y_1, y_2) \neq 0$

### 24.1.4 通解的结构定理

**定理**（齐次方程通解结构）：设 $y_1(x)$ 和 $y_2(x)$ 是齐次方程 $y'' + P(x)y' + Q(x)y = 0$ 的两个线性无关的解，则该方程的**通解**为

$$y = C_1 y_1(x) + C_2 y_2(x)$$

其中 $C_1, C_2$ 是任意常数。

**定理**（非齐次方程通解结构）：设 $y^*$ 是非齐次方程 $y'' + P(x)y' + Q(x)y = f(x)$ 的一个**特解**，$Y = C_1 y_1 + C_2 y_2$ 是对应齐次方程的**通解**，则非齐次方程的**通解**为

$$y = Y + y^* = C_1 y_1(x) + C_2 y_2(x) + y^*(x)$$

> **例题 24.1** 验证 $y_1 = e^x$ 和 $y_2 = e^{-x}$ 是方程 $y'' - y = 0$ 的两个线性无关解，并写出通解。

**解**：验证 $y_1 = e^x$ 是解：$y_1'' - y_1 = e^x - e^x = 0$。 ✓

验证 $y_2 = e^{-x}$ 是解：$y_2'' - y_2 = e^{-x} - e^{-x} = 0$。 ✓

计算Wronskian行列式：

$$W(y_1, y_2) = \begin{vmatrix} e^x & e^{-x} \\ e^x & -e^{-x} \end{vmatrix} = e^x \cdot (-e^{-x}) - e^{-x} \cdot e^x = -1 - 1 = -2 \neq 0$$

故 $y_1, y_2$ 线性无关，通解为 $y = C_1 e^x + C_2 e^{-x}$。 $\square$

---

## 24.2 常系数齐次方程

### 24.2.1 特征方程法

考虑**常系数齐次方程**：

$$y'' + py' + qy = 0$$

其中 $p, q$ 是常数。

**核心思想**：设 $y = e^{rx}$ 是方程的解，代入得：

$$r^2 e^{rx} + pr e^{rx} + q e^{rx} = 0$$

$$e^{rx}(r^2 + pr + q) = 0$$

由于 $e^{rx} \neq 0$，必有

$$\boxed{r^2 + pr + q = 0}$$

这称为原微分方程的**特征方程**，其根称为**特征根**。

### 24.2.2 三种情况

设特征方程的判别式 $\Delta = p^2 - 4q$。

**情况一：$\Delta > 0$，两个不相等实根 $r_1 \neq r_2$**

通解为：

$$\boxed{y = C_1 e^{r_1 x} + C_2 e^{r_2 x}}$$

> **例题 24.2** 求方程 $y'' - 5y' + 6y = 0$ 的通解。

**解**：特征方程为 $r^2 - 5r + 6 = 0$。

分解因式：$(r - 2)(r - 3) = 0$，得 $r_1 = 2$，$r_2 = 3$。

通解为 $y = C_1 e^{2x} + C_2 e^{3x}$。 $\square$

**情况二：$\Delta = 0$，两个相等实根 $r_1 = r_2 = r$**

此时只有一个解 $y_1 = e^{rx}$。需要找第二个线性无关解。

可以验证 $y_2 = xe^{rx}$ 也是方程的解（可用降阶法或直接代入验证）。

通解为：

$$\boxed{y = (C_1 + C_2 x) e^{rx}}$$

> **例题 24.3** 求方程 $y'' - 4y' + 4y = 0$ 的通解。

**解**：特征方程为 $r^2 - 4r + 4 = 0$，即 $(r - 2)^2 = 0$。

重根 $r = 2$。

通解为 $y = (C_1 + C_2 x) e^{2x}$。 $\square$

**情况三：$\Delta < 0$，共轭复根 $r_{1,2} = \alpha \pm \beta i$**

其中 $\alpha = -\dfrac{p}{2}$，$\beta = \dfrac{\sqrt{4q - p^2}}{2}$。

利用Euler公式 $e^{i\theta} = \cos\theta + i\sin\theta$，可得两个实值线性无关解：

$$y_1 = e^{\alpha x} \cos \beta x, \quad y_2 = e^{\alpha x} \sin \beta x$$

通解为：

$$\boxed{y = e^{\alpha x}(C_1 \cos \beta x + C_2 \sin \beta x)}$$

> **例题 24.4** 求方程 $y'' + 2y' + 5y = 0$ 的通解。

**解**：特征方程为 $r^2 + 2r + 5 = 0$。

$$r = \frac{-2 \pm \sqrt{4 - 20}}{2} = \frac{-2 \pm \sqrt{-16}}{2} = \frac{-2 \pm 4i}{2} = -1 \pm 2i$$

即 $\alpha = -1$，$\beta = 2$。

通解为 $y = e^{-x}(C_1 \cos 2x + C_2 \sin 2x)$。 $\square$

> **例题 24.5** 求初值问题 $\begin{cases} y'' + y = 0 \\ y(0) = 1, \ y'(0) = 0 \end{cases}$ 的解。

**解**：特征方程 $r^2 + 1 = 0$，得 $r = \pm i$（即 $\alpha = 0$，$\beta = 1$）。

通解为 $y = C_1 \cos x + C_2 \sin x$。

由 $y(0) = 1$：$C_1 = 1$。

$y' = -C_1 \sin x + C_2 \cos x$，由 $y'(0) = 0$：$C_2 = 0$。

特解为 $y = \cos x$。 $\square$

---

## 24.3 常系数非齐次方程

### 24.3.1 待定系数法

考虑**常系数非齐次方程**：

$$y'' + py' + qy = f(x)$$

根据通解结构定理，只需求出一个特解 $y^*$，再加上齐次方程的通解即可。

**待定系数法**的基本思想：根据 $f(x)$ 的形式，猜测特解 $y^*$ 的形式，代入方程确定待定系数。

### 24.3.2 类型一：$f(x) = e^{\lambda x} P_m(x)$

其中 $P_m(x)$ 是 $m$ 次多项式。

**特解形式**：设

$$y^* = x^k e^{\lambda x} Q_m(x)$$

其中 $Q_m(x)$ 是待定的 $m$ 次多项式，$k$ 的取值为：

- $k = 0$：若 $\lambda$ 不是特征根
- $k = 1$：若 $\lambda$ 是单特征根
- $k = 2$：若 $\lambda$ 是重特征根

> **例题 24.6** 求方程 $y'' - 3y' + 2y = e^{3x}$ 的一个特解。

**解**：特征方程 $r^2 - 3r + 2 = 0$，得 $r_1 = 1$，$r_2 = 2$。

$f(x) = e^{3x}$，这里 $\lambda = 3$，$P_m(x) = 1$（$m = 0$）。

由于 $\lambda = 3$ 不是特征根，取 $k = 0$。

设 $y^* = Ae^{3x}$。

代入方程：$9Ae^{3x} - 9Ae^{3x} + 2Ae^{3x} = e^{3x}$。

$2A = 1$，故 $A = \dfrac{1}{2}$。

特解为 $y^* = \dfrac{1}{2}e^{3x}$。 $\square$

> **例题 24.7** 求方程 $y'' - 2y' + y = xe^x$ 的一个特解。

**解**：特征方程 $r^2 - 2r + 1 = 0$，得重根 $r = 1$。

$f(x) = xe^x$，这里 $\lambda = 1$，$P_m(x) = x$（$m = 1$）。

由于 $\lambda = 1$ 是重特征根，取 $k = 2$。

设 $y^* = x^2(Ax + B)e^x = (Ax^3 + Bx^2)e^x$。

计算 $y^*{}'$ 和 $y^*{}''$（过程较繁，此处省略），代入方程后比较系数：

$6A = 1$，故 $A = \dfrac{1}{6}$，$B = 0$。

特解为 $y^* = \dfrac{1}{6}x^3 e^x$。 $\square$

### 24.3.3 类型二：$f(x) = e^{\alpha x}[P(x)\cos\beta x + Q(x)\sin\beta x]$

其中 $P(x), Q(x)$ 是多项式，设其最高次数为 $m$。

**特解形式**：设

$$y^* = x^k e^{\alpha x}[R_m(x)\cos\beta x + S_m(x)\sin\beta x]$$

其中 $R_m(x), S_m(x)$ 是待定的 $m$ 次多项式，$k$ 的取值为：

- $k = 0$：若 $\alpha + \beta i$ 不是特征根
- $k = 1$：若 $\alpha + \beta i$ 是特征根

> **例题 24.8** 求方程 $y'' + y = \cos x$ 的一个特解。

**解**：特征方程 $r^2 + 1 = 0$，得 $r = \pm i$。

$f(x) = \cos x$，这里 $\alpha = 0$，$\beta = 1$，$P(x) = 1$，$Q(x) = 0$。

由于 $\alpha + \beta i = i$ 是特征根，取 $k = 1$。

设 $y^* = x(A\cos x + B\sin x)$。

$y^*{}' = (A\cos x + B\sin x) + x(-A\sin x + B\cos x)$

$y^*{}'' = -2A\sin x + 2B\cos x - x(A\cos x + B\sin x)$

代入 $y'' + y = \cos x$：

$-2A\sin x + 2B\cos x = \cos x$

比较系数：$-2A = 0$，$2B = 1$，故 $A = 0$，$B = \dfrac{1}{2}$。

特解为 $y^* = \dfrac{x}{2}\sin x$。 $\square$

> **例题 24.9** 求方程 $y'' + 4y = \sin 2x$ 的通解。

**解**：特征方程 $r^2 + 4 = 0$，得 $r = \pm 2i$，齐次通解为 $Y = C_1\cos 2x + C_2\sin 2x$。

$f(x) = \sin 2x$，$\alpha = 0$，$\beta = 2$，$0 + 2i = 2i$ 是特征根，取 $k = 1$。

设 $y^* = x(A\cos 2x + B\sin 2x)$。

$y^*{}' = (A\cos 2x + B\sin 2x) + x(-2A\sin 2x + 2B\cos 2x)$

$y^*{}'' = -4A\sin 2x + 4B\cos 2x - 4x(A\cos 2x + B\sin 2x)$

代入 $y'' + 4y = \sin 2x$：

$-4A\sin 2x + 4B\cos 2x = \sin 2x$

比较系数：$-4A = 1$，$4B = 0$，故 $A = -\dfrac{1}{4}$，$B = 0$。

特解为 $y^* = -\dfrac{x}{4}\cos 2x$。

通解为 $y = C_1\cos 2x + C_2\sin 2x - \dfrac{x}{4}\cos 2x$。 $\square$

### 24.3.4 常数变易法

待定系数法只适用于 $f(x)$ 具有特殊形式的情况。对于一般的非齐次方程

$$y'' + P(x)y' + Q(x)y = f(x)$$

可以使用**常数变易法**（Variation of Parameters）来求特解。

**基本思想**：设齐次方程 $y'' + P(x)y' + Q(x)y = 0$ 的两个线性无关解为 $y_1(x)$ 和 $y_2(x)$，齐次通解为 $C_1 y_1 + C_2 y_2$。将常数 $C_1, C_2$ "变易"为未知函数 $C_1(x), C_2(x)$，设非齐次方程的特解为

$$y^* = C_1(x)\,y_1(x) + C_2(x)\,y_2(x)$$

**推导**：对 $y^*$ 求导：

$$y^*{}' = C_1' y_1 + C_1 y_1' + C_2' y_2 + C_2 y_2'$$

为简化计算，**附加条件**：

$$C_1' y_1 + C_2' y_2 = 0 \quad \cdots (*)$$

于是 $y^*{}' = C_1 y_1' + C_2 y_2'$。再求导：

$$y^*{}'' = C_1' y_1' + C_1 y_1'' + C_2' y_2' + C_2 y_2''$$

将 $y^*, y^*{}', y^*{}''$ 代入原方程 $y'' + Py' + Qy = f(x)$，利用 $y_1, y_2$ 分别满足齐次方程的条件消去含 $C_1, C_2$（不带撇）的项，得到：

$$C_1' y_1' + C_2' y_2' = f(x) \quad \cdots (**)$$

联立 $(*)$ 和 $(**)$，得到关于 $C_1'(x)$ 和 $C_2'(x)$ 的方程组：

$$\begin{cases} C_1' y_1 + C_2' y_2 = 0 \\ C_1' y_1' + C_2' y_2' = f(x) \end{cases}$$

由 Cramer 法则，其系数行列式恰好是 Wronskian 行列式 $W = y_1 y_2' - y_2 y_1' \neq 0$，解为：

$$C_1'(x) = -\frac{y_2(x)\,f(x)}{W(x)}, \quad C_2'(x) = \frac{y_1(x)\,f(x)}{W(x)}$$

分别积分即可求得 $C_1(x)$ 和 $C_2(x)$，从而得到特解。

> **例题 24.12** 用常数变易法求方程 $y'' + y = \dfrac{1}{\cos x}$ 的一个特解。

**解**：齐次方程 $y'' + y = 0$ 的通解为 $Y = C_1\cos x + C_2\sin x$。

取 $y_1 = \cos x$，$y_2 = \sin x$，则 $W = \cos x \cdot \cos x - \sin x \cdot (-\sin x) = 1$。

由公式：

$$C_1'(x) = -\frac{\sin x \cdot \frac{1}{\cos x}}{1} = -\tan x, \quad C_2'(x) = \frac{\cos x \cdot \frac{1}{\cos x}}{1} = 1$$

积分：

$$C_1(x) = -\int \tan x\,dx = \ln|\cos x|, \quad C_2(x) = \int 1\,dx = x$$

特解为 $y^* = \cos x \cdot \ln|\cos x| + x\sin x$。 $\square$

---

## 24.4 应用举例

### 24.4.1 弹簧振动问题

考虑一个质量为 $m$ 的物体悬挂在弹性系数为 $k$ 的弹簧下端。设 $x(t)$ 为物体相对于平衡位置的位移（向下为正）。

**无阻尼自由振动**：由Hooke定律和Newton第二定律：

$$m\frac{d^2x}{dt^2} = -kx$$

即

$$\frac{d^2x}{dt^2} + \omega^2 x = 0, \quad \omega = \sqrt{\frac{k}{m}}$$

通解为 $x = C_1\cos\omega t + C_2\sin\omega t = A\cos(\omega t - \varphi)$，这是**简谐振动**。

**有阻尼自由振动**：若存在与速度成正比的阻力 $-c\dfrac{dx}{dt}$：

$$m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = 0$$

**受迫振动**：若还有外力 $F(t) = F_0\cos\omega_0 t$：

$$m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = F_0\cos\omega_0 t$$

> **例题 24.10** 一弹簧振子，质量 $m = 1$ kg，弹性系数 $k = 4$ N/m，无阻尼。初始时物体在平衡位置下方 $0.1$ m 处静止释放。求运动规律。

**解**：方程为 $\dfrac{d^2x}{dt^2} + 4x = 0$，初始条件 $x(0) = 0.1$，$x'(0) = 0$。

特征方程 $r^2 + 4 = 0$，$r = \pm 2i$。

通解 $x = C_1\cos 2t + C_2\sin 2t$。

由 $x(0) = 0.1$：$C_1 = 0.1$。

$x' = -2C_1\sin 2t + 2C_2\cos 2t$，由 $x'(0) = 0$：$C_2 = 0$。

运动规律为 $x = 0.1\cos 2t$（单位：m），周期 $T = \pi$ s。 $\square$

### 24.4.2 RLC电路问题

在串联RLC电路中，设电容上的电荷为 $q(t)$，电流 $i = \dfrac{dq}{dt}$，外加电压为 $E(t)$。

由Kirchhoff电压定律：

$$L\frac{di}{dt} + Ri + \frac{q}{C} = E(t)$$

即

$$L\frac{d^2q}{dt^2} + R\frac{dq}{dt} + \frac{1}{C}q = E(t)$$

这是关于电荷 $q$ 的二阶线性微分方程。

> **例题 24.11** 一RLC串联电路，$L = 1$ H，$R = 2$ Ω，$C = 0.5$ F，外加电压 $E = 10$ V（直流）。初始时 $q(0) = 0$，$i(0) = 0$。求电荷 $q(t)$。

**解**：方程为 $\dfrac{d^2q}{dt^2} + 2\dfrac{dq}{dt} + 2q = 10$。

齐次方程的特征方程：$r^2 + 2r + 2 = 0$，$r = -1 \pm i$。

齐次通解：$Q = e^{-t}(C_1\cos t + C_2\sin t)$。

非齐次方程特解：设 $q^* = A$（常数），代入得 $2A = 10$，$A = 5$。

通解：$q = e^{-t}(C_1\cos t + C_2\sin t) + 5$。

由 $q(0) = 0$：$C_1 + 5 = 0$，$C_1 = -5$。

$q' = e^{-t}[(-C_1 + C_2)\cos t + (-C_1 - C_2)\sin t]$

由 $q'(0) = i(0) = 0$：$-C_1 + C_2 = 0$，$C_2 = C_1 = -5$。

$q(t) = e^{-t}(-5\cos t - 5\sin t) + 5 = 5[1 - e^{-t}(\cos t + \sin t)]$ 库仑。 $\square$

---

## 本章小结

1. **二阶线性方程的结构**：
   - 齐次方程 $y'' + Py' + Qy = 0$：通解为两个线性无关解的线性组合
   - 非齐次方程通解 $=$ 对应齐次方程通解 $+$ 特解
   - Wronskian行列式判断线性无关性

2. **常系数齐次方程** $y'' + py' + qy = 0$ 的特征方程法：

   | 判别式 | 特征根 | 通解形式 |
   |:---:|:---:|:---:|
   | $\Delta > 0$ | $r_1 \neq r_2$（实根） | $C_1 e^{r_1 x} + C_2 e^{r_2 x}$ |
   | $\Delta = 0$ | $r_1 = r_2 = r$（重根） | $(C_1 + C_2 x)e^{rx}$ |
   | $\Delta < 0$ | $r = \alpha \pm \beta i$（复根） | $e^{\alpha x}(C_1\cos\beta x + C_2\sin\beta x)$ |

3. **常系数非齐次方程的待定系数法**：
   - $f(x) = e^{\lambda x}P_m(x)$ 型：特解设为 $y^* = x^k e^{\lambda x}Q_m(x)$
   - $f(x) = e^{\alpha x}[P\cos\beta x + Q\sin\beta x]$ 型：特解设为 $y^* = x^k e^{\alpha x}[R_m\cos\beta x + S_m\sin\beta x]$
   - $k$ 值由 $\lambda$（或 $\alpha + \beta i$）是否为特征根决定

4. **应用**：
   - 弹簧振动：$m x'' + c x' + k x = F(t)$
   - RLC电路：$L q'' + R q' + \dfrac{1}{C}q = E(t)$

---

## 深度学习应用

### 概念回顾

二阶线性微分方程描述了包含加速度（二阶导数）的动态系统，如振动、电路等物理现象。其解的结构（齐次通解+特解）和特征方程法是求解的核心工具。

### 在深度学习中的应用

#### 1. 动量优化器的物理解释

带动量的梯度下降可以用二阶ODE描述。考虑优化问题 $\min_\theta \mathcal{L}(\theta)$：

**Heavy Ball 方法**（离散）：
$$\theta_{t+1} = \theta_t - \alpha \nabla \mathcal{L}(\theta_t) + \beta(\theta_t - \theta_{t-1})$$

**连续时间极限**（二阶ODE）：
$$\ddot{\theta} + \gamma \dot{\theta} + \nabla \mathcal{L}(\theta) = 0$$

这与有阻尼振动方程 $m\ddot{x} + c\dot{x} + kx = 0$ 结构相同！

- $\gamma$（阻尼系数）控制动量衰减
- $\nabla \mathcal{L}$（弹性力）驱动向最优点移动
- 欠阻尼 → 快速但可能振荡；过阻尼 → 稳定但缓慢

#### 2. Nesterov 加速梯度的 ODE 分析

Nesterov 加速梯度（NAG）的连续极限是：

$$\ddot{\theta} + \frac{3}{t}\dot{\theta} + \nabla \mathcal{L}(\theta) = 0$$

注意阻尼系数 $\frac{3}{t}$ 随时间减小！这解释了为什么 NAG 能达到 $O(1/t^2)$ 的加速收敛率（vs 普通梯度下降的 $O(1/t)$）。

#### 3. Physics-Informed Neural Networks (PINNs)

PINNs 将物理定律（通常是微分方程）作为约束嵌入神经网络训练：

对于二阶 ODE $y'' + py' + qy = f(x)$：

$$\mathcal{L}_{\text{physics}} = \|y''_\theta(x) + p \cdot y'_\theta(x) + q \cdot y_\theta(x) - f(x)\|^2$$

加上边界/初始条件损失：

$$\mathcal{L}_{\text{BC}} = \|y_\theta(x_0) - y_0\|^2 + \|y'_\theta(x_0) - v_0\|^2$$

总损失 $\mathcal{L} = \mathcal{L}_{\text{physics}} + \lambda \mathcal{L}_{\text{BC}}$

#### 4. 二阶 Neural ODE

扩展 Neural ODE 到二阶系统：

$$\ddot{h} = f(h, \dot{h}, t, \theta)$$

可以改写为一阶系统：
$$\frac{d}{dt}\begin{pmatrix} h \\ v \end{pmatrix} = \begin{pmatrix} v \\ f(h, v, t, \theta) \end{pmatrix}$$

这对建模物理系统（位置-速度）特别有效。

### 代码示例（Python/PyTorch）

```python
import torch
import torch.nn as nn
import numpy as np

# ========== 1. 动量优化器的 ODE 视角 ==========

class MomentumODEOptimizer:
    """
    用 ODE 视角实现动量优化器
    方程: θ'' + γθ' + ∇L = 0
    等价于: v' = -γv - ∇L, θ' = v
    """
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.gamma = (1 - momentum) / lr  # 阻尼系数
        self.velocities = [torch.zeros_like(p) for p in self.params]

    def step(self):
        dt = self.lr  # 时间步长
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            v = self.velocities[i]
            # Euler 更新: v' = -γv - ∇L
            v_new = v - dt * (self.gamma * v + p.grad)
            # θ' = v
            p.data = p.data + dt * v_new
            self.velocities[i] = v_new

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# ========== 2. PINN 求解二阶 ODE ==========

class PINN_SecondOrderODE(nn.Module):
    """
    用 PINN 求解: y'' + 2y' + 5y = 0, y(0)=1, y'(0)=0
    解析解: y = e^{-x}(cos(2x) + 0.5*sin(2x))
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

    def compute_derivatives(self, x):
        """计算 y, y', y'' """
        x = x.requires_grad_(True)
        y = self.forward(x)

        # 一阶导数 y'
        y_x = torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True
        )[0]

        # 二阶导数 y''
        y_xx = torch.autograd.grad(
            y_x, x, grad_outputs=torch.ones_like(y_x),
            create_graph=True
        )[0]

        return y, y_x, y_xx

    def physics_loss(self, x):
        """物理损失: y'' + 2y' + 5y = 0"""
        y, y_x, y_xx = self.compute_derivatives(x)
        residual = y_xx + 2 * y_x + 5 * y
        return torch.mean(residual ** 2)

    def boundary_loss(self):
        """边界条件: y(0)=1, y'(0)=0"""
        x0 = torch.tensor([[0.0]])
        y, y_x, _ = self.compute_derivatives(x0)
        loss_y0 = (y - 1.0) ** 2
        loss_yx0 = y_x ** 2
        return loss_y0 + loss_yx0

# 训练 PINN
def train_pinn():
    model = PINN_SecondOrderODE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(2000):
        optimizer.zero_grad()

        # 配点采样
        x_colloc = torch.rand(100, 1) * 5  # [0, 5] 区间

        # 总损失
        loss_phys = model.physics_loss(x_colloc)
        loss_bc = model.boundary_loss()
        loss = loss_phys + 10 * loss_bc

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Physics={loss_phys.item():.6f}, BC={loss_bc.item():.6f}")

    # 验证
    x_test = torch.linspace(0, 5, 100).reshape(-1, 1)
    y_pred = model(x_test).detach().numpy()
    y_exact = np.exp(-x_test.numpy()) * (np.cos(2*x_test.numpy()) + 0.5*np.sin(2*x_test.numpy()))

    error = np.mean(np.abs(y_pred - y_exact))
    print(f"平均绝对误差: {error:.6f}")

    return model

# ========== 3. 二阶 Neural ODE ==========

class SecondOrderNeuralODE(nn.Module):
    """
    二阶 Neural ODE: h'' = f(h, h', t)
    转化为一阶系统: [h, v]' = [v, f(h, v, t)]
    """
    def __init__(self, state_dim):
        super().__init__()
        # 输入: [h, v] 拼接, 输出: 加速度
        self.accel_net = nn.Sequential(
            nn.Linear(state_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, state_dim)
        )
        self.state_dim = state_dim

    def forward(self, t, state):
        """
        state = [h, v], shape: (batch, 2*state_dim)
        返回 d[h,v]/dt = [v, f(h,v)]
        """
        h = state[:, :self.state_dim]
        v = state[:, self.state_dim:]

        # 加速度由神经网络计算
        hv = torch.cat([h, v], dim=1)
        accel = self.accel_net(hv)

        # d[h,v]/dt = [v, accel]
        return torch.cat([v, accel], dim=1)

# 使用示例
print("=== 动量优化器演示 ===")
# 简单的二次损失
param = torch.tensor([5.0], requires_grad=True)
opt = MomentumODEOptimizer([param], lr=0.1, momentum=0.9)

for i in range(20):
    loss = (param - 1.0) ** 2  # 最优点在 1.0
    loss.backward()
    opt.step()
    opt.zero_grad()
    if i % 5 == 0:
        print(f"  迭代 {i}: param = {param.item():.4f}")

print("\n=== PINN 求解 y'' + 2y' + 5y = 0 ===")
# model = train_pinn()  # 取消注释以运行

print("\n=== 二阶 Neural ODE 结构 ===")
ode_func = SecondOrderNeuralODE(state_dim=8)
state0 = torch.randn(16, 16)  # [h0, v0]
dstate = ode_func(0, state0)
print(f"状态维度: {state0.shape}, 导数维度: {dstate.shape}")
```

### 延伸阅读

- Su, Boyd, Candes, "A Differential Equation for Modeling Nesterov's Accelerated Gradient Method" (JMLR 2016)
- Raissi et al., "Physics-Informed Neural Networks" (JCP 2019)
- Norcliffe et al., "On Second Order Behaviour in Augmented Neural ODEs" (NeurIPS 2020)
- Betancourt et al., "The Geometric Foundations of Hamiltonian Monte Carlo" (2017) - 与二阶ODE相关的采样方法

---

## 练习题

**1.** 求方程 $y'' - 4y' + 3y = 0$ 的通解。

**2.** 求方程 $y'' + 6y' + 9y = 0$ 的通解。

**3.** 求初值问题 $\begin{cases} y'' + 4y' + 13y = 0 \\ y(0) = 0, \ y'(0) = 6 \end{cases}$ 的解。

**4.** 求方程 $y'' - 2y' - 3y = e^{4x}$ 的通解。

**5.** 求方程 $y'' + 9y = 2\cos 3x$ 的通解。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.** 特征方程 $r^2 - 4r + 3 = 0$，$(r-1)(r-3) = 0$，得 $r_1 = 1$，$r_2 = 3$。

通解为 $y = C_1 e^x + C_2 e^{3x}$。

---

**2.** 特征方程 $r^2 + 6r + 9 = 0$，$(r+3)^2 = 0$，重根 $r = -3$。

通解为 $y = (C_1 + C_2 x)e^{-3x}$。

---

**3.** 特征方程 $r^2 + 4r + 13 = 0$，$r = \dfrac{-4 \pm \sqrt{16-52}}{2} = \dfrac{-4 \pm 6i}{2} = -2 \pm 3i$。

通解 $y = e^{-2x}(C_1\cos 3x + C_2\sin 3x)$。

由 $y(0) = 0$：$C_1 = 0$。

$y = C_2 e^{-2x}\sin 3x$，$y' = C_2 e^{-2x}(-2\sin 3x + 3\cos 3x)$。

由 $y'(0) = 6$：$3C_2 = 6$，$C_2 = 2$。

特解为 $y = 2e^{-2x}\sin 3x$。

---

**4.** 特征方程 $r^2 - 2r - 3 = 0$，$(r-3)(r+1) = 0$，$r_1 = 3$，$r_2 = -1$。

齐次通解 $Y = C_1 e^{3x} + C_2 e^{-x}$。

$f(x) = e^{4x}$，$\lambda = 4$ 不是特征根，取 $k = 0$，设 $y^* = Ae^{4x}$。

代入：$16Ae^{4x} - 8Ae^{4x} - 3Ae^{4x} = e^{4x}$，$5A = 1$，$A = \dfrac{1}{5}$。

通解为 $y = C_1 e^{3x} + C_2 e^{-x} + \dfrac{1}{5}e^{4x}$。

---

**5.** 特征方程 $r^2 + 9 = 0$，$r = \pm 3i$。

齐次通解 $Y = C_1\cos 3x + C_2\sin 3x$。

$f(x) = 2\cos 3x$，$\alpha = 0$，$\beta = 3$，$3i$ 是特征根，取 $k = 1$。

设 $y^* = x(A\cos 3x + B\sin 3x)$。

$y^*{}' = (A\cos 3x + B\sin 3x) + x(-3A\sin 3x + 3B\cos 3x)$

$y^*{}'' = -6A\sin 3x + 6B\cos 3x - 9x(A\cos 3x + B\sin 3x)$

代入 $y'' + 9y = 2\cos 3x$：

$-6A\sin 3x + 6B\cos 3x = 2\cos 3x$

比较系数：$-6A = 0$，$6B = 2$，得 $A = 0$，$B = \dfrac{1}{3}$。

特解 $y^* = \dfrac{x}{3}\sin 3x$。

通解为 $y = C_1\cos 3x + C_2\sin 3x + \dfrac{x}{3}\sin 3x$。

</details>
