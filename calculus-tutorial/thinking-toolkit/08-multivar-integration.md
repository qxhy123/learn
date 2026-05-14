# 多元积分变换

> **一例速记**：计算 $\displaystyle\iint_D e^{-(x^2+y^2)}\,dA$，$D$ 是单位圆盘 $x^2 + y^2 \leq 1$。
> 看到被积函数含 $x^2 + y^2$、积分区域是圆 → 立刻换**极坐标**：$x = r\cos\theta$，$y = r\sin\theta$，$dA = r\,dr\,d\theta$。
> 区域变为 $0 \leq r \leq 1$，$0 \leq \theta \leq 2\pi$。
> $\displaystyle\iint_D e^{-(x^2+y^2)}\,dA = \int_0^{2\pi}\int_0^1 e^{-r^2} \cdot r\,dr\,d\theta = 2\pi \int_0^1 r e^{-r^2}\,dr = 2\pi \cdot \frac{1-e^{-1}}{2} = \pi(1 - e^{-1})$。
> **$x^2+y^2$ + 圆形区域 = 极坐标，记住 $dA = r\,dr\,d\theta$。**

---

## 一、为什么坐标变换是多元积分的核心技巧

一元积分换元法（$\int f(g(x))g'(x)\,dx = \int f(u)\,du$）让很多原本无法直接积分的被积函数变得可积。多元积分中，同样的思想以更强大的形式出现：**坐标系的选择决定了积分的难易程度**。

在直角坐标下写得一塌糊涂的积分，换成极坐标或球坐标后，积分限可能变成常数区间，被积函数可能化为简单的多项式乘以一个指数，整个计算干净利落。

掌握多元积分变换有三个关键：知道**什么时候换**（区域形状识别）、知道**换成什么**（坐标系选择）、以及知道**怎么换**（Jacobian 行列式和积分限的重写）。

---

## 二、坐标系选择：区域形状决定坐标

### 2.1 速查表

| 区域形状 | 首选坐标系 | 面积/体积元 |
|---|---|---|
| 矩形 / 长方体 | **直角坐标** $(x, y, z)$ | $dA = dx\,dy$；$dV = dx\,dy\,dz$ |
| 圆盘 / 圆环 / 扇形 | **极坐标** $(r, \theta)$ | $dA = r\,dr\,d\theta$ |
| 圆柱 / 圆锥（绕 $z$ 轴）| **柱坐标** $(r, \theta, z)$ | $dV = r\,dr\,d\theta\,dz$ |
| 球体 / 球壳 / 半球 | **球坐标** $(\rho, \varphi, \theta)$ | $dV = \rho^2 \sin\varphi\,d\rho\,d\varphi\,d\theta$ |

### 2.2 极坐标（二维）

**变换公式**：$x = r\cos\theta$，$y = r\sin\theta$（$r \geq 0$，$0 \leq \theta < 2\pi$）

**Jacobian**：$\dfrac{\partial(x,y)}{\partial(r,\theta)} = \begin{vmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{vmatrix} = r\cos^2\theta + r\sin^2\theta = r$

故 $dA = dx\,dy = r\,dr\,d\theta$（**不要忘记因子 $r$！**）

**触发信号**：
- 被积函数含 $x^2 + y^2$（可换成 $r^2$）
- 积分区域是圆、圆环、半圆、扇形（$r$ 的范围是常数或简单函数）

**典型区域对应**：

| 区域 $D$ | 极坐标范围 |
|---|---|
| 单位圆盘 $x^2+y^2 \leq 1$ | $0 \leq r \leq 1$，$0 \leq \theta \leq 2\pi$ |
| 上半单位圆盘 | $0 \leq r \leq 1$，$0 \leq \theta \leq \pi$ |
| 圆环 $1 \leq x^2+y^2 \leq 4$ | $1 \leq r \leq 2$，$0 \leq \theta \leq 2\pi$ |
| 扇形（第一象限内 $r \leq 1$）| $0 \leq r \leq 1$，$0 \leq \theta \leq \pi/2$ |
| 过原点的圆 $x^2+y^2 \leq ax$（$a>0$）| $0 \leq r \leq a\cos\theta$，$-\pi/2 \leq \theta \leq \pi/2$ |

### 2.3 柱坐标（三维）

**变换公式**：$x = r\cos\theta$，$y = r\sin\theta$，$z = z$

**Jacobian**：$dV = r\,dr\,d\theta\,dz$（与极坐标相同，$z$ 方向不变）

**触发信号**：
- 三维区域有绕 $z$ 轴的圆对称性
- 区域是圆柱体 $\{r \leq a, 0 \leq z \leq h\}$
- 区域是圆锥 $\{z \geq r, z \leq 1\}$（即 $z \geq \sqrt{x^2+y^2}$）

### 2.4 球坐标（三维）

**变换公式**：
$$x = \rho\sin\varphi\cos\theta, \quad y = \rho\sin\varphi\sin\theta, \quad z = \rho\cos\varphi$$

其中 $\rho \geq 0$（径向距离），$0 \leq \varphi \leq \pi$（极角，从 $z$ 轴量起），$0 \leq \theta < 2\pi$（方位角）。

**Jacobian**：$dV = \rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$

**记忆技巧**：$\rho^2\sin\varphi$ 中，$\rho^2$ 来自径向，$\sin\varphi$ 来自极角（$\varphi = 0$ 时 $\sin\varphi = 0$，即北极点处"面积"为 $0$，符合直觉）。

**触发信号**：
- 三维区域含 $x^2 + y^2 + z^2$（可换成 $\rho^2$）
- 区域是球体 $\{x^2+y^2+z^2 \leq R^2\}$
- 区域是球壳 $\{a^2 \leq x^2+y^2+z^2 \leq b^2\}$
- 被积函数只依赖 $\rho = \sqrt{x^2+y^2+z^2}$

**典型球坐标范围**：

| 区域 | 球坐标范围 |
|---|---|
| 球体 $\rho \leq R$ | $0 \leq \rho \leq R$，$0 \leq \varphi \leq \pi$，$0 \leq \theta \leq 2\pi$ |
| 上半球 $z \geq 0$ | $0 \leq \varphi \leq \pi/2$（其余同）|
| 锥体内的球（$\varphi \leq \varphi_0$）| $0 \leq \varphi \leq \varphi_0$（其余同）|

---

## 三、Jacobian 行列式：体积元变换的几何意义

### 3.1 一般变量替换公式

设变量替换 $\mathbf{x} = \mathbf{x}(\mathbf{u})$（即 $(x, y) \to (u, v)$ 或 $(x, y, z) \to (u, v, w)$），则

$$\iint_D f(x, y)\,dx\,dy = \iint_{D'} f(\mathbf{x}(\mathbf{u})) \left|\frac{\partial(x, y)}{\partial(u, v)}\right|\,du\,dv$$

其中 $\left|\dfrac{\partial(x, y)}{\partial(u, v)}\right|$ 是 Jacobian 行列式（绝对值）：

$$\left|\frac{\partial(x,y)}{\partial(u,v)}\right| = \left|\begin{vmatrix} \partial x/\partial u & \partial x/\partial v \\ \partial y/\partial u & \partial y/\partial v \end{vmatrix}\right|$$

### 3.2 几何意义

Jacobian 行列式的绝对值 $|J|$ 表示：在变量替换下，**小面积（或体积）元的缩放比例**。

- 若 $|J| > 1$：变换"放大"面积
- 若 $|J| < 1$：变换"缩小"面积
- 若 $|J| = 1$：保面积（如平移、旋转）

**极坐标例子**：$|J| = r$。在原点附近（$r$ 小），极坐标小格子 $(dr, d\theta)$ 对应的实际面积很小（$r\,dr\,d\theta$ 趋于 $0$）；在远离原点处（$r$ 大），同样的 $(dr, d\theta)$ 对应更大的实际面积。这就是为什么极坐标面积元需要乘以 $r$。

### 3.3 各坐标系 Jacobian 一览

| 坐标变换 | Jacobian $|J|$ |
|---|---|
| 极坐标：$(r,\theta) \to (x,y)$ | $r$ |
| 柱坐标：$(r,\theta,z) \to (x,y,z)$ | $r$ |
| 球坐标：$(\rho,\varphi,\theta) \to (x,y,z)$ | $\rho^2\sin\varphi$ |
| 仿射变换：$(u,v) \to (au+bv, cu+dv)$ | $|ad - bc|$ |

---

## 四、二重积分换序：三步法

积分的计算顺序有时会决定难度。换序（交换积分顺序）的标准步骤：

### 4.1 三步法

**第一步**：用不等式描述区域 $D$，**画出区域草图**。

**第二步**：用另一种方式描述 $D$（若原来"外 $x$ 内 $y$"，改成"外 $y$ 内 $x$"，反之亦然）。

**第三步**：重写积分限，写出新的二重积分。

### 4.2 两种描述方式

"外 $x$ 内 $y$"（X 型区域）：

$$\iint_D f(x,y)\,dA = \int_a^b \left[\int_{\varphi_1(x)}^{\varphi_2(x)} f(x,y)\,dy\right]dx$$

区域 $D$：$a \leq x \leq b$，$\varphi_1(x) \leq y \leq \varphi_2(x)$

"外 $y$ 内 $x$"（Y 型区域）：

$$\iint_D f(x,y)\,dA = \int_c^d \left[\int_{\psi_1(y)}^{\psi_2(y)} f(x,y)\,dx\right]dy$$

区域 $D$：$c \leq y \leq d$，$\psi_1(y) \leq x \leq \psi_2(y)$

### 4.3 什么时候需要换序

| 触发信号 | 说明 |
|---|---|
| 内层积分对当前变量无法求（如 $\int e^{y^2}\,dy$ 无初等原函数）| 换序后可能变成 $\int e^{y^2}\,dy$ 在外层，变成可积形式 |
| 积分限含复杂函数，换序后变成常数 | 简化计算 |
| 区域在当前方向分段（需分块计算）| 换向后可能一块搞定 |

---

## 五、演示题：极坐标计算高斯积分

**题目**：计算 $\displaystyle\iint_D e^{-(x^2+y^2)}\,dA$，其中 $D$ 是单位圆盘 $x^2 + y^2 \leq 1$。

> **读题。** 被积函数是 $e^{-(x^2+y^2)}$，在直角坐标下，$x$ 和 $y$ 的变化范围由 $\sqrt{1-x^2}$ 等的条件限制，内层积分 $\int_{-\sqrt{1-x^2}}^{\sqrt{1-x^2}} e^{-(x^2+y^2)}\,dy$ 无初等原函数（$\int e^{-y^2}\,dy$ 不能用初等函数表示）。
>
> **选坐标。** 被积函数含 $x^2 + y^2$，区域是圆盘，两个信号都指向**极坐标**。
>
> **换坐标。** $x^2 + y^2 = r^2$，$dA = r\,dr\,d\theta$，区域变为 $0 \leq r \leq 1$，$0 \leq \theta \leq 2\pi$。
>
> $$\iint_D e^{-(x^2+y^2)}\,dA = \int_0^{2\pi}\int_0^1 e^{-r^2} \cdot r\,dr\,d\theta.$$
>
> **分离变量。** $\theta$ 和 $r$ 相互独立（被积函数不含 $\theta$）：
>
> $$= \left[\int_0^{2\pi} d\theta\right]\left[\int_0^1 r e^{-r^2}\,dr\right] = 2\pi \cdot \int_0^1 r e^{-r^2}\,dr.$$
>
> **计算内层。** 令 $u = r^2$，$du = 2r\,dr$：
>
> $$\int_0^1 r e^{-r^2}\,dr = \frac{1}{2}\int_0^1 e^{-u}\,du = \frac{1}{2}\left[-e^{-u}\right]_0^1 = \frac{1}{2}(1 - e^{-1}).$$
>
> **得出结果。**
>
> $$\iint_D e^{-(x^2+y^2)}\,dA = 2\pi \cdot \frac{1-e^{-1}}{2} = \pi(1 - e^{-1}) \approx 1.986.$$
>
> **回顾。** 极坐标使原本不可积的 $e^{-x^2}$ 和 $e^{-y^2}$ 合并成了 $e^{-r^2}$，再与因子 $r$ 结合，变成 $re^{-r^2}$——这正好是 $-e^{-r^2}$ 的导数，完美可积。换坐标是"发现隐藏结构"的过程。

---

## 六、思考路标

**路标 1**：看到 $x^2 + y^2$（二维）或 $x^2 + y^2 + z^2$（三维）→ 几乎必换极/球坐标。$x^2+y^2 = r^2$，$x^2+y^2+z^2 = \rho^2$，是最常见的触发信号。

**路标 2**：看到圆形/圆环区域（$a \leq x^2+y^2 \leq b$）→ 换极坐标，积分限变为 $r$ 的常数范围，大幅简化。

**路标 3**：极坐标换元时，$dA = r\,dr\,d\theta$ 中的 $r$ **绝对不能漏掉**。这是最常见的计算错误之一。记忆方法：Jacobian 就是 $r$，面积元从 $dx\,dy$ 变成 $r\,dr\,d\theta$。

**路标 4**：球坐标中，$dV = \rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$。$\rho^2$ 来自径向，$\sin\varphi$ 来自极角（当 $\varphi = 0$ 或 $\pi$ 即北极/南极时，$\sin\varphi = 0$，对应面积元退化为点，合理）。

**路标 5**：内层积分出现 $e^{y^2}$、$\sin(y^2)$、$\dfrac{\sin y}{y}$ 等"无初等原函数"被积函数 → 考虑**换积分顺序**。先画区域草图，换序后内外层互换，可能变得可积。

**路标 6**：看到对称区域（关于 $x$ 轴、$y$ 轴、原点对称）+ 奇/偶被积函数 → 用**对称性**：若 $D$ 关于 $x$ 轴对称且 $f(x, -y) = -f(x, y)$，则积分为 $0$；若 $f(x,-y) = f(x,y)$，则积分为上半部分的 $2$ 倍。

**路标 7**：计算体积时，可以先问"能不能用对称性/截面法（Cavalieri 原理）简化"，再决定坐标系。有时截面积已知（如标准球体、椭球体），直接积分截面积更快。

**路标 8**：遇到 Jacobian 行列式计算困难时，验证：已知坐标变换（极/球/柱）的 Jacobian 直接记住，不需要每次重新计算行列式。

---

## 七、典型应用 3 例

### 例 1：极坐标——计算含圆的区域积分

**题目**：计算 $\displaystyle\iint_D \sqrt{x^2 + y^2}\,dA$，其中 $D$ 是圆盘 $x^2 + y^2 \leq 2y$（即以 $(0,1)$ 为圆心、半径为 $1$ 的圆）。

**分析**：区域是圆，被积函数含 $\sqrt{x^2+y^2}$，换极坐标。

先改写区域：$x^2 + y^2 \leq 2y$ $\Leftrightarrow$ $r^2 \leq 2r\sin\theta$ $\Leftrightarrow$ $r \leq 2\sin\theta$（$0 \leq \theta \leq \pi$）。

$$\iint_D \sqrt{x^2+y^2}\,dA = \int_0^\pi \int_0^{2\sin\theta} r \cdot r\,dr\,d\theta = \int_0^\pi \left[\frac{r^3}{3}\right]_0^{2\sin\theta} d\theta = \int_0^\pi \frac{8\sin^3\theta}{3}\,d\theta.$$

$$\int_0^\pi \sin^3\theta\,d\theta = \int_0^\pi (1-\cos^2\theta)\sin\theta\,d\theta = \left[-\cos\theta + \frac{\cos^3\theta}{3}\right]_0^\pi = \frac{4}{3}.$$

故积分 $= \dfrac{8}{3} \cdot \dfrac{4}{3} = \dfrac{32}{9}$。

---

### 例 2：球坐标——计算球外壳上的积分

**题目**：计算 $\displaystyle\iiint_V (x^2 + y^2 + z^2)\,dV$，其中 $V$ 是球壳 $1 \leq x^2 + y^2 + z^2 \leq 4$。

**分析**：被积函数含 $x^2+y^2+z^2 = \rho^2$，区域是球壳，换球坐标。

$\rho$ 的范围：$1 \leq \rho \leq 2$；$\varphi$：$0 \leq \varphi \leq \pi$；$\theta$：$0 \leq \theta \leq 2\pi$。

$$\iiint_V (x^2+y^2+z^2)\,dV = \int_0^{2\pi}\int_0^\pi\int_1^2 \rho^2 \cdot \rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$$

$$= \left[\int_0^{2\pi}d\theta\right]\left[\int_0^\pi \sin\varphi\,d\varphi\right]\left[\int_1^2 \rho^4\,d\rho\right] = 2\pi \cdot 2 \cdot \frac{\rho^5}{5}\Big|_1^2 = 4\pi \cdot \frac{32-1}{5} = \frac{124\pi}{5}.$$

---

### 例 3：换积分顺序——处理无初等原函数的被积函数

**题目**：计算 $\displaystyle\int_0^1 \int_x^1 e^{y^2}\,dy\,dx$。

**分析**：内层积分 $\int_x^1 e^{y^2}\,dy$ 无初等原函数，必须换积分顺序。

**第一步（画区域）**：原积分的区域 $D$：$0 \leq x \leq 1$，$x \leq y \leq 1$，即三角形区域 $\{0 \leq x \leq y \leq 1\}$。

**第二步（换描述）**：同一三角形用"外 $y$ 内 $x$"描述：$0 \leq y \leq 1$，$0 \leq x \leq y$。

**第三步（重写积分）**：

$$\int_0^1 \int_x^1 e^{y^2}\,dy\,dx = \int_0^1 \int_0^y e^{y^2}\,dx\,dy = \int_0^1 e^{y^2} \cdot y\,dy.$$

内层积分变为 $\int_0^y dx = y$（$e^{y^2}$ 对 $x$ 积分，$y$ 视为常数），现在外层是 $\int_0^1 y e^{y^2}\,dy$，令 $u = y^2$：

$$\int_0^1 y e^{y^2}\,dy = \frac{1}{2}\int_0^1 e^u\,du = \frac{e-1}{2}.$$

---

## 八、自测题

**第 1 题**：计算 $\displaystyle\iint_D (x + y)\,dA$，$D$ 是第一象限内的四分之一单位圆盘：$x \geq 0$，$y \geq 0$，$x^2 + y^2 \leq 1$。

💡 提示：换极坐标，$D'$：$0 \leq r \leq 1$，$0 \leq \theta \leq \pi/2$。$x + y = r\cos\theta + r\sin\theta = r(\cos\theta + \sin\theta)$，$dA = r\,dr\,d\theta$。积分 $= \int_0^{\pi/2}\int_0^1 r^2(\cos\theta+\sin\theta)\,dr\,d\theta = \dfrac{1}{3}\int_0^{\pi/2}(\cos\theta+\sin\theta)\,d\theta = \dfrac{1}{3}[\sin\theta - \cos\theta]_0^{\pi/2} = \dfrac{2}{3}$。

**第 2 题**：计算 $\displaystyle\iiint_V z\,dV$，其中 $V$ 是锥体 $\{z \geq \sqrt{x^2+y^2}, z \leq 1\}$。

💡 提示：用柱坐标，区域：$0 \leq r \leq z$（即 $z \geq r$）且 $0 \leq z \leq 1$，换积分顺序：$0 \leq z \leq 1$，$0 \leq r \leq z$，$0 \leq \theta \leq 2\pi$。$\displaystyle\int_0^{2\pi}\int_0^1\int_0^z z \cdot r\,dr\,dz\,d\theta = 2\pi\int_0^1 z \cdot \frac{z^2}{2}\,dz = \pi\int_0^1 z^3\,dz = \dfrac{\pi}{4}$。

**第 3 题**：交换积分顺序：$\displaystyle\int_0^1\int_{\sqrt{y}}^1 f(x,y)\,dx\,dy$。

💡 提示：画出区域：$0 \leq y \leq 1$，$\sqrt{y} \leq x \leq 1$，即 $y \leq x^2$，$0 \leq x \leq 1$。换成外 $x$ 内 $y$：$0 \leq x \leq 1$，$0 \leq y \leq x^2$，即 $\displaystyle\int_0^1\int_0^{x^2} f(x,y)\,dy\,dx$。

**第 4 题**：计算球体 $x^2 + y^2 + z^2 \leq R^2$ 的体积（用球坐标，验证公式 $V = \frac{4}{3}\pi R^3$）。

💡 提示：$V = \displaystyle\int_0^{2\pi}\int_0^\pi\int_0^R \rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta = 2\pi \cdot 2 \cdot \dfrac{R^3}{3} = \dfrac{4\pi R^3}{3}$（三个独立积分各自计算后相乘）。

**第 5 题**（综合）：设 $f(x) = \int_x^1 e^{t^2}\,dt$（$0 \leq x \leq 1$），计算 $\displaystyle\int_0^1 xf(x)\,dx$。

💡 提示：直接计算 $\int_0^1 xf(x)\,dx = \int_0^1 x \left[\int_x^1 e^{t^2}\,dt\right]dx$。这是 $\displaystyle\int_0^1\int_0^t x\,dx\,dt$ 的形式（换序：原区域 $0 \leq x \leq t \leq 1$，外层 $t$，内层 $x$）。$= \displaystyle\int_0^1 e^{t^2}\cdot\dfrac{t^2}{2}\,dt$，再令 $u = t^2$：$= \dfrac{1}{4}\int_0^1 e^u\,du = \dfrac{e-1}{4}$。
