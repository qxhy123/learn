# 第19章 重积分

## 学习目标

通过本章学习，你将能够：

- 理解二重积分的概念，掌握其几何意义与物理意义
- 熟练运用累次积分计算二重积分，掌握交换积分次序的方法
- 掌握极坐标下二重积分的计算
- 理解三重积分的概念，掌握直角坐标、柱坐标、球坐标下的计算方法
- 理解重积分换元法的原理，掌握 Jacobi 行列式的计算
- 能够运用重积分求解曲面面积、质心、转动惯量等实际问题

---

## 19.1 二重积分的概念

### 19.1.1 从体积问题引入

设有一个以平面区域 $D$ 为底、以曲面 $z = f(x, y) \geq 0$ 为顶的曲顶柱体，如何求其体积？

**分割**：将区域 $D$ 分成 $n$ 个小区域 $\Delta\sigma_1, \Delta\sigma_2, \ldots, \Delta\sigma_n$，记各小区域的面积也为 $\Delta\sigma_i$。

**近似**：在每个小区域 $\Delta\sigma_i$ 上任取一点 $(\xi_i, \eta_i)$，以 $f(\xi_i, \eta_i)$ 为高的小柱体体积近似为 $f(\xi_i, \eta_i)\Delta\sigma_i$。

**求和**：曲顶柱体的体积近似为

$$V \approx \sum_{i=1}^{n} f(\xi_i, \eta_i)\Delta\sigma_i$$

**取极限**：令分割的最大直径 $\lambda = \max\{d_i\} \to 0$（其中 $d_i$ 是 $\Delta\sigma_i$ 的直径），体积为

$$V = \lim_{\lambda \to 0} \sum_{i=1}^{n} f(\xi_i, \eta_i)\Delta\sigma_i$$

### 19.1.2 二重积分的定义

**定义**：设 $f(x, y)$ 是有界闭区域 $D$ 上的有界函数。将 $D$ 任意分成 $n$ 个小区域 $\Delta\sigma_1, \Delta\sigma_2, \ldots, \Delta\sigma_n$，在每个 $\Delta\sigma_i$ 上任取一点 $(\xi_i, \eta_i)$，作和式

$$\sum_{i=1}^{n} f(\xi_i, \eta_i)\Delta\sigma_i$$

如果当各小区域的直径中的最大值 $\lambda \to 0$ 时，此和式的极限存在且与分割方式及点的取法无关，则称此极限为 $f(x, y)$ 在 $D$ 上的**二重积分**，记作

$$\iint_D f(x, y)\,d\sigma = \lim_{\lambda \to 0} \sum_{i=1}^{n} f(\xi_i, \eta_i)\Delta\sigma_i$$

其中 $f(x, y)$ 称为**被积函数**，$D$ 称为**积分区域**，$d\sigma$ 称为**面积元素**。

在直角坐标系中，$d\sigma = dx\,dy$，故二重积分也记作

$$\iint_D f(x, y)\,dx\,dy$$

**存在性定理**：若 $f(x, y)$ 在有界闭区域 $D$ 上连续，则二重积分 $\iint_D f(x, y)\,d\sigma$ 存在。

### 19.1.3 几何意义与物理意义

**几何意义**：
- 当 $f(x, y) > 0$ 时，$\iint_D f(x, y)\,d\sigma$ 表示以 $D$ 为底、以曲面 $z = f(x, y)$ 为顶的曲顶柱体的体积
- 当 $f(x, y)$ 有正有负时，积分值等于曲面上方体积减去曲面下方体积

**物理意义**：
- 若 $\rho(x, y)$ 表示平面薄板 $D$ 上的面密度，则 $\iint_D \rho(x, y)\,d\sigma$ 表示薄板的总质量
- 特别地，$\iint_D 1\,d\sigma = \iint_D d\sigma$ 等于区域 $D$ 的面积

**二重积分的性质**：

1. **线性性**：$\iint_D [af(x,y) + bg(x,y)]\,d\sigma = a\iint_D f\,d\sigma + b\iint_D g\,d\sigma$

2. **区域可加性**：若 $D = D_1 \cup D_2$，$D_1 \cap D_2$ 的面积为零，则
   $$\iint_D f\,d\sigma = \iint_{D_1} f\,d\sigma + \iint_{D_2} f\,d\sigma$$

3. **保号性**：若 $f(x,y) \geq 0$ 在 $D$ 上，则 $\iint_D f\,d\sigma \geq 0$

4. **估值定理**：若 $m \leq f(x,y) \leq M$ 在 $D$ 上，$S$ 为 $D$ 的面积，则
   $$mS \leq \iint_D f\,d\sigma \leq MS$$

5. **中值定理**：若 $f(x,y)$ 在有界闭区域 $D$ 上连续，则存在 $(\xi, \eta) \in D$ 使得
   $$\iint_D f(x,y)\,d\sigma = f(\xi, \eta) \cdot S$$

---

## 19.2 二重积分的计算

### 19.2.1 直角坐标下的计算（累次积分）

将二重积分化为两次定积分（累次积分）是计算的基本方法。

**X-型区域**：若区域 $D$ 可表示为

$$D = \{(x, y) \mid a \leq x \leq b, \, \varphi_1(x) \leq y \leq \varphi_2(x)\}$$

则二重积分化为先对 $y$ 后对 $x$ 的累次积分：

$$\iint_D f(x, y)\,dx\,dy = \int_a^b dx \int_{\varphi_1(x)}^{\varphi_2(x)} f(x, y)\,dy$$

**Y-型区域**：若区域 $D$ 可表示为

$$D = \{(x, y) \mid c \leq y \leq d, \, \psi_1(y) \leq x \leq \psi_2(y)\}$$

则二重积分化为先对 $x$ 后对 $y$ 的累次积分：

$$\iint_D f(x, y)\,dx\,dy = \int_c^d dy \int_{\psi_1(y)}^{\psi_2(y)} f(x, y)\,dx$$

> **例题 19.1** 计算 $\iint_D xy\,dx\,dy$，其中 $D$ 是由 $y = x$，$y = x^2$ 围成的区域。

**解**：两曲线交点为 $(0, 0)$ 和 $(1, 1)$。区域 $D$ 可表示为 X-型：

$$D = \{(x, y) \mid 0 \leq x \leq 1, \, x^2 \leq y \leq x\}$$

$$\iint_D xy\,dx\,dy = \int_0^1 dx \int_{x^2}^{x} xy\,dy = \int_0^1 x \left[\frac{y^2}{2}\right]_{x^2}^{x} dx$$

$$= \int_0^1 x \cdot \frac{1}{2}(x^2 - x^4)\,dx = \frac{1}{2}\int_0^1 (x^3 - x^5)\,dx$$

$$= \frac{1}{2}\left[\frac{x^4}{4} - \frac{x^6}{6}\right]_0^1 = \frac{1}{2}\left(\frac{1}{4} - \frac{1}{6}\right) = \frac{1}{2} \cdot \frac{1}{12} = \frac{1}{24}$$

### 19.2.2 交换积分次序

有时交换积分次序可以简化计算。关键是画出积分区域，然后用另一种方式表示。

> **例题 19.2** 交换积分次序：$\int_0^1 dx \int_x^1 f(x, y)\,dy$。

**解**：原积分区域为 $D = \{(x, y) \mid 0 \leq x \leq 1, \, x \leq y \leq 1\}$。

画图可知，这是由 $y = x$、$x = 0$、$y = 1$ 围成的三角形区域。

改写为 Y-型：$D = \{(x, y) \mid 0 \leq y \leq 1, \, 0 \leq x \leq y\}$。

交换后：$\int_0^1 dy \int_0^y f(x, y)\,dx$。

> **例题 19.3** 计算 $\int_0^1 dx \int_x^1 e^{y^2}\,dy$。

**解**：$e^{y^2}$ 没有初等原函数，无法直接积分。交换积分次序：

区域 $D = \{(x, y) \mid 0 \leq x \leq 1, \, x \leq y \leq 1\} = \{(x, y) \mid 0 \leq y \leq 1, \, 0 \leq x \leq y\}$

$$\int_0^1 dx \int_x^1 e^{y^2}\,dy = \int_0^1 dy \int_0^y e^{y^2}\,dx = \int_0^1 e^{y^2} \cdot y\,dy$$

$$= \frac{1}{2}\int_0^1 e^{y^2}\,d(y^2) = \frac{1}{2}\left[e^{y^2}\right]_0^1 = \frac{1}{2}(e - 1)$$

### 19.2.3 极坐标下的计算

当积分区域是圆形、扇形或被积函数含 $x^2 + y^2$ 时，使用极坐标往往更方便。

**坐标变换**：$x = r\cos\theta$，$y = r\sin\theta$

**面积元素**：$d\sigma = r\,dr\,d\theta$（注意多出的因子 $r$）

若积分区域为

$$D = \{(r, \theta) \mid \alpha \leq \theta \leq \beta, \, r_1(\theta) \leq r \leq r_2(\theta)\}$$

则

$$\iint_D f(x, y)\,dx\,dy = \int_\alpha^\beta d\theta \int_{r_1(\theta)}^{r_2(\theta)} f(r\cos\theta, r\sin\theta) \cdot r\,dr$$

> **例题 19.4** 计算 $\iint_D e^{-(x^2+y^2)}\,dx\,dy$，其中 $D = \{(x, y) \mid x^2 + y^2 \leq 1\}$。

**解**：区域 $D$ 是单位圆盘，极坐标下为 $0 \leq \theta \leq 2\pi$，$0 \leq r \leq 1$。

$$\iint_D e^{-(x^2+y^2)}\,dx\,dy = \int_0^{2\pi} d\theta \int_0^1 e^{-r^2} \cdot r\,dr$$

$$= 2\pi \int_0^1 re^{-r^2}\,dr = 2\pi \cdot \left[-\frac{1}{2}e^{-r^2}\right]_0^1 = 2\pi \cdot \frac{1}{2}(1 - e^{-1}) = \pi(1 - e^{-1})$$

---

## 19.3 三重积分

### 19.3.1 三重积分的定义

**定义**：设 $f(x, y, z)$ 是空间有界闭区域 $\Omega$ 上的有界函数。将 $\Omega$ 分成 $n$ 个小区域 $\Delta v_i$，在每个 $\Delta v_i$ 上任取一点 $(\xi_i, \eta_i, \zeta_i)$，若极限

$$\lim_{\lambda \to 0} \sum_{i=1}^{n} f(\xi_i, \eta_i, \zeta_i)\Delta v_i$$

存在，则称此极限为 $f(x, y, z)$ 在 $\Omega$ 上的**三重积分**，记作

$$\iiint_\Omega f(x, y, z)\,dv$$

在直角坐标系中，$dv = dx\,dy\,dz$。

**物理意义**：若 $\rho(x, y, z)$ 为空间物体 $\Omega$ 的体密度，则 $\iiint_\Omega \rho\,dv$ 为物体的总质量。

### 19.3.2 直角坐标下的计算

**投影法**：设区域 $\Omega$ 在 $xOy$ 面上的投影为 $D_{xy}$，且

$$\Omega = \{(x, y, z) \mid (x, y) \in D_{xy}, \, z_1(x, y) \leq z \leq z_2(x, y)\}$$

则

$$\iiint_\Omega f(x, y, z)\,dv = \iint_{D_{xy}} \left[\int_{z_1(x,y)}^{z_2(x,y)} f(x, y, z)\,dz\right] dx\,dy$$

> **例题 19.5** 计算 $\iiint_\Omega z\,dv$，其中 $\Omega$ 是由 $z = 0$，$z = 1 - x - y$，$x = 0$，$y = 0$ 围成的四面体。

**解**：在 $xOy$ 面上的投影 $D_{xy} = \{(x, y) \mid x \geq 0, y \geq 0, x + y \leq 1\}$。

$$\iiint_\Omega z\,dv = \iint_{D_{xy}} \left[\int_0^{1-x-y} z\,dz\right] dx\,dy = \iint_{D_{xy}} \frac{(1-x-y)^2}{2}\,dx\,dy$$

$$= \frac{1}{2}\int_0^1 dx \int_0^{1-x} (1-x-y)^2\,dy$$

令 $u = 1 - x - y$，则 $dy = -du$，当 $y = 0$ 时 $u = 1 - x$，当 $y = 1 - x$ 时 $u = 0$：

$$= \frac{1}{2}\int_0^1 dx \int_{1-x}^{0} u^2 \cdot (-du) = \frac{1}{2}\int_0^1 \left[\frac{u^3}{3}\right]_0^{1-x} dx = \frac{1}{6}\int_0^1 (1-x)^3\,dx$$

$$= \frac{1}{6} \cdot \left[-\frac{(1-x)^4}{4}\right]_0^1 = \frac{1}{6} \cdot \frac{1}{4} = \frac{1}{24}$$

### 19.3.3 柱坐标

**柱坐标**：$(r, \theta, z)$，其中

$$x = r\cos\theta, \quad y = r\sin\theta, \quad z = z$$

**体积元素**：$dv = r\,dr\,d\theta\,dz$

柱坐标适用于关于 $z$ 轴对称或含 $x^2 + y^2$ 的问题。

> **例题 19.6** 计算 $\iiint_\Omega (x^2 + y^2)\,dv$，其中 $\Omega$ 是由 $x^2 + y^2 = 1$，$z = 0$，$z = 2$ 围成的圆柱体。

**解**：用柱坐标，$\Omega$：$0 \leq r \leq 1$，$0 \leq \theta \leq 2\pi$，$0 \leq z \leq 2$。

$$\iiint_\Omega (x^2 + y^2)\,dv = \int_0^{2\pi} d\theta \int_0^1 r^2 \cdot r\,dr \int_0^2 dz$$

$$= 2\pi \cdot \left[\frac{r^4}{4}\right]_0^1 \cdot 2 = 2\pi \cdot \frac{1}{4} \cdot 2 = \pi$$

### 19.3.4 球坐标

**球坐标**：$(\rho, \varphi, \theta)$，其中

$$x = \rho\sin\varphi\cos\theta, \quad y = \rho\sin\varphi\sin\theta, \quad z = \rho\cos\varphi$$

这里 $\rho \geq 0$ 是到原点的距离，$\varphi \in [0, \pi]$ 是与 $z$ 轴正向的夹角，$\theta \in [0, 2\pi)$ 是在 $xOy$ 面上投影与 $x$ 轴正向的夹角。

**体积元素**：$dv = \rho^2 \sin\varphi\,d\rho\,d\varphi\,d\theta$

球坐标适用于球形区域或含 $x^2 + y^2 + z^2$ 的问题。

> **例题 19.7** 计算 $\iiint_\Omega \sqrt{x^2 + y^2 + z^2}\,dv$，其中 $\Omega$ 是球 $x^2 + y^2 + z^2 \leq R^2$。

**解**：用球坐标，$\Omega$：$0 \leq \rho \leq R$，$0 \leq \varphi \leq \pi$，$0 \leq \theta \leq 2\pi$。

$$\iiint_\Omega \sqrt{x^2 + y^2 + z^2}\,dv = \int_0^{2\pi} d\theta \int_0^{\pi} \sin\varphi\,d\varphi \int_0^R \rho \cdot \rho^2\,d\rho$$

$$= 2\pi \cdot [-\cos\varphi]_0^{\pi} \cdot \left[\frac{\rho^4}{4}\right]_0^R = 2\pi \cdot 2 \cdot \frac{R^4}{4} = \pi R^4$$

---

## 19.4 重积分的换元法

### 19.4.1 一般换元公式

设变换 $T: x = x(u, v)$，$y = y(u, v)$ 将 $uv$ 平面上的区域 $D'$ 一一映射到 $xy$ 平面上的区域 $D$，且变换具有连续偏导数。

**二重积分换元公式**：

$$\iint_D f(x, y)\,dx\,dy = \iint_{D'} f(x(u,v), y(u,v)) \cdot |J|\,du\,dv$$

### 19.4.2 Jacobi 行列式

**Jacobi 行列式**（雅可比行列式）定义为：

$$J = \frac{\partial(x, y)}{\partial(u, v)} = \begin{vmatrix} \dfrac{\partial x}{\partial u} & \dfrac{\partial x}{\partial v} \\[10pt] \dfrac{\partial y}{\partial u} & \dfrac{\partial y}{\partial v} \end{vmatrix}$$

**三重积分换元公式**：

$$\iiint_\Omega f(x, y, z)\,dx\,dy\,dz = \iiint_{\Omega'} f(x, y, z) \cdot |J|\,du\,dv\,dw$$

其中

$$J = \frac{\partial(x, y, z)}{\partial(u, v, w)} = \begin{vmatrix} \dfrac{\partial x}{\partial u} & \dfrac{\partial x}{\partial v} & \dfrac{\partial x}{\partial w} \\[10pt] \dfrac{\partial y}{\partial u} & \dfrac{\partial y}{\partial v} & \dfrac{\partial y}{\partial w} \\[10pt] \dfrac{\partial z}{\partial u} & \dfrac{\partial z}{\partial v} & \dfrac{\partial z}{\partial w} \end{vmatrix}$$

**常用坐标变换的 Jacobi 行列式**：

- 极坐标：$J = r$
- 柱坐标：$J = r$
- 球坐标：$J = \rho^2 \sin\varphi$

> **例题 19.8** 计算 $\iint_D e^{\frac{y-x}{y+x}}\,dx\,dy$，其中 $D$ 是由 $x = 0$，$y = 0$，$x + y = 1$ 围成的三角形。

**解**：令 $u = y - x$，$v = y + x$，则 $x = \dfrac{v - u}{2}$，$y = \dfrac{v + u}{2}$。

$$J = \frac{\partial(x, y)}{\partial(u, v)} = \begin{vmatrix} -\dfrac{1}{2} & \dfrac{1}{2} \\[8pt] \dfrac{1}{2} & \dfrac{1}{2} \end{vmatrix} = -\frac{1}{4} - \frac{1}{4} = -\frac{1}{2}$$

原区域边界变换：$x = 0 \Rightarrow u = v$；$y = 0 \Rightarrow u = -v$；$x + y = 1 \Rightarrow v = 1$。

新区域 $D' = \{(u, v) \mid 0 \leq v \leq 1, \, -v \leq u \leq v\}$。

$$\iint_D e^{\frac{y-x}{y+x}}\,dx\,dy = \iint_{D'} e^{\frac{u}{v}} \cdot \frac{1}{2}\,du\,dv = \frac{1}{2}\int_0^1 dv \int_{-v}^{v} e^{\frac{u}{v}}\,du$$

$$= \frac{1}{2}\int_0^1 \left[v \cdot e^{\frac{u}{v}}\right]_{-v}^{v} dv = \frac{1}{2}\int_0^1 v(e - e^{-1})\,dv = \frac{e - e^{-1}}{2} \cdot \frac{1}{2} = \frac{e - e^{-1}}{4}$$

---

## 19.5 重积分的应用

### 19.5.1 曲面面积

设曲面 $S$ 由方程 $z = f(x, y)$ 给出，$(x, y) \in D$，且 $f$ 有连续偏导数。曲面面积为：

$$A = \iint_D \sqrt{1 + \left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\,dx\,dy$$

> **例题 19.9** 求球面 $x^2 + y^2 + z^2 = R^2$ 的表面积。

**解**：由对称性，只计算上半球面 $z = \sqrt{R^2 - x^2 - y^2}$ 的面积再乘以 $2$。

$$\frac{\partial z}{\partial x} = \frac{-x}{\sqrt{R^2 - x^2 - y^2}}, \quad \frac{\partial z}{\partial y} = \frac{-y}{\sqrt{R^2 - x^2 - y^2}}$$

$$\sqrt{1 + z_x^2 + z_y^2} = \sqrt{1 + \frac{x^2 + y^2}{R^2 - x^2 - y^2}} = \frac{R}{\sqrt{R^2 - x^2 - y^2}}$$

投影区域为 $D = \{(x, y) \mid x^2 + y^2 \leq R^2\}$，用极坐标：

$$A_{上半球} = \iint_D \frac{R}{\sqrt{R^2 - x^2 - y^2}}\,dx\,dy = \int_0^{2\pi} d\theta \int_0^R \frac{R}{\sqrt{R^2 - r^2}} \cdot r\,dr$$

$$= 2\pi R \int_0^R \frac{r}{\sqrt{R^2 - r^2}}\,dr = 2\pi R \left[-\sqrt{R^2 - r^2}\right]_0^R = 2\pi R \cdot R = 2\pi R^2$$

故球面总面积为 $A = 2 \times 2\pi R^2 = 4\pi R^2$。

### 19.5.2 质心与转动惯量

**平面薄板的质心**：设薄板占据区域 $D$，面密度为 $\rho(x, y)$，则质心坐标为

$$\bar{x} = \frac{\iint_D x\rho(x, y)\,d\sigma}{\iint_D \rho(x, y)\,d\sigma}, \quad \bar{y} = \frac{\iint_D y\rho(x, y)\,d\sigma}{\iint_D \rho(x, y)\,d\sigma}$$

**空间物体的质心**：设物体占据区域 $\Omega$，体密度为 $\rho(x, y, z)$，则

$$\bar{x} = \frac{\iiint_\Omega x\rho\,dv}{\iiint_\Omega \rho\,dv}, \quad \bar{y} = \frac{\iiint_\Omega y\rho\,dv}{\iiint_\Omega \rho\,dv}, \quad \bar{z} = \frac{\iiint_\Omega z\rho\,dv}{\iiint_\Omega \rho\,dv}$$

**转动惯量**：质点系对某轴的转动惯量定义为 $I = \sum m_i r_i^2$，其中 $r_i$ 是质点到轴的距离。

对于连续分布的物体，平面薄板对 $x$ 轴、$y$ 轴、原点的转动惯量分别为：

$$I_x = \iint_D y^2 \rho\,d\sigma, \quad I_y = \iint_D x^2 \rho\,d\sigma, \quad I_O = \iint_D (x^2 + y^2)\rho\,d\sigma$$

> **例题 19.10** 求均匀薄板（密度 $\rho = 1$）$D = \{(x, y) \mid 0 \leq x \leq 1, 0 \leq y \leq 1\}$ 的质心和对原点的转动惯量。

**解**：质量 $M = \iint_D 1\,d\sigma = 1$。

$$\bar{x} = \iint_D x\,d\sigma = \int_0^1 \int_0^1 x\,dx\,dy = \int_0^1 \frac{1}{2}\,dy = \frac{1}{2}$$

由对称性，$\bar{y} = \dfrac{1}{2}$。质心为 $\left(\dfrac{1}{2}, \dfrac{1}{2}\right)$。

$$I_O = \iint_D (x^2 + y^2)\,d\sigma = \int_0^1 \int_0^1 (x^2 + y^2)\,dx\,dy = \int_0^1 \left(\frac{1}{3} + y^2\right) dy = \frac{1}{3} + \frac{1}{3} = \frac{2}{3}$$

---

## 本章小结

1. **二重积分**是"分割、近似、求和、取极限"的结果，几何上表示曲顶柱体的体积。

2. **计算方法**：
   - 直角坐标：化为累次积分 $\int_a^b dx \int_{\varphi_1(x)}^{\varphi_2(x)} f(x,y)\,dy$
   - 极坐标：$d\sigma = r\,dr\,d\theta$，适用于圆形区域或含 $x^2 + y^2$ 的被积函数

3. **交换积分次序**：画出积分区域，重新用另一种方式表示，可简化某些积分的计算。

4. **三重积分**：
   - 直角坐标：投影法化为先一后二的累次积分
   - 柱坐标：$dv = r\,dr\,d\theta\,dz$
   - 球坐标：$dv = \rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$

5. **换元公式**：$\iint_D f(x,y)\,dx\,dy = \iint_{D'} f(x(u,v), y(u,v)) |J|\,du\,dv$，其中 $J$ 是 Jacobi 行列式。

6. **应用**：曲面面积 $A = \iint_D \sqrt{1 + z_x^2 + z_y^2}\,d\sigma$；质心、转动惯量的计算。

---

## 深度学习应用

重积分在深度学习中有广泛应用，主要体现在概率期望计算、蒙特卡洛采样、变分推断和生成模型中。

### 期望的多重积分形式

设 $(X, Y)$ 是联合分布为 $p(x, y)$ 的连续随机向量，则函数 $f(X, Y)$ 的期望定义为二重积分：

$$\mathbb{E}[f(X,Y)] = \iint f(x,y)\, p(x,y)\, dx\, dy$$

更一般地，对于 $n$ 维随机向量 $\mathbf{x} \sim p(\mathbf{x})$，有

$$\mathbb{E}[f(\mathbf{x})] = \int \cdots \int f(\mathbf{x})\, p(\mathbf{x})\, d\mathbf{x}$$

**示例**：设 $(X, Y)$ 服从二维标准正态分布 $p(x,y) = \frac{1}{2\pi}e^{-\frac{x^2+y^2}{2}}$，则

$$\mathbb{E}[X^2 + Y^2] = \iint (x^2 + y^2)\, p(x,y)\, dx\, dy = 2$$

（即两个独立标准正态变量方差之和。）

### 蒙特卡洛积分

高维积分在深度学习中往往难以解析计算，蒙特卡洛方法用采样代替精确积分：

$$\int f(x)\, p(x)\, dx \approx \frac{1}{N}\sum_{i=1}^{N} f(x_i), \quad x_i \sim p(x)$$

**收敛性**：由大数定律，当 $N \to \infty$ 时，右侧均值依概率收敛到期望值。误差量级为 $O(1/\sqrt{N})$，与维度无关，这正是蒙特卡洛方法在高维场景中优于数值积分的原因。

在深度学习中，蒙特卡洛积分用于：
- **变分自编码器（VAE）**中对隐变量的期望估计
- **策略梯度方法**中对回报期望的估计
- **贝叶斯神经网络**中对后验预测分布的近似

### 变分推断中的 ELBO

变分推断的核心目标是最大化证据下界（Evidence Lower BOund，ELBO）。由 Jensen 不等式和重积分运算可推导：

$$\log p(x) \geq \mathbb{E}_q[\log p(x, z)] - \mathbb{E}_q[\log q(z)] =: \mathcal{L}(q)$$

其中各期望均为对隐变量 $z$ 的积分：

$$\mathbb{E}_q[\log p(x, z)] = \int \log p(x, z)\, q(z)\, dz$$

$$\mathbb{E}_q[\log q(z)] = \int \log q(z)\, q(z)\, dz \quad \text{（即 } q \text{ 的负熵）}$$

ELBO 可改写为：

$$\mathcal{L}(q) = \mathbb{E}_q[\log p(x \mid z)] - D_{\mathrm{KL}}(q(z) \,\|\, p(z))$$

第一项用蒙特卡洛估计（重构期望），第二项在高斯先验假设下有解析形式。

### 换元积分与生成模型（Normalizing Flows）

归一化流（Normalizing Flows）利用换元公式构造复杂分布。设双射 $g: \mathbf{z} \mapsto \mathbf{x}$，$\mathbf{z} \sim p_z(\mathbf{z})$，则 $\mathbf{x} = g(\mathbf{z})$ 的密度为

$$p_x(\mathbf{x}) = p_z(g^{-1}(\mathbf{x})) \cdot \left|\det\frac{\partial g^{-1}}{\partial \mathbf{x}}\right|$$

这正是多重积分换元公式中 Jacobi 行列式的直接体现：

$$\int_D f(\mathbf{x})\, d\mathbf{x} = \int_{D'} f(g(\mathbf{z}))\, |\det J_g(\mathbf{z})|\, d\mathbf{z}$$

训练时最大化对数似然：

$$\log p_x(\mathbf{x}) = \log p_z(g^{-1}(\mathbf{x})) + \log\left|\det\frac{\partial g^{-1}}{\partial \mathbf{x}}\right|$$

设计 Normalizing Flows 的关键挑战就在于构造 Jacobi 行列式易于计算的变换 $g$（如 RealNVP、Glow 等架构）。

### 代码示例

```python
import torch
import torch.distributions as dist

# 蒙特卡洛积分
def monte_carlo_integration(f, distribution, n_samples=10000):
    """使用蒙特卡洛方法计算 E[f(X)]"""
    samples = distribution.sample((n_samples,))
    return f(samples).mean()

# 计算 E[X^2] where X ~ N(0,1)，理论值 = 1
normal = dist.Normal(0, 1)
expectation = monte_carlo_integration(lambda x: x**2, normal)
print(f"E[X²] ≈ {expectation.item():.4f}, 理论值 = 1.0")

# 二维积分: E[X·Y] where (X,Y) ~ N(0,I)
mvn = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
samples = mvn.sample((10000,))
xy_product = (samples[:, 0] * samples[:, 1]).mean()
print(f"E[XY] ≈ {xy_product.item():.4f}, 理论值 = 0 (独立)")
```

---

## 练习题

**1.** 计算 $\iint_D (x + y)\,dx\,dy$，其中 $D$ 是由 $y = x$，$y = 2x$，$x = 1$ 围成的区域。

**2.** 交换积分次序并计算：$\int_0^2 dy \int_y^2 \sqrt{x^3 + 1}\,dx$。

**3.** 用极坐标计算 $\iint_D \sqrt{x^2 + y^2}\,dx\,dy$，其中 $D = \{(x, y) \mid 1 \leq x^2 + y^2 \leq 4\}$。

**4.** 计算 $\iiint_\Omega xyz\,dv$，其中 $\Omega$ 是由 $x = 0$，$y = 0$，$z = 0$，$x + y + z = 1$ 围成的四面体。

**5.** 求抛物面 $z = x^2 + y^2$ 在 $0 \leq z \leq 1$ 部分的曲面面积。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.** 区域 $D = \{(x, y) \mid 0 \leq x \leq 1, x \leq y \leq 2x\}$。

$$\iint_D (x + y)\,dx\,dy = \int_0^1 dx \int_x^{2x} (x + y)\,dy = \int_0^1 \left[xy + \frac{y^2}{2}\right]_x^{2x} dx$$

$$= \int_0^1 \left(2x^2 + 2x^2 - x^2 - \frac{x^2}{2}\right) dx = \int_0^1 \frac{5x^2}{2}\,dx = \frac{5}{2} \cdot \frac{1}{3} = \frac{5}{6}$$

---

**2.** 原积分区域 $D = \{(x, y) \mid 0 \leq y \leq 2, y \leq x \leq 2\}$。

交换次序：$D = \{(x, y) \mid 0 \leq x \leq 2, 0 \leq y \leq x\}$。

$$\int_0^2 dy \int_y^2 \sqrt{x^3 + 1}\,dx = \int_0^2 dx \int_0^x \sqrt{x^3 + 1}\,dy = \int_0^2 x\sqrt{x^3 + 1}\,dx$$

令 $u = x^3 + 1$，$du = 3x^2\,dx$，注意 $x\,dx = \dfrac{1}{3}x^{-1}\,du$...

更简单的方法：令 $t = x^3 + 1$，则 $dt = 3x^2 dx$，$x dx = \frac{1}{3} \cdot \frac{dt}{x} = \frac{dt}{3x}$，但 $x = (t-1)^{1/3}$...

直接计算：令 $u = x^3 + 1$，则当 $x = 0$ 时 $u = 1$，当 $x = 2$ 时 $u = 9$。

$$\int_0^2 x\sqrt{x^3 + 1}\,dx$$

令 $u = x^{3/2}$，不便。直接用换元 $u = x^3 + 1$：

$$\int_0^2 \sqrt{x^3 + 1} \cdot x\,dx = \frac{1}{3}\int_0^2 \sqrt{x^3 + 1} \cdot 3x^2 \cdot \frac{1}{x}\,dx$$

注意 $d(x^3+1) = 3x^2 dx$，所以：

$$= \frac{2}{9}\int_1^9 \sqrt{u}\,du = \frac{2}{9} \cdot \frac{2}{3}u^{3/2}\Big|_1^9 = \frac{4}{27}(27 - 1) = \frac{4 \cdot 26}{27} = \frac{104}{27}$$

---

**3.** 区域为圆环 $1 \leq r \leq 2$，$0 \leq \theta \leq 2\pi$。

$$\iint_D \sqrt{x^2 + y^2}\,dx\,dy = \int_0^{2\pi} d\theta \int_1^2 r \cdot r\,dr = 2\pi \int_1^2 r^2\,dr = 2\pi \cdot \frac{r^3}{3}\Big|_1^2 = 2\pi \cdot \frac{7}{3} = \frac{14\pi}{3}$$

---

**4.** 四面体在 $xOy$ 面上的投影为 $D_{xy} = \{(x,y) \mid x \geq 0, y \geq 0, x + y \leq 1\}$。

$$\iiint_\Omega xyz\,dv = \int_0^1 dx \int_0^{1-x} dy \int_0^{1-x-y} xyz\,dz$$

$$= \int_0^1 dx \int_0^{1-x} xy \cdot \frac{(1-x-y)^2}{2}\,dy$$

令 $I = \int_0^{1-x} y(1-x-y)^2\,dy$。设 $t = 1 - x - y$，$y = 1 - x - t$，$dy = -dt$：

$$I = \int_{1-x}^0 (1-x-t)t^2 \cdot (-dt) = \int_0^{1-x} [(1-x)t^2 - t^3]\,dt$$

$$= (1-x)\frac{(1-x)^3}{3} - \frac{(1-x)^4}{4} = (1-x)^4\left(\frac{1}{3} - \frac{1}{4}\right) = \frac{(1-x)^4}{12}$$

$$\iiint_\Omega xyz\,dv = \frac{1}{2}\int_0^1 x \cdot \frac{(1-x)^4}{12}\,dx = \frac{1}{24}\int_0^1 x(1-x)^4\,dx$$

$$= \frac{1}{24}\int_0^1 [(1-x)^4 - (1-x)^5]\,dx = \frac{1}{24}\left[\frac{1}{5} - \frac{1}{6}\right] = \frac{1}{24} \cdot \frac{1}{30} = \frac{1}{720}$$

---

**5.** 曲面 $z = x^2 + y^2$，投影区域 $D = \{(x,y) \mid x^2 + y^2 \leq 1\}$。

$$\frac{\partial z}{\partial x} = 2x, \quad \frac{\partial z}{\partial y} = 2y$$

$$\sqrt{1 + z_x^2 + z_y^2} = \sqrt{1 + 4x^2 + 4y^2} = \sqrt{1 + 4(x^2 + y^2)}$$

用极坐标：

$$A = \int_0^{2\pi} d\theta \int_0^1 \sqrt{1 + 4r^2} \cdot r\,dr = 2\pi \int_0^1 r\sqrt{1 + 4r^2}\,dr$$

令 $u = 1 + 4r^2$，$du = 8r\,dr$：

$$= 2\pi \cdot \frac{1}{8}\int_1^5 \sqrt{u}\,du = \frac{\pi}{4} \cdot \frac{2}{3}u^{3/2}\Big|_1^5 = \frac{\pi}{6}(5\sqrt{5} - 1)$$

</details>
