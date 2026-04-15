# 第20章 曲线积分

## 学习目标

通过本章学习，你将能够：

- 理解第一类曲线积分的定义，掌握其物理意义（曲线的质量）
- 掌握利用参数化计算第一类曲线积分的方法
- 理解第二类曲线积分的定义，掌握其物理意义（变力做功）
- 理解两类曲线积分之间的联系
- 掌握 Green 公式及其应用条件
- 理解路径无关的条件，掌握势函数的求法
- 能够运用曲线积分求解平面区域面积等实际问题

---

## 20.1 第一类曲线积分（对弧长的曲线积分）

### 20.1.1 物理背景：曲线的质量

设有一条平面曲线 $L$，其线密度为 $\rho(x, y)$（单位长度的质量）。如何求曲线的总质量？

**分割**：将曲线 $L$ 分成 $n$ 小段 $\Delta s_1, \Delta s_2, \ldots, \Delta s_n$。

**近似**：在每小段 $\Delta s_i$ 上任取一点 $(\xi_i, \eta_i)$，该小段的质量近似为 $\rho(\xi_i, \eta_i)\Delta s_i$。

**求和**：曲线的总质量近似为

$$M \approx \sum_{i=1}^{n} \rho(\xi_i, \eta_i)\Delta s_i$$

**取极限**：令分割的最大弧长 $\lambda = \max\{\Delta s_i\} \to 0$，得到

$$M = \lim_{\lambda \to 0} \sum_{i=1}^{n} \rho(\xi_i, \eta_i)\Delta s_i$$

### 20.1.2 第一类曲线积分的定义

**定义**：设 $L$ 是平面上的一条光滑曲线（或分段光滑曲线），$f(x, y)$ 是定义在 $L$ 上的有界函数。将 $L$ 任意分成 $n$ 小段，第 $i$ 段的弧长为 $\Delta s_i$，在其上任取一点 $(\xi_i, \eta_i)$，作和式

$$\sum_{i=1}^{n} f(\xi_i, \eta_i)\Delta s_i$$

如果当 $\lambda = \max\{\Delta s_i\} \to 0$ 时，此和式的极限存在且与分割方式及点的取法无关，则称此极限为 $f(x, y)$ 在曲线 $L$ 上的**第一类曲线积分**（或**对弧长的曲线积分**），记作

$$\int_L f(x, y)\,ds = \lim_{\lambda \to 0} \sum_{i=1}^{n} f(\xi_i, \eta_i)\Delta s_i$$

**存在性**：若 $f(x, y)$ 在光滑曲线 $L$ 上连续，则第一类曲线积分存在。

**性质**：

1. **线性性**：$\int_L [af + bg]\,ds = a\int_L f\,ds + b\int_L g\,ds$

2. **路径可加性**：若 $L = L_1 + L_2$，则 $\int_L f\,ds = \int_{L_1} f\,ds + \int_{L_2} f\,ds$

3. **与方向无关**：第一类曲线积分的值与曲线的方向无关

### 20.1.3 计算方法（参数化）

设曲线 $L$ 的参数方程为

$$\begin{cases} x = x(t) \\ y = y(t) \end{cases}, \quad t \in [\alpha, \beta]$$

其中 $x(t)$、$y(t)$ 具有连续导数，且 $[x'(t)]^2 + [y'(t)]^2 \neq 0$。

**弧长微元**：$ds = \sqrt{[x'(t)]^2 + [y'(t)]^2}\,dt$

**计算公式**：

$$\int_L f(x, y)\,ds = \int_\alpha^\beta f(x(t), y(t))\sqrt{[x'(t)]^2 + [y'(t)]^2}\,dt$$

**特殊情形**：

- 若曲线由 $y = y(x)$（$a \leq x \leq b$）给出，则 $ds = \sqrt{1 + [y'(x)]^2}\,dx$

$$\int_L f(x, y)\,ds = \int_a^b f(x, y(x))\sqrt{1 + [y'(x)]^2}\,dx$$

- 若曲线由极坐标 $r = r(\theta)$（$\alpha \leq \theta \leq \beta$）给出，则 $ds = \sqrt{r^2 + [r'(\theta)]^2}\,d\theta$

> **例题 20.1** 计算 $\int_L (x^2 + y^2)\,ds$，其中 $L$ 是圆周 $x^2 + y^2 = a^2$。

**解**：将圆周参数化：$x = a\cos t$，$y = a\sin t$，$t \in [0, 2\pi]$。

$$x'(t) = -a\sin t, \quad y'(t) = a\cos t$$

$$ds = \sqrt{a^2\sin^2 t + a^2\cos^2 t}\,dt = a\,dt$$

$$\int_L (x^2 + y^2)\,ds = \int_0^{2\pi} a^2 \cdot a\,dt = a^3 \int_0^{2\pi} dt = 2\pi a^3$$

> **例题 20.2** 计算 $\int_L y\,ds$，其中 $L$ 是抛物线 $y = x^2$ 从 $(0, 0)$ 到 $(1, 1)$ 的一段。

**解**：曲线由 $y = x^2$（$0 \leq x \leq 1$）给出。

$$ds = \sqrt{1 + (2x)^2}\,dx = \sqrt{1 + 4x^2}\,dx$$

$$\int_L y\,ds = \int_0^1 x^2\sqrt{1 + 4x^2}\,dx$$

令 $2x = \tan\theta$，则 $dx = \dfrac{1}{2}\sec^2\theta\,d\theta$，$\sqrt{1 + 4x^2} = \sec\theta$。

当 $x = 0$ 时 $\theta = 0$，当 $x = 1$ 时 $\theta = \arctan 2$。

$$= \int_0^{\arctan 2} \frac{\tan^2\theta}{4} \cdot \sec\theta \cdot \frac{1}{2}\sec^2\theta\,d\theta = \frac{1}{8}\int_0^{\arctan 2} \tan^2\theta\sec^3\theta\,d\theta$$

$$= \frac{1}{8}\int_0^{\arctan 2} (\sec^2\theta - 1)\sec^3\theta\,d\theta = \frac{1}{8}\int_0^{\arctan 2} (\sec^5\theta - \sec^3\theta)\,d\theta$$

利用递推公式或查表，最终得

$$\int_L y\,ds = \frac{1}{24}\left[(2 + \sqrt{5})\sqrt{5} - \ln(2 + \sqrt{5})\right] = \frac{5\sqrt{5} + 1}{24} - \frac{\ln(2 + \sqrt{5})}{24}$$

化简：$\int_L y\,ds = \dfrac{5\sqrt{5} + 1 - \ln(2 + \sqrt{5})}{24}$

---

## 20.2 第二类曲线积分（对坐标的曲线积分）

### 20.2.1 物理背景：变力做功

设质点在力场 $\mathbf{F}(x, y) = P(x, y)\mathbf{i} + Q(x, y)\mathbf{j}$ 的作用下，沿曲线 $L$ 从点 $A$ 移动到点 $B$。如何求力 $\mathbf{F}$ 所做的功？

**分割**：将曲线 $L$ 分成 $n$ 小段。

**近似**：在第 $i$ 小段上，力近似为常力 $\mathbf{F}(\xi_i, \eta_i)$，位移向量为 $(\Delta x_i, \Delta y_i)$，做功近似为

$$\Delta W_i \approx P(\xi_i, \eta_i)\Delta x_i + Q(\xi_i, \eta_i)\Delta y_i$$

**求和与取极限**：

$$W = \lim_{\lambda \to 0} \sum_{i=1}^{n} [P(\xi_i, \eta_i)\Delta x_i + Q(\xi_i, \eta_i)\Delta y_i]$$

### 20.2.2 第二类曲线积分的定义

**定义**：设 $L$ 是平面上从点 $A$ 到点 $B$ 的一条有向光滑曲线，$P(x, y)$、$Q(x, y)$ 是定义在 $L$ 上的有界函数。将 $L$ 任意分成 $n$ 小段，在第 $i$ 小段上任取一点 $(\xi_i, \eta_i)$，该小段在 $x$ 轴和 $y$ 轴上的投影分别为 $\Delta x_i$ 和 $\Delta y_i$。若极限

$$\lim_{\lambda \to 0} \sum_{i=1}^{n} P(\xi_i, \eta_i)\Delta x_i$$

存在，则称此极限为 $P(x, y)$ 在有向曲线 $L$ 上**对 $x$ 的曲线积分**，记作

$$\int_L P(x, y)\,dx$$

类似地定义 $\int_L Q(x, y)\,dy$。

**第二类曲线积分**的一般形式为：

$$\int_L P(x, y)\,dx + Q(x, y)\,dy = \int_L P\,dx + \int_L Q\,dy$$

也可写成向量形式：$\int_L \mathbf{F} \cdot d\mathbf{r}$，其中 $\mathbf{F} = (P, Q)$，$d\mathbf{r} = (dx, dy)$。

**性质**：

1. **线性性**：与第一类曲线积分类似

2. **路径可加性**：若 $L = L_1 + L_2$，则积分可加

3. **方向相关性**：若 $L^-$ 表示与 $L$ 方向相反的曲线，则
   $$\int_{L^-} P\,dx + Q\,dy = -\int_L P\,dx + Q\,dy$$

### 20.2.3 计算方法

设有向曲线 $L$ 的参数方程为 $x = x(t)$，$y = y(t)$，$t$ 从 $\alpha$ 变到 $\beta$（$\alpha < \beta$ 或 $\alpha > \beta$，取决于曲线方向）。

**计算公式**：

$$\int_L P\,dx + Q\,dy = \int_\alpha^\beta [P(x(t), y(t))x'(t) + Q(x(t), y(t))y'(t)]\,dt$$

> **例题 20.3** 计算 $\int_L y\,dx + x\,dy$，其中 $L$ 是从点 $(0, 0)$ 沿抛物线 $y = x^2$ 到点 $(1, 1)$。

**解**：以 $x$ 为参数，$y = x^2$，$x$ 从 $0$ 变到 $1$。

$$dy = 2x\,dx$$

$$\int_L y\,dx + x\,dy = \int_0^1 [x^2 + x \cdot 2x]\,dx = \int_0^1 3x^2\,dx = x^3\Big|_0^1 = 1$$

> **例题 20.4** 计算 $\int_L y\,dx + x\,dy$，其中 $L$ 是从点 $(0, 0)$ 沿直线 $y = x$ 到点 $(1, 1)$。

**解**：以 $x$ 为参数，$y = x$，$dy = dx$。

$$\int_L y\,dx + x\,dy = \int_0^1 [x + x]\,dx = \int_0^1 2x\,dx = x^2\Big|_0^1 = 1$$

**观察**：例题 20.3 和 20.4 中，沿不同路径但积分值相同，这与路径无关性有关（见 20.4 节）。

### 20.2.4 两类曲线积分的关系

设有向曲线 $L$ 在点 $(x, y)$ 处的单位切向量为 $\mathbf{T} = (\cos\alpha, \cos\beta)$，则

$$dx = \cos\alpha\,ds, \quad dy = \cos\beta\,ds$$

因此

$$\int_L P\,dx + Q\,dy = \int_L (P\cos\alpha + Q\cos\beta)\,ds$$

这表明第二类曲线积分可以化为第一类曲线积分。

---

## 20.3 Green 公式

### 20.3.1 公式陈述

**Green 公式**将平面区域上的二重积分与其边界曲线上的曲线积分联系起来。

**定理**（Green 公式）：设 $D$ 是平面上由分段光滑曲线 $L$ 围成的有界闭区域，函数 $P(x, y)$、$Q(x, y)$ 在 $D$ 上具有连续的一阶偏导数，则

$$\iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dx\,dy = \oint_L P\,dx + Q\,dy$$

其中 $L$ 取**正向**，即沿 $L$ 行走时区域 $D$ 始终在左侧（逆时针方向）。

### 20.3.2 证明思路

对于简单区域（既是 X-型又是 Y-型），分别证明：

$$\iint_D \frac{\partial P}{\partial y}\,dx\,dy = -\oint_L P\,dx$$

$$\iint_D \frac{\partial Q}{\partial x}\,dx\,dy = \oint_L Q\,dy$$

两式相减即得 Green 公式。

**第一个等式的证明**：设 $D = \{(x, y) \mid a \leq x \leq b, \, \varphi_1(x) \leq y \leq \varphi_2(x)\}$。

$$\iint_D \frac{\partial P}{\partial y}\,dx\,dy = \int_a^b dx \int_{\varphi_1(x)}^{\varphi_2(x)} \frac{\partial P}{\partial y}\,dy = \int_a^b [P(x, \varphi_2(x)) - P(x, \varphi_1(x))]\,dx$$

而 $\oint_L P\,dx$ 分为上下两段：

- 下边界 $L_1$：$y = \varphi_1(x)$，$x$ 从 $a$ 到 $b$，贡献 $\int_a^b P(x, \varphi_1(x))\,dx$
- 上边界 $L_2$：$y = \varphi_2(x)$，$x$ 从 $b$ 到 $a$（正向），贡献 $\int_b^a P(x, \varphi_2(x))\,dx = -\int_a^b P(x, \varphi_2(x))\,dx$

故 $\oint_L P\,dx = \int_a^b P(x, \varphi_1(x))\,dx - \int_a^b P(x, \varphi_2(x))\,dx = -\iint_D \frac{\partial P}{\partial y}\,dx\,dy$

### 20.3.3 应用条件

Green 公式要求：

1. **区域 $D$ 有界**，边界 $L$ 是分段光滑的闭曲线
2. **$P$、$Q$ 在 $D$（含边界）上有连续偏导数**
3. **$L$ 取正向**

**单连通区域**：区域内无"洞"，任何闭曲线都可以连续收缩为一点。

**复连通区域**：区域内有"洞"。此时需要引入割线将其化为单连通区域，或将积分分解到各边界上。

> **例题 20.5** 利用 Green 公式计算 $\oint_L (x^2 - y)\,dx + (y^2 + x)\,dy$，其中 $L$ 是圆周 $x^2 + y^2 = 1$ 的正向。

**解**：$P = x^2 - y$，$Q = y^2 + x$。

$$\frac{\partial Q}{\partial x} = 1, \quad \frac{\partial P}{\partial y} = -1$$

$$\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} = 1 - (-1) = 2$$

由 Green 公式：

$$\oint_L (x^2 - y)\,dx + (y^2 + x)\,dy = \iint_D 2\,dx\,dy = 2 \cdot \pi \cdot 1^2 = 2\pi$$

### 20.3.4 利用 Green 公式计算面积

取 $P = -y$，$Q = x$，则 $\dfrac{\partial Q}{\partial x} - \dfrac{\partial P}{\partial y} = 1 + 1 = 2$。

由 Green 公式：

$$\iint_D 2\,dx\,dy = \oint_L -y\,dx + x\,dy$$

故**平面区域的面积**为：

$$S = \frac{1}{2}\oint_L x\,dy - y\,dx$$

> **例题 20.6** 求椭圆 $\dfrac{x^2}{a^2} + \dfrac{y^2}{b^2} = 1$ 围成区域的面积。

**解**：椭圆的参数方程为 $x = a\cos t$，$y = b\sin t$，$t \in [0, 2\pi]$（正向）。

$$S = \frac{1}{2}\oint_L x\,dy - y\,dx = \frac{1}{2}\int_0^{2\pi} [a\cos t \cdot b\cos t - b\sin t \cdot (-a\sin t)]\,dt$$

$$= \frac{1}{2}\int_0^{2\pi} ab(\cos^2 t + \sin^2 t)\,dt = \frac{ab}{2}\int_0^{2\pi} 1\,dt = \pi ab$$

### 20.3.5 多连通区域的 Green 公式

前面讨论的 Green 公式适用于单连通区域（无"洞"的区域）。对于**多连通区域**（有一个或多个洞的区域），需要将公式作适当推广。

**多连通区域的 Green 公式**：设 $D$ 是由外边界 $L_0$（正向，逆时针）和内边界 $L_1, L_2, \ldots, L_k$（正向，**顺时针**，即使区域 $D$ 始终在边界的左侧）所围成的多连通区域。若 $P$、$Q$ 在 $D$（含所有边界）上有连续的一阶偏导数，则

$$\iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dx\,dy = \oint_{L_0} P\,dx + Q\,dy + \sum_{i=1}^{k} \oint_{L_i} P\,dx + Q\,dy$$

其中所有边界曲线都取**正向**（区域 $D$ 在边界的左侧）。

**注意**：对于多连通区域的正向规定——外边界取逆时针方向，内边界取顺时针方向。

**割线法（挖洞法）的基本思想**：

当被积函数 $P$、$Q$ 在区域的"洞"内（即某个内边界包围的区域内）不满足条件（例如存在奇点）时，不能直接对整个区域应用 Green 公式。此时可用**割线法**：

1. 在区域内作一条（或多条）割线，连接外边界和内边界，将多连通区域切割为单连通区域
2. 对切割后的单连通区域应用 Green 公式
3. 由于割线被经过了两次（方向相反），其上的积分互相抵消

等价地，可以用另一种直观的方法：**挖洞法**。在奇点周围作一个包含奇点的小闭曲线 $l$（通常取以奇点为圆心的小圆），利用 Green 公式将原积分转化为在这个小曲线上的积分。

**几何直观**：割线法的实质是将多连通区域的"洞"用割线"缝合"，使之变为单连通区域，从而可以使用标准的 Green 公式。

> **例题 20.7** 计算 $\oint_C \dfrac{-y\,dx + x\,dy}{x^2+y^2}$，其中 $C$ 是包围原点的任意一条正向简单闭曲线。

**解**：设 $P = \dfrac{-y}{x^2+y^2}$，$Q = \dfrac{x}{x^2+y^2}$。

首先验证：当 $(x,y) \neq (0,0)$ 时，

$$\frac{\partial P}{\partial y} = \frac{-(x^2+y^2) + y \cdot 2y}{(x^2+y^2)^2} = \frac{y^2 - x^2}{(x^2+y^2)^2}$$

$$\frac{\partial Q}{\partial x} = \frac{(x^2+y^2) - x \cdot 2x}{(x^2+y^2)^2} = \frac{y^2 - x^2}{(x^2+y^2)^2}$$

故 $\dfrac{\partial Q}{\partial x} = \dfrac{\partial P}{\partial y}$（在原点之外）。

但 $P$、$Q$ 在原点没有定义（奇点），不能直接对 $C$ 围成的区域应用 Green 公式得出积分为零的结论。

**挖洞法**：以原点为圆心，取充分小的圆 $l: x^2 + y^2 = \varepsilon^2$（$\varepsilon > 0$ 足够小使 $l$ 完全在 $C$ 内部），取**顺时针方向**（即 $l$ 的正向，使环形区域 $D$ 在边界左侧）。

在环形区域 $D$（$C$ 与 $l$ 之间的区域）上，$P$、$Q$ 有连续偏导数且 $\dfrac{\partial Q}{\partial x} - \dfrac{\partial P}{\partial y} = 0$。

由多连通区域的 Green 公式：

$$0 = \iint_D 0\,dx\,dy = \oint_C P\,dx + Q\,dy + \oint_{l^-} P\,dx + Q\,dy$$

其中 $l^-$ 表示 $l$ 取顺时针方向。因此

$$\oint_C P\,dx + Q\,dy = -\oint_{l^-} P\,dx + Q\,dy = \oint_l P\,dx + Q\,dy$$

其中 $l$ 取逆时针方向。

现在计算小圆 $l$ 上的积分。将 $l$ 参数化为 $x = \varepsilon\cos t$，$y = \varepsilon\sin t$，$t$ 从 $0$ 到 $2\pi$（逆时针）：

$$dx = -\varepsilon\sin t\,dt, \quad dy = \varepsilon\cos t\,dt$$

$$P = \frac{-\varepsilon\sin t}{\varepsilon^2} = \frac{-\sin t}{\varepsilon}, \quad Q = \frac{\varepsilon\cos t}{\varepsilon^2} = \frac{\cos t}{\varepsilon}$$

$$\oint_l P\,dx + Q\,dy = \int_0^{2\pi} \left[\frac{-\sin t}{\varepsilon} \cdot (-\varepsilon\sin t) + \frac{\cos t}{\varepsilon} \cdot \varepsilon\cos t\right] dt$$

$$= \int_0^{2\pi} (\sin^2 t + \cos^2 t)\,dt = \int_0^{2\pi} 1\,dt = 2\pi$$

因此

$$\oint_C \frac{-y\,dx + x\,dy}{x^2+y^2} = 2\pi$$

**注**：此结果与 $C$ 的具体形状无关，只要 $C$ 包围原点。这是因为 $P\,dx + Q\,dy$ 在原点之外满足 $\dfrac{\partial Q}{\partial x} = \dfrac{\partial P}{\partial y}$，故在不包含原点的区域内，积分与路径无关。但若 $C$ 不包围原点，则 $C$ 围成的区域内处处有 $\dfrac{\partial Q}{\partial x} - \dfrac{\partial P}{\partial y} = 0$，由 Green 公式直接得到积分值为 $0$。

从势函数的角度看，$\dfrac{-y\,dx + x\,dy}{x^2+y^2} = d(\arctan\dfrac{y}{x})$，而 $\arctan\dfrac{y}{x}$ 是多值函数（辐角函数），沿包围原点的闭曲线绕一圈后增加 $2\pi$，这正是积分值 $2\pi$ 的来源。 $\square$

---

## 20.4 路径无关与保守场

### 20.4.1 路径无关的概念

**定义**：设 $D$ 是平面上的一个区域，$P(x, y)$、$Q(x, y)$ 在 $D$ 内有定义。如果对于 $D$ 内任意两点 $A$、$B$，曲线积分 $\int_L P\,dx + Q\,dy$ 的值只与 $A$、$B$ 的位置有关，而与连接 $A$、$B$ 的路径 $L$（在 $D$ 内）无关，则称该曲线积分在 $D$ 内**与路径无关**。

### 20.4.2 路径无关的等价条件

**定理**：设 $D$ 是平面上的**单连通区域**，$P(x, y)$、$Q(x, y)$ 在 $D$ 内有连续的一阶偏导数，则以下四个条件等价：

1. $\int_L P\,dx + Q\,dy$ 在 $D$ 内与路径无关

2. 对 $D$ 内任意闭曲线 $C$，$\oint_C P\,dx + Q\,dy = 0$

3. 在 $D$ 内处处有 $\dfrac{\partial Q}{\partial x} = \dfrac{\partial P}{\partial y}$

4. 存在函数 $u(x, y)$ 使得 $du = P\,dx + Q\,dy$（即 $P\,dx + Q\,dy$ 是**全微分**）

**证明要点**：

- $(1) \Leftrightarrow (2)$：由定义直接可得
- $(2) \Leftrightarrow (3)$：由 Green 公式
- $(3) \Leftrightarrow (4)$：$du = P\,dx + Q\,dy$ 意味着 $\dfrac{\partial u}{\partial x} = P$，$\dfrac{\partial u}{\partial y} = Q$，故 $\dfrac{\partial P}{\partial y} = \dfrac{\partial^2 u}{\partial y\partial x} = \dfrac{\partial^2 u}{\partial x\partial y} = \dfrac{\partial Q}{\partial x}$

### 20.4.3 势函数

**定义**：若 $P\,dx + Q\,dy = du$，即 $\dfrac{\partial u}{\partial x} = P$，$\dfrac{\partial u}{\partial y} = Q$，则称 $u(x, y)$ 为向量场 $\mathbf{F} = (P, Q)$ 的**势函数**（或原函数）。此时向量场 $\mathbf{F}$ 称为**保守场**（或有势场）。

**势函数的求法**：

**方法一（折线法）**：利用

$$u(x, y) = \int_{(x_0, y_0)}^{(x, y)} P\,dx + Q\,dy$$

选择方便的路径（如先沿 $x$ 方向，再沿 $y$ 方向）。

**方法二（不定积分法）**：

1. 由 $\dfrac{\partial u}{\partial x} = P$，得 $u = \int P\,dx + \varphi(y)$（对 $x$ 积分，$\varphi(y)$ 待定）

2. 对上式关于 $y$ 求偏导，令其等于 $Q$，解出 $\varphi(y)$

> **例题 20.8** 验证 $(2x + y)\,dx + (x + 2y)\,dy$ 是全微分，并求其势函数。

**解**：$P = 2x + y$，$Q = x + 2y$。

$$\frac{\partial P}{\partial y} = 1, \quad \frac{\partial Q}{\partial x} = 1$$

由于 $\dfrac{\partial P}{\partial y} = \dfrac{\partial Q}{\partial x}$，故 $P\,dx + Q\,dy$ 是全微分。

**求势函数**：

由 $\dfrac{\partial u}{\partial x} = 2x + y$，得

$$u = \int (2x + y)\,dx = x^2 + xy + \varphi(y)$$

对 $y$ 求偏导：

$$\frac{\partial u}{\partial y} = x + \varphi'(y) = Q = x + 2y$$

故 $\varphi'(y) = 2y$，$\varphi(y) = y^2 + C$。

因此势函数为 $u(x, y) = x^2 + xy + y^2 + C$。

### 20.4.4 全微分方程的求解

**全微分方程**：形如 $P(x, y)\,dx + Q(x, y)\,dy = 0$ 的方程，若 $P\,dx + Q\,dy$ 是全微分，则称为全微分方程（或恰当方程）。

**解法**：若 $du = P\,dx + Q\,dy$，则方程变为 $du = 0$，其通解为 $u(x, y) = C$。

> **例题 20.9** 解方程 $(3x^2 + 6xy^2)\,dx + (6x^2y + 4y^3)\,dy = 0$。

**解**：$P = 3x^2 + 6xy^2$，$Q = 6x^2y + 4y^3$。

$$\frac{\partial P}{\partial y} = 12xy, \quad \frac{\partial Q}{\partial x} = 12xy$$

验证：$\dfrac{\partial P}{\partial y} = \dfrac{\partial Q}{\partial x}$，故为全微分方程。

求势函数：

$$u = \int P\,dx = \int (3x^2 + 6xy^2)\,dx = x^3 + 3x^2y^2 + \varphi(y)$$

$$\frac{\partial u}{\partial y} = 6x^2y + \varphi'(y) = Q = 6x^2y + 4y^3$$

故 $\varphi'(y) = 4y^3$，$\varphi(y) = y^4$。

通解为 $x^3 + 3x^2y^2 + y^4 = C$。

---

## 本章小结

1. **第一类曲线积分**（对弧长）：$\int_L f(x, y)\,ds$
   - 物理意义：曲线的质量（当 $f$ 为线密度时）
   - 计算：参数化后 $\int_\alpha^\beta f(x(t), y(t))\sqrt{x'^2 + y'^2}\,dt$
   - 与曲线方向无关

2. **第二类曲线积分**（对坐标）：$\int_L P\,dx + Q\,dy$
   - 物理意义：变力做功
   - 计算：参数化后 $\int_\alpha^\beta [Px'(t) + Qy'(t)]\,dt$
   - 与曲线方向有关

3. **Green 公式**：
   $$\oint_L P\,dx + Q\,dy = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dx\,dy$$
   - 条件：$D$ 有界，$L$ 为正向边界，$P$、$Q$ 有连续偏导数
   - 应用：计算曲线积分、求面积 $S = \dfrac{1}{2}\oint_L x\,dy - y\,dx$

4. **路径无关的条件**（单连通区域）：
   $$\frac{\partial Q}{\partial x} = \frac{\partial P}{\partial y}$$

5. **势函数**：若 $du = P\,dx + Q\,dy$，则 $u$ 是势函数。
   - 积分与路径无关时：$\int_A^B P\,dx + Q\,dy = u(B) - u(A)$

6. **全微分方程**：$P\,dx + Q\,dy = 0$，通解为 $u(x, y) = C$。

---

## 深度学习应用

曲线积分的概念在深度学习的优化理论中有着直接的对应。理解这些联系有助于从数学上更深刻地把握梯度下降的本质。

### 优化路径与损失曲面

神经网络训练的本质是在参数空间中寻找损失函数 $L(\theta)$ 的最小值。训练过程中参数 $\theta$ 随时间的变化轨迹构成参数空间中的一条曲线 $\Gamma$：

$$\theta(t): \theta_0 \to \theta_1 \to \cdots \to \theta_T$$

这条优化路径上的总损失变化可以用曲线积分（第二类）表达：

$$\Delta L = \int_{\Gamma} \nabla L(\theta) \cdot d\theta = \int_0^T \nabla L(\theta(t)) \cdot \dot{\theta}(t)\,dt$$

对于梯度下降，$\dot{\theta}(t) = -\nabla L(\theta(t))$，代入得：

$$\Delta L = -\int_0^T \|\nabla L(\theta(t))\|^2\,dt \leq 0$$

这从曲线积分的角度严格说明了梯度下降每步都在减小损失。

### 保守场与可积性

**损失函数是势函数**。梯度场 $\mathbf{F}(\theta) = \nabla L(\theta)$ 是保守场，因为 $L(\theta)$ 就是它的势函数，满足：

$$\frac{\partial F_i}{\partial \theta_j} = \frac{\partial^2 L}{\partial \theta_i \partial \theta_j} = \frac{\partial F_j}{\partial \theta_i}$$

（即 Hessian 矩阵的对称性）。

**路径无关性**：由于梯度场是保守场，从 $\theta_A$ 到 $\theta_B$ 的损失变化只取决于端点，与路径无关：

$$\int_\Gamma \nabla L \cdot d\theta = L(\theta_B) - L(\theta_A)$$

这对应于 20.4 节的定理：在保守场中，曲线积分与路径无关。

### 路径无关性与全局最优

**凸函数的情形**：当损失曲面是凸函数时（如 MSE 损失），任何梯度下降路径都保证收敛到全局最优点 $\theta^*$。数学上，凸性等价于 Hessian 矩阵半正定 $\nabla^2 L \succeq 0$，此时损失曲面无鞍点和局部极小值，路径的终点唯一。

**非凸函数的情形**：深度神经网络的损失曲面通常是非凸的，存在多个局部极小值和鞍点。不同的初始化点 $\theta_0$ 和学习率会导致不同的优化路径，最终收敛到不同的局部解。路径选择（即优化算法的选择）对结果有显著影响，这与一般场中曲线积分的路径依赖性完全对应。

### 代码示例

下面的示例演示如何收集神经网络训练过程中的参数轨迹，即参数空间中的优化路径：

```python
import torch
import torch.nn as nn

# 优化路径的可视化数据收集
def collect_optimization_path(model, loss_fn, x, y, lr=0.01, steps=100):
    """收集优化过程中的参数轨迹"""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    path = []
    for _ in range(steps):
        # 记录当前参数位置
        params = torch.cat([p.flatten() for p in model.parameters()])
        path.append(params.detach().clone())

        # 梯度下降步
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    return torch.stack(path)  # [steps, n_params]

# 简单示例
model = nn.Linear(2, 1)
x = torch.randn(10, 2)
y = torch.randn(10, 1)
path = collect_optimization_path(model, nn.MSELoss(), x, y)
print(f"优化路径形状: {path.shape}")
```

`path` 张量的每一行是参数空间中的一个点，相邻行之间的差向量就是一步梯度下降对应的 $d\theta$，整个序列构成了参数空间中的离散化曲线。

**联系总结**：

| 曲线积分概念 | 深度学习对应 |
|---|---|
| 势函数 $u(x, y)$ | 损失函数 $L(\theta)$ |
| 保守场 $\nabla u$ | 梯度场 $\nabla L$ |
| 路径无关性 | 损失变化只取决于初末参数值 |
| 凸区域中路径积分 | 凸损失曲面上梯度下降收敛到全局最优 |
| 参数化曲线 | 参数随训练步数的轨迹 |

---

## 练习题

**1.** 计算 $\int_L (x + y)\,ds$，其中 $L$ 是圆周 $x^2 + y^2 = 4$ 在第一象限的部分。

**2.** 计算 $\int_L xy\,dx + (x^2 + y^2)\,dy$，其中 $L$ 是从点 $(0, 0)$ 沿抛物线 $y^2 = x$ 到点 $(1, 1)$。

**3.** 利用 Green 公式计算 $\oint_L (x^2 + y^2)\,dx + (x^2 - y^2)\,dy$，其中 $L$ 是由 $y = x^2$ 和 $y = x$ 围成区域的正向边界。

**4.** 验证 $\dfrac{-y}{x^2 + y^2}\,dx + \dfrac{x}{x^2 + y^2}\,dy$ 在不包含原点的单连通区域内与路径无关，并求其势函数。

**5.** 解全微分方程 $(2xy + 3)\,dx + (x^2 + 4y)\,dy = 0$。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.** 第一象限的圆弧参数化为 $x = 2\cos t$，$y = 2\sin t$，$t \in [0, \dfrac{\pi}{2}]$。

$$ds = 2\,dt$$

$$\int_L (x + y)\,ds = \int_0^{\pi/2} (2\cos t + 2\sin t) \cdot 2\,dt = 4\int_0^{\pi/2} (\cos t + \sin t)\,dt$$

$$= 4[\sin t - \cos t]_0^{\pi/2} = 4[(1 - 0) - (0 - 1)] = 4 \cdot 2 = 8$$

---

**2.** 曲线 $y^2 = x$ 参数化为 $x = t^2$，$y = t$，$t$ 从 $0$ 到 $1$。

$$dx = 2t\,dt, \quad dy = dt$$

$$\int_L xy\,dx + (x^2 + y^2)\,dy = \int_0^1 [t^2 \cdot t \cdot 2t + (t^4 + t^2) \cdot 1]\,dt$$

$$= \int_0^1 (2t^4 + t^4 + t^2)\,dt = \int_0^1 (3t^4 + t^2)\,dt = \left[\frac{3t^5}{5} + \frac{t^3}{3}\right]_0^1 = \frac{3}{5} + \frac{1}{3} = \frac{14}{15}$$

---

**3.** $P = x^2 + y^2$，$Q = x^2 - y^2$。

$$\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} = 2x - 2y$$

区域 $D$ 由 $y = x^2$ 和 $y = x$（$0 \leq x \leq 1$）围成。

$$\oint_L P\,dx + Q\,dy = \iint_D (2x - 2y)\,dx\,dy = \int_0^1 dx \int_{x^2}^{x} (2x - 2y)\,dy$$

$$= \int_0^1 \left[2xy - y^2\right]_{x^2}^{x} dx = \int_0^1 [(2x^2 - x^2) - (2x^3 - x^4)]\,dx$$

$$= \int_0^1 (x^2 - 2x^3 + x^4)\,dx = \left[\frac{x^3}{3} - \frac{x^4}{2} + \frac{x^5}{5}\right]_0^1 = \frac{1}{3} - \frac{1}{2} + \frac{1}{5} = \frac{10 - 15 + 6}{30} = \frac{1}{30}$$

---

**4.** 设 $P = \dfrac{-y}{x^2 + y^2}$，$Q = \dfrac{x}{x^2 + y^2}$。

$$\frac{\partial P}{\partial y} = \frac{-(x^2 + y^2) + y \cdot 2y}{(x^2 + y^2)^2} = \frac{y^2 - x^2}{(x^2 + y^2)^2}$$

$$\frac{\partial Q}{\partial x} = \frac{(x^2 + y^2) - x \cdot 2x}{(x^2 + y^2)^2} = \frac{y^2 - x^2}{(x^2 + y^2)^2}$$

由于 $\dfrac{\partial P}{\partial y} = \dfrac{\partial Q}{\partial x}$（在 $(x, y) \neq (0, 0)$ 处），故在不包含原点的单连通区域内与路径无关。

**求势函数**：

$$u = \int Q\,dy = \int \frac{x}{x^2 + y^2}\,dy = \arctan\frac{y}{x} + \varphi(x)$$

$$\frac{\partial u}{\partial x} = \frac{-y/x^2}{1 + y^2/x^2} + \varphi'(x) = \frac{-y}{x^2 + y^2} + \varphi'(x) = P = \frac{-y}{x^2 + y^2}$$

故 $\varphi'(x) = 0$，$\varphi(x) = C$。

势函数为 $u = \arctan\dfrac{y}{x} + C$。

---

**5.** $P = 2xy + 3$，$Q = x^2 + 4y$。

$$\frac{\partial P}{\partial y} = 2x, \quad \frac{\partial Q}{\partial x} = 2x$$

验证：$\dfrac{\partial P}{\partial y} = \dfrac{\partial Q}{\partial x}$，是全微分方程。

求势函数：

$$u = \int P\,dx = \int (2xy + 3)\,dx = x^2y + 3x + \varphi(y)$$

$$\frac{\partial u}{\partial y} = x^2 + \varphi'(y) = Q = x^2 + 4y$$

故 $\varphi'(y) = 4y$，$\varphi(y) = 2y^2$。

通解为 $x^2y + 3x + 2y^2 = C$。

</details>
