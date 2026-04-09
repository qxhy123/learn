# 第3章：凸集与凸函数

> **前置知识**：第1章（向量与矩阵）、第2章（多元微分学）
>
> **本章难度**：★★★☆☆
>
> **预计学习时间**：5-6 小时

---

## 学习目标

学完本章后，你将能够：

- 给出凸集的精确定义，判断常见集合（球、多面体、超平面、半空间）的凸性，并利用凸包和凸锥构造新的凸集
- 掌握保凸运算（交集、仿射变换、透视映射、线性分数变换），在不求解复杂不等式的情况下快速判断集合凸性
- 从定义出发，利用一阶条件和二阶条件（Hessian 半正定）判断多元函数的凸性，并能区分凸函数、严格凸函数和强凸函数
- 理解并应用 Jensen 不等式，将其作为推导机器学习中期望不等式的统一工具
- 分析深度学习中常用损失函数的凸性，理解非凸优化的挑战，以及凸性视角对 ReLU 激活函数的解读

---

## 3.1 凸集的定义与例子

### 3.1.1 凸集的定义

**定义 3.1（凸集）**：集合 $C \subseteq \mathbb{R}^n$ 称为**凸集（convex set）**，若对任意 $\mathbf{x}, \mathbf{y} \in C$ 以及任意 $\theta \in [0, 1]$，有

$$\theta \mathbf{x} + (1 - \theta) \mathbf{y} \in C$$

**几何直觉**：凸集中任意两点的连线段仍在集合内。换句话说，集合"没有凹陷"或"没有洞"。

点 $\theta \mathbf{x} + (1-\theta)\mathbf{y}$（$\theta \in [0,1]$）称为 $\mathbf{x}$ 与 $\mathbf{y}$ 的**凸组合**，当 $\theta$ 从 0 变化到 1 时，它在线段 $[\mathbf{y}, \mathbf{x}]$ 上移动。

更一般地，$k$ 个点 $\mathbf{x}_1, \ldots, \mathbf{x}_k$ 的凸组合是

$$\sum_{i=1}^{k} \theta_i \mathbf{x}_i, \quad \theta_i \geq 0,\; \sum_{i=1}^{k} \theta_i = 1$$

> **直觉提示**：凸集是"胖"的——不会有尖角向内刺，不会有裂缝。典型的非凸集有月牙形、环形、带孔的区域。

### 3.1.2 常见凸集

**例 3.1（仿射集与超平面）**

超平面 $H = \{\mathbf{x} \mid \mathbf{a}^T \mathbf{x} = b\}$ 是凸集（事实上是仿射集）。

半空间 $\{\mathbf{x} \mid \mathbf{a}^T \mathbf{x} \leq b\}$ 也是凸集。

*验证*：设 $\mathbf{a}^T \mathbf{x} \leq b$ 且 $\mathbf{a}^T \mathbf{y} \leq b$，则对 $\theta \in [0,1]$：
$$\mathbf{a}^T[\theta \mathbf{x} + (1-\theta)\mathbf{y}] = \theta \mathbf{a}^T \mathbf{x} + (1-\theta)\mathbf{a}^T \mathbf{y} \leq \theta b + (1-\theta)b = b \checkmark$$

**例 3.2（球与椭球）**

欧氏球 $B(\mathbf{x}_0, r) = \{\mathbf{x} \mid \|\mathbf{x} - \mathbf{x}_0\|_2 \leq r\}$ 是凸集。

椭球 $\mathcal{E} = \{\mathbf{x} \mid (\mathbf{x} - \mathbf{x}_0)^T P^{-1} (\mathbf{x} - \mathbf{x}_0) \leq 1\}$（$P \succ 0$）也是凸集。

**例 3.3（多面体）**

多面体（polyhedron）$\mathcal{P} = \{\mathbf{x} \mid A\mathbf{x} \preceq \mathbf{b},\; C\mathbf{x} = \mathbf{d}\}$ 是有限个半空间与超平面的交，因此是凸集。线性规划的可行域即为多面体。

**例 3.4（正半定锥）**

对称正半定矩阵的集合 $\mathbb{S}^n_+ = \{X \in \mathbb{R}^{n \times n} \mid X = X^T,\; X \succeq 0\}$ 是凸集（半定规划的基础）。

**非凸集例子**：

- 有限点集（超过 1 个点时）
- 圆环 $\{(x,y) \mid 1 \leq x^2 + y^2 \leq 4\}$（含洞）
- 月牙形区域（有凹陷）

### 3.1.3 凸包

**定义 3.2（凸包）**：集合 $S$ 的**凸包（convex hull）** $\text{conv}(S)$ 是包含 $S$ 的最小凸集，等价地：

$$\text{conv}(S) = \left\{ \sum_{i=1}^{k} \theta_i \mathbf{x}_i \;\bigg|\; \mathbf{x}_i \in S,\; \theta_i \geq 0,\; \sum_{i=1}^k \theta_i = 1,\; k \geq 1 \right\}$$

**直觉**：$\text{conv}(S)$ 是将 $S$ 用"橡皮筋"包裹起来所形成的区域。

**例 3.5**：$S = \{(0,0),\,(1,0),\,(0,1)\}$（三个点），则 $\text{conv}(S)$ 是以这三点为顶点的三角形（含内部）。

### 3.1.4 凸锥

**定义 3.3（锥）**：集合 $K \subseteq \mathbb{R}^n$ 称为**锥（cone）**，若对任意 $\mathbf{x} \in K$ 和 $\lambda \geq 0$，有 $\lambda \mathbf{x} \in K$。

**定义 3.4（凸锥）**：既是凸集又是锥的集合称为**凸锥（convex cone）**。等价条件：对任意 $\mathbf{x}, \mathbf{y} \in K$ 和 $\lambda, \mu \geq 0$，有 $\lambda \mathbf{x} + \mu \mathbf{y} \in K$。

**常见凸锥**：

| 名称 | 定义 | 应用 |
|------|------|------|
| 非负象限 | $\mathbb{R}^n_+ = \{\mathbf{x} \mid x_i \geq 0\}$ | 线性规划 |
| 二阶锥（冰淇淋锥）| $\{(\mathbf{x}, t) \mid \|\mathbf{x}\|_2 \leq t\}$ | SOCP |
| 正半定锥 | $\mathbb{S}^n_+$ | SDP |
| 法锥/切锥 | 由约束集在边界处定义 | 最优性条件 |

---

## 3.2 保凸运算

判断一个集合是否为凸集，有时不需要直接验证定义，而是利用**保凸运算（convexity-preserving operations）**——这些运算将凸集映射为凸集。

### 3.2.1 交集

**命题 3.1**：任意多个（有限或无限）凸集的**交集**仍是凸集。

*证明*：设 $\{C_\alpha\}$ 均为凸集，$C = \bigcap_\alpha C_\alpha$。对任意 $\mathbf{x}, \mathbf{y} \in C$ 和 $\theta \in [0,1]$，由于 $\mathbf{x}, \mathbf{y} \in C_\alpha$（对所有 $\alpha$），利用每个 $C_\alpha$ 的凸性知 $\theta\mathbf{x}+(1-\theta)\mathbf{y} \in C_\alpha$，从而 $\theta\mathbf{x}+(1-\theta)\mathbf{y} \in C$。$\square$

> **注意**：凸集的**并集**通常不是凸集！例如两个不相交的圆盘的并。

**应用**：多面体 = 有限个半空间的交，故多面体是凸集。

### 3.2.2 仿射变换

**命题 3.2**：设 $C \subseteq \mathbb{R}^n$ 为凸集，$f(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ 为仿射映射，则：
- **像** $f(C) = \{A\mathbf{x} + \mathbf{b} \mid \mathbf{x} \in C\}$ 是凸集
- **原像** $f^{-1}(C) = \{\mathbf{x} \mid A\mathbf{x} + \mathbf{b} \in C\}$ 是凸集

**推论**：

- 凸集的**缩放**（$\lambda C$）和**平移**（$C + \mathbf{a}$）是凸集
- 凸集的**线性投影**是凸集
- 两个凸集的**Minkowski 和** $C_1 + C_2 = \{\mathbf{x} + \mathbf{y} \mid \mathbf{x} \in C_1, \mathbf{y} \in C_2\}$ 是凸集

### 3.2.3 透视映射与线性分数变换

**透视映射** $P: \mathbb{R}^{n+1} \to \mathbb{R}^n$，定义为

$$P(\mathbf{x}, t) = \frac{\mathbf{x}}{t}, \quad \text{dom}(P) = \{(\mathbf{x}, t) \mid t > 0\}$$

**命题 3.3**：若 $C \subseteq \text{dom}(P)$ 是凸集，则 $P(C)$ 是凸集；若 $D \subseteq \mathbb{R}^n$ 是凸集，则 $P^{-1}(D)$ 是凸集。

**线性分数变换**（透视映射与仿射映射的复合）同样保持凸性，这在射影几何与优化中有重要应用。

### 3.2.4 分离超平面定理

**定理 3.1（分离超平面定理）**：设 $C, D \subseteq \mathbb{R}^n$ 是两个非空不相交的凸集，则存在 $\mathbf{a} \neq \mathbf{0}$ 和 $b$ 使得

$$\mathbf{a}^T \mathbf{x} \leq b \;\forall\, \mathbf{x} \in C, \quad \mathbf{a}^T \mathbf{y} \geq b \;\forall\, \mathbf{y} \in D$$

超平面 $\{\mathbf{x} \mid \mathbf{a}^T \mathbf{x} = b\}$ 称为 $C$ 与 $D$ 的**分离超平面**。

**支撑超平面定理**：设 $C$ 是凸集，$\mathbf{x}_0$ 在 $C$ 的边界上，则存在 $\mathbf{a} \neq \mathbf{0}$ 使得 $\mathbf{a}^T \mathbf{x} \leq \mathbf{a}^T \mathbf{x}_0$ 对所有 $\mathbf{x} \in C$ 成立。这个超平面称为 $C$ 在 $\mathbf{x}_0$ 处的**支撑超平面**，是凸分析中联系几何与对偶理论的桥梁。

---

## 3.3 凸函数的定义与判定

### 3.3.1 凸函数的定义

**定义 3.5（凸函数）**：函数 $f: \mathbb{R}^n \to \mathbb{R}$（定义域 $\text{dom}(f)$ 为凸集）称为**凸函数（convex function）**，若对任意 $\mathbf{x}, \mathbf{y} \in \text{dom}(f)$ 和 $\theta \in [0,1]$，有

$$\boxed{f(\theta \mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y})}$$

**几何直觉**：函数图像上任意两点连线段在函数图像的**上方**（或与其相切）。

- 若不等号严格成立（$\theta \in (0,1)$，$\mathbf{x} \neq \mathbf{y}$）：**严格凸函数**
- 若 $-f$ 是凸函数：$f$ 是**凹函数（concave function）**

**上图集（epigraph）**：$f$ 是凸函数 $\iff$ 上图集 $\text{epi}(f) = \{(\mathbf{x}, t) \mid f(\mathbf{x}) \leq t\}$ 是凸集。

### 3.3.2 常见凸函数与凹函数

| 函数 | 凸/凹 | 条件 |
|------|-------|------|
| 仿射函数 $\mathbf{a}^T \mathbf{x} + b$ | 既凸又凹 | — |
| 指数函数 $e^{ax}$ | 凸 | 任意 $a$ |
| 幂函数 $x^p$（$x > 0$）| 凸 | $p \geq 1$ 或 $p \leq 0$ |
| 幂函数 $x^p$（$x > 0$）| 凹 | $0 \leq p \leq 1$ |
| 对数函数 $\log x$（$x > 0$）| 凹 | — |
| 二次型 $\mathbf{x}^T P \mathbf{x}$ | 凸 | $P \succeq 0$ |
| 范数 $\|\mathbf{x}\|$ | 凸 | 任意范数 |
| 最大值 $\max_i x_i$ | 凸 | — |
| 负熵 $x \log x$（$x > 0$）| 凸 | — |
| 对数行列式 $\log\det X$（$X \succ 0$）| 凹 | — |

### 3.3.3 一阶条件（可微情形）

**定理 3.2（一阶凸性条件）**：设 $f$ 在凸集上可微，则 $f$ 是凸函数 $\iff$ 对任意 $\mathbf{x}, \mathbf{y} \in \text{dom}(f)$：

$$\boxed{f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x})}$$

**几何含义**：凸函数在任意点处的**切平面**是函数的**全局下界**。这是凸优化的核心性质：

- 若 $\nabla f(\mathbf{x}^*) = \mathbf{0}$，则对所有 $\mathbf{y}$：$f(\mathbf{y}) \geq f(\mathbf{x}^*)$，即 $\mathbf{x}^*$ 是**全局最小值**。

*证明思路（$\Rightarrow$）*：由凸性定义，对 $\theta \in (0,1)$：
$$f(\mathbf{x} + \theta(\mathbf{y}-\mathbf{x})) \leq (1-\theta)f(\mathbf{x}) + \theta f(\mathbf{y})$$
整理得 $\frac{f(\mathbf{x} + \theta(\mathbf{y}-\mathbf{x})) - f(\mathbf{x})}{\theta} \leq f(\mathbf{y}) - f(\mathbf{x})$，令 $\theta \to 0^+$ 即得。

### 3.3.4 二阶条件（二次可微情形）

**定理 3.3（二阶凸性条件）**：设 $f$ 在凸集上二次连续可微，则

$$f \text{ 是凸函数} \iff \nabla^2 f(\mathbf{x}) \succeq 0 \;\; \forall\, \mathbf{x} \in \text{dom}(f)$$

即 Hessian 矩阵**半正定**。若 $\nabla^2 f(\mathbf{x}) \succ 0$（正定），则 $f$ 严格凸。

**一维情形**：$f''(x) \geq 0$ 等价于 $f$ 是凸函数（函数曲线"开口向上"）。

**例 3.6（验证二次函数的凸性）**：

$$f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T P \mathbf{x} + \mathbf{q}^T \mathbf{x} + r, \quad \nabla^2 f = P$$

因此 $f$ 凸 $\iff P \succeq 0$，$f$ 严格凸 $\iff P \succ 0$。

**例 3.7（验证 Softmax 损失的凸性）**：

交叉熵损失 $\ell(\mathbf{w}) = -\log \sigma(\mathbf{w}^T \mathbf{x})$（$\sigma$ 为 sigmoid）对 $\mathbf{w}$ 是凸函数，可通过计算 Hessian 并验证其半正定性来证明。

### 3.3.5 保凸的函数运算

如同集合的情形，函数也有保凸运算：

| 运算 | 条件 | 结果 |
|------|------|------|
| 非负线性组合 $\sum_i w_i f_i$ | $w_i \geq 0$，$f_i$ 凸 | 凸 |
| 逐点上确界 $\sup_\alpha f_\alpha(\mathbf{x})$ | 每个 $f_\alpha$ 凸 | 凸 |
| 仿射复合 $f(A\mathbf{x}+\mathbf{b})$ | $f$ 凸 | 凸 |
| 凸函数的凸单调非减复合 $g(f(\mathbf{x}))$ | $g$ 凸且非减，$f$ 凸 | 凸 |
| 下确界 $g(\mathbf{x}) = \inf_{\mathbf{y} \in C} f(\mathbf{x}, \mathbf{y})$ | $f$ 对 $(\mathbf{x},\mathbf{y})$ 联合凸，$C$ 凸 | 凸 |

---

## 3.4 凸函数的性质

### 3.4.1 Jensen 不等式

**定理 3.4（Jensen 不等式）**：设 $f$ 是凸函数，$\mathbf{x}_1, \ldots, \mathbf{x}_k \in \text{dom}(f)$，$\theta_1, \ldots, \theta_k \geq 0$，$\sum_i \theta_i = 1$，则

$$\boxed{f\!\left(\sum_{i=1}^k \theta_i \mathbf{x}_i\right) \leq \sum_{i=1}^k \theta_i f(\mathbf{x}_i)}$$

**概率形式**：若 $X$ 是随机变量，$f$ 是凸函数，则

$$\boxed{f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]}$$

**直觉**：对凸函数取期望，"先平均再取值"不超过"先取值再平均"。

**重要推论**（由 Jensen 不等式直接导出）：

- **AM-GM 不等式**：$\frac{x_1 + x_2}{2} \geq \sqrt{x_1 x_2}$（取 $f = -\log$，$f$ 凸）
- **对数和不等式**：$\sum_i a_i \log\frac{a_i}{b_i} \geq \left(\sum_i a_i\right)\log\frac{\sum_i a_i}{\sum_i b_i}$
- **KL 散度非负性**：$D_{\text{KL}}(P \| Q) = \mathbb{E}_P\left[\log\frac{P(X)}{Q(X)}\right] \geq 0$（取 $f = -\log$，凸函数，应用 Jensen）

**例 3.8（EM 算法的理论基础）**：

在 EM 算法中，利用 Jensen 不等式构造对数似然的下界：

$$\log p(\mathbf{x}; \theta) = \log \sum_{\mathbf{z}} p(\mathbf{x}, \mathbf{z}; \theta) = \log \mathbb{E}_q\left[\frac{p(\mathbf{x}, \mathbf{z}; \theta)}{q(\mathbf{z})}\right] \geq \mathbb{E}_q\left[\log \frac{p(\mathbf{x}, \mathbf{z}; \theta)}{q(\mathbf{z})}\right]$$

其中利用了 $\log$ 是凹函数，所以 $\mathbb{E}[\log X] \leq \log \mathbb{E}[X]$（Jensen 不等式取反方向）。

### 3.4.2 下水平集的凸性

**命题 3.4**：若 $f$ 是凸函数，则其**下水平集（sublevel set）**

$$L_\alpha = \{\mathbf{x} \mid f(\mathbf{x}) \leq \alpha\}$$

对任意 $\alpha$ 都是凸集。

*证明*：设 $\mathbf{x}, \mathbf{y} \in L_\alpha$，即 $f(\mathbf{x}) \leq \alpha$，$f(\mathbf{y}) \leq \alpha$。由凸性：
$$f(\theta\mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y}) \leq \theta\alpha + (1-\theta)\alpha = \alpha$$
故 $\theta\mathbf{x}+(1-\theta)\mathbf{y} \in L_\alpha$。$\square$

> **注意**：逆命题不成立——下水平集是凸集不能推出 $f$ 是凸函数（拟凸函数即满足此条件）。

### 3.4.3 局部最优即全局最优

**定理 3.5**：凸函数的**局部最优解（local minimizer）**一定是**全局最优解（global minimizer）**。

*证明*：设 $\mathbf{x}^*$ 是局部最优解，即存在 $r > 0$，对所有 $\|\mathbf{y} - \mathbf{x}^*\| < r$ 有 $f(\mathbf{y}) \geq f(\mathbf{x}^*)$。假设存在 $\mathbf{z}$ 使得 $f(\mathbf{z}) < f(\mathbf{x}^*)$，取 $\mathbf{y} = (1-\theta)\mathbf{x}^* + \theta\mathbf{z}$（$\theta > 0$ 足够小使得 $\|\mathbf{y} - \mathbf{x}^*\| < r$），由凸性：
$$f(\mathbf{y}) \leq (1-\theta)f(\mathbf{x}^*) + \theta f(\mathbf{z}) < f(\mathbf{x}^*)$$
与局部最优矛盾。$\square$

这是凸优化最重要的性质，它保证了梯度下降不会陷入"坏"的局部最优。

### 3.4.4 凸函数的连续性

**命题 3.5**：凸函数在其定义域的**相对内部**上是局部 Lipschitz 连续的，从而几乎处处可微。

这保证了即使凸函数不是处处可微（如 $|x|$ 在 $x=0$ 处），它的次梯度（subgradient）仍然存在，优化算法依然适用。

---

## 3.5 强凸函数与光滑函数

强凸性和光滑性是分析优化算法收敛速度的两个关键参数，共同决定了问题的"难度"。

### 3.5.1 强凸函数

**定义 3.6（强凸函数）**：函数 $f$ 称为**$m$-强凸函数（$m$-strongly convex）**（$m > 0$），若对任意 $\mathbf{x}, \mathbf{y} \in \text{dom}(f)$ 和 $\theta \in [0,1]$：

$$f(\theta\mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y}) - \frac{m}{2}\theta(1-\theta)\|\mathbf{x}-\mathbf{y}\|^2$$

等价地（可微情形），$f$ 是 $m$-强凸 $\iff$：

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x}) + \frac{m}{2}\|\mathbf{y}-\mathbf{x}\|^2$$

等价地（二次可微情形）：$\nabla^2 f(\mathbf{x}) \succeq mI$，即 Hessian 的最小特征值 $\geq m$。

**直觉**：强凸函数被某个"开口为 $m$ 的抛物面"从下方托住，函数"足够弯曲"。

**重要性质**：

- $m$-强凸函数有**唯一全局最小值**
- 强凸函数的最小值附近是"碗形"的，不存在平坦谷底

**例 3.9**：$f(\mathbf{x}) = \|\mathbf{x}\|^2$ 是 $2$-强凸（$\nabla^2 f = 2I$）；$f(\mathbf{x}) = \|\mathbf{x}\|^2 + \mathbf{w}^T\mathbf{x}$ 也是 $2$-强凸。

加 $L_2$ 正则项 $\frac{\lambda}{2}\|\mathbf{w}\|^2$（$\lambda > 0$）可将非强凸损失变为 $\lambda$-强凸，这正是 L2 正则化的优化理论依据。

### 3.5.2 光滑函数（Lipschitz 梯度）

**定义 3.7（$L$-光滑函数）**：可微函数 $f$ 称为**$L$-光滑（$L$-smooth）**，若其梯度满足 Lipschitz 条件：

$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\| \quad \forall\, \mathbf{x}, \mathbf{y}$$

等价地（二次可微情形）：$\nabla^2 f(\mathbf{x}) \preceq LI$，即 Hessian 最大特征值 $\leq L$。

等价地（二次上界）：

$$f(\mathbf{y}) \leq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x}) + \frac{L}{2}\|\mathbf{y}-\mathbf{x}\|^2$$

**直觉**：光滑函数的梯度变化不太剧烈，函数不会"太尖"——Hessian 的最大特征值有界。

**与步长选择的关系**：若 $f$ 是 $L$-光滑的，梯度下降步长 $\eta \leq \frac{1}{L}$ 可保证每步损失不增加（下降引理）。

### 3.5.3 条件数与收敛速度

设 $f$ 同时满足 $m$-强凸和 $L$-光滑，则 Hessian 的特征值满足 $m \leq \lambda_i(\nabla^2 f) \leq L$。

**定义 3.8（条件数）**：

$$\kappa = \frac{L}{m} \geq 1$$

**条件数的直觉**：

- $\kappa = 1$：等值线是正圆，梯度方向总指向最小值，收敛最快
- $\kappa \gg 1$：等值线是极扁椭圆，梯度方向偏离最优方向，收敛缓慢（"之"字形路径）

**梯度下降收敛率**：

$$f(\mathbf{x}^{(k)}) - f^* \leq \left(1 - \frac{1}{\kappa}\right)^k [f(\mathbf{x}^{(0)}) - f^*]$$

条件数越大，收敛越慢。当 $\kappa = 10^4$ 时，误差减小到原来 $10^{-4}$ 需要约 $10^4$ 步；当 $\kappa = 2$ 时仅需约 30 步。

### 3.5.4 强凸性、光滑性与凸性的关系

```
强凸（m > 0） ⊂ 严格凸 ⊂ 凸
光滑（L < ∞）+ 凸 ⊂ 凸
强凸 + 光滑：条件数 κ = L/m 决定收敛速度
```

| 性质 | 条件 | 推论 |
|------|------|------|
| 凸 | $f(\lambda\mathbf{x}+(1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x})+(1-\lambda)f(\mathbf{y})$ | 局部最优=全局最优 |
| 严格凸 | 不等号严格成立 | 最多一个全局最优点 |
| 强凸（$m>0$）| Hessian $\succeq mI$ | 唯一全局最优，线性收敛 |
| 光滑（$L$）| Hessian $\preceq LI$ | 步长 $1/L$ 安全，下降引理成立 |

---

## 本章小结

| 概念 | 定义/判定 | 关键性质 |
|------|-----------|---------|
| 凸集 | $\theta\mathbf{x}+(1-\theta)\mathbf{y} \in C$，$\forall\,\mathbf{x},\mathbf{y}\in C$，$\theta\in[0,1]$ | 交集保凸，仿射变换保凸 |
| 凸包 | 包含集合的最小凸集 | $\text{conv}(S)$ = 所有凸组合的集合 |
| 凸锥 | 凸集 + 对 $\lambda\geq 0$ 封闭 | 非负象限、二阶锥、正半定锥 |
| 凸函数（定义）| $f(\theta\mathbf{x}+(1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x})+(1-\theta)f(\mathbf{y})$ | 弦在图像上方 |
| 凸函数（一阶）| $f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x})$ | 切平面是全局下界 |
| 凸函数（二阶）| $\nabla^2 f(\mathbf{x}) \succeq 0$ | Hessian 半正定 |
| Jensen 不等式 | $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$ | 推导 KL 散度非负、EM 算法 |
| 下水平集 | $\{f(\mathbf{x}) \leq \alpha\}$ 是凸集 | 凸函数的直接推论 |
| 强凸（$m$）| $\nabla^2 f \succeq mI$ | 唯一最优解，线性收敛 |
| 光滑（$L$）| $\nabla^2 f \preceq LI$，梯度 Lipschitz | 安全步长 $1/L$ |
| 条件数 | $\kappa = L/m$ | 决定梯度下降收敛速度 |

---

## 深度学习应用

### 应用一：损失函数的凸性分析

深度学习中常见损失函数的凸性分析对理解优化难度至关重要。

#### 对参数的凸性

**凸损失函数**（对模型参数而言）：

| 损失函数 | 表达式 | 凸性 | 说明 |
|----------|--------|------|------|
| MSE（线性模型）| $\|X\mathbf{w} - \mathbf{y}\|^2$ | 凸（$\nabla^2 = 2X^TX \succeq 0$）| 唯一全局最优 |
| 逻辑回归（交叉熵）| $-\sum[y_i \log \sigma(\mathbf{w}^T\mathbf{x}_i) + (1-y_i)\log(1-\sigma(\mathbf{w}^T\mathbf{x}_i))]$ | 凸 | 全局最优有保证 |
| SVM（Hinge 损失）| $\sum \max(0, 1 - y_i \mathbf{w}^T \mathbf{x}_i)$ | 凸（逐点最大值）| 凸但不可微 |

**非凸损失函数**（对模型参数而言）：

| 损失函数/模型 | 非凸原因 |
|--------------|---------|
| 神经网络的 MSE | 非线性激活函数引入非凸性 |
| 矩阵分解（$\|UV^T - M\|^2$）| $U, V$ 的积是双线性，非凸 |
| K-means 目标 | 整数约束，组合非凸 |

**关键洞察**：深度神经网络的损失曲面是**高度非凸**的，存在大量鞍点，但在足够宽的网络中，局部极小值的质量通常接近全局最优（这是近年深度学习理论的重要发现）。

#### PyTorch 代码：验证逻辑回归损失的凸性

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 示例 1：验证逻辑回归损失对参数的凸性（通过 Hessian）
# ============================================================

def logistic_loss(w, X, y):
    """二分类逻辑回归损失（交叉熵）"""
    logits = X @ w  # (n,)
    # 数值稳定的交叉熵
    loss = torch.log(1 + torch.exp(-y * logits)).mean()
    return loss

# 生成二分类数据
torch.manual_seed(42)
n, d = 100, 2
X = torch.randn(n, d)
y = (X[:, 0] + X[:, 1] > 0).float() * 2 - 1  # {-1, +1}

# 在某点计算 Hessian（验证正半定）
w0 = torch.zeros(d, requires_grad=False)
w0 = torch.tensor([0.5, -0.3], dtype=torch.float32, requires_grad=True)

# 用 PyTorch autograd 计算 Hessian
def compute_hessian(w, X, y):
    """通过二阶自动微分计算 Hessian"""
    w = w.detach().requires_grad_(True)
    loss = logistic_loss(w, X, y)
    grad = torch.autograd.grad(loss, w, create_graph=True)[0]
    H = torch.zeros(d, d)
    for i in range(d):
        H[i] = torch.autograd.grad(grad[i], w, retain_graph=True)[0]
    return H.detach()

H = compute_hessian(w0, X, y)
eigenvalues = torch.linalg.eigvalsh(H)

print("=== 逻辑回归损失的 Hessian 分析 ===")
print(f"Hessian 矩阵:\n{H.numpy()}")
print(f"特征值: {eigenvalues.numpy()}")
print(f"最小特征值: {eigenvalues.min().item():.6f} (>= 0 说明半正定，损失是凸函数)")
print(f"条件数 κ = L/m = {eigenvalues.max().item() / eigenvalues.min().item():.2f}")

# ============================================================
# 示例 2：可视化线性模型 vs 非线性模型的损失曲面凸性
# ============================================================

# 一维参数扫描，直观对比凸与非凸损失
w_range = torch.linspace(-3, 3, 200)

# (a) 线性逻辑回归（凸）
losses_convex = []
for w_val in w_range:
    w = torch.tensor([w_val, 0.0])
    losses_convex.append(logistic_loss(w, X, y).item())

# (b) 简单非线性网络（非凸，用 sin 模拟参数化非凸性）
def nonconvex_loss(w_val, X, y):
    """模拟非凸损失：含激活函数的简单网络"""
    # 两层网络：隐藏层大小=1，激活 tanh
    # w_val 是第一层权重，固定第二层 = 1
    hidden = torch.tanh(X[:, 0:1] * w_val)  # (n, 1)
    pred = hidden.squeeze()
    # 二分类交叉熵
    loss = torch.log(1 + torch.exp(-y * pred)).mean()
    return loss.item()

losses_nonconvex = [nonconvex_loss(w_val.item(), X, y) for w_val in w_range]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(w_range.numpy(), losses_convex, 'b-', linewidth=2)
axes[0].set_title('逻辑回归损失（凸函数）\n—— 无局部极值', fontsize=12)
axes[0].set_xlabel('参数 $w_1$')
axes[0].set_ylabel('损失')
axes[0].grid(True, alpha=0.3)

axes[1].plot(w_range.numpy(), losses_nonconvex, 'r-', linewidth=2)
axes[1].set_title('单隐层网络损失（非凸函数）\n—— 含局部极值', fontsize=12)
axes[1].set_xlabel('第一层权重 $w$')
axes[1].set_ylabel('损失')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convex_vs_nonconvex_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n损失曲面对比图已保存")

# ============================================================
# 示例 3：L2 正则化如何引入强凸性
# ============================================================

def regularized_loss(w, X, y, lam=0.1):
    """带 L2 正则化的逻辑回归：凸 + λ||w||² → 强凸"""
    return logistic_loss(w, X, y) + lam * (w ** 2).sum() / 2

print("\n=== L2 正则化引入强凸性 ===")
lam_values = [0.0, 0.01, 0.1, 1.0]
for lam in lam_values:
    w_test = torch.tensor([0.5, -0.3], requires_grad=True)
    # 重新定义带 lambda 的 loss
    def loss_fn(w):
        return logistic_loss(w, X, y) + lam * (w ** 2).sum() / 2
    H_reg = compute_hessian(w_test, X, y)
    # 手动加正则化贡献: H_total = H_loss + λ * I
    H_total = H_reg + lam * torch.eye(d)
    eigs = torch.linalg.eigvalsh(H_total)
    print(f"λ={lam:.2f}: 最小特征值(强凸参数 m) = {eigs.min().item():.4f}, "
          f"条件数 κ = {eigs.max().item()/eigs.min().item():.2f}")
```

### 应用二：ReLU 与凸性

ReLU 激活函数 $\text{ReLU}(x) = \max(0, x)$ 在深度学习中无处不在，其凸性具有重要的理论含义。

#### ReLU 的凸性分析

**命题**：ReLU 函数 $f(x) = \max(0, x)$ 是**凸函数**。

*验证（二阶条件）*：$f''(x) = 0$（$x \neq 0$），即 Hessian 为 0，满足 $\geq 0$。但 $f$ 不是严格凸的（存在线性段），也不是强凸的。

*验证（定义）*：$\max(0, \theta x + (1-\theta)y) \leq \theta\max(0,x) + (1-\theta)\max(0,y)$（max 函数的次加性）。

**Leaky ReLU、ELU 与凸性**：

| 激活函数 | 定义 | 凸性 |
|----------|------|------|
| ReLU | $\max(0, x)$ | 凸 |
| Leaky ReLU | $\max(\alpha x, x)$（$\alpha < 1$）| 凸 |
| ELU | $x$（$x>0$），$\alpha(e^x-1)$（$x \leq 0$）| 凸（$\alpha > 0$）|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | 非凸（S 形曲线）|
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | 非凸 |
| Softplus | $\log(1 + e^x)$ | 凸（光滑近似 ReLU）|

**ReLU 网络的非凸性来源**：虽然单个 ReLU 是凸的，但多层 ReLU 网络的**复合**是非凸的——凸函数的复合不一定是凸函数（除非满足单调非减条件）。

#### PyTorch 代码：ReLU 凸性可视化与分析

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 示例 1：验证激活函数的凸性（通过定义）
# ============================================================

def check_convexity(f, x_range, n_tests=1000, tol=1e-5):
    """
    数值验证函数凸性：
    随机采样 (x, y, θ)，检查 f(θx+(1-θ)y) <= θf(x)+(1-θ)f(y)
    """
    x = torch.FloatTensor(n_tests).uniform_(x_range[0], x_range[1])
    y = torch.FloatTensor(n_tests).uniform_(x_range[0], x_range[1])
    theta = torch.FloatTensor(n_tests).uniform_(0, 1)

    z = theta * x + (1 - theta) * y
    lhs = f(z)
    rhs = theta * f(x) + (1 - theta) * f(y)

    violations = (lhs > rhs + tol).sum().item()
    max_violation = (lhs - rhs).max().item()
    return violations, max_violation

# 定义激活函数
activations = {
    'ReLU':      lambda x: torch.relu(x),
    'Leaky ReLU': lambda x: torch.where(x > 0, x, 0.1 * x),
    'Softplus':  lambda x: torch.log1p(torch.exp(x)),
    'Sigmoid':   lambda x: torch.sigmoid(x),
    'Tanh':      lambda x: torch.tanh(x),
}

print("=== 激活函数凸性数值验证（x ∈ [-3, 3]）===")
print(f"{'函数':<15} {'违反次数':>10} {'最大违反量':>12} {'是否凸'}")
print("-" * 52)
for name, f in activations.items():
    violations, max_viol = check_convexity(f, (-3, 3))
    is_convex = "是" if violations == 0 else "否"
    print(f"{name:<15} {violations:>10} {max_viol:>12.6f} {is_convex:>6}")

# ============================================================
# 示例 2：深层 ReLU 网络的损失曲面（展示非凸性）
# ============================================================

class ShallowNet(torch.nn.Module):
    """两层 ReLU 网络"""
    def __init__(self, hidden=4):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, hidden)
        self.fc2 = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# 生成简单回归数据：y = x^2
torch.manual_seed(0)
x_data = torch.linspace(-2, 2, 50).unsqueeze(1)
y_data = x_data ** 2

model = ShallowNet(hidden=4)
criterion = torch.nn.MSELoss()

# 固定除第一个参数外的所有参数，扫描第一个参数
param0 = list(model.parameters())[0]
original_val = param0.data[0, 0].item()

param_range = np.linspace(-5, 5, 200)
losses = []

with torch.no_grad():
    for val in param_range:
        param0.data[0, 0] = val
        pred = model(x_data)
        loss = criterion(pred, y_data)
        losses.append(loss.item())

# 恢复原始值
param0.data[0, 0] = original_val

# ============================================================
# 示例 3：ReLU 的次梯度（不可微点的处理）
# ============================================================

print("\n=== ReLU 在 x=0 处的次梯度 ===")
print("ReLU'(x) = 1 (x > 0), 0 (x < 0), [0, 1] (x = 0)")
print("PyTorch 约定: ReLU'(0) = 0（取次梯度集合中的一个值）")

x_test = torch.tensor([0.0], requires_grad=True)
y_test = torch.relu(x_test)
y_test.backward()
print(f"torch.relu'(0) = {x_test.grad.item()} (PyTorch 实现选择)")

# 可视化：ReLU 及其导数
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

x_plot = torch.linspace(-3, 3, 300)

# (a) 常见激活函数
ax = axes[0]
for name, f in activations.items():
    ax.plot(x_plot.numpy(), f(x_plot).numpy(), label=name, linewidth=2)
ax.set_title('常见激活函数', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlabel('x')
ax.set_ylim(-2, 3)

# (b) 两层 ReLU 网络损失曲面（非凸）
ax = axes[1]
ax.plot(param_range, losses, 'r-', linewidth=2)
ax.set_title('两层 ReLU 网络损失曲面\n（非凸，存在多个局部极值）', fontsize=11)
ax.set_xlabel('第一层第一个权重')
ax.set_ylabel('MSE 损失')
ax.grid(True, alpha=0.3)

# (c) 凸函数的弦在曲线上方示意
ax = axes[2]
x_cvx = torch.linspace(-2, 2, 200)
y_cvx = torch.relu(x_cvx)
ax.plot(x_cvx.numpy(), y_cvx.numpy(), 'b-', linewidth=2.5, label='ReLU(x)')
# 画两点连线（弦）
x1, x2 = -1.5, 1.5
y1, y2 = max(0, x1), max(0, x2)
chord_x = np.linspace(x1, x2, 100)
chord_y = y1 + (y2 - y1) * (chord_x - x1) / (x2 - x1)
ax.plot(chord_x, chord_y, 'r--', linewidth=2, label='弦（在曲线上方）')
ax.scatter([x1, x2], [y1, y2], color='red', s=60, zorder=5)
ax.fill_between(chord_x, [max(0, xi) for xi in chord_x], chord_y,
                alpha=0.2, color='green', label='弦高于函数值')
ax.set_title('ReLU 是凸函数\n（弦在图像上方）', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('relu_convexity_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 示例 4：强凸性如何帮助优化——条件数对收敛的影响
# ============================================================

print("\n=== 条件数对梯度下降收敛速度的影响 ===")

def gradient_descent_quadratic(A, b, x0, lr, n_steps):
    """最小化 f(x) = 0.5 * x^T A x - b^T x"""
    x = x0.clone()
    losses = []
    x_opt = torch.linalg.solve(A, b)  # 解析解
    f_opt = (0.5 * x_opt @ A @ x_opt - b @ x_opt).item()
    for _ in range(n_steps):
        f = 0.5 * x @ A @ x - b @ x
        losses.append((f - f_opt).item())
        grad = A @ x - b
        x = x - lr * grad
    return losses

d = 2
b = torch.zeros(d)
x0 = torch.ones(d)
n_steps = 200

# 不同条件数的问题
scenarios = {
    'κ=1 (圆形等值线)':  (torch.diag(torch.tensor([1.0, 1.0])), 0.9),
    'κ=10':              (torch.diag(torch.tensor([1.0, 10.0])), 0.09),
    'κ=100 (扁椭圆)':   (torch.diag(torch.tensor([1.0, 100.0])), 0.009),
}

print(f"{'场景':<22} {'100步后误差':>12} {'理论收敛率':>14}")
print("-" * 52)
for name, (A, lr) in scenarios.items():
    eigs = torch.linalg.eigvalsh(A)
    m, L = eigs.min().item(), eigs.max().item()
    kappa = L / m
    losses = gradient_descent_quadratic(A, b, x0, lr, n_steps)
    final_err = losses[-1] if losses[-1] > 0 else 1e-15
    theory_rate = (1 - 1/kappa) ** 100
    print(f"{name:<22} {final_err:>12.2e} {theory_rate:>14.2e}")
```

---

## 练习题

### 基础题

**练习 3.1**（凸集判定）

判断以下集合是否为凸集，并给出简要理由：

(a) $S_1 = \{(x, y) \in \mathbb{R}^2 \mid x^2 + y^2 \leq 1\}$（单位圆盘）

(b) $S_2 = \{(x, y) \in \mathbb{R}^2 \mid x^2 + y^2 \geq 1\}$（圆盘外部）

(c) $S_3 = \{(x, y) \in \mathbb{R}^2 \mid xy \geq 1,\; x > 0,\; y > 0\}$（双曲线右上方）

(d) $S_4 = \{(x, y, z) \in \mathbb{R}^3 \mid x + 2y - z \leq 3\}$（半空间）

**练习 3.2**（凸函数的一阶条件）

设 $f(x) = e^x$。

(a) 用一阶条件验证 $f$ 是凸函数：即证明 $e^y \geq e^x + e^x(y - x)$ 对所有 $x, y \in \mathbb{R}$ 成立。

(b) 利用 Jensen 不等式证明：对非负实数 $a, b$ 有 $\frac{e^a + e^b}{2} \geq e^{\frac{a+b}{2}}$。

### 中级题

**练习 3.3**（保凸运算与损失函数）

设 $f_i: \mathbb{R}^n \to \mathbb{R}$（$i = 1, \ldots, m$）均为凸函数，$w_i > 0$，$\sum_i w_i = 1$。

(a) 证明 $f(\mathbf{x}) = \sum_{i=1}^m w_i f_i(\mathbf{x})$ 是凸函数。

(b) 在深度学习中，批训练损失 $\mathcal{L}(\mathbf{w}) = \frac{1}{N}\sum_{i=1}^N \ell(\mathbf{w}; \mathbf{x}_i, y_i)$，其中每个样本损失 $\ell(\mathbf{w}; \mathbf{x}_i, y_i)$ 对 $\mathbf{w}$ 是凸的。用 (a) 的结论说明 $\mathcal{L}(\mathbf{w})$ 的凸性。

(c) 若 $\ell$ 是平方损失 $\ell(\mathbf{w}; \mathbf{x}, y) = (\mathbf{w}^T\mathbf{x} - y)^2$，计算 $\nabla^2_\mathbf{w} \mathcal{L}(\mathbf{w})$ 并验证其半正定性。

**练习 3.4**（强凸性与条件数）

设 $f(\mathbf{x}) = \frac{1}{2}\|\mathbf{x}\|^2 + \lambda\|\mathbf{x}\|^2 = \frac{1+2\lambda}{2}\|\mathbf{x}\|^2$（纯 $L_2$ 正则化的简化模型）。

(a) 证明 $f$ 是 $(1+2\lambda)$-强凸且 $(1+2\lambda)$-光滑的，从而条件数 $\kappa = 1$。

(b) 现设目标函数为 $g(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T A \mathbf{x} + \lambda\|\mathbf{x}\|^2$，其中 $A$ 的特征值在 $[m_0, L_0]$ 内（$0 < m_0 \leq L_0$）。计算 $g$ 的强凸参数、光滑参数和条件数，说明 $\lambda$ 如何改善条件数。

### 提高题

**练习 3.5**（Jensen 不等式的应用：KL 散度的非负性）

设 $p, q$ 是 $\mathcal{X}$ 上的两个概率分布，KL 散度定义为

$$D_{\text{KL}}(p \| q) = \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)}$$

（约定 $0 \log 0 = 0$，$p(x) \log \frac{p(x)}{0} = +\infty$）

(a) 利用 Jensen 不等式证明 $D_{\text{KL}}(p \| q) \geq 0$，等号成立当且仅当 $p = q$。

*提示*：注意 $f(t) = -\log t$ 是凸函数，对 $X = q(x)/p(x)$ 在分布 $p$ 下应用 Jensen 不等式。

(b) 将 (a) 的结论推广到连续分布情形：

$$D_{\text{KL}}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx \geq 0$$

(c) 利用 KL 散度的非负性，证明**信息不等式**：对任意概率分布 $p$ 和均匀分布 $u$（$u(x) = 1/|\mathcal{X}|$），有

$$H(p) \leq \log|\mathcal{X}|$$

其中 $H(p) = -\sum_x p(x)\log p(x)$ 是熵。这说明均匀分布使熵最大。

---

## 练习答案

### 练习 3.1 解答

**(a) $S_1$（单位圆盘）：凸集。**

设 $\mathbf{x}, \mathbf{y} \in S_1$，即 $\|\mathbf{x}\| \leq 1$，$\|\mathbf{y}\| \leq 1$，$\theta \in [0,1]$。由三角不等式：

$$\|\theta\mathbf{x} + (1-\theta)\mathbf{y}\| \leq \theta\|\mathbf{x}\| + (1-\theta)\|\mathbf{y}\| \leq \theta \cdot 1 + (1-\theta) \cdot 1 = 1$$

故 $\theta\mathbf{x} + (1-\theta)\mathbf{y} \in S_1$。

**(b) $S_2$（圆盘外部）：非凸集。**

反例：取 $\mathbf{x} = (1, 0)$，$\mathbf{y} = (-1, 0)$，二者均在 $S_2$ 中。但中点 $\frac{\mathbf{x}+\mathbf{y}}{2} = (0, 0)$ 满足 $\|(0,0)\| = 0 < 1$，不属于 $S_2$。

**(c) $S_3$（双曲线右上方）：凸集。**

设 $(x_1, y_1), (x_2, y_2) \in S_3$，即 $x_i y_i \geq 1$，$x_i, y_i > 0$。对 $\theta \in [0,1]$，令 $x = \theta x_1 + (1-\theta)x_2 > 0$，$y = \theta y_1 + (1-\theta)y_2 > 0$。由 AM-GM 不等式：

$$xy = (\theta x_1 + (1-\theta)x_2)(\theta y_1 + (1-\theta)y_2) \geq (\sqrt{x_1 x_2})^{2\theta \cdot 0} \cdots$$

更简洁地，用 $\log$ 凹函数和 Jensen 不等式：$\log(xy) = \log x + \log y \geq \theta \log(x_1 y_1) + (1-\theta)\log(x_2 y_2) \geq 0$，故 $xy \geq 1$。（注意 $\log$ 是凹函数）

**(d) $S_4$（半空间）：凸集。**

已在命题 3.1 之前的例 3.1 中给出证明。

---

### 练习 3.2 解答

**(a) 用一阶条件验证 $e^x$ 的凸性：**

需证 $e^y \geq e^x + e^x(y-x)$，即 $e^{y-x} \geq 1 + (y-x)$。

令 $t = y - x$，只需证 $e^t \geq 1 + t$ 对所有 $t \in \mathbb{R}$ 成立。

定义 $h(t) = e^t - 1 - t$，则 $h'(t) = e^t - 1$，$h''(t) = e^t > 0$。

$h'(0) = 0$ 是全局最小值点，$h(0) = 0$，故 $h(t) \geq 0$ 对所有 $t$ 成立。$\square$

**(b) 用 Jensen 不等式：**

取 $f = e^x$（凸函数），对点 $\{a, b\}$，权重 $\theta_1 = \theta_2 = \frac{1}{2}$，Jensen 不等式给出：

$$e^{\frac{a+b}{2}} = f\!\left(\frac{a+b}{2}\right) = f\!\left(\frac{1}{2} \cdot a + \frac{1}{2} \cdot b\right) \leq \frac{1}{2}f(a) + \frac{1}{2}f(b) = \frac{e^a + e^b}{2}$$

$\square$

---

### 练习 3.3 解答

**(a)** 对任意 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$，$\theta \in [0,1]$：

$$f(\theta\mathbf{x} + (1-\theta)\mathbf{y}) = \sum_i w_i f_i(\theta\mathbf{x}+(1-\theta)\mathbf{y}) \leq \sum_i w_i [\theta f_i(\mathbf{x}) + (1-\theta)f_i(\mathbf{y})] = \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y})$$

**(b)** 批训练损失 $\mathcal{L}(\mathbf{w}) = \frac{1}{N}\sum_{i=1}^N \ell_i(\mathbf{w})$，其中 $w_i = 1/N > 0$，$\sum w_i = 1$，每个 $\ell_i$ 凸。由 (a) 知 $\mathcal{L}$ 是凸函数。这说明：**即使整体网络是非凸的，线性模型的批损失仍具有良好的凸性结构**。

**(c)** 平方损失 $\ell(\mathbf{w}; \mathbf{x}, y) = (\mathbf{w}^T\mathbf{x} - y)^2$，$\nabla_\mathbf{w} \ell = 2(\mathbf{w}^T\mathbf{x} - y)\mathbf{x}$，$\nabla^2_\mathbf{w} \ell = 2\mathbf{x}\mathbf{x}^T$（秩 1 半正定矩阵）。

$$\nabla^2_\mathbf{w} \mathcal{L} = \frac{1}{N}\sum_{i=1}^N 2\mathbf{x}_i\mathbf{x}_i^T = \frac{2}{N} X^T X \succeq 0$$

其中 $X^TX$ 是 Gram 矩阵，总是半正定的（因为 $\mathbf{v}^T X^T X \mathbf{v} = \|X\mathbf{v}\|^2 \geq 0$）。

---

### 练习 3.4 解答

**(a)** $f(\mathbf{x}) = \frac{1+2\lambda}{2}\|\mathbf{x}\|^2$，$\nabla^2 f = (1+2\lambda)I$。

特征值恒为 $1+2\lambda > 0$，故 $m = L = 1+2\lambda$，$\kappa = L/m = 1$。

条件数为 1 意味着梯度下降一步即收敛（等值线是正圆）。

**(b)** $g(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T(A + 2\lambda I)\mathbf{x}$，$\nabla^2 g = A + 2\lambda I$。

特征值范围：$[m_0 + 2\lambda, L_0 + 2\lambda]$，因此：
- 强凸参数：$m = m_0 + 2\lambda$（$\lambda$ 越大，强凸参数越大）
- 光滑参数：$L = L_0 + 2\lambda$
- 条件数：$\kappa(\lambda) = \frac{L_0 + 2\lambda}{m_0 + 2\lambda}$

当 $\lambda \to \infty$ 时，$\kappa(\lambda) \to 1$，即正则化越强，条件数越接近 1，收敛越快。

这正是 L2 正则化（权重衰减）不仅防止过拟合，还能**改善优化条件数**的数学依据。

---

### 练习 3.5 解答

**(a)** 由于 $p$ 是概率分布（$\sum_x p(x) = 1$，$p(x) \geq 0$），我们对分布 $p$ 下的随机变量 $X = q(x)/p(x)$ 应用 Jensen 不等式。

$f(t) = -\log t$ 是凸函数（$f''(t) = 1/t^2 > 0$），由 Jensen 不等式：

$$\mathbb{E}_p[f(X)] \geq f(\mathbb{E}_p[X])$$

即

$$\sum_{x:p(x)>0} p(x) \left(-\log\frac{q(x)}{p(x)}\right) \geq -\log\!\left(\sum_{x:p(x)>0} p(x) \cdot \frac{q(x)}{p(x)}\right) = -\log\!\left(\sum_x q(x)\right) = -\log 1 = 0$$

故

$$D_{\text{KL}}(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = \sum_{x:p(x)>0} p(x)\left(-\log\frac{q(x)}{p(x)}\right) \geq 0$$

**等号条件**：$-\log t$ 严格凸，Jensen 等号成立 $\iff$ $X = q(x)/p(x)$ 在分布 $p$ 下是常数 $\iff$ $q(x)/p(x) = c$ 对所有 $p(x) > 0$。由 $\sum_x q(x) = 1 = \sum_x p(x)$ 得 $c = 1$，即 $p = q$。$\square$

**(b)** 连续情形将求和换为积分，Jensen 不等式在连续情形下仍成立：

$$D_{\text{KL}}(p\|q) = \int p(x)\log\frac{p(x)}{q(x)}dx = \int p(x)\left(-\log\frac{q(x)}{p(x)}\right)dx \geq -\log\int p(x)\cdot\frac{q(x)}{p(x)}dx = -\log 1 = 0$$

**(c)** 利用 KL 散度非负性：

$$0 \leq D_{\text{KL}}(p\|u) = \sum_x p(x)\log\frac{p(x)}{1/|\mathcal{X}|} = \sum_x p(x)\log p(x) + \sum_x p(x)\log|\mathcal{X}|$$

$$= -H(p) + \log|\mathcal{X}|$$

故 $H(p) \leq \log|\mathcal{X}|$，等号当且仅当 $p = u$（均匀分布）时成立。$\square$

这一结论在信息论中极为重要：均匀分布使离散熵最大，是"最大不确定性"的数学表达。

---

*下一章：第4章 最优性条件——我们将在凸集与凸函数的基础上，建立优化问题最优解的充分必要条件。*
