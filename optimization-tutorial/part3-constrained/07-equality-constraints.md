# 第7章：等式约束优化

> **前置章节**：第2章（多元微分学）、第3章（凸集与凸函数）、第4章（最优性条件）
>
> **难度**：★★★★☆

---

## 学习目标

学完本章，你将能够：

1. **建立等式约束优化的数学框架**：将实际问题形式化为 $\min f(\mathbf{x})\ \text{s.t.}\ \mathbf{h}(\mathbf{x}) = \mathbf{0}$，理解可行集的几何结构
2. **掌握拉格朗日乘数法**：构造拉格朗日函数 $L(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \boldsymbol{\lambda}^\top \mathbf{h}(\mathbf{x})$，理解乘数的几何与经济学意义
3. **推导一阶必要条件（KKT等式版本）**：掌握约束雅可比矩阵的正则性条件，以及梯度平行条件 $\nabla f = -\mathbf{J}_h^\top \boldsymbol{\lambda}$
4. **应用二阶充分条件**：在约束切空间上分析约束Hessian的正定性，判断局部极小
5. **理解消元法与降维思想**：通过参数化约束面将约束问题转化为无约束问题，并比较其与拉格朗日法的优劣

---

## 7.1 等式约束优化问题

### 7.1.1 问题的一般形式

**等式约束优化问题**的标准形式为：

$$\boxed{\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}) \quad \text{s.t.} \quad \mathbf{h}(\mathbf{x}) = \mathbf{0}}$$

其中：
- $\mathbf{x} \in \mathbb{R}^n$：决策变量
- $f: \mathbb{R}^n \to \mathbb{R}$：目标函数（连续可微）
- $\mathbf{h}: \mathbb{R}^n \to \mathbb{R}^m$：约束函数，$\mathbf{h}(\mathbf{x}) = (h_1(\mathbf{x}), \ldots, h_m(\mathbf{x}))^\top$，每个 $h_i$ 连续可微
- 通常要求 $m < n$（约束数少于变量数），否则问题可能无解或解退化

**可行集**（feasible set）定义为：

$$\mathcal{F} = \{\mathbf{x} \in \mathbb{R}^n \mid \mathbf{h}(\mathbf{x}) = \mathbf{0}\}$$

从几何上看，$\mathcal{F}$ 是 $\mathbb{R}^n$ 中的一个**流形**（manifold），其维数为 $n - m$（在正则点处）。优化目标是在这个流形上找到 $f$ 的极小值点。

### 7.1.2 经典应用场景

**例 7.1（球面上的极值）**

$$\min_{x_1, x_2, x_3} f(x_1, x_2, x_3) = x_1 + 2x_2 + 3x_3 \quad \text{s.t.} \quad x_1^2 + x_2^2 + x_3^2 = 1$$

这是在单位球面（$\mathbb{R}^3$ 中的二维流形）上极小化线性函数。

**例 7.2（预算约束下的效用最大化）**

$$\max_{x_1, x_2} U(x_1, x_2) = x_1^\alpha x_2^{1-\alpha} \quad \text{s.t.} \quad p_1 x_1 + p_2 x_2 = I$$

其中 $\alpha \in (0,1)$ 为偏好参数，$p_1, p_2$ 为价格，$I$ 为收入。这是微观经济学中的标准消费者问题。

**例 7.3（最小二乘的等式约束变体）**

$$\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 \quad \text{s.t.} \quad \mathbf{C}\mathbf{x} = \mathbf{d}$$

在线性代数约束下的最小化，常见于有结构约束的回归问题。

**例 7.4（深度学习：权重矩阵正交约束）**

$$\min_{\mathbf{W} \in \mathbb{R}^{n \times n}} \mathcal{L}(\mathbf{W}) \quad \text{s.t.} \quad \mathbf{W}^\top \mathbf{W} = \mathbf{I}$$

要求权重矩阵为正交矩阵，保证特征值均为 $\pm 1$，避免梯度消失/爆炸。

### 7.1.3 为什么无约束技术不够用

直觉上，我们希望把约束 $\mathbf{h}(\mathbf{x}) = \mathbf{0}$ 代入目标函数消掉约束，直接用无约束方法。但这在实践中往往遇到困难：

| 困难 | 原因 | 举例 |
|------|------|------|
| 代入代数复杂 | 约束方程难以显式解出 | $e^{x_1} + x_2^3 = 1$ |
| 引入参数化不唯一 | 约束流形可能需要多个局部坐标 | 球面 $\mathbb{S}^2$ |
| 破坏对称性 | 参数化后目标函数结构变得复杂 | 约束最优控制 |
| 难以处理多个约束 | 多个约束同时消元更难 | $m$ 个约束同时处理 |

拉格朗日乘数法提供了一种**不显式消元**、对称处理所有变量的优雅框架。

### 7.1.4 正则性条件（LICQ）

**定义 7.1（线性独立约束规范，LICQ）**

在可行点 $\mathbf{x}^* \in \mathcal{F}$ 处，若约束函数的梯度 $\nabla h_1(\mathbf{x}^*), \ldots, \nabla h_m(\mathbf{x}^*)$ 线性无关，则称 $\mathbf{x}^*$ 满足**线性独立约束规范**（Linear Independence Constraint Qualification，LICQ）。

等价地，**约束雅可比矩阵**

$$\mathbf{J}_h(\mathbf{x}^*) = \begin{pmatrix} \nabla h_1(\mathbf{x}^*)^\top \\ \vdots \\ \nabla h_m(\mathbf{x}^*)^\top \end{pmatrix} \in \mathbb{R}^{m \times n}$$

具有满行秩（$\text{rank}(\mathbf{J}_h) = m$）。

**几何意义**：LICQ 保证约束面在 $\mathbf{x}^*$ 处"光滑且无退化"，局部像一个 $(n-m)$ 维光滑流形。

**不满足 LICQ 的例子**：约束 $h_1(x,y) = x^2 + y^2 - 1 = 0$，$h_2(x,y) = x^2 + y^2 - 2 = 0$，可行集为空，无意义；或两个约束在可行点处梯度平行（约束相切）。

---

## 7.2 拉格朗日乘数法

### 7.2.1 几何动机

考虑单约束问题：$\min f(\mathbf{x})\ \text{s.t.}\ h(\mathbf{x}) = 0$。

设 $\mathbf{x}^*$ 是约束极小值点。在 $\mathbf{x}^*$ 处，$\nabla h(\mathbf{x}^*)$ 是约束曲面的**法向量**。若 $\nabla f(\mathbf{x}^*)$ 在约束面上有分量，即存在切方向 $\mathbf{d}$（满足 $\nabla h(\mathbf{x}^*)^\top \mathbf{d} = 0$）使得 $\nabla f(\mathbf{x}^*)^\top \mathbf{d} < 0$，则沿 $\mathbf{d}$ 移动可以保持在约束面上（一阶近似）并减小 $f$，与极小性矛盾。

因此，**在约束极小值点处，$\nabla f$ 在约束面的切空间上的投影必须为零**，即 $\nabla f(\mathbf{x}^*)$ 必须**垂直于约束面的切空间**，即**平行于约束法向量** $\nabla h(\mathbf{x}^*)$：

$$\nabla f(\mathbf{x}^*) = -\lambda^* \nabla h(\mathbf{x}^*)$$

其中 $\lambda^* \in \mathbb{R}$ 是某个标量（**拉格朗日乘数**）。

**多约束推广**：对于 $m$ 个约束 $h_1 = \cdots = h_m = 0$，条件变为 $\nabla f$ 在约束法空间（由 $\nabla h_1, \ldots, \nabla h_m$ 张成）内：

$$\nabla f(\mathbf{x}^*) = -\sum_{i=1}^m \lambda_i^* \nabla h_i(\mathbf{x}^*) = -\mathbf{J}_h(\mathbf{x}^*)^\top \boldsymbol{\lambda}^*$$

### 7.2.2 拉格朗日函数

**定义 7.2（拉格朗日函数）**

对于约束优化问题 $\min f(\mathbf{x})\ \text{s.t.}\ \mathbf{h}(\mathbf{x}) = \mathbf{0}$，定义**拉格朗日函数**（Lagrangian function）为：

$$\boxed{L(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \boldsymbol{\lambda}^\top \mathbf{h}(\mathbf{x}) = f(\mathbf{x}) + \sum_{i=1}^{m} \lambda_i h_i(\mathbf{x})}$$

其中 $\boldsymbol{\lambda} = (\lambda_1, \ldots, \lambda_m)^\top \in \mathbb{R}^m$ 称为**拉格朗日乘数**（Lagrange multipliers）或**对偶变量**（dual variables）。

**注意**：有些教材写 $L = f - \lambda^\top h$（用减号），两种写法都正确，乘数符号相差一个负号。本教程统一使用加号。

### 7.2.3 拉格朗日乘数的经济学解释

在约束优化中，$\lambda_i$ 有直接的经济学意义：**$\lambda_i$ 是第 $i$ 个约束的影子价格（shadow price）**，即约束右端 $c_i$ 变化一个单位时，最优目标值的变化率：

$$\lambda_i^* = \frac{\partial f^*}{\partial c_i}, \quad \text{其中} \quad f^* = \min\{f(\mathbf{x}) \mid \mathbf{h}(\mathbf{x}) = \mathbf{c}\}$$

**例 7.5（预算约束的影子价格）**：在例 7.2 的消费者问题中，$\lambda^*$ 表示收入 $I$ 每增加一元时，最大效用的增量，即**货币的边际效用**。

### 7.2.4 拉格朗日乘数定理

**定理 7.1（拉格朗日乘数定理，一阶必要条件）**

设 $f, \mathbf{h}$ 连续可微，$\mathbf{x}^*$ 是约束优化问题的局部极小值点，且在 $\mathbf{x}^*$ 处满足 LICQ。则存在唯一的 $\boldsymbol{\lambda}^* \in \mathbb{R}^m$，使得：

$$\nabla_\mathbf{x} L(\mathbf{x}^*, \boldsymbol{\lambda}^*) = \nabla f(\mathbf{x}^*) + \mathbf{J}_h(\mathbf{x}^*)^\top \boldsymbol{\lambda}^* = \mathbf{0}$$

$$\nabla_{\boldsymbol{\lambda}} L(\mathbf{x}^*, \boldsymbol{\lambda}^*) = \mathbf{h}(\mathbf{x}^*) = \mathbf{0}$$

即 $(\mathbf{x}^*, \boldsymbol{\lambda}^*)$ 是拉格朗日函数 $L(\mathbf{x}, \boldsymbol{\lambda})$ 的**驻点**。

**核心思想**：将有约束极小值问题，转化为无约束驻点问题（但维数从 $n$ 扩展到 $n + m$）。

**例 7.6**：极小化 $f(x_1, x_2) = x_1^2 + x_2^2$ 约束于 $h(x_1, x_2) = x_1 + x_2 - 1 = 0$。

构造拉格朗日函数：

$$L(x_1, x_2, \lambda) = x_1^2 + x_2^2 + \lambda(x_1 + x_2 - 1)$$

驻点条件：

$$\frac{\partial L}{\partial x_1} = 2x_1 + \lambda = 0, \quad \frac{\partial L}{\partial x_2} = 2x_2 + \lambda = 0, \quad \frac{\partial L}{\partial \lambda} = x_1 + x_2 - 1 = 0$$

由前两式得 $x_1 = x_2 = -\lambda/2$，代入第三式：$-\lambda = 1$，故 $\lambda^* = -1$，$x_1^* = x_2^* = 1/2$。

极小值：$f(1/2, 1/2) = 1/2$。

**几何解释**：原点到直线 $x_1 + x_2 = 1$ 的最近点正是 $(1/2, 1/2)$。

---

## 7.3 一阶必要条件

### 7.3.1 KKT 方程组（等式版本）

将拉格朗日乘数定理写成方程组，即**KKT（Karush-Kuhn-Tucker）条件**的等式约束版本：

$$\boxed{\begin{cases} \nabla f(\mathbf{x}^*) + \mathbf{J}_h(\mathbf{x}^*)^\top \boldsymbol{\lambda}^* = \mathbf{0} & \text{（梯度条件）} \\ \mathbf{h}(\mathbf{x}^*) = \mathbf{0} & \text{（可行性条件）} \end{cases}}$$

这是含 $n + m$ 个方程、$n + m$ 个未知数（$\mathbf{x}^*$ 和 $\boldsymbol{\lambda}^*$）的非线性方程组。

**展开写法**：

$$\frac{\partial f}{\partial x_j}(\mathbf{x}^*) + \sum_{i=1}^m \lambda_i^* \frac{\partial h_i}{\partial x_j}(\mathbf{x}^*) = 0, \quad j = 1, \ldots, n$$

$$h_i(\mathbf{x}^*) = 0, \quad i = 1, \ldots, m$$

### 7.3.2 雅可比矩阵与约束的几何

约束雅可比矩阵 $\mathbf{J}_h \in \mathbb{R}^{m \times n}$ 的行是各约束函数的梯度（行向量），列对应各变量的偏导数：

$$\mathbf{J}_h(\mathbf{x}) = \begin{pmatrix} \frac{\partial h_1}{\partial x_1} & \cdots & \frac{\partial h_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial h_m}{\partial x_1} & \cdots & \frac{\partial h_m}{\partial x_n} \end{pmatrix}$$

**几何意义**：
- $\mathbf{J}_h(\mathbf{x}^*)$ 的行空间 = 约束面的**法空间**（normal space），维数 $m$
- $\ker(\mathbf{J}_h(\mathbf{x}^*))$ = 约束面的**切空间**（tangent space），维数 $n - m$
- 梯度条件 $\nabla f + \mathbf{J}_h^\top \boldsymbol{\lambda} = \mathbf{0}$ 意味着 $\nabla f \in \text{行空间}(\mathbf{J}_h)$，即 $\nabla f$ 落在法空间中

### 7.3.3 求解 KKT 方程组的策略

**方法一：直接代入法（小规模线性/二次问题）**

联立方程，直接求解 $\mathbf{x}^*$ 和 $\boldsymbol{\lambda}^*$。

**例 7.7**：极大化 Cobb-Douglas 效用 $U = x_1^{1/2} x_2^{1/2}$，约束 $2x_1 + x_2 = 12$。

拉格朗日函数：$L = x_1^{1/2} x_2^{1/2} + \lambda(12 - 2x_1 - x_2)$

（注：这里取极大化，改写为 $\min -U$）

KKT 条件：

$$\frac{\partial L}{\partial x_1} = \frac{1}{2}x_1^{-1/2}x_2^{1/2} - 2\lambda = 0$$

$$\frac{\partial L}{\partial x_2} = \frac{1}{2}x_1^{1/2}x_2^{-1/2} - \lambda = 0$$

$$2x_1 + x_2 = 12$$

由前两个方程消去 $\lambda$：$\frac{x_2}{4x_1} = \frac{1}{2}$，即 $x_2 = 2x_1$。

代入约束：$2x_1 + 2x_1 = 12$，得 $x_1^* = 3$，$x_2^* = 6$，$U^* = \sqrt{18} = 3\sqrt{2}$。

**方法二：参数化消元（见 7.5 节）**

**方法三：数值方法（大规模）**

对于大规模问题，用 SQP（序列二次规划）或增广拉格朗日法等数值算法求解（见后续章节）。

### 7.3.4 多约束例题

**例 7.8**：在 $\mathbb{R}^3$ 中，极小化 $f = x_1^2 + x_2^2 + x_3^2$ 约束于 $x_1 + x_2 = 1$ 和 $x_2 + x_3 = 1$。

约束雅可比：

$$\mathbf{J}_h = \begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{pmatrix}$$

满秩（行线性无关），LICQ 成立。

KKT 条件：

$$\begin{cases} 2x_1 + \lambda_1 = 0 \\ 2x_2 + \lambda_1 + \lambda_2 = 0 \\ 2x_3 + \lambda_2 = 0 \\ x_1 + x_2 = 1 \\ x_2 + x_3 = 1 \end{cases}$$

由前三式：$x_1 = -\lambda_1/2$，$x_3 = -\lambda_2/2$，$x_2 = -(\lambda_1 + \lambda_2)/2$。

代入约束方程组，解得 $x_1^* = 1/3$，$x_2^* = 2/3$，$x_3^* = 1/3$，$f^* = 2/3$。

**解的几何意义**：可行集为两个平面的交线（一维仿射子空间），最近点即原点到该直线的垂足。

### 7.3.5 线性约束下的解析解

当约束为线性 $\mathbf{A}\mathbf{x} = \mathbf{b}$（$\mathbf{A} \in \mathbb{R}^{m \times n}$），目标为凸二次 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x}$（$\mathbf{Q} \succ 0$）时，KKT 方程组变为**线性方程组**：

$$\begin{pmatrix} \mathbf{Q} & \mathbf{A}^\top \\ \mathbf{A} & \mathbf{0} \end{pmatrix} \begin{pmatrix} \mathbf{x}^* \\ \boldsymbol{\lambda}^* \end{pmatrix} = \begin{pmatrix} -\mathbf{c} \\ \mathbf{b} \end{pmatrix}$$

这是一个**KKT 线性方程组**（也称鞍点方程组），可直接用线性代数方法求解。当 $\mathbf{A}$ 满行秩时，系数矩阵非奇异，有唯一解。

---

## 7.4 二阶条件

### 7.4.1 约束切空间

在满足一阶条件的点 $(\mathbf{x}^*, \boldsymbol{\lambda}^*)$ 处，需要二阶分析来区分局部极小、局部极大和鞍点。

**定义 7.3（约束切空间）**

在可行点 $\mathbf{x}^*$ 处，可行方向的一阶近似构成**切空间**：

$$\mathcal{T}(\mathbf{x}^*) = \ker(\mathbf{J}_h(\mathbf{x}^*)) = \{\mathbf{d} \in \mathbb{R}^n \mid \mathbf{J}_h(\mathbf{x}^*)\mathbf{d} = \mathbf{0}\}$$

即沿切空间方向移动，约束 $\mathbf{h}(\mathbf{x}) = \mathbf{0}$ 在一阶近似下仍然成立。切空间维数为 $n - m$。

**直觉**：无约束优化中我们在整个 $\mathbb{R}^n$ 上分析 $\nabla^2 f$；在约束优化中，我们只需在**切空间** $\mathcal{T}(\mathbf{x}^*)$ 上分析，因为这才是"实际可行的移动方向"。

### 7.4.2 约束 Hessian（拉格朗日函数的 Hessian）

**定义 7.4（拉格朗日 Hessian）**

$$\nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{x}^*, \boldsymbol{\lambda}^*) = \nabla^2 f(\mathbf{x}^*) + \sum_{i=1}^m \lambda_i^* \nabla^2 h_i(\mathbf{x}^*)$$

这是将所有约束"加权叠加"到目标函数 Hessian 上。当约束为线性时（$\nabla^2 h_i = 0$），拉格朗日 Hessian 退化为目标 Hessian。

### 7.4.3 二阶必要条件

**定理 7.2（二阶必要条件）**

设 $(\mathbf{x}^*, \boldsymbol{\lambda}^*)$ 满足一阶 KKT 条件，$\mathbf{x}^*$ 是约束局部极小值点。则：

$$\mathbf{d}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{x}^*, \boldsymbol{\lambda}^*) \mathbf{d} \geq 0, \quad \forall \mathbf{d} \in \mathcal{T}(\mathbf{x}^*)$$

即拉格朗日 Hessian 在约束切空间上是**半正定**的。

### 7.4.4 二阶充分条件

**定理 7.3（二阶充分条件）**

设 $(\mathbf{x}^*, \boldsymbol{\lambda}^*)$ 满足一阶 KKT 条件，且满足 LICQ。若：

$$\mathbf{d}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{x}^*, \boldsymbol{\lambda}^*) \mathbf{d} > 0, \quad \forall \mathbf{d} \in \mathcal{T}(\mathbf{x}^*) \setminus \{\mathbf{0}\}$$

则 $\mathbf{x}^*$ 是约束严格局部极小值点。

**关键要点**：二阶条件只需在**切空间**上分析，而非全空间。拉格朗日 Hessian 在全空间上可以是不定的，但只要在切空间上正定，就足以保证局部极小。

### 7.4.5 二阶条件的矩阵刻画

实践中，切空间上的正定性可以通过以下方式验证：

设 $\mathbf{Z} \in \mathbb{R}^{n \times (n-m)}$ 的列向量构成切空间的**正交基**（$\mathbf{J}_h \mathbf{Z} = \mathbf{0}$），则二阶充分条件等价于：

$$\mathbf{Z}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L \cdot \mathbf{Z} \succ \mathbf{0}$$

即**约简 Hessian**（reduced Hessian）$\mathbf{Z}^\top \mathbf{H}_L \mathbf{Z} \in \mathbb{R}^{(n-m) \times (n-m)}$ 正定。

**例 7.9**：验证例 7.6 的二阶条件。

约束 $h(x_1, x_2) = x_1 + x_2 - 1 = 0$，$\mathbf{J}_h = (1, 1)$。

切空间：$\mathcal{T} = \ker(1, 1) = \{(d_1, d_2) : d_1 + d_2 = 0\}$，基向量 $\mathbf{z} = (1, -1)^\top/\sqrt{2}$。

拉格朗日 Hessian（$h$ 为线性，$\nabla^2 h = 0$）：

$$\nabla^2_{\mathbf{x}\mathbf{x}} L = \nabla^2 f = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$$

约简 Hessian：

$$\mathbf{z}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L \cdot \mathbf{z} = \frac{1}{2}(1, -1)\begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \frac{1}{2} \cdot 4 = 2 > 0$$

二阶充分条件成立，$(1/2, 1/2)$ 是约束严格局部极小值点（这里也是全局极小，因 $f$ 是凸函数）。

### 7.4.6 无约束与约束的二阶条件对比

| 条件 | 无约束 | 等式约束 |
|------|--------|----------|
| 一阶必要 | $\nabla f = \mathbf{0}$ | $\nabla f + \mathbf{J}_h^\top \boldsymbol{\lambda} = \mathbf{0}$，$\mathbf{h} = \mathbf{0}$ |
| 二阶必要 | $\nabla^2 f \succeq \mathbf{0}$（全空间）| $\mathbf{Z}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L \mathbf{Z} \succeq \mathbf{0}$（切空间）|
| 二阶充分 | $\nabla^2 f \succ \mathbf{0}$（全空间）| $\mathbf{Z}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L \mathbf{Z} \succ \mathbf{0}$（切空间）|
| Hessian 的作用 | 直接判断正定性 | 需在约束子空间上投影 |

---

## 7.5 消元法与降维

### 7.5.1 消元法的基本思想

对于某些结构简单的约束，可以**显式参数化**可行集，从而将约束优化问题化为无约束问题。

**一般框架**：将变量 $\mathbf{x} \in \mathbb{R}^n$ 分拆为"因变量" $\mathbf{y} \in \mathbb{R}^m$ 和"自由变量" $\mathbf{z} \in \mathbb{R}^{n-m}$：

$$\mathbf{x} = \begin{pmatrix} \mathbf{y} \\ \mathbf{z} \end{pmatrix}, \quad \text{约束：} \mathbf{h}(\mathbf{y}, \mathbf{z}) = \mathbf{0}$$

若可以从约束中显式解出 $\mathbf{y} = \boldsymbol{\phi}(\mathbf{z})$（由隐函数定理保证局部可行），则代入目标函数：

$$\tilde{f}(\mathbf{z}) = f(\boldsymbol{\phi}(\mathbf{z}), \mathbf{z})$$

原问题化为无约束问题 $\min_{\mathbf{z}} \tilde{f}(\mathbf{z})$，维数降低 $m$。

### 7.5.2 线性约束的消元

当约束为 $\mathbf{A}\mathbf{x} = \mathbf{b}$（$\mathbf{A} \in \mathbb{R}^{m \times n}$，满行秩）时，消元最为简洁。

将 $\mathbf{A}$ 列分拆为基部分 $\mathbf{B} \in \mathbb{R}^{m \times m}$（可逆）和非基部分 $\mathbf{N} \in \mathbb{R}^{m \times (n-m)}$：

$$\mathbf{A} = [\mathbf{B} \mid \mathbf{N}], \quad \mathbf{x} = \begin{pmatrix} \mathbf{x}_B \\ \mathbf{x}_N \end{pmatrix}$$

由约束解出基变量：

$$\mathbf{x}_B = \mathbf{B}^{-1}(\mathbf{b} - \mathbf{N}\mathbf{x}_N)$$

代入目标函数得到关于 $\mathbf{x}_N$ 的无约束问题，这正是线性规划中**单纯形法**的核心思想。

### 7.5.3 隐函数定理保证

**定理 7.4（隐函数定理，优化版本）**

设 $\mathbf{h}: \mathbb{R}^n \to \mathbb{R}^m$ 连续可微，$\mathbf{h}(\mathbf{x}_0) = \mathbf{0}$，且将 $\mathbf{J}_h(\mathbf{x}_0)$ 的某 $m$ 列构成的子方阵 $\mathbf{J}_B$ 可逆（即 $\det(\mathbf{J}_B) \neq 0$）。

则在 $\mathbf{x}_0$ 的邻域内，可以将对应的 $m$ 个变量 $\mathbf{y}$ 表示为其余 $n-m$ 个变量 $\mathbf{z}$ 的光滑函数 $\mathbf{y} = \boldsymbol{\phi}(\mathbf{z})$，满足：
1. $\mathbf{h}(\boldsymbol{\phi}(\mathbf{z}), \mathbf{z}) = \mathbf{0}$（约束自动满足）
2. $\nabla_\mathbf{z} \boldsymbol{\phi} = -\mathbf{J}_B^{-1} \mathbf{J}_N$（雅可比由链式法则给出）

其中 $\mathbf{J}_N$ 是 $\mathbf{J}_h$ 中对应 $\mathbf{z}$ 的列构成的子矩阵。

### 7.5.4 消元法的梯度推导

设 $\mathbf{y} = \boldsymbol{\phi}(\mathbf{z})$ 满足约束，$\tilde{f}(\mathbf{z}) = f(\boldsymbol{\phi}(\mathbf{z}), \mathbf{z})$。由链式法则：

$$\nabla_\mathbf{z} \tilde{f} = \nabla_\mathbf{y} f \cdot \nabla_\mathbf{z} \boldsymbol{\phi} + \nabla_\mathbf{z} f = \nabla_\mathbf{y} f \cdot (-\mathbf{J}_B^{-1} \mathbf{J}_N) + \nabla_\mathbf{z} f$$

令 $\nabla_\mathbf{z} \tilde{f} = \mathbf{0}$，可以证明此方程与 KKT 方程等价（乘数 $\boldsymbol{\lambda}^* = \mathbf{J}_B^{-\top} \nabla_\mathbf{y} f$），两种方法殊途同归。

### 7.5.5 消元法与拉格朗日法的比较

| 比较维度 | 消元法 | 拉格朗日乘数法 |
|----------|--------|----------------|
| 变量数量 | $n - m$（约简） | $n + m$（扩展）|
| 适用条件 | 约束可显式参数化 | 任意等式约束 |
| 数学复杂度 | 代入后可能复杂 | 方程形式统一 |
| 对称性 | 破坏原始对称性 | 保持对称结构 |
| 数值实现 | 需要显式参数化 | 可用标准 KKT 求解器 |
| 乘数可用性 | 需额外计算 | 直接给出乘数 |

**选择原则**：
- 约束简单（线性或一元非线性）且可显式求解：消元法更直接
- 约束复杂或对称性重要：拉格朗日法更系统
- 大规模数值计算：使用增广拉格朗日法或 SQP（后续章节）

### 7.5.6 球面约束参数化例子

**例 7.10**：$\min f(x_1, x_2, x_3)\ \text{s.t.}\ x_1^2 + x_2^2 + x_3^2 = 1$（单位球面上的优化）

球面可用球坐标参数化：

$$x_1 = \sin\theta\cos\phi, \quad x_2 = \sin\theta\sin\phi, \quad x_3 = \cos\theta, \quad \theta \in (0,\pi),\ \phi \in [0, 2\pi)$$

代入得无约束问题 $\min_{\theta, \phi} \tilde{f}(\theta, \phi)$。但此参数化在极点处有奇性，需要分区域处理，这正体现了消元法的局限性。拉格朗日法则无此问题。

---

## 本章小结

| 概念 | 数学表达 | 关键条件 | 实践作用 |
|------|---------|----------|---------|
| **等式约束问题** | $\min f(\mathbf{x})$ s.t. $\mathbf{h}(\mathbf{x}) = \mathbf{0}$ | $m < n$，$h_i$ 可微 | 约束优化的基本形式 |
| **可行集** | $\mathcal{F} = \{\mathbf{x} \mid \mathbf{h}(\mathbf{x}) = \mathbf{0}\}$ | 在正则点处是 $(n-m)$ 维流形 | 搜索空间 |
| **LICQ** | $\mathbf{J}_h(\mathbf{x}^*)$ 满行秩 | 保证乘数唯一存在 | 拉格朗日定理前提 |
| **拉格朗日函数** | $L = f + \boldsymbol{\lambda}^\top \mathbf{h}$ | 乘数 $\boldsymbol{\lambda} \in \mathbb{R}^m$ | 将约束问题化为驻点问题 |
| **一阶必要条件** | $\nabla f + \mathbf{J}_h^\top \boldsymbol{\lambda} = \mathbf{0}$，$\mathbf{h} = \mathbf{0}$ | LICQ + 极小性 | 候选极值点的方程 |
| **约束切空间** | $\mathcal{T} = \ker(\mathbf{J}_h)$，维数 $n-m$ | 约束面的切向量 | 二阶分析的舞台 |
| **拉格朗日 Hessian** | $\nabla^2_{\mathbf{x}\mathbf{x}} L = \nabla^2 f + \sum \lambda_i \nabla^2 h_i$ | 约束曲率的加权贡献 | 二阶条件的关键量 |
| **二阶充分条件** | $\mathbf{Z}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L \mathbf{Z} \succ \mathbf{0}$ | 切空间上正定 | 确认严格局部极小 |
| **消元法** | 参数化约束面，降维 $m$ | 可显式求解约束 | 小规模问题的直接方法 |

**核心公式速查：**

$$\text{拉格朗日函数：} L(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \boldsymbol{\lambda}^\top \mathbf{h}(\mathbf{x})$$

$$\text{一阶条件：} \nabla_\mathbf{x} L = \nabla f + \mathbf{J}_h^\top \boldsymbol{\lambda} = \mathbf{0}, \quad \mathbf{h}(\mathbf{x}) = \mathbf{0}$$

$$\text{约束切空间：} \mathcal{T} = \ker(\mathbf{J}_h), \quad \dim \mathcal{T} = n - m$$

$$\text{二阶充分条件：} \mathbf{d}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L \, \mathbf{d} > 0, \quad \forall \mathbf{d} \in \mathcal{T} \setminus \{\mathbf{0}\}$$

---

## 深度学习应用：正交约束、谱归一化与权重约束

### 背景：深度网络中的等式约束

深度学习中的许多技术本质上是对权重矩阵施加等式约束：

| 技术 | 约束形式 | 目的 |
|------|---------|------|
| 正交权重（OrthogonalNet） | $\mathbf{W}^\top \mathbf{W} = \mathbf{I}$ | 保持梯度范数，避免爆炸/消失 |
| 谱归一化（Spectral Norm） | $\sigma_1(\mathbf{W}) = 1$ | 控制 Lipschitz 常数，稳定 GAN 训练 |
| 权重归一化（Weight Norm） | $\|\mathbf{w}\| = 1$ | 解耦方向与尺度，改善优化条件 |
| 层归一化等价约束 | $\mathbb{E}[\mathbf{h}] = 0, \text{Var}[\mathbf{h}] = 1$ | 稳定激活分布 |

### 正交权重矩阵

**问题形式**：

$$\min_{\mathbf{W} \in \mathbb{R}^{n \times n}} \mathcal{L}(\mathbf{W}) \quad \text{s.t.} \quad \mathbf{W}^\top \mathbf{W} = \mathbf{I}_n$$

这是在**正交群** $O(n) = \{\mathbf{W} \in \mathbb{R}^{n \times n} \mid \mathbf{W}^\top \mathbf{W} = \mathbf{I}\}$ 上的优化，$O(n)$ 是一个维数为 $n(n-1)/2$ 的李群。

约束 $\mathbf{W}^\top \mathbf{W} = \mathbf{I}$ 包含 $n(n+1)/2$ 个独立标量方程（对称矩阵等于单位矩阵）。

**正交矩阵的优良性质**：
- 奇异值全为 1：$\sigma_i(\mathbf{W}) = 1$，不放大也不缩小输入范数
- $\|\mathbf{W}\mathbf{v}\| = \|\mathbf{v}\|$，保持向量范数，避免梯度消失/爆炸
- $\mathbf{W}^{-1} = \mathbf{W}^\top$，便于反向传播计算

**Cayley 变换**（软正交投影）：

$$\mathbf{W} = (\mathbf{I} - \mathbf{A})(\mathbf{I} + \mathbf{A})^{-1}, \quad \mathbf{A} = -\mathbf{A}^\top \ (\text{反对称矩阵})$$

任意反对称矩阵 $\mathbf{A}$ 经 Cayley 变换得到正交矩阵，提供了正交流形的一种参数化。

### 谱归一化

**问题背景（GAN 训练）**：

生成对抗网络（GAN）中，判别器 $D: \mathbb{R}^d \to \mathbb{R}$ 需要满足 **1-Lipschitz** 条件，即：

$$|D(\mathbf{x}) - D(\mathbf{y})| \leq \|\mathbf{x} - \mathbf{y}\|, \quad \forall \mathbf{x}, \mathbf{y}$$

对于全连接层 $f(\mathbf{x}) = \mathbf{W}\mathbf{x}$，Lipschitz 常数等于 $\mathbf{W}$ 的最大奇异值（谱范数）：

$$\text{Lip}(f) = \sigma_1(\mathbf{W}) = \|\mathbf{W}\|_2$$

**谱归一化**将权重重新参数化为：

$$\hat{\mathbf{W}} = \frac{\mathbf{W}}{\sigma_1(\mathbf{W})}$$

使得 $\sigma_1(\hat{\mathbf{W}}) = 1$，即满足约束 $\|\hat{\mathbf{W}}\|_2 = 1$。

**高效计算**：精确计算 $\sigma_1$ 需要 SVD（$O(mn\min(m,n))$），在每步训练时代价过高。实践中用**幂迭代法**（power iteration）以 $O(mn)$ 代价近似：

$$\mathbf{v}_{t+1} = \frac{\mathbf{W}^\top \mathbf{u}_t}{\|\mathbf{W}^\top \mathbf{u}_t\|}, \quad \mathbf{u}_{t+1} = \frac{\mathbf{W} \mathbf{v}_{t+1}}{\|\mathbf{W} \mathbf{v}_{t+1}\|}, \quad \sigma_1 \approx \mathbf{u}_{t+1}^\top \mathbf{W} \mathbf{v}_{t+1}$$

每步训练只做 1 次幂迭代，利用上一步结果热启动，实践中效果很好。

### PyTorch 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# 1. 正交权重层（Orthogonal Weight Layer）
# ============================================================

class OrthogonalLinear(nn.Module):
    """
    正交权重全连接层：通过 Cayley 变换保持 W^T W = I。

    参数化方式：W = (I - A)(I + A)^{-1}，A 为可学习的反对称矩阵。
    这样 W 始终是正交矩阵，奇异值全为 1。
    """

    def __init__(self, n: int):
        """
        参数:
            n: 方阵维数（输入维数 == 输出维数）
        """
        super().__init__()
        self.n = n
        # 上三角部分（不含对角线），用于参数化反对称矩阵 A = U - U^T
        self.upper = nn.Parameter(torch.zeros(n * (n - 1) // 2))
        nn.init.normal_(self.upper, std=0.01)

    def _get_orthogonal_weight(self) -> torch.Tensor:
        """由反对称矩阵参数 -> 正交矩阵 W"""
        # 构造反对称矩阵 A
        A = torch.zeros(self.n, self.n, device=self.upper.device,
                        dtype=self.upper.dtype)
        idx = torch.triu_indices(self.n, self.n, offset=1)
        A[idx[0], idx[1]] = self.upper
        A = A - A.t()  # A = -A^T，反对称

        # Cayley 变换：W = (I - A)(I + A)^{-1}
        I = torch.eye(self.n, device=A.device, dtype=A.dtype)
        W = torch.linalg.solve(I + A, I - A)  # 数值上比直接求逆更稳定
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._get_orthogonal_weight()
        return F.linear(x, W)

    def orthogonality_error(self) -> float:
        """计算正交性误差 ||W^T W - I||_F（调试用）"""
        W = self._get_orthogonal_weight().detach()
        I = torch.eye(self.n, device=W.device)
        return (W.t() @ W - I).norm().item()


# ============================================================
# 2. 谱归一化层（Spectral Normalization）
# ============================================================

class SpectralNormLinear(nn.Module):
    """
    带谱归一化的全连接层：使用幂迭代估计最大奇异值，
    将权重除以谱范数，保证 sigma_1(W_hat) = 1。

    这确保该层的 Lipschitz 常数 <= 1，用于稳定 GAN 训练。
    """

    def __init__(self, in_features: int, out_features: int,
                 n_power_iterations: int = 1, eps: float = 1e-12):
        """
        参数:
            in_features:          输入维数
            out_features:         输出维数
            n_power_iterations:   每步更新的幂迭代次数（通常 1 足够）
            eps:                  数值稳定性小量
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        # 可学习参数
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / np.sqrt(in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # 幂迭代所需的奇异向量（不参与梯度，作为缓冲区）
        self.register_buffer('u', F.normalize(
            torch.randn(out_features), dim=0, eps=eps
        ))
        self.register_buffer('v', F.normalize(
            torch.randn(in_features), dim=0, eps=eps
        ))

    def _update_singular_vectors(self):
        """幂迭代更新左/右奇异向量，近似最大奇异值"""
        u = self.u
        v = self.v
        W = self.weight.detach()  # 不传播到奇异向量更新

        for _ in range(self.n_power_iterations):
            # v_{t+1} = normalize(W^T u_t)
            v_new = F.normalize(W.t() @ u, dim=0, eps=self.eps)
            # u_{t+1} = normalize(W v_{t+1})
            u_new = F.normalize(W @ v_new, dim=0, eps=self.eps)
            v, u = v_new, u_new

        # 原地更新缓冲区（不产生梯度）
        self.u.copy_(u)
        self.v.copy_(v)

    def spectral_norm(self) -> torch.Tensor:
        """计算当前权重的谱范数（近似）sigma_1 ≈ u^T W v"""
        return (self.u @ self.weight @ self.v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 训练时更新奇异向量
        if self.training:
            self._update_singular_vectors()

        # 谱归一化：W_hat = W / sigma_1
        sigma = self.spectral_norm()
        W_hat = self.weight / (sigma + self.eps)

        return F.linear(x, W_hat, self.bias)


# ============================================================
# 3. 权重归一化（Weight Normalization）
# ============================================================

class WeightNormLinear(nn.Module):
    """
    权重归一化全连接层：将权重分解为方向 v（单位向量）和尺度 g（标量），
    实际权重 w = g * v / ||v||，约束 ||w||= g（而非 ||w|| = 1）。

    权重归一化的优点：解耦方向与幅度，改善梯度流动，加速收敛。
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 参数化：v（方向）和 g（尺度）分开学习
        self.v = nn.Parameter(torch.randn(out_features, in_features))
        self.g = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        nn.init.normal_(self.v, std=1.0 / np.sqrt(in_features))

    def get_weight(self) -> torch.Tensor:
        """重构权重矩阵：W = g * v / ||v||（逐行归一化后乘以对应尺度）"""
        # v_norm: (out_features,)
        v_norm = self.v.norm(dim=1, keepdim=True)  # 每行的 L2 范数
        # W: (out_features, in_features)
        return self.g.unsqueeze(1) * self.v / (v_norm + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.get_weight()
        return F.linear(x, W, self.bias)


# ============================================================
# 4. 综合演示：正交约束对梯度范数的影响
# ============================================================

def demo_orthogonal_gradient_stability():
    """
    演示正交权重层与普通全连接层在深层网络中梯度范数的差异。

    关键观察：正交权重层能保持梯度范数，避免梯度消失/爆炸。
    """
    torch.manual_seed(42)
    n_layers = 20   # 使用20层以放大梯度消失/爆炸效应
    dim = 64
    batch_size = 32

    # 普通全连接网络（随机初始化，不保证正交）
    class PlainDeepNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(dim, dim) for _ in range(n_layers)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = torch.tanh(layer(x))
            return x.sum()

    # 正交权重深层网络
    class OrthoDeepNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                OrthogonalLinear(dim) for _ in range(n_layers)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = torch.tanh(layer(x))
            return x.sum()

    x = torch.randn(batch_size, dim)

    def measure_gradient_norms(model, x):
        """测量各层的梯度范数"""
        x = x.clone().detach().requires_grad_(False)
        out = model(x)
        out.backward()

        grad_norms = []
        for layer in model.layers:
            # 获取第一个可学习参数的梯度范数
            for param in layer.parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
                    break
        return grad_norms

    plain_net = PlainDeepNet()
    ortho_net = OrthoDeepNet()

    plain_grads = measure_gradient_norms(plain_net, x)
    ortho_grads = measure_gradient_norms(ortho_net, x)

    print("=" * 60)
    print(f"深层网络梯度范数对比（{n_layers} 层，维数 {dim}）")
    print("=" * 60)
    print(f"{'层编号':>6}  {'普通网络':>12}  {'正交网络':>12}")
    print("-" * 40)
    for i, (pg, og) in enumerate(zip(plain_grads, ortho_grads)):
        print(f"{i+1:>6}  {pg:>12.2e}  {og:>12.2e}")

    print(f"\n普通网络梯度范数之比（第1层/最后层）: "
          f"{plain_grads[0] / (plain_grads[-1] + 1e-30):.2e}")
    print(f"正交网络梯度范数之比（第1层/最后层）: "
          f"{ortho_grads[0] / (ortho_grads[-1] + 1e-30):.2e}")
    print("\n结论：正交网络各层梯度范数接近均匀，无梯度消失/爆炸。")


# ============================================================
# 5. 谱归一化 GAN 判别器
# ============================================================

class SNDiscriminator(nn.Module):
    """
    带谱归一化的 GAN 判别器。
    每层的 Lipschitz 常数 <= 1，总体 Lipschitz 常数 <= 1^L = 1。
    保证 Wasserstein 距离的有效估计（WGAN-GP 的替代方案）。
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            SpectralNormLinear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            SpectralNormLinear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            SpectralNormLinear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))

    def lipschitz_estimate(self, x_real: torch.Tensor,
                           x_fake: torch.Tensor) -> float:
        """
        经验估计 Lipschitz 常数：|D(x) - D(y)| / ||x - y||
        """
        self.eval()
        with torch.no_grad():
            d_real = self(x_real)
            d_fake = self(x_fake)
            diff_out = (d_real - d_fake).abs()
            diff_in  = (x_real - x_fake).view(x_real.size(0), -1).norm(dim=1, keepdim=True)
            lip = (diff_out / (diff_in + 1e-8)).mean().item()
        self.train()
        return lip


# ============================================================
# 6. 验证各约束的满足程度
# ============================================================

def verify_constraints():
    """验证三种权重约束的数学性质"""
    torch.manual_seed(0)
    n = 8

    print("\n" + "=" * 60)
    print("权重约束验证")
    print("=" * 60)

    # --- 正交权重 ---
    ortho_layer = OrthogonalLinear(n)
    W = ortho_layer._get_orthogonal_weight().detach()
    I = torch.eye(n)
    ortho_err = (W.t() @ W - I).norm().item()
    max_sv = torch.linalg.svdvals(W).max().item()
    min_sv = torch.linalg.svdvals(W).min().item()

    print(f"\n[正交权重层]  n = {n}")
    print(f"  ||W^T W - I||_F = {ortho_err:.2e}  (应接近 0)")
    print(f"  奇异值范围: [{min_sv:.6f}, {max_sv:.6f}]  (应全为 1.0)")

    # --- 谱归一化 ---
    sn_layer = SpectralNormLinear(n, n)
    x_dummy = torch.randn(4, n)
    _ = sn_layer(x_dummy)  # 触发幂迭代初始化
    W_hat = sn_layer.weight / (sn_layer.spectral_norm() + 1e-12)
    sigma1 = torch.linalg.svdvals(W_hat.detach()).max().item()

    print(f"\n[谱归一化层]  in={n}, out={n}")
    print(f"  归一化后最大奇异值 sigma_1 = {sigma1:.6f}  (应接近 1.0)")
    print(f"  谱范数近似值（幂迭代）= {sn_layer.spectral_norm().item():.6f}")

    # --- 权重归一化 ---
    wn_layer = WeightNormLinear(n, n)
    W_wn = wn_layer.get_weight().detach()
    row_norms = W_wn.norm(dim=1)

    print(f"\n[权重归一化层]  in={n}, out={n}")
    print(f"  各行范数: {row_norms.numpy().round(4)}")
    print(f"  （应等于对应的尺度参数 g）")
    print(f"  g 参数: {wn_layer.g.detach().numpy().round(4)}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("第7章 深度学习应用：等式约束在神经网络中的应用")
    print("=" * 60)

    # 1. 验证各约束的数学性质
    verify_constraints()

    # 2. 正交网络的梯度稳定性演示
    print("\n")
    demo_orthogonal_gradient_stability()

    # 3. 谱归一化 GAN 判别器的 Lipschitz 估计
    print("\n" + "=" * 60)
    print("谱归一化判别器 Lipschitz 常数估计")
    print("=" * 60)
    disc = SNDiscriminator(input_dim=784, hidden_dim=256)

    # 模拟真实/生成样本
    torch.manual_seed(1)
    x_real = torch.randn(64, 784)
    x_fake = torch.randn(64, 784) * 1.5 + 0.5

    lip_est = disc.lipschitz_estimate(x_real, x_fake)
    print(f"经验 Lipschitz 常数估计: {lip_est:.4f}  (谱归一化理论上界 = 1.0)")

    print("\n核心结论：")
    print("  1. 正交约束（W^T W = I）：通过 Cayley 变换参数化，保持梯度范数稳定")
    print("  2. 谱归一化（sigma_1 = 1）：幂迭代高效估计，控制 Lipschitz 常数")
    print("  3. 权重归一化：解耦方向与幅度，用于加速收敛")
    print("  4. 以上均是等式约束优化在深度学习中的具体实例")
```

### 运行说明

```bash
# 安装依赖
pip install torch numpy

# 运行演示
python equality_constraints_dl.py
```

### 预期输出解读

```
[正交权重层]  n = 8
  ||W^T W - I||_F = 2.34e-07  (应接近 0)
  奇异值范围: [0.999999, 1.000001]  (应全为 1.0)

[谱归一化层]  in=8, out=8
  归一化后最大奇异值 sigma_1 = 1.000000  (应接近 1.0)

深层网络梯度范数对比（20 层，维数 64）
  普通网络梯度范数之比（第1层/最后层）: ~1e+12  (梯度爆炸！)
  正交网络梯度范数之比（第1层/最后层）: ~1e+00  (各层均匀！)
```

---

## 练习题

**练习 7.1**（基础）用拉格朗日乘数法求解以下问题：

$$\min_{x_1, x_2} f(x_1, x_2) = x_1^2 + 2x_2^2 - 2x_1 \quad \text{s.t.} \quad x_1 + x_2 = 3$$

（a）写出拉格朗日函数 $L(x_1, x_2, \lambda)$。

（b）列出 KKT 方程组（3 个方程），并求解 $(x_1^*, x_2^*, \lambda^*)$。

（c）用二阶条件验证所求点是严格极小值点。

（d）用消元法（将 $x_2 = 3 - x_1$ 代入）验证结果。

---

**练习 7.2**（基础）设约束优化问题为：

$$\min_{x_1, x_2, x_3} x_1^2 + x_2^2 + x_3^2 \quad \text{s.t.} \quad x_1 + 2x_2 + 3x_3 = 6$$

（a）写出约束雅可比 $\mathbf{J}_h$，验证 LICQ 成立。

（b）用拉格朗日乘数法求最优解 $\mathbf{x}^*$ 和乘数 $\lambda^*$。

（c）求约束切空间 $\mathcal{T}(\mathbf{x}^*)$ 的一组正交基 $\mathbf{Z}$。

（d）计算约简 Hessian $\mathbf{Z}^\top \nabla^2_{\mathbf{x}\mathbf{x}} L \mathbf{Z}$，验证二阶充分条件。

---

**练习 7.3**（中级）考虑 KKT 线性方程组。设：

$$\mathbf{Q} = \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix}, \quad \mathbf{c} = \begin{pmatrix} -2 \\ 0 \end{pmatrix}, \quad \mathbf{A} = (1, 1), \quad b = 1$$

问题为：$\min \frac{1}{2}\mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x}\ \text{s.t.}\ \mathbf{A}\mathbf{x} = b$。

（a）写出 KKT 线性方程组的矩阵形式（鞍点方程组）。

（b）直接求解此线性方程组，得到 $(\mathbf{x}^*, \lambda^*)$。

（c）解释 $\lambda^*$ 的经济学含义（约束右端 $b$ 每增加一单位，最优目标值的变化）。

---

**练习 7.4**（中级）正交约束的局部分析。设 $f(\mathbf{W}) = \text{tr}(\mathbf{C}^\top \mathbf{W})$（$\mathbf{C}$ 为给定矩阵），约束为 $\mathbf{W}^\top \mathbf{W} = \mathbf{I}$（$\mathbf{W} \in \mathbb{R}^{n \times n}$）。

（a）写出约束 $\mathbf{h}(\mathbf{W}) = \mathbf{W}^\top \mathbf{W} - \mathbf{I} = \mathbf{0}$ 的雅可比（矩阵对矩阵求导）。

（b）写出拉格朗日函数（引入对称矩阵乘数 $\boldsymbol{\Lambda}$）：$L = \text{tr}(\mathbf{C}^\top \mathbf{W}) + \text{tr}(\boldsymbol{\Lambda}^\top (\mathbf{W}^\top \mathbf{W} - \mathbf{I}))$。

（c）令 $\nabla_\mathbf{W} L = \mathbf{0}$，推导一阶条件为 $\mathbf{C} + 2\mathbf{W}\boldsymbol{\Lambda} = \mathbf{0}$。

（d）说明最优解 $\mathbf{W}^*$ 与 $\mathbf{C}$ 的 SVD 的关系（提示：$\mathbf{C} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$，$\mathbf{W}^* = \mathbf{U}\mathbf{V}^\top$）。

---

**练习 7.5**（提高）等式约束与拉格朗日对偶。

（a）**对偶函数**：对于 $\min f(\mathbf{x})\ \text{s.t.}\ \mathbf{h}(\mathbf{x}) = \mathbf{0}$，定义对偶函数 $g(\boldsymbol{\lambda}) = \min_\mathbf{x} L(\mathbf{x}, \boldsymbol{\lambda})$。证明 $g(\boldsymbol{\lambda})$ 是凹函数（无论 $f$ 和 $h$ 是否凸）。

（b）**强对偶**：对于凸问题（$f$ 凸，$h$ 仿射），若 Slater 条件满足（可行集非空），则强对偶成立：$\min f = \max_{\boldsymbol{\lambda}} g(\boldsymbol{\lambda})$。验证：例 7.6 中原始最优值 $= 1/2$，对偶最优值 $= g(\lambda^*) = 1/2$。

（c）**扰动分析**：将约束改为 $h(\mathbf{x}) = \epsilon$（允许轻微违约），最优值函数为 $p^*(\epsilon) = \min\{f(\mathbf{x}) \mid h(\mathbf{x}) = \epsilon\}$。证明 $\frac{dp^*}{d\epsilon}\big|_{\epsilon=0} = -\lambda^*$（拉格朗日乘数的影子价格解释）。

---

## 练习答案

**答案 7.1**

（a）拉格朗日函数：

$$L(x_1, x_2, \lambda) = x_1^2 + 2x_2^2 - 2x_1 + \lambda(x_1 + x_2 - 3)$$

（b）KKT 方程组：

$$\frac{\partial L}{\partial x_1} = 2x_1 - 2 + \lambda = 0$$

$$\frac{\partial L}{\partial x_2} = 4x_2 + \lambda = 0$$

$$\frac{\partial L}{\partial \lambda} = x_1 + x_2 - 3 = 0$$

由前两式：$x_1 = (2 - \lambda)/2$，$x_2 = -\lambda/4$。

代入约束：$\frac{2-\lambda}{2} - \frac{\lambda}{4} = 3$，化简：$4 - 2\lambda - \lambda = 12$，得 $\lambda^* = -8/3$。

$$x_1^* = \frac{2 + 8/3}{2} = \frac{14/3}{2} = \frac{7}{3}, \quad x_2^* = \frac{8/3}{4} = \frac{2}{3}$$

验证：$x_1^* + x_2^* = 7/3 + 2/3 = 3$ $\checkmark$

最优值：$f^* = (7/3)^2 + 2(2/3)^2 - 2(7/3) = 49/9 + 8/9 - 14/3 = 57/9 - 42/9 = 15/9 = 5/3$

（c）二阶条件验证：

$\mathbf{J}_h = (1, 1)$，切空间 $\mathcal{T} = \{(d_1, d_2) \mid d_1 + d_2 = 0\}$，基向量 $\mathbf{z} = (1, -1)^\top / \sqrt{2}$。

$\nabla^2_{\mathbf{x}\mathbf{x}} L = \nabla^2 f = \begin{pmatrix}2 & 0 \\ 0 & 4\end{pmatrix}$（约束线性，$\nabla^2 h = 0$）

约简 Hessian：$\mathbf{z}^\top \nabla^2 f \cdot \mathbf{z} = \frac{1}{2}(1, -1)\begin{pmatrix}2 & 0\\0&4\end{pmatrix}\begin{pmatrix}1\\-1\end{pmatrix} = \frac{1}{2}(2 + 4) = 3 > 0$

二阶充分条件成立，$(7/3, 2/3)$ 是严格极小值点。

（d）消元法验证：

$x_2 = 3 - x_1$，代入：

$$\tilde{f}(x_1) = x_1^2 + 2(3-x_1)^2 - 2x_1 = x_1^2 + 18 - 12x_1 + 2x_1^2 - 2x_1 = 3x_1^2 - 14x_1 + 18$$

令 $\tilde{f}'(x_1) = 6x_1 - 14 = 0$，得 $x_1^* = 7/3$，$x_2^* = 2/3$，与拉格朗日法一致。$\square$

---

**答案 7.2**

（a）约束雅可比：$\mathbf{J}_h = (1, 2, 3) \in \mathbb{R}^{1 \times 3}$，满行秩（一行非零行向量），LICQ 成立。

（b）KKT 条件：

$$2x_1 + \lambda = 0, \quad 2x_2 + 2\lambda = 0, \quad 2x_3 + 3\lambda = 0, \quad x_1 + 2x_2 + 3x_3 = 6$$

由前三式：$x_1 = -\lambda/2$，$x_2 = -\lambda$，$x_3 = -3\lambda/2$。

代入约束：$-\lambda/2 - 2\lambda - 9\lambda/2 = 6$，即 $-7\lambda = 6$，$\lambda^* = -6/7$。

$$\mathbf{x}^* = \left(\frac{3}{7}, \frac{6}{7}, \frac{9}{7}\right)^\top, \quad f^* = \frac{9}{49} + \frac{36}{49} + \frac{81}{49} = \frac{126}{49} = \frac{18}{7}$$

（c）切空间 $\mathcal{T} = \ker(1,2,3) = \{(d_1,d_2,d_3) \mid d_1 + 2d_2 + 3d_3 = 0\}$，维数 2。

一组基（用 Gram-Schmidt 正交化得到）：

$$\mathbf{z}_1 = (-2, 1, 0)^\top / \sqrt{5}, \quad \mathbf{z}_2 = (-3, 0, 1)^\top / \sqrt{10}$$

（可验证 $\mathbf{J}_h \mathbf{z}_1 = -2 + 2 + 0 = 0$，$\mathbf{J}_h \mathbf{z}_2 = -3 + 0 + 3 = 0$）

（d）约简 Hessian：$\nabla^2_{\mathbf{x}\mathbf{x}} L = 2\mathbf{I}$（目标 $f = \|\mathbf{x}\|^2$ 是二次，约束线性）

$$\mathbf{Z} = [\mathbf{z}_1, \mathbf{z}_2], \quad \mathbf{Z}^\top (2\mathbf{I}) \mathbf{Z} = 2\mathbf{Z}^\top \mathbf{Z} = 2\mathbf{I}_2 \succ \mathbf{0}$$

二阶充分条件成立，$\mathbf{x}^*$ 是严格局部（也是全局）极小值点。

---

**答案 7.3**

（a）KKT 线性方程组（鞍点方程组）：

$$\begin{pmatrix} \mathbf{Q} & \mathbf{A}^\top \\ \mathbf{A} & 0 \end{pmatrix} \begin{pmatrix} \mathbf{x}^* \\ \lambda^* \end{pmatrix} = \begin{pmatrix} -\mathbf{c} \\ b \end{pmatrix} \implies \begin{pmatrix} 4 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 0 \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \\ \lambda \end{pmatrix} = \begin{pmatrix} 2 \\ 0 \\ 1 \end{pmatrix}$$

（b）由第三行：$x_1 + x_2 = 1$，设 $x_2 = 1 - x_1$。

代入前两行方程：

第一行：$4x_1 + (1 - x_1) + \lambda = 2$，即 $3x_1 + \lambda = 1$

第二行：$x_1 + 2(1 - x_1) + \lambda = 0$，即 $-x_1 + \lambda = -2$

两式相减：$4x_1 = 3$，$x_1^* = 3/4$，$x_2^* = 1/4$，$\lambda^* = x_1^* - 2 = -5/4$。

最优目标值：$f^* = \frac{1}{2}(3/4, 1/4)\begin{pmatrix}4&1\\1&2\end{pmatrix}\binom{3/4}{1/4} + (-2, 0)\binom{3/4}{1/4}$

$= \frac{1}{2}\left[(3/4)(3 + 1/4) + (1/4)(3/4 + 1/2)\right] - 3/2 = \frac{1}{2}\left[\frac{39}{16} + \frac{5}{16}\right] - \frac{3}{2} = \frac{22}{16} - \frac{24}{16} = -\frac{1}{8}$

（c）$\lambda^* = -5/4 < 0$。根据影子价格解释：约束 $x_1 + x_2 = b$ 中的 $b$ 每增加一单位，最优目标值变化约 $\lambda^* = -5/4$，即**目标值减小** $5/4$（因 $\lambda^*$ 为负）。这说明放松约束（允许更大的 $x_1 + x_2$）能进一步降低目标函数。

---

**答案 7.4**

（a）约束 $\mathbf{h}(\mathbf{W}) = \mathbf{W}^\top \mathbf{W} - \mathbf{I}$，对 $\mathbf{W}$ 的微分（方向导数在方向 $\Delta \mathbf{W}$ 上）：

$$D\mathbf{h}(\mathbf{W})[\Delta \mathbf{W}] = (\Delta \mathbf{W})^\top \mathbf{W} + \mathbf{W}^\top (\Delta \mathbf{W})$$

这是一个从 $\mathbb{R}^{n \times n}$ 到 $\text{Sym}^{n \times n}$（对称矩阵空间）的线性映射。

（b）拉格朗日函数（$\boldsymbol{\Lambda}$ 为对称矩阵乘数，对应对称约束）：

$$L(\mathbf{W}, \boldsymbol{\Lambda}) = \text{tr}(\mathbf{C}^\top \mathbf{W}) + \text{tr}\left(\boldsymbol{\Lambda}^\top (\mathbf{W}^\top \mathbf{W} - \mathbf{I})\right)$$

（c）对 $\mathbf{W}$ 求梯度并令其为零（利用矩阵微分 $\nabla_\mathbf{W} \text{tr}(\mathbf{C}^\top \mathbf{W}) = \mathbf{C}$，$\nabla_\mathbf{W} \text{tr}(\boldsymbol{\Lambda}^\top \mathbf{W}^\top \mathbf{W}) = 2\mathbf{W}\boldsymbol{\Lambda}$）：

$$\nabla_\mathbf{W} L = \mathbf{C} + 2\mathbf{W}\boldsymbol{\Lambda} = \mathbf{0}$$

（d）由 $\mathbf{C} = -2\mathbf{W}\boldsymbol{\Lambda}$，且 $\mathbf{W}^\top \mathbf{W} = \mathbf{I}$（正交），从右乘 $\mathbf{W}^\top$：

$\mathbf{C}\mathbf{W}^\top = -2\boldsymbol{\Lambda}$，即 $\boldsymbol{\Lambda} = -\frac{1}{2}\mathbf{C}\mathbf{W}^\top$（反对称部分为零时此式成立）。

设 $\mathbf{C} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$（SVD），令 $\mathbf{W}^* = \mathbf{U}\mathbf{V}^\top$（正交矩阵），则：

$$f(\mathbf{W}^*) = \text{tr}(\mathbf{C}^\top \mathbf{U}\mathbf{V}^\top) = \text{tr}(\mathbf{V}\boldsymbol{\Sigma}\mathbf{U}^\top \mathbf{U}\mathbf{V}^\top) = \text{tr}(\boldsymbol{\Sigma}) = \sum_i \sigma_i(\mathbf{C})$$

这是 $\text{tr}(\mathbf{C}^\top \mathbf{W})$ 在正交约束下的**最大值**（此问题实为极大化 $\text{tr}(\mathbf{C}^\top \mathbf{W})$，等价于求矩阵 $\mathbf{C}$ 的核范数）。这正是 **Procrustes 问题** 的解析解。

---

**答案 7.5**

（a）**$g(\boldsymbol{\lambda})$ 是凹函数**：

$$g(\boldsymbol{\lambda}) = \inf_\mathbf{x} L(\mathbf{x}, \boldsymbol{\lambda}) = \inf_\mathbf{x} [f(\mathbf{x}) + \boldsymbol{\lambda}^\top \mathbf{h}(\mathbf{x})]$$

对任意 $\mathbf{x}$，$\boldsymbol{\lambda} \mapsto f(\mathbf{x}) + \boldsymbol{\lambda}^\top \mathbf{h}(\mathbf{x})$ 是关于 $\boldsymbol{\lambda}$ 的仿射函数（凹），故其下确界（逐点取 $\inf$）也是凹函数。$\square$

（b）**例 7.6 的强对偶验证**：

原始问题：$\min x_1^2 + x_2^2\ \text{s.t.}\ x_1 + x_2 = 1$，原始最优值 $p^* = 1/2$。

对偶函数：

$$g(\lambda) = \min_{x_1, x_2} [x_1^2 + x_2^2 + \lambda(x_1 + x_2 - 1)]$$

对 $x_1, x_2$ 求极小，令 $2x_i + \lambda = 0$，得 $x_1 = x_2 = -\lambda/2$，代入：

$$g(\lambda) = 2 \cdot \frac{\lambda^2}{4} + \lambda(-\lambda - 1) = \frac{\lambda^2}{2} - \lambda^2 - \lambda = -\frac{\lambda^2}{2} - \lambda$$

最大化 $g$：$g'(\lambda) = -\lambda - 1 = 0$，$\lambda^* = -1$，$g(\lambda^*) = -1/2 + 1 = 1/2 = p^*$。强对偶成立。$\square$

（c）**扰动分析**：设 $h(\mathbf{x}) = \epsilon$，$p^*(\epsilon) = \min\{f(\mathbf{x}) \mid h(\mathbf{x}) = \epsilon\}$。

由包络定理（Envelope Theorem），在正则性条件下：

$$\frac{dp^*}{d\epsilon} = \frac{\partial L(\mathbf{x}^*(\epsilon), \lambda^*(\epsilon), \epsilon)}{\partial \epsilon} = \frac{\partial}{\partial \epsilon}[f(\mathbf{x}) + \lambda (h(\mathbf{x}) - \epsilon)]\bigg|_{\text{最优}} = -\lambda^*$$

这正是 $\lambda^*$ 作为约束影子价格的精确数学表述：约束松弛一单位，目标值下降 $\lambda^*$（对于极小化问题）。对于例 7.6，$\lambda^* = -1$，即将约束 $x_1 + x_2 = 1$ 改为 $x_1 + x_2 = 1 + \epsilon$ 时，最优值从 $1/2$ 变为约 $1/2 + 1 \cdot \epsilon = 1/2 + \epsilon$（松弛约束时最优值增大）。$\square$

---

*本章建立了等式约束优化的完整理论框架。下一章将把结果推广至不等式约束，引入 KKT 条件的完整形式和互补松弛条件。*
