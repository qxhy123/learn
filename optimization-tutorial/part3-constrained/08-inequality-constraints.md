# 第8章：不等式约束优化

> **前置知识**：第4章（最优性条件）、第7章（等式约束优化与 Lagrange 乘子法）
>
> **本章难度**：★★★★☆
>
> **预计学习时间**：6-8 小时

---

## 学习目标

学完本章后，你将能够：

1. **建立不等式约束优化的标准形式**，理解可行域的几何结构，区分活跃约束与非活跃约束
2. **推导并应用 KKT 条件**：掌握梯度条件、原始可行性、对偶可行性和互补松弛条件的完整含义
3. **理解互补松弛性的几何直觉**，利用互补松弛条件分析约束的"松紧"状态
4. **认识主要约束规范（CQ）**，理解 LICQ 和 Slater 条件，知晓 KKT 条件成立的前提
5. **掌握二阶充分条件**，利用约化 Hessian 判断 KKT 点是否为严格局部极小

---

## 8.1 不等式约束问题

### 8.1.1 标准问题形式

本章研究的一般约束优化问题为：

$$\boxed{\begin{aligned}
\min_{\mathbf{x} \in \mathbb{R}^n} \quad & f(\mathbf{x}) \\
\text{s.t.} \quad & g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m \\
& h_j(\mathbf{x}) = 0, \quad j = 1, \ldots, p
\end{aligned}}$$

其中：
- $f: \mathbb{R}^n \to \mathbb{R}$ 为**目标函数**（objective function）
- $g_i: \mathbb{R}^n \to \mathbb{R}$ 为**不等式约束函数**（inequality constraint）
- $h_j: \mathbb{R}^n \to \mathbb{R}$ 为**等式约束函数**（equality constraint）

**可行域**定义为满足所有约束的点集：

$$\mathcal{F} = \{\mathbf{x} \in \mathbb{R}^n \mid g_i(\mathbf{x}) \leq 0,\; i=1,\ldots,m;\; h_j(\mathbf{x})=0,\; j=1,\ldots,p\}$$

**假设**：本章假设 $f, g_i, h_j$ 均为二阶连续可微函数（$C^2$ 类）。

### 8.1.2 活跃约束与非活跃约束

**定义 8.1（活跃约束）**
设 $\mathbf{x}^* \in \mathcal{F}$。不等式约束 $g_i$ 在 $\mathbf{x}^*$ 处称为：
- **活跃约束**（active constraint）：若 $g_i(\mathbf{x}^*) = 0$
- **非活跃约束**（inactive constraint）：若 $g_i(\mathbf{x}^*) < 0$

活跃约束的下标集合称为**活跃集**（active set）：

$$\mathcal{A}(\mathbf{x}^*) = \{i \mid g_i(\mathbf{x}^*) = 0\}$$

**几何直觉：** 活跃约束意味着点 $\mathbf{x}^*$ 恰好位于约束边界上；非活跃约束则意味着点在约束的内部，该约束对当前点的局部行为不构成限制。

### 8.1.3 几何例子

**例 8.1** 考察二维问题：

$$\min_{x_1, x_2} \quad (x_1 - 2)^2 + (x_2 - 2)^2 \quad \text{s.t.} \quad x_1 + x_2 \leq 3,\; x_1 \geq 0,\; x_2 \geq 0.$$

无约束极小为 $(2, 2)$，但 $2 + 2 = 4 > 3$，违反第一个约束。因此最优解在约束边界 $x_1 + x_2 = 3$ 上（活跃约束），通过 Lagrange 分析可得 $\mathbf{x}^* = (3/2, 3/2)$。

此时 $g_1(\mathbf{x}^*) = 0$（活跃），$g_2(\mathbf{x}^*) = -3/2 < 0$，$g_3(\mathbf{x}^*) = -3/2 < 0$（非活跃）。

### 8.1.4 约束化简的思路

不等式约束问题的难点在于：**哪些约束在最优解处活跃是未知的**。如果我们事先知道活跃集 $\mathcal{A}(\mathbf{x}^*)$，则问题退化为等式约束问题（第7章的情形）。KKT 条件正是在不知道活跃集的情况下，给出最优性的必要条件。

---

## 8.2 KKT 条件

### 8.2.1 Lagrange 函数

对一般约束问题，定义**Lagrange 函数**（Lagrangian）：

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\mu}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_{i=1}^{m} \mu_i g_i(\mathbf{x}) + \sum_{j=1}^{p} \lambda_j h_j(\mathbf{x})$$

其中：
- $\mu_i \geq 0$ 为不等式约束 $g_i$ 的 **KKT 乘子**（亦称对偶变量）
- $\lambda_j \in \mathbb{R}$ 为等式约束 $h_j$ 的 **Lagrange 乘子**

**关键区别**：不等式约束的乘子要求 $\mu_i \geq 0$（对偶可行性），而等式约束乘子无符号约束。

### 8.2.2 KKT 条件的推导动机

设 $\mathbf{x}^*$ 是局部极小点。对于非活跃约束 $g_i(\mathbf{x}^*) < 0$，在 $\mathbf{x}^*$ 的某邻域内该约束不起作用，因此对局部分析无影响，可令 $\mu_i = 0$。

对于活跃约束 $g_i(\mathbf{x}^*) = 0$，情况类似于等式约束，但需额外考虑可行下降方向。直觉上，若存在方向 $\mathbf{d}$ 使得 $f$ 减小（$\nabla f^\top \mathbf{d} < 0$）且不违反活跃约束（$\nabla g_i^\top \mathbf{d} \leq 0$），则 $\mathbf{x}^*$ 不是极小点。

### 8.2.3 KKT 条件（一阶必要条件）

**定理 8.1（KKT 必要条件）**
设 $\mathbf{x}^*$ 是问题的局部极小点，且在 $\mathbf{x}^*$ 处某约束规范（见 8.4 节）成立。则存在乘子向量 $\boldsymbol{\mu}^* \in \mathbb{R}^m$，$\boldsymbol{\lambda}^* \in \mathbb{R}^p$，使得以下四组条件同时成立：

**（一）梯度条件（Stationarity）：**

$$\nabla f(\mathbf{x}^*) + \sum_{i=1}^{m} \mu_i^* \nabla g_i(\mathbf{x}^*) + \sum_{j=1}^{p} \lambda_j^* \nabla h_j(\mathbf{x}^*) = \mathbf{0}$$

即 $\nabla_\mathbf{x} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*) = \mathbf{0}$。

**（二）原始可行性（Primal Feasibility）：**

$$g_i(\mathbf{x}^*) \leq 0, \quad i = 1, \ldots, m$$
$$h_j(\mathbf{x}^*) = 0, \quad j = 1, \ldots, p$$

**（三）对偶可行性（Dual Feasibility）：**

$$\mu_i^* \geq 0, \quad i = 1, \ldots, m$$

**（四）互补松弛条件（Complementary Slackness）：**

$$\mu_i^* \cdot g_i(\mathbf{x}^*) = 0, \quad i = 1, \ldots, m$$

满足上述四组条件的点 $\mathbf{x}^*$ 称为 **KKT 点**，对应乘子 $(\boldsymbol{\mu}^*, \boldsymbol{\lambda}^*)$ 称为 **KKT 乘子**。

### 8.2.4 梯度条件的几何解释

梯度条件

$$\nabla f(\mathbf{x}^*) = -\sum_{i \in \mathcal{A}} \mu_i^* \nabla g_i(\mathbf{x}^*) - \sum_{j=1}^{p} \lambda_j^* \nabla h_j(\mathbf{x}^*)$$

表明：**在极小点处，目标函数的梯度必须可以用活跃约束梯度的非负线性组合（加上等式约束梯度的线性组合）来表示**。

几何上，$-\nabla f(\mathbf{x}^*)$ 必须在活跃不等式约束梯度和等式约束梯度张成的**法锥**（normal cone）内。

**一维直觉：** 若在可行域边界 $g(\mathbf{x}^*) = 0$ 处极小，则 $\nabla f$ 必须指向约束外侧（否则可沿约束内侧继续减小 $f$），即 $\nabla f = -\mu \nabla g$，$\mu > 0$。

### 8.2.5 完整例子：二维不等式约束问题

**例 8.2** 求解：

$$\min_{x_1, x_2} \quad f = x_1^2 + x_2^2 \quad \text{s.t.} \quad g: x_1 + x_2 - 1 \leq 0.$$

**分析：** 无约束极小为原点 $(0,0)$，满足 $g(0,0) = -1 < 0$（约束非活跃），故原点即为全局极小，KKT 乘子 $\mu^* = 0$。

**验证 KKT 条件：**
- 梯度条件：$\nabla f + \mu \nabla g = (0,0) + 0 \cdot (1,1) = (0,0)$。✓
- 原始可行：$0 + 0 - 1 = -1 \leq 0$。✓
- 对偶可行：$\mu^* = 0 \geq 0$。✓
- 互补松弛：$0 \cdot (-1) = 0$。✓

**例 8.3** 将上例约束改为 $g: x_1 + x_2 - 1 \geq 0$（等价地 $-(x_1+x_2-1) \leq 0$）。

此时无约束极小 $(0,0)$ 违反约束，最优解在边界 $x_1 + x_2 = 1$ 上。

令 $g'(\mathbf{x}) = 1 - x_1 - x_2 \leq 0$（改写为标准形式）。

KKT 条件：

$$\begin{pmatrix} 2x_1 \\ 2x_2 \end{pmatrix} + \mu \begin{pmatrix} -1 \\ -1 \end{pmatrix} = \mathbf{0}, \quad 1 - x_1 - x_2 \leq 0, \quad \mu \geq 0, \quad \mu(1 - x_1 - x_2) = 0.$$

由梯度条件：$x_1 = x_2 = \mu/2$。由互补松弛及约束活跃（因无约束极小在可行域外），$1 - x_1 - x_2 = 0$，解得 $x_1 = x_2 = 1/2$，$\mu = 1 > 0$。KKT 点为 $(1/2, 1/2)$，目标值 $f^* = 1/2$。

---

## 8.3 互补松弛性

### 8.3.1 互补松弛的含义

互补松弛条件 $\mu_i^* \cdot g_i(\mathbf{x}^*) = 0$ 意味着，对每个不等式约束，以下两种情形**至少有一个成立**：

| 情形 | 条件 | 解释 |
|------|------|------|
| **约束非活跃** | $g_i(\mathbf{x}^*) < 0$ | 点在约束内部，约束"松弛"，此时 $\mu_i^* = 0$ |
| **乘子为零** | $\mu_i^* = 0$ | 约束对最优解不施加"压力"，即使活跃也不影响梯度平衡 |
| **两者均成立** | $g_i(\mathbf{x}^*) < 0$ 且 $\mu_i^* = 0$ | 约束完全不活跃 |

**不可能**：$g_i(\mathbf{x}^*) < 0$ 且 $\mu_i^* > 0$（内部点有正乘子——物理上不合理）。

### 8.3.2 严格互补松弛

**定义 8.2（严格互补松弛）**
若对所有 $i \in \mathcal{A}(\mathbf{x}^*)$，均有 $\mu_i^* > 0$，则称严格互补松弛条件（strict complementarity）在 $\mathbf{x}^*$ 处成立。

严格互补松弛保证了活跃约束的乘子严格为正，是许多精细分析（如二阶条件、灵敏度分析）的正则性假设。

### 8.3.3 互补松弛的经济学解释

在资源分配问题中，不等式约束 $g_i(\mathbf{x}) \leq 0$ 可理解为"资源 $i$ 的容量限制"：

- $g_i(\mathbf{x}^*) < 0$：资源 $i$ **未被完全利用**（有剩余），边际价值 $\mu_i^* = 0$（增加容量不带来收益）。
- $g_i(\mathbf{x}^*) = 0$：资源 $i$ **被完全耗尽**（紧张），边际价值 $\mu_i^* \geq 0$（增加容量可能带来收益）。

互补松弛正是"不浪费有价值资源"原则的数学表达。

### 8.3.4 互补松弛与支撑向量机

互补松弛在机器学习中有重要应用。以**支撑向量机（SVM）**为例：

$$\min_{\mathbf{w}, b} \quad \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1, \quad i=1,\ldots,N.$$

改写为标准形式 $g_i = 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b) \leq 0$，KKT 互补松弛为：

$$\mu_i^* [1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)] = 0.$$

这说明：
- **支撑向量**（$\mu_i^* > 0$）：必然满足 $y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$（恰好在间隔边界上）。
- **非支撑向量**（$y_i(\mathbf{w}^\top \mathbf{x}_i + b) > 1$）：必然 $\mu_i^* = 0$，对最优解无贡献。

SVM 的稀疏性（仅少数支撑向量决定分类器）正是互补松弛的直接推论。

---

## 8.4 约束规范

### 8.4.1 为什么需要约束规范

KKT 条件是**必要条件**，但并不总是成立——需要在极小点处满足某种**约束规范**（Constraint Qualification, CQ），才能保证 KKT 乘子的存在性。

**反例（违反约束规范）：**

$$\min_{x_1, x_2} \quad x_2 \quad \text{s.t.} \quad g: x_1^2 - x_2 \leq 0,\; x_2 \leq 0.$$

最优解为 $\mathbf{x}^* = (0, 0)$。两个约束在 $\mathbf{x}^*$ 处均活跃，但 $\nabla g_1(0,0) = (0,-1)$，$\nabla g_2(0,0) = (0,1)$，两者方向相反，线性相关。KKT 条件 $\nabla f + \mu_1 \nabla g_1 + \mu_2 \nabla g_2 = 0$ 变为 $(0,1) + \mu_1(0,-1) + \mu_2(0,1) = 0$，即 $-\mu_1 + \mu_2 = -1$，$\mu_1, \mu_2 \geq 0$ 无解。此例违反 LICQ。

### 8.4.2 线性无关约束规范（LICQ）

**定义 8.3（LICQ）**
在点 $\mathbf{x}^*$ 处，称**线性无关约束规范**（Linear Independence Constraint Qualification）成立，若活跃约束的梯度集合

$$\{\nabla g_i(\mathbf{x}^*) \mid i \in \mathcal{A}(\mathbf{x}^*)\} \cup \{\nabla h_j(\mathbf{x}^*) \mid j=1,\ldots,p\}$$

线性无关。

**定理 8.2（LICQ 保证 KKT）**
若 $\mathbf{x}^*$ 是局部极小点且 LICQ 在 $\mathbf{x}^*$ 处成立，则存在唯一 KKT 乘子 $(\boldsymbol{\mu}^*, \boldsymbol{\lambda}^*)$。

LICQ 是最常用的约束规范，在大多数实际问题中成立。

### 8.4.3 Slater 条件（凸问题）

对于**凸优化问题**（$f, g_i$ 均为凸函数，$h_j$ 为仿射函数），有更弱的约束规范：

**定义 8.4（Slater 条件）**
若存在 $\tilde{\mathbf{x}} \in \mathbb{R}^n$（称为**Slater 点**）使得：

$$g_i(\tilde{\mathbf{x}}) < 0, \quad \forall\, i=1,\ldots,m, \qquad h_j(\tilde{\mathbf{x}}) = 0, \quad \forall\, j=1,\ldots,p,$$

则称 Slater 条件成立。

**定理 8.3（Slater 条件与强对偶性）**
若凸优化问题满足 Slater 条件，则：
1. KKT 条件是极小点的**充要条件**（不仅是必要条件）。
2. 强对偶性成立：原始问题与对偶问题的最优值相等（对偶间隙为零）。

**Slater 条件 vs LICQ 的比较：**

| 特征 | LICQ | Slater 条件 |
|------|------|-------------|
| 适用范围 | 一般光滑问题 | 凸问题 |
| 强度 | 较强（逐点线性无关）| 较弱（只需一个严格内点）|
| 乘子唯一性 | 保证唯一 | 不保证唯一 |
| 与对偶理论 | 关系不直接 | 直接推出强对偶 |

### 8.4.4 其他常见约束规范

| 约束规范 | 条件 | 特点 |
|----------|------|------|
| **LICQ** | 活跃约束梯度线性无关 | 最强，乘子唯一 |
| **MFCQ**（Mangasarian-Fromovitz）| 存在方向使所有活跃约束梯度严格负 | 弱于 LICQ，乘子可不唯一 |
| **CRCQ**（常秩约束规范）| 活跃约束梯度的秩在邻域内恒定 | 介于 LICQ 与 MFCQ 之间 |
| **Slater 条件** | 存在严格可行点（凸问题专用）| 最弱，最易验证 |

在实际应用中，**通常先验证 Slater 条件**（凸问题）或**假设 LICQ 成立**，再直接写出 KKT 条件。

---

## 8.5 二阶充分条件

### 8.5.1 KKT 点的进一步分析

KKT 条件是（在约束规范下的）必要条件，不是充分条件。KKT 点可能是极小点、极大点或鞍点。二阶条件提供了充分性判断。

### 8.5.2 临界锥

**定义 8.5（临界锥）**
设 $\mathbf{x}^*$ 是 KKT 点，乘子为 $(\boldsymbol{\mu}^*, \boldsymbol{\lambda}^*)$。**临界锥** $\mathcal{C}(\mathbf{x}^*, \boldsymbol{\mu}^*)$ 定义为：

$$\mathcal{C}(\mathbf{x}^*, \boldsymbol{\mu}^*) = \left\{ \mathbf{d} \in \mathbb{R}^n \;\middle|\;
\begin{array}{l}
\nabla h_j(\mathbf{x}^*)^\top \mathbf{d} = 0, \quad j=1,\ldots,p \\
\nabla g_i(\mathbf{x}^*)^\top \mathbf{d} = 0, \quad i \in \mathcal{A}(\mathbf{x}^*),\; \mu_i^* > 0 \\
\nabla g_i(\mathbf{x}^*)^\top \mathbf{d} \leq 0, \quad i \in \mathcal{A}(\mathbf{x}^*),\; \mu_i^* = 0
\end{array}
\right\}$$

临界锥是"可行下降方向"候选集的线性化近似——在这些方向上，一阶条件既不能排除也不能确认极小性，需要二阶信息。

### 8.5.3 约化 Hessian

**定义 8.6（Lagrange 函数的 Hessian）**

$$\nabla^2_{\mathbf{x}\mathbf{x}} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*) = \nabla^2 f(\mathbf{x}^*) + \sum_{i=1}^{m} \mu_i^* \nabla^2 g_i(\mathbf{x}^*) + \sum_{j=1}^{p} \lambda_j^* \nabla^2 h_j(\mathbf{x}^*)$$

这是 Lagrange 函数对 $\mathbf{x}$ 的二阶导数（在 KKT 点处计算）。

### 8.5.4 二阶充分条件

**定理 8.4（二阶充分条件，SOSC）**
设 $(\mathbf{x}^*, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*)$ 满足 KKT 条件。若

$$\mathbf{d}^\top \nabla^2_{\mathbf{x}\mathbf{x}} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*) \mathbf{d} > 0 \quad \forall\, \mathbf{d} \in \mathcal{C}(\mathbf{x}^*, \boldsymbol{\mu}^*) \setminus \{\mathbf{0}\},$$

则 $\mathbf{x}^*$ 是问题的**严格局部极小点**。

**直觉：** 在临界锥方向（"可疑方向"）上，Lagrange 函数的 Hessian 正定，则函数值沿这些方向是严格上凸的，不存在下降方向。

### 8.5.5 二阶必要条件

**定理 8.5（二阶必要条件，SONC）**
设 $\mathbf{x}^*$ 是局部极小点，LICQ 成立，$(\boldsymbol{\mu}^*, \boldsymbol{\lambda}^*)$ 为 KKT 乘子，则

$$\mathbf{d}^\top \nabla^2_{\mathbf{x}\mathbf{x}} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*) \mathbf{d} \geq 0 \quad \forall\, \mathbf{d} \in \mathcal{C}(\mathbf{x}^*, \boldsymbol{\mu}^*).$$

### 8.5.6 凸问题的特殊情形

**定理 8.6（凸问题的 KKT 充要条件）**
若 $f, g_i$ 均为凸函数，$h_j$ 为仿射函数，且 Slater 条件成立，则：

$$\mathbf{x}^* \text{ 是全局极小点} \iff (\mathbf{x}^*, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*) \text{ 满足 KKT 条件（某乘子存在）}$$

此时 SOSC 自动满足（凸函数的 Hessian 半正定），无需单独验证。

### 8.5.7 综合示例

**例 8.4** 分析问题：

$$\min_{x_1, x_2} \quad f = x_1^2 + x_2^2 - 2x_1 \quad \text{s.t.} \quad g: x_1^2 + x_2^2 - 1 \leq 0.$$

**步骤 1：无约束极小为 $(1, 0)$，满足 $g(1,0) = 1 - 1 = 0$（约束活跃），需用 KKT。**

**步骤 2：写出 KKT 条件：**

$$\begin{pmatrix} 2x_1 - 2 \\ 2x_2 \end{pmatrix} + \mu \begin{pmatrix} 2x_1 \\ 2x_2 \end{pmatrix} = \mathbf{0}, \quad x_1^2 + x_2^2 \leq 1, \quad \mu \geq 0, \quad \mu(x_1^2 + x_2^2 - 1) = 0.$$

**情形 1（约束非活跃，$\mu = 0$）：** $2x_1 - 2 = 0$，$2x_2 = 0$，得 $(1, 0)$。但 $g(1,0) = 0$（边界），矛盾。

**情形 2（约束活跃，$x_1^2 + x_2^2 = 1$）：** 由梯度条件 $x_2(1+\mu) = 0$，知 $x_2 = 0$ 或 $\mu = -1$（后者违反对偶可行性）。故 $x_2 = 0$，$x_1 = \pm 1$。

- $(x_1, x_2) = (1, 0)$：梯度条件给出 $\mu = 0$，满足所有 KKT 条件。
- $(x_1, x_2) = (-1, 0)$：梯度条件给出 $\mu = -1 < 0$，违反对偶可行性，非 KKT 点。

**步骤 3：验证 SOSC（$\mathbf{x}^* = (1,0)$，$\mu^* = 0$）：**

$$\nabla^2_{\mathbf{x}\mathbf{x}} \mathcal{L} = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} + 0 \cdot \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}.$$

临界锥：$\mu^* = 0$（退化情形，活跃约束 $g$ 约束方向 $\nabla g = (2,0)$）。临界锥 $\mathcal{C} = \{\mathbf{d} \mid \nabla g^\top \mathbf{d} \leq 0\} = \{d_1 \leq 0\}$（结合 SOSC 需在非零锥向量处验证）。

$\nabla^2 \mathcal{L}$ 正定，SOSC 成立，故 $(1, 0)$ 是严格局部极小点，$f^* = 1 - 2 = -1$。

---

## 本章小结

| 概念 | 数学表达 | 核心作用 |
|------|---------|---------|
| **标准不等式约束问题** | $\min f(\mathbf{x})$ s.t. $g_i \leq 0$, $h_j = 0$ | 约束优化的统一框架 |
| **活跃约束集** | $\mathcal{A}(\mathbf{x}^*) = \{i \mid g_i(\mathbf{x}^*) = 0\}$ | 识别在极小点处"起作用"的约束 |
| **Lagrange 函数** | $\mathcal{L} = f + \sum \mu_i g_i + \sum \lambda_j h_j$ | KKT 条件的统一表达载体 |
| **梯度条件** | $\nabla_\mathbf{x} \mathcal{L} = \mathbf{0}$ | 一阶稳定性：梯度由约束法向平衡 |
| **原始可行性** | $g_i(\mathbf{x}^*) \leq 0$，$h_j(\mathbf{x}^*) = 0$ | 解在可行域内 |
| **对偶可行性** | $\mu_i^* \geq 0$ | 不等式乘子非负 |
| **互补松弛** | $\mu_i^* g_i(\mathbf{x}^*) = 0$ | 非活跃约束的乘子为零 |
| **LICQ** | 活跃约束梯度线性无关 | 保证 KKT 乘子存在且唯一 |
| **Slater 条件** | 凸问题存在严格可行点 | 凸问题的弱约束规范，推出强对偶 |
| **SOSC** | $\mathbf{d}^\top \nabla^2 \mathcal{L}\, \mathbf{d} > 0$，$\forall \mathbf{d} \in \mathcal{C} \setminus\{\mathbf{0}\}$ | 充分判断严格局部极小 |
| **凸问题 KKT** | Slater + KKT $\iff$ 全局极小 | 凸优化的核心判别定理 |

---

## 深度学习应用：约束神经网络与投影梯度法

### 背景

深度学习中常见以下约束场景：
1. **权重裁剪（Weight Clipping）**：$\|\mathbf{w}\|_\infty \leq c$（Wasserstein GAN）
2. **谱范数约束**：$\sigma_{\max}(W) \leq 1$（谱归一化）
3. **概率单纯形约束**：$\mathbf{w} \geq 0$，$\mathbf{1}^\top \mathbf{w} = 1$（注意力权重）
4. **能量约束**：$\|\mathbf{w}\|_2 \leq R$（$L_2$ 球约束）

这些约束可通过 KKT 条件分析最优性，并用**投影梯度法**高效求解。

### 投影梯度法的原理

投影梯度法（Projected Gradient Descent, PGD）的更新规则为：

$$\mathbf{x}^{(k+1)} = \mathcal{P}_{\mathcal{F}}\!\left(\mathbf{x}^{(k)} - \eta \nabla f(\mathbf{x}^{(k)})\right)$$

其中 $\mathcal{P}_{\mathcal{F}}$ 是到可行域 $\mathcal{F}$ 的**正交投影**：

$$\mathcal{P}_{\mathcal{F}}(\mathbf{y}) = \arg\min_{\mathbf{x} \in \mathcal{F}} \|\mathbf{x} - \mathbf{y}\|_2^2$$

**KKT 与投影的关系：** 投影操作本质上在求解一个有约束的二次规划，其 KKT 条件正好给出投影点满足的最优性条件。

### 常见约束集的投影公式

| 约束集 | 定义 | 投影公式 |
|--------|------|---------|
| $L_2$ 球 | $\|\mathbf{x}\|_2 \leq R$ | $\mathcal{P}(\mathbf{y}) = R \cdot \mathbf{y} / \max(\|\mathbf{y}\|_2, R)$ |
| $L_\infty$ 球 | $\|\mathbf{x}\|_\infty \leq c$ | $\mathcal{P}(\mathbf{y}) = \text{clip}(\mathbf{y}, -c, c)$ |
| 非负象限 | $\mathbf{x} \geq \mathbf{0}$ | $\mathcal{P}(\mathbf{y}) = \max(\mathbf{y}, \mathbf{0})$ |
| 概率单纯形 | $\mathbf{x} \geq \mathbf{0}$，$\mathbf{1}^\top\mathbf{x} = 1$ | Duchi 等人（2008）排序投影算法 |

### 代码实现

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. KKT 条件的数值验证工具
# ============================================================

def check_kkt_conditions(f, g_list, h_list, x_star, mu_star, lam_star,
                          tol=1e-4, verbose=True):
    """
    验证候选点是否满足 KKT 条件

    参数:
        f       : 目标函数（torch 函数）
        g_list  : 不等式约束列表（g_i(x) <= 0）
        h_list  : 等式约束列表（h_j(x) = 0）
        x_star  : 候选极小点
        mu_star : 不等式乘子（与 g_list 等长）
        lam_star: 等式乘子（与 h_list 等长）
        tol     : 数值容差
    """
    x = x_star.clone().detach().requires_grad_(True)
    n = x.shape[0]

    # --- 计算目标梯度 ---
    fx = f(x)
    grad_f = torch.autograd.grad(fx, x, create_graph=False)[0].detach()

    # --- 计算约束梯度 ---
    grad_g = []
    g_vals = []
    for gi in g_list:
        xi = x_star.clone().detach().requires_grad_(True)
        gval = gi(xi)
        gg = torch.autograd.grad(gval, xi)[0].detach()
        grad_g.append(gg)
        g_vals.append(gval.item())

    grad_h = []
    h_vals = []
    for hj in h_list:
        xi = x_star.clone().detach().requires_grad_(True)
        hval = hj(xi)
        gh = torch.autograd.grad(hval, xi)[0].detach()
        grad_h.append(gh)
        h_vals.append(hval.item())

    # --- 检验梯度条件 ---
    stationarity = grad_f.clone()
    for mu_i, gg in zip(mu_star, grad_g):
        stationarity = stationarity + mu_i * gg
    for lam_j, gh in zip(lam_star, grad_h):
        stationarity = stationarity + lam_j * gh
    grad_norm = stationarity.norm().item()

    # --- 检验各条件 ---
    primal_feas_g = all(gv <= tol for gv in g_vals)
    primal_feas_h = all(abs(hv) <= tol for hv in h_vals)
    dual_feas     = all(mu >= -tol for mu in mu_star)
    comp_slack    = all(abs(mu_i * gv) <= tol
                        for mu_i, gv in zip(mu_star, g_vals))

    if verbose:
        print("=" * 55)
        print("KKT 条件验证")
        print("=" * 55)
        print(f"候选点 x*        : {x_star.tolist()}")
        print(f"目标值 f(x*)     : {fx.item():.6f}")
        print(f"  不等式约束 g_i : {[f'{v:.4f}' for v in g_vals]}")
        print(f"  等式约束 h_j   : {[f'{v:.4f}' for v in h_vals]}")
        print(f"  KKT 乘子 μ    : {[f'{v:.4f}' for v in mu_star]}")
        print(f"  KKT 乘子 λ    : {[f'{v:.4f}' for v in lam_star]}")
        print("-" * 55)
        print(f"梯度条件（稳定性）  : ‖∇L‖ = {grad_norm:.2e} "
              f"{'✓' if grad_norm <= tol else '✗'}")
        print(f"原始可行性（g ≤ 0）: "
              f"{'✓' if primal_feas_g else '✗'}")
        print(f"原始可行性（h = 0）: "
              f"{'✓' if primal_feas_h else '✗'}")
        print(f"对偶可行性（μ ≥ 0）: "
              f"{'✓' if dual_feas else '✗'}")
        print(f"互补松弛（μᵢgᵢ=0）: "
              f"{'✓' if comp_slack else '✗'}")
        all_ok = (grad_norm <= tol and primal_feas_g and
                  primal_feas_h and dual_feas and comp_slack)
        print("-" * 55)
        print(f"总体判断: {'满足 KKT 条件' if all_ok else '不满足 KKT 条件'}")

    return {
        "stationarity": grad_norm,
        "primal_g": primal_feas_g,
        "primal_h": primal_feas_h,
        "dual": dual_feas,
        "comp_slack": comp_slack,
    }


# ============================================================
# 2. 验证例 8.3 的 KKT 条件
# ============================================================

print("【验证例 8.3：带不等式约束的二维二次规划】")
# min x1^2 + x2^2, s.t. 1 - x1 - x2 <= 0
f_ex  = lambda x: x[0]**2 + x[1]**2
g_ex  = [lambda x: 1.0 - x[0] - x[1]]  # 1 - x1 - x2 <= 0

x_star  = torch.tensor([0.5, 0.5], dtype=torch.float64)
mu_star = [1.0]     # KKT 乘子
lam_star = []

check_kkt_conditions(f_ex, g_ex, [], x_star, mu_star, lam_star)


# ============================================================
# 3. 投影梯度法（PGD）实现
# ============================================================

def project_l2_ball(x, radius=1.0):
    """投影到 L2 球 ‖x‖₂ ≤ radius"""
    norm = x.norm()
    if norm > radius:
        return x * (radius / norm)
    return x.clone()


def project_linf_ball(x, clip_val=1.0):
    """投影到 L∞ 球 ‖x‖∞ ≤ clip_val（即分量逐个裁剪）"""
    return x.clamp(-clip_val, clip_val)


def project_simplex(v):
    """
    投影到概率单纯形 {x | x >= 0, sum(x) = 1}
    使用 Duchi et al. (2008) 排序算法，O(n log n)
    """
    n = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0)
    rho_candidates = torch.arange(1, n + 1, dtype=v.dtype)
    rho_mask = u > (cssv - 1.0) / rho_candidates
    rho = rho_candidates[rho_mask].max().item()
    theta = (cssv[int(rho) - 1] - 1.0) / rho
    return torch.clamp(v - theta, min=0.0)


def projected_gradient_descent(f, x0, project_fn,
                                lr=0.1, n_steps=200, tol=1e-6):
    """
    投影梯度下降

    参数:
        f          : 目标函数（可微）
        x0         : 初始点
        project_fn : 投影到可行域的函数
        lr         : 步长
        n_steps    : 最大迭代次数
        tol        : 收敛容差（梯度范数）
    返回:
        x_hist : 迭代轨迹
        f_hist : 目标函数值历史
    """
    x = x0.clone().float()
    x_hist = [x.clone()]
    f_hist = []

    for step in range(n_steps):
        x.requires_grad_(True)
        loss = f(x)
        f_hist.append(loss.item())

        grad = torch.autograd.grad(loss, x)[0]
        x = x.detach() - lr * grad.detach()
        x = project_fn(x)
        x_hist.append(x.clone())

        if grad.norm().item() < tol:
            print(f"  收敛于第 {step+1} 步，梯度范数 = {grad.norm().item():.2e}")
            break

    return torch.stack(x_hist), f_hist


# ============================================================
# 4. 实验：约束神经网络权重（L2 球约束和 L∞ 球约束）
# ============================================================

print("\n【实验：投影梯度法求解带约束的二次规划】")

# 目标：min f(x) = (x1 - 3)^2 + (x2 - 3)^2
# 约束：‖x‖₂ ≤ 1（L2 球内）
f_quadratic = lambda x: (x[0] - 3.0)**2 + (x[1] - 3.0)**2

x0 = torch.tensor([0.0, 0.0])

print("\n(a) L2 球约束 ‖x‖₂ ≤ 1：")
traj_l2, loss_l2 = projected_gradient_descent(
    f_quadratic, x0,
    project_fn=lambda x: project_l2_ball(x, radius=1.0),
    lr=0.2, n_steps=100
)
x_opt_l2 = traj_l2[-1]
print(f"  最优点: ({x_opt_l2[0]:.4f}, {x_opt_l2[1]:.4f})")
print(f"  理论解: (1/√2, 1/√2) ≈ ({1/2**0.5:.4f}, {1/2**0.5:.4f})")
print(f"  最优值: {f_quadratic(x_opt_l2):.6f}")

print("\n(b) L∞ 球约束 ‖x‖∞ ≤ 1：")
traj_linf, loss_linf = projected_gradient_descent(
    f_quadratic, x0,
    project_fn=lambda x: project_linf_ball(x, clip_val=1.0),
    lr=0.2, n_steps=100
)
x_opt_linf = traj_linf[-1]
print(f"  最优点: ({x_opt_linf[0]:.4f}, {x_opt_linf[1]:.4f})")
print(f"  理论解: (1.0, 1.0)")
print(f"  最优值: {f_quadratic(x_opt_linf):.6f}")

print("\n(c) 概率单纯形约束 x ≥ 0，Σx = 1（3维）：")
f_simplex = lambda x: (x[0] - 0.5)**2 + (x[1] - 0.3)**2 + (x[2] - 0.8)**2
x0_3d = torch.zeros(3)
traj_simp, loss_simp = projected_gradient_descent(
    f_simplex, x0_3d,
    project_fn=project_simplex,
    lr=0.3, n_steps=200
)
x_opt_s = traj_simp[-1]
print(f"  最优点: {x_opt_s.tolist()}")
print(f"  和为1验证: {x_opt_s.sum().item():.6f}")
print(f"  非负验证: {(x_opt_s >= -1e-6).all().item()}")


# ============================================================
# 5. 实际应用：带权重约束的神经网络（WGAN 权重裁剪）
# ============================================================

print("\n【实际应用：WGAN 风格的权重裁剪（L∞ 约束）】")


class ConstrainedLinear(nn.Module):
    """
    带 L∞ 范数约束的线性层
    约束：‖W‖∞ ≤ clip_val（Wasserstein GAN 的 Lipschitz 约束近似）
    """
    def __init__(self, in_features, out_features, clip_val=0.01):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.clip_val = clip_val

    def clamp_weights(self):
        """执行投影：将权重裁剪到 [-clip_val, clip_val]"""
        with torch.no_grad():
            self.linear.weight.clamp_(-self.clip_val, self.clip_val)

    def forward(self, x):
        return self.linear(x)


class WGANCritic(nn.Module):
    """WGAN 判别器（critic）：使用权重裁剪维持 Lipschitz 约束"""
    def __init__(self, input_dim=2, hidden_dim=16, clip_val=0.01):
        super().__init__()
        self.layers = nn.Sequential(
            ConstrainedLinear(input_dim, hidden_dim, clip_val),
            nn.LeakyReLU(0.2),
            ConstrainedLinear(hidden_dim, hidden_dim, clip_val),
            nn.LeakyReLU(0.2),
            ConstrainedLinear(hidden_dim, 1, clip_val)
        )
        self.clip_val = clip_val

    def forward(self, x):
        return self.layers(x)

    def clamp_all_weights(self):
        """训练步后对所有层执行权重裁剪（投影到 L∞ 球）"""
        for module in self.modules():
            if isinstance(module, ConstrainedLinear):
                module.clamp_weights()


# 演示权重裁剪的效果
torch.manual_seed(42)
critic = WGANCritic(input_dim=2, hidden_dim=16, clip_val=0.01)

print("  裁剪前权重统计（第一层）:")
w_before = list(critic.modules())[2].linear.weight.data
print(f"    max|W| = {w_before.abs().max().item():.4f}")
print(f"    ‖W‖∞ = {w_before.abs().max().item():.4f}")

# 执行 "训练步" + 裁剪
fake_loss = critic(torch.randn(16, 2)).mean()
fake_loss.backward()

# KKT 视角：约束 ‖W‖∞ ≤ 0.01，梯度步后立即投影
critic.clamp_all_weights()

print("  裁剪后权重统计（第一层）:")
w_after = list(critic.modules())[2].linear.weight.data
print(f"    max|W| = {w_after.abs().max().item():.6f}")
print(f"    ‖W‖∞ = {w_after.abs().max().item():.6f}  (≤ 0.01 ✓)")


# ============================================================
# 6. 可视化：投影梯度法轨迹与约束边界
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 绘制等高线和轨迹
x1_range = np.linspace(-1.5, 3.5, 300)
x2_range = np.linspace(-1.5, 3.5, 300)
X1, X2 = np.meshgrid(x1_range, x2_range)
F = (X1 - 3)**2 + (X2 - 3)**2

for ax_idx, (ax, traj, title, proj_shape) in enumerate(zip(
    axes,
    [traj_l2.numpy(), traj_linf.numpy()],
    ['L₂ 球约束（‖x‖₂ ≤ 1）', 'L∞ 球约束（‖x‖∞ ≤ 1）'],
    ['circle', 'square']
)):
    cs = ax.contourf(X1, X2, F, levels=20, cmap='Blues', alpha=0.7)
    ax.contour(X1, X2, F, levels=20, colors='k', linewidths=0.3, alpha=0.4)

    # 绘制约束边界
    if proj_shape == 'circle':
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), 'r-', linewidth=2,
                label='约束边界 ‖x‖₂=1')
        ax.fill(np.cos(theta), np.sin(theta), alpha=0.15, color='red')
    else:
        sq = plt.Rectangle((-1, -1), 2, 2, fill=True,
                            facecolor='red', alpha=0.15,
                            edgecolor='red', linewidth=2, label='约束域 ‖x‖∞≤1')
        ax.add_patch(sq)

    # 绘制轨迹
    ax.plot(traj[:, 0], traj[:, 1], 'go-', markersize=3,
            linewidth=1.5, alpha=0.7, label='PGD 轨迹')
    ax.scatter(traj[0, 0], traj[0, 1], c='blue', s=80, zorder=5,
               label='初始点', marker='s')
    ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, zorder=5,
               label='最优点', marker='*')
    ax.scatter([3], [3], c='orange', s=100, zorder=5,
               label='无约束极小', marker='D')

    ax.set_xlim(-1.5, 3.5)
    ax.set_ylim(-1.5, 3.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.colorbar(cs, ax=axes[1], label='目标函数值')
plt.tight_layout()
plt.savefig('pgd_constrained_optimization.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n投影梯度法轨迹图已保存至 pgd_constrained_optimization.png")
```

### 运行说明

```bash
# 安装依赖
pip install torch numpy matplotlib

# 运行代码
python constrained_deep_learning.py
```

### 预期输出

```
【验证例 8.3：带不等式约束的二维二次规划】
=======================================================
KKT 条件验证
=======================================================
候选点 x*        : [0.5, 0.5]
目标值 f(x*)     : 0.500000
  不等式约束 g_i : ['-0.0000']
  KKT 乘子 μ    : ['1.0000']
-------------------------------------------------------
梯度条件（稳定性）  : ‖∇L‖ = 0.00e+00 ✓
原始可行性（g ≤ 0）: ✓
对偶可行性（μ ≥ 0）: ✓
互补松弛（μᵢgᵢ=0）: ✓
总体判断: 满足 KKT 条件

【实验：投影梯度法求解带约束的二次规划】
(a) L2 球约束 ‖x‖₂ ≤ 1：
  最优点: (0.7071, 0.7071)
  理论解: (1/√2, 1/√2) ≈ (0.7071, 0.7071)
```

### 代码核心思想解读

**投影梯度法与 KKT 条件的关系：**

每次 PGD 迭代等价于求解一个小型约束二次规划。以 $L_2$ 球投影为例：

$$\mathcal{P}_{B_R}(\mathbf{y}) = \arg\min_{\mathbf{x}} \|\mathbf{x} - \mathbf{y}\|_2^2 \quad \text{s.t.} \quad \|\mathbf{x}\|_2^2 \leq R^2$$

其 KKT 条件：

$$2(\mathbf{x} - \mathbf{y}) + 2\mu \mathbf{x} = \mathbf{0} \implies \mathbf{x} = \frac{\mathbf{y}}{1 + \mu}, \quad \mu \geq 0, \quad \mu(\|\mathbf{x}\|_2 - R) = 0.$$

当 $\|\mathbf{y}\|_2 > R$ 时约束活跃，解 $\mu = \|\mathbf{y}\|_2 / R - 1 > 0$，$\mathbf{x}^* = R \cdot \mathbf{y} / \|\mathbf{y}\|_2$。这正是代码中的闭合公式。

---

## 练习题

**练习 8.1（KKT 条件应用）**

求解以下不等式约束问题，写出 KKT 条件并求最优解：

$$\min_{x_1, x_2} \quad (x_1 - 1)^2 + (x_2 - 2)^2 \quad \text{s.t.} \quad x_1^2 + x_2^2 \leq 4.$$

(a) 无约束极小是否可行？说明理由。
(b) 写出完整 KKT 条件（四组）。
(c) 分情形讨论（约束活跃 vs 非活跃），求出所有 KKT 点。
(d) 判断最终最优解。

---

**练习 8.2（互补松弛的物理意义）**

考察生产计划问题：某工厂生产两种产品（$x_1, x_2$），目标为最大化利润 $\pi = 3x_1 + 5x_2$，约束条件为：

$$x_1 \leq 4, \quad 2x_2 \leq 12, \quad 3x_1 + 5x_2 \leq 25, \quad x_1, x_2 \geq 0.$$

（改写为最小化 $f = -3x_1 - 5x_2$）

(a) 将问题写成标准不等式约束形式（$g_i \leq 0$）。
(b) 图解法（或代入角点法）求最优解 $(x_1^*, x_2^*)$。
(c) 写出 KKT 乘子，并用互补松弛条件解释哪些约束在最优解处活跃。
(d) 活跃约束的乘子 $\mu_i^* > 0$ 的经济学含义是什么？

---

**练习 8.3（约束规范的必要性）**

考察问题：

$$\min_{x_1, x_2} \quad x_1 \quad \text{s.t.} \quad g_1: (1-x_1)^3 \leq 0,\; g_2: x_2 \leq 0.$$

(a) 求最优解 $\mathbf{x}^*$（提示：直接分析可行域的结构）。
(b) 验证 LICQ 在 $\mathbf{x}^*$ 处是否成立。
(c) 尝试写出 KKT 条件，说明为何无法找到满足对偶可行性的乘子。
(d) 这说明约束规范对 KKT 条件成立的重要性是什么？

---

**练习 8.4（SVM 与 KKT）**

考察线性可分 SVM（硬间隔）：设训练集 $\{(+1, 0, +1), (0, +1, +1), (-1, 0, -1), (0, -1, -1)\}$（格式：$(x_1, x_2, y)$）。

(a) 写出 SVM 原始问题（最小化 $\frac{1}{2}\|\mathbf{w}\|^2$，约束 $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$）。
(b) 写出完整 KKT 条件，特别是梯度条件和互补松弛条件。
(c) 由对称性猜测 $\mathbf{w}^* = (w, w)^\top$ 的形式，利用 KKT 条件求出最优 $\mathbf{w}^*, b^*$。
(d) 哪些点是支撑向量？用互补松弛解释。

---

**练习 8.5（投影梯度法手动推导）**

考察约束问题：$\min_{\mathbf{x}} f(\mathbf{x}) = x_1^2 + 2x_2^2 - 2x_1 - 4x_2$，约束 $\|\mathbf{x}\|_2^2 \leq 2$。

(a) 求无约束极小，判断是否满足约束。
(b) 若约束活跃，写出 KKT 条件，求 KKT 点与乘子。
(c) 验证二阶充分条件，确认极小点类型。
(d) 描述投影梯度法的迭代格式（给出 $L_2$ 球投影公式），说明如何从初始点 $(0,0)$ 收敛到最优解。

---

## 练习答案

### 练习 8.1 答案

**(a) 无约束极小可行性检验：**

无约束极小为 $(1, 2)$。验证约束：$1^2 + 2^2 = 5 > 4$，约束 $g: x_1^2 + x_2^2 - 4 \leq 0$ 被违反。故无约束极小**不可行**，最优解必在约束边界上（约束活跃）。

**(b) KKT 条件（四组）：**

Lagrange 函数：$\mathcal{L} = (x_1-1)^2 + (x_2-2)^2 + \mu(x_1^2 + x_2^2 - 4)$。

- **梯度条件：**
$$2(x_1 - 1) + 2\mu x_1 = 0, \quad 2(x_2 - 2) + 2\mu x_2 = 0.$$
- **原始可行：** $x_1^2 + x_2^2 \leq 4$。
- **对偶可行：** $\mu \geq 0$。
- **互补松弛：** $\mu(x_1^2 + x_2^2 - 4) = 0$。

**(c) 分情形讨论：**

**情形 1（非活跃，$\mu = 0$）：** 梯度条件给出 $(x_1, x_2) = (1, 2)$，但 $1 + 4 = 5 > 4$，不可行，排除。

**情形 2（活跃，$x_1^2 + x_2^2 = 4$）：**

由梯度条件：
$$x_1(1 + \mu) = 1 \implies x_1 = \frac{1}{1+\mu}, \quad x_2(1 + \mu) = 2 \implies x_2 = \frac{2}{1+\mu}.$$

代入活跃约束：

$$\frac{1}{(1+\mu)^2} + \frac{4}{(1+\mu)^2} = 4 \implies (1+\mu)^2 = \frac{5}{4} \implies 1+\mu = \frac{\sqrt{5}}{2}.$$

（取正值，因为 $\mu \geq 0$ 要求 $1+\mu > 0$）

$$\mu = \frac{\sqrt{5}}{2} - 1 \approx 0.118 > 0. \quad \checkmark$$

$$x_1^* = \frac{2}{\sqrt{5}}, \quad x_2^* = \frac{4}{\sqrt{5}}.$$

**(d) 最优解：** $\mathbf{x}^* = \left(\dfrac{2}{\sqrt{5}}, \dfrac{4}{\sqrt{5}}\right)$，$f^* = \left(\dfrac{2}{\sqrt{5}} - 1\right)^2 + \left(\dfrac{4}{\sqrt{5}} - 2\right)^2 = (\sqrt{5}-2)^2 + (2\sqrt{5}/\sqrt{5}-2)^2 = (\sqrt{5}-2)^2 \approx 0.056$。

直觉验证：最优解是无约束极小 $(1,2)$ 在 $L_2$ 球 $\|\mathbf{x}\|_2 \leq 2$ 上的投影，投影公式 $\mathbf{x}^* = 2 \cdot (1,2) / \|(1,2)\|_2 = 2(1,2)/\sqrt{5} = (2/\sqrt{5}, 4/\sqrt{5})$，与 KKT 结果一致。

---

### 练习 8.2 答案

**(a) 标准形式：**

$\min f = -3x_1 - 5x_2$，约束：

$$g_1: x_1 - 4 \leq 0, \quad g_2: 2x_2 - 12 \leq 0, \quad g_3: 3x_1 + 5x_2 - 25 \leq 0, \quad g_4: -x_1 \leq 0, \quad g_5: -x_2 \leq 0.$$

**(b) 最优解（角点法）：**

可行域的极值点候选：$(0,0)$，$(4,0)$，$(0,5)$（由 $3\cdot0+5x_2=25$ 得 $x_2=5$），$(0,6)$（由 $g_2$ 得），$(4,13/5)$（由 $g_1, g_3$ 交点），$(10/3,3)$（由 $g_2, g_3$ 交点，$x_2=6$时 $g_2$ 先达到）。

计算 $g_3$ 在各角点的约束：$(4, 13/5)$：$3(4)+5(13/5) = 12+13 = 25$，满足。$f(4, 13/5) = -12 - 13 = -25$。

$(10/3, 3)$（$g_2$ 活跃时 $x_2=6$：$3x_1+30=25$，$x_1 < 0$，不可行）。重新分析：$g_2: 2x_2 \leq 12 \Rightarrow x_2 \leq 6$；$g_3: 3x_1+5x_2 \leq 25$；令 $x_2 = 6$：$x_1 \leq 5/3$；令 $x_1=4$：$x_2 \leq 13/5 = 2.6$。

目标在 $(4, 13/5)$ 处：$f = -3(4) - 5(13/5) = -12 - 13 = -25$（即利润 25）。目标在 $(0, 5)$ 处：$f = -25$（相同）。需进一步检查边 $g_3$ 上的极值：$\nabla f = (-3,-5) = \lambda(-3,-5) \cdot 5$ 等，得 $g_1, g_3$ 同时活跃时 $(4, 13/5)$ 为最优。

最优解：$x_1^* = 4$，$x_2^* = 13/5$，利润 $\pi^* = 25$。

**(c) KKT 乘子与互补松弛：**

活跃约束：$g_1(x^*) = 0$（$x_1 = 4$），$g_3(x^*) = 0$（$3(4)+5(13/5) = 25$），其余非活跃（$\mu_2 = \mu_4 = \mu_5 = 0$）。

梯度条件：
$$-3 + \mu_1 + 3\mu_3 = 0, \quad -5 + 5\mu_3 = 0 \implies \mu_3 = 1, \quad \mu_1 = 0.$$

**(d) 经济学解释：** $\mu_3 = 1$ 表示资源约束 $3x_1 + 5x_2 \leq 25$ 是"瓶颈"，其对应的影子价格（shadow price）为 1，即总资源增加 1 单位，利润提升 1 单位。$\mu_1 = 0$ 说明 $x_1 \leq 4$ 的约束虽然活跃但非"瓶颈"（边际价值为零）。

---

### 练习 8.3 答案

**(a) 最优解：**

$g_1 = (1-x_1)^3 \leq 0 \Rightarrow x_1 \geq 1$；$g_2 = x_2 \leq 0$。目标最小化 $x_1$，在 $x_1 \geq 1$ 约束下极小值在 $x_1 = 1$ 处，$x_2$ 可取 $0$。故 $\mathbf{x}^* = (1, 0)$。

**(b) LICQ 验证：**

活跃约束：$g_1(1,0) = 0$，$g_2(1,0) = 0$。
$$\nabla g_1(1,0) = \left(-3(1-x_1)^2\big|_{x_1=1}, 0\right) = (0, 0).$$

$\nabla g_1(\mathbf{x}^*) = \mathbf{0}$，活跃约束梯度含零向量，线性相关，**LICQ 不成立**。

**(c) KKT 条件失败：**

梯度条件：$\nabla f + \mu_1 \nabla g_1 + \mu_2 \nabla g_2 = (1,0) + \mu_1(0,0) + \mu_2(0,1) = (1, \mu_2) = (0,0)$。

第一分量：$1 = 0$，矛盾。无论 $\mu_1, \mu_2$ 如何取值，梯度条件均不满足，**KKT 条件无解**。

**(d) 结论：** 即使 $\mathbf{x}^*$ 是真正的局部极小点，若约束规范（如 LICQ）不成立，KKT 条件可能无法成立或乘子不存在。这说明**约束规范是 KKT 条件成立的前提**，在应用 KKT 条件前必须先验证约束规范。

---

### 练习 8.4 答案

**(a) SVM 原始问题：**

设 $\mathbf{w} = (w_1, w_2)^\top$，优化问题为：

$$\min_{\mathbf{w}, b} \frac{1}{2}(w_1^2 + w_2^2) \quad \text{s.t.} \quad g_i = 1 - y_i(w_1 x_{i1} + w_2 x_{i2} + b) \leq 0, \quad i=1,2,3,4.$$

四个约束展开：

$$g_1: 1-(w_1+b)\leq 0, \quad g_2: 1-(w_2+b)\leq 0, \quad g_3: 1+(w_1+b)\leq 0 \text{ (注：}y_3=-1), \quad g_4: 1+(w_2+b)\leq 0.$$

**(b) KKT 条件：**

- **梯度条件（对 $\mathbf{w}$）：** $\mathbf{w} = \sum_{i=1}^4 \mu_i y_i \mathbf{x}_i$。
- **梯度条件（对 $b$）：** $\sum_{i=1}^4 \mu_i y_i = 0$。
- **互补松弛：** $\mu_i [1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)] = 0$。

**(c) 由对称性求解：**

由问题对称性，$w_1 = w_2 = w$，$b = 0$。支撑向量应在 $y_i(\mathbf{w}^\top \mathbf{x}_i) = 1$ 的边界上。

对 $\mathbf{x}_1 = (1,0)$，$y_1 = 1$：$w \cdot 1 = 1 \Rightarrow w = 1$。

验证：$\mathbf{w}^* = (1, 1)^\top$，$b^* = 0$，间隔 $= 2/\|\mathbf{w}\|_2 = 2/\sqrt{2} = \sqrt{2}$。

对所有样本验证：$y_i(\mathbf{w}^\top \mathbf{x}_i) = 1$（对 $(1,0,+1)$：$1 \cdot 1 + 1 \cdot 0 = 1 \geq 1$）。所有四个点恰好在间隔边界上。

**(d) 支撑向量：** 所有四个训练点均满足 $y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$，即所有约束均活跃，由互补松弛 $\mu_i > 0$，所有点均为**支撑向量**（数据点恰好对称排列在单位圆上，特殊情形）。

---

### 练习 8.5 答案

**(a) 无约束极小：**

$$\nabla f = (2x_1 - 2, 4x_2 - 4) = \mathbf{0} \implies \mathbf{x}_{\text{unc}} = (1, 1).$$

约束验证：$1^2 + 1^2 = 2$，恰好满足 $\|\mathbf{x}\|_2^2 = 2$（边界）。故无约束极小恰好在约束边界上，约束**活跃**（但 $\mu = 0$ 仍可能满足 KKT）。

**(b) KKT 条件：**

Lagrange 函数：$\mathcal{L} = x_1^2 + 2x_2^2 - 2x_1 - 4x_2 + \mu(x_1^2 + x_2^2 - 2)$。

梯度条件：
$$2x_1 - 2 + 2\mu x_1 = 0 \implies x_1 = \frac{1}{1+\mu}, \quad 4x_2 - 4 + 2\mu x_2 = 0 \implies x_2 = \frac{2}{2+\mu}.$$

代入 $x_1^2 + x_2^2 = 2$（活跃约束）：

$$\frac{1}{(1+\mu)^2} + \frac{4}{(2+\mu)^2} = 2.$$

令 $\mu = 0$：$1 + 1 = 2$，方程成立！故 $\mu^* = 0$，$\mathbf{x}^* = (1, 1)$，验证对偶可行性 $\mu^* = 0 \geq 0$。✓

**(c) 二阶充分条件验证：**

$$\nabla^2_{\mathbf{x}\mathbf{x}} \mathcal{L} = \begin{pmatrix} 2+2\mu^* & 0 \\ 0 & 4+2\mu^* \end{pmatrix} = \begin{pmatrix} 2 & 0 \\ 0 & 4 \end{pmatrix} \succ \mathbf{0}.$$

临界锥（$\mu^* = 0$，活跃约束 $g: \mathbf{x}^\top \mathbf{x} - 2 \leq 0$，$\nabla g = 2\mathbf{x}^* = (2,2)$）：

$$\mathcal{C} = \{\mathbf{d} \mid (2,2)^\top \mathbf{d} \leq 0\}.$$

对任意 $\mathbf{d} \in \mathcal{C} \setminus \{\mathbf{0}\}$，$\mathbf{d}^\top \nabla^2 \mathcal{L}\, \mathbf{d} = 2d_1^2 + 4d_2^2 > 0$，SOSC 成立，$(1,1)$ 是严格局部极小（也是全局极小，因 $f$ 凸）。

**(d) 投影梯度法迭代格式：**

对 $L_2$ 球约束 $\|\mathbf{x}\|_2^2 \leq 2$（半径 $R = \sqrt{2}$）：

$$\mathbf{x}^{(k+1)} = \mathcal{P}\!\left(\mathbf{x}^{(k)} - \eta \nabla f(\mathbf{x}^{(k)})\right),$$

其中投影公式为：

$$\mathcal{P}_{\sqrt{2}}(\mathbf{y}) = \begin{cases} \mathbf{y} & \text{若 } \|\mathbf{y}\|_2 \leq \sqrt{2} \\ \sqrt{2} \cdot \dfrac{\mathbf{y}}{\|\mathbf{y}\|_2} & \text{否则} \end{cases}$$

从 $\mathbf{x}^{(0)} = (0,0)$ 出发，$\nabla f(0,0) = (-2,-4)$，第一步：$\mathbf{y}^{(1)} = (0,0) - \eta(-2,-4) = (2\eta, 4\eta)$。当 $\|\mathbf{y}^{(1)}\|_2 = \eta\sqrt{20} > \sqrt{2}$（对合理的 $\eta > 0$），投影后沿方向 $(1,2)/\sqrt{5}$ 缩放至半径 $\sqrt{2}$，逐步收敛至 $(1,1)$。
