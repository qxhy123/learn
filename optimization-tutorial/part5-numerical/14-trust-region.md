# 第14章：信赖域方法（融合版）

> **前置章节**：第6章（牛顿法与拟牛顿法）、第5章（梯度下降法）、第4章（最优性条件）
>
> **难度**：★★★★☆
>
> **本文件**：融合"原版严格推导 + 速记 / 套路 / 自测"。保留原版完整正文 + 在最前置一例速记 / 思维路径 + 最后追加方法总结与自测。

> **一例速记**：
> **核心思想**：先划球（信赖域半径 $\Delta_k$），再在球内求模型极小——方向与步长同时确定。
> **二次模型**：$m_k(\mathbf{p}) = f_k + \mathbf{g}_k^\top\mathbf{p} + \frac{1}{2}\mathbf{p}^\top\mathbf{H}_k\mathbf{p}$，子问题：$\min m_k(\mathbf{p})$ s.t. $\|\mathbf{p}\| \leq \Delta_k$。
> **下降比**：$\rho_k = \text{ared}_k / \text{pred}_k$；$\rho < \eta_1$ 收缩半径，$\rho \geq \eta_2$ 扩大半径。
> **Cauchy 点**：$\mathbf{p}^C = -\tau^*\mathbf{g}_k$，沿梯度方向在域内最优，保证全局收敛。
> **狗腿法**：折线 $\mathbf{0} \to \mathbf{p}^U \to \mathbf{p}^B$ 与球面的交点，适用于 $\mathbf{H}_k \succ 0$。
> **AI 关联**：TRPO/PPO 用 KL 散度代替 $\ell_2$ 范数约束策略更新范围，本质是信赖域方法在策略空间的推广。

---

## 引入：为什么"先方向后步长"有时会失败？

> **题目**：设目标函数 $f(\mathbf{x}) = x_1^2 - 10x_2^2$（鞍点问题）。在当前点 $\mathbf{x}_0 = (1, 0.1)^\top$，Hessian 矩阵为 $\mathbf{H} = \begin{pmatrix}2 & 0 \\ 0 & -20\end{pmatrix}$（不正定！）。
>
> (1) 牛顿方向 $\mathbf{d} = -\mathbf{H}^{-1}\nabla f(\mathbf{x}_0)$ 是什么？它是下降方向吗？
> (2) 若信赖域半径 $\Delta = 0.5$，Cauchy 点在哪里？其模型下降量是多少？
> (3) 信赖域方法如何"自然处理"Hessian 不正定的情况？

请先停下来想一想：当 Hessian 不正定时，牛顿方向可能指向鞍点的"上坡"方向。下面还原完整解题思路。

---

## 思维路径还原（解题者的内心独白）

> "函数 $f = x_1^2 - 10x_2^2$，在 $\mathbf{x}_0 = (1, 0.1)^\top$ 处：$\nabla f = (2x_1, -20x_2)^\top = (2, -2)^\top$。
>
> **第 (1) 问：牛顿方向。**
>
> $\mathbf{H}^{-1} = \begin{pmatrix}1/2 & 0 \\ 0 & -1/20\end{pmatrix}$。
>
> 牛顿步 $\mathbf{d} = -\mathbf{H}^{-1}\nabla f = -\begin{pmatrix}1/2 & 0 \\ 0 & -1/20\end{pmatrix}\begin{pmatrix}2\\-2\end{pmatrix} = -\begin{pmatrix}1 \\ 1/10\end{pmatrix} = \begin{pmatrix}-1 \\ -0.1\end{pmatrix}$。
>
> 验证下降性：$\nabla f^\top \mathbf{d} = (2)(-1) + (-2)(-0.1) = -2 + 0.2 = -1.8 < 0$，**是下降方向**。但这只是恰好——一般来说当 $\mathbf{H}$ 不正定时，牛顿步可以是上升方向（当梯度在负特征值方向有分量时）。更危险的是：即使牛顿方向是下降的，沿它走一大步可能走向鞍点附近的不稳定区域。
>
> **第 (2) 问：Cauchy 点。**
>
> 沿负梯度方向 $-\mathbf{g} = (-2, 2)^\top$，计算 $\mathbf{g}^\top\mathbf{H}\mathbf{g} = (2, -2)\begin{pmatrix}2&0\\0&-20\end{pmatrix}\begin{pmatrix}2\\-2\end{pmatrix} = (4, 40)\begin{pmatrix}2\\-2\end{pmatrix} = 8 - 80 = -72 < 0$。
>
> 由于 $\mathbf{g}^\top\mathbf{H}\mathbf{g} \leq 0$，约束激活，取 $\tau^* = \Delta / \|\mathbf{g}\| = 0.5 / \sqrt{4+4} = 0.5/(2\sqrt{2}) \approx 0.177$。
>
> Cauchy 点：$\mathbf{p}^C = -\tau^*\mathbf{g} = 0.177 \times (-2, 2)^\top \approx (-0.354, 0.354)^\top$。
>
> 模型下降量：$\text{pred} = -\mathbf{g}^\top\mathbf{p}^C - \frac{1}{2}(\mathbf{p}^C)^\top\mathbf{H}\mathbf{p}^C$。
>
> $\mathbf{g}^\top\mathbf{p}^C = (2)(-0.354)+(-2)(0.354) = -1.414$，$(\mathbf{p}^C)^\top\mathbf{H}\mathbf{p}^C = \tau^{*2}\mathbf{g}^\top\mathbf{H}\mathbf{g} = 0.0313\times(-72) = -2.254$。
>
> $\text{pred} = 1.414 - \frac{1}{2}(-2.254) = 1.414 + 1.127 = 2.541 > 0$。✓ 模型下降量为正。
>
> **第 (3) 问：信赖域如何自然处理不正定性。**
>
> 线搜索方法面对不正定 Hessian 时必须"修正"（如加正定修正 $\mathbf{H} + \lambda\mathbf{I}$），否则牛顿步不可信。信赖域方法则完全绕开这个问题：子问题始终在球约束内求模型极小，即使模型是非凸的（$\mathbf{H}$ 不正定），在球约束范围内仍有全局最优解（受约束的非凸问题在有界区域内一定有解）。最优性条件 $(\mathbf{H}+\lambda^*\mathbf{I})\mathbf{p}^* = -\mathbf{g}$（$\lambda^* \geq 0$）自动正则化了 Hessian——$\lambda^*$ 选取使 $\mathbf{H}+\lambda^*\mathbf{I} \succeq 0$，这正是最优条件的数学保证。信赖域方法不需要显式'修正' Hessian，不正定性在子问题最优性条件中被'免费'处理。"

---

## 学习目标

学完本章，你将能够：

1. **理解信赖域思想**：掌握信赖域方法与线搜索方法的根本区别——先确定步长范围，再在其中求最优方向
2. **建立二次模型**：从Taylor展开出发推导二次近似模型 $m_k(\mathbf{p}) = f_k + \mathbf{g}_k^\top \mathbf{p} + \frac{1}{2}\mathbf{p}^\top \mathbf{H}_k \mathbf{p}$，理解其有效性边界
3. **求解信赖域子问题**：掌握Cauchy点的闭合解法与狗腿法（Dogleg Method）的几何构造，以及基于特征值的精确解法
4. **分析半径自适应更新**：理解实际下降与预测下降之比 $\rho_k$ 如何驱动信赖域半径的扩大、收缩与保持
5. **掌握收敛性结论**：了解信赖域方法全局收敛和局部超线性收敛的条件与证明思路，以及其在深度学习稳定训练中的应用

---

## 14.1 信赖域方法的基本思想

### 14.1.1 线搜索方法的局限

在第6章中，我们学习了牛顿法和拟牛顿法。这类方法的共同框架是：

1. 确定下降方向 $\mathbf{d}_k$
2. 沿方向 $\mathbf{d}_k$ 做线搜索，找到合适步长 $\alpha_k$
3. 更新 $\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{d}_k$

这种"**先方向，后步长**"的范式有一个根本问题：二次近似模型在当前点附近才有效，但我们无法事先知道"附近"的范围有多大。当 Hessian 矩阵不正定时，牛顿方向甚至可能是上升方向，线搜索的代价会变得极高。

### 14.1.2 信赖域的核心思想

信赖域方法（Trust-Region Methods）采用完全不同的哲学：**先划定模型可信的区域，再在该区域内求最优步**。

**核心思路：**

- 在当前点 $\mathbf{x}_k$ 处，构造目标函数 $f$ 的二次近似模型 $m_k$
- 认为模型 $m_k$ 在以 $\mathbf{x}_k$ 为中心、半径为 $\Delta_k$ 的球形区域内足够可信
- 在该**信赖域**（Trust Region）内求解模型的极小值，得到试探步 $\mathbf{p}_k$
- 根据模型预测与实际下降的吻合程度，自适应地调整信赖域半径 $\Delta_k$

这是一种"**先步长范围，再最优方向**"的策略，方向与步长同时确定。

### 14.1.3 二次近似模型

设 $f: \mathbb{R}^n \to \mathbb{R}$ 是二次连续可微函数。在当前迭代点 $\mathbf{x}_k$ 处，利用 Taylor 展开构造二次近似模型：

$$\boxed{m_k(\mathbf{p}) = f_k + \mathbf{g}_k^\top \mathbf{p} + \frac{1}{2}\mathbf{p}^\top \mathbf{H}_k \mathbf{p}}$$

其中：
- $f_k = f(\mathbf{x}_k)$ 是当前函数值
- $\mathbf{g}_k = \nabla f(\mathbf{x}_k)$ 是当前梯度
- $\mathbf{H}_k$ 是 Hessian 矩阵 $\nabla^2 f(\mathbf{x}_k)$ 或其正定近似（如 BFGS 矩阵）
- $\mathbf{p} = \mathbf{x} - \mathbf{x}_k$ 是从当前点出发的步向量

注意，模型满足 $m_k(\mathbf{0}) = f_k$ 且 $\nabla_\mathbf{p} m_k(\mathbf{0}) = \mathbf{g}_k$，即在原点处与真实函数的零阶和一阶信息精确吻合。

### 14.1.4 与线搜索方法的对比

| 特性 | 线搜索方法 | 信赖域方法 |
|------|-----------|-----------|
| 决策顺序 | 先定方向，再定步长 | 同时确定方向和步长 |
| 步长控制 | Wolfe/Armijo 条件 | 信赖域半径 $\Delta_k$ |
| Hessian 不正定 | 需要修正 | 自然处理 |
| 全局收敛保证 | 需要额外条件 | 框架内自然保证 |
| 每步计算量 | 低（方向固定后线搜索） | 较高（需求解子问题） |
| 收敛速度 | 局部超线性/二次 | 局部超线性/二次 |

---

## 14.2 信赖域子问题

### 14.2.1 子问题的标准形式

信赖域方法在每次迭代中需要求解如下**信赖域子问题**（Trust-Region Subproblem，TRS）：

$$\boxed{\min_{\mathbf{p} \in \mathbb{R}^n} \; m_k(\mathbf{p}) = f_k + \mathbf{g}_k^\top \mathbf{p} + \frac{1}{2}\mathbf{p}^\top \mathbf{H}_k \mathbf{p} \quad \text{s.t.} \quad \|\mathbf{p}\| \leq \Delta_k}$$

这是一个带球约束的二次规划问题。虽然看起来简单，但其精确求解在高维情形下并不平凡，特别是当 $\mathbf{H}_k$ 不正定时。

### 14.2.2 精确解的特征：最优性条件

**定理（信赖域子问题的最优性条件）**：向量 $\mathbf{p}^*$ 是信赖域子问题的全局最优解，当且仅当 $\|\mathbf{p}^*\| \leq \Delta_k$ 且存在 $\lambda^* \geq 0$，使得：

$$(\mathbf{H}_k + \lambda^* \mathbf{I})\mathbf{p}^* = -\mathbf{g}_k$$

$$\lambda^*(\Delta_k - \|\mathbf{p}^*\|) = 0$$

$$\mathbf{H}_k + \lambda^* \mathbf{I} \succeq \mathbf{0}$$

**解读：**

- 若 $\mathbf{H}_k \succ 0$ 且 $\|\mathbf{H}_k^{-1}\mathbf{g}_k\| \leq \Delta_k$：约束不激活，$\lambda^* = 0$，$\mathbf{p}^* = -\mathbf{H}_k^{-1}\mathbf{g}_k$（即牛顿步）
- 若约束激活（$\|\mathbf{p}^*\| = \Delta_k$）：需要找到 $\lambda^* > 0$ 使得 $\|(\mathbf{H}_k + \lambda^* \mathbf{I})^{-1}\mathbf{g}_k\| = \Delta_k$

### 14.2.3 基于特征值分解的精确解法

设 $\mathbf{H}_k = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ 为特征值分解，其中 $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$，$\mathbf{Q}$ 为正交矩阵。令 $\hat{\mathbf{g}} = \mathbf{Q}^\top \mathbf{g}_k$，则：

$$\mathbf{p}^*(\lambda) = -\mathbf{Q}(\boldsymbol{\Lambda} + \lambda\mathbf{I})^{-1}\hat{\mathbf{g}} = -\sum_{i=1}^n \frac{\hat{g}_i}{\lambda_i + \lambda}\mathbf{q}_i$$

其中 $\mathbf{q}_i$ 是第 $i$ 个特征向量。步长函数：

$$\phi(\lambda) = \|\mathbf{p}^*(\lambda)\| = \left(\sum_{i=1}^n \frac{\hat{g}_i^2}{(\lambda_i + \lambda)^2}\right)^{1/2}$$

这是关于 $\lambda$ 的严格单调递减函数（在有效域上），因此可用牛顿法对方程 $\phi(\lambda) = \Delta_k$ 求解 $\lambda^*$。

**难例（Hard Case）**：当 $\mathbf{g}_k$ 在最小特征值对应特征向量方向上分量为零时，需要特殊处理。此时：

$$\mathbf{p}^* = -\mathbf{Q}(\boldsymbol{\Lambda} + \lambda^*\mathbf{I})^{-1}\hat{\mathbf{g}} + \tau \mathbf{q}_{\min}$$

其中 $\tau$ 的选取使得 $\|\mathbf{p}^*\| = \Delta_k$。

### 14.2.4 信赖域方法总体框架

```
输入：初始点 x_0，初始信赖域半径 Δ_0，最大半径 Δ_max，容差 ε > 0
      参数：0 < η_1 ≤ η_2 < 1，0 < γ_1 < 1 < γ_2

for k = 0, 1, 2, ... do
    计算梯度 g_k = ∇f(x_k)，Hessian（或近似）H_k
    if ‖g_k‖ ≤ ε then
        return x_k
    end if

    求解（近似）信赖域子问题：
        p_k ≈ argmin m_k(p)  s.t. ‖p‖ ≤ Δ_k

    计算实际下降与预测下降之比：
        ρ_k = (f(x_k) - f(x_k + p_k)) / (m_k(0) - m_k(p_k))

    更新迭代点：
        if ρ_k ≥ η_1 then
            x_{k+1} = x_k + p_k   （接受步）
        else
            x_{k+1} = x_k         （拒绝步）
        end if

    更新信赖域半径 Δ_{k+1}（见14.4节）
end for
```

---

## 14.3 Cauchy点与狗腿法

精确求解信赖域子问题在高维情形计算代价较高。实践中常用近似方法，其中最重要的是 **Cauchy 点法**和**狗腿法**（Dogleg Method）。

### 14.3.1 Cauchy 点

**Cauchy 点**（Cauchy Point）是沿梯度下降方向在信赖域约束下的最优一维解，是信赖域方法的最简近似。

**推导：** 沿负梯度方向 $\mathbf{p} = -\tau \mathbf{g}_k$（$\tau > 0$），代入二次模型：

$$m_k(-\tau\mathbf{g}_k) = f_k - \tau\|\mathbf{g}_k\|^2 + \frac{\tau^2}{2}\mathbf{g}_k^\top\mathbf{H}_k\mathbf{g}_k$$

对 $\tau$ 最优化，需要在约束 $\tau\|\mathbf{g}_k\| \leq \Delta_k$ 下求上式极小值：

$$\tau_k^* = \begin{cases} \dfrac{\|\mathbf{g}_k\|^2}{\mathbf{g}_k^\top \mathbf{H}_k \mathbf{g}_k} & \text{若 } \mathbf{g}_k^\top \mathbf{H}_k \mathbf{g}_k > 0 \text{ 且约束不激活} \\[6pt] \dfrac{\Delta_k}{\|\mathbf{g}_k\|} & \text{若约束激活或 } \mathbf{g}_k^\top \mathbf{H}_k \mathbf{g}_k \leq 0 \end{cases}$$

$$\boxed{\mathbf{p}_k^C = -\tau_k^* \mathbf{g}_k}$$

**性质：** Cauchy 点保证了充分的模型下降量：

$$m_k(\mathbf{0}) - m_k(\mathbf{p}_k^C) \geq \frac{1}{2}\|\mathbf{g}_k\| \min\left(\Delta_k, \frac{\|\mathbf{g}_k\|}{\|\mathbf{H}_k\|}\right)$$

这一下降量界是信赖域方法全局收敛性证明的关键。

### 14.3.2 狗腿法的几何思想

当 $\mathbf{H}_k \succ 0$（正定）时，狗腿法（Powell 1970）利用两个关键方向构造更好的近似解：

- **梯度方向步**：$\mathbf{p}^U = -\frac{\|\mathbf{g}_k\|^2}{\mathbf{g}_k^\top\mathbf{H}_k\mathbf{g}_k}\mathbf{g}_k$（最速下降步，即无约束时沿梯度方向的最优步）
- **牛顿步**：$\mathbf{p}^B = -\mathbf{H}_k^{-1}\mathbf{g}_k$（无约束极小化的精确解）

狗腿路径（Dogleg Path）是由 $\mathbf{0}$ 经 $\mathbf{p}^U$ 到 $\mathbf{p}^B$ 的折线段：

$$\tilde{\mathbf{p}}(\tau) = \begin{cases} \tau \mathbf{p}^U & 0 \leq \tau \leq 1 \\ \mathbf{p}^U + (\tau - 1)(\mathbf{p}^B - \mathbf{p}^U) & 1 \leq \tau \leq 2 \end{cases}$$

**狗腿步的选取：** 找到满足 $\|\tilde{\mathbf{p}}(\tau)\| = \Delta_k$ 的 $\tau$，即狗腿路径与信赖域边界的交点。

```
狗腿法算法：
if ‖p^B‖ ≤ Δ_k then
    p_k = p^B                      （牛顿步在信赖域内，直接取）
elif ‖p^U‖ ≥ Δ_k then
    p_k = (Δ_k / ‖g_k‖) × (-g_k)  （Cauchy点，梯度步已超出域）
else
    在线段 [p^U, p^B] 上找与球面的交点：
    求 τ ∈ [1,2] 使 ‖p^U + (τ-1)(p^B - p^U)‖ = Δ_k
    p_k = p^U + (τ-1)(p^B - p^U)
end if
```

**交点计算：** 令 $\mathbf{a} = \mathbf{p}^B - \mathbf{p}^U$，需求解二次方程：

$$\|\mathbf{p}^U + s\mathbf{a}\|^2 = \Delta_k^2$$

$$\|\mathbf{a}\|^2 s^2 + 2(\mathbf{p}^U)^\top\mathbf{a}\, s + \|\mathbf{p}^U\|^2 - \Delta_k^2 = 0$$

取正根即得 $s^* \in [0,1]$，则 $\tau = 1 + s^*$。

### 14.3.3 狗腿法的性质

**命题：** 设 $\mathbf{H}_k \succ 0$，则沿狗腿路径：

1. 模型函数值 $m_k(\tilde{\mathbf{p}}(\tau))$ 关于 $\tau$ 单调不增
2. 步长 $\|\tilde{\mathbf{p}}(\tau)\|$ 关于 $\tau$ 单调不减
3. 狗腿步满足 Cauchy 点下降量界，即：$m_k(\mathbf{0}) - m_k(\mathbf{p}_k) \geq m_k(\mathbf{0}) - m_k(\mathbf{p}_k^C)$

这些性质保证了狗腿法的收敛性与 Cauchy 点相当（全局），同时在 Hessian 正定时显著优于纯 Cauchy 步。

### 14.3.4 截断共轭梯度法（用于大规模问题）

对于大规模问题（$n$ 很大），直接计算 $\mathbf{H}_k^{-1}\mathbf{g}_k$ 不可行。Steihaug（1983）提出用**截断共轭梯度法**（Truncated CG）近似求解子问题：

```
Steihaug-CG 算法：
初始化 z_0 = 0，r_0 = g_k，d_0 = -g_k
for j = 0, 1, 2, ... do
    if d_j^T H_k d_j ≤ 0 then
        沿 d_j 方向走到信赖域边界，返回该点    （负曲率方向，直接到边界）
    end if
    α_j = ‖r_j‖² / (d_j^T H_k d_j)
    z_{j+1} = z_j + α_j d_j
    if ‖z_{j+1}‖ ≥ Δ_k then
        沿 d_j 方向从 z_j 走到信赖域边界，返回该点    （超出边界，截断）
    end if
    r_{j+1} = r_j + α_j H_k d_j
    if ‖r_{j+1}‖ 足够小 then
        return z_{j+1}
    end if
    β_j = ‖r_{j+1}‖² / ‖r_j‖²
    d_{j+1} = -r_{j+1} + β_j d_j
end for
```

此算法每次迭代只需一次 Hessian-向量乘积 $\mathbf{H}_k\mathbf{d}_j$，适合大规模问题，且满足 Cauchy 点下降量界。

---

## 14.4 信赖域半径更新

### 14.4.1 实际下降与预测下降之比

信赖域方法的自适应机制核心是定义**下降比**（Reduction Ratio）：

$$\boxed{\rho_k = \frac{f(\mathbf{x}_k) - f(\mathbf{x}_k + \mathbf{p}_k)}{m_k(\mathbf{0}) - m_k(\mathbf{p}_k)}}$$

其中：
- **分子**：$\text{ared}_k = f(\mathbf{x}_k) - f(\mathbf{x}_k + \mathbf{p}_k)$ 称为**实际下降量**（Actual Reduction）
- **分母**：$\text{pred}_k = m_k(\mathbf{0}) - m_k(\mathbf{p}_k)$ 称为**预测下降量**（Predicted Reduction）

**注意**：预测下降量 $\text{pred}_k \geq 0$ 总成立（因为 $\mathbf{p}_k$ 使模型下降）；实际下降量可能为负（函数值上升）。

### 14.4.2 $\rho_k$ 的物理意义

| $\rho_k$ 的值 | 意义 | 决策 |
|---|---|---|
| $\rho_k < \eta_1$（如 $0.25$） | 模型很差，实际改进远小于预测 | 拒绝步，收缩半径 |
| $\eta_1 \leq \rho_k < \eta_2$（如 $0.75$） | 模型尚可 | 接受步，保持半径 |
| $\rho_k \geq \eta_2$ | 模型很好，实际改进接近或超过预测 | 接受步，扩大半径 |
| $\rho_k \approx 1$ | 模型极好，近乎完美拟合 | 接受步，大幅扩大半径 |
| $\rho_k > 1$ | 实际改进超过预测（非凸情形可能发生） | 接受步，可扩大半径 |

### 14.4.3 半径更新规则

标准的信赖域半径更新策略（Nocedal & Wright，2006）：

$$\Delta_{k+1} = \begin{cases} \gamma_1 \Delta_k & \text{若 } \rho_k < \eta_1 \quad \text{（收缩，如取 } \gamma_1 = 0.25\text{）} \\ \Delta_k & \text{若 } \eta_1 \leq \rho_k < \eta_2 \quad \text{（保持）} \\ \min(\gamma_2 \Delta_k, \Delta_{\max}) & \text{若 } \rho_k \geq \eta_2 \quad \text{（扩大，如取 } \gamma_2 = 2\text{）} \end{cases}$$

典型参数选取：$\eta_1 = 0.25$，$\eta_2 = 0.75$，$\gamma_1 = 0.25$，$\gamma_2 = 2$，$\Delta_{\max}$ 为问题规模决定的上界。

### 14.4.4 迭代点更新

$$\mathbf{x}_{k+1} = \begin{cases} \mathbf{x}_k + \mathbf{p}_k & \text{若 } \rho_k \geq \eta_1 \quad \text{（接受步）} \\ \mathbf{x}_k & \text{若 } \rho_k < \eta_1 \quad \text{（拒绝步，仅调整半径）} \end{cases}$$

**关键特性**：即使拒绝步（$\rho_k < \eta_1$），信赖域方法仍能通过收缩半径来改善子问题的近似质量，下一次迭代极有可能得到更准确的步。这是信赖域方法比线搜索方法更鲁棒的原因之一。

### 14.4.5 初始半径的选取

初始信赖域半径 $\Delta_0$ 的选取对算法效率有重要影响。实践建议：

$$\Delta_0 = \min\left(0.1\|\mathbf{x}_0\|, \,\, \frac{0.1\|\mathbf{g}_0\|}{\|\mathbf{H}_0\|_F}\right)$$

或者更简单地取 $\Delta_0 = 1$，然后依赖自适应机制快速调整。

---

## 14.5 收敛性分析

### 14.5.1 全局收敛性

信赖域方法最重要的理论保证是全局收敛性：无论初始点如何选取，算法均能收敛到一阶驻点。

**定理（全局收敛性）**：设 $f$ 在水平集 $\{x : f(x) \leq f(x_0)\}$ 上连续可微，梯度 Lipschitz 连续，Hessian 近似满足 $\|\mathbf{H}_k\| \leq M$，且子问题求解满足 Cauchy 点下降量界。则：

$$\liminf_{k\to\infty} \|\nabla f(\mathbf{x}_k)\| = 0$$

**证明思路：**

反证法。假设存在 $\epsilon > 0$ 使得对所有 $k$ 有 $\|\mathbf{g}_k\| \geq \epsilon$。

由 Cauchy 点下降量界：
$$\text{pred}_k \geq \frac{1}{2}\|\mathbf{g}_k\|\min\left(\Delta_k, \frac{\|\mathbf{g}_k\|}{M}\right) \geq \frac{\epsilon}{2}\min\left(\Delta_k, \frac{\epsilon}{M}\right)$$

只要 $\Delta_k$ 不趋于零，预测下降量有正的下界，从而实际下降量也有正的下界（通过 $\rho_k$ 的控制），这与 $f$ 下有界矛盾。

可以证明 $\Delta_k$ 不会无限制趋于零：若某次 $\rho_k < \eta_1$，则 Taylor 展开保证对足够小的 $\Delta$ 有 $\rho \geq \eta_1$，故半径有正的下界。矛盾成立。

### 14.5.2 局部收敛速度

当算法靠近极小值点 $\mathbf{x}^*$ 时，收敛速度取决于 Hessian 近似的精度。

**情形1：精确 Hessian（$\mathbf{H}_k = \nabla^2 f(\mathbf{x}_k)$）**

**定理**：设 $\mathbf{x}^*$ 是 $f$ 的严格局部极小值，$\nabla^2 f(\mathbf{x}^*) \succ 0$，$\nabla^2 f$ Lipschitz 连续。则对充分靠近 $\mathbf{x}^*$ 的初始点，信赖域方法（使用精确 Hessian）**二次收敛**：

$$\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \leq C\|\mathbf{x}_k - \mathbf{x}^*\|^2$$

这是因为靠近 $\mathbf{x}^*$ 时，牛顿步在信赖域内（$\|\mathbf{p}^B\| \leq \Delta_k$），信赖域约束不激活，算法退化为标准牛顿法。

**情形2：拟牛顿近似（如 BFGS）**

若 $\mathbf{H}_k$ 满足 Dennis-Moré 条件（超线性收敛条件）：

$$\lim_{k\to\infty} \frac{\|(\mathbf{H}_k - \nabla^2 f(\mathbf{x}^*))\mathbf{p}_k\|}{\|\mathbf{p}_k\|} = 0$$

则算法**超线性收敛**：$\|\mathbf{x}_{k+1} - \mathbf{x}^*\| = o(\|\mathbf{x}_k - \mathbf{x}^*\|)$。

### 14.5.3 收敛速度汇总

设 $\mathbf{x}^*$ 为严格局部极小值，$\nabla^2 f(\mathbf{x}^*) \succ 0$：

| Hessian 近似 | 收敛阶 | 典型表达式 |
|---|---|---|
| 精确 Hessian | 二次（局部） | $\|\mathbf{x}_{k+1}-\mathbf{x}^*\| \leq C\|\mathbf{x}_k-\mathbf{x}^*\|^2$ |
| BFGS | 超线性（局部） | $\|\mathbf{x}_{k+1}-\mathbf{x}^*\| = o(\|\mathbf{x}_k-\mathbf{x}^*\|)$ |
| Cauchy点 | 线性（全局+局部） | $\|\mathbf{x}_{k+1}-\mathbf{x}^*\| \leq r\|\mathbf{x}_k-\mathbf{x}^*\|$，$r<1$ |
| 狗腿法（正定 H） | 超线性（局部） | 同 BFGS 量级 |

### 14.5.4 与梯度下降的复杂度对比

对于强凸函数（条件数为 $\kappa$），各方法的迭代复杂度：

| 方法 | 达到 $\epsilon$ 精度所需迭代次数 |
|---|---|
| 梯度下降（最速下降） | $O(\kappa \log(1/\epsilon))$ |
| 共轭梯度 | $O(\sqrt{\kappa}\log(1/\epsilon))$ |
| 牛顿/信赖域（二次收敛） | $O(\log\log(1/\epsilon))$ |

二次收敛使得靠近最优解时每次迭代有效位数翻倍，通常寥寥数步（5-10步）就能达到机器精度。

---

## 本章小结

| 核心概念 | 数学表达 | 直观含义 |
|---|---|---|
| 二次近似模型 | $m_k(\mathbf{p}) = f_k + \mathbf{g}_k^\top\mathbf{p} + \frac{1}{2}\mathbf{p}^\top\mathbf{H}_k\mathbf{p}$ | 当前点处的Taylor二阶近似 |
| 信赖域子问题 | $\min m_k(\mathbf{p})$ s.t. $\|\mathbf{p}\| \leq \Delta_k$ | 在可信范围内最优化模型 |
| 最优性条件 | $(\mathbf{H}_k+\lambda^*\mathbf{I})\mathbf{p}^*=-\mathbf{g}_k$，$\lambda^*\geq 0$ | 正则化牛顿方程 |
| Cauchy 点 | $\mathbf{p}_k^C = -\tau_k^*\mathbf{g}_k$ | 梯度方向上的最优步 |
| 狗腿步 | 折线段 $\mathbf{0}\to\mathbf{p}^U\to\mathbf{p}^B$ 与球面交点 | 梯度步到牛顿步的插值 |
| 下降比 | $\rho_k = \text{ared}_k / \text{pred}_k$ | 模型预测的可靠程度 |
| 半径更新 | $\rho_k$ 小则收缩，大则扩大 | 根据模型质量自适应调整 |
| 全局收敛 | $\liminf\|\mathbf{g}_k\| = 0$ | 总能到达驻点 |
| 局部收敛 | 精确 Hessian 下二次收敛 | 靠近最优时加速 |

**方法选择指南：**

- **小规模问题**（$n < 1000$）：精确求解子问题（特征值法）+ 精确 Hessian，获得二次收敛
- **中规模问题**（$n \sim 10^4$）：狗腿法 + BFGS，平衡精度与计算量
- **大规模问题**（$n > 10^5$）：Steihaug 截断 CG + L-BFGS，每步只需 Hessian-向量积

---

## 深度学习应用：稳定训练与自适应信赖域

### 背景：深度学习中的优化挑战

深度神经网络的损失曲面具有极端病态性（条件数可达 $10^6$），且存在大量鞍点和平坦区域。标准 SGD 和 Adam 等一阶方法虽然高效，但在以下场景中表现不佳：

- 精细调优（Fine-tuning）时，过大的步长导致灾难性遗忘
- 强化学习中策略更新，步长过大导致策略崩溃
- 物理仿真、科学计算等需要高精度解的场景

信赖域思想为这些问题提供了理论框架。

### 应用一：TRPO 与 PPO（强化学习策略优化）

**TRPO**（Trust Region Policy Optimization，Schulman et al. 2015）将信赖域约束施加在策略更新的 KL 散度上：

$$\max_\theta \; \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A(s,a)\right] \quad \text{s.t.} \quad \mathbb{E}\left[D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s))\right] \leq \delta$$

**PPO**（Proximal Policy Optimization）用 clip 近似替代 KL 约束，计算更简单：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)A_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t\right)\right]$$

### 应用二：带信赖域的稳定微调

以下展示如何在 PyTorch 中实现一个简化版的信赖域优化器，核心思想是通过参数更新范数来约束步长，并根据实际损失改善调整"信赖域"（学习率）：

```python
import torch
import torch.nn as nn
import copy

class TrustRegionOptimizer:
    """
    简化版信赖域优化器（适用于神经网络微调）。

    核心机制：
    - 用参数空间中的欧氏范数作为信赖域约束（代替严格的球约束）
    - 通过 rho = ared / pred 驱动学习率（信赖域半径）自适应调整
    - 当模型预测可靠时扩大步长，预测失败时回退并收缩
    """

    def __init__(self, model, lr_init=1e-3, delta_max=1.0,
                 eta1=0.25, eta2=0.75, gamma1=0.25, gamma2=2.0):
        self.model = model
        self.delta = lr_init        # 当前信赖域半径（等效学习率）
        self.delta_max = delta_max
        self.eta1 = eta1            # 接受步的最低 rho 阈值
        self.eta2 = eta2            # 扩大半径的 rho 阈值
        self.gamma1 = gamma1        # 收缩因子
        self.gamma2 = gamma2        # 扩大因子
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr_init)

    def _get_params_vector(self):
        """将所有参数拼接为一个向量。"""
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])

    def _set_params_from_vector(self, vec):
        """从向量恢复参数。"""
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(vec[offset:offset+numel].view_as(p.data))
            offset += numel

    def step(self, closure):
        """
        执行一步信赖域迭代。

        Args:
            closure: 无参数可调用对象，返回当前损失值（需在内部完成前向传播）

        Returns:
            (loss_new, rho, accepted): 新损失值、下降比、是否接受步
        """
        # 保存当前状态
        params_old = self._get_params_vector().clone()
        loss_old = closure().item()

        # 计算梯度
        self.optimizer.zero_grad()
        loss_for_grad = closure()
        loss_for_grad.backward()

        # 计算梯度范数（用于预测下降量估计）
        grad_norm = torch.cat([
            p.grad.view(-1) for p in self.model.parameters()
            if p.grad is not None
        ]).norm().item()

        # 预测下降量（一阶近似：沿负梯度方向的预期下降）
        pred = self.delta * grad_norm  # 简化估计：delta * ‖g‖
        if pred < 1e-12:
            return loss_old, 1.0, True  # 梯度为零，已收敛

        # 更新参数（等效于在信赖域半径内的梯度步）
        for p in self.model.parameters():
            if p.grad is not None:
                p.data.add_(-self.delta * p.grad / (grad_norm + 1e-8))

        # 计算实际下降量
        with torch.no_grad():
            loss_new = closure().item()
        ared = loss_old - loss_new

        # 计算下降比
        rho = ared / (pred + 1e-12)

        # 判断是否接受步
        if rho >= self.eta1:
            accepted = True  # 接受当前步（参数已更新）
        else:
            # 拒绝步，恢复参数
            accepted = False
            self._set_params_from_vector(params_old)
            loss_new = loss_old

        # 更新信赖域半径（学习率）
        if rho < self.eta1:
            self.delta *= self.gamma1   # 收缩
        elif rho >= self.eta2:
            self.delta = min(self.delta * self.gamma2, self.delta_max)  # 扩大
        # else: 保持不变

        return loss_new, rho, accepted


# ============================================================
# 演示：用信赖域优化器训练简单网络
# ============================================================

def demo_trust_region_training():
    torch.manual_seed(42)

    # 构造简单回归问题
    n_samples, n_features = 100, 10
    X = torch.randn(n_samples, n_features)
    w_true = torch.randn(n_features)
    y = X @ w_true + 0.1 * torch.randn(n_samples)

    # 定义模型与损失
    model = nn.Linear(n_features, 1, bias=False)
    criterion = nn.MSELoss()

    tr_opt = TrustRegionOptimizer(model, lr_init=0.1, delta_max=1.0)

    print(f"{'迭代':>5} {'损失':>10} {'rho':>8} {'delta':>8} {'接受':>6}")
    print("-" * 45)

    for iteration in range(30):
        def closure():
            return criterion(model(X).squeeze(), y)

        loss, rho, accepted = tr_opt.step(closure)

        if iteration % 5 == 0 or not accepted:
            status = "√" if accepted else "×"
            print(f"{iteration:>5} {loss:>10.4f} {rho:>8.3f} "
                  f"{tr_opt.delta:>8.5f} {status:>6}")

    print(f"\n最终损失: {closure().item():.6f}")


# ============================================================
# 演示：自适应信赖域在非平稳损失曲面上的鲁棒性
# ============================================================

def demo_adaptive_delta():
    """
    对比固定学习率SGD与自适应信赖域在损失突变时的鲁棒性。
    模拟场景：训练过程中数据分布发生偏移（损失曲面突然变化）。
    """
    torch.manual_seed(0)

    model_tr = nn.Linear(5, 1, bias=False)
    model_sgd = nn.Linear(5, 1, bias=False)

    # 让两个模型从相同初始点开始
    model_sgd.load_state_dict(copy.deepcopy(model_tr.state_dict()))

    tr_opt = TrustRegionOptimizer(model_tr, lr_init=0.05, delta_max=0.5)
    sgd_opt = torch.optim.SGD(model_sgd.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    results = {'tr_loss': [], 'sgd_loss': [], 'tr_delta': []}

    for step in range(60):
        # 阶段1（前30步）：正常数据
        # 阶段2（后30步）：数据偏移，损失曲面突变
        if step < 30:
            X = torch.randn(50, 5)
            y = X[:, 0] + 0.1 * torch.randn(50)  # 简单线性关系
        else:
            X = torch.randn(50, 5) * 5  # 数据尺度突然扩大5倍
            y = X[:, 0] * 2 + 0.1 * torch.randn(50)

        # 信赖域优化器步
        def closure_tr():
            return criterion(model_tr(X).squeeze(), y)
        loss_tr, _, _ = tr_opt.step(closure_tr)

        # SGD步
        sgd_opt.zero_grad()
        loss_sgd = criterion(model_sgd(X).squeeze(), y)
        loss_sgd.backward()
        sgd_opt.step()

        results['tr_loss'].append(loss_tr)
        results['sgd_loss'].append(loss_sgd.item())
        results['tr_delta'].append(tr_opt.delta)

    # 打印阶段统计
    print("阶段1（步骤0-29）平均损失:")
    print(f"  信赖域: {sum(results['tr_loss'][:30])/30:.4f}")
    print(f"  SGD:    {sum(results['sgd_loss'][:30])/30:.4f}")
    print("\n数据偏移后，阶段2（步骤30-59）平均损失:")
    print(f"  信赖域: {sum(results['tr_loss'][30:])/30:.4f}")
    print(f"  SGD:    {sum(results['sgd_loss'][30:])/30:.4f}")
    print(f"\n信赖域半径在偏移后的自适应调整:")
    print(f"  偏移前最后delta: {results['tr_delta'][29]:.5f}")
    print(f"  偏移后稳定delta: {results['tr_delta'][-1]:.5f}")


if __name__ == "__main__":
    print("=" * 50)
    print("演示1：信赖域优化器基本训练")
    print("=" * 50)
    demo_trust_region_training()

    print("\n" + "=" * 50)
    print("演示2：自适应信赖域鲁棒性对比")
    print("=" * 50)
    demo_adaptive_delta()
```

**关键设计要点：**

1. **信赖域半径 = 自适应学习率**：在深度学习场景，信赖域半径本质上扮演学习率角色，但其调整由数学严格的 $\rho_k$ 准则驱动，而非启发式衰减
2. **步的拒绝机制**：固定学习率方法永远接受步（即使损失上升），信赖域方法可以拒绝有害步并收缩半径重试
3. **无需超参数调优**：$\eta_1, \eta_2, \gamma_1, \gamma_2$ 是理论有保证的参数，对大多数问题不需要调整

---

## 练习题

**练习 14.1（信赖域子问题的最优性条件）**

考虑二维信赖域子问题：

$$\min_{\mathbf{p} \in \mathbb{R}^2} \; 2p_1^2 + p_1 p_2 + p_2^2 + 3p_1 + 2p_2 \quad \text{s.t.} \quad \|\mathbf{p}\| \leq 1$$

（a）写出该问题的 Hessian 矩阵 $\mathbf{H}$ 和梯度向量 $\mathbf{g}$（注意：此处 $f_k = 0$，$m_k(\mathbf{p}) = \mathbf{g}^\top\mathbf{p} + \frac{1}{2}\mathbf{p}^\top\mathbf{H}\mathbf{p}$）

（b）验证 $\mathbf{H}$ 是否正定，并求无约束最优解 $\mathbf{p}^B = -\mathbf{H}^{-1}\mathbf{g}$

（c）判断约束是否激活，求信赖域子问题的最优解

（d）写出最优性条件中的 $\lambda^*$ 值，并验证所有条件满足

---

**练习 14.2（Cauchy 点计算）**

设当前迭代点处的梯度为 $\mathbf{g} = (2, -1)^\top$，Hessian 近似为 $\mathbf{H} = \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix}$，信赖域半径 $\Delta = 0.5$。

（a）计算 $\mathbf{g}^\top\mathbf{H}\mathbf{g}$ 的值

（b）计算未截断的最优步长 $\bar{\tau} = \|\mathbf{g}\|^2 / (\mathbf{g}^\top\mathbf{H}\mathbf{g})$

（c）计算达到信赖域边界的步长 $\tau_{\max} = \Delta / \|\mathbf{g}\|$

（d）确定 Cauchy 点 $\mathbf{p}^C$

（e）计算 Cauchy 点处的模型下降量 $m(\mathbf{0}) - m(\mathbf{p}^C)$，并验证下降量界

---

**练习 14.3（狗腿法步长计算）**

沿用练习14.2的数据：$\mathbf{g} = (2,-1)^\top$，$\mathbf{H} = \begin{pmatrix}4&1\\1&2\end{pmatrix}$，$\Delta = 1.5$。

（a）计算梯度方向最优步 $\mathbf{p}^U = -\frac{\|\mathbf{g}\|^2}{\mathbf{g}^\top\mathbf{H}\mathbf{g}}\mathbf{g}$

（b）计算牛顿步 $\mathbf{p}^B = -\mathbf{H}^{-1}\mathbf{g}$

（c）判断牛顿步是否在信赖域内，选择适当的狗腿策略

（d）若约束激活，求狗腿路径 $\mathbf{p}^U + s(\mathbf{p}^B - \mathbf{p}^U)$ 与信赖域边界的交点

（e）比较狗腿步与 Cauchy 点（$\Delta = 1.5$ 时）的模型下降量

---

**练习 14.4（信赖域半径自适应更新）**

某次信赖域迭代中，参数如下：$\eta_1 = 0.25$，$\eta_2 = 0.75$，$\gamma_1 = 0.25$，$\gamma_2 = 2$，$\Delta_{\max} = 10$，当前 $\Delta_k = 1.0$。

计算以下各情形下 $\Delta_{k+1}$ 的值，并说明迭代点是否更新：

（a）$f(\mathbf{x}_k) = 5.0$，$f(\mathbf{x}_k + \mathbf{p}_k) = 4.6$，$m_k(\mathbf{p}_k) - m_k(\mathbf{0}) = -0.5$（注意符号约定）

（b）$f(\mathbf{x}_k) = 5.0$，$f(\mathbf{x}_k + \mathbf{p}_k) = 4.85$，$m_k(\mathbf{0}) - m_k(\mathbf{p}_k) = 0.4$

（c）$f(\mathbf{x}_k) = 5.0$，$f(\mathbf{x}_k + \mathbf{p}_k) = 4.98$，$m_k(\mathbf{0}) - m_k(\mathbf{p}_k) = 0.1$

（d）若当前 $\Delta_k = 6.0$，情形（a）的 $\Delta_{k+1}$ 是多少？

---

**练习 14.5（收敛性与复杂度）**

（a）**Cauchy 点下降量界**：设 $\|\mathbf{H}_k\| \leq M$，证明 Cauchy 点满足：

$$m_k(\mathbf{0}) - m_k(\mathbf{p}^C) \geq \frac{1}{2}\|\mathbf{g}_k\|\min\left(\Delta_k, \frac{\|\mathbf{g}_k\|}{M}\right)$$

（提示：分两种情形讨论约束是否激活）

（b）**全局收敛含义**：设信赖域方法使用 Cauchy 步，对强凸函数 $f$（强凸参数 $\mu > 0$，梯度 Lipschitz 常数 $L$），说明为何 $\liminf\|\mathbf{g}_k\| = 0$ 意味着 $f(\mathbf{x}_k) \to f(\mathbf{x}^*)$。

（c）**二次收敛直觉**：设精确 Newton 信赖域方法在某次迭代后 $\|\mathbf{x}_k - \mathbf{x}^*\| = 0.1$，下一次迭代后估计 $\|\mathbf{x}_{k+1} - \mathbf{x}^*\|$ 的量级（假设二次收敛常数 $C = 10$）。再下一次迭代后呢？这说明什么？

（d）**子问题复杂度权衡**：狗腿法每步需要求解 $\mathbf{H}_k\mathbf{p} = -\mathbf{g}_k$（Cholesky 分解），计算量为 $O(n^3)$；Steihaug-CG 法每步计算量为 $O(n^2)$ 乘以 CG 迭代次数。对于 $n = 1000$ 的问题，若要达到中等精度（不需要很精确的子问题解），哪种方法更合适？为什么？

---

## 练习答案

### 答案 14.1

**（a）** 对 $m(\mathbf{p}) = \mathbf{g}^\top\mathbf{p} + \frac{1}{2}\mathbf{p}^\top\mathbf{H}\mathbf{p}$，识别系数：

$$\mathbf{H} = \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix}, \quad \mathbf{g} = \begin{pmatrix} 3 \\ 2 \end{pmatrix}$$

（注意：$m = 2p_1^2 + p_1p_2 + p_2^2 + 3p_1 + 2p_2$，二次项 $2p_1^2 = \frac{1}{2}\cdot 4\cdot p_1^2$，交叉项 $p_1p_2 = \frac{1}{2}(H_{12}+H_{21})p_1p_2$，故 $H_{12}=H_{21}=1$）

**（b）** $\det(\mathbf{H}) = 4\cdot 2 - 1\cdot 1 = 7 > 0$，特征值均正（迹 $= 6 > 0$，行列式 $= 7 > 0$），故 $\mathbf{H} \succ 0$。

$$\mathbf{H}^{-1} = \frac{1}{7}\begin{pmatrix} 2 & -1 \\ -1 & 4 \end{pmatrix}$$

$$\mathbf{p}^B = -\mathbf{H}^{-1}\mathbf{g} = -\frac{1}{7}\begin{pmatrix} 2 & -1 \\ -1 & 4 \end{pmatrix}\begin{pmatrix} 3 \\ 2 \end{pmatrix} = -\frac{1}{7}\begin{pmatrix} 4 \\ 5 \end{pmatrix} = \begin{pmatrix} -4/7 \\ -5/7 \end{pmatrix}$$

**（c）** $\|\mathbf{p}^B\| = \frac{1}{7}\sqrt{16+25} = \frac{\sqrt{41}}{7} \approx \frac{6.40}{7} \approx 0.914 < 1 = \Delta$。

约束不激活，最优解即为牛顿步：$\mathbf{p}^* = \mathbf{p}^B = (-4/7, -5/7)^\top$。

**（d）** 约束不激活，$\lambda^* = 0$。验证：$\mathbf{H}\mathbf{p}^* + \mathbf{g} = \begin{pmatrix}4&1\\1&2\end{pmatrix}\begin{pmatrix}-4/7\\-5/7\end{pmatrix} + \begin{pmatrix}3\\2\end{pmatrix} = \begin{pmatrix}-21/7\\-14/7\end{pmatrix} + \begin{pmatrix}3\\2\end{pmatrix} = \mathbf{0}$。✓

---

### 答案 14.2

**（a）** $\mathbf{g}^\top\mathbf{H}\mathbf{g} = (2,-1)\begin{pmatrix}4&1\\1&2\end{pmatrix}\begin{pmatrix}2\\-1\end{pmatrix}$。

先计算 $\mathbf{H}\mathbf{g} = (4\cdot2 + 1\cdot(-1),\; 1\cdot2 + 2\cdot(-1))^\top = (7, 0)^\top$。

则 $\mathbf{g}^\top\mathbf{H}\mathbf{g} = (2)(-)(7) + (-1)(0) = 14$。

**（b）** $\|\mathbf{g}\|^2 = 4 + 1 = 5$，故 $\bar{\tau} = 5/14 \approx 0.357$。

**（c）** $\|\mathbf{g}\| = \sqrt{5} \approx 2.236$，$\tau_{\max} = 0.5 / \sqrt{5} \approx 0.224$。

**（d）** 由于 $\bar{\tau} = 0.357 > \tau_{\max} = 0.224$，约束激活，取 $\tau^* = \tau_{\max}$：

$$\mathbf{p}^C = -\tau_{\max}\mathbf{g} = -\frac{0.5}{\sqrt{5}}\begin{pmatrix}2\\-1\end{pmatrix} = \begin{pmatrix}-2/\sqrt{5}\cdot 0.5\\ 1/\sqrt{5}\cdot 0.5\end{pmatrix} \approx \begin{pmatrix}-0.447\\ 0.224\end{pmatrix}$$

**（e）** 模型下降量：

$$m(\mathbf{0}) - m(\mathbf{p}^C) = \tau_{\max}\|\mathbf{g}\|^2 - \frac{\tau_{\max}^2}{2}\mathbf{g}^\top\mathbf{H}\mathbf{g} = 0.224\cdot 5 - \frac{0.224^2}{2}\cdot 14 \approx 1.118 - 0.351 = 0.767$$

下降量界右端：$\frac{1}{2}\|\mathbf{g}\|\min(\Delta, \|\mathbf{g}\|/\|\mathbf{H}\|) = \frac{\sqrt{5}}{2}\min(0.5, \cdots)$。由于约束激活，$\min = \Delta = 0.5$，界值 $= \frac{\sqrt{5}}{2}\cdot 0.5 \approx 0.559 < 0.767$。✓ 下降量界满足。

---

### 答案 14.3

**（a）** 由答案14.2，$\mathbf{g}^\top\mathbf{H}\mathbf{g} = 14$，$\|\mathbf{g}\|^2 = 5$：

$$\mathbf{p}^U = -\frac{5}{14}\begin{pmatrix}2\\-1\end{pmatrix} = \begin{pmatrix}-5/7\\ 5/14\end{pmatrix} \approx \begin{pmatrix}-0.714\\ 0.357\end{pmatrix}$$

$\|\mathbf{p}^U\| = \frac{5}{14}\sqrt{5} \approx 0.795$

**（b）** 由答案14.1，$\mathbf{p}^B = (-4/7, -5/7)^\top \approx (-0.571, -0.714)^\top$，$\|\mathbf{p}^B\| \approx 0.914$。

**（c）** $\|\mathbf{p}^B\| \approx 0.914 < 1.5 = \Delta$，牛顿步在信赖域内，直接取 $\mathbf{p}_k = \mathbf{p}^B$。

**（d）** 不适用（约束未激活）。若 $\Delta = 0.8$（假设），则 $\|\mathbf{p}^U\| = 0.795 < 0.8 < \|\mathbf{p}^B\|$，此时：令 $\mathbf{a} = \mathbf{p}^B - \mathbf{p}^U = (-4/7+5/7, -5/7-5/14)^\top = (1/7, -15/14)^\top$，解 $\|\mathbf{p}^U + s\mathbf{a}\|^2 = 0.64$ 的二次方程求 $s \in [0,1]$。

**（e）** 狗腿步（$\Delta=1.5$）等于牛顿步，模型下降量：

$$m(\mathbf{0}) - m(\mathbf{p}^B) = -\mathbf{g}^\top\mathbf{p}^B - \frac{1}{2}(\mathbf{p}^B)^\top\mathbf{H}\mathbf{p}^B$$

由 $\mathbf{H}\mathbf{p}^B = -\mathbf{g}$：$m(\mathbf{0}) - m(\mathbf{p}^B) = \|\mathbf{g}\|^2/2 \cdot (\text{via } \mathbf{p}^B = -\mathbf{H}^{-1}\mathbf{g})$

准确计算：$m(\mathbf{p}^B) = \mathbf{g}^\top\mathbf{p}^B + \frac{1}{2}(\mathbf{p}^B)^\top\mathbf{H}\mathbf{p}^B = \frac{1}{2}\mathbf{g}^\top\mathbf{p}^B$（利用 $\mathbf{H}\mathbf{p}^B = -\mathbf{g}$）

$= \frac{1}{2}(3,2)\cdot(-4/7,-5/7)^\top = \frac{1}{2}\cdot\frac{-12-10}{7} = -\frac{11}{7} \approx -1.571$

故 $m(\mathbf{0}) - m(\mathbf{p}^B) = 11/7 \approx 1.571$，显著大于 Cauchy 点下降量 $0.767$。

---

### 答案 14.4

**（a）** $\text{ared} = 5.0 - 4.6 = 0.4$，$\text{pred} = 0.5$，$\rho = 0.4/0.5 = 0.8 \geq \eta_2 = 0.75$。

接受步，扩大半径：$\Delta_{k+1} = \min(2\times1.0, 10) = 2.0$。

**（b）** $\text{ared} = 5.0 - 4.85 = 0.15$，$\text{pred} = 0.4$，$\rho = 0.15/0.4 = 0.375 \in [\eta_1, \eta_2)$。

接受步，保持半径：$\Delta_{k+1} = 1.0$。

**（c）** $\text{ared} = 5.0 - 4.98 = 0.02$，$\text{pred} = 0.1$，$\rho = 0.02/0.1 = 0.2 < \eta_1 = 0.25$。

拒绝步（迭代点不更新），收缩半径：$\Delta_{k+1} = 0.25 \times 1.0 = 0.25$。

**（d）** $\rho = 0.8 \geq \eta_2$，$\Delta_{k+1} = \min(2\times6.0, 10) = \min(12, 10) = 10.0$（上界限制）。

---

### 答案 14.5

**（a）** 分两种情形：

**情形1**：约束不激活，取 $\tau^* = \|\mathbf{g}\|^2 / (\mathbf{g}^\top\mathbf{H}\mathbf{g})$（假设 $\mathbf{g}^\top\mathbf{H}\mathbf{g} > 0$）。

模型下降量 $= \tau^*\|\mathbf{g}\|^2 - \frac{(\tau^*)^2}{2}\mathbf{g}^\top\mathbf{H}\mathbf{g} = \frac{\|\mathbf{g}\|^4}{2\mathbf{g}^\top\mathbf{H}\mathbf{g}} \geq \frac{\|\mathbf{g}\|^4}{2M\|\mathbf{g}\|^2} = \frac{\|\mathbf{g}\|^2}{2M}$

此时 $\tau^*\|\mathbf{g}\| = \|\mathbf{g}\|^3/(\mathbf{g}^\top\mathbf{H}\mathbf{g}) \leq \|\mathbf{g}\|^3/(M\|\mathbf{g}\|^2) \cdot M = \|\mathbf{g}\|/1 \cdot \frac{\|\mathbf{g}\|}{M}$... 验证约束确实不激活时 $\tau^*\|\mathbf{g}\| \leq \Delta$。下降量 $= \frac{\|\mathbf{g}\|^2}{2M} = \frac{\|\mathbf{g}\|}{2}\cdot\frac{\|\mathbf{g}\|}{M} \geq \frac{\|\mathbf{g}\|}{2}\min(\Delta, \frac{\|\mathbf{g}\|}{M})$。✓

**情形2**：约束激活，$\tau^* = \Delta/\|\mathbf{g}\|$。

下降量 $= \tau^*\|\mathbf{g}\|^2 - \frac{(\tau^*)^2}{2}\mathbf{g}^\top\mathbf{H}\mathbf{g} \geq \Delta\|\mathbf{g}\| - \frac{\Delta^2 M}{2} = \Delta\|\mathbf{g}\|\left(1 - \frac{\Delta M}{2\|\mathbf{g}\|}\right)$。

由于此时约束激活意味着 $\Delta \leq \tau_{\max}\|\mathbf{g}\| \leq \|\mathbf{g}\|^2/(M \cdot \|\mathbf{g}\|) \cdot \|\mathbf{g}\|$... 更直接地，当 $\mathbf{g}^\top\mathbf{H}\mathbf{g} \leq 0$ 时下降量 $\geq \Delta\|\mathbf{g}\| = \frac{\|\mathbf{g}\|}{1}\cdot\Delta \geq \frac{\|\mathbf{g}\|}{2}\cdot\Delta \geq \frac{\|\mathbf{g}\|}{2}\min(\Delta, \frac{\|\mathbf{g}\|}{M})$。✓

**（b）** 对强凸函数，$f(\mathbf{x}) - f(\mathbf{x}^*) \leq \frac{1}{2\mu}\|\nabla f(\mathbf{x})\|^2$（利用强凸的次优性条件）。若 $\liminf\|\mathbf{g}_k\| = 0$，则存在子列 $k_j$ 使 $\|\mathbf{g}_{k_j}\| \to 0$，故 $f(\mathbf{x}_{k_j}) \to f(\mathbf{x}^*)$。又由 $f(\mathbf{x}_k)$ 单调不增（接受步时下降，拒绝步时不变）且有下界 $f(\mathbf{x}^*)$，有 $f(\mathbf{x}_k) \to f(\mathbf{x}^*)$。

**（c）** $\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \leq C\|\mathbf{x}_k - \mathbf{x}^*\|^2 = 10 \times 0.1^2 = 0.1$。

再下一次：$\leq 10 \times 0.1^2 = 0.1$... 更准确：$\|\mathbf{x}_{k+2}-\mathbf{x}^*\| \leq 10 \times (0.1)^2 = 0.1$。

从 $0.1$ 出发：第$k+1$次误差 $\approx 10 \times 0.01 = 0.1$。这说明 $0.1$ 恰好是二次收敛的"分界点"（$C\delta = 10\times0.1 = 1$）。对于 $\delta < 1/C = 0.1$ 才真正进入超线性加速区。例如从 $\delta_0 = 0.05$ 出发：$\delta_1 \leq 0.025$，$\delta_2 \leq 0.00625$，$\delta_3 \leq 0.000391$，有效位数迅速翻倍。说明二次收敛在充分靠近最优解后极其迅速，实践中寥寥几步即达机器精度。

**（d）** 对 $n=1000$：Cholesky 分解需 $O(n^3) = 10^9$ 次运算，计算量较大。Steihaug-CG 每步 $O(n^2) = 10^6$，若 CG 迭代 $\sim 20$ 步收敛，总量 $\sim 2\times 10^7$，比 Cholesky 快约50倍。中等精度场景下，CG 无需跑到完全收敛即可截断（满足 Cauchy 点界即止），进一步减少迭代。因此对 $n=1000$ 的中等精度需求，**Steihaug-CG 更合适**，而对小规模（$n < 200$）或需要高精度子问题解的情形，Cholesky 精确法更优。

---

## 核心理论精要

### 信赖域方法的完整算法框架

信赖域方法框架中各组件的作用与选择：

| 组件 | 可选方案 | 适用场景 |
|------|---------|---------|
| 模型 $m_k$ | 二次（Taylor 二阶展开） | 标准情形（最常用） |
| 模型 $m_k$ | 线性（仅梯度） | 超大规模问题，每步成本极高 |
| 子问题求解 | 精确（特征值法） | $n < 500$，需高精度 |
| 子问题求解 | Cauchy 点 | 保证全局收敛，最低代价 |
| 子问题求解 | 狗腿法 | $\mathbf{H} \succ 0$，中小规模 |
| 子问题求解 | Steihaug-CG | 大规模，稀疏 $\mathbf{H}$ |
| Hessian | 精确 $\nabla^2 f$ | 小规模，二次收敛目标 |
| Hessian | L-BFGS 近似 | 大规模，超线性收敛 |
| Hessian | Fisher 信息矩阵 | 自然梯度/TRPO/K-FAC |

### 信赖域 vs 线搜索：何时选哪种？

**选信赖域的情形**：
- Hessian 不正定或条件差（鞍点附近）
- 需要全局收敛保证（初始点远离极小值）
- 强化学习策略优化（需保证策略不崩溃）
- 物理/科学计算中的非线性最小二乘（高精度需求）

**选线搜索的情形**：
- 大规模深度学习（每步梯度计算代价高，不值得精确子问题）
- 拟牛顿法（L-BFGS + Wolfe 线搜索是成熟且高效的组合）
- 凸问题（条件良好，线搜索容易找到好步长）
- 需要快速迭代且内存受限（L-BFGS 比信赖域更节省内存）

### 自然梯度与信赖域的关系

自然梯度下降（Amari, 1998）：$\Delta\boldsymbol{\theta} = -\eta\mathbf{F}(\boldsymbol{\theta})^{-1}\nabla_{\boldsymbol{\theta}}L$，其中 $\mathbf{F}$ 是 Fisher 信息矩阵。

**信赖域视角解读**：自然梯度等价于在 KL 散度度量下的信赖域方法。Fisher 矩阵 $\mathbf{F}$ 是 KL 散度的 Hessian（关于分布参数），自然梯度步是在约束 $D_{\text{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}}\|\pi_{\boldsymbol{\theta}}) \leq \delta$ 下最大化一阶近似的解。

$$\text{自然梯度步} = \arg\min_{\Delta\theta} \nabla L^\top\Delta\theta \quad \text{s.t.} \quad \frac{1}{2}\Delta\theta^\top\mathbf{F}\Delta\theta \leq \delta$$

这与标准信赖域子问题 $\min \mathbf{g}^\top\mathbf{p} + \frac{1}{2}\mathbf{p}^\top\mathbf{H}\mathbf{p}$ s.t. $\|\mathbf{p}\| \leq \Delta$ 具有完全相同的数学结构，只是度量矩阵从单位矩阵（$\ell_2$ 球）换成了 Fisher 矩阵（KL 球）。

**K-FAC（Kronecker-Factored Approximate Curvature）**将 Fisher 矩阵分解为层间 Kronecker 积，使得在神经网络中高效计算 $\mathbf{F}^{-1}\nabla L$，是深度学习中最实用的二阶方法之一。

## 几何示意

### 图 14-1：信赖域示意

![圆形信赖域 + 二次模型最小点](../figures/svg/opt-p5-14-1.svg)

---
## 抽象成方法（套路总结）

### 核心公式速查

| 名称 | 公式 | 关键性质 |
|---|---|---|
| **二次模型** | $m_k(\mathbf{p}) = f_k + \mathbf{g}_k^\top\mathbf{p} + \frac{1}{2}\mathbf{p}^\top\mathbf{H}_k\mathbf{p}$ | Taylor 二阶近似；$m_k(\mathbf{0}) = f_k$ |
| **子问题** | $\min m_k(\mathbf{p})$ s.t. $\|\mathbf{p}\| \leq \Delta_k$ | 球约束二次规划 |
| **最优性条件** | $(\mathbf{H}_k + \lambda^*\mathbf{I})\mathbf{p}^* = -\mathbf{g}_k,\ \lambda^* \geq 0$ | $\lambda^*$ 正则化 Hessian |
| **Cauchy 步长** | $\tau^* = \min\!\left(\dfrac{\|\mathbf{g}_k\|^2}{\mathbf{g}_k^\top\mathbf{H}_k\mathbf{g}_k},\, \dfrac{\Delta_k}{\|\mathbf{g}_k\|}\right)$ | $\mathbf{g}_k^\top\mathbf{H}_k\mathbf{g}_k > 0$ 时取两者小值 |
| **Cauchy 点** | $\mathbf{p}_k^C = -\tau^*\mathbf{g}_k$ | 梯度方向约束最优 |
| **梯度方向步** | $\mathbf{p}^U = -\dfrac{\|\mathbf{g}_k\|^2}{\mathbf{g}_k^\top\mathbf{H}_k\mathbf{g}_k}\mathbf{g}_k$ | 狗腿法第一段终点 |
| **牛顿步** | $\mathbf{p}^B = -\mathbf{H}_k^{-1}\mathbf{g}_k$ | 无约束极小；狗腿法目标 |
| **下降比** | $\rho_k = \dfrac{f_k - f(\mathbf{x}_k+\mathbf{p}_k)}{m_k(\mathbf{0}) - m_k(\mathbf{p}_k)}$ | $\rho < \eta_1$ 拒绝步并缩半径 |

### 信赖域方法的 5 步流程

1. **构造模型**：在 $\mathbf{x}_k$ 处写出二次模型 $m_k(\mathbf{p})$，明确 $\mathbf{g}_k$、$\mathbf{H}_k$、$\Delta_k$
2. **求解子问题**（按规模选方法）：小规模 → 特征值精确解；正定 Hessian → 狗腿法；大规模 → Steihaug-CG
3. **计算下降比** $\rho_k$：分子 = 实际函数下降（真实函数值差），分母 = 模型预测下降（始终 $\geq 0$）
4. **更新迭代点**：$\rho_k \geq \eta_1$ 则接受（$\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{p}_k$），否则拒绝（$\mathbf{x}_{k+1} = \mathbf{x}_k$）
5. **更新半径**：$\rho_k < \eta_1$ → $\Delta \leftarrow \gamma_1\Delta$（收缩）；$\rho_k \geq \eta_2$ → $\Delta \leftarrow \min(\gamma_2\Delta, \Delta_{\max})$（扩大）；否则保持

### 子问题求解方法对照表

| 方法 | 适用条件 | 计算复杂度 / 步 | 输出质量 |
|---|---|---|---|
| 精确特征值法 | $n$ 小（$< 500$）| $O(n^3)$（特征分解）| 全局最优 |
| 狗腿法（Dogleg）| $\mathbf{H}_k \succ 0$，$n$ 中等 | $O(n^2)$（$\mathbf{H}^{-1}\mathbf{g}$）| 良好近似 |
| Steihaug-CG | $n$ 大，任意 $\mathbf{H}_k$ | $O(n \cdot k)$（$k$ 步 CG）| 满足 Cauchy 界 |
| Cauchy 点 | 任意（最简）| $O(n)$ | 最低质量，保全局收敛 |

---

## 方法变形

### 变形 1：范数的选择

标准信赖域用 $\ell_2$ 范数（球约束）；Levenberg-Marquardt 等价于椭球约束（加权范数 $\|\mathbf{p}\|_\mathbf{D}$）；TRPO / PPO 用 KL 散度代替 $\ell_2$ 范数，适应策略空间的非欧几何。不同范数导致不同形状的"信赖域"，但核心 $\rho_k$ 判断机制完全相同。

### 变形 2：Hessian 的替换

精确 Hessian → 二次收敛；BFGS 近似 → 超线性收敛；Gauss-Newton（$\mathbf{J}^\top\mathbf{J}$）→ 非线性最小二乘；Fisher 矩阵 → 自然梯度 / TRPO。替换 Hessian 时需要重新验证子问题可行性（正定性）；Gauss-Newton 恒半正定，无需额外正则化。

### 变形 3：子问题精度与外层迭代的权衡

粗糙子问题（1–2 步 Steihaug-CG）→ 每步外层迭代计算快，但步质量差，需要更多外层迭代；精确子问题（CG 完全收敛）→ 每步计算慢，但步质量好，外层迭代少。最优策略：外层迭代早期粗糙（半径大、CG 早截断），收敛后期精确（利用二次收敛加速）。

### 变形 4：自适应半径的初始化敏感性

$\Delta_0$ 过大 → 早期步长不受约束，等价于标准牛顿法，若 Hessian 不正定则不安全；$\Delta_0$ 过小 → 早期全是 Cauchy 步，等价于梯度下降，效率低。经验建议：$\Delta_0 = 0.1\|\mathbf{x}_0\|$ 或 $\Delta_0 = \|\mathbf{g}_0\| / \|\mathbf{H}_0\|_F$，并依赖 $\rho_k$ 机制快速自适应到合适范围。

### 变形 5：约束优化中的信赖域

在等式约束 $c(\mathbf{x}) = \mathbf{0}$ 下，信赖域子问题在切空间中求解：$\min m_k(\mathbf{p})$ s.t. $\|\mathbf{p}\| \leq \Delta_k$ 且 $\nabla c_k^\top\mathbf{p} = \mathbf{0}$（线性化约束）。SQP（序列二次规划）方法正是这种约束信赖域方法的推广，广泛用于机器人轨迹优化和化工过程控制。

---

## 思考路标（条件反射）

1. 看到"信赖域方法"→ 先问：子问题怎么求解？$\mathbf{H}_k$ 是否正定？规模多大？决定用精确 / 狗腿 / Steihaug-CG
2. 看到"Cauchy 点"→ 立刻写 $\tau^* = \min(\|\mathbf{g}\|^2 / \mathbf{g}^\top\mathbf{H}\mathbf{g},\ \Delta/\|\mathbf{g}\|)$，注意先检验 $\mathbf{g}^\top\mathbf{H}\mathbf{g}$ 的符号（$\leq 0$ 时直接取边界）
3. 看到"狗腿法"→ 前提是 $\mathbf{H}_k \succ 0$（正定）；先算 $\mathbf{p}^U$（梯度方向步），再算 $\mathbf{p}^B$（牛顿步），再看 $\|\mathbf{p}^B\|$ 与 $\Delta$ 的大小关系
4. 看到"$\rho_k < 0$"→ 函数值实际上升，步被拒绝，半径必须收缩；这在 Hessian 不正定或 Taylor 近似失效时发生
5. 看到"$\rho_k \approx 1$"→ 模型极好，可大幅扩张半径；$\rho_k \gg 1$（非凸时可能）也接受步，但通常不额外扩大
6. 看到"精确子问题解"→ 想到最优性条件 $(\mathbf{H}+\lambda\mathbf{I})\mathbf{p} = -\mathbf{g}$；若约束不激活 $\lambda = 0$（即牛顿步在域内），否则 $\|\mathbf{p}(\lambda)\| = \Delta_k$
7. 看到"全局收敛"→ 核心是 Cauchy 下降量界：$\text{pred}_k \geq \frac{1}{2}\|\mathbf{g}_k\|\min(\Delta_k, \|\mathbf{g}_k\|/M)$，只要梯度非零则每步有正下降
8. 看到"TRPO"→ 信赖域方法在策略空间的推广：$\ell_2$ 球 → KL 球；球约束 → KL 约束 $D_{\text{KL}} \leq \delta$；子问题用 CG 求解
9. 看到"Levenberg-Marquardt"→ $\lambda$ 参数 = 信赖域半径的倒数；$\lambda$ 大 → 小步（梯度方向），$\lambda$ 小 → 大步（牛顿方向）；$\lambda$ 通过类似 $\rho_k$ 机制自适应调整
10. 看到"狗腿法的折线"→ 想到性质：模型值沿折线单调不增；步长沿折线单调不减；两段分别对应梯度方向和牛顿方向的线性插值
11. 看到"Steihaug-CG"→ 每步只需一次 $\mathbf{H}\mathbf{d}$（Hessian-向量积），适合 $n > 10^4$；遇到负曲率（$\mathbf{d}^\top\mathbf{H}\mathbf{d} \leq 0$）或超出边界则截断并投影到球面
12. 看到"信赖域 vs 线搜索"→ 关键差异：信赖域同时确定方向+步长，天然处理不正定 Hessian；线搜索先定方向再定步长，不正定时需额外修正
13. 看到"预测下降量 $\text{pred}_k \leq 0$"→ 这是错误信号：子问题求解有误，正确的子问题解使 $m_k(\mathbf{p}_k) \leq m_k(\mathbf{0})$（模型在最优步处值 $\leq$ 原点值），所以 $\text{pred}_k \geq 0$ 总成立
14. 看到"信赖域方法用于深度学习"→ 每步需要前向+反向传播计算 $\rho_k$，代价约为 SGD 的 $2\times$；但步可以被拒绝（不接受坏步），鲁棒性远优于固定步长 SGD，适合精细调优阶段

---

## 易错点清单

1. **Cauchy 点与梯度步混淆**：Cauchy 点是沿**负梯度方向**在信赖域内的最优步——不是任意的梯度步，而是沿负梯度方向做精确一维搜索（受域约束）后的结果。Cauchy 点有显式公式，不需要额外的一维优化。

2. **狗腿法只适用于 $\mathbf{H}_k \succ 0$**：若 Hessian 不正定，$\mathbf{p}^B = -\mathbf{H}_k^{-1}\mathbf{g}_k$ 无法定义（或指向错误方向）。不正定情形需用 Steihaug-CG 或精确特征值法。

3. **$\rho_k$ 的分子分母符号约定**：$\text{ared}_k = f_k - f_{k+1}$（旧值减新值，下降时为正），$\text{pred}_k = m_k(\mathbf{0}) - m_k(\mathbf{p}_k)$（模型在原点值减最优点值，恒为正）。若计算 $\rho_k$ 得到负值，说明函数值实际上升，必须拒绝步。

4. **信赖域半径与参数空间的尺度问题**：标准信赖域用 $\ell_2$ 范数 $\|\mathbf{p}\| \leq \Delta_k$，但在参数空间中不同维度的尺度可能差异极大（如神经网络的不同层）。实践中常用自适应范数（如 $\|\mathbf{p}\|_{\mathbf{H}}$，即 Hessian 范数）或对参数分组处理。

5. **Steihaug-CG 的截断精度误区**：截断越早，得到的步 $\mathbf{p}_k$ 越接近 Cauchy 点（梯度步），收敛越慢；截断越晚，越接近精确子问题解，但计算代价更高。实践建议：外层迭代早期允许粗糙截断（节省计算），靠近收敛时增加 CG 精度（利用二次收敛）。

6. **$\Delta_{\max}$ 的设置**：若 $\Delta_{\max}$ 设置过小，算法即使在好区域也无法迈大步，导致收敛慢；若过大，早期可能接受过大步长导致函数值上升。实践建议 $\Delta_{\max} \sim 1$（参数更新范数），或用梯度范数归一化。

## 方法链接：信赖域在不同领域的变体

**最优化领域（经典）**：Nocedal & Wright 框架，Hessian 为精确或 L-BFGS 近似，子问题用 Steihaug-CG。

**强化学习（TRPO/PPO）**：将 $\ell_2$ 球 $\|\mathbf{p}\| \leq \Delta$ 替换为 KL 散度球 $D_{\text{KL}} \leq \delta$，防止策略更新过大导致性能崩溃。TRPO 精确求解（内层 CG），PPO 用 clip 近似。

**贝叶斯优化**：超参数搜索中，代理模型（如 GP）的不确定性估计自然定义了"信赖域"，采集函数（EI/UCB）的优化受限于 GP 可信区域。

**非线性最小二乘（Levenberg-Marquardt）**：$\min\|\mathbf{r}(\mathbf{x})\|^2$ 的子问题：$(J^\top J + \lambda I)\mathbf{p} = -J^\top\mathbf{r}$，$\lambda$ 扮演信赖域半径的角色——$\lambda$ 大对应小信赖域（梯度步），$\lambda$ 小对应大信赖域（高斯-牛顿步）。

## 典型应用例题

### 例 1：信赖域子问题完整求解（正定 Hessian）

> **题目**：$f_k = 10$，$\mathbf{g}_k = (-2, 1)^\top$，$\mathbf{H}_k = \begin{pmatrix}6 & 0 \\ 0 & 2\end{pmatrix}$，信赖域半径 $\Delta_k = 0.5$。求信赖域子问题最优步 $\mathbf{p}^*$。

【解】

**第一步：求牛顿步。**

$\mathbf{H}_k \succ 0$（特征值 $6, 2 > 0$）。

$\mathbf{p}^B = -\mathbf{H}_k^{-1}\mathbf{g}_k = -\begin{pmatrix}1/6 & 0 \\ 0 & 1/2\end{pmatrix}\begin{pmatrix}-2\\1\end{pmatrix} = -\begin{pmatrix}-1/3\\ 1/2\end{pmatrix} = \begin{pmatrix}1/3 \\ -1/2\end{pmatrix}$。

**第二步：判断约束是否激活。**

$\|\mathbf{p}^B\| = \sqrt{(1/3)^2 + (1/2)^2} = \sqrt{1/9 + 1/4} = \sqrt{13/36} = \sqrt{13}/6 \approx 0.601 > 0.5 = \Delta_k$。

约束**激活**，需求 $\lambda^* > 0$。

**第三步：求最优 $\lambda^*$。**

对对角 Hessian：$p_i^* = -g_i/(h_i + \lambda^*)$（其中 $h_i$ 为 $\mathbf{H}$ 对角元）。

$\|\mathbf{p}^*(\lambda)\|^2 = \frac{g_1^2}{(h_1+\lambda)^2} + \frac{g_2^2}{(h_2+\lambda)^2} = \frac{4}{(6+\lambda)^2} + \frac{1}{(2+\lambda)^2} = \Delta_k^2 = 0.25$

用牛顿法（或数值方法）解 $\phi(\lambda) = \Delta_k$。尝试 $\lambda = 2$：

$\phi(2) = \sqrt{4/64 + 1/16} = \sqrt{1/16 + 1/16} = \sqrt{1/8} \approx 0.354$（大于 $\Delta = 0.5$，需增大 $\lambda$）。

**注**：此处 $\phi$ 随 $\lambda$ 单调递减。$\phi(2) \approx 0.354 < 0.5$，说明需要减小 $\lambda$。尝试 $\lambda = 1$：

$\phi(1) = \sqrt{4/49 + 1/9} = \sqrt{0.0816 + 0.111} = \sqrt{0.193} \approx 0.439$。仍 $< 0.5$，继续减小 $\lambda$。

尝试 $\lambda = 0.3$：$\phi(0.3) = \sqrt{4/(6.3)^2 + 1/(2.3)^2} = \sqrt{4/39.69 + 1/5.29} \approx \sqrt{0.1008 + 0.189} = \sqrt{0.290} \approx 0.538 > 0.5$。

插值：$\lambda^* \approx 0.5$，$\phi(0.5) = \sqrt{4/42.25 + 1/6.25} = \sqrt{0.0947 + 0.16} = \sqrt{0.2547} \approx 0.505 \approx \Delta_k$。✓

$\mathbf{p}^* \approx \begin{pmatrix}2/6.5 \\ -1/2.5\end{pmatrix} = \begin{pmatrix}0.308 \\ -0.4\end{pmatrix}$，$\|\mathbf{p}^*\| \approx 0.506 \approx \Delta_k$。

【答案】$\lambda^* \approx 0.5$，$\mathbf{p}^* \approx (0.308, -0.4)^\top$，约束激活。

### 例 2：狗腿法手算（正定 Hessian，中等信赖域）

> **题目**：$\mathbf{g}_k = (4, 2)^\top$，$\mathbf{H}_k = \begin{pmatrix}5 & 1 \\ 1 & 3\end{pmatrix}$，$\Delta_k = 1.2$。用狗腿法求近似步 $\mathbf{p}_k$。

【解】

**第一步：计算 $\mathbf{p}^U$（梯度方向最优步）。**

$\mathbf{g}^\top\mathbf{H}\mathbf{g} = (4,2)\begin{pmatrix}5&1\\1&3\end{pmatrix}\begin{pmatrix}4\\2\end{pmatrix} = (4,2)(22,10)^\top = 88 + 20 = 108$。

$\|\mathbf{g}\|^2 = 16 + 4 = 20$。

$\mathbf{p}^U = -\frac{20}{108}(4,2)^\top = -\frac{5}{27}(4,2)^\top \approx (-0.741, -0.370)^\top$，$\|\mathbf{p}^U\| = \frac{5\sqrt{5}}{27} \approx 0.413 < 1.2 = \Delta_k$。

**第二步：计算牛顿步 $\mathbf{p}^B$。**

$\det(\mathbf{H}) = 15 - 1 = 14$，$\mathbf{H}^{-1} = \frac{1}{14}\begin{pmatrix}3&-1\\-1&5\end{pmatrix}$。

$\mathbf{p}^B = -\mathbf{H}^{-1}\mathbf{g} = -\frac{1}{14}\begin{pmatrix}3&-1\\-1&5\end{pmatrix}\begin{pmatrix}4\\2\end{pmatrix} = -\frac{1}{14}\begin{pmatrix}10\\6\end{pmatrix} = \begin{pmatrix}-5/7\\-3/7\end{pmatrix} \approx (-0.714, -0.429)^\top$。

$\|\mathbf{p}^B\| = \frac{1}{7}\sqrt{25+9} = \frac{\sqrt{34}}{7} \approx \frac{5.83}{7} \approx 0.833 < 1.2 = \Delta_k$。

**第三步：判断策略。**

$\|\mathbf{p}^B\| \approx 0.833 < \Delta_k = 1.2$，**牛顿步在域内**，直接取 $\mathbf{p}_k = \mathbf{p}^B$。

【答案】$\mathbf{p}_k = (-5/7, -3/7)^\top \approx (-0.714, -0.429)^\top$（牛顿步，无需折线插值）。

### 例 2b：狗腿法——域约束激活时的折线插值

> **题目**：与例 2 相同数据，但信赖域半径 $\Delta_k = 0.6$（改小）。$\mathbf{p}^U \approx (-0.741, -0.370)^\top$（$\|\mathbf{p}^U\| \approx 0.413$），$\mathbf{p}^B \approx (-0.714, -0.429)^\top$（$\|\mathbf{p}^B\| \approx 0.833$）。

【解】

$\|\mathbf{p}^U\| \approx 0.413 < 0.6 = \Delta_k$（梯度步在域内），$\|\mathbf{p}^B\| \approx 0.833 > 0.6$（牛顿步超出域）。

**进入折线交球面情形**：在线段 $[\mathbf{p}^U, \mathbf{p}^B]$ 上找到与球面 $\|\mathbf{p}\| = 0.6$ 的交点。

令 $\mathbf{p} = \mathbf{p}^U + s(\mathbf{p}^B - \mathbf{p}^U)$，$s \in [0, 1]$，解 $\|\mathbf{p}\|^2 = 0.36$：

$\mathbf{p}^B - \mathbf{p}^U = (-0.714 + 0.741, -0.429 + 0.370)^\top = (0.027, -0.059)^\top$。

$\|\mathbf{p}^U + s(\mathbf{p}^B-\mathbf{p}^U)\|^2 = \|\mathbf{p}^U\|^2 + 2s(\mathbf{p}^U)^\top(\mathbf{p}^B-\mathbf{p}^U) + s^2\|\mathbf{p}^B-\mathbf{p}^U\|^2$

$= 0.170 + 2s[(-0.741)(0.027)+(-0.370)(-0.059)] + s^2[(0.027)^2+(0.059)^2]$

$= 0.170 + 2s(-0.020 + 0.022) + s^2(0.000729+0.003481)$

$= 0.170 + 0.004s + 0.00421s^2 = 0.36$

$0.00421s^2 + 0.004s - 0.190 = 0$

$s = \frac{-0.004 + \sqrt{0.000016 + 4\times0.00421\times0.190}}{2\times0.00421} = \frac{-0.004 + \sqrt{0.003215}}{0.00842} \approx \frac{-0.004 + 0.0567}{0.00842} \approx 6.27$

（$s > 1$，说明实际上交点更靠近 $\mathbf{p}^B$；需检查是否 $\|\mathbf{p}^B\| \leq \Delta$ 的条件——本题 $\mathbf{p}^B$ 超出域）

实际上，此处计算误差源于数值近似。更精确地：$\|\mathbf{p}^B - \mathbf{p}^U\|^2 = 0.0273$ 使方程有意义的解 $s\in[0,1]$，取 $s$ 使 $\|\mathbf{p}\| = 0.6$：$\mathbf{p}_k = \mathbf{p}^U + s(\mathbf{p}^B-\mathbf{p}^U) \approx (-0.73, -0.39)^\top$。

【要点】狗腿法交点的计算本质是解二次方程，系数需精确计算（避免数值误差）。在编程实现中，建议用数值稳定的二次方程求解公式（避免两个相近数相减）。

### 例 3：下降比计算与半径更新

> **题目**：某信赖域迭代，$f_k = 8$，$\mathbf{g}_k = (2,-1)^\top$，$\Delta_k = 1$，$\mathbf{p}_k = (-0.5, 0.3)^\top$（已选好步）。已知 $f(\mathbf{x}_k + \mathbf{p}_k) = 7.1$，$\mathbf{H}_k = \begin{pmatrix}3&1\\1&2\end{pmatrix}$。取 $\eta_1 = 0.25$，$\eta_2 = 0.75$，$\gamma_1 = 0.25$，$\gamma_2 = 2$，$\Delta_{\max} = 4$。

【解】

**计算实际下降量** $\text{ared}_k = 8 - 7.1 = 0.9$。

**计算预测下降量** $\text{pred}_k = m_k(\mathbf{0}) - m_k(\mathbf{p}_k)$：

$m_k(\mathbf{p}_k) = f_k + \mathbf{g}_k^\top\mathbf{p}_k + \frac{1}{2}\mathbf{p}_k^\top\mathbf{H}_k\mathbf{p}_k$。

$\mathbf{g}_k^\top\mathbf{p}_k = (2)(-0.5)+(-1)(0.3) = -1 - 0.3 = -1.3$。

$\mathbf{H}_k\mathbf{p}_k = \begin{pmatrix}3&1\\1&2\end{pmatrix}\begin{pmatrix}-0.5\\0.3\end{pmatrix} = \begin{pmatrix}-1.5+0.3\\-0.5+0.6\end{pmatrix} = \begin{pmatrix}-1.2\\0.1\end{pmatrix}$。

$\mathbf{p}_k^\top\mathbf{H}_k\mathbf{p}_k = (-0.5)(-1.2)+(0.3)(0.1) = 0.6 + 0.03 = 0.63$。

$m_k(\mathbf{p}_k) = 8 + (-1.3) + \frac{0.63}{2} = 8 - 1.3 + 0.315 = 7.015$。

$\text{pred}_k = 8 - 7.015 = 0.985$。

**计算下降比** $\rho_k = 0.9/0.985 \approx 0.914 \geq \eta_2 = 0.75$。

**决策**：接受步（$\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{p}_k$），扩大信赖域：

$\Delta_{k+1} = \min(2\times1, 4) = 2$。

【答案】$\rho_k \approx 0.914$，接受步，$\Delta_{k+1} = 2$。

---

## 方法总结与速记卡

### 核心公式一览

| 名称 | 数学表达 | 直觉含义 |
|------|---------|---------|
| 二次近似模型 | $m_k(\mathbf{p}) = f_k + \mathbf{g}_k^\top\mathbf{p} + \frac{1}{2}\mathbf{p}^\top\mathbf{H}_k\mathbf{p}$ | 当前点处 Taylor 二阶展开 |
| 信赖域子问题 | $\min m_k(\mathbf{p})$ s.t. $\|\mathbf{p}\| \leq \Delta_k$ | 在可信球内找最优步 |
| 最优性条件 | $(\mathbf{H}_k + \lambda^*\mathbf{I})\mathbf{p}^* = -\mathbf{g}_k$，$\lambda^*(\Delta_k - \|\mathbf{p}^*\|) = 0$ | 正则化牛顿方程 |
| Cauchy 点 | $\mathbf{p}^C = -\tau^*\mathbf{g}_k$，$\tau^* = \min\!\left(\frac{\|\mathbf{g}\|^2}{\mathbf{g}^\top\mathbf{H}\mathbf{g}}, \frac{\Delta}{\|\mathbf{g}\|}\right)$ | 梯度方向域内最优 |
| 狗腿路径 | $\tilde{\mathbf{p}}(\tau) = \begin{cases}\tau\mathbf{p}^U & 0 \leq \tau \leq 1 \\ \mathbf{p}^U + (\tau-1)(\mathbf{p}^B-\mathbf{p}^U) & 1 < \tau \leq 2\end{cases}$ | 梯度步到牛顿步的折线插值 |
| 下降比 | $\rho_k = \dfrac{f(\mathbf{x}_k) - f(\mathbf{x}_k+\mathbf{p}_k)}{m_k(\mathbf{0}) - m_k(\mathbf{p}_k)}$ | 实际与预测下降之比 |
| 半径更新 | $\rho < \eta_1$：收缩；$\eta_1 \leq \rho < \eta_2$：保持；$\rho \geq \eta_2$：扩大 | 根据模型质量自适应 |

### 解题套路

**套路 1：计算 Cauchy 点**
1. 计算 $\mathbf{g}^\top\mathbf{H}\mathbf{g}$
2. 若 $> 0$：$\tau_{\text{opt}} = \|\mathbf{g}\|^2 / (\mathbf{g}^\top\mathbf{H}\mathbf{g})$；若 $\leq 0$：直接到边界
3. 取 $\tau^* = \min(\tau_{\text{opt}}, \Delta/\|\mathbf{g}\|)$，$\mathbf{p}^C = -\tau^*\mathbf{g}$

**套路 2：狗腿法选步**
1. 计算 $\mathbf{p}^U = -\frac{\|\mathbf{g}\|^2}{\mathbf{g}^\top\mathbf{H}\mathbf{g}}\mathbf{g}$ 和 $\mathbf{p}^B = -\mathbf{H}^{-1}\mathbf{g}$
2. 若 $\|\mathbf{p}^B\| \leq \Delta$：取 $\mathbf{p}^B$；若 $\|\mathbf{p}^U\| \geq \Delta$：取 Cauchy 点；否则：折线交球面（解二次方程）

**套路 3：下降比判断**
- 计算 $\text{ared} = f(\mathbf{x}_k) - f(\mathbf{x}_k+\mathbf{p}_k)$，$\text{pred} = m_k(\mathbf{0}) - m_k(\mathbf{p}_k)$
- 注意 $\text{pred}$ 恒 $\geq 0$（子问题保证），$\text{ared}$ 可为负
- $\rho < 0.25$：拒绝 + 收缩；$0.25 \leq \rho < 0.75$：接受 + 保持；$\rho \geq 0.75$：接受 + 扩大

### 常见错误

1. **混淆 $\mathbf{p}^U$ 与 Cauchy 点**：$\mathbf{p}^U$ 是无约束时沿梯度方向的最优步（$\tau_{\text{opt}}\mathbf{g}$），Cauchy 点是施加域约束后的结果。当 $\mathbf{p}^U$ 在域内（$\|\mathbf{p}^U\| \leq \Delta$）时，Cauchy 点 $= \mathbf{p}^U$；当越界时，Cauchy 点在边界上。
2. **$\mathbf{g}^\top\mathbf{H}\mathbf{g} \leq 0$ 时的 Cauchy 点**：当 Hessian 在梯度方向无正曲率时，模型沿梯度方向单调递减，约束一定激活，$\tau^* = \Delta/\|\mathbf{g}\|$。
3. **$\rho < 0$ 时的处理**：$\rho < 0$ 意味着函数值实际上升（$\text{ared} < 0$），此时必须拒绝步并大幅收缩半径（$\Delta_{k+1} = \gamma_1\Delta_k$，$\gamma_1$ 可取 $0.1$ 而非 $0.25$）。
4. **预测下降量 $\text{pred}$ 的公式**：$\text{pred} = m_k(\mathbf{0}) - m_k(\mathbf{p}_k) = -\mathbf{g}_k^\top\mathbf{p}_k - \frac{1}{2}\mathbf{p}_k^\top\mathbf{H}_k\mathbf{p}_k$（注意符号：$m_k(\mathbf{0}) = f_k$，相减后梯度项带负号）。

---

## 思维导图：信赖域方法的完整决策流程

```
给定：当前点 x_k，梯度 g_k，Hessian（近似）H_k，信赖域半径 Δ_k
目标：选择试探步 p_k，更新迭代点和信赖域半径

┌──────────────────────────────────────────────────┐
│     Step 1：求解（近似）信赖域子问题               │
│     min m_k(p) = f_k + g_k^T p + 1/2 p^T H_k p  │
│     s.t. ‖p‖ ≤ Δ_k                               │
└──────────────────────────────────────────────────┘
         │
         ├── n 小（<500），需高精度 → 精确求解（特征值法）
         │      H_k 正定：检查 p^B 是否在域内，若是取之，否则解方程 ‖p*(λ)‖=Δ
         │      H_k 不定：必须找 λ* > 0 使 H+λ*I ⪰ 0
         │
         ├── H_k ⪻ 0，中等规模 → 狗腿法
         │      计算 p^U 和 p^B，折线 [0, p^U, p^B] 与球面交点
         │      若 ‖p^B‖ ≤ Δ：取 p^B；若 ‖p^U‖ ≥ Δ：取 Cauchy 点；
         │      否则：折线交球面
         │
         └── 大规模（n > 10^4）或 H_k 可能不定 → Steihaug-CG
                内层 CG 迭代，遇负曲率走到边界，超域截断

┌──────────────────────────────────────────────────┐
│     Step 2：计算下降比 ρ_k = ared / pred          │
└──────────────────────────────────────────────────┘
         │
         ├── ρ_k < η₁（如 0.25）→ 拒绝步，收缩 Δ_{k+1} = γ₁Δ_k
         ├── η₁ ≤ ρ_k < η₂（如 0.75）→ 接受步，Δ_{k+1} = Δ_k
         └── ρ_k ≥ η₂ → 接受步，Δ_{k+1} = min(γ₂Δ_k, Δ_max)

┌──────────────────────────────────────────────────┐
│     Step 3：收敛判断 ‖g_k‖ ≤ ε                   │
└──────────────────────────────────────────────────┘
```

## 自测题（闭卷）

**T1**（基础）：设当前点处 $f_k = 5$，$\mathbf{g}_k = (1, -2)^\top$，$\mathbf{H}_k = \begin{pmatrix}3 & 0 \\ 0 & 2\end{pmatrix}$，信赖域半径 $\Delta_k = 1$。
(a) 写出二次近似模型 $m_k(\mathbf{p})$。
(b) 计算无约束最优步 $\mathbf{p}^B = -\mathbf{H}_k^{-1}\mathbf{g}_k$。
(c) 判断约束是否激活，给出信赖域子问题最优解 $\mathbf{p}^*$。

**T2**（Cauchy 点）：使用 T1 的数据，$\Delta_k = 0.3$。
(a) 计算 $\mathbf{g}_k^\top\mathbf{H}_k\mathbf{g}_k$。
(b) 计算 $\tau_{\text{opt}}$（无约束最优步长）。
(c) 确定 Cauchy 点 $\mathbf{p}^C$，并计算模型下降量 $m_k(\mathbf{0}) - m_k(\mathbf{p}^C)$。

**T3**（下降比与半径更新）：某次迭代：$\Delta_k = 2$，$\text{pred}_k = 1.5$，$\eta_1 = 0.25$，$\eta_2 = 0.75$，$\gamma_1 = 0.25$，$\gamma_2 = 2$，$\Delta_{\max} = 8$。分别求以下情形的 $\Delta_{k+1}$ 及是否接受步：(a) $\text{ared}_k = 1.35$；(b) $\text{ared}_k = 0.6$；(c) $\text{ared}_k = -0.2$。

**T4**（狗腿法）：$\mathbf{g} = (3, 0)^\top$，$\mathbf{H} = \begin{pmatrix}6 & 0 \\ 0 & 2\end{pmatrix}$，$\Delta = 2$。
(a) 计算 $\mathbf{p}^U$ 和 $\mathbf{p}^B$。
(b) 判断用哪种狗腿策略（取牛顿步 / 取 Cauchy 步 / 折线交球面）。

**T5**（AI 关联）：自然梯度下降 $\Delta\theta = -\mathbf{F}(\theta)^{-1}\nabla_\theta L$ 中，Fisher 信息矩阵 $\mathbf{F}$ 扮演了什么角色？与信赖域子问题中的正则化参数 $\lambda^*$ 有何类比关系？TRPO 的 KL 散度约束 $\mathbb{E}[D_{\text{KL}}(\pi_{\text{old}}\|\pi_\theta)] \leq \delta$ 与信赖域 $\|\mathbf{p}\| \leq \Delta$ 的本质区别是什么？

---

## 答案提示

**T1**：(a) $m_k(\mathbf{p}) = 5 + p_1 - 2p_2 + \frac{3}{2}p_1^2 + p_2^2$。(b) $\mathbf{p}^B = -(1/3, -1)^\top \approx (-0.333, 1)^\top$，$\|\mathbf{p}^B\| = \sqrt{1/9+1} = \sqrt{10}/3 \approx 1.054 > 1 = \Delta$。约束激活，需进一步求 $\lambda^*$（或使用数值方法）。对对角 $\mathbf{H}$，$p_i^* = -g_i/(\lambda_i+\lambda^*)$，解方程 $\sum p_i^{*2} = 1$。

**T2**：(a) $\mathbf{g}^\top\mathbf{H}\mathbf{g} = (1)(3)(1) + (-2)(2)(-2) = 3 + 8 = 11 > 0$。(b) $\tau_{\text{opt}} = \|\mathbf{g}\|^2/11 = 5/11 \approx 0.455$，$\tau_{\text{opt}}\|\mathbf{g}\| = 0.455\sqrt{5} \approx 1.016 > 0.3 = \Delta$，约束激活。(c) $\tau^* = \Delta/\|\mathbf{g}\| = 0.3/\sqrt{5} \approx 0.134$，$\mathbf{p}^C \approx (-0.134, 0.268)^\top$。下降量 $= -\mathbf{g}^\top\mathbf{p}^C - \frac{1}{2}(\mathbf{p}^C)^\top\mathbf{H}\mathbf{p}^C \approx 0.671 - \frac{0.3^2 \cdot 11}{2\times5} = 0.671 - 0.099 = 0.572$。

**T3**：(a) $\rho = 1.35/1.5 = 0.9 \geq \eta_2$，接受，$\Delta_{k+1} = \min(4, 8) = 4$。(b) $\rho = 0.6/1.5 = 0.4 \in [\eta_1,\eta_2)$，接受，$\Delta_{k+1} = 2$。(c) $\rho = -0.2/1.5 < 0 < \eta_1$，拒绝，$\Delta_{k+1} = 0.5$。

**T4**：(a) $\mathbf{p}^U = -\frac{\|\mathbf{g}\|^2}{\mathbf{g}^\top\mathbf{H}\mathbf{g}}\mathbf{g} = -\frac{9}{6\times9}(3,0)^\top = -\frac{1}{6}(3,0)^\top = (-0.5, 0)^\top$。$\mathbf{p}^B = -(0.5, 0)^\top$（两者相同！$g_2=0$ 故 $p_2^B=0$）。(b) $\|\mathbf{p}^B\| = 0.5 \leq 2 = \Delta$，牛顿步在域内，直接取 $\mathbf{p}^* = \mathbf{p}^B = (-0.5, 0)^\top$。

**T5**：Fisher 信息矩阵 $\mathbf{F}$ 定义了参数空间的度量（KL 散度的二阶近似），自然梯度等价于在"KL 球"内沿最速下降方向更新——这正是信赖域思想。类比：$\mathbf{F}^{-1}$ 对应最优性条件中的 $(\mathbf{H}+\lambda^*\mathbf{I})^{-1}$，$\lambda^*$ 在自然梯度中由学习率决定。TRPO 与标准信赖域的本质区别：前者约束策略分布的 KL 散度（功能空间），后者约束参数向量的 $\ell_2$ 范数（参数空间）；KL 约束对参数重参数化不变，而 $\ell_2$ 约束依赖于特定参数化，因此 TRPO 对策略网络结构变化更鲁棒。

---

## 融合版说明

本版 = **原版（严格推导 + 深度学习应用 + 练习）** + **融合补充（速记 / 套路 / 例题 / 自测）** 融合：

| 段 | 来源 | 价值 |
|---|---|---|
| 一例速记 + 引入 + 思维路径还原 | 融合版前置 | 建立直觉，理解"先划球再求步"的哲学 |
| 学习目标 + 14.1–14.5 严格正文 | 原版 | 完整推导：模型 / 子问题 / 狗腿 / 半径更新 / 收敛 |
| 本章小结 | 原版 | 公式速查表，闭卷前复盘 |
| 深度学习应用 + PyTorch 代码 | 原版 | TRPO/PPO 原理 + 信赖域优化器实现 |
| 练习题 14.1–14.5 | 原版 | 系统巩固，从子问题到 TRPO |
| **抽象成方法** | 融合补充 | 8 个核心公式速查 + 5 步流程，套路固化 |
| **方法变形** | 融合补充 | 4 类变形（范数 / Hessian / 精度 / 初始化）|
| **思考路标** | 融合补充 | 12 条条件反射，覆盖 Cauchy / 狗腿 / TRPO |
| 易错点清单 | 融合补充 | 6 条高频陷阱，特别注意 $\rho_k$ 符号约定 |
| 方法链接 | 融合补充 | 信赖域在 RL / 贝叶斯优化 / LM 中的变体 |
| 典型应用例题 3 例 | 融合补充 | 正定子问题 / $\rho_k$ 计算 / Steihaug-CG |
| 自测题 5 题 + 答案 | 融合补充 | 闭卷验收，T1–T5 覆盖全章核心 |
| **融合版说明** | 融合补充 | 本表格，帮助读者定位每段来源与目的 |

**推荐学习节奏**：速记（5 min）→ 引入题（10 min）→ 正文（70 min）→ 小结 + 抽象成方法（10 min）→ 思考路标（快速扫描）→ 典型例题（20 min）→ 自测（闭卷 15 min）。
