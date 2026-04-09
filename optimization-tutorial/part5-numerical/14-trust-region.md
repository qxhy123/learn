# 第14章：信赖域方法

> **前置章节**：第6章（牛顿法与拟牛顿法）、第5章（梯度下降法）、第4章（最优性条件）
>
> **难度**：★★★★☆

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
