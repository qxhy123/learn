# 第6章：牛顿法与拟牛顿法

> **前置章节**：第2章（多元微分学）、第4章（最优性条件）、第5章（梯度下降法）
>
> **难度**：★★★★☆

---

## 学习目标

学完本章，你将能够：

1. **推导牛顿法**：从二阶Taylor展开出发，理解牛顿方向 $\mathbf{d} = -\mathbf{H}^{-1}\nabla f$ 的几何含义与代数推导
2. **分析收敛性**：理解牛顿法的二次收敛率，并与梯度下降的线性收敛率进行对比
3. **掌握修正策略**：了解Hessian不正定时的修正技术，包括Cholesky修正和信赖域框架
4. **理解拟牛顿思想**：掌握BFGS和DFP更新公式的推导逻辑，以及它们如何在不计算真实Hessian的情况下逼近二阶信息
5. **应用L-BFGS**：了解有限内存拟牛顿法的存储优化策略，以及其在深度学习中的潜在价值与局限

---

## 6.1 牛顿法的推导

### 6.1.1 从Taylor展开出发

梯度下降法只利用了目标函数的一阶信息（梯度）。若我们进一步利用二阶信息（Hessian矩阵），能否设计出收敛更快的算法？这正是牛顿法的出发点。

设 $f: \mathbb{R}^n \to \mathbb{R}$ 是二次连续可微函数。在当前点 $\mathbf{x}_k$ 处，对 $f$ 进行二阶Taylor展开：

$$f(\mathbf{x}_k + \mathbf{d}) \approx f(\mathbf{x}_k) + \nabla f(\mathbf{x}_k)^\top \mathbf{d} + \frac{1}{2}\mathbf{d}^\top \nabla^2 f(\mathbf{x}_k) \mathbf{d}$$

记 $\mathbf{g}_k = \nabla f(\mathbf{x}_k)$，$\mathbf{H}_k = \nabla^2 f(\mathbf{x}_k)$。我们希望找到最优的下降方向 $\mathbf{d}$，即最小化上述二次模型：

$$\min_{\mathbf{d}} \; q(\mathbf{d}) = f(\mathbf{x}_k) + \mathbf{g}_k^\top \mathbf{d} + \frac{1}{2}\mathbf{d}^\top \mathbf{H}_k \mathbf{d}$$

对 $\mathbf{d}$ 求导并令其为零：

$$\nabla_{\mathbf{d}} q(\mathbf{d}) = \mathbf{g}_k + \mathbf{H}_k \mathbf{d} = \mathbf{0}$$

若 $\mathbf{H}_k$ 正定（可逆），则唯一解为：

$$\boxed{\mathbf{d}_k^{\text{Newton}} = -\mathbf{H}_k^{-1} \mathbf{g}_k}$$

这就是**牛顿方向**（Newton direction）。

### 6.1.2 牛顿法算法描述

**经典纯牛顿法（Pure Newton's Method）**：

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \mathbf{H}_k^{-1} \mathbf{g}_k$$

**带线搜索的牛顿法（Damped Newton's Method）**：

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{d}_k, \quad \mathbf{d}_k = -\mathbf{H}_k^{-1} \mathbf{g}_k$$

其中步长 $\alpha_k$ 通过线搜索（如Wolfe条件）确定。

**算法框架：**

```
输入：初始点 x_0，容差 ε > 0
输出：近似极小值点 x*

for k = 0, 1, 2, ... do
    计算梯度 g_k = ∇f(x_k)
    if ‖g_k‖ ≤ ε then
        return x_k
    end if
    计算 Hessian H_k = ∇²f(x_k)
    求解线性方程组 H_k d_k = -g_k
    用线搜索确定步长 α_k
    更新 x_{k+1} = x_k + α_k d_k
end for
```

### 6.1.3 几何直觉

牛顿法的直觉：**在当前点用一个二次函数拟合目标函数，然后跳到该二次函数的极小值点**。

对于严格凸二次函数 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{A} \mathbf{x} + \mathbf{b}^\top \mathbf{x} + c$，其中 $\mathbf{A} \succ 0$：

$$\mathbf{x}^* = -\mathbf{A}^{-1}\mathbf{b}$$

$$\mathbf{d}_0 = -\mathbf{A}^{-1}(\mathbf{A}\mathbf{x}_0 + \mathbf{b}) = -\mathbf{x}_0 - \mathbf{A}^{-1}\mathbf{b} = \mathbf{x}^* - \mathbf{x}_0$$

也就是说，**牛顿法对严格凸二次函数一步收敛**。这一性质是牛顿法快速收敛的根本原因。

**与梯度下降的对比：**

| 比较维度 | 梯度下降 | 牛顿法 |
|----------|----------|--------|
| 使用信息 | 一阶（梯度） | 二阶（梯度 + Hessian） |
| 搜索方向 | $-\nabla f$ | $-\mathbf{H}^{-1}\nabla f$ |
| 几何含义 | 最速下降方向 | 二次模型最优方向 |
| 条件数影响 | 受条件数严重影响 | 对条件数不敏感（仿射不变） |
| 每步计算量 | $O(n)$ | $O(n^2)$（Hessian）+ $O(n^3)$（求解线性方程组） |

### 6.1.4 仿射不变性

牛顿法的一个重要性质是**仿射不变性**（affine invariance）：对于可逆线性变换 $\mathbf{x} = \mathbf{T}\mathbf{y}$，若在 $\mathbf{x}$ 空间中用牛顿法，与在 $\mathbf{y}$ 空间中用牛顿法，产生完全相同的迭代轨迹（在适当坐标变换后）。

这说明牛顿法对变量尺度的伸缩不敏感，这正是梯度下降在病态条件（ill-conditioned）问题上表现差而牛顿法表现好的本质原因。

**例题 6.1**：用牛顿法极小化 $f(x_1, x_2) = x_1^2 + 100x_2^2$，从 $\mathbf{x}_0 = (1, 1)^\top$ 出发。

$$\nabla f = \begin{pmatrix} 2x_1 \\ 200x_2 \end{pmatrix}, \quad \mathbf{H} = \begin{pmatrix} 2 & 0 \\ 0 & 200 \end{pmatrix}$$

$$\mathbf{d}_0 = -\mathbf{H}^{-1}\nabla f(\mathbf{x}_0) = -\begin{pmatrix} 1/2 & 0 \\ 0 & 1/200 \end{pmatrix}\begin{pmatrix} 2 \\ 200 \end{pmatrix} = -\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

$$\mathbf{x}_1 = \mathbf{x}_0 + \mathbf{d}_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix} = \mathbf{x}^*$$

**一步收敛！** 而梯度下降在条件数为100的此问题上需要大量步骤才能收敛。

---

## 6.2 牛顿法的收敛性

### 6.2.1 局部二次收敛

牛顿法最显著的特点是**局部二次收敛**（quadratic convergence）。

**定理 6.1（牛顿法的局部收敛）**：设 $f: \mathbb{R}^n \to \mathbb{R}$ 二次连续可微，$\mathbf{x}^*$ 是局部极小值点，$\mathbf{H}^* = \nabla^2 f(\mathbf{x}^*)$ 正定。设 $\mathbf{H}^{-1}$ 在 $\mathbf{x}^*$ 附近满足 Lipschitz 条件：

$$\|\nabla^2 f(\mathbf{x}) - \nabla^2 f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\|$$

则当 $\mathbf{x}_0$ 充分靠近 $\mathbf{x}^*$ 时，纯牛顿法产生的迭代序列满足：

$$\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \leq C \|\mathbf{x}_k - \mathbf{x}^*\|^2$$

其中常数 $C = \frac{L\|(\mathbf{H}^*)^{-1}\|}{2}$。

**证明思路：**

利用 $\nabla f(\mathbf{x}^*) = \mathbf{0}$，对 $\nabla f$ 在 $\mathbf{x}^*$ 处展开：

$$\mathbf{g}_k = \nabla f(\mathbf{x}_k) = \mathbf{H}^*(\mathbf{x}_k - \mathbf{x}^*) + O(\|\mathbf{x}_k - \mathbf{x}^*\|^2)$$

$$\mathbf{H}_k = \mathbf{H}^* + O(\|\mathbf{x}_k - \mathbf{x}^*\|)$$

则：

$$\mathbf{x}_{k+1} - \mathbf{x}^* = \mathbf{x}_k - \mathbf{H}_k^{-1}\mathbf{g}_k - \mathbf{x}^*$$

代入展开式，化简可得误差以平方速率减小，即二次收敛。$\square$

### 6.2.2 收敛阶的比较

| 收敛类型 | 定义 | 代表算法 |
|----------|------|----------|
| 次线性收敛 | $\frac{\|\mathbf{e}_{k+1}\|}{\|\mathbf{e}_k\|} \to 1$ | 次梯度法 |
| 线性收敛 | $\|\mathbf{e}_{k+1}\| \leq r\|\mathbf{e}_k\|$，$0 < r < 1$ | 梯度下降 |
| 超线性收敛 | $\frac{\|\mathbf{e}_{k+1}\|}{\|\mathbf{e}_k\|} \to 0$ | 拟牛顿法 |
| 二次收敛 | $\|\mathbf{e}_{k+1}\| \leq C\|\mathbf{e}_k\|^2$ | 牛顿法 |

**直观感受**：若当前误差为 $\|\mathbf{e}_k\| = 10^{-3}$：
- 线性收敛（$r = 0.9$）：下一步误差 $\approx 9 \times 10^{-4}$，需要约20步到达 $10^{-6}$
- 二次收敛：下一步误差 $\approx 10^{-6}$，再下一步 $\approx 10^{-12}$，**2步内达到机器精度**

这就是为什么工程上常说牛顿法"最后几步非常快"——一旦进入二次收敛的区域，精度爆炸式提升。

### 6.2.3 全局收敛性

纯牛顿法只有局部收敛保证，对于非凸函数或初始点远离极小值时，可能发散或收敛到鞍点。

**阻尼牛顿法（带线搜索）的全局收敛**：若步长满足Wolfe条件，且 $\mathbf{H}_k$ 始终正定有界，则：

$$\lim_{k \to \infty} \|\nabla f(\mathbf{x}_k)\| = 0$$

即迭代序列收敛到驻点（一阶最优性条件满足）。

**牛顿法的三阶段行为：**
1. **远离极小值**（初始阶段）：可能行为不规律，依赖线搜索保证下降
2. **进入收敛域**（中间阶段）：超线性收敛逐渐显现
3. **接近极小值**（最终阶段）：完全二次收敛，精度极速提升

---

## 6.3 修正牛顿法

### 6.3.1 Hessian 不正定的问题

纯牛顿法要求 $\mathbf{H}_k \succ 0$，否则牛顿方向不是下降方向，甚至可能无解。

**问题情形**：
- 初始点远离极小值，Hessian 可能不定（indefinite）甚至负定
- 目标函数非凸，存在鞍点
- 数值计算误差导致Hessian近似不准确

**判断准则**：牛顿方向 $\mathbf{d}$ 是下降方向的充要条件为 $\nabla f^\top \mathbf{d} < 0$，即：

$$\mathbf{g}^\top \mathbf{d} = -\mathbf{g}^\top \mathbf{H}^{-1} \mathbf{g} < 0 \iff \mathbf{H} \succ 0$$

### 6.3.2 Hessian 修正方法

**方法一：加对角正则化（Levenberg-Marquardt 修正）**

$$\mathbf{H}_k^{\text{mod}} = \mathbf{H}_k + \mu_k \mathbf{I}$$

选取足够大的 $\mu_k > 0$ 使 $\mathbf{H}_k^{\text{mod}} \succ 0$，再用修正后的Hessian计算牛顿方向：

$$\mathbf{d}_k = -(\mathbf{H}_k + \mu_k \mathbf{I})^{-1}\mathbf{g}_k$$

自适应选取 $\mu_k$ 的策略：初始取 $\mu_k = 0$，若Cholesky分解失败，则翻倍 $\mu_k$ 直至成功。

**方法二：Cholesky 修正（Modified Cholesky）**

对 $\mathbf{H}_k$ 做修正Cholesky分解 $\mathbf{L}\mathbf{L}^\top = \mathbf{H}_k + \mathbf{E}_k$，其中 $\mathbf{E}_k$ 是最小的使分解成功的扰动。

**方法三：特征值修正**

对 $\mathbf{H}_k$ 做特征值分解 $\mathbf{H}_k = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$，将负特征值替换为小正数：

$$\mathbf{H}_k^{\text{mod}} = \mathbf{Q} \max(\boldsymbol{\Lambda}, \epsilon \mathbf{I}) \mathbf{Q}^\top$$

### 6.3.3 信赖域框架

信赖域方法（Trust Region）是处理Hessian不正定的另一种框架，见第14章详述。其核心思想是：不要求全局最小化二次模型，而是限制步长在一个球形区域 $\|\mathbf{d}\| \leq \Delta$ 内：

$$\min_{\mathbf{d}} \; \mathbf{g}_k^\top \mathbf{d} + \frac{1}{2}\mathbf{d}^\top \mathbf{H}_k \mathbf{d}, \quad \text{s.t. } \|\mathbf{d}\| \leq \Delta_k$$

信赖域半径 $\Delta_k$ 根据二次模型对真实函数的预测质量自适应调整。

### 6.3.4 牛顿法的计算瓶颈

牛顿法的主要计算开销：

| 步骤 | 计算量 | 存储量 |
|------|--------|--------|
| 计算梯度 $\mathbf{g}_k$ | $O(n)$ | $O(n)$ |
| 计算Hessian $\mathbf{H}_k$ | $O(n^2)$ | $O(n^2)$ |
| 求解 $\mathbf{H}_k \mathbf{d} = -\mathbf{g}_k$ | $O(n^3)$ | $O(n^2)$ |
| **总计** | $O(n^3)$ | $O(n^2)$ |

对于深度学习中 $n \sim 10^7$ 的情形，计算和存储都是**不可接受**的。这正是拟牛顿法与L-BFGS被提出的动机。

---

## 6.4 拟牛顿法（BFGS 与 DFP）

### 6.4.1 拟牛顿思想

拟牛顿法（Quasi-Newton Methods）的核心思想：**不显式计算Hessian或其逆，而是用一个矩阵 $\mathbf{B}_k$ 来近似 $\mathbf{H}_k$，或用 $\mathbf{W}_k$ 来近似 $\mathbf{H}_k^{-1}$**，并通过每步迭代的梯度变化信息对该近似进行低秩更新。

**割线条件（Secant Condition）**：

在两个相邻迭代点 $\mathbf{x}_k$ 和 $\mathbf{x}_{k+1}$ 之间，定义：

$$\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k \quad \text{（位移向量）}$$
$$\mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k) \quad \text{（梯度差向量）}$$

由二阶Taylor展开近似：$\nabla^2 f(\mathbf{x}_k) \mathbf{s}_k \approx \mathbf{y}_k$，因此要求近似矩阵满足**割线方程**：

$$\mathbf{B}_{k+1} \mathbf{s}_k = \mathbf{y}_k$$

等价地（对Hessian逆的近似 $\mathbf{W}_{k+1} \approx \mathbf{H}_{k+1}^{-1}$）：

$$\mathbf{W}_{k+1} \mathbf{y}_k = \mathbf{s}_k$$

**正定性保证**：要使 $\mathbf{B}_{k+1}$ 正定，需满足**曲率条件**：

$$\mathbf{s}_k^\top \mathbf{y}_k > 0$$

当线搜索满足强Wolfe条件时，此条件自动成立。

### 6.4.2 DFP 更新公式

Davidon-Fletcher-Powell（DFP）方法直接更新Hessian逆的近似 $\mathbf{W}_k \approx \mathbf{H}_k^{-1}$。

**推导**：在满足割线方程 $\mathbf{W}_{k+1}\mathbf{y}_k = \mathbf{s}_k$ 的所有正定矩阵中，找与 $\mathbf{W}_k$ 最近（加权Frobenius范数意义下）的矩阵：

$$\mathbf{W}_{k+1} = \underset{\mathbf{W} \succ 0, \, \mathbf{W}\mathbf{y}_k = \mathbf{s}_k}{\arg\min} \|\mathbf{W}^{1/2}(\mathbf{W} - \mathbf{W}_k)\mathbf{W}^{1/2}\|_F$$

求解得到**DFP更新公式**：

$$\boxed{\mathbf{W}_{k+1}^{\text{DFP}} = \mathbf{W}_k - \frac{\mathbf{W}_k \mathbf{y}_k \mathbf{y}_k^\top \mathbf{W}_k}{\mathbf{y}_k^\top \mathbf{W}_k \mathbf{y}_k} + \frac{\mathbf{s}_k \mathbf{s}_k^\top}{\mathbf{y}_k^\top \mathbf{s}_k}}$$

DFP更新是秩2更新：从 $\mathbf{W}_k$ 减去一个秩1矩阵，再加上一个秩1矩阵。

### 6.4.3 BFGS 更新公式

Broyden-Fletcher-Goldfarb-Shanno（BFGS）方法更新Hessian近似 $\mathbf{B}_k \approx \mathbf{H}_k$（而非其逆），然后通过Sherman-Morrison-Woodbury公式推导出对应的逆更新。

**Hessian近似的更新**（满足 $\mathbf{B}_{k+1}\mathbf{s}_k = \mathbf{y}_k$）：

$$\mathbf{B}_{k+1}^{\text{BFGS}} = \mathbf{B}_k - \frac{\mathbf{B}_k \mathbf{s}_k \mathbf{s}_k^\top \mathbf{B}_k}{\mathbf{s}_k^\top \mathbf{B}_k \mathbf{s}_k} + \frac{\mathbf{y}_k \mathbf{y}_k^\top}{\mathbf{y}_k^\top \mathbf{s}_k}$$

通过Sherman-Morrison-Woodbury公式 $(\mathbf{A} + \mathbf{U}\mathbf{V}^\top)^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(I + \mathbf{V}^\top \mathbf{A}^{-1}\mathbf{U})^{-1}\mathbf{V}^\top \mathbf{A}^{-1}$，可推导出Hessian逆的更新。

记 $\rho_k = \frac{1}{\mathbf{y}_k^\top \mathbf{s}_k}$，**BFGS对Hessian逆的更新**：

$$\boxed{\mathbf{W}_{k+1}^{\text{BFGS}} = \left(\mathbf{I} - \rho_k \mathbf{s}_k \mathbf{y}_k^\top\right) \mathbf{W}_k \left(\mathbf{I} - \rho_k \mathbf{y}_k \mathbf{s}_k^\top\right) + \rho_k \mathbf{s}_k \mathbf{s}_k^\top}$$

这是一种**秩2更新**，计算量 $O(n^2)$，远低于重新计算和求逆Hessian的 $O(n^3)$。

### 6.4.4 BFGS 算法流程

```
输入：初始点 x_0，初始逆Hessian近似 W_0 = I，容差 ε > 0
输出：近似极小值点 x*

for k = 0, 1, 2, ... do
    计算梯度 g_k = ∇f(x_k)
    if ‖g_k‖ ≤ ε then return x_k

    计算搜索方向 d_k = -W_k g_k
    用满足强Wolfe条件的线搜索确定步长 α_k

    更新点：x_{k+1} = x_k + α_k d_k
    计算 s_k = x_{k+1} - x_k = α_k d_k
    计算 y_k = ∇f(x_{k+1}) - ∇f(x_k)

    if s_k^T y_k > 0 then
        ρ_k = 1 / (y_k^T s_k)
        W_{k+1} = (I - ρ_k s_k y_k^T) W_k (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T
    else
        W_{k+1} = W_k  # 曲率条件不满足，跳过更新
    end if
end for
```

### 6.4.5 BFGS 与 DFP 的关系

BFGS 和 DFP 是互为对偶的：对 $\mathbf{B}$ 的更新（DFP用于 $\mathbf{W}$，BFGS用于 $\mathbf{B}$）通过交换 $\mathbf{s} \leftrightarrow \mathbf{y}$，$\mathbf{W} \leftrightarrow \mathbf{B}$ 可以从一个推导出另一个。

实践中，**BFGS 通常比 DFP 更稳健**，对不精确线搜索的鲁棒性更好，这使得BFGS成为最常用的拟牛顿方法。

**BFGS 的收敛性质**：
- 全局收敛（结合Wolfe线搜索）
- 局部超线性收敛（q-超线性）
- 对强凸函数：超线性收敛速率 $\|\mathbf{x}_{k+1} - \mathbf{x}^*\| = o(\|\mathbf{x}_k - \mathbf{x}^*\|)$

**例题 6.2**：用BFGS法极小化 $f(x_1, x_2) = (x_1 - 1)^2 + 10(x_2 - x_1^2)^2$（Rosenbrock函数），从 $\mathbf{x}_0 = (-1, 1)^\top$ 出发。

Rosenbrock函数是优化算法的经典测试案例，其极小值在 $\mathbf{x}^* = (1, 1)^\top$，$f^* = 0$。梯度下降在此函数上收敛极慢（需要数千步），而BFGS通常在数十步内收敛，体现了二阶方法的优越性。

---

## 6.5 有限内存拟牛顿法（L-BFGS）

### 6.5.1 标准BFGS的存储问题

BFGS算法的瓶颈：需要存储和更新 $n \times n$ 的矩阵 $\mathbf{W}_k$。

| 参数规模 | 矩阵存储量 | 每步矩阵更新计算量 |
|----------|------------|-------------------|
| $n = 1000$ | $\sim 8$ MB | $\sim 10^6$ 次浮点运算 |
| $n = 10^6$ | $\sim 8$ TB | $\sim 10^{12}$ 次浮点运算 |
| $n = 10^8$ | 不可行 | 不可行 |

对于大规模机器学习问题（参数量 $n \sim 10^7 \sim 10^9$），标准BFGS完全不可行。

### 6.5.2 L-BFGS 的核心思想

**有限内存BFGS（L-BFGS，Limited-memory BFGS）** 的关键洞察：

> 不存储完整的 $n \times n$ 矩阵 $\mathbf{W}_k$，而是只存储最近 $m$ 步的向量对 $\{(\mathbf{s}_i, \mathbf{y}_i)\}_{i=k-m}^{k-1}$，用这 $m$ 对向量隐式表示 $\mathbf{W}_k$，并利用它们高效计算矩阵-向量乘积 $\mathbf{W}_k \mathbf{g}_k$。

存储量：$O(mn)$，而非 $O(n^2)$。典型取 $m = 5 \sim 20$。

### 6.5.3 双循环递推（Two-Loop Recursion）

L-BFGS 计算方向 $\mathbf{d} = -\mathbf{W}_k \mathbf{g}_k$ 的高效算法：

**算法（L-BFGS Two-Loop Recursion）：**

```
输入：当前梯度 g = g_k，存储的向量对 {(s_i, y_i)}，i = k-m,...,k-1
      初始逆Hessian近似 W_0^k（通常取标量矩阵）
输出：搜索方向 d

q ← g

# 第一个循环（从最新到最旧）
for i = k-1, k-2, ..., k-m do
    ρ_i = 1 / (y_i^T s_i)
    α_i = ρ_i * (s_i^T q)
    q ← q - α_i * y_i
end for

r ← W_0^k * q   # 初始Hessian逆作用在 q 上（标量乘法）

# 第二个循环（从最旧到最新）
for i = k-m, k-m+1, ..., k-1 do
    β_i = ρ_i * (y_i^T r)
    r ← r + s_i * (α_i - β_i)
end for

d ← -r   # 搜索方向
```

**计算量**：$O(mn)$，仅需 $2m$ 次向量内积和 $2m$ 次 saxpy（向量加法）运算。

### 6.5.4 初始逆Hessian近似的选取

双循环中的初始近似 $\mathbf{W}_0^k$ 通常取标量矩阵：

$$\mathbf{W}_0^k = \gamma_k \mathbf{I}, \quad \gamma_k = \frac{\mathbf{s}_{k-1}^\top \mathbf{y}_{k-1}}{\mathbf{y}_{k-1}^\top \mathbf{y}_{k-1}}$$

这是用最近一步的曲率信息来估计Hessian的"尺度"，比始终用 $\mathbf{I}$ 效果好很多。

### 6.5.5 L-BFGS 的实际表现

- **内存效率**：存储 $O(mn)$，对大规模问题可行
- **计算效率**：每步 $O(mn)$，适合大规模优化
- **收敛速度**：超线性收敛（接近BFGS，当 $m$ 足够大时）
- **实践中**：$m = 10$ 通常足够，$m = 20$ 性能接近完整BFGS

L-BFGS 是解决**大规模无约束优化**（包括带简单约束的L-BFGS-B变体）的首选算法之一，被广泛应用于统计机器学习、科学计算、图像处理等领域。

**各方法性能对比（典型中等规模问题 $n \sim 10^4$）：**

| 方法 | 每步计算量 | 存储量 | 收敛速率 | 适用规模 |
|------|------------|--------|----------|----------|
| 梯度下降 | $O(n)$ | $O(n)$ | 线性 | 任意 |
| 共轭梯度 | $O(n)$ | $O(n)$ | 超线性（CG重启后） | 大规模 |
| L-BFGS | $O(mn)$ | $O(mn)$ | 超线性 | 大规模 |
| BFGS | $O(n^2)$ | $O(n^2)$ | 超线性 | 中小规模 |
| 牛顿法 | $O(n^3)$ | $O(n^2)$ | 二次 | 小规模 |

---

## 本章小结

| 方法 | 更新规则 | 收敛速率 | 每步代价 | 存储代价 | 关键要求 |
|------|----------|----------|----------|----------|----------|
| 经典牛顿法 | $\mathbf{x}_{k+1} = \mathbf{x}_k - \mathbf{H}_k^{-1}\mathbf{g}_k$ | 二次（局部） | $O(n^3)$ | $O(n^2)$ | $\mathbf{H}_k \succ 0$ |
| 修正牛顿法 | $\mathbf{d}_k = -(\mathbf{H}_k + \mu_k\mathbf{I})^{-1}\mathbf{g}_k$ | 二次（局部） | $O(n^3)$ | $O(n^2)$ | 线搜索 |
| DFP | $\mathbf{W}_{k+1} = \mathbf{W}_k - \frac{\mathbf{W}_k\mathbf{y}\mathbf{y}^\top\mathbf{W}_k}{\mathbf{y}^\top\mathbf{W}_k\mathbf{y}} + \frac{\mathbf{s}\mathbf{s}^\top}{\mathbf{y}^\top\mathbf{s}}$ | 超线性 | $O(n^2)$ | $O(n^2)$ | $\mathbf{s}^\top\mathbf{y} > 0$ |
| BFGS | $\mathbf{W}_{k+1} = (\mathbf{I}-\rho\mathbf{s}\mathbf{y}^\top)\mathbf{W}_k(\mathbf{I}-\rho\mathbf{y}\mathbf{s}^\top) + \rho\mathbf{s}\mathbf{s}^\top$ | 超线性 | $O(n^2)$ | $O(n^2)$ | Wolfe条件 |
| L-BFGS | 隐式 $m$ 对向量存储 | 超线性 | $O(mn)$ | $O(mn)$ | $m \geq 5$ |

**核心公式速查：**

$$\text{牛顿方向：}\mathbf{d}_k = -\mathbf{H}_k^{-1}\mathbf{g}_k$$

$$\text{割线方程：}\mathbf{B}_{k+1}\mathbf{s}_k = \mathbf{y}_k, \quad \mathbf{W}_{k+1}\mathbf{y}_k = \mathbf{s}_k$$

$$\text{曲率条件：}\mathbf{s}_k^\top\mathbf{y}_k > 0$$

$$\rho_k = \frac{1}{\mathbf{y}_k^\top\mathbf{s}_k}, \quad \mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k, \quad \mathbf{y}_k = \mathbf{g}_{k+1} - \mathbf{g}_k$$

---

## 深度学习应用：二阶优化器与 K-FAC

### 深度学习中二阶方法的挑战

深度神经网络的参数规模通常为 $n \sim 10^7 \sim 10^{11}$，这给二阶方法带来了根本性挑战：

| 挑战 | 具体问题 | 典型规模 |
|------|----------|----------|
| Hessian存储 | $n^2$ 个浮点数 | $10^{14}$ 字节 ≈ 不可行 |
| Hessian计算 | 反向传播计算Hessian列 | $O(n^2)$ 次前向传播 |
| 线性方程组求解 | $O(n^3)$ | 无法接受 |
| 随机噪声 | Mini-batch梯度有噪声 | 二阶信息更不准确 |
| 非凸性 | 大量鞍点、负曲率方向 | 需要特殊处理 |

尽管如此，二阶方法在深度学习中仍有重要价值，催生了多种近似方案。

### Hessian-Free 优化

Hessian-free（HF）方法避免显式计算Hessian，利用**共轭梯度法**求解牛顿方程 $\mathbf{H}\mathbf{d} = -\mathbf{g}$，每步CG迭代只需计算**Hessian-向量乘积** $\mathbf{H}\mathbf{v}$（可通过两次反向传播在 $O(n)$ 时间内完成，见第15章）。

### K-FAC 简介

**Kronecker-Factored Approximate Curvature（K-FAC）** 是目前最成功的深度学习二阶方法之一，由 Martens & Grosse（2015）提出。

**核心思想**：

对于神经网络，Fisher信息矩阵（作为Hessian的正半定近似，见第23章）具有特殊的块结构。对于全连接层的参数 $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，K-FAC将对应的Fisher块近似为两个矩阵的Kronecker积：

$$\mathbf{F}_{\text{layer}} \approx \mathbf{A} \otimes \mathbf{G}$$

其中：
- $\mathbf{A} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$：输入激活值的协方差矩阵
- $\mathbf{G} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}}$：输出梯度的协方差矩阵

**为何有效**：Kronecker积的逆可以高效计算：

$$(\mathbf{A} \otimes \mathbf{G})^{-1} = \mathbf{A}^{-1} \otimes \mathbf{G}^{-1}$$

将 $O(n^2)$ 的矩阵逆问题分解为两个 $O(d^2)$ 的小矩阵逆问题！

**K-FAC 更新规则（直觉）**：

$$\mathbf{W} \leftarrow \mathbf{W} - \alpha \cdot \mathbf{G}^{-1} \nabla_{\mathbf{W}} f \cdot \mathbf{A}^{-1}$$

其中矩阵乘法从两侧"预处理"了梯度，相当于在参数空间的自然几何度量下做梯度下降。

### PyTorch 代码：L-BFGS 优化器应用

PyTorch 内置了 `torch.optim.LBFGS`，常用于全批量（full-batch）优化，例如神经网络的精细调优或物理信息神经网络（PINN）。

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 示例 1：用 L-BFGS 拟合 Rosenbrock 函数
# ============================================================

def rosenbrock(x):
    """Rosenbrock 函数：f(x,y) = (1-x)^2 + 100*(y-x^2)^2，极小值在 (1,1)"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# 初始点
x = torch.tensor([-1.0, 1.0], requires_grad=True)

# L-BFGS 优化器
optimizer = torch.optim.LBFGS(
    [x],
    lr=1.0,
    max_iter=20,          # 每次 step() 调用内的最大迭代次数
    max_eval=None,        # 每次 step() 的最大函数评估次数
    tolerance_grad=1e-9,
    tolerance_change=1e-12,
    history_size=10,      # m：存储的向量对数
    line_search_fn='strong_wolfe'  # 使用强 Wolfe 线搜索
)

losses = []
print("L-BFGS 优化 Rosenbrock 函数")
print(f"初始点: {x.detach().numpy()}, f = {rosenbrock(x).item():.6f}")

for step in range(30):
    def closure():
        """L-BFGS 需要闭包（closure）来多次评估函数值"""
        optimizer.zero_grad()
        loss = rosenbrock(x)
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    losses.append(loss.item())

    if step % 5 == 0:
        print(f"Step {step:3d}: x = {x.detach().numpy()}, f = {loss.item():.2e}")

print(f"\n最终点: {x.detach().numpy()}")
print(f"最终值: {losses[-1]:.2e}")
print(f"理论最优: x* = (1, 1), f* = 0")


# ============================================================
# 示例 2：L-BFGS vs SGD 在简单神经网络上的对比
# ============================================================

torch.manual_seed(42)

# 生成训练数据：学习 sin(x)
X_train = torch.linspace(-3, 3, 100).unsqueeze(1)
y_train = torch.sin(X_train)

# 定义简单网络
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_lbfgs(model, X, y, max_steps=100):
    """用 L-BFGS 训练（适合全批量）"""
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        history_size=10,
        line_search_fn='strong_wolfe'
    )
    criterion = nn.MSELoss()

    loss_history = []

    for step in range(max_steps):
        def closure():
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_history.append(loss.item())

    return loss_history

def train_adam(model, X, y, lr=1e-3, max_steps=2000):
    """用 Adam 训练（对比基准）"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []

    for step in range(max_steps):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    return loss_history

# 训练
net_lbfgs = SimpleNet()
net_adam  = SimpleNet()

# 用相同初始化
state = net_lbfgs.state_dict()
net_adam.load_state_dict({k: v.clone() for k, v in state.items()})

print("\n训练对比：L-BFGS vs Adam")
loss_lbfgs = train_lbfgs(net_lbfgs, X_train, y_train, max_steps=100)
loss_adam  = train_adam(net_adam,  X_train, y_train, lr=1e-3, max_steps=2000)

print(f"L-BFGS 100步最终损失:  {loss_lbfgs[-1]:.2e}")
print(f"Adam  2000步最终损失: {loss_adam[-1]:.2e}")

# 注意：L-BFGS 100步可达到 Adam 2000步甚至更好的精度
# 但 L-BFGS 每步代价更高（全批量，多次函数评估）


# ============================================================
# 示例 3：手动实现 BFGS（教学用，小规模）
# ============================================================

import numpy as np
from scipy.optimize import line_search

def bfgs_demo(f, grad_f, x0, tol=1e-8, max_iter=1000):
    """
    简单的 BFGS 实现（教学演示版本）

    参数:
        f:       目标函数 R^n -> R
        grad_f:  梯度函数 R^n -> R^n
        x0:      初始点
        tol:     梯度范数容差
        max_iter: 最大迭代次数
    """
    n = len(x0)
    x = x0.copy()
    W = np.eye(n)   # 初始逆Hessian近似为单位矩阵

    history = {'x': [x.copy()], 'f': [f(x)], 'grad_norm': []}

    for k in range(max_iter):
        g = grad_f(x)
        grad_norm = np.linalg.norm(g)
        history['grad_norm'].append(grad_norm)

        if grad_norm < tol:
            print(f"收敛于第 {k} 步，梯度范数 = {grad_norm:.2e}")
            break

        # BFGS 搜索方向
        d = -W @ g

        # Wolfe 线搜索（使用 scipy）
        alpha, _, _, new_f, _, new_g = line_search(
            f, grad_f, x, d,
            c1=1e-4, c2=0.9
        )
        if alpha is None:
            alpha = 1e-4  # 退化为小步长

        # 更新
        s = alpha * d
        x_new = x + s
        y = grad_f(x_new) - g

        # BFGS 更新（仅当曲率条件满足时）
        sy = s @ y
        if sy > 1e-10:
            rho = 1.0 / sy
            I = np.eye(n)
            A = I - rho * np.outer(s, y)
            B = I - rho * np.outer(y, s)
            W = A @ W @ B + rho * np.outer(s, s)

        x = x_new
        history['x'].append(x.copy())
        history['f'].append(f(x))

    return x, history

# 测试 BFGS 在 Rosenbrock 函数上
def f_rosen(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_rosen(x):
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
         200*(x[1] - x[0]**2)
    ])

x0 = np.array([-1.0, 1.0])
x_opt, hist = bfgs_demo(f_rosen, grad_rosen, x0)

print(f"\nBFGS 结果：")
print(f"  最优点: {x_opt}")
print(f"  最优值: {f_rosen(x_opt):.2e}")
print(f"  迭代次数: {len(hist['f'])}")


# ============================================================
# 示例 4：K-FAC 思想演示（极简版，全连接层）
# ============================================================

class KFACLayer:
    """
    单层全连接网络的 K-FAC 近似曲率演示。

    对于权重 W (d_out x d_in)，Fisher 矩阵近似为：
        F ≈ A ⊗ G
    其中：
        A = E[a a^T]  (输入激活协方差，d_in x d_in)
        G = E[g g^T]  (输出梯度协方差，d_out x d_out)

    自然梯度更新：
        ΔW = G^{-1} ∇_W f A^{-1}
    """

    def __init__(self, d_in, d_out, damping=1e-3):
        self.d_in  = d_in
        self.d_out = d_out
        self.damping = damping

        # 累积统计量
        self.A = np.zeros((d_in,  d_in))    # 输入协方差
        self.G = np.zeros((d_out, d_out))   # 梯度协方差
        self.n_samples = 0

    def update_statistics(self, a, g):
        """
        更新 K-FAC 统计量。

        参数:
            a: 输入激活值向量 (d_in,)
            g: 对输出的梯度向量 (d_out,)
        """
        self.A += np.outer(a, a)
        self.G += np.outer(g, g)
        self.n_samples += 1

    def compute_natural_gradient(self, grad_W):
        """
        用 K-FAC 近似计算自然梯度：G^{-1} @ grad_W @ A^{-1}

        参数:
            grad_W: 标准梯度 (d_out x d_in)
        返回:
            natural_grad: 自然梯度 (d_out x d_in)
        """
        if self.n_samples == 0:
            return grad_W

        # 归一化
        A_mean = self.A / self.n_samples
        G_mean = self.G / self.n_samples

        # 添加阻尼项确保正定
        A_damp = A_mean + self.damping * np.eye(self.d_in)
        G_damp = G_mean + self.damping * np.eye(self.d_out)

        # 计算各自的逆
        A_inv = np.linalg.inv(A_damp)
        G_inv = np.linalg.inv(G_damp)

        # 自然梯度：G^{-1} @ grad_W @ A^{-1}
        return G_inv @ grad_W @ A_inv

# 演示 K-FAC
np.random.seed(42)
d_in, d_out = 4, 3

kfac = KFACLayer(d_in, d_out, damping=1e-2)

# 模拟积累统计量（实际中在训练过程中积累）
for _ in range(100):
    a = np.random.randn(d_in)      # 输入激活
    g = np.random.randn(d_out)     # 输出梯度（由链式法则传来）
    kfac.update_statistics(a, g)

# 对比标准梯度与自然梯度
grad_W = np.random.randn(d_out, d_in)   # 模拟标准梯度
nat_grad = kfac.compute_natural_gradient(grad_W)

print("\nK-FAC 自然梯度演示：")
print(f"标准梯度 Frobenius 范数:  {np.linalg.norm(grad_W):.4f}")
print(f"自然梯度 Frobenius 范数: {np.linalg.norm(nat_grad):.4f}")
print(f"K-FAC 对不同方向的曲率有自适应调节效果")

# 说明：在曲率大的方向（梯度方差大）步长会被缩小
# 在曲率小的方向（梯度方差小）步长会被放大
# 这正是二阶方法相比一阶方法的优势所在
```

### 何时在深度学习中使用二阶方法

**推荐使用二阶方法（L-BFGS/K-FAC）的场景**：
- **小批量或全批量训练**：梯度噪声小，二阶信息可靠
- **物理信息神经网络（PINN）**：全批量训练，L-BFGS 效果显著
- **迁移学习的最后阶段调优**：参数已接近收敛域，二次收敛优势明显
- **科学计算中的优化问题**：有精确函数值和梯度的场景
- **小型网络**：参数量 $n \lesssim 10^5$，BFGS 存储可接受

**不推荐使用二阶方法的场景**：
- 超大型语言模型（$n \sim 10^{10}$）：即使L-BFGS存储也困难
- 高度随机（large mini-batch variance）训练：二阶信息不可靠
- 需要分布式训练：矩阵运算难以并行

---

## 练习题

**练习 6.1**（基础）设 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{A}\mathbf{x} - \mathbf{b}^\top \mathbf{x}$，其中 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 对称正定。

（a）写出 $f$ 的梯度和Hessian矩阵。

（b）从任意初始点 $\mathbf{x}_0$ 出发，证明纯牛顿法一步收敛到 $\mathbf{x}^* = \mathbf{A}^{-1}\mathbf{b}$。

（c）若用步长为 $\alpha$ 的梯度下降法，收敛条件和收敛速率是什么？与牛顿法对比。

---

**练习 6.2**（基础）设当前点 $\mathbf{x}_k$ 处的梯度为 $\mathbf{g}_k = (3, 4)^\top$，Hessian矩阵为 $\mathbf{H}_k = \begin{pmatrix} 5 & 1 \\ 1 & 3 \end{pmatrix}$。

（a）计算牛顿方向 $\mathbf{d}_k = -\mathbf{H}_k^{-1}\mathbf{g}_k$。

（b）验证 $\mathbf{d}_k$ 是下降方向（计算 $\mathbf{g}_k^\top \mathbf{d}_k < 0$）。

（c）若改用梯度方向 $-\mathbf{g}_k$，比较两个方向的范数与在各自方向上的函数减小量（用二次近似）。

---

**练习 6.3**（中级）推导 BFGS 更新公式。

（a）写出BFGS对Hessian近似 $\mathbf{B}_{k+1}$ 的最优化问题：在满足割线方程的对称正定矩阵中，找与 $\mathbf{B}_k$ 最近的矩阵（加权Frobenius范数）。

（b）验证以下秩2更新满足割线方程 $\mathbf{B}_{k+1}\mathbf{s}_k = \mathbf{y}_k$：
$$\mathbf{B}_{k+1} = \mathbf{B}_k - \frac{\mathbf{B}_k\mathbf{s}_k\mathbf{s}_k^\top\mathbf{B}_k}{\mathbf{s}_k^\top\mathbf{B}_k\mathbf{s}_k} + \frac{\mathbf{y}_k\mathbf{y}_k^\top}{\mathbf{y}_k^\top\mathbf{s}_k}$$

（c）若 $\mathbf{B}_k \succ 0$ 且 $\mathbf{s}_k^\top\mathbf{y}_k > 0$，证明 $\mathbf{B}_{k+1} \succ 0$。（提示：用Sherman-Morrison公式或直接验证 $\mathbf{v}^\top\mathbf{B}_{k+1}\mathbf{v} > 0$）

---

**练习 6.4**（中级）L-BFGS 的双循环算法。

设 $m = 2$，当前存储了两对向量：

$$\mathbf{s}_0 = (1, 0)^\top, \quad \mathbf{y}_0 = (2, 1)^\top$$
$$\mathbf{s}_1 = (0, 1)^\top, \quad \mathbf{y}_1 = (1, 3)^\top$$

初始逆Hessian近似取 $\mathbf{W}_0 = \gamma \mathbf{I}$，其中 $\gamma = \frac{\mathbf{s}_1^\top\mathbf{y}_1}{\mathbf{y}_1^\top\mathbf{y}_1}$。

当前梯度为 $\mathbf{g} = (1, 2)^\top$，用双循环递推算法手动计算搜索方向 $\mathbf{d} = -\mathbf{W}_k\mathbf{g}$。

---

**练习 6.5**（提高）牛顿法的收敛分析与K-FAC的变体。

（a）**收敛域分析**：设 $f(x) = x^4 - 4x^2$（双阱势函数），极小值点为 $x^* = \pm\sqrt{2}$。
  - 写出纯牛顿法的迭代格式
  - 分析从 $x_0 = 0.1$ 出发，纯牛顿法收敛到哪个极小值？从 $x_0 = 0$ 出发呢？为什么？
  - 数值模拟验证（或手算前3步）

（b）**K-FAC 的维度分析**：对于有 $L$ 层的全连接网络，每层 $d \times d$ 的权重矩阵。
  - 标准 Fisher 矩阵的存储量是多少？
  - K-FAC 近似后的存储量是多少？
  - 若 $L = 10$，$d = 1000$，计算两者的具体数值并说明 K-FAC 的必要性。

---

## 练习答案

**答案 6.1**

（a）梯度与Hessian：

$$\nabla f(\mathbf{x}) = \mathbf{A}\mathbf{x} - \mathbf{b}, \quad \nabla^2 f(\mathbf{x}) = \mathbf{A}$$

（b）一步收敛证明：

$$\mathbf{d}_0 = -\mathbf{A}^{-1}\nabla f(\mathbf{x}_0) = -\mathbf{A}^{-1}(\mathbf{A}\mathbf{x}_0 - \mathbf{b}) = -\mathbf{x}_0 + \mathbf{A}^{-1}\mathbf{b}$$

$$\mathbf{x}_1 = \mathbf{x}_0 + \mathbf{d}_0 = \mathbf{A}^{-1}\mathbf{b} = \mathbf{x}^*$$

由于 $f$ 是二次函数，牛顿法的二次近似是精确的，因此一步直接跳到精确极小值。$\square$

（c）梯度下降收敛分析：

梯度下降迭代：$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha(\mathbf{A}\mathbf{x}_k - \mathbf{b})$

收敛条件：$0 < \alpha < \frac{2}{\lambda_{\max}(\mathbf{A})}$

最优步长：$\alpha^* = \frac{2}{\lambda_{\max}(\mathbf{A}) + \lambda_{\min}(\mathbf{A})}$

收敛速率：$\|\mathbf{x}_{k+1} - \mathbf{x}^*\|_\mathbf{A} \leq \frac{\kappa - 1}{\kappa + 1}\|\mathbf{x}_k - \mathbf{x}^*\|_\mathbf{A}$

其中 $\kappa = \lambda_{\max}/\lambda_{\min}$ 为条件数。当 $\kappa \gg 1$ 时（病态矩阵），梯度下降极慢，而牛顿法始终一步收敛。

---

**答案 6.2**

（a）计算 $\mathbf{H}_k^{-1}$：

$$\det(\mathbf{H}_k) = 5 \times 3 - 1 \times 1 = 14$$

$$\mathbf{H}_k^{-1} = \frac{1}{14}\begin{pmatrix} 3 & -1 \\ -1 & 5 \end{pmatrix}$$

牛顿方向：

$$\mathbf{d}_k = -\mathbf{H}_k^{-1}\mathbf{g}_k = -\frac{1}{14}\begin{pmatrix} 3 & -1 \\ -1 & 5 \end{pmatrix}\begin{pmatrix} 3 \\ 4 \end{pmatrix} = -\frac{1}{14}\begin{pmatrix} 5 \\ 17 \end{pmatrix} = \begin{pmatrix} -5/14 \\ -17/14 \end{pmatrix}$$

（b）下降性验证：

$$\mathbf{g}_k^\top \mathbf{d}_k = 3 \cdot (-5/14) + 4 \cdot (-17/14) = \frac{-15 - 68}{14} = -\frac{83}{14} < 0 \checkmark$$

（c）比较两个方向：

梯度方向 $\mathbf{p} = -\mathbf{g}_k = (-3, -4)^\top$，$\|\mathbf{p}\| = 5$

牛顿方向范数 $\|\mathbf{d}_k\| = \frac{1}{14}\sqrt{25 + 289} = \frac{\sqrt{314}}{14} \approx 1.267$

二次近似函数减小量（步长=1时）：

- 梯度方向：$\mathbf{g}_k^\top \mathbf{p} + \frac{1}{2}\mathbf{p}^\top\mathbf{H}_k\mathbf{p} = -25 + \frac{1}{2}(9 \times 5 + 2 \times 3 \times (-4) + 16 \times 3) = -25 + \frac{1}{2}(93) = 21.5$（增加！说明步长=1对梯度下降过大）

- 牛顿方向（步长=1为最优）：$-\frac{1}{2}\mathbf{g}_k^\top \mathbf{H}_k^{-1}\mathbf{g}_k = -\frac{83}{28} \approx -2.96$（减小）

这说明牛顿方向选择了二次模型的最优步长和方向。

---

**答案 6.3**

（a）BFGS最优化问题（对Hessian逆 $\mathbf{W}$，DFP目标）：

在所有满足 $\mathbf{W}\mathbf{y}_k = \mathbf{s}_k$，$\mathbf{W} = \mathbf{W}^\top$，$\mathbf{W} \succ 0$ 的矩阵中，最小化加权Frobenius距离：

$$\min_{\mathbf{W}} \|\mathbf{W}_k^{-1/2}(\mathbf{W} - \mathbf{W}_k)\mathbf{W}_k^{-1/2}\|_F^2$$

BFGS 是对 $\mathbf{B}$ 的对应问题：

$$\min_{\mathbf{B}} \|\mathbf{B}_k^{-1/2}(\mathbf{B} - \mathbf{B}_k)\mathbf{B}_k^{-1/2}\|_F^2, \quad \text{s.t. } \mathbf{B}\mathbf{s}_k = \mathbf{y}_k, \, \mathbf{B} \succ 0, \, \mathbf{B} = \mathbf{B}^\top$$

（b）验证割线方程：

令 $\mathbf{B}_{k+1}$ 为给定公式，计算 $\mathbf{B}_{k+1}\mathbf{s}_k$：

第一项：$\mathbf{B}_k \mathbf{s}_k$

第二项：$-\frac{\mathbf{B}_k\mathbf{s}_k\mathbf{s}_k^\top\mathbf{B}_k}{\mathbf{s}_k^\top\mathbf{B}_k\mathbf{s}_k}\mathbf{s}_k = -\frac{\mathbf{B}_k\mathbf{s}_k(\mathbf{s}_k^\top\mathbf{B}_k\mathbf{s}_k)}{\mathbf{s}_k^\top\mathbf{B}_k\mathbf{s}_k} = -\mathbf{B}_k\mathbf{s}_k$

第三项：$\frac{\mathbf{y}_k\mathbf{y}_k^\top}{\mathbf{y}_k^\top\mathbf{s}_k}\mathbf{s}_k = \frac{\mathbf{y}_k(\mathbf{y}_k^\top\mathbf{s}_k)}{\mathbf{y}_k^\top\mathbf{s}_k} = \mathbf{y}_k$

合计：$\mathbf{B}_k\mathbf{s}_k - \mathbf{B}_k\mathbf{s}_k + \mathbf{y}_k = \mathbf{y}_k$ $\checkmark$

（c）正定性证明：

对任意非零 $\mathbf{v} \in \mathbb{R}^n$，设 $\mathbf{u} = \mathbf{B}_k^{1/2}\mathbf{v}$：

$$\mathbf{v}^\top\mathbf{B}_{k+1}\mathbf{v} = \mathbf{v}^\top\mathbf{B}_k\mathbf{v} - \frac{(\mathbf{v}^\top\mathbf{B}_k\mathbf{s}_k)^2}{\mathbf{s}_k^\top\mathbf{B}_k\mathbf{s}_k} + \frac{(\mathbf{v}^\top\mathbf{y}_k)^2}{\mathbf{y}_k^\top\mathbf{s}_k}$$

前两项之差由 Cauchy-Schwarz 不等式知 $\geq 0$，最后一项当 $\mathbf{s}_k^\top\mathbf{y}_k > 0$ 时 $> 0$（除非 $\mathbf{v}^\top\mathbf{y}_k = 0$）。

若 $\mathbf{v}^\top\mathbf{y}_k = 0$，则需更细致的分析：由 $\mathbf{B}_k \succ 0$ 可得前两项严格正定（在 $\mathbf{y}_k$ 的正交补上）。综合可证 $\mathbf{B}_{k+1} \succ 0$。$\square$

---

**答案 6.4**

首先计算 $\rho$ 值：

$$\rho_0 = \frac{1}{\mathbf{s}_0^\top\mathbf{y}_0} = \frac{1}{1\cdot2 + 0\cdot1} = \frac{1}{2}$$

$$\rho_1 = \frac{1}{\mathbf{s}_1^\top\mathbf{y}_1} = \frac{1}{0\cdot1 + 1\cdot3} = \frac{1}{3}$$

初始尺度因子：

$$\gamma = \frac{\mathbf{s}_1^\top\mathbf{y}_1}{\mathbf{y}_1^\top\mathbf{y}_1} = \frac{3}{1+9} = \frac{3}{10}$$

**第一循环**（从最新 $i=1$ 到最旧 $i=0$，$\mathbf{q} \leftarrow \mathbf{g} = (1,2)^\top$）：

$i = 1$：$\alpha_1 = \rho_1(\mathbf{s}_1^\top\mathbf{q}) = \frac{1}{3}(0\cdot1 + 1\cdot2) = \frac{2}{3}$

$$\mathbf{q} \leftarrow \mathbf{q} - \alpha_1\mathbf{y}_1 = \begin{pmatrix}1\\2\end{pmatrix} - \frac{2}{3}\begin{pmatrix}1\\3\end{pmatrix} = \begin{pmatrix}1/3\\0\end{pmatrix}$$

$i = 0$：$\alpha_0 = \rho_0(\mathbf{s}_0^\top\mathbf{q}) = \frac{1}{2}(1\cdot\frac{1}{3} + 0\cdot0) = \frac{1}{6}$

$$\mathbf{q} \leftarrow \mathbf{q} - \alpha_0\mathbf{y}_0 = \begin{pmatrix}1/3\\0\end{pmatrix} - \frac{1}{6}\begin{pmatrix}2\\1\end{pmatrix} = \begin{pmatrix}0\\-1/6\end{pmatrix}$$

**初始近似作用**：

$$\mathbf{r} \leftarrow \gamma\mathbf{q} = \frac{3}{10}\begin{pmatrix}0\\-1/6\end{pmatrix} = \begin{pmatrix}0\\-1/20\end{pmatrix}$$

**第二循环**（从最旧 $i=0$ 到最新 $i=1$）：

$i = 0$：$\beta_0 = \rho_0(\mathbf{y}_0^\top\mathbf{r}) = \frac{1}{2}(2\cdot0 + 1\cdot(-\frac{1}{20})) = -\frac{1}{40}$

$$\mathbf{r} \leftarrow \mathbf{r} + \mathbf{s}_0(\alpha_0 - \beta_0) = \begin{pmatrix}0\\-1/20\end{pmatrix} + \begin{pmatrix}1\\0\end{pmatrix}\left(\frac{1}{6} + \frac{1}{40}\right) = \begin{pmatrix}23/120\\-1/20\end{pmatrix}$$

$i = 1$：$\beta_1 = \rho_1(\mathbf{y}_1^\top\mathbf{r}) = \frac{1}{3}(1\cdot\frac{23}{120} + 3\cdot(-\frac{1}{20})) = \frac{1}{3}\left(\frac{23}{120} - \frac{18}{120}\right) = \frac{5}{360} = \frac{1}{72}$

$$\mathbf{r} \leftarrow \mathbf{r} + \mathbf{s}_1(\alpha_1 - \beta_1) = \begin{pmatrix}23/120\\-1/20\end{pmatrix} + \begin{pmatrix}0\\1\end{pmatrix}\left(\frac{2}{3} - \frac{1}{72}\right) = \begin{pmatrix}23/120\\-1/20 + 47/72\end{pmatrix}$$

计算第二分量：$-\frac{1}{20} + \frac{47}{72} = -\frac{18}{360} + \frac{235}{360} = \frac{217}{360}$

**最终搜索方向**：

$$\mathbf{d} = -\mathbf{r} = \begin{pmatrix}-23/120\\-217/360\end{pmatrix} \approx \begin{pmatrix}-0.192\\-0.603\end{pmatrix}$$

---

**答案 6.5**

（a）牛顿法收敛域分析：

$f(x) = x^4 - 4x^2$，极小值 $x^* = \pm\sqrt{2}$，不稳定极大值 $x = 0$。

导数：$f'(x) = 4x^3 - 8x = 4x(x^2 - 2)$

二阶导数：$f''(x) = 12x^2 - 8$

牛顿迭代格式：

$$x_{k+1} = x_k - \frac{f'(x_k)}{f''(x_k)} = x_k - \frac{4x_k^3 - 8x_k}{12x_k^2 - 8} = x_k - \frac{x_k(x_k^2 - 2)}{3x_k^2 - 2}$$

化简：$x_{k+1} = \frac{x_k(3x_k^2 - 2) - x_k(x_k^2 - 2)}{3x_k^2 - 2} = \frac{2x_k^3}{3x_k^2 - 2}$

从 $x_0 = 0.1$ 出发：$x_1 = \frac{2 \times 0.001}{0.03 - 2} \approx \frac{0.002}{-1.97} \approx -0.001$

可以看到符号翻转，迭代接近0后向负方向（取决于数值积累），数值模拟显示会收敛到 $-\sqrt{2}$。

从 $x_0 = 0$ 出发：$f''(0) = -8 < 0$（鞍点/极大值），Hessian负定，牛顿方向是**上升方向**，不应继续；$f'(0) = 0$，已在驻点，但为极大值点，纯牛顿法会"停留"在此处。

这说明牛顿法对初始点敏感，在不正定区域需要修正（如加正则化）。

（b）K-FAC 维度分析：

**标准 Fisher 矩阵**：网络总参数 $n = L \cdot d^2$，Fisher 矩阵维度 $(n \times n)$

$$\text{标准存储量} = n^2 = (Ld^2)^2 = L^2 d^4$$

代入 $L = 10$，$d = 1000$：$n = 10 \times 10^6 = 10^7$，Fisher 矩阵 $= 10^{14}$ 个浮点数 $\approx 800$ TB（不可行）

**K-FAC 近似存储量**：每层存储两个矩阵 $\mathbf{A}_l \in \mathbb{R}^{d \times d}$ 和 $\mathbf{G}_l \in \mathbb{R}^{d \times d}$，共 $L$ 层：

$$\text{K-FAC存储量} = L \cdot 2d^2 = 2Ld^2$$

代入数值：$2 \times 10 \times 10^6 = 2 \times 10^7$ 个浮点数 $\approx 160$ MB（完全可行！）

**对比**：$\frac{\text{标准}}{\text{K-FAC}} = \frac{L^2 d^4}{2Ld^2} = \frac{Ld^2}{2} = \frac{10 \times 10^6}{2} = 5 \times 10^6$ 倍压缩，这正是 K-FAC 得以实用化的根本原因。

---

*本章详细推导了牛顿法族的理论基础与实用变体。下一章将进入约束优化领域，学习如何处理等式约束问题。*
