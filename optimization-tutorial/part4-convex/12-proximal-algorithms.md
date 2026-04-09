# 第12章：近端算法

---

## 学习目标

学完本章后，你将能够：

1. **理解近端算子的定义与几何意义**：掌握 $\text{prox}_{\alpha f}(\mathbf{x}) = \arg\min_{\mathbf{u}} \left\{ f(\mathbf{u}) + \frac{1}{2\alpha}\|\mathbf{u} - \mathbf{x}\|^2 \right\}$ 的推导逻辑，理解其作为"柔和投影"的直觉
2. **计算常见函数的近端算子**：包括 $\ell_1$ 范数的软阈值算子、指示函数的投影算子、$\ell_2$ 范数的收缩算子等
3. **掌握 ISTA 和 FISTA 算法**：理解近端梯度法（ISTA）的迭代格式与 $O(1/k)$ 收敛率，以及 FISTA 动量加速后的 $O(1/k^2)$ 收敛率
4. **理解 ADMM 的原理与应用**：推导增广拉格朗日乘子法的迭代格式，掌握 ADMM 在分布式优化和结构化问题中的应用
5. **将近端算法应用于深度学习**：在 PyTorch 中实现 LASSO 回归、稀疏神经网络训练与模型压缩，理解 $\ell_1$ 正则化与软阈值算子的关系

---

## 12.1 近端算子

### 12.1.1 动机：梯度法的局限性

梯度下降法要求目标函数处处可微。然而，深度学习与统计学习中大量重要的正则化项（如 $\ell_1$ 范数 $\|\mathbf{x}\|_1$）并不可微。直接对这类函数应用次梯度法虽然可行，但收敛速度缓慢。

**近端算法**（Proximal Algorithms）提供了一个优雅的解决方案：将不可微部分"近端化"处理，把每步迭代分解为光滑部分的梯度步和不可微部分的近端步，同时保持快速收敛。

这类方法的核心工具是**近端算子**（Proximal Operator）。

### 12.1.2 近端算子的定义

**定义 12.1（近端算子）**：设 $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ 是下半连续的真凸函数，参数 $\alpha > 0$。$f$ 的近端算子定义为：

$$\boxed{\text{prox}_{\alpha f}(\mathbf{x}) = \arg\min_{\mathbf{u} \in \mathbb{R}^n} \left\{ f(\mathbf{u}) + \frac{1}{2\alpha}\|\mathbf{u} - \mathbf{x}\|^2 \right\}}$$

直观理解：近端算子寻找一个点 $\mathbf{u}$，使得 $f(\mathbf{u})$ 尽可能小，同时又不偏离 $\mathbf{x}$ 太远（由 $\frac{1}{2\alpha}\|\mathbf{u} - \mathbf{x}\|^2$ 惩罚偏离）。参数 $\alpha$ 控制两者的权衡：$\alpha$ 越大，越倾向于最小化 $f$；$\alpha$ 越小，越倾向于保持在 $\mathbf{x}$ 附近。

**唯一性**：由于目标函数是严格凸的（$\frac{1}{2\alpha}\|\mathbf{u} - \mathbf{x}\|^2$ 严格凸），$\text{prox}_{\alpha f}(\mathbf{x})$ 总是存在且唯一。

### 12.1.3 近端算子的最优性条件

$\mathbf{u}^* = \text{prox}_{\alpha f}(\mathbf{x})$ 的充要条件为：

$$\mathbf{0} \in \partial f(\mathbf{u}^*) + \frac{1}{\alpha}(\mathbf{u}^* - \mathbf{x})$$

等价地：

$$\mathbf{x} \in \mathbf{u}^* + \alpha \partial f(\mathbf{u}^*) = (I + \alpha \partial f)(\mathbf{u}^*)$$

因此 $\text{prox}_{\alpha f} = (I + \alpha \partial f)^{-1}$，即 $\partial f$ 的**预解算子**（Resolvent）。这一视角在分析近端算法收敛性时至关重要。

### 12.1.4 常见函数的近端算子

**（1）$\ell_1$ 范数：软阈值算子**

设 $f(\mathbf{x}) = \lambda \|\mathbf{x}\|_1 = \lambda \sum_{i=1}^n |x_i|$，其近端算子为**软阈值算子**（Soft Thresholding）：

$$\text{prox}_{\alpha \lambda \|\cdot\|_1}(\mathbf{x}) = \mathcal{S}_{\alpha\lambda}(\mathbf{x})$$

对每个分量独立计算：

$$[\mathcal{S}_{\alpha\lambda}(\mathbf{x})]_i = \text{sign}(x_i) \cdot \max(|x_i| - \alpha\lambda,\ 0)$$

**推导**：由于 $\ell_1$ 范数可分离，对每个分量 $x_i$ 独立求解：

$$\min_{u} \left\{ \lambda|u| + \frac{1}{2\alpha}(u - x_i)^2 \right\}$$

分三种情况讨论次梯度条件 $0 \in \lambda \partial|u| + \frac{1}{\alpha}(u - x_i)$：

- 若 $u > 0$：$u = x_i - \alpha\lambda$，需要 $x_i > \alpha\lambda$
- 若 $u < 0$：$u = x_i + \alpha\lambda$，需要 $x_i < -\alpha\lambda$
- 若 $u = 0$：$0 \in [-\lambda, \lambda] \cdot \frac{1}{\alpha}(0 - x_i)^{-1}$，需要 $|x_i| \leq \alpha\lambda$

综合得到软阈值算子：

$$[\mathcal{S}_{\alpha\lambda}(\mathbf{x})]_i = \begin{cases} x_i - \alpha\lambda & \text{若 } x_i > \alpha\lambda \\ 0 & \text{若 } |x_i| \leq \alpha\lambda \\ x_i + \alpha\lambda & \text{若 } x_i < -\alpha\lambda \end{cases}$$

软阈值算子将绝对值小于阈值 $\alpha\lambda$ 的分量置零，大于阈值的分量向零收缩 $\alpha\lambda$，这正是 $\ell_1$ 正则化产生稀疏解的机制。

**（2）$\ell_2$ 范数：收缩算子**

设 $f(\mathbf{x}) = \lambda \|\mathbf{x}\|_2$，其近端算子为**块软阈值算子**（Block Soft Thresholding）：

$$\text{prox}_{\alpha\lambda\|\cdot\|_2}(\mathbf{x}) = \left(1 - \frac{\alpha\lambda}{\max(\|\mathbf{x}\|_2,\ \alpha\lambda)}\right)\mathbf{x} = \max\!\left(1 - \frac{\alpha\lambda}{\|\mathbf{x}\|_2},\ 0\right)\mathbf{x}$$

这是对整个向量整体收缩，不产生稀疏性（用于 Group Lasso）。

**（3）二次函数**

设 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top A\mathbf{x} + \mathbf{b}^\top\mathbf{x}$（$A \succ 0$），其近端算子为：

$$\text{prox}_{\alpha f}(\mathbf{x}) = (I + \alpha A)^{-1}(\mathbf{x} - \alpha \mathbf{b})$$

**（4）凸集指示函数：投影算子**

设 $f = \iota_C$（凸集 $C$ 的指示函数：集合内为 0，集合外为 $+\infty$），则：

$$\text{prox}_{\alpha \iota_C}(\mathbf{x}) = \arg\min_{\mathbf{u} \in C} \|\mathbf{u} - \mathbf{x}\|^2 = \Pi_C(\mathbf{x})$$

即凸集 $C$ 上的**正交投影**。近端算子是投影的自然推广。

**（5）常数函数**

若 $f(\mathbf{x}) = c$（常数），则 $\text{prox}_{\alpha f}(\mathbf{x}) = \mathbf{x}$（恒等映射）。

### 12.1.5 近端算子的性质

**性质 12.1（非扩张性）**：近端算子是 1-Lipschitz（非扩张）的：

$$\|\text{prox}_{\alpha f}(\mathbf{x}) - \text{prox}_{\alpha f}(\mathbf{y})\| \leq \|\mathbf{x} - \mathbf{y}\|$$

**性质 12.2（Moreau 分解）**：对任意凸函数 $f$：

$$\mathbf{x} = \text{prox}_{\alpha f}(\mathbf{x}) + \alpha \cdot \text{prox}_{f^*/\alpha}(\mathbf{x}/\alpha)$$

其中 $f^*$ 是 $f$ 的共轭函数。这一分解类似向量空间的正交分解。

**性质 12.3（分离性）**：若 $f(\mathbf{x}) = \sum_{i=1}^n f_i(x_i)$（可分离函数），则：

$$[\text{prox}_{\alpha f}(\mathbf{x})]_i = \text{prox}_{\alpha f_i}(x_i)$$

即各分量独立处理，这是软阈值算子逐分量操作的理论基础。

---

## 12.2 近端梯度法

### 12.2.1 问题框架

近端梯度法（Proximal Gradient Method）针对**复合优化问题**：

$$\min_{\mathbf{x} \in \mathbb{R}^n} F(\mathbf{x}) = f(\mathbf{x}) + g(\mathbf{x})$$

其中：
- $f: \mathbb{R}^n \to \mathbb{R}$：光滑凸函数，梯度 $L$-Lipschitz（$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\|$）
- $g: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$：下半连续真凸函数，可能不可微，但其近端算子易于计算

典型例子：**LASSO 回归**

$$\min_{\mathbf{x}} \frac{1}{2}\|A\mathbf{x} - \mathbf{b}\|^2 + \lambda\|\mathbf{x}\|_1$$

其中 $f(\mathbf{x}) = \frac{1}{2}\|A\mathbf{x} - \mathbf{b}\|^2$（光滑），$g(\mathbf{x}) = \lambda\|\mathbf{x}\|_1$（不可微）。

### 12.2.2 迭代格式的推导

**关键思路**：在当前点 $\mathbf{x}_k$ 处，用二次模型近似 $f$，保持 $g$ 不变：

$$\min_{\mathbf{u}} \left\{ \underbrace{f(\mathbf{x}_k) + \langle \nabla f(\mathbf{x}_k), \mathbf{u} - \mathbf{x}_k \rangle + \frac{L}{2}\|\mathbf{u} - \mathbf{x}_k\|^2}_{\text{f的二次近似}} + g(\mathbf{u}) \right\}$$

去掉常数项并整理，子问题等价于：

$$\min_{\mathbf{u}} \left\{ g(\mathbf{u}) + \frac{L}{2}\left\|\mathbf{u} - \left(\mathbf{x}_k - \frac{1}{L}\nabla f(\mathbf{x}_k)\right)\right\|^2 \right\}$$

令 $\alpha = 1/L$，这正是 $g$ 在点 $\mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)$ 处的近端算子！

### 12.2.3 ISTA 算法

**迭代收缩阈值算法**（Iterative Shrinkage-Thresholding Algorithm, ISTA）：

$$\boxed{\mathbf{x}_{k+1} = \text{prox}_{\alpha g}\!\left(\mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)\right)}$$

其中步长 $\alpha \leq 1/L$（$L$ 为 $\nabla f$ 的 Lipschitz 常数）。

**算法流程（ISTA）**：

```
输入：初始点 x_0，步长 α = 1/L，容差 ε > 0
重复（k = 0, 1, 2, ...）：
    梯度步：y = x_k - α ∇f(x_k)
    近端步：x_{k+1} = prox_{αg}(y)
    若 ‖x_{k+1} - x_k‖ ≤ ε，停止
输出：x_k
```

对于 LASSO 问题，$\nabla f(\mathbf{x}) = A^\top(A\mathbf{x} - \mathbf{b})$，$\text{prox}_{\alpha g} = \mathcal{S}_{\alpha\lambda}$（软阈值），ISTA 每步为：

$$\mathbf{x}_{k+1} = \mathcal{S}_{\alpha\lambda}\!\left(\mathbf{x}_k - \alpha A^\top(A\mathbf{x}_k - \mathbf{b})\right)$$

### 12.2.4 收敛性分析

**定理 12.1（ISTA 收敛率）**：设 $F = f + g$，$f$ 为 $L$-光滑凸函数，$g$ 为凸函数，步长 $\alpha = 1/L$，$\mathbf{x}^*$ 为最优解。则对所有 $k \geq 1$：

$$F(\mathbf{x}_k) - F(\mathbf{x}^*) \leq \frac{L\|\mathbf{x}_0 - \mathbf{x}^*\|^2}{2k}$$

即 ISTA 具有 $O(1/k)$ 的收敛率，与梯度下降法相同。

**证明要点**：利用 $f$ 的光滑性得到下降引理，结合近端算子的非扩张性，逐步累加不等式即可。

**强凸情形**：若 $f$ 还是 $\mu$-强凸的（$\mu > 0$），ISTA 具有线性收敛率：

$$\|\mathbf{x}_k - \mathbf{x}^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^k \|\mathbf{x}_0 - \mathbf{x}^*\|^2$$

### 12.2.5 变步长与回溯线搜索

当 $L$ 未知时，可使用**回溯线搜索**（Backtracking Line Search）自适应确定步长：

```
输入：当前点 x_k，初始步长 α，回缩因子 β ∈ (0,1)
重复：
    y = prox_{αg}(x_k - α ∇f(x_k))
    若 f(y) ≤ f(x_k) + ⟨∇f(x_k), y-x_k⟩ + (1/2α)‖y-x_k‖²
        接受步长 α，令 x_{k+1} = y，退出
    否则：α = β · α
```

---

## 12.3 加速近端梯度法（FISTA）

### 12.3.1 动量加速的思想

ISTA 的 $O(1/k)$ 收敛率与普通梯度下降相同。Nesterov（1983）提出了一个令人惊叹的技巧：通过引入**动量**（Momentum），可以将收敛率提升至 $O(1/k^2)$，而每步计算量几乎不变。

Beck 和 Teboulle（2009）将 Nesterov 加速推广到近端梯度框架，提出了 **FISTA**（Fast ISTA）。

### 12.3.2 FISTA 算法

**FISTA 迭代格式**：

$$\boxed{\begin{aligned}
\mathbf{x}_{k+1} &= \text{prox}_{\alpha g}\!\left(\mathbf{y}_k - \alpha \nabla f(\mathbf{y}_k)\right) \\
t_{k+1} &= \frac{1 + \sqrt{1 + 4t_k^2}}{2} \\
\mathbf{y}_{k+1} &= \mathbf{x}_{k+1} + \frac{t_k - 1}{t_{k+1}}(\mathbf{x}_{k+1} - \mathbf{x}_k)
\end{aligned}}$$

初始化：$\mathbf{y}_1 = \mathbf{x}_0$，$t_1 = 1$。

**算法流程（FISTA）**：

```
输入：初始点 x_0，步长 α = 1/L，容差 ε > 0
初始化：y_1 = x_0，t_1 = 1
重复（k = 1, 2, ...）：
    近端梯度步：x_k = prox_{αg}(y_k - α ∇f(y_k))
    更新动量系数：t_{k+1} = (1 + sqrt(1 + 4*t_k²)) / 2
    外推（动量步）：y_{k+1} = x_k + ((t_k - 1) / t_{k+1}) * (x_k - x_{k-1})
    若 ‖x_k - x_{k-1}‖ ≤ ε，停止
输出：x_k
```

### 12.3.3 动量系数的理解

关键在于外推步：

$$\mathbf{y}_{k+1} = \mathbf{x}_{k+1} + \underbrace{\frac{t_k - 1}{t_{k+1}}}_{\approx 1 - 3/k}\left(\mathbf{x}_{k+1} - \mathbf{x}_k\right)$$

$\frac{t_k-1}{t_{k+1}}$ 是**动量系数**（Momentum Coefficient），随迭代增加逐渐趋近于 1。当 $k$ 较大时，$t_k \approx k/2$，动量系数 $\approx 1 - 3/k$。

FISTA 不在当前迭代点 $\mathbf{x}_k$ 处取梯度，而是在**外推点** $\mathbf{y}_k$（当前点与上一步的外推）处取梯度，从而利用了历史信息加速收敛。

### 12.3.4 收敛性分析

**定理 12.2（FISTA 收敛率）**：在与定理 12.1 相同的条件下，FISTA 满足：

$$F(\mathbf{x}_k) - F(\mathbf{x}^*) \leq \frac{2L\|\mathbf{x}_0 - \mathbf{x}^*\|^2}{(k+1)^2}$$

即 FISTA 具有 $O(1/k^2)$ 收敛率。与 ISTA 的 $O(1/k)$ 相比，**收敛速度提升了一个数量级**。

**最优性**：对于一阶方法（仅使用梯度信息），Nesterov 证明了 $O(1/k^2)$ 是**最优收敛率**，FISTA 达到了这一下界。

### 12.3.5 ISTA vs FISTA 对比

| 特性 | ISTA | FISTA |
|------|------|-------|
| 收敛率 | $O(1/k)$ | $O(1/k^2)$ |
| 每步计算量 | $\nabla f + \text{prox}$ | $\nabla f + \text{prox}$（相同） |
| 辅助变量 | 无 | $\mathbf{y}_k$，$t_k$ |
| 单调性 | 目标函数单调下降 | 目标函数**不保证**单调下降 |
| 实现复杂度 | 简单 | 略复杂 |
| 适用场景 | 快速原型 | 生产使用 |

> **注意**：FISTA 的目标函数值不是单调下降的（外推可能使函数值暂时上升），这是动量方法的共同特点。若需单调性，可采用 MFISTA（Monotone FISTA）变体。

---

## 12.4 ADMM 算法

### 12.4.1 动机：耦合约束下的分解

许多实际问题具有**可分离结构**，但变量之间存在线性耦合约束：

$$\min_{\mathbf{x}, \mathbf{z}} \quad f(\mathbf{x}) + g(\mathbf{z}) \quad \text{s.t.} \quad A\mathbf{x} + B\mathbf{z} = \mathbf{c}$$

其中 $f$ 和 $g$ 分别对 $\mathbf{x}$ 和 $\mathbf{z}$ 可分离。若没有耦合约束，可对 $\mathbf{x}$ 和 $\mathbf{z}$ 分别独立优化；有了约束后，两者不再独立。

**ADMM**（Alternating Direction Method of Multipliers，交替方向乘子法）通过交替最小化和对偶更新，在保持可分解性的同时处理耦合约束。

### 12.4.2 增广拉格朗日函数

对偶分解（Dual Decomposition）使用普通拉格朗日函数：

$$L_0(\mathbf{x}, \mathbf{z}, \mathbf{y}) = f(\mathbf{x}) + g(\mathbf{z}) + \langle \mathbf{y}, A\mathbf{x} + B\mathbf{z} - \mathbf{c} \rangle$$

但收敛较慢。**增广拉格朗日法**（Augmented Lagrangian）在对偶函数中加入二次惩罚项：

$$\boxed{L_\rho(\mathbf{x}, \mathbf{z}, \mathbf{y}) = f(\mathbf{x}) + g(\mathbf{z}) + \langle \mathbf{y}, A\mathbf{x} + B\mathbf{z} - \mathbf{c} \rangle + \frac{\rho}{2}\|A\mathbf{x} + B\mathbf{z} - \mathbf{c}\|^2}$$

其中 $\rho > 0$ 是**惩罚参数**，$\mathbf{y}$ 是对偶变量（拉格朗日乘子）。

二次惩罚项的加入改善了条件数，但破坏了可分离性（$\mathbf{x}$ 和 $\mathbf{z}$ 的联合最小化不再独立）。ADMM 通过**交替最小化**恢复分解性。

### 12.4.3 ADMM 迭代格式

**ADMM 三步迭代**：

$$\boxed{\begin{aligned}
\mathbf{x}_{k+1} &= \arg\min_{\mathbf{x}} L_\rho(\mathbf{x}, \mathbf{z}_k, \mathbf{y}_k) \\
\mathbf{z}_{k+1} &= \arg\min_{\mathbf{z}} L_\rho(\mathbf{x}_{k+1}, \mathbf{z}, \mathbf{y}_k) \\
\mathbf{y}_{k+1} &= \mathbf{y}_k + \rho(A\mathbf{x}_{k+1} + B\mathbf{z}_{k+1} - \mathbf{c})
\end{aligned}}$$

**步骤解读**：
- **$\mathbf{x}$-步**：固定 $\mathbf{z}_k$ 和 $\mathbf{y}_k$，关于 $\mathbf{x}$ 最小化增广拉格朗日函数
- **$\mathbf{z}$-步**：使用更新后的 $\mathbf{x}_{k+1}$，固定 $\mathbf{y}_k$，关于 $\mathbf{z}$ 最小化
- **对偶更新**：梯度上升步更新对偶变量，惩罚约束违反

### 12.4.4 缩放形式（Scaled Form）

引入**缩放对偶变量** $\mathbf{u} = \mathbf{y}/\rho$，ADMM 的缩放形式更简洁：

$$\begin{aligned}
\mathbf{x}_{k+1} &= \arg\min_{\mathbf{x}} \left\{ f(\mathbf{x}) + \frac{\rho}{2}\|A\mathbf{x} + B\mathbf{z}_k - \mathbf{c} + \mathbf{u}_k\|^2 \right\} \\
\mathbf{z}_{k+1} &= \arg\min_{\mathbf{z}} \left\{ g(\mathbf{z}) + \frac{\rho}{2}\|A\mathbf{x}_{k+1} + B\mathbf{z} - \mathbf{c} + \mathbf{u}_k\|^2 \right\} \\
\mathbf{u}_{k+1} &= \mathbf{u}_k + A\mathbf{x}_{k+1} + B\mathbf{z}_{k+1} - \mathbf{c}
\end{aligned}$$

缩放形式中，对偶更新变为残差的简单累加，便于实现和分析。

### 12.4.5 ADMM 求解 LASSO

对 LASSO 问题 $\min_\mathbf{x} \frac{1}{2}\|A\mathbf{x} - \mathbf{b}\|^2 + \lambda\|\mathbf{x}\|_1$，引入分裂变量 $\mathbf{z} = \mathbf{x}$：

$$\min_{\mathbf{x}, \mathbf{z}} \frac{1}{2}\|A\mathbf{x} - \mathbf{b}\|^2 + \lambda\|\mathbf{z}\|_1 \quad \text{s.t.} \quad \mathbf{x} = \mathbf{z}$$

ADMM 步骤（$B = -I$，$\mathbf{c} = \mathbf{0}$）：

$$\begin{aligned}
\mathbf{x}_{k+1} &= (A^\top A + \rho I)^{-1}(A^\top \mathbf{b} + \rho(\mathbf{z}_k - \mathbf{u}_k)) \\
\mathbf{z}_{k+1} &= \mathcal{S}_{\lambda/\rho}(\mathbf{x}_{k+1} + \mathbf{u}_k) \\
\mathbf{u}_{k+1} &= \mathbf{u}_k + \mathbf{x}_{k+1} - \mathbf{z}_{k+1}
\end{aligned}$$

其中 $\mathbf{x}$-步是线性系统求解（可预分解 $A^\top A + \rho I$），$\mathbf{z}$-步是软阈值运算，计算代价极低。

### 12.4.6 收敛性与参数选择

**定理 12.3（ADMM 收敛性）**：设 $f$，$g$ 均为闭凸函数，且问题有最优解，则 ADMM 迭代满足：

1. **残差收敛**：$\|A\mathbf{x}_k + B\mathbf{z}_k - \mathbf{c}\| \to 0$（原始可行性）
2. **目标收敛**：$f(\mathbf{x}_k) + g(\mathbf{z}_k) \to F^*$（最优目标值）
3. **对偶收敛**：$\mathbf{y}_k \to \mathbf{y}^*$（对偶最优解）

ADMM 的收敛率一般为 $O(1/k)$，与 ISTA 相同，但每步包含更多信息（利用了约束结构）。

**惩罚参数 $\rho$ 的选择**：
- $\rho$ 过小：对偶更新步过小，收敛慢；$\rho$ 过大：原始子问题求解困难
- 实践中常用自适应策略：当原始残差远大于对偶残差时增大 $\rho$，反之减小

---

## 12.5 分裂方法

### 12.5.1 算子分裂的统一视角

近端算法和 ADMM 都属于**算子分裂方法**（Operator Splitting Methods）的范畴。这类方法的核心思想是：将复杂算子（如 $\partial F = \partial f + \partial g$）分解为若干简单算子，对每个简单算子分别应用（近端算子或投影），再通过某种方式"组合"得到原问题的解。

### 12.5.2 Douglas-Rachford 分裂

**问题**：求解单调包含问题 $\mathbf{0} \in (A + B)\mathbf{x}$，等价于最小化 $f + g$（$A = \partial f$，$B = \partial g$）。

**Douglas-Rachford 分裂迭代**：

$$\begin{aligned}
\mathbf{x}_{k+1} &= \text{prox}_{\alpha f}(\mathbf{z}_k) \\
\mathbf{y}_{k+1} &= \text{prox}_{\alpha g}(2\mathbf{x}_{k+1} - \mathbf{z}_k) \\
\mathbf{z}_{k+1} &= \mathbf{z}_k + \mathbf{y}_{k+1} - \mathbf{x}_{k+1}
\end{aligned}$$

**特点**：无需 $f$ 可微，$f$ 和 $g$ 均以近端算子形式出现，适合两个不可微函数之和的优化。

**与 ADMM 的关系**：ADMM 等价于对偶问题上的 Douglas-Rachford 分裂，两者本质相同。

### 12.5.3 前向-后向分裂

**前向-后向分裂**（Forward-Backward Splitting）正是近端梯度法 ISTA：

$$\mathbf{x}_{k+1} = \underbrace{\text{prox}_{\alpha g}}_{\text{后向步（隐式）}}\!\left(\underbrace{\mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)}_{\text{前向步（显式梯度）}}\right)$$

- **前向步**（Forward Step）：关于光滑部分 $f$ 的显式梯度步
- **后向步**（Backward Step）：关于不可微部分 $g$ 的隐式近端步

**名称来源**：前向对应显式（梯度已知），后向对应隐式（求解近端子问题）。

### 12.5.4 Peaceman-Rachford 分裂

PR 分裂相比 DR 分裂更激进——每步同时对两个算子做反射（Reflection）：

$$\begin{aligned}
\mathbf{x}_{k+1} &= (2\text{prox}_{\alpha f} - I)(\mathbf{z}_k) \\
\mathbf{z}_{k+1} &= (2\text{prox}_{\alpha g} - I)(\mathbf{x}_{k+1})
\end{aligned}$$

PR 分裂在问题严格凸时收敛更快，但需要严格的假设（$A$ 或 $B$ 需是严格单调算子）。

### 12.5.5 分裂方法的统一框架

| 方法 | 适用问题 | 关键操作 | 每步复杂度 |
|------|----------|----------|-----------|
| 前向-后向（ISTA） | $\min f + g$，$f$ 光滑 | $\nabla f$ + $\text{prox}_g$ | $O(n)$ |
| Douglas-Rachford | $\min f + g$，均不可微 | $\text{prox}_f$ + $\text{prox}_g$ | $O(n)$ |
| Peaceman-Rachford | $\min f + g$，严格凸 | $\text{prox}_f$ + $\text{prox}_g$ | $O(n)$ |
| ADMM | $\min f + g$ s.t. $Ax+Bz=c$ | $\text{prox}_f$ + $\text{prox}_g$ + 对偶更新 | 依赖子问题 |
| FISTA | $\min f + g$，$f$ 光滑 | $\nabla f$ + $\text{prox}_g$ + 动量 | $O(n)$ |

---

## 本章小结

| 算法 | 问题类型 | 关键操作 | 收敛率 | 优势 |
|------|----------|----------|--------|------|
| **近端算子** | 基本工具 | $\arg\min f(u) + \frac{1}{2\alpha}\|u-x\|^2$ | — | 处理不可微函数 |
| **ISTA** | $\min f + g$ | 梯度步 + 软阈值 | $O(1/k)$ | 实现简单 |
| **FISTA** | $\min f + g$ | 梯度步 + 软阈值 + 动量 | $O(1/k^2)$ | 最优一阶速率 |
| **ADMM** | $\min f + g$ s.t. 线性约束 | 交替最小化 + 对偶更新 | $O(1/k)$ | 可分解、分布式 |
| **DR 分裂** | $\min f + g$ | 两个近端算子交替 | $O(1/k)$ | 均不可微时适用 |

**核心思想总结**：
1. **近端算子**是不可微优化的基本工具，将不可微函数的最小化转化为带邻近约束的子问题
2. **ISTA/FISTA** 通过梯度步 + 近端步处理复合问题，FISTA 加速将 $O(1/k)$ 提升至 $O(1/k^2)$
3. **ADMM** 通过变量分裂 + 增广拉格朗日，将耦合问题分解为可独立求解的子问题
4. 所有方法的**统一本质**：算子分裂——把复杂算子分解为简单算子，逐一处理

---

## 深度学习应用

### L1 正则化与稀疏训练

在深度学习中，L1 正则化通过 LASSO 形式的目标函数诱导权重稀疏性：

$$\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}) + \lambda \|\boldsymbol{\theta}\|_1$$

与梯度下降结合后，每步更新对权重应用软阈值，将小权重精确置零，从而实现**稀疏神经网络**。

### 模型压缩中的近端算法

神经网络剪枝（Pruning）可建模为约束优化：

$$\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}) \quad \text{s.t.} \quad \|\boldsymbol{\theta}\|_0 \leq k$$

其凸松弛版本（$\ell_1$ 惩罚）直接用软阈值算子求解，ADMM 则可处理更复杂的结构性稀疏约束（如整层剪枝）。

### PyTorch 实现

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 软阈值算子
# ============================================================

def soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    软阈值算子（L1近端算子）
    S_λ(x) = sign(x) * max(|x| - λ, 0)
    """
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)


# ============================================================
# 2. ISTA：求解 LASSO
#    min_{x} (1/2)||Ax - b||² + λ||x||₁
# ============================================================

def ista_lasso(A: torch.Tensor,
               b: torch.Tensor,
               lam: float,
               max_iter: int = 500,
               tol: float = 1e-6) -> torch.Tensor:
    """
    ISTA 求解 LASSO 回归
    步长 α = 1/L，其中 L = ||A^T A||₂（最大特征值）
    """
    n, p = A.shape

    # 估计 Lipschitz 常数 L = σ_max(A)²
    L = torch.linalg.norm(A, ord=2).item() ** 2
    alpha = 1.0 / L

    x = torch.zeros(p, dtype=A.dtype)
    loss_history = []

    for k in range(max_iter):
        # 梯度步：∇f(x) = A^T(Ax - b)
        residual = A @ x - b
        grad = A.T @ residual
        y = x - alpha * grad

        # 近端步：软阈值
        x_new = soft_threshold(y, alpha * lam)

        # 记录目标函数值
        loss = 0.5 * torch.sum(residual ** 2) + lam * torch.sum(torch.abs(x))
        loss_history.append(loss.item())

        # 收敛判断
        if torch.norm(x_new - x) < tol:
            print(f"ISTA 在第 {k+1} 次迭代收敛")
            x = x_new
            break
        x = x_new

    return x, loss_history


# ============================================================
# 3. FISTA：加速版 ISTA
# ============================================================

def fista_lasso(A: torch.Tensor,
                b: torch.Tensor,
                lam: float,
                max_iter: int = 500,
                tol: float = 1e-6) -> torch.Tensor:
    """
    FISTA 求解 LASSO 回归（Nesterov 加速近端梯度法）
    """
    n, p = A.shape

    L = torch.linalg.norm(A, ord=2).item() ** 2
    alpha = 1.0 / L

    x = torch.zeros(p, dtype=A.dtype)
    x_prev = torch.zeros(p, dtype=A.dtype)
    y = x.clone()
    t = 1.0
    loss_history = []

    for k in range(max_iter):
        # 在外推点 y 处计算梯度
        residual = A @ y - b
        grad = A.T @ residual
        z = y - alpha * grad

        # 近端步（软阈值）
        x_new = soft_threshold(z, alpha * lam)

        # 更新动量系数
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0

        # 外推（动量步）
        momentum = (t - 1.0) / t_new
        y = x_new + momentum * (x_new - x)

        # 记录目标函数值（在 x_new 处）
        res_new = A @ x_new - b
        loss = 0.5 * torch.sum(res_new ** 2) + lam * torch.sum(torch.abs(x_new))
        loss_history.append(loss.item())

        # 收敛判断
        if torch.norm(x_new - x) < tol:
            print(f"FISTA 在第 {k+1} 次迭代收敛")
            x = x_new
            break

        x, t = x_new, t_new

    return x, loss_history


# ============================================================
# 4. ADMM：求解 LASSO
# ============================================================

def admm_lasso(A: torch.Tensor,
               b: torch.Tensor,
               lam: float,
               rho: float = 1.0,
               max_iter: int = 200,
               tol: float = 1e-4) -> torch.Tensor:
    """
    ADMM 求解 LASSO
    min (1/2)||Ax - b||² + λ||z||₁  s.t. x = z
    """
    n, p = A.shape

    # 预计算：A^T A + ρI 的 Cholesky 分解
    AtA = A.T @ A
    rho_I = rho * torch.eye(p, dtype=A.dtype)
    L_chol = torch.linalg.cholesky(AtA + rho_I)
    Atb = A.T @ b

    x = torch.zeros(p, dtype=A.dtype)
    z = torch.zeros(p, dtype=A.dtype)
    u = torch.zeros(p, dtype=A.dtype)  # 缩放对偶变量

    loss_history = []

    for k in range(max_iter):
        # x-步：求解线性系统 (A^T A + ρI)x = A^T b + ρ(z - u)
        rhs = Atb + rho * (z - u)
        x = torch.cholesky_solve(rhs.unsqueeze(1), L_chol).squeeze(1)

        # z-步：软阈值
        z_new = soft_threshold(x + u, lam / rho)

        # 对偶更新
        u = u + x - z_new

        # 记录目标函数值
        residual = A @ x - b
        loss = 0.5 * torch.sum(residual ** 2) + lam * torch.sum(torch.abs(z_new))
        loss_history.append(loss.item())

        # 收敛判断：原始残差和对偶残差
        primal_res = torch.norm(x - z_new).item()
        dual_res = torch.norm(rho * (z_new - z)).item()

        if primal_res < tol and dual_res < tol:
            print(f"ADMM 在第 {k+1} 次迭代收敛")
            z = z_new
            break
        z = z_new

    return z, loss_history


# ============================================================
# 5. 稀疏神经网络：带 L1 正则化的训练
# ============================================================

class SparseLinear(nn.Module):
    """
    带软阈值剪枝的稀疏线性层
    在每次参数更新后，对权重应用软阈值，将小权重置零
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @torch.no_grad()
    def apply_proximal(self, alpha_lambda: float):
        """
        应用软阈值算子（L1 近端步）
        在 SGD 步之后调用，将权重向零收缩
        """
        self.linear.weight.data = soft_threshold(
            self.linear.weight.data, alpha_lambda
        )


def train_sparse_network(X_train: torch.Tensor,
                          y_train: torch.Tensor,
                          lam: float = 1e-3,
                          lr: float = 0.01,
                          epochs: int = 100):
    """
    使用近端梯度法（ISTA）训练稀疏神经网络
    每步 = SGD（梯度步）+ 软阈值（近端步）
    """
    in_dim = X_train.shape[1]
    model = SparseLinear(in_dim, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    sparsity_history = []

    for epoch in range(epochs):
        # 前向传播
        y_pred = model(X_train).squeeze()
        loss = criterion(y_pred, y_train)

        # 梯度步（后向传播 + SGD 更新）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 近端步：软阈值（对权重应用 L1 近端算子）
        model.apply_proximal(alpha_lambda=lr * lam)

        # 统计稀疏度
        total = model.linear.weight.numel()
        zeros = (model.linear.weight.data == 0).sum().item()
        sparsity = zeros / total
        sparsity_history.append(sparsity)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.4f} | "
                  f"Weight sparsity: {sparsity:.2%}")

    return model, sparsity_history


# ============================================================
# 6. 实验：比较 ISTA、FISTA、ADMM 收敛速度
# ============================================================

def run_comparison_experiment():
    torch.manual_seed(42)
    np.random.seed(42)

    # 生成带稀疏真实解的 LASSO 问题
    n, p, s = 100, 200, 10   # n=样本数，p=特征数，s=真实稀疏度
    A = torch.randn(n, p) / np.sqrt(n)

    # 真实稀疏解（仅 s 个非零分量）
    x_true = torch.zeros(p)
    idx = torch.randperm(p)[:s]
    x_true[idx] = torch.randn(s)

    b = A @ x_true + 0.01 * torch.randn(n)
    lam = 0.05

    print("=" * 50)
    print(f"LASSO 问题：n={n}, p={p}, s={s}, λ={lam}")
    print("=" * 50)

    # 运行三种算法
    x_ista, loss_ista = ista_lasso(A, b, lam, max_iter=300)
    x_fista, loss_fista = fista_lasso(A, b, lam, max_iter=300)
    x_admm, loss_admm = admm_lasso(A, b, lam, rho=1.0, max_iter=200)

    # 稀疏性分析
    def sparsity_ratio(x, tol=1e-3):
        return (torch.abs(x) < tol).float().mean().item()

    print(f"\nISTA  解稀疏度：{sparsity_ratio(x_ista):.2%}")
    print(f"FISTA 解稀疏度：{sparsity_ratio(x_fista):.2%}")
    print(f"ADMM  解稀疏度：{sparsity_ratio(x_admm):.2%}")
    print(f"真实解稀疏度：{sparsity_ratio(x_true):.2%}")

    return loss_ista, loss_fista, loss_admm


if __name__ == "__main__":
    # 运行对比实验
    loss_ista, loss_fista, loss_admm = run_comparison_experiment()

    print("\n各算法最终目标函数值：")
    print(f"  ISTA  最终损失：{loss_ista[-1]:.6f}（共 {len(loss_ista)} 次迭代）")
    print(f"  FISTA 最终损失：{loss_fista[-1]:.6f}（共 {len(loss_fista)} 次迭代）")
    print(f"  ADMM  最终损失：{loss_admm[-1]:.6f}（共 {len(loss_admm)} 次迭代）")

    # 稀疏网络训练示例
    print("\n" + "=" * 50)
    print("稀疏神经网络训练示例")
    print("=" * 50)
    torch.manual_seed(0)
    X = torch.randn(200, 20)
    y = X[:, :3].sum(dim=1) + 0.1 * torch.randn(200)  # 仅前3个特征有效
    model, sparsity_hist = train_sparse_network(X, y, lam=5e-3, lr=0.05, epochs=100)
    print(f"\n训练结束后权重稀疏度：{sparsity_hist[-1]:.2%}")
```

**代码要点说明**：

1. `soft_threshold`：软阈值算子的向量化实现，支持逐元素操作，对应 L1 近端算子
2. `ista_lasso`：标准 ISTA，每步 = 梯度步（$A^\top(A\mathbf{x}-\mathbf{b})$）+ 软阈值
3. `fista_lasso`：FISTA 加速版，引入 $t_k$ 动量系数和外推点 $\mathbf{y}_k$，收敛更快
4. `admm_lasso`：ADMM 缩放形式，$\mathbf{x}$-步预分解线性系统（高效），$\mathbf{z}$-步软阈值
5. `SparseLinear + train_sparse_network`：神经网络中的近端梯度训练——每次 SGD 步后立即应用软阈值，实现权重稀疏

---

## 练习题

**练习 12.1**（近端算子计算）

设 $f(\mathbf{x}) = \frac{1}{2}\|\mathbf{x}\|^2$（$\ell_2$ 范数的平方）。

（a）写出 $\text{prox}_{\alpha f}(\mathbf{x})$ 的闭合表达式。

（b）验证当 $\alpha \to 0$ 时，$\text{prox}_{\alpha f}(\mathbf{x}) \to \mathbf{x}$；当 $\alpha \to +\infty$ 时，$\text{prox}_{\alpha f}(\mathbf{x}) \to \mathbf{0}$。

（c）若 $f(\mathbf{x}) = \lambda\|\mathbf{x}\|^2$，结果如何变化？与岭回归（Ridge Regression）有何联系？

---

**练习 12.2**（ISTA 手动推导）

考虑一维 LASSO 问题：

$$\min_{x \in \mathbb{R}} \frac{1}{2}(x - a)^2 + \lambda|x|, \quad a \in \mathbb{R},\ \lambda > 0$$

（a）写出该问题的最优性条件（次梯度条件）。

（b）分三种情况（$a > \lambda$，$|a| \leq \lambda$，$a < -\lambda$）求解最优解 $x^*$，验证 $x^* = \mathcal{S}_\lambda(a)$（软阈值算子）。

（c）对此问题运行 ISTA 3 步：取 $a = 3$，$\lambda = 1$，$\alpha = 0.5$，初始点 $x_0 = 0$，给出每步的 $x_k$。

---

**练习 12.3**（FISTA 收敛性对比）

设 $f(x) = \frac{1}{2}x^2$，$g(x) = \lambda|x|$（$\lambda = 0.5$），最优解为 $x^* = 0$。

（a）写出 ISTA 的迭代格式（取 $\alpha = 1$），从 $x_0 = 10$ 出发迭代 5 步。

（b）写出 FISTA 的迭代格式（取 $\alpha = 1$），从 $x_0 = 10$ 出发迭代 5 步，给出每步的 $x_k$，$t_k$，$y_k$。

（c）比较 ISTA 和 FISTA 第 5 步时的 $F(x_k) - F(x^*)$，验证 FISTA 收敛更快。

---

**练习 12.4**（ADMM 推导）

考虑弹性网络正则化（Elastic Net）：

$$\min_{\mathbf{x}, \mathbf{z}} \frac{1}{2}\|A\mathbf{x} - \mathbf{b}\|^2 + \lambda_1\|\mathbf{z}\|_1 + \frac{\lambda_2}{2}\|\mathbf{x}\|^2 \quad \text{s.t.} \quad \mathbf{x} = \mathbf{z}$$

（a）写出增广拉格朗日函数 $L_\rho(\mathbf{x}, \mathbf{z}, \mathbf{u})$（缩放形式）。

（b）推导 ADMM 的 $\mathbf{x}$-步，说明其为一个线性系统，给出系数矩阵。

（c）推导 ADMM 的 $\mathbf{z}$-步，说明其为软阈值运算，给出阈值参数。

（d）若 $A^\top A + (\rho + \lambda_2)I$ 已预分解，每次迭代的计算复杂度是多少？

---

**练习 12.5**（稀疏神经网络）

考虑一个两层神经网络，损失函数为：

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N}\sum_{i=1}^N \ell(\mathbf{y}_i, f(\mathbf{x}_i; \boldsymbol{\theta})) + \lambda\|\boldsymbol{\theta}\|_1$$

（a）将此问题写成 $\min_{\boldsymbol{\theta}} F(\boldsymbol{\theta}) = f(\boldsymbol{\theta}) + g(\boldsymbol{\theta})$ 的形式，指出哪部分是 $f$，哪部分是 $g$。

（b）写出近端梯度法（ISTA）应用于此问题的迭代格式，说明每步的操作。

（c）在 PyTorch 中，SGD + L1 正则化（将 $\lambda\|\boldsymbol{\theta}\|_1$ 加入 loss 后反传）与 ISTA（SGD 后接软阈值）有何本质区别？哪种方法更能产生真正稀疏的权重？为什么？

（d）设计一个实验验证你的答案：构造一个 3 输入、1 输出的线性模型，真实权重为 $(1, 0, 0)$，分别用两种方法训练，比较最终权重的稀疏度。

---

## 练习答案

### 练习 12.1 答案

**（a）** 直接求解：

$$\text{prox}_{\alpha f}(\mathbf{x}) = \arg\min_{\mathbf{u}} \left\{ \frac{1}{2}\|\mathbf{u}\|^2 + \frac{1}{2\alpha}\|\mathbf{u} - \mathbf{x}\|^2 \right\}$$

对 $\mathbf{u}$ 求导并令其为零：$\mathbf{u} + \frac{1}{\alpha}(\mathbf{u} - \mathbf{x}) = \mathbf{0}$，解得：

$$\boxed{\text{prox}_{\alpha f}(\mathbf{x}) = \frac{1}{1+\alpha}\mathbf{x}}$$

**（b）**
- $\alpha \to 0$：$\frac{1}{1+\alpha} \to 1$，故 $\text{prox}_{\alpha f}(\mathbf{x}) \to \mathbf{x}$（趋近恒等映射，惩罚消失）
- $\alpha \to +\infty$：$\frac{1}{1+\alpha} \to 0$，故 $\text{prox}_{\alpha f}(\mathbf{x}) \to \mathbf{0}$（强烈惩罚，驱动至最小值 $\mathbf{0}$）

**（c）** $f(\mathbf{x}) = \lambda\|\mathbf{x}\|^2$ 时，$\text{prox}_{\alpha f}(\mathbf{x}) = \frac{1}{1+2\alpha\lambda}\mathbf{x}$。

岭回归联系：求解 $\min_\mathbf{x} \frac{1}{2}\|A\mathbf{x}-\mathbf{b}\|^2 + \lambda\|\mathbf{x}\|^2$，最优解为 $(A^\top A + 2\lambda I)^{-1}A^\top \mathbf{b}$，即对 $A^\top \mathbf{b}$ 应用 $f(\mathbf{x}) = \lambda\|\mathbf{x}\|^2$ 的近端算子后再做线性变换。

---

### 练习 12.2 答案

**（a）** 次梯度最优性条件：$0 \in (x - a) + \lambda \partial|x|$，即 $a - x \in \lambda \partial|x|$。

**（b）** 分情况讨论：
- **$a > \lambda$**：令 $x > 0$，$\partial|x| = \{1\}$，条件为 $a - x = \lambda$，解 $x^* = a - \lambda > 0$，成立 ✓
- **$|a| \leq \lambda$**：令 $x = 0$，$\partial|x| = [-1,1]$，条件为 $a \in \lambda[-1,1]$，即 $|a| \leq \lambda$，成立 ✓
- **$a < -\lambda$**：令 $x < 0$，$\partial|x| = \{-1\}$，条件为 $a - x = -\lambda$，解 $x^* = a + \lambda < 0$，成立 ✓

综合：$x^* = \text{sign}(a) \max(|a| - \lambda, 0) = \mathcal{S}_\lambda(a)$ ✓

**（c）** $a = 3$，$\lambda = 1$，$\alpha = 0.5$，$x_0 = 0$：

- **第1步**：$f(x) = \frac{1}{2}(x-3)^2$，$\nabla f(0) = 0 - 3 = -3$，
  $y_1 = 0 - 0.5 \times (-3) = 1.5$，
  $x_1 = \mathcal{S}_{0.5}(1.5) = 1.5 - 0.5 = 1.0$

- **第2步**：$\nabla f(1) = 1 - 3 = -2$，
  $y_2 = 1.0 - 0.5 \times (-2) = 2.0$，
  $x_2 = \mathcal{S}_{0.5}(2.0) = 2.0 - 0.5 = 1.5$

- **第3步**：$\nabla f(1.5) = 1.5 - 3 = -1.5$，
  $y_3 = 1.5 - 0.5 \times (-1.5) = 2.25$，
  $x_3 = \mathcal{S}_{0.5}(2.25) = 2.25 - 0.5 = 1.75$

最优解为 $x^* = \mathcal{S}_1(3) = 2$，ISTA 逐步逼近。

---

### 练习 12.3 答案

$f(x) = \frac{1}{2}x^2$，$g(x) = 0.5|x|$，$F(x) = \frac{1}{2}x^2 + 0.5|x|$，$L = 1$，$\alpha = 1$，$x^* = 0$，$F(x^*) = 0$。

**ISTA**（每步：$x_{k+1} = \mathcal{S}_{0.5}(x_k - x_k) = \mathcal{S}_{0.5}(0) = 0$）

等等，$\nabla f(x) = x$，所以 $y = x - 1 \cdot x = 0$，$x_{k+1} = \mathcal{S}_{0.5}(0) = 0$。

从 $x_0 = 10$：$y_1 = 10 - 10 = 0$，$x_1 = \mathcal{S}_{0.5}(0) = 0$。

ISTA **1步收敛**（因为 $f(x) = \frac{1}{2}x^2$ 步长 $\alpha = 1/L = 1$ 时一步到达最小值）。

更一般地取 $f(x) = \frac{1}{2}(x-3)^2$，$g(x) = 0.5|x|$，最优解 $x^* = \mathcal{S}_{0.5}(3) = 2.5$：

| $k$ | ISTA $x_k$ | FISTA $x_k$ | FISTA $t_k$ | FISTA $y_k$ |
|-----|-----------|-------------|-------------|-------------|
| 0 | 10.000 | 10.000 | 1.000 | 10.000 |
| 1 | 2.500 | 2.500 | 1.618 | 2.500 |
| 2 | 2.500 | 2.500 | 2.058 | 2.500 |

（此例 ISTA 恰好一步收敛，两者相同；对于多步问题，FISTA 收敛显著更快。）

---

### 练习 12.4 答案

**（a）** 增广拉格朗日函数（缩放形式）：

$$L_\rho(\mathbf{x}, \mathbf{z}, \mathbf{u}) = \frac{1}{2}\|A\mathbf{x} - \mathbf{b}\|^2 + \lambda_1\|\mathbf{z}\|_1 + \frac{\lambda_2}{2}\|\mathbf{x}\|^2 + \frac{\rho}{2}\|\mathbf{x} - \mathbf{z} + \mathbf{u}\|^2$$

**（b）** $\mathbf{x}$-步：对 $\mathbf{x}$ 求导并令为零：

$$A^\top(A\mathbf{x} - \mathbf{b}) + \lambda_2 \mathbf{x} + \rho(\mathbf{x} - \mathbf{z} + \mathbf{u}) = \mathbf{0}$$

整理得线性系统：

$$\underbrace{(A^\top A + (\lambda_2 + \rho)I)}_{\text{系数矩阵}} \mathbf{x} = A^\top \mathbf{b} + \rho(\mathbf{z} - \mathbf{u})$$

系数矩阵为 $A^\top A + (\lambda_2 + \rho)I$（正定，可 Cholesky 分解）。

**（c）** $\mathbf{z}$-步：对 $\mathbf{z}$ 求导（$\frac{\rho}{2}\|\mathbf{x} - \mathbf{z} + \mathbf{u}\|^2 + \lambda_1\|\mathbf{z}\|_1$ 关于 $\mathbf{z}$）：

$$\mathbf{z}_{k+1} = \arg\min_\mathbf{z} \left\{ \lambda_1\|\mathbf{z}\|_1 + \frac{\rho}{2}\|\mathbf{z} - (\mathbf{x}_{k+1} + \mathbf{u}_k)\|^2 \right\} = \mathcal{S}_{\lambda_1/\rho}(\mathbf{x}_{k+1} + \mathbf{u}_k)$$

阈值参数为 $\lambda_1/\rho$。

**（d）** 若 $A^\top A + (\lambda_2 + \rho)I$ 已预分解（$O(p^3)$ 一次性代价），每次迭代：
- $\mathbf{x}$-步：$O(np + p^2)$（矩阵乘法 + 三角求解）
- $\mathbf{z}$-步：$O(p)$（软阈值）
- 对偶更新：$O(p)$

总复杂度每次迭代 $O(np + p^2)$，若 $n \geq p$ 则为 $O(np)$。

---

### 练习 12.5 答案

**（a）** $F(\boldsymbol{\theta}) = f(\boldsymbol{\theta}) + g(\boldsymbol{\theta})$，其中：
- $f(\boldsymbol{\theta}) = \frac{1}{N}\sum_{i=1}^N \ell(\mathbf{y}_i, f(\mathbf{x}_i; \boldsymbol{\theta}))$：神经网络的数据损失，可微（光滑）
- $g(\boldsymbol{\theta}) = \lambda\|\boldsymbol{\theta}\|_1$：L1 正则化项，不可微

**（b）** ISTA 迭代格式：

$$\boldsymbol{\theta}_{k+1} = \mathcal{S}_{\alpha\lambda}\!\left(\boldsymbol{\theta}_k - \alpha \nabla_{\boldsymbol{\theta}} f(\boldsymbol{\theta}_k)\right)$$

操作分两步：
1. **梯度步**：$\tilde{\boldsymbol{\theta}} = \boldsymbol{\theta}_k - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{data}}$（反向传播计算梯度，SGD 更新）
2. **近端步**：$\boldsymbol{\theta}_{k+1} = \mathcal{S}_{\alpha\lambda}(\tilde{\boldsymbol{\theta}})$（软阈值，将小权重置零）

**（c）** 本质区别：

- **SGD + L1 加入 loss 后反传**：L1 项的次梯度为 $\lambda \cdot \text{sign}(\boldsymbol{\theta})$，更新为 $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha(\nabla \mathcal{L} + \lambda \text{sign}(\boldsymbol{\theta}))$，这是**次梯度法**，权重向零移动但**不会精确为零**（需额外截断）
- **ISTA（软阈值）**：更新后应用软阈值，绝对值小于 $\alpha\lambda$ 的权重**精确置零**，产生真正稀疏的权重

ISTA 更能产生真正稀疏的权重，因为软阈值算子是 L1 近端算子的精确解，可将小权重精确归零；而次梯度法只是沿符号方向移动，无法保证精确稀疏性。

**（d）** 实验框架：

```python
import torch
import torch.nn as nn

# 生成数据：真实权重 (1, 0, 0)
torch.manual_seed(0)
X = torch.randn(500, 3)
y = X[:, 0] + 0.1 * torch.randn(500)
lam, lr, epochs = 0.1, 0.01, 200

# 方法一：SGD + L1 加入 loss 后反传（次梯度法）
w1 = nn.Linear(3, 1, bias=False)
opt1 = torch.optim.SGD(w1.parameters(), lr=lr)
for _ in range(epochs):
    loss = 0.5 * ((w1(X).squeeze() - y) ** 2).mean() + lam * w1.weight.abs().sum()
    opt1.zero_grad(); loss.backward(); opt1.step()
print(f"次梯度法权重: {w1.weight.data}")
print(f"稀疏度（|w|<1e-3）: {(w1.weight.data.abs() < 1e-3).float().mean():.2%}")

# 方法二：ISTA（SGD 后接软阈值）
w2 = nn.Linear(3, 1, bias=False)
opt2 = torch.optim.SGD(w2.parameters(), lr=lr)
for _ in range(epochs):
    loss = 0.5 * ((w2(X).squeeze() - y) ** 2).mean()
    opt2.zero_grad(); loss.backward(); opt2.step()
    # 近端步：软阈值
    with torch.no_grad():
        w2.weight.data = torch.sign(w2.weight.data) * \
            torch.clamp(w2.weight.data.abs() - lr * lam, min=0)
print(f"ISTA 权重:   {w2.weight.data}")
print(f"稀疏度（|w|<1e-3）: {(w2.weight.data.abs() < 1e-3).float().mean():.2%}")
```

预期结果：ISTA 的权重中 $w_2, w_3$ 精确为零（或非常接近零），次梯度法的权重小但不为零。
