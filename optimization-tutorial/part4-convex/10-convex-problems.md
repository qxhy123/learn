# 第10章：凸优化问题

> **前置知识**：第3章（凸集与凸函数）、第8章（不等式约束优化）、第9章（对偶理论）
>
> **本章难度**：★★★★☆
>
> **预计学习时间**：6-8 小时

---

## 学习目标

学完本章后，你将能够：

- 写出凸优化问题的标准形式，理解目标函数与可行域的凸性如何保证局部最优即全局最优，并说明约束规范（Slater 条件）对强对偶成立的意义
- 掌握线性规划（LP）的标准形式、几何解释（顶点最优性）以及单纯形法与内点法的基本思路，能将实际问题（如 L1 回归、网络流）建模为 LP
- 识别二次规划（QP）的结构特征，区分凸 QP 与非凸 QP，理解为何支持向量机（SVM）的训练本质上是一个凸 QP 问题
- 理解二阶锥规划（SOCP）对 QP 的推广，掌握鲁棒最优化与二阶锥约束之间的等价关系，并能将范数约束问题改写为 SOCP
- 掌握半定规划（SDP）的矩阵不等式约束形式，了解 LP $\subset$ QP $\subset$ SOCP $\subset$ SDP 的层次包含关系，以及 SDP 在谱范数约束与矩阵补全问题中的应用

---

## 10.1 凸优化问题的标准形式

### 10.1.1 一般形式

**定义 10.1（凸优化问题）**：凸优化问题具有如下结构：

$$
\begin{aligned}
\min_{\mathbf{x}} \quad & f_0(\mathbf{x}) \\
\text{s.t.} \quad & f_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m \\
& \mathbf{a}_j^T \mathbf{x} = b_j, \quad j = 1, \ldots, p
\end{aligned}
$$

其中 $f_0, f_1, \ldots, f_m : \mathbb{R}^n \to \mathbb{R}$ 均为**凸函数**，等式约束是**仿射**的（线性约束）。

**三个核心要素**：

| 要素 | 条件 | 含义 |
|------|------|------|
| 目标函数 | $f_0$ 凸 | 最小化凸函数 |
| 不等式约束 | $f_i$ 凸 | 可行域是凸集（$f_i(\mathbf{x}) \leq 0$ 的下水平集） |
| 等式约束 | 仿射 $A\mathbf{x} = \mathbf{b}$ | 保持可行域的凸性 |

> **为何等式约束必须是仿射的？** 若 $h(\mathbf{x}) = 0$ 中 $h$ 是非线性凸函数，则 $\{h(\mathbf{x}) = 0\}$ 通常不是凸集（例如 $x^2 = 0$ 只是单点，而 $x^2 - 1 = 0$ 是两个孤立点）。仿射等式 $\mathbf{a}^T\mathbf{x} = b$ 对应超平面，是凸集。

### 10.1.2 关键性质

**定理 10.1（局部最优即全局最优）**：凸优化问题的任何局部最优解都是全局最优解。

*证明*：设 $\mathbf{x}^*$ 是局部最优解，假设存在可行点 $\mathbf{z}$ 满足 $f_0(\mathbf{z}) < f_0(\mathbf{x}^*)$。对 $\theta \in (0,1)$，令 $\mathbf{y} = (1-\theta)\mathbf{x}^* + \theta\mathbf{z}$，由可行域的凸性知 $\mathbf{y}$ 可行。由 $f_0$ 的凸性：

$$f_0(\mathbf{y}) \leq (1-\theta)f_0(\mathbf{x}^*) + \theta f_0(\mathbf{z}) < f_0(\mathbf{x}^*)$$

当 $\theta$ 足够小时 $\mathbf{y}$ 在 $\mathbf{x}^*$ 的邻域内，与 $\mathbf{x}^*$ 为局部最优矛盾。$\square$

**定理 10.2（最优性条件）**：若目标函数可微，$\mathbf{x}^*$ 是可行点，则 $\mathbf{x}^*$ 是全局最优解当且仅当

$$\nabla f_0(\mathbf{x}^*)^T (\mathbf{y} - \mathbf{x}^*) \geq 0, \quad \forall \, \mathbf{y} \in \mathcal{F}$$

其中 $\mathcal{F}$ 是可行域。特别地，若 $\mathbf{x}^*$ 是内点（无有效约束），则 $\nabla f_0(\mathbf{x}^*) = \mathbf{0}$。

### 10.1.3 Slater 条件与强对偶

**定义 10.2（Slater 条件）**：凸优化问题满足 **Slater 约束规范**，若存在严格可行点 $\tilde{\mathbf{x}}$：

$$f_i(\tilde{\mathbf{x}}) < 0, \quad i = 1, \ldots, m, \qquad A\tilde{\mathbf{x}} = \mathbf{b}$$

**定理 10.3（强对偶定理）**：若凸优化问题满足 Slater 条件，则强对偶成立：对偶间隙为零，

$$p^* = d^*$$

且对偶最优解 $(\boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$ 存在。此时 KKT 条件不仅是必要条件，也是**充分条件**：

$$
\begin{cases}
f_i(\mathbf{x}^*) \leq 0, \quad A\mathbf{x}^* = \mathbf{b} & \text{（原始可行）} \\
\lambda_i^* \geq 0 & \text{（对偶可行）} \\
\lambda_i^* f_i(\mathbf{x}^*) = 0 & \text{（互补松弛）} \\
\nabla f_0(\mathbf{x}^*) + \sum_i \lambda_i^* \nabla f_i(\mathbf{x}^*) + A^T\boldsymbol{\nu}^* = \mathbf{0} & \text{（梯度条件）}
\end{cases}
$$

### 10.1.4 凸优化问题的层次结构

本章将介绍四类重要的凸优化问题，它们之间有严格的包含关系：

$$\text{LP} \subset \text{QP} \subset \text{SOCP} \subset \text{SDP}$$

```
SDP（半定规划）
  └── SOCP（二阶锥规划）
        └── QP（二次规划）
              └── LP（线性规划）
```

每一层都是更高层的特例，更高层具有更强的表达能力，但求解复杂度也相应增加。所有这些问题都能在多项式时间内求解（内点法）。

---

## 10.2 线性规划（LP）

### 10.2.1 标准形式

**定义 10.3（线性规划）**：线性规划（Linear Programming，LP）的**标准形式**为：

$$
\boxed{
\begin{aligned}
\min_{\mathbf{x}} \quad & \mathbf{c}^T \mathbf{x} \\
\text{s.t.} \quad & A\mathbf{x} = \mathbf{b} \\
& \mathbf{x} \succeq \mathbf{0}
\end{aligned}
}
$$

其中 $\mathbf{c} \in \mathbb{R}^n$（目标系数），$A \in \mathbb{R}^{m \times n}$，$\mathbf{b} \in \mathbb{R}^m$，$\mathbf{x} \succeq \mathbf{0}$ 表示各分量非负。

**不等式形式**（等价，通过引入松弛变量可互转）：

$$
\begin{aligned}
\min_{\mathbf{x}} \quad & \mathbf{c}^T \mathbf{x} \\
\text{s.t.} \quad & A\mathbf{x} \preceq \mathbf{b}
\end{aligned}
$$

**凸性验证**：$f_0(\mathbf{x}) = \mathbf{c}^T\mathbf{x}$ 是仿射函数（既凸又凹）；约束 $A\mathbf{x} \preceq \mathbf{b}$ 是仿射不等式，可行域（多面体）是凸集。LP 是凸优化问题的最简单情形。

### 10.2.2 几何解释

LP 的可行域是一个**多面体**（有限个半空间的交）。目标函数 $\mathbf{c}^T\mathbf{x} = \text{const}$ 的等值面是超平面，最优解位于沿 $-\mathbf{c}$ 方向"推进"这些超平面时最后碰到多面体的点。

**定理 10.4（顶点最优性）**：若 LP 有最优解，则在**顶点**（极点）处取到最优值。

*直觉*：若最优解在多面体内部或边的非端点，则沿某个方向仍可改善目标值，直至遇到约束边界。

**退化与无界情况**：
- 若可行域无界且 $\mathbf{c}$ 指向无界方向：LP 无界（$p^* = -\infty$）
- 若可行域为空：LP 不可行
- 若多个顶点都是最优：最优解集是一个面（但顶点仍是最优解）

### 10.2.3 常见应用形式

**例 10.1（L1 范数最小化）**：

$$\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_1 = \min_{\mathbf{x}} \sum_i |(\mathbf{A}\mathbf{x} - \mathbf{b})_i|$$

引入辅助变量 $\mathbf{t} \in \mathbb{R}^m$，等价 LP：

$$
\begin{aligned}
\min_{\mathbf{x}, \mathbf{t}} \quad & \mathbf{1}^T \mathbf{t} \\
\text{s.t.} \quad & -\mathbf{t} \preceq A\mathbf{x} - \mathbf{b} \preceq \mathbf{t}
\end{aligned}
$$

L1 损失对异常值鲁棒（稀疏残差），是鲁棒回归的常用工具。

**例 10.2（L∞ 范数最小化，Chebyshev 逼近）**：

$$\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_\infty = \min_{\mathbf{x}} \max_i |(A\mathbf{x} - \mathbf{b})_i|$$

引入标量 $t$，等价 LP：

$$
\begin{aligned}
\min_{\mathbf{x}, t} \quad & t \\
\text{s.t.} \quad & -t \cdot \mathbf{1} \preceq A\mathbf{x} - \mathbf{b} \preceq t \cdot \mathbf{1}
\end{aligned}
$$

**例 10.3（LP 对偶：零和博弈）**：

最大化最小收益的混合策略问题（极大极小定理）等价于求解一个 LP，这是博弈论与 LP 对偶理论的经典联系。

### 10.2.4 LP 的对偶问题

标准形式 LP 的 Lagrangian 对偶问题为：

$$
\begin{aligned}
\max_{\boldsymbol{\lambda}, \boldsymbol{\nu}} \quad & -\mathbf{b}^T\boldsymbol{\lambda} + \mathbf{b}^T\boldsymbol{\nu} \\
\text{s.t.} \quad & A^T\boldsymbol{\lambda} - A^T\boldsymbol{\nu} + \mathbf{c} \succeq \mathbf{0} \\
& \boldsymbol{\lambda} \succeq \mathbf{0}
\end{aligned}
$$

对于不等式形式的 LP $\min\{\mathbf{c}^T\mathbf{x} \mid A\mathbf{x} \preceq \mathbf{b}\}$，对偶为：

$$\max_{\boldsymbol{\lambda} \succeq \mathbf{0}} \; -\mathbf{b}^T\boldsymbol{\lambda} \quad \text{s.t.} \quad A^T\boldsymbol{\lambda} + \mathbf{c} = \mathbf{0}$$

**LP 强对偶**：Slater 条件对 LP 自动满足（只要原始可行域非空且问题有界），因此 LP 原始与对偶有相同最优值 $p^* = d^*$。

### 10.2.5 cvxpy 求解示例

```python
import cvxpy as cp
import numpy as np

# ============================================================
# 示例 1：标准 LP（最大覆盖问题）
# ============================================================

# 问题：资源分配
# 最大化利润：3x1 + 5x2
# 约束：x1 <= 4, 2x2 <= 12, 3x1 + 5x2 <= 25, x1,x2 >= 0

x = cp.Variable(2, nonneg=True)
c = np.array([-3.0, -5.0])   # 取负号，将最大化转为最小化
A = np.array([[1, 0],
              [0, 2],
              [3, 5]])
b = np.array([4.0, 12.0, 25.0])

prob = cp.Problem(cp.Minimize(c @ x), [A @ x <= b])
prob.solve(solver=cp.GLPK)

print("=== LP 资源分配 ===")
print(f"最优解: x = {x.value}")
print(f"最优利润: {-prob.value:.4f}")

# ============================================================
# 示例 2：L1 回归（鲁棒回归）
# ============================================================

np.random.seed(42)
n_samples, n_features = 50, 3
A_reg = np.random.randn(n_samples, n_features)
x_true = np.array([1.0, -2.0, 0.5])
noise = np.random.randn(n_samples) * 0.1
# 引入异常值
outliers = np.random.choice(n_samples, 5, replace=False)
noise[outliers] += 10.0
b_reg = A_reg @ x_true + noise

# L1 回归（鲁棒，等价 LP）
x_l1 = cp.Variable(n_features)
t = cp.Variable(n_samples)
prob_l1 = cp.Problem(
    cp.Minimize(cp.sum(t)),
    [A_reg @ x_l1 - b_reg <= t,
     -(A_reg @ x_l1 - b_reg) <= t]
)
prob_l1.solve(solver=cp.GLPK)

# L2 回归（普通最小二乘，受异常值影响）
x_l2 = cp.Variable(n_features)
prob_l2 = cp.Problem(cp.Minimize(cp.sum_squares(A_reg @ x_l2 - b_reg)))
prob_l2.solve()

print("\n=== L1 vs L2 回归（含异常值）===")
print(f"真实参数:    {x_true}")
print(f"L1 回归估计: {x_l1.value.round(4)}")
print(f"L2 回归估计: {x_l2.value.round(4)}")
print(f"L1 误差（更小）: {np.linalg.norm(x_l1.value - x_true):.4f}")
print(f"L2 误差（更大）: {np.linalg.norm(x_l2.value - x_true):.4f}")

# ============================================================
# 示例 3：Chebyshev 中心（L∞ 最小化）
# ============================================================
# 求多面体 {x: Ax <= b} 的 Chebyshev 中心（半径最大的内切球）

A_ch = np.array([[ 1,  0],
                 [-1,  0],
                 [ 0,  1],
                 [ 0, -1],
                 [ 1,  1]])
b_ch = np.array([2.0, 2.0, 3.0, 3.0, 4.0])

x_c = cp.Variable(2)
r   = cp.Variable()  # 半径

# 约束：对每个超平面 a_i^T x <= b_i，内切球中心需满足距离 >= r
# 即 a_i^T x_c + ||a_i|| * r <= b_i
rows = A_ch.shape[0]
constraints_ch = []
for i in range(rows):
    a_i = A_ch[i]
    constraints_ch.append(a_i @ x_c + np.linalg.norm(a_i) * r <= b_ch[i])

prob_ch = cp.Problem(cp.Maximize(r), constraints_ch)
prob_ch.solve(solver=cp.GLPK)

print("\n=== Chebyshev 中心（LP） ===")
print(f"Chebyshev 中心: {x_c.value.round(4)}")
print(f"最大内切球半径: {r.value:.4f}")
```

---

## 10.3 二次规划（QP）

### 10.3.1 标准形式

**定义 10.4（二次规划）**：二次规划（Quadratic Programming，QP）的标准形式为：

$$
\boxed{
\begin{aligned}
\min_{\mathbf{x}} \quad & \frac{1}{2}\mathbf{x}^T P \mathbf{x} + \mathbf{q}^T \mathbf{x} + r \\
\text{s.t.} \quad & G\mathbf{x} \preceq \mathbf{h} \\
& A\mathbf{x} = \mathbf{b}
\end{aligned}
}
$$

其中 $P \in \mathbb{S}^n$（对称矩阵），$G \in \mathbb{R}^{m \times n}$，$A \in \mathbb{R}^{p \times n}$。

**凸性条件**：QP 是凸优化问题 $\iff$ $P \succeq 0$（正半定）。

| $P$ 的性质 | 问题类型 | 特点 |
|-----------|---------|------|
| $P \succ 0$（正定）| 严格凸 QP | 唯一全局最小值 |
| $P \succeq 0$（半正定）| 凸 QP | 全局最小值存在（可能不唯一）|
| $P$ 有负特征值 | 非凸 QP | NP-hard，求解困难 |

当 $P = 0$ 时，QP 退化为 LP。因此 **LP $\subset$ QP**（LP 是 QP 的特例）。

### 10.3.2 二次约束二次规划（QCQP）

当约束也包含二次项时，得到**二次约束二次规划**（Quadratically Constrained QP，QCQP）：

$$
\begin{aligned}
\min_{\mathbf{x}} \quad & \frac{1}{2}\mathbf{x}^T P_0 \mathbf{x} + \mathbf{q}_0^T \mathbf{x} \\
\text{s.t.} \quad & \frac{1}{2}\mathbf{x}^T P_i \mathbf{x} + \mathbf{q}_i^T \mathbf{x} + r_i \leq 0, \quad i = 1, \ldots, m
\end{aligned}
$$

**凸性条件**：$P_0, P_1, \ldots, P_m \succeq 0$（所有矩阵正半定）。

### 10.3.3 应用：支持向量机（SVM）

SVM 的训练问题是凸 QP 的经典应用。

**硬间隔 SVM 原问题**：

给定线性可分数据 $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$（$y_i \in \{-1, +1\}$），最大化分类间隔：

$$
\begin{aligned}
\min_{\mathbf{w}, b} \quad & \frac{1}{2}\|\mathbf{w}\|^2 \\
\text{s.t.} \quad & y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, N
\end{aligned}
$$

这是一个凸 QP（$P = I \succ 0$，约束线性）。

**软间隔 SVM**（允许误分类，引入松弛变量 $\boldsymbol{\xi}$）：

$$
\begin{aligned}
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad & \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^N \xi_i \\
\text{s.t.} \quad & y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
\end{aligned}
$$

仍然是凸 QP，$C > 0$ 控制正则化强度。

**SVM 对偶问题**（核方法的基础）：

$$
\begin{aligned}
\max_{\boldsymbol{\alpha}} \quad & \sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j \\
\text{s.t.} \quad & \sum_{i=1}^N \alpha_i y_i = 0 \\
& 0 \leq \alpha_i \leq C
\end{aligned}
$$

其中 $\mathbf{x}_i^T\mathbf{x}_j$ 可替换为核函数 $K(\mathbf{x}_i, \mathbf{x}_j)$，得到非线性 SVM。

### 10.3.4 应用：投资组合优化（Markowitz 模型）

给定 $n$ 种资产的期望收益向量 $\boldsymbol{\mu}$ 和协方差矩阵 $\Sigma \succeq 0$，权重向量 $\mathbf{w}$：

$$
\begin{aligned}
\min_{\mathbf{w}} \quad & \mathbf{w}^T \Sigma \mathbf{w} \quad \text{（最小化方差/风险）} \\
\text{s.t.} \quad & \boldsymbol{\mu}^T \mathbf{w} \geq r_{\min} \quad \text{（期望收益不低于 $r_{\min}$）} \\
& \mathbf{1}^T \mathbf{w} = 1, \quad \mathbf{w} \succeq \mathbf{0}
\end{aligned}
$$

这是带等式约束和不等式约束的凸 QP（$\Sigma \succeq 0$）。

### 10.3.5 cvxpy 求解示例

```python
import cvxpy as cp
import numpy as np

# ============================================================
# 示例 1：硬间隔 SVM（凸 QP）
# ============================================================

np.random.seed(0)
# 生成线性可分数据
N = 40
X_pos = np.random.randn(N//2, 2) + np.array([2, 2])   # 正类
X_neg = np.random.randn(N//2, 2) + np.array([-2, -2])  # 负类
X_data = np.vstack([X_pos, X_neg])
y_data = np.hstack([np.ones(N//2), -np.ones(N//2)])

n_features = X_data.shape[1]
w = cp.Variable(n_features)
b = cp.Variable()

# 硬间隔 SVM：最小化 ||w||^2 / 2，约束 y_i(w^T x_i + b) >= 1
prob_svm = cp.Problem(
    cp.Minimize(0.5 * cp.sum_squares(w)),
    [cp.multiply(y_data, X_data @ w + b) >= 1]
)
prob_svm.solve(solver=cp.OSQP)

margin = 2.0 / np.linalg.norm(w.value)
print("=== 硬间隔 SVM（凸 QP）===")
print(f"权重 w = {w.value.round(4)}")
print(f"偏置 b = {b.value:.4f}")
print(f"间隔宽度 = {margin:.4f}")

# 验证所有点满足约束
violations = np.sum(y_data * (X_data @ w.value + b.value) < 1 - 1e-5)
print(f"约束违反数（应为 0）: {violations}")

# ============================================================
# 示例 2：软间隔 SVM（含噪声数据）
# ============================================================

# 引入噪声（使数据不可线性分）
np.random.seed(1)
X_noisy = np.vstack([X_pos, X_neg, np.random.randn(4, 2)])
y_noisy = np.hstack([np.ones(N//2), -np.ones(N//2),
                     np.array([1, 1, -1, -1])])
# 故意混入异类噪声
y_noisy[-4] = -y_noisy[-4]
y_noisy[-3] = -y_noisy[-3]

N2 = len(y_noisy)
w2  = cp.Variable(n_features)
b2  = cp.Variable()
xi  = cp.Variable(N2, nonneg=True)  # 松弛变量

C_val = 1.0   # 惩罚强度
prob_soft = cp.Problem(
    cp.Minimize(0.5 * cp.sum_squares(w2) + C_val * cp.sum(xi)),
    [cp.multiply(y_noisy, X_noisy @ w2 + b2) >= 1 - xi]
)
prob_soft.solve(solver=cp.OSQP)

print(f"\n=== 软间隔 SVM（C={C_val}）===")
print(f"权重 w = {w2.value.round(4)}")
print(f"松弛变量非零个数: {np.sum(xi.value > 1e-4)}")
print(f"目标函数值: {prob_soft.value:.4f}")

# ============================================================
# 示例 3：Markowitz 投资组合优化（凸 QP）
# ============================================================

np.random.seed(42)
n_assets = 5
# 随机生成期望收益和协方差矩阵
mu   = np.array([0.12, 0.10, 0.15, 0.08, 0.13])
# 构造正定协方差矩阵
A_tmp = np.random.randn(n_assets, n_assets)
Sigma = A_tmp @ A_tmp.T / n_assets + np.eye(n_assets) * 0.01

r_min_values = [0.09, 0.10, 0.11, 0.12]
print("\n=== Markowitz 投资组合优化 ===")
print(f"{'最低期望收益':>12} {'组合方差':>10} {'组合收益':>10}")
for r_min in r_min_values:
    w_port = cp.Variable(n_assets, nonneg=True)
    prob_port = cp.Problem(
        cp.Minimize(cp.quad_form(w_port, Sigma)),
        [mu @ w_port >= r_min,
         cp.sum(w_port) == 1]
    )
    prob_port.solve(solver=cp.OSQP)
    if prob_port.status == 'optimal':
        variance = w_port.value @ Sigma @ w_port.value
        ret = mu @ w_port.value
        print(f"{r_min:>12.2f} {variance:>10.6f} {ret:>10.4f}")
    else:
        print(f"{r_min:>12.2f}   不可行")
```

---

## 10.4 二阶锥规划（SOCP）

### 10.4.1 二阶锥的定义

**定义 10.5（二阶锥）**：$n+1$ 维**二阶锥**（Second-Order Cone，也称冰淇淋锥）定义为：

$$\mathcal{K}_n = \left\{(\mathbf{x}, t) \in \mathbb{R}^n \times \mathbb{R} \;\middle|\; \|\mathbf{x}\|_2 \leq t\right\}$$

**几何直觉**：$n=2$ 时这是三维空间中的圆锥面（及其内部）——截面为圆，顶点在原点。它是**凸锥**（对 $\lambda \geq 0$ 封闭，对凸组合封闭）。

**推广：旋转二阶锥**：

$$\mathcal{K}_n^r = \left\{(\mathbf{x}, u, v) \in \mathbb{R}^n \times \mathbb{R} \times \mathbb{R} \;\middle|\; \|\mathbf{x}\|_2^2 \leq uv,\; u, v \geq 0\right\}$$

旋转二阶锥可通过坐标变换从标准二阶锥得到。

### 10.4.2 SOCP 标准形式

**定义 10.6（二阶锥规划）**：二阶锥规划（Second-Order Cone Programming，SOCP）的标准形式为：

$$
\boxed{
\begin{aligned}
\min_{\mathbf{x}} \quad & \mathbf{f}^T \mathbf{x} \\
\text{s.t.} \quad & \|A_i \mathbf{x} + \mathbf{b}_i\|_2 \leq \mathbf{c}_i^T \mathbf{x} + d_i, \quad i = 1, \ldots, m \\
& F\mathbf{x} = \mathbf{g}
\end{aligned}
}
$$

其中 $A_i \in \mathbb{R}^{n_i \times n}$，$\mathbf{b}_i \in \mathbb{R}^{n_i}$，$\mathbf{c}_i \in \mathbb{R}^n$，$d_i \in \mathbb{R}$。约束 $\|A_i\mathbf{x}+\mathbf{b}_i\|_2 \leq \mathbf{c}_i^T\mathbf{x}+d_i$ 等价于 $(A_i\mathbf{x}+\mathbf{b}_i, \mathbf{c}_i^T\mathbf{x}+d_i) \in \mathcal{K}_{n_i}$。

**包含关系验证**：

- **LP 是 SOCP 特例**：取 $A_i = 0$，$\mathbf{b}_i = 0$，得线性约束 $0 \leq \mathbf{c}_i^T\mathbf{x}+d_i$。
- **QP 是 SOCP 特例**：$\frac{1}{2}\mathbf{x}^TP\mathbf{x} \leq t$ 等价于 $\|P^{1/2}\mathbf{x}\|_2^2 \leq 2t$（旋转二阶锥约束）。具体地，凸 QP 可改写为 SOCP：

$$\min_{\mathbf{x}, t} \; t + \mathbf{q}^T\mathbf{x} \quad \text{s.t.} \quad \|P^{1/2}\mathbf{x}\|_2 \leq \sqrt{2t}, \; G\mathbf{x} \preceq \mathbf{h}$$

### 10.4.3 应用：鲁棒线性规划

**问题背景**：数据矩阵 $A$ 存在不确定性，每行 $\mathbf{a}_i^T$ 属于椭球不确定集 $\mathcal{U}_i$：

$$\mathcal{U}_i = \{\bar{\mathbf{a}}_i + P_i\mathbf{u} \mid \|\mathbf{u}\|_2 \leq 1\}$$

其中 $\bar{\mathbf{a}}_i$ 是名义值，$P_i$ 描述不确定性形状。

**鲁棒约束**：对所有 $\mathbf{a}_i \in \mathcal{U}_i$ 均满足 $\mathbf{a}_i^T\mathbf{x} \leq b_i$，等价于：

$$\bar{\mathbf{a}}_i^T\mathbf{x} + \max_{\|\mathbf{u}\|_2 \leq 1} \mathbf{u}^T P_i^T \mathbf{x} \leq b_i$$

$$\Leftrightarrow \quad \bar{\mathbf{a}}_i^T\mathbf{x} + \|P_i^T\mathbf{x}\|_2 \leq b_i$$

这是一个**二阶锥约束**！因此，具有椭球不确定性的鲁棒 LP 等价于 SOCP，代价仅是每个约束增加一个范数项。

### 10.4.4 应用：最优化椭球估计

**最小体积椭球（MVEE）**：包含给定点集 $\{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$ 的最小体积椭球可表示为 SOCP 或 SDP（见 10.5 节）。

**Lorentz 锥约束的几何意义**：$\|\mathbf{x}\|_2 \leq t$ 的几何意义是点 $(\mathbf{x}, t)$ 位于 Lorentz 锥内，是欧氏球 $\|\mathbf{x}\|_2 \leq t$ 在高一维空间的表示。

### 10.4.5 cvxpy 求解示例

```python
import cvxpy as cp
import numpy as np

# ============================================================
# 示例 1：鲁棒线性规划（转化为 SOCP）
# ============================================================

np.random.seed(7)
n_vars = 3
m_con  = 4

# 名义约束矩阵和向量
A_nom = np.random.randn(m_con, n_vars)
b_nom = np.abs(np.random.randn(m_con)) + 2.0
c_obj = np.random.randn(n_vars)

# 不确定性矩阵（对角缩放，简化起见取各行独立椭球）
P_unc = [np.diag(np.abs(np.random.randn(n_vars)) * 0.1)
         for _ in range(m_con)]

x_robust = cp.Variable(n_vars)

# 鲁棒约束：a_i^T x + ||P_i^T x|| <= b_i
constraints_robust = []
for i in range(m_con):
    constraints_robust.append(
        A_nom[i] @ x_robust + cp.norm(P_unc[i].T @ x_robust, 2) <= b_nom[i]
    )

prob_robust = cp.Problem(cp.Minimize(c_obj @ x_robust), constraints_robust)
prob_robust.solve(solver=cp.ECOS)

# 名义 LP（无鲁棒性）
x_nominal = cp.Variable(n_vars)
prob_nominal = cp.Problem(
    cp.Minimize(c_obj @ x_nominal),
    [A_nom @ x_nominal <= b_nom]
)
prob_nominal.solve(solver=cp.ECOS)

print("=== 鲁棒 LP vs 名义 LP ===")
print(f"名义 LP 最优值:  {prob_nominal.value:.4f}")
print(f"鲁棒 SOCP 最优值: {prob_robust.value:.4f}  （更保守，代价更高）")
print(f"名义解: {x_nominal.value.round(4)}")
print(f"鲁棒解: {x_robust.value.round(4)}")

# 验证：用鲁棒解，在不确定性最差情形下的约束满足情况
print("\n最差情形约束检验（鲁棒解）:")
for i in range(m_con):
    worst = A_nom[i] @ x_robust.value + np.linalg.norm(P_unc[i].T @ x_robust.value)
    print(f"  约束 {i+1}: {worst:.4f} <= {b_nom[i]:.4f} -> {'满足' if worst <= b_nom[i]+1e-5 else '违反'}")

# ============================================================
# 示例 2：范数约束下的最小二乘（SOCP）
# ============================================================

# min ||Ax - b||_2
# s.t. ||x||_2 <= r  (L2 约束，可用 Tikhonov 正则化代替，也可直接写 SOCP)

np.random.seed(3)
n_s, n_f = 30, 5
A_ls = np.random.randn(n_s, n_f)
b_ls = np.random.randn(n_s)
r_bound = 1.5  # 约束球半径

x_socp = cp.Variable(n_f)
prob_socp = cp.Problem(
    cp.Minimize(cp.norm(A_ls @ x_socp - b_ls, 2)),
    [cp.norm(x_socp, 2) <= r_bound]
)
prob_socp.solve(solver=cp.ECOS)

# 对比：无约束最小二乘
x_ls  = np.linalg.lstsq(A_ls, b_ls, rcond=None)[0]

print("\n=== 约束 L2 最小二乘（SOCP） ===")
print(f"无约束 LS: ||x||={np.linalg.norm(x_ls):.4f}, 残差={np.linalg.norm(A_ls@x_ls-b_ls):.4f}")
print(f"约束 SOCP: ||x||={np.linalg.norm(x_socp.value):.4f} (<={r_bound}), 残差={np.linalg.norm(A_ls@x_socp.value-b_ls):.4f}")

# ============================================================
# 示例 3：多任务鲁棒估计（混合 L2/L-inf 范数，SOCP）
# ============================================================

# min sum_i t_i  s.t.  ||A_i x - b_i||_2 <= t_i （二阶锥约束组）

n_tasks = 3
results = []
x_mt = cp.Variable(n_f)
t_mt = cp.Variable(n_tasks, nonneg=True)
cons_mt = []

np.random.seed(9)
task_data = [(np.random.randn(10, n_f), np.random.randn(10)) for _ in range(n_tasks)]

for k, (Ak, bk) in enumerate(task_data):
    cons_mt.append(cp.norm(Ak @ x_mt - bk, 2) <= t_mt[k])

prob_mt = cp.Problem(cp.Minimize(cp.sum(t_mt)), cons_mt)
prob_mt.solve(solver=cp.ECOS)

print("\n=== 多任务鲁棒估计（SOCP） ===")
print(f"共享参数 x = {x_mt.value.round(3)}")
for k in range(n_tasks):
    Ak, bk = task_data[k]
    residual = np.linalg.norm(Ak @ x_mt.value - bk)
    print(f"  任务 {k+1} 残差: {residual:.4f}  (t_{k+1}={t_mt.value[k]:.4f})")
```

---

## 10.5 半定规划（SDP）

### 10.5.1 正半定锥

**定义 10.7（正半定锥）**：$n$ 阶实对称矩阵构成的正半定锥：

$$\mathbb{S}^n_+ = \{X \in \mathbb{R}^{n \times n} \mid X = X^T,\; X \succeq 0\}$$

其中 $X \succeq 0$ 表示 $X$ 正半定，等价于：所有特征值 $\geq 0$，或对所有向量 $\mathbf{v}$，$\mathbf{v}^T X \mathbf{v} \geq 0$。

**内积**：$\mathbb{S}^n$ 上的自然内积为迹内积：

$$\langle X, Y \rangle = \text{tr}(X^T Y) = \text{tr}(XY) = \sum_{i,j} X_{ij}Y_{ij}$$

### 10.5.2 SDP 标准形式

**定义 10.8（半定规划）**：半定规划（Semidefinite Programming，SDP）有两种等价形式。

**矩阵不等式形式**（标准形式之一）：

$$
\boxed{
\begin{aligned}
\min_{X \in \mathbb{S}^n} \quad & \text{tr}(C X) \\
\text{s.t.} \quad & \text{tr}(A_i X) = b_i, \quad i = 1, \ldots, m \\
& X \succeq 0
\end{aligned}
}
$$

其中 $C, A_1, \ldots, A_m \in \mathbb{S}^n$，$\mathbf{b} \in \mathbb{R}^m$，决策变量是**矩阵** $X \in \mathbb{S}^n_+$。

**向量变量形式**（另一标准形式）：

$$
\begin{aligned}
\min_{\mathbf{x} \in \mathbb{R}^n} \quad & \mathbf{c}^T\mathbf{x} \\
\text{s.t.} \quad & F_0 + x_1 F_1 + \cdots + x_n F_n \succeq 0 \\
& A\mathbf{x} = \mathbf{b}
\end{aligned}
$$

其中 $F_0, F_1, \ldots, F_n \in \mathbb{S}^m$，约束 $F(\mathbf{x}) = F_0 + \sum_i x_i F_i \succeq 0$ 称为**线性矩阵不等式（LMI）**。

**SDP 是最一般的锥规划**（本章范围内）：

$$\text{LP} \subset \text{QCQP} \subset \text{SOCP} \subset \text{SDP}$$

| 问题类型 | 约束锥 | 标量/矩阵 |
|---------|-------|---------|
| LP | $\mathbb{R}^n_+$（非负象限）| 标量变量 |
| SOCP | $\mathcal{K}_n$（二阶锥）| 标量变量 |
| SDP | $\mathbb{S}^n_+$（正半定锥）| 矩阵变量 |

**SOCP 是 SDP 特例**：约束 $\|\mathbf{x}\|_2 \leq t$ 等价于 LMI：

$$\begin{pmatrix} tI & \mathbf{x} \\ \mathbf{x}^T & t \end{pmatrix} \succeq 0$$

（由 Schur 补引理，该矩阵半正定 $\iff t^2 - \mathbf{x}^T(tI)^{-1}\mathbf{x} = t^2 - \|\mathbf{x}\|^2/t \cdot t = t^2 - \|\mathbf{x}\|^2 \geq 0$，即 $\|\mathbf{x}\|_2 \leq t$）。

### 10.5.3 Schur 补引理

**引理 10.1（Schur 补）**：分块矩阵

$$M = \begin{pmatrix} A & B \\ B^T & C \end{pmatrix}$$

设 $A \succ 0$，则 $M \succeq 0$ $\iff$ $C - B^T A^{-1} B \succeq 0$（$C$ 关于 $A$ 的 Schur 补非负定）。

**用途**：将非线性（含逆矩阵）约束改写为 LMI，是将 QP/SOCP 问题嵌入 SDP 的主要工具。

**例 10.4（二次不等式改写为 LMI）**：约束 $\mathbf{x}^T P \mathbf{x} \leq t$（$P \succ 0$）等价于：

$$\begin{pmatrix} t & \mathbf{x}^T \\ \mathbf{x} & P^{-1} \end{pmatrix} \succeq 0 \quad \Leftrightarrow \quad \begin{pmatrix} tP & P\mathbf{x} \\ \mathbf{x}^TP & t \end{pmatrix} \succeq 0$$

（乘以正定矩阵后保持半正定性）

### 10.5.4 应用：矩阵范数约束

**谱范数**（最大奇异值）：$\|A\|_2 = \sigma_{\max}(A)$，$\|A\|_2 \leq t$ 等价于 LMI：

$$\begin{pmatrix} tI & A \\ A^T & tI \end{pmatrix} \succeq 0$$

**核范数**（奇异值之和）：$\|A\|_* = \sum_i \sigma_i(A)$，核范数最小化（矩阵补全中的凸松弛）等价于 SDP：

$$\|A\|_* \leq t \quad \Leftrightarrow \quad \exists W_1, W_2 \succeq 0 : \begin{pmatrix} W_1 & A \\ A^T & W_2 \end{pmatrix} \succeq 0, \; \text{tr}(W_1) + \text{tr}(W_2) \leq 2t$$

### 10.5.5 应用：最大特征值最小化

**问题**：$\min_{\mathbf{x}} \lambda_{\max}(F_0 + \sum_i x_i F_i)$，其中 $F_i \in \mathbb{S}^m$。

这等价于 SDP（引入 $t = \lambda_{\max}$）：

$$
\begin{aligned}
\min_{\mathbf{x}, t} \quad & t \\
\text{s.t.} \quad & F_0 + \sum_i x_i F_i \preceq t I
\end{aligned}
$$

即 $tI - F(\mathbf{x}) \succeq 0$，这是一个 LMI 约束的线性目标，标准 SDP。

### 10.5.6 cvxpy 求解示例

```python
import cvxpy as cp
import numpy as np

# ============================================================
# 示例 1：基本 SDP——最大特征值最小化
# ============================================================

np.random.seed(5)
n_mat = 4
# 生成对称矩阵
def rand_sym(n):
    A = np.random.randn(n, n)
    return (A + A.T) / 2

F0 = rand_sym(n_mat)
F1 = rand_sym(n_mat)
F2 = rand_sym(n_mat)

x_sdp = cp.Variable(2)
# F(x) = F0 + x1*F1 + x2*F2
F_x = F0 + x_sdp[0] * F1 + x_sdp[1] * F2

# min lambda_max(F(x)) 等价于 min t s.t. t*I - F(x) >= 0
t_eig = cp.Variable()
prob_eig = cp.Problem(
    cp.Minimize(t_eig),
    [t_eig * np.eye(n_mat) - F_x >> 0]   # >> 表示半正定（PSD 约束）
)
prob_eig.solve(solver=cp.SCS)

print("=== 最大特征值最小化（SDP） ===")
print(f"最优 x = {x_sdp.value.round(4)}")
print(f"最小化后的最大特征值 = {t_eig.value:.4f}")
F_opt = F0 + x_sdp.value[0]*F1 + x_sdp.value[1]*F2
print(f"验证：lambda_max(F(x*)) = {np.linalg.eigvalsh(F_opt).max():.4f}")

# ============================================================
# 示例 2：核范数最小化（矩阵补全的凸松弛，SDP）
# ============================================================
# 已知部分矩阵元素（Omega 处），恢复低秩矩阵
# min  ||X||_*  s.t.  X_ij = M_ij, (i,j) in Omega

np.random.seed(11)
m_rows, n_cols = 5, 5
rank_true = 2

# 生成低秩真实矩阵
U = np.random.randn(m_rows, rank_true)
V = np.random.randn(n_cols, rank_true)
M_true = U @ V.T

# 随机观测：观测 60% 的元素
obs_mask = np.random.rand(m_rows, n_cols) < 0.6
print(f"\n=== 核范数最小化（矩阵补全） ===")
print(f"真实矩阵秩: {rank_true}，观测比例: {obs_mask.mean():.0%}")

X_mc = cp.Variable((m_rows, n_cols))
# 核范数 = 最小化奇异值之和
prob_mc = cp.Problem(
    cp.Minimize(cp.normNuc(X_mc)),
    [X_mc[obs_mask] == M_true[obs_mask]]
)
prob_mc.solve(solver=cp.SCS)

if X_mc.value is not None:
    singular_vals = np.linalg.svd(X_mc.value, compute_uv=False)
    effective_rank = np.sum(singular_vals > 1e-3)
    recovery_err  = np.linalg.norm(X_mc.value - M_true, 'fro') / np.linalg.norm(M_true, 'fro')
    print(f"恢复矩阵的有效秩: {effective_rank}（期望 {rank_true}）")
    print(f"相对恢复误差: {recovery_err:.4f}")
    print(f"前 {rank_true+1} 个奇异值: {singular_vals[:rank_true+1].round(4)}")

# ============================================================
# 示例 3：SDP 松弛——最大割问题（MAX-CUT）
# ============================================================
# MAX-CUT 是 NP-hard 问题，其 SDP 松弛（Goemans-Williamson 算法）
# 提供 0.878 近似比，是凸优化在组合优化中的经典应用

# 构造简单图：5 个节点，带权邻接矩阵
np.random.seed(2)
n_nodes = 5
# 随机对称权重矩阵（非负权）
W_adj = np.abs(np.random.randn(n_nodes, n_nodes))
W_adj = (W_adj + W_adj.T) / 2
np.fill_diagonal(W_adj, 0)

# SDP 松弛：
# MAX-CUT = max 0.25 * sum_{i,j} w_{ij}(1 - y_i*y_j)
# 等价于 max 0.25 * tr(L*Y) s.t. Y_ii = 1, Y >> 0
# 其中 L 是 Laplacian 矩阵
D = np.diag(W_adj.sum(axis=1))
L_lap = D - W_adj   # Laplacian

Y_sdp = cp.Variable((n_nodes, n_nodes), symmetric=True)
constraints_mc = [Y_sdp >> 0]
constraints_mc += [Y_sdp[i, i] == 1.0 for i in range(n_nodes)]

prob_cut = cp.Problem(
    cp.Maximize(0.25 * cp.trace(L_lap @ Y_sdp)),
    constraints_mc
)
prob_cut.solve(solver=cp.SCS)

print(f"\n=== MAX-CUT SDP 松弛 ===")
print(f"SDP 上界（松弛最优值）: {prob_cut.value:.4f}")

# Goemans-Williamson 随机舍入：获得近似割
if Y_sdp.value is not None:
    # 分解 Y* = V^T V
    eigvals, eigvecs = np.linalg.eigh(Y_sdp.value)
    eigvals = np.maximum(eigvals, 0)
    V_gw = eigvecs @ np.diag(np.sqrt(eigvals))

    best_cut = 0
    np.random.seed(0)
    for _ in range(100):
        r = np.random.randn(n_nodes)
        r = r / np.linalg.norm(r)
        y_round = np.sign(V_gw @ r)
        cut_val = 0.25 * np.trace(L_lap @ np.outer(y_round, y_round))
        best_cut = max(best_cut, cut_val)

    print(f"GW 随机舍入割值: {best_cut:.4f}")
    print(f"近似比: {best_cut/prob_cut.value:.4f}（理论保证 >= 0.878）")
```

---

## 本章小结

| 问题类型 | 标准形式（核心特征）| 约束锥 | 层次关系 | 典型应用 |
|---------|-------------------|-------|---------|---------|
| **LP** | $\min \mathbf{c}^T\mathbf{x}$，$A\mathbf{x} \preceq \mathbf{b}$，线性目标+线性约束 | $\mathbb{R}^n_+$ | 最基础 | L1/L∞ 回归、网络流、资源分配 |
| **QP** | $\min \frac{1}{2}\mathbf{x}^TP\mathbf{x}+\mathbf{q}^T\mathbf{x}$，$P \succeq 0$，线性约束 | $\mathbb{R}^n_+$ | $\supset$ LP | SVM、岭回归、投资组合优化 |
| **SOCP** | 线性目标，$\|A_i\mathbf{x}+\mathbf{b}_i\| \leq \mathbf{c}_i^T\mathbf{x}+d_i$ | $\mathcal{K}_n$（二阶锥）| $\supset$ QP | 鲁棒优化、范数约束、Lorentz 锥 |
| **SDP** | $\min \text{tr}(CX)$，$X \succeq 0$，LMI 约束 | $\mathbb{S}^n_+$（正半定锥）| $\supset$ SOCP | MAX-CUT、矩阵补全、谱范数约束 |

**求解复杂度（内点法）**：

| 问题 | 每次迭代复杂度 | 总迭代次数 |
|------|--------------|---------|
| LP（$n$ 变量，$m$ 约束）| $O((m+n)n^2)$ | $O(\sqrt{m+n})$ |
| QP | $O((m+n)n^2)$ | $O(\sqrt{m+n})$ |
| SOCP（$m$ 个锥，各维 $n_i$）| $O((\sum n_i)n^2)$ | $O(\sqrt{\sum n_i})$ |
| SDP（$n \times n$ 矩阵变量）| $O(n^{4.5})$ 至 $O(n^6)$ | $O(\sqrt{n})$ |

---

## 深度学习应用

### 应用一：稀疏编码（LP/SOCP）

**稀疏编码**旨在用字典 $D \in \mathbb{R}^{m \times k}$（$k > m$，过完备字典）的稀疏线性组合表示信号 $\mathbf{y}$：

$$\min_{\mathbf{x}} \|\mathbf{x}\|_0 \quad \text{s.t.} \quad \mathbf{y} = D\mathbf{x}$$

$\ell_0$ 范数最小化是 NP-hard 组合问题。**凸松弛**：将 $\|\mathbf{x}\|_0$ 替换为 $\|\mathbf{x}\|_1$，得到 **LASSO / Basis Pursuit**：

$$
\min_{\mathbf{x}} \|\mathbf{x}\|_1 \quad \text{s.t.} \quad \mathbf{y} = D\mathbf{x}
$$

或无约束形式（弹性 LASSO）：

$$\min_{\mathbf{x}} \frac{1}{2}\|D\mathbf{x} - \mathbf{y}\|_2^2 + \lambda\|\mathbf{x}\|_1$$

**等价 LP**：引入辅助变量 $\mathbf{u} \succeq \mathbf{0}$，$\mathbf{v} \succeq \mathbf{0}$，$\mathbf{x} = \mathbf{u} - \mathbf{v}$，$|\mathbf{x}| = \mathbf{u} + \mathbf{v}$：

$$\min_{\mathbf{u}, \mathbf{v}} \; \mathbf{1}^T(\mathbf{u}+\mathbf{v}) \quad \text{s.t.} \quad D(\mathbf{u}-\mathbf{v}) = \mathbf{y}, \; \mathbf{u}, \mathbf{v} \succeq \mathbf{0}$$

**压缩感知定理**：当字典 $D$ 满足**限制等距性质（RIP）**时，若真实信号 $\mathbf{x}^*$ 是 $s$-稀疏的，则 $\ell_1$ 松弛在 $m \geq O(s\log(k/s))$ 次测量下能精确恢复 $\mathbf{x}^*$。这是凸优化理论在信号处理中最深刻的结果之一。

```python
import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# 应用一：稀疏编码——L1 松弛（等价 LP）
# ============================================================

np.random.seed(42)

# 构建过完备字典（DCT 基 + 随机原子）
m_sig, k_dict = 32, 64   # 信号维度 32，字典 64 个原子
D_dict = np.random.randn(m_sig, k_dict)
# 归一化各列（字典原子单位化）
D_dict /= np.linalg.norm(D_dict, axis=0, keepdims=True)

# 生成稀疏真实信号（5-稀疏）
s_sparsity = 5
x_true_sc = np.zeros(k_dict)
support = np.random.choice(k_dict, s_sparsity, replace=False)
x_true_sc[support] = np.random.randn(s_sparsity) * 2
y_obs = D_dict @ x_true_sc + np.random.randn(m_sig) * 0.02  # 少量噪声

# L1 最小化（Basis Pursuit Denoising）
x_l1_sc = cp.Variable(k_dict)
lam_sc = 0.1
prob_sc = cp.Problem(
    cp.Minimize(0.5 * cp.sum_squares(D_dict @ x_l1_sc - y_obs) + lam_sc * cp.norm1(x_l1_sc))
)
prob_sc.solve(solver=cp.ECOS)

# L2 最小化（最小二乘，非稀疏）
x_l2_sc = np.linalg.lstsq(D_dict, y_obs, rcond=None)[0]

print("=== 稀疏编码：L1 松弛 vs L2 最小二乘 ===")
print(f"真实非零系数位置: {sorted(support)}")
print(f"L1 解非零元素数（阈值 0.1）: {np.sum(np.abs(x_l1_sc.value) > 0.1)}")
print(f"L2 解非零元素数（阈值 0.1）: {np.sum(np.abs(x_l2_sc) > 0.1)}")
print(f"L1 恢复误差: {np.linalg.norm(x_l1_sc.value - x_true_sc):.4f}")
print(f"L2 恢复误差: {np.linalg.norm(x_l2_sc - x_true_sc):.4f}")
print(f"真实支撑集恢复率（L1）: "
      f"{len(set(np.where(np.abs(x_l1_sc.value)>0.1)[0]) & set(support))}/{s_sparsity}")

# ============================================================
# 应用二：核方法——SVM 对偶（QP）
# ============================================================

# 使用 RBF 核的支持向量机（核 SVM）
def rbf_kernel(X1, X2, gamma=1.0):
    """RBF/高斯核：K(x, x') = exp(-gamma * ||x - x'||^2)"""
    diff = X1[:, None, :] - X2[None, :, :]   # (N1, N2, d)
    return np.exp(-gamma * np.sum(diff**2, axis=2))

np.random.seed(3)
N_svm = 60
# 生成非线性可分数据（两个圆环）
theta = np.random.uniform(0, 2*np.pi, N_svm)
r_inner = np.random.uniform(0.5, 1.2, N_svm//2)
r_outer = np.random.uniform(1.8, 2.5, N_svm//2)
X_svm = np.vstack([
    np.column_stack([r_inner * np.cos(theta[:N_svm//2]),
                     r_inner * np.sin(theta[:N_svm//2])]),
    np.column_stack([r_outer * np.cos(theta[N_svm//2:]),
                     r_outer * np.sin(theta[N_svm//2:])])
])
y_svm = np.hstack([np.ones(N_svm//2), -np.ones(N_svm//2)])

# 计算核矩阵
gamma_rbf = 1.0
K = rbf_kernel(X_svm, X_svm, gamma=gamma_rbf)
K_y = (y_svm[:, None] * y_svm[None, :]) * K   # 元素乘以 y_i * y_j

# 核 SVM 对偶问题（QP）
C_svm = 1.0
alpha = cp.Variable(N_svm, nonneg=True)
prob_ksvm = cp.Problem(
    cp.Maximize(cp.sum(alpha) - 0.5 * cp.quad_form(alpha, K_y + 1e-8 * np.eye(N_svm))),
    [alpha <= C_svm, y_svm @ alpha == 0]
)
prob_ksvm.solve(solver=cp.OSQP)

# 支持向量
sv_mask = alpha.value > 1e-4
n_sv = sv_mask.sum()
print(f"\n=== 核 SVM（QP 对偶，RBF 核 gamma={gamma_rbf}）===")
print(f"训练样本数: {N_svm}，支持向量数: {n_sv} ({n_sv/N_svm:.0%})")
print(f"对偶目标值: {prob_ksvm.value:.4f}")

# 计算训练精度
b_ksvm = np.mean(y_svm[sv_mask] - (alpha.value[sv_mask] * y_svm[sv_mask]) @ K[sv_mask][:, sv_mask].T)
decisions = K @ (alpha.value * y_svm) + b_ksvm
train_acc = np.mean(np.sign(decisions) == y_svm)
print(f"训练精度: {train_acc:.0%}")

# ============================================================
# 应用三：神经网络权重约束松弛（SDP/SOCP）
# ============================================================

print("\n=== 神经网络谱范数约束（SDP 松弛） ===")
print("谱范数约束：||W||_2 <= sigma_max 等价于 LMI:")
print("  [sigma*I  W ]")
print("  [W^T  sigma*I] >= 0")

# 演示：给定权重矩阵，投影到谱范数球内
torch.manual_seed(0)
W_nn = torch.randn(8, 6)
W_np = W_nn.numpy()

sigma_target = 1.0   # 目标谱范数上界

# 通过 SVD 投影到谱范数约束集合
# min ||W - W0||_F  s.t. ||W||_2 <= sigma_target
W_proj = cp.Variable(W_np.shape)
prob_proj = cp.Problem(
    cp.Minimize(cp.norm(W_proj - W_np, 'fro')),
    [cp.norm(W_proj, 2) <= sigma_target]   # CVXPY 内部用 SDP 实现
)
prob_proj.solve(solver=cp.SCS)

W_proj_val = W_proj.value
sv_before = np.linalg.svd(W_np, compute_uv=False)
sv_after  = np.linalg.svd(W_proj_val, compute_uv=False)

print(f"\n原始权重矩阵谱范数: {sv_before[0]:.4f}")
print(f"投影后谱范数: {sv_after[0]:.4f} (<= {sigma_target})")
print(f"Frobenius 距离（投影代价）: {np.linalg.norm(W_proj_val - W_np, 'fro'):.4f}")

# 对比：简单截断 SVD（标准方法）
U_t, s_t, Vt_t = np.linalg.svd(W_np, full_matrices=False)
s_clipped = np.minimum(s_t, sigma_target)
W_clipped = U_t @ np.diag(s_clipped) @ Vt_t
sv_clipped = np.linalg.svd(W_clipped, compute_uv=False)
print(f"截断 SVD 谱范数: {sv_clipped[0]:.4f}")
print(f"两种方法 Frobenius 差: {np.linalg.norm(W_proj_val - W_clipped, 'fro'):.6f}")
print("（两种方法等价：凸投影的解析解即截断 SVD）")

# ============================================================
# 应用四：层级汇总——各问题类型求解时间对比
# ============================================================

import time

print("\n=== 不同凸优化问题求解时间对比 ===")

def time_solve(prob):
    t0 = time.time()
    prob.solve(solver=cp.ECOS, verbose=False)
    return time.time() - t0

np.random.seed(0)
N_bench = 50

# LP
x_lp = cp.Variable(N_bench, nonneg=True)
prob_lp_b = cp.Problem(
    cp.Minimize(np.random.randn(N_bench) @ x_lp),
    [np.random.randn(N_bench, N_bench) @ x_lp <= np.abs(np.random.randn(N_bench))]
)

# QP
x_qp = cp.Variable(N_bench)
P_qp = np.random.randn(N_bench, N_bench)
P_qp = P_qp.T @ P_qp + np.eye(N_bench) * 0.1
prob_qp_b = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(x_qp, P_qp) + np.random.randn(N_bench) @ x_qp),
    [np.random.randn(10, N_bench) @ x_qp <= np.ones(10)]
)

# SOCP
x_socp_b = cp.Variable(N_bench)
prob_socp_b = cp.Problem(
    cp.Minimize(np.random.randn(N_bench) @ x_socp_b),
    [cp.norm(np.random.randn(10, N_bench) @ x_socp_b, 2) <= 5.0,
     cp.norm(np.random.randn(8, N_bench) @ x_socp_b, 2) <= 3.0]
)

benchmark = [
    ("LP  (n=50)", prob_lp_b),
    ("QP  (n=50)", prob_qp_b),
    ("SOCP(n=50)", prob_socp_b),
]

for name, prob in benchmark:
    t = time_solve(prob)
    status = prob.status
    print(f"  {name}: {t*1000:.1f} ms  状态: {status}")
```

---

## 练习题

**练习 10.1**（LP 建模）

某工厂生产两种产品 A 和 B，每件 A 需用 2 小时机器时间和 1 kg 原料，利润 3 元；每件 B 需用 1 小时机器时间和 2 kg 原料，利润 4 元。每天机器时间总量为 8 小时，原料总量为 10 kg，产量非负。

(a) 写出最大化利润的 LP 标准形式（转化为最小化问题）。

(b) 用图解法（画出可行域和目标函数等值线）求解。

(c) 写出该 LP 的对偶问题，验证强对偶定理（原始与对偶最优值相等）。

---

**练习 10.2**（QP 与 SVM）

考虑以下数据点（二维特征，二分类）：

正类（$y=+1$）：$(1, 2), (2, 1), (2, 3)$

负类（$y=-1$）：$(-1, -1), (-2, 0), (0, -2)$

(a) 写出硬间隔 SVM 的原问题（凸 QP）形式，指出目标函数矩阵 $P$ 和约束矩阵 $G$。

(b) 写出 SVM 的对偶 QP 问题，并说明核技巧如何通过将内积 $\mathbf{x}_i^T\mathbf{x}_j$ 替换为核函数 $K(\mathbf{x}_i,\mathbf{x}_j)$ 实现非线性分类。

(c) 若用凸 QP 求解器解出对偶变量 $\boldsymbol{\alpha}^*$，如何利用互补松弛条件识别支持向量？支持向量对应 $\alpha_i^*$ 满足什么条件？

---

**练习 10.3**（SOCP 鲁棒优化）

考虑如下名义 LP：

$$\min_{\mathbf{x}} \; \mathbf{c}^T\mathbf{x} \quad \text{s.t.} \quad \mathbf{a}_i^T\mathbf{x} \leq b_i, \; i = 1, 2, 3$$

其中 $\mathbf{a}_i$ 存在椭球不确定性 $\mathcal{U}_i = \{\bar{\mathbf{a}}_i + P_i\mathbf{u} \mid \|\mathbf{u}\|_2 \leq 1\}$。

(a) 证明"对所有 $\mathbf{a}_i \in \mathcal{U}_i$，$\mathbf{a}_i^T\mathbf{x} \leq b_i$ 成立"等价于 $\bar{\mathbf{a}}_i^T\mathbf{x} + \|P_i^T\mathbf{x}\|_2 \leq b_i$。

(b) 说明改写后的鲁棒约束是二阶锥约束，将完整的鲁棒优化问题写成 SOCP 标准形式。

(c) 若 $P_i = \epsilon I$（各向同性不确定性，半径 $\epsilon$），鲁棒约束简化为什么？这与正则化有何关系？

---

**练习 10.4**（SDP 与 Schur 补）

**Schur 补引理**：$\begin{pmatrix} A & B \\ B^T & C \end{pmatrix} \succeq 0$（$A \succ 0$）$\iff$ $C - B^T A^{-1} B \succeq 0$。

(a) 利用 Schur 补，将约束 $\mathbf{x}^T P \mathbf{x} \leq t$（$P \succ 0$）改写为等价的 LMI 形式。

(b) 将凸 QCQP 问题

$$\min_{\mathbf{x}} \; \mathbf{c}^T\mathbf{x} \quad \text{s.t.} \quad \mathbf{x}^T Q_i \mathbf{x} + 2\mathbf{r}_i^T\mathbf{x} \leq s_i, \; i = 1, \ldots, m \quad (Q_i \succeq 0)$$

改写为等价的 SDP（通过引入辅助变量 $t_i$，将每个二次约束改写为 LMI）。

(c) 说明这一转化的理论意义：为何 SDP 是比 QCQP 更强大的建模框架？

---

**练习 10.5**（核范数与矩阵补全）

核范数 $\|X\|_* = \sum_i \sigma_i(X)$ 是秩函数 $\text{rank}(X)$ 的凸包络（最紧凸下界）。

(a) 证明 $\|X\|_* = \min_{U, V: X = UV^T} \frac{1}{2}(\|U\|_F^2 + \|V\|_F^2)$，其中 $\|\cdot\|_F$ 是 Frobenius 范数。

*提示*：利用 SVD $X = U\Sigma V^T$，对上式右边取最小值，验证最优分解对应 $U = \hat{U}\Sigma^{1/2}$，$V = \hat{V}\Sigma^{1/2}$。

(b) 矩阵补全的凸松弛为

$$\min_{X} \; \|X\|_* \quad \text{s.t.} \quad X_{ij} = M_{ij},\; (i,j) \in \Omega$$

将此问题改写为 SDP：引入矩阵变量 $W_1 \in \mathbb{S}^m_+$，$W_2 \in \mathbb{S}^n_+$，并利用

$$\|X\|_* = \min_{W_1, W_2} \; \frac{1}{2}(\text{tr}(W_1) + \text{tr}(W_2)) \quad \text{s.t.} \quad \begin{pmatrix} W_1 & X \\ X^T & W_2 \end{pmatrix} \succeq 0$$

(c) 在什么条件下（关于观测数量 $|\Omega|$ 和矩阵的秩 $r$），核范数最小化能精确恢复真实矩阵？（简述相关定理的直觉，不要求严格证明。）

---

## 练习答案

### 练习 10.1 解答

**(a) LP 标准形式**

设 $x_1$（A 产品件数），$x_2$（B 产品件数）。

最大化利润 $3x_1 + 4x_2$ 转为最小化 $-3x_1 - 4x_2$：

$$
\begin{aligned}
\min_{x_1, x_2} \quad & -3x_1 - 4x_2 \\
\text{s.t.} \quad & 2x_1 + x_2 \leq 8 \quad \text{（机器时间）} \\
& x_1 + 2x_2 \leq 10 \quad \text{（原料）} \\
& x_1, x_2 \geq 0
\end{aligned}
$$

**(b) 图解法**

可行域由四个半空间的交确定，顶点为：$(0,0)$，$(4,0)$，$(2,4)$，$(0,5)$。

计算各顶点利润：

| 顶点 | 利润 $3x_1+4x_2$ |
|------|---------------|
| $(0, 0)$ | $0$ |
| $(4, 0)$ | $12$ |
| $(2, 4)$ | $6+16=22$ |
| $(0, 5)$ | $20$ |

最优解为 $(x_1^*, x_2^*) = (2, 4)$，最大利润 $p^* = 22$ 元。

**(c) 对偶问题**

引入对偶变量 $\lambda_1 \geq 0$（机器约束），$\lambda_2 \geq 0$（原料约束）：

$$
\begin{aligned}
\max_{\lambda_1, \lambda_2 \geq 0} \quad & -8\lambda_1 - 10\lambda_2 \\
\text{s.t.} \quad & -2\lambda_1 - \lambda_2 \geq -3 \; \Leftrightarrow \; 2\lambda_1 + \lambda_2 \leq 3\\
& -\lambda_1 - 2\lambda_2 \geq -4 \; \Leftrightarrow \; \lambda_1 + 2\lambda_2 \leq 4
\end{aligned}
$$

最大化 $-8\lambda_1 - 10\lambda_2$ 等价于最小化 $8\lambda_1 + 10\lambda_2$（取负号后为最大化问题）。

解对偶：由 KKT 互补松弛，原始最优解 $(2,4)$ 处两个约束均有效（等号成立），故 $\lambda_1, \lambda_2 > 0$。

由对偶可行性的互补松弛：$2\lambda_1 + \lambda_2 = 3$，$\lambda_1 + 2\lambda_2 = 4$，解得 $\lambda_1^* = 2/3$，$\lambda_2^* = 5/3$。

对偶最优值：$8 \times (2/3) + 10 \times (5/3) = 16/3 + 50/3 = 66/3 = 22$。

**强对偶验证**：$p^* = d^* = 22$，对偶间隙为零。$\square$

---

### 练习 10.2 解答

**(a) 硬间隔 SVM 原问题**

数据：6 个点，$n=2$ 维特征，$N=6$ 个约束。

$$
\min_{\mathbf{w} \in \mathbb{R}^2, b} \; \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \; i=1,\ldots,6
$$

写成标准 QP 形式，决策变量 $\mathbf{z} = (\mathbf{w}, b)^T \in \mathbb{R}^3$：

$$P = \begin{pmatrix} I_2 & \mathbf{0} \\ \mathbf{0}^T & 0 \end{pmatrix} \succeq 0, \quad \mathbf{q} = \mathbf{0}$$

约束矩阵 $G \in \mathbb{R}^{6 \times 3}$，第 $i$ 行为 $-y_i(\mathbf{x}_i^T, 1)$，$\mathbf{h} = -\mathbf{1}$。

**(b) 对偶问题与核技巧**

对偶 QP：

$$\max_{\boldsymbol{\alpha}} \; \mathbf{1}^T\boldsymbol{\alpha} - \frac{1}{2}\boldsymbol{\alpha}^T (K_y) \boldsymbol{\alpha} \quad \text{s.t.} \quad \boldsymbol{\alpha} \geq 0, \; \mathbf{y}^T\boldsymbol{\alpha} = 0$$

其中 $(K_y)_{ij} = y_i y_j \mathbf{x}_i^T\mathbf{x}_j$。

**核技巧**：将 $\mathbf{x}_i^T\mathbf{x}_j$ 替换为 $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)$（核函数），无需显式计算特征映射 $\phi$。对偶问题形式不变，但在原始空间中实现了非线性决策边界。常用核：RBF $e^{-\gamma\|\mathbf{x}-\mathbf{x}'\|^2}$、多项式 $(\mathbf{x}^T\mathbf{x}'+c)^d$。

**(c) 支持向量的识别**

由互补松弛条件：$\alpha_i^*[y_i(\mathbf{w}^{*T}\mathbf{x}_i + b^*) - 1] = 0$。

- $\alpha_i^* > 0$ $\Rightarrow$ $y_i(\mathbf{w}^{*T}\mathbf{x}_i + b^*) = 1$：点在间隔边界上，是**支持向量**
- $\alpha_i^* = 0$ $\Rightarrow$ $y_i(\mathbf{w}^{*T}\mathbf{x}_i + b^*) > 1$：点在间隔外，非支持向量

实际中取 $\alpha_i^* > \epsilon$（如 $10^{-5}$）的点作为支持向量。最优权重由支持向量线性表示：$\mathbf{w}^* = \sum_i \alpha_i^* y_i \mathbf{x}_i$。

---

### 练习 10.3 解答

**(a) 鲁棒约束等价性证明**

"对所有 $\mathbf{a}_i \in \mathcal{U}_i = \{\bar{\mathbf{a}}_i + P_i\mathbf{u} \mid \|\mathbf{u}\|_2 \leq 1\}$，$\mathbf{a}_i^T\mathbf{x} \leq b_i$"等价于：

$$\max_{\|\mathbf{u}\|_2 \leq 1} (\bar{\mathbf{a}}_i + P_i\mathbf{u})^T\mathbf{x} \leq b_i$$

$$\Leftrightarrow \quad \bar{\mathbf{a}}_i^T\mathbf{x} + \max_{\|\mathbf{u}\|_2 \leq 1} \mathbf{u}^T P_i^T\mathbf{x} \leq b_i$$

由 Cauchy-Schwarz 不等式，$\max_{\|\mathbf{u}\|_2 \leq 1} \mathbf{u}^T (P_i^T\mathbf{x}) = \|P_i^T\mathbf{x}\|_2$（最大值在 $\mathbf{u}^* = P_i^T\mathbf{x}/\|P_i^T\mathbf{x}\|_2$ 处取到）。

故鲁棒约束等价于 $\bar{\mathbf{a}}_i^T\mathbf{x} + \|P_i^T\mathbf{x}\|_2 \leq b_i$。$\square$

**(b) 鲁棒 LP 的 SOCP 形式**

$$
\begin{aligned}
\min_{\mathbf{x}} \quad & \mathbf{c}^T\mathbf{x} \\
\text{s.t.} \quad & \bar{\mathbf{a}}_i^T\mathbf{x} + \|P_i^T\mathbf{x}\|_2 \leq b_i, \quad i = 1, 2, 3
\end{aligned}
$$

引入辅助变量 $t_i = \|P_i^T\mathbf{x}\|_2$，改写为标准 SOCP：

$$
\begin{aligned}
\min_{\mathbf{x}, \mathbf{t}} \quad & \mathbf{c}^T\mathbf{x} \\
\text{s.t.} \quad & \|P_i^T\mathbf{x}\|_2 \leq t_i, \quad i = 1, 2, 3 \quad \text{（二阶锥约束）}\\
& \bar{\mathbf{a}}_i^T\mathbf{x} + t_i \leq b_i, \quad i = 1, 2, 3 \quad \text{（线性约束）}
\end{aligned}
$$

**(c) 各向同性不确定性的简化**

取 $P_i = \epsilon I$，则 $\|P_i^T\mathbf{x}\|_2 = \epsilon\|\mathbf{x}\|_2$，鲁棒约束变为：

$$\bar{\mathbf{a}}_i^T\mathbf{x} + \epsilon\|\mathbf{x}\|_2 \leq b_i$$

**与正则化的关系**：这等价于在名义约束中加入一个与 $\|\mathbf{x}\|_2$ 成正比的惩罚项，本质上是对解的范数施加了软约束。若将 $\epsilon\|\mathbf{x}\|_2$ 看作正则化项，则不确定性半径 $\epsilon$ 扮演正则化参数的角色——**不确定性越大，解被迫越小（更保守）**，这与 Tikhonov 正则化的效果完全一致。

---

### 练习 10.4 解答

**(a) 二次约束的 LMI 改写**

约束 $\mathbf{x}^T P \mathbf{x} \leq t$（$P \succ 0$）。引入分块矩阵：

$$M = \begin{pmatrix} t & \mathbf{x}^T \\ \mathbf{x} & P^{-1} \end{pmatrix}$$

由 Schur 补引理（$A = t > 0$，$B = \mathbf{x}$，$C = P^{-1}$）：

$$M \succeq 0 \iff P^{-1} - \mathbf{x} \cdot t^{-1} \cdot \mathbf{x}^T \succeq 0 \iff t P^{-1} - \mathbf{x}\mathbf{x}^T \succeq 0 \iff t \geq \mathbf{x}^T P \mathbf{x}$$

（最后一步左乘 $P^{1/2}$，右乘 $P^{1/2}$，得 $tI \succeq P^{1/2}\mathbf{x}\mathbf{x}^TP^{1/2}$，等价于 $t \geq \|P^{1/2}\mathbf{x}\|^2 = \mathbf{x}^TP\mathbf{x}$）

故 $\mathbf{x}^TP\mathbf{x} \leq t$ $\iff$ $\begin{pmatrix} t & \mathbf{x}^T \\ \mathbf{x} & P^{-1} \end{pmatrix} \succeq 0$。$\square$

**(b) QCQP 改写为 SDP**

对每个约束 $\mathbf{x}^T Q_i \mathbf{x} + 2\mathbf{r}_i^T\mathbf{x} \leq s_i$（$Q_i \succeq 0$），等价地写为：

$$(\mathbf{x}+Q_i^{\dagger}\mathbf{r}_i)^T Q_i (\mathbf{x}+Q_i^{\dagger}\mathbf{r}_i) \leq s_i + \mathbf{r}_i^T Q_i^{\dagger}\mathbf{r}_i := \tilde{s}_i$$

引入辅助变量 $t_i$，由 (a) 改写为 LMI：

$$\begin{pmatrix} t_i & (\mathbf{x}+Q_i^{\dagger}\mathbf{r}_i)^T \\ \mathbf{x}+Q_i^{\dagger}\mathbf{r}_i & Q_i^{\dagger} \end{pmatrix} \succeq 0, \quad t_i \leq \tilde{s}_i$$

从而整个 QCQP 等价于带若干 LMI 约束的 SDP。

**(c) SDP 的建模优势**

QCQP 中的正定性条件 $Q_i \succeq 0$ 限制了约束的形状，而 SDP 通过矩阵变量 $X$ 和正半定锥约束 $X \succeq 0$ 直接处理矩阵结构，可以表达：
- 约束涉及矩阵特征值（如谱范数约束 $\|A\|_2 \leq t$）
- 涉及迹内积的目标或约束（如最大特征值最小化）
- 涉及奇异值的问题（如矩阵低秩逼近的凸松弛）

QCQP 仅能处理标量二次关系，而 SDP 在矩阵半正定锥上优化，建模能力远超 QCQP。

---

### 练习 10.5 解答

**(a) 核范数的矩阵分解表示**

设 $X$ 的 SVD 为 $X = \hat{U}\Sigma\hat{V}^T$（$\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r)$，$r = \text{rank}(X)$）。

对任意满足 $X = UV^T$ 的分解，令 $U$ 的列为 $\mathbf{u}_i$，$V$ 的列为 $\mathbf{v}_i$：

$$\frac{1}{2}(\|U\|_F^2 + \|V\|_F^2) \geq \|U^TV\|_* \geq \text{tr}(\hat{U}^T U V^T \hat{V}\Sigma^{-1}) = \cdots = \sum_i \sigma_i = \|X\|_*$$

（AM-GM：$\frac{\|U\|_F^2 + \|V\|_F^2}{2} \geq \|U\|_F\|V\|_F \geq \cdots \geq \|X\|_*$）

取 $U^* = \hat{U}\Sigma^{1/2}$，$V^* = \hat{V}\Sigma^{1/2}$，则 $U^{*T}V^* = \Sigma^{1/2}\hat{U}^T\hat{V}\Sigma^{1/2} = \Sigma$（因 $\hat{U}^T\hat{V}$ 是酉矩阵乘积），且

$$\frac{1}{2}(\|U^*\|_F^2 + \|V^*\|_F^2) = \frac{1}{2}(\text{tr}(\Sigma) + \text{tr}(\Sigma)) = \text{tr}(\Sigma) = \|X\|_*$$

故最小值等于 $\|X\|_*$，在 $U=\hat{U}\Sigma^{1/2}$，$V=\hat{V}\Sigma^{1/2}$ 处取到。$\square$

**(b) 矩阵补全改写为 SDP**

$$
\begin{aligned}
\min_{X, W_1, W_2} \quad & \frac{1}{2}(\text{tr}(W_1) + \text{tr}(W_2)) \\
\text{s.t.} \quad & \begin{pmatrix} W_1 & X \\ X^T & W_2 \end{pmatrix} \succeq 0 \\
& X_{ij} = M_{ij}, \quad (i,j) \in \Omega \\
& W_1 \in \mathbb{S}^m_+, \quad W_2 \in \mathbb{S}^n_+
\end{aligned}
$$

这是一个标准 SDP：线性目标（迹函数），LMI 约束（分块正半定），线性等式约束（观测固定）。

**(c) 精确恢复的条件（直觉）**

**Candès-Recht 定理**（2009）给出了精确恢复的充分条件：

若 $m \times n$ 矩阵 $M$（不妨 $m \leq n$）的秩为 $r$，且满足**非相干性条件**（矩阵的奇异向量与标准基不太对齐），则当随机观测数 $|\Omega| \geq C \cdot \mu r n \log n$（$\mu$ 为非相干参数，$C$ 为常数）时，以高概率可精确恢复 $M$。

**直觉**：秩越低（更多结构），所需观测越少；矩阵越"扩散"（非相干性好），信息分布均匀，从部分观测恢复越容易。这是**随机矩阵感知**的矩阵版本，对应信号处理中的压缩感知理论。

---

*下一章：第11章 内点法——我们将深入探讨高效求解本章各类凸优化问题（LP、QP、SOCP、SDP）的内点法，包括障碍函数方法、中心路径和牛顿步的理论与实现。*
