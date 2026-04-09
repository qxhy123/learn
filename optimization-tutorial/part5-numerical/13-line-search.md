# 第13章：线搜索方法

---

## 学习目标

学完本章后，你将能够：

1. **理解线搜索的框架与作用**：掌握线搜索在迭代优化算法中的角色，理解为何每步迭代都需要选择步长，以及步长选择对算法收敛性的决定性影响
2. **掌握 Armijo 充分下降条件**：推导 Armijo 条件的数学形式，理解其几何意义——要求函数值下降量不低于线性预测值的固定比例，并能分析其防止步长过大的作用
3. **理解 Wolfe 条件与强 Wolfe 条件**：在 Armijo 条件基础上增加曲率条件，防止步长过小，掌握 Wolfe 条件在拟牛顿法中的重要性
4. **实现回溯线搜索算法**：掌握基于 Armijo 条件的回溯线搜索的算法流程，分析其有限步内必然终止的理论保证
5. **理解精确线搜索方法**：掌握黄金分割法和二分法（割线法）求解精确一维最小化的原理，分析其收敛速度与适用场景，并能与非精确线搜索进行比较权衡

---

## 13.1 线搜索框架

### 13.1.1 迭代优化中的步长问题

几乎所有基于梯度的迭代优化方法都具有如下通用框架：

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{d}_k$$

其中 $\mathbf{d}_k$ 是**搜索方向**（Search Direction），$\alpha_k > 0$ 是**步长**（Step Size）或**学习率**（Learning Rate）。

步长的选择直接决定算法的效率与收敛性：

- **步长过大**：可能跨越最小值，导致函数值上升甚至发散
- **步长过小**：每步进展甚微，需要大量迭代才能收敛
- **步长恰当**：充分利用搜索方向，加速收敛

**线搜索**（Line Search）是一种在给定方向 $\mathbf{d}_k$ 上系统确定步长 $\alpha_k$ 的方法。问题归结为一维优化：

$$\phi(\alpha) = f(\mathbf{x}_k + \alpha \mathbf{d}_k), \quad \alpha > 0$$

### 13.1.2 搜索方向的分类

搜索方向 $\mathbf{d}_k$ 必须满足**下降条件**，即：

$$\mathbf{d}_k^\top \nabla f(\mathbf{x}_k) < 0$$

这保证了沿 $\mathbf{d}_k$ 方向移动时函数值能够减小（至少在足够小的步长下）。常见搜索方向：

| 方法 | 搜索方向 $\mathbf{d}_k$ | 特点 |
|------|------------------------|------|
| 梯度下降 | $-\nabla f(\mathbf{x}_k)$ | 最速下降，计算简单 |
| 牛顿法 | $-[\nabla^2 f(\mathbf{x}_k)]^{-1}\nabla f(\mathbf{x}_k)$ | 利用曲率，收敛快 |
| 拟牛顿法（L-BFGS） | $-H_k^{-1}\nabla f(\mathbf{x}_k)$（近似 Hessian） | 兼顾效率与精度 |
| 共轭梯度法 | $-\nabla f(\mathbf{x}_k) + \beta_k \mathbf{d}_{k-1}$ | 利用历史方向 |

本章专注于**步长选择**问题，即给定合法搜索方向后如何高效确定 $\alpha_k$。

### 13.1.3 一维目标函数的性质

定义辅助函数 $\phi: \mathbb{R}_+ \to \mathbb{R}$：

$$\phi(\alpha) = f(\mathbf{x}_k + \alpha \mathbf{d}_k)$$

其导数为方向导数：

$$\phi'(\alpha) = \nabla f(\mathbf{x}_k + \alpha \mathbf{d}_k)^\top \mathbf{d}_k$$

特别地，在 $\alpha = 0$ 处：

$$\phi'(0) = \nabla f(\mathbf{x}_k)^\top \mathbf{d}_k < 0$$

（由下降条件保证）。这意味着在当前点沿方向 $\mathbf{d}_k$ 移动，函数值在初始时是严格减小的。

### 13.1.4 精确线搜索与非精确线搜索

**精确线搜索**（Exact Line Search）：求解

$$\alpha_k^* = \arg\min_{\alpha > 0} \phi(\alpha)$$

优点是每步步长最优，但计算代价高（需要大量函数值/梯度计算）。

**非精确线搜索**（Inexact Line Search）：不要求精确最优步长，只需找到"足够好"的步长，满足某些条件即可。典型条件包括 Armijo 条件和 Wolfe 条件。

在实践中，**非精确线搜索**几乎总是更受青睐，因为寻找精确最优步长的代价往往不值得。

```
线搜索方法分类：

非精确线搜索
├── Armijo 回溯线搜索（充分下降条件）
├── Wolfe 条件线搜索（充分下降 + 曲率条件）
└── 强 Wolfe 条件线搜索

精确线搜索
├── 黄金分割法（函数值，无需梯度）
├── 二分法/割线法（利用导数）
└── 解析法（二次/三次插值）
```

---

## 13.2 Armijo 条件

### 13.2.1 充分下降条件的推导

一个自然的要求是：步长 $\alpha$ 应使函数值有"足够大"的下降。但什么叫"足够大"？

最直接的想法是要求 $\phi(\alpha) < \phi(0)$，即函数值下降即可。然而这个条件太弱——即使步长极小，函数值也会（对于非零梯度）有所下降，但进展极其缓慢。

**线性下降模型**：在 $\alpha = 0$ 处对 $\phi(\alpha)$ 做一阶 Taylor 展开：

$$\phi(\alpha) \approx \phi(0) + \alpha \phi'(0)$$

这给出了函数值关于步长的线性预测。**Armijo 条件**要求实际下降量不低于线性预测下降量的 $c$ 倍：

$$\boxed{\phi(\alpha) \leq \phi(0) + c \alpha \phi'(0)}$$

其中 $c \in (0, 1)$ 是常数。展开后即为：

$$f(\mathbf{x}_k + \alpha \mathbf{d}_k) \leq f(\mathbf{x}_k) + c \alpha \nabla f(\mathbf{x}_k)^\top \mathbf{d}_k$$

由于 $\phi'(0) = \nabla f(\mathbf{x}_k)^\top \mathbf{d}_k < 0$，右侧是关于 $\alpha$ 的递减线性函数，Armijo 条件划定了一个"可接受区域"。

### 13.2.2 几何解释

```
φ(α)
  |
  |  φ(0) ────────────────────────────────── 水平线
  |      \
  |       \  φ(0) + c·α·φ'(0)（Armijo 界，斜率 = c·φ'(0)）
  |        \·············
  |         \           ···
  |          \       ········
  |           \  φ(α)（真实函数值）
  |            \   /
  |             \ /
  |              *  ← 精确最小值
  |
  +-----------------------------------------→ α
       可接受区域（φ(α) 低于 Armijo 界）
```

- Armijo 界的斜率为 $c \cdot \phi'(0)$，比实际初始斜率 $\phi'(0)$ 更平缓（因为 $c < 1$）
- Armijo 条件要求 $\phi(\alpha)$ 位于 Armijo 界之下
- 当 $\alpha$ 很小时，Armijo 条件总能满足（因为 $\phi(\alpha) \approx \phi(0) + \alpha\phi'(0) < \phi(0) + c\alpha\phi'(0)$，当 $c < 1$ 且 $\phi'(0) < 0$ 时）

### 13.2.3 参数 $c$ 的选择

常用取值 $c = 10^{-4}$（非常小），这意味着只需要实际下降量是线性预测值的万分之一即可。

取 $c$ 很小的原因：
- 目的是排除步长**过大**的情况（函数值基本不下降或上升）
- 不需要每步都接近精确最小值，只需"足够好"
- $c$ 越小，Armijo 界越低，接受区域越大，线搜索越容易终止

**单独的 Armijo 条件有一个缺陷**：它无法阻止步长过小。当 $\alpha$ 趋向 $0$ 时，Armijo 条件总成立，导致算法可能陷入步长极小、进展极慢的困境。这促使了 Wolfe 条件的提出（见 13.3 节）。

### 13.2.4 Armijo 条件的理论意义

**定理 13.1**（Armijo 条件保证的下降量）：若 $f$ 是 $L$-光滑函数，$\mathbf{d}_k$ 是下降方向，当步长 $\alpha$ 满足 Armijo 条件时，有：

$$f(\mathbf{x}_k) - f(\mathbf{x}_{k+1}) \geq c \alpha |\nabla f(\mathbf{x}_k)^\top \mathbf{d}_k|$$

这给出了每步函数值下降的显式下界。当 $\|\nabla f(\mathbf{x}_k)\|$ 不小时，每步下降量有界，防止算法停滞。

---

## 13.3 Wolfe 条件

### 13.3.1 曲率条件的动机

Armijo 条件只限制了步长不能太大，但未阻止步长过小。考虑极端情况：取 $\alpha$ 非常小，Armijo 条件总成立，但此时还远未到达一维问题的最优点，浪费了沿方向 $\mathbf{d}_k$ 的下降潜力。

**Wolfe 条件**在 Armijo 条件基础上增加**曲率条件**（Curvature Condition），要求步长不能太小，确保充分利用搜索方向：

$$\phi'(\alpha) \geq c_2 \phi'(0)$$

其中 $c_2 \in (c_1, 1)$，$c_1 = c$ 是 Armijo 参数。

**曲率条件的含义**：$\phi'(\alpha) = \nabla f(\mathbf{x}_k + \alpha \mathbf{d}_k)^\top \mathbf{d}_k$ 是在新点处沿方向 $\mathbf{d}_k$ 的方向导数。

- $\phi'(0) < 0$：初始时函数值在减小
- 曲率条件要求 $\phi'(\alpha) \geq c_2 \phi'(0)$：即在 $\mathbf{x}_{k+1}$ 处，沿 $\mathbf{d}_k$ 方向的斜率已经足够"平缓"
- 这意味着我们已经"越过"了函数值下降最快的区域，到达了相对平坦的地带

### 13.3.2 Wolfe 条件的完整形式

**Wolfe 条件**（两个条件同时满足）：

$$\boxed{\begin{aligned}
&\text{（W1，充分下降）} \quad f(\mathbf{x}_k + \alpha \mathbf{d}_k) \leq f(\mathbf{x}_k) + c_1 \alpha \nabla f(\mathbf{x}_k)^\top \mathbf{d}_k \\
&\text{（W2，曲率条件）} \quad \nabla f(\mathbf{x}_k + \alpha \mathbf{d}_k)^\top \mathbf{d}_k \geq c_2 \nabla f(\mathbf{x}_k)^\top \mathbf{d}_k
\end{aligned}}$$

参数满足 $0 < c_1 < c_2 < 1$。典型取值：$c_1 = 10^{-4}$，$c_2 = 0.9$（梯度类方法）或 $c_2 = 0.1$（共轭梯度法）。

### 13.3.3 强 Wolfe 条件

标准 Wolfe 条件的曲率条件 $\phi'(\alpha) \geq c_2 \phi'(0)$ 允许 $\phi'(\alpha)$ 为较大正值，即允许在函数值已经上升的地方停止。**强 Wolfe 条件**通过取绝对值加以约束：

$$\boxed{\begin{aligned}
&\text{（SW1）} \quad f(\mathbf{x}_k + \alpha \mathbf{d}_k) \leq f(\mathbf{x}_k) + c_1 \alpha \nabla f(\mathbf{x}_k)^\top \mathbf{d}_k \\
&\text{（SW2）} \quad |\nabla f(\mathbf{x}_k + \alpha \mathbf{d}_k)^\top \mathbf{d}_k| \leq c_2 |\nabla f(\mathbf{x}_k)^\top \mathbf{d}_k|
\end{aligned}}$$

**强 Wolfe 条件的含义**：不仅要求斜率足够平缓，还要求斜率的绝对值较小，即更接近一维问题的局部极小值点（$\phi'(\alpha) \approx 0$）。

强 Wolfe 条件与 Wolfe 条件的比较：

| 特性 | Wolfe 条件 | 强 Wolfe 条件 |
|------|-----------|--------------|
| 充分下降 | $\phi(\alpha) \leq \phi(0) + c_1\alpha\phi'(0)$ | 相同 |
| 曲率 | $\phi'(\alpha) \geq c_2\phi'(0)$ | $|\phi'(\alpha)| \leq c_2|\phi'(0)|$ |
| 接受区域 | 较宽 | 较窄（更接近极小值） |
| 主要用途 | 一般梯度类方法 | L-BFGS 等拟牛顿法 |

### 13.3.4 Wolfe 条件的存在性

**定理 13.2**（Wolfe 条件的存在性）：设 $f$ 连续可微且下有界，$\mathbf{d}_k$ 是下降方向（$\nabla f(\mathbf{x}_k)^\top \mathbf{d}_k < 0$），$0 < c_1 < c_2 < 1$。则存在满足 Wolfe 条件的步长区间。

**证明思路**：

设 $\phi(\alpha) = f(\mathbf{x}_k + \alpha\mathbf{d}_k)$。由于 $f$ 下有界，$\phi$ 也下有界。

首先，由连续性和 $\phi'(0) < 0$，对充分小的 $\alpha$，充分下降条件（W1）成立。

设 $\alpha_1$ 是满足 W1 的最大步长（即 $\phi(\alpha_1) = \phi(0) + c_1\alpha_1\phi'(0)$）。由中值定理，存在 $\xi \in (0, \alpha_1)$ 使得：

$$\phi'(\xi) = \frac{\phi(\alpha_1) - \phi(0)}{\alpha_1} = c_1\phi'(0) > c_2\phi'(0)$$

（最后一步因 $c_1 < c_2$ 且 $\phi'(0) < 0$）。因此 $\xi$ 满足曲率条件，且 $\xi \in (0, \alpha_1)$ 内也满足 W1。$\square$

### 13.3.5 Wolfe 条件在拟牛顿法中的重要性

Wolfe 条件的核心作用在拟牛顿法（如 L-BFGS）中尤为突出。拟牛顿法的 Hessian 近似更新（BFGS 公式）要求曲率条件：

$$\mathbf{y}_k^\top \mathbf{s}_k > 0, \quad \mathbf{s}_k = \alpha_k \mathbf{d}_k, \quad \mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$$

可以验证：若步长满足 Wolfe 条件，则 $\mathbf{y}_k^\top \mathbf{s}_k > 0$ 自动成立，从而保证 Hessian 近似矩阵始终正定。这是 L-BFGS 必须使用 Wolfe 条件线搜索的理论依据。

---

## 13.4 回溯线搜索

### 13.4.1 算法框架

**回溯线搜索**（Backtracking Line Search）是实现 Armijo 条件最简单、最常用的方法。核心思想：从一个较大的初始步长开始，逐步缩减，直到满足 Armijo 条件为止。

**算法 13.1（Armijo 回溯线搜索）**：

```
输入：
    当前点 x_k，搜索方向 d_k（下降方向）
    初始步长 ᾱ > 0（通常取 1）
    充分下降参数 c ∈ (0, 1)（通常 c = 1e-4）
    缩减因子 β ∈ (0, 1)（通常 β = 0.5）

过程：
    令 α = ᾱ
    当 f(x_k + α·d_k) > f(x_k) + c·α·∇f(x_k)ᵀ·d_k 时：
        α ← β · α    ← （步长缩减）

输出：步长 α
```

### 13.4.2 有限步终止的证明

**定理 13.3**（回溯线搜索有限步终止）：若 $f$ 连续可微，$\mathbf{d}_k$ 是下降方向（$\phi'(0) < 0$），则 Armijo 回溯线搜索在有限步内终止。

**证明**：

设 $\phi(\alpha) = f(\mathbf{x}_k + \alpha\mathbf{d}_k)$，则 $\phi(0) = f(\mathbf{x}_k)$，$\phi'(0) < 0$。

由 $f$ 的连续可微性，$\phi$ 也连续可微。在 $\alpha = 0$ 处做 Taylor 展开：

$$\phi(\alpha) = \phi(0) + \alpha\phi'(0) + o(\alpha)$$

当 $\alpha \to 0^+$ 时，$o(\alpha)/\alpha \to 0$，因此存在 $\bar{\delta} > 0$，对所有 $\alpha \in (0, \bar{\delta})$：

$$\phi(\alpha) - \phi(0) - c\alpha\phi'(0) = \alpha\phi'(0)(1 - c) + o(\alpha) < 0$$

（因为 $\phi'(0)(1-c) < 0$ 且 $o(\alpha)/\alpha \to 0$）。

所以当 $\alpha$ 足够小时，Armijo 条件成立。由于回溯过程 $\alpha = \bar{\alpha}, \beta\bar{\alpha}, \beta^2\bar{\alpha}, \ldots$ 是递减到 $0$ 的序列，必然在有限步（$m = O(\log_\beta(\bar{\delta}/\bar{\alpha}))$）内进入满足条件的区间。$\square$

### 13.4.3 参数选择的实践指南

**初始步长 $\bar{\alpha}$**：
- 梯度下降：取 $\bar{\alpha} = 1$（或 $\bar{\alpha} = 1/L$ 若 $L$ 已知）
- 牛顿法/拟牛顿法：强烈建议取 $\bar{\alpha} = 1$，因为在最优解附近二阶方法的单位步长是正确的
- 从上一步继承：$\bar{\alpha}_k = \alpha_{k-1}$，适用于步长变化平稳的问题

**缩减因子 $\beta$**：
- $\beta = 0.5$：最常用，每次步长减半
- $\beta = 0.8$：更精细的搜索，适合精度要求高的场景
- $\beta = 0.1$：粗放搜索，减少函数评估次数

**充分下降参数 $c$**：
- $c = 10^{-4}$：标准取值，宽松条件
- $c = 0.1$：较严格，要求更显著的下降

### 13.4.4 Wolfe 条件线搜索算法

满足 Wolfe 条件的线搜索比纯回溯更复杂，需要同时防止步长过大和过小，采用**括号法**（Bracketing）策略：

**算法 13.2（Wolfe 条件线搜索，概要）**：

```
阶段一：括号化（Bracketing）
    从 α_0 = 0 开始，逐步增大 α，直到找到一个区间 [α_lo, α_hi]，
    使得该区间内存在满足 Wolfe 条件的点。
    触发条件：
    (a) α 不满足充分下降条件（太大），或
    (b) φ(α) ≥ φ(α_prev)（函数值上升），或
    (c) φ'(α) ≥ 0（斜率非负，已越过极小值）

阶段二：精细化（Zoom）
    在区间 [α_lo, α_hi] 内用插值（二次/三次）缩小区间，
    直到找到满足 Wolfe 条件的步长。
```

这个"括号法 + 缩小"的策略保证在有限步内找到满足 Wolfe 条件的步长，被广泛用于 scipy、JAX 等科学计算库。

### 13.4.5 强 Wolfe 线搜索的收敛保证

**定理 13.4**（Zoutendijk 条件）：设 $f$ 连续可微且下有界，梯度 $L$-Lipschitz，每步搜索方向满足 $\mathbf{d}_k^\top\nabla f(\mathbf{x}_k) < 0$，步长满足 Wolfe 条件，且方向不过于垂直于梯度方向（$\cos\theta_k \geq \delta > 0$，$\theta_k$ 为 $\mathbf{d}_k$ 与 $-\nabla f(\mathbf{x}_k)$ 的夹角）。则：

$$\sum_{k=0}^{\infty} \cos^2\theta_k \|\nabla f(\mathbf{x}_k)\|^2 < +\infty$$

推论：$\liminf_{k\to\infty} \|\nabla f(\mathbf{x}_k)\| = 0$，即算法的某个极限点是驻点。

---

## 13.5 精确线搜索

### 13.5.1 精确线搜索的问题设置

精确线搜索求解一维最小化问题：

$$\alpha^* = \arg\min_{\alpha > 0} \phi(\alpha), \quad \phi(\alpha) = f(\mathbf{x}_k + \alpha \mathbf{d}_k)$$

对于一般函数 $f$，$\phi$ 没有解析解，需要数值方法。精确线搜索适用于：

- 每步方向计算代价极高，而函数值计算相对便宜
- 需要很高精度的应用场景
- 理论分析中作为参照基准

**单峰假设**：大多数精确线搜索方法假设 $\phi$ 在某区间 $[a, b]$ 内是**单峰函数**（Unimodal），即存在唯一极小值点，极小值点两侧函数值单调。

### 13.5.2 黄金分割法

**黄金分割法**（Golden Section Search）是一种无导数的精确线搜索方法，利用黄金分割比 $\tau = (\sqrt{5}-1)/2 \approx 0.618$ 进行区间缩减。

**核心思想**：在当前区间 $[a, b]$ 内选取两个内点 $c$ 和 $d$（$a < c < d < b$），比较 $\phi(c)$ 与 $\phi(d)$，根据比较结果缩小搜索区间：

- 若 $\phi(c) < \phi(d)$：极小值在 $[a, d]$ 内，令 $b \leftarrow d$
- 若 $\phi(c) \geq \phi(d)$：极小值在 $[c, b]$ 内，令 $a \leftarrow c$

内点的位置按黄金分割比确定：

$$c = a + (1-\tau)(b-a), \quad d = a + \tau(b-a)$$

其中黄金分割比 $\tau = (\sqrt{5}-1)/2$ 满足 $\tau^2 + \tau = 1$，这保证了每次迭代可以**复用一个函数值**（即每步只需一次新的函数值计算）。

**算法 13.3（黄金分割法）**：

```
输入：φ（目标函数），初始区间 [a, b]，容差 ε > 0
τ = (√5 - 1) / 2 ≈ 0.618
c = a + (1 - τ)(b - a)；  d = a + τ(b - a)
fc = φ(c)；  fd = φ(d)

重复：
    若 fc < fd：
        b ← d；  d ← c；  fd ← fc
        c ← a + (1-τ)(b-a)；  fc ← φ(c)
    否则：
        a ← c；  c ← d；  fc ← fd
        d ← a + τ(b-a)；  fd ← φ(d)

    若 b - a < ε，停止

输出：(a + b) / 2
```

**收敛速度**：每步区间长度缩小为原来的 $\tau \approx 0.618$ 倍。经过 $k$ 步后，区间长度为 $\tau^k (b-a)$。这是**线性收敛**，收敛比 $\tau \approx 0.618$。

要将区间长度缩减到 $\epsilon$，需要迭代次数：

$$k \geq \frac{\ln(\epsilon / (b-a))}{\ln \tau} \approx \frac{\ln((b-a)/\epsilon)}{0.481}$$

**优势**：不需要导数信息，实现简单，对目标函数仅要求连续性。

### 13.5.3 二分法（利用导数）

若 $\phi'(\alpha)$ 可以计算，可以用**二分法**（Bisection Method）更高效地求解 $\phi'(\alpha) = 0$。

**基本思想**：利用介值定理。若 $\phi'(a) < 0$（函数在减小）且 $\phi'(b) > 0$（函数在增大），则在 $[a, b]$ 内存在 $\phi'(\alpha^*) = 0$。

**算法 13.4（二分法求导数零点）**：

```
输入：φ'（方向导数函数），区间 [a, b] 满足 φ'(a)·φ'(b) < 0，容差 ε > 0

重复：
    m ← (a + b) / 2
    若 |φ'(m)| < ε，停止，输出 m
    若 φ'(a) · φ'(m) < 0：
        b ← m    （零点在左半段）
    否则：
        a ← m    （零点在右半段）

输出：m
```

**收敛速度**：每步区间减半，经过 $k$ 步后区间长度为 $(b-a)/2^k$。这也是线性收敛，但收敛比 $1/2 \approx 0.5$，优于黄金分割法（$0.618$）。

### 13.5.4 割线法（超线性收敛）

**割线法**（Secant Method）利用两点处的导数值，用线性插值近似 $\phi''$，从而更快地逼近 $\phi'(\alpha) = 0$：

$$\alpha_{k+1} = \alpha_k - \phi'(\alpha_k) \cdot \frac{\alpha_k - \alpha_{k-1}}{\phi'(\alpha_k) - \phi'(\alpha_{k-1})}$$

**收敛速度**：割线法具有**超线性收敛**（收敛阶约为 $1.618 = 1/\tau$），比二分法快但不如牛顿法（二阶收敛）稳定。

### 13.5.5 精确线搜索 vs 非精确线搜索

| 比较维度 | 精确线搜索 | 非精确线搜索（Armijo/Wolfe） |
|----------|-----------|------------------------------|
| 函数评估次数/步 | 多（$O(\log 1/\epsilon)$） | 少（通常 $1\sim 5$ 次） |
| 每步下降量 | 最大 | 有保证的下降 |
| 实现复杂度 | 较高 | 低（回溯）到中等（Wolfe） |
| 整体收敛速度 | 不一定更快 | 实践中通常更高效 |
| 适用场景 | 低维、$f$ 值便宜 | 高维、梯度主导代价 |
| 深度学习适用 | 否 | 部分（Armijo） |

**重要洞察**：精确线搜索在整体收敛中未必更优。以梯度下降为例，精确线搜索虽然每步下降最多，但下一步的梯度方向会与本步方向正交，导致"锯齿"行为。非精确线搜索配合好的搜索方向（如牛顿方向），往往整体更高效。

---

## 本章小结

| 概念 | 公式/要点 |
|------|-----------|
| 线搜索目标函数 | $\phi(\alpha) = f(\mathbf{x}_k + \alpha\mathbf{d}_k)$，$\phi'(0) = \nabla f(\mathbf{x}_k)^\top\mathbf{d}_k < 0$ |
| Armijo 条件（充分下降） | $\phi(\alpha) \leq \phi(0) + c_1\alpha\phi'(0)$，$c_1 \in (0,1)$，通常 $c_1 = 10^{-4}$ |
| Wolfe 曲率条件 | $\phi'(\alpha) \geq c_2\phi'(0)$，$c_2 \in (c_1, 1)$ |
| 强 Wolfe 曲率条件 | $|\phi'(\alpha)| \leq c_2|\phi'(0)|$，更接近一维极小值 |
| 回溯线搜索 | 从 $\bar\alpha$ 出发，每步乘以 $\beta \in (0,1)$，直至 Armijo 满足 |
| 回溯终止性 | $f$ 可微 + 下降方向 $\Rightarrow$ 有限步终止 |
| Zoutendijk 条件 | Wolfe 步长 $\Rightarrow$ $\sum_k\cos^2\theta_k\|\nabla f(\mathbf{x}_k)\|^2 < \infty$ |
| 黄金分割法收敛比 | $\tau = (\sqrt{5}-1)/2 \approx 0.618$，线性收敛，每步一次新函数评估 |
| 二分法收敛比 | $1/2$，线性收敛，需要导数 |
| 割线法收敛阶 | $\approx 1.618$，超线性收敛，需要两点导数 |
| L-BFGS 与 Wolfe | Wolfe 条件保证 $\mathbf{y}_k^\top\mathbf{s}_k > 0$，维持 Hessian 近似正定 |

**核心思想回顾**：

1. 线搜索将步长选择问题归结为一维优化，是大多数迭代优化算法的基础子程序
2. Armijo 条件防止步长过大，Wolfe 曲率条件防止步长过小，两者合力框定"合理步长区间"
3. 强 Wolfe 条件比 Wolfe 条件更严格，适用于拟牛顿法等需要精确步长的方法
4. 回溯线搜索实现简单、有限步终止，是实践中最常用的非精确线搜索
5. 精确线搜索（黄金分割、二分法）计算代价高，但无需调参，适用于低维或函数评估廉价的场景

---

## 深度学习应用：学习率调度与自适应步长

### 背景

深度学习中的"步长选择"问题具有特殊性：
- 损失函数 $\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n \ell(\mathbf{x}_i, y_i; \boldsymbol{\theta})$ 在大数据集上每次评估代价巨大
- 梯度本身是随机的（mini-batch SGD），传统线搜索假设（精确函数值/梯度）不成立
- 因此，深度学习通常使用**预设学习率调度**（Learning Rate Schedule）而非在线线搜索

然而，线搜索的思想在深度学习中以多种形式体现：Armijo 回溯偶尔用于训练初期，自适应学习率方法（Adam、AdaGrad）本质上是对每个参数的自适应步长选择。

### 学习率调度与自适应步长的 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 经典学习率调度策略
# ============================================================

def demo_lr_schedulers():
    """演示 PyTorch 内置的学习率调度器"""

    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    n_epochs = 100

    # (a) 阶梯式衰减：每 30 步学习率乘以 0.1
    scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # (b) 余弦退火：学习率按余弦曲线从初始值衰减到 0
    scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # (c) 指数衰减：每步学习率乘以 0.95
    scheduler_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # (d) ReduceLROnPlateau：当验证损失停止下降时自动降低学习率
    #     这是最接近"自适应线搜索"精神的调度器
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',       # 监控量越小越好
        factor=0.5,       # 学习率乘以 0.5（类似回溯的 β = 0.5）
        patience=10,      # 容忍 10 步无改善后才降低
        min_lr=1e-6,      # 学习率下限
        verbose=True
    )

    # 收集各调度器的学习率曲线
    lr_curves = {}
    for name, sched in [('StepLR', scheduler_step),
                         ('CosineAnnealing', scheduler_cos),
                         ('ExponentialLR', scheduler_exp)]:
        optimizer2 = optim.SGD(model.parameters(), lr=0.1)
        sched2 = type(sched)(optimizer2, **sched.state_dict().get('_last_lr', {})) \
            if False else None

        lr_history = []
        opt_tmp = optim.SGD(model.parameters(), lr=0.1)
        if name == 'StepLR':
            s = optim.lr_scheduler.StepLR(opt_tmp, step_size=30, gamma=0.1)
        elif name == 'CosineAnnealing':
            s = optim.lr_scheduler.CosineAnnealingLR(opt_tmp, T_max=n_epochs)
        else:
            s = optim.lr_scheduler.ExponentialLR(opt_tmp, gamma=0.95)

        for _ in range(n_epochs):
            lr_history.append(opt_tmp.param_groups[0]['lr'])
            s.step()

        lr_curves[name] = lr_history

    return lr_curves


# ============================================================
# 2. Armijo 回溯线搜索的完整实现
# ============================================================

def armijo_backtracking(
    f,
    grad_f,
    x: torch.Tensor,
    direction: torch.Tensor,
    alpha_init: float = 1.0,
    c: float = 1e-4,
    beta: float = 0.5,
    max_iter: int = 100
) -> tuple[float, int]:
    """
    Armijo 回溯线搜索

    参数：
        f        : 目标函数 f(x) -> scalar
        grad_f   : 梯度函数 grad_f(x) -> tensor
        x        : 当前点
        direction: 搜索方向（下降方向，满足 grad_f(x)^T direction < 0）
        alpha_init: 初始步长
        c        : Armijo 参数（充分下降系数）
        beta     : 步长缩减因子
        max_iter : 最大迭代次数（防止死循环）

    返回：
        alpha    : 满足 Armijo 条件的步长
        n_evals  : 函数评估次数
    """
    alpha = alpha_init
    fx = f(x)
    gx = grad_f(x)

    # 方向导数 φ'(0) = ∇f(x)^T d
    directional_deriv = torch.dot(gx.flatten(), direction.flatten()).item()
    assert directional_deriv < 0, "方向必须是下降方向"

    n_evals = 0
    for _ in range(max_iter):
        x_new = x + alpha * direction
        f_new = f(x_new)
        n_evals += 1

        # Armijo 条件：φ(α) ≤ φ(0) + c·α·φ'(0)
        if f_new.item() <= fx.item() + c * alpha * directional_deriv:
            break
        alpha *= beta

    return alpha, n_evals


def wolfe_line_search(
    f,
    grad_f,
    x: torch.Tensor,
    direction: torch.Tensor,
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 50
) -> tuple[float, int]:
    """
    强 Wolfe 条件线搜索（简化的括号法实现）

    满足：
        (SW1) f(x + α·d) ≤ f(x) + c1·α·∇f(x)^T·d  （充分下降）
        (SW2) |∇f(x + α·d)^T·d| ≤ c2·|∇f(x)^T·d|  （强曲率条件）
    """
    alpha_lo, alpha_hi = 0.0, float('inf')
    alpha = alpha_init

    fx = f(x).item()
    gx = grad_f(x)
    phi0 = fx
    dphi0 = torch.dot(gx.flatten(), direction.flatten()).item()

    alpha_prev = 0.0
    phi_prev = phi0
    n_evals = 0

    for i in range(max_iter):
        x_new = x + alpha * direction
        phi_alpha = f(x_new).item()
        n_evals += 1

        # 违反充分下降条件 或 函数值上升
        if phi_alpha > phi0 + c1 * alpha * dphi0 or (i > 0 and phi_alpha >= phi_prev):
            alpha_lo, alpha_hi = alpha_prev, alpha
            break

        dphi_alpha = torch.dot(grad_f(x_new).flatten(), direction.flatten()).item()
        n_evals += 1

        # 满足强 Wolfe 条件
        if abs(dphi_alpha) <= c2 * abs(dphi0):
            return alpha, n_evals

        if dphi_alpha >= 0:
            alpha_lo, alpha_hi = alpha, alpha_prev
            break

        alpha_prev = alpha
        phi_prev = phi_alpha
        alpha = min(2 * alpha, 10.0)  # 扩张步长

    # Zoom 阶段：在 [alpha_lo, alpha_hi] 内细化
    for _ in range(max_iter):
        alpha = (alpha_lo + alpha_hi) / 2  # 二分插值（简化）
        x_new = x + alpha * direction
        phi_alpha = f(x_new).item()
        n_evals += 1

        if phi_alpha > phi0 + c1 * alpha * dphi0 or phi_alpha >= f(x + alpha_lo * direction).item():
            alpha_hi = alpha
        else:
            dphi_alpha = torch.dot(grad_f(x_new).flatten(), direction.flatten()).item()
            n_evals += 1

            if abs(dphi_alpha) <= c2 * abs(dphi0):
                return alpha, n_evals

            if dphi_alpha * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha

        if abs(alpha_hi - alpha_lo) < 1e-10:
            break

    return alpha, n_evals


# ============================================================
# 3. 黄金分割精确线搜索
# ============================================================

def golden_section_search(
    phi,
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 200
) -> tuple[float, int]:
    """
    黄金分割法精确线搜索（单峰函数）

    参数：
        phi    : 一维目标函数 phi(alpha) -> float
        a, b   : 初始搜索区间（保证极小值在内）
        tol    : 收敛容差
        max_iter: 最大迭代次数

    返回：
        alpha* : 近似极小值点
        n_evals: 函数评估次数
    """
    tau = (np.sqrt(5) - 1) / 2  # 黄金分割比 ≈ 0.618

    # 初始两个内点
    c = a + (1 - tau) * (b - a)
    d = a + tau * (b - a)
    fc = phi(c)
    fd = phi(d)
    n_evals = 2

    for _ in range(max_iter):
        if b - a < tol:
            break

        if fc < fd:
            # 极小值在 [a, d]，缩小右界
            b = d
            d, fd = c, fc
            c = a + (1 - tau) * (b - a)
            fc = phi(c)
            n_evals += 1
        else:
            # 极小值在 [c, b]，缩小左界
            a = c
            c, fc = d, fd
            d = a + tau * (b - a)
            fd = phi(d)
            n_evals += 1

    return (a + b) / 2, n_evals


# ============================================================
# 4. 在梯度下降中对比三种线搜索策略
# ============================================================

def gradient_descent_with_line_search(
    f,
    grad_f,
    x0: torch.Tensor,
    line_search: str = 'armijo',
    max_iter: int = 200,
    tol: float = 1e-7
) -> tuple[list, list, list]:
    """
    梯度下降 + 指定线搜索策略

    line_search: 'armijo' | 'wolfe' | 'exact' | 'fixed'
    返回: (轨迹列表, 函数值列表, 步长列表)
    """
    x = x0.clone().requires_grad_(False)
    trajectory = [x.clone()]
    losses = [f(x).item()]
    alphas = []
    total_f_evals = 0

    for k in range(max_iter):
        g = grad_f(x)
        if g.norm().item() < tol:
            break

        d = -g  # 负梯度方向

        if line_search == 'armijo':
            alpha, evals = armijo_backtracking(f, grad_f, x, d)
            total_f_evals += evals

        elif line_search == 'wolfe':
            alpha, evals = wolfe_line_search(f, grad_f, x, d)
            total_f_evals += evals

        elif line_search == 'exact':
            # 用黄金分割法精确线搜索
            def phi(a):
                return f(x + a * d).item()

            # 先找一个包含极小值的区间
            a_gs, b_gs = 0.0, 1.0
            while phi(b_gs) < phi(a_gs):
                b_gs *= 2
                if b_gs > 1e4:
                    break
            alpha, evals = golden_section_search(phi, a_gs, b_gs, tol=1e-10)
            total_f_evals += evals

        else:  # 'fixed'，固定步长
            alpha = 0.01

        x = x + alpha * d
        trajectory.append(x.clone())
        losses.append(f(x).item())
        alphas.append(alpha)

    print(f"  [{line_search:8s}] 迭代 {len(losses)-1:4d} 步 | "
          f"最终损失 {losses[-1]:.2e} | 函数评估 {total_f_evals:4d} 次")
    return trajectory, losses, alphas


# ============================================================
# 5. 实验：对比各线搜索方法在二次问题上的表现
# ============================================================

def run_line_search_experiment():
    torch.manual_seed(42)

    # 构造一个条件数 κ=20 的二次函数
    # f(x) = 0.5 * x^T A x，A = diag(1, 5, 10, 20)
    diag_vals = torch.tensor([1.0, 5.0, 10.0, 20.0])
    A = torch.diag(diag_vals)
    b = torch.zeros(4)
    x_star = torch.zeros(4)  # 最优解
    f_star = 0.0

    def f(x):
        return 0.5 * (x * diag_vals * x).sum()

    def grad_f(x):
        return diag_vals * x

    x0 = torch.ones(4) * 2.0  # 初始点
    print("=" * 55)
    print(f"二次问题：A = diag{tuple(diag_vals.tolist())}，κ = {20}")
    print(f"初始点：x0 = {x0.tolist()}")
    print("=" * 55)

    methods = ['fixed', 'armijo', 'wolfe', 'exact']
    results = {}
    for method in methods:
        _, losses, alphas = gradient_descent_with_line_search(
            f, grad_f, x0, line_search=method, max_iter=500
        )
        results[method] = losses

    # 可视化收敛曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    labels = {
        'fixed': '固定步长 (α=0.01)',
        'armijo': 'Armijo 回溯',
        'wolfe': 'Wolfe 条件',
        'exact': '精确线搜索（黄金分割）'
    }
    styles = {'fixed': 'k--', 'armijo': 'b-', 'wolfe': 'r-', 'exact': 'g-'}

    for method in methods:
        losses = results[method]
        # 对齐纵轴为 f - f*
        gap = [l - f_star for l in losses]
        ax1.semilogy(gap, styles[method], label=labels[method], linewidth=2)

    ax1.set_xlabel('迭代次数 $k$')
    ax1.set_ylabel('$f(\\mathbf{x}_k) - f^*$（对数轴）')
    ax1.set_title('各线搜索方法的收敛比较（梯度下降）')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, min(200, min(len(r) for r in results.values()))])

    # 子图2：Armijo 步长接受区域可视化（一维截面）
    ax2 = axes[1]
    alpha_range = np.linspace(0, 2.0, 500)

    # 固定在 x0 处，沿负梯度方向的一维截面
    x_vis = torch.tensor([2.0, 0.1])
    d_vis = -torch.tensor([1.0, 0.5])  # 示意下降方向（非精确负梯度）
    d_vis = d_vis / d_vis.norm()

    def phi_vis(a):
        return 0.5 * ((x_vis + a * d_vis) ** 2).sum().item()

    phi_vals = [phi_vis(a) for a in alpha_range]
    phi0_val = phi_vis(0.0)
    dphi0_val = 2 * (x_vis * d_vis).sum().item()  # 梯度 = 2 * diag(1,1) * x，方向导数

    # Armijo 界（c=1e-4 和 c=0.1 两种）
    armijo_bound_tight = [phi0_val + 1e-4 * a * dphi0_val for a in alpha_range]
    armijo_bound_loose = [phi0_val + 0.1 * a * dphi0_val for a in alpha_range]

    ax2.plot(alpha_range, phi_vals, 'b-', linewidth=2.5, label=r'$\phi(\alpha)$（真实函数）')
    ax2.plot(alpha_range, armijo_bound_tight, 'g--', linewidth=1.5,
             label=r'Armijo 界 ($c=10^{-4}$)')
    ax2.plot(alpha_range, armijo_bound_loose, 'r--', linewidth=1.5,
             label=r'Armijo 界 ($c=0.1$)')
    ax2.axhline(phi0_val, color='gray', linestyle=':', linewidth=1)

    # 标注可接受区域
    accept_tight = [a for a, p, b in zip(alpha_range, phi_vals, armijo_bound_tight)
                    if p <= b]
    if accept_tight:
        ax2.axvspan(0, accept_tight[-1], alpha=0.08, color='green',
                    label='$c=10^{-4}$ 可接受区')
    accept_loose = [a for a, p, b in zip(alpha_range, phi_vals, armijo_bound_loose)
                    if p <= b]
    if accept_loose:
        ax2.axvspan(0, accept_loose[-1], alpha=0.08, color='red',
                    label='$c=0.1$ 可接受区')

    ax2.set_xlabel(r'步长 $\alpha$')
    ax2.set_ylabel(r'$\phi(\alpha)$')
    ax2.set_title('Armijo 条件的几何意义：可接受区域')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([min(phi_vals) * 0.8, phi0_val * 1.1])

    plt.tight_layout()
    plt.savefig('line_search_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n图像已保存为 line_search_comparison.png")


# ============================================================
# 6. 深度学习中的自适应学习率：Adam 的步长机制
# ============================================================

def demo_adaptive_lr():
    """
    Adam 优化器：自适应步长的直觉演示
    Adam 本质上是对每个参数维度独立估计"局部 Lipschitz 常数"，
    以此自动调整步长，类似于对每维参数做自适应线搜索。
    """
    torch.manual_seed(0)

    # 生成非线性回归数据
    n = 300
    X = torch.randn(n, 4)
    true_w = torch.tensor([2.0, -1.0, 0.5, 0.0])
    y = X @ true_w + 0.5 * torch.sin(X[:, 0]) + 0.1 * torch.randn(n)

    # 构建网络
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(4, 32), nn.Tanh(),
                nn.Linear(32, 16), nn.Tanh(),
                nn.Linear(16, 1)
            )
        def forward(self, x):
            return self.layers(x).squeeze()

    def train(optimizer_fn, n_epochs=200):
        net = Net()
        opt = optimizer_fn(net.parameters())
        crit = nn.MSELoss()
        loss_hist = []
        for _ in range(n_epochs):
            pred = net(X)
            loss = crit(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_hist.append(loss.item())
        return loss_hist

    print("\n训练神经网络：不同优化器（学习率策略）对比")
    configs = {
        'SGD (lr=0.1)':      lambda p: optim.SGD(p, lr=0.1),
        'SGD (lr=0.01)':     lambda p: optim.SGD(p, lr=0.01),
        'SGD+Momentum':      lambda p: optim.SGD(p, lr=0.05, momentum=0.9),
        'Adam (lr=1e-3)':    lambda p: optim.Adam(p, lr=1e-3),
        'Adam (lr=1e-2)':    lambda p: optim.Adam(p, lr=1e-2),
    }

    all_losses = {}
    for name, opt_fn in configs.items():
        losses = train(opt_fn)
        all_losses[name] = losses
        print(f"  {name:25s}：最终损失 = {losses[-1]:.4f}")

    # 绘制训练曲线
    plt.figure(figsize=(9, 5))
    styles = ['k--', 'k:', 'b-', 'r-', 'r--']
    for (name, losses), style in zip(all_losses.items(), styles):
        plt.semilogy(losses, style, label=name, linewidth=1.8)
    plt.xlabel('训练轮次')
    plt.ylabel('MSE 损失（对数轴）')
    plt.title('不同学习率策略的收敛比较（神经网络回归）')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图像已保存为 optimizer_comparison.png")


# ============================================================
# 7. ReduceLROnPlateau：深度学习中的"自适应回溯"
# ============================================================

def demo_reduce_lr_on_plateau():
    """
    ReduceLROnPlateau 调度器：
    - 当验证损失停止下降时，自动将学习率乘以 factor（如 0.5）
    - 这正是 Armijo 回溯精神在深度学习中的体现：
      检测到"充分下降"不足时，降低步长
    """
    torch.manual_seed(42)
    n, d = 500, 8
    X = torch.randn(n, d)
    y = (X[:, :3].sum(dim=1) > 0).float()

    model = nn.Sequential(
        nn.Linear(d, 32), nn.ReLU(),
        nn.Linear(32, 1), nn.Sigmoid()
    )
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

    # ReduceLROnPlateau：patience=5 步无改善，则 lr *= 0.5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
    )

    criterion = nn.BCELoss()
    loss_history = []
    lr_history = []

    for epoch in range(150):
        pred = model(X).squeeze()
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ReduceLROnPlateau 根据当前损失调整学习率
        scheduler.step(loss.item())

        loss_history.append(loss.item())
        lr_history.append(optimizer.param_groups[0]['lr'])

    # 绘制损失 + 学习率双轴图
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.semilogy(loss_history, 'b-', linewidth=1.5, label='训练损失')
    ax2.semilogy(lr_history, 'r--', linewidth=1.5, alpha=0.7, label='学习率')

    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('BCE 损失（对数轴）', color='b')
    ax2.set_ylabel('学习率（对数轴）', color='r')
    ax1.set_title('ReduceLROnPlateau：损失停滞时自动降低学习率（Armijo 精神）')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reduce_lr_on_plateau.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图像已保存为 reduce_lr_on_plateau.png")
    print(f"最终学习率：{lr_history[-1]:.2e}（初始 0.5，共降低 "
          f"{int(np.log2(0.5 / lr_history[-1]))} 次）")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("实验一：线搜索方法对比（梯度下降）")
    run_line_search_experiment()

    print("\n实验二：自适应学习率（神经网络）")
    demo_adaptive_lr()

    print("\n实验三：ReduceLROnPlateau（自适应步长调度）")
    demo_reduce_lr_on_plateau()
```

### 代码要点说明

1. **`armijo_backtracking`**：标准 Armijo 回溯实现，参数 `c=1e-4`，`beta=0.5`，每步乘以 $\beta$ 缩减步长直至充分下降条件满足

2. **`wolfe_line_search`**：实现强 Wolfe 条件线搜索的"括号化 + 缩小"策略：先扩张找到包含合法步长的区间，再二分精细化

3. **`golden_section_search`**：黄金分割精确线搜索，每步仅需一次新函数评估，区间以比率 $\tau \approx 0.618$ 缩减

4. **`ReduceLROnPlateau`**：深度学习中最接近 Armijo 回溯精神的调度器——检测到损失不再充分下降时，自动将学习率乘以 `factor`（默认 0.5），与回溯线搜索的 $\beta = 0.5$ 一脉相承

5. **Adam 的自适应步长本质**：Adam 维护每个参数的梯度一阶矩（$m_t$）和二阶矩（$v_t$）估计，有效步长为 $\alpha / \sqrt{v_t}$——梯度变化大的参数获得小步长，梯度稳定的参数获得大步长，这正是对每个参数维度自动做局部 Lipschitz 估计的体现

---

## 练习题

### 练习 13.1（基础）

设函数 $\phi(\alpha) = \alpha^3 - 3\alpha^2 + 4$（定义在 $\alpha \geq 0$ 上），当前点对应 $\phi(0) = 4$，方向导数 $\phi'(0) = 0$...

实际取 $f: \mathbb{R}^2 \to \mathbb{R}$，$f(\mathbf{x}) = x_1^2 + 4x_2^2$，当前点 $\mathbf{x}_k = (2, 1)^\top$，搜索方向 $\mathbf{d}_k = -\nabla f(\mathbf{x}_k)$。

(a) 计算 $\nabla f(\mathbf{x}_k)$ 和方向导数 $\phi'(0) = \nabla f(\mathbf{x}_k)^\top \mathbf{d}_k$。

(b) 写出辅助函数 $\phi(\alpha) = f(\mathbf{x}_k + \alpha \mathbf{d}_k)$ 的表达式，并化简为关于 $\alpha$ 的多项式。

(c) 取 $c = 10^{-4}$，Armijo 条件变为 $\phi(\alpha) \leq \phi(0) + c\alpha\phi'(0)$。验证：$\alpha = 0.2$ 是否满足 Armijo 条件？$\alpha = 1$ 是否满足？

---

### 练习 13.2（基础）

对上题中的函数 $f(\mathbf{x}) = x_1^2 + 4x_2^2$，从 $\mathbf{x}_0 = (2, 1)^\top$ 出发，搜索方向为负梯度方向。

(a) 推导精确线搜索的最优步长 $\alpha^* = \arg\min_{\alpha \geq 0} \phi(\alpha)$ 的解析表达式。（提示：令 $\phi'(\alpha^*) = 0$）

(b) 计算 $\mathbf{x}_1 = \mathbf{x}_0 + \alpha^* \mathbf{d}_0$，再计算 $\mathbf{d}_1 = -\nabla f(\mathbf{x}_1)$。

(c) 验证梯度下降在精确线搜索下，相邻两步搜索方向正交：$\mathbf{d}_1^\top \mathbf{d}_0 = 0$。

(d) 这种"锯齿"性质意味着什么？对收敛速度有何影响？

---

### 练习 13.3（中级）

**Wolfe 条件的理解**。设 $\phi: \mathbb{R}_+ \to \mathbb{R}$ 是可微函数，$\phi(0) = 1$，$\phi'(0) = -2$，Wolfe 参数 $c_1 = 10^{-4}$，$c_2 = 0.9$。

(a) 写出 Wolfe 条件（W1）和（W2）对 $\phi(\alpha)$ 和 $\phi'(\alpha)$ 的要求。

(b) 若 $\phi(0.5) = 0.2$，$\phi'(0.5) = -0.1$，验证 $\alpha = 0.5$ 是否满足强 Wolfe 条件。

(c) 若 $\phi(0.5) = 0.2$，$\phi'(0.5) = -2.5$，分析 $\alpha = 0.5$ 不满足 Wolfe 条件（但满足 Armijo 条件）的原因，并解释这意味着步长太小还是太大。

(d) 解释为什么拟牛顿法（L-BFGS）必须使用 Wolfe 条件，而不能只用 Armijo 条件。

---

### 练习 13.4（中级）

**黄金分割法的手动迭代**。对函数 $\phi(\alpha) = (\alpha - 1)^2 + 0.5$（极小值在 $\alpha^* = 1$），初始区间 $[0, 3]$，黄金分割比 $\tau = (\sqrt{5}-1)/2 \approx 0.618$。

(a) 计算第1步的内点 $c_1 = a + (1-\tau)(b-a)$，$d_1 = a + \tau(b-a)$，及函数值 $\phi(c_1)$，$\phi(d_1)$。

(b) 根据比较结果缩短区间，计算第2步的区间和新内点（注意复用一个函数值）。

(c) 经过3步迭代后，区间长度是多少？与初始区间 $[0,3]$ 的比是多少？验证该比值接近 $\tau^3$。

(d) 与等分法（每步区间缩小至 $1/2$）相比，黄金分割法有何劣势？它的优势是什么？

---

### 练习 13.5（提高）

**实验与分析**：实现并比较不同线搜索方法在 Rosenbrock 函数上的表现。

Rosenbrock 函数：$f(x_1, x_2) = (1-x_1)^2 + 100(x_2 - x_1^2)^2$，最优解 $\mathbf{x}^* = (1, 1)^\top$，$f^* = 0$。

(a) 推导 $\nabla f(\mathbf{x})$ 和 $\nabla^2 f(\mathbf{x})$ 的表达式。

(b) 在 Python 中实现梯度下降 + Armijo 回溯线搜索，从 $\mathbf{x}_0 = (-1, 1)^\top$ 出发，运行 5000 步，绘制函数值收敛曲线（对数纵轴）。

(c) 将搜索方向改为牛顿方向 $\mathbf{d}_k = -[\nabla^2 f(\mathbf{x}_k)]^{-1}\nabla f(\mathbf{x}_k)$，仍使用 Armijo 回溯，观察收敛速度的变化。

(d) 比较两种方法在（1）迭代次数、（2）总函数评估次数、（3）最终精度三个指标上的差异，分析原因。

---

## 练习答案

### 答案 13.1

$f(\mathbf{x}) = x_1^2 + 4x_2^2$，$\mathbf{x}_k = (2, 1)^\top$。

**(a)** 梯度：$\nabla f(\mathbf{x}) = (2x_1, 8x_2)^\top = (4, 8)^\top$。

搜索方向：$\mathbf{d}_k = -\nabla f(\mathbf{x}_k) = (-4, -8)^\top$。

方向导数：$\phi'(0) = \nabla f(\mathbf{x}_k)^\top \mathbf{d}_k = (4)(-4) + (8)(-8) = -16 - 64 = -80$。

**(b)** 辅助函数：

$$\phi(\alpha) = f\!\left(\begin{pmatrix}2 \\ 1\end{pmatrix} + \alpha\begin{pmatrix}-4 \\ -8\end{pmatrix}\right) = (2-4\alpha)^2 + 4(1-8\alpha)^2$$

$$= 4 - 16\alpha + 16\alpha^2 + 4(1 - 16\alpha + 64\alpha^2) = 20 - 80\alpha + 272\alpha^2$$

验证：$\phi(0) = 20 = f(2,1)$，$\phi'(0) = -80$（与(a)一致）。

**(c)** Armijo 条件：$\phi(\alpha) \leq 20 + 10^{-4} \cdot \alpha \cdot (-80) = 20 - 0.008\alpha$。

- $\alpha = 0.2$：$\phi(0.2) = 20 - 80(0.2) + 272(0.04) = 20 - 16 + 10.88 = 14.88$，界为 $20 - 0.0016 = 19.9984$。$14.88 \leq 19.9984$，**满足** ✓
- $\alpha = 1$：$\phi(1) = 20 - 80 + 272 = 212$，界为 $20 - 0.008 = 19.992$。$212 > 19.992$，**不满足** ✗

---

### 答案 13.2

**(a)** $\phi(\alpha) = 20 - 80\alpha + 272\alpha^2$（由答案 13.1(b)）。

令 $\phi'(\alpha) = -80 + 544\alpha = 0$，解得：

$$\alpha^* = \frac{80}{544} = \frac{5}{34} \approx 0.1471$$

**(b)** 更新点：

$$\mathbf{x}_1 = \begin{pmatrix}2 \\ 1\end{pmatrix} + \frac{5}{34}\begin{pmatrix}-4 \\ -8\end{pmatrix} = \begin{pmatrix}2 - 20/34 \\ 1 - 40/34\end{pmatrix} = \begin{pmatrix}48/34 \\ -6/34\end{pmatrix} \approx \begin{pmatrix}1.412 \\ -0.176\end{pmatrix}$$

新梯度：$\nabla f(\mathbf{x}_1) = (2 \times 48/34, 8 \times (-6/34))^\top = (96/34, -48/34)^\top$

下一步方向：$\mathbf{d}_1 = -(96/34, -48/34)^\top$

**(c)** 验证正交性：

$$\mathbf{d}_0^\top \mathbf{d}_1 = (-4) \cdot \left(-\frac{96}{34}\right) + (-8) \cdot \left(\frac{48}{34}\right) = \frac{384}{34} - \frac{384}{34} = 0 \checkmark$$

**(d)** 锯齿性质说明：精确线搜索后，下一步梯度方向与本步方向正交。这导致梯度下降呈"锯齿形"轨迹，在椭圆等高线上反复横跨，收敛极慢（条件数 $\kappa = 4$，收敛率 $\rho = (4-1)/(4+1) = 3/5$）。这是梯度下降配合精确线搜索的固有缺陷，动量方法和共轭梯度法正是为了克服这一问题而提出的。

---

### 答案 13.3

**(a)** Wolfe 条件（$\phi(0)=1$，$\phi'(0)=-2$，$c_1=10^{-4}$，$c_2=0.9$）：

- **W1（充分下降）**：$\phi(\alpha) \leq 1 + 10^{-4} \cdot \alpha \cdot (-2) = 1 - 2\times10^{-4}\alpha$
- **W2（曲率）**：$\phi'(\alpha) \geq 0.9 \times (-2) = -1.8$
- **强 W2**：$|\phi'(\alpha)| \leq 0.9 \times |-2| = 1.8$，即 $-1.8 \leq \phi'(\alpha) \leq 1.8$

**(b)** 验证 $\alpha = 0.5$，$\phi(0.5) = 0.2$，$\phi'(0.5) = -0.1$：

- W1：$0.2 \leq 1 - 10^{-4} = 0.9999$？$0.2 \leq 0.9999$，**满足** ✓
- 强 W2：$|-0.1| = 0.1 \leq 1.8$？**满足** ✓

因此 $\alpha = 0.5$ **满足强 Wolfe 条件**。

**(c)** $\alpha = 0.5$，$\phi(0.5) = 0.2$，$\phi'(0.5) = -2.5$：

- W1（Armijo）：$0.2 \leq 0.9999$，**满足** ✓
- W2（曲率）：$-2.5 \geq -1.8$？$-2.5 < -1.8$，**不满足** ✗

原因：$\phi'(0.5) = -2.5 < \phi'(0) = -2$，函数仍在以比初始更快的速度下降，意味着我们还未充分利用下降潜力，**步长太小**。

**(d)** 原因：L-BFGS 的 BFGS 更新要求 $\mathbf{y}_k^\top\mathbf{s}_k > 0$（以保持 Hessian 近似正定）。可以证明，Wolfe 条件中的曲率条件 W2 等价于 $\mathbf{y}_k^\top\mathbf{s}_k > 0$：

$$\mathbf{y}_k^\top\mathbf{s}_k = \alpha_k [\nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)]^\top\mathbf{d}_k = \alpha_k[\phi'(\alpha_k) - \phi'(0)] \geq \alpha_k(c_2-1)\phi'(0) > 0$$

（最后一步利用 W2 和 $\phi'(0) < 0$，$c_2 < 1$）。若只用 Armijo 条件，$\mathbf{y}_k^\top\mathbf{s}_k$ 可能为负，导致 Hessian 近似不定，BFGS 公式失效。

---

### 答案 13.4

$\phi(\alpha) = (\alpha-1)^2 + 0.5$，$[a,b] = [0, 3]$，$\tau \approx 0.618$，$1-\tau \approx 0.382$。

**(a)** 第1步内点：

$$c_1 = 0 + 0.382 \times 3 = 1.146, \quad d_1 = 0 + 0.618 \times 3 = 1.854$$

$$\phi(c_1) = (1.146-1)^2 + 0.5 = 0.0213 + 0.5 = 0.5213$$

$$\phi(d_1) = (1.854-1)^2 + 0.5 = 0.7294 + 0.5 = 1.2294$$

由于 $\phi(c_1) < \phi(d_1)$，极小值在 $[0, d_1] = [0, 1.854]$，令 $b \leftarrow 1.854$。

**(b)** 第2步：复用 $c_1 = 1.146$（作为新的 $d$），$\phi(c_1) = 0.5213$。

新区间 $[0, 1.854]$，长度 $= 1.854$。

新内点（左侧）：$c_2 = 0 + 0.382 \times 1.854 = 0.708$

$$\phi(c_2) = (0.708-1)^2 + 0.5 = 0.0852 + 0.5 = 0.5852$$

由于 $\phi(c_2) = 0.5852 > \phi(d_2) = \phi(c_1) = 0.5213$，极小值在 $[c_2, b] = [0.708, 1.854]$，令 $a \leftarrow 0.708$。

**(c)** 第3步区间：$[0.708, 1.854]$，长度 $= 1.146 = 3 \times 0.618^2 \approx 1.144$（约等于 $\tau^2 \times 3$）。

经3步后区间长度为初始的 $1.146/3 \approx 0.382 \approx (1-\tau) = \tau^2$（即 $\tau^3$ 的近似，此处有偏差因为第1步缩减比不完全是 $\tau$）。精确地，每步区间缩减比为 $\tau$，3步后为 $\tau^3 \approx 0.618^3 \approx 0.236$，即区间长 $\approx 0.708$，与实际接近。

**(d)** 与等分法（二分法）比较：

- **劣势**：黄金分割每步区间缩小为 $\tau \approx 0.618$，而二分法为 $0.5$，**黄金分割收敛更慢**
- **优势**：黄金分割每步只需**一次新函数评估**（复用一个旧值），而朴素的三点搜索需要两次。等分法寻找极小值时需要比较多点，而黄金分割利用了单峰性质的结构；此外，黄金分割**不需要导数**，适用于函数可微性未知的情况

---

### 答案 13.5

**(a)** Rosenbrock 函数 $f(x_1, x_2) = (1-x_1)^2 + 100(x_2-x_1^2)^2$：

$$\frac{\partial f}{\partial x_1} = -2(1-x_1) - 400x_1(x_2 - x_1^2) = -2 + 2x_1 - 400x_1 x_2 + 400x_1^3$$

$$\frac{\partial f}{\partial x_2} = 200(x_2 - x_1^2)$$

$$\nabla^2 f = \begin{pmatrix} 2 - 400x_2 + 1200x_1^2 & -400x_1 \\ -400x_1 & 200 \end{pmatrix}$$

在最优解 $(1,1)^\top$ 处：$\nabla^2 f = \begin{pmatrix} 802 & -400 \\ -400 & 200 \end{pmatrix}$，条件数 $\kappa \approx 2508$，极度病态。

**(b)** 梯度下降 + Armijo 回溯（代码框架，参考章节代码中 `armijo_backtracking` 函数）：

```python
import torch

def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    g1 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    g2 = 200*(x[1] - x[0]**2)
    return torch.tensor([g1, g2])

x = torch.tensor([-1.0, 1.0])
losses = []
for _ in range(5000):
    g = rosenbrock_grad(x)
    d = -g
    # Armijo 回溯
    alpha = 1.0
    fx = rosenbrock(x)
    dphi0 = (g * d).sum().item()
    while rosenbrock(x + alpha*d).item() > fx.item() + 1e-4 * alpha * dphi0:
        alpha *= 0.5
        if alpha < 1e-15:
            break
    x = x + alpha * d
    losses.append(rosenbrock(x).item())
```

梯度下降在 Rosenbrock 函数上收敛极慢（需要数万步才能接近最优），因为该函数在最优解附近的条件数极大（约 2500）。

**(c)** 牛顿方向 + Armijo 回溯：

```python
def rosenbrock_hessian(x):
    h11 = 2 - 400*x[1].item() + 1200*x[0].item()**2
    h12 = -400*x[0].item()
    h22 = 200.0
    return torch.tensor([[h11, h12], [h12, h22]])

# 牛顿方向：d = -H^{-1} g
H = rosenbrock_hessian(x)
g = rosenbrock_grad(x)
d = -torch.linalg.solve(H, g)  # 牛顿方向
```

牛顿法通常在数十步内收敛至高精度（在最优解附近具有二阶收敛），但每步需要计算并求解 Hessian，代价为 $O(n^3)$（此处 $n=2$ 可忽略）。

**(d)** 对比分析（典型结果）：

| 指标 | 梯度下降 + Armijo | 牛顿法 + Armijo |
|------|-----------------|----------------|
| 收敛迭代次数 | $\sim 5000$ 步仍未收敛 | $\sim 50$ 步达到 $10^{-10}$ |
| 函数评估次数 | 约 $5000\times 3 = 15000$ | 约 $50 \times 5 = 250$ |
| 最终精度 | $f \approx 10^{-4}$ | $f \approx 10^{-20}$ |

**原因分析**：Rosenbrock 函数极度病态（条件数 $\approx 2500$），梯度下降沿负梯度方向在弯曲的"香蕉形"山谷中锯齿前行，极难直接到达谷底。牛顿法通过 Hessian 逆变换消除条件数的影响，在局部二次近似框架内精确指向最优方向，因此具有二阶收敛速度。这正体现了线搜索方法的核心原则：**好的搜索方向比好的步长更重要**。

---

*本章结束。下一章：[信赖域方法](./14-trust-region.md)*
