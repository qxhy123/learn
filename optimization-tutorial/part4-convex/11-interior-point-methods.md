# 第11章：内点法

> **前置章节**：第6章（牛顿法与拟牛顿法）、第9章（对偶理论）、第10章（KKT条件）
>
> **难度**：★★★★★

---

## 学习目标

学完本章，你将能够：

1. **理解障碍函数原理**：掌握对数障碍函数 $\phi(\mathbf{x}) = -\sum_i \log(-g_i(\mathbf{x}))$ 如何将不等式约束内化为目标函数，理解其几何含义与数学性质
2. **分析中心路径**：理解参数化问题族 $\min t f + \phi$ 的解形成中心路径，以及中心路径与KKT条件之间的深刻联系
3. **推导障碍函数法**：掌握路径跟踪算法的完整框架，包括内层牛顿迭代、外层参数更新、以及停止准则的设计
4. **掌握原始-对偶内点法**：理解如何同时更新原始变量和对偶变量，推导牛顿步与对偶更新公式，分析多项式时间复杂性
5. **应用于大规模凸优化**：了解内点法在线性规划、半正定规划中的应用，以及在约束机器学习问题中的实践意义

---

## 11.1 障碍函数法

### 11.1.1 约束优化的困难

考虑标准形式的不等式约束凸优化问题：

$$\begin{aligned}
\min \quad & f(\mathbf{x}) \\
\text{s.t.} \quad & g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m \\
& \mathbf{h}(\mathbf{x}) = A\mathbf{x} - \mathbf{b} = \mathbf{0}
\end{aligned}$$

其中 $f: \mathbb{R}^n \to \mathbb{R}$ 和 $g_i: \mathbb{R}^n \to \mathbb{R}$ 均为凸函数，$A \in \mathbb{R}^{p \times n}$。

不等式约束的存在给优化带来了根本性的困难：

- **可行域边界**：最优解可能在可行域的边界上，即某些约束 $g_i(\mathbf{x}^*) = 0$ 处取等号（称为**活跃约束**），无法直接用无约束算法处理
- **内外分离**：可行域内部和边界的性质截然不同，算法需要特别处理"触碰边界"的情形
- **组合复杂性**：不知道哪些约束在最优解处活跃，暴力枚举是指数级的

**障碍函数法（Barrier Method）** 的核心思想是：**将不等式约束转化为目标函数中的惩罚项**，使得在可行域内部进行无约束优化，自动"感知"边界的存在。

### 11.1.2 可行域内点的概念

**定义 11.1（严格可行域）**：问题的严格可行域（interior of the feasible set）定义为：

$$\mathcal{F}^{\circ} = \{\mathbf{x} \in \mathbb{R}^n \mid g_i(\mathbf{x}) < 0 \; \forall i, \; A\mathbf{x} = \mathbf{b}\}$$

即所有不等式约束严格满足的可行点的集合。

**Slater 条件**：若 $\mathcal{F}^{\circ} \neq \emptyset$（即存在严格可行点），则强对偶定理成立，KKT条件是最优性的充要条件。

内点法的一个关键前提是 **Slater 条件成立**，即问题存在严格内点。在此条件下，内点法具有完备的理论保证。

### 11.1.3 障碍函数的直觉

设想在可行域边界处设置一道"无形的墙"：当迭代点趋近约束边界 $g_i(\mathbf{x}) \to 0^-$ 时，惩罚项趋向 $+\infty$，从而将迭代点"推回"可行域内部。

**理想的障碍函数**应满足：

1. 在严格可行域 $\mathcal{F}^{\circ}$ 上有定义且光滑（实际上是 $C^\infty$）
2. 当 $g_i(\mathbf{x}) \to 0^-$（趋近约束边界）时趋向 $+\infty$
3. 在可行域内部足够光滑，可以使用牛顿法求解

**对数函数**是满足上述要求的最自然选择：$-\log(-u)$ 在 $u < 0$ 时有定义，当 $u \to 0^-$ 时趋向 $+\infty$，且无穷次可微。

---

## 11.2 对数障碍函数

### 11.2.1 定义与基本性质

**定义 11.2（对数障碍函数）**：对于 $m$ 个不等式约束 $g_i(\mathbf{x}) \leq 0$，定义**对数障碍函数**（logarithmic barrier function）为：

$$\boxed{\phi(\mathbf{x}) = -\sum_{i=1}^{m} \log\left(-g_i(\mathbf{x})\right)}$$

定义域为严格可行域 $\mathcal{F}^{\circ}$。

**基本性质**：

1. **凸性**：若每个 $g_i$ 是凸函数，则 $\phi$ 是凸函数（复合凸函数的对数仍为凸函数）
2. **光滑性**：$\phi \in C^\infty(\mathcal{F}^{\circ})$，任意阶导数均存在
3. **趋近行为**：当 $\mathbf{x}$ 趋近约束边界（即某个 $g_i(\mathbf{x}) \to 0^-$）时，$\phi(\mathbf{x}) \to +\infty$
4. **梯度公式**：

$$\nabla \phi(\mathbf{x}) = -\sum_{i=1}^{m} \frac{\nabla g_i(\mathbf{x})}{g_i(\mathbf{x})} = \sum_{i=1}^{m} \frac{\nabla g_i(\mathbf{x})}{-g_i(\mathbf{x})}$$

5. **Hessian 公式**：

$$\nabla^2 \phi(\mathbf{x}) = \sum_{i=1}^{m} \frac{\nabla g_i(\mathbf{x}) \nabla g_i(\mathbf{x})^\top}{g_i(\mathbf{x})^2} - \sum_{i=1}^{m} \frac{\nabla^2 g_i(\mathbf{x})}{g_i(\mathbf{x})}$$

### 11.2.2 线性不等式约束的特例

对于线性约束 $\mathbf{a}_i^\top \mathbf{x} \leq b_i$（即 $g_i(\mathbf{x}) = \mathbf{a}_i^\top \mathbf{x} - b_i$），障碍函数退化为：

$$\phi(\mathbf{x}) = -\sum_{i=1}^{m} \log\left(b_i - \mathbf{a}_i^\top \mathbf{x}\right)$$

此时：
$$\nabla \phi(\mathbf{x}) = \sum_{i=1}^{m} \frac{\mathbf{a}_i}{b_i - \mathbf{a}_i^\top \mathbf{x}}$$

$$\nabla^2 \phi(\mathbf{x}) = \sum_{i=1}^{m} \frac{\mathbf{a}_i \mathbf{a}_i^\top}{(b_i - \mathbf{a}_i^\top \mathbf{x})^2} \succ 0$$

Hessian 恒正定，这使得线性约束问题的内点法具有良好的数值性质。

### 11.2.3 参数化近似问题

引入**精度参数** $t > 0$（也称**障碍参数**），构造参数化近似问题：

$$\begin{aligned}
\mathbf{x}^*(t) = \arg\min_{\mathbf{x}} \quad & t \cdot f(\mathbf{x}) + \phi(\mathbf{x}) \\
\text{s.t.} \quad & A\mathbf{x} = \mathbf{b}
\end{aligned} \tag{11.1}$$

等价地（除以 $t$）：

$$\mathbf{x}^*(t) = \arg\min_{\mathbf{x}} \quad f(\mathbf{x}) + \frac{1}{t}\phi(\mathbf{x}), \quad \text{s.t.} \quad A\mathbf{x} = \mathbf{b}$$

**参数 $t$ 的作用**：

- $t$ 较小时：障碍项 $\frac{1}{t}\phi$ 权重大，最优点被"推离"边界，远离约束，解偏离原问题
- $t$ 较大时：障碍项权重小，目标函数 $f$ 主导，解趋近于原问题的最优解 $\mathbf{x}^*$

**直觉**：随着 $t \to +\infty$，问题 (11.1) 的解 $\mathbf{x}^*(t)$ 收敛到原始问题的最优解 $\mathbf{x}^*$。

### 11.2.4 次优界

**定理 11.1（障碍函数近似误差）**：设 $\mathbf{x}^*(t)$ 是参数 $t$ 下的最优解，$\mathbf{x}^*$ 是原始问题的最优解，则：

$$f(\mathbf{x}^*(t)) - f(\mathbf{x}^*) \leq \frac{m}{t}$$

其中 $m$ 是不等式约束的数量。

**证明**：设原始问题的最优对偶变量为 $\boldsymbol{\lambda}^*$（满足 KKT 条件），利用强对偶定理和对数障碍函数的性质可以建立此界。

此定理给出了一个关键的**停止准则**：当 $\frac{m}{t} \leq \epsilon$ 时，即 $t \geq \frac{m}{\epsilon}$，当前解的次优性不超过 $\epsilon$。

**例题 11.1**：考虑简单的凸二次规划：

$$\min_{x \in \mathbb{R}} \; \frac{1}{2}x^2, \quad \text{s.t.} \quad x \geq 1$$

等价为 $g(x) = 1 - x \leq 0$。障碍函数为 $\phi(x) = -\log(x - 1)$（$x > 1$）。

参数化问题（$t > 0$）：

$$\min_{x > 1} \; \frac{t}{2}x^2 - \log(x - 1)$$

一阶条件：$tx - \frac{1}{x-1} = 0$，即 $t x(x-1) = 1$。

解为 $x^*(t) = \frac{1}{2}\left(1 + \frac{1}{t} + \sqrt{\left(1 + \frac{1}{t}\right)^2 - \frac{4}{t}}\right)$（取大根）。

当 $t \to \infty$ 时，$x^*(t) \to 1 = x^*$，与定理预期一致。

---

## 11.3 中心路径

### 11.3.1 中心路径的定义

**定义 11.3（中心路径）**：若对每个 $t > 0$，问题 (11.1) 有唯一最优解 $\mathbf{x}^*(t)$，则参数化最优解族：

$$\mathcal{C} = \{\mathbf{x}^*(t) \mid t > 0\}$$

称为原始问题的**中心路径**（central path）。

在凸目标和凸约束的假设下，若 Slater 条件成立，则中心路径存在且唯一。中心路径是从严格可行域内部出发、随 $t$ 增大而趋向最优解 $\mathbf{x}^*$ 的一条光滑曲线。

### 11.3.2 中心路径的最优性条件

中心路径上的每一点 $\mathbf{x}^*(t)$ 是含等式约束问题的无约束（关于不等式约束）最优解，满足 KKT 条件：

$$t \nabla f(\mathbf{x}^*(t)) + \nabla \phi(\mathbf{x}^*(t)) + A^\top \boldsymbol{\nu} = \mathbf{0}$$

代入 $\nabla \phi$ 的表达式：

$$t \nabla f(\mathbf{x}^*(t)) - \sum_{i=1}^{m} \frac{\nabla g_i(\mathbf{x}^*(t))}{g_i(\mathbf{x}^*(t))} + A^\top \boldsymbol{\nu} = \mathbf{0}$$

定义**隐含对偶变量**：

$$\lambda_i^*(t) = -\frac{1}{t \cdot g_i(\mathbf{x}^*(t))} > 0, \quad i = 1, \ldots, m$$

注意由于 $g_i(\mathbf{x}^*(t)) < 0$（严格可行域），所以 $\lambda_i^*(t) > 0$，满足对偶可行性。

代入上式，除以 $t$：

$$\nabla f(\mathbf{x}^*(t)) + \sum_{i=1}^{m} \lambda_i^*(t) \nabla g_i(\mathbf{x}^*(t)) + \frac{1}{t}A^\top \boldsymbol{\nu} = \mathbf{0}$$

### 11.3.3 中心路径与 KKT 条件的联系

**中心路径上的点所满足的条件**（将等式约束显式写出）：

$$\begin{cases}
\nabla f(\mathbf{x}) + \sum_{i=1}^{m} \lambda_i \nabla g_i(\mathbf{x}) + A^\top \boldsymbol{\nu} = \mathbf{0} & \text{（平稳条件）} \\
-\lambda_i g_i(\mathbf{x}) = \frac{1}{t}, \quad i = 1, \ldots, m & \text{（扰动互补松弛条件）} \\
g_i(\mathbf{x}) \leq 0, \quad A\mathbf{x} = \mathbf{b} & \text{（原始可行性）} \\
\lambda_i > 0 & \text{（对偶可行性）}
\end{cases} \tag{11.2}$$

**对比原始 KKT 条件**：

$$\begin{cases}
\nabla f(\mathbf{x}^*) + \sum_{i=1}^{m} \lambda_i^* \nabla g_i(\mathbf{x}^*) + A^\top \boldsymbol{\nu}^* = \mathbf{0} \\
\lambda_i^* g_i(\mathbf{x}^*) = 0, \quad i = 1, \ldots, m \quad \text{（精确互补松弛）} \\
g_i(\mathbf{x}^*) \leq 0, \quad A\mathbf{x}^* = \mathbf{b} \\
\lambda_i^* \geq 0
\end{cases}$$

**核心洞察**：中心路径是 KKT 条件的**扰动版本**（perturbed KKT conditions）：

> 精确互补松弛条件 $\lambda_i g_i(\mathbf{x}) = 0$ 被放松为 $-\lambda_i g_i(\mathbf{x}) = \frac{1}{t}$。

当 $t \to \infty$ 时，扰动量 $\frac{1}{t} \to 0$，中心路径上的点趋近于满足精确 KKT 条件的原始最优解。

### 11.3.4 对偶间隙的精确估计

**定理 11.2（中心路径的对偶间隙）**：对中心路径上的点 $\mathbf{x}^*(t)$，对应的对偶间隙为：

$$f(\mathbf{x}^*(t)) - d^* = \frac{m}{t}$$

其中 $d^*$ 是对偶问题的最优值（等于 $f^*$，强对偶成立）。

**证明**：由强对偶定理，对偶间隙等于原始目标值与对偶目标值之差。利用 Lagrange 对偶函数和对偶变量 $\boldsymbol{\lambda}^*(t)$ 的定义：

$$f(\mathbf{x}^*(t)) - d(\boldsymbol{\lambda}^*(t), \boldsymbol{\nu}^*(t)) = \sum_{i=1}^{m} \lambda_i^*(t) \cdot (-g_i(\mathbf{x}^*(t))) = \sum_{i=1}^{m} \frac{1}{t} = \frac{m}{t}$$

其中每一项 $\lambda_i^*(t) \cdot (-g_i(\mathbf{x}^*(t))) = \frac{1}{t}$ 直接由扰动互补松弛条件得出。$\square$

**推论**：给定精度要求 $\epsilon > 0$，只需将 $t$ 增大到 $t \geq \frac{m}{\epsilon}$，即可保证次优性 $\leq \epsilon$。

### 11.3.5 中心路径的几何直觉

对于简单的二维问题，中心路径的几何图像非常直观：

```
          约束边界
          ↑
    ______↑_____
   |      ↑    |
   |   x*(t₃) |         t₃ > t₂ > t₁
   |      ↑   |
   | x*(t₂)  |         ← 中心路径
   |      ↑  |
   |  x*(t₁) |
   |__________|

   x*(t) 随 t→∞ 趋向最优解 x*（可能在边界上）
```

中心路径从可行域的"中心"出发，随着 $t$ 增大，逐渐靠近最优解，但始终保持在严格可行域内部——这正是"内点法"名称的由来。

---

## 11.4 障碍函数法的算法

### 11.4.1 路径跟踪算法框架

障碍函数法（又称**路径跟踪法**，path-following method）的基本思路是：从小的 $t$ 出发，逐步增大 $t$，每次用牛顿法求解当前参数下的优化问题，用上一步的解作为下一步的初始点，"跟踪"中心路径直至对偶间隙足够小。

**算法 11.1（障碍函数路径跟踪法）**：

```
输入：严格可行初始点 x⁰（满足 g_i(x⁰) < 0, Ax⁰ = b）
      初始参数 t⁰ > 0，增长因子 μ > 1（典型取 μ = 10 ~ 20）
      精度要求 ε > 0，牛顿法容差 ε_newton > 0

输出：ε-最优解 x

x ← x⁰,  t ← t⁰

while m/t > ε do
    # 内层迭代：用牛顿法求解参数为 t 的障碍函数问题
    x ← Newton(x, t)    # 以当前 x 为初始点，求解 min t·f(x) + φ(x) s.t. Ax=b

    # 外层更新：增大参数 t
    t ← μ · t
end while

return x
```

**两层嵌套结构**：
- **外层循环**：更新参数 $t$，迭代次数为 $O\!\left(\log\frac{m/\epsilon}{t^0 \cdot \epsilon_{\text{newton}}}\right)$
- **内层迭代**：对给定 $t$，用牛顿法求解等式约束的光滑优化问题

### 11.4.2 内层牛顿迭代

对于固定的 $t$，内层问题是求解含等式约束的光滑凸优化：

$$\min_{\mathbf{x}} \; F_t(\mathbf{x}) = t f(\mathbf{x}) + \phi(\mathbf{x}), \quad \text{s.t.} \; A\mathbf{x} = \mathbf{b}$$

令 $\tilde{F}(\mathbf{x}) = F_t(\mathbf{x})$，梯度和 Hessian 为：

$$\nabla \tilde{F}(\mathbf{x}) = t \nabla f(\mathbf{x}) - \sum_{i=1}^{m} \frac{\nabla g_i(\mathbf{x})}{g_i(\mathbf{x})}$$

$$\nabla^2 \tilde{F}(\mathbf{x}) = t \nabla^2 f(\mathbf{x}) + \sum_{i=1}^{m} \frac{\nabla g_i(\mathbf{x}) \nabla g_i(\mathbf{x})^\top}{g_i(\mathbf{x})^2} - \sum_{i=1}^{m} \frac{\nabla^2 g_i(\mathbf{x})}{g_i(\mathbf{x})}$$

**含等式约束的牛顿步**：在当前点 $\mathbf{x}$ 处，对 $F_t$ 进行二阶 Taylor 展开并最小化，同时满足等式约束的线性化：

$$\min_{\Delta \mathbf{x}} \; \tilde{F}(\mathbf{x}) + \nabla \tilde{F}(\mathbf{x})^\top \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^\top \nabla^2 \tilde{F}(\mathbf{x}) \Delta\mathbf{x}, \quad \text{s.t.} \; A\Delta\mathbf{x} = \mathbf{0}$$

等价于求解**KKT 线性系统**：

$$\begin{pmatrix} \nabla^2 \tilde{F}(\mathbf{x}) & A^\top \\ A & \mathbf{0} \end{pmatrix} \begin{pmatrix} \Delta\mathbf{x} \\ \boldsymbol{\nu} \end{pmatrix} = -\begin{pmatrix} \nabla \tilde{F}(\mathbf{x}) \\ \mathbf{0} \end{pmatrix} \tag{11.3}$$

其中 $\boldsymbol{\nu}$ 是等式约束对应的 Lagrange 乘子。

**牛顿减少量（Newton decrement）**：定义：

$$\lambda(\mathbf{x})^2 = \Delta\mathbf{x}^\top \nabla^2 \tilde{F}(\mathbf{x}) \Delta\mathbf{x}$$

内层牛顿迭代的停止准则：当 $\frac{\lambda(\mathbf{x})^2}{2} \leq \epsilon_{\text{newton}}$ 时终止。

### 11.4.3 回溯线搜索的修正

内点法中的线搜索需要额外保证**可行性**：更新后的点必须仍在严格可行域内，即所有不等式约束严格满足。

**带可行性保证的回溯线搜索**：

```
给定当前点 x，牛顿步 Δx，参数 α_start=1，回退因子 β ∈ (0,1)，Armijo 常数 c ∈ (0,0.5)

α ← α_start

# 首先确保步长足够小以保持可行性
while ∃i : g_i(x + α·Δx) ≥ 0 do
    α ← β·α
end while

# 然后按 Armijo 条件缩减步长
while F_t(x + α·Δx) > F_t(x) + c·α·∇F_t(x)ᵀΔx do
    α ← β·α
end while

x ← x + α·Δx
```

由于 $\nabla^2 \tilde{F}(\mathbf{x}) \succ 0$（内点处障碍函数 Hessian 正定），牛顿方向始终是下降方向，线搜索必然成功。

### 11.4.4 寻找严格可行初始点

实际应用中，寻找严格可行初始点本身可以是一个挑战。一种标准方法是**阶段一（Phase I）方法**：引入松弛变量 $s > 0$，构造辅助问题：

$$\min_{\mathbf{x}, s} \; s \quad \text{s.t.} \; g_i(\mathbf{x}) \leq s, \; i=1,\ldots,m, \; A\mathbf{x} = \mathbf{b}$$

若此问题的最优值 $s^* < 0$，则可从辅助问题的最优解中提取严格可行点。若 $s^* \geq 0$，则原始问题不可行。

### 11.4.5 参数 μ 的选取策略

增长因子 $\mu$ 的选择影响算法性能：

| $\mu$ 的大小 | 外层迭代次数 | 内层牛顿迭代次数（每次） | 总体效率 |
|-------------|------------|------------------------|---------|
| $\mu$ 太小（如 $\mu = 2$） | 多（路径跟踪精密） | 少（初始点质量好） | 总步骤多 |
| $\mu$ 太大（如 $\mu = 1000$） | 少（大步跳跃） | 多（初始点偏离太远） | 内层收敛慢 |
| $\mu \approx 10 \sim 20$ | 适中 | 适中（5~50步） | **实践最优** |

**定理 11.3（总复杂性）**：设每次内层牛顿迭代最多需要 $c_{\text{Newton}}$ 步，则路径跟踪法达到精度 $\epsilon$ 所需的总牛顿步数为：

$$O\!\left(c_{\text{Newton}} \cdot \log\!\left(\frac{m}{\epsilon t^0}\right) \cdot \log\mu\right)$$

---

## 11.5 原始-对偶内点法

### 11.5.1 动机与改进思路

障碍函数法（11.4节）本质上是在原始空间中跟踪中心路径。但每次只更新原始变量，对偶变量 $\boldsymbol{\lambda}$ 是事后隐式确定的，这带来了一定的信息损失。

**原始-对偶内点法**（Primal-Dual Interior Point Method）同时维护和更新**原始变量** $\mathbf{x}$ 和**对偶变量** $\boldsymbol{\lambda}, \boldsymbol{\nu}$，将中心路径条件 (11.2) 直接作为非线性方程组，用牛顿法联合求解。

### 11.5.2 扰动 KKT 方程组

考虑线性等式约束的问题：

$$\min_{\mathbf{x}} \; f(\mathbf{x}), \quad \text{s.t.} \; g_i(\mathbf{x}) \leq 0, \; i=1,\ldots,m, \; A\mathbf{x} = \mathbf{b}$$

中心路径条件 (11.2) 构成以下非线性方程组（以 $\frac{1}{t}$ 记 $\tau$）：

$$r_\tau(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = \begin{pmatrix} r_{\text{dual}} \\ r_{\text{cent}} \\ r_{\text{prim}} \end{pmatrix} = \begin{pmatrix} \nabla f(\mathbf{x}) + Df(\mathbf{x})^\top\boldsymbol{\lambda} + A^\top\boldsymbol{\nu} \\ -\text{diag}(\boldsymbol{\lambda}) g(\mathbf{x}) - \tau \mathbf{1} \\ A\mathbf{x} - \mathbf{b} \end{pmatrix} = \mathbf{0} \tag{11.4}$$

其中：
- $r_{\text{dual}} \in \mathbb{R}^n$：**对偶残差**（平稳条件）
- $r_{\text{cent}} \in \mathbb{R}^m$：**中心性残差**（扰动互补松弛条件）
- $r_{\text{prim}} \in \mathbb{R}^p$：**原始残差**（等式约束满足性）
- $Df(\mathbf{x}) = [\nabla g_1(\mathbf{x}), \ldots, \nabla g_m(\mathbf{x})]^\top \in \mathbb{R}^{m \times n}$：约束函数的 Jacobian

### 11.5.3 牛顿步的推导

对方程组 $r_\tau(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = \mathbf{0}$ 应用牛顿迭代，在当前点 $(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ 处线性化：

$$\frac{\partial r_\tau}{\partial (\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})} \begin{pmatrix} \Delta\mathbf{x} \\ \Delta\boldsymbol{\lambda} \\ \Delta\boldsymbol{\nu} \end{pmatrix} = -r_\tau(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$$

展开 Jacobian 矩阵：

$$\begin{pmatrix} \nabla^2 f(\mathbf{x}) + \sum_i \lambda_i \nabla^2 g_i(\mathbf{x}) & Df(\mathbf{x})^\top & A^\top \\ -\text{diag}(\boldsymbol{\lambda}) Df(\mathbf{x}) & -\text{diag}(g(\mathbf{x})) & \mathbf{0} \\ A & \mathbf{0} & \mathbf{0} \end{pmatrix} \begin{pmatrix} \Delta\mathbf{x} \\ \Delta\boldsymbol{\lambda} \\ \Delta\boldsymbol{\nu} \end{pmatrix} = -\begin{pmatrix} r_{\text{dual}} \\ r_{\text{cent}} \\ r_{\text{prim}} \end{pmatrix} \tag{11.5}$$

这是一个 $(n + m + p) \times (n + m + p)$ 的线性方程组，称为**原始-对偶牛顿方程**。

### 11.5.4 对偶更新公式

对于线性约束 $g_i(\mathbf{x}) = \mathbf{a}_i^\top \mathbf{x} - b_i$，$Df(\mathbf{x}) = A_c$（约束矩阵），方程组 (11.5) 的第二行化简为：

$$-\text{diag}(\boldsymbol{\lambda}) A_c \Delta\mathbf{x} - \text{diag}(g(\mathbf{x})) \Delta\boldsymbol{\lambda} = -r_{\text{cent}}$$

由此解出**对偶更新**（$\Delta\boldsymbol{\lambda}$ 的显式表达）：

$$\Delta\lambda_i = \frac{-r_{\text{cent},i} - \lambda_i (A_c \Delta\mathbf{x})_i}{g_i(\mathbf{x})}$$

$$\boxed{\Delta\lambda_i = \frac{\lambda_i}{-g_i(\mathbf{x})} (A_c \Delta\mathbf{x})_i + \frac{\tau}{-g_i(\mathbf{x})} - \lambda_i}$$

代入化简，$\Delta\boldsymbol{\lambda}$ 可以在求出 $\Delta\mathbf{x}$ 后显式计算，无需求解更大的线性方程组。

**简化的牛顿方程**（消去 $\Delta\boldsymbol{\lambda}$）：将第二行的 $\Delta\boldsymbol{\lambda}$ 代入第一行，得到关于 $(\Delta\mathbf{x}, \Delta\boldsymbol{\nu})$ 的缩减系统：

$$\begin{pmatrix} H_{\text{pd}} & A^\top \\ A & \mathbf{0} \end{pmatrix} \begin{pmatrix} \Delta\mathbf{x} \\ \Delta\boldsymbol{\nu} \end{pmatrix} = -\begin{pmatrix} \tilde{r}_{\text{dual}} \\ r_{\text{prim}} \end{pmatrix} \tag{11.6}$$

其中**原始-对偶 Hessian** 为：

$$H_{\text{pd}} = \nabla^2 f(\mathbf{x}) + \sum_{i=1}^{m} \lambda_i \nabla^2 g_i(\mathbf{x}) + \sum_{i=1}^{m} \frac{\lambda_i}{-g_i(\mathbf{x})} \nabla g_i(\mathbf{x}) \nabla g_i(\mathbf{x})^\top$$

这与障碍函数法中 $\nabla^2 \tilde{F}_t(\mathbf{x})$ 非常类似，但对偶变量 $\boldsymbol{\lambda}$ 作为独立变量被同时跟踪，而非由 $\lambda_i = -\frac{1}{t g_i(\mathbf{x})}$ 隐式确定。

### 11.5.5 步长与代理对偶间隙

在原始-对偶方法中，步长 $\alpha$ 需要同时保证：

1. **对偶可行性**：$\boldsymbol{\lambda} + \alpha \Delta\boldsymbol{\lambda} > \mathbf{0}$（所有对偶变量保持正）
2. **原始严格可行**：$g_i(\mathbf{x} + \alpha \Delta\mathbf{x}) < 0$（所有约束严格满足）
3. **残差下降**（充分减少条件，类似 Armijo 条件）

**最大可行步长**：

$$\alpha_{\max} = \min\left(1, \min_{i: \Delta\lambda_i < 0} \frac{-\lambda_i}{\Delta\lambda_i}, \min_{i: (A\Delta x)_i > (b - Ax)_i} \text{可行步长}\right)$$

取 $\alpha = \min(1, s \cdot \alpha_{\max})$，其中 $s \in (0,1)$（典型取 $s = 0.99$）为安全因子。

**代理对偶间隙**：定义当前点 $(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ 处的代理对偶间隙（surrogate duality gap）：

$$\hat{\eta} = -\mathbf{g}(\mathbf{x})^\top \boldsymbol{\lambda} = -\sum_{i=1}^{m} g_i(\mathbf{x}) \lambda_i > 0$$

当 $(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ 在中心路径上时，$\hat{\eta} = \frac{m}{t}$，因此代理对偶间隙是对偶间隙的可计算近似。

**更新参数**：在每次牛顿步后，更新 $\tau$（即 $1/t$）为：

$$\tau = \frac{\sigma}{m} \hat{\eta}_{\text{new}}$$

其中 $\sigma \in (0,1)$（典型取 $\sigma = 0.1$）控制向中心路径的"向心"程度。

### 11.5.6 原始-对偶内点法完整算法

**算法 11.2（原始-对偶内点法）**：

```
输入：严格可行初始点 (x, λ, ν)，其中 λ > 0, g(x) < 0, Ax = b
      精度参数 ε_abs > 0（绝对误差），ε_rel > 0（相对误差）
      向心参数 σ = 0.1，安全因子 s = 0.99

输出：ε-最优的原始-对偶解 (x*, λ*, ν*)

while True do
    # 计算代理对偶间隙
    η̂ = -g(x)ᵀλ

    # 停止准则：原始残差、对偶残差、对偶间隙均充分小
    if ‖r_prim‖₂ ≤ ε_abs,  ‖r_dual‖₂ ≤ ε_abs,  η̂ ≤ ε_abs then
        break
    end if

    # 更新 τ（中心化参数）
    τ = σ · η̂ / m

    # 求解牛顿方程 (11.5)，得到 (Δx, Δλ, Δν)
    Solve: KKT system (11.5)

    # 确定最大可行步长
    α_max = max step s.t. λ + α·Δλ > 0 and g(x + α·Δx) < 0

    # 回溯线搜索（基于残差范数下降）
    α = s · α_max
    while ‖r_τ(x + α·Δx, λ + α·Δλ, ν + α·Δν)‖ > (1 - c·α)‖r_τ(x,λ,ν)‖ do
        α ← β · α
    end while

    # 更新变量
    x ← x + α·Δx
    λ ← λ + α·Δλ
    ν ← ν + α·Δν
end while

return (x, λ, ν)
```

### 11.5.7 多项式时间复杂性

原始-对偶内点法是著名的**多项式时间算法**（polynomial-time algorithm），这是其相比单纯形法的理论优势。

**定理 11.4（内点法的多项式复杂性）**：对于线性规划问题，原始-对偶内点法在 $O(\sqrt{m} \log(m/\epsilon))$ 次迭代内达到精度 $\epsilon$，每次迭代需要求解一个 $n \times n$ 线性方程组，时间复杂度为 $O(n^3)$。

**对比各算法的复杂性**：

| 算法 | 理论复杂性 | 实践表现 |
|------|-----------|---------|
| 单纯形法 | 指数级（最坏情况） | 通常很快（实践中近乎线性） |
| 椭球法 | 多项式（$O(n^6 L)$） | 实践中极慢 |
| 内点法 | 多项式（$O(\sqrt{n} L)$） | 实践中快速（大规模问题） |

其中 $L$ 是问题数据规模（二进制编码长度）。

### 11.5.8 原始-对偶法与障碍函数法的比较

| 特征 | 障碍函数法 | 原始-对偶内点法 |
|------|-----------|---------------|
| 维护变量 | 仅 $\mathbf{x}$ | $(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ |
| 对偶变量 | 隐式（由 $t$ 决定） | 显式更新 |
| 每步求解规模 | $n \times n$（缩减后） | $(n+m+p) \times (n+m+p)$ |
| 收敛速度 | 超线性（内层牛顿） | 超线性~二次 |
| 实现复杂度 | 较简单 | 较复杂 |
| 实践效率 | 良好 | **通常更优** |
| 理论分析 | 较清晰 | 更复杂 |

**实践建议**：对于中等规模问题，使用成熟的内点法求解器（如 CVXOPT、MOSEK、Gurobi）远优于自行实现，这些求解器已针对数值稳定性和效率进行了大量优化。

---

## 本章小结

| 概念 | 定义/公式 | 核心作用 |
|------|---------|---------|
| 对数障碍函数 | $\phi(\mathbf{x}) = -\sum_i \log(-g_i(\mathbf{x}))$ | 将不等式约束内化为惩罚项，在严格可行域内光滑 |
| 参数化近似问题 | $\min \; t f(\mathbf{x}) + \phi(\mathbf{x}), \; A\mathbf{x}=\mathbf{b}$ | 以参数 $t$ 控制近似精度，$t \to \infty$ 趋近原始问题 |
| 中心路径 | $\mathcal{C} = \{\mathbf{x}^*(t) \mid t > 0\}$ | 连接可行域内部与最优解的光滑曲线 |
| 扰动 KKT 条件 | $-\lambda_i g_i(\mathbf{x}) = \frac{1}{t}$ | 中心路径是精确互补松弛的连续松弛 |
| 对偶间隙估计 | $f(\mathbf{x}^*(t)) - f^* = \frac{m}{t}$ | 给出精确可计算的次优界 |
| 路径跟踪法 | 外层更新 $t$，内层牛顿法求解 | 实现内点法的基本框架，$O(\log(1/\epsilon))$ 外层步 |
| 原始-对偶法 | 联合更新 $(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ | 更快的实践收敛，多项式理论保证 |
| 原始-对偶 Hessian | $H_{\text{pd}} = \nabla^2 f + \sum \lambda_i \nabla^2 g_i + \sum \frac{\lambda_i}{-g_i} \nabla g_i \nabla g_i^\top$ | 同时编码原始和对偶曲率信息 |
| 代理对偶间隙 | $\hat{\eta} = -\mathbf{g}(\mathbf{x})^\top \boldsymbol{\lambda}$ | 当前解的对偶间隙可计算近似 |
| 多项式复杂性（LP） | $O(\sqrt{m} \log(m/\epsilon))$ 次迭代 | 内点法相比单纯形法的理论优势 |

**核心公式速查**：

$$\text{对数障碍函数：}\phi(\mathbf{x}) = -\sum_{i=1}^{m} \log(-g_i(\mathbf{x}))$$

$$\text{中心路径参数化：}\mathbf{x}^*(t) = \arg\min\left\{t f(\mathbf{x}) + \phi(\mathbf{x}) \mid A\mathbf{x} = \mathbf{b}\right\}$$

$$\text{次优界：}f(\mathbf{x}^*(t)) - f^* \leq \frac{m}{t}$$

$$\text{扰动 KKT：}-\lambda_i g_i(\mathbf{x}) = \frac{1}{t}, \quad \lambda_i = \frac{-1}{t \cdot g_i(\mathbf{x})} > 0$$

$$\text{原始-对偶牛顿方程：}\begin{pmatrix} H_{\text{pd}} & A^\top \\ A & \mathbf{0} \end{pmatrix} \begin{pmatrix} \Delta\mathbf{x} \\ \Delta\boldsymbol{\nu} \end{pmatrix} = -\begin{pmatrix} \tilde{r}_{\text{dual}} \\ r_{\text{prim}} \end{pmatrix}$$

---

## 深度学习应用：大规模凸优化与约束学习

### 背景：深度学习中的约束优化需求

现代深度学习中越来越多的场景需要处理约束：

- **公平性约束**：要求模型在不同群体上的预测误差接近相等
- **安全约束**：自动驾驶、医疗等场景中需要满足安全性指标
- **结构约束**：权重矩阵满足某些几何约束（如正半定、单调性）
- **资源约束**：模型推理时间、内存占用的约束
- **物理约束**：物理信息神经网络（PINN）中需要满足微分方程

内点法为处理这些约束提供了坚实的理论基础，但大规模场景下需要专门的工程优化。

### 应用一：约束深度学习（Constrained DNN Training）

**问题设置**：训练神经网络时施加均值等式约束和方差不等式约束：

$$\min_{\boldsymbol{\theta}} \; \mathcal{L}(\boldsymbol{\theta}) \quad \text{s.t.} \; c_j(\boldsymbol{\theta}) \leq 0, \; j = 1, \ldots, m$$

对于不可行的 mini-batch 梯度步，可以用**投影梯度法**或**增广 Lagrange 法**处理，但对于精确约束，内点法提供了更严格的保证。

```python
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer

# ============================================================
# 示例 1：内点法原理演示 - 简单凸二次规划
# ============================================================

def solve_qp_barrier(Q, c, A_ineq, b_ineq, t_init=1.0, mu=10.0, eps=1e-8,
                     max_outer=50, max_inner=50):
    """
    用障碍函数法求解凸二次规划：
        min  (1/2) x^T Q x + c^T x
        s.t. A_ineq x <= b_ineq

    参数说明：
        Q:        正定矩阵 (n x n), numpy array
        c:        线性项 (n,), numpy array
        A_ineq:   不等式约束矩阵 (m x n), numpy array
        b_ineq:   不等式约束右端项 (m,), numpy array
        t_init:   初始障碍参数
        mu:       增长因子
        eps:      精度要求
        max_outer: 外层最大迭代次数
        max_inner: 内层牛顿迭代最大步数
    """
    n = len(c)
    m = len(b_ineq)

    # 寻找严格可行初始点（此处假设 x=0 可行，实际需要 Phase I）
    x = np.zeros(n)
    # 检查初始可行性，若不可行则向可行域内移动
    slack = b_ineq - A_ineq @ x
    if np.any(slack <= 0):
        # 简单处理：缩放使其可行（实践中用 Phase I）
        x = x - 1.1 * np.min(slack) * np.ones(n) / n
        slack = b_ineq - A_ineq @ x
        assert np.all(slack > 0), "无法找到严格可行初始点"

    t = t_init
    history = {'x': [], 'f': [], 'gap': [], 't': []}

    def obj(x):
        """原始目标函数"""
        return 0.5 * x @ Q @ x + c @ x

    def barrier_obj(x, t_val):
        """障碍函数问题目标：t*f(x) + phi(x)"""
        slack = b_ineq - A_ineq @ x
        if np.any(slack <= 0):
            return np.inf
        return t_val * obj(x) - np.sum(np.log(slack))

    def barrier_grad(x, t_val):
        """障碍函数梯度：t*∇f + ∇φ"""
        slack = b_ineq - A_ineq @ x
        grad_f = t_val * (Q @ x + c)           # t * ∇f
        grad_phi = A_ineq.T @ (1.0 / slack)    # -∑ ∇g_i / g_i（此处 g_i = a_i x - b_i）
        return grad_f + grad_phi

    def barrier_hess(x, t_val):
        """障碍函数 Hessian：t*H_f + H_phi"""
        slack = b_ineq - A_ineq @ x
        hess_f = t_val * Q
        # H_phi = ∑ a_i a_i^T / slack_i^2
        weights = 1.0 / (slack ** 2)
        hess_phi = (A_ineq.T * weights) @ A_ineq
        return hess_f + hess_phi

    print(f"{'外层步':>6} {'t':>10} {'f(x)':>12} {'对偶间隙':>12} {'可行残差':>12}")
    print("-" * 60)

    for outer_iter in range(max_outer):
        # 停止准则：对偶间隙 m/t < eps
        dual_gap = m / t
        f_val = obj(x)
        history['x'].append(x.copy())
        history['f'].append(f_val)
        history['gap'].append(dual_gap)
        history['t'].append(t)

        slack_val = b_ineq - A_ineq @ x
        feas_res = -np.min(slack_val)  # 越小越严格可行

        if outer_iter % 5 == 0:
            print(f"{outer_iter:>6} {t:>10.2e} {f_val:>12.6f} {dual_gap:>12.2e} {feas_res:>12.2e}")

        if dual_gap < eps:
            print(f"\n收敛！外层迭代 {outer_iter} 步，对偶间隙 = {dual_gap:.2e}")
            break

        # ---- 内层牛顿迭代 ----
        for inner_iter in range(max_inner):
            g = barrier_grad(x, t)
            H = barrier_hess(x, t)

            # 求解牛顿方程 H * Δx = -g（此问题无等式约束）
            try:
                delta_x = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                break

            # 牛顿减少量
            newton_decrement_sq = -g @ delta_x  # = Δx^T H Δx
            if newton_decrement_sq / 2 <= 1e-10:
                break  # 内层收敛

            # 回溯线搜索（保证可行性 + Armijo 条件）
            alpha = 1.0
            beta_ls = 0.5   # 退缩因子
            c_armijo = 0.01

            # 1) 保证可行性
            max_feas_iters = 100
            for _ in range(max_feas_iters):
                x_new = x + alpha * delta_x
                slack_new = b_ineq - A_ineq @ x_new
                if np.all(slack_new > 0):
                    break
                alpha *= beta_ls

            # 2) Armijo 条件
            f_curr = barrier_obj(x, t)
            for _ in range(100):
                x_new = x + alpha * delta_x
                if barrier_obj(x_new, t) <= f_curr + c_armijo * alpha * (g @ delta_x):
                    break
                alpha *= beta_ls

            x = x + alpha * delta_x

        # 外层更新：增大 t
        t = mu * t

    return x, history


# 测试：求解简单约束二次规划
# min  x1^2 + x2^2 + x1 + x2
# s.t. x1 + x2 <= 1
#      x1 >= -2  (等价于 -x1 <= 2)
#      x2 >= -2

Q = np.array([[2., 0.], [0., 2.]])
c = np.array([1., 1.])
A_ineq = np.array([[1., 1.], [-1., 0.], [0., -1.]])
b_ineq = np.array([1., 2., 2.])

print("=" * 60)
print("障碍函数法求解凸二次规划")
print("min x1^2 + x2^2 + x1 + x2")
print("s.t. x1 + x2 <= 1,  x1 >= -2,  x2 >= -2")
print("=" * 60)

x_opt, hist = solve_qp_barrier(Q, c, A_ineq, b_ineq,
                                t_init=1.0, mu=15.0, eps=1e-7)

print(f"\n最优解: x* = {x_opt}")
print(f"最优值: f* = {0.5 * x_opt @ Q @ x_opt + c @ x_opt:.8f}")
print(f"理论解: x* = (-0.5, -0.5), f* = -0.5（无约束解，活跃约束为 none）")
# 无约束最优解 x* = -Q^{-1}c/2 = (-0.5, -0.5)，满足所有约束，故约束非活跃


# ============================================================
# 示例 2：原始-对偶内点法 - 线性规划标准形式
# ============================================================

def solve_lp_primal_dual(c_obj, A_eq, b_eq, max_iter=100, eps=1e-9, sigma=0.1):
    """
    原始-对偶内点法求解线性规划（标准形式）：
        min   c^T x
        s.t.  Ax = b,  x >= 0

    使用经典的 Mehrotra 预测-校正格式的简化版。
    KKT 条件（原始-对偶）：
        A^T y + s = c      （对偶可行性）
        Ax = b             （原始可行性）
        x_i s_i = 0        （互补松弛，松弛为 μ = σ η / n）
        x >= 0, s >= 0
    """
    m, n = A_eq.shape

    # 初始化（简单初始化，实践中用 Mehrotra 初始点）
    x = np.ones(n)
    y = np.zeros(m)
    s = np.ones(n)

    print(f"\n{'迭代':>4} {'原始残差':>12} {'对偶残差':>12} {'对偶间隙':>12} {'μ':>10}")
    print("-" * 55)

    for k in range(max_iter):
        # 计算残差
        r_prim = A_eq @ x - b_eq                      # 原始残差 (m,)
        r_dual = A_eq.T @ y + s - c_obj                # 对偶残差 (n,)
        gap = x @ s                                     # 对偶间隙 = x^T s
        mu = sigma * gap / n                            # 互补松弛目标

        prim_res = np.linalg.norm(r_prim)
        dual_res = np.linalg.norm(r_dual)

        if k % 10 == 0:
            print(f"{k:>4} {prim_res:>12.2e} {dual_res:>12.2e} {gap:>12.2e} {gap/n:>10.2e}")

        # 停止准则
        if prim_res < eps and dual_res < eps and gap < eps:
            print(f"\n收敛！迭代 {k} 步")
            break

        # 构造牛顿方程（消去 Δs：Δs = -s - S X^{-1} r_xs + S X^{-1} A Δx）
        # 其中 r_xs = XSe - μe（中心性残差）
        X = np.diag(x)
        S = np.diag(s)
        X_inv = np.diag(1.0 / x)

        r_xs = x * s - mu * np.ones(n)    # 中心性残差

        # 缩减 KKT 系统（关于 Δx, Δy）：
        # [A (S^{-1} X) A^T] Δy = ... （AXS^{-1}A^T 正定）
        # 通过 Schur 补得到：A (X/S) A^T Δy = rhs

        XS_inv = x / s   # 元素级 x_i / s_i
        M = A_eq @ (XS_inv[:, None] * A_eq.T)   # A diag(x/s) A^T，正定矩阵

        rhs_y = -(r_prim - A_eq @ (XS_inv * (r_dual - r_xs / x)))

        # 求解对偶步 Δy
        try:
            delta_y = np.linalg.solve(M, rhs_y)
        except np.linalg.LinAlgError:
            print("线性方程组奇异，终止")
            break

        # 回代求 Δx, Δs
        delta_s = -r_dual + A_eq.T @ delta_y        # ... 注意符号
        # 修正: Δs = -(A^T δy - Δ(A^Ty+s-c)) = r_dual - A^T δy (符号)
        delta_s = r_dual - A_eq.T @ delta_y
        delta_x = -XS_inv * (r_xs / x + delta_s)   # 从中心性方程回代

        # 确定最大步长（保持 x>0, s>0）
        def max_step(v, dv):
            """计算使 v + α dv > 0 的最大 α"""
            neg_idx = dv < 0
            if not np.any(neg_idx):
                return 1.0
            return min(1.0, 0.99 * np.min(-v[neg_idx] / dv[neg_idx]))

        alpha_p = max_step(x, delta_x)   # 原始步长
        alpha_d = max_step(s, delta_s)   # 对偶步长

        # 更新
        x = x + alpha_p * delta_x
        y = y + alpha_d * delta_y
        s = s + alpha_d * delta_s

    return x, y, s


# 测试 LP：min -x1 - 2x2  s.t.  x1+x2<=4, x1<=3, x2<=3, x1,x2>=0
# 标准化：引入松弛变量 x3, x4, x5
# min  c^T x  s.t. Ax = b, x >= 0
c_lp = np.array([-1., -2., 0., 0., 0.])
A_lp = np.array([
    [1., 1., 1., 0., 0.],
    [1., 0., 0., 1., 0.],
    [0., 1., 0., 0., 1.],
])
b_lp = np.array([4., 3., 3.])

print("\n" + "=" * 60)
print("原始-对偶内点法求解线性规划")
print("min  -x1 - 2x2")
print("s.t. x1+x2 <= 4,  x1 <= 3,  x2 <= 3,  x1,x2 >= 0")
print("=" * 60)

x_lp, y_lp, s_lp = solve_lp_primal_dual(c_lp, A_lp, b_lp, eps=1e-10)

print(f"\nLP 最优解: x1={x_lp[0]:.6f}, x2={x_lp[1]:.6f}")
print(f"LP 最优值: {c_lp @ x_lp:.6f}")
print(f"理论最优: x1=1, x2=3, f*=-7")
print(f"对偶变量 y: {y_lp[:3]}")


# ============================================================
# 示例 3：PyTorch 约束深度学习 - 公平性约束训练
# ============================================================

class FairClassifier(nn.Module):
    """
    带公平性约束的二分类器。
    约束：两个群体的假阳率之差不超过 ε_fair。
    使用增广 Lagrange 法（ALM）近似内点法思想处理约束。
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AugmentedLagrangianOptimizer:
    """
    增广 Lagrange 法（Augmented Lagrangian Method / ALM）。

    处理约束 h(θ) ≤ 0：
        L_ρ(θ, λ) = f(θ) + λ * h(θ) + (ρ/2) * max(h(θ), -λ/ρ)^2

    ALM 外层：更新 λ 和 ρ
    ALM 内层：固定 λ, ρ，对 θ 做无约束优化
    """
    def __init__(self, model, inner_optimizer, rho_init=1.0, rho_factor=2.0,
                 lambda_init=0.0):
        self.model = model
        self.inner_opt = inner_optimizer
        self.rho = rho_init
        self.rho_factor = rho_factor
        self.lam = lambda_init  # 对偶变量（Lagrange 乘子）

    def augmented_lagrangian_loss(self, base_loss, constraint_val):
        """
        计算增广 Lagrange 目标：f(θ) + λ h(θ) + (ρ/2) [max(h + λ/ρ, 0)]^2 - λ^2/(2ρ)

        参数:
            base_loss:       原始目标函数值（标量 tensor）
            constraint_val:  约束函数值 h(θ)（标量 tensor，<=0 为可行）
        """
        # 增广项：max(h + λ/ρ, 0)^2
        augmented_term = torch.clamp(
            constraint_val + self.lam / self.rho, min=0.0
        ) ** 2

        total_loss = base_loss + self.rho / 2.0 * augmented_term
        return total_loss

    def update_multipliers(self, constraint_val):
        """外层乘子更新（类比内点法的外层 t 更新）"""
        # λ ← max(0, λ + ρ * h(θ))（对不等式约束）
        self.lam = max(0.0, self.lam + self.rho * constraint_val)
        # 增大惩罚系数
        self.rho = self.rho * self.rho_factor


def train_fair_classifier(n_samples=1000, n_features=5, eps_fair=0.05,
                           n_epochs=50, n_inner=10):
    """
    演示带公平性约束的训练流程。
    公平性约束：两组的假阳率（FPR）之差 <= eps_fair
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # 生成模拟数据（两个群体）
    # 群体 A：特征均值 [0,...,0]
    # 群体 B：特征均值 [0.5,...,0.5]（有系统性差异）
    n_a = n_samples // 2
    n_b = n_samples - n_a

    X_a = torch.randn(n_a, n_features)
    X_b = torch.randn(n_b, n_features) + 0.5

    # 真实标签（两组相同的决策边界，确保本质公平）
    y_a = (X_a[:, 0] + X_a[:, 1] > 0).float()
    y_b = (X_b[:, 0] + X_b[:, 1] > 0).float()

    X = torch.cat([X_a, X_b], dim=0)
    y = torch.cat([y_a, y_b], dim=0)
    group = torch.cat([torch.zeros(n_a), torch.ones(n_b)])  # 群体标签

    model = FairClassifier(n_features, hidden_dim=32)
    inner_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    alm = AugmentedLagrangianOptimizer(model, inner_opt,
                                        rho_init=0.5, rho_factor=1.5)

    bce_loss = nn.BCELoss()

    print(f"\n{'Epoch':>6} {'分类损失':>10} {'约束违反':>12} {'λ':>10} {'ρ':>10}")
    print("-" * 55)

    for epoch in range(n_epochs):
        # ---- 内层优化：固定 λ, ρ，对 θ 做多步梯度下降 ----
        for inner_step in range(n_inner):
            inner_opt.zero_grad()

            pred = model(X)
            base_loss = bce_loss(pred, y)

            # 计算公平性约束违反量：|FPR_A - FPR_B| - eps_fair <= 0
            # FPR = E[ŷ | y=0]（对负样本预测为正的比例）
            mask_a_neg = (group == 0) & (y == 0)
            mask_b_neg = (group == 1) & (y == 0)

            if mask_a_neg.sum() > 0 and mask_b_neg.sum() > 0:
                fpr_a = pred[mask_a_neg].mean()
                fpr_b = pred[mask_b_neg].mean()
                constraint_val = torch.abs(fpr_a - fpr_b) - eps_fair
            else:
                constraint_val = torch.tensor(0.0)

            total_loss = alm.augmented_lagrangian_loss(base_loss, constraint_val)
            total_loss.backward()
            inner_opt.step()

        # ---- 外层更新：更新 λ 和 ρ ----
        with torch.no_grad():
            pred_eval = model(X)
            mask_a_neg = (group == 0) & (y == 0)
            mask_b_neg = (group == 1) & (y == 0)
            if mask_a_neg.sum() > 0 and mask_b_neg.sum() > 0:
                fpr_a_val = pred_eval[mask_a_neg].mean().item()
                fpr_b_val = pred_eval[mask_b_neg].mean().item()
                cv = abs(fpr_a_val - fpr_b_val) - eps_fair
            else:
                cv = 0.0

        alm.update_multipliers(cv)

        if epoch % 10 == 0:
            with torch.no_grad():
                base_l = bce_loss(model(X), y).item()
            print(f"{epoch:>6} {base_l:>10.4f} {cv:>12.4f} {alm.lam:>10.4f} {alm.rho:>10.4f}")

    # 最终评估
    with torch.no_grad():
        pred_final = model(X)
        acc = ((pred_final > 0.5).float() == y).float().mean().item()
        mask_a_neg = (group == 0) & (y == 0)
        mask_b_neg = (group == 1) & (y == 0)
        if mask_a_neg.sum() > 0 and mask_b_neg.sum() > 0:
            fpr_a_final = pred_final[mask_a_neg].mean().item()
            fpr_b_final = pred_final[mask_b_neg].mean().item()
            fpr_diff = abs(fpr_a_final - fpr_b_final)
        else:
            fpr_diff = 0.0

    print(f"\n训练完成！准确率: {acc:.4f}，FPR 差: {fpr_diff:.4f}（约束: <= {eps_fair}）")
    return model


print("\n" + "=" * 60)
print("带公平性约束的分类器训练（增广 Lagrange 法）")
print("=" * 60)
model_fair = train_fair_classifier()


# ============================================================
# 示例 4：CVXPY 调用内点法求解器 - 实际应用范式
# ============================================================

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

if CVXPY_AVAILABLE:
    print("\n" + "=" * 60)
    print("CVXPY + MOSEK/SCS 求解 Lasso 问题（调用内点法求解器）")
    print("=" * 60)

    np.random.seed(0)
    n_feat, n_samp = 50, 100
    A_data = np.random.randn(n_samp, n_feat)
    x_true = np.random.randn(n_feat)
    x_true[30:] = 0   # 稀疏真实解
    b_data = A_data @ x_true + 0.1 * np.random.randn(n_samp)

    # Lasso: min ||Ax-b||_2^2  s.t.  ||x||_1 <= t
    t_lasso = 5.0
    x_var = cp.Variable(n_feat)
    objective = cp.Minimize(cp.sum_squares(A_data @ x_var - b_data))
    constraints = [cp.norm1(x_var) <= t_lasso]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)  # SCS 是内点法求解器

    print(f"求解状态: {prob.status}")
    print(f"最优值 ||Ax-b||^2 = {prob.value:.4f}")
    print(f"解的稀疏度: {np.sum(np.abs(x_var.value) < 1e-3)} / {n_feat} 个分量接近零")
    print(f"对偶变量（约束乘子）: λ = {constraints[0].dual_value:.6f}")
else:
    print("\n注意：请安装 cvxpy（pip install cvxpy）以运行 CVXPY 示例。")
    print("CVXPY 是调用 MOSEK/SCS 等内点法求解器的 Python 建模语言。")
```

### 应用二：理解自适应梯度法与内点法的联系

自适应优化器（如 Adam、Adagrad）可以从内点法视角理解：

- **Adagrad 的坐标适应**：对每个坐标使用不同步长，类比于内点法中障碍 Hessian 对不同方向的自适应缩放
- **预条件**（Preconditioning）：对数障碍函数的 Hessian $\nabla^2\phi(\mathbf{x}) = \sum_i \frac{\mathbf{a}_i\mathbf{a}_i^\top}{(b_i - \mathbf{a}_i^\top\mathbf{x})^2}$ 自动在约束边界处增大曲率，提供了内置的预条件效果

**物理信息神经网络（PINN）中的应用**：PINN 将偏微分方程约束作为软约束（罚函数）加入损失，这正是障碍函数法的精神体现。随着训练进行逐渐增大约束权重，对应参数 $t$ 的增大。

---

## 练习题

**练习 11.1**（基础）设约束凸优化问题为：

$$\min_{x_1, x_2} \; x_1^2 + x_2^2, \quad \text{s.t.} \quad x_1 + x_2 \geq 1$$

（a）将约束写成标准形式 $g(\mathbf{x}) \leq 0$，写出对数障碍函数 $\phi(\mathbf{x})$。

（b）写出参数化问题 $\min_{x_1+x_2>1} \; t(x_1^2 + x_2^2) - \log(x_1 + x_2 - 1)$ 的一阶最优性条件。

（c）求解中心路径 $\mathbf{x}^*(t)$ 的解析表达式，验证当 $t \to \infty$ 时 $\mathbf{x}^*(t) \to (\frac{1}{2}, \frac{1}{2})$。

（d）计算在 $\mathbf{x}^*(t)$ 处的对偶间隙，验证等于 $\frac{m}{t} = \frac{1}{t}$。

---

**练习 11.2**（基础）对数障碍函数的性质。

（a）设 $g(\mathbf{x}) = \mathbf{a}^\top \mathbf{x} - b$（线性不等式），证明 $-\log(-g(\mathbf{x}))$ 是凸函数。

（b）设 $g(\mathbf{x}) = \mathbf{x}^\top Q \mathbf{x} - 1$（二次不等式，$Q \succ 0$，即单位椭球内部），在严格可行域（$\mathbf{x}^\top Q \mathbf{x} < 1$）内证明 $-\log(1 - \mathbf{x}^\top Q \mathbf{x})$ 是凸函数。

（c）比较：对于约束 $-1 \leq x \leq 1$（等价两个约束），写出对数障碍函数，计算其梯度和 Hessian，说明 Hessian 在 $x = 0$ 与 $x \to 1^-$ 处的变化趋势。

---

**练习 11.3**（中级）中心路径与对偶间隙。

考虑线性规划：

$$\min_{\mathbf{x}} \; \mathbf{c}^\top \mathbf{x}, \quad \text{s.t.} \; A\mathbf{x} = \mathbf{b}, \; \mathbf{x} \geq \mathbf{0}$$

等价写成 $g_i(\mathbf{x}) = -x_i \leq 0$。

（a）写出对应的对数障碍函数 $\phi(\mathbf{x}) = -\sum_{i=1}^n \log(x_i)$，以及参数化问题的 KKT 条件。

（b）证明中心路径上的点满足 $x_i s_i = \frac{1}{t}$，其中 $s_i = c_i - (A^\top \boldsymbol{\nu})_i$ 是对偶松弛变量。

（c）由此推导对偶间隙 $\mathbf{c}^\top \mathbf{x}^*(t) - \mathbf{b}^\top \boldsymbol{\nu}^*(t) = \frac{n}{t}$（LP 的互补松弛条件）。

（d）若要达到 $\epsilon = 10^{-8}$ 精度，且 $t^0 = 1$，增长因子 $\mu = 10$，需要多少次外层迭代？（$n$ 个变量，假设每次内层牛顿法恰好一步收敛）

---

**练习 11.4**（中级）原始-对偶内点法的牛顿步。

考虑如下简单的 LP（标准形式）：

$$\min \; -x_1 - 2x_2, \quad \text{s.t.} \; x_1 + x_2 = 3, \; x_1, x_2 \geq 0$$

（a）写出 KKT 条件（原始-对偶形式），确定变量为 $(x_1, x_2, \lambda_1, \lambda_2, \nu)$，其中 $\lambda_i$ 对应 $x_i \geq 0$ 的对偶变量。

（b）从初始点 $\mathbf{x}^{(0)} = (1, 2)^\top$，$\boldsymbol{\lambda}^{(0)} = (1, 1)^\top$，$\nu^{(0)} = 0$ 出发，取 $\tau = 0.5$，写出扰动 KKT 方程组的具体数值。

（c）求解牛顿方程，得到 $(\Delta\mathbf{x}, \Delta\boldsymbol{\lambda}, \Delta\nu)$。

（d）计算最大可行步长 $\alpha_{\max}$（保持 $x_i > 0$，$\lambda_i > 0$），取 $\alpha = 0.9 \alpha_{\max}$ 更新，计算更新后的代理对偶间隙 $\hat{\eta}$。

---

**练习 11.5**（提高）障碍函数法的收敛性分析与实现。

（a）**理论**：设增长因子为 $\mu > 1$，当前参数 $t_k$，对应的中心路径点为 $\mathbf{x}^*(t_k)$。当参数更新为 $t_{k+1} = \mu t_k$ 时，以 $\mathbf{x}^*(t_k)$ 为初始点，新问题的牛顿减少量满足：

$$\lambda(\mathbf{x}^*(t_k))^2 \leq \frac{(\mu - 1)^2 m^2}{\text{（某个与问题相关的量）}}$$

直觉上说明为什么 $\mu$ 不能太大（增长太快，内层难以收敛）。

（b）**实现**：用 Python/NumPy 实现障碍函数法，求解以下投资组合优化问题：

$$\min_{\mathbf{w}} \; -\boldsymbol{\mu}^\top \mathbf{w} + \gamma \mathbf{w}^\top \Sigma \mathbf{w}$$
$$\text{s.t.} \; \mathbf{1}^\top \mathbf{w} = 1, \; w_i \geq 0.05, \; w_i \leq 0.4, \; i = 1, \ldots, n$$

其中 $n = 5$ 个资产，$\boldsymbol{\mu}$ 为预期收益率，$\Sigma$ 为协方差矩阵，$\gamma = 2$ 为风险厌恶系数。记录每次外层迭代后的目标值和对偶间隙。

（c）**分析**：对比 $\mu = 2$、$\mu = 10$、$\mu = 100$ 三种增长因子下的总牛顿步数，验证 $\mu \approx 10$ 的经验选择。

---

## 练习答案

**答案 11.1**

（a）约束 $x_1 + x_2 \geq 1$ 等价于 $g(\mathbf{x}) = 1 - x_1 - x_2 \leq 0$。对数障碍函数：

$$\phi(\mathbf{x}) = -\log(-(1 - x_1 - x_2)) = -\log(x_1 + x_2 - 1)$$

定义域：$\{(x_1, x_2) \mid x_1 + x_2 > 1\}$。

（b）参数化问题的 Lagrange 函数对 $x_1$, $x_2$ 的偏导数为零：

$$\frac{\partial}{\partial x_1}\left[t(x_1^2 + x_2^2) - \log(x_1 + x_2 - 1)\right] = 2tx_1 - \frac{1}{x_1 + x_2 - 1} = 0$$

$$\frac{\partial}{\partial x_2}\left[t(x_1^2 + x_2^2) - \log(x_1 + x_2 - 1)\right] = 2tx_2 - \frac{1}{x_1 + x_2 - 1} = 0$$

（c）由对称性，两式相减得 $2t(x_1 - x_2) = 0$，即 $x_1 = x_2$。代入第一式：$2tx_1 = \frac{1}{2x_1 - 1}$，解出：

$$4tx_1^2 - 2tx_1 - 1 = 0 \implies x_1 = \frac{2t + \sqrt{4t^2 + 16t}}{8t} = \frac{1 + \sqrt{1 + 4/t}}{4}$$

（取使 $x_1 + x_2 > 1$ 的正根）

当 $t \to \infty$ 时，$\sqrt{1 + 4/t} \to 1$，得 $x_1 \to \frac{1}{2}$，$x_2 \to \frac{1}{2}$，即 $\mathbf{x}^*(t) \to (\frac{1}{2}, \frac{1}{2})$。$\checkmark$

（d）隐含对偶变量：$\lambda^*(t) = \frac{-1}{t \cdot g(\mathbf{x}^*(t))} = \frac{1}{t(x_1^*(t) + x_2^*(t) - 1)}$

对偶间隙：$f(\mathbf{x}^*(t)) - f^* = \lambda^*(t) \cdot (-g(\mathbf{x}^*(t))) = \lambda^*(t)(x_1^* + x_2^* - 1) = \frac{1}{t} = \frac{m}{t}$ $\checkmark$

---

**答案 11.2**

（a）$\phi(x) = -\log(b - \mathbf{a}^\top\mathbf{x})$。计算二阶导数：

$$\frac{\partial \phi}{\partial x_i} = \frac{a_i}{b - \mathbf{a}^\top\mathbf{x}}, \quad \nabla^2 \phi = \frac{\mathbf{a}\mathbf{a}^\top}{(b - \mathbf{a}^\top\mathbf{x})^2} \succeq 0$$

Hessian 半正定（秩1矩阵），故 $\phi$ 是凸函数。$\square$

（b）令 $u = \mathbf{x}^\top Q \mathbf{x}$，$\phi = -\log(1 - u)$。在 $u \in [0, 1)$ 上：

$$\frac{d^2}{du^2}(-\log(1 - u)) = \frac{1}{(1-u)^2} > 0$$

利用链式法则，$\nabla^2 \phi(\mathbf{x}) = \frac{2Q}{1 - \mathbf{x}^\top Q\mathbf{x}} + \frac{4(Q\mathbf{x})(Q\mathbf{x})^\top}{(1 - \mathbf{x}^\top Q\mathbf{x})^2} \succ 0$（$Q \succ 0$），故 $\phi$ 是严格凸函数。$\square$

（c）约束 $-1 \leq x \leq 1$ 等价于 $g_1(x) = x - 1 \leq 0$，$g_2(x) = -x - 1 \leq 0$，障碍函数：

$$\phi(x) = -\log(1 - x) - \log(1 + x)$$

梯度：$\phi'(x) = \frac{1}{1-x} - \frac{1}{1+x} = \frac{2x}{1 - x^2}$

Hessian（即二阶导数）：$\phi''(x) = \frac{1}{(1-x)^2} + \frac{1}{(1+x)^2} = \frac{2(1 + x^2)}{(1-x^2)^2}$

在 $x = 0$：$\phi''(0) = 2$（有限值，曲率适中）

当 $x \to 1^-$：$\phi''(x) \to +\infty$（曲率趋于无穷，对算法施加"无限阻力"防止越界）

---

**答案 11.3**

（a）对数障碍函数 $\phi(\mathbf{x}) = -\sum_{i=1}^n \log(x_i)$（$g_i(\mathbf{x}) = -x_i$）。

参数化问题 $\min t\mathbf{c}^\top\mathbf{x} - \sum_i \log x_i$，$A\mathbf{x} = \mathbf{b}$ 的 KKT 条件：

$$t c_i - \frac{1}{x_i} + (A^\top\boldsymbol{\nu})_i = 0, \quad \forall i = 1, \ldots, n$$

等价于 $\frac{1}{tx_i} = c_i - (A^\top\boldsymbol{\nu}/t)_i$。

（b）定义对偶松弛变量 $s_i = c_i - (A^\top\boldsymbol{\nu})_i$，则 KKT 条件变为 $\frac{1}{t} = x_i s_i$，即：

$$x_i^*(t) \cdot s_i^*(t) = \frac{1}{t}$$

这正是 LP 中心路径上的互补松弛（扰动）条件。$\square$

（c）对偶间隙：

$$\mathbf{c}^\top\mathbf{x}^*(t) - \mathbf{b}^\top\boldsymbol{\nu}^*(t) = \mathbf{s}^{*\top}(t)\mathbf{x}^*(t) + (A^\top\boldsymbol{\nu}^*(t))^\top\mathbf{x}^*(t) - \mathbf{b}^\top\boldsymbol{\nu}^*(t)$$

$$= \mathbf{s}^{*\top}(t)\mathbf{x}^*(t) + \boldsymbol{\nu}^{*\top}(t)(A\mathbf{x}^*(t) - \mathbf{b}) = \mathbf{s}^{*\top}(t)\mathbf{x}^*(t) = \sum_{i=1}^n x_i^*(t)s_i^*(t) = \frac{n}{t}$$

$\square$

（d）需要 $\frac{n}{t_k} \leq \epsilon$，即 $t_k \geq \frac{n}{\epsilon}$。

经过 $k$ 次外层迭代后 $t_k = \mu^k t_0 = 10^k$，需要 $10^k \geq \frac{n}{10^{-8}} = n \times 10^8$。

取 $n = 100$，则 $10^k \geq 10^{10}$，即 $k \geq 10$ 次外层迭代。

总牛顿步数 $\approx k \times 1 = 10$ 步（假设每次内层恰好一步），体现了内点法的高效性。

---

**答案 11.4**

（a）KKT 条件（扰动版，参数 $\tau$）：

$$\begin{cases}
-1 - \lambda_1 + \nu = 0 & \text{（}\partial/\partial x_1\text{，对偶残差）}\\
-2 - \lambda_2 + \nu = 0 & \text{（}\partial/\partial x_2\text{，对偶残差）}\\
-\lambda_1 \cdot (-x_1) - \tau = 0 \implies \lambda_1 x_1 = \tau & \text{（中心性残差）}\\
-\lambda_2 \cdot (-x_2) - \tau = 0 \implies \lambda_2 x_2 = \tau & \text{（中心性残差）}\\
x_1 + x_2 = 3 & \text{（原始可行性）}
\end{cases}$$

（b）代入初始点 $\mathbf{x}^{(0)} = (1, 2)^\top$，$\boldsymbol{\lambda}^{(0)} = (1, 1)^\top$，$\nu^{(0)} = 0$，$\tau = 0.5$：

$$r_{\text{dual},1} = -1 - 1 + 0 = -2, \quad r_{\text{dual},2} = -2 - 1 + 0 = -3$$
$$r_{\text{cent},1} = \lambda_1 x_1 - \tau = 1 \cdot 1 - 0.5 = 0.5, \quad r_{\text{cent},2} = 1 \cdot 2 - 0.5 = 1.5$$
$$r_{\text{prim}} = 1 + 2 - 3 = 0$$

（c）牛顿方程（LP 特殊情形，$\nabla^2 f = 0$）：

$$H_{\text{pd}} = \text{diag}\!\left(\frac{\lambda_i}{x_i}\right) = \text{diag}(1, 0.5)$$

等式约束矩阵 $A = [1, 1]$，$\mathbf{b} = 3$。

KKT 系统（缩减形式）：

$$\begin{pmatrix} 1 & 0 & 1 \\ 0 & 0.5 & 1 \\ 1 & 1 & 0 \end{pmatrix} \begin{pmatrix} \Delta x_1 \\ \Delta x_2 \\ \Delta\nu \end{pmatrix} = -\begin{pmatrix} r_{\text{dual},1}^{\text{mod}} \\ r_{\text{dual},2}^{\text{mod}} \\ r_{\text{prim}} \end{pmatrix}$$

其中修正对偶残差（代入 $\Delta\lambda$ 后）为：

$$\tilde{r}_{\text{dual},i} = r_{\text{dual},i} - \frac{r_{\text{cent},i}}{x_i} = -2 - \frac{0.5}{1} = -2.5, \quad -3 - \frac{1.5}{2} = -3.75$$

求解：$\Delta x_1 + \Delta\nu = 2.5$，$0.5\Delta x_2 + \Delta\nu = 3.75$，$\Delta x_1 + \Delta x_2 = 0$

由第三式 $\Delta x_2 = -\Delta x_1$，代入：$0.5(-\Delta x_1) + \Delta\nu = 3.75$，即 $-0.5\Delta x_1 + \Delta\nu = 3.75$。

与第一式相减：$1.5\Delta x_1 = -1.25$，得 $\Delta x_1 = -\frac{5}{6}$，$\Delta x_2 = \frac{5}{6}$，$\Delta\nu = 2.5 + \frac{5}{6} \approx 3.33$。

$\Delta\lambda_i = -\lambda_i + \frac{\tau - \lambda_i \Delta x_i}{x_i}$（由中心性方程回代）：

$$\Delta\lambda_1 = -1 + \frac{0.5 - 1 \cdot (-5/6)}{1} = -1 + 0.5 + 5/6 = 4/6 + 3/6 - 6/6 = 1/6 \approx 0.167$$

$$\Delta\lambda_2 = -1 + \frac{0.5 - 1 \cdot (5/6)}{2} = -1 + \frac{0.5 - 5/6}{2} = -1 + \frac{-1/3}{2} = -1 - 1/6 \approx -1.167$$

（d）最大可行步长：$\alpha_{\max}$ 需满足 $\lambda_i + \alpha\Delta\lambda_i > 0$，$x_i + \alpha\Delta x_i > 0$。

约束 $\lambda_2 + \alpha\Delta\lambda_2 > 0$：$1 + \alpha(-7/6) > 0 \implies \alpha < \frac{6}{7} \approx 0.857$

约束 $x_1 + \alpha\Delta x_1 > 0$：$1 + \alpha(-5/6) > 0 \implies \alpha < \frac{6}{5} = 1.2$（不紧）

故 $\alpha_{\max} = \frac{6}{7}$，取 $\alpha = 0.9 \times \frac{6}{7} \approx 0.771$。

更新后：$x_1 \approx 1 - 0.771 \times \frac{5}{6} \approx 0.358$，$x_2 \approx 2 + 0.771 \times \frac{5}{6} \approx 2.643$

代理对偶间隙：$\hat{\eta} = x_1 \lambda_1 + x_2 \lambda_2 \approx 0.358(1 + 0.771 \times \frac{1}{6}) + 2.643(1 - 0.771 \times \frac{7}{6}) \approx$ （一步后接近 $\frac{n}{\text{更大的}t}$，对偶间隙大幅缩小）

---

**答案 11.5**

（a）直觉分析：当 $\mu$ 很大时，$t_{k+1} = \mu t_k$ 远大于 $t_k$。以 $\mathbf{x}^*(t_k)$ 为初始点时，新问题中障碍项权重急剧缩小（相对目标项），当前点远离新中心路径点 $\mathbf{x}^*(t_{k+1})$，牛顿减少量 $\lambda(\mathbf{x}^*(t_k))^2$ 较大，需要更多牛顿步才能收敛到新的中心路径点。

理论上可以证明：若 $\mu \leq 1 + \frac{c}{\sqrt{m}}$（$c$ 为某常数），则每次内层恰好一次牛顿步即可（"短步"内点法）。这给出了外层迭代次数 $O(\sqrt{m}\log(1/\epsilon))$ 的理论界。

$\mu = 10 \sim 20$ 是实践中的折中：外层步数不多，同时每次内层收敛也足够快。

（b）投资组合优化实现（核心逻辑）：

$$\min_{\mathbf{w}} \; -\boldsymbol{\mu}^\top\mathbf{w} + \gamma\mathbf{w}^\top\Sigma\mathbf{w} + \frac{1}{t}\sum_{i=1}^n \left[\log(w_i - 0.05) + \log(0.4 - w_i)\right]^{-1}$$

$$\text{s.t.} \; \mathbf{1}^\top\mathbf{w} = 1$$

其中 $m = 2n$（每个资产贡献上下界两个约束），障碍函数为 $\phi(\mathbf{w}) = -\sum_i [\log(w_i - 0.05) + \log(0.4 - w_i)]$。

内层牛顿法需求解含单个等式约束的 KKT 系统（$n+1$ 维），Hessian 为 $2\gamma\Sigma + \nabla^2\phi(\mathbf{w})$（正定）。

（c）实验结果（理论预期）：

| 增长因子 $\mu$ | 外层步数 | 每次内层步数（平均） | 总牛顿步数（估计） |
|--------------|---------|-------------------|-----------------|
| $\mu = 2$ | $\approx 50$ | $2 \sim 3$ | $\approx 120$ |
| $\mu = 10$ | $\approx 15$ | $5 \sim 8$ | $\approx 90$（最优） |
| $\mu = 100$ | $\approx 6$ | $20 \sim 50$ | $\approx 180$ |

$\mu \approx 10$ 在两者之间取得最佳平衡，验证了经验选择的合理性。

---

*本章介绍了内点法的核心理论：从对数障碍函数到中心路径，再到原始-对偶牛顿更新。内点法是现代大规模凸优化的基石，也是约束机器学习问题的强力工具。下一章将讨论分解算法与交替方向乘子法（ADMM），进一步处理大规模分布式约束优化问题。*
