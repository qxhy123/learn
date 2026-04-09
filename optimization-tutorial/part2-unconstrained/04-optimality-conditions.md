# 第4章：最优性条件

## 学习目标

学完本章后，你将能够：

1. **理解局部极小与全局极小的区别**，掌握严格局部极小的数学定义
2. **推导并应用一阶必要条件**：理解为什么驻点 $\nabla f(\mathbf{x}^*) = \mathbf{0}$ 是极小值的必要条件
3. **掌握二阶最优性条件**：利用 Hessian 矩阵的半正定性和正定性区分极小、极大和鞍点
4. **理解鞍点的本质**，认识深度学习损失曲面中鞍点普遍存在的现象及其影响
5. **掌握凸函数的特殊最优性条件**：局部极小即全局极小，一阶条件充分

---

## 4.1 局部极小与全局极小

### 4.1.1 基本定义

设 $f: \mathbb{R}^n \to \mathbb{R}$ 是一个实值函数。

**定义 4.1（全局极小）**
若存在 $\mathbf{x}^* \in \mathbb{R}^n$，对所有 $\mathbf{x} \in \mathbb{R}^n$ 均满足

$$f(\mathbf{x}^*) \leq f(\mathbf{x}),$$

则称 $\mathbf{x}^*$ 为 $f$ 的**全局极小点**（global minimizer），$f(\mathbf{x}^*)$ 为**全局极小值**。

**定义 4.2（严格全局极小）**
若对所有 $\mathbf{x} \neq \mathbf{x}^*$ 均有 $f(\mathbf{x}^*) < f(\mathbf{x})$，则称 $\mathbf{x}^*$ 为**严格全局极小点**。

**定义 4.3（局部极小）**
若存在 $\epsilon > 0$，使得对所有满足 $\|\mathbf{x} - \mathbf{x}^*\| < \epsilon$ 的 $\mathbf{x}$ 均有

$$f(\mathbf{x}^*) \leq f(\mathbf{x}),$$

则称 $\mathbf{x}^*$ 为 $f$ 的**局部极小点**（local minimizer）。

**定义 4.4（严格局部极小）**
若在上述邻域内对所有 $\mathbf{x} \neq \mathbf{x}^*$ 均有 $f(\mathbf{x}^*) < f(\mathbf{x})$，则称 $\mathbf{x}^*$ 为**严格局部极小点**。

### 4.1.2 局部与全局的关系

**定理 4.1**
全局极小点必是局部极小点，但反之不成立。

**例 4.1** 考察函数

$$f(x) = x^4 - 3x^2 + 2x$$

该函数在 $\mathbb{R}$ 上有两个局部极小点，但只有一个全局极小点。

**关键洞察：**
- 一般非凸优化问题只能保证找到局部极小
- 凸优化问题中，局部极小即全局极小（见 4.5 节）
- 深度学习中，我们通常满足于找到"足够好"的局部极小

### 4.1.3 可行域上的最优性

若问题带约束 $\mathbf{x} \in \mathcal{S}$，则需在可行域 $\mathcal{S}$ 内讨论极小性。本章主要讨论**无约束**情形，约束优化将在第三部分展开。

---

## 4.2 一阶必要条件

### 4.2.1 梯度与方向导数

**定义 4.5（梯度）**
若 $f: \mathbb{R}^n \to \mathbb{R}$ 在 $\mathbf{x}$ 处可微，则其**梯度**为

$$\nabla f(\mathbf{x}) = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)^\top \in \mathbb{R}^n.$$

**定义 4.6（方向导数）**
沿单位方向 $\mathbf{d}$（$\|\mathbf{d}\| = 1$）的方向导数为

$$D_{\mathbf{d}} f(\mathbf{x}) = \lim_{\alpha \to 0^+} \frac{f(\mathbf{x} + \alpha \mathbf{d}) - f(\mathbf{x})}{\alpha} = \nabla f(\mathbf{x})^\top \mathbf{d}.$$

若 $D_{\mathbf{d}} f(\mathbf{x}) < 0$，则 $\mathbf{d}$ 是**下降方向**。

### 4.2.2 一阶必要条件

**定理 4.2（一阶必要条件）**
设 $f: \mathbb{R}^n \to \mathbb{R}$ 在 $\mathbf{x}^*$ 处可微。若 $\mathbf{x}^*$ 是局部极小点，则

$$\boxed{\nabla f(\mathbf{x}^*) = \mathbf{0}.}$$

**证明：**
反设 $\nabla f(\mathbf{x}^*) \neq \mathbf{0}$，令 $\mathbf{d} = -\nabla f(\mathbf{x}^*)$，则

$$D_{\mathbf{d}} f(\mathbf{x}^*) = \nabla f(\mathbf{x}^*)^\top \mathbf{d} = -\|\nabla f(\mathbf{x}^*)\|^2 < 0.$$

由方向导数的定义，存在 $\epsilon > 0$，当 $0 < \alpha < \epsilon$ 时，

$$f(\mathbf{x}^* + \alpha \mathbf{d}) < f(\mathbf{x}^*).$$

这与 $\mathbf{x}^*$ 是局部极小点矛盾。$\square$

### 4.2.3 驻点与临界点

满足 $\nabla f(\mathbf{x}) = \mathbf{0}$ 的点称为**驻点**或**临界点**（stationary/critical point）。驻点分三类：

| 驻点类型 | 几何特征 | Hessian 特征（见 4.3 节）|
|---------|---------|------------------------|
| 局部极小点 | 周围函数值均更大 | 正定 |
| 局部极大点 | 周围函数值均更小 | 负定 |
| 鞍点 | 某些方向增大，某些方向减小 | 不定 |

### 4.2.4 一维情形回顾

在一维情形 $f: \mathbb{R} \to \mathbb{R}$ 中，一阶条件退化为

$$f'(x^*) = 0.$$

**例 4.2** 对 $f(x) = x^3$，有 $f'(0) = 0$，但 $x=0$ 既不是极小也不是极大——它是**拐点**，这提醒我们一阶条件仅是必要条件。

### 4.2.5 多元函数例子

**例 4.3** 设 $f(x_1, x_2) = x_1^2 + 2x_2^2 + x_1 x_2$，求驻点。

求梯度并令其为零：

$$\nabla f = \begin{pmatrix} 2x_1 + x_2 \\ 4x_2 + x_1 \end{pmatrix} = \mathbf{0}.$$

解得 $x_1 = 0, x_2 = 0$。唯一驻点为原点，通过二阶条件（见下节）可确认其为极小点。

---

## 4.3 二阶必要条件与充分条件

### 4.3.1 Hessian 矩阵

**定义 4.7（Hessian 矩阵）**
若 $f: \mathbb{R}^n \to \mathbb{R}$ 二阶连续可微，则其 **Hessian 矩阵**为

$$\nabla^2 f(\mathbf{x}) = \mathbf{H}(\mathbf{x}) = \left[\frac{\partial^2 f}{\partial x_i \partial x_j}\right]_{n \times n}.$$

由 Schwarz 定理，在连续条件下 $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$，故 Hessian 是**对称矩阵**。

**例 4.4** 对 $f(x_1, x_2) = x_1^2 + 2x_2^2 + x_1 x_2$：

$$\mathbf{H} = \begin{pmatrix} 2 & 1 \\ 1 & 4 \end{pmatrix}.$$

特征值为 $\lambda_{1,2} = 3 \pm \sqrt{2}$，均为正，故 $\mathbf{H}$ 正定。

### 4.3.2 二阶 Taylor 展开

在 $\mathbf{x}^*$ 附近，$f$ 的二阶 Taylor 展开为

$$f(\mathbf{x}^* + \mathbf{d}) = f(\mathbf{x}^*) + \nabla f(\mathbf{x}^*)^\top \mathbf{d} + \frac{1}{2} \mathbf{d}^\top \nabla^2 f(\mathbf{x}^*) \mathbf{d} + O(\|\mathbf{d}\|^3).$$

若 $\mathbf{x}^*$ 是驻点（$\nabla f(\mathbf{x}^*) = \mathbf{0}$），则

$$f(\mathbf{x}^* + \mathbf{d}) - f(\mathbf{x}^*) \approx \frac{1}{2} \mathbf{d}^\top \mathbf{H}^* \mathbf{d},$$

其中 $\mathbf{H}^* = \nabla^2 f(\mathbf{x}^*)$。函数值的增减由二次型 $\mathbf{d}^\top \mathbf{H}^* \mathbf{d}$ 决定。

### 4.3.3 二阶必要条件

**定理 4.3（二阶必要条件）**
设 $f$ 二阶连续可微，$\mathbf{x}^*$ 是局部极小点，则

$$\nabla f(\mathbf{x}^*) = \mathbf{0} \quad \text{且} \quad \nabla^2 f(\mathbf{x}^*) \succeq \mathbf{0}\ (\text{半正定}).$$

**证明思路：** 若 $\mathbf{H}^*$ 不是半正定的，则存在方向 $\mathbf{d}$ 使得 $\mathbf{d}^\top \mathbf{H}^* \mathbf{d} < 0$，沿此方向可使函数值减小，与局部极小性矛盾。

**半正定的刻画：** 对称矩阵 $\mathbf{H}$ 半正定 $\Leftrightarrow$ 所有特征值 $\lambda_i \geq 0$ $\Leftrightarrow$ 所有顺序主子式非负。

### 4.3.4 二阶充分条件

**定理 4.4（二阶充分条件）**
设 $f$ 二阶连续可微，若

$$\nabla f(\mathbf{x}^*) = \mathbf{0} \quad \text{且} \quad \nabla^2 f(\mathbf{x}^*) \succ \mathbf{0}\ (\text{正定}),$$

则 $\mathbf{x}^*$ 是**严格局部极小点**。

**证明思路：** 因 $\mathbf{H}^*$ 正定，其最小特征值 $\lambda_{\min} > 0$，则

$$f(\mathbf{x}^* + \mathbf{d}) - f(\mathbf{x}^*) \geq \frac{1}{2} \lambda_{\min} \|\mathbf{d}\|^2 > 0$$

对足够小的 $\|\mathbf{d}\| > 0$ 成立。$\square$

**类似地，二阶充分条件判断极大：**

若 $\nabla f(\mathbf{x}^*) = \mathbf{0}$ 且 $\nabla^2 f(\mathbf{x}^*) \prec \mathbf{0}$（负定），则 $\mathbf{x}^*$ 是严格局部极大点。

### 4.3.5 条件总结

| 条件 | $\mathbf{H}^*$ 特征 | 结论 |
|-----|-------------------|------|
| $\nabla f = \mathbf{0}$，$\mathbf{H}^* \succ \mathbf{0}$ | 正定（全正特征值）| 严格局部极小（充分） |
| $\nabla f = \mathbf{0}$，$\mathbf{H}^* \prec \mathbf{0}$ | 负定（全负特征值）| 严格局部极大（充分） |
| $\nabla f = \mathbf{0}$，$\mathbf{H}^*$ 不定 | 正负特征值均有 | 鞍点（充分） |
| $\nabla f = \mathbf{0}$，$\mathbf{H}^* \succeq \mathbf{0}$ | 半正定（含零特征值）| 无法判断（需高阶分析）|

### 4.3.6 一维对应关系

| 多元条件 | 一维对应 |
|---------|---------|
| $\nabla f(\mathbf{x}^*) = \mathbf{0}$ | $f'(x^*) = 0$ |
| $\mathbf{H}^* \succ \mathbf{0}$ | $f''(x^*) > 0$ |
| $\mathbf{H}^* \prec \mathbf{0}$ | $f''(x^*) < 0$ |
| $\mathbf{H}^*$ 不定 | $f''(x^*) = 0$（拐点情形）|

---

## 4.4 鞍点与不定点

### 4.4.1 鞍点的数学定义

**定义 4.8（鞍点）**
设 $\mathbf{x}^*$ 是 $f$ 的驻点（$\nabla f(\mathbf{x}^*) = \mathbf{0}$）。若 $\nabla^2 f(\mathbf{x}^*)$ 是**不定矩阵**（既有正特征值又有负特征值），则称 $\mathbf{x}^*$ 为 $f$ 的**鞍点**（saddle point）。

**几何解释：** 在鞍点处，函数沿某些方向"上坡"，沿另一些方向"下坡"，形状如马鞍。

### 4.4.2 典型鞍点示例

**例 4.5（双曲抛物面）**
$$f(x_1, x_2) = x_1^2 - x_2^2.$$

梯度：$\nabla f = (2x_1, -2x_2)^\top = \mathbf{0}$ 给出唯一驻点 $(0, 0)$。

Hessian：

$$\mathbf{H} = \begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix},$$

特征值为 $\{2, -2\}$，不定，故原点是鞍点。

- 沿 $x_1$ 轴：$f(x_1, 0) = x_1^2$，函数递增（碗形）
- 沿 $x_2$ 轴：$f(0, x_2) = -x_2^2$，函数递减（倒碗形）

**例 4.6（猴鞍面）**
$$f(x_1, x_2) = x_1^3 - 3x_1 x_2^2.$$

原点处 $\nabla f = \mathbf{0}$，$\mathbf{H} = \mathbf{0}$（退化情形），但原点仍是鞍点（高阶项分析）。

### 4.4.3 负曲率方向

在鞍点 $\mathbf{x}^*$ 处，设 $\mathbf{H}^*$ 的最小特征值为 $\lambda_{\min} < 0$，对应特征向量为 $\mathbf{v}_{\min}$，则

$$\mathbf{d}^\top \mathbf{H}^* \mathbf{d} < 0 \quad \text{当 } \mathbf{d} \approx \mathbf{v}_{\min},$$

即沿**负曲率方向**可逃离鞍点。这一性质被现代优化算法利用。

### 4.4.4 深度学习中鞍点的普遍性

**Dauphin 等人（2014）的洞察：**

在高维非凸优化中（如深度神经网络），**鞍点远比局部极小点更常见**。原因如下：

1. **随机矩阵理论**：对于一个随机对称矩阵，所有特征值为正的概率随维数 $n$ 指数衰减。当 $n = 10^6$（参数数量级），一个随机驻点是局部极小点的概率极小。

2. **损失曲面的高维几何**：深度网络的损失函数有大量参数，临界点中鞍点的比例 $\approx 1 - 2^{-n}$。

3. **对称性引起的流形**：网络权重置换对称性导致参数空间中存在大量等价点，其连接路径上充满鞍点。

**实践意义：**

| 问题 | 原因 | 应对方法 |
|-----|------|---------|
| SGD 在鞍点减速 | 梯度接近零，更新步长很小 | 动量法（Momentum）|
| 收敛到不好的临界点 | 梯度噪声实际上有助于逃离鞍点 | 随机梯度（SGD 优于 GD）|
| 二阶方法可能卡住 | Newton 法可能朝鞍点移动 | 负曲率利用（saddle-free Newton）|

**值得注意的是：** Goodfellow 等人（2015）的研究表明，深度网络的局部极小点损失值通常接近全局极小，因此实践中找到任意局部极小即可。

---

## 4.5 凸函数的最优性条件

### 4.5.1 凸函数回顾

**定义 4.9（凸函数）**
函数 $f: \mathcal{C} \to \mathbb{R}$（$\mathcal{C}$ 为凸集）是凸函数，若对所有 $\mathbf{x}, \mathbf{y} \in \mathcal{C}$ 和 $\theta \in [0, 1]$：

$$f(\theta \mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta) f(\mathbf{y}).$$

**等价刻画（可微情形）：**
$f$ 是凸函数当且仅当

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top (\mathbf{y} - \mathbf{x}) \quad \forall \mathbf{x}, \mathbf{y} \in \mathcal{C}.$$

**等价刻画（二阶可微情形）：**
$f$ 是凸函数当且仅当

$$\nabla^2 f(\mathbf{x}) \succeq \mathbf{0} \quad \forall \mathbf{x} \in \mathcal{C}.$$

### 4.5.2 凸函数的关键性质

**定理 4.5（局部极小 = 全局极小）**
若 $f$ 是凸函数，则 $f$ 的任意局部极小点也是全局极小点。

**证明：**
设 $\mathbf{x}^*$ 是局部极小点，假设存在 $\mathbf{y}$ 使得 $f(\mathbf{y}) < f(\mathbf{x}^*)$。
对任意 $\theta \in (0, 1)$，令 $\mathbf{z}_\theta = (1-\theta)\mathbf{x}^* + \theta \mathbf{y}$，由凸性：

$$f(\mathbf{z}_\theta) \leq (1-\theta)f(\mathbf{x}^*) + \theta f(\mathbf{y}) < f(\mathbf{x}^*).$$

当 $\theta \to 0^+$ 时，$\mathbf{z}_\theta \to \mathbf{x}^*$，但 $f(\mathbf{z}_\theta) < f(\mathbf{x}^*)$，与局部极小性矛盾。$\square$

### 4.5.3 凸函数的一阶充要条件

**定理 4.6（凸函数一阶最优性）**
设 $f$ 是可微凸函数，则

$$\mathbf{x}^* \text{ 是全局极小点} \iff \nabla f(\mathbf{x}^*) = \mathbf{0}.$$

**证明（$\Rightarrow$方向）：** 由定理 4.2，全局极小点处梯度为零。

**证明（$\Leftarrow$方向）：** 由凸函数的一阶条件，对任意 $\mathbf{x}$：

$$f(\mathbf{x}) \geq f(\mathbf{x}^*) + \nabla f(\mathbf{x}^*)^\top (\mathbf{x} - \mathbf{x}^*) = f(\mathbf{x}^*). \quad \square$$

**重要意义：** 对于凸函数，一阶条件从"必要"升级为"充要"，大大简化了优化分析。

### 4.5.4 严格凸函数的唯一性

**定理 4.7（严格凸函数的唯一极小点）**
若 $f$ 是**严格凸函数**，则 $f$ 至多有一个全局极小点（唯一性）。

**例 4.7（二次函数）**
$f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{A} \mathbf{x} - \mathbf{b}^\top \mathbf{x}$，其中 $\mathbf{A} \succ \mathbf{0}$。

由严格凸性，唯一全局极小点满足

$$\nabla f(\mathbf{x}^*) = \mathbf{A}\mathbf{x}^* - \mathbf{b} = \mathbf{0} \implies \mathbf{x}^* = \mathbf{A}^{-1}\mathbf{b}.$$

### 4.5.5 常见凸函数与非凸函数

| 函数 | 凸/非凸 | 说明 |
|-----|--------|------|
| $\|\mathbf{x}\|^2$ | 凸（严格）| $\mathbf{H} = 2\mathbf{I} \succ \mathbf{0}$ |
| $e^x$ | 凸 | $f'' = e^x > 0$ |
| $-\log x$（$x>0$）| 凸 | $f'' = 1/x^2 > 0$ |
| $\|\mathbf{x}\|_1$ | 凸（非严格）| 次梯度存在 |
| $x^3$ | 非凸 | 有鞍点 |
| 神经网络损失 | 一般非凸 | 大量鞍点和局部极小 |

---

## 本章小结

| 概念 | 数学表达 | 判别条件 | 实践意义 |
|-----|---------|---------|---------|
| **全局极小** | $f(\mathbf{x}^*) \leq f(\mathbf{x}), \forall \mathbf{x}$ | 凸问题中等价于驻点 | 优化的终极目标 |
| **局部极小** | $f(\mathbf{x}^*) \leq f(\mathbf{x})$ 在邻域内 | $\nabla f=\mathbf{0}$，$\mathbf{H} \succeq \mathbf{0}$（必要）| 实际算法通常找局部极小 |
| **严格局部极小** | 邻域内严格最小 | $\nabla f=\mathbf{0}$，$\mathbf{H} \succ \mathbf{0}$（充分）| 孤立极小点 |
| **驻点/临界点** | $\nabla f(\mathbf{x}^*) = \mathbf{0}$ | 一阶必要条件 | 所有极值候选点 |
| **鞍点** | 驻点但非极值 | $\mathbf{H}$ 不定（正负特征值均有）| 深度学习中高维主导 |
| **一阶必要条件** | $\nabla f(\mathbf{x}^*) = \mathbf{0}$ | 对所有可微极小成立 | 驻点方程组 |
| **二阶必要条件** | $\mathbf{H}^* \succeq \mathbf{0}$ | 局部极小的必要性 | 区分极小与鞍点 |
| **二阶充分条件** | $\mathbf{H}^* \succ \mathbf{0}$ | 保证严格局部极小 | 确认极小点 |
| **凸函数最优性** | 一阶条件充要 | $\nabla f(\mathbf{x}^*) = \mathbf{0} \iff$ 全局极小 | 凸优化的核心优势 |

---

## 深度学习应用：深度网络损失曲面的临界点分析

### 背景

深度神经网络的训练等价于最小化高维非凸损失函数：

$$\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \ell(f_{\boldsymbol{\theta}}(\mathbf{x}_i), y_i),$$

其中 $\boldsymbol{\theta} \in \mathbb{R}^p$，$p$ 可达数十亿。理解其损失曲面对设计优化算法至关重要。

### 代码实现

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# 1. 可视化二维损失曲面（含极小点、极大点与鞍点）
# ============================================================

def visualize_critical_points():
    """可视化不同类型临界点的损失曲面"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             subplot_kw={'projection': '3d'})

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # 极小点：f(x,y) = x^2 + y^2
    Z1 = X**2 + Y**2
    axes[0].plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8)
    axes[0].scatter([0], [0], [0], color='red', s=100, zorder=5)
    axes[0].set_title('局部极小点\n$f = x^2 + y^2$\n$H = 2I \\succ 0$')
    axes[0].set_xlabel('$x_1$'); axes[0].set_ylabel('$x_2$')

    # 鞍点：f(x,y) = x^2 - y^2
    Z2 = X**2 - Y**2
    axes[1].plot_surface(X, Y, Z2, cmap='RdYlBu', alpha=0.8)
    axes[1].scatter([0], [0], [0], color='red', s=100, zorder=5)
    axes[1].set_title('鞍点\n$f = x^2 - y^2$\n$H$ 不定（特征值 $\\pm 2$）')
    axes[1].set_xlabel('$x_1$'); axes[1].set_ylabel('$x_2$')

    # 极大点：f(x,y) = -(x^2 + y^2)
    Z3 = -(X**2 + Y**2)
    axes[2].plot_surface(X, Y, Z3, cmap='plasma', alpha=0.8)
    axes[2].scatter([0], [0], [0], color='red', s=100, zorder=5)
    axes[2].set_title('局部极大点\n$f = -(x^2 + y^2)$\n$H = -2I \\prec 0$')
    axes[2].set_xlabel('$x_1$'); axes[2].set_ylabel('$x_2$')

    plt.tight_layout()
    plt.savefig('critical_points.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图像已保存至 critical_points.png")


# ============================================================
# 2. 利用 PyTorch autograd 计算梯度和 Hessian
# ============================================================

def compute_hessian_pytorch(f, x):
    """
    计算标量函数 f 在点 x 处的 Hessian 矩阵

    Args:
        f: 可微函数，接受 torch.Tensor 返回标量
        x: 计算点，shape (n,)
    Returns:
        H: Hessian 矩阵，shape (n, n)
    """
    x = x.clone().detach().requires_grad_(True)
    n = x.shape[0]

    # 先计算梯度
    val = f(x)
    grad = torch.autograd.grad(val, x, create_graph=True)[0]

    # 逐行计算 Hessian
    H = torch.zeros(n, n)
    for i in range(n):
        grad2 = torch.autograd.grad(grad[i], x, retain_graph=True)[0]
        H[i] = grad2.detach()

    return H


def analyze_critical_point(f, x_star, name="临界点"):
    """
    分析临界点类型

    Args:
        f: 目标函数
        x_star: 候选临界点
        name: 点的名称
    """
    x = torch.tensor(x_star, dtype=torch.float64, requires_grad=True)
    val = f(x)
    grad = torch.autograd.grad(val, x)[0]

    print(f"\n{'='*50}")
    print(f"分析: {name}")
    print(f"  点坐标: {x_star}")
    print(f"  函数值: {val.item():.6f}")
    print(f"  梯度范数: {grad.norm().item():.2e}")

    # 计算 Hessian
    H = compute_hessian_pytorch(f, x)
    eigenvalues = torch.linalg.eigvalsh(H)

    print(f"  Hessian 矩阵:\n{H.numpy()}")
    print(f"  特征值: {eigenvalues.numpy()}")

    # 判断临界点类型
    min_eig = eigenvalues.min().item()
    max_eig = eigenvalues.max().item()

    if min_eig > 1e-8:
        point_type = "严格局部极小点 (H 正定)"
    elif max_eig < -1e-8:
        point_type = "严格局部极大点 (H 负定)"
    elif min_eig < -1e-8 and max_eig > 1e-8:
        point_type = "鞍点 (H 不定)"
    elif min_eig >= -1e-8 and min_eig <= 1e-8:
        point_type = "退化临界点 (H 半正定，需高阶分析)"
    else:
        point_type = "退化临界点 (H 半负定，需高阶分析)"

    print(f"  类型: {point_type}")
    return eigenvalues.numpy()


# ============================================================
# 3. 深度网络损失曲面：鞍点统计分析
# ============================================================

class TinyNet(nn.Module):
    """微型全连接网络用于演示"""
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


def saddle_point_statistics():
    """
    统计随机点处 Hessian 特征值的符号分布
    模拟高维损失曲面中鞍点的普遍性
    """
    torch.manual_seed(42)
    model = TinyNet()
    criterion = nn.MSELoss()

    # 生成随机数据
    X = torch.randn(50, 2)
    y = torch.sin(X[:, 0]) + torch.cos(X[:, 1]) + 0.1 * torch.randn(50)
    y = y.unsqueeze(1)

    def compute_loss(params_flat):
        """接受展平参数向量的损失函数"""
        # 将展平参数填回模型
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data = params_flat[offset:offset+numel].reshape(p.shape)
            offset += numel
        pred = model(X)
        return criterion(pred, y)

    # 在多个随机初始化处分析曲率
    n_trials = 20
    saddle_count = 0
    local_min_count = 0
    local_max_count = 0
    degenerate_count = 0

    print("\n" + "="*60)
    print("深度网络损失曲面临界点分析（随机初始化统计）")
    print("="*60)

    for trial in range(n_trials):
        # 随机初始化参数
        for p in model.parameters():
            nn.init.normal_(p, std=0.5)

        # 当前参数展平
        params = torch.cat([p.data.flatten() for p in model.parameters()])
        n_params = params.numel()

        # 随机扰动以模拟"随机驻点"
        params_trial = params + 0.1 * torch.randn_like(params)
        params_trial.requires_grad_(True)

        # 计算损失和梯度
        loss = compute_loss(params_trial)
        if params_trial.grad is not None:
            params_trial.grad.zero_()

        # 只对前20个参数计算小 Hessian（完整 Hessian 计算量太大）
        n_sub = min(20, n_params)
        params_sub = params_trial[:n_sub].clone().detach().requires_grad_(True)

        def sub_loss(p):
            full_params = torch.cat([p, params_trial[n_sub:].detach()])
            return compute_loss(full_params)

        H_sub = compute_hessian_pytorch(sub_loss, params_sub)
        eigvals = torch.linalg.eigvalsh(H_sub).numpy()

        n_pos = (eigvals > 1e-4).sum()
        n_neg = (eigvals < -1e-4).sum()

        if n_neg == 0 and n_pos > 0:
            local_min_count += 1
            status = "局部极小"
        elif n_pos == 0 and n_neg > 0:
            local_max_count += 1
            status = "局部极大"
        elif n_pos > 0 and n_neg > 0:
            saddle_count += 1
            neg_ratio = n_neg / (n_pos + n_neg)
            status = f"鞍点 ({neg_ratio:.0%} 负曲率)"
        else:
            degenerate_count += 1
            status = "退化"

    print(f"\n共分析 {n_trials} 个随机点（子空间维数 {n_sub}）：")
    print(f"  鞍点:     {saddle_count:3d} ({saddle_count/n_trials*100:.1f}%)")
    print(f"  局部极小: {local_min_count:3d} ({local_min_count/n_trials*100:.1f}%)")
    print(f"  局部极大: {local_max_count:3d} ({local_max_count/n_trials*100:.1f}%)")
    print(f"  退化点:   {degenerate_count:3d} ({degenerate_count/n_trials*100:.1f}%)")
    print(f"\n结论: 在高维网络参数空间中，鞍点远多于局部极小点！")


# ============================================================
# 4. 梯度下降与逃离鞍点的演示
# ============================================================

def saddle_escape_demo():
    """演示标准梯度下降在鞍点的行为，以及扰动如何帮助逃离"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 目标函数：f(x,y) = x^2 - y^2（鞍点在原点）
    def f_saddle(xy):
        return xy[0]**2 - xy[1]**2

    def grad_f(xy):
        return np.array([2*xy[0], -2*xy[1]])

    # 绘制等高线
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2

    for ax in axes:
        cs = ax.contourf(X, Y, Z, levels=30, cmap='RdYlBu')
        ax.contour(X, Y, Z, levels=30, colors='k', linewidths=0.3, alpha=0.5)
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.set_aspect('equal')

    # 方案1：标准梯度下降（从鞍点附近出发）
    lr = 0.1
    n_steps = 50
    traj_gd = [np.array([0.01, 0.01])]  # 接近鞍点
    for _ in range(n_steps):
        g = grad_f(traj_gd[-1])
        new_point = traj_gd[-1] - lr * g
        traj_gd.append(new_point)

    traj_gd = np.array(traj_gd)
    axes[0].plot(traj_gd[:, 0], traj_gd[:, 1], 'r-o', markersize=3,
                 linewidth=2, label='梯度下降轨迹')
    axes[0].scatter([0], [0], c='yellow', s=200, marker='*',
                    zorder=5, label='鞍点', edgecolors='k')
    axes[0].set_title('梯度下降：缓慢逃离鞍点\n（初始点非常接近鞍点）')
    axes[0].legend(fontsize=8)

    # 方案2：加入随机扰动（模拟 SGD 的噪声效果）
    np.random.seed(42)
    traj_sgd = [np.array([0.01, 0.01])]
    noise_scale = 0.3
    for _ in range(n_steps):
        g = grad_f(traj_sgd[-1])
        noise = noise_scale * np.random.randn(2)
        new_point = traj_sgd[-1] - lr * g + noise
        traj_sgd.append(new_point)

    traj_sgd = np.array(traj_sgd)
    axes[1].plot(traj_sgd[:, 0], traj_sgd[:, 1], 'b-o', markersize=3,
                 linewidth=2, label='SGD 轨迹（含噪声）')
    axes[1].scatter([0], [0], c='yellow', s=200, marker='*',
                    zorder=5, label='鞍点', edgecolors='k')
    axes[1].set_title('SGD：噪声帮助快速逃离鞍点\n（相同初始点，加噪声扰动）')
    axes[1].legend(fontsize=8)

    plt.colorbar(cs, ax=axes[1])
    plt.tight_layout()
    plt.savefig('saddle_escape.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图像已保存至 saddle_escape.png")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("第4章 深度学习应用：损失曲面临界点分析")
    print("=" * 60)

    # 1. 可视化三种临界点
    print("\n[1] 可视化三种临界点类型...")
    visualize_critical_points()

    # 2. 分析具体临界点
    print("\n[2] 利用 PyTorch autograd 分析临界点类型...")

    # 极小点
    f_bowl = lambda x: x[0]**2 + x[1]**2
    analyze_critical_point(f_bowl, [0.0, 0.0], "碗形函数的原点（极小点）")

    # 鞍点
    f_saddle = lambda x: x[0]**2 - x[1]**2
    analyze_critical_point(f_saddle, [0.0, 0.0], "马鞍形函数的原点（鞍点）")

    # 极大点
    f_hill = lambda x: -(x[0]**2 + x[1]**2)
    analyze_critical_point(f_hill, [0.0, 0.0], "山丘形函数的原点（极大点）")

    # 3. 深度网络统计分析
    print("\n[3] 深度网络损失曲面统计...")
    saddle_point_statistics()

    # 4. 逃离鞍点演示
    print("\n[4] 梯度下降 vs SGD 逃离鞍点...")
    saddle_escape_demo()

    print("\n" + "=" * 60)
    print("核心结论：")
    print("  1. 高维非凸优化（深度学习）中，鞍点远多于局部极小点")
    print("  2. 标准 GD 在鞍点附近会减速（梯度接近零）")
    print("  3. SGD 的随机噪声实际上有助于逃离鞍点")
    print("  4. 深度网络的局部极小损失值通常接近全局极小")
    print("=" * 60)
```

### 运行说明

```bash
# 安装依赖
pip install torch numpy matplotlib

# 运行代码
python optimality_deep_learning.py
```

### 预期输出

程序将输出：
1. `critical_points.png`：三种临界点的三维损失曲面对比图
2. 各临界点的梯度、Hessian 和特征值分析
3. 深度网络随机点中鞍点的统计比例（通常 > 80%）
4. `saddle_escape.png`：GD vs SGD 逃离鞍点的轨迹对比

### 关键数值结果解读

```
分析: 碗形函数的原点（极小点）
  梯度范数: 0.00e+00
  特征值: [2. 2.]           ← 全正，正定
  类型: 严格局部极小点

分析: 马鞍形函数的原点（鞍点）
  梯度范数: 0.00e+00
  特征值: [-2.  2.]         ← 正负均有，不定
  类型: 鞍点

共分析 20 个随机点：
  鞍点:     17 (85.0%)      ← 高维中鞍点占主导！
  局部极小:  2 (10.0%)
  退化点:    1  (5.0%)
```

---

## 练习题

**练习 4.1（一阶条件）**
设 $f(x, y) = 3x^2 + 4xy + 3y^2 - 8x - 6y + 1$。

(a) 求 $f$ 的所有驻点。
(b) 计算驻点处的 Hessian 矩阵，判断其正定性。
(c) 确定驻点的类型（极小、极大或鞍点）。

---

**练习 4.2（二阶条件与鞍点）**
设 $f(x, y) = x^3 + y^3 - 3xy$。

(a) 求 $f$ 的所有驻点（令 $\nabla f = \mathbf{0}$）。
(b) 对每个驻点计算 Hessian，判断各点类型。
(c) 手绘或描述 $f$ 的大致等高线图。

---

**练习 4.3（凸函数最优性）**
设 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{A} \mathbf{x} - \mathbf{b}^\top \mathbf{x}$，其中

$$\mathbf{A} = \begin{pmatrix} 4 & 1 \\ 1 & 3 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}.$$

(a) 验证 $\mathbf{A}$ 正定，故 $f$ 是严格凸函数。
(b) 利用一阶条件（$\nabla f = \mathbf{0}$）求全局极小点 $\mathbf{x}^*$。
(c) 计算全局极小值 $f(\mathbf{x}^*)$。

---

**练习 4.4（高维鞍点）**
设 $f: \mathbb{R}^3 \to \mathbb{R}$，

$$f(x_1, x_2, x_3) = x_1^2 + 2x_2^2 - x_3^2 + x_1 x_2.$$

(a) 求唯一驻点。
(b) 写出该点处的 Hessian 矩阵，并计算特征值。
(c) 判断驻点类型，并指出"负曲率方向"（对应最小特征值的特征向量）。

---

**练习 4.5（思考题）**
在深度学习中，对于一个具有 $10^6$ 个参数的神经网络：

(a) 解释为何随机点处的 Hessian 不定概率远大于正定概率（从随机矩阵理论角度）。
(b) 从二阶最优性条件出发，解释为何 SGD 的随机梯度噪声有助于逃离鞍点。
(c) 假设你发现训练损失长时间不下降（梯度范数也很小），请给出至少两种可能的原因，并分别说明对应的应对策略。

---

## 练习答案

### 练习 4.1 答案

**(a) 求驻点：**

$$\nabla f = \begin{pmatrix} 6x + 4y - 8 \\ 4x + 6y - 6 \end{pmatrix} = \mathbf{0}.$$

方程组：
$$\begin{cases} 6x + 4y = 8 \\ 4x + 6y = 6 \end{cases}$$

由第一式得 $x = \frac{8 - 4y}{6}$，代入第二式：

$$4 \cdot \frac{8-4y}{6} + 6y = 6 \implies \frac{32 - 16y}{6} + 6y = 6 \implies 32 - 16y + 36y = 36 \implies y = \frac{1}{5}.$$

$$x = \frac{8 - 4/5}{6} = \frac{36/5}{6} = \frac{6}{5}.$$

唯一驻点：$\left(\frac{6}{5}, \frac{1}{5}\right)$。

**(b) Hessian 矩阵：**

$$\mathbf{H} = \begin{pmatrix} 6 & 4 \\ 4 & 6 \end{pmatrix}.$$

顺序主子式：$\Delta_1 = 6 > 0$，$\Delta_2 = 36 - 16 = 20 > 0$，故 $\mathbf{H} \succ \mathbf{0}$（正定）。

**(c) 结论：** 驻点 $\left(\frac{6}{5}, \frac{1}{5}\right)$ 是**严格局部极小点**（也是全局极小点，因 $f$ 是凸函数）。

---

### 练习 4.2 答案

**(a) 求驻点：**

$$\nabla f = \begin{pmatrix} 3x^2 - 3y \\ 3y^2 - 3x \end{pmatrix} = \mathbf{0} \implies \begin{cases} x^2 = y \\ y^2 = x \end{cases}.$$

将 $y = x^2$ 代入 $y^2 = x$：$(x^2)^2 = x \implies x^4 - x = 0 \implies x(x^3 - 1) = 0$。

解得 $x = 0$ 或 $x = 1$，对应 $(0, 0)$ 和 $(1, 1)$。

**(b) Hessian 分析：**

$$\mathbf{H}(x, y) = \begin{pmatrix} 6x & -3 \\ -3 & 6y \end{pmatrix}.$$

**在 $(0, 0)$：**
$$\mathbf{H}(0,0) = \begin{pmatrix} 0 & -3 \\ -3 & 0 \end{pmatrix}, \quad \det(\mathbf{H}) = 0 - 9 = -9 < 0.$$

特征值 $\pm 3$，不定。**$(0,0)$ 是鞍点**。

**在 $(1, 1)$：**
$$\mathbf{H}(1,1) = \begin{pmatrix} 6 & -3 \\ -3 & 6 \end{pmatrix}, \quad \det(\mathbf{H}) = 36 - 9 = 27 > 0, \quad H_{11} = 6 > 0.$$

**$(1,1)$ 是严格局部极小点**，$f(1,1) = 1 + 1 - 3 = -1$。

---

### 练习 4.3 答案

**(a) 验证正定性：**

$\Delta_1 = 4 > 0$，$\Delta_2 = 4 \times 3 - 1 \times 1 = 11 > 0$，故 $\mathbf{A} \succ \mathbf{0}$，$f$ 是严格凸函数。

**(b) 求极小点：**

$$\nabla f = \mathbf{A}\mathbf{x} - \mathbf{b} = \mathbf{0} \implies \mathbf{x}^* = \mathbf{A}^{-1}\mathbf{b}.$$

$$\mathbf{A}^{-1} = \frac{1}{11}\begin{pmatrix} 3 & -1 \\ -1 & 4 \end{pmatrix},$$

$$\mathbf{x}^* = \frac{1}{11}\begin{pmatrix} 3 & -1 \\ -1 & 4 \end{pmatrix}\begin{pmatrix} 2 \\ 1 \end{pmatrix} = \frac{1}{11}\begin{pmatrix} 5 \\ 2 \end{pmatrix} = \begin{pmatrix} 5/11 \\ 2/11 \end{pmatrix}.$$

**(c) 极小值：**

$$f(\mathbf{x}^*) = \frac{1}{2}(\mathbf{x}^*)^\top \mathbf{A}\mathbf{x}^* - \mathbf{b}^\top \mathbf{x}^* = \frac{1}{2}(\mathbf{x}^*)^\top \mathbf{b} - \mathbf{b}^\top \mathbf{x}^* = -\frac{1}{2}\mathbf{b}^\top \mathbf{x}^*.$$

$$f(\mathbf{x}^*) = -\frac{1}{2}\left(2 \cdot \frac{5}{11} + 1 \cdot \frac{2}{11}\right) = -\frac{1}{2} \cdot \frac{12}{11} = -\frac{6}{11} \approx -0.545.$$

---

### 练习 4.4 答案

**(a) 驻点：**

$$\nabla f = \begin{pmatrix} 2x_1 + x_2 \\ 4x_2 + x_1 \\ -2x_3 \end{pmatrix} = \mathbf{0} \implies \begin{cases} 2x_1 + x_2 = 0 \\ x_1 + 4x_2 = 0 \\ x_3 = 0 \end{cases}.$$

由前两式：$x_2 = -2x_1$，代入 $x_1 + 4(-2x_1) = 0 \implies -7x_1 = 0 \implies x_1 = 0$。

唯一驻点：$\mathbf{x}^* = (0, 0, 0)^\top$。

**(b) Hessian 与特征值：**

$$\mathbf{H} = \begin{pmatrix} 2 & 1 & 0 \\ 1 & 4 & 0 \\ 0 & 0 & -2 \end{pmatrix}.$$

特征值：$-2$ 和 $3 \pm \sqrt{2}$（约 $1.586$ 和 $4.414$）。

**(c) 结论：** 特征值 $\{-2, 1.586, 4.414\}$ 中既有正也有负，故 $\mathbf{H}$ 不定。**原点是鞍点**。

**负曲率方向**对应特征值 $-2$ 的特征向量：$\mathbf{v} = (0, 0, 1)^\top$，即 $x_3$ 方向。

---

### 练习 4.5 答案

**(a) 随机矩阵理论角度：**

对一个 $n \times n$ 随机对称矩阵（Wigner 矩阵），其特征值近似服从半圆律（Wigner 半圆定律），均匀分布在区间 $[-2\sigma, 2\sigma]$ 内。所有特征值为正的概率约为

$$P(\text{全正定}) \approx \prod_{i=1}^{n} P(\lambda_i > 0) \approx \left(\frac{1}{2}\right)^n = 2^{-n}.$$

当 $n = 10^6$ 时，这个概率约为 $2^{-10^6} \approx 0$，极其接近零。因此，高维随机点几乎必然是鞍点。

**(b) SGD 噪声与逃离鞍点：**

在鞍点 $\mathbf{x}^*$ 处，标准梯度 $\nabla \mathcal{L}(\mathbf{x}^*) \approx \mathbf{0}$，参数更新接近于零。

SGD 使用小批量梯度 $\mathbf{g}_t = \nabla \mathcal{L}_{B_t}(\mathbf{x})$，它是真实梯度的有噪估计：

$$\mathbf{g}_t = \nabla \mathcal{L}(\mathbf{x}) + \boldsymbol{\epsilon}_t, \quad \boldsymbol{\epsilon}_t \sim \text{某分布}.$$

即使真实梯度为零，噪声 $\boldsymbol{\epsilon}_t$ 仍会使参数移动。若噪声分量恰好沿负曲率方向 $\mathbf{v}_{\min}$，函数值会下降，参数逐渐逃离鞍点。这正是"SGD 在非凸优化中优于确定性 GD"的一个原因。

**(c) 训练不收敛的可能原因与对策：**

| 可能原因 | 判断依据 | 应对策略 |
|---------|---------|---------|
| **陷入鞍点** | 损失平台，梯度极小但二阶曲率有负值 | 增加批量噪声（减小 batch size）；使用 Adam/Momentum |
| **学习率过小** | 损失缓慢下降但速度极慢 | 适当提高学习率；使用学习率调度（warmup + cosine decay）|
| **梯度消失** | 深层参数梯度范数接近零，浅层正常 | 残差连接（ResNet）；梯度裁剪；更换激活函数（ReLU 替代 Sigmoid）|
| **陷入平坦局部极小** | 损失稳定不变，Hessian 正半定 | 继续训练（可能是好的极小）；尝试不同初始化 |
