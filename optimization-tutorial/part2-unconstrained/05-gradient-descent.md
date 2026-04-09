# 第5章：梯度下降法

---

## 学习目标

学完本章后，你将能够：

1. **理解梯度下降的几何直觉**：掌握负梯度方向作为最速下降方向的原理，理解迭代公式 $\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)$ 的推导依据
2. **掌握多种步长选择策略**：包括固定步长、精确线搜索和 Armijo 回溯线搜索的原理与适用场景
3. **分析凸函数的收敛性**：证明 $L$-光滑凸函数梯度下降的 $O(1/k)$ 收敛率
4. **理解强凸函数的线性收敛**：掌握条件数对收敛速度的影响，理解为何病态问题难以优化
5. **认识梯度下降的主要变体**：了解批量梯度下降、随机梯度下降与小批量梯度下降的区别与联系，并能用 PyTorch 实现 SGD

---

## 5.1 梯度下降的基本思想

### 5.1.1 从一维到多维：下山的直觉

想象你站在一座山上，目标是尽快到达山谷最低处。你的策略很自然：**沿着当前位置坡度最陡的方向向下走**。梯度下降法正是这一直觉的数学化表达。

对于可微函数 $f: \mathbb{R}^n \to \mathbb{R}$，我们希望求解：

$$\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})$$

**关键问题**：在当前点 $\mathbf{x}_k$，沿哪个方向移动一小步能使函数值下降最多？

### 5.1.2 最速下降方向的推导

设移动方向为单位向量 $\mathbf{d}$（$\|\mathbf{d}\| = 1$），移动步长为 $\epsilon > 0$。由 Taylor 展开：

$$f(\mathbf{x}_k + \epsilon \mathbf{d}) \approx f(\mathbf{x}_k) + \epsilon \langle \nabla f(\mathbf{x}_k), \mathbf{d} \rangle$$

函数值的变化量为 $\epsilon \langle \nabla f(\mathbf{x}_k), \mathbf{d} \rangle$。为使下降量最大，需最小化内积 $\langle \nabla f(\mathbf{x}_k), \mathbf{d} \rangle$。

由 Cauchy-Schwarz 不等式：

$$\langle \nabla f(\mathbf{x}_k), \mathbf{d} \rangle \geq -\|\nabla f(\mathbf{x}_k)\| \cdot \|\mathbf{d}\| = -\|\nabla f(\mathbf{x}_k)\|$$

等号成立当且仅当 $\mathbf{d} = -\dfrac{\nabla f(\mathbf{x}_k)}{\|\nabla f(\mathbf{x}_k)\|}$，即**负梯度方向**是最速下降方向。

### 5.1.3 梯度下降迭代公式

基于上述分析，梯度下降法的迭代格式为：

$$\boxed{\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \nabla f(\mathbf{x}_k)}$$

其中：
- $\mathbf{x}_k$：第 $k$ 步的当前点
- $\nabla f(\mathbf{x}_k)$：函数在 $\mathbf{x}_k$ 处的梯度
- $\alpha_k > 0$：第 $k$ 步的**步长**（学习率），控制每步移动的距离
- $\mathbf{x}_{k+1}$：更新后的新点

**算法流程**（梯度下降法）：

```
输入：初始点 x_0，终止准则 ε > 0
重复：
    计算梯度 g_k = ∇f(x_k)
    若 ‖g_k‖ ≤ ε，停止
    选择步长 α_k
    更新 x_{k+1} = x_k - α_k * g_k
    k = k + 1
输出：x_k（近似最优解）
```

### 5.1.4 一个简单例子

考虑二次函数 $f(x_1, x_2) = x_1^2 + 4x_2^2$，最优解为 $\mathbf{x}^* = (0, 0)$。

梯度为 $\nabla f(\mathbf{x}) = (2x_1,\ 8x_2)^\top$。

从初始点 $\mathbf{x}_0 = (1, 1)^\top$ 出发，取固定步长 $\alpha = 0.1$：

$$\mathbf{x}_1 = \begin{pmatrix}1 \\ 1\end{pmatrix} - 0.1 \begin{pmatrix}2 \\ 8\end{pmatrix} = \begin{pmatrix}0.8 \\ 0.2\end{pmatrix}$$

$$\mathbf{x}_2 = \begin{pmatrix}0.8 \\ 0.2\end{pmatrix} - 0.1 \begin{pmatrix}1.6 \\ 1.6\end{pmatrix} = \begin{pmatrix}0.64 \\ 0.04\end{pmatrix}$$

可以看到 $x_2$ 分量因梯度较大而收缩更快，但这也带来了震荡问题（见 5.4 节关于条件数的讨论）。

---

## 5.2 步长选择策略

步长 $\alpha_k$ 的选择对梯度下降的性能至关重要。步长太大可能导致震荡甚至发散；步长太小则收敛过慢。

### 5.2.1 固定步长

最简单的策略：取常数步长 $\alpha_k = \alpha$（对所有 $k$ 相同）。

**理论保证**：若 $f$ 是 $L$-光滑函数（梯度 Lipschitz 连续），即：

$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L \|\mathbf{x} - \mathbf{y}\|, \quad \forall \mathbf{x}, \mathbf{y}$$

则取步长 $\alpha \leq \dfrac{1}{L}$ 可保证每步函数值单调下降。

**直觉**：$L$ 是函数曲率的上界，步长不超过 $1/L$ 意味着我们的一阶近似足够可靠。

**优点**：实现简单，无需每步计算额外信息。
**缺点**：需要知道 $L$（通常未知），对不同问题需要调参。

### 5.2.2 精确线搜索

在每步沿负梯度方向做**精确一维最小化**：

$$\alpha_k = \arg\min_{\alpha \geq 0} f(\mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k))$$

**例子**：对二次函数 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{A} \mathbf{x} - \mathbf{b}^\top \mathbf{x}$（$\mathbf{A} \succ 0$），令方向 $\mathbf{d}_k = -\nabla f(\mathbf{x}_k)$，精确步长为：

$$\alpha_k = \frac{\|\mathbf{d}_k\|^2}{\mathbf{d}_k^\top \mathbf{A} \mathbf{d}_k}$$

**优点**：每步函数值下降最大，理论性质好。
**缺点**：计算成本高，一般函数无解析解，需额外迭代求解。

### 5.2.3 Armijo 回溯线搜索

实践中最常用的策略，通过**回溯**找到满足足够下降条件的步长。

**Armijo 条件**（sufficient decrease condition）：

$$f(\mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)) \leq f(\mathbf{x}_k) - c \cdot \alpha \|\nabla f(\mathbf{x}_k)\|^2$$

其中 $c \in (0, 1)$ 为常数（通常取 $c = 10^{-4}$）。

**算法**（回溯线搜索）：

```
输入：当前点 x_k，初始步长 α_0，参数 c ∈ (0,1)，缩减因子 β ∈ (0,1)
设 α = α_0
当 Armijo 条件不满足时：
    α = β * α    （通常 β = 0.5）
返回 α
```

**几何解释**：Armijo 条件要求函数值下降至少是线性预测下降量（$\alpha \|\nabla f\|^2$）的 $c$ 倍。这在理论上排除了步长过小的情况。

**Wolfe 条件**（进阶）：在 Armijo 条件基础上再加**曲率条件**（curvature condition），还要求步长不能太小：

$$\langle \nabla f(\mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)),\ -\nabla f(\mathbf{x}_k) \rangle \geq -c_2 \|\nabla f(\mathbf{x}_k)\|^2$$

其中 $c_2 \in (c, 1)$。Wolfe 条件在拟牛顿法中尤为重要（见第6章）。

### 5.2.4 步长选择的比较

| 策略 | 计算成本 | 收敛保证 | 实用性 |
|------|----------|----------|--------|
| 固定步长 $\alpha \leq 1/L$ | 极低 | 有（需知 $L$） | 中等 |
| 精确线搜索 | 高 | 强 | 低（一般问题） |
| Armijo 回溯 | 低-中 | 有 | 高 |
| Wolfe 条件线搜索 | 中 | 强 | 高（拟牛顿） |

---

## 5.3 收敛性分析

本节对梯度下降法的收敛性进行严格分析。分两个层次：一般凸函数和强凸函数。

### 5.3.1 基本假设

**假设 A（$L$-光滑性）**：$f$ 可微，梯度满足 Lipschitz 条件：

$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L \|\mathbf{x} - \mathbf{y}\|$$

等价地，对所有 $\mathbf{x}, \mathbf{y}$：

$$f(\mathbf{y}) \leq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle + \frac{L}{2}\|\mathbf{y} - \mathbf{x}\|^2$$

这是一个**二次上界**，直觉上说明函数弯曲程度有上界。

**假设 B（凸性）**：$f$ 是凸函数，即对所有 $\mathbf{x}, \mathbf{y}$：

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$$

这是一个**线性下界**，梯度给出的线性估计不会高估函数值。

### 5.3.2 凸函数的 $O(1/k)$ 收敛率

**定理 5.1**（凸函数收敛率）：设 $f$ 满足假设 A 和 B，$f^* = \min f$，$\mathbf{x}^*$ 为最优解。取固定步长 $\alpha = 1/L$，则：

$$f(\mathbf{x}_k) - f^* \leq \frac{L \|\mathbf{x}_0 - \mathbf{x}^*\|^2}{2k}$$

即收敛率为 $O(1/k)$。

**证明思路**：

**步骤1**：利用 $L$-光滑性建立一步下降不等式。

取 $\mathbf{y} = \mathbf{x}_{k+1} = \mathbf{x}_k - \frac{1}{L}\nabla f(\mathbf{x}_k)$，代入二次上界：

$$f(\mathbf{x}_{k+1}) \leq f(\mathbf{x}_k) - \frac{1}{L}\|\nabla f(\mathbf{x}_k)\|^2 + \frac{L}{2} \cdot \frac{1}{L^2}\|\nabla f(\mathbf{x}_k)\|^2$$

化简得：

$$f(\mathbf{x}_{k+1}) \leq f(\mathbf{x}_k) - \frac{1}{2L}\|\nabla f(\mathbf{x}_k)\|^2 \quad \cdots (*)$$

**步骤2**：利用凸性建立梯度与函数差的关系。

由凸函数定义，取 $\mathbf{y} = \mathbf{x}^*$：

$$f(\mathbf{x}^*) \geq f(\mathbf{x}_k) + \langle \nabla f(\mathbf{x}_k), \mathbf{x}^* - \mathbf{x}_k \rangle$$

所以：

$$f(\mathbf{x}_k) - f^* \leq \langle \nabla f(\mathbf{x}_k), \mathbf{x}_k - \mathbf{x}^* \rangle \leq \|\nabla f(\mathbf{x}_k)\| \cdot \|\mathbf{x}_k - \mathbf{x}^*\| \quad \cdots (**)$$

**步骤3**：利用距离的递推减少。

由 $(*)$ 和步骤2的技术引理可以推导出距离递推式，最终对 $k$ 步求和取平均，得到 $O(1/k)$ 收敛率。$\square$

**解释**：该定理说明，要使最优性间隙 $f(\mathbf{x}_k) - f^* \leq \epsilon$，需要迭代次数 $k = O(1/\epsilon)$。

### 5.3.3 函数值下降不等式的直觉

从 $(*)$ 式可以清晰看到梯度下降的本质：

$$\underbrace{f(\mathbf{x}_{k+1})}_{\text{新函数值}} \leq \underbrace{f(\mathbf{x}_k)}_{\text{旧函数值}} - \underbrace{\frac{1}{2L}\|\nabla f(\mathbf{x}_k)\|^2}_{\text{下降量}}$$

- 只要 $\nabla f(\mathbf{x}_k) \neq \mathbf{0}$，函数值就严格下降
- 下降量与梯度模的平方成正比
- 当 $\|\nabla f(\mathbf{x}_k)\| \to 0$ 时，更新趋于停止，说明接近驻点

---

## 5.4 收敛速度与条件数

凸函数的 $O(1/k)$ 收敛率在强凸情形下可显著加速。本节分析强凸函数的线性收敛及条件数的影响。

### 5.4.1 强凸函数的线性收敛

**定义**（$\mu$-强凸）：$f$ 是 $\mu$-强凸（$\mu > 0$），若对所有 $\mathbf{x}, \mathbf{y}$：

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle + \frac{\mu}{2}\|\mathbf{y} - \mathbf{x}\|^2$$

强凸性意味着函数有唯一最优解，且"碗形"不太平坦。

**定理 5.2**（强凸函数线性收敛）：设 $f$ 满足 $L$-光滑性和 $\mu$-强凸性，取步长 $\alpha = 1/L$，则：

$$\|\mathbf{x}_{k+1} - \mathbf{x}^*\|^2 \leq \left(1 - \frac{\mu}{L}\right) \|\mathbf{x}_k - \mathbf{x}^*\|^2$$

递推可得：

$$\|\mathbf{x}_k - \mathbf{x}^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^k \|\mathbf{x}_0 - \mathbf{x}^*\|^2$$

这是**线性收敛**（或**几何收敛**）：每步误差至少缩小常数比例 $(1 - \mu/L)$。

**比较**：
- 凸函数：$O(1/k)$ 收敛（次线性）
- 强凸函数：$O(\rho^k)$，$\rho = 1 - \mu/L < 1$（线性收敛）

要达到 $\epsilon$ 精度所需迭代次数：

| 问题类型 | 所需迭代次数 |
|----------|-------------|
| 凸函数 | $O(1/\epsilon)$ |
| 强凸函数 | $O\!\left(\frac{L}{\mu} \ln \frac{1}{\epsilon}\right)$ |

### 5.4.2 条件数的影响

**定义**（条件数）：对 $L$-光滑、$\mu$-强凸函数，定义**条件数**：

$$\kappa = \frac{L}{\mu} \geq 1$$

收敛率为 $1 - 1/\kappa$。当 $\kappa$ 大时：

- $1 - 1/\kappa \approx 1$，每步改进极小
- 需要 $O(\kappa \ln(1/\epsilon))$ 次迭代

**直观理解**：条件数衡量函数在不同方向上的曲率差异。对于二次函数 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{A} \mathbf{x}$（$\mathbf{A} \succ 0$）：

$$\mu = \lambda_{\min}(\mathbf{A}), \quad L = \lambda_{\max}(\mathbf{A}), \quad \kappa = \frac{\lambda_{\max}(\mathbf{A})}{\lambda_{\min}(\mathbf{A})}$$

当 $\kappa$ 很大时（**病态问题**），等高线呈现高度椭圆形，梯度下降会产生严重的**锯齿状震荡**，如下图所示：

```
高条件数（κ=100）的等高线：        低条件数（κ=2）的等高线：

    /--------\                        /---\
   /          \  ← 梯度下降路径       /     \
  | ...→..↙.. |     呈锯齿           |  ↘  |  ← 几乎直线
   \          /                       \     /
    \--------/                         \---/

收敛慢，震荡明显                    收敛快，路径近直线
```

**解决方案**：预条件（preconditioning）技术可以降低有效条件数；牛顿法利用 Hessian 自适应地消除条件数问题（见第6章）。

### 5.4.3 不同函数类型的收敛率汇总

| 函数类型 | 假设 | 收敛率 | 步长 |
|----------|------|--------|------|
| 光滑凸函数 | $L$-光滑 + 凸 | $O(1/k)$ | $1/L$ |
| 光滑强凸函数 | $L$-光滑 + $\mu$-强凸 | $O(\rho^k)$，$\rho = 1-\mu/L$ | $1/L$ |
| 非凸光滑函数 | $L$-光滑 | $O(1/\sqrt{k})$（找驻点） | $1/L$ |
| 最优步长强凸 | $L$-光滑 + $\mu$-强凸 | $O(\rho^k)$，$\rho = \frac{\kappa-1}{\kappa+1}$ | $\frac{2}{\mu+L}$ |

---

## 5.5 梯度下降的变体

### 5.5.1 三种梯度下降的分类

根据每次迭代使用的数据量，梯度下降分为三类：

**批量梯度下降**（Batch Gradient Descent, BGD）：

$$\nabla f(\mathbf{x}_k) = \frac{1}{n} \sum_{i=1}^n \nabla f_i(\mathbf{x}_k)$$

使用全部 $n$ 个样本计算精确梯度。
- 优点：梯度精确，收敛稳定
- 缺点：每步计算代价 $O(n)$，大数据集无法使用

**随机梯度下降**（Stochastic Gradient Descent, SGD）：

$$\nabla f(\mathbf{x}_k) \approx \nabla f_{i_k}(\mathbf{x}_k)$$

每步随机选一个样本 $i_k$，用其梯度代替全梯度。
- 优点：每步代价 $O(1)$，适合大规模问题
- 缺点：梯度噪声大，需要学习率衰减

**小批量梯度下降**（Mini-batch Gradient Descent）：

$$\nabla f(\mathbf{x}_k) \approx \frac{1}{B} \sum_{i \in \mathcal{B}_k} \nabla f_i(\mathbf{x}_k)$$

每步使用 $B$ 个样本（通常 $B = 32, 64, 128, 256$）。
- 优点：平衡精度与效率，充分利用 GPU 并行计算
- 缺点：需要调节批量大小

在实践中，深度学习几乎都使用小批量梯度下降，通称 SGD。

### 5.5.2 随机梯度的无偏性

SGD 之所以有效，关键在于随机梯度是无偏估计量：

$$\mathbb{E}_{i_k}[\nabla f_{i_k}(\mathbf{x}_k)] = \nabla f(\mathbf{x}_k)$$

其中期望对随机选取的样本下标 $i_k$ 取。这保证了在期望意义下，SGD 是向正确方向前进的。

### 5.5.3 SGD 的收敛性（概述）

SGD 的收敛分析比确定性梯度下降更复杂，需要考虑方差。核心结论：

- **凸函数**：SGD 以 $O(1/\sqrt{k})$ 收敛（慢于确定性 $O(1/k)$）
- **强凸函数**：SGD 以 $O(1/k)$ 收敛（慢于确定性线性收敛）
- **学习率衰减**：使用 $\alpha_k = \alpha_0 / k$ 可以达到最优收敛率

这些内容将在第16章（随机梯度下降）中详细展开。

### 5.5.4 动量方法简介

纯梯度下降在高条件数问题上表现差（见5.4节），**动量**方法通过积累历史梯度信息来加速：

$$\mathbf{v}_{k+1} = \beta \mathbf{v}_k + \nabla f(\mathbf{x}_k)$$
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \mathbf{v}_{k+1}$$

其中 $\beta \in [0, 1)$ 为动量系数（通常取 0.9）。

直觉上，动量类似物理中的惯性：在持续相同方向的梯度上积累速度，在震荡方向上抵消。这将在第17章（动量方法）中详细讨论。

---

## 本章小结

| 概念 | 公式/要点 |
|------|-----------|
| 迭代公式 | $\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \nabla f(\mathbf{x}_k)$ |
| 最速下降方向 | 负梯度方向 $-\nabla f(\mathbf{x}_k)$ |
| 固定步长条件 | $\alpha \leq 1/L$（$L$ 为梯度 Lipschitz 常数） |
| Armijo 条件 | $f(\mathbf{x}_k - \alpha g_k) \leq f(\mathbf{x}_k) - c\alpha\|g_k\|^2$ |
| 凸函数收敛率 | $f(\mathbf{x}_k) - f^* \leq \dfrac{L\|\mathbf{x}_0-\mathbf{x}^*\|^2}{2k}$，即 $O(1/k)$ |
| 强凸函数收敛率 | $\|\mathbf{x}_k - \mathbf{x}^*\|^2 \leq (1-\mu/L)^k \|\mathbf{x}_0-\mathbf{x}^*\|^2$，线性收敛 |
| 条件数 | $\kappa = L/\mu$，越大收敛越慢 |
| SGD（深度学习） | 小批量随机梯度，每步代价 $O(B)$ |
| 动量方法 | 积累历史梯度，改善病态问题收敛 |

**核心要点回顾**：

1. 梯度下降是负梯度方向的迭代优化，直觉来源于最速下降方向
2. 步长是关键超参数：固定步长简单但需调参，Armijo 回溯兼顾理论与实践
3. 凸函数有 $O(1/k)$ 保证，强凸有线性收敛保证
4. 条件数刻画了问题的难易程度，大条件数导致锯齿震荡
5. SGD 是深度学习的基础优化器，以牺牲部分收敛速度换取每步的效率

---

## 深度学习应用：SGD 基础与学习率的重要性

### 背景

在深度学习中，我们训练神经网络来最小化损失函数：

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n \ell(f(\mathbf{x}_i; \boldsymbol{\theta}),\ y_i)$$

其中 $\boldsymbol{\theta}$ 是网络参数，$\ell$ 是逐样本损失（如交叉熵）。由于 $n$ 可达数百万，批量梯度下降不可行，SGD 及其变体是标准做法。

### PyTorch 中的 SGD 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 基本用法：torch.optim.SGD
# ============================================================

# 定义一个简单的线性回归模型
model = nn.Linear(2, 1, bias=True)

# 创建 SGD 优化器
# lr: 学习率（步长 α）
# momentum: 动量系数 β（默认 0，即普通 SGD）
# weight_decay: L2 正则化系数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# 训练一步的标准流程
def train_step(model, optimizer, x_batch, y_batch, loss_fn):
    optimizer.zero_grad()          # 清零上一步的梯度
    y_pred = model(x_batch)        # 前向传播
    loss = loss_fn(y_pred, y_batch) # 计算损失
    loss.backward()                # 反向传播，计算梯度
    optimizer.step()               # 梯度下降更新参数
    return loss.item()


# ============================================================
# 2. 可视化学习率的影响
# ============================================================

def quadratic_loss(x, A, b):
    """二次损失函数 f(x) = 0.5 * x^T A x - b^T x"""
    return 0.5 * x @ A @ x - b @ x

def quadratic_grad(x, A, b):
    """二次函数的梯度"""
    return A @ x - b

def gradient_descent_trajectory(A, b, x0, lr, n_steps):
    """运行梯度下降，返回轨迹"""
    x = x0.clone()
    trajectory = [x.clone()]
    losses = []

    for _ in range(n_steps):
        grad = quadratic_grad(x, A, b)
        x = x - lr * grad
        trajectory.append(x.clone())
        losses.append(quadratic_loss(x, A, b).item())

    return trajectory, losses

# 构造一个二维二次问题（条件数 = 10）
torch.manual_seed(42)
# A = diag(1, 10)，特征值为 1 和 10，条件数 κ = 10
A = torch.tensor([[1.0, 0.0], [0.0, 10.0]])
b = torch.zeros(2)
x_star = torch.zeros(2)  # 最优解

x0 = torch.tensor([1.0, 1.0])  # 初始点
L = 10.0  # 最大特征值（梯度 Lipschitz 常数）

# 用不同学习率运行梯度下降
learning_rates = {
    r'$\alpha = 0.05 = 0.5/L$（保守）': 0.05,
    r'$\alpha = 0.1 = 1/L$（理论最优）': 0.1,
    r'$\alpha = 0.18 = 1.8/L$（接近临界）': 0.18,
    r'$\alpha = 0.25 = 2.5/L$（过大，发散）': 0.25,
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图1：不同学习率的函数值收敛曲线
ax1 = axes[0]
for label, lr in learning_rates.items():
    _, losses = gradient_descent_trajectory(A, b, x0, lr=lr, n_steps=100)
    # 限制显示范围避免发散情况破坏图形
    clipped = [min(l, 5.0) for l in losses]
    ax1.semilogy(clipped, label=label, linewidth=2)

ax1.set_xlabel('迭代次数 $k$')
ax1.set_ylabel('$f(\\mathbf{x}_k) - f^*$（对数轴）')
ax1.set_title('不同学习率的收敛曲线')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 100])

# 子图2：最优学习率下的梯度下降轨迹（等高线图）
ax2 = axes[1]
x1_range = np.linspace(-1.2, 1.2, 300)
x2_range = np.linspace(-1.2, 1.2, 300)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = 0.5 * (X1**2 + 10 * X2**2)  # 二次函数值

contour = ax2.contour(X1, X2, Z, levels=20, cmap='Blues')
ax2.clabel(contour, inline=True, fontsize=8)

# 绘制最优步长（1/L）的轨迹
traj, _ = gradient_descent_trajectory(A, b, x0, lr=0.1, n_steps=30)
traj_x = [t[0].item() for t in traj]
traj_y = [t[1].item() for t in traj]
ax2.plot(traj_x, traj_y, 'r-o', markersize=4, linewidth=1.5,
         label=r'$\alpha=1/L$，收敛路径')
ax2.plot(x0[0], x0[1], 'go', markersize=10, label='初始点 $\\mathbf{x}_0$')
ax2.plot(0, 0, 'r*', markersize=15, label='最优解 $\\mathbf{x}^*$')

ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_title(f'梯度下降轨迹（$\\kappa = {int(L/1)}$，等高线为 $f$）')
ax2.legend(fontsize=9)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent_demo.png', dpi=150, bbox_inches='tight')
plt.show()
print("图像已保存为 gradient_descent_demo.png")


# ============================================================
# 3. 在真实神经网络上演示 SGD
# ============================================================

# 生成合成分类数据集（二分类）
torch.manual_seed(0)
n_samples = 1000
X = torch.randn(n_samples, 2)
# 真实决策边界：x1 + 2*x2 > 0
y = ((X[:, 0] + 2 * X[:, 1]) > 0).float().unsqueeze(1)

# 构建两层神经网络
class TwoLayerNet(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 用不同学习率训练，比较收敛
def train_model(lr, n_epochs=200, batch_size=32):
    model = TwoLayerNet(hidden_size=16)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    train_losses = []
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(loader))

    return train_losses

print("训练神经网络（比较不同 SGD 学习率）...")
results = {}
for lr in [0.001, 0.01, 0.1, 1.0]:
    losses = train_model(lr)
    results[f'lr={lr}'] = losses
    final_loss = losses[-1]
    print(f"  学习率 {lr:5.3f}：最终损失 = {final_loss:.4f}")

# 绘制训练曲线
plt.figure(figsize=(8, 5))
for label, losses in results.items():
    plt.plot(losses, label=label, linewidth=2)
plt.xlabel('训练轮次（Epoch）')
plt.ylabel('训练损失（BCE）')
plt.title('SGD 在神经网络上的收敛：不同学习率的比较')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('sgd_neural_network.png', dpi=150, bbox_inches='tight')
plt.show()
print("图像已保存为 sgd_neural_network.png")


# ============================================================
# 4. Armijo 回溯线搜索的实现
# ============================================================

def armijo_line_search(f, grad_f, x, direction, alpha0=1.0, c=1e-4, beta=0.5):
    """
    Armijo 回溯线搜索

    参数:
        f: 目标函数
        grad_f: 梯度函数
        x: 当前点
        direction: 搜索方向（通常为负梯度）
        alpha0: 初始步长
        c: Armijo 参数（通常 1e-4）
        beta: 步长缩减因子（通常 0.5）

    返回:
        满足 Armijo 条件的步长
    """
    alpha = alpha0
    fx = f(x)
    gx = grad_f(x)
    slope = torch.dot(gx, direction).item()  # 方向导数

    max_iter = 50
    for _ in range(max_iter):
        x_new = x + alpha * direction
        if f(x_new).item() <= fx + c * alpha * slope:
            break  # Armijo 条件满足
        alpha *= beta

    return alpha

# 演示 Armijo 线搜索
print("\n演示 Armijo 回溯线搜索：")
x_test = torch.tensor([1.0, 1.0])
f_test = lambda x: quadratic_loss(x, A, b)
grad_test = lambda x: quadratic_grad(x, A, b)

for trial_x in [torch.tensor([1.0, 1.0]), torch.tensor([0.5, 0.5])]:
    direction = -grad_test(trial_x)
    alpha = armijo_line_search(f_test, grad_test, trial_x, direction)
    print(f"  点 {trial_x.numpy()}，负梯度方向，Armijo 步长 = {alpha:.4f}（理论最优 {1.0/L:.4f}）")
```

### 代码要点说明

1. **`optimizer.zero_grad()`**：PyTorch 默认累积梯度，每步训练前必须清零，否则梯度会叠加
2. **`loss.backward()`**：自动微分计算所有参数的梯度 $\partial \mathcal{L}/\partial \boldsymbol{\theta}$
3. **`optimizer.step()`**：执行参数更新 $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}$
4. **学习率的敏感性**：从实验可见，$\alpha$ 过小收敛慢，$\alpha$ 过大发散，选择合适学习率是深度学习实践的核心技巧
5. **Armijo 线搜索**：自动适应局部曲率，无需手动调参，但在深度学习中因计算成本高而较少使用

---

## 练习题

### 练习 5.1（基础）

设 $f(x) = \frac{1}{2}x^2 + 2x + 3$，从 $x_0 = 5$ 出发，取固定步长 $\alpha = 0.5$：

(a) 手动计算前3步梯度下降的迭代值 $x_1, x_2, x_3$。
(b) 求 $f$ 的最优解 $x^*$ 和最优值 $f^*$。
(c) 计算每步的函数值 $f(x_0), f(x_1), f(x_2), f(x_3)$，观察收敛趋势。

---

### 练习 5.2（基础）

对函数 $f(\mathbf{x}) = \frac{1}{2}(x_1^2 + 25x_2^2)$：

(a) 计算梯度 $\nabla f(\mathbf{x})$。
(b) 求梯度 Lipschitz 常数 $L$ 和强凸参数 $\mu$，以及条件数 $\kappa$。
(c) 从 $(1, 1)^\top$ 出发，用步长 $\alpha = 1/L$，计算一步更新后的点及函数值。

---

### 练习 5.3（中级）

证明：若 $f$ 是 $L$-光滑函数（梯度满足 Lipschitz 条件），则对所有 $\mathbf{x}, \mathbf{y}$：

$$f(\mathbf{y}) \leq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle + \frac{L}{2}\|\mathbf{y} - \mathbf{x}\|^2$$

（提示：利用 $f(\mathbf{y}) - f(\mathbf{x}) = \int_0^1 \langle \nabla f(\mathbf{x} + t(\mathbf{y}-\mathbf{x})), \mathbf{y} - \mathbf{x} \rangle \, dt$，再利用 Cauchy-Schwarz 不等式）

---

### 练习 5.4（中级）

考虑逻辑回归损失函数：

$$f(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \log(1 + e^{-y_i \mathbf{x}_i^\top \mathbf{w}})$$

(a) 推导梯度 $\nabla f(\mathbf{w})$ 的表达式（提示：设 $\sigma(z) = 1/(1+e^{-z})$ 为 sigmoid 函数）。
(b) 假设数据满足 $\|\mathbf{x}_i\| \leq R$，证明 $f$ 是 $L$-光滑的，并求 $L$ 的上界（提示：分析 Hessian 最大特征值）。
(c) 若使用 L2 正则化 $f_\lambda(\mathbf{w}) = f(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|^2$，为何 $f_\lambda$ 是强凸的？强凸参数 $\mu$ 是多少？

---

### 练习 5.5（提高）

**实验题**：实现并比较梯度下降的不同步长策略。

在 Python 中，对函数 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{A} \mathbf{x} - \mathbf{b}^\top \mathbf{x}$，其中 $\mathbf{A} = \text{diag}(1, 5, 10, 20)$（四维），$\mathbf{b} = \mathbf{1}$：

(a) 实现三种步长策略：固定步长（$\alpha = 1/L$）、精确线搜索（推导解析公式）、Armijo 回溯（参数 $c = 10^{-4}$，$\beta = 0.5$）。
(b) 从 $\mathbf{x}_0 = (2, 2, 2, 2)^\top$ 出发，运行 100 步，绘制三种策略的 $\|f(\mathbf{x}_k) - f^*\|$（对数纵轴）随迭代次数的曲线。
(c) 分析三种策略在每步计算成本和收敛速度上的权衡。

---

## 练习答案

### 答案 5.1

$f(x) = \frac{1}{2}x^2 + 2x + 3$，$\nabla f(x) = x + 2$。

**(a) 前3步迭代**：

$$x_1 = x_0 - \alpha(x_0 + 2) = 5 - 0.5 \times 7 = 1.5$$

$$x_2 = x_1 - \alpha(x_1 + 2) = 1.5 - 0.5 \times 3.5 = -0.25$$

$$x_3 = x_2 - \alpha(x_2 + 2) = -0.25 - 0.5 \times 1.75 = -1.125$$

**(b) 最优解**：令 $f'(x) = x + 2 = 0$，得 $x^* = -2$，$f^* = \frac{1}{2}(-2)^2 + 2(-2) + 3 = 1$。

**(c) 函数值**：

| $k$ | $x_k$ | $f(x_k)$ | $f(x_k) - f^*$ |
|-----|--------|----------|----------------|
| 0 | 5 | 21 | 20 |
| 1 | 1.5 | 7.625 | 6.625 |
| 2 | -0.25 | 2.53125 | 1.53125 |
| 3 | -1.125 | 1.382... | 0.382... |

函数值单调递减，向 $f^* = 1$ 收敛。

---

### 答案 5.2

$f(\mathbf{x}) = \frac{1}{2}(x_1^2 + 25x_2^2)$。

**(a)** $\nabla f(\mathbf{x}) = (x_1,\ 25x_2)^\top$。

**(b)** Hessian 矩阵 $\nabla^2 f = \text{diag}(1, 25)$。特征值为 1 和 25，因此：

$$\mu = \lambda_{\min} = 1, \quad L = \lambda_{\max} = 25, \quad \kappa = L/\mu = 25$$

**(c)** 从 $(1, 1)^\top$，步长 $\alpha = 1/25$：

$$\mathbf{x}_1 = \begin{pmatrix}1 \\ 1\end{pmatrix} - \frac{1}{25}\begin{pmatrix}1 \\ 25\end{pmatrix} = \begin{pmatrix}0.96 \\ 0\end{pmatrix}$$

函数值 $f(\mathbf{x}_1) = \frac{1}{2}(0.96^2 + 0) = 0.4608$（从初始 $f(\mathbf{x}_0) = 13$ 大幅下降）。

---

### 答案 5.3

**证明**：

由 $\mathbf{y} - \mathbf{x}$ 的积分路径展开：

$$f(\mathbf{y}) - f(\mathbf{x}) = \int_0^1 \langle \nabla f(\mathbf{x} + t(\mathbf{y}-\mathbf{x})), \mathbf{y} - \mathbf{x} \rangle \, dt$$

加减 $\langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$：

$$f(\mathbf{y}) - f(\mathbf{x}) - \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle = \int_0^1 \langle \nabla f(\mathbf{x} + t(\mathbf{y}-\mathbf{x})) - \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle \, dt$$

取绝对值，对右侧用 Cauchy-Schwarz 不等式：

$$\left|\cdot\right| \leq \int_0^1 \|\nabla f(\mathbf{x} + t(\mathbf{y}-\mathbf{x})) - \nabla f(\mathbf{x})\| \cdot \|\mathbf{y} - \mathbf{x}\| \, dt$$

再用 $L$-Lipschitz 条件（$\|\nabla f(\mathbf{x} + t(\mathbf{y}-\mathbf{x})) - \nabla f(\mathbf{x})\| \leq Lt\|\mathbf{y}-\mathbf{x}\|$）：

$$\leq \int_0^1 Lt \|\mathbf{y}-\mathbf{x}\|^2 \, dt = \frac{L}{2}\|\mathbf{y}-\mathbf{x}\|^2$$

因此：

$$f(\mathbf{y}) \leq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle + \frac{L}{2}\|\mathbf{y} - \mathbf{x}\|^2 \qquad \square$$

---

### 答案 5.4

**(a)** 设 $\sigma(z) = 1/(1+e^{-z})$，则 $\sigma'(z) = \sigma(z)(1-\sigma(z))$。

$$\nabla f(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^n y_i \sigma(-y_i \mathbf{x}_i^\top \mathbf{w}) \cdot \mathbf{x}_i = \frac{1}{n}\sum_{i=1}^n (\sigma(\mathbf{x}_i^\top \mathbf{w}) - \mathbb{1}[y_i = 1]) \mathbf{x}_i$$

即预测概率与真实标签之差的加权平均。

**(b)** Hessian 为：

$$\nabla^2 f(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n \sigma(y_i \mathbf{x}_i^\top \mathbf{w})(1 - \sigma(y_i \mathbf{x}_i^\top \mathbf{w})) \mathbf{x}_i \mathbf{x}_i^\top$$

由于 $\sigma(z)(1-\sigma(z)) \leq 1/4$（在 $z=0$ 处取到），因此：

$$\nabla^2 f(\mathbf{w}) \preceq \frac{1}{4n}\sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^\top \preceq \frac{R^2}{4} \mathbf{I}$$

所以 $L \leq R^2/4$。

**(c)** $f_\lambda(\mathbf{w}) = f(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|^2$ 的 Hessian 为：

$$\nabla^2 f_\lambda(\mathbf{w}) = \nabla^2 f(\mathbf{w}) + \lambda \mathbf{I} \succeq \lambda \mathbf{I}$$

因此 $f_\lambda$ 是 $\mu = \lambda$ 的强凸函数（正则化项强制了最小曲率为 $\lambda$）。

---

### 答案 5.5

**关键分析结论**（代码实现参考5.2.3节的 `armijo_line_search` 函数）：

对二次函数 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{A} \mathbf{x} - \mathbf{b}^\top \mathbf{x}$，$\mathbf{A} = \text{diag}(1, 5, 10, 20)$：

- $L = 20$（最大特征值），$\mu = 1$（最小特征值），$\kappa = 20$
- **固定步长** $\alpha = 1/20$：最简单，收敛率 $(1-1/20)^k = 0.95^k$，100步后误差约 $e^{-5} \approx 0.007$
- **精确线搜索**（二次函数解析公式 $\alpha_k = \|\mathbf{g}_k\|^2/(\mathbf{g}_k^\top \mathbf{A} \mathbf{g}_k)$）：每步计算量 $O(n)$，自适应步长，通常更快
- **Armijo 回溯**：步长在 $[1/L, 1]$ 之间自适应调整，性能介于两者之间

**(c) 权衡分析**：

| 策略 | 每步额外计算 | 收敛步数 | 实用性 |
|------|-------------|---------|--------|
| 固定步长 | 无 | 多 | 高（深度学习） |
| 精确线搜索 | $O(n)$（一次乘法） | 少 | 中（二次问题） |
| Armijo 回溯 | 若干次函数值计算 | 中 | 高（一般优化） |

在深度学习中，每次函数值计算需要一次完整的前向传播，成本极高，因此通常直接使用固定学习率（或学习率调度）而非线搜索。

---

*本章结束。下一章：[牛顿法与拟牛顿法](./06-newton-methods.md)*
