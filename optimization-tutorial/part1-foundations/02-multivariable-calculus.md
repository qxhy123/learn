# 第2章：多元微分学

> **前置知识**：单变量微积分、线性代数基础（向量、矩阵运算）
>
> **难度等级**：★★★☆☆
>
> **预计学习时间**：4–6 小时

---

## 学习目标

完成本章学习后，你将能够：

1. **理解偏导数的定义**，掌握多元函数对单个变量求导的方法，并能计算常见函数的偏导数。
2. **掌握梯度的几何意义**，理解梯度方向是函数增长最快的方向，并能计算方向导数。
3. **构造并分析 Hessian 矩阵**，利用其正定性判断驻点的类型（极小值、极大值、鞍点）。
4. **理解 Jacobian 矩阵的意义**，掌握向量值函数的链式法则矩阵形式，为反向传播算法奠定基础。
5. **应用多元 Taylor 展开**，构造函数的二次近似，理解优化算法中的一阶和二阶近似思想。

---

## 2.1 多元函数与偏导数

### 2.1.1 多元函数的基本概念

设 $\mathbf{x} = (x_1, x_2, \ldots, x_n)^\top \in \mathbb{R}^n$，**多元函数** $f: \mathbb{R}^n \to \mathbb{R}$ 将 $n$ 维向量映射为一个实数：

$$f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n)$$

**示例**：

- 二元二次函数：$f(x_1, x_2) = x_1^2 + 2x_1 x_2 + 3x_2^2$
- 神经网络损失函数：$L(\mathbf{w}) = \frac{1}{N}\sum_{i=1}^N \ell(f(\mathbf{x}_i; \mathbf{w}), y_i)$，其中 $\mathbf{w}$ 可能含有数百万个参数。

### 2.1.2 偏导数的定义

**偏导数**是对多元函数中某一个变量求导，保持其余变量不变。

$$\frac{\partial f}{\partial x_j}(\mathbf{x}) = \lim_{h \to 0} \frac{f(x_1, \ldots, x_j + h, \ldots, x_n) - f(\mathbf{x})}{h}$$

**计算规则**：对 $x_j$ 求偏导时，将其余变量视为常数，按单变量微分规则进行。

**示例**：设 $f(x_1, x_2) = x_1^2 x_2 + e^{x_1 x_2}$，则：

$$\frac{\partial f}{\partial x_1} = 2x_1 x_2 + x_2 e^{x_1 x_2}$$

$$\frac{\partial f}{\partial x_2} = x_1^2 + x_1 e^{x_1 x_2}$$

### 2.1.3 高阶偏导数

对偏导数继续求偏导，得到**高阶偏导数**：

$$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial}{\partial x_i}\left(\frac{\partial f}{\partial x_j}\right)$$

**Schwarz 定理**（混合偏导数的对称性）：若 $f$ 的二阶混合偏导数连续，则：

$$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$$

这意味着求偏导的顺序可以互换，这一性质对 Hessian 矩阵的对称性至关重要。

### 2.1.4 可微性

$f$ 在点 $\mathbf{x}_0$ **可微**，当且仅当存在线性映射（梯度向量）使得：

$$f(\mathbf{x}_0 + \Delta\mathbf{x}) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top \Delta\mathbf{x} + o(\|\Delta\mathbf{x}\|)$$

其中 $o(\|\Delta\mathbf{x}\|)$ 表示比 $\|\Delta\mathbf{x}\|$ 更高阶的无穷小量。偏导数存在并不保证可微，但若所有偏导数连续，则函数可微。

---

## 2.2 梯度与方向导数

### 2.2.1 梯度的定义

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 在点 $\mathbf{x}$ 处的**梯度**是由所有偏导数构成的列向量：

$$\nabla f(\mathbf{x}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix} \in \mathbb{R}^n$$

梯度算子 $\nabla$（nabla）将标量函数映射为向量场。

**示例**：设 $f(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x} + \mathbf{b}^\top \mathbf{x} + c$（二次函数），其中 $A$ 对称，则：

$$\nabla f(\mathbf{x}) = 2A\mathbf{x} + \mathbf{b}$$

特别地，若 $f(\mathbf{x}) = \|\mathbf{x}\|^2 = \mathbf{x}^\top \mathbf{x}$，则 $\nabla f(\mathbf{x}) = 2\mathbf{x}$。

### 2.2.2 梯度的几何意义

梯度具有以下重要几何性质：

1. **梯度方向是函数值增长最快的方向**：在点 $\mathbf{x}$ 处，沿 $\nabla f(\mathbf{x})$ 方向移动一小步，函数值增加最快。
2. **梯度的模是最大方向导数**：$\|\nabla f(\mathbf{x})\|$ 等于函数在点 $\mathbf{x}$ 处的最大方向导数。
3. **梯度垂直于等值面**：$\nabla f(\mathbf{x})$ 垂直于过点 $\mathbf{x}$ 的等值面（等高线）。

```
        ↑ ∇f(x)（梯度方向）
  f=3   |
  f=2   |    · x₀
  f=1   |
        +----------→
        等值线（水平集）
```

### 2.2.3 方向导数

函数 $f$ 沿单位向量 $\mathbf{u}$（$\|\mathbf{u}\| = 1$）方向的**方向导数**为：

$$D_{\mathbf{u}} f(\mathbf{x}) = \lim_{t \to 0} \frac{f(\mathbf{x} + t\mathbf{u}) - f(\mathbf{x})}{t} = \nabla f(\mathbf{x})^\top \mathbf{u}$$

由 Cauchy-Schwarz 不等式：

$$D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x})^\top \mathbf{u} \leq \|\nabla f(\mathbf{x})\| \cdot \|\mathbf{u}\| = \|\nabla f(\mathbf{x})\|$$

等号在 $\mathbf{u} = \frac{\nabla f(\mathbf{x})}{\|\nabla f(\mathbf{x})\|}$ 时成立，即**梯度方向是函数增加最快的方向**。

相应地，**负梯度方向** $-\frac{\nabla f(\mathbf{x})}{\|\nabla f(\mathbf{x})\|}$ 是函数减少最快的方向，这正是**梯度下降法**的理论基础。

### 2.2.4 常用函数的梯度

| 函数 $f(\mathbf{x})$ | 梯度 $\nabla f(\mathbf{x})$ | 备注 |
|:---:|:---:|:---:|
| $\mathbf{a}^\top \mathbf{x}$ | $\mathbf{a}$ | 线性函数 |
| $\mathbf{x}^\top A \mathbf{x}$（$A$ 对称） | $2A\mathbf{x}$ | 二次型 |
| $\|\mathbf{x}\|^2$ | $2\mathbf{x}$ | $A = I$ 的特殊情况 |
| $\|\mathbf{x} - \mathbf{a}\|^2$ | $2(\mathbf{x} - \mathbf{a})$ | 最小二乘 |
| $\log(\mathbf{a}^\top \mathbf{x})$ | $\frac{\mathbf{a}}{\mathbf{a}^\top \mathbf{x}}$ | 对数函数 |

---

## 2.3 Hessian 矩阵

### 2.3.1 Hessian 矩阵的定义

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 在点 $\mathbf{x}$ 处的 **Hessian 矩阵**（海森矩阵）是由所有二阶偏导数构成的 $n \times n$ 矩阵：

$$H_f(\mathbf{x}) = \nabla^2 f(\mathbf{x}) = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

由 Schwarz 定理，若二阶偏导数连续，则 $H_f(\mathbf{x})$ 是**对称矩阵**：$H_f = H_f^\top$。

### 2.3.2 Hessian 矩阵与曲率

Hessian 矩阵描述了函数的**二阶局部几何性质**（曲率）。

对于方向 $\mathbf{v}$（$\|\mathbf{v}\| = 1$），函数在该方向的**曲率**为：

$$\kappa_{\mathbf{v}} = \mathbf{v}^\top H_f(\mathbf{x}) \mathbf{v}$$

- 若 $\kappa_{\mathbf{v}} > 0$：函数在 $\mathbf{v}$ 方向上凸（碗形）
- 若 $\kappa_{\mathbf{v}} < 0$：函数在 $\mathbf{v}$ 方向上凹（山峰形）
- 若 $\kappa_{\mathbf{v}} = 0$：函数在 $\mathbf{v}$ 方向上线性

**示例**：$f(x_1, x_2) = x_1^2 + 4x_1 x_2 + 3x_2^2$

$$H_f = \begin{pmatrix} 2 & 4 \\ 4 & 6 \end{pmatrix}$$

特征值为 $\lambda_1, \lambda_2$（满足 $\det(H - \lambda I) = 0$），描述各主方向的曲率。

### 2.3.3 正定性与驻点分类

设 $\mathbf{x}^*$ 是驻点（即 $\nabla f(\mathbf{x}^*) = \mathbf{0}$），根据 Hessian 矩阵的正定性判断驻点类型：

| $H_f(\mathbf{x}^*)$ 的性质 | 特征值 | 驻点类型 |
|:---:|:---:|:---:|
| 正定（$\mathbf{v}^\top H \mathbf{v} > 0, \forall \mathbf{v} \neq \mathbf{0}$） | 全部 $> 0$ | 局部极小值 |
| 负定（$\mathbf{v}^\top H \mathbf{v} < 0, \forall \mathbf{v} \neq \mathbf{0}$） | 全部 $< 0$ | 局部极大值 |
| 不定（既有正也有负特征值） | 正负混杂 | 鞍点 |
| 半正/负定 | 有零特征值 | 需高阶分析 |

**正定矩阵的判断（Sylvester 准则）**：所有顺序主子式大于零。

$$H = \begin{pmatrix} a & b \\ b & c \end{pmatrix} \text{ 正定} \iff a > 0 \text{ 且 } ac - b^2 > 0$$

### 2.3.4 条件数与优化难度

Hessian 矩阵的**条件数** $\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}}$ 衡量函数曲率的各向异性程度：

- $\kappa \approx 1$：各方向曲率相近，等值线接近圆形，梯度下降收敛快。
- $\kappa \gg 1$：函数呈"峡谷"形，等值线扁平，梯度下降收敛慢（振荡问题）。

这正是深度学习中使用 **Adam** 等自适应优化器而非纯梯度下降的主要原因之一。

---

## 2.4 Jacobian 矩阵

### 2.4.1 向量值函数与 Jacobian 矩阵

对于向量值函数 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$：

$$\mathbf{f}(\mathbf{x}) = \begin{pmatrix} f_1(x_1, \ldots, x_n) \\ f_2(x_1, \ldots, x_n) \\ \vdots \\ f_m(x_1, \ldots, x_n) \end{pmatrix}$$

其 **Jacobian 矩阵**（雅可比矩阵）是 $m \times n$ 矩阵：

$$J_\mathbf{f}(\mathbf{x}) = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}$$

注意：梯度是标量函数 Jacobian 的转置，即 $\nabla f(\mathbf{x}) = J_f(\mathbf{x})^\top$（当 $m=1$ 时）。

### 2.4.2 Jacobian 矩阵的几何意义

Jacobian 矩阵是函数 $\mathbf{f}$ 在 $\mathbf{x}$ 处的**最佳线性近似（微分）**：

$$\mathbf{f}(\mathbf{x} + \Delta\mathbf{x}) \approx \mathbf{f}(\mathbf{x}) + J_\mathbf{f}(\mathbf{x}) \Delta\mathbf{x}$$

$|{\det(J_\mathbf{f})}|$ 表示函数在 $\mathbf{x}$ 附近的**局部体积缩放比**。

### 2.4.3 链式法则的矩阵形式

设 $\mathbf{h} = \mathbf{g} \circ \mathbf{f}$，即 $\mathbf{h}(\mathbf{x}) = \mathbf{g}(\mathbf{f}(\mathbf{x}))$，其中：

- $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^k$，Jacobian 为 $J_\mathbf{f} \in \mathbb{R}^{k \times n}$
- $\mathbf{g}: \mathbb{R}^k \to \mathbb{R}^m$，Jacobian 为 $J_\mathbf{g} \in \mathbb{R}^{m \times k}$

则复合函数的 Jacobian 为矩阵乘积：

$$J_\mathbf{h}(\mathbf{x}) = J_\mathbf{g}(\mathbf{f}(\mathbf{x})) \cdot J_\mathbf{f}(\mathbf{x}) \in \mathbb{R}^{m \times n}$$

对于标量损失函数 $L: \mathbb{R}^n \to \mathbb{R}$ 通过中间层 $\mathbf{z} = \mathbf{f}(\mathbf{x})$，梯度链式法则为：

$$\frac{\partial L}{\partial \mathbf{x}} = J_\mathbf{f}(\mathbf{x})^\top \frac{\partial L}{\partial \mathbf{z}}$$

这正是**反向传播算法**的核心数学结构。

### 2.4.4 神经网络层的 Jacobian

对于全连接层 $\mathbf{z} = W\mathbf{x} + \mathbf{b}$：

$$J_\mathbf{z} = \frac{\partial \mathbf{z}}{\partial \mathbf{x}} = W \in \mathbb{R}^{m \times n}$$

$$\frac{\partial \mathbf{z}}{\partial W} = \mathbf{x}^\top \otimes I_m \quad \text{（Kronecker 积形式）}$$

在实际反向传播中，更常用的形式是：$\frac{\partial L}{\partial W} = \boldsymbol{\delta} \mathbf{x}^\top$，其中 $\boldsymbol{\delta} = \frac{\partial L}{\partial \mathbf{z}}$。

---

## 2.5 Taylor 展开与二次近似

### 2.5.1 单变量 Taylor 展开回顾

单变量函数 $g: \mathbb{R} \to \mathbb{R}$ 在 $x_0$ 处的 Taylor 展开：

$$g(x_0 + h) = g(x_0) + g'(x_0)h + \frac{1}{2}g''(x_0)h^2 + \frac{1}{6}g'''(x_0)h^3 + \cdots$$

### 2.5.2 多元 Taylor 展开

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 在点 $\mathbf{x}_0$ 附近的 Taylor 展开（设 $\Delta\mathbf{x} = \mathbf{x} - \mathbf{x}_0$）：

$$f(\mathbf{x}_0 + \Delta\mathbf{x}) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top \Delta\mathbf{x} + \frac{1}{2} \Delta\mathbf{x}^\top H_f(\mathbf{x}_0) \Delta\mathbf{x} + O(\|\Delta\mathbf{x}\|^3)$$

各项含义：
- **零阶项** $f(\mathbf{x}_0)$：当前函数值
- **一阶项** $\nabla f(\mathbf{x}_0)^\top \Delta\mathbf{x}$：线性近似（梯度决定）
- **二阶项** $\frac{1}{2}\Delta\mathbf{x}^\top H_f(\mathbf{x}_0) \Delta\mathbf{x}$：曲率修正（Hessian 决定）

### 2.5.3 一阶近似与梯度下降

**一阶 Taylor 近似**（忽略二阶及更高阶项）：

$$f(\mathbf{x}_0 + \Delta\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top \Delta\mathbf{x}$$

梯度下降算法的更新规则 $\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)$ 正是基于一阶近似，在**负梯度方向**上移动步长 $\alpha$。

### 2.5.4 二阶近似与牛顿法

**二次近似**（保留到二阶项）是函数在 $\mathbf{x}_0$ 附近的抛物面拟合：

$$q(\Delta\mathbf{x}) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top \Delta\mathbf{x} + \frac{1}{2} \Delta\mathbf{x}^\top H_f(\mathbf{x}_0) \Delta\mathbf{x}$$

对 $q$ 关于 $\Delta\mathbf{x}$ 求最小值（令梯度为零）：

$$\nabla_{\Delta\mathbf{x}} q = \nabla f(\mathbf{x}_0) + H_f(\mathbf{x}_0) \Delta\mathbf{x} = \mathbf{0}$$

$$\Delta\mathbf{x}^* = -H_f(\mathbf{x}_0)^{-1} \nabla f(\mathbf{x}_0)$$

这就是**牛顿法**的更新规则：

$$\mathbf{x}_{k+1} = \mathbf{x}_k - H_f(\mathbf{x}_k)^{-1} \nabla f(\mathbf{x}_k)$$

牛顿法利用曲率信息，对于二次函数可以**一步收敛**，比梯度下降收敛快得多。

### 2.5.5 一阶与二阶方法的对比

| 性质 | 梯度下降（一阶） | 牛顿法（二阶） |
|:---:|:---:|:---:|
| 每步计算量 | $O(n)$ | $O(n^3)$（需矩阵求逆） |
| 收敛速度 | 线性收敛 | 二次收敛（超线性） |
| 内存需求 | $O(n)$ | $O(n^2)$（存储 Hessian） |
| 适用规模 | 大规模 $n$（深度学习） | 小/中规模 $n$ |
| 病态问题 | 收敛慢 | 对条件数不敏感 |

---

## 本章小结

| 概念 | 数学定义 | 维度 | 核心作用 |
|:---:|:---:|:---:|:---:|
| 偏导数 $\frac{\partial f}{\partial x_j}$ | 固定其余变量，对 $x_j$ 求导 | 标量 | 函数对单一变量的敏感度 |
| 梯度 $\nabla f(\mathbf{x})$ | $\left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)^\top$ | $\mathbb{R}^n$ 向量 | 函数增加最快的方向 |
| 方向导数 $D_\mathbf{u}f$ | $\nabla f^\top \mathbf{u}$ | 标量 | 沿任意方向的变化率 |
| Hessian $H_f$ | 二阶偏导数构成的对称矩阵 | $\mathbb{R}^{n\times n}$ 矩阵 | 曲率信息，判断驻点类型 |
| Jacobian $J_\mathbf{f}$ | $\frac{\partial \mathbf{f}}{\partial \mathbf{x}}$，一阶偏导数矩阵 | $\mathbb{R}^{m\times n}$ 矩阵 | 向量函数的线性近似，链式法则 |
| Taylor 展开（一阶） | $f(\mathbf{x}_0) + \nabla f^\top \Delta\mathbf{x}$ | — | 梯度下降的理论基础 |
| Taylor 展开（二阶） | 一阶 $+ \frac{1}{2}\Delta\mathbf{x}^\top H \Delta\mathbf{x}$ | — | 牛顿法的理论基础 |

**核心结论**：

- 梯度指向最速上升方向，其负方向是梯度下降的依据。
- Hessian 描述曲率，正定性决定驻点类型，条件数影响优化难度。
- Jacobian 是链式法则的矩阵形式，是反向传播算法的数学骨架。
- Taylor 展开将非线性函数局部线性化/二次化，是几乎所有优化算法的理论出发点。

---

## 深度学习应用：反向传播与自动微分

### 应用背景

深度神经网络的训练本质是求解一个高维优化问题：

$$\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) = \frac{1}{N}\sum_{i=1}^N \ell(f(\mathbf{x}_i; \boldsymbol{\theta}), y_i)$$

其中参数 $\boldsymbol{\theta}$ 可能有数十亿维。计算 $\nabla_{\boldsymbol{\theta}} L$ 需要高效的算法——这就是**反向传播（Backpropagation）**。

### 计算图与前向传播

**计算图**将神经网络表示为有向无环图（DAG），节点是操作，边是数据流。

以一个简单两层网络为例：

$$\mathbf{z}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}$$

$$\mathbf{a}^{(1)} = \sigma(\mathbf{z}^{(1)})$$

$$\mathbf{z}^{(2)} = W^{(2)}\mathbf{a}^{(1)} + \mathbf{b}^{(2)}$$

$$L = \ell(\mathbf{z}^{(2)}, \mathbf{y})$$

**前向传播**：从输入 $\mathbf{x}$ 出发，按计算图顺序计算各中间变量直至损失 $L$。

### 反向传播：链式法则的系统应用

**反向传播**从损失 $L$ 出发，逆向应用链式法则计算各参数的梯度：

$$\frac{\partial L}{\partial W^{(2)}} = \frac{\partial L}{\partial \mathbf{z}^{(2)}} \cdot \mathbf{a}^{(1)\top}$$

$$\frac{\partial L}{\partial W^{(1)}} = \left(W^{(2)\top} \frac{\partial L}{\partial \mathbf{z}^{(2)}}\right) \odot \sigma'(\mathbf{z}^{(1)}) \cdot \mathbf{x}^\top$$

其中 $\odot$ 表示逐元素乘法（Hadamard 积）。

关键：每一层的梯度都可以用**上层传来的梯度**乘以**本层的 Jacobian** 得到，这使得反向传播的计算复杂度与前向传播相同，均为 $O(|\text{参数数量}|)$。

### PyTorch 自动微分实例

#### 实例 1：标量函数梯度计算

```python
import torch

# 创建需要梯度的张量
x = torch.tensor([2.0, 3.0], requires_grad=True)

# 定义函数 f(x) = x1^2 + 2*x1*x2 + x2^2
f = x[0]**2 + 2*x[0]*x[1] + x[1]**2

# 反向传播：自动计算梯度
f.backward()

# 查看梯度 ∇f = [2x1 + 2x2, 2x1 + 2x2] = [10, 10]
print(f"f(x) = {f.item():.2f}")
print(f"∇f(x) = {x.grad}")
# 输出：∇f(x) = tensor([10., 10.])

# 手动验证：∂f/∂x1 = 2*2 + 2*3 = 10，∂f/∂x2 = 2*2 + 2*3 = 10
```

#### 实例 2：计算 Hessian 矩阵

```python
import torch

def compute_hessian(f_func, x):
    """计算标量函数在 x 处的 Hessian 矩阵"""
    n = x.shape[0]
    hessian = torch.zeros(n, n)

    # 计算梯度
    x = x.requires_grad_(True)
    f = f_func(x)
    grad = torch.autograd.grad(f, x, create_graph=True)[0]

    # 对每个梯度分量再求导
    for i in range(n):
        grad2 = torch.autograd.grad(grad[i], x, retain_graph=True)[0]
        hessian[i] = grad2.detach()

    return hessian

# 定义二次函数 f(x) = x1^2 + 4*x1*x2 + 3*x2^2
def f(x):
    return x[0]**2 + 4*x[0]*x[1] + 3*x[1]**2

x0 = torch.tensor([1.0, 1.0])
H = compute_hessian(f, x0)
print("Hessian 矩阵：")
print(H)
# 输出：
# tensor([[2., 4.],
#         [4., 6.]])

# 检查正定性（所有特征值 > 0?）
eigenvalues = torch.linalg.eigvalsh(H)
print(f"特征值：{eigenvalues}")
print(f"矩阵{'正定' if (eigenvalues > 0).all() else '非正定'}")
```

#### 实例 3：两层神经网络的前向传播与反向传播

```python
import torch
import torch.nn as nn

# 设定随机种子
torch.manual_seed(42)

# 网络参数（手动定义以展示计算过程）
n_in, n_hidden, n_out = 4, 8, 2
W1 = torch.randn(n_hidden, n_in,  requires_grad=True)
b1 = torch.zeros(n_hidden, requires_grad=True)
W2 = torch.randn(n_out,    n_hidden, requires_grad=True)
b2 = torch.zeros(n_out,    requires_grad=True)

# 激活函数
def relu(z):
    return torch.clamp(z, min=0)

# 前向传播
x = torch.randn(4)          # 输入
y = torch.tensor([1.0, 0.0]) # 目标

z1 = W1 @ x + b1            # 线性变换：z1 = W1·x + b1
a1 = relu(z1)                # 激活：a1 = ReLU(z1)
z2 = W2 @ a1 + b2           # 线性变换：z2 = W2·a1 + b2
loss = 0.5 * ((z2 - y)**2).sum()  # 均方误差损失

print(f"损失值：{loss.item():.4f}")

# 反向传播（自动微分）
loss.backward()

print(f"\n∂L/∂W1 形状：{W1.grad.shape}")
print(f"∂L/∂W2 形状：{W2.grad.shape}")
print(f"∂L/∂b1 形状：{b1.grad.shape}")
print(f"∂L/∂b2 形状：{b2.grad.shape}")

# ====== 手动验证反向传播 ======
with torch.no_grad():
    # 重新计算中间值
    z1_v = W1 @ x + b1
    a1_v = relu(z1_v)
    z2_v = W2 @ a1_v + b2

    # 从损失开始反向传播
    dL_dz2 = z2_v - y                          # ∂L/∂z2
    dL_dW2 = torch.outer(dL_dz2, a1_v)         # ∂L/∂W2 = δ2 · a1ᵀ
    dL_db2 = dL_dz2                             # ∂L/∂b2

    dL_da1 = W2.T @ dL_dz2                      # ∂L/∂a1 = W2ᵀ · δ2
    dL_dz1 = dL_da1 * (z1_v > 0).float()       # ReLU 导数（次梯度）
    dL_dW1 = torch.outer(dL_dz1, x)            # ∂L/∂W1 = δ1 · xᵀ
    dL_db1 = dL_dz1                             # ∂L/∂b1

# 对比自动微分和手动计算的结果
print(f"\n∂L/∂W2 最大误差（自动 vs 手动）：{(W2.grad - dL_dW2).abs().max().item():.2e}")
print(f"∂L/∂W1 最大误差（自动 vs 手动）：{(W1.grad - dL_dW1).abs().max().item():.2e}")
```

#### 实例 4：torch.autograd.grad 与高阶导数

```python
import torch

# 创建输入
x = torch.tensor(2.0, requires_grad=True)

# 定义函数 f(x) = sin(x^2)
f = torch.sin(x**2)

# 一阶导数：f'(x) = 2x·cos(x^2)
grad1 = torch.autograd.grad(f, x, create_graph=True)[0]
print(f"f'(2) = {grad1.item():.4f}")
print(f"理论值：{(2*2*torch.cos(torch.tensor(4.0))).item():.4f}")

# 二阶导数：f''(x) = 2·cos(x^2) - 4x²·sin(x^2)
grad2 = torch.autograd.grad(grad1, x)[0]
print(f"f''(2) = {grad2.item():.4f}")

# 对比数值微分（有限差分验证）
h = 1e-5
numerical = (torch.sin((x + h).detach()**2) - torch.sin((x - h).detach()**2)) / (2*h)
print(f"数值微分 f'(2) ≈ {numerical.item():.4f}")
```

### 自动微分的两种模式

| 模式 | 方向 | 计算量 | 适用场景 |
|:---:|:---:|:---:|:---:|
| **前向模式**（Forward AD） | 输入 → 输出 | $O(n)$ 次前向传播（$n$=输入维数） | $n \ll m$（输入少，输出多） |
| **反向模式**（Reverse AD / Backprop） | 输出 → 输入 | $O(m)$ 次反向传播（$m$=输出维数） | $m \ll n$（输出少如标量损失，输入多）|

深度学习中损失 $L$ 是标量（$m=1$），参数 $\boldsymbol{\theta}$ 维度极高（$n \gg 1$），因此**反向模式**（即反向传播）是唯一实用的选择。

PyTorch 的 `torch.autograd` 使用**反向模式自动微分**，在前向传播时记录计算图（tape），反向传播时沿计算图逆向应用链式法则。

---

## 练习题

### 练习 1（偏导数与梯度）

设 $f(x_1, x_2, x_3) = x_1^2 x_2 + e^{x_2 x_3} + \ln(x_1 + x_3)$。

(a) 求 $\frac{\partial f}{\partial x_1}$，$\frac{\partial f}{\partial x_2}$，$\frac{\partial f}{\partial x_3}$。

(b) 求在点 $(1, 0, 1)$ 处的梯度 $\nabla f(1, 0, 1)$。

(c) 求函数在该点沿方向 $\mathbf{u} = \frac{1}{\sqrt{3}}(1, 1, 1)^\top$ 的方向导数。

---

### 练习 2（Hessian 矩阵与驻点分析）

设 $f(x_1, x_2) = x_1^4 + x_2^4 - 4x_1 x_2$。

(a) 求所有驻点（满足 $\nabla f = \mathbf{0}$ 的点）。

(b) 对每个驻点，计算 Hessian 矩阵并判断其正定性。

(c) 分类各驻点（极小值、极大值或鞍点）。

---

### 练习 3（链式法则与 Jacobian）

设复合函数 $L = \|\mathbf{y} - W\mathbf{x}\|^2$，其中 $W \in \mathbb{R}^{m \times n}$，$\mathbf{x} \in \mathbb{R}^n$，$\mathbf{y} \in \mathbb{R}^m$。

(a) 设 $\mathbf{r} = \mathbf{y} - W\mathbf{x}$（残差向量），求 $\frac{\partial \mathbf{r}}{\partial \mathbf{x}}$（Jacobian 矩阵）。

(b) 利用链式法则求 $\frac{\partial L}{\partial \mathbf{x}}$。

(c) 令 $\frac{\partial L}{\partial \mathbf{x}} = \mathbf{0}$ 求最优解 $\mathbf{x}^*$（假设 $W$ 列满秩）。

---

### 练习 4（Taylor 展开与近似）

设 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top A \mathbf{x} - \mathbf{b}^\top \mathbf{x}$，其中 $A$ 是 $n \times n$ 正定矩阵。

(a) 计算 $\nabla f(\mathbf{x})$ 和 $H_f(\mathbf{x})$（Hessian 矩阵）。

(b) 写出 $f$ 在任意点 $\mathbf{x}_0$ 处的二阶 Taylor 展开，并与 $f$ 本身进行比较。

(c) 利用牛顿法更新规则 $\Delta\mathbf{x} = -H^{-1}\nabla f$，从任意初始点 $\mathbf{x}_0$ 出发，计算需要多少步牛顿迭代才能到达最小值点 $\mathbf{x}^* = A^{-1}\mathbf{b}$？

---

### 练习 5（PyTorch 实践）

```python
import torch

# 定义 Rosenbrock 函数（优化中的经典测试函数）
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

x = torch.tensor([-1.0, 1.0], requires_grad=True)
```

(a) 用 PyTorch 计算 $f(-1, 1)$ 的值和梯度 $\nabla f(-1, 1)$。

(b) 实现一个简单的梯度下降循环，步长 $\alpha = 0.001$，运行 1000 步，观察损失变化。

(c) 修改代码，使用 `torch.autograd.functional.hessian` 计算在起始点 $(-1, 1)$ 处的 Hessian 矩阵，并判断是否正定。

(d) 分析：Rosenbrock 函数的 Hessian 条件数是多少？这能解释为什么梯度下降在该函数上收敛慢吗？

---

## 练习答案

### 答案 1

**(a)** 分别对各变量求偏导（其余变量视为常数）：

$$\frac{\partial f}{\partial x_1} = 2x_1 x_2 + \frac{1}{x_1 + x_3}$$

$$\frac{\partial f}{\partial x_2} = x_1^2 + x_3 e^{x_2 x_3}$$

$$\frac{\partial f}{\partial x_3} = x_2 e^{x_2 x_3} + \frac{1}{x_1 + x_3}$$

**(b)** 代入 $(x_1, x_2, x_3) = (1, 0, 1)$：

$$\frac{\partial f}{\partial x_1}\bigg|_{(1,0,1)} = 2(1)(0) + \frac{1}{1+1} = \frac{1}{2}$$

$$\frac{\partial f}{\partial x_2}\bigg|_{(1,0,1)} = 1^2 + 1 \cdot e^0 = 1 + 1 = 2$$

$$\frac{\partial f}{\partial x_3}\bigg|_{(1,0,1)} = 0 \cdot e^0 + \frac{1}{1+1} = \frac{1}{2}$$

$$\nabla f(1,0,1) = \left(\frac{1}{2}, 2, \frac{1}{2}\right)^\top$$

**(c)** 方向导数：

$$D_\mathbf{u}f(1,0,1) = \nabla f(1,0,1)^\top \mathbf{u} = \left(\frac{1}{2}, 2, \frac{1}{2}\right) \cdot \frac{1}{\sqrt{3}}\begin{pmatrix}1\\1\\1\end{pmatrix} = \frac{1}{\sqrt{3}}\left(\frac{1}{2} + 2 + \frac{1}{2}\right) = \frac{3}{\sqrt{3}} = \sqrt{3}$$

---

### 答案 2

**(a)** 求驻点：令 $\nabla f = \mathbf{0}$：

$$\frac{\partial f}{\partial x_1} = 4x_1^3 - 4x_2 = 0 \implies x_2 = x_1^3$$

$$\frac{\partial f}{\partial x_2} = 4x_2^3 - 4x_1 = 0 \implies x_1 = x_2^3$$

代入：$x_1 = (x_1^3)^3 = x_1^9$，即 $x_1^9 - x_1 = 0$，$x_1(x_1^8 - 1) = 0$。

解得 $x_1 \in \{0, 1, -1\}$，对应 $(x_1, x_2) \in \{(0,0), (1,1), (-1,-1)\}$。

**(b)** Hessian 矩阵：

$$H_f = \begin{pmatrix} 12x_1^2 & -4 \\ -4 & 12x_2^2 \end{pmatrix}$$

- 在 $(0,0)$：$H = \begin{pmatrix} 0 & -4 \\ -4 & 0 \end{pmatrix}$，特征值 $\pm 4$，**不定矩阵**。
- 在 $(1,1)$：$H = \begin{pmatrix} 12 & -4 \\ -4 & 12 \end{pmatrix}$，$\det = 144 - 16 = 128 > 0$，迹 $= 24 > 0$，**正定**。
- 在 $(-1,-1)$：$H = \begin{pmatrix} 12 & -4 \\ -4 & 12 \end{pmatrix}$，与 $(1,1)$ 相同，**正定**。

**(c)** 分类：

- $(0,0)$：Hessian 不定 $\to$ **鞍点**
- $(1,1)$：Hessian 正定 $\to$ **局部极小值**，$f(1,1) = 1+1-4 = -2$
- $(-1,-1)$：Hessian 正定 $\to$ **局部极小值**，$f(-1,-1) = 1+1-4(-1)(-1) = -2$

两个极小值均为全局最小值（通过分析函数在无穷远处趋向 $+\infty$ 可知）。

---

### 答案 3

**(a)** $\mathbf{r} = \mathbf{y} - W\mathbf{x}$，对 $\mathbf{x}$ 求导：

$$\frac{\partial \mathbf{r}}{\partial \mathbf{x}} = -W \in \mathbb{R}^{m \times n}$$

**(b)** $L = \mathbf{r}^\top \mathbf{r} = \sum_i r_i^2$，用链式法则：

$$\frac{\partial L}{\partial \mathbf{x}} = \left(\frac{\partial \mathbf{r}}{\partial \mathbf{x}}\right)^\top \frac{\partial L}{\partial \mathbf{r}} = (-W)^\top (2\mathbf{r}) = -2W^\top(\mathbf{y} - W\mathbf{x}) = 2W^\top W\mathbf{x} - 2W^\top \mathbf{y}$$

**(c)** 令 $\frac{\partial L}{\partial \mathbf{x}} = \mathbf{0}$：

$$2W^\top W \mathbf{x}^* = 2W^\top \mathbf{y}$$

$$\boxed{\mathbf{x}^* = (W^\top W)^{-1} W^\top \mathbf{y}}$$

这正是**最小二乘法的正规方程（Normal Equations）**！当 $W$ 列满秩时，$W^\top W$ 可逆。

---

### 答案 4

**(a)** 梯度和 Hessian：

$$\nabla f(\mathbf{x}) = A\mathbf{x} - \mathbf{b}$$

$$H_f(\mathbf{x}) = A \quad \text{（常数，与 } \mathbf{x} \text{ 无关）}$$

**(b)** 在 $\mathbf{x}_0$ 处展开（$\Delta\mathbf{x} = \mathbf{x} - \mathbf{x}_0$）：

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + (A\mathbf{x}_0 - \mathbf{b})^\top \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^\top A \Delta\mathbf{x}$$

展开后可以验证这就等于 $f(\mathbf{x})$ 本身（因为 $f$ 本身就是二次函数，Taylor 展开精确！）。

**(c)** 牛顿法更新：

$$\Delta\mathbf{x} = -H^{-1}\nabla f(\mathbf{x}_0) = -A^{-1}(A\mathbf{x}_0 - \mathbf{b}) = \mathbf{x}_0 - A^{-1}\mathbf{b} \cdot (-1) = A^{-1}\mathbf{b} - \mathbf{x}_0$$

$$\mathbf{x}_1 = \mathbf{x}_0 + \Delta\mathbf{x} = \mathbf{x}_0 + (A^{-1}\mathbf{b} - \mathbf{x}_0) = A^{-1}\mathbf{b} = \mathbf{x}^*$$

**只需 1 步**即可到达精确最小值！这是因为 $f$ 是二次函数，牛顿法的二次近似精确，故一步收敛。

---

### 答案 5

```python
import torch

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# (a) 计算函数值和梯度
x = torch.tensor([-1.0, 1.0], requires_grad=True)
f = rosenbrock(x)
f.backward()
print(f"f(-1, 1) = {f.item()}")
print(f"∇f(-1, 1) = {x.grad}")
# f(-1, 1) = 4.0
# ∇f(-1, 1) = tensor([-2., 0.])
# 手动验证：∂f/∂x1 = -2(1-x1) - 400x1(x2-x1²) = -2(2) - 400(-1)(1-1) = -4+0 = -4?
# 重算：在(-1,1)：∂f/∂x1 = 2(x1-1) + 2·100·(x2-x1²)·(-2x1) = 2(-2) + 200·0·2 = -4
# 注：x2-x1²= 1-1=0，所以第二项为0，梯度第一分量=-4，第二分量=200·0=0

# (b) 梯度下降
x = torch.tensor([-1.0, 1.0], requires_grad=True)
alpha = 0.001
optimizer = torch.optim.SGD([x], lr=alpha)

losses = []
for step in range(1000):
    optimizer.zero_grad()
    loss = rosenbrock(x)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        losses.append(loss.item())
        print(f"Step {step:4d}: f = {loss.item():.4f}, x = {x.detach().numpy()}")

# (c) 计算 Hessian 矩阵
x0 = torch.tensor([-1.0, 1.0])
H = torch.autograd.functional.hessian(rosenbrock, x0)
print(f"\nHessian 矩阵：\n{H}")
eigenvalues = torch.linalg.eigvalsh(H)
print(f"特征值：{eigenvalues}")
is_pd = (eigenvalues > 0).all()
print(f"是否正定：{is_pd}")

# (d) 条件数分析
kappa = eigenvalues.max() / eigenvalues.min()
print(f"条件数 κ = {kappa.item():.1f}")
# 条件数极大（约 2500），说明函数等值面高度细长
# 梯度下降在细长"峡谷"中来回振荡，收敛极慢
# 这解释了为什么 Rosenbrock 函数是优化算法的经典挑战案例
```

**分析**：Rosenbrock 函数在 $(-1, 1)$ 处的 Hessian 条件数约为 2500，说明各方向曲率差异悬殊。梯度下降沿"峡谷"壁频繁振荡，只能缓慢接近谷底，需要数千甚至数万步才能收敛到全局最小值 $(1, 1)$。相比之下，牛顿法利用 Hessian 信息，可以在几十步内收敛。

---

*下一章预告*：**第3章 凸集与凸函数** ——我们将学习优化问题最重要的性质：凸性。理解凸函数与非凸函数的区别，以及全局最优解的存在性条件。
