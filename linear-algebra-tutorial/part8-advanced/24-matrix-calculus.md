# 第24章：矩阵微积分

> **前置知识**：第13章（线性映射）、第16章（内积与正交性）、第19章（特征值与特征向量）、第21章（对称矩阵与谱定理）
>
> **本章难度**：★★★★★
>
> **预计学习时间**：6-8 小时

---

## 学习目标

学完本章后，你将能够：

- 区分向量对向量、标量对向量、标量对矩阵三种求导布局，熟练使用分子布局（numerator layout）约定，理解 Jacobian 矩阵和梯度向量的几何含义
- 掌握 $\partial(\mathbf{A}\mathbf{x})/\partial\mathbf{x}$、$\partial(\mathbf{x}^T\mathbf{A}\mathbf{x})/\partial\mathbf{x}$、$\partial \operatorname{tr}(\mathbf{A}\mathbf{B})/\partial\mathbf{A}$ 等核心矩阵微分公式，并能用迹技巧（trace trick）推导复杂表达式的梯度
- 运用矩阵链式法则，将计算图中的梯度反向传播，从理论上完整推导多层神经网络的反向传播算法
- 理解自动微分（automatic differentiation）与矩阵链式法则的等价性，会用 PyTorch 验证手工推导的梯度公式
- 将本章所学融会贯通，把从第1章到第23章的线性代数知识统一到深度学习的梯度计算框架中

---

## 24.1 向量对向量的导数：Jacobian 矩阵

### 问题的提出

设 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$，即 $\mathbf{f}$ 将 $n$ 维向量映射为 $m$ 维向量：

$$\mathbf{f}(\mathbf{x}) = \begin{pmatrix} f_1(\mathbf{x}) \\ f_2(\mathbf{x}) \\ \vdots \\ f_m(\mathbf{x}) \end{pmatrix}, \quad \mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}$$

每个分量 $f_i$ 是 $n$ 个变量的标量函数。"$\mathbf{f}$ 关于 $\mathbf{x}$ 的导数"应该是什么？

### Jacobian 矩阵的定义

**定义**：映射 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ 在点 $\mathbf{x}_0$ 处的 **Jacobian 矩阵**（Jacobian matrix）是一个 $m \times n$ 矩阵，第 $(i,j)$ 元素为 $f_i$ 对 $x_j$ 的偏导数：

$$\boxed{\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{pmatrix} \dfrac{\partial f_1}{\partial x_1} & \dfrac{\partial f_1}{\partial x_2} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\[8pt] \dfrac{\partial f_2}{\partial x_1} & \dfrac{\partial f_2}{\partial x_2} & \cdots & \dfrac{\partial f_2}{\partial x_n} \\[4pt] \vdots & \vdots & \ddots & \vdots \\[4pt] \dfrac{\partial f_m}{\partial x_1} & \dfrac{\partial f_m}{\partial x_2} & \cdots & \dfrac{\partial f_m}{\partial x_n} \end{pmatrix} \in \mathbb{R}^{m \times n}}$$

这里采用**分子布局（numerator layout）**约定：结果矩阵的行由分子（$\mathbf{f}$）的维度决定，列由分母（$\mathbf{x}$）的维度决定。

> **注意**：矩阵微积分存在两种布局约定（分子布局和分母布局），不同教材可能不同，务必在阅读时先确认所用约定。本章统一使用**分子布局**。

### 几何意义

Jacobian 矩阵是多元函数在某点的**最佳线性近似**：

$$\mathbf{f}(\mathbf{x}_0 + \Delta\mathbf{x}) \approx \mathbf{f}(\mathbf{x}_0) + \mathbf{J}\,\Delta\mathbf{x}$$

当 $m = n$ 时，$|\det(\mathbf{J})|$ 表示函数 $\mathbf{f}$ 在该点的**局部体积缩放比例**——这正是多重积分换元公式中出现 Jacobian 行列式的原因。

### 典型例子

**例 1**：设 $\mathbf{f}(\mathbf{x}) = \mathbf{A}\mathbf{x}$，其中 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 是常数矩阵，则：

$$f_i(\mathbf{x}) = \sum_{k=1}^n a_{ik} x_k \implies \frac{\partial f_i}{\partial x_j} = a_{ij}$$

故：

$$\boxed{\frac{\partial (\mathbf{A}\mathbf{x})}{\partial \mathbf{x}} = \mathbf{A}}$$

线性映射的 Jacobian 就是变换矩阵本身——这与单变量微积分中"线性函数的导数等于斜率"完全类比。

**例 2**：设 $\mathbf{f}(\mathbf{x}) = \mathbf{x}$（恒等映射），则 $\partial \mathbf{x}/\partial \mathbf{x} = \mathbf{I}_n$。

**例 3**：设 $\mathbf{f}(\mathbf{x}) = \mathbf{x} \odot \mathbf{x}$（逐元素平方），$\mathbf{x} \in \mathbb{R}^n$：

$$f_i(\mathbf{x}) = x_i^2 \implies \frac{\partial f_i}{\partial x_j} = 2x_i \cdot \mathbf{1}_{[i=j]}$$

Jacobian 为对角矩阵 $\mathbf{J} = \operatorname{diag}(2x_1, 2x_2, \ldots, 2x_n) = 2\operatorname{diag}(\mathbf{x})$。

---

## 24.2 标量对向量的导数：梯度向量

### 梯度的定义

设 $f: \mathbb{R}^n \to \mathbb{R}$ 是标量函数（$m=1$ 的特例），其对 $\mathbf{x}$ 的导数是一个向量：

$$\boxed{\nabla_{\mathbf{x}} f = \frac{\partial f}{\partial \mathbf{x}} = \begin{pmatrix} \dfrac{\partial f}{\partial x_1} \\[8pt] \dfrac{\partial f}{\partial x_2} \\ \vdots \\ \dfrac{\partial f}{\partial x_n} \end{pmatrix} \in \mathbb{R}^n}$$

**布局约定说明**：在分子布局下，标量对列向量 $\mathbf{x}$ 的导数是**列向量**（与 $\mathbf{x}$ 同形）。这与 Jacobian 矩阵的定义一致：当 $m=1$ 时，$\mathbf{J} \in \mathbb{R}^{1 \times n}$ 是行向量，但梯度 $\nabla f$ 通常写成列向量（即 $\mathbf{J}^T$）。

### 几何意义

梯度 $\nabla f(\mathbf{x}_0)$ 指向 $f$ 在 $\mathbf{x}_0$ 处**增长最快的方向**，其模长等于该方向的变化率：

$$f(\mathbf{x}_0 + \epsilon \hat{\mathbf{v}}) \approx f(\mathbf{x}_0) + \epsilon \nabla f(\mathbf{x}_0)^T \hat{\mathbf{v}}$$

由 Cauchy-Schwarz 不等式，当 $\hat{\mathbf{v}} = \nabla f / \|\nabla f\|$ 时右端最大。负梯度 $-\nabla f$ 则指向**下降最陡的方向**，这正是梯度下降法的几何基础。

### 核心公式推导

**公式 1**：$f(\mathbf{x}) = \mathbf{a}^T \mathbf{x}$（线性函数）

$$f(\mathbf{x}) = \sum_{i=1}^n a_i x_i \implies \frac{\partial f}{\partial x_j} = a_j$$

$$\boxed{\frac{\partial (\mathbf{a}^T \mathbf{x})}{\partial \mathbf{x}} = \mathbf{a}}$$

**公式 2**：$f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x}$（二次型，$\mathbf{A}$ 对称）

展开：$f = \sum_{i,j} a_{ij} x_i x_j$。对 $x_k$ 求偏导：

$$\frac{\partial f}{\partial x_k} = \sum_j a_{kj} x_j + \sum_i a_{ik} x_i = (\mathbf{A}\mathbf{x})_k + (\mathbf{A}^T\mathbf{x})_k = 2(\mathbf{A}\mathbf{x})_k \quad (\text{当 } \mathbf{A} = \mathbf{A}^T)$$

$$\boxed{\frac{\partial (\mathbf{x}^T \mathbf{A} \mathbf{x})}{\partial \mathbf{x}} = 2\mathbf{A}\mathbf{x} \quad (\mathbf{A} \text{ 对称})}$$

**几何直觉**：二次型 $\mathbf{x}^T \mathbf{A} \mathbf{x}$ 在 $\mathbf{x}$ 处的梯度 $2\mathbf{A}\mathbf{x}$ 总是**径向向外**（在特征向量基下，每个方向的梯度分量 $= 2\lambda_i y_i$，正比于该方向的距离与曲率之积）。

**公式 3**：$f(\mathbf{x}) = \|\mathbf{x}\|^2 = \mathbf{x}^T\mathbf{x}$（取 $\mathbf{A} = \mathbf{I}$）

$$\boxed{\frac{\partial \|\mathbf{x}\|^2}{\partial \mathbf{x}} = 2\mathbf{x}}$$

**公式 4**：$f(\mathbf{x}) = \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$（最小二乘目标函数）

$$f = (\mathbf{A}\mathbf{x} - \mathbf{b})^T(\mathbf{A}\mathbf{x} - \mathbf{b}) = \mathbf{x}^T\mathbf{A}^T\mathbf{A}\mathbf{x} - 2\mathbf{b}^T\mathbf{A}\mathbf{x} + \mathbf{b}^T\mathbf{b}$$

$$\frac{\partial f}{\partial \mathbf{x}} = 2\mathbf{A}^T\mathbf{A}\mathbf{x} - 2\mathbf{A}^T\mathbf{b} = 2\mathbf{A}^T(\mathbf{A}\mathbf{x} - \mathbf{b})$$

令梯度为零，得**正规方程** $\mathbf{A}^T\mathbf{A}\mathbf{x} = \mathbf{A}^T\mathbf{b}$——最小二乘解的条件（详见第18章）。

---

## 24.3 标量对矩阵的导数：梯度矩阵

### 定义

设 $f: \mathbb{R}^{m \times n} \to \mathbb{R}$ 是以矩阵为自变量的标量函数，则 $f$ 对矩阵 $\mathbf{A}$ 的导数是同形矩阵：

$$\boxed{\frac{\partial f}{\partial \mathbf{A}} = \begin{pmatrix} \dfrac{\partial f}{\partial a_{11}} & \dfrac{\partial f}{\partial a_{12}} & \cdots & \dfrac{\partial f}{\partial a_{1n}} \\[8pt] \dfrac{\partial f}{\partial a_{21}} & \dfrac{\partial f}{\partial a_{22}} & \cdots & \dfrac{\partial f}{\partial a_{2n}} \\[4pt] \vdots & \vdots & \ddots & \vdots \\[4pt] \dfrac{\partial f}{\partial a_{m1}} & \dfrac{\partial f}{\partial a_{m2}} & \cdots & \dfrac{\partial f}{\partial a_{mn}} \end{pmatrix} \in \mathbb{R}^{m \times n}}$$

即结果矩阵的 $(i,j)$ 元素是 $\partial f / \partial a_{ij}$。

### 迹技巧（Trace Trick）

推导矩阵梯度最强大的工具是**迹技巧**。核心思想：任何标量都等于自身的迹，故 $f = \operatorname{tr}(f)$，然后利用迹的性质化简。

**迹的关键性质**：

1. **线性性**：$\operatorname{tr}(\alpha \mathbf{A} + \beta \mathbf{B}) = \alpha \operatorname{tr}(\mathbf{A}) + \beta \operatorname{tr}(\mathbf{B})$
2. **交换性**：$\operatorname{tr}(\mathbf{A}\mathbf{B}) = \operatorname{tr}(\mathbf{B}\mathbf{A})$（矩阵维度匹配即可）
3. **转置不变性**：$\operatorname{tr}(\mathbf{A}) = \operatorname{tr}(\mathbf{A}^T)$
4. **Frobenius 内积**：$\operatorname{tr}(\mathbf{A}^T \mathbf{B}) = \sum_{i,j} a_{ij} b_{ij} = \langle \mathbf{A}, \mathbf{B} \rangle_F$（Frobenius 内积）

**梯度识别定理**：若标量 $f$ 对矩阵 $\mathbf{A}$ 的微分可以写成：

$$df = \operatorname{tr}\!\left(\left(\frac{\partial f}{\partial \mathbf{A}}\right)^T d\mathbf{A}\right)$$

则括号内即为梯度矩阵 $\partial f/\partial \mathbf{A}$。

### 具体推导

**公式 5**：$f = \operatorname{tr}(\mathbf{A}\mathbf{B})$，对 $\mathbf{A}$ 求导（$\mathbf{B}$ 为常数）

$$df = \operatorname{tr}(d\mathbf{A} \cdot \mathbf{B}) = \operatorname{tr}(\mathbf{B} \, d\mathbf{A}) = \operatorname{tr}(\mathbf{B}^T \, d\mathbf{A})^T \cdot \ldots$$

更直接地，$\operatorname{tr}(d\mathbf{A} \cdot \mathbf{B}) = \operatorname{tr}(\mathbf{B} \cdot d\mathbf{A})$，对比识别定理（$df = \operatorname{tr}(\mathbf{G}^T d\mathbf{A})$），需要 $\mathbf{G}^T = \mathbf{B}$，即：

$$\boxed{\frac{\partial \operatorname{tr}(\mathbf{A}\mathbf{B})}{\partial \mathbf{A}} = \mathbf{B}^T}$$

**公式 6**：$f = \operatorname{tr}(\mathbf{A}^T \mathbf{A})= \|\mathbf{A}\|_F^2$（矩阵 Frobenius 范数的平方）

$$df = \operatorname{tr}(d\mathbf{A}^T \cdot \mathbf{A}) + \operatorname{tr}(\mathbf{A}^T \cdot d\mathbf{A}) = \operatorname{tr}(\mathbf{A}^T d\mathbf{A})^T + \operatorname{tr}(\mathbf{A}^T d\mathbf{A}) = 2\operatorname{tr}(\mathbf{A}^T d\mathbf{A})$$

对比 $df = \operatorname{tr}(\mathbf{G}^T d\mathbf{A})$，得 $\mathbf{G}^T = 2\mathbf{A}^T$，故：

$$\boxed{\frac{\partial \|\mathbf{A}\|_F^2}{\partial \mathbf{A}} = 2\mathbf{A}}$$

**公式 7**：$f = \log \det(\mathbf{A})$（$\mathbf{A}$ 正定）

利用 $\det(\mathbf{A}) = \det(\mathbf{A})$ 和矩阵行列式引理，可得：

$$df = \operatorname{tr}(\mathbf{A}^{-1} d\mathbf{A})$$

对比识别定理，$\mathbf{G}^T = \mathbf{A}^{-1}$，故：

$$\boxed{\frac{\partial \log \det(\mathbf{A})}{\partial \mathbf{A}} = \mathbf{A}^{-T} = (\mathbf{A}^{-1})^T}$$

当 $\mathbf{A}$ 对称时，$\mathbf{A}^{-T} = \mathbf{A}^{-1}$。这个公式在最大似然估计（高斯分布参数估计）中至关重要。

---

## 24.4 常用矩阵求导公式汇总

以下按分子布局给出最常用的矩阵微分公式。

### 向量对向量

| 函数 | 条件 | 梯度/Jacobian |
|:---|:---|:---|
| $\mathbf{A}\mathbf{x}$ | $\mathbf{A}$ 为常数 | $\mathbf{A}$ |
| $\mathbf{x}^T \mathbf{A}$ | $\mathbf{A}$ 为常数 | $\mathbf{A}^T$ |
| $\mathbf{x}$ | — | $\mathbf{I}$ |
| $\sigma(\mathbf{x})$（逐元素激活函数） | — | $\operatorname{diag}(\sigma'(\mathbf{x}))$ |

### 标量对向量

| 函数 $f$ | 条件 | $\partial f / \partial \mathbf{x}$ |
|:---|:---|:---|
| $\mathbf{a}^T \mathbf{x}$ | $\mathbf{a}$ 为常数 | $\mathbf{a}$ |
| $\mathbf{x}^T \mathbf{a}$ | $\mathbf{a}$ 为常数 | $\mathbf{a}$ |
| $\mathbf{x}^T \mathbf{A} \mathbf{x}$ | $\mathbf{A}$ 对称 | $2\mathbf{A}\mathbf{x}$ |
| $\mathbf{x}^T \mathbf{A} \mathbf{x}$ | $\mathbf{A}$ 一般 | $(\mathbf{A} + \mathbf{A}^T)\mathbf{x}$ |
| $\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$ | — | $2\mathbf{A}^T(\mathbf{A}\mathbf{x} - \mathbf{b})$ |
| $\|\mathbf{x}\|^2$ | — | $2\mathbf{x}$ |

### 标量对矩阵

| 函数 $f$ | 条件 | $\partial f / \partial \mathbf{A}$ |
|:---|:---|:---|
| $\operatorname{tr}(\mathbf{A})$ | $\mathbf{A}$ 为方阵 | $\mathbf{I}$ |
| $\operatorname{tr}(\mathbf{A}\mathbf{B})$ | $\mathbf{B}$ 为常数 | $\mathbf{B}^T$ |
| $\operatorname{tr}(\mathbf{A}^T \mathbf{B})$ | $\mathbf{B}$ 为常数 | $\mathbf{B}$ |
| $\operatorname{tr}(\mathbf{A}^T\mathbf{A})= \|\mathbf{A}\|_F^2$ | — | $2\mathbf{A}$ |
| $\log\det(\mathbf{A})$ | $\mathbf{A}$ 可逆 | $\mathbf{A}^{-T}$ |
| $\mathbf{a}^T \mathbf{A} \mathbf{b}$ | $\mathbf{a}, \mathbf{b}$ 为常数向量 | $\mathbf{a}\mathbf{b}^T$ |
| $\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$ | 对 $\mathbf{A}$ 求导 | $2(\mathbf{A}\mathbf{x}-\mathbf{b})\mathbf{x}^T$ |

### 维度一致性检验

**黄金法则**：导数的形状必须与分母（被对谁求导）的形状一致。

- $f \in \mathbb{R}$，$\mathbf{x} \in \mathbb{R}^n$ → $\partial f/\partial \mathbf{x} \in \mathbb{R}^n$
- $\mathbf{f} \in \mathbb{R}^m$，$\mathbf{x} \in \mathbb{R}^n$ → $\partial \mathbf{f}/\partial \mathbf{x} \in \mathbb{R}^{m \times n}$（Jacobian）
- $f \in \mathbb{R}$，$\mathbf{A} \in \mathbb{R}^{m \times n}$ → $\partial f/\partial \mathbf{A} \in \mathbb{R}^{m \times n}$

---

## 24.5 链式法则与计算图

### 矩阵链式法则

单变量链式法则 $\dfrac{dz}{dx} = \dfrac{dz}{dy} \cdot \dfrac{dy}{dx}$ 推广到向量/矩阵形式：

设 $\mathbf{x} \in \mathbb{R}^n \xrightarrow{\mathbf{g}} \mathbf{y} \in \mathbb{R}^m \xrightarrow{\mathbf{f}} \mathbf{z} \in \mathbb{R}^k$，则：

$$\boxed{\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}}$$

其中：
- $\partial \mathbf{z}/\partial \mathbf{x} \in \mathbb{R}^{k \times n}$（整体 Jacobian）
- $\partial \mathbf{z}/\partial \mathbf{y} \in \mathbb{R}^{k \times m}$（后段 Jacobian）
- $\partial \mathbf{y}/\partial \mathbf{x} \in \mathbb{R}^{m \times n}$（前段 Jacobian）

矩阵乘积 $\mathbb{R}^{k \times m} \cdot \mathbb{R}^{m \times n} = \mathbb{R}^{k \times n}$，维度自动匹配。

**当 $\mathbf{z}$ 为标量**（$k=1$，即 $z \in \mathbb{R}$）时，梯度为：

$$\frac{\partial z}{\partial \mathbf{x}} = \left(\frac{\partial z}{\partial \mathbf{y}}\right)^T \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \quad \in \mathbb{R}^n$$

这里 $\partial z/\partial \mathbf{y} \in \mathbb{R}^m$ 是梯度向量，$\partial \mathbf{y}/\partial \mathbf{x} \in \mathbb{R}^{m \times n}$ 是 Jacobian。

### 计算图与前向/反向传播

**计算图（computational graph）**是描述计算过程的有向无环图（DAG），每个节点代表一个中间变量，每条边代表函数关系。

以两层计算 $z = f(g(\mathbf{x}))$ 为例：

```
x → [g] → y → [f] → z
```

**前向传播**（forward pass）：从左至右计算各节点的值。

**反向传播**（backward pass）：从右至左，利用链式法则累积梯度：

$$\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}$$

对整个网络，梯度通过 Jacobian 矩阵的乘积从输出层反向流向输入层。

### 向量值链式法则的例子

设 $z = \|\mathbf{W}\mathbf{x} - \mathbf{b}\|^2$，分解为：

$$\mathbf{u} = \mathbf{W}\mathbf{x}, \quad \mathbf{v} = \mathbf{u} - \mathbf{b}, \quad z = \|\mathbf{v}\|^2$$

各段 Jacobian：

$$\frac{\partial z}{\partial \mathbf{v}} = 2\mathbf{v}^T, \quad \frac{\partial \mathbf{v}}{\partial \mathbf{u}} = \mathbf{I}, \quad \frac{\partial \mathbf{u}}{\partial \mathbf{x}} = \mathbf{W}$$

链式法则（标量对向量）：

$$\frac{\partial z}{\partial \mathbf{x}} = \mathbf{W}^T \cdot \mathbf{I}^T \cdot (2\mathbf{v}) = 2\mathbf{W}^T(\mathbf{W}\mathbf{x} - \mathbf{b})$$

与直接求导结果完全一致，验证了链式法则的正确性。

---

## 本章小结

- **Jacobian 矩阵** $\partial \mathbf{f}/\partial \mathbf{x} \in \mathbb{R}^{m \times n}$ 是向量值函数 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ 的导数，代表最佳线性近似；其行列式绝对值 $|\det \mathbf{J}|$ 是局部体积缩放因子。

- **梯度向量** $\nabla_{\mathbf{x}} f \in \mathbb{R}^n$ 是标量函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的导数，指向增长最快方向；核心公式 $\partial(\mathbf{x}^T\mathbf{A}\mathbf{x})/\partial\mathbf{x} = 2\mathbf{A}\mathbf{x}$（$\mathbf{A}$ 对称）是深度学习梯度计算的基石。

- **梯度矩阵** $\partial f/\partial \mathbf{A} \in \mathbb{R}^{m \times n}$ 的推导可借助**迹技巧**：将微分写成 $df = \operatorname{tr}(\mathbf{G}^T d\mathbf{A})$ 的形式，则 $\mathbf{G}$ 就是梯度。关键公式包括 $\partial \operatorname{tr}(\mathbf{A}\mathbf{B})/\partial \mathbf{A} = \mathbf{B}^T$ 和 $\partial \log\det(\mathbf{A})/\partial \mathbf{A} = \mathbf{A}^{-T}$。

- **维度一致性**是检验导数正确性的基本工具：梯度/Jacobian 的形状必须与被求导变量同形。

- **矩阵链式法则** $\partial \mathbf{z}/\partial \mathbf{x} = (\partial \mathbf{z}/\partial \mathbf{y})(\partial \mathbf{y}/\partial \mathbf{x})$ 通过 Jacobian 矩阵的乘积传递梯度，是反向传播算法的数学本质。

**本章在整个教程中的地位**：矩阵微积分是线性代数通往深度学习的桥梁。前23章建立的向量空间、矩阵运算、特征值分解、SVD 等工具，在此章通过"对矩阵求导"这一操作，汇聚为神经网络训练的核心机制——反向传播。

---

## 深度学习应用：反向传播的推导

### 神经网络的矩阵表示

考虑一个三层全连接网络（输入层 → 隐藏层 → 输出层）：

**前向传播**：

$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}, \quad \mathbf{a}^{(1)} = \sigma(\mathbf{z}^{(1)})$$

$$\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)} + \mathbf{b}^{(2)}, \quad \hat{\mathbf{y}} = \mathbf{a}^{(2)} = \sigma(\mathbf{z}^{(2)})$$

$$L = \frac{1}{2}\|\hat{\mathbf{y}} - \mathbf{y}\|^2$$

其中 $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$，$\mathbf{W}^{(2)} \in \mathbb{R}^{c \times h}$，$\sigma$ 是逐元素激活函数（如 ReLU 或 Sigmoid）。

### 反向传播的严格推导

**第一步**：输出层误差（loss 对 $\mathbf{z}^{(2)}$ 的梯度）

$$\boldsymbol{\delta}^{(2)} \triangleq \frac{\partial L}{\partial \mathbf{z}^{(2)}}$$

由链式法则：

$$\frac{\partial L}{\partial \mathbf{a}^{(2)}} = \hat{\mathbf{y}} - \mathbf{y}$$

$$\frac{\partial \mathbf{a}^{(2)}}{\partial \mathbf{z}^{(2)}} = \operatorname{diag}(\sigma'(\mathbf{z}^{(2)}))$$

故（标量对向量，链式法则）：

$$\boxed{\boldsymbol{\delta}^{(2)} = (\hat{\mathbf{y}} - \mathbf{y}) \odot \sigma'(\mathbf{z}^{(2)})}$$

其中 $\odot$ 表示逐元素乘法（Hadamard 乘积）——这正是 Jacobian 为对角矩阵时乘法的简化形式。

**第二步**：第二层权重的梯度

$$\frac{\partial L}{\partial \mathbf{W}^{(2)}} = \boldsymbol{\delta}^{(2)} (\mathbf{a}^{(1)})^T \in \mathbb{R}^{c \times h}$$

**推导**：$L = L(\mathbf{z}^{(2)})$，$\mathbf{z}^{(2)} = \mathbf{W}^{(2)}\mathbf{a}^{(1)} + \mathbf{b}^{(2)}$。对矩阵 $\mathbf{W}^{(2)}$ 的 $(i,j)$ 元素：

$$\frac{\partial L}{\partial w^{(2)}_{ij}} = \sum_k \frac{\partial L}{\partial z^{(2)}_k} \cdot \frac{\partial z^{(2)}_k}{\partial w^{(2)}_{ij}} = \delta^{(2)}_i \cdot a^{(1)}_j$$

写成矩阵形式即 $\boldsymbol{\delta}^{(2)} (\mathbf{a}^{(1)})^T$（外积）。

偏置梯度：$\partial L / \partial \mathbf{b}^{(2)} = \boldsymbol{\delta}^{(2)}$（仿照上式，$\partial z^{(2)}_k/\partial b^{(2)}_i = \mathbf{1}_{[k=i]}$）。

**第三步**：误差反向传播到隐藏层

$$\boldsymbol{\delta}^{(1)} = \frac{\partial L}{\partial \mathbf{z}^{(1)}} = \left[(\mathbf{W}^{(2)})^T \boldsymbol{\delta}^{(2)}\right] \odot \sigma'(\mathbf{z}^{(1)})$$

**推导**：$\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)}$，$\mathbf{a}^{(1)} = \sigma(\mathbf{z}^{(1)})$，链式法则：

$$\frac{\partial L}{\partial \mathbf{a}^{(1)}} = (\mathbf{W}^{(2)})^T \boldsymbol{\delta}^{(2)}$$

（注意转置！$\mathbf{W}^{(2)} \in \mathbb{R}^{c \times h}$，故 $(\mathbf{W}^{(2)})^T \in \mathbb{R}^{h \times c}$，乘以 $\boldsymbol{\delta}^{(2)} \in \mathbb{R}^c$ 得 $\mathbb{R}^h$ 向量，维度正确。）

再乘以 $\sigma'(\mathbf{z}^{(1)})$（逐元素），即得 $\boldsymbol{\delta}^{(1)}$。

**第四步**：第一层权重的梯度

$$\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^T$$

**反向传播总结**（可推广到任意深度 $L$ 层）：

$$\boxed{\boldsymbol{\delta}^{(l)} = \left[(\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}\right] \odot \sigma'(\mathbf{z}^{(l)}), \quad \frac{\partial L}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T}$$

这就是经典反向传播算法（Backpropagation）的完整矩阵形式，其数学本质是链式法则在计算图上的系统化应用。

### 自动微分与链式法则的关系

现代深度学习框架（PyTorch、JAX、TensorFlow）实现的**自动微分（automatic differentiation, autograd）**，本质上是以下两种模式之一：

**前向模式（forward mode AD）**：沿计算图正向同步传播导数，每次计算一个输入变量对所有输出的 Jacobian 列（即 Jacobian-向量积 $\mathbf{J}\mathbf{v}$）。适合"输入少、输出多"的场景。

**反向模式（reverse mode AD）**：沿计算图反向传播"伴随量"（adjoint），每次计算一个输出变量对所有输入的梯度（即向量-Jacobian 积 $\mathbf{v}^T\mathbf{J}$）。适合"输入多、输出少（如标量损失）"的场景——这正是神经网络训练的场景，PyTorch 的 `.backward()` 使用此模式。

**关键等价**：PyTorch 反向传播每步计算的**向量-Jacobian 积（VJP）**，在数学上与我们手工推导的矩阵链式法则完全一致——框架只是将此过程自动化、数值化，对用户隐藏了 Jacobian 矩阵的显式构造。

### PyTorch 代码：验证梯度公式

```python
import torch
import torch.nn as nn

# ── 1. 验证标量对向量的梯度公式 ─────────────────────────────────
print("=" * 55)
print("验证 ∂(x^T A x)/∂x = 2Ax  (A 对称正定)")
print("=" * 55)

torch.manual_seed(0)
n = 4
# 构造对称正定矩阵 A = B^T B + I
B = torch.randn(n, n)
A = B.T @ B + torch.eye(n)
x = torch.randn(n, requires_grad=True)

# 前向：计算二次型 f = x^T A x
f = x @ A @ x  # 等价于 (x^T A x)

# 自动微分
f.backward()
autograd_grad = x.grad.clone()

# 手工公式：∂f/∂x = 2Ax
manual_grad = 2 * A @ x.detach()

print(f"自动微分梯度:   {autograd_grad.numpy().round(4)}")
print(f"手工公式 2Ax:  {manual_grad.numpy().round(4)}")
print(f"最大误差: {(autograd_grad - manual_grad).abs().max().item():.2e}")

# ── 2. 验证 ∂‖Ax - b‖²/∂x = 2A^T(Ax - b) ───────────────────────
print("\n" + "=" * 55)
print("验证 ∂‖Ax - b‖²/∂x = 2A^T(Ax - b)")
print("=" * 55)

m = 6
A2 = torch.randn(m, n)
b = torch.randn(m)
x2 = torch.randn(n, requires_grad=True)

residual = A2 @ x2 - b
loss = (residual ** 2).sum()  # ‖Ax - b‖²
loss.backward()
autograd_grad2 = x2.grad.clone()

# 手工公式
manual_grad2 = 2 * A2.T @ (A2 @ x2.detach() - b)

print(f"最大误差: {(autograd_grad2 - manual_grad2).abs().max().item():.2e}")
print("公式验证通过!" if (autograd_grad2 - manual_grad2).abs().max() < 1e-5 else "公式不匹配!")

# ── 3. 完整两层 MLP 的手工反向传播与自动微分对比 ─────────────────
print("\n" + "=" * 55)
print("两层 MLP 手工反向传播 vs PyTorch autograd")
print("=" * 55)

torch.manual_seed(42)
d, h, c = 4, 8, 3   # 输入维度, 隐藏维度, 输出维度

W1 = torch.randn(h, d, requires_grad=True)
b1 = torch.zeros(h, requires_grad=True)
W2 = torch.randn(c, h, requires_grad=True)
b2 = torch.zeros(c, requires_grad=True)

x_in = torch.randn(d)
y_true = torch.randn(c)

# ── 前向传播 ──
z1 = W1 @ x_in + b1                   # (h,)
a1 = torch.relu(z1)                   # (h,)
z2 = W2 @ a1 + b2                     # (c,)
y_hat = torch.sigmoid(z2)             # (c,)
L = 0.5 * ((y_hat - y_true) ** 2).sum()

# ── PyTorch 自动微分 ──
L.backward()
autograd_dW1 = W1.grad.clone()
autograd_dW2 = W2.grad.clone()

# ── 手工反向传播 ──
# 保存前向计算中的中间值（已在上面计算）
with torch.no_grad():
    # Step 1: 输出层误差 δ² = (ŷ - y) ⊙ σ'(z²)
    sigmoid_prime_z2 = y_hat * (1 - y_hat)           # σ'(z²)
    delta2 = (y_hat - y_true) * sigmoid_prime_z2      # (c,)

    # Step 2: W² 的梯度 = δ² ⊗ a¹ (外积)
    manual_dW2 = torch.outer(delta2, a1)               # (c, h)

    # Step 3: 误差反传到隐藏层 δ¹ = (W²)^T δ² ⊙ ReLU'(z¹)
    relu_prime_z1 = (z1 > 0).float()                  # ReLU 的次梯度
    delta1 = (W2.T @ delta2) * relu_prime_z1           # (h,)

    # Step 4: W¹ 的梯度 = δ¹ ⊗ x (外积)
    manual_dW1 = torch.outer(delta1, x_in)             # (h, d)

print(f"∂L/∂W² 最大误差: {(autograd_dW2 - manual_dW2).abs().max().item():.2e}")
print(f"∂L/∂W¹ 最大误差: {(autograd_dW1 - manual_dW1).abs().max().item():.2e}")

if max((autograd_dW2 - manual_dW2).abs().max(),
       (autograd_dW1 - manual_dW1).abs().max()) < 1e-5:
    print("手工反向传播与 autograd 完全一致!")

# ── 4. 矩阵梯度公式验证：∂tr(AB)/∂A = B^T ─────────────────────
print("\n" + "=" * 55)
print("验证 ∂tr(AB)/∂A = B^T")
print("=" * 55)

m2, n2 = 3, 4
A_mat = torch.randn(m2, n2, requires_grad=True)
B_mat = torch.randn(n2, m2)   # 使 AB 为方阵

trace_AB = torch.trace(A_mat @ B_mat)
trace_AB.backward()
autograd_dA = A_mat.grad.clone()

# 手工公式：B^T
manual_dA = B_mat.T  # (m, n)

print(f"最大误差: {(autograd_dA - manual_dA).abs().max().item():.2e}")
print("公式 ∂tr(AB)/∂A = B^T 验证通过!" if (autograd_dA - manual_dA).abs().max() < 1e-5
      else "公式不匹配!")
```

**代码解读**：

- **第 1-2 部分**：分别验证二次型梯度公式和最小二乘梯度公式，每次都将 PyTorch 自动微分结果与手工公式逐元素对比，误差应在 $10^{-6}$ 量级（浮点精度）。

- **第 3 部分**：完整实现两层 MLP 的手工反向传播，严格按照推导的四步公式计算 $\boldsymbol{\delta}^{(2)}, \partial L/\partial\mathbf{W}^{(2)}, \boldsymbol{\delta}^{(1)}, \partial L/\partial\mathbf{W}^{(1)}$，并与 PyTorch 的 `.backward()` 逐元素对比。这直观证明了反向传播算法确实是矩阵链式法则的系统化实现。

- **第 4 部分**：验证矩阵梯度公式 $\partial \operatorname{tr}(\mathbf{A}\mathbf{B})/\partial\mathbf{A} = \mathbf{B}^T$，展示矩阵微积分公式与 autograd 的完全一致。

| 矩阵微积分概念 | 深度学习对应物 | 工程实现 |
|:---|:---|:---|
| 标量对向量梯度 $\partial L/\partial \mathbf{a}$ | 层输出的误差信号 | `.grad` 属性 |
| 标量对矩阵梯度 $\partial L/\partial \mathbf{W}$ | 权重更新方向 | `param.grad` |
| Jacobian 矩阵 $\partial \mathbf{a}/\partial \mathbf{z}$ | 激活函数的局部导数 | `create_graph=True` |
| 矩阵链式法则 | 反向传播算法 | `.backward()` |
| 向量-Jacobian 积（VJP） | 反向模式自动微分 | `torch.autograd.grad` |
| 外积 $\boldsymbol{\delta}(\mathbf{a}^{\text{prev}})^T$ | 权重梯度 = 误差 $\otimes$ 激活 | `einsum`, `outer` |

---

## 练习题

**练习 1**（基础——公式验证）

计算下列各式的梯度，并说明推导过程。

（a）$f(\mathbf{x}) = \mathbf{c}^T \mathbf{x} + \mathbf{x}^T \mathbf{x}$，求 $\partial f/\partial \mathbf{x}$。

（b）$f(\mathbf{x}) = \mathbf{x}^T \mathbf{A}^T \mathbf{A} \mathbf{x} - 2\mathbf{b}^T \mathbf{A} \mathbf{x}$（$\mathbf{A}$ 为常数矩阵），求 $\partial f/\partial \mathbf{x}$，并令梯度为零推导正规方程。

（c）$f(\mathbf{A}) = \mathbf{u}^T \mathbf{A} \mathbf{v}$（$\mathbf{u}, \mathbf{v}$ 为常数列向量，$\mathbf{A} \in \mathbb{R}^{m \times n}$），求 $\partial f/\partial \mathbf{A}$。（提示：将结果写成两个向量的外积形式。）

---

**练习 2**（基础——Jacobian 矩阵）

设 Softmax 函数 $\mathbf{s}: \mathbb{R}^n \to \mathbb{R}^n$ 定义为：

$$s_i(\mathbf{z}) = \frac{e^{z_i}}{\sum_{k=1}^n e^{z_k}}, \quad i = 1, \ldots, n$$

（a）计算 Jacobian 矩阵 $\partial \mathbf{s}/\partial \mathbf{z}$ 的 $(i,j)$ 元素 $\partial s_i/\partial z_j$。

（b）证明 Jacobian 矩阵可以写成 $\mathbf{J} = \operatorname{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^T$。

（c）说明为什么 $\mathbf{J}$ 是半正定矩阵，以及它为什么总有一个零特征值。

---

**练习 3**（中等——迹技巧与矩阵梯度）

设 $\mathbf{W} \in \mathbb{R}^{n \times n}$，计算以下函数对 $\mathbf{W}$ 的梯度。

（a）$f(\mathbf{W}) = \operatorname{tr}(\mathbf{W}^T \mathbf{W})$（即 $\|\mathbf{W}\|_F^2$）。

（b）$f(\mathbf{W}) = \operatorname{tr}(\mathbf{A}\mathbf{W}\mathbf{B})$，其中 $\mathbf{A} \in \mathbb{R}^{n \times n}$，$\mathbf{B} \in \mathbb{R}^{n \times n}$ 均为常数方阵。（提示：利用迹的循环不变性 $\operatorname{tr}(\mathbf{P}\mathbf{Q}\mathbf{R}) = \operatorname{tr}(\mathbf{R}\mathbf{P}\mathbf{Q})$。）

（c）$f(\mathbf{W}) = \|\mathbf{X}\mathbf{W} - \mathbf{Y}\|_F^2$，其中 $\mathbf{X} \in \mathbb{R}^{m \times d}$，$\mathbf{Y} \in \mathbb{R}^{m \times n}$ 均为常数矩阵。令梯度为零，求最优 $\mathbf{W}^*$（矩阵最小二乘）。

---

**练习 4**（中等——链式法则应用）

考虑如下计算链：

$$\mathbf{x} \xrightarrow{\mathbf{W}_1} \mathbf{z} = \mathbf{W}_1 \mathbf{x} \xrightarrow{\text{ReLU}} \mathbf{h} = \max(\mathbf{z}, \mathbf{0}) \xrightarrow{\mathbf{W}_2} \mathbf{y} = \mathbf{W}_2 \mathbf{h} \xrightarrow{\text{loss}} L = \frac{1}{2}\|\mathbf{y} - \hat{\mathbf{y}}\|^2$$

其中 $\mathbf{x} \in \mathbb{R}^d$，$\mathbf{W}_1 \in \mathbb{R}^{h \times d}$，$\mathbf{h} \in \mathbb{R}^h$，$\mathbf{W}_2 \in \mathbb{R}^c \times h$，$\hat{\mathbf{y}} \in \mathbb{R}^c$ 为目标。

（a）写出各段 Jacobian：$\partial \mathbf{z}/\partial \mathbf{x}$，$\partial \mathbf{h}/\partial \mathbf{z}$，$\partial \mathbf{y}/\partial \mathbf{h}$，$\partial L/\partial \mathbf{y}$。

（b）用链式法则逐步计算 $\partial L/\partial \mathbf{h}$，$\partial L/\partial \mathbf{z}$，$\partial L/\partial \mathbf{x}$。

（c）计算 $\partial L/\partial \mathbf{W}_1$ 和 $\partial L/\partial \mathbf{W}_2$，并验证维度一致性。

（d）写出一步梯度下降更新公式 $\mathbf{W}_1 \leftarrow \mathbf{W}_1 - \eta \partial L/\partial \mathbf{W}_1$，并解释为何梯度计算需要保存前向传播的中间变量 $\mathbf{z}$ 和 $\mathbf{h}$。

---

**练习 5**（进阶——从矩阵微积分推导 PCA）

主成分分析（PCA）的目标是找到单位向量 $\mathbf{w} \in \mathbb{R}^d$（$\|\mathbf{w}\| = 1$），使得数据矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$（已中心化）在 $\mathbf{w}$ 方向上的投影方差最大：

$$\max_{\mathbf{w}:\,\|\mathbf{w}\|=1} \operatorname{Var}(\mathbf{X}\mathbf{w}) = \max_{\mathbf{w}:\,\|\mathbf{w}\|=1} \frac{1}{n-1}\|\mathbf{X}\mathbf{w}\|^2$$

（a）设 $\mathbf{S} = \dfrac{1}{n-1}\mathbf{X}^T\mathbf{X}$ 为样本协方差矩阵，证明目标函数等价于 $\max_{\|\mathbf{w}\|=1} \mathbf{w}^T \mathbf{S} \mathbf{w}$。

（b）利用 Lagrange 乘数法，引入约束 $\|\mathbf{w}\|^2 = 1$，对增广函数 $g(\mathbf{w}, \lambda) = \mathbf{w}^T \mathbf{S} \mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$ 求梯度并令其为零。

（c）证明最优解满足特征方程 $\mathbf{S}\mathbf{w} = \lambda\mathbf{w}$，并由此得出 PCA 的最大方差方向就是协方差矩阵 $\mathbf{S}$ 的最大特征向量。

（d）将此结论与第21章的 Rayleigh 商极值定理联系起来，说明两种推导路径的统一性。

（e）若要找第 $k$ 个主成分方向（与前 $k-1$ 个正交），写出对应的约束优化问题，并说明解为 $\mathbf{S}$ 的第 $k$ 大特征向量（Courant-Fischer 定理的应用）。

---

## 练习答案

<details>
<summary>点击展开 练习1 答案</summary>

**（a）$f(\mathbf{x}) = \mathbf{c}^T \mathbf{x} + \mathbf{x}^T \mathbf{x}$**

分两项分别求导，再利用线性性相加：

$$\frac{\partial (\mathbf{c}^T\mathbf{x})}{\partial \mathbf{x}} = \mathbf{c}, \qquad \frac{\partial (\mathbf{x}^T\mathbf{x})}{\partial \mathbf{x}} = 2\mathbf{x}$$

（第二项用公式 $\partial(\mathbf{x}^T\mathbf{A}\mathbf{x})/\partial\mathbf{x} = 2\mathbf{A}\mathbf{x}$，取 $\mathbf{A} = \mathbf{I}$。）

$$\boxed{\frac{\partial f}{\partial \mathbf{x}} = \mathbf{c} + 2\mathbf{x}}$$

**（b）$f(\mathbf{x}) = \mathbf{x}^T \mathbf{A}^T \mathbf{A} \mathbf{x} - 2\mathbf{b}^T \mathbf{A} \mathbf{x}$**

令 $\mathbf{C} = \mathbf{A}^T\mathbf{A}$（对称矩阵），则 $f = \mathbf{x}^T\mathbf{C}\mathbf{x} - 2\mathbf{b}^T\mathbf{A}\mathbf{x}$。

$$\frac{\partial f}{\partial \mathbf{x}} = 2\mathbf{C}\mathbf{x} - 2\mathbf{A}^T\mathbf{b} = 2\mathbf{A}^T\mathbf{A}\mathbf{x} - 2\mathbf{A}^T\mathbf{b}$$

令梯度为零：$2\mathbf{A}^T\mathbf{A}\mathbf{x} = 2\mathbf{A}^T\mathbf{b}$，即**正规方程**：

$$\mathbf{A}^T\mathbf{A}\mathbf{x} = \mathbf{A}^T\mathbf{b}$$

这正是 $\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 = \mathbf{x}^T\mathbf{A}^T\mathbf{A}\mathbf{x} - 2\mathbf{b}^T\mathbf{A}\mathbf{x} + \|\mathbf{b}\|^2$ 最小化的必要条件（与第18章最小二乘完全吻合）。

**（c）$f(\mathbf{A}) = \mathbf{u}^T \mathbf{A} \mathbf{v}$**

$f = \sum_{i,j} u_i a_{ij} v_j$，故 $\partial f / \partial a_{ij} = u_i v_j$。

写成矩阵形式：

$$\boxed{\frac{\partial (\mathbf{u}^T\mathbf{A}\mathbf{v})}{\partial \mathbf{A}} = \mathbf{u}\mathbf{v}^T \in \mathbb{R}^{m \times n}}$$

（外积，$(i,j)$ 元素为 $u_i v_j$。）

**迹方法验证**：$f = \mathbf{u}^T\mathbf{A}\mathbf{v} = \operatorname{tr}(\mathbf{v}\mathbf{u}^T\mathbf{A}) = \operatorname{tr}(\mathbf{u}^T\mathbf{A}\mathbf{v})$（标量的迹等于自身）。

更直接地，$df = \operatorname{tr}(\mathbf{u}^T \, d\mathbf{A} \, \mathbf{v}) = \operatorname{tr}(\mathbf{v}\mathbf{u}^T \, d\mathbf{A}) = \operatorname{tr}((\mathbf{u}\mathbf{v}^T)^T d\mathbf{A})$，

对比识别定理，梯度为 $\mathbf{u}\mathbf{v}^T$。✓

</details>

<details>
<summary>点击展开 练习2 答案</summary>

**（a）Softmax 的偏导数**

设 $S = \sum_{k} e^{z_k}$，则 $s_i = e^{z_i}/S$。

当 $i = j$：

$$\frac{\partial s_i}{\partial z_i} = \frac{e^{z_i} \cdot S - e^{z_i} \cdot e^{z_i}}{S^2} = \frac{e^{z_i}}{S} - \left(\frac{e^{z_i}}{S}\right)^2 = s_i - s_i^2 = s_i(1 - s_i)$$

当 $i \neq j$：

$$\frac{\partial s_i}{\partial z_j} = \frac{0 \cdot S - e^{z_i} \cdot e^{z_j}}{S^2} = -\frac{e^{z_i}}{S} \cdot \frac{e^{z_j}}{S} = -s_i s_j$$

综合：

$$\frac{\partial s_i}{\partial z_j} = s_i(\mathbf{1}_{[i=j]} - s_j)$$

**（b）Jacobian 矩阵的矩阵形式**

$$(\mathbf{J})_{ij} = \frac{\partial s_i}{\partial z_j} = s_i \delta_{ij} - s_i s_j$$

其中 $s_i \delta_{ij}$ 构成对角矩阵 $\operatorname{diag}(\mathbf{s})$，$s_i s_j$ 构成外积 $\mathbf{s}\mathbf{s}^T$，故：

$$\boxed{\mathbf{J} = \operatorname{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^T}$$

**（c）半正定性与零特征值**

**半正定性**：对任意 $\mathbf{v} \in \mathbb{R}^n$：

$$\mathbf{v}^T \mathbf{J} \mathbf{v} = \sum_i s_i v_i^2 - \left(\sum_i s_i v_i\right)^2$$

由 Cauchy-Schwarz 不等式（以 $s_i > 0$ 为权重），$\sum_i s_i v_i^2 \geq \left(\sum_i s_i v_i\right)^2 / \sum_i s_i = \left(\sum_i s_i v_i\right)^2$（因为 $\sum_i s_i = 1$），故 $\mathbf{v}^T\mathbf{J}\mathbf{v} \geq 0$。

**零特征值**：取 $\mathbf{v} = \mathbf{1} = (1,1,\ldots,1)^T$：

$$\mathbf{J}\mathbf{1} = \operatorname{diag}(\mathbf{s})\mathbf{1} - \mathbf{s}\mathbf{s}^T\mathbf{1} = \mathbf{s} - \mathbf{s}(\mathbf{s}^T\mathbf{1}) = \mathbf{s} - \mathbf{s} \cdot 1 = \mathbf{0}$$

（因为 $\mathbf{s}^T\mathbf{1} = \sum_i s_i = 1$。）故 $\mathbf{1}$ 是 $\mathbf{J}$ 对应特征值 $0$ 的特征向量。

**直觉**：Softmax 输出的总和恒为 $1$，在全1方向上扰动 $\mathbf{z}$ 不改变 $\mathbf{s}$（因为分子分母同比例变化），故该方向的导数为零。

</details>

<details>
<summary>点击展开 练习3 答案</summary>

**（a）$f(\mathbf{W}) = \operatorname{tr}(\mathbf{W}^T\mathbf{W}) = \|\mathbf{W}\|_F^2$**

$$df = \operatorname{tr}(d\mathbf{W}^T \cdot \mathbf{W} + \mathbf{W}^T \cdot d\mathbf{W}) = \operatorname{tr}(\mathbf{W}^T d\mathbf{W})^T + \operatorname{tr}(\mathbf{W}^T d\mathbf{W}) = 2\operatorname{tr}(\mathbf{W}^T d\mathbf{W})$$

对比 $df = \operatorname{tr}(\mathbf{G}^T d\mathbf{W})$，得 $\mathbf{G}^T = 2\mathbf{W}^T$，即：

$$\boxed{\frac{\partial \|\mathbf{W}\|_F^2}{\partial \mathbf{W}} = 2\mathbf{W}}$$

**（b）$f(\mathbf{W}) = \operatorname{tr}(\mathbf{A}\mathbf{W}\mathbf{B})$**

$$df = \operatorname{tr}(\mathbf{A} \, d\mathbf{W} \, \mathbf{B})$$

利用迹的循环不变性 $\operatorname{tr}(\mathbf{A} \, d\mathbf{W} \, \mathbf{B}) = \operatorname{tr}(\mathbf{B}\mathbf{A} \, d\mathbf{W}) = \operatorname{tr}((\mathbf{A}^T\mathbf{B}^T)^T d\mathbf{W})$...

更准确地对比识别定理：$df = \operatorname{tr}(\mathbf{G}^T d\mathbf{W})$，需要 $\mathbf{G}^T d\mathbf{W}$ 与 $\mathbf{A} \, d\mathbf{W} \, \mathbf{B}$ 具有相同的迹。

由 $\operatorname{tr}(\mathbf{A} \, d\mathbf{W} \, \mathbf{B}) = \operatorname{tr}(\mathbf{B}\mathbf{A} \, d\mathbf{W})$（循环不变性），对比 $\operatorname{tr}(\mathbf{G}^T d\mathbf{W})$：

$$\mathbf{G}^T = \mathbf{B}\mathbf{A} \implies \mathbf{G} = \mathbf{A}^T\mathbf{B}^T$$

$$\boxed{\frac{\partial \operatorname{tr}(\mathbf{A}\mathbf{W}\mathbf{B})}{\partial \mathbf{W}} = \mathbf{A}^T\mathbf{B}^T}$$

**（c）$f(\mathbf{W}) = \|\mathbf{X}\mathbf{W} - \mathbf{Y}\|_F^2$**

设 $\mathbf{R} = \mathbf{X}\mathbf{W} - \mathbf{Y}$（残差矩阵），则 $f = \operatorname{tr}(\mathbf{R}^T\mathbf{R})$。

$$df = \operatorname{tr}(d\mathbf{R}^T \cdot \mathbf{R} + \mathbf{R}^T \cdot d\mathbf{R}) = 2\operatorname{tr}(\mathbf{R}^T d\mathbf{R})$$

而 $d\mathbf{R} = \mathbf{X} \, d\mathbf{W}$，故：

$$df = 2\operatorname{tr}(\mathbf{R}^T \mathbf{X} \, d\mathbf{W}) = 2\operatorname{tr}((\mathbf{X}^T\mathbf{R})^T d\mathbf{W})$$

对比识别定理：$\mathbf{G} = 2\mathbf{X}^T\mathbf{R} = 2\mathbf{X}^T(\mathbf{X}\mathbf{W} - \mathbf{Y})$。

$$\boxed{\frac{\partial \|\mathbf{X}\mathbf{W} - \mathbf{Y}\|_F^2}{\partial \mathbf{W}} = 2\mathbf{X}^T(\mathbf{X}\mathbf{W} - \mathbf{Y})}$$

令梯度为零：$\mathbf{X}^T\mathbf{X}\mathbf{W} = \mathbf{X}^T\mathbf{Y}$，最优解为：

$$\mathbf{W}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$$

（矩阵最小二乘，每列单独是向量最小二乘问题。）

</details>

<details>
<summary>点击展开 练习4 答案</summary>

**（a）各段 Jacobian**

- $\dfrac{\partial \mathbf{z}}{\partial \mathbf{x}} = \mathbf{W}_1 \in \mathbb{R}^{h \times d}$（线性层的 Jacobian 就是权重矩阵）

- $\dfrac{\partial \mathbf{h}}{\partial \mathbf{z}} = \operatorname{diag}(\mathbb{1}[\mathbf{z} > \mathbf{0}]) \in \mathbb{R}^{h \times h}$（ReLU 的 Jacobian 为对角矩阵，对角元为 $\mathbf{1}_{[z_i > 0]}$，即激活位置为1，非激活位置为0）

- $\dfrac{\partial \mathbf{y}}{\partial \mathbf{h}} = \mathbf{W}_2 \in \mathbb{R}^{c \times h}$

- $\dfrac{\partial L}{\partial \mathbf{y}} = \mathbf{y} - \hat{\mathbf{y}} \in \mathbb{R}^c$（标量对向量的梯度）

**（b）逐步反向传播**

$$\frac{\partial L}{\partial \mathbf{h}} = \left(\frac{\partial \mathbf{y}}{\partial \mathbf{h}}\right)^T \frac{\partial L}{\partial \mathbf{y}} = \mathbf{W}_2^T (\mathbf{y} - \hat{\mathbf{y}}) \in \mathbb{R}^h$$

$$\frac{\partial L}{\partial \mathbf{z}} = \left(\frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)^T \frac{\partial L}{\partial \mathbf{h}} = \operatorname{diag}(\mathbb{1}[\mathbf{z} > \mathbf{0}]) \cdot \mathbf{W}_2^T (\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{W}_2^T (\mathbf{y} - \hat{\mathbf{y}}) \odot \mathbb{1}[\mathbf{z} > \mathbf{0}] \in \mathbb{R}^h$$

（对角 Jacobian 与向量相乘等价于逐元素乘法。）

$$\frac{\partial L}{\partial \mathbf{x}} = \left(\frac{\partial \mathbf{z}}{\partial \mathbf{x}}\right)^T \frac{\partial L}{\partial \mathbf{z}} = \mathbf{W}_1^T \cdot \frac{\partial L}{\partial \mathbf{z}} \in \mathbb{R}^d$$

**（c）权重梯度**

设 $\boldsymbol{\delta}_2 = \mathbf{y} - \hat{\mathbf{y}}$，$\boldsymbol{\delta}_1 = \partial L/\partial \mathbf{z}$（已在（b）中求得）：

$$\frac{\partial L}{\partial \mathbf{W}_2} = \boldsymbol{\delta}_2 \mathbf{h}^T \in \mathbb{R}^{c \times h}$$

（维度：$\mathbb{R}^c \times \mathbb{R}^h \to \mathbb{R}^{c \times h}$，正确。）

$$\frac{\partial L}{\partial \mathbf{W}_1} = \boldsymbol{\delta}_1 \mathbf{x}^T \in \mathbb{R}^{h \times d}$$

（维度：$\mathbb{R}^h \times \mathbb{R}^d \to \mathbb{R}^{h \times d}$，与 $\mathbf{W}_1$ 同形，正确。）

**（d）梯度下降更新与中间变量的必要性**

$$\mathbf{W}_1 \leftarrow \mathbf{W}_1 - \eta \boldsymbol{\delta}_1 \mathbf{x}^T, \qquad \mathbf{W}_2 \leftarrow \mathbf{W}_2 - \eta \boldsymbol{\delta}_2 \mathbf{h}^T$$

计算 $\partial L/\partial \mathbf{W}_1$ 需要 $\boldsymbol{\delta}_1$，而 $\boldsymbol{\delta}_1 = \mathbf{W}_2^T\boldsymbol{\delta}_2 \odot \mathbb{1}[\mathbf{z}>\mathbf{0}]$，其中 $\mathbb{1}[\mathbf{z}>\mathbf{0}]$ 依赖前向传播时的 $\mathbf{z}$ 值；计算 $\partial L/\partial \mathbf{W}_2$ 需要 $\mathbf{h}$ 的值。因此，**前向传播时必须保存 $\mathbf{z}$ 和 $\mathbf{h}$**，这正是 PyTorch 在 `.backward()` 前需要保留计算图（`retain_graph`）的根本原因，也是 GPU 显存随模型深度增加的主要来源。

</details>

<details>
<summary>点击展开 练习5 答案</summary>

**（a）目标函数的等价性**

$$\operatorname{Var}(\mathbf{X}\mathbf{w}) = \frac{1}{n-1}\|\mathbf{X}\mathbf{w}\|^2 = \frac{1}{n-1}(\mathbf{X}\mathbf{w})^T(\mathbf{X}\mathbf{w}) = \frac{1}{n-1}\mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{w}^T \underbrace{\frac{\mathbf{X}^T\mathbf{X}}{n-1}}_{\mathbf{S}} \mathbf{w}$$

故 $\max_{\|\mathbf{w}\|=1} \operatorname{Var}(\mathbf{X}\mathbf{w}) \equiv \max_{\|\mathbf{w}\|=1} \mathbf{w}^T \mathbf{S} \mathbf{w}$。$\square$

**（b）Lagrange 乘数法**

引入约束 $\|\mathbf{w}\|^2 = 1$，Lagrangian：

$$g(\mathbf{w}, \lambda) = \mathbf{w}^T \mathbf{S} \mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$$

对 $\mathbf{w}$ 求梯度（用公式 $\partial(\mathbf{w}^T\mathbf{S}\mathbf{w})/\partial\mathbf{w} = 2\mathbf{S}\mathbf{w}$，$\mathbf{S}$ 对称）：

$$\frac{\partial g}{\partial \mathbf{w}} = 2\mathbf{S}\mathbf{w} - 2\lambda\mathbf{w} = \mathbf{0}$$

**（c）最优解是最大特征向量**

由梯度为零得：$\mathbf{S}\mathbf{w} = \lambda\mathbf{w}$，即 $\mathbf{w}$ 是 $\mathbf{S}$ 的特征向量。

代入目标函数：$\mathbf{w}^T\mathbf{S}\mathbf{w} = \mathbf{w}^T(\lambda\mathbf{w}) = \lambda\|\mathbf{w}\|^2 = \lambda$。

要最大化 $\mathbf{w}^T\mathbf{S}\mathbf{w} = \lambda$，需选取最大特征值 $\lambda_{\max}$ 对应的特征向量 $\mathbf{q}_{\max}$。

**结论**：PCA 的第一主成分方向 $\mathbf{w}_1^* = \mathbf{q}_{\max}$，最大方差 $= \lambda_{\max}(\mathbf{S})$。$\square$

**（d）与 Rayleigh 商极值定理的联系**

$\max_{\|\mathbf{w}\|=1} \mathbf{w}^T\mathbf{S}\mathbf{w}$ 正是 $\mathbf{S}$ 的 Rayleigh 商在单位球面上的最大值。由第21章定理，该最大值等于 $\lambda_{\max}(\mathbf{S})$，在最大特征向量处取到。

两种推导路径的统一：**Lagrange 乘数法**（矩阵微积分视角）与 **Rayleigh 商极值定理**（谱理论视角）殊途同归，其本质是同一个数学事实：约束二次型优化的最优解是矩阵的特征向量。

**（e）第 $k$ 个主成分**

优化问题：

$$\max_{\mathbf{w}:\,\|\mathbf{w}\|=1,\, \mathbf{w} \perp \mathbf{w}_1^*, \ldots, \mathbf{w} \perp \mathbf{w}_{k-1}^*} \mathbf{w}^T\mathbf{S}\mathbf{w}$$

由 Courant-Fischer 定理（第21章），在约束于与前 $k-1$ 个特征向量正交的 $k$ 维子空间中，Rayleigh 商的最大值等于第 $k$ 大特征值 $\lambda_k$，最优解是第 $k$ 大特征向量 $\mathbf{q}_k$。

因此，$k$ 个主成分方向 $\mathbf{w}_1^*, \ldots, \mathbf{w}_k^*$ 正是 $\mathbf{S}$ 按特征值从大到小排列的前 $k$ 个正交特征向量，主成分分析完全等价于协方差矩阵的谱分解（特征值分解）。这也解释了为什么 PCA 在实现上可以直接对 $\mathbf{X}$ 做 SVD：$\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$ 中的右奇异向量 $\mathbf{v}_1, \ldots, \mathbf{v}_k$ 就是 $\mathbf{S} = \mathbf{V}\mathbf{\Sigma}^2\mathbf{V}^T/(n-1)$ 的前 $k$ 个特征向量。

</details>
