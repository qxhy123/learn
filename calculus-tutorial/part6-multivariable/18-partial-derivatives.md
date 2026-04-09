# 第18章 偏导数

## 学习目标

通过本章学习，你将能够：

- 理解多元函数的定义，掌握二元函数的极限与连续性
- 掌握偏导数的定义和计算方法，理解其几何意义
- 理解高阶偏导数和混合偏导数相等定理
- 掌握全微分的概念，理解可微与可偏导的关系
- 熟练运用链式法则求多元复合函数的偏导数
- 掌握方向导数与梯度的概念及其应用
- 理解隐函数定理并能求隐函数的导数

---

## 18.1 多元函数的基本概念

### 18.1.1 二元函数的定义

**定义**：设 $D$ 是 $\mathbb{R}^2$ 的一个非空子集。如果对于每个有序数对 $(x, y) \in D$，按照某种对应法则 $f$，都有唯一确定的实数 $z$ 与之对应，则称 $f$ 是定义在 $D$ 上的**二元函数**，记作

$$z = f(x, y), \quad (x, y) \in D$$

其中 $D$ 称为函数的**定义域**，$x$、$y$ 称为**自变量**，$z$ 称为**因变量**。

> **例题 18.1** 求函数 $f(x, y) = \sqrt{1 - x^2 - y^2}$ 的定义域。

**解**：要使函数有意义，需要 $1 - x^2 - y^2 \geq 0$，即 $x^2 + y^2 \leq 1$。

因此定义域为 $D = \{(x, y) \mid x^2 + y^2 \leq 1\}$，这是以原点为圆心、半径为 $1$ 的圆盘（包括边界）。

### 18.1.2 多元函数的极限

**定义**：设函数 $f(x, y)$ 在点 $P_0(x_0, y_0)$ 的某去心邻域内有定义。如果对于任意给定的 $\varepsilon > 0$，总存在 $\delta > 0$，使得当 $0 < \sqrt{(x-x_0)^2 + (y-y_0)^2} < \delta$ 时，有

$$|f(x, y) - A| < \varepsilon$$

则称 $A$ 为函数 $f(x, y)$ 当 $(x, y) \to (x_0, y_0)$ 时的**极限**，记作

$$\lim_{(x,y) \to (x_0, y_0)} f(x, y) = A$$

**注意**：二元函数的极限要求点 $(x, y)$ 以**任意方式**趋近于 $(x_0, y_0)$ 时，函数值都趋于同一个常数 $A$。

> **例题 18.2** 证明极限 $\lim_{(x,y) \to (0,0)} \dfrac{xy}{x^2 + y^2}$ 不存在。

**解**：考虑沿不同路径趋近原点：

- 沿 $y = 0$：$\lim_{x \to 0} \dfrac{x \cdot 0}{x^2 + 0} = 0$
- 沿 $y = x$：$\lim_{x \to 0} \dfrac{x \cdot x}{x^2 + x^2} = \dfrac{x^2}{2x^2} = \dfrac{1}{2}$

沿不同路径趋近原点得到不同的极限值，因此该极限不存在。

### 18.1.3 多元函数的连续性

**定义**：设函数 $f(x, y)$ 在点 $P_0(x_0, y_0)$ 的某邻域内有定义。如果

$$\lim_{(x,y) \to (x_0, y_0)} f(x, y) = f(x_0, y_0)$$

则称 $f(x, y)$ 在点 $P_0$ 处**连续**。

**性质**：由初等函数组合而成的多元函数在其定义域内的所有内点都是连续的。

---

## 18.2 偏导数

### 18.2.1 偏导数的定义

**定义**：设函数 $z = f(x, y)$ 在点 $(x_0, y_0)$ 的某邻域内有定义。固定 $y = y_0$，如果极限

$$\lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x, y_0) - f(x_0, y_0)}{\Delta x}$$

存在，则称此极限为 $f(x, y)$ 在点 $(x_0, y_0)$ 处对 $x$ 的**偏导数**，记作

$$f_x(x_0, y_0) \quad \text{或} \quad \frac{\partial f}{\partial x}\bigg|_{(x_0, y_0)} \quad \text{或} \quad \frac{\partial z}{\partial x}\bigg|_{(x_0, y_0)}$$

类似地，对 $y$ 的偏导数定义为：

$$f_y(x_0, y_0) = \lim_{\Delta y \to 0} \frac{f(x_0, y_0 + \Delta y) - f(x_0, y_0)}{\Delta y}$$

**计算方法**：求 $f_x$ 时，将 $y$ 视为常数，对 $x$ 求导；求 $f_y$ 时，将 $x$ 视为常数，对 $y$ 求导。

> **例题 18.3** 设 $f(x, y) = x^2 + 3xy + y^2$，求 $f_x$ 和 $f_y$。

**解**：

$$f_x = \frac{\partial}{\partial x}(x^2 + 3xy + y^2) = 2x + 3y$$

$$f_y = \frac{\partial}{\partial y}(x^2 + 3xy + y^2) = 3x + 2y$$

> **例题 18.4** 设 $f(x, y) = e^{xy} \sin(x + y)$，求 $f_x$ 和 $f_y$。

**解**：

$$f_x = ye^{xy}\sin(x+y) + e^{xy}\cos(x+y) = e^{xy}[y\sin(x+y) + \cos(x+y)]$$

$$f_y = xe^{xy}\sin(x+y) + e^{xy}\cos(x+y) = e^{xy}[x\sin(x+y) + \cos(x+y)]$$

### 18.2.2 偏导数的几何意义

设曲面 $z = f(x, y)$，点 $P_0(x_0, y_0, z_0)$ 在曲面上。

- 用平面 $y = y_0$ 截曲面，得到曲线 $z = f(x, y_0)$
- $f_x(x_0, y_0)$ 是这条曲线在 $P_0$ 处切线的斜率（相对于 $x$ 轴）

同理，$f_y(x_0, y_0)$ 是用平面 $x = x_0$ 截得的曲线在 $P_0$ 处切线的斜率。

### 18.2.3 高阶偏导数

设 $z = f(x, y)$ 的偏导数 $f_x$ 和 $f_y$ 仍可偏导，则可以定义**二阶偏导数**：

$$\frac{\partial^2 z}{\partial x^2} = f_{xx} = \frac{\partial}{\partial x}\left(\frac{\partial z}{\partial x}\right)$$

$$\frac{\partial^2 z}{\partial y^2} = f_{yy} = \frac{\partial}{\partial y}\left(\frac{\partial z}{\partial y}\right)$$

$$\frac{\partial^2 z}{\partial x \partial y} = f_{xy} = \frac{\partial}{\partial y}\left(\frac{\partial z}{\partial x}\right)$$

$$\frac{\partial^2 z}{\partial y \partial x} = f_{yx} = \frac{\partial}{\partial x}\left(\frac{\partial z}{\partial y}\right)$$

其中 $f_{xy}$ 和 $f_{yx}$ 称为**混合偏导数**。

### 18.2.4 混合偏导数相等定理

**定理**（Schwarz 定理）：设函数 $z = f(x, y)$ 在区域 $D$ 内具有连续的二阶混合偏导数 $f_{xy}$ 和 $f_{yx}$，则在 $D$ 内

$$f_{xy} = f_{yx}$$

> **例题 18.5** 验证 $f(x, y) = x^3y + xy^3$ 满足混合偏导数相等。

**解**：

$$f_x = 3x^2y + y^3, \quad f_y = x^3 + 3xy^2$$

$$f_{xy} = \frac{\partial}{\partial y}(3x^2y + y^3) = 3x^2 + 3y^2$$

$$f_{yx} = \frac{\partial}{\partial x}(x^3 + 3xy^2) = 3x^2 + 3y^2$$

因此 $f_{xy} = f_{yx}$。

---

## 18.3 全微分

### 18.3.1 全微分的定义

**定义**：设函数 $z = f(x, y)$ 在点 $(x, y)$ 处的全增量

$$\Delta z = f(x + \Delta x, y + \Delta y) - f(x, y)$$

可以表示为

$$\Delta z = A\Delta x + B\Delta y + o(\rho)$$

其中 $A$、$B$ 是仅与 $x$、$y$ 有关的常数，$\rho = \sqrt{(\Delta x)^2 + (\Delta y)^2}$，$o(\rho)$ 是 $\rho$ 的高阶无穷小。则称 $f(x, y)$ 在点 $(x, y)$ 处**可微**，称 $A\Delta x + B\Delta y$ 为 $f(x, y)$ 在该点的**全微分**，记作

$$dz = A\,dx + B\,dy$$

**定理**：若 $f(x, y)$ 在点 $(x, y)$ 处可微，则 $A = f_x(x, y)$，$B = f_y(x, y)$。

因此全微分公式为：

$$dz = \frac{\partial z}{\partial x}dx + \frac{\partial z}{\partial y}dy$$

### 18.3.2 可微与可偏导的关系

**定理 1**：若函数在某点可微，则该点的偏导数必存在。

**定理 2**：偏导数存在不能保证函数可微。

**定理 3**（可微的充分条件）：若函数的偏导数在某点**连续**，则函数在该点可微。

> **例题 18.6** 求 $z = x^2y + y^3$ 的全微分。

**解**：

$$\frac{\partial z}{\partial x} = 2xy, \quad \frac{\partial z}{\partial y} = x^2 + 3y^2$$

因此

$$dz = 2xy\,dx + (x^2 + 3y^2)\,dy$$

### 18.3.3 全微分的几何意义

全微分 $dz$ 表示曲面 $z = f(x, y)$ 在点 $(x_0, y_0, z_0)$ 处的**切平面**上的增量，它是实际增量 $\Delta z$ 的线性主部。

切平面方程为：

$$z - z_0 = f_x(x_0, y_0)(x - x_0) + f_y(x_0, y_0)(y - y_0)$$

---

## 18.4 多元复合函数求导

### 18.4.1 链式法则

**定理**（链式法则）：设 $u = \varphi(t)$，$v = \psi(t)$ 在 $t$ 处可导，$z = f(u, v)$ 在对应点 $(u, v)$ 处偏导数连续，则复合函数 $z = f(\varphi(t), \psi(t))$ 在 $t$ 处可导，且

$$\frac{dz}{dt} = \frac{\partial z}{\partial u}\frac{du}{dt} + \frac{\partial z}{\partial v}\frac{dv}{dt}$$

**多元对多元的链式法则**：设 $z = f(u, v)$，$u = u(x, y)$，$v = v(x, y)$，则

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial x}$$

$$\frac{\partial z}{\partial y} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial y} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial y}$$

> **例题 18.7** 设 $z = e^{uv}$，$u = x + y$，$v = xy$，求 $\dfrac{\partial z}{\partial x}$。

**解**：

$$\frac{\partial z}{\partial u} = ve^{uv}, \quad \frac{\partial z}{\partial v} = ue^{uv}$$

$$\frac{\partial u}{\partial x} = 1, \quad \frac{\partial v}{\partial x} = y$$

由链式法则：

$$\frac{\partial z}{\partial x} = ve^{uv} \cdot 1 + ue^{uv} \cdot y = e^{uv}(v + uy) = e^{(x+y)xy}(xy + (x+y)y)$$

$$= e^{(x+y)xy}(xy + xy + y^2) = e^{(x+y)xy}(2xy + y^2)$$

### 18.4.2 全微分形式不变性

**定理**：无论 $u$、$v$ 是自变量还是中间变量，全微分的形式不变：

$$dz = \frac{\partial z}{\partial u}du + \frac{\partial z}{\partial v}dv$$

这个性质使得全微分在换元计算中非常方便。

---

## 18.5 方向导数与梯度

### 18.5.1 方向导数的定义

偏导数描述了函数沿坐标轴方向的变化率，方向导数则描述函数沿任意方向的变化率。

**定义**：设函数 $f(x, y)$ 在点 $P_0(x_0, y_0)$ 的某邻域内有定义，$\mathbf{l}$ 是从 $P_0$ 出发的单位方向向量，$\mathbf{l} = (\cos\alpha, \cos\beta)$。如果极限

$$\lim_{t \to 0^+} \frac{f(x_0 + t\cos\alpha, y_0 + t\cos\beta) - f(x_0, y_0)}{t}$$

存在，则称此极限为 $f$ 在点 $P_0$ 沿方向 $\mathbf{l}$ 的**方向导数**，记作 $\dfrac{\partial f}{\partial \mathbf{l}}$ 或 $D_{\mathbf{l}}f$。

**定理**：若 $f(x, y)$ 在点 $P_0$ 处可微，则沿任意方向 $\mathbf{l} = (\cos\alpha, \cos\beta)$ 的方向导数都存在，且

$$\frac{\partial f}{\partial \mathbf{l}} = f_x(x_0, y_0)\cos\alpha + f_y(x_0, y_0)\cos\beta$$

### 18.5.2 梯度

**定义**：设函数 $f(x, y)$ 在点 $P_0(x_0, y_0)$ 处具有一阶偏导数，称向量

$$\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right) = f_x\,\mathbf{i} + f_y\,\mathbf{j}$$

为 $f$ 在点 $P_0$ 处的**梯度**（gradient），记作 $\nabla f$ 或 $\text{grad}\,f$。

利用梯度，方向导数可以写成：

$$\frac{\partial f}{\partial \mathbf{l}} = \nabla f \cdot \mathbf{l} = |\nabla f| \cos\theta$$

其中 $\theta$ 是梯度与方向 $\mathbf{l}$ 的夹角。

### 18.5.3 梯度的几何意义

- 当 $\theta = 0$ 时，方向导数取得最大值 $|\nabla f|$，此时方向 $\mathbf{l}$ 与梯度同向
- 当 $\theta = \pi$ 时，方向导数取得最小值 $-|\nabla f|$
- 当 $\theta = \dfrac{\pi}{2}$ 时，方向导数为零

**结论**：**梯度的方向是函数增长最快的方向**，梯度的模是最大方向导数。

> **例题 18.8** 设 $f(x, y) = x^2 + y^2$，求在点 $(1, 2)$ 处沿从该点指向点 $(4, 6)$ 方向的方向导数。

**解**：方向向量为 $(4-1, 6-2) = (3, 4)$，单位方向向量为 $\mathbf{l} = \left(\dfrac{3}{5}, \dfrac{4}{5}\right)$。

$$f_x = 2x, \quad f_y = 2y$$

在点 $(1, 2)$ 处：$f_x(1, 2) = 2$，$f_y(1, 2) = 4$。

$$\frac{\partial f}{\partial \mathbf{l}} = 2 \cdot \frac{3}{5} + 4 \cdot \frac{4}{5} = \frac{6}{5} + \frac{16}{5} = \frac{22}{5}$$

---

## 18.6 隐函数定理

### 18.6.1 一个方程的隐函数

**隐函数定理**：设函数 $F(x, y)$ 满足：

1. $F(x_0, y_0) = 0$
2. $F(x, y)$ 在点 $(x_0, y_0)$ 的某邻域内具有连续偏导数
3. $F_y(x_0, y_0) \neq 0$

则方程 $F(x, y) = 0$ 在点 $(x_0, y_0)$ 附近唯一确定一个连续可微的隐函数 $y = f(x)$，且

$$\frac{dy}{dx} = -\frac{F_x}{F_y}$$

> **例题 18.9** 设 $x^2 + y^2 = 1$ 确定隐函数 $y = y(x)$，求 $\dfrac{dy}{dx}$。

**解**：设 $F(x, y) = x^2 + y^2 - 1$，则

$$F_x = 2x, \quad F_y = 2y$$

$$\frac{dy}{dx} = -\frac{F_x}{F_y} = -\frac{2x}{2y} = -\frac{x}{y} \quad (y \neq 0)$$

**三元情形**：方程 $F(x, y, z) = 0$ 在满足类似条件下确定隐函数 $z = z(x, y)$，且

$$\frac{\partial z}{\partial x} = -\frac{F_x}{F_z}, \quad \frac{\partial z}{\partial y} = -\frac{F_y}{F_z}$$

### 18.6.2 方程组的隐函数

**定理**：设方程组

$$\begin{cases} F(x, y, u, v) = 0 \\ G(x, y, u, v) = 0 \end{cases}$$

满足适当条件（函数连续可微，Jacobi行列式 $\dfrac{\partial(F, G)}{\partial(u, v)} \neq 0$），则方程组在某点附近确定隐函数 $u = u(x, y)$，$v = v(x, y)$，其偏导数可由以下方法求得：

对方程组两边分别对 $x$（或 $y$）求偏导，然后解关于 $\dfrac{\partial u}{\partial x}$、$\dfrac{\partial v}{\partial x}$ 的线性方程组。

---

## 本章小结

1. **多元函数的极限**要求点以任意方式趋近时函数值都趋于同一极限。若沿不同路径得到不同极限值，则极限不存在。

2. **偏导数**是固定其他变量、只对一个变量求导。$f_x$ 表示沿 $x$ 轴正向的变化率。

3. **高阶偏导数**：若二阶混合偏导数连续，则 $f_{xy} = f_{yx}$。

4. **全微分** $dz = f_x\,dx + f_y\,dy$ 是增量的线性主部。偏导数连续则函数可微。

5. **链式法则**：复合函数的偏导数等于各条"路径"的贡献之和。全微分形式不变性简化了计算。

6. **方向导数与梯度**：
   - 方向导数 $\dfrac{\partial f}{\partial \mathbf{l}} = \nabla f \cdot \mathbf{l}$
   - 梯度方向是函数增长最快的方向

7. **隐函数定理**提供了求隐函数导数的公式：$\dfrac{dy}{dx} = -\dfrac{F_x}{F_y}$。

---

## 深度学习应用

偏导数与梯度是深度学习的数学核心。本节展示第 18 章的概念如何直接映射到神经网络的训练与设计中。

### 18.7.1 多变量梯度与参数更新

神经网络的损失函数 $L$ 依赖于数千乃至数亿个参数 $\theta = (\theta_1, \theta_2, \ldots, \theta_n)$。对每个参数求偏导，构成**梯度向量**：

$$\nabla_\theta L = \left(\frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, \ldots, \frac{\partial L}{\partial \theta_n}\right)$$

梯度下降更新规则：

$$\theta_i \leftarrow \theta_i - \eta \frac{\partial L}{\partial \theta_i}, \quad i = 1, 2, \ldots, n$$

其中 $\eta$ 为学习率。每个参数根据其偏导数**独立更新**——这正是偏导数的核心意义：固定其他参数，衡量损失对该参数的敏感程度。

### 18.7.2 梯度的方向意义

由第 18.5 节的结论：

- **梯度 $\nabla_\theta L$ 指向损失函数增长最快的方向**
- **负梯度 $-\nabla_\theta L$ 是损失下降最快的方向**（梯度下降法的几何依据）
- 沿任意方向 $\mathbf{d}$（单位向量）的方向导数为 $D_{\mathbf{d}} L = \nabla_\theta L \cdot \mathbf{d}$

这解释了为何梯度下降优于随机方向搜索：负梯度方向是所有方向中损失下降最快的，即

$$\mathbf{d}^* = \arg\min_{\|\mathbf{d}\|=1} D_{\mathbf{d}} L = -\frac{\nabla_\theta L}{|\nabla_\theta L|}$$

动量方法（Momentum）、Adam 等优化器本质上是对该方向进行自适应修正。

### 18.7.3 Jacobian 矩阵与神经网络

神经网络每一层都是一个向量到向量的映射 $\mathbf{y} = f(\mathbf{x})$，其中 $\mathbf{x} \in \mathbb{R}^n$，$\mathbf{y} \in \mathbb{R}^m$。这个映射的完整导数信息由 **Jacobian 矩阵**给出：

$$J = \frac{\partial (y_1, y_2, \ldots, y_m)}{\partial (x_1, x_2, \ldots, x_n)} = \begin{pmatrix} \dfrac{\partial y_1}{\partial x_1} & \cdots & \dfrac{\partial y_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \dfrac{\partial y_m}{\partial x_1} & \cdots & \dfrac{\partial y_m}{\partial x_n} \end{pmatrix} \in \mathbb{R}^{m \times n}$$

反向传播本质上是对各层 Jacobian 矩阵的链式乘积（对应第 18.4 节的链式法则）：

$$\frac{\partial L}{\partial \mathbf{x}} = J^\top \frac{\partial L}{\partial \mathbf{y}}$$

对线性层 $\mathbf{y} = W\mathbf{x} + \mathbf{b}$，其 Jacobian 矩阵恰好等于权重矩阵 $W$。

### 18.7.4 隐函数定理与约束优化

许多神经网络训练问题带有约束，例如：

- 权重归一化：$\|\mathbf{w}\|_2 = 1$
- 公平性约束：$\mathbb{E}[\hat{y} \mid A=0] = \mathbb{E}[\hat{y} \mid A=1]$

**拉格朗日乘数法**（隐函数定理的直接应用）将约束优化转化为无约束问题：

$$\mathcal{L}(\theta, \lambda) = L(\theta) + \lambda \cdot g(\theta)$$

对 $\theta$ 和 $\lambda$ 分别求偏导并令其为零，即可求得约束极值。此外，隐函数定理保证了在满足约束的流形上，参数更新方向的存在性与唯一性。

### 18.7.5 代码示例：计算 Jacobian 矩阵

```python
import torch
import torch.nn as nn

# 计算 Jacobian 矩阵
def compute_jacobian(model, x):
    """计算 ∂y/∂x 的 Jacobian 矩阵"""
    x = x.requires_grad_(True)
    y = model(x)

    jacobian = []
    for i in range(y.shape[-1]):
        grad = torch.autograd.grad(y[..., i].sum(), x, retain_graph=True)[0]
        jacobian.append(grad)

    return torch.stack(jacobian, dim=-2)  # [batch, output_dim, input_dim]

# 示例
model = nn.Linear(3, 2)
x = torch.randn(1, 3)
J = compute_jacobian(model, x)
print(f"Jacobian shape: {J.shape}")  # [1, 2, 3]
print(f"Jacobian ≈ weight matrix:\n{J.squeeze()}")
print(f"Weight matrix:\n{model.weight}")
```

对线性层而言，Jacobian 矩阵与权重矩阵完全吻合，直观验证了上述理论推导。

---

## 练习题

**1.** 设 $f(x, y) = \ln(x + \sqrt{x^2 + y^2})$，求 $f_x$ 和 $f_y$。

**2.** 设 $z = \arctan\dfrac{y}{x}$，验证 $\dfrac{\partial^2 z}{\partial x^2} + \dfrac{\partial^2 z}{\partial y^2} = 0$（调和函数）。

**3.** 设 $z = f(x^2 - y^2, e^{xy})$，其中 $f$ 可微，求 $\dfrac{\partial z}{\partial x}$ 和 $\dfrac{\partial z}{\partial y}$。

**4.** 求函数 $f(x, y) = x^2 - xy + y^2$ 在点 $(1, 1)$ 处沿方向 $\mathbf{l} = \left(\dfrac{1}{\sqrt{2}}, \dfrac{1}{\sqrt{2}}\right)$ 的方向导数，并求该点处的梯度。

**5.** 设 $x^3 + y^3 - 3xy = 0$ 确定隐函数 $y = y(x)$，求 $\dfrac{dy}{dx}$ 和 $\dfrac{d^2y}{dx^2}$。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.** 设 $f(x, y) = \ln(x + \sqrt{x^2 + y^2})$。

$$f_x = \frac{1}{x + \sqrt{x^2 + y^2}} \cdot \left(1 + \frac{x}{\sqrt{x^2 + y^2}}\right) = \frac{1}{x + \sqrt{x^2 + y^2}} \cdot \frac{\sqrt{x^2 + y^2} + x}{\sqrt{x^2 + y^2}} = \frac{1}{\sqrt{x^2 + y^2}}$$

$$f_y = \frac{1}{x + \sqrt{x^2 + y^2}} \cdot \frac{y}{\sqrt{x^2 + y^2}} = \frac{y}{(x + \sqrt{x^2 + y^2})\sqrt{x^2 + y^2}}$$

---

**2.** 设 $z = \arctan\dfrac{y}{x}$。

$$\frac{\partial z}{\partial x} = \frac{1}{1 + \frac{y^2}{x^2}} \cdot \left(-\frac{y}{x^2}\right) = \frac{-y}{x^2 + y^2}$$

$$\frac{\partial z}{\partial y} = \frac{1}{1 + \frac{y^2}{x^2}} \cdot \frac{1}{x} = \frac{x}{x^2 + y^2}$$

$$\frac{\partial^2 z}{\partial x^2} = \frac{\partial}{\partial x}\left(\frac{-y}{x^2 + y^2}\right) = \frac{2xy}{(x^2 + y^2)^2}$$

$$\frac{\partial^2 z}{\partial y^2} = \frac{\partial}{\partial y}\left(\frac{x}{x^2 + y^2}\right) = \frac{-2xy}{(x^2 + y^2)^2}$$

因此 $\dfrac{\partial^2 z}{\partial x^2} + \dfrac{\partial^2 z}{\partial y^2} = \dfrac{2xy}{(x^2 + y^2)^2} - \dfrac{2xy}{(x^2 + y^2)^2} = 0$。

---

**3.** 设 $z = f(u, v)$，其中 $u = x^2 - y^2$，$v = e^{xy}$。

$$\frac{\partial z}{\partial x} = f_u \cdot \frac{\partial u}{\partial x} + f_v \cdot \frac{\partial v}{\partial x} = f_u \cdot 2x + f_v \cdot ye^{xy} = 2xf_u + ye^{xy}f_v$$

$$\frac{\partial z}{\partial y} = f_u \cdot \frac{\partial u}{\partial y} + f_v \cdot \frac{\partial v}{\partial y} = f_u \cdot (-2y) + f_v \cdot xe^{xy} = -2yf_u + xe^{xy}f_v$$

其中 $f_u = f_1(x^2 - y^2, e^{xy})$，$f_v = f_2(x^2 - y^2, e^{xy})$。

---

**4.** $f(x, y) = x^2 - xy + y^2$

$$f_x = 2x - y, \quad f_y = -x + 2y$$

在点 $(1, 1)$ 处：$f_x(1, 1) = 1$，$f_y(1, 1) = 1$。

方向导数：

$$\frac{\partial f}{\partial \mathbf{l}} = f_x \cdot \frac{1}{\sqrt{2}} + f_y \cdot \frac{1}{\sqrt{2}} = 1 \cdot \frac{1}{\sqrt{2}} + 1 \cdot \frac{1}{\sqrt{2}} = \frac{2}{\sqrt{2}} = \sqrt{2}$$

梯度：$\nabla f(1, 1) = (1, 1)$。

---

**5.** 设 $F(x, y) = x^3 + y^3 - 3xy$，则 $F_x = 3x^2 - 3y$，$F_y = 3y^2 - 3x$。

$$\frac{dy}{dx} = -\frac{F_x}{F_y} = -\frac{3x^2 - 3y}{3y^2 - 3x} = \frac{y - x^2}{y^2 - x}$$

对 $\dfrac{dy}{dx} = \dfrac{y - x^2}{y^2 - x}$ 两边再对 $x$ 求导：

$$\frac{d^2y}{dx^2} = \frac{(y' - 2x)(y^2 - x) - (y - x^2)(2yy' - 1)}{(y^2 - x)^2}$$

将 $y' = \dfrac{y - x^2}{y^2 - x}$ 代入并化简，得

$$\frac{d^2y}{dx^2} = \frac{-2xy(x^3 + y^3 - 3xy) + 2(x^3 - y^3)}{(y^2 - x)^3}$$

由于 $x^3 + y^3 - 3xy = 0$（原方程），故

$$\frac{d^2y}{dx^2} = \frac{2(x^3 - y^3)}{(y^2 - x)^3} = \frac{2xy}{(y^2 - x)^3}$$

（最后一步利用 $x^3 - y^3 = 3xy - 2y^3 - x^3 + y^3 = ...$，具体化简需要利用原方程）

</details>
