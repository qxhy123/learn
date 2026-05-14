# 多元链式与梯度

> **一例速记**：设 $z = \sin(xy^2)$，求 $\dfrac{\partial z}{\partial x}$ 和 $\dfrac{\partial z}{\partial y}$。
> 视 $y$ 为常数，对 $x$ 求导：$\dfrac{\partial z}{\partial x} = \cos(xy^2) \cdot y^2$（链式：外层 $\sin$ 求导，内层 $xy^2$ 对 $x$ 求导）。
> 视 $x$ 为常数，对 $y$ 求导：$\dfrac{\partial z}{\partial y} = \cos(xy^2) \cdot 2xy$。
> **偏导数 = "把其他变量当常数，对目标变量用单变量链式"**。

---

## 一、为什么多元链式是 AI 时代最重要的微积分工具

单变量链式法则 $\dfrac{dz}{dx} = \dfrac{dz}{du} \cdot \dfrac{du}{dx}$ 已经够用了吗？在机器学习的世界里远远不够。

神经网络是一个多层复合函数：输入经过线性变换 + 激活函数 → 中间层 → 再线性变换 + 激活 → 最终输出损失值。损失函数对**每一个参数**（权重矩阵的每个元素）的偏导数，需要沿着整个计算图反向传播。这个过程——**反向传播（Backpropagation）**——在数学上就是多元链式法则的递归应用。

掌握多元链式法则，你就掌握了理解反向传播的基础语言。

---

## 二、多元链式法则：树形依赖图

### 2.1 基本形式

设 $z = f(u, v)$，而 $u = u(x, y)$，$v = v(x, y)$（所有函数均可微），则：

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial x} + \frac{\partial z}{\partial v} \cdot \frac{\partial v}{\partial x}$$

$$\frac{\partial z}{\partial y} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial y} + \frac{\partial z}{\partial v} \cdot \frac{\partial v}{\partial y}$$

**口诀**：$z$ 对 $x$ 的偏导 = （$z$ 对 $u$ 的偏导）$\times$（$u$ 对 $x$ 的偏导）+ （$z$ 对 $v$ 的偏导）$\times$（$v$ 对 $x$ 的偏导）。对**每条从 $z$ 到 $x$ 的路径**，沿路径做乘积，再把所有路径的结果相加。

### 2.2 树形依赖图（"求导路由图"）

把变量依赖关系画成树：

```
        z
       / \
      u   v
     / \ / \
    x   y  x   y
```

（$u$ 依赖 $x, y$；$v$ 依赖 $x, y$；$z$ 依赖 $u, v$）

**规则**：
- 沿某条路径（从 $z$ 到 $x$），把路径上每段的偏导数**相乘**
- 所有从 $z$ 到 $x$ 的路径，把结果**相加**

这就是 $\dfrac{\partial z}{\partial x}$ 的计算方法。

### 2.3 更一般的情形

若 $z = f(u_1, u_2, \ldots, u_m)$，每个 $u_i = u_i(x_1, x_2, \ldots, x_n)$，则：

$$\frac{\partial z}{\partial x_j} = \sum_{i=1}^{m} \frac{\partial z}{\partial u_i} \cdot \frac{\partial u_i}{\partial x_j}$$

这是矩阵乘法的形式——事实上，这正是 **Jacobian 矩阵相乘**的规则。

### 2.4 常见特殊情形

| 情形 | 公式 | 说明 |
|---|---|---|
| $z = f(u)$，$u = u(x)$（单变量链式）| $\dfrac{dz}{dx} = f'(u) \cdot u'(x)$ | 最基本形式 |
| $z = f(x, y)$，$x = x(t)$，$y = y(t)$ | $\dfrac{dz}{dt} = \dfrac{\partial z}{\partial x}\dfrac{dx}{dt} + \dfrac{\partial z}{\partial y}\dfrac{dy}{dt}$ | 参数曲线上的全导数 |
| $z = f(u, v)$，$u = x$，$v = y$（无中间层）| $\dfrac{\partial z}{\partial x} = \dfrac{\partial f}{\partial x}$（直接偏导）| 退化为普通偏导数 |
| 隐函数 $F(x, y, z) = 0$ | $\dfrac{\partial z}{\partial x} = -\dfrac{F_x}{F_z}$，$\dfrac{\partial z}{\partial y} = -\dfrac{F_y}{F_z}$ | 由链式法则推导 |

**隐函数求导的推导**：$F(x, y, z) = 0$ 两边对 $x$ 求偏导，把 $z$ 视为 $x, y$ 的函数：

$$F_x + F_z \cdot \frac{\partial z}{\partial x} = 0 \implies \frac{\partial z}{\partial x} = -\frac{F_x}{F_z}$$

---

## 三、梯度：方向导数的最大化

### 3.1 方向导数

函数 $f(x, y)$ 在点 $P_0$ 沿单位向量 $\vec{u} = (\cos\theta, \sin\theta)$ 方向的**方向导数**定义为：

$$D_{\vec{u}} f = \lim_{t \to 0^+} \frac{f(P_0 + t\vec{u}) - f(P_0)}{t}$$

若 $f$ 在 $P_0$ 处可微，则方向导数可以用梯度表达：

$$D_{\vec{u}} f = \nabla f \cdot \vec{u} = |\nabla f| \cos\alpha$$

其中 $\alpha$ 是梯度 $\nabla f$ 与方向 $\vec{u}$ 的夹角。

### 3.2 梯度的几何意义

**梯度** $\nabla f = \left(\dfrac{\partial f}{\partial x},\, \dfrac{\partial f}{\partial y}\right)$（三元时加 $\dfrac{\partial f}{\partial z}$）有两个关键性质：

1. **梯度指向函数值增加最快的方向**：当 $\vec{u}$ 与 $\nabla f$ 同向（$\alpha = 0$），$D_{\vec{u}} f = |\nabla f|$ 取最大值。

2. **梯度垂直于等值线**（二元）/ **等值面**（三元）：在等值线 $f(x,y) = C$ 上移动时，$f$ 不变，故沿等值线方向的方向导数为 $0$，由 $\nabla f \cdot \vec{u} = 0$ 得梯度与等值线正交。

| 方向 | 方向导数 | 说明 |
|---|---|---|
| 沿 $\nabla f$（梯度方向）| $|\nabla f|$（最大）| 函数值增加最快 |
| 沿 $-\nabla f$（负梯度方向）| $-|\nabla f|$（最小）| 函数值减小最快 |
| 沿等值线切线方向 | $0$ | 函数值不变 |

### 3.3 直觉：山地地图类比

把 $f(x, y)$ 想象成地形图的海拔：
- **等值线** = 等高线（地图上的圆圈）
- **梯度方向** = 最陡上坡方向，总是垂直于等高线向外
- **梯度长度** = 坡度（梯度越大，坡越陡）
- **负梯度方向** = 最速下山方向（梯度下降算法的直觉来源）

---

## 四、Jacobian 矩阵：多元函数的线性近似

### 4.1 定义

设 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$，即 $\mathbf{f} = (f_1, f_2, \ldots, f_m)$，每个 $f_i$ 是 $x_1, \ldots, x_n$ 的函数，则 **Jacobian 矩阵**为 $m \times n$ 矩阵：

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{pmatrix} \dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\[6pt] \vdots & \ddots & \vdots \\[4pt] \dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n} \end{pmatrix}$$

特别地，当 $m = 1$（标量函数）时，Jacobian 退化为行向量 $(\partial f/\partial x_1, \ldots, \partial f/\partial x_n)$，即梯度的转置。

### 4.2 几何意义：最佳线性近似

Jacobian 矩阵是多元函数在 $\mathbf{x}_0$ 处的"最佳线性近似"：

$$\mathbf{f}(\mathbf{x}_0 + \delta) \approx \mathbf{f}(\mathbf{x}_0) + J(\mathbf{x}_0)\,\delta$$

这是单变量一阶 Taylor 近似 $f(x_0 + h) \approx f(x_0) + f'(x_0)h$ 的多元推广。$J\delta$ 是矩阵与向量的乘积，$J$ 就是"多元导数"。

### 4.3 链式法则的矩阵形式

若 $\mathbf{g}: \mathbb{R}^p \to \mathbb{R}^n$，$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$，$\mathbf{h} = \mathbf{f} \circ \mathbf{g}$，则

$$J_{\mathbf{h}} = J_{\mathbf{f}} \cdot J_{\mathbf{g}}$$

Jacobian 矩阵的链式法则就是**矩阵乘法**。这是神经网络反向传播中"误差沿层传播"的数学基础。

---

## 五、AI 应用：反向传播 = 链式法则的递归

### 神经网络的正向传播

考虑两层神经网络（简化）：

$$\mathbf{a}^{(1)} = W^{(1)} \mathbf{x} + \mathbf{b}^{(1)}, \quad \mathbf{h} = \sigma(\mathbf{a}^{(1)})$$
$$\mathbf{a}^{(2)} = W^{(2)} \mathbf{h} + \mathbf{b}^{(2)}, \quad \hat{y} = \text{softmax}(\mathbf{a}^{(2)})$$
$$L = \text{CrossEntropy}(\hat{y}, y)$$

这是一个复合函数链：$L \leftarrow \hat{y} \leftarrow \mathbf{a}^{(2)} \leftarrow \mathbf{h} \leftarrow \mathbf{a}^{(1)} \leftarrow \mathbf{x}, W^{(1)}, W^{(2)}$。

### 反向传播的链式展开

要求 $\dfrac{\partial L}{\partial W^{(1)}}$，需要沿整条链反向应用链式法则：

$$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \mathbf{a}^{(2)}} \cdot \frac{\partial \mathbf{a}^{(2)}}{\partial \mathbf{h}} \cdot \frac{\partial \mathbf{h}}{\partial \mathbf{a}^{(1)}} \cdot \frac{\partial \mathbf{a}^{(1)}}{\partial W^{(1)}}$$

每一个因子都是一个 Jacobian 矩阵（或其特殊形式）。反向传播算法的高效之处在于：从右往左计算，每个中间结果（"误差信号"）只需计算一次，就可以传递给前一层，避免重复计算。

### 自动微分

现代深度学习框架（PyTorch, TensorFlow）实现了**自动微分**：在正向传播时记录计算图（每个操作及其输入），反向传播时沿计算图自动应用链式法则，无需人工推导偏导数公式。这本质上是**计算机执行多元链式法则**。

---

## 六、演示题：链式法则求偏导

**题目**：设 $z = e^{u^2 + v}$，$u = x\cos y$，$v = x\sin y$，求 $\dfrac{\partial z}{\partial x}$。

> **画依赖图。** $z$ 依赖 $u, v$；$u, v$ 各依赖 $x, y$。路径从 $z$ 到 $x$ 有两条：$z \to u \to x$ 和 $z \to v \to x$。
>
> **写出链式公式。**
>
> $$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial x} + \frac{\partial z}{\partial v} \cdot \frac{\partial v}{\partial x}.$$
>
> **分别计算各偏导数。**
>
> $$\frac{\partial z}{\partial u} = e^{u^2 + v} \cdot 2u, \quad \frac{\partial z}{\partial v} = e^{u^2 + v} \cdot 1.$$
>
> $$\frac{\partial u}{\partial x} = \cos y, \quad \frac{\partial v}{\partial x} = \sin y.$$
>
> **代入链式公式。**
>
> $$\frac{\partial z}{\partial x} = e^{u^2 + v}(2u\cos y + \sin y).$$
>
> **回代 $u, v$。** $u = x\cos y$，$v = x\sin y$，故
>
> $$\frac{\partial z}{\partial x} = e^{x^2\cos^2 y + x\sin y}\left(2x\cos^2 y + \sin y\right).$$
>
> **回顾。** 两条路径的贡献分别是 $2u\cos y$（通过 $u$ 路径）和 $\sin y$（通过 $v$ 路径），相加得结果。如果只走了一条路径，就漏掉了一半——这是初学者最常犯的错误。

---

## 七、思考路标

**路标 1**：看到复合函数 $z = f(u, v)$，且 $u, v$ 又是 $x, y$ 的函数 → 立刻画树形依赖图，数从 $z$ 到目标变量有几条路径，每条路径相乘，所有路径相加。

**路标 2**：看到隐函数 $F(x, y, z) = 0$ 要求 $\partial z / \partial x$ → 公式 $-F_x / F_z$（前提：$F_z \neq 0$）。分子是"直接对 $x$ 的偏导"，分母是"对 $z$ 的偏导"，取负号。

**路标 3**：看到"梯度"二字 → 想到**最速上升方向**、**垂直于等值线**、**长度等于该方向的方向导数最大值**。这三个性质是梯度的三位一体。

**路标 4**：看到"方向导数" → 公式 $D_{\vec{u}} f = \nabla f \cdot \vec{u}$（$\vec{u}$ 必须是**单位向量**！）。若题目给的方向不是单位向量，先归一化。

**路标 5**：看到 $\dfrac{\partial z}{\partial u} \cdot \dfrac{\partial u}{\partial x}$ 的形式 → 问自己：**所有的路径都走到了吗**？常见漏项：$z$ 通过多个中间变量依赖 $x$，只写了一条路径。

**路标 6**：看到 Jacobian 矩阵 → 联想到"$\mathbf{f}(\mathbf{x}_0 + \delta) \approx \mathbf{f}(\mathbf{x}_0) + J\delta$"（线性近似），它是多元函数的"导数矩阵"，链式法则对应矩阵乘法。

**路标 7**：看到神经网络梯度计算（反向传播）→ 本质是从损失函数 $L$ 出发，沿计算图反向逐层应用链式法则。每一层的 Jacobian 乘以上层传来的梯度，得到本层的梯度。

---

## 八、典型应用 3 例

### 例 1：链式法则求全导数

**题目**：设 $f(x, y) = x^2 + xy + y^2$，$x = e^t$，$y = \ln t$（$t > 0$），求 $\dfrac{df}{dt}$。

**分析**：$\dfrac{df}{dt} = \dfrac{\partial f}{\partial x}\dfrac{dx}{dt} + \dfrac{\partial f}{\partial y}\dfrac{dy}{dt}$。

$\dfrac{\partial f}{\partial x} = 2x + y,\quad \dfrac{\partial f}{\partial y} = x + 2y.$

$\dfrac{dx}{dt} = e^t,\quad \dfrac{dy}{dt} = \dfrac{1}{t}.$

$$\frac{df}{dt} = (2x + y)e^t + (x + 2y)\frac{1}{t}.$$

回代 $x = e^t$，$y = \ln t$：

$$\frac{df}{dt} = (2e^t + \ln t)e^t + (e^t + 2\ln t)\frac{1}{t} = 2e^{2t} + e^t\ln t + \frac{e^t}{t} + \frac{2\ln t}{t}.$$

---

### 例 2：方向导数与最速上升

**题目**：函数 $f(x, y) = x^2 - 2xy + y^2$ 在点 $(1, 2)$ 处，沿哪个方向变化率最大？最大变化率是多少？

**分析**：

$$\nabla f = (2x - 2y,\; -2x + 2y).$$

在 $(1, 2)$：$\nabla f(1, 2) = (2 \cdot 1 - 2 \cdot 2,\; -2 \cdot 1 + 2 \cdot 2) = (-2, 2).$

最速上升方向：$(-2, 2)$（或单位化为 $(-1/\sqrt{2}, 1/\sqrt{2})$）。

最大变化率 $= |\nabla f| = \sqrt{(-2)^2 + 2^2} = \sqrt{8} = 2\sqrt{2}.$

注意：$f(x,y) = (x-y)^2$ 是两变量之差的平方，在 $(1,2)$ 处 $f = 1$；梯度方向是"最快增大"的方向，即沿着 $(-1, 1)$ 方向移动时 $(x-y)^2$ 增大最快（让差更负）。

---

### 例 3：Jacobian 矩阵

**题目**：设 $f_1 = x^2 + y$，$f_2 = xy - y^2$，写出 $\mathbf{f} = (f_1, f_2)$ 在点 $(1, -1)$ 处的 Jacobian 矩阵，并用它近似 $\mathbf{f}(1.1, -0.9)$。

**Jacobian 矩阵：**

$$J = \begin{pmatrix} \partial f_1/\partial x & \partial f_1/\partial y \\ \partial f_2/\partial x & \partial f_2/\partial y \end{pmatrix} = \begin{pmatrix} 2x & 1 \\ y & x - 2y \end{pmatrix}.$$

在 $(1, -1)$：$J(1,-1) = \begin{pmatrix} 2 & 1 \\ -1 & 3 \end{pmatrix}.$

**线性近似：** $\delta = (0.1, 0.1)$（从 $(1,-1)$ 到 $(1.1,-0.9)$）。

$$\mathbf{f}(1,-1) = (1^2 + (-1),\; 1\cdot(-1) - (-1)^2) = (0, -2).$$

$$\mathbf{f}(1.1, -0.9) \approx \mathbf{f}(1,-1) + J \cdot \delta = \begin{pmatrix} 0 \\ -2 \end{pmatrix} + \begin{pmatrix} 2 & 1 \\ -1 & 3 \end{pmatrix}\begin{pmatrix} 0.1 \\ 0.1 \end{pmatrix} = \begin{pmatrix} 0 \\ -2 \end{pmatrix} + \begin{pmatrix} 0.3 \\ 0.2 \end{pmatrix} = \begin{pmatrix} 0.3 \\ -1.8 \end{pmatrix}.$$

验证：精确值 $f_1(1.1, -0.9) = 1.21 - 0.9 = 0.31$，$f_2(1.1,-0.9) = -0.99 - 0.81 = -1.80$。近似误差约 $0.01$，符合预期（误差为 $|\delta|^2$ 量级）。

---

## 九、自测题

**第 1 题**：设 $z = \ln(u^2 + v^2)$，$u = x + y$，$v = x - y$，求 $\dfrac{\partial z}{\partial x} + \dfrac{\partial z}{\partial y}$。

💡 提示：$\dfrac{\partial z}{\partial x} = \dfrac{2u}{u^2+v^2}\cdot 1 + \dfrac{2v}{u^2+v^2} \cdot 1 = \dfrac{2(u+v)}{u^2+v^2}$；同理 $\dfrac{\partial z}{\partial y} = \dfrac{2(u-v)}{u^2+v^2}$；两式相加 $= \dfrac{4u}{u^2+v^2} = \dfrac{4(x+y)}{(x+y)^2+(x-y)^2} = \dfrac{4(x+y)}{2(x^2+y^2)} = \dfrac{2(x+y)}{x^2+y^2}$。

**第 2 题**：设 $F(x, y, z) = x^2 + y^2 + z^2 - 1 = 0$ 定义了隐函数 $z = z(x, y)$（上半球面），求 $\dfrac{\partial z}{\partial x}$。

💡 提示：$\dfrac{\partial z}{\partial x} = -\dfrac{F_x}{F_z} = -\dfrac{2x}{2z} = -\dfrac{x}{z}$（$z > 0$）。几何意义：在球面上，沿 $x$ 方向移动时，$z$ 的变化率是 $-x/z$。

**第 3 题**：$f(x, y) = x^3 + y^3 - 3xy$ 在哪些点的梯度为零？这些点叫什么？

💡 提示：$\nabla f = (3x^2 - 3y, 3y^2 - 3x) = (0, 0)$，即 $x^2 = y$ 且 $y^2 = x$，解得 $(0,0)$ 和 $(1,1)$。这些点是**驻点**（极值候选点），需用 Hessian 进一步判断极大/极小/鞍点。

**第 4 题**：函数 $f(x, y) = 2x^2 + y^2$ 在点 $(1, 1)$ 处，沿方向 $\vec{l} = (1, -1)$ 的方向导数是多少？

💡 提示：先归一化：$\vec{u} = (1, -1)/\sqrt{2}$。$\nabla f = (4x, 2y)$，在 $(1,1)$：$\nabla f = (4, 2)$。$D_{\vec{u}} f = (4, 2) \cdot (1/\sqrt{2}, -1/\sqrt{2}) = (4 - 2)/\sqrt{2} = \sqrt{2}$。

**第 5 题**（AI 向）：设神经网络一层的映射是 $\mathbf{h} = \sigma(W\mathbf{x})$，其中 $\sigma$ 逐元素应用（如 ReLU 或 sigmoid）。若损失 $L$ 对 $\mathbf{h}$ 的梯度为 $\delta = \partial L / \partial \mathbf{h}$，写出 $\partial L / \partial W$ 的表达式（用链式法则）。

💡 提示：$\dfrac{\partial L}{\partial W_{ij}} = \sum_k \dfrac{\partial L}{\partial h_k} \cdot \dfrac{\partial h_k}{\partial W_{ij}}$。$h_k = \sigma\!\left(\sum_j W_{kj} x_j\right)$，$\dfrac{\partial h_k}{\partial W_{ij}} = \sigma'(a_k)\, x_j\, \mathbf{1}_{k=i}$，故 $\dfrac{\partial L}{\partial W_{ij}} = \delta_i \sigma'(a_i) x_j$，矩阵形式为 $\partial L / \partial W = (\delta \odot \sigma'(\mathbf{a}))\, \mathbf{x}^\top$。
