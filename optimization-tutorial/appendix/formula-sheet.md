# 附录 A：公式速查表

本附录汇总优化理论与算法中最常用的公式，供快速查阅。所有向量默认为列向量，$\mathbb{R}^n$ 上的内积为标准欧氏内积。

---

## A.1 向量与矩阵公式

### 向量范数

| 范数 | 定义 | 说明 |
|------|------|------|
| $\ell_1$ 范数 | $\|x\|_1 = \sum_{i=1}^n |x_i|$ | 绝对值之和 |
| $\ell_2$ 范数（欧氏） | $\|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{x^\top x}$ | 默认范数 |
| $\ell_p$ 范数 | $\|x\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}$，$p \geq 1$ | |
| $\ell_\infty$ 范数 | $\|x\|_\infty = \max_i |x_i|$ | $p \to \infty$ 的极限 |

**范数等价关系**（$x \in \mathbb{R}^n$）：

$$\|x\|_\infty \leq \|x\|_2 \leq \|x\|_1 \leq \sqrt{n}\,\|x\|_2 \leq n\,\|x\|_\infty$$

### 矩阵范数

| 范数 | 定义 |
|------|------|
| Frobenius 范数 | $\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\operatorname{tr}(A^\top A)}$ |
| 谱范数（算子 $\ell_2$ 范数） | $\|A\|_2 = \sigma_{\max}(A)$（最大奇异值） |
| 算子 $\ell_1$ 范数 | $\|A\|_1 = \max_j \sum_i |a_{ij}|$（最大列绝对值和） |
| 算子 $\ell_\infty$ 范数 | $\|A\|_\infty = \max_i \sum_j |a_{ij}|$（最大行绝对值和） |
| 核范数 | $\|A\|_* = \sum_i \sigma_i(A)$（奇异值之和） |

**Frobenius 范数与迹的关系**：

$$\|A\|_F^2 = \operatorname{tr}(A^\top A) = \operatorname{tr}(AA^\top) = \sum_i \sigma_i^2(A)$$

**谱范数与 Frobenius 范数的关系**：

$$\|A\|_2 \leq \|A\|_F \leq \sqrt{\operatorname{rank}(A)}\,\|A\|_2$$

### 内积

**向量内积**：

$$\langle x, y \rangle = x^\top y = \sum_{i=1}^n x_i y_i$$

**矩阵内积**：

$$\langle A, B \rangle = \operatorname{tr}(A^\top B) = \sum_{i,j} a_{ij} b_{ij}$$

### Cauchy-Schwarz 不等式

$$|\langle x, y \rangle| \leq \|x\|_2 \|y\|_2$$

等号成立当且仅当 $x$ 与 $y$ 线性相关。

**Hölder 推广形式**（$\frac{1}{p} + \frac{1}{q} = 1$）：

$$|\langle x, y \rangle| \leq \|x\|_p \|y\|_q$$

### 迹的性质

$$\operatorname{tr}(A) = \sum_i a_{ii} = \sum_i \lambda_i(A)$$

$$\operatorname{tr}(AB) = \operatorname{tr}(BA), \quad \operatorname{tr}(ABC) = \operatorname{tr}(CAB) = \operatorname{tr}(BCA)$$

$$\operatorname{tr}(A^\top) = \operatorname{tr}(A), \quad \operatorname{tr}(\alpha A + \beta B) = \alpha\operatorname{tr}(A) + \beta\operatorname{tr}(B)$$

$$x^\top A x = \operatorname{tr}(x^\top A x) = \operatorname{tr}(A x x^\top)$$

---

## A.2 微分公式

### 梯度——常见函数

设 $f : \mathbb{R}^n \to \mathbb{R}$，$A \in \mathbb{R}^{n \times n}$，$b \in \mathbb{R}^n$：

| 函数 $f(x)$ | 梯度 $\nabla f(x)$ |
|-------------|-------------------|
| $a^\top x$ | $a$ |
| $x^\top A x$（$A$ 对称） | $2Ax$ |
| $x^\top A x$（$A$ 一般） | $(A + A^\top)x$ |
| $\|x\|_2^2 = x^\top x$ | $2x$ |
| $\|Ax - b\|_2^2$ | $2A^\top(Ax - b)$ |
| $\|x\|_2$ | $x / \|x\|_2$（$x \neq 0$） |
| $\log(a^\top x)$ | $a / (a^\top x)$ |
| $\exp(a^\top x)$ | $a\exp(a^\top x)$ |

设 $F : \mathbb{R}^{m \times n} \to \mathbb{R}$，$X \in \mathbb{R}^{m \times n}$：

| 函数 $F(X)$ | 梯度 $\nabla_X F$ |
|-------------|------------------|
| $\operatorname{tr}(A^\top X)$ | $A$ |
| $\operatorname{tr}(X^\top A X)$ | $(A + A^\top)X$ |
| $\|X\|_F^2$ | $2X$ |
| $\log\det(X)$（$X \succ 0$） | $X^{-1}$ |
| $\operatorname{tr}(X^{-1}A)$（$X \succ 0$） | $-X^{-\top}AX^{-\top}$ |

### 链式法则

**标量复合**：$h(x) = f(g(x))$，其中 $g : \mathbb{R}^n \to \mathbb{R}^m$，$f : \mathbb{R}^m \to \mathbb{R}$：

$$\nabla h(x) = J_g(x)^\top \nabla f(g(x))$$

其中 $J_g(x) \in \mathbb{R}^{m \times n}$ 是 $g$ 的 Jacobian 矩阵。

**向量复合**：$h(x) = f(g(x))$，其中 $f : \mathbb{R}^m \to \mathbb{R}^k$：

$$J_h(x) = J_f(g(x)) \cdot J_g(x)$$

### Hessian 矩阵

$$H_f(x) = \nabla^2 f(x) \in \mathbb{R}^{n \times n}, \quad [H_f(x)]_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

**常见 Hessian**：

| 函数 $f(x)$ | Hessian $\nabla^2 f(x)$ |
|-------------|------------------------|
| $x^\top A x$（$A$ 对称） | $2A$ |
| $\|Ax - b\|_2^2$ | $2A^\top A$ |
| $\log(1 + e^{a^\top x})$（逻辑损失） | $\sigma(1-\sigma) a a^\top$，$\sigma = \sigma(a^\top x)$ |

### Jacobian 矩阵

$f : \mathbb{R}^n \to \mathbb{R}^m$：

$$J_f(x) = \frac{\partial f}{\partial x} \in \mathbb{R}^{m \times n}, \quad [J_f(x)]_{ij} = \frac{\partial f_i}{\partial x_j}$$

### 方向导数

$$D_v f(x) = \langle \nabla f(x), v \rangle = \lim_{t \to 0} \frac{f(x + tv) - f(x)}{t}$$

### 泰勒展开

**一阶**：

$$f(y) = f(x) + \nabla f(x)^\top (y - x) + O(\|y - x\|^2)$$

**二阶**：

$$f(y) = f(x) + \nabla f(x)^\top (y - x) + \frac{1}{2}(y-x)^\top \nabla^2 f(x)(y-x) + O(\|y-x\|^3)$$

**二阶上界**（$\nabla^2 f \preceq LI$）：

$$f(y) \leq f(x) + \nabla f(x)^\top (y-x) + \frac{L}{2}\|y-x\|^2$$

---

## A.3 凸函数性质

### 凸函数定义

$f$ 为凸函数当且仅当对所有 $x, y \in \operatorname{dom}(f)$，$\theta \in [0,1]$：

$$f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$$

### 一阶凸性条件（可微时）

$$f \text{ 是凸函数} \iff f(y) \geq f(x) + \nabla f(x)^\top(y - x), \quad \forall x, y$$

### 二阶凸性条件（二阶可微时）

$$f \text{ 是凸函数} \iff \nabla^2 f(x) \succeq 0, \quad \forall x$$

$$f \text{ 是严格凸函数} \Leftarrow \nabla^2 f(x) \succ 0, \quad \forall x$$

### $\mu$-强凸性

$f$ 是 $\mu$-强凸（$\mu > 0$）当且仅当以下等价条件之一成立：

$$f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y) - \frac{\mu}{2}\theta(1-\theta)\|x-y\|^2$$

$$f(y) \geq f(x) + \nabla f(x)^\top(y-x) + \frac{\mu}{2}\|y-x\|^2, \quad \forall x, y$$

$$\nabla^2 f(x) \succeq \mu I, \quad \forall x$$

$$\langle \nabla f(x) - \nabla f(y), x - y \rangle \geq \mu \|x - y\|^2, \quad \forall x, y$$

**强凸性推论**：最优解唯一，且：

$$f(x) - f(x^*) \geq \frac{\mu}{2}\|x - x^*\|^2$$

### $L$-光滑性（梯度 Lipschitz）

$f$ 的梯度是 $L$-Lipschitz 连续（等价定义）：

$$\|\nabla f(x) - \nabla f(y)\|_2 \leq L\|x - y\|_2, \quad \forall x, y$$

$$f(y) \leq f(x) + \nabla f(x)^\top(y-x) + \frac{L}{2}\|y-x\|^2, \quad \forall x, y$$

$$\nabla^2 f(x) \preceq LI, \quad \forall x \quad \text{（二阶可微时）}$$

$$\langle \nabla f(x) - \nabla f(y), x-y \rangle \leq L\|x-y\|^2, \quad \forall x, y$$

**条件数**：$\kappa = L / \mu$（$\mu$-强凸且 $L$-光滑时）。

### 次梯度

$g \in \mathbb{R}^n$ 是凸函数 $f$ 在 $x$ 处的次梯度，若：

$$f(y) \geq f(x) + g^\top(y - x), \quad \forall y$$

次微分 $\partial f(x) = \{g : g \text{ 是 } f \text{ 在 } x \text{ 处的次梯度}\}$。

---

## A.4 最优性条件

### 无约束优化

**一阶必要条件**（$x^*$ 是局部极小值，$f$ 可微）：

$$\nabla f(x^*) = 0$$

**二阶必要条件**（$f$ 二阶可微）：

$$\nabla f(x^*) = 0 \quad \text{且} \quad \nabla^2 f(x^*) \succeq 0$$

**二阶充分条件**：

$$\nabla f(x^*) = 0 \quad \text{且} \quad \nabla^2 f(x^*) \succ 0 \implies x^* \text{ 是严格局部极小值}$$

**凸函数的全局最优性**：

$$\nabla f(x^*) = 0 \iff x^* \text{ 是全局极小值}$$

### 等式约束优化

**问题**：$\min f(x)$，s.t. $h(x) = 0$，$h : \mathbb{R}^n \to \mathbb{R}^p$。

**Lagrangian**：$\mathcal{L}(x, \nu) = f(x) + \nu^\top h(x)$

**KKT 一阶必要条件**（LICQ 正则性成立）：

$$\nabla_x \mathcal{L}(x^*, \nu^*) = \nabla f(x^*) + J_h(x^*)^\top \nu^* = 0$$
$$h(x^*) = 0$$

### 不等式约束优化（KKT 条件）

**问题**：$\min f(x)$，s.t. $g_i(x) \leq 0$（$i=1,\ldots,m$），$h_j(x) = 0$（$j=1,\ldots,p$）。

**Lagrangian**：

$$\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x)$$

**KKT 必要条件**（约束规范性成立）：

$$\nabla_x \mathcal{L}(x^*, \lambda^*, \nu^*) = 0 \quad \text{（稳定性）}$$
$$g_i(x^*) \leq 0, \quad i = 1, \ldots, m \quad \text{（原始可行性）}$$
$$h_j(x^*) = 0, \quad j = 1, \ldots, p \quad \text{（原始可行性）}$$
$$\lambda_i^* \geq 0, \quad i = 1, \ldots, m \quad \text{（对偶可行性）}$$
$$\lambda_i^* g_i(x^*) = 0, \quad i = 1, \ldots, m \quad \text{（互补松弛性）}$$

**充分性**：若 $f, g_i$ 为凸函数，$h_j$ 为仿射函数，则 KKT 条件既必要也充分。

---

## A.5 梯度下降收敛性

### 基本梯度下降更新

$$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$$

### 一般凸函数（$f$ 是 $L$-光滑凸函数）

步长 $\alpha = 1/L$：

$$f(x_k) - f(x^*) \leq \frac{L\|x_0 - x^*\|^2}{2k}$$

**收敛速率**：$O(1/k)$（次线性）。

达到 $\epsilon$ 精度所需迭代次数：$k = O(L/\epsilon)$。

### 强凸函数（$f$ 是 $\mu$-强凸且 $L$-光滑）

步长 $\alpha = 1/L$，$\kappa = L/\mu$（条件数）：

$$\|x_k - x^*\|^2 \leq \left(1 - \frac{1}{\kappa}\right)^k \|x_0 - x^*\|^2$$

$$f(x_k) - f(x^*) \leq \frac{L}{2}\left(1 - \frac{\mu}{L}\right)^k \|x_0 - x^*\|^2$$

**收敛速率**：$O\!\left(\left(1 - 1/\kappa\right)^k\right)$（线性收敛）。

达到 $\epsilon$ 精度所需迭代次数：$k = O(\kappa \log(1/\epsilon))$。

### 非凸函数（$f$ 是 $L$-光滑，下有界）

步长 $\alpha = 1/L$：

$$\min_{0 \leq k \leq K-1} \|\nabla f(x_k)\|^2 \leq \frac{2L(f(x_0) - f^*)}{K}$$

**收敛速率**：$O(1/\sqrt{K})$（收敛到稳定点）。

### Polyak-Lojasiewicz（PL）条件

若存在 $\mu > 0$ 使得：

$$\|\nabla f(x)\|^2 \geq 2\mu(f(x) - f^*), \quad \forall x$$

则步长 $\alpha = 1/L$ 的梯度下降满足：

$$f(x_k) - f^* \leq \left(1 - \frac{\mu}{L}\right)^k (f(x_0) - f^*)$$

（线性收敛，无需强凸性。）

---

## A.6 牛顿法与拟牛顿法

### 牛顿法更新

$$x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$$

**牛顿减量**：

$$\lambda(x)^2 = \nabla f(x)^\top [\nabla^2 f(x)]^{-1} \nabla f(x)$$

**二次收敛**（在 $x^*$ 邻域内）：

$$\|x_{k+1} - x^*\| \leq C\|x_k - x^*\|^2$$

### BFGS 拟牛顿法

**符号**：$s_k = x_{k+1} - x_k$，$y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$，$\rho_k = 1/(y_k^\top s_k)$。

**Hessian 近似更新**（$B_k \approx \nabla^2 f$）：

$$B_{k+1} = B_k - \frac{B_k s_k s_k^\top B_k}{s_k^\top B_k s_k} + \frac{y_k y_k^\top}{y_k^\top s_k}$$

**逆 Hessian 近似更新**（$H_k = B_k^{-1} \approx [\nabla^2 f]^{-1}$）：

$$H_{k+1} = (I - \rho_k s_k y_k^\top) H_k (I - \rho_k y_k s_k^\top) + \rho_k s_k s_k^\top$$

**曲率条件**（$H_{k+1}$ 正定的必要条件）：

$$y_k^\top s_k > 0$$

（Wolfe 线搜索保证此条件成立。）

### L-BFGS 双循环递归算法

给定 $\nabla f(x_k)$，存储最近 $m$ 步的 $\{s_i, y_i\}_{i=k-m}^{k-1}$，计算方向 $d = -H_k \nabla f(x_k)$：

**前向循环**（$i = k-1, k-2, \ldots, k-m$）：

$$\alpha_i = \rho_i s_i^\top q, \quad q \leftarrow q - \alpha_i y_i$$

**中心缩放**：

$$r = H_k^0 q, \quad H_k^0 = \frac{s_{k-1}^\top y_{k-1}}{\|y_{k-1}\|^2} I$$

**反向循环**（$i = k-m, \ldots, k-1$）：

$$\beta_i = \rho_i y_i^\top r, \quad r \leftarrow r + s_i(\alpha_i - \beta_i)$$

输出方向 $d = -r$，内存复杂度 $O(mn)$，每步计算 $O(mn)$。

---

## A.7 对偶理论

### Lagrangian

$$\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x)$$

### 对偶函数

$$d(\lambda, \nu) = \inf_{x \in \mathcal{X}} \mathcal{L}(x, \lambda, \nu)$$

对偶函数 $d(\lambda, \nu)$ 是凹函数（无论原始问题是否为凸）。

### 弱对偶性

对所有 $\lambda \geq 0$：

$$d(\lambda, \nu) \leq p^*$$

其中 $p^* = \inf_x f(x)$（原始最优值）。**对偶间隙** $= p^* - d^* \geq 0$。

### 强对偶性（Slater 条件）

若原始问题为凸，且存在严格可行点（Slater 点）$\tilde{x}$：

$$g_i(\tilde{x}) < 0 \quad (i=1,\ldots,m), \quad h_j(\tilde{x}) = 0 \quad (j=1,\ldots,p)$$

则强对偶成立：$d^* = p^*$，且对偶最优解 $(\lambda^*, \nu^*)$ 存在。

### 对偶问题

$$\max_{\lambda \geq 0,\, \nu} \; d(\lambda, \nu) = \max_{\lambda \geq 0,\, \nu} \inf_x \mathcal{L}(x, \lambda, \nu)$$

### 共轭函数

函数 $f : \mathbb{R}^n \to \mathbb{R}$ 的共轭（Fenchel 共轭）：

$$f^*(y) = \sup_x \left( y^\top x - f(x) \right)$$

$f^*$ 总是凸函数（下半连续）。

**Fenchel-Young 不等式**：

$$f(x) + f^*(y) \geq x^\top y, \quad \forall x, y$$

等号成立当且仅当 $y \in \partial f(x)$（即 $x \in \partial f^*(y)$）。

---

## A.8 随机优化

### 随机梯度下降（SGD）

$$x_{k+1} = x_k - \alpha_k g_k, \quad \mathbb{E}[g_k \mid x_k] = \nabla f(x_k)$$

**方差假设**：$\mathbb{E}[\|g_k - \nabla f(x_k)\|^2] \leq \sigma^2$。

**凸函数收敛**（步长 $\alpha_k = O(1/\sqrt{k})$，$f$ 为 $L$-光滑凸函数）：

$$\mathbb{E}[f(\bar{x}_K)] - f(x^*) \leq \frac{\|x_0 - x^*\|^2}{2\alpha K} + \frac{\alpha \sigma^2}{2}$$

其中 $\bar{x}_K = \frac{1}{K}\sum_{k=0}^{K-1} x_k$（均值输出），最优步长 $\alpha = \|x_0-x^*\|/(\sigma\sqrt{K})$ 给出速率 $O(\sigma/\sqrt{K})$。

**强凸函数收敛**（步长 $\alpha_k = 2/(\mu(k+1))$）：

$$\mathbb{E}[f(\tilde{x}_K)] - f(x^*) \leq \frac{2L\sigma^2}{\mu^2 K}$$

其中 $\tilde{x}_K = \frac{2}{K(K+1)}\sum_{k=0}^{K-1}(k+1)x_k$（加权均值输出）。

### 动量更新（经典动量 / Heavy Ball）

$$m_{k+1} = \beta m_k + \nabla f(x_k)$$
$$x_{k+1} = x_k - \alpha m_{k+1}$$

### Nesterov 加速梯度（NAG）

$$y_{k+1} = x_k - \alpha \nabla f(x_k)$$
$$x_{k+1} = y_{k+1} + \frac{k-1}{k+2}(y_{k+1} - y_k)$$

**收敛速率**（$f$ 为 $L$-光滑凸函数）：

$$f(x_k) - f(x^*) \leq \frac{2L\|x_0 - x^*\|^2}{(k+1)^2} = O\!\left(\frac{1}{k^2}\right)$$

### 方差缩减——SVRG

$$g_k^{\text{SVRG}} = \nabla f_{i_k}(x_k) - \nabla f_{i_k}(\tilde{x}) + \nabla f(\tilde{x})$$

其中 $\tilde{x}$ 为快照点，$\nabla f(\tilde{x}) = \frac{1}{n}\sum_{i=1}^n \nabla f_i(\tilde{x})$（完整梯度）。

**收敛速率**（$\mu$-强凸，$L$-光滑）：线性收敛 $O\!\left(\left(n + L/\mu\right)\log(1/\epsilon)\right)$（总梯度计算次数）。

---

## A.9 自适应方法

设参数 $\theta$（向量），$g_t = \nabla_\theta f_t(\theta)$ 为第 $t$ 步随机梯度，所有运算均为逐元素操作，$\epsilon$ 为数值稳定项（通常 $10^{-8}$）。

### AdaGrad

$$G_t = G_{t-1} + g_t \odot g_t \quad (G_0 = 0)$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t} + \epsilon} \odot g_t$$

累积梯度平方，学习率单调下降；适合稀疏梯度，不适合非稳态问题。

### RMSprop

$$v_t = \beta v_{t-1} + (1-\beta) g_t \odot g_t \quad (\beta \approx 0.9)$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_t} + \epsilon} \odot g_t$$

指数移动平均缓解 AdaGrad 学习率消失问题。

### Adam

**一阶矩（动量）**：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

**二阶矩（自适应学习率）**：

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t \odot g_t$$

**偏差修正**：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**参数更新**：

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t$$

**默认超参数**：$\alpha = 0.001$，$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$。

**有效步长**：$\Delta_t = \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$，受 $\alpha$ 界定。

### AdamW（权重衰减解耦）

$$\theta_{t+1} = \theta_t - \alpha\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

其中 $\lambda$ 为权重衰减系数（与梯度计算解耦）。

### AMSGrad（单调二阶矩）

$$\hat{v}_t^{\max} = \max(\hat{v}_{t-1}^{\max},\, \hat{v}_t)$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t^{\max}} + \epsilon} \odot \hat{m}_t$$

保证有效学习率单调不增，修复 Adam 的收敛问题。

---

## A.10 常用不等式

### Jensen 不等式

$f$ 为凸函数，$X$ 为随机变量：

$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

有限形式（$\sum_i w_i = 1$，$w_i \geq 0$）：

$$f\!\left(\sum_{i=1}^n w_i x_i\right) \leq \sum_{i=1}^n w_i f(x_i)$$

### AM-GM 不等式

$$\frac{a + b}{2} \geq \sqrt{ab}, \quad a, b \geq 0$$

一般形式：

$$\frac{\sum_{i=1}^n x_i}{n} \geq \left(\prod_{i=1}^n x_i\right)^{1/n}, \quad x_i \geq 0$$

加权形式（$\sum_i w_i = 1$，$w_i > 0$）：

$$\sum_{i=1}^n w_i x_i \geq \prod_{i=1}^n x_i^{w_i}$$

### Young 不等式

$p > 1$，$q > 1$，$\frac{1}{p} + \frac{1}{q} = 1$，$a, b \geq 0$：

$$ab \leq \frac{a^p}{p} + \frac{b^q}{q}$$

带参数形式（$\epsilon > 0$）：

$$ab \leq \frac{\epsilon}{2} a^2 + \frac{1}{2\epsilon} b^2$$

（$p = q = 2$ 的带权版本，常用于误差分析。）

### Hölder 不等式

$\frac{1}{p} + \frac{1}{q} = 1$，$p, q \geq 1$：

$$\sum_{i=1}^n |x_i y_i| \leq \|x\|_p \|y\|_q$$

积分形式：

$$\int |f(x)g(x)|\,dx \leq \|f\|_p \|g\|_q$$

**Cauchy-Schwarz** 为 $p = q = 2$ 的特例。

### 三角不等式

$$\|x + y\| \leq \|x\| + \|y\|$$

$$\big| \|x\| - \|y\| \big| \leq \|x - y\|$$

### Peter-Paul 不等式（$\epsilon > 0$）

$$2ab \leq \epsilon a^2 + \frac{1}{\epsilon} b^2$$

### 矩阵迹不等式

**von Neumann 迹不等式**：

$$|\operatorname{tr}(A^\top B)| \leq \sum_i \sigma_i(A)\sigma_i(B)$$

**矩阵 AM-GM**（$A, B \succ 0$）：

$$\operatorname{tr}(AB) \leq \frac{1}{2}\operatorname{tr}(A^2 + B^2)$$

### 谱范数与 Frobenius 范数

$$\|AB\|_F \leq \|A\|_2 \|B\|_F, \quad \|AB\|_F \leq \|A\|_F \|B\|_2$$

$$\|AB\|_2 \leq \|A\|_2 \|B\|_2, \quad \|AB\|_F \leq \|A\|_F \|B\|_F$$

### 下降引理（Descent Lemma）

$f$ 是 $L$-光滑函数，步长 $\alpha = 1/L$：

$$f(x_{k+1}) \leq f(x_k) - \frac{1}{2L}\|\nabla f(x_k)\|^2$$

### 强凸二次下界

$f$ 是 $\mu$-强凸函数：

$$f(x^*) \geq f(x) + \nabla f(x)^\top(x^* - x) + \frac{\mu}{2}\|x^* - x\|^2$$

$$\implies f(x) - f(x^*) \leq \frac{1}{2\mu}\|\nabla f(x)\|^2$$

---

## 符号索引

| 符号 | 含义 |
|------|------|
| $\nabla f(x)$ | $f$ 在 $x$ 处的梯度，$\in \mathbb{R}^n$ |
| $\nabla^2 f(x)$ | $f$ 在 $x$ 处的 Hessian 矩阵，$\in \mathbb{R}^{n \times n}$ |
| $J_f(x)$ | $f : \mathbb{R}^n \to \mathbb{R}^m$ 的 Jacobian，$\in \mathbb{R}^{m \times n}$ |
| $\partial f(x)$ | $f$ 在 $x$ 处的次微分（次梯度集合） |
| $A \succeq 0$ | $A$ 为半正定矩阵 |
| $A \succ 0$ | $A$ 为正定矩阵 |
| $\sigma_i(A)$ | $A$ 的第 $i$ 大奇异值 |
| $\lambda_i(A)$ | $A$ 的第 $i$ 个特征值 |
| $\operatorname{tr}(A)$ | $A$ 的迹 |
| $\operatorname{dom}(f)$ | $f$ 的有效定义域 |
| $f^*$ | $f$ 的共轭函数（Fenchel 共轭） |
| $x^*$ | 最优解 |
| $p^*$ | 原始最优值 |
| $d^*$ | 对偶最优值 |
| $\kappa = L/\mu$ | 条件数 |
| $\odot$ | 逐元素乘积（Hadamard 积） |
| $\mathbb{E}[\cdot]$ | 期望 |

---

*本速查表配合各章节内容使用，公式编号对应正文中的相关引用。如需推导细节，请参阅对应章节。*
