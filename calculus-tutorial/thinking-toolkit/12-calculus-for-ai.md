# 微积分中的 AI 思维

> **一例速记**：梯度下降的每一步是什么？
> 参数更新 $\theta \leftarrow \theta - \eta \nabla L(\theta)$，背后是泰勒一阶近似：$L(\theta - \eta g) \approx L(\theta) - \eta \|g\|^2$（其中 $g = \nabla L$）。只要 $\eta$ 足够小，每步都能让 $L$ 减小。沿负梯度方向移动，loss 下降最快——这是微积分方向导数理论的直接推论。
> **梯度下降 = 沿"最陡下坡"方向走一小步，每步由一阶泰勒近似保证有效。**

---

## 一、为什么 AI 需要微积分

现代 AI（特别是深度学习）的核心操作——训练神经网络——在数学上就是**大规模非线性优化**：在参数空间中寻找使损失函数最小的参数值。

微积分在这个过程中扮演不可或缺的角色：

- **梯度**（多元微分）告诉我们每个参数应该朝哪个方向调整；
- **链式法则**（复合函数求导）让我们能高效计算深层网络中每一层的梯度；
- **Hessian**（二阶导数）告诉我们损失函数的曲率，影响学习率的选取和收敛速度；
- **凸性理论**（Toolkit 10）解释了为什么某些模型（逻辑回归、SVM）训练必然成功，为什么神经网络训练在理论上更困难；
- **信息论中的散度**（KL 散度）是量化概率分布差异的工具，其非负性由 Jensen 不等式保证。

本篇把这些微积分工具和 AI 的具体算法连接起来，让你在读论文或调参时能"看到背后的数学"。

---

## 二、梯度下降的微积分基础

### 2.1 多元泰勒展开与梯度方向

设损失函数 $L: \mathbb{R}^n \to \mathbb{R}$ 是光滑的（充分可微）。在参数 $\theta$ 处做**一阶泰勒展开**：

$$L(\theta + \Delta\theta) \approx L(\theta) + \nabla L(\theta)^T \Delta\theta$$

其中 $\nabla L(\theta) = \left(\dfrac{\partial L}{\partial \theta_1}, \ldots, \dfrac{\partial L}{\partial \theta_n}\right)^T$ 是梯度。

**哪个方向让 $L$ 下降最快？** 设步长固定为 $\|\Delta\theta\| = \eta$，问哪个方向 $\Delta\theta$ 使 $\nabla L^T \Delta\theta$ 最小（最负）？

由 Cauchy-Schwarz：$\nabla L^T \Delta\theta \geq -\|\nabla L\|\cdot\|\Delta\theta\| = -\eta\|\nabla L\|$

等号在 $\Delta\theta = -\eta \dfrac{\nabla L}{\|\nabla L\|}$（即**负梯度方向**）时成立。

因此，**梯度下降更新规则**：

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

这是"沿最速下降方向走 $\eta$ 步"的实现，其中 $\eta$ 是**学习率**。

### 2.2 学习率的选择与 Lipschitz 连续

如果梯度本身变化缓慢（即 $\nabla L$ 是 Lipschitz 连续的，Lipschitz 常数为 $\beta$），则当学习率满足 $\eta \leq 1/\beta$ 时，可以证明每步都能保证损失下降：

$$L(\theta_{t+1}) \leq L(\theta_t) - \frac{\eta}{2}\|\nabla L(\theta_t)\|^2$$

这是梯度下降**收敛保证**的标准形式。$\beta$ 越大（梯度变化越剧烈），需要的学习率越小。

实践中 $\beta$ 未知，常用自适应学习率（Adam、RMSProp）或学习率调度来规避。

---

## 三、反向传播 = 多元链式法则的递归

### 3.1 神经网络的计算图

一个 $L$ 层神经网络可以写成复合函数：

$$\hat{y} = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$$

其中每层 $f_\ell(\mathbf{a}) = \sigma(W_\ell \mathbf{a} + \mathbf{b}_\ell)$（$\sigma$ 是激活函数，$W_\ell, \mathbf{b}_\ell$ 是该层参数）。

损失 $L = \ell(\hat{y}, y)$（如均方误差或交叉熵）是这整个复合函数的输出关于真实标签的差距。

### 3.2 链式法则的矩阵形式

设中间变量 $\mathbf{z}_\ell = W_\ell \mathbf{a}_{\ell-1} + \mathbf{b}_\ell$，$\mathbf{a}_\ell = \sigma(\mathbf{z}_\ell)$。定义**误差信号**（error signal）：

$$\boldsymbol{\delta}_\ell = \frac{\partial L}{\partial \mathbf{z}_\ell}$$

则链式法则给出递推关系（**反向传播公式**）：

$$\boldsymbol{\delta}_\ell = \left(\sigma'(\mathbf{z}_\ell) \odot \boldsymbol{\delta}_{\ell+1}^{\text{下一层的贡献}}\right)$$

更精确地写（矩阵形式）：

$$\boldsymbol{\delta}_\ell = (W_{\ell+1}^T \boldsymbol{\delta}_{\ell+1}) \odot \sigma'(\mathbf{z}_\ell)$$

其中 $\odot$ 是逐元素乘法（Hadamard 乘积），$\sigma'(\mathbf{z}_\ell)$ 是激活函数导数（逐元素）。

**对参数的梯度**（用于更新权重）：

$$\frac{\partial L}{\partial W_\ell} = \boldsymbol{\delta}_\ell \mathbf{a}_{\ell-1}^T, \qquad \frac{\partial L}{\partial \mathbf{b}_\ell} = \boldsymbol{\delta}_\ell$$

### 3.3 为什么是"反向"传播？

从输出层（$\ell = L$）开始计算 $\boldsymbol{\delta}_L$（很简单，直接是 $\partial\ell/\partial\hat{y}$ 经激活函数导数调整），然后**逐层向前**（向输入方向）递推 $\boldsymbol{\delta}_{L-1}, \ldots, \boldsymbol{\delta}_1$。

每层的计算仅依赖下一层的误差信号和当前层的前向计算结果（已在前向传播中保存）。

**计算复杂度**：总体 $O(\text{参数数量})$，与前向传播同阶——这正是反向传播的革命性之处。朴素的有限差分法估计梯度需要 $O(\text{参数数量}^2)$ 次前向传播。

---

## 四、Hessian 矩阵与 Newton 法

### 4.1 二阶泰勒展开

在参数 $\theta$ 处做**二阶泰勒展开**：

$$L(\theta + \Delta\theta) \approx L(\theta) + \nabla L(\theta)^T \Delta\theta + \frac{1}{2}\Delta\theta^T H \Delta\theta$$

其中 $H = \nabla^2 L(\theta)$ 是 Hessian 矩阵（$n \times n$ 对称矩阵），$H_{ij} = \dfrac{\partial^2 L}{\partial \theta_i \partial \theta_j}$。

最小化右端关于 $\Delta\theta$：令 $\nabla_{\Delta\theta}$ 为零，得 $\nabla L + H\Delta\theta = 0$，即 $\Delta\theta = -H^{-1}\nabla L$。

### 4.2 Newton 法

**Newton 更新规则**：

$$\theta_{t+1} = \theta_t - H_t^{-1} \nabla L(\theta_t)$$

其中 $H_t = \nabla^2 L(\theta_t)$。

**优点**：
- 收敛速度快（二阶收敛，每步迭代使误差平方——而梯度下降是线性收敛）；
- 自适应步长：$H$ 自动缩放搜索方向，无需人工调学习率。

**缺点**：
- 计算和存储 $H$（$n \times n$ 矩阵）代价高：存储 $O(n^2)$，求逆 $O(n^3)$——对大模型（$n \sim 10^9$）完全不可行；
- Hessian 可能不正定（在非凸问题中），导致更新方向变差。

**实践中的替代**：
- **拟 Newton 法**（L-BFGS）：用低秩更新近似 $H^{-1}$；
- **自然梯度**：用 Fisher 信息矩阵代替 Hessian，在参数空间上的几何意义更好；
- **Adam**：用梯度平方的移动平均近似 Hessian 的对角元素，是最常用的近似二阶方法。

### 4.3 Hessian 的 AI 意义

**鞍点问题**：在高维非凸优化中，鞍点（Hessian 不定——有正有负特征值）比局部极小更常见。梯度下降在鞍点附近的梯度接近 $0$，收敛变慢；引入随机性（SGD、Adam 的噪声）有助于逃出鞍点。

**曲率与学习率**：Hessian 最大特征值 $\lambda_{\max}$ 决定了梯度下降的稳定步长上界 $\eta < 2/\lambda_{\max}$。如果不同参数方向的曲率差距很大（条件数 $\lambda_{\max}/\lambda_{\min}$ 很大），固定学习率梯度下降会震荡——这正是归一化（BatchNorm）等技术的动机之一。

---

## 五、KL 散度与凸性

### 5.1 KL 散度的定义

**Kullback-Leibler 散度**（KL divergence）度量两个概率分布 $p$ 和 $q$ 之间的"差异"（严格来说不是距离，因为不对称）：

$$\text{KL}(p \| q) = \int p(x) \ln \frac{p(x)}{q(x)}\,dx$$

（离散版：$\text{KL}(p\|q) = \sum_x p(x) \ln \dfrac{p(x)}{q(x)}$）

**KL 散度的非负性**：$\text{KL}(p\|q) \geq 0$，等号成立当且仅当 $p = q$（几乎处处）。

### 5.2 用 Jensen 不等式证明 KL 非负

$-\ln$ 是凸函数（$(-\ln)'' = 1/x^2 > 0$），对任意概率分布 $p$，以 $q/p$ 为随机变量，$p$ 为权重分布，用 Jensen：

$$-\ln\!\left(\int p(x)\cdot\frac{q(x)}{p(x)}\,dx\right) \leq \int p(x)\cdot\left(-\ln\frac{q(x)}{p(x)}\right)\,dx$$

左端 $= -\ln\!\left(\int q(x)\,dx\right) = -\ln 1 = 0$（因为 $q$ 是概率分布，积分为 1）。

右端 $= \int p(x)\ln\dfrac{p(x)}{q(x)}\,dx = \text{KL}(p\|q)$。

故 $0 \leq \text{KL}(p\|q)$。$\blacksquare$

### 5.3 KL 散度在 AI 中的应用

| 应用场景 | KL 散度的角色 |
|---|---|
| **变分自编码器（VAE）** | 正则化项 $\text{KL}(q_\phi(z\|x) \| p(z))$ 使隐变量分布贴近先验 |
| **语言模型训练（PPO/RLHF）** | 约束策略更新 $\text{KL}(\pi_\theta \| \pi_{\text{ref}}) \leq \epsilon$，防止过大更新 |
| **知识蒸馏** | 最小化学生分布与教师分布的 KL：$\text{KL}(p_{\text{teacher}} \| p_{\text{student}})$ |
| **最大熵原理** | 在满足约束的分布中，最小化 $\text{KL}(p \| \text{uniform})$ 等价于最大化熵 |

---

## 六、自动微分（Automatic Differentiation）

### 6.1 两种模式

自动微分（AD）不是符号微分（手动展开公式），也不是数值微分（有限差分）——而是精确利用链式法则，**在程序执行过程中同时积累梯度**。

| 模式 | 方向 | 每次计算 | 适用场景 |
|---|---|---|---|
| **前向模式**（Forward mode） | 从输入到输出（顺着计算图） | 一个输入对所有输出的导数（Jacobian 的一列） | 少输入多输出（$n_{\text{in}} \ll n_{\text{out}}$） |
| **反向模式**（Backward mode） | 从输出到输入（逆着计算图） | 一个输出对所有输入的导数（Jacobian 的一行） | 多输入少输出（$n_{\text{in}} \gg n_{\text{out}}$，如 ML） |

深度学习中，参数数量 $n_{\text{in}} \sim 10^7$—$10^{11}$，而损失是标量（$n_{\text{out}} = 1$），因此**反向模式 AD 是标准选择**——这正是 PyTorch、TensorFlow 中"反向传播"的实现机制。

### 6.2 前向模式的对偶数实现

前向模式可以用**对偶数**（dual numbers）实现：设 $\epsilon$ 满足 $\epsilon^2 = 0$，对偶数 $a + b\epsilon$ 携带"实部"（函数值）和"对偶部"（导数值）。复合运算自动传播导数：

$$(a + b\epsilon) + (c + d\epsilon) = (a+c) + (b+d)\epsilon$$
$$(a + b\epsilon)(c + d\epsilon) = ac + (bc + ad)\epsilon \quad (\epsilon^2 = 0 \text{ 项丢弃})$$

若输入 $x = x_0 + 1 \cdot \epsilon$（$\dot{x} = 1$），通过复合运算自动得到 $f(x_0) + f'(x_0)\epsilon$。

### 6.3 计算图与梯度检查

实践技巧：实现新的神经网络结构时，用**数值梯度检验**（gradient check）验证反向传播实现是否正确：

$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + \epsilon \mathbf{e}_i) - L(\theta - \epsilon \mathbf{e}_i)}{2\epsilon}$$

计算所有参数的相对误差，若 $< 10^{-4}$ 则认为实现正确。（注意：使用 $\epsilon \sim 10^{-5}$ 的对称差分，精度 $O(\epsilon^2)$，优于单侧差分的 $O(\epsilon)$。）

---

## 七、Lagrangian 与对偶（SVM 核心推导）

### 7.1 SVM 的原始问题

**硬间隔 SVM** 的原始优化问题：在线性可分数据中找最大间隔超平面：

$$\min_{w, b} \frac{1}{2}\|w\|^2, \quad \text{s.t. } y_i(w^T x_i + b) \geq 1, \quad i = 1, \ldots, n$$

这是带不等式约束的二次规划（凸问题）。

### 7.2 Lagrangian 函数

引入 Lagrange 乘子 $\alpha_i \geq 0$（对每个约束），构造 Lagrangian：

$$\mathcal{L}(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^n \alpha_i[y_i(w^T x_i + b) - 1]$$

**对偶问题**：对 $w, b$ 最小化 Lagrangian，得到**对偶目标函数**（只含 $\alpha$）：

对 $w$：$\partial \mathcal{L}/\partial w = w - \sum \alpha_i y_i x_i = 0$，得 $w^* = \sum \alpha_i y_i x_i$；

对 $b$：$\partial \mathcal{L}/\partial b = -\sum \alpha_i y_i = 0$。

代入 Lagrangian，对偶问题变为：

$$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j, \quad \text{s.t. } \alpha_i \geq 0, \sum \alpha_i y_i = 0$$

### 7.3 为什么对偶有用？

- **核技巧**：对偶目标只含内积 $x_i^T x_j$，可以用**核函数** $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$ 替换，实现隐式非线性映射而无需显式计算高维特征；
- **稀疏性**：KKT 互补松弛 $\alpha_i[y_i(w^T x_i + b) - 1] = 0$ 意味着只有"支持向量"（约束活跃的点）对应 $\alpha_i > 0$，大多数样本 $\alpha_i = 0$——解是稀疏的；
- **强对偶**：由于原问题是凸的（Slater 条件满足），强对偶成立，对偶问题与原问题等价。

---

## 八、演示题：单个神经元的链式法则

**题目**：设单个 sigmoid 神经元 $y = \sigma(w_1 x_1 + w_2 x_2 + b)$，损失 $L = (y - t)^2$，用链式法则手算 $\dfrac{\partial L}{\partial w_1}$、$\dfrac{\partial L}{\partial w_2}$、$\dfrac{\partial L}{\partial b}$。

> **第一步：画出计算图，标清中间变量。**
>
> 设 $z = w_1 x_1 + w_2 x_2 + b$（线性部分），$y = \sigma(z)$（激活），$L = (y - t)^2$（损失）。
>
> 计算图：$(w_1, x_1, w_2, x_2, b) \to z \to y \to L$。

> **第二步：从输出到输入逐层计算偏导（反向传播）。**
>
> **$\partial L / \partial y$**：
> $$\frac{\partial L}{\partial y} = 2(y - t)$$
>
> **$\partial L / \partial z$**（链式法则，$y$ 是 $z$ 的函数）：
> $$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} = 2(y-t) \cdot \sigma'(z)$$
>
> 其中 $\sigma'(z) = \sigma(z)(1-\sigma(z)) = y(1-y)$（sigmoid 导数的精妙形式）。
>
> 故：
> $$\frac{\partial L}{\partial z} = 2(y - t) \cdot y(1 - y)$$

> **第三步：对各参数求导（$z$ 对参数是线性的）。**
>
> $$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w_1} = 2(y-t)y(1-y) \cdot x_1$$
>
> $$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w_2} = 2(y-t)y(1-y) \cdot x_2$$
>
> $$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial b} = 2(y-t)y(1-y) \cdot 1 = 2(y-t)y(1-y)$$

> **第四步：理解结构。**
>
> 梯度 $= $ 损失信号 $\times$ 激活函数导数 $\times$ 输入值。
>
> - **损失信号** $2(y-t)$：预测值与真实值的差，差越大梯度越大；
> - **激活函数导数** $y(1-y)$：sigmoid 在 $y \approx 0$ 或 $y \approx 1$ 时导数接近 $0$（**梯度消失**的来源），在 $y = 0.5$ 时最大（$= 0.25$）；
> - **输入值** $x_1, x_2$：哪个输入更大，对应权重的梯度也更大，更新更快。
>
> 这正是权重 $w_i$ 和输入 $x_i$ 对梯度贡献的**对称性**——也是 $L_2$ 正则化（让权重小）和输入标准化（让 $x_i$ 量级统一）的动机。

---

## 九、思考路标

**路标 1**：梯度下降的每一步背后是**一阶泰勒近似**。学习率 $\eta$ 控制步长，太大则近似失效（甚至导致 loss 上升），太小则收敛太慢。**Lipschitz 常数 $\beta$ 是理论上的最大学习率（$\eta \leq 1/\beta$）**；实践中用线搜索或自适应方法替代精确计算。

**路标 2**：反向传播的**本质是动态规划**——通过从输出到输入的递推，避免重复计算子问题（中间层的误差信号 $\boldsymbol{\delta}_\ell$ 只计算一次，被所有依赖它的上层参数共用）。理解这一点，就理解了为什么计算复杂度是 $O(\text{参数数量})$ 而非 $O(\text{参数数量}^2)$。

**路标 3**：**sigmoid 函数的梯度消失问题**来自 $\sigma'(z) = \sigma(z)(1-\sigma(z))$：当 $|z|$ 很大时，$\sigma(z) \approx 0$ 或 $\approx 1$，导数趋向 $0$。多层网络中这个导数连乘，指数速度衰减到接近 $0$——这是深层网络用 ReLU（$\sigma'(z) = 1$，无饱和区）代替 sigmoid 的根本原因。

**路标 4**：Newton 法 $\theta \leftarrow \theta - H^{-1}\nabla L$ 在理论上远比梯度下降快，但在 $n > 10^4$ 的规模下计算 $H^{-1}$ 不现实。**Adam 可以理解为对角 Hessian 的一阶近似**：用 $v_t \approx \text{diag}(H)$（梯度平方的指数移动平均近似各参数方向的曲率），从而实现自适应学习率而无需显式 Hessian。

**路标 5**：KL 散度 $\text{KL}(p\|q) \geq 0$ 的证明通过 **Jensen 不等式作用于凸函数 $-\ln$**——这条证明路线连接了信息论（KL 散度）、概率论（期望、分布）和微积分（凸性、Jensen）三个领域。深刻理解这条证明，就理解了三者的内在统一。

**路标 6**：SVM 的对偶问题引入了**核技巧的可能性**：原问题在参数 $w$ 空间求解，复杂度与特征维度相关；对偶问题在数据点 $\alpha$ 空间求解，通过内积 $x_i^T x_j$ 与特征空间解耦，使高维（甚至无穷维）特征映射在计算上可行。这是 Lagrangian 对偶在 ML 中最深刻的应用之一。

**路标 7**：前向模式 AD 和反向模式 AD 的复杂度比较：前向模式每次扫描计算一个输入的 Jacobian 列，共需 $n_{\text{in}}$ 次扫描；反向模式每次扫描计算一个输出的 Jacobian 行，共需 $n_{\text{out}}$ 次扫描。**深度学习：$n_{\text{in}} \gg 1$，$n_{\text{out}} = 1$，故反向模式仅需 1 次扫描**——这是 PyTorch `.backward()` 一次调用就能计算所有参数梯度的数学原因。

**路标 8**：理解"局部极小 vs 全局极小"在深度学习中的实际含义：理论上存在局部极小，但实验发现对大型网络，几乎所有局部极小的 loss 值都接近全局最小（高维空间的大多数临界点是鞍点而非局部极小）。**这不是理论保证，而是经验规律**；当前研究正在努力理解为什么深度学习在非凸优化上能成功。

---

## 十、典型应用例题

### 例 1：梯度下降收敛分析

**场景**：线性回归损失 $L(w) = \dfrac{1}{2n}\|Xw - y\|^2$（$X \in \mathbb{R}^{n \times d}$，$y \in \mathbb{R}^n$）。

**分析**：$\nabla L(w) = \dfrac{1}{n}X^T(Xw - y)$，Hessian $H = \dfrac{1}{n}X^TX$（半正定，$L$ 是凸函数）。

Hessian 的最大特征值 $\lambda_{\max} = \dfrac{1}{n}\sigma_{\max}^2(X)$（$X$ 最大奇异值的平方除以 $n$）。

**结论**：学习率 $\eta < \dfrac{2n}{\sigma_{\max}^2(X)}$ 时，梯度下降线性收敛：

$$L(w_t) - L(w^*) \leq \left(1 - \frac{\eta \lambda_{\min}(H)}{\lambda_{\max}(H)}\right)^t [L(w_0) - L(w^*)]$$

条件数 $\kappa = \lambda_{\max}/\lambda_{\min}$ 越大（特征高度共线时），收敛越慢——这是为什么特征标准化（让 $X$ 的各列量级相近，降低 $\kappa$）能大幅加速训练。

---

### 例 2：KL 散度与 VAE

**场景**：变分自编码器（VAE）将图像 $x$ 编码为隐变量 $z$ 的分布 $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x)I)$，先验 $p(z) = \mathcal{N}(0, I)$。

**KL 散度计算**（两个高斯之间的 KL 有闭合表达式）：

$$\text{KL}(q_\phi(z|x) \| p(z)) = \frac{1}{2}\sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - \ln\sigma_j^2 - 1\right)$$

**微积分推导**（单维，$q = \mathcal{N}(\mu, \sigma^2)$，$p = \mathcal{N}(0, 1)$，多维各维独立求和）：代入对数差
$$\ln\frac{q(z)}{p(z)} = -\tfrac{1}{2}\ln\sigma^2 - \frac{(z-\mu)^2}{2\sigma^2} + \frac{z^2}{2},$$
对 $z \sim q$ 取期望，用 $\mathbb{E}[(z-\mu)^2] = \sigma^2$、$\mathbb{E}[z^2] = \sigma^2 + \mu^2$：
$$\text{KL} = \mathbb{E}_q\!\left[\ln\frac{q}{p}\right] = -\tfrac{1}{2}\ln\sigma^2 - \tfrac{1}{2} + \tfrac{1}{2}(\sigma^2 + \mu^2) = \tfrac{1}{2}\big(\sigma^2 + \mu^2 - 1 - \ln\sigma^2\big).$$

**意义**：KL 项作为正则化，使编码器的输出分布贴近先验 $\mathcal{N}(0, I)$，确保隐空间的连续性（可以从先验采样然后解码生成新样本）。

---

### 例 3：自动微分——从 Python 到梯度

**场景**：实现简单的前向模式 AD，计算 $f(x) = x^2 \sin(x)$ 在 $x = \pi/4$ 处的导数。

**对偶数方法**：设 $x = (\pi/4) + 1\cdot\epsilon$，逐步传播：

- $x^2 = \left(\frac{\pi}{4}\right)^2 + 2\cdot\frac{\pi}{4}\cdot 1 \cdot \epsilon = \frac{\pi^2}{16} + \frac{\pi}{2}\epsilon$
- $\sin(x)$：$\sin\!\left(\frac{\pi}{4} + \epsilon\right) \approx \sin\frac{\pi}{4} + \cos\frac{\pi}{4}\cdot\epsilon = \frac{\sqrt{2}}{2} + \frac{\sqrt{2}}{2}\epsilon$
- $f = x^2\sin(x) = \left(\frac{\pi^2}{16} + \frac{\pi}{2}\epsilon\right)\!\left(\frac{\sqrt{2}}{2} + \frac{\sqrt{2}}{2}\epsilon\right)$
  $= \frac{\pi^2\sqrt{2}}{32} + \left(\frac{\pi\sqrt{2}}{4} + \frac{\pi^2\sqrt{2}}{32}\right)\epsilon$（丢弃 $\epsilon^2$ 项）

对偶部提取：$f'(\pi/4) = \dfrac{\pi\sqrt{2}}{4} + \dfrac{\pi^2\sqrt{2}}{32}$（= $2x\sin(x) + x^2\cos(x)$ 在 $x=\pi/4$ 的值，可验证）。

---

## 十一、自测题

**第 1 题**：设 $L(\theta) = \theta^4 - 4\theta^3 + 5\theta^2$（玩具损失函数），(1) 找所有驻点；(2) 判断哪些是局部极小/极大/鞍点；(3) 若用梯度下降（$\eta = 0.05$）从 $\theta_0 = 3$ 出发，第一步更新后 $\theta_1$ 是多少？

> 提示：$L' = 4\theta^3 - 12\theta^2 + 10\theta = 2\theta(2\theta^2 - 6\theta + 5)$，判别式 $36 - 40 < 0$，故 $2\theta^2 - 6\theta + 5 > 0$ 恒成立，唯一驻点 $\theta = 0$。$L'' = 12\theta^2 - 24\theta + 10$，$L''(0) = 10 > 0$，极小值。$\theta_1 = 3 - 0.05 \cdot L'(3) = 3 - 0.05(108 - 108 + 30) = 3 - 0.05\cdot30 = 1.5$。

**第 2 题**：手写 $y = \text{ReLU}(w_1x + b_1)$（第一层），$L = (y - t)^2$（损失），计算 $\partial L/\partial w_1$ 和 $\partial L/\partial b_1$（其中 $\text{ReLU}(z) = \max(0, z)$，当 $z > 0$ 时 $\text{ReLU}'(z) = 1$，否则 $= 0$）。

> 提示：设 $z = w_1 x + b_1$。$\partial L/\partial y = 2(y-t)$，$\partial y/\partial z = \mathbf{1}[z > 0]$，$\partial z/\partial w_1 = x$，$\partial z/\partial b_1 = 1$。链式法则：$\partial L/\partial w_1 = 2(y-t)\cdot\mathbf{1}[z>0]\cdot x$，$\partial L/\partial b_1 = 2(y-t)\cdot\mathbf{1}[z>0]$。注意：当 $z \leq 0$ 时梯度为 $0$（ReLU 的"死神经元"现象）。

**第 3 题**：解释为什么 $\text{KL}(p\|q) \neq \text{KL}(q\|p)$（举一个具体的概率分布对），并说明在 VAE 中为什么选用 $\text{KL}(q\|p)$ 而不是 $\text{KL}(p\|q)$。

> 提示：取 $p = \mathcal{N}(0, 1)$，$q = \mathcal{N}(0, 4)$（方差不同），由单维高斯 KL 闭式 $\text{KL}(\mathcal{N}(m_1,s_1^2)\|\mathcal{N}(m_2,s_2^2)) = \ln\dfrac{s_2}{s_1} + \dfrac{s_1^2 + (m_1-m_2)^2}{2s_2^2} - \dfrac{1}{2}$，得 $\text{KL}(p\|q) = \ln 2 - \dfrac{3}{8} \approx 0.318$，而 $\text{KL}(q\|p) = \dfrac{3}{2} - \ln 2 \approx 0.807$，两者不等，故 KL 不对称。VAE 中最大化 ELBO 自然导出 $\text{KL}(q_\phi(z|x)\|p(z))$（KL 是对 $q$ 的期望，而 $q$ 是编码器输出的可采样分布）；$\text{KL}(p(z)\|q_\phi)$ 要求对 $p$ 积分，而 $p(z|x)$（真后验）是无法直接计算的。

**第 4 题**：Newton 法在 $f(x) = x^2$ 上从 $x_0 = 3$ 出发，一步到达最优解 $x^* = 0$。验证这一点，并解释为什么 Newton 法对二次函数恰好一步收敛。

> 提示：$f'(x) = 2x$，$f''(x) = 2$，Newton 步：$x_1 = x_0 - f'(x_0)/f''(x_0) = 3 - 6/2 = 0 = x^*$ ✓。原因：Newton 法在每步用二阶泰勒展开近似 $f$，而对二次函数，二阶展开是精确的（无余项），因此一步就能跳到精确最优解。

**第 5 题**：设输入 $x = [x_1, x_2]^T$，网络 $y = \sigma(W x + b)$（$W \in \mathbb{R}^{1\times2}$，$b \in \mathbb{R}$，$\sigma$ 是 sigmoid），损失 $L = -[t\ln y + (1-t)\ln(1-y)]$（交叉熵）。推导 $\partial L/\partial W$ 和 $\partial L/\partial b$。

> 提示：$\partial L/\partial y = -t/y + (1-t)/(1-y) = (y-t)/[y(1-y)]$。$\partial y/\partial z = y(1-y)$（$z = Wx+b$）。$\partial L/\partial z = \partial L/\partial y \cdot \partial y/\partial z = (y-t)/[y(1-y)] \cdot y(1-y) = y - t$。故 $\partial L/\partial W = (y-t)x^T$（行向量），$\partial L/\partial b = y - t$。这正是 logistic 回归梯度的标准形式——交叉熵损失与 sigmoid 激活组合，使梯度表达式极其简洁。
