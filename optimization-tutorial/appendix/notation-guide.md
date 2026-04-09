# 附录 B：符号说明 (Notation Guide)

本附录汇总了本教程中使用的所有数学符号，供查阅参考。

---

## B.1 集合与空间

| 符号 | 含义 | 说明 |
|------|------|------|
| $\mathbb{R}$ | 实数集 | 所有实数的集合 |
| $\mathbb{R}^n$ | $n$ 维实向量空间 | $n$ 个实数组成的有序元组的集合 |
| $\mathbb{R}^{m \times n}$ | $m \times n$ 实矩阵空间 | 所有 $m$ 行 $n$ 列实矩阵的集合 |
| $\mathbb{R}_+$ | 非负实数集 | $\{x \in \mathbb{R} \mid x \geq 0\}$ |
| $\mathbb{R}_{++}$ | 正实数集 | $\{x \in \mathbb{R} \mid x > 0\}$ |
| $\mathbb{R}_+^n$ | $n$ 维非负正交卦限 | 各分量均非负的向量集合 |
| $\mathbb{S}^n$ | $n \times n$ 实对称矩阵空间 | $\{A \in \mathbb{R}^{n \times n} \mid A = A^\top\}$ |
| $\mathbb{S}^n_+$ | $n \times n$ 半正定矩阵锥 | $\{A \in \mathbb{S}^n \mid A \succeq 0\}$ |
| $\mathbb{S}^n_{++}$ | $n \times n$ 正定矩阵锥 | $\{A \in \mathbb{S}^n \mid A \succ 0\}$ |
| $\mathbb{Z}$ | 整数集 | 所有整数的集合 |
| $\mathbb{Z}_+$ | 非负整数集 | $\{0, 1, 2, \ldots\}$ |
| $\mathbb{N}$ | 自然数集 | $\{1, 2, 3, \ldots\}$（本教程约定不含 $0$）|
| $\emptyset$ | 空集 | 不含任何元素的集合 |
| $\mathcal{X}$ | 可行域 / 约束集 | 优化变量的允许取值范围 |
| $\mathcal{C}$ | 凸集 | 一般凸集，具体含义由上下文确定 |
| $\text{dom}(f)$ | 函数 $f$ 的定义域 | $\{x \mid f(x) < +\infty\}$（扩展值意义下）|
| $\text{int}(\mathcal{C})$ | 集合 $\mathcal{C}$ 的内部 | 所有内点构成的集合 |
| $\text{cl}(\mathcal{C})$ | 集合 $\mathcal{C}$ 的闭包 | 包含所有极限点的最小闭集 |
| $\text{bd}(\mathcal{C})$ | 集合 $\mathcal{C}$ 的边界 | $\text{cl}(\mathcal{C}) \setminus \text{int}(\mathcal{C})$ |
| $\text{conv}(\mathcal{C})$ | 集合 $\mathcal{C}$ 的凸包 | 包含 $\mathcal{C}$ 的最小凸集 |
| $[a, b]$ | 闭区间 | $\{x \in \mathbb{R} \mid a \leq x \leq b\}$ |
| $(a, b)$ | 开区间 | $\{x \in \mathbb{R} \mid a < x < b\}$ |
| $\{1:n\}$ | 指标集 | $\{1, 2, \ldots, n\}$ 的简写 |

---

## B.2 向量与矩阵

### B.2.1 基本约定

| 符号 | 含义 | 说明 |
|------|------|------|
| $x, y, z$ | 标量 | 斜体小写字母 |
| $\mathbf{x}, \mathbf{y}, \mathbf{z}$ | 列向量 | 粗体小写字母，默认为列向量 |
| $\mathbf{A}, \mathbf{B}, \mathbf{C}$ | 矩阵 | 粗体大写字母 |
| $x_i$ | 向量 $\mathbf{x}$ 的第 $i$ 个分量 | 下标为分量索引 |
| $A_{ij}$ 或 $[\mathbf{A}]_{ij}$ | 矩阵 $\mathbf{A}$ 的第 $(i,j)$ 元素 | 第 $i$ 行第 $j$ 列 |
| $\mathbf{a}_i$ | 矩阵 $\mathbf{A}$ 的第 $i$ 列 | 粗体小写，下标为列索引 |
| $\mathbf{a}_i^\top$ | 矩阵 $\mathbf{A}$ 的第 $i$ 行 | 以行向量形式表示 |

### B.2.2 特殊向量与矩阵

| 符号 | 含义 | 维度 |
|------|------|------|
| $\mathbf{0}$ | 零向量 / 零矩阵 | 由上下文确定 |
| $\mathbf{1}$ | 全一向量 | $(1, 1, \ldots, 1)^\top$ |
| $\mathbf{e}_i$ | 第 $i$ 个标准基向量 | 第 $i$ 位为 $1$，其余为 $0$ |
| $\mathbf{I}$ 或 $\mathbf{I}_n$ | 单位矩阵 | $n \times n$ |
| $\text{diag}(\mathbf{x})$ | 以 $\mathbf{x}$ 为对角元的对角矩阵 | $n \times n$ |
| $\text{diag}(\mathbf{A})$ | 矩阵 $\mathbf{A}$ 的对角元组成的向量 | $n \times 1$ |

### B.2.3 矩阵运算

| 符号 | 含义 | 说明 |
|------|------|------|
| $\mathbf{A}^\top$ | 矩阵转置 | 行列互换 |
| $\mathbf{A}^{-1}$ | 矩阵逆 | 要求 $\mathbf{A}$ 可逆 |
| $\mathbf{A}^{-\top}$ | 转置的逆（即逆的转置）| $(\mathbf{A}^{-1})^\top = (\mathbf{A}^\top)^{-1}$ |
| $\mathbf{A}^\dagger$ | Moore-Penrose 伪逆 | 适用于非方阵或奇异矩阵 |
| $\mathbf{A} \odot \mathbf{B}$ | Hadamard 积（逐元素乘积）| 要求 $\mathbf{A}$、$\mathbf{B}$ 同型 |
| $\mathbf{A} \otimes \mathbf{B}$ | Kronecker 积 | $(m \times n) \otimes (p \times q) \to (mp \times nq)$ |
| $\text{tr}(\mathbf{A})$ | 矩阵的迹 | 对角元之和 |
| $\det(\mathbf{A})$ 或 $|\mathbf{A}|$ | 矩阵的行列式 | — |
| $\text{rank}(\mathbf{A})$ | 矩阵的秩 | — |
| $\lambda_i(\mathbf{A})$ | 矩阵 $\mathbf{A}$ 的第 $i$ 个特征值 | 通常按从小到大排列 |
| $\lambda_{\min}(\mathbf{A})$ | 最小特征值 | — |
| $\lambda_{\max}(\mathbf{A})$ | 最大特征值 | — |
| $\sigma_i(\mathbf{A})$ | 矩阵 $\mathbf{A}$ 的第 $i$ 个奇异值 | 通常按从大到小排列 |
| $\mathbf{A} \succeq \mathbf{B}$ | 半正定序 | $\mathbf{A} - \mathbf{B} \in \mathbb{S}^n_+$ |
| $\mathbf{A} \succ \mathbf{B}$ | 正定序 | $\mathbf{A} - \mathbf{B} \in \mathbb{S}^n_{++}$ |
| $\text{vec}(\mathbf{A})$ | 矩阵向量化 | 将各列依次堆叠为长向量 |

---

## B.3 范数与内积

| 符号 | 名称 | 定义 |
|------|------|------|
| $\|\mathbf{x}\|$ 或 $\|\mathbf{x}\|_2$ | 欧几里得范数（$\ell_2$ 范数）| $\sqrt{\sum_i x_i^2}$ |
| $\|\mathbf{x}\|_1$ | $\ell_1$ 范数 | $\sum_i |x_i|$ |
| $\|\mathbf{x}\|_\infty$ | $\ell_\infty$ 范数（切比雪夫范数）| $\max_i |x_i|$ |
| $\|\mathbf{x}\|_0$ | $\ell_0$ "范数" | $\|\mathbf{x}\|_0 = |\{i \mid x_i \neq 0\}|$（非凸）|
| $\|\mathbf{x}\|_p$ | $\ell_p$ 范数 | $\left(\sum_i |x_i|^p\right)^{1/p}$，$p \geq 1$ |
| $\|\mathbf{A}\|_F$ | Frobenius 范数 | $\sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\text{tr}(\mathbf{A}^\top \mathbf{A})}$ |
| $\|\mathbf{A}\|_2$ | 矩阵谱范数（算子范数）| $\sigma_{\max}(\mathbf{A})$，即最大奇异值 |
| $\|\mathbf{A}\|_*$ | 核范数（迹范数）| $\sum_i \sigma_i(\mathbf{A})$，所有奇异值之和 |
| $\|\mathbf{A}\|_1$ | 矩阵 $\ell_1$ 范数 | $\max_j \sum_i |A_{ij}|$（最大列绝对值和）|
| $\langle \mathbf{x}, \mathbf{y} \rangle$ | 标准内积 | $\mathbf{x}^\top \mathbf{y} = \sum_i x_i y_i$ |
| $\langle \mathbf{A}, \mathbf{B} \rangle$ | 矩阵内积 | $\text{tr}(\mathbf{A}^\top \mathbf{B})$ |
| $d(\mathbf{x}, \mathbf{y})$ | 距离 | 欧几里得距离 $\|\mathbf{x} - \mathbf{y}\|_2$（除非另有说明）|

---

## B.4 微分算子

| 符号 | 名称 | 说明 |
|------|------|------|
| $\dfrac{\partial f}{\partial x_i}$ | 偏导数 | $f$ 对第 $i$ 个变量的偏导 |
| $\nabla f(\mathbf{x})$ | 梯度 | $\left[\dfrac{\partial f}{\partial x_1}, \ldots, \dfrac{\partial f}{\partial x_n}\right]^\top \in \mathbb{R}^n$，与 $\mathbf{x}$ 同维的列向量 |
| $\nabla^2 f(\mathbf{x})$ | Hessian 矩阵 | $\left[\dfrac{\partial^2 f}{\partial x_i \partial x_j}\right]_{n \times n} \in \mathbb{S}^n$ |
| $\mathbf{J}_f(\mathbf{x})$ | Jacobian 矩阵 | $f: \mathbb{R}^n \to \mathbb{R}^m$ 时，$[\mathbf{J}_f]_{ij} = \dfrac{\partial f_i}{\partial x_j}$，维度 $m \times n$ |
| $\partial f(\mathbf{x})$ | 次微分 | 凸函数 $f$ 在 $\mathbf{x}$ 处的次梯度集合 |
| $g \in \partial f(\mathbf{x})$ | 次梯度 | 满足 $f(\mathbf{y}) \geq f(\mathbf{x}) + \langle g, \mathbf{y}-\mathbf{x}\rangle$ 的向量 $g$ |
| $D_f(\mathbf{x}, \mathbf{y})$ | Bregman 散度 | $f(\mathbf{y}) - f(\mathbf{x}) - \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x}\rangle$ |
| $\nabla_{\mathbf{x}} \mathcal{L}$ | 对 $\mathbf{x}$ 求梯度 | Lagrange 函数（或其他多变量函数）对 $\mathbf{x}$ 的梯度 |
| $\dfrac{d}{dt}$ | 全导数 | 对标量变量 $t$ 求导 |
| $\ddot{f}$ | 二阶导数 | 标量函数的二阶导（有时用 $f''$）|

---

## B.5 优化符号

### B.5.1 问题描述

| 符号 | 含义 | 说明 |
|------|------|------|
| $f: \mathbb{R}^n \to \mathbb{R}$ | 目标函数 | 待最小化（或最大化）的函数 |
| $\min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$ | 最小化问题 | 在可行域 $\mathcal{X}$ 上求 $f$ 的最小值 |
| $\max_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$ | 最大化问题 | 在可行域 $\mathcal{X}$ 上求 $f$ 的最大值 |
| $\text{s.t.}$ | 约束条件（subject to）| 引出不等式或等式约束 |
| $g_i(\mathbf{x}) \leq 0$ | 不等式约束 | $i = 1, \ldots, m$ |
| $h_j(\mathbf{x}) = 0$ | 等式约束 | $j = 1, \ldots, p$ |
| $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ | Lagrange 函数 | $f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \nu_j h_j(\mathbf{x})$ |
| $d(\boldsymbol{\lambda}, \boldsymbol{\nu})$ | 对偶函数 | $\inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ |
| $p^*$ | 原问题最优值 | $\inf_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$ |
| $d^*$ | 对偶问题最优值 | $\sup_{\boldsymbol{\lambda} \geq 0, \boldsymbol{\nu}} d(\boldsymbol{\lambda}, \boldsymbol{\nu})$ |

### B.5.2 最优解

| 符号 | 含义 | 说明 |
|------|------|------|
| $f^*$ | 最优目标值 | $f(\mathbf{x}^*)$，即目标函数在最优点处的值 |
| $\mathbf{x}^*$ | 最优解（最优点）| 使目标函数取最优值的变量值 |
| $\arg\min_{\mathbf{x}} f(\mathbf{x})$ | 最小值点集合 | $\{\mathbf{x} \mid f(\mathbf{x}) = \inf_{\mathbf{z}} f(\mathbf{z})\}$ |
| $\arg\max_{\mathbf{x}} f(\mathbf{x})$ | 最大值点集合 | $\{\mathbf{x} \mid f(\mathbf{x}) = \sup_{\mathbf{z}} f(\mathbf{z})\}$ |
| $\mathbf{x}^*_{\text{loc}}$ | 局部最优解 | 邻域内使目标函数取最小值的点 |
| $\boldsymbol{\lambda}^*, \boldsymbol{\nu}^*$ | 最优对偶变量 | KKT 条件中的最优乘子 |
| $\boldsymbol{\lambda} \geq 0$ | 对偶可行性 | 不等式约束对应的 Lagrange 乘子非负 |
| $\lambda_i g_i(\mathbf{x}^*) = 0$ | 互补松弛条件 | KKT 条件之一 |

### B.5.3 凸分析

| 符号 | 含义 | 说明 |
|------|------|------|
| $f^*(\mathbf{y})$ | 凸共轭函数（Fenchel 共轭）| $\sup_{\mathbf{x}} \{\langle \mathbf{y}, \mathbf{x}\rangle - f(\mathbf{x})\}$ |
| $\text{epi}(f)$ | 函数 $f$ 的上图 | $\{(\mathbf{x}, t) \mid f(\mathbf{x}) \leq t\}$ |
| $\text{lev}_\alpha(f)$ | $f$ 的 $\alpha$-水平集 | $\{\mathbf{x} \mid f(\mathbf{x}) \leq \alpha\}$ |
| $\mu$ | 强凸参数 | $f(\mathbf{y}) \geq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y}-\mathbf{x}\rangle + \frac{\mu}{2}\|\mathbf{y}-\mathbf{x}\|^2$ |
| $L$ | Lipschitz 常数（梯度光滑常数）| $\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\|$ |
| $\kappa$ | 条件数 | $\kappa = L / \mu$，衡量优化问题的难易程度 |

---

## B.6 算法参数

| 符号 | 名称 | 典型取值范围 | 说明 |
|------|------|-------------|------|
| $\alpha$ 或 $\eta$ | 学习率 / 步长 | $(0, 1)$，常用 $10^{-4}$ 至 $10^{-1}$ | 控制每次迭代的更新幅度 |
| $\alpha_k$ | 第 $k$ 步学习率 | 随迭代递减 | 衰减学习率策略 |
| $\lambda$ | 正则化系数 | $[0, +\infty)$ | 权衡拟合精度与模型复杂度 |
| $\lambda_1$ | $\ell_1$ 正则化系数 | $\geq 0$ | Lasso 惩罚项系数 |
| $\lambda_2$ | $\ell_2$ 正则化系数 | $\geq 0$ | Ridge 惩罚项系数（有时记为 $\gamma$）|
| $\beta$ 或 $\beta_1$ | 动量系数 | $[0, 1)$，常用 $0.9$ | 历史梯度信息的保留比例（Adam 第一矩）|
| $\beta_2$ | 自适应学习率衰减系数 | $[0, 1)$，常用 $0.999$ | Adam 中第二矩的衰减系数 |
| $\epsilon$ | 数值稳定项 | $10^{-8}$ | 防止除零，Adam 中常用 |
| $\rho$ | 衰减率 | $[0, 1)$ | RMSProp 中梯度平方均值的衰减系数 |
| $\gamma$ | 学习率衰减因子 | $(0, 1]$ | 学习率调度中的乘法因子 |
| $\tau$ | 温度 / 软化参数 | $> 0$ | 软最大化等操作中的温度系数 |
| $T$ | 总迭代次数 | $\mathbb{Z}_+$ | 算法运行的最大轮数 |
| $B$ | 批大小（batch size）| $\mathbb{Z}_+$，常用 $32$、$64$、$128$ | 每次梯度计算使用的样本数 |
| $n$ | 总样本数 | $\mathbb{Z}_+$ | 训练集大小 |
| $t$ | 迭代轮次索引 | $0, 1, 2, \ldots$ | 当前迭代次数 |
| $c_1, c_2$ | Wolfe 条件参数 | $0 < c_1 < c_2 < 1$ | 线搜索中的充分下降与曲率条件参数 |

---

## B.7 概率与期望

| 符号 | 含义 | 说明 |
|------|------|------|
| $\mathbb{P}(A)$ | 事件 $A$ 的概率 | $\mathbb{P}: \mathcal{F} \to [0,1]$ |
| $\mathbb{P}(A \mid B)$ | 条件概率 | 给定事件 $B$ 发生时 $A$ 的概率 |
| $\mathbb{E}[X]$ | 随机变量 $X$ 的期望 | $\int x \, dP(x)$ 或 $\sum_x x P(X=x)$ |
| $\mathbb{E}_{\mathbf{x} \sim p}[f(\mathbf{x})]$ | 对分布 $p$ 的期望 | 显式注明采样分布 |
| $\text{Var}(X)$ | 方差 | $\mathbb{E}[(X - \mathbb{E}[X])^2]$ |
| $\text{Cov}(X, Y)$ | 协方差 | $\mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$ |
| $\boldsymbol{\Sigma}$ | 协方差矩阵 | $[\boldsymbol{\Sigma}]_{ij} = \text{Cov}(X_i, X_j)$ |
| $p(\mathbf{x})$ 或 $p_\theta(\mathbf{x})$ | 概率密度 / 质量函数 | 下标 $\theta$ 表示参数化分布 |
| $\hat{\nabla} f$ | 随机梯度 | 真实梯度 $\nabla f$ 的无偏或有偏估计 |
| $\mathbb{E}[\hat{\nabla} f] = \nabla f$ | 无偏性 | SGD 收敛分析的关键假设 |
| $\sigma^2$ | 梯度方差上界 | $\mathbb{E}[\|\hat{\nabla} f - \nabla f\|^2] \leq \sigma^2$ |
| $\overset{d}{=}$ | 分布相等 | 两个随机变量具有相同分布 |
| $X \sim \mathcal{N}(\mu, \sigma^2)$ | 正态分布 | 均值 $\mu$、方差 $\sigma^2$ 的正态分布 |

---

## B.8 下标与上标约定

### B.8.1 迭代索引

| 约定 | 含义 | 示例 |
|------|------|------|
| 上标 $^{(t)}$ 或 $^k$ | 第 $t$（或 $k$）次迭代 | $\mathbf{x}^{(t)}$、$\mathbf{x}^k$ 表示第 $t$ 步的迭代点 |
| $\mathbf{x}^{(t+1)}$ | 下一迭代步的变量值 | 更新后的参数 |
| $\nabla f^{(t)}$ | 第 $t$ 步的梯度 | $\nabla f(\mathbf{x}^{(t)})$ 的简写 |
| $\alpha^{(t)}$ 或 $\alpha_t$ | 第 $t$ 步学习率 | 时变学习率 |

### B.8.2 神经网络层索引

| 约定 | 含义 | 示例 |
|------|------|------|
| 上标 $^{[l]}$ | 第 $l$ 层 | $\mathbf{W}^{[l]}$ 为第 $l$ 层权重矩阵 |
| $\mathbf{a}^{[l]}$ | 第 $l$ 层激活值 | — |
| $\mathbf{z}^{[l]}$ | 第 $l$ 层预激活值 | $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$ |
| $\mathbf{b}^{[l]}$ | 第 $l$ 层偏置向量 | — |
| $L$ | 总层数 | 神经网络的深度 |

### B.8.3 批次与样本索引

| 约定 | 含义 | 示例 |
|------|------|------|
| 上标 $(i)$ | 第 $i$ 个训练样本 | $\mathbf{x}^{(i)}$ 为第 $i$ 个输入样本 |
| 下标 $i$ 或 $j$ | 向量 / 矩阵的分量索引 | $x_i$ 为向量 $\mathbf{x}$ 的第 $i$ 个分量 |
| $\mathcal{B}$ 或 $\mathcal{B}_t$ | 第 $t$ 步的小批量 | 随机选取的样本下标集合 |
| $|\mathcal{B}|$ | 批大小 | 小批量中的样本数 |
| $\hat{y}^{(i)}$ | 第 $i$ 个样本的预测值 | 区别于真实标签 $y^{(i)}$ |

### B.8.4 其他常用约定

| 约定 | 含义 |
|------|------|
| 星号上标 $^*$ | 最优值或最优解，如 $\mathbf{x}^*$、$f^*$ |
| 波浪号 $\tilde{\cdot}$ | 近似量，如 $\tilde{f}$ 为 $f$ 的近似 |
| 帽子 $\hat{\cdot}$ | 估计量，如 $\hat{\mathbf{x}}$ 为 $\mathbf{x}$ 的估计 |
| 横线 $\bar{\cdot}$ | 均值 / 平均量，如 $\bar{\mathbf{x}} = \frac{1}{T}\sum_t \mathbf{x}^{(t)}$ |

---

## B.9 特殊函数

| 符号 | 名称 | 定义 / 说明 |
|------|------|------------|
| $\text{sign}(x)$ | 符号函数 | $\begin{cases} 1 & x > 0 \\ 0 & x = 0 \\ -1 & x < 0 \end{cases}$ |
| $[x]_+$ 或 $\max(x, 0)$ | 正部截断 | ReLU 激活函数 |
| $\sigma(x)$ | Sigmoid 激活函数 | $\dfrac{1}{1+e^{-x}}$，值域 $(0, 1)$ |
| $\tanh(x)$ | 双曲正切激活函数 | $\dfrac{e^x - e^{-x}}{e^x + e^{-x}}$，值域 $(-1, 1)$ |
| $\text{ReLU}(x)$ | 线性整流函数 | $\max(0, x)$ |
| $\text{softmax}(\mathbf{x})_i$ | Softmax 函数 | $\dfrac{e^{x_i}}{\sum_j e^{x_j}}$，输出概率向量 |
| $\text{prox}_{\alpha f}(\mathbf{v})$ | 近端算子（proximal operator）| $\arg\min_{\mathbf{x}} \left\{f(\mathbf{x}) + \dfrac{1}{2\alpha}\|\mathbf{x}-\mathbf{v}\|^2\right\}$ |
| $\text{prox}_{\alpha\|\cdot\|_1}(\mathbf{v})$ | 软阈值算子 | $\text{sign}(v_i)\max(|v_i|-\alpha, 0)$（逐元素）|
| $\text{proj}_{\mathcal{C}}(\mathbf{v})$ | 投影算子 | $\arg\min_{\mathbf{x} \in \mathcal{C}} \|\mathbf{x} - \mathbf{v}\|$，将 $\mathbf{v}$ 投影到集合 $\mathcal{C}$ |
| $\delta_{\mathcal{C}}(\mathbf{x})$ | 示性函数（indicator function）| $\begin{cases} 0 & \mathbf{x} \in \mathcal{C} \\ +\infty & \mathbf{x} \notin \mathcal{C} \end{cases}$ |
| $\iota_{\mathcal{C}}(\mathbf{x})$ | 示性函数（备用记法）| 同 $\delta_{\mathcal{C}}(\mathbf{x})$ |
| $\text{dist}(\mathbf{x}, \mathcal{C})$ | 点到集合的距离 | $\inf_{\mathbf{y} \in \mathcal{C}} \|\mathbf{x} - \mathbf{y}\|$ |
| $\ell(\hat{y}, y)$ | 损失函数 | 预测值 $\hat{y}$ 与真实值 $y$ 之差的度量 |
| $\mathcal{L}(\theta)$ 或 $J(\theta)$ | 整体损失 / 代价函数 | 在全部训练数据上的平均损失 |
| $\log$ | 自然对数 | $\ln$（底数为 $e$），除非另有说明 |
| $\log_2$ | 以 $2$ 为底的对数 | 信息论中常用 |
| $\exp(x)$ 或 $e^x$ | 指数函数 | — |
| $\lceil x \rceil$ | 上取整 | 不小于 $x$ 的最小整数 |
| $\lfloor x \rfloor$ | 下取整 | 不大于 $x$ 的最大整数 |
| $O(\cdot)$ | 大 $O$ 符号 | 渐近上界，$f(n) = O(g(n))$ 表示 $f$ 的增长速度不快于 $g$ |
| $\Omega(\cdot)$ | 大 $\Omega$ 符号 | 渐近下界 |
| $\Theta(\cdot)$ | 大 $\Theta$ 符号 | 渐近紧界 |

---

## B.10 缩写对照表

### B.10.1 优化算法

| 缩写 | 全称 | 中文说明 |
|------|------|---------|
| GD | Gradient Descent | 梯度下降法 |
| SGD | Stochastic Gradient Descent | 随机梯度下降法 |
| mini-batch SGD | Mini-batch Stochastic Gradient Descent | 小批量随机梯度下降法 |
| NGD | Natural Gradient Descent | 自然梯度下降法 |
| HB | Heavy Ball | 重球法（Polyak 动量）|
| NAG | Nesterov Accelerated Gradient | Nesterov 加速梯度法 |
| RMSProp | Root Mean Square Propagation | 均方根传播法 |
| Adam | Adaptive Moment Estimation | 自适应矩估计 |
| AdaGrad | Adaptive Gradient Algorithm | 自适应梯度算法 |
| AdaDelta | Adaptive Delta | 自适应增量法 |
| AMSGrad | AMS Gradient | Adam 的改进变体（保证收敛）|
| LBFGS | Limited-memory BFGS | 有限内存 BFGS 算法 |
| BFGS | Broyden–Fletcher–Goldfarb–Shanno | 拟牛顿法之一 |
| DFP | Davidon–Fletcher–Powell | 拟牛顿法之一 |
| Newton-CG | Newton Conjugate-Gradient | 牛顿共轭梯度法 |
| CG | Conjugate Gradient | 共轭梯度法 |
| CD | Coordinate Descent | 坐标下降法 |
| BCD | Block Coordinate Descent | 块坐标下降法 |
| PG | Proximal Gradient | 近端梯度法 |
| ISTA | Iterative Shrinkage-Thresholding Algorithm | 迭代收缩阈值算法 |
| FISTA | Fast ISTA | 快速迭代收缩阈值算法 |
| ADMM | Alternating Direction Method of Multipliers | 交替方向乘子法 |
| SVRG | Stochastic Variance Reduced Gradient | 随机方差缩减梯度法 |
| SAG | Stochastic Average Gradient | 随机平均梯度法 |
| SAGA | — | SAG 的改进版本（无偏梯度估计）|
| SARAH | StochAstic Recursive grAdient algoritHm | 随机递归梯度算法 |
| SpiderBoost | — | 基于 Spider 估计器的方差缩减方法 |

### B.10.2 最优性条件与理论

| 缩写 | 全称 | 中文说明 |
|------|------|---------|
| KKT | Karush–Kuhn–Tucker | KKT 最优性条件 |
| LICQ | Linear Independence Constraint Qualification | 线性无关约束品质 |
| MFCQ | Mangasarian–Fromovitz Constraint Qualification | MF 约束品质 |
| CQ | Constraint Qualification | 约束品质条件（通称）|
| LP | Linear Programming | 线性规划 |
| QP | Quadratic Programming | 二次规划 |
| SDP | Semidefinite Programming | 半正定规划 |
| SOCP | Second-Order Cone Programming | 二阶锥规划 |
| NLP | Nonlinear Programming | 非线性规划 |
| MIP | Mixed-Integer Programming | 混合整数规划 |
| CO | Combinatorial Optimization | 组合优化 |
| DP | Dynamic Programming | 动态规划 |

### B.10.3 机器学习与统计

| 缩写 | 全称 | 中文说明 |
|------|------|---------|
| ML | Machine Learning | 机器学习 |
| DL | Deep Learning | 深度学习 |
| ERM | Empirical Risk Minimization | 经验风险最小化 |
| SRM | Structural Risk Minimization | 结构风险最小化 |
| SVM | Support Vector Machine | 支持向量机 |
| SVM-QP | SVM Quadratic Programming | 支持向量机的二次规划形式 |
| MAP | Maximum A Posteriori | 最大后验估计 |
| MLE | Maximum Likelihood Estimation | 最大似然估计 |
| PCA | Principal Component Analysis | 主成分分析 |
| Lasso | Least Absolute Shrinkage and Selection Operator | 最小绝对值收缩与选择算子 |
| Ridge | — | 岭回归（$\ell_2$ 正则化线性回归）|
| ElasticNet | — | 弹性网络（$\ell_1 + \ell_2$ 正则化）|
| BN | Batch Normalization | 批归一化 |
| LN | Layer Normalization | 层归一化 |
| WD | Weight Decay | 权重衰减（等价于 $\ell_2$ 正则化）|
| LR | Learning Rate | 学习率 |

### B.10.4 其他常用缩写

| 缩写 | 全称 | 中文说明 |
|------|------|---------|
| w.r.t. | with respect to | 关于（对……求导 / 优化）|
| i.i.d. | independent and identically distributed | 独立同分布 |
| a.s. | almost surely | 几乎必然（概率论）|
| w.p.1 | with probability one | 以概率 $1$（同 a.s.）|
| s.t. | subject to | 约束于（优化问题格式）|
| WLOG | without loss of generality | 不失一般性 |
| iff | if and only if | 当且仅当 |
| LHS / RHS | left-hand side / right-hand side | 左侧 / 右侧 |

---

## B.11 希腊字母速查

| 大写 | 小写 | 名称 | 在本教程中的常见用途 |
|------|------|------|-------------------|
| $A$ | $\alpha$ | alpha | 学习率、步长、Wolfe 条件参数 |
| $B$ | $\beta$ | beta | 动量系数、Adam 衰减系数 |
| $\Gamma$ | $\gamma$ | gamma | 学习率衰减因子、折扣因子 |
| $\Delta$ | $\delta$ | delta | 变化量 $\Delta\mathbf{x}$、信任域半径 |
| $E$ | $\epsilon$ | epsilon | 数值稳定项、收敛容差 |
| $Z$ | $\zeta$ | zeta | — |
| $H$ | $\eta$ | eta | 学习率（与 $\alpha$ 等价使用）|
| $\Theta$ | $\theta$ | theta | 模型参数（通用）|
| $I$ | $\iota$ | iota | 示性函数（有时用）|
| $K$ | $\kappa$ | kappa | 条件数 |
| $\Lambda$ | $\lambda$ | lambda | 正则化系数、Lagrange 乘子、特征值 |
| $M$ | $\mu$ | mu | 强凸参数、均值 |
| $N$ | $\nu$ | nu | 等式约束的 Lagrange 乘子 |
| $\Xi$ | $\xi$ | xi | 随机变量（有时用）|
| $\Pi$ | $\pi$ | pi | 策略（强化学习）、圆周率 |
| $P$ | $\rho$ | rho | 步长比、RMSProp 衰减率 |
| $\Sigma$ | $\sigma$ | sigma | 奇异值、标准差、激活函数（Sigmoid）|
| $T$ | $\tau$ | tau | 温度参数、时间步 |
| $\Upsilon$ | $\upsilon$ | upsilon | — |
| $\Phi$ | $\phi$ | phi | 特征映射、约束函数 |
| $X$ | $\chi$ | chi | — |
| $\Psi$ | $\psi$ | psi | — |
| $\Omega$ | $\omega$ | omega | 权重（有时用）|

---

*本符号表将随教程更新持续完善。如遇符号含义不明，请参考相应章节的首次定义。*
