# 第16章：随机梯度下降

---

## 学习目标

学完本章后，你将能够：

1. **区分期望风险与经验风险**：理解机器学习优化目标的统计本质，掌握从总体分布到有限样本近似的建模思路，以及随机优化与确定性优化的根本区别
2. **理解并推导无偏梯度估计**：掌握随机梯度作为真实梯度无偏估计量的理论依据，以及为何 SGD 在期望意义下沿正确方向下降
3. **分析方差对收敛的影响**：理解梯度噪声的来源与量化方式，掌握方差减小（Variance Reduction）的核心思想，以及梯度噪声与收敛速度的内在联系
4. **掌握 SGD 的 $O(1/\sqrt{T})$ 收敛率**：在凸函数假设下完整理解 SGD 的收敛性证明框架，与确定性梯度下降的 $O(1/T)$ 率做定量比较，理解代价与收益的权衡
5. **掌握 Mini-batch 与并行化策略**：理解批量大小如何影响方差、收敛率和计算效率，以及大批量训练的理论与实践权衡，能在 PyTorch 中高效实现 Mini-batch SGD

---

## 16.1 随机优化问题

### 16.1.1 机器学习中的优化：从总体到样本

机器学习的核心目标是找到一个模型参数 $\boldsymbol{\theta}$，使其在**未见过的数据**上表现良好。设数据服从未知分布 $\mathcal{D}$，每个样本为 $(\mathbf{x}, y) \sim \mathcal{D}$，我们真正想最小化的是**期望风险**（Expected Risk）：

$$\boxed{R(\boldsymbol{\theta}) = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}}\left[\ell(\boldsymbol{\theta}; \mathbf{x}, y)\right]}$$

其中 $\ell(\boldsymbol{\theta}; \mathbf{x}, y)$ 是单样本损失（如均方误差、交叉熵）。

**问题**：我们无法直接计算 $R(\boldsymbol{\theta})$，因为分布 $\mathcal{D}$ 未知。

**实践替代**：给定 $n$ 个训练样本 $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$，用**经验风险**（Empirical Risk）近似：

$$\hat{R}_n(\boldsymbol{\theta}) = \frac{1}{n} \sum_{i=1}^n \ell(\boldsymbol{\theta}; \mathbf{x}_i, y_i)$$

这就是**经验风险最小化**（ERM）框架，绝大多数机器学习模型的训练目标都可以归结为此。

### 16.1.2 期望风险 vs 经验风险

两者之间的关系与区别至关重要：

| 概念 | 表达式 | 性质 |
|------|--------|------|
| 期望风险 $R(\boldsymbol{\theta})$ | $\mathbb{E}_{\mathcal{D}}[\ell(\boldsymbol{\theta}; \mathbf{x}, y)]$ | 真实目标，不可计算 |
| 经验风险 $\hat{R}_n(\boldsymbol{\theta})$ | $\frac{1}{n}\sum_{i=1}^n \ell(\boldsymbol{\theta}; \mathbf{x}_i, y_i)$ | 可计算的近似 |
| 泛化误差 | $R(\boldsymbol{\theta}) - \hat{R}_n(\boldsymbol{\theta})$ | 统计学习理论的核心研究对象 |

由大数定律，当 $n \to \infty$ 时，$\hat{R}_n(\boldsymbol{\theta}) \to R(\boldsymbol{\theta})$（逐点收敛）。

**深层联系**：SGD 直接优化的是经验风险 $\hat{R}_n$，但由于每步只用一个（或少量）样本，其梯度估计实际上在逼近对**整个训练集**的梯度，而训练集本身又是对真实分布的近似。SGD 因此天然具有一定的正则化效果（梯度噪声抑制了对训练集的过拟合）。

### 16.1.3 有限和结构与随机优化

经验风险具有**有限和**（Finite Sum）结构：

$$\hat{R}_n(\boldsymbol{\theta}) = \frac{1}{n} \sum_{i=1}^n f_i(\boldsymbol{\theta}), \quad f_i(\boldsymbol{\theta}) \triangleq \ell(\boldsymbol{\theta}; \mathbf{x}_i, y_i)$$

**批量梯度下降**（Full Gradient Descent）：

$$\nabla \hat{R}_n(\boldsymbol{\theta}) = \frac{1}{n} \sum_{i=1}^n \nabla f_i(\boldsymbol{\theta})$$

每次迭代需遍历全部 $n$ 个样本，计算代价为 $O(n)$。当 $n = 10^6$（现代数据集的典型规模）时，每步梯度计算极其昂贵。

**随机优化的核心思想**：用少量样本的梯度代替全梯度，将每步计算代价降至 $O(1)$ 或 $O(B)$（$B$ 为批量大小），以允许更多次迭代换取更低的单步代价。

### 16.1.4 在线学习视角

SGD 还可以从**在线学习**（Online Learning）角度理解：假设数据以流式方式逐个到达，模型在每个样本到达时立即更新一次。此时：

- 无需存储全部数据
- 可处理无穷数据流（对应期望风险最小化）
- 迭代公式为：$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha_t \nabla f_{\xi_t}(\boldsymbol{\theta}_t)$，其中 $\xi_t$ 是第 $t$ 步到达的随机样本

在线学习视角下，SGD 直接优化期望风险 $R(\boldsymbol{\theta})$，而非经验风险 $\hat{R}_n$，这使得 SGD 天然具有泛化优势。

---

## 16.2 SGD 算法

### 16.2.1 算法定义

**随机梯度下降**（Stochastic Gradient Descent，SGD）在每次迭代中随机选取一个（或少量）样本，用其梯度代替全梯度：

$$\boxed{\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha_t \tilde{\nabla} f(\boldsymbol{\theta}_t)}$$

其中 $\tilde{\nabla} f(\boldsymbol{\theta}_t)$ 是梯度的随机估计，$\alpha_t > 0$ 是第 $t$ 步的学习率（步长）。

**算法流程（SGD）**：

```
输入：初始点 θ_0，学习率序列 {α_t}，迭代次数 T
重复（t = 0, 1, ..., T-1）：
    随机选取样本下标 i_t ∈ {1, ..., n}（均匀分布）
    计算随机梯度：g_t = ∇f_{i_t}(θ_t)
    更新参数：θ_{t+1} = θ_t - α_t * g_t
输出：θ_T 或 θ_avg = (1/T) Σ θ_t（运行均值）
```

### 16.2.2 无偏梯度估计

SGD 有效性的数学基础是**无偏性**（Unbiasedness）。

**命题 16.1（随机梯度的无偏性）**：设样本 $i_t$ 从 $\{1, \ldots, n\}$ 中均匀随机选取，则：

$$\mathbb{E}_{i_t}\!\left[\nabla f_{i_t}(\boldsymbol{\theta}_t)\right] = \frac{1}{n} \sum_{i=1}^n \nabla f_i(\boldsymbol{\theta}_t) = \nabla \hat{R}_n(\boldsymbol{\theta}_t)$$

**证明**：直接由期望的线性性和均匀分布：

$$\mathbb{E}_{i_t}\!\left[\nabla f_{i_t}(\boldsymbol{\theta}_t)\right] = \sum_{i=1}^n \frac{1}{n} \nabla f_i(\boldsymbol{\theta}_t) = \frac{1}{n}\sum_{i=1}^n \nabla f_i(\boldsymbol{\theta}_t) = \nabla \hat{R}_n(\boldsymbol{\theta}_t) \qquad \square$$

**重要性**：无偏性保证了 SGD 在**期望意义下**与梯度下降方向一致，这是收敛性分析的关键前提。

在线学习视角下，若样本 $\xi_t \sim \mathcal{D}$ 独立同分布，则：

$$\mathbb{E}_{\xi_t}\!\left[\nabla_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta}_t; \xi_t)\right] = \nabla_{\boldsymbol{\theta}} R(\boldsymbol{\theta}_t)$$

SGD 直接估计了期望风险的梯度，理论上等价于直接对真实目标做梯度下降。

### 16.2.3 学习率策略

学习率 $\alpha_t$ 的选择对 SGD 收敛至关重要。与确定性梯度下降不同，SGD **不能使用过小的固定学习率**（否则梯度噪声导致参数在最优解附近随机游走，无法真正收敛），也**不能使用过大的固定学习率**（导致震荡发散）。

**常见学习率策略**：

**（1）固定学习率**：$\alpha_t = \alpha$（常数）

SGD 以固定学习率不会精确收敛到最优解，而是收敛到最优解的一个**邻域**，邻域大小由 $\alpha$ 控制。适合大规模深度学习训练（配合批量归一化等技巧）。

**（2）多项式衰减**：$\alpha_t = \frac{\alpha_0}{t^\gamma}$，$\gamma \in (0, 1]$

最常见的是 $\gamma = 1$（调和序列）：

$$\alpha_t = \frac{\alpha_0}{t}$$

理论上满足 Robbins-Monro 条件（见 16.4 节），保证精确收敛。

**（3）分段常数衰减**：每隔若干轮将学习率乘以衰减因子 $\gamma < 1$

深度学习实践中最常用：

$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / \text{step} \rfloor}$$

**（4）余弦退火**（Cosine Annealing）：

$$\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\!\frac{\pi t}{T}\right)$$

在深度学习中广泛使用，允许学习率周期性"重启"（Warm Restart）逃离局部极小。

### 16.2.4 与批量梯度下降的比较

| 特性 | 批量梯度下降（GD） | 随机梯度下降（SGD） |
|------|-------------------|---------------------|
| 每步梯度 | 精确：$\nabla \hat{R}_n(\boldsymbol{\theta})$ | 随机：$\nabla f_{i_t}(\boldsymbol{\theta})$ |
| 每步计算量 | $O(n)$ | $O(1)$ |
| 同等时间内迭代数 | $1$ | $n$ |
| 梯度噪声 | 无 | 有（方差 $\sigma^2$） |
| 收敛速度（凸函数） | $O(1/T)$ | $O(1/\sqrt{T})$ |
| 内存需求 | 需存全部梯度 | 极低 |
| 逃离局部极小 | 弱（确定性） | 强（噪声有益） |

在深度学习中，批量梯度下降几乎从不使用；所谓的"SGD"通常指 **Mini-batch SGD**（见 16.5 节）。

---

## 16.3 方差与噪声

### 16.3.1 随机梯度的方差

随机梯度 $\tilde{g}_t = \nabla f_{i_t}(\boldsymbol{\theta}_t)$ 是全梯度 $g_t = \nabla \hat{R}_n(\boldsymbol{\theta}_t)$ 的无偏估计，但存在**方差**：

$$\sigma_t^2 = \mathbb{E}\!\left[\|\tilde{g}_t - g_t\|^2\right] = \mathbb{E}\!\left[\|\nabla f_{i_t}(\boldsymbol{\theta}_t) - \nabla \hat{R}_n(\boldsymbol{\theta}_t)\|^2\right]$$

方差 $\sigma_t^2$ 衡量了随机梯度偏离真实方向的程度，是分析 SGD 收敛性的核心量。

**有界方差假设**：在大多数收敛性分析中，假设存在常数 $\sigma^2 > 0$，使得对所有 $\boldsymbol{\theta}$ 和所有 $t$：

$$\mathbb{E}\!\left[\|\tilde{g}_t\|^2\right] \leq \sigma^2 + \|\nabla \hat{R}_n(\boldsymbol{\theta}_t)\|^2$$

等价地，存在 $\sigma^2$ 使得：

$$\mathbb{E}\!\left[\|\tilde{g}_t - \nabla \hat{R}_n(\boldsymbol{\theta}_t)\|^2\right] \leq \sigma^2$$

这是一个合理的假设：当梯度有界（如损失函数的梯度被特征范数控制）时自然成立。

### 16.3.2 方差的几何直觉

方差的影响可以从以下角度理解：

**SGD 更新的分解**：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha_t \underbrace{\nabla \hat{R}_n(\boldsymbol{\theta}_t)}_{\text{信号（真实梯度）}} - \alpha_t \underbrace{(\tilde{g}_t - \nabla \hat{R}_n(\boldsymbol{\theta}_t))}_{\text{噪声（零均值）}}$$

- **信号项**使参数沿梯度方向下降（有益）
- **噪声项**使参数产生随机游走（在最优解附近有害）

当学习率固定为 $\alpha$ 时，SGD 在最优解 $\boldsymbol{\theta}^*$ 附近的"稳态误差"大约为：

$$\mathbb{E}\!\left[\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2\right] \approx \frac{\alpha \sigma^2}{2\mu}$$

其中 $\mu$ 为强凸参数。学习率越大、噪声越大，稳态误差越大。

### 16.3.3 为何方差会减小收敛速度

**直觉对比**：

- **确定性 GD**：每步沿负梯度方向移动，类比在有路灯的夜晚下山。
- **SGD**：每步方向带有随机扰动，类比在微弱光线下下山，偶尔走错方向需要纠正。

这种随机性使得 SGD 无法像 GD 一样以 $O(1/T)$ 收敛——即使在最优解附近，噪声仍然驱动参数偏离。数学上，标准凸函数 SGD 的收敛率为 $O(1/\sqrt{T})$（见 16.4 节），比 GD 慢了 $\sqrt{T}$ 倍。

然而，若将**总计算量**（而非迭代次数）作为衡量标准，SGD 的优势就显现出来：

- GD：$T$ 次迭代，每次 $O(n)$ 计算，总计 $O(nT)$，达到精度 $\epsilon = O(1/T)$
- SGD：$T$ 次迭代，每次 $O(1)$ 计算，总计 $O(T)$，达到精度 $\epsilon = O(1/\sqrt{T})$

若令两者达到相同精度 $\epsilon$：GD 需要 $O(n/\epsilon)$ 计算，SGD 只需 $O(1/\epsilon^2)$ 计算。当 $n \gg 1/\epsilon$ 时（大数据集、低精度需求），SGD 更优。

### 16.3.4 梯度噪声的有益性

在非凸优化（如深度学习）中，梯度噪声并非总是有害的：

1. **逃离鞍点**：确定性 GD 可能在鞍点附近停滞，而 SGD 的随机扰动有助于逃离
2. **逃离尖锐局部极小**：噪声使 SGD 更倾向于收敛到"平坦"（flat）极小值，这类点通常泛化性更好
3. **隐式正则化**：SGD 的噪声对模型参数有类似 L2 正则化的隐式约束效果

这些性质使 SGD 在深度学习中成为一流优化器，即便存在更快收敛的确定性方法。

---

## 16.4 收敛性分析

### 16.4.1 基本假设

**假设 16.1（$L$-光滑性）**：$\hat{R}_n$ 的梯度满足 Lipschitz 条件：

$$\|\nabla \hat{R}_n(\boldsymbol{\theta}) - \nabla \hat{R}_n(\boldsymbol{\phi})\| \leq L\|\boldsymbol{\theta} - \boldsymbol{\phi}\|, \quad \forall \boldsymbol{\theta}, \boldsymbol{\phi}$$

等价地：

$$\hat{R}_n(\boldsymbol{\phi}) \leq \hat{R}_n(\boldsymbol{\theta}) + \langle \nabla \hat{R}_n(\boldsymbol{\theta}), \boldsymbol{\phi} - \boldsymbol{\theta} \rangle + \frac{L}{2}\|\boldsymbol{\phi} - \boldsymbol{\theta}\|^2$$

**假设 16.2（有界方差）**：随机梯度的方差一致有界：

$$\mathbb{E}\!\left[\|\tilde{g}_t - \nabla \hat{R}_n(\boldsymbol{\theta}_t)\|^2\right] \leq \sigma^2, \quad \forall t$$

**假设 16.3（凸性）**（在凸函数分析中额外需要）：$\hat{R}_n$ 是凸函数：

$$\hat{R}_n(\boldsymbol{\phi}) \geq \hat{R}_n(\boldsymbol{\theta}) + \langle \nabla \hat{R}_n(\boldsymbol{\theta}), \boldsymbol{\phi} - \boldsymbol{\theta} \rangle, \quad \forall \boldsymbol{\theta}, \boldsymbol{\phi}$$

### 16.4.2 Robbins-Monro 条件

收敛到精确最优解要求学习率满足 **Robbins-Monro 条件**：

$$\sum_{t=1}^\infty \alpha_t = +\infty \quad \text{（步长之和无穷大，保证能到达任何点）}$$

$$\sum_{t=1}^\infty \alpha_t^2 < +\infty \quad \text{（步长平方之和有限，保证噪声逐步消除）}$$

典型满足条件的选择：$\alpha_t = \frac{c}{t}$（$c > 0$）。此时 $\sum \alpha_t = c \sum 1/t = +\infty$，$\sum \alpha_t^2 = c^2 \sum 1/t^2 < +\infty$。

直观上：第一个条件保证步长不会"过早死亡"（能持续移动）；第二个条件保证随机噪声的累积效果可控（噪声被逐渐压制）。

### 16.4.3 凸函数的 $O(1/\sqrt{T})$ 收敛率

**定理 16.1（SGD 收敛率，凸函数）**：设 $\hat{R}_n$ 满足假设 16.1–16.3，$\boldsymbol{\theta}^* = \arg\min \hat{R}_n$，初始距离 $\|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\| \leq D$。取固定学习率 $\alpha_t = \alpha = \dfrac{D}{\sigma\sqrt{T}}$，则 SGD 的运行均值 $\bar{\boldsymbol{\theta}}_T = \dfrac{1}{T}\sum_{t=0}^{T-1} \boldsymbol{\theta}_t$ 满足：

$$\mathbb{E}\!\left[\hat{R}_n(\bar{\boldsymbol{\theta}}_T)\right] - \hat{R}_n(\boldsymbol{\theta}^*) \leq \frac{D\sigma}{\sqrt{T}}$$

即收敛率为 $O\!\left(\dfrac{1}{\sqrt{T}}\right)$。

**证明框架**：

**步骤 1**：利用 $L$-光滑性建立一步下降关系。

$$\hat{R}_n(\boldsymbol{\theta}_{t+1}) \leq \hat{R}_n(\boldsymbol{\theta}_t) + \langle \nabla \hat{R}_n(\boldsymbol{\theta}_t), \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t \rangle + \frac{L}{2}\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t\|^2$$

代入 $\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t = -\alpha \tilde{g}_t$：

$$\hat{R}_n(\boldsymbol{\theta}_{t+1}) \leq \hat{R}_n(\boldsymbol{\theta}_t) - \alpha \langle \nabla \hat{R}_n(\boldsymbol{\theta}_t), \tilde{g}_t \rangle + \frac{L\alpha^2}{2}\|\tilde{g}_t\|^2$$

**步骤 2**：对随机性取期望。

$$\mathbb{E}_t\!\left[\hat{R}_n(\boldsymbol{\theta}_{t+1})\right] \leq \hat{R}_n(\boldsymbol{\theta}_t) - \alpha \|\nabla \hat{R}_n(\boldsymbol{\theta}_t)\|^2 + \frac{L\alpha^2}{2}\left(\sigma^2 + \|\nabla \hat{R}_n(\boldsymbol{\theta}_t)\|^2\right)$$

**步骤 3**：利用凸性将梯度范数与函数间隙关联。

由凸性：$\hat{R}_n(\boldsymbol{\theta}_t) - \hat{R}_n(\boldsymbol{\theta}^*) \leq \langle \nabla \hat{R}_n(\boldsymbol{\theta}_t), \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rangle$

**步骤 4**：利用"望远镜"求和技巧。

对距离递推式 $\mathbb{E}\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\|^2 \leq \mathbb{E}\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2 - 2\alpha\mathbb{E}[\hat{R}_n(\boldsymbol{\theta}_t) - \hat{R}_n(\boldsymbol{\theta}^*)] + \alpha^2(\sigma^2 + \|\nabla \hat{R}_n(\boldsymbol{\theta}_t)\|^2)$，对 $t = 0, \ldots, T-1$ 求和：

$$2\alpha \sum_{t=0}^{T-1} \mathbb{E}\!\left[\hat{R}_n(\boldsymbol{\theta}_t) - \hat{R}_n(\boldsymbol{\theta}^*)\right] \leq \|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\|^2 + \alpha^2 T\sigma^2 \leq D^2 + \alpha^2 T\sigma^2$$

**步骤 5**：取均值并代入最优学习率 $\alpha = D/(\sigma\sqrt{T})$：

$$\mathbb{E}\!\left[\hat{R}_n(\bar{\boldsymbol{\theta}}_T)\right] - \hat{R}_n(\boldsymbol{\theta}^*) \leq \frac{D^2 + \alpha^2 T\sigma^2}{2\alpha T} = \frac{D^2}{2\alpha T} + \frac{\alpha \sigma^2}{2} = \frac{D\sigma}{2\sqrt{T}} + \frac{D\sigma}{2\sqrt{T}} = \frac{D\sigma}{\sqrt{T}} \qquad \square$$

### 16.4.4 强凸函数的 $O(1/T)$ 收敛率

当 $\hat{R}_n$ 满足 $\mu$-强凸性时，SGD 可以达到更快的 $O(1/T)$ 收敛率。

**定理 16.2（SGD 收敛率，强凸函数）**：设 $\hat{R}_n$ 是 $L$-光滑、$\mu$-强凸函数，使用衰减学习率 $\alpha_t = \dfrac{2}{\mu(t+1)}$，则：

$$\mathbb{E}\!\left[\hat{R}_n(\bar{\boldsymbol{\theta}}_T^w)\right] - \hat{R}_n(\boldsymbol{\theta}^*) \leq \frac{2L\sigma^2}{\mu^2 T}$$

其中 $\bar{\boldsymbol{\theta}}_T^w = \dfrac{\sum_{t=1}^T t\,\boldsymbol{\theta}_t}{\sum_{t=1}^T t}$ 是加权运行均值。即收敛率为 $O(1/T)$。

### 16.4.5 收敛率汇总

| 问题类型 | 方法 | 收敛率 | 最优学习率 |
|----------|------|--------|-----------|
| 凸函数 | GD（确定性） | $O(1/T)$ | $1/L$ |
| 凸函数 | SGD | $O(1/\sqrt{T})$ | $D/(\sigma\sqrt{T})$ |
| 强凸函数 | GD（确定性） | $O(\rho^T)$，$\rho = 1-\mu/L$ | $1/L$ |
| 强凸函数 | SGD | $O(1/T)$ | $O(1/(\mu t))$ |
| 非凸函数 | GD（确定性） | $O(1/\sqrt{T})$（驻点） | $1/L$ |
| 非凸函数 | SGD | $O(1/\sqrt{T})$（驻点期望） | $O(1/\sqrt{T})$ |

**关键观察**：
- SGD 在凸函数上比 GD 慢（$O(1/\sqrt{T})$ vs $O(1/T)$），代价是以迭代次数计量
- 但 SGD 每次迭代的计算代价低 $n$ 倍，综合来看在大数据集上具有优势
- 非凸函数下，GD 和 SGD 在迭代次数意义上具有相同渐近率，但 SGD 的实际表现更好

### 16.4.6 下界的最优性

SGD 的 $O(1/\sqrt{T})$ 收敛率并非算法的缺陷，而是**信息论下界**（Information-theoretic Lower Bound）：

**定理 16.3（SGD 下界，非正式版本）**：对于任意使用有界方差随机梯度的一阶方法，在凸函数上，存在问题实例使得任意算法的收敛率不优于 $\Omega(1/\sqrt{T})$。

这说明 SGD 的 $O(1/\sqrt{T})$ 率是**最优的**（对一阶方法而言），无法在不改变信息获取方式的情况下进一步提升。要打破此下界，必须使用**方差减小技术**（如 SVRG、SARAH；见进阶内容）。

---

## 16.5 Mini-batch 与并行化

### 16.5.1 Mini-batch SGD

**Mini-batch SGD** 在每次迭代中随机选取 $B$ 个样本（批量，batch），用其平均梯度作为全梯度的估计：

$$\tilde{g}_t^B = \frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla f_i(\boldsymbol{\theta}_t)$$

其中 $\mathcal{B}_t$ 是第 $t$ 步从 $\{1, \ldots, n\}$ 中均匀随机选取的大小为 $B$ 的子集。

更新规则：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha_t \tilde{g}_t^B$$

Mini-batch SGD 是现代深度学习的标准训练范式，通常 $B \in \{32, 64, 128, 256\}$。

### 16.5.2 Mini-batch 的方差减小

Mini-batch 梯度相比单样本梯度有更小的方差。

**命题 16.2（Mini-batch 方差）**：设单样本梯度的方差为 $\sigma^2$，则 Mini-batch 梯度的方差为：

$$\text{Var}\!\left[\tilde{g}_t^B\right] = \frac{\sigma^2}{B}$$

**证明**：设 $\tilde{g}_i = \nabla f_i(\boldsymbol{\theta}_t)$，则各样本独立时：

$$\text{Var}\!\left[\frac{1}{B}\sum_{i \in \mathcal{B}_t} \tilde{g}_i\right] = \frac{1}{B^2} \sum_{i \in \mathcal{B}_t} \text{Var}[\tilde{g}_i] = \frac{B \sigma^2}{B^2} = \frac{\sigma^2}{B} \qquad \square$$

**直接推论**：将 Mini-batch 方差 $\sigma^2/B$ 代入定理 16.1，可得 Mini-batch SGD 的收敛率：

$$\mathbb{E}\!\left[\hat{R}_n(\bar{\boldsymbol{\theta}}_T)\right] - \hat{R}_n(\boldsymbol{\theta}^*) \leq \frac{D(\sigma/\sqrt{B})}{\sqrt{T}} = \frac{D\sigma}{\sqrt{BT}}$$

即 $O\!\left(\dfrac{1}{\sqrt{BT}}\right)$——批量大小 $B$ 直接参与改善收敛率。

### 16.5.3 批量大小与计算效率的权衡

Mini-batch SGD 在以下几个维度上实现权衡：

**（1）计算效率**：现代 GPU/TPU 高度并行，小批量和大批量的实际计算时间差异远小于理论上的 $B$ 倍。典型情况下，批量从 1 增加到 64 时，速度提升接近 $B$ 倍；但批量从 64 增加到 256 时，提升仅有 1-2 倍（受内存带宽等硬件限制）。

**（2）梯度噪声**：批量越大，梯度越精确（方差 $\sim 1/B$），但超过某个临界值后改善效果递减。

**（3）泛化性能**：大量实验表明，过大的批量（如 $B > 8192$）会导致模型泛化性能下降，即"大批量训练泛化陷阱"（Large Batch Generalization Gap）。主流解释认为大批量倾向于收敛到尖锐极小值，泛化性更差。

**（4）学习率缩放规则（Linear Scaling Rule）**：为补偿大批量的方差减小效果，实践中通常按如下规则调整学习率：

$$\alpha_{\text{large}} = \frac{B_{\text{large}}}{B_{\text{base}}} \cdot \alpha_{\text{base}}$$

即批量大小加倍，学习率也加倍（称为"线性缩放规则"，Goyal et al., 2017）。直觉上，批量增大 $k$ 倍相当于方差减小 $k$ 倍，可以用更大步长"安全地"移动。

### 16.5.4 数据并行训练

大批量训练天然支持**数据并行**（Data Parallelism）：

**同步数据并行**（Synchronous Data Parallel）：

1. 将批量 $\mathcal{B}$ 均匀分配到 $P$ 个设备，每台设备处理子批量 $\mathcal{B}^{(p)}$（$|\mathcal{B}^{(p)}| = B/P$）
2. 每台设备独立计算子批量梯度：$\tilde{g}^{(p)} = \frac{P}{B}\sum_{i \in \mathcal{B}^{(p)}} \nabla f_i(\boldsymbol{\theta}_t)$
3. **AllReduce**：跨设备对梯度求平均：$\tilde{g}_t^B = \frac{1}{P}\sum_{p=1}^P \tilde{g}^{(p)}$
4. 所有设备用相同梯度更新参数

从优化角度，同步数据并行等价于在单设备上运行批量为 $B$ 的 Mini-batch SGD。

```
设备1: θ → ∇f_{B/P个样本}(θ) ─┐
设备2: θ → ∇f_{B/P个样本}(θ) ─┤ AllReduce → g_avg → θ_{t+1}
...                            │
设备P: θ → ∇f_{B/P个样本}(θ) ─┘
```

**梯度累积**（Gradient Accumulation）：当显存不足时，通过 $k$ 次前向-后向传播（每次批量 $B/k$）累积梯度再更新，等效于批量 $B$，无需增加显存：

```python
optimizer.zero_grad()
for micro_batch in split(batch, k):
    loss = model(micro_batch) / k
    loss.backward()        # 梯度自动累积
optimizer.step()           # 一次更新
```

### 16.5.5 随机洗牌与 Epoch

**Epoch**（轮次）是遍历一次完整训练集的过程。实践中 SGD 通常以 Epoch 为单位组织训练：

```
对每个 Epoch：
    随机洗牌训练数据
    对每个 Mini-batch（批量 B）：
        计算梯度并更新参数
```

**为何随机洗牌重要**：不洗牌时，相邻 Mini-batch 的样本高度相关，梯度估计的有效方差增大；洗牌保证每步梯度尽量接近独立同分布的采样，与理论分析的假设一致。

**无放回采样 vs 有放回采样**：

- **有放回**（理论分析通常假设）：每步独立随机选取，满足无偏性和独立性条件，理论分析简洁
- **无放回（洗牌）**（实践标准）：每个 Epoch 内每个样本恰好被使用一次，梯度估计方差更小，实践效果更好

---

## 本章小结

| 概念 | 公式/要点 |
|------|-----------|
| 期望风险 | $R(\boldsymbol{\theta}) = \mathbb{E}_{\mathcal{D}}[\ell(\boldsymbol{\theta}; \mathbf{x}, y)]$，真实目标不可计算 |
| 经验风险 | $\hat{R}_n(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n f_i(\boldsymbol{\theta})$，有限样本近似 |
| SGD 更新 | $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha_t \nabla f_{i_t}(\boldsymbol{\theta}_t)$ |
| 无偏性 | $\mathbb{E}_{i_t}[\nabla f_{i_t}(\boldsymbol{\theta}_t)] = \nabla \hat{R}_n(\boldsymbol{\theta}_t)$ |
| 有界方差假设 | $\mathbb{E}\|\tilde{g}_t - \nabla \hat{R}_n\|^2 \leq \sigma^2$ |
| 凸函数收敛率 | $O(1/\sqrt{T})$，学习率 $\alpha = D/(\sigma\sqrt{T})$ |
| 强凸函数收敛率 | $O(1/T)$，学习率 $\alpha_t = 2/(\mu(t+1))$ |
| Robbins-Monro | $\sum \alpha_t = \infty$，$\sum \alpha_t^2 < \infty$（精确收敛条件） |
| Mini-batch 方差 | $\text{Var}[\tilde{g}^B] = \sigma^2/B$，方差与批量成反比 |
| Mini-batch 收敛率 | $O(1/\sqrt{BT})$，批量大小改善收敛 |
| 线性缩放规则 | 批量加倍，学习率加倍，保持有效步长不变 |

**核心要点回顾**：

1. SGD 以单样本随机梯度代替全梯度，将每步计算代价从 $O(n)$ 降至 $O(1)$，适合大规模优化
2. 随机梯度是全梯度的无偏估计，保证 SGD 在期望意义下沿正确方向下降
3. 梯度方差是制约 SGD 收敛速度的核心因素，导致凸函数上 $O(1/\sqrt{T})$（慢于确定性 GD 的 $O(1/T)$）
4. $O(1/\sqrt{T})$ 是有界方差一阶方法的信息论最优率，不可改进（在不减小方差的前提下）
5. Mini-batch 将方差降低 $B$ 倍，是精度与效率的最佳平衡，是现代深度学习的标准训练范式

---

## 深度学习应用：深度学习训练的核心算法

### 背景

深度神经网络的训练本质上是求解：

$$\min_{\boldsymbol{\theta} \in \mathbb{R}^d} \mathcal{L}(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n \ell(f(\mathbf{x}_i; \boldsymbol{\theta}),\ y_i)$$

其中：
- $\boldsymbol{\theta}$ 是网络参数（$d$ 可达数十亿）
- $n$ 是训练样本数（可达数百亿）
- $f(\cdot; \boldsymbol{\theta})$ 是神经网络的计算图
- $\ell$ 是逐样本损失（分类用交叉熵，回归用 MSE）

这是一个**大规模非凸随机优化**问题，Mini-batch SGD 是目前为止最有效的求解算法。

### PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 1. 生成实验数据：多类分类
# ============================================================

def make_classification_dataset(
    n_samples: int = 2000,
    n_features: int = 20,
    n_classes: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成线性可分的多类分类数据集"""
    X_list, y_list = [], []
    for c in range(n_classes):
        center = torch.randn(n_features) * 3
        X_c = center + torch.randn(n_samples // n_classes, n_features)
        y_c = torch.full((n_samples // n_classes,), c, dtype=torch.long)
        X_list.append(X_c)
        y_list.append(y_c)
    return torch.cat(X_list), torch.cat(y_list)


X_all, y_all = make_classification_dataset()
n_total = len(X_all)
n_train = int(0.8 * n_total)

# 划分训练/验证集
perm = torch.randperm(n_total)
X_train, y_train = X_all[perm[:n_train]], y_all[perm[:n_train]]
X_val, y_val = X_all[perm[n_train:]], y_all[perm[n_train:]]

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
print(f"特征维度: {X_train.shape[1]}, 类别数: 5")


# ============================================================
# 2. 定义神经网络
# ============================================================

class MLP(nn.Module):
    """多层感知机（MLP）用于多类分类"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 3. Mini-batch SGD 训练循环（标准深度学习流程）
# ============================================================

def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loader: data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """训练一个 Epoch，返回平均损失"""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # SGD 三步：清零梯度 → 前向传播 → 反向传播 → 更新参数
        optimizer.zero_grad()                    # 清零上一步累积梯度
        logits = model(X_batch)                  # 前向传播
        loss = criterion(logits, y_batch)        # 计算损失
        loss.backward()                          # 反向传播（自动微分）
        optimizer.step()                         # 梯度下降更新 θ ← θ - α·∇L

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """计算损失和准确率"""
    model.eval()
    X, y = X.to(device), y.to(device)
    logits = model(X)
    loss = criterion(logits, y).item()
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


def run_experiment(
    batch_size: int,
    lr: float,
    lr_schedule: str = "constant",
    n_epochs: int = 80,
    device: torch.device = torch.device("cpu")
) -> dict:
    """
    运行单个训练实验

    参数:
        batch_size: Mini-batch 大小
        lr: 初始学习率
        lr_schedule: 学习率策略 ("constant" / "step" / "cosine")
        n_epochs: 训练轮次
    """
    # 构建数据加载器（随机洗牌）
    dataset = data.TensorDataset(X_train, y_train)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 模型与优化器
    model = MLP(in_dim=20, hidden_dim=64, out_dim=5).to(device)
    # momentum=0.9：标准 SGD 加动量（几乎所有深度学习都用动量）
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 学习率调度器
    if lr_schedule == "step":
        # 每 30 轮学习率乘以 0.1
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif lr_schedule == "cosine":
        # 余弦退火
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    else:
        scheduler = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, optimizer, loader, criterion, device)
        val_loss, val_acc = evaluate(model, X_val, y_val, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if scheduler is not None:
            scheduler.step()

    return history


# ============================================================
# 4. 实验一：不同批量大小对收敛的影响
# ============================================================

print("\n实验一：不同批量大小对收敛的影响")
print("-" * 50)

batch_sizes = [8, 32, 128, 512]
batch_results = {}

for B in batch_sizes:
    hist = run_experiment(batch_size=B, lr=0.01 * (B / 32) ** 0.5, lr_schedule="cosine")
    batch_results[B] = hist
    final_acc = hist["val_acc"][-1]
    print(f"  批量 B={B:4d}，有效学习率={0.01*(B/32)**0.5:.4f}，"
          f"最终验证准确率={final_acc:.4f}")


# ============================================================
# 5. 实验二：不同学习率调度策略
# ============================================================

print("\n实验二：不同学习率调度策略")
print("-" * 50)

schedule_configs = {
    "固定学习率 (α=0.01)": {"lr": 0.01, "lr_schedule": "constant"},
    "分段衰减 (StepLR)":    {"lr": 0.1,  "lr_schedule": "step"},
    "余弦退火 (Cosine)":    {"lr": 0.1,  "lr_schedule": "cosine"},
}

schedule_results = {}
for name, cfg in schedule_configs.items():
    hist = run_experiment(batch_size=64, **cfg)
    schedule_results[name] = hist
    final_acc = hist["val_acc"][-1]
    print(f"  {name}：最终验证准确率={final_acc:.4f}")


# ============================================================
# 6. 可视化
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1：不同批量大小 - 训练损失（按 Epoch）
ax = axes[0, 0]
for B, hist in batch_results.items():
    ax.plot(hist["train_loss"], label=f"B={B}", linewidth=1.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("训练损失")
ax.set_title("不同批量大小的训练损失曲线")
ax.legend()
ax.grid(True, alpha=0.3)

# 子图2：不同批量大小 - 验证准确率
ax = axes[0, 1]
for B, hist in batch_results.items():
    ax.plot(hist["val_acc"], label=f"B={B}", linewidth=1.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("验证准确率")
ax.set_title("不同批量大小的验证准确率")
ax.legend()
ax.grid(True, alpha=0.3)

# 子图3：不同学习率调度 - 验证损失
ax = axes[1, 0]
for name, hist in schedule_results.items():
    ax.plot(hist["val_loss"], label=name, linewidth=1.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("验证损失")
ax.set_title("不同学习率调度策略的验证损失")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 子图4：学习率随 Epoch 的变化
ax = axes[1, 1]
for name, hist in schedule_results.items():
    ax.plot(hist["lr"], label=name, linewidth=1.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("学习率 α")
ax.set_title("学习率随训练轮次的变化")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

plt.tight_layout()
plt.savefig("sgd_experiments.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n图像已保存为 sgd_experiments.png")


# ============================================================
# 7. 梯度方差的实验验证：批量大小与方差的 1/B 关系
# ============================================================

print("\n实验三：验证 Mini-batch 方差 ∝ 1/B")
print("-" * 50)

# 固定一个已初始化的模型，在不同批量大小下估计梯度方差
model_fixed = MLP(in_dim=20, hidden_dim=64, out_dim=5)
criterion = nn.CrossEntropyLoss()

def estimate_gradient_variance(model, X, y, batch_size, n_trials=100):
    """
    通过多次随机采样估计给定批量大小下的梯度方差。
    返回第一层权重梯度的方差（标量，作为方差的代理指标）。
    """
    grads = []
    for _ in range(n_trials):
        idx = torch.randperm(len(X))[:batch_size]
        X_b, y_b = X[idx], y[idx]

        model.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()

        # 取第一层权重梯度的 Frobenius 范数平方作为方差代理
        g = list(model.parameters())[0].grad.detach().clone()
        grads.append(g.norm().item() ** 2)

    return np.var(grads)

batch_sizes_var = [1, 4, 16, 64, 256]
variances = []
for B in batch_sizes_var:
    var = estimate_gradient_variance(model_fixed, X_train, y_train, B, n_trials=50)
    variances.append(var)
    print(f"  B={B:4d}：梯度方差代理 = {var:.4f}")

# 验证 1/B 关系
print("\n  方差比（相对 B=1）：")
for i, B in enumerate(batch_sizes_var):
    ratio = variances[i] / variances[0]
    theoretical = 1.0 / B
    print(f"  B={B:4d}：实测比={ratio:.4f}，理论值 1/B={theoretical:.4f}")


# ============================================================
# 8. 梯度累积：模拟大批量训练
# ============================================================

print("\n示例：梯度累积（等效大批量训练）")
print("-" * 50)

def train_with_gradient_accumulation(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    effective_batch: int,   # 有效批量大小
    accumulation_steps: int, # 累积步数
    lr: float = 0.01,
    n_epochs: int = 20
) -> List[float]:
    """
    梯度累积训练。
    micro_batch = effective_batch // accumulation_steps
    等效于在 effective_batch 上训练，但每次只加载 micro_batch 个样本，
    节省显存。
    """
    micro_batch = effective_batch // accumulation_steps
    dataset = data.TensorDataset(X, y)
    loader = data.DataLoader(dataset, batch_size=micro_batch, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        step_count = 0

        for batch_idx, (X_b, y_b) in enumerate(loader):
            # 损失除以累积步数，等效于对 effective_batch 取平均
            loss = criterion(model(X_b), y_b) / accumulation_steps
            loss.backward()
            step_count += 1

            # 每累积 accumulation_steps 步执行一次参数更新
            if step_count % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step_count = 0

            epoch_loss += loss.item() * accumulation_steps

        losses.append(epoch_loss / len(loader))

    return losses

# 比较：直接大批量 vs 梯度累积（有效批量均为 256）
model_direct = MLP(in_dim=20, hidden_dim=64, out_dim=5)
model_accum = MLP(in_dim=20, hidden_dim=64, out_dim=5)
# 两个模型使用相同初始化
model_accum.load_state_dict(model_direct.state_dict())

# 直接大批量（B=256）
loader_large = data.DataLoader(
    data.TensorDataset(X_train, y_train), batch_size=256, shuffle=True
)
optimizer_direct = optim.SGD(model_direct.parameters(), lr=0.01 * (256/32)**0.5, momentum=0.9)
criterion = nn.CrossEntropyLoss()
losses_direct = []
for _ in range(20):
    ep_loss = 0.0
    for Xb, yb in loader_large:
        optimizer_direct.zero_grad()
        loss = criterion(model_direct(Xb), yb)
        loss.backward()
        optimizer_direct.step()
        ep_loss += loss.item()
    losses_direct.append(ep_loss / len(loader_large))

# 梯度累积（micro_batch=32，累积8步 → 有效批量256）
losses_accum = train_with_gradient_accumulation(
    model_accum, X_train, y_train,
    effective_batch=256, accumulation_steps=8, lr=0.01 * (256/32)**0.5, n_epochs=20
)

print(f"  直接大批量（B=256）最终损失：{losses_direct[-1]:.4f}")
print(f"  梯度累积（等效B=256）最终损失：{losses_accum[-1]:.4f}")
print("  两者损失差异（理论上应接近0）：",
      f"{abs(losses_direct[-1] - losses_accum[-1]):.4f}")
```

### 代码要点说明

1. **`optimizer.zero_grad()` 的位置**：必须在每次 `loss.backward()` 之前调用，否则梯度会跨步累积。梯度累积是特意利用这一机制，在不调用 `zero_grad` 的情况下累积多个小批量的梯度。

2. **`momentum=0.9` 的标准用法**：在深度学习中，SGD 几乎总是配合动量使用。纯 SGD（`momentum=0`）在实践中很少单独使用，动量 SGD 通过积累历史梯度显著加速收敛（见第17章）。

3. **学习率调度的重要性**：固定学习率难以兼顾早期快速收敛和后期精细收敛。余弦退火允许学习率从大到小平滑变化，实践效果通常优于分段衰减。

4. **方差验证实验**：代码中通过多次随机采样估计梯度方差，验证了理论预测的 $\text{Var} \propto 1/B$，批量增大时方差显著减小。

5. **梯度累积**：将 `effective_batch // accumulation_steps` 的微批量梯度累积后一次更新，数学上等价于直接使用 `effective_batch` 的大批量，是在显存受限时模拟大批量训练的标准技巧。

---

## 练习题

### 练习 16.1（基础）

设经验风险为 $\hat{R}_n(\theta) = \frac{1}{n}\sum_{i=1}^n f_i(\theta)$，其中 $f_i(\theta) = \frac{1}{2}(\theta - a_i)^2$，$a_i$ 为第 $i$ 个样本值。

(a) 计算经验风险 $\hat{R}_n(\theta)$ 的最优解 $\hat{\theta}^*$，证明它等于样本均值 $\bar{a} = \frac{1}{n}\sum_{i=1}^n a_i$。

(b) 写出 SGD 更新公式（每步随机选取样本 $i_t$），计算 $\mathbb{E}[\nabla f_{i_t}(\theta)]$ 并验证无偏性。

(c) 设 $n = 4$，$a = (1, 3, 5, 7)$，从 $\theta_0 = 0$ 出发，用固定学习率 $\alpha = 0.5$，手动模拟 SGD 的前4步（每步选不同样本）并观察参数的随机游走。

---

### 练习 16.2（中级）

考虑二次函数 $f(\theta) = \frac{1}{2}\theta^2$，随机梯度为 $\tilde{g}(\theta) = \theta + \xi$，其中 $\xi \sim \mathcal{N}(0, \sigma^2)$ 为独立噪声。

(a) 验证 $\tilde{g}(\theta)$ 是 $\nabla f(\theta) = \theta$ 的无偏估计。

(b) 取固定学习率 $\alpha$，从 $\theta_0$ 出发，计算 $\mathbb{E}[\theta_t]$ 和 $\mathbb{E}[\theta_t^2]$（提示：利用递推关系）。

(c) 求 $\mathbb{E}[\theta_t^2]$ 的稳态值（$t \to \infty$），说明稳态误差与 $\alpha$、$\sigma^2$ 的关系。

(d) 若要稳态误差满足 $\mathbb{E}[\theta_\infty^2] \leq \epsilon$，学习率 $\alpha$ 应满足什么条件？

---

### 练习 16.3（中级）

证明定理 16.1 的关键步骤：在凸函数假设下，SGD 满足以下距离递推式：

$$\mathbb{E}_t\!\left[\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\|^2\right] \leq \|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2 - 2\alpha_t\left(\hat{R}_n(\boldsymbol{\theta}_t) - \hat{R}_n(\boldsymbol{\theta}^*)\right) + \alpha_t^2\left(\sigma^2 + \|\nabla \hat{R}_n(\boldsymbol{\theta}_t)\|^2\right)$$

（提示：展开 $\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\|^2 = \|\boldsymbol{\theta}_t - \alpha_t \tilde{g}_t - \boldsymbol{\theta}^*\|^2$，再分别利用无偏性、凸性和有界方差假设）

---

### 练习 16.4（中级）

分析 Mini-batch SGD 在凸函数上的收敛率。

(a) 设单样本梯度方差为 $\sigma^2$，推导批量为 $B$ 的 Mini-batch 梯度的方差。

(b) 将批量 $B$ 代入定理 16.1，写出 Mini-batch SGD 的最优收敛率表达式（用 $D$、$\sigma$、$B$、$T$ 表示）。

(c) 若固定总计算量（即总样本读取次数）为 $C = BT$，Mini-batch SGD 的收敛率作为 $B$ 的函数如何变化？是否存在最优批量大小？

(d) 给出在以下两种约束下的最优批量大小策略：(i) 固定 Epoch 数（即 $T = n/B \times \text{Epochs}$）；(ii) 固定总梯度计算次数（即 $T = C/B$）。

---

### 练习 16.5（提高）

**实验题**：在 PyTorch 中实现并比较 SGD 的理论与实践。

设 $f(\theta) = \frac{1}{2}\theta^T A\theta - b^T\theta$（二次函数），其中 $A = \text{diag}(1, 2, 3, 4, 5) \in \mathbb{R}^{5\times 5}$，$b = \mathbf{1}$。为模拟随机性，每步梯度加入零均值高斯噪声 $\xi \sim \mathcal{N}(0, \sigma^2 I)$，$\sigma = 0.5$。

(a) 实现固定学习率 SGD（$\alpha = 0.05$）、多项式衰减 SGD（$\alpha_t = 1/t$）和 Mini-batch SGD（$B = 10$，$\alpha = 0.05$），运行 $T = 500$ 步，绘制 $f(\theta_t) - f^*$ 的收敛曲线（对数纵轴）。

(b) 对多次独立运行（如 20 次）结果求均值和标准差，绘制带置信区间（$\pm 1$ 标准差）的收敛曲线，观察各方法的稳定性差异。

(c) 验证：固定学习率 SGD 在最优解附近不收敛（稳态误差 $> 0$），而衰减学习率 SGD 可以收敛。

(d) 将收敛曲线与理论率 $O(1/\sqrt{T})$（凸函数）和 $O(1/T)$（强凸函数）在同一图中比较，分析与理论的吻合程度。

---

## 练习答案

### 答案 16.1

**(a)** $\hat{R}_n(\theta) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}(\theta - a_i)^2$。

对 $\theta$ 求导：

$$\frac{d\hat{R}_n}{d\theta} = \frac{1}{n}\sum_{i=1}^n (\theta - a_i) = \theta - \frac{1}{n}\sum_{i=1}^n a_i = \theta - \bar{a}$$

令导数为零，得 $\hat{\theta}^* = \bar{a} = \frac{1}{n}\sum_{i=1}^n a_i$（样本均值）。这正是经典的均方误差最小化估计量——样本均值是均方损失下的最优估计。

**(b)** $\nabla f_{i_t}(\theta) = \theta - a_{i_t}$，其中 $i_t$ 均匀分布于 $\{1, \ldots, n\}$：

$$\mathbb{E}_{i_t}[\nabla f_{i_t}(\theta)] = \frac{1}{n}\sum_{i=1}^n (\theta - a_i) = \theta - \bar{a} = \nabla \hat{R}_n(\theta) \qquad \text{（无偏性）}\checkmark$$

**(c)** $a = (1, 3, 5, 7)$，$\bar{a} = 4 = \hat{\theta}^*$，$\alpha = 0.5$，$\theta_0 = 0$：

按顺序选样本 $i = 1, 2, 3, 4$（模拟随机选取）：

- $t=0$，选 $a_1 = 1$：$g = 0 - 1 = -1$，$\theta_1 = 0 - 0.5 \times (-1) = 0.5$
- $t=1$，选 $a_2 = 3$：$g = 0.5 - 3 = -2.5$，$\theta_2 = 0.5 - 0.5 \times (-2.5) = 1.75$
- $t=2$，选 $a_3 = 5$：$g = 1.75 - 5 = -3.25$，$\theta_3 = 1.75 - 0.5 \times (-3.25) = 3.375$
- $t=3$，选 $a_4 = 7$：$g = 3.375 - 7 = -3.625$，$\theta_4 = 3.375 - 0.5 \times (-3.625) = 5.1875$

观察到参数逐步向最优解 $\hat{\theta}^* = 4$ 靠近，但存在随机游走；4步后超过了最优解（$5.1875 > 4$），体现了 SGD 的随机性。若持续训练并适当衰减学习率，参数会收敛至 $4$。

---

### 答案 16.2

**(a)** $\mathbb{E}[\tilde{g}(\theta)] = \mathbb{E}[\theta + \xi] = \theta + 0 = \theta = \nabla f(\theta)$。无偏性成立。

**(b)** 更新规则：$\theta_{t+1} = \theta_t - \alpha(\theta_t + \xi_t) = (1-\alpha)\theta_t - \alpha\xi_t$

- **期望**：$\mathbb{E}[\theta_{t+1}] = (1-\alpha)\mathbb{E}[\theta_t]$（因 $\mathbb{E}[\xi_t] = 0$）

  递推：$\mathbb{E}[\theta_t] = (1-\alpha)^t \theta_0$。当 $\alpha \in (0, 2)$ 时，$|1-\alpha| < 1$，$\mathbb{E}[\theta_t] \to 0$（期望收敛）。

- **二阶矩**：$\mathbb{E}[\theta_{t+1}^2] = (1-\alpha)^2\mathbb{E}[\theta_t^2] + \alpha^2\sigma^2$

  递推求解（等比数列）：

$$\mathbb{E}[\theta_t^2] = (1-\alpha)^{2t}\theta_0^2 + \alpha^2\sigma^2 \cdot \frac{1-(1-\alpha)^{2t}}{1-(1-\alpha)^2}$$

**(c)** 令 $t \to \infty$（假设 $\alpha \in (0, 2)$，$(1-\alpha)^{2t} \to 0$）：

$$\mathbb{E}[\theta_\infty^2] = \frac{\alpha^2\sigma^2}{1-(1-\alpha)^2} = \frac{\alpha^2\sigma^2}{2\alpha - \alpha^2} = \frac{\alpha\sigma^2}{2-\alpha} \approx \frac{\alpha\sigma^2}{2} \quad (\alpha \ll 1)$$

**结论**：稳态误差 $\approx \alpha\sigma^2/2$，与学习率 $\alpha$ 和方差 $\sigma^2$ 均成正比。学习率越大、噪声越大，收敛精度越差。

**(d)** 要使 $\mathbb{E}[\theta_\infty^2] \leq \epsilon$：

$$\frac{\alpha\sigma^2}{2-\alpha} \leq \epsilon \implies \alpha \leq \frac{2\epsilon}{\sigma^2 + \epsilon} \approx \frac{2\epsilon}{\sigma^2} \quad (\epsilon \ll \sigma^2)$$

因此精度要求越高（$\epsilon$ 越小），学习率上界越小，SGD 需要更谨慎的步长。

---

### 答案 16.3

**证明**：

展开 $\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\|^2$：

$$\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\|^2 = \|\boldsymbol{\theta}_t - \alpha_t \tilde{g}_t - \boldsymbol{\theta}^*\|^2$$

$$= \|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2 - 2\alpha_t \langle \tilde{g}_t, \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rangle + \alpha_t^2\|\tilde{g}_t\|^2$$

对 $t$ 时刻的随机性（给定 $\boldsymbol{\theta}_t$）取条件期望，利用 $\mathbb{E}_t[\tilde{g}_t] = \nabla \hat{R}_n(\boldsymbol{\theta}_t)$：

$$\mathbb{E}_t\!\left[\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\|^2\right] = \|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2 - 2\alpha_t \langle \nabla \hat{R}_n(\boldsymbol{\theta}_t), \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rangle + \alpha_t^2 \mathbb{E}_t\!\left[\|\tilde{g}_t\|^2\right]$$

**利用凸性**：$\langle \nabla \hat{R}_n(\boldsymbol{\theta}_t), \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rangle \geq \hat{R}_n(\boldsymbol{\theta}_t) - \hat{R}_n(\boldsymbol{\theta}^*)$，故：

$$-2\alpha_t \langle \nabla \hat{R}_n(\boldsymbol{\theta}_t), \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rangle \leq -2\alpha_t\left(\hat{R}_n(\boldsymbol{\theta}_t) - \hat{R}_n(\boldsymbol{\theta}^*)\right)$$

**利用有界方差**：$\mathbb{E}_t[\|\tilde{g}_t\|^2] = \mathbb{E}_t[\|\tilde{g}_t - \nabla \hat{R}_n(\boldsymbol{\theta}_t) + \nabla \hat{R}_n(\boldsymbol{\theta}_t)\|^2] \leq \sigma^2 + \|\nabla \hat{R}_n(\boldsymbol{\theta}_t)\|^2$

综合两个不等式：

$$\mathbb{E}_t\!\left[\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\|^2\right] \leq \|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2 - 2\alpha_t\left(\hat{R}_n(\boldsymbol{\theta}_t) - \hat{R}_n(\boldsymbol{\theta}^*)\right) + \alpha_t^2\left(\sigma^2 + \|\nabla \hat{R}_n(\boldsymbol{\theta}_t)\|^2\right) \qquad \square$$

---

### 答案 16.4

**(a)** 设 $\mathcal{B}_t = \{i_1, \ldots, i_B\}$ 为均匀无放回（近似独立）采样，$\tilde{g}^B = \frac{1}{B}\sum_{j=1}^B \nabla f_{i_j}(\boldsymbol{\theta})$：

$$\text{Var}\!\left[\tilde{g}^B\right] = \frac{1}{B^2}\sum_{j=1}^B \text{Var}\!\left[\nabla f_{i_j}\right] = \frac{B\sigma^2}{B^2} = \frac{\sigma^2}{B}$$

批量增大 $B$ 倍，方差降低 $B$ 倍。

**(b)** 将 $\sigma^2/B$ 代入定理 16.1（将 $\sigma$ 替换为 $\sigma/\sqrt{B}$）：

$$\mathbb{E}\!\left[\hat{R}_n(\bar{\boldsymbol{\theta}}_T)\right] - \hat{R}_n(\boldsymbol{\theta}^*) \leq \frac{D \cdot (\sigma/\sqrt{B})}{\sqrt{T}} = \frac{D\sigma}{\sqrt{BT}}$$

最优学习率为 $\alpha = D\sqrt{B}/(\sigma\sqrt{T})$（较单样本 SGD 大 $\sqrt{B}$ 倍，对应线性缩放规则）。

**(c)** 固定总计算量 $C = BT$，则 $T = C/B$：

$$\frac{D\sigma}{\sqrt{BT}} = \frac{D\sigma}{\sqrt{B \cdot (C/B)}} = \frac{D\sigma}{\sqrt{C}}$$

收敛率 $O(1/\sqrt{C})$ 与批量大小 $B$ **无关**！在固定总计算量的约束下，任何批量大小都给出相同的理论收敛率。因此不存在理论上的最优批量大小——批量大小的选择更多是由硬件效率（GPU 并行度）决定的。

**(d)**
- **(i) 固定 Epoch 数**（$E$ 轮）：$T = nE/B$，收敛率 $\sim 1/\sqrt{BT} = 1/\sqrt{nE}$，与 $B$ 无关。实践上，大 $B$ 的每 Epoch 计算量不变，但梯度更精确，因此每 Epoch 进步更大，应尽可能增大 $B$（受硬件和泛化性限制）。
- **(ii) 固定总梯度计算次数**（$C$ 次）：如 (c) 所示，$1/\sqrt{C}$ 与 $B$ 无关，此时选择 $B$ 主要看 GPU 并行效率，通常在 $B = 64 \sim 256$ 之间。

---

### 答案 16.5

**关键分析结论**（完整代码实现参考本章深度学习应用部分）：

对二次函数 $f(\theta) = \frac{1}{2}\theta^T A\theta - b^T\theta$，$A = \text{diag}(1,2,3,4,5)$：

- $L = \lambda_{\max}(A) = 5$，$\mu = \lambda_{\min}(A) = 1$，$\kappa = 5$（强凸）
- 最优解 $\theta^* = A^{-1}b = (1, 1/2, 1/3, 1/4, 1/5)^T$，$f^* = -\frac{1}{2}b^T\theta^*$

**(a/b) 预期收敛行为**：

| 方法 | 稳态行为 | 收敛率（理论） |
|------|----------|---------------|
| 固定学习率 SGD ($\alpha=0.05$) | 不收敛，在 $O(\alpha\sigma^2/\mu)$ 邻域随机游走 | — |
| 衰减学习率 SGD ($\alpha_t = 1/t$) | 精确收敛 | $O(1/T)$（强凸） |
| Mini-batch SGD ($B=10$) | 方差减小10倍，收敛域缩小 | $O(1/\sqrt{10T})$ |

**(c) 稳态误差验证**：固定 $\alpha = 0.05$，$\sigma = 0.5$，稳态误差理论值 $\approx \alpha\sigma^2/(2\mu) = 0.05 \times 0.25 / 2 = 0.00625$，实验中应在此量级附近波动（不再单调下降）。

**(d) 与理论率对比**：

- 强凸函数（$\mu = 1$）确定性 GD 线性收敛：$O((1-1/\kappa)^T) = O(0.8^T)$
- 强凸函数 SGD（衰减率）：$O(1/T)$，图中应表现为对数轴上斜率 $-1$ 的直线
- 凸函数 SGD（固定率）：$O(1/\sqrt{T})$，对数轴上斜率 $-1/2$ 的直线

实验曲线的初始阶段通常比理论率更快（常数因子有利），后期（固定学习率时）会在稳态误差处饱和。

---

*本章结束。下一章：[动量方法与加速梯度下降](./17-momentum-methods.md)*
