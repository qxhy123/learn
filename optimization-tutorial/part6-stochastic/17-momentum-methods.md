# 第17章：动量方法

> **前置知识**：第5章（梯度下降）、第6章（Newton方法）、第16章（随机梯度下降）
>
> **本章难度**：★★★★☆
>
> **预计学习时间**：6-8 小时

---

## 学习目标

学完本章后，你将能够：

- 从物理直觉出发，理解动量方法为何能加速优化，并将"惯性"概念映射到参数更新规则上
- 掌握经典动量（Polyak重球法）的迭代格式 $v_{t+1} = \beta v_t + \nabla f(x_t)$，并理解动量系数 $\beta$ 的调节作用
- 深刻理解Nesterov加速梯度（NAG）的"预测-校正"思路，区分其与经典动量的本质差别
- 对强凸光滑函数证明动量方法的最优收敛率 $O\!\left(\left(1-\sqrt{1/\kappa}\right)^t\right)$，理解为何加速是"最优"的
- 在深度学习实践中正确使用带动量的SGD，掌握 $\beta$ 的选择原则与热身调度策略，并能用PyTorch复现典型训练曲线

---

## 17.1 动量的物理直觉

### 17.1.1 从物理学到优化

想象一个小球在山坡上滚动。普通梯度下降好比一只**没有惯性的虫子**——每次只沿当前最陡的方向挪动一小步，完全不记得自己之前的运动方向。当坡面弯弯曲曲时，虫子会反复折返，走出"之"字形轨迹。

而现实中的球拥有**惯性（momentum）**：它会把之前积累的速度延续下去，不会因为当前坡度的细微变化而骤然转向。这使得它在平缓方向上越滚越快，在振荡方向上自然抵消。

**核心类比**：

| 物理量 | 优化对应 |
|--------|---------|
| 质点位置 $x$ | 参数 $\theta$ |
| 速度 $v$ | 参数更新量（动量向量） |
| 力 $F = -\nabla U$ | 负梯度 $-\nabla f(\theta)$ |
| 阻尼系数 $\gamma$ | 动量衰减系数 $\beta$ |
| 质量 $m$ | 影响步长的超参数 |

### 17.1.2 梯度下降的病态条件问题

考虑一个经典的**病态二次函数**：

$$f(x_1, x_2) = \frac{1}{2}(ax_1^2 + bx_2^2), \quad a \ll b$$

其等值线是极扁的椭圆，条件数 $\kappa = b/a \gg 1$。

**普通梯度下降**的更新为：

$$x_1^{(t+1)} = x_1^{(t)} - \eta a x_1^{(t)}, \quad x_2^{(t+1)} = x_2^{(t)} - \eta b x_2^{(t)}$$

为使 $x_2$ 方向收敛，需要 $\eta \leq 2/b$；但此时 $x_1$ 方向的收缩率为 $|1 - \eta a| \approx 1 - a/b = 1 - 1/\kappa$，收敛极慢。**步数量级为 $O(\kappa)$**。

**直觉上**，我们希望：
- 在曲率低（梯度变化慢）的方向上加速前进
- 在曲率高（梯度来回振荡）的方向上相互抵消

这正是动量方法的核心机制。

### 17.1.3 动量的"低通滤波"视角

将梯度更新看作信号，动量相当于对梯度历史做**指数移动平均（EMA）**：

$$v_t = \sum_{k=0}^{t} \beta^{t-k} g_k, \quad g_k = \nabla f(x_k)$$

其中 $\beta \in (0, 1)$ 是衰减因子。这是一个**低通滤波器**：

- **低频成分**（持续的梯度方向）被放大，有效步长增大为 $\frac{1}{1-\beta}$ 倍
- **高频成分**（来回振荡的噪声方向）被平均抵消

当 $\beta = 0.9$ 时，有效步长约为普通梯度下降的 $\frac{1}{1-0.9} = 10$ 倍——这就是为什么深度学习中 $\beta = 0.9$ 是默认值。

> **关键洞察**：动量不是万能加速器。它加速的前提是梯度方向保持一致；若梯度方向随机翻转（纯噪声），动量反而无益。

---

## 17.2 经典动量（Polyak动量）

### 17.2.1 重球法（Heavy Ball Method）

Boris Polyak 于1964年提出**重球法（Heavy Ball Method）**，其名称来自物理类比——像一个在摩擦地面上运动的重球。

**迭代格式**：

$$\boxed{v_{t+1} = \beta v_t + \nabla f(x_t)}$$

$$\boxed{x_{t+1} = x_t - \alpha v_{t+1}}$$

其中：
- $v_t \in \mathbb{R}^d$：动量向量（"速度"）
- $\beta \in [0, 1)$：动量系数（"衰减率"），通常取 $0.9$
- $\alpha > 0$：学习率（"步长"）
- 初始化 $v_0 = \mathbf{0}$

**等价展开形式**：将 $v$ 展开代入：

$$x_{t+1} = x_t - \alpha \nabla f(x_t) - \alpha \beta v_t = x_t - \alpha \nabla f(x_t) + \beta (x_t - x_{t-1})$$

最后一步利用了 $v_t = (x_t - x_{t-1})/\alpha \cdot \alpha$。这揭示了另一种直观理解：**新的更新 = 梯度下降步 + 惯性项（上一步位移的 $\beta$ 倍）**。

> **另一种常见写法**（深度学习框架使用）：
>
> $$v_{t+1} = \beta v_t + (1-\beta)\nabla f(x_t)$$
>
> 这是将梯度乘以 $(1-\beta)$ 以保持速度的物理量纲一致，此时有效步长为 $\alpha/(1-\beta)$。两种写法等价，区别在于 $\alpha$ 和 $\beta$ 的吸收方式。

### 17.2.2 动量系数 $\beta$ 的作用

**定理 17.1（二次函数上的重球法）**：对强凸二次函数 $f(x) = \frac{1}{2}x^TAx$（$A$ 的特征值在 $[m, L]$ 内），取最优参数：

$$\alpha^* = \frac{4}{(\sqrt{L}+\sqrt{m})^2}, \quad \beta^* = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2$$

此时收敛率为：

$$\|x_t - x^*\| \leq \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^t \|x_0 - x^*\|$$

**对比梯度下降**：梯度下降的最优收敛率为 $\left(\frac{\kappa-1}{\kappa+1}\right)^t \approx \left(1 - \frac{2}{\kappa}\right)^t$。

当 $\kappa$ 很大时：

$$\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1} \approx 1 - \frac{2}{\sqrt{\kappa}} \quad \text{（动量方法）}$$

$$\frac{\kappa-1}{\kappa+1} \approx 1 - \frac{2}{\kappa} \quad \text{（梯度下降）}$$

动量方法将收敛速度从 $O(\kappa)$ 步降低到 $O(\sqrt{\kappa})$ 步——**这是数量级的提升**！

**$\beta$ 的选择直觉**：

| $\beta$ 值 | 效果 |
|-----------|------|
| $\beta = 0$ | 退化为普通梯度下降 |
| $\beta = 0.9$ | 深度学习默认值，平滑效果好 |
| $\beta = 0.99$ | 更强惯性，适合梯度噪声大的场景，但需小心过冲 |
| $\beta \to 1$ | 不稳定，动量积累过大导致振荡 |

### 17.2.3 动量的几何解释

以二维二次函数为例，设 $f(x, y) = \frac{1}{2}(x^2 + 10y^2)$（条件数 $\kappa = 10$）。

**梯度下降**的路径：在 $y$ 方向剧烈振荡，$x$ 方向缓慢前进。

**动量方法**：$y$ 方向的振荡梯度相互抵消（$+\partial f/\partial y$ 和 $-\partial f/\partial y$ 交替出现），动量向量在 $y$ 方向趋近于零；$x$ 方向的梯度符号一致，动量持续累积，步长不断增大。

**关键等式**（稳态分析）：当 $y$ 方向振荡时，连续两步的动量贡献为：

$$\beta \cdot g_y + g_y' \approx \beta \cdot g_y - g_y = (\beta - 1)g_y$$

当 $\beta \to 1$ 时这个量趋近于零，$y$ 方向的振荡被完全抑制。

### 17.2.4 重球法的局限性

尽管在二次函数上表现完美，重球法在非凸问题上存在**理论局限**：

1. **没有一般非凸函数的加速保证**：最优参数 $(\alpha^*, \beta^*)$ 依赖于全局曲率信息 $(m, L)$，而这在非凸问题中不存在统一的值。

2. **可能绕过极小值**：动量过大时，算法可能"飞越"极小值点，在其周围振荡甚至发散。

3. **对学习率更敏感**：相比梯度下降，重球法的稳定性条件更严格。

这些局限促使 Nesterov 在1983年提出了更优雅的加速方案。

---

## 17.3 Nesterov加速梯度

### 17.3.1 Nesterov方法的直觉：先预测，再校正

Yurii Nesterov 于1983年提出了一种惊人的方法，将凸函数优化从 $O(1/t)$ 加速到 $O(1/t^2)$。

**核心思想**：普通动量是在当前位置 $x_t$ 计算梯度；Nesterov方法是先用动量做一个**预测步**，到达"未来位置" $y_t$，然后在 $y_t$ 处计算梯度进行校正。

**直觉**：如果我们知道下一步会往哪走（由动量决定），为什么不直接在那个位置"看"一眼梯度，而非在当前位置？这就像开车时往前看而非只看脚下。

### 17.3.2 Nesterov加速梯度（NAG）的迭代格式

**标准形式**（"预测-校正"视角）：

$$\boxed{y_t = x_t + \frac{\beta_t - 1}{\beta_{t+1}}(x_t - x_{t-1})}$$

$$\boxed{x_{t+1} = y_t - \alpha \nabla f(y_t)}$$

其中动量系数 $\{\beta_t\}$ 是一个特殊序列（见下文）。

**等价的速度形式**（更便于实现）：

$$v_{t+1} = \beta v_t + \nabla f(x_t + \beta v_t)$$

$$x_{t+1} = x_t - \alpha v_{t+1}$$

注意梯度是在**预测位置** $\tilde{x}_t = x_t + \beta v_t$ 处计算，而非当前位置 $x_t$。

**与经典动量的对比**：

| | 经典动量 | Nesterov动量 |
|--|---------|------------|
| 梯度计算位置 | $x_t$（当前位置）| $x_t + \beta v_t$（预测位置）|
| 直觉 | 走了再看路 | 先预测走到哪，再决定方向 |
| 凸函数收敛率 | $O(1/t)$（无加速保证）| $O(1/t^2)$（最优！）|
| 强凸收敛率 | $O(\rho^t)$，$\rho = 1 - 1/\sqrt{\kappa}$ | 同上，但理论更严格 |

### 17.3.3 动量系数序列的设计

Nesterov方法中，动量系数 $\{\lambda_t\}$（或 $\{\beta_t\}$）不是常数，而是一个**精心设计的增长序列**：

**经典选择**（FISTA风格）：

$$\lambda_0 = 1, \quad \lambda_{t+1} = \frac{1 + \sqrt{1 + 4\lambda_t^2}}{2}$$

对应的动量系数：

$$\theta_t = \frac{\lambda_t - 1}{\lambda_{t+1}}$$

当 $t \to \infty$ 时，$\lambda_t \approx t/2$，$\theta_t \approx \frac{t-2}{t+1} \to 1^-$。

**直觉**：随着迭代进行，算法越来越"相信"动量的方向，动量系数从小到大逐渐增大，越来越充分利用历史信息。

**常数动量版本**（深度学习常用）：固定 $\beta \in [0.8, 0.99]$，牺牲理论最优性换取超参数简洁：

$$v_{t+1} = \beta v_t + \nabla f(x_t + \beta v_t)$$

$$x_{t+1} = x_t - \alpha v_{t+1}$$

### 17.3.4 FISTA：稀疏优化中的Nesterov方法

**问题**：LASSO（$L_1$ 正则化）的目标函数 $f(x) = \frac{1}{2}\|Ax-b\|^2 + \lambda\|x\|_1$ 不可微。

**ISTA（迭代收缩阈值算法）**：每步做梯度步 + 近端算子（软阈值）。

**FISTA（Fast ISTA）**：在ISTA基础上加入Nesterov动量，收敛率从 $O(1/t)$ 提升至 $O(1/t^2)$：

$$y_{t+1} = \text{prox}_{\alpha\lambda\|\cdot\|_1}(x_t - \alpha A^T(Ax_t - b))$$

$$x_{t+1} = y_{t+1} + \frac{\lambda_t - 1}{\lambda_{t+1}}(y_{t+1} - y_t)$$

这是Nesterov方法在非光滑优化中的经典应用，验证了加速思想的普适性。

---

## 17.4 动量的收敛分析

### 17.4.1 凸函数的 $O(1/t^2)$ 下界

**定理 17.2（Nesterov下界，1983）**：对任意一阶方法（仅利用梯度信息），在 $L$-光滑凸函数的最坏情形下，达到精度 $\epsilon$ 所需的梯度查询次数至少为：

$$\Omega\!\left(\sqrt{\frac{L}{\epsilon}}\right)$$

**推论**：凸光滑函数的最优收敛率为 $O(1/t^2)$，而梯度下降只能达到 $O(1/t)$。

Nesterov方法是**信息论意义上最优的一阶方法**——任何一阶方法都不能比它快（在最坏情况下）。

### 17.4.2 Nesterov方法的 $O(1/t^2)$ 上界

**定理 17.3（Nesterov加速，凸情形）**：设 $f$ 是 $L$-光滑凸函数，$f^*$ 是最小值。取步长 $\alpha = 1/L$，运行Nesterov方法 $t$ 步后：

$$f(x_t) - f^* \leq \frac{2L\|x_0 - x^*\|^2}{(t+1)^2}$$

**证明思路（势函数法）**：

定义势函数（Lyapunov函数）：

$$\Phi_t = \lambda_t^2(f(x_t) - f^*) + \frac{L}{2}\|z_t - x^*\|^2$$

其中 $z_t$ 是一个辅助序列。证明 $\Phi_t$ 单调不增：

1. **梯度步的进展**：由 $L$-光滑性（下降引理），梯度步使得目标值下降 $\frac{1}{2L}\|\nabla f(y_t)\|^2$。

2. **动量步的抵消**：精心设计的 $\lambda_t$ 序列使得每步进展恰好补偿势函数中的 $\lambda_t^2$ 增长。

3. **迭代求和**：$\lambda_t^2 \geq (t+1)^2/4$，从而 $f(x_t) - f^* \leq \Phi_t/\lambda_t^2 = O(1/t^2)$。

完整证明约需3页，核心是以下不等式链：

$$\underbrace{\lambda_{t+1}^2(f(x_{t+1})-f^*)}_{\text{新势函数}} \leq \underbrace{\lambda_t^2(f(x_t)-f^*)}_{\text{旧势函数}} - \underbrace{\lambda_{t+1}(\lambda_{t+1}-1)(f(x_t)-f^*)}_{\geq 0} - \underbrace{\text{（梯度进展）}}_{\geq 0}$$

### 17.4.3 强凸情形的线性收敛

**定理 17.4（强凸加速）**：设 $f$ 是 $m$-强凸且 $L$-光滑的函数，条件数 $\kappa = L/m$。Nesterov方法以常数动量 $\beta = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$ 和步长 $\alpha = \frac{1}{L}$ 运行，满足：

$$f(x_t) - f^* \leq L\|x_0 - x^*\|^2 \cdot \left(1 - \frac{1}{\sqrt{\kappa}}\right)^t$$

**对比梯度下降**（强凸）：收敛率 $\left(1 - \frac{1}{\kappa}\right)^t$。

**加速倍数**：在 $\kappa = 10^4$ 的问题上，达到 $\epsilon = 10^{-4}$ 精度：
- 梯度下降：约 $\kappa \ln(1/\epsilon) \approx 10^4 \times 9 \approx 90000$ 步
- Nesterov方法：约 $\sqrt{\kappa} \ln(1/\epsilon) \approx 100 \times 9 = 900$ 步

**加速了约100倍**（$\sqrt{\kappa}$ 倍）。

### 17.4.4 理解加速的本质

为什么Nesterov方法能超越梯度下降的"自然速率"？

**关键机制**：Nesterov方法实际上在每步隐式地利用了**更多历史信息**——不仅仅是当前梯度，而是通过动量积累了历史梯度的加权平均。这使得它等效于在一个更"全局"的近似函数上做优化。

**从多项式近似角度**（Chebyshev多项式）：

最优化方法可以视为用一个多项式 $P_t(\lambda)$ 来近似函数，其中 $\lambda$ 是 Hessian 的特征值。梯度下降对应一次多项式（每步只用当前信息），而Nesterov方法可以利用 $t$ 次多项式，其误差界与 **Chebyshev多项式**的极小极大性质相关：

$$\min_{\text{$t$阶多项式} P_t, P_t(0)=1} \max_{\lambda \in [m, L]} |P_t(\lambda)| = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^t$$

这恰好是Nesterov方法的收敛率——它是**代数意义上最优的一阶方法**。

### 17.4.5 随机情形下的动量

在随机梯度下降（SGD）中，梯度 $\tilde{g}_t = \nabla f(x_t) + \xi_t$ 带有随机噪声 $\xi_t$。

**关键观察**：Nesterov方法的 $O(1/t^2)$ 加速**不能**移植到随机情形。原因：

1. 梯度噪声通过动量**累积**（而非平均），放大了噪声影响
2. 理论上，在随机情形下，$O(1/\sqrt{t})$（无方差缩减时）或 $O(1/t)$（有方差缩减时）已是最优

**实践中**，动量仍然有益，但作用从"加速"变为：
- 减少梯度估计的方差（类似滑动平均）
- 改善优化路径的平滑性
- 在早期迭代（接近确定性梯度时）实现实质性加速

---

## 17.5 动量在深度学习中的作用

### 17.5.1 SGD with Momentum是深度学习的基础优化器

深度学习中广泛使用的 **SGD with Momentum** 结合了随机梯度和动量：

$$v_{t+1} = \beta v_t + \nabla_\theta \mathcal{L}(\theta_t; \mathcal{B}_t)$$

$$\theta_{t+1} = \theta_t - \alpha v_{t+1}$$

其中 $\mathcal{B}_t$ 是第 $t$ 步的随机mini-batch，$\nabla_\theta \mathcal{L}(\theta_t; \mathcal{B}_t)$ 是批梯度。

**为什么深度学习偏爱SGD+Momentum而非Adam？**

在图像分类（ResNet, ViT）等任务中，SGD+Momentum 配合良好的学习率调度（余弦退火、预热）往往能达到**更好的泛化性能**，尽管Adam收敛更快。原因尚无定论，但主流假说是：

1. **平坦极小值假说**：SGD+Momentum倾向于找到"更宽"（更平坦）的极小值，这些极小值泛化更好
2. **隐式正则化**：SGD的随机性提供了隐式正则化，与动量结合后既快又稳
3. **批归一化兼容性**：BN层与SGD+Momentum的配合优于自适应方法

### 17.5.2 学习率与动量的协同调整

**关键原则**：学习率 $\alpha$ 和动量系数 $\beta$ 不是独立的——它们共同决定有效步长。

**有效步长**（近似）：

$$\alpha_{\text{eff}} \approx \frac{\alpha}{1 - \beta}$$

因此：
- $\beta: 0.9 \to 0.99$（增大动量）时，有效步长增大约10倍，此时应**相应减小** $\alpha$ 约10倍
- 动量预热（从小到大增大 $\beta$）有助于训练初期的稳定性

**常见调参策略**：

| 阶段 | $\beta$ | $\alpha$ | 说明 |
|------|--------|---------|------|
| 预热期（前5-10个epoch）| $0.5 \to 0.9$ 渐增 | 从小到正常 | 避免初期不稳定 |
| 主训练期 | $0.9$（固定）| 余弦衰减 | 标准配置 |
| 精调期（最后若干epoch）| $0.9$（固定）| 很小的固定值 | 精细收敛 |

### 17.5.3 动量与批归一化的相互作用

批归一化（BN）对优化景观有深刻影响，与动量的相互作用值得注意：

1. **BN使损失景观更光滑**：BN压缩了梯度的尺度变化，相当于提高了全局 Lipschitz 常数的稳定性，使动量累积更可靠

2. **Ghost BN问题**：当batch size很小时，BN统计量不准确，梯度噪声大，此时 $\beta$ 应适当减小

3. **分布式训练**：多GPU同步SGD中，等效batch size增大，常规做法是**线性缩放规则**：batch size倍增时，$\alpha$ 成比例增大，$\beta$ 不变

### 17.5.4 动量在Transformer训练中的特殊性

现代大语言模型（LLM）训练主要使用Adam/AdamW，但理解动量的作用仍有价值：

- **梯度爆炸与动量**：Transformer训练中的梯度裁剪（grad clipping）会截断梯度范数，与动量结合时要注意：裁剪发生在累积动量 $v$ 之前还是之后，影响行为不同

- **预热的重要性**：在训练初期，动量向量从零开始积累，有效步长逐渐增大。不做学习率预热直接用大学习率+大动量会导致初期训练不稳定

- **梯度噪声估计**：$\|v_t\| / \|\nabla f(x_t)\|$ 可用于监控动量是否有效累积——若比值 $\gg 1$，说明动量在起作用；若 $\approx 1$，说明梯度方向频繁翻转，动量效益有限

---

## 本章小结

| 方法 | 更新规则 | 收敛率（强凸）| 收敛率（凸）| 关键超参 |
|------|---------|------------|-----------|---------|
| 梯度下降 | $x_{t+1} = x_t - \alpha\nabla f(x_t)$ | $O\!\left(\left(1-\frac{1}{\kappa}\right)^t\right)$ | $O(1/t)$ | $\alpha$ |
| 重球法（Polyak）| $v_{t+1}=\beta v_t+\nabla f(x_t)$; $x_{t+1}=x_t-\alpha v_{t+1}$ | $O\!\left(\left(1-\frac{1}{\sqrt{\kappa}}\right)^t\right)$（二次最优）| 无理论保证 | $\alpha, \beta$ |
| Nesterov加速梯度 | 在预测位置 $y_t$ 处计算梯度 | $O\!\left(\left(1-\frac{1}{\sqrt{\kappa}}\right)^t\right)$（最优！）| $O(1/t^2)$（最优！）| $\alpha, \{\lambda_t\}$ |
| SGD+Momentum | Polyak动量 + 随机梯度 | $O(1/\sqrt{t})$（无VR）| $O(1/\sqrt{t})$ | $\alpha, \beta$ |

**核心要点**：

- 动量通过积累历史梯度信息，在低曲率方向加速、在高曲率方向抑制振荡
- Nesterov方法在凸函数上是**信息论最优**的一阶方法，加速来自"向前看"的预测梯度
- 深度学习实践中，$\beta = 0.9$ 是经验黄金值；增大 $\beta$ 时需相应减小 $\alpha$
- 随机情形下，加速理论不成立，但动量仍通过方差缩减效果改善实践性能

---

## 深度学习应用：SGD with Momentum的实现与调参

### 完整实现与超参数分析

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 工具函数：梯度下降轨迹可视化
# ============================================================

def get_quadratic_landscape(a=1.0, b=10.0, grid_size=100):
    """生成病态二次函数的等值线数据
    f(x, y) = 0.5 * (a*x^2 + b*y^2)，条件数 κ = b/a
    """
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (a * X**2 + b * Y**2)
    return X, Y, Z


def optimize_quadratic(method, a=1.0, b=10.0, lr=0.1, beta=0.9,
                        x_init=(-2.5, 1.2), n_steps=50):
    """
    在病态二次函数上运行不同优化方法，返回轨迹。

    method: 'gd'（梯度下降）, 'momentum'（经典动量）, 'nesterov'（NAG）
    """
    x = torch.tensor(list(x_init), dtype=torch.float64, requires_grad=False)
    trajectory = [x.numpy().copy()]

    if method in ('momentum', 'nesterov'):
        v = torch.zeros_like(x)

    for _ in range(n_steps):
        if method == 'gd':
            # 梯度下降
            grad = torch.tensor([a * x[0].item(), b * x[1].item()], dtype=torch.float64)
            x = x - lr * grad

        elif method == 'momentum':
            # 经典动量（Polyak重球法）
            # v_{t+1} = β*v_t + ∇f(x_t)
            # x_{t+1} = x_t - α*v_{t+1}
            grad = torch.tensor([a * x[0].item(), b * x[1].item()], dtype=torch.float64)
            v = beta * v + grad
            x = x - lr * v

        elif method == 'nesterov':
            # Nesterov加速梯度
            # 在预测位置 x_t + β*v_t 处计算梯度
            # v_{t+1} = β*v_t + ∇f(x_t + β*v_t)
            # x_{t+1} = x_t - α*v_{t+1}
            x_lookahead = x + beta * v
            grad = torch.tensor([a * x_lookahead[0].item(),
                                  b * x_lookahead[1].item()], dtype=torch.float64)
            v = beta * v + grad
            x = x - lr * v

        trajectory.append(x.detach().numpy().copy())

    return np.array(trajectory)


# ============================================================
# 实验 1：三种方法在病态二次函数上的轨迹对比
# ============================================================

print("=" * 60)
print("实验 1：病态二次函数上的优化轨迹对比")
print("f(x, y) = 0.5 * (x² + 10y²),  条件数 κ = 10")
print("=" * 60)

a, b = 1.0, 10.0     # 条件数 κ = 10
lr_gd  = 0.18        # 梯度下降步长（接近 2/L = 2/10 = 0.2 的上界）
lr_mom = 0.03        # 动量方法步长（需较小避免振荡）
beta   = 0.9

traj_gd  = optimize_quadratic('gd',       a, b, lr=lr_gd,  n_steps=60)
traj_mom = optimize_quadratic('momentum', a, b, lr=lr_mom, beta=beta, n_steps=60)
traj_nes = optimize_quadratic('nesterov', a, b, lr=lr_mom, beta=beta, n_steps=60)

# 计算每步的函数值（残差）
f_val = lambda traj: 0.5 * (a * traj[:, 0]**2 + b * traj[:, 1]**2)
loss_gd  = f_val(traj_gd)
loss_mom = f_val(traj_mom)
loss_nes = f_val(traj_nes)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：优化轨迹
X, Y, Z = get_quadratic_landscape(a, b)
ax = axes[0]
ax.contour(X, Y, Z, levels=20, colors='gray', alpha=0.4, linewidths=0.8)
ax.plot(traj_gd[:, 0],  traj_gd[:, 1],  'b.-', markersize=4, label='梯度下降', linewidth=1.5)
ax.plot(traj_mom[:, 0], traj_mom[:, 1], 'r.-', markersize=4, label=f'经典动量 β={beta}', linewidth=1.5)
ax.plot(traj_nes[:, 0], traj_nes[:, 1], 'g.-', markersize=4, label=f'Nesterov β={beta}', linewidth=1.5)
ax.scatter([0], [0], color='black', s=100, zorder=5, marker='*', label='最优点')
ax.set_title('优化轨迹对比\n（病态二次函数，κ=10）', fontsize=12)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 右图：收敛曲线（对数坐标）
ax = axes[1]
steps = np.arange(len(loss_gd))
ax.semilogy(steps, loss_gd  + 1e-12, 'b-',  linewidth=2, label='梯度下降')
ax.semilogy(steps, loss_mom + 1e-12, 'r-',  linewidth=2, label=f'经典动量 β={beta}')
ax.semilogy(steps, loss_nes + 1e-12, 'g-',  linewidth=2, label=f'Nesterov β={beta}')
ax.set_title('收敛曲线（对数坐标）', fontsize=12)
ax.set_xlabel('迭代步数')
ax.set_ylabel('$f(x_t) - f^*$（对数尺度）')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('momentum_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()

# 打印收敛步数统计
threshold = 1e-4
for name, loss in [('梯度下降', loss_gd), ('经典动量', loss_mom), ('Nesterov', loss_nes)]:
    converged = np.where(loss < threshold)[0]
    if len(converged) > 0:
        print(f"{name}: 在第 {converged[0]} 步达到精度 {threshold}")
    else:
        print(f"{name}: {len(loss)} 步内未达到精度 {threshold}，最终值 = {loss[-1]:.2e}")


# ============================================================
# 实验 2：动量系数 β 的影响
# ============================================================

print("\n" + "=" * 60)
print("实验 2：不同 β 值的收敛行为")
print("=" * 60)

beta_values = [0.0, 0.5, 0.9, 0.95, 0.99]
lr_fixed = 0.03
n_steps = 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：不同 β 的收敛曲线
ax = axes[0]
for beta_val in beta_values:
    if beta_val == 0.0:
        traj = optimize_quadratic('gd', a, b, lr=lr_fixed, n_steps=n_steps)
        label = 'β=0（梯度下降）'
    else:
        traj = optimize_quadratic('momentum', a, b, lr=lr_fixed, beta=beta_val, n_steps=n_steps)
        label = f'β={beta_val}'
    loss = f_val(traj)
    ax.semilogy(np.arange(len(loss)), loss + 1e-15, linewidth=2, label=label)

ax.set_title('动量系数 β 对收敛速度的影响\n（经典动量，固定 α=0.03）', fontsize=11)
ax.set_xlabel('迭代步数')
ax.set_ylabel('$f(x_t)$（对数尺度）')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 右图：有效步长 α/(1-β) 与 β 的关系
ax = axes[1]
betas = np.linspace(0, 0.99, 200)
effective_lr = lr_fixed / (1 - betas)
ax.plot(betas, effective_lr, 'b-', linewidth=2)
ax.axhline(y=lr_fixed, color='gray', linestyle='--', label=f'α={lr_fixed}（无动量）')
ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.7, label='β=0.9（常用默认值）')
ax.scatter([0.9], [lr_fixed / 0.1], color='red', s=80, zorder=5)
ax.annotate(f'有效步长≈{lr_fixed/0.1:.2f}', xy=(0.9, lr_fixed/0.1),
            xytext=(0.7, lr_fixed/0.1 * 1.5), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='red'))
ax.set_title('有效步长 α/(1-β) 与动量系数 β 的关系', fontsize=11)
ax.set_xlabel('动量系数 β')
ax.set_ylabel('有效步长')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('momentum_beta_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 实验 3：神经网络训练——SGD vs SGD+Momentum vs Nesterov
# ============================================================

print("\n" + "=" * 60)
print("实验 3：MNIST子集上的神经网络训练对比")
print("=" * 60)

torch.manual_seed(42)
np.random.seed(42)

# 生成模拟数据（替代真实MNIST，无需下载）
n_samples = 2000
n_features = 100   # 模拟扁平化图像特征
n_classes  = 10

X_data = torch.randn(n_samples, n_features)
y_data = torch.randint(0, n_classes, (n_samples,))

train_size = int(0.8 * n_samples)
X_train, X_val = X_data[:train_size], X_data[train_size:]
y_train, y_val = y_data[:train_size], y_data[train_size:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=256)


def make_mlp(n_features, n_classes, hidden=128):
    """构建两层 MLP"""
    return nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_classes)
    )


def train_one_epoch(model, loader, optimizer, criterion, device='cpu'):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def eval_model(model, loader, criterion, device='cpu'):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)
    return total_loss / total, correct / total


criterion = nn.CrossEntropyLoss()
n_epochs = 50
lr = 0.01

# 定义参与比较的优化器配置
configs = [
    ('SGD（无动量）',      lambda p: torch.optim.SGD(p, lr=lr, momentum=0.0)),
    ('SGD + Momentum β=0.9', lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9)),
    ('SGD + Nesterov β=0.9', lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9, nesterov=True)),
    ('Adam（对比基准）',   lambda p: torch.optim.Adam(p, lr=1e-3)),
]

results = {}
for name, opt_fn in configs:
    print(f"  训练中：{name}...")
    model = make_mlp(n_features, n_classes)
    optimizer = opt_fn(model.parameters())
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(n_epochs):
        tr_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = eval_model(model, val_loader, criterion)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

    results[name] = history
    final_acc = history['val_acc'][-1]
    print(f"    最终验证准确率: {final_acc*100:.1f}%")

# 绘制训练曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
epochs = np.arange(1, n_epochs + 1)

ax = axes[0]
for (name, _), color in zip(configs, colors):
    ax.plot(epochs, results[name]['train_loss'], color=color, linewidth=2, label=name)
ax.set_title('训练损失曲线', fontsize=12)
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross-Entropy Loss')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
for (name, _), color in zip(configs, colors):
    ax.plot(epochs, [acc * 100 for acc in results[name]['val_acc']],
            color=color, linewidth=2, label=name)
ax.set_title('验证准确率曲线', fontsize=12)
ax.set_xlabel('Epoch')
ax.set_ylabel('准确率 (%)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sgd_momentum_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 实验 4：学习率预热 + 动量预热策略
# ============================================================

print("\n" + "=" * 60)
print("实验 4：动量预热（Momentum Warmup）策略")
print("=" * 60)

class MomentumWarmupScheduler:
    """
    动量预热调度器：
    - 前 warmup_epochs 个epoch，β 从 beta_start 线性增大到 beta_end
    - 之后保持 beta_end
    """
    def __init__(self, optimizer, warmup_epochs, beta_start=0.5, beta_end=0.9):
        self.optimizer    = optimizer
        self.warmup_epochs = warmup_epochs
        self.beta_start   = beta_start
        self.beta_end     = beta_end
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            beta = self.beta_start + (self.beta_end - self.beta_start) * \
                   (self.current_epoch / self.warmup_epochs)
        else:
            beta = self.beta_end
        for pg in self.optimizer.param_groups:
            pg['momentum'] = beta
        return beta


# 对比：固定 β=0.9 vs 动量预热
warmup_configs = [
    ('固定 β=0.9',     False),
    ('动量预热（0.5→0.9）', True),
]

warmup_results = {}
for name, use_warmup in warmup_configs:
    model = make_mlp(n_features, n_classes)
    # 初始 momentum 值：使用预热时从 0.5 开始，否则直接 0.9
    init_beta = 0.5 if use_warmup else 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=init_beta)
    scheduler = MomentumWarmupScheduler(optimizer, warmup_epochs=10) if use_warmup else None

    history = {'train_loss': [], 'val_acc': [], 'beta': []}
    for epoch in range(n_epochs):
        current_beta = optimizer.param_groups[0]['momentum']
        history['beta'].append(current_beta)
        tr_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion)
        _, vl_acc  = eval_model(model, val_loader, criterion)
        history['train_loss'].append(tr_loss)
        history['val_acc'].append(vl_acc)
        if scheduler:
            scheduler.step()

    warmup_results[name] = history
    print(f"  {name}: 最终验证准确率 = {history['val_acc'][-1]*100:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

ax = axes[0]
for name in warmup_results:
    ax.plot(epochs, warmup_results[name]['beta'], linewidth=2, label=name)
ax.set_title('动量系数 β 随 Epoch 的变化', fontsize=11)
ax.set_xlabel('Epoch')
ax.set_ylabel('β')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
for name in warmup_results:
    ax.plot(epochs, warmup_results[name]['train_loss'], linewidth=2, label=name)
ax.set_title('训练损失', fontsize=11)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[2]
for name in warmup_results:
    ax.plot(epochs, [a*100 for a in warmup_results[name]['val_acc']],
            linewidth=2, label=name)
ax.set_title('验证准确率', fontsize=11)
ax.set_xlabel('Epoch')
ax.set_ylabel('准确率 (%)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('momentum_warmup.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 实验 5：动量向量分析——监控动量的有效性
# ============================================================

print("\n" + "=" * 60)
print("实验 5：动量向量与梯度的比值分析")
print("说明：ratio = ||v_t|| / ||g_t|| 衡量动量累积程度")
print("=" * 60)

model = make_mlp(n_features, n_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

ratios_per_epoch = []
for epoch in range(30):
    model.train()
    epoch_ratios = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()

        # 计算动量向量范数与梯度范数的比值
        total_mom_sq, total_grad_sq = 0.0, 0.0
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                total_grad_sq += p.grad.data.norm(2).item() ** 2
                # 访问动量缓冲区（先step后才有，这里用grad近似）
        total_grad_sq = max(total_grad_sq, 1e-12)

        optimizer.step()

        # step后访问动量缓冲区
        total_mom_sq = 0.0
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'momentum_buffer' in state and state['momentum_buffer'] is not None:
                    total_mom_sq += state['momentum_buffer'].norm(2).item() ** 2

        ratio = (total_mom_sq ** 0.5) / (total_grad_sq ** 0.5)
        epoch_ratios.append(ratio)

    ratios_per_epoch.append(np.mean(epoch_ratios))

print("\n各Epoch平均 ||v||/||g|| 比值:")
for ep, ratio in enumerate(ratios_per_epoch):
    bar = '#' * int(ratio * 2)
    print(f"  Epoch {ep+1:2d}: {ratio:6.2f}  {bar}")

print(f"\n解读：初期比值小（动量从零积累），随训练逐渐稳定在约{np.mean(ratios_per_epoch[-5:]):.1f}x")
print("比值 >> 1 说明动量在放大有效步长；比值 ≈ 1 说明梯度方向随机，动量效益有限。")
```

### 超参数调参指南

```python
# ============================================================
# 超参数调参最佳实践总结
# ============================================================

print("""
SGD with Momentum 调参清单
==========================

1. 动量系数 β 的选择：
   - β = 0.9  : 深度学习默认值，适用于绝大多数场景
   - β = 0.99 : 适用于 batch size 很大（梯度噪声小）的情形
   - β = 0.5  : 适用于训练初期预热，或梯度方差极大的情形
   规则：增大 β 时，同比减小 α（有效步长 ≈ α/(1-β)）

2. 学习率 α 与 β 的联动：
   - 标准配置：α=0.1, β=0.9 → 有效步长 ≈ 1.0
   - 若将 β 从 0.9 改为 0.99：α 也应从 0.1 改为 0.01
   - 线性缩放规则（分布式）：batch size × k → α × k，β 不变

3. 学习率调度（配合动量）：
   - 余弦退火（CosineAnnealingLR）：最常用
   - 带预热的余弦退火（CosineAnnealingWarmRestarts）
   - 分段常数（MultiStepLR）：在特定epoch降低10倍
   注：调度的是 α，而非 β；β 通常固定

4. 梯度裁剪与动量的配合：
   - 先 loss.backward()，再 clip_grad_norm_()，最后 optimizer.step()
   - 裁剪发生在梯度累积到动量之前，避免大梯度污染动量缓冲区

5. PyTorch 中的 Nesterov vs 经典动量：
   optimizer = torch.optim.SGD(
       model.parameters(),
       lr=0.01,
       momentum=0.9,
       nesterov=True   # 改为 False 即为经典动量
   )
   注：使用 nesterov=True 时，momentum 必须 > 0 且 dampening == 0

6. 常见问题诊断：
   - 训练初期 loss 爆炸：减小 α 或添加学习率预热
   - 训练后期 loss 振荡：降低 α（余弦衰减末期自动处理）
   - 收敛比 Adam 慢：尝试增大 β 或使用预热，SGD 通常需要更多调参
   - 验证集性能比 Adam 差：延长训练或改善数据增强，SGD 通常泛化更好但需更长时间
""")
```

---

## 练习题

**练习 17.1**（动量的等价形式）

经典动量的迭代格式为：

$$v_{t+1} = \beta v_t + \nabla f(x_t), \quad x_{t+1} = x_t - \alpha v_{t+1}$$

(a) 将上述格式展开，证明它等价于：

$$x_{t+1} = x_t - \alpha \nabla f(x_t) + \beta(x_t - x_{t-1})$$

即新的更新 = 梯度下降步 + 惯性项（上一步位移的 $\beta$ 倍）。

(b) 进一步证明：$v_t = \sum_{k=0}^{t-1} \beta^{t-1-k} \nabla f(x_k)$，即动量是历史梯度的指数加权平均。

(c) 当目标函数为 $f(x) = \frac{c}{2}x^2$（一维二次函数）时，写出 $x_t$ 关于 $x_0$ 的闭合表达式（特征方程法），并分析稳定性条件（$|x_t| \to 0$）对 $(\alpha, \beta)$ 的约束。

---

**练习 17.2**（Nesterov方法的预测步）

Nesterov方法（速度形式）为：

$$\tilde{x}_t = x_t + \beta v_t \quad \text{（预测位置）}$$

$$v_{t+1} = \beta v_t + \nabla f(\tilde{x}_t) \quad \text{（在预测位置计算梯度）}$$

$$x_{t+1} = x_t - \alpha v_{t+1}$$

(a) 与经典动量相比，Nesterov方法的唯一区别是在哪里计算梯度？用一句话说明这带来的直觉优势。

(b) 在一维二次函数 $f(x) = \frac{c}{2}x^2$ 上，分别写出经典动量和Nesterov方法的一步更新格式，并比较两者在 $x_0 = 1, v_0 = 0$ 时第一步的更新差异（取 $\alpha = 0.1, \beta = 0.9, c = 1$）。

(c) PyTorch中 `torch.optim.SGD(..., momentum=0.9, nesterov=True)` 和 `nesterov=False` 的实现有何区别？阅读PyTorch文档并用伪代码写出两者的更新步骤。

---

**练习 17.3**（收敛率的量级分析）

设 $f$ 是 $m$-强凸且 $L$-光滑的函数，条件数 $\kappa = L/m$。

(a) 梯度下降（最优步长）达到精度 $\|x_t - x^*\| \leq \epsilon$ 所需步数约为 $O(\kappa \log(1/\epsilon))$。证明：取步长 $\alpha = 2/(L+m)$，梯度下降的收敛率为 $\left(\frac{\kappa-1}{\kappa+1}\right)^t$。

(b) Nesterov方法的收敛率为 $\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^t$。当 $\kappa = 100$ 时，分别计算梯度下降和Nesterov方法达到 $\epsilon = 10^{-6}$ 所需的步数（保留整数）。

(c) 填写下表（$\epsilon = 10^{-6}$）：

| 条件数 $\kappa$ | 梯度下降步数 | Nesterov步数 | 加速比 |
|------------|-----------|------------|------|
| 10 | | | |
| 100 | | | |
| $10^4$ | | | |
| $10^6$ | | | |

---

**练习 17.4**（重球法的最优参数）

对一维二次函数 $f(x) = \frac{c}{2}x^2$，重球法的迭代为：

$$x_{t+1} = x_t - \alpha c x_t + \beta(x_t - x_{t-1}) = (1 - \alpha c + \beta)x_t - \beta x_{t-1}$$

记 $r_1, r_2$ 为特征方程 $r^2 - (1 - \alpha c + \beta)r + \beta = 0$ 的两根。

(a) 证明收敛条件为 $|r_1| < 1$ 且 $|r_2| < 1$，等价于 $0 < \alpha c < 2(1+\beta)$ 且 $|\beta| < 1$。

(b) 当 $c$ 的取值范围为 $[m, L]$ 时（多维情形的类比），最优参数为：

$$\alpha^* = \frac{4}{(\sqrt{L}+\sqrt{m})^2}, \quad \beta^* = \left(\frac{\sqrt{L}-\sqrt{m}}{\sqrt{L}+\sqrt{m}}\right)^2$$

请验证：此时特征方程的两根的绝对值均等于 $\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$。

(c) 对比：梯度下降的最优收敛率为 $\frac{\kappa-1}{\kappa+1}$，重球法为 $\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$。当 $\kappa = 100$ 时，分别计算需要多少步使误差缩小到 $10^{-4}$（即 $\rho^t \leq 10^{-4}$）。

---

**练习 17.5**（深度学习实践）

本题要求用PyTorch实现并分析SGD+Momentum的行为。

(a) **有效步长验证**：实现以下实验——在一个固定的二次损失上，对比以下三个配置的实际收敛速度是否相同：
- 配置A：$\alpha = 0.1$，$\beta = 0$（无动量）
- 配置B：$\alpha = 0.01$，$\beta = 0.9$（有效步长约0.1）
- 配置C：$\alpha = 0.001$，$\beta = 0.99$（有效步长约0.1）

理论上三者有效步长相同，实际表现如何？分析差异的原因。

(b) **梯度裁剪与动量**：实现以下两种梯度裁剪策略，并比较训练10个epoch后的损失：
- 策略1：在 `optimizer.step()` 之前裁剪梯度（标准做法）
- 策略2：不做梯度裁剪

在梯度偶尔爆炸的场景中（人为加入大梯度噪声），哪种策略更稳定？

(c) **Nesterov vs 经典动量**：在CIFAR-10（或等效模拟数据）上，对比 `nesterov=True` 和 `nesterov=False` 的训练曲线，分析：训练初期（前10个epoch）、中期（第10-50个epoch）、后期（最后10个epoch）各自的差异模式是什么？

---

## 练习答案

### 练习 17.1 解答

**(a) 等价形式的证明**：

由递推关系 $v_t = \beta v_{t-1} + \nabla f(x_{t-1})$，代入 $x_{t+1} = x_t - \alpha v_{t+1}$：

$$x_{t+1} = x_t - \alpha(\beta v_t + \nabla f(x_t))$$

$$= x_t - \alpha \nabla f(x_t) - \alpha \beta v_t$$

由于 $x_t = x_{t-1} - \alpha v_t$，即 $\alpha v_t = x_{t-1} - x_t$，代入：

$$x_{t+1} = x_t - \alpha \nabla f(x_t) - \beta(x_{t-1} - x_t) = x_t - \alpha \nabla f(x_t) + \beta(x_t - x_{t-1})$$

$\square$

**(b) 指数加权平均展开**：

由 $v_{t+1} = \beta v_t + \nabla f(x_t)$，递推展开：

$$v_t = \beta^t v_0 + \sum_{k=0}^{t-1} \beta^{t-1-k} \nabla f(x_k)$$

取 $v_0 = \mathbf{0}$，即得 $v_t = \sum_{k=0}^{t-1} \beta^{t-1-k} \nabla f(x_k)$。

权重 $\beta^{t-1-k}$：对最近的梯度（$k=t-1$，权重 $\beta^0 = 1$）权重最大；对最远的梯度（$k=0$，权重 $\beta^{t-1}$）权重最小，呈指数衰减。$\square$

**(c) 一维二次函数的特征方程**：

$f(x) = \frac{c}{2}x^2$ 时，$\nabla f(x) = cx$，更新为 $x_{t+1} = (1 - \alpha c + \beta)x_t - \beta x_{t-1}$。

引入向量 $\mathbf{u}_t = (x_t, x_{t-1})^T$，系数矩阵 $A = \begin{pmatrix} 1-\alpha c + \beta & -\beta \\ 1 & 0 \end{pmatrix}$。

特征方程：$\lambda^2 - (1-\alpha c + \beta)\lambda + \beta = 0$。

稳定性条件（$x_t \to 0$）：两个特征值 $|\lambda_i| < 1$，由 Jury 稳定性准则等价于：

1. $\beta < 1$（$|\det A| = |\beta| < 1$）
2. $|1 - \alpha c + \beta| < 1 + \beta$，即 $0 < \alpha c < 2(1+\beta)$

---

### 练习 17.2 解答

**(a) 核心区别**：经典动量在**当前位置** $x_t$ 计算梯度，Nesterov方法在**预测的未来位置** $\tilde{x}_t = x_t + \beta v_t$ 计算梯度。直觉优势：如果动量方向正确，在"前方"计算梯度能更早感知到目标附近的曲率，提前修正方向，避免过冲。

**(b) 数值对比**（$x_0 = 1, v_0 = 0, \alpha = 0.1, \beta = 0.9, c = 1$）：

**第一步**（$t=0$）：

*经典动量*：
- $v_1 = 0.9 \times 0 + 1 \times 1 = 1$（梯度在 $x_0 = 1$ 处）
- $x_1 = 1 - 0.1 \times 1 = 0.9$

*Nesterov*：
- $\tilde{x}_0 = 1 + 0.9 \times 0 = 1$（预测位置与当前位置相同，因 $v_0 = 0$）
- $v_1 = 0.9 \times 0 + 1 = 1$
- $x_1 = 1 - 0.1 \times 1 = 0.9$

第一步两者相同（因 $v_0 = 0$），差异从第二步开始体现。

**第二步**（$t=1$）：

*经典动量*：梯度在 $x_1 = 0.9$ 处，$g_1 = 0.9$。
- $v_2 = 0.9 \times 1 + 0.9 = 1.8$
- $x_2 = 0.9 - 0.1 \times 1.8 = 0.72$

*Nesterov*：预测位置 $\tilde{x}_1 = 0.9 + 0.9 \times 1 = 1.8$，梯度 $= 1.8$（反向！）
- $v_2 = 0.9 \times 1 + 1.8 = 2.7$（看似更大）
- $x_2 = 0.9 - 0.1 \times 2.7 = 0.63$

Nesterov在此例中步子更激进（因预测位置超过了最优点 $x^*=0$），但在接近最优解时会自动修正。

**(c) PyTorch 实现**（伪代码）：

```
# 经典动量 (nesterov=False)：
buf = momentum * buf + grad
param -= lr * buf

# Nesterov 动量 (nesterov=True)：
buf = momentum * buf + grad
param -= lr * (grad + momentum * buf)
# 等价于：先做 buf 更新，再在"前方"补一个额外的 momentum * buf 步
```

---

### 练习 17.3 解答

**(a) 梯度下降收敛率的推导**：

设 $f$ 是 $m$-强凸且 $L$-光滑，取步长 $\alpha = 2/(L+m)$，则：

$$x_{t+1} - x^* = (I - \alpha \nabla^2 f)(x_t - x^*) + O(\|x_t - x^*\|^2)$$

对二次函数精确成立，特征值缩放因子为 $|1 - \alpha \lambda_i| \leq \max(|1-\alpha m|, |1-\alpha L|)$。

代入 $\alpha = 2/(L+m)$：

$$|1 - \alpha m| = 1 - \frac{2m}{L+m} = \frac{L-m}{L+m} = \frac{\kappa-1}{\kappa+1}$$

$$|1 - \alpha L| = \frac{L-m}{L+m} = \frac{\kappa-1}{\kappa+1}$$

两端相等，均为 $\frac{\kappa-1}{\kappa+1}$，这正是"最优步长使两端的缩放因子相等"的Chebyshev原理。$\square$

**(b) $\kappa = 100$ 时的步数计算**：

梯度下降：$\rho_{GD} = \frac{99}{101} \approx 0.9802$，需 $t$ 步使 $0.9802^t \leq \epsilon$：

$$t \geq \frac{\ln(1/\epsilon)}{\ln(1/\rho_{GD})} = \frac{\ln(10^6)}{\ln(101/99)} \approx \frac{13.816}{0.0202} \approx 684 \text{ 步}$$

Nesterov：$\rho_{Nes} = \frac{\sqrt{100}-1}{\sqrt{100}+1} = \frac{9}{11} \approx 0.8182$：

$$t \geq \frac{13.816}{\ln(11/9)} \approx \frac{13.816}{0.2007} \approx 69 \text{ 步}$$

加速约 $684/69 \approx 10 \approx \sqrt{\kappa}$ 倍。

**(c) 完整表格**（$\epsilon = 10^{-6}$，$\ln(10^6) \approx 13.816$）：

| 条件数 $\kappa$ | 梯度下降步数 | Nesterov步数 | 加速比 |
|------------|-----------|------------|------|
| 10 | $\approx 130$ | $\approx 28$ | $\approx 4.6\approx\sqrt{10}$ |
| 100 | $\approx 684$ | $\approx 69$ | $\approx 9.9\approx\sqrt{100}$ |
| $10^4$ | $\approx 63800$ | $\approx 638$ | $\approx 100=\sqrt{10^4}$ |
| $10^6$ | $\approx 6.4\times10^6$ | $\approx 6380$ | $\approx 1000=\sqrt{10^6}$ |

加速比精确地等于 $\sqrt{\kappa}$，验证了理论。

---

### 练习 17.4 解答

**(a) 稳定性分析**：

两根满足 Vieta 定理：$r_1 + r_2 = 1 - \alpha c + \beta$，$r_1 r_2 = \beta$。

由 $|r_1|, |r_2| < 1$ 的充要条件（实数系数二次方程稳定性）：
1. $|r_1 r_2| = \beta < 1$，即 $|\beta| < 1$（同时保证 $\beta > -1$，通常取 $\beta \in (0,1)$）
2. $|r_1 + r_2| < 1 + r_1 r_2$，即 $|1 - \alpha c + \beta| < 1 + \beta$

展开绝对值：$-(1+\beta) < 1 - \alpha c + \beta < 1 + \beta$，

左边：$\alpha c < 2 + 2\beta = 2(1+\beta)$；右边：$\alpha c > 0$。

合并即为 $0 < \alpha c < 2(1+\beta)$。$\square$

**(b) 最优参数的特征根验证**：

代入 $\alpha^* c = \frac{4c}{(\sqrt{L}+\sqrt{m})^2}$（以 $c = L$ 为例，极端情形）：

特征方程 $r^2 - (1 - \alpha^* L + \beta^*)r + \beta^* = 0$，其中：

$$1 - \alpha^* L + \beta^* = 1 - \frac{4L}{(\sqrt{L}+\sqrt{m})^2} + \left(\frac{\sqrt{L}-\sqrt{m}}{\sqrt{L}+\sqrt{m}}\right)^2$$

记 $s = \sqrt{L/m} = \sqrt{\kappa}$，$\alpha^* m = \frac{4}{(1+\sqrt{\kappa}/s \cdot s)^2}$，化简后判别式 $\Delta = 0$（重根），双重根为：

$$r = \frac{1-\alpha^*c+\beta^*}{2} = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$$

故 $|r_1| = |r_2| = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$。$\square$

**(c) 步数对比**（$\kappa = 100$，误差缩小到 $10^{-4}$）：

梯度下降：$\rho_{GD} = \frac{99}{101}$，需 $t$ 步使 $\rho_{GD}^t \leq 10^{-4}$：

$$t \geq \frac{4\ln 10}{\ln(101/99)} \approx \frac{9.21}{0.0202} \approx 456 \text{ 步}$$

重球法：$\rho_{HB} = \frac{9}{11}$：

$$t \geq \frac{4\ln 10}{\ln(11/9)} \approx \frac{9.21}{0.2007} \approx 46 \text{ 步}$$

重球法约快 10 倍（$\approx\sqrt{\kappa}$ 倍）。

---

### 练习 17.5 解答

**(a) 有效步长验证**：

理论上三种配置有效步长相同（$\approx 0.1$），但实际表现有差异：
- 配置A（无动量）：纯梯度下降，收敛稳定但无加速
- 配置B（$\beta=0.9$）：动量累积需要约 $1/(1-0.9)=10$ 步"预热"，初期较慢，之后加速
- 配置C（$\beta=0.99$）：预热期更长（约100步），初期极慢；但一旦动量积累完成，有效步长更稳定

**差异原因**：有效步长公式 $\alpha/(1-\beta)$ 是稳态近似，在初期（动量从零积累）不成立。$\beta$ 越大，需要越多步才能接近稳态有效步长。

**(b) 梯度裁剪**：

策略1（裁剪后再step）更稳定。原因：
- 在梯度爆炸时，策略1将大梯度裁剪后再累积到动量中，动量缓冲区不受污染
- 策略2若一次大梯度进入了动量缓冲区，接下来多步都会受其影响（$v_t$ 带有大量残留），需要 $1/(1-\beta)$ 步才能"忘记"这次爆炸

**实践建议**：总是在 `optimizer.step()` 之前调用 `torch.nn.utils.clip_grad_norm_()`。

**(c) Nesterov vs 经典动量的训练差异**：

- **训练初期（前10 epoch）**：两者差异较小（动量向量从零开始积累，预测位置接近当前位置），有时Nesterov略快
- **训练中期（第10-50 epoch）**：Nesterov收敛更平滑，振荡更少——因为预测步能提前"感知"鞍点和弯道，避免过冲
- **训练后期（最后10 epoch）**：两者最终精度相近，但Nesterov通常更快到达低损失区域；若使用学习率衰减，两者差异进一步缩小（小学习率时动量效果减弱）

---

*下一章：第18章 自适应学习率方法——从AdaGrad到Adam，我们将看到如何为每个参数自动调整步长，彻底解决动量方法对学习率敏感性的问题。*
