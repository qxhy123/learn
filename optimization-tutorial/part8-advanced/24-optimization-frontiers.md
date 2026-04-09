# 第24章 优化理论前沿

## 学习目标

完成本章学习后，你将能够：

1. 理解非凸优化中鞍点逃逸的理论机制，掌握扰动梯度下降的收敛保证
2. 解释隐式正则化现象和稳定边缘（Edge of Stability）的数学原理
3. 推导神经切线核（NTK）理论框架，理解无限宽网络的线性化行为
4. 分析损失曲面的几何结构，理解模式连通性与彩票假说的含义
5. 掌握PAC-Bayes界和平坦性-泛化关系的理论联系

---

## 24.1 非凸优化理论进展

### 24.1.1 非凸优化的挑战

深度学习的损失函数是高度非凸的。经典优化理论无法直接应用，因为：

- **局部极小值**：可能有指数多个，质量参差不齐
- **鞍点**：梯度为零但非极小值，梯度下降可能在此停滞
- **平坦区域**：梯度极小，收敛极慢

令 $f: \mathbb{R}^d \to \mathbb{R}$ 为损失函数。一个点 $x^*$ 称为：
- **局部极小值**：$\nabla f(x^*) = 0$，$\nabla^2 f(x^*) \succeq 0$
- **严格鞍点**：$\nabla f(x^*) = 0$，$\nabla^2 f(x^*)$ 有至少一个负特征值
- **退化鞍点**：$\nabla f(x^*) = 0$，$\nabla^2 f(x^*)$ 半正定但非正定

**关键洞见**：在高维空间中，局部极小值远比鞍点稀少。随机矩阵理论表明，当维度 $d \to \infty$ 时，一个随机临界点是局部极小值的概率趋近于零。

### 24.1.2 严格鞍点性质

**定义（严格鞍点性质）**：函数 $f$ 满足严格鞍点性质，若所有鞍点都是严格的，即每个临界点要么是局部极小值，要么是严格鞍点（Hessian 有负特征值）。

满足此性质的函数包括：
- 矩阵分解：$f(U, V) = \|M - UV^\top\|_F^2$
- 相位恢复
- 字典学习
- 单隐层神经网络（在某些条件下）

**定理（Du et al., 2017；Lee et al., 2016）**：设 $f$ 满足严格鞍点性质且二阶连续可微。从几乎所有初始点出发，梯度下降（步长足够小）以概率1收敛到局部极小值，而非鞍点。

**证明思路**：鞍点的稳定流形（stable manifold）是零测集。若初始点不在稳定流形上，梯度下降的迭代序列不会收敛到鞍点。

### 24.1.3 扰动梯度下降（Perturbed GD）

然而，梯度下降逃离鞍点的速度可能极慢（指数时间）。扰动梯度下降（PGD）通过主动添加噪声来加速逃逸。

**算法（NEON/PGD，Jin et al., 2017）**：

$$x_{t+1} = x_t - \eta \nabla f(x_t) + \xi_t$$

其中 $\xi_t \sim \text{Uniform}(B(0, r))$ 在满足某条件时添加扰动。

**定理（Jin et al., 2017）**：设 $f$ 是 $\ell$-光滑的，且所有鞍点的 Hessian 最小特征值 $\lambda_{\min}(\nabla^2 f) \leq -\gamma$（$\gamma$-非退化鞍点）。PGD 在

$$O\!\left(\frac{\ell (f(x_0) - f^*)}{\epsilon^2} \log^4\frac{d \ell (f(x_0)-f^*)}{\epsilon^2 \delta}\right)$$

步内以概率 $1-\delta$ 找到一个 $\epsilon$-近似二阶驻点 $x$，满足：

$$\|\nabla f(x)\| \leq \epsilon, \quad \lambda_{\min}(\nabla^2 f(x)) \geq -\sqrt{\epsilon \ell}$$

与梯度下降的 $O(1/\epsilon^2)$ 相比，对数因子的代价可以接受。

### 24.1.4 随机梯度下降的隐式逃鞍机制

有趣的是，**SGD 的噪声本身就能帮助逃离鞍点**，无需显式扰动。

**直觉**：在鞍点附近，损失曲面在某些方向上是下降的。SGD 的随机梯度噪声在这些方向上有分量，从而自然地"滑离"鞍点。

**Langevin 动力学视角**：SGD 近似于 Langevin 扩散：

$$dx_t = -\nabla f(x_t) \, dt + \sqrt{2\beta^{-1}} \, dW_t$$

其中 $\beta$ 是逆温度（学习率的倒数），$W_t$ 是标准布朗运动。Langevin 动力学的平稳分布是 Gibbs 分布 $p^*(x) \propto e^{-\beta f(x)}$，在低温（大学习率）时集中于全局极小值附近。

---

## 24.2 隐式正则化

### 24.2.1 过参数化与隐式偏差

现代神经网络参数数量远超训练样本数量（严重过参数化），但仍能泛化。经典统计学习理论无法解释这一现象。

**关键观察**：梯度下降在过参数化模型中会隐式地偏向某类解，即使没有显式正则化项。

**线性模型的例子**：考虑线性回归 $y = Xw$，样本数 $n <$ 参数数 $d$。存在无穷多个零训练误差的解 $\{w : Xw = y\}$。

**定理（梯度下降的隐式正则化）**：从零初始化出发，梯度下降收敛到最小 $\ell_2$ 范数解：

$$w^* = \arg\min_{w} \|w\|_2 \quad \text{s.t.} \quad Xw = y = X X^\dagger y$$

即 $w^* = X^\top (XX^\top)^{-1} y$（Moore-Penrose 伪逆）。

**证明**：设 $w_t$ 为第 $t$ 步的参数。梯度更新保持 $w_t$ 在 $X$ 的行空间内（$w_t \in \text{row}(X)$），因为初始化为零且梯度 $\nabla \mathcal{L} = X^\top(Xw_t - y)$ 始终在行空间内。因此收敛点是行空间内满足 $Xw = y$ 的唯一解，即最小范数解。

### 24.2.2 矩阵分解中的隐式正则化

**矩阵分解**：$\min_{U \in \mathbb{R}^{m \times r}, V \in \mathbb{R}^{n \times r}} \frac{1}{2}\|P_\Omega(M - UV^\top)\|_F^2$

其中 $P_\Omega$ 是观测位置的投影算子。

**定理（Gunasekar et al., 2017）**：在平衡初始化（$U_0 U_0^\top = V_0 V_0^\top$）和梯度流（连续时间梯度下降）下，收敛到的解最小化核范数：

$$\min_{Z} \|Z\|_* \quad \text{s.t.} \quad P_\Omega(Z) = P_\Omega(M)$$

**直觉**：梯度流偏向低秩解，核范数是矩阵秩的凸松弛，因此隐式地执行了核范数正则化。

### 24.2.3 稳定边缘（Edge of Stability）

**Cohen et al. (2021)** 发现了一个令人惊讶的现象：在全批次梯度下降训练神经网络时：

1. **渐进阶段**：损失单调下降，Hessian 最大特征值 $\lambda_{\max}(\nabla^2 \mathcal{L})$ 稳定增大
2. **稳定边缘**：$\lambda_{\max}$ 稳定在 $2/\eta$（$\eta$ 为学习率）附近
3. **EOS 后**：损失非单调但整体下降，$\lambda_{\max}$ 在 $2/\eta$ 附近震荡

**理论背景**：对于二次函数 $f(x) = \frac{1}{2}x^\top A x$，梯度下降稳定的条件是所有特征值 $\lambda_i$ 满足 $\eta \lambda_i < 2$，即 $\lambda_{\max} < 2/\eta$。

当 $\lambda_{\max} = 2/\eta$ 时，梯度下降恰好在稳定边缘。

**EOS 的含义**：神经网络的优化轨迹会自适应地将曲率控制在稳定边缘，这是一种隐式的自我调节机制。

**EOS 与隐式正则化的联系**：在稳定边缘，优化器隐式地偏向曲率（Hessian 迹）小的解，这与泛化性有关（见24.5节）。

**数学描述**：设 $\phi: \mathbb{R} \to \mathbb{R}$ 为非线性激活函数，训练损失为 $\mathcal{L}(\theta)$。EOS 现象表明：

$$\lim_{t \to \infty} \lambda_{\max}(\nabla^2_\theta \mathcal{L}(\theta_t)) \approx \frac{2}{\eta}$$

这一收敛不依赖于初始化，是损失曲面几何与学习率之间的深层联系。

---

## 24.3 神经切线核

### 24.3.1 无限宽网络的线性化

**神经切线核（NTK）**由 Jacot, Gabriel & Hongler（2018）提出，揭示了无限宽神经网络在训练过程中的线性化行为。

**设置**：考虑参数化为 $\theta \in \mathbb{R}^P$ 的神经网络 $f_\theta: \mathbb{R}^d \to \mathbb{R}$。在梯度流下：

$$\dot{\theta}_t = -\nabla_\theta \mathcal{L}(\theta_t) = -\frac{1}{n} \sum_{i=1}^n (f_{\theta_t}(x_i) - y_i) \nabla_\theta f_{\theta_t}(x_i)$$

网络输出的动态为：

$$\dot{f}_{\theta_t}(x) = \nabla_\theta f_{\theta_t}(x)^\top \dot{\theta}_t = -\frac{1}{n}\sum_{i=1}^n K_t(x, x_i)(f_{\theta_t}(x_i) - y_i)$$

其中**神经切线核**定义为：

$$K_t(x, x') = \nabla_\theta f_{\theta_t}(x)^\top \nabla_\theta f_{\theta_t}(x')$$

### 24.3.2 NTK 的不动性定理

**定理（Jacot et al., 2018）**：对于适当参数化的无限宽神经网络（$n_1, \ldots, n_L \to \infty$），NTK 在训练过程中保持不变：

$$K_t(x, x') \xrightarrow{P \to \infty} K_\infty(x, x'), \quad \forall t \geq 0$$

其中 $K_\infty$ 是确定性的核，由网络架构和激活函数决定，与初始化和训练过程无关。

**推论**：无限宽网络等价于核方法，其训练动态线性化为：

$$\dot{f}_t = -K_\infty (f_t - y)$$

其中 $f_t = (f_t(x_1), \ldots, f_t(x_n))^\top$，$K_\infty$ 是 $n \times n$ 的核矩阵。

### 24.3.3 NTK 的递推公式

**全连接网络的 NTK**：对于 $L$ 层全连接网络，NTK 满足递推关系：

设 $h^{(0)}(x) = x$，第 $l$ 层的核为：

$$\Sigma^{(1)}(x, x') = x^\top x' / d_0$$

$$\Lambda^{(l)}(x, x') = \begin{pmatrix} \Sigma^{(l)}(x,x) & \Sigma^{(l)}(x,x') \\ \Sigma^{(l)}(x',x) & \Sigma^{(l)}(x',x') \end{pmatrix}$$

$$\Sigma^{(l+1)}(x, x') = \mathbb{E}_{(u,v) \sim \mathcal{N}(0, \Lambda^{(l)})}[\sigma(u)\sigma(v)]$$

$$\dot{\Sigma}^{(l+1)}(x, x') = \mathbb{E}_{(u,v) \sim \mathcal{N}(0, \Lambda^{(l)})}[\sigma'(u)\sigma'(v)]$$

NTK 递推为：

$$\Theta^{(1)}(x, x') = \Sigma^{(1)}(x, x')$$

$$\Theta^{(L+1)}(x, x') = \Theta^{(L)}(x, x') \cdot \dot{\Sigma}^{(L+1)}(x, x') + \Sigma^{(L+1)}(x, x')$$

### 24.3.4 NTK 的训练动态

在 NTK 框架下，训练动态精确可解。设均方误差损失：

$$\mathcal{L}(\theta) = \frac{1}{2n}\|f_\theta(X) - y\|^2$$

则函数空间的梯度流解为：

$$f_t(X) - y = e^{-K_\infty t/n}(f_0(X) - y)$$

**测试误差**：对测试点 $x^*$，预测为：

$$f_t(x^*) = f_0(x^*) + K_\infty(x^*, X) K_\infty(X, X)^{-1}(y - f_0(X))(I - e^{-K_\infty(X,X)t/n})$$

当 $t \to \infty$（完全训练）：

$$f_\infty(x^*) = f_0(x^*) + K_\infty(x^*, X) K_\infty(X, X)^{-1}(y - f_0(X))$$

这正是以 $K_\infty$ 为核的**核岭回归**（正则化参数为零）的预测！

### 24.3.5 NTK 理论的局限性

尽管 NTK 理论优美，但存在重要局限：

1. **无限宽极限与实践脱节**：实际网络宽度有限，特征会随训练变化（特征学习）
2. **NTK 对应的核往往劣于有限宽网络**：NTK 预测的泛化误差通常差于实际网络
3. **无法解释迁移学习**：NTK 框架下无法发生特征学习
4. **平均场理论**（Yang & Hu, 2021）提供了超越 NTK 的框架，允许描述有限宽度下的特征学习

---

## 24.4 损失曲面的几何结构

### 24.4.1 局部极小值的等价性

经典理论担心局部极小值的质量差异，但实验表明深度网络的局部极小值质量相近。

**定理（Goodfellow et al., 2015；实验性）**：对于过参数化的深度网络，沿梯度下降路径上的线性插值，损失单调下降——这暗示局部极小值附近的损失曲面相对平坦。

**定理（过参数化线性网络，Kawaguchi 2016）**：对于深度线性网络，所有局部极小值都是全局极小值，且所有鞍点都是严格鞍点。

### 24.4.2 模式连通性（Mode Connectivity）

**Garipov et al. (2018)** 和 **Draxler et al. (2018)** 独立发现：

**现象**：两个独立训练得到的局部极小值（"模式"）可以用一条低损失路径连接，而非被高损失壁垒分隔。

**数学表述**：设 $\theta_1, \theta_2$ 为两个局部极小值。存在路径 $\phi: [0,1] \to \mathbb{R}^P$，$\phi(0) = \theta_1$，$\phi(1) = \theta_2$，使得：

$$\max_{t \in [0,1]} \mathcal{L}(\phi(t)) \approx \mathcal{L}(\theta_1) \approx \mathcal{L}(\theta_2)$$

**寻找连接路径的方法**：

1. **线性插值**：$\phi(t) = (1-t)\theta_1 + t\theta_2$（通常经过高损失区域）
2. **贝塞尔曲线**：$\phi(t) = (1-t)^2\theta_1 + 2t(1-t)\theta_m + t^2\theta_2$，优化中间点 $\theta_m$
3. **折线路径**：$\phi(t)$ 为经过中间节点的分段线性路径，优化节点位置

**损失面板**（loss barrier）：线性插值路径的最大损失与端点损失之差：

$$\Delta(\theta_1, \theta_2) = \max_{t \in [0,1]} \mathcal{L}((1-t)\theta_1 + t\theta_2) - \frac{\mathcal{L}(\theta_1) + \mathcal{L}(\theta_2)}{2}$$

**模型平均**：模式连通性的实用意义是，沿连接路径的中间模型通常比端点模型泛化更好（**SWA**：随机权重平均）。

### 24.4.3 彩票假说（Lottery Ticket Hypothesis）

**Frankle & Carlin (2019)** 提出的彩票假说：

**假说**：一个大型随机初始化的神经网络包含一个小的子网络（"中奖彩票"），如果从原始初始化权重出发单独训练这个子网络，可以达到与完整网络相当的精度。

**形式化**：设 $f(\theta; m)$ 为应用掩码 $m \in \{0,1\}^{|\theta|}$ 后的网络。存在掩码 $m^*$ 和初始化 $\theta_0$，使得：

$$\mathcal{L}(f(\theta^*(m^*); m^*)) \approx \mathcal{L}(f(\theta^*; \mathbf{1}))$$

其中 $|m^*| \ll |\theta|$（中奖彩票远小于完整网络），$\theta^*(m^*)$ 是从 $\theta_0 \odot m^*$ 出发训练的权重。

**发现中奖彩票的算法（迭代幅度剪枝）**：

1. 随机初始化 $\theta_0$
2. 训练 $j$ 步得到 $\theta_j$
3. 剪掉幅度最小的 $p\%$ 权重，得到掩码 $m$
4. 将未被剪掉的权重重置为 $\theta_0 \odot m$
5. 重复步骤 2-4

**彩票假说的深层含义**：

- 网络初始化的质量远比想象中重要
- 稀疏性是深度学习的内在属性
- 解释了为何网络剪枝有效

**线性模式连通性**：**Frankle et al. (2020)** 发现，训练几步后中奖彩票（但非完整网络）满足线性模式连通性——这为彩票假说提供了几何解释。

---

## 24.5 优化与泛化的统一理论

### 24.5.1 PAC-Bayes 界

**PAC-Bayes 框架**（McAllester, 1999）将泛化误差与参数空间上的概率测度联系起来。

**定理（PAC-Bayes 界）**：设 $P$ 为参数先验（训练前确定），$Q$ 为训练后的后验。对任意 $\delta > 0$，以概率 $1-\delta$（对训练集采样）：

$$\mathbb{E}_{\theta \sim Q}[\mathcal{L}_{test}(\theta)] \leq \mathbb{E}_{\theta \sim Q}[\mathcal{L}_{train}(\theta)] + \sqrt{\frac{KL(Q \| P) + \ln(2\sqrt{n}/\delta)}{2n}}$$

**PAC-Bayes 对深度学习的应用**（Dziugaite & Roy, 2017）：取 $Q = \mathcal{N}(\theta^*, \sigma^2 I)$（以训练解为中心的高斯），$P = \mathcal{N}(0, \sigma_0^2 I)$（标准高斯先验）：

$$KL(Q \| P) = \frac{d(\sigma^2 + \|\theta^*\|^2/d)}{\sigma_0^2} - d + d\ln\frac{\sigma_0^2}{\sigma^2}$$

通过优化 $\sigma$，可以得到非平凡的泛化界，揭示：**参数范数小且损失曲面平坦的解泛化更好**。

### 24.5.2 平坦性与泛化

**Hochreiter & Schmidhuber (1997)** 最早提出平坦极小值泛化更好的直觉：若 $\theta^*$ 周围半径 $r$ 内的参数都有低训练损失，则参数量化误差（相当于参数扰动 $\leq r$）不影响训练误差，泛化更好。

**Hessian 迹作为平坦性度量**：

$$\text{Flat}(\theta) = \text{tr}(\nabla^2 \mathcal{L}(\theta)) = \sum_{i=1}^d \lambda_i(\nabla^2 \mathcal{L}(\theta))$$

**SAM（Sharpness-Aware Minimization，Foret et al., 2021）**：直接最小化最坏情况下的损失：

$$\min_\theta \max_{\|\epsilon\| \leq \rho} \mathcal{L}(\theta + \epsilon)$$

**SAM 的更新规则**：

1. 计算最坏扰动：$\hat{\epsilon}(\theta) = \rho \frac{\nabla_\theta \mathcal{L}(\theta)}{\|\nabla_\theta \mathcal{L}(\theta)\|}$
2. 计算扰动点的梯度：$g = \nabla_\theta \mathcal{L}(\theta + \hat{\epsilon}(\theta))$
3. 更新：$\theta \leftarrow \theta - \eta g$

**定理（SAM 的 PAC-Bayes 解释）**：SAM 近似地最小化了以 $\theta$ 为中心、方差为 $\rho^2/d$ 的高斯分布的平均训练损失，从而隐式地最小化了 PAC-Bayes 界中的 KL 散度项。

### 24.5.3 双下降现象

**Belkin et al. (2019)** 发现了**双下降**（Double Descent）现象，挑战了经典偏差-方差权衡理论：

**经典理论**：测试误差随模型复杂度先降后升（U形曲线）。

**双下降**：在过参数化区间，测试误差再次下降，形成双峰形状：

$$\text{Test Error} = \begin{cases} \text{偏差主导（欠拟合）} & \text{参数数} \ll n \\ \text{插值门槛处峰值} & \text{参数数} \approx n \\ \text{再次下降} & \text{参数数} \gg n \end{cases}$$

**数学解释（线性模型）**：设 $X \in \mathbb{R}^{n \times d}$，$y = X\beta^* + \epsilon$，$\epsilon \sim \mathcal{N}(0, \sigma^2 I)$。最小范数插值解 $\hat{\beta} = X^\dagger y$ 的风险为：

$$R(\hat{\beta}) = \underbrace{\sigma^2 \text{tr}(X^\dagger (X^\dagger)^\top)}_{\text{方差}} + \underbrace{\|(I - X^\dagger X)\beta^*\|^2}_{\text{偏差}}$$

在过参数化区间（$d > n$），随着 $d$ 增大，$X^\dagger$ 的范数减小，方差降低，测试误差再次下降。

### 24.5.4 神经正切核与泛化

NTK 框架给出了泛化误差的核方法界：

**测试误差**（Cao & Gu, 2019）：设 $\|y\|^2 \leq B$，核矩阵的最小特征值 $\lambda_{\min}(K_\infty) \geq \lambda_0 > 0$，则过拟合梯度下降解的泛化误差满足：

$$\mathbb{E}[\mathcal{L}_{test}] - \mathbb{E}[\mathcal{L}_{train}] \leq O\left(\sqrt{\frac{B^2 \text{tr}(K_\infty) / \lambda_0^2}{n}}\right)$$

这一界表明：**NTK 迹越小，泛化越好**，与平坦性的直觉吻合。

---

## 本章小结

| 主题 | 核心结论 | 关键工具/方法 |
|------|----------|---------------|
| 非凸优化与鞍点 | 严格鞍点处梯度下降以概率1逃逸；PGD 多项式时间逃离 | 稳定流形理论；随机矩阵理论 |
| 隐式正则化 | GD 偏向最小范数/低秩解；EOS 使曲率自稳定在 $2/\eta$ | 梯度流分析；Langevin 动力学 |
| 神经切线核 | 无限宽网络等价于核方法；训练动态线性化 | 递推核公式；核岭回归 |
| 损失曲面几何 | 局部极小值质量相近；存在低损失连接路径；彩票子网络 | 贝塞尔曲线优化；迭代幅度剪枝 |
| 优化与泛化统一 | PAC-Bayes 界联系 KL 散度与泛化；SAM 寻找平坦极小值；双下降挑战经典理论 | PAC-Bayes；SAM；最小范数解分析 |

**核心主线**：

$$\text{过参数化} \xrightarrow{\text{隐式正则化}} \text{特定解} \xrightarrow{\text{几何结构}} \text{低损失流形} \xrightarrow{\text{PAC-Bayes}} \text{泛化保证}$$

---

## 深度学习应用：可视化模式连通性与 NTK

### 代码示例 1：模式连通性可视化

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# ── 工具函数 ─────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ── 简单 MLP ──────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 数据生成 ──────────────────────────────────────────────────────────────

def make_dataset(n: int = 400, noise: float = 0.2):
    """生成两类螺旋数据。"""
    set_seed(0)
    theta = torch.linspace(0, 4 * np.pi, n // 2)
    r = torch.linspace(0.1, 1.0, n // 2)

    x0 = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    x1 = torch.stack([r * torch.cos(theta + np.pi), r * torch.sin(theta + np.pi)], dim=1)

    X = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(n // 2, dtype=torch.long),
                   torch.ones(n // 2, dtype=torch.long)])
    X += noise * torch.randn_like(X)
    return X, y


# ── 训练函数 ──────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-2,
) -> list:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    losses = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
    return losses


# ── 提取参数向量 ──────────────────────────────────────────────────────────

def params_to_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def vector_to_params(model: nn.Module, vec: torch.Tensor):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vec[idx: idx + numel].view(p.shape))
        idx += numel


# ── 计算路径损失 ──────────────────────────────────────────────────────────

def path_loss(
    model: nn.Module,
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    n_points: int = 21,
) -> tuple:
    """计算线性插值路径上各点的损失。"""
    criterion = nn.CrossEntropyLoss()
    ts = np.linspace(0, 1, n_points)
    losses = []
    with torch.no_grad():
        for t in ts:
            theta_t = (1 - t) * theta1 + t * theta2
            vector_to_params(model, theta_t)
            loss = criterion(model(X), y).item()
            losses.append(loss)
    return ts, losses


# ── 贝塞尔曲线路径优化 ────────────────────────────────────────────────────

def find_bezier_connection(
    model: nn.Module,
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
) -> torch.Tensor:
    """优化二次贝塞尔曲线的中间点，使路径损失最低。"""
    criterion = nn.CrossEntropyLoss()

    # 初始化中间点为线性中点
    theta_mid = nn.Parameter(((theta1 + theta2) / 2).clone())
    optimizer_mid = optim.Adam([theta_mid], lr=lr)

    for _ in range(epochs):
        optimizer_mid.zero_grad()
        # 随机采样路径上的点
        t = torch.rand(1).item()
        theta_t = (1 - t) ** 2 * theta1 + 2 * t * (1 - t) * theta_mid + t ** 2 * theta2
        vector_to_params(model, theta_t)
        # 前向传播需要梯度（通过 theta_mid）
        # 重新计算以保留计算图
        theta_t_grad = (1 - t) ** 2 * theta1 + 2 * t * (1 - t) * theta_mid + t ** 2 * theta2
        # 临时设置参数
        idx = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(theta_t_grad[idx: idx + numel].view(p.shape).detach())
            idx += numel
        out = model(X)
        loss = criterion(out, y)
        # 通过 theta_mid 反向传播
        loss.backward()
        optimizer_mid.step()

    return theta_mid.detach()


# ── 贝塞尔路径损失计算 ────────────────────────────────────────────────────

def bezier_path_loss(
    model: nn.Module,
    theta1: torch.Tensor,
    theta2: torch.Tensor,
    theta_mid: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    n_points: int = 21,
) -> tuple:
    criterion = nn.CrossEntropyLoss()
    ts = np.linspace(0, 1, n_points)
    losses = []
    with torch.no_grad():
        for t in ts:
            theta_t = (1 - t) ** 2 * theta1 + 2 * t * (1 - t) * theta_mid + t ** 2 * theta2
            vector_to_params(model, theta_t)
            loss = criterion(model(X), y).item()
            losses.append(loss)
    return ts, losses


# ── 主实验 ────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    X, y = make_dataset(n=400)

    # 训练两个独立模型
    model1 = MLP()
    model2 = MLP()
    print("训练模型1...")
    losses1 = train_model(model1, X, y, epochs=300)
    print(f"  最终损失: {losses1[-1]:.4f}")

    print("训练模型2...")
    set_seed(123)
    model2 = MLP()
    losses2 = train_model(model2, X, y, epochs=300)
    print(f"  最终损失: {losses2[-1]:.4f}")

    theta1 = params_to_vector(model1)
    theta2 = params_to_vector(model2)

    # 线性插值路径
    ts_lin, losses_lin = path_loss(model1, theta1, theta2, X, y)

    # 贝塞尔曲线路径（简化版：使用均匀采样而非完整优化）
    # 这里演示概念；实际中需更多优化步骤
    theta_mid = (theta1 + theta2) / 2
    # 微调中间点（简化）
    model_tmp = MLP()
    criterion = nn.CrossEntropyLoss()
    theta_mid_param = nn.Parameter(theta_mid.clone())
    opt_mid = optim.Adam([theta_mid_param], lr=1e-2)
    for step in range(200):
        opt_mid.zero_grad()
        t = np.random.random()
        theta_t = (1-t)**2 * theta1 + 2*t*(1-t) * theta_mid_param + t**2 * theta2
        vector_to_params(model_tmp, theta_t.detach())
        out = model_tmp(X)
        loss = criterion(out, y)
        # 近似梯度传递
        g = torch.autograd.grad(
            criterion(model_tmp(X), y),
            model_tmp.parameters(),
            allow_unused=True,
        )
        g_vec = torch.cat([gi.flatten() if gi is not None else torch.zeros(p.numel())
                           for gi, p in zip(g, model_tmp.parameters())])
        theta_mid_param.grad = 2 * t * (1 - t) * g_vec
        opt_mid.step()

    ts_bez, losses_bez = bezier_path_loss(
        model1, theta1, theta2, theta_mid_param.detach(), X, y
    )

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：路径损失对比
    ax = axes[0]
    ax.plot(ts_lin, losses_lin, 'b-o', markersize=4, label='线性插值')
    ax.plot(ts_bez, losses_bez, 'r-s', markersize=4, label='贝塞尔曲线')
    ax.axhline(losses1[-1], color='b', linestyle='--', alpha=0.5, label=f'模型1损失={losses1[-1]:.3f}')
    ax.axhline(losses2[-1], color='g', linestyle='--', alpha=0.5, label=f'模型2损失={losses2[-1]:.3f}')
    ax.set_xlabel('插值参数 t')
    ax.set_ylabel('训练损失')
    ax.set_title('模式连通性：损失路径对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：训练曲线
    ax = axes[1]
    ax.plot(losses1, label='模型1', alpha=0.8)
    ax.plot(losses2, label='模型2', alpha=0.8)
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('损失')
    ax.set_title('独立训练的两个模型')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mode_connectivity.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存至 mode_connectivity.png")

    # 计算损失壁垒（loss barrier）
    barrier = max(losses_lin) - (losses1[-1] + losses2[-1]) / 2
    bez_barrier = max(losses_bez) - (losses1[-1] + losses2[-1]) / 2
    print(f"\n线性路径损失壁垒: {barrier:.4f}")
    print(f"贝塞尔路径损失壁垒: {bez_barrier:.4f}")


if __name__ == "__main__":
    main()
```

### 代码示例 2：神经切线核计算与可视化

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


# ── NTK 计算（雅可比向量积方法）──────────────────────────────────────────

def compute_jacobian(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    计算 df/dtheta 的雅可比矩阵。
    输出形状: (n_samples, n_params)
    """
    n = x.shape[0]
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)

    jac = torch.zeros(n, n_params)
    for i in range(n):
        model.zero_grad()
        out = model(x[i:i+1])  # 形状 (1, output_dim)
        # 对每个输出维度（假设回归，output_dim=1）
        out.backward(torch.ones_like(out))
        grad_vec = torch.cat([p.grad.flatten() for p in params if p.grad is not None])
        jac[i] = grad_vec

    return jac


def compute_ntk(
    model: nn.Module,
    X: torch.Tensor,
    X2: torch.Tensor = None,
) -> torch.Tensor:
    """
    计算神经切线核矩阵 K(X, X2)。
    K[i,j] = <df/dtheta(x_i), df/dtheta(x_j)>
    """
    if X2 is None:
        X2 = X

    J1 = compute_jacobian(model, X)    # (n1, P)
    J2 = compute_jacobian(model, X2)   # (n2, P)
    K = J1 @ J2.T                      # (n1, n2)
    return K


# ── 理论 NTK（单隐层，ReLU 激活）──────────────────────────────────────────

def arc_cosine_kernel(x1: np.ndarray, x2: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Arc-cosine 核，对应 ReLU 激活的无限宽单隐层网络 NTK。

    K^(1)(x, x') = (1/pi) * ||x|| * ||x'|| * (sin(theta) + (pi - theta) * cos(theta))
    其中 theta = arccos(x·x' / (||x|| * ||x'||))
    """
    norm1 = np.linalg.norm(x1, axis=-1, keepdims=True)
    norm2 = np.linalg.norm(x2, axis=-1, keepdims=True)

    # 归一化内积
    cos_theta = np.clip(
        (x1 @ x2.T) / (norm1 @ norm2.T + 1e-8), -1.0, 1.0
    )
    theta = np.arccos(cos_theta)

    if order == 0:
        K = (np.pi - theta) / np.pi
    elif order == 1:
        K = (norm1 @ norm2.T) * (np.sin(theta) + (np.pi - theta) * cos_theta) / np.pi
    else:
        raise ValueError("只支持 order=0 或 1")
    return K


# ── NTK 随宽度的变化实验 ─────────────────────────────────────────────────

class WideNet(nn.Module):
    """单隐层宽网络（NTK 参数化）。"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
        # NTK 参数化：用 1/sqrt(hidden_dim) 初始化第二层
        nn.init.normal_(self.fc1.weight, std=1.0)
        nn.init.normal_(self.fc2.weight, std=1.0 / np.sqrt(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x) / np.sqrt(self.fc1.weight.shape[1]))
        return self.fc2(h)


def ntk_convergence_experiment():
    """验证 NTK 随宽度增大趋于确定性极限。"""
    torch.manual_seed(42)
    d = 5
    n = 10
    X = torch.randn(n, d)

    widths = [16, 64, 256, 1024]
    n_trials = 5
    ntk_std_list = []

    for width in widths:
        ntks = []
        for trial in range(n_trials):
            torch.manual_seed(trial)
            model = WideNet(d, width)
            K = compute_ntk(model, X).detach().numpy()
            ntks.append(K)

        ntks = np.array(ntks)  # (n_trials, n, n)
        # 各试验间 NTK 的标准差（归一化）
        std = ntks.std(axis=0) / (ntks.mean(axis=0) + 1e-8)
        ntk_std_list.append(std.mean())
        print(f"宽度={width:4d}: NTK 相对标准差 = {ntk_std_list[-1]:.4f}")

    # 与理论 NTK 对比
    X_np = X.numpy()
    K_theory = arc_cosine_kernel(X_np, X_np, order=1)

    # 最大宽度的 NTK
    torch.manual_seed(0)
    model_wide = WideNet(d, 2048)
    K_empirical = compute_ntk(model_wide, X).detach().numpy()

    # 归一化以便比较
    K_theory_norm = K_theory / K_theory.diagonal().mean()
    K_empirical_norm = K_empirical / K_empirical.diagonal().mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 左图：理论 NTK
    im0 = axes[0].imshow(K_theory_norm, cmap='hot', aspect='auto')
    axes[0].set_title('理论 NTK（Arc-cosine，无限宽）')
    plt.colorbar(im0, ax=axes[0])

    # 中图：经验 NTK（宽=2048）
    im1 = axes[1].imshow(K_empirical_norm, cmap='hot', aspect='auto')
    axes[1].set_title(f'经验 NTK（宽度=2048）')
    plt.colorbar(im1, ax=axes[1])

    # 右图：NTK 方差随宽度的变化
    axes[2].loglog(widths, ntk_std_list, 'bo-', markersize=8, label='实验数据')
    # 理论预测：标准差 ~ 1/sqrt(width)
    theory_std = [ntk_std_list[0] * np.sqrt(widths[0] / w) for w in widths]
    axes[2].loglog(widths, theory_std, 'r--', label=r'理论 $\propto 1/\sqrt{n}$')
    axes[2].set_xlabel('隐层宽度')
    axes[2].set_ylabel('NTK 相对标准差')
    axes[2].set_title('NTK 收敛性：方差随宽度衰减')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ntk_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存至 ntk_visualization.png")


# ── NTK 动力学：训练预测 vs 实际 ─────────────────────────────────────────

def ntk_dynamics_experiment():
    """
    验证 NTK 预测的训练动态是否与实际梯度流吻合。
    对足够宽的网络，NTK 预测应与实际轨迹一致。
    """
    torch.manual_seed(42)
    n, d = 20, 3
    X = torch.randn(n, d)
    y = torch.randn(n, 1)

    width = 512
    model = WideNet(d, width)

    # 计算初始 NTK（假设训练中保持不变）
    K = compute_ntk(model, X).detach()  # (n, n)

    # NTK 预测动态：f_t = f_0 + K(K_nn)^{-1}(y - f_0)(I - e^{-K t/n})
    f0 = model(X).detach()  # (n, 1)

    # 梯度流模拟（连续时间）
    lr = 0.01
    n_steps = 500
    actual_losses = []
    ntk_pred_losses = []

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(X)
        loss = 0.5 * F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        actual_losses.append(loss.item())

        # NTK 预测
        t = step * lr
        # 解析解：f_t - y = exp(-K t/n)(f0 - y)
        K_np = K.numpy()
        f0_np = f0.numpy().flatten()
        y_np = y.numpy().flatten()
        eigenvalues, eigenvectors = np.linalg.eigh(K_np / n)
        decay = np.exp(-eigenvalues * t)
        residual0 = f0_np - y_np
        residual_t = eigenvectors @ (decay * (eigenvectors.T @ residual0))
        ntk_loss = 0.5 * np.mean(residual_t ** 2)
        ntk_pred_losses.append(ntk_loss)

    plt.figure(figsize=(8, 5))
    steps = np.arange(n_steps)
    plt.semilogy(steps, actual_losses, 'b-', label='实际梯度下降', alpha=0.8)
    plt.semilogy(steps, ntk_pred_losses, 'r--', label='NTK 理论预测', alpha=0.8)
    plt.xlabel('训练步数')
    plt.ylabel('训练损失（对数尺度）')
    plt.title(f'NTK 预测 vs 实际训练动态（宽度={width}）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ntk_dynamics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存至 ntk_dynamics.png")


if __name__ == "__main__":
    print("=== NTK 收敛性实验 ===")
    ntk_convergence_experiment()
    print("\n=== NTK 动力学实验 ===")
    ntk_dynamics_experiment()
```

### 代码示例 3：稳定边缘（Edge of Stability）可视化

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class SmallNet(nn.Module):
    """用于演示 EOS 的小型全连接网络。"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def compute_hessian_max_eigenvalue(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    n_power_iter: int = 20,
) -> float:
    """
    用幂迭代法估计 Hessian 最大特征值 lambda_max。
    避免存储完整 Hessian 矩阵（参数量大时内存友好）。
    """
    loss = criterion(model(X), y)
    params = [p for p in model.parameters() if p.requires_grad]
    grad = torch.autograd.grad(loss, params, create_graph=True)
    grad_vec = torch.cat([g.flatten() for g in grad])

    # 随机初始化方向向量
    v = torch.randn_like(grad_vec)
    v = v / v.norm()

    for _ in range(n_power_iter):
        # Hessian-向量积（HVP）
        Hv = torch.autograd.grad(
            grad_vec, params,
            grad_outputs=torch.autograd.grad(
                grad_vec, params,
                grad_outputs=[vi.reshape(p.shape) for vi, p in zip(
                    v.split([p.numel() for p in params]), params
                )],
                retain_graph=True,
            ),
            retain_graph=True,
        )
        Hv_vec = torch.cat([h.flatten() for h in Hv])
        lambda_max = Hv_vec.dot(v).item()
        v = Hv_vec / (Hv_vec.norm() + 1e-10)

    return lambda_max


def eos_experiment():
    """演示 Edge of Stability 现象。"""
    torch.manual_seed(42)
    n, d = 200, 10
    X = torch.randn(n, d)
    true_w = torch.randn(d)
    y = X @ true_w + 0.1 * torch.randn(n)

    learning_rate = 0.5  # 较大学习率以触发 EOS
    model = SmallNet()
    criterion = nn.MSELoss()

    # 全批次梯度下降
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)

    losses = []
    lambda_maxes = []

    n_steps = 300
    check_interval = 10  # 每隔 check_interval 步计算 Hessian（计算量大）

    print(f"学习率: {learning_rate}, 稳定边缘: {2/learning_rate:.2f}")

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % check_interval == 0:
            # 估计 Hessian 最大特征值
            model.zero_grad()
            lam = compute_hessian_max_eigenvalue(model, X, y, criterion)
            lambda_maxes.append((step, lam))
            print(f"步骤 {step:3d}: 损失={loss.item():.4f}, λ_max={lam:.2f}, 2/η={2/learning_rate:.2f}")

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # 上图：损失曲线
    ax = axes[0]
    ax.semilogy(losses, 'b-', alpha=0.7, linewidth=1)
    ax.set_xlabel('训练步数')
    ax.set_ylabel('损失（对数尺度）')
    ax.set_title('Edge of Stability：损失曲线（注意非单调性）')
    ax.grid(True, alpha=0.3)

    # 下图：Hessian 最大特征值
    ax = axes[1]
    steps_lam, lam_vals = zip(*lambda_maxes)
    ax.plot(steps_lam, lam_vals, 'ro-', markersize=5, label=r'$\lambda_{\max}(\nabla^2 \mathcal{L})$')
    ax.axhline(2 / learning_rate, color='k', linestyle='--',
               label=f'$2/\\eta = {2/learning_rate:.1f}$（稳定边缘）')
    ax.set_xlabel('训练步数')
    ax.set_ylabel(r'$\lambda_{\max}$')
    ax.set_title('Hessian 最大特征值随训练的变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('edge_of_stability.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存至 edge_of_stability.png")


if __name__ == "__main__":
    eos_experiment()
```

---

## 练习题

### 基础题

**24.1** （严格鞍点逃逸）

考虑函数 $f(x, y) = x^2 - y^2$（一个鞍点位于原点）。

(a) 证明原点是严格鞍点，并写出 Hessian 矩阵及其特征值。

(b) 从初始点 $(x_0, y_0) = (0.1, 0.0)$ 出发，使用学习率 $\eta = 0.1$ 的梯度下降，手动计算前 3 步的迭代结果，说明梯度下降的行为。

(c) 若初始点 $(x_0, y_0) = (0, \epsilon)$（$\epsilon > 0$ 极小），梯度下降能否逃离鞍点？说明原因并给出理论保证。

---

**24.2** （隐式正则化）

考虑过参数化线性回归：$X = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$，$y = \begin{pmatrix} 1 \\ 1 \\ 2 \end{pmatrix}$，参数 $w \in \mathbb{R}^2$。

(a) 验证 $w^* = (1, 1)^\top$ 是零训练误差的解。是否存在其他零误差解？

(b) 从 $w_0 = (0, 0)^\top$ 出发使用梯度下降最小化均方误差 $\mathcal{L}(w) = \frac{1}{2n}\|Xw - y\|^2$，证明梯度 $\nabla \mathcal{L}(w_t) \in \text{row}(X)$（即始终在 $X$ 的行空间内）。

(c) 求梯度下降收敛的极限解，并验证它是最小范数解。

---

### 中级题

**24.3** （神经切线核推导）

考虑单隐层网络 $f_\theta(x) = \frac{1}{\sqrt{m}}\sum_{j=1}^m a_j \sigma(w_j^\top x)$，其中 $\theta = \{w_j, a_j\}_{j=1}^m$，$w_j \in \mathbb{R}^d$，$a_j \in \mathbb{R}$，$\sigma$ 为 ReLU 激活。

(a) 计算 $\frac{\partial f_\theta}{\partial w_j}$ 和 $\frac{\partial f_\theta}{\partial a_j}$。

(b) 写出 NTK 的表达式 $K_t(x, x') = \nabla_\theta f_\theta(x)^\top \nabla_\theta f_\theta(x')$，分别写出 $w_j$ 和 $a_j$ 贡献的部分。

(c) 当 $m \to \infty$（无限宽）时，利用大数定律说明 NTK 如何收敛到确定性极限，写出极限 NTK 的积分形式。

---

**24.4** （PAC-Bayes 界分析）

设 $Q = \mathcal{N}(\theta^*, \sigma^2 I_d)$，$P = \mathcal{N}(0, I_d)$，参数维度为 $d$。

(a) 计算 $KL(Q \| P)$。

(b) 设 $\|\theta^*\|^2 = d$（参数平均范数为 1），训练样本量为 $n$。写出 PAC-Bayes 界，并分析：要使泛化界 $\leq \epsilon$，需要 $\sigma$ 满足什么条件？

(c) 直觉上，为什么 $\sigma$ 大（解附近损失扁平）有利于泛化？结合 SAM 算法解释。

---

### 进阶题

**24.5** （双下降与最小范数解）

考虑线性回归：$y = X\beta^* + \epsilon$，$X \in \mathbb{R}^{n \times d}$（行为样本，列为特征），$\epsilon \sim \mathcal{N}(0, \sigma^2 I_n)$，真实参数 $\beta^* \in \mathbb{R}^d$，$\|\beta^*\|^2 = r^2$。

设特征矩阵满足：$\frac{1}{n}X^\top X \to \Sigma$（正定矩阵），且 $d/n \to \phi > 1$（过参数化比例）。

最小范数解为 $\hat{\beta} = X^\top(XX^\top)^{-1}y$。

(a) 利用 Sherman-Morrison-Woodbury 公式，将 $\hat{\beta}$ 的偏差项 $\|(I - X^\dagger X)\beta^*\|^2$ 用 $X$、$\beta^*$ 表示，说明当 $d \to \infty$（$\phi \to \infty$）时偏差的行为趋势。

(b) 将方差项 $\sigma^2 \text{tr}((XX^\top)^{-2}XX^\top)$ 简化，利用随机矩阵理论的 Marchenko-Pastur 定律，说明当 $\phi > 1$ 时方差随 $\phi$ 的变化趋势。

(c) 综合 (a)(b) 说明双下降现象：在 $\phi \approx 1$（参数数接近样本数）时为何出现峰值，在 $\phi \gg 1$ 时测试误差为何再次下降？与过参数化的隐式正则化联系起来解释。

---

## 练习答案

### 24.1 答案

**(a)** $\nabla f = (2x, -2y)^\top$，在原点 $\nabla f(0,0) = 0$，为临界点。

Hessian：$\nabla^2 f = \begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix}$

特征值为 $\lambda_1 = 2 > 0$，$\lambda_2 = -2 < 0$。由于存在负特征值，原点是**严格鞍点**。

**(b)** 梯度下降更新：$\begin{cases} x_{t+1} = x_t - 0.1 \cdot 2x_t = 0.8 x_t \\ y_{t+1} = y_t - 0.1 \cdot (-2y_t) = 1.2 y_t \end{cases}$

从 $(0.1, 0.0)$ 出发：
- 步骤 1：$(0.08, 0.0)$
- 步骤 2：$(0.064, 0.0)$
- 步骤 3：$(0.0512, 0.0)$

由于 $y_0 = 0$，$y$ 坐标始终为 0，梯度下降沿 $x$ 轴下降收敛到原点。**注意**：这演示了当初始点恰好在鞍点的稳定流形上时，梯度下降无法逃逸！

**(c)** 从 $(0, \epsilon)$ 出发，$x_t = 0$ 恒成立（梯度在 $x$ 方向为零），$y_{t+1} = 1.2 y_t$，$y$ 坐标指数增长趋向 $-\infty$（因为 $f(0,y) = -y^2$，沿负 $y^2$ 方向下降）。

梯度下降**不能逃**到 $x \neq 0$ 的区域——它沿负曲率方向（$y$ 轴）逃逸，但这恰好是鞍点稳定流形（$y=0$）之外的情形。

**理论保证**：由 Lee et al. (2016)，鞍点的稳定流形 $\{(0, y): y \in \mathbb{R}\}$ 是 $\mathbb{R}^2$ 中的零测集（一维子流形）。从几乎所有初始点（除 $(x_0, 0)$ 以外），梯度下降都能逃离。

---

### 24.2 答案

**(a)** 验证：$Xw^* = \begin{pmatrix}1\\1\\2\end{pmatrix} = y$，故零训练误差。

由于 $X$ 有 3 行 2 列，秩为 2（满列秩），方程 $Xw = y$ 有唯一解 $w^* = (1,1)^\top$。

（注：若样本数小于参数数则存在多解；本题参数数=2 < 样本数=3，此时方程组过定，最小二乘解唯一。）

**(b)** 梯度：$\nabla \mathcal{L}(w) = \frac{1}{n}X^\top(Xw - y)$。

由于 $\nabla \mathcal{L}(w) \in \text{col}(X^\top) = \text{row}(X)$，初始点 $w_0 = 0 \in \text{row}(X)$（平凡地），梯度更新 $w_{t+1} = w_t - \eta \nabla \mathcal{L}(w_t)$ 始终保持 $w_t \in \text{row}(X) = \mathbb{R}^2$（本题 $X$ 满列秩，行空间即 $\mathbb{R}^2$）。

**(c)** 本题 $X$ 满列秩，最小二乘解唯一（非欠定），即 $w^* = (X^\top X)^{-1}X^\top y = (1,1)^\top$。梯度下降收敛到此唯一解，它恰好也是最小范数解（唯一解即最小范数解）。

---

### 24.3 答案

**(a)**
$$\frac{\partial f_\theta}{\partial w_j} = \frac{a_j}{\sqrt{m}} \sigma'(w_j^\top x) \cdot x \in \mathbb{R}^d$$

$$\frac{\partial f_\theta}{\partial a_j} = \frac{1}{\sqrt{m}} \sigma(w_j^\top x) \in \mathbb{R}$$

**(b)** NTK 为（两部分之和）：

$$K(x, x') = \underbrace{\sum_{j=1}^m \frac{\partial f}{\partial w_j}(x)^\top \frac{\partial f}{\partial w_j}(x')}_{\text{第一层贡献}} + \underbrace{\sum_{j=1}^m \frac{\partial f}{\partial a_j}(x) \frac{\partial f}{\partial a_j}(x')}_{\text{第二层贡献}}$$

$$= \frac{1}{m}\sum_{j=1}^m a_j^2 \sigma'(w_j^\top x)\sigma'(w_j^\top x') (x^\top x') + \frac{1}{m}\sum_{j=1}^m \sigma(w_j^\top x)\sigma(w_j^\top x')$$

**(c)** 当 $m \to \infty$，$w_j \sim \mathcal{N}(0, I_d)$，$a_j \sim \mathcal{N}(0,1)$ i.i.d. 时，由大数定律：

$$K_\infty(x, x') = \mathbb{E}_{w \sim \mathcal{N}(0,I), a \sim \mathcal{N}(0,1)}\left[a^2 \sigma'(w^\top x)\sigma'(w^\top x')(x^\top x') + \sigma(w^\top x)\sigma(w^\top x')\right]$$

$$= \mathbb{E}_w[\sigma'(w^\top x)\sigma'(w^\top x')](x^\top x') + \mathbb{E}_w[\sigma(w^\top x)\sigma(w^\top x')]$$

这正是 arc-cosine 核的两项组合，完全由 $x$、$x'$ 及激活函数决定，与 $\theta$ 无关。

---

### 24.4 答案

**(a)** 对高斯分布，KL 散度的封闭形式：

$$KL(\mathcal{N}(\mu, \sigma^2 I) \| \mathcal{N}(0, I)) = \frac{1}{2}\left(d\sigma^2 + \|\mu\|^2 - d - d\ln\sigma^2\right)$$

代入 $\mu = \theta^*$，$\|\theta^*\|^2 = d$：

$$KL(Q\|P) = \frac{d}{2}\left(\sigma^2 + 1 - 1 - \ln\sigma^2\right) = \frac{d}{2}(\sigma^2 - \ln\sigma^2 - 1)$$

**(b)** PAC-Bayes 界为：

$$\mathbb{E}_Q[\mathcal{L}_{test}] \leq \mathbb{E}_Q[\mathcal{L}_{train}] + \sqrt{\frac{d(\sigma^2 - \ln\sigma^2 - 1)/2 + \ln(2\sqrt{n}/\delta)}{2n}}$$

要使右侧第二项 $\leq \epsilon$，需 $\frac{d(\sigma^2 - \ln\sigma^2 - 1)}{4n} \leq \epsilon^2$，即 $\sigma^2 - \ln\sigma^2 - 1 \leq \frac{4n\epsilon^2}{d}$。

当 $\sigma \to 0$ 时，$-\ln\sigma^2 \to +\infty$，KL 散度增大；当 $\sigma \to \infty$ 时，$\sigma^2$ 主导。最优 $\sigma^2 = 1$ 时 KL=0，但此时 $Q = P$，随机参数无训练误差优势。

实践中需平衡：$\sigma$ 足够大使训练误差保持低（解在高概率区域），同时 $\sigma$ 不太大避免 KL 爆炸。当损失曲面平坦（大 $\sigma$ 仍低训练误差）时，可以在更小 KL 代价下保证泛化。

**(c)** $\sigma$ 大意味着在 $\theta^*$ 附近半径 $\sigma$ 的球内，几乎所有参数都有低训练损失——这正是**平坦极小值**的定义。

SAM 直接最小化 $\max_{\|\epsilon\| \leq \rho}\mathcal{L}(\theta+\epsilon)$（最坏扰动下的损失），等价于寻找在 $\rho$ 半径内损失一致低的解，即平坦极小值。由 (b)，平坦解对应小 KL，从而 PAC-Bayes 泛化界更紧——这从理论上解释了为何 SAM 改善泛化。

---

### 24.5 答案

**(a)** 最小范数解 $\hat{\beta} = X^\top(XX^\top)^{-1}y = X^\top(XX^\top)^{-1}(X\beta^* + \epsilon)$。

令 $H = X^\top(XX^\top)^{-1}X$（$d \times d$ 投影矩阵，秩为 $n < d$），则：

$$\hat{\beta} - \beta^* = (H - I)\beta^* + X^\top(XX^\top)^{-1}\epsilon$$

偏差项：$\|(I - H)\beta^*\|^2$。由于 $H$ 是行空间 $\text{row}(X)$ 上的投影，$(I-H)\beta^*$ 是 $\beta^*$ 在 $\text{null}(X)$ 上的分量。

当 $d \to \infty$（$\phi \to \infty$）时，$\text{null}(X)$ 的维数为 $d - n \to \infty$，而 $\beta^*$ 的方向越来越"随机"地落在这个大零空间中，$(I-H)\beta^*$ 趋于 $\beta^*$（偏差趋于 $\|\beta^*\|^2 = r^2$）——**偏差趋于常数**，不随 $d$ 增大而发散。

**(b)** 方差项：$\sigma^2 \text{tr}((XX^\top)^{-2}XX^\top) = \sigma^2 \text{tr}((XX^\top)^{-1}) = \sigma^2 \sum_{i=1}^n \frac{1}{\lambda_i(XX^\top)}$

$\frac{1}{n}XX^\top$ 的特征值分布由 Marchenko-Pastur 定律给出（$\phi = d/n$）：

$$\mu_\phi(d\lambda) = \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{2\pi \lambda} d\lambda, \quad \lambda_{\pm} = (1 \pm \phi^{-1/2})^2$$

（注意这里 $\phi > 1$ 时矩阵 $XX^\top$ 满秩，所有特征值非零。）

当 $\phi \to \infty$（$d \gg n$），特征值集中在 $\phi^{-1}$ 附近（因为 $XX^\top/d$ 趋于 $I_n$），故 $\lambda_i(XX^\top) \approx d/n$，方差 $\approx \sigma^2 n/(d/n) = \sigma^2 n^2/d \to 0$。

**方差随 $\phi$ 增大而减小。**

**(c)** 综合分析：

- **欠参数化区域** ($d < n$)：最小二乘解有偏（欠拟合），偏差大；方差有限。随 $d$ 增大，偏差减小。
- **插值门槛** ($d \approx n$)：恰好插值。$XX^\top$ 接近奇异，最小特征值趋近 0，方差项 $\sum 1/\lambda_i \to \infty$。**测试误差出现峰值**——这正是双下降的第一个下降与峰值之间的区域。
- **过参数化区域** ($d \gg n$)：存在无穷多插值解。梯度下降选择最小范数解（隐式正则化）。根据 (b)，方差 $\to 0$；根据 (a)，偏差趋于常数 $r^2$（不发散）。因此**总测试误差再次下降**，形成双下降的第二段下降。

这一分析揭示了隐式正则化（最小范数解）是过参数化下良好泛化的关键：它不仅控制了方差，而且偏差被几何结构自然限制，从而在 $d \gg n$ 时实现低测试误差。
