# 概率论中的 AI 思维

> **一例速记**：VAE 如何"压缩"一张图片到隐变量，再"还原"出来？
> 编码器把图片 $x$ 映射到隐空间的分布 $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$；解码器从分布中采样 $z$，重建 $\hat{x} = p_\theta(x|z)$。训练目标：最大化 ELBO = 重建好 + 隐变量分布接近先验。
> **生成式 AI = 概率分布的参数化学习：编码器建分布，解码器从分布生成。**

---

## 一、为什么概率论是生成式 AI 的语言

判别式模型（分类器）学习 $p(y|x)$——给定输入预测标签。生成式模型学习 $p(x)$ 或 $p(x|y)$——建模数据本身的分布，然后从中采样生成新数据（图像、文本、音频）。

概率论贯穿现代生成式 AI 的每个核心模块：

| 模型 | 核心概率操作 |
|---|---|
| **VAE** | 变分推断（近似后验）+ 重参数化梯度 |
| **扩散模型** | 随机过程（加噪/去噪）+ 得分匹配 |
| **Normalizing Flow** | 变量变换公式 + Jacobian 行列式 |
| **GAN** | 博弈论 + JS 散度最小化 |
| **贝叶斯神经网络** | 参数后验 + 变分推断 |

理解这五类模型背后的概率机制，就能在读论文时"看穿"公式，在调参时"理解"行为。

---

## 二、ELBO 与变分推断的统一框架

### 2.1 变分推断的基本问题

**目标**：在贝叶斯模型中，给定观测数据 $x$，计算隐变量 $z$ 的后验分布：

$$p(z|x) = \frac{p(x|z)\,p(z)}{p(x)} = \frac{p(x|z)\,p(z)}{\int p(x|z)\,p(z)\,dz}$$

分母 $p(x) = \int p(x|z)\,p(z)\,dz$ 是**边际似然**（evidence），在连续高维空间中难以精确计算——这是贝叶斯推断的根本困难。

**变分推断的策略**：用参数族 $\{q_\phi(z|x)\}$ 中的分布近似 $p(z|x)$，通过最优化 $\phi$ 使近似质量最高。

### 2.2 ELBO 的推导

最小化 $D_{\mathrm{KL}}(q_\phi(z|x) \| p(z|x))$，展开：

$$D_{\mathrm{KL}}(q_\phi(z|x)\|p(z|x)) = \mathbb{E}_{q_\phi}\!\left[\log\frac{q_\phi(z|x)}{p(z|x)}\right]$$

代入 $p(z|x) = p(x|z)p(z)/p(x)$：

$$= \mathbb{E}_{q_\phi}\!\left[\log q_\phi(z|x) - \log p(x|z) - \log p(z) + \log p(x)\right]$$

整理：

$$D_{\mathrm{KL}}(q_\phi(z|x)\|p(z|x)) = \log p(x) - \underbrace{\left[\mathbb{E}_{q_\phi}[\log p(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))\right]}_{\mathcal{L}(\phi;\,x)\;\triangleq\;\text{ELBO}}$$

由于 KL 散度 $\geq 0$，故：

$$\log p(x) \geq \mathcal{L}(\phi;\,x) = \mathbb{E}_{q_\phi}[\log p(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))$$

**ELBO**（Evidence Lower BOund）是 $\log p(x)$ 的下界，**最大化 ELBO $\Leftrightarrow$ 最小化 $D_{\mathrm{KL}}(q_\phi \| p(\cdot|x))$（同时间接最大化 $\log p(x)$）**。

---

## 三、VAE：隐变量生成模型

### 3.1 模型结构

VAE（Variational Autoencoder）由两个神经网络组成：

- **编码器** $q_\phi(z|x)$：将输入 $x$ 映射到隐变量 $z$ 的近似后验（参数为 $\phi$）；
- **解码器** $p_\theta(x|z)$：从隐变量 $z$ 重建数据 $x$（参数为 $\theta$）。

先验：$p(z) = \mathcal{N}(0, I)$（各向同性高斯）。

通常设 $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$（均值和方差由编码器神经网络输出）。

### 3.2 ELBO 的两项分解

$$\mathcal{L}(\phi, \theta;\,x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重建损失（负）}} - \underbrace{D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))}_{\text{KL 正则化项}}$$

- **重建项**：鼓励编码再解码后能高保真重建原始输入（对应最大似然）；
- **KL 项**：惩罚编码器分布偏离先验 $\mathcal{N}(0, I)$，让隐空间平滑、连续，可以从先验采样生成新样本。

**对高斯 $q_\phi$，KL 项有闭合公式**（每维独立）：

$$D_{\mathrm{KL}}(q_\phi(z|x)\|p(z)) = \frac{1}{2}\sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - \ln\sigma_j^2 - 1\right)$$

（参见 Toolkit 10 例 2 的推导。）

### 3.3 重参数化技巧（Reparameterization Trick）

**问题**：ELBO 中含 $\mathbb{E}_{q_\phi(z|x)}[\cdot]$，直接求关于 $\phi$ 的梯度时，期望符号内含 $\phi$（分布本身依赖 $\phi$），不能把梯度算子移入期望。

**解决方案**：将 $z$ 的随机性与参数解耦：

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

（$\odot$ 是逐元素乘。）此时 $z$ 是 $\epsilon$（固定分布，与 $\phi$ 无关）的确定性函数，梯度可以通过 $\mu_\phi$、$\sigma_\phi$ 传到编码器参数 $\phi$：

$$\nabla_\phi \mathcal{L} = \nabla_\phi \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}[\log p_\theta(x|\mu_\phi + \sigma_\phi \odot \epsilon)] - \nabla_\phi D_{\mathrm{KL}}$$

期望中的梯度可以用 Monte Carlo 估计（单样本 $\epsilon$）：

$$\nabla_\phi \mathcal{L} \approx \nabla_\phi [\log p_\theta(x|z)] \big|_{z = \mu_\phi + \sigma_\phi \odot \epsilon} - \nabla_\phi D_{\mathrm{KL}}$$

**重参数化的意义**：让基于采样的期望变成关于参数可微——这是将贝叶斯推断与深度学习（SGD 训练）融合的关键技术。

---

## 四、扩散模型：逐步加噪与学习去噪

### 4.1 前向过程（加噪）

**扩散模型**（Diffusion Model）的前向过程定义为 $T$ 步加噪的马尔可夫链：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t;\;\sqrt{1-\beta_t}\,x_{t-1},\;\beta_t I)$$

其中 $\{\beta_t\}$ 是噪声调度（noise schedule，从小到大），$T$ 步后 $x_T \approx \mathcal{N}(0, I)$（纯噪声）。

**便捷的闭合公式**：定义 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$，则：

$$q(x_t | x_0) = \mathcal{N}(x_t;\;\sqrt{\bar{\alpha}_t}\,x_0,\;(1-\bar{\alpha}_t)I)$$

任意时刻 $t$ 的含噪图像可以直接由 $x_0$ 生成：$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$，$\epsilon \sim \mathcal{N}(0,I)$。

### 4.2 反向过程（去噪）

训练一个神经网络 $p_\theta(x_{t-1}|x_t)$ 来"逆转"加噪过程：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\;\mu_\theta(x_t, t),\;\Sigma_\theta(x_t, t))$$

**训练目标**（简化版，DDPM）：预测加在 $x_0$ 上的噪声 $\epsilon$：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\;t)\|^2\right]$$

即让网络 $\epsilon_\theta$ 从含噪图像和时间步 $t$ 预测噪声——简洁有力。

### 4.3 得分匹配（Score Matching）

得分函数（score function）：$s_\theta(x, t) = \nabla_x \log p_t(x)$（对数密度的梯度）。

$\epsilon_\theta$ 与得分函数的关系：$\epsilon_\theta(x_t, t) = -\sqrt{1-\bar{\alpha}_t}\,s_\theta(x_t, t)$。

预测噪声等价于学习得分函数——生成时用 Langevin 动力学（或 ODE 求解器）沿得分方向走，从噪声走回真实数据分布。

---

## 五、Normalizing Flow：精确似然的生成模型

### 5.1 变量变换公式

设 $z \sim p_Z(z)$（简单分布，如 $\mathcal{N}(0,I)$），通过可逆变换 $x = f(z)$（$f$ 是双射），则 $x$ 的密度：

$$p_X(x) = p_Z(f^{-1}(x)) \cdot \left|\det\frac{\partial f^{-1}}{\partial x}\right| = p_Z(z) \cdot \left|\det\frac{\partial f}{\partial z}\right|^{-1}$$

取对数：

$$\log p_X(x) = \log p_Z(z) - \log\left|\det J_f(z)\right|$$

其中 $J_f(z) = \partial f/\partial z$ 是 Jacobian 矩阵。

### 5.2 Normalizing Flow 的训练

叠加 $K$ 个可逆变换 $f = f_K \circ \cdots \circ f_1$，则：

$$\log p_X(x) = \log p_Z(z_0) - \sum_{k=1}^K \log|\det J_{f_k}(z_{k-1})|$$

**训练**：最大化数据的对数似然（精确计算，无需近似）。

**设计挑战**：Jacobian 行列式通常是 $O(d^3)$ 的运算，需要特殊网络结构使其高效可计算：
- **RealNVP / GLOW**：耦合层（coupling layers）使 Jacobian 为三角矩阵，行列式 $= $ 对角线元素之积；
- **Neural ODE**：连续时间版本，Jacobian 迹替代行列式；
- **Autoregressive Flow**：自回归结构使每维的条件密度可以用神经网络建模。

---

## 六、GAN：博弈论视角

### 6.1 min-max 博弈

GAN（Generative Adversarial Network）由生成器 $G_\theta(z)$（$z \sim p_Z$）和判别器 $D_\psi(x)$ 构成，训练目标：

$$\min_\theta \max_\psi \mathbb{E}_{x \sim p_{\text{data}}}[\log D_\psi(x)] + \mathbb{E}_{z \sim p_Z}[\log(1 - D_\psi(G_\theta(z)))]$$

**最优判别器**（固定 $G$）：$D^*(x) = \dfrac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}$

将最优判别器代回目标函数，可以证明：

$$\max_\psi V(G, D) = -\ln 4 + 2\,D_{\mathrm{JS}}(p_{\text{data}} \| p_G)$$

其中 **JS 散度**（Jensen-Shannon divergence）：

$$D_{\mathrm{JS}}(p \| q) = \frac{1}{2}D_{\mathrm{KL}}\!\left(p\,\Big\|\,\frac{p+q}{2}\right) + \frac{1}{2}D_{\mathrm{KL}}\!\left(q\,\Big\|\,\frac{p+q}{2}\right)$$

JS 散度是对称的（$D_{\mathrm{JS}}(p\|q) = D_{\mathrm{JS}}(q\|p)$）、有界的（$\in [0, \ln 2]$），是衡量分布差异的度量。

**训练 $G$ = 最小化 JS 散度** = 让生成分布逼近真实数据分布。

### 6.2 原始 GAN 的问题：梯度消失与模式崩溃

**梯度消失**：当 $p_{\text{data}}$ 与 $p_G$ 支撑不重叠（高维空间中常见），JS 散度 $= \ln 2$（常数），梯度为 0，$G$ 无法获得有效梯度。

**解决方案——Wasserstein GAN（WGAN）**：用 Wasserstein 距离（Earth Mover's Distance）替代 JS 散度：

$$W(p, q) = \inf_{\gamma \in \Pi(p,q)} \mathbb{E}_{(x,y)\sim\gamma}[\|x-y\|]$$

Wasserstein 距离即使分布不重叠也提供有意义的梯度信号，且满足度量的三角不等式。

---

## 七、贝叶斯神经网络

### 7.1 参数后验

普通神经网络给出点估计 $\hat{\theta}$；贝叶斯神经网络（BNN）维护参数的完整后验分布 $p(\theta|D)$，用于不确定性量化：

$$p(\theta|D) \propto p(D|\theta)\,p(\theta)$$

**挑战**：参数维度 $d \sim 10^7$—$10^{11}$，精确后验不可行。

### 7.2 近似推断方法

**变分推断（VI）**：用 $q_\phi(\theta)$ 近似后验，最大化 ELBO：

$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}[\log p(D|\theta)] - D_{\mathrm{KL}}(q_\phi(\theta)\|p(\theta))$$

通常设 $q_\phi(\theta) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$（均值场近似，各参数独立），用重参数化技巧梯度优化。

**Laplace 近似**：在 MAP 估计 $\hat{\theta}$ 处做二阶泰勒展开，用高斯 $\mathcal{N}(\hat{\theta}, H^{-1})$ 近似后验（$H$ 是负对数后验的 Hessian）。计算一次 Hessian 即可，不需要迭代；缺点是 Hessian 存储成本 $O(d^2)$，大模型用低秩近似（KFAC, subspace Laplace）。

**MC Dropout**：训练时使用 Dropout，推断时也保持 Dropout 并运行 $T$ 次前向，输出的方差作为不确定性估计。与变分推断等价（Dropout 训练 = 近似变分推断，Gal & Ghahramani 2016）。

---

## 八、演示题：从 ELBO 推出"重建损失 + KL 正则"

**题目**：从 VAE 的 ELBO 定义出发，完整推导"重建损失 + KL 正则"的分解：

$$\mathcal{L}(\phi, \theta;\,x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))$$

并解释为什么最大化 ELBO 等价于同时最小化重建误差和 KL 散度。

> **第 1 步：从贝叶斯推断出发写 $\log p(x)$。**
>
> 边际对数似然：
> $$\log p_\theta(x) = \log \int p_\theta(x|z)\,p(z)\,dz$$
>
> 由于 $p(z)$ 是先验，直接计算困难。引入任意分布 $q_\phi(z|x)$（编码器），乘以 $q_\phi/q_\phi = 1$：
>
> $$\log p_\theta(x) = \log \int p_\theta(x|z)\,p(z)\,\frac{q_\phi(z|x)}{q_\phi(z|x)}\,dz = \log \mathbb{E}_{q_\phi(z|x)}\!\left[\frac{p_\theta(x|z)\,p(z)}{q_\phi(z|x)}\right]$$

> **第 2 步：用 Jensen 不等式（$\log$ 是凹函数）提取下界。**
>
> $$\log p_\theta(x) = \log \mathbb{E}_{q_\phi}\!\left[\frac{p_\theta(x|z)\,p(z)}{q_\phi(z|x)}\right] \geq \mathbb{E}_{q_\phi}\!\left[\log\frac{p_\theta(x|z)\,p(z)}{q_\phi(z|x)}\right]$$
>
> 右端即为 ELBO：
>
> $$\mathcal{L} = \mathbb{E}_{q_\phi}\!\left[\log p_\theta(x|z) + \log p(z) - \log q_\phi(z|x)\right]$$

> **第 3 步：拆分为两项。**
>
> $$\mathcal{L} = \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] + \mathbb{E}_{q_\phi}\!\left[\log\frac{p(z)}{q_\phi(z|x)}\right]$$
>
> 第二项恰好是 KL 散度的相反数：
>
> $$\mathbb{E}_{q_\phi}\!\left[\log\frac{p(z)}{q_\phi(z|x)}\right] = -\mathbb{E}_{q_\phi}\!\left[\log\frac{q_\phi(z|x)}{p(z)}\right] = -D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))$$
>
> 故：
>
> $$\boxed{\mathcal{L}(\phi, \theta;\,x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重建项（越大越好）}} - \underbrace{D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))}_{\text{KL 正则（越小越好）}}}$$

> **第 4 步：语义解释两项分别的作用。**
>
> **重建项** $\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]$：
> - 直觉：从编码分布 $q_\phi(z|x)$ 中采样 $z$，然后解码器重建 $x$ 的对数概率，期望越大越好；
> - 若解码器输出高斯分布（方差固定），则 $\log p_\theta(x|z) = -\|x - \mu_\theta(z)\|^2 / (2\sigma^2) + \text{const}$，最大化等价于最小化均方误差（MSE）；
> - 若解码器输出伯努利分布（图像像素），则等价于最小化二元交叉熵。
>
> **KL 项** $D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))$：
> - 直觉：惩罚编码分布偏离先验 $\mathcal{N}(0, I)$；
> - 功能：正则化编码器，使隐空间平滑——隐空间中相邻点解码出相似的图像，可以插值和生成；
> - $\beta$-VAE（$\beta > 1$）加强此项，鼓励隐变量解耦（不同维度控制不同语义属性）。
>
> **训练的张力**：重建项希望编码器把每张图"精确编码"（$\sigma_\phi \to 0$，后验集中），KL 项希望编码器接近先验（$\mu_\phi \to 0, \sigma_\phi \to 1$）。两项的平衡决定 VAE 生成质量 vs 重建质量的权衡。

> **第 5 步：等号成立条件。**
>
> Jensen 不等式等号成立当且仅当 $\log$ 内的随机变量为常数：
>
> $$\frac{p_\theta(x|z)\,p(z)}{q_\phi(z|x)} = \text{const}$$
>
> 即 $q_\phi(z|x) \propto p_\theta(x|z)\,p(z) = p(z|x)$（后验）。
>
> **结论**：ELBO 等于 $\log p(x)$ 当且仅当编码器 $q_\phi$ 精确等于真实后验 $p(z|x)$——ELBO 的"缺口"正好是 $D_{\mathrm{KL}}(q_\phi \| p(\cdot|x))$，VAE 通过最大化 ELBO 间接最小化这个 KL 散度。

---

## 九、思考路标

**路标 1**：ELBO $= \log p(x) - D_{\mathrm{KL}}(q_\phi \| p(\cdot|x))$ 这个等式是变分推断的核心。**最大化 ELBO 同时做两件事**：提高模型的边际似然（让数据更"可信"）+ 让近似后验更接近真实后验。理解这个等式，就理解了为什么 VAE 的训练目标是合理的。

**路标 2**：重参数化技巧 $z = \mu_\phi + \sigma_\phi \odot \epsilon$（$\epsilon \sim \mathcal{N}(0,I)$）的本质是**将随机性从参数化变量中"分离"出来**，让梯度可以通过确定性路径传播。这一技巧不只适用于高斯分布——任何可以表示为确定性变换（路径梯度）的分布都可以用重参数化（如 Gumbel-softmax 让离散变量可微分）。

**路标 3**：扩散模型的前向加噪 $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$ 是一个**数学上极优雅的设计**：直接计算任意 $t$ 时刻的含噪图像，训练时每步独立采样 $(t, \epsilon)$，无需模拟完整马尔可夫链。这得益于高斯分布的链式卷积性质（两个高斯相加仍是高斯）。

**路标 4**：VAE（最大化 ELBO，反向 KL），Flow（精确最大似然），GAN（JS 散度）和扩散模型（去噪得分匹配）本质上都是**让模型分布逼近数据分布的不同度量/策略**：用不同的散度/距离，权衡计算效率与样本质量。Wasserstein GAN 换用 Wasserstein 距离，解决了原始 GAN 的梯度消失问题——这说明"选哪种散度"不是小事，直接影响训练稳定性。

**路标 5**：贝叶斯神经网络的不确定性分为两类——**偶然不确定性**（aleatoric，数据本身的噪声，不可降低）和**认知不确定性**（epistemic，模型知识的不足，可以通过更多数据降低）。BNN 的后验方差主要捕捉认知不确定性，这在主动学习、异常检测、安全 AI 中至关重要。普通神经网络的 softmax 输出不是真实的不确定性度量（即使在分布外数据上也可能输出高置信度）。

**路标 6**：Normalizing Flow 的 Jacobian 行列式计算 $\log|\det J|$ 是设计流模型时的根本约束。**自回归流**利用三角 Jacobian（对角线以下为零）使行列式 $= $ 对角线元素之积（$O(d)$），代价是不能并行采样（必须顺序生成每一维）。**耦合层**（RealNVP）牺牲表达能力换取生成和训练的并行性。这是"能力-效率"权衡的典型例子。

---

## 十、典型应用例题

### 例 1：VAE 的重参数化梯度计算

**场景**：设编码器输出 $\mu_\phi = 1.0$，$\sigma_\phi = 0.5$（标量），采样 $\epsilon = 0.3$，则 $z = 1.0 + 0.5 \times 0.3 = 1.15$。

解码器输出高斯 $p_\theta(x|z) = \mathcal{N}(\mu_\theta(z), 1)$，设 $\mu_\theta(1.15) = 0.9$，真实值 $x = 1.0$。

**重建损失**（MSE形式）：$-\log p_\theta(x|z) = \frac{(x-\mu_\theta(z))^2}{2} = \frac{(1.0-0.9)^2}{2} = 0.005$

**KL 项**（一维）：$\frac{1}{2}(\mu_\phi^2 + \sigma_\phi^2 - \ln\sigma_\phi^2 - 1) = \frac{1}{2}(1 + 0.25 - \ln 0.25 - 1) = \frac{1}{2}(0.25 + 1.386) = 0.818$

**ELBO**（取负，即为 loss）：$0.005 + 0.818 = 0.823$

**梯度流**：重建梯度 $\to \mu_\theta \to z = \mu_\phi + \sigma_\phi\epsilon \to \mu_\phi, \sigma_\phi$（可微路径）。KL 梯度直接对 $\mu_\phi, \sigma_\phi$ 求导（闭合公式）。

### 例 2：扩散模型的一步训练

**场景**：$T = 1000$，$\bar{\alpha}_{500} = 0.1$（第 500 步）。给定清晰图像 $x_0$（取值 $0.8$），采样 $\epsilon = 1.2$（从 $\mathcal{N}(0,1)$）。

含噪图像：$x_{500} = \sqrt{0.1}\times 0.8 + \sqrt{0.9}\times 1.2 = 0.253 + 1.138 = 1.391$

训练目标：网络 $\epsilon_\theta(x_{500}, 500)$ 应输出 $1.2$（还原原始噪声）。

损失：$\|\epsilon - \epsilon_\theta(x_{500}, 500)\|^2 = (1.2 - \hat{\epsilon}_\theta)^2$

**直觉**：$\bar{\alpha}_{500} = 0.1$ 说明第 500 步已非常嘈杂（原始信号占比 $\sqrt{0.1} \approx 31.6\%$，噪声占比 $\sqrt{0.9} \approx 94.9\%$），网络此时主要学习"从噪声中辨认原始信号方向"。

### 例 3：Dropout 作为变分推断

**场景**：一个有 Dropout（丢弃率 $p = 0.5$）的两层网络，推断时运行 $T = 50$ 次前向传播，输出预测分布 $\{\hat{y}_1, \ldots, \hat{y}_{50}\}$。

**不确定性估计**：
- 预测均值：$\bar{y} = \frac{1}{50}\sum_{t=1}^{50}\hat{y}_t$（点预测）
- 预测方差：$\text{Var}(\hat{y}) = \frac{1}{50}\sum_{t=1}^{50}(\hat{y}_t - \bar{y})^2$（认知不确定性）

**MC Dropout 的等价性**：每次前向相当于从近似后验 $q(\theta)$（二元混合高斯，每个权重要么正常取值要么为 0）中采样参数，输出方差近似参数后验不确定性（Gal & Ghahramani 2016）。

---

## 十一、自测题

**第 1 题**：写出 VAE 的完整训练目标（ELBO）的两项，并说明为什么每项都不能单独作为训练目标。

> 提示：只有重建项时，编码器会让 $\sigma_\phi \to 0$（后验退化为确定性编码），失去生成多样性；只有 KL 项时，解码器无法学到有用结构（隐变量与输入无关，$q_\phi \equiv p(z) = \mathcal{N}(0,I)$ 总能使 KL $=0$）。两项共同作用才能学到有意义的隐表示：KL 约束隐空间结构，重建项要求隐空间包含足够的输入信息。

**第 2 题**：扩散模型中，为什么前向过程设计为高斯马尔可夫链（而非其他分布），且可以写成 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ 的闭合形式？

> 提示：高斯分布在线性叠加下封闭（两个独立高斯之和仍是高斯），且参数可以解析追踪。具体地，$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_t$，递推展开：$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sum_{s=1}^t \sqrt{\bar{\alpha}_t/\bar{\alpha}_s}\cdot\sqrt{1-\alpha_s}\cdot\epsilon_s$，利用方差叠加性质化简为单个高斯，系数恰好为 $\sqrt{1-\bar{\alpha}_t}$。这是高斯分布"可合并性"的直接应用。

**第 3 题**：GAN 的判别器 $D(x)$ 在训练收敛时，$D^*(x) = p_{\text{data}}(x)/[p_{\text{data}}(x)+p_G(x)]$。当 $p_G = p_{\text{data}}$ 时，$D^*(x) = ?$，此时 JS 散度是多少？

> 提示：$D^* = 1/2$（判别器完全无法区分真实与生成数据）；$D_{\mathrm{JS}}(p_{\text{data}}\|p_G) = 0$（两分布相同时 KL $= 0$，故 JS $= 0$）。此时对应 GAN 的"Nash 均衡"——生成器达到全局最优。

**第 4 题**：重参数化技巧 $z = \mu + \sigma\epsilon$（$\epsilon \sim \mathcal{N}(0,1)$）允许梯度通过采样步骤传播。设 ELBO 的重建项（简化为一维 MSE）$L_{\text{recon}} = (x - z)^2$，$x = 1.0$，$\mu = 0.5$，$\sigma = 0.8$，$\epsilon = 1.0$。计算 $\partial L_{\text{recon}} / \partial \mu$ 和 $\partial L_{\text{recon}} / \partial \sigma$。

> 提示：$z = 0.5 + 0.8 \times 1.0 = 1.3$，$L = (1.0 - 1.3)^2 = 0.09$。$\partial L/\partial z = 2(z-x) = 2(1.3-1.0) = 0.6$。$\partial z/\partial \mu = 1$，$\partial z/\partial \sigma = \epsilon = 1.0$（重参数化关键！梯度直接通过 $\epsilon$ 传到 $\sigma$）。$\partial L/\partial \mu = 0.6 \times 1 = 0.6$，$\partial L/\partial \sigma = 0.6 \times 1.0 = 0.6$。若没有重参数化，$\partial L/\partial \sigma$ 无法计算（采样操作不可微）。

**第 5 题**：贝叶斯神经网络的预测不确定性分解为认知不确定性和偶然不确定性，写出回归问题中两者的数学表达式（利用全方差公式 $\text{Var}[y^*] = \mathbb{E}[\text{Var}(y^*|\theta)] + \text{Var}[\mathbb{E}(y^*|\theta)]$）。

> 提示：$\text{Var}[y^*|x^*, D] = \underbrace{\mathbb{E}_\theta[\sigma_\theta^2(x^*)]}_{\text{偶然不确定性（数据噪声均值）}} + \underbrace{\text{Var}_\theta[\mu_\theta(x^*)]}_{\text{认知不确定性（预测均值的方差）}}$。MC Dropout 在实践中近似计算后者：$\widehat{\text{Var}}_{\text{epist}} = \frac{1}{T}\sum_t \hat{y}_t^2 - \bar{y}^2$（第二项）。偶然不确定性需要模型显式输出 $\sigma_\theta^2(x^*)$（异方差回归）。
