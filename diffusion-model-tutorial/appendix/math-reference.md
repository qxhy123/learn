# 附录A：数学速查表

> 本附录汇总教程中频繁使用的数学公式，按主题分类，方便快速查阅。

---

## A.1 概率论基础

### 基本公式

| 公式 | 名称 |
|------|------|
| $p(x, y) = p(x|y)p(y) = p(y|x)p(x)$ | 联合概率分解 |
| $p(x|y) = \frac{p(y|x)p(x)}{p(y)}$ | 贝叶斯定理 |
| $p(x) = \int p(x|z)p(z)dz$ | 边缘化（全概率公式） |
| $\mathbb{E}[f(X)] = \int f(x)p(x)dx$ | 期望定义 |

### 高斯分布

**一维**：$p(x) = \mathcal{N}(\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$

**多维**：$p(\mathbf{x}) = \mathcal{N}(\boldsymbol\mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol\mu)^T\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\right)$

**关键性质**：
- 线性变换：$A\mathbf{x} + b \sim \mathcal{N}(A\boldsymbol\mu + b, A\Sigma A^T)$（$\mathbf{x}\sim\mathcal{N}(\boldsymbol\mu,\Sigma)$）
- 独立高斯之和：$\mathcal{N}(\mu_1,\sigma_1^2) + \mathcal{N}(\mu_2,\sigma_2^2) = \mathcal{N}(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$
- 条件高斯：$\mathbf{x}_1|\mathbf{x}_2 \sim \mathcal{N}(\boldsymbol\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2-\boldsymbol\mu_2),\ \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21})$

**KL散度（高斯）**：

$$D_{KL}(\mathcal{N}(\mu_1,\sigma_1^2)\|\mathcal{N}(\mu_2,\sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

**特殊情形**（$p=\mathcal{N}(\mu,\text{diag}(\sigma^2))$，$q=\mathcal{N}(0,I)$）：

$$D_{KL}(p\|q) = \frac{1}{2}\sum_{j=1}^d\left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)$$

---

## A.2 信息论

| 公式 | 名称 |
|------|------|
| $H(X) = -\mathbb{E}[\log p(X)]$ | 熵 |
| $D_{KL}(p\|q) = \mathbb{E}_p[\log p/q] \geq 0$ | KL散度（非对称！） |
| $I(X;Y) = D_{KL}(p(x,y)\|p(x)p(y))$ | 互信息 |
| $\log p(x) = \mathcal{L}(\theta,\phi;x) + D_{KL}(q_\phi\|p_\theta(\cdot|x))$ | ELBO等式 |

**Jensen不等式**：$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$（$f$为凸函数，反向对凹函数）

---

## A.3 变分推断与ELBO

**ELBO定义**：

$$\mathcal{L}(\theta,\phi;x) = \mathbb{E}_{q_\phi(z|x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z|x)}\right] = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]}_{\text{重建项}} - \underbrace{D_{KL}(q_\phi(z|x)\|p(z))}_{\text{KL正则项}}$$

**重参数化**：$z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon$，$\epsilon\sim\mathcal{N}(0,I)$

---

## A.4 扩散模型核心公式

### 正向过程

$$q(x_t|x_0) = \mathcal{N}(x_t;\ \sqrt{\bar\alpha_t}x_0,\ (1-\bar\alpha_t)I)$$

$$x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\quad\epsilon\sim\mathcal{N}(0,I)$$

$$\bar\alpha_t = \prod_{s=1}^t\alpha_s = \prod_{s=1}^t(1-\beta_s)$$

### 后验分布

$$q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1};\ \tilde\mu_t(x_t,x_0),\ \tilde\beta_t I)$$

$$\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$

$$\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$

### 训练目标

$$\mathcal{L}_{simple} = \mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\ t)\|^2\right]$$

### 等价预测参数化

| 预测目标 | 公式 |
|----------|------|
| $\epsilon$-预测 | $\epsilon_\theta(x_t, t)$ |
| $x_0$-预测 | $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta)/\sqrt{\bar\alpha_t}$ |
| $v$-预测 | $v_\theta = \sqrt{\bar\alpha_t}\epsilon - \sqrt{1-\bar\alpha_t}x_0$ |
| 分数函数 | $s_\theta(x_t,t) = -\epsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$ |

### 信噪比

$$\text{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t}$$

---

## A.5 采样算法

### DDPM采样

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right) + \sqrt{\beta_t}\mathbf{z},\quad\mathbf{z}\sim\mathcal{N}(0,I)$$

### DDIM采样（$\sigma_t=0$）

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\underbrace{\left(\frac{x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta}{\sqrt{\bar\alpha_t}}\right)}_{\hat x_0} + \sqrt{1-\bar\alpha_{t-1}}\epsilon_\theta(x_t,t)$$

### CFG（无分类器引导）

$$\tilde\epsilon = \epsilon_\theta(x_t,t,\emptyset) + w\cdot[\epsilon_\theta(x_t,t,y) - \epsilon_\theta(x_t,t,\emptyset)]$$

等价形式：$\tilde\epsilon = (1+w)\epsilon_\theta(x_t,t,y) - w\cdot\epsilon_\theta(x_t,t,\emptyset)$

---

## A.6 SDE框架

### 正向SDE（VP-SDE）

$$d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}\,dt + \sqrt{\beta(t)}\,dW_t$$

### Anderson 1982逆向SDE

$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x},t) - g(t)^2\nabla_\mathbf{x}\log p_t(\mathbf{x})\right]dt + g(t)d\bar W_t$$

### 概率流ODE（DDIM对应）

$$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x},t) - \frac{1}{2}g(t)^2\nabla_\mathbf{x}\log p_t(\mathbf{x})$$

### Fokker-Planck方程

$$\frac{\partial p_t}{\partial t} = -\nabla\cdot(fp_t) + \frac{g^2}{2}\nabla^2 p_t$$

---

## A.7 流匹配

### Rectified Flow插值

$$x_t = (1-t)x_0 + t\epsilon,\quad\epsilon\sim\mathcal{N}(0,I),\quad t\in[0,1]$$

### 速度场目标

$$\mathcal{L}_{RF} = \mathbb{E}_{t,x_0,\epsilon}\left[\|v_\theta(x_t,t) - (\epsilon - x_0)\|^2\right]$$

---

## A.8 注意力机制

### 缩放点积注意力

$$\text{Attn}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Q、K、V的来源**：

| 类型 | Q来源 | K/V来源 |
|------|-------|---------|
| 自注意力 | 图像特征 | 图像特征 |
| 交叉注意力 | 图像特征 | 文本特征 |

### AdaLN（扩散变换器）

$$\text{AdaLN}(x, t, y) = (1 + \gamma(t,y))\cdot\text{LayerNorm}(x) + \delta(t,y)$$

---

## A.9 常用不等式与恒等式

| 不等式/恒等式 | 条件 |
|--------------|------|
| $\mathbb{E}[f(X)] \geq f(\mathbb{E}[X])$ | $f$为凸函数（Jensen） |
| $D_{KL}(p\|q) \geq 0$，等号当且仅当 $p=q$ | — |
| $\log(x) \leq x - 1$，等号当 $x=1$ | — |
| $(dW_t)^2 = dt$（均方意义下） | 伊藤积分 |
| $\text{tr}(\nabla_x s_\theta) = \sum_i\partial_i[s_\theta]_i$ | Hyvärinen分数匹配 |

---

## A.10 符号表

| 符号 | 含义 |
|------|------|
| $x_0$ | 原始数据 |
| $x_t$ | $t$时刻加噪数据 |
| $\epsilon$ | 高斯噪声 $\mathcal{N}(0,I)$ |
| $\epsilon_\theta$ | 噪声预测网络 |
| $\alpha_t = 1-\beta_t$ | 单步信号保留比例 |
| $\bar\alpha_t = \prod_{s=1}^t\alpha_s$ | 累积信号保留比例 |
| $\beta_t$ | 扩散步长（噪声调度） |
| $T$ | 总扩散步数 |
| $s_\theta$ | 分数函数估计 |
| $p_t(x)$ | $t$时刻的边缘分布 |
| $q(x_t|x_0)$ | 正向过程条件分布 |
| $p_\theta(x_{t-1}|x_t)$ | 逆向过程条件分布 |
| $\phi_\theta$ | 神经网络参数 |
| $W_t$ | 维纳过程（布朗运动） |
| $\text{SNR}(t)$ | 信噪比 $= \bar\alpha_t/(1-\bar\alpha_t)$ |
| $w$ | CFG引导强度 |
| $y$ | 条件（文本、类别等） |
| $z$ | 潜在变量（VAE / LDM） |

---

[返回目录](../README.md)
