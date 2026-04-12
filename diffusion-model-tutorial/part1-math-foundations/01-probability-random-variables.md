# 第一章：概率论基础与随机变量

> **本章导读**：扩散模型的核心是在概率分布之间进行转换。理解随机变量、概率分布和重参数化技巧，是掌握DDPM的数学基础。本章从概率空间出发，逐步建立扩散模型所需的全部概率论工具。

**前置知识**：微积分基础（积分、偏导数），线性代数（矩阵运算）
**预计学习时间**：90分钟

---

## 学习目标

完成本章学习后，你将能够：
1. 理解概率空间的三要素，掌握条件概率和贝叶斯定理
2. 区分离散和连续随机变量，计算期望、方差和协方差
3. 推导多元高斯分布的PDF，理解边缘化和条件化
4. 理解指数族分布的统一形式，识别常见分布的指数族表达
5. 掌握重参数化技巧，理解其在扩散模型梯度计算中的作用

---

## 1.1 概率空间与基本概念

### 概率空间的三要素

一个概率空间由三元组 $(\Omega, \mathcal{F}, P)$ 定义：

- **样本空间** $\Omega$：所有可能结果的集合。例如掷骰子：$\Omega = \{1,2,3,4,5,6\}$
- **事件域** $\mathcal{F}$：$\Omega$ 的子集构成的 $\sigma$-代数，代表所有"可测"事件
- **概率测度** $P: \mathcal{F} \to [0,1]$：满足Kolmogorov公理：
  1. $P(\Omega) = 1$（规范化）
  2. $P(A) \geq 0$（非负性）
  3. 互斥事件的可加性：$P\left(\bigcup_i A_i\right) = \sum_i P(A_i)$

### 条件概率与独立性

**条件概率**：在事件 $B$ 已发生的条件下，事件 $A$ 发生的概率：

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**全概率公式**：设 $\{B_1, ..., B_n\}$ 为样本空间的一个划分：

$$P(A) = \sum_{i=1}^n P(A|B_i) P(B_i)$$

**贝叶斯定理**：

$$P(B_i|A) = \frac{P(A|B_i)P(B_i)}{\sum_j P(A|B_j)P(B_j)}$$

> **在扩散模型中**：贝叶斯定理是推导逆向过程 $q(x_{t-1}|x_t, x_0)$ 的核心工具（见第7章）。

**独立性**：若 $P(A \cap B) = P(A)P(B)$，则 $A$ 与 $B$ 独立。

---

## 1.2 随机变量与分布

### 随机变量

**随机变量** $X: \Omega \to \mathbb{R}$ 是从样本空间到实数的可测函数。

**概率质量函数（PMF）**（离散）：$p_X(x) = P(X = x)$

**概率密度函数（PDF）**（连续）：$f_X(x)$ 满足 $P(a \leq X \leq b) = \int_a^b f_X(x)dx$

**累积分布函数（CDF）**：$F_X(x) = P(X \leq x)$

### 期望与方差

$$\mathbb{E}[X] = \int_{-\infty}^{+\infty} x f_X(x) dx$$

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**期望的线性性**（极为重要）：$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$

**LOTUS法则**（无需先求 $Y=g(X)$ 的分布）：

$$\mathbb{E}[g(X)] = \int g(x) f_X(x) dx$$

### 重要分布

**一维高斯分布** $\mathcal{N}(\mu, \sigma^2)$：

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

性质：$\mathbb{E}[X] = \mu$，$\text{Var}(X) = \sigma^2$，$X$ 的线性变换仍为高斯。

---

## 1.3 多元随机变量

### 联合分布与边缘分布

对于随机向量 $\mathbf{X} = (X_1, ..., X_d)^T$，联合PDF为 $f(\mathbf{x})$。

**边缘化**：

$$f_{X_1}(x_1) = \int f(x_1, x_2) dx_2$$

**条件分布**：

$$f(x_1 | x_2) = \frac{f(x_1, x_2)}{f_{X_2}(x_2)}$$

### 协方差与协方差矩阵

**协方差**：$\text{Cov}(X_i, X_j) = \mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)]$

**协方差矩阵** $\Sigma \in \mathbb{R}^{d \times d}$：

$$\Sigma_{ij} = \text{Cov}(X_i, X_j)$$

$\Sigma$ 是对称半正定矩阵（$\Sigma = \Sigma^T$，$\mathbf{v}^T \Sigma \mathbf{v} \geq 0$）。

### 多元高斯分布

$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ 的PDF：

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

**关键性质**：
1. 线性变换：若 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$，则 $A\mathbf{X} + \mathbf{b} \sim \mathcal{N}(A\boldsymbol{\mu}+\mathbf{b}, A\Sigma A^T)$
2. 独立高斯之和：$\mathcal{N}(\mu_1, \sigma_1^2) + \mathcal{N}(\mu_2, \sigma_2^2) = \mathcal{N}(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$（扩散过程的关键！）
3. 边缘分布仍为高斯
4. 条件分布仍为高斯

---

## 1.4 指数族分布

许多常见分布都属于**指数族**，具有统一形式：

$$p(x|\eta) = h(x) \exp(\eta^T T(x) - A(\eta))$$

其中：
- $\eta$：自然参数（natural parameters）
- $T(x)$：充分统计量（sufficient statistics）
- $A(\eta)$：对数配分函数（log-partition function），保证归一化
- $h(x)$：基础测度

**高斯分布的指数族形式**：

$$p(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(\frac{\mu}{\sigma^2}x - \frac{1}{2\sigma^2}x^2 - \frac{\mu^2}{2\sigma^2} - \log\sigma\right)$$

其中 $\eta = (\frac{\mu}{\sigma^2}, -\frac{1}{2\sigma^2})^T$，$T(x) = (x, x^2)^T$。

> **为什么重要**：扩散模型的逆向过程 $p_\theta(x_{t-1}|x_t)$ 被参数化为高斯分布，理解高斯的性质直接影响对模型的理解。

---

## 1.5 变量变换与重参数化技巧

### 变量变换公式

设 $Y = g(X)$，$g$ 是单调可微函数，则：

$$f_Y(y) = f_X(g^{-1}(y)) \cdot \left|\frac{d}{dy}g^{-1}(y)\right|$$

多维情形（Jacobian）：设 $\mathbf{Y} = g(\mathbf{X})$：

$$f_{\mathbf{Y}}(\mathbf{y}) = f_{\mathbf{X}}(g^{-1}(\mathbf{y})) \cdot |\det J_{g^{-1}}(\mathbf{y})|$$

### 重参数化技巧（Reparameterization Trick）

**问题**：如何对 $\mathbb{E}_{x \sim q_\phi(x)}[f(x)]$ 关于 $\phi$ 求梯度？

直接对期望中的分布求梯度在计算上非常困难（需要REINFORCE等高方差估计）。

**重参数化**：将随机性分离出来。若 $x \sim \mathcal{N}(\mu_\phi, \sigma_\phi^2)$，则等价于：

$$x = \mu_\phi + \sigma_\phi \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

现在 $\nabla_\phi \mathbb{E}_\epsilon[f(\mu_\phi + \sigma_\phi \epsilon)] = \mathbb{E}_\epsilon[\nabla_\phi f(\mu_\phi + \sigma_\phi \epsilon)]$，梯度可以直接穿透！

> **在扩散模型中**：正向过程 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ 正是重参数化技巧的核心应用。

---

## 1.6 贝叶斯推断

### 贝叶斯框架

$$\underbrace{p(\theta|x)}_{\text{后验}} = \frac{\underbrace{p(x|\theta)}_{\text{似然}} \cdot \underbrace{p(\theta)}_{\text{先验}}}{\underbrace{p(x)}_{\text{证据}}}$$

**边缘似然（模型证据）**：

$$p(x) = \int p(x|\theta) p(\theta) d\theta$$

这个积分通常难以计算，这就是变分推断（第3章）的动机。

### 共轭先验

若先验和后验属于同一分布族，则称先验为**共轭先验**。

**高斯-高斯共轭**：若 $x|\theta \sim \mathcal{N}(\theta, \sigma^2)$，$\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)$，则：

$$\theta|x \sim \mathcal{N}\left(\frac{\sigma^2 \mu_0 + \sigma_0^2 x}{\sigma^2 + \sigma_0^2}, \frac{\sigma^2 \sigma_0^2}{\sigma^2 + \sigma_0^2}\right)$$

---

## 代码实战

```python
"""
第一章代码实战：概率论基础与重参数化
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# ============================================================
# 1. 常见分布采样
# ============================================================

def sample_distributions():
    """演示常见分布的采样和PDF"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 标准正态分布
    samples = torch.randn(10000)
    axes[0, 0].hist(samples.numpy(), bins=100, density=True, alpha=0.7)
    x = torch.linspace(-4, 4, 200)
    pdf = torch.exp(-x**2 / 2) / (2 * np.pi)**0.5
    axes[0, 0].plot(x.numpy(), pdf.numpy(), 'r-', lw=2)
    axes[0, 0].set_title('标准正态分布 N(0,1)')
    
    # 多元高斯分布
    mu = torch.tensor([1.0, 2.0])
    sigma = torch.tensor([[1.0, 0.8], [0.8, 2.0]])
    dist = torch.distributions.MultivariateNormal(mu, sigma)
    samples_2d = dist.sample((1000,))
    axes[0, 1].scatter(samples_2d[:, 0], samples_2d[:, 1], alpha=0.3, s=5)
    axes[0, 1].set_title('多元高斯分布 N(μ, Σ)')
    
    plt.tight_layout()
    plt.savefig('distributions.png', dpi=100, bbox_inches='tight')
    print("图像已保存为 distributions.png")


# ============================================================
# 2. 重参数化技巧演示
# ============================================================

class GaussianSampler(nn.Module):
    """演示重参数化技巧的梯度流动"""
    
    def __init__(self):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
    
    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)
    
    def sample_naive(self, n: int) -> torch.Tensor:
        """朴素采样：梯度无法回传！"""
        return torch.normal(self.mu.detach(), self.sigma.detach(), size=(n,))
    
    def sample_reparam(self, n: int) -> torch.Tensor:
        """重参数化采样：梯度可以回传"""
        epsilon = torch.randn(n)          # 随机性分离到 epsilon
        return self.mu + self.sigma * epsilon  # 梯度通过 mu 和 sigma 回传
    
    def forward(self, n: int) -> torch.Tensor:
        return self.sample_reparam(n)


def demo_reparameterization():
    """演示重参数化技巧的梯度效果"""
    sampler = GaussianSampler()
    optimizer = torch.optim.Adam(sampler.parameters(), lr=0.1)
    
    target_mu = 3.0
    losses = []
    
    for step in range(200):
        samples = sampler(100)
        # 目标：样本均值接近 target_mu
        loss = (samples.mean() - target_mu) ** 2
        
        optimizer.zero_grad()
        loss.backward()  # 重参数化使得梯度可以回传
        optimizer.step()
        losses.append(loss.item())
    
    print(f"训练后 mu = {sampler.mu.item():.4f} (目标: {target_mu})")
    print(f"训练后 sigma = {sampler.sigma.item():.4f}")
    return losses


# ============================================================
# 3. 高斯分布的线性变换（扩散模型预览）
# ============================================================

def demo_gaussian_sum():
    """
    演示独立高斯之和仍为高斯：
    X ~ N(μ₁, σ₁²), Y ~ N(μ₂, σ₂²), X+Y ~ N(μ₁+μ₂, σ₁²+σ₂²)
    
    这是扩散模型正向过程的数学基础！
    """
    torch.manual_seed(42)
    
    mu1, sigma1 = 1.0, 0.5
    mu2, sigma2 = 2.0, 0.8
    
    X = torch.normal(mu1, sigma1, size=(100000,))
    Y = torch.normal(mu2, sigma2, size=(100000,))
    Z = X + Y
    
    print(f"X: mean={X.mean():.4f} (理论:{mu1}), std={X.std():.4f} (理论:{sigma1})")
    print(f"Y: mean={Y.mean():.4f} (理论:{mu2}), std={Y.std():.4f} (理论:{sigma2})")
    print(f"Z=X+Y: mean={Z.mean():.4f} (理论:{mu1+mu2}), std={Z.std():.4f} (理论:{(sigma1**2+sigma2**2)**0.5:.4f})")


# ============================================================
# 4. 贝叶斯更新演示
# ============================================================

def bayesian_update(prior_mu: float, prior_sigma: float,
                    likelihood_sigma: float,
                    observations: torch.Tensor) -> Tuple[float, float]:
    """
    高斯-高斯贝叶斯更新（共轭先验）
    
    Args:
        prior_mu: 先验均值
        prior_sigma: 先验标准差
        likelihood_sigma: 似然标准差（已知）
        observations: 观测值
    
    Returns:
        (posterior_mu, posterior_sigma): 后验参数
    """
    n = len(observations)
    x_bar = observations.mean().item()
    
    # 精度（方差的倒数）
    prior_precision = 1 / prior_sigma**2
    likelihood_precision = n / likelihood_sigma**2
    
    posterior_precision = prior_precision + likelihood_precision
    posterior_sigma = 1 / posterior_precision**0.5
    posterior_mu = (prior_precision * prior_mu + likelihood_precision * x_bar) / posterior_precision
    
    return posterior_mu, posterior_sigma


if __name__ == "__main__":
    print("=" * 50)
    print("演示1：重参数化技巧")
    losses = demo_reparameterization()
    print(f"最终损失: {losses[-1]:.6f}")
    
    print("\n演示2：高斯之和性质")
    demo_gaussian_sum()
    
    print("\n演示3：贝叶斯更新")
    true_mu = 5.0
    observations = torch.normal(true_mu, 1.0, size=(20,))
    post_mu, post_sigma = bayesian_update(0.0, 10.0, 1.0, observations)
    print(f"先验: N(0, 100), 观测均值: {observations.mean():.4f}")
    print(f"后验: N({post_mu:.4f}, {post_sigma**2:.4f})")
```

---

## 本章小结

| 概念 | 关键公式 | 在扩散模型中的作用 |
|------|----------|-------------------|
| 条件概率 | $P(A\|B) = P(A\cap B)/P(B)$ | 正向/逆向过程的条件分布 |
| 贝叶斯定理 | $p(\theta\|x) \propto p(x\|\theta)p(\theta)$ | 推导 $q(x_{t-1}\|x_t,x_0)$ |
| 高斯PDF | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}$ | 逆向过程的参数化形式 |
| 高斯之和 | $\mathcal{N}(\mu_1,\sigma_1^2)+\mathcal{N}(\mu_2,\sigma_2^2)=\mathcal{N}(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$ | 正向加噪闭合形式 |
| 重参数化 | $x = \mu + \sigma\epsilon, \epsilon\sim\mathcal{N}(0,I)$ | 扩散训练中梯度的计算 |

---

## 练习题

### 基础题

**1.1** 设 $X \sim \mathcal{N}(2, 4)$（即均值2，方差4），计算：
   - (a) $P(X > 4)$
   - (b) $P(0 < X < 4)$
   - (c) $\mathbb{E}[X^2]$

**1.2** 证明：若 $X \sim \mathcal{N}(\mu, \sigma^2)$，则 $aX + b \sim \mathcal{N}(a\mu + b, a^2\sigma^2)$。

### 中级题

**1.3** 实现一个函数，给定两个独立高斯分布的参数 $(\mu_1, \sigma_1^2)$ 和 $(\mu_2, \sigma_2^2)$，验证它们的和服从 $\mathcal{N}(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$（用Monte Carlo方法，并与理论值比较）。

**1.4** 在重参数化技巧中，解释为什么 $\mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}[f(\mu + \sigma\epsilon)]$ 关于 $\mu$ 的梯度等于 $\mathbb{E}_\epsilon[\nabla_\mu f(\mu + \sigma\epsilon)]$。这个交换（期望和梯度的顺序）在什么条件下成立？

### 提高题

**1.5** 扩散模型的正向过程定义为：
$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$
利用高斯分布的线性变换性质，证明：
$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$
其中 $\alpha_t = 1-\beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。
（提示：用归纳法，利用 $x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{\beta_t}\epsilon_t$ 的重参数化形式。）

---

## 练习答案

**1.1** 设 $Z = (X-2)/2 \sim \mathcal{N}(0,1)$：
- (a) $P(X>4) = P(Z>1) = 1 - \Phi(1) \approx 0.1587$
- (b) $P(0<X<4) = P(-1<Z<1) = 2\Phi(1)-1 \approx 0.6827$
- (c) $\mathbb{E}[X^2] = \text{Var}(X) + (\mathbb{E}[X])^2 = 4 + 4 = 8$

**1.2** 设 $Y = aX + b$。由期望线性性：$\mathbb{E}[Y] = a\mu + b$。方差：$\text{Var}(Y) = a^2\text{Var}(X) = a^2\sigma^2$。高斯分布的线性变换性质（特征函数可证）保证 $Y$ 仍为高斯。

**1.3** 代码验证略（见代码实战中 `demo_gaussian_sum`）。

**1.4** 交换成立的条件：$f$ 关于 $\mu$ 可微，且满足控制收敛定理的条件（例如梯度有界或满足某种可积条件）。直觉上：$\mu$ 通过 $x = \mu + \sigma\epsilon$ 影响 $f$，梯度 $\nabla_\mu f = \nabla_x f \cdot 1 = \nabla_x f$，因此 $\nabla_\mu \mathbb{E}_\epsilon[f] = \mathbb{E}_\epsilon[\nabla_\mu f]$。

**1.5** 归纳证明：
- 基础：$q(x_1|x_0) = \mathcal{N}(\sqrt{\alpha_1}x_0, \beta_1 I) = \mathcal{N}(\sqrt{\bar{\alpha}_1}x_0, (1-\bar{\alpha}_1)I)$ ✓
- 归纳步：假设 $q(x_{t-1}|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})I)$
- 则 $x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{\beta_t}\epsilon_t$，其中 $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon'$
- 代入：$x_t = \sqrt{\alpha_t\bar{\alpha}_{t-1}}x_0 + \sqrt{\alpha_t(1-\bar{\alpha}_{t-1})}\epsilon' + \sqrt{\beta_t}\epsilon_t$
- 后两项为独立高斯：方差 $= \alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t = \alpha_t - \alpha_t\bar{\alpha}_{t-1} + \beta_t = 1 - \bar{\alpha}_t$ ✓

---

## 延伸阅读

1. **Bishop, C. (2006)**. *Pattern Recognition and Machine Learning*, Chapter 2（概率分布）
2. **Murphy, K. (2022)**. *Probabilistic Machine Learning: An Introduction*, Chapter 2-3
3. **Kingma & Welling (2013)**. *Auto-Encoding Variational Bayes* — 重参数化技巧的原始论文

---

[下一章：高斯分布与马尔科夫链 →](./02-gaussian-markov-chains.md)

[返回目录](../README.md)
