# 第二章：高斯分布与马尔科夫链

> **本章导读**：扩散模型的正向过程本质上是一条高斯马尔科夫链。本章深入研究多元高斯分布的条件化、高斯分布的卷积性质，以及马尔科夫链的基本理论，为理解DDPM的加噪过程打下坚实基础。

**前置知识**：第一章内容，矩阵运算（行列式、逆矩阵）
**预计学习时间**：100分钟

---

## 学习目标

完成本章学习后，你将能够：
1. 推导多元高斯分布的条件分布公式，理解其几何意义
2. 证明独立高斯变量之和的分布，理解高斯的卷积性质
3. 计算两个高斯分布之间的KL散度（闭合形式）
4. 理解马尔科夫性质，分析马尔科夫链的平稳分布
5. 将高斯马尔科夫链与DDPM正向过程建立对应关系

---

## 2.1 多元高斯分布的条件化

### 分块高斯分布

设 $\mathbf{x} = \begin{pmatrix}\mathbf{x}_1 \\ \mathbf{x}_2\end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix}\boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2\end{pmatrix}, \begin{pmatrix}\Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}\end{pmatrix}\right)$

**条件分布**（非常重要，在第7章推导后验时用到）：

$$\mathbf{x}_1 | \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \Sigma_{1|2})$$

其中：
$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$
$$\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$$

**推导思路**：
利用 $\mathbf{x}_1 | \mathbf{x}_2 = \mathbf{x}_1 - \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2) + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$，第一项与 $\mathbf{x}_2$ 独立。

### 几何直觉

条件均值 $\boldsymbol{\mu}_{1|2}$ 是先验均值 $\boldsymbol{\mu}_1$ 加上根据观测 $\mathbf{x}_2$ 的修正项。修正量由 $\Sigma_{12}$ 控制：**协方差越大，观测对预测的影响越大**。

条件方差 $\Sigma_{1|2}$ 比先验方差 $\Sigma_{11}$ 小，体现了观测带来的信息增益。

---

## 2.2 高斯分布的卷积性质

### 独立高斯之和

**定理**：若 $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ 与 $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ 独立，则：

$$X + Y \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$$

**证明**（特征函数法）：高斯分布的特征函数为 $\phi_X(t) = \exp(i\mu t - \frac{\sigma^2 t^2}{2})$。

独立变量之和的特征函数等于各自特征函数之积：
$$\phi_{X+Y}(t) = \phi_X(t)\phi_Y(t) = \exp(i(\mu_1+\mu_2)t - \frac{(\sigma_1^2+\sigma_2^2)t^2}{2})$$

这正是 $\mathcal{N}(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$ 的特征函数。$\square$

### 缩放与平移后的求和

更一般地，若 $\epsilon \sim \mathcal{N}(0, I)$，则：

$$ax_0 + b\epsilon \sim \mathcal{N}(ax_0, b^2 I)$$

这是扩散模型正向过程的基础：$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$。

---

## 2.3 KL散度与信息论

### KL散度定义

两个分布 $p, q$ 之间的**KL散度（相对熵）**：

$$D_{KL}(p \| q) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right] = \int p(x) \log \frac{p(x)}{q(x)} dx$$

**性质**：
- $D_{KL}(p\|q) \geq 0$（Gibbs不等式），等号当且仅当 $p = q$ 几乎处处成立
- **不对称**：$D_{KL}(p\|q) \neq D_{KL}(q\|p)$（一般情况）
- 不满足三角不等式，不是严格意义上的距离

### 两个高斯分布的KL散度

设 $p = \mathcal{N}(\mu_1, \Sigma_1)$，$q = \mathcal{N}(\mu_2, \Sigma_2)$，$d$ 维，闭合形式：

$$D_{KL}(p \| q) = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^T \Sigma_2^{-1}(\mu_2-\mu_1)\right]$$

**特殊情况**：$p = \mathcal{N}(\mu, \sigma^2 I)$，$q = \mathcal{N}(0, I)$（VAE的KL项）：

$$D_{KL}(p \| q) = \frac{1}{2}\left(-d\log\sigma^2 - d + d\sigma^2 + \|\mu\|^2\right) = \frac{1}{2}\sum_{j=1}^d\left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)$$

> **在扩散模型中**：DDPM的训练目标包含 $D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))$，两个高斯的KL散度有闭合形式。

---

## 2.4 马尔科夫链基础

### 马尔科夫性质

随机过程 $\{X_t\}_{t \geq 0}$ 具有**马尔科夫性质**（无记忆性）：

$$P(X_{t+1} = x_{t+1} | X_t = x_t, X_{t-1} = x_{t-1}, ..., X_0 = x_0) = P(X_{t+1} = x_{t+1} | X_t = x_t)$$

即：**未来只依赖当前状态，与历史无关**。

### 转移核与平稳分布

**转移核**（连续状态）：$T(x'|x) = P(X_{t+1} \in dx' | X_t = x)$

**平稳分布** $\pi$：满足 $\pi(x') = \int T(x'|x)\pi(x)dx$

**细致平衡条件**（充分条件）：$\pi(x)T(x'|x) = \pi(x')T(x|x')$

满足细致平衡的马尔科夫链是**可逆的**，MCMC算法（如Metropolis-Hastings）利用这一性质构造具有目标分布为平稳分布的马尔科夫链。

---

## 2.5 高斯马尔科夫链

### 线性高斯系统

考虑如下**高斯马尔科夫链**：

$$x_t = \sqrt{\alpha} x_{t-1} + \sqrt{1-\alpha} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

其中 $0 < \alpha < 1$。这正是DDPM正向过程（令 $\alpha = \alpha_t = 1 - \beta_t$）。

**边缘分布推导**：
- $t=1$：$x_1 \sim \mathcal{N}(\sqrt{\alpha}x_0, (1-\alpha)I)$
- $t=2$：$x_2 = \sqrt{\alpha}x_1 + \sqrt{1-\alpha}\epsilon_2$，代入 $x_1$：
  $$x_2 = \alpha x_0 + \sqrt{\alpha(1-\alpha)}\epsilon_1 + \sqrt{1-\alpha}\epsilon_2$$
  后两项独立高斯之和，方差 $= \alpha(1-\alpha) + (1-\alpha) = (1-\alpha^2)$
  $$x_2 \sim \mathcal{N}(\alpha x_0, (1-\alpha^2)I)$$

**一般形式**（归纳得）：

$$x_t \sim \mathcal{N}(\alpha^{t/2} x_0, (1-\alpha^t)I)$$

### DDPM正向过程（非均匀调度）

当每步调度 $\alpha_t = 1 - \beta_t$ 不同时，令 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$：

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

**重参数化形式**：$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，$\epsilon \sim \mathcal{N}(0,I)$

当 $t \to T$ 且 $\bar{\alpha}_T \approx 0$：$x_T \approx \mathcal{N}(0, I)$（数据信息被完全销毁）。

---

## 2.6 扩散过程预览

### 信噪比（SNR）

定义信号强度与噪声强度之比：

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

- $t=0$：$\text{SNR} = \infty$（纯信号）
- $t=T$：$\text{SNR} \approx 0$（纯噪声）

SNR随时间单调递减，不同的噪声调度（线性、余弦等）对应不同的SNR衰减曲线，影响训练时各时间步的学习难度分布。

### 从马尔科夫链到SDE

当 $T \to \infty$，$\beta_t \to 0$，离散马尔科夫链趋近于连续时间SDE（随机微分方程）：

$$dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dW_t$$

这是VP-SDE（Variance Preserving SDE），第6章将详细讨论。

---

## 代码实战

```python
"""
第二章代码实战：高斯分布与马尔科夫链
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# ============================================================
# 1. 条件高斯分布
# ============================================================

def conditional_gaussian(mu1: torch.Tensor, mu2: torch.Tensor,
                          Sigma11: torch.Tensor, Sigma12: torch.Tensor,
                          Sigma22: torch.Tensor,
                          x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算多元高斯的条件分布参数
    
    Args:
        mu1, mu2: 分块均值向量
        Sigma11, Sigma12, Sigma22: 分块协方差矩阵
        x2: 条件化的观测值
    
    Returns:
        (mu_cond, Sigma_cond): 条件均值和协方差
    """
    Sigma22_inv = torch.linalg.inv(Sigma22)
    mu_cond = mu1 + Sigma12 @ Sigma22_inv @ (x2 - mu2)
    Sigma_cond = Sigma11 - Sigma12 @ Sigma22_inv @ Sigma12.T
    return mu_cond, Sigma_cond


# ============================================================
# 2. KL散度（两个高斯）
# ============================================================

def kl_gaussian(mu1: torch.Tensor, sigma1: torch.Tensor,
                mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """
    计算 KL(N(mu1, diag(sigma1^2)) || N(mu2, diag(sigma2^2)))
    
    Args:
        mu1, mu2: 均值向量，shape: (d,)
        sigma1, sigma2: 标准差向量（对角协方差），shape: (d,)
    
    Returns:
        KL散度（标量）
    """
    d = mu1.shape[0]
    kl = 0.5 * (
        2 * torch.log(sigma2 / sigma1).sum()
        - d
        + (sigma1**2 / sigma2**2).sum()
        + ((mu2 - mu1)**2 / sigma2**2).sum()
    )
    return kl


def kl_to_standard_normal(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, diag(sigma^2)) || N(0, I)) — VAE中常用"""
    return 0.5 * (mu**2 + sigma**2 - 2*torch.log(sigma) - 1).sum()


# ============================================================
# 3. 高斯马尔科夫链模拟（扩散过程）
# ============================================================

class GaussianMarkovChain:
    """
    模拟线性高斯马尔科夫链：x_t = sqrt(alpha)*x_{t-1} + sqrt(1-alpha)*eps
    这是DDPM正向过程的简化版本（均匀调度）
    """
    
    def __init__(self, alpha: float, T: int):
        """
        Args:
            alpha: 每步信号保留比例（0 < alpha < 1）
            T: 总步数
        """
        self.alpha = alpha
        self.T = T
        # 计算各时间步的累积 alpha_bar
        self.alpha_bar = torch.tensor([alpha**t for t in range(T+1)])
    
    def q_sample(self, x0: torch.Tensor, t: int) -> torch.Tensor:
        """
        直接从 q(x_t|x_0) 采样（闭合形式，无需迭代）
        
        Args:
            x0: 初始状态，shape: (d,) or (B, d)
            t: 时间步
        
        Returns:
            x_t: 第t步状态
        """
        alpha_bar_t = self.alpha_bar[t]
        eps = torch.randn_like(x0)
        # shape: same as x0
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps
    
    def simulate(self, x0: torch.Tensor) -> list:
        """
        逐步模拟马尔科夫链
        
        Returns:
            trajectory: 轨迹列表 [x_0, x_1, ..., x_T]
        """
        trajectory = [x0.clone()]
        x = x0.clone()
        for t in range(1, self.T + 1):
            eps = torch.randn_like(x)
            x = torch.sqrt(torch.tensor(self.alpha)) * x + torch.sqrt(torch.tensor(1 - self.alpha)) * eps
            trajectory.append(x.clone())
        return trajectory
    
    def snr(self, t: int) -> float:
        """计算时间步t的信噪比"""
        ab = self.alpha_bar[t].item()
        return ab / (1 - ab) if ab < 1 else float('inf')


def visualize_markov_chain():
    """可视化高斯马尔科夫链的演化过程"""
    torch.manual_seed(42)
    
    chain = GaussianMarkovChain(alpha=0.98, T=100)
    
    # 1D信号演示
    x0 = torch.tensor([3.0])
    trajectory = chain.simulate(x0)
    trajectory_tensor = torch.stack(trajectory).squeeze()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 轨迹图
    axes[0].plot(trajectory_tensor.numpy())
    axes[0].axhline(0, color='r', linestyle='--', alpha=0.5, label='均值(最终)')
    axes[0].set_xlabel('时间步 t')
    axes[0].set_ylabel('x_t')
    axes[0].set_title('高斯马尔科夫链轨迹')
    axes[0].legend()
    
    # 闭合形式验证
    t_steps = [0, 20, 50, 80, 100]
    x0_many = torch.tensor(3.0).expand(1000)
    for t in t_steps:
        samples = chain.q_sample(x0_many, t)
        mean = samples.mean().item()
        std = samples.std().item()
        expected_mean = (chain.alpha_bar[t]**0.5 * 3.0).item()
        expected_std = (1 - chain.alpha_bar[t]).sqrt().item()
        print(f"t={t:3d}: mean={mean:.4f} (理论:{expected_mean:.4f}), std={std:.4f} (理论:{expected_std:.4f})")
    
    # SNR曲线
    snr_values = [chain.snr(t) for t in range(chain.T + 1)]
    axes[2].semilogy(snr_values)
    axes[2].set_xlabel('时间步 t')
    axes[2].set_ylabel('SNR (对数尺度)')
    axes[2].set_title('信噪比随时间的衰减')
    
    plt.tight_layout()
    plt.savefig('markov_chain.png', dpi=100, bbox_inches='tight')


if __name__ == "__main__":
    print("=" * 50)
    print("演示1：条件高斯分布")
    mu1 = torch.tensor([0.0, 0.0])
    mu2 = torch.tensor([0.0])
    Sigma11 = torch.tensor([[1.0, 0.8], [0.8, 2.0]])
    Sigma12 = torch.tensor([[0.5], [0.5]])
    Sigma22 = torch.tensor([[1.0]])
    x2 = torch.tensor([1.0])
    mu_c, Sigma_c = conditional_gaussian(mu1, mu2, Sigma11, Sigma12, Sigma22, x2)
    print(f"条件均值: {mu_c}")
    print(f"条件协方差:\n{Sigma_c}")
    
    print("\n演示2：KL散度计算")
    mu = torch.tensor([1.0, 2.0])
    sigma = torch.tensor([0.5, 0.8])
    kl = kl_to_standard_normal(mu, sigma)
    print(f"KL(N(mu,sigma²)||N(0,I)) = {kl.item():.4f}")
    
    print("\n演示3：高斯马尔科夫链")
    visualize_markov_chain()
```

---

## 本章小结

| 概念 | 公式 | 扩散模型应用 |
|------|------|-------------|
| 条件高斯均值 | $\mu_{1\|2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2)$ | 推导 $q(x_{t-1}\|x_t,x_0)$ |
| 条件高斯方差 | $\Sigma_{1\|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$ | 后验方差 $\tilde{\beta}_t$ |
| 高斯卷积 | $\mathcal{N}(\mu_1,\sigma_1^2)+\mathcal{N}(\mu_2,\sigma_2^2)=\mathcal{N}(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$ | 正向过程闭合形式 |
| 高斯KL散度 | $\frac{1}{2}[\log\frac{\|\Sigma_2\|}{\|\Sigma_1\|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + \Delta\mu^T\Sigma_2^{-1}\Delta\mu]$ | DDPM训练目标 |
| 马尔科夫链 | $q(x_t\|x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$ | 正向过程直接采样 |

---

## 练习题

### 基础题

**2.1** 设 $X \sim \mathcal{N}(\mu, \sigma^2)$，$Y = aX + bZ$，其中 $Z \sim \mathcal{N}(0, 1)$ 与 $X$ 独立，$a, b \in \mathbb{R}$。求 $Y$ 的分布。

**2.2** 计算 $D_{KL}(\mathcal{N}(1, 1) \| \mathcal{N}(0, 1))$ 的精确值。

### 中级题

**2.3** 实现 `GaussianMarkovChain`，并对一个二维数据点 $x_0 = (3, 4)^T$ 运行100步马尔科夫链。绘制轨迹，并验证：在每个时间步 $t$，模拟结果的均值和方差与闭合公式 $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}^t}x_0, (1-\alpha^t)I)$ 的预测一致（$\alpha = 0.98$）。

**2.4** 证明：若 $p = \mathcal{N}(\mu, \sigma^2 I)$ 且 $q = \mathcal{N}(0, I)$，则：
$$D_{KL}(p \| q) = \frac{1}{2}(\|\mu\|^2 + d\sigma^2 - d\log\sigma^2 - d)$$

### 提高题

**2.5** 在DDPM中，正向过程的后验分布 $q(x_{t-1}|x_t, x_0)$ 是高斯分布。利用本章的条件高斯公式，推导其均值 $\tilde{\mu}_t$ 和方差 $\tilde{\beta}_t$：

$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t$$
$$\tilde{\beta}_t = \frac{(1-\bar{\alpha}_{t-1})\beta_t}{1-\bar{\alpha}_t}$$

（提示：$q(x_{t-1}|x_t, x_0) \propto q(x_t|x_{t-1})q(x_{t-1}|x_0)$，均为高斯，相乘后配方。）

---

## 练习答案

**2.1** $Y = aX + bZ \sim \mathcal{N}(a\mu, a^2\sigma^2 + b^2)$（高斯的线性变换+独立高斯之和）。

**2.2** 代入公式：$D_{KL}(\mathcal{N}(1,1)\|\mathcal{N}(0,1)) = \frac{1}{2}(1^2 + 1 - \log 1 - 1) = \frac{1}{2}$。

**2.3** 见代码实战中 `visualize_markov_chain` 函数。

**2.4** 代入对角协方差的KL公式：$\frac{1}{2}[\log(1/\sigma^{2d}) - d + d\sigma^2 + \|\mu\|^2] = \frac{1}{2}(\|\mu\|^2 + d\sigma^2 - d\log\sigma^2 - d)$。

**2.5** 配方法：
- $q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}x_{t-1}, \beta_t I)$，$q(x_{t-1}|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})I)$
- $\log q(x_{t-1}|x_t,x_0) = \log q(x_t|x_{t-1}) + \log q(x_{t-1}|x_0) + \text{const}$
- 展开并关于 $x_{t-1}$ 配方，得到均值和方差如题目所示。

---

## 延伸阅读

1. **Anderson (1958)**. *An Introduction to Multivariate Statistical Analysis* — 多元高斯的经典参考
2. **Norris (1997)**. *Markov Chains* — 马尔科夫链的严格处理
3. **Song et al. (2020)**. *Score-Based Generative Modeling through SDEs* — 连接马尔科夫链与SDE

---

[← 上一章：概率论基础与随机变量](./01-probability-random-variables.md)

[下一章：变分推断基础 →](./03-variational-inference.md)

[返回目录](../README.md)
