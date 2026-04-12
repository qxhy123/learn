# 第五章：基于分数的生成模型

> **本章导读**：分数匹配（Score Matching）是理解扩散模型的另一个重要视角。本章介绍分数函数、朗之万动力学采样，以及去噪分数匹配——这些概念构成了扩散模型的另一半理论基础，并将在第8章与DDPM统一。

**前置知识**：前四章，梯度计算，基础概率论
**预计学习时间**：100分钟

---

## 学习目标

完成本章学习后，你将能够：
1. 定义分数函数并理解其几何意义
2. 推导朗之万动力学采样算法，理解其收敛性
3. 掌握去噪分数匹配（DSM）的目标函数推导
4. 理解多尺度噪声分数网络（NCSN）的设计原理
5. 证明DDPM的噪声预测等价于加权分数匹配

---

## 5.1 分数函数

### 定义与直觉

对于数据分布 $p_{data}(\mathbf{x})$，定义**分数函数**（Score Function）：

$$s(\mathbf{x}) = \nabla_\mathbf{x} \log p_{data}(\mathbf{x})$$

**几何直觉**：
- $\log p(\mathbf{x})$ 是数据的"势能"（Potential），高概率区域势能高
- 分数是势能的梯度：**指向数据概率增大最快的方向**
- 类比物理：分数是"力场"，将随机游走的粒子引导向高密度区域

**为什么使用分数而非概率**：
- $p(\mathbf{x})$ 需要计算归一化常数 $Z = \int \exp(E(\mathbf{x}))d\mathbf{x}$（通常intractable）
- $\nabla_\mathbf{x} \log p(\mathbf{x}) = \nabla_\mathbf{x} E(\mathbf{x})$：梯度消除了 $Z$！
- 分数可以从能量函数直接计算，无需归一化

### 直接分数匹配

**目标**：训练神经网络 $s_\theta(\mathbf{x}) \approx \nabla_\mathbf{x} \log p_{data}(\mathbf{x})$。

**显式分数匹配（ESM）**：

$$\mathcal{L}_{ESM} = \mathbb{E}_{p_{data}}\left[\|s_\theta(\mathbf{x}) - \nabla_\mathbf{x} \log p_{data}(\mathbf{x})\|^2\right]$$

**问题**：需要知道 $\nabla_\mathbf{x} \log p_{data}$，这正是我们想学的！

---

## 5.2 朗之万动力学采样

### 朗之万方程

朗之万动力学（Langevin Dynamics）通过迭代更新从分布 $p(\mathbf{x})$ 采样：

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \frac{\epsilon}{2} \nabla_\mathbf{x} \log p(\mathbf{x}_k) + \sqrt{\epsilon} \mathbf{z}_k$$

其中 $\mathbf{z}_k \sim \mathcal{N}(0, I)$，$\epsilon$ 是步长。

**理论保证**：当 $\epsilon \to 0$，迭代次数 $K \to \infty$ 时，$\mathbf{x}_K$ 的分布收敛到 $p(\mathbf{x})$（在正则性条件下）。

**直觉**：
- 梯度项 $\frac{\epsilon}{2}\nabla_\mathbf{x}\log p$ 引导粒子向高概率区域移动
- 噪声项 $\sqrt{\epsilon}\mathbf{z}$ 保证遍历性（不会陷在局部极值）

### 与扩散模型的联系

朗之万采样只需分数函数！如果我们能学到好的分数估计 $s_\theta(\mathbf{x}) \approx \nabla_\mathbf{x}\log p(\mathbf{x})$，就可以用朗之万动力学从 $p$ 中采样——这正是分数生成模型的基础。

---

## 5.3 去噪分数匹配（DSM）

### Hyvärinen分数匹配

Hyvärinen (2005) 证明了ESM可以等价转换为无需真实分数的形式：

$$\mathcal{L}_{SM} = \mathbb{E}_{p_{data}}\left[\text{tr}(\nabla_\mathbf{x} s_\theta(\mathbf{x})) + \frac{1}{2}\|s_\theta(\mathbf{x})\|^2\right]$$

**问题**：Jacobian的迹 $\text{tr}(\nabla_\mathbf{x} s_\theta)$ 计算量为 $O(d^2)$，高维数据不可行。

### Vincent (2011) 去噪分数匹配

**关键思想**：在含噪数据上学习分数，更简单且可扩展！

定义含噪分布 $q_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 I)$，则：

$$\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) = \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2}$$

**去噪分数匹配目标**：

$$\mathcal{L}_{DSM} = \mathbb{E}_{p_{data}(\mathbf{x})}\mathbb{E}_{q_\sigma(\tilde{\mathbf{x}}|\mathbf{x})}\left[\left\|s_\theta(\tilde{\mathbf{x}}) - \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2}\right\|^2\right]$$

**等价性定理**（Vincent 2011）：

$$\mathcal{L}_{DSM} = \mathcal{L}_{ESM} + \text{const}$$

即：在含噪数据上的去噪等价于学习原始数据分布的分数！

**直觉**：分数 $\nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}}|\mathbf{x})$ 指向"去噪方向"，也就是 $(\mathbf{x} - \tilde{\mathbf{x}})/\sigma^2$——学习去噪就是学习分数！

---

## 5.4 噪声条件分数网络（NCSN）

### 多尺度噪声

单一噪声水平的问题：
- 噪声太小：分布的低密度区域（不同模式之间）分数估计不准
- 噪声太大：学到的分数对应的是高度模糊的分布

**Song & Ermon (2019)** 提出使用**多尺度噪声** $\{\sigma_1 > \sigma_2 > ... > \sigma_L\}$：

$$\mathcal{L}_{NCSN} = \frac{1}{L}\sum_{i=1}^L \lambda(\sigma_i) \mathbb{E}_{p(\mathbf{x})}\mathbb{E}_{\mathcal{N}(\tilde{\mathbf{x}};\mathbf{x},\sigma_i^2 I)}\left[\|s_\theta(\tilde{\mathbf{x}}, \sigma_i) - \frac{\mathbf{x}-\tilde{\mathbf{x}}}{\sigma_i^2}\|^2\right]$$

其中 $\lambda(\sigma_i) = \sigma_i^2$（使各尺度损失量纲一致）。

### 退火朗之万采样

**采样算法**（从粗到细）：

```
从 x ~ N(0, σ₁²I) 开始
For σ_i from σ₁ (大) to σ_L (小):
    用步长 ε_i = α·σ_i²/σ_L² 运行 T 步朗之万：
    x ← x + ε_i/2 · s_θ(x, σ_i) + √ε_i · z,  z ~ N(0,I)
```

**直觉**：先在大噪声下"粗定位"（确定大致位置），再逐步精细化。这正是扩散模型逆向过程的离散版本！

---

## 5.5 分数匹配与DDPM的统一

### Song et al. (2020) 的统一

**定理**：DDPM的噪声预测等价于加权去噪分数匹配。

**推导**：

DDPM训练目标：$\mathbb{E}_{t, x_0, \epsilon}[||\epsilon - \epsilon_\theta(x_t, t)||^2]$

其中 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，$\epsilon \sim \mathcal{N}(0,I)$。

注意到正向过程的分数：

$$\nabla_{x_t}\log q(x_t|x_0) = \frac{-(x_t - \sqrt{\bar{\alpha}_t}x_0)}{1-\bar{\alpha}_t} = \frac{-\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

因此噪声预测与分数的关系：

$$\epsilon_\theta(x_t, t) \approx \epsilon \implies s_\theta(x_t, t) = \frac{-\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}} \approx \nabla_{x_t}\log q(x_t|x_0)$$

**结论**：学习噪声 $\epsilon$ = 学习分数（差一个负的缩放因子）！

### 统一框架

| 方法 | 训练目标 | 等价分数估计 |
|------|----------|-------------|
| DSM | $\|s_\theta(\tilde{x}) - (x-\tilde{x})/\sigma^2\|^2$ | $\nabla_{\tilde{x}}\log q_\sigma(\tilde{x})$ |
| NCSN | 多尺度DSM | $\nabla_{x_t}\log q_{\sigma_i}(x)$ |
| DDPM | $\|\epsilon - \epsilon_\theta(x_t,t)\|^2$ | $-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$ |

---

## 代码实战

```python
"""
第五章代码实战：分数匹配与朗之万采样
在二维混合高斯上演示分数场可视化
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 二维高斯混合分布
# ============================================================

class GaussianMixture:
    """二维高斯混合分布"""
    
    def __init__(self, means: list, stds: list, weights: list = None):
        self.means = [torch.tensor(m, dtype=torch.float32) for m in means]
        self.stds = stds
        self.n = len(means)
        self.weights = weights or [1.0 / self.n] * self.n
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """对数概率，shape: (N,) -> (N,)"""
        log_probs = []
        for mu, std, w in zip(self.means, self.stds, self.weights):
            dist = torch.distributions.MultivariateNormal(mu, std**2 * torch.eye(2))
            log_probs.append(torch.log(torch.tensor(w)) + dist.log_prob(x))
        return torch.logsumexp(torch.stack(log_probs, dim=0), dim=0)
    
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """真实分数函数（解析计算）"""
        x.requires_grad_(True)
        log_p = self.log_prob(x).sum()
        grad = torch.autograd.grad(log_p, x)[0]
        return grad.detach()
    
    def sample(self, n: int) -> torch.Tensor:
        """采样"""
        samples = []
        counts = torch.multinomial(torch.tensor(self.weights), n, replacement=True)
        for i, (mu, std) in enumerate(zip(self.means, self.stds)):
            n_i = (counts == i).sum().item()
            if n_i > 0:
                s = mu + std * torch.randn(n_i, 2)
                samples.append(s)
        return torch.cat(samples, dim=0)


# ============================================================
# 2. 分数网络（MLP）
# ============================================================

class ScoreNetwork(nn.Module):
    """简单MLP分数估计网络 s_θ(x) ≈ ∇_x log p(x)"""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 3. 去噪分数匹配训练
# ============================================================

def train_dsm(distribution: GaussianMixture, sigma: float = 0.3,
              n_steps: int = 5000) -> ScoreNetwork:
    """
    去噪分数匹配训练
    目标：s_θ(x̃) ≈ (x - x̃) / σ²
    """
    model = ScoreNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for step in range(n_steps):
        x = distribution.sample(256)                     # shape: (256, 2)
        noise = sigma * torch.randn_like(x)              # shape: (256, 2)
        x_tilde = x + noise                              # shape: (256, 2)
        
        # 目标分数：(x - x̃) / σ²
        target_score = (x - x_tilde) / sigma**2         # shape: (256, 2)
        
        # 预测分数
        pred_score = model(x_tilde)                      # shape: (256, 2)
        
        # DSM损失
        loss = ((pred_score - target_score)**2).sum(-1).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 1000 == 0:
            print(f"Step {step+1}, Loss: {loss.item():.4f}")
    
    return model


# ============================================================
# 4. 朗之万采样
# ============================================================

@torch.no_grad()
def langevin_sampling(score_fn, n_samples: int = 500,
                      n_steps: int = 1000, step_size: float = 0.01,
                      init_std: float = 2.0) -> torch.Tensor:
    """
    朗之万动力学采样
    x_{k+1} = x_k + ε/2 * s_θ(x_k) + √ε * z_k
    """
    x = init_std * torch.randn(n_samples, 2)
    
    for k in range(n_steps):
        score = score_fn(x)
        noise = torch.randn_like(x)
        x = x + step_size / 2 * score + step_size**0.5 * noise
    
    return x


def visualize_score_field(model: ScoreNetwork, distribution: GaussianMixture,
                          xlim=(-4, 4), ylim=(-4, 4), grid_size=20):
    """可视化真实分数场和学习到的分数场"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 创建网格
    x_grid = torch.linspace(*xlim, grid_size)
    y_grid = torch.linspace(*ylim, grid_size)
    xx, yy = torch.meshgrid(x_grid, y_grid, indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (grid_size^2, 2)
    
    # 真实分数场
    true_scores = distribution.score(grid.clone())
    
    # 预测分数场
    with torch.no_grad():
        pred_scores = model(grid)
    
    # 可视化
    for ax, scores, title in zip(axes[:2], [true_scores, pred_scores],
                                  ['真实分数场', '预测分数场（DSM）']):
        ax.quiver(grid[:, 0].numpy(), grid[:, 1].numpy(),
                  scores[:, 0].numpy(), scores[:, 1].numpy(),
                  alpha=0.7, scale=30)
        # 叠加真实分布等高线
        real_samples = distribution.sample(500).numpy()
        ax.scatter(*real_samples.T, s=5, alpha=0.3, c='gray')
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    # 朗之万采样结果
    lang_samples = langevin_sampling(model, n_samples=500).numpy()
    axes[2].scatter(*lang_samples.T, s=5, alpha=0.5, c='blue', label='朗之万采样')
    real_samples = distribution.sample(500).numpy()
    axes[2].scatter(*real_samples.T, s=5, alpha=0.3, c='red', label='真实数据')
    axes[2].legend()
    axes[2].set_title('朗之万采样 vs 真实数据')
    
    plt.tight_layout()
    plt.savefig('score_matching.png', dpi=100)
    print("分数场可视化已保存")


if __name__ == "__main__":
    # 创建二维混合高斯
    dist = GaussianMixture(
        means=[[-2, 0], [2, 0], [0, 2]],
        stds=[0.5, 0.5, 0.5]
    )
    
    print("训练去噪分数匹配...")
    model = train_dsm(dist, sigma=0.3, n_steps=3000)
    
    print("\n可视化分数场...")
    visualize_score_field(model, dist)
    
    print("\n分数与噪声预测的关系（DDPM预览）:")
    print("s_θ(x_t, t) = -ε_θ(x_t, t) / √(1 - ᾱ_t)")
    print("预测噪声 ≡ 学习分数（差一个负的缩放因子）")
```

---

## 本章小结

| 概念 | 公式 | 意义 |
|------|------|------|
| 分数函数 | $s(\mathbf{x}) = \nabla_\mathbf{x}\log p(\mathbf{x})$ | 指向概率增大方向，无需归一化常数 |
| 朗之万采样 | $x_{k+1} = x_k + \frac{\epsilon}{2}s(x_k) + \sqrt{\epsilon}z_k$ | 用分数从任意分布采样 |
| 去噪分数匹配 | $\|s_\theta(\tilde{x}) - (x-\tilde{x})/\sigma^2\|^2$ | 学习去噪等价于学习分数 |
| DDPM-分数关系 | $s_\theta(x_t,t) = -\epsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$ | 两个框架等价 |

---

## 练习题

### 基础题

**5.1** 对于一维高斯 $p(x) = \mathcal{N}(\mu, \sigma^2)$，计算分数函数 $\nabla_x \log p(x)$，并解释其几何含义。

**5.2** 朗之万动力学中，为什么噪声项 $\sqrt{\epsilon}\mathbf{z}$ 是必要的？如果去掉噪声项，会发生什么？

### 中级题

**5.3** 证明DSM目标等价于ESM目标（差一个常数）。具体地，从 $\mathbb{E}_{p(\mathbf{x})q_\sigma(\tilde{\mathbf{x}}|\mathbf{x})}[\|s_\theta(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}}|\mathbf{x})\|^2]$ 出发，展开并化简。

**5.4** 实现NCSN的多尺度分数匹配（使用 $L=5$ 个噪声水平 $\{\sigma_i\}$ 按几何级数分布），并实现退火朗之万采样。在二维混合高斯上验证采样质量。

### 提高题

**5.5** Song et al. 2021 证明DDPM等价于对连续时间SDE的离散化，其逆向SDE为：
$$dx = [f(x,t) - g(t)^2 \nabla_x\log p_t(x)]dt + g(t)d\bar{W}$$
其中 $f(x,t) = -\frac{1}{2}\beta(t)x$（VP-SDE漂移项），$g(t) = \sqrt{\beta(t)}$。利用本章学到的分数估计 $s_\theta(x,t) \approx \nabla_x\log p_t(x)$，实现VP-SDE的数值逆向求解（Euler-Maruyama），并与DDPM采样结果对比。

---

## 练习答案

**5.1** $\nabla_x\log p(x) = \nabla_x[-\frac{(x-\mu)^2}{2\sigma^2}] = -\frac{x-\mu}{\sigma^2}$。几何含义：指向均值方向，距均值越远，力越大（线性恢复力，类似弹簧）。

**5.2** 无噪声项 $\Rightarrow$ 确定性梯度下降 $\Rightarrow$ 粒子收敛到概率密度的**局部极大值**（众数），而非从整个分布采样。噪声提供遍历性。

**5.3** 展开 $\|s_\theta(\tilde{x}) - \nabla\log q_\sigma(\tilde{x}|x)\|^2 = \|s_\theta\|^2 - 2s_\theta^T\nabla\log q_\sigma + \|\nabla\log q_\sigma\|^2$，最后一项是关于 $\theta$ 的常数，展开第二项后与ESM的 $\text{tr}(\nabla s_\theta) + \frac{1}{2}\|s_\theta\|^2$ 进行分部积分对比。

**5.4** 见代码实战扩展（在 `train_dsm` 中循环多个 $\sigma_i$）。

**5.5** 核心：将 $s_\theta(x,t)$ 代入逆向SDE，用Euler-Maruyama：$x_{t-dt} = x_t - [f(x_t,t) - g(t)^2 s_\theta(x_t,t)]dt + g(t)\sqrt{dt}z$。

---

## 延伸阅读

1. **Hyvärinen (2005)**. *Estimation of Non-Normalized Statistical Models by Score Matching*
2. **Vincent (2011)**. *A Connection Between Score Matching and Denoising Autoencoders*
3. **Song & Ermon (2019)**. *Generative Modeling by Estimating Gradients of the Data Distribution* — NCSN
4. **Song et al. (2021)**. *Score-Based Generative Modeling through SDEs* — 统一框架

---

[← 上一章：生成模型全景与VAE回顾](./04-generative-models-overview-vae.md)

[下一章：随机微分方程入门 →](./06-stochastic-differential-equations.md)

[返回目录](../README.md)
