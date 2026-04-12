# 第八章：逆向去噪过程与DDPM

> **本章导读**：上一章我们完整推导了正向扩散过程——将数据逐步变为噪声。本章进入扩散模型的核心：如何学习逆向过程，从纯噪声中逐步恢复出数据。我们将从变分下界（ELBO）出发，严格推导DDPM的训练目标，展示 Ho et al. 2020 如何通过巧妙的参数化将复杂的变分目标简化为一个简洁的噪声预测损失 $L_{simple} = \mathbb{E}\|\epsilon - \epsilon_\theta(x_t, t)\|^2$，并完整实现训练和采样算法。

**前置知识**: 正向扩散过程与闭合形式（第七章），KL散度（第六章），变分推断基础（第六章）

**预计学习时间**: 4-5小时

## 学习目标

1. 理解逆向过程 $q(x_{t-1}|x_t)$ 为何无法直接计算，以及神经网络近似的必要性
2. 掌握DDPM变分下界（ELBO）的完整推导和分解
3. 理解从ELBO到简化目标 $L_{simple}$ 的推导过程，以及噪声预测参数化的动机
4. 能够独立实现DDPM的训练算法（Algorithm 1）和采样算法（Algorithm 2）
5. 分析DDPM的实验行为，包括步数 $T$、FID分数和多样性-保真度权衡

---

## 8.1 逆向过程的困难

### 8.1.1 为什么 $q(x_{t-1}|x_t)$ 无法直接计算

正向过程中，$q(x_t|x_{t-1})$ 是我们人为设计的高斯分布，完全可控。但逆向过程的真实分布：

$$q(x_{t-1}|x_t) = \frac{q(x_t|x_{t-1}) q(x_{t-1})}{q(x_t)}$$

需要知道边际分布 $q(x_{t-1})$ 和 $q(x_t)$。这些边际分布是对所有可能的 $x_0$ 的积分：

$$q(x_t) = \int q(x_t|x_0) q(x_0)\, dx_0$$

而 $q(x_0)$ 就是真实的数据分布——这正是我们想要建模的未知分布。所以我们陷入了循环：要计算逆向过程，需要知道数据分布；但学习数据分布正是我们的目标。

### 8.1.2 一个关键的数学事实

尽管 $q(x_{t-1}|x_t)$ 无法直接计算，但条件后验 $q(x_{t-1}|x_t, x_0)$ 是可以解析计算的——这正是上一章7.5节推导的结果：

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1};\, \tilde{\mu}_t(x_t, x_0),\, \tilde{\beta}_t I)$$

这个分布在训练时可用（因为训练数据提供了 $x_0$），为我们提供了监督信号。

### 8.1.3 神经网络近似的动机

核心思路：用一个参数化的神经网络 $p_\theta(x_{t-1}|x_t)$ 来近似真实的逆向分布 $q(x_{t-1}|x_t)$。由于当 $\beta_t$ 足够小时，$q(x_{t-1}|x_t)$ 近似为高斯分布（Feller, 1949），我们可以将 $p_\theta$ 也设为高斯形式：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t))$$

训练目标：让 $p_\theta(x_{t-1}|x_t)$ 尽可能接近 $q(x_{t-1}|x_t, x_0)$（在训练数据上）。

---

## 8.2 参数化逆向过程

### 8.2.1 逆向马尔科夫链

完整的逆向生成过程定义为：

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)$$

其中 $p(x_T) = \mathcal{N}(x_T; 0, I)$ 是起点分布（纯高斯噪声），每一步的逆向转移核为：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t))$$

生成过程：从 $x_T \sim \mathcal{N}(0, I)$ 出发，依次采样 $x_{T-1}, x_{T-2}, \ldots, x_0$。

### 8.2.2 Ho et al. 2020 的方差选择

DDPM（Ho et al., 2020）做了一个关键的简化：**固定方差**，不用网络预测。

$$\Sigma_\theta(x_t, t) = \sigma_t^2 I$$

两种常见选择：

**选择1**：$\sigma_t^2 = \beta_t$

这对应于逆向过程的上界方差。当 $x_0$ 是确定性的（方差为零），逆向过程的最优方差为 $\beta_t$。

**选择2**：$\sigma_t^2 = \tilde{\beta}_t = \frac{(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\beta_t$

这是真实后验方差（第七章推导得出），对应于 $x_0 \sim \mathcal{N}(0, I)$ 时的最优方差。

两者的关系：$\tilde{\beta}_t \leq \beta_t$，且在 $t$ 较大时两者接近。Ho et al. 报告实验中两者差异不大，默认使用 $\sigma_t^2 = \beta_t$。

> **注意**：后续工作（Nichol & Dhariwal, 2021）证明学习方差可以在较少步数时显著提升质量，但对于理解DDPM核心原理，固定方差已经足够。

### 8.2.3 只需预测均值

固定方差后，$p_\theta(x_{t-1}|x_t)$ 只有均值 $\mu_\theta(x_t, t)$ 需要学习。问题简化为：

$$\text{找到 } \mu_\theta \text{ 使得 } \mu_\theta(x_t, t) \approx \tilde{\mu}_t(x_t, x_0) \text{ 对所有训练样本成立}$$

但 $\tilde{\mu}_t$ 依赖于 $x_0$，而在生成时我们没有 $x_0$。解决方案是让网络从 $x_t$ 预测 $x_0$（或等价地，预测噪声 $\epsilon$）。

---

## 8.3 训练目标推导（完整ELBO）

### 8.3.1 变分下界出发

对于任意生成模型，对数似然的变分下界为：

$$\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T}|x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] \equiv -L$$

即 $-\log p_\theta(x_0) \leq L$。我们的目标是最小化 $L$（即最大化下界）。

展开 $L$：

$$L = \mathbb{E}_q\left[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$

$$= \mathbb{E}_q\left[-\log \frac{p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^T q(x_t|x_{t-1})}\right]$$

$$= \mathbb{E}_q\left[-\log p(x_T) - \sum_{t=1}^T \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]$$

### 8.3.2 关键的重写技巧

为了将 $L$ 分解为可计算的KL散度之和，需要引入条件后验。对 $t \geq 2$，利用贝叶斯定理：

$$q(x_t|x_{t-1}) = \frac{q(x_{t-1}|x_t, x_0) \cdot q(x_t|x_0)}{q(x_{t-1}|x_0)}$$

代入并重新组织求和项（对 $t = 2, \ldots, T$）：

$$\sum_{t=2}^{T}\log\frac{q(x_t|x_{t-1})}{p_\theta(x_{t-1}|x_t)} = \sum_{t=2}^{T}\log\frac{q(x_{t-1}|x_t,x_0) \cdot q(x_t|x_0)}{q(x_{t-1}|x_0) \cdot p_\theta(x_{t-1}|x_t)}$$

$$= \sum_{t=2}^{T}\log\frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)} + \sum_{t=2}^{T}\log\frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}$$

第二个求和是一个伸缩（telescoping）和：

$$\sum_{t=2}^{T}\log\frac{q(x_t|x_0)}{q(x_{t-1}|x_0)} = \log\frac{q(x_T|x_0)}{q(x_1|x_0)}$$

### 8.3.3 完整分解

将所有项合并，$L$ 分解为三部分：

$$\boxed{L = \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^{T}\underbrace{D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} + \underbrace{(-\mathbb{E}_q[\log p_\theta(x_0|x_1)])}_{L_0}}$$

逐项解释：

**$L_T = D_{KL}(q(x_T|x_0) \| p(x_T))$**

正向过程终态 $q(x_T|x_0)$ 与先验 $p(x_T) = \mathcal{N}(0, I)$ 的KL散度。当 $T$ 足够大时 $\bar{\alpha}_T \approx 0$，$q(x_T|x_0) \approx \mathcal{N}(0, I) = p(x_T)$，所以 $L_T \approx 0$。此项不含可训练参数。

**$L_{t-1} = D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))$**（$t = 2, \ldots, T$）

这是训练的核心项。两个高斯分布之间的KL散度有解析形式。

**$L_0 = -\mathbb{E}_q[\log p_\theta(x_0|x_1)]$**

重建项。实践中通常用离散化的高斯对数似然来处理像素值。

### 8.3.4 KL散度的闭合形式

对于 $L_{t-1}$，两个高斯分布的KL散度（方差为标量的情况）：

$$D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t)) = \frac{1}{2\sigma_t^2}\|\tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2 + C$$

其中 $C$ 是与 $\theta$ 无关的常数（包含方差的对数比项）。

因此，训练目标归结为：让 $\mu_\theta(x_t, t)$ 逼近真实后验均值 $\tilde{\mu}_t(x_t, x_0)$。

---

## 8.4 噪声预测的简化目标

### 8.4.1 $\mu_\theta$ 的参数化选择

从上一节，我们需要预测后验均值 $\tilde{\mu}_t(x_t, x_0)$。利用第七章的结果：

$$\tilde{\mu}_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon\right)$$

其中 $\epsilon$ 是生成 $x_t$ 时使用的噪声：$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$。

有三种等价的参数化方式：

**方式A：预测 $x_0$**

让网络直接预测原始数据 $\hat{x}_0 = f_\theta(x_t, t)$，然后代入后验均值公式：

$$\mu_\theta(x_t, t) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} f_\theta(x_t, t)$$

**方式B：预测噪声 $\epsilon$（DDPM默认）**

让网络预测噪声 $\hat{\epsilon} = \epsilon_\theta(x_t, t)$，然后：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$$

**方式C：直接预测均值**

让网络直接输出 $\mu_\theta(x_t, t)$。

### 8.4.2 噪声预测目标的推导

选择方式B（预测 $\epsilon$），将 $\mu_\theta$ 代入 $L_{t-1}$：

$$L_{t-1} = \frac{1}{2\sigma_t^2}\left\|\tilde{\mu}_t - \mu_\theta\right\|^2 = \frac{1}{2\sigma_t^2}\left\|\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}(\epsilon - \epsilon_\theta(x_t, t))\right\|^2$$

$$= \frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}\|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

因此：

$$L_{t-1} = w(t) \cdot \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

其中权重 $w(t) = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)}$ 依赖于 $t$。

### 8.4.3 简化目标

Ho et al. 发现，去掉权重 $w(t)$，使用简化目标效果更好：

$$\boxed{L_{simple} = \mathbb{E}_{t \sim \text{Uniform}(1,T),\, x_0 \sim q(x_0),\, \epsilon \sim \mathcal{N}(0,I)}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]}$$

其中 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$。

### 8.4.4 为什么简化有效

移除权重 $w(t)$ 后，$L_{simple}$ 相当于对各时间步使用均匀权重。分析表明：

- 原始权重 $w(t)$ 在 $t$ 较小时非常大（因为 $\beta_t$ 小使分母小），导致训练集中在低噪声区域
- 简化目标让高噪声时间步（$t$ 较大）也获得足够的梯度信号
- 这对生成质量有益，因为逆向过程从高噪声开始，如果高噪声步预测不准，误差会累积
- 直觉上，$L_{simple}$ 相当于"下采样"低噪声步的权重、"上采样"高噪声步的权重

从另一个角度看，$L_{simple}$ 可以解释为一种多任务学习：网络在不同噪声水平下学习去噪，均匀权重确保了各任务之间的平衡。

---

## 8.5 训练算法

### 8.5.1 Algorithm 1: Training

DDPM的训练算法出奇地简单：

---
**算法1：DDPM训练**

**重复**以下步骤直到收敛：
1. 从数据集采样 $x_0 \sim q(x_0)$
2. 随机采样时间步 $t \sim \text{Uniform}(\{1, 2, \ldots, T\})$
3. 采样噪声 $\epsilon \sim \mathcal{N}(0, I)$
4. 计算加噪样本 $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$
5. 梯度下降步：$\nabla_\theta \|\epsilon - \epsilon_\theta(x_t, t)\|^2$

---

每次迭代的计算开销仅为：一次前向传播（网络预测 $\epsilon_\theta$）+ 一次反向传播。不需要模拟整条马尔科夫链。

### 8.5.2 Algorithm 2: Sampling

---
**算法2：DDPM采样**

1. 初始化 $x_T \sim \mathcal{N}(0, I)$
2. **for** $t = T, T-1, \ldots, 1$ **do**:
   - 如果 $t > 1$，采样 $z \sim \mathcal{N}(0, I)$；否则 $z = 0$
   - 计算 $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z$
3. **返回** $x_0$

---

采样公式的推导：由逆向过程 $p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$，通过重参数化：

$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$$

其中均值为：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$$

注意最后一步（$t = 1$）不加噪声（$z = 0$），因为我们要输出确定性的 $x_0$。

### 8.5.3 采样过程的直觉

采样的每一步可以分解为两个操作：

$$x_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}}x_t}_{\text{缩放恢复}} - \underbrace{\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(x_t, t)}_{\text{预测噪声去除}} + \underbrace{\sigma_t z}_{\text{随机性注入}}$$

1. **缩放恢复**：补偿正向过程中的 $\sqrt{1-\beta_t}$ 缩放
2. **去噪**：根据网络预测的噪声方向，进行"纠正"
3. **随机性**：添加适量噪声以保持分布的多样性

如果没有第三步（$\sigma_t = 0$），采样会退化为确定性过程，失去多样性——这实际上就是DDIM（Denoising Diffusion Implicit Models, Song et al., 2020）的特殊情况。

---

## 8.6 DDPM实验分析

### 8.6.1 图像质量 vs 步数 $T$

DDPM使用 $T = 1000$ 步采样，这是一个巨大的计算负担。步数对质量的影响：

| 步数 $T$ | FID (CIFAR-10) | 采样时间 | 分析 |
|:---:|:---:|:---:|:---|
| 50 | $\sim 50$ | 快 | 每步变化大，高斯近似不准确 |
| 200 | $\sim 15$ | 中 | 质量显著提升 |
| 1000 | 3.17 | 慢 | 接近最优 |
| 4000 | $\sim 3.1$ | 非常慢 | 收益递减 |

$T$ 过小时，逆向每一步的变化过大，高斯近似的误差累积导致生成质量崩溃。

### 8.6.2 FID分数分析

FID（Frechet Inception Distance）是衡量生成图像质量的核心指标（详见第九章），分数越低越好。DDPM的里程碑结果：

- **CIFAR-10**：FID = 3.17，IS = 9.46（当时与GAN竞争力相当）
- **LSUN Bedroom 256x256**：FID = 4.90
- **LSUN Church 256x256**：FID = 7.89
- **CelebA-HQ 256x256**：高质量人脸生成

DDPM的重要贡献在于证明了扩散模型可以生成与GAN媲美的图像质量，同时具有更稳定的训练过程和更好的模式覆盖。

### 8.6.3 多样性 vs 保真度权衡

与GAN相比，DDPM有一个重要优势：

- **GAN**倾向于mode collapse，生成多样性有限但单张图像质量高
- **DDPM**的随机采样过程天然保证了多样性，但可能牺牲单张图像的锐利度

可以通过方差 $\sigma_t$ 的选择来调节这个权衡：
- $\sigma_t$ 较大 $\to$ 更大随机性 $\to$ 更高多样性
- $\sigma_t$ 较小 $\to$ 更确定性 $\to$ 更高保真度（更锐利）

### 8.6.4 渐进式生成的特性

DDPM的采样过程揭示了一个有趣的现象——**由粗到精的生成**：

- $t \approx T$（早期步骤）：决定全局结构（物体位置、大致颜色）
- $t \approx T/2$（中期步骤）：确定中层特征（物体形状、主要纹理）
- $t \approx 0$（后期步骤）：精细化细节（边缘锐化、高频纹理）

这与人类绘画从草图到细节的过程惊人地相似，也解释了为什么扩散模型在生成复杂场景时表现出色。

---

## 代码实战

```python
"""
第八章代码实战：完整DDPM实现——训练与采样
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm


# ============================================================
# 1. 噪声调度器（从第七章复用，略作增强）
# ============================================================

class NoiseScheduler:
    """DDPM噪声调度器，封装所有与调度相关的计算"""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "linear",
        device: str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif schedule_type == "cosine":
            steps = torch.arange(num_timesteps + 1, dtype=torch.float64, device=device)
            s = 0.008
            f = torch.cos(((steps / num_timesteps) + s) / (1 + s) * (np.pi / 2)) ** 2
            alpha_bars = f / f[0]
            betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
            self.betas = torch.clamp(betas, min=0.0, max=0.999).float()
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alphas = 1.0 - self.betas  # (T,)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # (T,)
        self.alpha_bars_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alpha_bars[:-1]]
        )  # (T,)
        
        # 正向过程参数
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        
        # 逆向过程参数
        self.sqrt_recip_alphas = 1.0 / torch.sqrt(self.alphas)
        self.posterior_variance = (
            (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars) * self.betas
        )
        # 防止 t=0 时方差为 0 导致 log 出问题
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        
        # 采样时的标准差（两种选择）
        self.sigma_beta = torch.sqrt(self.betas)  # sigma_t = sqrt(beta_t)
        self.sigma_posterior = torch.sqrt(self.posterior_variance)  # sigma_t = sqrt(tilde_beta_t)


# ============================================================
# 2. 简单的噪声预测网络（用于2D数据演示）
# ============================================================

class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码，将时间步 t 编码为高维向量"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) 时间步索引
        Returns:
            emb: (B, dim) 位置编码
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = np.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale
        )  # (half_dim,)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return emb


class SimpleNoisePredictor(nn.Module):
    """用于2D数据的简单噪声预测网络
    
    输入：(x_t, t) -> 输出：预测的噪声 epsilon_hat
    """
    
    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 256,
        time_emb_dim: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 输入层：数据 + 时间嵌入
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        
        # 中间层
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            ])
        self.network = nn.Sequential(*layers)
        
        # 输出层
        self.output_proj = nn.Linear(hidden_dim, data_dim)
    
    def forward(
        self,
        x_t: torch.Tensor,  # (B, D)
        t: torch.Tensor,    # (B,)
    ) -> torch.Tensor:      # (B, D)
        """预测噪声"""
        t_emb = self.time_embed(t)          # (B, hidden_dim)
        x_emb = self.input_proj(x_t)        # (B, hidden_dim)
        h = x_emb + t_emb                   # (B, hidden_dim) — 加法注入
        h = self.network(h)                  # (B, hidden_dim)
        noise_pred = self.output_proj(h)     # (B, D)
        return noise_pred


# ============================================================
# 3. DDPM核心类
# ============================================================

class DDPM:
    """Denoising Diffusion Probabilistic Model
    
    封装训练和采样逻辑。
    """
    
    def __init__(
        self,
        model: nn.Module,
        scheduler: NoiseScheduler,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device
    
    # -------- 训练相关 --------
    
    def compute_loss(
        self,
        x_0: torch.Tensor,  # (B, D) 或 (B, C, H, W)
    ) -> torch.Tensor:
        """计算 L_simple 损失
        
        Algorithm 1 的单步实现。
        """
        batch_size = x_0.shape[0]
        
        # 1. 随机采样时间步
        t = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,), device=self.device
        )  # (B,)
        
        # 2. 采样噪声
        noise = torch.randn_like(x_0)  # (B, D)
        
        # 3. 计算 x_t（闭合形式加噪）
        sqrt_alpha_bar = self.scheduler.sqrt_alpha_bars[t]  # (B,)
        sqrt_one_minus_alpha_bar = self.scheduler.sqrt_one_minus_alpha_bars[t]  # (B,)
        
        # 调整维度以便广播
        while sqrt_alpha_bar.dim() < x_0.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise  # (B, D)
        
        # 4. 网络预测噪声
        noise_pred = self.model(x_t, t)  # (B, D)
        
        # 5. L_simple: 均方误差
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    # -------- 采样相关 --------
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        return_intermediates: bool = False,
        intermediate_steps: Optional[list[int]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, list[torch.Tensor]]:
        """Algorithm 2: DDPM采样
        
        Args:
            shape: 生成样本的形状，如 (batch_size, data_dim)
            return_intermediates: 是否返回中间步骤
            intermediate_steps: 需要记录的中间步骤列表
            
        Returns:
            生成的样本 x_0，以及可选的中间步骤列表
        """
        self.model.eval()
        
        # 1. 初始化 x_T ~ N(0, I)
        x_t = torch.randn(shape, device=self.device)  # (B, D)
        
        intermediates = []
        if intermediate_steps is None:
            intermediate_steps = list(range(0, self.scheduler.num_timesteps, 100))
        
        # 2. 迭代去噪
        for t_val in reversed(range(self.scheduler.num_timesteps)):
            t = torch.full(
                (shape[0],), t_val, dtype=torch.long, device=self.device
            )  # (B,)
            
            # 预测噪声
            noise_pred = self.model(x_t, t)  # (B, D)
            
            # 计算均值
            sqrt_recip_alpha = self.scheduler.sqrt_recip_alphas[t_val]
            beta = self.scheduler.betas[t_val]
            sqrt_one_minus_alpha_bar = self.scheduler.sqrt_one_minus_alpha_bars[t_val]
            
            mu = sqrt_recip_alpha * (
                x_t - beta / sqrt_one_minus_alpha_bar * noise_pred
            )  # (B, D)
            
            # 添加噪声（最后一步不加）
            if t_val > 0:
                sigma = self.scheduler.sigma_beta[t_val]
                z = torch.randn_like(x_t)  # (B, D)
                x_t = mu + sigma * z
            else:
                x_t = mu
            
            # 记录中间结果
            if return_intermediates and t_val in intermediate_steps:
                intermediates.append(x_t.clone())
        
        if return_intermediates:
            return x_t, intermediates
        return x_t


# ============================================================
# 4. 2D数据集
# ============================================================

def make_swiss_roll(n_samples: int = 10000) -> torch.Tensor:
    """生成2D瑞士卷数据集"""
    t = torch.linspace(1.5 * np.pi, 4.5 * np.pi, n_samples)
    x = t * torch.cos(t) / (4.5 * np.pi)
    y = t * torch.sin(t) / (4.5 * np.pi)
    data = torch.stack([x, y], dim=1)  # (N, 2)
    data += torch.randn_like(data) * 0.02  # 少量噪声
    return data


def make_two_moons(n_samples: int = 10000) -> torch.Tensor:
    """生成双月数据集"""
    n = n_samples // 2
    # 上半月
    theta1 = torch.linspace(0, np.pi, n)
    x1 = torch.cos(theta1)
    y1 = torch.sin(theta1)
    # 下半月
    theta2 = torch.linspace(0, np.pi, n_samples - n)
    x2 = 1 - torch.cos(theta2)
    y2 = 1 - torch.sin(theta2) - 0.5
    
    x = torch.cat([x1, x2])
    y = torch.cat([y1, y2])
    data = torch.stack([x, y], dim=1)  # (N, 2)
    data += torch.randn_like(data) * 0.05
    return data


# ============================================================
# 5. 训练循环
# ============================================================

def train_ddpm(
    ddpm: DDPM,
    dataset: torch.Tensor,       # (N, D)
    num_epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    log_every: int = 20,
) -> list[float]:
    """完整的DDPM训练循环
    
    Returns:
        losses: 每个epoch的平均损失列表
    """
    optimizer = torch.optim.Adam(ddpm.model.parameters(), lr=learning_rate)
    dataset = dataset.to(ddpm.device)
    n_samples = dataset.shape[0]
    
    losses = []
    
    for epoch in range(num_epochs):
        # 随机打乱数据
        perm = torch.randperm(n_samples, device=ddpm.device)
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch = dataset[perm[i:i + batch_size]]  # (B, D)
            
            optimizer.zero_grad()
            loss = ddpm.compute_loss(batch)
            loss.backward()
            # 梯度裁剪（稳定训练）
            torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch+1:>4d}/{num_epochs} | Loss: {avg_loss:.6f}")
    
    return losses


# ============================================================
# 6. 可视化
# ============================================================

def visualize_sampling_process(
    ddpm: DDPM,
    n_samples: int = 1000,
    data_dim: int = 2,
) -> plt.Figure:
    """可视化采样过程的中间步骤"""
    T = ddpm.scheduler.num_timesteps
    steps_to_show = [T - 1, int(T * 0.75), int(T * 0.5), int(T * 0.25), int(T * 0.1), 0]
    
    x_final, intermediates = ddpm.sample(
        shape=(n_samples, data_dim),
        return_intermediates=True,
        intermediate_steps=steps_to_show,
    )
    
    # intermediates 是按时间倒序存储的
    fig, axes = plt.subplots(1, len(intermediates), figsize=(3 * len(intermediates), 3))
    
    for i, (x_inter, t_val) in enumerate(zip(intermediates, steps_to_show)):
        x_np = x_inter.cpu().numpy()
        axes[i].scatter(x_np[:, 0], x_np[:, 1], s=1, alpha=0.5, c="steelblue")
        axes[i].set_title(f"t = {t_val}")
        axes[i].set_xlim(-2, 2)
        axes[i].set_ylim(-2, 2)
        axes[i].set_aspect("equal")
        axes[i].grid(True, alpha=0.3)
    
    fig.suptitle("DDPM Sampling: From Noise to Data", fontsize=14)
    plt.tight_layout()
    return fig


def plot_training_results(
    losses: list[float],
    real_data: torch.Tensor,
    generated_data: torch.Tensor,
) -> plt.Figure:
    """绘制训练损失曲线和生成结果对比"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 损失曲线
    axes[0].plot(losses, linewidth=1.5, color="steelblue")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")
    
    # 真实数据
    real_np = real_data.cpu().numpy()
    axes[1].scatter(real_np[:, 0], real_np[:, 1], s=1, alpha=0.5, c="steelblue")
    axes[1].set_title("Real Data")
    axes[1].set_xlim(-2, 2)
    axes[1].set_ylim(-2, 2)
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)
    
    # 生成数据
    gen_np = generated_data.cpu().numpy()
    axes[2].scatter(gen_np[:, 0], gen_np[:, 1], s=1, alpha=0.5, c="coral")
    axes[2].set_title("Generated Data")
    axes[2].set_xlim(-2, 2)
    axes[2].set_ylim(-2, 2)
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================
# 7. 主程序：完整的2D DDPM演示
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 数据准备 ---
    data = make_swiss_roll(n_samples=10000)
    print(f"Dataset shape: {data.shape}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    
    # --- 创建模型 ---
    T = 1000
    scheduler = NoiseScheduler(
        num_timesteps=T,
        beta_start=1e-4,
        beta_end=0.02,
        schedule_type="linear",
        device=device,
    )
    
    model = SimpleNoisePredictor(
        data_dim=2,
        hidden_dim=256,
        time_emb_dim=128,
        num_layers=4,
    )
    
    ddpm = DDPM(model=model, scheduler=scheduler, device=device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # --- 训练 ---
    print("\n=== Training ===")
    losses = train_ddpm(
        ddpm=ddpm,
        dataset=data,
        num_epochs=200,
        batch_size=256,
        learning_rate=3e-4,
        log_every=20,
    )
    
    # --- 采样 ---
    print("\n=== Sampling ===")
    generated = ddpm.sample(shape=(5000, 2))
    if isinstance(generated, tuple):
        generated = generated[0]
    print(f"Generated samples shape: {generated.shape}")
    
    # --- 可视化 ---
    # 1. 训练结果
    fig_results = plot_training_results(losses, data, generated.cpu())
    plt.savefig("ddpm_training_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("训练结果已保存为 ddpm_training_results.png")
    
    # 2. 采样过程
    fig_sampling = visualize_sampling_process(ddpm, n_samples=2000, data_dim=2)
    plt.savefig("ddpm_sampling_process.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("采样过程已保存为 ddpm_sampling_process.png")
    
    # --- 定量分析 ---
    print("\n=== 定量分析 ===")
    
    # 比较真实数据和生成数据的统计量
    print(f"真实数据 - 均值: {data.mean(0).numpy()}, 标准差: {data.std(0).numpy()}")
    gen_cpu = generated.cpu()
    print(f"生成数据 - 均值: {gen_cpu.mean(0).numpy()}, 标准差: {gen_cpu.std(0).numpy()}")
    
    # 损失分析
    print(f"\n最终损失: {losses[-1]:.6f}")
    print(f"最低损失: {min(losses):.6f} (epoch {np.argmin(losses) + 1})")
    
    # 不同时间步的去噪质量分析
    print("\n=== 各时间步损失分析 ===")
    ddpm.model.eval()
    t_values = [10, 50, 100, 250, 500, 750, 999]
    test_data = data[:1000].to(device)
    
    for t_val in t_values:
        t = torch.full((1000,), t_val, dtype=torch.long, device=device)
        noise = torch.randn_like(test_data)
        sqrt_ab = scheduler.sqrt_alpha_bars[t_val]
        sqrt_1_ab = scheduler.sqrt_one_minus_alpha_bars[t_val]
        x_t = sqrt_ab * test_data + sqrt_1_ab * noise
        
        with torch.no_grad():
            noise_pred = ddpm.model(x_t, t)
            loss_t = F.mse_loss(noise_pred, noise).item()
        
        print(f"t={t_val:>4d} | alpha_bar={scheduler.alpha_bars[t_val]:.4f} | loss={loss_t:.6f}")
```

---

## 本章小结

| 概念 | 公式 / 要点 |
|:---|:---|
| 逆向过程 | $p_\theta(x_{t-1}\|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \sigma_t^2 I)$ |
| ELBO分解 | $L = L_T + \sum_{t>1} L_{t-1} + L_0$ |
| $L_{t-1}$ | $D_{KL}(q(x_{t-1}\|x_t,x_0) \|\| p_\theta(x_{t-1}\|x_t))$ |
| 噪声预测均值 | $\mu_\theta = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t))$ |
| 简化目标 | $L_{simple} = \mathbb{E}\|\epsilon - \epsilon_\theta(x_t,t)\|^2$ |
| 训练 | 采样 $t$，加噪 $x_t$，预测 $\epsilon$，MSE损失 |
| 采样 | $x_{t-1} = \mu_\theta + \sigma_t z$，从 $t=T$ 迭代到 $t=0$ |
| 方差选择 | $\sigma_t^2 = \beta_t$ 或 $\sigma_t^2 = \tilde{\beta}_t$，影响不大 |

---

## 练习题

### 基础题

**练习 8.1**（ELBO理解）

解释ELBO分解中每一项的直觉含义：$L_T$ 为什么近似为零？$L_{t-1}$ 为什么是训练的核心？$L_0$ 在图像生成中如何处理？

**练习 8.2**（采样公式推导）

从逆向过程 $p_\theta(x_{t-1}|x_t)$ 的高斯形式出发，利用重参数化技巧，推导出采样公式：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right) + \sigma_t z$$

### 中级题

**练习 8.3**（权重分析）

完整写出原始ELBO中 $L_{t-1}$ 的权重 $w(t) = \frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}$。分别对 $\sigma_t^2 = \beta_t$ 和 $\sigma_t^2 = \tilde{\beta}_t$ 两种情况画出 $w(t)$ 关于 $t$ 的曲线（使用标准线性调度参数）。解释为什么 Ho et al. 选择去掉权重。

**练习 8.4**（$x_0$ 预测 vs $\epsilon$ 预测的等价性）

证明：如果 $\hat{x}_0 = f_\theta(x_t, t)$ 是网络对 $x_0$ 的预测，定义 $\hat{\epsilon} = \frac{x_t - \sqrt{\bar{\alpha}_t}\hat{x}_0}{\sqrt{1-\bar{\alpha}_t}}$，则 $||\epsilon - \hat{\epsilon}||^2 = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}||x_0 - \hat{x}_0||^2$。讨论这对不同时间步损失权重的影响。

### 提高题

**练习 8.5**（DDPM与分数匹配的联系）

证明 DDPM 的噪声预测目标 $\mathbb{E}\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ 等价于去噪分数匹配（Denoising Score Matching）目标。具体地，说明 $\epsilon_\theta(x_t, t) = -\sqrt{1-\bar{\alpha}_t}\, s_\theta(x_t, t)$，其中 $s_\theta$ 是对分数函数 $\nabla_{x_t} \log q(x_t)$ 的估计。提示：利用 $\nabla_{x_t}\log q(x_t|x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$。

---

## 练习答案

### 练习 8.1 解答

**$L_T = D_{KL}(q(x_T|x_0) \| p(x_T))$**

$L_T$ 度量正向过程终态与先验分布的差异。当 $T = 1000$ 且使用标准调度时，$\bar{\alpha}_T \approx 0$，$q(x_T|x_0) \approx \mathcal{N}(0, I) = p(x_T)$，所以 $L_T \approx 0$。此项不含可训练参数，在训练中可以忽略。

**$L_{t-1} = D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))$**

$L_{t-1}$ 是训练的核心，因为它直接度量逆向模型 $p_\theta$ 与真实后验的差距。由于两边都是高斯分布，KL散度归结为均值之差的平方。共有 $T-1$ 个这样的项，覆盖了整个逆向过程。

**$L_0 = -\mathbb{E}_q[\log p_\theta(x_0|x_1)]$**

$L_0$ 是重建项，衡量从 $x_1$（非常接近 $x_0$ 的略带噪声版本）重建 $x_0$ 的能力。对于图像数据，通常将像素值离散化为 $\{0, 1, \ldots, 255\}$，使用离散化的高斯对数似然。在使用 $L_{simple}$ 时，$L_0$ 被统一到整体MSE目标中。

### 练习 8.2 解答

逆向过程定义为 $p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$。

由重参数化：$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$，$z \sim \mathcal{N}(0, I)$。

代入 $\mu_\theta$ 的表达式（噪声预测参数化）：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$$

得到：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z$$

最后一步 $t = 1$ 时，$z = 0$（不加随机性），直接输出 $\mu_\theta$ 作为 $x_0$。

### 练习 8.3 解答

权重公式为 $w(t) = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)}$。

**情况1**：$\sigma_t^2 = \beta_t$

$$w(t) = \frac{\beta_t^2}{2\beta_t \alpha_t (1-\bar{\alpha}_t)} = \frac{\beta_t}{2\alpha_t(1-\bar{\alpha}_t)}$$

在 $t$ 较小时，$\beta_t \approx 10^{-4}$，$\alpha_t \approx 1$，$1-\bar{\alpha}_t \approx t \cdot 10^{-4}$，所以 $w(t) \approx \frac{1}{2t}$，随 $t$ 减小而急剧增大。

**情况2**：$\sigma_t^2 = \tilde{\beta}_t = \frac{(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\beta_t$

$$w(t) = \frac{\beta_t^2}{2 \cdot \frac{(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\beta_t \cdot \alpha_t (1-\bar{\alpha}_t)} = \frac{\beta_t}{2\alpha_t(1-\bar{\alpha}_{t-1})}$$

类似的行为，在 $t$ 较小时 $w(t) \to \infty$。

Ho et al. 选择去掉权重因为：（1）原始权重使训练过度关注低噪声区域（小 $t$），而这些区域的预测相对容易；（2）均匀权重让高噪声区域也获得足够的学习信号，这对生成质量至关重要。

### 练习 8.4 解答

给定 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，有：

$$\hat{\epsilon} = \frac{x_t - \sqrt{\bar{\alpha}_t}\hat{x}_0}{\sqrt{1-\bar{\alpha}_t}}$$

$$\epsilon - \hat{\epsilon} = \epsilon - \frac{x_t - \sqrt{\bar{\alpha}_t}\hat{x}_0}{\sqrt{1-\bar{\alpha}_t}}$$

代入 $x_t$：

$$= \epsilon - \frac{\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon - \sqrt{\bar{\alpha}_t}\hat{x}_0}{\sqrt{1-\bar{\alpha}_t}}$$

$$= \epsilon - \epsilon - \frac{\sqrt{\bar{\alpha}_t}(x_0 - \hat{x}_0)}{\sqrt{1-\bar{\alpha}_t}}$$

$$= -\frac{\sqrt{\bar{\alpha}_t}}{\sqrt{1-\bar{\alpha}_t}}(x_0 - \hat{x}_0)$$

因此：

$$\|\epsilon - \hat{\epsilon}\|^2 = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}\|x_0 - \hat{x}_0\|^2 = \text{SNR}(t) \cdot \|x_0 - \hat{x}_0\|^2$$

这意味着 $\epsilon$-预测目标自动对 $x_0$-预测施加了SNR加权：在高SNR（小 $t$）时，$x_0$ 预测误差被放大；在低SNR（大 $t$）时被缩小。这是合理的，因为高SNR时应该能更准确地恢复 $x_0$。

### 练习 8.5 解答

对于加噪分布 $q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$，分数函数为：

$$\nabla_{x_t}\log q(x_t|x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t}$$

利用 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，有 $x_t - \sqrt{\bar{\alpha}_t}x_0 = \sqrt{1-\bar{\alpha}_t}\epsilon$，因此：

$$\nabla_{x_t}\log q(x_t|x_0) = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

去噪分数匹配目标为：

$$\mathbb{E}_{x_0, x_t}\left[\left\|s_\theta(x_t,t) - \nabla_{x_t}\log q(x_t|x_0)\right\|^2\right] = \mathbb{E}\left[\left\|s_\theta(x_t,t) + \frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}\right\|^2\right]$$

定义 $\epsilon_\theta(x_t, t) = -\sqrt{1-\bar{\alpha}_t}\, s_\theta(x_t, t)$，代入：

$$= \mathbb{E}\left[\left\|-\frac{\epsilon_\theta}{\sqrt{1-\bar{\alpha}_t}} + \frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}\right\|^2\right] = \frac{1}{1-\bar{\alpha}_t}\mathbb{E}\left[\|\epsilon - \epsilon_\theta(x_t,t)\|^2\right]$$

这与DDPM的 $L_{simple}$ 仅差一个时间步相关的常数因子 $\frac{1}{1-\bar{\alpha}_t}$。因此DDPM的噪声预测本质上就是在学习数据的分数函数（score function），扩散模型也因此被称为"基于分数的生成模型"（Score-Based Generative Models）。

---

## 延伸阅读

1. **Ho, J., Jain, A., & Abbeel, P.** (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020.* -- DDPM原始论文，本章的主要参考。

2. **Song, Y., & Ermon, S.** (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS 2019.* -- 分数匹配视角的扩散模型（NCSN）。

3. **Song, J., Meng, C., & Ermon, S.** (2020). "Denoising Diffusion Implicit Models." *ICLR 2021.* -- DDIM：将DDPM推广为非马尔科夫过程，允许确定性采样和加速。

4. **Nichol, A. Q., & Dhariwal, P.** (2021). "Improved Denoising Diffusion Probabilistic Models." *ICML 2021.* -- 改进DDPM：学习方差、余弦调度、重要性采样。

5. **Kingma, D. P., & Welling, M.** (2013). "Auto-Encoding Variational Bayes." *ICLR 2014.* -- VAE原始论文，变分下界推导的经典参考。

6. **Luo, C.** (2022). "Understanding Diffusion Models: A Unified Perspective." *arXiv:2208.11970.* -- 扩散模型的统一综述，推荐作为本章的补充阅读。

---

[上一章：正向扩散过程与加噪](./07-diffusion-process-forward.md) | [目录](../README.md) | [下一章：噪声预测网络与训练目标](./09-noise-prediction-network-training.md)
