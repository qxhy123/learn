# 第三章：变分推断基础

> **本章导读**：扩散模型的训练目标本质上是一个变分下界（ELBO）。本章从推断问题的困难出发，系统介绍变分推断的数学基础，重点推导VAE的ELBO，为理解DDPM的目标函数做好准备。

**前置知识**：前两章内容，KL散度，重参数化技巧
**预计学习时间**：110分钟

---

## 学习目标

完成本章学习后，你将能够：
1. 理解贝叶斯推断的计算困难，掌握变分推断的基本思想
2. 从第一性原理推导变分下界（ELBO）
3. 理解均场假设，推导坐标上升更新规则
4. 实现并训练VAE，理解重建项与KL项的权衡
5. 将VAE的ELBO与扩散模型的训练目标建立联系

---

## 3.1 推断问题的困难

### 后验计算的挑战

在潜变量模型中，我们有：
- 观测变量 $\mathbf{x}$（例如图像）
- 潜变量 $\mathbf{z}$（例如语义编码）
- 生成模型 $p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z})$

**推断任务**：给定观测 $\mathbf{x}$，计算后验 $p_\theta(\mathbf{z}|\mathbf{x})$。

由贝叶斯定理：

$$p_\theta(\mathbf{z}|\mathbf{x}) = \frac{p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{p_\theta(\mathbf{x})}$$

**关键困难**：边缘似然（模型证据）：

$$p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

这个积分通常是**不可解析计算的**（intractable）：
- 高维积分，数值方法计算量指数增长
- 被积函数可能非常复杂（神经网络定义的似然）

### 变分推断的基本思想

**核心想法**：用一个参数化的、容易计算的分布族 $q_\phi(\mathbf{z}|\mathbf{x})$ 来**近似**真实后验 $p_\theta(\mathbf{z}|\mathbf{x})$。

通过优化参数 $\phi$，使 $q_\phi(\mathbf{z}|\mathbf{x})$ 尽量接近 $p_\theta(\mathbf{z}|\mathbf{x})$（用KL散度衡量距离）。

---

## 3.2 变分下界（ELBO）

### 推导过程

对数边缘似然可以分解为：

$$\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z}) d\mathbf{z}$$

引入变分分布 $q_\phi(\mathbf{z}|\mathbf{x})$，乘以1：

$$= \log \int q_\phi(\mathbf{z}|\mathbf{x}) \frac{p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} d\mathbf{z}$$

由Jensen不等式（$\log$ 是凹函数）：

$$\geq \int q_\phi(\mathbf{z}|\mathbf{x}) \log \frac{p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} d\mathbf{z}$$

这就是**证据下界（ELBO）**：

$$\log p_\theta(\mathbf{x}) \geq \mathcal{L}(\theta, \phi; \mathbf{x}) \triangleq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log \frac{p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]$$

### ELBO的两种等价分解

**分解一**（重建 + 正则）：

$$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{重建项（越大越好）}} - \underbrace{D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{KL正则项（越小越好）}}$$

**分解二**（log似然 - KL距离）：

$$\log p_\theta(\mathbf{x}) = \mathcal{L}(\theta,\phi;\mathbf{x}) + D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x}))$$

因为 $D_{KL} \geq 0$，所以 $\mathcal{L} \leq \log p_\theta(\mathbf{x})$，等号成立当且仅当 $q_\phi = p_\theta(\cdot|\mathbf{x})$。

> **关键洞察**：最大化ELBO等价于同时：(1) 最大化对数似然，(2) 最小化变分后验与真实后验的KL散度。

---

## 3.3 均场变分推断

### 均场假设

**均场假设**：变分分布可以完全分解：

$$q(\mathbf{z}) = \prod_{j=1}^d q_j(z_j)$$

各个维度之间**没有依赖关系**（这是一个强假设，但使推导可行）。

### 坐标上升变分推断（CAVI）

在均场假设下，对 $j$-th 因子的最优分布为：

$$\log q_j^*(z_j) = \mathbb{E}_{q_{-j}}[\log p(\mathbf{x}, \mathbf{z})] + \text{const}$$

其中 $q_{-j}$ 表示对所有其他因子取期望。

**算法**：轮流更新每个因子 $q_j^*$，直到收敛（ELBO不再增加）。

---

## 3.4 变分自编码器（VAE）

### VAE的概率图模型

- **先验**：$p(\mathbf{z}) = \mathcal{N}(0, I)$
- **生成模型（解码器）**：$p_\theta(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mu_\theta(\mathbf{z}), \sigma_\theta^2(\mathbf{z})I)$ 或 Bernoulli
- **推断模型（编码器）**：$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mu_\phi(\mathbf{x}), \text{diag}(\sigma_\phi^2(\mathbf{x})))$

### VAE的ELBO

$$\mathcal{L}_{VAE} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{重建损失}} - \underbrace{D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| \mathcal{N}(0,I))}_{\text{KL正则项（闭合形式）}}$$

**KL项的闭合形式**（对角高斯先验）：

$$D_{KL}(q_\phi \| p) = \frac{1}{2}\sum_{j=1}^d \left(\mu_{\phi,j}^2 + \sigma_{\phi,j}^2 - \log\sigma_{\phi,j}^2 - 1\right)$$

### 重参数化使反向传播可行

问题：重建项包含对 $q_\phi(\mathbf{z}|\mathbf{x})$ 的期望，采样操作不可微分。

解决方案（重参数化）：

$$\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

梯度通过 $\mu_\phi$ 和 $\sigma_\phi$ 反向传播，$\epsilon$ 是外部随机噪声。

---

## 3.5 重参数化梯度估计

### 两种梯度估计方法

**方法1：REINFORCE（得分函数估计）**

$$\nabla_\phi \mathbb{E}_{q_\phi}[f(\mathbf{z})] = \mathbb{E}_{q_\phi}[f(\mathbf{z})\nabla_\phi \log q_\phi(\mathbf{z})]$$

优点：通用；缺点：**方差极高**，需要大量样本。

**方法2：重参数化梯度**

$$\nabla_\phi \mathbb{E}_{q_\phi}[f(\mathbf{z})] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}[\nabla_\phi f(\mu_\phi + \sigma_\phi \odot \epsilon)]$$

优点：**低方差**；缺点：要求分布可重参数化。

实践中，重参数化梯度的方差通常比REINFORCE小1-3个数量级。

---

## 3.6 ELBO与扩散模型的联系

### 扩散模型的ELBO

扩散模型也最大化一个ELBO。考虑正向过程 $q(x_{1:T}|x_0)$，逆向过程 $p_\theta(x_{0:T})$：

$$\log p_\theta(x_0) \geq \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] = -\mathcal{L}_{DDPM}$$

展开后（第8章详细推导）：

$$\mathcal{L}_{DDPM} = \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T \approx 0} + \sum_{t=2}^T \underbrace{D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} + \underbrace{\mathbb{E}_q[-\log p_\theta(x_0|x_1)]}_{L_0}$$

**本质上与VAE相同**：
- 编码器 = 正向扩散过程（无参数）
- 解码器 = 逆向去噪过程（神经网络参数化）
- 重建项 = $L_0$
- KL项 = $\sum L_{t-1} + L_T$

---

## 代码实战

```python
"""
第三章代码实战：VAE完整实现
在MNIST上训练，可视化潜在空间
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Tuple


class Encoder(nn.Module):
    """
    VAE编码器：x -> (mu, log_var)
    
    输入: x, shape: (B, 1, 28, 28)
    输出: mu, log_var, shape: (B, latent_dim)
    """
    
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(200, latent_dim)
        self.log_var_layer = nn.Linear(200, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)                      # shape: (B, 200)
        mu = self.mu_layer(h)                # shape: (B, latent_dim)
        log_var = self.log_var_layer(h)      # shape: (B, latent_dim)
        return mu, log_var


class Decoder(nn.Module):
    """
    VAE解码器：z -> x_reconstructed
    
    输入: z, shape: (B, latent_dim)
    输出: x_recon, shape: (B, 1, 28, 28)，值域[0,1]
    """
    
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)  # shape: (B, 1, 28, 28)


class VAE(nn.Module):
    """变分自编码器"""
    
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        重参数化采样：z = mu + std * eps
        
        Args:
            mu: 编码器输出的均值，shape: (B, latent_dim)
            log_var: 编码器输出的对数方差，shape: (B, latent_dim)
        
        Returns:
            z: 采样的潜变量，shape: (B, latent_dim)
        """
        std = torch.exp(0.5 * log_var)       # shape: (B, latent_dim)
        eps = torch.randn_like(std)           # shape: (B, latent_dim)
        return mu + std * eps                 # shape: (B, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
    
    def elbo_loss(self, x: torch.Tensor, x_recon: torch.Tensor,
                  mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        计算负ELBO（训练目标为最小化）
        
        Args:
            x: 原始输入，shape: (B, 1, 28, 28)
            x_recon: 重建输出，shape: (B, 1, 28, 28)
            mu, log_var: 编码器输出，shape: (B, latent_dim)
        
        Returns:
            loss: 标量损失值（负ELBO / batch_size）
        """
        # 重建损失（二元交叉熵，假设输出为Bernoulli分布）
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # KL散度：KL(N(mu, sigma^2) || N(0, I))（闭合形式）
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return (recon_loss + kl_loss) / x.shape[0]


def train_vae(epochs: int = 10, latent_dim: int = 2, device: str = 'cpu'):
    """训练VAE并可视化结果"""
    
    # 数据加载
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # 模型和优化器
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            
            x_recon, mu, log_var = model(x)
            loss = model.elbo_loss(x, x_recon, mu, log_var)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    return model, losses


if __name__ == "__main__":
    print("训练VAE（MNIST）...")
    model, losses = train_vae(epochs=5, latent_dim=2)
    print(f"训练完成！最终损失: {losses[-1]:.4f}")
    print("\n关键联系：VAE的ELBO = 重建项 - KL项")
    print("DDPM的ELBO = -sum(KL项) - 重建项")
    print("两者在数学上具有相同的变分推断结构！")
```

---

## 本章小结

| 概念 | 公式 | 说明 |
|------|------|------|
| ELBO | $\mathcal{L} = \mathbb{E}_q[\log p(x\|z)] - D_{KL}(q(z\|x)\|p(z))$ | 对数似然的下界 |
| ELBO等式 | $\log p(x) = \mathcal{L} + D_{KL}(q\|p_\theta(\cdot\|x))$ | 等号当 $q=p_\theta$ |
| 高斯KL | $\frac{1}{2}\sum_j(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1)$ | VAE正则项闭合形式 |
| 重参数化 | $z=\mu+\sigma\odot\epsilon$，$\epsilon\sim\mathcal{N}(0,I)$ | 使梯度可反向传播 |
| 扩散ELBO | $-\mathcal{L}=\sum_t D_{KL}(q(x_{t-1}\|x_t,x_0)\|p_\theta)$ | 第8章详细推导 |

---

## 练习题

### 基础题

**3.1** 从ELBO的定义出发：$\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x,z) - \log q_\phi(z|x)]$，证明 $\log p_\theta(x) \geq \mathcal{L}$，并指出等号成立的条件。

**3.2** 对于标准VAE，计算KL散度 $D_{KL}(\mathcal{N}(\mu, \text{diag}(\sigma^2)) \| \mathcal{N}(0, I))$ 的闭合形式（维度为$d$）。

### 中级题

**3.3** 在VAE的训练中，"posterior collapse"是一个常见问题：KL项变为0，解码器忽略潜变量。解释为什么会发生这种现象，并描述至少两种缓解方法（例如：KL退火，$\beta$-VAE等）。

**3.4** 实现一个简单的VAE（可用玩具2D数据集），并对比：
   - (a) 使用重参数化梯度时的训练曲线
   - (b) 使用REINFORCE（REINFORCE梯度估计，需要绕过采样）时的训练曲线
   比较两者的收敛速度和稳定性。

### 提高题

**3.5** 推导扩散模型的完整ELBO（不需要知道DDPM，利用通用变分推断框架）：
设 $p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)$，$q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})$。
证明：$\log p_\theta(x_0) \geq -D_{KL}(q(x_T|x_0)\|p(x_T)) - \sum_{t=2}^T \mathbb{E}_q[D_{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))] + \mathbb{E}_q[\log p_\theta(x_0|x_1)]$

---

## 练习答案

**3.1** $\log p_\theta(x) = \log \int q_\phi \frac{p_\theta(x,z)}{q_\phi} dz \geq \int q_\phi \log\frac{p_\theta(x,z)}{q_\phi}dz = \mathcal{L}$（Jensen不等式）。等号成立当且仅当 $\frac{p_\theta(x,z)}{q_\phi}$ 为常数，即 $q_\phi(z|x) = p_\theta(z|x)$。

**3.2** $D_{KL} = \frac{1}{2}\sum_{j=1}^d(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1)$。

**3.3** 原因：若解码器足够强大，可以在不依赖 $z$ 的情况下重建 $x$；此时KL项被优化器推向0。缓解方法：(1) KL退火：训练初期KL权重为0，逐渐增加到1；(2) $\beta$-VAE：增大KL权重至 $\beta>1$，强制信息压缩；(3) 较弱的解码器（限制容量）。

**3.4** 见代码实战。重参数化梯度通常收敛快5-10倍，REINFORCE方差极大。

**3.5** 提示：分子分母同乘以 $q(x_{1:T}|x_0)$，利用全概率分解，应用Jensen不等式，然后逐项整理。关键步骤是利用马尔科夫性将 $q(x_{1:T}|x_0)$ 分解。

---

## 延伸阅读

1. **Kingma & Welling (2013)**. *Auto-Encoding Variational Bayes* — VAE原始论文
2. **Blei et al. (2017)**. *Variational Inference: A Review for Statisticians* — 全面综述
3. **Lucas et al. (2019)**. *Don't Blame the ELBO! A Linear VAE Perspective* — ELBO的深入分析

---

[← 上一章：高斯分布与马尔科夫链](./02-gaussian-markov-chains.md)

[下一章：生成模型全景与VAE回顾 →](../part2-generative-models/04-generative-models-overview-vae.md)

[返回目录](../README.md)
