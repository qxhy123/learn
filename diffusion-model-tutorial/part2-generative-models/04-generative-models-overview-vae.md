# 第四章：生成模型全景与VAE回顾

> **本章导读**：扩散模型是众多生成模型中的一员。理解各类生成模型的优缺点，能帮助你深刻理解扩散模型为何能在2020年后成为主流。本章系统梳理生成模型家族，并深化对VAE的理解。

**前置知识**：前三章内容，卷积神经网络基础
**预计学习时间**：90分钟

---

## 学习目标

完成本章学习后，你将能够：
1. 区分显式密度和隐式密度生成模型，理解各自的优缺点
2. 解释自回归模型的原理和局限性
3. 分析GAN的训练困难（模式崩塌、梯度消失），理解Wasserstein改进
4. 深化VAE理解：$\beta$-VAE、VQ-VAE、层次VAE
5. 准确说明扩散模型相比其他生成模型的核心优势

---

## 4.1 生成模型分类

### 显式密度 vs 隐式密度

| 类别 | 子类 | 代表方法 | 优点 | 缺点 |
|------|------|----------|------|------|
| **显式密度** | 精确 | PixelCNN, GPT | 精确似然，训练稳定 | 序列生成慢 |
| | 近似 | VAE, DDPM | 理论完备，多样性好 | 样本质量略逊 |
| | 可解析 | 归一化流 | 精确似然+快速采样 | 可逆约束限制模型 |
| **隐式密度** | — | GAN | 样本质量高，速度快 | 训练不稳定，模式崩塌 |

### 扩散模型的定位

扩散模型属于**显式密度模型（近似类）**：
- 通过优化变分下界（ELBO）最大化数据似然
- 训练稳定，生成质量高，多样性强
- 代价：采样需要多步（通常50-1000步）

---

## 4.2 自回归模型

### 链式法则分解

自回归模型将联合分布分解为条件分布的乘积：

$$p(\mathbf{x}) = \prod_{i=1}^d p(x_i | x_1, ..., x_{i-1})$$

**图像**：像素按光栅扫描顺序，每个像素依赖之前所有像素（PixelCNN）。

**文本**：每个词依赖之前所有词（GPT系列）。

### 优缺点

**优点**：
- 精确计算对数似然 $\log p(\mathbf{x})$
- 训练稳定（交叉熵损失）
- 模型容量大（Transformer）

**缺点**：
- 采样是**顺序的**：必须逐维生成，无法并行
- 对于 $512\times512$ 图像：需要 $786432$ 步！
- 全局一致性差（早期错误会累积）

---

## 4.3 GAN与其困难

### GAN的基本框架

生成器 $G_\theta: z \to x$（从噪声生成数据），判别器 $D_\phi: x \to [0,1]$（区分真假）。

**目标**（零和博弈）：

$$\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1-D(G(z)))]$$

### 训练困难

**1. 模式崩塌（Mode Collapse）**：生成器只学会生成少数类型的样本（几个"众数"），忽视数据分布的多样性。

**2. 梯度消失**：当判别器过强时，$\log(1-D(G(z))) \approx 0$，梯度消失，生成器无法学习。

**3. 训练不稳定**：生成器和判别器的博弈容易陷入振荡，超参数极为敏感。

### Wasserstein GAN（WGAN）

WGAN用Wasserstein距离替代JS散度，提供更平滑的梯度：

$$W(p, q) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p}[f(x)] - \mathbb{E}_{x \sim q}[f(x)]$$

其中 $f$ 是1-Lipschitz函数（用梯度裁剪或梯度惩罚保证）。

WGAN改善了训练稳定性，但模式崩塌和训练技巧问题依然存在。

---

## 4.4 VAE深入

### $\beta$-VAE

在标准VAE的基础上，增强KL正则项的权重：

$$\mathcal{L}_{\beta-VAE} = \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) \| p(z)), \quad \beta > 1$$

**效果**：
- $\beta$ 越大：潜在表示越解耦（每个维度独立控制一个语义因素），但重建质量略降
- Higgins et al. 2017证明 $\beta > 1$ 有助于学习解耦表示

### VQ-VAE（离散潜在空间）

标准VAE使用连续潜变量；VQ-VAE使用**离散码本**（codebook）：

1. 编码器输出连续向量 $z_e$，shape: $(B, H, W, d)$
2. 在码本 $\{e_k\}_{k=1}^K$ 中找最近邻：$z_q = e_{k^*}$，$k^* = \arg\min_k \|z_e - e_k\|$
3. 解码器从 $z_q$ 重建（直通梯度估计：将梯度直接复制到 $z_e$）

**优势**：离散潜在空间更适合文本、音频等离散数据；码本提供紧凑表示。

**损失函数**：

$$\mathcal{L} = \|x - \hat{x}\|^2 + \|sg(z_e) - e\|^2 + \gamma\|z_e - sg(e)\|^2$$

其中 $sg$ 表示停止梯度（stop-gradient）。

### 层次VAE（HVAE）

通过多层潜变量建立层次化的先验：

$$p(x, z_1, ..., z_L) = p(x|z_1)p(z_1|z_2)\cdots p(z_L)$$

每层潜变量捕获不同粒度的特征（从全局语义到局部细节）。VDVAE（Very Deep VAE）使用数十层潜变量，生成质量接近GAN。

---

## 4.5 归一化流

### 可逆变换

归一化流通过一系列可逆变换将简单分布（如高斯）变换为复杂数据分布：

$$x = f(z), \quad z = f^{-1}(x), \quad z \sim p_z$$

精确对数似然（变量替换公式）：

$$\log p(x) = \log p_z(f^{-1}(x)) + \log|\det J_{f^{-1}}(x)|$$

**关键挑战**：计算Jacobian行列式必须高效（$O(d)$ 而非 $O(d^3)$）。

### Real-NVP（实值非体积保持）

通过仿射耦合层（Affine Coupling Layer）实现高效的行列式计算：

$$x_{1:d/2} = z_{1:d/2}$$
$$x_{d/2+1:d} = z_{d/2+1:d} \odot \exp(s(z_{1:d/2})) + t(z_{1:d/2})$$

行列式为 $\exp(\sum_i s_i)$（对角矩阵），$O(d)$ 计算。

**局限**：可逆约束限制了模型的表达能力；无法使用标准卷积等非可逆操作。

---

## 4.6 为什么扩散模型胜出

### 对比表格

| 指标 | 自回归 | GAN | VAE | 归一化流 | **扩散模型** |
|------|--------|-----|-----|----------|------------|
| 样本质量 | 中 | 高 | 中低 | 中 | **极高** |
| 生成多样性 | 高 | 低（模式崩塌） | 高 | 高 | **高** |
| 训练稳定性 | 高 | 低 | 高 | 中 | **高** |
| 精确似然 | ✓ | ✗ | 近似 | ✓ | 近似 |
| 采样速度 | 极慢 | 极快 | 快 | 快 | **慢（可优化）** |
| 条件生成 | 中等 | 困难 | 中等 | 困难 | **极佳（CFG）** |
| 理论完备性 | 高 | 低 | 高 | 高 | **高** |

### 扩散模型的关键优势

1. **训练目标简单**：噪声预测是一个简单的回归任务，无对抗训练
2. **天然支持条件生成**：CFG只需在推理时修改采样，无需重新训练
3. **分层次生成**：不同时间步对应不同频率的特征，自然形成层次化
4. **可扩展性强**：U-Net→DiT的转变证明了Scaling Law的有效性

---

## 代码实战

```python
"""
第四章代码实战：生成模型比较
比较简单的VAE和GAN在二维Moon数据集上的表现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np


def get_moon_data(n: int = 1000) -> torch.Tensor:
    """生成二维月牙形数据集"""
    X, _ = make_moons(n_samples=n, noise=0.1)
    return torch.FloatTensor(X)


# ============================================================
# 简单VAE（2D数据）
# ============================================================

class VAE2D(nn.Module):
    """用于2D数据的简单VAE"""
    
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.mu_head = nn.Linear(64, latent_dim)
        self.logvar_head = nn.Linear(64, latent_dim)
        
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2),
        )
    
    def encode(self, x):
        h = self.encoder_net(x)
        return self.mu_head(h), self.logvar_head(h)
    
    def decode(self, z):
        return self.decoder_net(z)
    
    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss(self, x, recon, mu, logvar):
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        return (recon_loss + kl_loss) / x.shape[0]
    
    @torch.no_grad()
    def generate(self, n: int) -> torch.Tensor:
        z = torch.randn(n, 2)
        return self.decode(z)


# ============================================================
# 简单GAN（2D数据）
# ============================================================

class Generator(nn.Module):
    def __init__(self, z_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2),
        )
    
    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        return self.net(x)


def train_models(n_epochs: int = 500):
    """训练并比较VAE和GAN"""
    data = get_moon_data(2000)
    
    # 训练VAE
    vae = VAE2D(latent_dim=2)
    vae_opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        idx = torch.randint(0, len(data), (256,))
        x_batch = data[idx]
        recon, mu, logvar = vae(x_batch)
        loss = vae.loss(x_batch, recon, mu, logvar)
        vae_opt.zero_grad()
        loss.backward()
        vae_opt.step()
    
    # 训练GAN
    G = Generator(z_dim=2)
    D = Discriminator()
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    for epoch in range(n_epochs):
        idx = torch.randint(0, len(data), (256,))
        real = data[idx]
        z = torch.randn(256, 2)
        fake = G(z)
        
        # 训练判别器
        d_loss = -torch.mean(D(real)) + torch.mean(D(fake.detach()))
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()
        
        # 训练生成器
        g_loss = -torch.mean(D(G(torch.randn(256, 2))))
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()
    
    # 可视化比较
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].scatter(*data.T.numpy(), s=5, alpha=0.5)
    axes[0].set_title('真实数据（月牙形）')
    
    vae_samples = vae.generate(1000).numpy()
    axes[1].scatter(*vae_samples.T, s=5, alpha=0.5, c='blue')
    axes[1].set_title('VAE生成')
    
    with torch.no_grad():
        gan_samples = G(torch.randn(1000, 2)).numpy()
    axes[2].scatter(*gan_samples.T, s=5, alpha=0.5, c='green')
    axes[2].set_title('GAN生成')
    
    plt.tight_layout()
    plt.savefig('generative_comparison.png', dpi=100)
    print("对比图已保存")


if __name__ == "__main__":
    train_models(epochs=300)
```

---

## 本章小结

| 模型 | 训练目标 | 采样速度 | 模式覆盖 | 训练稳定性 |
|------|----------|----------|----------|------------|
| 自回归 | $\log p(x)$ | 极慢（顺序） | 好 | 高 |
| GAN | 对抗博弈 | 极快（单步） | 差 | 低 |
| VAE | ELBO | 快 | 好 | 高 |
| 归一化流 | 精确$\log p(x)$ | 快 | 好 | 中 |
| **扩散模型** | **噪声预测MSE** | **中（多步）** | **极好** | **高** |

---

## 练习题

### 基础题

**4.1** GAN的纳什均衡状态是什么？在均衡状态下，$D(x) = ?$，$G$ 的分布是什么？

**4.2** 解释为什么VAE生成的图像有时会"模糊"，而GAN生成的图像通常更清晰。从损失函数的角度分析。

### 中级题

**4.3** 实现VQ-VAE的直通梯度估计器：`z_q = z_e + (z_q_sg - z_e).detach()`，解释这行代码如何将梯度从 $z_q$ 传递到 $z_e$。

**4.4** 对比实验：在相同的2D数据集上训练VAE和WGAN，分析：(a) 哪个模型更好地覆盖所有模式？(b) 哪个训练更稳定？

### 提高题

**4.5** 研究题：$\beta$-VAE声称更大的$\beta$有助于学习解耦表示。设计一个实验验证这一点：使用2D数据集（如旋转的两个高斯混合），训练不同$\beta$的VAE，用互信息或相关系数量化解耦程度。

---

## 练习答案

**4.1** 均衡状态：$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} = \frac{1}{2}$（对所有$x$），此时$p_g = p_{data}$，生成器完美模拟真实分布。

**4.2** VAE使用像素级MSE/BCE重建损失：$\mathbb{E}[||x - \hat{x}||^2]$。最小化MSE等价于预测所有可能重建的**均值**，导致模糊。GAN无显式重建损失，直接优化判别器，可以产生清晰样本，但多样性差。

**4.3** `z_q = z_e + (z_q_sg - z_e).detach()`：前向时值为`z_q_sg`（码本映射结果），反向时`.detach()`部分梯度为0，因此梯度 $\nabla_{z_q} L$ 直接传到 $\nabla_{z_e} L$，实现"直通"。

**4.4** 见代码实战（上方`train_models`函数可扩展为WGAN）。

**4.5** 解耦量化方法：对每个潜变量维度单独变化，观察生成图像变化是否只影响单一语义因素。互信息 $I(z_i; v_j)$ 应接近0（$i \neq j$ 时），其中 $v_j$ 是真实生成因子。

---

## 延伸阅读

1. **Goodfellow et al. (2014)**. *Generative Adversarial Nets* — GAN原始论文
2. **Arjovsky et al. (2017)**. *Wasserstein GAN* — WGAN
3. **Oord et al. (2017)**. *Neural Discrete Representation Learning* — VQ-VAE
4. **Dhariwal & Nichol (2021)**. *Diffusion Models Beat GANs on Image Synthesis* — 扩散超越GAN

---

[← 上一章：变分推断基础](../part1-math-foundations/03-variational-inference.md)

[下一章：基于分数的生成模型 →](./05-score-based-generative-models.md)

[返回目录](../README.md)
