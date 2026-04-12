# 附录C：习题答案（详解）

> 本附录收录各章提高题的详细解答，包含完整的数学推导和代码实现。基础题与中级题的简要答案已在各章末尾给出。

---

## 第一部分：基础数学

### 第1章：概率论与信息论

**1.5（提高题）** 证明高斯分布之间的KL散度闭式公式，并分析维度 $d$ 对KL值的影响。

**证明**：设 $p = \mathcal{N}(\mu_1, \Sigma_1)$，$q = \mathcal{N}(\mu_2, \Sigma_2)$，

$$D_{KL}(p\|q) = \mathbb{E}_p\left[\log\frac{p(x)}{q(x)}\right]$$

展开对数项：

$$\log\frac{p}{q} = -\frac{1}{2}(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1) + \frac{1}{2}(x-\mu_2)^T\Sigma_2^{-1}(x-\mu_2) + \frac{1}{2}\log\frac{|\Sigma_2|}{|\Sigma_1|}$$

在 $p$ 下取期望，利用 $\mathbb{E}_p[(x-\mu_1)(x-\mu_1)^T] = \Sigma_1$：

$$D_{KL}(p\|q) = \frac{1}{2}\left[\text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^T\Sigma_2^{-1}(\mu_2-\mu_1) - d + \log\frac{|\Sigma_2|}{|\Sigma_1|}\right]$$

**维度分析**：对角情形 $p = \mathcal{N}(0, \sigma^2 I)$，$q = \mathcal{N}(0, I)$：

$$D_{KL}(p\|q) = \frac{d}{2}(\sigma^2 - \log\sigma^2 - 1)$$

KL值与维度 $d$ 线性增长。当 $\sigma \neq 1$ 时，高维空间中分布差异被放大。

---

### 第2章：变分自编码器

**2.5（提高题）** 实现层次VAE（HVAE）：两层潜在变量 $z_1, z_2$，其中 $p(z_1|z_2)$ 是条件高斯。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalVAE(nn.Module):
    """两层层次VAE
    潜在层次：z2（高层，全局结构）→ z1（低层，细节）→ x
    后验：q(z1,z2|x) = q(z2|x)·q(z1|x,z2)
    """
    
    def __init__(self, input_dim: int = 784, h_dim: int = 256,
                 z1_dim: int = 32, z2_dim: int = 16):
        super().__init__()
        
        # 编码器：x → z2（高层语义）
        self.enc_shared = nn.Sequential(
            nn.Linear(input_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
        )
        self.enc_z2_mu = nn.Linear(h_dim, z2_dim)
        self.enc_z2_logvar = nn.Linear(h_dim, z2_dim)
        
        # 编码器：(x, z2) → z1（低层细节）
        self.enc_z1_net = nn.Sequential(
            nn.Linear(h_dim + z2_dim, h_dim), nn.ReLU(),
        )
        self.enc_z1_mu = nn.Linear(h_dim, z1_dim)
        self.enc_z1_logvar = nn.Linear(h_dim, z1_dim)
        
        # 先验：z2 → p(z1|z2)
        self.prior_z1_net = nn.Sequential(
            nn.Linear(z2_dim, h_dim), nn.ReLU(),
        )
        self.prior_z1_mu = nn.Linear(h_dim, z1_dim)
        self.prior_z1_logvar = nn.Linear(h_dim, z1_dim)
        
        # 解码器：(z1, z2) → x
        self.decoder = nn.Sequential(
            nn.Linear(z1_dim + z2_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, input_dim),
        )
    
    def encode(self, x):
        h = self.enc_shared(x)
        
        # q(z2|x)
        z2_mu = self.enc_z2_mu(h)
        z2_logvar = self.enc_z2_logvar(h)
        z2 = self.reparameterize(z2_mu, z2_logvar)
        
        # q(z1|x, z2)
        h1 = self.enc_z1_net(torch.cat([h, z2], dim=-1))
        z1_mu = self.enc_z1_mu(h1)
        z1_logvar = self.enc_z1_logvar(h1)
        z1 = self.reparameterize(z1_mu, z1_logvar)
        
        return z1, z1_mu, z1_logvar, z2, z2_mu, z2_logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        z1, z1_mu, z1_logvar, z2, z2_mu, z2_logvar = self.encode(x)
        
        # 先验 p(z1|z2)
        h_prior = self.prior_z1_net(z2)
        prior_z1_mu = self.prior_z1_mu(h_prior)
        prior_z1_logvar = self.prior_z1_logvar(h_prior)
        
        # 重建
        x_recon = self.decoder(torch.cat([z1, z2], dim=-1))
        
        # ELBO = -重建损失 - KL(q(z2)||p(z2)) - KL(q(z1|x,z2)||p(z1|z2))
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum')
        
        # KL(z2)：相对于标准正态
        kl_z2 = -0.5 * torch.sum(1 + z2_logvar - z2_mu.pow(2) - z2_logvar.exp())
        
        # KL(z1)：相对于条件先验 p(z1|z2)
        kl_z1 = self.gaussian_kl(z1_mu, z1_logvar, prior_z1_mu, prior_z1_logvar)
        
        loss = recon_loss + kl_z2 + kl_z1
        return loss, recon_loss, kl_z2, kl_z1
    
    def gaussian_kl(self, mu1, logvar1, mu2, logvar2):
        """KL(N(mu1,var1) || N(mu2,var2))"""
        return 0.5 * torch.sum(
            logvar2 - logvar1 + (logvar1.exp() + (mu1-mu2).pow(2)) / logvar2.exp() - 1
        )
```

**结果分析**：层次VAE的ELBO = -重建损失 - KL(z2) - KL(z1|z2)。z2捕获全局语义（如数字类别），z1捕获局部细节（如笔画粗细）。两层KL都需要监控，避免后验坍缩。

---

### 第3章：扩散模型数学基础

**3.5（提高题）** 推导扩散模型后验 $q(x_{t-1}|x_t, x_0)$ 的均值 $\tilde\mu_t$，验证其为计算最优去噪方向的充分统计量。

**推导**：

利用贝叶斯公式和马尔可夫性质：

$$q(x_{t-1}|x_t, x_0) \propto q(x_t|x_{t-1}) \cdot q(x_{t-1}|x_0)$$

其中：
- $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, \beta_t I)$
- $q(x_{t-1}|x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar\alpha_{t-1}}x_0, (1-\bar\alpha_{t-1})I)$

两个高斯的乘积仍为高斯（在 $x_{t-1}$ 上）。完成配方：

$$\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$

$$\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$

**充分统计量分析**：用 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ 代入，得到 $\epsilon$-预测等价形式：

$$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right)$$

这表明：给定 $x_t$，预测 $\epsilon$ 与预测 $x_0$ 在最小化后验均值MSE意义上等价。$\tilde\mu_t$ 包含了从 $x_t$ 去噪到 $x_{t-1}$ 所需的全部信息。

---

## 第二部分：DDPM与Score Matching

### 第4章：DDPM

**4.5（提高题）** 实现连续时间扩散模型（VP-SDE），并验证其在离散极限下与DDPM等价。

```python
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp

class VPSDEScheduler:
    """连续时间VP-SDE调度器
    dX = -0.5·β(t)·X dt + sqrt(β(t)) dW
    β(t) = β_min + t·(β_max - β_min)，t∈[0,1]
    """
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """E[x_t|x_0] = sqrt(alpha_bar(t)) * x_0"""
        # 对线性β积分: ∫_0^t β(s)ds = β_min*t + 0.5*(β_max-β_min)*t²
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
        return torch.exp(-0.5 * integral)
    
    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor):
        """q(x_t|x_0) = N(mean, std²I)"""
        ab = self.alpha_bar(t)[:, None, None, None]
        mean = ab.sqrt() * x0
        std = (1 - ab).sqrt()
        return mean, std
    
    def forward_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean, std = self.marginal_prob(x0, t)
        eps = torch.randn_like(x0)
        return mean + std * eps, eps
    
    def ddpm_discretize(self, T: int = 1000):
        """离散化验证：VP-SDE ↔ DDPM"""
        ts = torch.linspace(0, 1, T + 1)
        alpha_bars = self.alpha_bar(ts)
        
        # DDPM betas: beta_t ≈ 1 - alpha_bar_t / alpha_bar_{t-1}
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        return betas.clamp(0, 0.999)


class ProbabilityFlowODE:
    """概率流ODE采样（确定性，等价于DDIM η=0）
    dx/dt = -0.5·β(t)·x - 0.5·β(t)·score(x,t)
    """
    
    def __init__(self, score_fn, scheduler: VPSDEScheduler):
        self.score_fn = score_fn
        self.scheduler = scheduler
    
    def ode_fn(self, t_scalar, x_flat, shape):
        """scipy求解器接口"""
        t = torch.tensor([t_scalar], dtype=torch.float32)
        x = torch.tensor(x_flat, dtype=torch.float32).reshape(shape)
        
        beta_t = self.scheduler.beta(t).item()
        score = self.score_fn(x, t)
        
        dxdt = -0.5 * beta_t * (x + score)
        return dxdt.flatten().numpy()
    
    def sample(self, shape, n_steps: int = 100):
        """从t=1反向积分到t=0"""
        # 初始噪声
        x_T = torch.randn(shape)
        t_span = [1.0, 1e-3]  # 从T到0（避免t=0奇点）
        
        sol = solve_ivp(
            lambda t, x: self.ode_fn(t, x, shape),
            t_span, x_T.flatten().numpy(),
            method='RK45', dense_output=False,
            t_eval=np.linspace(1.0, 1e-3, n_steps)
        )
        x_0 = torch.tensor(sol.y[:, -1], dtype=torch.float32).reshape(shape)
        return x_0
```

**等价性验证**：当 $T \to \infty$ 时，VP-SDE的离散化 $\beta_t = 1 - \bar\alpha_t / \bar\alpha_{t-1}$ 恰好还原为线性噪声调度。概率流ODE的Euler离散化 = DDIM（$\eta=0$）。

---

### 第5章：Score Matching

**5.5（提高题）** 实现Sliced Score Matching，对比与DSM的计算效率和估计方差。

```python
def sliced_score_matching_loss(score_fn, x: torch.Tensor, n_slices: int = 1):
    """
    Sliced Score Matching（Song et al. 2020）
    避免计算完整Hessian，用随机投影估计迹：
    L_SSM = E_v[v^T·∇_x s(x)·v + 0.5·||s(x)||²]
    其中v为随机单位向量（Rademacher或高斯）
    
    Args:
        score_fn: 分数网络 s_θ(x)
        x: 训练数据，shape (B, D)
        n_slices: 随机投影数量（越多方差越小）
    
    Returns:
        scalar loss
    """
    x = x.requires_grad_(True)
    score = score_fn(x)  # shape: (B, D)
    
    loss = 0.5 * (score ** 2).sum(dim=-1).mean()  # ||s||²项
    
    for _ in range(n_slices):
        # Rademacher随机向量（±1）
        v = torch.randint_like(x, low=0, high=2).float() * 2 - 1  # shape: (B, D)
        v = v / v.norm(dim=-1, keepdim=True)
        
        # 计算 v^T·J·v（无需完整Jacobian）
        sv = (score * v).sum(dim=-1)  # shape: (B,)
        grad = torch.autograd.grad(sv.sum(), x, create_graph=True)[0]  # shape: (B, D)
        
        loss += (v * grad).sum(dim=-1).mean()
    
    return loss / n_slices


def compare_score_matching():
    """对比SSM和DSM的效率"""
    import time
    
    D, B = 784, 64
    x = torch.randn(B, D)
    score_fn = nn.Sequential(nn.Linear(D, 256), nn.SiLU(), nn.Linear(256, D))
    
    # DSM
    sigma = 0.1
    x_noisy = x + sigma * torch.randn_like(x)
    start = time.time()
    score_pred = score_fn(x_noisy)
    dsm_loss = 0.5 * ((score_pred + (x_noisy - x) / sigma**2)**2).sum(-1).mean()
    dsm_time = time.time() - start
    
    # SSM
    start = time.time()
    ssm_loss = sliced_score_matching_loss(score_fn, x, n_slices=1)
    ssm_time = time.time() - start
    
    print(f"DSM: loss={dsm_loss.item():.4f}, time={dsm_time*1000:.1f}ms")
    print(f"SSM: loss={ssm_loss.item():.4f}, time={ssm_time*1000:.1f}ms")
    print(f"SSM比DSM慢约 {ssm_time/dsm_time:.1f}x（需要autograd二阶）")
    print("但SSM不需要噪声，适用于任意未知数据分布")
```

---

### 第6章：DDPM实现

**6.5（提高题）** 对比x0-预测与ε-预测的训练稳定性，分析SNR加权对不同时间步学习的影响。

**分析**：

训练目标的等价推导：

| 预测目标 | 损失函数 | 隐含的SNR加权 |
|---------|---------|-------------|
| $\epsilon$-预测 | $\mathbb{E}[\|\epsilon - \epsilon_\theta\|^2]$ | 均匀（各时间步权重相同） |
| $x_0$-预测 | $\mathbb{E}[\|x_0 - \hat x_0\|^2]$ | 按 $\text{SNR}(t) = \bar\alpha_t/(1-\bar\alpha_t)$ 加权 |
| $v$-预测 | $\mathbb{E}[\|v - v_\theta\|^2]$ | 均匀（类似ε） |

```python
def analyze_snr_weighting():
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T)
    alpha_bars = torch.cumprod(1 - betas, dim=0)
    snr = alpha_bars / (1 - alpha_bars)
    
    print("SNR值范围：")
    print(f"  t=1 (最小噪声): SNR={snr[0]:.1f}")
    print(f"  t=500: SNR={snr[499]:.3f}")
    print(f"  t=1000 (最大噪声): SNR={snr[-1]:.6f}")
    
    # x0-预测的隐含加权
    x0_weights = snr  # 高SNR时间步（低噪声）权重大
    
    # ε-预测等价的x0预测加权 = SNR²/(1+SNR)²（推导略）
    eps_equiv_weights = snr**2 / (1 + snr)**2
    
    print("\n低噪声区(t<200)权重占比：")
    print(f"  x0-预测: {x0_weights[:200].sum()/x0_weights.sum():.1%}")
    print(f"  ε-预测: {eps_equiv_weights[:200].sum()/eps_equiv_weights.sum():.1%}")

# 结论：x0-预测对低噪声时间步的权重极高（>99%），导致忽略高噪声步骤；
# ε-预测更均匀，训练更稳定；Min-SNR截断（Hang et al. 2023）是折中方案。
```

---

## 第三部分：进阶理论

### 第7章：潜在扩散模型

**7.5（提高题）** 分析VAE潜在空间的分布对扩散模型训练的影响，推导KL正则系数 $\lambda_{KL}$ 的最优选择。

**分析**：

LDM的总损失：

$$\mathcal{L}_{LDM} = \mathbb{E}_{t,z,\epsilon}\left[\|\epsilon - \epsilon_\theta(z_t, t)\|^2\right] + \lambda_{KL} \cdot D_{KL}(q_\phi(z|x)\|p(z))$$

**KL系数的影响**：
- $\lambda_{KL}$ 过大：VAE潜在空间被强制为标准正态，压缩过度，细节丢失（信息瓶颈增强）
- $\lambda_{KL}$ 过小：潜在空间不规则，噪声调度不匹配（$q(z) \neq \mathcal{N}(0,I)$）

**最优选择（经验规律）**：Rombach et al. 使用 $\lambda_{KL} = 10^{-6}$（极小值），主要依靠感知损失 $\mathcal{L}_{LPIPS}$ 约束潜在空间质量；另外用调度缩放因子 `scaling_factor ≈ 0.18`（SD1.5）补偿潜在空间的实际方差不为1。

```python
def calibrate_scaling_factor(vae, dataloader, n_batches=100):
    """校准VAE潜在空间的缩放因子
    使得 z/scaling_factor ~ N(0, I)
    """
    all_stds = []
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            z = vae.encode(x).latent_dist.sample()
            all_stds.append(z.std().item())
    
    mean_std = sum(all_stds) / len(all_stds)
    scaling_factor = mean_std
    print(f"建议 scaling_factor = {scaling_factor:.4f}")
    print(f"（SD1.5实际值为 0.18215）")
    return scaling_factor
```

---

### 第8章：SDE框架

**8.5（提高题）** 推导VP-SDE的概率流ODE，实现Euler-Maruyama（随机）和RK45（确定性）两种采样器，对比生成质量与多样性。

```python
class VPSDESampler:
    """VP-SDE的两种采样器"""
    
    def __init__(self, score_fn, beta_min=0.1, beta_max=20.0, T=1.0):
        self.score_fn = score_fn
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
    
    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def euler_maruyama(self, shape, n_steps=1000, eps=1e-3):
        """随机Euler-Maruyama（DDPM等价）"""
        dt = -(self.T - eps) / n_steps
        x = torch.randn(shape)
        
        for i in range(n_steps):
            t_val = self.T + i * dt
            t = torch.full((shape[0],), t_val)
            
            beta_t = self.beta(torch.tensor([t_val])).item()
            score = self.score_fn(x, t)
            
            # 逆向SDE: dx = [-0.5β(t)x - β(t)·score]dt + sqrt(β(t))·dW
            drift = (-0.5 * beta_t * x - beta_t * score) * (-dt)
            diffusion = (beta_t * (-dt)).sqrt() * torch.randn_like(x)
            x = x + drift + diffusion
        
        return x
    
    def probability_flow_ode(self, shape, n_steps=100, eps=1e-3):
        """确定性概率流ODE（DDIM等价）
        dx/dt = -0.5·β(t)·[x + score(x,t)]
        """
        dt = -(self.T - eps) / n_steps
        x = torch.randn(shape)
        
        for i in range(n_steps):
            t_val = self.T + i * dt
            t = torch.full((shape[0],), t_val)
            
            beta_t = self.beta(torch.tensor([t_val])).item()
            score = self.score_fn(x, t)
            
            # ODE（无扩散项）
            drift = -0.5 * beta_t * (x + score) * (-dt)
            x = x + drift
        
        return x

# 对比分析：
# Euler-Maruyama（随机）：每次生成结果不同，多样性高
# 概率流ODE（确定性）：固定噪声→固定输出，可用于插值
# 相同步数下，ODE通常需要更少步骤达到相同质量（因为无扩散项不放大误差）
```

---

### 第9章：NCSN与连续分数匹配

**9.5（提高题）** 实现多尺度NCSN，分析噪声尺度序列的设计对生成质量的影响（几何级数 vs 线性级数）。

```python
class MultiScaleNCSN(nn.Module):
    """多尺度NCSN（Noise Conditional Score Network）
    在L个不同噪声水平上估计分数函数
    """
    
    def __init__(self, dim: int, sigma_min: float = 0.01,
                 sigma_max: float = 50.0, L: int = 10,
                 schedule: str = 'geometric'):
        super().__init__()
        self.dim = dim
        
        if schedule == 'geometric':
            # 几何级数（Song & Ermon 2019推荐）
            self.sigmas = torch.exp(
                torch.linspace(np.log(sigma_min), np.log(sigma_max), L)
            )
        else:
            # 线性级数（分布不均匀，高噪声级别过多）
            self.sigmas = torch.linspace(sigma_min, sigma_max, L)
        
        # 共享网络（σ通过时间嵌入注入）
        self.sigma_embed = nn.Embedding(L, 32)
        self.net = nn.Sequential(
            nn.Linear(dim + 32, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, dim),
        )
    
    def forward(self, x: torch.Tensor, sigma_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 加噪数据，shape (B, dim)
            sigma_idx: 噪声水平索引，shape (B,)
        Returns:
            score: 归一化分数 s(x,σ) = score_raw / σ
        """
        sigma_emb = self.sigma_embed(sigma_idx)          # shape: (B, 32)
        h = torch.cat([x, sigma_emb], dim=-1)             # shape: (B, dim+32)
        score_raw = self.net(h)                           # shape: (B, dim)
        
        # 归一化：s(x,σ) * σ 的尺度不依赖σ
        sigmas = self.sigmas[sigma_idx][:, None]          # shape: (B, 1)
        return score_raw / sigmas
    
    def dsm_loss(self, x: torch.Tensor) -> torch.Tensor:
        """多尺度去噪分数匹配损失"""
        B = x.shape[0]
        L = len(self.sigmas)
        
        # 随机选择噪声水平
        idx = torch.randint(0, L, (B,))
        sigmas = self.sigmas[idx][:, None]                # shape: (B, 1)
        
        # 加噪
        eps = torch.randn_like(x)
        x_noisy = x + sigmas * eps                       # shape: (B, dim)
        
        # 分数估计 vs 真实分数（-eps/sigma）
        score_pred = self(x_noisy, idx)                   # shape: (B, dim)
        score_true = -eps / sigmas                        # shape: (B, dim)
        
        # 加权损失（权重 = sigma²，使各尺度贡献均等）
        loss = (sigmas**2 * (score_pred - score_true)**2).sum(-1).mean()
        return loss
```

**噪声尺度分析**：
- 几何级数：相邻噪声比率恒定，Langevin采样时每步接受率均匀，实践效果更好
- 线性级数：低噪声区间密度不足，模型难以学到精细细节；高噪声区间过密，浪费算力

---

## 第四部分：采样加速

### 第10章：DDIM

**10.5（提高题）** 推导DDIM的连接到概率流ODE：证明当步长 $\Delta t \to 0$ 时，DDIM更新公式收敛到概率流ODE的Euler离散化。

**证明**：

DDIM更新（$\sigma_t = 0$）：

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\hat x_0 + \sqrt{1-\bar\alpha_{t-1}}\epsilon_\theta(x_t, t)$$

其中 $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta) / \sqrt{\bar\alpha_t}$。

连续时间变量替换：令 $\alpha(t) = \bar\alpha_t$，$\sigma(t) = \sqrt{1-\bar\alpha_t}$，则：

$$x_{t-\Delta t} = \frac{\alpha(t-\Delta t)}{\alpha(t)}x_t + \left(\sigma(t-\Delta t) - \frac{\alpha(t-\Delta t)\sigma(t)}{\alpha(t)}\right)\epsilon_\theta$$

令 $\Delta t \to 0$，利用导数定义，得到：

$$\frac{dx}{dt} = \frac{d\log\alpha}{dt}x - \left(\sigma\frac{d\log\alpha}{dt} - \frac{d\sigma}{dt}\right)\epsilon_\theta$$

代入 $\epsilon_\theta = -\sigma \cdot \nabla_x \log p_t(x)$（分数函数关系），化简后得到：

$$\frac{dx}{dt} = \mathbf{f}(x,t) - \frac{g(t)^2}{2}\nabla_x \log p_t(x)$$

这正是VP-SDE对应的概率流ODE，证毕。

---

### 第11章：噪声调度设计

**11.5（提高题）** 实现EDM（Karras et al. 2022）的完整训练和推理流程，包括预条件化和Heun采样器。

```python
class EDMPreconditioned(nn.Module):
    """EDM预条件化包装器
    网络实际预测: D(x; σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x; c_noise(σ))
    """
    
    def __init__(self, backbone: nn.Module,
                 sigma_data: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.sigma_data = sigma_data
    
    def precond_params(self, sigma: torch.Tensor):
        s_d = self.sigma_data
        c_skip = s_d**2 / (sigma**2 + s_d**2)
        c_out = sigma * s_d / (sigma**2 + s_d**2).sqrt()
        c_in = 1 / (sigma**2 + s_d**2).sqrt()
        c_noise = sigma.log() / 4
        return c_skip, c_out, c_in, c_noise
    
    def forward(self, x_noisy, sigma):
        # sigma: shape (B,) → (B, 1, 1, 1)
        s = sigma[:, None, None, None]
        c_skip, c_out, c_in, c_noise = self.precond_params(s)
        
        # 预条件化输入
        x_in = c_in * x_noisy
        
        # 骨干网络（输出形状与输入相同）
        F_out = self.backbone(x_in, c_noise.squeeze())
        
        # 预条件化输出
        return c_skip * x_noisy + c_out * F_out
    
    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """EDM训练损失（对σ的对数均匀采样）"""
        B = x0.shape[0]
        
        # 对数正态采样σ
        log_sigma = torch.randn(B) * 1.2 - 1.2  # P_mean=-1.2, P_std=1.2
        sigma = log_sigma.exp()[:, None, None, None]  # shape: (B, 1, 1, 1)
        
        # 加噪
        noise = torch.randn_like(x0)
        x_noisy = x0 + sigma * noise
        
        # 目标（带λ加权）
        s_d = self.sigma_data
        lambda_weight = (sigma**2 + s_d**2) / (sigma * s_d)**2
        
        x0_pred = self(x_noisy, sigma.squeeze((-1, -2, -3)))
        return (lambda_weight * (x0_pred - x0)**2).mean()


def heun_sampler(model: EDMPreconditioned, shape,
                 sigma_min=0.002, sigma_max=80.0,
                 rho=7, n_steps=18):
    """EDM Heun采样器（二阶Runge-Kutta）"""
    # σ序列：多项式衰减
    sigmas = (sigma_max**(1/rho) + torch.arange(n_steps+1)/(n_steps) *
              (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    sigmas[-1] = 0
    
    x = torch.randn(shape) * sigma_max
    
    for i in range(n_steps):
        s_curr = sigmas[i].item()
        s_next = sigmas[i+1].item()
        
        with torch.no_grad():
            # 一阶Euler步
            d_curr = (x - model(x, torch.full((shape[0],), s_curr))) / s_curr
            x_next = x + (s_next - s_curr) * d_curr
            
            if s_next > 0:
                # Heun修正（二阶）
                d_next = (x_next - model(x_next, torch.full((shape[0],), s_next))) / s_next
                d_avg = (d_curr + d_next) / 2
                x_next = x + (s_next - s_curr) * d_avg
        
        x = x_next
    
    return x
```

---

## 第五部分：条件生成

### 第13章：分类器引导

**13.5（提高题）** 分析分类器引导中的"对抗梯度"问题：为什么大引导尺度 $s$ 会导致图像质量下降？

**分析**：

分类器梯度的分解：

$$\nabla_x \log p_\psi(y|x_t) = \underbrace{\nabla_x \log p(x_t|y)}_{\text{语义方向}} + \underbrace{\nabla_x \log p(y)}_{\text{常数，=0}} - \underbrace{\nabla_x \log p(x_t)}_{\text{流形法向量}}$$

噪声分类器在远离数据流形的区域（高噪声 $x_t$）容易过拟合，其梯度包含"对抗扰动"成分——沿最大化分类器输出但不在自然图像流形上的方向。

```python
def visualize_guidance_effect():
    """演示不同引导尺度的梯度方向"""
    # 假设理想分类器：梯度 = 语义方向 + 对抗方向
    # 实际分类器梯度 = alpha * semantic + (1-alpha) * adversarial
    # 其中 alpha 随t减小（高噪声时对抗成分更强）
    
    scales = [1, 3, 5, 10, 20]
    
    print("引导尺度 → 图像质量的倒U型关系：")
    print(f"{'尺度':>6} | {'语义对齐':>8} | {'图像质量':>8} | {'多样性':>8}")
    print("-" * 40)
    
    for s in scales:
        # 简化模型
        semantic_alignment = min(s / 5, 1.0)  # 饱和
        adversarial_effect = max(0, (s - 5) / 15)  # 超过5后劣化
        image_quality = 1.0 - adversarial_effect
        diversity = 1.0 / s  # 多样性随尺度单调下降
        
        print(f"{s:>6} | {semantic_alignment:>8.2f} | {image_quality:>8.2f} | {diversity:>8.2f}")
    
    print("\n结论：s=3-7通常是质量-多样性的帕累托最优区间")
    print("CFG（无分类器引导）通过训练消除了对抗梯度问题")
```

---

### 第14章：无分类器引导

**14.5（提高题）** 推导CFG引导的最优引导尺度 $w^*$，使CLIP相似度最大化同时保持图像多样性。

**理论分析**：

CFG预测的噪声：$\tilde\epsilon = \epsilon_\text{uncond} + w(\epsilon_\text{cond} - \epsilon_\text{uncond})$

等价于从增强分布采样：$\tilde p(x|y) \propto p(x|y)^{1+w} \cdot p(x)^{-w}$

当 $w$ 增大时：
- CLIP相似度单调增（图像更符合条件）
- FID先降后升（过大的引导破坏图像质量）
- FID-CLIP权衡曲线存在帕累托前沿

```python
def cfg_pareto_analysis():
    """分析CFG引导尺度的帕累托前沿（理论模型）"""
    import numpy as np
    
    ws = np.linspace(0, 20, 100)
    
    # 简化的经验模型（基于SD1.5的典型结果）
    clip_score = 1 - np.exp(-ws / 5)          # 趋近于1
    diversity = np.exp(-ws / 8)               # 单调下降
    quality = np.exp(-(ws - 7.5)**2 / 25)     # 峰值在w≈7.5
    
    fid_proxy = 1 - 0.5 * (quality + diversity)  # FID越低越好（此处取反）
    
    # 帕累托最优点：quality和clip的权衡
    best_idx = np.argmax(quality * clip_score)
    
    print(f"最优引导尺度（质量×CLIP）: w = {ws[best_idx]:.1f}")
    print(f"典型推荐: w = 7.5 (SD1.5), w = 5.0 (SDXL), w = 3.5 (FLUX)")
    print("\n实践建议：")
    print("  - 精确文本匹配（商业用途）：w = 10-15")
    print("  - 高质量图像（艺术创作）：w = 5-8")
    print("  - 多样性探索：w = 2-5")
```

---

## 第六部分：架构

### 第16章：U-Net架构

**16.5（提高题）** 分析跳跃连接在扩散模型U-Net中的作用：与分割U-Net的区别，以及注意力层的必要性。

```python
# 实验：消融跳跃连接对扩散模型训练的影响

def ablation_study_skip_connections():
    """量化跳跃连接对不同噪声水平的影响"""
    
    # 理论分析：跳跃连接将编码器特征直接传递给解码器
    # 在扩散模型中，其作用与分割网络有重要区别：
    
    print("跳跃连接在扩散模型中的特殊作用：")
    print()
    print("1. 多尺度噪声处理：")
    print("   - 低噪声(t小)：跳跃连接保留细节（高频信息）")
    print("   - 高噪声(t大)：注意力层主导全局结构（低频信息）")
    print()
    print("2. 与分割U-Net的区别：")
    print("   - 分割：每次推理独立，跳跃连接=特征融合")
    print("   - 扩散：跨时间步迭代，跳跃连接=历史信息保留")
    print()
    print("3. 注意力层的必要性（Dhariwal & Nichol 2021实验）：")
    
    results = {
        "纯卷积（无注意力）": {"FID": 8.2, "params": "100M"},
        "8×8注意力": {"FID": 5.9, "params": "114M"},
        "8×8+16×16注意力": {"FID": 4.6, "params": "128M"},
        "全分辨率注意力": {"FID": 4.3, "params": "256M"},
    }
    
    for arch, metrics in results.items():
        print(f"   {arch}: FID={metrics['FID']}, 参数量={metrics['params']}")
    
    print()
    print("结论：注意力层在中间分辨率(16×16, 8×8)效果最显著；")
    print("      全分辨率注意力参数效率低（DiT通过patch化解决此问题）")
```

---

### 第17章：Transformer扩散（DiT）

**17.5（提高题）** 推导DiT中AdaLN的反向传播，分析条件 $y$（类别嵌入）通过 $\gamma, \delta$ 影响梯度流的机制。

**推导**：

AdaLN前向：$h = (1+\gamma(y)) \cdot \text{LN}(x) + \delta(y)$

对 $x$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial h} \cdot (1+\gamma(y)) \cdot \frac{\partial \text{LN}(x)}{\partial x}$$

对条件 $y$ 的梯度（通过 $\gamma, \delta$）：

$$\frac{\partial \mathcal{L}}{\partial y} = \frac{\partial \mathcal{L}}{\partial h} \cdot \text{LN}(x) \cdot \frac{\partial \gamma}{\partial y} + \frac{\partial \mathcal{L}}{\partial h} \cdot \frac{\partial \delta}{\partial y}$$

**关键分析**：
1. $(1+\gamma)$ 充当**梯度门控**：当 $\gamma = -1$ 时完全截断梯度；当 $\gamma > 0$ 时放大梯度
2. $\delta$ 是**加性偏移**，不影响归一化，但影响激活函数后的分布
3. 与Cross-Attention条件注入对比：AdaLN计算效率更高（无序列长度的二次复杂度），但表达能力略弱（全局调制 vs 位置自适应）

```python
class AdaLNAnalysis(nn.Module):
    """分析AdaLN的梯度动力学"""
    
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Linear(cond_dim, 2 * dim)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mod = self.modulation(y)                  # shape: (B, 2D)
        gamma, delta = mod.chunk(2, dim=-1)        # each: (B, D)
        gamma = gamma.unsqueeze(1)                 # shape: (B, 1, D)
        delta = delta.unsqueeze(1)                 # shape: (B, 1, D)
        
        x_norm = self.norm(x)                      # shape: (B, N, D)
        return (1 + gamma) * x_norm + delta        # shape: (B, N, D)
    
    def analyze_gradient_scale(self, y_range=(-2, 2)):
        """分析gamma对梯度尺度的影响"""
        gamma_vals = torch.linspace(*y_range, 100)
        grad_scales = 1 + gamma_vals  # 梯度放大/缩小倍数
        
        print("gamma值 → 梯度缩放因子：")
        for g in [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            print(f"  γ={g:+.1f} → 梯度×{1+g:.1f}")
        
        print("\n初始化策略（DiT论文）：")
        print("  最后一层MLP初始化为0 → gamma=0, delta=0")
        print("  训练初期：AdaLN等价于普通LayerNorm（梯度稳定）")
        print("  训练中后期：网络学会调制条件信息")
```

---

## 第七部分：前沿模型

### 第19章：Stable Diffusion

**19.5（提高题）** 分析SD v1→v2→v3的架构演进对生成质量的影响，设计消融实验验证各改进的贡献。

```python
def sd_architecture_comparison():
    """SD各版本架构对比"""
    
    versions = {
        "SD v1.5": {
            "文本编码器": "CLIP-L (123M)",
            "UNet参数": "860M",
            "VAE潜在维度": "4 (f=8)",
            "训练分辨率": "512×512",
            "关键改进": "基础版本，广泛使用",
            "FID (MS-COCO)": "~8.6",
        },
        "SD v2.1": {
            "文本编码器": "OpenCLIP-ViT-H (354M)",
            "UNet参数": "865M",
            "VAE潜在维度": "4 (f=8)",
            "训练分辨率": "768×768",
            "关键改进": "更大文本编码器，v-prediction",
            "FID (MS-COCO)": "~7.8",
        },
        "SD v3 (Medium)": {
            "文本编码器": "T5-XXL + CLIP-L + CLIP-G",
            "UNet→DiT": "MMDiT 2B参数",
            "VAE潜在维度": "16 (f=8)",
            "训练分辨率": "1024×1024",
            "关键改进": "Flow Matching, 16-ch VAE, 多编码器",
            "FID (MS-COCO)": "~6.1",
        },
    }
    
    print("Stable Diffusion架构演进：\n")
    for version, specs in versions.items():
        print(f"【{version}】")
        for key, val in specs.items():
            print(f"  {key}: {val}")
        print()
    
    print("消融实验设计（建议验证各改进的独立贡献）：")
    ablations = [
        ("OpenCLIP vs CLIP-L", "固定UNet，只换文本编码器，测CLIP-FID"),
        ("v-prediction vs ε-pred", "固定其他，只换预测目标，测FID和细节质量"),
        ("Flow Matching vs DDPM", "相同网络结构，只换训练目标，测采样步数-质量曲线"),
        ("16ch VAE vs 4ch VAE", "固定扩散模型，只换VAE，测重建质量(PSNR/SSIM)"),
        ("DiT vs U-Net", "相同参数量，对比架构，测FID-参数效率"),
    ]
    
    for exp, method in ablations:
        print(f"  [{exp}] {method}")
```

---

### 第22章：流匹配与一致性模型

**22.5（提高题）** 推导一致性模型（CM）的训练目标，分析其与蒸馏目标的联系。

**推导**：

一致性函数 $f_\theta(x_t, t)$ 满足自一致性条件：对任意 $(x_t, t)$ 和 $(x_s, s)$（同一轨迹上），有 $f_\theta(x_t, t) = f_\theta(x_s, s)$。

**一致性蒸馏损失**（CM-D）：

$$\mathcal{L}_{CD} = \mathbb{E}\left[d\left(f_\theta(x_{t_{n+1}}, t_{n+1}),\ f_{\theta^-}(\hat x_{t_n}^{\phi}, t_n)\right)\right]$$

其中：
- $\hat x_{t_n}^{\phi}$：用ODE求解器（如DDIM一步）从 $x_{t_{n+1}}$ 估计 $x_{t_n}$
- $\theta^-$：EMA参数（停止梯度）
- $d$：距离函数（LPIPS效果优于L2）

**联系分析**：

当 $n \to \infty$（步长无穷小）时，$\hat x_{t_n}^{\phi} \to x_{t_n}$（精确ODE解），CM训练变为Score Distillation的极限形式。CM-D等价于从预训练扩散模型中蒸馏知识，使学生网络在任意时间步都输出与教师一致的去噪结果。

```python
class ConsistencyModel(nn.Module):
    """一致性模型"""
    
    def __init__(self, backbone: nn.Module, T: float = 80.0,
                 eps: float = 0.002):
        super().__init__()
        self.backbone = backbone
        self.T = T
        self.eps = eps
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: 加噪数据，shape (B, C, H, W)
            t: 噪声水平（连续），shape (B,)
        Returns:
            x_0_pred: 原始数据预测，shape (B, C, H, W)
        """
        # c_skip, c_out保证在t=eps时 f(x,eps)=x（边界条件）
        c_skip = self.eps**2 / ((t - self.eps)**2 + self.eps**2)
        c_out = (t - self.eps) * self.eps / ((t - self.eps)**2 + self.eps**2).sqrt()
        
        c_skip = c_skip[:, None, None, None]
        c_out = c_out[:, None, None, None]
        
        F_out = self.backbone(x_t, t)
        return c_skip * x_t + c_out * F_out
    
    def consistency_distillation_loss(self, x0: torch.Tensor,
                                      teacher_score_fn,
                                      ema_model,
                                      n_timesteps: int = 40):
        """一致性蒸馏训练损失"""
        B = x0.shape[0]
        
        # 随机选择相邻时间步对 (t_{n+1}, t_n)
        n = torch.randint(1, n_timesteps, (B,))
        sigmas = self.get_sigma_schedule(n_timesteps)
        
        t_n1 = sigmas[n]                           # shape: (B,)
        t_n = sigmas[n - 1]                        # shape: (B,)
        
        # 从 x0 采样 x_{t_{n+1}}
        noise = torch.randn_like(x0)
        x_t_n1 = x0 + t_n1[:, None, None, None] * noise  # shape: (B, C, H, W)
        
        # 用DDIM一步估计 x_{t_n}（教师模型）
        with torch.no_grad():
            score = teacher_score_fn(x_t_n1, t_n1)
            dt = t_n - t_n1
            x_t_n_hat = x_t_n1 + dt[:, None, None, None] * score
        
        # 一致性目标：f_θ(x_{n+1}, t_{n+1}) ≈ f_{θ-}(x̂_n, t_n)
        f_n1 = self(x_t_n1, t_n1)
        with torch.no_grad():
            f_n_hat = ema_model(x_t_n_hat, t_n)
        
        # LPIPS或L2距离
        loss = F.mse_loss(f_n1, f_n_hat)
        return loss
    
    def get_sigma_schedule(self, n: int) -> torch.Tensor:
        rho = 7
        i_vals = torch.arange(n + 1)
        return (self.eps**(1/rho) + i_vals/n * (self.T**(1/rho) - self.eps**(1/rho)))**rho
```

---

## 第八部分：工程实践

### 第23章：推理优化

**23.5（提高题）** 实现扩散模型的投机解码（Speculative Decoding）变体，分析接受率与加速比的关系。

```python
class SpeculativeDiffusionSampler:
    """扩散模型的投机采样
    用小型草稿模型预测多步，再用完整模型验证
    
    有效条件：草稿与完整模型的预测KL散度较小
    """
    
    def __init__(self, full_model, draft_model,
                 scheduler, gamma: int = 4):
        """
        Args:
            full_model: 完整扩散模型（质量高，速度慢）
            draft_model: 草稿模型（质量低，速度快，如少层版本）
            scheduler: DDIM调度器
            gamma: 草稿步数（每次验证前运行几步草稿）
        """
        self.full = full_model
        self.draft = draft_model
        self.scheduler = scheduler
        self.gamma = gamma
    
    def speculative_step(self, x: torch.Tensor,
                         timesteps: list, start_idx: int,
                         cond=None) -> tuple:
        """
        执行gamma步草稿预测，然后用完整模型验证
        
        Returns:
            x_new: 更新后的状态
            accepted_steps: 实际接受的步数
        """
        gamma = min(self.gamma, len(timesteps) - start_idx - 1)
        
        # 阶段1：草稿预测（不计梯度）
        draft_predictions = []
        x_draft = x.clone()
        
        with torch.no_grad():
            for i in range(gamma):
                t = timesteps[start_idx + i]
                t_next = timesteps[start_idx + i + 1]
                
                eps_draft = self.draft(x_draft, t, cond)
                x_draft_next, x0_pred = self.scheduler.ddim_step(
                    x_draft, eps_draft, t, t_next
                )
                draft_predictions.append({
                    'x': x_draft.clone(),
                    'x_next': x_draft_next.clone(),
                    'eps': eps_draft.clone(),
                    't': t, 't_next': t_next,
                })
                x_draft = x_draft_next
        
        # 阶段2：完整模型验证（并行批处理所有草稿步）
        # 将gamma个时间步合并为一个批次
        x_batch = torch.stack([p['x'] for p in draft_predictions])  # (gamma, B, C, H, W)
        t_batch = torch.tensor([p['t'] for p in draft_predictions])  # (gamma,)
        
        B = x.shape[0]
        x_batch_flat = x_batch.reshape(-1, *x.shape[1:])            # (gamma*B, C, H, W)
        t_batch_flat = t_batch.repeat_interleave(B)                  # (gamma*B,)
        
        with torch.no_grad():
            eps_full_flat = self.full(x_batch_flat, t_batch_flat, cond)
        
        eps_full = eps_full_flat.reshape(gamma, B, *x.shape[1:])     # (gamma, B, C, H, W)
        
        # 阶段3：逐步验证（比较草稿和完整模型的输出）
        accepted_steps = 0
        x_current = x.clone()
        
        for i in range(gamma):
            eps_d = draft_predictions[i]['eps']
            eps_f = eps_full[i]
            t = draft_predictions[i]['t']
            t_next = draft_predictions[i]['t_next']
            
            # 计算接受概率（基于噪声预测的一致性）
            cos_sim = F.cosine_similarity(
                eps_f.reshape(B, -1), eps_d.reshape(B, -1), dim=-1
            ).mean().item()
            
            acceptance_threshold = 0.9  # 可调参数
            
            if cos_sim > acceptance_threshold:
                # 接受草稿步
                x_current = draft_predictions[i]['x_next']
                accepted_steps += 1
            else:
                # 拒绝：用完整模型重新计算此步
                x_current, _ = self.scheduler.ddim_step(
                    x_current, eps_f, t, t_next
                )
                accepted_steps += 1  # 此步仍然有效
                break  # 停止，丢弃后续草稿步
        
        return x_current, accepted_steps
    
    def analyze_acceptance_rate(self, x: torch.Tensor, timesteps: list, cond=None):
        """分析不同时间步区间的接受率"""
        n = len(timesteps)
        
        early_sims = []   # 高噪声区（t > T/2）
        late_sims = []    # 低噪声区（t < T/2）
        
        for i in range(0, n-1, 5):
            t = timesteps[i]
            t_next = timesteps[i+1]
            
            with torch.no_grad():
                eps_draft = self.draft(x, t, cond)
                eps_full = self.full(x, t, cond)
            
            sim = F.cosine_similarity(
                eps_full.reshape(x.shape[0], -1),
                eps_draft.reshape(x.shape[0], -1), dim=-1
            ).mean().item()
            
            if i < n // 2:
                early_sims.append(sim)
            else:
                late_sims.append(sim)
            
            # 模拟DDIM步
            with torch.no_grad():
                x, _ = self.scheduler.ddim_step(x, eps_full, t, t_next)
        
        print("草稿模型接受率分析：")
        print(f"  高噪声区（早期步骤）平均余弦相似度: {sum(early_sims)/len(early_sims):.3f}")
        print(f"  低噪声区（后期步骤）平均余弦相似度: {sum(late_sims)/len(late_sims):.3f}")
        print()
        print("理论加速比公式：")
        print("  speedup = (γ+1) / (γ·(1-α) + 1)")
        print("  其中 α = 接受率，γ = 草稿步数")
        print()
        for alpha in [0.7, 0.8, 0.9, 0.95]:
            for gamma in [2, 4, 8]:
                speedup = (gamma + 1) / (gamma * (1 - alpha) + 1)
                print(f"  α={alpha:.2f}, γ={gamma}: speedup≈{speedup:.2f}x")
```

---

### 第24章：完整项目

**24.5（提高题）** 为FashionMNIST条件扩散模型实现LoRA微调，在仅5张图像的小样本场景下生成高质量新样式。

```python
import torch
import torch.nn as nn
from typing import Optional

class LoRALinear(nn.Module):
    """LoRA线性层
    W_new = W_0 + α/r · B·A
    其中 W_0 冻结，只训练 A, B
    """
    
    def __init__(self, original_layer: nn.Linear,
                 rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA矩阵（低秩分解）
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 冻结原始权重
        for param in self.original.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)              # 原始输出
        lora_out = x @ self.lora_A.T @ self.lora_B.T  # LoRA增量
        return base_out + (self.alpha / self.rank) * lora_out


def inject_lora(model: nn.Module, target_modules: list = None,
                rank: int = 4, alpha: float = 1.0):
    """将LoRA注入模型的目标线性层
    
    Args:
        model: 预训练扩散模型
        target_modules: 要注入的模块名称列表（None=所有Linear层）
        rank: LoRA秩
        alpha: 缩放因子
    """
    if target_modules is None:
        target_modules = ['q', 'k', 'v', 'out_proj']
    
    def replace_module(parent, name, child):
        if isinstance(child, nn.Linear):
            # 检查是否是目标模块
            if any(target in name for target in target_modules):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(parent, name, lora_layer)
                return True
        return False
    
    replaced = 0
    for name, module in model.named_children():
        if replace_module(model, name, module):
            replaced += 1
        else:
            # 递归处理子模块
            replaced += inject_lora(module, target_modules, rank, alpha)
    
    return replaced


def few_shot_lora_finetune(base_model, few_shot_images: torch.Tensor,
                           few_shot_labels: torch.Tensor,
                           scheduler, n_steps: int = 500,
                           lr: float = 1e-3):
    """
    5-shot LoRA微调
    
    Args:
        base_model: 预训练条件扩散模型
        few_shot_images: 5张目标风格图像，shape (5, C, H, W)
        few_shot_labels: 对应类别标签，shape (5,)
        scheduler: 噪声调度器
        n_steps: 微调步数
    """
    # 注入LoRA（只训练注意力层）
    n_replaced = inject_lora(base_model, rank=4, alpha=1.0)
    print(f"注入LoRA到 {n_replaced} 个线性层")
    
    # 统计可训练参数
    total = sum(p.numel() for p in base_model.parameters())
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"总参数: {total:,}, LoRA可训练: {trainable:,} ({trainable/total:.2%})")
    
    optimizer = torch.optim.AdamW(
        [p for p in base_model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    
    base_model.train()
    
    for step in range(n_steps):
        # 数据增强：从5张图中随机采样（随机翻转、裁剪）
        idx = torch.randint(0, len(few_shot_images), (4,))
        x0 = few_shot_images[idx]
        labels = few_shot_labels[idx]
        
        # 随机时间步
        t = torch.randint(0, scheduler.T, (x0.shape[0],))
        
        # 加噪
        x_t, eps = scheduler.q_sample(x0, t)
        
        # 预测（随机条件dropout用于CFG）
        use_null = torch.rand(x0.shape[0]) < 0.1
        cond = torch.where(use_null[:, None].expand_as(labels.unsqueeze(-1)),
                           torch.zeros_like(labels),
                           labels)
        
        eps_pred = base_model(x_t, t, cond)
        loss = F.mse_loss(eps_pred, eps)
        
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止LoRA矩阵过大）
        nn.utils.clip_grad_norm_(
            [p for p in base_model.parameters() if p.requires_grad],
            max_norm=1.0
        )
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")
    
    print(f"\n微调完成！LoRA权重可保存为增量更新（约{trainable*4/1024:.1f}KB）")
    return base_model
```

**实验分析**：
- 无LoRA全量微调5步：过拟合，泛化差
- rank=4 LoRA微调500步：保留预训练的生成能力，同时适应新风格
- 微调的关键是低学习率（1e-4）+ 梯度裁剪 + 较小的rank（不超过16）

---

[返回目录](../README.md)

[← 附录B：Diffusers API速查](./diffusers-api.md)
