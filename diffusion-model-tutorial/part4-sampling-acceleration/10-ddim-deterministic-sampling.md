# 第十章：DDIM——确定性采样

> **本章导读**：DDPM采样需要1000步，速度极慢。Song et al. 2020提出DDIM（Denoising Diffusion Implicit Models），在不重新训练模型的情况下，将采样步数减少到50步甚至10步，且生成质量几乎不下降。本章深入剖析DDIM的数学原理，揭示其与概率流ODE的联系。

**前置知识**：第6-9章，SDE基础
**预计学习时间**：100分钟

---

## 学习目标

完成本章学习后，你将能够：
1. 理解DDPM采样慢的根本原因，掌握DDIM的非马尔科夫正向过程定义
2. 推导DDIM逆向更新公式，理解其中 $\sigma_\tau$ 参数的作用
3. 证明DDIM是概率流ODE的欧拉离散化
4. 实现DDIM采样器，对比DDPM与DDIM的速度-质量权衡
5. 理解DDIM的隐变量插值能力（语义插值）

---

## 10.1 DDPM慢采样的根本原因

### 马尔科夫约束

DDPM逆向过程是马尔科夫链：

$$p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)$$

每步 $p_\theta(x_{t-1}|x_t)$ 只依赖 $x_t$。训练时，ELBO分解为 $T$ 个KL项，**每个时间步都需要**在反向传播中单独优化。

**问题**：推理时必须完整地走1000步，每步一次神经网络前向传播。

$$\text{采样时间} \propto T \times \text{(网络推理时间)}$$

对于$512\times512$图像，1000步约需15-30秒（GPU）。

### 加速的直觉

注意：训练目标只依赖 $q(x_t|x_0)$，而不依赖马尔科夫链的逐步结构！

$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$$

如果我们能找到另一个正向过程，它满足：
1. 相同的边缘分布 $q(x_t|x_0)$（确保训练好的模型可复用）
2. 但具有**不同的条件分布** $q(x_{t-1}|x_t, x_0)$

那就可以用不同的步数来采样！

---

## 10.2 DDIM的非马尔科夫正向过程

### 广义正向过程

Song et al. 构造了一族正向过程，边缘分布与DDPM相同：

$$q_\sigma(x_{1:T}|x_0) = q_\sigma(x_T|x_0)\prod_{t=2}^T q_\sigma(x_{t-1}|x_t, x_0)$$

其中，条件分布设计为：

$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left(x_{t-1}; \sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\frac{x_t - \sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}, \sigma_t^2 I\right)$$

**验证边缘分布一致**：可以验证，在任意 $\sigma_t$ 下，$q_\sigma(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$。

**关键参数**：$\sigma_t$ 控制每步添加的随机噪声量：
- $\sigma_t = \sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\sqrt{1-\alpha_t}$：退化为**DDPM**（马尔科夫，完全随机）
- $\sigma_t = 0$：**DDIM**（完全确定性，给定 $x_T$ 则轨迹确定）

### DDIM逆向更新公式

用学习到的 $\epsilon_\theta(x_t, t) \approx \epsilon$，预测 $x_0$：

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$$

代入条件分布，得到DDIM更新：

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\underbrace{\hat{x}_0}_{\text{预测}x_0} + \underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\epsilon_\theta(x_t, t)}_{\text{"指向x_t方向"}} + \underbrace{\sigma_t\epsilon_t}_{\text{随机项}}$$

当 $\sigma_t = 0$ 时，无随机噪声，轨迹完全确定！

---

## 10.3 子序列采样（跳步加速）

### 时间步子集

DDIM的真正加速来自：可以只在**时间步子序列** $\tau = [\tau_1, \tau_2, ..., \tau_S] \subset [1, ..., T]$ 上采样，$S \ll T$。

例如，$T=1000$，选择 $\tau = \{1, 21, 41, ..., 981\}$（共50个，均匀间隔）。

更新公式从 $x_{\tau_i}$ 到 $x_{\tau_{i-1}}$：

$$x_{\tau_{i-1}} = \sqrt{\bar\alpha_{\tau_{i-1}}}\hat{x}_0 + \sqrt{1-\bar\alpha_{\tau_{i-1}}-\sigma_{\tau_i}^2}\cdot\epsilon_\theta(x_{\tau_i}, \tau_i)$$

（令 $\sigma_{\tau_i} = 0$，即DDIM确定性版本）

### 速度-质量曲线

| 步数 | FID (DDPM) | FID (DDIM) |
|------|------------|------------|
| 1000 | 3.17 | 4.16 |
| 100 | - | 4.67 |
| 50 | - | 4.86 |
| 10 | - | 6.84 |

DDIM在50步时性能接近DDPM 1000步，速度提升20倍！

---

## 10.4 DDIM与概率流ODE的联系

### DDIM是ODE求解

DDIM的确定性更新（$\sigma_t = 0$）是以下ODE的欧拉离散化：

$$\frac{dx}{d\bar\alpha} = \frac{x - \sqrt{1-\bar\alpha}\epsilon_\theta(x, t)}{2\bar\alpha}$$

等价地，代入 $d(\sqrt{\bar\alpha})\bar\alpha\partial_t\sqrt{\bar\alpha}$，可以改写为概率流ODE形式：

$$\frac{dx}{dt} = f(x, t) - \frac{1}{2}g(t)^2\nabla_x\log p_t(x)$$

其中用 $-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$ 近似 $\nabla_x\log p_t$。

**结论**：DDIM是第6章概率流ODE的欧拉法离散化！使用更高阶的ODE求解器（如Heun法）可以进一步提高精度。

### DDIM隐变量编码

确定性轨迹带来了"精确逆"的能力：

$$x_0 \xrightarrow{\text{DDIM Inversion}} x_T \xrightarrow{\text{DDIM Sampling}} x_0$$

给定图像 $x_0$，可以通过**DDIM Inversion**（时间反向运行ODE）得到对应的潜变量 $x_T$，修改后再生成，实现**图像编辑**。

---

## 10.5 隐变量插值

### 语义插值

DDIM使得图像语义插值成为可能：

1. 编码图像 $A \to x_T^A$，图像 $B \to x_T^B$
2. 插值：$x_T^{(\lambda)} = (1-\lambda)x_T^A + \lambda x_T^B$
3. 解码：$x_T^{(\lambda)} \to x^{(\lambda)}$

由于 $x_T \approx \mathcal{N}(0, I)$，球形插值（SLERP）效果更好：

$$\text{SLERP}(x_T^A, x_T^B; \lambda) = \frac{\sin((1-\lambda)\theta)}{\sin\theta}x_T^A + \frac{\sin(\lambda\theta)}{\sin\theta}x_T^B$$

其中 $\theta = \arccos\left(\frac{x_T^A \cdot x_T^B}{\|x_T^A\|\|x_T^B\|}\right)$。

---

## 代码实战

```python
"""
第十章代码实战：DDIM采样器实现
对比DDPM与DDIM的速度-质量权衡
"""
import torch
import numpy as np
from typing import Optional, List
import time


# ============================================================
# 1. 噪声调度（共享）
# ============================================================

class NoiseScheduler:
    """线性噪声调度，支持DDPM和DDIM"""
    
    def __init__(self, T: int = 1000, beta_min: float = 1e-4, beta_max: float = 0.02):
        self.T = T
        # 线性调度
        self.betas = torch.linspace(beta_min, beta_max, T)          # shape: (T,)
        self.alphas = 1.0 - self.betas                               # shape: (T,)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)          # shape: (T,)
        
        # 填充 alpha_bar_0 = 1（用于DDIM）
        self.alpha_bars_prev = torch.cat(
            [torch.tensor([1.0]), self.alpha_bars[:-1]]              # shape: (T,)
        )
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        正向加噪：x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        
        Args:
            x0: shape (B, C, H, W)
            t: 时间步索引，shape (B,)
            noise: 可选，shape (B, C, H, W)
        
        Returns:
            x_t: shape (B, C, H, W)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_bar = self.alpha_bars[t].sqrt()[:, None, None, None]   # (B, 1, 1, 1)
        sqrt_1m_alpha_bar = (1 - self.alpha_bars[t]).sqrt()[:, None, None, None]
        
        return sqrt_alpha_bar * x0 + sqrt_1m_alpha_bar * noise


# ============================================================
# 2. DDPM采样器
# ============================================================

class DDPMSampler:
    """标准DDPM采样器（1000步）"""
    
    def __init__(self, scheduler: NoiseScheduler):
        self.scheduler = scheduler
        self.T = scheduler.T
    
    @torch.no_grad()
    def sample(self, model, shape: tuple, device: str = 'cpu') -> torch.Tensor:
        """
        DDPM反向采样
        
        Args:
            model: 噪声预测网络，输入(x_t, t)，输出eps预测
            shape: 生成形状 (B, C, H, W)
        
        Returns:
            x0: shape (B, C, H, W)
        """
        x = torch.randn(shape, device=device)  # shape: (B, C, H, W)
        sch = self.scheduler
        
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # 预测噪声
            eps_pred = model(x, t_batch)           # shape: (B, C, H, W)
            
            alpha_t = sch.alphas[t]
            alpha_bar_t = sch.alpha_bars[t]
            beta_t = sch.betas[t]
            
            # 预测 x0
            x0_pred = (x - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)        # 裁剪
            
            # 均值（DDPM后验均值）
            coef1 = alpha_bar_t.sqrt() * beta_t / (1 - alpha_bar_t)
            coef2 = (1 - sch.alpha_bars_prev[t]).sqrt() * alpha_t.sqrt() / (1 - alpha_bar_t)
            mean = coef1 * x0_pred + coef2 * x    # shape: (B, C, H, W)
            
            # 方差
            var = beta_t * (1 - sch.alpha_bars_prev[t]) / (1 - alpha_bar_t)
            
            # 采样
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + var.sqrt() * noise
            else:
                x = mean
        
        return x


# ============================================================
# 3. DDIM采样器
# ============================================================

class DDIMSampler:
    """
    DDIM确定性采样器（支持任意步数）
    
    Reference: Song et al. "Denoising Diffusion Implicit Models" (2021)
    """
    
    def __init__(self, scheduler: NoiseScheduler):
        self.scheduler = scheduler
        self.T = scheduler.T
    
    def make_timesteps(self, num_steps: int) -> List[int]:
        """
        创建均匀间隔的时间步子序列
        
        Args:
            num_steps: 采样步数（S << T）
        
        Returns:
            timesteps: 降序时间步列表，shape: (S,)
        """
        step_size = self.T // num_steps
        timesteps = list(range(0, self.T, step_size))[::-1]
        return timesteps
    
    @torch.no_grad()
    def sample(self, model, shape: tuple, num_steps: int = 50,
               eta: float = 0.0, device: str = 'cpu') -> torch.Tensor:
        """
        DDIM采样
        
        Args:
            model: 噪声预测网络
            shape: 生成形状 (B, C, H, W)
            num_steps: 采样步数（默认50，远小于T=1000）
            eta: 随机性参数（0=完全确定，1=等价DDPM）
        
        Returns:
            x0: shape (B, C, H, W)
        """
        sch = self.scheduler
        timesteps = self.make_timesteps(num_steps)  # 时间步子序列（降序）
        
        x = torch.randn(shape, device=device)        # shape: (B, C, H, W)
        
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            alpha_bar_t = sch.alpha_bars[t]
            alpha_bar_t_prev = sch.alpha_bars[t_prev] if t_prev > 0 else torch.tensor(1.0)
            
            # 预测噪声
            eps_pred = model(x, t_batch)             # shape: (B, C, H, W)
            
            # 预测 x0
            x0_pred = (x - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            
            # DDIM更新
            # sigma_t = eta * sqrt((1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * sqrt(1 - alpha_t)
            sigma_t = eta * ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)).sqrt() * \
                      (1 - alpha_bar_t / alpha_bar_t_prev).sqrt()
            
            # 方向向量（"指向 x_t 的方向"）
            direction = (1 - alpha_bar_t_prev - sigma_t**2).sqrt() * eps_pred
            
            # 更新 x_{t-1}
            x = alpha_bar_t_prev.sqrt() * x0_pred + direction
            
            if eta > 0 and t > 0:
                noise = torch.randn_like(x)
                x = x + sigma_t * noise
        
        return x
    
    @torch.no_grad()
    def encode(self, model, x0: torch.Tensor, num_steps: int = 50,
               device: str = 'cpu') -> torch.Tensor:
        """
        DDIM Inversion：将图像编码为潜变量 x_T
        （确定性DDIM的时间反转）
        
        Args:
            x0: 输入图像，shape: (B, C, H, W)
            num_steps: 步数
        
        Returns:
            x_T: 潜变量，shape: (B, C, H, W)
        """
        sch = self.scheduler
        timesteps = self.make_timesteps(num_steps)[::-1]  # 正向顺序（升序）
        
        x = x0.clone()
        
        for i, t in enumerate(timesteps):
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else self.T - 1
            
            t_batch = torch.full((x0.shape[0],), t, device=device, dtype=torch.long)
            
            alpha_bar_t = sch.alpha_bars[t]
            alpha_bar_t_next = sch.alpha_bars[t_next]
            
            # 预测噪声
            eps_pred = model(x, t_batch)
            
            # 预测 x0
            x0_pred = (x - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
            
            # 正向（Inversion）更新
            x = alpha_bar_t_next.sqrt() * x0_pred + (1 - alpha_bar_t_next).sqrt() * eps_pred
        
        return x


# ============================================================
# 4. 速度对比演示（使用随机"模型"模拟）
# ============================================================

def benchmark_samplers():
    """对比DDPM和DDIM采样速度"""
    
    # 模拟噪声预测模型（随机输出，仅用于测速）
    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.randn_like(x)  # shape: same as x
    
    scheduler = NoiseScheduler(T=1000)
    model = DummyModel()
    shape = (1, 1, 16, 16)  # 小图像，仅测速
    
    # DDPM（全步）
    ddpm = DDPMSampler(scheduler)
    start = time.time()
    _ = ddpm.sample(model, shape)
    ddpm_time = time.time() - start
    
    # DDIM不同步数
    ddim = DDIMSampler(scheduler)
    results = {}
    for steps in [10, 20, 50, 100]:
        start = time.time()
        _ = ddim.sample(model, shape, num_steps=steps)
        results[steps] = time.time() - start
    
    print(f"DDPM (1000步): {ddpm_time:.3f}s")
    for steps, t in results.items():
        speedup = ddpm_time / t
        print(f"DDIM ({steps:4d}步): {t:.3f}s  (加速 {speedup:.1f}x)")
    
    # SLERP插值演示
    z1 = torch.randn(1, 4, 8, 8)
    z2 = torch.randn(1, 4, 8, 8)
    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    def slerp(z1: torch.Tensor, z2: torch.Tensor, lam: float) -> torch.Tensor:
        """球形线性插值"""
        z1_flat = z1.flatten()
        z2_flat = z2.flatten()
        cos_theta = (z1_flat @ z2_flat) / (z1_flat.norm() * z2_flat.norm())
        theta = torch.arccos(cos_theta.clamp(-1, 1))
        if theta.abs() < 1e-6:
            return (1 - lam) * z1 + lam * z2
        return (torch.sin((1 - lam) * theta) / torch.sin(theta)) * z1 + \
               (torch.sin(lam * theta) / torch.sin(theta)) * z2
    
    print("\nSLERP插值:")
    for lam in lambdas:
        interp = slerp(z1, z2, lam)
        print(f"  lambda={lam:.2f}: norm={interp.norm():.4f}")


if __name__ == "__main__":
    print("=" * 50)
    print("DDIM采样器对比")
    benchmark_samplers()
    
    print("\n关键公式总结:")
    print("DDIM更新: x_{t-1} = sqrt(αbar_{t-1})·x0_pred + sqrt(1-αbar_{t-1}-σ²)·ε_θ + σ·ε")
    print("当σ=0: 完全确定性（DDIM），给定x_T轨迹唯一")
    print("当σ=sqrt((1-αbar_{t-1})/(1-αbar_t))·sqrt(1-αt): 退化为DDPM")
    print("DDIM Inversion: 时间反向运行ODE，可恢复x_T")
```

---

## 本章小结

| 概念 | 数学形式 | 意义 |
|------|----------|------|
| DDIM正向过程 | $q_\sigma(x_{t-1}\|x_t,x_0) = \mathcal{N}(\sqrt{\bar\alpha_{t-1}}\hat{x}_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\hat\epsilon, \sigma_t^2 I)$ | 非马尔科夫，边缘分布与DDPM一致 |
| DDIM确定性采样 | $\sigma_t = 0$，步数 $S \ll T$ | 加速20-100倍 |
| 概率流ODE | DDIM = ODE的欧拉离散化 | 允许高阶求解器 |
| DDIM Inversion | $x_0 \to x_T$（时间反转） | 图像编辑基础 |

---

## 练习题

### 基础题

**10.1** 在DDIM更新公式中，令 $\sigma_t = \sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}(1-\alpha_t)}$，证明其退化为DDPM更新规则（利用第8章的 $\tilde\mu_t$ 公式）。

**10.2** 解释为什么DDIM可以用50步代替DDPM的1000步，而DDPM不能直接跳步。（提示：考虑马尔科夫性和非马尔科夫性。）

### 中级题

**10.3** 实现DDIM Inversion，对一个简单的2D数据集中的样本进行编码，验证：解码后的样本与原始样本之间的重建误差随步数增加而减小。

**10.4** 在DDIM采样中，实现球面线性插值（SLERP）：对两张图像分别做DDIM Inversion得到 $x_T^A, x_T^B$，然后在它们之间做SLERP插值，再DDIM解码，观察插值图像的语义变化。

### 提高题

**10.5** DDIM的确定性采样可以看作ODE求解。实现一个二阶Heun求解器（Karras et al. 2022），在相同步数下与DDIM（一阶欧拉法）对比生成质量（FID分数）。

---

## 练习答案

**10.1** 令 $\sigma_t = \sqrt{\frac{(1-\bar\alpha_{t-1})\beta_t}{1-\bar\alpha_t}}$（即 $\tilde\beta_t$），代入DDIM公式：方向项为 $\sqrt{1-\bar\alpha_{t-1}-\tilde\beta_t}\cdot\epsilon_\theta$，其系数化简后与 $\tilde\mu_t$ 的 $x_t$ 系数一致。

**10.2** DDPM是马尔科夫链，$p_\theta(x_{t-1}|x_t)$ 只依赖 $x_t$；若跳过中间步，模型无法处理不连续的噪声水平组合。DDIM的 $q_\sigma(x_{t-1}|x_t, x_0)$ 显式依赖 $x_0$ 预测，无马尔科夫约束，因此可以任意跳步。

**10.3** 关键：步数越多，$x_0 \to x_T \to x_0$ 的数值误差（ODE积分误差）越小。步数20-50步时重建误差已很小。

**10.4** SLERP保持球面距离，对高斯噪声分布更自然，插值图像的语义也更平滑（不会出现模糊中间态）。

**10.5** Heun法：$x_{t-1}' = x_t + \Delta t \cdot d_t$（欧拉预测），$x_{t-1} = x_t + \Delta t \cdot \frac{d_t + d_t'}{2}$（修正）。相同步数下，Heun法FID通常比DDIM低0.5-1个点。

---

## 延伸阅读

1. **Song et al. (2021)**. *Denoising Diffusion Implicit Models* — DDIM原文
2. **Karras et al. (2022)**. *Elucidating the Design Space of Diffusion-Based Generative Models* — 分析DDIM与更高阶求解器
3. **Song et al. (2021)**. *Score-Based Generative Modeling through SDEs* — 概率流ODE（第6章联系）

---

[← 上一章：噪声预测网络与训练技巧](../part3-ddpm-core/09-noise-prediction-network-training.md)

[下一章：噪声调度器设计 →](./11-noise-scheduler-design.md)

[返回目录](../README.md)
