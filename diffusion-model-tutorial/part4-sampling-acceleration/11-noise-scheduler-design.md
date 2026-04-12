# 第十一章：噪声调度器设计

> **本章导读**：噪声调度（Noise Schedule）决定了扩散模型如何将数据逐步变成噪声，以及每个时间步的信噪比分布。Ho et al. 2020使用线性调度，Nichol & Dhariwal 2021发现余弦调度更好。本章系统分析各种调度的数学性质和设计原则。

**前置知识**：第7-8章，第10章
**预计学习时间**：80分钟

---

## 学习目标

完成本章学习后，你将能够：
1. 理解噪声调度的数学定义及其对信噪比（SNR）的影响
2. 分析线性、余弦、Sigmoid调度的优缺点
3. 实现连续时间调度，理解EDM（Karras et al. 2022）的调度框架
4. 理解调度与训练时间步采样分布的联系
5. 为不同任务（图像、视频、音频）选择合适的调度策略

---

## 11.1 噪声调度的数学框架

### 基本定义

噪声调度由序列 $\{\beta_t\}_{t=1}^T$ 或等价的 $\{\bar\alpha_t\}_{t=1}^T$ 定义：

$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$$

**信噪比（SNR）**：

$$\text{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t}$$

$t=0$时 $\text{SNR} \to \infty$（纯信号），$t=T$时 $\text{SNR} \approx 0$（纯噪声）。

调度的本质就是设计SNR随时间的衰减曲线。

### 关键约束

1. $\bar\alpha_0 \approx 1$（初始基本不加噪声）
2. $\bar\alpha_T \approx 0$（最终接近标准高斯）
3. $\bar\alpha_t$ 单调递减

---

## 11.2 常用调度方案

### 线性调度（DDPM原始）

$$\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})$$

典型值：$\beta_{\min} = 10^{-4}$，$\beta_{\max} = 0.02$，$T = 1000$。

**问题**：对于高分辨率图像（$256\times256$），前几步噪声太大，后几步噪声太小，SNR曲线在两端浪费了大量步数。

具体地，$\bar\alpha_T = \prod_{t=1}^T(1-\beta_t)$ 在线性调度下约为 $10^{-4}$（够小），但中间步数分布不够均匀。

### 余弦调度（improved DDPM）

Nichol & Dhariwal (2021) 提出余弦调度：

$$\bar\alpha_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left(\frac{t/T + s}{1 + s}\cdot\frac{\pi}{2}\right)$$

其中 $s = 0.008$ 是偏移量（防止端点奇异性）。

**优势**：
- SNR在两端变化平缓，中间变化均匀
- 图像质量明显提升（尤其低分辨率）
- $\bar\alpha_T \approx 0$，但不会过于接近0而导致数值问题

**对比**：

| 时间步 | 线性调度 $\bar\alpha_t$ | 余弦调度 $\bar\alpha_t$ |
|--------|------------------------|------------------------|
| $t=0$  | 1.000                  | 1.000                  |
| $t=250$| 0.671                  | 0.924                  |
| $t=500$| 0.334                  | 0.691                  |
| $t=750$| 0.077                  | 0.312                  |
| $t=1000$| 0.0001               | 0.0001                 |

### Sigmoid调度

$$\bar\alpha_t = \sigma\left(-\frac{t - T/2}{T/(2\cdot k)}\right)$$

其中 $\sigma$ 是sigmoid函数，$k$ 控制过渡陡峭度。Sigmoid调度在SD3等模型中使用，可以控制噪声过渡的集中区域。

---

## 11.3 EDM调度框架

Karras et al. (2022) 提出了统一的EDM（Elucidating Diffusion Models）框架，将噪声调度与ODE求解器解耦。

### EDM噪声参数化

用 $\sigma(t)$ 直接参数化噪声标准差（而非 $\bar\alpha_t$）：

$$x_t = x_0 + \sigma(t)\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

（VE-SDE形式，不保方差）

**训练分布**：$p(\sigma) = \ln\mathcal{N}(\sigma; P_{mean}, P_{std}^2)$，典型值 $P_{mean} = -1.2$，$P_{std} = 1.2$。

### EDM预条件化

EDM对网络预条件化，消除不同噪声水平下的数值不稳定：

$$D_\theta(x, \sigma) = c_{skip}(\sigma)x + c_{out}(\sigma)F_\theta(c_{in}(\sigma)x, c_{noise}(\sigma))$$

其中：
$$c_{skip}(\sigma) = \frac{\sigma_{data}^2}{\sigma^2 + \sigma_{data}^2}, \quad c_{out}(\sigma) = \frac{\sigma\cdot\sigma_{data}}{\sqrt{\sigma^2 + \sigma_{data}^2}}$$
$$c_{in}(\sigma) = \frac{1}{\sqrt{\sigma^2 + \sigma_{data}^2}}, \quad c_{noise}(\sigma) = \frac{1}{4}\ln\sigma$$

**目的**：网络 $F_\theta$ 的输入/输出始终在 $O(1)$ 量级，训练更稳定。

### EDM采样时间步分布

推理时，EDM选择时间步序列使ODE求解误差均匀分布：

$$\sigma_i = \left(\sigma_{max}^{1/\rho} + \frac{i}{n-1}(\sigma_{min}^{1/\rho} - \sigma_{max}^{1/\rho})\right)^\rho$$

其中 $\rho = 7$（默认），$\sigma_{max} = 80$，$\sigma_{min} = 0.002$。

---

## 11.4 连续时间调度

对于SDE框架（第6章），调度由连续函数 $\beta(t)$ 定义：

$$d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}\,dt + \sqrt{\beta(t)}\,dW_t$$

**VP-SDE线性调度**：$\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})$

此时 $\bar\alpha_t = e^{-\frac{1}{2}\int_0^t\beta(s)ds} = e^{-\frac{1}{2}(\beta_{\min}t + \frac{\beta_{\max}-\beta_{\min}}{2}t^2)}$

---

## 11.5 训练时间步采样

训练时，时间步 $t$ 通常从 $\{1, ..., T\}$ 均匀采样。但不同 $t$ 对损失的贡献不同：

- 小 $t$（低噪声）：噪声预测容易，梯度小
- 大 $t$（高噪声）：噪声预测困难，梯度大

**Min-SNR加权**（Hang et al. 2023）：对损失加权以均衡不同时间步的贡献：

$$\lambda(t) = \min\left(\text{SNR}(t), \gamma\right), \quad \gamma = 5$$

大 $t$ 时SNR小，损失被放大；小 $t$ 时SNR大，但被截断到 $\gamma$。

**Logit-Normal采样**（SD3）：对连续时间 $t \in [0,1]$，用Logit-Normal分布采样更多中间时间步：

$$\text{logit}(t) = \log\frac{t}{1-t} \sim \mathcal{N}(\mu, \sigma^2)$$

---

## 代码实战

```python
"""
第十一章代码实战：噪声调度器实现与对比
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


# ============================================================
# 1. 各种调度器实现
# ============================================================

class LinearScheduler:
    """线性噪声调度（DDPM原始）"""
    
    def __init__(self, T: int = 1000, beta_min: float = 1e-4, beta_max: float = 0.02):
        self.T = T
        self.betas = torch.linspace(beta_min, beta_max, T)        # shape: (T,)
        self.alphas = 1.0 - self.betas                            # shape: (T,)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)       # shape: (T,)
    
    def snr(self, t: int) -> float:
        ab = self.alpha_bars[t].item()
        return ab / (1 - ab)


class CosineScheduler:
    """余弦噪声调度（Improved DDPM）"""
    
    def __init__(self, T: int = 1000, s: float = 0.008):
        self.T = T
        ts = torch.arange(T + 1, dtype=torch.float32) / T        # shape: (T+1,)
        f = torch.cos((ts + s) / (1 + s) * torch.pi / 2) ** 2   # shape: (T+1,)
        self.alpha_bars = (f / f[0]).clamp(min=1e-8)[:T]         # shape: (T,)
        
        # 从 alpha_bars 推导 betas
        alpha_bars_prev = torch.cat([torch.tensor([1.0]), self.alpha_bars[:-1]])
        self.alphas = self.alpha_bars / alpha_bars_prev
        self.betas = 1.0 - self.alphas
        self.betas = self.betas.clamp(max=0.999)
    
    def snr(self, t: int) -> float:
        ab = self.alpha_bars[t].item()
        return ab / (1 - ab)


class SigmoidScheduler:
    """Sigmoid噪声调度"""
    
    def __init__(self, T: int = 1000, start: float = -3.0, end: float = 3.0):
        self.T = T
        ts = torch.linspace(start, end, T)
        self.alpha_bars = torch.sigmoid(-ts)                      # shape: (T,)
        alpha_bars_prev = torch.cat([torch.tensor([1.0]), self.alpha_bars[:-1]])
        self.alphas = self.alpha_bars / alpha_bars_prev
        self.betas = (1.0 - self.alphas).clamp(max=0.999)
    
    def snr(self, t: int) -> float:
        ab = self.alpha_bars[t].item()
        return ab / max(1 - ab, 1e-8)


class EDMScheduler:
    """EDM噪声调度（连续时间，sigma参数化）"""
    
    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0,
                 rho: float = 7.0, n_steps: int = 40):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.n_steps = n_steps
        
        # 计算离散时间步对应的sigma
        steps = torch.arange(n_steps + 1, dtype=torch.float32)
        self.sigmas = (
            sigma_max ** (1/rho) +
            steps / n_steps * (sigma_min ** (1/rho) - sigma_max ** (1/rho))
        ) ** rho                                                   # shape: (n_steps+1,)
        self.sigmas[-1] = 0.0  # 最终步sigma=0
    
    def c_skip(self, sigma: torch.Tensor, sigma_data: float = 0.5) -> torch.Tensor:
        return sigma_data**2 / (sigma**2 + sigma_data**2)
    
    def c_out(self, sigma: torch.Tensor, sigma_data: float = 0.5) -> torch.Tensor:
        return sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    
    def c_in(self, sigma: torch.Tensor, sigma_data: float = 0.5) -> torch.Tensor:
        return 1.0 / (sigma**2 + sigma_data**2).sqrt()


# ============================================================
# 2. Min-SNR损失加权
# ============================================================

def min_snr_weight(scheduler, t_batch: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    """
    Min-SNR加权
    
    Args:
        t_batch: 时间步索引，shape (B,)
        gamma: SNR上界
    
    Returns:
        weights: shape (B,)
    """
    snr_values = torch.tensor(
        [scheduler.snr(t.item()) for t in t_batch],
        dtype=torch.float32
    )                                                              # shape: (B,)
    return torch.minimum(snr_values, torch.tensor(gamma)) / gamma


# ============================================================
# 3. Logit-Normal时间步采样
# ============================================================

def logit_normal_sample(n: int, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    """
    Logit-Normal时间步采样（用于Rectified Flow/SD3）
    
    Args:
        n: 样本数
    
    Returns:
        t: 时间步，shape (n,)，范围[0, 1]
    """
    u = torch.randn(n) * sigma + mu                               # shape: (n,)
    return torch.sigmoid(u)                                       # shape: (n,)


# ============================================================
# 4. 可视化对比
# ============================================================

def visualize_schedulers():
    """可视化不同调度器的SNR曲线"""
    T = 1000
    schedulers = {
        'Linear (DDPM)': LinearScheduler(T),
        'Cosine (iDDPM)': CosineScheduler(T),
        'Sigmoid': SigmoidScheduler(T),
    }
    
    t_steps = list(range(T))
    
    print("SNR对比（部分时间步）：")
    print(f"{'时间步':<10}", end='')
    for name in schedulers:
        print(f"{name:<25}", end='')
    print()
    
    for t in [0, 100, 250, 500, 750, 900, 999]:
        print(f"{t:<10}", end='')
        for name, sch in schedulers.items():
            snr = sch.snr(t)
            print(f"{snr:<25.6f}", end='')
        print()
    
    print("\nalpha_bar对比（部分时间步）：")
    print(f"{'时间步':<10}", end='')
    for name in schedulers:
        print(f"{name:<25}", end='')
    print()
    
    for t in [0, 100, 250, 500, 750, 900, 999]:
        print(f"{t:<10}", end='')
        for name, sch in schedulers.items():
            ab = sch.alpha_bars[t].item()
            print(f"{ab:<25.6f}", end='')
        print()
    
    # Min-SNR加权演示
    print("\nMin-SNR加权（γ=5）：")
    linear = LinearScheduler()
    for t_val in [0, 100, 500, 900, 999]:
        t_batch = torch.tensor([t_val])
        w = min_snr_weight(linear, t_batch, gamma=5.0)
        print(f"  t={t_val}: SNR={linear.snr(t_val):.4f}, 权重={w.item():.4f}")
    
    # Logit-Normal采样分布
    print("\nLogit-Normal时间步采样分布：")
    samples = logit_normal_sample(10000, mu=0.0, sigma=1.0)
    for percentile in [10, 25, 50, 75, 90]:
        p = torch.quantile(samples, percentile / 100)
        print(f"  P{percentile}: t={p.item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("噪声调度器对比")
    visualize_schedulers()
    
    print("\nEDM调度器：")
    edm = EDMScheduler(n_steps=40)
    print(f"  sigma范围: [{edm.sigmas[-2].item():.4f}, {edm.sigmas[0].item():.2f}]")
    print(f"  中间步sigma(20): {edm.sigmas[20].item():.4f}")
    
    sigma = torch.tensor([1.0])
    print(f"\nEDM预条件（sigma=1.0）:")
    print(f"  c_skip = {edm.c_skip(sigma).item():.4f}")
    print(f"  c_out  = {edm.c_out(sigma).item():.4f}")
    print(f"  c_in   = {edm.c_in(sigma).item():.4f}")
```

---

## 本章小结

| 调度方案 | 公式 | 优缺点 | 代表模型 |
|----------|------|--------|----------|
| 线性 | $\beta_t = \beta_{\min} + \frac{t}{T}(\beta_{\max}-\beta_{\min})$ | 简单；高分辨率效果差 | DDPM |
| 余弦 | $\bar\alpha_t = \cos^2\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right)$ | SNR均匀；效果好 | iDDPM, SD2 |
| EDM | 直接参数化 $\sigma(t)$，对数正态训练 | 最灵活；支持高阶求解器 | EDM, DiT-EDM |
| Rectified Flow | $x_t=(1-t)x_0+t\epsilon$ | 直线轨迹；少步采样 | SD3, FLUX |

---

## 练习题

### 基础题

**11.1** 对线性调度（$T=1000$，$\beta_{\min}=10^{-4}$，$\beta_{\max}=0.02$），计算 $t=500$ 时的 $\bar\alpha_{500}$ 和 $\text{SNR}(500)$（使用数值方法）。对余弦调度重复相同计算，对比两者的差异。

**11.2** 解释为什么余弦调度在低分辨率图像（$32\times32$）上改进不如高分辨率（$256\times256$）明显。（提示：考虑数据的频率内容。）

### 中级题

**11.3** 实现Min-SNR加权，对比有无Min-SNR加权时各时间步的有效学习率（梯度范数）。在二维扩散模型上训练，观察加权是否使各时间步的预测误差更均匀。

**11.4** 实现EDM调度器，并使用Heun ODE求解器（二阶精度）在二维数据上生成样本，对比与Euler法（DDIM）在相同步数下的质量差异。

### 提高题

**11.5** Karras et al. (2022) 证明存在一个最优时间步分布使得ODE求解误差均匀。推导（或数值验证）：对于VE-SDE（$dx = \sqrt{2t}dW$），最优离散时间步满足等式 $\sigma_{i+1}/\sigma_i = \text{const}$（几何序列）。

---

## 练习答案

**11.1** 线性：$\bar\alpha_{500} = \prod_{t=1}^{500}(1-\beta_t) \approx 0.334$，$\text{SNR} \approx 0.502$。余弦：$\bar\alpha_{500} \approx 0.691$，$\text{SNR} \approx 2.237$。余弦调度在中间步保留更多信号。

**11.2** 低分辨率图像的高频内容少，线性调度的早期大噪声对低频结构损害不大。高分辨率图像高频细节丰富，线性调度的前期过大噪声会过早破坏这些细节。

**11.3** 无Min-SNR时，$t \approx 0$的损失权重很小（梯度小），模型在低噪声步训练不足。加权后，各时间步有效贡献更均衡。

**11.4** Heun法：欧拉预测步 $\hat x_{t-1} = x_t + h\cdot d_t$，然后修正 $x_{t-1} = x_t + h(d_t + d_{\hat t-1})/2$。相同步数下，FID通常低1-2个点。

**11.5** 对VE-SDE，ODE轨迹曲率正比于 $d^2x/dt^2 \propto 1/t^2$。Euler误差 $\propto h_i^2 \cdot \kappa_i$。令 $h_i \cdot \sigma_i^{-1} = \text{const}$ → $\sigma_i / h_i = \sigma_{i+1}/h_{i+1}$，得到 $\sigma_{i+1}/\sigma_i = \text{const}$（几何序列）。

---

## 延伸阅读

1. **Nichol & Dhariwal (2021)**. *Improved Denoising Diffusion Probabilistic Models* — 余弦调度
2. **Karras et al. (2022)**. *Elucidating the Design Space of Diffusion-Based Generative Models* — EDM框架
3. **Hang et al. (2023)**. *Efficient Diffusion Training via Min-SNR Weighting Strategy* — Min-SNR
4. **Chen (2023)**. *On the Importance of Noise Scheduling for Diffusion Models* — 不同分辨率的调度对比

---

[← 上一章：DDIM确定性采样](./10-ddim-deterministic-sampling.md)

[下一章：加速采样综述 →](./12-accelerated-sampling-survey.md)

[返回目录](../README.md)
