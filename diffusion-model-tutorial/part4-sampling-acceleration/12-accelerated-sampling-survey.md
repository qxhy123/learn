# 第十二章：加速采样综述

> **本章导读**：DDIM之后，加速扩散模型采样成为研究热点。本章系统梳理主要加速方法：高阶ODE求解器（DPM-Solver、DEIS）、基于蒸馏的方法（Progressive Distillation、LCM），以及无训练加速（PLMS、UniPC）等，为理解最新高效扩散模型奠定基础。

**前置知识**：第10-11章，数值方法基础
**预计学习时间**：100分钟

---

## 学习目标

完成本章学习后，你将能够：
1. 理解扩散ODE的特殊结构（半线性），掌握指数积分器方法
2. 推导DPM-Solver的核心公式，理解其比DDIM精度更高的原因
3. 理解Progressive Distillation和LCM的蒸馏原理
4. 对比无训练加速方法（PLMS、UniPC）的实现思路
5. 根据实际需求（步数、质量、是否可重新训练）选择合适的加速策略

---

## 12.1 扩散ODE的特殊结构

### 半线性ODE

概率流ODE具有特殊的**半线性**（semi-linear）结构：

$$\frac{dx}{dt} = f(t)x + g(t)\cdot s_\theta(x, t)$$

其中 $f(t)x$ 是线性部分（可精确求解），$g(t)\cdot s_\theta$ 是非线性部分（神经网络）。

对于VP-SDE概率流ODE：

$$\frac{dx}{dt} = -\frac{\beta(t)}{2}x - \frac{\beta(t)}{2}\nabla_x\log p_t(x)$$

令 $\lambda_t = \ln(\bar\alpha_t / \sqrt{1-\bar\alpha_t})$（对数SNR），可以将ODE改写为：

$$\frac{d\hat x_\lambda}{d\lambda} = -e^\lambda \hat\epsilon_\theta(\hat x_\lambda, \lambda)$$

其中 $\hat x_\lambda$ 是变量替换后的等价量，非线性项只含 $\epsilon_\theta$。

### 精确积分线性部分

利用ODE的Duhamel原理，线性部分可以**精确积分**：

$$x_{t_1} = e^{\int_t^{t_1}f(s)ds}x_t + \int_t^{t_1}e^{\int_s^{t_1}f(r)dr}g(s)s_\theta(x_s, s)ds$$

DDIM相当于对非线性积分项 $\int \cdots s_\theta ds$ 做**零阶近似**（常数）。

---

## 12.2 DPM-Solver

### DPM-Solver-2（二阶精度）

Lu et al. (2022) 提出DPM-Solver，利用指数积分器对非线性项做**Taylor展开**：

$$s_\theta(x_s, s) \approx s_\theta(x_t, t) + (s - t)\frac{d}{ds}s_\theta(x_s, s)\bigg|_{s=t}$$

**DPM-Solver-2更新规则**（每步两次网络评估）：

$$x_{t_1} = \frac{\alpha_{t_1}}{\alpha_{t}}x_t - \sigma_{t_1}\left[\left(e^h - 1\right)\epsilon_\theta(x_t, t) + \frac{1}{2}\left(e^h - 1 - h\right)\frac{\epsilon_\theta(x_{t_m}, t_m) - \epsilon_\theta(x_t, t)}{r_1 h}\right]$$

其中 $h = \lambda_{t_1} - \lambda_t$，$t_m$ 是中间时间步。

**优势**：相同精度下，步数比DDIM减少约2-4倍（20步DDIM ≈ 10步DPM-Solver-2）。

### DPM-Solver++（改进版）

DPM-Solver++对 $x_0$ 而非 $\epsilon_\theta$ 进行Taylor展开（在高引导强度下更稳定）：

$$\hat x_0^{(t)} = (x_t - \sigma_t\epsilon_\theta(x_t, t)) / \alpha_t$$

在使用CFG时，$x_0$ 预测的误差比 $\epsilon$ 预测更小，FID进一步下降。

---

## 12.3 PLMS与UniPC

### PLMS（伪线性多步法）

Liu et al. (2022) 类比Adams-Bashforth多步法，缓存历史步的 $\epsilon_\theta$：

$$x_{t_{i-1}} \approx \cdots \text{（利用之前}k\text{步的噪声预测组合）}$$

**优势**：无需额外网络评估（单求值/步）；缺点：非自启动，前几步精度低。

### UniPC

Zhao et al. (2023) 将预测-校正（Predictor-Corrector）与DPM-Solver框架统一：

- **预测器**：DPM-Solver多步法（利用历史）
- **校正器**：在同一 $t$ 处用新的 $\epsilon_\theta$ 修正

UniPC在5-10步时表现优异，是目前无训练加速方法中效果最好之一。

---

## 12.4 知识蒸馏加速方法

### Progressive Distillation（渐进蒸馏）

Salimans & Ho (2022)：将2步的采样结果蒸馏为1步。

**算法**：
1. 用教师模型（$N$步）做2步采样：$x_T \to x_{T/2} \to \hat x_0$
2. 训练学生模型：用1步预测同样的 $\hat x_0$：$\mathcal{L} = \|\hat x_0^{student} - \hat x_0^{teacher}\|^2$
3. 学生成为新教师，重复（$N \to N/2 \to N/4 \to \cdots \to 1$）

**优点**：高质量，只需4步；**缺点**：需要多轮蒸馏训练。

### LCM（潜在一致性模型）

Luo et al. (2023) 将一致性蒸馏应用到**潜在空间**：

$$\mathcal{L}_{LCM}(\theta, \theta^-) = \mathbb{E}\left[\|f_\theta(x_{t_{n+1}}, t_{n+1}, c) - f_{\theta^-}(\hat x_{t_n}^{\psi,w}, t_n, c)\|^2\right]$$

其中 $c$ 是文本条件，$\hat x_{t_n}^{\psi,w}$ 是带引导的ODE步。

**关键改进**：将CFG引导整合进蒸馏过程（引导一致性蒸馏，Guided CM）。

**效果**：从预训练SD模型出发，只需训练数小时，可实现4步高质量生成。

### LCM-LoRA

LCM-LoRA将一致性蒸馏与LoRA结合：只训练LoRA权重（占总参数的1%），即可将任意微调后的SD模型转为少步模型，无需重新蒸馏！

---

## 12.5 方法对比

| 方法 | 类别 | 典型步数 | 需要重训练 | FID（COCO） |
|------|------|----------|------------|-------------|
| DDPM | SDE | 1000 | — | ~3.2 |
| DDIM | ODE | 50 | 否 | ~4.1 |
| DPM-Solver-2 | ODE | 20 | 否 | ~3.8 |
| DPM-Solver++ | ODE | 15-20 | 否 | ~3.6 |
| UniPC | ODE | 10 | 否 | ~3.9 |
| Prog. Distill. | 蒸馏 | 4 | 是 | ~4.5 |
| LCM | 蒸馏 | 4 | 是（轻量） | ~4.7 |
| Consistency Models | 一致性 | 1-2 | 是 | ~5-7 |

---

## 代码实战

```python
"""
第十二章代码实战：DPM-Solver-2实现
对比DDIM与DPM-Solver-2的步数-质量权衡
"""
import torch
import numpy as np
from typing import Optional, Callable


# ============================================================
# 1. DPM-Solver-2实现（简化版）
# ============================================================

class DPMSolver2:
    """
    DPM-Solver-2：二阶精度ODE求解器
    
    Reference: Lu et al. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic
    Model Sampling in Around 10 Steps" (NeurIPS 2022)
    """
    
    def __init__(self, alpha_bars: torch.Tensor):
        """
        Args:
            alpha_bars: 所有时间步的累积 alpha，shape: (T,)
        """
        self.alpha_bars = alpha_bars
        T = len(alpha_bars)
        
        # 计算 lambda_t = log(alpha_bar_t / sigma_t)
        sigma_bars = (1 - alpha_bars).sqrt()
        self.lambda_t = torch.log(alpha_bars.sqrt()) - torch.log(sigma_bars)  # shape: (T,)
    
    def _get_params(self, t: int):
        """获取时间步 t 的参数"""
        ab = self.alpha_bars[t]
        lam = self.lambda_t[t]
        alpha = ab.sqrt()
        sigma = (1 - ab).sqrt()
        return alpha, sigma, lam
    
    @torch.no_grad()
    def sample(self, model: Callable, shape: tuple,
               num_steps: int = 10, device: str = 'cpu') -> torch.Tensor:
        """
        DPM-Solver-2采样
        
        Args:
            model: 噪声预测函数 (x_t, t) -> eps_pred
            shape: (B, C, H, W)
            num_steps: 采样步数（10步即可达到DDIM 50步的质量）
        
        Returns:
            x0: shape (B, C, H, W)
        """
        T = len(self.alpha_bars)
        
        # 选择均匀间隔的时间步（按 lambda 均匀间隔）
        # 简化版：均匀按时间步选
        step_size = T // num_steps
        timesteps = list(range(T - 1, 0, -step_size))[:num_steps]  # 降序
        
        x = torch.randn(shape, device=device)  # shape: (B, C, H, W)
        
        for i, t in enumerate(timesteps):
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            
            alpha_t, sigma_t, lam_t = self._get_params(t)
            alpha_next, sigma_next, lam_next = self._get_params(t_next)
            h = lam_next - lam_t                # lambda 步长（负数，因为降噪）
            
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps_t = model(x, t_batch)           # shape: (B, C, H, W)，第一次评估
            
            # ---- 一阶预测步（得到中间点）----
            # 等价于 DDIM 一步
            r = 0.5  # 中间时间步比例
            h_half = r * h
            lam_mid = lam_t + h_half
            
            # 用线性插值找中间时间步的 alpha/sigma
            # 简化：用 (t + t_next) // 2
            t_mid = (t + t_next) // 2
            alpha_mid, sigma_mid, _ = self._get_params(t_mid)
            
            x_mid = (alpha_mid / alpha_t) * x - sigma_mid * (torch.exp(-h_half) - 1) * eps_t
            
            t_mid_batch = torch.full((shape[0],), t_mid, device=device, dtype=torch.long)
            eps_mid = model(x_mid, t_mid_batch)  # shape: (B, C, H, W)，第二次评估
            
            # ---- 二阶修正步 ----
            D = eps_t + (eps_mid - eps_t) / (2 * r)
            x = (alpha_next / alpha_t) * x - sigma_next * (torch.exp(-h) - 1) * D
        
        return x


# ============================================================
# 2. LCM损失（演示框架）
# ============================================================

class LCMLoss:
    """
    潜在一致性模型（LCM）蒸馏损失
    """
    
    def __init__(self, k: int = 20, w_min: float = 1.0, w_max: float = 15.0):
        """
        Args:
            k: ODE求解步数（每次更新只用k步之内的区间）
            w_min, w_max: 引导强度范围
        """
        self.k = k
        self.w_min = w_min
        self.w_max = w_max
    
    def boundary_cond(self, x: torch.Tensor) -> torch.Tensor:
        """边界条件的简单实现（将 x_0 映射到 x_0）"""
        return x.clamp(-1, 1)
    
    def loss(self, f_theta: Callable, f_ema: Callable,
             x0: torch.Tensor, t_n: torch.Tensor,
             ode_step: Callable, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        LCM蒸馏损失（简化版）
        
        Args:
            f_theta: 学生一致性模型（可训练）
            f_ema: 学生EMA（目标网络，固定参数）
            x0: 真实数据，shape (B, D)
            t_n: 当前时间步，shape (B,)
            ode_step: ODE推进一步的函数
            c: 文本条件（可选）
        
        Returns:
            loss: 标量
        """
        # 1. 加噪到 x_{t_{n+1}}
        eps = torch.randn_like(x0)
        x_t = x0 + t_n[:, None].float() * eps       # 简化的VE加噪
        
        # 2. ODE步：从 t_{n+1} 到 t_n（用教师/目标模型）
        x_t_prev = ode_step(x_t, t_n)               # shape: (B, D)
        
        # 3. 一致性函数评估
        target = f_ema(x_t_prev, t_n, c).detach()   # 目标（固定EMA参数）
        pred = f_theta(x_t, t_n + 1, c)             # 预测（可训练参数）
        
        # 4. 损失（LPIPS/L2）
        return ((pred - target) ** 2).mean()


# ============================================================
# 3. 基准测试
# ============================================================

def benchmark_methods():
    """对比不同方法的采样速度（用随机模型模拟）"""
    import time
    
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    class DummyModel:
        def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.randn_like(x)
    
    model = DummyModel()
    shape = (1, 4, 8, 8)  # 模拟潜在空间
    solver = DPMSolver2(alpha_bars)
    
    print("各方法采样时间对比（随机模型，仅测速）：")
    
    for steps in [5, 10, 15, 20]:
        start = time.time()
        _ = solver.sample(model, shape, num_steps=steps)
        elapsed = time.time() - start
        nfe = steps * 2  # DPM-Solver-2每步2次评估
        print(f"  DPM-Solver-2 {steps:2d}步 (NFE={nfe}): {elapsed:.3f}s")
    
    print("\n参考：DDIM 50步 = 50次评估 ≈ DPM-Solver-2 25步 = 50次评估")
    print("但DPM-Solver-2精度更高（二阶 vs 一阶）")
    
    print("\n方法选择建议：")
    print("  • 不可重训练 + 高质量: DPM-Solver++(15-20步) 或 UniPC(10步)")
    print("  • 可重训练 + 4步以内: LCM 或 Progressive Distillation")
    print("  • 实时交互（1-2步）: LCM-LoRA 或 Consistency Models")
    print("  • 研究场景: EDM + Heun(35步)")


if __name__ == "__main__":
    print("=" * 60)
    print("加速采样方法对比")
    benchmark_methods()
    
    print("\nDPM-Solver关键公式:")
    print("DDIM（一阶）: x_{t-1} = (α_{t-1}/α_t)·x_t - σ_{t-1}(e^h - 1)·ε_θ(x_t, t)")
    print("DPM-Solver-2（二阶）: 同上，但ε用中间步修正: D = ε_t + (ε_{mid} - ε_t)/(2r)")
```

---

## 本章小结

| 方法 | 核心思路 | 优点 | 缺点 |
|------|----------|------|------|
| DDIM | ODE欧拉法 | 简单，无需重训 | 一阶精度 |
| DPM-Solver | 指数积分器+Taylor展开 | 高精度，20步≈DDIM 50步 | 需要2次NFE/步 |
| UniPC | 预测-校正统一 | 10步高质量，无需重训 | 实现复杂 |
| Progressive Distillation | 步数减半蒸馏 | 4步高质量 | 多轮蒸馏耗时 |
| LCM | 一致性蒸馏+CFG | 4步，可与LoRA结合 | 需微调 |

---

## 练习题

### 基础题

**12.1** DPM-Solver-2每步需要2次网络评估（NFE=2），DDIM每步1次。若要达到相同精度，DPM-Solver-2在总NFE相同时，为什么精度仍优于DDIM？（提示：考虑数值积分的阶数。）

**12.2** 解释为什么DPM-Solver++在使用CFG时比DPM-Solver更稳定。（提示：思考高引导强度下 $\epsilon_\theta$ 的量级与 $\hat x_0$ 的量级差异。）

### 中级题

**12.3** 实现PLMS（伪线性多步法）：缓存前3步的噪声预测，用Adams-Bashforth公式组合，对比与DDIM在相同NFE下的生成质量。

**12.4** 实现渐进蒸馏的一步蒸馏（从32步蒸馏到16步）：用32步模型做2步采样作为教师目标，训练16步模型匹配，然后从16步蒸馏到8步。验证每次蒸馏后模型的生成质量。

### 提高题

**12.5** 推导DPM-Solver-2的二阶精度。具体地，展示DDIM（一阶近似）的截断误差为 $O(h^2)$（每步），而DPM-Solver-2的截断误差为 $O(h^3)$（每步），其中 $h = \lambda_{t+1} - \lambda_t$。

---

## 练习答案

**12.1** 数值积分的误差阶：一阶（DDIM欧拉法）截断误差 $O(h^2)$/步，总误差 $O(h)$（$N$步时 $h = \Delta T/N$）。二阶（DPM-Solver-2）截断误差 $O(h^3)$/步，总误差 $O(h^2)$。在相同NFE下，二阶方法精度远高于一阶。

**12.2** 高引导强度下 $\epsilon_\theta^{guided}$ 幅值极大（$\|\epsilon\| \propto w$），Taylor展开在 $\epsilon$ 上的误差也按 $w$ 放大。$\hat x_0$ 有范围约束（图像范围有界），Taylor展开误差较小。

**12.3** PLMS-3步公式：$D = (23\epsilon_t - 16\epsilon_{t-1} + 5\epsilon_{t-2})/12$，类比Adams-Bashforth三步法。

**12.4** 核心代码：`target = teacher_model(x_t, t+1); target = teacher_model(target, t); loss = ||student(x_{t+1}, t+1) - target||`

**12.5** 设 $g(\lambda) = e^{-\lambda}\hat\epsilon_\theta$，DDIM等价于 $\int_{0}^{h}g(\lambda_t)ds$（零阶）误差$O(h^2)$。DPM-Solver-2：$\int_0^h[g(\lambda_t) + s \cdot g'(\lambda_t)] ds$（一阶Taylor），误差$O(h^3)$。

---

## 延伸阅读

1. **Lu et al. (2022)**. *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling* — DPM-Solver原文
2. **Lu et al. (2022)**. *DPM-Solver++: Fast Solver for Guided Sampling* — DPM-Solver++
3. **Salimans & Ho (2022)**. *Progressive Distillation for Fast Sampling* — 渐进蒸馏
4. **Luo et al. (2023)**. *Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference* — LCM
5. **Zhao et al. (2023)**. *UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models* — UniPC

---

[← 上一章：噪声调度器设计](./11-noise-scheduler-design.md)

[下一章：分类器引导 →](../part5-conditional-generation/13-classifier-guidance.md)

[返回目录](../README.md)
