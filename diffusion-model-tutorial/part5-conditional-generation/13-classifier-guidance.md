# 第十三章：分类器引导

> **本章导读**：如何在不重新训练扩散模型的情况下实现条件生成？Dhariwal & Nichol (2021) 提出分类器引导（Classifier Guidance），通过在采样时使用预训练分类器的梯度来"引导"生成方向，实现类别条件生成。本章深入分析其数学原理和实现细节。

**前置知识**：第8-10章，基础神经网络
**预计学习时间**：90分钟

---

## 学习目标

完成本章学习后，你将能够：
1. 从贝叶斯视角推导条件分数函数，理解分类器梯度的数学作用
2. 实现分类器引导采样算法（DDPM和DDIM版本）
3. 理解引导强度 $s$ 对多样性-质量权衡的影响
4. 分析训练噪声分类器的必要性及其与干净图像分类器的区别
5. 比较分类器引导与无分类器引导（CFG）的优缺点

---

## 13.1 条件生成的贝叶斯视角

### 条件分布的分解

我们想从条件分布 $p(x|y)$ 采样（$y$ 是类别标签）。由贝叶斯定理：

$$p(x|y) = \frac{p(x)p(y|x)}{p(y)} \propto p(x)p(y|x)$$

取对数梯度（分数函数）：

$$\nabla_x \log p(x|y) = \nabla_x \log p(x) + \nabla_x \log p(y|x)$$

**关键洞察**：
- $\nabla_x \log p(x)$：**无条件分数函数**，由扩散模型学到
- $\nabla_x \log p(y|x)$：**分类器的梯度**，指向"更像类别 $y$"的方向

### 时间步条件分数

在扩散模型中，需要带噪声的条件分数：

$$\nabla_{x_t}\log p_t(x_t|y) = \nabla_{x_t}\log p_t(x_t) + \nabla_{x_t}\log p_t(y|x_t)$$

第一项由扩散模型提供（用 $-\epsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$ 近似），第二项需要**噪声感知分类器** $p_\phi(y|x_t)$。

---

## 13.2 引导强度

### 放大引导

Dhariwal & Nichol引入**引导强度** $s > 0$：

$$\tilde\epsilon_\theta(x_t, t, y) = \epsilon_\theta(x_t, t) - s\sqrt{1-\bar\alpha_t}\nabla_{x_t}\log p_\phi(y|x_t)$$

等价地，用修改后的分数：

$$\nabla_{x_t}\log\tilde p_t(x_t|y) = \nabla_{x_t}\log p_t(x_t) + s\cdot\nabla_{x_t}\log p_t(y|x_t)$$

**参数 $s$ 的效果**：
- $s = 0$：无条件生成（忽略类别）
- $s = 1$：贝叶斯最优条件分布
- $s > 1$：增强引导，更符合类别 $y$，但多样性降低
- $s \gg 1$：过度引导，生成典型（representative）但不自然的图像

**隐含的温度调整**：大 $s$ 使分布更尖锐，等效于降低"温度"：

$$\tilde p(x|y) \propto p(x)p(y|x)^s$$

---

## 13.3 噪声感知分类器的训练

### 为什么需要噪声分类器？

标准分类器在干净图像 $x_0$ 上训练，但扩散采样需要在各种噪声水平 $x_t$ 上计算梯度。

在 $t$ 较大时（$x_t$ 几乎是纯噪声），干净分类器的梯度几乎无意义！

### 训练细节

噪声感知分类器 $p_\phi(y|x_t, t)$ 的训练：

1. 对训练数据 $(x_0, y)$ 采样
2. 均匀采样时间步 $t \sim \text{Uniform}(1, T)$
3. 加噪：$x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$
4. 训练分类器预测：$\mathcal{L} = -\mathbb{E}[\log p_\phi(y|x_t, t)]$

网络结构：与扩散模型的U-Net类似，需要接受时间步 $t$ 作为额外输入。

---

## 13.4 实现细节

### DDPM版分类器引导

```
循环 t = T, T-1, ..., 1:
    epsilon_pred = model(x_t, t)          # 无条件噪声预测
    
    # 计算分类器梯度
    with torch.enable_grad():
        x_t.requires_grad_(True)
        log_prob = classifier(x_t, t)[target_class]
        grad = torch.autograd.grad(log_prob, x_t)[0]
    
    # 修改噪声预测
    epsilon_pred = epsilon_pred - s * sqrt(1 - alpha_bar_t) * grad
    
    # 正常DDPM更新
    x_{t-1} = ddpm_update(x_t, epsilon_pred, t)
```

**关键**：梯度计算需要 `torch.enable_grad()`（分类器前向必须在计算图中）。

### DDIM版分类器引导

类似，但在预测 $x_0$ 后修改方向向量：

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\hat x_0 + \sqrt{1-\bar\alpha_{t-1}}\underbrace{(\tilde\epsilon_\theta)}_{\text{修改后}}$$

---

## 代码实战

```python
"""
第十三章代码实战：分类器引导采样
在二维混合高斯上演示引导效果
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


# ============================================================
# 1. 简单扩散模型（二维示例）
# ============================================================

class SimpleDiffusionModel(nn.Module):
    """
    简单噪声预测网络（2D数据）
    输入：(x_t, t)，输出：预测噪声 eps
    """
    
    def __init__(self, hidden_dim: int = 128, T: int = 1000):
        super().__init__()
        self.T = T
        # 时间嵌入
        self.time_embed = nn.Embedding(T, 32)
        self.net = nn.Sequential(
            nn.Linear(2 + 32, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, 2)
            t: shape (B,) — 时间步整数
        Returns:
            eps_pred: shape (B, 2)
        """
        t_emb = self.time_embed(t)             # shape: (B, 32)
        h = torch.cat([x, t_emb], dim=-1)      # shape: (B, 34)
        return self.net(h)                      # shape: (B, 2)


# ============================================================
# 2. 噪声感知分类器
# ============================================================

class NoisyClassifier(nn.Module):
    """
    噪声感知分类器 p_phi(y | x_t, t)
    输入：(x_t, t)，输出：类别 logits
    """
    
    def __init__(self, n_classes: int = 2, hidden_dim: int = 128, T: int = 1000):
        super().__init__()
        self.time_embed = nn.Embedding(T, 32)
        self.net = nn.Sequential(
            nn.Linear(2 + 32, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, n_classes),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            logits: shape (B, n_classes)
        """
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


# ============================================================
# 3. 噪声调度
# ============================================================

class SimpleScheduler:
    def __init__(self, T: int = 1000):
        self.T = T
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)  # shape: (T,)


# ============================================================
# 4. 分类器引导采样
# ============================================================

class ClassifierGuidedSampler:
    """分类器引导采样器"""
    
    def __init__(self, model: SimpleDiffusionModel,
                 classifier: NoisyClassifier,
                 scheduler: SimpleScheduler):
        self.model = model
        self.classifier = classifier
        self.scheduler = scheduler
        self.T = scheduler.T
    
    @torch.no_grad()
    def ddpm_update(self, x: torch.Tensor, eps_pred: torch.Tensor,
                    t: int) -> torch.Tensor:
        """
        DDPM单步更新（使用修改后的eps_pred）
        
        Args:
            x: 当前状态，shape (B, 2)
            eps_pred: （修改后的）噪声预测，shape (B, 2)
            t: 时间步
        
        Returns:
            x_prev: shape (B, 2)
        """
        sch = self.scheduler
        alpha_bar_t = sch.alpha_bars[t]
        alpha_bar_t_prev = sch.alpha_bars[t - 1] if t > 1 else torch.tensor(1.0)
        beta_t = 1 - alpha_bar_t / alpha_bar_t_prev
        
        # 预测 x0
        x0_pred = (x - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
        x0_pred = x0_pred.clamp(-3, 3)
        
        # 后验均值
        coef1 = alpha_bar_t_prev.sqrt() * beta_t / (1 - alpha_bar_t)
        coef2 = (1 - alpha_bar_t_prev).sqrt() * (alpha_bar_t / alpha_bar_t_prev).sqrt() / (1 - alpha_bar_t)
        mean = coef1 * x0_pred + coef2 * x
        
        var = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
        
        if t > 1:
            noise = torch.randn_like(x)
            return mean + var.sqrt() * noise
        return mean
    
    def sample_with_guidance(self, n_samples: int, target_class: int,
                              guidance_scale: float = 1.0,
                              n_steps: int = 200) -> torch.Tensor:
        """
        带分类器引导的采样
        
        Args:
            n_samples: 生成样本数
            target_class: 目标类别
            guidance_scale: 引导强度 s（>1 增强引导）
            n_steps: 采样步数（使用子序列加速）
        
        Returns:
            x0_samples: shape (n_samples, 2)
        """
        device = next(self.model.parameters()).device
        x = torch.randn(n_samples, 2, device=device)  # shape: (n_samples, 2)
        
        sch = self.scheduler
        step_size = self.T // n_steps
        timesteps = list(range(self.T - 1, 0, -step_size))
        
        for t in timesteps:
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            
            # 无条件噪声预测
            with torch.no_grad():
                eps_uncond = self.model(x, t_batch)     # shape: (n_samples, 2)
            
            # 分类器梯度
            x_input = x.detach().requires_grad_(True)
            logits = self.classifier(x_input, t_batch)  # shape: (n_samples, n_classes)
            log_prob = F.log_softmax(logits, dim=-1)[:, target_class].sum()
            grad = torch.autograd.grad(log_prob, x_input)[0]  # shape: (n_samples, 2)
            
            # 修改噪声预测
            alpha_bar_t = sch.alpha_bars[t]
            eps_modified = eps_uncond - guidance_scale * (1 - alpha_bar_t).sqrt() * grad.detach()
            
            # 更新
            with torch.no_grad():
                x = self.ddpm_update(x, eps_modified, t)
        
        return x


# ============================================================
# 5. 可视化引导效果
# ============================================================

def demo_classifier_guidance():
    """演示不同引导强度的效果（使用随机初始化的网络，仅展示结构）"""
    torch.manual_seed(42)
    
    scheduler = SimpleScheduler(T=200)
    model = SimpleDiffusionModel(T=200)
    classifier = NoisyClassifier(n_classes=2, T=200)
    
    sampler = ClassifierGuidedSampler(model, classifier, scheduler)
    
    print("分类器引导采样（随机初始化模型，仅展示框架）：")
    
    for scale in [0.0, 1.0, 3.0, 7.5]:
        samples = sampler.sample_with_guidance(
            n_samples=50,
            target_class=0,
            guidance_scale=scale,
            n_steps=50
        )
        print(f"  引导强度 s={scale:.1f}: 均值=({samples[:,0].mean():.3f}, {samples[:,1].mean():.3f})")
    
    print("\n总结：")
    print("s=0.0: 无引导（无条件生成）")
    print("s=1.0: 标准贝叶斯引导")
    print("s=3.0: 增强引导（更确定属于目标类）")
    print("s=7.5: 强引导（高质量但多样性低）")
    
    print("\n分类器引导 vs CFG:")
    print("引导方式     | 需要单独训练分类器 | 推理时需要分类器 | 计算开销")
    print("分类器引导   |       是           |       是         |   高（两次前向+梯度）")
    print("无分类器引导 |       否           |       否         |   低（两次前向，无梯度）")


if __name__ == "__main__":
    print("=" * 50)
    print("分类器引导演示")
    demo_classifier_guidance()
    
    print("\n数学总结:")
    print("条件分数 = 无条件分数 + s × 分类器梯度")
    print("∇_x log p(x|y) = ∇_x log p(x) + s·∇_x log p(y|x)")
    print("修改噪声: ε̃ = ε_θ(x_t, t) - s·√(1-αbar_t)·∇_{x_t} log p_φ(y|x_t)")
```

---

## 本章小结

| 概念 | 数学形式 | 实践意义 |
|------|----------|----------|
| 条件分数 | $\nabla_x\log p(x\|y) = \nabla_x\log p(x) + \nabla_x\log p(y\|x)$ | 贝叶斯分解 |
| 修改噪声预测 | $\tilde\epsilon = \epsilon_\theta - s\sqrt{1-\bar\alpha_t}\nabla_{x_t}\log p_\phi(y\|x_t)$ | 引导采样 |
| 引导强度 $s$ | $s>1$ 增强引导，$s<1$ 削弱引导 | 质量-多样性权衡 |
| 噪声分类器 | 在 $(x_t, t)$ 上训练，而非 $(x_0)$ | 必须感知噪声水平 |

---

## 练习题

### 基础题

**13.1** 从贝叶斯定理推导：$\nabla_x\log p(x|y) = \nabla_x\log p(x) + \nabla_x\log p(y|x)$，并解释分类器引导为什么需要在含噪图像 $x_t$ 上计算梯度而非 $x_0$。

**13.2** 引导强度 $s>1$ 等效于从分布 $p(x)p(y|x)^s$ 采样。分析当 $s \to \infty$ 时，生成结果趋向何种图像？（提示：考虑类别后验 $p(y|x) = 1$ 意味着什么）

### 中级题

**13.3** 实现一个简单的噪声感知分类器，在二维月牙形数据集上训练，并可视化分类器梯度场（在不同噪声水平 $t = 0, 100, 500, 999$ 下分别绘制梯度方向）。

**13.4** 分析引导强度对FID分数和Inception Score的影响：
- 当 $s$ 从0增大到20时，哪个指标先恶化？
- 这说明了质量-多样性权衡的什么规律？

### 提高题

**13.5** 研究题：Dhariwal & Nichol (2021) 发现分类器引导配合改进的U-Net（ADM）超越了GAN（BigGAN-deep）。查阅论文，列出他们对U-Net架构的具体改进，并分析每项改进对生成质量的贡献（从消融实验角度）。

---

## 练习答案

**13.1** $\nabla_x\log p(x|y) = \nabla_x[\log p(x) + \log p(y|x) - \log p(y)]$，最后一项与 $x$ 无关，梯度为0。必须在 $x_t$ 上计算梯度，因为采样过程在噪声空间进行；在 $x_0$ 上的梯度无法直接用于 $x_t$ 的更新步骤。

**13.2** 当 $s \to \infty$：$p(x)p(y|x)^s$ 集中在使 $p(y|x) = 1$ 的点，即分类器最确定地认为属于类别 $y$ 的点。这些往往是"极端代表性"的图像，而非真实数据中的典型样本（例如，"完美无瑕的猫脸"而非真实猫照片的多样性）。

**13.3** 见代码实战框架，扩展`NoisyClassifier`在月牙形数据上训练即可。

**13.4** 通常Recall（多样性）先恶化，Precision（质量）随 $s$ 增大而提高。这体现了引导是在质量（类别符合度）和多样性（覆盖全类别分布）之间的权衡。

**13.5** 主要改进：(1) 增加更多注意力头；(2) 在更多分辨率层使用注意力；(3) 大容量模型（BigADM）；(4) 使用自适应归一化（AdaGN，类似AdaIN）注入类别信息。

---

## 延伸阅读

1. **Dhariwal & Nichol (2021)**. *Diffusion Models Beat GANs on Image Synthesis* — 分类器引导原文
2. **Song et al. (2021)**. *Score-Based Generative Modeling through SDEs* — 分数引导的SDE视角
3. **Ho & Salimans (2022)**. *Classifier-Free Diffusion Guidance* — 无分类器引导（下一章）

---

[← 上一章：加速采样综述](../part4-sampling-acceleration/12-accelerated-sampling-survey.md)

[下一章：无分类器引导（CFG） →](./14-classifier-free-guidance.md)

[返回目录](../README.md)
