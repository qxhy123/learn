# 第十四章：无分类器引导（CFG）

> **本章导读**：分类器引导需要单独训练噪声分类器，计算代价高。Ho & Salimans (2022) 提出无分类器引导（Classifier-Free Guidance, CFG），通过联合训练条件和无条件模型，在推理时只需两次网络评估即可实现强大的条件控制。CFG是现代扩散模型（SD、DALL-E、Midjourney）中最重要的技术之一。

**前置知识**：第13章（分类器引导），第8章（DDPM）
**预计学习时间**：90分钟

---

## 学习目标

完成本章学习后，你将能够：
1. 推导无分类器引导的数学基础，理解其与分类器引导的等价性
2. 实现CFG采样（条件/无条件混合推理）
3. 理解引导强度 $w$ 对生成质量和多样性的影响，以及最优 $w$ 的选择
4. 掌握条件Dropout训练策略（随机丢弃条件以训练无条件预测）
5. 理解CFG的变体：PNDM引导、负面提示词、动态CFG等

---

## 14.1 CFG的数学推导

### 等价性推导

回顾分类器引导：

$$\nabla_x\log p(x|y) = \nabla_x\log p(x) + s\cdot\nabla_x\log p(y|x)$$

注意贝叶斯定理：$\log p(y|x) = \log p(x|y) - \log p(x) + \text{const}$

所以：

$$\nabla_x\log p(x|y) + (s-1)\nabla_x\log p(y|x) = s\cdot\nabla_x\log p(x|y) - (s-1)\cdot\nabla_x\log p(x)$$

即，引导的条件分数等价于：

$$\tilde s(x, t, y) = s\cdot\nabla_x\log p(x|y) - (s-1)\cdot\nabla_x\log p(x)$$

**关键洞察**：不需要单独的分类器！只需要：
- $\epsilon_\theta(x_t, t, y)$：**条件**噪声预测
- $\epsilon_\theta(x_t, t, \emptyset)$：**无条件**噪声预测（$y = \emptyset$ 表示无条件）

### CFG修正噪声预测

$$\tilde\epsilon_\theta(x_t, t, y) = \epsilon_\theta(x_t, t, \emptyset) + w \cdot [\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset)]$$

等价地：

$$\tilde\epsilon_\theta = (1 + w)\epsilon_\theta(x_t, t, y) - w\cdot\epsilon_\theta(x_t, t, \emptyset)$$

其中 $w = s - 1$ 是引导强度（$w = 0$ 无引导，$w = 7.5$ 是SD默认值）。

**直觉**：CFG在"条件方向"上放大预测，同时减去"无条件分量"。

---

## 14.2 条件Dropout训练

### 联合训练策略

训练时，以概率 $p_{uncond}$（通常10-20%）随机将条件 $y$ 替换为空（$\emptyset$）：

```python
if random.random() < p_uncond:
    y = null_condition  # 固定的"空"嵌入
```

这样，同一个模型既能做条件预测（给定 $y$）也能做无条件预测（给定 $\emptyset$）。

**实现上**，$\emptyset$ 可以是：
- 全零嵌入
- 固定的"空文本"嵌入（""的CLIP嵌入）
- 可学习的空嵌入（null token）

### 推理时的两次前向传播

推理时，每步需要**两次**网络前向：

```python
# 1. 无条件预测
eps_uncond = model(x_t, t, null_condition)     # shape: (B, C, H, W)
# 2. 条件预测
eps_cond = model(x_t, t, text_embedding)       # shape: (B, C, H, W)
# 3. CFG混合
eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

**计算开销**：推理时间是无引导的2倍。优化技巧：将条件和无条件的 $x_t$ 拼接，做一次批量前向：

```python
# 批量合并，一次前向得到两个输出
x_double = torch.cat([x_t, x_t], dim=0)        # shape: (2B, C, H, W)
cond_double = torch.cat([null_cond, text_emb]) # shape: (2B, dim)
eps_double = model(x_double, t_double, cond_double)  # shape: (2B, C, H, W)
eps_uncond, eps_cond = eps_double.chunk(2)
```

---

## 14.3 引导强度的选择

### 质量-多样性权衡

| 引导强度 $w$ | 效果 |
|------------|------|
| $w = 0$ | 无引导，多样性最高，与条件无关 |
| $w = 1$ | 弱引导，接近贝叶斯最优 |
| $w = 7.5$ | 标准SD设置，高质量、符合提示词 |
| $w = 15$ | 强引导，图像过于典型，细节可能失真 |

**FID vs CLIP Score**：随 $w$ 增大，CLIP Score（文本-图像对齐度）提高，FID先降后升。

### 动态CFG

Chen et al. (2024) 的**AutoGuidance**和**PAG**（Perturbed Attention Guidance）发现：
- 早期步骤（$t$ 大）：高引导有益（确定构图）
- 晚期步骤（$t$ 小）：低引导更好（保留细节多样性）

**时变CFG**：$w(t) = w_{max} \cdot (t/T) + w_{min} \cdot (1 - t/T)$

---

## 14.4 负面提示词（Negative Prompts）

### 数学形式

负面提示词 $y_{neg}$（如"模糊, 低质量"）可以通过：

$$\tilde\epsilon = \epsilon_\theta(x_t, t, y_{neg}) + w\cdot[\epsilon_\theta(x_t, t, y_{pos}) - \epsilon_\theta(x_t, t, y_{neg})]$$

将无条件分量替换为负面条件分量：
- "推离"负面描述的方向
- "拉向"正面描述的方向

等价于在 $y_{neg}$ 和 $y_{pos}$ 之间做引导。

---

## 代码实战

```python
"""
第十四章代码实战：CFG完整实现
在DDIM采样器中集成CFG
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


# ============================================================
# 1. 条件U-Net（简化版）
# ============================================================

class ConditionalDiffusionModel(nn.Module):
    """
    条件噪声预测网络，支持条件/无条件预测
    用于演示CFG的条件Dropout训练
    """
    
    def __init__(self, data_dim: int = 2, cond_dim: int = 8,
                 hidden_dim: int = 128, T: int = 1000):
        super().__init__()
        self.T = T
        self.null_cond = nn.Parameter(torch.zeros(1, cond_dim))  # 可学习的空条件
        
        self.time_embed = nn.Sequential(
            nn.Embedding(T, 32),
            nn.Linear(32, 32), nn.SiLU(),
        )
        
        self.cond_proj = nn.Linear(cond_dim, 32)  # 条件嵌入投影
        
        self.net = nn.Sequential(
            nn.Linear(data_dim + 32 + 32, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                use_null: bool = False) -> torch.Tensor:
        """
        Args:
            x: 噪声数据，shape (B, data_dim)
            t: 时间步，shape (B,)
            cond: 条件嵌入，shape (B, cond_dim)；None则用null条件
            use_null: 强制使用null条件
        
        Returns:
            eps_pred: 预测噪声，shape (B, data_dim)
        """
        B = x.shape[0]
        t_emb = self.time_embed(t)                                # shape: (B, 32)
        
        if cond is None or use_null:
            cond_emb = self.null_cond.expand(B, -1)               # shape: (B, cond_dim)
        else:
            cond_emb = cond                                        # shape: (B, cond_dim)
        
        cond_proj = self.cond_proj(cond_emb)                      # shape: (B, 32)
        h = torch.cat([x, t_emb, cond_proj], dim=-1)             # shape: (B, data_dim+64)
        return self.net(h)                                        # shape: (B, data_dim)


# ============================================================
# 2. 条件Dropout训练
# ============================================================

def train_with_cfg_dropout(model: ConditionalDiffusionModel,
                            x_data: torch.Tensor,
                            cond_data: torch.Tensor,
                            n_steps: int = 5000,
                            p_uncond: float = 0.1,
                            T: int = 1000) -> None:
    """
    CFG训练：随机以概率p_uncond丢弃条件
    
    Args:
        x_data: 训练数据，shape (N, data_dim)
        cond_data: 对应条件，shape (N, cond_dim)
        p_uncond: 条件Dropout概率（通常0.1~0.2）
    """
    betas = torch.linspace(1e-4, 0.02, T)
    alpha_bars = torch.cumprod(1 - betas, dim=0)                  # shape: (T,)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    N = x_data.shape[0]
    
    for step in range(n_steps):
        idx = torch.randperm(N)[:256]
        x_batch = x_data[idx]                                     # shape: (256, 2)
        c_batch = cond_data[idx]                                  # shape: (256, cond_dim)
        
        t = torch.randint(0, T, (256,))                           # shape: (256,)
        eps = torch.randn_like(x_batch)                           # shape: (256, 2)
        
        # 加噪
        sqrt_ab = alpha_bars[t].sqrt()[:, None]                   # shape: (256, 1)
        sqrt_1mab = (1 - alpha_bars[t]).sqrt()[:, None]           # shape: (256, 1)
        x_t = sqrt_ab * x_batch + sqrt_1mab * eps                # shape: (256, 2)
        
        # 条件Dropout：以p_uncond的概率使用null条件
        use_null = torch.rand(256) < p_uncond                     # shape: (256,) bool
        cond_input = c_batch.clone()
        cond_input[use_null] = model.null_cond.expand(use_null.sum(), -1).detach()
        
        # 预测
        eps_pred = model(x_t, t, cond_input)                     # shape: (256, 2)
        
        loss = F.mse_loss(eps_pred, eps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 1000 == 0:
            print(f"Step {step+1}: loss={loss.item():.4f}")


# ============================================================
# 3. CFG采样器
# ============================================================

class CFGSampler:
    """
    CFG采样（DDIM + CFG）
    """
    
    def __init__(self, model: ConditionalDiffusionModel, T: int = 1000):
        self.model = model
        self.T = T
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)            # shape: (T,)
    
    @torch.no_grad()
    def sample(self, cond: torch.Tensor, guidance_scale: float = 7.5,
               num_steps: int = 50, neg_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        CFG采样
        
        Args:
            cond: 正向条件，shape (B, cond_dim)
            guidance_scale: 引导强度 w
            neg_cond: 负面条件（None则用null条件），shape (B, cond_dim)
        
        Returns:
            x0: 生成样本，shape (B, data_dim)
        """
        B = cond.shape[0]
        device = cond.device
        data_dim = 2  # 假设2D数据
        
        x = torch.randn(B, data_dim, device=device)               # shape: (B, 2)
        
        sch = self.alpha_bars
        step_size = self.T // num_steps
        timesteps = list(range(self.T - 1, 0, -step_size))
        
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # ---- CFG：两次前向传播 ----
            # 方式1：分开计算
            eps_cond = self.model(x, t_batch, cond)               # shape: (B, 2)
            
            if neg_cond is not None:
                eps_uncond = self.model(x, t_batch, neg_cond)     # 负面提示词
            else:
                eps_uncond = self.model(x, t_batch, use_null=True)  # null条件
            
            # CFG混合
            eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            
            # ---- DDIM更新 ----
            ab_t = sch[t]
            ab_t_prev = sch[t_prev] if t_prev > 0 else torch.tensor(1.0)
            
            x0_pred = (x - (1 - ab_t).sqrt() * eps_cfg) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-3, 3)
            
            direction = (1 - ab_t_prev).sqrt() * eps_cfg
            x = ab_t_prev.sqrt() * x0_pred + direction
        
        return x
    
    @torch.no_grad()
    def batch_cfg_forward(self, x: torch.Tensor, t: torch.Tensor,
                          cond: torch.Tensor) -> tuple:
        """
        优化版CFG：批量合并条件/无条件，一次前向
        
        Returns:
            (eps_uncond, eps_cond): 各 shape (B, data_dim)
        """
        B = x.shape[0]
        
        # 拼接：[x_uncond, x_cond]
        x_double = torch.cat([x, x], dim=0)                       # shape: (2B, 2)
        t_double = torch.cat([t, t], dim=0)                       # shape: (2B,)
        
        # 拼接条件：[null, cond]
        null_cond = self.model.null_cond.expand(B, -1)            # shape: (B, cond_dim)
        cond_double = torch.cat([null_cond, cond], dim=0)         # shape: (2B, cond_dim)
        
        eps_double = self.model(x_double, t_double, cond_double)  # shape: (2B, 2)
        eps_uncond, eps_cond = eps_double.chunk(2, dim=0)         # each: (B, 2)
        
        return eps_uncond, eps_cond


# ============================================================
# 4. 引导强度对比演示
# ============================================================

def demo_guidance_scale():
    """演示不同引导强度的效果"""
    torch.manual_seed(42)
    T = 200
    model = ConditionalDiffusionModel(data_dim=2, cond_dim=4, T=T)
    sampler = CFGSampler(model, T=T)
    
    # 创建测试条件
    cond = torch.randn(20, 4)  # 随机条件
    
    print("不同引导强度的生成结果（随机初始化模型，仅展示框架）：")
    for w in [0.0, 1.0, 3.0, 7.5, 15.0]:
        samples = sampler.sample(cond, guidance_scale=w, num_steps=20)
        print(f"  w={w:4.1f}: 样本范围=[{samples.min():.3f}, {samples.max():.3f}]，"
              f"方差={samples.var():.4f}")
    
    print("\nCFG公式:")
    print("ε̃ = ε_uncond + w·(ε_cond - ε_uncond)")
    print("等价于: ε̃ = (1+w)·ε_cond - w·ε_uncond")
    print(f"w=0: 无引导")
    print(f"w=7.5: SD默认")
    print(f"负面提示词: 将ε_uncond替换为ε_neg")


if __name__ == "__main__":
    print("=" * 60)
    print("CFG（无分类器引导）演示")
    demo_guidance_scale()
    
    print("\n效率优化：批量前向（一次前向计算条件+无条件）")
    model = ConditionalDiffusionModel(data_dim=2, cond_dim=4, T=200)
    sampler = CFGSampler(model, T=200)
    x = torch.randn(4, 2)
    t = torch.zeros(4, dtype=torch.long)
    cond = torch.randn(4, 4)
    eps_u, eps_c = sampler.batch_cfg_forward(x, t, cond)
    print(f"批量CFG前向成功: eps_uncond={eps_u.shape}, eps_cond={eps_c.shape}")
```

---

## 本章小结

| 概念 | 数学形式 | 实践 |
|------|----------|------|
| CFG修正 | $\tilde\epsilon = \epsilon_{uncond} + w(\epsilon_{cond} - \epsilon_{uncond})$ | 推理时2次前向 |
| 条件Dropout | 训练时以 $p=0.1$ 丢弃条件 | 同一模型兼顾有/无条件 |
| 负面提示词 | 用负向条件代替null条件 | 排除不想要的特征 |
| 引导强度 $w$ | $w=7.5$ 为SD默认 | 质量-多样性权衡 |

---

## 练习题

### 基础题

**14.1** 验证：$\tilde\epsilon = \epsilon_{uncond} + w(\epsilon_{cond} - \epsilon_{uncond})$ 与 $(1+w)\epsilon_{cond} - w\cdot\epsilon_{uncond}$ 完全等价。证明当 $w=1$ 时，CFG等价于贝叶斯最优条件生成。

**14.2** 解释为什么条件Dropout比单独训练无条件模型更好：若分别训练有条件和无条件的两个模型，CFG是否还能有效工作？

### 中级题

**14.3** 实现负面提示词采样：给定正向提示词"一只猫（条件A）"和负向提示词"模糊，低质量（条件B）"，实现采样并可视化不同引导方向对生成图像的影响。

**14.4** 实现动态CFG（时变引导强度）：前50%时间步使用 $w=10$，后50%使用 $w=3$，与固定 $w=7.5$ 对比生成质量（FID）和文本对齐度（CLIP Score）。

### 提高题

**14.5** PAG（Perturbed Attention Guidance）不使用无条件模型，而是用扰动的注意力层作为"引导基准"。推导其等价的分数函数形式，并解释为什么它在单步/少步采样中比CFG更稳定。

---

## 练习答案

**14.1** $(1+w)\epsilon_{cond} - w\epsilon_{uncond} = \epsilon_{cond} + w\epsilon_{cond} - w\epsilon_{uncond} = \epsilon_{uncond} + (w+1)(\epsilon_{cond}-\epsilon_{uncond}) + \epsilon_{uncond} - \epsilon_{uncond}$... 直接展开：$\epsilon_{uncond}+w(\epsilon_{cond}-\epsilon_{uncond}) = \epsilon_{uncond}+w\epsilon_{cond}-w\epsilon_{uncond} = (1+w)\epsilon_{cond}-w\epsilon_{uncond}$。✓ 当$w=1$：$\tilde\epsilon = 2\epsilon_{cond} - \epsilon_{uncond}$，对应分数 $2\nabla\log p(x|y) - \nabla\log p(x) = \nabla\log p(y|x)$，即贝叶斯条件分数。

**14.2** 若分开训练两个模型，两者的特征空间不对齐（不同初始化），$\epsilon_{cond} - \epsilon_{uncond}$ 的差值无意义。条件Dropout使同一模型的两类预测共享相同特征空间，差值才代表"条件方向"。

**14.3** 关键代码：`eps = eps_neg + w * (eps_pos - eps_neg)`，效果：同时向"好猫"方向推进、远离"模糊低质量"区域。

**14.4** 动态CFG的直觉：早期步骤（$t$大）决定构图，需要强引导；晚期步骤（$t$小）完善细节，弱引导保留多样性。

**14.5** PAG用自注意力层被置换（如full-attention matrix替换为恒等注意力）的模型作为基准，等价于一种结构化的"无引导"基准。优点：不需要无条件训练；少步时比null条件更稳定的梯度。

---

## 延伸阅读

1. **Ho & Salimans (2022)**. *Classifier-Free Diffusion Guidance* — CFG原文
2. **Nichol et al. (2021)**. *GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models* — 大规模CFG应用
3. **Ahn et al. (2024)**. *Tuning-Free Image Customization with Image and Text Guidance* — 负面提示词高级应用

---

[← 上一章：分类器引导](./13-classifier-guidance.md)

[下一章：文本条件生成 →](./15-text-conditional-generation.md)

[返回目录](../README.md)
