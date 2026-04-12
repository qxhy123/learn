# 第十五章：文本条件生成

> **本章导读**：文本到图像生成（Text-to-Image）是扩散模型最具影响力的应用。本章介绍如何将文本信息注入扩散模型：CLIP文本编码器、交叉注意力机制，以及完整的文本条件扩散模型架构（以Stable Diffusion为例）。本章也介绍图像编辑（InstructPix2Pix）和视觉-语言条件化。

**前置知识**：第14章（CFG），第16-17章（U-Net与注意力）—— 可先看概述
**预计学习时间**：100分钟

---

## 学习目标

完成本章学习后，你将能够：
1. 理解CLIP文本编码器的工作原理，掌握文本特征的提取方式
2. 推导交叉注意力机制，理解文本如何注入U-Net
3. 实现简化版的文本条件扩散模型
4. 理解InstructPix2Pix的图像编辑原理
5. 分析多条件融合（文本+图像、文本+掩码）的实现策略

---

## 15.1 文本编码器

### CLIP编码器

CLIP（Contrastive Language-Image Pretraining，Radford et al. 2021）通过对比学习将文本和图像映射到共享语义空间。

**文本编码器**：基于Transformer（GPT-2风格）：

$$\text{text\_feat} = \text{CLIP\_TextEncoder}(\text{tokenize}(\text{prompt}))$$

对于长度 $L$ 的token序列（SD中 $L=77$），输出形状为 $(B, L, d)$，$d=768$（CLIP-L）或 $d=1024$（CLIP-G）。

**CLIP嵌入的特性**：
- 每个token有独立的语义（不只是最后一个）
- 包含丰富的视觉语义（因为与图像对齐训练）
- 最大长度限制：77 tokens（超出截断）

### T5文本编码器

Imagen（Saharia et al. 2022）发现，更大的T5文本编码器（特别是T5-XXL，4.6B参数）效果优于CLIP：

- T5编码器保留更完整的语言语义
- SD3使用T5-XXL + CLIP-L + CLIP-G三个编码器的拼接

---

## 15.2 交叉注意力

### 文本注入机制

文本特征通过**交叉注意力**（Cross-Attention）注入U-Net的中间层：

**Query**来自图像特征，**Key/Value**来自文本特征：

$$\text{CrossAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q = W_Q \phi(x_t)$：图像特征线性投影，shape $(B, HW, d_k)$
- $K = W_K \psi_y$：文本特征线性投影，shape $(B, L, d_k)$
- $V = W_V \psi_y$：文本特征线性投影，shape $(B, L, d_v)$

$\phi(x_t)$ 是U-Net中间特征（flatten后的空间维度），$\psi_y$ 是文本编码序列。

**输出**：shape $(B, HW, d_v)$，reshape回空间维度 $(B, H, W, d_v)$。

### 注意力图的意义

交叉注意力的权重矩阵 $A \in \mathbb{R}^{HW \times L}$ 对每个空间位置 $(h, w)$ 指定了最关注哪个文本token。这提供了**空间-语义对齐**的可解释性——Prompt-to-Prompt等方法利用注意力图实现图像编辑。

---

## 15.3 条件U-Net架构（简化版）

### 整体结构

文本条件扩散模型的U-Net包含三种条件注入：

1. **时间步嵌入**（加法注入，通过AdaGN）：每一层都加入时间信息
2. **文本条件**（交叉注意力注入）：在各分辨率的中间层注入
3. **图像条件**（可选，通道拼接或加法）：用于图像编辑等任务

典型结构：

```
输入 x_t (B, 4, H/8, W/8)  ← 潜在空间
↓
Encoder Block × N（每层含 ResBlock + CrossAttn + DownSample）
↓
Middle Block（ResBlock + SelfAttn + CrossAttn）
↓
Decoder Block × N（每层含 ResBlock + CrossAttn + UpSample）
↓
输出 eps_pred (B, 4, H/8, W/8)
```

---

## 15.4 InstructPix2Pix

### 图像编辑方法

Brooks et al. (2022) 提出InstructPix2Pix：给定图像 $x$ 和指令（"将天空改成夜晚"），生成编辑后的图像。

**关键思想**：双重CFG引导——同时对图像条件和文本条件做引导：

$$\tilde\epsilon = \epsilon(\emptyset, \emptyset) + w_{img}[\epsilon(x, \emptyset) - \epsilon(\emptyset, \emptyset)] + w_{txt}[\epsilon(x, c) - \epsilon(x, \emptyset)]$$

其中：
- $\epsilon(\emptyset, \emptyset)$：无图像、无文本的无条件预测
- $\epsilon(x, \emptyset)$：有图像条件、无文本的预测
- $\epsilon(x, c)$：有图像条件、有文本条件的预测

**训练**：需要 $(原图, 指令, 编辑后图) $ 的三元组数据（通过GPT-4 + SD自动生成）。

---

## 代码实战

```python
"""
第十五章代码实战：文本条件扩散模型（简化版）
实现交叉注意力文本注入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ============================================================
# 1. 简单文本编码器（模拟CLIP）
# ============================================================

class SimpleTextEncoder(nn.Module):
    """
    简单文本编码器（替代CLIP，用于演示）
    将词汇索引映射为序列嵌入
    """
    
    def __init__(self, vocab_size: int = 1000, seq_len: int = 16,
                 embed_dim: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: shape (B, seq_len) — token索引
        
        Returns:
            text_feat: shape (B, seq_len, embed_dim)
        """
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)  # shape: (1, L)
        x = self.token_embed(input_ids) + self.pos_embed(pos)         # shape: (B, L, embed_dim)
        return self.transformer(x)                                    # shape: (B, L, embed_dim)


# ============================================================
# 2. 交叉注意力层
# ============================================================

class CrossAttention(nn.Module):
    """
    文本-图像交叉注意力
    Query 来自图像，Key/Value 来自文本
    """
    
    def __init__(self, query_dim: int, context_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Query投影（从图像特征）
        self.q = nn.Linear(query_dim, query_dim, bias=False)
        # Key/Value投影（从文本特征）
        self.k = nn.Linear(context_dim, query_dim, bias=False)
        self.v = nn.Linear(context_dim, query_dim, bias=False)
        
        self.out_proj = nn.Linear(query_dim, query_dim)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 图像特征（Query来源），shape (B, HW, query_dim)
            context: 文本特征（Key/Value来源），shape (B, L, context_dim)
            mask: 注意力掩码（可选），shape (B, HW, L)
        
        Returns:
            out: shape (B, HW, query_dim)
        """
        B, S, _ = x.shape    # S = HW（空间维度展平）
        L = context.shape[1]  # L = 文本序列长度
        
        # 计算Q, K, V
        Q = self.q(x)                                              # shape: (B, S, query_dim)
        K = self.k(context)                                        # shape: (B, L, query_dim)
        V = self.v(context)                                        # shape: (B, L, query_dim)
        
        # 拆分多头
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            B, N, D = t.shape
            return t.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            # shape: (B, n_heads, N, head_dim)
        
        Q, K, V = map(split_heads, [Q, K, V])
        
        # 注意力权重
        attn = (Q @ K.transpose(-2, -1)) * self.scale             # shape: (B, n_heads, S, L)
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attn_weights = attn.softmax(dim=-1)                        # shape: (B, n_heads, S, L)
        
        # 注意力输出
        out = (attn_weights @ V)                                   # shape: (B, n_heads, S, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, S, -1)           # shape: (B, S, query_dim)
        
        return self.out_proj(out)                                  # shape: (B, S, query_dim)


# ============================================================
# 3. 带文本条件的Transformer块
# ============================================================

class TextConditionedBlock(nn.Module):
    """
    单个文本条件Transformer块
    包含：自注意力 + 交叉注意力 + FFN
    """
    
    def __init__(self, dim: int, context_dim: int, n_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, context_dim, n_heads)
        
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像特征，shape (B, HW, dim)
            context: 文本特征，shape (B, L, context_dim)
        
        Returns:
            out: shape (B, HW, dim)
        """
        # 自注意力（图像内部）
        x_norm = self.norm1(x)
        sa_out, _ = self.self_attn(x_norm, x_norm, x_norm)        # shape: (B, HW, dim)
        x = x + sa_out
        
        # 交叉注意力（文本-图像）
        x = x + self.cross_attn(self.norm2(x), context)           # shape: (B, HW, dim)
        
        # FFN
        x = x + self.ffn(self.norm3(x))                           # shape: (B, HW, dim)
        
        return x


# ============================================================
# 4. 简化版文本条件扩散模型
# ============================================================

class TextCondDiffusion(nn.Module):
    """
    简化版文本条件扩散模型
    输入：(x_t, t, text_ids)，输出：预测噪声
    """
    
    def __init__(self, img_size: int = 8, dim: int = 64,
                 text_dim: int = 64, T: int = 1000,
                 vocab_size: int = 100, seq_len: int = 16):
        super().__init__()
        self.img_size = img_size
        self.dim = dim
        
        # 文本编码器
        self.text_encoder = SimpleTextEncoder(vocab_size, seq_len, text_dim)
        
        # 时间步嵌入
        self.time_embed = nn.Embedding(T, dim)
        
        # 图像嵌入（将像素patch展平）
        self.img_embed = nn.Linear(1, dim)  # 假设灰度图
        
        # Transformer块（含交叉注意力）
        self.blocks = nn.ModuleList([
            TextConditionedBlock(dim, text_dim, n_heads=4)
            for _ in range(4)
        ])
        
        # 输出头
        self.out = nn.Linear(dim, 1)
        
        # Null条件（用于CFG训练）
        self.null_text = nn.Parameter(torch.zeros(1, seq_len, text_dim))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor,
                text_ids: Optional[torch.Tensor] = None,
                use_null: bool = False) -> torch.Tensor:
        """
        Args:
            x: 噪声图像，shape (B, 1, H, W)
            t: 时间步，shape (B,)
            text_ids: 文本token索引，shape (B, seq_len)；None时用null条件
            use_null: 强制使用null条件（用于CFG无条件预测）
        
        Returns:
            eps_pred: 预测噪声，shape (B, 1, H, W)
        """
        B, C, H, W = x.shape
        
        # 图像特征（B, HW, dim）
        x_flat = x.reshape(B, C, H * W).permute(0, 2, 1)         # shape: (B, HW, 1)
        x_feat = self.img_embed(x_flat)                           # shape: (B, HW, dim)
        
        # 时间嵌入（加到图像特征）
        t_emb = self.time_embed(t)[:, None, :]                    # shape: (B, 1, dim)
        x_feat = x_feat + t_emb                                   # shape: (B, HW, dim)
        
        # 文本特征
        if use_null or text_ids is None:
            text_feat = self.null_text.expand(B, -1, -1)          # shape: (B, seq_len, text_dim)
        else:
            text_feat = self.text_encoder(text_ids)               # shape: (B, seq_len, text_dim)
        
        # Transformer块
        for block in self.blocks:
            x_feat = block(x_feat, text_feat)                     # shape: (B, HW, dim)
        
        # 输出预测噪声
        out = self.out(x_feat)                                    # shape: (B, HW, 1)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)           # shape: (B, 1, H, W)
        
        return out


# ============================================================
# 5. 完整的文本到图像采样流程
# ============================================================

def text_to_image_demo():
    """演示文本条件生成的完整流程"""
    torch.manual_seed(42)
    
    model = TextCondDiffusion(img_size=8, dim=32, text_dim=32, T=200)
    
    B = 2
    # 模拟文本输入（token ids）
    text_ids = torch.randint(0, 100, (B, 16))  # shape: (B, 16)
    
    # CFG采样
    T = 200
    betas = torch.linspace(1e-4, 0.02, T)
    alpha_bars = torch.cumprod(1 - betas, dim=0)
    
    x = torch.randn(B, 1, 8, 8)               # 从噪声开始
    guidance_scale = 7.5
    
    n_steps = 20
    step_size = T // n_steps
    timesteps = list(range(T - 1, 0, -step_size))
    
    print(f"文本到图像生成（CFG w={guidance_scale}，步数={n_steps}）：")
    
    with torch.no_grad():
        for i, t in enumerate(timesteps[:3]):  # 仅演示前3步
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            t_batch = torch.full((B,), t, dtype=torch.long)
            
            # CFG：两次前向
            eps_cond = model(x, t_batch, text_ids=text_ids)
            eps_uncond = model(x, t_batch, use_null=True)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            
            # DDIM更新
            ab = alpha_bars[t]
            ab_prev = alpha_bars[t_prev] if t_prev > 0 else torch.tensor(1.0)
            x0_pred = (x - (1-ab).sqrt() * eps) / ab.sqrt()
            x = ab_prev.sqrt() * x0_pred + (1-ab_prev).sqrt() * eps
            
            print(f"  步骤 {i+1} (t={t}): x范围=[{x.min():.3f}, {x.max():.3f}]")
    
    print("\n架构总结：")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")
    print(f"  文本编码器参数: {sum(p.numel() for p in model.text_encoder.parameters()):,}")
    print(f"  交叉注意力已集成在 {len(model.blocks)} 个Transformer块中")
    
    print("\n文本条件对比：")
    with torch.no_grad():
        eps_with_text = model(x, t_batch, text_ids=text_ids)
        eps_null = model(x, t_batch, use_null=True)
        diff = (eps_with_text - eps_null).abs().mean()
        print(f"  条件vs无条件噪声预测差异（均值绝对差）: {diff.item():.6f}")
        print(f"  （训练未进行，差异接近0属于正常初始化状态）")


if __name__ == "__main__":
    print("=" * 60)
    print("文本条件扩散模型演示")
    text_to_image_demo()
    
    print("\n交叉注意力单独测试：")
    cross_attn = CrossAttention(query_dim=64, context_dim=32, n_heads=4)
    img_feat = torch.randn(2, 16, 64)   # (B=2, HW=16, dim=64)
    text_feat = torch.randn(2, 8, 32)   # (B=2, L=8, context_dim=32)
    out = cross_attn(img_feat, text_feat)
    print(f"  交叉注意力输出: {out.shape}")  # 应为 (2, 16, 64)
    
    print("\nInstructPix2Pix双重引导公式：")
    print("ε̃ = ε(∅,∅) + w_img·[ε(img,∅) - ε(∅,∅)] + w_txt·[ε(img,c) - ε(img,∅)]")
    print("其中：w_img=1.5（图像引导强度），w_txt=7.5（文本指令强度）")
```

---

## 本章小结

| 组件 | 作用 | 输入 → 输出 |
|------|------|-------------|
| CLIP文本编码器 | 文本→语义嵌入序列 | $(B, L)$ → $(B, L, d)$ |
| 交叉注意力 | 文本注入图像特征 | Q:$(B,HW,d)$, KV:$(B,L,d)$ → $(B,HW,d)$ |
| 条件Dropout | 训练无条件分支 | 以概率 $p$ 替换为null条件 |
| CFG | 推理时放大文本引导 | 2次前向 → 混合预测 |
| 负面提示词 | 排除不想要特征 | 用负向条件代替null条件 |

---

## 练习题

### 基础题

**15.1** 在交叉注意力中，为什么Query来自图像特征而不是文本特征？交换Q和K/V有什么后果？

**15.2** CLIP文本编码器的最大输入长度为77个token，超出会被截断。分析这对长提示词的影响，并描述一种缓解方法（例如，多段编码取均值）。

### 中级题

**15.3** 实现**Prompt-to-Prompt**（Hertz et al. 2022）的核心：在两次生成中（不同提示词）共享交叉注意力图，实现保持图像结构但改变某部分语义的编辑效果（例如，"一只猫坐在桌子上" → "一只狗坐在桌子上"）。

**15.4** 实现InstructPix2Pix的双重CFG采样：给定源图像条件和编辑指令，用双重引导生成编辑后图像，并对比不同的 $(w_{img}, w_{txt})$ 组合。

### 提高题

**15.5** 研究题：DALL-E 3与SD的文本条件注入有何不同？查阅论文，分析以下方面：(a) 文本编码器选择（CLIP vs T5 vs 自训练）；(b) 是否使用多个文本编码器及其融合方式；(c) 对复杂文本（如物体关系、多对象）的处理能力差异。

---

## 练习答案

**15.1** Query来自图像：每个空间位置决定"关注哪些文本信息"，即图像向文本查询。若Q来自文本，则是文本向图像查询，输出维度是文本序列，无法映射回图像空间。

**15.2** 截断影响：长提示词的末尾部分被忽略（如"a cat, highly detailed, cinematic lighting"中后面的修饰词可能丢失）。缓解：将提示词分段，分别编码后取均值或连接（SD的CLIP skip技巧）。

**15.3** 关键：用钩子（hook）提取并覆写交叉注意力权重。两次前向（源提示词和目标提示词）时，用源的注意力图覆盖目标的，保持空间结构。

**15.4** 关键代码：`eps = eps_null_null + w_img*(eps_img_null - eps_null_null) + w_txt*(eps_img_text - eps_img_null)`

**15.5** 主要差异：DALL-E 3使用T5-XXL（更强的语言理解）+ 自训练的图像分词器；SD1/2使用CLIP-L；SD3使用三编码器（T5-XXL + CLIP-L + CLIP-G）拼接。多编码器融合对复杂关系的理解明显优于单一CLIP。

---

## 延伸阅读

1. **Rombach et al. (2022)**. *High-Resolution Image Synthesis with Latent Diffusion Models* — SD/LDM原文（交叉注意力）
2. **Saharia et al. (2022)**. *Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding* — Imagen（T5编码器）
3. **Hertz et al. (2022)**. *Prompt-to-Prompt Image Editing with Cross Attention Control* — 注意力图编辑
4. **Brooks et al. (2022)**. *InstructPix2Pix: Learning to Follow Image Editing Instructions* — 指令编辑

---

[← 上一章：无分类器引导（CFG）](./14-classifier-free-guidance.md)

[下一章：U-Net架构详解 →](../part6-architecture/16-unet-architecture.md)

[返回目录](../README.md)
