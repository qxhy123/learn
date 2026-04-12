# 第二十一章：DiT：扩散Transformer

> **本章导读**：Diffusion Transformer（DiT，Peebles & Xie 2023）将扩散模型的骨干网络从传统的U-Net替换为纯Transformer架构，这一看似简单的架构替换产生了深远的影响。DiT不仅在ImageNet生成上取得了SOTA（FID=2.27），更重要的是它证明了扩散模型遵循Transformer的Scaling Laws——更大的模型持续带来更好的质量。这一发现直接催生了Sora、SD3/FLUX等后续突破。本章将从U-Net的局限出发，详细分析DiT的patch化策略、adaLN-Zero条件机制和缩放特性，并提供完整的PyTorch实现。

**前置知识**：Vision Transformer（ViT）基础、DDPM/DDIM扩散模型、Classifier-Free Guidance、Latent Diffusion

**预计学习时间**：4-5小时

---

## 学习目标

完成本章学习后，你将能够：

1. 解释从U-Net到Transformer架构转变的动机与优势
2. 理解图像patch化处理如何将2D图像转换为1D token序列
3. 掌握adaLN-Zero条件机制的设计细节及其相对于其他条件化方法的优越性
4. 分析DiT的Scaling Laws并理解不同规模变体的性能差异
5. 从零实现一个完整的DiT模型并在图像数据集上训练

---

## 21.1 从U-Net到Transformer

### 21.1.1 U-Net的归纳偏置与局限

U-Net自DDPM以来一直是扩散模型的标准骨干网络，它具有以下归纳偏置：

1. **局部性**：卷积操作天然关注局部区域，这对图像的低层特征（纹理、边缘）非常有效
2. **层次性**：编码器-解码器结构通过下采样/上采样捕获多尺度特征
3. **跳跃连接**：encoder特征直接传给decoder，帮助保留细节

然而，这些归纳偏置也带来了限制：

- **扩展性差**：U-Net的扩展方式（增加通道、增加层级）不如Transformer的简单堆叠那么系统化
- **缺乏Scaling Laws**：U-Net在不同规模下的行为不可预测，没有类似LLM的幂律关系
- **全局交互受限**：虽然SD的U-Net加入了Self-Attention，但只在低分辨率层级（16x16、8x8），高分辨率层级仍依赖局部卷积
- **架构复杂**：ResBlock、SpatialTransformer、上下采样、跳跃连接——组件太多，难以系统优化

### 21.1.2 Vision Transformer的启示

ViT（Dosovitskiy et al. 2021）证明了纯Transformer在图像分类上可以超越CNN，关键思想是将图像分割为固定大小的patch，每个patch视为一个token：

$$x \in \mathbb{R}^{H \times W \times C} \xrightarrow{\text{patchify}} z \in \mathbb{R}^{N \times D}, \quad N = \frac{HW}{p^2}$$

ViT的成功有两个关键因素：
1. **简单性**：几乎是标准Transformer的直接应用
2. **可扩展性**：遵循与语言模型相同的Scaling Laws

DiT的核心问题就是：**能否在扩散模型中用Transformer替换U-Net，同时继承Transformer的Scaling Laws？**

### 21.1.3 DiT的核心贡献

Peebles & Xie (2023)在DiT论文中证明了：

1. 纯Transformer在扩散模型中可以完全替代U-Net
2. DiT遵循Scaling Laws：FID随模型大小和训练计算量单调下降
3. 在ImageNet 256x256上，DiT-XL/2达到了SOTA FID=2.27
4. 在相同Gflops下，DiT优于U-Net（计算效率更高）

---

## 21.2 图像Patch化

### 21.2.1 从图像到Token序列

DiT在VAE的潜在空间中工作（与SD类似），因此输入已经是压缩后的表示。Patch化过程：

$$z \in \mathbb{R}^{h \times w \times c} \xrightarrow{\text{PatchEmbed}} \text{tokens} \in \mathbb{R}^{N \times D}$$

其中 $h = H/8, w = W/8$（VAE压缩），$c = 4$（VAE通道数），$N = hw/p^2$（token数），$D$ 是Transformer的隐藏维度。

以ImageNet 256x256为例：
- VAE编码后：$32 \times 32 \times 4$
- $p=2$（DiT默认）：$N = 32 \times 32 / 4 = 256$ 个token
- $p=4$：$N = 32 \times 32 / 16 = 64$ 个token
- $p=8$：$N = 32 \times 32 / 64 = 16$ 个token

### 21.2.2 Patch大小的影响

Patch大小 $p$ 是计算量和质量之间的关键权衡：

| Patch大小 $p$ | Token数 $N$ | Self-Attention复杂度 | FID（DiT-XL） |
|--------------|-------------|---------------------|---------------|
| 2 | 256 | $O(256^2) = 65536$ | **2.27** |
| 4 | 64 | $O(64^2) = 4096$ | 3.04 |
| 8 | 16 | $O(16^2) = 256$ | 9.62 |

$p=2$ 在质量上最优（更多token保留更多空间信息），但计算量最大。DiT论文的主要结果使用 $p=2$。

### 21.2.3 位置嵌入

DiT使用**固定正弦位置嵌入**（与原始Transformer一致），也可以替换为可学习的位置嵌入。对于2D图像，位置嵌入需要编码二维坐标：

$$\text{PE}(i, j) = [\sin(i/\omega_1), \cos(i/\omega_1), ..., \sin(j/\omega_1), \cos(j/\omega_1), ...]$$

其中 $(i, j)$ 是patch的行列坐标。

**重要性质**：固定正弦嵌入支持分辨率外推——在256x256上训练的模型可以（在一定程度上）在512x512上推理（通过插值位置嵌入）。

### 21.2.4 Unpatchify（输出重建）

DiT的输出层需要将token序列恢复为2D特征图。对于噪声预测：

$$\text{tokens} \in \mathbb{R}^{N \times D} \xrightarrow{\text{Linear}} \mathbb{R}^{N \times (p^2 \cdot c)} \xrightarrow{\text{reshape}} \mathbb{R}^{h \times w \times c}$$

即每个token通过线性层预测其对应patch区域的所有像素值。

---

## 21.3 DiT Block设计

### 21.3.1 条件化机制的探索

DiT需要将两种条件信息注入到Transformer中：
- **时间步 $t$**：当前去噪阶段（标量）
- **类别标签 $y$**（或文本等）：生成条件

论文探索了4种条件化方案：

**方案1：In-Context Conditioning**

将条件嵌入为额外token，拼接到序列前面：

$$\text{input} = [t_{token}, y_{token}, x_1, x_2, ..., x_N]$$

简单但效果最差——条件token只通过注意力间接影响图像token。

**方案2：Cross-Attention**

类似SD的U-Net，在Self-Attention之后加入Cross-Attention层：

$$Q = x, \quad K = V = [t_{emb}, y_{emb}]$$

有效但增加了约15%的参数量。

**方案3：adaLN（Adaptive Layer Normalization）**

用条件信息回归LayerNorm的缩放和偏移参数：

$$\gamma, \beta = \text{MLP}(\text{concat}(t_{emb}, y_{emb}))$$

$$\text{adaLN}(x) = \gamma \odot \text{LayerNorm}(x) + \beta$$

效果很好，但训练初期不稳定。

**方案4：adaLN-Zero（最终选择）**

在adaLN基础上增加**零初始化**策略：

$$\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2 = \text{MLP}(c), \quad c = t_{emb} + y_{emb}$$

其中 $\alpha$ 是额外的缩放参数，初始化MLP使得 $\alpha = 0$，保证训练开始时DiT Block是恒等映射。

### 21.3.2 adaLN-Zero的完整数学描述

给定输入 $x \in \mathbb{R}^{N \times D}$ 和条件嵌入 $c \in \mathbb{R}^{D}$：

**步骤1**：回归6个调制参数

$$[\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2] = \text{Linear}(\text{SiLU}(\text{Linear}(c)))$$

所有参数 $\in \mathbb{R}^{D}$，最后一个Linear层的权重和偏置初始化为零。

**步骤2**：调制Self-Attention

$$h_1 = \text{LayerNorm}(x) \cdot (1 + \gamma_1) + \beta_1$$

$$x = x + \alpha_1 \cdot \text{MultiHeadAttention}(h_1)$$

**步骤3**：调制FFN

$$h_2 = \text{LayerNorm}(x) \cdot (1 + \gamma_2) + \beta_2$$

$$x = x + \alpha_2 \cdot \text{FFN}(h_2)$$

**零初始化的意义**：在训练开始时 $\alpha_1 = \alpha_2 = 0$，因此DiT Block的输出等于输入（恒等映射）。这意味着：
1. 初始模型等价于一个什么都不做的变换
2. 训练从一个合理的起点开始（比随机初始化稳定得多）
3. 条件信息的影响从零开始逐渐增长

### 21.3.3 性能对比

论文的消融实验结果（DiT-XL/2, ImageNet 256x256, 400K训练步）：

| 条件化方式 | FID↓ |
|-----------|------|
| In-Context | 5.95 |
| Cross-Attention | 3.75 |
| adaLN | 3.15 |
| **adaLN-Zero** | **2.27** |

adaLN-Zero显著优于其他方案，且不增加额外参数（Cross-Attention需要额外的KV投影层）。

### 21.3.4 FFN设计

DiT使用标准的GELU FFN，扩展比为4：

$$\text{FFN}(x) = \text{Linear}_2(\text{GELU}(\text{Linear}_1(x)))$$

其中 $\text{Linear}_1: D \rightarrow 4D$，$\text{Linear}_2: 4D \rightarrow D$。

---

## 21.4 DiT的缩放

### 21.4.1 模型变体

DiT定义了4个标准变体，遵循ViT的命名惯例：

| 变体 | 层数 | 隐藏维度 | 注意力头 | 参数量 | Gflops |
|------|------|---------|---------|--------|--------|
| DiT-S | 12 | 384 | 6 | 33M | 6.1 |
| DiT-B | 12 | 768 | 12 | 130M | 23.0 |
| DiT-L | 24 | 1024 | 16 | 458M | 80.7 |
| DiT-XL | 28 | 1152 | 16 | 675M | 118.6 |

### 21.4.2 Scaling Laws

DiT论文的关键发现是扩散Transformer遵循幂律缩放关系：

$$\text{FID} \propto C^{-\alpha}$$

其中 $C$ 是训练计算量（Gflops $\times$ 训练步数）。这意味着：

1. **更大的模型始终更好**：在相同的训练计算预算下，大模型优于小模型
2. **更长的训练始终有益**：即使小模型训练更长时间，也难以追上大模型
3. **可预测性**：可以通过小规模实验预测大模型的性能

| 训练计算量 | DiT-S FID | DiT-B FID | DiT-L FID | DiT-XL FID |
|-----------|-----------|-----------|-----------|------------|
| 100K步 | 68.4 | 43.5 | 23.3 | 19.5 |
| 400K步 | 27.4 | 10.7 | 5.02 | 2.27 |
| 7M步 | 9.62 | 3.04 | 2.27 | **2.27** |

### 21.4.3 与U-Net的计算效率对比

在相同Gflops预算下的比较：

- DiT-XL/2（119 Gflops, FID=2.27）vs ADM-U（296 Gflops, FID=3.94）
- DiT在使用**不到一半计算量**的情况下获得了更好的FID

这说明Transformer架构在扩散模型中的计算效率高于U-Net——每单位计算量产生的信息利用率更高。

### 21.4.4 为什么Transformer更高效？

理论解释：
1. **全局注意力**：每个token在每一层都能attend所有其他token，信息流动效率高
2. **统一架构**：没有U-Net中的不同组件（ResBlock、Transformer、上下采样），优化更一致
3. **密集计算**：Transformer的矩阵乘法操作对GPU更友好，实际硬件利用率更高

---

## 21.5 潜在空间DiT（Latent DiT）

### 21.5.1 与VAE的结合

DiT在VAE的潜在空间中运行，这与Stable Diffusion的LDM思路完全一致：

```
图像 x ∈ [B, 3, 256, 256]
    │
    ▼ (VAE Encoder)
潜在表示 z ∈ [B, 4, 32, 32]
    │
    ▼ (Patch化, p=2)
Token序列 ∈ [B, 256, D]
    │
    ▼ (DiT, T步扩散)
去噪Token ∈ [B, 256, D]
    │
    ▼ (Unpatchify)
去噪潜在 z₀ ∈ [B, 4, 32, 32]
    │
    ▼ (VAE Decoder)
生成图像 x̂ ∈ [B, 3, 256, 256]
```

### 21.5.2 训练流程

DiT的训练流程与标准DDPM类似，只是骨干网络换成了Transformer：

1. 从数据集采样图像 $x$，通过VAE编码得到 $z_0$
2. 采样时间步 $t \sim \text{Uniform}(1, T)$ 和噪声 $\epsilon \sim \mathcal{N}(0, I)$
3. 构造噪声输入 $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$
4. DiT预测噪声（或 $v$-prediction）：$\hat{\epsilon} = \text{DiT}(z_t, t, y)$
5. 优化 $\mathcal{L} = \|\epsilon - \hat{\epsilon}\|^2$

### 21.5.3 ImageNet上的SOTA结果

DiT-XL/2在ImageNet 256x256上的性能：

| 模型 | FID↓ | IS↑ | Precision | Recall |
|------|------|-----|-----------|--------|
| ADM-U (Dhariwal 2021) | 3.94 | 215.8 | 0.83 | 0.53 |
| LDM-4 (Rombach 2022) | 3.60 | 247.7 | - | - |
| **DiT-XL/2** | **2.27** | **278.2** | **0.83** | **0.57** |

注意这些结果使用了CFG（guidance_scale=1.5-4.0）。

---

## 21.6 DiT的影响与后续

### 21.6.1 Sora（OpenAI, 2024）

Sora是DiT在视频生成领域的直接应用——Spacetime DiT：

- **架构**：将视频视为3D体积（帧 $\times$ 高 $\times$ 宽），用3D patch化
- **Spacetime patch**：例如 $p_t=2, p_h=2, p_w=2$，时间和空间同时patch化
- **可变分辨率/时长**：通过调整token数量适配不同尺寸
- **缩放**：据报道参数量可能在3B-30B之间

Sora的成功直接证明了DiT架构在视频领域的可扩展性。

### 21.6.2 PixArt-α（2023）

PixArt-α提出了高效的DiT训练策略：

1. **三阶段训练**：像素→概念→高质量（逐步提升数据质量和分辨率）
2. **分解Cross-Attention**：将文本条件和类别条件分开处理
3. **训练效率**：仅用SD 10.8%的训练成本达到同等质量

### 21.6.3 SD3与FLUX：MMDiT

Stable Diffusion 3（Esser et al. 2024）和FLUX（Black Forest Labs）引入了**Multimodal DiT（MMDiT）**：

**MMDiT的核心创新**：文本token和图像token在同一个Transformer中联合处理，而非通过Cross-Attention交互。

```
文本token: [t₁, t₂, ..., t_L]     (来自T5/CLIP编码器)
图像token: [x₁, x₂, ..., x_N]     (来自VAE + patch化)
                │
                ▼ (拼接)
联合序列: [t₁, ..., t_L, x₁, ..., x_N]
                │
                ▼ (Joint Self-Attention)
所有token之间自由交互
                │
                ▼ (分离)
更新后的文本token + 更新后的图像token
```

这种设计允许文本和图像在同一注意力计算中双向交互，比单向的Cross-Attention更强大。

**SD3的其他改进**：
- 采用Flow Matching（见第22章）替代DDPM
- 使用Rectified Flow进行更直的采样轨迹
- 三个文本编码器：CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL

### 21.6.4 架构趋势

从DiT出发，扩散模型架构的演进趋势清晰：

1. **U-Net → Transformer**：纯Transformer成为主流
2. **Cross-Attention → Joint Attention**：文本图像在同一空间交互
3. **DDPM → Flow Matching**：更直的采样轨迹
4. **图像 → 视频 → 3D**：token化方式的泛化

---

## 代码实战

### 完整DiT实现

```python
"""
Diffusion Transformer (DiT) 完整实现
包含所有核心组件和简化版训练循环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ============================================================
# 基础组件
# ============================================================

class PatchEmbed(nn.Module):
    """
    将2D图像（或潜在表示）分割为patch并嵌入到D维空间
    
    输入: [B, C, H, W]
    输出: [B, N, D], N = H*W / p^2
    """
    
    def __init__(
        self,
        img_size: int = 32,      # 潜在空间尺寸（256/8=32）
        patch_size: int = 2,
        in_channels: int = 4,    # VAE潜在通道数
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # N
        
        # 使用卷积实现patch化 + 线性嵌入
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )  # [B, C, H, W] -> [B, D, H/p, W/p]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B = x.shape[0]
        x = self.proj(x)                  # [B, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class TimestepEmbedder(nn.Module):
    """
    时间步嵌入：正弦编码 + MLP
    
    输入: [B] (整数时间步)
    输出: [B, D]
    """
    
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def sinusoidal_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: float = 10000.0,
    ) -> torch.Tensor:
        """
        生成正弦位置编码
        
        Args:
            t: [B] 时间步
            dim: 嵌入维度
        Returns:
            emb: [B, dim]
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )  # [dim/2]
        args = t[:, None].float() * freqs[None, :]  # [B, dim/2]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding  # [B, dim]
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        t_freq = self.sinusoidal_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)  # [B, D]
        return t_emb


class LabelEmbedder(nn.Module):
    """
    类别标签嵌入（用于类条件生成）
    支持Classifier-Free Guidance的dropout
    
    输入: [B] (整数类别标签)
    输出: [B, D]
    """
    
    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
    
    def token_drop(
        self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """在训练时随机将标签替换为无条件标签（用于CFG）"""
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels
    
    def forward(
        self,
        labels: torch.Tensor,
        train: bool = True,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)  # [B, D]
        return embeddings


# ============================================================
# adaLN-Zero 调制层
# ============================================================

class AdaLNZeroModulation(nn.Module):
    """
    adaLN-Zero: 自适应LayerNorm + 零初始化
    从条件嵌入c回归6个调制参数
    """
    
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 6 * hidden_size)
        # 零初始化：训练开始时所有调制参数为零
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, c: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """
        Args:
            c: [B, D] 条件嵌入（时间步 + 类别）
        Returns:
            gamma1, beta1, alpha1, gamma2, beta2, alpha2: 各 [B, 1, D]
        """
        params = self.linear(self.silu(c))       # [B, 6*D]
        params = params.unsqueeze(1)              # [B, 1, 6*D]
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)
        return gamma1, beta1, alpha1, gamma2, beta2, alpha2


# ============================================================
# DiT Block
# ============================================================

class DiTBlock(nn.Module):
    """
    DiT Block: Self-Attention + FFN，使用adaLN-Zero条件化
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        
        self.adaLN_modulation = AdaLNZeroModulation(hidden_size)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] 图像token序列
            c: [B, D] 条件嵌入
        Returns:
            x: [B, N, D] 更新后的token序列
        """
        # 获取调制参数
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = (
            self.adaLN_modulation(c)
        )  # 各 [B, 1, D]
        
        # adaLN-Zero调制 + Self-Attention
        h = self.norm1(x) * (1 + gamma1) + beta1  # [B, N, D]
        attn_out, _ = self.attn(h, h, h)           # [B, N, D]
        x = x + alpha1 * attn_out                  # [B, N, D]
        
        # adaLN-Zero调制 + FFN
        h = self.norm2(x) * (1 + gamma2) + beta2   # [B, N, D]
        mlp_out = self.mlp(h)                       # [B, N, D]
        x = x + alpha2 * mlp_out                    # [B, N, D]
        
        return x


# ============================================================
# 最终输出层
# ============================================================

class FinalLayer(nn.Module):
    """
    DiT的最终输出层：adaLN + 线性投影到patch空间
    """
    
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels
        )
        # adaLN参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            c: [B, D]
        Returns:
            x: [B, N, p*p*C]
        """
        params = self.adaLN_modulation(c).unsqueeze(1)  # [B, 1, 2*D]
        gamma, beta = params.chunk(2, dim=-1)
        x = self.norm(x) * (1 + gamma) + beta
        x = self.linear(x)  # [B, N, p*p*C]
        return x


# ============================================================
# 完整DiT模型
# ============================================================

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT)
    
    在VAE潜在空间中运行，使用adaLN-Zero条件化。
    """
    
    def __init__(
        self,
        img_size: int = 32,       # 潜在空间尺寸
        patch_size: int = 2,
        in_channels: int = 4,     # VAE通道数
        hidden_size: int = 768,
        depth: int = 12,          # Transformer层数
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        cfg_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels  # 输出通道数等于输入
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch嵌入
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_channels, hidden_size
        )
        
        # 位置嵌入（可学习）
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )
        
        # 条件嵌入
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, cfg_dropout)
        
        # Transformer块
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # 最终输出层
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """权重初始化"""
        # 位置嵌入初始化
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # 所有线性层和嵌入层的标准初始化
        def _init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        
        self.apply(_init_weights)
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        将patch序列恢复为2D特征图
        
        输入: [B, N, p*p*C]
        输出: [B, C, H, W]
        """
        p = self.patch_size
        c = self.out_channels
        h = w = self.img_size // p
        
        x = x.reshape(-1, h, w, p, p, c)     # [B, h, w, p, p, C]
        x = x.permute(0, 5, 1, 3, 2, 4)      # [B, C, h, p, w, p]
        x = x.reshape(-1, c, h * p, w * p)    # [B, C, H, W]
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        DiT前向传播
        
        Args:
            x: [B, C, H, W] 带噪声的潜在表示
            t: [B] 时间步
            y: [B] 类别标签
        Returns:
            noise_pred: [B, C, H, W] 预测的噪声
        """
        # Patch嵌入 + 位置嵌入
        x = self.patch_embed(x) + self.pos_embed  # [B, N, D]
        
        # 条件嵌入
        t_emb = self.t_embedder(t)                 # [B, D]
        y_emb = self.y_embedder(y, self.training)  # [B, D]
        c = t_emb + y_emb                          # [B, D]
        
        # Transformer块
        for block in self.blocks:
            x = block(x, c)                        # [B, N, D]
        
        # 最终输出
        x = self.final_layer(x, c)                 # [B, N, p*p*C]
        x = self.unpatchify(x)                     # [B, C, H, W]
        
        return x


# ============================================================
# 模型变体工厂函数
# ============================================================

def DiT_S_2(**kwargs) -> DiT:
    """DiT-S/2: 33M参数"""
    return DiT(depth=12, hidden_size=384, num_heads=6, patch_size=2, **kwargs)

def DiT_B_2(**kwargs) -> DiT:
    """DiT-B/2: 130M参数"""
    return DiT(depth=12, hidden_size=768, num_heads=12, patch_size=2, **kwargs)

def DiT_L_2(**kwargs) -> DiT:
    """DiT-L/2: 458M参数"""
    return DiT(depth=24, hidden_size=1024, num_heads=16, patch_size=2, **kwargs)

def DiT_XL_2(**kwargs) -> DiT:
    """DiT-XL/2: 675M参数"""
    return DiT(depth=28, hidden_size=1152, num_heads=16, patch_size=2, **kwargs)


# ============================================================
# 简化版训练循环（MNIST/CIFAR）
# ============================================================

def train_dit_simple(
    dataset_name: str = "cifar10",
    img_size: int = 32,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-4,
    num_timesteps: int = 1000,
    device: str = "cuda",
) -> DiT:
    """
    在MNIST/CIFAR上训练简化版DiT
    注意：这里直接在像素空间训练（无VAE），仅用于教学目的
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    # 数据集
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # 归一化到[-1, 1]
    ])
    
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root="./data", train=True,
                                   download=True, transform=transform)
        in_channels = 3
        num_classes = 10
    elif dataset_name == "mnist":
        dataset = datasets.MNIST(root="./data", train=True,
                                 download=True, transform=transform)
        in_channels = 1
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    
    # 模型（使用小型DiT）
    model = DiT(
        img_size=img_size,
        patch_size=2,
        in_channels=in_channels,
        hidden_size=384,
        depth=6,          # 减小深度用于快速训练
        num_heads=6,
        num_classes=num_classes,
        cfg_dropout=0.1,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    
    # 噪声调度
    betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"数据集: {dataset_name}, 样本数: {len(dataset)}")
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in dataloader:
            images = images.to(device)   # [B, C, H, W]
            labels = labels.to(device)   # [B]
            B = images.shape[0]
            
            # 采样时间步和噪声
            t = torch.randint(0, num_timesteps, (B,), device=device)
            noise = torch.randn_like(images)
            
            # 前向扩散
            sqrt_alpha_cumprod = alphas_cumprod[t].sqrt()[:, None, None, None]
            sqrt_one_minus = (1 - alphas_cumprod[t]).sqrt()[:, None, None, None]
            noisy_images = sqrt_alpha_cumprod * images + sqrt_one_minus * noise
            
            # DiT预测噪声
            noise_pred = model(noisy_images, t.float(), labels)
            
            # MSE损失
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


# ============================================================
# 参数量和速度对比
# ============================================================

def compare_dit_variants() -> None:
    """对比不同DiT变体的参数量和推理速度"""
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variants = {
        "DiT-S/2": DiT_S_2,
        "DiT-B/2": DiT_B_2,
        "DiT-L/2": DiT_L_2,
        "DiT-XL/2": DiT_XL_2,
    }
    
    print(f"{'变体':<12} {'参数量':>10} {'GFLOPs':>10} {'推理时间(ms)':>14}")
    print("-" * 50)
    
    for name, factory in variants.items():
        model = factory(num_classes=1000).to(device).eval()
        
        # 参数量
        num_params = sum(p.numel() for p in model.parameters())
        
        # 推理时间
        x = torch.randn(1, 4, 32, 32, device=device)
        t = torch.tensor([500.0], device=device)
        y = torch.tensor([0], device=device)
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(x, t, y)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(x, t, y)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10 * 1000  # ms
        
        print(f"{name:<12} {num_params/1e6:>8.1f}M {0:>10} {elapsed:>12.1f}")
        
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 创建DiT-S/2模型
    model = DiT_S_2(img_size=32, in_channels=4, num_classes=1000)
    print(f"DiT-S/2 参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    x = torch.randn(2, 4, 32, 32)    # [B, C, H, W] 潜在表示
    t = torch.tensor([100.0, 500.0])  # [B] 时间步
    y = torch.tensor([0, 5])          # [B] 类别标签
    
    out = model(x, t, y)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")     # 应该与输入相同: [2, 4, 32, 32]
    
    # 对比变体
    if torch.cuda.is_available():
        compare_dit_variants()
```

---

## 本章小结

| 概念 | 核心要点 |
|------|---------|
| 动机 | U-Net缺乏系统化的缩放规律，Transformer的Scaling Laws可迁移到扩散模型 |
| Patch化 | 图像→patch序列，$p=2$时256个token（32x32潜在空间） |
| 位置嵌入 | 固定正弦2D编码，支持分辨率外推 |
| adaLN-Zero | 零初始化的自适应LayerNorm，6个调制参数，训练初期为恒等映射 |
| 条件化对比 | adaLN-Zero > adaLN > Cross-Attention > In-Context |
| Scaling Laws | FID随模型规模和计算量单调下降，遵循幂律关系 |
| DiT-XL/2 | 675M参数，ImageNet 256x256 FID=2.27（SOTA） |
| 计算效率 | 相同Gflops下，DiT优于U-Net |
| Sora | Spacetime DiT用于视频生成 |
| SD3/FLUX | MMDiT：文本和图像token在同一Transformer中联合处理 |

---

## 练习题

### 基础题

**练习1**：如果输入潜在空间尺寸为 $64 \times 64 \times 4$（对应SD的512x512图像），patch大小 $p=2$，计算token序列长度。与SD的U-Net在最低分辨率（8x8）的Self-Attention序列长度相比如何？讨论计算量的差异。

**练习2**：解释adaLN-Zero中零初始化的具体实现和作用。如果不使用零初始化（改为随机初始化），训练会出现什么问题？

### 中级题

**练习3**：修改上面的DiT实现，添加以下功能：
- (a) 支持 $\epsilon$-prediction和 $v$-prediction两种参数化
- (b) 在 $v$-prediction模式下，输出包含速度的预测

提示：$v$-prediction定义为 $v = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0$。

**练习4**：DiT-XL/2有675M参数，其中哪些组件贡献了最多的参数量？请分别估算：
- (a) Patch嵌入层
- (b) 位置嵌入
- (c) 所有DiTBlock（含注意力和FFN）
- (d) 时间步和标签嵌入层

### 提高题

**练习5**：设计一个MMDiT Block（SD3使用的多模态DiT块），使得文本token和图像token可以在同一个Self-Attention中交互，但使用独立的FFN。具体要求：
- (a) 画出MMDiT Block的数据流图
- (b) 写出关键代码
- (c) 分析相比Cross-Attention方式的参数量和计算量差异
- (d) 讨论双向注意力相比单向Cross-Attention的理论优势

---

## 练习答案

### 练习1

**Token序列长度计算**：

$$N = \frac{64 \times 64}{2^2} = \frac{4096}{4} = 1024 \text{ 个token}$$

**与SD U-Net的对比**：
- SD U-Net在最低分辨率（8x8）的Self-Attention序列长度：$8 \times 8 = 64$
- DiT的序列长度：1024
- 比例：$1024 / 64 = 16\times$

**计算量差异**：
Self-Attention的复杂度为 $O(N^2 D)$：
- DiT：$O(1024^2 \times D) = O(1048576 D)$
- U-Net（8x8层）：$O(64^2 \times D) = O(4096 D)$

DiT的单层注意力计算量约为U-Net最低分辨率层的256倍。但需注意：
1. DiT没有卷积层的开销
2. U-Net在多个分辨率层级都有注意力
3. DiT的总层数可能更少（28 vs U-Net的多分辨率堆叠）
4. 矩阵乘法对GPU更友好，实际throughput可能更高

### 练习2

**零初始化的具体实现**：

在`AdaLNZeroModulation`中，最后一个线性层的权重和偏置都初始化为零：

```python
nn.init.zeros_(self.linear.weight)  # W = 0
nn.init.zeros_(self.linear.bias)    # b = 0
```

这意味着训练开始时：
- $\gamma_1 = \beta_1 = \alpha_1 = \gamma_2 = \beta_2 = \alpha_2 = 0$
- adaLN变为：$\text{LN}(x) \cdot (1 + 0) + 0 = \text{LN}(x)$
- 缩放为：$x + 0 \cdot \text{Attn/FFN}(\text{LN}(x)) = x$

因此每个DiTBlock是恒等映射，整个DiT初始等价于"什么都不做"。

**不使用零初始化的问题**：
1. **训练不稳定**：随机初始化的调制参数可能产生极端的缩放/偏移值，导致梯度爆炸
2. **初始输出噪声**：模型的初始预测是随机的，与真实噪声差距很大，初始损失很高
3. **收敛变慢**：模型需要额外的训练步来学会"首先不要乱改输入"
4. 这与ResNet中使用零初始化残差分支（"fixup"初始化）的思想一致

### 练习3

```python
class DiTWithVPred(DiT):
    """支持epsilon-prediction和v-prediction的DiT"""
    
    def __init__(self, prediction_type: str = "epsilon", **kwargs):
        super().__init__(**kwargs)
        self.prediction_type = prediction_type
        
        if prediction_type == "v":
            # v-prediction输出与输入同维度
            pass  # 输出层结构不变
    
    def forward(
        self,
        x: torch.Tensor,      # [B, C, H, W] 带噪输入
        t: torch.Tensor,       # [B] 时间步
        y: torch.Tensor,       # [B] 类别标签
    ) -> torch.Tensor:
        """输出根据prediction_type不同而含义不同"""
        # 网络输出
        output = super().forward(x, t, y)  # [B, C, H, W]
        return output  # epsilon或v的预测
    
    def compute_loss(
        self,
        x_0: torch.Tensor,         # [B, C, H, W] 干净数据
        noise: torch.Tensor,        # [B, C, H, W] 采样噪声
        t: torch.Tensor,            # [B] 时间步
        y: torch.Tensor,            # [B] 类别标签
        alphas_cumprod: torch.Tensor,  # [T] 累积alpha
    ) -> torch.Tensor:
        """计算训练损失"""
        sqrt_alpha = alphas_cumprod[t.long()].sqrt()[:, None, None, None]
        sqrt_one_minus = (1 - alphas_cumprod[t.long()]).sqrt()[:, None, None, None]
        
        # 构造带噪输入
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        
        # 网络预测
        pred = self.forward(x_t, t.float(), y)
        
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v":
            # v = sqrt(alpha_bar) * epsilon - sqrt(1 - alpha_bar) * x_0
            target = sqrt_alpha * noise - sqrt_one_minus * x_0
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        return F.mse_loss(pred, target)
```

### 练习4

DiT-XL/2的参数分布估算（$D=1152$, $L=28$, $N=256$, $\text{heads}=16$）：

**(a) Patch嵌入层**：
Conv2d(4, 1152, 2, 2): $4 \times 1152 \times 2 \times 2 + 1152 = 18432 + 1152 \approx$ **20K**

**(b) 位置嵌入**：
$256 \times 1152 =$ **295K**

**(c) 所有DiTBlock**（28个块）：
- Self-Attention: $4 \times D^2 = 4 \times 1152^2 \approx 5.3M$（Q,K,V,O投影）
- FFN: $2 \times D \times 4D = 2 \times 1152 \times 4608 \approx 10.6M$
- adaLN-Zero: $D \times 6D + 6D \approx 8.0M$（含SiLU后的Linear）
- LayerNorm: 无可学习参数（elementwise_affine=False）
- 每块合计: $\approx 23.9M$
- 28块总计: $28 \times 23.9M \approx$ **669M**

**(d) 时间步和标签嵌入**：
- TimestepEmbedder: $256 \times 1152 + 1152 \times 1152 \approx 1.6M$
- LabelEmbedder: $1001 \times 1152 \approx 1.2M$
- 合计: $\approx$ **2.8M**

**总结**：DiTBlock占了约99%的参数量，这与标准Transformer一致——嵌入层和输出层相对轻量。

### 练习5

**(a) MMDiT Block数据流图**：

```
文本token z_t ∈ [B, L, D]    图像token z_x ∈ [B, N, D]
     │                              │
     ▼                              ▼
  adaLN(z_t, c)                 adaLN(z_x, c)
     │                              │
     └──────── 拼接 ────────────────┘
                 │
                 ▼
      Joint Self-Attention
      (所有L+N个token互相attend)
                 │
                 ▼
           按位置分离
          ┌──────┴──────┐
          │              │
      z_t_attn       z_x_attn
          │              │
          ▼              ▼
     FFN_text(z_t)  FFN_image(z_x)    ← 独立的FFN
          │              │
          ▼              ▼
    更新z_t          更新z_x
```

**(b) 关键代码**：

```python
class MMDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm_t = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm_x = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ffn_text = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.ffn_image = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
    
    def forward(self, z_t, z_x, c):
        # Joint attention
        h_t, h_x = self.norm_t(z_t), self.norm_x(z_x)
        joint = torch.cat([h_t, h_x], dim=1)  # [B, L+N, D]
        joint_out, _ = self.attn(joint, joint, joint)
        t_out, x_out = joint_out.split([z_t.size(1), z_x.size(1)], dim=1)
        z_t = z_t + t_out
        z_x = z_x + x_out
        # Independent FFN
        z_t = z_t + self.ffn_text(self.norm_t(z_t))
        z_x = z_x + self.ffn_image(self.norm_x(z_x))
        return z_t, z_x
```

**(c) 参数量和计算量**：
- Cross-Attention额外增加 $K,V$ 投影（$2D^2$），MMDiT没有额外投影但多一个FFN（$8D^2$）
- MMDiT计算量更高（Joint Attention的序列长度为 $L+N$），但信息交换更充分
- 总体MMDiT参数量略多（因为双FFN），但attention参数量相同

**(d) 双向注意力的优势**：
- Cross-Attention是单向的：图像token可以attend文本token，但文本token不能attend图像token
- Joint Attention是双向的：文本token也会根据图像上下文被更新
- 这使得文本表示可以根据生成过程动态调整，实现更深层的多模态交互

---

## 延伸阅读

1. Peebles, W. & Xie, S. (2023). "Scalable Diffusion Models with Transformers." *ICCV 2023*. (DiT原论文)
2. Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*. (ViT)
3. Esser, P., et al. (2024). "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." *ICML 2024*. (SD3 / MMDiT)
4. Chen, J., et al. (2023). "PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis." *ICLR 2024*.
5. Brooks, T., et al. (2024). "Video generation models as world simulators." *OpenAI Technical Report*. (Sora)
6. Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models." *arXiv:2001.08361*. (Transformer Scaling Laws)

---

<div align="center">

[⬅️ 第二十章：DALL-E 2与层次式生成](20-dalle2-clip-conditioning.md) | [📖 目录](../README.md) | [第二十二章：Flow Matching与一致性模型 ➡️](22-flow-matching-consistency-models.md)

</div>
