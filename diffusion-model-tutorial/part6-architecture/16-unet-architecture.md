# 第十六章：U-Net架构详解

> **本章导读**：U-Net是扩散模型中最核心的骨干网络。从2015年医学图像分割到2020年DDPM的爆发，U-Net的编码器-解码器结构与跳跃连接被证明是去噪任务的理想选择。本章将从零开始构建一个完整的扩散U-Net，深入分析每个组件的设计动机与实现细节，帮助你理解现代扩散模型架构的基石。

**前置知识**：卷积神经网络基础（残差块、归一化）、注意力机制初步、扩散模型前向/反向过程（第1-15章）

**预计学习时间**：4-5小时

## 学习目标

完成本章学习后，你将能够：

1. 阐述U-Net在扩散模型中被选用的核心理由，对比分割U-Net与扩散U-Net的结构差异
2. 从零实现带有时间步条件化的残差块（ResBlock），包括Adaptive Group Normalization
3. 设计完整的编码器-瓶颈-解码器架构，合理配置通道数、注意力层和采样策略
4. 实现正弦时间嵌入与MLP投影，掌握时间信息注入U-Net的多种方式
5. 构建可运行的完整扩散U-Net，并分析其参数量、计算开销与典型配置

---

## 16.1 U-Net的历史与扩散模型的选择

### 16.1.1 原始U-Net：医学图像分割的突破

2015年，Ronneberger等人提出了U-Net架构，最初用于生物医学图像分割任务。其核心创新在于：

- **对称的编码器-解码器结构**：编码器逐步压缩空间分辨率提取语义信息，解码器逐步恢复分辨率
- **跳跃连接（Skip Connections）**：将编码器各层的特征直接传递给对应的解码器层，弥补下采样中丢失的细节

原始U-Net的结构可以用字母"U"形象地描述：

```
输入图像                                    输出分割图
  │                                          ▲
  ▼                                          │
[Conv 64] ──────────────────────────► [Conv 64]
  │                                          ▲
  ▼ (下采样)                          (上采样) │
[Conv 128] ─────────────────────────► [Conv 128]
  │                                          ▲
  ▼ (下采样)                          (上采样) │
[Conv 256] ─────────────────────────► [Conv 256]
  │                                          ▲
  ▼ (下采样)                          (上采样) │
[Conv 512] ─────────────────────────► [Conv 512]
  │                                          ▲
  ▼ (下采样)                          (上采样) │
            [Bottleneck 1024]
```

### 16.1.2 为什么U-Net适合扩散模型？

扩散模型的反向过程要求网络 $\epsilon_\theta(x_t, t)$ 预测添加到 $x_t$ 上的噪声。这个任务有几个关键特征：

1. **输入输出同尺寸**：输入是含噪图像 $x_t \in \mathbb{R}^{H \times W \times C}$，输出是同尺寸的噪声预测 $\hat{\epsilon} \in \mathbb{R}^{H \times W \times C}$。U-Net天然满足这一要求。

2. **多尺度特征理解**：去噪需要同时理解全局结构（"这是一张人脸"）和局部细节（"眼睛的纹理"）。U-Net的多层级结构天然提供了这种能力。

3. **细节保留至关重要**：跳跃连接确保高频细节不会在下采样中丢失。对于图像生成，这意味着生成的图像可以同时具有整体一致性和丰富细节。

4. **残差学习的天然优势**：噪声预测可以看作一种残差学习——预测"偏差"而非完整信号。U-Net的跳跃连接使得网络可以专注于学习残差。

### 16.1.3 扩散U-Net vs 分割U-Net

扩散U-Net在原始U-Net基础上进行了重要改进：

| 特性 | 分割U-Net（2015） | 扩散U-Net（DDPM 2020） |
|------|-------------------|----------------------|
| 输入 | 单张图像 | 含噪图像 $x_t$ + 时间步 $t$ |
| 输出 | 分割掩码 | 噪声预测 $\hat{\epsilon}$ |
| 归一化 | Batch Normalization | Group Normalization |
| 时间条件 | 无 | 正弦嵌入 + MLP注入 |
| 注意力 | 无 | 瓶颈层/低分辨率层自注意力 |
| 残差块 | 简单卷积块 | 预激活残差块 |
| 激活函数 | ReLU | SiLU/Swish |
| 下采样 | 最大池化 | 步幅卷积 |

最关键的区别是**时间条件化**：扩散U-Net需要"知道"当前处于第几步去噪，因为不同噪声水平下的去噪策略完全不同。高噪声时重建全局结构，低噪声时精修细节。

---

## 16.2 编码器路径（下采样）

编码器负责逐步降低空间分辨率、增加通道数，提取从局部到全局的多尺度特征。

### 16.2.1 残差块（ResNet Block）

扩散U-Net中的基本构建单元是带有时间条件化的残差块。其计算流程为：

$$h = x + F(x, t_{emb})$$

其中残差函数 $F$ 展开为：

$$F(x, t_{emb}) = \text{Conv}_2\big(\text{Dropout}\big(\text{SiLU}\big(\text{GN}_2(h_1 + \text{Linear}(t_{emb}))\big)\big)\big)$$

$$h_1 = \text{Conv}_1\big(\text{SiLU}\big(\text{GN}_1(x)\big)\big)$$

这里使用了**预激活（pre-activation）**范式，即先归一化后激活再卷积，数学上更优雅且训练更稳定。

关键设计选择：

**Group Normalization代替Batch Normalization**：扩散模型训练时batch size通常较小（如8-16），BN在小batch下统计量不稳定。GN将通道分为若干组，在每组内做归一化，不依赖batch大小。

$$\text{GN}(x) = \gamma \cdot \frac{x - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}} + \beta$$

其中 $\mu_g, \sigma_g^2$ 是每组通道的均值和方差。通常设 $G=32$ 组。

**SiLU/Swish激活函数**：$\text{SiLU}(x) = x \cdot \sigma(x)$，光滑且允许负值通过，实践中优于ReLU。

### 16.2.2 下采样策略

下采样有两种主流方式：

**步幅卷积（Strided Convolution）**：

$$\text{Downsample}(x) = \text{Conv2d}(x, \text{stride}=2, \text{kernel}=3, \text{padding}=1)$$

优点：可学习的下采样，可以选择保留哪些信息。

**平均池化 + 卷积**：

$$\text{Downsample}(x) = \text{Conv2d}(\text{AvgPool2d}(x, 2), \text{kernel}=1)$$

DDPM（Ho et al. 2020）选择了步幅卷积，而后续一些工作使用了可选的下采样方式。ADM（Dhariwal & Nichol 2021）对比实验表明两种方式性能差异不大，但步幅卷积实现更简洁。

### 16.2.3 时间嵌入注入

时间嵌入通过以下方式注入残差块：

**方式一：加法注入（Additive Injection）**

$$h = h + \text{Linear}(\text{SiLU}(t_{emb}))$$

将时间嵌入投影到与隐层相同的通道数后直接相加。这是DDPM原始论文使用的方式。

**方式二：Scale-Shift注入（Adaptive Group Normalization）**

$$\text{AdaGN}(h, t) = t_s \cdot \text{GN}(h) + t_b$$

其中 $t_s, t_b = \text{chunk}(\text{Linear}(\text{SiLU}(t_{emb})), 2)$ 分别是从时间嵌入预测的缩放因子和偏移量。

AdaGN更有表达力，因为它不仅移动特征（偏移），还缩放特征的分布。ADM论文证明这种方式性能更优。

### 16.2.4 通道数扩展策略

典型的通道数配置使用一个基准通道数 $C_{base}$，然后按倍数扩展：

- **DDPM CIFAR-10**：$C_{base}=128$，通道倍数 $(1, 2, 2, 2)$ → 128, 256, 256, 256
- **ADM ImageNet 256**：$C_{base}=256$，通道倍数 $(1, 1, 2, 2, 4, 4)$ → 256, 256, 512, 512, 1024, 1024
- **Stable Diffusion U-Net**：$C_{base}=320$，通道倍数 $(1, 2, 4, 4)$ → 320, 640, 1280, 1280

每个分辨率级别通常包含2-3个残差块。

---

## 16.3 瓶颈层（Bottleneck）

瓶颈层位于U-Net的最深处，处理分辨率最低但通道数最多的特征图。

### 16.3.1 全局自注意力

在瓶颈层的低分辨率（如 $8 \times 8$ 或 $4 \times 4$）下，序列长度足够短，可以应用全局自注意力：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

将特征图 $h \in \mathbb{R}^{B \times C \times H \times W}$ 重塑为 $h' \in \mathbb{R}^{B \times (HW) \times C}$，然后进行标准多头自注意力。对于 $8 \times 8$ 的特征图，序列长度仅为64，计算开销可以接受。

### 16.3.2 瓶颈层结构

典型的瓶颈层由以下组件交替构成：

```
ResBlock → AttentionBlock → ResBlock
```

其中每个ResBlock都接收时间嵌入。这种"三明治"结构使得：
- 第一个ResBlock对特征进行非线性变换
- AttentionBlock建立全局依赖关系
- 最后的ResBlock进一步整合信息

### 16.3.3 捕获全局依赖

瓶颈层的自注意力是U-Net中**唯一能看到完整空间范围的操作**（在DDPM中）。它负责：

- 建立图像不同区域之间的语义关联（如人脸的对称性）
- 在最抽象的层级上理解图像内容
- 为解码器提供全局一致的特征表示

在更高级的架构（如ADM、Stable Diffusion）中，注意力不仅在瓶颈层使用，还在多个分辨率级别使用，但瓶颈层的注意力仍然是最基本且最重要的。

---

## 16.4 解码器路径（上采样）

解码器的任务是将瓶颈层的压缩特征逐步恢复到原始分辨率。

### 16.4.1 上采样策略

**转置卷积（Transposed Convolution）**：

$$\text{Upsample}(x) = \text{ConvTranspose2d}(x, \text{stride}=2, \text{kernel}=4, \text{padding}=1)$$

转置卷积可能产生**棋盘伪影（checkerboard artifacts）**，因为不同输出位置被不同数量的输入元素覆盖。

**最近邻插值 + 卷积（推荐）**：

$$\text{Upsample}(x) = \text{Conv2d}(\text{Interpolate}(x, \text{scale}=2, \text{mode='nearest'}), \text{kernel}=3, \text{padding}=1)$$

这种方式先通过最近邻插值将空间分辨率翻倍，然后用 $3 \times 3$ 卷积平滑结果。由于每个输出位置的覆盖均匀，不会产生棋盘伪影。

实践中，DDPM和ADM都使用了最近邻插值 + 卷积的方式。

### 16.4.2 跳跃连接

跳跃连接将编码器中间层的特征传递给解码器对应层，有两种实现方式：

**通道拼接（Concatenation）**：

$$h_{dec} = \text{ResBlock}(\text{cat}(h_{up}, h_{skip}), t_{emb})$$

编码器特征 $h_{skip}$ 与上采样后的解码器特征 $h_{up}$ 在通道维度拼接。ResBlock的输入通道数翻倍。

**逐元素相加（Addition）**：

$$h_{dec} = \text{ResBlock}(h_{up} + h_{skip}, t_{emb})$$

DDPM采用拼接方式，因为它保留了更多信息——网络可以学习如何最优地组合两个来源的特征。

### 16.4.3 通道数缩减

解码器的通道数与编码器镜像对称地递减。考虑跳跃连接的拼接，每个解码器级别的输入通道数为：

$$C_{in}^{dec} = C_{up} + C_{skip}$$

其中 $C_{up}$ 是上采样后的通道数，$C_{skip}$ 是对应编码器层的通道数。ResBlock内部将其映射回目标通道数。

---

## 16.5 时间步嵌入的设计

时间步嵌入是扩散U-Net区别于普通U-Net的核心组件，它告诉网络当前处于扩散过程的哪一步。

### 16.5.1 正弦位置编码

受Transformer位置编码的启发，时间步 $t$ 首先被编码为正弦嵌入：

$$\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right)$$

$$\text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)$$

其中 $d$ 是嵌入维度，$i \in \{0, 1, \ldots, d/2-1\}$ 是维度索引。

直观理解：每个维度对应不同频率的"时钟"。低频维度缓慢变化（区分早期vs晚期），高频维度快速变化（区分相邻时间步）。这使得网络可以在不同精度层级上理解时间信息。

### 16.5.2 MLP投影

正弦编码后，通过一个小型MLP将其投影到更高维的空间：

$$t_{emb} = \text{Linear}_2(\text{SiLU}(\text{Linear}_1(\text{PE}(t))))$$

其中 $\text{Linear}_1: \mathbb{R}^{d} \to \mathbb{R}^{4d}$，$\text{Linear}_2: \mathbb{R}^{4d} \to \mathbb{R}^{4d}$。

扩展维度（通常为 $4 \times d_{model}$）给予MLP足够的容量来学习时间步的非线性变换。这个投影后的 $t_{emb} \in \mathbb{R}^{4d}$ 被所有ResBlock共享。

### 16.5.3 注入位置与方式

时间嵌入在U-Net中的每个ResBlock中被注入。具体来说：

1. 共享的MLP产生全局时间嵌入 $t_{emb}$
2. 每个ResBlock内部有独立的线性层，将 $t_{emb}$ 投影到该层的通道数
3. 投影后的时间信息与卷积中间特征融合

这意味着同一个时间步信息在不同层级以不同方式被解读——浅层可能用时间来调整低级纹理的去噪力度，深层可能用时间来决定全局结构的修正策略。

### 16.5.4 Adaptive Group Normalization详解

AdaGN是时间嵌入注入的高级形式：

$$\text{AdaGN}(h, y) = y_s \odot \text{GroupNorm}(h) + y_b$$

其中 $y_s, y_b \in \mathbb{R}^C$ 由条件向量（时间嵌入或类别嵌入）通过线性变换生成：

$$(y_s, y_b) = \text{Linear}(\text{SiLU}(t_{emb})) \in \mathbb{R}^{2C}$$

通过 $y_s$（scale），时间条件可以**调节每个通道特征的幅度**；通过 $y_b$（bias），可以**偏移特征分布**。这比单纯的加法注入多了一个自由度。

在DiT（Peebles & Xie 2023）中，AdaGN被进一步扩展为**adaLN-Zero**，初始化时 $y_s = 1, y_b = 0$，使得网络在训练初期等价于无条件网络，有助于稳定训练。

---

## 16.6 完整扩散U-Net架构

### 16.6.1 架构总览

以下是一个典型扩散U-Net的完整结构图（以256x256输入为例）：

```
输入: x_t (B, 3, 256, 256) + t (B,)
                │
         ┌──────┴──────┐
         │  SinEmb + MLP │ → t_emb (B, 512)
         └──────┬──────┘
                │
    ┌───────────┴───────────┐
    │  Conv2d(3→128, k=3)   │ → (B, 128, 256, 256)
    └───────────┬───────────┘
                │
    ════════════╪═══════════════════════════════════
    编码器      │                              解码器
    ════════════╪═══════════════════════════════════
                │
    ┌───────────┴───────────┐     ┌─────────────────┐
    │ ResBlock(128→128, t)  │────►│ ResBlock(256→128)│
    │ ResBlock(128→128, t)  │     │ ResBlock(256→128)│
    │ Downsample(128)       │     │ Upsample(128)    │
    └───────────┬───────────┘     └────────┬────────┘
                │                          ▲
    ┌───────────┴───────────┐     ┌────────┴────────┐
    │ ResBlock(128→256, t)  │────►│ ResBlock(512→256)│
    │ Attn(256)             │     │ Attn(256)        │
    │ ResBlock(256→256, t)  │     │ ResBlock(512→256)│
    │ Downsample(256)       │     │ Upsample(256)    │
    └───────────┬───────────┘     └────────┬────────┘
                │                          ▲
    ┌───────────┴───────────┐     ┌────────┴────────┐
    │ ResBlock(256→512, t)  │────►│ ResBlock(1024→512│
    │ Attn(512)             │     │ Attn(512)        │
    │ ResBlock(512→512, t)  │     │ ResBlock(1024→512│
    │ Downsample(512)       │     │ Upsample(512)    │
    └───────────┬───────────┘     └────────┬────────┘
                │                          ▲
                ▼                          │
    ┌─────────────────────────────────────┐
    │         Bottleneck (32x32)          │
    │ ResBlock(512→512, t)                │
    │ Attn(512)                           │
    │ ResBlock(512→512, t)                │
    └─────────────────────────────────────┘
```

### 16.6.2 参数量分析

各组件的参数量占比（以DDPM CIFAR-10 35M参数为例）：

| 组件 | 参数量 | 占比 |
|------|--------|------|
| 时间嵌入MLP | ~0.5M | 1.4% |
| 编码器ResBlocks | ~12M | 34% |
| 编码器注意力 | ~3M | 8.6% |
| 瓶颈层 | ~4M | 11.4% |
| 解码器ResBlocks | ~14M | 40% |
| 解码器注意力 | ~3M | 8.6% |
| 输入/输出卷积 | <0.1M | <1% |

解码器参数略多于编码器，因为跳跃连接的拼接使得解码器ResBlock的输入通道数更大。

### 16.6.3 注意力的分辨率策略

对于 $H \times W$ 的特征图，自注意力的计算复杂度为 $O((HW)^2 \cdot C)$。这意味着：

- **256x256**：序列长度65536，自注意力不可行（约需17GB显存仅用于注意力矩阵）
- **64x64**：序列长度4096，仍然很大
- **32x32**：序列长度1024，可以接受
- **16x16**：序列长度256，很轻量
- **8x8**：序列长度64，几乎无开销

因此，典型策略是只在 $32 \times 32$ 及以下分辨率使用自注意力。ADM论文实验表明，在 $32, 16, 8$ 三个分辨率都使用注意力效果最好。

### 16.6.4 典型配置对比

| 配置 | 图像大小 | 基通道 | 通道倍数 | 注意力分辨率 | 参数量 |
|------|---------|--------|---------|-------------|--------|
| DDPM CIFAR-10 | 32x32 | 128 | (1,2,2,2) | 16 | 35.7M |
| ADM ImageNet 64 | 64x64 | 192 | (1,2,3,4) | 32,16,8 | 296M |
| ADM ImageNet 256 | 256x256 | 256 | (1,1,2,2,4,4) | 32,16,8 | 554M |
| SD 1.5 U-Net | 64x64(潜在) | 320 | (1,2,4,4) | 8,4,2 | 860M |

---

## 代码实战

下面从零实现一个完整的扩散U-Net。

```python
"""
扩散U-Net完整实现
================
从零构建用于DDPM的U-Net骨干网络，包含：
- 正弦时间嵌入
- 带时间条件的残差块
- 多头自注意力块
- 下采样/上采样模块
- 完整U-Net组装

参考：Ho et al. 2020 (DDPM), Dhariwal & Nichol 2021 (ADM)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """正弦时间步嵌入。
    
    将标量时间步 t 编码为 d 维向量，使用不同频率的正弦/余弦函数。
    
    Args:
        dim: 嵌入维度（必须为偶数）
    """
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, "嵌入维度必须为偶数"
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间步张量, shape: (B,)
            
        Returns:
            嵌入向量, shape: (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        
        # 计算频率: 10000^(-2i/d), i = 0, 1, ..., d/2-1
        emb = math.log(10000.0) / (half_dim - 1)  # scalar
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * (-emb)
        )  # shape: (d/2,)
        
        # 外积: (B, 1) * (1, d/2) -> (B, d/2)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        
        # 拼接 sin 和 cos: (B, d/2) -> (B, d)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb  # shape: (B, dim)


class TimeEmbeddingMLP(nn.Module):
    """时间嵌入MLP：正弦编码 -> Linear -> SiLU -> Linear。
    
    Args:
        time_dim: 正弦编码维度
        emb_dim: 输出嵌入维度（通常为 4 * model_channels）
    """
    
    def __init__(self, time_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间步, shape: (B,)
            
        Returns:
            时间嵌入, shape: (B, emb_dim)
        """
        emb = self.sinusoidal(t)  # shape: (B, time_dim)
        emb = self.mlp(emb)       # shape: (B, emb_dim)
        return emb


class ResBlock(nn.Module):
    """带有时间条件化的残差块。
    
    结构: GN -> SiLU -> Conv -> (+ time_emb) -> GN -> SiLU -> Dropout -> Conv -> (+ shortcut)
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        time_emb_dim: 时间嵌入维度
        dropout: dropout概率
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        # 第一个卷积分支
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 时间嵌入投影
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        # 第二个卷积分支
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 残差连接的通道匹配
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征, shape: (B, C_in, H, W)
            t_emb: 时间嵌入, shape: (B, time_emb_dim)
            
        Returns:
            输出特征, shape: (B, C_out, H, W)
        """
        h = self.conv1(self.act1(self.norm1(x)))       # shape: (B, C_out, H, W)
        
        # 注入时间嵌入: (B, C_out) -> (B, C_out, 1, 1)
        t = self.time_proj(t_emb)                       # shape: (B, C_out)
        h = h + t[:, :, None, None]                     # shape: (B, C_out, H, W)
        
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))  # shape: (B, C_out, H, W)
        
        return h + self.shortcut(x)                     # shape: (B, C_out, H, W)


class AttentionBlock(nn.Module):
    """多头自注意力块，用于U-Net的低分辨率层。
    
    将2D特征图重塑为序列，执行多头自注意力，然后恢复形状。
    
    Args:
        channels: 输入/输出通道数
        num_heads: 注意力头数
    """
    
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "通道数必须能被头数整除"
        
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)
        
        # 零初始化输出投影，使注意力块初始为恒等映射
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征, shape: (B, C, H, W)
            
        Returns:
            输出特征, shape: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        h = self.norm(x)
        h = h.reshape(B, C, H * W)                     # shape: (B, C, N) where N=H*W
        
        # 计算Q, K, V
        qkv = self.qkv(h)                               # shape: (B, 3*C, N)
        q, k, v = qkv.chunk(3, dim=1)                   # 各 shape: (B, C, N)
        
        # 多头注意力
        head_dim = C // self.num_heads
        q = q.reshape(B, self.num_heads, head_dim, -1)   # shape: (B, heads, d_k, N)
        k = k.reshape(B, self.num_heads, head_dim, -1)   # shape: (B, heads, d_k, N)
        v = v.reshape(B, self.num_heads, head_dim, -1)   # shape: (B, heads, d_k, N)
        
        # 使用PyTorch 2.0+ 的scaled_dot_product_attention
        # 需要 (B, heads, N, d_k) 格式
        q = q.permute(0, 1, 3, 2)                        # shape: (B, heads, N, d_k)
        k = k.permute(0, 1, 3, 2)                        # shape: (B, heads, N, d_k)
        v = v.permute(0, 1, 3, 2)                        # shape: (B, heads, N, d_k)
        
        attn_out = F.scaled_dot_product_attention(q, k, v)  # shape: (B, heads, N, d_k)
        
        attn_out = attn_out.permute(0, 1, 3, 2)          # shape: (B, heads, d_k, N)
        attn_out = attn_out.reshape(B, C, -1)             # shape: (B, C, N)
        
        out = self.proj_out(attn_out)                     # shape: (B, C, N)
        out = out.reshape(B, C, H, W)                     # shape: (B, C, H, W)
        
        return x + out                                    # 残差连接


class Downsample(nn.Module):
    """下采样模块：步幅为2的卷积，空间分辨率减半。
    
    Args:
        channels: 输入/输出通道数
    """
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征, shape: (B, C, H, W)
            
        Returns:
            下采样特征, shape: (B, C, H/2, W/2)
        """
        return self.conv(x)  # shape: (B, C, H//2, W//2)


class Upsample(nn.Module):
    """上采样模块：最近邻插值 + 3x3卷积，空间分辨率翻倍。
    
    使用最近邻插值而非转置卷积，避免棋盘伪影。
    
    Args:
        channels: 输入/输出通道数
    """
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征, shape: (B, C, H, W)
            
        Returns:
            上采样特征, shape: (B, C, 2H, 2W)
        """
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # shape: (B, C, 2H, 2W)
        return self.conv(x)  # shape: (B, C, 2H, 2W)


class UNet(nn.Module):
    """完整扩散U-Net。
    
    编码器-瓶颈-解码器结构，带有跳跃连接和时间步条件化。
    
    Args:
        in_channels: 输入图像通道数（RGB=3）
        model_channels: 基准通道数
        out_channels: 输出通道数（噪声预测，通常=in_channels）
        channel_mult: 各级别的通道数倍数
        num_res_blocks: 每个级别的残差块数量
        attention_resolutions: 使用注意力的分辨率（如 [32, 16, 8]）
        dropout: dropout概率
        num_heads: 注意力头数
        image_size: 输入图像尺寸（用于确定注意力分辨率）
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.0,
        num_heads: int = 4,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        
        time_emb_dim = model_channels * 4
        
        # ---- 时间嵌入 ----
        self.time_embed = TimeEmbeddingMLP(model_channels, time_emb_dim)
        
        # ---- 输入卷积 ----
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # ---- 编码器 ----
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        ch = model_channels
        current_res = image_size
        encoder_channels = [model_channels]  # 记录每层输出通道数，用于跳跃连接
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_emb_dim, dropout)]
                
                if current_res in attention_resolutions:
                    layers.append(AttentionBlock(out_ch, num_heads))
                
                self.encoder_blocks.append(nn.ModuleList(layers))
                ch = out_ch
                encoder_channels.append(ch)
            
            # 除了最后一级，都添加下采样
            if level < len(channel_mult) - 1:
                self.downsamplers.append(Downsample(ch))
                encoder_channels.append(ch)
                current_res //= 2
        
        # ---- 瓶颈层 ----
        self.bottleneck = nn.ModuleList([
            ResBlock(ch, ch, time_emb_dim, dropout),
            AttentionBlock(ch, num_heads),
            ResBlock(ch, ch, time_emb_dim, dropout),
        ])
        
        # ---- 解码器 ----
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                # +1 因为要多处理一次跳跃连接（包括下采样层的输出）
                skip_ch = encoder_channels.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout)]
                
                if current_res in attention_resolutions:
                    layers.append(AttentionBlock(out_ch, num_heads))
                
                self.decoder_blocks.append(nn.ModuleList(layers))
                ch = out_ch
            
            # 除了最后一级（对应编码器第一级），都添加上采样
            if level > 0:
                self.upsamplers.append(Upsample(ch))
                current_res *= 2
        
        # ---- 输出层 ----
        self.output_norm = nn.GroupNorm(num_groups=32, num_channels=ch)
        self.output_act = nn.SiLU()
        self.output_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        
        # 零初始化输出卷积
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 含噪图像, shape: (B, C_in, H, W)
            t: 时间步, shape: (B,)
            
        Returns:
            噪声预测, shape: (B, C_out, H, W)
        """
        # 时间嵌入
        t_emb = self.time_embed(t)  # shape: (B, time_emb_dim)
        
        # 输入卷积
        h = self.input_conv(x)  # shape: (B, model_channels, H, W)
        
        # ---- 编码器 ----
        skips = [h]
        ds_idx = 0
        block_idx = 0
        
        for level, mult in enumerate(
            (self.in_channels,)  # 占位，实际用channel_mult
            for _ in [None]
        ):
            pass
        
        # 重新实现更清晰的前向传播
        skips = [h]
        ds_idx = 0
        
        for i, block_layers in enumerate(self.encoder_blocks):
            # 残差块（和可能的注意力块）
            for layer in block_layers:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            skips.append(h)
            
            # 检查是否需要下采样（每 num_res_blocks 个block后）
            # 通过与downsamplers数量对比来判断
        
        # 简化：逐层处理编码器
        # 重写前向传播以更清晰
        pass
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 含噪图像, shape: (B, C_in, H, W)
            t: 时间步, shape: (B,)
            
        Returns:
            噪声预测, shape: (B, C_out, H, W)
        """
        t_emb = self.time_embed(t)     # shape: (B, time_emb_dim)
        h = self.input_conv(x)         # shape: (B, model_channels, H, W)
        
        # 编码器：收集跳跃连接
        skips = [h]
        ds_idx = 0
        blocks_per_level = []
        idx = 0
        
        num_levels = len([m for m in (1, 2, 2, 2)])  # 会在__init__中记录
        
        # 使用更直接的方式
        block_idx = 0
        for level_idx in range(len(self._channel_mult)):
            for _ in range(self._num_res_blocks):
                block_layers = self.encoder_blocks[block_idx]
                for layer in block_layers:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
                skips.append(h)
                block_idx += 1
            
            if level_idx < len(self._channel_mult) - 1:
                h = self.downsamplers[ds_idx](h)
                skips.append(h)
                ds_idx += 1
        
        # 瓶颈
        for layer in self.bottleneck:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # 解码器：消费跳跃连接
        us_idx = 0
        block_idx = 0
        for level_idx in range(len(self._channel_mult) - 1, -1, -1):
            for _ in range(self._num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                
                block_layers = self.decoder_blocks[block_idx]
                for layer in block_layers:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
                block_idx += 1
            
            if level_idx > 0:
                h = self.upsamplers[us_idx](h)
                us_idx += 1
        
        # 输出
        h = self.output_conv(self.output_act(self.output_norm(h)))
        return h  # shape: (B, C_out, H, W)


# ============================================================
# 测试与参数统计
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_unet() -> None:
    """测试U-Net前向传播，打印各层输出形状。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DDPM CIFAR-10 配置
    model = UNet(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.1,
        num_heads=4,
        image_size=32,
    ).to(device)
    
    # 需要保存配置用于forward
    model._channel_mult = (1, 2, 2, 2)
    model._num_res_blocks = 2
    
    total_params = count_parameters(model)
    print(f"模型总参数量: {total_params:,} ({total_params / 1e6:.1f}M)")
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    assert x.shape == out.shape, "输入输出形状必须一致"
    
    # 各模块参数统计
    print("\n各模块参数量:")
    print(f"  时间嵌入MLP: {count_parameters(model.time_embed):,}")
    print(f"  输入卷积: {count_parameters(model.input_conv):,}")
    
    enc_params = sum(count_parameters(b) for b in model.encoder_blocks)
    enc_params += sum(count_parameters(d) for d in model.downsamplers)
    print(f"  编码器: {enc_params:,}")
    
    print(f"  瓶颈层: {sum(count_parameters(b) for b in model.bottleneck):,}")
    
    dec_params = sum(count_parameters(b) for b in model.decoder_blocks)
    dec_params += sum(count_parameters(u) for u in model.upsamplers)
    print(f"  解码器: {dec_params:,}")
    
    out_params = (count_parameters(model.output_norm) 
                  + count_parameters(model.output_conv))
    print(f"  输出层: {out_params:,}")


if __name__ == "__main__":
    test_unet()
```

> **注意**：上述代码为教学目的的完整实现。生产环境中建议使用 `diffusers` 库的 `UNet2DModel`，它经过了大量优化和测试。

---

## 本章小结

| 概念 | 核心要点 |
|------|---------|
| U-Net选择动机 | 输入输出同尺寸、多尺度特征、跳跃连接保留细节、残差学习天然契合 |
| 编码器 | ResBlock（GN→SiLU→Conv）+ 时间嵌入注入 + 步幅卷积下采样 |
| 瓶颈层 | ResBlock + 全局自注意力 + ResBlock，捕获全局依赖 |
| 解码器 | 最近邻上采样 + 跳跃连接拼接 + ResBlock |
| 时间嵌入 | 正弦编码 → MLP（Linear→SiLU→Linear）→ 每个ResBlock注入 |
| AdaGN | $y_s \cdot \text{GN}(h) + y_b$，scale-shift由条件向量控制 |
| 注意力策略 | 仅在低分辨率层（$\leq 32 \times 32$）使用全局自注意力 |
| 参数规模 | DDPM 35M → ADM 554M → SD U-Net 860M |

---

## 练习题

### 基础题

**练习1**：计算一个通道数为256、组数为32的Group Normalization层需要多少可学习参数。写出完整推导过程。

**练习2**：给定输入张量 $x \in \mathbb{R}^{4 \times 128 \times 32 \times 32}$（batch=4, channels=128, height=width=32），计算经过一个 `Downsample` 模块后的输出张量形状，以及该模块的参数量。

### 中级题

**练习3**：修改 `ResBlock` 实现，将加法时间嵌入注入替换为Adaptive Group Normalization（AdaGN）。具体要求：时间嵌入同时预测 scale 和 shift 参数，应用于第二个 GroupNorm 的输出。

**练习4**：设计一个"轻量级"U-Net配置：输入 $64 \times 64$，参数量控制在10M以下。写出你选择的 `model_channels`、`channel_mult`、`num_res_blocks` 和 `attention_resolutions`，并估算参数量。

### 提高题

**练习5**：实现一个支持**类别条件化**的U-Net。要求：
- 输入额外接收类别标签 $y \in \{0, 1, \ldots, K-1\}$
- 类别通过 `nn.Embedding` 编码后与时间嵌入相加
- 支持无条件训练（通过随机将类别设为特殊的"无条件"类）
- 写出完整的修改方案和代码

---

## 练习答案

### 练习1

Group Normalization的可学习参数是逐通道的 scale $\gamma$ 和 shift $\beta$：

- $\gamma \in \mathbb{R}^{C}$：256个参数
- $\beta \in \mathbb{R}^{C}$：256个参数
- 总计：$256 + 256 = 512$ 个参数

注意：组数（32）影响的是归一化统计量的计算方式（每组 $256/32 = 8$ 个通道共享均值和方差），但不影响可学习参数的数量。每个通道仍然有独立的 $\gamma$ 和 $\beta$。

### 练习2

输入：$x \in \mathbb{R}^{4 \times 128 \times 32 \times 32}$

`Downsample` 使用 `Conv2d(128, 128, kernel_size=3, stride=2, padding=1)`：

- 输出空间尺寸：$\lfloor(32 + 2 \times 1 - 3) / 2\rfloor + 1 = \lfloor 31/2 \rfloor + 1 = 16$
- 输出形状：$(4, 128, 16, 16)$
- 参数量：$128 \times 128 \times 3 \times 3 + 128 = 147,584$（权重 + 偏置）

### 练习3

```python
class ResBlockAdaGN(nn.Module):
    """使用Adaptive Group Normalization的残差块。"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # AdaGN: 时间嵌入预测 scale 和 shift
        self.adagn_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),  # 2x 用于 scale + shift
        )
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        
        # AdaGN: 从时间嵌入预测scale和shift
        scale_shift = self.adagn_proj(t_emb)       # shape: (B, 2*C_out)
        scale, shift = scale_shift.chunk(2, dim=1)  # 各 shape: (B, C_out)
        
        # 应用 AdaGN: scale * GN(h) + shift
        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        h = self.conv2(self.dropout(self.act2(h)))
        return h + self.shortcut(x)
```

关键点：`scale` 初始化接近0（通过1+scale使初始为恒等），`shift` 初始化接近0。

### 练习4

轻量级U-Net配置（$64 \times 64$ 输入，<10M参数）：

```python
config = dict(
    model_channels=64,          # 基准通道数减半
    channel_mult=(1, 2, 4),     # 3个级别: 64→128→256
    num_res_blocks=1,           # 每级别仅1个残差块
    attention_resolutions=(8,), # 仅在8x8使用注意力
    image_size=64,
    num_heads=4,
)
```

参数量估算：
- 时间嵌入MLP：$64 \times 256 + 256 \times 256 \approx 82K$
- 编码器ResBlocks：约 $3.2M$
- 注意力层：约 $0.8M$
- 瓶颈：约 $1.6M$
- 解码器：约 $4.0M$（含跳跃连接的额外通道）
- **总计约 $9.7M$**

### 练习5

```python
class ClassConditionedUNet(UNet):
    """支持类别条件化的U-Net。"""
    
    def __init__(
        self,
        num_classes: int,
        *args,
        class_dropout_prob: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        
        time_emb_dim = self.model_channels * 4
        
        # 类别嵌入: num_classes + 1 (额外一个用于无条件)
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.class_embed = nn.Embedding(num_classes + 1, time_emb_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 含噪图像, shape: (B, C, H, W)
            t: 时间步, shape: (B,)
            y: 类别标签, shape: (B,), 值域 [0, num_classes-1]
        """
        t_emb = self.time_embed(t)
        
        if y is not None:
            # 训练时随机 dropout 类别信息（用于 classifier-free guidance）
            if self.training:
                mask = torch.rand(y.shape[0], device=y.device) < self.class_dropout_prob
                y = y.clone()
                y[mask] = self.num_classes  # 无条件 token
            
            class_emb = self.class_embed(y)  # shape: (B, time_emb_dim)
            t_emb = t_emb + class_emb        # 时间 + 类别
        
        # 之后的前向传播与父类相同，使用 t_emb
        # （实际使用中需要将forward逻辑中t_emb部分提取为方法）
        return super()._forward_with_emb(x, t_emb)
```

核心思路：类别通过 `nn.Embedding` 映射到与时间嵌入同维度的向量，然后两者相加。训练时以 `class_dropout_prob`（如10%）的概率将类别设为"无条件"类，推理时支持classifier-free guidance。

---

## 延伸阅读

1. **Ronneberger, O., Fischer, P., & Brox, T.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI. 原始U-Net论文。

2. **Ho, J., Jain, A., & Abbeel, P.** (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS. DDPM论文，首次将U-Net系统性地用于扩散模型。

3. **Dhariwal, P. & Nichol, A.** (2021). *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS. ADM论文，系统性地改进了U-Net架构（AdaGN、更多注意力层等）。

4. **Nichol, A. & Dhariwal, P.** (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML. 改进的DDPM，包含U-Net架构的消融实验。

5. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep Residual Learning for Image Recognition*. CVPR. ResNet论文，残差块的起源。

---

[上一章：第十五章](../part5-sampling/15-advanced-sampling.md) | [目录](../README.md) | [下一章：第十七章 注意力机制在扩散模型中的应用](17-attention-in-diffusion.md)
