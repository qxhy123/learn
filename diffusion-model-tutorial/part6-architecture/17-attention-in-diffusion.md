# 第十七章：注意力机制在扩散模型中的应用

> **本章导读**：注意力机制是扩散模型从"能生成图像"跃升到"能生成高质量、全局一致图像"的关键技术。自注意力让模型捕获长距离依赖，交叉注意力让文本指令精确控制图像生成。本章将深入分析注意力在扩散模型中的实现细节、高效变体以及可解释性应用，为理解Stable Diffusion的核心机制奠定基础。

**前置知识**：Transformer注意力基础（Query/Key/Value）、U-Net架构（第16章）、卷积神经网络

**预计学习时间**：4-5小时

## 学习目标

完成本章学习后，你将能够：

1. 解释卷积的局部感受野限制以及注意力如何解决扩散模型中的全局一致性问题
2. 实现空间自注意力（Spatial Self-Attention），掌握2D特征图与序列之间的转换
3. 实现交叉注意力（Cross-Attention）以将文本条件注入扩散模型，理解其在Stable Diffusion中的核心地位
4. 对比Flash Attention、Linear Attention、Windowed Attention等高效注意力机制的原理与适用场景
5. 利用注意力图进行可视化与图像编辑（如Prompt-to-Prompt方法）

---

## 17.1 为什么扩散模型需要注意力

### 17.1.1 卷积的局部感受野限制

标准卷积操作的感受野受限于卷积核大小。一个 $3 \times 3$ 卷积核只能看到每个位置周围的9个像素。虽然通过堆叠多层卷积可以逐渐扩大感受野，但这种扩展是线性的：$L$ 层 $3 \times 3$ 卷积的理论感受野为 $(2L+1) \times (2L+1)$。

在 $256 \times 256$ 分辨率下，为了让一个像素"看到"对角的像素，理论上需要约128层卷积。实际上，由于有效感受野远小于理论感受野（Luo et al. 2016），需要的层数更多。

**这对图像生成意味着什么？**

考虑生成一张人脸图像：左眼和右眼需要保持对称、相同颜色。如果网络只能通过局部卷积传递信息，左眼区域的信息需要经过许多层才能影响到右眼区域，这使得全局协调变得困难。

### 17.1.2 全局一致性要求

高质量图像生成对全局一致性有严格要求：

- **语义一致性**：同一物体的不同部分需要协调（人脸的左右对称、文字的整体布局）
- **风格一致性**：整幅图像的光照、色彩、纹理应保持统一
- **结构一致性**：建筑的线条需要对齐，自然场景的透视关系要正确

纯卷积网络在这些方面表现不佳，这也是早期GAN生成的图像经常出现不自然的原因之一。

### 17.1.3 注意力的计算复杂度挑战

标准自注意力的计算复杂度为：

$$\text{Complexity} = O(N^2 \cdot d)$$

其中 $N$ 是序列长度，$d$ 是特征维度。对于图像，$N = H \times W$：

| 分辨率 | $N$ | 注意力矩阵大小 | 内存占用（float32）|
|--------|-----|----------------|-------------------|
| $8 \times 8$ | 64 | $64 \times 64$ | 16 KB |
| $16 \times 16$ | 256 | $256 \times 256$ | 256 KB |
| $32 \times 32$ | 1,024 | $1024 \times 1024$ | 4 MB |
| $64 \times 64$ | 4,096 | $4096 \times 4096$ | 64 MB |
| $128 \times 128$ | 16,384 | $16384 \times 16384$ | 1 GB |
| $256 \times 256$ | 65,536 | $65536 \times 65536$ | 16 GB |

在 $256 \times 256$ 分辨率下，仅存储一个注意力矩阵就需要16GB内存，这对单张GPU来说完全不可行。这就是为什么注意力只在U-Net的低分辨率层使用的根本原因。

---

## 17.2 自注意力在U-Net中的实现

### 17.2.1 空间维度到序列的转换

自注意力是为序列数据设计的（NLP中的token序列），要将其应用于2D特征图，需要进行维度转换：

$$x \in \mathbb{R}^{B \times C \times H \times W} \xrightarrow{\text{reshape}} x' \in \mathbb{R}^{B \times (HW) \times C}$$

即把 $H \times W$ 个空间位置视为 $N = HW$ 个"token"，每个token的特征维度为 $C$。注意力之后再恢复原始形状：

$$\text{out}' \in \mathbb{R}^{B \times (HW) \times C} \xrightarrow{\text{reshape}} \text{out} \in \mathbb{R}^{B \times C \times H \times W}$$

### 17.2.2 标准多头自注意力

在U-Net中的实现与标准Transformer的多头注意力几乎相同：

$$Q = W_Q x', \quad K = W_K x', \quad V = W_V x'$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中 $d_k = C / h$，$h$ 是头数。多头注意力将 $C$ 个通道分为 $h$ 个头，每个头独立计算注意力：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$$

### 17.2.3 位置编码的讨论

一个有趣的设计选择：扩散模型的U-Net中**通常不添加显式的空间位置编码**。原因包括：

1. **卷积隐含位置信息**：ResBlock中的卷积操作已经编码了相对空间关系。每个位置的特征已经包含了来自其局部邻域的信息。

2. **跳跃连接保留位置**：编码器中各级别的跳跃连接将包含位置信息的特征传递给解码器。

3. **平移等变性的保持**：不加位置编码使得注意力保持平移等变性，这对图像生成是有利的。

4. **实验验证**：在实践中，添加显式位置编码对扩散模型的质量提升有限。

不过在DiT（Diffusion Transformer）架构中，由于没有卷积，确实需要添加位置编码。

### 17.2.4 注意力在不同分辨率层的配置

实践中的典型配置（以ADM为例）：

| 分辨率 | 使用注意力 | 原因 |
|--------|-----------|------|
| $256 \times 256$ | 否 | $N=65536$，计算量太大 |
| $128 \times 128$ | 否 | $N=16384$，仍然太大 |
| $64 \times 64$ | 否 | $N=4096$，边界情况 |
| $32 \times 32$ | 是 | $N=1024$，可以接受 |
| $16 \times 16$ | 是 | $N=256$，很轻量 |
| $8 \times 8$ | 是 | $N=64$，几乎无开销 |

在Stable Diffusion中，由于在 $64 \times 64$ 的潜在空间操作，各层分辨率更低，注意力覆盖更广。

---

## 17.3 交叉注意力（Cross-Attention）用于条件化

### 17.3.1 从自注意力到交叉注意力

交叉注意力是Stable Diffusion实现文本-图像对齐的核心机制。与自注意力不同，交叉注意力从两个不同来源生成Q和K/V：

$$Q = W_Q \cdot x_{\text{img}}, \quad K = W_K \cdot h_{\text{text}}, \quad V = W_V \cdot h_{\text{text}}$$

其中：
- $x_{\text{img}} \in \mathbb{R}^{B \times N_{\text{img}} \times C_{\text{img}}}$ 是图像特征（来自U-Net）
- $h_{\text{text}} \in \mathbb{R}^{B \times N_{\text{text}} \times C_{\text{text}}}$ 是文本特征（来自CLIP/T5等文本编码器）
- $N_{\text{img}} = H \times W$ 是图像token数，$N_{\text{text}}$ 是文本token数（通常77）

注意力矩阵的形状为 $N_{\text{img}} \times N_{\text{text}}$，它描述了**每个图像位置对每个文本token的关注程度**。

### 17.3.2 交叉注意力的计算流程

详细计算步骤：

$$A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times h \times N_{\text{img}} \times N_{\text{text}}}$$

$$\text{out} = A \cdot V \in \mathbb{R}^{B \times h \times N_{\text{img}} \times d_v}$$

注意力矩阵 $A$ 的每一行（对应一个图像位置）是一个关于文本token的概率分布。例如，生成"红色汽车"时，图像中汽车区域的像素在注意力中会高度关注"红色"和"汽车"这两个token。

### 17.3.3 Cross-Attention在Stable Diffusion中的角色

在Stable Diffusion的U-Net中，每个Transformer Block包含三个子层：

```
self_attention → cross_attention → feed_forward
    │                  │
    │                  └── 文本条件注入
    └── 图像内部依赖
```

交叉注意力在每个分辨率级别的每个Transformer Block中都存在，这意味着文本条件在多个尺度上反复注入，确保了精确的文本-图像对齐。

### 17.3.4 注意力图的可解释性

交叉注意力图提供了直观的可解释性：

对于提示词 "a cat sitting on a chair"，各token的注意力图（热力图）大致如下：

- **"cat"** → 高权重集中在猫的区域
- **"sitting"** → 分散在猫和椅子的接触区域
- **"chair"** → 高权重集中在椅子区域
- **"a"** / **"on"** → 较均匀分布

这种对应关系使得通过操控注意力图来实现精细的图像编辑成为可能。

---

## 17.4 高效注意力机制

### 17.4.1 Flash Attention

Flash Attention（Dao et al. 2022）是标准注意力的**精确加速实现**，不改变数学结果，而是优化了内存访问模式：

**核心思想**：标准注意力的瓶颈不是计算，而是内存带宽。它需要将整个 $N \times N$ 的注意力矩阵写入GPU高带宽内存（HBM），然后再读取进行softmax和矩阵乘法。

Flash Attention通过**分块计算（tiling）**避免了这一瓶颈：

1. 将 $Q, K, V$ 分成小块
2. 在GPU SRAM（快速片上内存）中计算每个小块的注意力
3. 使用在线softmax技巧逐块累积结果
4. 永远不需要在HBM中存储完整的 $N \times N$ 注意力矩阵

$$\text{内存}: O(N^2) \to O(N)$$
$$\text{速度}: 2\text{-}4\times \text{加速}$$
$$\text{结果}: \text{精确一致（非近似）}$$

在PyTorch 2.0+中，`F.scaled_dot_product_attention` 已自动调用Flash Attention（当可用时）。

### 17.4.2 Linear Attention

Linear Attention（Katharopoulos et al. 2020）将注意力的复杂度从 $O(N^2)$ 降低到 $O(N)$。核心思想是将softmax注意力重新表达为核函数：

$$\text{Attention}(Q, K, V) = \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q) \sum_j \phi(K_j)^T}$$

其中 $\phi$ 是特征映射函数。通过先计算 $\phi(K)^T V$（$d \times d$ 矩阵），避免了 $N \times N$ 的注意力矩阵：

$$O(N^2 d) \to O(N d^2)$$

当 $d \ll N$ 时（低分辨率层通常如此），这可以大幅减少计算。

**缺点**：近似质量不如标准注意力，在需要精确长距离依赖的任务上性能下降。

### 17.4.3 Windowed Attention（窗口注意力）

受Swin Transformer启发，窗口注意力将图像分割为不重叠的局部窗口，在每个窗口内部执行自注意力：

$$\text{Complexity}: O(N^2 d) \to O(w^2 N d)$$

其中 $w$ 是窗口大小。例如 $w=8$ 时，复杂度降低为原来的 $w^2/N = 64/N$ 倍。

为了实现跨窗口信息交换，通常交替使用：
- 常规窗口（Regular Window）
- 移位窗口（Shifted Window）

### 17.4.4 分辨率自适应注意力策略

现代扩散模型通常在不同层使用不同注意力策略：

| 分辨率 | 策略 | 原因 |
|--------|------|------|
| 高（$\geq 64$） | 无注意力或窗口注意力 | 序列太长，全局注意力不可行 |
| 中（$32$） | 标准/Flash注意力 | 序列长度适中，全局注意力可行 |
| 低（$\leq 16$） | 标准注意力 | 序列很短，注意力开销可忽略 |

---

## 17.5 注意力图分析与可解释性

### 17.5.1 Cross-Attention图可视化

提取和可视化交叉注意力图的步骤：

1. 在U-Net的前向传播中，hook每个交叉注意力层
2. 提取注意力权重矩阵 $A \in \mathbb{R}^{h \times N_{\text{img}} \times N_{\text{text}}}$
3. 对头维度取平均：$\bar{A} = \frac{1}{h} \sum_{i=1}^{h} A_i$
4. 对于每个文本token $j$，将 $\bar{A}[:, j]$ reshape为 $H \times W$ 的热力图
5. 上采样到原始图像分辨率进行叠加可视化

### 17.5.2 Prompt-to-Prompt：通过注意力编辑图像

Hertz et al. (2022) 提出的Prompt-to-Prompt方法利用注意力图实现精细的图像编辑。核心思想：

**注意力图控制布局**：交叉注意力图决定了"什么内容出现在什么位置"。

三种编辑操作：

1. **词语替换（Word Swap）**：将"cat"替换为"dog"，保持其他token的注意力图不变
   - 结果：猫变成狗，但位置、姿态、背景保持不变

2. **新增关注（Adding Attention）**：添加修饰词"red"并调节其注意力图
   - 结果：在保持原始结构的基础上添加红色属性

3. **注意力重加权（Attention Reweighting）**：增强或减弱某个词的注意力
   - 结果：增强"fluffy"会让毛发更蓬松

形式化表达：

$$\hat{A}_t = \text{Edit}(A_t^{\text{source}}, A_t^{\text{target}})$$

其中 $A_t^{\text{source}}$ 是原始提示的注意力图，$A_t^{\text{target}}$ 是编辑后提示的注意力图。

### 17.5.3 Attention Rollout

Attention Rollout（Abnar & Zuidema 2020）用于追踪信息在多层注意力中的流动：

$$\hat{A}^{(l)} = A^{(l)} \cdot \hat{A}^{(l-1)}$$

其中 $A^{(l)}$ 是第 $l$ 层的注意力矩阵，$\hat{A}^{(0)} = I$。通过连乘各层注意力矩阵，可以估计最终输出中每个位置受原始输入各位置的影响程度。

在扩散模型中，Rollout可以帮助理解：
- 哪些区域对最终生成结果影响最大
- 信息是如何在不同分辨率层之间传递的
- 跳跃连接在信息传递中的作用

---

## 17.6 Transformer Block变体

### 17.6.1 Pre-Norm vs Post-Norm

**Post-Norm（原始Transformer）**：

$$x_{l+1} = \text{LN}(x_l + \text{Attn}(x_l))$$

**Pre-Norm（现代扩散模型默认）**：

$$x_{l+1} = x_l + \text{Attn}(\text{LN}(x_l))$$

Pre-Norm的优势：
- 梯度流更稳定（残差分支的梯度不经过归一化）
- 训练更容易收敛
- 不需要learning rate warmup

扩散模型中几乎全部使用Pre-Norm。

### 17.6.2 SwiGLU前馈网络

标准FFN：

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

SwiGLU FFN（Shazeer 2020）：

$$\text{SwiGLU}(x) = W_2 \cdot (\text{Swish}(W_{\text{gate}} x) \odot (W_{\text{up}} x)) + b_2$$

其中 $\odot$ 是逐元素乘法。SwiGLU引入了一个门控机制，允许网络学习"让哪些特征通过"。在同等参数量下，SwiGLU通常优于标准GELU FFN。

### 17.6.3 DiT中的adaLN-Zero

Peebles & Xie (2023) 的DiT（Diffusion Transformer）提出了adaLN-Zero方案：

$$\gamma, \beta, \alpha = \text{MLP}(c)$$

$$h = x + \alpha \cdot \text{Attn}(\gamma \cdot \text{LN}(x) + \beta)$$

关键创新点：
1. 条件向量 $c$（时间步+类别）通过MLP生成三组参数：scale $\gamma$、shift $\beta$、gate $\alpha$
2. **零初始化**：$\alpha$ 初始化为0，使得网络初始状态等价于恒等映射
3. 这使得训练初期更稳定，因为网络从"什么都不做"开始逐渐学习

adaLN-Zero在DiT中取得了显著优于标准Cross-Attention条件化的结果。

### 17.6.4 归一化策略对训练稳定性的影响

| 归一化方式 | 特点 | 适用场景 |
|-----------|------|---------|
| Layer Norm | 对整个通道维度归一化 | Transformer内部 |
| Group Norm | 分组归一化，不依赖batch size | U-Net中的Conv层 |
| RMS Norm | 只做缩放不做偏移，计算更快 | LLaMA、一些新架构 |
| Adaptive LN | 条件控制归一化参数 | DiT |

训练稳定性建议：
- 使用Pre-Norm配置
- GroupNorm用于卷积层，LayerNorm用于注意力
- 输出层零初始化（注意力的 $W_O$ 和FFN的 $W_2$）
- 考虑使用QK-Norm（对Q和K做归一化），防止注意力logits爆炸

---

## 代码实战

```python
"""
扩散模型注意力机制完整实现
========================
包含：
1. 空间自注意力 (Spatial Self-Attention)
2. 交叉注意力 (Cross-Attention)
3. 完整Transformer Block
4. Flash Attention对比测试
5. 注意力图可视化
6. 不同分辨率的计算开销测试

参考：Vaswani et al. 2017, Rombach et al. 2022 (Stable Diffusion), Dao et al. 2022 (Flash Attention)
"""

import math
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 空间自注意力 (Spatial Self-Attention)
# ============================================================

class SpatialSelfAttention(nn.Module):
    """空间自注意力：将2D特征图转为序列，执行多头自注意力。
    
    Args:
        channels: 输入通道数（也是输出通道数）
        num_heads: 注意力头数
        head_dim: 每个头的维度（若为None，则 head_dim = channels // num_heads）
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim or (channels // num_heads)
        self.inner_dim = self.num_heads * self.head_dim
        
        # 归一化
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        
        # QKV投影
        self.to_qkv = nn.Linear(channels, self.inner_dim * 3, bias=False)
        
        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, channels),
            nn.Dropout(0.0),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图, shape: (B, C, H, W)
            
        Returns:
            输出特征图, shape: (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x
        
        # 归一化
        x = self.norm(x)  # shape: (B, C, H, W)
        
        # 2D -> 序列: (B, C, H, W) -> (B, H*W, C)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # shape: (B, N, C)
        
        # 计算 Q, K, V
        qkv = self.to_qkv(x)  # shape: (B, N, 3 * inner_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # 各 shape: (B, N, inner_dim)
        
        # 分头: (B, N, inner_dim) -> (B, num_heads, N, head_dim)
        q = q.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # shape: (B, num_heads, N, head_dim)
        
        # 缩放点积注意力（自动使用Flash Attention加速，如果可用）
        attn_out = F.scaled_dot_product_attention(
            q, k, v
        )  # shape: (B, num_heads, N, head_dim)
        
        # 合并头: (B, num_heads, N, head_dim) -> (B, N, inner_dim)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, H * W, self.inner_dim)
        
        # 输出投影
        out = self.to_out(attn_out)  # shape: (B, N, C)
        
        # 序列 -> 2D: (B, N, C) -> (B, C, H, W)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # shape: (B, C, H, W)
        
        return residual + out  # 残差连接


# ============================================================
# 2. 交叉注意力 (Cross-Attention)
# ============================================================

class CrossAttention(nn.Module):
    """交叉注意力：图像特征为Q，条件特征（如文本）为K和V。
    
    这是Stable Diffusion中将文本条件注入U-Net的核心机制。
    
    Args:
        query_dim: Q的输入维度（来自图像特征）
        context_dim: K/V的输入维度（来自文本编码器）
        num_heads: 注意力头数
        head_dim: 每个头的维度
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int = 768,
        num_heads: int = 8,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        
        # Q来自图像特征
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # K, V来自条件特征（文本）
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        # 输出投影
        self.to_out = nn.Linear(inner_dim, query_dim)
        
        # 存储注意力权重用于可视化
        self._attn_weights: Optional[torch.Tensor] = None
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        store_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: 图像特征序列, shape: (B, N_img, query_dim)
            context: 条件特征序列, shape: (B, N_text, context_dim)
            store_attention: 是否存储注意力权重用于可视化
            
        Returns:
            输出特征, shape: (B, N_img, query_dim)
        """
        B, N_img, _ = x.shape
        N_text = context.shape[1]
        
        # 投影
        q = self.to_q(x)        # shape: (B, N_img, inner_dim)
        k = self.to_k(context)   # shape: (B, N_text, inner_dim)
        v = self.to_v(context)   # shape: (B, N_text, inner_dim)
        
        # 分头
        q = q.reshape(B, N_img, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N_text, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N_text, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # shape: (B, heads, N_*, head_dim)
        
        if store_attention:
            # 手动计算注意力以存储权重
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            # shape: (B, heads, N_img, N_text)
            
            self._attn_weights = attn_weights.detach().cpu()
            attn_out = torch.matmul(attn_weights, v)
        else:
            # 使用高效实现
            attn_out = F.scaled_dot_product_attention(q, k, v)
        
        # shape: (B, heads, N_img, head_dim)
        
        # 合并头
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N_img, -1)
        # shape: (B, N_img, inner_dim)
        
        return self.to_out(attn_out)  # shape: (B, N_img, query_dim)
    
    def get_attention_maps(self) -> Optional[torch.Tensor]:
        """获取最近一次前向传播的注意力权重。
        
        Returns:
            注意力权重, shape: (B, heads, N_img, N_text) 或 None
        """
        return self._attn_weights


# ============================================================
# 3. 完整Transformer Block
# ============================================================

class FeedForward(nn.Module):
    """前馈网络（FFN）：使用GEGLU激活函数。
    
    GEGLU: out = Linear(GELU(W_gate @ x) * (W_up @ x))
    
    Args:
        dim: 输入/输出维度
        mult: 中间层维度倍数（默认4倍）
        dropout: dropout概率
    """
    
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim * mult
        # GEGLU需要双倍维度（一半用于gate）
        self.proj_in = nn.Linear(dim, inner_dim * 2)
        self.proj_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape: (B, N, dim)
        Returns:
            shape: (B, N, dim)
        """
        hidden, gate = self.proj_in(x).chunk(2, dim=-1)
        # shape: 各 (B, N, inner_dim)
        x = hidden * F.gelu(gate)  # GEGLU激活
        return self.proj_out(x)


class TransformerBlock(nn.Module):
    """完整的Transformer Block：Self-Attention + Cross-Attention + FFN。
    
    与标准Transformer的区别：
    - 使用Pre-Norm（归一化在注意力之前）
    - 自注意力使用图像特征自身
    - 交叉注意力使用外部条件（文本）
    
    Args:
        dim: 特征维度
        num_heads: 注意力头数
        head_dim: 每个头的维度
        context_dim: 条件特征维度（用于交叉注意力）
        ff_mult: FFN扩展倍数
        dropout: dropout概率
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        context_dim: int = 768,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        # Self-Attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CrossAttention(
            query_dim=dim,
            context_dim=dim,  # 自注意力: context = 自身
            num_heads=num_heads,
            head_dim=head_dim,
        )
        
        # Cross-Attention
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        
        # Feed-Forward Network
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mult=ff_mult, dropout=dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        store_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: 图像特征序列, shape: (B, N, dim)
            context: 条件特征序列, shape: (B, N_ctx, context_dim)
            store_attention: 是否存储交叉注意力权重
            
        Returns:
            输出特征, shape: (B, N, dim)
        """
        # Self-Attention (Pre-Norm)
        normed = self.norm1(x)
        x = x + self.self_attn(normed, normed)  # Q=K=V=自身
        
        # Cross-Attention (Pre-Norm)
        if context is not None:
            normed = self.norm2(x)
            x = x + self.cross_attn(normed, context, store_attention=store_attention)
        
        # Feed-Forward (Pre-Norm)
        x = x + self.ffn(self.norm3(x))
        
        return x


class SpatialTransformerBlock(nn.Module):
    """空间Transformer块：处理2D特征图的完整Transformer。
    
    将2D特征图转为序列 -> Transformer Block -> 转回2D。
    这是Stable Diffusion U-Net中使用的完整模块。
    
    Args:
        channels: 输入通道数
        num_heads: 注意力头数
        head_dim: 每个头的维度
        context_dim: 条件特征维度
        num_layers: Transformer层数
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: int = 64,
        context_dim: int = 768,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        inner_dim = num_heads * head_dim
        
        # 输入投影: conv -> 序列
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Linear(channels, inner_dim)
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=inner_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                context_dim=context_dim,
            )
            for _ in range(num_layers)
        ])
        
        # 输出投影: 序列 -> conv
        self.proj_out = nn.Linear(inner_dim, channels)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入特征图, shape: (B, C, H, W)
            context: 条件特征, shape: (B, N_ctx, context_dim)
            
        Returns:
            输出特征图, shape: (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x
        
        # 归一化 + 投影到序列
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # shape: (B, N, C)
        x = self.proj_in(x)  # shape: (B, N, inner_dim)
        
        # 通过Transformer层
        for block in self.transformer_blocks:
            x = block(x, context)
        
        # 投影回通道维度
        x = self.proj_out(x)  # shape: (B, N, C)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # shape: (B, C, H, W)
        
        return residual + x  # 残差连接


# ============================================================
# 4. Flash Attention 对比标准注意力
# ============================================================

def benchmark_attention(
    batch_size: int = 2,
    seq_len: int = 1024,
    num_heads: int = 8,
    head_dim: int = 64,
    num_runs: int = 50,
) -> None:
    """对比标准注意力与Flash Attention的速度和内存。
    
    Args:
        batch_size: 批大小
        seq_len: 序列长度（对应图像的 H*W）
        num_heads: 注意力头数
        head_dim: 每个头的维度
        num_runs: 重复次数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"配置: B={batch_size}, N={seq_len}, heads={num_heads}, d_k={head_dim}")
    print(f"相当于分辨率: {int(seq_len**0.5)}x{int(seq_len**0.5)}")
    print("-" * 60)
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    
    # 标准注意力（手动实现）
    def standard_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1])
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, h, N, N)
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)
    
    # 预热
    for _ in range(5):
        _ = standard_attention(q, k, v)
        _ = F.scaled_dot_product_attention(q, k, v)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # 测试标准注意力
    start = time.perf_counter()
    for _ in range(num_runs):
        out_standard = standard_attention(q, k, v)
    if device.type == "cuda":
        torch.cuda.synchronize()
    time_standard = (time.perf_counter() - start) / num_runs
    
    # 测试 scaled_dot_product_attention（自动Flash Attention）
    start = time.perf_counter()
    for _ in range(num_runs):
        out_sdpa = F.scaled_dot_product_attention(q, k, v)
    if device.type == "cuda":
        torch.cuda.synchronize()
    time_sdpa = (time.perf_counter() - start) / num_runs
    
    # 验证结果一致性
    max_diff = (out_standard - out_sdpa).abs().max().item()
    
    print(f"标准注意力:          {time_standard*1000:.2f} ms")
    print(f"SDPA (Flash Attn):   {time_sdpa*1000:.2f} ms")
    print(f"加速比:              {time_standard/time_sdpa:.2f}x")
    print(f"最大差异:            {max_diff:.2e}")
    
    # 注意力矩阵内存占用
    attn_matrix_size = batch_size * num_heads * seq_len * seq_len * 4  # float32
    print(f"注意力矩阵大小:      {attn_matrix_size / 1024 / 1024:.1f} MB")


# ============================================================
# 5. 注意力图可视化
# ============================================================

def visualize_cross_attention(
    attn_maps: torch.Tensor,
    tokens: list,
    image_size: int,
) -> None:
    """可视化交叉注意力图。
    
    Args:
        attn_maps: 注意力权重, shape: (B, heads, N_img, N_text)
        tokens: 文本token列表
        image_size: 原始图像尺寸（用于reshape）
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return
    
    # 取第一个样本，对头维度取平均
    attn = attn_maps[0].mean(dim=0)  # shape: (N_img, N_text)
    
    # 推断特征图尺寸
    spatial_size = int(attn.shape[0] ** 0.5)
    
    # 可视化每个token的注意力图
    num_tokens = min(len(tokens), attn.shape[1])
    fig, axes = plt.subplots(1, num_tokens, figsize=(3 * num_tokens, 3))
    
    if num_tokens == 1:
        axes = [axes]
    
    for i, (ax, token) in enumerate(zip(axes, tokens[:num_tokens])):
        # 取第 i 个token的注意力图
        attn_map = attn[:, i].reshape(spatial_size, spatial_size)
        
        # 上采样到原始图像尺寸
        attn_map_upscaled = F.interpolate(
            attn_map.unsqueeze(0).unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        
        ax.imshow(attn_map_upscaled.numpy(), cmap="hot")
        ax.set_title(f'"{token}"', fontsize=10)
        ax.axis("off")
    
    plt.suptitle("Cross-Attention Maps", fontsize=14)
    plt.tight_layout()
    plt.savefig("cross_attention_maps.png", dpi=150, bbox_inches="tight")
    print("注意力图已保存至 cross_attention_maps.png")


# ============================================================
# 6. 不同分辨率的计算开销对比
# ============================================================

def benchmark_resolutions() -> None:
    """测试不同分辨率下自注意力的计算开销。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    resolutions = [8, 16, 32, 64]
    channels = 256
    num_heads = 8
    batch_size = 2
    
    print(f"自注意力计算开销对比 (channels={channels}, heads={num_heads}, batch={batch_size})")
    print(f"{'分辨率':>8} | {'序列长度':>8} | {'时间(ms)':>10} | {'内存(MB)':>10} | {'注意力矩阵':>12}")
    print("-" * 70)
    
    for res in resolutions:
        seq_len = res * res
        attn_matrix_mb = batch_size * num_heads * seq_len * seq_len * 4 / 1024 / 1024
        
        attn_module = SpatialSelfAttention(
            channels=channels, num_heads=num_heads
        ).to(device)
        
        x = torch.randn(batch_size, channels, res, res, device=device)
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = attn_module(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # 计时
        start = time.perf_counter()
        num_runs = 20
        with torch.no_grad():
            for _ in range(num_runs):
                _ = attn_module(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_runs * 1000
        
        print(
            f"{res:>4}x{res:<3} | {seq_len:>8} | {elapsed:>8.2f} ms | "
            f"{attn_matrix_mb:>8.1f} MB | {seq_len}x{seq_len}"
        )
    
    del attn_module, x
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# 完整演示
# ============================================================

def demo_all() -> None:
    """运行所有演示。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("扩散模型注意力机制演示")
    print("=" * 70)
    
    # --- 1. 空间自注意力测试 ---
    print("\n[1] 空间自注意力测试")
    sa = SpatialSelfAttention(channels=256, num_heads=8).to(device)
    x = torch.randn(2, 256, 16, 16, device=device)
    with torch.no_grad():
        out = sa(x)
    print(f"  输入: {x.shape} -> 输出: {out.shape}")
    params = sum(p.numel() for p in sa.parameters())
    print(f"  参数量: {params:,}")
    
    # --- 2. 交叉注意力测试 ---
    print("\n[2] 交叉注意力测试")
    ca = CrossAttention(
        query_dim=256, context_dim=768, num_heads=8, head_dim=32
    ).to(device)
    img_feat = torch.randn(2, 256, 256, device=device)   # 16x16 图像特征
    text_feat = torch.randn(2, 77, 768, device=device)    # CLIP文本特征
    with torch.no_grad():
        out = ca(img_feat, text_feat, store_attention=True)
    print(f"  图像特征: {img_feat.shape}")
    print(f"  文本特征: {text_feat.shape}")
    print(f"  输出: {out.shape}")
    attn_maps = ca.get_attention_maps()
    print(f"  注意力图: {attn_maps.shape}")  # (B, heads, N_img, N_text)
    
    # --- 3. 完整Transformer Block测试 ---
    print("\n[3] 空间Transformer Block测试")
    stb = SpatialTransformerBlock(
        channels=256, num_heads=8, head_dim=32, context_dim=768
    ).to(device)
    x = torch.randn(2, 256, 16, 16, device=device)
    context = torch.randn(2, 77, 768, device=device)
    with torch.no_grad():
        out = stb(x, context)
    print(f"  输入: {x.shape}, 条件: {context.shape} -> 输出: {out.shape}")
    params = sum(p.numel() for p in stb.parameters())
    print(f"  参数量: {params:,}")
    
    # --- 4. 注意力图可视化演示 ---
    print("\n[4] 注意力图可视化演示")
    tokens = ["a", "red", "car", "on", "the", "road"]
    if attn_maps is not None:
        # 模拟不同token的注意力分布
        print(f"  注意力图形状: {attn_maps.shape}")
        print(f"  每个token对应 16x16={16*16} 个图像位置")
        print(f"  文本tokens: {tokens}")
        # visualize_cross_attention(attn_maps, tokens, 256)  # 取消注释以保存图片
    
    # --- 5. Flash Attention对比 ---
    print("\n[5] Flash Attention vs 标准注意力")
    benchmark_attention(batch_size=2, seq_len=1024, num_heads=8, head_dim=64)
    
    # --- 6. 分辨率对比 ---
    print("\n[6] 不同分辨率的注意力开销")
    benchmark_resolutions()


if __name__ == "__main__":
    demo_all()
```

---

## 本章小结

| 概念 | 核心要点 |
|------|---------|
| 自注意力的必要性 | 卷积感受野有限，注意力提供 $O(1)$ 长距离依赖 |
| 空间自注意力 | $(B,C,H,W) \to (B,HW,C)$，执行标准多头注意力，再恢复形状 |
| 交叉注意力 | $Q$ 来自图像，$K,V$ 来自文本条件；Stable Diffusion的核心机制 |
| Flash Attention | IO-aware精确加速，$O(N)$ 内存，2-4x速度提升 |
| Linear Attention | $O(Nd^2)$ 近似注意力，质量有损 |
| Windowed Attention | 局部窗口注意力 + 移位窗口实现跨窗口信息交换 |
| Prompt-to-Prompt | 通过操控注意力图实现图像编辑 |
| adaLN-Zero | DiT的条件化方案：零初始化的scale/shift/gate |
| Pre-Norm | 先归一化后注意力，训练更稳定 |
| GEGLU/SwiGLU | 门控FFN，同参数量下优于标准GELU FFN |

---

## 练习题

### 基础题

**练习1**：计算 $32 \times 32$ 分辨率特征图、通道数256、8头自注意力的总FLOPs。分别计算QKV投影、注意力矩阵计算、加权求和、输出投影的FLOPs。

**练习2**：解释为什么交叉注意力中 $Q$ 来自图像特征而 $K,V$ 来自文本特征（而不是反过来），从信息流的角度阐述。

### 中级题

**练习3**：实现一个支持可选 $Q$-$K$ 归一化（QK-Norm）的自注意力。QK-Norm在计算注意力分数前对 $Q$ 和 $K$ 分别做RMS归一化，然后乘以一个可学习的温度参数 $\tau$。写出数学公式和PyTorch代码。

**练习4**：实现窗口自注意力（Window Self-Attention）。要求：
- 输入 $(B, C, H, W)$，窗口大小 $w$
- 将特征图分割为 $\frac{H}{w} \times \frac{W}{w}$ 个窗口
- 在每个窗口内执行自注意力
- 确保 $H, W$ 能被 $w$ 整除

### 提高题

**练习5**：实现一个简化版的Prompt-to-Prompt图像编辑框架。要求：
- 定义注意力存储和注入的hook函数
- 实现"词语替换"编辑：保持布局不变，替换指定词语
- 实现"注意力重加权"：增强或减弱指定词语的影响
- 解释为什么需要在特定的去噪步骤中操控注意力（前几步控制结构，后几步控制细节）

---

## 练习答案

### 练习1

配置：$H=W=32$，$C=256$，$h=8$ 头，$d_k = 256/8 = 32$，$N = 32 \times 32 = 1024$。

**QKV投影**：3个线性变换 $(N, C) \times (C, C)$

$$\text{FLOPs}_{QKV} = 3 \times 2 \times N \times C \times C = 3 \times 2 \times 1024 \times 256 \times 256 \approx 402M$$

**注意力矩阵** $QK^T$：$(N, d_k) \times (d_k, N)$，每个头独立

$$\text{FLOPs}_{QK^T} = h \times 2 \times N \times d_k \times N = 8 \times 2 \times 1024 \times 32 \times 1024 \approx 537M$$

**加权求和** $AV$：$(N, N) \times (N, d_k)$，每个头独立

$$\text{FLOPs}_{AV} = h \times 2 \times N \times N \times d_k = 8 \times 2 \times 1024 \times 1024 \times 32 \approx 537M$$

**输出投影**：$(N, C) \times (C, C)$

$$\text{FLOPs}_{out} = 2 \times N \times C \times C = 2 \times 1024 \times 256 \times 256 \approx 134M$$

**总计**：$402M + 537M + 537M + 134M \approx 1.61 \text{ GFLOPs}$

### 练习2

交叉注意力中 $Q$ 来自图像、$K,V$ 来自文本的原因：

**信息流视角**：注意力可以理解为"查询-检索"机制。$Q$ 代表"我在找什么"，$K$ 代表"这里有什么"，$V$ 代表"这里的内容是什么"。

- **$Q$ 来自图像**：每个图像位置发出查询——"我这个位置应该生成什么？"
- **$K,V$ 来自文本**：文本提供信息库——"有红色、有汽车、在路上..."
- **注意力矩阵** $A \in \mathbb{R}^{N_{img} \times N_{text}}$：每个图像位置根据自身需要，从文本中检索最相关的语义信息

如果反过来（$Q$ 来自文本，$K,V$ 来自图像），意义变为"文本在图像中找什么"，这在图像描述（captioning）中有意义，但在图像生成中方向错误——我们需要让**图像去查询文本**来获得生成指导。

从维度上看：输出形状与 $Q$ 相同，即 $(N_{img}, d)$，正好是我们需要的图像特征。

### 练习3

**QK-Norm数学公式**：

$$\hat{Q} = \frac{Q}{\text{RMS}(Q)} \cdot \tau, \quad \hat{K} = \frac{K}{\text{RMS}(K)}$$

其中 $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$，$\tau$ 是可学习的标量温度参数。

```python
class QKNormAttention(nn.Module):
    """带有QK归一化的自注意力。"""
    
    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.to_qkv = nn.Linear(channels, channels * 3, bias=False)
        self.to_out = nn.Linear(channels, channels)
        
        # 可学习的温度参数，每个头一个
        self.temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1) * math.log(10.0)
        )
    
    @staticmethod
    def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """RMS归一化，沿最后一个维度。"""
        rms = x.square().mean(dim=-1, keepdim=True).add(eps).sqrt()
        return x / rms
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # QK-Norm
        q = self.rms_norm(q)
        k = self.rms_norm(k)
        
        # 使用温度缩放（代替标准的 1/sqrt(d_k)）
        tau = self.temperature.exp()  # shape: (heads, 1, 1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * tau
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        out = self.to_out(out)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return residual + out
```

### 练习4

```python
class WindowSelfAttention(nn.Module):
    """窗口自注意力。"""
    
    def __init__(
        self,
        channels: int,
        window_size: int = 8,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.to_qkv = nn.Linear(channels, channels * 3, bias=False)
        self.to_out = nn.Linear(channels, channels)
    
    def window_partition(
        self, x: torch.Tensor, w: int
    ) -> torch.Tensor:
        """将特征图分割为窗口。
        
        Args:
            x: (B, H, W, C)
            w: 窗口大小
        Returns:
            (B * num_windows, w*w, C)
        """
        B, H, W, C = x.shape
        x = x.reshape(B, H // w, w, W // w, w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, C)
        return x
    
    def window_unpartition(
        self, x: torch.Tensor, w: int, H: int, W: int, B: int
    ) -> torch.Tensor:
        """将窗口合并回特征图。
        
        Args:
            x: (B * num_windows, w*w, C)
        Returns:
            (B, H, W, C)
        """
        num_h, num_w = H // w, W // w
        x = x.reshape(B, num_h, num_w, w, w, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = self.window_size
        assert H % w == 0 and W % w == 0, f"H={H}, W={W}必须能被w={w}整除"
        
        residual = x
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        # 分割窗口
        x = self.window_partition(x, w)  # (B*nW, w^2, C)
        
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        BnW = q.shape[0]
        q = q.reshape(BnW, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(BnW, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(BnW, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(BnW, w * w, C)
        out = self.to_out(out)
        
        # 合并窗口
        out = self.window_unpartition(out, w, H, W, B)
        out = out.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return residual + out
```

### 练习5

```python
class AttentionStore:
    """注意力存储器：用于Prompt-to-Prompt编辑。"""
    
    def __init__(self) -> None:
        self.attention_maps: dict = {}
        self.step: int = 0
    
    def store(self, name: str, attn: torch.Tensor) -> None:
        if self.step not in self.attention_maps:
            self.attention_maps[self.step] = {}
        self.attention_maps[self.step][name] = attn.detach().cpu()
    
    def next_step(self) -> None:
        self.step += 1
    
    def reset(self) -> None:
        self.attention_maps.clear()
        self.step = 0


class PromptToPromptController:
    """Prompt-to-Prompt编辑控制器。"""
    
    def __init__(
        self,
        source_store: AttentionStore,
        edit_type: str = "replace",  # "replace" | "reweight"
        word_idx: int = 0,
        weight: float = 1.0,
        cross_replace_steps: float = 0.8,  # 前80%步骤替换交叉注意力
    ) -> None:
        self.source_store = source_store
        self.edit_type = edit_type
        self.word_idx = word_idx
        self.weight = weight
        self.cross_replace_steps = cross_replace_steps
        self.step = 0
        self.total_steps = 50
    
    def get_edited_attention(
        self, name: str, target_attn: torch.Tensor
    ) -> torch.Tensor:
        step_ratio = self.step / self.total_steps
        
        if self.edit_type == "replace" and step_ratio < self.cross_replace_steps:
            # 词语替换: 使用源注意力图的布局
            source_attn = self.source_store.attention_maps[self.step][name]
            return source_attn.to(target_attn.device)
        
        elif self.edit_type == "reweight":
            # 注意力重加权: 增强/减弱指定词语的注意力
            edited = target_attn.clone()
            edited[:, :, :, self.word_idx] *= self.weight
            # 重新归一化
            edited = edited / edited.sum(dim=-1, keepdim=True)
            return edited
        
        return target_attn
    
    def next_step(self) -> None:
        self.step += 1

# 使用说明：
# 1. 先用源提示生成图像，用 AttentionStore 记录所有注意力图
# 2. 用目标提示重新生成，通过 PromptToPromptController 替换注意力
# 3. 前80%步骤使用源注意力（保持结构），后20%使用目标注意力（引入新细节）
```

**为什么需要在特定步骤操控注意力？**

去噪过程有明确的分工：
- **前期步骤**（$t$ 大）：噪声强，网络确定全局结构和布局
- **后期步骤**（$t$ 小）：噪声弱，网络精修细节和纹理

因此，在前期步骤替换注意力图可以保持布局不变，后期步骤释放注意力让新内容自然融入。`cross_replace_steps=0.8` 意味着前80%步骤强制使用源布局，后20%步骤允许自由生成。

---

## 延伸阅读

1. **Vaswani, A., et al.** (2017). *Attention Is All You Need*. NeurIPS. Transformer和自注意力的开创性论文。

2. **Dao, T., et al.** (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS. Flash Attention的原始论文。

3. **Hertz, A., et al.** (2022). *Prompt-to-Prompt Image Editing with Cross Attention Control*. ICLR 2023. 通过操控注意力图实现图像编辑。

4. **Rombach, R., et al.** (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR. Stable Diffusion论文，交叉注意力条件化的核心参考。

5. **Peebles, W. & Xie, S.** (2023). *Scalable Diffusion Models with Transformers (DiT)*. ICCV. adaLN-Zero和纯Transformer扩散模型。

6. **Dao, T.** (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. Flash Attention的改进版。

7. **Katharopoulos, A., et al.** (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML. Linear Attention的理论基础。

---

[上一章：第十六章 U-Net架构详解](16-unet-architecture.md) | [目录](../README.md) | [下一章：第十八章 潜在扩散模型](18-latent-diffusion-models.md)
