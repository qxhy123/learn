# 第二十章：DALL-E 2与层次式生成

> **本章导读**：DALL-E 2（Ramesh et al. 2022）是OpenAI推出的层次式文本到图像生成系统，其核心创新在于将生成过程分为两个阶段：先从文本生成CLIP图像嵌入（"先验"），再从CLIP图像嵌入生成像素。这种层次化设计不仅提高了生成质量，还解锁了图像变体、图像插值等独特能力。本章将深入分析DALL-E 2的unCLIP框架、扩散先验模型的设计动机，以及层次化生成范式对后续工作的影响。

**前置知识**：CLIP模型基础、扩散模型（DDPM/DDIM）、Classifier-Free Guidance、超分辨率扩散

**预计学习时间**：3-4小时

---

## 学习目标

完成本章学习后，你将能够：

1. 理解DALL-E 2的两阶段生成框架（先验 + 解码器）及其设计动机
2. 解释扩散先验模型为什么比直接从文本嵌入生成更有效
3. 掌握unCLIP解码器的架构细节与CLIP嵌入注入方式
4. 利用CLIP嵌入实现图像变体生成与语义插值
5. 对比分析DALL-E 2、Stable Diffusion和Imagen等不同架构的优劣

---

## 20.1 DALL-E 2的创新

### 20.1.1 从DALL-E 1到DALL-E 2

DALL-E 1（Ramesh et al. 2021）使用的是完全不同的架构：

| 特性 | DALL-E 1 | DALL-E 2 |
|------|----------|----------|
| 架构 | 自回归Transformer + dVAE | 扩散模型 + CLIP先验 |
| 图像表示 | 离散tokens（VQ-VAE码本） | 连续像素/潜在 |
| 文本编码 | BPE tokens | CLIP嵌入 |
| 参数量 | 12B | ~3.5B |
| 分辨率 | 256x256 | 1024x1024 |
| 多样性控制 | Top-k采样 | CFG + 扩散采样 |

DALL-E 2的核心突破在于引入了CLIP作为文本-图像的桥梁，而非直接从文本token生成图像token。

### 20.1.2 unCLIP框架概述

DALL-E 2的论文标题实际上是"Hierarchical Text-Conditional Image Generation with CLIP Latents"，作者将其方法称为**unCLIP**——因为它在概念上是"反转"CLIP的过程：

- **CLIP**：图像 → CLIP图像嵌入
- **unCLIP**：CLIP图像嵌入 → 图像

完整的生成流程：

$$\text{文本} \xrightarrow{\text{CLIP文本编码}} z_t \xrightarrow{\text{Prior}} z_i \xrightarrow{\text{Decoder}} \text{图像}$$

其中 $z_t$ 是CLIP文本嵌入，$z_i$ 是CLIP图像嵌入，Prior负责从 $z_t$ 预测 $z_i$。

### 20.1.3 为什么需要两�段？

一个自然的问题：为什么不直接从文本生成图像？答案涉及到CLIP嵌入空间的特殊性质：

1. **多对一映射**：同一个文本对应多种合理的图像。Prior学习的是这个"一对多"的条件分布 $p(z_i | z_t)$
2. **语义丰富性**：CLIP图像嵌入包含了更丰富的视觉语义（光照、构图、风格），而文本嵌入可能只有高层概念
3. **解耦生成**：先验负责"理解what"（语义规划），解码器负责"实现how"（细节渲染）

---

## 20.2 CLIP图像嵌入的先验模型

### 20.2.1 先验模型的任务

先验模型 $P(z_i | z_t, \text{text})$ 的目标是：给定CLIP文本嵌入，预测对应的CLIP图像嵌入。

注意这不是一个确定性映射——同一段文本可以对应多种合理的图像嵌入。因此先验模型需要学习一个分布。

### 20.2.2 自回归先验

DALL-E 2论文中探索了两种先验架构，第一种是自回归先验：

1. 将CLIP图像嵌入 $z_i \in \mathbb{R}^{768}$ 量化为离散token序列
2. 使用因果Transformer（类似GPT）自回归地从 $z_t$ 生成 $z_i$ 的token序列
3. 条件输入包括：CLIP文本嵌入 $z_t$、原始文本的BPE token、timestep

### 20.2.3 扩散先验（最终选择）

第二种是扩散先验，这是DALL-E 2最终采用的方案，因为它在质量和多样性上都更好。

扩散先验在CLIP图像嵌入空间（而非像素空间）中进行扩散：

**前向过程**：

$$z_i^{(t)} = \sqrt{\bar{\alpha}_t} \cdot z_i + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $z_i \in \mathbb{R}^{768}$（CLIP ViT-L/14的图像嵌入维度）。

**训练目标**：

$$\mathcal{L}_{\text{prior}} = \mathbb{E}_{t, z_i, \epsilon}\left[\|z_i - f_\theta(z_i^{(t)}, t, z_t)\|^2\right]$$

这里使用的是 $x_0$-prediction（预测原始 $z_i$）而非 $\epsilon$-prediction，因为在低维空间（768维）中直接预测目标更稳定。

**先验模型的Transformer架构**：

```
输入序列:
[z_t (CLIP文本嵌入), text_tokens (BPE编码), t (时间步嵌入), z_i^(t) (带噪图像嵌入)]
    │
    ▼
因果Transformer (24层, 隐藏维度2048)
    │
    ▼
输出: z_i_pred (预测的CLIP图像嵌入) ∈ R^768
```

### 20.2.4 为什么需要先验？直接方案的问题

如果跳过先验，直接将 $z_t$（CLIP文本嵌入）送入解码器，实验发现生成质量明显下降。原因：

1. **模态差异（Modality Gap）**：CLIP的文本嵌入和图像嵌入虽然在同一空间中，但它们的分布有系统性差异。文本嵌入的分布与图像嵌入的分布之间存在一个"gap"
2. **信息缺失**：文本 "a dog" 没有指定狗的品种、颜色、姿态、背景等。先验模型的工作是在条件分布中"填补"这些缺失的视觉信息
3. **多模态分布**：一段文本可能对应多种不同风格/构图的图像。先验作为生成模型可以采样不同的 $z_i$，产生多样化的结果

$$p(z_i | z_t) \neq \delta(z_i - g(z_t))$$

先验模型本质上将"文本空间中的一个点"映射到"图像空间中的一个分布"。

---

## 20.3 unCLIP解码器

### 20.3.1 解码器架构

unCLIP的解码器基于GLIDE（Nichol et al. 2021）架构——一个在像素空间运行的扩散模型。

```
CLIP图像嵌入 z_i ∈ R^768
    │
    ├──→ 投影到时间步嵌入维度，与t_emb相加
    │
    ├──→ 投影为4个额外token，拼接到文本token序列后
    │
    ▼
U-Net (GLIDE架构)
    │
    ▼
64x64 图像
```

### 20.3.2 CLIP嵌入的注入方式

DALL-E 2通过两种方式将CLIP图像嵌入注入到解码器中：

**方式1：加到时间步嵌入**

$$t_{emb}' = t_{emb} + \text{Linear}(z_i)$$

这为整个网络提供全局的语义信息。

**方式2：作为Cross-Attention的额外token**

$$K' = [K_{text}; K_{clip}], \quad V' = [V_{text}; V_{clip}]$$

其中 $K_{clip}, V_{clip}$ 是 $z_i$ 通过线性投影得到的4个额外token。这样解码器在每个空间位置都能attend到CLIP嵌入。

### 20.3.3 级联超分辨率

DALL-E 2使用级联（cascade）方式逐步提升分辨率：

```
Stage 1: 文本 + z_i → 64x64图像    (base模型)
    │
    ▼
Stage 2: 64x64 → 256x256           (超分辨率模型1)
    │
    ▼
Stage 3: 256x256 → 1024x1024       (超分辨率模型2)
```

每个阶段都是独立训练的扩散模型。超分辨率模型以低分辨率图像（经过噪声增强）为条件：

$$p(x_{high} | x_{low}, z_i, \text{text})$$

**噪声增强（Noise Augmentation）**：在训练和推理时，对低分辨率输入添加少量噪声，防止超分模型过度依赖低分辨率输入的细节（避免错误放大伪影）。

### 20.3.4 解码器训练

解码器的训练目标是标准的扩散去噪目标，但条件信息更丰富：

$$\mathcal{L}_{\text{decoder}} = \mathbb{E}_{t, x, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t, z_i, z_t, \text{text})\|^2\right]$$

训练时使用真实图像的CLIP嵌入 $z_i = \text{CLIP}_{image}(x)$ 作为条件。推理时使用先验模型生成的 $z_i$。

---

## 20.4 图像变体与混合

### 20.4.1 图像变体生成

DALL-E 2的两阶段设计天然支持图像变体生成：

1. 给定输入图像 $x$，提取CLIP图像嵌入 $z_i = \text{CLIP}_{image}(x)$
2. 直接将 $z_i$ 送入解码器（跳过先验）
3. 由于扩散模型的随机性，每次采样产生不同但语义相似的图像

$$x_{variant} \sim p(x | z_i)$$

这些变体保留了原图的核心语义（物体、场景类型、整体氛围），但改变了具体细节（姿态、光照、布局）。

### 20.4.2 图像嵌入插值

在CLIP嵌入空间中进行线性插值，可以产生平滑的图像变化：

$$z_{interp} = (1 - \lambda) z_{i_1} + \lambda z_{i_2}, \quad \lambda \in [0, 1]$$

将 $z_{interp}$ 送入解码器，可以生成从图像1到图像2的平滑过渡。由于CLIP嵌入空间是语义结构化的，这种插值在语义层面是连贯的——不是简单的像素混合，而是概念的渐变。

### 20.4.3 文本-图像混合

更灵活地，可以混合文本嵌入和图像嵌入：

$$z_{mix} = \alpha \cdot z_t + (1 - \alpha) \cdot z_i, \quad \alpha \in [0, 1]$$

- $\alpha = 0$：纯图像条件（图像变体）
- $\alpha = 1$：纯文本条件（文本到图像）
- $\alpha = 0.5$：文本和图像的混合

注意这里需要先将 $z_t$ 和 $z_i$ 归一化到相同的尺度，因为文本嵌入和图像嵌入的norm可能不同。

### 20.4.4 嵌入空间的算术

类似于Word2Vec的词向量算术，CLIP嵌入空间也支持语义算术：

$$z_{result} = z_{image} - z_{text_A} + z_{text_B}$$

例如：狗的图像嵌入 - "dog"的文本嵌入 + "cat"的文本嵌入 ≈ 猫的图像嵌入（保留原图的姿态和背景）。

这种属性在图像编辑中非常有用，但实际效果取决于CLIP嵌入空间的局部线性性。

---

## 20.5 与Stable Diffusion的对比

### 20.5.1 架构对比

| 维度 | DALL-E 2 | Stable Diffusion |
|------|----------|-----------------|
| 生成空间 | 像素空间（+级联超分） | VAE潜在空间 |
| 文本编码 | CLIP文本嵌入（单向量） | CLIP文本token序列 |
| 条件机制 | CLIP先验 + 嵌入注入 | Cross-Attention |
| 文本理解 | 通过CLIP先验间接理解 | 直接token级注意力 |
| 分辨率策略 | 64→256→1024级联 | 直接512x512 |
| 参数总量 | ~3.5B | ~890M |
| 开源 | 否 | 是 |

### 20.5.2 条件机制的关键差异

**DALL-E 2**使用CLIP图像嵌入（$\in \mathbb{R}^{768}$，单个向量）作为主要条件。这个向量携带了丰富的语义信息，但是是**全局的、压缩的**——它知道图像中有什么，但不精确知道在哪里。

**Stable Diffusion**使用CLIP文本的token序列（$\in \mathbb{R}^{77 \times 768}$）通过Cross-Attention注入。这种方式允许图像的每个空间位置**选择性地关注**文本的不同部分。

**后果**：SD在处理复杂构图（"左边一只猫，右边一条狗"）时通常优于DALL-E 2，因为Cross-Attention提供了空间-语义的细粒度对应。

### 20.5.3 质量对比

在早期评测中（2022年中）：

- **FID**：DALL-E 2略优（在COCO上）
- **文本一致性**：SD和DALL-E 2各有优劣，取决于场景复杂度
- **多样性**：DALL-E 2的两阶段设计天然支持更高多样性
- **速度**：SD显著更快（潜在空间 + 单阶段 vs 像素空间 + 级联）

### 20.5.4 生态差异

SD的开源特性使其形成了庞大的社区生态（ControlNet、LoRA、各种插件），而DALL-E 2仅通过API提供服务。这一差异最终使SD成为了研究和应用的主流平台。

---

## 20.6 层次化生成的一般化

### 20.6.1 级联扩散模型

DALL-E 2使用的级联策略是一种通用范式：

$$p(x) = p(x_{low}) \cdot p(x_{mid} | x_{low}) \cdot p(x_{high} | x_{mid})$$

每一级都是一个独立的扩散模型，专注于不同分辨率的细节。

**优点**：
- 各级模型更易训练（任务更简单）
- 低分辨率模型可以快速迭代
- 超分辨率模型可以独立改进

**缺点**：
- 多模型部署和推理更复杂
- 低分辨率阶段的错误会传播并放大
- 总推理时间可能很长

### 20.6.2 Imagen（Google, 2022）

Imagen采用了类似的级联策略，但在文本编码上做了关键创新：

- **文本编码器**：使用T5-XXL（4.6B参数的纯语言模型）替代CLIP
- **关键发现**：更大的纯语言模型比更大的图像模型更能提升生成质量
- **级联**：64x64 → 256x256 → 1024x1024
- **文本条件**：直接Cross-Attention，无需CLIP先验

Imagen的训练目标同样是标准的扩散去噪，但条件来自T5编码的文本嵌入：

$$\mathcal{L} = \mathbb{E}_{t, x, \epsilon}\left[\|w_t(\epsilon - \epsilon_\theta(x_t, t, c_{T5}))\|^2\right]$$

其中 $w_t$ 是依赖于 $t$ 的权重函数（v-prediction参数化）。

### 20.6.3 Parti（Google, 2022）

Parti采用了完全不同的路线——自回归方式：

1. **文本编码**：使用ViT-VQGAN将图像编码为离散token
2. **生成**：使用20B参数的Transformer自回归生成图像token
3. **后处理**：可选的超分辨率扩散模型

$$p(x_1, ..., x_N | \text{text}) = \prod_{i=1}^{N} p(x_i | x_{<i}, \text{text})$$

### 20.6.4 不同方法的系统比较

| 方法 | 文本编码 | 生成方式 | 层次 | FID↓ |
|------|---------|---------|------|------|
| DALL-E 2 | CLIP | 扩散+先验 | 64→256→1024 | 10.39 |
| Imagen | T5-XXL | 扩散 | 64→256→1024 | 7.27 |
| Parti | ViT-VQGAN | 自回归 | 256→1024 | 7.23 |
| SD v1.5 | CLIP ViT-L | 潜在扩散 | 直接512 | ~8.5 |

**趋势总结**：
- 更大的文本编码器比更大的图像模型更有效（Imagen的发现）
- 潜在空间方法在效率上远超像素空间方法
- 级联可以提升最终分辨率但增加系统复杂度
- 自回归方法可以利用语言模型的缩放经验，但速度较慢

---

## 代码实战

### 实战1：CLIP图像/文本嵌入提取与分析

```python
"""
DALL-E 2核心概念实战：
- CLIP嵌入提取
- 图像变体（概念演示）
- 嵌入空间插值
- 文本+图像混合
- CLIP嵌入空间PCA可视化
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


# ============================================================
# 1. CLIP嵌入提取
# ============================================================

class CLIPEmbeddingExtractor:
    """CLIP嵌入提取器，用于图像和文本"""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor
        
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        提取文本的CLIP嵌入
        
        Args:
            texts: 文本列表
        Returns:
            text_embeds: [N, 768] 归一化的文本嵌入
        """
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        
        text_embeds = self.model.get_text_features(**inputs)  # [N, 768]
        text_embeds = F.normalize(text_embeds, dim=-1)        # L2归一化
        return text_embeds
    
    @torch.no_grad()
    def encode_image(self, images: List["PIL.Image"]) -> torch.Tensor:
        """
        提取图像的CLIP嵌入
        
        Args:
            images: PIL图像列表
        Returns:
            image_embeds: [N, 768] 归一化的图像嵌入
        """
        inputs = self.processor(
            images=images,
            return_tensors="pt",
        ).to(self.device)
        
        image_embeds = self.model.get_image_features(**inputs)  # [N, 768]
        image_embeds = F.normalize(image_embeds, dim=-1)        # L2归一化
        return image_embeds
    
    @torch.no_grad()
    def compute_similarity(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算文本-图像相似度矩阵
        
        Args:
            text_embeds: [N_t, 768]
            image_embeds: [N_i, 768]
        Returns:
            similarity: [N_t, N_i] 余弦相似度矩阵
        """
        return text_embeds @ image_embeds.T  # [N_t, N_i]


# ============================================================
# 2. 图像变体生成（使用CLIP嵌入 + 解码器概念演示）
# ============================================================

class ImageVariationGenerator:
    """
    图像变体生成器（概念演示）
    真正的DALL-E 2解码器未开源，这里展示核心概念
    """
    
    def __init__(
        self,
        clip_extractor: CLIPEmbeddingExtractor,
        device: str = "cuda",
    ) -> None:
        self.clip = clip_extractor
        self.device = device
    
    def generate_variation_embeddings(
        self,
        image: "PIL.Image",
        noise_scale: float = 0.1,
        n_variations: int = 5,
    ) -> torch.Tensor:
        """
        生成图像变体的CLIP嵌入
        通过在CLIP嵌入空间添加噪声来模拟变体
        
        Args:
            image: 输入图像
            noise_scale: 噪声强度
            n_variations: 变体数量
        Returns:
            varied_embeds: [n_variations, 768]
        """
        # 提取原始图像嵌入
        original_embed = self.clip.encode_image([image])  # [1, 768]
        
        # 在嵌入空间添加高斯噪声
        noise = torch.randn(
            n_variations, 768, device=self.device
        ) * noise_scale  # [n_variations, 768]
        
        varied_embeds = original_embed + noise  # [n_variations, 768]
        varied_embeds = F.normalize(varied_embeds, dim=-1)
        
        return varied_embeds


# ============================================================
# 3. 图像嵌入插值
# ============================================================

def interpolate_clip_embeddings(
    embed_1: torch.Tensor,      # [768]
    embed_2: torch.Tensor,      # [768]
    n_steps: int = 10,
    method: str = "slerp",
) -> torch.Tensor:
    """
    在CLIP嵌入空间中进行插值
    
    Args:
        embed_1: 起始嵌入
        embed_2: 终止嵌入
        n_steps: 插值步数
        method: 插值方法 ("linear" 或 "slerp")
    Returns:
        interpolated: [n_steps, 768] 插值结果
    """
    lambdas = torch.linspace(0, 1, n_steps)
    
    if method == "linear":
        # 线性插值
        interpolated = torch.stack([
            (1 - lam) * embed_1 + lam * embed_2
            for lam in lambdas
        ])
        interpolated = F.normalize(interpolated, dim=-1)
    
    elif method == "slerp":
        # 球面线性插值（SLERP）——在单位球面上的等角速插值
        # 对于归一化的嵌入向量更合理
        dot = torch.dot(embed_1, embed_2).clamp(-1, 1)
        omega = torch.acos(dot)  # 两向量间的角度
        
        if omega.abs() < 1e-6:
            # 两个嵌入几乎相同，退化为线性插值
            interpolated = torch.stack([
                (1 - lam) * embed_1 + lam * embed_2
                for lam in lambdas
            ])
        else:
            interpolated = torch.stack([
                (torch.sin((1 - lam) * omega) / torch.sin(omega)) * embed_1
                + (torch.sin(lam * omega) / torch.sin(omega)) * embed_2
                for lam in lambdas
            ])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return interpolated  # [n_steps, 768]


# ============================================================
# 4. 文本+图像嵌入混合
# ============================================================

def mix_text_image_embeddings(
    text_embed: torch.Tensor,   # [768]
    image_embed: torch.Tensor,  # [768]
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    混合文本和图像的CLIP嵌入
    
    Args:
        text_embed: CLIP文本嵌入
        image_embed: CLIP图像嵌入
        alpha: 混合系数 (0=纯图像, 1=纯文本)
    Returns:
        mixed: [768] 混合后的嵌入
    """
    # 归一化到相同尺度
    text_embed = F.normalize(text_embed, dim=-1)
    image_embed = F.normalize(image_embed, dim=-1)
    
    # 混合
    mixed = alpha * text_embed + (1 - alpha) * image_embed
    mixed = F.normalize(mixed, dim=-1)
    
    return mixed


def semantic_arithmetic(
    image_embed: torch.Tensor,
    subtract_text: str,
    add_text: str,
    clip_extractor: CLIPEmbeddingExtractor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    CLIP嵌入空间的语义算术
    result = image_embed - text_A + text_B
    
    Args:
        image_embed: 原始图像嵌入 [768]
        subtract_text: 要减去的语义概念
        add_text: 要添加的语义概念
        clip_extractor: CLIP提取器
        scale: 编辑强度
    Returns:
        result: [768] 编辑后的嵌入
    """
    sub_embed = clip_extractor.encode_text([subtract_text])[0]  # [768]
    add_embed = clip_extractor.encode_text([add_text])[0]       # [768]
    
    # 语义算术
    delta = add_embed - sub_embed  # [768]
    result = image_embed + scale * delta
    result = F.normalize(result, dim=-1)
    
    return result


# ============================================================
# 5. CLIP嵌入空间PCA可视化
# ============================================================

def visualize_clip_embedding_space(
    texts: List[str],
    images: Optional[List["PIL.Image"]] = None,
    clip_extractor: Optional[CLIPEmbeddingExtractor] = None,
    save_path: str = "clip_pca.png",
) -> None:
    """
    使用PCA将CLIP嵌入空间降维到2D进行可视化
    
    Args:
        texts: 文本列表
        images: 可选的图像列表
        clip_extractor: CLIP提取器
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    if clip_extractor is None:
        clip_extractor = CLIPEmbeddingExtractor()
    
    # 提取嵌入
    text_embeds = clip_extractor.encode_text(texts).cpu().numpy()  # [N_t, 768]
    
    all_embeds = text_embeds
    labels = [f"T: {t[:20]}" for t in texts]
    colors = ["blue"] * len(texts)
    
    if images is not None:
        image_embeds = clip_extractor.encode_image(images).cpu().numpy()
        all_embeds = np.concatenate([all_embeds, image_embeds], axis=0)
        labels += [f"I: image_{i}" for i in range(len(images))]
        colors += ["red"] * len(images)
    
    # PCA降维
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_embeds)  # [N, 2]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (x, y) in enumerate(reduced):
        ax.scatter(x, y, c=colors[i], s=100, zorder=5)
        ax.annotate(
            labels[i],
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("CLIP Embedding Space (PCA Projection)")
    ax.legend(["Text", "Image"] if images else ["Text"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    
    print(f"PCA可视化已保存到 {save_path}")
    print(f"前2个主成分解释的方差比例: "
          f"{pca.explained_variance_ratio_[:2].sum():.1%}")


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 初始化CLIP提取器
    extractor = CLIPEmbeddingExtractor(device="cuda")
    
    # 文本嵌入提取
    texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a beautiful sunset over the ocean",
        "a modern city skyline at night",
        "an abstract painting with vivid colors",
    ]
    text_embeds = extractor.encode_text(texts)  # [5, 768]
    print(f"文本嵌入形状: {text_embeds.shape}")
    
    # 计算文本间的相似度
    sim_matrix = text_embeds @ text_embeds.T  # [5, 5]
    print("文本相似度矩阵:")
    for i, t1 in enumerate(texts):
        for j, t2 in enumerate(texts):
            if j > i:
                print(f"  '{t1[:25]}' <-> '{t2[:25]}': {sim_matrix[i,j]:.3f}")
    
    # 嵌入插值演示
    cat_embed = text_embeds[0]   # "a photo of a cat"
    dog_embed = text_embeds[1]   # "a photo of a dog"
    
    interpolated = interpolate_clip_embeddings(
        cat_embed, dog_embed, n_steps=5, method="slerp"
    )
    print(f"\n猫→狗的SLERP插值嵌入形状: {interpolated.shape}")
    
    # 计算插值点与原始嵌入的相似度
    cat_sims = interpolated @ cat_embed  # [5]
    dog_sims = interpolated @ dog_embed  # [5]
    print("插值过程中的相似度变化:")
    for i in range(5):
        print(f"  Step {i}: cat_sim={cat_sims[i]:.3f}, dog_sim={dog_sims[i]:.3f}")
    
    # PCA可视化
    visualize_clip_embedding_space(texts, clip_extractor=extractor)
```

---

## 本章小结

| 概念 | 核心要点 |
|------|---------|
| unCLIP框架 | 文本 → CLIP先验 → CLIP图像嵌入 → 解码器 → 图像 |
| 扩散先验 | 在768维CLIP嵌入空间中的扩散模型，使用 $x_0$-prediction |
| 先验的必要性 | 弥合文本-图像模态差异，将"一个点"映射为"一个分布" |
| 解码器 | 基于GLIDE的像素空间扩散模型，级联超分到1024x1024 |
| CLIP嵌入注入 | 同时通过时间步加法和Cross-Attention额外token注入 |
| 图像变体 | 直接从图像的CLIP嵌入解码，利用扩散随机性产生变化 |
| 嵌入插值 | SLERP插值在语义上更连贯，实现平滑的概念过渡 |
| 语义算术 | $z_{result} = z_{image} - z_{A} + z_{B}$，在嵌入空间中编辑语义 |
| 与SD对比 | SD使用token级Cross-Attention，空间控制更精确 |
| Imagen/Parti | T5文本编码更有效（Imagen）；自回归可利用LM缩放（Parti） |

---

## 练习题

### 基础题

**练习1**：解释DALL-E 2中"先验模型"的输入和输出分别是什么？为什么使用 $x_0$-prediction而非 $\epsilon$-prediction？

**练习2**：DALL-E 2的解码器通过哪两种方式注入CLIP图像嵌入？各自的作用是什么？

### 中级题

**练习3**：假设你有两张图像A（一只站立的猫）和B（一只坐着的狗），你在它们的CLIP嵌入之间进行插值。请分析：
- (a) 当 $\lambda = 0.5$ 时，生成的图像最可能是什么样的？
- (b) 线性插值（lerp）与球面线性插值（slerp）的结果有什么理论差异？
- (c) 如何修改插值方案使得中间图像更像"坐着的猫"？

**练习4**：Imagen使用T5-XXL（纯语言模型）替代CLIP作为文本编码器，并发现文本编码器的缩放比图像模型的缩放更能提升质量。请解释这一现象的可能原因，并讨论其对后续模型设计的影响。

### 提高题

**练习5**：设计一个实验来验证CLIP嵌入空间是否满足"局部线性性"假设。具体来说：
- (a) 定义"局部线性性"的数学形式
- (b) 设计测量方法（使用现有的CLIP模型和图像数据集）
- (c) 预测在哪些情况下线性假设会失效
- (d) 线性性的失效对DALL-E 2的图像插值和语义算术分别有什么影响？

---

## 练习答案

### 练习1

**先验模型的输入与输出**：

- **输入**：
  - CLIP文本嵌入 $z_t \in \mathbb{R}^{768}$
  - 原始文本的BPE token序列
  - 当前时间步 $t$
  - 带噪的CLIP图像嵌入 $z_i^{(t)} \in \mathbb{R}^{768}$

- **输出**：预测的干净CLIP图像嵌入 $\hat{z}_i \in \mathbb{R}^{768}$

**使用 $x_0$-prediction的原因**：

1. **低维空间特性**：CLIP嵌入仅768维（远小于像素空间的数十万维）。在低维空间中，$x_0$-prediction的方差更小，训练更稳定
2. **直接监督**：$x_0$-prediction直接预测目标嵌入，训练信号更直接
3. **嵌入空间的结构**：CLIP嵌入是归一化的、结构化的向量空间。$\epsilon$-prediction中的噪声方向可能破坏这种结构，而 $x_0$-prediction始终在有意义的嵌入空间中操作

### 练习2

DALL-E 2解码器注入CLIP图像嵌入的两种方式：

**方式1：加到时间步嵌入（Additive）**

$$t_{emb}' = t_{emb} + \text{Linear}(z_i)$$

- **作用**：提供全局的语义条件，影响整个网络的行为
- **类比**：类似于对整个图像设定一个"语义基调"
- **影响范围**：通过时间步嵌入传播到所有ResBlock

**方式2：Cross-Attention额外token**

将 $z_i$ 通过线性层投影为4个key-value对，拼接到文本token之后。

- **作用**：提供可被空间选择性关注的语义信息
- **类比**：每个空间位置可以"查询"CLIP嵌入以获取相关的视觉特征
- **影响范围**：通过注意力机制在不同空间位置施加不同强度的影响

两种方式互补：additive提供全局语义基调，Cross-Attention提供空间可选的细粒度信息。

### 练习3

**(a) $\lambda = 0.5$ 的插值结果**：

中间嵌入 $z_{0.5} = 0.5 z_A + 0.5 z_B$ 位于猫和狗嵌入的中间位置。解码器生成的图像可能是：
- 一只具有猫和狗特征的动物（模糊的物种边界）
- 或者生成两种动物之一，但可能在姿态上取中间值

实际效果取决于CLIP嵌入空间在这两点之间的拓扑结构。

**(b) Lerp vs Slerp的差异**：

- **Lerp**：$z = (1-\lambda)z_1 + \lambda z_2$。中间点的模长会小于端点（向量加法的几何性质），归一化后可能导致不均匀的语义变化
- **Slerp**：在单位球面上等角度插值。保持模长恒定，语义变化更均匀。数学上更适合归一化的嵌入向量

对于归一化的CLIP嵌入，Slerp通常产生更平滑的过渡。

**(c) 生成"坐着的猫"**：

使用语义算术而非简单插值：

$$z_{result} = z_A + (z_B - z_{B'}) \cdot \alpha$$

其中 $z_{B'} = \text{CLIP}(\text{"a standing dog"})$，这样 $z_B - z_{B'}$ 编码了"坐→站"的语义差异，加到 $z_A$ 上实现姿态迁移。

或者更直接地：

$$z_{result} = z_A - \text{CLIP}(\text{"standing"}) + \text{CLIP}(\text{"sitting"})$$

### 练习4

**T5-XXL优于CLIP的可能原因**：

1. **语言理解深度**：T5-XXL（11B参数）在海量纯文本上预训练，对语言的理解（特别是复杂句法、属性绑定、空间关系）远超CLIP的文本编码器。CLIP的文本编码器本质上是为对比学习优化的，而非为语言理解优化
2. **长尾概念**：T5接触的文本覆盖面远大于CLIP的图文对数据，能更好地理解罕见概念和复杂描述
3. **组合性泛化**：T5的自注意力机制在处理"A and B"这类组合描述时更精确

**对后续设计的影响**：
- SDXL采用双编码器（CLIP + OpenCLIP），增加文本理解能力
- SD3/FLUX进一步采用T5编码器
- 趋势：生成模型的质量瓶颈越来越多在"理解力"而非"绘画力"上

### 练习5

**(a) 局部线性性的数学定义**：

CLIP嵌入空间在点 $z_0$ 附近的局部线性性指：对于小扰动 $\delta_1, \delta_2$，存在解码函数 $D$ 使得：

$$D(z_0 + \alpha\delta_1 + \beta\delta_2) \approx D(z_0) + \alpha D'(\delta_1) + \beta D'(\delta_2)$$

即嵌入空间中的线性组合应对应于图像空间中的语义线性组合。

**(b) 测量方法**：

1. 选取一组图像 $\{x_i\}$，提取嵌入 $\{z_i\}$
2. 对于每对图像 $(i, j)$，计算中间插值 $z_{0.5} = 0.5 z_i + 0.5 z_j$
3. 使用图像检索：找到与 $z_{0.5}$ 最近的真实图像
4. 人工评估该图像是否确实是语义的"中间态"
5. 量化指标：定义语义距离函数 $d_s$，检验 $d_s(D(z_{0.5}), D(z_i)) \approx d_s(D(z_{0.5}), D(z_j))$

**(c) 线性性失效的场景**：

- **跨类别插值**：猫↔汽车之间缺乏有意义的中间类别
- **对抗性方向**：某些方向可能导致嵌入离开自然图像的流形
- **高曲率区域**：在语义边界附近（如"猫"和"狗"之间），流形曲率可能很大

**(d) 失效的影响**：

- **图像插值**：中间帧可能出现不自然的伪影或模糊
- **语义算术**：编辑方向不准确，可能改变预期外的属性（如改颜色时同时改了形状）

---

## 延伸阅读

1. Ramesh, A., et al. (2022). "Hierarchical Text-Conditional Image Generation with CLIP Latents." *arXiv:2204.06125*. (DALL-E 2 / unCLIP)
2. Ramesh, A., et al. (2021). "Zero-Shot Text-to-Image Generation." *ICML 2021*. (DALL-E 1)
3. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021*. (CLIP)
4. Saharia, C., et al. (2022). "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding." *NeurIPS 2022*. (Imagen)
5. Yu, J., et al. (2022). "Scaling Autoregressive Models for Content-Rich Text-to-Image Generation." *TMLR 2022*. (Parti)
6. Nichol, A., et al. (2021). "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models." *ICML 2022*.

---

<div align="center">

[⬅️ 第十九章：Stable Diffusion原理与实现](19-stable-diffusion.md) | [📖 目录](../README.md) | [第二十一章：DiT：扩散Transformer ➡️](21-dit-diffusion-transformer.md)

</div>
