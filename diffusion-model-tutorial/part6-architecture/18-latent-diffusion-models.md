# 第十八章：潜在扩散模型（LDM）

> **本章导读**：潜在扩散模型（Latent Diffusion Model, LDM）是Stable Diffusion的理论基础，也是扩散模型从学术研究走向工业应用的关键突破。它的核心思想极其优雅：既然高分辨率像素空间的扩散计算代价高昂，为什么不先用自编码器将图像压缩到低维潜在空间，然后在这个紧凑的空间中做扩散？本章将完整解析LDM的每个组件，从VAE感知压缩到潜在空间扩散，从条件注入到完整推理管线。

**前置知识**：变分自编码器（VAE）基础、扩散模型正向/反向过程、U-Net架构（第16章）、交叉注意力（第17章）

**预计学习时间**：5-6小时

## 学习目标

完成本章学习后，你将能够：

1. 分析像素空间扩散的计算瓶颈，解释LDM将扩散转移到潜在空间的动机与理论依据
2. 实现KL正则化VAE（AutoencoderKL），包括编码器、解码器和对角高斯分布采样
3. 理解LDM两阶段训练流程（先训练VAE，再冻结VAE训练扩散模型）并实现完整的训练循环
4. 掌握交叉注意力条件化的统一框架，理解文本/图像/分割图等不同模态如何注入扩散模型
5. 对比潜在空间与像素空间的扩散质量、计算开销和语义特性，使用预训练模型进行实际实验

---

## 18.1 像素空间扩散的局限

### 18.1.1 高分辨率图像的计算挑战

在像素空间直接进行扩散的计算开销与图像分辨率的平方成正比。考虑一张 $512 \times 512 \times 3$ 的RGB图像：

$$\text{像素数} = 512 \times 512 \times 3 = 786,432$$

U-Net需要对这个高维空间中的张量执行大量卷积和注意力操作。具体的计算瓶颈包括：

| 操作 | 256x256 | 512x512 | 1024x1024 |
|------|---------|---------|-----------|
| 特征图内存 | 2 GB | 8 GB | 32 GB |
| 卷积FLOPs | ~50 GFLOPs | ~200 GFLOPs | ~800 GFLOPs |
| 注意力(32x32) | 可行 | 勉强 | 不可行 |
| 训练时间（单V100） | ~1周 | ~1月 | >1年 |

### 18.1.2 自然图像中的冗余

自然图像包含大量的冗余信息，这可以从信息论角度理解：

**空间冗余**：相邻像素高度相关。一张 $512 \times 512$ 的自然图像，其信息熵远低于 $512 \times 512 \times 8 \times 3 = 6.3M$ bit的理论上限。JPEG压缩通常可以将自然图像压缩到原始大小的5-10%而不产生明显的质量损失。

**频谱冗余**：自然图像的能量集中在低频分量。对自然图像做傅里叶变换会发现，高频分量的能量占比很小。扩散模型在像素空间操作时，大量计算浪费在了这些低信息量的高频细节上。

**感知冗余**：人类视觉系统对某些变化不敏感（如色度的微小差异）。在像素空间中，扩散模型需要精确重建每个像素值，包括人眼无法分辨的差异。

### 18.1.3 训练代价分析

Dhariwal & Nichol (2021) 的ADM模型在ImageNet 256x256上训练：

- **模型参数**：554M
- **训练数据**：120万张图像
- **训练时间**：约2000 V100 GPU小时
- **FID-50K**：4.59

而DALL-E 2和Imagen等更大模型的训练代价达到了数千乃至数万GPU天。

**核心问题**：对于图像生成这个任务，我们是否真的需要在 $512 \times 512 \times 3 = 786K$ 维的空间中做扩散？

### 18.1.4 解决思路

LDM的核心洞察（Rombach et al. 2022）：

> 将高维像素空间的扩散过程分解为两个步骤：
> 1. **感知压缩**（Perceptual Compression）：用自编码器将图像压缩到低维潜在空间
> 2. **语义扩散**（Semantic Diffusion）：在紧凑的潜在空间中执行扩散过程

$$x \in \mathbb{R}^{H \times W \times 3} \xrightarrow{\mathcal{E}} z \in \mathbb{R}^{h \times w \times c} \xrightarrow{\text{扩散}} \hat{z} \xrightarrow{\mathcal{D}} \hat{x} \in \mathbb{R}^{H \times W \times 3}$$

其中 $h = H/f$，$w = W/f$，$f$ 是下采样因子。以 $f=8$ 为例：

$$512 \times 512 \times 3 \xrightarrow{\mathcal{E}} 64 \times 64 \times 4$$

维度从786,432降低到16,384，**减少了48倍**。

---

## 18.2 感知压缩（VAE编码器-解码器）

### 18.2.1 LDM中的VAE设计

LDM使用的VAE（Variational Autoencoder）不是标准的VAE，而是一个经过精心设计的**感知压缩模型**。其目标不是学习生成模型（那是扩散模型的工作），而是学习一个高质量的图像压缩器-解压缩器。

编码器 $\mathcal{E}$ 将图像映射到潜在空间：

$$\mathcal{E}: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{h \times w \times 2c}$$

注意输出通道数是 $2c$，因为VAE编码器输出的是高斯分布的均值 $\mu$ 和对数方差 $\log \sigma^2$，各占 $c$ 个通道。

通过重参数化采样：

$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

得到潜在码 $z \in \mathbb{R}^{h \times w \times c}$。

解码器 $\mathcal{D}$ 将潜在码恢复为图像：

$$\mathcal{D}: \mathbb{R}^{h \times w \times c} \to \mathbb{R}^{H \times W \times 3}$$

### 18.2.2 压缩因子的选择

压缩因子 $f$ 是一个关键的设计选择，它决定了信息保留量和计算节省量之间的平衡：

| 压缩因子 $f$ | 潜在空间大小(512输入) | 维度比 | 重建质量 | 用途 |
|:---:|:---:|:---:|:---:|:---:|
| 1 | $512 \times 512 \times c$ | ~1x | 完美 | 无压缩（像素扩散） |
| 2 | $256 \times 256 \times c$ | ~4x | 极好 | 轻度压缩 |
| 4 | $128 \times 128 \times c$ | ~16x | 好 | SD 2.x, SDXL |
| 8 | $64 \times 64 \times c$ | ~64x | 较好 | SD 1.x |
| 16 | $32 \times 32 \times c$ | ~256x | 一般 | 压缩过度 |

Rombach et al. (2022) 的实验表明：
- $f \leq 4$：重建质量极好，但计算节省有限
- $f = 4 \sim 8$：重建质量好，计算节省显著，**最佳平衡点**
- $f \geq 16$：重建质量下降明显，信息损失过大

Stable Diffusion 1.x使用 $f=8$，$c=4$，即 $512 \times 512$ 图像被压缩到 $64 \times 64 \times 4$。

### 18.2.3 KL正则化 VAE vs VQ-VAE

LDM支持两种VAE变体：

**KL-reg VAE（连续潜在空间）**：

$$\mathcal{L}_{KL} = D_{KL}(q(z|x) \| \mathcal{N}(0, I))$$

对编码器输出的高斯分布施加KL正则化，使其接近标准正态分布。这使得潜在空间平滑且连续，适合扩散过程。

Stable Diffusion使用的就是KL-reg VAE，其KL权重很小（$\lambda_{KL} \approx 10^{-6}$），主要靠重建损失驱动，KL项只是轻度正则化以防止潜在空间崩塌。

**VQ-VAE（离散潜在空间）**：

$$z_q = \text{Quantize}(z_e) = \arg\min_{e_k \in \mathcal{C}} \|z_e - e_k\|$$

将连续的编码器输出量化到离散的codebook向量。VQ-VAE在某些任务上重建质量更好，但离散空间不太适合连续的扩散过程。

### 18.2.4 感知损失与判别器

VAE的训练目标不仅仅是像素级重建，还包括感知质量：

$$\mathcal{L}_{VAE} = \underbrace{\mathcal{L}_{rec}}_{\text{重建}} + \underbrace{\lambda_{KL} \cdot \mathcal{L}_{KL}}_{\text{正则化}} + \underbrace{\lambda_{perc} \cdot \mathcal{L}_{perc}}_{\text{感知}} + \underbrace{\lambda_{adv} \cdot \mathcal{L}_{adv}}_{\text{对抗}}$$

各项含义：

- **$\mathcal{L}_{rec}$**：L1或L2像素重建损失
- **$\mathcal{L}_{KL}$**：KL散度正则化
- **$\mathcal{L}_{perc}$**：LPIPS感知损失（基于VGG特征的感知相似度）
- **$\mathcal{L}_{adv}$**：PatchGAN判别器的对抗损失

PatchGAN判别器是一个小型CNN，它不判断整张图像是真还是假，而是对图像的每个 $N \times N$ 的patch独立判断，输出一个空间概率图。这种设计在保持感知质量的同时降低了判别器的计算开销。

---

## 18.3 潜在空间中的扩散

### 18.3.1 前向过程

在潜在空间中，前向扩散过程与像素空间完全一致，只是操作对象从 $x$ 变为 $z$：

$$z_t = \sqrt{\bar{\alpha}_t} \cdot z_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $z_0 = \mathcal{E}(x)$ 是编码后的潜在码。

### 18.3.2 训练目标

LDM的训练目标：

$$\mathcal{L}_{LDM} = \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,I), t}\left[\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|^2\right]$$

其中：
- $z_t$ 是第 $t$ 步的含噪潜在码
- $\epsilon_\theta$ 是U-Net噪声预测网络
- $\tau_\theta(y)$ 是条件编码器（如CLIP文本编码器），将条件 $y$ 编码为特征序列
- $y$ 可以是文本、类别标签、图像等任意条件

关键区别：U-Net在 $64 \times 64 \times 4$（而非 $512 \times 512 \times 3$）的空间中操作，计算量大幅减少。

### 18.3.3 推理过程

LDM的推理管线：

$$\hat{z}_T \sim \mathcal{N}(0, I) \xrightarrow{\text{逐步去噪}} \hat{z}_{T-1} \to \cdots \to \hat{z}_0 \xrightarrow{\mathcal{D}} \hat{x}$$

完整步骤：
1. 从标准正态分布采样初始噪声 $\hat{z}_T \in \mathbb{R}^{h \times w \times c}$
2. 使用训练好的U-Net逐步去噪（DDPM/DDIM/DPM-Solver等采样器）
3. 得到干净的潜在码 $\hat{z}_0$
4. 使用VAE解码器重建图像：$\hat{x} = \mathcal{D}(\hat{z}_0)$

### 18.3.4 为什么潜在空间更好？

从多个角度理解潜在空间扩散的优势：

**计算效率**：以 $f=8$ 为例，潜在空间的空间尺寸是像素空间的 $1/64$。U-Net的卷积操作（FLOPs与空间尺寸平方成正比）和注意力操作（与空间尺寸四次方成正比）都得到了巨大的加速。

**语义空间**：VAE的潜在空间比像素空间更接近语义空间。像素空间中一个微小的扰动（如一个像素的亮度变化1/255）在语义上几乎无意义，但在数值上是一个确定的变化。潜在空间中的变化更倾向于对应有意义的语义变化。

**信息密度**：潜在空间中的每个"像素"携带的信息量远高于像素空间。VAE已经去除了空间冗余和感知冗余，留下的是高信息密度的表示。扩散模型因此可以将计算预算集中在生成有意义的内容上。

$$\text{计算节省} \approx f^2 = 64 \text{倍（空间尺寸）}$$

$$\text{注意力节省} \approx f^4 = 4096 \text{倍（若用同分辨率的注意力）}$$

---

## 18.4 条件机制（Cross-Attention注入）

### 18.4.1 统一的条件化框架

LDM提出了一个优雅的统一条件化框架。不管条件 $y$ 是什么模态（文本、图像、分割图、边缘图...），都通过以下流程注入：

$$y \xrightarrow{\tau_\theta} h_y \in \mathbb{R}^{M \times d_\tau} \xrightarrow{\text{Cross-Attention}} \text{U-Net各层}$$

其中 $\tau_\theta$ 是领域特定的条件编码器：

| 条件类型 | 编码器 $\tau_\theta$ | 输出维度 $M \times d_\tau$ |
|---------|---------------------|-------------------------|
| 文本 | CLIP ViT-L/14 | $77 \times 768$ |
| 文本（v2） | OpenCLIP ViT-H/14 | $77 \times 1024$ |
| 文本（SDXL） | CLIP + OpenCLIP | $77 \times 2048$ |
| 类别标签 | nn.Embedding | $1 \times d$ |
| 图像 | CLIP Image Encoder | $257 \times 1024$ |
| 分割图 | 卷积编码器 | 空间对齐特征 |

### 18.4.2 文本条件的编码

以Stable Diffusion 1.5为例，文本条件处理流程：

1. **Tokenization**：使用CLIP tokenizer将文本分词，填充或截断到77个token
2. **CLIP编码**：通过CLIP ViT-L/14的文本编码器得到 $h_{text} \in \mathbb{R}^{77 \times 768}$
3. **Cross-Attention注入**：在U-Net的每个SpatialTransformer层中，$h_{text}$ 作为K和V

每个文本token对应的768维向量包含了该token在CLIP语义空间中的表示。交叉注意力使得U-Net中的每个空间位置都可以"查询"文本信息，决定在该位置生成什么内容。

### 18.4.3 多模态条件注入

LDM框架的灵活性在于，交叉注意力的K/V可以来自任意编码器：

**ControlNet**（Zhang et al. 2023）：将边缘图、深度图、姿态等空间条件通过一个额外的编码器分支注入U-Net的跳跃连接。

**IP-Adapter**（Ye et al. 2023）：将参考图像通过CLIP图像编码器编码，然后通过解耦的交叉注意力（与文本交叉注意力并行）注入U-Net。

**T2I-Adapter**：更轻量的条件注入方式，通过特征加法而非交叉注意力。

### 18.4.4 Classifier-Free Guidance在LDM中的应用

Classifier-Free Guidance（CFG）在LDM中的实现：

$$\hat{\epsilon}_\theta(z_t, t, c) = (1 + w) \cdot \epsilon_\theta(z_t, t, c) - w \cdot \epsilon_\theta(z_t, t, \varnothing)$$

其中 $w$ 是引导强度（guidance scale），$c$ 是条件，$\varnothing$ 是无条件（空文本""）。

在训练时，以一定概率（如10%）随机将条件替换为空条件，这样同一个模型既能做条件生成也能做无条件生成。

推理时：
- $w = 1$：无引导，等同于条件生成
- $w = 7.5$：Stable Diffusion的默认值，平衡质量和多样性
- $w > 15$：过度引导，图像会过饱和和过度锐化

---

## 18.5 潜在空间的特性

### 18.5.1 潜在空间的可视化

VAE的潜在空间具有有趣的几何结构：

**PCA可视化**：对大量图像的潜在码做PCA，前几个主成分通常对应于：
- PC1：亮度/曝光
- PC2：色调/色彩
- PC3：复杂度/纹理密度

**t-SNE可视化**：相似图像的潜在码在t-SNE图中聚类在一起，说明潜在空间编码了有意义的语义信息。

### 18.5.2 潜在空间插值

两个图像 $x_A$ 和 $x_B$ 之间可以在潜在空间中进行平滑插值：

**线性插值（LERP）**：

$$z_\alpha = (1 - \alpha) \cdot z_A + \alpha \cdot z_B$$

**球面线性插值（SLERP）**：

$$z_\alpha = \frac{\sin((1-\alpha)\theta)}{\sin\theta} z_A + \frac{\sin(\alpha\theta)}{\sin\theta} z_B$$

其中 $\theta = \arccos\left(\frac{z_A \cdot z_B}{\|z_A\| \|z_B\|}\right)$。

SLERP通常产生更自然的插值结果，因为高维高斯分布的样本倾向于分布在球壳上（而非球体内部），线性插值的中间点可能落入低概率区域。

### 18.5.3 潜在空间的语义结构

与GAN的潜在空间类似，VAE的潜在空间也具有一定的语义结构：

- **语义方向**：存在对应特定语义属性的方向（如微笑方向、年龄方向）
- **局部线性性**：在潜在空间的局部区域，语义变化近似线性
- **分辨率解耦**：低频信息和高频信息在潜在空间中有一定程度的解耦

### 18.5.4 潜在空间 vs 像素空间的质量对比

Rombach et al. (2022) 的对比实验（ImageNet 256x256，class-conditional）：

| 模型 | 空间 | 参数量 | FID↓ | IS↑ | 训练成本 |
|------|------|--------|------|-----|---------|
| ADM | 像素 | 554M | 10.94 | 101.0 | 1000 V100天 |
| ADM-G | 像素 | 554M | 4.59 | 186.7 | 1000 V100天 |
| LDM-4 | 潜在($f$=4) | 400M | 10.56 | 103.5 | 35 V100天 |
| LDM-8 | 潜在($f$=8) | 400M | 15.51 | 79.0 | 5 V100天 |
| LDM-4-G | 潜在($f$=4) | 400M | 3.60 | 247.7 | 35 V100天 |

关键发现：
- LDM在训练成本**降低20-200倍**的情况下达到了可比甚至更好的FID
- $f=4$ 是质量最优的压缩率，$f=8$ 在质量和效率之间取得了好的平衡
- 加上Classifier-Free Guidance后，LDM超越了像素空间的ADM

---

## 18.6 LDM的训练与推理流程

### 18.6.1 两阶段训练

**阶段一：训练VAE**

VAE独立训练，目标是学习高质量的图像压缩：

$$\mathcal{L}_{VAE} = \|x - \mathcal{D}(\mathcal{E}(x))\|_1 + \lambda_{perc} \cdot \text{LPIPS}(x, \mathcal{D}(\mathcal{E}(x))) + \lambda_{KL} \cdot D_{KL}(q(z|x) \| \mathcal{N}(0, I)) + \lambda_{adv} \cdot \mathcal{L}_{GAN}$$

训练完成后VAE被冻结，作为扩散模型的"翻译层"。

**阶段二：训练扩散模型（U-Net）**

VAE参数完全冻结，只训练U-Net和条件编码器（如果需要微调的话）：

```
训练循环:
  1. 从数据集加载 (图像x, 条件y)
  2. 编码: z = E(x)  [VAE编码器，无梯度]
  3. 采样噪声和时间步: ε ~ N(0,I), t ~ U(1,T)
  4. 加噪: z_t = sqrt(ᾱ_t) * z + sqrt(1-ᾱ_t) * ε
  5. 预测噪声: ε̂ = UNet(z_t, t, τ(y))
  6. 计算损失: L = ||ε - ε̂||²
  7. 反向传播，更新UNet参数
```

这种两阶段解耦的好处：
- VAE可以在大规模数据上预训练，然后复用于多个扩散模型
- 扩散模型在低维空间训练，大幅降低计算成本
- 两个组件可以独立优化和迭代

### 18.6.2 推理管线

以Stable Diffusion为例的完整推理流程：

```
文本: "a photo of a cat on a beach, sunset"
  │
  ▼
[CLIP Text Encoder] → h_text ∈ R^{77×768}
  │
  ▼
[采样初始噪声] z_T ~ N(0, I) ∈ R^{64×64×4}
  │
  ▼
[U-Net去噪循环] (50步 DDIM / 20步 DPM-Solver++)
  │  每步: ε̂ = UNet(z_t, t, h_text)
  │        z_{t-1} = 采样器更新(z_t, ε̂, t)
  │
  ▼
[VAE Decoder] z_0 → x̂ ∈ R^{512×512×3}
  │
  ▼
输出图像
```

### 18.6.3 内存优化技术

在实际部署中，以下技术用于减少内存占用：

**Gradient Checkpointing**：在U-Net的前向传播中不保存中间激活值，反向传播时重新计算。用时间换内存，通常增加30%训练时间，但可节省60-70%激活内存。

**xFormers**：使用memory-efficient attention实现，将注意力的内存占用从 $O(N^2)$ 降低到 $O(N)$。

**VAE Tiling**：对于超高分辨率图像，VAE编码/解码时将图像分割为小块分别处理，然后拼合。

**Half Precision（FP16/BF16）**：使用混合精度训练，将大部分计算转为16位浮点数，内存减半且速度提升。

**模型卸载（Model Offloading）**：在推理时，将暂不使用的模型（如VAE）卸载到CPU，只在需要时加载到GPU。

---

## 代码实战

```python
"""
潜在扩散模型（LDM）完整实现
==========================
包含：
1. VAE编码器/解码器
2. 对角高斯分布（重参数化）
3. AutoencoderKL
4. 潜在空间扩散训练循环
5. 潜在空间可视化
6. 像素空间vs潜在空间开销对比
7. 使用预训练SD VAE的演示

参考：Rombach et al. 2022 (LDM / Stable Diffusion)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 辅助模块
# ============================================================

class ResBlock2d(nn.Module):
    """2D残差块，用于VAE编码器和解码器。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape: (B, C_in, H, W)
        Returns:
            shape: (B, C_out, H, W)
        """
        h = F.silu(self.norm1(x))
        h = self.conv1(h)         # shape: (B, C_out, H, W)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)         # shape: (B, C_out, H, W)
        return h + self.shortcut(x)


class SelfAttention2d(nn.Module):
    """轻量级自注意力，用于VAE的瓶颈层。
    
    Args:
        channels: 通道数
    """
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)  # shape: (B, 3, C, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # 各 shape: (B, C, N)
        
        q = q.permute(0, 2, 1)  # shape: (B, N, C)
        # 使用 sdpa
        attn_out = F.scaled_dot_product_attention(
            q, k.permute(0, 2, 1), v.permute(0, 2, 1)
        )  # shape: (B, N, C)
        
        attn_out = attn_out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(attn_out)


# ============================================================
# 1. VAE 编码器
# ============================================================

class Encoder(nn.Module):
    """VAE编码器：将图像压缩到潜在空间。
    
    结构：输入卷积 → [ResBlock + Downsample] × N → 瓶颈(ResBlock + Attn) → 输出卷积
    
    Args:
        in_channels: 输入图像通道数（RGB=3）
        base_channels: 基准通道数
        channel_mult: 各级别通道数倍数
        num_res_blocks: 每个级别的残差块数量
        latent_channels: 潜在空间通道数（输出为2倍，用于均值和方差）
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
    ) -> None:
        super().__init__()
        
        # 输入卷积
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            block = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                block.append(ResBlock2d(ch, out_ch))
                ch = out_ch
            
            # 除了最后一层，都添加下采样
            if i < len(channel_mult) - 1:
                block.append(
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1)  # 下采样
                )
            
            self.down_blocks.append(block)
        
        # 瓶颈层
        self.mid_block = nn.ModuleList([
            ResBlock2d(ch, ch),
            SelfAttention2d(ch),
            ResBlock2d(ch, ch),
        ])
        
        # 输出层：归一化 + 卷积到 2*latent_channels（均值+对数方差）
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, 2 * latent_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像, shape: (B, 3, H, W)
            
        Returns:
            潜在分布参数, shape: (B, 2*latent_channels, H/f, W/f)
        """
        h = self.conv_in(x)  # shape: (B, base_ch, H, W)
        
        for block in self.down_blocks:
            for layer in block:
                h = layer(h)
        
        for layer in self.mid_block:
            h = layer(h)
        
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)  # shape: (B, 2*latent_ch, H/f, W/f)
        
        return h


# ============================================================
# 2. VAE 解码器
# ============================================================

class Decoder(nn.Module):
    """VAE解码器：将潜在码恢复为图像。
    
    结构：输入卷积 → 瓶颈(ResBlock + Attn) → [ResBlock + Upsample] × N → 输出卷积
    
    Args:
        out_channels: 输出图像通道数（RGB=3）
        base_channels: 基准通道数
        channel_mult: 各级别通道数倍数（与编码器镜像）
        num_res_blocks: 每个级别的残差块数量
        latent_channels: 潜在空间通道数
    """
    
    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
    ) -> None:
        super().__init__()
        
        # 最深层的通道数
        ch = base_channels * channel_mult[-1]
        
        # 输入卷积
        self.conv_in = nn.Conv2d(latent_channels, ch, 3, padding=1)
        
        # 瓶颈层
        self.mid_block = nn.ModuleList([
            ResBlock2d(ch, ch),
            SelfAttention2d(ch),
            ResBlock2d(ch, ch),
        ])
        
        # 上采样路径（从深到浅）
        self.up_blocks = nn.ModuleList()
        
        for i, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = base_channels * mult
            block = nn.ModuleList()
            
            for _ in range(num_res_blocks + 1):  # 解码器多一个ResBlock
                block.append(ResBlock2d(ch, out_ch))
                ch = out_ch
            
            # 除了第一级（最浅层），都添加上采样
            if i > 0:
                block.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(ch, ch, 3, padding=1),
                    )
                )
            
            self.up_blocks.append(block)
        
        # 输出层
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 潜在码, shape: (B, latent_channels, h, w)
            
        Returns:
            重建图像, shape: (B, 3, H, W)
        """
        h = self.conv_in(z)  # shape: (B, ch, h, w)
        
        for layer in self.mid_block:
            h = layer(h)
        
        for block in self.up_blocks:
            for layer in block:
                h = layer(h)
        
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)  # shape: (B, 3, H, W)
        
        return h


# ============================================================
# 3. 对角高斯分布
# ============================================================

class DiagonalGaussianDistribution:
    """对角高斯分布：用于VAE的重参数化采样。
    
    接收编码器输出的 (mean, logvar) 参数，支持采样和KL散度计算。
    
    Args:
        parameters: 编码器输出, shape: (B, 2*C, H, W)，前C个通道为均值，后C个为对数方差
    """
    
    def __init__(self, parameters: torch.Tensor) -> None:
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        # 截断对数方差以保持数值稳定
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
    
    def sample(self) -> torch.Tensor:
        """重参数化采样: z = mean + std * epsilon。
        
        Returns:
            采样结果, shape: (B, C, H, W)
        """
        eps = torch.randn_like(self.std)
        return self.mean + self.std * eps
    
    def kl_divergence(self) -> torch.Tensor:
        """计算与标准正态分布的KL散度。
        
        $$D_{KL}(q(z|x) || N(0,I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)$$
        
        Returns:
            每个样本的KL散度, shape: (B,)
        """
        kl = -0.5 * torch.sum(
            1.0 + self.logvar - self.mean.pow(2) - self.logvar.exp(),
            dim=[1, 2, 3],
        )
        return kl
    
    def mode(self) -> torch.Tensor:
        """返回分布的众数（即均值）。
        
        Returns:
            均值, shape: (B, C, H, W)
        """
        return self.mean


# ============================================================
# 4. AutoencoderKL（完整KL-VAE）
# ============================================================

class AutoencoderKL(nn.Module):
    """KL正则化变分自编码器。
    
    这是Stable Diffusion中使用的VAE架构的简化版本。
    
    Args:
        in_channels: 输入图像通道数
        base_channels: 基准通道数
        channel_mult: 通道数倍数
        num_res_blocks: 残差块数量
        latent_channels: 潜在空间通道数
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
    ) -> None:
        super().__init__()
        
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels,
        )
        
        self.decoder = Decoder(
            out_channels=in_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels,
        )
        
        # 缩放因子（SD使用0.18215使潜在码方差接近1）
        self.scale_factor = 0.18215
    
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """编码图像到潜在分布。
        
        Args:
            x: 输入图像, shape: (B, 3, H, W)
            
        Returns:
            潜在高斯分布
        """
        params = self.encoder(x)  # shape: (B, 2*latent_ch, H/f, W/f)
        return DiagonalGaussianDistribution(params)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码潜在码到图像。
        
        Args:
            z: 潜在码, shape: (B, latent_ch, H/f, W/f)
            
        Returns:
            重建图像, shape: (B, 3, H, W)
        """
        return self.decoder(z)
    
    def forward(
        self, x: torch.Tensor, sample_posterior: bool = True
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """完整前向传播：编码 -> 采样 -> 解码。
        
        Args:
            x: 输入图像, shape: (B, 3, H, W)
            sample_posterior: 是否从后验采样（False则使用均值）
            
        Returns:
            (重建图像, 潜在分布)
        """
        posterior = self.encode(x)
        
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        
        reconstruction = self.decode(z)
        return reconstruction, posterior


# ============================================================
# 5. 潜在扩散模型
# ============================================================

class SimpleUNetForLatent(nn.Module):
    """简化版U-Net，用于潜在空间扩散。
    
    这是一个教学用的精简实现，展示LDM中U-Net的核心结构。
    生产环境请使用第16章的完整U-Net或diffusers库。
    
    Args:
        latent_channels: 潜在空间通道数
        model_channels: 基准模型通道数
        channel_mult: 通道倍数
        context_dim: 条件特征维度（用于交叉注意力）
    """
    
    def __init__(
        self,
        latent_channels: int = 4,
        model_channels: int = 256,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        context_dim: int = 768,
    ) -> None:
        super().__init__()
        
        time_dim = model_channels * 4
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # 输入卷积
        self.conv_in = nn.Conv2d(latent_channels, model_channels, 3, padding=1)
        
        # 简化的编码器：3级下采样
        ch = model_channels
        self.enc1 = TimestepResBlock(ch, ch * channel_mult[0], time_dim)
        self.down1 = nn.Conv2d(ch * channel_mult[0], ch * channel_mult[0], 3, stride=2, padding=1)
        
        ch1 = ch * channel_mult[0]
        self.enc2 = TimestepResBlock(ch1, ch * channel_mult[1], time_dim)
        self.down2 = nn.Conv2d(ch * channel_mult[1], ch * channel_mult[1], 3, stride=2, padding=1)
        
        ch2 = ch * channel_mult[1]
        self.enc3 = TimestepResBlock(ch2, ch * channel_mult[2], time_dim)
        
        # 瓶颈层
        ch3 = ch * channel_mult[2]
        self.mid1 = TimestepResBlock(ch3, ch3, time_dim)
        self.mid_attn = SimpleCrossAttention(ch3, context_dim)
        self.mid2 = TimestepResBlock(ch3, ch3, time_dim)
        
        # 简化的解码器：3级上采样（含跳跃连接）
        self.dec3 = TimestepResBlock(ch3 + ch3, ch2, time_dim)  # skip连接
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch2, ch2, 3, padding=1),
        )
        
        self.dec2 = TimestepResBlock(ch2 + ch2, ch1, time_dim)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch1, ch1, 3, padding=1),
        )
        
        self.dec1 = TimestepResBlock(ch1 + ch1, model_channels, time_dim)
        
        # 输出层
        self.norm_out = nn.GroupNorm(32, model_channels)
        self.conv_out = nn.Conv2d(model_channels, latent_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_t: 含噪潜在码, shape: (B, latent_ch, h, w)
            t: 时间步, shape: (B,)
            context: 条件特征, shape: (B, N_ctx, context_dim)
            
        Returns:
            噪声预测, shape: (B, latent_ch, h, w)
        """
        t_emb = self.time_mlp(t)  # shape: (B, time_dim)
        
        # 编码器
        h = self.conv_in(z_t)          # shape: (B, model_ch, h, w)
        s1 = self.enc1(h, t_emb)       # shape: (B, ch*m[0], h, w)
        h = self.down1(s1)             # shape: (B, ch*m[0], h/2, w/2)
        s2 = self.enc2(h, t_emb)       # shape: (B, ch*m[1], h/2, w/2)
        h = self.down2(s2)             # shape: (B, ch*m[1], h/4, w/4)
        s3 = self.enc3(h, t_emb)       # shape: (B, ch*m[2], h/4, w/4)
        
        # 瓶颈
        h = self.mid1(s3, t_emb)       # shape: (B, ch*m[2], h/4, w/4)
        h = self.mid_attn(h, context)  # 交叉注意力条件化
        h = self.mid2(h, t_emb)        # shape: (B, ch*m[2], h/4, w/4)
        
        # 解码器（含跳跃连接）
        h = self.dec3(torch.cat([h, s3], dim=1), t_emb)  # shape: (B, ch*m[1], h/4, w/4)
        h = self.up2(h)                                     # shape: (B, ch*m[1], h/2, w/2)
        h = self.dec2(torch.cat([h, s2], dim=1), t_emb)  # shape: (B, ch*m[0], h/2, w/2)
        h = self.up1(h)                                     # shape: (B, ch*m[0], h, w)
        h = self.dec1(torch.cat([h, s1], dim=1), t_emb)  # shape: (B, model_ch, h, w)
        
        # 输出
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)  # shape: (B, latent_ch, h, w)


class SinusoidalEmbedding(nn.Module):
    """正弦时间嵌入。"""
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepResBlock(nn.Module):
    """带时间步条件化的残差块。"""
    
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SimpleCrossAttention(nn.Module):
    """简化版交叉注意力，用于LDM的条件注入。"""
    
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(context_dim, channels, bias=False)
        self.to_v = nn.Linear(context_dim, channels, bias=False)
        self.to_out = nn.Linear(channels, channels)
    
    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 图像特征, shape: (B, C, H, W)
            context: 条件特征, shape: (B, N, context_dim) 或 None
        """
        if context is None:
            return x  # 无条件时直接返回
        
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # shape: (B, N_img, C)
        
        q = self.to_q(x)             # shape: (B, N_img, C)
        k = self.to_k(context)        # shape: (B, N_ctx, C)
        v = self.to_v(context)        # shape: (B, N_ctx, C)
        
        # 分头
        q = q.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        out = self.to_out(out)
        
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return residual + out


# ============================================================
# 6. 完整LDM
# ============================================================

class LatentDiffusionModel(nn.Module):
    """完整的潜在扩散模型。
    
    整合VAE和扩散U-Net，提供训练和推理接口。
    
    Args:
        vae: 预训练的AutoencoderKL
        unet: 扩散U-Net
        num_timesteps: 扩散步数
        scale_factor: VAE潜在空间缩放因子
    """
    
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: SimpleUNetForLatent,
        num_timesteps: int = 1000,
        scale_factor: float = 0.18215,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.num_timesteps = num_timesteps
        self.scale_factor = scale_factor
        
        # 冻结VAE参数
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # 注册噪声调度器参数
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码图像到潜在空间（无梯度）。
        
        Args:
            x: 图像, shape: (B, 3, H, W), 值域 [-1, 1]
            
        Returns:
            潜在码, shape: (B, latent_ch, H/f, W/f)
        """
        posterior = self.vae.encode(x)
        z = posterior.sample() * self.scale_factor
        return z
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码潜在码到图像（无梯度）。
        
        Args:
            z: 潜在码, shape: (B, latent_ch, h, w)
            
        Returns:
            图像, shape: (B, 3, H, W)
        """
        z = z / self.scale_factor
        return self.vae.decode(z)
    
    def q_sample(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向扩散：给潜在码添加噪声。
        
        z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * epsilon
        
        Args:
            z_0: 干净潜在码, shape: (B, C, h, w)
            t: 时间步, shape: (B,)
            noise: 可选的噪声, shape: (B, C, h, w)
            
        Returns:
            含噪潜在码, shape: (B, C, h, w)
        """
        if noise is None:
            noise = torch.randn_like(z_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alpha * z_0 + sqrt_one_minus_alpha * noise
    
    def training_loss(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算LDM训练损失。
        
        Args:
            x: 训练图像, shape: (B, 3, H, W)
            context: 条件特征, shape: (B, N, context_dim)
            
        Returns:
            MSE损失, shape: scalar
        """
        # 编码到潜在空间（无梯度）
        z_0 = self.encode(x)  # shape: (B, latent_ch, h, w)
        
        # 采样噪声和时间步
        B = z_0.shape[0]
        noise = torch.randn_like(z_0)
        t = torch.randint(0, self.num_timesteps, (B,), device=z_0.device)
        
        # 加噪
        z_t = self.q_sample(z_0, t, noise)  # shape: (B, latent_ch, h, w)
        
        # 预测噪声
        noise_pred = self.unet(z_t, t, context)  # shape: (B, latent_ch, h, w)
        
        # MSE损失
        loss = F.mse_loss(noise_pred, noise)
        return loss
    
    @torch.no_grad()
    def sample_ddim(
        self,
        shape: Tuple[int, ...],
        context: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """DDIM采样。
        
        Args:
            shape: 采样形状 (B, C, h, w)
            context: 条件特征
            num_steps: 采样步数
            eta: DDIM的随机性参数（0=确定性）
            guidance_scale: CFG引导强度
            
        Returns:
            生成的图像, shape: (B, 3, H, W)
        """
        device = self.betas.device
        B = shape[0]
        
        # 构建时间步序列
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps + 1, device=device
        ).long()
        
        # 初始噪声
        z = torch.randn(shape, device=device)
        
        for i in range(num_steps):
            t_cur = timesteps[i].expand(B)
            t_next = timesteps[i + 1].expand(B)
            
            # 预测噪声
            if guidance_scale > 1.0 and context is not None:
                # Classifier-Free Guidance
                noise_cond = self.unet(z, t_cur, context)
                noise_uncond = self.unet(z, t_cur, None)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = self.unet(z, t_cur, context)
            
            # DDIM更新
            alpha_cur = self.alphas_cumprod[t_cur[0]]
            alpha_next = self.alphas_cumprod[t_next[0]] if t_next[0] >= 0 else torch.tensor(1.0)
            
            # 预测 x_0
            z_0_pred = (z - (1 - alpha_cur).sqrt() * noise_pred) / alpha_cur.sqrt()
            
            # 方向指向 z_t
            dir_zt = (1 - alpha_next - eta**2 * (1 - alpha_next) / (1 - alpha_cur) * (1 - alpha_cur / alpha_next)).sqrt() * noise_pred
            
            # 随机噪声
            if eta > 0 and t_next[0] > 0:
                sigma = eta * ((1 - alpha_cur / alpha_next) * (1 - alpha_next) / (1 - alpha_cur)).sqrt()
                noise = torch.randn_like(z)
            else:
                sigma = 0
                noise = 0
            
            z = alpha_next.sqrt() * z_0_pred + dir_zt + sigma * noise
        
        # 解码到图像空间
        images = self.decode(z)
        return images


# ============================================================
# 7. 对比像素空间vs潜在空间开销
# ============================================================

def compare_pixel_vs_latent() -> None:
    """对比像素空间和潜在空间扩散的计算开销。"""
    print("=" * 70)
    print("像素空间 vs 潜在空间扩散模型 计算开销对比")
    print("=" * 70)
    
    configs = [
        ("像素空间 256x256", (2, 3, 256, 256), 256),
        ("像素空间 512x512", (2, 3, 512, 512), 512),
        ("潜在空间 32x32 (f=8, 256px)", (2, 4, 32, 32), 32),
        ("潜在空间 64x64 (f=8, 512px)", (2, 4, 64, 64), 64),
    ]
    
    print(f"\n{'配置':<35} | {'维度':>20} | {'元素数':>10} | {'相对大小':>10}")
    print("-" * 85)
    
    base_elements = 256 * 256 * 3
    for name, shape, _ in configs:
        elements = 1
        for s in shape[1:]:
            elements *= s
        relative = elements / base_elements
        dim_str = "x".join(str(s) for s in shape[1:])
        print(f"{name:<35} | {dim_str:>20} | {elements:>10,} | {relative:>9.2f}x")
    
    print("\n关键指标:")
    print(f"  256x256像素 -> 32x32潜在: 维度减少 {256*256*3 / (32*32*4):.0f}x")
    print(f"  512x512像素 -> 64x64潜在: 维度减少 {512*512*3 / (64*64*4):.0f}x")
    print(f"  注意力矩阵 (32x32 vs 256x256): {(256*256)**2 / (32*32)**2:.0f}x 差异")


# ============================================================
# 8. 使用预训练Stable Diffusion VAE的演示
# ============================================================

def demo_pretrained_vae() -> None:
    """演示使用diffusers库的预训练SD VAE。
    
    需要安装: pip install diffusers transformers accelerate
    """
    try:
        from diffusers import AutoencoderKL as DiffusersVAE
        import numpy as np
    except ImportError:
        print("需要安装 diffusers: pip install diffusers transformers accelerate")
        print("跳过预训练VAE演示。")
        return
    
    print("\n" + "=" * 70)
    print("使用预训练Stable Diffusion VAE")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载预训练VAE
    print("加载预训练VAE (stabilityai/sd-vae-ft-mse)...")
    vae = DiffusersVAE.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32
    ).to(device)
    
    # 创建测试图像（随机噪声图像）
    test_image = torch.randn(1, 3, 512, 512, device=device)
    test_image = test_image.clamp(-1, 1)
    
    # 编码
    with torch.no_grad():
        latent_dist = vae.encode(test_image)
        z = latent_dist.latent_dist.sample()
        z_scaled = z * 0.18215
    
    print(f"\n输入图像:    {test_image.shape} ({test_image.numel():,} 元素)")
    print(f"潜在码:      {z.shape} ({z.numel():,} 元素)")
    print(f"压缩比:      {test_image.numel() / z.numel():.1f}x")
    print(f"潜在码统计:  mean={z.mean():.4f}, std={z.std():.4f}")
    print(f"缩放后统计:  mean={z_scaled.mean():.4f}, std={z_scaled.std():.4f}")
    
    # 解码
    with torch.no_grad():
        reconstruction = vae.decode(z).sample
    
    # 重建质量
    mse = F.mse_loss(test_image, reconstruction)
    psnr = 10 * torch.log10(4.0 / mse)  # 值域[-1,1]，范围为4
    print(f"\n重建MSE:     {mse.item():.6f}")
    print(f"重建PSNR:    {psnr.item():.2f} dB")
    
    # 潜在空间插值演示
    print("\n潜在空间插值演示:")
    z_A = torch.randn(1, 4, 64, 64, device=device) * 0.18215
    z_B = torch.randn(1, 4, 64, 64, device=device) * 0.18215
    
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_interp = (1 - alpha) * z_A + alpha * z_B
        with torch.no_grad():
            img = vae.decode(z_interp / 0.18215).sample
        print(f"  alpha={alpha:.2f}: 图像范围 [{img.min():.2f}, {img.max():.2f}]")


# ============================================================
# 完整演示
# ============================================================

def demo_all() -> None:
    """运行所有演示。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("潜在扩散模型 (LDM) 完整演示")
    print("=" * 70)
    
    # --- 1. AutoencoderKL 测试 ---
    print("\n[1] AutoencoderKL 测试")
    vae = AutoencoderKL(
        in_channels=3,
        base_channels=64,   # 缩小版，用于测试
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=1,
        latent_channels=4,
    ).to(device)
    
    x = torch.randn(2, 3, 256, 256, device=device)
    reconstruction, posterior = vae(x)
    z = posterior.sample()
    kl = posterior.kl_divergence()
    
    print(f"  输入:     {x.shape}")
    print(f"  潜在码:   {z.shape}")
    print(f"  重建:     {reconstruction.shape}")
    print(f"  KL散度:   {kl.mean().item():.2f}")
    print(f"  压缩比:   {x.numel() / z.numel():.1f}x")
    
    vae_params = sum(p.numel() for p in vae.parameters())
    print(f"  VAE参数:  {vae_params:,} ({vae_params/1e6:.1f}M)")
    
    # --- 2. LDM完整训练步骤测试 ---
    print("\n[2] LDM训练步骤测试")
    unet = SimpleUNetForLatent(
        latent_channels=4,
        model_channels=128,
        channel_mult=(1, 2, 4),
        context_dim=768,
    ).to(device)
    
    ldm = LatentDiffusionModel(
        vae=vae,
        unet=unet,
        num_timesteps=1000,
    ).to(device)
    
    # 模拟训练步骤
    x_train = torch.randn(2, 3, 256, 256, device=device)
    context = torch.randn(2, 77, 768, device=device)  # CLIP文本特征
    
    loss = ldm.training_loss(x_train, context)
    print(f"  训练损失: {loss.item():.4f}")
    
    unet_params = sum(p.numel() for p in unet.parameters())
    print(f"  U-Net参数: {unet_params:,} ({unet_params/1e6:.1f}M)")
    
    # --- 3. 对比分析 ---
    print("\n[3] 像素空间 vs 潜在空间对比")
    compare_pixel_vs_latent()
    
    # --- 4. 预训练VAE演示 ---
    demo_pretrained_vae()


if __name__ == "__main__":
    demo_all()
```

---

## 本章小结

| 概念 | 核心要点 |
|------|---------|
| 像素空间瓶颈 | 高分辨率图像的计算/内存开销巨大，自然图像存在大量冗余 |
| LDM核心思想 | 先用VAE压缩到潜在空间，再在潜在空间中做扩散 |
| 感知压缩VAE | KL-reg VAE + LPIPS感知损失 + PatchGAN判别器 |
| 压缩因子 | $f=4\sim8$ 为最佳平衡点，SD 1.x用 $f=8$ |
| 潜在空间扩散 | 训练目标与像素扩散完全一致，只是操作空间不同 |
| 条件化 | 统一的Cross-Attention框架，支持文本/图像/分割图等任意模态 |
| 两阶段训练 | 先训练VAE → 冻结VAE → 训练U-Net |
| 推理管线 | 文本编码 → 潜在空间去噪 → VAE解码 |
| 计算节省 | 空间维度减少 $f^2$ 倍，注意力减少 $f^4$ 倍 |
| 内存优化 | Gradient Checkpointing, xFormers, FP16, 模型卸载 |

---

## 练习题

### 基础题

**练习1**：一张 $768 \times 768 \times 3$ 的图像，经过 $f=8$ 的VAE编码后，潜在码的形状是什么（假设潜在通道数 $c=4$）？计算压缩比（元素数比）。如果在此潜在空间上做 $16 \times 16$ 分辨率的自注意力，注意力矩阵的大小是多少？

**练习2**：解释为什么LDM中VAE的KL正则化权重 $\lambda_{KL}$ 设置得很小（约 $10^{-6}$），而不是像标准VAE那样设为1。从潜在空间的使用目的角度分析。

### 中级题

**练习3**：实现VAE的训练循环，包含以下损失项：
- L1重建损失
- KL散度正则化（带权重 $\lambda_{KL}$）
- 感知损失（使用VGG特征，可以用 `torchvision.models.vgg16` 的中间层特征）
写出完整的训练步骤代码。

**练习4**：实现球面线性插值（SLERP）函数，并对比SLERP与线性插值（LERP）在高维高斯分布上的行为差异。提示：采样两个高维高斯向量，计算插值路径上各点的范数变化。

### 提高题

**练习5**：设计并实现一个实验，比较以下三种LDM配置的训练效率-质量权衡：
- 配置A：$f=4$，$c=4$，潜在空间 $128 \times 128 \times 4$
- 配置B：$f=8$，$c=4$，潜在空间 $64 \times 64 \times 4$
- 配置C：$f=8$，$c=16$，潜在空间 $64 \times 64 \times 16$

对于每个配置，估算：(a) VAE重建质量，(b) 扩散模型训练的计算开销，(c) 潜在空间的信息容量。讨论哪个配置在什么场景下最优。

---

## 练习答案

### 练习1

输入：$768 \times 768 \times 3$，$f=8$，$c=4$

潜在码形状：$\frac{768}{8} \times \frac{768}{8} \times 4 = 96 \times 96 \times 4$

元素数比（压缩比）：

$$\frac{768 \times 768 \times 3}{96 \times 96 \times 4} = \frac{1,769,472}{36,864} = 48.0$$

即压缩了48倍。

如果在 $96 \times 96$ 的潜在空间上做自注意力（假设不进一步下采样到16x16，而是在某个U-Net层的16x16分辨率）：

在U-Net内部的 $16 \times 16$ 分辨率层，序列长度 $N = 16 \times 16 = 256$，注意力矩阵大小为 $256 \times 256 = 65,536$ 个元素。

如果直接在 $96 \times 96$ 做全局注意力，序列长度 $N = 96 \times 96 = 9,216$，注意力矩阵大小为 $9,216 \times 9,216 \approx 85M$ 个元素，内存约需 $85M \times 4B = 340MB$（单头），这在GPU上可行但开销较大。

### 练习2

标准VAE的KL权重为1，是为了让潜在空间尽可能接近标准正态分布，便于采样。但在LDM中：

1. **采样由扩散模型负责**：不需要VAE的潜在空间是"可采样的"。扩散模型从噪声开始逐步去噪，它会自己学习潜在空间的分布。

2. **重建质量优先**：VAE的首要任务是高质量压缩-解压缩。强KL正则化会强制潜在空间接近标准正态，牺牲重建质量。而低KL权重允许VAE使用更多的潜在空间容量来保留图像信息。

3. **轻度正则化的作用**：非零但很小的 $\lambda_{KL}$ 防止潜在空间"崩塌"（某些维度方差趋近于零）或"爆炸"（方差趋近于无穷），保持潜在空间的数值稳定性。

4. **经验发现**：Rombach et al. 发现 $\lambda_{KL} \approx 10^{-6}$ 时，VAE重建质量最好，同时潜在空间足够平滑使得扩散模型可以有效工作。

### 练习3

```python
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """基于VGG16的感知损失。"""
    
    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        # 使用VGG16的前16层（到relu3_3）
        self.feature_extractor = nn.Sequential(
            *list(vgg.features[:16])
        ).eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # VGG的归一化参数
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 将 [-1, 1] 范围转换为 VGG 期望的归一化范围
        x = (x + 1) / 2  # -> [0, 1]
        target = (target + 1) / 2
        
        x = (x - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        feat_x = self.feature_extractor(x)
        feat_target = self.feature_extractor(target)
        
        return F.mse_loss(feat_x, feat_target)


def train_vae_step(
    vae: AutoencoderKL,
    perceptual_loss_fn: VGGPerceptualLoss,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    lambda_kl: float = 1e-6,
    lambda_perc: float = 1.0,
) -> dict:
    """VAE单步训练。
    
    Returns:
        包含各项损失的字典
    """
    optimizer.zero_grad()
    
    # 前向传播
    reconstruction, posterior = vae(x, sample_posterior=True)
    
    # 重建损失 (L1)
    l_rec = F.l1_loss(reconstruction, x)
    
    # KL散度
    l_kl = posterior.kl_divergence().mean()
    
    # 感知损失
    l_perc = perceptual_loss_fn(reconstruction, x)
    
    # 总损失
    loss = l_rec + lambda_kl * l_kl + lambda_perc * l_perc
    
    loss.backward()
    optimizer.step()
    
    return {
        "loss": loss.item(),
        "l_rec": l_rec.item(),
        "l_kl": l_kl.item(),
        "l_perc": l_perc.item(),
    }
```

### 练习4

```python
def slerp(
    z_A: torch.Tensor,
    z_B: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """球面线性插值 (SLERP)。
    
    Args:
        z_A: 起始向量, shape: (B, ...)
        z_B: 终止向量, shape: (B, ...)
        alpha: 插值参数, 范围 [0, 1]
    
    Returns:
        插值结果, shape: (B, ...)
    """
    # 展平为2D
    shape = z_A.shape
    z_A_flat = z_A.reshape(shape[0], -1)  # (B, D)
    z_B_flat = z_B.reshape(shape[0], -1)  # (B, D)
    
    # 归一化
    z_A_norm = F.normalize(z_A_flat, dim=-1)
    z_B_norm = F.normalize(z_B_flat, dim=-1)
    
    # 计算角度
    cos_theta = (z_A_norm * z_B_norm).sum(dim=-1, keepdim=True)
    cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)
    
    # 保持原始范数
    norm_A = z_A_flat.norm(dim=-1, keepdim=True)
    norm_B = z_B_flat.norm(dim=-1, keepdim=True)
    norm_interp = (1 - alpha) * norm_A + alpha * norm_B
    
    # SLERP
    sin_theta = torch.sin(theta)
    # 处理theta接近0的情况（退化为LERP）
    safe_mask = sin_theta.abs() > 1e-6
    
    coeff_A = torch.where(
        safe_mask,
        torch.sin((1 - alpha) * theta) / sin_theta,
        torch.tensor(1 - alpha, device=z_A.device),
    )
    coeff_B = torch.where(
        safe_mask,
        torch.sin(alpha * theta) / sin_theta,
        torch.tensor(alpha, device=z_A.device),
    )
    
    result = coeff_A * z_A_flat + coeff_B * z_B_flat
    
    # 恢复范数
    result = F.normalize(result, dim=-1) * norm_interp
    
    return result.reshape(shape)


def compare_lerp_slerp() -> None:
    """对比LERP和SLERP在高维空间中的行为。"""
    dim = 64 * 64 * 4  # 模拟潜在空间维度
    z_A = torch.randn(1, dim)
    z_B = torch.randn(1, dim)
    
    print(f"维度: {dim}")
    print(f"z_A 范数: {z_A.norm().item():.2f}")
    print(f"z_B 范数: {z_B.norm().item():.2f}")
    print(f"期望范数 (高维高斯): {dim**0.5:.2f}")
    print()
    
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print(f"{'alpha':>6} | {'LERP范数':>10} | {'SLERP范数':>10} | {'差异%':>8}")
    print("-" * 45)
    
    for alpha in alphas:
        z_lerp = (1 - alpha) * z_A + alpha * z_B
        z_slerp = slerp(z_A, z_B, alpha)
        
        norm_lerp = z_lerp.norm().item()
        norm_slerp = z_slerp.norm().item()
        diff_pct = (norm_lerp - norm_slerp) / norm_slerp * 100
        
        print(f"{alpha:>6.1f} | {norm_lerp:>10.2f} | {norm_slerp:>10.2f} | {diff_pct:>7.1f}%")
    
    print("\n结论: LERP在alpha=0.5处范数下降最多（'范数凹陷'），")
    print("SLERP保持了更一致的范数，避免穿过低概率区域。")
```

### 练习5

**分析：**

**配置A**：$f=4$, $c=4$, 潜在空间 $128 \times 128 \times 4$

- *VAE重建质量*：极好。$f=4$ 的压缩很轻度，几乎无视觉损失。PSNR通常>35dB。
- *扩散训练开销*：中等偏高。$128 \times 128 = 16384$ 个空间位置，如果使用注意力（比如在 $32 \times 32$ 分辨率层），仍然可行。但整体计算量约为配置B的4倍。
- *信息容量*：$128 \times 128 \times 4 = 65536$ 元素。

**配置B**：$f=8$, $c=4$, 潜在空间 $64 \times 64 \times 4$

- *VAE重建质量*：好。部分高频细节有损失，PSNR约30-33dB。对于大多数应用足够。
- *扩散训练开销*：低。$64 \times 64 = 4096$ 个空间位置。这是Stable Diffusion 1.x的配置，在单张A100上可以合理训练。
- *信息容量*：$64 \times 64 \times 4 = 16384$ 元素。

**配置C**：$f=8$, $c=16$, 潜在空间 $64 \times 64 \times 16$

- *VAE重建质量*：好（同配置B的空间分辨率，但更多通道保留更多信息）。PSNR可能略优于配置B。
- *扩散训练开销*：中等。空间尺寸同B，但通道数4倍。U-Net的卷积参数量和计算量增加。
- *信息容量*：$64 \times 64 \times 16 = 65536$ 元素（与配置A相同）。

**推荐场景**：
- **注重生成质量**：配置A（$f=4$），VAE损失最小，代价是训练较慢
- **平衡效率和质量**：配置B（$f=8, c=4$），这也是实践中最常用的配置
- **需要更多语义信息**：配置C（$f=8, c=16$），适用于需要精细控制的场景（如视频生成中的时序一致性），SDXL的VAE就增大了通道数

---

## 延伸阅读

1. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B.** (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR. LDM/Stable Diffusion原始论文。

2. **Esser, P., Rombach, R., & Ommer, B.** (2021). *Taming Transformers for High-Resolution Image Synthesis*. CVPR. VQ-VAE用于图像压缩的关键工作。

3. **Kingma, D.P. & Welling, M.** (2014). *Auto-Encoding Variational Bayes*. ICLR. VAE的开创性论文。

4. **van den Oord, A., Vinyals, O., & Kavukcuoglu, K.** (2017). *Neural Discrete Representation Learning*. NeurIPS. VQ-VAE原始论文。

5. **Podell, D., et al.** (2023). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*. Stable Diffusion XL的技术报告。

6. **Zhang, L., Rao, A., & Agrawala, M.** (2023). *Adding Conditional Control to Text-to-Image Diffusion Models*. ICCV. ControlNet论文。

7. **Ho, J. & Salimans, T.** (2022). *Classifier-Free Diffusion Guidance*. NeurIPS Workshop. CFG的原始论文。

---

[上一章：第十七章 注意力机制在扩散模型中的应用](17-attention-in-diffusion.md) | [目录](../README.md) | [下一章：第十九章](../part7-advanced/19-next-topic.md)
