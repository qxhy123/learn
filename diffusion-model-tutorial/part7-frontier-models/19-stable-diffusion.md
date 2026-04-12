# 第十九章：Stable Diffusion原理与实现

> **本章导读**：Stable Diffusion（SD）是2022年由Stability AI开源的潜在扩散模型，它将文本到图像生成的能力从封闭实验室推向了每一位开发者。SD的核心创新在于将扩散过程从像素空间转移到VAE的潜在空间，结合CLIP文本编码器实现文本引导生成。本章将系统拆解SD的三大模块——CLIP文本编码器、VAE感知压缩器和U-Net去噪网络——深入理解推理管线的每一步，并通过代码实战掌握文本到图像、图像到图像等多种生成范式。

**前置知识**：潜在扩散模型（LDM）、Classifier-Free Guidance（CFG）、DDIM采样、VAE基础、注意力机制

**预计学习时间**：4-5小时

---

## 学习目标

完成本章学习后，你将能够：

1. 完整描述Stable Diffusion的三大模块（CLIP、VAE、U-Net）及其参数规模与数据流
2. 理解CLIP对比学习的训练目标及其文本编码器在SD中的具体作用
3. 掌握SD推理管线的全流程，包括CFG的批量处理技巧
4. 实现图像到图像（img2img）生成，理解噪声强度参数对输出的控制
5. 了解SD生态扩展（ControlNet、LoRA、SDXL）的核心思想与应用场景

---

## 19.1 Stable Diffusion的架构全景

### 19.1.1 SD的历史背景

2022年8月，Stability AI联合CompVis（慕尼黑大学）和Runway正式开源了Stable Diffusion v1，这是人工智能领域的一个里程碑事件。在此之前，DALL-E 2（OpenAI）和Imagen（Google）虽然展示了惊人的文本到图像能力，但均未开源。SD的开源彻底改变了AI生成图像的格局——任何人都可以在消费级GPU上运行高质量的图像生成。

Stable Diffusion的核心论文是Rombach et al. 2022的《High-Resolution Image Synthesis with Latent Diffusion Models》（即LDM论文），其关键洞察是：

> 将扩散过程从高维像素空间转移到低维潜在空间，可以在保持生成质量的同时大幅降低计算成本。

### 19.1.2 三大模块概览

SD由三个核心模块组成：

```
文本输入 ─→ [CLIP Text Encoder] ─→ 文本嵌入 h_text
                                         │
                                         ▼
随机噪声 z_T ─→ [U-Net (LDM)] ←─── Cross-Attention
                     │
                     ▼
                潜在表示 z_0 ─→ [VAE Decoder] ─→ 生成图像 x_0
```

| 模块 | 模型 | 参数量 | 功能 |
|------|------|--------|------|
| 文本编码器 | CLIP ViT-L/14 | ~123M | 将文本转换为语义嵌入 |
| 去噪网络 | U-Net (含Transformer) | ~860M | 在潜在空间预测噪声 |
| 图像编码/解码 | KL-VAE | ~84M | 像素空间 ↔ 潜在空间 |
| **总计** | | **~890M** | |

注意这里的参数统计是SD v1.5的数据。VAE的编码器（约34M）在推理时不一定使用（仅img2img需要），解码器约50M参数。

### 19.1.3 训练数据

SD v1.x的训练使用了LAION-Aesthetic数据集，这是从LAION-5B中筛选出的子集：

- **LAION-5B**：约58.5亿图文对，从Common Crawl爬取
- **LAION-Aesthetic**：使用美学评分模型过滤，保留评分 $\geq 5.0$ 的约6亿图文对
- **过滤策略**：去除NSFW内容、低分辨率（$< 256 \times 256$）图像、重复图像
- **训练流程**：先在LAION-2B（$\geq 256$分辨率）预训练，再在LAION-Aesthetic（$\geq 512$分辨率）微调

### 19.1.4 潜在空间的压缩比

SD使用的KL-VAE将图像压缩到潜在空间，压缩比为 $f=8$：

$$x \in \mathbb{R}^{H \times W \times 3} \xrightarrow{\text{Encoder}} z \in \mathbb{R}^{H/8 \times W/8 \times 4}$$

以512x512图像为例：

- 像素空间：$512 \times 512 \times 3 = 786,432$ 维
- 潜在空间：$64 \times 64 \times 4 = 16,384$ 维
- 压缩比：$786432 / 16384 = 48\times$

这种高压缩比是SD能在消费级GPU上运行的关键——扩散过程只需在16K维空间中进行，而非786K维。

---

## 19.2 CLIP文本编码器详解

### 19.2.1 CLIP对比预训练

CLIP（Contrastive Language-Image Pre-training，Radford et al. 2021）是OpenAI训练的多模态模型，通过对比学习将图像和文本映射到共享的嵌入空间。

训练目标：对于一个批次中的 $N$ 个图文对 $(t_i, v_i)$，最大化匹配对的相似度，最小化不匹配对的相似度：

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{\exp(\text{sim}(t_i, v_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(t_i, v_j)/\tau)} + \log\frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(v_i, t_j)/\tau)}\right]$$

其中 $\text{sim}(a, b) = \frac{a \cdot b}{\|a\|\|b\|}$ 为余弦相似度，$\tau$ 为可学习的温度参数。

### 19.2.2 文本编码器架构

SD v1.x使用CLIP ViT-L/14的文本编码器部分：

| 参数 | 值 |
|------|-----|
| Transformer层数 | 12 |
| 隐藏维度 | 768 |
| 注意力头数 | 12 |
| 最大序列长度 | 77 tokens |
| 词汇表大小 | 49408（BPE编码） |

**关键细节**：SD使用的是CLIP文本编码器的**最后一层hidden states**（而非最终的池化向量），输出形状为 $[B, 77, 768]$。这是因为生成任务需要序列级的细粒度语义信息，单个向量无法携带足够的空间和属性信息。

### 19.2.3 文本编码流程

```
输入文本: "a beautiful sunset over the ocean"
    │
    ▼
BPE Tokenize: [49406, 320, 4108, 12938, 962, 518, 4890, 49407, 0, 0, ..., 0]
    │           ↑SOT                                      ↑EOT   ↑padding
    ▼
Token Embedding + Positional Embedding
    │
    ▼
12层 Transformer (Causal Attention)
    │
    ▼
输出: h_text ∈ [1, 77, 768]   (所有token的hidden states)
```

**注意**：SD v1.x使用的CLIP文本编码器有77 token的限制。对于短文本，不足77 token的部分用padding填充（token id为0或49407后的填充）；对于长文本，超出部分会被截断。这一限制在SD 2.x中通过使用更大的CLIP模型（ViT-H/14, 1024维）得到部分缓解，在SDXL中通过双文本编码器进一步改善。

### 19.2.4 Negative Prompt机制

Negative Prompt是CFG（Classifier-Free Guidance）框架的自然延伸。回顾CFG公式：

$$\hat{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t, \varnothing) + w \cdot [\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \varnothing)]$$

其中 $\varnothing$ 是"无条件"嵌入。在SD中：

- **正常CFG**：$\varnothing$ 对应空文本 `""` 的CLIP编码
- **Negative Prompt**：将 $\varnothing$ 替换为负面提示词（如 `"blurry, low quality"`）的CLIP编码

$$\hat{\epsilon}_\theta(z_t, t, c_{pos}, c_{neg}) = \epsilon_\theta(z_t, t, c_{neg}) + w \cdot [\epsilon_\theta(z_t, t, c_{pos}) - \epsilon_\theta(z_t, t, c_{neg})]$$

这样，生成过程会同时向正面提示靠近、远离负面提示。

---

## 19.3 U-Net架构（SD版本）

### 19.3.1 整体结构

SD的U-Net基于DDPM的U-Net，但关键改进是在每个分辨率层级加入了**SpatialTransformer Block**以实现文本条件注入。完整结构：

```
输入: z_t ∈ [B, 4, 64, 64], t ∈ [B], h_text ∈ [B, 77, 768]

Encoder:
  [64x64] → ResBlock + SpatialTransformer × 2  (320 channels)
  ↓ Downsample
  [32x32] → ResBlock + SpatialTransformer × 2  (640 channels)
  ↓ Downsample
  [16x16] → ResBlock + SpatialTransformer × 2  (1280 channels)
  ↓ Downsample
  [8x8]   → ResBlock + SpatialTransformer × 2  (1280 channels)

Middle:
  [8x8]   → ResBlock + SpatialTransformer + ResBlock  (1280 channels)

Decoder (with skip connections):
  [8x8]   → ResBlock + SpatialTransformer × 3  (1280 channels)
  ↑ Upsample
  [16x16] → ResBlock + SpatialTransformer × 3  (1280 channels)
  ↑ Upsample
  [32x32] → ResBlock + SpatialTransformer × 3  (640 channels)
  ↑ Upsample
  [64x64] → ResBlock + SpatialTransformer × 3  (320 channels)

输出: ε_θ ∈ [B, 4, 64, 64]
```

### 19.3.2 SpatialTransformer Block

SpatialTransformer是SD中实现文本-图像交互的核心组件：

```
输入特征 x ∈ [B, C, H, W]
    │
    ▼
GroupNorm → Conv1x1 → Reshape to [B, H*W, C]
    │
    ▼
Self-Attention: Q=K=V=x    (图像内部的空间注意力)
    │
    ▼
Cross-Attention: Q=x, K=V=h_text   (文本→图像的条件注入)
    │
    ▼
Feed-Forward Network (GEGLU激活)
    │
    ▼
Reshape to [B, C, H, W] → Conv1x1
    │
    ▼
残差连接: output = input + projected_output
```

**Cross-Attention的数学形式**：

$$Q = W_Q \cdot x, \quad K = W_K \cdot h_{text}, \quad V = W_V \cdot h_{text}$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中 $x \in \mathbb{R}^{(H \cdot W) \times d}$（展平的空间特征），$h_{text} \in \mathbb{R}^{77 \times 768}$（文本嵌入），$d_k$ 是注意力头维度。这使得图像的每个空间位置都能关注文本的每个token。

### 19.3.3 时间步嵌入

时间步 $t$ 通过正弦位置编码+MLP映射为时间嵌入向量：

$$t_{emb} = \text{MLP}(\text{SinusoidalEmbed}(t)) \in \mathbb{R}^{1280}$$

时间嵌入通过以下方式注入ResBlock：

$$h = h + \text{Linear}(t_{emb})$$

即以加法方式融合到每个ResBlock的中间特征中。

### 19.3.4 参数配置细节

SD v1.5的U-Net关键配置：

```python
model_channels = 320                    # 基础通道数
channel_mult = (1, 2, 4, 4)            # 通道倍数: 320, 640, 1280, 1280
num_res_blocks = 2                      # 每个层级的ResBlock数
attention_resolutions = [4, 2, 1]       # 在哪些下采样率添加注意力 (对应8x8, 16x16, 32x32)
num_heads = 8                           # 注意力头数
context_dim = 768                       # CLIP文本嵌入维度
```

---

## 19.4 推理管线全流程

### 19.4.1 文本到图像（txt2img）完整流程

SD的推理过程可以分为4个清晰的步骤：

**步骤1：文本编码**

$$\text{prompt} \xrightarrow{\text{CLIP}} h_{text} \in \mathbb{R}^{[B, 77, 768]}$$

同时编码正面和负面提示：

$$h_{pos} = \text{CLIP}(\text{prompt}), \quad h_{neg} = \text{CLIP}(\text{negative\_prompt})$$

**步骤2：初始化噪声**

$$z_T \sim \mathcal{N}(0, I), \quad z_T \in \mathbb{R}^{[B, 4, 64, 64]}$$

对于512x512图像，潜在空间为 $64 \times 64$（因为VAE压缩比 $f=8$）。

**步骤3：DDIM去噪循环**

使用DDIM采样器进行50步（默认）去噪：

```python
for i, t in enumerate(reversed(timesteps)):  # t: 999, 979, 959, ..., 0
    # CFG: 同时预测条件和无条件噪声
    z_input = torch.cat([z_t, z_t])           # [2B, 4, 64, 64]
    h_input = torch.cat([h_neg, h_pos])       # [2B, 77, 768]
    
    noise_pred = unet(z_input, t, h_input)    # [2B, 4, 64, 64]
    noise_uncond, noise_cond = noise_pred.chunk(2)
    
    # CFG合并
    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    
    # DDIM更新
    z_t = ddim_step(z_t, noise_pred, t, t_prev)
```

**DDIM单步更新公式**：

$$z_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \underbrace{\frac{z_t - \sqrt{1-\bar{\alpha}_t} \cdot \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}}_{\text{预测的 } z_0} + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta$$

**步骤4：VAE解码**

$$z_0 \xrightarrow{\text{VAE Decoder}} x_0 \in \mathbb{R}^{[B, 3, 512, 512]}$$

注意解码前需要进行缩放：$z_0 = z_0 / 0.18215$（SD特有的缩放因子）。

### 19.4.2 CFG的批量处理技巧

在实际实现中，为了利用GPU并行性，条件和无条件预测是在同一个batch中完成的：

$$\text{batch\_input} = \begin{bmatrix} z_t \\ z_t \end{bmatrix} \in \mathbb{R}^{[2B, 4, 64, 64]}, \quad \text{cond\_input} = \begin{bmatrix} h_{neg} \\ h_{pos} \end{bmatrix} \in \mathbb{R}^{[2B, 77, 768]}$$

这样只需一次前向传播即可同时得到两个预测，然后通过chunk操作分开。

### 19.4.3 采样器选择

SD支持多种采样器，它们在速度和质量之间有不同的权衡：

| 采样器 | 步数 | 特点 |
|--------|------|------|
| DDIM | 20-50 | 确定性，可进行插值 |
| PLMS | 20-50 | 多步预测，更快收敛 |
| Euler | 20-30 | 简单高效 |
| Euler Ancestral | 20-30 | 随机性更强，多样性更好 |
| DPM-Solver++ | 15-25 | 高阶ODE求解器，快速且高质量 |
| UniPC | 10-20 | 统一预测-校正框架 |

---

## 19.5 图像到图像（img2img）

### 19.5.1 基本原理

img2img的核心思想：不从纯噪声 $z_T$ 出发，而是从已有图像的带噪版本出发。

**步骤1**：将输入图像编码到潜在空间

$$x_{input} \xrightarrow{\text{VAE Encoder}} z_0^{input} \in \mathbb{R}^{[B, 4, 64, 64]}$$

**步骤2**：根据强度参数 $s \in (0, 1]$ 选择起始时间步 $t^* = \lfloor s \cdot T \rfloor$，并添加噪声

$$z_{t^*} = \sqrt{\bar{\alpha}_{t^*}} \cdot z_0^{input} + \sqrt{1 - \bar{\alpha}_{t^*}} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**步骤3**：从 $t^*$ 开始去噪（而非从 $T$ 开始）

- $s = 1.0$：等价于txt2img（完全忽略输入图像）
- $s = 0.5$：保留输入图像的大致结构，改变细节
- $s = 0.2$：仅做微小修改（如调色、增加细节）

### 19.5.2 Inpainting

Inpainting是img2img的变体，允许用户指定mask区域进行局部编辑：

$$z_t^{final} = m \cdot z_t^{denoised} + (1-m) \cdot z_t^{noised\_input}$$

其中 $m$ 是二值mask（1表示需要重绘的区域），每一步去噪后都将非mask区域替换回原图的噪声版本，从而保持非编辑区域不变。

SD专用的inpainting模型（`sd-v1-5-inpainting`）进一步将mask和被遮罩的图像作为额外输入通道（共9通道输入：4+4+1）。

### 19.5.3 DDIM Inversion

DDIM Inversion允许从真实图像精确恢复其对应的噪声 $z_T$：

$$z_{t+1} = \sqrt{\bar{\alpha}_{t+1}} \cdot \hat{z}_0 + \sqrt{1 - \bar{\alpha}_{t+1}} \cdot \epsilon_\theta(z_t, t, c)$$

其中 $\hat{z}_0 = \frac{z_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$。

这在图像编辑中非常有用：先对原图做DDIM Inversion得到 $z_T$，然后用新的prompt从 $z_T$ 开始去噪，实现精确的文本引导编辑。

---

## 19.6 Stable Diffusion生态

### 19.6.1 ControlNet

ControlNet（Zhang et al. 2023）通过额外的控制信号实现精确的空间控制：

- **Canny边缘**：保持生成图像的边缘结构与参考一致
- **人体姿态（OpenPose）**：控制生成人物的姿势
- **深度图**：控制生成场景的空间深度关系
- **语义分割图**：按区域指定生成内容

ControlNet的架构巧妙地复制了U-Net编码器的权重作为"可训练副本"，通过zero convolution将控制信号注入到原始U-Net中：

$$y_c = \mathcal{F}(x; \Theta) + \mathcal{Z}(\mathcal{F}(x + \mathcal{Z}(c; \Theta_{z1}); \Theta_c); \Theta_{z2})$$

其中 $\mathcal{Z}$ 是zero convolution（初始权重为零的1x1卷积），保证训练开始时ControlNet不影响原始模型。

### 19.6.2 LoRA微调

LoRA（Low-Rank Adaptation，Hu et al. 2021）是一种参数高效的微调方法：

$$W' = W + \Delta W = W + BA$$

其中 $W \in \mathbb{R}^{d \times k}$ 是原始权重，$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d,k)$。

在SD中，LoRA通常应用于U-Net的Cross-Attention层（$W_Q, W_K, W_V, W_O$），文件大小一般为2-50MB（取决于rank $r$），远小于完整模型的2GB+。

### 19.6.3 SDXL

SDXL（Podell et al. 2023）是SD的重大升级：

| 特性 | SD v1.5 | SDXL |
|------|---------|------|
| U-Net参数 | ~860M | ~2.6B |
| 文本编码器 | CLIP ViT-L/14 | CLIP ViT-L/14 + OpenCLIP ViT-bigG/14 |
| 默认分辨率 | 512x512 | 1024x1024 |
| 生成流程 | 单阶段 | Base + Refiner两阶段 |
| 条件注入 | 文本 | 文本 + 尺寸条件 + 裁剪条件 |

SDXL的双文本编码器输出拼接为2048维的条件向量，提供更丰富的语义信息。

---

## 代码实战

### 实战1：使用diffusers库的完整SD推理

```python
"""
Stable Diffusion完整推理实战
包含txt2img、img2img、inpainting、手动CFG循环
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np

# ============================================================
# 1. 文本到图像生成 (txt2img)
# ============================================================

def txt2img_with_diffusers(
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    seed: int = 42,
) -> "PIL.Image":
    """使用diffusers库进行文本到图像生成"""
    from diffusers import StableDiffusionPipeline
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to("cuda")
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    
    return image


# ============================================================
# 2. 图像到图像生成 (img2img)
# ============================================================

def img2img_with_diffusers(
    prompt: str,
    init_image: "PIL.Image",
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
) -> "PIL.Image":
    """使用diffusers库进行图像到图像生成"""
    from diffusers import StableDiffusionImg2ImgPipeline
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to("cuda")
    
    # strength控制去噪起始点: 1.0=完全重新生成, 0.0=不改变
    image = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]
    
    return image


# ============================================================
# 3. Inpainting
# ============================================================

def inpainting_with_diffusers(
    prompt: str,
    init_image: "PIL.Image",
    mask_image: "PIL.Image",  # 白色区域表示需要重绘
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
) -> "PIL.Image":
    """使用diffusers库进行图像修复"""
    from diffusers import StableDiffusionInpaintPipeline
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to("cuda")
    
    image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]
    
    return image


# ============================================================
# 4. 手动实现CFG采样循环（无封装）
# ============================================================

@torch.no_grad()
def manual_cfg_sampling(
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    seed: int = 42,
    device: str = "cuda",
) -> torch.Tensor:
    """
    手动实现Stable Diffusion的CFG采样循环
    不依赖Pipeline封装，展示底层数据流
    """
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
    
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # 加载各组件
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16
    ).to(device)
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # ---------- 步骤1: 文本编码 ----------
    # 编码正面提示
    pos_tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)                         # [1, 77]
    pos_embeds = text_encoder(pos_tokens)[0]       # [1, 77, 768]
    
    # 编码负面提示（空文本作为无条件）
    neg_tokens = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)                         # [1, 77]
    neg_embeds = text_encoder(neg_tokens)[0]       # [1, 77, 768]
    
    # 拼接为单个batch以并行处理
    text_embeds = torch.cat([neg_embeds, pos_embeds])  # [2, 77, 768]
    
    # ---------- 步骤2: 初始化噪声 ----------
    latent_h = height // 8   # 64
    latent_w = width // 8    # 64
    
    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn(
        (1, 4, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=torch.float16,
    )  # [1, 4, 64, 64]
    
    # 设置采样时间步
    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents * scheduler.init_noise_sigma
    
    # ---------- 步骤3: DDIM去噪循环 ----------
    intermediate_latents: List[torch.Tensor] = []
    
    for i, t in enumerate(scheduler.timesteps):
        # CFG: 将latent复制一份，分别用于无条件和有条件预测
        latent_input = torch.cat([latents, latents])  # [2, 4, 64, 64]
        latent_input = scheduler.scale_model_input(latent_input, t)
        
        # U-Net前向传播
        noise_pred = unet(
            latent_input,
            t,
            encoder_hidden_states=text_embeds,
        ).sample  # [2, 4, 64, 64]
        
        # 分离无条件和有条件预测
        noise_uncond, noise_cond = noise_pred.chunk(2)
        
        # CFG合并
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        # DDIM更新
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # 保存中间结果（每10步一次）
        if i % 10 == 0:
            intermediate_latents.append(latents.clone())
    
    # ---------- 步骤4: VAE解码 ----------
    latents = latents / 0.18215  # SD的缩放因子
    image = vae.decode(latents).sample  # [1, 3, 512, 512]
    image = (image / 2 + 0.5).clamp(0, 1)  # 归一化到[0, 1]
    
    return image, intermediate_latents


# ============================================================
# 5. 可视化中间去噪步骤
# ============================================================

def visualize_denoising_steps(
    intermediate_latents: List[torch.Tensor],
    vae: "AutoencoderKL",
) -> None:
    """可视化去噪过程的中间结果"""
    import matplotlib.pyplot as plt
    
    n_steps = len(intermediate_latents)
    fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))
    
    for idx, latent in enumerate(intermediate_latents):
        # VAE解码中间latent
        with torch.no_grad():
            decoded = vae.decode(latent / 0.18215).sample  # [1, 3, H, W]
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
        
        # 转换为numpy用于显示
        img_np = decoded[0].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
        
        axes[idx].imshow(img_np)
        axes[idx].set_title(f"Step {idx * 10}")
        axes[idx].axis("off")
    
    plt.suptitle("Denoising Process Visualization", fontsize=14)
    plt.tight_layout()
    plt.savefig("denoising_steps.png", dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# 6. LoRA权重加载演示
# ============================================================

def load_lora_weights(
    pipe: "StableDiffusionPipeline",
    lora_path: str,
    lora_scale: float = 1.0,
) -> "StableDiffusionPipeline":
    """
    加载LoRA权重到SD管线
    
    Args:
        pipe: 已加载的StableDiffusionPipeline
        lora_path: LoRA权重文件路径或HuggingFace模型ID
        lora_scale: LoRA强度缩放因子 (0.0-1.0)
    """
    # 方法1: 使用diffusers内置支持
    pipe.load_lora_weights(lora_path)
    
    # 调整LoRA强度（可在推理时动态调整）
    pipe.fuse_lora(lora_scale=lora_scale)
    
    return pipe


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 文本到图像
    image = txt2img_with_diffusers(
        prompt="a majestic castle on a floating island, sunset, fantasy art",
        negative_prompt="blurry, low quality, distorted",
        guidance_scale=7.5,
        num_inference_steps=50,
    )
    image.save("txt2img_result.png")
    
    # 手动CFG采样
    result, intermediates = manual_cfg_sampling(
        prompt="a serene mountain lake at dawn, photorealistic",
        negative_prompt="cartoon, painting, illustration",
        guidance_scale=7.5,
        num_inference_steps=50,
    )
    print(f"生成图像形状: {result.shape}")  # [1, 3, 512, 512]
    print(f"中间步骤数量: {len(intermediates)}")
```

---

## 本章小结

| 概念 | 核心要点 |
|------|---------|
| SD架构 | CLIP文本编码器 + KL-VAE + U-Net，共~890M参数 |
| CLIP编码 | 输出 $[B, 77, 768]$ 的token级嵌入序列（非单一向量） |
| U-Net改进 | 加入SpatialTransformer实现Cross-Attention文本注入 |
| 潜在空间 | $f=8$ 压缩比，512x512 → 64x64x4，48倍数据压缩 |
| CFG实现 | 条件/无条件batch并行处理，guidance_scale控制强度 |
| img2img | 从中间时间步开始去噪，strength参数控制保留程度 |
| Negative Prompt | 替换CFG中的无条件嵌入为负面语义嵌入 |
| ControlNet | zero convolution注入控制信号，精确空间控制 |
| LoRA | 低秩适配，2-50MB实现风格迁移 |
| SDXL | 2.6B U-Net + 双文本编码器 + 两阶段生成 |

---

## 练习题

### 基础题

**练习1**：解释为什么SD使用CLIP文本编码器的最后一层hidden states而非池化向量。如果使用池化后的单一向量 $\in \mathbb{R}^{768}$，对生成质量会有什么影响？

**练习2**：假设我们要生成768x768的图像（而非512x512），请计算：
- (a) 对应的潜在空间维度
- (b) Self-Attention的序列长度变化（相对于512x512）
- (c) 内存需求的大致增长比例

### 中级题

**练习3**：在img2img中，如果strength=0.6，总步数为50步，请回答：
- (a) 实际执行多少步去噪？
- (b) 起始时间步 $t^*$ 大约是多少？
- (c) 为什么strength=0时不等于"不改变图像"（提示：考虑VAE的重建误差）

**练习4**：修改手动CFG采样代码，实现"Prompt Weighting"功能：对于输入 `"a (beautiful:1.5) sunset"`，将"beautiful"对应token的嵌入向量乘以1.5。讨论这与直接提高guidance_scale的区别。

### 提高题

**练习5**：设计一个实验，比较以下三种文本条件注入方式在SD U-Net中的效果差异：
- (a) Cross-Attention（SD当前方式）
- (b) 将文本嵌入加到时间步嵌入上（additive）
- (c) 将文本嵌入通过FiLM层调制特征（$\gamma \cdot x + \beta$）

请从理论角度分析各方式的表达能力，并设计合理的消融实验方案。

---

## 练习答案

### 练习1

CLIP文本编码器的输出有两种形式：
- **池化向量**：$\in \mathbb{R}^{768}$，是对整个文本语义的压缩表示
- **Hidden states**：$\in \mathbb{R}^{77 \times 768}$，保留了每个token的独立语义信息

SD选择使用hidden states的原因：

1. **空间对应性**：Cross-Attention机制需要图像的每个空间位置与文本的每个token进行注意力交互。如果只有一个向量，就无法建立这种细粒度的空间-语义对应关系。例如，"a red car and a blue house"需要不同的空间区域关注"red car"和"blue house"。

2. **信息瓶颈**：768维向量无法编码复杂场景中的所有对象、属性、空间关系。77x768的序列提供了$\sim$60倍的信息容量。

3. **实验证据**：Imagen论文（Saharia et al. 2022）的消融实验明确显示，使用token序列的生成质量远优于使用池化向量。

如果使用单一池化向量，可能通过additive或FiLM方式注入，但会导致：
- 无法精确控制多物体场景中各物体的属性
- 颜色绑定（color binding）问题更严重
- 复杂构图能力显著下降

### 练习2

768x768图像的计算：

**(a) 潜在空间维度**：
$$\frac{768}{8} \times \frac{768}{8} \times 4 = 96 \times 96 \times 4 = 36,864 \text{ 维}$$

**(b) Self-Attention序列长度变化**：
- 512x512：序列长度 $= 64 \times 64 = 4096$
- 768x768：序列长度 $= 96 \times 96 = 9216$
- 变化比例：$9216 / 4096 = 2.25\times$

**(c) 内存增长**：
Self-Attention的复杂度为 $O(N^2)$，因此注意力层的内存增长为 $2.25^2 = 5.0625\times$。
但U-Net中还有卷积层（$O(N)$复杂度），综合来看整体内存增长约为 $2\text{-}3\times$。

### 练习3

strength=0.6，总步数50步：

**(a) 实际执行步数**：
$$\text{实际步数} = \lfloor 0.6 \times 50 \rfloor = 30 \text{ 步}$$

**(b) 起始时间步**：
DDIM将1000个时间步均匀划分为50步，因此时间步间隔为20。
从第20步（$50 - 30 = 20$）开始，对应时间步 $t^* \approx 20 \times 20 = 400$（约为总时间的40%处开始去噪）。

更精确地说，scheduler会选择从时间步序列的第 $(1 - 0.6) \times 50 = 20$ 个位置开始。

**(c) strength=0为什么不等于不改变**：
即使strength=0（不执行任何去噪步骤），VAE的编码-解码过程也会引入重建误差：

$$x \xrightarrow{\text{Encode}} z_0 \xrightarrow{\text{Decode}} \hat{x} \neq x$$

VAE不是完美的自编码器，存在信息损失。实际测试中，SD的VAE重建PSNR约为27-30dB，人眼可以注意到细微差异（尤其在文字、纹理区域）。

### 练习4

Prompt Weighting实现思路：

```python
def apply_prompt_weights(
    prompt: str,
    tokenizer: "CLIPTokenizer",
    text_encoder: "CLIPTextModel",
    device: str = "cuda",
) -> torch.Tensor:
    """
    解析加权prompt并调整token嵌入
    格式: "a (beautiful:1.5) sunset" -> 将beautiful的嵌入乘以1.5
    """
    import re
    
    # 解析权重标记
    pattern = r'\((\w+):([\d.]+)\)'
    weights = {}
    clean_prompt = prompt
    for match in re.finditer(pattern, prompt):
        word, weight = match.group(1), float(match.group(2))
        weights[word] = weight
        clean_prompt = clean_prompt.replace(match.group(0), word)
    
    # Tokenize
    tokens = tokenizer(
        clean_prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    
    # 获取token嵌入
    embeds = text_encoder(tokens.input_ids)[0]  # [1, 77, 768]
    
    # 找到目标token并应用权重
    token_ids = tokens.input_ids[0].tolist()
    for word, weight in weights.items():
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        for i in range(len(token_ids) - len(word_tokens) + 1):
            if token_ids[i:i+len(word_tokens)] == word_tokens:
                embeds[0, i:i+len(word_tokens)] *= weight
    
    return embeds
```

**与提高guidance_scale的区别**：
- **Prompt Weighting**：选择性地放大特定概念的语义强度，不影响其他概念
- **提高guidance_scale**：全局增强所有文本条件的影响，可能导致过度饱和和伪影
- Prompt Weighting更精细，guidance_scale更全局

### 练习5

三种文本条件注入方式的分析：

**(a) Cross-Attention（SD当前方式）**：
- **表达能力**：最强。每个空间位置独立地与所有文本token交互，实现了 $O(HW \times L)$ 的交互容量（$L$为文本序列长度）
- **优势**：天然支持多物体场景的空间-语义绑定
- **劣势**：计算开销最大

**(b) Additive（加到时间步嵌入上）**：
- **表达能力**：最弱。文本信息被压缩为单一向量，对所有空间位置施加相同的偏置
- **优势**：计算开销最小
- **劣势**：无法区分不同空间位置应该生成什么

**(c) FiLM调制**：
- **表达能力**：中等。$\gamma, \beta$ 按通道调制，可以控制哪些特征通道被激活
- **优势**：计算高效，比additive表达能力更强
- **劣势**：仍然是全局调制，空间选择性不足

**消融实验方案**：
1. 固定VAE和训练数据，仅替换U-Net中的条件注入模块
2. 评价指标：FID（生成质量）、CLIP Score（文本-图像一致性）、DrawBench（复杂场景测试）
3. 重点测试场景：(i) 单物体属性描述，(ii) 多物体空间关系，(iii) 颜色绑定测试
4. 预期结果：Cross-Attention在(ii)(iii)显著优于其他两种方式

---

## 延伸阅读

1. Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*.
2. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021*. (CLIP论文)
3. Zhang, L., et al. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models." *ICCV 2023*. (ControlNet)
4. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
5. Podell, D., et al. (2023). "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis." *ICLR 2024*.
6. Song, J., Meng, C., & Ermon, S. (2021). "Denoising Diffusion Implicit Models." *ICLR 2021*. (DDIM)
7. Lu, C., et al. (2022). "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling." *NeurIPS 2022*.

---

<div align="center">

[⬅️ 第十八章：条件生成与Classifier-Free Guidance](../part6-conditional-generation/18-cfg-conditional-generation.md) | [📖 目录](../README.md) | [第二十章：DALL-E 2与层次式生成 ➡️](20-dalle2-clip-conditioning.md)

</div>
