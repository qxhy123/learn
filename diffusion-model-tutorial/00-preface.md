# 前言：扩散模型的崛起与本教程使用指南

---

## 一场悄然发生的革命

2020年6月，Jonathan Ho等人在NeurIPS上发表了《Denoising Diffusion Probabilistic Models》（DDPM），这篇论文在当时并未立即引爆社区——毕竟GAN在图像生成领域已独霸多年。然而，仅仅两年后，Stable Diffusion、DALL-E 2、Midjourney相继问世，"文字生成图像"从科幻走进了每个人的日常。

扩散模型（Diffusion Models）成为了这场革命的核心引擎。

**为什么是扩散模型，而不是GAN？**

GAN曾经是生成模型的王者，但它有一个根本性的缺陷：训练极不稳定。生成器和判别器之间的博弈经常陷入模式崩塌（mode collapse），生成的图像多样性严重不足。研究者们花费数年时间开发各种稳定训练的技巧，却始终无法根治这一顽疾。

扩散模型的出现提供了一条截然不同的路径：

- **训练稳定**：目标是预测噪声，损失函数简单明确，无对抗训练
- **生成多样**：从高斯噪声出发，随机采样保证了生成多样性
- **质量卓越**：通过足够的去噪步骤，可以生成极高质量的样本
- **易于条件化**：Classifier-Free Guidance让条件生成变得优雅而强大

这些优势使扩散模型迅速席卷了图像、音频、视频、蛋白质结构预测等几乎所有生成任务。

---

## 扩散模型的发展脉络

理解历史有助于理解现在。让我们简要回顾扩散模型的发展：

### 2015：热力学的启示

Sohl-Dickstein等人发表《Deep Unsupervised Learning using Nonequilibrium Thermodynamics》，首次将热力学扩散过程引入生成模型。正向过程缓慢地将数据"破坏"为噪声，逆向过程学习"修复"噪声。这个想法超前于时代，在当时并未引起广泛关注。

### 2019：分数匹配的崛起

Song & Ermon发表《Generative Modeling by Estimating Gradients of the Data Distribution》，提出用神经网络估计数据分布的**分数函数**（梯度），并通过朗之万动力学采样。多尺度噪声分数网络（NCSN）的提出为扩散模型的统一框架铺垫了基础。

### 2020：DDPM的突破

Ho等人的DDPM论文将扩散模型与分数匹配统一，提出了简洁的噪声预测目标 $\mathbb{E}[||\epsilon - \epsilon_\theta(x_t, t)||^2]$，在CIFAR-10上实现了当时的最佳FID。更重要的是，DDPM的实现足够简单，社区可以快速复现和改进。

### 2021：加速与条件化

- **DDIM**（Song等）：非马尔科夫采样，将1000步压缩到50步，质量几乎不损失
- **ADM**（Dhariwal & Nichol）：改进U-Net架构，分类器引导，超越GAN的FID
- **分数匹配与SDE的统一**（Song等）：将扩散模型纳入随机微分方程框架

### 2021-2022：走向实用

- **Latent Diffusion Model**（Rombach等）：在压缩潜在空间中扩散，计算效率提升数十倍
- **GLIDE**（OpenAI）：CLIP引导的文本到图像生成
- **DALL-E 2**（OpenAI）：层次式生成，unCLIP框架
- **Imagen**（Google Brain）：T5编码器 + 级联超分辨率
- **Stable Diffusion**（CompVis/Stability AI）：开源LDM，掀起社区狂潮

### 2023-2024：前沿探索

- **一致性模型**（Song等）：少步甚至单步生成
- **DiT**（Peebles & Xie）：用Transformer取代U-Net，更好的Scaling
- **Flow Matching**（Lipman等, Liu等）：直线轨迹，更简单的ODE
- **Stable Diffusion 3 / FLUX**：Flow Matching + 多模态DiT
- **Sora**（OpenAI）：Spacetime Patches + DiT，视频生成新范式

---

## 本教程的定位

本教程的目标读者是：**有深度学习基础、希望系统掌握扩散模型原理与实践的学习者**。

我们相信，真正的理解来自三个层面的融合：

1. **数学推导**：不只是"知道公式"，而是能推导每一步
2. **代码实现**：不只是"会调用API"，而是能从零写出每个组件
3. **直觉理解**：在数学和代码之间，能用语言描述"为什么"

因此，本教程拒绝以下两种极端：
- 纯数学教程：公式堆砌，无法动手
- 纯应用教程：只调用`pipeline(prompt)`，不知其所以然

我们的路径是：**直觉 → 数学 → 代码 → 实验 → 反思**。

---

## 教程结构说明

### 第一部分：数学基础（第1-3章）

这部分是"地基"。扩散模型深度依赖概率论和变分推断。如果你对这些内容已经熟悉，可以快速浏览或直接跳过。但如果你想真正理解DDPM的ELBO推导，这部分不可或缺。

### 第二部分：生成模型概览（第4-6章）

了解扩散模型在生成模型家族中的位置。我们会快速回顾VAE和GAN，介绍分数匹配，并从SDE视角统一理解扩散过程。这部分建立了更广阔的视野。

### 第三部分：DDPM核心原理（第7-9章）

这是本教程的核心。我们会一步步推导：
- 正向加噪过程的闭合形式
- 逆向过程的ELBO
- 简化训练目标
- 采样算法

读完这三章，你就能从零实现DDPM。

### 第四部分：采样与加速（第10-12章）

DDPM需要1000步采样太慢了。这部分介绍如何在保持质量的前提下大幅减少采样步数，这是让扩散模型走向实用的关键。

### 第五部分：条件生成（第13-15章）

"生成一只橘猫坐在书桌上"——条件生成让用户掌控内容。从分类器引导到无分类器引导（CFG），再到文本条件化，这部分揭示了现代文生图系统的核心机制。

### 第六部分：架构设计（第16-18章）

U-Net、注意力机制、潜在扩散模型——这是让扩散模型在高分辨率图像上高效运行的工程基础。理解Stable Diffusion必须先理解LDM。

### 第七部分：前沿模型（第19-22章）

Stable Diffusion、DALL-E 2、DiT、Flow Matching——这部分覆盖2022-2024年的主要突破。我们会解析每个模型的创新点和工程实现。

### 第八部分：工程实践（第23-24章）

将知识转化为产品。从推理优化到完整文生图服务，这部分帮助你将扩散模型部署到实际应用中。

---

## 如何使用本教程

### 对于初学者

1. 按顺序阅读，不要跳跃
2. 每章的代码实战必须动手运行，不要只看
3. 基础练习题是检验理解的最低标准
4. 遇到不懂的数学，先接受结论继续学，之后再回来推导

### 对于有经验的读者

1. 可以根据学习路径建议跳读
2. 重点关注与已有知识的**差异和联系**
3. 提高题往往涉及论文复现，挑战性更强
4. 代码实战中，尝试在动手前预测输出形状

### 数学符号约定

| 符号 | 含义 | 说明 |
|------|------|------|
| $x_0$ | 原始数据 | 无噪声的真实数据 |
| $x_t$ | 第t步的加噪数据 | $t \in \{1, ..., T\}$ |
| $\epsilon$ | 高斯噪声 | $\epsilon \sim \mathcal{N}(0, I)$ |
| $\beta_t$ | 噪声调度 | 控制每步加噪量 |
| $\alpha_t$ | $1 - \beta_t$ | 信号保留比例 |
| $\bar{\alpha}_t$ | $\prod_{s=1}^t \alpha_s$ | 累积信号保留 |
| $\epsilon_\theta$ | 噪声预测网络 | 以$\theta$为参数 |
| $q$ | 正向过程分布 | 固定，无参数 |
| $p_\theta$ | 逆向过程分布 | 可学习 |

### 代码约定

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

# 张量形状注释格式：# shape: (B, C, H, W)
# B: batch_size, C: channels, H: height, W: width
# T: time_steps, D: dim/embedding_dim

# 类名：大驼峰
class DiffusionModel(nn.Module): ...

# 函数名：小写下划线
def compute_snr(alphas_cumprod: torch.Tensor) -> torch.Tensor: ...

# 常量：全大写
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
```

---

## 环境准备

### 硬件建议

- **最低要求**：CPU（第1-9章可完整运行）
- **推荐**：GPU（>=8GB显存）用于图像生成实验
- **理想**：GPU（>=16GB显存）用于第24章完整项目

### 安装

```bash
# 1. 创建虚拟环境
python -m venv diffusion-env
source diffusion-env/bin/activate

# 2. 安装PyTorch（选择适合你的CUDA版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装扩散模型相关库
pip install diffusers transformers accelerate

# 4. 安装工具库
pip install numpy matplotlib seaborn einops tqdm
pip install torchmetrics  # 用于FID等评估指标

# 5. 验证
python -c "
import torch, diffusers
print(f'PyTorch: {torch.__version__}')
print(f'Diffusers: {diffusers.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

---

## 与本系列其他教程的关系

```
线性代数教程 ──┐
概率统计教程 ──┤──→ 扩散模型教程（本教程）
微积分教程   ──┘         │
                    ┌────┼────┐
                    ↓    ↓    ↓
                 图像  音频  视频
                 生成  生成  生成
```

如果对本系列其他教程感兴趣，可参考项目根目录的相关链接。

---

## 致谢

本教程的内容基于以下核心工作：

- **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- **DDIM**: Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021)
- **Score SDE**: Song et al., "Score-Based Generative Modeling through SDEs" (ICLR 2021)
- **LDM**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022)
- **CFG**: Ho & Salimans, "Classifier-Free Diffusion Guidance" (NeurIPS Workshop 2021)
- **DiT**: Peebles & Xie, "Scalable Diffusion Models with Transformers" (ICCV 2023)
- **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)

感谢所有开源社区的贡献者，尤其是HuggingFace Diffusers团队。

---

*准备好了吗？让我们从概率论出发，一步步构建对扩散模型的完整理解。*

[开始第一章：概率论基础与随机变量 →](./part1-math-foundations/01-probability-random-variables.md)

[返回目录](./README.md)
