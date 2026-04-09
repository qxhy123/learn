# 第21章：多模态Transformer

> 语言是思维的符号，图像是世界的投影。当Transformer同时学会阅读文字和观察图像，人工智能便迈向了真正理解世界的第一步。

---

## 学习目标

完成本章学习后，你将能够：

1. **理解Vision Transformer（ViT）的原理**：掌握如何将图像分块后送入标准Transformer编码器，以及[CLS] Token在分类任务中的作用
2. **掌握图像Patch Embedding的实现**：理解将 $H \times W \times C$ 图像切分为 $N$ 个 $P \times P$ 的patch并线性投影到嵌入空间的完整流程
3. **理解CLIP的对比学习方法**：掌握双塔架构、InfoNCE损失函数，以及Zero-shot迁移的工作原理
4. **了解多模态融合策略**：区分Early Fusion与Late Fusion，理解Cross-Attention在Flamingo和LLaVA等架构中的应用
5. **能够实现简单的视觉Transformer**：从零用PyTorch搭建PatchEmbedding、ViT编码器，以及CLIP风格的对比学习训练循环

---

## 21.1 Vision Transformer（ViT）

### 21.1.1 从CNN到ViT

在ViT出现之前，计算机视觉领域由卷积神经网络（CNN）主导长达十年。CNN的设计哲学源于视觉信号的局部性——相邻像素之间关系紧密，远距离像素关系稀疏，因此用局部感受野的卷积核逐层提取特征是合理的归纳偏置（Inductive Bias）。

然而，CNN存在两个根本性局限：

- **全局建模能力有限**：CNN依赖堆叠多层来扩大感受野，远距离依赖关系的建模效率低
- **归纳偏置过强**：平移等变性假设并非对所有视觉任务都成立，限制了模型的表达灵活性

2020年，Google Brain团队提出了 **An Image is Worth 16x16 Words**（Dosovitskiy et al., 2020），证明：只要训练数据足够大，去掉CNN的归纳偏置，纯Transformer架构可以在图像分类任务上超越最先进的CNN。

这一发现的核心思想极为简洁：**把图像当成一段"视觉词序列"输入标准Transformer**。

```
CNN路线：                    ViT路线：
Image                        Image
  │                            │
卷积层（局部特征）             切块（Patch）
  │                            │
池化层（降采样）              线性投影（Embedding）
  │                            │
全连接层（分类头）            Transformer编码器
  │                            │
Logits                       [CLS] → 分类头 → Logits
```

### 21.1.2 图像分块（Patch Embedding）

ViT的核心预处理步骤是将连续的图像切分为固定大小的非重叠块（patch）。

设输入图像的尺寸为 $H \times W$，通道数为 $C$，patch大小为 $P \times P$，则：

$$N = \frac{H \times W}{P^2}$$

其中 $N$ 是patch的总数量，也是Transformer接收的序列长度。每个patch是一个 $P \times P \times C$ 的三维张量，将其展平后得到维度为 $P^2 C$ 的向量，再经过一个线性投影矩阵 $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$ 映射到模型维度 $D$：

$$\mathbf{z}_i = \text{Flatten}(\text{patch}_i) \cdot \mathbf{E}, \quad i = 1, \ldots, N$$

以ViT-B/16为例：输入图像 $224 \times 224 \times 3$，patch大小 $16 \times 16$，则：
- $N = 224^2 / 16^2 = 196$ 个patch
- 每个patch展平后维度 $16 \times 16 \times 3 = 768$
- 投影后维度 $D = 768$

这196个向量构成一个长度为196的序列，与NLP中的token序列在结构上完全等价。

### 21.1.3 位置编码

由于Transformer本身不感知序列的位置顺序，ViT同样需要加入位置信息。ViT原论文测试了三种方案：

| 方案 | 描述 | 效果 |
|:-----|:-----|:-----|
| 无位置编码 | 纯无序集合输入 | 准确率显著下降 |
| 1D可学习位置编码 | 每个patch分配一个可学习向量 | 与2D相当，原论文默认 |
| 2D可学习位置编码 | 按行列分别编码再相加 | 略有提升但差距小 |
| 相对位置编码 | 编码patch之间的相对距离 | 在某些任务上更好 |

ViT原论文令人意外地发现，1D可学习位置编码与2D编码效果相差无几——模型能够从数据中自行学习到patch的空间排列关系。

最终的输入序列构造如下：

$$\mathbf{z}_0 = [\mathbf{x}_{\text{cls}}; \mathbf{z}_1; \mathbf{z}_2; \ldots; \mathbf{z}_N] + \mathbf{E}_{\text{pos}}$$

其中 $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ 是可学习的位置嵌入矩阵，分号表示沿序列方向拼接。

### 21.1.4 [CLS] Token用于分类

借鉴BERT的设计，ViT在序列最前端插入一个可学习的 **[CLS] token**（Classification Token）。这个特殊token的初始值是随机初始化的，通过Transformer的自注意力机制与所有patch交互，最终汇聚全局信息。

在编码器输出后，只取 [CLS] token对应位置的输出向量 $\mathbf{z}_0^{(L)}$，送入一个简单的MLP分类头：

$$\mathbf{y} = \text{MLP}(\mathbf{z}_0^{(L)})$$

这一设计的优雅之处在于：不需要对N个patch的输出做池化（全局平均/最大池化），而是让模型自主学习如何从各patch中"汇总"有用信息。实验表明，也可以用所有patch输出的平均值（Global Average Pooling）替代 [CLS] token，效果相近。

### 21.1.5 ViT配置

ViT有多个标准配置，命名规则为 **ViT-{规模}/{patch大小}**：

| 配置 | 层数 $L$ | 隐藏维度 $D$ | 注意力头数 $h$ | MLP维度 | 参数量 |
|:-----|:--------:|:------------:|:--------------:|:-------:|:------:|
| ViT-Ti/16 | 12 | 192 | 3 | 768 | 5.7M |
| ViT-S/16 | 12 | 384 | 6 | 1536 | 22M |
| ViT-B/16 | 12 | 768 | 12 | 3072 | 86M |
| ViT-L/16 | 24 | 1024 | 16 | 4096 | 307M |
| ViT-H/14 | 32 | 1280 | 16 | 5120 | 632M |
| ViT-G/14 | 40 | 1408 | 16 | 6144 | 1.8B |

**命名解读**：ViT-B/16 表示 Base 规模、patch大小为 $16 \times 16$；ViT-L/14 表示 Large 规模、patch大小为 $14 \times 14$（patch越小，序列越长，细节越丰富，计算量也越大）。

ViT对训练数据量非常敏感。在ImageNet-1k（128万张）上从头训练，ViT-B/16比同参数量的CNN略差；而在JFT-300M（3亿张）上预训练后微调，则大幅超越CNN。这印证了"ViT需要用数据来弥补归纳偏置的缺失"这一核心结论。

---

## 21.2 Patch Embedding

### 21.2.1 图像分块过程

理解Patch Embedding的最直观方式是把它看作一种**结构化的图像切割**。以 $224 \times 224$ 图像、$16 \times 16$ patch为例：

```
原始图像（224×224）
┌─────────────────────────────┐
│  patch(0,0)  patch(0,1) ... │  ← 第0行，共14列
│  patch(1,0)  patch(1,1) ... │  ← 第1行
│     ...         ...         │
│  patch(13,0) ...patch(13,13)│  ← 第13行
└─────────────────────────────┘
总计：14 × 14 = 196 个patch

每个patch形状：16 × 16 × 3 = 768维向量（展平后）
```

分块操作可以用 `torch.Tensor` 的 `unfold` 或直接用步长等于patch大小的二维卷积实现。

### 21.2.2 线性投影

展平后的patch向量维度为 $P^2 C$，通过线性变换映射到模型维度 $D$：

$$\mathbf{e}_i = \mathbf{W}_E \cdot \text{Flatten}(\mathbf{x}_i^{\text{patch}}) + \mathbf{b}_E$$

其中 $\mathbf{W}_E \in \mathbb{R}^{D \times P^2C}$，$\mathbf{b}_E \in \mathbb{R}^D$。

这个线性投影本质上是在学习一组 $D$ 个"视觉滤波器"，每个滤波器对一个patch的像素值做线性响应，生成该patch的低维表示。与CNN第一层卷积的作用类似，但没有局部连接的约束。

### 21.2.3 完整公式推导

设图像 $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$，patch大小 $P$，模型维度 $D$，则完整的Patch Embedding流程为：

$$
\begin{aligned}
N &= \frac{HW}{P^2} \quad \text{（patch数量）} \\
\mathbf{x}_i &= \text{Reshape}(\mathbf{X}[r_i:r_i+P,\ c_i:c_i+P,\ :]) \in \mathbb{R}^{P^2C} \quad \text{（第 } i \text{ 个patch）} \\
\mathbf{e}_i &= \mathbf{W}_E \mathbf{x}_i + \mathbf{b}_E \in \mathbb{R}^D \quad \text{（线性投影）} \\
\mathbf{z}_0 &= [\mathbf{v}_{\text{cls}}; \mathbf{e}_1; \ldots; \mathbf{e}_N] + \mathbf{E}_{\text{pos}} \quad \text{（加入CLS和位置编码）}
\end{aligned}
$$

最终 $\mathbf{z}_0 \in \mathbb{R}^{(N+1) \times D}$ 是Transformer的输入序列。

### 21.2.4 卷积实现技巧

在实际工程中，线性投影 $+ $ 分块操作可以用**单次卷积**高效实现：

使用一个卷积核大小为 $P \times P$、步长为 $P$、输出通道数为 $D$ 的二维卷积，一步完成切块和线性投影：

```python
# 等价操作：patch embedding = Conv2d(kernel=P, stride=P, out_channels=D)
# 输入：(B, C, H, W)
# 输出：(B, D, H/P, W/P) → reshape → (B, N, D)
```

这一等价性来自：步长等于核大小的卷积恰好实现非重叠滑窗，每个窗口对应一个patch；卷积核参数与线性投影矩阵一一对应。这种实现在GPU上可以利用高度优化的CUDNN卷积内核，比手动unfold + matmul快得多。

---

## 21.3 CLIP

### 21.3.1 图像-文本对比学习

2021年，OpenAI发布了 **CLIP**（Contrastive Language-Image Pre-training，Radford et al., 2021），用4亿个从互联网爬取的图像-文本对进行对比学习预训练，开创了视觉-语言联合表示学习的新范式。

CLIP的训练目标非常简单：**给定一批图像-文本对，让匹配的对的相似度尽可能高，不匹配的对的相似度尽可能低**。

```
训练数据（一个batch，N个对）：
  图像：[img_1, img_2, ..., img_N]
  文本：[txt_1, txt_2, ..., txt_N]

目标：
  相似度矩阵 S（N×N）中，对角线元素最大（匹配对），
  非对角线元素最小（不匹配对）
```

这一目标不需要手动标注类别标签，所有监督信号来自"图像与哪段文字一起出现"这一天然的配对关系，因此可以轻松扩展到数十亿规模的数据。

### 21.3.2 双塔架构

CLIP采用**双塔（Dual Encoder）**架构，图像和文本分别由独立的编码器处理：

```
图像输入                    文本输入
    │                           │
Image Encoder               Text Encoder
（ViT-B/32 或 ResNet）      （Transformer）
    │                           │
图像嵌入 f_i ∈ R^D         文本嵌入 g_i ∈ R^D
    │                           │
    └──────── 余弦相似度 ────────┘
              s_ij = f_i · g_j / (‖f_i‖ ‖g_j‖)
```

两个编码器的输出都经过L2归一化，投影到同一个 $D$ 维语义空间（原论文 $D=512$ 或 $D=768$）。在这个共享空间中，相似的图像和文本应该彼此靠近。

**Image Encoder** 可以是ResNet-50/101或ViT（ViT-B/32、ViT-L/14等）。CLIP原论文测试了多个规模，发现ViT-L/14@336px效果最佳。

**Text Encoder** 是一个12层的Transformer，最大序列长度77个token，取 [EOS] token的输出作为文本表示。

### 21.3.3 InfoNCE损失

CLIP使用 **InfoNCE（Noise Contrastive Estimation）** 损失，也称为对称交叉熵损失：

设batch大小为 $N$，$\mathbf{F} \in \mathbb{R}^{N \times D}$ 为图像嵌入矩阵，$\mathbf{G} \in \mathbb{R}^{N \times D}$ 为文本嵌入矩阵（均已L2归一化），温度参数为 $\tau$（可学习），则相似度矩阵为：

$$\mathbf{S} = \frac{\mathbf{F} \mathbf{G}^T}{\tau} \in \mathbb{R}^{N \times N}$$

对比损失从两个方向计算：

$$\mathcal{L}_{\text{img}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\mathbf{S}_{ii})}{\sum_{j=1}^N \exp(\mathbf{S}_{ij})} \quad \text{（以图像为查询）}$$

$$\mathcal{L}_{\text{txt}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\mathbf{S}_{ii})}{\sum_{j=1}^N \exp(\mathbf{S}_{ji})} \quad \text{（以文本为查询）}$$

$$\mathcal{L}_{\text{CLIP}} = \frac{\mathcal{L}_{\text{img}} + \mathcal{L}_{\text{txt}}}{2}$$

这个损失的直观含义：对于每张图像，正确配对的文本在所有文本中的得分最高；对于每段文本，正确配对的图像在所有图像中的得分最高。两个方向对称，因此叫"对称交叉熵"。

**温度参数 $\tau$ 的作用**：$\tau$ 控制相似度分布的"尖锐程度"。$\tau$ 小时分布更尖锐，模型被迫在相似的负样本中精准区分；$\tau$ 大时分布平坦，训练信号弱。CLIP将 $\tau$ 设为可学习参数（初始值 $\approx 0.07$），实验显示这比固定值效果更好。

### 21.3.4 Zero-shot图像分类

CLIP最令人印象深刻的能力是**Zero-shot分类**：无需在目标数据集上微调，直接用文本描述类别进行分类。

流程如下：

```
1. 准备类别文本：
   "a photo of a dog"
   "a photo of a cat"
   "a photo of a car"
   ...（N_class 个类别）

2. 用 Text Encoder 编码所有类别文本，得到 N_class 个文本嵌入

3. 用 Image Encoder 编码待分类图像，得到图像嵌入

4. 计算图像嵌入与所有类别文本嵌入的余弦相似度

5. 取相似度最高的类别作为预测结果
```

CLIP在零样本设置下，在ImageNet上达到76.2%的Top-1准确率，接近同期有监督训练的ResNet-50（76.1%）。这意味着CLIP从未见过ImageNet的训练图像，仅凭在网络数据上学到的视觉-语言对齐，就能完成分类——这是多模态预训练能力的有力证明。

**Prompt Engineering**：文本模板的选择对Zero-shot性能影响显著。"a photo of a {class}" 比直接用 "{class}" 提升约3.5%。更复杂的ensemble提示（如80个不同模板的平均）可以进一步提升。

---

## 21.4 多模态融合

### 21.4.1 Early Fusion vs Late Fusion

多模态融合的核心问题是**何时融合、如何融合**。最基本的两种策略：

**Early Fusion（早期融合）**：在特征提取的早期阶段将不同模态的信息合并。

```
图像 ──→ [图像特征]──┐
                     ├──→ 融合表示 ──→ Transformer ──→ 输出
文本 ──→ [文本特征]──┘
```

- 优点：模态间可以尽早交互，充分利用跨模态信息
- 缺点：要求不同模态的特征空间对齐，对齐困难时训练不稳定
- 代表：ViLBERT的双流模型中的某些变体、单流VLP模型

**Late Fusion（晚期融合）**：每个模态独立提取高层语义特征后再融合。

```
图像 ──→ Image Encoder ──→ 图像语义嵌入 ──┐
                                          ├──→ 融合/对比 ──→ 输出
文本 ──→ Text Encoder  ──→ 文本语义嵌入 ──┘
```

- 优点：模态解耦，各编码器可以独立预训练，部署灵活
- 缺点：缺乏细粒度的跨模态交互（CLIP的文字只能描述整体，无法细粒度定位）
- 代表：CLIP、ALIGN

两种策略各有适用场景：检索类任务（图文匹配）适合Late Fusion；理解类任务（VQA、图像描述）适合Early Fusion。

### 21.4.2 Cross-Attention融合

**Cross-Attention**（交叉注意力）是当前多模态融合的主流机制，允许一个模态的表示以另一个模态的表示为键值对（Key-Value）进行信息查询：

$$\text{CrossAttention}(\mathbf{Q}_{\text{img}}, \mathbf{K}_{\text{txt}}, \mathbf{V}_{\text{txt}}) = \text{softmax}\left(\frac{\mathbf{Q}_{\text{img}} \mathbf{K}_{\text{txt}}^T}{\sqrt{d_k}}\right)\mathbf{V}_{\text{txt}}$$

这里图像特征作为Query，文本特征作为Key和Value：模型可以动态地根据图像内容去"查询"最相关的文本信息。反之也可以用文本特征查询图像特征。

Cross-Attention使得图像区域和文本词语之间能够建立细粒度的对应关系，例如："左上角的红色物体"可以与文本中的"苹果"产生高注意力权重。

### 21.4.3 Flamingo架构简介

**Flamingo**（Alayrac et al., DeepMind 2022）是将预训练视觉编码器和预训练语言模型结合的里程碑工作，核心思想是**冻结两个预训练模型，只训练连接它们的跨模态接口**。

Flamingo的关键设计：

**Perceiver Resampler**：由于ViT输出的patch token数量随图像分辨率变化，Flamingo用一个Perceiver模块将可变长度的视觉特征压缩为固定数量（如64个）的视觉token，降低后续计算量：

```
ViT输出：N_patch 个视觉token（N_patch 可变）
           │
     Perceiver Resampler
    （64个可学习的latent queries
      通过cross-attention聚合视觉信息）
           │
        64个压缩视觉token（固定长度）
```

**Gated Cross-Attention**：在语言模型的每个Transformer层之间插入带门控的Cross-Attention层，让语言模型"看到"视觉信息：

$$\mathbf{h}' = \mathbf{h} + \tanh(\alpha) \cdot \text{CrossAttention}(\mathbf{h}, \mathbf{v})$$

其中 $\alpha$ 是可学习的标量（初始值为0），保证训练初始阶段跨模态层不破坏语言模型的原有输出，训练更稳定。预训练的语言模型权重被冻结，只训练交叉注意力层和Perceiver。

这种设计使Flamingo具备了 few-shot 多模态能力：给出几个图文示例后，无需微调就能完成新任务。

### 21.4.4 LLaVA架构简介

**LLaVA**（Liu et al., 2023）取"Large Language and Vision Assistant"之意，是开源多模态大模型的重要里程碑，其设计思路极为简洁：

```
图像
  │
CLIP ViT-L/14（冻结）
  │
视觉特征矩阵（N×1024）
  │
线性投影层（唯一可训练的视觉桥接模块）
  │
视觉token序列（N×4096）
  │
与文本token拼接
  │
LLaMA/Vicuna LLM（可微调）
  │
文本输出
```

LLaVA最重要的贡献不在架构创新，而在于**数据构造方法**：利用GPT-4生成高质量的视觉指令微调数据（158k样本），证明了小规模高质量数据也能产生强大的多模态理解能力。

LLaVA-1.5进一步将线性投影层替换为两层MLP，并使用CLIP ViT-L/14@336px，在多个基准上达到了当时的最优性能。

---

## 21.5 视觉语言模型

### 21.5.1 从CLIP到多模态LLM

多模态大语言模型（Multimodal LLM，MLLM）的发展可以分为三个阶段：

```
阶段一：对比学习预训练（2021）
  CLIP、ALIGN ── 建立图文共同语义空间，Zero-shot能力强，但不能生成

阶段二：图文生成预训练（2022）
  Flamingo、CoCa ── 结合对比学习与自回归生成，支持图文交替输入/输出

阶段三：指令微调时代（2023至今）
  LLaVA、InstructBLIP、GPT-4V ── 用指令数据对齐人类意图，支持对话式交互
```

核心技术转变：从"判别式"（判断图像属于哪类）到"生成式"（根据图像生成文本描述、回答问题）。

### 21.5.2 视觉编码器的选择

视觉编码器是MLLM感知世界的"眼睛"，选择至关重要：

| 编码器 | 预训练方式 | 优势 | 劣势 |
|:-------|:----------|:-----|:-----|
| CLIP ViT-L/14 | 图文对比 | 语义对齐好，通用性强 | 细粒度局部特征弱 |
| CLIP ViT-L/14@336px | 图文对比（高分辨率） | 分辨率更高，细节更好 | 计算量大（576 tokens） |
| DINOv2 ViT-L/14 | 自监督（知识蒸馏） | 局部特征丰富，适合定位 | 语言对齐较弱 |
| SigLIP ViT-So400M | Sigmoid对比损失 | 大规模，性能强 | 参数量大 |
| ConvNeXt | 监督分类 | 局部纹理细节好 | 全局语义弱 |

目前主流MLLM（如LLaVA-1.6、InternVL、Qwen-VL）倾向于使用**高分辨率CLIP ViT**或**SigLIP**作为视觉编码器，并冻结其参数以保留预训练的视觉语义表示。

### 21.5.3 投影层设计

投影层（Projection Layer / Visual Adapter）负责将视觉编码器的输出特征维度 $D_v$ 映射到语言模型的嵌入维度 $D_l$：

**线性投影**（LLaVA v1）：
$$\mathbf{H}_v = \mathbf{W} \mathbf{F}_v, \quad \mathbf{W} \in \mathbb{R}^{D_l \times D_v}$$

参数量小，训练快，但表达能力有限。

**MLP投影**（LLaVA v1.5）：
$$\mathbf{H}_v = \text{MLP}(\mathbf{F}_v) = \text{GELU}(\mathbf{W}_1 \mathbf{F}_v + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$

引入非线性，效果明显提升，是当前主流选择。

**Q-Former**（BLIP-2）：使用一组可学习的query token通过Cross-Attention从视觉特征中提取信息，可以灵活控制传入LLM的视觉token数量，适合压缩高分辨率图像特征。

**Token 数量权衡**：投影层输出的视觉token数量直接影响LLM的上下文长度。224px图像经ViT-L/14产生196个token，336px产生576个token。更多token带来更多细节，但增加计算量（自注意力 $O(L^2)$）。高分辨率方案（如LLaVA-HR）将图像切分为多个低分辨率子图，保留细节的同时控制token数量。

### 21.5.4 指令微调

指令微调（Instruction Fine-tuning）是让模型从"能看图"到"能交互"的关键步骤，将MLLM对齐到人类对话习惯。

典型的指令微调数据格式：

```
[System]: 你是一个有帮助的视觉助手。
[User]: <图像> 这张图片里有什么动物？
[Assistant]: 图片中有一只橙色的猫咪，正趴在阳光照射的窗台上。
```

数据来源：
- **人工标注**：质量高，成本极高
- **GPT-4V生成**：利用闭源强模型生成训练数据，成本较低
- **已有VQA数据集转换**：如VQAv2、GQA、TextVQA等改造为对话格式

两阶段训练流程（以LLaVA为例）：

1. **预训练阶段**：冻结视觉编码器和LLM，只训练投影层；使用595k图文对，让投影层学会对齐视觉和语言空间（约1小时，单卡A100）
2. **指令微调阶段**：解冻LLM（视觉编码器仍冻结），用158k指令数据微调；模型学习遵循指令和多轮对话（约15小时，8卡A100）

---

## 本章小结

| 模型/方法 | 年份 | 核心创新 | 输入 | 输出 | 代表性能 |
|:---------|:----:|:--------|:-----|:-----|:--------|
| ViT-B/16 | 2020 | 图像切块→Transformer | 图像 | 类别概率 | ImageNet 81.8%（JFT预训练） |
| CLIP | 2021 | 图文对比学习 | 图像+文本 | 相似度 | ImageNet 76.2%（Zero-shot） |
| Flamingo-80B | 2022 | 冻结LM + Gated Cross-Attn | 图像/视频+文本 | 文本 | VQAv2 82.0（few-shot） |
| BLIP-2 | 2023 | Q-Former桥接 | 图像+文本 | 文本 | VQAv2 82.2（zero-shot） |
| LLaVA-1.5 | 2023 | MLP投影 + 指令微调 | 图像+文本 | 文本 | MM-Bench 85.9 |
| InternVL-Chat | 2024 | 大规模视觉编码器 | 图像+文本 | 文本 | MMBench 92.3 |

**关键设计轴线**：
- **模态融合时机**：CLIP（晚期，对比）→ Flamingo（中期，交叉注意力）→ 单流模型（早期，共享Transformer）
- **训练策略**：全量预训练 → 冻结预训练模型 + 训练桥接 → 指令微调对齐
- **参数效率**：LLaVA仅需训练投影层（~数百万参数）即可激活LLM的视觉理解能力

---

## 代码实战

### 完整实现：PatchEmbedding与ViT

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────
# 1. Patch Embedding
# ─────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """
    将图像切分为 patch 并投影到模型维度。

    等价于：将图像 reshape 为 patch 序列后做线性变换，
    但用卷积实现更高效。
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 用步长等于 patch_size 的卷积一步完成切块 + 线性投影
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, (
            f"输入图像尺寸 ({H}x{W}) 与预期 ({self.image_size}x{self.image_size}) 不符"
        )

        # (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        # (B, embed_dim, N_h * N_w) -> (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


# ─────────────────────────────────────────────
# 2. Multi-Head Self-Attention
# ─────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        # 生成 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, h, N, d_h)
        q, k, v = qkv.unbind(0)            # 各自 (B, h, N, d_h)

        # 缩放点积注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        x = self.proj(x)
        return x


# ─────────────────────────────────────────────
# 3. Transformer Block
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# 4. Vision Transformer（ViT）
# ─────────────────────────────────────────────
class VisionTransformer(nn.Module):
    """
    ViT-Base/16 默认配置：
      image_size=224, patch_size=16, embed_dim=768,
      depth=12, num_heads=12, num_classes=1000
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Patch Embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 可学习的 [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 可学习的位置编码（长度 = num_patches + 1，+1 是 CLS token）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(dropout)

        # Transformer 编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        # 位置编码用截断正态分布初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # 线性层和 LayerNorm 标准初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # 1. Patch Embedding: (B, N, D)
        x = self.patch_embed(x)

        # 2. 拼接 [CLS] token: (B, N+1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 3. 加入位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Transformer 编码器
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 5. 取 [CLS] token 输出做分类
        cls_output = x[:, 0]          # (B, D)
        logits = self.head(cls_output)  # (B, num_classes)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """返回 [CLS] token 的特征向量，用于下游任务"""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]


# ─────────────────────────────────────────────
# 5. CLIP 风格的对比学习
# ─────────────────────────────────────────────
class CLIPModel(nn.Module):
    """
    简化版 CLIP 模型，演示双塔对比学习框架。
    Image Encoder: 上面实现的 ViT（精简版）
    Text Encoder:  单层 Transformer（演示用）
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 512,
        vision_depth: int = 6,
        text_vocab_size: int = 49408,  # CLIP 词表大小
        text_max_len: int = 77,
        text_depth: int = 6,
        num_heads: int = 8,
        temperature_init: float = 0.07,
    ):
        super().__init__()

        # 图像编码器（ViT，输出 embed_dim 维特征）
        self.image_encoder = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=vision_depth,
            num_heads=num_heads,
            num_classes=embed_dim,  # 用分类头做投影
        )
        # 把原来的分类头换成投影头（映射到共享空间）
        self.image_encoder.head = nn.Identity()
        self.image_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 文本编码器
        self.text_embedding = nn.Embedding(text_vocab_size, embed_dim)
        self.text_pos_embed = nn.Parameter(
            torch.zeros(1, text_max_len, embed_dim)
        )
        self.text_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(text_depth)
        ])
        self.text_norm = nn.LayerNorm(embed_dim)
        self.text_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 可学习温度参数（log scale 以保证正值）
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(temperature_init))
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """返回 L2 归一化的图像特征"""
        features = self.image_encoder.get_features(images)
        features = self.image_proj(features)
        return F.normalize(features, dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, L) 文本 token id
        返回 L2 归一化的文本特征（取最后一个非 padding token 的输出）
        """
        B, L = tokens.shape
        x = self.text_embedding(tokens)           # (B, L, D)
        x = x + self.text_pos_embed[:, :L, :]
        for block in self.text_blocks:
            x = block(x)
        x = self.text_norm(x)

        # 取每个序列的最后一个有效 token（EOS token）
        # 简化版：直接取序列末尾位置
        text_features = x[:, -1, :]              # (B, D)
        text_features = self.text_proj(text_features)
        return F.normalize(text_features, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
    ) -> dict:
        """
        计算 CLIP 对比损失。
        images: (B, C, H, W)
        tokens: (B, L) 文本 token id
        """
        image_features = self.encode_image(images)  # (B, D)
        text_features = self.encode_text(tokens)    # (B, D)

        # 温度缩放的相似度矩阵
        temperature = torch.exp(self.log_temperature)
        logits = (image_features @ text_features.T) / temperature  # (B, B)

        # 对角线为正样本标签
        B = images.shape[0]
        labels = torch.arange(B, device=images.device)

        # 对称交叉熵损失
        loss_img = F.cross_entropy(logits, labels)
        loss_txt = F.cross_entropy(logits.T, labels)
        loss = (loss_img + loss_txt) / 2.0

        return {
            "loss": loss,
            "loss_img": loss_img,
            "loss_txt": loss_txt,
            "temperature": temperature.item(),
            "logits": logits,
        }


# ─────────────────────────────────────────────
# 6. 演示与验证
# ─────────────────────────────────────────────
def demo_vit():
    print("=" * 50)
    print("ViT-Base/16 前向传播演示")
    print("=" * 50)

    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=1000,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.1f}M")
    print(f"Patch 数量: {model.patch_embed.num_patches}")

    # 模拟一批图像
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        logits = model(images)
        features = model.get_features(images)

    print(f"输入形状: {images.shape}")
    print(f"输出 logits 形状: {logits.shape}")
    print(f"特征向量形状: {features.shape}")


def demo_patch_embedding():
    print("\n" + "=" * 50)
    print("PatchEmbedding 分步演示")
    print("=" * 50)

    patch_embed = PatchEmbedding(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
    )

    x = torch.randn(2, 3, 224, 224)
    out = patch_embed(x)

    print(f"输入图像: {x.shape}  (B, C, H, W)")
    print(f"Patch 序列: {out.shape}  (B, N, embed_dim)")
    print(f"其中 N = {out.shape[1]} = (224/16)^2 = 14*14 = 196")


def demo_clip():
    print("\n" + "=" * 50)
    print("CLIP 对比学习演示")
    print("=" * 50)

    model = CLIPModel(
        image_size=224,
        patch_size=16,
        embed_dim=512,
        vision_depth=4,    # 演示用，实际应更深
        text_depth=4,
        num_heads=8,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.1f}M")

    batch_size = 8
    images = torch.randn(batch_size, 3, 224, 224)
    tokens = torch.randint(0, 49408, (batch_size, 77))

    output = model(images, tokens)
    print(f"对比损失: {output['loss'].item():.4f}")
    print(f"图像方向损失: {output['loss_img'].item():.4f}")
    print(f"文本方向损失: {output['loss_txt'].item():.4f}")
    print(f"当前温度参数: {output['temperature']:.4f}")
    print(f"相似度矩阵形状: {output['logits'].shape}  (B, B)")


def zero_shot_demo():
    """演示 Zero-shot 分类逻辑（不加载真实权重，仅展示流程）"""
    print("\n" + "=" * 50)
    print("Zero-shot 分类流程演示")
    print("=" * 50)

    embed_dim = 512

    # 模拟已预训练的编码器（随机初始化仅做流程演示）
    image_encoder = lambda img: F.normalize(
        torch.randn(img.shape[0], embed_dim), dim=-1
    )
    text_encoder = lambda txt: F.normalize(
        torch.randn(len(txt), embed_dim), dim=-1
    )

    # 类别文本模板
    class_names = ["猫", "狗", "汽车", "飞机", "船"]
    class_prompts = [f"a photo of a {c}" for c in class_names]

    # 编码类别文本
    text_features = text_encoder(class_prompts)  # (num_classes, D)
    print(f"类别文本特征形状: {text_features.shape}")

    # 编码待分类图像
    test_images = torch.randn(3, 3, 224, 224)
    image_features = image_encoder(test_images)  # (B, D)
    print(f"图像特征形状: {image_features.shape}")

    # 计算相似度并分类
    similarity = image_features @ text_features.T  # (B, num_classes)
    predictions = similarity.argmax(dim=-1)

    print("分类结果（随机权重，仅演示流程）:")
    for i, pred in enumerate(predictions):
        print(f"  图像 {i+1} -> 预测类别: {class_names[pred.item()]}")


if __name__ == "__main__":
    demo_patch_embedding()
    demo_vit()
    demo_clip()
    zero_shot_demo()
```

**运行输出**（参考）：

```
==================================================
PatchEmbedding 分步演示
==================================================
输入图像: torch.Size([2, 3, 224, 224])  (B, C, H, W)
Patch 序列: torch.Size([2, 196, 768])  (B, N, embed_dim)
其中 N = 196 = (224/16)^2 = 14*14 = 196

==================================================
ViT-Base/16 前向传播演示
==================================================
总参数量: 86.6M
Patch 数量: 196
输入形状: torch.Size([4, 3, 224, 224])
输出 logits 形状: torch.Size([4, 1000])
特征向量形状: torch.Size([4, 768])

==================================================
CLIP 对比学习演示
==================================================
总参数量: 91.2M
对比损失: 2.0794
图像方向损失: 2.0812
文本方向损失: 2.0776
当前温度参数: 0.0700
相似度矩阵形状: torch.Size([8, 8])
```

---

## 练习题

### 基础题

**题目 21-1**（基础）：在ViT-B/16中，输入图像尺寸为 $224 \times 224$，patch大小为 $16 \times 16$。请计算：
1. Transformer编码器接收的序列长度（包含 [CLS] token）
2. PatchEmbedding层的参数量（投影矩阵 $\mathbf{E}$ 和偏置 $\mathbf{b}$）

---

**题目 21-2**（基础）：CLIP使用对称InfoNCE损失，设batch大小为 $N=256$，温度 $\tau=0.07$。解释：
1. 为什么较大的batch size对CLIP训练有益？
2. 温度 $\tau=0.07$ 与 $\tau=1.0$ 在训练时有何区别？梯度信号有何不同？

---

### 中级题

**题目 21-3**（中级）：在上述代码中，`PatchEmbedding` 使用了 `Conv2d(kernel_size=P, stride=P)` 实现。请用纯矩阵操作（`unfold` + `linear`）重新实现 `PatchEmbedding`，并验证两种实现的输出数值上等价（提示：需要正确对齐卷积权重的排列顺序）。

---

**题目 21-4**（中级）：LLaVA将冻结的CLIP ViT输出的视觉特征通过MLP投影后拼接到文本序列中。假设：
- 视觉编码器：CLIP ViT-L/14@336px，输出维度 1024，共 576 个视觉 token
- LLM：LLaMA-7B，嵌入维度 4096
- 问题长度：50 个文本 token

请计算：
1. 投影层MLP（两层，中间维度4096）的参数量
2. LLM处理一条样本时，自注意力的输入序列长度
3. 如果将336px改为224px（256个视觉token），自注意力的计算量（矩阵乘法次数，与序列长度平方成比例）变化比例是多少？

---

### 提高题

**题目 21-5**（提高）：Flamingo使用Gated Cross-Attention层，公式为：

$$\mathbf{h}' = \mathbf{h} + \tanh(\alpha) \cdot \text{CrossAttention}(\mathbf{h}, \mathbf{v})$$

其中 $\alpha$ 初始化为0。

1. 当 $\alpha = 0$ 时，$\tanh(0) = 0$，说明训练初始阶段该层不改变语言模型的输出。这样设计有什么好处？
2. 如果不用门控（即直接 $\mathbf{h}' = \mathbf{h} + \text{CrossAttention}(\mathbf{h}, \mathbf{v})$），训练会遇到什么问题？
3. 请用PyTorch实现一个带门控的Cross-Attention模块（`GatedCrossAttention`），要求：接受语言特征 $\mathbf{H} \in \mathbb{R}^{B \times L \times D}$ 和视觉特征 $\mathbf{V} \in \mathbb{R}^{B \times M \times D}$ 作为输入，输出与 $\mathbf{H}$ 同维度的更新特征。

---

## 练习答案

### 题目 21-1 答案

**第1问：序列长度**

$$N_{\text{patch}} = \left(\frac{224}{16}\right)^2 = 14^2 = 196$$

加上 [CLS] token，Transformer编码器的输入序列长度为：

$$N_{\text{total}} = 196 + 1 = \boxed{197}$$

**第2问：PatchEmbedding参数量**

投影矩阵 $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D} = \mathbb{R}^{768 \times 768}$，参数量为 $768 \times 768 = 589{,}824$。

偏置 $\mathbf{b} \in \mathbb{R}^{768}$，参数量为 $768$。

总计：$589{,}824 + 768 = \boxed{590{,}592} \approx 0.59\text{M}$

（等价地，`Conv2d(3, 768, 16, 16)` 的参数：$768 \times 3 \times 16 \times 16 + 768 = 590{,}592$，验证一致）

---

### 题目 21-2 答案

**第1问：batch size的作用**

InfoNCE损失中，每个正样本需要与 $N-1$ 个负样本对比。**batch size越大，负样本越多，对比任务越难，模型学到的表示越精细**。具体来说：

- 小batch（如32）：负样本少，模型只需区分32个选项，任务简单，学到的边界粗糙
- 大batch（如32768）：负样本多，模型需要在更大集合中精准定位，学到的特征更有判别力

这也是为什么CLIP原论文使用32,768的超大batch大小，需要数百块GPU协同训练。

**第2问：温度参数的影响**

给定相似度 $s_{ij}$，softmax分母为 $\sum_j \exp(s_{ij}/\tau)$：

- $\tau = 0.07$（小温度）：相似度差异被放大，softmax分布更"尖锐"。模型对区分相似负样本产生强梯度，学习信号强，但容易在训练初期因分布过于集中而不稳定。
- $\tau = 1.0$（大温度）：softmax分布平坦，所有负样本的权重接近均等。梯度弱，收敛慢，但训练稳定。

CLIP将 $\tau$ 设为可学习参数，训练过程中自动从较大值减小到约0.07，兼顾了稳定性和最终性能。

---

### 题目 21-3 答案

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbeddingUnfold(nn.Module):
    """用 unfold + linear 实现的 PatchEmbedding"""
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.linear = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        P = self.patch_size

        # unfold 按 patch 大小切块
        # unfold(dim, size, step): 在 dim 维度上，以 size 为窗口、step 为步长滑动
        x = x.unfold(2, P, P).unfold(3, P, P)
        # x 形状: (B, C, H/P, W/P, P, P)

        N_h, N_w = H // P, W // P
        # 重排为 (B, N, patch_dim)
        x = x.contiguous().reshape(B, C, N_h * N_w, P * P)
        x = x.permute(0, 2, 1, 3).reshape(B, N_h * N_w, C * P * P)

        return self.linear(x)  # (B, N, embed_dim)


def verify_equivalence():
    """验证两种实现的数值等价性"""
    B, C, H, P, D = 2, 3, 224, 16, 768

    # 创建两个模块
    conv_embed = PatchEmbedding(H, P, C, D)
    unfold_embed = PatchEmbeddingUnfold(H, P, C, D)

    # 将 Conv2d 权重复制到 Linear
    # Conv2d 权重形状: (D, C, P, P) -> reshape -> (D, C*P*P) = Linear 权重 (D, C*P*P)
    with torch.no_grad():
        unfold_embed.linear.weight.copy_(
            conv_embed.projection.weight.reshape(D, -1)
        )
        unfold_embed.linear.bias.copy_(conv_embed.projection.bias)

    x = torch.randn(B, C, H, H)

    with torch.no_grad():
        out_conv = conv_embed(x)
        out_unfold = unfold_embed(x)

    max_diff = (out_conv - out_unfold).abs().max().item()
    print(f"两种实现最大数值差异: {max_diff:.2e}")
    print(f"等价性验证: {'通过' if max_diff < 1e-5 else '失败'}")


# verify_equivalence()
```

关键点：`Conv2d` 权重形状为 `(out_channels, in_channels, kH, kW)` 即 `(D, C, P, P)`，reshape 为 `(D, C*P*P)` 后等价于 `Linear` 的权重矩阵（转置后为 `(C*P*P, D)`）。

---

### 题目 21-4 答案

**第1问：MLP投影层参数量**

两层MLP：Linear(1024→4096) + GELU + Linear(4096→4096)

$$\text{参数量} = (1024 \times 4096 + 4096) + (4096 \times 4096 + 4096)$$
$$= 4{,}198{,}400 + 16{,}781{,}312 = \boxed{20{,}979{,}712} \approx 21\text{M}$$

**第2问：自注意力输入序列长度**

$$L = N_{\text{visual}} + N_{\text{text}} = 576 + 50 = \boxed{626}$$

**第3问：计算量变化比例**

自注意力计算量与序列长度平方成比例（忽略视觉token本身的微小差异）：

336px时序列长度：$L_1 = 576 + 50 = 626$

224px时序列长度：$L_2 = 256 + 50 = 306$

计算量比例：

$$\frac{L_2^2}{L_1^2} = \frac{306^2}{626^2} = \frac{93{,}636}{391{,}876} \approx 0.239$$

即224px方案的自注意力计算量约为336px方案的 **23.9%**，降低了约 **76%**。这说明减少视觉token数量对推理效率有极大提升，这也是高分辨率图像处理的核心挑战所在。

---

### 题目 21-5 答案

**第1问：门控初始化为0的好处**

当 $\alpha = 0$，$\tanh(0) = 0$，Cross-Attention层输出为零，即：

$$\mathbf{h}' = \mathbf{h} + 0 \cdot \text{CrossAttention}(\mathbf{h}, \mathbf{v}) = \mathbf{h}$$

这保证了**训练初始阶段，预训练语言模型的行为不受破坏**。好处有两点：

1. **训练稳定性**：语言模型是高度调优的系统，随机初始化的Cross-Attention层在训练初期产生噪声输出，直接相加会破坏语言模型的分布，导致梯度爆炸或loss震荡。门控确保噪声从零开始逐步引入。
2. **正则化效果**：模型只在有明确训练信号证明视觉信息有用时，才逐渐增大 $\alpha$ 的值，避免过拟合。

**第2问：不用门控的问题**

随机初始化的Cross-Attention在前向传播中会产生任意值，直接加到语言特征上，相当于对语言模型施加随机扰动。具体问题：
- 语言建模loss在训练最初几步会急剧升高
- 可能触发梯度爆炸（Cross-Attention输出的L2范数可能很大）
- 即使后续能收敛，收敛速度也慢于带门控版本

**第3问：GatedCrossAttention 实现**

```python
class GatedCrossAttention(nn.Module):
    """
    Flamingo 风格的门控交叉注意力层。

    语言特征作为 Query，视觉特征作为 Key 和 Value。
    使用可学习的 tanh 门控保证训练初始阶段不破坏语言模型输出。
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q 来自语言特征，K/V 来自视觉特征
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.norm_lang = nn.LayerNorm(embed_dim)
        self.norm_vis = nn.LayerNorm(embed_dim)

        # 门控参数，初始化为 0
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        lang_features: torch.Tensor,   # (B, L, D)  语言特征
        vis_features: torch.Tensor,    # (B, M, D)  视觉特征
    ) -> torch.Tensor:
        B, L, D = lang_features.shape
        M = vis_features.shape[1]
        h = self.num_heads

        # LayerNorm
        lang_normed = self.norm_lang(lang_features)
        vis_normed = self.norm_vis(vis_features)

        # 投影并分头
        Q = self.q_proj(lang_normed).reshape(B, L, h, self.head_dim).transpose(1, 2)
        K = self.k_proj(vis_normed).reshape(B, M, h, self.head_dim).transpose(1, 2)
        V = self.v_proj(vis_normed).reshape(B, M, h, self.head_dim).transpose(1, 2)
        # Q: (B, h, L, d_h), K/V: (B, h, M, d_h)

        # 缩放点积注意力（语言 query 对视觉 key-value）
        attn = (Q @ K.transpose(-2, -1)) * self.scale   # (B, h, L, M)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, L, D)  # (B, L, D)
        out = self.out_proj(out)

        # 门控残差连接：初始 gate=0 时等价于恒等映射
        return lang_features + torch.tanh(self.gate) * out


# 验证
def test_gated_cross_attention():
    B, L, M, D, h = 2, 20, 64, 512, 8
    layer = GatedCrossAttention(D, h)

    lang = torch.randn(B, L, D)
    vis = torch.randn(B, M, D)

    # gate=0 时，输出应等于输入
    with torch.no_grad():
        out = layer(lang, vis)

    diff = (out - lang).abs().max().item()
    print(f"gate=0 时输出与输入的最大差异: {diff:.2e}")
    print(f"（应接近0，验证门控初始化的正确性）")
    print(f"输出形状: {out.shape}")


# test_gated_cross_attention()
```

运行 `test_gated_cross_attention()` 可以验证：当 `gate=0` 时，`tanh(0) * out = 0`，模块输出恰好等于输入，预训练语言模型的行为完全不变。

---

*本章完*
