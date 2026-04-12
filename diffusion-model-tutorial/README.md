# 扩散生成模型从零到高阶教程

> 系统、完整的扩散模型学习资源，从概率论基础出发，逐步深入DDPM核心原理、条件生成、架构设计，直至Stable Diffusion、DiT、Flow Matching等前沿方法，最终覆盖工程部署实践。

**本教程特色**：每章均包含从零实现的PyTorch代码，配备严格数学推导与丰富可视化，专为中文读者设计。

---

## 前置知识要求

| 类别 | 要求 |
|------|------|
| 编程 | Python 3.9+，熟悉面向对象编程 |
| 框架 | PyTorch 2.0+（张量操作、nn.Module、autograd） |
| 数学 | 微积分（偏导数、链式法则）、线性代数（矩阵运算） |
| 深度学习 | 反向传播、梯度下降、卷积神经网络基础 |
| 推荐 | 了解VAE或GAN（第四章会复习） |

---

## 章节导航

### 开始之前

- [前言：扩散模型的崛起与本教程使用指南](./00-preface.md)

---

### 第一部分：数学基础

| 章节 | 标题 | 核心内容 | 代码实战 |
|------|------|----------|----------|
| 第1章 | [概率论基础与随机变量](./part1-math-foundations/01-probability-random-variables.md) | 概率空间、随机变量、期望方差、重参数化技巧 | 分布采样与可视化 |
| 第2章 | [高斯分布与马尔科夫链](./part1-math-foundations/02-gaussian-markov-chains.md) | 多元高斯、条件分布、KL散度、高斯马尔科夫链 | 马尔科夫链模拟与可视化 |
| 第3章 | [变分推断基础](./part1-math-foundations/03-variational-inference.md) | ELBO推导、均场变分、VAE、重参数化梯度 | MNIST上VAE完整实现 |

---

### 第二部分：生成模型概览

| 章节 | 标题 | 核心内容 | 代码实战 |
|------|------|----------|----------|
| 第4章 | [生成模型全景与VAE回顾](./part2-generative-models/04-generative-models-overview-vae.md) | AR模型、GAN、VAE、归一化流、扩散模型对比 | VAE与GAN并排实现 |
| 第5章 | [基于分数的生成模型](./part2-generative-models/05-score-based-generative-models.md) | 分数函数、朗之万采样、去噪分数匹配、NCSN | 二维分数场可视化 |
| 第6章 | [随机微分方程入门](./part2-generative-models/06-stochastic-differential-equations.md) | 布朗运动、伊藤积分、正向/逆向SDE、概率流ODE | VP-SDE与VE-SDE轨迹对比 |

---

### 第三部分：DDPM核心原理

| 章节 | 标题 | 核心内容 | 代码实战 |
|------|------|----------|----------|
| 第7章 | [正向扩散过程与加噪](./part3-ddpm-core/07-diffusion-process-forward.md) | 马尔科夫加噪、闭合形式推导、后验分布、SNR分析 | 噪声调度器实现与可视化 |
| 第8章 | [逆向去噪过程与DDPM](./part3-ddpm-core/08-reverse-denoising-ddpm.md) | 逆向马尔科夫链、ELBO分解、简化训练目标 | DDPM完整训练与采样 |
| 第9章 | [噪声预测网络与训练目标](./part3-ddpm-core/09-noise-prediction-network-training.md) | 时间嵌入、ε/x₀/v预测、Min-SNR加权、EMA | 完整训练框架实现 |

---

### 第四部分：采样与加速

| 章节 | 标题 | 核心内容 | 代码实战 |
|------|------|----------|----------|
| 第10章 | [DDIM与确定性采样](./part4-sampling-acceleration/10-ddim-deterministic-sampling.md) | 非马尔科夫正向过程、子序列采样、DDIM Inversion | DDIM采样器与Inversion |
| 第11章 | [噪声调度器设计](./part4-sampling-acceleration/11-noise-scheduler-design.md) | 线性/余弦/Sigmoid调度、SNR分布、连续时间扩展 | 调度器对比实验 |
| 第12章 | [加速采样技术综述](./part4-sampling-acceleration/12-accelerated-sampling-survey.md) | DPM-Solver、PNDM、一致性模型、渐进蒸馏 | 质量-速度权衡对比 |

---

### 第五部分：条件生成

| 章节 | 标题 | 核心内容 | 代码实战 |
|------|------|----------|----------|
| 第13章 | [分类器引导生成](./part5-conditional-generation/13-classifier-guidance.md) | 条件分数推导、ADM架构、CLIP引导、引导强度分析 | 分类器引导采样实现 |
| 第14章 | [无分类器引导（CFG）](./part5-conditional-generation/14-classifier-free-guidance.md) | CFG推导、训练策略、CFG Scale选择、蒸馏加速 | CFG完整实现与可视化 |
| 第15章 | [文本条件图像生成](./part5-conditional-generation/15-text-conditional-generation.md) | CLIP/T5编码器、交叉注意力条件化、评估指标 | 文本条件扩散模型 |

---

### 第六部分：架构设计

| 章节 | 标题 | 核心内容 | 代码实战 |
|------|------|----------|----------|
| 第16章 | [U-Net架构详解](./part6-architecture/16-unet-architecture.md) | 残差块、时间嵌入注入、跳跃连接、AdaGN | 扩散U-Net从零实现 |
| 第17章 | [注意力机制在扩散模型中的应用](./part6-architecture/17-attention-in-diffusion.md) | 空间自注意力、交叉注意力、Flash Attention、注意力图 | Self-Attn + Cross-Attn实现 |
| 第18章 | [潜在扩散模型（LDM）](./part6-architecture/18-latent-diffusion-models.md) | KL-VAE感知压缩、潜在空间扩散、两阶段训练 | AutoencoderKL + LDM实现 |

---

### 第七部分：前沿模型

| 章节 | 标题 | 核心内容 | 代码实战 |
|------|------|----------|----------|
| 第19章 | [Stable Diffusion原理与实现](./part7-frontier-models/19-stable-diffusion.md) | SD架构全景、CLIP编码、img2img、ControlNet生态 | diffusers库完整推理管线 |
| 第20章 | [DALL-E 2与层次式生成](./part7-frontier-models/20-dalle2-clip-conditioning.md) | unCLIP框架、扩散先验、图像插值混合、级联超分 | CLIP嵌入操作实验 |
| 第21章 | [DiT：扩散Transformer](./part7-frontier-models/21-dit-diffusion-transformer.md) | Patch化、adaLN-Zero、Scaling Laws、Sora影响 | DiT从零实现 |
| 第22章 | [Flow Matching与一致性模型](./part7-frontier-models/22-flow-matching-consistency-models.md) | 直线流、FM训练目标、一致性函数、SD3/FLUX | Flow Matching玩具实验 |

---

### 第八部分：工程实践

| 章节 | 标题 | 核心内容 | 代码实战 |
|------|------|----------|----------|
| 第23章 | [扩散模型推理优化](./part8-engineering/23-inference-optimization.md) | 量化、Flash Attention、torch.compile、TensorRT | 性能基准测试 |
| 第24章 | [完整项目实战：文生图系统](./part8-engineering/24-complete-project-text-to-image.md) | 端到端系统设计、训练框架、FastAPI服务、评估 | 可运行文生图项目 |

---

### 附录

| 附录 | 标题 | 内容 |
|------|------|------|
| 附录A | [数学符号与公式速查](./appendix/math-reference.md) | 符号表、关键公式索引、高斯恒等式 |
| 附录B | [HuggingFace Diffusers API参考](./appendix/diffusers-api.md) | Pipeline、调度器、模型类速查 |
| 附录C | [练习题答案汇总](./appendix/answers.md) | 全部24章×5题详细解答 |

---

## 学习路径建议

### 路径一：入门路径（约4周）

适合有深度学习基础、首次接触扩散模型的学习者：

```
第1章 → 第2章 → 第7章 → 第8章 → 第9章 → 第16章 → 第19章
```

重点掌握：概率论基础 → DDPM核心 → U-Net架构 → Stable Diffusion使用

### 路径二：研究路径（约10周）

适合希望深入理论、从事生成模型研究的学习者：

```
第1-6章（全部数学基础）→ 第7-9章（DDPM）→ 第10-15章（采样与条件）→ 第16-22章（架构与前沿）
```

### 路径三：工程路径（约4周）

适合希望快速构建应用的工程师：

```
第7章 → 第8章 → 第10章 → 第14章 → 第16章 → 第18章 → 第23章 → 第24章
```

---

## 环境配置

```bash
# 推荐环境
Python >= 3.9
PyTorch >= 2.0
diffusers >= 0.25
transformers >= 4.36
accelerate >= 0.25

# 安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate
pip install numpy matplotlib einops tqdm
pip install torchmetrics  # FID计算
```

---

## 教程特色

- **24章完整内容**：从概率论基础到工程部署，覆盖扩散模型完整知识体系
- **从零实现**：核心组件均有手写PyTorch实现，不依赖黑盒封装
- **120道练习题**：每章5道精选习题（基础×2 + 中级×2 + 提高×1），含详细解答
- **严格数学推导**：所有公式均附完整LaTeX推导过程
- **前沿跟踪**：覆盖Flow Matching、DiT、一致性模型等2023-2024最新方法
- **中文优先**：专为中文读者设计，术语准确，表达清晰

---

## 许可证

MIT License — 自由使用、修改、分发，请保留原始署名。

---

*发现错误或有改进建议？欢迎提交 Issue 或 Pull Request。*
