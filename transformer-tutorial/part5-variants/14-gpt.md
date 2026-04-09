# 第14章：GPT系列——自回归语言模型的崛起

## 学习目标

学完本章后，你将能够：

1. 理解自回归语言模型的数学原理及其与双向模型的本质区别
2. 掌握GPT的架构设计，包括Pre-LN、因果掩码等关键组件
3. 理解GPT-1、GPT-2、GPT-3的技术演进脉络
4. 掌握In-Context Learning的概念、形式及涌现能力
5. 从零实现完整的GPT模型，包括多种文本生成采样策略

---

## 14.1 自回归语言模型

### 14.1.1 语言建模目标

语言模型的核心任务是对自然语言序列赋予概率分布。给定一个词序列 $w_1, w_2, \ldots, w_n$，语言模型需要计算该序列的联合概率：

$$P(w_1, w_2, \ldots, w_n)$$

**自回归分解**利用链式法则，将联合概率分解为一系列条件概率的乘积：

$$P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, w_2, \ldots, w_{i-1}) = \prod_{i=1}^{n} P(w_i \mid w_{<i})$$

其中 $w_{<i} = (w_1, w_2, \ldots, w_{i-1})$ 表示位置 $i$ 之前的所有词。

这个分解具有深刻的直觉意义：**每个词的生成只依赖于它之前的上下文**。这恰好对应人类书写文字的过程——我们从左到右逐词书写，每个词的选择都基于已写出的内容。

模型的训练目标是最大化训练语料的对数似然：

$$\mathcal{L} = \sum_{i=1}^{n} \log P(w_i \mid w_{<i}; \theta)$$

等价地，最小化负对数似然（交叉熵损失）：

$$\mathcal{L}_{\text{LM}} = -\frac{1}{n} \sum_{i=1}^{n} \log P(w_i \mid w_{<i}; \theta)$$

### 14.1.2 与BERT双向模型的对比

GPT和BERT代表了两种截然不同的预训练范式：

| 维度 | GPT（自回归） | BERT（自编码） |
|------|-------------|--------------|
| 建模方向 | 单向（从左到右） | 双向（全局上下文） |
| 训练目标 | 下一词预测 | Masked Language Model |
| 上下文范围 | 只能看到过去的词 | 可以看到前后的词 |
| 生成能力 | 天然支持文本生成 | 不直接支持生成 |
| 典型任务 | 文本生成、补全 | 分类、NER、QA |
| 信息利用 | 序列信息完整 | 每次只预测15%的词 |

**双向模型的优势**在于理解任务：对于"我爱北京[MASK]安门"，BERT可以同时利用"我爱北京"和"安门"来推断被掩码的词，上下文信息更丰富。

**自回归模型的优势**在于生成任务：文本生成天然是从左到右的过程，自回归模型与生成过程高度契合，不需要额外的解码策略。

### 14.1.3 因果掩码的必要性

为了实现自回归建模，GPT使用**因果掩码（Causal Mask）**，也称为**下三角掩码**。

在训练时，模型一次性接收整个序列，并行计算所有位置的预测。但是，位置 $i$ 的预测只能基于 $w_{<i}$，不能"偷看"未来的词。因果掩码通过将注意力矩阵的上三角部分设为 $-\infty$（在softmax后变为0），强制实现这一约束：

$$\text{Mask}[i, j] = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

带掩码的注意力计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 $M$ 是因果掩码矩阵。

**可视化示意**（4个词的序列）：

```
注意力矩阵（允许注意到的位置用1表示）：

        w1  w2  w3  w4
w1  [   1   0   0   0  ]
w2  [   1   1   0   0  ]
w3  [   1   1   1   0  ]
w4  [   1   1   1   1  ]
```

这保证了：训练时并行高效，推理时保持因果一致性。

### 14.1.4 生成能力

自回归模型的生成过程非常自然：

1. 给定提示（Prompt）$x_1, x_2, \ldots, x_k$
2. 计算 $P(x_{k+1} \mid x_1, \ldots, x_k)$，采样得到 $x_{k+1}$
3. 将 $x_{k+1}$ 加入序列，计算 $P(x_{k+2} \mid x_1, \ldots, x_{k+1})$
4. 重复直到生成结束标记或达到最大长度

这种**逐步自回归生成**（Autoregressive Generation）是GPT系列强大生成能力的基础。

---

## 14.2 GPT架构

### 14.2.1 只使用Decoder的设计

原始Transformer包含Encoder和Decoder两个组件。GPT的核心设计决策是**只使用Decoder部分**，并移除其中的交叉注意力层（Cross-Attention）。

这一设计的逻辑非常清晰：
- 语言建模不需要编码器——输入和输出都是同一个序列
- 移除交叉注意力后，每一层只有自注意力和前馈网络
- 配合因果掩码，每个位置只能注意到自己及之前的位置

GPT的整体结构：

```
输入文本
    ↓
Token Embedding + Position Embedding
    ↓
┌─────────────────────────────┐
│  Transformer Decoder Block  │ × N 层
│  ┌─────────────────────────┐│
│  │  Layer Norm (Pre-LN)    ││
│  │  Masked Self-Attention  ││
│  │  Residual Connection    ││
│  ├─────────────────────────┤│
│  │  Layer Norm (Pre-LN)    ││
│  │  Feed-Forward Network   ││
│  │  Residual Connection    ││
│  └─────────────────────────┘│
└─────────────────────────────┘
    ↓
Layer Norm
    ↓
Linear + Softmax（词表投影）
    ↓
下一词概率分布
```

### 14.2.2 Pre-LN与GELU

**Pre-LN（Pre-Layer Normalization）**

原始Transformer使用Post-LN，即在残差连接之后进行层归一化：

$$x_{l+1} = \text{LayerNorm}(x_l + \text{Sublayer}(x_l))$$

GPT-2引入Pre-LN，将层归一化移到子层之前：

$$x_{l+1} = x_l + \text{Sublayer}(\text{LayerNorm}(x_l))$$

Pre-LN的优势：
- 训练更稳定，梯度流动更好
- 不需要学习率预热（warm-up）也能稳定训练
- 在深层模型中表现更好

**GELU激活函数**

GPT用GELU（Gaussian Error Linear Unit）替代ReLU：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

近似计算：

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)$$

与ReLU相比，GELU在负值区域有平滑的梯度，在0附近不存在梯度突变，实践中通常带来更好的性能。

### 14.2.3 与原始Transformer Decoder的区别

| 组件 | 原始Transformer Decoder | GPT |
|------|------------------------|-----|
| 自注意力 | 有（带因果掩码） | 有（带因果掩码） |
| 交叉注意力 | 有（注意Encoder输出） | **无** |
| 前馈网络 | 有 | 有 |
| 层归一化位置 | Post-LN | **Pre-LN** |
| 激活函数 | ReLU | **GELU** |
| 位置编码 | 正弦位置编码 | **可学习位置嵌入** |

### 14.2.4 配置参数

GPT系列的主要超参数：

| 参数 | 含义 | GPT-1 | GPT-2 Small | GPT-2 Large | GPT-3 |
|------|------|-------|-------------|-------------|-------|
| $n_{\text{layer}}$ | Transformer层数 | 12 | 12 | 36 | 96 |
| $d_{\text{model}}$ | 隐藏层维度 | 768 | 768 | 1280 | 12288 |
| $n_{\text{head}}$ | 注意力头数 | 12 | 12 | 20 | 96 |
| $d_{\text{ff}}$ | 前馈网络维度 | 3072 | 3072 | 5120 | 49152 |
| $n_{\text{ctx}}$ | 最大上下文长度 | 512 | 1024 | 1024 | 2048 |
| 参数量 | 总参数数 | 117M | 117M | 774M | 175B |

---

## 14.3 GPT系列演进

### 14.3.1 GPT-1：预训练+微调范式（2018）

**论文**：*Improving Language Understanding by Generative Pre-Training*（Radford et al., 2018）

GPT-1的核心贡献是确立了**预训练-微调（Pre-train + Fine-tune）**范式：

**第一阶段：无监督预训练**

在大规模文本语料（BooksCorpus，约8亿词）上训练语言模型：

$$\mathcal{L}_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \ldots, u_{i-1}; \theta)$$

**第二阶段：有监督微调**

在下游任务的标注数据上微调，同时保留语言模型目标：

$$\mathcal{L}_3(\mathcal{C}) = \mathcal{L}_2(\mathcal{C}) + \lambda \cdot \mathcal{L}_1(\mathcal{C})$$

其中 $\mathcal{L}_2$ 是任务特定的监督损失，$\lambda$ 是权重系数。

**输入变换技巧**：不同NLP任务需要不同的输入格式，GPT-1设计了统一的输入变换方案：

- 分类任务：`[Start] 文本 [Extract]`
- 文本蕴含：`[Start] 前提 [Delim] 假设 [Extract]`
- 相似度：两个方向各一次
- 多项选择：每个选项单独构成一个输入

**成果**：在12个NLP任务中的9个达到了当时最优性能。

### 14.3.2 GPT-2：Zero-shot学习（2019）

**论文**：*Language Models are Unsupervised Multitask Learners*（Radford et al., 2019）

GPT-2的核心假设：**足够强大的语言模型可以在不微调的情况下完成下游任务**。

关键洞察：任何NLP任务都可以用自然语言描述，语言模型在预训练时隐式学习了这些任务。

**规模提升**：
- 参数量：117M → 1.5B（最大版本）
- 训练数据：BooksCorpus → WebText（约40GB，来自Reddit高质量链接）
- 词表：从 ~32K → 50,257（BPE编码）

**Zero-shot能力展示**：

```
任务：翻译
提示：将英文翻译成法文：
      sea otter => loutre de mer
      peppercorn => poivre
      cheese => fromage
      steak =>
模型输出：steak（无需微调，直接给出答案）
```

GPT-2在当时引发了广泛讨论，OpenAI甚至以"安全考虑"为由分阶段发布模型，这是大模型安全问题首次进入公众视野。

### 14.3.3 GPT-3：Few-shot与In-Context Learning（2020）

**论文**：*Language Models are Few-Shot Learners*（Brown et al., 2020）

GPT-3将规模提升至前所未有的程度，并系统性地研究了**上下文学习（In-Context Learning，ICL）**。

**三种学习范式**：

1. **Zero-shot**：只给任务描述，不给示例
2. **One-shot**：给一个示例
3. **Few-shot**：给几个示例（通常3-100个）

这些都在**推理阶段**完成，不更新任何模型参数。

### 14.3.4 参数量和数据规模对比

| 模型 | 发布时间 | 参数量 | 训练数据 | 上下文长度 | 关键特性 |
|------|---------|--------|---------|-----------|---------|
| GPT-1 | 2018.06 | 117M | 4.5GB | 512 | 预训练+微调 |
| GPT-2 | 2019.02 | 117M~1.5B | 40GB | 1024 | Zero-shot |
| GPT-3 | 2020.05 | 175B | 570GB（过滤后） | 2048 | Few-shot / ICL |
| GPT-3.5 | 2022.11 | ~175B | - | 4096 | RLHF微调 |
| GPT-4 | 2023.03 | 未公开 | 未公开 | 128K | 多模态、推理 |

**训练计算量（FLOPs）的增长**：

$$C \approx 6 \times N \times D$$

其中 $N$ 是参数量，$D$ 是训练tokens数。GPT-3使用约300B tokens，总计算量约 $3.14 \times 10^{23}$ FLOPs。

---

## 14.4 In-Context Learning

### 14.4.1 三种学习形式

**Zero-shot Learning**

不提供任何示例，只给任务指令：

```
指令：将以下评论分类为正面或负面。
评论：这部电影太精彩了！
分类：
```

**One-shot Learning**

提供一个示例：

```
指令：将以下评论分类为正面或负面。
示例：这家餐厅的服务很差。→ 负面
评论：这部电影太精彩了！
分类：
```

**Few-shot Learning**

提供多个示例（通常称为Demonstrations）：

```
指令：将以下评论分类为正面或负面。
评论：这家餐厅的服务很差。→ 负面
评论：这部电影节奏紧凑，演技出色。→ 正面
评论：包裹迟到了，客服态度也不好。→ 负面
评论：这部电影太精彩了！→
```

### 14.4.2 Prompt设计

Prompt的质量对ICL性能有显著影响。几个关键设计原则：

**示例选择**：
- 示例应与测试输入分布相近
- 示例应覆盖不同类别/情况
- 示例数量与质量之间存在权衡

**格式一致性**：
- 输入-输出格式要统一
- 标签词的选择影响性能（"正面/负面" vs "好/坏"）

**顺序效应**：
- 示例的排列顺序会影响结果
- 最后一个示例对结果的影响最大

**指令清晰度**：
- 明确的任务描述通常优于隐式描述
- Chain-of-Thought（CoT）提示可以大幅提升推理任务性能

Chain-of-Thought 示例：

```
问：Roger有5个网球。他又买了2罐网球，每罐3个。他现在有多少网球？
答：Roger一开始有5个网球。2罐，每罐3个，共6个球。5+6=11。答案是11。

问：食堂有23个苹果。如果他们用掉20个做午餐，又买了6个，现在有多少苹果？
答：
```

### 14.4.3 涌现能力

**涌现能力（Emergent Abilities）**是指在小模型中几乎不存在，但在模型规模超过某一阈值后突然出现的能力。

典型的涌现能力包括：

| 能力 | 大约出现的规模 |
|------|-------------|
| 3位数加法 | ~10B |
| 多步推理 | ~100B |
| 代码生成（有意义） | ~100B |
| 复杂逻辑推理 | ~100B |
| 指令跟随（无微调） | ~100B |

涌现能力的重要性：
- 它意味着扩大规模不仅是量变，还会产生质变
- 预测能力的出现时机非常困难
- 这是"Scaling Laws"研究的重要动机

**注意**：部分研究者认为"涌现"是度量标准选择的产物，在连续度量下，能力的提升可能是平滑的。这一争议仍在进行中。

### 14.4.4 ICL的工作原理假说

关于ICL为何有效，目前存在几种假说：

**假说一：梯度下降隐喻**

Akyürek et al. (2022) 和 von Oswald et al. (2022) 证明，Transformer的前向传播在数学上等价于对示例执行隐式梯度下降。即ICL是在激活值中"执行"了一次小型优化。

**假说二：任务识别**

Pan et al. (2023) 认为，GPT的主要能力是从示例中识别出任务，然后调用预训练中已经学到的对应技能。示例的主要作用是任务定位，而非传授新技能。

**假说三：贝叶斯推断**

从贝叶斯角度，ICL可以看作对任务先验的后验更新：

$$P(\text{output} \mid \text{input, demos}) \propto P(\text{demos} \mid \text{task}) \cdot P(\text{task}) \cdot P(\text{output} \mid \text{input, task})$$

**实践启示**：无论ICL的机制如何，它在大规模模型中确实有效，且为我们提供了一种无需微调就能适配新任务的强大工具。

---

## 14.5 GPT实现

### 14.5.1 GPTConfig配置类

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257      # GPT-2的词表大小
    block_size: int = 1024       # 最大上下文长度
    n_layer: int = 12            # Transformer层数
    n_head: int = 12             # 注意力头数
    n_embd: int = 768            # 嵌入维度
    dropout: float = 0.1         # Dropout概率
    bias: bool = True            # 线性层是否使用偏置

    @property
    def head_size(self):
        return self.n_embd // self.n_head

    @classmethod
    def gpt2_small(cls):
        return cls(n_layer=12, n_head=12, n_embd=768)

    @classmethod
    def gpt2_medium(cls):
        return cls(n_layer=24, n_head=16, n_embd=1024)

    @classmethod
    def gpt2_large(cls):
        return cls(n_layer=36, n_head=20, n_embd=1280)

    @classmethod
    def gpt2_xl(cls):
        return cls(n_layer=48, n_head=25, n_embd=1600)
```

### 14.5.2 GPT模型实现

完整的GPT模型实现（详见本章代码实战部分）。

### 14.5.3 文本生成策略

生成质量的关键在于**采样策略**的选择。

**贪心搜索（Greedy Search）**

每步选择概率最高的词：

$$w_t = \arg\max_{w} P(w \mid w_{<t})$$

优点：确定性，快速。缺点：容易陷入重复循环，缺乏多样性。

**Temperature采样**

通过温度参数 $T$ 调整分布的"尖锐程度"：

$$P_T(w) = \frac{\exp(\log P(w) / T)}{\sum_{w'} \exp(\log P(w') / T)}$$

- $T \to 0$：退化为贪心搜索
- $T = 1$：原始分布采样
- $T > 1$：分布更平坦，更多随机性（更有创意，但可能不连贯）
- $0 < T < 1$：分布更尖锐，更保守

**Top-k采样**

只从概率最高的 $k$ 个词中采样：

$$P_{\text{top-k}}(w) \propto P(w) \cdot \mathbf{1}[w \in \text{Top-k}]$$

问题：$k$ 是固定的，无法适应不同情况下概率分布的差异。

**Top-p（Nucleus）采样**

选择累积概率超过 $p$ 的最小词集合 $V^{(p)}$：

$$V^{(p)} = \arg\min_{V' \subseteq V} \left\{ |V'| : \sum_{w \in V'} P(w \mid w_{<t}) \geq p \right\}$$

Top-p能够自适应：分布尖锐时选少量词，分布平坦时选更多词，更加灵活。

**Beam Search**

维护 $B$ 个最优候选序列：

- 优点：比贪心搜索找到更高概率的序列
- 缺点：生成文本重复性高，对开放生成效果不好
- 适用场景：机器翻译、摘要等有标准答案的任务

---

## 本章小结

| 特性 | GPT-1 | GPT-2 | GPT-3 |
|------|-------|-------|-------|
| 参数量 | 117M | 117M~1.5B | 175B |
| 训练数据 | 4.5GB (BooksCorpus) | 40GB (WebText) | 570GB (过滤) |
| 核心范式 | 预训练+微调 | Zero-shot | Few-shot / ICL |
| 上下文长度 | 512 | 1024 | 2048 |
| 词表大小 | ~32K | 50,257 | 50,257 |
| 位置编码 | 可学习 | 可学习 | 可学习 |
| 层归一化 | Post-LN | Pre-LN | Pre-LN |
| 训练技巧 | — | 梯度裁剪 | 梯度裁剪、混合精度 |
| 主要贡献 | 确立预训练范式 | 证明Zero-shot可行性 | 证明规模是关键 |

GPT系列的演进揭示了一个核心规律：**规模即能力（Scale is all you need）**。随着模型规模和数据规模的提升，新的能力会涌现，这为后续的LLM研究（包括ChatGPT、GPT-4等）奠定了基础。

---

## 代码实战

```python
"""
GPT从零实现
包含：模型定义、文本生成、简单训练示例
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


# ============================================================
# 配置
# ============================================================

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True

    @classmethod
    def gpt2_small(cls):
        return cls(n_layer=12, n_head=12, n_embd=768)

    @classmethod
    def tiny(cls):
        """用于演示的微型配置"""
        return cls(
            vocab_size=256,
            block_size=128,
            n_layer=4,
            n_head=4,
            n_embd=128,
            dropout=0.1
        )


# ============================================================
# 模型组件
# ============================================================

class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Q, K, V投影（合并为一个矩阵提高效率）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果掩码（下三角矩阵）
        # 注册为buffer，不参与梯度计算，但会随模型保存
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # 计算Q、K、V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 重塑为多头形式 (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 注意力分数
        scale = 1.0 / math.sqrt(self.head_size)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, n_head, T, T)

        # 应用因果掩码
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Softmax + Dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 加权求和
        out = att @ v  # (B, n_head, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(out))


class MLP(nn.Module):
    """前馈网络（使用GELU激活）"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """GPT Transformer Block（Pre-LN设计）"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN：先归一化，再做子层，最后残差连接
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ============================================================
# GPT主模型
# ============================================================

class GPT(nn.Module):
    """GPT语言模型"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),   # token embedding
            'wpe': nn.Embedding(config.block_size, config.n_embd),   # position embedding
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd, bias=config.bias),
        })
        # 语言模型头：将隐藏状态映射到词表
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重绑定：嵌入层和LM头共享权重（节省参数，提升性能）
        self.transformer['wte'].weight = self.lm_head.weight

        # 参数初始化
        self.apply(self._init_weights)
        # 对残差投影使用特殊初始化（GPT-2做法）
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx: (B, T) token indices
        targets: (B, T) target token indices（训练时使用）
        返回：(logits, loss)
        """
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"序列长度 {T} 超过最大长度 {self.config.block_size}"

        device = idx.device

        # Token + Position Embedding
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)
        tok_emb = self.transformer['wte'](idx)   # (B, T, n_embd)
        pos_emb = self.transformer['wpe'](pos)   # (T, n_embd)
        x = self.transformer['drop'](tok_emb + pos_emb)

        # 依次通过所有Transformer块
        for block in self.transformer['h']:
            x = block(x)

        # 最终层归一化
        x = self.transformer['ln_f'](x)

        if targets is not None:
            # 训练模式：计算所有位置的logits和loss
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # 推理模式：只计算最后一个位置的logits（节省计算）
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    def get_num_params(self, non_embedding: bool = True) -> int:
        """返回参数量（默认不含位置嵌入）"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer['wpe'].weight.numel()
        return n_params

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str
    ) -> torch.optim.Optimizer:
        """配置AdamW优化器，对嵌入和偏置不应用权重衰减"""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # 维度>=2的参数（权重矩阵）应用权重衰减；其他（偏置、LN参数）不应用
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        use_fused = (device_type == 'cuda') and ('fused' in torch.optim.AdamW.__init__.__code__.co_varnames)
        extra_args = {'fused': True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer


# ============================================================
# 文本生成
# ============================================================

@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
) -> torch.Tensor:
    """
    自回归文本生成

    参数：
        model: GPT模型
        idx: (B, T) 输入token序列（prompt）
        max_new_tokens: 最多生成多少个新token
        temperature: 温度参数（>1更随机，<1更确定）
        top_k: Top-k采样的k值
        top_p: Top-p（Nucleus）采样的p值
        do_sample: True=采样，False=贪心

    返回：
        (B, T + max_new_tokens) 完整序列
    """
    model.eval()
    config = model.config

    for _ in range(max_new_tokens):
        # 如果序列超过最大长度，截断到最近的block_size个token
        idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size:]

        # 前向传播，获取最后位置的logits
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (B, vocab_size)

        # 应用Temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k截断
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Top-p（Nucleus）截断
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # 移除累积概率超过p的token
            sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_indices_to_remove] = float('-inf')
            # 还原到原始顺序
            logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

        # 转换为概率分布
        probs = F.softmax(logits, dim=-1)

        if do_sample:
            # 采样
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # 贪心
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        # 追加到序列
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# ============================================================
# 字符级简单训练示例
# ============================================================

def build_char_dataset(text: str):
    """构建字符级数据集"""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode, vocab_size


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """随机采样一个batch"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def train_demo():
    """
    字符级GPT训练演示（使用一段莎士比亚文本）
    """
    # 示例文本（实际使用时应加载更大语料）
    text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them.
    """ * 50  # 重复使文本足够长

    encode, decode, vocab_size = build_char_dataset(text)
    data = torch.tensor(encode(text), dtype=torch.long)

    # 训练/验证集分割
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"词表大小: {vocab_size}")
    print(f"训练集大小: {len(train_data)} 字符")

    # 使用微型配置
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1
    )
    model = GPT(config).to(device)
    print(f"模型参数量: {model.get_num_params():,}")

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=3e-4,
        betas=(0.9, 0.95),
        device_type=device
    )

    # 训练循环
    batch_size = 32
    max_iters = 1000
    eval_interval = 200

    for step in range(max_iters):
        # 评估
        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_data, config.block_size, batch_size, device)
                _, val_loss = model(xv, yv)
            print(f"Step {step}: val_loss = {val_loss.item():.4f}")

        # 训练步
        model.train()
        x, y = get_batch(train_data, config.block_size, batch_size, device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print("训练完成！")
    return model, encode, decode, config


# ============================================================
# 采样策略对比演示
# ============================================================

def compare_sampling_strategies(model: GPT, encode, decode, config: GPTConfig, prompt: str):
    """对比不同采样策略的生成效果"""
    device = next(model.parameters()).device
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt!r}")
    print(f"{'='*60}")

    strategies = [
        {"name": "贪心搜索",          "do_sample": False, "temperature": 1.0},
        {"name": "Temperature=0.5",   "do_sample": True,  "temperature": 0.5},
        {"name": "Temperature=1.0",   "do_sample": True,  "temperature": 1.0},
        {"name": "Temperature=1.5",   "do_sample": True,  "temperature": 1.5},
        {"name": "Top-k (k=10)",      "do_sample": True,  "temperature": 1.0, "top_k": 10},
        {"name": "Top-p (p=0.9)",     "do_sample": True,  "temperature": 1.0, "top_p": 0.9},
    ]

    for strategy in strategies:
        name = strategy.pop("name")
        torch.manual_seed(42)
        output = generate(model, context.clone(), max_new_tokens=50, **strategy)
        generated = decode(output[0].tolist()[len(encode(prompt)):])
        print(f"\n[{name}]")
        print(f"{prompt}{generated}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GPT从零实现演示")
    print("=" * 60)

    # 1. 训练一个字符级GPT
    model, encode, decode, config = train_demo()

    # 2. 对比采样策略
    compare_sampling_strategies(
        model, encode, decode, config,
        prompt="To be"
    )

    # 3. 展示模型结构
    print("\n模型结构：")
    print(model)

    # 4. 验证参数量计算
    total = model.get_num_params()
    print(f"\n总参数量（不含位置嵌入）: {total:,}")

    expected = (
        config.vocab_size * config.n_embd +              # wte（与lm_head共享）
        config.n_layer * (
            3 * config.n_embd * config.n_embd +          # c_attn
            config.n_embd * config.n_embd +              # c_proj
            4 * config.n_embd * config.n_embd +          # c_fc
            4 * config.n_embd * config.n_embd            # c_proj in MLP
        )
    )
    print(f"估算参数量（含偏置但粗略）: ~{expected:,}")
```

---

## 练习题

### 基础题

**题目1**（基础）：解释因果掩码的作用。在一个长度为5的序列中，位置3（从0开始）的token可以注意到哪些位置？请画出注意力掩码矩阵的第3行。

**题目2**（基础）：比较GPT-1、GPT-2、GPT-3在以下方面的区别：(a) 参数量；(b) 训练数据规模；(c) 主要推理范式。完成下表：

| 模型 | 参数量 | 训练数据 | 推理范式 |
|------|--------|---------|---------|
| GPT-1 | ? | ? | ? |
| GPT-2 | ? | ? | ? |
| GPT-3 | ? | ? | ? |

### 中级题

**题目3**（中级）：GPT使用Pre-LN而非Post-LN。请解释：(a) 两者的区别（写出公式）；(b) Pre-LN在训练稳定性方面的优势；(c) 在深层网络（如96层的GPT-3）中，为什么Pre-LN尤为重要？

**题目4**（中级）：分析以下三种采样策略的优缺点，并说明它们各自适合哪种应用场景：
- Temperature采样（T=0.3）
- Top-k采样（k=50）
- Top-p采样（p=0.9）

### 提高题

**题目5**（提高）：实现一个支持**Beam Search**的文本生成函数。函数签名如下：

```python
def beam_search(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    beam_size: int = 4,
) -> torch.Tensor:
    """
    使用Beam Search生成文本
    返回得分最高的序列
    """
    pass
```

要求：
- 维护beam_size个候选序列及其累积对数概率
- 在每一步扩展所有候选序列，保留得分最高的beam_size个
- 考虑序列长度归一化（length normalization）防止偏向短序列

---

## 练习答案

### 答案1

因果掩码的作用是**防止模型在预测位置 $i$ 时看到未来位置 $j > i$ 的信息**，从而维持自回归生成的因果性。

在长度为5的序列中，位置3（索引从0开始）可以注意到位置0、1、2、3，不能注意到位置4。

注意力掩码矩阵（$M$，第3行，值1表示允许注意，0表示被掩码）：

$$\text{掩码第3行} = [1, 1, 1, 1, 0]$$

在实际计算中，被掩码的位置被设为 $-\infty$，softmax后变为0：

```
         pos0  pos1  pos2  pos3  pos4
位置3的掩码：[  1     1     1     1     0  ]
对应掩码值：[  0     0     0     0    -∞  ]
```

---

### 答案2

| 模型 | 参数量 | 训练数据 | 推理范式 |
|------|--------|---------|---------|
| GPT-1 | 117M | 4.5GB（BooksCorpus） | 预训练+有监督微调 |
| GPT-2 | 117M~1.5B | ~40GB（WebText） | Zero-shot（不微调直接推理） |
| GPT-3 | 175B | ~570GB（过滤后） | Few-shot / In-Context Learning |

关键区别在于**推理范式**的演进：GPT-1依然需要针对每个任务微调参数；GPT-2尝试完全零样本；GPT-3则系统化地展示了通过在提示中提供少量示例即可适配任意任务，且无需更新参数。

---

### 答案3

**(a) 公式区别**

Post-LN（原始Transformer）：

$$x_{l+1} = \text{LayerNorm}(x_l + \text{Sublayer}(x_l))$$

Pre-LN（GPT-2及以后）：

$$x_{l+1} = x_l + \text{Sublayer}(\text{LayerNorm}(x_l))$$

区别在于归一化操作的位置：Post-LN在残差加法之后，Pre-LN在子层输入之前。

**(b) 训练稳定性优势**

在Pre-LN中，每个子层的输入已被归一化，不存在极大或极小值，梯度不容易爆炸或消失。更具体地：

- 梯度可以通过残差连接的"高速公路"直接传播到底层，因为加法路径 $x_l$ 没有经过任何归一化
- 训练初期（随机初始化时），子层输出接近零，Pre-LN能保持合理的梯度量级
- 不需要学习率预热（Warm-up），可以直接使用较大的学习率

**(c) 深层网络中的重要性**

在96层的GPT-3中，Post-LN会导致严重的梯度消失问题：每一层归一化操作都会改变梯度的尺度，96层的累积效应导致底层几乎没有梯度信号。Pre-LN的残差路径保持了梯度的直通通道，使深层网络的训练成为可能。实验表明，96层的Post-LN模型在没有精心设计的学习率调度时往往无法收敛。

---

### 答案4

**Temperature采样（T=0.3）**

- 优点：生成内容确定性强，质量稳定，语句连贯
- 缺点：多样性低，容易重复，创意不足
- 适合场景：代码补全、SQL生成、标准格式文本生成等需要准确性的任务

**Top-k采样（k=50）**

- 优点：避免采样到概率极低的词，实现简单
- 缺点：k是固定的，无法自适应概率分布；当分布平坦时，50个候选可能仍然很多；当分布尖锐时，50个候选可能过多
- 适合场景：聊天对话、故事生成等对质量和多样性都有要求的任务

**Top-p采样（p=0.9）**

- 优点：自适应地根据概率分布调整候选集大小；分布尖锐时选少量词（保证质量），分布平坦时选更多词（增加多样性）
- 缺点：实现稍复杂，需要排序操作
- 适合场景：创意写作、开放域对话等需要高质量且多样性兼顾的场景

**实践建议**：通常将Temperature（0.7~0.9）与Top-p（0.9~0.95）结合使用，这是目前最广泛采用的生成策略。

---

### 答案5

```python
def beam_search(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    beam_size: int = 4,
    length_penalty: float = 0.6,
) -> torch.Tensor:
    """
    使用Beam Search生成文本

    参数：
        model: GPT模型
        idx: (1, T) 输入序列（batch_size必须为1）
        max_new_tokens: 最多生成多少新token
        beam_size: beam数量
        length_penalty: 长度惩罚系数（>1倾向长序列，<1倾向短序列）

    返回：
        (1, T + new_len) 最优序列
    """
    model.eval()
    config = model.config
    device = idx.device

    assert idx.size(0) == 1, "Beam Search要求batch_size=1"

    # 初始化：所有beam从同一输入开始
    # beams: list of (序列tensor, 累积对数概率)
    beams = [(idx.clone(), 0.0)]
    completed = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            all_candidates = []

            for seq, score in beams:
                # 截断到最大长度
                seq_cond = seq if seq.size(1) <= config.block_size else seq[:, -config.block_size:]

                # 前向传播
                logits, _ = model(seq_cond)
                logits = logits[:, -1, :]  # (1, vocab_size)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (vocab_size,)

                # 取top beam_size个候选
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                for i in range(beam_size):
                    token = top_indices[i].unsqueeze(0).unsqueeze(0)  # (1, 1)
                    new_seq = torch.cat([seq, token], dim=1)
                    # 长度归一化的得分
                    new_len = new_seq.size(1)
                    new_score = score + top_log_probs[i].item()
                    normalized_score = new_score / (new_len ** length_penalty)
                    all_candidates.append((new_seq, new_score, normalized_score))

            # 按归一化得分排序，保留top beam_size个
            all_candidates.sort(key=lambda x: x[2], reverse=True)
            beams = [(seq, score) for seq, score, _ in all_candidates[:beam_size]]

    # 返回得分最高的序列
    best_seq, best_score = max(beams, key=lambda x: x[1] / (x[0].size(1) ** length_penalty))
    return best_seq


# 使用示例（配合前面的train_demo）：
# model, encode, decode, config = train_demo()
# prompt = "To be"
# device = next(model.parameters()).device
# context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
# output = beam_search(model, context, max_new_tokens=50, beam_size=4)
# print(decode(output[0].tolist()))
```

**关键实现要点**：

1. **候选维护**：每步维护 `beam_size` 个(序列, 累积对数概率)对
2. **扩展**：每个beam扩展出 `beam_size` 个新候选，共 `beam_size²` 个候选
3. **剪枝**：按归一化得分保留最优的 `beam_size` 个
4. **长度归一化**：用 `length_penalty` 防止偏向短序列（短序列的负对数概率累积更少）
5. **最终选择**：从所有beam中选归一化得分最高的序列返回
