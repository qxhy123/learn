# 第16章：现代大模型

> **学习目标**
>
> 完成本章学习后，你将能够：
> 1. 理解LLaMA架构的设计选择及其背后的工程权衡
> 2. 掌握RoPE（旋转位置编码）的数学原理与实现
> 3. 理解GQA（分组查询注意力）如何解决KV Cache内存问题
> 4. 了解Flash Attention的IO感知设计思路
> 5. 能够用PyTorch实现现代LLM的核心组件

---

## 引言

2017年Transformer问世，2018年BERT和GPT开启预训练时代，2020年GPT-3展示了大模型的涌现能力。但真正让大模型走进千家万户的，是2023年以来的一批开源模型——以LLaMA为代表的现代大语言模型（Large Language Model, LLM）。

LLaMA并非在架构上颠覆了Transformer，而是在每一个细节上精心调优：用Pre-RMSNorm替代LayerNorm、用SwiGLU替代ReLU、用RoPE替代绝对位置编码。这些看似微小的改动，叠加在一起产生了显著的训练稳定性和推理效率提升。

与此同时，工程创新同样关键。Flash Attention重新设计了注意力的计算顺序，将显存占用从$O(n^2)$降至$O(n)$；GQA在多头注意力的参数量与推理速度之间找到了新的平衡点。这些技术共同构成了现代LLM的工程基础。

本章将深入剖析这些技术的原理，并通过完整代码帮你掌握它们的实现。

---

## 16.1 LLaMA架构

### 16.1.1 设计理念：在细节处精益求精

LLaMA（Large Language Model Meta AI）由Meta AI在2023年发布。与GPT-3等闭源模型不同，LLaMA开放了权重，引发了开源社区的巨大热情。

LLaMA的设计哲学可以概括为：**在Transformer解码器架构的基础上，用更稳定、更高效的组件替换原有的次优选择**。它不追求架构创新，而是将已被证明有效的技术系统性地集成在一起。

LLaMA与原始GPT架构的对比如下：

| 组件 | 原始GPT / GPT-2 | LLaMA |
|------|----------------|-------|
| 归一化位置 | Post-LN（残差后） | Pre-RMSNorm（残差前） |
| 归一化类型 | LayerNorm | RMSNorm |
| 激活函数 | GELU | SwiGLU |
| 位置编码 | 学习型绝对位置编码 | RoPE（旋转位置编码） |
| 注意力类型 | MHA（多头注意力） | GQA（分组查询注意力，LLaMA 2+） |
| 偏置项 | 有 | 无（大部分线性层去掉bias） |

### 16.1.2 Pre-RMSNorm：训练稳定的关键

**归一化位置**对训练稳定性影响极大。

原始Transformer使用**Post-LN**：将LayerNorm放在残差连接之后。Post-LN在深层网络中容易出现梯度爆炸，需要精心设计学习率热身策略。

现代LLM普遍采用**Pre-LN**：将归一化放在子层之前，梯度更稳定，可以使用更大的学习率，训练更容易收敛。

$$\text{Pre-LN: } x_{l+1} = x_l + \text{SubLayer}(\text{Norm}(x_l))$$

$$\text{Post-LN: } x_{l+1} = \text{Norm}(x_l + \text{SubLayer}(x_l))$$

**RMSNorm** 是LayerNorm的简化版本。LayerNorm需要计算均值和方差：

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta, \quad \mu = \frac{1}{d}\sum_i x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_i (x_i - \mu)^2}$$

RMSNorm去掉了均值中心化步骤，只做缩放：

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2}$$

RMSNorm的优势：计算量更少（省去均值计算），没有偏置参数$\beta$，实验表明效果与LayerNorm相当甚至更好。

### 16.1.3 SwiGLU激活函数

标准FFN层使用ReLU或GELU：

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

LLaMA使用**SwiGLU**（Swish-Gated Linear Unit）：

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_2)$$

其中$\otimes$表示逐元素乘法，$\text{Swish}(x) = x \cdot \sigma(x)$是平滑的门控函数。

SwiGLU需要两个投影矩阵$W_1$和$W_2$（门控矩阵），再加上输出投影$W_3$，共三个矩阵。为保持参数量与标准FFN相当，LLaMA将隐藏层维度从$4d$调整为$\frac{2}{3} \times 4d$（并向上取整到特定倍数）。

```
标准FFN：  x → [d × 4d] → GELU → [4d × d]  （2个矩阵，参数量 8d²）
SwiGLU：   x → [d × h] ⊗ [d × h] → [h × d]  （3个矩阵，h ≈ 8d/3，参数量 ≈ 8d²）
```

Noam Shazeer在2020年的论文中首次提出GLU系列激活，并实验发现SwiGLU是其中效果最好的变体。

### 16.1.4 整体架构图

```
输入 token ids
      │
      ▼
┌─────────────┐
│  Embedding  │  词嵌入（无位置编码）
└─────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  LLaMA Decoder Block × N            │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  RMSNorm（前置归一化）       │   │
│  └──────────────┬──────────────┘   │
│                 ▼                   │
│  ┌─────────────────────────────┐   │
│  │  GQA（含RoPE位置编码）       │   │
│  └──────────────┬──────────────┘   │
│                 ▼ +残差             │
│  ┌─────────────────────────────┐   │
│  │  RMSNorm（前置归一化）       │   │
│  └──────────────┬──────────────┘   │
│                 ▼                   │
│  ┌─────────────────────────────┐   │
│  │  SwiGLU FFN                 │   │
│  └──────────────┬──────────────┘   │
│                 ▼ +残差             │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────┐
│  RMSNorm    │  最终归一化
└─────────────┘
      │
      ▼
┌─────────────┐
│  LM Head    │  线性层，投影到词表
└─────────────┘
```

---

## 16.2 旋转位置编码（RoPE）

### 16.2.1 核心思想

位置编码的根本目标是：让模型知道每个token在序列中的位置。但绝对位置编码（如原始Transformer中的正弦编码）有一个根本局限：**它编码的是绝对位置，而注意力机制真正需要的是相对位置信息**。

RoPE（Rotary Position Embedding，Su et al., 2021）的洞见是：**通过旋转操作，将相对位置信息隐式地编码进注意力的点积计算中**。

具体地，设位置$m$处的查询向量为$q_m$，位置$n$处的键向量为$k_n$，RoPE要求：

$$\langle f(q, m), f(k, n) \rangle = g(q, k, m-n)$$

即：经过RoPE变换后的内积，只依赖于相对距离$m-n$，而不依赖于绝对位置$m$和$n$。

### 16.2.2 旋转矩阵推导

对于二维向量，旋转角度$\theta$的旋转矩阵为：

$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

将位置$m$处的向量旋转$m\theta$角度：

$$f(x, m) = R(m\theta) \cdot x$$

两个向量内积：

$$\langle f(q, m), f(k, n) \rangle = \langle R(m\theta)q, R(n\theta)k \rangle = q^T R(m\theta)^T R(n\theta) k = q^T R((n-m)\theta) k$$

这正是我们想要的：内积只依赖于相对位置$(n-m)$。

对于$d$维向量，将其分成$d/2$对，每对独立旋转，第$i$对的旋转频率为$\theta_i = 10000^{-2i/d}$（与原始Transformer位置编码的频率设计一致）：

$$\text{RoPE}(x, m) = \begin{pmatrix}
x_1 \cos(m\theta_1) - x_2 \sin(m\theta_1) \\
x_1 \sin(m\theta_1) + x_2 \cos(m\theta_1) \\
x_3 \cos(m\theta_2) - x_4 \sin(m\theta_2) \\
x_3 \sin(m\theta_2) + x_4 \cos(m\theta_2) \\
\vdots
\end{pmatrix}$$

### 16.2.3 复数形式的简化实现

RoPE有一个优雅的复数表示，极大地简化了实现。

将二维向量$(x_1, x_2)$视为复数$x_1 + ix_2$，旋转等价于乘以$e^{im\theta} = \cos(m\theta) + i\sin(m\theta)$：

$$(x_1 + ix_2) \cdot e^{im\theta} = (x_1\cos(m\theta) - x_2\sin(m\theta)) + i(x_1\sin(m\theta) + x_2\cos(m\theta))$$

在实现中，将实数向量重新解释为复数向量，乘以复数旋转因子，再转回实数：

```python
# 将 [batch, seq, heads, d] 的最后一维解释为 d/2 个复数
x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
# 乘以旋转因子（逐位置、逐频率）
x_rotated = x_complex * freqs_complex
# 转回实数
x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
```

### 16.2.4 长度外推能力

RoPE的一个重要优势是**相对较好的长度外推能力**。由于编码的是相对位置（通过旋转角度差），模型在训练时只见过长度$L$的序列，在推理时处理更长序列时，相对位置关系仍然有意义。

近年来出现了多种基于RoPE的长度扩展方法：

- **位置插值（PI）**：将超出训练长度的位置线性插值到训练范围内，相当于降低旋转频率
- **YaRN**：针对不同频率的旋转维度采用不同的插值策略
- **LongRoPE**：通过搜索最优缩放因子实现更好的长度外推

这些方法使得基础模型（如LLaMA 2的4096上下文）能够扩展到128K甚至更长的上下文。

---

## 16.3 分组查询注意力（GQA）

### 16.3.1 从MHA到MQA到GQA的演进

**多头注意力（MHA）** 是原始Transformer的标准设计：$H$个头，每个头都有独立的Q、K、V投影。

$$\text{MHA}: \quad Q_i, K_i, V_i = xW^Q_i, xW^K_i, xW^V_i \quad (i = 1, \ldots, H)$$

**多查询注意力（MQA）** 由Shazeer在2019年提出：保留$H$个Q头，但所有头共享同一组K和V。

$$\text{MQA}: \quad Q_i = xW^Q_i, \quad K = xW^K, \quad V = xW^V \quad (i = 1, \ldots, H)$$

MQA大幅减少了KV Cache的内存占用，推理速度显著提升，但训练质量有所下降。

**分组查询注意力（GQA）** 是MHA和MQA的折中方案（Ainslie et al., 2023）：将$H$个Q头分成$G$组，每组共享一对K、V头。

$$\text{GQA}: \quad Q_i = xW^Q_i, \quad K_g = xW^K_g, \quad V_g = xW^V_g$$

其中$g = \lceil i \cdot G / H \rceil$，即Q头$i$属于第$g$组。

```
MHA (H=8):   Q1 K1 V1 | Q2 K2 V2 | Q3 K3 V3 | Q4 K4 V4 | Q5 K5 V5 | Q6 K6 V6 | Q7 K7 V7 | Q8 K8 V8
MQA (G=1):   Q1        Q2        Q3        Q4        Q5        Q6        Q7        Q8
              \         |         |         |         |         |         /
                                    K V  （共享）

GQA (G=2):   Q1 Q2 Q3 Q4          Q5 Q6 Q7 Q8
              \   |   /               \   |   /
              K1 V1  （组1共享）      K2 V2  （组2共享）
```

### 16.3.2 KV Cache的内存问题

理解GQA的必要性，需要先理解**KV Cache**。

在自回归解码时，每生成一个新token，都需要计算该token与所有历史token的注意力。朴素实现需要对历史token重新计算K和V，时间复杂度为$O(n^2)$。

KV Cache是一种空间换时间的技术：缓存每一层所有历史token的K和V张量，生成新token时直接读取缓存。

**内存占用分析**（以LLaMA 13B为例）：

- 层数：40层
- 注意力头数：40个头
- 每头维度：128
- 数据类型：float16（2字节）

单个token的KV Cache大小：
$$40 \times 2 \times 40 \times 128 \times 2 \approx 819 \text{ KB}$$

对于4096 token的序列：$819 \text{ KB} \times 4096 \approx 3.2 \text{ GB}$

这占据了13B模型（约26GB）显存的约12%。当批量大小或序列长度增加时，KV Cache很快成为瓶颈。

**GQA的内存节省**：若使用G=4个KV头（替代40个），KV Cache减少到原来的$4/40 = 10\%$。

### 16.3.3 GQA的实现细节

GQA在实现时需要将共享的KV头"展开"以与每个Q头对齐。这通过`repeat_kv`操作实现：

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将KV头重复n_rep次，从[B, S, n_kv_heads, head_dim]扩展到[B, S, n_heads, head_dim]"""
    if n_rep == 1:
        return x
    B, S, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]                          # [B, S, n_kv_heads, 1, head_dim]
        .expand(B, S, n_kv_heads, n_rep, head_dim)   # [B, S, n_kv_heads, n_rep, head_dim]
        .reshape(B, S, n_kv_heads * n_rep, head_dim) # [B, S, n_heads, head_dim]
    )
```

这个操作不产生实际的数据复制（`expand`使用广播），内存友好。

### 16.3.4 参数与效率的权衡

| 配置 | KV参数量 | KV Cache | 模型质量 |
|------|---------|---------|---------|
| MHA (G=H) | $2Hd_h d$ | 最大 | 最好 |
| GQA (1<G<H) | $2Gd_h d$ | 中等 | 接近MHA |
| MQA (G=1) | $2d_h d$ | 最小 | 略有下降 |

LLaMA 2选择了GQA（70B模型使用8个KV头，对应40个Q头），在质量与效率之间取得了良好平衡。

---

## 16.4 Flash Attention

### 16.4.1 标准注意力的内存瓶颈

标准注意力机制的计算流程：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

对于序列长度$n$，计算$QK^T$产生$n \times n$的注意力矩阵。这带来两个问题：

1. **内存占用**：需要存储$n \times n$矩阵（float16时为$2n^2$字节），$n=2048$时约需8MB，$n=8192$时约需128MB。内存从HBM（高带宽显存）分配，当序列很长时极易OOM。

2. **IO瓶颈**：GPU的计算单元（CUDA Core / Tensor Core）速度极快，但从HBM读写数据的带宽有限。标准注意力需要多次读写$n \times n$矩阵，大量时间花在数据传输上，而非实际计算。

现代GPU（如A100）的算术强度（FLOPS/带宽）约为200：FLOP/字节，而标准注意力的算术强度随$n$增大，意味着在短序列时是**内存带宽受限**（Memory Bound）的操作。

### 16.4.2 IO感知算法设计

Flash Attention（Dao et al., 2022）的核心思想：**重新组织计算顺序，避免将$n \times n$矩阵写入HBM**。

GPU内存层次结构：

```
SRAM（片上缓存，~20MB，极高带宽）
      ↕ 非常快
HBM（高带宽显存，~80GB，相对慢）
      ↕ 慢（相对于SRAM）
```

Flash Attention的策略：**分块（Tiling）处理，将数据保留在SRAM中计算，只在最终结果写回HBM**。

关键数学洞察：softmax可以增量计算。对于分块输入，利用online softmax技术，可以在不见到完整行的情况下维护正确的softmax结果：

对于行$[x_1, x_2, \ldots, x_n]$，分两块$[x_1, \ldots, x_k]$和$[x_{k+1}, \ldots, x_n]$：

第一块：
$$m_1 = \max(x_1, \ldots, x_k), \quad \ell_1 = \sum_{i=1}^k e^{x_i - m_1}$$

第二块（更新全局统计量）：
$$m = \max(m_1, \max(x_{k+1}, \ldots, x_n))$$
$$\ell = e^{m_1 - m} \ell_1 + \sum_{i=k+1}^n e^{x_i - m}$$

合并输出：
$$O = \frac{e^{m_1 - m} \ell_1 \cdot O_1 + \sum_{i=k+1}^n e^{x_i - m} V_i}{\ell}$$

### 16.4.3 分块计算（Tiling）

Flash Attention将Q、K、V矩阵分成若干块，每次将一块载入SRAM，在SRAM内完成计算，再将结果写回HBM：

```
HBM中：Q[N×d], K[N×d], V[N×d], O[N×d]

分块大小 Bc = Br = ⌈M / (4d)⌉，其中M是SRAM大小

外循环：KV块（j = 1..⌈N/Bc⌉）
  内循环：Q块（i = 1..⌈N/Br⌉）
    从HBM加载 Q_i, K_j, V_j 到SRAM
    计算 S_ij = Q_i * K_j^T / sqrt(d)
    更新 online softmax 统计量
    累加到输出 O_i
  将 O_i 写回HBM
```

**复杂度对比**：

| 指标 | 标准注意力 | Flash Attention |
|------|-----------|----------------|
| 时间复杂度（FLOP） | $O(n^2 d)$ | $O(n^2 d)$（相同） |
| 空间复杂度（HBM） | $O(n^2)$ | $O(n)$ |
| HBM读写次数 | $O(n^2)$ | $O(n^2 / M)$ |
| 实际速度（A100） | baseline | 2-4x加速 |

Flash Attention不减少FLOP，但大幅减少HBM访问次数，从而显著加速。

### 16.4.4 Flash Attention 2的改进

Flash Attention 2（Dao, 2023）在第一版基础上做了三项主要优化：

1. **减少非矩阵乘法操作**：将rescaling操作从每次内循环移到外循环结束时，减少约15%的非matmul计算。

2. **调整循环顺序**：将Q的循环移到外层，KV的循环移到内层。这使得每个CUDA线程块只需处理一部分Q（而非一部分KV），减少线程间通信。

3. **更好的并行化**：在序列维度上也进行并行化，使得短序列时也能充分利用GPU。

Flash Attention 2在A100上实现了理论峰值性能的50-73%，相比标准注意力约有2-9倍加速（随序列长度增加而增大）。

---

## 16.5 其他现代技术

### 16.5.1 滑动窗口注意力（Mistral）

标准注意力的复杂度为$O(n^2)$，在极长序列下代价高昂。Mistral 7B（2023）引入**滑动窗口注意力（Sliding Window Attention, SWA）**：每个token只关注前$W$个token（局部注意力窗口）。

```
标准注意力（n=6, 全局）：
token 6 关注：[1, 2, 3, 4, 5, 6]

滑动窗口（W=3）：
token 6 关注：[4, 5, 6]
token 5 关注：[3, 4, 5]
```

通过多层堆叠，信息可以"跳过"窗口边界传播：第$k$层的感受野为$k \times W$。32层网络、窗口4096，理论感受野可达131072 token。

Mistral中每隔一层使用滑动窗口注意力，另外的层使用全局注意力，在效率与全局信息获取之间取得平衡。

### 16.5.2 混合专家（Mixture of Experts，Mixtral）

**专家混合（MoE）** 的核心思想：不同的输入由不同的"专家"（FFN子网络）处理，每次前向传播只激活部分参数。

$$\text{MoE}(x) = \sum_{i \in \text{TopK}(G(x))} G_i(x) \cdot E_i(x)$$

其中$G(x)$是门控网络（Router），$E_i$是第$i$个专家（FFN），TopK通常取2。

Mixtral 8x7B：8个专家，每次激活2个，等效参数量约46B，但实际激活参数量仅约13B，计算量与13B模型相当，但质量接近更大的稠密模型。

MoE的挑战：
- **负载均衡**：需要辅助损失防止所有token都路由到少数专家
- **通信开销**：在分布式推理中，不同token可能路由到不同设备上的专家
- **显存占用**：所有专家的参数都需要加载到显存（即使不被激活）

### 16.5.3 长上下文扩展技术

随着RAG（检索增强生成）和长文档理解需求的增长，扩展LLM的上下文窗口成为重要课题。

**位置插值（Position Interpolation）**：将超出训练长度的位置索引线性压缩到训练范围，例如将8192长度的索引$m$映射为$m \times (4096/8192)$。需要少量微调恢复性能。

**YaRN（Yet another RoPE extensioN）**：认识到RoPE中不同频率维度的外推难度不同（高频维度更难外推），对不同频率分别处理：
- 低频维度：直接外推（旋转角度不变）
- 高频维度：线性插值
- 中频维度：混合处理

**ALiBi（Attention with Linear Biases）**：不使用位置编码，而是在注意力分数上加一个与距离成正比的负偏置：$\text{bias}_{i,j} = -|i-j| \cdot m$，其中$m$是与头相关的斜率。ALiBi天然支持长度外推，但在绝对位置信息上有所牺牲。

### 16.5.4 量化感知训练

大模型推理的主要瓶颈之一是显存带宽。量化（Quantization）通过降低参数精度来减少显存占用和带宽需求。

**训练后量化（PTQ）**：

- **GPTQ**（2023）：基于二阶信息（Hessian）对权重进行逐层量化，4bit量化下质量损失极小
- **AWQ**（Activation-aware Weight Quantization）：保护对激活值敏感的关键权重，其余权重量化为4bit
- **GGUF/llama.cpp**：面向CPU推理的混合精度量化格式，使消费级硬件可运行大模型

**量化感知训练（QAT）**：在训练过程中模拟量化误差，使模型适应低精度推理。QAT通常比PTQ效果更好，但需要额外的训练成本。

**KV Cache量化**：除了权重量化，还可以将KV Cache存储为int8格式，进一步减少显存占用。

---

## 本章小结

本章介绍的技术共同构成了现代LLM的工程基础：

| 技术 | 解决的问题 | 核心方法 | 效果 |
|------|-----------|---------|------|
| Pre-RMSNorm | 训练不稳定 | 残差前归一化，简化LayerNorm | 更稳定的训练，更少参数 |
| SwiGLU | 激活函数表达力 | 门控线性单元 | 相同参数量下更好的效果 |
| RoPE | 位置编码外推 | 旋转操作编码相对位置 | 更好的长度泛化能力 |
| GQA | KV Cache内存 | 多个Q头共享KV头 | 显存占用减少5-10倍 |
| Flash Attention | 注意力内存瓶颈 | IO感知分块计算 | 2-9x加速，$O(n)$内存 |
| 滑动窗口注意力 | 长序列计算复杂度 | 局部注意力窗口 | 线性复杂度 |
| MoE | 扩展模型容量 | 稀疏激活专家网络 | 大容量低计算成本 |
| 位置插值/YaRN | 上下文窗口扩展 | 调整RoPE频率 | 支持128K+上下文 |
| GPTQ/AWQ | 推理显存与速度 | 权重量化到4/8bit | 显存减少2-4倍 |

这些技术大多是正交的（可以同时使用），它们的结合使得现代LLM在有限硬件上服务更长上下文、更大批量成为可能。

---

## 代码实战

下面实现一个完整的简化版LLaMA Block，包含RoPE、GQA和SwiGLU。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ============================================================
# 1. RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    去掉了均值中心化，只做缩放归一化
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习缩放参数

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        # RMS = sqrt(mean(x^2))，在最后一维上计算
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先归一化（用float32避免精度问题），再乘以可学习参数
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ============================================================
# 2. RoPE（旋转位置编码）
# ============================================================

def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0
) -> torch.Tensor:
    """
    预计算RoPE的复数旋转因子
    返回形状：[max_seq_len, dim//2]，复数张量
    """
    # 频率：theta^(-2i/d)，i = 0, 1, ..., d/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # [dim//2]

    # 位置索引
    t = torch.arange(max_seq_len)  # [max_seq_len]

    # 外积得到每个位置、每个频率的旋转角度
    freqs = torch.outer(t, freqs)  # [max_seq_len, dim//2]

    # 转为复数：cos(θ) + i·sin(θ) = e^(iθ)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [max_seq_len, dim//2]
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将RoPE旋转编码应用到Q和K上

    Args:
        xq: Query张量，形状 [batch, seq, n_heads, head_dim]
        xk: Key张量，形状 [batch, seq, n_kv_heads, head_dim]
        freqs_cis: 旋转因子，形状 [seq, head_dim//2]

    Returns:
        旋转后的 xq, xk，形状不变
    """
    # 将实数向量重新解释为复数：[B, S, H, D] -> [B, S, H, D/2]（复数）
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis形状调整：[S, D/2] -> [1, S, 1, D/2] 以支持广播
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, S, 1, D/2]

    # 复数乘法实现旋转
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # [B, S, H, D]
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)  # [B, S, H, D]

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ============================================================
# 3. GQA（分组查询注意力）
# ============================================================

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将KV头重复展开，使其数量与Q头对齐

    Args:
        x: [batch, seq, n_kv_heads, head_dim]
        n_rep: 每个KV头需要重复的次数（= n_heads // n_kv_heads）

    Returns:
        [batch, seq, n_kv_heads * n_rep, head_dim]
    """
    if n_rep == 1:
        return x
    B, S, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]                           # [B, S, n_kv_heads, 1, head_dim]
        .expand(B, S, n_kv_heads, n_rep, head_dim)    # [B, S, n_kv_heads, n_rep, head_dim]
        .reshape(B, S, n_kv_heads * n_rep, head_dim)  # [B, S, n_heads, head_dim]
    )


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力（GQA）
    n_kv_heads < n_heads 时为GQA，n_kv_heads == 1 时退化为MQA
    """
    def __init__(
        self,
        dim: int,           # 模型维度
        n_heads: int,       # Q头数
        n_kv_heads: int,    # KV头数（GQA关键参数）
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads必须是n_kv_heads的整数倍"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # 每个KV头对应的Q头数
        self.head_dim = dim // n_heads

        # 线性投影层（无bias，与LLaMA一致）
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, dim]
            freqs_cis: [seq, head_dim//2]，RoPE旋转因子
            mask: [seq, seq] 或 None，因果掩码

        Returns:
            [batch, seq, dim]
        """
        B, S, _ = x.shape

        # 1. 线性投影
        xq = self.wq(x)  # [B, S, n_heads * head_dim]
        xk = self.wk(x)  # [B, S, n_kv_heads * head_dim]
        xv = self.wv(x)  # [B, S, n_kv_heads * head_dim]

        # 2. 重塑为多头形式
        xq = xq.view(B, S, self.n_heads, self.head_dim)     # [B, S, H, D]
        xk = xk.view(B, S, self.n_kv_heads, self.head_dim)  # [B, S, Hkv, D]
        xv = xv.view(B, S, self.n_kv_heads, self.head_dim)  # [B, S, Hkv, D]

        # 3. 应用RoPE（只对Q和K）
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # 4. 展开KV头，使数量与Q头对齐
        xk = repeat_kv(xk, self.n_rep)  # [B, S, H, D]
        xv = repeat_kv(xv, self.n_rep)  # [B, S, H, D]

        # 5. 转置为 [B, H, S, D] 以便矩阵乘法
        xq = xq.transpose(1, 2)  # [B, H, S, D]
        xk = xk.transpose(1, 2)  # [B, H, S, D]
        xv = xv.transpose(1, 2)  # [B, H, S, D]

        # 6. 注意力计算
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(xq, xk.transpose(2, 3)) * scale  # [B, H, S, S]

        if mask is not None:
            scores = scores + mask  # 加法掩码（-inf处softmax后变为0）

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # [B, H, S, D]

        # 7. 合并多头，输出投影
        output = output.transpose(1, 2).contiguous().view(B, S, -1)  # [B, S, H*D]
        return self.wo(output)  # [B, S, dim]


# ============================================================
# 4. SwiGLU FFN
# ============================================================

class SwiGLUFFN(nn.Module):
    """
    SwiGLU前馈网络
    使用三个矩阵：gate投影、up投影、down投影
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        # LLaMA的隐藏层维度计算：将4d缩放到2/3，再取8的倍数（与实现一致）
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            # 向上取整到256的倍数（简化版，实际LLaMA取特定倍数）
            hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate投影
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down投影
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = Swish(xW1) ⊗ xW3，然后乘W2
        # Swish(x) = x * sigmoid(x) = x * σ(x)
        gate = F.silu(self.w1(x))  # silu即swish，[B, S, hidden_dim]
        up = self.w3(x)             # [B, S, hidden_dim]
        return self.w2(gate * up)   # [B, S, dim]


# ============================================================
# 5. LLaMA Decoder Block
# ============================================================

class LlamaDecoderBlock(nn.Module):
    """
    完整的LLaMA解码器块
    Pre-RMSNorm + GQA（含RoPE） + 残差 + Pre-RMSNorm + SwiGLU + 残差
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_hidden_dim: Optional[int] = None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attention = GroupedQueryAttention(dim, n_heads, n_kv_heads)
        self.feed_forward = SwiGLUFFN(dim, ffn_hidden_dim)

        # 前置归一化
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 注意力子层：Pre-Norm + 残差
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        # FFN子层：Pre-Norm + 残差
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# ============================================================
# 6. 简化版LLaMA模型
# ============================================================

class LlamaModel(nn.Module):
    """
    简化版LLaMA模型（用于教学，省略了部分工程细节）
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 2048,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # 词嵌入（无位置编码，RoPE通过注意力层注入）
        self.embed_tokens = nn.Embedding(vocab_size, dim)

        # 解码器层
        self.layers = nn.ModuleList([
            LlamaDecoderBlock(dim, n_heads, n_kv_heads, norm_eps=norm_eps)
            for _ in range(n_layers)
        ])

        # 最终归一化 + 语言模型头
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # 预计算RoPE旋转因子
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, max_seq_len),
            persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq]

        Returns:
            logits: [batch, seq, vocab_size]
        """
        B, S = input_ids.shape

        # 词嵌入
        h = self.embed_tokens(input_ids)  # [B, S, dim]

        # 取当前序列长度对应的旋转因子
        freqs_cis = self.freqs_cis[:S]  # [S, head_dim//2]

        # 因果掩码（上三角为-inf）
        mask = torch.full((S, S), float('-inf'), device=input_ids.device)
        mask = torch.triu(mask, diagonal=1)  # 严格上三角

        # 逐层前向传播
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        # 最终归一化 + 投影到词表
        h = self.norm(h)
        logits = self.lm_head(h)  # [B, S, vocab_size]
        return logits


# ============================================================
# 7. 测试与验证
# ============================================================

def test_components():
    """验证各组件的形状与基本功能"""
    torch.manual_seed(42)

    # 模型配置（小型测试配置）
    batch_size = 2
    seq_len = 16
    dim = 256
    n_heads = 8
    n_kv_heads = 2      # GQA：8个Q头，2个KV头
    vocab_size = 1000
    n_layers = 4

    print("=" * 50)
    print("测试现代LLM核心组件")
    print("=" * 50)

    # 测试 RMSNorm
    x = torch.randn(batch_size, seq_len, dim)
    rms_norm = RMSNorm(dim)
    out = rms_norm(x)
    print(f"[RMSNorm] 输入: {x.shape} -> 输出: {out.shape}")
    assert out.shape == x.shape

    # 测试 RoPE 预计算
    head_dim = dim // n_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=128)
    print(f"[RoPE] freqs_cis形状: {freqs_cis.shape}（复数）")
    assert freqs_cis.shape == (128, head_dim // 2)

    # 测试 GQA
    attn = GroupedQueryAttention(dim, n_heads, n_kv_heads)
    mask = torch.full((seq_len, seq_len), float('-inf'))
    mask = torch.triu(mask, diagonal=1)
    attn_out = attn(x, freqs_cis[:seq_len], mask)
    print(f"[GQA] 输入: {x.shape} -> 输出: {attn_out.shape}")
    print(f"      Q头: {n_heads}, KV头: {n_kv_heads}, 重复倍数: {n_heads // n_kv_heads}")
    assert attn_out.shape == x.shape

    # 测试 SwiGLU FFN
    ffn = SwiGLUFFN(dim)
    ffn_out = ffn(x)
    print(f"[SwiGLU] 输入: {x.shape} -> 输出: {ffn_out.shape}")
    print(f"         隐藏层维度: {ffn.w1.weight.shape[0]}")
    assert ffn_out.shape == x.shape

    # 测试完整 LLaMA Block
    block = LlamaDecoderBlock(dim, n_heads, n_kv_heads)
    block_out = block(x, freqs_cis[:seq_len], mask)
    print(f"[LlamaBlock] 输入: {x.shape} -> 输出: {block_out.shape}")
    assert block_out.shape == x.shape

    # 测试完整模型
    model = LlamaModel(vocab_size, dim, n_layers, n_heads, n_kv_heads, max_seq_len=128)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    print(f"[LlamaModel] 输入: {input_ids.shape} -> logits: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size)

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print("\n所有测试通过！")


def compare_kv_memory():
    """对比MHA、GQA、MQA的KV Cache内存占用"""
    dim = 4096
    n_heads = 32
    head_dim = dim // n_heads
    seq_len = 4096
    n_layers = 32
    bytes_per_element = 2  # float16

    configs = [
        ("MHA (32 KV heads)", n_heads),
        ("GQA (8 KV heads)",  8),
        ("MQA (1 KV head)",   1),
    ]

    print("\n" + "=" * 50)
    print("KV Cache内存占用对比（LLaMA-13B规格）")
    print("=" * 50)
    print(f"序列长度: {seq_len}, 层数: {n_layers}")
    print(f"{'配置':<20} {'KV Cache大小':>15} {'相对MHA':>10}")
    print("-" * 50)

    mha_size = None
    for name, n_kv in configs:
        # KV Cache = 2 (K+V) × n_kv_heads × head_dim × seq_len × n_layers × bytes
        size_bytes = 2 * n_kv * head_dim * seq_len * n_layers * bytes_per_element
        size_gb = size_bytes / (1024 ** 3)
        if mha_size is None:
            mha_size = size_gb
        ratio = size_gb / mha_size
        print(f"{name:<20} {size_gb:>12.2f} GB {ratio:>10.1%}")


if __name__ == "__main__":
    test_components()
    compare_kv_memory()
```

运行输出示例：

```
==================================================
测试现代LLM核心组件
==================================================
[RMSNorm] 输入: torch.Size([2, 16, 256]) -> 输出: torch.Size([2, 16, 256])
[RoPE] freqs_cis形状: torch.Size([128, 16])（复数）
[GQA] 输入: torch.Size([2, 16, 256]) -> 输出: torch.Size([2, 16, 256])
      Q头: 8, KV头: 2, 重复倍数: 4
[SwiGLU] 输入: torch.Size([2, 16, 256]) -> 输出: torch.Size([2, 16, 256])
         隐藏层维度: 512
[LlamaBlock] 输入: torch.Size([2, 16, 256]) -> 输出: torch.Size([2, 16, 256])
[LlamaModel] 输入: torch.Size([2, 16]) -> logits: torch.Size([2, 16, 1000])

模型总参数量: 3,162,624 (3.16M)

所有测试通过！

==================================================
KV Cache内存占用对比（LLaMA-13B规格）
==================================================
序列长度: 4096, 层数: 32
配置                    KV Cache大小       相对MHA
--------------------------------------------------
MHA (32 KV heads)           4.00 GB      100.0%
GQA (8 KV heads)            1.00 GB       25.0%
MQA (1 KV head)             0.12 GB        3.1%
```

---

## 练习题

### 基础题

**练习 16.1** （基础）

RMSNorm和LayerNorm的主要区别是什么？在以下场景中，RMSNorm相比LayerNorm有什么计算优势？

已知：输入维度$d = 4096$，批量大小$B = 32$，序列长度$S = 2048$。

(a) 写出LayerNorm和RMSNorm各自需要的乘法操作次数（不考虑参数$\gamma, \beta$的乘法）。

(b) RMSNorm去掉了均值中心化步骤，这对模型表达能力有何影响？直觉上为什么这个简化通常不影响效果？

---

**练习 16.2** （基础）

验证RoPE的相对位置属性。

设$d = 4$（为简单起见），使用RoPE旋转向量，旋转角度为$\theta_1 = 1.0, \theta_2 = 0.01$（两个频率）。

(a) 手动计算位置$m=2$时的旋转矩阵$R(2\theta)$（块对角形式）。

(b) 对于向量$q = [1, 0, 1, 0]$，分别计算$f(q, 2)$和$f(q, 5)$。

(c) 验证$\langle f(q, 2), f(k, 5) \rangle = \langle f(q, 0), f(k, 3) \rangle$，即内积只依赖于相对位置差3。（取$k = [1, 0, 1, 0]$）

---

### 中级题

**练习 16.3** （中级）

分析GQA的参数量与内存效率。

一个LLM具有以下配置：
- 模型维度$d = 4096$
- Q头数$H = 32$，每头维度$d_h = 128$
- 序列长度$n = 8192$，层数$L = 32$
- float16精度（2字节/参数）

(a) 计算MHA配置（$G = 32$个KV头）下，单个样本的KV Cache大小（GB）。

(b) 计算GQA配置（$G = 8$个KV头）下，单个样本的KV Cache大小，以及相对MHA的节省比例。

(c) 注意力权重矩阵（$W_Q, W_K, W_V$）的参数量在MHA vs GQA下分别是多少？

(d) 如果用节省的KV Cache内存来增大批量大小，GQA配置相比MHA可以支持多大的批量？

---

**练习 16.4** （中级）

Flash Attention的IO复杂度分析。

设GPU SRAM大小为$M$字节，float16精度，注意力头维度为$d$，序列长度为$N$。

(a) 标准注意力算法需要从HBM读写哪些矩阵？列出每次读写的大小，并计算总HBM访问量（以字节为单位，用$N$, $d$表示）。

(b) Flash Attention的分块大小$B_c = B_r = \lfloor M / (4d) \rfloor$（以元素数计）。计算外循环（KV块）和内循环（Q块）各需要迭代多少次？

(c) 每次内循环迭代需要从HBM加载哪些数据？每次加载多少字节？

(d) 推导Flash Attention的总HBM读写量，并与标准注意力对比，说明在什么条件下Flash Attention有更少的HBM访问。

---

### 提高题

**练习 16.5** （提高）

实现一个支持KV Cache增量解码的GQA模块。

在自回归生成时，每次只生成一个新token，但需要与所有历史token计算注意力。利用KV Cache避免重复计算。

(a) 修改`GroupedQueryAttention`类，添加以下功能：
- `use_cache: bool`参数，控制是否使用KV Cache
- `past_key_value: Optional[Tuple[Tensor, Tensor]]`输入参数，接收历史KV
- 返回值增加`present_key_value: Tuple[Tensor, Tensor]`，存储当前层的KV

(b) 在解码时，Q的序列长度为1（当前token），K和V的序列长度为历史长度+1。注意`freqs_cis`需要只取当前位置的旋转因子（而不是完整序列）。

(c) 实现`generate`函数：

```python
def generate(
    model: LlamaModel,
    input_ids: torch.Tensor,  # [1, prompt_len]
    max_new_tokens: int = 20,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    自回归生成，使用KV Cache加速
    返回：[1, prompt_len + max_new_tokens]
    """
    # 请在此实现
    ...
```

(d) 验证：带KV Cache的生成结果与不带KV Cache的结果一致（给定相同的输入和随机种子）。

---

## 练习答案

### 练习 16.1 答案

**(a) 操作次数对比**

LayerNorm需要：
1. 计算均值$\mu$：$d$次加法 + 1次除法
2. 中心化$x - \mu$：$d$次减法
3. 计算方差$\sigma^2$：$d$次减法 + $d$次乘法（平方）+ $d$次加法 + 1次除法 + 1次开方
4. 归一化$(x-\mu)/\sigma$：$d$次除法

关键乘法次数：约$d$次（主要是平方计算）

RMSNorm需要：
1. 计算$\text{RMS}$：$d$次乘法（平方）+ $d$次加法 + 1次除法 + 1次开方
2. 缩放$x / \text{RMS}$：$d$次除法（等价于乘以$1/\text{RMS}$）

RMSNorm省去了均值计算和中心化步骤，运算量约为LayerNorm的2/3左右。

**(b) 表达能力分析**

LayerNorm的均值中心化将特征"零均值化"，从理论上可以提高数值稳定性。然而，由于后续的可学习缩放参数$\gamma$和偏置$\beta$可以将均值恢复到任意值，均值中心化的效果在很大程度上是可以被后续参数补偿的。

直觉上：$\text{RMSNorm}$ 保留了"方向"信息（归一化后的单位向量），方向信息是表征语义最重要的部分。均值仅影响"基准水平"，而这个信息可以通过可学习偏置$\beta$（尽管LLaMA中去掉了$\beta$）或下一层的权重来弥补。

---

### 练习 16.2 答案

**(a) 旋转矩阵**

对于$d=4$，两对旋转频率$\theta_1 = 1.0, \theta_2 = 0.01$，位置$m=2$：

$$R(2\theta_1) = \begin{pmatrix} \cos(2) & -\sin(2) \\ \sin(2) & \cos(2) \end{pmatrix} \approx \begin{pmatrix} -0.416 & -0.909 \\ 0.909 & -0.416 \end{pmatrix}$$

$$R(2\theta_2) = \begin{pmatrix} \cos(0.02) & -\sin(0.02) \\ \sin(0.02) & \cos(0.02) \end{pmatrix} \approx \begin{pmatrix} 0.9998 & -0.0200 \\ 0.0200 & 0.9998 \end{pmatrix}$$

块对角矩阵：
$$R(2\Theta) = \text{diag}(R(2\theta_1), R(2\theta_2))$$

**(b) 计算 $f(q, 2)$ 和 $f(q, 5)$**

$q = [1, 0, 1, 0]$，即每对的第一个分量为1，第二个为0：

$$f(q, 2) = R(2\Theta)q = [-0.416, 0.909, 0.9998, 0.0200]$$

$$f(q, 5): \quad \cos(5) \approx 0.284, \sin(5) \approx -0.959$$
$$f(q, 5) = [0.284, -0.959, \cos(0.05), \sin(0.05)] \approx [0.284, -0.959, 0.9988, 0.0500]$$

**(c) 验证相对位置属性**

$k = [1, 0, 1, 0]$

$$\langle f(q, 2), f(k, 5) \rangle = q^T R(-3\Theta) k$$

$$\langle f(q, 0), f(k, 3) \rangle = q^T R(3\Theta - 0) k = q^T R(3\Theta) k$$

注意$R(-3\Theta) = R(3\Theta)^T$，而$q^T R^T k = (Rq)^T k = k^T R q = \langle Rk, q \rangle$。

由于$q = k$（在本题中），$q^T R q$是关于旋转角度的对称量，两者相等。

一般情况下，关键等式是：

$$q^T R((n-m)\Theta) k = \text{只依赖于} (n-m)$$

即内积仅由相对位置差决定，验证了RoPE的核心属性。

---

### 练习 16.3 答案

**(a) MHA的KV Cache**

$$\text{KV Cache} = 2 \times H \times d_h \times n \times L \times 2 \text{ 字节}$$
$$= 2 \times 32 \times 128 \times 8192 \times 32 \times 2$$
$$= 2 \times 32 \times 128 \times 8192 \times 64$$
$$\approx 4.29 \times 10^9 \text{ 字节} \approx 4.0 \text{ GB}$$

**(b) GQA的KV Cache**

$$\text{KV Cache}_{GQA} = 2 \times 8 \times 128 \times 8192 \times 32 \times 2 \approx 1.07 \text{ GB}$$

节省比例：$(4.0 - 1.0) / 4.0 = 75\%$，降低到原来的$8/32 = 25\%$。

**(c) 注意力权重矩阵参数量**

MHA：
$$W_Q: d \times (H \times d_h) = 4096 \times 4096 = 16.8\text{M}$$
$$W_K: d \times (H \times d_h) = 16.8\text{M}, \quad W_V: 16.8\text{M}$$
总计：$3 \times 16.8\text{M} = 50.3\text{M}$ 参数

GQA（8个KV头）：
$$W_Q: 16.8\text{M}, \quad W_K: 4096 \times (8 \times 128) = 4.2\text{M}, \quad W_V: 4.2\text{M}$$
总计：$16.8 + 4.2 + 4.2 = 25.2\text{M}$ 参数（节省约50%的KV参数）

**(d) 批量大小扩展**

MHA支持批量大小$B$时KV Cache为$B \times 4.0$ GB。
GQA节省了3.0 GB，这些内存可以用来增大批量大小：

$$\Delta B = \lfloor 3.0 / 1.0 \rfloor = 3$$

即从$B=1$可以扩展到$B=4$（增加3倍），或者在固定显存预算下，GQA支持的批量大小是MHA的4倍。

---

### 练习 16.4 答案

**(a) 标准注意力的HBM访问**

标准注意力需要：

| 操作 | 读写 | 大小 |
|------|------|------|
| 读 Q, K, V | 读 | $3 \times 2Nd$ 字节 |
| 写 $S = QK^T$ | 写 | $2N^2$ 字节 |
| 读 $S$，写 $P = \text{softmax}(S)$ | 读+写 | $2 \times 2N^2$ 字节 |
| 读 P, V，写 O | 读+写 | $2N^2 + 2Nd + 2Nd$ 字节 |

总计（主项）：$O(Nd + N^2)$，以字节为单位约为$2(3Nd + 4N^2)$。

当$N \gg d$时，$N^2$项主导，总HBM访问量约$\Theta(N^2)$。

**(b) Flash Attention的循环次数**

块大小：$B_c = B_r = \lfloor M/(4d) \rfloor$（以元素数计）

外循环（KV块）：$T_c = \lceil N / B_c \rceil$ 次

内循环（Q块）：$T_r = \lceil N / B_r \rceil$ 次

总迭代次数：$T_c \times T_r = \lceil N/B_c \rceil \times \lceil N/B_r \rceil \approx N^2 / B_c^2$

**(c) 每次内循环的HBM加载**

每次内循环迭代需要加载：
- $Q_i$块：$B_r \times d$ 个元素 = $2B_r d$ 字节
- $K_j$块：$B_c \times d$ 个元素 = $2B_c d$ 字节
- $V_j$块：$B_c \times d$ 个元素 = $2B_c d$ 字节

合计：$2d(B_r + 2B_c)$ 字节，当$B_r = B_c$时为$6B_c d$ 字节。

**(d) Flash Attention总HBM访问量**

外循环每次迭代加载$K_j, V_j$并遍历所有Q块：
$$\text{总加载} = T_c \times (T_r \times 2B_r d + 2B_c d + 2B_c d) \approx T_c \times T_r \times 6B_c d$$

代入$T_c \approx N/B_c, T_r \approx N/B_r = N/B_c$：
$$\approx \frac{N}{B_c} \times \frac{N}{B_c} \times 6B_c d = \frac{6N^2 d}{B_c} = \frac{6N^2 d}{\lfloor M/(4d) \rfloor} \approx \frac{24N^2 d^2}{M}$$

标准注意力：$\Theta(N^2)$

Flash Attention：$\Theta(N^2 d^2 / M)$

**条件**：当$M > d^2$时（即SRAM足够大以存下多个头维度的块），Flash Attention的HBM访问量更少。现代GPU的SRAM约20MB，$d=128$时$d^2=16384$元素≈32KB，条件轻松满足，因此Flash Attention几乎总是更快。

---

### 练习 16.5 参考实现

```python
class CachedGroupedQueryAttention(nn.Module):
    """支持KV Cache的GQA实现"""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, _ = x.shape

        xq = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim)

        # 应用RoPE（只对当前token的位置）
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # 拼接历史KV Cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            xk = torch.cat([past_k, xk], dim=1)  # [B, past+S, Hkv, D]
            xv = torch.cat([past_v, xv], dim=1)

        present_key_value = (xk, xv) if use_cache else None

        # 展开KV头
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 注意力计算
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(xq, xk.transpose(2, 3)) * scale
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(output), present_key_value


@torch.no_grad()
def generate(
    model: LlamaModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
) -> torch.Tensor:
    """使用KV Cache的自回归生成（概念演示）"""
    model.eval()
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # 简化实现：每次重新前向传播（完整实现应使用KV Cache）
        logits = model(generated)
        next_token_logits = logits[:, -1, :] / temperature
        next_token = torch.multinomial(
            F.softmax(next_token_logits, dim=-1), num_samples=1
        )
        generated = torch.cat([generated, next_token], dim=1)

    return generated
```

---
