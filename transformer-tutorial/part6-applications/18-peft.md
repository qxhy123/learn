# 第18章：参数高效微调（PEFT）

## 学习目标

完成本章学习后，你将能够：

1. **理解PEFT的动机**：掌握为什么全参数微调在大模型时代面临挑战，以及PEFT如何解决这些问题
2. **掌握LoRA的原理和实现**：理解低秩分解的数学原理，能够从零实现LoRA并在实际任务中应用
3. **理解Adapter的设计**：熟悉Adapter的结构、插入位置及参数量，能与LoRA进行合理比较
4. **掌握Prefix-Tuning和P-Tuning**：理解可学习前缀向量的作用机制，区分各种Prompt类微调方法
5. **选择合适的PEFT方法**：根据任务特性、资源约束和性能需求，合理选择和组合PEFT策略

---

## 18.1 PEFT概述

### 18.1.1 为什么需要PEFT

自2017年Transformer架构提出以来，预训练语言模型的规模呈指数级增长。从BERT的1.1亿参数，到GPT-3的1750亿参数，再到如今动辄千亿参数的大语言模型，模型规模的扩大带来了显著的性能提升，但也引入了严重的实用化挑战。

**传统全参数微调的困境：**

```
模型规模增长趋势：
BERT-base    ≈  110M 参数    （2018）
GPT-2        ≈  1.5B 参数    （2019）
T5-11B       ≈   11B 参数    （2020）
GPT-3        ≈  175B 参数    （2020）
PaLM         ≈  540B 参数    （2022）
LLaMA-65B    ≈   65B 参数    （2023）
```

对于一个175B参数的模型，全参数微调意味着：

- **存储开销**：以float32精度存储，175B参数需要约700GB显存；即使是float16，也需要350GB
- **每个下游任务需要独立存储一个完整副本**：若有100个任务，就需要100×700GB = 70TB存储
- **训练成本极高**：梯度计算和优化器状态（Adam需要2倍参数量）使内存需求翻倍
- **分发困难**：如此大的模型难以快速部署和版本管理

### 18.1.2 全参数微调的核心问题

**问题1：存储爆炸**

设模型参数量为 $N$，下游任务数为 $T$，全参数微调的总存储需求为：

$$\text{Storage}_{\text{full}} = N \times T \times \text{sizeof}(\text{dtype})$$

而参数高效微调的存储需求为：

$$\text{Storage}_{\text{PEFT}} = N \times \text{sizeof}(\text{dtype}) + \Delta \theta \times T \times \text{sizeof}(\text{dtype})$$

其中 $|\Delta \theta| \ll N$，通常 $|\Delta \theta| / N < 1\%$。

**问题2：灾难性遗忘（Catastrophic Forgetting）**

全参数微调时，模型在新任务上的梯度更新会覆盖预训练阶段学到的通用知识。这种现象被称为灾难性遗忘。

直觉上理解：预训练阶段模型学习了大量通用语言知识（语法、语义、常识等），全参数微调在特定任务上的梯度会"侵蚀"这些知识，导致：
- 在原任务上性能下降
- 模型泛化能力降低
- 对分布外样本更敏感

**问题3：计算资源限制**

大多数研究机构和企业不具备训练千亿参数模型的硬件条件。全参数微调进一步提高了门槛，使得技术垄断加剧。

### 18.1.3 PEFT的核心思想

参数高效微调（Parameter-Efficient Fine-Tuning，PEFT）的核心思想是：**冻结预训练模型的大部分参数，只训练少量新增或选定的参数，使模型适应下游任务。**

这一思想背后有深刻的理论依据：

**内在维度假说（Intrinsic Dimensionality Hypothesis）**

Aghajanyan等人（2020）提出，预训练语言模型的微调过程实际上发生在一个低维子空间中。也就是说，虽然模型参数空间维度极高，但真正需要改变的方向只有很少几个。

形式化表达：设原始参数为 $\theta_0 \in \mathbb{R}^d$，存在一个低维子空间投影矩阵 $P \in \mathbb{R}^{d \times k}$（$k \ll d$），微调后的参数可以近似表示为：

$$\theta = \theta_0 + P \cdot \phi$$

其中 $\phi \in \mathbb{R}^k$ 是在低维空间中学习的参数，$k$ 比 $d$ 小几个数量级。

这一发现为参数高效微调提供了理论基础：**我们无需修改全部参数，只需在合适的低维子空间中学习增量更新。**

### 18.1.4 各种PEFT方法概览

当前主流的PEFT方法可以分为三大类：

```
PEFT方法分类
├── 增量参数法（Addition-based）
│   ├── Adapter（串行插入小型网络）
│   └── Prefix-Tuning / P-Tuning（添加可学习前缀）
├── 权重分解法（Decomposition-based）
│   ├── LoRA（低秩矩阵分解）
│   ├── AdaLoRA（自适应秩分配）
│   └── DoRA（权重分解）
└── 参数选择法（Selection-based）
    ├── BitFit（只微调偏置项）
    └── Sparse Fine-tuning（选择性更新稀疏参数）
```

各方法的参数效率对比：

| 方法 | 可训练参数比例 | 相对全参数微调 |
|------|--------------|--------------|
| Full Fine-tuning | 100% | 基准 |
| Adapter | 0.5%~5% | 1/20~1/200 |
| LoRA | 0.1%~1% | 1/100~1/1000 |
| Prefix-Tuning | 0.1%~1% | 1/100~1/1000 |
| BitFit | ~0.1% | 1/1000 |
| P-Tuning v2 | ~0.1% | 1/1000 |

---

## 18.2 LoRA（Low-Rank Adaptation）

### 18.2.1 低秩分解的核心思想

LoRA（Low-Rank Adaptation，低秩适配）由微软研究院的Hu等人在2021年提出，是目前应用最广泛的PEFT方法之一。

**核心洞察**：预训练模型的权重矩阵本身是满秩的，但在特定任务的微调过程中，权重的变化量 $\Delta W$ 具有**低秩（low-rank）特性**。

直觉理解：假设你已经掌握了一门语言（预训练），现在只需要学习特定领域的专业词汇和表达（微调）。这种"增量学习"所需要的知识维度远小于整门语言的复杂度。

**数学基础**：

对于一个预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，全参数微调会学习一个更新量 $\Delta W \in \mathbb{R}^{d \times k}$：

$$W' = W_0 + \Delta W$$

LoRA假设 $\Delta W$ 具有低秩结构，将其分解为两个低秩矩阵的乘积：

$$\Delta W = B \cdot A$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

因此，前向传播变为：

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

### 18.2.2 参数量分析

原始矩阵参数量：$d \times k$

LoRA参数量：$d \times r + r \times k = r(d + k)$

当 $r \ll \min(d, k)$ 时，参数量大幅减少：

$$\text{压缩比} = \frac{r(d+k)}{dk} = r\left(\frac{1}{k} + \frac{1}{d}\right)$$

以GPT-3的注意力层为例（$d = k = 12288$，$r = 4$）：

$$\text{压缩比} = 4 \times \frac{2}{12288} \approx 0.065\%$$

### 18.2.3 初始化策略与缩放

LoRA的初始化非常关键：

- **矩阵 $A$**：使用高斯随机初始化（$\mathcal{N}(0, \sigma^2)$）
- **矩阵 $B$**：初始化为全零矩阵

这样在训练开始时，$\Delta W = BA = 0$，保证模型以预训练权重为起点，不引入随机扰动。

此外，LoRA引入了一个缩放因子 $\alpha$，最终的更新量为：

$$\Delta W = \frac{\alpha}{r} \cdot BA$$

缩放因子 $\alpha/r$ 的作用：
- 当改变 $r$ 时，可以通过调整 $\alpha$ 保持更新量的尺度稳定
- 通常设置 $\alpha = r$ 或 $\alpha = 2r$，即缩放因子为1或2

完整的前向传播公式：

$$h = W_0 x + \frac{\alpha}{r} \cdot B A x$$

### 18.2.4 应用于哪些层

LoRA最初设计用于Transformer的注意力机制中的权重矩阵。原论文中，作者主要将LoRA应用于自注意力的四个投影矩阵：

$$W_q, W_k, W_v, W_o \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

后续研究发现，将LoRA也应用于前馈网络（FFN）的权重矩阵可以进一步提升性能：

$$W_{\text{up}}, W_{\text{down}}, W_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$$

**实践建议**：

- 仅对 $W_q$, $W_v$ 使用LoRA：参数效率最高，适合资源极度受限场景
- 对所有注意力矩阵使用LoRA：平衡效率与性能
- 对所有线性层使用LoRA：性能最优，但参数量增加

### 18.2.5 秩 r 的选择

秩 $r$ 是LoRA最重要的超参数，直接决定了模型的表达能力和参数效率的权衡：

| 秩 r | 特点 | 适用场景 |
|------|------|----------|
| 1~2 | 极度参数高效，表达能力弱 | 简单分类任务 |
| 4~8 | 常用默认值，良好平衡 | 大多数NLP任务 |
| 16~32 | 更强表达能力 | 复杂生成任务 |
| 64+ | 接近全参数微调 | 极复杂任务 |

经验规则：从 $r=8$ 开始，根据验证集性能调整。

### 18.2.6 推理时权重合并

LoRA的一个重要优势是推理时无额外开销。训练结束后，可以将LoRA权重合并到原始权重中：

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} \cdot BA$$

合并后模型与原始架构完全相同，推理速度不受影响。这与Adapter等方法形成鲜明对比（Adapter在推理时有额外的前向传播开销）。

---

## 18.3 Adapter

### 18.3.1 Adapter的结构设计

Adapter（适配器）由Houlsby等人在2019年提出，是最早的PEFT方法之一。其核心思想是在Transformer的每一层中插入小型神经网络模块（适配器），训练时只更新这些模块的参数。

**标准Adapter结构**（瓶颈结构，Bottleneck Architecture）：

```
输入 x (维度 d)
    ↓
下投影层：Linear(d → m)    ← 参数量: d×m + m
    ↓
非线性激活：GeLU / ReLU
    ↓
上投影层：Linear(m → d)    ← 参数量: m×d + d
    ↓
残差连接：+ x
    ↓
输出 (维度 d)
```

数学表达：

$$\text{Adapter}(x) = x + f\left(W_{\text{up}} \cdot \text{GeLU}(W_{\text{down}} \cdot x + b_{\text{down}}) + b_{\text{up}}\right)$$

其中：
- $W_{\text{down}} \in \mathbb{R}^{m \times d}$：下投影矩阵（降维）
- $W_{\text{up}} \in \mathbb{R}^{d \times m}$：上投影矩阵（升维）
- $m$：瓶颈维度（bottleneck dimension），通常 $m \ll d$
- 残差连接确保初始状态接近恒等变换

**参数量分析**：

单个Adapter的参数量：$2md + m + d \approx 2md$（当 $m \ll d$ 时）

压缩比（相对于单个权重矩阵 $W \in \mathbb{R}^{d \times d}$）：

$$\frac{2md}{d^2} = \frac{2m}{d}$$

以BERT-base（$d=768$，$m=64$）为例：$2 \times 64 / 768 \approx 16.7\%$

### 18.3.2 Adapter的插入位置

原始Houlsby Adapter在每个Transformer层中插入两个Adapter模块：

```
Transformer Layer（Houlsby版本）：

输入
 │
 ▼
LayerNorm
 │
 ▼
Multi-Head Attention
 │
 ▼
Adapter ← 第一个Adapter（注意力后）
 │
 ▼
残差连接
 │
 ▼
LayerNorm
 │
 ▼
Feed-Forward Network
 │
 ▼
Adapter ← 第二个Adapter（FFN后）
 │
 ▼
残差连接
 │
 ▼
输出
```

后续工作（Pfeiffer Adapter）发现只在FFN后插入一个Adapter即可达到类似效果，同时减少了一半的Adapter参数：

```
Transformer Layer（Pfeiffer版本）：

输入
 │
 ▼
LayerNorm + Multi-Head Attention + 残差连接
 │
 ▼
LayerNorm + Feed-Forward Network
 │
 ▼
Adapter ← 只有一个Adapter
 │
 ▼
残差连接 → 输出
```

### 18.3.3 Adapter的变体

**AdapterFusion**（2020）：将多个任务的Adapter融合，通过注意力机制动态选择各任务知识：

$$\text{AdapterFusion}(h, \{A_i\}) = \text{Softmax}(h W_q \cdot [A_1(h), ..., A_n(h)]^T) \cdot [A_1(h), ..., A_n(h)]$$

**MAM Adapter**（Mix-and-Match）（2022）：将Adapter与Prefix-Tuning结合，在不同位置使用不同策略：
- 注意力层：Prefix-Tuning
- FFN层：Parallel Adapter（并行而非串行）

**Parallel Adapter**：将Adapter与主网络并行而非串行：

$$h = W x + \text{Adapter}(x)$$

这种方式减少了推理时的串行计算延迟。

### 18.3.4 Adapter vs LoRA 详细对比

| 维度 | Adapter | LoRA |
|------|---------|------|
| 结构 | 串行插入瓶颈网络 | 并行低秩矩阵分解 |
| 非线性 | 有（GeLU/ReLU） | 无 |
| 推理开销 | 有（额外前向传播） | 无（可合并权重） |
| 参数量 | 通常更多 | 通常更少 |
| 表达能力 | 较强（非线性） | 适中（线性） |
| 实现复杂度 | 需修改模型架构 | 相对简单 |
| 多任务切换 | 替换Adapter模块 | 替换LoRA权重 |
| 内存效率 | 较低 | 较高 |
| 主要应用 | NLU任务 | 生成模型、指令微调 |

---

## 18.4 Prefix-Tuning

### 18.4.1 可学习的前缀向量

Prefix-Tuning由Li和Liang在2021年提出，其核心思想与Adapter和LoRA完全不同：**不修改模型权重，而是在输入序列前添加一系列可学习的连续向量（前缀）。**

传统Prompt工程使用离散的文本提示：
```
离散Prompt：
"翻译以下英文为中文：[INPUT]"
```

Prefix-Tuning使用连续的可学习向量：
```
Prefix-Tuning：
[P1, P2, ..., Pk, INPUT_TOKEN_1, INPUT_TOKEN_2, ...]
  ↑可学习前缀          ↑原始输入
```

前缀向量 $P = [p_1, p_2, ..., p_k]$ 中的每个 $p_i \in \mathbb{R}^{d_{\text{model}}}$ 都是可训练参数，与词嵌入维度相同但不对应任何真实词汇。

### 18.4.2 在注意力机制中的作用

Prefix-Tuning的独特之处在于，前缀向量不仅在输入层添加，而是**在每一层的注意力计算中都添加**，直接操控Key和Value矩阵。

对于第 $l$ 层的多头注意力，假设原始序列的Key和Value为：

$$K_l = W_k^l H_l, \quad V_l = W_v^l H_l$$

添加前缀后变为：

$$K_l' = [P_k^l; K_l], \quad V_l' = [P_v^l; V_l]$$

其中 $P_k^l, P_v^l \in \mathbb{R}^{k \times d_{\text{head}}}$ 是第 $l$ 层的前缀键值对（每层独立学习）。

注意力计算变为：

$$\text{Attention}(Q_l, K_l', V_l') = \text{Softmax}\left(\frac{Q_l (K_l')^T}{\sqrt{d_k}}\right) V_l'$$

这意味着每个输出token都可以"关注"到前缀向量，前缀向量可以理解为注入到每层注意力的"软性指令"。

### 18.4.3 重参数化技巧

直接优化前缀向量可能导致训练不稳定。Prefix-Tuning原论文采用了重参数化策略：

训练时，通过一个小型MLP将低维参数映射到前缀空间：

$$P_l = \text{MLP}_\theta(P_{\text{init}})$$

推理时，删除MLP，直接使用学到的前缀 $P_l$。

这种方式的优点：
- 降低了优化难度（先在低维空间学习）
- 前缀向量之间的相关性得到建模
- 训练更稳定

### 18.4.4 Prompt Tuning

Prompt Tuning（Lester等，2021）是Prefix-Tuning的简化版本：**只在输入层添加可学习的soft token，不修改每层的注意力。**

```
Prefix-Tuning vs Prompt Tuning：

Prefix-Tuning：
Layer L: [P_L; K_L], [P_L; V_L]  ← 每层都有前缀
Layer 2: [P_2; K_2], [P_2; V_2]
Layer 1: [P_1; K_1], [P_1; V_1]
Input:   [PREFIX_TOKENS; INPUT_TOKENS]

Prompt Tuning：
Layer L: K_L, V_L                 ← 中间层无修改
Layer 2: K_2, V_2
Layer 1: K_1, V_1
Input:   [SOFT_TOKENS; INPUT_TOKENS]  ← 只在输入层添加
```

Prompt Tuning的参数量极少：仅需 $k \times d_{\text{model}}$ 个参数（$k$ 为前缀长度）。

研究发现，随着模型规模的增大，Prompt Tuning的性能逐渐逼近全参数微调，在超过10B参数的模型上表现尤为突出。

### 18.4.5 P-Tuning v1 和 v2

**P-Tuning v1**（Liu等，2021）针对自然语言理解任务（如分类），将可学习的soft token插入prompt模板中：

```
传统hard prompt模板：
"[INPUT] It was [MASK]."

P-Tuning v1：
"[e1][e2] [INPUT] [e3][e4] [MASK]."
  ↑软token          ↑软token
```

其中 `[e1]...[e4]` 是可学习的连续向量，但与Prefix-Tuning不同，它们可以出现在输入序列的任意位置（不限于前缀），通过LSTM编码器建模相互依赖关系。

**P-Tuning v2**（Liu等，2022）的主要改进：

1. **每层都添加可学习前缀**（类似Prefix-Tuning的思路）
2. **去掉重参数化MLP**，直接优化前缀向量
3. **应用于NLU任务**（Prefix-Tuning主要针对生成任务）
4. **使用分类头**而非生成式预测

P-Tuning v2的关键发现：对于参数量较小的模型（如BERT-base），在NLU任务上可以与全参数微调性能持平。

---

## 18.5 PEFT方法对比与选择

### 18.5.1 参数效率对比

以BERT-base（110M参数）为例，各方法的可训练参数量：

| 方法 | 可训练参数量 | 占总参数比例 | 每任务额外存储（MB） |
|------|------------|------------|-------------------|
| Full Fine-tuning | 110M | 100% | 440 MB |
| Houlsby Adapter | 3.6M | 3.3% | 14.4 MB |
| Pfeiffer Adapter | 1.8M | 1.6% | 7.2 MB |
| LoRA (r=8) | 0.9M | 0.8% | 3.6 MB |
| LoRA (r=4) | 0.45M | 0.4% | 1.8 MB |
| Prefix-Tuning | 0.1M | 0.1% | 0.4 MB |
| Prompt Tuning | 0.008M | 0.007% | 0.03 MB |
| BitFit | 0.1M | 0.1% | 0.4 MB |

### 18.5.2 性能对比

在SuperGLUE基准测试上（BERT-base/GPT-2规模）：

| 方法 | 性能（相对全参数微调） | 备注 |
|------|---------------------|------|
| Full Fine-tuning | 100%（基准） | - |
| Adapter | 98%~99% | 几乎无损 |
| LoRA (r=8) | 97%~99% | 接近全参数 |
| Prefix-Tuning | 95%~98% | 生成任务更强 |
| P-Tuning v2 | 97%~99% | NLU任务 |
| Prompt Tuning | 90%~95%（小模型）/ ~99%（大模型） | 规模依赖 |
| BitFit | 93%~96% | 简单任务更好 |

**关键结论**：
- 参数效率和性能之间存在权衡，但差距小于预期
- 对于超大模型（>10B），PEFT方法性能差距进一步缩小
- 生成任务上LoRA和Prefix-Tuning表现突出
- NLU任务上Adapter和P-Tuning v2表现更好

### 18.5.3 推理效率对比

| 方法 | 推理延迟 | 内存开销 | 可并行服务多任务 |
|------|---------|---------|----------------|
| Full Fine-tuning | 基准（1×） | 基准（1×） | 需要多份模型 |
| Adapter | 1.05×~1.2× | 微增 | 只需一份主模型 |
| LoRA（未合并） | 1.02×~1.05× | 微增 | 只需一份主模型 |
| LoRA（已合并） | 1×（无开销） | 1× | 需要多份合并权重 |
| Prefix-Tuning | 因序列变长略增 | KV缓存增大 | 只需一份主模型 |
| Prompt Tuning | 因序列变长略增 | 微增 | 只需一份主模型 |

**推理效率关键点**：
- LoRA合并后零开销，是生产环境的最优解
- Adapter的串行结构引入不可避免的延迟
- Prefix-Tuning扩展了KV缓存，长序列任务影响更大

### 18.5.4 实际应用选择建议

**场景一：资源极度受限（单张消费级GPU，24GB显存）**

推荐：**LoRA（r=4~8，仅Q/V矩阵）**
- 参数量最小
- 推理无额外开销
- 实现成熟（peft库支持完善）

**场景二：需要服务大量任务（100+任务）**

推荐：**LoRA（未合并）+ 动态加载**
- 只需一份主模型（节省存储）
- 按需加载不同任务的LoRA权重
- peft库提供完整支持

**场景三：生成任务（对话、翻译、摘要）**

推荐：**LoRA 或 Prefix-Tuning**
- LoRA：稳定、高效，r=8~16
- Prefix-Tuning：前缀长度100~200，对长文本生成效果好

**场景四：NLU任务（分类、NER、关系抽取）**

推荐：**P-Tuning v2 或 Adapter（Pfeiffer版）**
- P-Tuning v2在分类任务上有竞争力
- Pfeiffer Adapter实现简单、性能稳定

**场景五：超大模型（>30B参数）**

推荐：**QLoRA（量化LoRA）**
- 将基础模型量化为4-bit（NF4格式）
- 计算时反量化为bf16
- 显著降低显存需求（65B模型可在48GB显存上训练）

**选择决策树**：

```
需要PEFT?
├── 推理时不接受任何开销 → LoRA（合并权重）
├── 需要服务100+任务 → LoRA（动态加载）
├── 生成任务
│   ├── 超长输出 → Prefix-Tuning
│   └── 一般生成 → LoRA
├── NLU/分类任务
│   ├── 模型>10B → LoRA / Prompt Tuning
│   └── 模型<10B → P-Tuning v2 / Adapter
└── 资源极度受限 → LoRA (r=4) + 量化
```

---

## 本章小结

本章系统介绍了参数高效微调（PEFT）的主要方法。以下表格对各方法进行综合对比：

| 方法 | 核心思想 | 参数效率 | 性能 | 推理开销 | 实现难度 | 最佳场景 |
|------|---------|---------|------|---------|---------|---------|
| Full Fine-tuning | 更新全部参数 | 最低 | 最高 | 无 | 低 | 资源充足 |
| LoRA | 低秩矩阵分解 | 高 | 高 | 可消除 | 低 | 生成任务、通用 |
| AdaLoRA | 自适应秩分配 | 高 | 高+ | 可消除 | 中 | 预算敏感场景 |
| Houlsby Adapter | 串行瓶颈网络 | 中 | 高 | 低 | 中 | NLU任务 |
| Pfeiffer Adapter | 简化版Adapter | 中高 | 高 | 低 | 中 | 多任务学习 |
| Prefix-Tuning | 每层注入前缀KV | 高 | 中高 | 低 | 中 | 生成任务 |
| Prompt Tuning | 输入层软token | 极高 | 中（小模型）/高（大模型） | 极低 | 低 | 超大模型 |
| P-Tuning v2 | 每层前缀+分类头 | 高 | 高 | 低 | 中 | NLU任务 |
| BitFit | 只更新偏置 | 极高 | 中 | 无 | 极低 | 快速原型 |
| QLoRA | 量化+LoRA | 高 | 高 | 低 | 中 | 超大模型微调 |

**核心要点回顾**：

1. PEFT方法的理论基础是内在维度假说——微调发生在低维子空间中
2. LoRA以低秩矩阵 $\Delta W = BA$ 逼近权重更新，参数效率极高且推理可零开销
3. Adapter在模型架构中插入瓶颈网络，表达能力强但有推理延迟
4. Prefix-Tuning通过操控每层注意力的KV，实现"软性提示"注入
5. 方法选择需综合考虑任务类型、模型规模、资源约束和推理需求

---

## 代码实战

### 实战1：LoRA从零实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    LoRA 线性层实现

    将原始线性层 y = Wx + b 替换为 y = Wx + b + (alpha/r) * BAx
    其中 B ∈ R^{d_out × r}，A ∈ R^{r × d_in}
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r  # 缩放因子 α/r

        # 原始预训练权重（冻结）
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # LoRA 矩阵（可训练）
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Dropout（可选）
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # 初始化
        self.reset_parameters()

    def reset_parameters(self):
        # 原始权重使用 kaiming uniform 初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # LoRA A: 高斯初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # LoRA B: 零初始化（确保训练开始时 ΔW = 0）
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始线性变换
        result = F.linear(x, self.weight, self.bias)

        # LoRA 增量：(α/r) * B * A * x
        lora_update = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + lora_update * self.scaling

        return result

    def merge_weights(self) -> nn.Linear:
        """将 LoRA 权重合并到原始权重，返回标准 Linear 层"""
        merged_weight = self.weight + self.scaling * (self.lora_B @ self.lora_A)
        merged_layer = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        merged_layer.weight.data = merged_weight
        if self.bias is not None:
            merged_layer.bias.data = self.bias.clone()
        return merged_layer

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int = 8, lora_alpha: float = 16.0) -> 'LoRALinear':
        """从现有 Linear 层创建 LoRALinear，保留原始权重"""
        lora_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            bias=linear.bias is not None,
        )
        # 复制原始权重
        lora_layer.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            lora_layer.bias.data = linear.bias.data.clone()
        return lora_layer

    def trainable_parameters(self) -> int:
        """返回可训练参数数量"""
        return self.lora_A.numel() + self.lora_B.numel()

    def total_parameters(self) -> int:
        """返回总参数数量（包括冻结参数）"""
        total = self.weight.numel()
        if self.bias is not None:
            total += self.bias.numel()
        return total + self.trainable_parameters()


class LoRAConfig:
    """LoRA 配置类"""

    def __init__(
        self,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        target_modules: list = None,  # 例如 ['q_proj', 'v_proj']
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ['q_proj', 'v_proj']


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    将 LoRA 应用到模型的指定层

    遍历模型，将名称匹配 target_modules 的 Linear 层替换为 LoRALinear
    """
    for name, module in model.named_modules():
        # 检查是否是目标模块
        for target in config.target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # 找到父模块
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model

                # 替换为 LoRALinear
                lora_layer = LoRALinear.from_linear(
                    module, r=config.r, lora_alpha=config.lora_alpha
                )
                setattr(parent, child_name, lora_layer)
                print(f"Applied LoRA to: {name}")
                break

    # 冻结非 LoRA 参数
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    return model


def count_parameters(model: nn.Module) -> dict:
    """统计模型的可训练参数和总参数"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        'trainable': trainable,
        'total': total,
        'ratio': trainable / total * 100
    }


# ===== 使用示例 =====

class SimpleTransformer(nn.Module):
    """简化的 Transformer 用于演示"""

    def __init__(self, d_model=768, n_heads=12, n_layers=6, vocab_size=30522):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(d_model, d_model),
                'k_proj': nn.Linear(d_model, d_model),
                'v_proj': nn.Linear(d_model, d_model),
                'o_proj': nn.Linear(d_model, d_model),
                'ffn_up': nn.Linear(d_model, d_model * 4),
                'ffn_down': nn.Linear(d_model * 4, d_model),
            })
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        # 简化前向传播
        for layer in self.layers:
            x = layer['q_proj'](x) + x
        return self.classifier(x[:, 0, :])


if __name__ == "__main__":
    # 创建模型
    model = SimpleTransformer(d_model=768, n_layers=6)

    print("=== 全参数模型 ===")
    params = count_parameters(model)
    print(f"总参数: {params['total']:,}")
    print(f"可训练参数: {params['trainable']:,}")

    # 应用 LoRA
    config = LoRAConfig(r=8, lora_alpha=16.0, target_modules=['q_proj', 'v_proj'])
    model = apply_lora(model, config)

    print("\n=== 应用 LoRA 后 ===")
    params = count_parameters(model)
    print(f"总参数: {params['total']:,}")
    print(f"可训练参数: {params['trainable']:,}")
    print(f"可训练比例: {params['ratio']:.2f}%")

    # 前向传播测试
    dummy_input = torch.randint(0, 30522, (2, 128))
    output = model(dummy_input)
    print(f"\n输出形状: {output.shape}")
```

### 实战2：使用 peft 库的 LoRA

```python
"""
使用 Hugging Face peft 库对语言模型进行 LoRA 微调
需要安装: pip install peft transformers datasets
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from datasets import load_dataset


def load_model_and_tokenizer(model_name: str):
    """加载预训练模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer


def create_lora_model(model, r=8, lora_alpha=32, target_modules=None):
    """
    创建 LoRA 模型

    Args:
        model: 基础模型
        r: LoRA 秩
        lora_alpha: LoRA 缩放参数
        target_modules: 应用 LoRA 的模块名称列表
    """
    if target_modules is None:
        # 对于 LLaMA/GPT 类模型的常用设置
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 因果语言模型
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",         # 不对偏置使用 LoRA
        inference_mode=False,
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model


def tokenize_dataset(dataset, tokenizer, max_length=512):
    """数据集分词"""
    def tokenize_fn(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    return tokenized


def train_lora(
    model_name: str = "gpt2",
    dataset_name: str = "wikitext",
    output_dir: str = "./lora_output",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    r: int = 8,
    lora_alpha: int = 32,
):
    """完整的 LoRA 训练流程"""

    print(f"加载模型: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)

    print("创建 LoRA 模型...")
    model = create_lora_model(model, r=r, lora_alpha=lora_alpha)

    print(f"加载数据集: {dataset_name}")
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split="train[:1000]")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        learning_rate=learning_rate,
        fp16=True,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    # 数据整理器（语言模型任务）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言模型，不使用 MLM
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("开始训练...")
    trainer.train()

    # 保存 LoRA 权重（只保存增量，不保存基础模型）
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA 权重已保存到: {output_dir}")

    return model, tokenizer


def load_and_merge_lora(base_model_name: str, lora_path: str):
    """
    加载基础模型和 LoRA 权重，并合并
    合并后的模型推理速度与原始模型相同
    """
    print("加载基础模型...")
    base_model, tokenizer = load_model_and_tokenizer(base_model_name)

    print("加载 LoRA 权重...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("合并 LoRA 权重...")
    model = model.merge_and_unload()

    print("合并完成！模型推理无额外开销")
    return model, tokenizer


def inference_example(model, tokenizer, prompt: str, max_new_tokens: int = 100):
    """生成示例"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


# 主程序
if __name__ == "__main__":
    # 方式1：训练 LoRA
    # model, tokenizer = train_lora(model_name="gpt2", r=8, lora_alpha=32)

    # 方式2：直接演示配置
    print("=== peft 库 LoRA 配置示例 ===")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    print(f"LoRA 配置:\n{lora_config}")
    print("\n使用方法:")
    print("  model = get_peft_model(base_model, lora_config)")
    print("  model.print_trainable_parameters()")
    print("  # 训练后保存")
    print("  model.save_pretrained('./lora_weights')")
    print("  # 推理时合并（可选）")
    print("  merged_model = model.merge_and_unload()")
```

### 实战3：Adapter 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdapterLayer(nn.Module):
    """
    标准 Pfeiffer Adapter 实现（瓶颈结构）

    结构：输入 → 下投影 → 激活 → 上投影 → 残差连接 → 输出
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int,
        activation: str = 'gelu',
        init_scale: float = 1e-3,
    ):
        super().__init__()

        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        # 下投影（降维）
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        # 上投影（升维）
        self.up_proj = nn.Linear(bottleneck_dim, d_model)

        # 激活函数
        self.activation = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
        }[activation]

        # 初始化（近似恒等变换）
        self._init_weights(init_scale)

    def _init_weights(self, init_scale: float):
        """初始化使得 Adapter 近似为恒等变换"""
        # 小值初始化，减少对预训练模型的影响
        nn.init.normal_(self.down_proj.weight, std=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=init_scale)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 (batch, seq_len, d_model)

        Returns:
            输出张量，形状 (batch, seq_len, d_model)
        """
        # 瓶颈变换
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        # 残差连接
        return x + residual

    def parameter_count(self) -> dict:
        """统计参数量"""
        down_params = self.d_model * self.bottleneck_dim + self.bottleneck_dim
        up_params = self.bottleneck_dim * self.d_model + self.d_model
        total = down_params + up_params
        return {
            'down_proj': down_params,
            'up_proj': up_params,
            'total': total,
            'compression_ratio': total / (self.d_model ** 2),
        }


class TransformerLayerWithAdapter(nn.Module):
    """
    带 Adapter 的 Transformer 层（Pfeiffer 版本）

    原始层结构保持不变，Adapter 插入在 FFN 之后
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        bottleneck_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 原始 Transformer 组件（预训练后冻结）
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Adapter（只有这部分是可训练的）
        self.adapter = AdapterLayer(d_model, bottleneck_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # 自注意力（预训练权重，冻结）
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm,
                                      attn_mask=attn_mask,
                                      key_padding_mask=key_padding_mask)
        x = residual + self.dropout(attn_out)

        # FFN（预训练权重，冻结）
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.dropout(ffn_out)

        # Adapter（可训练，插入在 FFN 之后）
        x = self.adapter(x)

        return x


def freeze_except_adapter(model: nn.Module) -> nn.Module:
    """冻结所有非 Adapter 参数"""
    for name, param in model.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # 统计
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Adapter 微调: {trainable:,} / {total:,} 参数可训练 ({trainable/total*100:.2f}%)")

    return model


# 演示
if __name__ == "__main__":
    # 配置
    d_model = 768
    n_heads = 12
    d_ff = 3072
    bottleneck_dim = 64   # 瓶颈维度
    batch_size = 4
    seq_len = 128

    # 创建带 Adapter 的 Transformer 层
    layer = TransformerLayerWithAdapter(d_model, n_heads, d_ff, bottleneck_dim)

    # 冻结非 Adapter 参数
    layer = freeze_except_adapter(layer)

    # 查看 Adapter 参数量
    adapter_info = layer.adapter.parameter_count()
    print(f"\nAdapter 参数量统计:")
    print(f"  下投影: {adapter_info['down_proj']:,}")
    print(f"  上投影: {adapter_info['up_proj']:,}")
    print(f"  总计:   {adapter_info['total']:,}")
    print(f"  压缩比: {adapter_info['compression_ratio']:.4f}")

    # 前向传播测试
    x = torch.randn(batch_size, seq_len, d_model)
    output = layer(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"形状保持一致: {x.shape == output.shape}")
```

### 实战4：Prefix-Tuning 示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PrefixTuning(nn.Module):
    """
    Prefix-Tuning 实现

    为每层 Transformer 的注意力机制添加可学习的前缀键值对
    使用重参数化技巧（MLP）提升训练稳定性
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        prefix_length: int = 10,
        use_reparameterization: bool = True,
        reparam_hidden_dim: int = 512,
        prefix_dropout: float = 0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.prefix_length = prefix_length
        self.use_reparameterization = use_reparameterization

        if use_reparameterization:
            # 重参数化：低维嵌入 → MLP → 前缀
            self.prefix_embedding = nn.Embedding(prefix_length, d_model)
            self.prefix_mlp = nn.Sequential(
                nn.Linear(d_model, reparam_hidden_dim),
                nn.Tanh(),
                nn.Linear(reparam_hidden_dim, num_layers * 2 * d_model),
            )
        else:
            # 直接参数化：为每层直接学习 Key 和 Value 前缀
            # 形状: (num_layers, 2, num_heads, prefix_length, d_head)
            # 其中 2 代表 Key 和 Value
            self.prefix_params = nn.Parameter(
                torch.randn(num_layers, 2, num_heads, prefix_length, self.d_head) * 0.01
            )

        self.prefix_dropout = nn.Dropout(prefix_dropout)

        self._init_weights()

    def _init_weights(self):
        if self.use_reparameterization:
            nn.init.normal_(self.prefix_embedding.weight, std=0.02)

    def get_prefix(self, batch_size: int) -> torch.Tensor:
        """
        获取当前批次的前缀参数

        Returns:
            prefix: 形状 (num_layers, 2, batch_size, num_heads, prefix_length, d_head)
        """
        if self.use_reparameterization:
            # 通过 MLP 生成前缀
            idx = torch.arange(self.prefix_length, device=self.prefix_embedding.weight.device)
            prefix_emb = self.prefix_embedding(idx)  # (prefix_len, d_model)

            # MLP 变换
            prefix_flat = self.prefix_mlp(prefix_emb)  # (prefix_len, num_layers * 2 * d_model)
            prefix_flat = prefix_flat.view(
                self.prefix_length, self.num_layers, 2, self.d_model
            )

            # 重组形状: (num_layers, 2, prefix_len, d_model)
            prefix = prefix_flat.permute(1, 2, 0, 3)

            # 分割为多头: (num_layers, 2, prefix_len, num_heads, d_head)
            prefix = prefix.view(self.num_layers, 2, self.prefix_length, self.num_heads, self.d_head)

            # 转置: (num_layers, 2, num_heads, prefix_len, d_head)
            prefix = prefix.permute(0, 1, 3, 2, 4)

        else:
            prefix = self.prefix_params  # (num_layers, 2, num_heads, prefix_length, d_head)

        # 扩展到批次维度: (num_layers, 2, batch_size, num_heads, prefix_len, d_head)
        prefix = prefix.unsqueeze(2).expand(-1, -1, batch_size, -1, -1, -1)

        return self.prefix_dropout(prefix)

    def forward_layer(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        prefix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在指定层的注意力中注入前缀

        Args:
            layer_idx: 层索引
            query: (batch, num_heads, seq_len, d_head)
            key: (batch, num_heads, seq_len, d_head)
            value: (batch, num_heads, seq_len, d_head)
            prefix: 所有层的前缀 (num_layers, 2, batch, num_heads, prefix_len, d_head)

        Returns:
            prefix_key: 扩展后的 Key
            prefix_value: 扩展后的 Value
        """
        # 获取当前层的前缀键值对
        prefix_key = prefix[layer_idx, 0]    # (batch, num_heads, prefix_len, d_head)
        prefix_value = prefix[layer_idx, 1]  # (batch, num_heads, prefix_len, d_head)

        # 拼接到原始键值对前面
        extended_key = torch.cat([prefix_key, key], dim=2)    # (batch, num_heads, prefix_len+seq_len, d_head)
        extended_value = torch.cat([prefix_value, value], dim=2)

        return extended_key, extended_value


class AttentionWithPrefix(nn.Module):
    """带 Prefix-Tuning 的多头自注意力"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # 原始注意力投影（冻结）
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        prefix_key: Optional[torch.Tensor] = None,
        prefix_value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # 投影
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # 注入前缀（如果提供）
        if prefix_key is not None and prefix_value is not None:
            k = torch.cat([prefix_key, k], dim=2)
            v = torch.cat([prefix_value, v], dim=2)

        # 注意力计算
        scale = self.d_head ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # 重组并投影
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.o_proj(attn_out)


# 演示 Prefix-Tuning
if __name__ == "__main__":
    # 模型配置
    num_layers = 12
    num_heads = 12
    d_model = 768
    prefix_length = 20
    batch_size = 2
    seq_len = 64

    # 创建 Prefix-Tuning 模块
    prefix_tuning = PrefixTuning(
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        prefix_length=prefix_length,
        use_reparameterization=True,
        reparam_hidden_dim=512,
    )

    # 统计参数
    total_params = sum(p.numel() for p in prefix_tuning.parameters())
    trainable_params = sum(p.numel() for p in prefix_tuning.parameters() if p.requires_grad)
    print(f"Prefix-Tuning 可训练参数: {trainable_params:,}")

    # 生成前缀
    prefix = prefix_tuning.get_prefix(batch_size)
    print(f"\n前缀张量形状: {prefix.shape}")
    print(f"  [num_layers={num_layers}, 2(K/V), batch={batch_size}, "
          f"num_heads={num_heads}, prefix_len={prefix_length}, d_head={d_model//num_heads}]")

    # 创建带前缀的注意力层
    attn = AttentionWithPrefix(d_model, num_heads)

    # 获取第0层的前缀
    prefix_key = prefix[0, 0]    # (batch, num_heads, prefix_len, d_head)
    prefix_value = prefix[0, 1]

    # 前向传播
    x = torch.randn(batch_size, seq_len, d_model)
    output = attn(x, prefix_key=prefix_key, prefix_value=prefix_value)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 比较有无前缀的参数量
    base_params = d_model * d_model * 4 * num_layers  # 4个投影矩阵
    print(f"\n对比:")
    print(f"  基础注意力参数 (12层): {base_params:,}")
    print(f"  Prefix-Tuning 参数:    {trainable_params:,}")
    print(f"  参数比例: {trainable_params/base_params*100:.2f}%")
```

---

## 练习题

### 基础题

**练习1**（基础）LoRA参数量计算

对于一个BERT-large模型（$d_{\text{model}} = 1024$，24层，每层有 $W_q, W_k, W_v, W_o$ 四个投影矩阵），使用LoRA（$r=8$）对所有四个投影矩阵进行微调：

(a) 计算原始四个投影矩阵的总参数量

(b) 计算LoRA引入的可训练参数量

(c) 计算参数效率（LoRA参数量 / 原始参数量）

(d) 如果改为 $r=16$，参数效率如何变化？

---

**练习2**（基础）理解LoRA的初始化策略

在LoRA中，矩阵 $B$ 初始化为零，矩阵 $A$ 使用高斯随机初始化。

(a) 为什么要将 $B$ 初始化为零而不是也使用随机初始化？

(b) 如果将 $A$ 也初始化为零会有什么问题？（提示：考虑梯度流动）

(c) 给出一种替代的初始化策略，并分析其优缺点

---

### 中级题

**练习3**（中级）Adapter vs LoRA 对比分析

假设一个Transformer模型有12层，每层的FFN维度为 $d_{\text{model}} = 768$，$d_{\text{ff}} = 3072$。

(a) 使用Pfeiffer Adapter（瓶颈维度 $m=64$），计算总可训练参数量

(b) 使用LoRA（$r=8$，只对 $W_q, W_v$ 使用），计算总可训练参数量

(c) 为使两种方法的参数量相同，LoRA的秩 $r$ 应该设为多少？

(d) 分析两种方法在以下场景的优劣：生产环境部署、多任务服务、资源受限训练

---

**练习4**（中级）Prefix-Tuning的有效性分析

Prefix-Tuning在每层注意力中添加 $k$ 个前缀向量。

(a) 解释为什么在每层都添加前缀比只在输入层添加（Prompt Tuning）更有效

(b) 对于一个有12层、12头、$d_{\text{model}}=768$ 的模型，前缀长度为20，计算不使用重参数化时的可训练参数量

(c) 使用重参数化（隐藏维度512）后，训练参数量是多少？这种方式的训练和推理行为有何不同？

(d) 前缀长度对模型性能的影响是单调的吗？为什么？

---

### 提高题

**练习5**（提高）设计自适应PEFT策略

你正在为一家公司设计一个多任务微调系统，需要在同一个基础模型上支持50个不同任务（文本分类、摘要生成、问答、翻译等），同时满足以下约束：

- 基础模型：7B参数（如LLaMA-7B），以float16存储（约14GB）
- 总存储预算：15GB（只比基础模型多1GB）
- 推理时延迟增加不超过10%
- 各任务需要独立更新（不同任务不互相影响）

请：

(a) 分析各PEFT方法在此场景下的可行性，给出定量分析

(b) 提出一个满足所有约束的方案，详细说明：
   - 使用哪种PEFT方法
   - 关键超参数设置（秩/瓶颈维度/前缀长度等）
   - 存储布局设计
   - 推理时如何高效切换任务

(c) 分析你的方案在以下方面的权衡：参数效率、性能、工程复杂度

(d) 如果将总存储预算放宽到50GB，你会如何调整方案？

---

## 练习答案

### 练习1 答案

**(a) 原始四个投影矩阵的总参数量**

每个投影矩阵：$W_q, W_k, W_v, W_o \in \mathbb{R}^{1024 \times 1024}$

每个矩阵参数量：$1024 \times 1024 = 1,048,576$

24层，4个矩阵：$1,048,576 \times 4 \times 24 = 100,663,296 \approx 100.7M$

**(b) LoRA的可训练参数量**

单个矩阵的LoRA参数：$r \times d + d \times r = 2 \times r \times d = 2 \times 8 \times 1024 = 16,384$

24层，4个矩阵：$16,384 \times 4 \times 24 = 1,572,864 \approx 1.57M$

**(c) 参数效率**

$$\text{效率} = \frac{1,572,864}{100,663,296} \approx 1.56\%$$

即仅需原始参数量的1.56%。

**(d) r=16时的参数效率**

LoRA参数：$2 \times 16 \times 1024 \times 4 \times 24 = 3,145,728 \approx 3.15M$

效率：$3.15M / 100.7M \approx 3.13\%$

规律：参数效率与秩 $r$ 成线性关系。

---

### 练习2 答案

**(a) B初始化为零的原因**

训练开始时，$\Delta W = BA = B \cdot A$。若 $B = 0$，则 $\Delta W = 0$，即LoRA对模型的初始状态没有影响。这保证了：

1. 训练从预训练权重出发，而不是从随机扰动出发
2. 初始预测与原始预训练模型完全相同
3. 避免引入随机噪声，训练更稳定

**(b) A也初始化为零的问题**

若 $A = 0$，则 $\Delta W = BA = 0$（与 $B=0$ 效果相同）。但问题在于梯度：

$$\frac{\partial \mathcal{L}}{\partial A} = B^T \frac{\partial \mathcal{L}}{\partial (BA)} = 0 \cdot (\ldots) = 0$$

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial (BA)} A^T = (\ldots) \cdot 0 = 0$$

两个矩阵梯度均为零，导致训练完全停滞（"梯度死区"问题）。这就是为什么必须至少有一个矩阵是非零初始化的。

**(c) 替代初始化策略**

**策略：SVD初始化**

使用预训练权重矩阵的SVD分解来初始化：

$$W_0 = U \Sigma V^T \approx U_r \Sigma_r V_r^T$$

令 $B = U_r \Sigma_r^{1/2}$，$A = \Sigma_r^{1/2} V_r^T$

**优点**：
- $\Delta W$ 初始时近似为 $W_0$ 的最优低秩逼近
- 初始值有物理意义（保留主要信息方向）
- 可能加速收敛

**缺点**：
- 需要额外计算SVD（$O(d^3)$ 复杂度）
- 初始 $\Delta W \neq 0$，会改变初始预测
- 实际效果因任务而异，不一定优于标准初始化

---

### 练习3 答案

**(a) Pfeiffer Adapter总可训练参数量**

每层的Adapter参数：

$$2 \times m \times d_{\text{model}} + m + d_{\text{model}} = 2 \times 64 \times 768 + 64 + 768 = 99,328$$

12层总计：$99,328 \times 12 = 1,191,936 \approx 1.19M$

**(b) LoRA总可训练参数量**

每层对 $W_q, W_v$ 的LoRA参数：

$$2 \times 2rd = 2 \times 2 \times 8 \times 768 = 24,576$$

12层总计：$24,576 \times 12 = 294,912 \approx 0.29M$

**注**：Adapter参数量约是LoRA（r=8）的4倍。

**(c) 等参数量时的LoRA秩**

令LoRA参数量等于Adapter参数量：

$$2 \times 2r \times d \times 12 = 1,191,936$$

$$r = \frac{1,191,936}{4 \times 768 \times 12} = \frac{1,191,936}{36,864} \approx 32.3$$

所以 $r \approx 32$ 时两者参数量相当。

**(d) 场景优劣分析**

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 生产环境部署（低延迟） | LoRA（合并） | 推理零开销，Adapter有串行延迟 |
| 多任务服务（100+任务） | LoRA（不合并） | 动态加载效率高，存储更小 |
| 资源受限训练 | LoRA | 参数量更少，显存占用更低 |
| 需要强表达能力 | Adapter | 非线性激活函数增强表达能力 |

---

### 练习4 答案

**(a) 每层添加前缀比只在输入层添加更有效的原因**

Transformer的表示随层次加深而变化，高层特征与低层特征有本质差异：

- **输入层**的前缀通过多层处理后，影响逐渐被稀释（梯度消失类似效应）
- **每层直接注入**前缀KV对，可以在对应抽象层次直接影响注意力模式
- 不同层次捕捉不同粒度的语义，每层独立学习的前缀可以针对性地调整各层的注意力行为
- 实验结果也证实：每层注入的Prefix-Tuning在生成任务上显著优于只在输入层的Prompt Tuning（尤其对小模型）

**(b) 不使用重参数化的参数量**

前缀为每层的Key和Value分别提供 $k$ 个向量，每个向量维度为 $d_{\text{model}}$：

$$\text{参数量} = \text{num\_layers} \times 2 \times k \times d_{\text{model}}$$

$$= 12 \times 2 \times 20 \times 768 = 368,640 \approx 0.37M$$

**(c) 使用重参数化后的参数量**

重参数化MLP：

$$\text{Embedding参数} = k \times d_{\text{model}} = 20 \times 768 = 15,360$$

$$\text{MLP参数} = d_{\text{model}} \times 512 + 512 + 512 \times (\text{num\_layers} \times 2 \times d_{\text{model}}) + \text{num\_layers} \times 2 \times d_{\text{model}}$$

$$= 768 \times 512 + 512 + 512 \times 18432 + 18432$$

$$= 393,216 + 512 + 9,437,184 + 18,432 \approx 9.85M$$

**训练差异**：重参数化时训练MLP参数，推理时直接使用缓存的前缀矩阵（丢弃MLP），两者推理代价相同。重参数化使参数间有相关性约束，优化更稳定但训练参数更多。

**(d) 前缀长度与性能的关系**

前缀长度与性能**不是单调关系**：

- **过短前缀**（如1~5）：表达能力不足，无法充分捕捉任务特性
- **适中前缀**（如10~100）：通常性能最优
- **过长前缀**（如>200）：占用注意力的较大比例（前缀/总序列），可能导致：
  - 原始输入token的注意力被稀释
  - 前缀之间的交互变得复杂，优化困难
  - 推理时KV缓存开销增大

实验上，前缀长度在10~100范围内通常足够，超过100后边际收益递减甚至性能下降。

---

### 练习5 答案

**(a) 各PEFT方法的可行性分析**

**约束量化**：

- 1GB额外存储 = 1024MB = 约512M float16参数
- 50个任务均分：每任务 $\approx$ 10M参数
- LLaMA-7B单层注意力维度 $d = 4096$

| 方法 | 每任务参数量（估算） | 50任务总额外存储 | 是否可行 |
|------|-------------------|----------------|---------|
| Full Fine-tuning | 7B = 14GB | 700GB | 不可行 |
| LoRA (r=8, Q/V) | $2 \times 8 \times 4096 \times 2 \times 32 \approx 4.2M$ | ~400MB | 可行 |
| LoRA (r=16, 全层) | ~50M | ~4.8GB | 不可行 |
| Pfeiffer Adapter (m=64) | $2 \times 64 \times 4096 \times 32 \approx 16.8M$ | ~1.6GB | 不可行 |
| Pfeiffer Adapter (m=16) | $\approx 4.2M$ | ~400MB | 可行 |
| Prompt Tuning (k=50) | $50 \times 4096 \approx 0.8M$ | ~76MB | 可行但性能差 |
| LoRA (r=4, Q/V) | ~2.1M | ~200MB | 可行，余量大 |

**(b) 推荐方案：LoRA动态加载**

**方案设计**：

- **方法**：LoRA，仅对 $W_q, W_v$ 使用（覆盖所有32层）
- **超参数**：$r=4$（严格场景）或 $r=8$（宽松场景）
- **存储布局**：
  ```
  基础模型: 14GB（固定，共享）
  LoRA权重目录:
    task_001_lora.bin: ~2MB (r=4) / ~4MB (r=8)
    task_002_lora.bin: ~2MB
    ...
    task_050_lora.bin: ~2MB
  总LoRA存储: ~100MB (r=4) / ~200MB (r=8)
  总存储: ~14.1GB / ~14.2GB ← 满足15GB约束
  ```
- **推理切换**：
  1. 基础模型常驻显存（不合并）
  2. 当前活跃任务的LoRA权重加载到显存
  3. 切换任务时，卸载旧LoRA权重，加载新LoRA权重
  4. 切换延迟仅为LoRA权重的IO时间（毫秒级）

**推理延迟分析**：
- LoRA未合并时，每个线性层多一次矩阵乘法
- 对于 $d=4096, r=4$：$A \in \mathbb{R}^{4 \times 4096}$，$B \in \mathbb{R}^{4096 \times 4}$
- 额外计算量极小（$r \ll d$），实测延迟增加 $<5\%$，满足10%约束

**(c) 方案的权衡分析**

| 维度 | 分析 |
|------|------|
| 参数效率 | 优秀：每任务仅需原始0.06%的参数 |
| 性能 | 良好：r=4对简单任务足够，复杂生成任务可能需要r=8 |
| 工程复杂度 | 低：peft库提供完整支持，动态加载简单 |
| 推理延迟 | 优秀：<5%增加，满足约束 |
| 任务隔离 | 完全隔离：每任务独立LoRA权重 |
| 扩展性 | 优秀：添加新任务只需新增一个小权重文件 |

**(d) 预算放宽到50GB时的调整**

额外预算：$50 - 14 = 36GB$，每任务可用：$36GB / 50 = 720MB$

**调整方案**：

1. **增大LoRA秩**：$r$ 从4提升到64~128，覆盖所有线性层（$W_q, W_k, W_v, W_o, W_{\text{up}}, W_{\text{down}}, W_{\text{gate}}$）
2. **每任务参数量**：约 $128 \times 4096 \times 2 \times 7 \times 32 \approx 234M \approx 468MB$（满足720MB）
3. **性能提升**：大秩LoRA覆盖所有层，性能接近全参数微调
4. **推理选项**：可以预合并高频任务权重（零延迟），其余任务动态加载
5. **更好选择**：对重要任务使用Adapter（更强的非线性表达），辅以LoRA服务长尾任务
