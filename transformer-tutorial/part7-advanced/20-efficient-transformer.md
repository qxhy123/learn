# 第20章：高效Transformer

> 当序列长度从512增长到16384，标准注意力的内存需求增长了1024倍。高效Transformer的核心任务，就是在保持模型能力的同时，打破这一平方复杂度的枷锁。

---

## 学习目标

完成本章学习后，你将能够：

1. **理解标准注意力的计算瓶颈**：分析 $O(L^2)$ 复杂度的来源及其对长序列任务的实际影响
2. **掌握稀疏注意力的原理**：理解固定模式和学习模式如何将计算量降低到 $O(L \sqrt{L})$ 或 $O(L \log L)$
3. **理解线性注意力的设计**：通过核函数近似将注意力计算降低到 $O(L)$
4. **了解Longformer和BigBird的实现**：掌握滑动窗口、全局Token和随机注意力的组合策略
5. **能够根据场景选择合适的高效方案**：在序列长度、任务类型、精度要求之间做出合理权衡

---

## 20.1 注意力的效率问题

> 延伸阅读：本章会多次提到 FlashAttention、块稀疏注意力、cuBLAS 和自定义 CUDA kernel。若你希望从 GPU 执行与访存角度理解这些实现背景，可继续阅读 [CUDA 教程：共享内存与分块优化](../../cuda-tutorial/part3-memory-and-execution/08-shared-memory-and-tiling.md)、[访存合并与 Occupancy](../../cuda-tutorial/part5-performance-and-tooling/13-memory-coalescing-and-occupancy.md) 与 [cuBLAS、cuDNN 与 CUB](../../cuda-tutorial/part7-libraries-and-dl/19-cublas-cudnn-and-cub.md)。

### 20.1.1 $O(L^2)$ 的时间和空间复杂度

标准自注意力机制的核心计算是：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

对于长度为 $L$、维度为 $d$ 的序列，这一计算涉及：

- **注意力矩阵计算** $QK^T$：形状为 $L \times L$，需要 $O(L^2 d)$ 次乘法
- **Softmax操作**：对 $L \times L$ 矩阵逐行归一化，$O(L^2)$
- **加权求和** $AV$：形状 $L \times L$ 乘以 $L \times d$，$O(L^2 d)$
- **内存占用**：注意力矩阵本身需要存储 $L^2$ 个浮点数

以float32计算，不同序列长度下单头注意力矩阵的内存需求：

| 序列长度 $L$ | 注意力矩阵大小 | 内存（float32） |
|:-----------:|:-------------:|:--------------:|
| 512         | 262,144       | 1 MB           |
| 2,048       | 4,194,304     | 16 MB          |
| 4,096       | 16,777,216    | 64 MB          |
| 16,384      | 268,435,456   | 1 GB           |
| 32,768      | 1,073,741,824 | 4 GB           |

当考虑多头注意力（通常12到64个头）和批次维度时，内存消耗迅速超出单块GPU的容量。

### 20.1.2 长序列的计算瓶颈

长序列任务在NLP和其他领域大量存在：

- **文档级NLP**：法律合同（数万词）、学术论文（数千词）、书籍摘要
- **代码理解**：大型代码库中的跨文件依赖分析
- **生物信息学**：蛋白质序列（数千氨基酸）、基因组序列
- **时间序列**：长周期传感器数据、音频信号（采样率高时序列极长）

BERT的512 token限制并非偶然，而是标准注意力在当时硬件条件下的上限。GPT-2的1024 token同理。即使是现代大模型，其长上下文能力往往也依赖专门的高效注意力实现。

### 20.1.3 内存问题的深层分析

内存瓶颈有两个维度：

**峰值内存（Peak Memory）**：注意力矩阵 $A \in \mathbb{R}^{L \times L}$ 必须在内存中完整存在（至少在计算softmax时），这是硬性约束。

**反向传播内存**：训练时需要保存前向计算的中间结果以计算梯度，注意力矩阵通常需要保存两份（前向值 + 梯度）。

**FlashAttention**（Dao et al., 2022）通过IO感知的分块计算部分解决了峰值内存问题，但其时间复杂度仍是 $O(L^2)$，只是减少了HBM读写次数。要真正打破平方复杂度，需要从算法层面改变注意力的计算方式。

### 20.1.4 各种优化方向概览

高效Transformer的主要技术路线可以归纳为四类：

```
高效注意力优化方向
├── 稀疏化注意力
│   ├── 固定模式（Local Window、Strided）
│   ├── 学习的稀疏模式（Adaptively Sparse）
│   └── 组合模式（Longformer、BigBird）
├── 低秩近似
│   ├── 线性注意力（核函数方法）
│   ├── Linformer（低秩投影K、V）
│   └── Performer（FAVOR+）
├── 哈希/聚类方法
│   ├── Reformer（LSH注意力）
│   └── Routing Transformer（在线聚类）
└── 硬件感知优化
    ├── FlashAttention（IO感知分块）
    └── PagedAttention（KV Cache管理）
```

---

## 20.2 稀疏注意力

### 20.2.1 稀疏注意力的基本思想

标准注意力中，每个token都与序列中的所有token交互。稀疏注意力的核心思想是：**并非所有token对都需要直接交互**，通过精心设计的稀疏模式，可以用 $O(L \cdot k)$ 的计算量（$k \ll L$）近似 $O(L^2)$ 的完整注意力。

形式化地，设 $\mathcal{A}(i)$ 为位置 $i$ 关注的位置集合，稀疏注意力定义为：

$$\text{SparseAttention}(Q, K, V)_i = \text{softmax}\left(\frac{q_i k_j^T}{\sqrt{d_k}}, j \in \mathcal{A}(i)\right) \cdot V_{\mathcal{A}(i)}$$

关键在于 $|\mathcal{A}(i)| \ll L$。

### 20.2.2 固定模式稀疏：Local Window

最自然的稀疏模式是**局部窗口注意力（Local Window Attention）**：每个token只关注其附近 $w$ 个token（$w/2$ 个前缀位置和 $w/2$ 个后续位置）：

$$\mathcal{A}_{\text{local}}(i) = \left\{j : |i - j| \leq \frac{w}{2}\right\}$$

复杂度降为 $O(L \cdot w)$。对于 $w = 512, L = 16384$，节省约32倍计算量。

局部窗口注意力的局限是**感受野受限**：信息传播需要 $L/w$ 层才能从序列头传到尾。这在浅层模型中是显著问题。

### 20.2.3 固定模式稀疏：Strided（步幅）注意力

**步幅注意力（Strided Attention）**引入了长距离依赖的捷径：

$$\mathcal{A}_{\text{strided}}(i) = \{j : j \equiv i \pmod{s}\} \cup \mathcal{A}_{\text{local}}(i)$$

其中 $s$ 是步幅。每个token除了关注局部邻居，还关注每隔 $s$ 个位置的token。这类似于膨胀卷积（dilated convolution），能以 $O(L \cdot (w + L/s))$ 的复杂度覆盖全局信息。

Sparse Transformer（Child et al., 2019）将Local和Strided以多头方式组合：一半的头使用局部注意力，另一半使用步幅注意力，在图像生成任务上取得了与完整注意力可比的效果。

### 20.2.4 学习的稀疏模式

固定模式是启发式设计，不能适应输入。**学习的稀疏模式**让模型自己决定哪些位置对应该交互：

**Top-K稀疏注意力**：计算完整的注意力分数后，只保留每行最大的 $k$ 个值：
$$A_{ij} = \begin{cases} \frac{\exp(q_i k_j^T / \sqrt{d})}{\sum_{l \in \text{top-k}(i)} \exp(q_i k_l^T / \sqrt{d})} & j \in \text{top-k}(i) \\ 0 & \text{otherwise} \end{cases}$$

问题是Top-K操作本身需要 $O(L^2)$ 来计算所有分数再选择。

**Routing Transformer**（Roy et al., 2021）用在线k-means聚类将token分组，同组token相互注意，复杂度 $O(L^{1.5})$。

### 20.2.5 稀疏注意力的实现挑战

稀疏操作在GPU上的实现并不像理论分析那样简单：

- GPU的并行计算单元针对稠密矩阵乘法优化（cuBLAS等库）
- 稀疏矩阵格式（CSR、COO）在小规模稀疏时效率反而更差
- 需要专门的CUDA kernel实现块稀疏（Block Sparse）操作

OpenAI的Sparse Transformer提供了针对GPU优化的块稀疏注意力实现，以块为单位（如 $64 \times 64$）进行稀疏化，在保持GPU利用率的同时实现稀疏计算。

---

## 20.3 线性注意力

### 20.3.1 核函数近似的基本思想

softmax注意力的关键结构是：

$$\text{Attention}(Q, K, V)_i = \frac{\sum_j \exp(q_i^T k_j / \sqrt{d}) v_j}{\sum_j \exp(q_i^T k_j / \sqrt{d})}$$

瓶颈在于分子的 $\sum_j \exp(q_i^T k_j / \sqrt{d}) v_j$：必须先计算所有 $\exp(q_i^T k_j)$，才能加权求和，无法改变求和顺序。

**核函数近似**的思路：用某个特征映射 $\phi: \mathbb{R}^d \to \mathbb{R}^r$ 来近似指数核：

$$\exp(q^T k / \sqrt{d}) \approx \phi(q)^T \phi(k)$$

如果这个近似成立，则：

$$\text{Attention}_i \approx \frac{\sum_j \phi(q_i)^T \phi(k_j) v_j}{\sum_j \phi(q_i)^T \phi(k_j)} = \frac{\phi(q_i)^T \sum_j \phi(k_j) v_j^T}{\phi(q_i)^T \sum_j \phi(k_j)}$$

关键变换：**将 $\phi(q_i)^T$ 移出求和符号**，先计算 $S = \sum_j \phi(k_j) v_j^T \in \mathbb{R}^{r \times d}$ 和 $z = \sum_j \phi(k_j) \in \mathbb{R}^r$，然后：

$$\text{Attention}_i \approx \frac{\phi(q_i)^T S}{\phi(q_i)^T z}$$

$S$ 和 $z$ 只需计算一次（$O(Lrd)$），每个位置的输出计算为 $O(rd)$，总体复杂度 $O(Lrd)$——如果 $r$ 是常数，则为 $O(L)$！

### 20.3.2 公式推导详解

设 $\phi: \mathbb{R}^d \to \mathbb{R}^r$ 为特征映射，定义：

$$\Phi_Q = \phi(Q) \in \mathbb{R}^{L \times r}, \quad \Phi_K = \phi(K) \in \mathbb{R}^{L \times r}$$

线性注意力定义为：

$$\text{LinearAttn}(Q, K, V) = \frac{\Phi_Q (\Phi_K^T V)}{\Phi_Q \Phi_K^T \mathbf{1}_L}$$

其中分母是逐行的归一化因子。

计算顺序至关重要：

- **错误顺序**：先计算 $\Phi_Q \Phi_K^T \in \mathbb{R}^{L \times L}$，再乘以 $V$，复杂度 $O(L^2 r)$
- **正确顺序**：先计算 $\Phi_K^T V \in \mathbb{R}^{r \times d}$（复杂度 $O(Lrd)$），再计算 $\Phi_Q (\Phi_K^T V) \in \mathbb{R}^{L \times d}$（复杂度 $O(Lrd)$）

总复杂度 $O(Lrd)$，当 $r \ll L$ 时相比 $O(L^2 d)$ 有极大优势。

因果掩码（Causal Masking）下的线性注意力也有高效递推形式：

$$S_t = S_{t-1} + \phi(k_t) v_t^T, \quad z_t = z_{t-1} + \phi(k_t)$$
$$\text{output}_t = \frac{\phi(q_t)^T S_t}{\phi(q_t)^T z_t}$$

这是一个 $O(1)$ 递推，整体推理复杂度 $O(L)$，特别适合自回归生成。

### 20.3.3 特征映射的选择

好的 $\phi$ 需要满足：$\phi(q)^T \phi(k) \geq 0$（保证权重非负）且尽可能准确近似 $\exp(q^T k / \sqrt{d})$。

**简单选择**（Katharopoulos et al., 2020 "Linear Transformers"）：

$$\phi(x) = \text{elu}(x) + 1 = \begin{cases} x + 1 & x \geq 0 \\ e^x & x < 0 \end{cases}$$

优点是计算简单，但对指数核的近似质量有限。

**随机傅里叶特征**：利用Bochner定理，平移不变核 $k(q, k) = k(q-k)$ 可以用随机特征近似：

$$k(q, k) = \mathbb{E}_{\omega \sim p(\omega)}[\exp(i\omega^T q) \exp(-i\omega^T k)]$$

对于高斯核，$\omega \sim \mathcal{N}(0, I)$，使用随机特征 $\phi(x) = \frac{1}{\sqrt{r}}[\cos(\omega_1^T x), \sin(\omega_1^T x), \ldots, \cos(\omega_r^T x), \sin(\omega_r^T x)]$。

Performer使用FAVOR+（Fast Attention Via positive Orthogonal Random features）对此进行了改进，使用正交随机特征并保证非负性。

### 20.3.4 线性注意力的局限

线性注意力并非免费的午餐：

1. **近似质量**：对指数核的近似在某些情况下精度不足，在需要sharp注意力（即注意力集中在少数token上）的任务中表现下降明显
2. **训练稳定性**：归一化因子 $\phi(q_i)^T z$ 可能趋近于零，导致数值不稳定
3. **任务差异**：在需要精确匹配（如问答、实体识别）的任务上，线性注意力通常弱于softmax注意力；在需要聚合（如文档分类、摘要）的任务上差距较小

---

## 20.4 Longformer

### 20.4.1 Longformer的设计动机

Beltagy et al.（2020）提出的Longformer针对长文档NLP任务，目标是将BERT扩展到能处理数千token的文档，同时保持预训练模型的迁移学习能力。

Longformer的核心洞察是：**不同类型的token需要不同类型的注意力**。局部上下文token需要滑动窗口注意力，而特殊的全局token（如[CLS]）需要能关注整个序列。

### 20.4.2 滑动窗口注意力

Longformer的基础注意力模式是以窗口大小 $w$ 的**滑动窗口（Sliding Window）**：

$$\mathcal{A}_{\text{window}}(i) = \left\{j : \max(0, i-w/2) \leq j \leq \min(L-1, i+w/2)\right\}$$

每个token关注其左右各 $w/2$ 个token，计算复杂度 $O(L \cdot w)$。

不同层可以使用不同窗口大小：浅层使用较小窗口（捕捉局部语法），深层使用较大窗口（捕捉更广泛的语义）。类似于CNN的感受野随层数增长：$l$ 层网络的有效感受野为 $O(l \cdot w)$。

### 20.4.3 全局注意力Token

滑动窗口无法直接支持需要全局上下文的任务（如分类、问答）。Longformer引入**全局注意力（Global Attention）**：

选定一些特殊位置 $\mathcal{G}$（如[CLS] token，或问答中的问题token），这些位置：
- **作为Query**：关注序列中的所有token
- **作为Key/Value**：被序列中的所有token关注

$$\mathcal{A}(i) = \begin{cases} \{0, 1, \ldots, L-1\} & i \in \mathcal{G} \\ \mathcal{A}_{\text{window}}(i) \cup \mathcal{G} & i \notin \mathcal{G} \end{cases}$$

全局注意力token的数量通常很少（几个到几十个），其计算复杂度为 $O(|\mathcal{G}| \cdot L)$，远小于完整 $O(L^2)$。

### 20.4.4 膨胀滑动窗口

为了在不增加参数的情况下扩大感受野，Longformer可选地使用**膨胀滑动窗口（Dilated Sliding Window）**：

$$\mathcal{A}_{\text{dilated}}(i, d) = \{j : j \equiv i \pmod{d}, |i-j| \leq w \cdot d\}$$

膨胀因子 $d$ 使窗口跨越更大范围，但维持相同的token数。在高层使用更大的膨胀率，类似于WaveNet的膨胀因果卷积。

### 20.4.5 实现细节

Longformer的关键实现技巧是**分块矩阵操作（Chunked Matrix Operations）**：

将序列分成长度为 $w$ 的块，对每个块内部以及与相邻块的交叉部分进行注意力计算，利用GPU的并行性高效处理。

实际实现中，Longformer维护两组独立的投影矩阵：
- **局部投影** $W_Q^{\text{local}}, W_K^{\text{local}}, W_V^{\text{local}}$：用于滑动窗口注意力
- **全局投影** $W_Q^{\text{global}}, W_K^{\text{global}}, W_V^{\text{global}}$：用于全局注意力

这使得全局注意力有更大的表达能力。

### 20.4.6 Longformer的实验结果

在长文档任务上，Longformer在以下数据集上显著优于BERT等短窗口模型：

- **WikiHop**：需要跨多个段落推理
- **TriviaQA**：长文档问答
- **HotpotQA**：多跳推理
- **IMDB**：长文档情感分析

Longformer-base（12层，768维）在序列长度4096时，显存占用约为RoBERTa-base的1/8，同时在长文档任务上性能更好。

---

## 20.5 其他高效架构

### 20.5.1 BigBird：三种模式的组合

Zaheer et al.（2020）的BigBird从理论角度出发，证明了稀疏注意力可以近似任意完整注意力的表达能力，前提是满足以下条件的组合：

**三种注意力模式**：

1. **随机注意力（Random Attention）**：每个token随机选择 $r$ 个token进行注意力，实现 $O(Lr)$ 的复杂度，确保图的连通性
2. **局部窗口注意力（Window Attention）**：覆盖局部上下文，与Longformer相同
3. **全局Token注意力（Global Token Attention）**：少数全局token（如[CLS]）关注所有位置

$$\mathcal{A}_{\text{BigBird}}(i) = \mathcal{A}_{\text{random}}(i) \cup \mathcal{A}_{\text{window}}(i) \cup \mathcal{A}_{\text{global}}(i)$$

**理论保证**：BigBird证明这种稀疏注意力是图灵完备的，且是完整二部图（full bigraph）的通用近似器——意味着稀疏BigBird在足够深时可以计算任何完整注意力能计算的函数。

BigBird在基因组序列（长度达4096+）的分类任务和长文档QA上取得了SOTA效果。

### 20.5.2 Reformer：LSH注意力

Kitaev et al.（2020）的Reformer针对极长序列（$L \geq 64K$），引入两项关键创新：

**LSH注意力（Locality Sensitive Hashing Attention）**：

softmax注意力的大部分权重集中在少数高相似度的key上（$\text{softmax}$ 的锐化效应）。LSH用于高效找到这些高相似度的 key-query 对：

1. 对 Q 和 K 使用相同的投影矩阵（tied QK）：$Q = K = X W_{QK}$
2. 使用随机投影LSH将向量分配到哈希桶：$h(x) = \arg\max([xR; -xR])$，其中 $R \in \mathbb{R}^{d \times b/2}$ 是随机矩阵
3. 同一哈希桶中的向量更可能互相注意，只在桶内计算注意力

复杂度降为 $O(L \log L)$（LSH分桶的复杂度），实践中相当于 $O(L \cdot b_{\text{size}})$，其中 $b_{\text{size}}$ 是桶大小。

**可逆残差（Reversible Residual）**：受RevNet启发，Reformer使用可逆残差层：

$$Y_1 = X_1 + \text{Attention}(X_2)$$
$$Y_2 = X_2 + \text{FFN}(Y_1)$$

正向计算可以从输出反推输入，无需存储所有层的激活值，内存从 $O(n \cdot L)$（$n$ 为层数）降为 $O(L)$。

### 20.5.3 Linformer：低秩投影

Wang et al.（2020）观察到注意力矩阵在实践中通常是低秩的。Linformer在此基础上提出：不近似注意力计算，而是直接对 K 和 V 做低秩投影：

$$\bar{K} = E K \in \mathbb{R}^{k \times d}, \quad \bar{V} = F V \in \mathbb{R}^{k \times d}$$

其中 $E, F \in \mathbb{R}^{k \times L}$ 是可学习的投影矩阵，$k \ll L$。

注意力计算变为：

$$\text{LinformerAttn}(Q, K, V) = \text{softmax}\left(\frac{Q \bar{K}^T}{\sqrt{d}}\right) \bar{V} \in \mathbb{R}^{L \times d}$$

注意力矩阵形状变为 $L \times k$（而非 $L \times L$），复杂度 $O(Lkd)$。当 $k = O(\log L)$ 时，总复杂度为 $O(L \log L)$。

Linformer的限制：$E, F$ 依赖于固定的序列长度 $L$，无法直接推广到训练时未见过的序列长度（不支持length extrapolation）。

### 20.5.4 效率对比

| 方法 | 时间复杂度 | 空间复杂度 | 适用场景 | 主要局限 |
|:----:|:----------:|:----------:|:--------:|:--------:|
| 标准Attention | $O(L^2 d)$ | $O(L^2)$ | 短序列 | 长序列OOM |
| Sparse Transformer | $O(L\sqrt{L}d)$ | $O(L\sqrt{L})$ | 图像生成 | 固定模式 |
| Longformer | $O(Lwd)$ | $O(Lw)$ | 长文档NLP | 需预训练 |
| BigBird | $O(Lrd)$ | $O(Lr)$ | 长文档+基因组 | 随机性 |
| Reformer | $O(L\log L \cdot d)$ | $O(L\log L)$ | 极长序列 | Tied QK限制 |
| Linformer | $O(Lkd)$ | $O(Lk)$ | 固定长度 | 不支持外推 |
| Linear Attention | $O(Lrd)$ | $O(r)$ | 流式推理 | 近似误差 |
| Performer | $O(Lrd)$ | $O(r)$ | 通用 | 理论保证弱 |
| FlashAttention | $O(L^2 d)$ | $O(L)$ | 训练加速 | 不降算术复杂度 |

---

## 代码实战

### 完整实现：高效注意力模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple


# ============================================================
# 1. 滑动窗口注意力（Sliding Window Attention）
# ============================================================

class SlidingWindowAttention(nn.Module):
    """
    滑动窗口注意力：每个token只关注其周围window_size个token。
    复杂度：O(L * window_size * d)
    """

    def __init__(self, d_model: int, num_heads: int, window_size: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) 可选的padding mask
        Returns:
            output: (batch, seq_len, d_model)
        """
        B, L, D = x.shape
        w = self.window_size
        H = self.num_heads
        d = self.d_head

        Q = self.W_q(x).view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        K = self.W_k(x).view(B, L, H, d).transpose(1, 2)
        V = self.W_v(x).view(B, L, H, d).transpose(1, 2)

        output = torch.zeros(B, H, L, d, device=x.device, dtype=x.dtype)

        # 对每个位置i，收集窗口内的K和V
        for i in range(L):
            start = max(0, i - w // 2)
            end = min(L, i + w // 2 + 1)

            # q_i: (B, H, 1, d)
            q_i = Q[:, :, i:i+1, :]
            # k_window: (B, H, window, d)
            k_window = K[:, :, start:end, :]
            # v_window: (B, H, window, d)
            v_window = V[:, :, start:end, :]

            # 注意力分数: (B, H, 1, window)
            scores = torch.matmul(q_i, k_window.transpose(-2, -1)) / math.sqrt(d)
            attn = F.softmax(scores, dim=-1)

            # 加权求和: (B, H, 1, d)
            output[:, :, i:i+1, :] = torch.matmul(attn, v_window)

        # 拼接多头输出
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(output)


class SlidingWindowAttentionFast(nn.Module):
    """
    向量化实现的滑动窗口注意力（避免Python循环）。
    使用unfold操作将窗口展开为批次维度。
    """

    def __init__(self, d_model: int, num_heads: int, window_size: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        w = self.window_size
        H = self.num_heads
        d = self.d_head

        Q = self.W_q(x).view(B, L, H, d)  # (B, L, H, d)
        K = self.W_k(x).view(B, L, H, d)
        V = self.W_v(x).view(B, L, H, d)

        # Padding: 两侧各补 w//2 个零
        pad = w // 2
        K_padded = F.pad(K.permute(0, 2, 3, 1), (pad, pad))  # (B, H, d, L+2*pad)
        V_padded = F.pad(V.permute(0, 2, 3, 1), (pad, pad))  # (B, H, d, L+2*pad)

        # unfold展开窗口: (B, H, d, L, w)
        K_windows = K_padded.unfold(-1, w, 1)
        V_windows = V_padded.unfold(-1, w, 1)

        # 调整维度: (B, H, L, d, w) -> (B, H, L, w, d)
        K_windows = K_windows.permute(0, 1, 3, 4, 2)
        V_windows = V_windows.permute(0, 1, 3, 4, 2)

        # Q: (B, L, H, d) -> (B, H, L, 1, d)
        Q = Q.permute(0, 2, 1, 3).unsqueeze(-2)

        # 注意力分数: (B, H, L, 1, w)
        scores = torch.matmul(Q, K_windows.transpose(-2, -1)) / math.sqrt(d)
        attn = F.softmax(scores, dim=-1)

        # 加权求和: (B, H, L, 1, d) -> (B, H, L, d)
        out = torch.matmul(attn, V_windows).squeeze(-2)

        # 恢复形状
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.W_o(out)


# ============================================================
# 2. 线性注意力（Linear Attention）
# ============================================================

class LinearAttention(nn.Module):
    """
    线性注意力实现。
    使用 phi(x) = elu(x) + 1 作为特征映射，近似 exp(q^T k)。
    复杂度：O(L * r * d)，其中r为特征维度（等于d_head）。
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 用于数值稳定的epsilon
        self.eps = 1e-6

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """特征映射 phi(x) = elu(x) + 1，保证非负性"""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            causal: 是否使用因果掩码（自回归场景）
        """
        B, L, D = x.shape
        H = self.num_heads
        d = self.d_head

        Q = self.W_q(x).view(B, L, H, d)  # (B, L, H, d)
        K = self.W_k(x).view(B, L, H, d)
        V = self.W_v(x).view(B, L, H, d)

        # 应用特征映射
        Q = self.feature_map(Q)  # (B, L, H, d)
        K = self.feature_map(K)

        if not causal:
            # 非因果（双向）线性注意力
            # 先计算 K^T V: (B, H, d, d)
            KtV = torch.einsum('blhd,blhe->bhde', K, V)  # (B, H, d, d)
            # 计算 Q (K^T V): (B, L, H, d)
            QKtV = torch.einsum('blhd,bhde->blhe', Q, KtV)  # (B, L, H, d)

            # 归一化因子: K的列和 (B, H, d)
            Ksum = K.sum(dim=1)  # (B, H, d)
            # Q 点乘 K的和: (B, L, H)
            denom = torch.einsum('blhd,bhd->blh', Q, Ksum)  # (B, L, H)
            denom = denom.unsqueeze(-1).clamp(min=self.eps)  # (B, L, H, 1)

            out = QKtV / denom  # (B, L, H, d)
        else:
            # 因果（单向）线性注意力：递推计算
            out = torch.zeros(B, L, H, d, device=x.device, dtype=x.dtype)
            S = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)  # 累积 K^T V
            z = torch.zeros(B, H, d, device=x.device, dtype=x.dtype)    # 累积 K

            for t in range(L):
                k_t = K[:, t, :, :]  # (B, H, d)
                v_t = V[:, t, :, :]  # (B, H, d)
                q_t = Q[:, t, :, :]  # (B, H, d)

                # 更新累积状态
                S = S + torch.einsum('bhd,bhe->bhde', k_t, v_t)
                z = z + k_t

                # 计算当前位置输出
                num = torch.einsum('bhd,bhde->bhe', q_t, S)  # (B, H, d)
                den = torch.einsum('bhd,bhd->bh', q_t, z).unsqueeze(-1).clamp(min=self.eps)
                out[:, t, :, :] = num / den

        out = out.contiguous().view(B, L, D)
        return self.W_o(out)


# ============================================================
# 3. Longformer风格的注意力
# ============================================================

class LongformerAttention(nn.Module):
    """
    Longformer风格注意力：滑动窗口 + 全局Token注意力。

    全局token（如[CLS]）关注所有位置，同时被所有位置关注。
    其余token使用滑动窗口注意力。
    """

    def __init__(self, d_model: int, num_heads: int, window_size: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.d_head = d_model // num_heads

        # 局部注意力投影
        self.W_q_local = nn.Linear(d_model, d_model)
        self.W_k_local = nn.Linear(d_model, d_model)
        self.W_v_local = nn.Linear(d_model, d_model)

        # 全局注意力投影（独立权重，给全局token更多表达能力）
        self.W_q_global = nn.Linear(d_model, d_model)
        self.W_k_global = nn.Linear(d_model, d_model)
        self.W_v_global = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        global_token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            global_token_mask: (batch, seq_len) bool tensor，True表示全局token
        """
        B, L, D = x.shape
        H = self.num_heads
        d = self.d_head
        w = self.window_size

        if global_token_mask is None:
            # 默认第一个token为全局token（类似[CLS]）
            global_token_mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
            global_token_mask[:, 0] = True

        output = torch.zeros(B, L, D, device=x.device, dtype=x.dtype)

        # ---- Step 1: 计算所有token的局部注意力输出 ----
        Q_l = self.W_q_local(x).view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        K_l = self.W_k_local(x).view(B, L, H, d).transpose(1, 2)
        V_l = self.W_v_local(x).view(B, L, H, d).transpose(1, 2)

        local_output = torch.zeros(B, H, L, d, device=x.device, dtype=x.dtype)

        for i in range(L):
            start = max(0, i - w // 2)
            end = min(L, i + w // 2 + 1)

            q_i = Q_l[:, :, i:i+1, :]           # (B, H, 1, d)
            k_win = K_l[:, :, start:end, :]      # (B, H, win, d)
            v_win = V_l[:, :, start:end, :]

            scores = torch.matmul(q_i, k_win.transpose(-2, -1)) / math.sqrt(d)
            attn = F.softmax(scores, dim=-1)
            local_output[:, :, i:i+1, :] = torch.matmul(attn, v_win)

        local_output = local_output.transpose(1, 2).contiguous().view(B, L, D)
        output = local_output

        # ---- Step 2: 全局Token的全局注意力（覆盖其局部注意力结果） ----
        Q_g = self.W_q_global(x).view(B, L, H, d).transpose(1, 2)
        K_g = self.W_k_global(x).view(B, L, H, d).transpose(1, 2)
        V_g = self.W_v_global(x).view(B, L, H, d).transpose(1, 2)

        # 找到全局token的索引（取batch中第一个样本的mask，简化实现）
        global_indices = global_token_mask[0].nonzero(as_tuple=True)[0]

        if len(global_indices) > 0:
            # 全局token关注所有位置
            Q_global_tokens = Q_g[:, :, global_indices, :]  # (B, H, n_global, d)
            scores_global = torch.matmul(Q_global_tokens, K_g.transpose(-2, -1)) / math.sqrt(d)
            attn_global = F.softmax(scores_global, dim=-1)
            out_global = torch.matmul(attn_global, V_g)  # (B, H, n_global, d)

            # 将全局token的输出写回
            out_global = out_global.transpose(1, 2).contiguous().view(B, len(global_indices), D)
            output[:, global_indices, :] = out_global

            # 非全局token也关注全局token（在其局部窗口注意力中额外加入全局token的影响）
            # 简化实现：对非全局token，额外加一个对全局token的注意力项
            non_global_mask = ~global_token_mask[0]
            non_global_indices = non_global_mask.nonzero(as_tuple=True)[0]

            if len(non_global_indices) > 0:
                Q_non_global = Q_l[:, :, non_global_indices, :]  # (B, H, n_non_global, d)
                K_global_tokens = K_g[:, :, global_indices, :]   # (B, H, n_global, d)
                V_global_tokens = V_g[:, :, global_indices, :]

                scores_ng = torch.matmul(Q_non_global, K_global_tokens.transpose(-2, -1)) / math.sqrt(d)
                attn_ng = F.softmax(scores_ng, dim=-1)
                out_ng = torch.matmul(attn_ng, V_global_tokens)  # (B, H, n_non_global, d)
                out_ng = out_ng.transpose(1, 2).contiguous().view(B, len(non_global_indices), D)

                # 将全局注意力输出与局部注意力输出平均（简化融合策略）
                output[:, non_global_indices, :] = (
                    output[:, non_global_indices, :] + out_ng
                ) / 2

        return self.W_o(output)


# ============================================================
# 4. 效率基准测试
# ============================================================

def benchmark_attention(
    attention_module: nn.Module,
    seq_lengths: list,
    d_model: int = 256,
    batch_size: int = 2,
    num_runs: int = 5,
    device: str = 'cpu'
) -> dict:
    """
    对不同序列长度进行速度和内存基准测试。

    Args:
        attention_module: 要测试的注意力模块
        seq_lengths: 测试的序列长度列表
        d_model: 模型维度
        batch_size: 批次大小
        num_runs: 重复测试次数（取平均）
        device: 运行设备

    Returns:
        包含各序列长度下时间和内存统计的字典
    """
    attention_module = attention_module.to(device)
    results = {'seq_lengths': seq_lengths, 'times': [], 'memory_mb': []}

    for L in seq_lengths:
        x = torch.randn(batch_size, L, d_model, device=device)

        # Warmup
        with torch.no_grad():
            _ = attention_module(x)

        # 测试时间
        times = []
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = attention_module(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转为毫秒

        avg_time = sum(times) / len(times)
        results['times'].append(avg_time)

        # 估算内存（简化：使用输入输出的内存）
        memory_mb = x.element_size() * x.nelement() / (1024 ** 2)
        results['memory_mb'].append(memory_mb)

        print(f"  L={L:5d}: {avg_time:.2f} ms")

    return results


def run_comparison():
    """运行所有注意力实现的对比测试"""
    d_model = 256
    num_heads = 8
    window_size = 64
    batch_size = 2
    device = 'cpu'

    print("=" * 60)
    print("高效注意力机制效率对比")
    print("=" * 60)
    print(f"配置: d_model={d_model}, heads={num_heads}, window={window_size}")
    print()

    # 测试序列长度
    seq_lengths = [128, 256, 512, 1024]

    # 初始化各模块
    modules = {
        'SlidingWindow（慢速版）': SlidingWindowAttention(d_model, num_heads, window_size),
        'SlidingWindow（快速版）': SlidingWindowAttentionFast(d_model, num_heads, window_size),
        'LinearAttention': LinearAttention(d_model, num_heads),
        'LongformerStyle': LongformerAttention(d_model, num_heads, window_size),
    }

    all_results = {}
    for name, module in modules.items():
        print(f"\n{name}:")
        results = benchmark_attention(
            module, seq_lengths, d_model, batch_size, device=device
        )
        all_results[name] = results

    # 打印对比表
    print("\n" + "=" * 70)
    print(f"{'方法':<25}", end='')
    for L in seq_lengths:
        print(f"  L={L:<6}", end='')
    print()
    print("-" * 70)

    for name, results in all_results.items():
        print(f"{name:<25}", end='')
        for t in results['times']:
            print(f"  {t:7.1f}ms", end='')
        print()

    print("=" * 70)


# ============================================================
# 5. 标准注意力（用于对比）
# ============================================================

class StandardAttention(nn.Module):
    """标准多头自注意力，用于基准对比"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads
        d = self.d_head

        Q = self.W_q(x).view(B, L, H, d).transpose(1, 2)
        K = self.W_k(x).view(B, L, H, d).transpose(1, 2)
        V = self.W_v(x).view(B, L, H, d).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)


def complexity_comparison():
    """演示各方法的复杂度差异"""
    print("\n复杂度实验：随序列长度增长的运行时间")
    print("-" * 50)

    d_model = 128
    num_heads = 4
    window_size = 32
    batch_size = 1
    seq_lengths = [64, 128, 256, 512]

    std_attn = StandardAttention(d_model, num_heads)
    lin_attn = LinearAttention(d_model, num_heads)
    sw_attn = SlidingWindowAttentionFast(d_model, num_heads, window_size)

    print(f"\n{'L':>6} | {'标准O(L²)':>12} | {'线性O(L)':>12} | {'滑动窗口O(Lw)':>14}")
    print("-" * 50)

    for L in seq_lengths:
        x = torch.randn(batch_size, L, d_model)
        times = {}

        for name, module in [('std', std_attn), ('lin', lin_attn), ('sw', sw_attn)]:
            runs = []
            for _ in range(10):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = module(x)
                end = time.perf_counter()
                runs.append((end - start) * 1000)
            times[name] = sum(runs) / len(runs)

        print(f"{L:>6} | {times['std']:>10.2f}ms | {times['lin']:>10.2f}ms | {times['sw']:>12.2f}ms")


if __name__ == '__main__':
    # 功能验证
    print("功能验证")
    print("-" * 40)
    d_model, num_heads, window_size = 64, 4, 8
    B, L = 2, 32

    x = torch.randn(B, L, d_model)

    sw = SlidingWindowAttentionFast(d_model, num_heads, window_size)
    la = LinearAttention(d_model, num_heads)
    lf = LongformerAttention(d_model, num_heads, window_size)

    print(f"输入形状: {x.shape}")
    print(f"SlidingWindow输出: {sw(x).shape}")
    print(f"LinearAttention输出: {la(x).shape}")
    print(f"LongformerStyle输出: {lf(x).shape}")
    print()

    # 线性注意力因果模式
    la_causal_out = la(x, causal=True)
    print(f"LinearAttention（因果）输出: {la_causal_out.shape}")

    # 运行基准测试
    run_comparison()

    # 复杂度对比
    complexity_comparison()
```

---

## 本章小结

本章系统介绍了高效Transformer的主要技术路线。标准注意力的 $O(L^2)$ 复杂度是处理长序列的根本瓶颈，各种高效方法从不同角度突破这一限制：

| 技术路线 | 核心思想 | 代表方法 | 复杂度 | 精度损失 | 适用场景 |
|:--------:|:--------:|:--------:|:------:|:--------:|:--------:|
| 固定稀疏模式 | 只计算局部邻居 | Longformer | $O(Lw)$ | 低（有全局token） | 长文档NLP |
| 组合稀疏 | 局部+随机+全局 | BigBird | $O(Lr)$ | 低 | 长文档+基因组 |
| 哈希稀疏 | LSH找近邻 | Reformer | $O(L\log L)$ | 中（tied QK） | 极长序列 |
| 低秩投影 | K/V降维 | Linformer | $O(Lk)$ | 中 | 固定长度序列 |
| 核近似 | 特征映射分解 | Performer | $O(Lr)$ | 中-高 | 流式推理 |
| IO感知 | 减少内存读写 | FlashAttention | $O(L^2)$ | 无 | 训练加速 |

**选择指南**：
- 序列长度 $< 2K$：标准注意力 + FlashAttention即可
- 长文档分类/QA（$2K - 16K$）：Longformer或BigBird
- 极长序列（$> 16K$）且精度要求高：BigBird或窗口注意力
- 流式推理/在线学习：线性注意力（因果递推形式）
- 需要预训练权重迁移：Longformer（可从BERT初始化）

---

## 练习题

### 基础题

**练习20.1**（基础）

分析标准自注意力的内存复杂度。对于批大小 $B=4$，头数 $H=8$，序列长度 $L=4096$，计算：
- (a) 注意力矩阵（所有头）占用多少GB显存（float32）？
- (b) 若序列长度翻倍到 $L=8192$，显存增加多少倍？

**练习20.2**（基础）

线性注意力的核函数近似将 $\exp(q^T k)$ 近似为 $\phi(q)^T \phi(k)$。

- (a) 使用 $\phi(x) = \text{elu}(x) + 1$ 的线性注意力计算公式推导：说明如何从 $O(L^2)$ 降低到 $O(L \cdot r)$，关键步骤是什么？
- (b) 为什么 $\phi(x)$ 需要保证输出非负？（提示：考虑归一化因子的含义）

---

### 中级题

**练习20.3**（中级）

Longformer使用滑动窗口 + 全局token的组合注意力。

- (a) 对于长度 $L=4096$、窗口大小 $w=512$、全局token数 $g=16$ 的配置，计算总的注意力操作数（以乘加操作计），并与完整注意力对比。
- (b) 在下面三种任务中，应该选择哪些token作为全局token？解释原因。
  - 文档分类任务
  - 问答任务（问题在文档前面拼接）
  - 命名实体识别任务

**练习20.4**（中级）

实现带**因果掩码**的滑动窗口注意力（只能关注当前位置之前的 $w$ 个token，用于语言模型）：

```python
def causal_sliding_window_attention(
    Q: torch.Tensor,  # (B, H, L, d)
    K: torch.Tensor,  # (B, H, L, d)
    V: torch.Tensor,  # (B, H, L, d)
    window_size: int
) -> torch.Tensor:  # (B, H, L, d)
    # 请实现此函数
    pass
```

要求：
- 位置 $i$ 只能关注 $[\max(0, i-w+1), i]$ 范围内的token
- 使用向量化操作（避免对 $L$ 的Python循环）

---

### 提高题

**练习20.5**（提高）

**线性注意力的精度分析与改进**

线性注意力在需要"精确匹配"的任务（如拷贝任务、联想记忆）上表现较差，原因是特征映射 $\phi$ 对 softmax 的近似质量不足。

- (a) **理论分析**：对于拷贝任务（输出等于某个特定位置的输入），softmax注意力如何实现精确定位？线性注意力为什么难以做到同样的事？（从注意力权重的分布角度分析）

- (b) **实验验证**：设计一个简单的拷贝任务（序列长度32，词表大小16），分别训练标准注意力层和线性注意力层，比较两者在测试集上的准确率。给出代码实现。

- (c) **改进方案**：提出至少两种缓解线性注意力精度问题的方法，并说明各自的优缺点。

---

## 练习答案

### 答案20.1

**(a) 注意力矩阵显存计算：**

单个头的注意力矩阵形状为 $(B, L, L) = (4, 4096, 4096)$。

$$\text{总元素数} = B \times H \times L \times L = 4 \times 8 \times 4096 \times 4096 = 536,870,912$$

$$\text{显存（float32，4字节）} = 536,870,912 \times 4 \text{ bytes} = 2,147,483,648 \text{ bytes} = 2 \text{ GB}$$

这仅是注意力矩阵本身。训练时还需要保存梯度（再乘以2），加上 $Q, K, V$ 投影等，实际占用约6-8 GB。

**(b) 序列长度翻倍的影响：**

$$\text{新显存} = B \times H \times (2L)^2 \times 4 = 4 \times \text{原显存}$$

序列长度翻倍，注意力矩阵显存增加 **4倍**（平方关系）。$L: 4096 \to 8192$，显存从2 GB增到8 GB。

---

### 答案20.2

**(a) 复杂度降低的关键步骤：**

原始计算：
$$\text{output}_i = \frac{\sum_j \phi(q_i)^T \phi(k_j) \cdot v_j}{\sum_j \phi(q_i)^T \phi(k_j)}$$

**关键观察**：$\phi(q_i)^T \phi(k_j)$ 可以分解为两个向量的内积，因此：

$$\sum_j \phi(q_i)^T \phi(k_j) \cdot v_j = \phi(q_i)^T \underbrace{\left(\sum_j \phi(k_j) v_j^T\right)}_{S \in \mathbb{R}^{r \times d}}$$

先计算累积矩阵 $S = \sum_j \phi(k_j) v_j^T$：需要 $L$ 次外积运算，每次 $O(rd)$，总计 $O(Lrd)$。

再计算 $\phi(q_i)^T S$：每次 $O(rd)$，共 $L$ 次，总计 $O(Lrd)$。

相比原来的 $O(L^2 d)$（必须先建立 $L \times L$ 矩阵再乘以 $V$），降为 $O(Lrd)$。当 $r \ll L$ 时效果显著。

**(b) 非负性的必要性：**

归一化因子 $\phi(q_i)^T z$（其中 $z = \sum_j \phi(k_j)$）在线性注意力中扮演着softmax分母的角色，应当为正值。

若 $\phi$ 允许负值，则 $\phi(q_i)^T \phi(k_j)$ 可能为负，导致：
- 注意力"权重"出现负值，失去概率分布的含义
- 归一化因子 $\phi(q_i)^T z$ 可能为零甚至负数，造成数值不稳定或除以零错误

因此 $\phi(x) = \text{elu}(x) + 1$ 保证输出恒正，使整个框架在数学上自洽。

---

### 答案20.3

**(a) 操作数计算：**

**滑动窗口部分**（非全局token）：
$$L_{\text{local}} = L - g = 4096 - 16 = 4080 \text{ 个非全局token}$$
$$\text{每个token关注} w = 512 \text{ 个位置}$$
$$\text{操作数}_{\text{local}} = L_{\text{local}} \times w = 4080 \times 512 = 2,088,960$$

**全局token的全局注意力**：
$$\text{操作数}_{\text{global}} = g \times L = 16 \times 4096 = 65,536$$

**非全局token关注全局token**：
$$\text{操作数}_{\text{ng \to g}} = L_{\text{local}} \times g = 4080 \times 16 = 65,280$$

$$\text{总操作数} = 2,088,960 + 65,536 + 65,280 \approx 2.2 \times 10^6$$

**完整注意力**：$L^2 = 4096^2 \approx 1.68 \times 10^7$

**加速比**：约 $\mathbf{7.6} \times$。

**(b) 全局token选择：**

- **文档分类任务**：选择 **[CLS] token**。分类标签从[CLS]的表示预测，它需要聚合整个文档的信息，天然需要全局注意力。

- **问答任务**：选择 **所有问题token**（通常在文档前部）。答案的定位依赖于将问题中的实体和概念与文档中的对应位置对齐，问题token需要全局可见以实现跨段落匹配。

- **命名实体识别任务**：通常**不需要**专门的全局token，或使用**[SEP] token**作为轻量级全局锚点。NER是局部的序列标注任务，每个token的标签主要取决于其局部上下文，滑动窗口通常已足够。若需要处理跨越长距离的实体（如代词消解），可将每段的第一个token设为全局token。

---

### 答案20.4

```python
def causal_sliding_window_attention(
    Q: torch.Tensor,  # (B, H, L, d)
    K: torch.Tensor,  # (B, H, L, d)
    V: torch.Tensor,  # (B, H, L, d)
    window_size: int
) -> torch.Tensor:  # (B, H, L, d)
    """
    因果滑动窗口注意力：位置i只关注[max(0, i-w+1), i]范围内的token。
    使用向量化的unfold操作避免Python循环。
    """
    B, H, L, d = Q.shape
    w = window_size

    # 将K和V在左侧填充(w-1)个零，使得每个位置i对应K[i-w+1:i+1]
    # 填充后: K_padded形状 (B, H, L+w-1, d)
    K_padded = F.pad(K, (0, 0, w - 1, 0))  # 在序列维度左侧填充
    V_padded = F.pad(V, (0, 0, w - 1, 0))

    # unfold: 提取每个位置的窗口
    # K_padded: (B, H, L+w-1, d) -> 转置为 (B, H, d, L+w-1)
    # unfold(dim=-1, size=w, step=1): (B, H, d, L, w)
    K_windows = K_padded.transpose(-2, -1).unfold(-1, w, 1)  # (B, H, d, L, w)
    V_windows = V_padded.transpose(-2, -1).unfold(-1, w, 1)  # (B, H, d, L, w)

    # 调整为 (B, H, L, w, d)
    K_windows = K_windows.permute(0, 1, 3, 4, 2)  # (B, H, L, w, d)
    V_windows = V_windows.permute(0, 1, 3, 4, 2)

    # Q: (B, H, L, d) -> (B, H, L, 1, d)
    Q_expanded = Q.unsqueeze(-2)

    # 注意力分数: (B, H, L, 1, w)
    scores = torch.matmul(Q_expanded, K_windows.transpose(-2, -1)) / math.sqrt(d)

    # 因果掩码：窗口中左侧w-1个填充位置应被屏蔽
    # 对于位置i，有效窗口大小为min(i+1, w)，前面的填充位置无效
    # 构建掩码: position j in window is valid if j >= (w - min(i+1, w))
    positions = torch.arange(L, device=Q.device)  # (L,)
    valid_len = (positions + 1).clamp(max=w)  # (L,)，每个位置的有效窗口大小

    # 创建掩码 (L, w): mask[i, j] = True if j >= w - valid_len[i]
    window_idx = torch.arange(w, device=Q.device).unsqueeze(0)  # (1, w)
    padding_len = (w - valid_len).unsqueeze(-1)  # (L, 1)
    causal_mask = window_idx < padding_len  # (L, w): True表示无效（填充）位置

    # 将掩码应用到分数
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-2), float('-inf'))

    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V_windows).squeeze(-2)  # (B, H, L, d)

    return out
```

**验证（可选）**：
```python
# 快速验证：输出形状正确，且位置0只关注位置0（只有一个有效token）
import torch.nn.functional as F
import math

B, H, L, d, w = 1, 1, 5, 4, 3
Q = torch.randn(B, H, L, d)
K = torch.randn(B, H, L, d)
V = torch.randn(B, H, L, d)
out = causal_sliding_window_attention(Q, K, V, w)
print(f"输出形状: {out.shape}")  # 应为 (1, 1, 5, 4)
```

---

### 答案20.5

**(a) 理论分析：精确匹配的困难**

**softmax注意力如何实现精确定位**：

对于拷贝任务，需要将注意力集中在某一个位置 $j^*$ 上。softmax的"指数放大"特性允许这一点：当 $q_i^T k_{j^*} \gg q_i^T k_j$（$j \neq j^*$）时：

$$\text{softmax}(q_i^T k_j / \sqrt{d})_{j=j^*} = \frac{e^{q_i^T k_{j^*}/\sqrt{d}}}{\sum_j e^{q_i^T k_j/\sqrt{d}}} \approx 1$$

softmax的这种"获胜者全得"（winner-takes-all）特性使得注意力权重可以任意接近独热分布。

**线性注意力的困难**：

线性注意力的权重为 $w_j = \phi(q_i)^T \phi(k_j) / \sum_l \phi(q_i)^T \phi(k_l)$。

要让 $w_{j^*} \approx 1$，需要：
$$\phi(q_i)^T \phi(k_{j^*}) \gg \phi(q_i)^T \phi(k_j) \text{ for all } j \neq j^*$$

但特征映射 $\phi$ 是有界函数（$\phi(x) = \text{elu}(x) + 1$ 在 $x > 0$ 时线性增长），内积无法像指数那样以任意大的倍率放大。因此线性注意力的权重分布天然趋向"平滑"，难以实现精确的单点选择。

从信息论角度：线性注意力相当于用有限维向量（秩为 $r$ 的矩阵）来存储所有key-value对的信息，当序列很长时，单个位置的信息被稀释。

**(b) 实验代码：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def run_copy_task_experiment():
    """在拷贝任务上对比标准注意力和线性注意力"""

    # 任务配置
    VOCAB_SIZE = 16
    SEQ_LEN = 32
    D_MODEL = 64
    NUM_HEADS = 4
    BATCH_SIZE = 64
    NUM_EPOCHS = 200

    # 生成拷贝任务数据：输入前半部分，目标是拷贝第一个token到最后
    def generate_data(n_samples):
        x = torch.randint(0, VOCAB_SIZE, (n_samples, SEQ_LEN))
        # 任务：输出序列每个位置拷贝输入的第一个token
        y = x[:, 0:1].expand(-1, SEQ_LEN)
        return x, y

    class CopyModel(nn.Module):
        def __init__(self, attn_type='standard'):
            super().__init__()
            self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
            if attn_type == 'standard':
                self.attn = StandardAttention(D_MODEL, NUM_HEADS)
            else:
                self.attn = LinearAttention(D_MODEL, NUM_HEADS)
            self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

        def forward(self, x):
            h = self.embed(x)   # (B, L, D)
            h = self.attn(h)    # (B, L, D)
            return self.head(h) # (B, L, VOCAB_SIZE)

    results = {}
    for attn_type in ['standard', 'linear']:
        model = CopyModel(attn_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(NUM_EPOCHS):
            x, y = generate_data(BATCH_SIZE)
            logits = model(x)  # (B, L, V)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 评估
        x_test, y_test = generate_data(500)
        with torch.no_grad():
            logits = model(x_test)
            preds = logits.argmax(dim=-1)  # (B, L)
            acc = (preds == y_test).float().mean().item()

        results[attn_type] = acc
        print(f"{attn_type:10s} 注意力在拷贝任务上的准确率: {acc:.1%}")

    return results

# 运行实验（注释掉以避免训练时间过长）
# results = run_copy_task_experiment()
```

典型结果：标准注意力准确率 $>95\%$，线性注意力通常在 $40\%-70\%$。

**(c) 改进方案：**

**方案1：混合注意力（Hybrid Attention）**

在不同层交替使用softmax注意力和线性注意力：前几层使用softmax（精确的局部依赖），后几层使用线性（全局聚合）。

- 优点：兼顾精度和效率，可以通过层比例调节权衡
- 缺点：需要调整softmax层的比例，工程复杂度增加

**方案2：改进特征映射（Enhanced Feature Map）**

使用更好的近似，如FAVOR+（正交随机特征）：

$$\phi(x) = \frac{1}{\sqrt{r}} \exp\left(\omega_i^T x - \frac{\|x\|^2}{2}\right), \quad \omega_i \sim \mathcal{N}(0, I)$$

正交约束使特征向量之间减少干扰，提升近似质量。

- 优点：理论上有更好的近似保证，无偏估计
- 缺点：训练时随机特征需要固定（推理时重新采样会破坏一致性）；维度 $r$ 需要足够大

**方案3：线性注意力 + 残差Softmax门控**

在线性注意力的输出上加一个轻量的softmax门控：

$$\text{output}_i = \lambda \cdot \text{LinearAttn}_i + (1-\lambda) \cdot \text{SoftmaxAttn}_i^{\text{local}}$$

其中 $\lambda$ 是可学习标量，局部softmax注意力只在小窗口内计算（低开销）。

- 优点：用少量softmax开销弥补线性注意力在精确任务上的劣势
- 缺点：增加了计算量（虽然仍远小于完整softmax注意力）；两路注意力的融合需要精细调优
