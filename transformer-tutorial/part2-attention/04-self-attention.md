# 第4章：自注意力机制

## 学习目标

完成本章学习后，你将能够：

1. **理解自注意力与普通注意力的区别**：掌握两者在 Query/Key/Value 来源上的本质差异
2. **掌握 Query、Key、Value 的线性投影**：理解为什么需要投影矩阵，以及参数量的计算
3. **理解自注意力的计算复杂度**：推导时间复杂度和空间复杂度，与 RNN 进行对比
4. **能够从零实现自注意力层**：用 PyTorch 完整实现，并逐步追踪每个张量的形状
5. **了解自注意力的优势和局限**：在实际应用中做出合理的架构选择

---

## 4.1 从注意力到自注意力

### 4.1.1 普通注意力的局限

在第2章中，我们学习了注意力机制的基本形式。在经典的序列到序列（Seq2Seq）模型中，注意力机制用于连接编码器和解码器：

- **Query（查询）**：来自解码器的当前隐状态
- **Key（键）**：来自编码器的所有隐状态
- **Value（值）**：来自编码器的所有隐状态

```
编码器输出序列: [h1, h2, h3, h4]  ← Key 和 Value 的来源
                       ↑
            注意力权重（跨序列）
                       ↓
解码器隐状态: [s_t]  ← Query 的来源
```

这种注意力机制解决了跨序列对齐的问题，但它只能建模**两个不同序列之间**的关系。如果我们想分析一个句子内部词语之间的关系，比如"猫追了老鼠，它很快"中的"它"指代"猫"还是"老鼠"，普通的编码器-解码器注意力就无能为力了。

### 4.1.2 自注意力的核心思想

**自注意力（Self-Attention）** 的关键创新是：让序列中的**每个位置都向序列中所有其他位置"查询"信息**。

> Q、K、V 全部来自同一个输入序列。

形式化地说，给定输入序列 $X \in \mathbb{R}^{L \times d_{model}}$，自注意力的 Q、K、V 均由 $X$ 经线性变换得到：

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中 $W^Q, W^K, W^V$ 是可学习的投影矩阵。

### 4.1.3 一个直观的例子

考虑句子："**银行**下午关门了，我明天再去那里**取钱**。"

使用自注意力时，"银行"这个词在计算自身的表示时，会同时考虑句子中所有其他词的信息：

| 当前词（Query） | 关注的词（Key） | 注意力权重 | 含义 |
|----------------|----------------|-----------|------|
| 银行 | 取钱 | **高** | "银行"与"取钱"强相关，是金融机构 |
| 银行 | 关门 | 中 | "关门"是银行的行为 |
| 银行 | 明天 | 低 | 时间修饰词，关联较弱 |
| 取钱 | 银行 | **高** | "取钱"的地点是"银行" |

通过这种方式，自注意力让"银行"和"取钱"的语义相互增强，模型能够消歧"银行"是金融机构而非河岸。

### 4.1.4 普通注意力与自注意力的对比

| 特征 | 普通注意力（编码器-解码器） | 自注意力 |
|------|--------------------------|---------|
| Query 来源 | 解码器隐状态 | 输入序列本身 |
| Key/Value 来源 | 编码器输出 | 输入序列本身 |
| 建模关系 | 跨序列对齐 | 序列内部依赖 |
| 典型用途 | 翻译、摘要的解码阶段 | 句子表示、上下文理解 |
| 位置限制 | Query 只看当前步 | 所有位置相互可见 |

---

## 4.2 Query、Key、Value 的线性投影

### 4.2.1 为什么不直接用原始输入？

一个自然的问题是：既然 Q、K、V 都来自同一个输入 $X$，为什么不直接让 $Q = K = V = X$？

原因有以下几点：

**1. 维度解耦**：原始 $d_{model}$ 维度可能过大，通过投影可以将查询空间（$d_k$）和值空间（$d_v$）设置为不同的维度。

**2. 功能分离**：通过不同的投影矩阵，模型可以学到：
   - $W^Q$：将输入变换为"我在找什么"的表示
   - $W^K$：将输入变换为"我能提供什么"的表示
   - $W^V$：将输入变换为"我实际携带的信息"的表示

**3. 增加表达能力**：投影矩阵引入了额外的可学习参数，让模型可以在不同子空间中捕捉不同类型的关系。

> 类比：图书馆检索系统中，书名（Key）和内容摘要（Value）是同一本书的不同表示；而你的检索词（Query）经过规范化后才与书名进行匹配。

### 4.2.2 投影矩阵的定义

设输入维度为 $d_{model}$，投影后的 Q/K 维度为 $d_k$，V 维度为 $d_v$，则三个投影矩阵为：

$$W^Q \in \mathbb{R}^{d_{model} \times d_k}, \quad W^K \in \mathbb{R}^{d_{model} \times d_k}, \quad W^V \in \mathbb{R}^{d_{model} \times d_v}$$

计算投影：

$$Q = XW^Q \in \mathbb{R}^{L \times d_k}$$
$$K = XW^K \in \mathbb{R}^{L \times d_k}$$
$$V = XW^V \in \mathbb{R}^{L \times d_v}$$

在原始 Transformer 论文中，通常取 $d_k = d_v = d_{model} / h$，其中 $h$ 是注意力头的数量（多头注意力在第5章讲解）。对于单头自注意力，常取 $d_k = d_v = d_{model}$。

### 4.2.3 参数量分析

三个投影矩阵的参数量（不含偏置）：

$$\text{参数量} = d_{model} \times d_k + d_{model} \times d_k + d_{model} \times d_v = 2 \cdot d_{model} \cdot d_k + d_{model} \cdot d_v$$

若 $d_k = d_v = d_{model}$：

$$\text{参数量} = 3 \times d_{model}^2$$

以 $d_{model} = 512$ 为例，三个投影矩阵共有 $3 \times 512^2 = 786,432 \approx 79$ 万参数。这是自注意力层中**唯一的可学习参数**（注意力计算本身不含参数）。

```python
# 验证参数量
import torch.nn as nn

d_model = 512
d_k = d_v = 512

W_q = nn.Linear(d_model, d_k, bias=False)
W_k = nn.Linear(d_model, d_k, bias=False)
W_v = nn.Linear(d_model, d_v, bias=False)

total_params = sum(p.numel() for p in [*W_q.parameters(),
                                        *W_k.parameters(),
                                        *W_v.parameters()])
print(f"三个投影矩阵参数量: {total_params:,}")
# 输出: 三个投影矩阵参数量: 786,432
print(f"理论计算: 3 × {d_model}² = {3 * d_model**2:,}")
# 输出: 理论计算: 3 × 512² = 786,432
```

---

## 4.3 自注意力的计算过程

### 4.3.1 完整计算公式

自注意力的完整计算公式为：

$$\text{SelfAttention}(X) = \text{softmax}\!\left(\frac{XW^Q (XW^K)^T}{\sqrt{d_k}}\right) XW^V$$

展开后等价于缩放点积注意力：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中 $Q = XW^Q$，$K = XW^K$，$V = XW^V$。

### 4.3.2 逐步计算过程

以下以批量大小 $B=2$，序列长度 $L=4$，模型维度 $d_{model}=8$，投影维度 $d_k=d_v=8$ 为例，逐步追踪张量形状。

**步骤 1：输入序列**

$$X \in \mathbb{R}^{B \times L \times d_{model}} = \mathbb{R}^{2 \times 4 \times 8}$$

**步骤 2：线性投影**

$$Q = XW^Q \in \mathbb{R}^{2 \times 4 \times 8}$$
$$K = XW^K \in \mathbb{R}^{2 \times 4 \times 8}$$
$$V = XW^V \in \mathbb{R}^{2 \times 4 \times 8}$$

**步骤 3：计算注意力分数**

$$\text{scores} = QK^T \in \mathbb{R}^{2 \times 4 \times 4}$$

注意：这里 $QK^T$ 是批量矩阵乘法（bmm），每个位置 $i$ 与所有位置 $j$ 的点积：

$$\text{scores}[b, i, j] = Q[b, i, :] \cdot K[b, j, :]^T$$

**步骤 4：缩放**

$$\text{scores\_scaled} = \frac{\text{scores}}{\sqrt{d_k}} \in \mathbb{R}^{2 \times 4 \times 4}$$

缩放因子 $\sqrt{d_k}$ 防止点积值过大导致 softmax 梯度消失（详见第2章）。

**步骤 5：Softmax 归一化**

$$\text{weights} = \text{softmax}(\text{scores\_scaled}) \in \mathbb{R}^{2 \times 4 \times 4}$$

对最后一个维度（$j$ 维度）做 softmax，每行（每个 Query 位置）的权重之和为 1。

**步骤 6：加权求和**

$$\text{output} = \text{weights} \cdot V \in \mathbb{R}^{2 \times 4 \times 8}$$

输出序列与输入序列形状相同，每个位置的表示是所有位置 Value 的加权平均。

### 4.3.3 张量形状变化总结

```
输入 X:            (B, L, d_model)  = (2, 4, 8)
         ↓ 线性投影 W^Q, W^K, W^V
Q, K, V:           (B, L, d_k)      = (2, 4, 8)
         ↓ Q @ K^T（批量矩阵乘）
scores:            (B, L, L)         = (2, 4, 4)
         ↓ 除以 √d_k
scores_scaled:     (B, L, L)         = (2, 4, 4)
         ↓ softmax(dim=-1)
weights:           (B, L, L)         = (2, 4, 4)
         ↓ weights @ V（批量矩阵乘）
输出 output:       (B, L, d_v)       = (2, 4, 8)
```

注意力权重矩阵 `weights` 的形状是 $(B, L, L)$，这是自注意力**二次复杂度**的直观体现：每个位置都要与所有其他位置计算相似度。

---

## 4.4 复杂度分析

### 4.4.1 时间复杂度

自注意力的计算瓶颈在于注意力分数矩阵 $QK^T$ 的计算：

- $Q \in \mathbb{R}^{L \times d_k}$，$K \in \mathbb{R}^{L \times d_k}$
- 矩阵乘法 $QK^T$ 需要 $O(L^2 \cdot d_k)$ 次乘法

同时，加权求和 $\text{weights} \cdot V$ 也是 $O(L^2 \cdot d_v)$。

忽略常数，总时间复杂度为：

$$T_{\text{self-attn}} = O(L^2 \cdot d)$$

其中 $d = \max(d_k, d_v)$。

**线性投影的代价**：三次矩阵乘法各需 $O(L \cdot d_{model} \cdot d_k)$，即 $O(L \cdot d^2)$。当序列较短（$L < d$）时，线性投影是主要计算开销；当序列很长（$L \gg d$）时，注意力计算成为主要瓶颈。

### 4.4.2 空间复杂度

需要存储以下中间量：

| 中间量 | 形状 | 空间 |
|--------|------|------|
| Q, K, V 矩阵 | $(L \times d_k)$ × 3 | $O(L \cdot d)$ |
| 注意力分数矩阵 | $L \times L$ | $O(L^2)$ |
| 注意力权重矩阵 | $L \times L$ | $O(L^2)$ |
| 输出矩阵 | $L \times d_v$ | $O(L \cdot d)$ |

总空间复杂度：

$$S_{\text{self-attn}} = O(L^2 + L \cdot d)$$

当序列很长时，$O(L^2)$ 的注意力矩阵是主要内存瓶颈。

### 4.4.3 与 RNN 的复杂度对比

| 模型 | 时间复杂度（每层） | 空间复杂度 | 最长路径长度 | 并行度 |
|------|--------------------|-----------|-------------|--------|
| RNN/LSTM | $O(L \cdot d^2)$ | $O(L \cdot d)$ | $O(L)$ | 无（串行） |
| 自注意力 | $O(L^2 \cdot d)$ | $O(L^2 + L \cdot d)$ | $O(1)$ | 完全并行 |
| 卷积（局部） | $O(k \cdot L \cdot d^2)$ | $O(L \cdot d)$ | $O(\log_k L)$ | 部分并行 |

**最长路径长度**（Maximum Path Length）是衡量模型捕捉长距离依赖能力的关键指标：
- RNN 中，位置 1 和位置 $L$ 之间的信息需要经过 $L-1$ 步才能传递，梯度消失风险高
- 自注意力中，任意两个位置只需 1 步即可直接交互，极大地缓解了长距离依赖问题

### 4.4.4 长序列的计算瓶颈

当序列长度 $L$ 很大时，$O(L^2)$ 的复杂度会导致严重的性能问题：

```python
# 不同序列长度下的注意力矩阵大小（以 float32 计算）
import torch

for L in [128, 512, 1024, 4096, 16384]:
    # 每个元素 4 bytes（float32）
    memory_mb = L * L * 4 / (1024 ** 2)
    print(f"L={L:6d}: 注意力矩阵 = {L}×{L} = {memory_mb:.1f} MB")

# 输出:
# L=   128: 注意力矩阵 = 128×128 = 0.1 MB
# L=   512: 注意力矩阵 = 512×512 = 1.0 MB
# L=  1024: 注意力矩阵 = 1024×1024 = 4.0 MB
# L=  4096: 注意力矩阵 = 4096×4096 = 64.0 MB
# L= 16384: 注意力矩阵 = 16384×16384 = 1024.0 MB
```

可以看到，$L = 16384$ 时，单张注意力矩阵就占 1 GB。在批量训练（多头 + 多层 + 批量大小）时，内存压力会更加突出。这也是近年来出现 Longformer、BigBird、Flash Attention 等高效注意力方法的原因（详见第20章）。

---

## 4.5 自注意力的优势与局限

### 4.5.1 核心优势

**1. 完全并行计算**

与 RNN 不同，自注意力中每个位置的计算不依赖于其他位置的结果（在推理时不计，训练时 $QK^T$ 可以一次性完成），因此可以高效地利用 GPU/TPU 的并行计算能力。

```
RNN（串行）:   x1 → h1 → h2 → h3 → h4    (必须按顺序)
              x2 ↗      x3 ↗      x4 ↗

自注意力（并行）:
x1, x2, x3, x4 → 同时计算所有 QKV → 同时计算所有注意力分数
```

**2. 全局感受野**

每个位置在计算表示时可以直接看到序列中的所有其他位置，感受野是整个序列。这使得模型天然具备建模长距离依赖的能力。

**3. 动态权重**

注意力权重根据输入内容动态计算，而不是固定的（如卷积核的权重）。同一个"银行"在不同上下文中会有不同的注意力分布，体现了动态的上下文感知能力。

**4. 可解释性强**

注意力权重矩阵 $(L \times L)$ 可以直接可视化，直观地展示模型关注了哪些位置，便于分析和调试。

### 4.5.2 主要局限

**1. 二次复杂度**

$O(L^2 \cdot d)$ 的时间复杂度和 $O(L^2)$ 的空间复杂度是最主要的局限，处理长文档、高分辨率图像等长序列时代价极高。

**2. 缺少位置信息**

自注意力本身是**位置无关**的（Permutation-Invariant）：将输入序列打乱顺序，输出仍然是对应打乱顺序的版本。这意味着模型无法区分"狗咬了人"和"人咬了狗"。

这就是为什么 Transformer 需要**位置编码**（第3章）来显式地注入位置信息。

**3. 对短序列效率不高**

当序列很短时，$O(L^2 \cdot d)$ 中的 $L^2$ 很小，但线性投影的 $O(L \cdot d^2)$ 仍然不变。此时 RNN 或 CNN 可能更高效。

**4. 推理时无法增量更新**

标准的自注意力在每次推理时需要重新计算所有位置的 QKV（不过可以用 KV Cache 缓存历史，这是 GPT 等模型推理加速的核心技术，详见第23章）。

### 4.5.3 实际应用中的权衡

| 场景 | 序列长度 | 推荐方案 |
|------|---------|---------|
| 短文本分类（<512 词） | 短 | 标准自注意力，性能优先 |
| 长文档理解（>2K 词） | 长 | 稀疏注意力（Longformer）或分块处理 |
| 代码生成（几千 tokens） | 中长 | Flash Attention 优化 |
| 高分辨率图像（ViT） | 极长 | 分块 Patch + 局部注意力 |
| 实时推理服务 | 中 | KV Cache + 量化 |

---

## 本章小结

| 概念 | 定义 | 关键公式/参数 |
|------|------|-------------|
| 自注意力 | Q、K、V 来自同一序列的注意力 | $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$ |
| 线性投影 | 将输入映射到 Q/K/V 空间 | $Q=XW^Q$，$K=XW^K$，$V=XW^V$ |
| 投影矩阵维度 | Q/K 投影到 $d_k$，V 投影到 $d_v$ | 通常 $d_k=d_v=d_{model}$ |
| 缩放因子 | 防止点积过大导致梯度消失 | 除以 $\sqrt{d_k}$ |
| 时间复杂度 | 注意力矩阵计算 $QK^T$ | $O(L^2 \cdot d)$ |
| 空间复杂度 | 存储注意力矩阵 | $O(L^2 + L \cdot d)$ |
| 最长路径 | 任意两位置交互所需步数 | $O(1)$（优于 RNN 的 $O(L)$） |
| 位置无关性 | 自注意力本身不含位置信息 | 需配合位置编码使用 |
| 参数量 | 仅在线性投影中 | $3 \times d_{model} \times d_k$（不含偏置） |

---

## 代码实战

### 完整实现：从零构建自注意力层

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'  # 支持中文显示（macOS 可改为 'Arial Unicode MS'）
matplotlib.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# Part 1：核心函数：缩放点积注意力
# ─────────────────────────────────────────────

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    缩放点积注意力。

    参数:
        query:     (B, L_q, d_k)
        key:       (B, L_k, d_k)
        value:     (B, L_v, d_v)  — 通常 L_k == L_v
        mask:      (B, L_q, L_k) 或可广播形状，True 表示屏蔽该位置
        dropout_p: 注意力权重上的 dropout 概率

    返回:
        output:  (B, L_q, d_v)
        weights: (B, L_q, L_k)
    """
    d_k = query.size(-1)

    # 步骤1：计算注意力分数：(B, L_q, d_k) × (B, d_k, L_k) → (B, L_q, L_k)
    scores = torch.bmm(query, key.transpose(-2, -1))
    print(f"  [步骤1] scores 形状: {scores.shape}  (B, L_q, L_k)")

    # 步骤2：缩放
    scores = scores / math.sqrt(d_k)
    print(f"  [步骤2] scaled scores 形状: {scores.shape}  ÷ √{d_k}={math.sqrt(d_k):.2f}")

    # 步骤3：应用掩码（可选）
    if mask is not None:
        # mask=True 的位置填充 -inf，softmax 后趋近于 0
        scores = scores.masked_fill(mask, float('-inf'))
        print(f"  [步骤3] 应用掩码后 scores 形状: {scores.shape}")

    # 步骤4：Softmax 归一化（对最后一维）
    weights = F.softmax(scores, dim=-1)
    print(f"  [步骤4] attention weights 形状: {weights.shape}  (B, L_q, L_k)")

    # 步骤5：Dropout（仅训练阶段）
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p)

    # 步骤6：加权求和：(B, L_q, L_k) × (B, L_k, d_v) → (B, L_q, d_v)
    output = torch.bmm(weights, value)
    print(f"  [步骤6] output 形状: {output.shape}  (B, L_q, d_v)")

    return output, weights


# ─────────────────────────────────────────────
# Part 2：自注意力层
# ─────────────────────────────────────────────

class SelfAttention(nn.Module):
    """
    单头自注意力层。

    参数:
        d_model: 输入/输出的特征维度
        d_k:     Q、K 的投影维度（默认等于 d_model）
        d_v:     V 的投影维度（默认等于 d_model）
        dropout: 注意力权重上的 dropout 概率
        bias:    投影矩阵是否含偏置
    """

    def __init__(
        self,
        d_model: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model
        self.dropout = dropout

        # 三个线性投影层
        self.W_q = nn.Linear(d_model, self.d_k, bias=bias)
        self.W_k = nn.Linear(d_model, self.d_k, bias=bias)
        self.W_v = nn.Linear(d_model, self.d_v, bias=bias)

        # Xavier 均匀初始化（PyTorch 默认也是这个，显式写出更清晰）
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x:    (B, L, d_model)  输入序列
            mask: (B, L, L) 或 None，True 表示屏蔽该位置

        返回:
            output:  (B, L, d_v)
            weights: (B, L, L)
        """
        B, L, _ = x.shape
        print(f"\n[SelfAttention.forward]")
        print(f"  输入 x 形状: {x.shape}  (B={B}, L={L}, d_model={self.d_model})")

        # ── 线性投影 ──
        Q = self.W_q(x)   # (B, L, d_k)
        K = self.W_k(x)   # (B, L, d_k)
        V = self.W_v(x)   # (B, L, d_v)
        print(f"  Q 形状: {Q.shape}  K 形状: {K.shape}  V 形状: {V.shape}")

        # ── 缩放点积注意力 ──
        output, weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_p=self.dropout if self.training else 0.0
        )

        return output, weights

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, d_k={self.d_k}, "
                f"d_v={self.d_v}, dropout={self.dropout}")


# ─────────────────────────────────────────────
# Part 3：单步调试——形状追踪
# ─────────────────────────────────────────────

def demo_shape_trace():
    print("=" * 60)
    print("【演示1】张量形状逐步追踪")
    print("=" * 60)

    torch.manual_seed(42)
    B, L, d_model = 2, 4, 8

    # 构造输入
    x = torch.randn(B, L, d_model)
    print(f"\n输入序列 x: {x.shape}")

    # 初始化自注意力
    attn = SelfAttention(d_model=d_model, d_k=8, d_v=8)
    attn.eval()

    output, weights = attn(x)

    print(f"\n最终输出 output: {output.shape}")
    print(f"注意力权重 weights: {weights.shape}")

    # 验证权重行和为 1
    weight_sum = weights.sum(dim=-1)
    print(f"\n注意力权重行和（应为全1）:\n{weight_sum}")


# ─────────────────────────────────────────────
# Part 4：与 nn.MultiheadAttention 对比验证
# ─────────────────────────────────────────────

def demo_compare_with_pytorch():
    """
    将本章实现与 PyTorch 官方 nn.MultiheadAttention（num_heads=1）对比。
    注意：PyTorch 的实现含有输出投影 W_o，本实现无输出投影，
    因此我们手动对齐权重来验证核心注意力计算逻辑是否一致。
    """
    print("\n" + "=" * 60)
    print("【演示2】与 nn.MultiheadAttention 对比（逻辑验证）")
    print("=" * 60)

    torch.manual_seed(0)
    B, L, d_model = 1, 5, 16
    x = torch.randn(B, L, d_model)

    # ── 本章实现 ──
    my_attn = SelfAttention(d_model=d_model, d_k=d_model, d_v=d_model, bias=False)
    my_attn.eval()

    # ── PyTorch 官方实现（单头，无偏置，无输出投影的等效写法）──
    # nn.MultiheadAttention 内部有 in_proj_weight (3*d_model, d_model) 和
    # out_proj，我们手动设置 in_proj_weight 与本章实现一致
    pytorch_attn = nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=1,
        bias=False,
        batch_first=True,   # 使用 (B, L, d) 格式
    )
    pytorch_attn.eval()

    # 将 PyTorch 的 in_proj_weight 拆分为 W_q, W_k, W_v 并与本章实现对齐
    with torch.no_grad():
        # PyTorch 的 in_proj_weight 形状: (3*d_model, d_model)
        wq = my_attn.W_q.weight  # (d_k, d_model)
        wk = my_attn.W_k.weight  # (d_k, d_model)
        wv = my_attn.W_v.weight  # (d_v, d_model)
        pytorch_attn.in_proj_weight.copy_(torch.cat([wq, wk, wv], dim=0))

        # 输出投影设为单位矩阵（等效于无输出投影）
        nn.init.eye_(pytorch_attn.out_proj.weight)

    # ── 前向传播 ──
    with torch.no_grad():
        my_output, my_weights = my_attn(x)
        # PyTorch 接口：query, key, value 分别传入
        pt_output, pt_weights = pytorch_attn(x, x, x, need_weights=True, average_attn_weights=False)

    print(f"\n本章实现  output 形状: {my_output.shape}")
    print(f"PyTorch   output 形状: {pt_output.shape}")

    # 比较注意力权重（注意 PyTorch 返回 (B, num_heads, L, L)，本章返回 (B, L, L)）
    pt_weights_squeezed = pt_weights.squeeze(1)  # (B, 1, L, L) → (B, L, L)
    max_diff_weights = (my_weights - pt_weights_squeezed).abs().max().item()
    max_diff_output = (my_output - pt_output).abs().max().item()

    print(f"\n注意力权重最大绝对误差: {max_diff_weights:.2e}")
    print(f"输出最大绝对误差:         {max_diff_output:.2e}")

    if max_diff_weights < 1e-5 and max_diff_output < 1e-5:
        print("验证通过：本章实现与 PyTorch 官方结果一致！")
    else:
        print("注意：存在差异，请检查权重对齐方式（可能由输出投影引起）。")


# ─────────────────────────────────────────────
# Part 5：注意力权重可视化
# ─────────────────────────────────────────────

def demo_attention_visualization():
    """
    用一个简单的例子可视化注意力权重矩阵。
    """
    print("\n" + "=" * 60)
    print("【演示3】注意力权重可视化")
    print("=" * 60)

    torch.manual_seed(7)
    tokens = ["猫", "追", "了", "老鼠"]
    L = len(tokens)
    d_model = 16

    # 构造随机输入（实际中是词嵌入）
    x = torch.randn(1, L, d_model)

    attn = SelfAttention(d_model=d_model, d_k=16, d_v=16)
    attn.eval()

    with torch.no_grad():
        output, weights = attn(x)

    # weights 形状: (1, L, L)，去掉批次维度
    weights_np = weights.squeeze(0).numpy()

    print(f"\n注意力权重矩阵 ({L}×{L}):")
    print("行=Query（当前词），列=Key（被关注的词）")
    print(f"{'':>6}", end="")
    for t in tokens:
        print(f"{t:>8}", end="")
    print()
    for i, t_q in enumerate(tokens):
        print(f"{t_q:>6}", end="")
        for j in range(L):
            print(f"{weights_np[i, j]:>8.3f}", end="")
        print()

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(weights_np, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels(tokens, fontsize=12)
    ax.set_yticklabels(tokens, fontsize=12)
    ax.set_xlabel("Key（被关注的词）", fontsize=11)
    ax.set_ylabel("Query（当前词）", fontsize=11)
    ax.set_title("自注意力权重热力图", fontsize=13)
    plt.colorbar(im, ax=ax)

    # 在每个格子内标注数值
    for i in range(L):
        for j in range(L):
            ax.text(j, i, f"{weights_np[i, j]:.2f}",
                    ha='center', va='center', fontsize=10,
                    color='white' if weights_np[i, j] > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig("self_attention_weights.png", dpi=150, bbox_inches='tight')
    print("\n热力图已保存至 self_attention_weights.png")
    plt.show()


# ─────────────────────────────────────────────
# Part 6：复杂度测试
# ─────────────────────────────────────────────

def demo_complexity():
    """
    测量不同序列长度下自注意力的实际运行时间，验证 O(L²·d) 的复杂度。
    """
    print("\n" + "=" * 60)
    print("【演示4】不同序列长度的运行时间（验证 O(L²·d) 复杂度）")
    print("=" * 60)

    import time

    d_model = 64
    B = 1
    lengths = [64, 128, 256, 512, 1024]

    attn = SelfAttention(d_model=d_model)
    attn.eval()

    print(f"\n{'序列长度 L':>12} {'理论 L²':>10} {'实测时间(ms)':>14} {'时间比（相对L=64）':>18}")
    print("-" * 60)

    times = []
    for L in lengths:
        x = torch.randn(B, L, d_model)
        # 预热
        with torch.no_grad():
            for _ in range(3):
                attn(x)
        # 计时
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                attn(x)
        elapsed = (time.perf_counter() - start) / 20 * 1000  # ms
        times.append(elapsed)

    base_time = times[0]
    for L, t in zip(lengths, times):
        l2_ratio = (L / lengths[0]) ** 2
        t_ratio = t / base_time
        print(f"{L:>12} {L**2:>10,} {t:>14.3f} {t_ratio:>18.2f}x  (理论 {l2_ratio:.1f}x)")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == "__main__":
    demo_shape_trace()
    demo_compare_with_pytorch()
    demo_attention_visualization()
    demo_complexity()
```

### 预期输出

运行 `demo_shape_trace()` 后，你会看到：

```
============================================================
【演示1】张量形状逐步追踪
============================================================

[SelfAttention.forward]
  输入 x 形状: torch.Size([2, 4, 8])  (B=2, L=4, d_model=8)
  Q 形状: torch.Size([2, 4, 8])  K 形状: torch.Size([2, 4, 8])  V 形状: torch.Size([2, 4, 8])
  [步骤1] scores 形状: torch.Size([2, 4, 4])  (B, L_q, L_k)
  [步骤2] scaled scores 形状: torch.Size([2, 4, 4])  ÷ √8=2.83
  [步骤4] attention weights 形状: torch.Size([2, 4, 4])  (B, L_q, L_k)
  [步骤6] output 形状: torch.Size([2, 4, 8])  (B, L_q, d_v)

最终输出 output: torch.Size([2, 4, 8])
注意力权重 weights: torch.Size([2, 4, 4])

注意力权重行和（应为全1）:
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.]])
```

---

## 练习题

### 基础题

**练习 4.1（基础）**

给定输入序列 $X \in \mathbb{R}^{3 \times 4}$（序列长度为3，特征维度为4），投影矩阵 $W^Q, W^K \in \mathbb{R}^{4 \times 2}$，$W^V \in \mathbb{R}^{4 \times 4}$，手动计算（不用代码）自注意力输出的形状，并写出每一步中间张量的形状。

---

**练习 4.2（基础）**

解释为什么计算注意力分数后需要除以 $\sqrt{d_k}$，而不是 $d_k$ 或 $\sqrt{d_{model}}$。如果不做缩放会发生什么？请用代码验证：当 $d_k$ 很大时（如 $d_k = 512$），不缩放和缩放后的 softmax 输出分布有何不同。

---

### 中级题

**练习 4.3（中级）**

在本章的 `SelfAttention` 类基础上，添加一个**输出投影层** $W^O \in \mathbb{R}^{d_v \times d_{model}}$，使得最终输出维度重新变为 $d_{model}$（即使 $d_v \neq d_{model}$）。修改 `forward` 方法并验证：

```python
attn = SelfAttentionWithOutputProjection(d_model=16, d_k=8, d_v=8)
x = torch.randn(2, 5, 16)
output, weights = attn(x)
assert output.shape == (2, 5, 16)   # 输出维度恢复为 d_model
assert weights.shape == (2, 5, 5)
```

---

**练习 4.4（中级）**

实现**带因果掩码（Causal Mask）的自注意力**，使得每个位置只能关注当前及之前的位置（即未来的位置被屏蔽）。这是 GPT 等自回归语言模型的核心机制。

要求：
1. 生成下三角因果掩码矩阵
2. 集成到 `SelfAttention.forward` 中（通过 `causal=True` 参数控制）
3. 验证屏蔽后注意力权重矩阵的上三角部分（不含对角线）为 0

```python
attn = SelfAttention(d_model=8, causal=True)
x = torch.randn(1, 4, 8)
output, weights = attn(x)
# weights[0] 应为下三角矩阵（上三角含对角线以上部分为0）
print(weights[0])
```

---

### 提高题

**练习 4.5（提高）**

**复现论文中的自注意力可视化实验**

参考 "Attention Is All You Need" 论文（第3.2节），对以下句子完成一个迷你实验：

```python
sentence = "The animal didn't cross the street because it was too tired"
```

1. 使用 Hugging Face 加载 BERT-base-uncased 模型，提取第一层第一个注意力头的权重
2. 可视化注意力权重矩阵（热力图）
3. 找出 "it" 这个词最关注哪些词（权重前3名），并解释模型是否正确识别了指代关系
4. 对比不同层（第1层、第6层、第12层）的注意力模式，描述你的发现

提示：
```python
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
inputs = tokenizer(sentence, return_tensors='pt')
outputs = model(**inputs)
# outputs.attentions: tuple of (B, num_heads, L, L) per layer
```

---

## 练习答案

### 答案 4.1

**逐步追踪形状：**

- 输入：$X \in \mathbb{R}^{3 \times 4}$
- 步骤1（线性投影）：
  - $Q = XW^Q \in \mathbb{R}^{3 \times 2}$（$3 \times 4$ 乘 $4 \times 2$）
  - $K = XW^K \in \mathbb{R}^{3 \times 2}$
  - $V = XW^V \in \mathbb{R}^{3 \times 4}$（$3 \times 4$ 乘 $4 \times 4$）
- 步骤2（注意力分数）：$QK^T \in \mathbb{R}^{3 \times 3}$（$3 \times 2$ 乘 $2 \times 3$）
- 步骤3（缩放）：$\mathbb{R}^{3 \times 3}$，形状不变
- 步骤4（softmax）：$\mathbb{R}^{3 \times 3}$，形状不变
- 步骤5（加权求和）：$\mathbb{R}^{3 \times 4}$（$3 \times 3$ 乘 $3 \times 4$）

最终输出形状：$\mathbb{R}^{3 \times 4}$，与 $V$ 的形状相同。

---

### 答案 4.2

**为什么除以 $\sqrt{d_k}$？**

当 $d_k$ 很大时，$Q$ 和 $K$ 的点积 $Q \cdot K^T = \sum_{i=1}^{d_k} q_i k_i$ 是 $d_k$ 个随机变量之和。若 $q_i, k_i \sim \mathcal{N}(0,1)$，则点积的方差为 $d_k$，标准差为 $\sqrt{d_k}$。

- 不除 $\sqrt{d_k}$：点积值随 $d_k$ 增大，softmax 输入极端化，梯度接近 0（one-hot 分布）
- 除以 $d_k$：过度缩放，注意力权重趋于均匀分布，失去区分能力
- 除以 $\sqrt{d_k}$：将方差归一化为 1，既保留了区分能力，又避免了梯度消失

代码验证：

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
d_k = 512
q = torch.randn(1, 1, d_k)
k = torch.randn(1, 10, d_k)

# 不缩放
scores_raw = torch.bmm(q, k.transpose(-2, -1)).squeeze()
weights_raw = F.softmax(scores_raw, dim=-1)

# 缩放
scores_scaled = scores_raw / (d_k ** 0.5)
weights_scaled = F.softmax(scores_scaled, dim=-1)

print(f"不缩放 - 分数范围: [{scores_raw.min():.1f}, {scores_raw.max():.1f}]")
print(f"不缩放 - 最大权重: {weights_raw.max():.4f}（接近1.0，近似one-hot）")
print(f"缩放后 - 分数范围: [{scores_scaled.min():.1f}, {scores_scaled.max():.1f}]")
print(f"缩放后 - 最大权重: {weights_scaled.max():.4f}（分布更均匀）")
# 输出示例:
# 不缩放 - 分数范围: [-45.3, 52.8]
# 不缩放 - 最大权重: 1.0000（近似one-hot）
# 缩放后 - 分数范围: [-2.0, 2.3]
# 缩放后 - 最大权重: 0.3521（分布更均匀）
```

---

### 答案 4.3

```python
class SelfAttentionWithOutputProjection(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        # 输出投影：将 d_v 映射回 d_model
        self.W_o = nn.Linear(d_v, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Q = self.W_q(x)  # (B, L, d_k)
        K = self.W_k(x)  # (B, L, d_k)
        V = self.W_v(x)  # (B, L, d_v)

        # 缩放点积注意力
        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        weights = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            weights = F.dropout(weights, p=self.dropout)

        context = torch.bmm(weights, V)  # (B, L, d_v)

        # 输出投影：(B, L, d_v) → (B, L, d_model)
        output = self.W_o(context)
        return output, weights


# 验证
attn = SelfAttentionWithOutputProjection(d_model=16, d_k=8, d_v=8)
x = torch.randn(2, 5, 16)
output, weights = attn(x)
assert output.shape == (2, 5, 16), f"期望 (2,5,16)，实际 {output.shape}"
assert weights.shape == (2, 5, 5), f"期望 (2,5,5)，实际 {weights.shape}"
print(f"输出形状: {output.shape}  验证通过！")
```

---

### 答案 4.4

```python
class SelfAttentionCausal(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.d_k = d_model
        self.dropout = dropout

    def forward(self, x: torch.Tensor, causal: bool = True):
        B, L, _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if causal:
            # 生成上三角掩码（不含对角线），True 表示屏蔽
            # triu(k=1): 对角线以上的元素为 True
            causal_mask = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1
            )
            # 扩展到 batch 维度: (1, L, L) 可广播
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        weights = F.softmax(scores, dim=-1)
        output = torch.bmm(weights, V)
        return output, weights


# 验证
torch.manual_seed(0)
attn = SelfAttentionCausal(d_model=8)
attn.eval()
x = torch.randn(1, 4, 8)
output, weights = attn(x, causal=True)

print("因果掩码注意力权重矩阵（应为下三角）:")
print(weights[0].detach().numpy().round(3))

# 验证上三角为 0（位置 i 不能关注位置 j>i）
upper_tri = weights[0].triu(diagonal=1)
assert upper_tri.abs().max().item() < 1e-6, "上三角不为零，掩码有误！"
print("验证通过：上三角元素全为 0。")

# 示例输出:
# [[1.000 0.000 0.000 0.000]
#  [0.312 0.688 0.000 0.000]
#  [0.211 0.439 0.350 0.000]
#  [0.087 0.352 0.289 0.272]]
```

---

### 答案 4.5

```python
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
import numpy as np

sentence = "The animal didn't cross the street because it was too tired"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
model.eval()

inputs = tokenizer(sentence, return_tensors='pt')
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions: 12层，每层 (1, 12, L, L)
attentions = outputs.attentions  # tuple of 12 tensors

# 找 "it" 的位置
it_idx = tokens.index('it')
print(f"Token 列表: {tokens}")
print(f"'it' 的位置索引: {it_idx}")

# 查看第6层第1个头中 "it" 的注意力分布
layer_idx = 5   # 第6层（0-indexed）
head_idx = 0    # 第1个头

it_attn = attentions[layer_idx][0, head_idx, it_idx, :].numpy()
top3_idx = it_attn.argsort()[-3:][::-1]
print(f"\n第6层第1头中，'it' 最关注的3个词:")
for rank, idx in enumerate(top3_idx, 1):
    print(f"  {rank}. '{tokens[idx]}' (位置 {idx})  权重 = {it_attn[idx]:.4f}")

# 可视化三层的注意力（第1、6、12层，第1头）
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, layer_i, title in zip(axes, [0, 5, 11],
                               ["第1层", "第6层", "第12层"]):
    attn_matrix = attentions[layer_i][0, 0].numpy()  # (L, L)
    im = ax.imshow(attn_matrix, cmap='Blues')
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_title(f"{title}（第1头）", fontsize=12)
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("bert_attention_layers.png", dpi=150, bbox_inches='tight')
plt.show()

# 典型发现：
# - 第1层：注意力主要集中在邻近词（局部句法）
# - 第6层：'it' 开始关注 'animal'（指代消解）
# - 第12层：注意力更加分散，捕捉高层语义
```

---

[下一章：多头注意力](./05-multi-head-attention.md)

[返回目录](../README.md)
