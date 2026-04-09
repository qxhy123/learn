# 第8章：解码器结构

## 学习目标

完成本章学习后，你将能够：

1. **理解解码器与编码器的区别** —— 掌握解码器三层结构与编码器的本质差异
2. **掌握自回归生成的原理** —— 理解序列生成时"一步一步预测下一个词"的机制
3. **理解编码器-解码器注意力（交叉注意力）** —— 掌握 Q、K、V 分别来自哪里及其作用
4. **能够从零实现完整的解码器** —— 用 PyTorch 手写 DecoderLayer 和 TransformerDecoder
5. **了解解码器在不同任务中的应用** —— 机器翻译、文本摘要、代码生成等场景

---

## 8.1 解码器整体结构

### 8.1.1 解码器层的三个子层

在原始 Transformer 论文中，**编码器**由 6 层堆叠，每层包含 2 个子层：
1. 多头自注意力（Multi-Head Self-Attention）
2. 前馈网络（Feed-Forward Network）

**解码器**同样由 6 层堆叠，但每层包含 **3 个子层**：

| 子层 | 名称 | 作用 |
|------|------|------|
| 子层 1 | 掩码多头自注意力（Masked Multi-Head Self-Attention） | 解码器内部词之间的关系，但不能看未来 |
| 子层 2 | 编码器-解码器注意力（Encoder-Decoder Attention） | 将编码器的语义信息引入解码器 |
| 子层 3 | 前馈网络（Feed-Forward Network） | 非线性变换，增强表达能力 |

每个子层之后同样使用**残差连接**和**层归一化**（Layer Normalization）：

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

### 8.1.2 与编码器的对比

```
编码器（Encoder）层结构：
┌─────────────────────────────┐
│  输入嵌入 + 位置编码         │
│           ↓                 │
│  ┌─────────────────────┐    │
│  │ 多头自注意力         │    │
│  └─────────┬───────────┘    │
│            ↓                │
│       残差 + LayerNorm       │
│            ↓                │
│  ┌─────────────────────┐    │
│  │ 前馈网络             │    │
│  └─────────┬───────────┘    │
│            ↓                │
│       残差 + LayerNorm       │
│            ↓                │
│         输出                │
└─────────────────────────────┘

解码器（Decoder）层结构：
┌─────────────────────────────────────┐
│  目标序列嵌入 + 位置编码             │
│           ↓                         │
│  ┌─────────────────────────────┐    │
│  │ 掩码多头自注意力（因果掩码）  │    │
│  └─────────┬───────────────────┘    │
│            ↓                        │
│       残差 + LayerNorm               │
│            ↓                        │
│  ┌─────────────────────────────┐    │
│  │ 编码器-解码器注意力          │ ←── 来自编码器的 K, V
│  └─────────┬───────────────────┘    │
│            ↓                        │
│       残差 + LayerNorm               │
│            ↓                        │
│  ┌─────────────────────────────┐    │
│  │ 前馈网络                    │    │
│  └─────────┬───────────────────┘    │
│            ↓                        │
│       残差 + LayerNorm               │
│            ↓                        │
│         输出                        │
└─────────────────────────────────────┘
```

### 8.1.3 输入输出流程

在**机器翻译**任务中，整体流程如下：

```
源语言句子（如英文）
        ↓
   [编码器] × N 层
        ↓
   编码器输出（memory）
        ↓ ←──────────── Key, Value
   [解码器] × N 层 ← 目标语言（如中文，右移一位）
        ↓
   线性层 + Softmax
        ↓
   目标语言词汇表概率分布
        ↓
   预测下一个词
```

解码器的输入是**目标序列右移一位**（shifted right）：

- 训练时输入：`<BOS> 我 爱 学 习`
- 训练时标签：`我 爱 学 习 <EOS>`

这样每个位置只需预测下一个词，而不是当前词本身。

---

## 8.2 掩码自注意力

### 8.2.1 为什么需要掩码

在标准自注意力中，每个位置都可以看到序列中的**所有**其他位置。这在编码器中完全合理——我们希望理解整个源语言句子的上下文。

然而，解码器在生成时必须遵循**因果性约束**（Causality Constraint）：

> **位置 $i$ 只能依赖于位置 $1, 2, \ldots, i$ 的信息，不能看到未来的词。**

举例说明：假设目标序列是"我 爱 学 习"

- 当预测"爱"时，只能看到"我"
- 当预测"学"时，只能看到"我 爱"
- 当预测"习"时，只能看到"我 爱 学"

如果允许看到未来，模型在训练时可以直接"抄答案"，推理时却无法看到未来词汇，导致训练和推理行为不一致。

### 8.2.2 因果掩码（Causal Mask）的应用

因果掩码是一个**上三角矩阵**，将注意力矩阵中位置 $i$ 对位置 $j > i$ 的分数设为 $-\infty$：

对于长度为 4 的序列，掩码矩阵如下：

$$M = \begin{pmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{pmatrix}$$

在计算注意力分数时：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

其中 $M$ 是上述掩码矩阵。经过 $\text{softmax}$ 后，$-\infty$ 对应的权重变为 0，从而屏蔽了未来信息。

```python
import torch

def create_causal_mask(seq_len):
    """
    创建因果掩码（上三角为 True，即需要被屏蔽的位置）
    返回: (seq_len, seq_len) 的布尔张量
    """
    # triu 返回上三角矩阵，diagonal=1 表示不包括主对角线
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask

# 示例：长度为 4 的因果掩码
mask = create_causal_mask(4)
print(mask)
# tensor([[False,  True,  True,  True],
#         [False, False,  True,  True],
#         [False, False, False,  True],
#         [False, False, False, False]])
```

### 8.2.3 训练时的并行计算

因果掩码的一个重要优点是：**训练时可以并行处理整个目标序列**。

在推理时，我们必须逐步生成（第 $t$ 步依赖前 $t-1$ 步的输出）。但在训练时，由于我们已知完整的目标序列，只需用掩码防止信息泄露，就可以在一次前向传播中同时计算所有位置的损失。

```
训练时（并行）：
输入:  [<BOS>, 我,  爱,  学]  ← 一次性输入，用掩码防止看到未来
标签:  [我,    爱,  学,  习]  ← 每个位置预测下一个词

推理时（顺序）：
步骤1: 输入 [<BOS>]        → 预测 "我"
步骤2: 输入 [<BOS>, 我]    → 预测 "爱"
步骤3: 输入 [<BOS>, 我, 爱] → 预测 "学"
...
```

这种设计使得训练效率大幅提升，是 Transformer 相比 RNN 的重要优势之一。

---

## 8.3 编码器-解码器注意力

### 8.3.1 交叉注意力的 Q、K、V 来源

编码器-解码器注意力（也称**交叉注意力**，Cross-Attention）是解码器中最独特的模块。它的 Q、K、V 来自不同的地方：

| 参数 | 来源 | 含义 |
|------|------|------|
| **Query（Q）** | 解码器当前层的输出 | "我现在需要什么信息？" |
| **Key（K）** | 编码器最终层的输出 | "源语言每个位置能提供什么？" |
| **Value（V）** | 编码器最终层的输出 | "源语言每个位置的实际内容" |

数学上：

$$\text{CrossAttention}(Q_{dec}, K_{enc}, V_{enc}) = \text{softmax}\left(\frac{Q_{dec} K_{enc}^T}{\sqrt{d_k}}\right) V_{enc}$$

其中：
- $Q_{dec} \in \mathbb{R}^{T_{tgt} \times d_{model}}$，$T_{tgt}$ 是目标序列长度
- $K_{enc}, V_{enc} \in \mathbb{R}^{T_{src} \times d_{model}}$，$T_{src}$ 是源序列长度

### 8.3.2 Query 来自解码器，Key/Value 来自编码器

这一设计的直觉非常清晰：

```
解码器当前状态（Query）:
"我已经生成了'我爱'，现在我想生成下一个词，
 我需要从源语言中找到什么信息？"

编码器输出（Key/Value）:
"源语言句子'I love machine learning'中，
 每个词的语义表示"

交叉注意力的结果：
解码器通过 Q 与 K 计算相似度，找到源语言中
最相关的位置，然后加权聚合 V 中的信息。
```

在翻译"机器"时，交叉注意力会对"machine"位置赋予较高权重；翻译"学习"时，会对"learning"赋予较高权重。这正是注意力机制的核心价值——动态对齐（Alignment）。

### 8.3.3 信息流分析

完整的信息流如下：

```
源序列 "I love machine learning"
           ↓
    [编码器 × N 层]
           ↓
    编码器输出 memory
    shape: (batch, src_len, d_model)
           ↓
    ┌──────────────┐
    │  K = W_K × memory   │
    │  V = W_V × memory   │
    └──────┬───────┘
           │
           ↓ (在每一个解码器层中)
    ┌──────────────────────────────┐
    │  解码器层内部状态              │
    │  Q = W_Q × decoder_hidden    │
    │                              │
    │  CrossAttn(Q, K, V)          │
    │  → 融合了源语言信息的解码器状态 │
    └──────────────────────────────┘
```

注意：**K 和 V 在所有解码步骤中保持不变**（都是编码器最终输出），只有 Q 随着解码的进行而变化。这意味着编码器只需运行一次，极大提升了推理效率。

---

## 8.4 自回归生成

### 8.4.1 训练模式 vs 推理模式

解码器在训练和推理时有本质区别：

**训练模式（Teacher Forcing）：**

```
输入: <BOS> 我  爱  机  器  学  习
       ↓    ↓   ↓   ↓   ↓   ↓   ↓
      [D]  [D] [D] [D] [D] [D] [D]   ← 并行计算所有位置
       ↓    ↓   ↓   ↓   ↓   ↓   ↓
      我   爱  机  器  学  习  <EOS>  ← 与标签比较，计算损失
```

**推理模式（自回归）：**

```
步骤 1: 输入 [<BOS>]                → 输出"我"
步骤 2: 输入 [<BOS>, 我]            → 输出"爱"
步骤 3: 输入 [<BOS>, 我, 爱]        → 输出"机"
步骤 4: 输入 [<BOS>, 我, 爱, 机]    → 输出"器"
步骤 5: 输入 [<BOS>, 我, 爱, 机, 器] → 输出"学"
...
直到输出 <EOS>
```

### 8.4.2 Teacher Forcing

Teacher Forcing 指训练时**始终使用真实的目标词作为下一步输入**，而非模型自己预测的词。

**优点：**
- 避免"错误累积"（一步预测错误导致后续全错）
- 训练稳定，收敛快
- 支持并行计算

**缺点：**
- **曝光偏差（Exposure Bias）**：训练时看到的是真实词，推理时看到的是自己预测的词，存在分布差异
- 可能导致模型推理时对错误更脆弱

一些改进方法（如 Scheduled Sampling）会在训练过程中逐渐用预测词替代真实词，以减轻曝光偏差。

### 8.4.3 推理时的逐步生成

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    """
    贪婪解码：每一步选择概率最高的词
    """
    # 1. 编码源序列（只需一次）
    memory = model.encode(src, src_mask)

    # 2. 初始化目标序列，以 <BOS> 开头
    tgt = torch.ones(1, 1).fill_(start_symbol).long()

    for i in range(max_len - 1):
        # 3. 解码当前目标序列
        out = model.decode(memory, src_mask, tgt)

        # 4. 取最后一个位置的输出，预测下一个词
        prob = model.generator(out[:, -1])
        next_word = prob.argmax(dim=-1).item()

        # 5. 将预测词追加到目标序列
        tgt = torch.cat([tgt, torch.ones(1, 1).long().fill_(next_word)], dim=1)

        # 6. 检查终止条件
        if next_word == end_symbol:
            break

    return tgt
```

### 8.4.4 终止条件

自回归生成有以下几种终止条件：

1. **生成 `<EOS>` 标记**：最常见的终止方式，模型学会了在适当位置输出结束符
2. **达到最大长度限制**：防止生成无限循环，通常设置为源序列长度的 1.5-2 倍
3. **所有序列都已结束**（批量解码时）：批次中所有序列都输出了 `<EOS>`

在生产系统中，通常同时设置多个条件，满足任意一个即停止。

---

## 8.5 完整解码器层实现

### 8.5.1 DecoderLayer 类实现

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """多头注意力（复用编码器中的实现）"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, V), attn_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        x, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(x)


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class DecoderLayer(nn.Module):
    """
    单个解码器层，包含三个子层：
    1. 掩码多头自注意力
    2. 编码器-解码器（交叉）注意力
    3. 前馈网络
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # 子层 1：掩码自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 子层 2：交叉注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # 子层 3：前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        参数：
            tgt: 目标序列，shape (batch, tgt_len, d_model)
            memory: 编码器输出，shape (batch, src_len, d_model)
            tgt_mask: 目标序列掩码（因果掩码），防止看到未来
            memory_mask: 编码器输出掩码（通常是 padding 掩码）
        """
        # --- 子层 1：掩码自注意力 ---
        # Q=K=V 都来自目标序列，但加了因果掩码
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_output))

        # --- 子层 2：交叉注意力 ---
        # Q 来自解码器，K 和 V 来自编码器
        cross_output = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(cross_output))

        # --- 子层 3：前馈网络 ---
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_output))

        return tgt
```

### 8.5.2 TransformerDecoder 类实现

```python
class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    """
    完整的 Transformer 解码器
    包含：词嵌入 + 位置编码 + N 个 DecoderLayer + 最终输出层
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, max_len=5000, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 解码器层堆叠
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最终线性投影层（d_model → vocab_size）
        self.output_projection = nn.Linear(d_model, vocab_size)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier 均匀初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_causal_mask(self, seq_len, device):
        """创建因果掩码"""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        参数：
            tgt: 目标序列 token id，shape (batch, tgt_len)
            memory: 编码器输出，shape (batch, src_len, d_model)
            tgt_mask: 目标掩码，None 时自动生成因果掩码
            memory_mask: 源序列 padding 掩码
        返回：
            logits，shape (batch, tgt_len, vocab_size)
        """
        tgt_len = tgt.size(1)

        # 自动生成因果掩码（如果未提供）
        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(tgt_len, tgt.device)

        # 嵌入 + 位置编码，缩放嵌入向量
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # 通过 N 个解码器层
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)

        # 输出投影到词表空间
        logits = self.output_projection(x)
        return logits
```

### 8.5.3 解码器的参数量分析

以标准 Transformer（`d_model=512, num_heads=8, d_ff=2048, num_layers=6, vocab_size=32000`）为例：

**单个 DecoderLayer 的参数量：**

| 模块 | 参数量 | 计算公式 |
|------|--------|----------|
| 掩码自注意力（W_Q, W_K, W_V, W_O） | 4 × 512 × 512 = 1,048,576 | $4 \times d_{model}^2$ |
| 交叉注意力（W_Q, W_K, W_V, W_O） | 4 × 512 × 512 = 1,048,576 | $4 \times d_{model}^2$ |
| 前馈网络（两层线性） | 512×2048 + 2048×512 = 2,097,152 | $2 \times d_{model} \times d_{ff}$ |
| LayerNorm × 3（γ, β） | 3 × 2 × 512 = 3,072 | $3 \times 2 \times d_{model}$ |
| **单层合计** | **~4,197,376** | |

**整个解码器的参数量：**

$$\text{总参数量} = N \times (4 \times 2 \times d_{model}^2 + 2 \times d_{model} \times d_{ff}) + V \times d_{model}$$

$$= 6 \times (4 \times 2 \times 512^2 + 2 \times 512 \times 2048) + 32000 \times 512$$

$$\approx 6 \times 4.2M + 16.4M \approx 41.6M$$

其中词嵌入层（$V \times d_{model}$）占据较大比例。对比编码器（无交叉注意力，约 25.2M），解码器因为多了一组交叉注意力，参数量约为编码器的 1.65 倍。

```python
# 验证参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

decoder = TransformerDecoder(
    vocab_size=32000,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6
)

total_params = count_parameters(decoder)
print(f"解码器总参数量: {total_params:,}")
print(f"约 {total_params / 1e6:.1f}M 参数")
```

---

## 本章小结

| 特性 | 编码器（Encoder） | 解码器（Decoder） |
|------|-----------------|-----------------|
| **子层数量** | 2（自注意力 + FFN） | 3（掩码自注意力 + 交叉注意力 + FFN） |
| **自注意力类型** | 双向（可看所有位置） | 单向/因果（只看当前及之前位置） |
| **是否有交叉注意力** | 无 | 有（Q 来自解码器，K/V 来自编码器） |
| **输入** | 源序列 | 目标序列（右移一位） |
| **训练方式** | 完整并行 | Teacher Forcing（并行训练） |
| **推理方式** | 一次性计算 | 自回归逐步生成 |
| **典型应用** | BERT、文本分类、NER | GPT、机器翻译（目标端）、文本生成 |
| **参数量（标准配置）** | ~25.2M | ~41.6M |

**核心要点回顾：**
1. 解码器比编码器多一个交叉注意力子层，用于从编码器获取源语言信息
2. 因果掩码确保解码器不会"偷看"未来的词，保持自回归特性
3. 交叉注意力中 Q 来自解码器，K/V 来自编码器，实现动态对齐
4. 训练时用 Teacher Forcing 并行计算，推理时自回归逐步生成
5. 两种模式之间的差异（曝光偏差）是需要关注的训练-推理不一致问题

---

## 代码实战：完整文本生成示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ─────────────────────────────────────────────
# 1. 辅助组件（与上文定义相同，此处完整列出）
# ─────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, V), attn_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        x, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────
# 2. 解码器层
# ─────────────────────────────────────────────

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 子层 1：掩码自注意力
        attn_out = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_out))
        # 子层 2：交叉注意力
        cross_out = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(cross_out))
        # 子层 3：前馈网络
        ff_out = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_out))
        return tgt


# ─────────────────────────────────────────────
# 3. 完整 Transformer 解码器
# ─────────────────────────────────────────────

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt_len = tgt.size(1)
        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(tgt_len, tgt.device)
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.output_projection(x)


# ─────────────────────────────────────────────
# 4. 贪婪解码函数
# ─────────────────────────────────────────────

def greedy_decode(decoder, memory, src_mask, max_len,
                  bos_token_id, eos_token_id, device):
    """
    贪婪解码：每步选择概率最高的词

    参数：
        decoder: TransformerDecoder 实例
        memory: 编码器输出，shape (1, src_len, d_model)
        src_mask: 源序列掩码
        max_len: 最大生成长度
        bos_token_id: 开始符 id
        eos_token_id: 结束符 id
        device: 运行设备

    返回：
        生成的 token id 列表（不含 <BOS>）
    """
    decoder.eval()

    # 初始化：以 <BOS> 开头
    generated = [bos_token_id]

    with torch.no_grad():
        for step in range(max_len):
            # 构建当前目标序列
            tgt = torch.tensor([generated], dtype=torch.long, device=device)

            # 前向传播（自动生成因果掩码）
            logits = decoder(tgt, memory, tgt_mask=None, memory_mask=src_mask)

            # 取最后一个位置的 logits，选择概率最大的词
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)
            next_token_id = next_token_logits.argmax(dim=-1).item()

            # 追加到生成序列
            generated.append(next_token_id)

            # 遇到 <EOS> 则停止
            if next_token_id == eos_token_id:
                break

    # 去掉开头的 <BOS>，返回生成的 token id
    return generated[1:]


# ─────────────────────────────────────────────
# 5. 简单示例：随机初始化，测试前向传播
# ─────────────────────────────────────────────

def demo():
    # 超参数
    vocab_size = 1000
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 2
    batch_size = 2
    src_len = 10
    tgt_len = 8

    # 特殊 token
    BOS_ID = 1
    EOS_ID = 2
    PAD_ID = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建解码器
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=0.1
    ).to(device)

    print(f"解码器参数量: {sum(p.numel() for p in decoder.parameters()):,}")

    # 模拟编码器输出（实际应由编码器生成）
    memory = torch.randn(batch_size, src_len, d_model).to(device)

    # 模拟目标序列（训练时输入，右移版本）
    tgt_input = torch.randint(3, vocab_size, (batch_size, tgt_len)).to(device)
    tgt_input[:, 0] = BOS_ID  # 第一个 token 是 <BOS>

    # ---- 训练模式：前向传播 ----
    decoder.train()
    logits = decoder(tgt_input, memory)
    print(f"\n训练模式前向传播:")
    print(f"  输入形状: {tgt_input.shape}")
    print(f"  编码器输出形状: {memory.shape}")
    print(f"  解码器输出形状 (logits): {logits.shape}")

    # 计算交叉熵损失（模拟）
    tgt_labels = torch.randint(3, vocab_size, (batch_size, tgt_len)).to(device)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        tgt_labels.view(-1),
        ignore_index=PAD_ID
    )
    print(f"  损失值: {loss.item():.4f}")

    # ---- 推理模式：贪婪解码 ----
    single_memory = memory[0:1]  # 取第一个样本
    print(f"\n推理模式（贪婪解码）:")

    generated_ids = greedy_decode(
        decoder=decoder,
        memory=single_memory,
        src_mask=None,
        max_len=20,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        device=device
    )
    print(f"  生成的 token ids: {generated_ids}")
    print(f"  生成长度: {len(generated_ids)}")


if __name__ == '__main__':
    demo()
```

**预期输出示例：**

```
使用设备: cpu
解码器参数量: 1,474,744

训练模式前向传播:
  输入形状: torch.Size([2, 8])
  编码器输出形状: torch.Size([2, 10, 128])
  解码器输出形状 (logits): torch.Size([2, 8, 1000])
  损失值: 6.9078

推理模式（贪婪解码）:
  生成的 token ids: [537, 241, 89, ...]
  生成长度: 20
```

---

## 练习题

### 基础题

**练习 8.1**（基础）

以下代码创建了一个 4×4 的矩阵，请问它代表什么？填写注释并解释每个元素的含义：

```python
import torch
mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()
print(mask)
```

输出为：
```
tensor([[False,  True,  True,  True],
        [False, False,  True,  True],
        [False, False, False,  True],
        [False, False, False, False]])
```

请回答：
1. `True` 的位置表示什么含义？
2. 在注意力计算中，如何使用这个掩码？
3. 如果将 `diagonal=1` 改为 `diagonal=0`，结果会有什么变化？

---

**练习 8.2**（基础）

分析以下 DecoderLayer 的 `forward` 方法，指出每个步骤中张量的形状变化：

```python
def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
    # tgt:    (batch=2, tgt_len=5,  d_model=512)
    # memory: (batch=2, src_len=10, d_model=512)

    attn_out = self.self_attn(tgt, tgt, tgt, tgt_mask)
    # 问题 1：attn_out 的形状是？

    tgt = self.norm1(tgt + self.dropout(attn_out))
    # 问题 2：此时 tgt 的形状是？

    cross_out = self.cross_attn(tgt, memory, memory, memory_mask)
    # 问题 3：cross_out 的形状是？
    # 注意：Q 来自 tgt（长度5），K/V 来自 memory（长度10）

    tgt = self.norm2(tgt + self.dropout(cross_out))
    tgt = self.norm3(tgt + self.dropout(self.feed_forward(tgt)))
    return tgt
    # 问题 4：最终返回的 tgt 形状是？
```

---

### 中级题

**练习 8.3**（中级）

实现一个带有**温度采样**（Temperature Sampling）的解码函数，替代贪婪解码。温度参数 $\tau$ 控制生成的随机性：

$$P(w_i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

- 当 $\tau \to 0$ 时，等效于贪婪解码
- 当 $\tau = 1$ 时，使用原始概率分布
- 当 $\tau > 1$ 时，分布更平滑，生成更多样

要求：
1. 函数签名为 `temperature_decode(decoder, memory, src_mask, max_len, bos_id, eos_id, device, temperature=1.0)`
2. 处理 `temperature=0` 的边界情况

---

**练习 8.4**（中级）

在当前的 `TransformerDecoder` 实现中，`create_causal_mask` 生成的掩码形状为 `(tgt_len, tgt_len)`。

在多头注意力中，掩码需要广播到 `(batch, num_heads, tgt_len, tgt_len)` 的形状。

请修改 `TransformerDecoder.forward` 方法，使掩码能正确广播，并验证不同批次大小下的正确性。同时，解释为什么 PyTorch 的 `masked_fill` 可以直接使用 2D 掩码而不报错。

---

### 提高题

**练习 8.5**（提高）

实现一个支持**KV 缓存**（Key-Value Cache）的解码器，用于加速推理。

**背景**：在标准自回归解码中，每一步都需要重新计算所有历史 token 的 K 和 V。KV 缓存通过缓存已计算的 K、V 矩阵，每步只需计算新增 token 的 K、V，从而将时间复杂度从 $O(T^2)$ 降低到 $O(T)$（T 为当前序列长度）。

要求：
1. 修改 `MultiHeadAttention` 的 `forward` 方法，支持传入 `cache` 参数
2. 修改 `DecoderLayer` 的 `forward` 方法，在自注意力中使用 KV 缓存
3. 实现带缓存的贪婪解码函数 `cached_greedy_decode`
4. 验证带缓存和不带缓存的输出完全一致
5. 对比两种方式在序列长度 100 时的推理时间

**提示**：KV 缓存只用于自注意力（目标序列内部），交叉注意力的 K、V 来自编码器，始终固定不变，无需缓存（其实编码器输出本身就是"缓存"）。

---

## 练习答案

### 练习 8.1 答案

```python
import torch
mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()
print(mask)
# tensor([[False,  True,  True,  True],
#         [False, False,  True,  True],
#         [False, False, False,  True],
#         [False, False, False, False]])
```

**解答：**

1. **`True` 的位置含义**：`True` 表示该位置需要被**屏蔽（mask）**，即禁止注意力流动。位于行 $i$、列 $j$ 的 `True` 表示位置 $i$ 不能关注位置 $j$（$j > i$，即未来位置）。例如，第 0 行的 `True` 在列 1、2、3，说明位置 0 不能看到位置 1、2、3（未来）。

2. **在注意力计算中的使用**：
```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
# mask 为 True 的位置填充 -inf，经 softmax 后变为 0
scores = scores.masked_fill(mask, float('-inf'))
attn_weights = torch.softmax(scores, dim=-1)
# 被屏蔽的位置权重为 0，不会影响输出
```

3. **`diagonal=0` 的变化**：
```python
mask_d0 = torch.triu(torch.ones(4, 4), diagonal=0).bool()
# tensor([[ True,  True,  True,  True],
#         [False,  True,  True,  True],
#         [False, False,  True,  True],
#         [False, False, False,  True]])
```
主对角线也被设为 `True`，意味着每个位置连**自身**都无法关注，这通常不是我们想要的（无法学习自身表示）。标准因果掩码使用 `diagonal=1`，允许每个位置关注自身。

---

### 练习 8.2 答案

```python
# tgt:    (batch=2, tgt_len=5,  d_model=512)
# memory: (batch=2, src_len=10, d_model=512)

attn_out = self.self_attn(tgt, tgt, tgt, tgt_mask)
# 答案 1：attn_out 的形状是 (2, 5, 512)
# 自注意力中 Q, K, V 都来自 tgt，输出形状与输入相同

tgt = self.norm1(tgt + self.dropout(attn_out))
# 答案 2：tgt 的形状是 (2, 5, 512)
# 残差连接后 LayerNorm，不改变形状

cross_out = self.cross_attn(tgt, memory, memory, memory_mask)
# 答案 3：cross_out 的形状是 (2, 5, 512)
# 关键点：输出形状由 Q（来自 tgt）决定，不是 K/V（来自 memory）
# Q 形状：(2, 5, 512) → 输出也是 (2, 5, 512)
# K/V 形状：(2, 10, 512) → 只影响注意力矩阵大小 (2, h, 5, 10)，不影响输出形状

tgt = self.norm2(tgt + self.dropout(cross_out))
tgt = self.norm3(tgt + self.dropout(self.feed_forward(tgt)))
return tgt
# 答案 4：最终返回的 tgt 形状是 (2, 5, 512)
# 整个 DecoderLayer 的输入输出形状完全相同
```

**关键规律**：注意力的输出形状由 **Query** 决定，与 Key/Value 的序列长度无关。这是为什么交叉注意力中源序列和目标序列长度可以不同，但输出形状始终等于目标序列形状。

---

### 练习 8.3 答案

```python
def temperature_decode(decoder, memory, src_mask, max_len,
                       bos_id, eos_id, device, temperature=1.0):
    """
    温度采样解码

    temperature → 0：趋近贪婪解码（选最高概率词）
    temperature = 1：标准采样（使用原始概率）
    temperature > 1：更随机，生成更多样化
    """
    decoder.eval()
    generated = [bos_id]

    with torch.no_grad():
        for _ in range(max_len):
            tgt = torch.tensor([generated], dtype=torch.long, device=device)
            logits = decoder(tgt, memory, tgt_mask=None, memory_mask=src_mask)
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)

            if temperature == 0 or temperature < 1e-6:
                # 边界情况：等效贪婪解码
                next_token_id = next_token_logits.argmax(dim=-1).item()
            else:
                # 温度缩放
                scaled_logits = next_token_logits / temperature
                # 转为概率分布
                probs = torch.softmax(scaled_logits, dim=-1)
                # 按概率采样
                next_token_id = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token_id)

            if next_token_id == eos_id:
                break

    return generated[1:]


# 测试：不同温度的效果
device = torch.device('cpu')
vocab_size, d_model = 100, 64
decoder = TransformerDecoder(vocab_size, d_model, 4, 256, 2).to(device)
memory = torch.randn(1, 5, d_model).to(device)

print("温度 0.0（贪婪）:")
for _ in range(3):
    ids = temperature_decode(decoder, memory, None, 10, 1, 2, device, temperature=0.0)
    print(f"  {ids}")  # 每次相同

print("\n温度 1.0（标准采样）:")
for _ in range(3):
    ids = temperature_decode(decoder, memory, None, 10, 1, 2, device, temperature=1.0)
    print(f"  {ids}")  # 每次不同

print("\n温度 2.0（高温，更随机）:")
for _ in range(3):
    ids = temperature_decode(decoder, memory, None, 10, 1, 2, device, temperature=2.0)
    print(f"  {ids}")  # 每次不同，且差异更大
```

---

### 练习 8.4 答案

**关于广播机制的解释：**

当掩码形状为 `(tgt_len, tgt_len)` 时，PyTorch 会自动将其广播到注意力分数的形状 `(batch, num_heads, tgt_len, tgt_len)`。这是因为 PyTorch 广播规则：**从右侧对齐维度，缺失的维度自动重复**。

```python
# scores 形状: (batch, num_heads, tgt_len, tgt_len)
# mask  形状: (tgt_len, tgt_len)
# 广播后 mask 等效于: (1, 1, tgt_len, tgt_len)
# 自动扩展到: (batch, num_heads, tgt_len, tgt_len)
scores = scores.masked_fill(mask, float('-inf'))  # 可以正常工作
```

**修改 forward 方法以明确广播：**

```python
def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
    batch_size = tgt.size(0)
    tgt_len = tgt.size(1)

    if tgt_mask is None:
        # 生成 2D 因果掩码
        tgt_mask = self.create_causal_mask(tgt_len, tgt.device)

    # 明确扩展到 4D：(1, 1, tgt_len, tgt_len)，让 PyTorch 广播到 batch 和 heads
    if tgt_mask.dim() == 2:
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
        # 现在形状为 (1, 1, tgt_len, tgt_len)，广播到 (batch, num_heads, tgt_len, tgt_len)

    x = self.embedding(tgt) * math.sqrt(self.d_model)
    x = self.pos_encoding(x)
    for layer in self.layers:
        x = layer(x, memory, tgt_mask, memory_mask)
    return self.output_projection(x)


# 验证不同批次大小
for batch_size in [1, 4, 16]:
    decoder = TransformerDecoder(vocab_size=100, d_model=64, num_heads=4,
                                 d_ff=256, num_layers=2)
    memory = torch.randn(batch_size, 10, 64)
    tgt = torch.randint(0, 100, (batch_size, 5))

    output = decoder(tgt, memory)
    assert output.shape == (batch_size, 5, 100), f"批次 {batch_size} 形状错误"
    print(f"批次大小 {batch_size:2d}: 输出形状 {output.shape} ✓")
```

---

### 练习 8.5 答案

```python
class MultiHeadAttentionWithCache(nn.Module):
    """支持 KV 缓存的多头注意力"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, cache=None):
        """
        cache: dict，包含 'k' 和 'v' 键，形状 (batch, num_heads, past_len, d_k)
        返回: (output, updated_cache)
        """
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 如果有缓存，将新的 K/V 拼接到历史 K/V 上
        if cache is not None:
            if 'k' in cache:
                K = torch.cat([cache['k'], K], dim=2)  # 在序列维度拼接
                V = torch.cat([cache['v'], V], dim=2)
            # 更新缓存
            new_cache = {'k': K, 'v': V}
        else:
            new_cache = None

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # 推理时 mask 形状为 (1, cur_pos+1)，需要处理
            scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        x = torch.matmul(attn_weights, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(x), new_cache


class DecoderLayerWithCache(nn.Module):
    """支持 KV 缓存的解码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithCache(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # 交叉注意力不需要 KV 缓存（K/V 来自编码器，始终固定）
        self.cross_attn = MultiHeadAttentionWithCache(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, self_attn_cache=None):
        # 掩码自注意力（使用缓存）
        attn_out, new_cache = self.self_attn(tgt, tgt, tgt, tgt_mask, cache=self_attn_cache)
        tgt = self.norm1(tgt + self.dropout(attn_out))

        # 交叉注意力（不需要缓存，K/V 固定）
        cross_out, _ = self.cross_attn(tgt, memory, memory, memory_mask, cache=None)
        tgt = self.norm2(tgt + self.dropout(cross_out))

        ff_out = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_out))

        return tgt, new_cache


def cached_greedy_decode(decoder_layers, embedding, pos_encoding, output_proj,
                          memory, max_len, bos_id, eos_id, device, d_model):
    """
    带 KV 缓存的贪婪解码
    每步只输入最新的一个 token，利用缓存避免重复计算
    """
    import time

    # 初始化每层的缓存
    caches = [None] * len(decoder_layers)

    generated = [bos_id]

    with torch.no_grad():
        for step in range(max_len):
            # 只输入最新的一个 token（推理加速的关键）
            cur_token = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)

            # 嵌入 + 位置编码（只对当前 token）
            x = embedding(cur_token) * math.sqrt(d_model)
            # 位置编码：使用当前步骤的位置
            pos = pos_encoding.pe[:, step:step+1]
            x = x + pos

            # 通过每个解码器层（使用缓存）
            new_caches = []
            for i, layer in enumerate(decoder_layers):
                x, new_cache = layer(x, memory, tgt_mask=None,
                                     memory_mask=None, self_attn_cache=caches[i])
                new_caches.append(new_cache)
            caches = new_caches

            # 预测下一个 token
            logits = output_proj(x[:, -1, :])
            next_token_id = logits.argmax(dim=-1).item()
            generated.append(next_token_id)

            if next_token_id == eos_id:
                break

    return generated[1:]


# 性能对比测试
import time

vocab_size, d_model = 5000, 256
num_heads, d_ff, num_layers = 4, 1024, 4
device = torch.device('cpu')

# 标准解码器
std_decoder = TransformerDecoder(vocab_size, d_model, num_heads, d_ff, num_layers).to(device)
memory = torch.randn(1, 20, d_model).to(device)

# 标准贪婪解码
start = time.time()
std_result = greedy_decode(std_decoder, memory, None, 100, 1, 2, device)
std_time = time.time() - start

print(f"标准贪婪解码: {std_time:.3f}s, 生成 {len(std_result)} 个 token")
# 注：完整 KV 缓存实现需要对 MultiHeadAttention 内部结构进行更多修改
# 上述框架展示了核心思路：缓存历史 K/V，每步只计算新增 token 的表示
```

**KV 缓存的核心原理总结：**

| 方面 | 无缓存 | 有 KV 缓存 |
|------|--------|-----------|
| 每步计算量 | $O(T^2 \cdot d)$ | $O(T \cdot d)$ |
| 内存占用 | $O(1)$（不保存中间结果） | $O(T \cdot d \cdot L)$（L 层，T 历史长度） |
| 适用场景 | 训练、短序列推理 | 长序列推理、实时生成 |
| 实现复杂度 | 简单 | 较复杂（需维护缓存状态） |

KV 缓存是现代大语言模型推理加速的核心技术，例如 vLLM、FlashAttention 等框架都实现了高效的 KV 缓存管理。
