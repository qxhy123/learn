# 第6章：掩码注意力（Masked Attention）

> **系列导航**：Part 1 基础数学 → Part 2 注意力机制（当前）→ Part 3 Transformer 架构

---

## 学习目标

完成本章后，你将能够：

1. **理解掩码的作用和类型**：掌握掩码在注意力机制中的核心功能及应用场景
2. **掌握 Padding 掩码的实现**：处理变长序列，屏蔽填充位置的注意力
3. **掌握因果掩码（Causal Mask）的实现**：实现自回归模型的核心机制，防止"看见未来"
4. **理解交叉注意力掩码**：在编码器-解码器架构中组合多种掩码
5. **能够在 PyTorch 中正确使用掩码**：掌握掩码的形状处理、广播机制和性能优化技巧

**前置知识**：第4章自注意力、第5章多头注意力、PyTorch 张量操作

---

## 6.1 掩码的必要性

---

### 6.1.1 为什么需要掩码

在实际应用中，注意力机制面临两个关键问题：

**问题 1：变长序列的批量处理**

在批量训练时，同一批次中的序列长度往往不同：

```
序列1: "我 爱 自然 语言 处理"       长度 = 5
序列2: "深度 学习"                 长度 = 2
序列3: "Transformer 模型 很 强大"  长度 = 4
```

为了能够批量计算，我们需要将所有序列 padding 到同一长度（如 5）。Padding 位置不包含有效信息，**必须被屏蔽**，否则会污染注意力分布。

**问题 2：自回归生成的时序约束**

在语言模型（如 GPT）和解码器中，生成第 $t$ 个词时，**只能看到位置 $1, 2, \ldots, t$ 的词**，不能看到未来的词（位置 $t+1, t+2, \ldots$）。

如果不加掩码，模型在训练时会"作弊"——直接看到正确答案，导致训练和推理的不一致。

### 6.1.2 掩码的作用机制

掩码通过在 softmax 之前将某些位置的注意力分数设置为 $-\infty$，使得 softmax 后这些位置的权重趋近于 0：

$$\text{scores}_{ij} = \frac{Q_i \cdot K_j^T}{\sqrt{d_k}}$$

$$\text{masked\_scores}_{ij} = \begin{cases}
\text{scores}_{ij} & \text{if allowed} \\
-\infty & \text{if masked}
\end{cases}$$

$$\text{weights}_{ij} = \frac{\exp(\text{masked\_scores}_{ij})}{\sum_k \exp(\text{masked\_scores}_{ik})} \approx \begin{cases}
\text{normalized} & \text{if allowed} \\
0 & \text{if masked}
\end{cases}$$

**关键性质**：$\lim_{x \to -\infty} \frac{e^x}{\sum_k e^{x_k}} = 0$，因此 $-\infty$ 位置的权重为 0，对加权求和没有贡献。

### 6.1.3 掩码在注意力计算中的位置

掩码在**缩放后、softmax 前**应用：

```
           QK^T              ÷ √d_k           +mask           softmax          @ V
(B,h,L,L) ────→ scores ────→ scaled ────→ masked_scores ────→ weights ────→ context
          注意力分数     缩放        屏蔽无效位置      归一化       加权求和
```

如果在 softmax **之后**应用掩码，会破坏概率分布的归一化性质（权重和不为 1）。

### 6.1.4 掩码的三种主要类型

| 掩码类型 | 英文名 | 作用 | 应用场景 |
|---------|--------|------|---------|
| **Padding 掩码** | Padding Mask | 屏蔽序列中的填充位置 | 所有 Transformer 模型 |
| **因果掩码** | Causal Mask / Look-ahead Mask | 屏蔽未来位置，只能看到当前及之前 | GPT、解码器自注意力 |
| **交叉注意力掩码** | Cross-Attention Mask | 编码器-解码器之间的掩码 | 机器翻译、图像描述生成 |

本章将逐一深入讲解这三种掩码的实现和使用。

---

## 6.2 Padding 掩码

---

### 6.2.1 变长序列的处理

在自然语言处理中，句子的长度各不相同。为了批量处理，我们使用特殊的 `<PAD>` token 将短序列填充到批次中的最大长度：

```python
# 示例：批量处理三个句子
sentences = [
    "我 爱 NLP",            # 长度 3
    "深度 学习 很 有趣",     # 长度 4
    "Transformer",         # 长度 1
]

# Padding 到最大长度 4
padded = [
    "我 爱 NLP <PAD>",
    "深度 学习 很 有趣",
    "Transformer <PAD> <PAD> <PAD>",
]
```

在注意力计算中，`<PAD>` 位置不应该：
1. **被其他位置关注**（Key 的角度）
2. **关注任何位置**（Query 的角度，虽然这通常影响较小）

### 6.2.2 Padding 掩码的计算

Padding 掩码通常从输入序列的有效长度或 token ID 生成：

**方法 1：从序列长度生成**

```python
import torch

def create_padding_mask_from_lengths(
    seq_lengths: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """
    从序列长度生成 padding 掩码

    参数:
        seq_lengths: (B,) 每个序列的有效长度
        max_len: 序列的最大长度

    返回:
        mask: (B, max_len) bool 张量，True 表示有效位置，False 表示 padding
    """
    batch_size = seq_lengths.size(0)
    # 生成位置索引: (1, max_len)
    positions = torch.arange(max_len).unsqueeze(0)  # (1, max_len)
    # 广播比较: (B, 1) vs (1, max_len) -> (B, max_len)
    mask = positions < seq_lengths.unsqueeze(1)
    return mask

# 示例
seq_lengths = torch.tensor([3, 4, 1])  # 三个序列的长度
max_len = 4
mask = create_padding_mask_from_lengths(seq_lengths, max_len)
print(mask)
# tensor([[ True,  True,  True, False],   # 前3个有效
#         [ True,  True,  True,  True],    # 全部有效
#         [ True, False, False, False]])   # 只有第1个有效
```

**方法 2：从 token ID 生成**

```python
def create_padding_mask_from_tokens(
    token_ids: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    从 token ID 生成 padding 掩码

    参数:
        token_ids: (B, L) token ID 序列
        pad_token_id: padding token 的 ID

    返回:
        mask: (B, L) bool 张量，True 表示有效位置
    """
    return token_ids != pad_token_id

# 示例
token_ids = torch.tensor([
    [5, 12, 8, 0],    # 最后一个是 PAD
    [3, 7, 9, 2],     # 无 PAD
    [11, 0, 0, 0],    # 后三个是 PAD
])
mask = create_padding_mask_from_tokens(token_ids, pad_token_id=0)
print(mask)
# tensor([[ True,  True,  True, False],
#         [ True,  True,  True,  True],
#         [ True, False, False, False]])
```

### 6.2.3 在注意力中应用 Padding 掩码

在多头注意力中，掩码需要广播到注意力分数矩阵的形状 `(B, h, L_q, L_k)`：

```python
def apply_padding_mask_in_attention(
    scores: torch.Tensor,        # (B, h, L_q, L_k)
    key_padding_mask: torch.Tensor,  # (B, L_k)
) -> torch.Tensor:
    """
    在注意力分数上应用 padding 掩码

    参数:
        scores: 注意力分数 (B, h, L_q, L_k)
        key_padding_mask: (B, L_k), True 表示有效位置

    返回:
        masked_scores: (B, h, L_q, L_k)
    """
    # 将掩码扩展为 (B, 1, 1, L_k)，可以广播到 (B, h, L_q, L_k)
    mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_k)

    # 将 False (padding) 位置填充为 -inf
    masked_scores = scores.masked_fill(~mask, float('-inf'))
    return masked_scores

# 示例
B, h, L_q, L_k = 2, 4, 3, 5
scores = torch.randn(B, h, L_q, L_k)
key_padding_mask = torch.tensor([
    [True, True, True, True, False],  # 最后一个位置是 padding
    [True, True, False, False, False], # 后三个位置是 padding
])

masked_scores = apply_padding_mask_in_attention(scores, key_padding_mask)

# 验证：padding 位置应为 -inf
print("原始分数的最后一列（第5个 key）:")
print(scores[0, 0, :, 4])  # 第1个样本，第1个头，所有 query，第5个 key

print("\n掩码后的最后一列:")
print(masked_scores[0, 0, :, 4])  # 应全为 -inf
```

### 6.2.4 完整示例：带 Padding 掩码的自注意力

```python
import torch.nn.functional as F
import math

def self_attention_with_padding_mask(
    x: torch.Tensor,              # (B, L, d_model)
    W_q: torch.nn.Linear,
    W_k: torch.nn.Linear,
    W_v: torch.nn.Linear,
    padding_mask: torch.Tensor,   # (B, L), True 表示有效位置
) -> torch.Tensor:
    """带 padding 掩码的自注意力"""
    B, L, d_model = x.shape

    # 线性投影
    Q = W_q(x)  # (B, L, d_k)
    K = W_k(x)  # (B, L, d_k)
    V = W_v(x)  # (B, L, d_v)

    d_k = Q.size(-1)

    # 计算注意力分数
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)  # (B, L, L)

    # 应用 padding 掩码
    # padding_mask: (B, L) -> (B, 1, L) 可广播到 (B, L, L)
    mask = padding_mask.unsqueeze(1)  # (B, 1, L)
    scores = scores.masked_fill(~mask, float('-inf'))

    # Softmax 和加权求和
    weights = F.softmax(scores, dim=-1)

    # 处理全为 -inf 的行（所有 key 都被 mask）
    weights = torch.nan_to_num(weights, nan=0.0)

    output = torch.bmm(weights, V)  # (B, L, d_v)

    return output, weights

# 测试
B, L, d_model = 2, 5, 8
x = torch.randn(B, L, d_model)
W_q = torch.nn.Linear(d_model, d_model, bias=False)
W_k = torch.nn.Linear(d_model, d_model, bias=False)
W_v = torch.nn.Linear(d_model, d_model, bias=False)

# 第1个序列长度3，第2个序列长度5（无 padding）
padding_mask = torch.tensor([
    [True, True, True, False, False],
    [True, True, True, True, True],
])

output, weights = self_attention_with_padding_mask(x, W_q, W_k, W_v, padding_mask)

print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")

# 检查第1个样本的注意力权重（后2列应全为0）
print("\n第1个样本的注意力权重（Key维度）:")
print(weights[0].detach().numpy().round(3))
# 最后两列（索引3,4）应全为0，因为它们是 padding
```

---

## 6.3 因果掩码（Causal Mask）

---

### 6.3.1 自回归生成的需求

在**自回归语言模型**（如 GPT）中，生成序列是逐步进行的：

```
输入:  <BOS> The  cat  sat  on   the
目标:       The  cat  sat  on   the  mat

时刻 t=1: 只看 <BOS>          → 预测 "The"
时刻 t=2: 只看 <BOS> The      → 预测 "cat"
时刻 t=3: 只看 <BOS> The cat  → 预测 "sat"
...
```

在训练时，我们有完整的句子，但必须模拟推理时的逐步生成过程。**因果掩码**确保位置 $t$ 只能关注位置 $\leq t$ 的信息，不能"看见未来"。

### 6.3.2 下三角掩码矩阵

因果掩码是一个**下三角矩阵**（包含对角线）：

$$\text{CausalMask}_{ij} = \begin{cases}
1 & \text{if } j \leq i \\
0 & \text{if } j > i
\end{cases}$$

对于长度 $L=4$ 的序列：

$$\text{CausalMask} = \begin{bmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1
\end{bmatrix}$$

- 行 $i$：位置 $i$ 作为 Query
- 列 $j$：位置 $j$ 作为 Key
- $\text{Mask}_{ij} = 1$：允许位置 $i$ 关注位置 $j$
- $\text{Mask}_{ij} = 0$：屏蔽，位置 $i$ 不能关注位置 $j$

**解读**：
- 第1行：位置0只能看到位置0（自己）
- 第2行：位置1可以看到位置0和1
- 第3行：位置2可以看到位置0、1、2
- 第4行：位置3可以看到位置0、1、2、3

### 6.3.3 PyTorch 实现

**方法 1：使用 `torch.tril`（推荐）**

```python
def create_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """
    生成因果掩码（下三角矩阵）

    参数:
        seq_len: 序列长度
        device: 设备（CPU/CUDA）

    返回:
        mask: (seq_len, seq_len) bool 张量，True 表示可以关注
    """
    # torch.tril 生成下三角矩阵（包含对角线）
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask

# 示例
mask = create_causal_mask(5)
print(mask.int())
# tensor([[1, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0],
#         [1, 1, 1, 0, 0],
#         [1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 1]])
```

**方法 2：使用 `torch.triu` 生成反向掩码**

```python
def create_causal_mask_v2(seq_len: int, device=None) -> torch.Tensor:
    """使用上三角函数生成因果掩码"""
    # torch.triu(diagonal=1) 生成上三角（不含对角线）
    # 这些位置是"未来"，应该被屏蔽
    future_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1
    )
    # 取反：未来位置为 False，允许位置为 True
    return ~future_mask

mask = create_causal_mask_v2(5)
print(mask.int())
# 输出与方法1相同
```

**方法 3：广播生成（更灵活，支持不同 query/key 长度）**

```python
def create_causal_mask_broadcast(L_q: int, L_k: int, device=None) -> torch.Tensor:
    """
    使用广播生成因果掩码（支持 cross-attention）

    参数:
        L_q: query 序列长度
        L_k: key 序列长度

    返回:
        mask: (L_q, L_k) bool 张量
    """
    # query 位置索引: (L_q, 1)
    q_indices = torch.arange(L_q, device=device).unsqueeze(1)
    # key 位置索引: (1, L_k)
    k_indices = torch.arange(L_k, device=device).unsqueeze(0)
    # 广播比较: (L_q, 1) >= (1, L_k) -> (L_q, L_k)
    mask = q_indices >= k_indices
    return mask

mask = create_causal_mask_broadcast(4, 6)
print(mask.int())
# tensor([[1, 0, 0, 0, 0, 0],  # query 0 只能看到 key 0
#         [1, 1, 0, 0, 0, 0],  # query 1 可以看到 key 0,1
#         [1, 1, 1, 0, 0, 0],  # query 2 可以看到 key 0,1,2
#         [1, 1, 1, 1, 0, 0]]) # query 3 可以看到 key 0,1,2,3
```

### 6.3.4 在注意力中应用因果掩码

```python
def scaled_dot_product_attention_causal(
    Q: torch.Tensor,  # (B, h, L, d_k)
    K: torch.Tensor,  # (B, h, L, d_k)
    V: torch.Tensor,  # (B, h, L, d_v)
) -> torch.Tensor:
    """带因果掩码的缩放点积注意力"""
    B, h, L, d_k = Q.shape

    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, h, L, L)

    # 生成因果掩码
    causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=Q.device))

    # 扩展掩码维度以匹配 scores: (L, L) -> (1, 1, L, L) -> (B, h, L, L)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # 应用掩码：将上三角（未来）位置设为 -inf
    scores = scores.masked_fill(~causal_mask, float('-inf'))

    # Softmax 和加权求和
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)

    return output, weights

# 测试
B, h, L, d_k = 1, 1, 5, 8
Q = torch.randn(B, h, L, d_k)
K = torch.randn(B, h, L, d_k)
V = torch.randn(B, h, L, d_k)

output, weights = scaled_dot_product_attention_causal(Q, K, V)

print("因果注意力权重矩阵:")
print(weights[0, 0].detach().numpy().round(3))
# 应该是下三角矩阵，上三角全为0
```

### 6.3.5 因果掩码的可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_causal_mask(seq_len: int = 8):
    """可视化因果掩码"""
    mask = create_causal_mask(seq_len).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：掩码矩阵
    axes[0].imshow(mask, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title('因果掩码矩阵\n(白色=允许关注, 蓝色=屏蔽)', fontsize=12)
    axes[0].set_xlabel('Key 位置', fontsize=10)
    axes[0].set_ylabel('Query 位置', fontsize=10)

    # 添加网格和数值
    for i in range(seq_len):
        for j in range(seq_len):
            text = axes[0].text(j, i, int(mask[i, j]),
                               ha='center', va='center',
                               color='white' if mask[i, j] else 'gray',
                               fontsize=10)

    # 右图：模拟的注意力权重
    torch.manual_seed(42)
    Q = torch.randn(1, 1, seq_len, 16)
    K = torch.randn(1, 1, seq_len, 16)
    V = torch.randn(1, 1, seq_len, 16)
    _, weights = scaled_dot_product_attention_causal(Q, K, V)

    weights_np = weights[0, 0].detach().numpy()
    im = axes[1].imshow(weights_np, cmap='viridis', aspect='auto')
    axes[1].set_title('应用因果掩码后的注意力权重\n(上三角全为0)', fontsize=12)
    axes[1].set_xlabel('Key 位置', fontsize=10)
    axes[1].set_ylabel('Query 位置', fontsize=10)
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig('causal_mask_visualization.png', dpi=150, bbox_inches='tight')
    print("可视化已保存至 causal_mask_visualization.png")
    plt.show()

visualize_causal_mask()
```

---

## 6.4 交叉注意力掩码

---

### 6.4.1 编码器-解码器注意力中的掩码

在 Transformer 的编码器-解码器架构中，交叉注意力（Cross-Attention）让解码器关注编码器的输出：

```
编码器输入:  "The cat sat on the mat"  (源语言，英文)
解码器输入:  "<BOS> 猫 坐 在 垫子 上"   (目标语言，中文)

交叉注意力:
  Query  来自解码器
  Key    来自编码器
  Value  来自编码器
```

交叉注意力需要同时处理：
1. **解码器端的因果掩码**（自注意力层）：不能看到未来的目标词
2. **源序列的 Padding 掩码**（交叉注意力层）：不关注编码器输出的 padding 位置

### 6.4.2 组合多种掩码

在一个完整的 Transformer 解码器层中，掩码的使用场景：

| 子层 | 掩码类型 | 形状 | 作用 |
|-----|---------|------|------|
| 自注意力 | 因果掩码 + Padding掩码 | (L_tgt, L_tgt) | 当前位置不能看未来 + 不关注目标序列的 padding |
| 交叉注意力 | Padding掩码 | (L_tgt, L_src) | 不关注源序列的 padding |
| 前馈网络 | 无掩码 | - | - |

**组合策略：逻辑与（AND）**

当需要同时应用多个掩码时，使用逻辑与操作：

```python
def combine_masks(
    causal_mask: torch.Tensor,     # (L, L)
    padding_mask: torch.Tensor,    # (B, L)
) -> torch.Tensor:
    """
    组合因果掩码和 padding 掩码

    返回:
        combined_mask: (B, L, L)
    """
    B, L = padding_mask.shape

    # 扩展因果掩码: (L, L) -> (1, L, L)
    causal = causal_mask.unsqueeze(0)  # (1, L, L)

    # 扩展 padding 掩码: (B, L) -> (B, 1, L)
    padding = padding_mask.unsqueeze(1)  # (B, 1, L)

    # 逻辑与: (1, L, L) & (B, 1, L) -> (B, L, L)
    combined = causal & padding

    return combined

# 示例
L = 5
B = 2
causal_mask = create_causal_mask(L)
padding_mask = torch.tensor([
    [True, True, True, False, False],  # 前3个有效
    [True, True, True, True, True],     # 全部有效
])

combined = combine_masks(causal_mask, padding_mask)

print("第1个样本的组合掩码:")
print(combined[0].int())
# tensor([[1, 0, 0, 0, 0],  # 位置0只能看自己（有效）
#         [1, 1, 0, 0, 0],  # 位置1可以看0,1（有效）
#         [1, 1, 1, 0, 0],  # 位置2可以看0,1,2（有效）
#         [0, 0, 0, 0, 0],  # 位置3是padding，不关注任何位置
#         [0, 0, 0, 0, 0]]) # 位置4是padding，不关注任何位置
```

### 6.4.3 交叉注意力掩码示例

```python
def cross_attention_with_mask(
    query: torch.Tensor,      # (B, L_tgt, d_model) 来自解码器
    key: torch.Tensor,        # (B, L_src, d_model) 来自编码器
    value: torch.Tensor,      # (B, L_src, d_model) 来自编码器
    W_q: torch.nn.Linear,
    W_k: torch.nn.Linear,
    W_v: torch.nn.Linear,
    src_padding_mask: torch.Tensor,  # (B, L_src)
) -> torch.Tensor:
    """
    带掩码的交叉注意力

    参数:
        src_padding_mask: 源序列的 padding 掩码，True 表示有效位置
    """
    B, L_tgt, _ = query.shape
    L_src = key.size(1)

    Q = W_q(query)  # (B, L_tgt, d_k)
    K = W_k(key)    # (B, L_src, d_k)
    V = W_v(value)  # (B, L_src, d_v)

    d_k = Q.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)  # (B, L_tgt, L_src)

    # 应用源序列的 padding 掩码
    # src_padding_mask: (B, L_src) -> (B, 1, L_src) -> 广播到 (B, L_tgt, L_src)
    mask = src_padding_mask.unsqueeze(1)  # (B, 1, L_src)
    scores = scores.masked_fill(~mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)

    output = torch.bmm(weights, V)  # (B, L_tgt, d_v)

    return output, weights

# 测试
B, L_tgt, L_src, d_model = 2, 4, 6, 8
query = torch.randn(B, L_tgt, d_model)    # 解码器状态
key = torch.randn(B, L_src, d_model)       # 编码器输出
value = key  # 通常 key 和 value 相同

W_q = torch.nn.Linear(d_model, d_model, bias=False)
W_k = torch.nn.Linear(d_model, d_model, bias=False)
W_v = torch.nn.Linear(d_model, d_model, bias=False)

# 源序列的 padding 掩码
src_padding_mask = torch.tensor([
    [True, True, True, True, False, False],  # 前4个有效
    [True, True, True, True, True, True],     # 全部有效
])

output, weights = cross_attention_with_mask(
    query, key, value, W_q, W_k, W_v, src_padding_mask
)

print(f"交叉注意力输出形状: {output.shape}")  # (2, 4, 8)
print(f"注意力权重形状: {weights.shape}")      # (2, 4, 6)

# 检查第1个样本的权重（最后2列应全为0）
print("\n第1个样本的注意力权重:")
print(weights[0].detach().numpy().round(3))
```

### 6.4.4 完整解码器层的掩码逻辑

```python
class TransformerDecoderLayerWithMask(torch.nn.Module):
    """简化的 Transformer 解码器层，展示掩码的使用"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        from torch.nn import MultiheadAttention

        self.self_attn = MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn = MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,              # (B, L_tgt, d_model)
        memory: torch.Tensor,           # (B, L_src, d_model) 来自编码器
        tgt_mask: torch.Tensor = None,  # (L_tgt, L_tgt) 因果掩码
        tgt_key_padding_mask: torch.Tensor = None,  # (B, L_tgt)
        memory_key_padding_mask: torch.Tensor = None,  # (B, L_src)
    ):
        """
        参数:
            tgt_mask: 自注意力的因果掩码（下三角）
            tgt_key_padding_mask: 目标序列的 padding 掩码
            memory_key_padding_mask: 源序列的 padding 掩码
        """
        # 1. 自注意力（带因果掩码 + padding 掩码）
        attn_output, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,                         # 因果掩码
            key_padding_mask=~tgt_key_padding_mask      # padding 掩码（注意取反）
            if tgt_key_padding_mask is not None else None
        )
        tgt = self.norm1(tgt + attn_output)

        # 2. 交叉注意力（只有源序列的 padding 掩码）
        attn_output, _ = self.cross_attn(
            tgt, memory, memory,
            key_padding_mask=~memory_key_padding_mask   # padding 掩码
            if memory_key_padding_mask is not None else None
        )
        tgt = self.norm2(tgt + attn_output)

        # 3. 前馈网络
        ffn_output = self.ffn(tgt)
        tgt = self.norm3(tgt + ffn_output)

        return tgt

# 测试
B, L_tgt, L_src, d_model = 2, 5, 7, 64
num_heads = 4

decoder_layer = TransformerDecoderLayerWithMask(d_model, num_heads)

tgt = torch.randn(B, L_tgt, d_model)
memory = torch.randn(B, L_src, d_model)

# 生成掩码
causal_mask = ~create_causal_mask(L_tgt)  # PyTorch 的 attn_mask 使用 True 表示屏蔽
tgt_padding = torch.tensor([
    [True, True, True, False, False],
    [True, True, True, True, True],
])
src_padding = torch.tensor([
    [True, True, True, True, True, False, False],
    [True, True, True, True, True, True, True],
])

output = decoder_layer(
    tgt, memory,
    tgt_mask=causal_mask,
    tgt_key_padding_mask=tgt_padding,
    memory_key_padding_mask=src_padding,
)

print(f"解码器输出形状: {output.shape}")  # (2, 5, 64)
```

---

## 6.5 掩码实现技巧

---

### 6.5.1 掩码的形状处理

掩码在不同层次需要不同的形状，理解广播机制至关重要：

| 掩码类型 | 原始形状 | 扩展后形状 | 目标形状（注意力分数）|
|---------|---------|-----------|------------------|
| Padding掩码 | (B, L) | (B, 1, 1, L) | (B, h, L_q, L_k) |
| 因果掩码 | (L, L) | (1, 1, L, L) | (B, h, L, L) |
| 组合掩码 | (B, L, L) | (B, 1, L, L) | (B, h, L, L) |

**广播原则**：

```python
# 示例：掩码广播
B, h, L = 2, 4, 5

# 原始掩码
padding_mask = torch.tensor([[True, True, False, False, False],
                              [True, True, True, True, True]])  # (B, L)
causal_mask = create_causal_mask(L)  # (L, L)

# 方式1：手动扩展维度
padding_expanded = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
causal_expanded = causal_mask.unsqueeze(0).unsqueeze(0)    # (1, 1, L, L)

# 方式2：使用 view 和 -1
padding_view = padding_mask.view(B, 1, 1, L)
causal_view = causal_mask.view(1, 1, L, L)

# 方式3：使用 None 索引（等价于 unsqueeze）
padding_none = padding_mask[:, None, None, :]  # (B, 1, 1, L)
causal_none = causal_mask[None, None, :, :]    # (1, 1, L, L)

# 验证可以广播到目标形状
scores = torch.randn(B, h, L, L)
masked_scores = scores.masked_fill(~padding_expanded, float('-inf'))
print(f"成功广播！形状: {masked_scores.shape}")  # (2, 4, 5, 5)
```

### 6.5.2 布尔掩码 vs 加性掩码

有两种主流的掩码实现方式：

**方式1：布尔掩码 + masked_fill（推荐）**

```python
# 布尔掩码：True 表示有效，False 表示屏蔽
mask = torch.tensor([[True, True, False],
                     [True, False, False]])  # (B, L)

scores = torch.randn(2, 3)
# 将 False 位置填充为 -inf
masked_scores = scores.masked_fill(~mask, float('-inf'))

weights = F.softmax(masked_scores, dim=-1)
print(weights)
# tensor([[0.4502, 0.5498, 0.0000],  # 第3个位置被屏蔽，权重为0
#         [1.0000, 0.0000, 0.0000]]) # 第2、3个位置被屏蔽
```

**方式2：加性掩码（注意力偏置）**

```python
# 加性掩码：0 表示无偏置，-inf 表示屏蔽
additive_mask = torch.tensor([[0., 0., float('-inf')],
                               [0., float('-inf'), float('-inf')]])

scores = torch.randn(2, 3)
masked_scores = scores + additive_mask  # 直接相加

weights = F.softmax(masked_scores, dim=-1)
print(weights)
# 结果与方式1相同
```

**对比**：

| 特性 | 布尔掩码 | 加性掩码 |
|------|---------|---------|
| 存储 | bool (1 byte) | float32 (4 bytes) |
| 内存 | 更省内存（4倍） | 更占内存 |
| 语义 | 清晰（True/False） | 稍复杂（需理解 -inf） |
| 灵活性 | 只能屏蔽/不屏蔽 | 可以有不同偏置值 |
| 推荐场景 | 常规掩码 | 需要相对位置偏置时 |

**推荐使用布尔掩码**，除非需要实现相对位置编码等高级功能。

### 6.5.3 性能优化

**优化1：避免不必要的掩码创建**

```python
# 不推荐：每次 forward 都创建掩码
def forward_slow(self, x):
    L = x.size(1)
    mask = torch.tril(torch.ones(L, L))  # 每次都创建
    # ...

# 推荐：缓存掩码，仅在序列长度变化时重新创建
class EfficientAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('causal_mask', None)
        self.max_seq_len_cached = 0

    def _get_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        if seq_len > self.max_seq_len_cached:
            # 创建并缓存更大的掩码
            self.causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
            )
            self.max_seq_len_cached = seq_len
        return self.causal_mask[:seq_len, :seq_len]

    def forward(self, x):
        L = x.size(1)
        mask = self._get_causal_mask(L, x.device)  # 复用缓存
        # ...
```

**优化2：使用 `torch.backends.cuda.sdp_kernel`（PyTorch 2.0+）**

```python
# PyTorch 2.0 引入的高效注意力实现，自动处理掩码
import torch.nn.functional as F

def efficient_attention_pytorch20(Q, K, V, mask=None):
    """
    使用 PyTorch 2.0 的优化实现
    内部使用 Flash Attention 等优化算法
    """
    # scaled_dot_product_attention 支持直接传入掩码
    output = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=mask,      # 加性掩码或布尔掩码均可
        is_causal=True,      # 自动应用因果掩码（可选）
    )
    return output

# 示例
B, h, L, d_k = 2, 4, 128, 64
Q = torch.randn(B, h, L, d_k, device='cuda')
K = torch.randn(B, h, L, d_k, device='cuda')
V = torch.randn(B, h, L, d_k, device='cuda')

# 自动应用因果掩码，无需手动创建
output = efficient_attention_pytorch20(Q, K, V)
```

**优化3：避免冗余的 `nan_to_num` 操作**

```python
# 仅在可能出现全 -inf 行时使用
def safe_softmax(scores: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)

    # 检查是否有 NaN（仅在确实可能出现全 mask 的行时）
    if mask is not None and (~mask).all(dim=-1).any():
        weights = torch.nan_to_num(weights, nan=0.0)

    return weights
```

### 6.5.4 Flash Attention 中的掩码

Flash Attention（Dao et al., 2022）是一种优化的注意力实现，可以大幅降低内存占用和计算时间。它对掩码的处理有特殊优化：

```python
# 使用 xformers 或 flash-attn 库
try:
    from flash_attn import flash_attn_func

    def flash_attention_with_mask(Q, K, V, causal=False):
        """
        使用 Flash Attention 计算注意力

        参数:
            Q, K, V: (B, L, num_heads, d_k) 注意：维度顺序不同！
            causal: 是否应用因果掩码
        """
        # Flash Attention 内部优化了因果掩码的计算
        output = flash_attn_func(Q, K, V, causal=causal)
        return output

except ImportError:
    print("flash-attn 未安装，使用标准实现")
    # 降级到标准实现
```

**Flash Attention 的优势**：
- 不显式存储 $(L \times L)$ 的注意力矩阵，内存从 $O(L^2)$ 降至 $O(L)$
- 对因果掩码有特殊优化，自动跳过上三角的计算
- 在长序列（$L > 1024$）上可获得 2-4x 的加速

---

## 本章小结

---

| 掩码类型 | 形状 | 作用 | 实现方式 | 应用场景 |
|---------|------|------|---------|---------|
| **Padding掩码** | (B, L) | 屏蔽填充位置 | `token_ids != pad_id` | 所有模型（处理变长序列） |
| **因果掩码** | (L, L) | 屏蔽未来位置 | `torch.tril(ones(L,L))` | GPT、解码器自注意力 |
| **交叉注意力掩码** | (B, L_tgt, L_src) | 屏蔽源序列padding | 源序列的 Padding掩码 | Encoder-Decoder 架构 |
| **组合掩码** | (B, L, L) | 多种掩码同时应用 | 逻辑与（&） | 解码器自注意力 |

**关键要点**：

1. **位置**：掩码在 softmax **之前**应用，将无效位置设为 $-\infty$
2. **形状**：理解广播机制，正确扩展掩码维度
3. **类型**：布尔掩码（推荐）vs 加性掩码
4. **性能**：缓存掩码、使用 PyTorch 2.0 优化、考虑 Flash Attention

---

## 代码实战

---

以下完整代码整合了本章所有内容，可直接运行：

```python
"""
第6章完整代码实战：掩码注意力机制
文件名：chapter6_masked_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


# ============================================================
# Part 1: 掩码生成函数
# ============================================================

def create_padding_mask(
    seq_lengths: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """从序列长度生成 padding 掩码"""
    batch_size = seq_lengths.size(0)
    positions = torch.arange(max_len, device=seq_lengths.device).unsqueeze(0)
    mask = positions < seq_lengths.unsqueeze(1)
    return mask  # (B, max_len), True 表示有效位置


def create_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """生成因果掩码（下三角矩阵）"""
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask  # (seq_len, seq_len), True 表示可以关注


def combine_masks(
    causal_mask: torch.Tensor,     # (L, L)
    padding_mask: torch.Tensor,    # (B, L)
) -> torch.Tensor:
    """组合因果掩码和 padding 掩码"""
    B, L = padding_mask.shape
    # 扩展维度
    causal = causal_mask.unsqueeze(0)     # (1, L, L)
    padding = padding_mask.unsqueeze(1)   # (B, 1, L)
    # 逻辑与
    combined = causal & padding           # (B, L, L)
    return combined


# ============================================================
# Part 2: 带掩码的缩放点积注意力
# ============================================================

def scaled_dot_product_attention_with_mask(
    Q: torch.Tensor,  # (B, h, L_q, d_k)
    K: torch.Tensor,  # (B, h, L_k, d_k)
    V: torch.Tensor,  # (B, h, L_v, d_v)
    mask: Optional[torch.Tensor] = None,  # 兼容多种形状
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    带掩码的缩放点积注意力

    参数:
        mask: 可以是以下形状之一（自动广播）
              - (L_q, L_k)        因果掩码
              - (B, L_k)          padding 掩码
              - (B, L_q, L_k)     组合掩码
              - (B, 1, L_q, L_k)  完整掩码
              True 表示有效位置，False 表示屏蔽
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, h, L_q, L_k)

    if mask is not None:
        # 自动处理掩码的维度扩展
        if mask.dim() == 2:
            # 因果掩码 (L_q, L_k) -> (1, 1, L_q, L_k)
            if mask.size(0) == mask.size(1):
                mask = mask.unsqueeze(0).unsqueeze(0)
            # Padding掩码 (B, L_k) -> (B, 1, 1, L_k)
            else:
                mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 3:
            # 组合掩码 (B, L_q, L_k) -> (B, 1, L_q, L_k)
            mask = mask.unsqueeze(1)

        # 应用掩码
        scores = scores.masked_fill(~mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)

    # 处理全被屏蔽的情况
    if mask is not None:
        weights = torch.nan_to_num(weights, nan=0.0)

    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p)

    output = torch.matmul(weights, V)  # (B, h, L_q, d_v)

    return output, weights


# ============================================================
# Part 3: 带掩码的多头注意力
# ============================================================

class MultiHeadAttentionWithMask(nn.Module):
    """支持完整掩码功能的多头注意力"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None

        # 缓存因果掩码
        self.register_buffer('causal_mask_cache', None)
        self.max_seq_len_cached = 0

    def _get_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """获取因果掩码（带缓存）"""
        if seq_len > self.max_seq_len_cached:
            self.causal_mask_cache = create_causal_mask(seq_len, device)
            self.max_seq_len_cached = seq_len
        return self.causal_mask_cache[:seq_len, :seq_len]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        参数:
            mask: 自定义掩码（padding 等）
            causal: 是否应用因果掩码
        """
        B, L_q, _ = query.shape

        # 线性投影并分头
        Q = self.W_q(query).view(B, L_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, key.size(1), self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, value.size(1), self.num_heads, self.d_k).transpose(1, 2)

        # 组合掩码
        final_mask = mask
        if causal:
            causal_mask = self._get_causal_mask(L_q, query.device)
            if final_mask is not None:
                # 组合因果掩码和自定义掩码
                if final_mask.dim() == 2:  # (B, L_k)
                    final_mask = combine_masks(causal_mask, final_mask)
                else:
                    final_mask = final_mask & causal_mask.unsqueeze(0)
            else:
                final_mask = causal_mask

        # 注意力计算
        context, attn = scaled_dot_product_attention_with_mask(
            Q, K, V, final_mask, dropout_p=self.dropout.p if self.training else 0.0
        )

        self.attn_weights = attn.detach()

        # 合并头并输出投影
        context = context.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        output = self.W_o(context)

        if return_attn:
            return output, self.attn_weights
        return output, None


# ============================================================
# Part 4: 演示和可视化
# ============================================================

def demo_padding_mask():
    """演示 Padding 掩码"""
    print("=" * 60)
    print("【演示1】Padding 掩码")
    print("=" * 60)

    B, L, d_model = 2, 5, 16
    num_heads = 2

    # 模拟变长序列
    seq_lengths = torch.tensor([3, 5])  # 第1个序列长度3，第2个长度5
    x = torch.randn(B, L, d_model)

    # 生成 padding 掩码
    padding_mask = create_padding_mask(seq_lengths, L)
    print(f"\nPadding 掩码 (B={B}, L={L}):")
    print(padding_mask.int())

    # 应用到注意力
    mha = MultiHeadAttentionWithMask(d_model, num_heads)
    output, attn = mha(x, x, x, mask=padding_mask, return_attn=True)

    print(f"\n输出形状: {output.shape}")
    print(f"注意力权重形状: {attn.shape}")

    # 检查第1个样本的注意力（最后2列应为0）
    print("\n第1个样本，第1个头的注意力权重:")
    print(attn[0, 0].detach().numpy().round(3))
    print("(最后两列应全为0，因为是 padding)")


def demo_causal_mask():
    """演示因果掩码"""
    print("\n" + "=" * 60)
    print("【演示2】因果掩码（自回归）")
    print("=" * 60)

    L, d_model = 6, 16
    num_heads = 2

    tokens = ["<BOS>", "The", "cat", "sat", "on", "mat"]
    x = torch.randn(1, L, d_model)

    # 生成因果掩码
    causal_mask = create_causal_mask(L)
    print(f"\n因果掩码 (L={L}):")
    print(causal_mask.int())

    # 应用到注意力
    mha = MultiHeadAttentionWithMask(d_model, num_heads)
    output, attn = mha(x, x, x, causal=True, return_attn=True)

    print(f"\n输出形状: {output.shape}")

    # 可视化
    attn_np = attn[0, 0].detach().numpy()
    print("\n注意力权重矩阵（第1个头）:")
    print("      ", "  ".join([f"{t:>5}" for t in tokens]))
    for i, t in enumerate(tokens):
        print(f"{t:>5} ", "  ".join([f"{attn_np[i,j]:.3f}" for j in range(L)]))
    print("(上三角应全为0)")


def demo_combined_mask():
    """演示组合掩码"""
    print("\n" + "=" * 60)
    print("【演示3】组合掩码（因果 + Padding）")
    print("=" * 60)

    B, L, d_model = 2, 5, 16
    num_heads = 2

    # 序列长度
    seq_lengths = torch.tensor([3, 5])
    padding_mask = create_padding_mask(seq_lengths, L)

    print(f"\nPadding 掩码:")
    print(padding_mask.int())

    # 因果掩码
    causal_mask = create_causal_mask(L)
    print(f"\n因果掩码:")
    print(causal_mask.int())

    # 组合
    combined = combine_masks(causal_mask, padding_mask)
    print(f"\n组合掩码（第1个样本）:")
    print(combined[0].int())
    print("(位置3和4是padding，所有行的这两列都应为0)")

    # 应用
    x = torch.randn(B, L, d_model)
    mha = MultiHeadAttentionWithMask(d_model, num_heads)
    output, attn = mha(x, x, x, mask=padding_mask, causal=True, return_attn=True)

    print(f"\n第1个样本的注意力权重（第1个头）:")
    print(attn[0, 0].detach().numpy().round(3))


def visualize_masks():
    """可视化三种掩码"""
    print("\n" + "=" * 60)
    print("【演示4】掩码类型对比可视化")
    print("=" * 60)

    L = 8
    B = 2
    seq_lengths = torch.tensor([5, 8])

    # 生成三种掩码
    padding_mask = create_padding_mask(seq_lengths, L)
    causal_mask = create_causal_mask(L)
    combined = combine_masks(causal_mask, padding_mask)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Padding 掩码
    ax = axes[0, 0]
    im = ax.imshow(padding_mask[0].unsqueeze(0).int(), cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_title('Padding 掩码（第1个样本）\n长度=5，后3个位置被屏蔽', fontsize=11)
    ax.set_ylabel('样本')
    ax.set_xlabel('序列位置')
    ax.set_yticks([0])
    ax.set_yticklabels(['样本1'])
    plt.colorbar(im, ax=ax)

    # 2. 因果掩码
    ax = axes[0, 1]
    im = ax.imshow(causal_mask.int(), cmap='Greens', vmin=0, vmax=1, aspect='auto')
    ax.set_title('因果掩码（下三角）\n上三角=未来位置，被屏蔽', fontsize=11)
    ax.set_ylabel('Query 位置')
    ax.set_xlabel('Key 位置')
    plt.colorbar(im, ax=ax)

    # 3. 组合掩码（第1个样本）
    ax = axes[1, 0]
    im = ax.imshow(combined[0].int(), cmap='Oranges', vmin=0, vmax=1, aspect='auto')
    ax.set_title('组合掩码（第1个样本）\n因果 AND Padding', fontsize=11)
    ax.set_ylabel('Query 位置')
    ax.set_xlabel('Key 位置')
    plt.colorbar(im, ax=ax)

    # 4. 实际注意力权重
    ax = axes[1, 1]
    torch.manual_seed(42)
    d_model, num_heads = 32, 1
    x = torch.randn(1, L, d_model)
    mha = MultiHeadAttentionWithMask(d_model, num_heads)

    # 只取第1个样本的 mask
    mask_sample = padding_mask[0:1]  # (1, L)
    _, attn = mha(x, x, x, mask=mask_sample, causal=True, return_attn=True)

    attn_np = attn[0, 0].detach().numpy()
    im = ax.imshow(attn_np, cmap='viridis', aspect='auto')
    ax.set_title('应用组合掩码后的注意力权重', fontsize=11)
    ax.set_ylabel('Query 位置')
    ax.set_xlabel('Key 位置')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('mask_comparison.png', dpi=150, bbox_inches='tight')
    print("可视化已保存至 mask_comparison.png")
    plt.show()


def demo_cross_attention_mask():
    """演示交叉注意力掩码"""
    print("\n" + "=" * 60)
    print("【演示5】交叉注意力掩码")
    print("=" * 60)

    B, L_tgt, L_src, d_model = 2, 4, 6, 32
    num_heads = 2

    # 目标序列（解码器）
    tgt = torch.randn(B, L_tgt, d_model)
    # 源序列（编码器输出）
    src = torch.randn(B, L_src, d_model)

    # 源序列的 padding 掩码
    src_lengths = torch.tensor([4, 6])
    src_padding_mask = create_padding_mask(src_lengths, L_src)

    print(f"源序列 padding 掩码 (B={B}, L_src={L_src}):")
    print(src_padding_mask.int())

    # 交叉注意力：query 来自 tgt，key/value 来自 src
    mha = MultiHeadAttentionWithMask(d_model, num_heads)
    output, attn = mha(tgt, src, src, mask=src_padding_mask, return_attn=True)

    print(f"\n交叉注意力输出形状: {output.shape}")  # (2, 4, 32)
    print(f"注意力权重形状: {attn.shape}")        # (2, 2, 4, 6)

    print(f"\n第1个样本的注意力权重（第1个头）:")
    print(attn[0, 0].detach().numpy().round(3))
    print("(最后2列应全为0，因为源序列的后2个位置是 padding)")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("第6章：掩码注意力 - 完整代码演示")
    print("=" * 60)

    demo_padding_mask()
    demo_causal_mask()
    demo_combined_mask()
    visualize_masks()
    demo_cross_attention_mask()

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
```

---

## 练习题

---

### 基础题

**练习 6.1（基础）**

给定一个批次，包含3个序列，长度分别为 4、6、5，padding 到最大长度6。

(a) 手动写出 padding 掩码矩阵（3×6）

(b) 如果这3个序列进行自注意力计算（不使用因果掩码），注意力分数矩阵的形状是多少？

(c) padding 掩码应该如何广播到注意力分数矩阵的形状？写出每一步的形状变化。

---

**练习 6.2（基础）**

考虑因果掩码矩阵（下三角，长度为5）：

```
[[1, 0, 0, 0, 0],
 [1, 1, 0, 0, 0],
 [1, 1, 1, 0, 0],
 [1, 1, 1, 1, 0],
 [1, 1, 1, 1, 1]]
```

(a) 为什么对角线元素必须是1（每个位置可以关注自己）？

(b) 如果将对角线也设为0（不能关注自己），会发生什么？

(c) 编写代码生成"只能看到前1个位置"的掩码（例如位置3只能看位置2，不能看位置0、1、3、4）。

---

### 中级题

**练习 6.3（中级）**

实现一个**滑动窗口注意力掩码**，每个位置只能关注前后各 $w$ 个位置（共 $2w+1$ 个位置）。

要求：
- 函数签名：`create_sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor`
- 返回形状 `(seq_len, seq_len)` 的 bool 掩码
- 边界处理：序列开头和结尾的位置窗口可能不完整

示例：`window_size=1, seq_len=5` 时（每个位置可以看前1个、自己、后1个）：

```
[[1, 1, 0, 0, 0],   # 位置0: 只能看0,1
 [1, 1, 1, 0, 0],   # 位置1: 可以看0,1,2
 [0, 1, 1, 1, 0],   # 位置2: 可以看1,2,3
 [0, 0, 1, 1, 1],   # 位置3: 可以看2,3,4
 [0, 0, 0, 1, 1]]   # 位置4: 只能看3,4
```

---

**练习 6.4（中级）**

在 Transformer 解码器中，自注意力层需要同时应用因果掩码和 padding 掩码。假设：
- 批次大小 $B = 2$
- 序列长度 $L = 5$
- 第1个序列有效长度3，第2个序列有效长度5

(a) 绘制第1个样本的组合掩码矩阵（5×5）

(b) 解释为什么 padding 位置的行（Query）可以全为0（即 padding 位置不关注任何位置）

(c) 如果 padding 位置的行不全为0，softmax 后会发生什么？写代码验证。

---

### 提高题

**练习 6.5（提高）**

**实现 ALiBi（Attention with Linear Biases）位置编码**

ALiBi（Press et al., 2022）不使用传统的位置编码，而是在注意力分数中加入与相对距离成正比的负偏置：

$$\text{scores}_{ij} = \frac{Q_i \cdot K_j^T}{\sqrt{d_k}} - m \cdot |i - j|$$

其中 $m$ 是每个头的专属斜率（slope），不同头的 $m$ 不同。

要求：

(a) 实现 `create_alibi_bias(num_heads: int, seq_len: int) -> torch.Tensor`，返回形状 `(num_heads, seq_len, seq_len)` 的偏置矩阵。

(b) 斜率 $m$ 的生成公式（原论文）：
$$m_k = 2^{-\frac{8k}{n}}, \quad k = 1, 2, \ldots, n$$
其中 $n$ 是头数。例如 $n=4$ 时，斜率为 $[2^{-2}, 2^{-4}, 2^{-6}, 2^{-8}] = [0.25, 0.0625, 0.0156, 0.0039]$

(c) 修改 `MultiHeadAttentionWithMask` 类，添加 `use_alibi=True` 参数，在注意力计算中加入 ALiBi 偏置。

(d) 可视化不同头的 ALiBi 偏置模式，解释为什么斜率越小的头关注范围越广。

---

## 练习答案

---

### 答案 6.1

**(a) Padding 掩码矩阵**

```python
# 序列长度: [4, 6, 5]，padding 到 6
padding_mask = torch.tensor([
    [True, True, True, True, False, False],    # 样本1: 前4个有效
    [True, True, True, True, True, True],       # 样本2: 全部有效
    [True, True, True, True, True, False],      # 样本3: 前5个有效
])
print(padding_mask.int())
```

**(b) 注意力分数矩阵的形状**

对于自注意力，Query 和 Key 都来自输入序列，因此：
- 单头：`(B, L, L) = (3, 6, 6)`
- 多头（$h$ 个头）：`(B, h, L, L) = (3, h, 6, 6)`

**(c) 广播过程**

```python
# 原始 padding 掩码: (B, L) = (3, 6)
# 目标形状（多头）: (B, h, L_q, L_k) = (3, h, 6, 6)

# 步骤1: 扩展 Key 维度
mask = padding_mask.unsqueeze(1)  # (3, 1, 6)

# 步骤2: 扩展 Query 维度（也可以在步骤1之前）
mask = mask.unsqueeze(2)  # (3, 1, 1, 6)

# 现在 mask 形状为 (3, 1, 1, 6)，可以广播到 (3, h, 6, 6)
# - 维度1: 1 -> h（广播到所有头）
# - 维度2: 1 -> 6（广播到所有 Query 位置）
# - 维度3: 6 保持不变（Key 位置）
```

---

### 答案 6.2

**(a) 为什么对角线必须是1？**

因为每个位置生成其表示时，需要包含自身的信息。如果屏蔽自己，该位置只能依赖历史信息，无法捕获当前 token 的语义。

在自回归生成中，位置 $t$ 的输入是前 $t-1$ 个 token 加上当前位置的嵌入，因此需要关注自己。

**(b) 对角线为0会怎样？**

```python
# 生成无对角线的因果掩码
L = 5
mask = torch.tril(torch.ones(L, L)) - torch.eye(L)
print(mask.int())
# [[0, 0, 0, 0, 0],  # 位置0不能看任何位置！
#  [1, 0, 0, 0, 0],  # 位置1只能看位置0
#  [1, 1, 0, 0, 0],
#  [1, 1, 1, 0, 0],
#  [1, 1, 1, 1, 0]]

# 问题：第一个位置没有任何可关注的位置，softmax 会产生 NaN
scores = torch.randn(1, L, L)
scores = scores.masked_fill(mask == 0, float('-inf'))
weights = F.softmax(scores, dim=-1)
print(weights[0, 0])  # tensor([nan, nan, nan, nan, nan])
```

**(c) "只能看前1个位置"的掩码**

```python
def create_prev_only_mask(seq_len: int) -> torch.Tensor:
    """每个位置只能看前1个位置（不包括自己）"""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    # 次对角线（diagonal=-1）为 True
    mask += torch.diag(torch.ones(seq_len - 1), diagonal=-1).bool()
    return mask

mask = create_prev_only_mask(5)
print(mask.int())
# tensor([[0, 0, 0, 0, 0],  # 位置0: 无法看任何位置（会出现 NaN）
#         [1, 0, 0, 0, 0],  # 位置1: 只看位置0
#         [0, 1, 0, 0, 0],  # 位置2: 只看位置1
#         [0, 0, 1, 0, 0],  # 位置3: 只看位置2
#         [0, 0, 0, 1, 0]]) # 位置4: 只看位置3

# 实际应用中需要特殊处理第一个位置，例如允许看自己：
mask[0, 0] = True
```

---

### 答案 6.3

```python
def create_sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """
    生成滑动窗口注意力掩码

    参数:
        seq_len: 序列长度
        window_size: 窗口半径（前后各 window_size 个位置）

    返回:
        mask: (seq_len, seq_len) bool 张量
    """
    # 方法1: 使用循环（直观但较慢）
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = True
    return mask


# 方法2: 向量化实现（推荐）
def create_sliding_window_mask_v2(seq_len: int, window_size: int) -> torch.Tensor:
    """向量化实现"""
    # 生成位置索引
    q_idx = torch.arange(seq_len).unsqueeze(1)  # (L, 1)
    k_idx = torch.arange(seq_len).unsqueeze(0)  # (1, L)

    # 计算相对距离
    dist = torch.abs(q_idx - k_idx)  # (L, L)

    # 在窗口内的位置为 True
    mask = dist <= window_size
    return mask


# 测试
mask = create_sliding_window_mask_v2(8, window_size=2)
print("滑动窗口掩码 (window_size=2):")
print(mask.int())

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 5))
plt.imshow(mask.numpy(), cmap='Blues', aspect='auto')
plt.title('滑动窗口注意力掩码 (window_size=2)')
plt.xlabel('Key 位置')
plt.ylabel('Query 位置')
plt.colorbar()
plt.savefig('sliding_window_mask.png', dpi=150, bbox_inches='tight')
plt.show()
```

**应用场景**：
- Longformer（Beltagy et al., 2020）使用滑动窗口注意力处理长文档（最长 4096 tokens）
- 窗口越小，计算复杂度越低（从 $O(L^2)$ 降至 $O(L \cdot w)$）

---

### 答案 6.4

**(a) 第1个样本的组合掩码矩阵**

```python
L = 5
causal = torch.tril(torch.ones(L, L))
padding = torch.tensor([True, True, True, False, False])

# 组合: causal (5,5) AND padding (5,)
combined = causal.bool() & padding.unsqueeze(0)
print(combined.int())
# tensor([[1, 0, 0, 0, 0],  # Query 0: 可以看Key 0
#         [1, 1, 0, 0, 0],  # Query 1: 可以看Key 0,1
#         [1, 1, 1, 0, 0],  # Query 2: 可以看Key 0,1,2
#         [0, 0, 0, 0, 0],  # Query 3: padding，所有Key都被屏蔽
#         [0, 0, 0, 0, 0]]) # Query 4: padding，所有Key都被屏蔽
```

**(b) 为什么 padding 位置的行可以全为0？**

Padding 位置不包含有效信息，其输出不会被后续层使用（会被掩码掉）。因此，padding 位置关注谁并不重要，将其全部屏蔽可以：
1. 避免计算浪费
2. 保持语义清晰（padding 不参与任何注意力）

**(c) 如果 padding 行不全为0？**

```python
# 模拟：padding 位置的行没有被完全屏蔽
L = 5
scores = torch.randn(1, L, L)

# 只屏蔽 Key 的 padding（列），不屏蔽 Query 的 padding（行）
key_mask = torch.tensor([[True, True, True, False, False]])  # (1, L)
key_mask = key_mask.unsqueeze(1)  # (1, 1, L)
scores_masked = scores.masked_fill(~key_mask, float('-inf'))

weights = F.softmax(scores_masked, dim=-1)
print("注意力权重:")
print(weights[0].detach().numpy().round(3))

# 观察：第4、5行（padding Query）仍然有非零权重（关注前3个有效Key）
# 但这些权重不会被使用，因为 padding 位置的输出会被丢弃

# 更严格的做法：同时屏蔽 Query 的 padding 行
# 方法：在损失计算或后续使用时，忽略 padding 位置的输出
```

---

### 答案 6.5

```python
import math

def create_alibi_bias(num_heads: int, seq_len: int, device=None) -> torch.Tensor:
    """
    生成 ALiBi 位置偏置

    参数:
        num_heads: 注意力头数
        seq_len: 序列长度

    返回:
        bias: (num_heads, seq_len, seq_len) 偏置矩阵
    """
    # (a) 计算每个头的斜率
    def get_slopes(n):
        """原论文的斜率计算"""
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            # 不是2的幂时的处理
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            # 填充剩余的斜率
            extra = n - closest_power_of_2
            slopes.extend(get_slopes_power_of_2(2 * closest_power_of_2)[:extra])
            return slopes

    slopes = torch.tensor(get_slopes(num_heads), device=device)  # (num_heads,)

    # (b) 计算相对距离矩阵
    q_pos = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
    k_pos = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
    relative_dist = torch.abs(q_pos - k_pos).float()  # (seq_len, seq_len)

    # (c) 应用斜率：bias = -slope * distance
    # slopes: (num_heads,) -> (num_heads, 1, 1)
    # relative_dist: (seq_len, seq_len) -> (1, seq_len, seq_len)
    bias = -slopes.view(num_heads, 1, 1) * relative_dist.unsqueeze(0)  # (num_heads, L, L)

    return bias


# (c) 修改多头注意力类
class MultiHeadAttentionALiBi(nn.Module):
    """支持 ALiBi 的多头注意力"""

    def __init__(self, d_model: int, num_heads: int, use_alibi: bool = True):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_alibi = use_alibi

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # 缓存 ALiBi 偏置
        self.register_buffer('alibi_bias', None)
        self.max_seq_len_cached = 0

    def _get_alibi_bias(self, seq_len: int, device) -> torch.Tensor:
        """获取 ALiBi 偏置（带缓存）"""
        if not self.use_alibi:
            return None

        if seq_len > self.max_seq_len_cached:
            self.alibi_bias = create_alibi_bias(self.num_heads, seq_len, device)
            self.max_seq_len_cached = seq_len

        return self.alibi_bias[:, :seq_len, :seq_len]

    def forward(self, query, key, value, mask=None):
        B, L_q, _ = query.shape

        Q = self.W_q(query).view(B, L_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, key.size(1), self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, value.size(1), self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 添加 ALiBi 偏置
        if self.use_alibi:
            alibi = self._get_alibi_bias(L_q, query.device)
            scores = scores + alibi.unsqueeze(0)  # (1, h, L, L) -> (B, h, L, L)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)
        context = context.transpose(1, 2).contiguous().view(B, L_q, self.d_model)

        return self.W_o(context), weights


# (d) 可视化 ALiBi 偏置
def visualize_alibi():
    num_heads = 4
    seq_len = 16

    bias = create_alibi_bias(num_heads, seq_len)

    fig, axes = plt.subplots(1, num_heads, figsize=(16, 4))

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(bias[h].numpy(), cmap='RdBu_r', aspect='auto')
        ax.set_title(f'Head {h+1}\n斜率={2**(-2**(h+1)):.4f}')
        ax.set_xlabel('Key 位置')
        ax.set_ylabel('Query 位置')
        plt.colorbar(im, ax=ax)

    plt.suptitle('ALiBi 位置偏置（不同头的斜率）', fontsize=14)
    plt.tight_layout()
    plt.savefig('alibi_bias.png', dpi=150, bbox_inches='tight')
    print("ALiBi 可视化已保存至 alibi_bias.png")
    plt.show()

visualize_alibi()

# 测试
mha_alibi = MultiHeadAttentionALiBi(d_model=64, num_heads=4, use_alibi=True)
x = torch.randn(2, 10, 64)
output, _ = mha_alibi(x, x, x)
print(f"输出形状: {output.shape}")  # (2, 10, 64)
```

**ALiBi 的优势**：
1. **外推能力**：训练时用短序列（如512），推理时可以直接处理更长序列（如2048），注意力模式仍然合理
2. **无需显式位置编码**：简化模型，减少参数
3. **性能相当或更好**：在多个任务上与传统位置编码性能相当

---

> **下一章预告**：第7章将介绍 **Transformer 编码器架构**，整合多头注意力、前馈网络、层归一化和残差连接，构建完整的编码器模块。

---

[返回目录](../README.md)
