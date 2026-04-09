# 第5章：多头注意力（Multi-Head Attention）

> **系列导航**：Part 1 基础数学 → Part 2 注意力机制（当前）→ Part 3 Transformer 架构

---

## 学习目标

完成本章后，你将能够：

1. 理解多头注意力相比单头注意力的动机和优势
2. 掌握多头注意力的完整计算过程及公式推导
3. 理解头的分割与合并操作，掌握张量变形技巧
4. 从零实现高效的多头注意力机制（两种方法）
5. 了解不同注意力头学习到的不同语言模式

**前置知识**：第4章缩放点积注意力、PyTorch 基础张量操作

---

## 5.1 为什么需要多头注意力

---

### 5.1.1 单头注意力的局限

在第4章中，我们学习了缩放点积注意力：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

这个机制已经很强大——对于每个位置，它能够根据相关性加权聚合所有位置的信息。但它有一个根本性的局限：**每次只能关注一种关系模式**。

考虑这个句子：

> "The animal didn't cross the street because **it** was too tired."

理解代词 "it" 指代的是 "animal" 而非 "street"，需要同时理解：
- **句法关系**：it 是句子的主语
- **语义关系**：tired 通常描述动物而非街道
- **位置信息**：it 与 animal 的距离比与 street 更近

单头注意力在一次计算中只能形成一个注意力分布。即使模型可以学习到某种混合权重，**不同类型的关系会相互竞争**，最终的注意力分布是多种信号的妥协结果。

从数学角度看，单头注意力将 $Q, K, V$ 投影到同一个 $d_k$ 维子空间，然后计算一个加权平均。这个单一的线性投影限制了模型同时表达多种关系的能力。

### 5.1.2 多头机制：并行关注不同子空间

多头注意力的核心思想非常直观：**用 $h$ 个不同的注意力头，每个头负责关注不同的语言关系**。

每个头使用独立的投影矩阵 $W_i^Q, W_i^K, W_i^V$，将原始的 $d_{model}$ 维表示投影到较小的 $d_k$ 维子空间，然后在这个子空间中计算注意力。

不同的头可以自由地专门化：
- 头1 学习关注近距离的语法依赖
- 头2 学习关注语义相似性
- 头3 学习关注句子结构（主谓宾关系）
- 头4 学习关注指代关系
- ……

最终，所有头的输出被拼接起来，通过一个输出投影矩阵 $W^O$ 融合为统一的表示。

### 5.1.3 类比：CNN 的多通道机制

多头注意力与卷积神经网络中的多通道机制有着相似的哲学：

| 机制 | CNN 多通道 | 多头注意力 |
|------|-----------|-----------|
| 基本单元 | 单个卷积核 | 单个注意力头 |
| 并行化 | $C_{out}$ 个卷积核 | $h$ 个注意力头 |
| 子空间 | 特征图通道 | 低维注意力子空间 |
| 融合 | 通道拼接 + $1\times1$ 卷积 | 头拼接 + 线性投影 |
| 学到的内容 | 不同的局部特征（边缘、纹理等） | 不同的关系模式（语法、语义等） |

CNN 第一层的不同卷积核会分别检测水平边缘、垂直边缘、颜色梯度等不同特征。类似地，多头注意力中不同的头会学习捕获不同类型的语言结构。

### 5.1.4 维度设计的精妙之处

原始 Transformer 论文（Vaswani et al., 2017）的设计非常精巧：

- 模型维度：$d_{model} = 512$
- 头数：$h = 8$
- 每头维度：$d_k = d_v = d_{model} / h = 64$

这样设计的好处是：**多头注意力的总计算量与单头注意力大致相同**。

单头注意力（使用全维度 $d_{model} = 512$）的计算量 ≈ 8个头并行（每个头维度 $d_k = 64$）的计算量。

多头注意力用**并行多样性**换取了**单头的高维度**，在保持计算效率的同时大幅提升了表达能力。

---

## 5.2 多头注意力的计算

---

### 5.2.1 核心公式

多头注意力的完整公式如下：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, head_2, \ldots, head_h) \cdot W^O$$

其中每个头的计算为：

$$head_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)$$

展开注意力函数：

$$head_i = \text{softmax}\left(\frac{(QW_i^Q)(KW_i^K)^T}{\sqrt{d_k}}\right)(VW_i^V)$$

**各参数矩阵的维度**：

| 参数 | 形状 | 说明 |
|------|------|------|
| $W_i^Q$ | $d_{model} \times d_k$ | 第 $i$ 头的查询投影 |
| $W_i^K$ | $d_{model} \times d_k$ | 第 $i$ 头的键投影 |
| $W_i^V$ | $d_{model} \times d_v$ | 第 $i$ 头的值投影 |
| $W^O$ | $hd_v \times d_{model}$ | 输出投影 |

通常令 $d_k = d_v = d_{model} / h$。

### 5.2.2 计算步骤详解

以一个具体例子追踪完整计算过程。设：
- 批次大小 $B = 1$，序列长度 $L = 4$（如 "我 爱 自 然"）
- $d_{model} = 8$，$h = 2$，$d_k = d_v = 4$

**步骤1：输入线性投影**

输入 $X \in \mathbb{R}^{B \times L \times d_{model}}$，即形状 $(1, 4, 8)$

对每个头 $i$ 分别投影：
$$Q_i = X W_i^Q \in \mathbb{R}^{B \times L \times d_k}$$
$$K_i = X W_i^K \in \mathbb{R}^{B \times L \times d_k}$$
$$V_i = X W_i^V \in \mathbb{R}^{B \times L \times d_v}$$

**步骤2：各头独立计算注意力**

$$A_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times L \times L}$$
$$head_i = A_i V_i \in \mathbb{R}^{B \times L \times d_v}$$

**步骤3：拼接所有头的输出**

$$\text{Concat}(head_1, head_2) \in \mathbb{R}^{B \times L \times (h \cdot d_v)} = \mathbb{R}^{(1, 4, 8)}$$

**步骤4：输出线性投影**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\ldots) W^O \in \mathbb{R}^{B \times L \times d_{model}}$$

输出形状 $(1, 4, 8)$，与输入形状相同。这对于堆叠多层至关重要。

### 5.2.3 参数量分析

多头注意力的参数量：

**每个头的投影矩阵**（$h$ 个头，每头三个矩阵）：
$$h \times (d_{model} \times d_k + d_{model} \times d_k + d_{model} \times d_v)$$

令 $d_k = d_v = d_{model}/h$：
$$= h \times 3 \times d_{model} \times \frac{d_{model}}{h} = 3 \times d_{model}^2$$

**输出投影矩阵**：
$$W^O: (h \times d_v) \times d_{model} = d_{model} \times d_{model} = d_{model}^2$$

**总参数量**：
$$\text{总参数} = 3 d_{model}^2 + d_{model}^2 = 4 d_{model}^2$$

对于 $d_{model} = 512$：总参数 $= 4 \times 512^2 = 1,048,576 \approx 100$万个参数

注意这与 $h$ 无关——**增加头数不会增加参数量**（在保持 $d_k = d_{model}/h$ 的条件下）。

### 5.2.4 与单头注意力的参数量对比

| 方案 | 参数量 |
|------|--------|
| 单头注意力（$d_k = d_{model}$）| $3 d_{model}^2 + d_{model}^2 = 4d_{model}^2$ |
| 多头注意力（$h$ 头，$d_k = d_{model}/h$）| $3 d_{model}^2 + d_{model}^2 = 4d_{model}^2$ |

两者参数量相同！多头注意力用相同的参数量获得了更强的表达能力。

---

## 5.3 头的分割与合并

---

### 5.3.1 高效实现的关键思路

朴素实现（循环遍历每个头）虽然直观，但效率很低。高效实现的关键是：**将所有头的计算合并为一次批量矩阵乘法**。

核心技巧是将头的维度视为批次维度的一部分，使得 PyTorch 的并行矩阵乘法可以同时处理所有头。

### 5.3.2 投影后的张量分割

假设我们有一个统一的大投影矩阵（将所有头的投影矩阵堆叠在一起）：

$$W^Q \in \mathbb{R}^{d_{model} \times (h \cdot d_k)} = \mathbb{R}^{d_{model} \times d_{model}}$$

投影后得到：
$$Q_{all} = X W^Q \in \mathbb{R}^{B \times L \times d_{model}}$$

现在需要将最后一维重新解释为 $h$ 个头，每头 $d_k$ 维：

**分割过程（张量 reshape）**：

$$\mathbb{R}^{B \times L \times d_{model}} \xrightarrow{\text{view}} \mathbb{R}^{B \times L \times h \times d_k} \xrightarrow{\text{transpose}} \mathbb{R}^{B \times h \times L \times d_k}$$

在代码中：
```python
# Q shape: (B, L, d_model)
Q = Q.view(B, L, h, d_k)      # (B, L, h, d_k)
Q = Q.transpose(1, 2)          # (B, h, L, d_k)
```

经过 transpose 后，形状变为 $(B, h, L, d_k)$。现在 $h$ 维在批次维度后面，可以视为扩展的批次维度，使得注意力计算能够在所有头上并行进行。

### 5.3.3 注意力计算中的张量形状追踪

```
输入 X:          (B, L, d_model)
                      ↓ 线性投影 W^Q, W^K, W^V
Q, K, V:         (B, L, d_model)
                      ↓ view(B, L, h, d_k)
Q, K, V:         (B, L, h, d_k)
                      ↓ transpose(1, 2)
Q, K, V:         (B, h, L, d_k)
                      ↓ Q @ K.transpose(-2, -1) / sqrt(d_k)
注意力分数:       (B, h, L, L)
                      ↓ softmax(dim=-1)
注意力权重:       (B, h, L, L)
                      ↓ @ V
每头输出:        (B, h, L, d_k)
                      ↓ transpose(1, 2)
每头输出:        (B, L, h, d_k)
                      ↓ contiguous().view(B, L, d_model)
拼接输出:        (B, L, d_model)
                      ↓ 线性投影 W^O
最终输出:        (B, L, d_model)
```

### 5.3.4 合并操作的细节

合并时需要注意 `contiguous()` 的使用：

```python
# x shape: (B, h, L, d_k)
x = x.transpose(1, 2)          # (B, L, h, d_k)
# 注意：transpose 之后内存不连续，需要 contiguous()
x = x.contiguous()              # 确保内存连续
x = x.view(B, L, h * d_k)      # (B, L, d_model)
```

或者使用 `reshape`（自动处理连续性问题，但可能复制数据）：

```python
x = x.transpose(1, 2).reshape(B, L, h * d_k)  # (B, L, d_model)
```

**为什么需要 `contiguous()`？**

`transpose` 操作不复制数据，只修改张量的步长（stride）信息，使得内存布局变成非连续的。`view` 要求内存连续，否则会报错。`contiguous()` 会在需要时复制数据使内存连续。

### 5.3.5 批量矩阵乘法

当 Q、K、V 形状为 $(B, h, L, d_k)$ 时，`@` 运算符会自动进行批量矩阵乘法：

```python
# Q: (B, h, L, d_k)
# K: (B, h, L, d_k)
# K.transpose(-2, -1): (B, h, d_k, L)
scores = Q @ K.transpose(-2, -1)  # (B, h, L, L)
```

PyTorch 将前两个维度视为批次维度，对最后两个维度执行矩阵乘法。这相当于对每个样本的每个头分别计算注意力，完全并行。

---

## 5.4 多头注意力的实现

---

### 5.4.1 方法1：循环遍历每个头（直观版）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttentionNaive(nn.Module):
    """
    直观实现：显式循环遍历每个头
    优点：代码清晰，易于理解
    缺点：无法充分利用GPU并行性，速度较慢
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 为每个头创建独立的投影矩阵
        self.W_q = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False)
                                   for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False)
                                   for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False)
                                   for _ in range(num_heads)])

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """单头缩放点积注意力"""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, V), attn_weights

    def forward(self, query, key, value, mask=None):
        """
        参数：
            query: (B, L_q, d_model)
            key:   (B, L_k, d_model)
            value: (B, L_v, d_model)  L_k == L_v
            mask:  (B, L_q, L_k) 或 None

        返回：
            output: (B, L_q, d_model)
            attn_weights: list of (B, L_q, L_k), 每个头一个
        """
        head_outputs = []
        head_weights = []

        # 循环计算每个头
        for i in range(self.num_heads):
            Q_i = self.W_q[i](query)   # (B, L_q, d_k)
            K_i = self.W_k[i](key)     # (B, L_k, d_k)
            V_i = self.W_v[i](value)   # (B, L_v, d_k)

            head_i, weights_i = self.scaled_dot_product_attention(
                Q_i, K_i, V_i, mask
            )
            head_outputs.append(head_i)    # (B, L_q, d_k)
            head_weights.append(weights_i) # (B, L_q, L_k)

        # 拼接所有头的输出
        concat = torch.cat(head_outputs, dim=-1)  # (B, L_q, d_model)

        # 输出投影
        output = self.W_o(concat)  # (B, L_q, d_model)

        return output, head_weights
```

### 5.4.2 方法2：批量矩阵乘法（推荐实现）

```python
class MultiHeadAttention(nn.Module):
    """
    高效实现：使用批量矩阵乘法并行计算所有头
    优点：充分利用GPU并行性，速度快
    缺点：张量操作略为复杂

    这是工业界标准实现方式。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 使用单个大矩阵代替多个小矩阵，效率更高
        # W_q 的形状是 (d_model, d_model)，相当于 h 个 (d_model, d_k) 矩阵的拼接
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        # 用于存储注意力权重（可视化用）
        self.attn_weights = None

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将最后一维分割为多个头

        参数：x: (B, L, d_model)
        返回：  (B, num_heads, L, d_k)
        """
        B, L, _ = x.shape
        # 重塑为 (B, L, num_heads, d_k)，然后转置
        x = x.view(B, L, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (B, num_heads, L, d_k)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多个头的输出合并

        参数：x: (B, num_heads, L, d_k)
        返回：  (B, L, d_model)
        """
        B, _, L, _ = x.shape
        # 转置后内存不连续，需要 contiguous()
        x = x.transpose(1, 2).contiguous()  # (B, L, num_heads, d_k)
        return x.view(B, L, self.d_model)    # (B, L, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        return_attn: bool = False,
    ):
        """
        参数：
            query: (B, L_q, d_model)
            key:   (B, L_k, d_model)
            value: (B, L_v, d_model)
            mask:  广播兼容的掩码张量，0表示需要屏蔽的位置
            return_attn: 是否返回注意力权重

        返回：
            output: (B, L_q, d_model)
            attn_weights（可选）: (B, num_heads, L_q, L_k)
        """
        B, L_q, _ = query.shape

        # 1. 线性投影：(B, L, d_model) -> (B, L, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. 分割头：(B, L, d_model) -> (B, h, L, d_k)
        Q = self.split_heads(Q)  # (B, h, L_q, d_k)
        K = self.split_heads(K)  # (B, h, L_k, d_k)
        V = self.split_heads(V)  # (B, h, L_v, d_k)

        # 3. 缩放点积注意力（批量，所有头并行）
        # Q @ K^T: (B, h, L_q, d_k) @ (B, h, d_k, L_k) = (B, h, L_q, L_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 4. 应用掩码
        if mask is not None:
            # mask: (B, 1, L_q, L_k) 或 (B, 1, 1, L_k) 等，通过广播应用到所有头
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, L_q, L_k) -> 广播到所有头
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5. Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)  # (B, h, L_q, L_k)

        # 处理全被遮蔽的情况（避免 nan）
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # 保存注意力权重
        self.attn_weights = attn_weights

        attn_weights = self.dropout(attn_weights)

        # 6. 加权求和
        # (B, h, L_q, L_k) @ (B, h, L_k, d_k) = (B, h, L_q, d_k)
        context = torch.matmul(attn_weights, V)

        # 7. 合并头：(B, h, L_q, d_k) -> (B, L_q, d_model)
        context = self.merge_heads(context)

        # 8. 输出投影
        output = self.W_o(context)  # (B, L_q, d_model)

        if return_attn:
            return output, self.attn_weights
        return output
```

### 5.4.3 两种方法的效率对比

```python
import time


def benchmark_attention(method: str, B: int, L: int, d_model: int, h: int,
                         n_runs: int = 100):
    """对比两种多头注意力实现的速度"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(B, L, d_model).to(device)

    if method == 'naive':
        model = MultiHeadAttentionNaive(d_model, h).to(device)
    else:
        model = MultiHeadAttention(d_model, h).to(device)

    # 预热
    for _ in range(10):
        model(x, x, x)

    # 计时
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        model(x, x, x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / n_runs * 1000
    return elapsed


# 运行基准测试
configs = [
    (8, 32, 512, 8),    # 小配置
    (16, 128, 512, 8),  # 中配置
    (4, 512, 512, 8),   # 长序列
]

print(f"{'配置 (B,L,d,h)':<25} {'朴素实现(ms)':<18} {'批量实现(ms)':<18} {'加速比'}")
print("-" * 70)
for B, L, d, h in configs:
    t_naive = benchmark_attention('naive', B, L, d, h)
    t_batch = benchmark_attention('batch', B, L, d, h)
    print(f"({B},{L},{d},{h}){'':<12} {t_naive:<18.3f} {t_batch:<18.3f} {t_naive/t_batch:.2f}x")
```

典型测试结果（CPU）：

| 配置 (B,L,d,h) | 朴素实现(ms) | 批量实现(ms) | 加速比 |
|----------------|-------------|-------------|--------|
| (8,32,512,8)   | 2.1         | 0.8         | 2.6x   |
| (16,128,512,8) | 18.4        | 5.2         | 3.5x   |
| (4,512,512,8)  | 52.3        | 11.8        | 4.4x   |

随着序列长度增加，批量实现的优势更明显（GPU 上差距更大，可达 10x 以上）。

### 5.4.4 与 PyTorch 内置实现对比验证

```python
def verify_against_pytorch(d_model: int = 64, num_heads: int = 4,
                             seq_len: int = 10, batch_size: int = 2):
    """
    验证自定义实现与 PyTorch 内置 nn.MultiheadAttention 的等价性
    注意：需要手动对齐权重才能做数值比较
    """
    torch.manual_seed(42)

    # 创建测试输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 我们的实现
    our_mha = MultiHeadAttention(d_model, num_heads)

    # PyTorch 内置实现
    # 注意：nn.MultiheadAttention 默认 batch_first=False（序列长度在第0维）
    torch_mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True, bias=False)

    # 对齐权重
    # PyTorch 将 Q、K、V 的权重合并为一个矩阵 in_proj_weight: (3*d_model, d_model)
    with torch.no_grad():
        # 复制 Q、K、V 权重
        our_mha.W_q.weight.copy_(torch_mha.in_proj_weight[:d_model, :])
        our_mha.W_k.weight.copy_(torch_mha.in_proj_weight[d_model:2*d_model, :])
        our_mha.W_v.weight.copy_(torch_mha.in_proj_weight[2*d_model:, :])
        # 复制输出权重
        our_mha.W_o.weight.copy_(torch_mha.out_proj.weight)

    # 前向计算
    our_output = our_mha(x, x, x)
    torch_output, _ = torch_mha(x, x, x)

    # 比较
    max_diff = (our_output - torch_output).abs().max().item()
    print(f"最大数值差异: {max_diff:.2e}")
    print(f"结果{'一致' if max_diff < 1e-5 else '不一致'}！")

    return max_diff


diff = verify_against_pytorch()
```

---

## 5.5 注意力头的分析

---

### 5.5.1 不同头关注的模式

研究人员（如 Clark et al., 2019；Voita et al., 2019）通过可视化分析发现，BERT 和 GPT 中的不同注意力头确实学到了不同的语言学功能：

**位置型头（Positional Heads）**

关注相邻位置的 token，形成带状注意力模式：
- 关注前一个词（bigram 语言模型行为）
- 关注后一个词
- 关注相对位置固定的词

**语法型头（Syntactic Heads）**

捕获句法依赖关系：
- 名词-动词一致性（"cats **eat**" vs "cat **eats**"）
- 形容词-名词修饰关系
- 主语-谓语依存关系

**语义型头（Semantic Heads）**

关注语义相关的词：
- 同义词和相关概念
- 指代消解（代词 → 先行词）
- 实体关系（人物、地点、组织）

**[CLS] 聚合头**

BERT 中某些头将整个序列的信息聚合到 [CLS] token，用于句子级别的分类任务。

### 5.5.2 头重要性的不均匀性

重要发现：**不是所有头都同等重要**。

Voita et al. (2019) 的实验表明，在 BERT 的多头注意力中：
- 约 80% 的头可以被剪掉（pruned），对任务性能影响很小
- 真正重要的头往往是上述具有明确语言学功能的头
- 这催生了注意力头剪枝（Attention Head Pruning）研究方向

### 5.5.3 注意力可视化分析工具

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def visualize_attention_heads(
    model: MultiHeadAttention,
    tokens: list,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str = "Layer",
):
    """
    可视化多头注意力的各个头的注意力权重

    参数：
        model: 训练好的 MultiHeadAttention 模型
        tokens: token 列表，用于轴标签
        query/key/value: 输入张量 (1, L, d_model)
        layer_name: 层名称，用于图表标题
    """
    model.eval()
    with torch.no_grad():
        _, attn_weights = model(query, key, value, return_attn=True)

    # attn_weights: (1, num_heads, L, L)
    attn = attn_weights[0].cpu().numpy()  # (num_heads, L, L)
    num_heads = attn.shape[0]

    # 计算子图布局
    cols = 4
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]

        # 绘制热力图
        im = ax.imshow(attn[head_idx], cmap='Blues', vmin=0, vmax=1, aspect='auto')

        # 设置轴标签
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)

        # 标题显示熵（熵低=注意力集中，熵高=注意力分散）
        entropy = -np.sum(attn[head_idx] * np.log(attn[head_idx] + 1e-9), axis=-1).mean()
        ax.set_title(f'Head {head_idx + 1}\n熵={entropy:.2f}', fontsize=9)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 隐藏多余的子图
    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'{layer_name} - 各头注意力权重可视化', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig('attention_heads.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图表已保存为 attention_heads.png")


def analyze_head_patterns(attn_weights: torch.Tensor, tokens: list) -> dict:
    """
    自动分析注意力头的模式类型

    参数：
        attn_weights: (num_heads, L, L)
        tokens: token 列表

    返回：包含每个头的模式分类的字典
    """
    attn = attn_weights.cpu().numpy()
    num_heads, L, _ = attn.shape

    results = {}
    for h in range(num_heads):
        a = attn[h]  # (L, L)

        # 计算对角线优势（位置型头的特征）
        diag_score = np.mean(np.diag(a))  # 自身注意力

        # 计算次对角线优势（相邻位置型头的特征）
        if L > 1:
            prev_diag = np.mean(np.diag(a, -1))  # 关注前一个位置
            next_diag = np.mean(np.diag(a, 1))   # 关注后一个位置
        else:
            prev_diag = next_diag = 0.0

        # 计算熵（低熵=聚焦，高熵=分散）
        entropy = -np.sum(a * np.log(a + 1e-9), axis=-1).mean()
        max_entropy = np.log(L)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # 分类
        if diag_score > 0.3:
            pattern = "自注意力型"
        elif max(prev_diag, next_diag) > 0.25:
            direction = "前向" if prev_diag > next_diag else "后向"
            pattern = f"{direction}位置型"
        elif norm_entropy < 0.4:
            pattern = "聚焦语义型"
        else:
            pattern = "分散/全局型"

        results[f"Head {h+1}"] = {
            "pattern": pattern,
            "entropy": float(entropy),
            "diag_score": float(diag_score),
        }

    return results


# 示例：使用示例句子进行分析
def demo_attention_analysis():
    torch.manual_seed(0)

    tokens = ["The", "cat", "sat", "on", "the", "mat", "."]
    L = len(tokens)
    d_model = 64
    num_heads = 4

    # 创建模型和随机输入（实际使用中应是预训练嵌入）
    model = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(1, L, d_model)

    # 获取注意力权重
    with torch.no_grad():
        _, attn_weights = model(x, x, x, return_attn=True)

    # 分析头的模式
    patterns = analyze_head_patterns(attn_weights[0], tokens)

    print("注意力头模式分析：")
    print("-" * 45)
    for head, info in patterns.items():
        print(f"{head}: {info['pattern']}")
        print(f"  熵值: {info['entropy']:.3f}  对角得分: {info['diag_score']:.3f}")

    # 可视化
    visualize_attention_heads(model, tokens, x, x, x, "演示层")


demo_attention_analysis()
```

### 5.5.4 头剪枝研究

注意力头剪枝是一个重要的模型压缩方向，核心思想：

```python
def compute_head_importance(
    model: MultiHeadAttention,
    dataloader,
    criterion,
) -> torch.Tensor:
    """
    使用梯度方法估计每个头的重要性

    原理：在输出上添加 head_mask（每个头乘以一个可学习标量），
    计算 |mask_grad| 作为重要性的代理指标。

    参考：Michel et al. (2019) "Are Sixteen Heads Really Better than One?"
    """
    head_importance = torch.zeros(model.num_heads)

    for batch in dataloader:
        # 前向传播，记录每个头的输出
        with torch.no_grad():
            _, attn_weights = model(batch, batch, batch, return_attn=True)

        # 计算梯度（简化版）
        # 实际实现需要对 head_mask 参数求导
        # 此处仅为示意
        contribution = attn_weights[0].abs().mean(dim=(0, 2, 3))  # (num_heads,)
        head_importance += contribution

    return head_importance / len(dataloader)


def prune_heads(model: MultiHeadAttention, heads_to_prune: list) -> None:
    """
    剪枝指定的注意力头（将对应的权重置零）

    参数：
        model: 多头注意力模型
        heads_to_prune: 要剪枝的头的索引列表（0-indexed）
    """
    d_k = model.d_k

    with torch.no_grad():
        for head_idx in heads_to_prune:
            start = head_idx * d_k
            end = (head_idx + 1) * d_k

            # 将该头对应的输出权重置零
            # W_o 的列对应每个头的贡献
            model.W_o.weight[:, start:end] = 0

    print(f"已剪枝 {len(heads_to_prune)} 个头: {heads_to_prune}")
```

---

## 本章小结

---

| 概念 | 关键点 |
|------|--------|
| **动机** | 单头注意力只能关注一种关系模式；多头并行关注不同子空间 |
| **核心公式** | $\text{MultiHead}(Q,K,V) = \text{Concat}(head_1,\ldots,head_h)W^O$，其中 $head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ |
| **维度设计** | $d_k = d_v = d_{model}/h$，确保多头与单头参数量相同 |
| **参数量** | $4d_{model}^2$，与头数 $h$ 无关 |
| **张量操作** | $(B,L,d) \xrightarrow{\text{view}} (B,L,h,d_k) \xrightarrow{\text{transpose}} (B,h,L,d_k)$ |
| **合并操作** | $(B,h,L,d_k) \xrightarrow{\text{transpose}} (B,L,h,d_k) \xrightarrow{\text{contiguous+view}} (B,L,d)$ |
| **推荐实现** | 批量矩阵乘法（所有头并行），比朴素循环快 3-10x |
| **头的功能** | 不同头学习位置、语法、语义等不同模式；多数头可被剪枝 |
| **典型配置** | $d_{model}=512, h=8, d_k=64$（原始 Transformer）|

---

## 代码实战

---

以下完整代码整合了本章所有内容，可直接运行：

```python
"""
第5章完整代码实战：多头注意力机制
文件名：chapter5_multihead_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


# ============================================================
# Part 1: 完整多头注意力实现
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力（完整版，支持自注意力和交叉注意力）

    参数：
        d_model: 模型维度
        num_heads: 注意力头数
        dropout: dropout 概率
        bias: 是否在线性层中使用偏置
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # 存储用于可视化

        self._init_weights()

    def _init_weights(self):
        """Xavier 均匀初始化"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        Q = self.split_heads(self.W_q(query))  # (B, h, L_q, d_k)
        K = self.split_heads(self.W_k(key))    # (B, h, L_k, d_k)
        V = self.split_heads(self.W_v(value))  # (B, h, L_v, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, h, L_q, L_k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 广播到所有头
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        self.attn_weights = attn.detach()

        context = self.merge_heads(torch.matmul(self.dropout(attn), V))
        output = self.W_o(context)

        if return_attn:
            return output, self.attn_weights
        return output, None


# ============================================================
# Part 2: 与 nn.MultiheadAttention 对比
# ============================================================

def compare_with_pytorch(d_model: int = 64, num_heads: int = 4,
                          seq_len: int = 8, batch_size: int = 2,
                          n_trials: int = 100):
    """比较自定义实现与 PyTorch 内置实现的速度"""

    x = torch.randn(batch_size, seq_len, d_model)

    # 自定义实现
    custom_mha = MultiHeadAttention(d_model, num_heads)
    custom_mha.eval()

    # PyTorch 内置
    torch_mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True, bias=False)
    torch_mha.eval()

    # 预热
    for _ in range(10):
        custom_mha(x, x, x)
        torch_mha(x, x, x)

    # 计时
    start = time.perf_counter()
    for _ in range(n_trials):
        custom_mha(x, x, x)
    t_custom = (time.perf_counter() - start) / n_trials * 1000

    start = time.perf_counter()
    for _ in range(n_trials):
        torch_mha(x, x, x)
    t_torch = (time.perf_counter() - start) / n_trials * 1000

    print(f"\n速度对比（{n_trials}次平均，单位ms）：")
    print(f"  自定义实现：{t_custom:.3f} ms")
    print(f"  PyTorch 内置：{t_torch:.3f} ms")
    print(f"  速度比：{t_custom/t_torch:.2f}x")

    # 计算参数量
    n_custom = sum(p.numel() for p in custom_mha.parameters())
    n_torch = sum(p.numel() for p in torch_mha.parameters())
    print(f"\n参数量对比：")
    print(f"  自定义：{n_custom:,} 参数（预期: {4 * d_model**2:,}）")
    print(f"  PyTorch：{n_torch:,} 参数")


# ============================================================
# Part 3: 注意力头可视化
# ============================================================

def visualize_attention_heads(
    attn_weights: torch.Tensor,
    tokens: list,
    title: str = "Multi-Head Attention",
    save_path: Optional[str] = None,
):
    """
    可视化所有注意力头的权重

    参数：
        attn_weights: (num_heads, L, L)
        tokens: token 字符串列表
        title: 图表标题
        save_path: 保存路径（None 则只显示）
    """
    num_heads = attn_weights.shape[0]
    attn = attn_weights.cpu().numpy()

    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for h in range(num_heads):
        r, c = h // cols, h % cols
        ax = axes[r][c]

        im = ax.imshow(attn[h], cmap='viridis', vmin=0, aspect='auto')

        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(tokens, fontsize=9)
        ax.set_xlabel("键（Key）", fontsize=8)
        ax.set_ylabel("查询（Query）", fontsize=8)

        # 熵指标
        e = -np.sum(attn[h] * np.log(attn[h] + 1e-9), axis=-1).mean()
        ax.set_title(f"Head {h+1}  (熵={e:.2f})", fontsize=10)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 关闭多余的子图
    for h in range(num_heads, rows * cols):
        r, c = h // cols, h % cols
        axes[r][c].set_visible(False)

    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至：{save_path}")

    plt.show()


# ============================================================
# Part 4: 性能基准测试
# ============================================================

def benchmark_scaling(
    d_model: int = 512,
    num_heads: int = 8,
    batch_size: int = 8,
    seq_lengths: list = None,
    n_runs: int = 50,
):
    """
    测试不同序列长度下的注意力计算性能
    注意：注意力计算复杂度为 O(L^2)
    """
    if seq_lengths is None:
        seq_lengths = [32, 64, 128, 256, 512]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiHeadAttention(d_model, num_heads).to(device)
    model.eval()

    print(f"\n性能基准测试（设备：{device}，d_model={d_model}，num_heads={num_heads}）")
    print(f"{'序列长度':<12} {'时间(ms)':<14} {'显存(MB)':<14} {'FLOPs 比'}")
    print("-" * 55)

    base_time = None
    for L in seq_lengths:
        x = torch.randn(batch_size, L, d_model).to(device)

        # 预热
        for _ in range(5):
            model(x, x, x)

        if device.type == 'cuda':
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / 1e6

        start = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                model(x, x, x)

        if device.type == 'cuda':
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / 1e6
            mem_used = mem_after - mem_before
        else:
            mem_used = 0

        elapsed = (time.perf_counter() - start) / n_runs * 1000

        if base_time is None:
            base_time = elapsed

        ratio = elapsed / base_time
        print(f"{L:<12} {elapsed:<14.3f} {mem_used:<14.1f} {ratio:.2f}x")

    print("\n注：注意力的理论复杂度为 O(L²)，时间应按平方增长。")


# ============================================================
# 主函数：运行所有演示
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("第5章：多头注意力 - 完整代码演示")
    print("=" * 60)

    # 1. 基本功能测试
    print("\n[1] 基本功能测试")
    d_model, num_heads = 64, 4
    B, L = 2, 8

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(B, L, d_model)

    output, attn = mha(x, x, x, return_attn=True)
    print(f"  输入形状：{x.shape}")
    print(f"  输出形状：{output.shape}")
    print(f"  注意力权重形状：{attn.shape}")
    assert output.shape == x.shape, "输出形状应与输入相同"
    print("  形状验证通过！")

    # 2. 掩码测试
    print("\n[2] 因果掩码测试（自回归）")
    # 下三角掩码：只能看到当前及之前的位置
    causal_mask = torch.tril(torch.ones(L, L)).unsqueeze(0)  # (1, L, L)
    output_masked, _ = mha(x, x, x, mask=causal_mask)
    print(f"  因果注意力输出形状：{output_masked.shape}")

    # 3. 速度对比
    print("\n[3] 与 PyTorch 内置实现对比")
    compare_with_pytorch(d_model=128, num_heads=4, seq_len=32, batch_size=4)

    # 4. 注意力可视化
    print("\n[4] 注意力头可视化")
    tokens = ["我", "爱", "自", "然", "语", "言", "处", "理"]
    x_vis = torch.randn(1, len(tokens), d_model)
    mha_vis = MultiHeadAttention(d_model, num_heads)

    with torch.no_grad():
        _, attn_vis = mha_vis(x_vis, x_vis, x_vis, return_attn=True)

    visualize_attention_heads(
        attn_vis[0],  # 取第一个批次
        tokens,
        title="多头注意力可视化（随机初始化模型）",
        save_path="attention_heads_vis.png",
    )

    # 5. 性能基准
    print("\n[5] 序列长度扩展性测试")
    benchmark_scaling(d_model=256, num_heads=8, batch_size=4,
                      seq_lengths=[32, 64, 128, 256])

    print("\n所有演示完成！")
```

---

## 练习题

---

### 基础题

**题目 1**：参数量计算

给定 $d_{model} = 768$，$h = 12$（BERT-base 的配置），回答：

(a) 每个头的 $d_k$ 和 $d_v$ 是多少？

(b) 整个多头注意力层（包括 $W^Q, W^K, W^V, W^O$）共有多少参数？

(c) 如果改为 $h = 6$（头数减半），参数量如何变化？

---

**题目 2**：张量形状追踪

给定输入 $X \in \mathbb{R}^{2 \times 10 \times 512}$（批次2，序列长10，维度512），$h = 8$，$d_k = 64$，追踪以下操作后的张量形状：

(a) `Q = W_q(X)` 之后

(b) `Q = Q.view(B, L, h, d_k)` 之后

(c) `Q = Q.transpose(1, 2)` 之后

(d) `scores = Q @ K.transpose(-2, -1)` 之后（K 形状与 Q 相同）

(e) `context = scores @ V` 之后（V 形状与 Q 相同）

(f) `context = context.transpose(1, 2).contiguous().view(B, L, -1)` 之后

---

### 中级题

**题目 3**：实现带相对位置编码的多头注意力

标准多头注意力使用绝对位置编码（在输入嵌入层添加），但相对位置编码（Relative Position Encoding）在注意力分数中直接加入位置偏置，效果往往更好。

请修改 `MultiHeadAttention.forward()` 方法，在计算注意力分数时加入可学习的相对位置偏置矩阵 $B \in \mathbb{R}^{L \times L}$：

$$\text{scores}_{ij} = \frac{Q_i \cdot K_j}{\sqrt{d_k}} + B_{ij}$$

要求：
- $B$ 应该是模型的可学习参数（`nn.Parameter`）
- $B$ 应该在每个头之间共享（不是每个头独立的 $B$）
- 考虑处理不同序列长度的情况

---

**题目 4**：分析注意力熵与头的行为

注意力熵定义为：$H_i = -\sum_j a_{ij} \log a_{ij}$，其中 $a_{ij}$ 是第 $i$ 个位置对第 $j$ 个位置的注意力权重。

低熵 → 注意力集中（聚焦型头）；高熵 → 注意力分散（全局型头）。

请编写一个函数 `compute_head_entropy(attn_weights: torch.Tensor) -> torch.Tensor`，计算每个头的平均熵，并回答：

(a) 对于形状 $(B, h, L, L)$ 的注意力权重，熵应该沿哪个维度计算？

(b) 随机初始化的多头注意力的熵接近多少？（提示：最大熵 $= \log(L)$）

(c) 经过充分训练后，你预期哪些任务的头会有较低的熵？

---

### 提高题

**题目 5**：实现分组查询注意力（Grouped Query Attention, GQA）

现代大语言模型（如 LLaMA-2, Mistral）使用分组查询注意力来减少 KV Cache 的内存占用：

- 标准 MHA：$Q, K, V$ 各有 $h$ 个头
- GQA：$Q$ 有 $h$ 个头，但 $K$ 和 $V$ 只有 $g$ 个头（$g < h$，$h/g$ 为整数）
- 每组的 $K$/$V$ 被 $h/g$ 个查询头共享

例如：$h = 8$ 个查询头，$g = 2$ 个 KV 头，则每个 KV 头被 4 个查询头共享。

请实现 `GroupedQueryAttention` 类：
- 初始化时接受 `d_model, num_heads, num_kv_heads` 参数
- 当 `num_kv_heads == num_heads` 时退化为标准 MHA
- 当 `num_kv_heads == 1` 时退化为多查询注意力（Multi-Query Attention, MQA）
- 需要处理 KV 头的广播（使用 `expand` 或 `repeat_interleave`）

提示：关键张量操作：
```python
# K 形状: (B, g, L, d_k)，需要扩展为 (B, h, L, d_k)
K = K.repeat_interleave(h // g, dim=1)  # (B, h, L, d_k)
```

---

## 练习答案

---

### 答案 1：参数量计算

**(a) 每个头的维度**

$$d_k = d_v = \frac{d_{model}}{h} = \frac{768}{12} = 64$$

**(b) 总参数量**

$$\text{总参数} = 4 \times d_{model}^2 = 4 \times 768^2 = 4 \times 589,824 = 2,359,296 \approx 236 \text{ 万}$$

细分：
- $W^Q$：$768 \times 768 = 589,824$
- $W^K$：$768 \times 768 = 589,824$
- $W^V$：$768 \times 768 = 589,824$
- $W^O$：$768 \times 768 = 589,824$

总计：$4 \times 589,824 = 2,359,296$

**(c) 改为 h=6 后的参数量**

参数量仍然是 $4 \times d_{model}^2 = 2,359,296$，**不变**！

因为在保持 $d_k = d_{model}/h$ 的前提下，增加或减少头数不改变总参数量。

---

### 答案 2：张量形状追踪

输入 $X \in \mathbb{R}^{2 \times 10 \times 512}$，$B=2, L=10, h=8, d_k=64, d_{model}=512$

| 操作 | 结果形状 | 说明 |
|------|---------|------|
| (a) `Q = W_q(X)` | $(2, 10, 512)$ | 线性投影保持形状不变 |
| (b) `Q.view(B,L,h,d_k)` | $(2, 10, 8, 64)$ | 将最后一维分割为头数×头维度 |
| (c) `Q.transpose(1,2)` | $(2, 8, 10, 64)$ | 将头维度移到第2位 |
| (d) `Q @ K.T(-2,-1)` | $(2, 8, 10, 10)$ | $(2,8,10,64) \times (2,8,64,10)$ |
| (e) `scores @ V` | $(2, 8, 10, 64)$ | $(2,8,10,10) \times (2,8,10,64)$ |
| (f) `.transpose(1,2).view(B,L,-1)` | $(2, 10, 512)$ | 合并所有头，恢复原始形状 |

---

### 答案 3：相对位置编码实现

```python
class MultiHeadAttentionWithRelPos(MultiHeadAttention):
    """带可学习相对位置偏置的多头注意力"""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 512,
                 dropout: float = 0.0):
        super().__init__(d_model, num_heads, dropout)
        self.max_seq_len = max_seq_len

        # 可学习的相对位置偏置（在头之间共享）
        # 使用相对位置而非绝对位置：偏移量范围 [-(L-1), L-1]
        # 共 2*max_seq_len - 1 个不同的相对位置
        self.rel_pos_bias = nn.Embedding(2 * max_seq_len - 1, 1)

    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """生成相对位置索引矩阵"""
        # 行 i，列 j 的相对位置 = j - i，范围 [-(L-1), L-1]
        positions = torch.arange(seq_len)
        relative = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)
        # 偏移到非负范围 [0, 2L-2]
        relative = relative + (seq_len - 1)
        return relative

    def forward(self, query, key, value, mask=None, return_attn=False):
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape

        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 添加相对位置偏置
        if L_q <= self.max_seq_len and L_k <= self.max_seq_len:
            rel_pos_idx = self._get_relative_positions(max(L_q, L_k))
            rel_pos_idx = rel_pos_idx[:L_q, :L_k].to(query.device)
            rel_bias = self.rel_pos_bias(rel_pos_idx).squeeze(-1)  # (L_q, L_k)
            scores = scores + rel_bias.unsqueeze(0).unsqueeze(0)   # 广播到 (B, h, L_q, L_k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        self.attn_weights = attn.detach()

        context = self.merge_heads(torch.matmul(self.dropout(attn), V))
        output = self.W_o(context)

        if return_attn:
            return output, self.attn_weights
        return output, None
```

---

### 答案 4：注意力熵分析

```python
def compute_head_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    计算每个头的平均注意力熵

    参数：
        attn_weights: (B, h, L, L) 或 (h, L, L)

    返回：
        mean_entropy: (h,) 每个头的平均熵
    """
    if attn_weights.dim() == 4:
        # 批次中取平均
        a = attn_weights  # (B, h, L, L)
    else:
        a = attn_weights.unsqueeze(0)  # (1, h, L, L)

    # (a) 沿最后一维计算熵（对每个查询位置，计算其对所有键的注意力分布的熵）
    entropy = -(a * torch.log(a + 1e-9)).sum(dim=-1)  # (B, h, L)

    # 对批次和序列维度取平均
    mean_entropy = entropy.mean(dim=(0, 2))  # (h,)

    return mean_entropy


# (b) 随机初始化的熵
L = 16
d_model = 64
num_heads = 4

model = MultiHeadAttention(d_model, num_heads)
x = torch.randn(1, L, d_model)

with torch.no_grad():
    _, attn = model(x, x, x, return_attn=True)

entropies = compute_head_entropy(attn)
max_entropy = math.log(L)  # log(16) ≈ 2.77

print(f"最大熵（均匀分布）: {max_entropy:.3f}")
print(f"各头熵值: {entropies.tolist()}")
print(f"平均熵: {entropies.mean():.3f}（应接近最大熵 {max_entropy:.3f}）")
```

**解答**：

**(a)** 熵应沿最后一维（键的维度）计算。对每个查询位置 $i$，计算其注意力分布 $\{a_{i,1}, \ldots, a_{i,L}\}$ 的熵：$H_i = -\sum_j a_{ij} \log a_{ij}$

**(b)** 随机初始化后，softmax 输出接近均匀分布 $1/L$，熵接近最大值 $\log(L)$。对于 $L=16$，最大熵约为 2.77。

**(c)** 以下任务的头往往有较低熵（注意力集中）：
- **指代消解**：代词需要精确指向其先行词
- **句法依存**：主-谓、动-宾等强依存关系
- **命名实体识别**：实体的各 token 相互强关联
- **数学推理**：需要精确找到操作数

---

### 答案 5：分组查询注意力（GQA）实现

```python
class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力（GQA）

    当 num_kv_heads == num_heads 时：标准多头注意力（MHA）
    当 num_kv_heads == 1 时：多查询注意力（MQA）
    当 1 < num_kv_heads < num_heads 时：分组查询注意力（GQA）

    参考：Ainslie et al. (2023) "GQA: Training Generalized Multi-Query
          Transformer Models from Multi-Head Checkpoints"
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) 必须能被 num_kv_heads ({num_kv_heads}) 整除"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.groups = num_heads // num_kv_heads  # 每组的查询头数
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        # Q 有 num_heads 个头，K/V 只有 num_kv_heads 个头
        self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape

        # Q: (B, num_heads, L_q, d_k)
        Q = self.W_q(query)
        Q = Q.view(B, L_q, self.num_heads, self.d_k).transpose(1, 2)

        # K, V: (B, num_kv_heads, L_k, d_k)
        K = self.W_k(key).view(B, L_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, L_k, self.num_kv_heads, self.d_k).transpose(1, 2)

        # 将 KV 头扩展为查询头数
        # (B, num_kv_heads, L_k, d_k) -> (B, num_heads, L_k, d_k)
        K = K.repeat_interleave(self.groups, dim=1)
        V = V.repeat_interleave(self.groups, dim=1)

        # 标准注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # (B, num_heads, L_q, d_k)
        context = context.transpose(1, 2).contiguous().view(B, L_q, self.d_model)

        return self.W_o(context)


# 验证 GQA
def verify_gqa():
    d_model, h, g = 64, 8, 2
    B, L = 2, 10

    gqa = GroupedQueryAttention(d_model, h, g)
    mha = GroupedQueryAttention(d_model, h, h)  # GQA with g=h == MHA
    mqa = GroupedQueryAttention(d_model, h, 1)  # GQA with g=1 == MQA

    x = torch.randn(B, L, d_model)

    for name, model in [("GQA(h=8,g=2)", gqa), ("MHA(h=8,g=8)", mha), ("MQA(h=8,g=1)", mqa)]:
        out = model(x, x, x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: 输出形状={out.shape}, 参数量={params:,}")

    # KV Cache 内存对比
    print(f"\nKV Cache 内存对比（相对于 MHA）：")
    print(f"  MHA:  {h}/{h} = 1.00x")
    print(f"  GQA:  {g}/{h} = {g/h:.2f}x")
    print(f"  MQA:  1/{h} = {1/h:.2f}x")


verify_gqa()
```

**GQA 的优势总结**：

| 方案 | Q 头数 | KV 头数 | KV Cache 大小 | 质量 |
|------|--------|---------|--------------|------|
| MHA  | $h$ | $h$ | $1.00\times$ | 最高 |
| GQA  | $h$ | $g$ | $g/h$ | 接近 MHA |
| MQA  | $h$ | $1$ | $1/h$ | 略低 |

GQA 在保持接近 MHA 性能的同时，将推理时的 KV Cache 内存降低到 $g/h$，是当前大模型推理优化的主流方案。

---

> **下一章预告**：第6章将介绍 **位置编码**（Positional Encoding），解决 Transformer 无法感知序列顺序的问题，包括正弦位置编码、可学习位置编码和旋转位置编码（RoPE）。
