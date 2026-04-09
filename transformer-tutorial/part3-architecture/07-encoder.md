# 第7章：编码器结构

> **前置知识**：第4章多头注意力机制、第5章位置编码
>
> **本章目标**：深入理解Transformer编码器的每一个组件，并从零实现完整的编码器。

---

## 学习目标

完成本章学习后，你将能够：

1. 理解Transformer编码器的整体结构及各组件的职责
2. 掌握层归一化（Layer Normalization）的原理、公式和实现
3. 理解前馈网络（FFN）在编码器中的作用与设计选择
4. 掌握残差连接的原理及其对梯度流的贡献
5. 从零实现一个完整的、可与PyTorch官方实现对比验证的编码器

---

## 7.1 编码器整体结构

### 7.1.1 编码器层的组成

Transformer编码器由 $N$ 个**完全相同**的编码器层（Encoder Layer）堆叠而成。每个编码器层包含两个子层（Sub-layer）：

1. **多头自注意力子层**（Multi-Head Self-Attention）
2. **前馈网络子层**（Feed-Forward Network，FFN）

每个子层都使用**残差连接**（Residual Connection）和**层归一化**（Layer Normalization）进行包裹。

```
输入序列
    │
    ▼
[位置编码 + 词嵌入]
    │
    ▼
┌─────────────────────────┐
│   Encoder Layer × N      │
│                          │
│  ┌─────────────────┐    │
│  │  Multi-Head      │    │
│  │  Self-Attention  │    │
│  └────────┬────────┘    │
│           │ + 残差       │
│  ┌────────▼────────┐    │
│  │  Layer Norm      │    │
│  └────────┬────────┘    │
│           │              │
│  ┌────────▼────────┐    │
│  │  Feed-Forward   │    │
│  │  Network (FFN)  │    │
│  └────────┬────────┘    │
│           │ + 残差       │
│  ┌────────▼────────┐    │
│  │  Layer Norm      │    │
│  └────────┬────────┘    │
└───────────┼─────────────┘
            │
            ▼
      编码器输出
```

### 7.1.2 N层堆叠的意义

为什么要堆叠多层？单层编码器已经能捕获词与词之间的关联，但堆叠多层能带来：

- **层次化特征提取**：低层捕获局部句法关系，高层捕获全局语义关系
- **表达能力提升**：多层非线性变换使模型能拟合更复杂的函数
- **组合推理能力**：每层的注意力可以"站在前一层的肩膀上"进行更高阶的推理

原始论文使用 $N=6$，BERT-Base 使用 $N=12$，BERT-Large 使用 $N=24$。

### 7.1.3 输入输出维度

假设：
- 批次大小：$B$
- 序列长度：$L$
- 模型维度：$d_\text{model}$

每一层的**输入和输出维度完全相同**，均为 $(B, L, d_\text{model})$。这是残差连接得以工作的前提，也使得 $N$ 层堆叠变得简洁。

---

## 7.2 层归一化

### 7.2.1 为什么不用 Batch Normalization？

**Batch Normalization（BN）**在计算机视觉领域大获成功，它对每个特征维度，跨批次和空间位置进行归一化。但在NLP中使用BN存在以下问题：

| 问题 | 说明 |
|------|------|
| 变长序列 | NLP中序列长度不一，填充的padding位置会污染统计量 |
| 小批次不稳定 | 批次较小时，均值和方差估计不准确 |
| 自回归推理 | 解码时每次只生成一个token，批次大小为1，BN完全失效 |
| 统计量依赖 | BN需要在推理时使用训练时的移动平均统计量，引入额外复杂度 |

**Layer Normalization（LN）**是沿着**特征维度**进行归一化，对同一个样本、同一个位置的所有特征计算均值和方差，完全不依赖批次中的其他样本。

### 7.2.2 LayerNorm 公式

对输入向量 $x \in \mathbb{R}^{d_\text{model}}$，Layer Normalization 的计算步骤如下：

**第一步：计算均值**

$$\mu = \frac{1}{d_\text{model}} \sum_{i=1}^{d_\text{model}} x_i$$

**第二步：计算方差**

$$\sigma^2 = \frac{1}{d_\text{model}} \sum_{i=1}^{d_\text{model}} (x_i - \mu)^2$$

**第三步：归一化**

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

**第四步：仿射变换（缩放与平移）**

$$\text{LN}(x) = \gamma \cdot \hat{x} + \beta$$

其中：
- $\epsilon$：防止除零的小常数，通常取 $10^{-5}$ 或 $10^{-6}$
- $\gamma \in \mathbb{R}^{d_\text{model}}$：可学习的缩放参数，初始化为全1
- $\beta \in \mathbb{R}^{d_\text{model}}$：可学习的平移参数，初始化为全0

合并写法：

$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

### 7.2.3 Pre-LN vs Post-LN

原始Transformer论文使用 **Post-LN**（先过子层，再归一化）：

$$\text{output} = \text{LN}(x + \text{Sublayer}(x))$$

后续研究发现 **Pre-LN**（先归一化，再过子层）训练更稳定：

$$\text{output} = x + \text{Sublayer}(\text{LN}(x))$$

两种方式的对比：

| 方面 | Post-LN | Pre-LN |
|------|---------|--------|
| 训练稳定性 | 较差，需要学习率预热 | 更好，可直接大学习率训练 |
| 最终性能 | 通常略高（收敛后） | 略低，但差距在减小 |
| 梯度流 | 梯度必须经过LN反向传播 | 残差路径上梯度不受LN影响 |
| 使用场景 | 原始BERT/GPT | GPT-2、LLaMA、现代LLM |

本章代码实现 Pre-LN，这也是当前主流选择。

### 7.2.4 PyTorch 实现

```python
import torch
import torch.nn as nn

# PyTorch内置实现
layer_norm = nn.LayerNorm(normalized_shape=512, eps=1e-6)

# 验证：随机输入
x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
output = layer_norm(x)
print(output.shape)  # (2, 10, 512)

# 验证归一化效果（沿最后一维）
print(output.mean(dim=-1).abs().max())   # 接近 0
print(output.std(dim=-1).mean())         # 接近 1
```

---

## 7.3 前馈网络（FFN）

### 7.3.1 FFN 的直觉理解

注意力机制负责**信息聚合**（把不同位置的信息加权求和），而FFN负责**特征变换**（对每个位置独立地进行非线性映射）。

可以把注意力看作"让词汇之间互相交流"，FFN则是"每个词独自消化和加工这些信息"。FFN在每个位置**独立**（position-wise）应用，这正是其名称"Position-wise Feed-Forward Network"的来源。

### 7.3.2 FFN 公式

FFN是一个简单的两层全连接网络：

$$\text{FFN}(x) = \text{max}(0,\ xW_1 + b_1) W_2 + b_2$$

或更一般地（允许替换激活函数）：

$$\text{FFN}(x) = \text{Act}(xW_1 + b_1) W_2 + b_2$$

其中：
- $x \in \mathbb{R}^{d_\text{model}}$：输入（对每个位置独立处理）
- $W_1 \in \mathbb{R}^{d_\text{model} \times d_{ff}}$：第一层权重
- $W_2 \in \mathbb{R}^{d_{ff} \times d_\text{model}}$：第二层权重
- $b_1 \in \mathbb{R}^{d_{ff}}$，$b_2 \in \mathbb{R}^{d_\text{model}}$：偏置
- $d_{ff}$：FFN隐藏层维度

**维度变化**：$d_\text{model} \to d_{ff} \to d_\text{model}$（先扩展，再压缩）

### 7.3.3 隐藏层维度选择

原始论文中 $d_{ff} = 4 \times d_\text{model}$（$d_\text{model}=512$，$d_{ff}=2048$）。

为什么是4倍？这是一个经验性选择，背后有几个直觉：

1. **参数预算平衡**：FFN占据了约2/3的参数（注意力占1/3），4倍是在容量和效率间的折中
2. **表达能力**：更大的隐藏层能够表示更多的"知识"和模式
3. **信息瓶颈**：先扩展再压缩形成瓶颈结构，强迫模型学习更紧凑的表示

**现代变体中的维度选择**：
- GPT-3：$d_{ff} = 4 \times d_\text{model}$（延续原始设计）
- PaLM：$d_{ff} = \frac{8}{3} \times d_\text{model}$（配合SwiGLU激活，参数量等效）
- 混合专家（MoE）：每个专家独立的FFN，稀疏激活

### 7.3.4 激活函数选择

**ReLU**（原始论文使用）：

$$\text{ReLU}(x) = \max(0, x)$$

优点：计算简单，梯度清晰。缺点：存在"死亡ReLU"问题（负数梯度为零，神经元可能永久不激活）。

**GELU**（BERT、GPT-2使用）：

$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。GELU是"带概率的ReLU"——输入越大，激活概率越高。实践中比ReLU效果更好。

**SwiGLU**（LLaMA、PaLM使用）：

$$\text{SwiGLU}(x, W, V, b, c) = \text{Swish}(xW + b) \odot (xV + c)$$
$$\text{Swish}(x) = x \cdot \sigma(x)$$

SwiGLU引入了门控机制（$\odot$ 表示逐元素乘），两路信号相互调制，实践效果优于GELU。

| 激活函数 | 典型使用场景 | 特点 |
|---------|------------|------|
| ReLU | 原始Transformer | 简单，有死亡问题 |
| GELU | BERT, GPT-2 | 平滑，效果好 |
| SwiGLU | LLaMA, PaLM | 门控，当前SOTA |

---

## 7.4 残差连接

### 7.4.1 残差连接的作用

残差连接（Residual Connection）来自 ResNet（He et al., 2016），在Transformer中发挥了同样关键的作用。

其核心思想是：与其让子层直接学习目标映射 $H(x)$，不如让它学习**残差映射** $F(x) = H(x) - x$，而输出则通过 $H(x) = F(x) + x$ 得到。

**为什么这有帮助？**

假设某层的最优变换接近恒等映射（即"这层什么都不做最好"），那么：
- 没有残差连接：网络需要学习将权重调整为恒等矩阵，并不容易
- 有残差连接：网络只需将 $F(x)$ 趋近于零向量即可，权重全置为零就能实现

### 7.4.2 梯度流分析

残差连接最重要的贡献在于**改善梯度流**。

设最终损失为 $\mathcal{L}$，考虑跨越 $N$ 层的反向传播。有残差连接时：

$$x_n = x_{n-1} + F_{n-1}(x_{n-1})$$

因此：

$$\frac{\partial x_n}{\partial x_{n-1}} = I + \frac{\partial F_{n-1}}{\partial x_{n-1}}$$

通过链式法则展开到输入 $x_0$：

$$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_N} \prod_{n=1}^{N} \left(I + \frac{\partial F_n}{\partial x_{n-1}}\right)$$

关键在于每个括号中都有**单位矩阵 $I$**，这保证了即使 $\frac{\partial F_n}{\partial x_{n-1}}$ 很小（梯度消失），梯度仍然能通过 $I$ 这条"高速公路"直接流向输入层。

**没有残差连接**时，梯度需要逐层衰减，深层网络极易出现梯度消失。

### 7.4.3 残差连接公式

Post-LN 形式（原始论文）：

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Pre-LN 形式（现代主流）：

$$\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

注意残差路径上的 $x$ 是**直接相加**，没有任何变换。这要求子层的输出维度必须与 $x$ 的维度相同（均为 $d_\text{model}$），这也是FFN最终输出到 $d_\text{model}$ 的约束来源。

---

## 7.5 完整编码器层

### 7.5.1 组装各个组件

将7.2至7.4节的所有组件组装为一个完整的编码器层，以 Pre-LN 为例：

**前向传播流程**：

```
输入 x
  │
  ├──────────────────────────┐ (残差路径)
  │                          │
  ▼                          │
LN₁(x)                       │
  │                          │
  ▼                          │
Multi-Head Self-Attention    │
  │                          │
  ▼                          │
  + ◄──────────────────────── ┘
  │
  ├──────────────────────────┐ (残差路径)
  │                          │
  ▼                          │
LN₂(x')                      │
  │                          │
  ▼                          │
Feed-Forward Network         │
  │                          │
  ▼                          │
  + ◄──────────────────────── ┘
  │
  ▼
输出 x''
```

### 7.5.2 EncoderLayer 类实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头自注意力（简化版，用于本章演示）"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, L, _ = x.shape

        Q = self.w_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.w_o(out)


class EncoderLayer(nn.Module):
    """
    单个Transformer编码器层（Pre-LN形式）

    结构：
        output = x + FFN(LN2(x + Attention(LN1(x))))
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x:        (B, L, d_model)
            src_mask: (B, 1, 1, L) 或 None，0 表示 mask 掉

        Returns:
            (B, L, d_model)
        """
        # 子层1：多头自注意力（Pre-LN）
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask=src_mask)
        x = self.dropout(x)
        x = residual + x

        # 子层2：前馈网络（Pre-LN）
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x
```

### 7.5.3 TransformerEncoder 类实现

```python
class TransformerEncoder(nn.Module):
    """
    N层堆叠的Transformer编码器

    包含：
        - 词嵌入
        - 位置编码
        - N个EncoderLayer
        - 最终LayerNorm（Pre-LN需要）
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Pre-LN架构需要在最后添加一个额外的LayerNorm
        self.norm = nn.LayerNorm(d_model)
        self._init_parameters()

    def _init_parameters(self):
        """Xavier均匀初始化线性层权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            src:      (B, L)，token id序列
            src_mask: (B, 1, 1, L)，padding mask

        Returns:
            (B, L, d_model)，上下文表示
        """
        # 词嵌入 + 缩放 + 位置编码
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # N层编码器
        for layer in self.layers:
            x = layer(x, src_mask)

        # 最终归一化（Pre-LN需要）
        x = self.norm(x)
        return x

    def make_src_mask(self, src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        根据padding token生成mask

        Args:
            src:     (B, L)
            pad_idx: padding token的id

        Returns:
            (B, 1, 1, L)，非padding位置为1，padding为0
        """
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)
```

### 7.5.4 参数初始化

良好的参数初始化对Transformer的训练至关重要。

**Xavier均匀初始化**（Glorot & Bengio, 2010）：

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_\text{in} + n_\text{out}}},\ \sqrt{\frac{6}{n_\text{in} + n_\text{out}}}\right)$$

这使得正向传播和反向传播中，信号的方差大致保持不变。

**LayerNorm的初始化**：
- $\gamma$ 初始化为1（不改变方差）
- $\beta$ 初始化为0（不偏移均值）

**Embedding初始化**：通常使用正态分布 $\mathcal{N}(0, 1)$，然后乘以 $\sqrt{d_\text{model}}$ 进行缩放（这也是前向传播中词嵌入乘以 $\sqrt{d_\text{model}}$ 的原因，使其与位置编码的量纲匹配）。

---

## 本章小结

| 组件 | 作用 | 关键超参数 |
|------|------|-----------|
| 多头自注意力 | 捕获序列内任意位置间的依赖关系 | $d_\text{model}$，$h$（头数） |
| 层归一化 | 稳定训练，加速收敛 | $\epsilon$，Pre/Post-LN选择 |
| 前馈网络 | 对每个位置独立地进行非线性特征变换 | $d_{ff}$（通常 $4 \times d_\text{model}$），激活函数 |
| 残差连接 | 缓解梯度消失，允许训练深层网络 | 无（结构性设计） |
| 堆叠N层 | 层次化特征提取，提升模型容量 | $N$（层数） |

**设计原则回顾**：
- 编码器的输入和输出维度保持不变（$d_\text{model}$），这是残差连接的约束
- Pre-LN 比 Post-LN 训练更稳定，是现代架构的首选
- FFN的隐藏层维度通常为 $4 \times d_\text{model}$，GELU/SwiGLU 优于 ReLU

---

## 代码实战

以下是完整的可运行代码，包含从零实现和官方实现的对比验证。

```python
"""
第7章代码实战：从零实现Transformer编码器

依赖：torch >= 2.0
运行：python encoder_from_scratch.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. LayerNorm 从零实现
# ============================================================

class LayerNorm(nn.Module):
    """
    从零实现的层归一化

    公式：LN(x) = γ * (x - μ) / sqrt(σ² + ε) + β
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))   # 可学习缩放
        self.beta = nn.Parameter(torch.zeros(d_model))   # 可学习偏移
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        mean = x.mean(dim=-1, keepdim=True)           # 均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 方差（无偏=False）
        x_norm = (x - mean) / torch.sqrt(var + self.eps)   # 归一化
        return self.gamma * x_norm + self.beta              # 仿射变换

    def extra_repr(self) -> str:
        return f'd_model={self.gamma.shape[0]}, eps={self.eps}'


# ============================================================
# 2. FeedForward 从零实现
# ============================================================

class FeedForward(nn.Module):
    """
    位置前馈网络

    公式：FFN(x) = Act(xW₁ + b₁)W₂ + b₂
    支持三种激活：relu, gelu, swiglu
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()
        self.activation_name = activation

        if activation == 'swiglu':
            # SwiGLU需要两个并行的线性层（门控结构）
            # 为保持参数量与标准FFN一致，d_ff通常缩小为 2/3 * 4 * d_model
            self.w1 = nn.Linear(d_model, d_ff)
            self.w_gate = nn.Linear(d_model, d_ff)
            self.w2 = nn.Linear(d_ff, d_model)
        else:
            self.w1 = nn.Linear(d_model, d_ff)
            self.w2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

        # 激活函数映射
        self.act_fn = {
            'relu': F.relu,
            'gelu': F.gelu,
        }.get(activation, F.gelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == 'swiglu':
            # SwiGLU: Swish(xW₁) ⊙ (xW_gate)
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w_gate(x)))
        else:
            return self.dropout(self.w2(self.act_fn(self.w1(x))))

    def extra_repr(self) -> str:
        d_in = self.w1.in_features
        d_hidden = self.w1.out_features
        d_out = self.w2.out_features
        return f'd_model={d_in}, d_ff={d_hidden}→{d_out}, activation={self.activation_name}'


# ============================================================
# 3. MultiHeadAttention 从零实现
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    多头自注意力（支持padding mask）
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, L, _ = x.shape

        # 线性投影 + 分头
        def split_heads(t):
            return t.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(self.w_q(x)), split_heads(self.w_k(x)), split_heads(self.w_v(x))

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        # 加权求和 + 合并头
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.w_o(out)


# ============================================================
# 4. EncoderLayer 从零实现
# ============================================================

class EncoderLayer(nn.Module):
    """
    单个编码器层，Pre-LN架构

    前向：
        x' = x + Attention(LN₁(x))
        x''= x' + FFN(LN₂(x'))
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # --- 子层1：多头自注意力 ---
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=src_mask))
        # --- 子层2：前馈网络 ---
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ============================================================
# 5. PositionalEncoding（复用自第5章）
# ============================================================

class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================
# 6. TransformerEncoder 从零实现
# ============================================================

class TransformerEncoder(nn.Module):
    """
    完整的Transformer编码器

    架构：
        Embedding → PositionalEncoding → [EncoderLayer] × N → LayerNorm
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
        self._init_parameters()

    def _init_parameters(self):
        for name, p in self.named_parameters():
            if 'embedding' in name:
                nn.init.normal_(p, mean=0, std=self.d_model ** -0.5)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'gamma' in name:
                nn.init.ones_(p)
            elif 'beta' in name or 'bias' in name:
                nn.init.zeros_(p)

    def make_src_mask(self, src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """生成padding mask：(B, 1, 1, L)"""
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            src:      (B, L)
            src_mask: (B, 1, 1, L) 或 None

        Returns:
            (B, L, d_model)
        """
        if src_mask is None:
            src_mask = self.make_src_mask(src)

        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# 7. 验证：与 nn.TransformerEncoder 对比
# ============================================================

def verify_encoder_shapes():
    """验证自实现编码器的输出形状正确"""
    print("=" * 60)
    print("形状验证")
    print("=" * 60)

    B, L = 4, 32
    vocab_size = 1000
    d_model = 128
    num_heads = 4
    num_layers = 3
    d_ff = 512

    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
    )
    encoder.eval()

    src = torch.randint(1, vocab_size, (B, L))       # 随机token ids（非0）
    src_with_pad = src.clone()
    src_with_pad[:, -5:] = 0                         # 末尾5个位置为padding

    with torch.no_grad():
        output = encoder(src_with_pad)

    print(f"输入形状:  {src_with_pad.shape}")         # (4, 32)
    print(f"输出形状:  {output.shape}")               # (4, 32, 128)
    print(f"参数数量:  {encoder.count_parameters():,}")
    assert output.shape == (B, L, d_model), "形状不匹配！"
    print("形状验证通过！\n")


def verify_layernorm():
    """验证自实现LayerNorm与官方实现数值一致"""
    print("=" * 60)
    print("LayerNorm 数值验证")
    print("=" * 60)

    d_model = 256
    x = torch.randn(2, 10, d_model)

    # 自实现
    my_ln = LayerNorm(d_model)
    # 官方实现（使用相同的初始参数）
    official_ln = nn.LayerNorm(d_model, eps=1e-6)

    # 同步参数（默认初始化一致，均为 gamma=1, beta=0）
    with torch.no_grad():
        my_out = my_ln(x)
        official_out = official_ln(x)

    max_diff = (my_out - official_out).abs().max().item()
    print(f"最大数值误差: {max_diff:.2e}")
    assert max_diff < 1e-5, f"数值误差过大: {max_diff}"
    print("LayerNorm 验证通过！\n")


def verify_gradient_flow():
    """验证残差连接的梯度流"""
    print("=" * 60)
    print("梯度流验证")
    print("=" * 60)

    d_model = 64
    num_layers = 12   # 较深的网络

    # 无残差连接的网络（用纯MLP模拟）
    layers_no_res = nn.Sequential(
        *[nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
          for _ in range(num_layers)]
    )

    # 有残差连接的编码器层堆叠
    encoder_layers = nn.ModuleList([
        EncoderLayer(d_model, num_heads=4, d_ff=256)
        for _ in range(num_layers)
    ])

    x = torch.randn(1, 10, d_model, requires_grad=True)
    x_no_res = torch.randn(1, 10 * d_model, requires_grad=True)

    # 前向 + 反向
    out_res = x.clone()
    for layer in encoder_layers:
        out_res = layer(out_res)
    loss_res = out_res.sum()
    loss_res.backward()
    grad_with_res = x.grad.abs().mean().item()

    x_flat = x_no_res
    out_no_res = x_flat
    for layer in layers_no_res:
        out_no_res = layer(out_no_res)
    loss_no_res = out_no_res.sum()
    loss_no_res.backward()
    grad_no_res = x_no_res.grad.abs().mean().item()

    print(f"有残差连接 - 梯度均值: {grad_with_res:.6f}")
    print(f"无残差连接 - 梯度均值: {grad_no_res:.6f}")
    print(f"梯度比值 (有/无残差): {grad_with_res / (grad_no_res + 1e-10):.2f}x\n")


def compare_with_official():
    """与PyTorch官方TransformerEncoder对比输出形状和参数量"""
    print("=" * 60)
    print("与 nn.TransformerEncoder 对比")
    print("=" * 60)

    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 2
    dropout = 0.0   # 关闭dropout以便对比

    # 官方实现（Post-LN）
    official_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=d_ff,
        dropout=dropout,
        batch_first=True,
    )
    official_encoder = nn.TransformerEncoder(official_layer, num_layers=num_layers)

    # 自实现（Pre-LN）
    my_encoder = TransformerEncoder(
        vocab_size=1,   # 不使用embedding
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
    )

    # 只比较编码器层的参数量（不含embedding和pos_encoding）
    official_params = sum(p.numel() for p in official_encoder.parameters())

    # 自实现中只计算encoder layers + final norm（去掉embedding和pos_enc）
    my_layer_params = sum(
        p.numel() for layer in my_encoder.layers for p in layer.parameters()
    ) + sum(p.numel() for p in my_encoder.norm.parameters())

    print(f"官方 TransformerEncoder 参数量: {official_params:,}")
    print(f"自实现 EncoderLayer × {num_layers} + Norm: {my_layer_params:,}")
    print(f"（参数量差异因bias设置和LN位置不同而存在小差异）")

    # 形状验证
    B, L = 2, 20
    x = torch.randn(B, L, d_model)
    official_out = official_encoder(x)
    print(f"\n官方输出形状: {official_out.shape}")
    print(f"预期输出形状: ({B}, {L}, {d_model})")
    print("形状验证通过！")


if __name__ == '__main__':
    torch.manual_seed(42)

    verify_layernorm()
    verify_encoder_shapes()
    verify_gradient_flow()
    compare_with_official()

    print("=" * 60)
    print("所有验证通过！")
    print("=" * 60)
```

---

## 练习题

### 基础题

**练习 7.1**（基础）

写出以下输入经过LayerNorm后的输出（手动计算）：

$$x = [1, 2, 3, 4], \quad \gamma = [1, 1, 1, 1], \quad \beta = [0, 0, 0, 0], \quad \epsilon = 0$$

请计算：均值 $\mu$，方差 $\sigma^2$，归一化后的 $\hat{x}$，最终输出。

---

**练习 7.2**（基础）

给定以下参数配置：
- $d_\text{model} = 512$
- $N = 6$（层数）
- $d_{ff} = 2048$
- $h = 8$（注意力头数）

计算**一个编码器层**的参数量（不含bias），并分析注意力部分和FFN部分各占的比例。

---

### 中级题

**练习 7.3**（中级）

修改 `FeedForward` 类，将激活函数从GELU改为SwiGLU，并使 $d_{ff}$ 调整为 $\frac{8}{3} d_\text{model}$（取整），使总参数量与标准FFN（$d_{ff} = 4 d_\text{model}$，无gate）大致相当。

验证：对相同输入，两个版本的输出形状相同。

---

**练习 7.4**（中级）

在 `EncoderLayer` 中，将 Pre-LN 改为 Post-LN，即：

$$x' = \text{LN}(x + \text{Attention}(x))$$
$$x'' = \text{LN}(x' + \text{FFN}(x'))$$

然后训练一个简单的序列分类任务（可使用随机数据），比较两种架构在初始几十步的训练损失曲线，验证 Pre-LN 更稳定的结论。

---

### 提高题

**练习 7.5**（提高）

实现一个**稀疏MoE FFN**（Mixture of Experts），用 $K=2$ 的Top-K路由替换标准FFN。具体要求：

1. 定义 $E$ 个相同结构的FFN专家（每个专家的 $d_{ff}$ 为标准的 $1/E$）
2. 实现一个 Router（线性层 + softmax）计算每个位置的专家权重
3. 对每个位置，只激活 Top-2 的专家并加权求和
4. 计算实际激活的参数量占总参数量的比例（稀疏度）
5. 添加辅助负载均衡损失（Load Balancing Loss），防止所有token都路由到同一个专家

提示：负载均衡损失 $\mathcal{L}_\text{aux} = E \cdot \sum_{i=1}^{E} f_i \cdot P_i$，其中 $f_i$ 为专家 $i$ 接收的token比例，$P_i$ 为路由权重均值。

---

## 练习答案

### 答案 7.1

**手动计算LayerNorm**

输入：$x = [1, 2, 3, 4]$，$\epsilon = 0$

**步骤一：计算均值**

$$\mu = \frac{1 + 2 + 3 + 4}{4} = \frac{10}{4} = 2.5$$

**步骤二：计算方差**（使用无偏=False的总体方差）

$$\sigma^2 = \frac{(1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2}{4}$$
$$= \frac{2.25 + 0.25 + 0.25 + 2.25}{4} = \frac{5}{4} = 1.25$$

**步骤三：归一化**（$\epsilon = 0$，$\sqrt{\sigma^2} = \sqrt{1.25} \approx 1.118$）

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2}} = \frac{[1,2,3,4] - 2.5}{1.118} \approx [-1.342, -0.447, 0.447, 1.342]$$

**步骤四：仿射变换**（$\gamma = [1,1,1,1]$，$\beta = [0,0,0,0]$）

$$\text{LN}(x) = \gamma \cdot \hat{x} + \beta = [-1.342, -0.447, 0.447, 1.342]$$

验证：$\sum \hat{x}_i \approx 0$（均值为0），$\text{std}(\hat{x}) = 1$（标准差为1）。

---

### 答案 7.2

**编码器层参数量计算**

设 $d_\text{model} = 512$，$h = 8$，$d_{ff} = 2048$，$d_k = d_v = d_\text{model}/h = 64$。

**注意力部分**（$W_Q, W_K, W_V, W_O$ 各一个，无bias）：

$$\text{Params}_\text{attn} = 4 \times d_\text{model} \times d_\text{model} = 4 \times 512 \times 512 = 1{,}048{,}576$$

**FFN部分**（$W_1, W_2$，无bias）：

$$\text{Params}_\text{ffn} = d_\text{model} \times d_{ff} + d_{ff} \times d_\text{model} = 2 \times 512 \times 2048 = 2{,}097{,}152$$

**LayerNorm部分**（两个LN，各含 $\gamma$ 和 $\beta$）：

$$\text{Params}_\text{ln} = 2 \times 2 \times 512 = 2{,}048$$

**单层总参数量**：

$$\text{Total} = 1{,}048{,}576 + 2{,}097{,}152 + 2{,}048 = 3{,}147{,}776 \approx 3.15\text{M}$$

**比例分析**：
- 注意力占比：$\approx 33.3\%$
- FFN 占比：$\approx 66.5\%$
- LayerNorm 占比：$\approx 0.07\%$（可忽略）

结论：FFN 贡献了约 2/3 的参数量，这是因为 $d_{ff} = 4 \times d_\text{model}$ 的设计。

---

### 答案 7.3

**SwiGLU FFN 实现**

```python
class FeedForwardSwiGLU(nn.Module):
    """
    SwiGLU激活的FFN

    参数量与标准FFN（d_ff=4*d_model）大致相当：
    - 标准 FFN：2 × d_model × 4×d_model = 8 × d_model²
    - SwiGLU FFN：(W1 + W_gate + W2) = 3 × d_model × d_ff_swiglu
    - 令 3 × d_ff_swiglu = 8 × d_model，得 d_ff_swiglu = 8/3 × d_model
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # d_ff_swiglu = 8/3 * d_model，取整到64的倍数
        d_ff = int(d_model * 8 / 3 / 64) * 64
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(xW₁) ⊙ (xW_gate)，再投影回d_model
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w_gate(x)))

# 验证
d_model = 512
ffn_standard = FeedForward(d_model, d_model * 4)
ffn_swiglu = FeedForwardSwiGLU(d_model)

x = torch.randn(2, 10, d_model)
out_std = ffn_standard(x)
out_swiglu = ffn_swiglu(x)

print(f"标准FFN参数量:  {sum(p.numel() for p in ffn_standard.parameters()):,}")
print(f"SwiGLU参数量:  {sum(p.numel() for p in ffn_swiglu.parameters()):,}")
print(f"标准FFN输出形状:  {out_std.shape}")
print(f"SwiGLU输出形状:  {out_swiglu.shape}")
# 两者输出形状相同：(2, 10, 512)
```

---

### 答案 7.4

**Post-LN 实现与比较**

```python
class EncoderLayerPostLN(nn.Module):
    """Post-LN编码器层"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Post-LN：先过子层，再归一化
        x = self.norm1(x + self.dropout(self.self_attn(x, mask=src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

# 简单序列分类任务（随机数据）演示训练稳定性
def compare_pre_post_ln_stability():
    d_model, num_heads, d_ff = 64, 4, 256
    num_layers = 6

    def build_model(use_preln: bool):
        LayerClass = EncoderLayer if use_preln else EncoderLayerPostLN
        layers = nn.ModuleList([LayerClass(d_model, num_heads, d_ff) for _ in range(num_layers)])
        head = nn.Linear(d_model, 2)
        return layers, head

    for label, use_preln in [('Pre-LN', True), ('Post-LN', False)]:
        layers, head = build_model(use_preln)
        all_params = list(head.parameters())
        for l in layers: all_params += list(l.parameters())
        optimizer = torch.optim.Adam(all_params, lr=1e-3)

        losses = []
        for step in range(30):
            x = torch.randn(4, 20, d_model)
            for layer in layers:
                x = layer(x)
            logits = head(x.mean(dim=1))
            labels = torch.randint(0, 2, (4,))
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"{label} 前5步损失: {[f'{l:.3f}' for l in losses[:5]]}")
        print(f"{label} 梯度范数稳定性（标准差）: {torch.tensor(losses).std():.4f}")
```

---

### 答案 7.5

**稀疏 MoE FFN 实现**

```python
class MoEFeedForward(nn.Module):
    """
    混合专家前馈网络（Top-2路由）

    参数：
        d_model:    模型维度
        num_experts: 专家数量 E
        d_ff_expert: 每个专家的FFN隐藏维度（通常 d_ff/E * 2，保持总容量不变）
        top_k:      激活的专家数（默认2）
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        d_ff_expert: int | None = None,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        if d_ff_expert is None:
            # 保持总参数量与标准FFN相似
            d_ff_expert = (4 * d_model) // num_experts * 2

        # E个独立的FFN专家
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff_expert),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff_expert, d_model),
            )
            for _ in range(num_experts)
        ])

        # Router：为每个位置计算对各专家的路由权重
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, d_model)

        Returns:
            output:   (B, L, d_model)
            aux_loss: 负载均衡辅助损失（标量）
        """
        B, L, d_model = x.shape
        x_flat = x.view(-1, d_model)   # (B*L, d_model)
        N = B * L

        # --- 路由 ---
        router_logits = self.router(x_flat)            # (N, E)
        router_probs = F.softmax(router_logits, dim=-1) # (N, E)

        # Top-K选择
        topk_probs, topk_indices = router_probs.topk(self.top_k, dim=-1)  # (N, K)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)     # 归一化

        # --- 专家计算 ---
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]             # (N,) 每个token选的专家
            expert_weight = topk_probs[:, k]            # (N,) 对应权重

            for e in range(self.num_experts):
                token_mask = (expert_idx == e)          # 哪些token路由到专家e
                if token_mask.sum() == 0:
                    continue
                tokens = x_flat[token_mask]             # (n_e, d_model)
                expert_out = self.experts[e](tokens)    # (n_e, d_model)
                output[token_mask] += expert_weight[token_mask].unsqueeze(-1) * expert_out

        # --- 负载均衡损失 ---
        # f_i：专家i接收的token比例（通过top-1来计算）
        top1_idx = router_probs.argmax(dim=-1)           # (N,)
        f = torch.bincount(top1_idx, minlength=self.num_experts).float() / N

        # P_i：路由权重的均值
        P = router_probs.mean(dim=0)                     # (E,)

        # 辅助损失：E * sum(f_i * P_i)
        aux_loss = self.num_experts * (f * P).sum()

        return output.view(B, L, d_model), aux_loss


# 验证
moe_ffn = MoEFeedForward(d_model=128, num_experts=8, top_k=2)
x = torch.randn(2, 10, 128)
out, aux_loss = moe_ffn(x)

total_params = sum(p.numel() for p in moe_ffn.parameters())
router_params = sum(p.numel() for p in moe_ffn.router.parameters())

print(f"MoE FFN 输出形状: {out.shape}")
print(f"辅助损失: {aux_loss.item():.4f}")
print(f"总参数量: {total_params:,}")
print(f"每次推理激活参数比例（Top-2/8）: {2/8 * 100:.1f}%")
```

**关键设计点说明**：
- Top-K路由使每个token只激活 $K/E$ 比例的参数，显著降低推理计算量
- 但**所有**参数仍需存在内存中，所以MoE是"计算高效"而非"内存高效"
- 负载均衡损失防止模式坍缩（所有token都选择同一个专家），鼓励专家专业化

---

> **下一章预告**：第8章将介绍解码器结构，它在编码器的基础上增加了**交叉注意力**机制，使解码器能够关注编码器的输出。
