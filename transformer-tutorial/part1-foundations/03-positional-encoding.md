# 第3章：位置编码

> **系列导航**：[第1章：注意力机制](01-attention-mechanism.md) | [第2章：自注意力](02-self-attention.md) | **第3章：位置编码** | [第4章：多头注意力](04-multi-head-attention.md)

---

## 学习目标

完成本章学习后，你将能够：

1. **理解为什么Transformer需要位置编码** —— 从排列不变性问题出发，明白位置信息的必要性
2. **掌握正弦位置编码的数学原理** —— 理解公式推导、频率选择以及相对位置的线性变换性质
3. **理解可学习位置编码的实现** —— 掌握 `nn.Embedding` 方式及其与正弦编码的对比
4. **了解相对位置编码的概念** —— 了解Shaw相对编码、T5偏置和ALiBi的思想
5. **能够实现和可视化各种位置编码** —— 从零动手写出多种位置编码并用热力图观察其规律

---

## 3.1 为什么需要位置编码

### 3.1.1 Transformer的排列不变性

回顾上一章的自注意力机制。给定输入序列 $X \in \mathbb{R}^{n \times d}$，注意力输出为：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

其中 $Q = XW^Q$，$K = XW^K$，$V = XW^V$。

考虑两个序列：

- 序列 A：["猫", "追", "狗"]
- 序列 B：["狗", "追", "猫"]

如果我们把序列 A 的 token 顺序打乱，注意力输出中每个位置的值只取决于"哪些 token 之间有关联"，而**不知道它们出现的先后顺序**。形式上，若对输入施加任意置换矩阵 $P$，则：

$$\text{Attention}(PX) = P \cdot \text{Attention}(X)$$

这意味着：自注意力本质上是一个**集合操作（set operation）**，而非序列操作。"猫追狗"和"狗追猫"在纯注意力眼中，除了词义不同，结构上是对称的。

### 3.1.2 没有位置信息会怎样

以语言为例，位置信息决定了句子的语法结构：

| 无位置编码的模型所见 | 实际含义 |
|:---|:---|
| {"John", "loves", "Mary"} | 无法区分主语和宾语 |
| {"bank", "river", "near"} | 无法判断是"河边的银行"还是"银行的河边" |

对于需要严格顺序的任务（机器翻译、代码生成、时间序列），缺少位置信息会导致模型无法学习到任何与顺序相关的模式。

**实验验证**：在没有位置编码的 Transformer 上训练翻译任务，BLEU 分数会从 ~28 骤降到 ~5，几乎等同于随机翻转词序。

### 3.1.3 位置编码的作用

位置编码（Positional Encoding）的核心思想是：**给每个位置 $pos$ 分配一个固定的向量 $PE_{pos} \in \mathbb{R}^{d_{model}}$，然后将其加到对应 token 的嵌入向量上**：

$$\tilde{x}_{pos} = x_{pos} + PE_{pos}$$

通过这种"注入"方式，位置信息和语义信息被融合进同一向量空间。注意力机制在计算 $QK^\top$ 时，自然就能感知到两个 token 之间的位置关系。

一个好的位置编码需要满足以下性质：

1. **唯一性**：每个位置有唯一的编码
2. **平滑性**：相邻位置的编码之间距离小，不跳跃
3. **有界性**：编码的值在固定范围内，不会影响嵌入向量的数值稳定性
4. **可外推性**：能处理训练时未见过的序列长度（或至少性能优雅降级）

---

## 3.2 正弦位置编码

### 3.2.1 公式定义

原始 Transformer 论文《Attention Is All You Need》（Vaswani et al., 2017）提出了基于正弦函数的固定位置编码：

$$\boxed{PE_{(pos,\, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)}$$

$$\boxed{PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)}$$

其中：
- $pos \in \{0, 1, \ldots, n-1\}$：token 在序列中的位置（从0开始）
- $i \in \{0, 1, \ldots, d_{model}/2 - 1\}$：维度索引（每对维度共用同一频率）
- $d_{model}$：模型的嵌入维度（原始论文中为512）

对于位置 $pos$，其完整的位置编码向量为：

$$PE_{pos} = \begin{bmatrix} \sin(\omega_0 \cdot pos) \\ \cos(\omega_0 \cdot pos) \\ \sin(\omega_1 \cdot pos) \\ \cos(\omega_1 \cdot pos) \\ \vdots \\ \sin(\omega_{d/2-1} \cdot pos) \\ \cos(\omega_{d/2-1} \cdot pos) \end{bmatrix}, \quad \omega_i = \frac{1}{10000^{2i/d_{model}}}$$

### 3.2.2 为什么使用正弦和余弦

**频率选择：多尺度表示**

不同维度对应不同的频率 $\omega_i$：
- $i=0$：$\omega_0 = 1$，周期为 $2\pi \approx 6.28$（最高频，感知相邻位置变化）
- $i=d/4$：$\omega_{d/4} = 1/100$，周期约为 628（中等频率）
- $i=d/2-1$：$\omega_{d/2-1} \approx 1/10000$，周期约为 62832（最低频，感知全局位置）

这类似于二进制计数：低位快速翻转，高位缓慢变化。每个位置在不同"分辨率"下都有独特的"指纹"。

**为什么同时使用 sin 和 cos**

单独使用 $\sin$ 时，若 $\sin(\omega \cdot pos) = \sin(\omega \cdot pos')$，无法唯一区分位置。配对使用后，$(\sin(\omega \cdot pos), \cos(\omega \cdot pos))$ 等价于单位圆上的坐标，对于任意 $pos \neq pos'$（在一个周期内），这对坐标是唯一的。

### 3.2.3 相对位置的线性变换性质

这是正弦编码最精妙的设计。对于固定偏移量 $k$，存在一个**线性变换矩阵** $M_k$，使得：

$$PE_{pos+k} = M_k \cdot PE_{pos}$$

推导过程如下。对第 $i$ 对维度，令 $\theta = \omega_i$，则：

$$\begin{pmatrix} PE_{pos+k,\, 2i} \\ PE_{pos+k,\, 2i+1} \end{pmatrix} = \begin{pmatrix} \sin(\theta(pos+k)) \\ \cos(\theta(pos+k)) \end{pmatrix}$$

利用和角公式：

$$\sin(\theta \cdot pos + \theta k) = \sin(\theta \cdot pos)\cos(\theta k) + \cos(\theta \cdot pos)\sin(\theta k)$$
$$\cos(\theta \cdot pos + \theta k) = \cos(\theta \cdot pos)\cos(\theta k) - \sin(\theta \cdot pos)\sin(\theta k)$$

写成矩阵形式：

$$\begin{pmatrix} \sin(\theta(pos+k)) \\ \cos(\theta(pos+k)) \end{pmatrix} = \underbrace{\begin{pmatrix} \cos(\theta k) & \sin(\theta k) \\ -\sin(\theta k) & \cos(\theta k) \end{pmatrix}}_{M_k^{(i)}} \begin{pmatrix} \sin(\theta \cdot pos) \\ \cos(\theta \cdot pos) \end{pmatrix}$$

这个 $M_k^{(i)}$ 正是一个**二维旋转矩阵**！

**重要推论**：当模型计算注意力分数 $q_j^\top k_i$ 时，由于 $Q, K$ 是位置编码嵌入的线性变换，$q_j^\top k_i$ 中会包含 $PE_{pos_j}^\top PE_{pos_i}$ 项，而这一项可以被改写成只依赖于相对位置 $|pos_j - pos_i|$ 的函数。因此，**注意力权重自然地感知到了相对距离**，而不仅仅是绝对位置。

### 3.2.4 外推能力分析

正弦编码的外推能力较强：

- **优势**：函数是连续的，对训练集之外的位置 $pos > n_{train}$，$PE_{pos}$ 仍然有明确定义且数值有界
- **劣势**：超长序列时，位置之间的"可分辨性"下降。当 $pos \gg 10000$ 时，低频维度（高 $i$）几乎不再变化，信息冗余

**实践结论**：原始正弦编码在序列长度超过训练长度 1.5-2 倍时，性能通常会显著下降。

---

## 3.3 可学习位置编码

### 3.3.1 基本思路

与正弦编码不同，可学习位置编码将 $PE$ 视为模型参数，通过梯度下降从数据中学习。实现上，就是一个标准的嵌入层：

```python
self.pos_embedding = nn.Embedding(max_seq_len, d_model)
```

前向传播时：

```python
positions = torch.arange(seq_len)        # [0, 1, ..., seq_len-1]
pos_enc = self.pos_embedding(positions)   # shape: (seq_len, d_model)
x = token_emb + pos_enc
```

### 3.3.2 BERT 的位置编码

BERT（Devlin et al., 2019）正是使用可学习位置编码的代表：

```
输入嵌入 = Token嵌入 + 分段嵌入 + 位置嵌入
```

BERT 的最大序列长度为512，因此位置嵌入矩阵形状为 $(512, 768)$，共 $512 \times 768 = 393216$ 个参数。

**BERT 的实验发现**：论文作者测试了正弦编码和可学习编码，发现两者效果"几乎没有差别"（"nearly identical results"），因此选择了更简洁的可学习编码。

### 3.3.3 优缺点对比

| 特性 | 正弦位置编码 | 可学习位置编码 |
|:---|:---:|:---:|
| 参数量 | 0（无需训练） | $n_{max} \times d_{model}$ |
| 外推能力 | 较好（数学性质保证） | 差（超出 $n_{max}$ 无法泛化） |
| 任务适应性 | 固定，无法针对任务优化 | 自适应，从数据中学习 |
| 实现复杂度 | 稍复杂（计算公式） | 极简（一行代码） |
| 典型应用 | 原始Transformer、编解码器 | BERT、GPT系列 |

**关键洞见**：对于序列长度固定（如文本分类，最长512词），可学习编码通常表现稍好；对于需要处理变长序列的任务（如长文档摘要），正弦编码或相对位置编码更合适。

---

## 3.4 相对位置编码

### 3.4.1 绝对位置 vs 相对位置

绝对位置编码回答的是："这个 token 在序列的第几个位置？"
相对位置编码回答的是："这两个 token 之间相差几个位置？"

对于许多 NLP 任务，**相对距离比绝对位置更重要**：

> "John 在第3个位置" → 无意义的绝对信息
> "John 和 Mary 相差2个位置" → 直接反映句法关系的相对信息

### 3.4.2 Shaw 等人的相对位置编码（2018）

Shaw et al. 在自注意力的 $Q, K$ 计算中直接引入相对位置：

$$e_{ij} = \frac{(x_i W^Q)(x_j W^K + a_{ij}^K)^\top}{\sqrt{d_k}}$$

其中 $a_{ij}^K \in \mathbb{R}^{d_k}$ 是相对位置 $\text{clip}(j - i,\ {-k},\ k)$ 对应的可学习向量（$k$ 为最大相对距离）。

**特点**：
- 参数量为 $O(k \cdot d)$，远少于绝对编码的 $O(n \cdot d)$
- 对称性更好：距离为 $\delta$ 的两个 token，无论在序列哪个位置，共用同一套参数
- 缺点：需要修改注意力计算，实现略复杂

### 3.4.3 T5 的相对位置偏置

T5（Raffel et al., 2020）采用了更简洁的方式——在注意力分数上直接加一个**标量偏置**：

$$e_{ij} = \frac{q_i k_j^\top}{\sqrt{d_k}} + b_{i-j}$$

其中 $b_{i-j}$ 是与相对位置 $(i-j)$ 对应的可学习标量。T5 将相对位置分桶（bucket），近距离精细划分，远距离粗糙划分：

$$\text{bucket}(\delta) = \begin{cases} \delta & \text{if } |\delta| < k \\ k + \lfloor \log(\delta/k) / \log(n_{max}/k) \cdot (n - k) \rfloor & \text{otherwise} \end{cases}$$

**优点**：参数量极少（仅几十个偏置标量）；可以用于编码器和解码器的跨注意力。

### 3.4.4 ALiBi（Attention with Linear Biases）

ALiBi（Press et al., 2022）提出了最简洁的相对位置方案——**不使用任何位置嵌入**，直接在注意力分数上减去一个与距离成正比的惩罚：

$$e_{ij} = q_i k_j^\top - m \cdot |i - j|$$

其中 $m$ 是每个注意力头的固定超参数（不可学习），按几何级数设置：

$$m_h = \frac{1}{2^{h \cdot 8/H}}, \quad h = 1, 2, \ldots, H$$

**ALiBi 的优势**：
1. **零额外参数**：$m$ 是固定值，无需学习
2. **强外推能力**：在 1024 token 上训练，在 2048 token 上推理，性能与直接在 2048 上训练相当
3. **训练效率高**：无需在输入层添加位置编码，节省了加法计算

**为什么距离惩罚有效**：ALiBi 对模型施加了一个"关注附近 token"的归纳偏置（inductive bias）。距离越远的 token，注意力分数越低，促使模型优先利用局部上下文，而不是远处的 token。

---

## 3.5 旋转位置编码（RoPE）

### 3.5.1 基本思想

RoPE（Rotary Position Embedding，Su et al., 2021）是当前最流行的位置编码方案之一，被 LLaMA、Mistral、Qwen 等主流大语言模型广泛采用。

核心思想：**不直接将位置编码加到嵌入向量上，而是在计算注意力时，用旋转矩阵对 $Q$ 和 $K$ 进行变换，使得 $q_m^\top k_n$ 只依赖于相对位置 $m - n$。**

RoPE 的目标：找到一个函数 $f(x, pos)$，使得：

$$\langle f(q, m),\ f(k, n) \rangle = g(q, k, m-n)$$

即内积结果只依赖于相对位置差，而不是绝对位置。

### 3.5.2 二维情形推导

先从二维向量入手，推广到高维。

设 $q = (q_0, q_1)^\top$，$k = (k_0, k_1)^\top$，位置分别为 $m$ 和 $n$。定义旋转变换：

$$f(x, pos) = R(\theta \cdot pos) \cdot x = \begin{pmatrix} \cos(\theta \cdot pos) & -\sin(\theta \cdot pos) \\ \sin(\theta \cdot pos) & \cos(\theta \cdot pos) \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \end{pmatrix}$$

则内积为：

$$f(q,m)^\top f(k,n) = q^\top R(-\theta m)^\top R(\theta n) k = q^\top R(\theta(n-m)) k$$

完美！内积只依赖于相对位置 $n - m$。

### 3.5.3 高维推广

对于维度为 $d$ 的向量，将其分为 $d/2$ 组，每组2维，独立施加旋转：

$$R_{\Theta,m}^d = \begin{pmatrix} R(\theta_0 m) & & & \\ & R(\theta_1 m) & & \\ & & \ddots & \\ & & & R(\theta_{d/2-1} m) \end{pmatrix}$$

其中 $\theta_i = 10000^{-2i/d}$（与正弦编码的频率选择完全一致）。

**实现技巧**：不需要显式构建稀疏旋转矩阵，可以用复数乘法高效实现：

$$f(x, pos) = x \cdot e^{i \Theta pos}$$

等价于将向量 $x$ 拆成实部和虚部，逐元素乘以旋转因子。

### 3.5.4 RoPE 在 LLaMA 中的应用

LLaMA 系列的 RoPE 实现将旋转应用于注意力层的 Q 和 K，但不修改 V：

```
注意力分数 = softmax((RQ)(RK)^T / sqrt(d_k)) V
```

**RoPE vs 其他方案**：

| 特性 | RoPE | ALiBi | T5相对编码 | 正弦绝对 |
|:---|:---:|:---:|:---:|:---:|
| 额外参数 | 0 | 0 | 少量 | 0 |
| 外推能力 | 好（需YaRN等增强） | 很好 | 一般 | 一般 |
| 计算开销 | 低 | 极低 | 低 | 低 |
| FlashAttention 兼容 | 是 | 部分 | 否 | 是 |
| 主要使用者 | LLaMA/Qwen/Mistral | MPT/BLOOM | T5/Flan | 原始Transformer |

### 3.5.5 长度外推增强

原始 RoPE 在超出训练长度时也会退化。主要改进方案：

- **PI（Position Interpolation）**：将位置下采样，将长度 $L'$ 的序列映射到 $[0, L_{train}]$ 范围内，线性插值
- **YaRN（Yet another RoPE extensioN）**：对不同频率维度使用不同的缩放因子，低频维度插值，高频维度外推，效果优于简单插值
- **LongRoPE**：动态调整旋转基，支持超过100万 token 的上下文窗口

---

## 本章小结

### 各种位置编码方法对比

| 方法 | 类型 | 参数量 | 外推 | 典型模型 | 核心机制 |
|:---|:---:|:---:|:---:|:---:|:---|
| 正弦编码 | 绝对，固定 | 0 | 一般 | Transformer | 多频率 sin/cos 叠加 |
| 可学习绝对 | 绝对，可学习 | $n_{max} \times d$ | 差 | BERT, GPT-2 | nn.Embedding |
| Shaw相对 | 相对，可学习 | $2k \times d$ | 好 | Transformer-XL | K/V加相对向量 |
| T5相对偏置 | 相对，可学习 | $n_{bucket}$ | 好 | T5, Flan-T5 | 注意力分数加标量偏置 |
| ALiBi | 相对，固定 | 0 | 很好 | MPT, BLOOM | 线性距离惩罚 |
| RoPE | 相对，固定 | 0 | 好（需增强） | LLaMA, Qwen, Mistral | 旋转矩阵变换Q/K |

**选型建议**：
- 序列长度固定、资源充足 → 可学习绝对编码（简单、效果好）
- 需要外推到更长序列 → ALiBi 或 RoPE+YaRN
- 编码器-解码器架构 → T5相对位置偏置
- 追求最新最好效果的LLM → RoPE

---

## 代码实战

### 环境准备

```python
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'SimHei'  # 中文显示
matplotlib.rcParams['axes.unicode_minus'] = False
```

---

### 实现一：正弦位置编码

```python
class SinusoidalPositionalEncoding(nn.Module):
    """
    经典正弦位置编码（Attention Is All You Need, 2017）

    Args:
        d_model: 嵌入维度
        max_len: 支持的最大序列长度
        dropout: dropout 概率
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 构建位置编码矩阵 PE: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # pos: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: (d_model/2,)
        # 等价于 1 / (10000 ^ (2i / d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)   # (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)   # (max_len, d_model/2)

        # 注册为 buffer（不作为模型参数，但随模型保存和加载）
        # 增加 batch 维度: (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        # 只取前 seq_len 个位置的编码，并广播加到 batch 上
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---- 测试 ----
d_model = 64
max_len = 100

pe_layer = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)

# 查看编码矩阵形状
pe_matrix = pe_layer.pe.squeeze(0)  # (max_len, d_model)
print(f"PE矩阵形状: {pe_matrix.shape}")  # torch.Size([100, 64])

# 验证相对位置线性变换性质
pos_0 = pe_matrix[0]    # 位置0的编码
pos_k = pe_matrix[5]    # 位置5的编码

# 对第 i=0 对维度，验证旋转矩阵关系
theta = 1.0 / (10000 ** (0 / d_model))  # i=0 对应的频率
k = 5
M_k = torch.tensor([
    [math.cos(theta * k),  math.sin(theta * k)],
    [-math.sin(theta * k), math.cos(theta * k)]
])
transformed = M_k @ pos_0[:2]
print(f"旋转变换结果: {transformed}")
print(f"实际PE[5]前2维: {pos_k[:2]}")
print(f"误差: {(transformed - pos_k[:2]).abs().max():.2e}")  # 应接近0
```

---

### 实现二：可学习位置编码

```python
class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码（BERT风格）

    Args:
        d_model: 嵌入维度
        max_len: 最大序列长度（超过此长度无法处理）
        dropout: dropout 概率
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 可学习参数矩阵
        self.pos_embedding = nn.Embedding(max_len, d_model)
        # 使用截断正态初始化（与BERT一致）
        nn.init.trunc_normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)  # (seq_len,)
        pos_enc = self.pos_embedding(positions)              # (seq_len, d_model)
        return self.dropout(x + pos_enc)

    def get_embedding_matrix(self) -> torch.Tensor:
        """返回位置嵌入矩阵，用于可视化"""
        return self.pos_embedding.weight.detach()


# ---- 测试 ----
learnable_pe = LearnablePositionalEncoding(d_model=64, max_len=512, dropout=0.0)

# 显示参数量
total_params = sum(p.numel() for p in learnable_pe.parameters())
print(f"可学习位置编码参数量: {total_params}")  # 64 * 512 = 32768

# 模拟前向传播
batch_size, seq_len = 4, 128
dummy_input = torch.randn(batch_size, seq_len, 64)
output = learnable_pe(dummy_input)
print(f"输入形状: {dummy_input.shape}, 输出形状: {output.shape}")
```

---

### 实现三：RoPE

```python
def precompute_freqs_cis(d: int, seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """
    预计算旋转频率的复数表示 e^{i * theta_j * pos}

    Args:
        d: 头的维度（必须为偶数）
        seq_len: 最大序列长度
        base: 基数（默认10000）
    Returns:
        freqs_cis: (seq_len, d//2) 复数张量
    """
    # theta_j = base^{-2j/d}, j = 0, 1, ..., d/2-1
    freqs = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))  # (d/2,)
    # 位置序列
    t = torch.arange(seq_len, dtype=torch.float)  # (seq_len,)
    # 外积：每个位置 * 每个频率
    freqs = torch.outer(t, freqs)  # (seq_len, d/2)
    # 转为复数 e^{i*freq}
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # (seq_len, d/2)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    将旋转位置编码应用到向量 x 上

    Args:
        x: (..., seq_len, d) 实数张量（Q 或 K）
        freqs_cis: (seq_len, d//2) 复数张量
    Returns:
        旋转后的向量，形状与 x 相同
    """
    # 将实数向量视为复数：(x0, x1) -> x0 + i*x1
    # reshape: (..., seq_len, d) -> (..., seq_len, d/2, 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs_cis: (seq_len, d/2) -> (1, ..., 1, seq_len, d/2) 广播用
    freqs_cis = freqs_cis.view(*([1] * (x_complex.dim() - 2)), *freqs_cis.shape)
    # 复数乘法 = 旋转操作
    x_rotated = x_complex * freqs_cis
    # 转回实数
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.type_as(x)


class RoPEMultiHeadAttention(nn.Module):
    """
    集成 RoPE 的多头注意力（简化版，用于演示）

    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        max_seq_len: 最大序列长度
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # 预计算旋转频率
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer('freqs_cis', freqs_cis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        B, L, D = x.shape
        H, head_dim = self.num_heads, self.head_dim

        # 线性变换
        Q = self.wq(x).view(B, L, H, head_dim)  # (B, L, H, head_dim)
        K = self.wk(x).view(B, L, H, head_dim)
        V = self.wv(x).view(B, L, H, head_dim)

        # 应用旋转位置编码（只对 Q 和 K，不对 V）
        freqs = self.freqs_cis[:L]  # 取前 L 个位置
        Q = apply_rotary_emb(Q, freqs)
        K = apply_rotary_emb(K, freqs)

        # 转置为 (B, H, L, head_dim) 进行注意力计算
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 缩放点积注意力
        scale = math.sqrt(head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, L, L)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, L, head_dim)

        # 合并多头输出
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(out)


# ---- 测试 RoPE ----
d_model, num_heads, seq_len, batch_size = 64, 8, 32, 2
rope_attn = RoPEMultiHeadAttention(d_model=d_model, num_heads=num_heads)
dummy = torch.randn(batch_size, seq_len, d_model)
output = rope_attn(dummy)
print(f"RoPE注意力输出形状: {output.shape}")  # (2, 32, 64)

# 验证相对位置不变性（简单检验）
# 如果将所有位置整体平移，注意力分数应不变
# 这里用旋转频率验证：pos=5 vs pos=3 的内积，应等于 pos=10 vs pos=8 的内积
freqs = precompute_freqs_cis(d=8, seq_len=20)
q = torch.randn(1, 1, 1, 8)
k = torch.randn(1, 1, 1, 8)
q_3 = apply_rotary_emb(q.expand(1, 20, 1, 8), freqs)[:, 3]  # 位置3
q_5 = apply_rotary_emb(q.expand(1, 20, 1, 8), freqs)[:, 5]  # 位置5
k_8 = apply_rotary_emb(k.expand(1, 20, 1, 8), freqs)[:, 8]  # 位置8
k_10 = apply_rotary_emb(k.expand(1, 20, 1, 8), freqs)[:, 10] # 位置10

dot_1 = (q_3 * k_5).sum() if False else (q_3 * k_8).sum()   # 相对距离=5
dot_2 = (q_5 * k_10).sum()                                   # 相对距离=5
print(f"位置(3,8)内积: {dot_1.item():.4f}, 位置(5,10)内积: {dot_2.item():.4f}")
print("注：两者应接近相等（验证相对位置不变性）")
```

---

### 实现四：位置编码可视化

```python
def visualize_positional_encodings():
    """可视化并对比三种位置编码"""

    d_model = 64
    seq_len = 50

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ---- 图1：正弦位置编码热力图 ----
    pe_layer = SinusoidalPositionalEncoding(d_model=d_model, max_len=seq_len, dropout=0.0)
    pe_matrix = pe_layer.pe.squeeze(0)[:seq_len].numpy()  # (seq_len, d_model)

    im1 = axes[0].imshow(pe_matrix, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[0].set_title('正弦位置编码', fontsize=14)
    axes[0].set_xlabel('维度索引 (d)', fontsize=12)
    axes[0].set_ylabel('位置 (pos)', fontsize=12)
    plt.colorbar(im1, ax=axes[0])

    # ---- 图2：不同维度的正弦波形 ----
    positions = np.arange(seq_len)
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    dim_indices = [0, 4, 16, 32]  # 选取4个维度观察

    for idx, dim in enumerate(dim_indices):
        if dim < d_model:
            axes[1].plot(
                positions,
                pe_matrix[:, dim],
                color=colors[idx],
                label=f'dim={dim}',
                linewidth=2
            )
    axes[1].set_title('各维度的正弦波形', fontsize=14)
    axes[1].set_xlabel('位置 (pos)', fontsize=12)
    axes[1].set_ylabel('编码值', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ---- 图3：位置编码的余弦相似度矩阵 ----
    pe_tensor = torch.tensor(pe_matrix)
    # 归一化
    pe_norm = pe_tensor / (pe_tensor.norm(dim=-1, keepdim=True) + 1e-8)
    # 计算所有位置对之间的余弦相似度
    similarity = (pe_norm @ pe_norm.T).numpy()  # (seq_len, seq_len)

    im3 = axes[2].imshow(similarity, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[2].set_title('位置编码余弦相似度矩阵\n（对角线为自相似=1）', fontsize=14)
    axes[2].set_xlabel('位置 j', fontsize=12)
    axes[2].set_ylabel('位置 i', fontsize=12)
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle('正弦位置编码可视化', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('sinusoidal_pe_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图像已保存至 sinusoidal_pe_visualization.png")


def visualize_rope_patterns():
    """可视化 RoPE 旋转频率"""

    d_head = 64
    seq_len = 128
    freqs_cis = precompute_freqs_cis(d=d_head, seq_len=seq_len)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1：各维度的旋转角度（实部和虚部）
    freqs_real = freqs_cis.real.numpy()  # (seq_len, d/2)
    freqs_imag = freqs_cis.imag.numpy()

    positions = np.arange(seq_len)
    colors = plt.cm.plasma(np.linspace(0, 1, 4))
    for i, dim in enumerate([0, 8, 16, 31]):
        axes[0].plot(positions, freqs_real[:, dim], color=colors[i],
                     label=f'维度对{dim}', linewidth=1.5)
    axes[0].set_title('RoPE旋转因子（实部）', fontsize=13)
    axes[0].set_xlabel('位置', fontsize=12)
    axes[0].set_ylabel('cos(θ·pos)', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 图2：相对位置内积热力图（随机生成一个查询向量）
    torch.manual_seed(42)
    q = torch.randn(d_head)
    # 对每个位置旋转 q
    q_all = apply_rotary_emb(
        q.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, seq_len, 1, d_head),
        freqs_cis
    ).squeeze()  # (seq_len, d_head)

    # 计算所有位置对的内积
    dots = (q_all @ q_all.T).detach().numpy()  # (seq_len, seq_len)

    im = axes[1].imshow(dots, cmap='RdYlBu', aspect='auto')
    axes[1].set_title('RoPE内积矩阵\n（对角线：自内积；偏离对角线：相对距离增大）', fontsize=12)
    axes[1].set_xlabel('位置 j', fontsize=12)
    axes[1].set_ylabel('位置 i', fontsize=12)
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig('rope_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("RoPE可视化已保存至 rope_visualization.png")


# 运行可视化
visualize_positional_encodings()
visualize_rope_patterns()
```

---

### 完整对比实验

```python
def compare_position_encodings():
    """
    对比三种位置编码在简单序列分类任务上的性能
    任务：判断序列是否按升序排列（依赖顺序信息）
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)

    # 数据生成
    def generate_data(n_samples: int, seq_len: int = 10, vocab_size: int = 20):
        """生成随机序列，标签=1表示序列单调不降"""
        seqs = torch.randint(1, vocab_size, (n_samples, seq_len))
        # 标签：序列是否为升序（允许相等）
        labels = (seqs.diff(dim=1) >= 0).all(dim=1).float()
        return seqs, labels

    n_train, n_val = 2000, 500
    seq_len, vocab_size, d_model = 10, 20, 32

    X_train, y_train = generate_data(n_train, seq_len, vocab_size)
    X_val, y_val = generate_data(n_val, seq_len, vocab_size)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    class SimpleTransformer(nn.Module):
        def __init__(self, pe_type: str, vocab_size: int, d_model: int, seq_len: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)

            if pe_type == 'sinusoidal':
                self.pe = SinusoidalPositionalEncoding(d_model, seq_len, dropout=0.0)
            elif pe_type == 'learnable':
                self.pe = LearnablePositionalEncoding(d_model, seq_len, dropout=0.0)
            elif pe_type == 'none':
                self.pe = None

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=64,
                dropout=0.0, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.classifier = nn.Linear(d_model, 1)

        def forward(self, x):
            emb = self.embedding(x)  # (B, L, D)
            if self.pe is not None:
                emb = self.pe(emb)
            out = self.transformer(emb)
            pooled = out.mean(dim=1)  # 全局平均池化
            return self.classifier(pooled).squeeze(-1)

    def train_and_eval(pe_type: str) -> tuple:
        model = SimpleTransformer(pe_type, vocab_size, d_model, seq_len)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(20):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = (model(X_batch) > 0).float()
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
        return correct / total

    results = {}
    for pe_type in ['none', 'sinusoidal', 'learnable']:
        acc = train_and_eval(pe_type)
        results[pe_type] = acc
        print(f"位置编码={pe_type:12s}  验证准确率: {acc:.4f}")

    print("\n结论：有位置编码的模型准确率应显著高于无位置编码的模型")
    return results


results = compare_position_encodings()
```

---

## 练习题

### 基础题

**练习 3-1（基础）**：手动计算正弦位置编码

给定 $d_{model} = 4$，计算位置 $pos = 0, 1, 2$ 的完整位置编码向量（精确到小数点后4位）。

要求：
1. 写出每个维度的 $\omega_i$ 值
2. 列出完整的 $PE$ 矩阵
3. 用代码验证你的手算结果

---

**练习 3-2（基础）**：可学习位置编码的参数量分析

BERT-base 使用 $d_{model} = 768$，最大序列长度 512。

1. 计算位置嵌入的参数量
2. BERT-base 的总参数量约为 1.1 亿，位置嵌入占比是多少？
3. 如果将最大长度扩展到 4096，参数量如何变化？这会带来哪些问题？

---

### 中级题

**练习 3-3（中级）**：验证 ALiBi 的外推能力

实现一个简单的 ALiBi 注意力层，并用以下实验验证其外推能力：

1. 在序列长度 $\leq 64$ 的数据上训练一个带 ALiBi 的单层 Transformer 分类器
2. 在序列长度 128 的数据上测试（不重新训练）
3. 对比：同样设置下，使用可学习绝对位置编码的模型在长度128上的表现
4. 分析实验结果，解释为什么 ALiBi 能外推

---

**练习 3-4（中级）**：实现 RoPE 并验证相对位置性质

1. 完整实现 `apply_rotary_emb` 函数（不允许直接复制本章代码，需要从数学推导出发重写）
2. 实验验证：随机生成两个向量 $q$ 和 $k$，证明 $\langle R_m q,\ R_n k \rangle$ 只依赖于 $m - n$（使用多组不同的 $m, n$ 进行数值验证）
3. 绘制 $d_{head} = 64$ 时，RoPE 内积随相对距离变化的曲线

---

### 提高题

**练习 3-5（提高）**：实现 YaRN 长度外推

YaRN 的核心思想是对 RoPE 的不同频率维度使用不同的缩放策略：
- 高频维度（$\omega_i > \omega_{high}$）：不缩放（保持原始频率）
- 低频维度（$\omega_i < \omega_{low}$）：线性缩放（扩大有效上下文）
- 中间维度：线性插值

具体公式参考 Peng et al. (2023)《YaRN: Efficient Context Window Extension of Large Language Models》。

任务：
1. 实现 `YaRNPositionalEncoding`，支持参数 `original_max_len`、`extended_max_len`、`alpha`、`beta`
2. 在 `seq_len=512` 上训练带 RoPE 的语言模型（可用字符级语言模型简化）
3. 不重新训练，对比普通 RoPE 和 YaRN-RoPE 在 `seq_len=1024` 上的 perplexity
4. 写一段 200 字以内的分析，解释 YaRN 为何优于简单线性插值

---

## 练习答案

### 答案 3-1

**第一步：计算 $\omega_i$**

$d_{model} = 4$，有 $d/2 = 2$ 个频率：

$$\omega_0 = \frac{1}{10000^{0/4}} = \frac{1}{10000^0} = 1.0000$$

$$\omega_1 = \frac{1}{10000^{2/4}} = \frac{1}{10000^{0.5}} = \frac{1}{100} = 0.0100$$

**第二步：计算各位置的编码**

| pos | $\sin(\omega_0 \cdot pos)$ | $\cos(\omega_0 \cdot pos)$ | $\sin(\omega_1 \cdot pos)$ | $\cos(\omega_1 \cdot pos)$ |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 0.8415 | 0.5403 | 0.0100 | 0.9999 |
| 2 | 0.9093 | -0.4161 | 0.0200 | 0.9998 |

**验证代码**：

```python
import torch
import math

d_model = 4
positions = torch.tensor([0, 1, 2], dtype=torch.float)

pe = torch.zeros(3, d_model)
for i in range(d_model // 2):
    omega = 1.0 / (10000 ** (2 * i / d_model))
    pe[:, 2 * i]     = torch.sin(omega * positions)
    pe[:, 2 * i + 1] = torch.cos(omega * positions)

print("PE矩阵：")
print(pe)
# tensor([[ 0.0000,  1.0000,  0.0000,  1.0000],
#         [ 0.8415,  0.5403,  0.0100,  0.9999],
#         [ 0.9093, -0.4161,  0.0200,  0.9998]])
```

---

### 答案 3-2

**第一问：参数量**

$$\text{参数量} = 512 \times 768 = 393{,}216 \approx 39.3 \text{万}$$

**第二问：占比**

$$\text{占比} = \frac{393216}{110{,}000{,}000} \approx 0.36\%$$

位置嵌入仅占 BERT-base 总参数的约 0.36%，说明其对总体复杂度影响极小。

**第三问：扩展到 4096**

$$\text{新参数量} = 4096 \times 768 = 3{,}145{,}728 \approx 314 \text{万}$$

增加了 8 倍（$4096/512 = 8$）。主要问题：
1. **无法泛化**：对 $pos > 512$ 的位置，嵌入从未被训练，相当于随机初始化，会引入噪声
2. **存储增加**：314万参数虽然在现代模型中不算多，但若直接 fine-tune 则需要大量4096长度的数据
3. **根本缺陷**：可学习位置编码不能"插值"到新位置，而正弦编码或 RoPE 可以

---

### 答案 3-3

**实现 ALiBi 注意力**：

```python
class ALiBiAttention(nn.Module):
    """ALiBi: Attention with Linear Biases"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # 预计算每个头的斜率 m_h
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)  # (num_heads,)

    @staticmethod
    def _get_slopes(n: int) -> torch.Tensor:
        """计算 ALiBi 的斜率（几何级数）"""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]

        if math.log2(n).is_integer():
            slopes = get_slopes_power_of_2(n)
        else:
            # 不是2的幂次时的处理
            closest_power = 2 ** math.floor(math.log2(n))
            slopes = get_slopes_power_of_2(closest_power)
            extra = get_slopes_power_of_2(2 * closest_power)[0::2][:n - closest_power]
            slopes = slopes + extra
        return torch.tensor(slopes, dtype=torch.float)

    def _build_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """构建 ALiBi 偏置矩阵: (num_heads, seq_len, seq_len)"""
        # 相对位置矩阵：bias[i,j] = -(j - i)（只计算因果情况下的距离）
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)
        # ALiBi 使用 -|distance| 作为偏置
        alibi = -distance.abs().float()  # (L, L)
        # 每个头乘以对应斜率
        alibi = self.slopes.view(-1, 1, 1) * alibi.unsqueeze(0)  # (H, L, L)
        return alibi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, d_k = self.num_heads, self.d_k

        Q = self.wq(x).view(B, L, H, d_k).transpose(1, 2)  # (B, H, L, d_k)
        K = self.wk(x).view(B, L, H, d_k).transpose(1, 2)
        V = self.wv(x).view(B, L, H, d_k).transpose(1, 2)

        # 注意力分数 + ALiBi 偏置
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        alibi_bias = self._build_alibi_bias(L, x.device)  # (H, L, L)
        scores = scores + alibi_bias.unsqueeze(0)          # (B, H, L, L)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(out)
```

**外推实验结论**：

ALiBi 在 64 长度上训练、128 长度上推理，准确率下降通常 <5%，因为：
- ALiBi 的偏置是基于距离的**数学公式**，天然支持任意长度
- 斜率 $m$ 使远距离 token 自动受到更大的注意力惩罚，这种"局部优先"的归纳偏置与真实语言的局部依赖性吻合

可学习绝对编码在超长序列上通常准确率骤降 >20%，因为它从未见过 $pos > 64$ 的位置嵌入。

---

### 答案 3-4

**核心推导**（从零推导 `apply_rotary_emb`）：

对第 $j$ 对维度 $(x_{2j}, x_{2j+1})$，旋转变换定义为：

$$\begin{pmatrix} x'_{2j} \\ x'_{2j+1} \end{pmatrix} = \begin{pmatrix} \cos\theta_j \cdot pos & -\sin\theta_j \cdot pos \\ \sin\theta_j \cdot pos & \cos\theta_j \cdot pos \end{pmatrix} \begin{pmatrix} x_{2j} \\ x_{2j+1} \end{pmatrix}$$

即：

$$x'_{2j} = x_{2j} \cos(\theta_j \cdot pos) - x_{2j+1} \sin(\theta_j \cdot pos)$$
$$x'_{2j+1} = x_{2j} \sin(\theta_j \cdot pos) + x_{2j+1} \cos(\theta_j \cdot pos)$$

等价于复数乘法：$(x_{2j} + i \cdot x_{2j+1}) \cdot e^{i\theta_j \cdot pos}$

**从零实现**：

```python
def apply_rotary_emb_from_scratch(
    x: torch.Tensor,      # (..., seq_len, head_dim)
    positions: torch.Tensor,  # (seq_len,)
    base: float = 10000.0
) -> torch.Tensor:
    head_dim = x.shape[-1]
    assert head_dim % 2 == 0
    d_half = head_dim // 2

    # 计算频率 theta_j = base^{-2j/d}
    j = torch.arange(d_half, dtype=torch.float, device=x.device)
    theta = 1.0 / (base ** (2 * j / head_dim))  # (d_half,)

    # 计算每个位置每个频率的旋转角度
    angles = positions.float().unsqueeze(-1) * theta  # (seq_len, d_half)
    cos_val = angles.cos()  # (seq_len, d_half)
    sin_val = angles.sin()  # (seq_len, d_half)

    # 将 x 分为偶数维和奇数维
    x_even = x[..., 0::2]  # (..., seq_len, d_half)
    x_odd  = x[..., 1::2]  # (..., seq_len, d_half)

    # 旋转
    out_even = x_even * cos_val - x_odd * sin_val
    out_odd  = x_even * sin_val + x_odd * cos_val

    # 交织合并
    out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
    return out.type_as(x)


# 验证相对位置不变性
torch.manual_seed(0)
d = 16
q = torch.randn(d)
k = torch.randn(d)

print("验证 <R_m q, R_n k> 只依赖于 (m-n)：")
for m, n in [(1, 3), (5, 7), (10, 12), (0, 2)]:
    q_rot = apply_rotary_emb_from_scratch(
        q.unsqueeze(0).unsqueeze(0),
        torch.tensor([m])
    ).squeeze()
    k_rot = apply_rotary_emb_from_scratch(
        k.unsqueeze(0).unsqueeze(0),
        torch.tensor([n])
    ).squeeze()
    dot = (q_rot * k_rot).sum().item()
    print(f"  m={m:2d}, n={n:2d}, m-n={m-n:3d}: 内积={dot:.6f}")
# 所有 m-n=-2 的行，内积应相同（或非常接近）
```

---

### 答案 3-5

**YaRN 关键代码框架**：

```python
def get_yarn_scaling_factor(
    dim_idx: int,
    d_model: int,
    original_max_len: int,
    extended_max_len: int,
    alpha: float = 1.0,
    beta: float = 32.0,
    base: float = 10000.0
) -> float:
    """
    计算每个频率维度的 YaRN 缩放因子

    alpha: 控制高频维度的截止（默认1，即完整周期内不插值）
    beta:  控制低频维度的截止（默认32，对应T5基础的设置）
    """
    scale = extended_max_len / original_max_len

    # 该维度对应的频率
    omega = base ** (-2 * dim_idx / d_model)
    # 该频率在原始长度下的周期（以token数计）
    period = 2 * math.pi / omega

    if period < original_max_len / beta:
        # 高频：周期很短，位置变化足够稳定，不缩放
        return 1.0
    elif period > original_max_len * alpha:
        # 低频：周期很长，需要线性插值才能感知新位置
        return scale
    else:
        # 中间频率：线性插值（平滑过渡）
        ramp = (period / (original_max_len / beta) - 1) / (alpha * beta - 1)
        ramp = max(0.0, min(1.0, ramp))
        return 1.0 + (scale - 1.0) * ramp


def precompute_freqs_cis_yarn(
    d: int,
    seq_len: int,
    original_max_len: int,
    extended_max_len: int,
    alpha: float = 1.0,
    beta: float = 32.0,
    base: float = 10000.0
) -> torch.Tensor:
    """YaRN 版的旋转频率预计算"""
    # 计算每个维度的缩放因子
    scaling_factors = [
        get_yarn_scaling_factor(i, d, original_max_len, extended_max_len, alpha, beta, base)
        for i in range(d // 2)
    ]
    scaling = torch.tensor(scaling_factors, dtype=torch.float)

    # 原始频率
    freqs = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
    # 缩放后的频率（低频维度频率降低 = 周期拉长 = 支持更长序列）
    freqs_scaled = freqs / scaling

    t = torch.arange(seq_len, dtype=torch.float)
    freqs_mat = torch.outer(t, freqs_scaled)
    return torch.polar(torch.ones_like(freqs_mat), freqs_mat)
```

**为何优于线性插值**：

简单线性插值对所有维度统一缩放，会破坏高频维度原本已经足够稳定的局部位置表示（相邻 token 的区分度下降）。YaRN 的关键洞察是：**高频维度已经"收敛"，不需要缩放；真正需要扩展的是低频维度**，因为低频维度才是感知远距离关系的通道。通过差异化缩放，YaRN 在保留局部精度的同时，大幅增强了全局上下文感知能力，因此 perplexity 比线性插值低 15-30%（具体数字因任务和长度而异）。

---

> **下一章预告**：[第4章：多头注意力](04-multi-head-attention.md) —— 我们将把位置编码融入完整的多头注意力机制，并实现带 RoPE 的 Transformer 编码器块。
