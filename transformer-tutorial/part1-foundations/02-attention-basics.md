# 第2章：注意力机制原理

> **前置知识**：第1章 Transformer概览、基础线性代数（矩阵乘法、softmax函数）
>
> **本章目标**：从直觉出发，逐步推导注意力机制的数学原理，并通过代码实现加深理解

---

## 学习目标

完成本章学习后，你将能够：

1. **理解注意力机制的直觉和动机**：从人类注意力的类比出发，理解为什么序列模型需要注意力
2. **掌握点积注意力的计算过程**：清楚 Query、Key、Value 三者的关系及完整矩阵运算步骤
3. **理解缩放点积注意力的必要性**：通过方差分析推导缩放因子 $\sqrt{d_k}$ 的来源
4. **实现和可视化注意力权重**：用 PyTorch 代码实现注意力并用热力图展示
5. **了解注意力在序列到序列模型中的应用**：对比 Bahdanau 和 Luong 注意力的异同

---

## 2.1 注意力的直觉

### 2.1.1 人类注意力的类比

当你阅读这句话时：

> "银行宣布降低贷款**利率**，储户对此表示担忧"

你的大脑在处理"利率"这个词时，会自动将注意力集中在"银行"、"贷款"这些词上，而不是"储户"、"担忧"。这就是**选择性注意力**——在信息海洋中，只关注当前任务最相关的部分。

机器学习中的注意力机制正是对这一过程的模拟。

```
输入序列：  银行  宣布  降低  贷款  利率  储户  对此  表示  担忧
              ↓    ↓    ↓    ↓    ↑    ↓    ↓    ↓    ↓
注意力权重： 0.3  0.05 0.1  0.4  ●   0.05 0.02 0.02 0.06
             （处理"利率"时，"银行"和"贷款"获得最高权重）
```

### 2.1.2 加权求和的思想

注意力机制的核心操作是**加权求和**。给定一组信息，我们根据其与当前任务的相关程度分配权重，再将所有信息按权重合并。

设有 $n$ 个信息向量 $v_1, v_2, \ldots, v_n$，以及对应的注意力权重 $\alpha_1, \alpha_2, \ldots, \alpha_n$（满足 $\sum_i \alpha_i = 1$），则注意力输出为：

$$\text{output} = \sum_{i=1}^{n} \alpha_i v_i$$

这本质上是对信息的**软选择**（soft selection）：不像 argmax 那样只选一个，而是给每个元素一个连续的权重，使整个过程可微分。

### 2.1.3 注意力解决了什么问题

在注意力机制出现之前，序列到序列（Seq2Seq）模型依赖 RNN 将整个输入序列压缩成一个固定长度的上下文向量：

```
输入序列 → [RNN编码器] → 上下文向量（固定维度）→ [RNN解码器] → 输出序列
              h₁h₂h₃h₄           c                    y₁y₂y₃
```

**固定上下文向量的瓶颈**：

| 问题 | 表现 |
|------|------|
| 信息压缩损失 | 长序列后期的信息会覆盖早期信息 |
| 无法动态关注 | 生成每个输出词时使用相同的上下文 |
| 长程依赖退化 | 序列越长，梯度消失越严重 |

注意力机制的引入允许解码器在生成每个词时，**动态查阅**编码器的所有隐状态，按相关性加权使用，彻底解除了固定向量的瓶颈。

---

## 2.2 点积注意力

### 2.2.1 Query、Key、Value 的概念

注意力机制借用了**信息检索**（Information Retrieval）的类比：

- **Query（查询 $Q$）**：当前的查询需求，"我想要什么信息？"
- **Key（键 $K$）**：数据库中每条记录的索引，"我能提供什么？"
- **Value（值 $V$）**：数据库中每条记录的实际内容，"我的具体内容是什么？"

```
信息检索类比：
  搜索关键词 → Query
  网页标题   → Key
  网页内容   → Value

过程：
  Query 与所有 Key 计算相似度
       ↓
  归一化为概率分布（softmax）
       ↓
  按概率加权 Value 求和
       ↓
  返回与 Query 最相关的信息
```

在神经网络中，$Q$、$K$、$V$ 都是通过线性变换从原始输入得到的矩阵：

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中 $X \in \mathbb{R}^{n \times d_{model}}$ 是输入序列（$n$ 个词，每词 $d_{model}$ 维），$W^Q, W^K, W^V$ 是可学习的权重矩阵。

### 2.2.2 点积计算相似度

两个向量的**点积**（dot product）可以衡量它们的相似程度：若两个向量方向相近，点积大；若正交，点积为零；若反向，点积为负。

对于 Query 向量 $q \in \mathbb{R}^{d_k}$ 和一组 Key 向量 $k_1, k_2, \ldots, k_n \in \mathbb{R}^{d_k}$，相似度分数为：

$$\text{score}(q, k_i) = q \cdot k_i = \sum_{j=1}^{d_k} q_j \cdot k_{ij}$$

矩阵形式（同时计算所有 Query 对所有 Key 的得分）：

$$\text{Scores} = QK^T \in \mathbb{R}^{n_q \times n_k}$$

其中 $n_q$ 是 Query 的数量，$n_k$ 是 Key 的数量。

### 2.2.3 基础点积注意力公式

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}(QK^T)V}$$

对每一行（每个 Query）独立应用 softmax，得到归一化的注意力权重矩阵 $A \in \mathbb{R}^{n_q \times n_k}$，再乘以 $V$。

### 2.2.4 详细的矩阵运算步骤

我们通过一个具体例子来拆解计算过程。

**设定**：序列长度 $n = 3$，维度 $d_k = 4$

```
输入（已投影为 Q、K、V）：

Q = [[1, 0, 1, 0],    ← Query₁
     [0, 1, 0, 1],    ← Query₂
     [1, 1, 0, 0]]    ← Query₃

K = [[1, 0, 1, 0],    ← Key₁
     [0, 1, 0, 1],    ← Key₂
     [1, 1, 0, 0]]    ← Key₃

V = [[1, 2, 3, 4],    ← Value₁
     [5, 6, 7, 8],    ← Value₂
     [9,10,11,12]]    ← Value₃
```

**第一步：计算得分矩阵** $S = QK^T$

$$S = QK^T = \begin{bmatrix} 1&0&1&0 \\ 0&1&0&1 \\ 1&1&0&0 \end{bmatrix} \begin{bmatrix} 1&0&1 \\ 0&1&1 \\ 1&0&0 \\ 0&1&0 \end{bmatrix} = \begin{bmatrix} 2&0&1 \\ 0&2&1 \\ 1&1&1 \end{bmatrix}$$

其中 $S_{ij}$ 表示 $\text{Query}_i$ 与 $\text{Key}_j$ 的相似度。

**第二步：对每行应用 softmax**

对第一行 $[2, 0, 1]$：

$$\text{softmax}([2, 0, 1]) = \frac{[e^2, e^0, e^1]}{e^2 + e^0 + e^1} = \frac{[7.389, 1.000, 2.718]}{11.107} \approx [0.665, 0.090, 0.245]$$

$$A = \text{softmax}(S) \approx \begin{bmatrix} 0.665 & 0.090 & 0.245 \\ 0.090 & 0.665 & 0.245 \\ 0.333 & 0.333 & 0.333 \end{bmatrix}$$

**第三步：加权求和 Value**

$$\text{Output} = AV = \begin{bmatrix} 0.665&0.090&0.245 \\ 0.090&0.665&0.245 \\ 0.333&0.333&0.333 \end{bmatrix} \begin{bmatrix} 1&2&3&4 \\ 5&6&7&8 \\ 9&10&11&12 \end{bmatrix}$$

$$\text{Output}_1 = 0.665 \times [1,2,3,4] + 0.090 \times [5,6,7,8] + 0.245 \times [9,10,11,12]$$
$$\approx [0.665, 1.33, 1.995, 2.66] + [0.45, 0.54, 0.63, 0.72] + [2.205, 2.45, 2.695, 2.94]$$
$$\approx [3.32, 4.32, 5.32, 6.32]$$

每个输出向量都是 Value 向量的加权组合，权重由 Query-Key 相似度决定。

---

## 2.3 缩放点积注意力

### 2.3.1 为什么需要缩放

当维度 $d_k$ 较大时，点积的量级会增大，导致 softmax 进入梯度极小的饱和区域。

**直觉理解**：假设 $d_k = 64$，Query 和 Key 的每个分量均从 $\mathcal{N}(0, 1)$ 独立采样，则点积：

$$q \cdot k = \sum_{j=1}^{d_k} q_j k_j$$

由于各分量独立，根据方差加法性：

$$\text{Var}(q \cdot k) = \sum_{j=1}^{d_k} \text{Var}(q_j k_j) = d_k \cdot \text{Var}(q_j) \cdot \text{Var}(k_j) = d_k$$

（利用 $\text{Var}(XY) = \text{Var}(X)\text{Var}(Y)$ 当 $X, Y$ 独立且均值为 0 时）

因此点积的标准差为 $\sqrt{d_k}$，当 $d_k = 64$ 时，标准差为 8，得分可能达到数十量级。

**softmax 的饱和问题**：

```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# 小量级：正常分布
scores_small = np.array([2.0, 1.0, 0.5])
print("小量级：", softmax(scores_small))
# 输出：[0.576, 0.212, 0.131]  ← 梯度正常

# 大量级（d_k=64 时常见）：极端分布
scores_large = np.array([16.0, 8.0, 4.0])
print("大量级：", softmax(scores_large))
# 输出：[0.9997, 0.0003, 0.0000]  ← 几乎 one-hot，梯度消失
```

当 softmax 输出接近 one-hot 时，除最大值外所有梯度趋近于零，模型无法有效学习多样化的注意力模式。

### 2.3.2 缩放因子 $\sqrt{d_k}$ 的推导

为了将点积的方差控制回 1，只需除以标准差 $\sqrt{d_k}$：

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{(\sqrt{d_k})^2} = \frac{d_k}{d_k} = 1$$

这样无论 $d_k$ 取何值，缩放后的得分始终具有单位方差，softmax 工作在稳定的梯度区域。

### 2.3.3 缩放点积注意力公式

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V}$$

这就是论文《Attention Is All You Need》中提出的**缩放点积注意力**（Scaled Dot-Product Attention），也是 Transformer 的基础构件。

**与基础点积注意力的对比**：

| 特性 | 点积注意力 | 缩放点积注意力 |
|------|-----------|--------------|
| 公式 | $\text{softmax}(QK^T)V$ | $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$ |
| 大 $d_k$ 时的梯度 | 容易消失 | 稳定 |
| 计算复杂度 | $O(n^2 d_k)$ | $O(n^2 d_k)$（相同） |
| 实现难度 | 简单 | 仅多一个除法 |

### 2.3.4 Mask 机制（可选）

在解码器中，为防止位置 $i$ 看到未来位置 $j > i$ 的信息，需在 softmax 前将对应位置的得分设为 $-\infty$：

$$\text{score}_{ij} = \begin{cases} \dfrac{q_i \cdot k_j}{\sqrt{d_k}} & \text{若 } j \leq i \\ -\infty & \text{若 } j > i \end{cases}$$

$e^{-\infty} = 0$，因此这些位置的注意力权重恰好为 0，实现了因果掩码（Causal Mask）。

---

## 2.4 注意力权重可视化

### 2.4.1 注意力矩阵的含义

注意力权重矩阵 $A \in \mathbb{R}^{n \times n}$ 是理解模型行为的重要窗口：

- **行** $i$：第 $i$ 个输出位置在生成时关注了哪些输入位置
- **列** $j$：输入位置 $j$ 被各输出位置关注的程度
- **$A_{ij}$**：输出位置 $i$ 对输入位置 $j$ 的注意力强度（0到1之间）

```
注意力矩阵示例（机器翻译 EN→ZH）：

         The  cat  sat  on  the  mat
"这只"  [0.7  0.2  0.0  0.0  0.0  0.1]  ← 主要关注 "The" 和 "cat"
"猫"    [0.1  0.8  0.0  0.0  0.0  0.1]  ← 强烈关注 "cat"
"坐"    [0.0  0.1  0.9  0.0  0.0  0.0]  ← 强烈关注 "sat"
"在"    [0.0  0.0  0.1  0.8  0.1  0.0]  ← 主要关注 "on"
"垫子上" [0.0  0.1  0.0  0.1  0.2  0.6]  ← 主要关注 "mat"
```

这种对角线模式（近似对齐）是翻译任务的典型特征。

### 2.4.2 热力图可视化

注意力权重最常用热力图（heatmap）来展示，颜色越深表示注意力越强：

```
颜色映射示例：
  0.0 → 白色/浅色
  0.5 → 中等颜色
  1.0 → 深色/黑色

典型工具：matplotlib.pyplot.imshow() 或 seaborn.heatmap()
```

### 2.4.3 不同任务的注意力模式

研究者发现不同任务、不同层的注意力头呈现出规律性的模式：

| 模式名称 | 视觉特征 | 语言学含义 |
|---------|---------|----------|
| **对角线模式** | 主对角线亮 | 词与自身对应（复制操作） |
| **局部窗口** | 对角线附近的带状区域 | 关注相邻词（局部语法） |
| **全局关注** | 某几行/列整体较亮 | 关键词（句号、[CLS]等） |
| **翻译对齐** | 近似对角线（含偏移） | 源语言与目标语言的词对齐 |
| **句法依存** | 稀疏非对角线 | 主谓宾等句法关系 |

不同头（head）往往专注于不同的模式，这也是多头注意力（Multi-Head Attention）的设计动机——我们将在第3章详细介绍。

---

## 2.5 序列到序列中的注意力

### 2.5.1 Encoder-Decoder 架构

经典的 Seq2Seq 模型分为两个部分：

```
Encoder（编码器）：
  输入序列 x₁, x₂, ..., xₙ
       ↓ RNN/LSTM
  隐状态 h₁, h₂, ..., hₙ

Decoder（解码器）：
  上下文 + 前一时刻输出
       ↓ RNN/LSTM
  输出序列 y₁, y₂, ..., yₘ
```

**加入注意力后**，解码器在生成 $y_t$ 时，不再只依赖一个固定的上下文向量，而是动态计算对编码器所有隐状态的注意力：

$$c_t = \sum_{j=1}^{n} \alpha_{tj} h_j$$

其中 $\alpha_{tj}$ 是解码器在时刻 $t$ 对编码器位置 $j$ 的注意力权重。

### 2.5.2 Bahdanau 注意力（加性注意力）

Bahdanau 等人（2014）提出了第一个用于机器翻译的注意力机制，又称**加性注意力**（Additive Attention）：

**相似度函数**：

$$e_{tj} = v_a^T \tanh(W_a s_{t-1} + U_a h_j)$$

其中：
- $s_{t-1}$：解码器前一时刻的隐状态（Query）
- $h_j$：编码器第 $j$ 个隐状态（Key）
- $W_a, U_a, v_a$：可学习的参数矩阵/向量

**权重归一化**：

$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{n} \exp(e_{tk})}$$

**上下文向量**：

$$c_t = \sum_{j=1}^{n} \alpha_{tj} h_j$$

Bahdanau 注意力的特点：
- 引入额外的神经网络层来计算相似度
- 参数量较多（$W_a$、$U_a$、$v_a$）
- 使用 $\tanh$ 非线性变换
- 被称为"加性"是因为 $W_a s + U_a h$ 是两者的加法组合

### 2.5.3 Luong 注意力（乘性注意力）

Luong 等人（2015）提出了更简洁的注意力机制，提供了多种相似度计算方式：

**dot（点积）**：

$$e_{tj} = s_t^T h_j$$

**general（一般化）**：

$$e_{tj} = s_t^T W_a h_j$$

**concat（拼接，与 Bahdanau 类似）**：

$$e_{tj} = v_a^T \tanh(W_a [s_t; h_j])$$

Luong 注意力还有一个关键区别：使用**当前时刻**的解码器状态 $s_t$ 计算注意力，而 Bahdanau 使用**前一时刻**的 $s_{t-1}$。

### 2.5.4 从加性注意力到乘性注意力

| 对比维度 | Bahdanau（加性） | Luong（乘性） | 缩放点积 |
|---------|----------------|-------------|---------|
| 提出年份 | 2014 | 2015 | 2017 |
| 相似度计算 | $v^T\tanh(Ws+Uh)$ | $s^TWh$ | $\frac{qk^T}{\sqrt{d_k}}$ |
| 参数量 | 多（$W_a, U_a, v_a$） | 中（$W_a$） | 无额外参数 |
| 计算效率 | 较低 | 中等 | 高（矩阵乘法并行） |
| 主要优势 | 首创，效果好 | 简洁，多种变体 | 可并行，适合Transformer |

**演化趋势**：从需要额外神经网络（加性）→ 直接矩阵乘法（乘性）→ 并行化缩放点积，参数更少，速度更快，是 Transformer 成功的关键。

---

## 本章小结

本章从人类注意力的直觉出发，系统介绍了注意力机制的数学原理和发展历程。

### 各种注意力机制对比

| 机制 | 公式 | 特点 | 应用场景 |
|------|------|------|---------|
| **加权求和**（基础） | $\sum_i \alpha_i v_i$ | 最简形式 | 概念理解 |
| **点积注意力** | $\text{softmax}(QK^T)V$ | 无缩放 | 维度较小时 |
| **缩放点积注意力** | $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$ | 方差稳定 | Transformer 标准 |
| **Bahdanau 注意力** | $v^T\tanh(W_a s + U_a h)$ | 加性，参数多 | 早期 Seq2Seq |
| **Luong dot 注意力** | $s^T h$ | 最简，无参数 | 轻量 Seq2Seq |
| **Luong general 注意力** | $s^T W_a h$ | 引入变换矩阵 | 中等 Seq2Seq |
| **带 Mask 的缩放点积** | 在 $-\infty$ 掩盖未来 | 自回归生成 | Transformer 解码器 |

### 关键公式回顾

$$\text{缩放点积注意力：}\quad \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $Q \in \mathbb{R}^{n_q \times d_k}$：Query 矩阵
- $K \in \mathbb{R}^{n_k \times d_k}$：Key 矩阵
- $V \in \mathbb{R}^{n_k \times d_v}$：Value 矩阵
- 输出 $\in \mathbb{R}^{n_q \times d_v}$

---

## 代码实战

### 完整实现：缩放点积注意力

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'  # macOS 中文支持
# Linux 用户改为: matplotlib.rcParams['font.family'] = 'WenQuanYi Micro Hei'

# ─────────────────────────────────────────────
# 1. 核心实现：缩放点积注意力
# ─────────────────────────────────────────────

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    缩放点积注意力。

    Args:
        query: (batch, n_q, d_k)
        key:   (batch, n_k, d_k)
        value: (batch, n_k, d_v)
        mask:  (batch, n_q, n_k) 或 None，True 表示被遮蔽的位置
        dropout: dropout 概率

    Returns:
        output:  (batch, n_q, d_v)  加权求和后的输出
        weights: (batch, n_q, n_k)  注意力权重（可用于可视化）
    """
    d_k = query.size(-1)

    # 第一步：计算得分矩阵 (batch, n_q, n_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

    # 第二步：应用 mask（解码器自注意力或 padding mask）
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # 第三步：softmax 归一化为注意力权重
    weights = F.softmax(scores, dim=-1)

    # 防止 NaN（当某行全为 -inf 时）
    weights = torch.nan_to_num(weights, nan=0.0)

    # 第四步：可选的 dropout
    if dropout > 0.0 and torch.is_grad_enabled():
        weights = F.dropout(weights, p=dropout)

    # 第五步：加权求和 Value
    output = torch.matmul(weights, value)

    return output, weights


# ─────────────────────────────────────────────
# 2. 基础测试：验证形状和数值
# ─────────────────────────────────────────────

def test_basic():
    """验证注意力机制的基本行为。"""
    torch.manual_seed(42)
    batch, n_q, n_k, d_k, d_v = 2, 5, 7, 64, 64

    Q = torch.randn(batch, n_q, d_k)
    K = torch.randn(batch, n_k, d_k)
    V = torch.randn(batch, n_k, d_v)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("=== 基础功能测试 ===")
    print(f"Q 形状: {Q.shape}")
    print(f"K 形状: {K.shape}")
    print(f"V 形状: {V.shape}")
    print(f"输出形状: {output.shape}")
    print(f"权重形状: {weights.shape}")
    print(f"权重行和（应为全1）: {weights.sum(dim=-1)[0]}")  # 每行和为1

    # 验证权重非负且和为1
    assert (weights >= 0).all(), "权重应为非负数"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, n_q), atol=1e-5), \
        "每行权重之和应为1"
    print("所有断言通过！\n")


# ─────────────────────────────────────────────
# 3. 手动计算示例（与 2.2.4 节对应）
# ─────────────────────────────────────────────

def manual_example():
    """复现章节中 2.2.4 的手动计算示例。"""
    Q = torch.tensor([[1., 0., 1., 0.],
                      [0., 1., 0., 1.],
                      [1., 1., 0., 0.]])
    K = torch.tensor([[1., 0., 1., 0.],
                      [0., 1., 0., 1.],
                      [1., 1., 0., 0.]])
    V = torch.tensor([[ 1.,  2.,  3.,  4.],
                      [ 5.,  6.,  7.,  8.],
                      [ 9., 10., 11., 12.]])

    # 无批次维度：添加 batch=1
    Q, K, V = Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)

    # 不缩放的基础点积注意力（复现章节公式）
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (1, 3, 3)
    weights_no_scale = F.softmax(scores, dim=-1)
    output_no_scale = torch.matmul(weights_no_scale, V)

    print("=== 手动计算示例（不缩放）===")
    print(f"得分矩阵 QKᵀ:\n{scores.squeeze(0).numpy()}")
    print(f"\n注意力权重（softmax后）:\n{weights_no_scale.squeeze(0).numpy().round(3)}")
    print(f"\n输出（加权Value）:\n{output_no_scale.squeeze(0).numpy().round(3)}")
    print()


# ─────────────────────────────────────────────
# 4. 缩放的必要性演示
# ─────────────────────────────────────────────

def scaling_necessity_demo():
    """展示不缩放时 softmax 饱和的问题。"""
    torch.manual_seed(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    dimensions = [4, 64, 512]

    for ax, d_k in zip(axes, dimensions):
        # 从 N(0,1) 采样 Q 和 K
        Q = torch.randn(1, 8, d_k)
        K = torch.randn(1, 8, d_k)

        # 不缩放的得分
        scores_raw = torch.matmul(Q, K.transpose(-2, -1)).squeeze(0)
        # 缩放后的得分
        scores_scaled = scores_raw / (d_k ** 0.5)

        weights_raw = F.softmax(scores_raw, dim=-1).detach().numpy()
        weights_scaled = F.softmax(scores_scaled, dim=-1).detach().numpy()

        # 绘制热力图对比
        combined = np.vstack([weights_raw[0:1], weights_scaled[0:1]])
        im = ax.imshow(combined, aspect='auto', cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'd_k = {d_k}\n上: 不缩放，下: 缩放后', fontsize=11)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['原始', '缩放'])
        ax.set_xlabel('Key 位置')
        plt.colorbar(im, ax=ax, fraction=0.04)

        # 打印熵（越高表示分布越均匀，越低表示越集中）
        entropy_raw = -(weights_raw[0] * np.log(weights_raw[0] + 1e-9)).sum()
        entropy_scaled = -(weights_scaled[0] * np.log(weights_scaled[0] + 1e-9)).sum()
        print(f"d_k={d_k:3d} | 原始熵={entropy_raw:.3f} | 缩放后熵={entropy_scaled:.3f} "
              f"| 理论最大熵={np.log(8):.3f}")

    plt.suptitle('缩放对注意力权重分布的影响（d_k 越大，不缩放问题越严重）', fontsize=12)
    plt.tight_layout()
    plt.savefig('scaling_effect.png', dpi=150, bbox_inches='tight')
    print("\n图像已保存至 scaling_effect.png\n")


# ─────────────────────────────────────────────
# 5. 注意力权重可视化
# ─────────────────────────────────────────────

def visualize_attention(
    weights: np.ndarray,
    x_labels: list,
    y_labels: list,
    title: str = "注意力权重",
    figsize: tuple = (8, 6)
):
    """
    可视化注意力权重热力图。

    Args:
        weights: (n_q, n_k) 的注意力权重矩阵
        x_labels: Key 侧的标签列表（横轴）
        y_labels: Query 侧的标签列表（纵轴）
        title: 图表标题
        figsize: 图表尺寸
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(weights, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='注意力权重')

    # 设置刻度和标签
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_xlabel("Key（被关注的位置）", fontsize=12)
    ax.set_ylabel("Query（当前位置）", fontsize=12)
    ax.set_title(title, fontsize=13, pad=15)

    # 在格子中显示数值
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            val = weights[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=color)

    plt.tight_layout()
    return fig


def attention_visualization_demo():
    """演示不同场景下的注意力模式。"""
    torch.manual_seed(42)

    # 场景1：翻译对齐（模拟英→中词对齐）
    en_tokens = ['The', 'cat', 'sat', 'on', 'mat']
    zh_tokens = ['这只猫', '坐', '在', '垫子上']

    # 模拟对齐矩阵（实际中从模型中提取）
    align_weights = np.array([
        [0.75, 0.10, 0.05, 0.05, 0.05],  # "这只猫" 关注 "The", "cat"
        [0.05, 0.10, 0.80, 0.03, 0.02],  # "坐" 关注 "sat"
        [0.05, 0.03, 0.05, 0.82, 0.05],  # "在" 关注 "on"
        [0.02, 0.05, 0.03, 0.05, 0.85],  # "垫子上" 关注 "mat"
    ])

    fig1 = visualize_attention(
        align_weights, en_tokens, zh_tokens,
        title="机器翻译注意力（英→中）\n近似对角线 = 词对齐",
        figsize=(8, 5)
    )
    fig1.savefig('attention_translation.png', dpi=150, bbox_inches='tight')
    print("翻译注意力热力图已保存至 attention_translation.png")

    # 场景2：自注意力（句子内部）
    tokens = ['猫', '追', '逐', '老鼠', '。']
    n = len(tokens)

    # 随机初始化并计算注意力
    Q = torch.randn(1, n, 32)
    K = torch.randn(1, n, 32)
    V = torch.randn(1, n, 32)
    _, self_attn_weights = scaled_dot_product_attention(Q, K, V)
    self_attn_np = self_attn_weights.squeeze(0).detach().numpy()

    fig2 = visualize_attention(
        self_attn_np, tokens, tokens,
        title="自注意力权重（随机初始化示例）\n训练后会出现语义模式",
        figsize=(7, 6)
    )
    fig2.savefig('attention_self.png', dpi=150, bbox_inches='tight')
    print("自注意力热力图已保存至 attention_self.png\n")

    plt.show()


# ─────────────────────────────────────────────
# 6. Causal Mask（因果掩码）演示
# ─────────────────────────────────────────────

def causal_mask_demo():
    """演示解码器自注意力中的因果掩码。"""
    tokens = ['<start>', '我', '爱', '学习']
    n = len(tokens)

    # 构建上三角掩码（掩蔽未来位置）
    # True 表示需要被掩蔽（-inf）
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    mask = mask.unsqueeze(0)  # (1, n, n)

    print("=== 因果掩码演示 ===")
    print(f"掩码矩阵（True=遮蔽未来）:")
    print(mask.squeeze(0).numpy().astype(int))
    print()

    Q = torch.randn(1, n, 32)
    K = torch.randn(1, n, 32)
    V = torch.randn(1, n, 32)

    _, weights_no_mask = scaled_dot_product_attention(Q, K, V)
    _, weights_with_mask = scaled_dot_product_attention(Q, K, V, mask=mask)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, weights, title in zip(
        axes,
        [weights_no_mask, weights_with_mask],
        ['无掩码（双向注意力）', '有因果掩码（自回归）']
    ):
        w = weights.squeeze(0).detach().numpy()
        im = ax.imshow(w, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tokens, fontsize=11)
        ax.set_yticklabels(tokens, fontsize=11)
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{w[i,j]:.2f}', ha='center', va='center',
                        fontsize=8, color='white' if w[i,j] > 0.5 else 'black')

    plt.suptitle('因果掩码效果对比', fontsize=13)
    plt.tight_layout()
    plt.savefig('causal_mask.png', dpi=150, bbox_inches='tight')
    print("因果掩码热力图已保存至 causal_mask.png\n")
    plt.show()


# ─────────────────────────────────────────────
# 7. 简单翻译模型示例（Bahdanau 注意力）
# ─────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    """Bahdanau（加性）注意力机制。"""

    def __init__(self, encoder_hidden: int, decoder_hidden: int, attn_dim: int):
        super().__init__()
        self.W_encoder = nn.Linear(encoder_hidden, attn_dim, bias=False)
        self.W_decoder = nn.Linear(decoder_hidden, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        decoder_state: torch.Tensor,  # (batch, decoder_hidden)
        encoder_outputs: torch.Tensor  # (batch, src_len, encoder_hidden)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context: (batch, encoder_hidden)  上下文向量
            weights: (batch, src_len)          注意力权重
        """
        # 将编码器和解码器状态投影到同一空间
        enc_proj = self.W_encoder(encoder_outputs)         # (batch, src_len, attn_dim)
        dec_proj = self.W_decoder(decoder_state).unsqueeze(1)  # (batch, 1, attn_dim)

        # 加性得分
        energy = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(-1)  # (batch, src_len)

        # softmax 归一化
        weights = F.softmax(energy, dim=-1)  # (batch, src_len)

        # 加权求和
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, enc_hidden)

        return context, weights


class SimpleSeq2SeqWithAttention(nn.Module):
    """
    带 Bahdanau 注意力的简单 Seq2Seq 模型（用于演示）。
    编码器和解码器均使用单层 GRU。
    """

    def __init__(self, src_vocab: int, tgt_vocab: int,
                 embed_dim: int = 64, hidden_dim: int = 128, attn_dim: int = 64):
        super().__init__()
        # 编码器
        self.src_embedding = nn.Embedding(src_vocab, embed_dim, padding_idx=0)
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # 解码器
        self.tgt_embedding = nn.Embedding(tgt_vocab, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, hidden_dim, attn_dim)
        self.decoder_gru = nn.GRUCell(embed_dim + hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, tgt_vocab)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """编码源序列。"""
        embedded = self.src_embedding(src)               # (batch, src_len, embed_dim)
        outputs, hidden = self.encoder_gru(embedded)     # (batch, src_len, hidden), (1, batch, hidden)
        return outputs, hidden.squeeze(0)

    def decode_step(
        self,
        tgt_token: torch.Tensor,          # (batch,)
        decoder_state: torch.Tensor,      # (batch, hidden)
        encoder_outputs: torch.Tensor     # (batch, src_len, hidden)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步解码。"""
        embedded = self.tgt_embedding(tgt_token)          # (batch, embed_dim)
        context, attn_weights = self.attention(decoder_state, encoder_outputs)
        gru_input = torch.cat([embedded, context], dim=-1)  # (batch, embed_dim + hidden)
        new_state = self.decoder_gru(gru_input, decoder_state)  # (batch, hidden)
        logits = self.output_proj(new_state)               # (batch, tgt_vocab)
        return logits, new_state, attn_weights

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        前向传播（训练时使用 teacher forcing）。

        Args:
            src: (batch, src_len) 源序列 token ids
            tgt: (batch, tgt_len) 目标序列 token ids（含 <start>）

        Returns:
            logits: (batch, tgt_len-1, tgt_vocab)
        """
        encoder_outputs, decoder_state = self.encode(src)

        all_logits = []
        for t in range(tgt.size(1) - 1):
            logits, decoder_state, _ = self.decode_step(
                tgt[:, t], decoder_state, encoder_outputs
            )
            all_logits.append(logits.unsqueeze(1))

        return torch.cat(all_logits, dim=1)  # (batch, tgt_len-1, tgt_vocab)


def demo_seq2seq():
    """演示 Seq2Seq + Bahdanau 注意力的前向传播。"""
    torch.manual_seed(42)

    src_vocab, tgt_vocab = 1000, 1200
    batch, src_len, tgt_len = 4, 10, 8

    model = SimpleSeq2SeqWithAttention(src_vocab, tgt_vocab)

    src = torch.randint(1, src_vocab, (batch, src_len))
    tgt = torch.randint(1, tgt_vocab, (batch, tgt_len))

    logits = model(src, tgt)

    print("=== Seq2Seq + Bahdanau 注意力演示 ===")
    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    print(f"输出 logits 形状: {logits.shape}  (batch, tgt_len-1, tgt_vocab)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")
    print()

    # 提取注意力权重用于可视化
    encoder_outputs, decoder_state = model.encode(src[:1])  # 取第一个样本
    attn_weights_list = []
    tgt_single = tgt[:1]
    for t in range(tgt_len - 1):
        _, decoder_state, attn_w = model.decode_step(
            tgt_single[:, t], decoder_state, encoder_outputs
        )
        attn_weights_list.append(attn_w.squeeze(0).detach().numpy())

    attn_matrix = np.stack(attn_weights_list)  # (tgt_len-1, src_len)

    fig = visualize_attention(
        attn_matrix,
        x_labels=[f'src{i}' for i in range(src_len)],
        y_labels=[f'tgt{i}' for i in range(tgt_len - 1)],
        title="Bahdanau 注意力权重（随机初始化）",
        figsize=(10, 5)
    )
    fig.savefig('bahdanau_attention.png', dpi=150, bbox_inches='tight')
    print("Bahdanau 注意力热力图已保存至 bahdanau_attention.png")
    plt.show()


# ─────────────────────────────────────────────
# 主程序入口
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print("第2章代码实战：注意力机制原理")
    print("=" * 50)
    print()

    test_basic()
    manual_example()
    scaling_necessity_demo()
    attention_visualization_demo()
    causal_mask_demo()
    demo_seq2seq()

    print("所有演示完成！生成的图像文件：")
    print("  - scaling_effect.png      缩放效果对比")
    print("  - attention_translation.png  翻译注意力")
    print("  - attention_self.png      自注意力")
    print("  - causal_mask.png         因果掩码")
    print("  - bahdanau_attention.png  Bahdanau注意力")
```

---

## 练习题

### 基础题

**题目 2.1**（基础）

给定以下 Query 和 Key 向量（无缩放、无批次维度）：

$$Q = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad V = \begin{bmatrix} 10 & 20 \\ 30 & 40 \\ 50 & 60 \end{bmatrix}$$

请手动计算：
1. 得分矩阵 $S = QK^T$
2. 对第一行 $S_1$ 应用 softmax（保留3位小数）
3. $\text{Query}_1$ 对应的注意力输出向量（保留2位小数）

---

**题目 2.2**（基础）

下列说法中，哪些是**正确**的？（多选）

A. 缩放因子 $\sqrt{d_k}$ 使得点积的均值从 $d_k$ 缩放到 1

B. 缩放因子 $\sqrt{d_k}$ 使得点积的方差从 $d_k$ 缩放到 1

C. 当 $d_k = 1$ 时，缩放点积注意力与基础点积注意力等价

D. Bahdanau 注意力不需要任何可学习参数

E. 因果掩码将未来位置的得分设为 $-\infty$，使其注意力权重为 0

---

### 中级题

**题目 2.3**（中级）

**分析缩放的数值效应**

设 $d_k = 256$，有两个随机向量 $q, k \in \mathbb{R}^{256}$，各分量独立地从 $\mathcal{N}(0, 1)$ 采样。

1. 推导不缩放时 $q \cdot k$ 的期望和方差
2. 推导缩放后 $\frac{q \cdot k}{\sqrt{256}}$ 的期望和方差
3. 设注意力序列长度为 $n=8$，得分向量为 $[s_1, \ldots, s_8]$，其中 $s_1 = \max_i s_i = 32$，$s_j = 0 \; (j \neq 1)$。
   - 计算不缩放时 $\text{softmax}$ 的输出（精确到小数点后6位）
   - 计算缩放后（除以 $\sqrt{256}$）$\text{softmax}$ 的输出
   - 比较两者梯度的差异

---

**题目 2.4**（中级）

**实现 Luong general 注意力**

参考代码中 `BahdanauAttention` 的实现，补全以下 Luong general 注意力的代码：

```python
class LuongGeneralAttention(nn.Module):
    def __init__(self, encoder_hidden: int, decoder_hidden: int):
        super().__init__()
        # TODO: 定义所需的线性层（只需一个）

    def forward(
        self,
        decoder_state: torch.Tensor,   # (batch, decoder_hidden)
        encoder_outputs: torch.Tensor  # (batch, src_len, encoder_hidden)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: 实现 Luong general 注意力
        # 公式：e_tj = s_t^T W_a h_j
        # 注意矩阵维度的处理
        pass
```

要求：
1. 实现 `__init__` 和 `forward` 方法
2. 在 `forward` 中添加详细注释说明每步的维度变化
3. 与 `BahdanauAttention` 相比，参数量有什么变化？在什么条件下两者参数量相同？

---

### 提高题

**题目 2.5**（提高）

**注意力复杂度分析与优化思考**

标准缩放点积注意力的时间和空间复杂度均为 $O(n^2)$（$n$ 为序列长度），这在长序列上成为瓶颈。

1. **复杂度推导**：详细推导 $Q, K, V \in \mathbb{R}^{n \times d}$ 时，计算 $\text{softmax}(QK^T / \sqrt{d})V$ 的时间和空间复杂度，说明 $O(n^2)$ 的来源。

2. **稀疏化思路**：考虑局部窗口注意力（每个位置只关注其前后 $w$ 个位置），请：
   - 写出修改后的 mask 矩阵构造方式（伪代码或 PyTorch 代码）
   - 分析此时的时间和空间复杂度
   - 讨论局部窗口注意力的局限性

3. **线性注意力**（了解）：某些工作提出将注意力近似为 $\phi(Q)(\phi(K)^T V)$ 来规避 $QK^T$ 的二次复杂度，其中 $\phi$ 是某核函数。
   - 说明括号重结合如何将复杂度降为 $O(n)$
   - 这种近似需要什么前提条件？

---

## 练习答案

### 题目 2.1 答案

**第一步：计算得分矩阵** $S = QK^T$

$$Q = \begin{bmatrix} 1&2 \\ 3&4 \end{bmatrix}, \quad K^T = \begin{bmatrix} 1&0&1 \\ 0&1&1 \end{bmatrix}$$

$$S = QK^T = \begin{bmatrix} 1\times1+2\times0 & 1\times0+2\times1 & 1\times1+2\times1 \\ 3\times1+4\times0 & 3\times0+4\times1 & 3\times1+4\times1 \end{bmatrix} = \begin{bmatrix} 1 & 2 & 3 \\ 3 & 4 & 7 \end{bmatrix}$$

**第二步：对第一行** $[1, 2, 3]$ **应用 softmax**

$$e^1 = 2.718, \quad e^2 = 7.389, \quad e^3 = 20.086$$

$$\text{sum} = 2.718 + 7.389 + 20.086 = 30.193$$

$$\text{softmax}([1,2,3]) = [0.090, 0.245, 0.665]$$

**第三步：计算** $\text{Query}_1$ **的输出**

$$\text{output}_1 = 0.090 \times [10,20] + 0.245 \times [30,40] + 0.665 \times [50,60]$$

$$= [0.90, 1.80] + [7.35, 9.80] + [33.25, 39.90] = [41.50, 51.50]$$

---

### 题目 2.2 答案

**正确答案：B、C、E**

- **A 错误**：缩放的是**方差**而非均值。当 $Q, K$ 各分量均值为零时，$q \cdot k$ 的均值也为零，缩放前后均值均为 0。
- **B 正确**：$\text{Var}(q \cdot k) = d_k$，缩放后 $\text{Var}(\frac{q \cdot k}{\sqrt{d_k}}) = 1$。
- **C 正确**：当 $d_k = 1$ 时，$\frac{QK^T}{\sqrt{1}} = QK^T$，两者完全相同。
- **D 错误**：Bahdanau 注意力有三组可学习参数 $W_a \in \mathbb{R}^{d\times d_{attn}},\; U_a \in \mathbb{R}^{d\times d_{attn}},\; v_a \in \mathbb{R}^{d_{attn}}$。
- **E 正确**：$e^{-\infty} = 0$，softmax 归一化后该位置权重恰好为 0。

---

### 题目 2.3 答案

**第1问：不缩放时的期望和方差**

设 $q_j, k_j \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)$，则：

$$E[q \cdot k] = \sum_{j=1}^{256} E[q_j k_j] = \sum_{j=1}^{256} E[q_j] E[k_j] = 0$$

$$\text{Var}(q \cdot k) = \sum_{j=1}^{256} \text{Var}(q_j k_j) = \sum_{j=1}^{256} (E[q_j^2 k_j^2] - (E[q_j k_j])^2)$$

由于 $E[q_j^2] = 1,\; E[k_j^2] = 1$，故 $E[q_j^2 k_j^2] = 1$，

$$\text{Var}(q \cdot k) = 256 \times (1 - 0) = 256, \quad \text{标准差} = 16$$

**第2问：缩放后的期望和方差**

$$E\!\left[\frac{q\cdot k}{\sqrt{256}}\right] = 0$$

$$\text{Var}\!\left(\frac{q\cdot k}{\sqrt{256}}\right) = \frac{256}{256} = 1, \quad \text{标准差} = 1$$

**第3问：梯度对比**

不缩放时，得分向量 $s = [32, 0, 0, \ldots, 0]$（共8个）：

$$\text{softmax}(s)_1 = \frac{e^{32}}{e^{32} + 7 \times e^0} = \frac{7.896 \times 10^{13}}{7.896 \times 10^{13} + 7} \approx 0.999999$$

$$\text{softmax}(s)_j \approx \frac{1}{7.896 \times 10^{13}} \approx 1.27 \times 10^{-14} \quad (j \neq 1)$$

缩放后，$s / 16 = [2, 0, 0, \ldots, 0]$：

$$\text{softmax}(s/16)_1 = \frac{e^2}{e^2 + 7} = \frac{7.389}{14.389} \approx 0.514$$

$$\text{softmax}(s/16)_j \approx \frac{1}{14.389} \approx 0.069 \quad (j \neq 1)$$

softmax 的梯度 $\frac{\partial \text{softmax}_i}{\partial s_j} = \text{softmax}_i (\delta_{ij} - \text{softmax}_j)$，当输出接近 one-hot 时梯度趋近于 0，训练信号几乎消失；缩放后分布更平均，梯度更大，训练更有效。

---

### 题目 2.4 答案

```python
class LuongGeneralAttention(nn.Module):
    """Luong general 注意力：e_tj = s_t^T W_a h_j"""

    def __init__(self, encoder_hidden: int, decoder_hidden: int):
        super().__init__()
        # 只需一个权重矩阵：将编码器隐状态映射到解码器空间
        self.W_a = nn.Linear(encoder_hidden, decoder_hidden, bias=False)

    def forward(
        self,
        decoder_state: torch.Tensor,   # (batch, decoder_hidden)
        encoder_outputs: torch.Tensor  # (batch, src_len, encoder_hidden)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 第一步：将编码器输出投影
        # (batch, src_len, encoder_hidden) → (batch, src_len, decoder_hidden)
        projected = self.W_a(encoder_outputs)

        # 第二步：计算得分 e_tj = s_t^T (W_a h_j)
        # decoder_state: (batch, decoder_hidden) → (batch, decoder_hidden, 1)
        # bmm: (batch, src_len, decoder_hidden) × (batch, decoder_hidden, 1)
        #    = (batch, src_len, 1)
        energy = torch.bmm(projected, decoder_state.unsqueeze(-1)).squeeze(-1)
        # energy: (batch, src_len)

        # 第三步：softmax 归一化
        weights = F.softmax(energy, dim=-1)  # (batch, src_len)

        # 第四步：加权求和
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        # (batch, 1, src_len) × (batch, src_len, encoder_hidden) = (batch, encoder_hidden)

        return context, weights
```

**参数量分析**：

设 $d_e = \text{encoder\_hidden}$，$d_d = \text{decoder\_hidden}$，$d_{attn} = \text{attn\_dim}$：

| 模型 | 参数量 |
|------|--------|
| `LuongGeneralAttention` | $d_e \times d_d$（只有 $W_a$） |
| `BahdanauAttention` | $d_e \times d_{attn} + d_d \times d_{attn} + d_{attn}$（$U_a + W_a + v_a$） |

当 $d_{attn} = 1$ 时，Bahdanau 的参数量 $= d_e + d_d + 1$，而 Luong general 为 $d_e \times d_d$，Luong 反而更多。当 $d_{attn}$ 足够大（$d_{attn} > \frac{d_e d_d}{d_e + d_d + 1}$）时，Bahdanau 参数量反超 Luong。通常 $d_{attn}$ 取与隐状态相同量级，Bahdanau 参数量约为 Luong 的两倍。

---

### 题目 2.5 答案

**第1问：复杂度推导**

设 $Q, K, V \in \mathbb{R}^{n \times d}$：

- $QK^T$：$(n \times d) \times (d \times n)$，需要 $n^2 d$ 次乘法，**时间 $O(n^2 d)$**，产生 $n \times n$ 矩阵，**空间 $O(n^2)$**
- $\text{softmax}$：逐行操作，$O(n^2)$
- $AV$：$(n \times n) \times (n \times d)$，需要 $n^2 d$ 次乘法，**时间 $O(n^2 d)$**

总时间：$O(n^2 d)$，**主要瓶颈是空间** $O(n^2)$（需存储完整注意力矩阵），当 $n = 16384$ 时，仅注意力矩阵（fp32）就需要 $16384^2 \times 4 \approx 1 \text{ GB}$。

**第2问：局部窗口注意力**

```python
def local_window_mask(n: int, w: int) -> torch.Tensor:
    """
    构造局部窗口 mask：每个位置只关注 [i-w, i+w] 范围内的位置。
    True 表示需要被遮蔽（不关注）。
    """
    mask = torch.ones(n, n, dtype=torch.bool)
    for i in range(n):
        lo = max(0, i - w)
        hi = min(n - 1, i + w)
        mask[i, lo:hi+1] = False  # 窗口内不遮蔽
    return mask
```

复杂度分析：每个位置只与 $2w+1$ 个位置交互，总操作数 $O(n \cdot w \cdot d)$，空间 $O(n \cdot w)$，**均为线性**（$w$ 固定时）。

局限性：
- 无法捕获长程依赖（如句子开头和结尾的关系）
- 对于语言任务，某些结构需要全局信息（如指代消解）
- 实际实现需要特殊的稀疏矩阵计算，普通 GPU 上难以高效实现

**第3问：线性注意力**

标准注意力：$O = \text{softmax}(QK^T)V$，括号顺序是 $((QK^T)V)$，先算 $QK^T$，得到 $n\times n$ 矩阵，再乘 $V$。

线性注意力将 softmax 替换为核分解：

$$O_i = \frac{\sum_j \phi(q_i)^T \phi(k_j) v_j^T}{\sum_j \phi(q_i)^T \phi(k_j)}$$

改变括号顺序为 $\phi(q_i)^T (\sum_j \phi(k_j) v_j^T)$，先计算 $\sum_j \phi(k_j) v_j^T \in \mathbb{R}^{d' \times d}$，只需 $O(nd'd)$ 时间和 $O(d'd)$ 空间，再对每个 query 查询，总复杂度 $O(nd'd)$，**线性于 $n$**。

前提条件：核函数 $\phi$ 必须满足 $K(q, k) = \phi(q)^T\phi(k)$（核函数正定），即原始的注意力函数可被分解为两个映射的内积。而 $\text{softmax}$ 对应的 $\exp$ 核并不能精确分解，只能近似，因此线性注意力是对标准注意力的**近似**，在某些任务上性能有所损失。

---

## 延伸阅读

- **Bahdanau et al. (2014)**：*Neural Machine Translation by Jointly Learning to Align and Translate* — 注意力机制的奠基性工作
- **Luong et al. (2015)**：*Effective Approaches to Attention-based Neural Machine Translation* — 多种注意力变体的系统对比
- **Vaswani et al. (2017)**：*Attention Is All You Need* — 提出缩放点积注意力和 Transformer
- **Clark et al. (2019)**：*What Does BERT Look at? An Analysis of BERT's Attention* — 可视化分析 BERT 的注意力模式
- **Tay et al. (2022)**：*Efficient Transformers: A Survey* — 高效注意力机制综述（含线性注意力）

---

## 下一章预告

第3章将在本章基础上介绍**多头注意力**（Multi-Head Attention）：
- 为什么单头注意力不够用
- 多头的并行计算实现
- 位置编码（Positional Encoding）的设计
- 完整的 Transformer 编码器层

> 关键问题思考：如果每个注意力头使用相同的 $Q, K, V$ 投影矩阵，多头注意力还有意义吗？请带着这个问题进入第3章。
