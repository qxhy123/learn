# 第1章：序列建模基础

> **本章导读**：在深入Transformer之前，我们需要理解它所要解决的问题。本章从序列建模的基本概念出发，回顾RNN和LSTM的原理与局限，最终揭示注意力机制诞生的动机。

---

## 学习目标

学完本章后，你将能够：

1. **理解序列到序列问题的定义**：清晰描述什么是序列数据，以及序列到序列任务的典型形式
2. **掌握RNN和LSTM的基本原理**：理解循环结构的工作方式，推导前向传播公式
3. **理解长距离依赖问题**：解释为何经典循环模型在长序列上会失效
4. **了解注意力机制的动机**：从RNN的缺陷出发，理解注意力机制要解决的核心问题
5. **能够用PyTorch实现简单的RNN**：编写可运行的RNN/LSTM代码，并可视化梯度消失现象

---

## 1.1 序列建模问题

### 什么是序列数据

在自然界和人类社会中，**序列数据**无处不在。与普通的独立同分布数据不同，序列数据中的每一个元素都与它的前后元素存在依赖关系，顺序本身携带了重要信息。

**文本（Text）**

自然语言是最典型的序列数据。句子"我爱北京天安门"中，每个字符的含义都依赖于上下文：

```
我 → 爱 → 北京 → 天安门
```

如果打乱顺序为"天安门爱我北京"，意思完全改变。语言模型的任务就是学习这种序列中的条件概率：

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

**时间序列（Time Series）**

股票价格、气温变化、心电图信号都属于时间序列。预测明天的股价需要参考过去一段时间的走势：

```
价格: [100, 102, 98, 105, 103, ?, ?]
         ↑    ↑    ↑    ↑    ↑
       t-4  t-3  t-2  t-1   t  → 预测未来
```

**音频（Audio）**

语音信号是以极高采样率（如44100 Hz）记录的连续波形。语音识别需要将音频序列转换为文字序列，这是一个典型的序列到序列问题。

**视频（Video）**

视频是帧序列，动作识别、视频字幕生成都需要对时序信息建模。

### 序列到序列任务

**Seq2Seq（Sequence-to-Sequence）** 任务是序列建模的核心范式：给定输入序列 $X = (x_1, x_2, \ldots, x_m)$，输出序列 $Y = (y_1, y_2, \ldots, y_n)$，其中输入和输出的长度 $m$ 和 $n$ 可以不同。

| 任务 | 输入序列 | 输出序列 |
|------|---------|---------|
| 机器翻译 | 中文句子 | 英文句子 |
| 文本摘要 | 长篇文章 | 简短摘要 |
| 对话系统 | 用户提问 | 系统回答 |
| 语音识别 | 音频帧序列 | 文字序列 |
| 代码补全 | 已有代码 | 补全代码 |
| 图像描述 | 图像特征序列 | 描述文字 |

**机器翻译示例**：

```
输入（中文）: 我 爱 自然 语言 处理
              ↓   ↓   ↓    ↓    ↓
输出（英文）: I  love natural language processing
```

**文本摘要示例**：

```
输入（长文）: 今日，科学家在某项研究中发现，通过对数千名志愿者
             长达十年的跟踪研究，证明了每天锻炼30分钟与降低
             心血管疾病风险之间存在显著相关性...（500字）

输出（摘要）: 研究证明每日运动30分钟可显著降低心血管疾病风险。
```

### 序列建模的挑战

序列建模面临若干独特挑战，这些挑战直接驱动了模型架构的演进：

**挑战1：可变长度（Variable Length）**

不同样本的序列长度各不相同：一篇新闻可能有500个词，一条推文只有20个词。模型必须能处理任意长度的输入。

**挑战2：长距离依赖（Long-range Dependencies）**

序列中相距很远的元素可能存在语义关联：

```
"The animal didn't cross the street because it was too tired."
 ↑                                              ↑
 animal                                        it → 指代 animal
```

"it"指代的是"animal"还是"street"？这取决于句子后半部分的"tired"（动物才会累，街道不会）。距离越远，依赖越难捕捉。

**挑战3：顺序处理的并行瓶颈（Sequential Processing Bottleneck）**

传统RNN必须按时间步顺序处理，$t$ 时刻的计算依赖于 $t-1$ 时刻的结果，无法并行化。对于长序列，训练效率极低。

**挑战4：梯度消失与爆炸（Vanishing/Exploding Gradients）**

在反向传播时，梯度需要通过所有时间步传递。长序列上梯度会指数级衰减（消失）或增大（爆炸），导致模型无法学习有效的长期依赖。

---

## 1.2 循环神经网络（RNN）

### RNN的基本结构

循环神经网络（Recurrent Neural Network, RNN）是处理序列数据的经典架构。其核心思想是：**通过隐藏状态（hidden state）在时间步之间传递信息**。

```
输入序列: x_1  x_2  x_3  ...  x_T
           ↓    ↓    ↓         ↓
隐藏状态: h_1  h_2  h_3  ...  h_T
           ↓    ↓    ↓         ↓
输出序列: y_1  y_2  y_3  ...  y_T

其中每个 h_t 由 h_{t-1} 和 x_t 共同决定
```

**展开图**：

```
x_1 → [RNN] → h_1 → [RNN] → h_2 → [RNN] → h_3
          ↑               ↑               ↑
         参数W            参数W            参数W（共享）
```

关键特点是：**所有时间步共享同一组参数**（$W_{hh}$, $W_{xh}$, $b$），这使得模型的参数量不随序列长度增加而增长。

### 前向传播公式

RNN的前向传播由以下公式描述：

**隐藏状态更新**：

$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$

**输出计算**（可选，取决于任务）：

$$y_t = W_{hy}h_t + b_y$$

其中：
- $x_t \in \mathbb{R}^{d_x}$：第 $t$ 个时间步的输入向量
- $h_t \in \mathbb{R}^{d_h}$：第 $t$ 个时间步的隐藏状态
- $h_0$：初始隐藏状态，通常初始化为零向量
- $W_{hh} \in \mathbb{R}^{d_h \times d_h}$：隐藏到隐藏的权重矩阵
- $W_{xh} \in \mathbb{R}^{d_h \times d_x}$：输入到隐藏的权重矩阵
- $b_h \in \mathbb{R}^{d_h}$：偏置向量
- $\tanh$：双曲正切激活函数，将输出压缩到 $(-1, 1)$

**矩阵形式**（更紧凑）：

将 $h_{t-1}$ 和 $x_t$ 拼接，可以写成：

$$h_t = \tanh\left(\begin{bmatrix} W_{hh} & W_{xh} \end{bmatrix} \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_h\right)$$

### PyTorch实现

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 方法1：手动实现RNN（理解原理）
# ============================================================

class ManualRNN(nn.Module):
    """手动实现的RNN，便于理解内部计算"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 参数矩阵
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_xh = nn.Linear(input_size, hidden_size, bias=True)

    def forward(self, x, h_0=None):
        """
        参数:
            x: (batch_size, seq_len, input_size)
            h_0: (batch_size, hidden_size)，初始隐藏状态

        返回:
            outputs: (batch_size, seq_len, hidden_size)
            h_n: (batch_size, hidden_size)，最后一步隐藏状态
        """
        batch_size, seq_len, _ = x.shape

        # 初始化隐藏状态
        if h_0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = h_0

        outputs = []

        # 按时间步逐步计算
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)

            # 核心公式: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)
            h_t = torch.tanh(self.W_hh(h_t) + self.W_xh(x_t))
            outputs.append(h_t)

        # 堆叠所有时间步的输出
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)

        return outputs, h_t


# ============================================================
# 方法2：使用PyTorch内置nn.RNN
# ============================================================

class SimpleRNNModel(nn.Module):
    """基于nn.RNN的序列分类模型"""

    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, num_layers=1):
        super().__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # RNN层
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,    # 输入格式: (batch, seq, feature)
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        """
        参数:
            x: (batch_size, seq_len) 词索引序列

        返回:
            logits: (batch_size, num_classes)
        """
        # 词嵌入
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # RNN前向传播
        # output: (batch, seq_len, hidden_size) - 所有时间步的隐藏状态
        # h_n:    (num_layers, batch, hidden_size) - 最后时间步的隐藏状态
        output, h_n = self.rnn(embedded)

        # 取最后一个时间步的隐藏状态作为序列表示
        last_hidden = h_n[-1]  # (batch, hidden_size)

        # 分类
        logits = self.classifier(last_hidden)

        return logits


# 快速测试
def test_rnn():
    batch_size = 4
    seq_len = 10
    input_size = 8
    hidden_size = 16

    x = torch.randn(batch_size, seq_len, input_size)

    # 测试手动实现
    manual_rnn = ManualRNN(input_size, hidden_size)
    out, h_n = manual_rnn(x)
    print(f"ManualRNN - output: {out.shape}, h_n: {h_n.shape}")
    # 预期: output: (4, 10, 16), h_n: (4, 16)

    # 测试nn.RNN
    torch_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    out2, h_n2 = torch_rnn(x)
    print(f"nn.RNN   - output: {out2.shape}, h_n: {h_n2.shape}")
    # 预期: output: (4, 10, 16), h_n: (1, 4, 16)


test_rnn()
```

### RNN的问题：梯度消失与爆炸

RNN的训练依赖**基于时间的反向传播（Backpropagation Through Time, BPTT）**。设损失函数为 $L$，对早期参数的梯度为：

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

其中每个 Jacobian 矩阵为：

$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh'(\cdot)) \cdot W_{hh}$$

连乘 $T-1$ 项后，若 $W_{hh}$ 的最大奇异值 $\sigma_1$：
- $\sigma_1 < 1$：梯度指数级**消失** $\rightarrow$ 早期时间步无法接收有效梯度信号
- $\sigma_1 > 1$：梯度指数级**爆炸** $\rightarrow$ 训练数值不稳定

```
梯度消失示意：
t=1   t=2   t=3   t=4   t=5   t=6   t=7   t=8
 ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑
0.001 0.003 0.01  0.03  0.1   0.3   1.0   3.0  ← 梯度大小（从右向左）

早期时间步（t=1,2）的梯度几乎为0，这些参数无法被有效更新
```

---

## 1.3 长短期记忆网络（LSTM）

### LSTM的动机

为了解决RNN的梯度消失问题，Hochreiter和Schmidhuber在1997年提出了**长短期记忆网络（Long Short-Term Memory, LSTM）**。LSTM的核心思想是引入一个**细胞状态（cell state）** $c_t$ 作为"信息高速公路"，通过**门控机制（gating mechanism）** 选择性地读写信息。

```
RNN:   h_{t-1} ──→ [tanh] ──→ h_t
                    ↑
                   x_t

LSTM:  h_{t-1} ────────────────────────────→ h_t
       c_{t-1} ──[遗忘门]──[输入门]──[输出门]── c_t
                    ↑          ↑          ↑
                   x_t        x_t        x_t
```

### 门控机制详解

LSTM包含三个门，每个门都是一个0到1之间的向量，控制信息的流动量：

**遗忘门（Forget Gate）** - 决定从细胞状态中"丢弃"哪些信息：

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**输入门（Input Gate）** - 决定将哪些新信息"写入"细胞状态：

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

**细胞状态更新** - 结合遗忘和输入门更新细胞状态：

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

其中 $\odot$ 表示逐元素乘法（Hadamard积）。

**输出门（Output Gate）** - 决定从细胞状态中"读取"哪些信息作为隐藏状态：

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(c_t)$$

**关键洞察**：细胞状态的更新是**加法操作**（$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$），而非RNN中的乘法/激活函数嵌套。加法操作使梯度能够在长序列上近乎无阻碍地流动，这是LSTM缓解梯度消失的核心机制。

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t \approx 1 \text{（当遗忘门接近1时）}$$

### PyTorch实现

```python
import torch
import torch.nn as nn

# ============================================================
# 方法1：手动实现LSTM（展示所有门的计算）
# ============================================================

class ManualLSTMCell(nn.Module):
    """单步LSTM单元，展示四个门的完整计算"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 将四个门的线性变换合并为一个大矩阵（高效实现）
        # 顺序: [input_gate, forget_gate, cell_gate, output_gate]
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        """
        参数:
            x_t: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)
            c_prev: (batch_size, hidden_size)

        返回:
            h_t: (batch_size, hidden_size)
            c_t: (batch_size, hidden_size)
        """
        # 拼接输入和上一步隐藏状态
        combined = torch.cat([h_prev, x_t], dim=1)  # (batch, hidden+input)

        # 计算四个门的原始值（一次矩阵乘法，高效）
        gates = self.linear(combined)  # (batch, 4 * hidden_size)

        # 拆分四个门
        i, f, g, o = gates.chunk(4, dim=1)

        # 应用激活函数
        i_t = torch.sigmoid(i)   # 输入门: [0, 1]
        f_t = torch.sigmoid(f)   # 遗忘门: [0, 1]
        g_t = torch.tanh(g)      # 候选细胞: [-1, 1]
        o_t = torch.sigmoid(o)   # 输出门: [0, 1]

        # 更新细胞状态（加法，保护梯度流）
        c_t = f_t * c_prev + i_t * g_t

        # 计算隐藏状态
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class ManualLSTM(nn.Module):
    """完整的手动LSTM，展示序列处理"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = ManualLSTMCell(input_size, hidden_size)

    def forward(self, x, states=None):
        """
        参数:
            x: (batch_size, seq_len, input_size)
            states: (h_0, c_0) 初始状态元组

        返回:
            outputs: (batch_size, seq_len, hidden_size)
            (h_n, c_n): 最后时间步的状态
        """
        batch_size, seq_len, _ = x.shape

        if states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t, c_t = states

        outputs = []
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], h_t, c_t)
            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=1)
        return outputs, (h_t, c_t)


# ============================================================
# 方法2：使用PyTorch内置nn.LSTM
# ============================================================

class LSTMClassifier(nn.Module):
    """基于LSTM的文本分类模型"""

    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 num_layers=2, bidirectional=False, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 双向LSTM输出维度翻倍
        direction_factor = 2 if bidirectional else 1

        self.classifier = nn.Linear(hidden_size * direction_factor, num_classes)

    def forward(self, x):
        """
        参数:
            x: (batch_size, seq_len) 词索引

        返回:
            logits: (batch_size, num_classes)
        """
        embedded = self.dropout(self.embedding(x))

        # output: (batch, seq_len, hidden * directions)
        # (h_n, c_n): 各层最后时间步的状态
        output, (h_n, c_n) = self.lstm(embedded)

        # 取最后一层的最后时间步隐藏状态
        # h_n shape: (num_layers * directions, batch, hidden_size)
        last_hidden = h_n[-1]  # (batch, hidden_size)

        logits = self.classifier(last_hidden)
        return logits


# 测试LSTM
def test_lstm():
    batch_size, seq_len, input_size, hidden_size = 4, 15, 8, 16
    x = torch.randn(batch_size, seq_len, input_size)

    # 手动实现
    manual_lstm = ManualLSTM(input_size, hidden_size)
    out, (h_n, c_n) = manual_lstm(x)
    print(f"ManualLSTM - output: {out.shape}, h_n: {h_n.shape}, c_n: {c_n.shape}")

    # PyTorch内置
    torch_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    out2, (h_n2, c_n2) = torch_lstm(x)
    print(f"nn.LSTM    - output: {out2.shape}, h_n: {h_n2.shape}, c_n: {c_n2.shape}")


test_lstm()
```

### LSTM vs RNN 对比

| 特性 | 简单RNN | LSTM |
|------|--------|------|
| 状态 | $h_t$（1个向量） | $h_t, c_t$（2个向量） |
| 更新方式 | 乘法嵌套 | 加法更新细胞状态 |
| 门控机制 | 无 | 3个门（遗忘/输入/输出） |
| 梯度流 | 容易消失/爆炸 | 较好，但仍有问题 |
| 参数量 | $4 \times d_h^2$ | $4 \times 4 \times d_h^2$（约4倍） |
| 训练速度 | 较快 | 较慢 |
| 长序列性能 | 差 | 显著提升，但有上限 |
| 适用场景 | 短序列，资源受限 | 大多数序列任务 |

---

## 1.4 长距离依赖问题

### 理论分析：信息瓶颈

即使LSTM缓解了梯度消失，序列建模仍然面临一个根本性的**信息瓶颈**：所有历史信息都被压缩进固定维度的隐藏状态 $h_t \in \mathbb{R}^{d_h}$。

```
输入序列（任意长）:  x_1, x_2, ..., x_100, ..., x_500
                                                    ↓
隐藏状态（固定大小）: ─────────────────────────────→ h_500 ∈ ℝ^{512}

问题：500个词的所有信息都要压缩进一个512维向量！
```

想象你需要把一本书的全部内容压缩成一段话来传给翻译者——这就是Seq2Seq模型在做的事，而且对于越长的书（序列），压缩损失越大。

### 实验演示：长序列性能下降

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def generate_copy_task(seq_len, batch_size=32, vocab_size=10):
    """
    复制任务：模型需要记住序列开头的内容，在序列末尾输出它

    输入: [x_1, ..., x_k, PAD, PAD, ..., PAD, SEP]
    目标: 在最后k步输出 x_1, ..., x_k

    这个任务要求模型记住距离当前位置很远的信息
    """
    prefix_len = 5  # 需要记住的内容长度

    # 生成随机前缀
    prefix = torch.randint(1, vocab_size, (batch_size, prefix_len))

    # 中间填充（测试长距离记忆）
    gap = seq_len - 2 * prefix_len - 1
    padding = torch.zeros(batch_size, gap, dtype=torch.long)

    # 分隔符
    sep = torch.full((batch_size, 1), vocab_size, dtype=torch.long)

    # 构建输入
    inputs = torch.cat([prefix, padding, sep, prefix], dim=1)

    # 目标是输出前缀
    targets = prefix

    return inputs, targets


def evaluate_copy_task(model, seq_lengths, num_trials=100):
    """在不同序列长度上评估模型的复制准确率"""
    accuracies = []

    model.eval()
    with torch.no_grad():
        for seq_len in seq_lengths:
            correct = 0
            total = 0

            for _ in range(num_trials // 32 + 1):
                inputs, targets = generate_copy_task(seq_len)

                # 简化：只评估最后一步的预测
                outputs, _ = model(inputs)
                last_output = outputs[:, -1, :]  # 取最后时间步

                # 计算最可能的预测
                pred = last_output.argmax(dim=-1)

                # 与目标的第一个元素比较（简化评估）
                correct += (pred == targets[:, 0]).sum().item()
                total += targets.size(0)

            acc = correct / total
            accuracies.append(acc)

    return accuracies


# 可视化长距离依赖问题
def visualize_long_range_problem():
    """展示RNN和LSTM在不同序列长度上的性能"""

    # 模拟实验结果（实际训练需要更多时间）
    seq_lengths = [10, 20, 50, 100, 200, 500]

    # 典型的实验观察（近似值，体现趋势）
    rnn_accuracy =  [0.92, 0.78, 0.45, 0.23, 0.12, 0.08]
    lstm_accuracy = [0.94, 0.89, 0.75, 0.58, 0.42, 0.31]
    transformer_accuracy = [0.96, 0.95, 0.93, 0.92, 0.91, 0.89]  # 预览：第3章内容

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：准确率 vs 序列长度
    ax1 = axes[0]
    ax1.plot(seq_lengths, rnn_accuracy, 'o-', color='#e74c3c',
             label='Simple RNN', linewidth=2, markersize=8)
    ax1.plot(seq_lengths, lstm_accuracy, 's-', color='#3498db',
             label='LSTM', linewidth=2, markersize=8)
    ax1.plot(seq_lengths, transformer_accuracy, '^--', color='#2ecc71',
             label='Transformer (预览)', linewidth=2, markersize=8, alpha=0.7)

    ax1.set_xlabel('序列长度', fontsize=12)
    ax1.set_ylabel('复制任务准确率', fontsize=12)
    ax1.set_title('长距离依赖：性能随序列长度的衰减', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='随机基线')

    # 右图：有效记忆距离
    ax2 = axes[1]
    models = ['Simple RNN', 'LSTM', 'GRU', 'Transformer']
    effective_memory = [10, 50, 45, 512]  # 大约的有效记忆距离（词数）
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']

    bars = ax2.bar(models, effective_memory, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('有效记忆距离（token数）', fontsize=12)
    ax2.set_title('不同架构的有效记忆范围对比', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, effective_memory):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig('long_range_dependency.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图表已保存: long_range_dependency.png")


visualize_long_range_problem()
```

### 梯度流可视化

理解梯度如何在时间步之间流动，是理解长距离依赖问题的关键：

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def visualize_gradient_flow(model_type='rnn', seq_len=50, hidden_size=64):
    """
    可视化梯度在时间步之间的流动

    通过记录每个时间步的梯度范数，可以看到：
    - RNN: 梯度随时间步指数级消失
    - LSTM: 梯度流更稳定（但仍有衰减）
    """

    input_size = 16
    batch_size = 1

    # 创建模型
    if model_type == 'rnn':
        model = nn.RNN(input_size, hidden_size, batch_first=True)
        title = 'Simple RNN'
        color = '#e74c3c'
    else:
        model = nn.LSTM(input_size, hidden_size, batch_first=True)
        title = 'LSTM'
        color = '#3498db'

    # 创建随机输入，需要梯度
    x = torch.randn(batch_size, seq_len, input_size, requires_grad=True)

    # 前向传播
    if model_type == 'rnn':
        outputs, h_n = model(x)
    else:
        outputs, (h_n, c_n) = model(x)

    # 计算损失（对最后时间步的输出求和）
    loss = outputs[:, -1, :].sum()

    # 反向传播
    loss.backward()

    # 收集每个时间步输出的梯度（通过hook）
    grad_norms = []

    # 重新运行，这次用hook记录梯度
    model.zero_grad()
    x2 = torch.randn(batch_size, seq_len, input_size)

    hooks = []
    step_grads = [None] * seq_len

    if model_type == 'rnn':
        outputs2, _ = model(x2)
    else:
        outputs2, _ = model(x2)

    # 计算每个时间步输出对最终损失的梯度
    final_loss = outputs2[:, -1, :].sum()

    # 计算对每个中间输出的梯度
    for t in range(seq_len):
        output_t = outputs2[:, t, :]

        # 保留计算图以便多次backward
        if t < seq_len - 1:
            grad = torch.autograd.grad(final_loss, output_t,
                                        retain_graph=True,
                                        allow_unused=True)[0]
        else:
            grad = torch.autograd.grad(final_loss, output_t,
                                        allow_unused=True)[0]

        if grad is not None:
            grad_norms.append(grad.norm().item())
        else:
            grad_norms.append(0.0)

    return grad_norms


# 可视化对比
def plot_gradient_comparison():
    seq_len = 40

    # 注意：这里展示理想化的梯度范数模式
    # 实际值取决于随机初始化，以下是典型趋势

    t = np.arange(seq_len)

    # RNN: 指数衰减（从最后一步向前）
    rnn_grads = np.exp(-0.15 * (seq_len - 1 - t)) * (1 + 0.3 * np.random.randn(seq_len))
    rnn_grads = np.abs(rnn_grads)

    # LSTM: 更平缓的衰减（细胞状态保护梯度流）
    lstm_grads = np.exp(-0.04 * (seq_len - 1 - t)) * (1 + 0.2 * np.random.randn(seq_len))
    lstm_grads = np.abs(lstm_grads)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # RNN梯度图
    ax1 = axes[0]
    ax1.bar(t, rnn_grads, color='#e74c3c', alpha=0.7, label='梯度范数')
    ax1.axvline(x=seq_len*0.3, color='orange', linestyle='--', alpha=0.8)
    ax1.text(seq_len*0.3 + 0.5, max(rnn_grads)*0.8, '这里之前的梯度已近乎为0',
             fontsize=9, color='orange')
    ax1.set_title('Simple RNN: 梯度随时间步快速消失', fontsize=12, fontweight='bold')
    ax1.set_ylabel('梯度范数（对数尺度）', fontsize=10)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('时间步 t（从早到晚）')

    # LSTM梯度图
    ax2 = axes[1]
    ax2.bar(t, lstm_grads, color='#3498db', alpha=0.7, label='梯度范数')
    ax2.set_title('LSTM: 细胞状态保护梯度，衰减更平缓', fontsize=12, fontweight='bold')
    ax2.set_ylabel('梯度范数（对数尺度）', fontsize=10)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('时间步 t（从早到晚）')

    plt.tight_layout()
    plt.savefig('gradient_flow.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("梯度流图表已保存: gradient_flow.png")


# 设置随机种子以保证可重复性
np.random.seed(42)
plot_gradient_comparison()
```

---

## 1.5 从RNN到注意力

### RNN的固有限制

经过深入分析，RNN/LSTM存在三个**结构性限制**，这些限制无法通过简单的超参数调整来解决：

**限制1：信息瓶颈（Information Bottleneck）**

```
编码器:  x_1 → x_2 → x_3 → ... → x_n → [压缩到单个向量 c]
                                                     ↓
解码器:  [从 c 生成] → y_1 → y_2 → ... → y_m

问题: 整个输入序列的语义信息必须全部压缩进固定维度的向量 c
      对于长序列，这是不可能完成的压缩任务
```

**限制2：顺序处理，无法并行（Sequential Processing）**

```python
# RNN的计算依赖链（无法并行化）
h_1 = f(x_1, h_0)    # 必须先算h_1
h_2 = f(x_2, h_1)    # 才能算h_2  ← 依赖h_1
h_3 = f(x_3, h_2)    # 才能算h_3  ← 依赖h_2
...
h_n = f(x_n, h_{n-1})

# 时间复杂度: O(n) 串行步骤，无论计算资源多少
```

在现代GPU拥有数千个并行计算核心的时代，这是巨大的浪费。Transformer通过自注意力机制使所有位置的计算可以**完全并行化**。

**限制3：距离惩罚（Distance Penalty）**

任意两个位置 $i$ 和 $j$ 之间的信息交互路径长度为 $O(|i-j|)$（需要逐步传递），而Transformer中任意两个位置的路径长度为 $O(1)$（直接注意力）。

```
RNN中位置1和位置100的交互:
1 → 2 → 3 → ... → 99 → 100    路径长度: 99步

Transformer中位置1和位置100的交互:
1 ←────────────────── 100      路径长度: 1步（直接注意力）
```

### 注意力机制的直觉

**注意力机制（Attention Mechanism）** 的核心直觉非常自然：**在生成输出时，不要依赖固定的压缩向量，而是让模型"看"整个输入序列，并决定每个时刻关注哪些部分**。

这就像人类翻译句子时的行为：

```
中文: "猫  坐  在  垫子  上"
英文: "The cat sat  on  the  mat"

翻译 "cat" 时 → 重点关注 "猫"        (注意力权重: 猫=0.9, 其他≈0)
翻译 "sat"  时 → 重点关注 "坐"        (注意力权重: 坐=0.85, 其他≈0)
翻译 "mat"  时 → 重点关注 "垫子"+"上" (注意力权重: 垫子=0.7, 上=0.2)
```

**注意力得分计算（形式化）**：

给定解码器在时刻 $t$ 的隐藏状态 $s_t$ 和编码器所有时刻的输出 $\{h_1, h_2, \ldots, h_n\}$：

**第一步：计算注意力得分（对齐打分）**

$$e_{t,i} = \text{score}(s_t, h_i)$$

常用的打分函数：
- 点积：$e_{t,i} = s_t^T h_i$
- 加性（Bahdanau）：$e_{t,i} = v^T \tanh(W_s s_t + W_h h_i)$
- 缩放点积（Transformer用）：$e_{t,i} = \frac{s_t^T h_i}{\sqrt{d_k}}$

**第二步：Softmax归一化为注意力权重**

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{n} \exp(e_{t,j})}$$

**第三步：对编码器输出加权求和**

$$c_t = \sum_{i=1}^{n} \alpha_{t,i} h_i$$

现在 $c_t$ 是一个**动态的上下文向量**，每个解码步都不同——这解决了信息瓶颈问题。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class BahdanauAttention(nn.Module):
    """
    Bahdanau注意力机制（加性注意力）
    来自论文: "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
    """

    def __init__(self, encoder_hidden_size, decoder_hidden_size, attention_size):
        super().__init__()

        # W_s * s_t: 解码器状态的线性变换
        self.W_decoder = nn.Linear(decoder_hidden_size, attention_size, bias=False)

        # W_h * h_i: 编码器状态的线性变换
        self.W_encoder = nn.Linear(encoder_hidden_size, attention_size, bias=False)

        # v^T: 输出分数
        self.v = nn.Linear(attention_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        """
        参数:
            decoder_state:   (batch_size, decoder_hidden)      解码器当前状态
            encoder_outputs: (batch_size, src_len, encoder_hidden) 编码器所有输出

        返回:
            context:         (batch_size, encoder_hidden)       加权上下文向量
            attention_weights: (batch_size, src_len)            注意力权重（可视化用）
        """
        src_len = encoder_outputs.size(1)

        # 扩展解码器状态以匹配序列长度
        # (batch, decoder_hidden) → (batch, src_len, decoder_hidden)
        decoder_expanded = decoder_state.unsqueeze(1).expand(-1, src_len, -1)

        # 计算注意力得分: e_i = v^T * tanh(W_s * s + W_h * h_i)
        energy = self.v(torch.tanh(
            self.W_decoder(decoder_expanded) +  # (batch, src_len, attn_size)
            self.W_encoder(encoder_outputs)      # (batch, src_len, attn_size)
        )).squeeze(-1)  # (batch, src_len)

        # Softmax归一化
        attention_weights = F.softmax(energy, dim=-1)  # (batch, src_len)

        # 加权求和上下文向量
        # (batch, 1, src_len) × (batch, src_len, encoder_hidden)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (batch, encoder_hidden)

        return context, attention_weights


def visualize_attention(attention_matrix, src_tokens, tgt_tokens):
    """
    可视化注意力矩阵

    参数:
        attention_matrix: (tgt_len, src_len) numpy数组
        src_tokens: 源语言词列表
        tgt_tokens: 目标语言词列表
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')

    # 设置坐标轴
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, fontsize=12)
    ax.set_yticklabels(tgt_tokens, fontsize=12)

    # 在每个格子显示权重值
    for i in range(len(tgt_tokens)):
        for j in range(len(src_tokens)):
            val = attention_matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=9, color=color, fontweight='bold')

    ax.set_xlabel('源语言（编码器输入）', fontsize=13)
    ax.set_ylabel('目标语言（解码器输出）', fontsize=13)
    ax.set_title('注意力权重可视化\n颜色越深 = 注意力权重越大', fontsize=14, fontweight='bold')

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


# 演示：机器翻译中的注意力对齐
src_tokens = ['我', '爱', '自然', '语言', '处理', '<EOS>']
tgt_tokens = ['I', 'love', 'natural', 'language', 'processing', '<EOS>']

# 模拟理想的注意力对齐矩阵（接近对角线）
attention_matrix = np.array([
    [0.90, 0.05, 0.02, 0.01, 0.01, 0.01],  # I ← 我
    [0.05, 0.88, 0.04, 0.01, 0.01, 0.01],  # love ← 爱
    [0.02, 0.03, 0.87, 0.05, 0.02, 0.01],  # natural ← 自然
    [0.01, 0.02, 0.05, 0.89, 0.02, 0.01],  # language ← 语言
    [0.01, 0.01, 0.03, 0.04, 0.89, 0.02],  # processing ← 处理
    [0.01, 0.01, 0.01, 0.01, 0.03, 0.93],  # <EOS> ← <EOS>
])

visualize_attention(attention_matrix, src_tokens, tgt_tokens)
```

### 为什么Transformer抛弃了循环结构

注意力机制的成功引出了一个激进的问题：**如果注意力可以直接建模任意两个位置的关系，我们还需要循环结构来传递信息吗？**

2017年，Google在论文《Attention Is All You Need》中给出了答案：**不需要**。

**Transformer的核心思想**：

```
RNN思路: 通过时间步逐步传递信息
         x_1 → h_1 → h_2 → ... → h_n（串行）

Transformer思路: 每个位置直接与所有其他位置交互
         x_1 ←→ x_2 ←→ x_3 ←→ ... ←→ x_n（并行）
         （通过自注意力机制，同时处理所有位置）
```

**三大优势**：

| 维度 | RNN/LSTM | Transformer |
|------|---------|------------|
| 并行性 | $O(n)$ 串行步骤 | $O(1)$ 并行步骤 |
| 最大路径长度 | $O(n)$ | $O(1)$ |
| 计算复杂度/层 | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$ |
| 长距离依赖 | 难以捕捉 | 直接建模 |

*注：$n$ 为序列长度，$d$ 为模型维度*

这个"抛弃循环，全用注意力"的大胆决策，催生了GPT、BERT、T5等现代大语言模型的诞生。

---

## 本章小结

| 模型 | 核心公式 | 优点 | 缺点 | 典型应用 |
|------|---------|------|------|---------|
| **简单RNN** | $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t)$ | 结构简单，参数少 | 梯度消失，无法建模长依赖 | 短序列分类 |
| **LSTM** | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ | 缓解梯度消失，长期记忆 | 顺序处理，信息瓶颈，参数多 | 语言模型，机器翻译（2017前） |
| **GRU** | $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ | LSTM的简化版，更高效 | 同LSTM，但略差 | 资源受限场景 |
| **带注意力的Seq2Seq** | $c_t = \sum_i \alpha_{t,i} h_i$ | 解决信息瓶颈，可解释 | 仍需顺序编码，解码串行 | 翻译，摘要 |
| **Transformer** | $\text{Attn}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$ | 完全并行，直接长依赖 | 二次空间复杂度 | **现代NLP所有任务** |

**核心要点回顾**：

1. **序列数据**具有顺序依赖性，序列建模的核心挑战是捕捉**长距离依赖**
2. **RNN**通过隐藏状态传递信息，但面临**梯度消失**和**顺序处理瓶颈**
3. **LSTM**通过门控机制和细胞状态的**加法更新**缓解梯度消失，但未解决并行化问题
4. **注意力机制**通过直接计算任意位置对的相关性，绕过了信息瓶颈
5. **Transformer**完全基于注意力，实现了序列处理的**完全并行化**，是现代大模型的基础

---

## 代码实战：完整实验

```python
"""
完整实验代码：RNN vs LSTM 梯度消失可视化与长序列性能对比

运行环境: Python 3.8+, PyTorch 1.12+, matplotlib 3.5+
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import List, Tuple

# ──────────────────────────────────────────────────────────────
# 1. 梯度消失可视化实验
# ──────────────────────────────────────────────────────────────

def compute_gradient_norms(model_type: str, seq_len: int = 50,
                            hidden_size: int = 64, input_size: int = 16,
                            seed: int = 42) -> List[float]:
    """
    计算RNN/LSTM在每个时间步的梯度范数

    返回: 从t=0到t=T的梯度范数列表（越靠前越早）
    """
    torch.manual_seed(seed)

    if model_type == 'rnn':
        model = nn.RNN(input_size, hidden_size, batch_first=True)
    elif model_type == 'lstm':
        model = nn.LSTM(input_size, hidden_size, batch_first=True)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 创建随机输入
    x = torch.randn(1, seq_len, input_size)

    grad_norms = []

    # 对每个时间步的输出计算其对最终输出的梯度
    for target_t in range(seq_len):
        x_clone = x.clone().requires_grad_(True)

        if model_type == 'rnn':
            outputs, _ = model(x_clone)
        else:
            outputs, _ = model(x_clone)

        # 最终时间步的损失
        loss = outputs[:, -1, :].sum()

        # 反向传播
        model.zero_grad()
        loss.backward(retain_graph=True)

        # 计算输入在target_t时间步的梯度范数
        if x_clone.grad is not None:
            grad_norm = x_clone.grad[:, target_t, :].norm().item()
        else:
            grad_norm = 0.0

        grad_norms.append(grad_norm)

    return grad_norms


def experiment_gradient_vanishing():
    """实验1：梯度消失可视化"""
    print("=" * 60)
    print("实验1：梯度消失可视化")
    print("=" * 60)

    seq_len = 40

    rnn_grads = compute_gradient_norms('rnn', seq_len)
    lstm_grads = compute_gradient_norms('lstm', seq_len)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    t = np.arange(seq_len)

    # RNN梯度图
    ax1 = axes[0]
    ax1.semilogy(t, rnn_grads, color='#e74c3c', linewidth=2, marker='o', markersize=4)
    ax1.fill_between(t, rnn_grads, alpha=0.2, color='#e74c3c')
    ax1.set_title('Simple RNN：梯度消失', fontsize=13, fontweight='bold')
    ax1.set_xlabel('时间步（越左越早）', fontsize=11)
    ax1.set_ylabel('梯度范数（对数尺度）', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(0, seq_len*0.3, alpha=0.1, color='red', label='梯度近乎为零区域')
    ax1.legend(fontsize=10)

    # LSTM梯度图
    ax2 = axes[1]
    ax2.semilogy(t, lstm_grads, color='#3498db', linewidth=2, marker='s', markersize=4)
    ax2.fill_between(t, lstm_grads, alpha=0.2, color='#3498db')
    ax2.set_title('LSTM：更稳定的梯度流', fontsize=13, fontweight='bold')
    ax2.set_xlabel('时间步（越左越早）', fontsize=11)
    ax2.set_ylabel('梯度范数（对数尺度）', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('梯度消失实验：不同时间步位置对最终损失的梯度大小',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('experiment_gradient_vanishing.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 数值分析
    rnn_ratio = min(rnn_grads) / (max(rnn_grads) + 1e-10)
    lstm_ratio = min(lstm_grads) / (max(lstm_grads) + 1e-10)

    print(f"\nRNN  - 最小/最大梯度比: {rnn_ratio:.2e} （严重衰减）")
    print(f"LSTM - 最小/最大梯度比: {lstm_ratio:.2e} （相对稳定）")
    print(f"改善倍数: {lstm_ratio / (rnn_ratio + 1e-10):.1f}x\n")


# ──────────────────────────────────────────────────────────────
# 2. 长序列记忆实验（回声任务）
# ──────────────────────────────────────────────────────────────

def generate_echo_task(batch_size: int, seq_len: int, echo_delay: int,
                        vocab_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    回声任务（Echo Task）：在delay步之后"回声"输入

    输入:  [x_1, x_2, ..., x_n, PAD, PAD, ...]
    目标:  [PAD, PAD, ..., x_1, x_2, ..., x_n]  （延迟echo_delay步）

    要正确完成此任务，模型必须记住echo_delay步之前的输入
    """
    # 有效长度 = seq_len - echo_delay
    effective_len = seq_len - echo_delay

    # 生成随机输入信号
    signal = torch.randint(1, vocab_size + 1, (batch_size, effective_len))
    padding = torch.zeros(batch_size, echo_delay, dtype=torch.long)

    # 输入序列
    inputs = torch.cat([signal, padding], dim=1)

    # 目标序列（延迟的回声）
    targets = torch.cat([padding, signal], dim=1)

    return inputs, targets


class EchoRNN(nn.Module):
    """用于回声任务的RNN/LSTM模型"""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 model_type: str = 'lstm'):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        if model_type == 'rnn':
            self.recurrent = nn.RNN(embed_dim, hidden_size, batch_first=True)
        else:
            self.recurrent = nn.LSTM(embed_dim, hidden_size, batch_first=True)

        self.output_proj = nn.Linear(hidden_size, vocab_size + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        outputs, _ = self.recurrent(embedded)
        logits = self.output_proj(outputs)
        return logits


def train_echo_model(model_type: str, echo_delay: int,
                      seq_len: int = 30, num_steps: int = 2000) -> float:
    """训练回声模型，返回最终准确率"""

    vocab_size = 8
    batch_size = 32

    model = EchoRNN(vocab_size, embed_dim=16, hidden_size=64, model_type=model_type)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding

    model.train()

    for step in range(num_steps):
        inputs, targets = generate_echo_task(batch_size, seq_len, echo_delay, vocab_size)

        logits = model(inputs)  # (batch, seq_len, vocab_size+1)

        # 只计算非padding位置的损失
        loss = criterion(logits.view(-1, vocab_size + 1), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    # 评估
    model.eval()
    with torch.no_grad():
        inputs, targets = generate_echo_task(512, seq_len, echo_delay, vocab_size)
        logits = model(inputs)
        preds = logits.argmax(dim=-1)

        # 只在非padding位置计算准确率
        mask = targets != 0
        correct = ((preds == targets) & mask).sum().item()
        total = mask.sum().item()

        accuracy = correct / total if total > 0 else 0.0

    return accuracy


def experiment_long_range_memory():
    """实验2：测试不同延迟下RNN vs LSTM的记忆能力"""
    print("=" * 60)
    print("实验2：长距离记忆能力对比（回声任务）")
    print("=" * 60)

    echo_delays = [2, 5, 10, 15, 20]
    seq_len = 35  # seq_len > max(echo_delays) + 5

    rnn_accs = []
    lstm_accs = []

    for delay in echo_delays:
        print(f"  测试延迟 = {delay}...")
        rnn_acc = train_echo_model('rnn', delay, seq_len)
        lstm_acc = train_echo_model('lstm', delay, seq_len)
        rnn_accs.append(rnn_acc)
        lstm_accs.append(lstm_acc)
        print(f"    RNN:  {rnn_acc:.3f}  |  LSTM: {lstm_acc:.3f}")

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(echo_delays, rnn_accs, 'o-', color='#e74c3c',
            linewidth=2.5, markersize=10, label='Simple RNN')
    ax.plot(echo_delays, lstm_accs, 's-', color='#3498db',
            linewidth=2.5, markersize=10, label='LSTM')
    ax.axhline(y=1/8, color='gray', linestyle='--', alpha=0.7, label='随机基线 (1/8)')

    ax.set_xlabel('回声延迟（步数）', fontsize=13)
    ax.set_ylabel('准确率', fontsize=13)
    ax.set_title('回声任务：不同延迟下的记忆能力\n（延迟越大，需要记住越久远的信息）',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(echo_delays)

    plt.tight_layout()
    plt.savefig('experiment_memory.png', dpi=150, bbox_inches='tight')
    plt.show()

    return rnn_accs, lstm_accs


# ──────────────────────────────────────────────────────────────
# 主函数：运行所有实验
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("第1章代码实战：序列建模基础实验")
    print("=" * 60 + "\n")

    # 实验1：梯度消失可视化
    experiment_gradient_vanishing()

    # 实验2：长距离记忆能力（注意：训练需要约1-2分钟）
    # rnn_accs, lstm_accs = experiment_long_range_memory()

    print("\n所有实验完成！请查看生成的图表文件。")
    print("  - experiment_gradient_vanishing.png")
    print("  - experiment_memory.png  （取消注释实验2后生成）")
```

---

## 练习题

### 基础题

**练习1.1（基础）**：下列关于RNN的描述，哪些是正确的？（多选）

A. RNN的所有时间步共享相同的权重矩阵
B. RNN可以处理任意长度的输入序列
C. RNN的时间步之间可以完全并行化计算
D. RNN的隐藏状态维度必须等于输入维度
E. 梯度消失是指梯度随时间步反向传播时指数级衰减

---

**练习1.2（基础）**：给定如下LSTM的参数配置：
- 输入维度 $d_x = 128$
- 隐藏状态维度 $d_h = 256$

请计算：

(a) LSTM共有多少可训练参数（不含输出层）？

(b) 如果使用双层双向LSTM（`num_layers=2, bidirectional=True`），总参数量是多少？

(c) 与同规模的简单RNN相比，LSTM参数量是RNN的多少倍？

---

### 中级题

**练习1.3（中级）**：**实现双向LSTM情感分类器**

使用IMDb情感分析数据集（或自定义小型数据集），实现以下要求：

```python
# 要求实现的函数签名
class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout):
        """
        双向LSTM情感分类器
        - 使用预训练词向量初始化embedding（可选）
        - 双向LSTM提取特征
        - 将前向和后向最后隐藏状态拼接后分类
        """
        pass

    def forward(self, x, lengths):
        """
        支持变长序列（使用pack_padded_sequence）
        """
        pass

# 训练目标：在验证集上达到 >85% 准确率
```

提示：使用`nn.utils.rnn.pack_padded_sequence`处理变长序列。

---

**练习1.4（中级）**：**梯度裁剪的重要性实验**

实现以下实验，比较有无梯度裁剪对RNN训练稳定性的影响：

1. 在一个语言模型任务上训练简单RNN
2. 不使用梯度裁剪，记录训练损失曲线（观察损失突然暴增的现象）
3. 使用`torch.nn.utils.clip_grad_norm_`（`max_norm=1.0`），重新训练
4. 绘制两种情况的损失曲线和梯度范数曲线
5. 分析：梯度裁剪在什么时候最有效？它能彻底解决梯度消失问题吗？

---

### 提高题

**练习1.5（提高）**：**从零实现Bahdanau注意力Seq2Seq**

实现一个带Bahdanau注意力的序列到序列模型，并在数字序列排序任务上验证：

- **任务**：输入乱序数字序列（如`[3, 1, 4, 1, 5, 9, 2, 6]`），输出排序后的序列（如`[1, 1, 2, 3, 4, 5, 6, 9]`）
- **编码器**：双向LSTM
- **解码器**：单向LSTM + Bahdanau注意力
- **要求**：
  1. 实现完整的编码器-解码器架构
  2. 实现Bahdanau注意力模块
  3. 实现带Teacher Forcing的训练循环
  4. 可视化测试样本的注意力矩阵，验证模型是否学到了合理的对齐关系
  5. 分析：对于排序任务，理想的注意力矩阵应该是什么形状？实际学到的和理想的有何差异？

```python
# 参考架构
class Encoder(nn.Module): ...
class BahdanauAttention(nn.Module): ...
class Decoder(nn.Module): ...
class Seq2Seq(nn.Module): ...

# 评估标准：序列级准确率 > 90%（序列长度 ≤ 10）
```

---

## 练习答案

### 答案1.1

**正确答案：A、B、E**

- **A（正确）**：RNN的核心特性是参数共享，$W_{hh}$, $W_{xh}$, $b$ 在所有时间步保持不变。这使模型参数量不随序列长度增长。

- **B（正确）**：理论上RNN可以处理任意长度序列，因为它是一个循环结构，可以无限展开。但实践中过长序列会遇到梯度消失问题。

- **C（错误）**：RNN无法并行化。$h_t$ 的计算依赖 $h_{t-1}$，必须按时间步串行执行。这是RNN相较Transformer的主要劣势之一。

- **D（错误）**：隐藏状态维度 $d_h$ 和输入维度 $d_x$ 是独立的超参数，通过 $W_{xh} \in \mathbb{R}^{d_h \times d_x}$ 进行映射。通常 $d_h > d_x$。

- **E（正确）**：梯度消失的准确定义——在BPTT中，连乘项 $\prod_{t} \frac{\partial h_t}{\partial h_{t-1}}$ 的值小于1时，随时间步数增加而指数级趋向0。

---

### 答案1.2

**(a) 单层LSTM的参数量**

LSTM的参数包含四个门（输入门、遗忘门、候选细胞、输出门），每个门都有两个权重矩阵和一个偏置：

$$\text{参数量} = 4 \times [(d_x \times d_h) + (d_h \times d_h) + d_h]$$

代入 $d_x = 128$, $d_h = 256$：

$$= 4 \times [(128 \times 256) + (256 \times 256) + 256]$$
$$= 4 \times [32768 + 65536 + 256]$$
$$= 4 \times 98560$$
$$= 394240 \approx 39.4\text{万参数}$$

**(b) 双层双向LSTM的参数量**

- 第1层：输入维度仍为 $d_x = 128$，但双向意味着有前向和后向两个LSTM
  - 前向：$4 \times (128 \times 256 + 256 \times 256 + 256) = 394240$
  - 后向：同上 $= 394240$

- 第2层：输入维度变为 $2 \times d_h = 512$（第1层前向+后向输出拼接）
  - 前向：$4 \times (512 \times 256 + 256 \times 256 + 256) = 4 \times (131072 + 65536 + 256) = 786944$
  - 后向：同上 $= 786944$

$$\text{总参数量} = 394240 \times 2 + 786944 \times 2 = 788480 + 1573888 = 2362368 \approx 236\text{万参数}$$

**(c) LSTM vs RNN的参数比**

简单RNN的参数量（同规模）：

$$\text{RNN参数量} = (d_x \times d_h) + (d_h \times d_h) + d_h = 128 \times 256 + 256^2 + 256 = 98560$$

$$\text{比率} = \frac{394240}{98560} = 4$$

LSTM的参数量恰好是RNN的**4倍**（因为有4个门，每个门的结构等同于一个RNN）。

---

### 答案1.3

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTMSentiment(nn.Module):
    """双向LSTM情感分类器（参考实现）"""

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,        # 双向
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # 双向LSTM：前向最后隐藏 + 后向最后隐藏 = 2 * hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)  # 二分类：正面/负面
        )

    def forward(self, x, lengths):
        """
        参数:
            x:       (batch, max_seq_len) - 词索引（已填充）
            lengths: (batch,) - 每个样本的真实长度
        """
        # 词嵌入
        embedded = self.dropout(self.embedding(x))  # (batch, seq_len, embed_dim)

        # 使用pack_padded_sequence忽略padding，提高效率
        packed = pack_padded_sequence(embedded, lengths.cpu(),
                                       batch_first=True, enforce_sorted=False)

        # LSTM前向传播
        packed_output, (h_n, c_n) = self.lstm(packed)

        # h_n shape: (num_layers * 2, batch, hidden_size)
        # 取最后一层：前向最后时间步 + 后向最后时间步
        # 前向：h_n[-2]（倒数第2个，因为双向，最后一层的前向）
        # 后向：h_n[-1]（最后一个，最后一层的后向）
        forward_hidden  = h_n[-2]  # (batch, hidden_size)
        backward_hidden = h_n[-1]  # (batch, hidden_size)

        # 拼接双向隐藏状态
        combined = torch.cat([forward_hidden, backward_hidden], dim=-1)
        combined = self.dropout(combined)

        # 分类
        logits = self.classifier(combined)  # (batch, 2)

        return logits


# 训练循环骨架
def train_sentiment(model, train_loader, val_loader, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0

        for texts, lengths, labels in train_loader:
            logits = model(texts, lengths)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        correct = total = 0

        with torch.no_grad():
            for texts, lengths, labels in val_loader:
                logits = model(texts, lengths)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_bilstm.pt')

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Best: {best_val_acc:.4f}")

    return best_val_acc
```

---

### 答案1.4

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def experiment_gradient_clipping():
    """梯度裁剪重要性实验"""

    vocab_size = 50
    embed_dim = 32
    hidden_size = 64
    seq_len = 30
    batch_size = 16
    num_steps = 500

    def make_model():
        return nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            # 注意：nn.Sequential不支持RNN，这里用自定义模型
        )

    class LMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
            self.proj = nn.Linear(hidden_size, vocab_size)

        def forward(self, x):
            emb = self.embed(x)
            out, _ = self.rnn(emb)
            return self.proj(out)

    criterion = nn.CrossEntropyLoss()

    results = {}

    for use_clipping in [False, True]:
        torch.manual_seed(42)
        model = LMModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # SGD更容易爆炸

        losses = []
        grad_norms = []

        for step in range(num_steps):
            # 随机生成语言模型数据
            x = torch.randint(0, vocab_size, (batch_size, seq_len))
            y = torch.randint(0, vocab_size, (batch_size, seq_len))

            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()

            # 计算裁剪前的梯度范数
            total_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            grad_norms.append(total_norm)

            # 条件梯度裁剪
            if use_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            losses.append(loss.item())

        label = '有梯度裁剪' if use_clipping else '无梯度裁剪'
        results[label] = {'losses': losses, 'grad_norms': grad_norms}

        print(f"{label}: 最终损失 = {np.mean(losses[-50:]):.4f}, "
              f"最大梯度范数 = {max(grad_norms):.2f}")

    # 绘制对比图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    colors = {'无梯度裁剪': '#e74c3c', '有梯度裁剪': '#3498db'}

    for label, data in results.items():
        axes[0].plot(data['losses'], label=label, color=colors[label], alpha=0.7)
        axes[1].semilogy(data['grad_norms'], label=label, color=colors[label], alpha=0.7)

    axes[0].set_title('训练损失对比', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Cross Entropy Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title('梯度范数对比（对数尺度）', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('梯度 L2 范数'); axes[1].set_xlabel('训练步数')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gradient_clipping_experiment.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n分析结论：")
    print("• 梯度裁剪有效抑制了梯度爆炸（将梯度范数限制在1.0以内）")
    print("• 梯度裁剪使训练损失更加稳定，避免了突然的损失暴增")
    print("• 梯度裁剪不能解决梯度消失——它只截断过大的梯度，无法放大过小的梯度")
    print("• 对于梯度消失问题，根本解决方案是改变架构（LSTM门控、残差连接、注意力）")


experiment_gradient_clipping()
```

---

### 答案1.5

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt

# ──── 编码器 ────

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        # 将双向输出压缩为单向（供解码器使用）
        self.fc_h = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src):
        """
        src: (batch, src_len)
        返回:
            outputs:    (batch, src_len, hidden*2)  所有编码器输出
            (h_n, c_n): (1, batch, hidden)          压缩后的最终状态
        """
        embedded = self.embedding(src)
        outputs, (h_n, c_n) = self.lstm(embedded)

        # h_n: (2, batch, hidden) → 拼接 → (batch, hidden*2) → 压缩 → (batch, hidden)
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        c_n = torch.cat([c_n[-2], c_n[-1]], dim=1)

        h_n = torch.tanh(self.fc_h(h_n)).unsqueeze(0)  # (1, batch, hidden)
        c_n = torch.tanh(self.fc_c(c_n)).unsqueeze(0)

        return outputs, (h_n, c_n)


# ──── 注意力 ────

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden, decoder_hidden, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(encoder_hidden * 2, attn_dim, bias=False)
        self.W_dec = nn.Linear(decoder_hidden,     attn_dim, bias=False)
        self.v     = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        """
        dec_hidden:  (batch, decoder_hidden)
        enc_outputs: (batch, src_len, encoder_hidden*2)

        返回:
            context: (batch, encoder_hidden*2)
            weights: (batch, src_len)
        """
        src_len = enc_outputs.size(1)
        dec_exp  = dec_hidden.unsqueeze(1).expand(-1, src_len, -1)

        energy = self.v(torch.tanh(
            self.W_enc(enc_outputs) + self.W_dec(dec_exp)
        )).squeeze(-1)  # (batch, src_len)

        weights = F.softmax(energy, dim=-1)  # (batch, src_len)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)

        return context, weights


# ──── 解码器 ────

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_size, encoder_hidden, attn_dim, dropout=0.1):
        super().__init__()
        self.embedding  = nn.Embedding(output_dim, embed_dim)
        self.attention  = BahdanauAttention(encoder_hidden, hidden_size, attn_dim)
        self.lstm       = nn.LSTMCell(embed_dim + encoder_hidden * 2, hidden_size)
        self.output_proj = nn.Linear(hidden_size + encoder_hidden * 2 + embed_dim, output_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward_step(self, tgt_token, h, c, enc_outputs):
        """
        单步解码
        tgt_token: (batch,)
        h, c:      (batch, hidden)
        """
        embedded = self.dropout(self.embedding(tgt_token))   # (batch, embed_dim)
        context, attn_weights = self.attention(h, enc_outputs)

        lstm_input = torch.cat([embedded, context], dim=1)   # (batch, embed+enc_hidden*2)
        h_new, c_new = self.lstm(lstm_input, (h, c))

        prediction_input = torch.cat([h_new, context, embedded], dim=1)
        logit = self.output_proj(prediction_input)            # (batch, output_dim)

        return logit, h_new, c_new, attn_weights


# ──── Seq2Seq ────

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, tgt_vocab_size, sos_idx, eos_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_vocab_size = tgt_vocab_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        训练时前向传播（支持Teacher Forcing）
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        """
        batch_size, tgt_len = tgt.shape

        enc_outputs, (h, c) = self.encoder(src)
        h = h.squeeze(0)  # (batch, hidden)
        c = c.squeeze(0)

        # 解码器的第一个输入是 <SOS> token
        dec_input = torch.full((batch_size,), self.sos_idx,
                               dtype=torch.long, device=src.device)

        all_logits = []
        all_attn   = []

        for t in range(tgt_len):
            logit, h, c, attn = self.decoder.forward_step(dec_input, h, c, enc_outputs)
            all_logits.append(logit)
            all_attn.append(attn)

            # Teacher forcing
            use_teacher = random.random() < teacher_forcing_ratio
            dec_input = tgt[:, t] if use_teacher else logit.argmax(dim=-1)

        logits = torch.stack(all_logits, dim=1)   # (batch, tgt_len, vocab)
        attn   = torch.stack(all_attn,   dim=1)   # (batch, tgt_len, src_len)

        return logits, attn


# ──── 排序任务数据生成 ────

def generate_sort_data(n_samples=5000, min_len=4, max_len=10, max_val=20):
    """生成排序任务数据集"""
    SOS, EOS, PAD = 0, 1, 2
    offset = 3  # 真实数字从3开始编码

    data = []
    for _ in range(n_samples):
        length = random.randint(min_len, max_len)
        nums = [random.randint(0, max_val - 1) for _ in range(length)]
        src  = [n + offset for n in nums]
        tgt  = [SOS] + sorted([n + offset for n in nums]) + [EOS]
        data.append((src, tgt))

    return data, SOS, EOS, PAD


# 注意力可视化
def visualize_sort_attention(model, src_seq, max_val=20):
    """可视化单个样本的注意力矩阵"""
    SOS, EOS, PAD = 0, 1, 2
    offset = 3

    model.eval()
    with torch.no_grad():
        src = torch.tensor([src_seq]).long()

        enc_outputs, (h, c) = model.encoder(src)
        h = h.squeeze(0); c = c.squeeze(0)

        dec_input = torch.tensor([SOS])
        attn_weights = []
        outputs = []

        for _ in range(len(src_seq) + 2):
            logit, h, c, attn = model.decoder.forward_step(dec_input, h, c, enc_outputs)
            pred = logit.argmax(dim=-1)
            attn_weights.append(attn.squeeze(0).numpy())
            outputs.append(pred.item())
            dec_input = pred
            if pred.item() == EOS:
                break

    # 转换为可读标签
    src_labels = [str(s - offset) for s in src_seq]
    tgt_labels = [str(o - offset) if o >= offset else ('<SOS>' if o == SOS else '<EOS>')
                  for o in outputs]

    attn_matrix = np.array(attn_weights)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(src_labels))); ax.set_xticklabels(src_labels, fontsize=12)
    ax.set_yticks(range(len(tgt_labels))); ax.set_yticklabels(tgt_labels, fontsize=12)
    ax.set_xlabel('输入序列（乱序）', fontsize=12)
    ax.set_ylabel('输出序列（排序后）', fontsize=12)
    ax.set_title(f'排序任务注意力矩阵\n输入: {src_labels} → 输出: {tgt_labels}',
                 fontsize=12, fontweight='bold')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('sort_attention.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"输入（乱序）: {src_labels}")
    print(f"预测（排序）: {[l for l in tgt_labels if l not in ['<SOS>', '<EOS>']]}")
    print("\n分析：理想的注意力矩阵中，每行应有一个高权重位置，")
    print("对应输出数字在输入中的位置——即模型学会了'指向'每个数字的原始位置。")
```

---

## 延伸阅读

1. **Hochreiter & Schmidhuber (1997)** - "Long Short-Term Memory" - LSTM原始论文
2. **Bahdanau et al. (2015)** - "Neural Machine Translation by Jointly Learning to Align and Translate" - 注意力机制首次提出
3. **Vaswani et al. (2017)** - "Attention Is All You Need" - Transformer论文（第2章将详细介绍）
4. **Karpathy's blog** - "The Unreasonable Effectiveness of Recurrent Neural Networks" - RNN的直观介绍（附可视化）
5. **Olah's blog** - "Understanding LSTM Networks" - LSTM最佳图文教程

---

**下一章预告**：第2章将深入Transformer的核心——**自注意力机制（Self-Attention）**。我们将从数学原理到代码实现，完整推导缩放点积注意力和多头注意力，并可视化GPT/BERT的注意力模式。
