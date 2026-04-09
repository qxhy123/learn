# 第23章：序列模型与注意力机制

## 学习目标

完成本章学习后，你将能够：

1. 理解循环神经网络（RNN）的工作原理，并掌握梯度消失/爆炸问题的解决思路
2. 熟练使用 LSTM 和 GRU 处理长序列依赖问题
3. 构建双向 RNN 和多层 RNN 网络结构
4. 理解注意力机制（Attention）的数学原理并用 PyTorch 实现
5. 掌握 Transformer 的核心组件：自注意力（Self-Attention）和位置编码

---

## 23.1 循环神经网络（RNN）

### 23.1.1 为什么需要序列模型

在处理文本、语音、时间序列等数据时，数据的顺序至关重要。传统的全连接网络将每个输入独立处理，无法捕捉序列中的上下文关系。

例如，理解"我去银行取钱"中的"银行"，需要结合上下文；预测股票明天的价格，需要参考历史走势。这类问题需要**序列模型**。

### 23.1.2 RNN 基本结构

RNN 的核心思想是引入**隐藏状态（hidden state）**，将前一时刻的信息传递给下一时刻：

```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

其中：
- `x_t`：t 时刻的输入
- `h_t`：t 时刻的隐藏状态（"记忆"）
- `y_t`：t 时刻的输出
- `W_hh, W_xh, W_hy`：可学习的权重矩阵

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 手动实现一个简单的 RNN 前向传播
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
        self.b_h = np.zeros((1, hidden_size))
        self.b_y = np.zeros((1, output_size))

    def forward(self, inputs):
        """
        inputs: list of numpy arrays, shape (1, input_size)
        返回每个时间步的输出和隐藏状态
        """
        h = np.zeros((1, self.W_hh.shape[0]))  # 初始隐藏状态
        outputs = []
        hidden_states = [h.copy()]

        for x in inputs:
            # 更新隐藏状态
            h = np.tanh(x @ self.W_xh + h @ self.W_hh + self.b_h)
            # 计算输出
            y = h @ self.W_hy + self.b_y
            outputs.append(y)
            hidden_states.append(h.copy())

        return outputs, hidden_states


# 使用示例：处理长度为 5 的序列
input_size = 4
hidden_size = 8
output_size = 2
seq_len = 5

rnn = SimpleRNN(input_size, hidden_size, output_size)
inputs = [np.random.randn(1, input_size) for _ in range(seq_len)]
outputs, hidden_states = rnn.forward(inputs)

print(f"输入序列长度: {seq_len}")
print(f"每个时间步输出形状: {outputs[0].shape}")
print(f"隐藏状态数量: {len(hidden_states)}")
```

### 23.1.3 使用 PyTorch 实现 RNN

```python
import torch
import torch.nn as nn

# PyTorch 内置 RNN
class TextClassifierRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(
            input_size=embed_size,
            hidden_size=hidden_size,
            batch_first=True  # 输入形状: (batch, seq, feature)
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embed = self.embedding(x)           # (batch, seq, embed)
        output, hidden = self.rnn(embed)    # output: (batch, seq, hidden)
                                            # hidden: (1, batch, hidden)
        # 取最后时刻的隐藏状态用于分类
        last_hidden = hidden.squeeze(0)     # (batch, hidden)
        out = self.fc(last_hidden)          # (batch, num_classes)
        return out


# 创建模型并测试
vocab_size = 1000
embed_size = 64
hidden_size = 128
num_classes = 5
batch_size = 32
seq_len = 20

model = TextClassifierRNN(vocab_size, embed_size, hidden_size, num_classes)
x = torch.randint(0, vocab_size, (batch_size, seq_len))
output = model(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")  # (32, 5)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

### 23.1.4 梯度消失与梯度爆炸

RNN 在处理长序列时面临严重问题：**梯度消失**和**梯度爆炸**。

**原因分析：**

反向传播时，梯度需要通过时间步链式传播。设每步的梯度因子为 `λ`：
- 若 `|λ| < 1`：梯度随时间步指数递减 → **梯度消失**，早期信息被遗忘
- 若 `|λ| > 1`：梯度随时间步指数增长 → **梯度爆炸**，训练不稳定

```python
# 演示梯度消失问题
def demonstrate_vanishing_gradient():
    seq_len = 50
    # 模拟 tanh 激活函数的梯度（tanh'(x) ≤ 1）
    gradient_factor = 0.9  # 假设每步梯度缩小为 0.9

    gradients = [gradient_factor ** t for t in range(seq_len)]

    plt.figure(figsize=(10, 4))
    plt.plot(gradients)
    plt.xlabel("距离当前时刻的步数")
    plt.ylabel("梯度大小")
    plt.title("RNN 中的梯度消失：距离越远，梯度越小")
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("vanishing_gradient.png", dpi=100)
    plt.show()

    print(f"第 1 步的梯度: {gradients[0]:.4f}")
    print(f"第 10 步的梯度: {gradients[9]:.6f}")
    print(f"第 50 步的梯度: {gradients[49]:.10f}")

demonstrate_vanishing_gradient()
```

**缓解梯度爆炸：梯度裁剪**

```python
# 梯度裁剪（Gradient Clipping）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环中
def train_step(model, x, y, optimizer, max_grad_norm=1.0):
    optimizer.zero_grad()
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()

    # 梯度裁剪：将梯度范数限制在 max_grad_norm 以内
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()
    return loss.item()
```

---

## 23.2 LSTM 与 GRU

### 23.2.1 LSTM：长短期记忆网络

LSTM（Long Short-Term Memory）通过引入**门控机制**和**细胞状态**解决梯度消失问题。

**LSTM 的四个核心组件：**

| 组件 | 作用 | 公式 |
|------|------|------|
| 遗忘门 (Forget Gate) | 决定遗忘多少旧信息 | `f_t = σ(W_f·[h_{t-1}, x_t] + b_f)` |
| 输入门 (Input Gate) | 决定写入多少新信息 | `i_t = σ(W_i·[h_{t-1}, x_t] + b_i)` |
| 候选细胞 (Cell Candidate) | 生成候选新信息 | `C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)` |
| 输出门 (Output Gate) | 决定输出多少信息 | `o_t = σ(W_o·[h_{t-1}, x_t] + b_o)` |

细胞状态更新：
```
C_t = f_t * C_{t-1} + i_t * C̃_t
h_t = o_t * tanh(C_t)
```

```python
# 手动实现 LSTM 单元（用于理解原理）
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 将四个门的矩阵合并为一个大矩阵，提升效率
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h_prev, c_prev):
        # 拼接输入和上一时刻隐藏状态
        combined = torch.cat([x, h_prev], dim=1)  # (batch, input+hidden)

        # 一次性计算四个门
        gates = self.linear(combined)  # (batch, 4*hidden)

        # 拆分四个门
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        # 应用激活函数
        i = torch.sigmoid(i_gate)   # 输入门
        f = torch.sigmoid(f_gate)   # 遗忘门
        g = torch.tanh(g_gate)      # 候选细胞
        o = torch.sigmoid(o_gate)   # 输出门

        # 更新细胞状态和隐藏状态
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


# 测试 LSTM 单元
batch_size = 4
input_size = 10
hidden_size = 20

lstm_cell = LSTMCell(input_size, hidden_size)
x = torch.randn(batch_size, input_size)
h = torch.zeros(batch_size, hidden_size)
c = torch.zeros(batch_size, hidden_size)

h_new, c_new = lstm_cell(x, h, c)
print(f"新隐藏状态形状: {h_new.shape}")   # (4, 20)
print(f"新细胞状态形状: {c_new.shape}")   # (4, 20)
```

### 23.2.2 使用 PyTorch LSTM 处理时间序列

```python
class LSTMForecaster(nn.Module):
    """使用 LSTM 进行时间序列预测"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM 前向传播
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out: (batch, seq_len, hidden_size)

        # 取最后时刻输出
        out = self.dropout(out[:, -1, :])  # (batch, hidden_size)
        out = self.fc(out)                  # (batch, output_size)
        return out


# 生成正弦波数据并训练
def generate_sine_data(n_samples=1000, seq_len=50, pred_len=1):
    t = np.linspace(0, 4 * np.pi, n_samples + seq_len)
    data = np.sin(t) + 0.1 * np.random.randn(len(t))

    X, y = [], []
    for i in range(n_samples):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_len])

    return np.array(X), np.array(y)


X, y = generate_sine_data()
X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # (1000, 50, 1)
y_tensor = torch.FloatTensor(y)                  # (1000, 1)

# 创建模型
model = LSTMForecaster(
    input_size=1,
    hidden_size=64,
    num_layers=2,
    output_size=1,
    dropout=0.2
)

# 简单训练循环
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

model.train()
for epoch in range(5):
    optimizer.zero_grad()
    output = model(X_tensor[:100])
    loss = criterion(output, y_tensor[:100])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")
```

### 23.2.3 GRU：门控循环单元

GRU（Gated Recurrent Unit）是 LSTM 的简化版本，将遗忘门和输入门合并为**更新门**，去掉了独立的细胞状态：

```
z_t = σ(W_z·[h_{t-1}, x_t])   # 更新门
r_t = σ(W_r·[h_{t-1}, x_t])   # 重置门
h̃_t = tanh(W·[r_t*h_{t-1}, x_t])  # 候选隐藏状态
h_t = (1 - z_t)*h_{t-1} + z_t*h̃_t  # 最终隐藏状态
```

```python
# LSTM vs GRU 对比
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embed = self.embedding(x)
        output, hidden = self.gru(embed)
        # GRU 只有一个隐藏状态（不像 LSTM 有 h 和 c）
        out = self.fc(hidden.squeeze(0))
        return out


# 参数量对比
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

vocab_size, embed_size, hidden_size, num_classes = 5000, 128, 256, 10

lstm_model = TextClassifierRNN(vocab_size, embed_size, hidden_size, num_classes)
# 重新定义一个用LSTM的版本
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        embed = self.embedding(x)
        _, (hidden, _) = self.lstm(embed)
        return self.fc(hidden.squeeze(0))

gru_model = GRUClassifier(vocab_size, embed_size, hidden_size, num_classes)
lstm_model2 = LSTMClassifier(vocab_size, embed_size, hidden_size, num_classes)

print(f"GRU  参数量: {count_params(gru_model):,}")
print(f"LSTM 参数量: {count_params(lstm_model2):,}")
print(f"GRU 比 LSTM 少约 {(1 - count_params(gru_model)/count_params(lstm_model2))*100:.1f}% 参数")
```

**LSTM vs GRU 选择建议：**

| 特点 | LSTM | GRU |
|------|------|-----|
| 参数量 | 多（4个门） | 少（3个门） |
| 训练速度 | 较慢 | 较快 |
| 长序列能力 | 更强 | 接近 |
| 适用场景 | 极长序列、复杂任务 | 大多数序列任务 |

---

## 23.3 双向 RNN 与多层 RNN

### 23.3.1 双向 RNN

单向 RNN 只能利用过去的信息。**双向 RNN** 同时从前向后和从后向前处理序列，在每个时刻结合两个方向的信息：

```
前向：h_t→ = RNN(x_t, h_{t-1}→)
后向：h_t← = RNN(x_t, h_{t+1}←)
输出：h_t = [h_t→; h_t←]  （拼接）
```

```python
class BiLSTMClassifier(nn.Module):
    """双向 LSTM 文本分类"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.bilstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True   # 关键参数
        )
        # 双向：输出维度是 hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embed = self.dropout(self.embedding(x))
        # output: (batch, seq, hidden*2)  <-- 前后向拼接
        # hidden: (2, batch, hidden)  <-- 2 = num_directions
        output, (hidden, cell) = self.bilstm(embed)

        # 拼接最后时刻的前向和后向隐藏状态
        # hidden[0]: 前向最后隐藏状态
        # hidden[1]: 后向最后隐藏状态
        last_hidden = torch.cat([hidden[0], hidden[1]], dim=1)  # (batch, hidden*2)
        out = self.dropout(last_hidden)
        return self.fc(out)


# 测试
model = BiLSTMClassifier(5000, 128, 64, 5)
x = torch.randint(1, 5000, (16, 30))  # batch=16, seq_len=30
out = model(x)
print(f"双向 LSTM 输出形状: {out.shape}")  # (16, 5)
```

### 23.3.2 多层 RNN（堆叠 RNN）

**多层 RNN** 将多个 RNN 层垂直堆叠，下层的输出作为上层的输入，学习更抽象的特征：

```python
class DeepBiLSTM(nn.Module):
    """多层双向 LSTM，带残差连接"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size * 2
            self.layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    bidirectional=True
                )
            )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq, input_size)
        out = x
        for i, lstm in enumerate(self.layers):
            out, _ = lstm(out)
            if i < len(self.layers) - 1:
                out = self.dropout(out)
        # 取序列最后一步
        last = out[:, -1, :]  # (batch, hidden*2)
        return self.fc(last)


# 验证多层架构
model = DeepBiLSTM(
    input_size=64,
    hidden_size=128,
    num_layers=3,
    num_classes=10
)
x = torch.randn(8, 50, 64)  # batch=8, seq=50, feature=64
out = model(x)
print(f"3层双向LSTM输出: {out.shape}")  # (8, 10)
print(f"参数总量: {sum(p.numel() for p in model.parameters()):,}")
```

### 23.3.3 处理变长序列：PackedSequence

实际中序列长度不同，需要填充（padding）。PyTorch 提供 `pack_padded_sequence` 高效处理：

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EfficientLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        embed = self.embedding(x)

        # 打包变长序列（忽略 padding 位置的计算）
        packed = pack_padded_sequence(
            embed, lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, (hidden, _) = self.lstm(packed)

        # 解包（恢复 padding）
        output, _ = pad_packed_sequence(packed_out, batch_first=True)

        return self.fc(hidden.squeeze(0))


# 模拟变长序列
vocab_size = 1000
batch_size = 4
max_len = 20

# 实际长度各不相同
lengths = torch.tensor([20, 15, 10, 8])
x = torch.randint(1, vocab_size, (batch_size, max_len))
# 用 0 填充短序列
for i, l in enumerate(lengths):
    x[i, l:] = 0

model = EfficientLSTM(vocab_size, 64, 128, 5)
out = model(x, lengths)
print(f"变长序列分类输出: {out.shape}")  # (4, 5)
```

---

## 23.4 注意力机制（Attention）

### 23.4.1 注意力机制的直觉

传统 Seq2Seq 模型将整个源序列压缩为一个固定长度的向量，这是性能瓶颈所在。**注意力机制**允许解码器在每步动态"关注"源序列的不同部分。

**类比：** 翻译长句时，翻译每个词时你会回头看原文中最相关的部分，而非死记整句话。

### 23.4.2 注意力分数计算

注意力机制的三要素：
- **Query (Q)**：当前解码器状态（"我想找什么"）
- **Key (K)**：编码器各时刻状态（"每个位置是什么"）
- **Value (V)**：编码器各时刻状态（"每个位置的内容"）

计算流程：
```
1. 计算相似度（注意力分数）: score(Q, K_i) = Q · K_i  或  Q^T W K_i
2. 归一化: α_i = softmax(score_i)
3. 加权求和: context = Σ α_i * V_i
```

```python
class BahdanauAttention(nn.Module):
    """Bahdanau（加法）注意力机制"""
    def __init__(self, encoder_hidden, decoder_hidden, attention_size):
        super().__init__()
        self.W_encoder = nn.Linear(encoder_hidden, attention_size, bias=False)
        self.W_decoder = nn.Linear(decoder_hidden, attention_size, bias=False)
        self.v = nn.Linear(attention_size, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: (batch, src_len, encoder_hidden)
        decoder_hidden:  (batch, decoder_hidden)
        返回: context (batch, encoder_hidden), attention_weights (batch, src_len)
        """
        # 扩展 decoder_hidden 以便广播
        decoder_hidden = decoder_hidden.unsqueeze(1)  # (batch, 1, decoder_hidden)

        # 计算注意力能量
        energy = torch.tanh(
            self.W_encoder(encoder_outputs) +  # (batch, src_len, attn)
            self.W_decoder(decoder_hidden)      # (batch, 1, attn) → 广播
        )  # (batch, src_len, attn)

        # 计算注意力分数
        attention = self.v(energy).squeeze(-1)  # (batch, src_len)
        attention_weights = torch.softmax(attention, dim=1)  # 归一化

        # 加权求和得到上下文向量
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, src_len)
            encoder_outputs                   # (batch, src_len, encoder_hidden)
        ).squeeze(1)  # (batch, encoder_hidden)

        return context, attention_weights


# 测试注意力机制
batch_size = 4
src_len = 10
encoder_hidden = 256
decoder_hidden = 256
attention_size = 128

attention = BahdanauAttention(encoder_hidden, decoder_hidden, attention_size)
encoder_outputs = torch.randn(batch_size, src_len, encoder_hidden)
decoder_state = torch.randn(batch_size, decoder_hidden)

context, weights = attention(encoder_outputs, decoder_state)
print(f"上下文向量形状: {context.shape}")      # (4, 256)
print(f"注意力权重形状: {weights.shape}")       # (4, 10)
print(f"注意力权重之和: {weights.sum(dim=1)}")  # 应为全 1.0
```

### 23.4.3 缩放点积注意力

Transformer 使用的**缩放点积注意力（Scaled Dot-Product Attention）**：

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

除以 `√d_k` 是为了防止点积结果过大导致 softmax 梯度消失。

```python
class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query: (batch, heads, seq_q, d_k)
        key:   (batch, heads, seq_k, d_k)
        value: (batch, heads, seq_v, d_v)
        mask:  (batch, 1, 1, seq_k) 或 (batch, 1, seq_q, seq_k)
        """
        d_k = query.size(-1)

        # 计算注意力分数: Q·K^T / √d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        # scores: (batch, heads, seq_q, seq_k)

        # 应用掩码（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax 归一化
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, value)
        # output: (batch, heads, seq_q, d_v)

        return output, attention_weights


# 测试
sdp_attn = ScaledDotProductAttention(dropout=0.1)
batch, heads, seq_len, d_k = 2, 4, 10, 64

Q = torch.randn(batch, heads, seq_len, d_k)
K = torch.randn(batch, heads, seq_len, d_k)
V = torch.randn(batch, heads, seq_len, d_k)

output, weights = sdp_attn(Q, K, V)
print(f"注意力输出形状: {output.shape}")  # (2, 4, 10, 64)
print(f"注意力权重形状: {weights.shape}") # (2, 4, 10, 10)
```

---

## 23.5 Transformer 基础

### 23.5.1 多头注意力（Multi-Head Attention）

**多头注意力**将 Q、K、V 投影到多个子空间并行计算注意力，使模型能从不同角度关注序列信息：

```python
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q、K、V 的投影矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def split_heads(self, x):
        """将最后一维拆分为 (num_heads, d_k)"""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, heads, seq, d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        residual = query  # 残差连接

        # 线性投影并拆分多头
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # 并行计算多头注意力
        x, attention_weights = self.attention(Q, K, V, mask)
        # x: (batch, heads, seq, d_k)

        # 合并多头
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        # x: (batch, seq, d_model)

        # 输出投影
        x = self.W_o(x)

        # 残差连接 + 层归一化
        x = self.layer_norm(residual + self.dropout(x))
        return x, attention_weights


# 测试多头注意力
d_model = 512
num_heads = 8
seq_len = 20
batch_size = 4

mha = MultiHeadAttention(d_model, num_heads)
x = torch.randn(batch_size, seq_len, d_model)
out, weights = mha(x, x, x)  # 自注意力：Q=K=V=x

print(f"多头注意力输出: {out.shape}")    # (4, 20, 512)
print(f"注意力权重:     {weights.shape}") # (4, 8, 20, 20)
```

### 23.5.2 位置编码（Positional Encoding）

Transformer 没有循环结构，无法感知位置信息，因此需要显式添加**位置编码**：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

```python
class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # 频率系数：1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维：sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维：cos

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 不参与梯度计算

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 可视化位置编码
def visualize_positional_encoding():
    d_model = 128
    max_len = 50
    pe_module = PositionalEncoding(d_model, max_len)
    pe_matrix = pe_module.pe.squeeze(0).numpy()  # (max_len, d_model)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(pe_matrix, aspect='auto', cmap='RdBu')
    plt.colorbar()
    plt.xlabel("编码维度")
    plt.ylabel("序列位置")
    plt.title("位置编码矩阵热图")

    plt.subplot(1, 2, 2)
    for dim in [0, 1, 4, 8, 16]:
        plt.plot(pe_matrix[:, dim], label=f"维度 {dim}")
    plt.xlabel("序列位置")
    plt.ylabel("编码值")
    plt.title("不同维度的位置编码波形")
    plt.legend()

    plt.tight_layout()
    plt.savefig("positional_encoding.png", dpi=100)
    plt.show()

visualize_positional_encoding()
```

### 23.5.3 前馈网络（Feed-Forward Network）

Transformer 中每个注意力层后面都有一个位置级别的前馈网络：

```python
class PositionWiseFeedForward(nn.Module):
    """位置级前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc2(self.dropout(self.activation(self.fc1(x))))
        return self.layer_norm(residual + self.dropout(out))


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器层 = 多头注意力 + 前馈网络"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, src_mask=None):
        x, attention = self.self_attention(x, x, x, src_mask)
        x = self.feed_forward(x)
        return x, attention


class TransformerEncoder(nn.Module):
    """完整的 Transformer 编码器"""
    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        # 词嵌入 + 位置编码（乘以 √d_model 放大嵌入）
        x = self.pos_encoding(self.embedding(src) * (self.d_model ** 0.5))

        attention_maps = []
        for layer in self.layers:
            x, attn = layer(x, src_mask)
            attention_maps.append(attn)

        return x, attention_maps


# 测试完整 Transformer 编码器
encoder = TransformerEncoder(
    vocab_size=10000,
    d_model=256,
    num_heads=8,
    d_ff=1024,
    num_layers=4,
    dropout=0.1
)

src = torch.randint(1, 10000, (4, 30))  # batch=4, seq=30
encoded, attentions = encoder(src)

print(f"编码器输出形状: {encoded.shape}")           # (4, 30, 256)
print(f"注意力层数: {len(attentions)}")             # 4
print(f"第1层注意力形状: {attentions[0].shape}")    # (4, 8, 30, 30)
print(f"参数总量: {sum(p.numel() for p in encoder.parameters()):,}")
```

---

## 本章小结

| 模型 | 核心特点 | 优势 | 局限 |
|------|----------|------|------|
| **RNN** | 循环隐藏状态 | 简单，适合短序列 | 梯度消失，长依赖弱 |
| **LSTM** | 遗忘/输入/输出门 + 细胞状态 | 长序列能力强 | 参数多，速度慢 |
| **GRU** | 更新/重置门 | 速度快，参数少 | 稍弱于 LSTM |
| **双向 RNN** | 双向上下文 | 全文感知能力强 | 不适合自回归生成 |
| **注意力机制** | 动态关注相关位置 | 解决信息瓶颈 | 需配合 RNN 编码器 |
| **Transformer** | 全并行自注意力 | 最强，GPU 友好 | 计算复杂度 O(n²) |

**关键公式总结：**

```
LSTM:  C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
GRU:   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
注意力: Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

---

## 深度学习应用：简单机器翻译（Seq2Seq + Attention）

下面实现一个完整的序列到序列（Seq2Seq）翻译模型，带有注意力机制：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# ==================== 数据准备 ====================

# 简单的英-中词汇表（演示用）
EN_VOCAB = {
    '<pad>': 0, '<sos>': 1, '<eos>': 2,
    'i': 3, 'love': 4, 'you': 5, 'hate': 6,
    'eat': 7, 'apple': 8, 'like': 9, 'cat': 10,
    'dog': 11, 'am': 12, 'happy': 13, 'sad': 14, 'go': 15
}

ZH_VOCAB = {
    '<pad>': 0, '<sos>': 1, '<eos>': 2,
    '我': 3, '爱': 4, '你': 5, '恨': 6,
    '吃': 7, '苹果': 8, '喜欢': 9, '猫': 10,
    '狗': 11, '是': 12, '开心': 13, '伤心': 14, '走': 15
}

EN_VOCAB_SIZE = len(EN_VOCAB)
ZH_VOCAB_SIZE = len(ZH_VOCAB)
EN_ID2WORD = {v: k for k, v in EN_VOCAB.items()}
ZH_ID2WORD = {v: k for k, v in ZH_VOCAB.items()}

# 训练数据（英文ID序列，中文ID序列）
TRAIN_DATA = [
    ([3, 4, 5], [3, 4, 5]),      # i love you -> 我 爱 你
    ([3, 9, 10], [3, 9, 10]),    # i like cat -> 我 喜欢 猫
    ([3, 7, 8], [3, 7, 8]),      # i eat apple -> 我 吃 苹果
    ([3, 6, 11], [3, 6, 11]),    # i hate dog -> 我 恨 狗
    ([3, 12, 13], [3, 12, 13]),  # i am happy -> 我 是 开心
]


# ==================== 编码器 ====================

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, embed)
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: (batch, src_len, hidden) ← 所有时刻的输出，用于注意力
        # hidden:  (num_layers, batch, hidden)
        return outputs, hidden, cell


# ==================== 注意力机制 ====================

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden:  (batch, hidden)
        encoder_outputs: (batch, src_len, hidden)
        """
        src_len = encoder_outputs.size(1)

        # 扩展解码器隐藏状态以匹配 src_len
        hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # (batch, src_len, hidden)

        # 拼接并计算能量
        energy = torch.tanh(
            self.attn(torch.cat([hidden_expanded, encoder_outputs], dim=2))
        )  # (batch, src_len, hidden)

        attention = self.v(energy).squeeze(2)  # (batch, src_len)
        return torch.softmax(attention, dim=1)   # 归一化


# ==================== 解码器 ====================

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attention = Attention(hidden_size)
        # 输入 = 当前词嵌入 + 注意力上下文
        self.lstm = nn.LSTM(
            embed_size + hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_size * 2 + embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_token, hidden, cell, encoder_outputs):
        """
        tgt_token:       (batch,)  当前输入词
        hidden:          (num_layers, batch, hidden)
        cell:            (num_layers, batch, hidden)
        encoder_outputs: (batch, src_len, hidden)
        """
        tgt_token = tgt_token.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(tgt_token))  # (batch, 1, embed)

        # 计算注意力权重（用最后一层 hidden）
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        # attn_weights: (batch, src_len)

        # 计算上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        # context: (batch, 1, hidden)

        # 拼接嵌入和上下文，输入 LSTM
        lstm_input = torch.cat([embedded, context], dim=2)  # (batch, 1, embed+hidden)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: (batch, 1, hidden)

        # 输出预测
        embedded = embedded.squeeze(1)          # (batch, embed)
        output = output.squeeze(1)              # (batch, hidden)
        context = context.squeeze(1)            # (batch, hidden)

        prediction = self.fc_out(
            torch.cat([output, context, embedded], dim=1)
        )  # (batch, vocab_size)

        return prediction, hidden, cell, attn_weights


# ==================== Seq2Seq 整体模型 ====================

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        teacher_forcing_ratio: 使用真实标签作为下一步输入的概率
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.vocab_size

        # 存储每步的输出
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # 编码
        encoder_outputs, hidden, cell = self.encoder(src)

        # 第一个解码器输入：<sos> 标记
        decoder_input = tgt[:, 0]  # (batch,)

        for t in range(1, tgt_len):
            output, hidden, cell, _ = self.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            outputs[:, t, :] = output

            # Teacher Forcing：决定下一步使用预测值还是真实值
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=1)  # 贪心预测
            decoder_input = tgt[:, t] if teacher_force else top1

        return outputs


# ==================== 训练 ====================

def collate_fn(batch):
    """将变长序列批处理"""
    src_seqs, tgt_seqs = zip(*batch)
    src_max = max(len(s) for s in src_seqs)
    tgt_max = max(len(t) for t in tgt_seqs) + 2  # +2 for <sos> and <eos>

    src_padded = torch.zeros(len(src_seqs), src_max, dtype=torch.long)
    tgt_padded = torch.zeros(len(tgt_seqs), tgt_max, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_seqs, tgt_seqs)):
        src_padded[i, :len(src)] = torch.tensor(src)
        tgt_padded[i, 0] = ZH_VOCAB['<sos>']
        tgt_padded[i, 1:len(tgt)+1] = torch.tensor(tgt)
        tgt_padded[i, len(tgt)+1] = ZH_VOCAB['<eos>']

    return src_padded, tgt_padded


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 超参数
EMBED_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.1
LEARNING_RATE = 0.001
N_EPOCHS = 30

# 构建模型
encoder = Encoder(EN_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
decoder = Decoder(ZH_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

# 权重初始化
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=ZH_VOCAB['<pad>'])

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 训练循环
model.train()
for epoch in range(N_EPOCHS):
    total_loss = 0
    for src_seq, tgt_seq in TRAIN_DATA:
        # 简单起见，每次用单条数据（batch_size=1）
        src_tensor = torch.tensor([src_seq], dtype=torch.long).to(device)
        tgt_ids = [ZH_VOCAB['<sos>']] + tgt_seq + [ZH_VOCAB['<eos>']]
        tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)

        optimizer.zero_grad()
        output = model(src_tensor, tgt_tensor, teacher_forcing_ratio=0.5)
        # output: (1, tgt_len, vocab_size)

        # 忽略第一个 <sos>，从第1步开始计算损失
        output_flat = output[:, 1:, :].reshape(-1, ZH_VOCAB_SIZE)
        tgt_flat = tgt_tensor[:, 1:].reshape(-1)
        loss = criterion(output_flat, tgt_flat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:2d}/{N_EPOCHS}], Loss: {total_loss/len(TRAIN_DATA):.4f}")


# ==================== 推理（贪心解码） ====================

def translate(model, src_seq, max_len=10):
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor([src_seq], dtype=torch.long).to(device)
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

        decoder_input = torch.tensor([ZH_VOCAB['<sos>']], dtype=torch.long).to(device)
        translated = []
        attention_maps = []

        for _ in range(max_len):
            output, hidden, cell, attn = model.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            attention_maps.append(attn.squeeze(0).cpu().numpy())
            pred_token = output.argmax(dim=1).item()

            if pred_token == ZH_VOCAB['<eos>']:
                break

            translated.append(ZH_ID2WORD[pred_token])
            decoder_input = torch.tensor([pred_token], dtype=torch.long).to(device)

    return translated, attention_maps


# 翻译测试
test_sentences = [
    ([3, 4, 5], "i love you"),
    ([3, 9, 10], "i like cat"),
    ([3, 7, 8], "i eat apple"),
]

print("\n=== 翻译测试 ===")
for src_ids, src_text in test_sentences:
    translation, _ = translate(model, src_ids)
    print(f"英文: {src_text}")
    print(f"中文: {''.join(translation)}")
    print()
```

---

## 练习题

### 基础题

**练习 1：** 修改下面的代码，为 `TextClassifierRNN` 添加 Dropout 正则化，并说明在哪些位置添加最合适：

```python
class TextClassifierRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    # 请修改此类，在合适位置添加 Dropout
```

**练习 2：** 给定如下 GRU 模型，计算并解释输出形状：

```python
import torch
import torch.nn as nn

# 分析每一步的张量形状
batch_size = 8
seq_len = 25
input_size = 50
hidden_size = 100
num_layers = 3

gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
             bidirectional=True, batch_first=True)
x = torch.randn(batch_size, seq_len, input_size)
output, hidden = gru(x)

# 问题：output 和 hidden 的形状分别是什么？为什么？
print(f"output.shape = {output.shape}")  # 填写答案
print(f"hidden.shape = {hidden.shape}")  # 填写答案
```

### 进阶题

**练习 3：** 实现一个 `DotProductAttention` 类，使用最简单的点积（不缩放）计算注意力，并与 `ScaledDotProductAttention` 比较，在序列长度为 100 和维度为 512 时，两者 softmax 输入的方差有什么不同？

**练习 4：** 以下 Transformer 编码器存在一个常见错误，找出并修复：

```python
class BuggyTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # 注意：PyTorch MultiheadAttention 默认输入格式是 (seq, batch, d_model)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(attn_out)         # Bug 在哪里？
        ff_out = self.ff(x)
        x = self.norm2(ff_out)           # Bug 在哪里？
        return x
```

### 挑战题

**练习 5：** 扩展本章的 Seq2Seq 机器翻译系统，实现**束搜索（Beam Search）**解码，将贪心解码替换为 beam_size=3 的束搜索，比较两者在翻译质量上的差异（提示：维护 beam_size 个候选序列，每步保留概率最高的 beam_size 个）。

---

## 练习答案

### 练习 1 答案

```python
class TextClassifierRNNWithDropout(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 1. Embedding 后添加 Dropout：防止过拟合词嵌入
        self.embed_dropout = nn.Dropout(dropout)
        # 2. RNN 内部的 dropout（多层时层间添加）
        self.rnn = nn.RNN(
            embed_size, hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=dropout  # 只在多层间有效
        )
        # 3. 全连接层前添加 Dropout：防止最终分类器过拟合
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embed = self.embed_dropout(self.embedding(x))  # Dropout 位置 1
        output, hidden = self.rnn(embed)
        last_hidden = self.fc_dropout(hidden[-1])       # Dropout 位置 2
        return self.fc(last_hidden)

# 关键原则：
# - 嵌入层后的 Dropout：减少词嵌入的协同适应
# - RNN 层间 Dropout：需要 num_layers > 1 才有效
# - 分类器前的 Dropout：减少全连接层过拟合
```

### 练习 2 答案

```python
# output.shape = (8, 25, 200)
# 解释：batch=8, seq_len=25, hidden_size*2=200（双向，所以乘2）

# hidden.shape = (6, 8, 100)
# 解释：num_layers * num_directions = 3*2=6，batch=8，hidden_size=100
# 注意：双向时 hidden 存储顺序为：
# [前向第1层, 后向第1层, 前向第2层, 后向第2层, 前向第3层, 后向第3层]

import torch
import torch.nn as nn

batch_size = 8
seq_len = 25
input_size = 50
hidden_size = 100
num_layers = 3

gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
             bidirectional=True, batch_first=True)
x = torch.randn(batch_size, seq_len, input_size)
output, hidden = gru(x)

print(f"output.shape = {output.shape}")  # torch.Size([8, 25, 200])
print(f"hidden.shape = {hidden.shape}")  # torch.Size([6, 8, 100])

# 提取最后一个时刻的前向和后向状态
forward_last = output[:, -1, :hidden_size]   # 最后位置的前向
backward_last = output[:, 0, hidden_size:]   # 第一位置的后向（后向的最后时刻）
combined = torch.cat([forward_last, backward_last], dim=1)
print(f"combined.shape = {combined.shape}")  # (8, 200)
```

### 练习 3 答案

```python
import torch
import torch.nn as nn

class DotProductAttention(nn.Module):
    """未缩放的点积注意力"""
    def forward(self, query, key, value):
        # query/key/value: (batch, seq, d_k)
        scores = torch.matmul(query, key.transpose(-2, -1))
        # 不除以 √d_k
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, value), weights


# 方差分析
batch = 1
seq_len = 100
d_k = 512

torch.manual_seed(42)
Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)

# 未缩放：分数方差 ≈ d_k
scores_raw = torch.matmul(Q, K.transpose(-2, -1))
print(f"未缩放分数的方差: {scores_raw.var().item():.2f}")
print(f"期望方差（≈d_k）: {d_k}")

# 缩放后：分数方差 ≈ 1
scores_scaled = scores_raw / (d_k ** 0.5)
print(f"缩放后分数的方差: {scores_scaled.var().item():.4f}")
print(f"期望方差（≈1）: 1.0")

# 当方差大时，softmax 输出趋于 one-hot，梯度几乎为零
# 这就是为什么 Transformer 必须缩放

soft_raw = torch.softmax(scores_raw[0, 0], dim=-1)
soft_scaled = torch.softmax(scores_scaled[0, 0], dim=-1)
print(f"\n未缩放 softmax 的熵 (越低越 one-hot): "
      f"{-(soft_raw * (soft_raw + 1e-9).log()).sum().item():.4f}")
print(f"缩放后 softmax 的熵 (越高越均匀): "
      f"{-(soft_scaled * (soft_scaled + 1e-9).log()).sum().item():.4f}")
```

### 练习 4 答案

```python
# Bug 分析：
# 原代码中残差连接缺失（Pre-Norm 或 Post-Norm 都需要 x = norm(x + sublayer(x))）

class FixedTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                               batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # 修复1：添加残差连接 x = LayerNorm(x + Attention(x))
        attn_out, _ = self.attention(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))   # ← 正确：残差 + LayerNorm

        # 修复2：同样，前馈层也需要残差连接
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))      # ← 正确：残差 + LayerNorm
        return x


# 验证
import torch
d_model, num_heads, batch, seq = 128, 4, 2, 10
layer = FixedTransformerLayer(d_model, num_heads)
x = torch.randn(batch, seq, d_model)
out = layer(x)
print(f"修复后 Transformer 层输出: {out.shape}")  # (2, 10, 128)
print("残差连接确保了梯度可以直接反向传播，防止梯度消失")
```

### 练习 5 答案（束搜索）

```python
def beam_search_translate(model, src_seq, beam_size=3, max_len=15):
    """
    束搜索解码
    维护 beam_size 个候选序列，每步扩展并保留最优 beam_size 个
    """
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor([src_seq], dtype=torch.long).to(device)
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

        # 初始束：[(累计对数概率, 词序列, hidden, cell)]
        sos_id = ZH_VOCAB['<sos>']
        eos_id = ZH_VOCAB['<eos>']

        beams = [(0.0, [sos_id], hidden, cell)]
        completed = []

        for step in range(max_len):
            new_beams = []

            for score, seq, h, c in beams:
                if seq[-1] == eos_id:
                    completed.append((score, seq))
                    continue

                # 当前词
                curr_token = torch.tensor([seq[-1]], dtype=torch.long).to(device)

                # 解码一步
                output, new_h, new_c, _ = model.decoder(
                    curr_token, h, c, encoder_outputs
                )
                # output: (1, vocab_size)

                # 计算对数概率，取 top-beam_size
                log_probs = torch.log_softmax(output, dim=-1).squeeze(0)
                top_probs, top_ids = log_probs.topk(beam_size)

                for prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
                    new_score = score + prob
                    new_seq = seq + [token_id]
                    new_beams.append((new_score, new_seq, new_h, new_c))

            if not new_beams:
                break

            # 按累计分数降序，保留 beam_size 个
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]

            # 若所有束都结束，提前退出
            if all(b[1][-1] == eos_id for b in beams):
                break

        # 收集剩余束
        for score, seq, h, c in beams:
            completed.append((score, seq))

        # 选最高分
        if not completed:
            return []
        best_score, best_seq = max(completed, key=lambda x: x[0])

        # 解码词汇（去掉 <sos> 和 <eos>）
        result = []
        for token_id in best_seq[1:]:
            if token_id == eos_id:
                break
            result.append(ZH_ID2WORD.get(token_id, '<unk>'))

        return result


# 对比贪心和束搜索
print("=== 贪心解码 vs 束搜索 对比 ===")
test_cases = [
    ([3, 4, 5], "i love you"),
    ([3, 9, 10], "i like cat"),
]

for src_ids, src_text in test_cases:
    greedy_result, _ = translate(model, src_ids)
    beam_result = beam_search_translate(model, src_ids, beam_size=3)

    print(f"源句:     {src_text}")
    print(f"贪心解码: {''.join(greedy_result)}")
    print(f"束搜索:   {''.join(beam_result)}")
    print()

# 说明：
# 贪心解码每步选局部最优，可能陷入次优解
# 束搜索维护多个候选，全局搜索空间更大，通常质量更好
# 在实际 NMT 系统中，beam_size=4~6 是常用配置
```
