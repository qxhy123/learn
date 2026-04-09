# 附录 A：数学符号速查

本附录汇总 Transformer 教程中使用的全部数学符号、公式及对应 PyTorch 实现，供随时查阅。

---

## A.1 符号约定

### 基本表示

| 符号 | 含义 | 示例 |
|------|------|------|
| $a$ | 标量（小写斜体） | 学习率 $\alpha$，温度 $\tau$ |
| $\mathbf{v}$ | 向量（小写粗体） | 词嵌入 $\mathbf{e}$，查询向量 $\mathbf{q}$ |
| $\mathbf{M}$ | 矩阵（大写粗体） | 权重矩阵 $\mathbf{W}$，注意力矩阵 $\mathbf{A}$ |
| $\mathcal{V}$ | 集合或空间（花体） | 词表 $\mathcal{V}$，特征空间 $\mathcal{H}$ |
| $\hat{y}$ | 预测值（帽符号） | 预测概率 $\hat{p}$ |
| $y^*$ | 最优值（星号） | 最优标签 $y^*$ |

### 常用下标含义

| 下标 | 含义 |
|------|------|
| $i, j$ | 序列位置索引（通常从 1 开始） |
| $k$ | 注意力头索引，或第 $k$ 层 |
| $t$ | 时间步或 token 位置 |
| $d$ | 特征维度索引 |
| $b$ | 批次索引 |
| $h$ | 注意力头编号 |
| $n$ | 序列长度 |
| $v$ | 词表中词的索引 |

### 维度约定

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $B$ | 批次大小（batch size） | 32, 64, 128 |
| $n$ 或 $L$ | 序列长度（sequence length） | 512, 1024, 2048 |
| $d_{\text{model}}$ | 模型隐藏维度 | 512, 768, 1024 |
| $d_k$ | 查询/键的维度 | $d_{\text{model}} / H$ |
| $d_v$ | 值的维度 | $d_{\text{model}} / H$ |
| $d_{\text{ff}}$ | 前馈网络中间维度 | $4 \times d_{\text{model}}$ |
| $H$ | 注意力头数 | 8, 12, 16 |
| $V$ | 词表大小 | 30000, 50257 |

```python
# 维度示例
B, n, d_model = 32, 512, 768
H = 12
d_k = d_v = d_model // H  # 64
d_ff = 4 * d_model         # 3072
```

---

## A.2 矩阵运算

### 矩阵乘法

$$\mathbf{C} = \mathbf{A}\mathbf{B}, \quad \mathbf{A} \in \mathbb{R}^{m \times k},\ \mathbf{B} \in \mathbb{R}^{k \times n},\ \mathbf{C} \in \mathbb{R}^{m \times n}$$

$$C_{ij} = \sum_{l=1}^{k} A_{il} B_{lj}$$

```python
import torch
A = torch.randn(m, k)
B = torch.randn(k, n)
C = A @ B          # 或 torch.matmul(A, B)
```

### 转置

$$(\mathbf{A}^{\top})_{ij} = A_{ji}$$

$$(\mathbf{AB})^{\top} = \mathbf{B}^{\top}\mathbf{A}^{\top}$$

```python
A_T = A.transpose(-2, -1)   # 通用写法，兼容批量维度
A_T = A.T                   # 仅适用于二维张量
```

### 批量矩阵乘法

输入形状为 $(B, m, k)$ 和 $(B, k, n)$，输出形状为 $(B, m, n)$，在批次维度上独立执行矩阵乘法：

$$\mathbf{C}[b] = \mathbf{A}[b]\,\mathbf{B}[b], \quad b = 1,\ldots,B$$

```python
A = torch.randn(B, m, k)
B_mat = torch.randn(B, k, n)
C = torch.bmm(A, B_mat)     # 严格三维
C = A @ B_mat               # 更通用，支持任意批量维度
```

### Einsum 表示法

Einsum 用下标字符串描述张量缩并，简洁且高效：

| 操作 | Einsum 字符串 | 等价写法 |
|------|--------------|---------|
| 矩阵乘法 | `"ij,jk->ik"` | `A @ B` |
| 批量矩阵乘法 | `"bij,bjk->bik"` | `torch.bmm(A, B)` |
| 点积 | `"i,i->"` | `torch.dot(a, b)` |
| 外积 | `"i,j->ij"` | `torch.outer(a, b)` |
| 迹 | `"ii->"` | `torch.trace(A)` |
| 多头注意力分数 | `"bhid,bhjd->bhij"` | `(Q @ K.T) / sqrt(d_k)` |

```python
# 注意力分数的 einsum 写法
import torch
scores = torch.einsum("bhid,bhjd->bhij", Q, K)

# 注意力输出
output = torch.einsum("bhij,bhjd->bhid", attn_weights, V)
```

---

## A.3 Softmax 与交叉熵

### Softmax 公式

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad i = 1, \ldots, K$$

**性质：**
- 输出为概率分布：$\sum_i \text{softmax}(\mathbf{z})_i = 1$，且每项 $> 0$
- 平移不变性：$\text{softmax}(\mathbf{z}) = \text{softmax}(\mathbf{z} + c)$（利用此性质做数值稳定化）
- 梯度：$\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = \text{softmax}(\mathbf{z})_i(\delta_{ij} - \text{softmax}(\mathbf{z})_j)$

```python
import torch.nn.functional as F
probs = F.softmax(logits, dim=-1)
```

### 交叉熵公式

对于真实标签 $y$（one-hot）和预测概率 $\hat{\mathbf{p}}$：

$$\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{K} y_k \log \hat{p}_k$$

当 $y$ 为类别索引时简化为：

$$\mathcal{L}_{\text{CE}} = -\log \hat{p}_{y}$$

```python
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
# logits: (B, K)，targets: (B,) 为类别索引
loss = criterion(logits, targets)

# 等价的手动实现（数值稳定版本）
log_probs = F.log_softmax(logits, dim=-1)
loss = F.nll_loss(log_probs, targets)
```

### 温度参数的影响

引入温度 $\tau > 0$ 控制分布的尖锐程度：

$$\text{softmax}\!\left(\frac{\mathbf{z}}{\tau}\right)_i = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$$

| 温度 $\tau$ | 效果 |
|------------|------|
| $\tau \to 0^+$ | 趋向 argmax，分布极度集中 |
| $\tau = 1$ | 标准 softmax |
| $\tau \to \infty$ | 趋向均匀分布 |

```python
def softmax_with_temperature(logits, temperature=1.0):
    return F.softmax(logits / temperature, dim=-1)
```

### 数值稳定性

直接计算 $e^{z_i}$ 容易溢出，稳定写法利用平移不变性：

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}$$

```python
# PyTorch 的 F.softmax 内部已做此处理
# 手动实现：
z_shifted = z - z.max(dim=-1, keepdim=True).values
probs = z_shifted.exp() / z_shifted.exp().sum(dim=-1, keepdim=True)
```

---

## A.4 层归一化

### LayerNorm 公式

对单个样本的特征维度做归一化（与 BatchNorm 不同，不依赖批次统计）：

$$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$

其中：
$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \qquad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$$

- $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^d$：可学习的缩放和偏移参数
- $\epsilon$：防止除零的小常数（通常 $10^{-5}$）
- $\odot$：逐元素乘法

```python
import torch.nn as nn
layer_norm = nn.LayerNorm(d_model)
x_normed = layer_norm(x)   # x: (..., d_model)

# 手动实现
mean = x.mean(dim=-1, keepdim=True)
var  = x.var(dim=-1, keepdim=True, unbiased=False)
x_normed = (x - mean) / (var + 1e-5).sqrt()
x_normed = x_normed * gamma + beta
```

### RMSNorm 公式

去掉均值中心化，只做均方根归一化，计算更高效：

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}, \qquad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

LLaMA、Mistral 等模型使用 RMSNorm。

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight
```

### BatchNorm 对比

| 特性 | LayerNorm | RMSNorm | BatchNorm |
|------|-----------|---------|-----------|
| 归一化维度 | 特征维度 | 特征维度 | 批次维度 |
| 批次依赖 | 否 | 否 | 是 |
| 均值中心化 | 是 | 否 | 是 |
| 可学习参数 | $\boldsymbol{\gamma}, \boldsymbol{\beta}$ | $\boldsymbol{\gamma}$ | $\boldsymbol{\gamma}, \boldsymbol{\beta}$ |
| NLP 使用 | 广泛 | LLaMA 等 | 少用 |
| 推理时统计量 | 实时计算 | 实时计算 | 运行均值/方差 |

---

## A.5 注意力公式

### 缩放点积注意力

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\right)\mathbf{V}$$

其中 $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$，$\mathbf{K} \in \mathbb{R}^{m \times d_k}$，$\mathbf{V} \in \mathbb{R}^{m \times d_v}$，输出 $\in \mathbb{R}^{n \times d_v}$。

缩放因子 $\sqrt{d_k}$ 防止点积过大导致 softmax 梯度消失。

```python
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (..., n, m)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)             # (..., n, m)
    return attn_weights @ V                              # (..., n, d_v)
```

### 多头注意力

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\,\mathbf{W}^O$$

$$\text{head}_h = \text{Attention}(\mathbf{Q}\mathbf{W}_h^Q,\ \mathbf{K}\mathbf{W}_h^K,\ \mathbf{V}\mathbf{W}_h^V)$$

参数矩阵：$\mathbf{W}_h^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$\mathbf{W}_h^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$\mathbf{W}_h^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$，$\mathbf{W}^O \in \mathbb{R}^{Hd_v \times d_{\text{model}}}$。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, H: int):
        super().__init__()
        self.H, self.d_k = H, d_model // H
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, n, _ = Q.shape
        # 线性投影并分头: (B, H, n, d_k)
        q = self.W_q(Q).view(B, n, self.H, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, -1, self.H, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, -1, self.H, self.d_k).transpose(1, 2)
        out = scaled_dot_product_attention(q, k, v, mask)  # (B, H, n, d_k)
        out = out.transpose(1, 2).reshape(B, n, -1)        # (B, n, d_model)
        return self.W_o(out)
```

### 自注意力

自注意力是多头注意力的特例，Q、K、V 均来自同一输入序列 $\mathbf{X}$：

$$\text{SelfAttention}(\mathbf{X}) = \text{MultiHead}(\mathbf{X}, \mathbf{X}, \mathbf{X})$$

编码器使用**双向**自注意力（无掩码），解码器使用**因果**（causal）自注意力（下三角掩码）：

```python
# 因果掩码（下三角）
def causal_mask(n: int, device=None) -> torch.Tensor:
    return torch.tril(torch.ones(n, n, device=device)).bool()

# 使用示例
mask = causal_mask(seq_len, device=x.device)
out = self_attention(X, X, X, mask=mask)
```

### 交叉注意力

查询来自解码器，键和值来自编码器输出 $\mathbf{M}$：

$$\text{CrossAttention}(\mathbf{X}_{\text{dec}}, \mathbf{M}) = \text{MultiHead}(\mathbf{X}_{\text{dec}},\ \mathbf{M},\ \mathbf{M})$$

```python
# 解码器中的交叉注意力
cross_out = cross_attention(
    Q=decoder_hidden,    # (B, tgt_len, d_model)
    K=encoder_output,    # (B, src_len, d_model)
    V=encoder_output,    # (B, src_len, d_model)
)
```

---

## A.6 位置编码公式

### 正弦位置编码（Sinusoidal PE）

原始 Transformer 使用固定正弦位置编码：

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

其中 $pos$ 为位置，$i$ 为维度索引（$0 \le i < d_{\text{model}}/2$）。

```python
def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()          # (max_len, 1)
    div = torch.exp(
        torch.arange(0, d_model, 2).float()
        * (-math.log(10000.0) / d_model)
    )                                                          # (d_model/2,)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (max_len, d_model)
```

### 旋转位置编码（RoPE）

RoPE 通过旋转矩阵将位置信息编码进查询和键，使注意力分数只依赖相对位置：

对查询向量 $\mathbf{q}$ 的第 $(2i, 2i+1)$ 维施加旋转：

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(pos \cdot \theta_i) & -\sin(pos \cdot \theta_i) \\ \sin(pos \cdot \theta_i) & \cos(pos \cdot \theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

其中 $\theta_i = 10000^{-2i/d_k}$，相同操作也施加于键向量 $\mathbf{k}$。

旋转后点积满足：$\langle \mathbf{q}'_{pos}, \mathbf{k}'_{pos'} \rangle$ 只依赖 $pos - pos'$。

```python
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, n, d_k); cos/sin: (n, d_k)
    x1, x2 = x[..., ::2], x[..., 1::2]      # 奇偶维度分离
    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos + rotated * sin

def precompute_rope(d_k: int, max_len: int, base: float = 10000.0):
    theta = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
    pos   = torch.arange(max_len).float()
    freqs = torch.outer(pos, theta)                  # (max_len, d_k/2)
    freqs = torch.cat([freqs, freqs], dim=-1)        # (max_len, d_k)
    return freqs.cos(), freqs.sin()
```

### 相对位置编码

相对位置编码直接在注意力分数中加入相对位置偏置（以 ALiBi 为例）：

$$\text{score}_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} - m_h \cdot |i - j|$$

其中 $m_h$ 为第 $h$ 头的斜率，按几何级数设置：$m_h = 2^{-8h/H}$。

```python
def alibi_slopes(H: int) -> torch.Tensor:
    # 计算 ALiBi 各头斜率
    n = 2 ** math.floor(math.log2(H))
    slopes = 2 ** (-8 * torch.arange(1, n + 1).float() / n)
    if H > n:
        extra = 2 ** (-4 * torch.arange(1, 2 * (H - n) + 1, 2).float() / n)
        slopes = torch.cat([slopes, extra])
    return slopes  # (H,)

def alibi_bias(H: int, n: int) -> torch.Tensor:
    slopes = alibi_slopes(H).view(H, 1, 1)  # (H, 1, 1)
    dist   = torch.arange(n).unsqueeze(0) - torch.arange(n).unsqueeze(1)
    return -slopes * dist.abs().unsqueeze(0)  # (H, n, n)
```

---

## A.7 损失函数

### 语言模型损失（因果 LM）

自回归语言模型最大化序列的对数似然，等价于最小化下一个 token 的交叉熵：

$$\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1};\, \theta)$$

```python
def lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: (B, T, V)；labels: (B, T)
    # 将输入向右移一位：用 x_{0..T-2} 预测 x_{1..T-1}
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

### MLM 损失（掩码语言模型）

BERT 等模型随机掩盖 15% 的 token，预测被掩盖的原始 token：

$$\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|}\sum_{t \in \mathcal{M}} \log P(x_t \mid \tilde{\mathbf{x}};\, \theta)$$

其中 $\mathcal{M}$ 为被掩盖位置的集合，$\tilde{\mathbf{x}}$ 为带 `[MASK]` 的输入序列。

```python
def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: (B, T, V)；labels: (B, T)，非掩码位置为 -100
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
```

### 困惑度（Perplexity）

困惑度是语言模型的标准评估指标，等于每个 token 的平均负对数似然的指数：

$$\text{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t})\right) = \exp(\mathcal{L}_{\text{LM}})$$

困惑度越低，模型越好；随机模型的困惑度等于词表大小 $V$。

```python
import math

def perplexity(loss: float) -> float:
    return math.exp(loss)

# 在评估循环中
total_loss, total_tokens = 0.0, 0
with torch.no_grad():
    for batch in eval_loader:
        loss = model(**batch).loss
        total_loss   += loss.item() * batch["input_ids"].numel()
        total_tokens += batch["input_ids"].numel()
ppl = math.exp(total_loss / total_tokens)
```

---

## A.8 常用激活函数

### ReLU

$$\text{ReLU}(x) = \max(0, x)$$

- 梯度：$\text{ReLU}'(x) = \mathbf{1}[x > 0]$
- 问题：负区间梯度为零（"神经元死亡"）

```python
F.relu(x)
```

### GELU

高斯误差线性单元，GPT 系列默认激活函数：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$$

快速近似（Hendrycks & Gimpel, 2016）：

$$\text{GELU}(x) \approx 0.5x\left[1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right]$$

```python
F.gelu(x)                          # 精确版
F.gelu(x, approximate='tanh')      # 近似版（更快）
```

### SiLU / Swish

$$\text{SiLU}(x) = \text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

- 自门控，平滑且无界（正方向）
- LLaMA、Mistral 等模型广泛使用

```python
F.silu(x)    # PyTorch 内置
```

### SwiGLU

将 SiLU 与门控线性单元（GLU）结合，LLaMA 前馈层的标准选择：

$$\text{SwiGLU}(\mathbf{x}) = \text{SiLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \odot (\mathbf{W}_2 \mathbf{x} + \mathbf{b}_2)$$

使用 SwiGLU 时，前馈网络中间维度通常取 $\frac{2}{3} \times 4d_{\text{model}}$ 而非 $4d_{\text{model}}$，以保持参数量不变。

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up   = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff,   d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.W_gate(x))
        up   = self.W_up(x)
        return self.W_down(gate * up)
```

### 激活函数对比

| 函数 | 公式 | 范围 | 主要用途 |
|------|------|------|---------|
| ReLU | $\max(0,x)$ | $[0, +\infty)$ | CNN，早期 Transformer |
| GELU | $x\Phi(x)$ | $(-0.17, +\infty)$ | BERT，GPT 系列 |
| SiLU | $x\sigma(x)$ | $(-0.28, +\infty)$ | LLaMA，EfficientNet |
| SwiGLU | $\text{SiLU}(Wx)\odot W'x$ | $(-\infty, +\infty)$ | LLaMA FFN 层 |

---

## 快速索引

| 公式 | 所在节 |
|------|--------|
| Scaled Dot-Product Attention | A.5 |
| Multi-Head Attention | A.5 |
| Softmax（数值稳定版） | A.3 |
| Cross-Entropy Loss | A.3 |
| LayerNorm | A.4 |
| RMSNorm | A.4 |
| Sinusoidal PE | A.6 |
| RoPE | A.6 |
| ALiBi | A.6 |
| 语言模型损失 | A.7 |
| 困惑度 | A.7 |
| GELU | A.8 |
| SwiGLU | A.8 |
