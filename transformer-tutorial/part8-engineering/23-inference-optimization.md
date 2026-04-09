# 第23章：推理优化

> 训练一个模型只需要一次，但推理需要运行数百万次。推理优化是将研究成果转化为实际产品的关键桥梁。

## 学习目标

完成本章学习后，你将能够：

1. 理解KV Cache的工作原理，并用代码实现它
2. 掌握INT8/INT4量化技术，了解PTQ与QAT的区别
3. 理解知识蒸馏的原理，能够训练一个蒸馏模型
4. 将PyTorch模型导出为ONNX格式并在ONNX Runtime上运行推理
5. 了解投机解码、连续批处理等前沿推理优化技术

---

## 23.1 KV Cache

### 为什么需要KV Cache

自回归生成（Autoregressive Generation）是大语言模型最核心的推理范式。模型逐词生成输出，每次生成一个新token时，都需要处理**所有**已生成的token。

考虑生成一段长度为 $T$ 的文本。在第 $t$ 步生成第 $t$ 个token时，注意力机制需要计算：

$$\text{Attention}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{Q_t K_{1:t}^T}{\sqrt{d_k}}\right) V_{1:t}$$

如果每一步都重新计算所有位置的 $K$ 和 $V$，总计算量为：

$$\text{总计算量} = \sum_{t=1}^{T} O(t \cdot d) = O(T^2 \cdot d)$$

其中 $d$ 是隐层维度。对于 $T=2048$，这意味着每生成一个新token，都要重新处理2048个位置，浪费了大量重复计算。

**问题的根源**：前 $t-1$ 个位置的 $K$、$V$ 值在每一步都是相同的，却被反复计算。

### 内存换计算

KV Cache（键值缓存）的核心思想是**空间换时间**：将计算过的 $K$、$V$ 矩阵缓存起来，避免重复计算。

```
不使用KV Cache：
步骤1：计算 K1, V1           → 生成 token1
步骤2：计算 K1,K2, V1,V2     → 生成 token2  （重复计算K1,V1）
步骤3：计算 K1,K2,K3, V1,V2,V3 → 生成 token3  （重复计算K1,V1,K2,V2）

使用KV Cache：
步骤1：计算 K1, V1           → 缓存(K1,V1)    → 生成 token1
步骤2：计算 K2, V2           → 缓存(K1,K2,V1,V2)  → 生成 token2
步骤3：计算 K3, V3           → 缓存(K1,K2,K3,V1,V2,V3) → 生成 token3
```

使用KV Cache后，每步只需计算当前token的 $K$、$V$，计算量从 $O(T^2 \cdot d)$ 降低到 $O(T \cdot d)$，是线性复杂度。

**代价**：需要存储所有历史步的 $K$、$V$。对于一个 $L$ 层、隐层维度 $d$、序列长度 $T$ 的模型，KV Cache的内存占用为：

$$\text{KV Cache 内存} = 2 \times L \times T \times d \times \text{sizeof(dtype)}$$

以GPT-3（96层，$d=12288$，$T=2048$，FP16）为例：

$$2 \times 96 \times 2048 \times 12288 \times 2 \approx 9.6 \text{ GB}$$

这就是为什么大模型推理需要大量GPU显存。

### KV Cache的实现

KV Cache的实现分为两个阶段：

**预填充阶段（Prefill Phase）**：处理输入提示词（prompt），一次性计算所有token的 $K$、$V$ 并缓存。

**解码阶段（Decode Phase）**：每次只输入一个新token，利用缓存的 $K$、$V$ 计算注意力。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MultiHeadAttentionWithKVCache(nn.Module):
    """带KV Cache的多头注意力"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) -> (batch, heads, seq, d_k)"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            past_key_value: 缓存的(K, V)，形状为(batch, heads, past_len, d_k)
            use_cache: 是否返回更新后的KV Cache

        Returns:
            output: (batch, seq_len, d_model)
            present_key_value: 更新后的(K, V) cache
        """
        batch, seq_len, _ = x.shape

        # 计算当前输入的Q, K, V
        Q = self.split_heads(self.W_q(x))  # (batch, heads, seq, d_k)
        K = self.split_heads(self.W_k(x))  # (batch, heads, seq, d_k)
        V = self.split_heads(self.W_v(x))  # (batch, heads, seq, d_k)

        # 如果有历史缓存，拼接到当前K、V前面
        if past_key_value is not None:
            past_K, past_V = past_key_value
            K = torch.cat([past_K, K], dim=2)  # (batch, heads, past+seq, d_k)
            V = torch.cat([past_V, V], dim=2)  # (batch, heads, past+seq, d_k)

        # 保存当前步的KV Cache
        present_key_value = (K, V) if use_cache else None

        # 计算注意力
        scale = self.d_k ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # 因果掩码：当前token只能看到自身及之前的token
        total_len = K.shape[2]
        q_len = Q.shape[2]
        # scores: (batch, heads, q_len, total_len)
        # 只掩盖未来的位置
        if q_len > 1:  # 预填充阶段需要因果掩码
            causal_mask = torch.triu(
                torch.ones(q_len, total_len, device=x.device),
                diagonal=total_len - q_len + 1
            ).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (batch, heads, q_len, d_k)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, q_len, self.d_model)
        output = self.W_o(attn_output)

        return output, present_key_value


class TransformerBlockWithKVCache(nn.Module):
    """带KV Cache的Transformer块"""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.attention = MultiHeadAttentionWithKVCache(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, past_key_value=None, use_cache=False):
        # 注意力 + 残差
        attn_out, present_kv = self.attention(
            self.norm1(x), past_key_value, use_cache
        )
        x = x + attn_out

        # FFN + 残差
        x = x + self.ffn(self.norm2(x))

        return x, present_kv
```

在推理时，调用方式如下：

```python
def generate_with_kv_cache(model_blocks, token_ids, max_new_tokens=50):
    """使用KV Cache进行自回归生成"""
    past_key_values = [None] * len(model_blocks)  # 每层一个缓存

    # 预填充阶段：处理输入prompt
    x = embed(token_ids)  # (batch, prompt_len, d_model)
    for i, block in enumerate(model_blocks):
        x, past_key_values[i] = block(x, use_cache=True)

    generated = []

    # 解码阶段：逐步生成
    for _ in range(max_new_tokens):
        next_token_id = sample_next_token(x[:, -1, :])  # 从最后一个位置采样
        generated.append(next_token_id)

        # 只输入新token
        x = embed(next_token_id.unsqueeze(1))  # (batch, 1, d_model)

        for i, block in enumerate(model_blocks):
            x, past_key_values[i] = block(
                x,
                past_key_value=past_key_values[i],  # 使用缓存
                use_cache=True
            )

    return generated
```

### PagedAttention简介

传统KV Cache有一个严重问题：必须**预先分配**连续的内存。由于不知道生成序列的最终长度，通常需要按最大序列长度分配内存，导致大量内存浪费（内存碎片化）。

**PagedAttention**（由vLLM团队提出）借鉴了操作系统中虚拟内存分页的思想：

- 将KV Cache分割成固定大小的**页（Page）**
- 每个序列的KV Cache可以存储在**不连续**的内存页中
- 通过**块表（Block Table）**记录逻辑页号到物理页号的映射
- 不同请求可以**共享**相同的前缀（prompt sharing）

```
传统KV Cache（连续内存）：
[=====Seq1=====][==Seq2==][=====Seq3=====]
│   已分配但未使用 ↑↑↑    │

PagedAttention（分页内存）：
Page0: [Seq1块0] Page1: [Seq2块0] Page2: [Seq1块1]
Page3: [Seq3块0] Page4: [空闲]    Page5: [Seq3块1]
       按需分配，几乎无浪费
```

PagedAttention将GPU内存利用率从通常的30-40%提升到接近90%，显著提升吞吐量。这是vLLM能高效服务大模型的核心技术之一。

---

## 23.2 模型量化

### 量化基础

神经网络训练通常使用FP32（32位浮点数）或FP16（16位浮点数）。**量化（Quantization）**是将权重和激活值从高精度浮点数压缩为低精度整数（INT8、INT4等）的技术。

**为什么量化有效？**

对于INT8量化，将浮点值 $x$ 映射到 $[-128, 127]$ 的整数：

$$x_q = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right) + z, \, q_{\min}, \, q_{\max}\right)$$

其中 $s$ 是**缩放因子（scale）**，$z$ 是**零点（zero point）**。

反量化（dequantize）时：

$$\hat{x} = s \cdot (x_q - z)$$

**量化的收益**：

| 精度 | 位宽 | 内存占用 | 典型加速比 |
|------|------|----------|-----------|
| FP32 | 32位 | 1x       | 1x        |
| FP16 | 16位 | 0.5x     | 2-3x      |
| INT8 | 8位  | 0.25x    | 3-4x      |
| INT4 | 4位  | 0.125x   | 4-8x（理论）|

**对称量化 vs 非对称量化**：

- 对称量化：零点 $z=0$，适合权重（通常以0为中心对称分布）
- 非对称量化：零点 $z \neq 0$，适合激活值（ReLU后全为正数）

### 训练后量化（PTQ）

Post-Training Quantization（PTQ）在训练完成后对模型进行量化，无需重新训练，是最常用的量化方式。

**逐层量化（Per-tensor）**：整个张量使用一个 $s$ 和 $z$，量化误差较大。

**逐通道量化（Per-channel）**：每个输出通道使用独立的 $s$ 和 $z$，精度更高。

```python
import torch
import torch.nn as nn
import numpy as np

def quantize_tensor_int8(tensor: torch.Tensor, per_channel: bool = False):
    """INT8对称量化"""
    if per_channel:
        # 逐输出通道量化（对线性层）
        # tensor: (out_features, in_features)
        max_vals = tensor.abs().max(dim=1, keepdim=True).values
    else:
        max_vals = tensor.abs().max()

    # scale：将最大值映射到127
    scale = max_vals / 127.0
    scale = scale.clamp(min=1e-8)  # 避免除零

    # 量化
    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)

    return quantized, scale

def dequantize_tensor(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """INT8反量化"""
    return quantized.float() * scale


class QuantizedLinear(nn.Module):
    """量化的线性层：权重INT8，计算时反量化"""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        # 量化权重
        q_weight, scale = quantize_tensor_int8(linear.weight.data, per_channel=True)
        self.register_buffer('q_weight', q_weight)
        self.register_buffer('scale', scale)
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 推理时反量化权重，再做矩阵乘法
        # 实际硬件实现中，INT8矩阵乘法在整数域直接完成，效率更高
        weight = dequantize_tensor(self.q_weight, self.scale)
        return F.linear(x, weight, self.bias)

    @property
    def weight_size_bytes(self):
        return self.q_weight.numel()  # INT8: 每个元素1字节


def ptq_model(model: nn.Module) -> nn.Module:
    """对模型中所有Linear层进行PTQ量化"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, QuantizedLinear(module))
        else:
            ptq_model(module)  # 递归处理子模块
    return model


# 使用示例
import torch.nn.functional as F

class SimpleTransformer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, num_layers=4, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

# 量化前后对比
model = SimpleTransformer()
model.eval()

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return total, total_bytes

n, b = count_parameters(model)
print(f"原始模型: {n:,}参数, {b/1024:.1f} KB")

quantized_model = ptq_model(model)

# 统计量化后的大小（只计算量化层的权重）
def count_quantized_size(model):
    total_bytes = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            total_bytes += module.weight_size_bytes  # INT8: 1字节/参数
        elif isinstance(module, nn.Embedding):
            total_bytes += module.weight.numel() * 4  # 保持FP32
    return total_bytes

qb = count_quantized_size(quantized_model)
print(f"量化模型权重: {qb/1024:.1f} KB")
print(f"压缩比: {b/qb:.1f}x")
```

### 量化感知训练（QAT）

Quantization-Aware Training（QAT）在训练阶段模拟量化误差，让模型"适应"量化带来的精度损失，通常比PTQ精度更高，但需要重新训练。

QAT的关键是**伪量化（Fake Quantization）**节点：

```python
class FakeQuantize(torch.autograd.Function):
    """伪量化：前向传播量化，反向传播直通估计（STE）"""

    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max):
        # 模拟量化过程
        x_q = torch.clamp(
            torch.round(x / scale + zero_point),
            quant_min, quant_max
        )
        x_dq = (x_q - zero_point) * scale  # 反量化
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        # 直通估计（Straight-Through Estimator, STE）
        # 量化函数不可微，用梯度直通来近似
        return grad_output, None, None, None, None
```

直通估计（STE）是QAT的核心技巧：量化操作（round函数）梯度为0，但用STE将梯度"直通"，使得上游参数可以被更新。

### GPTQ、AWQ、GGML

大语言模型量化面临特殊挑战：权重规模庞大，精度要求高。以下是三种主流方案：

**GPTQ（GPT Quantization）**

逐层最小化量化误差。对于权重矩阵 $W$，找到量化后的 $\hat{W}$ 使重构误差最小：

$$\hat{W} = \arg\min_{\hat{W}} \|WX - \hat{W}X\|_2^2$$

利用Hessian矩阵信息，按列顺序量化权重，量化一列后调整剩余列以补偿误差（类似Cholesky分解）。支持INT4量化，精度损失极小。

**AWQ（Activation-aware Weight Quantization）**

观察到权重并非等重要：某些权重对应高激活值的通道，对精度影响更大，称为"重要权重（salient weights）"。

AWQ为重要通道乘以缩放因子，将其"保护"起来，避免量化误差过大：

$$\hat{W} = \text{Quant}(W \cdot \text{diag}(s)^{-1}) \cdot \text{diag}(s)$$

其中 $s$ 是根据激活统计确定的逐通道缩放因子。

**GGML/GGUF**

由llama.cpp社区开发的量化格式，专注于CPU推理。支持多种量化粒度（Q4_0, Q4_K, Q5_K, Q8_0等），可在普通笔记本上运行7B参数模型。GGUF是其升级版格式，存储更多元数据。

---

## 23.3 知识蒸馏

### 蒸馏的基本原理

**知识蒸馏（Knowledge Distillation）**由Hinton等人在2015年提出。核心思想是：用一个大型的**教师模型（Teacher）**指导训练一个小型的**学生模型（Student）**。

学生模型不仅学习真实标签（硬标签），还学习教师模型的**软预测（软标签）**。软标签包含更丰富的信息——例如，对于"猫"的图片，教师模型可能预测猫80%，老虎15%，狗5%，这种类间相似性信息对学生非常有价值。

蒸馏的总损失函数：

$$\mathcal{L} = \alpha \mathcal{L}_{\text{CE}}(y, \hat{y}_s) + (1-\alpha) \mathcal{L}_{\text{KD}}(z_t, z_s, T)$$

其中：
- $\mathcal{L}_{\text{CE}}$ 是学生预测与真实标签的交叉熵损失
- $\mathcal{L}_{\text{KD}}$ 是蒸馏损失（学生与教师软预测的KL散度）
- $\alpha$ 是权重超参数
- $T$ 是温度参数

### 软标签蒸馏

温度 $T$ 软化了概率分布：

$$p_i^T = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

当 $T=1$ 时为正常softmax；$T > 1$ 时，分布更平滑，携带更多类间关系信息；$T < 1$ 时，分布更尖锐。

蒸馏损失（对称KL散度近似，实践中常用KL散度）：

$$\mathcal{L}_{\text{KD}} = T^2 \cdot \text{KL}\left(p^T_{\text{teacher}} \| p^T_{\text{student}}\right)$$

乘以 $T^2$ 是为了补偿梯度在大温度下的缩小效应（Hinton原文中的技巧）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """知识蒸馏损失"""

    def __init__(self, alpha: float = 0.5, temperature: float = 4.0):
        """
        Args:
            alpha: 硬标签损失的权重
            temperature: 软化温度T
        """
        super().__init__()
        self.alpha = alpha
        self.T = temperature

    def forward(
        self,
        student_logits: torch.Tensor,   # (batch, num_classes)
        teacher_logits: torch.Tensor,   # (batch, num_classes)
        labels: torch.Tensor,           # (batch,)
    ) -> torch.Tensor:
        # 硬标签损失：学生预测 vs 真实标签
        hard_loss = F.cross_entropy(student_logits, labels)

        # 软标签损失：学生软预测 vs 教师软预测
        student_soft = F.log_softmax(student_logits / self.T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=-1)

        # KL散度：KL(teacher || student)
        soft_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.T ** 2)  # 补偿温度缩放

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


def distill_training_step(
    teacher: nn.Module,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: DistillationLoss,
    batch: tuple,
):
    """一个蒸馏训练步骤"""
    inputs, labels = batch

    # 教师模型不参与梯度计算
    with torch.no_grad():
        teacher_logits = teacher(inputs)

    # 学生模型前向传播
    student_logits = student(inputs)

    # 计算蒸馏损失
    loss = criterion(student_logits, teacher_logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

### 特征蒸馏

除了最终的输出logits，还可以让学生模型学习教师模型的**中间特征（Feature Distillation）**，传递更细粒度的知识。

常用方法：

**FitNets**：让学生中间层的特征图尽量接近教师对应层：

$$\mathcal{L}_{\text{feat}} = \frac{1}{HW} \|F_s - F_t\|_F^2$$

由于学生和教师的中间层维度可能不同，需要添加一个适配层（hint layer）。

**注意力图蒸馏（Attention Transfer）**：蒸馏注意力权重矩阵，传递"关注哪里"的知识：

$$\mathcal{L}_{\text{attn}} = \sum_l \|A_s^l - A_t^l\|_F^2$$

```python
class FeatureDistillationLoss(nn.Module):
    """特征蒸馏损失（FitNets风格）"""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        # 适配层：将学生特征映射到教师特征空间
        self.adapter = nn.Linear(student_dim, teacher_dim)

    def forward(
        self,
        student_features: list,   # 每层的特征列表
        teacher_features: list,   # 教师对应层的特征列表
    ) -> torch.Tensor:
        total_loss = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            # 适配维度
            s_feat_adapted = self.adapter(s_feat)
            # L2损失（对特征图归一化后计算）
            s_norm = F.normalize(s_feat_adapted, dim=-1)
            t_norm = F.normalize(t_feat.detach(), dim=-1)
            total_loss += F.mse_loss(s_norm, t_norm)
        return total_loss
```

### DistilBERT案例

DistilBERT是最著名的BERT蒸馏模型，由HuggingFace在2019年提出：

| 指标 | BERT-base | DistilBERT |
|------|-----------|------------|
| 层数 | 12层      | 6层        |
| 参数量 | 110M    | 66M（40%↓）|
| 推理速度 | 1x    | 60%↑       |
| GLUE得分 | 79.6  | 77.0（97.5%保留）|

DistilBERT的蒸馏策略：

1. **架构**：删除BERT一半的层（12层→6层），初始化时每隔一层取一个教师层的权重
2. **三重损失**：MLM损失（语言模型）+ 软标签蒸馏损失 + 余弦嵌入损失（隐层对齐）
3. **训练数据**：使用与BERT相同的预训练语料（English Wikipedia + BookCorpus）

---

## 23.4 ONNX导出

### ONNX格式介绍

**ONNX（Open Neural Network Exchange）**是由微软、Facebook等公司联合推出的开放神经网络交换格式。它的目标是成为不同深度学习框架之间的**通用中间表示（IR）**。

```
PyTorch ──────────────────────────┐
TensorFlow ─── 导出为ONNX格式 ──→ ONNX Runtime → 部署（CPU/GPU/Edge）
JAX ──────────────────────────────┘
```

ONNX的优势：
- **跨框架**：训练用PyTorch，部署用ONNX Runtime，各取所长
- **硬件优化**：ONNX Runtime针对不同硬件（Intel CPU、NVIDIA GPU、ARM等）有深度优化
- **图优化**：算子融合、常量折叠、内存优化
- **量化支持**：集成量化工具链

### PyTorch模型导出

```python
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

class SimpleTransformerForExport(nn.Module):
    """用于ONNX导出的简单Transformer"""

    def __init__(self, vocab_size=1000, d_model=128, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(512, d_model)  # 固定位置编码
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, 2)  # 二分类

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.embedding(input_ids) + self.pos_encoding(positions)

        # 转换attention_mask为Transformer期望的格式
        # True表示忽略该位置（padding）
        src_key_padding_mask = (attention_mask == 0)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 取[CLS]位置（第一个token）进行分类
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


def export_to_onnx(
    model: nn.Module,
    onnx_path: str,
    batch_size: int = 1,
    seq_len: int = 32,
):
    """将PyTorch模型导出为ONNX格式"""
    model.eval()

    # 构造虚拟输入（仅用于追踪计算图）
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # 导出
    torch.onnx.export(
        model,
        args=(dummy_input_ids, dummy_attention_mask),  # 模型输入
        f=onnx_path,
        export_params=True,          # 导出模型权重
        opset_version=14,            # ONNX算子集版本
        do_constant_folding=True,    # 常量折叠优化
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        # 动态轴：允许batch_size和seq_len在推理时变化
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'},
        }
    )
    print(f"模型已导出到: {onnx_path}")

    return onnx_path
```

### 动态轴处理

ONNX导出的一个重要特性是**动态轴（Dynamic Axes）**。默认情况下，ONNX导出的模型形状是固定的（与虚拟输入相同）。通过指定动态轴，可以让模型接受不同大小的输入。

```python
# 静态导出（仅支持batch=1, seq=32）
torch.onnx.export(model, dummy_input, "model_static.onnx")

# 动态导出（支持任意batch和seq长度）
torch.onnx.export(
    model, dummy_input, "model_dynamic.onnx",
    dynamic_axes={
        'input_ids':      {0: 'batch', 1: 'seq'},
        'attention_mask': {0: 'batch', 1: 'seq'},
        'logits':         {0: 'batch'},
    }
)
```

注意，动态轴会略微降低推理速度（编译器无法做最激进的优化），需要在灵活性和性能之间权衡。

### ONNX Runtime推理

```python
import onnxruntime as ort
import numpy as np
import time

def verify_onnx_model(onnx_path: str):
    """验证ONNX模型的正确性"""
    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX模型验证通过")

    # 打印图信息
    print(f"输入: {[inp.name for inp in model.graph.input]}")
    print(f"输出: {[out.name for out in model.graph.output]}")


def create_onnx_session(onnx_path: str, use_gpu: bool = False):
    """创建ONNX Runtime推理会话"""
    # 选择执行提供程序（Execution Provider）
    providers = []
    if use_gpu:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')  # 回退到CPU

    # 创建会话选项
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4  # 使用4个线程

    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=providers
    )
    return session


def onnx_inference(session: ort.InferenceSession, input_ids: np.ndarray, attention_mask: np.ndarray):
    """使用ONNX Runtime进行推理"""
    # ONNX Runtime需要numpy数组，数据类型必须匹配
    inputs = {
        'input_ids': input_ids.astype(np.int64),
        'attention_mask': attention_mask.astype(np.int64),
    }

    outputs = session.run(None, inputs)  # None表示获取所有输出
    return outputs[0]  # logits


def benchmark_pytorch_vs_onnx(
    pytorch_model: nn.Module,
    onnx_session: ort.InferenceSession,
    batch_size: int = 8,
    seq_len: int = 128,
    num_runs: int = 100,
):
    """对比PyTorch和ONNX Runtime的推理速度"""
    pytorch_model.eval()

    # 准备测试数据
    input_ids_pt = torch.randint(0, 1000, (batch_size, seq_len))
    attn_mask_pt = torch.ones(batch_size, seq_len, dtype=torch.long)
    input_ids_np = input_ids_pt.numpy()
    attn_mask_np = attn_mask_pt.numpy()

    # 预热（排除JIT编译时间）
    for _ in range(10):
        with torch.no_grad():
            _ = pytorch_model(input_ids_pt, attn_mask_pt)
        _ = onnx_inference(onnx_session, input_ids_np, attn_mask_np)

    # PyTorch推理计时
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = pytorch_model(input_ids_pt, attn_mask_pt)
    pt_time = (time.perf_counter() - start) / num_runs * 1000

    # ONNX Runtime推理计时
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = onnx_inference(onnx_session, input_ids_np, attn_mask_np)
    ort_time = (time.perf_counter() - start) / num_runs * 1000

    print(f"PyTorch推理:      {pt_time:.2f} ms/batch")
    print(f"ONNX Runtime推理: {ort_time:.2f} ms/batch")
    print(f"加速比: {pt_time/ort_time:.2f}x")
```

---

## 23.5 其他优化技术

### 投机解码（Speculative Decoding）

自回归解码的核心瓶颈在于：每次只能生成一个token，GPU的大量并行计算能力被浪费。

**投机解码（Speculative Decoding）**的思路：

1. 用一个小型**草稿模型（Draft Model）**快速生成 $k$ 个候选token
2. 用大型**目标模型（Target Model）**并行验证这 $k$ 个token
3. 接受草稿模型猜对的前缀，从第一个错误位置重新生成

```
草稿模型（快速）：[the][quick][brown][fox]  → 4个候选token
目标模型（准确）：并行验证，发现前3个正确，第4个错误
→ 接受"the quick brown"，从"fox"的位置重新生成 → 实际获得3.x个token的输出
```

理论加速分析：若草稿模型的接受率为 $\alpha$（即每个token被目标模型接受的概率），则期望每次目标模型推理获得的token数为：

$$\mathbb{E}[\text{接受token数}] = \frac{1 - \alpha^{k+1}}{1 - \alpha}$$

当 $\alpha=0.8, k=5$ 时，期望接受 $\approx 3.4$ 个token，相比传统每次1个token有约3倍加速。

关键约束：**投机解码在数学上等价于目标模型的输出分布**，不损失任何精度。

```python
def speculative_decode_step(
    draft_model,
    target_model,
    input_ids: torch.Tensor,
    k: int = 5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """投机解码的单步实现（示意性代码）"""

    # 步骤1：草稿模型生成k个候选token
    draft_tokens = []
    draft_probs = []
    current_ids = input_ids.clone()

    for _ in range(k):
        with torch.no_grad():
            logits = draft_model(current_ids)[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            draft_tokens.append(next_token)
            draft_probs.append(probs)
            current_ids = torch.cat([current_ids, next_token], dim=1)

    # 步骤2：目标模型并行验证所有k个草稿token
    candidate_ids = torch.cat([input_ids] + [t for t in draft_tokens], dim=1)
    with torch.no_grad():
        target_logits = target_model(candidate_ids)

    # 步骤3：逐token验证（接受/拒绝采样）
    accepted_count = 0
    for i in range(k):
        target_probs = F.softmax(target_logits[:, len(input_ids[0]) - 1 + i, :] / temperature, dim=-1)
        draft_token = draft_tokens[i]

        # 接受概率：min(1, p_target / p_draft)
        accept_prob = torch.min(
            torch.ones_like(target_probs.gather(1, draft_token)),
            target_probs.gather(1, draft_token) / draft_probs[i].gather(1, draft_token)
        )

        if torch.rand(1) < accept_prob:
            accepted_count += 1
        else:
            break  # 拒绝，停止接受后续token

    # 返回接受的token
    if accepted_count > 0:
        return torch.cat(draft_tokens[:accepted_count], dim=1)
    else:
        # 从目标模型在拒绝位置重新采样
        reject_probs = F.softmax(target_logits[:, len(input_ids[0]) - 1, :] / temperature, dim=-1)
        return torch.multinomial(reject_probs, 1)
```

### 连续批处理（Continuous Batching）

传统批处理（Static Batching）的问题：一批请求中，有的序列很短，有的很长，短序列完成后必须等待长序列，导致GPU利用率低（"气泡"问题）。

**连续批处理**（又称动态批处理）允许在批处理中途插入新请求、剔除已完成请求：

```
时间步:  1  2  3  4  5  6  7  8
--------------------------------------
静态批处理:
请求A:   ■  ■  ■  ■  □  □  □  □  (4步完成，等待B)
请求B:   ■  ■  ■  ■  ■  ■  ■  ■  (8步)
GPU利用: 满  满  满  满  半  半  半  半  ← 利用率50%

连续批处理:
请求A:   ■  ■  ■  ■               (4步完成)
请求B:   ■  ■  ■  ■  ■  ■  ■  ■
请求C（新）:         ■  ■  ■  ■   (A完成后插入)
GPU利用: 满  满  满  满  满  满  满  满  ← 利用率接近100%
```

连续批处理配合PagedAttention，是现代LLM推理引擎（vLLM、TGI等）高吞吐量的核心机制。

### Tensor并行推理

对于超大模型（如GPT-3 175B），单张GPU无法容纳全部权重，需要**模型并行（Model Parallelism）**。

**Tensor并行（Tensor Parallelism）**将单个矩阵运算分割到多GPU上：

对于线性层 $Y = XW$，可以按列分割 $W$：

$$Y = X[W_1 | W_2] = [XW_1 | XW_2]$$

每个GPU计算部分结果，最后AllReduce合并：

```
GPU0: Y0 = X @ W0   (W的前半部分)
GPU1: Y1 = X @ W1   (W的后半部分)
↓ AllReduce
Y = concat(Y0, Y1)
```

Megatron-LM将Tensor并行系统化，支持Transformer中的注意力层和FFN层的列并行/行并行分割。

### vLLM简介

**vLLM**是UC Berkeley开发的高吞吐量LLM推理引擎，集成了本章几乎所有关键技术：

| 技术 | vLLM实现 |
|------|----------|
| KV Cache管理 | PagedAttention |
| 批处理 | 连续批处理（Continuous Batching）|
| 量化 | GPTQ、AWQ、INT8 |
| 并行 | Tensor并行、Pipeline并行 |
| 采样 | 投机解码（可选）|

基本使用：

```python
# pip install vllm
from vllm import LLM, SamplingParams

# 加载模型（自动应用PagedAttention等优化）
llm = LLM(model="facebook/opt-1.3b", tensor_parallel_size=1)

# 批量推理
prompts = [
    "The future of AI is",
    "In machine learning, the key challenge is",
]
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
    print()
```

相比原生HuggingFace推理，vLLM在高并发场景下吞吐量通常有**5-24倍**的提升。

---

## 本章小结

本章介绍了Transformer推理优化的核心技术栈：

| 技术 | 核心思想 | 收益 | 适用场景 |
|------|----------|------|----------|
| KV Cache | 缓存历史K/V，避免重复计算 | 将解码从O(T²)降至O(T) | 所有自回归生成 |
| PagedAttention | 分页管理KV Cache内存 | 内存利用率从~40%→~90% | 多并发服务 |
| INT8量化（PTQ） | 权重从FP16→INT8 | 模型体积减半，推理加速2x | 部署，精度要求适中 |
| INT4量化（GPTQ/AWQ） | 权重从FP16→INT4 | 模型体积压缩4x | 边缘设备，内存受限 |
| 知识蒸馏 | 小模型学大模型输出分布 | 模型缩小40-60%，精度保留95%+ | 需要训练的场景 |
| ONNX Runtime | 跨框架优化图执行 | CPU推理加速1.5-3x | 跨平台部署 |
| 投机解码 | 小模型猜测+大模型验证 | 吞吐量提升2-4x | 延迟敏感应用 |
| 连续批处理 | 动态管理推理批次 | 服务吞吐量提升5-10x | 在线推理服务 |
| Tensor并行 | 矩阵运算分布到多GPU | 支持超大模型，近线性加速 | 超大模型（>13B） |

**技术选择建议**：

- **追求最快上线**：PTQ（INT8）+ ONNX Runtime + KV Cache
- **极限压缩模型**：知识蒸馏 → GPTQ/AWQ量化
- **高并发服务**：vLLM（集成PagedAttention + 连续批处理）
- **延迟优先**：投机解码 + KV Cache

---

## 代码实战

以下是一个完整的推理优化实验，涵盖KV Cache、量化和ONNX导出三个核心技术，并进行性能基准测试。

```python
"""
推理优化综合实验
包含：KV Cache实现、INT8量化、ONNX导出、性能基准测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from typing import Optional, Tuple, List

# ─── 1. 模型定义 ─────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """因果自注意力（支持KV Cache）"""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # 预计算因果掩码
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        B, T, C = x.shape

        QKV = self.qkv_proj(x)
        Q, K, V = QKV.split(self.d_model, dim=-1)

        # 分割多头
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # 拼接历史KV Cache
        if past_kv is not None:
            K = torch.cat([past_kv[0], K], dim=2)
            V = torch.cat([past_kv[1], V], dim=2)

        present_kv = (K, V) if use_cache else None
        total_T = K.shape[2]

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 应用因果掩码
        if T > 1:
            mask = self.causal_mask[total_T-T:total_T, :total_T]
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out, present_kv


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, present_kv = self.attn(self.ln1(x), past_kv, use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, present_kv


class MiniGPT(nn.Module):
    """小型GPT模型，用于演示"""

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        B, T = input_ids.shape

        # 如果有缓存，从偏移位置开始计算位置编码
        past_len = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]

        pos = torch.arange(past_len, past_len + T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        present_key_values = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values else None
            x, present_kv = block(x, past_kv, use_cache)
            present_key_values.append(present_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, present_key_values if use_cache else None


# ─── 2. KV Cache生成 ─────────────────────────────────────────────────────────

@torch.no_grad()
def generate_without_cache(model: MiniGPT, prompt_ids: torch.Tensor, max_new_tokens: int = 50):
    """不使用KV Cache的朴素生成"""
    model.eval()
    ids = prompt_ids.clone()

    for _ in range(max_new_tokens):
        logits, _ = model(ids)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_token], dim=1)

    return ids


@torch.no_grad()
def generate_with_cache(model: MiniGPT, prompt_ids: torch.Tensor, max_new_tokens: int = 50):
    """使用KV Cache的加速生成"""
    model.eval()

    # 预填充阶段
    logits, past_key_values = model(prompt_ids, use_cache=True)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]

    # 解码阶段：每次只输入一个token
    for _ in range(max_new_tokens - 1):
        logits, past_key_values = model(
            next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)

    return torch.cat([prompt_ids] + generated, dim=1)


# ─── 3. 量化工具 ──────────────────────────────────────────────────────────────

def quantize_model_weights(model: nn.Module, bits: int = 8) -> dict:
    """量化模型所有Linear层的权重，返回量化后的参数字典"""
    quantized_state = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data

            if bits == 8:
                # INT8对称量化
                max_val = weight.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
                scale = max_val / 127.0
                q_weight = (weight / scale).round().clamp(-128, 127).to(torch.int8)
                quantized_state[name] = {'weight': q_weight, 'scale': scale, 'bits': 8}
            elif bits == 4:
                # INT4量化（简化版，实际INT4需要打包存储）
                max_val = weight.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
                scale = max_val / 7.0
                q_weight = (weight / scale).round().clamp(-8, 7).to(torch.int8)
                quantized_state[name] = {'weight': q_weight, 'scale': scale, 'bits': 4}

    return quantized_state


def estimate_model_size(model: nn.Module) -> dict:
    """估计模型各精度的存储大小"""
    total_params = sum(p.numel() for p in model.parameters())

    return {
        'FP32 (MB)': total_params * 4 / 1024 / 1024,
        'FP16 (MB)': total_params * 2 / 1024 / 1024,
        'INT8 (MB)': total_params * 1 / 1024 / 1024,
        'INT4 (MB)': total_params * 0.5 / 1024 / 1024,
        'total_params': total_params,
    }


# ─── 4. ONNX导出（需要onnx和onnxruntime库）────────────────────────────────────

class MiniGPTForONNX(nn.Module):
    """去掉KV Cache接口的简化版，适合ONNX导出"""

    def __init__(self, vocab_size=1000, d_model=128, num_heads=4, num_layers=2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.transformer(x)
        x = self.ln(x)
        return self.head(x)  # (B, T, vocab_size)


def export_and_verify_onnx(model: nn.Module, save_path: str = "/tmp/mini_gpt.onnx"):
    """导出ONNX并验证输出一致性"""
    model.eval()

    # 虚拟输入
    dummy_ids = torch.randint(0, 1000, (2, 16))

    # 导出
    torch.onnx.export(
        model,
        dummy_ids,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq'},
            'logits':    {0: 'batch', 1: 'seq'},
        },
        verbose=False,
    )

    # 验证输出一致性
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(save_path, providers=['CPUExecutionProvider'])

        with torch.no_grad():
            pt_out = model(dummy_ids).numpy()

        ort_out = session.run(None, {'input_ids': dummy_ids.numpy().astype(np.int64)})[0]

        max_diff = np.abs(pt_out - ort_out).max()
        print(f"导出成功：{save_path}")
        print(f"PyTorch vs ONNX最大差异: {max_diff:.6f} (应接近0)")

        return session
    except ImportError:
        print(f"导出成功：{save_path}（未安装onnxruntime，跳过验证）")
        return None


# ─── 5. 性能基准测试 ─────────────────────────────────────────────────────────

def run_benchmarks():
    """运行所有基准测试"""
    print("=" * 60)
    print("Transformer推理优化基准测试")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"运行设备: {device}\n")

    # 创建模型
    model = MiniGPT(
        vocab_size=1000, d_model=256, num_heads=8, num_layers=6
    ).to(device)
    model.eval()

    # 测试参数
    batch_size = 4
    prompt_len = 32
    gen_len = 64
    num_runs = 5

    prompt = torch.randint(0, 1000, (batch_size, prompt_len), device=device)

    # ── 基准测试1：KV Cache对比 ──
    print("【基准测试1】KV Cache vs 无缓存生成")
    print(f"Batch: {batch_size}, Prompt: {prompt_len}, 生成: {gen_len} tokens")

    # 不使用KV Cache
    times_no_cache = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = generate_without_cache(model, prompt, max_new_tokens=gen_len)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_no_cache.append(time.perf_counter() - start)
    avg_no_cache = np.mean(times_no_cache) * 1000

    # 使用KV Cache
    times_cache = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = generate_with_cache(model, prompt, max_new_tokens=gen_len)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_cache.append(time.perf_counter() - start)
    avg_cache = np.mean(times_cache) * 1000

    print(f"  无KV Cache:    {avg_no_cache:.1f} ms")
    print(f"  有KV Cache:    {avg_cache:.1f} ms")
    print(f"  KV Cache加速:  {avg_no_cache / avg_cache:.2f}x\n")

    # ── 基准测试2：模型大小对比 ──
    print("【基准测试2】量化压缩比")
    size_info = estimate_model_size(model)
    print(f"  参数量:    {size_info['total_params']:,}")
    print(f"  FP32大小:  {size_info['FP32 (MB)']:.2f} MB")
    print(f"  FP16大小:  {size_info['FP16 (MB)']:.2f} MB")
    print(f"  INT8大小:  {size_info['INT8 (MB)']:.2f} MB")
    print(f"  INT4大小:  {size_info['INT4 (MB)']:.2f} MB\n")

    # ── 基准测试3：ONNX导出 ──
    print("【基准测试3】ONNX导出与验证")
    onnx_model = MiniGPTForONNX().eval()
    ort_session = export_and_verify_onnx(onnx_model, save_path="/tmp/mini_gpt_demo.onnx")

    if ort_session is not None:
        # ONNX Runtime vs PyTorch 速度对比
        test_ids = torch.randint(0, 1000, (batch_size, 64))

        # PyTorch
        times_pt = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = onnx_model(test_ids)
            times_pt.append(time.perf_counter() - start)

        # ONNX Runtime
        import onnxruntime as ort
        times_ort = []
        for _ in range(20):
            start = time.perf_counter()
            _ = ort_session.run(None, {'input_ids': test_ids.numpy().astype(np.int64)})
            times_ort.append(time.perf_counter() - start)

        avg_pt = np.mean(times_pt) * 1000
        avg_ort = np.mean(times_ort) * 1000
        print(f"  PyTorch推理:      {avg_pt:.2f} ms")
        print(f"  ONNX Runtime推理: {avg_ort:.2f} ms")
        print(f"  加速比: {avg_pt/avg_ort:.2f}x")

    print("\n" + "=" * 60)
    print("基准测试完成")
    print("=" * 60)


if __name__ == '__main__':
    run_benchmarks()
```

---

## 练习题

### 基础题

**题目1**：KV Cache内存计算

一个Transformer模型有以下参数：
- 层数 $L = 32$
- 注意力头数 $H = 32$
- 每头维度 $d_k = 128$（即 $d_{model} = 4096$）
- 序列长度 $T = 4096$
- 精度：FP16（每个元素2字节）

请计算：
1. 单个样本的KV Cache内存占用（MB）
2. 同时服务100个并发请求时的KV Cache总内存占用（GB）
3. 若使用INT8量化KV Cache，可以节省多少内存？

**题目2**：量化误差分析

对以下权重向量进行INT8对称量化，然后反量化，计算量化误差（均方误差）：

$$W = [0.1, -0.5, 1.2, -0.8, 0.3, -1.5, 0.7, -0.2]$$

1. 计算缩放因子 $s$
2. 量化后的整数值
3. 反量化后的近似值
4. 均方误差（MSE）

---

### 中级题

**题目3**：实现带温度的投机解码验证

投机解码的接受/拒绝步骤需要保持与目标模型分布一致。设草稿模型在某位置的预测概率为：

$$p_{\text{draft}} = [0.5, 0.3, 0.1, 0.1]$$（对应4个token）

目标模型的预测概率为：

$$p_{\text{target}} = [0.4, 0.35, 0.15, 0.1]$$

草稿模型采样了token 0（概率0.5的那个）。

1. 计算接受概率 $\min(1, p_{\text{target}}[0] / p_{\text{draft}}[0])$
2. 如果token 0被拒绝，需要从调整后的分布 $p'$ 中重新采样，其中 $p'_i \propto \max(0, p_{\text{target}}[i] - p_{\text{draft}}[i])$。计算 $p'$。
3. 验证：若每个位置的接受率均为 $\alpha=0.7$，草稿步数 $k=4$，期望接受token数是多少？

**题目4**：知识蒸馏超参数影响

给定教师模型的logits为 $z_t = [3.0, 1.0, -1.0, -3.0]$，学生模型的logits为 $z_s = [2.0, 1.5, 0.5, -4.0]$，真实标签为类别0。

分别在温度 $T \in \{1, 2, 4, 8\}$ 下：
1. 计算教师模型的软标签概率分布
2. 计算蒸馏损失 $T^2 \cdot \text{KL}(p_t^T \| p_s^T)$
3. 分析温度如何影响梯度信号的"强度"

---

### 提高题

**题目5**：设计一个完整的量化-蒸馏联合优化流程

考虑以下场景：你需要将一个12层、768维的BERT模型部署到移动设备（内存限制100MB，延迟要求50ms）。

原始BERT-base大小：约440MB，延迟约200ms（CPU）。

请设计一个**两阶段优化方案**：

**第一阶段**：通过知识蒸馏压缩模型
- 目标：从12层蒸馏到4层，保留95%以上的GLUE精度
- 需要设计：学生模型架构、蒸馏损失函数、训练策略

**第二阶段**：通过量化进一步压缩
- 目标：将蒸馏后的模型从FP32量化到INT8
- 需要考虑：哪些层更敏感？是用PTQ还是QAT？

请回答：
1. 设计蒸馏学生模型的具体架构（层数、隐层维度、注意力头数）
2. 给出蒸馏阶段的完整损失函数（含各项权重的选择依据）
3. 分析哪些层量化后精度损失最大（embedding层、注意力层、FFN层），给出处理策略
4. 估算最终模型大小和推理延迟，验证是否满足部署要求

---

## 练习答案

### 答案1：KV Cache内存计算

**解题过程**：

KV Cache由K矩阵和V矩阵组成，每层各一个。

单层KV Cache大小（字节数）：

$$\text{单层} = 2 \times T \times d_{\text{model}} \times \text{bytes} = 2 \times 4096 \times 4096 \times 2$$

$$= 67,108,864 \text{ 字节} = 64 \text{ MB}$$

全部32层（单样本）：

$$\text{总KV Cache} = 32 \times 64 \text{ MB} = 2048 \text{ MB} = 2 \text{ GB}$$

100个并发请求：

$$\text{总计} = 100 \times 2 \text{ GB} = 200 \text{ GB}$$

这就是为什么A100（80GB显存）在服务大模型时并发数很有限，PagedAttention通过内存共享和按需分配显著缓解了这个问题。

INT8量化KV Cache（每元素从2字节→1字节）：

$$\text{节省} = 200 \text{ GB} \times 50\% = 100 \text{ GB}$$

---

### 答案2：量化误差分析

**解题过程**：

$$W = [0.1, -0.5, 1.2, -0.8, 0.3, -1.5, 0.7, -0.2]$$

**步骤1**：计算缩放因子

$$\max(|W|) = 1.5, \quad s = \frac{1.5}{127} \approx 0.01181$$

**步骤2**：量化

$$W_q = \text{round}(W / s) = \text{round}([8.47, -42.34, 101.61, -67.74, 25.40, -127.0, 59.27, -16.94])$$

$$W_q = [8, -42, 102, -68, 25, -127, 59, -17]$$

**步骤3**：反量化

$$\hat{W} = W_q \times s = [0.0945, -0.4960, 1.2045, -0.8031, 0.2953, -1.5000, 0.6968, -0.2008]$$

**步骤4**：均方误差

$$\text{MSE} = \frac{1}{8}\sum_{i=1}^{8}(W_i - \hat{W}_i)^2$$

$$= \frac{(0.1-0.0945)^2 + (-0.5+0.496)^2 + ... }{8} \approx 0.0000213$$

MSE约为 $2.13 \times 10^{-5}$，说明INT8量化精度损失很小。

---

### 答案3：投机解码验证

**1. 接受概率**

$$\text{接受概率} = \min\left(1, \frac{p_{\text{target}}[0]}{p_{\text{draft}}[0]}\right) = \min\left(1, \frac{0.4}{0.5}\right) = 0.8$$

有80%的概率接受草稿模型的token 0。

**2. 拒绝后重采样分布**

$$p'_i = \max(0, p_{\text{target}}[i] - p_{\text{draft}}[i])$$

$$p'_0 = \max(0, 0.4 - 0.5) = 0$$
$$p'_1 = \max(0, 0.35 - 0.3) = 0.05$$
$$p'_2 = \max(0, 0.15 - 0.1) = 0.05$$
$$p'_3 = \max(0, 0.1 - 0.1) = 0$$

归一化后：$p' = [0, 0.5, 0.5, 0]$

即若token 0被拒绝，等概率在token 1和token 2中采样。

**3. 期望接受token数**

$$\mathbb{E}[\text{接受数}] = \sum_{n=1}^{k} n \cdot \alpha^{n-1}(1-\alpha) + k \cdot \alpha^k$$

当 $\alpha=0.7, k=4$：

$$= 1 \cdot 0.3 + 2 \cdot 0.7 \times 0.3 + 3 \cdot 0.49 \times 0.3 + 4 \times 0.343 \times 0.3 + 4 \times 0.2401$$

$$= 0.3 + 0.42 + 0.441 + 0.4116 + 0.9604 \approx 2.53 \text{ tokens}$$

相比传统每次生成1个token，吞吐量提升约2.5倍。

---

### 答案4：知识蒸馏温度分析

以 $T=1$ 为例（其余类推）：

教师softmax：$\text{softmax}([3, 1, -1, -3]/1) = [0.867, 0.117, 0.016, 0.002]$（近似）

学生softmax：$\text{softmax}([2, 1.5, 0.5, -4]/1) = [0.567, 0.343, 0.085, 0.001]$（近似）

$$\text{KL}(T^2=1) = \sum_i p_t \log\frac{p_t}{p_s} \approx 0.867 \log\frac{0.867}{0.567} + ... \approx 0.325$$

当 $T=4$ 时，分布变平滑：

教师：$\approx [0.40, 0.27, 0.18, 0.12]$（更均匀）

学生：$\approx [0.35, 0.31, 0.22, 0.08]$（更均匀）

$$\text{KL}(T^2=16) = 16 \times \text{KL}_{\text{软}} \approx 16 \times 0.012 \approx 0.19$$

**分析**：高温度使分布趋于均匀，KL散度的绝对值减小，但乘以 $T^2$ 后与低温时量级相近。高温的真正价值在于：使"非答案类别"的相对关系更清晰，传递更多类间知识。

---

### 答案5：量化-蒸馏联合优化方案

**目标**：BERT-base（440MB, 200ms）→ 移动端模型（<100MB, <50ms）

#### 第一阶段：知识蒸馏

**学生模型架构**（4层BERT-small变体）：
- 层数：4层（原12层的1/3）
- 隐层维度：384（原768的1/2）
- 注意力头数：6（保持每头维度64不变）
- FFN维度：1536（4倍隐层维度）
- 参数量：约29M，大小约116MB（FP32）

**蒸馏损失**：

$$\mathcal{L} = \underbrace{0.5 \cdot \mathcal{L}_{\text{task}}}_{\text{任务损失}} + \underbrace{0.3 \cdot \mathcal{L}_{\text{logit-KD}}(T=4)}_{\text{输出蒸馏}} + \underbrace{0.2 \cdot \mathcal{L}_{\text{attn}}}_{\text{注意力蒸馏}}$$

选择依据：
- 保留任务损失（0.5权重）确保最终性能
- 输出蒸馏（0.3，$T=4$）传递类间知识
- 注意力蒸馏（0.2）传递中间层语义

初始化：取教师第1、4、8、12层权重初始化学生4层（每隔3层取一个），显著加速蒸馏收敛。

#### 第二阶段：量化（PTQ → QAT）

**敏感性分析**：
- **Embedding层**：参数多（30522×768≈23M）但对量化不敏感，保持FP16
- **第一层和最后层注意力**：接收原始输入/产生最终输出，量化最敏感，建议QAT或保持FP16
- **FFN中间层**：经过GELU后激活分布复杂，使用逐通道量化
- **中间注意力层**：相对不敏感，INT8 PTQ通常损失<0.5 GLUE

**推荐策略**：先用PTQ（INT8）获得基线，若精度损失>1%，对敏感层用QAT微调（1-2 epoch）。

**最终估算**：
- 蒸馏后：4层，约29M参数，FP32约116MB
- INT8量化后（仅Linear层）：约32MB
- 量化后推理延迟估算（基于BERT-small线性外推）：$200 \times \frac{4}{12} \times \frac{384^2}{768^2} \times 0.5 \approx 28 \text{ ms}$

结论：满足100MB内存和50ms延迟要求。

**精度预期**：DistilBERT（6层）保留97.5%精度。4层蒸馏+量化预计保留93-95%精度，对大多数移动端应用可接受。
