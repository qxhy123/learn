# 前言：如何使用本教程

## 教程设计理念

Transformer架构自2017年提出以来，已经彻底改变了自然语言处理和计算机视觉领域。从BERT到GPT，从ViT到CLIP，几乎所有现代AI系统都建立在Transformer的基础之上。本教程的设计基于以下理念：

1. **从零实现优先**：不使用高层封装，手写每个核心组件
2. **数学与代码并重**：公式推导与PyTorch实现相辅相成
3. **循序渐进**：从注意力机制到完整LLM，逐步构建知识体系
4. **实战驱动**：每章都有可运行的完整代码示例
5. **中文优先**：专为中文读者设计，术语准确，表达清晰

## 为什么从零实现？

### 使用高层API的问题

```python
# 使用PyTorch的nn.Transformer
model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6)
output = model(src, tgt)  # 一行代码，但你真的理解发生了什么吗？
```

### 从零实现的价值

```python
# 从零实现多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性投影
        Q = self.W_q(query)  # (batch, seq, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # 分割为多头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和
        context = torch.matmul(attn_weights, V)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        return output, attn_weights
```

通过从零实现，你将真正理解：
- Q、K、V的含义和作用
- 多头是如何并行计算的
- 掩码是如何工作的
- 每一步张量的形状变化

## 章节结构

每章采用统一的结构，便于系统学习：

### 1. 学习目标
每章开头列出5个学习目标，帮助读者明确本章重点。

### 2. 正文内容
每章包含5个小节，循序渐进地展开主题：
- 概念引入与直觉理解
- 数学公式与推导
- PyTorch代码实现
- 常见问题与技巧
- 综合案例演示

### 3. 本章小结
以表格或列表形式总结核心概念与公式。

### 4. 代码实战
每章的特色部分，包含：
- 完整的可运行代码
- 详细的注释说明
- 可视化输出

### 5. 练习题
每章5道练习题，分三个难度级别：
- 基础题（2道）：检验概念理解
- 中级题（2道）：代码实现与调试
- 提高题（1道）：综合项目或论文复现

### 6. 练习答案
详细的解答过程，帮助自学者检验学习效果。

## 数学符号约定

本教程使用以下数学符号：

| 符号 | 含义 | 示例 |
|------|------|------|
| $d_{model}$ | 模型维度 | 512, 768, 1024 |
| $d_k, d_v$ | 键/值维度 | 64 |
| $h$ | 注意力头数 | 8, 12, 16 |
| $N$ | 编码器/解码器层数 | 6, 12 |
| $L$ | 序列长度 | 512, 2048 |
| $B$ | 批次大小 | 32 |
| $V$ | 词表大小 | 30000, 50000 |

### 张量形状约定

```python
# 输入序列: (batch_size, seq_length, d_model)
x = torch.randn(32, 128, 512)  # B=32, L=128, d=512

# 注意力权重: (batch_size, num_heads, seq_length, seq_length)
attn = torch.randn(32, 8, 128, 128)  # B=32, h=8, L=128

# 输出序列: (batch_size, seq_length, d_model)
output = torch.randn(32, 128, 512)
```

## 代码约定

本教程使用以下代码约定：

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# 类名使用大驼峰
class TransformerEncoder(nn.Module):
    pass

# 函数名使用小写下划线
def scaled_dot_product_attention(query, key, value, mask=None):
    pass

# 变量使用小写下划线
batch_size = 32
seq_length = 512
d_model = 768

# 常量使用全大写
MAX_SEQ_LENGTH = 2048
NUM_HEADS = 12
NUM_LAYERS = 6
```

## 输出格式说明

代码示例的输出使用以下格式：

```python
x = torch.randn(2, 4, 8)
print(x.shape)
# 输出: torch.Size([2, 4, 8])

attn = MultiHeadAttention(d_model=8, num_heads=2)
output, weights = attn(x, x, x)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
# 输出: Output shape: torch.Size([2, 4, 8])
# 输出: Attention weights shape: torch.Size([2, 2, 4, 4])
```

## 环境准备

### 推荐开发环境

1. **Python解释器**：Python 3.9或更高版本
2. **深度学习框架**：PyTorch 2.0+（推荐支持Flash Attention）
3. **代码编辑器**：VS Code或PyCharm
4. **计算资源**：建议有GPU（至少8GB显存）

### 安装依赖

```bash
# 1. 创建虚拟环境
python -m venv transformer-env
source transformer-env/bin/activate  # Linux/macOS

# 2. 安装PyTorch（选择适合你系统的版本）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU only
pip install torch torchvision torchaudio

# 3. 安装其他依赖
pip install transformers datasets tokenizers
pip install numpy pandas matplotlib seaborn
pip install jupyter tqdm einops

# 4. 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

### 验证GPU

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## 学习建议

### 对于深度学习初学者

1. **先修课程**：确保理解反向传播、梯度下降、损失函数
2. **PyTorch基础**：熟悉张量操作、nn.Module、forward方法
3. **线性代数**：矩阵乘法、转置、点积
4. **逐章学习**：不要跳过基础章节

### 对于有经验的开发者

1. **关注实现细节**：即使概念熟悉，也要理解实现
2. **对比不同实现**：比较本教程与官方实现的差异
3. **做提高题**：挑战论文复现和性能优化
4. **阅读源码**：结合Hugging Face Transformers源码学习

### 对于研究者

1. **深入数学推导**：每个公式都要能手推
2. **关注第16章**：现代大模型的最新技术
3. **关注第20-22章**：高效Transformer和可解释性
4. **扩展阅读**：参考每章末尾的论文列表

## 与其他教程的关系

本教程是"从零到高阶"系列的一部分：

```
Python教程 ────→ 深度学习基础
                      │
                      ↓
              Transformer教程（本教程）
                      │
              ┌───────┼───────┐
              ↓       ↓       ↓
           NLP应用  CV应用  多模态应用
```

建议先完成Python教程中的深度学习基础部分，再学习本教程。

## 排版说明

- **粗体**：重要概念、关键词
- *斜体*：术语首次出现、强调
- `代码字体`：代码、类名、函数名、变量名
- > 引用块：重要提示、注意事项
- $数学公式$：行内公式
- $$数学公式$$：独立公式块

## 常见问题

**Q: 学习本教程需要多长时间？**
A: 因人而异。有深度学习基础的学习者完整学习约需2-3个月，重点学习预训练模型部分约需1个月。

**Q: 需要GPU吗？**
A: 基础章节（1-12章）在CPU上即可运行。预训练和微调章节（13-24章）强烈建议使用GPU。

**Q: 遇到问题怎么办？**
A: 首先仔细阅读错误信息，检查张量形状是否匹配。然后参考PyTorch文档，也可以在项目Issue中提问。

**Q: 可以直接学习预训练模型章节吗？**
A: 如果已熟悉注意力机制和Transformer架构，可以直接从第13章开始。否则建议先完成前9章。

**Q: 本教程与Hugging Face教程的区别？**
A: 本教程注重从零实现，帮助理解原理；Hugging Face教程注重使用API，帮助快速上手。两者互补。

---

*准备好了吗？让我们开始Transformer之旅！*

[下一章：序列建模基础](./part1-foundations/01-sequence-modeling.md)
