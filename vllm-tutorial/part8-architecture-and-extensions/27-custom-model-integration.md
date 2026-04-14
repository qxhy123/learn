# 第27章：自定义模型接入

> vLLM 已经支持了大量模型架构，但如果你有一个自定义模型，理解如何将它接入 vLLM 是掌握系统的最佳方式。

---

## 学习目标

学完本章，你将能够：

1. 理解 vLLM 模型接口的基本要求
2. 将一个 HuggingFace 模型转换为 vLLM 兼容格式
3. 实现自定义注意力层以支持 PagedAttention
4. 注册和测试自定义模型
5. 处理自定义模型的常见问题

---

## 27.1 模型接口要求

### vLLM 模型与 HF 模型的区别

```python
# HuggingFace 模型
class HFModel(nn.Module):
    def forward(self, input_ids, attention_mask, past_key_values=None):
        # attention_mask: 标准掩码
        # past_key_values: 元组的元组，连续存储
        ...

# vLLM 模型
class VLLMModel(nn.Module):
    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        # positions: 位置编码索引
        # kv_caches: PagedAttention 管理的分页 KV Cache
        # attn_metadata: 注意力计算的元数据（块表等）
        ...
```

核心区别：
1. KV Cache 由 PagedAttention 管理（分页、不连续）
2. 注意力计算使用 vLLM 的注意力后端
3. 需要适配 vLLM 的权重加载接口

---

## 27.2 实现步骤

### 步骤 1：创建模型文件

```python
# vllm/model_executor/models/my_model.py
from typing import Optional, List
import torch
from torch import nn
from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, RowParallelLinear
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead
)


class MyAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.qkv_proj = ColumnParallelLinear(
            hidden_size, (num_heads + 2 * num_kv_heads) * head_dim
        )
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size)
        self.rotary_emb = get_rope(head_dim, ...)

        # 关键：使用 vLLM 的 Attention 层
        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            num_kv_heads=num_kv_heads,
        )

    def forward(self, hidden_states, positions, kv_cache, attn_metadata):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([...], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        # 使用 vLLM 的 Attention（自动处理 PagedAttention）
        output = self.attn(q, k, v, kv_cache, attn_metadata)
        return self.o_proj(output)
```

### 步骤 2：实现权重加载

```python
class MyForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MyDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, positions, kv_caches[i], attn_metadata)
        logits = self.lm_head(hidden)
        return logits

    def load_weights(self, weights):
        """映射 HuggingFace 权重名到 vLLM 参数名"""
        params_mapping = {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.layers.{}.self_attn.q_proj": "layers.{}.attn.qkv_proj",
            # ... 其他映射
        }
        for name, param in weights:
            mapped_name = self._map_name(name, params_mapping)
            self.get_parameter(mapped_name).copy_(param)
```

### 步骤 3：注册模型

```python
# vllm/model_executor/models/__init__.py
# 添加模型注册

_MODELS = {
    # ... 现有模型
    "MyModelForCausalLM": ("my_model", "MyForCausalLM"),
}
```

### 步骤 4：测试

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/my-model",
    trust_remote_code=True,  # 如果需要
)

outputs = llm.generate(["Hello!"], SamplingParams(max_tokens=50))
print(outputs[0].outputs[0].text)
```

---

## 27.3 常见挑战

### 挑战 1：权重名映射

HF 模型和 vLLM 模型的参数名通常不同，需要正确映射。

### 挑战 2：注意力层适配

必须使用 vLLM 的 `Attention` 层来获得 PagedAttention 支持。不能直接用 `F.scaled_dot_product_attention`。

### 挑战 3：张量并行

如果需要支持多 GPU，线性层需要使用 `ColumnParallelLinear` 和 `RowParallelLinear`。

### 挑战 4：特殊结构

某些模型有特殊结构（如 MoE、滑动窗口注意力），需要额外适配。

---

## 27.4 使用 trust_remote_code

对于带有自定义代码的 HF 模型，vLLM 可以尝试自动适配：

```python
llm = LLM(
    model="org/custom-model",
    trust_remote_code=True,
)
```

但这不是万能的——复杂的自定义架构仍然可能需要手动适配。

---

## 本章小结

| 步骤 | 要点 |
|------|------|
| 1. 创建模型文件 | 使用 vLLM 的 Attention 层和并行线性层 |
| 2. 权重加载 | 正确映射 HF 权重名到 vLLM 参数名 |
| 3. 注册模型 | 在 `__init__.py` 中注册 |
| 4. 测试验证 | 对比与 HF 推理的输出一致性 |

---

## 练习题

### 思考题

1. 为什么 vLLM 的注意力层必须替换标准的 `F.scaled_dot_product_attention`？
2. 模型接入 vLLM 后，哪些优化特性（PagedAttention、连续批处理等）是自动获得的？
3. 如果一个模型使用了自定义的位置编码方案，接入 vLLM 时需要注意什么？
