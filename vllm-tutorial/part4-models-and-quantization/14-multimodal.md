# 第14章：多模态模型推理

> 大语言模型不再只理解文字。通过视觉编码器，它们可以"看"图片、"看"视频。vLLM 的多模态支持让你可以用统一的 API 处理图文混合输入。

---

## 学习目标

学完本章，你将能够：

1. 理解多模态模型（VLM）的基本架构
2. 使用 vLLM 进行图文推理
3. 通过 OpenAI 兼容 API 发送多模态请求
4. 处理多图输入和视频输入
5. 了解多模态推理的性能特点和限制

---

## 14.1 多模态模型架构

### VLM 的基本结构

视觉语言模型（Vision-Language Model, VLM）通常由三部分组成：

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Vision      │     │  Projection  │     │  Language     │
│  Encoder     │ ──→ │  Layer       │ ──→ │  Model        │
│  (ViT/CLIP)  │     │  (MLP/QFormer)│     │  (LLM)        │
└──────────────┘     └──────────────┘     └──────────────┘
     ↑                                          ↑
   图像输入                                    文本输入
```

1. **视觉编码器**：将图像编码为特征向量序列
2. **投影层**：将视觉特征映射到语言模型的嵌入空间
3. **语言模型**：处理融合后的图文 token 序列

### 对推理系统的影响

- 图像 token 通常很多（一张图 ~576-2048 个 token）
- Prefill 阶段计算量显著增加
- KV Cache 也需要为图像 token 分配空间

---

## 14.2 离线多模态推理

### 使用 LLM 类

```python
from vllm import LLM, SamplingParams

# 加载多模态模型
llm = LLM(
    model="Qwen/Qwen2-VL-7B-Instruct",
    max_model_len=4096,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
```

### 图文推理

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    max_model_len=4096,
)

# 使用 chat 格式发送图文输入
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
            {"type": "text", "text": "这张图片里有什么？"},
        ],
    }
]

outputs = llm.chat([messages], SamplingParams(max_tokens=256))
print(outputs[0].outputs[0].text)
```

### 本地图片

```python
import base64

# 从本地文件读取图片
with open("photo.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
            {"type": "text", "text": "描述这张图片。"},
        ],
    }
]
```

---

## 14.3 API 服务器多模态推理

### 启动多模态服务

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct \
    --max-model-len 4096
```

### 使用 OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/chart.png"},
                },
                {"type": "text", "text": "请解读这张图表的数据趋势。"},
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0].message.content)
```

### 多图输入

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "image1.jpg"}},
                {"type": "image_url", "image_url": {"url": "image2.jpg"}},
                {"type": "text", "text": "比较这两张图片的区别。"},
            ],
        }
    ],
    max_tokens=300,
)
```

---

## 14.4 V1 源码中的多模态调度机制

在 V1 架构中，多模态推理不只是"图片变成更多 token"这么简单。当前源码有两个专门的子系统来管理多模态资源。

### Encoder Cache Manager

多模态模型的视觉编码器输出需要缓存，调度器通过 `EncoderCacheManager`（`v1/core/encoder_cache_manager.py`）管理：

- 每个多模态输入有独立的 encoder cache 槽位
- 调度器在接纳请求时要同时检查 encoder cache 是否有空间
- 请求完成后 encoder cache 被释放

### Encoder Compute Budget

调度器维护一个独立的 `encoder_compute_budget`，和 token budget 并行约束：

```python
encoder_compute_budget = self.max_num_encoder_input_tokens
```

每当一个带多模态输入的请求被调度，encoder budget 会被扣减。这样可以防止太多图像/视频请求同时 prefill，导致 GPU 计算量暴增。

### 对调度的直接影响

在 `scheduler.py` 的 running 和 waiting 流程中，encoder 相关逻辑随处可见：

```python
if request.has_encoder_inputs:
    (encoder_inputs_to_schedule, num_new_tokens, new_encoder_compute_budget,
     external_load_encoder_input) = self._try_schedule_encoder_inputs(
        request, num_computed_tokens, num_new_tokens, encoder_compute_budget)
    if num_new_tokens == 0:
        break  # encoder 预算耗尽
```

这意味着多模态请求有**双重准入门禁**：既要满足 token budget，又要满足 encoder budget。

---

## 14.5 性能特点

### 图像 token 的影响

```
纯文本请求 (100 个 token):
  Prefill: ~5ms
  KV Cache: ~0.1 GB

图文请求 (100 文本 + 576 图像 token):
  Prefill: ~30ms  (6× 更慢)
  KV Cache: ~0.7 GB (7× 更大)
```

### 显存规划

多模态推理需要额外考虑：

1. **视觉编码器的显存**：通常 1-3 GB
2. **图像 token 的 KV Cache**：每张图 576-2048 个 token
3. **图像预处理的临时显存**：图像解码和变换

```python
# 估算图文请求的 KV Cache
text_tokens = 100
image_tokens = 576  # LLaVA 默认

total_tokens = text_tokens + image_tokens
# KV Cache 按 total_tokens 计算
```

### 优化建议

```bash
# 限制图像分辨率（减少图像 token 数）
vllm serve model --max-num-seqs 32  # 多模态并发数通常要设小一些

# 如果显存紧张
vllm serve model --max-model-len 2048  # 限制总序列长度
```

---

## 14.6 支持的多模态模型

| 模型 | 输入模态 | 图像 token 数 | 说明 |
|------|---------|-------------|------|
| LLaVA 1.5 | 图+文 | ~576 | 经典 VLM |
| Qwen2-VL | 图+文+视频 | 动态 | 支持任意分辨率 |
| InternVL 2 | 图+文 | 动态 | 高性能中文 VLM |
| Llama 3.2 Vision | 图+文 | ~1024 | Meta 官方 |
| PaliGemma | 图+文 | 256 | Google 轻量 VLM |
| Phi-3 Vision | 图+文 | ~768 | Microsoft |

---

## 本章小结

| 概念 | 要点 |
|------|------|
| VLM 架构 | 视觉编码器 + 投影层 + 语言模型 |
| 图像 token | 一张图 ~576-2048 个 token |
| API 兼容 | 使用 OpenAI Vision API 格式 |
| 显存影响 | 图像 token 增加 KV Cache 和 prefill 开销 |
| 并发限制 | 多模态请求显存更大，并发数需适当调低 |

---

## 动手实验

### 实验 1：图文推理

加载一个 VLM 模型，对不同类型的图片（照片、图表、截图）进行推理。

### 实验 2：纯文本 vs 图文性能对比

测量相同模型在纯文本和图文请求下的 TTFT 和吞吐差异。

---

## 练习题

### 基础题

1. VLM 的三个主要组成部分是什么？
2. 为什么多模态推理的 prefill 阶段更慢？
3. 图像 token 数对 KV Cache 有什么影响？

### 思考题

4. 如果你的应用需要处理高分辨率图片（4K），对推理系统有什么挑战？
5. 多图输入时，显存压力如何变化？如何优化？
