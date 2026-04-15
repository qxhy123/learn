# 第12章：量化推理

> 量化是用精度换速度和显存的艺术。选对量化方案，可以在几乎不损失质量的前提下，让同一张 GPU 多服务 2-4 倍的请求。

---

## 学习目标

学完本章，你将能够：

1. 理解量化的基本原理和它对推理的影响
2. 掌握 vLLM 支持的量化方案（AWQ、GPTQ、FP8 等）及其区别
3. 加载和使用量化模型进行推理
4. 评估量化对显存、速度和生成质量的影响
5. 根据场景选择合适的量化方案

---

## 12.1 量化基础

### 什么是量化？

量化（Quantization）是将模型权重从高精度数据类型转换为低精度数据类型的过程。

```
FP16 (16 bit): ████████████████  → 基线
INT8  (8 bit): ████████          → 显存减半
INT4  (4 bit): ████              → 显存减至 1/4
```

### 量化为什么对推理有效？

回顾第 1 章的分析：decode 阶段是**内存带宽密集**的。量化通过减少权重大小：

1. **减少显存占用**：同样的 GPU 可以加载更大模型或服务更多请求
2. **减少数据传输量**：权重从显存到计算单元的传输更快
3. **潜在的计算加速**：INT8/INT4 运算在某些硬件上有专用指令

### 量化的代价

- **精度损失**：低精度表示引入舍入误差
- **某些任务质量下降**：复杂推理、数学计算可能受影响
- **量化过程需要校准数据**：部分方案需要少量数据做校准

---

## 12.2 vLLM 支持的量化方案

### 方案对比

| 方案 | 精度 | 量化方式 | 速度 | 质量 | 推荐场景 |
|------|------|---------|------|------|---------|
| AWQ | W4A16 | 权重 4bit，激活 FP16 | 快 | 好 | 通用推荐 |
| GPTQ | W4A16 | 权重 4bit，激活 FP16 | 快 | 好 | 社区模型多 |
| FP8 | W8A8 | 权重和激活 FP8 | 最快 | 最好 | H100/Ada GPU |
| SqueezeLLM | W4 | 非均匀量化 | 中等 | 较好 | 特殊需求 |
| GGUF | 多种 | llama.cpp 格式 | 中等 | 取决于精度 | 社区共享 |
| BitsAndBytes | W4/W8 | 动态量化 | 较慢 | 好 | 灵活性需求 |

**W4A16** = 权重 4bit，激活值（computation）使用 FP16

### AWQ（Activation-aware Weight Quantization）

**原理**：不是均匀量化所有权重，而是根据激活值的分布，保护对输出影响大的权重通道。

```python
# 加载 AWQ 量化模型
from vllm import LLM, SamplingParams

llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-AWQ",
    quantization="awq",
)

# 或使用 Qwen 的 AWQ 模型
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    quantization="awq",
)
```

**优势**：质量好、速度快、社区模型多
**适用**：大多数场景的首选

### GPTQ

**原理**：基于 OBQ（Optimal Brain Quantization），逐列量化权重矩阵，最小化量化误差。

```python
llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-GPTQ",
    quantization="gptq",
)
```

**优势**：社区生态成熟，模型资源丰富
**注意**：有不同的 GPTQ 变体（如 GPTQ-Marlin），性能不同

### FP8

**原理**：使用 8 位浮点数（FP8）表示权重和/或激活值。

```python
# FP8 量化（需要 H100/Ada Lovelace 或更新的 GPU）
llm = LLM(
    model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
    quantization="fp8",
)
```

**优势**：精度损失最小、速度最快（在支持的硬件上）
**限制**：需要 FP8 硬件支持（Compute Capability >= 8.9）

---

## 12.3 显存节省分析

### 模型权重显存对比

以 Llama-3.1-8B 为例：

| 精度 | 每参数字节数 | 模型权重大小 | 节省 |
|------|------------|------------|------|
| FP32 | 4 | ~32 GB | 基线 |
| FP16/BF16 | 2 | ~16 GB | 50% |
| FP8 | 1 | ~8 GB | 75% |
| INT4 (AWQ/GPTQ) | 0.5 | ~4 GB | 87.5% |

### 对并发能力的影响

```python
# 以 A100 80GB 为例
# FP16: 模型 16GB + KV Cache 56GB → ~56 并发
# INT4: 模型  4GB + KV Cache 68GB → ~68 并发 → 多出 21%

# 对于更大的模型效果更明显
# 70B FP16: 需要 2 张 A100 → 模型 140GB
# 70B INT4: 只需 1 张 A100 → 模型  35GB + KV Cache 37GB
```

### KV Cache 量化

vLLM 还支持 KV Cache 本身的量化：

```python
# KV Cache FP8 量化
llm = LLM(
    model="...",
    kv_cache_dtype="fp8",  # 默认是 auto（跟随模型精度）
)
```

KV Cache FP8 可以进一步减少 KV Cache 显存占用 50%，从而大幅提升并发能力。

### KV Cache 量化在 V1 源码中的落点

KV Cache 量化不只是"把 KV 存成 FP8"，它还影响注意力后端的行为。在 `v1/attention/backends/flash_attn.py` 中：

- `FlashAttentionBackend` 会根据 `kv_cache_dtype` 判断是否启用量化 KV cache
- 使用 `is_quantized_kv_cache()` 工具函数检测
- 量化 KV cache 需要额外的 `k_scale` / `v_scale` 参数传入 `write_to_paged_cache`
- 部分后端（如 FlashInfer）对 FP8 KV cache 有专门的优化路径

这也是为什么 KV cache FP8 在 H100 上效果最好——不仅存储减半，还能利用硬件的原生 FP8 支持减少 dequantize 开销。

---

## 12.4 质量评估

### 量化质量损失

不同任务对量化的敏感度不同：

| 任务 | INT4 质量影响 | FP8 质量影响 |
|------|-------------|-------------|
| 通用对话 | 几乎无 | 无 |
| 文本摘要 | 轻微 | 无 |
| 代码生成 | 轻微 | 无 |
| 数学推理 | 中等 | 轻微 |
| 复杂逻辑 | 中等 | 轻微 |

### 简单评估方法

```python
from vllm import LLM, SamplingParams

# 对比 FP16 和 AWQ 的输出
prompts = [
    "计算 123 × 456 的结果。",
    "解释相对论的基本概念。",
    "用 Python 实现快速排序。",
]

params = SamplingParams(temperature=0, max_tokens=300)

# FP16 推理
llm_fp16 = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="float16")
outputs_fp16 = llm_fp16.generate(prompts, params)

# AWQ 推理
llm_awq = LLM(model="Qwen/Qwen2.5-7B-Instruct-AWQ", quantization="awq")
outputs_awq = llm_awq.generate(prompts, params)

# 对比
for i, prompt in enumerate(prompts):
    print(f"\nPrompt: {prompt}")
    print(f"FP16: {outputs_fp16[i].outputs[0].text[:200]}")
    print(f"AWQ:  {outputs_awq[i].outputs[0].text[:200]}")
```

---

## 12.5 量化方案选择指南

### 决策树

```
你的 GPU 支持 FP8 吗？(H100/L40S/RTX 4090)
├── 是 → 使用 FP8（质量最好、速度最快）
└── 否 → 你需要最大的显存节省吗？
    ├── 是 → 使用 AWQ INT4
    └── 否 → 你需要中等显存节省且保证质量？
        └── 是 → 考虑 INT8 或混合精度方案
```

### 快速推荐

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 通用部署 | AWQ | 质量和显存的最佳平衡 |
| H100 部署 | FP8 | 硬件加速，质量最好 |
| 显存极限 | GPTQ INT4 + KV Cache FP8 | 最大化并发 |
| 质量优先 | FP16/BF16 | 无精度损失 |

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 量化目的 | 减少显存占用和数据传输量 |
| AWQ | W4A16，质量好，通用推荐 |
| GPTQ | W4A16，社区资源丰富 |
| FP8 | W8A8，质量最好但需要新硬件 |
| KV Cache 量化 | 独立于权重量化，进一步节省显存 |
| 质量影响 | 通用对话几乎无损，复杂推理略有影响 |

---

## 动手实验

### 实验 1：量化前后对比

加载同一模型的 FP16 和 AWQ 版本，对比显存占用、推理速度和生成质量。

### 实验 2：并发能力对比

分别用 FP16 和 AWQ 模型，测量在相同 GPU 上的最大并发数。

---

## 练习题

### 基础题

1. 量化为什么能加速 LLM 推理？
2. AWQ 和 GPTQ 的主要区别是什么？
3. FP8 量化需要什么硬件支持？

### 实践题

4. 加载一个 AWQ 量化模型，与 FP16 版本对比 10 个 prompt 的输出质量。

### 思考题

5. 为什么 INT4 量化对通用对话影响很小，但对数学推理影响较大？
6. 如果你需要在单张 24GB GPU 上部署 70B 模型，你会选择什么方案？
