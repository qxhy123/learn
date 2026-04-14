# 第1章：LLM 推理的挑战

> 训练是计算密集的，推理是内存密集的。理解这一差异，是理解所有 LLM 推理优化技术的起点。

---

## 学习目标

学完本章，你将能够：

1. 理解自回归生成的基本过程以及它为什么天然慢
2. 区分 LLM 推理中的 prefill 阶段和 decode 阶段
3. 理解 LLM 推理为什么是内存带宽瓶颈而非计算瓶颈
4. 掌握 TTFT、TPOT、吞吐等推理性能指标的定义
5. 建立对 LLM 推理系统设计空间的总体认识

---

## 1.1 自回归生成：LLM 推理的基本模式

### 什么是自回归生成？

当你向 ChatGPT 提问并看到它逐字输出回答时，背后发生的就是自回归生成（autoregressive generation）。

自回归生成的过程可以用一句话概括：

> 每次只生成一个 token，将这个 token 追加到已有序列中，然后再生成下一个 token，直到生成结束标记或达到长度上限。

用伪代码表示：

```python
def autoregressive_generate(model, prompt_tokens, max_new_tokens):
    tokens = prompt_tokens
    for _ in range(max_new_tokens):
        # 将整个序列送入模型，得到下一个 token 的概率分布
        logits = model.forward(tokens)
        next_token = sample(logits[-1])  # 只用最后一个位置的 logits
        tokens.append(next_token)
        if next_token == EOS:
            break
    return tokens
```

### 为什么自回归生成天然慢？

关键问题在于：**每生成一个 token，都需要一次完整的模型前向计算**。

对于一个 7B 参数的模型，生成 100 个 token 意味着：

- 模型的所有权重被从 GPU 显存读取了 100 次
- 每次读取约 14GB 数据（FP16 下 7B 参数）
- 但每次读取只产出 1 个 token

这就好比你每查一个字，都要把整本字典从书架上取下来、翻一遍、再放回去。

### 与训练的对比

训练时，一个 batch 中可能有数千个 token，每次前向计算可以同时计算所有位置的 loss，GPU 的计算单元被充分利用。

推理时（尤其是 decode 阶段），每次只处理 1 个新 token，GPU 的大量计算单元处于空闲状态，瓶颈变成了"把模型权重从显存读到计算单元"的带宽。

| 维度 | 训练 | 推理（decode） |
|------|------|----------------|
| 每次处理的 token 数 | 数百到数千 | 1 |
| 瓶颈 | 计算（compute-bound） | 内存带宽（memory-bound） |
| GPU 利用率 | 高 | 低（除非大 batch） |
| 优化方向 | 更多 FLOPS | 更高带宽、更大 batch |

---

## 1.2 推理的两个阶段：Prefill 与 Decode

一次完整的 LLM 推理过程可以分为两个明显不同的阶段。

### Prefill 阶段（预填充）

当用户输入一段 prompt 时，模型需要先处理这段输入，计算出所有 prompt token 对应的内部状态。

特点：

- **输入**：所有 prompt token 一次性送入模型
- **计算模式**：类似训练的前向传播，可以并行处理所有 token
- **瓶颈**：通常是计算密集（compute-bound），因为一次处理大量 token
- **输出**：第一个生成的 token + 所有 token 的 KV Cache

Prefill 阶段的耗时直接决定了用户看到第一个字的等待时间，即 **TTFT（Time To First Token）**。

### Decode 阶段（解码）

Prefill 完成后，模型进入逐 token 生成的循环。

特点：

- **输入**：每次只有 1 个新 token（加上之前缓存的 KV Cache）
- **计算模式**：极小的矩阵-向量乘法，计算量很小
- **瓶颈**：内存带宽密集（memory-bound），因为每次都要读取全部模型权重
- **输出**：1 个新 token

Decode 阶段每个 token 的生成时间即 **TPOT（Time Per Output Token）**。

### 两阶段的差异可视化

```
用户输入: "请解释什么是量子计算"  (假设 10 个 token)

=== Prefill 阶段 ===
[tok1, tok2, tok3, ..., tok10] → 模型一次性处理 → 第一个输出 token + KV Cache
                                                    ↓
                                              用户看到第一个字
                                              (TTFT = prefill 耗时)

=== Decode 阶段 ===
[新 tok1] + KV Cache → 模型处理 → 新 tok2    (TPOT)
[新 tok2] + KV Cache → 模型处理 → 新 tok3    (TPOT)
[新 tok3] + KV Cache → 模型处理 → 新 tok4    (TPOT)
...
[新 tokN] + KV Cache → 模型处理 → EOS        (生成结束)
```

### 为什么区分两个阶段很重要？

因为它们的优化策略完全不同：

| 优化目标 | Prefill 优化 | Decode 优化 |
|----------|-------------|-------------|
| 关键指标 | TTFT | TPOT |
| 计算特点 | 大矩阵乘法 | 小矩阵-向量乘 |
| 瓶颈类型 | 计算密集 | 内存带宽密集 |
| 优化手段 | Flash Attention、量化 | 批处理、KV Cache 优化 |

vLLM 的很多核心设计——包括 PagedAttention、连续批处理、前缀缓存——都是围绕这两个阶段的不同特性来设计的。

---

## 1.3 内存带宽：推理的真正瓶颈

### 为什么 GPU 算力用不满？

一张 A100 GPU 的峰值算力是 312 TFLOPS（FP16），但在 decode 阶段，实际利用率可能不到 5%。

原因很简单：

- 每次 decode 只处理 1 个 token
- 需要的计算量约为 2 × 模型参数数（对于 7B 模型约 14 GFLOP）
- 但需要从 HBM 读取的数据量约为模型参数量（7B × 2 bytes = 14 GB）

A100 的 HBM 带宽约为 2 TB/s，读取 14 GB 需要约 7ms。

而 14 GFLOP 的计算在 312 TFLOPS 算力下只需要 0.045ms。

这意味着计算单元在 99% 的时间里都在等待数据从显存送达。

### 算术强度：判断瓶颈的关键指标

**算术强度（arithmetic intensity）** = 计算量（FLOP） / 数据传输量（Bytes）

| 场景 | 计算量 | 数据量 | 算术强度 | 瓶颈 |
|------|--------|--------|----------|------|
| 训练（batch=1024） | 高 | 相对低 | 高 | 计算 |
| Prefill（长 prompt） | 中高 | 中 | 中 | 取决于长度 |
| Decode（batch=1） | 低 | 高 | ~1 | 内存带宽 |
| Decode（batch=32） | 中 | 高 | ~32 | 接近平衡 |

关键洞察：**增大 batch size 是提升 decode 阶段 GPU 利用率的最直接方法**。这也是 vLLM 的连续批处理如此重要的原因。

### Roofline 模型直觉

Roofline 模型用一张图展示了计算密集和内存密集的分界线：

```
性能 (FLOPS)
    │
    │         ╱‾‾‾‾‾‾‾‾‾‾‾‾‾  计算 Roofline (峰值算力)
    │        ╱
    │       ╱
    │      ╱   ← 内存带宽 Roofline
    │     ╱
    │    ╱
    │   ╱  ← Decode (batch=1) 在这里
    │  ╱
    │ ╱
    │╱
    └────────────────────────── 算术强度 (FLOP/Byte)
         ↑                  ↑
     Decode (batch=1)   Prefill / 训练
```

当算术强度低于拐点时，系统受内存带宽限制；高于拐点时，受计算能力限制。Decode 阶段几乎总是落在左侧。

---

## 1.4 推理性能指标

理解推理性能需要关注以下核心指标：

### 延迟指标

**TTFT（Time To First Token，首 token 延迟）**

- 定义：从发送请求到收到第一个生成 token 的时间
- 影响因素：prompt 长度、prefill 计算量、排队等待时间
- 用户感知：用户点击"发送"后等多久才开始看到输出

**TPOT（Time Per Output Token，token 间延迟）**

- 定义：生成相邻两个 token 之间的时间间隔
- 影响因素：模型大小、batch size、KV Cache 大小
- 用户感知：输出文字的流畅程度

**端到端延迟（End-to-End Latency）**

- 定义：从发送请求到收到完整回答的总时间
- 计算：TTFT + TPOT × 输出 token 数

### 吞吐指标

**请求吞吐（Request Throughput）**

- 定义：单位时间内完成的请求数（requests/second）
- 关注点：整体系统处理能力

**Token 吞吐（Token Throughput）**

- 定义：单位时间内生成的 token 数（tokens/second）
- 分为输入吞吐和输出吞吐

### 延迟与吞吐的权衡

这是 LLM 推理系统设计中最核心的权衡：

- **更大的 batch** → 更高吞吐，但单个请求延迟可能增加
- **更小的 batch** → 更低延迟，但 GPU 利用率低、吞吐差

```
吞吐
  │      ╱‾‾‾‾‾‾‾‾‾‾ 吞吐上限（显存/计算限制）
  │     ╱
  │    ╱
  │   ╱
  │  ╱
  │ ╱
  │╱
  └──────────────── 并发请求数
  
延迟
  │                 ╱
  │               ╱
  │             ╱
  │           ╱
  │  ‾‾‾‾‾‾‾╱   ← 延迟开始增加（排队效应）
  │
  └──────────────── 并发请求数
```

好的推理系统会在吞吐和延迟之间找到一个合适的工作点，而不是一味追求某一端。

---

## 1.5 朴素推理的问题

如果直接用 Hugging Face Transformers 的 `model.generate()` 来做推理服务，会遇到以下问题：

### 问题 1：显存浪费严重

朴素实现中，KV Cache 按最大长度预分配，导致大量显存浪费。

```python
# 假设 max_seq_len=2048，但实际请求只有 100 token
# 朴素实现仍然为 2048 token 分配 KV Cache
# 浪费比例：(2048 - 100) / 2048 = 95%
```

### 问题 2：无法高效批处理

不同请求的输入长度和输出长度不同，朴素的静态批处理要么等最长请求完成（浪费算力），要么预先 padding（浪费显存）。

```
静态批处理的浪费：

请求 A: [████████████████████░░░░░░░░░░]  已完成，在等待
请求 B: [██████████████████████████████]  仍在生成
请求 C: [████████████████░░░░░░░░░░░░░░]  已完成，在等待
                                           ↑ B 完成前，A 和 C 的 GPU 资源被浪费
```

### 问题 3：请求间无法共享内存

当多个请求使用相同的系统 prompt（如 "你是一个有帮助的助手..."），每个请求都独立计算并存储这部分 KV Cache，无法复用。

### 问题 4：缺乏生产级特性

- 没有请求排队和调度
- 没有优雅的超时和取消
- 没有流式输出（或实现不完善）
- 没有并发安全保障
- 没有监控指标暴露

这些问题正是 vLLM 等推理引擎被创造出来要解决的。

---

## 1.6 推理优化技术全景

在深入 vLLM 之前，先建立对推理优化技术的全局认识：

### 模型级优化

| 技术 | 原理 | 效果 |
|------|------|------|
| 量化（Quantization） | 降低权重精度（FP16→INT8/INT4） | 减少显存、提高带宽利用 |
| 蒸馏（Distillation） | 用大模型教小模型 | 更小模型、更快推理 |
| 剪枝（Pruning） | 去掉不重要的参数 | 减少计算量 |

### 算法级优化

| 技术 | 原理 | 效果 |
|------|------|------|
| KV Cache | 缓存已计算的 K、V 向量 | 避免重复计算 |
| Flash Attention | IO 感知的注意力算法 | 减少显存访问 |
| 投机解码 | 用小模型预测、大模型验证 | 降低延迟 |

### 系统级优化

| 技术 | 原理 | 效果 |
|------|------|------|
| 连续批处理 | 动态添加/移除请求 | 提高 GPU 利用率 |
| PagedAttention | 分页管理 KV Cache | 减少显存碎片 |
| 前缀缓存 | 复用共享前缀的 KV Cache | 减少重复计算 |
| 张量并行 | 跨 GPU 切分模型 | 支持更大模型 |

### 硬件级优化

| 技术 | 原理 | 效果 |
|------|------|------|
| Tensor Core | 矩阵乘加专用硬件 | 加速计算 |
| HBM3 | 更高带宽显存 | 缓解内存瓶颈 |
| NVLink | GPU 间高速互联 | 加速多 GPU 通信 |

vLLM 主要在**系统级优化**领域做出了核心贡献，同时集成了模型级和算法级的优化能力。

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 自回归生成 | 每次只生成一个 token，需要 N 次前向计算生成 N 个 token |
| Prefill 阶段 | 处理 prompt，计算密集，决定 TTFT |
| Decode 阶段 | 逐 token 生成，内存带宽密集，决定 TPOT |
| 内存带宽瓶颈 | Decode 阶段 GPU 计算单元大量空闲，等待数据传输 |
| 提升吞吐的关键 | 增大 batch size，让更多请求共享一次权重读取 |
| 朴素推理的问题 | 显存浪费、无法高效批处理、无法共享内存、缺乏生产特性 |

---

## 动手实验

### 实验 1：测量朴素推理的性能

使用 Hugging Face Transformers 测量单请求推理的基本性能：

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

prompt = "请用三句话解释量子计算的基本原理。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 测量 TTFT 和总延迟
start = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs, max_new_tokens=200, do_sample=False
    )
total_time = time.time() - start

generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
print(f"生成 {generated_tokens} 个 token")
print(f"总延迟: {total_time:.2f}s")
print(f"平均 TPOT: {total_time / generated_tokens * 1000:.1f}ms")
print(f"Token 吞吐: {generated_tokens / total_time:.1f} tokens/s")
```

### 实验 2：观察 batch size 对吞吐的影响

```python
# 对比 batch_size=1 和 batch_size=8 的吞吐差异
prompts_1 = ["Hello, how are you?"]
prompts_8 = ["Hello, how are you?"] * 8

# 分别测量，观察：
# 1. batch=8 的总时间是否是 batch=1 的 8 倍？
# 2. token 吞吐提升了多少？
# 3. 单请求延迟有何变化？
```

### 实验 3：估算显存占用

```python
# 估算 7B 模型的显存占用
param_count = 7e9
bytes_per_param_fp16 = 2

model_weight_gb = param_count * bytes_per_param_fp16 / 1e9
print(f"模型权重 (FP16): {model_weight_gb:.1f} GB")

# 估算 KV Cache 显存
# 假设：32 层，32 头，128 维度/头，序列长度 2048，batch=1
num_layers = 32
num_heads = 32
head_dim = 128
seq_len = 2048
batch_size = 1

kv_cache_bytes = (
    2  # K 和 V
    * num_layers
    * num_heads
    * head_dim
    * seq_len
    * batch_size
    * 2  # FP16
)
kv_cache_gb = kv_cache_bytes / 1e9
print(f"KV Cache (单请求, seq={seq_len}): {kv_cache_gb:.2f} GB")
print(f"KV Cache (32 并发): {kv_cache_gb * 32:.2f} GB")
```

---

## 练习题

### 基础题

1. 为什么自回归生成每次只能产出 1 个 token？能否一次性生成所有 token？
2. Prefill 阶段和 Decode 阶段分别是计算密集还是内存带宽密集？为什么？
3. 什么是 TTFT 和 TPOT？它们分别受哪些因素影响？

### 实践题

4. 计算一个 70B 参数模型在 FP16 下的权重显存占用。如果使用 INT4 量化，能节省多少显存？
5. 对于一个 32 层、32 头、head_dim=128 的模型，计算一个请求在序列长度分别为 512、2048、8192 时的 KV Cache 显存占用。

### 思考题

6. 为什么说"增大 batch size 是提升 decode 吞吐的最直接方法"？增大 batch size 的上限在哪里？
7. 如果你的应用场景是实时聊天（低延迟优先）而非离线文本生成（高吞吐优先），你会如何调整推理系统的配置？
