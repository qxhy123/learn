# 第3章：vLLM 概览与定位

> vLLM 的核心洞察很简单：把操作系统管理物理内存的方式，用到 GPU 显存上管理 KV Cache。这个类比不仅优雅，而且真的有效。

---

## 学习目标

学完本章，你将能够：

1. 理解 vLLM 的设计目标和它要解决的核心问题
2. 掌握 vLLM 的三大核心技术：PagedAttention、连续批处理、高效调度
3. 了解 vLLM 与其他推理引擎（TGI、TensorRT-LLM、SGLang）的定位差异
4. 建立对 vLLM 整体架构的初步认识
5. 运行第一个 vLLM 推理，直观感受与朴素推理的性能差异

---

## 3.1 vLLM 是什么？

vLLM 是一个**高吞吐、低延迟的 LLM 推理和服务引擎**，由加州大学伯克利分校的 Kwon 等人于 2023 年提出。

它的名字中的 "v" 代表 "virtual"——虚拟内存，因为它的核心创新 PagedAttention 借鉴了操作系统虚拟内存管理的思想。

### vLLM 的核心价值主张

用一句话概括：**vLLM 通过高效管理 KV Cache 显存，让同一张 GPU 能同时服务更多请求，从而大幅提升推理吞吐。**

具体来说，vLLM 在以下几个维度提供价值：

| 维度 | vLLM 的贡献 |
|------|------------|
| 显存效率 | PagedAttention 消除 KV Cache 碎片，显存浪费从 60-80% 降至接近 0 |
| 吞吐量 | 连续批处理 + 更多并发 = 2-4× 吞吐提升 |
| 易用性 | OpenAI 兼容 API，一行命令启动服务 |
| 模型生态 | 支持 HuggingFace 上绝大多数主流模型 |
| 生产就绪 | 分布式推理、量化、监控、流式输出 |

### vLLM 的发展时间线

```
2023.06  论文 "Efficient Memory Management for Large Language Model Serving
          with PagedAttention" 发表
2023.06  vLLM v0.1 开源发布
2023-24  快速迭代：量化支持、多模态、投机解码、前缀缓存...
2024     成为最流行的开源 LLM 推理引擎之一
2025     持续发展：V1 引擎重构、Disaggregated Prefill、FlashInfer 集成...
```

---

## 3.2 三大核心技术

### 核心技术 1：PagedAttention

**问题**：朴素的 KV Cache 管理导致大量显存浪费（预分配 + 碎片）。

**解决方案**：借鉴操作系统的虚拟内存分页机制：

- 将 KV Cache 分成固定大小的**块（block）**
- 每个块存储固定数量 token 的 KV 向量
- 块可以在物理显存中不连续分布
- 通过**块表（block table）**维护逻辑块到物理块的映射

```
操作系统虚拟内存              vLLM PagedAttention
─────────────────          ─────────────────────
虚拟页面                    逻辑 KV 块
物理页帧                    物理 KV 块
页表                        块表
按需分配页面                  按需分配 KV 块
消除内存碎片                  消除显存碎片
```

效果：KV Cache 的显存浪费从 60-80% 降至不到 4%（仅最后一个块可能有少量内部碎片）。

详细原理将在第 7 章展开。

### 核心技术 2：连续批处理（Continuous Batching）

**问题**：静态批处理中，一个请求完成后必须等其他请求都完成，GPU 资源被浪费。

**解决方案**：在每个 iteration 级别动态添加和移除请求。

```
静态批处理：                  连续批处理：
                            
[A████████████]             [A████████████]
[B████████]                 [B████████][D████]
[C██████████████████]       [C██████████████████]
                            [         E████████]
↑ A,B完成后仍占用资源       ↑ A,B完成后立刻让位给新请求
```

效果：GPU 利用率显著提升，吞吐量提升 2-4×。

详细原理将在第 8 章展开。

### 核心技术 3：统一调度

**问题**：高并发场景下，如何让长 prompt、短 prompt、decode 请求、prefix cache 命中请求和 speculative decoding 请求共享同一套资源，而不是互相把系统拖死？

**当前仓库里的解决方案**：vLLM V1 使用统一 token 预算调度，而不是把 prefill/decode 拆成两套几乎独立的流程。

- **统一进度抽象**：用 `num_computed_tokens` 和 `num_tokens_with_spec` 表示“已经算到哪里”和“应该算到哪里”
- **统一预算模型**：所有请求竞争同一套 `max_num_batched_tokens` / `max_num_seqs`
- **运行中请求优先**：先尽量推进已经持有 KV Cache 的请求，减少浪费
- **抢占以 recompute 为主**：当前 V1 主路径是 `PREEMPTED + 重新调度/重算`，而不是旧资料常写的 CPU swap 队列

详细原理将在第 9 章展开。

---

## 3.3 vLLM 架构总览

结合当前仓库，vLLM 更适合被理解成下面这几层：

```text
┌────────────────────────────────────────────────────┐
│                    入口层                          │
│  entrypoints/llm.py  /  openai/api_server.py      │
├────────────────────────────────────────────────────┤
│                  外层引擎层                         │
│  v1/engine/llm_engine.py                          │
│  InputProcessor / EngineCoreClient / OutputProcessor │
├────────────────────────────────────────────────────┤
│                  核心内循环层                       │
│  v1/engine/core.py                                │
│  Scheduler / KVCacheManager / StructuredOutputManager │
├────────────────────────────────────────────────────┤
│                  执行与设备层                        │
│  v1/executor/  →  v1/worker/                      │
│  GPUWorker / GPUModelRunner                        │
├────────────────────────────────────────────────────┤
│                模型与注意力后端层                    │
│  model_executor/models/                            │
│  v1/attention/backends/ / ops/paged_attn.py        │
└────────────────────────────────────────────────────┘
```

### 各层职责

**入口层**：接收 Python 调用或 HTTP 请求，把协议层输入转换成引擎可消费的请求。

**外层引擎层**：负责把输入整理成 `EngineCoreRequest`，并把底层输出重新拼成用户能直接消费的 `RequestOutput`。

**核心内循环层**：真正做调度、KV 块管理、结构化输出编排和执行批次组织。

**执行与设备层**：把 scheduler 给出的 batch 送到具体设备上，由 worker/model runner 完成前向执行。

**模型与注意力后端层**：实现具体模型结构、PagedAttention、FlashAttention/FlashInfer/Triton 等后端。

### 当前仓库里的源码对照

| 主题 | 关键文件 |
|------|----------|
| 离线入口 | `vllm/vllm/entrypoints/llm.py` |
| 在线入口 | `vllm/vllm/entrypoints/openai/api_server.py` |
| 外层引擎 | `vllm/vllm/v1/engine/llm_engine.py` |
| 核心内循环 | `vllm/vllm/v1/engine/core.py` |
| 调度器 | `vllm/vllm/v1/core/sched/scheduler.py` |
| KV 管理 | `vllm/vllm/v1/core/kv_cache_manager.py` |
| 设备执行 | `vllm/vllm/v1/worker/gpu_worker.py` / `vllm/vllm/v1/worker/gpu/model_runner.py` |
| 注意力后端 | `vllm/vllm/v1/attention/backends/` |

### 一次请求的生命周期

```text
1. 用户请求进入入口层
2. InputProcessor 生成 EngineCoreRequest
3. EngineCore 把请求交给 Scheduler
4. Scheduler 通过 KVCacheManager 查询前缀命中并分配 block
5. Executor / Worker / GPUModelRunner 执行模型前向
6. OutputProcessor 汇总和流式返回输出
7. 请求完成后，KV block 被释放或保留在 prefix cache 中复用
```

---

## 3.4 与其他推理引擎的对比

### vLLM vs Hugging Face TGI

| 维度 | vLLM | TGI |
|------|------|-----|
| 核心技术 | PagedAttention | Continuous Batching |
| 显存管理 | 分页管理，碎片极低 | 预分配，碎片较高 |
| 吞吐 | 更高（得益于更好的显存效率） | 中等 |
| 模型支持 | 广泛 | 广泛 |
| 易用性 | pip install + 一行命令 | Docker 为主 |
| 开发语言 | Python + CUDA | Rust + Python |
| 社区 | 非常活跃 | 活跃 |

### vLLM vs TensorRT-LLM

| 维度 | vLLM | TensorRT-LLM |
|------|------|--------------|
| 核心优势 | 灵活性、易用性、显存效率 | 极致性能优化 |
| 性能 | 高 | 最高（NVIDIA 官方优化） |
| 模型支持 | 广泛，社区驱动 | 需要手动转换模型 |
| 易用性 | 非常容易 | 相对复杂 |
| 硬件支持 | CUDA 为主 | 仅 NVIDIA |
| 适用场景 | 通用、快速迭代 | 极致性能需求 |

### vLLM vs SGLang

| 维度 | vLLM | SGLang |
|------|------|--------|
| 核心优势 | 成熟的推理引擎 | RadixAttention + 前端编程 |
| 显存管理 | PagedAttention | RadixAttention（基数树管理） |
| 前缀缓存 | 支持（hash 匹配） | 原生集成（Radix Tree） |
| 编程模型 | API 调用 | 结构化生成DSL |
| 成熟度 | 更成熟 | 快速发展中 |
| 社区 | 更大 | 较小但活跃 |

### 如何选择？

- **通用场景、快速上手**：vLLM
- **极致性能、NVIDIA 平台**：TensorRT-LLM
- **复杂生成逻辑、前缀复用密集**：SGLang
- **Hugging Face 生态集成**：TGI

---

## 3.5 第一次运行 vLLM

### 离线推理（最简单的方式）

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

# 生成
prompts = [
    "请用简单的语言解释什么是机器学习。",
    "写一首关于春天的五言绝句。",
    "Python 和 Java 的主要区别是什么？",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Output: {generated}")
    print("---")
```

### 在线服务

```bash
# 启动 OpenAI 兼容的 API 服务器
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000

# 另一个终端，用 curl 调用
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### 与 Hugging Face 基线对比

```python
import time
from vllm import LLM, SamplingParams

# --- vLLM ---
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
prompts = ["Explain quantum computing in simple terms."] * 32
params = SamplingParams(max_tokens=200, temperature=0)

start = time.time()
vllm_outputs = llm.generate(prompts, params)
vllm_time = time.time() - start

total_tokens = sum(len(o.outputs[0].token_ids) for o in vllm_outputs)
print(f"vLLM: {total_tokens} tokens in {vllm_time:.1f}s")
print(f"  吞吐: {total_tokens / vllm_time:.0f} tokens/s")
```

你应该能观察到 vLLM 在批量推理场景下的吞吐优势明显。

---

## 本章小结

| 概念 | 要点 |
|------|------|
| vLLM 定位 | 高吞吐、低延迟的 LLM 推理和服务引擎 |
| PagedAttention | 分页管理 KV Cache，消除显存碎片和浪费 |
| 连续批处理 | iteration 级别动态调度请求，提升 GPU 利用率 |
| 架构分层 | API 层 → 引擎层（调度+块管理）→ 执行层（Worker+Model）→ 硬件层 |
| vs 其他引擎 | vLLM 在易用性和显存效率间取得了出色平衡 |

---

## 动手实验

### 实验 1：对比 vLLM 与 HF Transformers 的吞吐

分别用 vLLM 和 HF Transformers 对同一组 prompt 做批量生成（建议 32-128 条），记录：
- 总生成时间
- 总 token 数
- token 吞吐（tokens/s）
- GPU 显存占用（`nvidia-smi`）

### 实验 2：感受连续批处理

启动 vLLM 服务器后，用多线程/多进程同时发送 50 个请求，每个请求的输出长度不同（通过不同 prompt 控制）。观察：
- 所有请求是否大致同时开始返回？
- 短请求是否比长请求更早完成？
- 整体吞吐与顺序执行相比如何？

### 实验 3：探索 vLLM 启动参数

用不同参数启动 vLLM 并观察效果：

```bash
# 限制 GPU 显存使用
vllm serve model --gpu-memory-utilization 0.5

# 限制最大序列长度
vllm serve model --max-model-len 2048

# 调整最大并发数
vllm serve model --max-num-seqs 64
```

---

## 练习题

### 基础题

1. vLLM 的名字中 "v" 代表什么？它与 vLLM 的核心技术有什么关系？
2. 连续批处理相比静态批处理的核心优势是什么？
3. 列出 vLLM 架构的四个主要层次及其职责。

### 实践题

4. 安装 vLLM 并运行本章的离线推理示例。记录加载时间和生成速度。
5. 启动 vLLM OpenAI 兼容服务器，并用 Python 的 `openai` 库成功调用。

### 思考题

6. 在什么场景下 TensorRT-LLM 可能比 vLLM 更合适？在什么场景下 vLLM 的优势更明显？
7. 如果你的应用是一个在线聊天机器人（并发 1000 用户），你最关心 vLLM 的哪些特性？为什么？
