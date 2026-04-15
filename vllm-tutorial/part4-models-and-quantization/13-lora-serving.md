# 第13章：LoRA 多适配器服务

> LoRA 让你用一个基座模型同时服务多个不同的微调版本。对推理系统来说，这意味着一张 GPU 可以"变身"成多个专用模型。

---

## 学习目标

学完本章，你将能够：

1. 理解 LoRA 在推理中的工作方式
2. 使用 vLLM 同时服务多个 LoRA 适配器
3. 掌握 LoRA 的动态加载和卸载
4. 分析 LoRA 服务的显存开销
5. 理解多 LoRA 服务的性能考量

---

## 13.1 LoRA 推理原理

### 回顾 LoRA

LoRA（Low-Rank Adaptation）通过低秩矩阵分解，在不修改原始模型权重的情况下添加可训练参数：

```
原始前向: y = W × x
LoRA 前向: y = W × x + (B × A) × x

其中:
  W: [d_out, d_in]     原始权重矩阵 (冻结)
  A: [r, d_in]          LoRA 下投影矩阵
  B: [d_out, r]         LoRA 上投影矩阵
  r: rank (通常 8-64)    远小于 d_in 和 d_out
```

### 为什么 LoRA 适合推理服务？

```
传统多模型部署:                    LoRA 多适配器部署:

GPU 1: [模型A 全部权重 16GB]       GPU 1: [基座模型 16GB]
GPU 2: [模型B 全部权重 16GB]              + [LoRA-A 0.1GB]
GPU 3: [模型C 全部权重 16GB]              + [LoRA-B 0.1GB]
                                          + [LoRA-C 0.1GB]
总计: 48 GB × 3 GPU                总计: 16.3 GB × 1 GPU
```

关键优势：
- LoRA 权重很小（通常是基座模型的 0.1%-1%）
- 多个 LoRA 共享同一个基座模型
- 可以动态加载/卸载，无需重启

---

## 13.2 vLLM 中的 LoRA 使用

### 启动时启用 LoRA

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 启用 LoRA 支持
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_loras=4,          # 同时加载的最大 LoRA 数
    max_lora_rank=64,     # 支持的最大 LoRA rank
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

# 创建 LoRA 请求
lora_request = LoRARequest(
    lora_name="my-adapter",
    lora_int_id=1,
    lora_local_path="/path/to/lora-adapter",
)

# 使用 LoRA 生成
outputs = llm.generate(
    ["Tell me a joke."],
    sampling_params,
    lora_request=lora_request,
)
```

### 同时服务多个 LoRA

```python
# 定义多个 LoRA
lora_chat = LoRARequest("chat-adapter", 1, "/path/to/chat-lora")
lora_code = LoRARequest("code-adapter", 2, "/path/to/code-lora")
lora_math = LoRARequest("math-adapter", 3, "/path/to/math-lora")

# 不同请求使用不同 LoRA
prompts_and_loras = [
    ("你好，今天天气怎么样？", lora_chat),
    ("def fibonacci(n):", lora_code),
    ("求解方程 2x + 3 = 7", lora_math),
]

for prompt, lora in prompts_and_loras:
    outputs = llm.generate([prompt], sampling_params, lora_request=lora)
    print(f"[{lora.lora_name}] {outputs[0].outputs[0].text[:100]}")
```

### 不使用 LoRA 的请求

```python
# 某些请求可以不带 LoRA，直接使用基座模型
outputs = llm.generate(
    ["Hello!"],
    sampling_params,
    # 不传 lora_request → 使用基座模型
)
```

---

## 13.3 API 服务器中的 LoRA

### 启动带 LoRA 的服务器

```bash
vllm serve meta-llama/Llama-3.1-8B \
    --enable-lora \
    --lora-modules chat-lora=/path/to/chat-lora \
                   code-lora=/path/to/code-lora \
    --max-loras 4 \
    --max-lora-rank 64
```

### 通过 API 指定 LoRA

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 使用 chat-lora
response = client.chat.completions.create(
    model="chat-lora",  # 使用 LoRA 的名字作为模型名
    messages=[{"role": "user", "content": "Hello!"}],
)

# 使用 code-lora
response = client.chat.completions.create(
    model="code-lora",
    messages=[{"role": "user", "content": "Write a Python function."}],
)

# 使用基座模型
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

---

## 13.4 显存开销分析

### LoRA 权重的显存

```python
# LoRA 权重大小估算
# 假设对所有 attention 层的 Q, K, V, O 投影做 LoRA

num_layers = 32
target_modules = 4  # Q, K, V, O
hidden_dim = 4096
rank = 16
dtype_bytes = 2  # FP16

lora_params = num_layers * target_modules * (hidden_dim * rank * 2)
lora_size_mb = lora_params * dtype_bytes / (1024**2)
print(f"LoRA 权重: {lora_size_mb:.1f} MB")  # ~128 MB (rank=16)
```

### 多 LoRA 的显存规划

```
基座模型:         16 GB
KV Cache:        56 GB
LoRA 适配器 ×4:   0.5 GB (每个 ~128 MB)
LoRA 计算缓冲:     0.5 GB
──────────────────────
总计:            ~73 GB (A100 80GB 可承受)
```

### max_loras vs 加载的 LoRA 数

- `max_loras`：同时**活跃**的 LoRA 数（影响显存预分配）
- 实际注册的 LoRA 可以更多，vLLM 按需加载和卸载

---

## 13.5 V1 源码中的 LoRA 集成

当前 V1 架构中，LoRA 的集成不是简单的"加载权重然后算一下增量"。它贯穿了调度器、worker 和 model runner 三层。

### 调度器层：LoRA 约束

在 `v1/core/sched/scheduler.py` 的 waiting 流程中，调度器会检查当前 batch 已有的 LoRA 种类：

```python
# 如果已调度的 LoRA 数量达到 max_loras，
# 且新请求使用的 LoRA 不在已调度集合中，则跳过该请求
if (self.lora_config
    and request.lora_request
    and len(scheduled_loras) == self.lora_config.max_loras
    and request.lora_request.lora_int_id not in scheduled_loras):
    step_skipped_waiting.prepend_request(request)
    continue
```

这意味着 `max_loras` 不只是限制"GPU 上加载几个 LoRA 权重"，还直接影响调度决策——同一 batch 中最多混合几种 LoRA。

### Worker 层：LoraState

`v1/worker/gpu/lora_utils.py` 中的 `LoraState` 管理 GPU 侧每个请求的 LoRA 映射：

```python
class LoraState:
    def __init__(self, max_num_reqs):
        self.lora_ids = np.zeros(max_num_reqs, dtype=np.int32)
        self.lora_requests: dict[str, LoRARequest] = {}

    def make_lora_inputs(self, req_ids, idx_mapping, num_scheduled_tokens):
        # 生成 prompt_lora_mapping 和 token_lora_mapping
        # 用于告诉 LoRA kernel 每个 token 属于哪个 LoRA
```

### Model Runner 层：LoRAModelRunnerMixin

`v1/worker/lora_model_runner_mixin.py` 中的 `LoRAModelRunnerMixin` 负责：

1. 用 `LRUCacheWorkerLoRAManager` 管理 LoRA 权重的加载/卸载（LRU 缓存策略）
2. 在每个前向计算前调用 `_set_active_loras()` 设置当前 batch 的 LoRA 映射
3. 使用 SGMV kernel（fused LoRA 计算）在 GPU 上高效执行多 LoRA batch

### 源码对照

| 层级 | 文件 | 职责 |
|------|------|------|
| 调度约束 | `v1/core/sched/scheduler.py` | `max_loras` 检查、LoRA 集合管理 |
| 请求对象 | `vllm/lora/request.py` | `LoRARequest` 定义 |
| Worker 状态 | `v1/worker/gpu/lora_utils.py` | `LoraState`，token→LoRA 映射 |
| Model Runner | `v1/worker/lora_model_runner_mixin.py` | LoRA 权重加载、SGMV kernel |

---

## 13.6 性能考量

### LoRA 对推理速度的影响

LoRA 引入额外的矩阵乘法，但因为 rank 很小，开销通常 < 5%：

```
原始:  y = W × x                    1次大矩阵乘
LoRA: y = W × x + B × (A × x)      1次大矩阵乘 + 2次小矩阵乘
```

当前 V1 使用 SGMV（Sparse Grouped Matrix-Vector）kernel 做融合 LoRA 计算，比 naive 实现更高效。

### 多 LoRA 批处理

当同一 batch 中有使用不同 LoRA 的请求时：

```
Batch 中的请求:
  请求 1: 基座模型
  请求 2: LoRA-A
  请求 3: LoRA-B
  请求 4: LoRA-A

处理方式:
  1. 基座模型的前向计算（所有请求）
  2. LoRA-A 的增量计算（请求 2, 4）
  3. LoRA-B 的增量计算（请求 3）
```

不同 LoRA 的请求越多，额外开销越大（因为需要分开计算 LoRA 增量）。

### LoRA 权重的 LRU 缓存

V1 使用 `LRUCacheWorkerLoRAManager` 管理 GPU 上的 LoRA 权重。当注册的 LoRA 数量超过 `max_loras` 时，最近最少使用的 LoRA 会被卸载，新请求的 LoRA 会被动态加载。这个过程对用户透明，但频繁切换 LoRA 会带来额外的加载开销。

---

## 本章小结

| 概念 | 要点 |
|------|------|
| LoRA 推理 | 基座模型 + 小型适配器权重 |
| 多适配器 | 一个 GPU 同时服务多个微调版本 |
| 显存开销 | LoRA 权重很小（~100MB/个），显存友好 |
| API 使用 | 通过 model 字段指定不同 LoRA |
| 性能影响 | 通常 < 5% 的额外延迟 |

---

## 动手实验

### 实验 1：LoRA 加载与推理

使用 Hugging Face 上的公开 LoRA 适配器，加载并进行推理。

### 实验 2：多 LoRA 并发

启动带多个 LoRA 的服务器，并发发送使用不同 LoRA 的请求，观察性能。

---

## 练习题

### 基础题

1. LoRA 推理时，基座模型的权重需要修改吗？
2. `max_loras` 参数的作用是什么？
3. LoRA rank=16 和 rank=64 对推理性能有什么影响？

### 思考题

4. 在什么场景下，LoRA 多适配器服务比部署多个独立模型更有优势？
5. 如果 batch 中所有请求都使用不同的 LoRA，对性能有什么影响？
