# 第5章：离线批量推理

> 离线推理是理解 vLLM 最直接的方式。没有服务器、没有 HTTP、没有并发——就是把一批 prompt 丢进去，拿到结果。

---

## 学习目标

学完本章，你将能够：

1. 使用 `LLM` 类和 `SamplingParams` 完成批量推理
2. 理解 vLLM 离线推理的核心 API 及其参数
3. 正确解析 `RequestOutput` 的结构并提取结果
4. 使用 chat 模板进行多轮对话推理
5. 掌握常用引擎参数对推理行为的影响

---

## 5.1 最简离线推理

### 核心 API

vLLM 离线推理只需要两个类：

```python
from vllm import LLM, SamplingParams
```

- **`LLM`**：模型引擎，负责加载模型和执行推理
- **`SamplingParams`**：采样参数，控制生成行为

### Hello World

```python
from vllm import LLM, SamplingParams

# 1. 加载模型
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

# 2. 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
)

# 3. 批量生成
prompts = [
    "什么是深度学习？用一句话回答。",
    "Python 的 GIL 是什么？",
    "为什么天空是蓝色的？",
]
outputs = llm.generate(prompts, sampling_params)

# 4. 输出结果
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("---")
```

---

## 5.2 SamplingParams 详解

`SamplingParams` 控制模型如何从概率分布中选择下一个 token。

### 常用参数

```python
params = SamplingParams(
    # --- 基础控制 ---
    max_tokens=256,        # 最大生成 token 数
    min_tokens=0,          # 最少生成 token 数（防止过早停止）
    stop=["\n\n", "END"],  # 停止词列表

    # --- 温度与采样 ---
    temperature=0.7,       # 控制随机性：0=确定性，>1=更随机
    top_p=0.9,             # nucleus sampling：只从累积概率前 90% 的 token 中采样
    top_k=50,              # 只从概率最高的 50 个 token 中采样

    # --- 重复控制 ---
    repetition_penalty=1.1,  # 重复惩罚，>1 减少重复
    frequency_penalty=0.0,   # 频率惩罚
    presence_penalty=0.0,    # 存在惩罚

    # --- 多输出 ---
    n=1,                   # 每个 prompt 生成几个结果
    best_of=1,             # 生成几个候选，返回最好的

    # --- 其他 ---
    seed=42,               # 随机种子，保证可复现
    skip_special_tokens=True,  # 跳过特殊 token
)
```

### 温度的直觉

```python
# temperature=0：完全确定性，每次结果相同（贪心解码）
params_greedy = SamplingParams(temperature=0, max_tokens=100)

# temperature=0.3：低随机性，适合事实性问答
params_low = SamplingParams(temperature=0.3, max_tokens=100)

# temperature=0.7：中等随机性，适合通用对话
params_mid = SamplingParams(temperature=0.7, max_tokens=100)

# temperature=1.2：高随机性，适合创意写作
params_high = SamplingParams(temperature=1.2, max_tokens=100)
```

### top_p 与 top_k 的关系

```python
# top_p=0.9：动态截断，保留概率总和达到 90% 的 token
# 效果：高置信时选择少，低置信时选择多

# top_k=50：固定截断，只保留概率最高的 50 个 token
# 效果：不管置信度如何，候选数量固定

# 两者可以同时使用，取交集
params = SamplingParams(top_p=0.9, top_k=50, temperature=0.7, max_tokens=100)
```

---

## 5.3 RequestOutput 结构解析

`llm.generate()` 返回一个 `RequestOutput` 列表，每个元素对应一个输入 prompt。

### 结构层次

```python
output = outputs[0]  # 一个 RequestOutput

# 基本信息
output.request_id     # 请求 ID
output.prompt         # 原始 prompt 文本
output.prompt_token_ids  # prompt 的 token ID 列表

# 生成结果（可能有多个，取决于 n 参数）
output.outputs        # List[CompletionOutput]

# 单个生成结果
completion = output.outputs[0]
completion.text              # 生成的文本
completion.token_ids         # 生成的 token ID 列表
completion.cumulative_logprob  # 累积对数概率
completion.finish_reason     # 结束原因: "stop" 或 "length"
```

### 完整解析示例

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    n=3,  # 每个 prompt 生成 3 个结果
)

outputs = llm.generate(["写一个关于 AI 的笑话。"], params)

output = outputs[0]
print(f"Request ID: {output.request_id}")
print(f"Prompt tokens: {len(output.prompt_token_ids)}")

for i, completion in enumerate(output.outputs):
    print(f"\n--- 结果 {i+1} ---")
    print(f"Text: {completion.text}")
    print(f"Tokens: {len(completion.token_ids)}")
    print(f"Finish reason: {completion.finish_reason}")
    print(f"Cumulative logprob: {completion.cumulative_logprob:.2f}")
```

---

## 5.4 Chat 模板推理

大多数 Instruct 模型需要特定的对话格式。vLLM 支持使用模型自带的 chat 模板。

### 使用 chat 方法

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
params = SamplingParams(temperature=0.7, max_tokens=256)

# 使用 chat 格式
messages_list = [
    # 第一个对话
    [
        {"role": "system", "content": "你是一个专业的 Python 导师。"},
        {"role": "user", "content": "解释一下装饰器。"},
    ],
    # 第二个对话
    [
        {"role": "system", "content": "你是一个数学老师。"},
        {"role": "user", "content": "什么是贝叶斯定理？"},
    ],
]

outputs = llm.chat(messages_list, params)

for output in outputs:
    print(output.outputs[0].text)
    print("---")
```

### 多轮对话

```python
# 模拟多轮对话
conversation = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "Python 的列表和元组有什么区别？"},
    {"role": "assistant", "content": "列表是可变的，元组是不可变的..."},
    {"role": "user", "content": "什么场景下应该用元组？"},
]

outputs = llm.chat([conversation], params)
print(outputs[0].outputs[0].text)
```

### 手动应用 chat 模板

如果你需要更细粒度的控制：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
]

# 手动应用模板
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(prompt)  # 查看实际发送给模型的文本

# 然后用 generate 方法
outputs = llm.generate([prompt], params)
```

---

## 5.5 LLM 引擎参数

`LLM` 类的构造参数控制引擎的行为。

### 常用引擎参数

```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",

    # --- 资源控制 ---
    dtype="auto",                  # 数据类型: auto/float16/bfloat16/float32
    gpu_memory_utilization=0.9,    # GPU 显存使用比例 (0-1)
    max_model_len=4096,            # 最大序列长度（过长会 OOM）

    # --- 并行 ---
    tensor_parallel_size=1,        # 张量并行 GPU 数
    pipeline_parallel_size=1,      # 流水线并行 GPU 数

    # --- 量化 ---
    quantization=None,             # 量化方案: awq/gptq/squeezellm/fp8

    # --- 模型信任 ---
    trust_remote_code=False,       # 是否信任模型仓库中的自定义代码

    # --- 调度 ---
    max_num_seqs=256,              # 最大并发序列数
    max_num_batched_tokens=None,   # 每次迭代最大 token 数

    # --- KV Cache ---
    block_size=16,                 # KV Cache 块大小（通常不需要修改）
    swap_space=4,                  # CPU swap 空间 (GB)

    # --- 前缀缓存 ---
    enable_prefix_caching=False,   # 是否启用前缀缓存
)
```

### 显存利用率调优

```python
# gpu_memory_utilization 控制 vLLM 可以使用多少 GPU 显存
# 剩余部分作为安全缓冲

# 保守设置（适合显存紧张时）
llm = LLM(model="...", gpu_memory_utilization=0.7)

# 激进设置（最大化吞吐）
llm = LLM(model="...", gpu_memory_utilization=0.95)

# 默认值
llm = LLM(model="...", gpu_memory_utilization=0.9)
```

### max_model_len 的影响

```python
# max_model_len 决定了最大序列长度（prompt + 生成）
# 更短 = 更少 KV Cache 显存 = 更多并发
# 更长 = 支持长文本 = 更少并发

# 对于短对话场景，可以限制 max_model_len 来提升并发
llm = LLM(model="...", max_model_len=2048)

# 对于长文本场景
llm = LLM(model="...", max_model_len=32768)
```

---

## 5.6 批量推理的性能观察

### 批大小对吞吐的影响

```python
import time
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
params = SamplingParams(temperature=0, max_tokens=200)

prompt = "Explain the concept of neural networks in detail."

for batch_size in [1, 4, 16, 64, 128]:
    prompts = [prompt] * batch_size
    start = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed

    print(f"Batch={batch_size:3d}  "
          f"Time={elapsed:.1f}s  "
          f"Tokens={total_tokens:5d}  "
          f"Throughput={throughput:.0f} tok/s")
```

你应该能观察到：
- 吞吐量随 batch 增大而提升
- 提升逐渐趋于平缓（受显存和计算限制）
- 单请求延迟可能略有增加

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 核心 API | `LLM` 加载模型，`SamplingParams` 控制生成 |
| generate vs chat | generate 接受原始文本，chat 自动应用对话模板 |
| RequestOutput | 包含 prompt 信息和生成结果（可能多个） |
| 引擎参数 | gpu_memory_utilization、max_model_len、dtype 最常调整 |
| 批量推理 | vLLM 自动优化批处理，吞吐随 batch 增大提升 |

---

## 动手实验

### 实验 1：采样参数对比

用相同的 prompt，分别设置 temperature=0, 0.5, 1.0, 1.5，观察生成结果的差异。每组设置 n=3 生成多个结果，比较多样性。

### 实验 2：批量推理吞吐测试

运行 5.6 节的吞吐测试代码，记录不同 batch size 下的吞吐量，画出吞吐-batch 曲线。

### 实验 3：引擎参数调优

用同一个模型，分别设置 `gpu_memory_utilization=0.5` 和 `0.95`，对比：
- 最大可服务的并发请求数
- batch=64 时的吞吐量

---

## 练习题

### 基础题

1. `SamplingParams` 中 `temperature=0` 意味着什么？与 `temperature=1` 有什么区别？
2. `RequestOutput.outputs` 为什么是一个列表而不是单个值？
3. `gpu_memory_utilization=0.9` 意味着什么？

### 实践题

4. 编写代码，用 vLLM 批量翻译 10 个中文句子为英文。使用 chat 格式，设置合适的 system prompt。
5. 测量你的 GPU 上 vLLM 加载 Qwen2.5-1.5B-Instruct 后的显存占用。

### 思考题

6. 为什么 vLLM 批量处理 100 个 prompt 的总时间，远小于逐个处理 100 次的总时间？
7. `max_model_len` 设置得越大越好吗？它的权衡点在哪里？
