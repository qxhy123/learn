# 第11章：模型加载与支持列表

> vLLM 的一大优势是广泛的模型支持——几乎所有主流的 Hugging Face 模型都可以直接加载。但理解模型加载的过程和限制，能帮你避免很多坑。

---

## 学习目标

学完本章，你将能够：

1. 了解 vLLM 支持的模型架构和限制
2. 理解模型加载的过程和权重格式
3. 掌握不同模型来源（HF Hub、本地、GGUF）的加载方式
4. 正确处理 `trust_remote_code` 和模型认证
5. 诊断和解决模型加载中的常见问题

---

## 11.1 支持的模型架构

### 主流支持架构

vLLM 支持的架构覆盖了绝大多数主流 LLM：

| 架构 | 代表模型 | 说明 |
|------|---------|------|
| LlamaForCausalLM | Llama 2/3/3.1, CodeLlama | 最广泛使用 |
| MistralForCausalLM | Mistral 7B, Mixtral | 包括 MoE |
| Qwen2ForCausalLM | Qwen 2/2.5 系列 | 阿里通义 |
| GemmaForCausalLM | Gemma, Gemma 2 | Google |
| GPTNeoXForCausalLM | Pythia, GPT-NeoX | EleutherAI |
| PhiForCausalLM | Phi-2, Phi-3 | Microsoft |
| ChatGLMModel | ChatGLM 3/4 | 智谱 AI |
| DeepseekV2ForCausalLM | DeepSeek V2/V3 | DeepSeek |
| InternLMForCausalLM | InternLM 2/2.5 | 上海 AI Lab |

### 多模态模型支持

| 架构 | 代表模型 | 模态 |
|------|---------|------|
| LlavaForConditionalGeneration | LLaVA | 图文 |
| Qwen2VLForConditionalGeneration | Qwen2-VL | 图文/视频 |
| InternVLChatModel | InternVL | 图文 |
| PaliGemmaForConditionalGeneration | PaliGemma | 图文 |
| MllamaForConditionalGeneration | Llama 3.2 Vision | 图文 |

### 如何检查模型是否支持

```python
# 方法 1：直接尝试加载
from vllm import LLM
try:
    llm = LLM(model="model-name")
    print("模型加载成功")
except ValueError as e:
    print(f"不支持: {e}")

# 方法 2：查看官方文档的支持列表
# https://docs.vllm.ai/en/latest/models/supported_models.html
```

---

## 11.2 模型加载过程

### 加载流程

```
1. 确定模型架构
   └── 读取 config.json 中的 architectures → ModelRegistry 解析

2. 下载权重文件
   └── 从 HF Hub 或本地路径加载 safetensors/bin 文件

3. 初始化模型结构
   └── 根据 config 创建 vLLM 内部的模型实例

4. 加载权重
   └── 将权重张量映射到 vLLM 模型参数（通过 load_weights）

5. 显存 Profiling + KV Cache 分配
   └── 根据剩余显存计算可用块数

6. 预热
   └── 运行 dummy 推理确保 CUDA kernel 编译完成
```

### V1 中步骤 5 的真实实现

步骤 5 在 V1 中由 `EngineCore._initialize_kv_caches()` 完成，这是整个初始化流程中最关键的环节：

```python
# 1. 询问模型需要哪些 KV cache 规格（每层的 head 数、head dim、dtype 等）
kv_cache_specs = self.model_executor.get_kv_cache_specs()

# 2. 做一次 dummy 前向来 profile 峰值显存，得出"除模型权重/激活外还剩多少显存"
available_gpu_memory = self.model_executor.determine_available_memory()

# 3. 根据可用显存和 KV cache 规格，计算每组 cache 能分成多少个 block
kv_cache_configs = get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory)

# 4. 真正在 GPU 上分配 KV cache 张量并预热模型
self.model_executor.initialize_from_config(kv_cache_configs)
```

这个 profiling 过程解释了为什么 vLLM 启动时有一段明显的"初始化时间"——它在真正做一次完整的显存探测，而不是简单估算。日志中的 `init engine (profile, create kv cache, warmup model) took X.XX seconds` 就是这个阶段。

如果 profiling 后发现可用显存不足以支持配置的 `max_model_len`，V1 还会自动尝试缩减 `max_model_len`（auto-fit），并将新值同步到所有 worker。

### 权重格式

vLLM 支持以下权重格式：

```python
# SafeTensors (推荐，更安全、更快)
# 文件: model-00001-of-00004.safetensors, ...

# PyTorch bin 格式
# 文件: pytorch_model-00001-of-00004.bin, ...

# GGUF 格式 (社区量化格式)
# 文件: model-Q4_K_M.gguf
```

---

## 11.3 加载方式

### 从 Hugging Face Hub 加载

```python
# 标准方式
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# 指定版本/分支
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", revision="main")
```

### 从本地路径加载

```python
# 先下载到本地
# huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./models/llama

llm = LLM(model="./models/llama")
```

### 加载量化模型

```python
# AWQ 量化模型
llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-AWQ",
    quantization="awq",
)

# GPTQ 量化模型
llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-GPTQ",
    quantization="gptq",
)
```

### 需要 trust_remote_code 的模型

某些模型（如 ChatGLM、部分 Qwen 模型）包含自定义代码：

```python
# ⚠️ 只在信任模型来源时使用
llm = LLM(
    model="THUDM/chatglm3-6b",
    trust_remote_code=True,
)
```

### 数据类型控制

```python
# 自动选择（推荐）
llm = LLM(model="...", dtype="auto")

# 强制 FP16
llm = LLM(model="...", dtype="float16")

# 强制 BF16（需要 Ampere 或更新的 GPU）
llm = LLM(model="...", dtype="bfloat16")
```

---

## 11.4 常见问题

### 问题 1：模型太大，显存不够

```python
# 方案 1：使用量化
llm = LLM(model="...", quantization="awq")

# 方案 2：减少最大序列长度
llm = LLM(model="...", max_model_len=2048)

# 方案 3：降低显存利用率（给其他进程留空间）
llm = LLM(model="...", gpu_memory_utilization=0.8)

# 方案 4：多卡张量并行
llm = LLM(model="...", tensor_parallel_size=2)
```

### 问题 2：max_model_len 超出模型支持

```
ValueError: The model's max seq len (32768) is larger than the maximum
number of tokens that can be stored in KV cache.
```

```python
# 显式限制最大长度
llm = LLM(model="...", max_model_len=8192)
```

### 问题 3：模型架构不支持

```
ValueError: Model architectures ['MyCustomModel'] are not supported.
```

解决方案：
- 检查是否需要 `trust_remote_code=True`
- 查看 vLLM 是否支持该架构
- 考虑使用 Hugging Face Transformers 直接推理

### 问题 4：权重加载慢

```python
# 使用 tensorizer 加速加载（实验性）
llm = LLM(
    model="...",
    load_format="tensorizer",
)

# 确保使用 safetensors 格式（比 .bin 快）
```

---

## 11.5 模型选择建议

### 按场景选择

| 场景 | 推荐模型 | 显存需求 |
|------|---------|---------|
| 快速测试 | Qwen2.5-0.5B-Instruct | 2 GB |
| 通用对话 | Qwen2.5-7B-Instruct | 16 GB |
| 代码生成 | CodeLlama-7B-Instruct | 16 GB |
| 中文能力 | Qwen2.5-14B-Instruct | 32 GB |
| 高质量 | Llama-3.1-70B-Instruct | 140 GB (2×A100) |
| 图文理解 | Qwen2-VL-7B-Instruct | 18 GB |

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 支持范围 | 覆盖 Llama、Qwen、Mistral、Gemma 等主流架构 |
| 加载方式 | HF Hub、本地路径、量化模型 |
| 权重格式 | SafeTensors（推荐）、PyTorch bin、GGUF |
| 常见问题 | OOM → 量化/限长/多卡；不支持 → trust_remote_code |

---

## 动手实验

### 实验 1：加载不同模型

分别加载 2-3 个不同架构的模型，记录加载时间和显存占用。

### 实验 2：对比数据类型

用同一模型的 FP16 和 BF16 版本，对比显存占用和生成质量。

---

## 练习题

### 基础题

1. vLLM 支持的权重格式有哪些？哪个推荐使用？
2. `trust_remote_code=True` 有什么安全风险？
3. 模型加载时，vLLM 如何确定可以分配多少 KV Cache 块？

### 思考题

4. 为什么 vLLM 不像 HuggingFace Transformers 那样支持所有模型？支持新模型需要哪些工作？
