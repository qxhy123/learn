# 第26章：vLLM 源码架构

> 当前仓库里的 vLLM 已经全面切到 V1 核心架构。真正值得你掌握的，不再是“某个旧目录里有没有 `block_manager.py`”，而是请求如何在 `InputProcessor -> EngineCore -> Scheduler -> KVCacheManager -> Worker -> OutputProcessor` 这条路径上流动。

---

## 学习目标

学完本章，你将能够：

1. 理解当前仓库里 V1 架构的顶层模块划分
2. 跟踪离线推理和 OpenAI 服务两条入口的真实调用链
3. 识别 `LLMEngine`、`EngineCore`、`Scheduler`、`KVCacheManager`、`GPUModelRunner` 的职责边界
4. 理解多进程架构下 API server、engine core、GPU worker 之间如何协作
5. 知道读源码时应该从哪里切入，而不是陷在过时路径里

---

## 26.1 先校准一个事实：当前仓库的核心是 V1

官方文档 `vllm/docs/usage/v1_guide.md` 已经明确说明：

- V0 已被 fully deprecated
- 当前仓库主线是 V1 core engine

这意味着你在读源码时要先建立一个正确心智模型：

### 仍然存在的“兼容层”

仓库里仍然能看到这些目录和类：

- `vllm/vllm/engine/llm_engine.py`
- `vllm/vllm/engine/async_llm_engine.py`
- `vllm/vllm/entrypoints/openai/api_server.py`

它们没有消失，但更多是：

- 用户接口
- 兼容层
- 入口包装

### 真正决定调度和执行行为的“核心层”

当前仓库真正的核心实现集中在：

- `vllm/vllm/v1/engine/`
- `vllm/vllm/v1/core/`
- `vllm/vllm/v1/executor/`
- `vllm/vllm/v1/worker/`
- `vllm/vllm/v1/attention/`

所以，如果你看到教程里还在大量引用：

- `core/block_manager.py`
- `worker/model_runner.py`
- `swapped queue`

那就要立刻警觉：这通常是在讲旧抽象，不是当前仓库的主路径。

---

## 26.2 目录结构：哪些目录真的值得读？

下面是结合当前仓库整理后的阅读优先级版本：

```text
vllm/
├── vllm/
│   ├── entrypoints/                # Python LLM、OpenAI server、CLI 等入口
│   ├── engine/                     # 入口参数和兼容层
│   ├── config/                     # 各类配置对象，含 SchedulerConfig
│   ├── sampling_params.py          # 采样与 structured_outputs 参数
│   ├── model_executor/             # 模型定义、加载、层实现、kernel glue
│   ├── v1/
│   │   ├── engine/                 # LLMEngine、AsyncLLM、EngineCore、Input/OutputProcessor
│   │   │   │                       #   core_client、detokenizer、coordinator 等
│   │   ├── core/                   # Scheduler、KVCacheManager、KVCacheCoordinator、
│   │   │   │                       #   BlockPool、request queue、encoder cache
│   │   │   └── sched/              # 调度器核心：scheduler、async_scheduler、output、interface
│   │   ├── executor/               # 单进程/多进程/Ray 执行器
│   │   ├── worker/                 # GPU/CPU/XPU worker 和 model runner
│   │   │   └── gpu/                # GPU 专属：model_runner、input_batch、block_table、
│   │   │       │                   #   sample/、spec_decode/、mm/、structured_outputs 等
│   │   ├── attention/              # V1 注意力后端与 paged attention op
│   │   │   ├── backends/           # flash_attn、flashinfer、triton、MLA、mamba、flex 等
│   │   │   └── ops/                # paged_attn、merge_attn_states、prefix_prefill 等算子
│   │   ├── sample/                 # V1 采样器：logits processor、sampler、rejection sampler
│   │   ├── spec_decode/            # speculative decoding 实现
│   │   ├── structured_output/      # grammar/backend/bitmask 管理
│   │   ├── kv_offload/             # KV offload（CPU offload、策略、worker）
│   │   ├── simple_kv_offload/      # 简化版 KV offload
│   │   ├── pool/                   # pooling / late interaction 推理
│   │   └── metrics/                # 运行时指标与统计
│   ├── distributed/                # KV transfer、通信、DP/TP 等
│   └── ...
├── docs/
│   ├── design/                     # 架构、prefix caching、metrics 等设计文档
│   └── usage/                      # V1 guide、使用说明
└── examples/                       # 离线推理、prefix caching、spec decode 等示例
```

### 关键模块职责

| 模块 | 当前职责 |
|------|----------|
| `entrypoints/` | 用户入口，接收 HTTP 请求或 Python 调用 |
| `v1/engine/` | 输入处理、引擎主循环、输出聚合、detokenizer、core_client |
| `v1/core/` | 调度、KV 块管理（KVCacheManager → Coordinator → BlockPool）、等待队列 |
| `v1/core/sched/` | 调度器核心：同步/异步 scheduler、输出定义、请求队列、调度策略 |
| `v1/executor/` | 把调度好的 batch 分发到 worker（uniproc/multiproc/ray） |
| `v1/worker/` | 单设备上的模型执行、cudagraph、输入 batch |
| `v1/worker/gpu/` | GPU 专属子模块：model_runner、sample/、spec_decode/、mm/、structured_outputs |
| `v1/attention/` | 注意力后端和 paged attention 算子 |
| `v1/sample/` | V1 采样器：logits processor、sampler、rejection sampler |
| `model_executor/models/` | 各模型架构在 vLLM 中的实现 |
| `sampling_params.py` | 采样参数、structured outputs 参数定义 |

---

## 26.3 两条真实入口：离线 `LLM` 和在线 `serve`

### 入口 1：离线推理 `LLM`

用户最常写的是：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
outputs = llm.generate(["hello"], SamplingParams(max_tokens=16))
```

这条路的关键入口在：

- `vllm/vllm/entrypoints/llm.py`

它最终会组装出 V1 的 `LLMEngine`。

### 入口 2：OpenAI 兼容服务

服务模式常见启动方式：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct
```

这条路的关键入口在：

- `vllm/vllm/entrypoints/cli/main.py`
- `vllm/vllm/entrypoints/openai/api_server.py`

API server 负责：

- 接收 HTTP 请求
- 校验/解析协议
- 渲染 chat template、处理多模态输入
- 通过 engine client 把请求交给 EngineCore

离线 `LLM` 和在线 `serve` 最终都会落到 V1 engine core，只是上游包装不同。

---

## 26.4 一次请求的真实调用链

结合当前仓库，最值得记住的路径是：

```text
用户请求
  ↓
entrypoints/llm.py 或 entrypoints/openai/api_server.py
  ↓
v1/engine/llm_engine.py
  ↓
InputProcessor
  ↓
EngineCoreClient
  ↓
EngineCore
  ↓
Scheduler.schedule()
  ↓
KVCacheManager.get_computed_blocks() / allocate_slots()
  ↓
Executor
  ↓
GPU Worker / GPUModelRunner
  ↓
attention backend / model forward / sampler
  ↓
EngineCore.update_from_output(...)
  ↓
OutputProcessor
  ↓
流式或最终输出返回用户
```

### 对应源码位置

| 阶段 | 关键文件 |
|------|----------|
| 入口 | `vllm/vllm/entrypoints/llm.py` / `vllm/vllm/entrypoints/openai/api_server.py` |
| 引擎外层 | `vllm/vllm/v1/engine/llm_engine.py` / `vllm/vllm/v1/engine/async_llm.py` |
| 输入处理 | `vllm/vllm/v1/engine/input_processor.py` |
| engine core 客户端 | `vllm/vllm/v1/engine/core_client.py` |
| engine core | `vllm/vllm/v1/engine/core.py` |
| 调度 | `vllm/vllm/v1/core/sched/scheduler.py` / `async_scheduler.py` |
| 内存管理 | `vllm/vllm/v1/core/kv_cache_manager.py` → `kv_cache_coordinator.py` → `block_pool.py` |
| 执行器 | `vllm/vllm/v1/executor/uniproc_executor.py` / `multiproc_executor.py` / `ray_executor.py` |
| worker | `vllm/vllm/v1/worker/gpu_worker.py` |
| model runner | `vllm/vllm/v1/worker/gpu/model_runner.py` |
| 输出处理 | `vllm/vllm/v1/engine/output_processor.py` |

---

## 26.5 `LLMEngine` 在当前仓库里扮演什么角色？

`vllm/vllm/v1/engine/llm_engine.py` 里的 `LLMEngine` 可以理解成“外层编排器”。

它在初始化时做了几件关键事：

1. 保存 `vllm_config`
2. 创建 `renderer`（负责 chat template / tokenizer）
3. 创建 `InputProcessor`（把用户输入转成 `EngineCoreRequest`）
4. 创建 `OutputProcessor`（把 `EngineCoreOutputs` 转成 `RequestOutput`）
5. 通过 `EngineCoreClient.make_client(...)` 建立和 engine core 的连接
6. 可选创建 `StatLoggerManager` 进行指标统计

你可以把它看成：

```text
LLMEngine
  ├── renderer (tokenizer + chat template)
  ├── InputProcessor
  ├── EngineCoreClient
  ├── OutputProcessor
  └── StatLoggerManager (可选)
```

注意：异步服务场景下，对应的入口是 `vllm/vllm/v1/engine/async_llm.py` 中的 `AsyncLLM`，它和 `LLMEngine` 共享相同的 Input/Output Processor 架构，但使用异步 `EngineCoreClient`。

这层自己不实现真正的调度算法，也不直接操作 KV block。
它更像一个“胶水层”：

- 往里把请求变成 `EngineCoreRequest`
- 往外把 `EngineCoreOutputs` 变成人类能用的 `RequestOutput`

如果你是从使用者视角切入源码，`LLMEngine.add_request()` 和 `LLMEngine.step()` 是非常好的起点。

---

## 26.6 `EngineCore`：真正的内循环

如果说 `LLMEngine` 是外层编排器，那：

- `vllm/vllm/v1/engine/core.py`

里的 `EngineCore` 就是真正的内循环。

它初始化时会做这些关键工作：

1. 创建 `model_executor`（通过 `executor_class`）
2. 调 `_initialize_kv_caches()` 进行显存 profiling、计算 KV cache 容量并初始化
3. 创建 `StructuredOutputManager`（结构化输出 grammar 管理）
4. 根据 `SchedulerConfig.get_scheduler_cls()` 选择调度器类（同步 `Scheduler` 或异步 `AsyncScheduler`）并初始化
5. 初始化 batch queue（用于 pipeline parallelism 消除 pipeline bubbles）
6. 初始化 request block hasher（用于 prefix caching）
7. 选择 `step_fn`：根据是否启用 batch queue 选择 `step` 或 `step_with_batch_queue`

### `EngineCore.step()` 的核心结构

`step()` 的源码可以直接展开：

```python
def step(self):
    if not self.scheduler.has_requests():
        return {}, False

    # 1. 调度：决定这一轮谁跑、分配 KV block
    scheduler_output = self.scheduler.schedule()

    # 2. 执行模型前向（非阻塞）
    future = self.model_executor.execute_model(scheduler_output, non_block=True)

    # 3. 获取结构化输出的 grammar bitmask（CPU 侧并行）
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

    # 4. 等待模型执行完成
    model_output = future.result()

    # 5. 如果模型没有直接采样，用 grammar bitmask 做采样
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)

    # 6. 处理本轮中发生的 abort
    self._process_aborts_queue()

    # 7. 根据模型输出更新所有请求状态
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

有几个值得注意的设计：

1. **模型执行和 grammar bitmask 是并行的**：`execute_model` 以 `non_block=True` 启动，然后 CPU 侧立刻去算 grammar bitmask，再等模型结果。
2. **采样可以延迟**：如果模型前向没有直接产出采样结果（例如需要先应用 bitmask），采样在 `sample_tokens` 中完成。
3. **speculative decoding 的 draft token 更新**：在 `post_step()` 中，如果不是异步调度，会从 executor 获取 draft token ids 并更新到调度器。

### Pipeline Parallelism 时的变体：`step_with_batch_queue`

当 `batch_queue_size > 1` 时（PP 场景），engine core 使用 `step_with_batch_queue` 代替 `step`。它通过一个双端队列实现批次的异步调度和执行：

```text
调度 batch N+1（非阻塞）→ 加入队列
                         ↓
                   等队列头部 batch N 执行完
                         ↓
                   处理 batch N 的输出
```

这样可以在 batch N 还在 GPU 上执行时，CPU 侧提前完成 batch N+1 的调度，消除 pipeline bubbles。

---

## 26.7 `Scheduler` 和 `KVCacheManager` 的边界

源码阅读时一个很容易混淆的问题是：

> “到底是调度器在管理显存，还是 KVCacheManager 在管理显存？”

答案是：

- 调度器负责决策
- `KVCacheManager` 负责资源账本和块生命周期

### `Scheduler` 负责的事

- running/waiting 请求谁先上
- 这轮每个请求给多少 token 预算
- 不够块时抢占谁
- 何时接纳新请求
- 何时推进 structured output / remote KV / streaming input 等阻塞态

### `KVCacheManager` 负责的事

`KVCacheManager` 内部通过 `KVCacheCoordinator`（在 `kv_cache_coordinator.py`）和 `BlockPool`（在 `block_pool.py`）分层管理：

- prefix cache 命中查询（`get_computed_blocks`）
- block 是否够用（通过 coordinator 判断）
- 新 block 分配（`allocate_slots`）
- block hash 和 cache blocks 维护（coordinator → block pool）
- block 释放（`free`）
- prefix cache reset

当前架构是三层：`KVCacheManager` → `KVCacheCoordinator` → `BlockPool` + `FreeKVCacheBlockQueue`

这也是为什么当前教程里应该讲：

- `Scheduler.schedule()`
- `KVCacheManager.allocate_slots()`

而不是继续围绕旧的 `block_manager.py` 展开。

---

## 26.8 Worker、Executor 和 Model Runner 怎么分工？

### Executor：批次的分发者

执行器在：

- `vllm/vllm/v1/executor/uniproc_executor.py`（单进程，调试和小规模场景）
- `vllm/vllm/v1/executor/multiproc_executor.py`（多进程，TP/PP 场景）
- `vllm/vllm/v1/executor/ray_executor.py` / `ray_executor_v2.py`（Ray 分布式场景）
- `vllm/vllm/v1/executor/abstract.py`（抽象基类 `Executor`）

它负责把调度器给出的 batch：

- 组织成 worker 能吃的执行请求
- 发往单进程、多进程或 Ray worker

### Worker：设备级的运行实体

GPU worker 主要在：

- `vllm/vllm/v1/worker/gpu_worker.py`

它负责：

- 设备初始化
- 加载模型
- 管理本设备侧缓存和执行状态

### GPUModelRunner：真正准备 batch 并前向执行的人

最值得看的文件之一是：

- `vllm/vllm/v1/worker/gpu/model_runner.py`

这里通常会处理：

- 输入 batch 整理
- block table / slot mapping 准备
- attention metadata
- cudagraph / workspace
- 调模型前向
- 采样前后的辅助逻辑

可以粗略理解成：

```text
Scheduler 决定”谁跑”
Executor 负责”怎么送过去”
GPUModelRunner 决定”送过去之后怎么真正算”
```

### GPUModelRunner 的内部子模块

当前 `v1/worker/gpu/` 目录已经非常丰富，model_runner 的职责被进一步拆分：

| 子模块 | 职责 |
|--------|------|
| `input_batch.py` | 管理 GPU 侧的 input batch 状态 |
| `block_table.py` | GPU 侧的 block table 管理 |
| `sample/` | 采样器（sampler、logprob、penalties、min_p、bad_words 等）|
| `spec_decode/` | GPU 侧的 speculative decoding（rejection sampler、EAGLE speculator）|
| `mm/` | 多模态：encoder_cache、encoder_runner、RoPE 处理 |
| `structured_outputs.py` | 结构化输出的 bitmask 应用 |
| `cudagraph_utils.py` | CUDAGraph 捕获和回放 |
| `warmup.py` | 模型预热逻辑 |

---

## 26.9 注意力后端与模型实现在哪里接上？

### 模型实现

模型定义主要在：

- `vllm/vllm/model_executor/models/`

例如 Llama、Qwen、Mistral 等都在这里有对应实现。

### 注意力后端

V1 的注意力后端在：

- `vllm/vllm/v1/attention/backends/`

当前已经非常丰富，主要包括：

| 后端 | 说明 |
|------|------|
| `flash_attn.py` | Flash Attention 主力后端 |
| `flashinfer.py` | FlashInfer 后端 |
| `triton_attn.py` | Triton 实现的注意力 |
| `rocm_attn.py` / `rocm_aiter_fa.py` | AMD ROCm 后端 |
| `mla/` | Multi-head Latent Attention（MLA）后端族，含 flashmla、cutlass_mla、triton_mla 等多种实现 |
| `flex_attention.py` | PyTorch Flex Attention 后端 |
| `mamba_attn.py` / `mamba1_attn.py` / `mamba2_attn.py` | Mamba SSM 系列后端 |
| `linear_attn.py` | 线性注意力后端 |
| `tree_attn.py` | 树形注意力（用于 speculative decoding 验证） |
| `cpu_attn.py` | CPU 注意力后端 |

后端注册和选择逻辑在 `registry.py` 和 `selector.py`。

### PagedAttention 相关算子

更底层的 paged attention op 在：

- `vllm/vllm/v1/attention/ops/paged_attn.py`

这里能看到诸如：

- `split_kv_cache(...)`
- `write_to_paged_cache(...)`

这部分是“理论上的 PagedAttention”真正落到内存布局和 kernel 调用的地方。

---

## 26.10 结构化输出、投机解码、多模态分别挂在哪？

这是理解“V1 为何更模块化”的最好例子。

### 结构化输出

- 参数入口：`vllm/vllm/sampling_params.py`（`StructuredOutputsParams`）
- runtime 管理：`vllm/vllm/v1/structured_output/`（后端选择、grammar 编译）
- engine 集成：`StructuredOutputManager`（在 `EngineCore` 初始化时创建）
- worker 侧：`vllm/vllm/v1/worker/gpu/structured_outputs.py`（bitmask 应用）

### 投机解码

- `vllm/vllm/v1/spec_decode/`

这里有：

- `draft_model.py`（传统小模型草稿）
- `ngram_proposer.py` / `ngram_proposer_gpu.py`（N-gram 猜测，含 GPU 加速版）
- `eagle.py`（EAGLE/EAGLE3 方法）
- `medusa.py`（Medusa 多头预测）
- `dflash.py`（DFlash 方法）
- `suffix_decoding.py`（后缀解码）
- `extract_hidden_states.py`（提取隐藏状态，供 EAGLE 等使用）
- `metadata.py` / `metrics.py` / `utils.py`（元数据、指标统计、工具函数）

此外，`v1/worker/gpu/spec_decode/` 下还有 worker 侧的 rejection sampler 和 EAGLE speculator 实现。

调度器并不会为它们单独写一套“特判流程”，而是通过：

- `request.spec_token_ids`
- `num_tokens_with_spec`
- `scheduled_spec_decode_tokens`

把它们并入统一调度抽象。

### 多模态

- `vllm/vllm/multimodal/`
- `vllm/vllm/renderers/`
- `vllm/vllm/v1/core/encoder_cache_manager.py`

多模态输入最终也会回到：

- encoder budget
- scheduler token budget
- worker 侧 batch 构造

---

## 26.11 当前仓库里最容易踩的“读源码误区”

### 误区 1：把兼容层当成核心层

`engine/async_llm_engine.py` 还在，并不等于当前主逻辑还在那里。
真正的执行内核已经转到 `v1/engine/core.py` 等目录。

### 误区 2：沿着旧的 `block_manager.py` 找下去

当前仓库主线更应该去看：

- `v1/core/kv_cache_manager.py`
- `v1/core/kv_cache_utils.py`

### 误区 3：默认认为还有 `swapped` 队列

当前 V1 主路径的请求状态里，关键状态是：

- `WAITING`
- `RUNNING`
- `PREEMPTED`

而不是旧资料常写的 `swapped`。

### 误区 4：以为 structured outputs 还是 `guided_*` API

当前仓库已切到：

- `structured_outputs`
- `response_format`

相关逻辑也都挂在 `v1/structured_output/` 下，而不是旧的 `guided_decoding` 流程。

---

## 26.12 推荐阅读顺序

如果你准备第一次系统读 vLLM 源码，我推荐这个顺序：

1. `vllm/vllm/v1/request.py`（请求对象和状态定义）
2. `vllm/vllm/config/scheduler.py`（调度器所有可调参数）
3. `vllm/vllm/v1/core/sched/scheduler.py`（核心调度算法）
4. `vllm/vllm/v1/core/kv_cache_manager.py`（KV 块管理外观接口）
5. `vllm/vllm/v1/core/kv_cache_coordinator.py`（coordinator 调度块分配策略）
6. `vllm/vllm/v1/core/block_pool.py`（block pool 和 prefix cache 实现）
7. `vllm/vllm/v1/engine/core.py`（引擎主循环）
8. `vllm/vllm/v1/engine/llm_engine.py`（外层编排器）
9. `vllm/vllm/v1/worker/gpu/model_runner.py`（GPU 侧 batch 准备和模型前向）
10. `vllm/vllm/v1/attention/backends/flash_attn.py`（Flash Attention 后端）

为什么是这个顺序？

- 先搞清请求对象和调度配置
- 再搞清调度和内存管理（KVCacheManager → Coordinator → BlockPool 三层）
- 然后回到 engine core 把主循环串起来
- 最后再下探到 worker 和 attention backend

这样不会一开始就淹死在 kernel 细节里。

---

## 本章小结

| 概念 | 当前仓库里的真实位置 |
|------|----------------------|
| 入口层 | `entrypoints/` |
| 外层引擎 | `v1/engine/llm_engine.py`（同步）/ `v1/engine/async_llm.py`（异步）|
| 核心内循环 | `v1/engine/core.py` |
| 调度 | `v1/core/sched/scheduler.py`（同步）/ `async_scheduler.py`（异步）|
| KV 管理 | `v1/core/kv_cache_manager.py` → `kv_cache_coordinator.py` → `block_pool.py` |
| 执行器 | `v1/executor/`（uniproc / multiproc / ray）|
| 设备执行 | `v1/worker/gpu_worker.py` + `v1/worker/gpu/model_runner.py` |
| 注意力后端 | `v1/attention/backends/`（flash_attn、flashinfer、MLA、mamba 等）|
| 采样器 | `v1/sample/`（logits processor、sampler）|
| 模型实现 | `model_executor/models/` |

---

## 练习题

### 实践题

1. 从 `vllm/vllm/entrypoints/llm.py` 出发，追到 `EngineCore.step()`，画出离线 `LLM.generate()` 的调用链。
2. 在 `scheduler.py` 中找到 `allocate_slots()` 的调用点，再追到 `kv_cache_manager.py`，确认调度和内存管理的职责边界。
3. 打开 `vllm/vllm/v1/worker/gpu/model_runner.py`，找出“准备输入 batch”和“调用模型前向”的关键函数。

### 思考题

4. 为什么 V1 要把 `InputProcessor` / `EngineCore` / `OutputProcessor` 明确拆开？
5. 如果你要给 vLLM 增加一个新的请求级约束能力，应该优先考虑改 `sampling_params.py`、`Scheduler`，还是 `GPUModelRunner`？
6. 为什么投机解码和结构化输出在 V1 里都能通过“统一调度抽象”挂进去，而不需要两套完全独立的引擎？
