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
│   │   ├── engine/                 # LLMEngine、EngineCore、Input/OutputProcessor
│   │   ├── core/                   # Scheduler、KVCacheManager、request queue
│   │   ├── executor/               # 单进程/多进程执行器
│   │   ├── worker/                 # GPU/CPU/XPU worker 和 model runner
│   │   ├── attention/              # V1 注意力后端与 paged attention op
│   │   ├── spec_decode/            # speculative decoding 实现
│   │   ├── structured_output/      # grammar/backend/bitmask 管理
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
| `v1/engine/` | 输入处理、引擎主循环、输出聚合 |
| `v1/core/` | 调度、KV 块管理、等待队列、请求状态 |
| `v1/executor/` | 把调度好的 batch 分发到 worker |
| `v1/worker/` | 单设备上的模型执行、cudagraph、输入 batch |
| `v1/attention/` | 注意力后端和 paged attention 算子 |
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
| 引擎外层 | `vllm/vllm/v1/engine/llm_engine.py` |
| 输入处理 | `vllm/vllm/v1/engine/input_processor.py` |
| engine core | `vllm/vllm/v1/engine/core.py` |
| 调度 | `vllm/vllm/v1/core/sched/scheduler.py` |
| 内存管理 | `vllm/vllm/v1/core/kv_cache_manager.py` |
| 执行器 | `vllm/vllm/v1/executor/uniproc_executor.py` / `multiproc_executor.py` |
| worker | `vllm/vllm/v1/worker/gpu_worker.py` |
| model runner | `vllm/vllm/v1/worker/gpu/model_runner.py` |
| 输出处理 | `vllm/vllm/v1/engine/output_processor.py` |

---

## 26.5 `LLMEngine` 在当前仓库里扮演什么角色？

`vllm/vllm/v1/engine/llm_engine.py` 里的 `LLMEngine` 可以理解成“外层编排器”。

它在初始化时做了几件关键事：

1. 保存 `vllm_config`
2. 创建 `InputProcessor`
3. 创建 `OutputProcessor`
4. 通过 `EngineCoreClient.make_client(...)` 建立和 engine core 的连接

你可以把它看成：

```text
LLMEngine
  ├── InputProcessor
  ├── EngineCoreClient
  └── OutputProcessor
```

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

1. 创建 `model_executor`
2. 调 `_initialize_kv_caches()` 计算和初始化 KV cache
3. 创建 `StructuredOutputManager`
4. 根据 `SchedulerConfig` 选择调度器类并初始化 scheduler
5. 初始化 batch queue、connector、指标统计等

### `EngineCore.step()` 的核心结构

`step()` 的职责可以概括成：

```text
收集请求
  → 调度
  → 执行模型
  → 更新请求状态
  → 产出输出
```

这也是你理解整个 V1 的最短路径。

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

- prefix cache 命中查询
- block 是否够用
- 新 block 分配
- block hash 和 cache blocks 维护
- block 释放
- prefix cache reset

这也是为什么当前教程里应该讲：

- `Scheduler.schedule()`
- `KVCacheManager.allocate_slots()`

而不是继续围绕旧的 `block_manager.py` 展开。

---

## 26.8 Worker、Executor 和 Model Runner 怎么分工？

### Executor：批次的分发者

执行器在：

- `vllm/vllm/v1/executor/uniproc_executor.py`
- `vllm/vllm/v1/executor/multiproc_executor.py`

它负责把调度器给出的 batch：

- 组织成 worker 能吃的执行请求
- 发往单进程或多进程 worker

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
Scheduler 决定“谁跑”
Executor 负责“怎么送过去”
GPUModelRunner 决定“送过去之后怎么真正算”
```

---

## 26.9 注意力后端与模型实现在哪里接上？

### 模型实现

模型定义主要在：

- `vllm/vllm/model_executor/models/`

例如 Llama、Qwen、Mistral 等都在这里有对应实现。

### 注意力后端

V1 的注意力后端在：

- `vllm/vllm/v1/attention/backends/`

常见后端包括：

- `flash_attn.py`
- `flashinfer.py`
- `triton_attn.py`
- `rocm_attn.py`

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

- 参数入口：`vllm/vllm/sampling_params.py`
- runtime 管理：`vllm/vllm/v1/structured_output/`
- engine 集成：`StructuredOutputManager`

### 投机解码

- `vllm/vllm/v1/spec_decode/`

这里有：

- `draft_model.py`
- `ngram_proposer.py`
- `eagle.py`
- `medusa.py`
- `dflash.py`
- `suffix_decoding.py`

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

1. `vllm/vllm/v1/request.py`
2. `vllm/vllm/config/scheduler.py`
3. `vllm/vllm/v1/core/sched/scheduler.py`
4. `vllm/vllm/v1/core/kv_cache_manager.py`
5. `vllm/vllm/v1/engine/core.py`
6. `vllm/vllm/v1/engine/llm_engine.py`
7. `vllm/vllm/v1/worker/gpu/model_runner.py`
8. `vllm/vllm/v1/attention/backends/flash_attn.py`

为什么是这个顺序？

- 先搞清请求对象和调度配置
- 再搞清调度和内存管理
- 然后回到 engine core 把主循环串起来
- 最后再下探到 worker 和 attention backend

这样不会一开始就淹死在 kernel 细节里。

---

## 本章小结

| 概念 | 当前仓库里的真实位置 |
|------|----------------------|
| 入口层 | `entrypoints/` |
| 外层引擎 | `v1/engine/llm_engine.py` |
| 核心内循环 | `v1/engine/core.py` |
| 调度 | `v1/core/sched/scheduler.py` |
| KV 管理 | `v1/core/kv_cache_manager.py` |
| 执行器 | `v1/executor/` |
| 设备执行 | `v1/worker/` |
| 注意力后端 | `v1/attention/backends/` |
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
