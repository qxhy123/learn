# 第26章：vLLM 源码架构

> 读源码不是目的，理解设计决策才是。本章带你走一遍 vLLM 的关键路径，让你在遇到问题时知道该去哪里找答案。

---

## 学习目标

学完本章，你将能够：

1. 理解 vLLM 源码的模块划分和职责
2. 跟踪一个请求从 API 到输出的完整路径
3. 定位关键组件的源码位置
4. 理解 V1 引擎重构的设计目标
5. 具备阅读和修改 vLLM 源码的基础能力

---

## 26.1 代码结构概览

### 顶层目录

```
vllm/
├── vllm/
│   ├── entrypoints/       # API 入口（OpenAI server, offline LLM）
│   ├── engine/            # 引擎核心（LLMEngine, AsyncLLMEngine）
│   ├── core/              # 调度器和块管理
│   ├── worker/            # GPU Worker 和 ModelRunner
│   ├── model_executor/    # 模型执行和模型定义
│   ├── attention/         # 注意力后端（PagedAttention 等）
│   ├── lora/              # LoRA 支持
│   ├── spec_decode/       # 投机解码
│   ├── distributed/       # 分布式通信
│   ├── transformers_utils/ # HuggingFace 适配
│   └── ...
```

### 关键模块职责

| 模块 | 职责 |
|------|------|
| `entrypoints/` | 接收 HTTP 请求，解析参数，调用引擎 |
| `engine/` | 协调调度器、分词器和 Worker |
| `core/scheduler.py` | 调度决策：谁运行、谁等待、谁被抢占 |
| `core/block_manager.py` | KV Cache 块的分配、释放、CoW |
| `worker/worker.py` | 单 GPU 上的执行管理 |
| `worker/model_runner.py` | 准备模型输入、执行前向计算 |
| `model_executor/models/` | 各模型架构的 vLLM 实现 |
| `attention/backends/` | 不同注意力 kernel 的实现 |

---

## 26.2 请求生命周期

### 完整调用链

```
1. 用户发送 HTTP 请求
   ↓
2. entrypoints/openai/api_server.py
   接收请求，解析参数
   ↓
3. engine/async_llm_engine.py
   AsyncLLMEngine.add_request()
   ↓
4. core/scheduler.py
   Scheduler.schedule()
   决定本次 iteration 运行哪些请求
   ↓
5. core/block_manager.py
   BlockSpaceManager.allocate() / append_slots()
   为请求分配或追加 KV Cache 块
   ↓
6. worker/worker.py
   Worker.execute_model()
   ↓
7. worker/model_runner.py
   ModelRunner.execute_model()
   准备输入张量 → 调用模型前向 → 采样
   ↓
8. model_executor/models/llama.py (例)
   LlamaForCausalLM.forward()
   ↓
9. attention/backends/flash_attn.py (例)
   FlashAttentionBackend.forward()
   执行 PagedAttention kernel
   ↓
10. 返回采样结果 → 更新序列 → 返回给用户
```

### 核心循环

vLLM 的核心是一个无限循环：

```python
# 简化的核心循环
while True:
    # 1. 调度
    scheduler_output = self.scheduler.schedule()

    # 2. 执行
    if scheduler_output.has_work():
        output = self.model_executor.execute_model(scheduler_output)

    # 3. 处理结果
    for seq, sample in zip(scheduler_output.sequences, output.samples):
        seq.append_token(sample.token_id)
        if seq.is_finished():
            self.scheduler.free_seq(seq)
            self.output_queue.put(seq.get_output())
```

---

## 26.3 关键源码阅读指南

### 调度器

```python
# vllm/core/scheduler.py
class Scheduler:
    def schedule(self) -> SchedulerOutputs:
        """每次 iteration 的核心调度逻辑"""
        # 1. 处理 running 队列
        # 2. 处理 swapped 队列
        # 3. 处理 waiting 队列
        # 4. 返回调度结果
```

### 块管理器

```python
# vllm/core/block_manager.py (v1) 或 block/ 目录 (v2)
class BlockSpaceManager:
    def can_allocate(self, seq) -> bool:
        """检查是否有足够的物理块"""

    def allocate(self, seq):
        """为新序列分配初始块"""

    def append_slots(self, seq):
        """为生成中的序列追加新 slot/块"""

    def free(self, seq):
        """释放序列的所有块"""
```

### 模型实现

```python
# vllm/model_executor/models/llama.py
class LlamaForCausalLM(nn.Module):
    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        """模型前向计算"""
        # 1. 嵌入层
        # 2. 逐层 Transformer
        #    - 自注意力 (使用 PagedAttention)
        #    - FFN
        # 3. LM Head → logits
```

---

## 26.4 V1 引擎重构

vLLM 正在进行 V1 引擎的重构，主要变化包括：

### 设计目标

```
V0 引擎:
  - 成熟稳定
  - 但代码复杂度增长快
  - 某些优化难以实现（如 disaggregated prefill）

V1 引擎:
  - 更清晰的抽象层次
  - 更好的可扩展性
  - 支持新的调度模式
  - 更高的性能天花板
```

### 关键变化

1. **块管理重构**：更灵活的块分配策略
2. **调度器重设计**：支持 disaggregated prefill/decode
3. **执行引擎抽象**：更容易适配不同硬件
4. **异步化改进**：减少 CPU-GPU 同步开销

---

## 26.5 参与贡献

### 开发环境搭建

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e ".[dev]"

# 运行测试
pytest tests/test_basic.py -v

# 代码风格检查
ruff check vllm/
mypy vllm/
```

### 贡献方向

| 方向 | 难度 | 示例 |
|------|------|------|
| Bug 修复 | 低 | 修复特定模型的加载问题 |
| 新模型支持 | 中 | 添加新架构的 vLLM 实现 |
| 性能优化 | 高 | 优化注意力 kernel |
| 新特性 | 高 | 实现新的调度策略 |

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 模块划分 | entrypoints → engine → scheduler/block_manager → worker → model |
| 核心循环 | schedule → execute → process_output |
| 关键代码 | scheduler.py、block_manager.py、model_runner.py |
| V1 重构 | 更清晰的抽象，支持新调度模式 |

---

## 练习题

### 实践题

1. 克隆 vLLM 源码，找到 Llama 模型的 `forward` 方法，阅读其调用链。
2. 在 `scheduler.py` 中找到抢占逻辑的实现位置。

### 思考题

3. vLLM 为什么需要自己实现模型（而不是直接用 HuggingFace 的实现）？
4. 如果你要给 vLLM 添加一个新特性（如请求优先级），需要修改哪些模块？
