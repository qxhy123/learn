# 第15章：投机解码

> 投机解码的直觉没有变，仍然是“先猜，再验证”；但当前仓库里的实现已经比最早期的 draft-model 方案更丰富，包含 `ngram`、`draft_model`、`eagle/eagle3`、`mtp` 等多种路径，并且这些能力已经被调度器统一吸收到同一套 token 预算模型里。

---

## 学习目标

学完本章，你将能够：

1. 理解投机解码为什么能减少大模型 decode 次数
2. 区分当前仓库中不同 speculative 方法的实现路径
3. 用当前仓库真实支持的 `speculative_config` 接口启用投机解码
4. 理解 speculative token 如何进入 V1 调度器
5. 结合源码和指标观察 acceptance length / acceptance rate

---

## 15.1 为什么 decode 阶段适合做投机？

decode 的核心问题是：

- 每一步通常只生成 1 个 token
- 却要重新走一遍大模型的大量权重读写与算子调度

所以直觉上你会问：

> 能不能一次让大模型“确认”多个 token，而不是每次只确认 1 个？

投机解码给出的答案是：

1. 先用更便宜的方法生成候选 token
2. 再让主模型一次性验证一串候选
3. 接受连续正确的前缀
4. 在第一个不一致的位置退回主模型真实结果

只要候选质量足够好，就能把：

- “一次主模型前向 = 一个 token”

变成：

- “一次主模型前向 = 多个被接受 token”

---

## 15.2 当前仓库里有哪些 speculative 方法？

直接看源码目录：

- `vllm/vllm/v1/spec_decode/`

你会看到这些实现文件：

- `ngram_proposer.py` / `ngram_proposer_gpu.py`（含 GPU 加速版）
- `draft_model.py`
- `eagle.py`
- `medusa.py`
- `dflash.py`
- `suffix_decoding.py`
- `extract_hidden_states.py`（提取隐藏状态，供 EAGLE 等方法使用）
- `metadata.py` / `metrics.py` / `utils.py`（元数据、指标、工具）

此外，worker 侧的 speculative decoding 实现在 `v1/worker/gpu/spec_decode/` 下，包括：

- `rejection_sampler.py`（rejection sampling 验证逻辑）
- `eagle/speculator.py`（EAGLE speculator 实现）
- `eagle/cudagraph.py`（EAGLE cudagraph 支持）

从当前仓库公开示例 `examples/offline_inference/spec_decode.py` 看，教程里最值得掌握的是：

| 方法 | 说明 | 典型场景 |
|------|------|----------|
| `ngram` | 不需要额外草稿模型，直接从 prompt / 历史模式做猜测（含 GPU 加速版） | 翻译、改写、重复模式强 |
| `draft_model` | 传统”小模型草稿 + 大模型验证” | target 很大、draft 很小 |
| `eagle` / `eagle3` | 使用专门训练的 speculative 模型路径 | 延迟优化重点场景 |
| `medusa` | 多头并行预测 | 模型有对应 Medusa head 时 |

仓库里还能看到 `dflash`、`suffix_decoding` 等实现，可根据具体场景选择。

---

## 15.3 当前 V1 里，投机 token 是怎么进入调度器的？

这是最值得结合源码理解的一点。

V1 并没有给 speculative decoding 单独搞一套调度器，而是把它并入了统一抽象：

- `request.spec_token_ids`
- `request.num_tokens_with_spec`
- `scheduled_spec_decode_tokens`

在 `vllm/vllm/v1/core/sched/scheduler.py` 顶部的注释里，源码已经明确说明：

- 调度器只关心“请求应该算到哪里”和“已经算到哪里”
- speculative token 只是把 `num_tokens_with_spec` 往前推得更远

因此，同一个 `schedule()` 可以同时处理：

- 普通 prefill
- 普通 decode
- chunked prefill
- prefix caching
- speculative decoding

这正是 V1 架构比旧实现更整洁的地方。

---

## 15.4 一次 speculative 迭代在源码层面的直觉

你可以把它粗略理解成下面这个过程：

```text
当前上下文
  ↓
proposer 先给出 spec token 序列
  ↓
request.spec_token_ids 被写入请求状态
  ↓
Scheduler 按 num_tokens_with_spec 给该请求分配预算
  ↓
Worker / attention backend 一次处理更多位置
  ↓
根据验证结果接受前缀、拒绝第一个不一致位置
  ↓
更新真实 output token 和下一轮状态
```

也就是说，投机解码的关键不只是“猜”，还包括：

- 如何让调度器为这些 speculative positions 留预算
- 如何让 worker 知道哪些 token 是 draft，哪些是 bonus/verified
- 如何把 acceptance 统计出来

这些逻辑在当前仓库里散布于：

- `v1/spec_decode/`
- `v1/core/sched/scheduler.py`
- `v1/attention/backend.py`
- `v1/worker/gpu/model_runner.py`

---

## 15.5 用当前仓库真实接口启用投机解码

### 离线推理：推荐使用 `speculative_config`

当前仓库官方示例写法是：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_config={
        "method": "draft_model",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "num_speculative_tokens": 5,
    },
)

params = SamplingParams(temperature=0, max_tokens=200)
outputs = llm.generate(["Explain quantum computing."], params)
```

### `ngram` 示例

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 4,
        "prompt_lookup_min": 2,
        "prompt_lookup_max": 5,
    },
)
```

### `eagle3` 示例

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle3",
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
    },
)
```

### 为什么教程不再把 `speculative_model=...` 作为主写法？

因为当前仓库更稳定、也更通用的接口已经是：

- `speculative_config={...}`

它能表达：

- 具体方法
- 草稿模型
- speculative token 数
- ngram 查找参数
- eagle 并行 drafting 等额外开关

---

## 15.6 服务模式的真实配置方式

当前仓库服务端已经把 speculative 参数收敛成一个 JSON 配置：

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --speculative-config '{
    "method": "draft_model",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "num_speculative_tokens": 5
  }'
```

对于 `ngram`：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "ngram",
    "num_speculative_tokens": 4,
    "prompt_lookup_min": 2,
    "prompt_lookup_max": 5
  }'
```

这比旧教程里分散的 `--speculative-model`、`--num-speculative-tokens` 更贴近当前源码和参数系统。

---

## 15.7 接受率、接受长度和加速效果怎么理解？

### 关键不是“猜了多少”，而是“接受了多少”

真正决定收益的指标不是：

- 每次 draft 了几个 token

而是：

- 平均每次主模型验证后，能接受几个 token

当前仓库的官方示例就会统计：

- `vllm:spec_decode_num_drafts`
- `vllm:spec_decode_num_draft_tokens`
- `vllm:spec_decode_num_accepted_tokens`
- `vllm:spec_decode_num_accepted_tokens_per_pos`

### 经验判断

| 场景 | 典型现象 |
|------|----------|
| 草稿和目标模型高度同族 | acceptance 通常更高 |
| 翻译、格式转换、固定模板输出 | `ngram`/spec decode 更容易赚钱 |
| 创意写作、强随机采样 | acceptance 往往下降 |
| 已经高 batch 跑满 GPU | speculative 收益会被摊薄 |

### `num_speculative_tokens` 不是越大越好

它太小：

- 主模型一次验证不了多少 token

它太大：

- 后段 token 很容易被拒绝
- draft 成本反而浪费

工程上通常从 `2~5` 开始试最合理。

---

## 15.8 当前仓库里和投机解码耦合最紧的模块

| 主题 | 关键文件 | 说明 |
|------|----------|------|
| 参数入口 | `vllm/vllm/engine/arg_utils.py` | `--speculative-config` 解析 |
| 示例用法 | `vllm/examples/offline_inference/spec_decode.py` | 当前仓库最直观的配置参考 |
| 核心实现 | `vllm/vllm/v1/spec_decode/` | 各 speculative 方法（proposer 层） |
| Worker 侧实现 | `vllm/vllm/v1/worker/gpu/spec_decode/` | rejection sampler、EAGLE speculator |
| 调度协同 | `vllm/vllm/v1/core/sched/scheduler.py` | spec token 如何进入调度预算 |
| 注意力元数据 | `vllm/vllm/v1/attention/backend.py` | speculative positions 对 attention 的影响 |
| 树形注意力 | `vllm/vllm/v1/attention/backends/tree_attn.py` | 投机验证时的 tree attention |
| 运行时指标 | `vllm/vllm/v1/spec_decode/metrics.py` | acceptance 相关指标 |

---

## 15.9 源码阅读建议

如果你打算顺着 speculative decoding 读一遍源码，建议顺序是：

1. `examples/offline_inference/spec_decode.py`
2. `engine/arg_utils.py` 中 `speculative_config` 的解析
3. `v1/spec_decode/` 中你关心的方法实现
4. `scheduler.py` 中 `request.spec_token_ids` 的处理
5. 指标与 acceptance 统计

这么读的好处是：

- 先把“怎么配”搞清楚
- 再把“怎么跑”串起来
- 最后再下沉到某个具体 proposer

---

## 本章小结

| 概念 | 当前仓库中的真实语义 |
|------|----------------------|
| 主配置接口 | `speculative_config={...}` |
| 主方法族 | `ngram`（含 GPU 加速版）、`draft_model`、`eagle/eagle3`、`medusa` |
| 调度抽象 | speculative token 并入统一 token 预算模型 |
| 验证机制 | rejection sampling（在 `v1/worker/gpu/spec_decode/rejection_sampler.py`）|
| 关键收益指标 | acceptance length / acceptance rate |
| 典型误区 | 只盯 draft 长度，不看实际接受长度 |

---

## 练习题

### 基础题

1. 为什么 speculative decoding 能减少主模型 decode 次数？
2. 当前仓库为什么更推荐 `speculative_config`，而不是孤立的 `speculative_model` 参数？
3. `ngram` 和 `draft_model` 的最大区别是什么？

### 实践题

4. 打开 `examples/offline_inference/spec_decode.py`，列出当前示例支持的全部 method。
5. 在 `scheduler.py` 中找到 `request.spec_token_ids` 的处理逻辑，确认 speculative token 是如何进入本轮调度的。

### 思考题

6. 为什么说 speculative decoding 在高 batch、GPU 已经接近打满时收益会下降？
7. 如果 acceptance 很低，你应该优先减小 `num_speculative_tokens`、换 proposer，还是先调采样温度？
