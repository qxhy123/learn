# 第17章：结构化输出

> 让模型“尽量输出 JSON”只是提示词工程；让模型“只能输出符合约束的 token”才是推理引擎层面的结构化输出。当前仓库里，这套能力已经不再围绕旧的 `guided_*` API，而是统一收敛到 `structured_outputs` 和 `response_format`。

---

## 学习目标

学完本章，你将能够：

1. 理解当前仓库中结构化输出的真实接口和执行路径
2. 使用 JSON Schema、正则、choice、grammar 等约束方式
3. 理解 `StructuredOutputManager`、grammar 编译和 token bitmask 的关系
4. 了解 `xgrammar`、`guidance`、`outlines` 等后端在源码中的挂载方式
5. 判断结构化输出对延迟和吞吐的影响来自哪里

---

## 17.1 为什么不能只靠 prompt？

只靠 prompt 约束的问题很直接：

- 模型可能在 JSON 前后多说废话
- 可能漏引号、漏逗号
- 可能输出额外字段
- 可能在分类任务中返回“我认为答案是 positive”而不是单个标签

结构化输出的目标不是“提高服从性”，而是：

> 在采样阶段直接屏蔽所有不合法 token。

这样模型即使想偏，也没有可偏的 token 空间。

---

## 17.2 当前仓库里的真实接口

### 接口 1：OpenAI 风格 `response_format`

这是当前最推荐的在线服务写法：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "介绍一下北京。"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "city_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "country": {"type": "string"},
                    "population": {"type": "number"},
                },
                "required": ["name", "country", "population"],
            },
        },
    },
)
```

在线协议转换的源码入口主要在：

- `vllm/vllm/entrypoints/openai/chat_completion/`
- `vllm/vllm/entrypoints/openai/completion/`

这些协议层会把 `response_format` 进一步转成内部的：

- `StructuredOutputsParams`

### 接口 2：OpenAI extra body `structured_outputs`

如果你不走 `response_format`，也可以显式传：

```python
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Classify this sentiment: vLLM is wonderful!",
        }
    ],
    extra_body={
        "structured_outputs": {
            "choice": ["positive", "negative"]
        }
    },
)
```

### 接口 3：离线推理 `SamplingParams(structured_outputs=...)`

当前仓库的离线接口应写成：

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

params = SamplingParams(
    temperature=0,
    max_tokens=128,
    structured_outputs=StructuredOutputsParams(
        json={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "confidence"],
        }
    ),
)

outputs = llm.generate(["法国的首都是什么？"], sampling_params=params)
print(outputs[0].outputs[0].text)
```

### 一个重要纠偏

如果你还看到这些旧字段：

- `guided_json`
- `guided_regex`
- `guided_choice`
- `guided_grammar`
- `guided_decoding_backend`

那说明资料已经过时。官方 `docs/features/structured_outputs.md` 也明确把它们标成 deprecated/removed 路线。

---

## 17.3 支持哪些约束类型？

`vllm/vllm/sampling_params.py` 里的 `StructuredOutputsParams` 当前主要支持：

| 字段 | 含义 |
|------|------|
| `json` | 约束为 JSON Schema（`str | dict | None`）|
| `json_object` | 只保证输出为合法 JSON 对象 |
| `regex` | 输出必须匹配正则 |
| `choice` | 输出只能是候选列表之一 |
| `grammar` | 输出必须符合 CFG / EBNF 文法 |
| `structural_tag` | 在指定 tag 内遵循结构化约束 |

以上 6 个字段互斥（只能设一个）。另外还有几个控制选项：

| 选项 | 含义 |
|------|------|
| `disable_any_whitespace` | 禁止 JSON 中的额外空白 |
| `disable_additional_properties` | 禁止 JSON Schema 中未定义的额外属性 |
| `whitespace_pattern` | 自定义空白匹配模式 |

### 例 1：choice

```python
params = SamplingParams(
    max_tokens=1,
    structured_outputs=StructuredOutputsParams(
        choice=["A", "B", "C", "D"]
    ),
)
```

### 例 2：regex

```python
params = SamplingParams(
    max_tokens=64,
    structured_outputs=StructuredOutputsParams(
        regex=r"\d{4}-\d{2}-\d{2}"
    ),
)
```

### 例 3：grammar

```python
sql_grammar = """
root ::= select_statement
select_statement ::= "SELECT " column " FROM " table
column ::= "name" | "email"
table ::= "users" | "accounts"
"""

params = SamplingParams(
    max_tokens=64,
    structured_outputs=StructuredOutputsParams(grammar=sql_grammar),
)
```

---

## 17.4 当前源码里它是怎么工作的？

### 第一步：请求先带着结构化约束进入系统

当 `SamplingParams.structured_outputs` 不为空时：

- `Request` 会构造 `StructuredOutputRequest`
- 请求初始状态可能被设成 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`

关键文件：

- `vllm/vllm/v1/request.py`
- `vllm/vllm/v1/structured_output/request.py`

### 第二步：`StructuredOutputManager` 选择后端并编译 grammar

核心类在：

- `vllm/vllm/v1/structured_output/__init__.py`

它在第一次遇到结构化请求时，会：

1. 读取 `structured_outputs_config.backend`
2. 选择具体 backend
3. 把请求的 schema / regex / grammar 编译成内部 grammar 对象

源码里能看到这些后端：

- `XgrammarBackend`
- `GuidanceBackend`
- `OutlinesBackend`
- `LMFormatEnforcerBackend`

### 第三步：采样前生成 token bitmask

真正限制 token 可选集合的关键机制是：

- grammar 状态机
- token bitmask

`StructuredOutputManager.grammar_bitmask(...)` 会为 batch 中的结构化请求生成位掩码，告诉 sampler：

- 哪些 token 合法
- 哪些 token 必须设成不可选

你可以把它粗略理解成：

```text
原始 logits
  ↓
根据 grammar 生成合法 token mask
  ↓
把不合法 token 置为 -inf
  ↓
再做 softmax / top-k / top-p / sample
```

### 第四步：如果开了 speculative decoding，还要为多个位置准备 mask

源码里还有一个很容易被忽略的细节：

- 如果启用了 speculative decoding
- 结构化输出不能只准备“当前 token”一个 mask
- 还要为 speculative positions / bonus token 准备额外 bitmask

这就是 `grammar_bitmask(...)` 里为什么会根据 `num_speculative_tokens` 预留更大张量。

---

## 17.5 结构化输出和调度器怎么配合？

结构化输出并不是“模型前向结束后的后处理”，它会影响请求生命周期。

### 影响 1：请求可能先卡在 grammar 编译阶段

如果 grammar 编译是异步的，请求会先处于：

- `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`

调度器遍历 waiting 队列时，会先尝试把它 promote 回普通 `WAITING`，只有 grammar ready 后才真正接纳。

相关逻辑在：

- `vllm/vllm/v1/core/sched/scheduler.py`

### 影响 2：每轮采样都要额外生成 mask

这会带来：

- CPU 侧 grammar 状态推进
- backend 侧 mask 构造
- 更大的 host 参与度

所以结构化输出的开销，本质上来自“每个 token 的合法集合维护”，而不只是一个一次性的 schema 解析。

---

## 17.6 后端选择：源码里有哪些真实差异？

### 默认模式：`auto`

当前服务端支持：

```bash
vllm serve model --structured-outputs-config.backend auto
```

`auto` 会根据请求形态和 backend 能力选择具体实现。

### 你需要知道的工程差异

| 后端 | 典型特点 |
|------|----------|
| `xgrammar` | 当前常用主力后端之一，速度与能力较平衡 |
| `guidance` | 支持较强的 grammar 约束能力 |
| `outlines` | 生态常见，适合部分 schema/regex 场景 |
| `lm-format-enforcer` | 另一个兼容型后端 |

不同后端支持的：

- regex 语法
- JSON Schema 细节
- 性能表现

并不完全一致，所以生产上最好固定一个你验证过的 backend，而不是只依赖默认行为。

---

## 17.7 性能影响该怎么理解？

### 开销来自哪里？

结构化输出的额外成本通常来自：

1. grammar 编译
2. 每轮 bitmask 生成
3. grammar 状态推进
4. speculative decode 场景下的多位置 mask

### 什么时候开销最明显？

- batch 很大
- 输出 token 很长
- grammar 很复杂
- 同时还开了 speculative decoding

### 为什么仍然值得？

因为它往往能替代：

- 无数次“解析失败后重试”
- 后处理修补 JSON
- 应用端复杂的兜底逻辑

所以很多业务里，结构化输出虽然略增 TPOT，却能显著降低整体系统复杂度。

---

## 17.8 源码对照：本章该看哪些文件？

| 主题 | 关键文件 | 重点 |
|------|----------|------|
| 参数定义 | `vllm/vllm/sampling_params.py` | `StructuredOutputsParams` |
| OpenAI 协议映射 | `vllm/vllm/entrypoints/openai/chat_completion/` | `response_format -> structured_outputs` |
| 请求对象 | `vllm/vllm/v1/structured_output/request.py` | grammar future、key 生成 |
| 引擎级管理器 | `vllm/vllm/v1/structured_output/__init__.py` | backend 选择、grammar 编译、bitmask |
| Worker 侧应用 | `vllm/vllm/v1/worker/gpu/structured_outputs.py` | bitmask 在 GPU 侧的实际应用 |
| 调度器协同 | `vllm/vllm/v1/core/sched/scheduler.py` | `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` |
| 官方特性文档 | `vllm/docs/features/structured_outputs.md` | 当前支持能力与示例 |

---

## 本章小结

| 概念 | 当前仓库中的真实语义 |
|------|----------------------|
| 主接口 | `structured_outputs` / `response_format` |
| 旧接口 | `guided_*` 已不应作为主线教程写法 |
| 核心管理器 | `StructuredOutputManager` |
| 运行机制 | grammar 编译 + token bitmask |
| 与调度关系 | grammar 未准备好时，请求会暂时卡在等待态 |

---

## 练习题

### 基础题

1. 为什么说结构化输出的核心不是 prompt，而是 token mask？
2. `response_format` 和 `structured_outputs` 在当前仓库里是什么关系？
3. 为什么结构化输出请求会进入 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`？

### 实践题

4. 在 `sampling_params.py` 中找到 `StructuredOutputsParams`，列出它支持的所有互斥约束类型。
5. 在 `v1/structured_output/__init__.py` 中找到 backend 初始化逻辑，确认当前仓库支持哪些 backend。

### 思考题

6. 如果你同时打开 structured outputs 和 speculative decoding，为什么 bitmask 逻辑会变复杂？
7. 在你的业务里，如果模型输出 JSON 经常错，结构化输出带来的额外 TPOT 是否值得？
