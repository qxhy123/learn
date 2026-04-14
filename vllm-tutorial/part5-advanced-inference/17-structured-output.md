# 第17章：结构化输出

> 让 LLM 输出合法的 JSON 不应该靠"请一定要输出 JSON 哦"。结构化输出通过在解码时直接约束 token 选择，从根本上保证输出符合指定格式。

---

## 学习目标

学完本章，你将能够：

1. 理解 guided decoding 的原理
2. 使用 JSON Schema 约束 vLLM 的输出格式
3. 使用正则表达式引导生成
4. 理解结构化输出对性能的影响
5. 在实际应用中选择合适的约束方式

---

## 17.1 为什么需要结构化输出？

### Prompt 约束的局限

```python
# 不可靠的方式：靠 prompt 约束
prompt = """请以 JSON 格式回答，格式如下：
{"name": "...", "age": ..., "city": "..."}
"""
# 模型可能输出：
# 1. 正确 JSON ✓
# 2. JSON 前加了解释文字 ✗
# 3. 漏掉引号或逗号 ✗
# 4. 添加了额外字段 ✗
```

### Guided Decoding 的保证

```python
# 可靠的方式：guided decoding
# 在每一步 token 采样时，只允许选择符合格式的 token
# 100% 保证输出是合法的 JSON
```

---

## 17.2 JSON Schema 约束

### 基本使用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "介绍一下北京。"},
    ],
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
                    "landmarks": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "description": {"type": "string"},
                },
                "required": ["name", "country", "population", "landmarks"],
            },
        },
    },
    max_tokens=300,
)

import json
result = json.loads(response.choices[0].message.content)
print(json.dumps(result, ensure_ascii=False, indent=2))
# 保证是合法 JSON，且符合指定 schema
```

### 简单 JSON 模式

```python
# 只约束输出为合法 JSON（不指定 schema）
response = client.chat.completions.create(
    model="model-name",
    messages=[...],
    response_format={"type": "json_object"},
    max_tokens=300,
)
```

### 离线推理中使用

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

params = SamplingParams(
    temperature=0.7,
    max_tokens=300,
)

# 使用 guided_decoding 参数
from vllm.sampling_params import GuidedDecodingParams

guided_params = GuidedDecodingParams(
    json_schema={
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["answer", "confidence"],
    }
)

params = SamplingParams(
    temperature=0.7,
    max_tokens=300,
    guided_decoding=guided_params,
)

outputs = llm.generate(["What is the capital of France?"], params)
```

---

## 17.3 正则表达式约束

### 使用正则引导

```python
# 约束输出为邮箱格式
guided_params = GuidedDecodingParams(
    regex=r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
)

# 约束输出为日期格式
guided_params = GuidedDecodingParams(
    regex=r"\d{4}-\d{2}-\d{2}"
)

# 约束输出为选项
guided_params = GuidedDecodingParams(
    choice=["positive", "negative", "neutral"]
)
```

### 选择约束

```python
# 限制输出为指定选项之一
guided_params = GuidedDecodingParams(
    choice=["A", "B", "C", "D"]
)

params = SamplingParams(max_tokens=1, guided_decoding=guided_params)
outputs = llm.generate(["The answer is:"], params)
# 输出一定是 A、B、C 或 D
```

---

## 17.4 工作原理

### Token Masking

Guided decoding 在每一步采样时，通过 token masking 实现约束：

```
正常采样:
  logits: [tok_a: 5.2, tok_b: 3.1, tok_c: 4.8, tok_d: 2.0, ...]
  所有 token 都可以被选择

Guided 采样 (约束为 JSON，当前期望 "{""):
  logits: [tok_a: -inf, tok_b: -inf, "{": 4.8, tok_d: -inf, ...]
  只有 "{" 可以被选择
  
更灵活的场景 (JSON 字符串值中):
  logits: [tok_a: 5.2, tok_b: 3.1, "\"": 4.8, tok_d: 2.0, ...]
  所有文本 token 和引号都可以被选择
```

### 状态机驱动

vLLM 使用有限状态自动机（FSM）或上下文无关文法（CFG）来追踪当前的生成状态：

```
JSON 生成的状态示例:

START → 期望 "{" 
  → OBJECT_KEY → 期望 引号开始
    → STRING → 任意字符或引号结束
      → COLON → 期望 ":"
        → VALUE → 数字/字符串/布尔/数组/对象
          → COMMA_OR_END → "," 或 "}"
```

---

## 17.5 性能影响

### 额外开销

```
结构化输出的性能开销:
  - Token masking: 每步需要计算合法 token 集合
  - 状态转换: 维护 FSM/CFG 状态
  - 典型开销: 5-15% TPOT 增加
  
但节省的成本:
  - 不需要输出后重试（失败的 JSON 解析）
  - 不需要后处理修复
  - 应用端代码更简单
```

### 优化

vLLM 使用 outlines 或 xgrammar 作为结构化输出的后端，这些库已经做了大量优化：

- 预编译 FSM
- 向量化的 mask 计算
- 缓存状态转换

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 核心原理 | 每步采样时 mask 不合法的 token |
| JSON Schema | 保证输出严格符合指定结构 |
| 正则约束 | 输出匹配指定正则表达式 |
| 选择约束 | 限制输出为给定选项之一 |
| 性能开销 | 5-15% TPOT 增加，但省去重试成本 |

---

## 动手实验

### 实验 1：JSON Schema 输出

定义一个复杂的 JSON Schema（嵌套对象、数组），让 vLLM 生成符合 schema 的输出。

### 实验 2：分类任务

用 choice 约束实现一个情感分类任务，保证输出一定是预定义的类别之一。

---

## 练习题

### 基础题

1. Guided decoding 如何保证输出一定是合法 JSON？
2. 正则约束和 JSON Schema 约束有什么区别？

### 思考题

3. 结构化输出会影响生成质量吗？为什么？
4. 在什么场景下，结构化输出的价值最大？
