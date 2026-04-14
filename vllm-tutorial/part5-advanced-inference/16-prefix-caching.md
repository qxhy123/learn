# 第16章：前缀缓存与 Prompt 复用

> 如果 1000 个请求共享同一个系统 prompt，为什么要计算 1000 次？前缀缓存的答案很简单：只计算一次。

---

## 学习目标

学完本章，你将能够：

1. 理解前缀缓存的工作原理和适用场景
2. 在 vLLM 中启用和配置前缀缓存
3. 分析前缀缓存对 TTFT 和吞吐的影响
4. 掌握 hash 匹配机制和缓存命中条件
5. 识别最能从前缀缓存获益的应用场景

---

## 16.1 问题：重复的 Prefill 计算

### 场景分析

在实际应用中，大量请求共享相同的前缀：

```
请求 A: [系统prompt(500 tok)] + [用户A的问题(20 tok)]
请求 B: [系统prompt(500 tok)] + [用户B的问题(30 tok)]
请求 C: [系统prompt(500 tok)] + [用户C的问题(15 tok)]
...
请求 N: [系统prompt(500 tok)] + [用户N的问题(25 tok)]
```

每个请求的 prefill 都要重新计算 500 个 token 的系统 prompt，但它们的 KV Cache 完全相同。

### 浪费有多大？

假设系统 prompt 长度 = 1000 token，用户消息平均 50 token：

```
没有前缀缓存:
  每个请求的 prefill 计算: 1000 + 50 = 1050 tokens
  1000 个请求总计: 1,050,000 tokens 的 prefill 计算

有前缀缓存:
  第 1 个请求: 1050 tokens (完整计算)
  后续 999 个请求: 50 tokens 的 prefill + 缓存命中的 1000 tokens
  总计: 1050 + 999 × 50 = 51,000 tokens 的 prefill 计算
  
  节省: 95% 的 prefill 计算
```

---

## 16.2 前缀缓存的工作原理

### Hash 匹配

vLLM 的前缀缓存通过 token 内容的 hash 来匹配：

```
1. 将 prompt 的 token 序列按 block_size 分块
2. 计算每个块的 hash 值
3. 检查 hash 是否已在缓存中存在
4. 命中 → 直接复用 KV Cache 块
5. 未命中 → 正常计算并存入缓存

块 hash 示例:
  [tok₁,...,tok₁₆]  → hash: 0x3a7f → 命中! 复用物理块 23
  [tok₁₇,...,tok₃₂] → hash: 0x8b12 → 命中! 复用物理块 45
  [tok₃₃,...,tok₄₈] → hash: 0xc4e9 → 未命中 → 计算并缓存
```

### 与 PagedAttention 的配合

前缀缓存天然与 PagedAttention 的块管理配合：

```
请求 A 的块表: [块23, 块45, 块67, 块89]  (前 2 个是共享前缀)
请求 B 的块表: [块23, 块45, 块12, 块34]  (前 2 个复用)
请求 C 的块表: [块23, 块45, 块56]        (前 2 个复用)
                ↑      ↑
            共享的物理块 (ref_count = 3)
```

### 缓存命中条件

前缀缓存命中需要满足：

1. **token 序列完全匹配**（逐 token 对比 hash）
2. **必须从序列开头连续匹配**（中间不能跳过）
3. **以 block 为粒度**（部分 block 不算命中）

```
可以命中:
  请求 A: [系统prompt] [用户A]  → 缓存
  请求 B: [系统prompt] [用户B]  → 命中系统prompt 部分 ✓

不能命中:
  请求 A: [prompt_X] [中间内容] [prompt_Y]
  请求 B: [prompt_Z] [中间内容] [prompt_Y]
  → prompt_Y 虽然相同，但不在开头，不是连续前缀 ✗
```

---

## 16.3 使用前缀缓存

### 启用前缀缓存

```python
# 离线推理
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enable_prefix_caching=True,  # 启用前缀缓存
)

# 第一个请求（完整 prefill）
outputs1 = llm.generate(
    ["System: You are helpful.\nUser: Hello!"],
    SamplingParams(max_tokens=100),
)

# 第二个请求（前缀命中，只需 prefill 差异部分）
outputs2 = llm.generate(
    ["System: You are helpful.\nUser: What is AI?"],
    SamplingParams(max_tokens=100),
)
```

### API 服务器

```bash
vllm serve model --enable-prefix-caching
```

前缀缓存对 API 调用是透明的——用户不需要做任何修改。

---

## 16.4 性能影响

### TTFT 改善

前缀缓存最直接的效果是减少 TTFT：

```
无前缀缓存:
  TTFT = prefill(系统prompt + 用户输入) = prefill(1000 + 50) ≈ 50ms

有前缀缓存（命中）:
  TTFT = prefill(仅用户输入) = prefill(50) ≈ 5ms

改善: 10× TTFT 降低
```

### 吞吐提升

前缀缓存也间接提升吞吐：
- prefill 计算减少 → GPU 有更多时间做 decode
- KV Cache 共享 → 显存效率提升 → 更多并发

### 缓存命中率

```
命中率取决于:
  ✓ 系统 prompt 比例高 → 高命中率
  ✓ 请求到达模式集中 → 高命中率（缓存还没被逐出）
  ✗ 每个请求完全不同 → 几乎不命中
  ✗ 请求间隔太长 → 缓存已被逐出
```

---

## 16.5 最佳应用场景

### 最适合前缀缓存的场景

| 场景 | 共享前缀比例 | 预期收益 |
|------|------------|---------|
| 聊天机器人（相同系统 prompt） | 80-95% | 高 |
| RAG（相同上下文文档） | 50-80% | 中高 |
| 代码补全（相同文件上下文） | 60-90% | 高 |
| 多轮对话（共享历史） | 30-70% | 中 |
| Few-shot 推理（相同示例） | 70-90% | 高 |
| 独立短问答 | 0-10% | 低 |

### 设计 Prompt 以最大化前缀缓存

```python
# ✓ 好的做法：系统 prompt 在前，用户输入在后
prompt = f"{SYSTEM_PROMPT}\nUser: {user_message}"

# ✗ 不好的做法：每次都在前面插入变化内容
prompt = f"Time: {current_time}\n{SYSTEM_PROMPT}\nUser: {user_message}"
# current_time 变化导致整个前缀无法命中
```

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 核心思想 | 相同前缀的 KV Cache 只计算一次 |
| 匹配方式 | Hash 匹配，block 粒度 |
| TTFT 改善 | 命中时可降低 5-10× |
| 最佳场景 | 长系统 prompt + 短用户输入 |
| 使用方式 | `--enable-prefix-caching`，对用户透明 |

---

## 练习题

### 基础题

1. 前缀缓存的 hash 匹配以什么为粒度？
2. 为什么前缀缓存必须从序列开头连续匹配？

### 思考题

3. 如何设计 prompt 结构以最大化前缀缓存的命中率？
4. 前缀缓存与 Copy-on-Write 有什么关系？
