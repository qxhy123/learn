# 第15章：投机解码

> 投机解码的核心思想出奇地简单：用一个小模型猜，用大模型验证。猜对了就赚到了，猜错了也不亏——因为验证可以并行完成。

---

## 学习目标

学完本章，你将能够：

1. 理解投机解码的核心思想和数学保证
2. 掌握 draft model 和 target model 的协作机制
3. 在 vLLM 中配置和使用投机解码
4. 分析投机解码在不同场景下的加速效果
5. 选择合适的草稿模型和投机长度

---

## 15.1 为什么需要投机解码？

### Decode 阶段的瓶颈回顾

Decode 阶段每步只生成 1 个 token，但需要读取全部模型权重。GPU 的计算单元大量空闲。

```
标准 decode: 每步 1 个 token
  Step 1: 读权重(14GB) → 算 1 个 token → 写回
  Step 2: 读权重(14GB) → 算 1 个 token → 写回
  Step 3: 读权重(14GB) → 算 1 个 token → 写回
  ...
  
  10 个 token 需要 10 次完整的权重读取
```

**核心问题**：能不能每次读取权重时，多产出几个 token？

### 投机解码的直觉

```
投机解码: 每步可能产出多个 token

  1. 小模型快速"猜"出 k 个 token
  2. 大模型一次性验证这 k 个 token（只需 1 次前向计算）
  3. 接受连续正确的部分，拒绝第一个错误的

  如果 5 个全猜对: 1 次大模型前向 → 5 个 token ✓
  如果前 3 个对:   1 次大模型前向 → 3 个 token ✓ + 1 个修正 token
```

---

## 15.2 工作原理

### Draft-Then-Verify

投机解码由两个模型协作：

- **Draft Model（草稿模型）**：小而快的模型，用于生成候选 token
- **Target Model（目标模型）**：大而准的模型，用于验证候选 token

### 一次投机迭代的流程

```
1. Draft Phase（草稿阶段）:
   小模型连续生成 k 个 token: [d₁, d₂, d₃, d₄, d₅]
   
   耗时: 很短（小模型速度快）

2. Verify Phase（验证阶段）:
   大模型一次性处理 [original_tokens, d₁, d₂, d₃, d₄, d₅]
   同时计算每个位置的概率分布
   
   耗时: ≈ 1 次标准 decode（因为可以并行计算）

3. Accept/Reject（接受/拒绝）:
   逐个检查: 大模型是否同意小模型的选择？
   
   位置 1: P_target(d₁) 足够高 → 接受 ✓
   位置 2: P_target(d₂) 足够高 → 接受 ✓
   位置 3: P_target(d₃) 足够高 → 接受 ✓
   位置 4: P_target(d₄) 太低   → 拒绝 ✗ → 用大模型重新采样
   位置 5: 不再检查（前面已拒绝）
   
   结果: 接受 3 个 token + 1 个修正 = 4 个 token
```

### 数学保证

投机解码有一个重要的理论保证：**无论草稿模型质量如何，最终输出的分布与只用目标模型完全一致。**

接受概率的计算：

```
对于草稿模型生成的 token x:
  如果 p_target(x) >= p_draft(x):  直接接受
  否则: 以概率 p_target(x) / p_draft(x) 接受
```

这意味着投机解码**不影响输出质量**，只影响生成速度。

---

## 15.3 在 vLLM 中使用投机解码

### 使用独立的 Draft 模型

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.1-8B-Instruct",
    num_speculative_tokens=5,    # 每次投机的 token 数
)

params = SamplingParams(temperature=0, max_tokens=200)
outputs = llm.generate(["Explain quantum computing."], params)
```

### 使用 ngram 投机（无需额外模型）

```python
# ngram 投机：基于输入中的 n-gram 模式预测
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_model="[ngram]",
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,
)
```

ngram 投机适合 prompt 中包含大量可复用模式的场景（如翻译、改写）。

### API 服务器配置

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.1-8B-Instruct \
    --num-speculative-tokens 5 \
    --tensor-parallel-size 4
```

---

## 15.4 加速效果分析

### 加速倍数

加速倍数取决于**接受率（acceptance rate）**：

```
加速倍数 ≈ 平均每次接受的 token 数 / (draft 时间 + verify 时间 / 标准 decode 时间)
```

典型的加速效果：

| 场景 | 接受率 | 加速倍数 |
|------|--------|---------|
| 同系列小-大模型（Llama 8B→70B） | 70-85% | 1.5-2× |
| 翻译/改写（输入与输出相似） | 80-90% | 2-3× |
| 创意生成（输出不可预测） | 40-60% | 1.0-1.3× |
| ngram（重复模式多） | 取决于内容 | 1.0-2× |

### 什么时候投机解码最有效？

```
✓ 大模型远大于小模型（如 70B vs 8B）
✓ 任务输出相对可预测（翻译、摘要、格式化）
✓ 延迟敏感的场景（实时交互）
✓ 大模型是 memory-bound 的（batch size 较小）

✗ 大模型不够大（7B vs 3B，加速有限）
✗ 创意性生成（接受率低）
✗ 已经是高 batch 场景（GPU 已经利用率高）
✗ 显存紧张（需要额外空间放小模型）
```

### num_speculative_tokens 的选择

```
太少 (1-2): 开销不够分摊，加速有限
太多 (10+): 后面的 token 接受率低，浪费小模型计算
推荐: 3-7，视接受率调整

经验法则: 如果接受率 > 80%，可以增加投机长度
          如果接受率 < 50%，应该减少投机长度
```

---

## 15.5 Draft 模型选择

### 选择原则

1. **同系列更好**：Llama-8B 作为 Llama-70B 的 draft，比随机小模型效果好
2. **足够小**：draft 模型应该比 target 快很多（至少 5-10×）
3. **词表一致**：draft 和 target 必须使用相同的 tokenizer/词表
4. **领域匹配**：fine-tuned target 最好配同领域的 draft

### 常见 Draft-Target 配对

| Target Model | Draft Model | 建议 |
|-------------|-------------|------|
| Llama-3.1-70B | Llama-3.1-8B | 推荐 |
| Qwen2.5-72B | Qwen2.5-1.5B | 推荐 |
| Mistral Large | Mistral 7B | 可用 |

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 核心思想 | 小模型快速猜测，大模型并行验证 |
| 数学保证 | 输出分布与只用大模型完全一致 |
| 典型加速 | 1.5-2× 延迟改善 |
| 适用场景 | 延迟敏感、大模型、可预测输出 |
| 不适用场景 | 高 batch、显存紧张、创意生成 |

---

## 动手实验

### 实验 1：投机解码加速测试

对比有无投机解码时的 TPOT 和端到端延迟。

### 实验 2：接受率观察

用不同类型的 prompt（翻译、创意写作、代码生成）测试接受率差异。

---

## 练习题

### 基础题

1. 投机解码为什么不影响输出质量？
2. 什么情况下投机解码的加速效果最好？
3. draft model 和 target model 的 tokenizer 为什么必须一致？

### 思考题

4. 如果所有投机的 token 都被拒绝，投机解码比标准 decode 更慢还是一样快？
5. 投机解码与连续批处理如何配合？高 batch 时投机解码还有意义吗？
