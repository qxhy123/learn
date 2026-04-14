# 第18章：采样参数与解码策略

> 同一个模型，不同的采样参数可以产出完全不同的结果——从死板的复读机到天马行空的诗人。理解采样参数是用好 LLM 的基本功。

---

## 学习目标

学完本章，你将能够：

1. 深入理解 temperature、top-p、top-k 的数学原理和效果
2. 掌握 beam search 与采样解码的区别和适用场景
3. 理解重复惩罚、频率惩罚、存在惩罚的作用
4. 根据不同应用场景选择最优的采样配置
5. 诊断和解决常见的生成质量问题

---

## 18.1 从 Logits 到 Token

### 基本流程

```
模型输出 logits → 变换/过滤 → 概率分布 → 采样 → token

1. logits: [2.1, 0.5, 3.8, -1.2, 1.7, ...]  (词表大小维度)
2. temperature 缩放: logits / T
3. top-k 过滤: 只保留最大的 k 个
4. top-p 过滤: 只保留累积概率达到 p 的
5. softmax: 转为概率分布
6. 采样: 从概率分布中选择一个 token
```

---

## 18.2 Temperature

### 原理

Temperature 通过缩放 logits 来控制概率分布的"尖锐度"：

```python
# 概率计算: softmax(logits / temperature)

# temperature = 1.0 (默认): 使用原始分布
# temperature < 1.0: 分布更集中 → 更确定
# temperature > 1.0: 分布更均匀 → 更随机
# temperature → 0:   等价于贪心解码 (argmax)
```

### 直觉理解

```
logits = [5.0, 3.0, 2.0, 1.0, 0.5]

T=0.1 (几乎确定):  [0.99, 0.01, 0.00, 0.00, 0.00]  → 几乎总是选第一个
T=0.5 (较确定):    [0.85, 0.10, 0.03, 0.01, 0.01]  → 偶尔会选其他
T=1.0 (标准):      [0.55, 0.15, 0.10, 0.08, 0.06]  → 有一定多样性
T=2.0 (很随机):    [0.35, 0.20, 0.17, 0.15, 0.13]  → 相当随机
```

### 应用场景

| 场景 | 推荐 Temperature | 原因 |
|------|-----------------|------|
| 事实性问答 | 0 | 要求准确、一致 |
| 代码生成 | 0 - 0.3 | 代码需要正确性 |
| 通用对话 | 0.5 - 0.8 | 自然但不失控 |
| 创意写作 | 0.8 - 1.2 | 需要多样性 |
| 头脑风暴 | 1.0 - 1.5 | 鼓励发散思维 |

---

## 18.3 Top-p（Nucleus Sampling）

### 原理

Top-p 动态截断概率分布：只保留累积概率达到 p 的最小 token 集合。

```
概率分布: [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]

top_p = 0.9:
  累积: 0.40 → 0.65 → 0.80 → 0.90 ← 到这里停止
  保留: [0.40, 0.25, 0.15, 0.10]  (4 个 token)
  重新归一化: [0.44, 0.28, 0.17, 0.11]

top_p = 0.5:
  累积: 0.40 → 0.65 ← 到这里停止
  保留: [0.40, 0.25]  (2 个 token)
```

### 与 top-k 的区别

```
top_k = 5: 固定保留 5 个候选 (不管概率分布如何)
top_p = 0.9: 动态保留，概率集中时少，分散时多

高置信场景: logits = [10, 2, 1, 0, -1, ...]
  top_k=5: 保留 5 个 (但后 4 个概率极低，无意义)
  top_p=0.9: 可能只保留 1-2 个 (更合理)

低置信场景: logits = [3, 2.8, 2.5, 2.3, 2.0, ...]
  top_k=5: 只保留 5 个 (可能过少)
  top_p=0.9: 可能保留 10+ 个 (更灵活)
```

---

## 18.4 重复惩罚

### 三种惩罚机制

**repetition_penalty**（重复惩罚）：

```python
# 对已出现的 token，将其 logit 除以 penalty 值
# repetition_penalty > 1: 减少重复
# repetition_penalty = 1: 不惩罚
# 推荐范围: 1.0 - 1.3

params = SamplingParams(repetition_penalty=1.1)
```

**frequency_penalty**（频率惩罚）：

```python
# logit -= frequency_penalty * token 出现次数
# 出现次数越多，惩罚越大
# 推荐范围: 0.0 - 1.0

params = SamplingParams(frequency_penalty=0.5)
```

**presence_penalty**（存在惩罚）：

```python
# logit -= presence_penalty * (1 if token 出现过 else 0)
# 只要出现过就惩罚，不区分出现几次
# 推荐范围: 0.0 - 1.0

params = SamplingParams(presence_penalty=0.5)
```

### 选择哪种惩罚？

| 问题 | 推荐方案 |
|------|---------|
| 模型反复输出相同的句子 | repetition_penalty=1.1 |
| 模型过度使用某些词 | frequency_penalty=0.3 |
| 想鼓励模型使用新词 | presence_penalty=0.3 |
| 正常情况 | 都设为默认值 (不惩罚) |

---

## 18.5 Beam Search

### 与采样的区别

```
采样解码 (Sampling):
  每步随机选一个 token → 一条路径 → 一个结果
  优点: 多样性好
  缺点: 不保证全局最优

Beam Search:
  每步保留 top-b 条路径 → b 条路径 → 选最好的
  优点: 更可能找到高概率序列
  缺点: 多样性差, 更慢, 显存占用更大
```

### 在 vLLM 中使用 Beam Search

```python
params = SamplingParams(
    use_beam_search=True,
    best_of=4,        # beam width
    temperature=0,    # beam search 通常配合 T=0
    max_tokens=200,
)
```

### 使用建议

- **通常不推荐 beam search**：现代 LLM 在采样模式下表现更好
- 适合少数场景：机器翻译、摘要等需要"最可能"输出的任务
- 采样 + 合理参数在大多数场景下效果更好

---

## 18.6 常见配置模板

### 精确问答

```python
params = SamplingParams(
    temperature=0,        # 贪心解码
    max_tokens=500,
)
```

### 自然对话

```python
params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.05,
    max_tokens=500,
)
```

### 创意写作

```python
params = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    presence_penalty=0.3,
    max_tokens=1000,
)
```

### 代码生成

```python
params = SamplingParams(
    temperature=0.2,
    top_p=0.95,
    max_tokens=1000,
    stop=["\n\n\n", "```"],
)
```

### 分类/选择题

```python
params = SamplingParams(
    temperature=0,
    max_tokens=1,
)
```

---

## 本章小结

| 参数 | 作用 | 推荐范围 |
|------|------|---------|
| temperature | 控制随机性 | 0 - 1.5 |
| top_p | 动态截断概率分布 | 0.8 - 1.0 |
| top_k | 固定截断候选数 | 20 - 100 |
| repetition_penalty | 减少重复 | 1.0 - 1.3 |
| frequency_penalty | 惩罚高频词 | 0.0 - 1.0 |
| presence_penalty | 鼓励新词 | 0.0 - 1.0 |
| max_tokens | 最大输出长度 | 视任务而定 |

---

## 动手实验

### 实验 1：Temperature 对比

同一个 prompt，用 T=0, 0.5, 1.0, 1.5 各生成 5 次，对比输出的多样性和质量。

### 实验 2：惩罚参数效果

找一个容易产生重复的 prompt，逐步增加 repetition_penalty 观察效果。

---

## 练习题

### 基础题

1. temperature=0 等价于什么解码策略？
2. top_p=0.9 意味着什么？
3. frequency_penalty 和 presence_penalty 的区别是什么？

### 思考题

4. 为什么 top_p 比 top_k 更灵活？给出一个例子。
5. 对于一个 RAG（检索增强生成）应用，你会选择什么采样参数？为什么？
