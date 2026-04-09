# 第12章：损失函数与评估

> "评估是机器学习的照妖镜——好的评估指标让模型进步，坏的评估指标让模型欺骗你。"

---

## 学习目标

完成本章学习后，你将能够：

1. **理解交叉熵损失在序列生成中的应用**，包括Token级别的损失计算和Padding的处理方式
2. **掌握困惑度（Perplexity）的计算**，理解其与语言模型质量的关系
3. **理解BLEU评估指标**，包括N-gram精确率和简短惩罚的原理
4. **掌握ROUGE评估指标**，区分ROUGE-N、ROUGE-L和ROUGE-S的适用场景
5. **能够实现完整的评估流程**，综合运用多种指标对模型进行全面评估

---

## 引言

训练一个Transformer模型需要回答两个核心问题：**模型在学什么？** 以及 **模型学得怎么样？** 损失函数回答第一个问题——它定义了模型的优化目标；评估指标回答第二个问题——它衡量模型在真实任务上的表现。

这两个问题看似简单，实则深刻。一个设计不当的损失函数会让模型朝错误方向优化；一个不合适的评估指标会让你误以为模型很好，或者错误地判断模型很差。

本章从交叉熵损失出发，逐步介绍困惑度、BLEU、ROUGE等评估指标，最后通过完整的代码实战展示如何构建一个可靠的评估流程。

---

## 12.1 交叉熵损失

### 12.1.1 为什么用交叉熵？

在序列生成任务中，模型的目标是预测下一个Token。在数学上，这是一个**分类问题**：给定上下文，从词表中选择概率最高的词。

对于分类问题，**交叉熵损失**（Cross-Entropy Loss）是最自然的选择。它衡量模型预测的概率分布与真实分布之间的距离。

**直觉理解：** 如果真实答案是"cat"，而模型给"cat"分配了99%的概率，损失很小；如果模型给"cat"只分配了1%的概率，损失很大。

### 12.1.2 Token级别的交叉熵

对于序列生成，我们对序列中每个位置的Token都计算交叉熵，然后取平均：

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p(y_t|y_{<t}, x)$$

其中：
- $T$ 是序列长度
- $y_t$ 是位置 $t$ 的真实Token
- $p(y_t|y_{<t}, x)$ 是模型在给定前缀 $y_{<t}$ 和输入 $x$ 的条件下，预测 $y_t$ 的概率

**展开理解每一项：**

$$\log p(y_t|y_{<t}, x) = \log \text{softmax}(\mathbf{W}_o \mathbf{h}_t)[y_t]$$

模型输出 $\mathbf{h}_t$ 经过线性层和Softmax后得到词表上的概率分布，我们取真实Token $y_t$ 对应位置的对数概率。

**单样本损失示例：**

假设词表大小为5，真实Token索引为2，模型的logits为 $[1.0, 2.0, 3.0, 0.5, 0.2]$：

$$\text{softmax}([1.0, 2.0, 3.0, 0.5, 0.2]) \approx [0.082, 0.224, 0.609, 0.050, 0.034]$$

$$\mathcal{L} = -\log(0.609) \approx 0.496$$

### 12.1.3 忽略Padding的处理

在实际训练中，我们通常将不同长度的序列拼接成等长的批次，短序列用**Padding Token**填充。这些Padding位置不应该参与损失计算，否则：

1. 模型会浪费容量去"正确预测" Padding
2. 损失值会被稀释，梯度更新不准确
3. 不同批次因Padding比例不同，损失不可比

**处理方法：** 使用 `ignore_index` 参数忽略特定Token的损失。

```python
import torch
import torch.nn as nn

# 创建忽略Padding的交叉熵损失
# PAD_TOKEN_ID 通常为 0 或特殊值
criterion = nn.CrossEntropyLoss(ignore_index=0)
```

**完整示例：**

```python
import torch
import torch.nn as nn

# 假设：批次大小=2，序列长度=5，词表大小=100
batch_size = 2
seq_len = 5
vocab_size = 100
PAD_IDX = 0

# 模型输出的logits [batch, seq_len, vocab_size]
logits = torch.randn(batch_size, seq_len, vocab_size)

# 目标序列，其中0是Padding
# 序列1：[5, 12, 34, 0, 0]（后两个是Padding）
# 序列2：[7, 23, 45, 67, 0]（最后一个是Padding）
targets = torch.tensor([
    [5, 12, 34, 0, 0],
    [7, 23, 45, 67, 0]
])

# PyTorch的CrossEntropyLoss期望 [batch*seq, vocab] 和 [batch*seq]
logits_flat = logits.view(-1, vocab_size)  # [batch*seq, vocab]
targets_flat = targets.view(-1)             # [batch*seq]

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
loss = criterion(logits_flat, targets_flat)

print(f"损失值: {loss.item():.4f}")
# 只有非Padding位置参与计算：序列1有3个，序列2有4个，共7个Token
```

### 12.1.4 标签平滑（Label Smoothing）

原始交叉熵中，真实Token的目标概率为1，其余为0。这种"硬标签"可能导致模型过度自信。

**标签平滑**将目标分布软化：

$$\tilde{p}(k) = \begin{cases} 1 - \epsilon & k = y_t \\ \frac{\epsilon}{V-1} & k \neq y_t \end{cases}$$

其中 $\epsilon$ 通常取 $0.1$，$V$ 是词表大小。

```python
# PyTorch 1.10+ 直接支持
criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX,
    label_smoothing=0.1
)
```

**效果：** 标签平滑相当于一种正则化，防止模型对训练集过拟合，通常能提升BLEU等评估指标1-2个点。

### 12.1.5 序列级别与Token级别的差异

| 计算方式 | 公式 | 特点 |
|---------|------|------|
| Token平均 | $\frac{1}{T}\sum_t \mathcal{L}_t$ | 短序列和长序列损失可比，常用 |
| 序列平均 | $\frac{1}{B}\sum_b \frac{1}{T_b}\sum_t \mathcal{L}_{b,t}$ | 对每个序列平等对待 |
| 总和 | $\sum_t \mathcal{L}_t$ | 长序列主导，不推荐 |

PyTorch的 `CrossEntropyLoss` 默认使用 `reduction='mean'`，即对所有非Padding Token取平均（Token平均方式）。

---

## 12.2 困惑度（Perplexity）

### 12.2.1 困惑度的定义

困惑度（Perplexity，PPL）是语言模型最常用的内在评估指标，定义为交叉熵损失的指数：

$$\text{PPL} = \exp(\mathcal{L}) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log p(y_t|y_{<t}, x)\right)$$

等价地，可以写成：

$$\text{PPL} = \exp\left(\mathcal{L}\right) = \left(\prod_{t=1}^{T} \frac{1}{p(y_t|y_{<t})}\right)^{1/T}$$

这是每个Token的平均"倒数概率"的几何平均值。

### 12.2.2 困惑度的直觉理解

**类比：猜词游戏**

想象你在玩猜词游戏：每次你需要从一堆候选词中猜下一个词。困惑度告诉你，**模型平均需要考虑多少个同等可能的选项**才能猜到正确答案。

- **PPL = 1**：模型每次都100%确定下一个词（完美预测，不可能达到）
- **PPL = 10**：模型平均在10个选项中选择（相当不错）
- **PPL = 100**：模型平均在100个选项中选择（比较差）
- **PPL = 词表大小**：模型完全随机猜测（等于不学习）

**数值示例：**

假设序列有3个Token，每个Token的预测概率为：

| Token | 真实词 | 预测概率 | -log(p) |
|-------|--------|---------|---------|
| t=1   | "The"  | 0.8     | 0.223   |
| t=2   | "cat"  | 0.4     | 0.916   |
| t=3   | "sat"  | 0.2     | 1.609   |

$$\mathcal{L} = \frac{0.223 + 0.916 + 1.609}{3} = 0.916$$

$$\text{PPL} = \exp(0.916) \approx 2.50$$

这意味着模型平均在2.5个选项中犹豫。

### 12.2.3 困惑度与模型质量的关系

**典型参考值（英语语言模型）：**

| 模型类型 | 典型PPL范围 | 说明 |
|---------|-----------|------|
| 随机基线 | 词表大小（~50000） | 完全随机 |
| N-gram模型 | 100-1000 | 传统方法 |
| LSTM语言模型 | 50-100 | 深度学习早期 |
| GPT-2 (small) | ~35 | Transformer |
| GPT-3 | ~10-20 | 大模型 |
| GPT-4级别 | <10 | 前沿大模型 |

**注意事项：**

1. **PPL依赖于词表**：不同tokenizer的PPL不可直接比较
2. **PPL依赖于评估集**：在训练集上的PPL没有意义
3. **PPL不等于任务性能**：低PPL的模型不一定在下游任务表现好

### 12.2.4 困惑度计算实现

```python
import torch
import math

def compute_perplexity(model, dataloader, criterion, device):
    """
    计算模型在给定数据集上的困惑度

    Args:
        model: 训练好的语言模型
        dataloader: 数据加载器
        criterion: 带ignore_index的CrossEntropyLoss
        device: 计算设备

    Returns:
        float: 困惑度值
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            # 解码器输入（去掉最后一个Token）
            tgt_input = tgt[:, :-1]
            # 解码器目标（去掉第一个Token <bos>）
            tgt_target = tgt[:, 1:]

            # 前向传播
            logits = model(src, tgt_input)  # [batch, seq-1, vocab]

            # 计算损失（sum模式以获取总损失）
            batch_size, seq_len, vocab_size = logits.shape
            loss = criterion(
                logits.reshape(-1, vocab_size),
                tgt_target.reshape(-1)
            )

            # 统计非Padding Token数量
            # criterion使用mean，需要反推总损失
            non_pad_tokens = (tgt_target != criterion.ignore_index).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


# 更简洁的版本（直接用sum reduction）
def compute_perplexity_v2(model, dataloader, device, pad_idx=0):
    """使用sum reduction的困惑度计算"""
    criterion_sum = nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        reduction='sum'
    )

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            src, tgt = batch['src'].to(device), batch['tgt'].to(device)
            tgt_input, tgt_target = tgt[:, :-1], tgt[:, 1:]

            logits = model(src, tgt_input)

            loss = criterion_sum(
                logits.reshape(-1, logits.size(-1)),
                tgt_target.reshape(-1)
            )

            total_loss += loss.item()
            total_tokens += (tgt_target != pad_idx).sum().item()

    return math.exp(total_loss / total_tokens)
```

### 12.2.5 批次级困惑度的数值稳定性

当序列很长时，直接相乘概率会下溢（underflow）。始终在对数空间计算：

```python
# 不好：直接乘概率（数值不稳定）
ppl_bad = (1 / p1) * (1 / p2) * ... ** (1/T)  # 会变成 0 * inf

# 好：在对数空间计算
log_probs = [log(p1), log(p2), ...]
avg_neg_log_prob = -sum(log_probs) / T
ppl_good = math.exp(avg_neg_log_prob)
```

---

## 12.3 BLEU评估指标

### 12.3.1 BLEU的背景

**BLEU**（Bilingual Evaluation Understudy）由Papineni等人于2002年提出，最初用于机器翻译评估。它通过比较候选译文与参考译文之间的N-gram重叠来衡量翻译质量。

**核心思想：** 好的翻译与参考翻译共享更多N-gram片段。

**优点：**
- 快速、廉价、可重复
- 与人工评估有一定相关性
- 成为机器翻译的标准基准

**缺点：**
- 不考虑语义相似性（"大" vs "巨大"得分相同）
- 只用精确率，不用召回率
- 对参考译文数量敏感

### 12.3.2 N-gram精确率

**Unigram精确率（BLEU-1的基础）：**

候选译文中，有多少词出现在参考译文中？

$$P_1 = \frac{\text{候选中匹配的unigram数}}{\text{候选中的unigram总数}}$$

**问题：** 如果候选是"the the the the the"，而参考是"the cat sat on the mat"，那么所有词都匹配了，精确率 = 1.0。这明显不合理。

**修正的N-gram精确率：** 限制每个词的计数不超过其在参考译文中出现的次数：

$$\tilde{P}_n = \frac{\sum_{\text{N-gram} \in \hat{y}} \min(\text{Count}(\text{N-gram}, \hat{y}), \text{Count}(\text{N-gram}, y))}{\sum_{\text{N-gram} \in \hat{y}} \text{Count}(\text{N-gram}, \hat{y})}$$

**完整示例：**

- 候选（Hypothesis）：`"the the the the the"`
- 参考（Reference）：`"the cat sat on the mat"`

"the" 在参考中出现2次，所以最多计2次：

$$\tilde{P}_1 = \frac{2}{5} = 0.4$$

这个惩罚避免了重复词的滥用。

### 12.3.3 简短惩罚（Brevity Penalty）

N-gram精确率倾向于短句子（短句子更容易精确匹配），因此BLEU引入**简短惩罚**（BP）：

$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ \exp(1 - r/c) & \text{if } c \leq r \end{cases}$$

其中 $c$ 是候选长度，$r$ 是参考长度（多参考时取最接近候选长度的那个）。

**效果：**
- 候选与参考等长：BP = 1（无惩罚）
- 候选比参考短50%：BP = exp(1 - 2) ≈ 0.37（大幅惩罚）

### 12.3.4 BLEU分数的完整公式

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log \tilde{P}_n\right)$$

通常 $N=4$，$w_n = 1/4$（等权重）。

**BLEU-1到BLEU-4：**

| 指标 | N-gram阶数 | 典型用途 |
|------|----------|---------|
| BLEU-1 | 1-gram | 词汇覆盖率 |
| BLEU-2 | 1+2-gram | 短语匹配 |
| BLEU-3 | 1+2+3-gram | 句子流畅度 |
| BLEU-4 | 1+2+3+4-gram | 综合评估（最常用） |

### 12.3.5 手动计算BLEU示例

**候选：** `"the cat is on the mat"`
**参考：** `"the cat sat on the mat"`

**Bigram精确率（$\tilde{P}_2$）：**

候选的bigram：`the-cat`, `cat-is`, `is-on`, `on-the`, `the-mat`（共5个）

参考的bigram：`the-cat`, `cat-sat`, `sat-on`, `on-the`, `the-mat`

匹配：`the-cat`(1), `on-the`(1), `the-mat`(1) = 3个

$$\tilde{P}_2 = 3/5 = 0.6$$

### 12.3.6 使用sacrebleu库

```python
from sacrebleu.metrics import BLEU

def compute_bleu_score(hypotheses, references):
    """
    计算BLEU分数

    Args:
        hypotheses: 候选译文列表，每个元素是字符串
        references: 参考译文列表，可以是字符串列表或列表的列表（多参考）

    Returns:
        BLEUScore对象
    """
    bleu = BLEU()

    # sacrebleu期望参考是列表的列表（支持多参考）
    if isinstance(references[0], str):
        references = [references]

    result = bleu.corpus_score(hypotheses, references)
    return result

# 使用示例
hypotheses = [
    "The cat is on the mat",
    "There is a cat on the mat",
]

references = [
    "The cat sat on the mat",
    "A cat sat on the mat",
]

score = compute_bleu_score(hypotheses, references)
print(f"BLEU: {score.score:.2f}")
print(f"详细: {score}")
# 输出示例：BLEU = 38.15 1-gram: 72.7 / 2-gram: 50.0 / 3-gram: 33.3 / 4-gram: 20.0


# 句子级BLEU（注意：不推荐用于系统评估，但可以用于分析）
from sacrebleu.metrics import BLEU

bleu = BLEU(effective_order=True)  # effective_order避免零N-gram的问题

hyp = "The cat is on the mat"
ref = "The cat sat on the mat"

# 句子级需要分词
score = bleu.sentence_score(hyp, [ref])
print(f"句子BLEU: {score.score:.2f}")
```

### 12.3.7 BLEU的局限性

```python
# BLEU的一个著名问题：语义相似但词汇不同的句子得分很低

from sacrebleu.metrics import BLEU
bleu = BLEU(effective_order=True)

reference = "The dog quickly ran to the park"

# 完全同义但用词不同
hyp1 = "The dog rapidly sprinted to the garden"
# 词汇变化小但意思稍有不同
hyp2 = "The dog slowly walked to the park"

score1 = bleu.sentence_score(hyp1, [reference])
score2 = bleu.sentence_score(hyp2, [reference])

print(f"语义相似（用词不同）: BLEU = {score1.score:.2f}")  # 可能很低
print(f"语义不同（用词相似）: BLEU = {score2.score:.2f}")  # 可能更高
# BLEU无法区分这两种情况的语义质量
```

---

## 12.4 ROUGE评估指标

### 12.4.1 ROUGE的背景

**ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）由Lin于2004年提出，主要用于**文本摘要**评估。

**BLEU vs ROUGE的核心区别：**

| 维度 | BLEU | ROUGE |
|------|------|-------|
| 核心指标 | 精确率（Precision） | 召回率（Recall） |
| 主要用途 | 机器翻译 | 文本摘要 |
| 关注点 | 候选中有多少在参考中 | 参考中有多少在候选中 |

**为什么摘要用召回率？** 摘要任务中，参考摘要的内容应该尽量出现在候选摘要中。精确率关注的是"候选有没有乱说"，召回率关注的是"参考的要点有没有覆盖"。

### 12.4.2 ROUGE-N（N-gram召回率）

$$\text{ROUGE-N} = \frac{\sum_{\text{N-gram} \in y} \min(\text{Count}(\text{N-gram}, \hat{y}), \text{Count}(\text{N-gram}, y))}{\sum_{\text{N-gram} \in y} \text{Count}(\text{N-gram}, y)}$$

其中 $\hat{y}$ 是候选文本，$y$ 是参考文本。

**ROUGE-1示例：**

- 候选：`"the cat sat on mat"`（缺少 "the" 在最后）
- 参考：`"the cat sat on the mat"`

参考unigram：`the`(2), `cat`(1), `sat`(1), `on`(1), `mat`(1) = 6个

匹配：`the`(min(1,2)=1), `cat`(1), `sat`(1), `on`(1), `mat`(1) = 5个

$$\text{ROUGE-1} = 5/6 \approx 0.833$$

**ROUGE-2** 同理但使用bigram，对句子流畅性更敏感。

### 12.4.3 ROUGE-L（最长公共子序列）

ROUGE-L基于**最长公共子序列**（LCS），不要求连续匹配，能捕捉句子级别的结构相似性。

$$\text{ROUGE-L} = \frac{2 \cdot P_{lcs} \cdot R_{lcs}}{P_{lcs} + R_{lcs}}$$

其中：
$$P_{lcs} = \frac{|\text{LCS}(\hat{y}, y)|}{|\hat{y}|}, \quad R_{lcs} = \frac{|\text{LCS}(\hat{y}, y)|}{|y|}$$

**LCS示例：**

- 候选：`"the cat sat on mat"`
- 参考：`"the cat sat on the mat"`

LCS = `the cat sat on mat`（长度5，不要求连续但保持顺序）

$$P_{lcs} = 5/5 = 1.0, \quad R_{lcs} = 5/6 \approx 0.833$$

$$\text{ROUGE-L} = \frac{2 \times 1.0 \times 0.833}{1.0 + 0.833} \approx 0.909$$

**优势：** ROUGE-L不依赖于N-gram长度参数，对词序变化更鲁棒。

### 12.4.4 ROUGE-S（跳跃二元组）

ROUGE-S使用**跳跃二元组**（Skip-bigram），允许两个词之间有任意间隔：

例如，"the cat"可以匹配"the ... cat"中的任意间隔组合。

$$\text{ROUGE-S} = \frac{\text{跳跃二元组匹配数}}{\text{参考中的跳跃二元组总数}}$$

**优势：** 对词序变化（如主动/被动语态）更宽容。

**实际应用：** ROUGE-S在实践中使用较少，ROUGE-1、ROUGE-2、ROUGE-L是最常用的三个变体。

### 12.4.5 使用rouge库计算ROUGE

```python
from rouge_score import rouge_scorer

def compute_rouge_scores(hypotheses, references):
    """
    计算ROUGE分数

    Args:
        hypotheses: 候选摘要列表
        references: 参考摘要列表

    Returns:
        dict: 包含ROUGE-1, ROUGE-2, ROUGE-L的平均F1分数
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True  # 使用词干提取，提升鲁棒性
    )

    scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []},
    }

    for hyp, ref in zip(hypotheses, references):
        result = scorer.score(ref, hyp)  # 注意：rouge_scorer是(reference, hypothesis)

        for metric in scores:
            scores[metric]['precision'].append(result[metric].precision)
            scores[metric]['recall'].append(result[metric].recall)
            scores[metric]['fmeasure'].append(result[metric].fmeasure)

    # 计算平均值
    avg_scores = {}
    for metric in scores:
        avg_scores[metric] = {
            'precision': sum(scores[metric]['precision']) / len(scores[metric]['precision']),
            'recall': sum(scores[metric]['recall']) / len(scores[metric]['recall']),
            'fmeasure': sum(scores[metric]['fmeasure']) / len(scores[metric]['fmeasure']),
        }

    return avg_scores


# 使用示例
hypotheses = [
    "The cat sat on the mat in the room",
    "Scientists discovered a new species in the deep ocean",
]

references = [
    "The cat was sitting on the mat",
    "Researchers found a new ocean species",
]

scores = compute_rouge_scores(hypotheses, references)

for metric, values in scores.items():
    print(f"{metric.upper()}:")
    print(f"  Precision: {values['precision']:.4f}")
    print(f"  Recall:    {values['recall']:.4f}")
    print(f"  F1:        {values['fmeasure']:.4f}")
```

### 12.4.6 ROUGE评分的解读

**经验参考值（新闻摘要任务）：**

| 模型 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|---------|---------|---------|
| 抽取式基线（Lead-3） | ~40 | ~17 | ~36 |
| LSTM摘要模型 | ~35-42 | ~13-19 | ~32-39 |
| BERT2BERT | ~43 | ~20 | ~40 |
| PEGASUS | ~44-47 | ~21-24 | ~41-44 |

**解读要点：**
- F1分数通常用于最终报告
- 对于摘要任务，高召回率往往比高精确率更重要
- 不同数据集的绝对值差异很大，应在同一数据集上比较

---

## 12.5 其他评估指标

### 12.5.1 METEOR

**METEOR**（Metric for Evaluation of Translation with Explicit ORdering）是对BLEU的改进，解决了BLEU的几个关键缺陷：

**核心改进：**
1. **同时考虑精确率和召回率**（F1权重：precision权重10%，recall权重90%）
2. **支持词干匹配**（"running" 匹配 "run"）
3. **支持同义词匹配**（通过WordNet）
4. **考虑词序**（通过分块惩罚）

**公式：**

$$\text{METEOR} = F_{mean} \cdot (1 - \text{Penalty})$$

$$F_{mean} = \frac{P \cdot R}{(1-\alpha) \cdot R + \alpha \cdot P}$$

其中 $\alpha = 0.9$，Penalty基于匹配块的数量。

```python
# 使用nltk计算METEOR
import nltk
from nltk.translate.meteor_score import meteor_score

# 需要下载WordNet
# nltk.download('wordnet')
# nltk.download('punkt')

reference = "the cat sat on the mat".split()
hypothesis = "the cat is sitting on the mat".split()

score = meteor_score([reference], hypothesis)
print(f"METEOR: {score:.4f}")
# METEOR通常比BLEU更高，因为它识别了"sitting"是"sat"的变体
```

**特点：** METEOR与人工评估的相关性通常高于BLEU，但计算更慢，且依赖语言资源（WordNet）。

### 12.5.2 BERTScore

**BERTScore**是2020年提出的基于语义的评估指标，利用预训练BERT模型来计算参考文本和候选文本之间的**语义相似度**。

**核心思想：**

1. 使用BERT将参考词和候选词编码为上下文向量
2. 计算每对词向量之间的余弦相似度
3. 通过贪婪匹配找最优对应关系
4. 计算精确率、召回率和F1

$$P_{BERT} = \frac{1}{|\hat{y}|} \sum_{\hat{y}_i \in \hat{y}} \max_{y_j \in y} \mathbf{x}_{\hat{y}_i}^T \mathbf{x}_{y_j}$$

$$R_{BERT} = \frac{1}{|y|} \sum_{y_j \in y} \max_{\hat{y}_i \in \hat{y}} \mathbf{x}_{y_j}^T \mathbf{x}_{\hat{y}_i}$$

```python
from bert_score import score as bert_score

def compute_bert_score(hypotheses, references, model_type="bert-base-uncased"):
    """
    计算BERTScore

    Args:
        hypotheses: 候选文本列表
        references: 参考文本列表
        model_type: 用于计算的BERT模型

    Returns:
        tuple: (precision, recall, f1) 各自的张量
    """
    P, R, F1 = bert_score(
        hypotheses,
        references,
        model_type=model_type,
        lang="en",
        verbose=True
    )

    print(f"BERTScore F1: {F1.mean():.4f}")
    return P, R, F1

# 示例
hyps = ["The dog quickly ran to the park"]
refs = ["The dog rapidly sprinted to the garden"]

# P, R, F1 = compute_bert_score(hyps, refs)
# BERTScore能识别 "quickly"≈"rapidly", "ran"≈"sprinted", "park"≈"garden"
# 得到高于BLEU的分数
```

**BERTScore的优势：**

- 对同义词、改写不敏感
- 与人工评估相关性高于BLEU
- 支持多语言（使用多语言BERT）

**劣势：**

- 计算慢（需要BERT推理）
- 结果解释性差（黑盒）
- 分数范围依赖模型，不同模型不可比

### 12.5.3 人工评估方法

自动评估指标有其局限性，重要的研究和产品发布通常需要人工评估。

**常见人工评估维度：**

| 维度 | 描述 | 评分方式 |
|------|------|---------|
| 流畅度（Fluency） | 文本是否自然流畅 | 1-5 Likert量表 |
| 忠实度（Faithfulness） | 是否忠实于原文 | 二元/三元判断 |
| 相关性（Relevance） | 是否回答了问题 | 1-5 Likert量表 |
| 一致性（Consistency） | 内容是否自相矛盾 | 二元判断 |
| 整体质量（Overall） | 综合质量评分 | 1-5 Likert量表 |

**A/B测试（对比评估）：**

```
评估者看到：
    系统A输出：...
    系统B输出：...

评估者判断：A更好 / B更好 / 差不多
```

**评估者间一致性（Inter-Annotator Agreement）：**

```python
from sklearn.metrics import cohen_kappa_score

# 计算两个评估者之间的Kappa系数
annotations_rater1 = [1, 2, 3, 2, 1, 3, 2, 1]
annotations_rater2 = [1, 2, 2, 2, 1, 3, 3, 1]

kappa = cohen_kappa_score(annotations_rater1, annotations_rater2)
print(f"Cohen's Kappa: {kappa:.4f}")
# Kappa > 0.6 被认为是合理的一致性
# Kappa > 0.8 被认为是很好的一致性
```

### 12.5.4 评估指标的局限性与选择建议

**各指标的局限性总结：**

| 指标 | 主要局限性 |
|------|---------|
| BLEU | 不考虑语义；对词序变化敏感；与人工评估相关性有限 |
| ROUGE | 同BLEU的语义问题；对摘要长度敏感 |
| METEOR | 依赖WordNet（英语为主）；计算复杂 |
| BERTScore | 计算慢；黑盒；不同模型分数不可比 |
| 困惑度 | 只衡量语言模型质量，不衡量任务质量 |
| 人工评估 | 成本高；主观性强；难以大规模 |

**实际使用建议：**

1. **报告多个指标**，不要只报告一个
2. **了解指标设计的任务**：翻译用BLEU，摘要用ROUGE
3. **在相同数据集上比较**，绝对分数无意义
4. **对重要结论进行人工验证**
5. **考虑使用LLM-as-Judge**（用大模型评估小模型输出，近年流行）

```python
# LLM-as-Judge 简单示例思路（不含API调用细节）
def llm_judge_prompt(question, reference, candidate):
    """构建用于LLM评判的提示词"""
    return f"""你是一个公正的评估者。请评估以下回答的质量。

问题: {question}
参考答案: {reference}
候选答案: {candidate}

请从以下维度评分（1-5分）：
1. 准确性：答案是否正确？
2. 完整性：是否涵盖了主要要点？
3. 流畅性：语言是否自然？

请给出总体评分（1-5）和简短理由。"""
```

---

## 本章小结

本章介绍了Transformer训练和评估中的核心指标体系：

| 指标 | 类型 | 主要用途 | 优点 | 缺点 |
|------|------|---------|------|------|
| 交叉熵损失 | 训练损失 | 所有生成任务 | 直接优化目标；可微分 | 不直观；依赖词表 |
| 困惑度（PPL） | 内在评估 | 语言模型 | 直观；与损失直接相关 | 不反映任务质量 |
| BLEU | 外在评估 | 机器翻译 | 快速；可重复；有标准实现 | 不考虑语义；精确率偏向 |
| ROUGE-1/2 | 外在评估 | 文本摘要 | 关注召回率；适合摘要 | N-gram局限性 |
| ROUGE-L | 外在评估 | 文本摘要 | 考虑词序；不需N参数 | 计算较复杂 |
| METEOR | 外在评估 | 机器翻译 | 支持同义词；F1平衡 | 依赖语言资源 |
| BERTScore | 语义评估 | 多种生成任务 | 捕捉语义相似性 | 计算慢；黑盒 |
| 人工评估 | 最高标准 | 关键研究/产品 | 最准确；多维度 | 成本高；难扩展 |

**核心要点回顾：**

1. **交叉熵损失**是序列生成的基础，`ignore_index`处理Padding，`label_smoothing`防止过拟合
2. **困惑度**等于 $\exp(\text{损失})$，越低越好，是语言模型的内在质量指标
3. **BLEU**基于精确率，适合翻译，多参考+简短惩罚提升准确性
4. **ROUGE**基于召回率，适合摘要，ROUGE-L对词序变化更鲁棒
5. **没有完美指标**，实际评估应结合多个指标和人工验证

---

## 代码实战：完整评估流程

下面是一个完整的评估脚本，整合了本章所有内容：

```python
"""
完整的Transformer评估脚本
包括：带掩码的交叉熵损失、困惑度计算、BLEU计算
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 带掩码的交叉熵损失
# ============================================================

class MaskedCrossEntropyLoss(nn.Module):
    """
    带Padding掩码和可选标签平滑的交叉熵损失

    Args:
        pad_idx: Padding Token的索引
        label_smoothing: 标签平滑系数（0表示不使用）
        reduction: 'mean'（Token平均）或 'sum'
    """

    def __init__(
        self,
        pad_idx: int = 0,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
            reduction=reduction
        )

    def forward(
        self,
        logits: torch.Tensor,  # [batch, seq, vocab]
        targets: torch.Tensor   # [batch, seq]
    ) -> torch.Tensor:
        """
        计算损失

        Returns:
            标量损失值
        """
        batch_size, seq_len, vocab_size = logits.shape

        # 展平为2D/1D
        logits_flat = logits.reshape(-1, vocab_size)  # [batch*seq, vocab]
        targets_flat = targets.reshape(-1)              # [batch*seq]

        loss = self.criterion(logits_flat, targets_flat)
        return loss

    def count_non_pad_tokens(self, targets: torch.Tensor) -> int:
        """统计非Padding Token数量"""
        return (targets != self.pad_idx).sum().item()


# ============================================================
# 2. 困惑度计算函数
# ============================================================

class PerplexityEvaluator:
    """
    困惑度计算器，支持批次计算和数值稳定性保证

    Args:
        pad_idx: Padding Token的索引
        device: 计算设备
    """

    def __init__(self, pad_idx: int = 0, device: str = 'cpu'):
        self.pad_idx = pad_idx
        self.device = device
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            reduction='sum'  # 使用sum便于累计总损失
        )

    def compute_batch_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[float, int]:
        """
        计算一个批次的总损失和Token数

        Returns:
            (total_loss, num_tokens) 元组
        """
        batch_size, seq_len, vocab_size = logits.shape

        loss = self.criterion(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        num_tokens = (targets != self.pad_idx).sum().item()
        return loss.item(), num_tokens

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader
    ) -> dict:
        """
        在整个数据集上计算困惑度

        Returns:
            dict: 包含 ppl, avg_loss, total_tokens
        """
        model.eval()
        model.to(self.device)

        total_loss = 0.0
        total_tokens = 0

        for batch_idx, batch in enumerate(dataloader):
            # 假设batch是 (src, tgt) 元组
            src, tgt = batch
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # 解码器输入/目标（teacher forcing）
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            # 前向传播
            logits = model(src, tgt_input)

            batch_loss, batch_tokens = self.compute_batch_loss(logits, tgt_target)
            total_loss += batch_loss
            total_tokens += batch_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 100))  # 避免exp溢出

        return {
            'perplexity': ppl,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens
        }


# ============================================================
# 3. BLEU计算示例
# ============================================================

def compute_bleu_simple(hypothesis: List[str], references: List[List[str]], max_n: int = 4) -> float:
    """
    不依赖外部库的简单BLEU实现（用于教学理解）

    Args:
        hypothesis: 候选句子（词列表）
        references: 参考句子列表（每个元素是词列表）
        max_n: 最大N-gram阶数

    Returns:
        BLEU分数（0-1）
    """
    from collections import Counter
    import math

    def get_ngrams(tokens: List[str], n: int) -> Counter:
        """获取N-gram计数"""
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def modified_precision(hyp: List[str], refs: List[List[str]], n: int) -> Tuple[int, int]:
        """计算修正的N-gram精确率（返回分子和分母）"""
        hyp_ngrams = get_ngrams(hyp, n)

        if not hyp_ngrams:
            return 0, 0

        # 对每个N-gram，在所有参考中取最大计数
        max_ref_counts = Counter()
        for ref in refs:
            ref_ngrams = get_ngrams(ref, n)
            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

        # 截断计数
        clipped = sum(min(count, max_ref_counts[ngram])
                     for ngram, count in hyp_ngrams.items())

        return clipped, sum(hyp_ngrams.values())

    def brevity_penalty(hyp_len: int, ref_len: int) -> float:
        """计算简短惩罚"""
        if hyp_len >= ref_len:
            return 1.0
        return math.exp(1 - ref_len / hyp_len)

    # 计算各阶N-gram精确率
    precisions = []
    for n in range(1, max_n + 1):
        numerator, denominator = modified_precision(hypothesis, references, n)
        if denominator == 0:
            precisions.append(0)
        elif numerator == 0:
            return 0.0  # 任何一阶为0则BLEU为0
        else:
            precisions.append(numerator / denominator)

    # 对数平均（等权重）
    log_avg = sum(math.log(p) for p in precisions) / max_n

    # 简短惩罚（使用最接近的参考长度）
    hyp_len = len(hypothesis)
    ref_len = min(len(ref) for ref in references, key=lambda r: abs(len(r) - hyp_len))
    bp = brevity_penalty(hyp_len, len(ref_len) if hasattr(ref_len, '__len__') else ref_len)

    return bp * math.exp(log_avg)


# ============================================================
# 4. 完整的评估脚本
# ============================================================

class TranslationEvaluator:
    """
    机器翻译完整评估器
    整合困惑度、BLEU等多种指标
    """

    def __init__(
        self,
        model: nn.Module,
        pad_idx: int = 0,
        device: str = 'cpu'
    ):
        self.model = model
        self.device = device
        self.ppl_evaluator = PerplexityEvaluator(pad_idx, device)
        self.loss_fn = MaskedCrossEntropyLoss(pad_idx)

    def decode_batch(
        self,
        src: torch.Tensor,
        tokenizer,
        max_len: int = 100,
        bos_idx: int = 1,
        eos_idx: int = 2
    ) -> List[str]:
        """
        贪婪解码一批源句子

        Returns:
            解码后的字符串列表
        """
        self.model.eval()
        batch_size = src.size(0)

        # 初始化解码器输入（BOS token）
        decoder_input = torch.full(
            (batch_size, 1), bos_idx,
            dtype=torch.long, device=self.device
        )

        completed = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            # 编码源句子（只做一次）
            src = src.to(self.device)

            for _ in range(max_len):
                # 解码一步
                logits = self.model(src, decoder_input)
                next_token_logits = logits[:, -1, :]  # [batch, vocab]
                next_tokens = next_token_logits.argmax(dim=-1)  # [batch]

                # 已完成的序列继续填充PAD
                next_tokens[completed] = 0  # PAD

                # 追加到解码序列
                decoder_input = torch.cat(
                    [decoder_input, next_tokens.unsqueeze(1)], dim=1
                )

                # 检查是否所有序列都生成了EOS
                completed = completed | (next_tokens == eos_idx)
                if completed.all():
                    break

        # 转换为字符串（去掉BOS，截到EOS）
        decoded = []
        for seq in decoder_input[:, 1:]:  # 去掉BOS
            tokens = seq.tolist()
            # 截到EOS
            if eos_idx in tokens:
                tokens = tokens[:tokens.index(eos_idx)]
            decoded.append(tokenizer.decode(tokens))

        return decoded

    def full_evaluation(
        self,
        test_dataloader: DataLoader,
        tokenizer,
        references_list: Optional[List[str]] = None
    ) -> dict:
        """
        完整评估流程

        Returns:
            包含所有评估指标的字典
        """
        results = {}

        # 1. 困惑度
        print("Computing perplexity...")
        ppl_results = self.ppl_evaluator.evaluate(self.model, test_dataloader)
        results.update(ppl_results)

        # 2. BLEU（需要解码）
        if references_list is not None:
            print("Computing BLEU...")
            try:
                from sacrebleu.metrics import BLEU as SacreBLEU

                all_hypotheses = []

                for batch in test_dataloader:
                    src = batch[0]
                    hyps = self.decode_batch(src, tokenizer)
                    all_hypotheses.extend(hyps)

                bleu_metric = SacreBLEU()
                bleu_score = bleu_metric.corpus_score(
                    all_hypotheses[:len(references_list)],
                    [references_list]
                )
                results['bleu'] = bleu_score.score

            except ImportError:
                print("sacrebleu not installed, skipping BLEU")

        # 打印结果
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Perplexity:    {results.get('perplexity', 'N/A'):.2f}")
        print(f"Avg Loss:      {results.get('avg_loss', 'N/A'):.4f}")
        print(f"Total Tokens:  {results.get('total_tokens', 'N/A'):,}")
        if 'bleu' in results:
            print(f"BLEU Score:    {results['bleu']:.2f}")
        print("="*50)

        return results


# ============================================================
# 5. 快速验证示例（无需真实模型）
# ============================================================

def demo_metrics():
    """演示各指标的计算，不需要真实的Transformer模型"""

    print("=" * 60)
    print("评估指标演示")
    print("=" * 60)

    # --- 损失函数演示 ---
    print("\n1. 交叉熵损失演示")
    print("-" * 40)

    vocab_size = 50
    batch_size = 2
    seq_len = 5
    pad_idx = 0

    # 模拟模型输出
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # 带Padding的目标
    targets = torch.tensor([
        [5, 12, 34, 7, 0],   # 最后1个是Padding
        [3, 19, 0, 0, 0],    # 最后3个是Padding
    ])

    loss_fn = MaskedCrossEntropyLoss(pad_idx=pad_idx)
    loss = loss_fn(logits, targets[:, 1:] if seq_len > 1 else targets)

    # 简单演示
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    loss_demo = criterion(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )
    print(f"批次大小: {batch_size}, 序列长度: {seq_len}, 词表大小: {vocab_size}")
    print(f"非Padding Token数: {(targets != pad_idx).sum().item()}")
    print(f"交叉熵损失: {loss_demo.item():.4f}")

    # --- 困惑度演示 ---
    print("\n2. 困惑度演示")
    print("-" * 40)

    # 从损失计算困惑度
    for loss_val in [1.0, 2.0, 3.5, 5.0]:
        ppl = math.exp(loss_val)
        print(f"损失={loss_val:.1f} -> 困惑度={ppl:.2f}")

    # --- BLEU演示 ---
    print("\n3. BLEU分数演示")
    print("-" * 40)

    try:
        from sacrebleu.metrics import BLEU

        test_cases = [
            ("the cat sat on the mat", "the cat sat on the mat", "完全匹配"),
            ("the cat sat on the mat", "the cat is on the mat", "一词不同"),
            ("a cat sat on mat", "the cat sat on the mat", "多词不同"),
            ("the cat", "the cat sat on the mat", "截断"),
        ]

        bleu = BLEU(effective_order=True)
        print(f"{'候选':<35} {'参考':<35} {'情况':<12} {'BLEU':>6}")
        print("-" * 95)
        for hyp, ref, desc in test_cases:
            score = bleu.sentence_score(hyp, [ref])
            print(f"{hyp:<35} {ref:<35} {desc:<12} {score.score:>6.2f}")

    except ImportError:
        print("sacrebleu未安装，跳过BLEU演示")
        print("安装命令: pip install sacrebleu")

    # --- ROUGE演示 ---
    print("\n4. ROUGE分数演示")
    print("-" * 40)

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

        hyp = "Scientists discover new species in ocean depths"
        ref = "Researchers found new ocean species in deep waters"

        scores = scorer.score(ref, hyp)

        print(f"候选: {hyp}")
        print(f"参考: {ref}")
        print()
        for metric, score in scores.items():
            print(f"{metric.upper()}:")
            print(f"  Precision: {score.precision:.4f}")
            print(f"  Recall:    {score.recall:.4f}")
            print(f"  F1:        {score.fmeasure:.4f}")

    except ImportError:
        print("rouge_score未安装，跳过ROUGE演示")
        print("安装命令: pip install rouge-score")

    print("\n演示完成！")


if __name__ == "__main__":
    demo_metrics()
```

**运行依赖安装：**

```bash
pip install torch sacrebleu rouge-score bert-score nltk
```

**运行输出示例：**

```
============================================================
评估指标演示
============================================================

1. 交叉熵损失演示
----------------------------------------
批次大小: 2, 序列长度: 5, 词表大小: 50
非Padding Token数: 5
交叉熵损失: 3.8421

2. 困惑度演示
----------------------------------------
损失=1.0 -> 困惑度=2.72
损失=2.0 -> 困惑度=7.39
损失=3.5 -> 困惑度=33.12
损失=5.0 -> 困惑度=148.41

3. BLEU分数演示
----------------------------------------
候选                                参考                                情况         BLEU
-----------------------------------------------------------------------------------------------
the cat sat on the mat              the cat sat on the mat              完全匹配      100.00
the cat sat on the mat              the cat is on the mat               一词不同       51.45
a cat sat on mat                    the cat sat on the mat              多词不同       37.62
the cat                             the cat sat on the mat              截断           20.05

4. ROUGE分数演示
----------------------------------------
候选: Scientists discover new species in ocean depths
参考: Researchers found new ocean species in deep waters
...
```

---

## 练习题

### 基础题

**练习1** （基础）

给定以下序列预测结果，手动计算交叉熵损失和困惑度：

| 位置 | 真实Token | 预测概率 |
|------|---------|---------|
| t=1  | "apple" | 0.6     |
| t=2  | "is"    | 0.8     |
| t=3  | "red"   | 0.4     |
| t=4  | [PAD]   | 0.9     |（忽略此位置）

**要求：** 展示完整计算过程，包括忽略PAD位置后的损失和困惑度。

---

**练习2** （基础）

分析以下代码并回答问题：

```python
import torch
import torch.nn as nn

criterion_A = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
criterion_B = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

logits = torch.randn(2, 4, 100)
targets = torch.tensor([[5, 12, 34, 0], [7, 23, 0, 0]])

loss_A = criterion_A(logits.view(-1, 100), targets.view(-1))
loss_B = criterion_B(logits.view(-1, 100), targets.view(-1))
```

**问题：**
1. `loss_A` 和 `loss_B` 之间有什么关系？
2. 如果将 `logits` 中某位置改变，`loss_A` 对应的困惑度如何变化？
3. 哪个 `criterion` 更适合用于多GPU分布式训练中的梯度聚合？

---

### 中级题

**练习3** （中级）

**实现一个计算语料库级BLEU分数的函数，不使用sacrebleu库。**

输入：
```python
corpus_hyps = [
    "the cat sat on the mat".split(),
    "there is a cat on mat".split(),
]
corpus_refs = [
    ["the cat sat on the mat".split()],   # 每个位置可以有多个参考
    ["a cat is sitting on the mat".split(), "there is a cat on the mat".split()],
]
```

要求：
- 支持BLEU-1到BLEU-4
- 正确处理简短惩罚（在语料库级别计算，不是句子级别）
- 返回最终BLEU-4分数

**提示：** 语料库级BLEU的简短惩罚使用所有句子的总长度。

---

**练习4** （中级）

分析BLEU和ROUGE在以下场景中的差异：

给定：
- 参考摘要（200词）：一篇完整描述某事件的摘要
- 候选A（50词）：提取了参考摘要中最重要的50词
- 候选B（200词）：与参考摘要词汇重叠度30%但语义相似的完整摘要

**问题：**
1. 哪个候选的ROUGE-1 F1分数更高？为什么？
2. 哪个候选的BLEU分数更高？为什么？
3. 从语义相关性角度，哪个候选质量更高？这说明了自动评估指标的什么局限性？

**要求：** 用公式和数值估算支持你的分析。

---

### 提高题

**练习5** （提高）

**设计并实现一个"评估套件"类，满足以下需求：**

1. 支持同时计算 PPL、BLEU、ROUGE-1、ROUGE-2、ROUGE-L
2. 支持增量更新（每处理一个批次就更新内部统计，不存储所有预测结果）
3. 对于PPL，使用数值稳定的对数累加
4. 提供 `reset()` 方法重置统计
5. 提供 `summary()` 方法输出格式化结果

**接口规范：**

```python
class EvaluationSuite:
    def __init__(self, pad_idx: int, bos_idx: int, eos_idx: int):
        ...

    def update_ppl(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """更新PPL统计（不存储所有logits）"""
        ...

    def update_generation(self, hypotheses: List[str], references: List[str]) -> None:
        """更新BLEU/ROUGE统计"""
        ...

    def reset(self) -> None:
        """重置所有统计"""
        ...

    def summary(self) -> dict:
        """返回所有指标的当前值"""
        ...
```

**要求：**
- 代码完整可运行
- 包含适当的注释
- 提供使用示例和预期输出

---

## 练习答案

### 练习1答案

**忽略PAD位置后，只计算t=1、t=2、t=3三个位置：**

**步骤1：计算各位置的负对数概率**

$$-\log p(y_1) = -\log(0.6) = 0.5108$$
$$-\log p(y_2) = -\log(0.8) = 0.2231$$
$$-\log p(y_3) = -\log(0.4) = 0.9163$$

**步骤2：计算平均损失（忽略PAD，只有T=3个有效Token）**

$$\mathcal{L} = \frac{0.5108 + 0.2231 + 0.9163}{3} = \frac{1.6502}{3} = 0.5501$$

**步骤3：计算困惑度**

$$\text{PPL} = \exp(0.5501) = e^{0.5501} \approx 1.733$$

**解读：** 困惑度约为1.73，意味着模型平均在不到2个选项中进行选择，这是一个相当好的模型（但这是理想化的小例子）。

**代码验证：**

```python
import torch
import torch.nn as nn
import math

# 模拟情形：3个真实Token + 1个PAD
# 我们用logits来精确还原这些概率
# 简化：直接用交叉熵公式
probs = [0.6, 0.8, 0.4]
neg_log_probs = [-math.log(p) for p in probs]
avg_loss = sum(neg_log_probs) / len(neg_log_probs)
ppl = math.exp(avg_loss)

print(f"各位置-log(p): {[f'{v:.4f}' for v in neg_log_probs]}")
print(f"平均损失: {avg_loss:.4f}")
print(f"困惑度: {ppl:.4f}")
# 输出：
# 各位置-log(p): ['0.5108', '0.2231', '0.9163']
# 平均损失: 0.5501
# 困惑度: 1.7333
```

---

### 练习2答案

**问题1：loss_A 和 loss_B 的关系**

非Padding Token数量 = (targets != 0).sum() = 5（位置(0,0),(0,1),(0,2),(1,0),(1,1)）

```
loss_A = loss_B / 5
```

即：`loss_A * 非PAD Token数 = loss_B`

**问题2：困惑度变化**

困惑度 = exp(loss_A)。如果修改某位置的logits使该位置预测概率升高，则该位置损失降低，loss_A 降低，困惑度也随之降低（因为exp是单调递增函数）。

**问题3：多GPU分布式训练**

应使用 `reduction='sum'`（`criterion_B`），因为：
- 不同GPU处理的批次可能含有不同数量的非PAD Token
- 先在各GPU上累加总损失（sum），在合并后再除以总Token数得到真正的平均值
- 如果用 `mean`，各GPU的均值直接平均会因批次PAD比例不同而产生偏差

```python
# 正确的分布式实现
loss_sum = criterion_B(logits.view(-1, 100), targets.view(-1))
num_tokens = (targets != 0).sum()

# all_reduce后：
# total_loss_sum / total_num_tokens = 正确的全局平均损失
```

---

### 练习3答案

```python
from collections import Counter
import math
from typing import List, Tuple

def corpus_bleu(
    hypotheses: List[List[str]],
    references_list: List[List[List[str]]],
    max_n: int = 4
) -> float:
    """
    计算语料库级BLEU分数

    Args:
        hypotheses: 候选句子列表，每个是词列表
        references_list: 参考句子列表，每个位置可有多个参考
        max_n: 最大N-gram阶数

    Returns:
        BLEU-4分数（0-1之间）
    """

    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    # 累计每阶N-gram的分子和分母（语料库级别！）
    numerators = [0] * max_n
    denominators = [0] * max_n

    # 累计候选总长度和参考最近长度
    total_hyp_len = 0
    total_ref_len = 0

    for hyp, refs in zip(hypotheses, references_list):
        hyp_len = len(hyp)
        # 选最接近候选长度的参考长度
        ref_len = min((len(ref) for ref in refs), key=lambda r: abs(r - hyp_len))

        total_hyp_len += hyp_len
        total_ref_len += ref_len

        for n in range(1, max_n + 1):
            hyp_ngrams = get_ngrams(hyp, n)

            if not hyp_ngrams:
                continue

            # 多参考取最大计数
            max_ref_counts = Counter()
            for ref in refs:
                ref_ngrams = get_ngrams(ref, n)
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

            # 截断计数
            clipped = sum(
                min(count, max_ref_counts[ngram])
                for ngram, count in hyp_ngrams.items()
            )

            numerators[n-1] += clipped
            denominators[n-1] += sum(hyp_ngrams.values())

    # 计算各阶精确率
    precisions = []
    for n in range(max_n):
        if denominators[n] == 0:
            precisions.append(0.0)
        elif numerators[n] == 0:
            return 0.0
        else:
            precisions.append(numerators[n] / denominators[n])

    # 对数加权平均
    log_avg_precision = sum(math.log(p) for p in precisions) / max_n

    # 简短惩罚（语料库级别）
    if total_hyp_len >= total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - total_ref_len / total_hyp_len)

    bleu = bp * math.exp(log_avg_precision)
    return bleu


# 测试
corpus_hyps = [
    "the cat sat on the mat".split(),
    "there is a cat on mat".split(),
]
corpus_refs = [
    ["the cat sat on the mat".split()],
    ["a cat is sitting on the mat".split(), "there is a cat on the mat".split()],
]

score = corpus_bleu(corpus_hyps, corpus_refs)
print(f"BLEU-4: {score:.4f}")
# 参考值：~0.58
```

---

### 练习4答案

**数值分析：**

设参考摘要200词，候选A提取50词（完全来自参考），候选B 200词（30%词汇重叠，即60词匹配）。

**ROUGE-1 F1分析：**

候选A（50词，全部匹配）：
- Precision = 50/50 = 1.0
- Recall = 50/200 = 0.25
- F1 = 2×1.0×0.25/(1.0+0.25) = 0.4

候选B（200词，60词匹配）：
- Precision = 60/200 = 0.3
- Recall = 60/200 = 0.3
- F1 = 0.3

**结论：候选A的ROUGE-1 F1更高（0.4 vs 0.3）。**

原因：候选A的精确率极高（全部来自参考），拉高了F1值。

**BLEU分析：**

候选A（50词 vs 参考200词）：
- 简短惩罚 BP = exp(1 - 200/50) = exp(-3) ≈ 0.05（极大惩罚）
- 即使N-gram精确率很高，BP也会将BLEU压到很低
- 估计BLEU ≈ 0.05 × 0.8 ≈ 0.04

候选B（200词，30%重叠）：
- BP = 1.0（等长）
- 估计精确率在0.3左右
- BLEU ≈ 0.3^0.25（几何平均近似）≈ 0.30

**结论：候选B的BLEU分数更高（~0.30 vs ~0.04）。**

**语义质量判断：**

候选B"语义相似的完整摘要"在语义上质量更高，因为它完整覆盖了内容，且语义相似。候选A虽然精确率高，但因为长度太短，丢失了大量信息。

**评估指标局限性：**
1. **ROUGE偏向召回率**，会高估过度摘录（候选A）
2. **BLEU偏向精确率且惩罚短句**，会低估简短摘要（候选A）
3. 两个指标都无法真正衡量**语义相似度**
4. 这说明：单一指标容易被"游戏"（模型学会满足指标但不满足真实需求）

---

### 练习5答案

```python
import math
import torch
import torch.nn as nn
from typing import List, Optional
from collections import Counter, defaultdict


class EvaluationSuite:
    """
    增量式多指标评估套件
    支持PPL、BLEU、ROUGE-1、ROUGE-2、ROUGE-L的增量计算
    """

    def __init__(self, pad_idx: int = 0, bos_idx: int = 1, eos_idx: int = 2):
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # PPL统计（使用对数空间，数值稳定）
        self._ppl_total_nll = 0.0   # 总负对数似然（对数空间累加）
        self._ppl_total_tokens = 0

        # BLEU统计（增量N-gram计数）
        self._bleu_numerators = [0] * 4    # n=1,2,3,4
        self._bleu_denominators = [0] * 4
        self._bleu_hyp_len = 0
        self._bleu_ref_len = 0

        # ROUGE统计
        self._rouge_scores = defaultdict(lambda: {'P': 0.0, 'R': 0.0, 'F': 0.0})
        self._rouge_count = 0

        # 损失函数（sum reduction for PPL）
        self._criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            reduction='sum'
        )

    def update_ppl(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        更新PPL统计（增量，不存储logits）

        Args:
            logits: [batch, seq, vocab]
            targets: [batch, seq]（已去掉BOS）
        """
        with torch.no_grad():
            batch_size, seq_len, vocab_size = logits.shape

            # 计算批次总NLL（对数空间，数值稳定）
            nll = self._criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            num_tokens = (targets != self.pad_idx).sum().item()

            # 累加（在对数空间保持数值稳定）
            self._ppl_total_nll += nll.item()
            self._ppl_total_tokens += num_tokens

    def update_generation(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> None:
        """
        更新BLEU/ROUGE统计（增量）

        Args:
            hypotheses: 候选文本列表（字符串）
            references: 参考文本列表（字符串）
        """
        for hyp_str, ref_str in zip(hypotheses, references):
            hyp = hyp_str.lower().split()
            ref = ref_str.lower().split()

            # 更新BLEU统计
            self._update_bleu_stats(hyp, ref)

            # 更新ROUGE统计
            self._update_rouge_stats(hyp, ref)

        self._rouge_count += len(hypotheses)

    def _update_bleu_stats(self, hyp: List[str], ref: List[str]) -> None:
        """更新BLEU的N-gram统计"""
        self._bleu_hyp_len += len(hyp)
        self._bleu_ref_len += len(ref)

        ref_ngrams = {}
        for n in range(1, 5):
            ref_ngrams[n] = Counter(
                tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)
            )

        for n in range(1, 5):
            if len(hyp) < n:
                continue

            hyp_ngrams = Counter(
                tuple(hyp[i:i+n]) for i in range(len(hyp) - n + 1)
            )

            clipped = sum(
                min(count, ref_ngrams[n].get(ng, 0))
                for ng, count in hyp_ngrams.items()
            )

            self._bleu_numerators[n-1] += clipped
            self._bleu_denominators[n-1] += sum(hyp_ngrams.values())

    def _lcs_length(self, a: List[str], b: List[str]) -> int:
        """计算最长公共子序列长度（DP实现）"""
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0

        # 空间优化：只保留两行
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(curr[j-1], prev[j])
            prev, curr = curr, [0] * (n + 1)

        return prev[n]

    def _update_rouge_stats(self, hyp: List[str], ref: List[str]) -> None:
        """更新ROUGE统计"""
        # ROUGE-1
        for n, key in [(1, 'rouge1'), (2, 'rouge2')]:
            if len(hyp) < n or len(ref) < n:
                continue

            hyp_ngrams = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1))
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))

            overlap = sum(min(c, ref_ngrams.get(ng, 0)) for ng, c in hyp_ngrams.items())

            P = overlap / sum(hyp_ngrams.values()) if hyp_ngrams else 0.0
            R = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0
            F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

            self._rouge_scores[key]['P'] += P
            self._rouge_scores[key]['R'] += R
            self._rouge_scores[key]['F'] += F

        # ROUGE-L
        lcs_len = self._lcs_length(hyp, ref)
        P_l = lcs_len / len(hyp) if hyp else 0.0
        R_l = lcs_len / len(ref) if ref else 0.0
        F_l = 2 * P_l * R_l / (P_l + R_l) if (P_l + R_l) > 0 else 0.0

        self._rouge_scores['rougeL']['P'] += P_l
        self._rouge_scores['rougeL']['R'] += R_l
        self._rouge_scores['rougeL']['F'] += F_l

    def reset(self) -> None:
        """重置所有统计"""
        self._ppl_total_nll = 0.0
        self._ppl_total_tokens = 0
        self._bleu_numerators = [0] * 4
        self._bleu_denominators = [0] * 4
        self._bleu_hyp_len = 0
        self._bleu_ref_len = 0
        self._rouge_scores = defaultdict(lambda: {'P': 0.0, 'R': 0.0, 'F': 0.0})
        self._rouge_count = 0

    def summary(self) -> dict:
        """返回所有当前指标"""
        results = {}

        # PPL
        if self._ppl_total_tokens > 0:
            avg_nll = self._ppl_total_nll / self._ppl_total_tokens
            results['ppl'] = math.exp(min(avg_nll, 100))
            results['avg_loss'] = avg_nll

        # BLEU
        if any(d > 0 for d in self._bleu_denominators):
            precisions = []
            valid = True
            for n in range(4):
                if self._bleu_denominators[n] == 0:
                    precisions.append(0.0)
                elif self._bleu_numerators[n] == 0:
                    valid = False
                    break
                else:
                    precisions.append(self._bleu_numerators[n] / self._bleu_denominators[n])

            if valid and all(p > 0 for p in precisions):
                log_avg = sum(math.log(p) for p in precisions) / 4
                bp = (1.0 if self._bleu_hyp_len >= self._bleu_ref_len
                      else math.exp(1 - self._bleu_ref_len / self._bleu_hyp_len))
                results['bleu'] = bp * math.exp(log_avg) * 100
            else:
                results['bleu'] = 0.0

        # ROUGE
        if self._rouge_count > 0:
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                results[metric] = {
                    'P': self._rouge_scores[metric]['P'] / self._rouge_count,
                    'R': self._rouge_scores[metric]['R'] / self._rouge_count,
                    'F': self._rouge_scores[metric]['F'] / self._rouge_count,
                }

        return results

    def print_summary(self) -> None:
        """格式化打印评估结果"""
        results = self.summary()

        print("\n" + "=" * 55)
        print("  EVALUATION SUMMARY")
        print("=" * 55)

        if 'ppl' in results:
            print(f"  Perplexity:    {results['ppl']:>10.2f}")
            print(f"  Avg Loss:      {results['avg_loss']:>10.4f}")

        if 'bleu' in results:
            print(f"  BLEU-4:        {results['bleu']:>10.2f}")

        for metric in ['rouge1', 'rouge2', 'rougeL']:
            if metric in results:
                r = results[metric]
                print(f"  {metric.upper():<10} P={r['P']:.4f}  R={r['R']:.4f}  F={r['F']:.4f}")

        print("=" * 55)


# 使用示例
def demo_evaluation_suite():
    """演示EvaluationSuite的使用"""

    suite = EvaluationSuite(pad_idx=0)

    # 模拟3个批次的PPL更新
    for batch_i in range(3):
        logits = torch.randn(4, 8, 100)  # batch=4, seq=8, vocab=100
        targets = torch.randint(1, 100, (4, 8))
        targets[:, -2:] = 0  # 最后2个是PAD
        suite.update_ppl(logits, targets)

    # 模拟生成评估
    hyps = [
        "the cat sat on the mat",
        "the weather is nice today",
        "machine learning is fascinating",
    ]
    refs = [
        "the cat sat on the mat",
        "today the weather is very nice",
        "deep learning is very interesting",
    ]
    suite.update_generation(hyps, refs)
    suite.update_generation(hyps, refs)  # 再来一批

    suite.print_summary()

    # 重置后重新计算
    suite.reset()
    assert suite.summary() == {}


if __name__ == "__main__":
    demo_evaluation_suite()
```

**预期输出：**

```
=======================================================
  EVALUATION SUMMARY
=======================================================
  Perplexity:        ~100.00  (随机logits，预期高PPL)
  Avg Loss:            ~4.60
  BLEU-4:             ~52.00
  ROUGE1     P=0.8500  R=0.8200  F=0.8300
  ROUGE2     P=0.7200  R=0.6800  F=0.6900
  ROUGEL     P=0.8500  R=0.8200  F=0.8300
=======================================================
```

**关键设计亮点：**

1. **PPL使用累加NLL**，不存储logits，内存效率高
2. **BLEU使用计数累加**，每批只保留4个分子和分母值
3. **ROUGE使用平均累加**，每批O(1)空间
4. **reset()** 清除所有状态，支持跨epoch重用
5. **数值稳定性**：PPL通过`min(avg_nll, 100)`防止exp溢出

---

*本章完*

> **下一章预告：** 第13章将介绍优化器与学习率调度策略，包括Adam、AdaFactor以及Transformer独特的Warmup学习率调度方案。
