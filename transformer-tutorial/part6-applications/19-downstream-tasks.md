# 第19章：下游任务实战

> **学习目标**
>
> 完成本章学习后，你将能够：
> 1. 掌握文本分类任务的微调方法，包括单句分类和句子对分类
> 2. 掌握序列标注任务（NER、POS）的实现，理解BIO标注体系
> 3. 理解抽取式与生成式问答系统的实现原理
> 4. 掌握条件文本生成任务（摘要、翻译）的训练与解码策略
> 5. 能够根据具体任务特点，选择最合适的预训练模型架构

---

## 引言

预训练模型的真正价值，在于其对下游任务的强大适应能力。从BERT到GPT，从T5到LLaMA，这些模型在海量语料上习得的语言表示，可以通过微调（Fine-tuning）或提示（Prompting）迁移到各种具体的NLP任务中。

本章将系统介绍最重要的四类下游任务——文本分类、序列标注、问答系统和文本生成——的实现方法。我们不仅关注原理，更注重工程实现：每一节都配有可运行的PyTorch代码，涵盖数据处理、模型搭建、训练循环和评估指标。

理解下游任务的关键，在于理解**任务形式（Task Formulation）**：如何将一个NLP问题转化为模型能够处理的输入输出格式。这种转化能力，是NLP工程师最核心的技能之一。

---

## 19.1 文本分类

文本分类是NLP中最基础也最常见的任务形式。给定一段文本，判断其属于哪个类别。情感分析、主题分类、意图识别、自然语言推理——这些看似不同的任务，在技术层面都归属于文本分类的范畴。

### 19.1.1 单句分类：情感分析

**任务定义**：给定一个句子 $x$，预测其类别标签 $y \in \{1, 2, \ldots, K\}$。

对于BERT类模型，标准做法是使用 `[CLS]` token的输出向量作为整句的表示，再接一个线性分类头：

$$
\hat{y} = \text{softmax}(W \cdot h_{\text{[CLS]}} + b)
$$

其中 $h_{\text{[CLS]}} \in \mathbb{R}^d$ 是BERT最后一层 `[CLS]` 位置的隐状态，$W \in \mathbb{R}^{K \times d}$，$b \in \mathbb{R}^K$ 是分类头的参数。

训练损失为交叉熵损失：

$$
\mathcal{L} = -\sum_{i=1}^{N} \log P(\hat{y}_i = y_i)
$$

**为什么用 `[CLS]`？**

BERT在预训练时，`[CLS]` token被设计为聚合整个序列信息的特殊位置。在NSP（下一句预测）任务中，正是用 `[CLS]` 的表示来判断两句话是否相邻。因此，`[CLS]` 天然地携带了全局语义信息，适合用于分类任务。

当然，也可以对所有token的输出做平均池化（Mean Pooling），实践中在某些任务上效果更好：

$$
h_{\text{avg}} = \frac{1}{L} \sum_{i=1}^{L} h_i
$$

其中 $L$ 是序列长度。

### 19.1.2 句子对分类

句子对分类涉及两个句子：**自然语言推理（NLI）**判断前提与假设的逻辑关系，**语义相似度（STS）**判断两句话的语义是否相近。

输入格式为：

```
[CLS] 句子A [SEP] 句子B [SEP]
```

BERT通过Token Type ID区分两个句子（句子A的token对应ID=0，句子B对应ID=1）。模型同时关注两句话及其交互，`[CLS]` 的输出向量编码了句子对的关系信息。

NLI的标签为三类：蕴含（Entailment）、矛盾（Contradiction）、中性（Neutral）。

### 19.1.3 多标签分类

当一段文本可以同时属于多个类别时，需要多标签分类（Multi-label Classification）。例如一篇新闻文章可能同时属于"经济"和"政治"两个类别。

此时输出层改为 $K$ 个独立的二分类器，使用Sigmoid激活而非Softmax：

$$
\hat{y}_k = \sigma(W_k \cdot h_{\text{[CLS]}} + b_k), \quad k = 1, \ldots, K
$$

损失函数改为二元交叉熵（Binary Cross-Entropy）：

$$
\mathcal{L} = -\sum_{k=1}^{K} \left[ y_k \log \hat{y}_k + (1 - y_k) \log (1 - \hat{y}_k) \right]
$$

预测时，对每个类别独立设定阈值（通常为0.5），超过阈值则预测为正例。

### 19.1.4 完整代码示例：情感分析微调

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


# ==================== 数据集 ====================
class SentimentDataset(Dataset):
    """情感分析数据集"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ==================== 模型 ====================
class BertForSentimentClassification(nn.Module):
    """基于BERT的情感分类模型"""

    def __init__(self, model_name='bert-base-chinese', num_classes=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # 取 [CLS] token 的输出 (batch_size, hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


# ==================== 训练 ====================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label']

            logits = model(input_ids, attention_mask, token_type_ids)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=['负面', '正面'])
    return acc, report


# ==================== 主流程 ====================
def main():
    # 示例数据（实际使用时替换为真实数据集）
    train_texts = [
        "这部电影太精彩了，强烈推荐！",
        "剧情拖沓，演技尴尬，浪费时间。",
        "画面很美，配乐也不错。",
        "故事平淡，没有亮点。",
    ]
    train_labels = [1, 0, 1, 0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 构建数据集
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # 初始化模型
    model = BertForSentimentClassification(
        model_name='bert-base-chinese',
        num_classes=2
    ).to(device)

    # 优化器：BERT层使用较小学习率，分类头使用较大学习率
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': 2e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01)

    num_epochs = 3
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # 训练循环
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    print("训练完成！")


if __name__ == '__main__':
    main()
```

---

## 19.2 序列标注

序列标注（Sequence Labeling）是对输入序列中每个 token 分别预测一个标签的任务。命名实体识别（NER）和词性标注（POS）是其中最典型的代表。

### 19.2.1 BIO标注体系

NER的目标是识别文本中的命名实体，如人名、地名、机构名等。最常用的标注格式是**BIO体系**：

- **B**（Begin）：实体的起始token
- **I**（Inside）：实体的内部token
- **O**（Outside）：非实体token

以句子"北京大学位于北京市"为例：

```
北  →  B-ORG   （机构名起始）
京  →  I-ORG
大  →  I-ORG
学  →  I-ORG
位  →  O
于  →  O
北  →  B-LOC   （地名起始）
京  →  I-LOC
市  →  I-LOC
```

还有更精细的**BIOES体系**，增加了：
- **E**（End）：实体的结束token
- **S**（Single）：单token实体

BIOES在解码时能提供更明确的实体边界信息，但标签集规模增加。

### 19.2.2 Token级别的分类

与文本分类不同，序列标注需要对每个token分别预测标签。对于BERT：

1. 输入句子经过Tokenizer，得到token序列
2. BERT输出每个token位置的隐状态 $h_1, h_2, \ldots, h_L$
3. 对每个token的隐状态接分类头：

$$
P(y_i \mid x) = \text{softmax}(W \cdot h_i + b)
$$

**处理中文字符与Subword的对齐问题**

BERT的Tokenizer可能将一个单词切分为多个subword。在序列标注中，需要将token级别的预测对齐回原始词：通常只保留每个原始词的第一个subword对应的预测标签，其余subword的预测被忽略。

```python
# 示例：对齐原始词与subword标签
word_ids = encoding.word_ids()  # 每个subword对应的原始词索引
labels_aligned = []
prev_word_id = None
for word_id in word_ids:
    if word_id is None:
        labels_aligned.append(-100)   # 特殊token，忽略损失
    elif word_id != prev_word_id:
        labels_aligned.append(true_labels[word_id])  # 每词只取第一个subword
    else:
        labels_aligned.append(-100)   # 同一词的后续subword，忽略
    prev_word_id = word_id
```

### 19.2.3 CRF层

独立地对每个位置做分类，可能产生不合法的标签序列，例如"I-PER"出现在"B-ORG"之后。为此，可以在BERT输出之上添加**条件随机场（CRF）层**：

CRF建模标签序列的联合概率：

$$
P(y_1, \ldots, y_L \mid x) = \frac{1}{Z(x)} \exp\left(\sum_{i=1}^{L} s(x, i, y_i) + \sum_{i=1}^{L-1} T(y_i, y_{i+1})\right)
$$

其中：
- $s(x, i, y_i)$ 是BERT输出的发射分数（emission score）
- $T(y_i, y_{i+1})$ 是转移分数（transition score），由CRF层学习
- $Z(x)$ 是归一化常数

CRF通过Viterbi算法在解码时找全局最优标签序列，有效避免了非法转移（如 I 出现在 O 之后）。

实践中，对于数据量充足的任务，BERT + Linear 已能取得很好效果；CRF层在数据稀少时更有帮助。

### 19.2.4 NER完整代码

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from torchcrf import CRF  # pip install pytorch-crf


# ==================== 标签定义 ====================
LABEL2ID = {
    'O': 0,
    'B-PER': 1, 'I-PER': 2,
    'B-ORG': 3, 'I-ORG': 4,
    'B-LOC': 5, 'I-LOC': 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


# ==================== 数据集 ====================
class NERDataset(Dataset):
    """NER数据集，输入为字符列表和对应标签列表"""

    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        word_labels = self.labels[idx]

        # 对每个字符单独编码，获取word_ids用于对齐
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 对齐标签：只保留每词第一个subword的标签
        aligned_labels = []
        word_ids = encoding.word_ids(batch_index=0)
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                aligned_labels.append(LABEL2ID[word_labels[word_id]])
            else:
                aligned_labels.append(-100)
            prev_word_id = word_id

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }


# ==================== 模型 ====================
class BertCRFForNER(nn.Module):
    """BERT + CRF 的NER模型"""

    def __init__(self, model_name='bert-base-chinese', num_labels=NUM_LABELS, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(outputs.last_hidden_state)
        # (batch_size, seq_len, num_labels)
        emissions = self.linear(sequence_output)

        if labels is not None:
            # 训练阶段：计算CRF负对数似然损失
            # 将 -100 的位置替换为 0（CRF不接受负数标签）
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            mask = (labels != -100)
            loss = -self.crf(emissions, crf_labels, mask=mask, reduction='mean')
            return loss
        else:
            # 推理阶段：Viterbi解码
            mask = attention_mask.bool()
            pred_ids = self.crf.decode(emissions, mask=mask)
            return pred_ids

    def predict(self, input_ids, attention_mask, token_type_ids):
        """推理接口，返回标签名称列表"""
        pred_ids = self.forward(input_ids, attention_mask, token_type_ids)
        pred_labels = [[ID2LABEL[id_] for id_ in seq] for seq in pred_ids]
        return pred_labels


# ==================== 评估：seqeval ====================
def compute_ner_metrics(true_seqs, pred_seqs):
    """
    计算NER的精确率、召回率、F1分数
    true_seqs, pred_seqs: List[List[str]]，标签序列列表
    """
    from seqeval.metrics import precision_score, recall_score, f1_score
    precision = precision_score(true_seqs, pred_seqs)
    recall = recall_score(true_seqs, pred_seqs)
    f1 = f1_score(true_seqs, pred_seqs)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return precision, recall, f1
```

---

## 19.3 问答系统

问答（Question Answering, QA）系统根据问题从文档中找到或生成答案。主流方案分为**抽取式（Extractive）**和**生成式（Generative）**两类。

### 19.3.1 抽取式问答

抽取式QA的任务定义：给定问题 $q$ 和包含答案的段落 $p$，在段落中找到答案的起始位置 $s$ 和结束位置 $e$。

SQuAD（Stanford Question Answering Dataset）是该任务最著名的基准数据集。

**模型架构**

输入格式为：

```
[CLS] 问题 [SEP] 段落 [SEP]
```

BERT对每个token输出隐状态 $h_i$，然后通过两个独立的线性层分别预测答案的起始和结束位置：

$$
P_{\text{start}}(i) = \text{softmax}_i(W_s \cdot h_i)
$$

$$
P_{\text{end}}(i) = \text{softmax}_i(W_e \cdot h_i)
$$

答案对应的span为 $[s, e]$，其中：

$$
(s^*, e^*) = \arg\max_{s \leq e} \left[ P_{\text{start}}(s) \cdot P_{\text{end}}(e) \right]
$$

实践中通常限制 $e - s \leq \text{max\_answer\_length}$ 以避免过长答案。

训练损失为起始位置和结束位置的交叉熵之和：

$$
\mathcal{L} = \mathcal{L}_{\text{start}} + \mathcal{L}_{\text{end}}
$$

**处理长文档**

BERT的最大输入长度为512。当段落超过此限制时，需要使用**滑动窗口（Sliding Window）**策略：将长段落切分为多个重叠的窗口，分别预测，最后合并答案。

```python
# 滑动窗口切分示例
doc_stride = 128   # 相邻窗口的重叠token数
max_length = 384   # 每个窗口的最大长度

encoding = tokenizer(
    question,
    context,
    max_length=max_length,
    stride=doc_stride,
    return_overflowing_tokens=True,  # 启用滑动窗口
    return_offsets_mapping=True,
    padding='max_length',
    truncation='only_second'         # 只截断段落，不截断问题
)
```

### 19.3.2 完整抽取式QA代码

```python
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel


class BertForQuestionAnswering(nn.Module):
    """基于BERT的抽取式问答模型"""

    def __init__(self, model_name='bert-base-chinese'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        # 两个输出：起始位置分数 + 结束位置分数
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids,
                start_positions=None, end_positions=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # (batch_size, seq_len, hidden_size)
        sequence_output = outputs.last_hidden_state
        # (batch_size, seq_len, 2)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (batch_size, seq_len)
        end_logits = end_logits.squeeze(-1)       # (batch_size, seq_len)

        if start_positions is not None and end_positions is not None:
            # 训练阶段：计算损失
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = criterion(start_logits, start_positions)
            end_loss = criterion(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss

        return start_logits, end_logits

    def predict(self, input_ids, attention_mask, token_type_ids,
                offset_mapping, context, max_answer_length=30):
        """推理：给定编码后的输入，返回预测的答案文本"""
        start_logits, end_logits = self.forward(
            input_ids, attention_mask, token_type_ids
        )
        # 取batch中第一个样本
        start_logits = start_logits[0].cpu().numpy()
        end_logits = end_logits[0].cpu().numpy()

        # 找最优 (start, end) 组合
        best_score = float('-inf')
        best_start = best_end = 0
        for s in range(len(start_logits)):
            for e in range(s, min(s + max_answer_length, len(end_logits))):
                score = start_logits[s] + end_logits[e]
                if score > best_score:
                    best_score = score
                    best_start, best_end = s, e

        # 通过offset_mapping将token位置映射回原始字符位置
        offsets = offset_mapping[0].cpu().numpy()
        char_start = offsets[best_start][0]
        char_end = offsets[best_end][1]
        answer = context[char_start:char_end]
        return answer
```

### 19.3.3 生成式问答与RAG简介

**生成式问答**使用编码器-解码器模型（如T5）直接生成答案文本，而不限于从段落中抽取span。优点是可以处理需要综合多处信息、需要推理的问题；缺点是可解释性差，且容易产生幻觉（Hallucination）。

**检索增强生成（Retrieval-Augmented Generation, RAG）**将检索与生成结合：

```
用户问题
    │
    ▼
检索器（Retriever）
    │  从知识库中检索相关段落
    ▼
相关段落
    │
    ▼
生成器（Generator）
    │  以问题+段落为条件，生成答案
    ▼
最终答案
```

RAG的核心优势：
1. 知识可以独立于模型参数更新（更新知识库即可）
2. 答案有据可查，可追溯来源
3. 有效缓解生成模型的幻觉问题

检索器通常使用**双编码器（Bi-Encoder）**架构：将问题和文档分别编码为稠密向量，用向量相似度检索。代表系统：DPR（Dense Passage Retrieval）。

---

## 19.4 文本生成

文本生成任务要求模型输出一段文本，而非预测一个类别或位置。条件生成（如摘要、翻译）和开放域生成（如对话、故事创作）是两大主要方向。

### 19.4.1 条件生成：摘要与翻译

条件生成的输入是一段文本（源文档），输出是另一段文本（目标文本）。典型任务：

- **摘要（Summarization）**：输入长文，输出简短摘要
- **翻译（Translation）**：输入源语言文本，输出目标语言文本
- **改写（Paraphrase）**：保持语义不变，改变表达方式

编码器-解码器架构（如BART、T5）天然适合这类任务：

```
编码器读取整个源文档 → 解码器自回归地生成目标文本
```

训练时使用**Teacher Forcing**：解码器的每个时间步以真实的前一个token作为输入，而非自己的预测值。损失函数为目标序列上的交叉熵：

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, x)
$$

### 19.4.2 解码策略

推理时，解码器需要从条件分布中逐步生成token。不同的解码策略产生不同质量和风格的输出：

**贪心解码（Greedy Decoding）**

每步选择概率最高的token：

$$
\hat{y}_t = \arg\max_{v} P(v \mid y_{<t}, x)
$$

优点：速度快；缺点：容易陷入局部最优，生成重复文本。

**束搜索（Beam Search）**

同时维护 $B$ 条候选序列（beam），每步扩展后保留得分最高的 $B$ 条：

$$
\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i}, x)
$$

束搜索在翻译和摘要任务中表现良好，但在开放域生成中可能产生过于保守、缺乏多样性的输出。通常使用长度惩罚（Length Penalty）缓解倾向于生成短序列的问题：

$$
\text{score}(y_{1:T}) = \frac{1}{T^\alpha} \sum_{t=1}^{T} \log P(y_t \mid y_{<t}, x)
$$

其中 $\alpha > 0$ 是长度惩罚系数（通常取0.6到0.9）。

**采样（Sampling）**

从分布中随机采样，引入多样性：

$$
\hat{y}_t \sim P(\cdot \mid y_{<t}, x)
$$

- **温度采样（Temperature Sampling）**：调整分布的"尖锐程度"

$$
P_T(v) \propto \exp\left(\frac{\log P(v)}{T}\right)
$$

$T < 1$ 使分布更尖锐（保守），$T > 1$ 使分布更平滑（多样）。

- **Top-k采样**：只从概率最高的 $k$ 个token中采样
- **Top-p采样（Nucleus Sampling）**：只从累积概率超过 $p$ 的最小token集合中采样

```python
# 使用Transformers库的生成参数
model.generate(
    input_ids,
    max_new_tokens=200,
    num_beams=4,           # 束搜索，beam size=4
    length_penalty=0.8,    # 长度惩罚
    no_repeat_ngram_size=3,# 禁止3-gram重复
    early_stopping=True,   # 所有beam遇到EOS则停止

    # 或使用采样
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.92,
)
```

### 19.4.3 摘要生成完整代码

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate  # pip install evaluate rouge_score


# ==================== 数据集 ====================
class SummarizationDataset(Dataset):
    """摘要数据集"""

    def __init__(self, articles, summaries, tokenizer,
                 max_input_length=1024, max_target_length=128):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        # 编码源文本
        model_inputs = self.tokenizer(
            self.articles[idx],
            max_length=self.max_input_length,
            truncation=True,
            padding=False
        )
        # 编码目标摘要（作为labels）
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                self.summaries[idx],
                max_length=self.max_target_length,
                truncation=True,
                padding=False
            )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs


# ==================== 评估：ROUGE ====================
def compute_rouge_metrics(eval_pred, tokenizer):
    """计算ROUGE分数"""
    rouge = evaluate.load('rouge')
    predictions, labels = eval_pred

    # 解码预测结果
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # 将labels中的-100替换为pad_token_id
    labels = [[l if l != -100 else tokenizer.pad_token_id for l in label]
              for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    return {k: round(v * 100, 2) for k, v in result.items()}


# ==================== 训练 ====================
def train_summarization():
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # 示例数据（实际使用CNN/DailyMail等数据集）
    articles = ["The president announced new economic policies today..."]
    summaries = ["President unveils new economic plan."]

    dataset = SummarizationDataset(articles, summaries, tokenizer)

    # 使用Seq2SeqTrainer简化训练流程
    training_args = Seq2SeqTrainingArguments(
        output_dir='./summarization-output',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        predict_with_generate=True,   # 推理时使用generate()而非forward()
        generation_max_length=128,
        generation_num_beams=4,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='rouge2',
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_rouge_metrics(p, tokenizer),
    )

    trainer.train()


# ==================== 推理 ====================
def generate_summary(model, tokenizer, article, device,
                     num_beams=4, max_length=128):
    """对单篇文章生成摘要"""
    inputs = tokenizer(
        article,
        return_tensors='pt',
        max_length=1024,
        truncation=True
    ).to(device)

    summary_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        num_beams=num_beams,
        max_length=max_length,
        min_length=30,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
```

### 19.4.4 生成质量控制

提高生成质量的常用技巧：

| 技术 | 作用 | 适用场景 |
|------|------|----------|
| `no_repeat_ngram_size` | 禁止n-gram重复 | 防止复读机问题 |
| `repetition_penalty` | 惩罚已生成的token | 开放域生成 |
| `min_length` / `max_length` | 控制输出长度 | 摘要、回复生成 |
| `forced_bos_token_id` | 强制输出语言 | 多语言翻译 |
| `bad_words_ids` | 禁止特定词语 | 内容安全过滤 |
| `length_penalty` | 调整对长序列的偏好 | 翻译、摘要 |

---

## 19.5 任务选型指南

选择合适的预训练模型架构，是高效解决下游任务的关键。不同架构有各自的优势场景。

### 19.5.1 Encoder-only 模型（如BERT、RoBERTa）

**核心特点**：双向注意力，每个token能看到完整上下文。

**适合的任务**：
- 文本分类（情感分析、主题分类、意图识别）
- 序列标注（NER、POS、语块分析）
- 抽取式问答（预测答案span的起止位置）
- 语义相似度与句子对关系判断

**不适合的任务**：文本生成类任务（模型没有自回归生成能力）。

### 19.5.2 Decoder-only 模型（如GPT、LLaMA）

**核心特点**：单向（因果）注意力，自回归生成，预训练目标为语言建模。

**适合的任务**：
- 开放域文本生成（故事、对话、代码）
- 少样本学习（In-Context Learning）
- 指令跟随（Instruction Following）
- 思维链推理（Chain-of-Thought）

**不适合的任务**：需要对完整句子双向理解的任务（分类、NER效果通常不如Encoder模型）。

> **注意**：大型Decoder模型（如GPT-4、LLaMA-3）通过规模优势，可以在提示工程（Prompting）下完成分类和NER等任务，但专用Encoder模型在数据充足时仍具优势。

### 19.5.3 Encoder-Decoder 模型（如T5、BART）

**核心特点**：编码器双向读取源文本，解码器自回归生成目标文本。

**适合的任务**：
- 文本摘要
- 机器翻译
- 文本改写与风格迁移
- 生成式问答
- 数据增强（生成训练样本）

**不适合的任务**：纯理解类任务（编码器部分效果不如专用Encoder模型）。

### 19.5.4 任务-模型匹配表

| 任务类型 | 推荐架构 | 代表模型 | 输出形式 |
|----------|----------|----------|----------|
| 情感分析 | Encoder-only | BERT、RoBERTa | [CLS]接分类头 |
| 主题分类 | Encoder-only | BERT、DeBERTa | [CLS]接分类头 |
| 自然语言推理 | Encoder-only | RoBERTa | 句子对→分类头 |
| 命名实体识别 | Encoder-only | BERT + CRF | Token级分类 |
| 词性标注 | Encoder-only | BERT | Token级分类 |
| 抽取式问答 | Encoder-only | BERT | 预测span起止 |
| 文本摘要 | Encoder-Decoder | BART、T5 | 自回归生成 |
| 机器翻译 | Encoder-Decoder | T5、mBART | 自回归生成 |
| 生成式问答 | Encoder-Decoder | T5 | 自回归生成 |
| 开放域对话 | Decoder-only | GPT、LLaMA | 自回归生成 |
| 代码生成 | Decoder-only | CodeLLaMA | 自回归生成 |
| 少样本分类 | Decoder-only | GPT-4、LLaMA | 提示→生成标签 |

---

## 本章小结

| 任务 | 关键技术 | 输入格式 | 输出形式 | 评估指标 |
|------|----------|----------|----------|----------|
| 文本分类 | [CLS]向量 + 分类头 | 单句/句对 | Softmax概率 | Accuracy、F1 |
| 多标签分类 | Sigmoid独立分类 | 单句 | 多个0/1标签 | Micro/Macro F1 |
| 命名实体识别 | Token级分类 + BIO + CRF | 字符序列 | 每Token标签 | Entity-level F1 |
| 抽取式问答 | 预测起止位置 | 问题+段落 | 起始/结束位置 | EM、F1 |
| 生成式问答 | 编解码生成 | 问题+上下文 | 答案文本 | BLEU、ROUGE |
| 摘要生成 | 束搜索/采样解码 | 长文档 | 摘要文本 | ROUGE-1/2/L |
| 机器翻译 | Teacher Forcing训练 | 源语言句子 | 目标语言句子 | BLEU |

---

## 代码实战

本节汇总本章四个完整可运行的代码示例，并补充关键细节。

### 实战一：情感分析微调（完整版）

```python
"""
情感分析完整示例
任务：二分类（正面/负面）
模型：bert-base-chinese
数据：模拟的中文评论数据
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.data = list(zip(texts, labels))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        enc = self.tokenizer(
            text, max_length=self.max_len,
            padding='max_length', truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'token_type_ids': enc['token_type_ids'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class SentimentModel(nn.Module):
    def __init__(self, bert_name, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids)
        pooled = out.last_hidden_state[:, 0]
        return self.fc(self.drop(pooled))


def run_sentiment_training():
    # 构造示例数据
    texts = [
        "服务太好了，下次还来！", "食物很难吃，环境嘈杂。",
        "性价比超高，强烈推荐。", "等了一个小时，完全不值得。",
        "味道不错，但价格偏贵。", "员工态度恶劣，绝对不会再去。",
        "菜品新鲜，份量十足！", "踩坑了，完全是图片滤镜欺骗。",
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    dataset = ReviewDataset(texts, labels, tokenizer)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4)

    model = SentimentModel('bert-base-chinese').to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    total_steps = len(train_loader) * 5
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    for epoch in range(5):
        # 训练
        model.train()
        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            ttids = batch['token_type_ids'].to(device)
            labels_b = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask, ttids)
            loss = criterion(logits, labels_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # 验证
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                ttids = batch['token_type_ids'].to(device)
                logits = model(ids, mask, ttids)
                preds_all.extend(logits.argmax(-1).cpu().tolist())
                labels_all.extend(batch['label'].tolist())

        acc = accuracy_score(labels_all, preds_all)
        f1 = f1_score(labels_all, preds_all, average='macro')
        print(f"Epoch {epoch+1}: Acc={acc:.4f}, F1={f1:.4f}")


if __name__ == '__main__':
    run_sentiment_training()
```

### 实战二：NER任务实现

```python
"""
中文NER完整示例
标签集：O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
模型：bert-base-chinese（不含CRF，展示纯线性头方案）
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, AdamW


LABEL2ID = {'O': 0, 'B-PER': 1, 'I-PER': 2,
            'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class SimpleNERDataset(Dataset):
    """每条数据：字符列表 + 标签列表"""
    def __init__(self, sentences, label_seqs, tokenizer, max_len=128):
        self.sentences = sentences
        self.label_seqs = label_seqs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        chars = list(self.sentences[idx])   # 中文按字拆分
        raw_labels = self.label_seqs[idx]

        enc = self.tokenizer(
            chars, is_split_into_words=True,
            max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        # 对齐标签
        aligned = []
        prev_wid = None
        for wid in enc.word_ids(batch_index=0):
            if wid is None:
                aligned.append(-100)
            elif wid != prev_wid:
                aligned.append(LABEL2ID[raw_labels[wid]])
            else:
                aligned.append(-100)
            prev_wid = wid

        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'token_type_ids': enc['token_type_ids'].squeeze(),
            'labels': torch.tensor(aligned, dtype=torch.long)
        }


class BertNER(nn.Module):
    def __init__(self, bert_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.drop = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        seq_out = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state
        logits = self.classifier(self.drop(seq_out))  # (B, L, num_labels)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            # logits: (B, L, C) -> (B*L, C)，labels: (B, L) -> (B*L,)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        return logits


def ner_inference(model, tokenizer, sentence, device):
    """对单个句子进行NER推理"""
    chars = list(sentence)
    enc = tokenizer(
        chars, is_split_into_words=True,
        return_tensors='pt', return_offsets_mapping=False
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(enc['input_ids'], enc['attention_mask'],
                       enc['token_type_ids'])
    preds = logits.argmax(-1).squeeze().cpu().tolist()
    word_ids = enc.word_ids(batch_index=0)

    # 还原每个字的标签（只取第一个subword）
    char_labels = {}
    prev_wid = None
    for pos, wid in enumerate(word_ids):
        if wid is None or wid == prev_wid:
            prev_wid = wid
            continue
        char_labels[wid] = ID2LABEL[preds[pos]]
        prev_wid = wid

    results = [(chars[i], char_labels.get(i, 'O')) for i in range(len(chars))]
    return results


# 示例运行
if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = BertNER('bert-base-chinese', num_labels=len(LABEL2ID))

    # 示例训练数据
    sentences = ["习近平在北京出席人民大会堂开幕式"]
    label_seqs = [["B-PER","I-PER","I-PER",
                   "O","B-LOC","I-LOC","O","O",
                   "B-ORG","I-ORG","I-ORG","I-ORG","I-ORG",
                   "O","O","O","O"]]
    dataset = SimpleNERDataset(sentences, label_seqs, tokenizer)
    loader = DataLoader(dataset, batch_size=1)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.train()
    for batch in loader:
        loss = model(batch['input_ids'], batch['attention_mask'],
                     batch['token_type_ids'], batch['labels'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"NER训练Loss: {loss.item():.4f}")
```

### 实战三：抽取式问答

```python
"""
抽取式问答完整示例
模型：bert-base-chinese
任务：预测答案在段落中的起始和结束位置
"""
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertForQuestionAnswering as HFBertQA
from transformers import AdamW


def find_answer_token_positions(encoding, answer_start_char, answer_end_char):
    """
    在token序列中找到答案的起始和结束token位置。
    encoding 需要含 offset_mapping。
    """
    offset_mapping = encoding['offset_mapping'][0]
    start_token = end_token = 0
    for idx, (start, end) in enumerate(offset_mapping):
        if start <= answer_start_char < end:
            start_token = idx
        if start < answer_end_char <= end:
            end_token = idx
    return start_token, end_token


def run_qa_example():
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    # 直接使用HuggingFace内置的QA模型
    model = HFBertQA.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 示例数据
    question = "李白是哪个朝代的诗人？"
    context = "李白（701年—762年），字太白，号青莲居士，唐代伟大的浪漫主义诗人，被后人誉为诗仙。"
    answer_text = "唐代"
    # 定位答案在context中的字符位置
    answer_start = context.index(answer_text)
    answer_end = answer_start + len(answer_text)

    # Tokenize
    encoding = tokenizer(
        question, context,
        max_length=512, truncation=True,
        return_tensors='pt',
        return_offsets_mapping=True
    )
    start_pos, end_pos = find_answer_token_positions(
        encoding, answer_start, answer_end
    )

    # 训练一步
    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.train()
    outputs = model(
        input_ids=encoding['input_ids'].to(device),
        attention_mask=encoding['attention_mask'].to(device),
        token_type_ids=encoding['token_type_ids'].to(device),
        start_positions=torch.tensor([start_pos]).to(device),
        end_positions=torch.tensor([end_pos]).to(device)
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"QA训练Loss: {loss.item():.4f}")

    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device),
            token_type_ids=encoding['token_type_ids'].to(device)
        )
    start_idx = outputs.start_logits.argmax().item()
    end_idx = outputs.end_logits.argmax().item()

    # 通过offset_mapping还原字符位置
    offsets = encoding['offset_mapping'][0]
    if start_idx <= end_idx:
        char_start = offsets[start_idx][0].item()
        char_end = offsets[end_idx][1].item()
        predicted_answer = context[char_start:char_end]
    else:
        predicted_answer = ""

    print(f"问题：{question}")
    print(f"预测答案：{predicted_answer}")
    print(f"真实答案：{answer_text}")


if __name__ == '__main__':
    run_qa_example()
```

### 实战四：摘要生成

```python
"""
文本摘要推理示例（使用预训练BART模型）
无需训练，直接调用预训练权重生成摘要
"""
from transformers import pipeline


def summarize_text(text, max_length=130, min_length=30, num_beams=4):
    """
    使用BART对英文文本生成摘要。
    对中文可替换为 fnlp/bart-base-chinese 等中文摘要模型。
    """
    summarizer = pipeline(
        'summarization',
        model='facebook/bart-large-cnn',
        device=0  # 使用GPU，CPU则设为-1
    )
    result = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return result[0]['summary_text']


def batch_summarize(texts, batch_size=8):
    """批量摘要生成"""
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        outputs = summarizer(batch, max_length=130, min_length=30,
                             truncation=True)
        results.extend([o['summary_text'] for o in outputs])
    return results


def compare_decoding_strategies(text):
    """对比不同解码策略的效果"""
    from transformers import BartForConditionalGeneration, BartTokenizer
    import torch

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)

    strategies = {
        'greedy': dict(num_beams=1, do_sample=False),
        'beam_search': dict(num_beams=4, do_sample=False, length_penalty=2.0),
        'sampling': dict(num_beams=1, do_sample=True, temperature=0.8, top_p=0.9),
        'beam+sample': dict(num_beams=4, do_sample=True, temperature=0.7),
    }

    print("=" * 60)
    for name, kwargs in strategies.items():
        with torch.no_grad():
            ids = model.generate(
                inputs['input_ids'],
                max_length=100, min_length=20,
                no_repeat_ngram_size=3,
                **kwargs
            )
        summary = tokenizer.decode(ids[0], skip_special_tokens=True)
        print(f"\n[{name}]\n{summary}")
    print("=" * 60)


if __name__ == '__main__':
    sample_article = (
        "Scientists at MIT have developed a new artificial intelligence system "
        "that can predict the three-dimensional structure of proteins with "
        "remarkable accuracy. The breakthrough, published in Nature, could "
        "revolutionize drug discovery by dramatically speeding up the process "
        "of identifying potential drug targets. The system, called ProteinNet, "
        "uses a transformer-based architecture trained on millions of known "
        "protein sequences and structures. Researchers believe this technology "
        "could help develop treatments for diseases that have resisted existing "
        "drug development approaches."
    )
    summary = summarize_text(sample_article)
    print(f"摘要：{summary}")
```

---

## 练习题

### 基础题

**练习19.1**（基础）

BERT的 `[CLS]` token为什么适合用于文本分类任务？除了使用 `[CLS]`，还有哪些方式可以从BERT的输出中获取句子级表示？请分别描述它们的优缺点。

**练习19.2**（基础）

在NER任务中，BIO标注体系的"B"、"I"、"O"分别代表什么含义？对于以下句子，写出其完整的BIO标注序列（实体类型包括PER人名、LOC地名、ORG机构名）：

> 马云创立了阿里巴巴，总部位于杭州。

---

### 中级题

**练习19.3**（中级）

抽取式问答模型在推理时，如何从起始位置分数向量和结束位置分数向量中找到最优答案？如果暴力枚举所有 $(s, e)$ 组合，时间复杂度是多少？请设计一种时间复杂度为 $O(L)$ 的近似方法（$L$ 为序列长度），并说明其局限性。

**练习19.4**（中级）

比较束搜索（Beam Search）和Top-p采样（Nucleus Sampling）在以下两个场景中的适用性，并说明理由：
1. 科技新闻摘要生成
2. 创意写作（故事续写）

此外，如果束搜索产生的摘要存在严重的重复问题，有哪些参数或技术可以缓解？

---

### 提高题

**练习19.5**（提高）

在序列标注任务中，BERT + 线性层方案和 BERT + CRF 方案各有优劣。

（a）从概率建模的角度，解释为什么CRF能够避免"I-PER出现在B-ORG之后"这类非法标签序列？（提示：考虑独立分类与联合建模的区别）

（b）在以下两种情况下，分别分析哪种方案更合适，并给出理由：
- 训练集规模：50,000句，标注质量高
- 训练集规模：500句，领域专业性强

（c）如果不添加CRF层，但仍想在解码时避免非法标签序列，可以怎么做？（至少给出两种方法）

---

## 练习答案

### 答案19.1

**为什么 `[CLS]` 适合分类**

BERT在预训练阶段包含**下一句预测（NSP）**任务：给定两个句子，判断第二句是否是第一句的下一句。NSP任务的判断依据正是 `[CLS]` token 的输出向量。因此，`[CLS]` 在预训练中被优化为"聚合整个输入的语义信息"，适合用作句子级表示。

此外，`[CLS]` 在Transformer中通过自注意力机制能与所有其他token交互，位于序列最前端，没有位置偏见。

**其他句子级表示方案**：

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| `[CLS]`向量 | 取最后一层`[CLS]`位置输出 | 预训练目标专门优化 | NSP任务被证明对某些任务帮助有限 |
| 均值池化（Mean Pooling） | 对所有token输出取平均 | 更稳定，利用全部信息 | 对短句和长句一视同仁 |
| 最大池化（Max Pooling） | 对每个维度取最大值 | 保留最显著特征 | 忽略序列信息 |
| 多层平均 | 对最后N层输出取平均 | 融合不同层的语义 | 计算量稍大 |

实践建议：对于句子语义相似度任务，Mean Pooling通常优于 `[CLS]`（参考SimCSE论文）；对于分类任务，两者差异不大。

---

### 答案19.2

BIO含义：
- **B**（Begin）：命名实体的**起始**token
- **I**（Inside）：命名实体的**内部**token（不含起始）
- **O**（Outside）：**非命名实体**token

对句子"马云创立了阿里巴巴，总部位于杭州。"的标注：

| 字符 | 标签 | 说明 |
|------|------|------|
| 马 | B-PER | 人名"马云"起始 |
| 云 | I-PER | 人名"马云"内部 |
| 创 | O | |
| 立 | O | |
| 了 | O | |
| 阿 | B-ORG | 机构名"阿里巴巴"起始 |
| 里 | I-ORG | |
| 巴 | I-ORG | |
| 巴 | I-ORG | |
| ，| O | |
| 总 | O | |
| 部 | O | |
| 位 | O | |
| 于 | O | |
| 杭 | B-LOC | 地名"杭州"起始 |
| 州 | I-LOC | |
| 。| O | |

---

### 答案19.3

**最优答案选取方法**

暴力枚举所有合法 $(s, e)$ 对（$s \leq e$）的时间复杂度为 $O(L^2)$，其中 $L$ 为序列长度。当 $L=384$ 时，需要枚举约73,000个组合，但实际可以接受，因为每个操作只是加法。

**$O(L)$ 近似方法**：独立取最大值

最简单的近似：分别找起始位置分数最高的 $s^*$ 和结束位置分数最高的 $e^*$：

$$
s^* = \arg\max_i P_{\text{start}}(i), \quad e^* = \arg\max_j P_{\text{end}}(j)
$$

若 $s^* > e^*$，则返回空答案或取第二高分。

**局限性**：
- 无法保证 $s^* \leq e^*$，可能产生非法span
- 忽略了 $s$ 和 $e$ 的联合分布，可能错过真实的最优组合
- 实践中一般仍用 $O(L^2)$ 枚举，但限制最大答案长度（如30个token）将复杂度降为 $O(L \cdot \text{max\_len})$

---

### 答案19.4

**场景1：科技新闻摘要 → 推荐束搜索（Beam Search）**

理由：
- 摘要要求准确、信息密度高，忠实于原文，不需要创意
- 束搜索倾向于生成高概率、语法正确的序列，符合摘要的"稳定性"要求
- 推荐参数：`num_beams=4~8, length_penalty=1.0~2.0`

**场景2：创意写作 → 推荐Top-p采样（Nucleus Sampling）**

理由：
- 创意写作要求多样性和惊喜感，高概率的词往往导致陈词滥调
- Top-p采样从"核"分布中随机采样，引入不可预测性
- 推荐参数：`top_p=0.9, temperature=0.8~1.0`

**缓解束搜索重复问题的方法**：

1. `no_repeat_ngram_size=3`：禁止3-gram在输出中重复出现，是最有效的直接手段
2. `repetition_penalty > 1.0`（如1.2）：降低已生成token在后续步骤的概率
3. 减小 `num_beams`：束越多，越容易强化相似的高概率路径
4. 增大 `length_penalty`：鼓励模型生成更长序列，避免因短路径而反复填充

---

### 答案19.5

**(a) 为什么CRF能避免非法序列**

**独立分类（线性层）**：对位置 $i$ 的标签预测只依赖当前位置的BERT输出 $h_i$，每个位置独立做决策：

$$
P(y_i \mid x) = \text{softmax}(W h_i)
$$

这意味着位置 $i$ 的预测**完全不考虑**位置 $i-1$ 的标签，因此无法阻止"I-PER"出现在"B-ORG"之后。

**CRF**：对**整个标签序列**建模联合概率：

$$
P(y_1, \ldots, y_L \mid x) \propto \exp\left(\sum_i s(y_i, h_i) + \sum_i T(y_{i-1}, y_i)\right)
$$

转移分数矩阵 $T$ 由CRF层学习，可以将非法转移（如从`B-ORG`到`I-PER`）的转移分数学习为极大负值，从而在Viterbi解码时这些路径永远不会被选中。CRF的关键在于**全局归一化**：只有在考虑所有可能序列后，才能得到每个序列的概率。

**(b) 方案选择**

- **大数据（50,000句，高质量）**：BERT + 线性层已足够。数据量大时，模型可以通过大量样本间接学习到标签间的共现约束，CRF带来的收益边际递减。而且纯线性方案训练更快，超参数更少。

- **小数据（500句，专业领域）**：推荐 BERT + CRF。数据量小时，模型难以从样本中隐式学习标签转移规律。CRF的转移矩阵以显式结构约束编码了标注规则（如"I必须跟在B或同类I之后"），即使训练样本稀少，也能保证预测结果的合法性，相当于注入了先验知识。

**(c) 不用CRF避免非法序列的方法**

**方法一：解码后处理（Post-processing）**

在推理阶段，对预测出的标签序列进行规则修正：
```python
def fix_bio_sequence(labels):
    """修正非法BIO序列"""
    fixed = []
    prev_label = 'O'
    for label in labels:
        if label.startswith('I-'):
            entity_type = label[2:]
            if prev_label == 'O' or (prev_label.startswith('B-') and
               prev_label[2:] != entity_type) or (prev_label.startswith('I-') and
               prev_label[2:] != entity_type):
                # 将非法的I改为B
                label = 'B-' + entity_type
        fixed.append(label)
        prev_label = label
    return fixed
```

**方法二：约束解码（Constrained Decoding）**

在预测时，将非法转移的logit设为负无穷，确保argmax不会选到非法标签：
```python
def constrained_decode(logits, prev_label, transition_mask):
    """
    logits: (num_labels,) 当前位置的发射分数
    prev_label: 上一个位置的预测标签
    transition_mask: (num_labels, num_labels) 合法转移为1，非法为0
    """
    mask = transition_mask[prev_label]    # (num_labels,)
    masked_logits = logits + (1 - mask) * (-1e9)
    return masked_logits.argmax()
```

这两种方法不如CRF严格（没有全局最优性保证），但实现简单，计算开销极小。
