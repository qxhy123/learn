# 第22章：自然语言处理基础

> **系列**：Python深度学习教程 · 第七部分：深度学习应用
> **前置知识**：第20章（PyTorch基础）、第21章（卷积神经网络）
> **难度**：中级

---

## 学习目标

完成本章学习后，你将能够：

1. 掌握文本预处理的完整流程，包括分词、清洗和标准化
2. 构建词汇表并将文本编码为数值表示
3. 理解词嵌入的原理，能够使用 `nn.Embedding` 层
4. 构建基于 RNN/LSTM 的文本分类模型
5. 正确处理变长序列，使用 padding 和 `pack_padded_sequence`

---

## 22.1 文本预处理（分词、清洗、标准化）

自然语言处理（NLP）的第一步是将原始文本转换为模型可以处理的格式。这个过程称为**文本预处理**，包含多个子步骤。

### 22.1.1 为什么需要预处理？

原始文本存在很多"噪声"：

- HTML 标签、特殊符号、多余空格
- 大小写不统一（"Apple" 和 "apple" 语义相同）
- 标点符号干扰（"good." 和 "good" 应被视为同一词）
- 停用词（"the"、"a"、"is" 等对语义贡献很少）

### 22.1.2 基本清洗

```python
import re
import string

def clean_text(text: str) -> str:
    """基本文本清洗"""
    # 转为小写
    text = text.lower()
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除 URL
    text = re.sub(r'http\S+|www\S+', '', text)
    # 去除特殊字符，保留字母、数字和基本标点
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?]', '', text)
    # 压缩多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 示例
raw = "<p>This is a GREAT movie!! Visit http://example.com for more.</p>"
print(clean_text(raw))
# 输出: this is a great movie visit  for more.
```

### 22.1.3 分词（Tokenization）

分词是将文本分割为最小语义单元（token）的过程。

```python
# 方法一：简单空格分词
def simple_tokenize(text: str) -> list[str]:
    return text.split()

# 方法二：使用正则表达式（更精确）
def regex_tokenize(text: str) -> list[str]:
    # 匹配单词（包含缩写如 don't）
    pattern = r"\b\w+(?:'\w+)?\b"
    return re.findall(pattern, text)

# 方法三：使用 NLTK（推荐用于英文）
# pip install nltk
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

text = "Don't stop believing. It's a wonderful life!"
print(simple_tokenize(text))
# ["Don't", 'stop', 'believing.', "It's", 'a', 'wonderful', 'life!']

print(regex_tokenize(text))
# ["Don't", 'stop', 'believing', "It's", 'a', 'wonderful', 'life']

print(word_tokenize(text))
# ['Do', "n't", 'stop', 'believing', '.', 'It', "'s", 'a', 'wonderful', 'life', '!']
```

> **注意**：不同的分词策略会影响词汇表大小和模型性能。在实际项目中应根据任务选择合适的分词器。

### 22.1.4 停用词过滤与词形还原

```python
# 停用词过滤
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'it', 'its',
    'this', 'that', 'these', 'those', 'of', 'in', 'to', 'for',
    'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into',
}

def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.lower() not in STOPWORDS]

# 词形还原（Lemmatization）：将单词还原为基本形式
# pip install nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    return [lemmatizer.lemmatize(t) for t in tokens]

# 示例流程
text = "The cats are running quickly in the beautiful gardens"
tokens = regex_tokenize(text.lower())
tokens = remove_stopwords(tokens)
tokens = lemmatize_tokens(tokens)
print(tokens)
# ['cat', 'running', 'quickly', 'beautiful', 'garden']
```

### 22.1.5 完整预处理流水线

```python
class TextPreprocessor:
    """文本预处理流水线"""

    def __init__(self, remove_stops: bool = True, lemmatize: bool = False):
        self.remove_stops = remove_stops
        self.lemmatize = lemmatize
        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> list[str]:
        # 步骤1：清洗
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)       # 去 HTML
        text = re.sub(r'http\S+|www\S+', '', text) # 去 URL
        text = re.sub(r'[^a-z0-9\s]', ' ', text)  # 去特殊字符
        text = re.sub(r'\s+', ' ', text).strip()   # 压缩空格

        # 步骤2：分词
        tokens = text.split()

        # 步骤3：过滤停用词
        if self.remove_stops:
            tokens = [t for t in tokens if t not in STOPWORDS]

        # 步骤4：词形还原
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens

# 使用示例
preprocessor = TextPreprocessor(remove_stops=True, lemmatize=True)
result = preprocessor.preprocess(
    "<b>The movie was absolutely AMAZING!</b> I loved the acting."
)
print(result)
# ['movie', 'absolutely', 'amazing', 'loved', 'acting']
```

---

## 22.2 词汇表构建与编码

### 22.2.1 词汇表（Vocabulary）

词汇表是文本中所有唯一词的集合，并为每个词分配一个唯一的整数 ID。

```python
from collections import Counter
from typing import Optional

class Vocabulary:
    """词汇表类"""

    # 特殊标记
    PAD_TOKEN = '<pad>'  # 填充标记，ID=0
    UNK_TOKEN = '<unk>'  # 未知词标记，ID=1
    BOS_TOKEN = '<bos>'  # 句子开始
    EOS_TOKEN = '<eos>'  # 句子结束

    def __init__(self, min_freq: int = 1, max_size: Optional[int] = None):
        """
        Args:
            min_freq: 词出现的最低频次，低于此频次的词被视为未知词
            max_size: 词汇表最大大小（不含特殊标记）
        """
        self.min_freq = min_freq
        self.max_size = max_size

        # 词 -> ID 映射
        self.word2idx: dict[str, int] = {}
        # ID -> 词 映射
        self.idx2word: dict[int, str] = {}

        # 初始化特殊标记
        self._add_special_tokens()

    def _add_special_tokens(self):
        for token in [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def build(self, corpus: list[list[str]]):
        """
        从语料库构建词汇表
        Args:
            corpus: 分词后的文本列表，例如 [['hello', 'world'], ['foo', 'bar']]
        """
        # 统计词频
        counter = Counter()
        for tokens in corpus:
            counter.update(tokens)

        # 按频次排序
        sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 过滤低频词，限制词汇表大小
        if self.max_size:
            sorted_words = sorted_words[:self.max_size]

        for word, freq in sorted_words:
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, tokens: list[str]) -> list[int]:
        """将词列表编码为 ID 列表"""
        unk_idx = self.word2idx[self.UNK_TOKEN]
        return [self.word2idx.get(t, unk_idx) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """将 ID 列表解码为词列表"""
        return [self.idx2word.get(i, self.UNK_TOKEN) for i in ids]

    def __len__(self) -> int:
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[self.UNK_TOKEN]
```

### 22.2.2 使用词汇表

```python
# 示例语料库（已分词）
corpus = [
    ['the', 'movie', 'was', 'great', 'acting', 'was', 'superb'],
    ['terrible', 'movie', 'waste', 'of', 'time'],
    ['great', 'story', 'loved', 'every', 'minute'],
    ['boring', 'and', 'predictable', 'plot'],
]

# 构建词汇表（最低词频=1，最多保留20个词）
vocab = Vocabulary(min_freq=1, max_size=20)
vocab.build(corpus)

print(f"词汇表大小: {len(vocab)}")
# 词汇表大小: 22 (4个特殊标记 + 18个普通词)

# 编码示例
tokens = ['great', 'movie', 'unknown_word']
ids = vocab.encode(tokens)
print(f"编码: {tokens} -> {ids}")
# 编码: ['great', 'movie', 'unknown_word'] -> [6, 5, 1]

# 解码示例
decoded = vocab.decode(ids)
print(f"解码: {ids} -> {decoded}")
# 解码: [6, 5, 1] -> ['great', 'movie', '<unk>']
```

### 22.2.3 文本数值化与数据集

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """文本分类数据集"""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: Vocabulary,
        preprocessor: TextPreprocessor,
        max_len: int = 200,
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.labels = labels

        # 预处理并编码所有文本
        self.encoded = []
        for text in texts:
            tokens = preprocessor.preprocess(text)
            ids = vocab.encode(tokens)
            # 截断到最大长度
            ids = ids[:max_len]
            self.encoded.append(ids)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.encoded[idx], self.labels[idx]
```

---

## 22.3 词嵌入（Word2Vec原理、nn.Embedding）

### 22.3.1 为什么需要词嵌入？

如果直接使用词的整数 ID，模型无法理解词之间的语义关系。例如：
- "king"=15, "queen"=16, "apple"=17

从 ID 来看，"king" 和 "apple" 的数值差距（2）比 "king" 和 "queen"（1）还大，但语义上 "king" 和 "queen" 显然更相近。

**词嵌入**将每个词映射到一个低维连续向量空间，在这个空间中，语义相近的词距离更近。

### 22.3.2 Word2Vec 原理

Word2Vec 是 Google 在 2013 年提出的词嵌入训练方法，有两种架构：

**CBOW（Continuous Bag of Words）**：用上下文词预测中心词
```
上下文：[w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}]  →  预测：w_t
```

**Skip-gram**：用中心词预测上下文词
```
中心词：w_t  →  预测：[w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}]
```

核心思想：**出现在相似上下文中的词，应该有相似的向量表示**。

经过训练的词向量具有惊人的语义特性：
```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

### 22.3.3 PyTorch 中的 nn.Embedding

```python
import torch
import torch.nn as nn

# 创建嵌入层
vocab_size = 10000   # 词汇表大小
embed_dim = 128      # 嵌入维度

embedding = nn.Embedding(
    num_embeddings=vocab_size,  # 词汇表大小
    embedding_dim=embed_dim,    # 嵌入向量维度
    padding_idx=0,              # PAD token 的 ID，其梯度设为0
)

# 查看参数
print(f"嵌入层参数量: {embedding.weight.numel():,}")
# 嵌入层参数量: 1,280,000

# 前向传播
# 输入：整数索引张量，形状 [batch_size, seq_len]
batch_size, seq_len = 4, 20
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

# 输出：嵌入向量，形状 [batch_size, seq_len, embed_dim]
embedded = embedding(input_ids)
print(f"输入形状: {input_ids.shape}")    # torch.Size([4, 20])
print(f"输出形状: {embedded.shape}")     # torch.Size([4, 20, 128])
```

### 22.3.4 使用预训练词向量

在实际应用中，通常使用预训练的词向量（如 GloVe、fastText）初始化嵌入层：

```python
import numpy as np

def load_glove_vectors(glove_path: str, vocab: Vocabulary,
                       embed_dim: int = 100) -> torch.Tensor:
    """
    从 GloVe 文件加载预训练词向量
    Args:
        glove_path: GloVe 文件路径（如 glove.6B.100d.txt）
        vocab: 词汇表对象
        embed_dim: 嵌入维度
    Returns:
        shape [vocab_size, embed_dim] 的权重矩阵
    """
    # 初始化：未找到预训练向量的词使用随机初始化
    vectors = np.random.uniform(-0.1, 0.1, (len(vocab), embed_dim))
    # PAD token 全零
    vectors[vocab.pad_idx] = np.zeros(embed_dim)

    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab.word2idx:
                idx = vocab.word2idx[word]
                vectors[idx] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"找到预训练向量: {found}/{len(vocab)} 个词")
    return torch.FloatTensor(vectors)


def create_embedding_layer(vocab: Vocabulary, embed_dim: int,
                           pretrained_vectors: torch.Tensor = None,
                           freeze: bool = False) -> nn.Embedding:
    """创建嵌入层，可选择加载预训练向量"""
    embedding = nn.Embedding(
        num_embeddings=len(vocab),
        embedding_dim=embed_dim,
        padding_idx=vocab.pad_idx,
    )

    if pretrained_vectors is not None:
        embedding.weight.data.copy_(pretrained_vectors)
        if freeze:
            # 冻结预训练权重，不参与训练
            embedding.weight.requires_grad = False

    return embedding
```

---

## 22.4 文本分类模型

### 22.4.1 基于 LSTM 的分类器

**LSTM（Long Short-Term Memory）** 是处理序列数据的经典架构，能够捕获长程依赖。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    """基于双向 LSTM 的文本分类模型"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,        # 输入形状：[batch, seq, feature]
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # 双向 LSTM 的输出维度是 hidden_dim * 2
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor,
                lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len] 的整数张量
            lengths: [batch_size] 每个序列的实际长度（可选）
        Returns:
            [batch_size, num_classes] 的分类 logits
        """
        # [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.embedding(input_ids)

        if lengths is not None:
            # 使用 pack_padded_sequence 忽略 padding（见 22.5 节）
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, _) = self.lstm(packed)
            # hidden: [num_layers * num_directions, batch, hidden_dim]
        else:
            _, (hidden, _) = self.lstm(embedded)

        # 取最后一层的隐藏状态
        # 对于双向 LSTM：拼接正向和反向的最终隐藏状态
        if self.lstm.bidirectional:
            # hidden[-2]: 正向最后一层；hidden[-1]: 反向最后一层
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        # hidden: [batch_size, hidden_dim * 2]

        return self.classifier(hidden)
```

### 22.4.2 基于 TextCNN 的分类器

TextCNN 使用卷积提取 n-gram 特征，速度更快：

```python
class TextCNN(nn.Module):
    """基于卷积的文本分类模型"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        num_filters: int = 128,
        filter_sizes: list[int] = None,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super().__init__()
        if filter_sizes is None:
            filter_sizes = [2, 3, 4]  # 提取 bigram, trigram, 4-gram 特征

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )

        # 多尺度卷积核
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=fs,
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.embedding(input_ids)

        # Conv1d 期望 [batch, channels, length]
        # [batch, seq_len, embed_dim] -> [batch, embed_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)

        # 对每种卷积核大小：卷积 -> ReLU -> 全局最大池化
        pooled_outputs = []
        for conv in self.convs:
            # [batch, num_filters, seq_len - filter_size + 1]
            conved = F.relu(conv(embedded))
            # [batch, num_filters, 1] -> [batch, num_filters]
            pooled = conved.max(dim=2).values
            pooled_outputs.append(pooled)

        # [batch, num_filters * len(filter_sizes)]
        cat = self.dropout(torch.cat(pooled_outputs, dim=1))

        return self.fc(cat)
```

### 22.4.3 训练循环

```python
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids, labels, lengths = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids, labels, lengths = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total
```

---

## 22.5 处理变长序列（padding、pack_padded_sequence）

### 22.5.1 问题：序列长度不一致

在一个 batch 中，不同文本的长度通常不同：
```
句子1: ["great", "movie"]          长度 2
句子2: ["i", "really", "loved", "it"]  长度 4
句子3: ["bad"]                      长度 1
```

PyTorch 的张量要求固定形状，因此需要统一序列长度。

### 22.5.2 Padding

将所有序列填充到 batch 中最长序列的长度：

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch: list[tuple[list[int], int]]):
    """
    DataLoader 的 collate 函数，将变长序列打包为批次
    Args:
        batch: [(token_ids, label), ...]
    Returns:
        padded_ids: [batch_size, max_seq_len]
        labels: [batch_size]
        lengths: [batch_size] 每个序列的实际长度
    """
    # 按序列长度降序排列（pack_padded_sequence 的要求）
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    ids_list, labels = zip(*batch)
    lengths = torch.tensor([len(ids) for ids in ids_list])

    # 转换为张量列表
    ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in ids_list]

    # pad_sequence 自动用 0 填充到最长序列的长度
    # [max_seq_len, batch_size] -> 转置为 [batch_size, max_seq_len]
    padded = pad_sequence(ids_tensors, batch_first=True, padding_value=0)

    labels = torch.tensor(labels, dtype=torch.long)

    return padded, labels, lengths


# 使用 DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
)
```

### 22.5.3 手动 Padding 示例

```python
def pad_sequences(sequences: list[list[int]],
                  max_len: int = None,
                  pad_value: int = 0,
                  truncate: str = 'post',
                  padding: str = 'post') -> torch.Tensor:
    """
    手动填充序列到固定长度

    Args:
        sequences: token ID 列表的列表
        max_len: 目标长度，None 时使用最长序列长度
        pad_value: 填充值（PAD token ID）
        truncate: 截断方向，'pre' 或 'post'
        padding: 填充方向，'pre' 或 'post'
    Returns:
        [len(sequences), max_len] 的张量
    """
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    result = []
    for seq in sequences:
        # 截断
        if truncate == 'post':
            seq = seq[:max_len]
        else:
            seq = seq[-max_len:]

        # 计算需要填充的数量
        pad_len = max_len - len(seq)

        # 填充
        if padding == 'post':
            padded = seq + [pad_value] * pad_len
        else:
            padded = [pad_value] * pad_len + seq

        result.append(padded)

    return torch.tensor(result, dtype=torch.long)


# 示例
seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
padded = pad_sequences(seqs, max_len=6)
print(padded)
# tensor([[ 1,  2,  3,  0,  0,  0],
#         [ 4,  5,  0,  0,  0,  0],
#         [ 6,  7,  8,  9, 10,  0]])
```

### 22.5.4 pack_padded_sequence

Padding 引入了大量无效计算（LSTM 会处理 PAD token）。`pack_padded_sequence` 通过压缩序列避免这一问题：

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 准备数据
embed_dim = 8
hidden_dim = 16

# 模拟嵌入后的张量 [batch=3, max_seq=5, embed_dim=8]
# 实际长度为 [5, 3, 2]
embedded = torch.randn(3, 5, embed_dim)
lengths = torch.tensor([5, 3, 2])

# 打包：将变长序列压缩，LSTM 只处理有效 token
packed = pack_padded_sequence(
    embedded,
    lengths=lengths,
    batch_first=True,
    enforce_sorted=False,  # True 时要求按长度降序
)

print(f"打包后的数据量: {packed.data.shape[0]} 个有效 token")
# 5 + 3 + 2 = 10 个有效 token

lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
packed_output, (hidden, cell) = lstm(packed)

# 解包：还原为填充序列
output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
print(f"输出形状: {output.shape}")
# [3, 5, hidden_dim]
print(f"序列长度: {output_lengths}")
# tensor([5, 3, 2])
```

### 22.5.5 掩码机制（Attention Mask）

在 Transformer 架构中，通常使用掩码矩阵标记哪些位置是有效的：

```python
def create_padding_mask(lengths: torch.Tensor,
                        max_len: int) -> torch.Tensor:
    """
    创建填充掩码，True 表示有效位置，False 表示填充位置
    Args:
        lengths: [batch_size]
        max_len: 最大序列长度
    Returns:
        [batch_size, max_len] 的布尔张量
    """
    batch_size = lengths.size(0)
    # [1, max_len]
    indices = torch.arange(max_len).unsqueeze(0)
    # [batch_size, 1]
    lengths = lengths.unsqueeze(1)
    # 广播比较：[batch_size, max_len]
    mask = indices < lengths
    return mask

# 示例
lengths = torch.tensor([4, 2, 5])
mask = create_padding_mask(lengths, max_len=5)
print(mask)
# tensor([[ True,  True,  True,  True, False],
#         [ True,  True, False, False, False],
#         [ True,  True,  True,  True,  True]])
```

---

## 本章小结

| 概念 | 说明 | 关键工具/方法 |
|------|------|---------------|
| 文本预处理 | 清洗、分词、去停用词、词形还原 | `re`, `nltk`, 自定义流水线 |
| 词汇表 | 词→ID 双向映射，含特殊标记 | `Vocabulary` 类，`Counter` |
| 词嵌入 | 将词映射为低维密集向量 | `nn.Embedding`，GloVe/Word2Vec |
| LSTM 分类器 | 双向 LSTM 捕获序列特征 | `nn.LSTM`，隐藏状态拼接 |
| TextCNN | 多尺度卷积提取 n-gram 特征 | `nn.Conv1d`，全局最大池化 |
| Padding | 统一 batch 内序列长度 | `pad_sequence`，`collate_fn` |
| PackedSequence | 忽略 PAD token 的高效计算 | `pack_padded_sequence`，`pad_packed_sequence` |
| 掩码 | 标记有效/填充位置 | `create_padding_mask`，广播比较 |

**核心流程回顾**：

```
原始文本
  ↓ 清洗（去 HTML、URL、特殊字符）
  ↓ 分词（按空格/正则/NLTK）
  ↓ 去停用词 + 词形还原
  ↓ 词汇表编码（词→ID）
  ↓ Padding（统一长度）
  ↓ nn.Embedding（ID→向量）
  ↓ LSTM/TextCNN
  ↓ 分类器
预测结果
```

---

## 深度学习应用：情感分析项目

### 项目目标

基于 IMDB 电影评论数据集，构建一个二分类情感分析模型，判断评论为**正面（positive）**或**负面（negative）**。

### 项目结构

```
sentiment_analysis/
├── data.py        # 数据加载与预处理
├── model.py       # 模型定义
├── train.py       # 训练脚本
└── predict.py     # 推理脚本
```

### 完整代码

```python
# ============================================================
# sentiment_analysis.py - 基于 IMDB 的情感分析完整代码
# ============================================================

import re
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from collections import Counter
from typing import Optional

# 固定随机种子，保证可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")


# ============================================================
# 1. 数据预处理
# ============================================================

STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'i', 'me', 'my', 'we', 'our',
    'you', 'your', 'it', 'its', 'this', 'that', 'of', 'in',
    'to', 'for', 'on', 'with', 'at', 'by', 'from',
}


def clean_text(text: str) -> str:
    """清洗 IMDB 文本：去 HTML、小写化、标准化"""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)        # 去 HTML 标签
    text = re.sub(r'[^a-z0-9\s]', ' ', text)   # 去非字母数字字符
    text = re.sub(r'\s+', ' ', text).strip()   # 压缩空格
    return text


def tokenize(text: str, remove_stops: bool = True) -> list[str]:
    """分词，可选去停用词"""
    tokens = clean_text(text).split()
    if remove_stops:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class Vocabulary:
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'

    def __init__(self, min_freq: int = 2, max_size: Optional[int] = 25000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        # 先添加特殊标记
        for tok in [self.PAD_TOKEN, self.UNK_TOKEN]:
            idx = len(self.word2idx)
            self.word2idx[tok] = idx
            self.idx2word[idx] = tok

    def build(self, corpus: list[list[str]]):
        counter = Counter(t for tokens in corpus for t in tokens)
        sorted_words = sorted(counter.items(), key=lambda x: -x[1])
        if self.max_size:
            sorted_words = sorted_words[:self.max_size]
        for word, freq in sorted_words:
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        print(f"词汇表构建完成，共 {len(self)} 个词（含特殊标记）")

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.word2idx[self.UNK_TOKEN]
        return [self.word2idx.get(t, unk) for t in tokens]

    def __len__(self) -> int:
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.PAD_TOKEN]


# ============================================================
# 2. 数据集
# ============================================================

class IMDBDataset(Dataset):
    """IMDB 情感分析数据集"""

    def __init__(self, texts: list[str], labels: list[int],
                 vocab: Vocabulary, max_len: int = 256):
        self.labels = labels
        self.max_len = max_len

        # 预处理所有文本
        self.data: list[list[int]] = []
        for text in texts:
            tokens = tokenize(text)
            ids = vocab.encode(tokens)[:max_len]
            self.data.append(ids)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.data[idx], self.labels[idx]


def collate_fn(batch: list[tuple[list[int], int]]):
    """将变长序列打包为批次张量"""
    # 按长度降序排列
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    ids_list, labels = zip(*batch)

    lengths = torch.tensor([max(len(ids), 1) for ids in ids_list])
    ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in ids_list]
    padded = pad_sequence(ids_tensors, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded, labels, lengths


def load_imdb_data(data_dir: str = None):
    """
    加载 IMDB 数据集。
    如果没有本地数据，从 torchtext 下载（需要 torchtext）。
    返回 (train_texts, train_labels, test_texts, test_labels)
    """
    try:
        # 方案A：使用 torchtext（推荐）
        from torchtext.datasets import IMDB
        train_iter = IMDB(split='train')
        test_iter = IMDB(split='test')

        train_texts, train_labels = [], []
        for label, text in train_iter:
            train_texts.append(text)
            train_labels.append(1 if label == 'pos' else 0)

        test_texts, test_labels = [], []
        for label, text in test_iter:
            test_texts.append(text)
            test_labels.append(1 if label == 'pos' else 0)

        return train_texts, train_labels, test_texts, test_labels

    except ImportError:
        # 方案B：生成合成数据（用于演示，无需下载）
        print("警告: torchtext 未安装，使用合成数据演示")
        return _generate_synthetic_data()


def _generate_synthetic_data():
    """生成用于演示的合成情感数据"""
    positive_phrases = [
        "this movie is absolutely amazing and wonderful",
        "great performances by all the actors brilliant film",
        "loved every single minute of this masterpiece",
        "fantastic story with incredible character development",
        "best film I have seen in years highly recommended",
        "outstanding direction and superb cinematography throughout",
    ]
    negative_phrases = [
        "terrible movie complete waste of time and money",
        "awful acting poor direction boring plot throughout",
        "hated this film nothing good about it at all",
        "worst movie ever made incredibly disappointing experience",
        "boring predictable and poorly written script bad film",
        "dreadful performances and painfully slow pacing awful",
    ]

    random.seed(42)
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for _ in range(2000):
        if random.random() > 0.5:
            phrase = random.choice(positive_phrases)
            label = 1
        else:
            phrase = random.choice(negative_phrases)
            label = 0
        # 通过重复和随机拼接增加长度
        n = random.randint(2, 6)
        text = ' '.join(random.choices(
            positive_phrases + negative_phrases, k=n
        ) if label == 1 else [phrase] * n)
        train_texts.append(text)
        train_labels.append(label)

    for _ in range(500):
        if random.random() > 0.5:
            text = random.choice(positive_phrases) * 2
            label = 1
        else:
            text = random.choice(negative_phrases) * 2
            label = 0
        test_texts.append(text)
        test_labels.append(label)

    return train_texts, train_labels, test_texts, test_labels


# ============================================================
# 3. 模型
# ============================================================

class SentimentLSTM(nn.Module):
    """用于情感分析的双向 LSTM 模型"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, input_ids: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        # [batch, seq] -> [batch, seq, embed_dim]
        embedded = self.embedding(input_ids)

        # 打包序列
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, _) = self.lstm(packed)

        # 解包
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # output: [batch, seq_len, hidden*2]

        # 注意力机制：对序列中所有时间步加权平均
        # [batch, seq_len, 1]
        attn_weights = self.attention(output)

        # 掩码：将 PAD 位置设为 -inf
        seq_len = output.size(1)
        indices = torch.arange(seq_len, device=output.device).unsqueeze(0)
        mask = indices >= lengths.unsqueeze(1).to(output.device)
        attn_weights = attn_weights.squeeze(-1)
        attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        # softmax 归一化
        attn_weights = F.softmax(attn_weights, dim=1)  # [batch, seq_len]

        # 加权求和
        context = torch.bmm(
            attn_weights.unsqueeze(1), output
        ).squeeze(1)  # [batch, hidden*2]

        return self.classifier(context)


# ============================================================
# 4. 训练与评估
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for input_ids, labels, lengths in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for input_ids, labels, lengths in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


# ============================================================
# 5. 主程序
# ============================================================

def main():
    # 超参数
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.4
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LR = 1e-3
    MAX_LEN = 256
    MIN_FREQ = 2

    print("=" * 60)
    print("IMDB 情感分析")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    train_texts, train_labels, test_texts, test_labels = load_imdb_data()
    print(f"训练集: {len(train_texts)} 条，测试集: {len(test_texts)} 条")

    # 2. 构建词汇表
    print("\n[2/5] 构建词汇表...")
    vocab = Vocabulary(min_freq=MIN_FREQ, max_size=25000)
    train_tokens = [tokenize(t) for t in train_texts]
    vocab.build(train_tokens)

    # 3. 创建数据集
    print("\n[3/5] 创建数据集...")
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, MAX_LEN)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, MAX_LEN)

    # 划分验证集（取训练集的 10%）
    n_val = len(train_dataset) // 10
    n_train = len(train_dataset) - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    print(f"训练批次: {len(train_loader)}，验证批次: {len(val_loader)}")

    # 4. 创建模型
    print("\n[4/5] 初始化模型...")
    model = SentimentLSTM(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=vocab.pad_idx,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    # 5. 训练
    print("\n[5/5] 开始训练...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>11} "
          f"{'Val Loss':>10} {'Val Acc':>9} {'Time':>8}")
    print("-" * 65)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        elapsed = time.time() - t0

        print(f"{epoch:>6} {train_loss:>12.4f} {train_acc:>10.2%} "
              f"{val_loss:>10.4f} {val_acc:>9.2%} {elapsed:>7.1f}s")

        scheduler.step(val_acc)

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 加载最优模型并评估测试集
    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\n最优验证准确率: {best_val_acc:.2%}")
    print(f"测试集准确率:   {test_acc:.2%}")

    return model, vocab


# ============================================================
# 6. 推理函数
# ============================================================

def predict_sentiment(text: str, model: nn.Module,
                      vocab: Vocabulary, device: torch.device) -> dict:
    """
    对单条文本进行情感预测

    Args:
        text: 原始评论文本
        model: 训练好的模型
        vocab: 词汇表
        device: 推理设备
    Returns:
        包含预测标签和置信度的字典
    """
    model.eval()
    with torch.no_grad():
        tokens = tokenize(text)
        if not tokens:
            return {'label': 'unknown', 'confidence': 0.0, 'probabilities': {}}

        ids = vocab.encode(tokens)[:256]
        input_ids = torch.tensor([ids], dtype=torch.long).to(device)
        lengths = torch.tensor([len(ids)])

        logits = model(input_ids, lengths)
        probs = F.softmax(logits, dim=1).squeeze()

        pred_idx = probs.argmax().item()
        label = 'positive' if pred_idx == 1 else 'negative'
        confidence = probs[pred_idx].item()

    return {
        'label': label,
        'confidence': confidence,
        'probabilities': {
            'negative': probs[0].item(),
            'positive': probs[1].item(),
        }
    }


# ============================================================
# 运行示例
# ============================================================

if __name__ == '__main__':
    model, vocab = main()

    # 推理示例
    print("\n" + "=" * 60)
    print("推理示例")
    print("=" * 60)

    test_reviews = [
        "This movie was absolutely fantastic! The acting was superb "
        "and the story was deeply moving. Highly recommended!",

        "What a terrible waste of time. The plot made no sense, "
        "the acting was awful, and I fell asleep halfway through.",

        "It's an okay film. Nothing special but not terrible either. "
        "Some good moments but overall pretty average.",
    ]

    for review in test_reviews:
        result = predict_sentiment(review, model, vocab, DEVICE)
        print(f"\n评论: {review[:60]}...")
        print(f"预测: {result['label']:>10} | 置信度: {result['confidence']:.1%}")
        print(f"  负面概率: {result['probabilities']['negative']:.1%}  "
              f"正面概率: {result['probabilities']['positive']:.1%}")
```

### 运行预期输出

```
使用设备: cpu
============================================================
IMDB 情感分析
============================================================

[1/5] 加载数据...
训练集: 2000 条，测试集: 500 条

[2/5] 构建词汇表...
词汇表构建完成，共 312 个词（含特殊标记）

[3/5] 创建数据集...
训练批次: 29，验证批次: 4

[4/5] 初始化模型...
模型参数量: 556,034

[5/5] 开始训练...
 Epoch   Train Loss   Train Acc   Val Loss   Val Acc     Time
-----------------------------------------------------------------
     1       0.6821      54.19%     0.6714    56.00%      1.2s
     2       0.6532      61.78%     0.6231    65.00%      1.1s
     ...
    10       0.2341      91.23%     0.3102    87.00%      1.3s

最优验证准确率: 88.00%
测试集准确率:   85.80%
```

---

## 练习题

### 基础题

**1. 词汇表与编码**

给定以下句子列表，完成词汇表构建和编码任务：

```python
sentences = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog outpaces a lazy fox",
    "the dog jumped over the fox",
]
```

a) 构建一个 `Vocabulary` 对象，`min_freq=1`
b) 将第一个句子编码为 ID 序列
c) 验证 `<pad>` 的 ID 是 0，`<unk>` 的 ID 是 1
d) 对一个包含未知词的句子编码，验证未知词被替换为 `<unk>` 的 ID

**2. Padding 与 collate_fn**

实现一个 `collate_fn` 函数，要求：

a) 接受 `[(ids_list, label), ...]` 格式的 batch
b) 使用 `pad_sequence` 将序列填充到 batch 内最长长度
c) 返回 `(padded_ids, labels, lengths)` 三元组
d) 验证 padded_ids 的形状为 `[batch_size, max_seq_len]`

### 中级题

**3. 自注意力改进**

当前 `SentimentLSTM` 模型使用简单线性注意力。将其改造为**缩放点积注意力（Scaled Dot-Product Attention）**：

- 引入查询（Q）、键（K）、值（V）线性层
- 实现 `attention(Q, K, V, mask)` 函数
- 注意力分数计算：$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- 将改进后的注意力集成到模型的 `forward` 方法中

**4. 文本 CNN 与 LSTM 对比实验**

在合成数据集上同时训练 `TextCNN` 和 `SentimentLSTM`，并完成：

a) 记录两个模型的训练时间（每个 epoch）
b) 对比两个模型在测试集上的准确率
c) 分析两种架构各自的优缺点（至少各两点）
d) 绘制两个模型的训练/验证 loss 曲线（使用 matplotlib）

### 进阶题

**5. 预训练词向量集成**

扩展当前项目以支持预训练词向量（GloVe）：

a) 实现 `load_glove_vectors` 函数（参考 22.3.4 节），能够从 GloVe 文本文件中读取向量
b) 修改 `SentimentLSTM.__init__`，添加 `pretrained_vectors` 和 `freeze_embed` 参数
c) 设计对比实验，比较以下三种配置：
   - 随机初始化嵌入层（可训练）
   - 预训练 GloVe 初始化 + 冻结
   - 预训练 GloVe 初始化 + 微调
d) 分析实验结果，讨论预训练词向量对模型性能的影响以及在不同数据规模下的适用场景

---

## 练习答案

### 答案1：词汇表与编码

```python
sentences = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog outpaces a lazy fox",
    "the dog jumped over the fox",
]

# a) 构建词汇表
vocab = Vocabulary(min_freq=1)
corpus = [s.split() for s in sentences]
vocab.build(corpus)
print(f"词汇表大小: {len(vocab)}")

# b) 编码第一个句子
tokens = sentences[0].split()
ids = vocab.encode(tokens)
print(f"编码: {ids}")

# c) 验证特殊标记 ID
assert vocab.word2idx['<pad>'] == 0, "PAD 的 ID 应为 0"
assert vocab.word2idx['<unk>'] == 1, "UNK 的 ID 应为 1"
print("PAD ID:", vocab.word2idx['<pad>'])   # 0
print("UNK ID:", vocab.word2idx['<unk>'])   # 1

# d) 包含未知词的编码
unknown_sentence = ["the", "fox", "ate", "sushi"]
ids_with_unk = vocab.encode(unknown_sentence)
print(f"含未知词的编码: {ids_with_unk}")
# "ate" 和 "sushi" 不在词汇表中，应被编码为 1 (<unk>)
assert 1 in ids_with_unk, "未知词应编码为 UNK ID (1)"
```

### 答案2：Padding 与 collate_fn

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn_solution(batch: list[tuple[list[int], int]]):
    """
    将变长序列批次转换为填充后的张量
    """
    # 按长度降序排列（有利于 pack_padded_sequence）
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    ids_list, labels = zip(*batch)

    # 计算实际长度
    lengths = torch.tensor([max(len(ids), 1) for ids in ids_list])

    # 转换为张量并填充
    ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in ids_list]
    # pad_sequence 默认用 0 填充
    padded_ids = pad_sequence(ids_tensors, batch_first=True, padding_value=0)

    labels = torch.tensor(labels, dtype=torch.long)

    # d) 验证形状
    assert padded_ids.shape == (len(batch), lengths[0].item()), \
        f"形状不匹配: {padded_ids.shape}"

    return padded_ids, labels, lengths


# 测试
sample_batch = [
    ([1, 2, 3, 4, 5], 1),
    ([6, 7], 0),
    ([8, 9, 10], 1),
]

padded, labels, lengths = collate_fn_solution(sample_batch)
print(f"padded_ids 形状: {padded.shape}")   # [3, 5]
print(f"labels: {labels}")                  # tensor([1, 1, 0])
print(f"lengths: {lengths}")               # tensor([5, 3, 2])
print(f"填充张量:\n{padded}")
# tensor([[ 1,  2,  3,  4,  5],
#         [ 8,  9, 10,  0,  0],
#         [ 6,  7,  0,  0,  0]])
```

### 答案3：缩放点积注意力

```python
class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch, seq_len, hidden_dim]
        mask: [batch, seq_len] True 表示有效位置
        """
        Q = self.W_q(x)  # [batch, seq, hidden]
        K = self.W_k(x)
        V = self.W_v(x)

        # 注意力分数 [batch, seq, seq]
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        # 掩码：将 PAD 位置的注意力分数设为 -inf
        if mask is not None:
            # mask [batch, seq] -> [batch, 1, seq]
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # [batch, seq, seq]

        # 对 NaN 处理（全 PAD 行）
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # 加权求和 [batch, seq, hidden]
        context = torch.bmm(attn_weights, V)

        # 取 [CLS] 位置（第一个 token）或平均池化
        output = context.mean(dim=1)  # [batch, hidden]

        return output, attn_weights


class SentimentLSTMWithAttention(nn.Module):
    """带缩放点积注意力的 LSTM 分类器"""

    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2,
                 dropout: float = 0.4, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = ScaledDotProductAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, input_ids: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # 创建掩码
        seq_len = output.size(1)
        indices = torch.arange(seq_len, device=output.device).unsqueeze(0)
        mask = indices < lengths.unsqueeze(1).to(output.device)

        context, _ = self.attention(output, mask)
        return self.classifier(context)
```

### 答案4：TextCNN 与 LSTM 对比实验

```python
import time
import matplotlib.pyplot as plt

def run_comparison_experiment():
    """在合成数据上对比 TextCNN 和 SentimentLSTM"""
    train_texts, train_labels, test_texts, test_labels = _generate_synthetic_data()

    vocab = Vocabulary(min_freq=1)
    vocab.build([tokenize(t) for t in train_texts])

    train_dataset = IMDBDataset(train_texts, train_labels, vocab)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(train_dataset, batch_size=32,
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             shuffle=False, collate_fn=collate_fn)

    models = {
        'LSTM': SentimentLSTM(len(vocab), pad_idx=vocab.pad_idx).to(DEVICE),
        'TextCNN': TextCNN(len(vocab), embed_dim=128,
                           num_classes=2, pad_idx=vocab.pad_idx).to(DEVICE),
    }

    results = {}

    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        history = {'train_loss': [], 'val_loss': [], 'epoch_times': []}

        for epoch in range(5):
            t0 = time.time()

            # TextCNN 的 forward 不需要 lengths 参数
            if name == 'TextCNN':
                model.train()
                epoch_loss = 0
                for ids, labels, lengths in train_loader:
                    ids, labels = ids.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(ids), labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                train_loss = epoch_loss / len(train_loader)
            else:
                train_loss, _ = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

            elapsed = time.time() - t0
            history['train_loss'].append(train_loss)
            history['epoch_times'].append(elapsed)

        results[name] = history

    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, hist in results.items():
        axes[0].plot(hist['train_loss'], label=name)
        axes[1].plot(hist['epoch_times'], label=name)

    axes[0].set_title('训练 Loss 对比')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].set_title('每 Epoch 训练时间对比（秒）')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('时间（s）')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()

    print("\n架构对比分析：")
    print("TextCNN 优点：训练速度快，可并行化；参数量少")
    print("TextCNN 缺点：无法捕获长程依赖；不考虑词序信息")
    print("LSTM 优点：能捕获长程依赖和顺序信息；适合时序任务")
    print("LSTM 缺点：训练较慢（序列依赖）；梯度消失风险")
```

### 答案5：预训练词向量集成

```python
def load_glove_vectors(glove_path: str, vocab: Vocabulary,
                       embed_dim: int = 100) -> torch.Tensor:
    """从 GloVe 文件加载词向量"""
    vectors = np.random.uniform(-0.1, 0.1, (len(vocab), embed_dim)).astype(np.float32)
    vectors[vocab.pad_idx] = np.zeros(embed_dim)

    found = 0
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != embed_dim + 1:
                    continue
                word = parts[0]
                if word in vocab.word2idx:
                    idx = vocab.word2idx[word]
                    vectors[idx] = np.array(parts[1:], dtype=np.float32)
                    found += 1
    except FileNotFoundError:
        print(f"警告: GloVe 文件 {glove_path} 不存在，使用随机初始化")

    print(f"GloVe 覆盖率: {found}/{len(vocab)} ({found/len(vocab):.1%})")
    return torch.FloatTensor(vectors)


class SentimentLSTMWithGloVe(SentimentLSTM):
    """支持预训练词向量的 LSTM 情感分析模型"""

    def __init__(self, *args, pretrained_vectors: torch.Tensor = None,
                 freeze_embed: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if pretrained_vectors is not None:
            self.embedding.weight.data.copy_(pretrained_vectors)
            if freeze_embed:
                self.embedding.weight.requires_grad = False
                print("嵌入层已冻结（不参与训练）")
            else:
                print("使用预训练词向量初始化，嵌入层参与微调")


def glove_experiment(glove_path: str, vocab: Vocabulary,
                     train_loader, test_loader):
    """三种嵌入配置的对比实验"""
    configs = [
        {'name': '随机初始化', 'pretrained': None, 'freeze': False},
        {'name': 'GloVe + 冻结', 'pretrained': True, 'freeze': True},
        {'name': 'GloVe + 微调', 'pretrained': True, 'freeze': False},
    ]

    glove_vecs = load_glove_vectors(glove_path, vocab, embed_dim=100)

    for cfg in configs:
        print(f"\n配置: {cfg['name']}")
        model = SentimentLSTMWithGloVe(
            vocab_size=len(vocab),
            embed_dim=100,
            pad_idx=vocab.pad_idx,
            pretrained_vectors=glove_vecs if cfg['pretrained'] else None,
            freeze_embed=cfg['freeze'],
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(5):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, DEVICE
            )

        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        print(f"  测试准确率: {test_acc:.2%}")

    print("\n结论：")
    print("- 数据量充足时，GloVe + 微调通常表现最佳")
    print("- 数据量较少时，GloVe + 冻结可以防止过拟合")
    print("- 领域特定任务中，随机初始化有时优于通用预训练向量")
```

---

*下一章：第23章 循环神经网络与序列模型*
