# 附录B：PyTorch API参考

本附录汇总了实现和使用Transformer模型时最常用的PyTorch及相关库API，包含参数说明和典型使用示例。

---

## B.1 PyTorch核心API

### B.1.1 `nn.Transformer`

完整的Encoder-Decoder Transformer模型。

**签名**

```python
torch.nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',        # 'relu' 或 'gelu'
    custom_encoder=None,
    custom_decoder=None,
    layer_norm_eps=1e-5,
    batch_first=False,        # False: (seq, batch, feature)
    norm_first=False,         # True: Pre-LN；False: Post-LN
    bias=True,
    device=None,
    dtype=None
)
```

**主要参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `d_model` | int | 512 | 模型维度（embedding维度） |
| `nhead` | int | 8 | 多头注意力的头数，须整除 `d_model` |
| `num_encoder_layers` | int | 6 | Encoder层数 |
| `num_decoder_layers` | int | 6 | Decoder层数 |
| `dim_feedforward` | int | 2048 | FFN隐层维度 |
| `dropout` | float | 0.1 | Dropout比例 |
| `activation` | str/callable | `'relu'` | 激活函数 |
| `batch_first` | bool | False | True时输入形状为 `(batch, seq, feature)` |
| `norm_first` | bool | False | True时使用Pre-LayerNorm |

**前向方法**

```python
output = transformer(
    src,                    # (S, N, E) 或 (N, S, E) if batch_first
    tgt,                    # (T, N, E) 或 (N, T, E) if batch_first
    src_mask=None,          # (S, S) 加法mask
    tgt_mask=None,          # (T, T) 加法mask
    memory_mask=None,       # (T, S) 加法mask
    src_key_padding_mask=None,    # (N, S) bool mask
    tgt_key_padding_mask=None,    # (N, T) bool mask
    memory_key_padding_mask=None  # (N, S) bool mask
)
```

**使用示例**

```python
import torch
import torch.nn as nn

model = nn.Transformer(
    d_model=256,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=512,
    dropout=0.1,
    batch_first=True
)

batch_size, src_len, tgt_len, d_model = 4, 10, 8, 256
src = torch.randn(batch_size, src_len, d_model)
tgt = torch.randn(batch_size, tgt_len, d_model)

# 生成因果mask（防止decoder看到未来token）
tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len)

output = model(src, tgt, tgt_mask=tgt_mask)
# output shape: (batch_size, tgt_len, d_model)
print(output.shape)  # torch.Size([4, 8, 256])
```

---

### B.1.2 `nn.TransformerEncoderLayer` 和 `nn.TransformerEncoder`

**`nn.TransformerEncoderLayer` 签名**

```python
torch.nn.TransformerEncoderLayer(
    d_model,
    nhead,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',
    layer_norm_eps=1e-5,
    batch_first=False,
    norm_first=False,
    bias=True,
    device=None,
    dtype=None
)
```

**`nn.TransformerEncoder` 签名**

```python
torch.nn.TransformerEncoder(
    encoder_layer,      # TransformerEncoderLayer实例
    num_layers,         # 层数
    norm=None,          # 可选的最终LayerNorm
    enable_nested_tensor=True,
    mask_check=True
)
```

**前向方法**

```python
output = encoder(
    src,                           # (S, N, E) 或 (N, S, E)
    mask=None,                     # (S, S) 注意力mask
    src_key_padding_mask=None,     # (N, S) padding mask
    is_causal=False
)
```

**使用示例**

```python
import torch
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True,
    norm_first=True    # Pre-LN，训练更稳定
)

encoder = nn.TransformerEncoder(
    encoder_layer=encoder_layer,
    num_layers=6,
    norm=nn.LayerNorm(512)
)

src = torch.randn(2, 20, 512)  # (batch, seq_len, d_model)

# padding mask：True的位置被忽略
src_key_padding_mask = torch.zeros(2, 20, dtype=torch.bool)
src_key_padding_mask[0, 15:] = True  # 第一个样本后5个token是padding

output = encoder(src, src_key_padding_mask=src_key_padding_mask)
print(output.shape)  # torch.Size([2, 20, 512])
```

---

### B.1.3 `nn.TransformerDecoderLayer` 和 `nn.TransformerDecoder`

**`nn.TransformerDecoderLayer` 签名**

```python
torch.nn.TransformerDecoderLayer(
    d_model,
    nhead,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',
    layer_norm_eps=1e-5,
    batch_first=False,
    norm_first=False,
    bias=True,
    device=None,
    dtype=None
)
```

**`nn.TransformerDecoder` 签名**

```python
torch.nn.TransformerDecoder(
    decoder_layer,      # TransformerDecoderLayer实例
    num_layers,
    norm=None
)
```

**前向方法**

```python
output = decoder(
    tgt,                           # (T, N, E) 或 (N, T, E)
    memory,                        # 来自Encoder的输出
    tgt_mask=None,                 # (T, T) 因果mask
    memory_mask=None,              # (T, S) cross-attention mask
    tgt_key_padding_mask=None,     # (N, T)
    memory_key_padding_mask=None,  # (N, S)
    tgt_is_causal=False,
    memory_is_causal=False
)
```

**使用示例**

```python
decoder_layer = nn.TransformerDecoderLayer(
    d_model=512, nhead=8, batch_first=True
)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

memory = torch.randn(2, 20, 512)   # Encoder输出
tgt = torch.randn(2, 15, 512)      # Decoder输入

tgt_mask = nn.Transformer.generate_square_subsequent_mask(15)
output = decoder(tgt, memory, tgt_mask=tgt_mask, tgt_is_causal=True)
print(output.shape)  # torch.Size([2, 15, 512])
```

---

### B.1.4 `nn.MultiheadAttention`

独立的多头注意力模块，可灵活用于自注意力或交叉注意力。

**签名**

```python
torch.nn.MultiheadAttention(
    embed_dim,              # 输入/输出维度
    num_heads,              # 注意力头数
    dropout=0.0,
    bias=True,
    add_bias_kv=False,
    add_zero_attn=False,
    kdim=None,              # Key维度，默认等于embed_dim
    vdim=None,              # Value维度，默认等于embed_dim
    batch_first=False,
    device=None,
    dtype=None
)
```

**前向方法**

```python
attn_output, attn_weights = mha(
    query,                          # (T, N, E) 或 (N, T, E)
    key,                            # (S, N, E) 或 (N, S, E)
    value,                          # (S, N, E) 或 (N, S, E)
    key_padding_mask=None,          # (N, S) bool
    need_weights=True,              # 是否返回注意力权重
    attn_mask=None,                 # (T, S) 或 (N*heads, T, S) 加法mask
    average_attn_weights=True,      # 对多头取平均
    is_causal=False
)
# attn_output: (T, N, E) 或 (N, T, E)
# attn_weights: (N, T, S) 或 None
```

**主要参数**

| 参数 | 说明 |
|------|------|
| `embed_dim` | 模型维度，每头维度为 `embed_dim // num_heads` |
| `num_heads` | 头数，须整除 `embed_dim` |
| `kdim` / `vdim` | 跨模态注意力时可设为不同维度 |
| `batch_first` | True时 `(batch, seq, feature)` 格式 |

**使用示例**

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    batch_first=True
)

# 自注意力
x = torch.randn(2, 10, 512)
out, weights = mha(x, x, x)
print(out.shape)      # torch.Size([2, 10, 512])
print(weights.shape)  # torch.Size([2, 10, 10])

# 交叉注意力（query来自decoder，key/value来自encoder）
query = torch.randn(2, 5, 512)
kv = torch.randn(2, 10, 512)
out, weights = mha(query, kv, kv)
print(out.shape)  # torch.Size([2, 5, 512])

# 因果mask（下三角）
causal_mask = torch.triu(torch.ones(5, 5), diagonal=1).bool()
out, _ = mha(query, query, query, attn_mask=causal_mask)
```

---

## B.2 常用层

### B.2.1 `nn.Embedding`

**签名**

```python
torch.nn.Embedding(
    num_embeddings,     # 词表大小
    embedding_dim,      # embedding维度
    padding_idx=None,   # 该索引的梯度设为0
    max_norm=None,      # L2范数上限
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    _weight=None,
    _freeze=False,
    device=None,
    dtype=None
)
```

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| `num_embeddings` | int | 词表大小（token数量） |
| `embedding_dim` | int | 每个token的向量维度 |
| `padding_idx` | int | 该索引位置的embedding恒为0且不更新梯度 |
| `max_norm` | float | 若设置，embedding向量L2范数超过此值时被归一化 |
| `scale_grad_by_freq` | bool | 梯度按token频率缩放（稀疏数据有用） |

**使用示例**

```python
import torch
import torch.nn as nn

vocab_size, d_model = 10000, 512
embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

token_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
embedded = embedding(token_ids)
print(embedded.shape)  # torch.Size([2, 5, 512])

# 从预训练权重初始化
pretrained_weights = torch.randn(vocab_size, d_model)
embedding = nn.Embedding.from_pretrained(
    pretrained_weights,
    freeze=False,        # False允许继续微调
    padding_idx=0
)
```

---

### B.2.2 `nn.LayerNorm`

**签名**

```python
torch.nn.LayerNorm(
    normalized_shape,   # int 或 List[int]，对最后N维归一化
    eps=1e-5,
    elementwise_affine=True,   # 是否有可学习的 gamma/beta
    bias=True,
    device=None,
    dtype=None
)
```

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| `normalized_shape` | int/list | 归一化的维度，通常为 `d_model` |
| `eps` | float | 数值稳定的小量 |
| `elementwise_affine` | bool | True时有可学习缩放参数 `weight`（gamma）和偏置 `bias`（beta） |

**使用示例**

```python
layer_norm = nn.LayerNorm(512)

x = torch.randn(2, 10, 512)
out = layer_norm(x)
print(out.shape)   # torch.Size([2, 10, 512])
print(out.mean(-1).abs().max().item())   # 接近0
print(out.std(-1).mean().item())         # 接近1

# 对多维归一化
layer_norm_2d = nn.LayerNorm([10, 512])
```

---

### B.2.3 `nn.Linear`

**签名**

```python
torch.nn.Linear(
    in_features,        # 输入特征维度
    out_features,       # 输出特征维度
    bias=True,
    device=None,
    dtype=None
)
```

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| `in_features` | int | 输入向量维度 |
| `out_features` | int | 输出向量维度 |
| `bias` | bool | 是否添加偏置项 |

计算：`y = x @ W^T + b`，权重形状为 `(out_features, in_features)`。

**使用示例**

```python
# FFN中的两个线性层
linear1 = nn.Linear(512, 2048)
linear2 = nn.Linear(2048, 512, bias=False)

x = torch.randn(2, 10, 512)
out = linear2(torch.relu(linear1(x)))
print(out.shape)  # torch.Size([2, 10, 512])

# 输出投影（无偏置节省参数）
proj = nn.Linear(512, 10000, bias=False)
logits = proj(out)
print(logits.shape)  # torch.Size([2, 10, 10000])
```

---

### B.2.4 `nn.Dropout`

**签名**

```python
torch.nn.Dropout(p=0.5, inplace=False)
```

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| `p` | float | 置零概率，训练时随机将元素置0并缩放 `1/(1-p)` |
| `inplace` | bool | True时原地修改（节省内存，但影响反向传播） |

**注意**：调用 `model.eval()` 后Dropout自动禁用；`model.train()` 恢复。

**使用示例**

```python
dropout = nn.Dropout(p=0.1)

x = torch.randn(2, 10, 512)
# 训练模式
dropout.train()
out = dropout(x)   # 约10%的元素被置0

# 推理模式
dropout.eval()
out = dropout(x)   # 恒等变换，out == x
```

---

### B.2.5 常用层参数汇总

| 层 | 关键参数 | 输入形状 | 输出形状 |
|----|---------|---------|---------|
| `nn.Embedding` | `num_embeddings`, `embedding_dim` | `(*, )` int | `(*, embedding_dim)` |
| `nn.LayerNorm` | `normalized_shape`, `eps` | `(*, normalized_shape)` | 同输入 |
| `nn.Linear` | `in_features`, `out_features` | `(*, in_features)` | `(*, out_features)` |
| `nn.Dropout` | `p` | 任意 | 同输入 |
| `nn.MultiheadAttention` | `embed_dim`, `num_heads` | `(T, N, E)` | `(T, N, E)` |

---

## B.3 Hugging Face Transformers

### B.3.1 `AutoModel` 和 `AutoTokenizer`

Auto类根据模型名称或路径自动选择正确的模型/分词器架构。

**`AutoTokenizer`**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path,   # 模型名或本地路径
    cache_dir=None,           # 缓存目录
    use_fast=True,            # 使用Rust实现的fast tokenizer
    token=None,               # HuggingFace Hub token
    trust_remote_code=False,  # 是否信任远程代码
    **kwargs
)

# 编码
encoding = tokenizer(
    text,                    # str 或 List[str]
    padding=True,            # 'max_length', True, False
    truncation=True,         # 是否截断
    max_length=512,
    return_tensors='pt',     # 'pt', 'tf', 'np'
    add_special_tokens=True
)
# encoding.input_ids: (batch, seq_len)
# encoding.attention_mask: (batch, seq_len)
# encoding.token_type_ids: (batch, seq_len) [BERT]
```

**`AutoModel`**

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    cache_dir=None,
    ignore_mismatched_sizes=False,
    torch_dtype=None,           # torch.float16 等
    device_map=None,            # 'auto', 'cpu', 'cuda'
    load_in_8bit=False,         # 需要bitsandbytes
    load_in_4bit=False,
    **kwargs
)

outputs = model(**encoding)
# outputs.last_hidden_state: (batch, seq_len, hidden_size)
# outputs.pooler_output: (batch, hidden_size) [仅部分模型]
```

---

### B.3.2 `from_pretrained` 详解

```python
# 加载预训练模型并适配下游任务
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,               # 自定义分类数
    torch_dtype=torch.float16,  # 半精度节省显存
    device_map='auto'           # 自动分配多GPU
)

# 保存模型
model.save_pretrained('./my_model')
tokenizer.save_pretrained('./my_model')

# 从本地加载
model = AutoModelForSequenceClassification.from_pretrained('./my_model')
```

**常用任务后缀**

| 类名后缀 | 任务 | 典型输出 |
|---------|------|---------|
| `ForSequenceClassification` | 文本分类 | logits `(batch, num_labels)` |
| `ForTokenClassification` | NER、序列标注 | logits `(batch, seq, num_labels)` |
| `ForQuestionAnswering` | 抽取式问答 | start/end logits |
| `ForMaskedLM` | MLM预训练/填空 | logits `(batch, seq, vocab)` |
| `ForCausalLM` | 自回归生成 | logits `(batch, seq, vocab)` |
| `ForSeq2SeqLM` | 翻译、摘要 | logits `(batch, seq, vocab)` |

---

### B.3.3 常用模型类

**`BertModel`**

```python
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, world!", return_tensors='pt')
outputs = model(**inputs)

# last_hidden_state: (1, seq_len, 768)
# pooler_output: (1, 768)  [CLS token经过线性+tanh]
print(outputs.last_hidden_state.shape)
print(outputs.pooler_output.shape)
```

**`GPT2Model` 和 `GPT2LMHeadModel`**

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

inputs = tokenizer("The future of AI is", return_tensors='pt')
outputs = model(**inputs, labels=inputs['input_ids'])

loss = outputs.loss           # 语言模型损失
logits = outputs.logits       # (1, seq_len, vocab_size=50257)

# 生成文本
generated = model.generate(
    inputs['input_ids'],
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_p=0.95
)
print(tokenizer.decode(generated[0], skip_special_tokens=True))
```

**`T5ForConditionalGeneration`**

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

input_text = "translate English to French: Hello, how are you?"
inputs = tokenizer(input_text, return_tensors='pt')

outputs = model.generate(
    inputs['input_ids'],
    max_length=50,
    num_beams=4,         # beam search
    early_stopping=True
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

### B.3.4 Trainer API

`Trainer` 封装了训练循环、评估、混合精度、分布式训练等细节。

**`TrainingArguments`**

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=5e-5,
    lr_scheduler_type='linear',
    evaluation_strategy='epoch',  # 'no', 'steps', 'epoch'
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    fp16=True,                    # 混合精度
    dataloader_num_workers=4,
    logging_dir='./logs',
    report_to='tensorboard'       # 'wandb', 'none'
)
```

**`Trainer`**

```python
from transformers import Trainer
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.save_model('./final_model')
```

---

## B.4 Hugging Face Datasets

### B.4.1 `load_dataset`

**签名**

```python
from datasets import load_dataset

dataset = load_dataset(
    path,               # 数据集名称或本地路径
    name=None,          # 子集名，如 'wikitext-2-raw-v1'
    split=None,         # 'train', 'test', 'validation', 'train[:80%]'
    data_files=None,    # 本地文件路径字典
    cache_dir=None,
    num_proc=None,      # 并行处理进程数
    trust_remote_code=False,
    **kwargs
)
```

**使用示例**

```python
from datasets import load_dataset

# 加载公开数据集
dataset = load_dataset('imdb')
# DatasetDict({'train': Dataset, 'test': Dataset, 'unsupervised': Dataset})

# 加载特定split
train_data = load_dataset('imdb', split='train')
# Dataset({features: ['text', 'label'], num_rows: 25000})

# 按比例划分
splits = load_dataset('imdb', split=[
    'train[:80%]', 'train[80%:]'
])

# 加载本地CSV
dataset = load_dataset('csv', data_files={
    'train': 'train.csv',
    'test': 'test.csv'
})

# 加载本地JSON
dataset = load_dataset('json', data_files='data.jsonl')
```

---

### B.4.2 `map` 和 `filter`

**`map`** — 对数据集每条样本应用函数

```python
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,          # 批量处理（速度更快）
    batch_size=1000,
    num_proc=4,            # 多进程
    remove_columns=['text'],  # 删除原始列
    desc='Tokenizing'
)

# 添加新列
def add_length(example):
    return {'length': len(example['text'].split())}

dataset = dataset.map(add_length)
```

**`filter`** — 过滤不满足条件的样本

```python
# 只保留正面评论
positive_dataset = dataset.filter(
    lambda x: x['label'] == 1
)

# 过滤过短文本
filtered = dataset.filter(
    lambda x: len(x['text'].split()) > 10,
    num_proc=4
)

print(f"原始大小: {len(dataset)}")
print(f"过滤后: {len(filtered)}")
```

**其他常用操作**

```python
# 排序
sorted_dataset = dataset.sort('length')

# 打乱
shuffled = dataset.shuffle(seed=42)

# 选取子集
small_dataset = dataset.select(range(1000))

# 重命名列
dataset = dataset.rename_column('label', 'labels')

# 设置格式
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
```

---

### B.4.3 `DatasetDict`

```python
from datasets import DatasetDict, Dataset

# 手动创建DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

# 访问split
train = dataset_dict['train']

# 对所有split应用map
tokenized = dataset_dict.map(tokenize_function, batched=True)

# 保存和加载到磁盘
dataset_dict.save_to_disk('./my_dataset')
loaded = DatasetDict.load_from_disk('./my_dataset')
```

---

### B.4.4 常用数据集

| 数据集 | 加载方式 | 任务 | 规模 |
|-------|---------|------|------|
| IMDB | `load_dataset('imdb')` | 情感分析 | 50K |
| SST-2 | `load_dataset('glue', 'sst2')` | 情感分析 | 67K |
| SQuAD | `load_dataset('squad')` | 问答 | 88K |
| WikiText-103 | `load_dataset('wikitext', 'wikitext-103-raw-v1')` | 语言模型 | 100M tokens |
| WMT14 en-de | `load_dataset('wmt14', 'de-en')` | 翻译 | 4.5M |
| CNN/DailyMail | `load_dataset('cnn_dailymail', '3.0.0')` | 摘要 | 300K |
| MNLI | `load_dataset('glue', 'mnli')` | 自然语言推断 | 433K |
| C4 | `load_dataset('c4', 'en')` | 预训练语料 | ~750GB |

---

## B.5 其他有用的库

### B.5.1 einops

`einops` 提供直观的张量重排和操作，语法类似爱因斯坦求和符号。

**安装**

```bash
pip install einops
```

**`rearrange`** — 张量重排

```python
from einops import rearrange

x = torch.randn(2, 8, 512)   # (batch, seq, d_model)

# 分离多头
# d_model = num_heads * head_dim = 8 * 64
x_heads = rearrange(x, 'b s (h d) -> b h s d', h=8)
print(x_heads.shape)   # (2, 8, 8, 64)

# 合并多头
x_merged = rearrange(x_heads, 'b h s d -> b s (h d)')
print(x_merged.shape)  # (2, 8, 512)

# 转置batch和seq
x_T = rearrange(x, 'b s d -> s b d')

# 展平batch和seq
x_flat = rearrange(x, 'b s d -> (b s) d')
print(x_flat.shape)    # (16, 512)

# 图像patches（Vision Transformer预处理）
img = torch.randn(4, 3, 224, 224)   # (batch, C, H, W)
patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
print(patches.shape)   # (4, 196, 768)
```

**`repeat`** — 张量重复扩展

```python
from einops import repeat

# 扩展位置编码
pos_enc = torch.randn(1, 100, 512)  # (1, seq, d)
pos_enc_batch = repeat(pos_enc, '1 s d -> b s d', b=4)
print(pos_enc_batch.shape)  # (4, 100, 512)

# 复制cls token
cls_token = torch.randn(1, 1, 512)
cls_tokens = repeat(cls_token, '1 1 d -> b 1 d', b=4)
print(cls_tokens.shape)  # (4, 1, 512)
```

**`reduce`** — 带操作的降维

```python
from einops import reduce

x = torch.randn(4, 10, 512)

# 全局平均池化
mean_pool = reduce(x, 'b s d -> b d', 'mean')
print(mean_pool.shape)  # (4, 512)

# 最大池化
max_pool = reduce(x, 'b s d -> b d', 'max')
```

---

### B.5.2 accelerate

`accelerate` 让PyTorch代码无缝运行在CPU、单GPU、多GPU或TPU上，只需少量修改。

**安装**

```bash
pip install accelerate
```

**基本使用**

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision='fp16',     # 'no', 'fp16', 'bf16'
    gradient_accumulation_steps=4,
    log_with='tensorboard',
    project_dir='./logs'
)

# 准备所有对象
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

# 训练循环（与普通PyTorch几乎相同）
for batch in train_dataloader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

# 保存
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained('./model', save_function=accelerator.save)
```

**配置命令**

```bash
# 交互式配置（单机多卡、DeepSpeed等）
accelerate config

# 启动训练脚本
accelerate launch --num_processes 4 train.py
```

---

### B.5.3 bitsandbytes

`bitsandbytes` 提供8-bit和4-bit量化，大幅降低模型显存需求。

**安装**

```bash
pip install bitsandbytes
```

**8-bit量化加载**

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit量化（约节省50%显存）
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    load_in_8bit=True,
    device_map='auto'
)
print(model.get_memory_footprint() / 1e9, 'GB')
```

**4-bit量化（QLoRA）**

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,   # 嵌套量化，进一步节省显存
    bnb_4bit_quant_type='nf4',        # 'fp4' 或 'nf4'（推荐）
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    quantization_config=bnb_config,
    device_map='auto'
)
```

**常用量化参数**

| 参数 | 说明 |
|------|------|
| `load_in_8bit` | LLM.int8()量化，推理时动态反量化 |
| `load_in_4bit` | 4-bit量化，通常配合LoRA使用 |
| `bnb_4bit_quant_type` | `nf4`（正态浮点4位，推荐）或 `fp4` |
| `bnb_4bit_compute_dtype` | 计算时的精度，推荐 `bfloat16` |
| `bnb_4bit_use_double_quant` | 对量化常数再量化，额外节省约0.5 bit/参数 |

---

### B.5.4 peft（Parameter-Efficient Fine-Tuning）

`peft` 提供LoRA、Prefix Tuning、Prompt Tuning等参数高效微调方法。

**安装**

```bash
pip install peft
```

**LoRA微调**

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # 任务类型
    r=16,                           # LoRA秩（低秩矩阵的维度）
    lora_alpha=32,                  # 缩放系数，实际lr = lora_alpha/r
    target_modules=['q_proj', 'v_proj'],  # 应用LoRA的模块
    lora_dropout=0.1,
    bias='none',                    # 'none', 'all', 'lora_only'
    inference_mode=False
)

# 将预训练模型转换为PEFT模型
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

**LoRA参数说明**

| 参数 | 说明 |
|------|------|
| `r` | 低秩分解的秩，越大表达能力越强，参数越多（通常4-64） |
| `lora_alpha` | 缩放因子，控制LoRA更新的幅度 |
| `target_modules` | 应用LoRA的层名，通常是注意力的Q/V投影 |
| `lora_dropout` | LoRA层的Dropout |
| `bias` | 是否训练偏置 |

**保存和加载LoRA权重**

```python
# 只保存LoRA权重（很小，通常几十MB）
peft_model.save_pretrained('./lora_weights')

# 加载并合并到基础模型
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
model = PeftModel.from_pretrained(base_model, './lora_weights')

# 将LoRA权重合并到基础模型（加速推理）
merged_model = model.merge_and_unload()
merged_model.save_pretrained('./merged_model')
```

**4-bit + LoRA（QLoRA完整示例）**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 4-bit量化加载
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    quantization_config=bnb_config,
    device_map='auto'
)

# 2. 为kbit训练准备（启用梯度检查点、转换LayerNorm精度）
model = prepare_model_for_kbit_training(model)

# 3. 添加LoRA适配器
lora_config = LoraConfig(
    r=64, lora_alpha=16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05, bias='none',
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 约0.5%的参数可训练，在消费级GPU上可微调70亿参数模型
```

---

## 快速查阅索引

| API | 所属库 | 用途 |
|-----|--------|------|
| `nn.Transformer` | PyTorch | 完整Encoder-Decoder Transformer |
| `nn.TransformerEncoder` | PyTorch | 仅Encoder（BERT风格） |
| `nn.TransformerDecoder` | PyTorch | 仅Decoder（GPT风格） |
| `nn.MultiheadAttention` | PyTorch | 独立多头注意力模块 |
| `nn.Embedding` | PyTorch | Token/位置嵌入 |
| `nn.LayerNorm` | PyTorch | 层归一化 |
| `AutoModel` | Transformers | 自动加载预训练模型 |
| `AutoTokenizer` | Transformers | 自动加载分词器 |
| `Trainer` | Transformers | 封装训练循环 |
| `load_dataset` | Datasets | 加载公开/本地数据集 |
| `rearrange` | einops | 张量直觉式重排 |
| `Accelerator` | accelerate | 多硬件无缝训练 |
| `BitsAndBytesConfig` | bitsandbytes | 量化配置 |
| `LoraConfig` | peft | LoRA参数高效微调 |
