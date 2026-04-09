# 第24章：完整项目实战

## 学习目标

完成本章学习后，你将能够：

1. 从零开始训练一个小型GPT语言模型，理解完整的训练流程
2. 掌握数据准备和预处理的完整流程，包括Tokenizer训练
3. 理解分布式训练的基本配置和梯度累积技术
4. 熟练使用Hugging Face生态系统，包括Hub上传和Trainer使用
5. 能够将训练好的模型部署为生产级API服务

---

## 24.1 项目概述

本章将带你完成一个端到端的GPT模型训练和部署项目。我们将训练一个拥有约1.2亿参数的小型GPT模型，使用公开数据集，最终将其部署为一个支持流式响应的API服务。

### 24.1.1 项目目标

我们要构建的是一个**词级小型GPT**（GPT-Small），规格如下：

| 参数 | 值 |
|------|-----|
| 词汇表大小 | 50,257 (GPT-2标准) |
| 上下文长度 | 1024 |
| 嵌入维度 | 768 |
| 注意力头数 | 12 |
| Transformer层数 | 12 |
| 总参数量 | ~117M |

这个规模的模型既有足够的能力展示真实的语言建模效果，又可以在单张消费级GPU（如RTX 3090/4090）上完成训练。

### 24.1.2 数据集选择

我们提供两个难度级别的数据集选项：

**入门级：Shakespeare数据集**
- 大小：约1MB
- 特点：单一风格，容易收敛，适合快速验证流程
- 下载：直接从网络获取
- 预计训练时间：单GPU约30分钟可看到效果

**标准级：WikiText-103**
- 大小：约500MB（压缩后）
- 特点：多样化的英文维基百科文章
- 特点：真实世界文本，训练结果更有实用价值
- 预计训练时间：单GPU约2-3天完整训练

本章以WikiText-103为主，代码同时支持两种数据集。

### 24.1.3 硬件需求估算

训练117M参数模型的资源需求：

**内存估算：**

$$\text{模型参数内存} = N_{params} \times 4 \text{ bytes} = 117M \times 4 = 468 \text{ MB}$$

$$\text{优化器状态(Adam)} = N_{params} \times 8 \text{ bytes} = 936 \text{ MB}$$

$$\text{梯度} = N_{params} \times 4 \text{ bytes} = 468 \text{ MB}$$

$$\text{总计（不含激活值）} \approx 1.9 \text{ GB}$$

激活值内存与批次大小和序列长度成正比。使用梯度检查点可以显著降低激活值内存。

**推荐配置：**

| 配置 | GPU | 批次大小 | 预计训练时间 |
|------|-----|----------|------------|
| 最低 | RTX 3080 (10GB) | 4 | ~4天 |
| 推荐 | RTX 3090 (24GB) | 16 | ~2天 |
| 理想 | A100 (40GB) | 32 | ~16小时 |

### 24.1.4 项目结构规划

```
gpt-project/
├── config.py          # 所有超参数和路径配置
├── data.py            # 数据集下载、Tokenizer训练、Dataset类
├── model.py           # GPT模型定义
├── train.py           # 训练主循环
├── serve.py           # FastAPI推理服务
├── requirements.txt   # 依赖列表
├── Dockerfile         # 容器化配置
├── docker-compose.yml # 服务编排
├── data/              # 数据存放目录
│   ├── raw/           # 原始数据
│   └── processed/     # 处理后数据
├── checkpoints/       # 模型检查点
├── logs/              # 训练日志
└── hf_model/          # Hugging Face格式模型
```

---

## 24.2 数据准备

数据质量直接决定模型质量。本节涵盖从原始文本到可训练数据集的完整流程。

### 24.2.1 数据集下载和清洗

```python
# data.py - 数据下载和清洗部分

import os
import re
import json
import hashlib
import requests
from pathlib import Path
from typing import Optional, List, Iterator
from datasets import load_dataset
import unicodedata
from tqdm import tqdm


def download_shakespeare(data_dir: str = "data/raw") -> str:
    """下载Shakespeare数据集（入门级）"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    save_path = Path(data_dir) / "shakespeare.txt"

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    if not save_path.exists():
        print(f"下载Shakespeare数据集...")
        response = requests.get(url)
        response.raise_for_status()
        save_path.write_text(response.text, encoding='utf-8')
        print(f"已保存到 {save_path}，大小：{save_path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"Shakespeare数据集已存在：{save_path}")

    return str(save_path)


def download_wikitext103(data_dir: str = "data/raw") -> dict:
    """下载WikiText-103数据集（标准级）"""
    print("下载WikiText-103数据集（使用Hugging Face datasets）...")

    # 使用HF datasets库下载
    dataset = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        cache_dir=data_dir
    )

    print(f"训练集大小：{len(dataset['train'])} 条")
    print(f"验证集大小：{len(dataset['validation'])} 条")
    print(f"测试集大小：{len(dataset['test'])} 条")

    return dataset


def clean_text(text: str) -> str:
    """
    文本清洗函数

    处理步骤：
    1. Unicode标准化（将全角字符转为半角等）
    2. 去除维基百科特有的标记
    3. 合并多余空白字符
    4. 过滤过短的段落
    """
    # 1. Unicode标准化
    text = unicodedata.normalize('NFKC', text)

    # 2. 去除维基百科章节标题标记（= Title =）
    text = re.sub(r'^\s*=+\s*.+\s*=+\s*$', '', text, flags=re.MULTILINE)

    # 3. 去除多余的空白行（保留最多一个空行）
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 4. 去除行首尾空白
    lines = [line.strip() for line in text.split('\n')]

    # 5. 过滤过短的行（少于10个字符的非空行）
    lines = [
        line for line in lines
        if len(line) == 0 or len(line) >= 10
    ]

    return '\n'.join(lines).strip()


def prepare_wikitext_corpus(
    dataset,
    output_path: str = "data/processed/corpus.txt",
    max_docs: Optional[int] = None
) -> str:
    """
    将WikiText数据集处理为单个语料文件

    Args:
        dataset: HF datasets对象
        output_path: 输出文件路径
        max_docs: 最大文档数量（None表示全部）

    Returns:
        输出文件路径
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    total_docs = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for split in ['train', 'validation', 'test']:
            data = dataset[split]

            # 合并相邻文本行，形成完整文档
            current_doc = []

            for i, example in enumerate(tqdm(data, desc=f"处理{split}")):
                if max_docs and total_docs >= max_docs:
                    break

                text = example['text']

                # WikiText用空行分隔文档
                if text.strip() == '':
                    if current_doc:
                        doc_text = clean_text('\n'.join(current_doc))
                        if len(doc_text) > 100:  # 过滤过短文档
                            f.write(doc_text + '\n\n')
                            total_chars += len(doc_text)
                            total_docs += 1
                        current_doc = []
                else:
                    current_doc.append(text)

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"语料处理完成：{total_docs} 篇文档，{total_chars/1e6:.1f}M 字符，{size_mb:.1f} MB")

    return output_path
```

### 24.2.2 Tokenizer训练（BPE）

字节对编码（Byte Pair Encoding，BPE）是现代语言模型最常用的分词算法。它通过迭代合并最频繁出现的字符对来构建词汇表。

**BPE算法核心思想：**

$$\text{初始化：} V = \{\text{所有Unicode字符}\}$$

$$\text{迭代：} \text{找到最频繁的字符对 } (a, b) \text{，合并为新token } ab$$

$$\text{直到：} |V| = \text{目标词汇表大小}$$

```python
# data.py - Tokenizer训练部分

from tokenizers import Tokenizer, trainers, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers import normalizers
from transformers import PreTrainedTokenizerFast
import os


def train_bpe_tokenizer(
    corpus_path: str,
    vocab_size: int = 50257,
    save_dir: str = "data/tokenizer",
    special_tokens: Optional[List[str]] = None
) -> PreTrainedTokenizerFast:
    """
    训练BPE Tokenizer

    Args:
        corpus_path: 语料文件路径
        vocab_size: 目标词汇表大小（50257与GPT-2一致）
        save_dir: 保存目录
        special_tokens: 特殊token列表

    Returns:
        HuggingFace格式的Tokenizer
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>"]

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 1. 初始化BPE模型
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

    # 2. 配置预分词器（按空白和标点分割）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 3. 配置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 4. 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,           # 最少出现2次才纳入词汇表
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 5. 从文件训练
    print(f"开始训练BPE Tokenizer，目标词汇量：{vocab_size}...")

    # 流式读取大文件
    def corpus_iterator(file_path: str, chunk_size: int = 1000):
        """生成器：批量读取语料"""
        with open(file_path, 'r', encoding='utf-8') as f:
            chunk = []
            for line in f:
                chunk.append(line.strip())
                if len(chunk) >= chunk_size:
                    yield from chunk
                    chunk = []
            if chunk:
                yield from chunk

    tokenizer.train_from_iterator(
        corpus_iterator(corpus_path),
        trainer=trainer,
        length=sum(1 for _ in open(corpus_path))  # 用于进度条
    )

    # 6. 保存原始tokenizer
    raw_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(raw_path)
    print(f"原始tokenizer已保存：{raw_path}")

    # 7. 包装为HuggingFace格式（方便与HF生态集成）
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=raw_path,
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )

    hf_tokenizer.save_pretrained(save_dir)
    print(f"HuggingFace tokenizer已保存：{save_dir}")

    # 8. 打印统计信息
    test_text = "The quick brown fox jumps over the lazy dog."
    tokens = hf_tokenizer.encode(test_text)
    print(f"\nTokenizer测试：")
    print(f"  输入：{test_text}")
    print(f"  Token数量：{len(tokens)}")
    print(f"  Token IDs：{tokens[:10]}...")

    return hf_tokenizer


def load_tokenizer(tokenizer_dir: str) -> PreTrainedTokenizerFast:
    """加载已训练的Tokenizer"""
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
```

### 24.2.3 数据集类实现

```python
# data.py - Dataset类

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import mmap


class TextDataset(Dataset):
    """
    高效的文本数据集类，支持内存映射以处理大文件

    将文本预先tokenize并保存为二进制格式，
    训练时直接读取，避免重复tokenization的开销。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        seq_len: int = 1024,
        stride: Optional[int] = None,
        cache_dir: str = "data/cache"
    ):
        """
        Args:
            data_path: 文本文件路径
            tokenizer: 分词器
            seq_len: 序列长度（包含输入和目标）
            stride: 滑动窗口步长，None表示等于seq_len（无重叠）
            cache_dir: tokenized数据缓存目录
        """
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        # 生成缓存文件路径（基于文件哈希，确保数据一致性）
        file_hash = self._hash_file(data_path)
        cache_path = Path(cache_dir) / f"{file_hash}_vocab{tokenizer.vocab_size}.npy"

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            print(f"从缓存加载tokenized数据：{cache_path}")
            self.data = np.load(str(cache_path), allow_pickle=False)
        else:
            print(f"首次运行，tokenizing数据（可能需要几分钟）...")
            self.data = self._tokenize_and_cache(
                data_path, tokenizer, str(cache_path)
            )

        # 计算样本数量
        self.n_samples = max(0, (len(self.data) - seq_len) // self.stride)
        print(f"数据集大小：{len(self.data):,} tokens → {self.n_samples:,} 样本")

    def _hash_file(self, file_path: str) -> str:
        """计算文件的MD5哈希（用于缓存验证）"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()[:8]

    def _tokenize_and_cache(
        self,
        data_path: str,
        tokenizer,
        cache_path: str,
        batch_size: int = 1000
    ) -> np.ndarray:
        """批量tokenize并缓存为numpy数组"""
        all_ids = []

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = []
            for line in tqdm(f, desc="读取文件"):
                lines.append(line.strip())

                if len(lines) >= batch_size:
                    # 批量编码（比逐行更快）
                    encodings = tokenizer(
                        lines,
                        add_special_tokens=True,
                        return_attention_mask=False,
                    )
                    for ids in encodings['input_ids']:
                        all_ids.extend(ids)
                    lines = []

            # 处理剩余行
            if lines:
                encodings = tokenizer(
                    lines,
                    add_special_tokens=True,
                    return_attention_mask=False,
                )
                for ids in encodings['input_ids']:
                    all_ids.extend(ids)

        # 保存为uint16（词汇表<=65535时节省内存）
        data = np.array(all_ids, dtype=np.uint16)
        np.save(cache_path, data)
        print(f"已缓存 {len(data):,} tokens 到 {cache_path}")

        return data

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """
        返回输入序列和目标序列

        目标序列是输入序列向右移位一个token（标准语言建模目标）
        即：给定 x[0..n-1]，预测 x[1..n]
        """
        start = idx * self.stride
        end = start + self.seq_len + 1  # +1 用于构造目标

        chunk = self.data[start:end].astype(np.int64)

        x = torch.from_numpy(chunk[:-1])  # 输入：前seq_len个token
        y = torch.from_numpy(chunk[1:])   # 目标：后seq_len个token

        return {'input_ids': x, 'labels': y}
```

### 24.2.4 DataLoader配置

```python
# data.py - DataLoader配置

def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    config,
    num_workers: int = 4
) -> tuple:
    """
    创建训练和验证DataLoader

    Args:
        train_path: 训练集路径
        val_path: 验证集路径
        tokenizer: 分词器
        config: 训练配置对象
        num_workers: 数据加载进程数

    Returns:
        (train_loader, val_loader) 元组
    """
    train_dataset = TextDataset(
        train_path,
        tokenizer,
        seq_len=config.seq_len,
        stride=config.seq_len,       # 训练集：无重叠
    )

    val_dataset = TextDataset(
        val_path,
        tokenizer,
        seq_len=config.seq_len,
        stride=config.seq_len // 2,  # 验证集：50%重叠，更全面的评估
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,         # 加速GPU数据传输
        drop_last=True,          # 丢弃最后不完整的批次
        persistent_workers=True  # 保持worker进程存活（减少重启开销）
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # 验证时可用更大批次
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

    print(f"训练集：{len(train_dataset):,} 样本，{len(train_loader):,} 批次")
    print(f"验证集：{len(val_dataset):,} 样本，{len(val_loader):,} 批次")

    return train_loader, val_loader
```

---

## 24.3 模型训练

### 24.3.1 模型配置

```python
# config.py - 完整配置文件

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class GPTConfig:
    """GPT模型架构配置"""
    # 词汇表和序列
    vocab_size: int = 50257
    seq_len: int = 1024

    # 模型规模
    n_layer: int = 12           # Transformer层数
    n_head: int = 12            # 注意力头数
    n_embd: int = 768           # 嵌入维度

    # 正则化
    dropout: float = 0.1
    bias: bool = True           # LayerNorm和Linear层是否使用偏置

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, \
            f"嵌入维度 {self.n_embd} 必须能被头数 {self.n_head} 整除"


@dataclass
class TrainConfig:
    """训练超参数配置"""
    # 数据
    train_data: str = "data/processed/train.txt"
    val_data: str = "data/processed/val.txt"
    tokenizer_dir: str = "data/tokenizer"

    # 批次和序列
    batch_size: int = 8                  # 每GPU批次大小
    seq_len: int = 1024
    grad_accum_steps: int = 8            # 梯度累积步数
    # 等效批次大小 = batch_size * grad_accum_steps * n_gpus

    # 优化器
    learning_rate: float = 6e-4          # 峰值学习率
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0               # 梯度裁剪阈值

    # 学习率调度（余弦退火with warmup）
    warmup_steps: int = 2000
    max_steps: int = 100000
    min_lr_ratio: float = 0.1           # 最小学习率 = peak_lr * ratio

    # 混合精度
    use_amp: bool = True
    amp_dtype: str = "bfloat16"         # bfloat16 或 float16

    # 检查点
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000              # 每N步保存一次
    keep_last_n: int = 3               # 保留最近N个检查点

    # 日志
    log_dir: str = "logs"
    log_every: int = 10                # 每N步打印一次
    eval_every: int = 500              # 每N步评估一次
    eval_steps: int = 100              # 评估时使用的步数

    # WandB（可选）
    use_wandb: bool = False
    wandb_project: str = "gpt-training"
    wandb_run_name: Optional[str] = None

    # 分布式训练
    distributed: bool = False

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    @property
    def amp_dtype_torch(self):
        return torch.bfloat16 if self.amp_dtype == "bfloat16" else torch.float16
```

### 24.3.2 GPT模型实现

```python
# model.py - GPT模型

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    """因果自注意力（带Flash Attention支持）"""

    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # QKV投影（合并为一个线性层，提高效率）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 检查是否支持Flash Attention
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            # 注册因果掩码（不参与梯度计算）
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.seq_len, config.seq_len))
                     .view(1, 1, config.seq_len, config.seq_len)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # 批次, 序列长度, 嵌入维度

        # 计算QKV
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 重塑为多头格式
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.flash:
            # PyTorch 2.0+ Flash Attention（自动选择最优实现）
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True  # 自动生成因果掩码
            )
        else:
            # 标准注意力（兼容旧版本）
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v

        # 合并多头
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    """前馈网络（使用GELU激活）"""

    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        # GPT-2使用4倍扩展
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """完整的Transformer块（Pre-LN架构）"""

    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: 先归一化再计算（比Post-LN更稳定）
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    GPT语言模型

    架构：Embedding → N × TransformerBlock → LM Head
    使用权重绑定（embedding和lm_head共享权重）
    """

    def __init__(self, config: 'GPTConfig'):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),   # token embedding
            'wpe': nn.Embedding(config.seq_len, config.n_embd),      # position embedding
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd, bias=config.bias),
        })

        # 语言模型头（预测下一个token的logits）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重绑定：embedding矩阵 = lm_head矩阵的转置
        # 这样可以减少参数量并提升性能
        self.transformer['wte'].weight = self.lm_head.weight

        # 初始化权重
        self.apply(self._init_weights)

        # 对残差投影使用特殊初始化（GPT-2论文）
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT参数量：{n_params/1e6:.1f}M")

    def _init_weights(self, module: nn.Module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: [B, T] 输入token IDs
            labels: [B, T] 目标token IDs（用于计算损失）

        Returns:
            (loss, logits) 元组，loss在labels为None时也为None
        """
        B, T = input_ids.shape
        assert T <= self.config.seq_len, \
            f"序列长度 {T} 超过最大长度 {self.config.seq_len}"

        device = input_ids.device

        # Token和位置嵌入
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer['wte'](input_ids)  # [B, T, C]
        pos_emb = self.transformer['wpe'](pos)        # [T, C]

        x = self.transformer['drop'](tok_emb + pos_emb)

        # 通过所有Transformer层
        for block in self.transformer['h']:
            x = block(x)

        # 最终LayerNorm
        x = self.transformer['ln_f'](x)

        # 计算logits和损失
        if labels is not None:
            logits = self.lm_head(x)  # [B, T, V]
            # 展平后计算交叉熵
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1
            )
        else:
            # 推理时只需计算最后一个位置的logits
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return loss, logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        文本生成（支持temperature、top-k、top-p采样）

        Args:
            input_ids: [1, T] 输入token IDs
            max_new_tokens: 最大生成token数
            temperature: 采样温度（越高越随机）
            top_k: Top-K采样
            top_p: Top-P（核采样）
            eos_token_id: 终止token ID

        Returns:
            包含生成token的张量 [1, T + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 裁剪到最大上下文长度
            idx_cond = input_ids[:, -self.config.seq_len:]

            # 前向传播
            _, logits = self(idx_cond)
            logits = logits[:, -1, :]  # 取最后一个位置 [B, V]

            # 应用temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-K过滤
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-P（核采样）过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_indices_to_remove] = float('-inf')

                logits.scatter_(1, sorted_indices, sorted_logits)

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # 检查终止条件
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return input_ids
```

### 24.3.3 训练循环

```python
# train.py - 主训练脚本

import os
import time
import math
import glob
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional
import logging

from config import GPTConfig, TrainConfig
from model import GPT
from data import create_dataloaders, load_tokenizer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_lr(step: int, config: TrainConfig) -> float:
    """
    余弦退火学习率调度（带线性warmup）

    公式：
    - warmup阶段：lr = peak_lr * (step / warmup_steps)
    - 退火阶段：lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
    - progress = (step - warmup_steps) / (max_steps - warmup_steps)
    """
    min_lr = config.learning_rate * config.min_lr_ratio

    if step < config.warmup_steps:
        # 线性warmup
        return config.learning_rate * (step + 1) / config.warmup_steps

    if step > config.max_steps:
        return min_lr

    # 余弦退火
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr + coeff * (config.learning_rate - min_lr)


def configure_optimizer(model: nn.Module, config: TrainConfig) -> AdamW:
    """
    配置AdamW优化器

    关键细节：对一维参数（偏置、LayerNorm权重）不使用权重衰减
    这是GPT训练的重要trick
    """
    # 将参数分为两组
    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            # 2D及以上参数（矩阵权重）：使用weight decay
            decay_params.append(param)
        else:
            # 1D参数（偏置、LN权重）：不使用weight decay
            nodecay_params.append(param)

    n_decay = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in nodecay_params)
    logger.info(f"参数分组：decay={n_decay/1e6:.1f}M，no_decay={n_nodecay/1e6:.1f}M")

    optimizer = AdamW(
        [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=True  # PyTorch 2.0+ 融合实现，速度更快
    )

    return optimizer


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader,
    config: TrainConfig,
    device: torch.device
) -> float:
    """在验证集上评估模型，返回平均loss"""
    model.eval()

    total_loss = 0.0
    n_batches = min(config.eval_steps, len(val_loader))

    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss, _ = model(input_ids, labels)

        total_loss += loss.item()

    model.train()
    return total_loss / n_batches


def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scaler: GradScaler,
    step: int,
    loss: float,
    config: TrainConfig
):
    """保存训练检查点"""
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'config': config,
    }

    path = os.path.join(config.checkpoint_dir, f"ckpt_step{step:07d}.pt")
    torch.save(checkpoint, path)
    logger.info(f"检查点已保存：{path}")

    # 删除旧检查点，只保留最近N个
    checkpoints = sorted(
        glob.glob(os.path.join(config.checkpoint_dir, "ckpt_step*.pt"))
    )
    while len(checkpoints) > config.keep_last_n:
        os.remove(checkpoints.pop(0))


def load_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scaler: GradScaler,
    config: TrainConfig
) -> int:
    """从最新检查点恢复训练，返回起始步数"""
    checkpoints = sorted(
        glob.glob(os.path.join(config.checkpoint_dir, "ckpt_step*.pt"))
    )

    if not checkpoints:
        logger.info("未找到检查点，从头开始训练")
        return 0

    latest = checkpoints[-1]
    logger.info(f"从检查点恢复：{latest}")

    checkpoint = torch.load(latest, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_step = checkpoint['step'] + 1
    logger.info(f"从步骤 {start_step} 继续训练")

    return start_step


def train(model_config: GPTConfig, train_config: TrainConfig):
    """主训练函数"""

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备：{device}")

    if device.type == 'cuda':
        logger.info(f"GPU：{torch.cuda.get_device_name(0)}")
        logger.info(f"GPU内存：{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # 设置随机种子（可重现性）
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True   # 允许TF32（A100上更快）
    torch.backends.cudnn.allow_tf32 = True

    # 初始化WandB（可选）
    if train_config.use_wandb:
        import wandb
        wandb.init(
            project=train_config.wandb_project,
            name=train_config.wandb_run_name,
            config={**vars(model_config), **vars(train_config)}
        )

    # 加载tokenizer和数据
    tokenizer = load_tokenizer(train_config.tokenizer_dir)
    train_loader, val_loader = create_dataloaders(
        train_config.train_data,
        train_config.val_data,
        tokenizer,
        train_config
    )

    # 创建模型
    model = GPT(model_config).to(device)

    # 编译模型（PyTorch 2.0+，显著提升速度）
    if hasattr(torch, 'compile'):
        logger.info("编译模型（torch.compile）...")
        model = torch.compile(model)

    # 配置优化器和GradScaler
    optimizer = configure_optimizer(model, train_config)
    scaler = GradScaler(enabled=train_config.use_amp and device.type == 'cuda')

    # 从检查点恢复（如果存在）
    start_step = load_checkpoint(model, optimizer, scaler, train_config)

    # TensorBoard日志
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=train_config.log_dir)

    # 训练循环
    model.train()
    train_iter = iter(train_loader)

    logger.info(f"开始训练，从步骤 {start_step} 到 {train_config.max_steps}")
    logger.info(f"等效批次大小：{train_config.effective_batch_size} tokens/step × {train_config.seq_len} seq_len")

    t0 = time.time()

    for step in range(start_step, train_config.max_steps + 1):
        # 动态调整学习率
        lr = get_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 梯度累积训练步
        optimizer.zero_grad()

        total_loss = 0.0

        for micro_step in range(train_config.grad_accum_steps):
            # 获取下一批数据
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # 混合精度前向传播
            amp_ctx = torch.amp.autocast(
                device_type='cuda',
                dtype=train_config.amp_dtype_torch
            ) if train_config.use_amp else nullcontext()

            with amp_ctx:
                loss, _ = model(input_ids, labels)
                # 损失除以累积步数（等效于对累积批次求平均）
                loss = loss / train_config.grad_accum_steps

            # 反向传播
            scaler.scale(loss).backward()
            total_loss += loss.item()

        # 梯度裁剪（防止梯度爆炸）
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            train_config.grad_clip
        )

        # 更新参数
        scaler.step(optimizer)
        scaler.update()

        # 日志记录
        if step % train_config.log_every == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # 计算吞吐量（tokens/second）
            tokens_per_sec = (
                train_config.batch_size *
                train_config.seq_len *
                train_config.grad_accum_steps *
                train_config.log_every / dt
            )

            logger.info(
                f"步骤 {step:6d} | "
                f"loss {total_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"梯度范数 {grad_norm:.2f} | "
                f"吞吐量 {tokens_per_sec/1e3:.1f}K tok/s"
            )

            writer.add_scalar('train/loss', total_loss, step)
            writer.add_scalar('train/lr', lr, step)
            writer.add_scalar('train/grad_norm', grad_norm, step)

            if train_config.use_wandb:
                import wandb
                wandb.log({
                    'train/loss': total_loss,
                    'train/lr': lr,
                    'train/grad_norm': grad_norm,
                    'train/tokens_per_sec': tokens_per_sec,
                }, step=step)

        # 定期评估
        if step % train_config.eval_every == 0:
            val_loss = evaluate(model, val_loader, train_config, device)
            val_ppl = math.exp(val_loss)

            logger.info(f"步骤 {step:6d} | 验证loss {val_loss:.4f} | 困惑度 {val_ppl:.2f}")

            writer.add_scalar('val/loss', val_loss, step)
            writer.add_scalar('val/perplexity', val_ppl, step)

        # 保存检查点
        if step % train_config.save_every == 0 and step > 0:
            save_checkpoint(model, optimizer, scaler, step, total_loss, train_config)

    writer.close()
    logger.info("训练完成！")


if __name__ == "__main__":
    model_config = GPTConfig()
    train_config = TrainConfig()
    train(model_config, train_config)
```

---

## 24.4 Hugging Face集成

### 24.4.1 将模型转换为HF格式

Hugging Face生态系统提供了统一的模型接口，方便分享和复用。我们将自定义GPT转换为HF兼容格式。

```python
# hf_convert.py - 模型格式转换

import torch
import json
import shutil
from pathlib import Path
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForCausalLM
)
from model import GPT
from config import GPTConfig


def convert_to_hf_format(
    checkpoint_path: str,
    tokenizer_dir: str,
    output_dir: str,
    model_config: GPTConfig
):
    """
    将自定义GPT检查点转换为Hugging Face GPT-2格式

    HF的GPT2LMHeadModel与我们的模型架构基本兼容，
    主要区别在于权重名称映射。

    Args:
        checkpoint_path: PyTorch检查点路径
        tokenizer_dir: Tokenizer目录
        output_dir: 输出目录
        model_config: 模型配置
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. 加载自定义模型
    print("加载自定义模型...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    our_model = GPT(model_config)
    # 如果使用了torch.compile，需要处理_orig_mod前缀
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    our_model.load_state_dict(state_dict)
    our_model.eval()

    # 2. 创建HF配置
    hf_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.seq_len,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        resid_pdrop=model_config.dropout,
        embd_pdrop=model_config.dropout,
        attn_pdrop=model_config.dropout,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
    )

    # 3. 创建HF模型并复制权重
    print("创建HF模型并映射权重...")
    hf_model = GPT2LMHeadModel(hf_config)

    # 权重名称映射表
    # 格式：(我们的名称, HF的名称)
    mapping = {
        'transformer.wte.weight': 'transformer.wte.weight',
        'transformer.wpe.weight': 'transformer.wpe.weight',
        'transformer.ln_f.weight': 'transformer.ln_f.weight',
        'transformer.ln_f.bias': 'transformer.ln_f.bias',
        'lm_head.weight': 'lm_head.weight',
    }

    # 每层的映射
    for i in range(model_config.n_layer):
        layer_mapping = {
            f'transformer.h.{i}.ln_1.weight': f'transformer.h.{i}.ln_1.weight',
            f'transformer.h.{i}.ln_1.bias': f'transformer.h.{i}.ln_1.bias',
            f'transformer.h.{i}.ln_2.weight': f'transformer.h.{i}.ln_2.weight',
            f'transformer.h.{i}.ln_2.bias': f'transformer.h.{i}.ln_2.bias',
            f'transformer.h.{i}.attn.c_attn.weight': f'transformer.h.{i}.attn.c_attn.weight',
            f'transformer.h.{i}.attn.c_attn.bias': f'transformer.h.{i}.attn.c_attn.bias',
            f'transformer.h.{i}.attn.c_proj.weight': f'transformer.h.{i}.attn.c_proj.weight',
            f'transformer.h.{i}.attn.c_proj.bias': f'transformer.h.{i}.attn.c_proj.bias',
            f'transformer.h.{i}.mlp.c_fc.weight': f'transformer.h.{i}.mlp.c_fc.weight',
            f'transformer.h.{i}.mlp.c_fc.bias': f'transformer.h.{i}.mlp.c_fc.bias',
            f'transformer.h.{i}.mlp.c_proj.weight': f'transformer.h.{i}.mlp.c_proj.weight',
            f'transformer.h.{i}.mlp.c_proj.bias': f'transformer.h.{i}.mlp.c_proj.bias',
        }
        mapping.update(layer_mapping)

    # 复制权重
    our_state = our_model.state_dict()
    hf_state = hf_model.state_dict()

    for our_key, hf_key in mapping.items():
        if our_key in our_state and hf_key in hf_state:
            hf_state[hf_key].copy_(our_state[our_key])

    hf_model.load_state_dict(hf_state)

    # 4. 保存HF格式模型
    print(f"保存HF格式模型到 {output_dir}...")
    hf_model.save_pretrained(output_dir)

    # 5. 复制tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    tokenizer.save_pretrained(output_dir)

    # 6. 验证模型
    print("验证模型...")
    test_input = tokenizer("Once upon a time", return_tensors='pt')
    with torch.no_grad():
        output = hf_model(**test_input, labels=test_input['input_ids'])
    print(f"测试loss：{output.loss.item():.4f} （应该是一个合理的正数）")

    print(f"转换完成！HF格式模型保存在：{output_dir}")

    return hf_model, tokenizer
```

### 24.4.2 推送到Hugging Face Hub

```python
# hf_upload.py - 上传到HF Hub

from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


def push_to_hub(
    model_dir: str,
    repo_id: str,            # 格式：username/model-name
    private: bool = False,
    commit_message: str = "Add model"
):
    """
    将模型推送到Hugging Face Hub

    前置条件：
    1. 已安装 huggingface_hub
    2. 已运行 huggingface-cli login 登录

    Args:
        model_dir: 本地模型目录
        repo_id: Hub仓库ID，格式为 "username/model-name"
        private: 是否设为私有仓库
        commit_message: 提交信息
    """
    api = HfApi()

    # 1. 创建仓库（如果不存在）
    print(f"创建/确认仓库：{repo_id}")
    create_repo(
        repo_id=repo_id,
        private=private,
        exist_ok=True,  # 已存在时不报错
        repo_type="model"
    )

    # 2. 上传模型文件
    print(f"上传文件到 {repo_id}...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message
    )

    print(f"上传完成！模型地址：https://huggingface.co/{repo_id}")


def create_model_card(
    model_dir: str,
    repo_id: str,
    model_description: str,
    training_details: dict
) -> str:
    """
    生成模型卡（README.md）

    模型卡是HF Hub上的模型说明文档，
    包含模型用途、训练细节、使用示例等。
    """
    username, model_name = repo_id.split('/')

    card_content = f"""---
language:
- en
license: mit
tags:
- text-generation
- gpt
- causal-lm
datasets:
- wikitext
---

# {model_name}

{model_description}

## 模型详情

- **架构**：GPT（仅解码器Transformer）
- **参数量**：{training_details.get('n_params', '~117M')}
- **训练数据**：{training_details.get('dataset', 'WikiText-103')}
- **上下文长度**：{training_details.get('seq_len', 1024)} tokens
- **词汇表大小**：{training_details.get('vocab_size', 50257)}

## 训练配置

| 参数 | 值 |
|------|----|
| 训练步数 | {training_details.get('max_steps', 100000)} |
| 批次大小（等效） | {training_details.get('effective_batch_size', 64)} |
| 学习率 | {training_details.get('learning_rate', '6e-4')} |
| 优化器 | AdamW |

## 使用示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")

# 文本生成
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 局限性

本模型是用于教学目的的小型模型，不适合生产环境使用。
"""

    card_path = Path(model_dir) / "README.md"
    card_path.write_text(card_content, encoding='utf-8')

    print(f"模型卡已生成：{card_path}")
    return card_content
```

### 24.4.3 使用HF Trainer（可选路径）

```python
# hf_trainer.py - 使用Hugging Face Trainer

from transformers import (
    Trainer,
    TrainingArguments,
    GPT2LMHeadModel,
    GPT2Config,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
import torch


def train_with_hf_trainer(
    model_config_dict: dict,
    train_texts: list,
    val_texts: list,
    tokenizer,
    output_dir: str = "hf_training_output"
):
    """
    使用Hugging Face Trainer训练GPT-2

    HF Trainer自动处理：
    - 梯度累积
    - 混合精度
    - 分布式训练
    - 检查点保存
    - 评估循环
    """
    # 1. 创建模型
    config = GPT2Config(**model_config_dict)
    model = GPT2LMHeadModel(config)

    # 2. 创建数据集
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=1024,
            padding=False
        )

    train_dataset = Dataset.from_dict({'text': train_texts})
    val_dataset = Dataset.from_dict({'text': val_texts})

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    # 3. 数据整理器（自动生成labels）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT是因果语言模型，不是掩码语言模型
    )

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,

        # 批次配置
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=8,

        # 训练步数
        max_steps=100000,
        warmup_steps=2000,

        # 优化器
        learning_rate=6e-4,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,

        # 混合精度
        bf16=True,

        # 日志和保存
        logging_steps=100,
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=3,
        evaluation_strategy="steps",
        load_best_model_at_end=True,

        # 报告
        report_to="tensorboard",
    )

    # 5. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 6. 开始训练
    trainer.train()

    # 7. 保存最终模型
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer
```

---

## 24.5 部署上线

### 24.5.1 FastAPI推理服务

```python
# serve.py - FastAPI推理服务

import asyncio
import time
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional, AsyncIterator
import json
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

logger = logging.getLogger(__name__)


# ==================== 全局模型状态 ====================

class ModelState:
    """线程安全的模型状态管理"""
    model: Optional[AutoModelForCausalLM] = None
    tokenizer = None
    device: torch.device = None


model_state = ModelState()


# ==================== 请求/响应模型 ====================

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="输入文本提示", min_length=1)
    max_new_tokens: int = Field(200, ge=1, le=2048, description="最大生成token数")
    temperature: float = Field(0.8, ge=0.01, le=2.0, description="采样温度")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-P核采样")
    top_k: int = Field(50, ge=0, le=500, description="Top-K采样")
    stream: bool = Field(False, description="是否使用流式响应")


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float


# ==================== 应用生命周期 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时加载模型，关闭时释放"""
    # 启动
    model_dir = "hf_model"  # 或 Hugging Face Hub ID

    logger.info(f"加载模型：{model_dir}")

    model_state.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model_state.tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_state.model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,  # 半精度推理，节省内存
        device_map='auto',           # 自动分配到可用设备
    )
    model_state.model.eval()

    # 编译模型（可选，推理速度提升约20-30%）
    if hasattr(torch, 'compile') and model_state.device.type == 'cuda':
        logger.info("编译模型用于推理...")
        model_state.model = torch.compile(
            model_state.model,
            mode='reduce-overhead'  # 针对推理优化
        )

    logger.info(f"模型已就绪，设备：{model_state.device}")

    yield

    # 关闭
    logger.info("释放模型资源...")
    del model_state.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="GPT推理服务",
    description="小型GPT模型的生产级推理API",
    version="1.0.0",
    lifespan=lifespan
)


# ==================== API端点 ====================

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_loaded": model_state.model is not None,
        "device": str(model_state.device),
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    标准文本生成端点（非流式）

    适用于需要完整响应的场景，如批量处理。
    """
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    t0 = time.time()

    # Tokenize输入
    inputs = model_state.tokenizer(
        request.prompt,
        return_tensors='pt',
        truncation=True,
        max_length=model_state.model.config.n_positions - request.max_new_tokens
    ).to(model_state.device)

    input_length = inputs['input_ids'].shape[1]

    # 生成
    with torch.no_grad():
        outputs = model_state.model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=True,
            pad_token_id=model_state.tokenizer.eos_token_id,
        )

    # 解码（只返回新生成的部分）
    generated_ids = outputs[0][input_length:]
    generated_text = model_state.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    elapsed = time.time() - t0
    n_tokens = len(generated_ids)

    return GenerateResponse(
        text=generated_text,
        tokens_generated=n_tokens,
        time_seconds=elapsed,
        tokens_per_second=n_tokens / elapsed
    )


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """
    流式文本生成端点（Server-Sent Events）

    逐token返回生成结果，适用于聊天界面等需要实时展示的场景。

    客户端示例（JavaScript）：
        const eventSource = new EventSource('/generate/stream');
        eventSource.onmessage = (e) => {
            const data = JSON.parse(e.data);
            console.log(data.token);
        };
    """
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    async def token_generator() -> AsyncIterator[str]:
        """异步token生成器"""

        # 使用TextIteratorStreamer实现流式输出
        streamer = TextIteratorStreamer(
            model_state.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True  # 不重复输出prompt
        )

        inputs = model_state.tokenizer(
            request.prompt,
            return_tensors='pt',
            truncation=True,
            max_length=model_state.model.config.n_positions - request.max_new_tokens
        ).to(model_state.device)

        # 在独立线程中运行生成（避免阻塞异步事件循环）
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=True,
            pad_token_id=model_state.tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = Thread(
            target=model_state.model.generate,
            kwargs=generation_kwargs,
            daemon=True
        )
        thread.start()

        # 逐token发送SSE事件
        for text in streamer:
            if text:
                data = json.dumps({"token": text, "done": False})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0)  # 让出控制权给事件循环

        # 发送完成信号
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
        }
    )
```

### 24.5.2 批处理推理

```python
# serve.py - 批处理推理部分

class BatchGenerateRequest(BaseModel):
    prompts: list[str] = Field(..., description="批量提示词列表", min_items=1, max_items=32)
    max_new_tokens: int = Field(100, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.01, le=2.0)


@app.post("/generate/batch")
async def generate_batch(request: BatchGenerateRequest):
    """
    批量文本生成端点

    同时处理多个请求，比逐一处理效率更高。
    GPU利用率提升约2-4倍（取决于批次大小和序列长度）。
    """
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    t0 = time.time()

    # 批量tokenize（自动padding到相同长度）
    inputs = model_state.tokenizer(
        request.prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(model_state.device)

    input_lengths = inputs['attention_mask'].sum(dim=1)  # 每个样本的真实长度

    with torch.no_grad():
        outputs = model_state.model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=model_state.tokenizer.eos_token_id,
        )

    # 解码每个样本（去除padding和原始prompt）
    results = []
    for i, (output, input_len) in enumerate(zip(outputs, input_lengths)):
        generated = output[input_len:]
        text = model_state.tokenizer.decode(generated, skip_special_tokens=True)
        results.append({"prompt": request.prompts[i], "generated": text})

    elapsed = time.time() - t0

    return {
        "results": results,
        "batch_size": len(request.prompts),
        "time_seconds": elapsed,
        "avg_time_per_prompt": elapsed / len(request.prompts)
    }


if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        workers=1,         # GPU服务通常使用单worker
        log_level="info",
    )
```

### 24.5.3 Docker容器化

```dockerfile
# Dockerfile

# 使用官方CUDA基础镜像（支持GPU）
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 系统依赖
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 先复制依赖文件（利用Docker层缓存）
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 预下载模型（可选，也可在运行时挂载）
# RUN python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('./hf_model')"

# 暴露服务端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python3", "-m", "uvicorn", "serve:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  gpt-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      # 挂载模型目录（避免将大文件打包进镜像）
      - ./hf_model:/app/hf_model:ro
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_DIR=/app/hf_model
    restart: unless-stopped

  # 可选：Nginx反向代理（负载均衡、SSL终止）
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - gpt-api
```

```
# requirements.txt

torch>=2.0.0
transformers>=4.35.0
tokenizers>=0.14.0
datasets>=2.14.0
huggingface_hub>=0.19.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
tensorboard>=2.14.0
wandb>=0.16.0
tqdm>=4.65.0
numpy>=1.24.0
requests>=2.31.0
```

---

## 本章小结

本章完成了一个完整的GPT模型训练和部署项目：

1. **数据准备**：实现了数据下载、清洗、BPE tokenizer训练、高效Dataset类和DataLoader配置的完整流程。关键技术包括内存映射加速大文件读取、基于哈希的数据缓存机制。

2. **模型训练**：实现了带Flash Attention的GPT模型，配置了余弦退火学习率调度、参数分组的AdamW优化器、梯度累积和混合精度训练。使用TensorBoard和WandB进行训练监控。

3. **Hugging Face集成**：完成了自定义模型到HF格式的转换，实现了Hub上传和模型卡生成，并展示了HF Trainer的使用方法。

4. **生产级部署**：使用FastAPI构建了支持流式响应和批处理推理的API服务，并通过Docker容器化实现了可重现的部署环境。

---

## 代码实战

以下是完整的训练和部署示例：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
python -c "
from data import download_wikitext103, prepare_wikitext_corpus, train_bpe_tokenizer

# 下载数据集
dataset = download_wikitext103('data/raw')

# 处理语料
prepare_wikitext_corpus(dataset, 'data/processed/corpus.txt')

# 训练Tokenizer
tokenizer = train_bpe_tokenizer('data/processed/corpus.txt')
"

# 3. 分割数据集（简化示例）
python -c "
import random
with open('data/processed/corpus.txt') as f:
    lines = f.readlines()

random.shuffle(lines)
n = len(lines)
train_lines = lines[:int(n*0.9)]
val_lines = lines[int(n*0.9):]

with open('data/processed/train.txt', 'w') as f:
    f.writelines(train_lines)

with open('data/processed/val.txt', 'w') as f:
    f.writelines(val_lines)

print(f'训练集：{len(train_lines)} 行，验证集：{len(val_lines)} 行')
"

# 4. 开始训练
python train.py

# 5. 转换为HF格式
python -c "
from hf_convert import convert_to_hf_format, create_model_card
from config import GPTConfig

model_config = GPTConfig()
# 替换为实际检查点路径
hf_model, tokenizer = convert_to_hf_format(
    checkpoint_path='checkpoints/ckpt_step0100000.pt',
    tokenizer_dir='data/tokenizer',
    output_dir='hf_model',
    model_config=model_config
)

# 生成模型卡
create_model_card(
    model_dir='hf_model',
    repo_id='your-username/gpt-small-wikitext103',
    model_description='A small GPT model trained on WikiText-103',
    training_details={'n_params': '117M', 'max_steps': 100000}
)
"

# 6. 推送到Hub（需要先 huggingface-cli login）
python -c "
from hf_upload import push_to_hub
push_to_hub('hf_model', 'your-username/gpt-small-wikitext103')
"

# 7. 启动推理服务
python serve.py

# 8. 测试API
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "The future of artificial intelligence",
    "max_new_tokens": 100,
    "temperature": 0.8
  }'

# 9. Docker部署
docker-compose up -d
```

```python
# 完整的推理客户端示例

import requests
import json


API_BASE = "http://localhost:8000"


def generate_text(prompt: str, max_tokens: int = 200) -> str:
    """调用标准生成API"""
    response = requests.post(
        f"{API_BASE}/generate",
        json={
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": 0.8,
            "top_p": 0.9,
        }
    )
    response.raise_for_status()
    result = response.json()

    print(f"生成 {result['tokens_generated']} tokens，"
          f"速度 {result['tokens_per_second']:.1f} tok/s")

    return result['text']


def generate_stream(prompt: str, max_tokens: int = 200):
    """调用流式生成API"""
    response = requests.post(
        f"{API_BASE}/generate/stream",
        json={
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": 0.8,
            "stream": True,
        },
        stream=True
    )

    print(f"Prompt: {prompt}")
    print("Generated: ", end="", flush=True)

    for line in response.iter_lines():
        if line and line.startswith(b"data: "):
            data = json.loads(line[6:])
            if not data['done']:
                print(data['token'], end="", flush=True)

    print()  # 换行


if __name__ == "__main__":
    # 测试标准生成
    text = generate_text("Once upon a time in a land far away,")
    print(f"\n生成文本：{text[:200]}...")

    # 测试流式生成
    generate_stream("The theory of relativity states that")
```

---

## 练习题

### 基础题

**题1：配置一个更小的GPT模型**

修改 `GPTConfig` 类，创建一个参数量约为25M的"GPT-Tiny"配置，要求：
- 保持12个注意力头
- 调整 `n_embd` 和 `n_layer` 使总参数约为25M
- 验证配置是否满足 `n_embd % n_head == 0` 约束

**题2：添加早停机制**

在训练循环中实现早停（Early Stopping）功能：
- 监控验证loss，如果连续 `patience` 次评估没有改善则停止训练
- 保存验证loss最低时的检查点
- 确保与现有检查点保存逻辑兼容

### 中级题

**题3：实现KV-Cache推理加速**

当前 `generate()` 函数在每个推理步骤重新计算所有token的注意力。实现KV-Cache来加速推理：
- 修改 `CausalSelfAttention.forward()` 接受可选的 `past_key_values` 参数
- 修改 `GPT.generate()` 使用缓存
- 测量加速比（生成200 tokens时，无缓存 vs 有缓存的时间对比）

**题4：实现梯度检查点**

对于24GB以下的GPU，训练时可能遇到OOM（显存不足）问题。实现梯度检查点：
- 在 `TransformerBlock.forward()` 中使用 `torch.utils.checkpoint.checkpoint`
- 在 `TrainConfig` 中添加 `use_gradient_checkpointing` 选项
- 测量启用前后的显存使用量和训练速度变化

### 提高题

**题5：实现多GPU分布式数据并行训练**

使用 `torch.distributed` 和 `DistributedDataParallel (DDP)` 实现多GPU训练：

1. 在 `train.py` 中添加 DDP 初始化代码（使用 `torch.distributed.init_process_group`）
2. 将模型包装在 `DDP` 中，并正确处理 `module` 属性访问
3. 修改 `DataLoader` 使用 `DistributedSampler` 确保每个GPU看到不同的数据
4. 确保只在 `rank=0` 的进程上保存检查点和打印日志
5. 提供完整的启动命令（使用 `torchrun`）
6. 验证：2个GPU上的等效批次大小应是单GPU的2倍

---

## 练习答案

### 题1答案

```python
from config import GPTConfig

# GPT-2 Small: 117M 参数，n_layer=12, n_embd=768, n_head=12
# 目标：约25M参数

# 参数量估算公式（近似）：
# 嵌入层：vocab_size * n_embd = 50257 * n_embd
# 每层：12 * n_embd^2 (注意力) + 8 * n_embd^2 (FFN) ≈ 20 * n_embd^2
# n_layer 层：n_layer * 20 * n_embd^2

# 求解：25M ≈ 50257 * n_embd + n_layer * 20 * n_embd^2
# 尝试 n_embd=384, n_layer=6：
# ≈ 50257*384 + 6*20*384^2
# ≈ 19.3M + 17.7M ≈ 37M （太大）

# 尝试 n_embd=256, n_layer=6：
# ≈ 50257*256 + 6*20*256^2
# ≈ 12.9M + 7.9M ≈ 20.8M （接近）

# 尝试 n_embd=288, n_layer=6（288/12=24，满足整除约束）：
# ≈ 50257*288 + 6*20*288^2
# ≈ 14.5M + 10.0M ≈ 24.5M （约25M）

tiny_config = GPTConfig(
    vocab_size=50257,
    seq_len=512,      # 缩短序列长度以进一步节省内存
    n_layer=6,
    n_head=12,        # 保持12个头
    n_embd=288,       # 288 / 12 = 24，满足整除约束
    dropout=0.1,
)

# 验证
from model import GPT
model = GPT(tiny_config)
# 输出：GPT参数量：约25M
```

### 题2答案

```python
# 在 train.py 中的 train() 函数内添加早停逻辑

def train_with_early_stopping(
    model_config: GPTConfig,
    train_config: TrainConfig,
    patience: int = 5          # 连续N次评估未改善则停止
):
    # ... 前置代码不变 ...

    # 早停状态
    best_val_loss = float('inf')
    patience_counter = 0
    best_checkpoint_path = None

    for step in range(start_step, train_config.max_steps + 1):
        # ... 训练步骤代码不变 ...

        # 定期评估
        if step % train_config.eval_every == 0 and step > 0:
            val_loss = evaluate(model, val_loader, train_config, device)

            if val_loss < best_val_loss:
                # 验证loss改善
                best_val_loss = val_loss
                patience_counter = 0

                # 保存最佳检查点
                best_path = os.path.join(
                    train_config.checkpoint_dir,
                    "best_model.pt"
                )
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                }, best_path)
                logger.info(f"步骤 {step}：新的最佳验证loss {val_loss:.4f}，已保存")
            else:
                # 验证loss未改善
                patience_counter += 1
                logger.info(
                    f"步骤 {step}：验证loss {val_loss:.4f} 未改善 "
                    f"（最佳：{best_val_loss:.4f}，patience：{patience_counter}/{patience}）"
                )

                if patience_counter >= patience:
                    logger.info(f"早停触发！连续 {patience} 次评估未改善")
                    break

    # 加载最佳模型
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"已加载最佳模型（验证loss：{checkpoint['val_loss']:.4f}）")

    return model
```

### 题3答案

```python
# model.py - 带KV-Cache的注意力层

class CausalSelfAttentionWithCache(nn.Module):
    """支持KV-Cache的因果自注意力"""

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: Optional[tuple] = None  # (past_k, past_v)
    ) -> tuple:
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 拼接历史KV
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=2)  # 在序列维度拼接
            v = torch.cat([past_v, v], dim=2)

        # 保存当前KV用于下一步
        current_key_values = (k, v)

        # 注意力计算
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(past_key_values is None))
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y, current_key_values


# 加速比测试
def benchmark_kvcache(model, tokenizer, prompt, max_new_tokens=200):
    import time

    inputs = tokenizer(prompt, return_tensors='pt')

    # 无缓存
    t0 = time.time()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=False)
    t_no_cache = time.time() - t0

    # 有缓存
    t0 = time.time()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    t_with_cache = time.time() - t0

    print(f"无缓存：{t_no_cache:.2f}s，有缓存：{t_with_cache:.2f}s")
    print(f"加速比：{t_no_cache/t_with_cache:.1f}x")
    # 预期加速比：随序列增长约 2-10x
```

### 题4答案

```python
# model.py - 带梯度检查点的TransformerBlock

import torch.utils.checkpoint as checkpoint_utils


class TransformerBlock(nn.Module):

    def forward(
        self,
        x: torch.Tensor,
        use_gradient_checkpointing: bool = False
    ) -> torch.Tensor:

        if use_gradient_checkpointing and self.training:
            # 使用梯度检查点：前向时不保存中间激活值
            # 反向时重新计算（用时间换空间）
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            x = x + checkpoint_utils.checkpoint(
                create_custom_forward(self.attn),
                self.ln_1(x),
                use_reentrant=False
            )
            x = x + checkpoint_utils.checkpoint(
                create_custom_forward(self.mlp),
                self.ln_2(x),
                use_reentrant=False
            )
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))

        return x

# 显存测量
# 无梯度检查点（n_layer=12, batch=8, seq_len=1024）：约18GB
# 有梯度检查点：约9GB（节省约50%）
# 速度损失：约20-30%（因为需要重新计算激活值）
```

### 题5答案

```python
# train_ddp.py - 分布式数据并行训练

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os


def setup_distributed():
    """初始化分布式训练环境"""
    # torchrun 自动设置这些环境变量
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    dist.init_process_group(
        backend='nccl',          # GPU通信使用NCCL
        init_method='env://',    # 从环境变量读取配置
    )

    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def train_ddp(model_config, train_config):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # 只在rank 0打印日志
    is_master = (rank == 0)

    if is_master:
        logger.info(f"分布式训练：world_size={world_size}")

    # 创建模型
    model = GPT(model_config).to(device)
    model = DDP(model, device_ids=[local_rank])

    # 使用DistributedSampler确保数据不重叠
    train_dataset = TextDataset(
        train_config.train_data,
        tokenizer,
        seq_len=train_config.seq_len
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=train_sampler,  # 使用分布式sampler，不设shuffle
        num_workers=4,
        pin_memory=True,
    )

    # 优化器（访问DDP包装的模型需要.module）
    optimizer = configure_optimizer(model.module, train_config)

    for step in range(train_config.max_steps):
        # 设置epoch（影响DistributedSampler的随机种子）
        train_sampler.set_epoch(step // len(train_loader))

        # 训练步骤（与单GPU相同）
        # ...

        # 只在主进程保存检查点
        if is_master and step % train_config.save_every == 0:
            save_checkpoint(
                model.module,  # 注意：保存.module，不保存DDP包装
                optimizer,
                scaler,
                step,
                total_loss,
                train_config
            )

    dist.destroy_process_group()


# 启动命令（2个GPU）：
# torchrun --nproc_per_node=2 train_ddp.py
#
# 等效批次大小 = batch_size * grad_accum_steps * world_size
# = 8 * 8 * 2 = 128 tokens/step（相比单GPU的64翻倍）
```
