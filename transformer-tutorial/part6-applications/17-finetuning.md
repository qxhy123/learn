# 第17章：微调技术

> "微调预训练模型就像站在巨人的肩膀上——你不需要重新学会如何理解语言，只需要学会如何将这种理解应用到你的特定问题上。" —— 改编自 Howard & Ruder, 2018

---

## 学习目标

完成本章后，你将能够：

1. 理解预训练模型微调的原理及其相对于从头训练的优势
2. 掌握全参数微调的实现方式及注意事项
3. 理解不同冻结策略的适用场景与选择依据
4. 掌握微调中学习率设置的关键技巧（分层学习率、预热等）
5. 能够针对文本分类、序列标注等下游任务完成端到端微调

---

## 17.1 微调概述

### 17.1.1 预训练-微调范式

现代NLP的核心工作流程已经从"为每个任务从头训练一个模型"演变为"在大规模语料上预训练一个通用模型，然后针对具体任务进行微调"。这一范式转变由BERT（2018）正式确立，并被后续的GPT、T5、RoBERTa等模型沿用至今。

```
预训练阶段（昂贵，一次性）
─────────────────────────────────────────────────────────
海量无标注文本（TB级别）
          ↓
  自监督预训练目标（MLM / 语言建模 / 去噪等）
          ↓
  通用语言表示（数亿 ~ 数千亿参数）

微调阶段（轻量，任务特定）
─────────────────────────────────────────────────────────
预训练权重（冻结或可更新）
          ↓
  加入任务特定Head（分类层、CRF层、生成解码器等）
          ↓
  在少量标注数据上微调（数百 ~ 数万样本）
          ↓
  下游任务（分类、NER、QA、摘要、翻译...）
```

这个范式的优势在于：预训练模型已经在海量数据上学到了丰富的语言知识（词义、语法、常识推理等），微调只需要让模型"适应"特定任务的格式与分布，所需数据量和计算量大幅减少。

### 17.1.2 为什么微调有效

微调之所以能够奏效，根本原因在于**表示的可迁移性**。预训练模型在各层学到的特征可以分层理解：

| 层次 | 学到的特征 | 可迁移性 |
|------|-----------|---------|
| 底层（1-4层） | 词形、词素、基础句法（词性、依存关系） | 极高，几乎对所有任务通用 |
| 中层（5-8层） | 短语结构、实体信息、局部语义 | 较高，对大多数NLU任务有用 |
| 高层（9-12层） | 长距离依赖、任务相关语义、推理 | 中等，更偏向预训练任务本身 |

这一分层特性已被大量可视化研究（如 probing task 实验）所证实。底层特征几乎对所有下游任务都有用，而高层特征则相对更任务特定，这也是冻结策略的理论基础。

从损失曲面的角度来看，预训练将模型参数推入了一个"良好盆地"（good basin）——该区域在参数空间中更平坦、泛化性更好。微调在这个盆地内进行局部优化，比从随机初始化出发的从头训练更容易收敛到好的解。

$$\theta^* = \underset{\theta}{\arg\min} \mathcal{L}_{\text{finetune}}(\theta) \quad \text{以 } \theta_{\text{pretrained}} \text{ 为起点}$$

### 17.1.3 迁移学习的本质

迁移学习的核心假设是：**源任务（预训练）与目标任务（微调）共享有用的特征表示**。对于NLP而言，这一假设几乎总是成立的，因为所有任务都依赖对自然语言的理解。

迁移学习的效果受以下因素影响：

- **领域相似性**：预训练语料与下游任务语料越相似，迁移效果越好（如用法律文本预训练的模型在法律任务上表现更优）
- **任务相似性**：预训练目标与下游任务越接近，微调所需数据越少（如用MLM预训练的模型在填空类任务上表现尤为出色）
- **模型容量**：更大的模型通常具有更强的表示能力，迁移效果也更好，但计算成本同样更高

### 17.1.4 微调的类型

根据对预训练参数的处理方式，微调可以分为以下几种主要类型：

**类型一：特征提取（Feature Extraction）**

预训练模型的参数完全冻结，仅将其输出的隐层表示作为特征，输入一个轻量级的下游分类器（如线性层、SVM）。

```
[冻结的BERT] → [CLS表示] → [线性分类器] → [预测结果]
     ↑
  参数不更新
```

优点：计算成本低，不存在灾难性遗忘，适合数据极少的场景。
缺点：无法让表示适应目标任务的特定语义，上限较低。

**类型二：全参数微调（Full Fine-tuning）**

所有参数（包括预训练模型和任务头）都在下游任务数据上更新。这是最常见、效果通常最好的方式。

```
[BERT（所有参数可更新）] → [任务头] → [预测结果]
           ↑
      所有参数更新
```

**类型三：部分微调（Partial Fine-tuning）**

只更新部分层的参数（如顶部若干层 + 任务头），底层冻结。这是特征提取与全参数微调之间的折中方案。

**类型四：参数高效微调（Parameter-Efficient Fine-tuning，PEFT）**

在预训练参数基本冻结的前提下，通过添加少量可训练参数（如 Adapter、LoRA、Prefix-Tuning）实现高效微调。（本教程将在第18章专门介绍PEFT）

| 微调类型 | 可训练参数量 | 效果上限 | 计算成本 | 适用场景 |
|---------|-----------|---------|---------|---------|
| 特征提取 | 极少（仅任务头） | 低 | 极低 | 数据极少、资源受限 |
| 部分微调 | 少 | 中 | 低 | 数据较少、资源有限 |
| 全参数微调 | 全部 | 高 | 高 | 数据充足、资源充裕 |
| PEFT（LoRA等） | 少（0.1%~1%） | 接近全参数 | 低~中 | 资源受限、多任务部署 |

---

## 17.2 全参数微调

### 17.2.1 全参数微调的流程

全参数微调（Full Fine-tuning）是指将预训练模型的所有参数都纳入梯度更新。整个流程可以总结为：

```
1. 加载预训练模型（包含所有预训练权重）
2. 在模型顶部添加任务特定的 Head（如分类层）
3. 在下游任务数据上运行正向传播，计算任务损失
4. 通过反向传播计算所有参数的梯度
5. 使用小学习率更新所有参数
6. 重复步骤 3-5 直至收敛
```

对于分类任务，典型的损失函数为交叉熵：

$$\mathcal{L} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c}$$

其中 $N$ 是样本数，$C$ 是类别数，$y_{i,c}$ 是真实标签的 one-hot 编码，$\hat{y}_{i,c}$ 是模型预测的概率。

### 17.2.2 灾难性遗忘问题

全参数微调面临的最大挑战是**灾难性遗忘（Catastrophic Forgetting）**：模型在适应新任务时，可能"忘记"预训练阶段学到的通用知识，导致：

- 过拟合下游训练集，泛化能力差
- 模型在与下游任务分布略有不同的数据上性能骤降
- 多任务能力丧失（微调后的模型只擅长单一任务）

灾难性遗忘的根本原因是：当学习率过大或训练步数过多时，梯度更新会大幅偏离预训练权重所在的"良好盆地"。

**量化"遗忘程度"的方式：**

可以计算微调前后权重的 $L_2$ 距离来评估参数偏移量：

$$\Delta\theta = \|\theta_{\text{fine-tuned}} - \theta_{\text{pretrained}}\|_2$$

研究表明，微调效果好的模型通常 $\Delta\theta$ 较小，说明参数在"良好盆地"内移动，而非大幅跳跃。

**缓解灾难性遗忘的策略：**

1. **使用小学习率**：微调学习率通常比预训练低 1~2 个数量级（如 $2 \times 10^{-5}$ vs $1 \times 10^{-4}$）
2. **早停（Early Stopping）**：监控验证集性能，在过拟合发生前停止训练
3. **L2正则化（权重衰减）**：在损失中加入 $\lambda \|\theta\|_2^2$ 项，防止参数偏移过大
4. **分层学习率**：底层使用更小的学习率，减少底层特征的破坏（见17.4节）
5. **混合训练**：在微调数据中混入少量预训练语料，保持通用能力

### 17.2.3 微调数据量的影响

微调数据量对最终效果有显著影响，但其与模型大小的关系并非线性：

| 数据规模 | 典型行为 | 建议策略 |
|---------|---------|---------|
| < 100 条 | 极易过拟合，效果不稳定 | 特征提取或少样本提示 |
| 100 ~ 1,000 条 | 过拟合风险高 | 部分微调 + 强正则化 |
| 1,000 ~ 10,000 条 | 可以全参数微调，需谨慎调参 | 全参数微调 + 早停 |
| 10,000 ~ 100,000 条 | 全参数微调表现稳定 | 标准全参数微调流程 |
| > 100,000 条 | 接近继续预训练 | 可考虑更大学习率，更多 epochs |

经验规则：数据量越少，应该"动"的参数越少（即冻结更多层），使用的学习率越小，训练的 epoch 数越少。

### 17.2.4 何时使用全参数微调

全参数微调适用于以下场景：

- **标注数据充足**（通常 > 10,000 条）：有足够的监督信号来稳健地更新所有参数
- **目标任务与预训练任务差异较大**：需要对高层特征进行较大幅度的调整
- **对性能要求极高**：在计算资源允许的情况下，全参数微调通常给出最佳效果
- **单任务部署**：不需要同时服务多个任务，可以为每个任务保存一份完整的参数副本

反之，当数据量少、计算资源有限、或需要同时服务多个任务时，应优先考虑 PEFT 方法（第18章）。

---

## 17.3 冻结策略

### 17.3.1 冻结底层、微调顶层

最直观的冻结策略是：**冻结底层 Transformer 块，只微调顶部若干层和任务头**。

其背后的逻辑是：底层学习的是通用语言特征（词法、句法），这些特征对几乎所有任务都有用，不应该被破坏；而顶层学习的是更抽象、更任务相关的特征，需要针对下游任务进行调整。

```python
# 冻结BERT的前N层
def freeze_bottom_layers(model, num_frozen_layers):
    # 冻结embedding层
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    # 冻结前num_frozen_layers个Transformer块
    for i in range(num_frozen_layers):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False
```

**如何选择冻结多少层？**

一个粗略的经验准则是：

- 数据量极少（< 1,000）：冻结约 75% 的层（如 BERT-base 冻结前 9 层，只微调后 3 层 + 任务头）
- 数据量较少（1,000 ~ 5,000）：冻结约 50% 的层
- 数据量充足（> 10,000）：不冻结或只冻结底部 1~2 层

### 17.3.2 渐进式解冻（Gradual Unfreezing）

渐进式解冻（由 Howard & Ruder 在 ULMFiT 中提出）是一种更精细的策略：从任务头开始，逐步"解冻"并微调更深的层。

```
第1阶段：只训练任务头（最高层）
第2阶段：解冻最后几个Transformer块，继续训练
第3阶段：解冻更多层，继续训练
...
最终阶段：所有层都解冻，以极小的学习率继续训练
```

这种策略的优点是：
- 在解冻深层之前，任务头已经学到了稳定的任务相关表示
- 每次解冻新层时，梯度信号来自已经适应任务的顶层，而非随机初始化的任务头
- 大幅降低灾难性遗忘的风险

渐进式解冻通常用于数据量较少但希望尽可能利用预训练知识的场景。

### 17.3.3 冻结 Embedding 层

Embedding 层（包括词向量、位置编码、token type embedding）是一个特殊的冻结对象。冻结 Embedding 的理由是：

1. **Embedding 层存储了最基础的词汇语义知识**，这些知识在几乎所有任务中都是有用且正确的
2. **Embedding 层参数量大**（词汇表大小 × 隐层维度，如 30,522 × 768 ≈ 2300 万参数），冻结后可以显著减少需要更新的参数数量
3. **微调数据中可能没有覆盖所有词汇**，允许未见词的 embedding 更新可能导致噪声

```python
# 冻结Embedding层
for param in model.bert.embeddings.parameters():
    param.requires_grad = False

# 验证冻结效果
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,}")
print(f"可训练比例: {trainable_params/total_params*100:.1f}%")
```

### 17.3.4 选择性微调

选择性微调（Selective Fine-tuning）指根据参数的重要性或特性，有针对性地选择哪些参数进行更新。常见的选择性策略包括：

**只微调 LayerNorm 参数**：一些研究发现，只更新 Transformer 中 LayerNorm 的 $\gamma$ 和 $\beta$ 参数（参数量 < 0.1%）就能获得接近全参数微调的效果，同时极大地保留了预训练知识。

**只微调 Attention 的偏置项**：BitFit（Ben Zaken et al., 2022）发现，只更新 attention 和 FFN 中的偏置参数（约 0.1% 的参数量）就能在多个 NLU 任务上达到全参数微调 90%+ 的性能。

**基于梯度幅度的选择**：计算每个参数在验证集上的梯度幅度，只更新梯度最大的 k% 的参数。

| 策略 | 可训练参数比例 | 效果（相对全参数微调）| 实现难度 |
|------|-------------|-------------------|---------|
| 只微调任务头 | < 1% | 70-80% | 简单 |
| 冻结底层N层 | 20-80% | 85-95% | 简单 |
| 渐进式解冻 | 最终100% | 95-100% | 中等 |
| 只微调LayerNorm | < 0.1% | 85-92% | 中等 |
| BitFit（偏置微调）| 约0.1% | 88-95% | 中等 |

---

## 17.4 学习率设置

### 17.4.1 微调学习率的基本原则

微调学习率通常比预训练学习率小 1~2 个数量级。这是因为：

- 预训练权重已经处于一个"良好盆地"，我们希望在盆地内小步移动，而非大跨步跳出
- 大学习率会导致灾难性遗忘，破坏预训练学到的通用表示
- 下游任务数据集通常较小，大学习率容易过拟合

**典型学习率对比：**

| 训练阶段 | 典型学习率范围 |
|---------|-------------|
| 预训练（从头） | $1 \times 10^{-4}$ ~ $5 \times 10^{-4}$ |
| 全参数微调 | $1 \times 10^{-5}$ ~ $5 \times 10^{-5}$ |
| 顶层微调 | $1 \times 10^{-4}$ ~ $1 \times 10^{-3}$ |
| 任务头（随机初始化）| $1 \times 10^{-3}$ ~ $1 \times 10^{-2}$ |

### 17.4.2 分层学习率（Layer-wise LR Decay）

分层学习率衰减（LLRD, Layer-wise Learning Rate Decay）是微调中的重要技术，由 Howard & Ruder（2018）在 ULMFiT 中提出，并在 BERT 微调的最佳实践中被广泛采用。

其核心思想是：**越靠近输入端（底层）的参数，使用越小的学习率；越靠近输出端（顶层）的参数，使用越大的学习率**。

数学上，若顶层学习率为 $\eta$，衰减因子为 $\alpha$（通常取 0.8~0.95），则第 $l$ 层（从顶部数起第 $l$ 层）的学习率为：

$$\eta_l = \eta \cdot \alpha^{l}$$

例如，对于 12 层的 BERT-base，若顶层学习率为 $2 \times 10^{-5}$，衰减因子为 $0.9$：

| 层编号 | 层类型 | 学习率 |
|-------|-------|-------|
| 12（顶层）| 最后一个 Transformer 块 | $2.00 \times 10^{-5}$ |
| 11 | 倒数第2层 | $1.80 \times 10^{-5}$ |
| 10 | 倒数第3层 | $1.62 \times 10^{-5}$ |
| ... | ... | ... |
| 1（底层）| 第1个 Transformer 块 | $6.97 \times 10^{-6}$ |
| 0 | Embedding 层 | $6.28 \times 10^{-6}$ |

```python
def get_layerwise_lr_groups(model, base_lr, decay_factor=0.9):
    """
    为BERT模型生成分层学习率参数组

    Args:
        model: BERT模型
        base_lr: 顶层（Transformer最后一层）的学习率
        decay_factor: 每下降一层的学习率乘以该因子

    Returns:
        optimizer的参数组列表
    """
    # 获取BERT的层数
    num_layers = model.config.num_hidden_layers  # 通常为12

    param_groups = []

    # 任务头使用更高的学习率（通常是base_lr的10倍）
    param_groups.append({
        'params': model.classifier.parameters(),
        'lr': base_lr * 10
    })

    # Transformer层：从顶层到底层，学习率递减
    for layer_idx in range(num_layers - 1, -1, -1):
        # 从顶层数起的距离
        distance_from_top = num_layers - 1 - layer_idx
        layer_lr = base_lr * (decay_factor ** distance_from_top)

        param_groups.append({
            'params': model.bert.encoder.layer[layer_idx].parameters(),
            'lr': layer_lr
        })

    # Embedding层使用最小的学习率
    embedding_lr = base_lr * (decay_factor ** num_layers)
    param_groups.append({
        'params': model.bert.embeddings.parameters(),
        'lr': embedding_lr
    })

    return param_groups
```

### 17.4.3 判别式微调（Discriminative Fine-tuning）

判别式微调（Discriminative Fine-tuning）与分层学习率本质相同，但名称来自 ULMFiT 论文。其思想是：模型的不同层应该以不同速率"区分性地"学习，而不是用统一的学习率更新所有层。

从优化的角度理解，统一学习率会导致：
- 底层参数（已经很好地优化）被过度更新，丢失通用知识
- 高层参数（需要适应新任务）更新幅度可能不够

判别式微调通过为不同层分配不同学习率，让优化器能够针对各层的"最优更新量"进行调整。

### 17.4.4 学习率预热（Warmup）

学习率预热是指在训练初期，将学习率从极小值线性增加到目标学习率，然后再按某种策略衰减。

**为什么微调也需要预热？**

微调初期，任务头是随机初始化的，其梯度可能远大于预训练层的梯度。如果一开始就使用全量学习率，较大的梯度可能导致预训练权重在任务头稳定之前就被大幅破坏。预热期让任务头先"稳定下来"，再以正常学习率更新整个模型。

常用的学习率调度策略：

**线性预热 + 线性衰减：**

$$\eta(t) = \begin{cases} \eta_{\max} \cdot \dfrac{t}{t_{\text{warmup}}} & t \leq t_{\text{warmup}} \\ \eta_{\max} \cdot \dfrac{T - t}{T - t_{\text{warmup}}} & t > t_{\text{warmup}} \end{cases}$$

**线性预热 + 余弦衰减：**

$$\eta(t) = \begin{cases} \eta_{\max} \cdot \dfrac{t}{t_{\text{warmup}}} & t \leq t_{\text{warmup}} \\ \dfrac{\eta_{\max}}{2} \left(1 + \cos\left(\pi \cdot \dfrac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}}\right)\right) & t > t_{\text{warmup}} \end{cases}$$

其中 $T$ 是总训练步数，$t_{\text{warmup}}$ 是预热步数（通常取总步数的 6%~10%）。

```python
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# 线性预热 + 线性衰减（Hugging Face默认）
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps * 0.1,   # 10%预热
    num_training_steps=total_steps
)

# 线性预热 + 余弦衰减
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps * 0.1,
    num_training_steps=total_steps
)
```

---

## 17.5 微调实战技巧

### 17.5.1 早停（Early Stopping）

早停是防止过拟合最直接有效的策略。基本思路是：在每个 epoch（或每隔 N 步）评估验证集性能，若验证集指标在连续 $k$ 次评估中没有改善，则停止训练。

```
训练过程中的性能曲线：
          ↑ 性能
          │          ★ ← 最佳验证集性能（此处保存checkpoint）
          │        ╱   ╲
          │      ╱       ╲ ← 开始过拟合
          │    ╱           ╲
  训练集 ─┼──────────────────────→ 时间
  验证集 ─┼────★──────
          │         ↑
          │    Early Stop 触发点
```

关键参数：
- **patience**（耐心值）：允许验证集性能不提升的最大评估次数，通常设为 3~10
- **min_delta**：认为性能"有改善"的最小阈值，避免噪声触发提前停止
- **monitor**：监控的指标（验证集 loss、accuracy、F1 等）

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # 'max'表示越大越好（如accuracy），'min'表示越小越好（如loss）
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta
```

**早停的最佳实践：**

1. **保存最佳 checkpoint**：每次验证集性能创新高时保存模型，训练结束后加载最佳 checkpoint 进行推理
2. **合理设置 patience**：太小容易过早停止（欠拟合），太大容易训练太长（浪费时间）
3. **评估频率**：对于小数据集，每个 epoch 评估一次；对于大数据集，可以每 N 步评估一次

### 17.5.2 正则化策略

**权重衰减（Weight Decay / L2正则化）**

在 AdamW 优化器中，权重衰减直接作用于参数更新（而非梯度），等效于 L2 正则化：

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L} - \eta \lambda \theta_t$$

其中 $\lambda$ 是权重衰减系数（通常取 $0.01$）。注意：通常不对偏置项（bias）和 LayerNorm 的参数施加权重衰减。

```python
# 对不同参数组应用不同的权重衰减
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    },
    {
        'params': [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
```

**Dropout**

预训练模型中已经内置了 Dropout，微调时通常保持与预训练相同的 dropout rate（BERT 中为 0.1）。对于数据极少的场景，可以适当增大 dropout rate（如 0.2~0.3）来增强正则化效果。

**标签平滑（Label Smoothing）**

将硬标签（0或1）替换为软标签（$\epsilon/(C-1)$ 或 $1-\epsilon$），防止模型对预测过度自信：

$$\mathcal{L}_{\text{smooth}} = (1-\epsilon) \mathcal{L}_{\text{CE}} + \frac{\epsilon}{C} \sum_{c=1}^{C} (-\log \hat{y}_c)$$

其中 $\epsilon$ 通常取 0.1。标签平滑在数据量较少时效果尤为明显。

### 17.5.3 数据增强

对于 NLP 微调，常用的数据增强方法包括：

**词汇替换（Token Replacement）**

- **同义词替换**：随机将句子中的词替换为其同义词（可使用 WordNet）
- **随机删除（Random Deletion）**：以概率 $p$ 随机删除句子中的每个词
- **随机交换（Random Swap）**：随机交换句子中两个词的位置

**回译（Back Translation）**

将文本翻译成另一种语言，再翻译回来，得到语义相同但表述不同的文本：

```
原始：The movie was absolutely fantastic!
→ 中文：这部电影真是太精彩了！
→ 回译：The film was truly spectacular!
```

**EDA（Easy Data Augmentation）**

Wei & Zou (2019) 提出的简单而有效的四种操作：
1. 同义词替换（SR）：将 $n$ 个随机词替换为其同义词
2. 随机插入（RI）：随机在句子中插入某个词的同义词
3. 随机交换（RS）：随机交换两个词的位置，重复 $n$ 次
4. 随机删除（RD）：以概率 $p$ 随机删除每个词

**Mixup for NLP**

在 embedding 空间对两个样本进行线性插值：

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$

### 17.5.4 混合精度微调（Mixed Precision Training）

混合精度训练使用 FP16（半精度浮点数）进行前向传播和梯度计算，用 FP32（单精度）保存主权重（master weights），从而在保持模型精度的同时大幅减少显存占用和加快训练速度。

**为什么混合精度适用于微调？**

- 预训练模型的权重值通常在 FP16 可表示的范围内
- 微调的梯度更小（小学习率），梯度下溢是主要风险，需通过梯度缩放（Loss Scaling）解决
- GPU（尤其是 Volta 架构后的 NVIDIA GPU）对 FP16 Tensor Core 有专门的硬件加速，速度可提升 2~3 倍

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # 自动混合精度前向传播
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss

    # 缩放损失，反向传播
    scaler.scale(loss).backward()

    # 梯度裁剪（在反缩放后进行）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 更新参数
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
```

**FP16 vs BF16：**

| 格式 | 指数位 | 尾数位 | 动态范围 | 精度 | 适用硬件 |
|-----|-------|-------|---------|-----|---------|
| FP32 | 8 | 23 | 大 | 高 | 所有GPU |
| FP16 | 5 | 10 | 小（易上下溢出）| 中 | 所有支持半精度的GPU |
| BF16 | 8 | 7 | 大（与FP32相同）| 较低 | A100, H100, TPU |

对于微调，BF16 通常比 FP16 更稳定（无需 Loss Scaling），在支持 BF16 的硬件（如 A100）上优先使用。

```python
# 使用BF16（需要支持的硬件）
model = model.to(torch.bfloat16)
# 或在Trainer中指定
from transformers import TrainingArguments
args = TrainingArguments(bf16=True, ...)
```

---

## 本章小结

本章系统介绍了 Transformer 预训练模型的微调技术，涵盖从基本原理到实战技巧的完整知识体系。

**各种微调策略对比总结：**

| 维度 | 特征提取 | 部分微调 | 渐进式解冻 | 全参数微调 | PEFT（第18章）|
|-----|---------|---------|----------|----------|-------------|
| 可训练参数 | 极少（仅任务头）| 少~中 | 最终全部 | 全部 | 极少（0.1%~1%）|
| 数据需求 | 极少 | 少~中 | 少~中 | 充足 | 少~中 |
| 灾难性遗忘风险 | 无 | 低 | 低 | 中等 | 低 |
| 性能上限 | 较低 | 中 | 较高 | 最高 | 接近全参数 |
| 计算成本 | 极低 | 低~中 | 中 | 高 | 低~中 |
| 适用场景 | 资源极限/数据极少 | 中小数据 | 中小数据 | 大数据/高性能 | 资源受限/多任务 |

**关键技术总结：**

| 技术 | 关键参数 | 典型取值 | 作用 |
|-----|---------|---------|-----|
| 学习率 | `lr` | $1$~$5 \times 10^{-5}$ | 防止灾难性遗忘 |
| 分层学习率衰减 | `decay_factor` | 0.8~0.95 | 保护底层特征 |
| 预热步数 | `warmup_ratio` | 6%~10% | 稳定初期训练 |
| 权重衰减 | `weight_decay` | 0.01 | L2正则化 |
| 早停 patience | `patience` | 3~10 | 防止过拟合 |
| Batch Size | `batch_size` | 16~32 | 影响收敛稳定性 |
| 训练 Epochs | `num_epochs` | 3~5 | 充分利用数据 |

**选择微调策略的决策流程：**

```
数据量 < 100 条？ → 使用提示学习或少样本学习（第20章）
           ↓ 否
计算资源严重受限？ → 使用PEFT（第18章，LoRA等）
           ↓ 否
数据量 < 5,000 条？ → 部分微调（冻结底层50%~75%）+ 强正则化
           ↓ 否
追求极致性能？ → 全参数微调 + 分层学习率 + 混合精度
           ↓ 否
标准场景 → 全参数微调（BERT官方推荐配置）
```

---

## 代码实战

### 完整微调代码：文本分类

以下是使用 Hugging Face Transformers 对 BERT 进行情感分类微调的完整代码，包含分层学习率、早停、混合精度等所有最佳实践。

```python
"""
第17章代码实战：使用BERT进行文本分类微调
包含：分层学习率、冻结层、早停、混合精度训练
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 1. 数据集定义
# ============================================================

class TextClassificationDataset(Dataset):
    """文本分类数据集"""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BertTokenizer,
        max_length: int = 128
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings['token_type_ids'][idx],
            'labels': self.labels[idx]
        }


# ============================================================
# 2. 冻结层工具函数
# ============================================================

def freeze_layers(model: BertForSequenceClassification, num_frozen_layers: int):
    """
    冻结BERT的底部N层（包括Embedding层）

    Args:
        model: BERT分类模型
        num_frozen_layers: 冻结的Transformer层数（从底部算起）
    """
    # 冻结Embedding层
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    logger.info("已冻结: Embedding层")

    # 冻结前num_frozen_layers个Transformer块
    for i in range(num_frozen_layers):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False
        logger.info(f"已冻结: Transformer层 {i}")

    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable:,} / 总参数: {total:,} ({trainable/total*100:.1f}%)")


def unfreeze_all_layers(model: BertForSequenceClassification):
    """解冻所有层"""
    for param in model.parameters():
        param.requires_grad = True
    logger.info("已解冻所有层")


# ============================================================
# 3. 分层学习率
# ============================================================

def get_optimizer_with_layerwise_lr(
    model: BertForSequenceClassification,
    base_lr: float = 2e-5,
    decay_factor: float = 0.9,
    weight_decay: float = 0.01
) -> AdamW:
    """
    创建带分层学习率的AdamW优化器

    Args:
        model: BERT分类模型
        base_lr: 最顶层Transformer的学习率
        decay_factor: 每向下一层，学习率乘以该因子
        weight_decay: 权重衰减系数

    Returns:
        配置好分层学习率的AdamW优化器
    """
    no_decay = ['bias', 'LayerNorm.weight']
    num_layers = model.config.num_hidden_layers  # BERT-base: 12

    param_groups = []

    # 任务头（分类器）：使用10倍的base_lr
    classifier_lr = base_lr * 10
    param_groups.extend([
        {
            'params': [p for n, p in model.classifier.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'lr': classifier_lr,
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.classifier.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'lr': classifier_lr,
            'weight_decay': 0.0
        }
    ])
    logger.info(f"分类器层学习率: {classifier_lr:.2e}")

    # Transformer层：从顶层到底层，学习率递减
    for layer_idx in range(num_layers - 1, -1, -1):
        distance_from_top = (num_layers - 1) - layer_idx
        layer_lr = base_lr * (decay_factor ** distance_from_top)
        layer = model.bert.encoder.layer[layer_idx]

        param_groups.extend([
            {
                'params': [p for n, p in layer.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                'lr': layer_lr,
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in layer.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                'lr': layer_lr,
                'weight_decay': 0.0
            }
        ])
        if layer_idx in [0, num_layers // 2, num_layers - 1]:
            logger.info(f"Transformer层 {layer_idx} 学习率: {layer_lr:.2e}")

    # Embedding层：使用最小的学习率
    embedding_lr = base_lr * (decay_factor ** num_layers)
    param_groups.extend([
        {
            'params': [p for n, p in model.bert.embeddings.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            'lr': embedding_lr,
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.bert.embeddings.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            'lr': embedding_lr,
            'weight_decay': 0.0
        }
    ])
    logger.info(f"Embedding层学习率: {embedding_lr:.2e}")

    return AdamW(param_groups)


# ============================================================
# 4. 早停
# ============================================================

class EarlyStopping:
    """早停机制，监控验证集性能"""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = 'max'):
        """
        Args:
            patience: 允许不改善的最大评估次数
            min_delta: 认为"有改善"的最小变化量
            mode: 'max'（如accuracy/F1）或 'min'（如loss）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def step(self, score: float, epoch: int) -> bool:
        """
        更新早停状态

        Returns:
            True 表示应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: {self.counter}/{self.patience} (最佳: {self.best_score:.4f} @ epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered! 最佳性能在 epoch {self.best_epoch}")

        return self.should_stop

    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


# ============================================================
# 5. 训练与评估函数
# ============================================================

def train_epoch(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    use_amp: bool = True,
    max_grad_norm: float = 1.0
) -> float:
    """训练一个epoch，返回平均损失"""
    model.train()
    scaler = GradScaler() if use_amp else None
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        # 将数据移动到设备
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            logger.info(f"  Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float]:
    """评估模型，返回(loss, accuracy, macro_f1)"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast():
                outputs = model(**batch)

            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, macro_f1


# ============================================================
# 6. 完整微调流程
# ============================================================

def finetune_bert(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    num_classes: int,
    model_name: str = 'bert-base-chinese',
    max_length: int = 128,
    batch_size: int = 32,
    num_epochs: int = 5,
    base_lr: float = 2e-5,
    decay_factor: float = 0.9,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    num_frozen_layers: int = 0,        # 0表示不冻结任何层
    use_layerwise_lr: bool = True,
    use_amp: bool = True,
    early_stopping_patience: int = 5,
    output_dir: str = './finetuned_model'
) -> BertForSequenceClassification:
    """
    完整的BERT微调流程

    Args:
        train_texts: 训练集文本列表
        train_labels: 训练集标签列表
        val_texts: 验证集文本列表
        val_labels: 验证集标签列表
        num_classes: 分类类别数
        model_name: 预训练模型名称或路径
        max_length: 最大序列长度
        batch_size: 批大小
        num_epochs: 最大训练轮数
        base_lr: 基础学习率（顶层Transformer使用该学习率）
        decay_factor: 分层学习率衰减因子
        warmup_ratio: 预热步数占总步数的比例
        weight_decay: 权重衰减系数
        num_frozen_layers: 冻结底部的层数
        use_layerwise_lr: 是否使用分层学习率
        use_amp: 是否使用混合精度训练
        early_stopping_patience: 早停的patience值
        output_dir: 模型保存目录

    Returns:
        微调后的最佳模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # ---- 1. 加载tokenizer和模型 ----
    logger.info(f"加载预训练模型: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    model = model.to(device)

    # ---- 2. 冻结层（如果需要）----
    if num_frozen_layers > 0:
        freeze_layers(model, num_frozen_layers)

    # ---- 3. 创建数据集和DataLoader ----
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=4)

    # ---- 4. 创建优化器 ----
    if use_layerwise_lr:
        optimizer = get_optimizer_with_layerwise_lr(
            model, base_lr=base_lr, decay_factor=decay_factor, weight_decay=weight_decay
        )
    else:
        # 简单版本：统一学习率，但不对bias和LayerNorm做weight_decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer = AdamW([
            {'params': [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ], lr=base_lr)

    # ---- 5. 创建学习率调度器 ----
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    logger.info(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")

    # ---- 6. 早停 ----
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
    best_model_state = None

    # ---- 7. 训练循环 ----
    logger.info("开始训练...")

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{num_epochs}")
        logger.info(f"{'='*50}")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, use_amp)

        # 评估
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # 保存最佳模型
        if early_stopping.best_score is None or val_f1 > early_stopping.best_score:
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(f"保存最佳模型 (Val F1: {val_f1:.4f})")

        # 早停检查
        if early_stopping.step(val_f1, epoch):
            break

    # ---- 8. 加载最佳模型 ----
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"已加载最佳模型 (来自 epoch {early_stopping.best_epoch})")

    # ---- 9. 保存模型 ----
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"模型已保存至: {output_dir}")

    return model


# ============================================================
# 7. 使用示例
# ============================================================

def demo_finetuning():
    """
    微调演示：使用模拟数据进行情感分类
    实际使用时替换为真实数据集
    """
    # 模拟训练数据（实际使用时替换为真实数据）
    train_texts = [
        "这部电影真的很精彩，强烈推荐！",
        "剧情拖沓，演技一般，浪费时间。",
        "特效震撼，故事感人，值得一看。",
        "无聊透顶，毫无逻辑，烂片一部。",
        # ... 实际应有数千条数据
    ] * 100  # 模拟扩充数据量

    train_labels = [1, 0, 1, 0] * 100  # 1=正面，0=负面

    val_texts = [
        "画面精美，配乐动听，非常享受。",
        "故事老套，没有新意，不推荐。",
    ] * 20

    val_labels = [1, 0] * 20

    # 运行微调
    model = finetune_bert(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        num_classes=2,
        model_name='bert-base-chinese',
        max_length=128,
        batch_size=32,
        num_epochs=5,
        base_lr=2e-5,
        decay_factor=0.9,
        warmup_ratio=0.1,
        weight_decay=0.01,
        num_frozen_layers=0,        # 不冻结层
        use_layerwise_lr=True,      # 使用分层学习率
        use_amp=True,               # 使用混合精度
        early_stopping_patience=3,
        output_dir='./sentiment_model'
    )

    return model


# ============================================================
# 8. 渐进式解冻示例
# ============================================================

def gradual_unfreezing_example(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_stages: int = 4
):
    """
    渐进式解冻微调示例

    策略：
      阶段1: 只训练任务头
      阶段2: 解冻顶部1/3的Transformer层
      阶段3: 解冻顶部2/3的Transformer层
      阶段4: 解冻所有层（包括Embedding）
    """
    num_layers = model.config.num_hidden_layers  # 12
    layers_per_stage = num_layers // (num_stages - 1)  # 每阶段解冻的层数

    # 初始：冻结所有BERT层
    for param in model.bert.parameters():
        param.requires_grad = False

    for stage in range(1, num_stages + 1):
        logger.info(f"\n{'='*40}")
        logger.info(f"渐进式解冻 - 阶段 {stage}/{num_stages}")

        if stage == 1:
            # 第一阶段：只训练任务头
            logger.info("只训练分类头")
            trainable_params = model.classifier.parameters()
        elif stage < num_stages:
            # 中间阶段：逐步解冻更多层
            unfrozen_layers = min((stage - 1) * layers_per_stage, num_layers)
            start_layer = num_layers - unfrozen_layers
            for i in range(start_layer, num_layers):
                for param in model.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
            logger.info(f"解冻Transformer层 {start_layer}~{num_layers-1}")
        else:
            # 最后阶段：解冻所有层
            for param in model.bert.parameters():
                param.requires_grad = True
            logger.info("解冻所有层（包括Embedding）")

        # 统计可训练参数
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"可训练参数: {trainable:,} ({trainable/total*100:.1f}%)")

        # 每个阶段使用递减的学习率
        stage_lr = 2e-5 * (0.5 ** (stage - 1))
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer = AdamW([
            {'params': [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ], lr=stage_lr)

        total_steps = len(train_loader) * 2  # 每阶段训练2个epoch
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )

        # 训练当前阶段
        for epoch_in_stage in range(2):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
            logger.info(f"  Epoch {epoch_in_stage+1}: Train={train_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")


if __name__ == '__main__':
    # 运行演示
    demo_finetuning()
```

---

## 练习题

### 基础题

**练习17.1**（基础）：以下关于微调的说法中，哪些是正确的？请逐条判断并说明理由。

a) 微调时应该始终使用与预训练相同的学习率。

b) 冻结底层参数可以减少灾难性遗忘的风险。

c) 特征提取（冻结所有预训练参数）在任何情况下都比全参数微调效果差。

d) 分层学习率衰减的核心思想是越靠近输入端的层使用越小的学习率。

e) 早停的 patience 值越大，越容易出现过拟合。

---

**练习17.2**（基础）：给定一个 BERT-base 模型（12 层 Transformer，隐层维度 768），配置如下：
- 顶层（第12层）学习率：$\eta = 2 \times 10^{-5}$
- 分层衰减因子：$\alpha = 0.9$

请计算：
(a) 第 8 层的学习率是多少？
(b) Embedding 层（第0层底部）的学习率是多少？（视为第13层的衰减）
(c) 如果将衰减因子改为 $\alpha = 0.8$，第 1 层的学习率变为多少？

---

### 中级题

**练习17.3**（中级）：你有一个二分类情感分析任务，训练集只有 800 条标注数据，验证集 200 条。请设计一个完整的微调方案，包括：
- 选择哪种微调类型（特征提取/部分微调/全参数微调）及理由
- 冻结策略（冻结哪些层）
- 学习率设置
- 正则化策略
- 训练 epoch 数的估计

并说明每个选择的依据。

---

**练习17.4**（中级）：下面的代码实现了一个简单的微调流程，但存在 3 个问题（可能导致训练不稳定或效果差）。请找出并修复这些问题。

```python
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 问题代码
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(20):
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 直接用训练集最后的checkpoint进行评估
accuracy = evaluate_model(model, test_loader)
```

---

### 提高题

**练习17.5**（提高）：设计并实现一个**自适应冻结策略**：根据每层参数梯度的范数（gradient norm）动态决定是否冻结该层。

具体要求：
1. 在训练初期（前 N 步），让所有层可训练，并记录每层的平均梯度范数
2. 训练 N 步后，冻结梯度范数低于某阈值 $\tau$ 的层（这些层"变化意愿"小，说明预训练权重对当前任务已经适用）
3. 每隔 M 步重新评估是否需要解冻（如果该层的损失贡献变大，则解冻）

请实现该策略，并分析其优缺点。（可在代码框架基础上实现，不强制要求完整可运行）

---

## 练习答案

### 练习17.1 答案

**a) 错误。**微调学习率应远小于预训练学习率（通常小 1~2 个数量级）。原因：预训练权重已经收敛到良好区域，大学习率会导致灾难性遗忘，破坏已学到的通用语言知识。典型微调学习率为 $1$~$5 \times 10^{-5}$，而预训练学习率通常为 $1$~$5 \times 10^{-4}$。

**b) 正确。**底层学习的是通用语言特征（词法、句法），对几乎所有任务都有用。冻结底层意味着这些通用特征不会被下游任务的梯度破坏，从而降低灾难性遗忘的风险。

**c) 错误。**在训练数据极少（如 < 100 条）的场景下，特征提取的效果可能优于全参数微调，因为：(1) 数据量不足以稳健地更新所有参数；(2) 全参数微调在极少数据下极易过拟合；(3) 预训练的通用特征本身已经很强，直接用于简单分类器效果良好。

**d) 正确。**这正是分层学习率衰减（LLRD）的核心思想：底层参数学习的是更通用的特征，不需要大幅更新，因此使用更小的学习率；顶层参数需要适应特定任务，使用相对较大的学习率。

**e) 正确。**patience 值越大，允许验证集性能不提升的次数越多，模型在过拟合区域训练的时间越长，越容易最终过拟合。但也不能设置得太小（如1~2），否则可能因为偶然的性能波动而过早停止训练。

---

### 练习17.2 答案

**(a) 第 8 层的学习率：**

第 8 层从顶部（第 12 层）数起处于第 $12 - 8 = 4$ 层位置（顶层计为第 0 层）：

$$\eta_8 = \eta \cdot \alpha^{12-8} = 2 \times 10^{-5} \times 0.9^4 = 2 \times 10^{-5} \times 0.6561 \approx 1.31 \times 10^{-5}$$

**(b) Embedding 层的学习率：**

将 Embedding 层视为从顶部数起第 12 层位置（在最后一个 Transformer 层之下）：

$$\eta_{\text{emb}} = \eta \cdot \alpha^{12} = 2 \times 10^{-5} \times 0.9^{12} = 2 \times 10^{-5} \times 0.2824 \approx 5.65 \times 10^{-6}$$

**(c) 第 1 层在 $\alpha=0.8$ 时的学习率：**

第 1 层从顶部数起处于第 $12 - 1 = 11$ 层位置：

$$\eta_1 = 2 \times 10^{-5} \times 0.8^{11} = 2 \times 10^{-5} \times 0.0859 \approx 1.72 \times 10^{-6}$$

可以看出，衰减因子从 0.9 减小到 0.8 时，底层的学习率衰减幅度更大（约为顶层的 8.6% vs 31.4%），这意味着底层参数更新更保守，对预训练知识的保护更强。

---

### 练习17.3 答案

**推荐方案：部分微调 + 强正则化**

**1. 微调类型选择：部分微调（冻结底层）**

理由：800 条数据属于"数据量较少"的情形，全参数微调（约 1.1 亿参数）极易过拟合。纯特征提取上限太低（无法充分利用预训练知识），部分微调是最佳折中。

**2. 冻结策略：冻结前 8 层（共 12 层），只微调后 4 层 + 任务头**

```python
freeze_layers(model, num_frozen_layers=8)
# 可训练参数约为总参数的 35%（约 3800 万参数）
```

理由：数据量约 800 条，遵循经验准则（< 1,000 条冻结约 75%），后 4 层包含更任务相关的高层语义特征，足以适应情感分类任务。

**3. 学习率设置**

```python
# 任务头：较大学习率（随机初始化需要快速收敛）
classifier_lr = 5e-4

# 顶层 Transformer（第9~12层）：中等学习率
top_layers_lr = 2e-5

# 使用分层衰减（仅对可训练的4层）
decay_factor = 0.9
```

**4. 正则化策略**

```python
weight_decay = 0.05       # 较强的L2正则化（数据少时增强）
dropout = 0.2             # 稍大于默认值0.1
label_smoothing = 0.1     # 防止过度自信
early_stopping_patience = 5
```

**5. 训练 epoch 数**

建议：10~15 个 epoch（数据量少，需要多轮才能充分学习），结合早停（patience=5）防止过拟合。每个 epoch 约 25 个 batch（batch_size=32），总训练步数约 250~375 步。

**补充建议**：考虑数据增强（EDA 将 800 条扩充到 2000+ 条），以及 k 折交叉验证（如 5 折）来充分利用有限数据，得到更可靠的性能评估。

---

### 练习17.4 答案

代码存在以下 **3 个问题**：

**问题1：使用 SGD 而非 AdamW**

```python
# 错误
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 修复
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

原因：(a) 学习率 `1e-3` 对于微调预训练模型太大，会导致灾难性遗忘；(b) SGD 在 NLP 微调中收敛慢且不稳定，AdamW 是 Transformer 微调的标准优化器，具有自适应学习率和去耦合权重衰减；(c) 缺少权重衰减正则化。

**问题2：训练 epoch 数过多且缺少早停**

```python
# 错误
for epoch in range(20):
    ...
# 没有早停，没有保存最佳模型

# 修复
early_stopping = EarlyStopping(patience=5)
best_model_state = None

for epoch in range(20):  # 20作为上限
    train_epoch(...)
    val_loss, val_acc, _ = evaluate(model, val_loader, device)

    # 保存最佳模型
    if best_model_state is None or val_acc > early_stopping.best_score:
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 早停检查
    if early_stopping.step(val_acc, epoch):
        break

model.load_state_dict(best_model_state)
```

原因：20 个 epoch 对于微调预训练模型通常过多（BERT 论文推荐 3~5 epoch），容易过拟合。缺少早停和最佳 checkpoint 保存会导致最终使用的是过拟合后的模型。

**问题3：`optimizer.zero_grad()` 的位置错误**

```python
# 错误（zero_grad在step之后）
loss.backward()
optimizer.step()
optimizer.zero_grad()   # ← 应在step之前或backward之前

# 修复（zero_grad在backward之前）
optimizer.zero_grad()   # ← 在计算梯度之前清零
outputs = model(**batch)
loss = outputs.loss
loss.backward()
optimizer.step()
```

原因：将 `zero_grad()` 放在 `step()` 之后在逻辑上等价（清零后的梯度不会影响已经完成的 `step()`），但这是一种不好的习惯，在某些特殊场景（如梯度累积）下可能导致 Bug。标准做法是在每次反向传播之前清零梯度。

**修复后的完整代码：**

```python
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 修复1：使用AdamW，正确的学习率
no_decay = ['bias', 'LayerNorm.weight']
optimizer = AdamW([
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
], lr=2e-5)

# 添加学习率调度器
total_steps = len(train_loader) * 5
scheduler = get_linear_schedule_with_warmup(optimizer,
                                             num_warmup_steps=total_steps//10,
                                             num_training_steps=total_steps)

# 修复2：添加早停和最佳模型保存
early_stopping = EarlyStopping(patience=5, mode='max')
best_model_state = None
best_val_acc = 0

for epoch in range(20):  # 20作为上限，早停控制实际轮数
    for batch in train_loader:
        # 修复3：zero_grad在backward之前
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    val_acc = evaluate_model(model, val_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    if early_stopping.step(val_acc, epoch):
        break

# 加载最佳模型进行测试
model.load_state_dict(best_model_state)
accuracy = evaluate_model(model, test_loader)
```

---

### 练习17.5 答案（参考实现）

```python
class AdaptiveFreezing:
    """
    自适应冻结策略：基于梯度范数动态决定冻结哪些层
    """

    def __init__(
        self,
        model: BertForSequenceClassification,
        warmup_steps: int = 100,      # 前N步收集统计信息
        threshold_ratio: float = 0.1,  # 低于平均梯度范数的10%时冻结
        recheck_interval: int = 50     # 每M步重新检查
    ):
        self.model = model
        self.warmup_steps = warmup_steps
        self.threshold_ratio = threshold_ratio
        self.recheck_interval = recheck_interval
        self.step_count = 0

        # 记录每层的梯度范数历史
        self.num_layers = model.config.num_hidden_layers
        self.grad_norms: Dict[str, List[float]] = {
            f'layer_{i}': [] for i in range(self.num_layers)
        }
        self.grad_norms['embeddings'] = []

        # 当前冻结状态
        self.frozen_layers = set()

    def collect_grad_norms(self):
        """收集当前步的梯度范数"""
        for i in range(self.num_layers):
            layer = self.model.bert.encoder.layer[i]
            grad_norm = self._compute_grad_norm(layer)
            self.grad_norms[f'layer_{i}'].append(grad_norm)

        emb_norm = self._compute_grad_norm(self.model.bert.embeddings)
        self.grad_norms['embeddings'].append(emb_norm)

    def _compute_grad_norm(self, module: nn.Module) -> float:
        """计算模块所有参数梯度的L2范数"""
        total_norm = 0.0
        for param in module.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def update_freezing(self):
        """根据梯度范数统计更新冻结状态"""
        # 计算各层平均梯度范数
        avg_norms = {}
        for layer_name, norms in self.grad_norms.items():
            if norms:
                avg_norms[layer_name] = np.mean(norms[-50:])  # 最近50步的均值

        if not avg_norms:
            return

        # 计算所有层的平均值
        overall_avg = np.mean(list(avg_norms.values()))
        threshold = overall_avg * self.threshold_ratio

        logger.info(f"\n自适应冻结检查 (step {self.step_count}):")
        logger.info(f"  全局平均梯度范数: {overall_avg:.6f}")
        logger.info(f"  冻结阈值: {threshold:.6f}")

        for layer_name, avg_norm in avg_norms.items():
            if avg_norm < threshold:
                # 梯度很小，冻结该层
                if layer_name not in self.frozen_layers:
                    self._freeze_layer(layer_name)
                    self.frozen_layers.add(layer_name)
                    logger.info(f"  冻结 {layer_name} (梯度范数: {avg_norm:.6f} < {threshold:.6f})")
            else:
                # 梯度较大，解冻该层
                if layer_name in self.frozen_layers:
                    self._unfreeze_layer(layer_name)
                    self.frozen_layers.discard(layer_name)
                    logger.info(f"  解冻 {layer_name} (梯度范数: {avg_norm:.6f} >= {threshold:.6f})")

    def _freeze_layer(self, layer_name: str):
        module = self._get_module(layer_name)
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_layer(self, layer_name: str):
        module = self._get_module(layer_name)
        for param in module.parameters():
            param.requires_grad = True

    def _get_module(self, layer_name: str) -> nn.Module:
        if layer_name == 'embeddings':
            return self.model.bert.embeddings
        else:
            idx = int(layer_name.split('_')[1])
            return self.model.bert.encoder.layer[idx]

    def step(self):
        """每个训练步调用一次"""
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # 预热期：收集梯度统计，不冻结
            self.collect_grad_norms()
            if self.step_count == self.warmup_steps:
                logger.info("预热完成，开始自适应冻结")
                self.update_freezing()
        elif self.step_count % self.recheck_interval == 0:
            # 定期重新评估
            self.collect_grad_norms()
            self.update_freezing()


# 使用示例
def train_with_adaptive_freezing(model, train_loader, optimizer, scheduler, device):
    adaptive_freezing = AdaptiveFreezing(
        model,
        warmup_steps=100,
        threshold_ratio=0.1,
        recheck_interval=50
    )

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()

        # 收集梯度统计（在optimizer.step之前）
        adaptive_freezing.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
```

**优缺点分析：**

| 方面 | 优点 | 缺点 |
|-----|------|------|
| **自适应性** | 根据实际梯度动态调整，比手动设置更灵活 | 需要额外计算梯度范数，有一定开销 |
| **理论依据** | 梯度范数反映了参数"想要更新的强度"，是合理的冻结指标 | 梯度范数受学习率影响，可能不准确 |
| **鲁棒性** | 可以随训练进度自动解冻（适应分布变化）| 动态冻结/解冻可能导致训练不稳定 |
| **可解释性** | 可以观察哪些层被冻结，了解任务特性 | 调参复杂（threshold_ratio, recheck_interval 等）|

实际中，这种方法在研究场景下比在工业应用中更常见，生产环境通常倾向于使用更简单可预测的静态冻结策略。
