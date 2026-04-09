# 第19章：训练循环与优化

## 学习目标

完成本章学习后，你将能够：

1. 理解并使用 PyTorch 中常见的损失函数（MSELoss、CrossEntropyLoss、BCELoss），根据任务类型选择合适的损失函数
2. 掌握主流优化器（SGD、Adam、AdamW）的原理与参数配置，理解学习率对训练的影响
3. 编写规范的训练循环，正确处理前向传播、损失计算、反向传播与参数更新
4. 实现完整的验证与测试流程，正确使用 `model.eval()` 和 `torch.no_grad()`
5. 掌握模型的保存与加载方法，包括保存完整模型、状态字典以及训练检查点

---

## 19.1 损失函数

损失函数（Loss Function）衡量模型预测值与真实值之间的差距，是训练过程的核心指标。PyTorch 在 `torch.nn` 模块中提供了丰富的损失函数。

### 19.1.1 均方误差损失：MSELoss

均方误差（Mean Squared Error）适用于**回归任务**，计算预测值与真实值差值的平方均值。

$$\text{MSELoss} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

```python
import torch
import torch.nn as nn

# 创建损失函数实例
criterion = nn.MSELoss()

# 模拟预测值与真实值（回归任务）
predictions = torch.tensor([2.5, 3.0, 4.1, 5.2], dtype=torch.float32)
targets     = torch.tensor([2.0, 3.5, 4.0, 5.0], dtype=torch.float32)

loss = criterion(predictions, targets)
print(f"MSE Loss: {loss.item():.4f}")
# MSE Loss: 0.1350

# reduction 参数控制归约方式
criterion_sum  = nn.MSELoss(reduction='sum')   # 求和而非平均
criterion_none = nn.MSELoss(reduction='none')  # 返回每个样本的损失

loss_sum  = criterion_sum(predictions, targets)
loss_none = criterion_none(predictions, targets)
print(f"Sum  reduction: {loss_sum.item():.4f}")
print(f"None reduction: {loss_none}")
# Sum  reduction: 0.5400
# None reduction: tensor([0.2500, 0.2500, 0.0100, 0.0400])
```

### 19.1.2 交叉熵损失：CrossEntropyLoss

交叉熵损失（Cross Entropy Loss）适用于**多分类任务**。PyTorch 的 `CrossEntropyLoss` 内部已包含 Softmax，因此模型最后一层输出原始 logits 即可，**不需要**手动添加 Softmax。

$$\text{CrossEntropyLoss} = -\sum_{c=1}^{C} y_c \log(\hat{p}_c)$$

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# logits shape: (batch_size, num_classes)
# 3个样本，4个类别
logits  = torch.tensor([
    [2.0, 1.0, 0.5, 0.1],   # 样本1：预测类别0概率最大
    [0.3, 3.0, 0.2, 0.1],   # 样本2：预测类别1概率最大
    [0.1, 0.2, 0.3, 4.0],   # 样本3：预测类别3概率最大
])
# 真实标签（类别索引）
labels = torch.tensor([0, 1, 3])

loss = criterion(logits, labels)
print(f"CrossEntropy Loss: {loss.item():.4f}")
# CrossEntropy Loss: 0.2368

# 类别权重：处理样本不均衡
weights = torch.tensor([1.0, 2.0, 1.0, 1.5])
criterion_weighted = nn.CrossEntropyLoss(weight=weights)
loss_w = criterion_weighted(logits, labels)
print(f"Weighted Loss: {loss_w.item():.4f}")

# ignore_index：忽略特定标签（常用于 NLP padding）
criterion_ignore = nn.CrossEntropyLoss(ignore_index=-100)
labels_with_pad  = torch.tensor([0, -100, 3])   # -100 会被忽略
loss_ignore = criterion_ignore(logits, labels_with_pad)
print(f"Ignore Index Loss: {loss_ignore.item():.4f}")
```

### 19.1.3 二元交叉熵损失：BCELoss 与 BCEWithLogitsLoss

适用于**二分类**或**多标签分类**任务。

```python
import torch
import torch.nn as nn

# ---- BCELoss：输入必须先经过 Sigmoid ----
sigmoid   = nn.Sigmoid()
criterion = nn.BCELoss()

logits  = torch.tensor([1.5, -0.5, 2.0, -1.0])
targets = torch.tensor([1.0,  0.0, 1.0,  0.0])

probs = sigmoid(logits)           # 先 Sigmoid
loss  = criterion(probs, targets)
print(f"BCE Loss: {loss.item():.4f}")

# ---- BCEWithLogitsLoss：数值更稳定，推荐使用 ----
# 内部将 Sigmoid 与 BCE 合并，数值稳定性更好
criterion_logits = nn.BCEWithLogitsLoss()
loss2 = criterion_logits(logits, targets)
print(f"BCEWithLogits Loss: {loss2.item():.4f}")
# 两者结果相同，但 BCEWithLogitsLoss 更稳定

# pos_weight：处理正负样本不均衡
pos_weight = torch.tensor([3.0])   # 正样本权重放大3倍
criterion_pw = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss3 = criterion_pw(logits, targets)
print(f"Pos-weighted Loss: {loss3.item():.4f}")
```

### 19.1.4 损失函数选择指南

| 任务类型 | 推荐损失函数 | 输出层激活 |
|---|---|---|
| 回归 | `MSELoss` / `L1Loss` | 无（线性输出） |
| 二分类 | `BCEWithLogitsLoss` | 无（内含 Sigmoid） |
| 多分类 | `CrossEntropyLoss` | 无（内含 Softmax） |
| 多标签分类 | `BCEWithLogitsLoss` | 无（内含 Sigmoid） |
| 序列生成 | `CrossEntropyLoss` | 无（内含 Softmax） |

---

## 19.2 优化器

优化器（Optimizer）根据梯度更新模型参数，目标是最小化损失函数。

### 19.2.1 随机梯度下降：SGD

SGD 是最基础的优化器，支持动量（Momentum）和权重衰减（Weight Decay）。

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}$$

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 简单模型
model = nn.Linear(10, 1)

# 基础 SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 带动量的 SGD（推荐）：momentum 使更新方向平滑
optimizer_momentum = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,       # 动量系数，常用 0.9
    weight_decay=1e-4,  # L2 正则化，防止过拟合
    nesterov=True,      # Nesterov 动量，通常效果更好
)

# 为不同层设置不同学习率（常用于微调）
optimizer_diff_lr = optim.SGD([
    {'params': model.weight, 'lr': 0.001},  # 权重学习率
    {'params': model.bias,   'lr': 0.01},   # 偏置学习率更大
], lr=0.005)   # 默认学习率
```

### 19.2.2 自适应学习率优化器：Adam

Adam（Adaptive Moment Estimation）结合了动量和自适应学习率，是深度学习中最常用的优化器。

```python
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# Adam 优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,            # 默认学习率
    betas=(0.9, 0.999), # 一阶/二阶矩估计的衰减系数
    eps=1e-8,           # 数值稳定性项
    weight_decay=0,     # 不建议在 Adam 中使用（见 AdamW）
)

# 学习率调度：余弦退火
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# 训练步骤示意
for epoch in range(100):
    # ... 训练代码 ...
    optimizer.step()
    scheduler.step()   # 每个 epoch 更新学习率
    if epoch % 10 == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: lr = {current_lr:.6f}")
```

### 19.2.3 解耦权重衰减优化器：AdamW

AdamW 修正了 Adam 中权重衰减的实现方式，将 L2 正则化从梯度更新中解耦，是当前 Transformer 类模型的标准选择。

```python
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# AdamW：推荐用于 Transformer 和大型模型
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,  # 权重衰减在 AdamW 中正确解耦
)

# 实践技巧：不对偏置和 LayerNorm 应用权重衰减
def get_optimizer_groups(model, weight_decay=0.01):
    """将参数分组：有衰减 vs 无衰减"""
    decay_params     = []
    no_decay_params  = []
    no_decay_names   = {'bias', 'LayerNorm.weight', 'LayerNorm.bias'}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {'params': decay_params,    'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

param_groups = get_optimizer_groups(model)
optimizer_grouped = optim.AdamW(param_groups, lr=1e-4)
```

### 19.2.4 学习率调度器

学习率调度器在训练过程中动态调整学习率，通常能显著提升模型性能。

```python
import torch.optim as optim
import matplotlib.pyplot as plt

model     = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 常用调度器
# 1. StepLR：每 step_size 个 epoch 将 lr 乘以 gamma
step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 2. MultiStepLR：在指定 milestone epoch 降低学习率
milestones_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 90], gamma=0.1
)

# 3. CosineAnnealingLR：余弦退火，平滑降低学习率
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# 4. ReduceLROnPlateau：验证集损失不降时降低 lr（自适应）
plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# 5. OneCycleLR：超级收敛策略，先升后降
onecycle_scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, steps_per_epoch=100, epochs=30
)

# ReduceLROnPlateau 的使用方式不同（传入验证损失）
val_loss = 0.5
plateau_scheduler.step(val_loss)   # 传入指标，而非 epoch
```

---

## 19.3 训练循环编写

规范的训练循环是深度学习工程的核心，需要正确处理前向传播、损失计算、反向传播和参数更新四个步骤。

### 19.3.1 基础训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---- 准备数据 ----
torch.manual_seed(42)
X = torch.randn(1000, 20)
y = (X[:, 0] + X[:, 1] > 0).long()   # 二分类标签

dataset    = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ---- 定义模型 ----
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---- 训练循环 ----
def train_one_epoch(model, dataloader, criterion, optimizer, device='cpu'):
    model.train()          # 切换到训练模式（启用 Dropout、BatchNorm 等）
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 步骤 1：清零梯度（必须在每次迭代前执行）
        optimizer.zero_grad()

        # 步骤 2：前向传播
        outputs = model(inputs)

        # 步骤 3：计算损失
        loss = criterion(outputs, labels)

        # 步骤 4：反向传播（计算梯度）
        loss.backward()

        # 步骤 5：梯度裁剪（可选，防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 步骤 6：更新参数
        optimizer.step()

        # 统计指标
        total_loss    += loss.item() * inputs.size(0)
        preds          = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# 运行训练
for epoch in range(5):
    train_loss, train_acc = train_one_epoch(model, dataloader, criterion, optimizer)
    print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
```

### 19.3.2 梯度裁剪与梯度累积

```python
# ---- 梯度裁剪：防止梯度爆炸 ----
optimizer.zero_grad()
outputs = model(inputs)
loss    = criterion(outputs, labels)
loss.backward()

# 按范数裁剪（常用于 RNN/Transformer）
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,     # 梯度 L2 范数上限
)
print(f"Gradient norm: {grad_norm:.4f}")

optimizer.step()

# ---- 梯度累积：模拟大 batch（内存不足时使用） ----
accumulation_steps = 4   # 等效 batch_size 扩大 4 倍

optimizer.zero_grad()
for step, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss    = criterion(outputs, labels)

    # 缩放损失，保持梯度量级一致
    loss = loss / accumulation_steps
    loss.backward()

    # 每 accumulation_steps 步更新一次参数
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 19.4 验证与测试

验证集用于监控训练过程、调整超参数；测试集用于最终性能评估。两者都需要关闭梯度计算。

### 19.4.1 验证循环

```python
import torch

def evaluate(model, dataloader, criterion, device='cpu'):
    """在验证集或测试集上评估模型"""
    model.eval()       # 关键：切换到评估模式
                       # 禁用 Dropout，BatchNorm 使用全局统计量

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():   # 关键：禁用梯度计算，节省内存与计算
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            total_loss    += loss.item() * inputs.size(0)
            preds          = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# model.train() 与 model.eval() 的区别
print("训练模式效果示例：")
model_with_dropout = nn.Sequential(
    nn.Linear(10, 20),
    nn.Dropout(p=0.5),   # 训练时随机置零
    nn.Linear(20, 1),
)

x = torch.ones(3, 10)

model_with_dropout.train()
out_train1 = model_with_dropout(x)
out_train2 = model_with_dropout(x)
print(f"训练模式（两次结果不同）: {torch.allclose(out_train1, out_train2)}")

model_with_dropout.eval()
out_eval1 = model_with_dropout(x)
out_eval2 = model_with_dropout(x)
print(f"评估模式（两次结果相同）: {torch.allclose(out_eval1, out_eval2)}")
```

### 19.4.2 完整训练与验证流程

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(42)

# 数据准备
X = torch.randn(2000, 20)
y = (X[:, 0] * X[:, 1] + X[:, 2] > 0).long()
dataset = TensorDataset(X, y)

# 划分训练集 / 验证集 / 测试集 = 70% / 15% / 15%
n_total = len(dataset)
n_train = int(0.7 * n_total)
n_val   = int(0.15 * n_total)
n_test  = n_total - n_train - n_val

train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=128)
test_loader  = DataLoader(test_set,  batch_size=128)

# 模型与优化器
model     = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32,  2),
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5, verbose=False
)

# 早停机制
best_val_loss   = float('inf')
patience_count  = 0
early_stop_pat  = 10

# 主训练循环
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

for epoch in range(50):
    # 训练
    train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer)

    # 验证
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    # 学习率调度
    scheduler.step(val_loss)

    # 记录历史
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # 早停检测
    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        patience_count = 0
        # 保存最佳模型权重（见 19.5 节）
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_count += 1

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if patience_count >= early_stop_pat:
        print(f"Early stopping at epoch {epoch}")
        break

# 恢复最佳权重，在测试集上评估
model.load_state_dict(best_weights)
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"\n最终测试集结果 | Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
```

### 19.4.3 多分类评估指标

```python
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def get_predictions(model, dataloader, device='cpu'):
    """收集所有预测结果与真实标签"""
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs  = inputs.to(device)
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy(),
    )

preds, labels, probs = get_predictions(model, test_loader)

# 详细分类报告
print(classification_report(labels, preds, target_names=['Class 0', 'Class 1']))

# 混淆矩阵
cm = confusion_matrix(labels, preds)
print("混淆矩阵:")
print(cm)
```

---

## 19.5 模型保存与加载

PyTorch 提供多种模型持久化方式，根据需求选择合适的策略。

### 19.5.1 保存与加载状态字典（推荐方式）

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 10),
)

# ---- 保存状态字典 ----
# 只保存参数（权重和偏置），不包含模型结构
torch.save(model.state_dict(), 'model_weights.pth')

# ---- 加载状态字典 ----
# 必须先定义与原来相同的模型结构
model_loaded = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 10),
)
model_loaded.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model_loaded.eval()   # 加载后切换到评估模式

print("模型权重加载成功")

# 查看状态字典内容
state_dict = model.state_dict()
for name, param in state_dict.items():
    print(f"{name}: shape={param.shape}")
```

### 19.5.2 保存完整模型

```python
# ---- 保存完整模型（结构 + 权重）----
torch.save(model, 'model_full.pth')

# ---- 加载完整模型 ----
model_full = torch.load('model_full.pth', weights_only=False)
model_full.eval()
# 注意：依赖于保存时的类定义，迁移性较差，不推荐用于生产
```

### 19.5.3 训练检查点（Checkpoint）

检查点保存训练过程的完整状态，支持断点续训。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import os

def save_checkpoint(state: dict, filepath: str) -> None:
    """保存训练检查点"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(filepath: str, model: nn.Module,
                    optimizer: optim.Optimizer = None):
    """加载训练检查点，返回恢复的 epoch 和最佳损失"""
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch     = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"Checkpoint loaded: epoch={epoch}, best_val_loss={best_loss:.4f}")
    return epoch, best_loss

# ---- 使用示例 ----
model     = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

checkpoint_dir = 'checkpoints'
best_val_loss  = float('inf')
start_epoch    = 0

# 断点续训：若检查点存在则加载
checkpoint_path = os.path.join(checkpoint_dir, 'latest.pth')
if os.path.exists(checkpoint_path):
    start_epoch, best_val_loss = load_checkpoint(
        checkpoint_path, model, optimizer
    )

# 模拟训练循环
for epoch in range(start_epoch, 20):
    # ... 实际训练代码 ...
    simulated_val_loss = 1.0 / (epoch + 1)   # 模拟递减的验证损失

    # 每个 epoch 保存最新检查点
    save_checkpoint({
        'epoch':               epoch + 1,
        'model_state_dict':    model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss':       best_val_loss,
        'current_val_loss':    simulated_val_loss,
    }, checkpoint_path)

    # 保存最佳模型
    if simulated_val_loss < best_val_loss:
        best_val_loss = simulated_val_loss
        save_checkpoint({
            'epoch':            epoch + 1,
            'model_state_dict': model.state_dict(),
            'best_val_loss':    best_val_loss,
        }, os.path.join(checkpoint_dir, 'best.pth'))

    scheduler.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Val Loss: {simulated_val_loss:.4f} "
              f"| Best: {best_val_loss:.4f}")
```

### 19.5.4 跨设备加载（CPU / GPU 迁移）

```python
import torch

# 保存时使用 GPU，加载到 CPU
def load_to_cpu(filepath: str, model: nn.Module):
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint)
    return model

# 保存时使用 CPU，加载到 GPU
def load_to_gpu(filepath: str, model: nn.Module, device: str = 'cuda:0'):
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    return model

# 通用跨设备加载
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('model_weights.pth', map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
model = model.to(device)
print(f"Model loaded on: {device}")
```

---

## 本章小结

| 知识点 | 核心要点 |
|---|---|
| **损失函数** | 回归→MSELoss；多分类→CrossEntropyLoss（含Softmax）；二分类→BCEWithLogitsLoss（含Sigmoid） |
| **优化器** | SGD 适合图像分类（配合动量）；Adam 通用性好；AdamW 是 Transformer 标配 |
| **学习率调度** | ReduceLROnPlateau（自适应）/ CosineAnnealingLR（平滑）/ OneCycleLR（超级收敛） |
| **训练循环** | 必须顺序执行：`zero_grad()` → 前向 → `loss.backward()` → `optimizer.step()` |
| **验证模式** | `model.eval()` + `torch.no_grad()` 组合使用，验证集不更新参数 |
| **早停机制** | 监控验证损失，patience 轮无改善则停止，恢复最佳权重 |
| **模型保存** | 优先保存 `state_dict`；断点续训使用包含优化器状态的完整 checkpoint |
| **跨设备加载** | `torch.load(..., map_location=device)` 处理 CPU/GPU 迁移 |

---

## 深度学习应用：完整训练管道

本节展示一个端到端的分类模型训练代码，整合本章所有知识点，可直接作为项目模板使用。

```python
"""
完整训练管道示例
任务：手写数字分类（使用随机数据模拟 MNIST 结构）
包含：数据加载、模型定义、训练循环、验证、保存/加载
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, List, Tuple

# ===========================================================
# 1. 配置
# ===========================================================
CONFIG = {
    'seed':          42,
    'batch_size':    128,
    'epochs':        30,
    'lr':            1e-3,
    'weight_decay':  1e-4,
    'patience':      7,         # 早停耐心值
    'checkpoint_dir': 'checkpoints',
    'device':        'cuda' if torch.cuda.is_available() else 'cpu',
}

torch.manual_seed(CONFIG['seed'])
device = torch.device(CONFIG['device'])
print(f"使用设备: {device}")


# ===========================================================
# 2. 数据准备（模拟 MNIST：28×28 灰度图，10 类）
# ===========================================================
def make_dataset(n_samples: int = 5000) -> TensorDataset:
    X = torch.randn(n_samples, 1, 28, 28)   # (N, C, H, W)
    y = torch.randint(0, 10, (n_samples,))  # 0~9
    return TensorDataset(X, y)

full_dataset = make_dataset(n_samples=5000)
n_total = len(full_dataset)
n_train = int(0.7 * n_total)
n_val   = int(0.15 * n_total)
n_test  = n_total - n_train - n_val

train_set, val_set, test_set = random_split(
    full_dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(CONFIG['seed'])
)

train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

print(f"数据集大小 -> 训练: {n_train} | 验证: {n_val} | 测试: {n_test}")


# ===========================================================
# 3. 模型定义（小型 CNN）
# ===========================================================
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28×28→28×28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # 28×28→14×14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14×14→14×14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # 14×14→7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SmallCNN(num_classes=10).to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")


# ===========================================================
# 4. 损失函数与优化器
# ===========================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)   # 标签平滑
optimizer = optim.AdamW(
    model.parameters(),
    lr=CONFIG['lr'],
    weight_decay=CONFIG['weight_decay'],
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=CONFIG['epochs'],
    eta_min=1e-6,
)


# ===========================================================
# 5. 训练与验证函数
# ===========================================================
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    model.train()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss    += loss.item() * inputs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += inputs.size(0)

    return total_loss / total_samples, total_correct / total_samples


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            total_loss    += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += inputs.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ===========================================================
# 6. 主训练循环（含早停与检查点）
# ===========================================================
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

history: Dict[str, List[float]] = {
    'train_loss': [], 'train_acc': [],
    'val_loss':   [], 'val_acc':   [],
    'lr':         [],
}

best_val_loss     = float('inf')
patience_counter  = 0
best_model_path   = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')

print("\n开始训练...\n" + "="*60)
for epoch in range(1, CONFIG['epochs'] + 1):
    # 训练
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    # 验证
    val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
    # 更新学习率
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    # 记录历史
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)

    # 日志输出
    print(f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"LR: {current_lr:.2e}")

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            'epoch':            epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss':         val_loss,
            'val_acc':          val_acc,
            'config':           CONFIG,
        }, best_model_path)
        print(f"  -> 最佳模型已保存 (val_loss={best_val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG['patience']:
            print(f"\n早停触发（{CONFIG['patience']} 轮未改善），停止训练")
            break


# ===========================================================
# 7. 测试集最终评估
# ===========================================================
print("\n" + "="*60)
print("加载最佳模型进行测试集评估...")
checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
print(f"测试集结果 | Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
print(f"最佳验证结果来自 Epoch {checkpoint['epoch']}: "
      f"Val Loss={checkpoint['val_loss']:.4f}, "
      f"Val Acc={checkpoint['val_acc']:.4f}")
```

---

## 练习题

### 基础题

**练习 1**：编写一个使用 `MSELoss` 的简单线性回归训练循环。数据为 $y = 2x + 1 + \epsilon$，其中 $\epsilon$ 为均值 0、标准差 0.1 的高斯噪声。训练 100 个 epoch 后打印最终学习到的权重与偏置。

**练习 2**：对比 `BCELoss` 和 `BCEWithLogitsLoss` 的数值稳定性。生成一批极端 logits（如 100.0 和 -100.0），分别计算两种损失并观察结果差异。解释为什么 `BCEWithLogitsLoss` 更稳定。

### 进阶题

**练习 3**：实现一个带**学习率预热**（warmup）的训练循环。前 5 个 epoch 学习率从 0 线性增长到目标学习率，之后使用余弦退火衰减到 0。使用 `LambdaLR` 调度器实现，并绘制学习率曲线。

**练习 4**：在练习 3 的基础上，添加以下功能：
- 梯度累积（accumulation_steps=4）
- 梯度范数监控（记录每步的梯度范数并打印均值）
- 验证集上的 Top-3 准确率计算

### 挑战题

**练习 5**：实现一个通用的 `Trainer` 类，封装本章所有功能：
- 支持任意 `model`、`criterion`、`optimizer`、`scheduler`
- 支持早停（`patience` 参数）
- 支持检查点保存与断点续训（`resume_from` 参数）
- 提供 `fit(train_loader, val_loader, epochs)` 方法
- 返回训练历史字典，包含各 epoch 的 loss、acc、lr

---

## 练习答案

### 答案 1：线性回归训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

# 生成数据：y = 2x + 1 + noise
X = torch.linspace(-5, 5, 500).unsqueeze(1)
y = 2 * X + 1 + torch.randn_like(X) * 0.1

# 模型：单层线性
model     = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss    = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.6f}")

w = model.weight.item()
b = model.bias.item()
print(f"\n学习到的参数: weight={w:.4f} (真值 2.0), bias={b:.4f} (真值 1.0)")
```

### 答案 2：数值稳定性对比

```python
import torch
import torch.nn as nn

# 极端 logits（梯度爆炸场景）
logits  = torch.tensor([100.0, -100.0, 50.0, -50.0])
targets = torch.tensor([1.0,    0.0,   1.0,   0.0])

# BCELoss：需先手动 Sigmoid，极端值会导致 nan
probs = torch.sigmoid(logits)
print(f"Sigmoid 极端值结果: {probs}")   # [1., 0., 1., 0.] — 精度丢失

try:
    loss_bce = nn.BCELoss()(probs, targets)
    print(f"BCELoss: {loss_bce.item()}")
except Exception as e:
    print(f"BCELoss 错误: {e}")

# BCEWithLogitsLoss：内部使用 log-sum-exp 技巧，数值稳定
loss_logits = nn.BCEWithLogitsLoss()(logits, targets)
print(f"BCEWithLogitsLoss: {loss_logits.item():.6f}")

# 原理：BCEWithLogits 使用等价但稳定的公式
# loss = max(x, 0) - x*y + log(1 + exp(-|x|))
# 避免 exp(100) 溢出
```

### 答案 3：学习率预热 + 余弦退火

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math

model      = nn.Linear(10, 2)
optimizer  = optim.Adam(model.parameters(), lr=1e-3)

warmup_epochs = 5
total_epochs  = 50

def lr_lambda(epoch: int) -> float:
    """线性预热 + 余弦退火"""
    if epoch < warmup_epochs:
        # 线性预热：0 → 1
        return (epoch + 1) / warmup_epochs
    else:
        # 余弦退火：1 → 0
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

lr_history = []
for epoch in range(total_epochs):
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    scheduler.step()

print("学习率预热阶段 (前5 epochs):")
for i, lr in enumerate(lr_history[:5]):
    print(f"  Epoch {i+1}: lr = {lr:.6f}")

print("\n余弦退火阶段 (每10 epochs):")
for i in range(5, 50, 10):
    print(f"  Epoch {i+1}: lr = {lr_history[i]:.6f}")
```

### 答案 4：梯度累积 + 梯度监控 + Top-K 准确率

```python
import torch
import torch.nn as nn

def top_k_accuracy(outputs: torch.Tensor, labels: torch.Tensor, k: int = 3) -> float:
    """计算 Top-K 准确率"""
    _, top_k_preds = outputs.topk(k, dim=1)       # (batch, k)
    correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))
    return correct.any(dim=1).float().mean().item()

def train_with_grad_accum(model, loader, criterion, optimizer,
                          device, accumulation_steps=4):
    model.train()
    grad_norms   = []
    total_loss   = 0.0
    total_top3   = 0.0
    n_batches    = 0

    optimizer.zero_grad()

    for step, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss    = criterion(outputs, labels) / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            # 记录梯度范数
            grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            grad_norms.append(grad_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        total_top3 += top_k_accuracy(outputs.detach(), labels, k=3)
        n_batches  += 1

    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    print(f"平均梯度范数: {avg_grad_norm:.4f}")
    print(f"Top-3 准确率: {total_top3 / n_batches:.4f}")

# 演示（使用随机数据）
demo_model  = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
demo_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.randn(256, 20), torch.randint(0, 10, (256,))
    ), batch_size=32
)
train_with_grad_accum(
    demo_model, demo_loader, nn.CrossEntropyLoss(),
    optim.Adam(demo_model.parameters()), torch.device('cpu')
)
```

### 答案 5：通用 Trainer 类

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List

class Trainer:
    """通用训练器，封装完整训练流程"""

    def __init__(
        self,
        model:          nn.Module,
        criterion:      nn.Module,
        optimizer:      optim.Optimizer,
        scheduler:      Optional[object] = None,
        device:         str = 'cpu',
        patience:       int = 10,
        checkpoint_dir: str = 'checkpoints',
        resume_from:    Optional[str] = None,
    ):
        self.model          = model.to(device)
        self.criterion      = criterion
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.device         = torch.device(device)
        self.patience       = patience
        self.checkpoint_dir = checkpoint_dir
        self.start_epoch    = 0
        self.best_val_loss  = float('inf')

        os.makedirs(checkpoint_dir, exist_ok=True)

        # 断点续训
        if resume_from and os.path.exists(resume_from):
            self._load_checkpoint(resume_from)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.start_epoch   = ckpt.get('epoch', 0)
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"已从 {path} 恢复训练（epoch={self.start_epoch}）")

    def _save_checkpoint(self, epoch: int, val_loss: float,
                         is_best: bool = False) -> None:
        state = {
            'epoch':                epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss':        self.best_val_loss,
            'current_val_loss':     val_loss,
        }
        torch.save(state, os.path.join(self.checkpoint_dir, 'latest.pth'))
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, 'best.pth'))

    def _run_epoch(self, loader: DataLoader, training: bool) -> tuple:
        self.model.train() if training else self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if training:
                    self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss    = self.criterion(outputs, labels)
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()

                total_loss    += loss.item() * inputs.size(0)
                total_correct += (outputs.argmax(1) == labels).sum().item()
                total_samples += inputs.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int,
    ) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            'train_loss': [], 'train_acc': [],
            'val_loss':   [], 'val_acc':   [], 'lr': [],
        }
        patience_counter = 0

        for epoch in range(self.start_epoch + 1, self.start_epoch + epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            val_loss,   val_acc   = self._run_epoch(val_loader,   training=False)

            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler,
                              optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter   = 0
            else:
                patience_counter  += 1

            self._save_checkpoint(epoch, val_loss, is_best)

            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e}"
                  + (" *" if is_best else ""))

            if patience_counter >= self.patience:
                print(f"早停（patience={self.patience}），最佳 val_loss={self.best_val_loss:.4f}")
                break

        return history


# ---- 使用示例 ----
if __name__ == '__main__':
    import torch.utils.data as data

    torch.manual_seed(42)
    X_data = torch.randn(2000, 20)
    y_data = torch.randint(0, 5, (2000,))
    dataset = data.TensorDataset(X_data, y_data)

    n_train = 1600
    train_ds, val_ds = data.random_split(dataset, [n_train, 400])
    t_loader = data.DataLoader(train_ds, batch_size=64, shuffle=True)
    v_loader = data.DataLoader(val_ds,   batch_size=64)

    net = nn.Sequential(
        nn.Linear(20, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32,  5),
    )
    opt  = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    crit = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=net, criterion=crit, optimizer=opt,
        scheduler=sch, patience=5, checkpoint_dir='./trainer_ckpts'
    )
    hist = trainer.fit(t_loader, v_loader, epochs=20)
    print(f"\n训练完成，最终 Val Acc: {hist['val_acc'][-1]:.4f}")
```
