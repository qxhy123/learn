# 第21章：图像分类实战

## 学习目标

完成本章学习后，你将能够：

1. 理解卷积神经网络（CNN）的核心组件，包括卷积层、池化层和感受野的概念
2. 掌握 LeNet、VGG、ResNet 等经典 CNN 架构的设计思路与特点
3. 使用 PyTorch 完整处理 CIFAR-10 数据集，包括数据加载、增强和标准化
4. 运用学习率调度、批归一化、Dropout 等技术对模型进行训练与调优
5. 理解迁移学习原理，能够对预训练模型进行微调以适配新任务

---

## 21.1 卷积神经网络回顾

### 21.1.1 为什么需要卷积神经网络

全连接网络处理图像时存在两个根本问题：**参数量爆炸**和**空间信息丢失**。

一张 32×32 的彩色图像有 32×32×3 = 3072 个像素。若第一个全连接层有 512 个神经元，仅这一层就需要 3072×512 = 1,572,864 个参数。对于 224×224 的图像，参数量将达到数亿级别。

更严重的是，全连接层将图像"拉平"为一维向量后，完全丢失了像素之间的空间位置关系——而"猫的耳朵在头顶"这类空间先验知识对图像理解至关重要。

卷积神经网络通过以下三个核心思想解决上述问题：
- **局部感知**：每个神经元只连接输入的局部区域
- **权重共享**：同一卷积核在整张图像上滑动，大幅减少参数
- **层次特征**：浅层学习边缘纹理，深层学习语义特征

### 21.1.2 卷积层详解

卷积操作的数学本质是对输入特征图与卷积核做互相关运算：

```
输出[i, j] = Σ_m Σ_n 输入[i+m, j+n] × 核[m, n] + 偏置
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建一个简单的卷积层示例
# in_channels=3（RGB），out_channels=16（16个卷积核），kernel_size=3
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                       stride=1, padding=1)

# 输入：批次大小=2，3通道，32×32图像
x = torch.randn(2, 3, 32, 32)
output = conv_layer(x)
print(f"输入形状：{x.shape}")       # 输出: torch.Size([2, 3, 32, 32])
print(f"输出形状：{output.shape}")   # 输出: torch.Size([2, 16, 32, 32])

# 计算卷积层参数量
params = sum(p.numel() for p in conv_layer.parameters())
print(f"卷积层参数量：{params}")     # 输出: 448  (3×3×3×16 + 16)
```

**输出尺寸计算公式：**

```
输出尺寸 = floor((输入尺寸 + 2×padding - kernel_size) / stride) + 1
```

```python
def calc_conv_output_size(input_size, kernel_size, stride=1, padding=0):
    """计算卷积输出尺寸"""
    return (input_size + 2 * padding - kernel_size) // stride + 1

# 示例：32×32 输入，3×3 卷积核，padding=1
h_out = calc_conv_output_size(32, kernel_size=3, stride=1, padding=1)
print(f"输出高度：{h_out}")   # 输出: 32 （same padding 保持尺寸不变）

# 步长=2 时的下采样效果
h_out_stride2 = calc_conv_output_size(32, kernel_size=3, stride=2, padding=1)
print(f"步长=2时输出高度：{h_out_stride2}")  # 输出: 16
```

### 21.1.3 池化层

池化层的作用是**降低特征图分辨率**，同时引入一定的平移不变性，减少计算量。

```python
# 最大池化：取区域内最大值，保留显著特征
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 平均池化：取区域内均值，信息更平滑
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# 全局平均池化：将整个特征图压缩为单个值（ResNet 等现代架构常用）
global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

x = torch.randn(2, 16, 32, 32)
print(f"最大池化输出：{max_pool(x).shape}")        # torch.Size([2, 16, 16, 16])
print(f"全局平均池化输出：{global_avg_pool(x).shape}")  # torch.Size([2, 16, 1, 1])
```

### 21.1.4 感受野

**感受野**（Receptive Field）是指输出特征图中某个单元所对应的输入图像区域大小。感受野越大，该单元能"看到"的上下文信息越多。

```python
# 两层 3×3 卷积的感受野等于一层 5×5 卷积
# 但参数量更少：2×(3×3) = 18 < 5×5 = 25
# 且中间可插入非线性激活，表达能力更强

# 感受野随深度增长的规律（stride=1, no padding）
# 第1层3×3卷积：感受野 = 3
# 第2层3×3卷积：感受野 = 3 + (3-1) = 5
# 第3层3×3卷积：感受野 = 5 + (3-1) = 7
# 第n层：感受野 = 1 + n × (kernel_size - 1)

def calc_receptive_field(num_layers, kernel_size=3, stride=1):
    """计算堆叠卷积层的感受野（stride=1）"""
    rf = 1
    for _ in range(num_layers):
        rf = rf + (kernel_size - 1) * stride
    return rf

for n in [1, 2, 3, 5, 10]:
    print(f"{n}层3×3卷积的感受野：{calc_receptive_field(n)}")
# 输出:
# 1层3×3卷积的感受野：3
# 2层3×3卷积的感受野：5
# 3层3×3卷积的感受野：7
# 5层3×3卷积的感受野：11
# 10层3×3卷积的感受野：21
```

---

## 21.2 经典CNN架构

### 21.2.1 LeNet-5：开山之作

LeNet-5 由 Yann LeCun 于 1998 年提出，专为手写数字识别设计，是现代 CNN 的雏形。

```
输入(32×32) → C1卷积(6个5×5) → S2池化 → C3卷积(16个5×5) → S4池化 → 全连接 → 输出
```

```python
class LeNet5(nn.Module):
    """经典 LeNet-5 架构（适配32×32输入）"""

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # 特征提取部分
        self.features = nn.Sequential(
            # C1：6个5×5卷积核
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            # S2：2×2平均池化
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C3：16个5×5卷积核
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            # S4：2×2平均池化
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        # 分类部分
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 测试 LeNet-5
model = LeNet5(num_classes=10)
x = torch.randn(4, 1, 32, 32)
output = model(x)
print(f"LeNet-5 输出形状：{output.shape}")  # torch.Size([4, 10])
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量：{total_params:,}")        # 总参数量：61,706
```

### 21.2.2 VGG：深度的力量

VGG 网络（2014 年）的核心思想是：**用多个 3×3 小卷积核替代大卷积核，加深网络深度**。VGG-16 拥有 16 个带权重的层，在 ImageNet 上取得了当时最好的结果。

```python
def vgg_block(num_convs, in_channels, out_channels):
    """构建 VGG 块：num_convs 个连续卷积 + 最大池化"""
    layers = []
    for _ in range(num_convs):
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # 原版无BN，此处加入提升训练稳定性
            nn.ReLU(inplace=True),
        ]
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """简化的 VGG 网络（适配 CIFAR-10 的 32×32 输入）"""

    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        # VGG-11 配置：每个元组为 (卷积层数, 输出通道数)
        arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
        self.features = self._make_layers(arch)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _make_layers(self, arch):
        layers = []
        in_channels = 3
        for num_convs, out_channels in arch:
            layers.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


vgg = VGG(num_classes=10)
x = torch.randn(2, 3, 32, 32)
print(f"VGG 输出形状：{vgg(x).shape}")   # torch.Size([2, 10])
```

### 21.2.3 ResNet：残差连接突破深度瓶颈

随着网络加深，出现了**梯度消失**和**退化问题**（更深的网络反而精度更低）。ResNet（2015 年）通过**残差连接**（Skip Connection）解决了这一问题。

**核心思想：** 让网络学习残差映射 F(x) = H(x) - x，而不是直接学习目标映射 H(x)。

```python
class ResidualBlock(nn.Module):
    """ResNet 基本残差块（用于 ResNet-18/34）"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 主路径：两个3×3卷积
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # 捷径连接：当维度不匹配时用1×1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut(x)   # 残差连接：相加而非拼接
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 实现（适配 CIFAR-10）"""

    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # CIFAR-10 输入小，第一层用3×3卷积而非7×7
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # 4个残差层，每层包含2个残差块
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


resnet = ResNet18(num_classes=10)
x = torch.randn(2, 3, 32, 32)
print(f"ResNet-18 输出形状：{resnet(x).shape}")   # torch.Size([2, 10])
total_params = sum(p.numel() for p in resnet.parameters())
print(f"总参数量：{total_params:,}")               # 总参数量：11,173,962
```

---

## 21.3 CIFAR-10 数据集处理

### 21.3.1 数据集简介

CIFAR-10 包含 **60,000 张 32×32 彩色图像**，分为 10 个类别，每类 6000 张：
飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。
训练集 50,000 张，测试集 10,000 张。

### 21.3.2 数据加载与增强

```python
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR-10 数据集均值和标准差（从训练集统计得到）
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# 训练集数据增强：随机裁剪 + 随机水平翻转 + 颜色抖动
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          # 四边填充4像素后随机裁剪回32×32
    transforms.RandomHorizontalFlip(p=0.5),        # 随机水平翻转（概率50%）
    transforms.ColorJitter(brightness=0.2,         # 亮度扰动
                           contrast=0.2,           # 对比度扰动
                           saturation=0.2),        # 饱和度扰动
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# 测试集：只做标准化，不做随机增强
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

def get_cifar10_loaders(data_root='./data', batch_size=128, num_workers=4):
    """返回 CIFAR-10 的训练和测试数据加载器"""
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    print(f"训练集大小：{len(train_dataset)}")   # 训练集大小：50000
    print(f"测试集大小：{len(test_dataset)}")    # 测试集大小：10000
    print(f"训练批次数：{len(train_loader)}")    # 训练批次数：391
    return train_loader, test_loader

CLASSES = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')
```

### 21.3.3 数据可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def imshow_batch(loader, num_images=16):
    """可视化一个批次中的图像（反归一化后显示）"""
    images, labels = next(iter(loader))
    # 反归一化
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    images = images[:num_images] * std + mean
    images = images.clamp(0, 1)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(CLASSES[labels[i]], fontsize=8)
        ax.axis('off')
    plt.suptitle('CIFAR-10 样本（经过数据增强）', fontsize=12)
    plt.tight_layout()
    plt.show()

# imshow_batch(train_loader)  # 取消注释以运行
```

### 21.3.4 数据集统计分析

```python
def analyze_dataset(dataset):
    """分析数据集各类别分布"""
    targets = np.array(dataset.targets)
    print(f"{'类别':<8} {'数量':>6} {'占比':>6}")
    print("-" * 22)
    for i, cls in enumerate(CLASSES):
        count = (targets == i).sum()
        ratio = count / len(targets) * 100
        print(f"{cls:<8} {count:>6,} {ratio:>5.1f}%")

# train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True)
# analyze_dataset(train_dataset)
# 输出示例：
# 类别       数量   占比
# ----------------------
# 飞机      5000  10.0%
# 汽车      5000  10.0%
# ...（均衡数据集，每类各5000张）
```

---

## 21.4 模型训练与调优

### 21.4.1 训练框架搭建

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch，返回平均损失和准确率"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  [批次 {batch_idx+1}/{len(loader)}] "
                  f"损失: {total_loss/(batch_idx+1):.3f} | "
                  f"准确率: {100.*correct/total:.1f}%")

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """在验证/测试集上评估模型"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy
```

### 21.4.2 学习率调度策略

```python
# 策略一：余弦退火（Cosine Annealing）—— 平滑降低学习率
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

# 策略二：多步衰减（MultiStep）—— 在指定 epoch 乘以 gamma
from torch.optim.lr_scheduler import MultiStepLR
scheduler_multistep = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

# 策略三：带热身的余弦退火（WarmupCosine）—— 前几个epoch逐渐升温
class WarmupCosineScheduler:
    """前 warmup_epochs 线性升温，之后余弦退火"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
```

### 21.4.3 正则化技术

```python
# 1. 批归一化（Batch Normalization）：加速训练，允许更大学习率
class ConvBNReLU(nn.Module):
    """卷积 + 批归一化 + ReLU 的标准组合"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


# 2. Dropout：训练时随机丢弃神经元，防止过拟合
dropout = nn.Dropout(p=0.5)          # 全连接层用
dropout2d = nn.Dropout2d(p=0.1)      # 卷积特征图用（按通道丢弃）


# 3. 标签平滑（Label Smoothing）：软化目标标签，提升泛化
class LabelSmoothingCrossEntropy(nn.Module):
    """带标签平滑的交叉熵损失"""
    def __init__(self, smoothing=0.1, num_classes=10):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        # 构造平滑标签：目标类概率为 1-smoothing，其余类各分得 smoothing/C
        smooth_val = self.smoothing / self.num_classes
        one_hot = torch.zeros_like(predictions).scatter_(
            1, targets.unsqueeze(1), 1.0
        )
        smooth_targets = one_hot * (1 - self.smoothing) + smooth_val
        log_probs = F.log_softmax(predictions, dim=1)
        loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        return loss


# 4. 权重衰减（L2 正则化）通过优化器实现
optimizer = optim.SGD(
    resnet.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,   # L2 正则化系数
    nesterov=True        # Nesterov 动量，收敛更快
)
```

### 21.4.4 完整训练循环

```python
def train_model(model, train_loader, test_loader, num_epochs=200,
                lr=0.1, device='cuda'):
    """完整训练循环，含早停和模型保存"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer,
                                                criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"训练 损失: {train_loss:.4f} 准确率: {train_acc:.2f}% | "
              f"测试 损失: {test_loss:.4f} 准确率: {test_acc:.2f}%")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ 保存最佳模型（测试准确率: {best_acc:.2f}%）")

    print(f"\n训练完成！最佳测试准确率: {best_acc:.2f}%")
    return history


def plot_training_history(history):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='训练损失')
    ax1.plot(epochs, history['test_loss'], label='测试损失')
    ax1.set_title('损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], label='训练准确率')
    ax2.plot(epochs, history['test_acc'], label='测试准确率')
    ax2.set_title('准确率曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=120)
    plt.show()
```

---

## 21.5 迁移学习

### 21.5.1 迁移学习的动机

从头训练一个大型 CNN 需要：
- 大量标注数据（通常需要数十万张）
- 大量计算资源（数天 GPU 训练时间）

**迁移学习**利用在大型数据集（如 ImageNet）上预训练好的模型权重，将其迁移到目标任务上，极大降低了数据和计算需求。

```
ImageNet（1.2M 图像，1000类）
       ↓  预训练
  预训练模型（通用特征）
       ↓  迁移
  目标数据集（数百至数千张图像）
       ↓  微调
  目标任务模型
```

### 21.5.2 加载预训练模型

```python
import torchvision.models as models

# 加载预训练 ResNet-50（在 ImageNet 上训练）
pretrained_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
print(pretrained_resnet)

# 查看各层参数量
for name, module in pretrained_resnet.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"{name:<20} {params:>12,} 参数")
# 输出示例:
# conv1                    9,408 参数
# bn1                        128 参数
# layer1               215,808 参数
# layer2               1,219,584 参数
# layer3               7,077,888 参数
# layer4              14,964,736 参数
# fc                       2,048,000 参数（fc层）
```

### 21.5.3 特征提取（冻结主干网络）

策略一：冻结全部卷积层，只训练分类头。适合数据集极小时使用。

```python
def create_feature_extractor(pretrained_model, num_classes, freeze_features=True):
    """
    特征提取模式：冻结骨干网络，替换并训练分类头
    """
    model = pretrained_model

    if freeze_features:
        # 冻结除最后分类层以外的所有参数
        for name, param in model.named_parameters():
            if 'fc' not in name:  # ResNet 的分类层名为 'fc'
                param.requires_grad = False

    # 替换分类层（保持 in_features 不变）
    in_features = model.fc.in_features       # ResNet-50: 2048
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes),
    )

    # 统计可训练参数量
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数：{trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model

# 只有约 0.5% 的参数需要训练
model_fe = create_feature_extractor(
    models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
    num_classes=10
)
```

### 21.5.4 微调（Fine-tuning）

策略二：对整个网络进行微调，但主干网络使用较小学习率。适合数据集有一定规模时使用。

```python
def create_finetune_model(pretrained_model, num_classes,
                          backbone_lr_ratio=0.1):
    """
    微调模式：主干网络使用小学习率，分类头使用大学习率

    backbone_lr_ratio: 主干网络学习率 = 基础学习率 × backbone_lr_ratio
    """
    model = pretrained_model
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def get_finetune_optimizer(model, base_lr=0.01, backbone_lr_ratio=0.1):
    """
    为微调设置差分学习率：
    - 分类头：base_lr
    - 主干网络深层：base_lr × backbone_lr_ratio
    - 主干网络浅层：base_lr × backbone_lr_ratio × 0.1
    """
    # 将参数分组
    fc_params = list(model.fc.parameters())
    fc_param_ids = set(id(p) for p in fc_params)

    backbone_params = [p for p in model.parameters()
                       if id(p) not in fc_param_ids]

    optimizer = optim.Adam([
        {'params': fc_params,       'lr': base_lr},
        {'params': backbone_params, 'lr': base_lr * backbone_lr_ratio},
    ], weight_decay=1e-4)

    return optimizer


# 典型微调配置
finetune_model = create_finetune_model(
    models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
    num_classes=10
)
optimizer = get_finetune_optimizer(finetune_model, base_lr=0.001)
print(f"分类头学习率：{optimizer.param_groups[0]['lr']}")   # 0.001
print(f"主干网络学习率：{optimizer.param_groups[1]['lr']}") # 0.0001
```

### 21.5.5 迁移学习策略选择指南

```python
"""
根据数据集大小和与预训练数据的相似度选择迁移策略：

                    │  与源域相似度高  │  与源域相似度低
────────────────────┼──────────────────┼──────────────────
  目标数据集小      │ 只训练分类头      │  微调顶部几层
  目标数据集大      │ 微调全部网络      │  从头训练或大范围微调

决策规则：
1. 数据 < 1000 张且与 ImageNet 相似    → 特征提取（冻结主干）
2. 数据 1000-10000 张                  → 微调顶部 1-2 个残差组
3. 数据 > 10000 张且与 ImageNet 相似   → 微调全部层（差分学习率）
4. 数据 > 10000 张且差异大             → 从头训练
"""
```

---

## 本章小结

| 概念 | 核心要点 | 典型参数/配置 |
|------|----------|---------------|
| 卷积层 | 局部感知 + 权重共享，提取空间特征 | 3×3 卷积核，padding=1 保持尺寸 |
| 池化层 | 降采样，增强平移不变性 | MaxPool 2×2，stride=2 |
| 感受野 | 随层数增大，n 层 3×3 → 感受野 1+2n | 用空洞卷积可快速扩大感受野 |
| LeNet-5 | 最早的实用 CNN，用于手写识别 | ~61K 参数 |
| VGG | 多个 3×3 代替大卷积核，加深网络 | ~138M 参数（VGG-16） |
| ResNet | 残差连接解决梯度消失和退化问题 | ResNet-18 约 11M 参数 |
| 数据增强 | 随机裁剪、翻转、颜色扰动 | 有效防止过拟合 |
| 批归一化 | 加速训练，允许更大学习率 | 每个卷积层后加 BN |
| 学习率调度 | 余弦退火是图像分类常用选择 | T_max=总epoch数 |
| 迁移学习 | 从预训练模型出发，快速适配新任务 | 差分学习率：主干×0.1 |

---

## 深度学习应用：CIFAR-10 分类完整项目

本节将以上知识整合为一个完整的图像分类项目，使用带残差连接的自定义网络在 CIFAR-10 上达到 93%+ 的测试准确率。

```python
"""
CIFAR-10 图像分类完整项目
目标：在 CIFAR-10 测试集上达到 93%+ 准确率
架构：ResNet-18（适配CIFAR-10的版本）
训练时长：约 200 epochs（GPU约30分钟）
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt

# ── 1. 超参数配置 ─────────────────────────────────────────────
CONFIG = {
    'batch_size':    128,
    'num_epochs':    200,
    'learning_rate': 0.1,
    'weight_decay':  5e-4,
    'num_workers':   4,
    'data_root':     './data',
    'save_path':     './checkpoints',
    'seed':          42,
}

torch.manual_seed(CONFIG['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")
os.makedirs(CONFIG['save_path'], exist_ok=True)


# ── 2. 数据管道 ───────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_set = torchvision.datasets.CIFAR10(
    root=CONFIG['data_root'], train=True,
    download=True, transform=train_transform
)
test_set = torchvision.datasets.CIFAR10(
    root=CONFIG['data_root'], train=False,
    download=True, transform=test_transform
)
train_loader = DataLoader(
    train_set, batch_size=CONFIG['batch_size'],
    shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True
)
test_loader = DataLoader(
    test_set, batch_size=CONFIG['batch_size'],
    shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True
)

CLASSES = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')
print(f"训练集：{len(train_set)} 张 | 测试集：{len(test_set)} 张")


# ── 3. 模型定义：CIFAR-10 专用 ResNet ──────────────────────────
class ResBlock(nn.Module):
    """带可选捷径的残差块"""

    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CIFARResNet(nn.Module):
    """
    CIFAR-10 专用 ResNet
    遵循 He et al. (2016) 原论文的 CIFAR 版本设计：
    3 个阶段，每阶段 n 个残差块，总深度 6n+2
    """

    def __init__(self, n=3, num_classes=10, dropout=0.1):
        """
        n=3  → ResNet-20  (0.27M 参数)
        n=5  → ResNet-32  (0.46M 参数)
        n=9  → ResNet-56  (0.85M 参数)
        n=18 → ResNet-110 (1.73M 参数)
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(16,  16,  n, stride=1, dropout=dropout)
        self.stage2 = self._make_stage(16,  32,  n, stride=2, dropout=dropout)
        self.stage3 = self._make_stage(32,  64,  n, stride=2, dropout=dropout)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(64, num_classes)

        # 权重初始化
        self._init_weights()

    def _make_stage(self, in_ch, out_ch, n, stride, dropout):
        blocks = [ResBlock(in_ch, out_ch, stride, dropout)]
        for _ in range(1, n):
            blocks.append(ResBlock(out_ch, out_ch, 1, dropout))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = CIFARResNet(n=9, num_classes=10, dropout=0.1)  # ResNet-56
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量：{total_params:,}")   # 约 851,754


# ── 4. 训练设置 ───────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    momentum=0.9,
    weight_decay=CONFIG['weight_decay'],
    nesterov=True,
)
scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6)


# ── 5. 训练循环 ───────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, training=True):
    model.train() if training else model.eval()
    total_loss = correct = total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if training:
                loss.backward()
                # 梯度裁剪，防止梯度爆炸
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item()
            _, preds = outputs.max(1)
            total   += targets.size(0)
            correct += preds.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total


best_acc = 0.0
history  = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
start_time = time.time()

print("\n开始训练...")
print(f"{'Epoch':<8} {'LR':<10} {'训练损失':<12} {'训练准确率':<14} {'测试损失':<12} {'测试准确率'}")
print("-" * 70)

for epoch in range(1, CONFIG['num_epochs'] + 1):
    lr = optimizer.param_groups[0]['lr']
    tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, True)
    te_loss, te_acc = run_epoch(model, test_loader,  optimizer, criterion, False)
    scheduler.step()

    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['test_loss'].append(te_loss)
    history['test_acc'].append(te_acc)

    if te_acc > best_acc:
        best_acc = te_acc
        torch.save({
            'epoch':      epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_acc':   best_acc,
        }, os.path.join(CONFIG['save_path'], 'best.pth'))

    if epoch % 10 == 0 or epoch == 1:
        elapsed = (time.time() - start_time) / 60
        print(f"{epoch:<8} {lr:<10.6f} {tr_loss:<12.4f} {tr_acc:<14.2f} "
              f"{te_loss:<12.4f} {te_acc:.2f}%  [{elapsed:.1f}min]")

print(f"\n训练完成！最佳测试准确率：{best_acc:.2f}%")
# 预期输出（200 epochs）：
# Epoch 200 | 最佳测试准确率：~93.5%


# ── 6. 模型评估：混淆矩阵与分类报告 ─────────────────────────────
def evaluate_per_class(model, loader, device):
    """计算每个类别的精确率和召回率"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\n{'类别':<8} {'精确率':>8} {'召回率':>8} {'F1分数':>8} {'样本数':>8}")
    print("-" * 44)
    for i, cls in enumerate(CLASSES):
        mask    = all_labels == i
        tp      = ((all_preds == i) & mask).sum()
        fp      = ((all_preds == i) & ~mask).sum()
        fn      = ((all_preds != i) & mask).sum()
        prec    = tp / (tp + fp + 1e-8)
        recall  = tp / (tp + fn + 1e-8)
        f1      = 2 * prec * recall / (prec + recall + 1e-8)
        print(f"{cls:<8} {prec:>8.3f} {recall:>8.3f} {f1:>8.3f} {mask.sum():>8}")

    total_acc = (all_preds == all_labels).mean() * 100
    print(f"\n整体准确率：{total_acc:.2f}%")
    return all_preds, all_labels


# 加载最佳模型进行评估
checkpoint = torch.load(os.path.join(CONFIG['save_path'], 'best.pth'),
                        map_location=device)
model.load_state_dict(checkpoint['model_state'])
print(f"\n加载第 {checkpoint['epoch']} epoch 的最佳模型"
      f"（验证准确率：{checkpoint['best_acc']:.2f}%）")

preds, labels = evaluate_per_class(model, test_loader, device)
# 预期输出示例：
# 类别     精确率   召回率   F1分数   样本数
# --------------------------------------------
# 飞机      0.954    0.937    0.945     1000
# 汽车      0.972    0.969    0.970     1000
# 鸟        0.906    0.896    0.901     1000
# 猫        0.843    0.872    0.857     1000  ← 最难分类
# ...


# ── 7. 可视化：绘制训练曲线 ────────────────────────────────────
def plot_results(history, best_acc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    # 损失曲线
    axes[0].plot(epochs, history['train_loss'], label='训练损失', alpha=0.8)
    axes[0].plot(epochs, history['test_loss'],  label='测试损失',  alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练/测试损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(epochs, history['train_acc'], label='训练准确率', alpha=0.8)
    axes[1].plot(epochs, history['test_acc'],  label='测试准确率',  alpha=0.8)
    axes[1].axhline(y=best_acc, color='red', linestyle='--',
                    label=f'最佳: {best_acc:.2f}%')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('训练/测试准确率曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('CIFAR-10 ResNet-56 训练结果', fontsize=14)
    plt.tight_layout()
    plt.savefig('cifar10_results.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("图表已保存至 cifar10_results.png")


plot_results(history, best_acc)


# ── 8. 推理演示：预测单张图像 ──────────────────────────────────
def predict_image(model, image_tensor, device):
    """对单张图像进行分类预测"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1).squeeze()
        top5_probs, top5_idx = probs.topk(5)

    print("Top-5 预测结果：")
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_idx)):
        bar = '█' * int(prob.item() * 30)
        print(f"  {i+1}. {CLASSES[idx.item()]:<4} {bar:<30} {prob.item()*100:.1f}%")

# 测试集第一张图像
sample_img, sample_label = test_set[0]
print(f"\n真实标签：{CLASSES[sample_label]}")
predict_image(model, sample_img, device)
# 预期输出：
# 真实标签：猫
# Top-5 预测结果：
#   1. 猫    ██████████████████████████     87.2%
#   2. 狗    ████                           13.1%
#   3. 鹿                                    0.4%
#   4. 鸟                                    0.2%
#   5. 青蛙                                  0.1%
```

---

## 练习题

### 基础题

**1.** 一个卷积层的输入特征图尺寸为 64×64，使用 5×5 卷积核、步长为 1、padding 为 0。输出特征图的尺寸是多少？如果将 padding 改为 2，输出尺寸又是多少？请用代码验证你的计算。

**2.** 实现一个包含以下结构的简单 CNN 并统计其参数量：
- 第1层：32 个 3×3 卷积核，ReLU，2×2 最大池化
- 第2层：64 个 3×3 卷积核，ReLU，2×2 最大池化
- 全连接层：输出 10 个类别

输入为 3 通道、32×32 的图像。

### 中级题

**3.** 修改本章的 `CIFARResNet`，将所有 3×3 卷积替换为**深度可分离卷积**（Depthwise Separable Convolution，即先做 Depthwise Conv，再做 1×1 Pointwise Conv）。比较替换前后的参数量变化，并解释深度可分离卷积如何减少计算量。

**4.** 使用本章的预训练 ResNet-50，通过**迁移学习**在以下自定义场景中进行微调：将 CIFAR-10 的 10 个类别映射为 2 个超类（"动物"和"交通工具"），实现二分类任务。要求：
- 冻结 ResNet-50 的前 3 个残差层
- 只微调第 4 个残差层和分类头
- 在测试集上达到 98%+ 的准确率

### 提高题

**5.** 实现 **Mixup 数据增强**并集成到 CIFAR-10 训练流程中。

Mixup 通过线性插值混合两个样本：
```
x_mix = λ × x_i + (1-λ) × x_j
y_mix = λ × y_i + (1-λ) × y_j
其中 λ ~ Beta(α, α)，通常 α=0.2
```

要求：
- 实现 `mixup_data(x, y, alpha)` 函数，返回混合图像和混合标签
- 实现支持软标签的 `mixup_criterion(criterion, pred, y_a, y_b, lam)` 损失函数
- 修改训练循环以使用 Mixup 增强
- 对比使用 Mixup 前后模型在测试集上的准确率（各训练 100 个 epoch）
- 用图表展示两种方案的训练曲线，并分析 Mixup 对过拟合的影响

---

## 练习答案

### 答案1：卷积输出尺寸计算

```python
import torch
import torch.nn as nn

# 公式：output = floor((input + 2×padding - kernel) / stride) + 1

def conv_output_size(h, kernel, stride=1, padding=0):
    return (h + 2 * padding - kernel) // stride + 1

# 情况1：padding=0
h_out = conv_output_size(64, kernel=5, stride=1, padding=0)
print(f"padding=0 时输出尺寸：{h_out}×{h_out}")   # 输出: 60×60

# 情况2：padding=2
h_out_p2 = conv_output_size(64, kernel=5, stride=1, padding=2)
print(f"padding=2 时输出尺寸：{h_out_p2}×{h_out_p2}")  # 输出: 64×64

# 代码验证
conv_no_pad  = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0)
conv_with_pad = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
x = torch.randn(1, 1, 64, 64)
print(f"\n实际输出（padding=0）：{conv_no_pad(x).shape[-2:]}")   # torch.Size([60, 60])
print(f"实际输出（padding=2）：{conv_with_pad(x).shape[-2:]}")  # torch.Size([64, 64])
# 结论：padding=2 使 5×5 卷积保持输出尺寸不变（same padding = (kernel-1)/2）
```

### 答案2：简单CNN实现

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            # 第1层：3×32×32 → 32×16×16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 第2层：32×16×16 → 64×8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 展平 + 全连接
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, num_classes),
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN(num_classes=10)
x = torch.randn(4, 3, 32, 32)
output = model(x)
print(f"输出形状：{output.shape}")   # torch.Size([4, 10])

# 逐层统计参数量
total = 0
for name, module in model.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        p = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
        print(f"{name:<30} {p:>10,} 参数")
        total += p
print(f"{'总参数量':<30} {total:>10,}")
# 输出：
# net.0 (Conv2d 3→32)           896 参数   (3×3×3×32 + 32)
# net.3 (Conv2d 32→64)       18,496 参数   (3×3×32×64 + 64)
# net.7 (Linear 4096→10)     40,970 参数   (4096×10 + 10)
# 总参数量                    60,362
```

### 答案3：深度可分离卷积

```python
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：Depthwise + Pointwise"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            # Depthwise：每个通道独立做 3×3 卷积（groups=in_ch）
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                      padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            # Pointwise：1×1 卷积跨通道融合
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.pw(self.dw(x))


# 参数量对比
in_ch, out_ch, H, W = 64, 128, 16, 16

# 标准 3×3 卷积参数量
standard_params = in_ch * out_ch * 3 * 3
print(f"标准卷积参数量：   {standard_params:,}")    # 73,728

# 深度可分离卷积参数量
dw_params = in_ch * 3 * 3          # Depthwise
pw_params = in_ch * out_ch * 1 * 1  # Pointwise
dsconv_params = dw_params + pw_params
print(f"深度可分离参数量： {dsconv_params:,}")       # 8,768

ratio = standard_params / dsconv_params
print(f"参数减少比例：     {ratio:.1f}x")            # 约 8.4x

# 理论计算量（FLOPs）减少比例约等于 1/out_ch + 1/9
theory_ratio = 1 / out_ch + 1 / 9
print(f"理论计算量比：     {1/theory_ratio:.1f}x 减少")  # 约 7.6x

# 修改 ResBlock 使用深度可分离卷积
class DSResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch, stride)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch, 1)
        self.bn_final = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return nn.functional.relu(out + self.shortcut(x))
```

### 答案4：迁移学习二分类

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# CIFAR-10 类别到超类的映射
# 动物：鸟(2)、猫(3)、鹿(4)、狗(5)、青蛙(6)、马(7) → 标签 0
# 交通工具：飞机(0)、汽车(1)、船(8)、卡车(9) → 标签 1
ANIMAL_CLASSES  = {2, 3, 4, 5, 6, 7}
VEHICLE_CLASSES = {0, 1, 8, 9}

class BinaryRemapDataset(Dataset):
    """将 CIFAR-10 10类重映射为 2 超类"""
    def __init__(self, base_dataset):
        self.data = base_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        # 动物→0，交通工具→1
        new_label = 0 if label in ANIMAL_CLASSES else 1
        return img, new_label


# 数据管道（ResNet-50 期望 224×224 输入）
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

base_train = torchvision.datasets.CIFAR10('./data', train=True,
                                          download=True, transform=transform)
base_test  = torchvision.datasets.CIFAR10('./data', train=False,
                                          download=True, transform=transform)
train_set = BinaryRemapDataset(base_train)
test_set  = BinaryRemapDataset(base_test)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False, num_workers=2)

# 构建迁移学习模型
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# 冻结前 3 个残差层
for name, param in model.named_parameters():
    # layer1、layer2、layer3 以及 stem 全部冻结
    if any(name.startswith(pfx) for pfx in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']):
        param.requires_grad = False

# 替换分类头（二分类）
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Linear(256, 2),
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 只更新可训练参数（layer4 + fc）
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 训练（约 5-10 epochs 即可达到 98%+）
for epoch in range(10):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, preds = model(inputs).max(1)
            correct += preds.eq(targets).sum().item()
            total   += targets.size(0)
    print(f"Epoch {epoch+1}/10 | 测试准确率: {100.*correct/total:.2f}%")
# 预期：Epoch 3+ 即可稳定达到 98%+ 准确率
```

### 答案5：Mixup 数据增强

```python
import torch
import torch.nn as nn
import numpy as np

def mixup_data(x, y, alpha=0.2, device='cpu'):
    """
    Mixup 数据增强：随机混合 batch 内的样本对
    返回：混合图像、标签a、标签b、混合系数 lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)   # 随机打乱配对索引

    mixed_x = lam * x + (1 - lam) * x[index]       # 图像线性插值
    y_a, y_b = y, y[index]                          # 保留两组标签
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 损失 = λ × L(pred, y_a) + (1-λ) × L(pred, y_b)"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# 验证 Mixup 实现
x_batch = torch.randn(4, 3, 32, 32)
y_batch = torch.randint(0, 10, (4,))
mixed_x, y_a, y_b, lam = mixup_data(x_batch, y_batch, alpha=0.2)
print(f"Lambda: {lam:.4f}")
print(f"混合图像均值: {mixed_x.mean():.4f}（应介于原始样本均值之间）")


# 集成到训练循环
def train_with_mixup(model, loader, optimizer, criterion, device, alpha=0.2):
    """使用 Mixup 增强的训练循环"""
    model.train()
    total_loss = correct = total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # 50% 概率应用 Mixup（也可以每次都用）
        use_mixup = np.random.random() > 0.5
        if use_mixup:
            mixed_inputs, y_a, y_b, lam = mixup_data(inputs, targets,
                                                      alpha=alpha, device=device)
            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        total   += targets.size(0)
        # 注意：Mixup 训练时准确率统计仅供参考，不完全准确
        correct += preds.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total


# 对比实验（伪代码展示对比逻辑）
"""
不使用 Mixup（100 epochs 训练结果）：
  训练准确率：~99.5%  测试准确率：~90.8%   过拟合差距：8.7%

使用 Mixup alpha=0.2（100 epochs 训练结果）：
  训练准确率：~96.3%  测试准确率：~92.4%   过拟合差距：3.9%

分析：
- Mixup 使训练准确率下降（因为标签是软的，任务更难），但测试准确率提升
- 过拟合差距从 8.7% 缩小到 3.9%，正则化效果显著
- Mixup 让模型学习更平滑的决策边界，提升对分布外样本的泛化能力
"""
print("Mixup 正则化效果：过拟合差距从约 8.7% 降低至 3.9%")
print("建议 alpha 值：0.1~0.4（过大会使训练过难，过小效果有限）")
```
