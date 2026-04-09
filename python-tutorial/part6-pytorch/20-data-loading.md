# 第20章：数据加载与预处理

> **本章主题**：掌握PyTorch的数据加载体系，学会构建高效、可复用的数据管道，为模型训练提供稳定的数据流。

---

## 学习目标

完成本章学习后，你将能够：

1. **理解Dataset抽象**：掌握`__len__`和`__getitem__`协议，能够实现符合PyTorch规范的数据集类
2. **构建自定义Dataset**：针对图像、文本、CSV等各类数据源，编写灵活的自定义Dataset
3. **熟练使用DataLoader**：理解`batch_size`、`shuffle`、`num_workers`等关键参数的作用与调优方法
4. **应用数据变换**：使用`transforms`和`Compose`构建可组合的数据预处理流水线
5. **实现数据增强**：使用随机裁剪、随机翻转、归一化等技术提升模型泛化能力

---

## 20.1 Dataset类（`__len__`、`__getitem__`）

### 为什么需要Dataset抽象

深度学习训练通常需要处理大量数据。如果一次性将所有数据加载到内存，对于ImageNet（超过100万张图片）这类大型数据集来说几乎不可能。PyTorch通过`Dataset`抽象解决了这个问题：

- **按需加载**：只在需要时读取数据，节约内存
- **统一接口**：无论数据来自文件、数据库还是网络，都用相同的方式访问
- **与DataLoader协作**：支持并行加载、批量打包、随机打乱

### Dataset的两个核心方法

`torch.utils.data.Dataset`是一个抽象基类，任何自定义数据集都必须实现两个方法：

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __len__(self):
        """返回数据集的总样本数"""
        ...

    def __getitem__(self, idx):
        """根据索引返回单个样本（及其标签）"""
        ...
```

这两个方法对应Python的迭代协议：
- `__len__`：`len(dataset)` 调用此方法
- `__getitem__`：`dataset[i]` 调用此方法

### 最简单的Dataset示例

```python
import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        """
        参数:
            data:   特征数据，形状 (N, ...)
            labels: 标签数据，形状 (N,)
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 创建示例数据
features = torch.randn(100, 10)   # 100个样本，每个10维
labels = torch.randint(0, 3, (100,))  # 3分类标签

dataset = SimpleDataset(features, labels)

print(f"数据集大小: {len(dataset)}")
# 输出: 数据集大小: 100

sample, label = dataset[0]
print(f"单个样本形状: {sample.shape}, 标签: {label}")
# 输出: 单个样本形状: torch.Size([10]), 标签: 1
```

### 索引的灵活性

`__getitem__`不仅支持整数索引，还支持切片（当与DataLoader配合时，DataLoader会传入整数列表）：

```python
class FlexibleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # idx 可以是整数或整数列表
        return {
            'features': self.data[idx],
            'label': self.labels[idx],
            'index': idx  # 保留原始索引，便于调试
        }
```

> **注意**：`__getitem__`的返回值可以是张量、numpy数组、字典或元组。DataLoader会自动将它们组合成批次（batch）。但字典的值必须是可堆叠的张量类型。

---

## 20.2 自定义Dataset

### 从CSV文件加载数据

许多机器学习数据集以CSV格式存储。以下是一个完整的CSV数据集实现：

```python
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    """从CSV文件加载表格数据"""

    def __init__(self, csv_path, feature_cols, label_col, transform=None):
        """
        参数:
            csv_path:     CSV文件路径
            feature_cols: 特征列名列表
            label_col:    标签列名
            transform:    可选的数据变换函数
        """
        df = pd.read_csv(csv_path)
        self.features = torch.tensor(
            df[feature_cols].values, dtype=torch.float32
        )
        self.labels = torch.tensor(
            df[label_col].values, dtype=torch.long
        )
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y


# 使用示例（假设有 iris.csv 文件）
# dataset = CSVDataset(
#     csv_path='iris.csv',
#     feature_cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
#     label_col='species'
# )
```

### 从图像文件夹加载数据

图像分类任务中最常见的数据组织方式是"类别即文件夹"结构：

```
data/
  train/
    cat/    img001.jpg, img002.jpg, ...
    dog/    img101.jpg, img102.jpg, ...
    bird/   img201.jpg, img202.jpg, ...
  val/
    cat/    ...
    dog/    ...
```

```python
import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    """从文件夹结构加载图像数据集（类别即文件夹名）"""

    # 支持的图像格式
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    def __init__(self, root_dir, transform=None):
        """
        参数:
            root_dir:  根目录，包含各类别子目录
            transform: 图像变换（torchvision.transforms）
        """
        self.root_dir = root_dir
        self.transform = transform

        # 构建类别列表（排序保证顺序一致性）
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 扫描所有图像文件
        self.samples = self._make_dataset()

    def _make_dataset(self):
        """扫描目录，返回 (图像路径, 类别索引) 列表"""
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(self.IMG_EXTENSIONS):
                    path = os.path.join(class_dir, fname)
                    samples.append((path, class_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 用PIL打开图像，转为RGB（处理灰度图和RGBA图）
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# 使用示例
# dataset = ImageFolderDataset(root_dir='data/train', transform=my_transform)
# print(f"类别: {dataset.classes}")
# print(f"样本数: {len(dataset)}")
```

### 支持数据集分割

实际项目中，常需要将数据集分割为训练集和验证集：

```python
from torch.utils.data import random_split, Subset

# 方法1：random_split（随机按比例分割）
dataset = SimpleDataset(features, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # 固定随机种子
)

print(f"训练集: {len(train_dataset)} 样本")
# 输出: 训练集: 80 样本
print(f"验证集: {len(val_dataset)} 样本")
# 输出: 验证集: 20 样本

# 方法2：Subset（手动指定索引）
indices = list(range(len(dataset)))
train_indices = indices[:80]
val_indices = indices[80:]

train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)
```

### 内存中的完整数据集：TensorDataset

对于能完全放入内存的数据，`TensorDataset`是最简单的选择：

```python
from torch.utils.data import TensorDataset

X = torch.randn(200, 5)   # 200个样本，5个特征
y = torch.randint(0, 2, (200,))  # 二分类标签

# TensorDataset 自动实现 __len__ 和 __getitem__
dataset = TensorDataset(X, y)

print(f"样本数: {len(dataset)}")
# 输出: 样本数: 200

x_sample, y_sample = dataset[0]
print(f"特征: {x_sample.shape}, 标签: {y_sample}")
# 输出: 特征: torch.Size([5]), 标签: 0
```

---

## 20.3 DataLoader（`batch_size`、`shuffle`、`num_workers`）

### DataLoader的作用

`DataLoader`是数据集与模型训练之间的桥梁，它负责：

1. **批量采样**：将单个样本打包成 mini-batch
2. **随机打乱**：每个epoch随机重排样本顺序
3. **并行加载**：使用多进程加速数据读取
4. **内存钉扎**：将数据固定在内存，加速GPU传输

```python
from torch.utils.data import DataLoader

# 基本使用
loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 迭代一个epoch
for batch_features, batch_labels in loader:
    print(f"批次形状: {batch_features.shape}, 标签形状: {batch_labels.shape}")
    # 输出: 批次形状: torch.Size([32, 10]), 标签形状: torch.Size([32])
    break
```

### 关键参数详解

#### `batch_size`：批次大小

批次大小是影响训练的重要超参数：

```python
# 小批次：更多梯度更新步骤，内存占用低，训练噪声更大（有正则化效果）
small_loader = DataLoader(dataset, batch_size=8)

# 大批次：每个epoch更新次数少，内存占用高，但梯度估计更准确
large_loader = DataLoader(dataset, batch_size=256)

# 全批次梯度下降（不推荐用于大数据集）
full_loader = DataLoader(dataset, batch_size=len(dataset))
```

| 批次大小 | 优点 | 缺点 |
|----------|------|------|
| 小（8-32） | 内存友好，有正则化效果 | 训练慢，梯度方差大 |
| 中（64-256） | 平衡 | — |
| 大（512+） | 训练快，梯度稳定 | 需要大内存，易陷入局部最优 |

#### `shuffle`：随机打乱

```python
# 训练集：必须打乱，避免模型记住样本顺序
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 验证集/测试集：不需要打乱（保证结果可复现）
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
```

#### `num_workers`：并行加载进程数

```python
import torch
import time

# 测试不同 num_workers 的加载速度
for num_workers in [0, 1, 2, 4]:
    loader = DataLoader(dataset, batch_size=32, num_workers=num_workers)
    start = time.time()
    for _ in loader:
        pass
    elapsed = time.time() - start
    print(f"num_workers={num_workers}: {elapsed:.3f}秒")
```

> **最佳实践**：
> - **Windows系统**：`num_workers` 需要在 `if __name__ == '__main__':` 块内设置，或设为 0
> - **Linux/macOS**：通常设为 CPU核心数的一半（如4核CPU设为2）
> - **GPU训练**：加大 `num_workers` 可以避免GPU等待数据
> - **调试阶段**：设为 0，单进程便于排查问题

#### 其他重要参数

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,       # 将数据固定在内存，加速GPU传输
    drop_last=True,        # 丢弃最后一个不完整批次（避免批归一化出错）
    prefetch_factor=2,     # 每个worker预取的批次数（num_workers>0时有效）
    persistent_workers=True  # 保持worker进程存活（避免epoch间重新启动）
)
```

### 理解迭代行为

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# 创建10个样本的数据集
X = torch.arange(10, dtype=torch.float32).reshape(10, 1)
y = torch.arange(10)
dataset = TensorDataset(X, y)

# batch_size=3，drop_last=False（默认）
loader = DataLoader(dataset, batch_size=3, shuffle=False, drop_last=False)
print(f"批次数量: {len(loader)}")
# 输出: 批次数量: 4  （3+3+3+1，最后一个批次只有1个样本）

for i, (x_batch, y_batch) in enumerate(loader):
    print(f"批次{i}: x={x_batch.squeeze().tolist()}")
# 输出:
# 批次0: x=[0.0, 1.0, 2.0]
# 批次1: x=[3.0, 4.0, 5.0]
# 批次2: x=[6.0, 7.0, 8.0]
# 批次3: x=[9.0]

# drop_last=True：丢弃最后不足一个批次的样本
loader_drop = DataLoader(dataset, batch_size=3, shuffle=False, drop_last=True)
print(f"批次数量(drop_last=True): {len(loader_drop)}")
# 输出: 批次数量(drop_last=True): 3
```

### 自定义collate_fn

默认情况下，DataLoader将样本列表堆叠成张量。对于变长序列（如NLP任务），需要自定义`collate_fn`：

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_variable_length(batch):
    """将变长序列填充到相同长度"""
    sequences, labels = zip(*batch)
    # pad_sequence 将序列填充到批次中最长序列的长度
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded, labels

# 假设每个样本是不同长度的序列
class VariableLengthDataset(Dataset):
    def __init__(self):
        self.data = [
            (torch.randn(torch.randint(3, 10, ()).item()), torch.tensor(i % 3))
            for i in range(20)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

vl_dataset = VariableLengthDataset()
vl_loader = DataLoader(
    vl_dataset,
    batch_size=4,
    collate_fn=collate_variable_length
)
```

---

## 20.4 数据变换（`transforms`、`Compose`）

### torchvision.transforms概述

`torchvision.transforms`提供了一套标准的图像预处理操作，每个变换都是一个可调用的类：

```python
from torchvision import transforms
from PIL import Image
import torch

# 单个变换
to_tensor = transforms.ToTensor()
resize = transforms.Resize((224, 224))
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

### Compose：组合多个变换

`Compose`将多个变换串联成一个流水线，按顺序依次应用：

```python
# 标准的ImageNet预处理流水线
imagenet_transform = transforms.Compose([
    transforms.Resize(256),          # 短边缩放到256
    transforms.CenterCrop(224),      # 中心裁剪224x224
    transforms.ToTensor(),           # PIL Image -> Tensor，像素值从[0,255]缩放到[0,1]
    transforms.Normalize(            # 标准化到标准正态分布
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 应用变换
# image = Image.open('photo.jpg')
# tensor = imagenet_transform(image)  # 形状: (3, 224, 224)
```

### 理解ToTensor的作用

```python
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# 模拟一张小图像
img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
pil_image = Image.fromarray(img_array)

to_tensor = transforms.ToTensor()
tensor = to_tensor(pil_image)

print(f"PIL图像大小: {pil_image.size}")         # 输出: PIL图像大小: (64, 64)
print(f"Tensor形状: {tensor.shape}")             # 输出: Tensor形状: torch.Size([3, 64, 64])
print(f"像素值范围: [{tensor.min():.3f}, {tensor.max():.3f}]")
# 输出: 像素值范围: [0.000, 1.000]
# ToTensor 做了两件事：
# 1. HxWxC（PIL格式）-> CxHxW（PyTorch格式）
# 2. uint8 [0,255] -> float32 [0.0,1.0]
```

### 训练集与验证集使用不同变换

训练集需要数据增强，验证集只需标准化：

```python
from torchvision import transforms

# 训练集变换：包含数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),    # 随机裁剪并缩放
    transforms.RandomHorizontalFlip(),    # 随机水平翻转
    transforms.ColorJitter(               # 颜色抖动
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 验证集变换：只做标准化，不做随机增强
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 自定义变换

自定义变换只需实现`__call__`方法：

```python
import torch
from torchvision import transforms

class AddGaussianNoise:
    """向图像张量添加高斯噪声（数据增强）"""

    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class SquarePad:
    """将图像填充为正方形（保持纵横比）"""

    def __call__(self, image):
        w, h = image.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        pad_right = max_dim - w - pad_left
        pad_bottom = max_dim - h - pad_top
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return transforms.functional.pad(image, padding, fill=0)


# 在流水线中使用自定义变换
custom_transform = transforms.Compose([
    SquarePad(),
    transforms.Resize(224),
    transforms.ToTensor(),
    AddGaussianNoise(std=0.02),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

### 新版API：v2 transforms（PyTorch 2.0+）

PyTorch 2.0引入了`torchvision.transforms.v2`，支持同时变换图像和标注框：

```python
# torchvision >= 0.15
from torchvision.transforms import v2

# v2 API 可以同时变换图像和边界框/掩码
transform_v2 = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),  # 替代 ToTensor
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## 20.5 数据增强（RandomCrop、RandomFlip、Normalize）

### 为什么需要数据增强

数据增强通过对训练样本施加随机变换，人为扩大数据集规模，减少模型对特定视角、位置、光照条件的过拟合：

```
原始图像 → [随机翻转] → [随机裁剪] → [颜色抖动] → 增强后图像
             （有50%概率）  （随机区域）   （随机调整）
```

> **关键原则**：数据增强只在**训练阶段**使用，验证和测试阶段使用确定性变换。

### RandomCrop：随机裁剪

```python
from torchvision import transforms
from PIL import Image
import numpy as np

# 模拟32x32图像（如CIFAR-10）
img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))

# RandomCrop：从图像内随机裁剪指定大小
# padding=4 先在四周填充4像素，再裁剪回32x32，相当于平移增强
random_crop = transforms.RandomCrop(
    size=32,
    padding=4,            # 先填充再裁剪
    padding_mode='reflect' # 填充模式：reflect（镜像）或 constant（0填充）
)

# 每次调用结果不同
for i in range(3):
    cropped = random_crop(img)
    print(f"第{i+1}次裁剪后大小: {cropped.size}")
    # 输出: 第1次裁剪后大小: (32, 32)

# RandomResizedCrop：同时随机缩放和裁剪（ImageNet标准增强）
rand_resized_crop = transforms.RandomResizedCrop(
    size=224,
    scale=(0.08, 1.0),   # 裁剪面积占原图面积的比例范围
    ratio=(3/4, 4/3)     # 裁剪区域的宽高比范围
)
```

### RandomHorizontalFlip / RandomVerticalFlip：随机翻转

```python
from torchvision import transforms

# 水平翻转（左右镜像）——自然图像最常用
h_flip = transforms.RandomHorizontalFlip(p=0.5)  # p=翻转概率

# 垂直翻转（上下颠倒）——航拍图像、医学图像常用
v_flip = transforms.RandomVerticalFlip(p=0.5)

# 随机旋转
random_rotate = transforms.RandomRotation(
    degrees=15,          # 旋转角度范围：[-15, 15]度
    fill=0               # 旋转后空白区域的填充值
)

# 验证翻转效果
import torch

dummy = transforms.ToTensor()(
    Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4, 1).repeat(3, axis=2))
)
print("原始（第0行）:", dummy[0, 0, :])
# 输出: 原始（第0行）: tensor([0., 1., 2., 3.]) / 255

flipped = transforms.functional.hflip(dummy)
print("翻转（第0行）:", flipped[0, 0, :])
# 输出: 翻转（第0行）: tensor([3., 2., 1., 0.]) / 255
```

### Normalize：归一化

归一化是最重要的预处理步骤之一，它将像素值调整到标准分布，加速模型收敛：

```python
from torchvision import transforms
import torch

# Normalize 公式：output = (input - mean) / std
# 对每个通道独立计算

# CIFAR-10的均值和标准差（在训练集上计算得到）
cifar10_normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2470, 0.2435, 0.2616]
)

# ImageNet的均值和标准差
imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# 演示归一化效果
tensor = torch.ones(3, 4, 4) * 0.5  # 所有像素值为0.5
normalized = imagenet_normalize(tensor)
print(f"归一化前: {tensor[0, 0, 0]:.3f}")
# 输出: 归一化前: 0.500
print(f"归一化后（R通道）: {normalized[0, 0, 0]:.3f}")
# 输出: 归一化后（R通道）: 0.077  （(0.5-0.485)/0.229 ≈ 0.066）

# 如何计算自定义数据集的均值和标准差
def compute_mean_std(dataset, batch_size=64):
    """计算数据集的均值和标准差"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    for images, _ in loader:
        batch_size = images.size(0)
        # images: (N, C, H, W)
        images = images.view(batch_size, images.size(1), -1)  # (N, C, H*W)
        mean += images.mean(dim=[0, 2]) * batch_size
        std += images.std(dim=[0, 2]) * batch_size
        total_samples += batch_size

    mean /= total_samples
    std /= total_samples
    return mean, std
```

### 综合增强策略：CIFAR-10示例

```python
from torchvision import transforms

# CIFAR-10 标准训练增强策略
cifar10_train_transform = transforms.Compose([
    # 1. 随机裁剪（平移不变性）
    transforms.RandomCrop(32, padding=4),

    # 2. 随机水平翻转（左右对称性）
    transforms.RandomHorizontalFlip(p=0.5),

    # 3. 颜色抖动（光照和颜色变化不变性）
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    # 4. 转为张量
    transforms.ToTensor(),

    # 5. 归一化
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

cifar10_val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

# 验证集不使用随机增强，确保评估结果可复现
print("训练变换:", cifar10_train_transform)
print("验证变换:", cifar10_val_transform)
```

### 高级增强技术

```python
from torchvision import transforms
import torch

# AutoAugment：自动搜索最优增强策略（ImageNet/CIFAR效果显著）
auto_augment = transforms.AutoAugment(
    policy=transforms.AutoAugmentPolicy.CIFAR10
)

# RandAugment：随机选择增强操作
rand_augment = transforms.RandAugment(
    num_ops=2,    # 每次随机应用2个操作
    magnitude=9   # 操作强度（0-30）
)

# TrivialAugmentWide：简单且效果好的增强策略
trivial_augment = transforms.TrivialAugmentWide()

# Cutout/RandomErasing：随机遮挡（模拟遮挡场景）
random_erasing = transforms.RandomErasing(
    p=0.5,           # 应用概率
    scale=(0.02, 0.33),  # 遮挡区域占图像面积的比例
    ratio=(0.3, 3.3),    # 遮挡区域的宽高比
    value=0          # 填充值（0=黑色）
)

# MixUp 和 CutMix 需要在DataLoader层面实现（涉及两个样本的混合）
def mixup_data(x, y, alpha=0.2):
    """MixUp数据增强：混合两个样本"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
```

---

## 本章小结

| 概念 | 类/函数 | 核心职责 |
|------|---------|---------|
| 数据集基类 | `torch.utils.data.Dataset` | 定义数据访问接口，必须实现`__len__`和`__getitem__` |
| 张量数据集 | `TensorDataset` | 将内存中的张量包装成Dataset |
| 数据集分割 | `random_split` / `Subset` | 按比例或按索引分割数据集 |
| 数据加载器 | `DataLoader` | 批量加载、随机打乱、并行预取 |
| 批次大小 | `batch_size` | 每次迭代的样本数，影响训练速度和稳定性 |
| 随机打乱 | `shuffle=True` | 每个epoch重新随机排列样本（训练集必须） |
| 并行加载 | `num_workers` | 使用多进程加速数据读取，典型值为4 |
| 内存固定 | `pin_memory=True` | 加速CPU到GPU的数据传输 |
| 丢弃尾批 | `drop_last=True` | 避免最后一批样本数不足影响批归一化 |
| 变换流水线 | `transforms.Compose` | 将多个变换串联，按顺序依次应用 |
| 张量转换 | `transforms.ToTensor` | PIL Image → CxHxW张量，像素值缩放到[0,1] |
| 归一化 | `transforms.Normalize` | 按通道减均值除标准差，加速收敛 |
| 随机裁剪 | `RandomCrop` | 随机位置裁剪，增强位置不变性 |
| 随机翻转 | `RandomHorizontalFlip` | 随机镜像，增强对称不变性 |
| 颜色抖动 | `ColorJitter` | 随机调整亮度/对比度/饱和度/色调 |
| 随机遮挡 | `RandomErasing` | 随机遮挡区域，模拟遮挡场景 |
| 自动增强 | `AutoAugment` | 自动搜索的最优增强策略 |

---

## 深度学习应用：高效数据管道

### 应用背景

本节展示如何为图像分类任务（以CIFAR-10为例）构建完整的数据加载流程。CIFAR-10包含10个类别、共60000张32×32彩色图像（50000训练 + 10000测试）。

### 完整实现

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
import os


# ============================================================
# 第一步：定义数据变换流水线
# ============================================================

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

# 训练集：带数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

# 验证集/测试集：只做标准化
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])


# ============================================================
# 第二步：加载数据集
# ============================================================

def get_cifar10_loaders(
    data_dir='./data',
    batch_size=128,
    val_split=0.1,
    num_workers=4,
    pin_memory=True
):
    """
    构建CIFAR-10的训练、验证、测试数据加载器

    参数:
        data_dir:    数据存储目录
        batch_size:  批次大小
        val_split:   从训练集中划分验证集的比例
        num_workers: 并行加载进程数
        pin_memory:  是否使用内存固定（GPU训练时设为True）

    返回:
        train_loader, val_loader, test_loader
    """
    # 下载并加载CIFAR-10（torchvision内置数据集）
    full_train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform
    )

    # 将训练集分割为训练集和验证集
    n_total = len(full_train_set)   # 50000
    n_val   = int(n_total * val_split)  # 5000
    n_train = n_total - n_val           # 45000

    train_set, val_set_raw = random_split(
        full_train_set,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # 验证集需要使用 eval_transform（不做数据增强）
    # 由于 random_split 返回的 Subset 共用父数据集的 transform，
    # 需要单独创建一个使用 eval_transform 的数据集
    val_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=eval_transform
    )
    from torch.utils.data import Subset
    val_indices = val_set_raw.indices
    val_set = Subset(val_set, val_indices)

    # 构建 DataLoader
    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,   # 确保每个批次大小一致
        **common_kwargs
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,  # 验证时可以用更大的批次
        shuffle=False,
        **common_kwargs
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size * 2,
        shuffle=False,
        **common_kwargs
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 第三步：验证数据管道
# ============================================================

def inspect_dataloader(loader, name="DataLoader"):
    """检查数据加载器的基本信息"""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  批次数量:   {len(loader)}")
    print(f"  总样本数:   {len(loader.dataset)}")
    print(f"  批次大小:   {loader.batch_size}")

    # 检查第一个批次
    images, labels = next(iter(loader))
    print(f"  图像形状:   {images.shape}")
    print(f"  标签形状:   {labels.shape}")
    print(f"  像素值范围: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  均值:       {images.mean(dim=[0,2,3])}")
    print(f"  标准差:     {images.std(dim=[0,2,3])}")


# ============================================================
# 第四步：性能基准测试
# ============================================================

def benchmark_loader(loader, num_batches=50):
    """测试数据加载速度"""
    start = time.time()
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break
        # 模拟GPU传输
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
    elapsed = time.time() - start
    throughput = num_batches * loader.batch_size / elapsed
    print(f"  {num_batches}批次耗时: {elapsed:.2f}s  |  吞吐量: {throughput:.0f} 样本/秒")


# ============================================================
# 第五步：简单的CNN模型（验证数据管道可用性）
# ============================================================

class SimpleCNN(nn.Module):
    """用于CIFAR-10的简单卷积网络"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16x16 -> 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)) # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# 第六步：训练循环（展示数据管道的完整使用）
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # 将数据移动到计算设备
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ============================================================
# 主程序
# ============================================================

def main():
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 构建数据管道
    print("\n正在构建数据管道...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir='./data',
        batch_size=128,
        num_workers=4 if os.name != 'nt' else 0,  # Windows使用0
        pin_memory=torch.cuda.is_available()
    )

    # 检查数据加载器
    inspect_dataloader(train_loader, "训练集 DataLoader")
    inspect_dataloader(val_loader,   "验证集 DataLoader")
    inspect_dataloader(test_loader,  "测试集 DataLoader")

    # 性能基准测试
    print("\n数据加载速度测试:")
    benchmark_loader(train_loader)

    # 构建模型
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20
    )

    # 训练5个epoch演示
    print("\n开始训练（演示5个epoch）...")
    print(f"{'Epoch':>6} | {'训练损失':>8} | {'训练准确率':>10} | {'验证损失':>8} | {'验证准确率':>10}")
    print("-" * 60)

    for epoch in range(1, 6):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(f"{epoch:>6} | {train_loss:>8.4f} | {train_acc*100:>9.2f}% | "
              f"{val_loss:>8.4f} | {val_acc*100:>9.2f}%")

    # 最终测试集评估
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n最终测试集准确率: {test_acc*100:.2f}%")


if __name__ == '__main__':
    main()
```

### 预期输出

```
使用设备: cpu

正在构建数据管道...
下载 CIFAR-10...

==================================================
  训练集 DataLoader
==================================================
  批次数量:   351
  总样本数:   45000
  批次大小:   128
  图像形状:   torch.Size([128, 3, 32, 32])
  标签形状:   torch.Size([128])
  像素值范围: [-2.429, 2.754]
  均值:       tensor([-0.0023,  0.0031, -0.0018])
  标准差:     tensor([0.9987, 0.9992, 0.9989])

==================================================
  验证集 DataLoader
==================================================
  批次数量:   20
  总样本数:   5000
  批次大小:   256
  ...

数据加载速度测试:
  50批次耗时: 3.21s  |  吞吐量: 1994 样本/秒

开始训练（演示5个epoch）...
 Epoch | 训练损失 | 训练准确率 |  验证损失 | 验证准确率
------------------------------------------------------------
     1 |   1.6823 |    39.54%  |    1.5121 |    44.60%
     2 |   1.3201 |    52.71%  |    1.2834 |    53.82%
     3 |   1.1456 |    59.38%  |    1.1203 |    60.14%
     4 |   1.0213 |    63.89%  |    1.0421 |    63.20%
     5 |   0.9287 |    67.23%  |    0.9876 |    65.40%

最终测试集准确率: 64.82%
```

### 关键设计要点

1. **训练集与验证集使用不同变换**：训练集包含随机增强，验证集只做标准化——这是常见的错误来源
2. **`non_blocking=True`**：配合`pin_memory=True`使CPU-GPU数据传输与计算重叠，提升GPU利用率
3. **`drop_last=True`**：批归一化（BatchNorm）对批次大小敏感，丢弃不完整批次避免异常
4. **`persistent_workers=True`**：避免每个epoch重新启动Worker进程，减少开销
5. **固定随机种子**：`random_split`使用`torch.Generator().manual_seed(42)`确保每次运行划分相同

---

## 练习题

### 基础题

**练习1**：实现一个`RangeDataset`，它的样本是从0到`n-1`的整数，标签是样本值的奇偶性（0表示偶数，1表示奇数）。验证`len()`和索引访问均正常工作。

**练习2**：使用`DataLoader`加载练习1的`RangeDataset`（`n=20`，`batch_size=4`，`shuffle=False`），打印所有批次的样本和标签，验证无数据遗漏且批次划分正确。

### 中级题

**练习3**：实现一个`NoisyDataset`装饰器类，它接受一个已有的Dataset，并在`__getitem__`中对样本添加标准正态噪声（乘以系数`sigma`）。要求：只对特征添加噪声，不改变标签；支持通过参数控制噪声强度。

**练习4**：实现一个自定义`transforms`类`RandomBrightness`，随机将图像张量的亮度调整为原来的`[1-delta, 1+delta]`倍（delta由用户指定），并将结果裁剪到`[0, 1]`范围。将其集成到一个`Compose`流水线中并测试。

### 提高题

**练习5**：构建一个通用的`FolderDataset`类，支持：
- 从任意文件夹结构（类别即子目录名）加载图像
- 支持`train_transform`和`val_transform`分别用于训练和验证模式
- 支持通过`split='train'`或`split='val'`参数在同一个根目录下加载不同子目录
- 提供`class_weights`属性，返回每个类别的样本权重（用于处理类别不平衡问题，权重 = 总样本数 / (类别数 × 该类别样本数)）
- 编写测试代码，用随机生成的假图像数据验证所有功能

---

## 练习答案

### 答案1

```python
import torch
from torch.utils.data import Dataset

class RangeDataset(Dataset):
    """样本为0到n-1的整数，标签为奇偶性"""

    def __init__(self, n):
        self.n = n
        self.data = torch.arange(n, dtype=torch.float32)
        self.labels = self.data.long() % 2  # 偶数→0，奇数→1

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 测试
dataset = RangeDataset(10)
print(f"数据集大小: {len(dataset)}")
# 输出: 数据集大小: 10

for i in range(10):
    sample, label = dataset[i]
    print(f"  样本={sample.item():.0f}, 标签={label.item()} ({'偶' if label==0 else '奇'})")
# 输出:
#   样本=0, 标签=0 (偶)
#   样本=1, 标签=1 (奇)
#   样本=2, 标签=0 (偶)
#   ...
```

### 答案2

```python
from torch.utils.data import DataLoader

dataset = RangeDataset(20)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

print(f"总批次数: {len(loader)}")
# 输出: 总批次数: 5

total_samples = 0
for batch_idx, (samples, labels) in enumerate(loader):
    print(f"批次{batch_idx}: 样本={samples.long().tolist()}, 标签={labels.tolist()}")
    total_samples += len(samples)

print(f"总样本数: {total_samples}")
# 输出:
# 批次0: 样本=[0, 1, 2, 3], 标签=[0, 1, 0, 1]
# 批次1: 样本=[4, 5, 6, 7], 标签=[0, 1, 0, 1]
# 批次2: 样本=[8, 9, 10, 11], 标签=[0, 1, 0, 1]
# 批次3: 样本=[12, 13, 14, 15], 标签=[0, 1, 0, 1]
# 批次4: 样本=[16, 17, 18, 19], 标签=[0, 1, 0, 1]
# 总样本数: 20
```

### 答案3

```python
import torch
from torch.utils.data import Dataset

class NoisyDataset(Dataset):
    """
    为已有Dataset的特征添加高斯噪声的装饰器

    参数:
        base_dataset: 原始数据集
        sigma:        噪声标准差（默认0.1）
    """

    def __init__(self, base_dataset, sigma=0.1):
        self.base_dataset = base_dataset
        self.sigma = sigma

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample, label = self.base_dataset[idx]
        # 只对特征添加噪声，标签不变
        noise = torch.randn_like(sample) * self.sigma
        noisy_sample = sample + noise
        return noisy_sample, label


# 测试
base = RangeDataset(10)
noisy = NoisyDataset(base, sigma=0.5)

print(f"装饰后大小: {len(noisy)}")
# 输出: 装饰后大小: 10

orig_sample, label = base[3]
noisy_sample, noisy_label = noisy[3]
print(f"原始样本: {orig_sample.item():.3f}, 标签: {label.item()}")
print(f"噪声样本: {noisy_sample.item():.3f}, 标签: {noisy_label.item()}")
# 输出: 原始样本: 3.000, 标签: 1
# 输出: 噪声样本: 3.127, 标签: 1  （噪声随机，值会不同）
```

### 答案4

```python
import torch
from torchvision import transforms


class RandomBrightness:
    """随机亮度调整变换"""

    def __init__(self, delta=0.2):
        """
        参数:
            delta: 亮度变化范围，调整倍数为 [1-delta, 1+delta]
        """
        assert 0 <= delta <= 1, "delta 必须在 [0, 1] 范围内"
        self.delta = delta

    def __call__(self, tensor):
        """
        参数:
            tensor: 形状为 (C, H, W) 的浮点张量，值域 [0, 1]
        返回:
            亮度随机调整后的张量
        """
        factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.delta
        return torch.clamp(tensor * factor, 0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(delta={self.delta})"


# 集成到 Compose 流水线
import numpy as np
from PIL import Image

pipeline = transforms.Compose([
    transforms.ToTensor(),          # PIL -> Tensor [0,1]
    RandomBrightness(delta=0.3),    # 随机调整亮度
    transforms.Normalize(           # 标准化
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# 测试
dummy_img = Image.fromarray(
    (np.ones((32, 32, 3)) * 128).astype(np.uint8)  # 全灰图像
)

# 运行多次验证随机性
results = []
for _ in range(5):
    result = pipeline(dummy_img)
    results.append(result.mean().item())

print("5次亮度调整后的均值:", [f"{v:.3f}" for v in results])
# 输出: 5次亮度调整后的均值: ['-0.019', '0.134', '-0.092', '0.201', '-0.063']
# （值各不相同，说明随机性有效）

print(f"\n变换流水线:\n{pipeline}")
```

### 答案5

```python
import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class FolderDataset(Dataset):
    """
    通用图像文件夹数据集

    目录结构:
        root/
            train/
                class_a/  img1.jpg, img2.jpg, ...
                class_b/  ...
            val/
                class_a/  ...
                class_b/  ...
    """

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

    def __init__(self, root, split='train', transform=None):
        """
        参数:
            root:      根目录（包含 train/ 和 val/ 子目录）
            split:     'train' 或 'val'
            transform: 图像变换流水线
        """
        assert split in ('train', 'val'), "split 必须为 'train' 或 'val'"
        self.root = os.path.join(root, split)
        self.split = split
        self.transform = transform

        # 扫描类别
        self.classes = sorted([
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 扫描样本
        self.samples = self._scan_samples()

    def _scan_samples(self):
        samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root, cls)
            cls_idx = self.class_to_idx[cls]
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(self.IMG_EXTENSIONS):
                    samples.append((os.path.join(cls_dir, fname), cls_idx))
        return samples

    @property
    def class_weights(self):
        """
        计算每个类别的样本权重，用于处理类别不平衡

        返回:
            weights: 长度为样本数的张量，每个样本的权重
        """
        n_total = len(self.samples)
        n_classes = len(self.classes)

        # 统计每个类别的样本数
        class_counts = torch.zeros(n_classes)
        for _, label in self.samples:
            class_counts[label] += 1

        # 计算每个类别的权重：总样本数 / (类别数 × 该类别样本数)
        weights_per_class = n_total / (n_classes * class_counts)

        # 为每个样本分配对应类别的权重
        sample_weights = torch.tensor([
            weights_per_class[label].item()
            for _, label in self.samples
        ])
        return sample_weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# --------------------------------------------------------
# 测试代码：用随机生成的假图像数据验证所有功能
# --------------------------------------------------------

import tempfile
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

def create_fake_dataset(root, splits, classes, samples_per_class):
    """创建测试用的假图像目录结构"""
    for split in splits:
        for cls, n in zip(classes, samples_per_class):
            cls_dir = os.path.join(root, split, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(n):
                # 创建随机32x32 RGB图像
                arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(arr).save(os.path.join(cls_dir, f'img_{i:04d}.jpg'))


# 在临时目录中创建测试数据集（不平衡：cat=30, dog=10, bird=5）
with tempfile.TemporaryDirectory() as tmp_dir:
    classes = ['bird', 'cat', 'dog']
    create_fake_dataset(
        root=tmp_dir,
        splits=['train', 'val'],
        classes=classes,
        samples_per_class=[5, 30, 10]   # 不平衡数据集
    )

    # 定义变换
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # 加载数据集
    train_ds = FolderDataset(tmp_dir, split='train', transform=train_tf)
    val_ds   = FolderDataset(tmp_dir, split='val',   transform=val_tf)

    print("=== FolderDataset 功能验证 ===")
    print(f"训练集大小: {len(train_ds)}")   # 输出: 训练集大小: 45
    print(f"验证集大小: {len(val_ds)}")     # 输出: 验证集大小: 45
    print(f"类别: {train_ds.classes}")      # 输出: 类别: ['bird', 'cat', 'dog']

    # 验证 class_weights
    weights = train_ds.class_weights
    print(f"\n样本权重形状: {weights.shape}")
    print(f"前5个样本权重: {weights[:5].tolist()}")

    # 使用权重采样器（处理类别不平衡）
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_ds),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        sampler=sampler,  # 使用权重采样器替代 shuffle=True
        num_workers=0
    )

    # 验证一个批次
    images, labels = next(iter(train_loader))
    print(f"\n批次图像形状: {images.shape}")  # 输出: 批次图像形状: torch.Size([8, 3, 32, 32])
    print(f"批次标签: {labels.tolist()}")

    # 统计采样器的类别分布（应接近均匀）
    label_counts = torch.zeros(3)
    for _, batch_labels in train_loader:
        for l in batch_labels:
            label_counts[l] += 1
    print(f"\n加权采样类别分布: {label_counts.long().tolist()}")
    # 输出应接近均匀，而不是原始的 [5, 30, 10]
    # 例如: [14, 17, 14]

    print("\n所有功能验证通过！")
```

---

*下一章：[第21章：图像分类实战](../part7-deep-learning/21-image-classification.md)*
