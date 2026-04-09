# 第18章：神经网络构建

## 学习目标

完成本章学习后，你将能够：

1. 理解并掌握 `nn.Module` 的基本结构，能够自定义神经网络层和模型
2. 熟练使用 PyTorch 中常用的网络层（Linear、Conv2d、BatchNorm、Dropout）
3. 了解各种激活函数（ReLU、Sigmoid、Tanh、Softmax）的特性与适用场景
4. 使用 Sequential、ModuleList、ModuleDict 等容器灵活组织网络结构
5. 掌握模型参数的查看、提取与保存/加载方法

---

## 18.1 nn.Module 基础

### 18.1.1 什么是 nn.Module

`nn.Module` 是 PyTorch 中所有神经网络模块的基类。无论是单个层（如全连接层）还是整个网络，都应当继承自 `nn.Module`。它提供了以下核心功能：

- 自动追踪模型中的所有可学习参数（`Parameter`）
- 支持 `.train()` 和 `.eval()` 模式切换（影响 Dropout、BatchNorm 等层的行为）
- 支持 `.to(device)` 将所有参数移动到指定设备（CPU/GPU）
- 支持 `.state_dict()` 和 `.load_state_dict()` 进行模型序列化

### 18.1.2 自定义一个简单模型

定义神经网络的标准方式是继承 `nn.Module` 并实现两个方法：

- `__init__`：在此定义网络层（子模块和参数）
- `forward`：定义数据的前向传播逻辑

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # 必须调用父类 __init__
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 全连接 + 激活
        x = self.fc2(x)           # 输出层（无激活）
        return x

# 实例化模型
model = SimpleNet(input_size=784, hidden_size=256, output_size=10)
print(model)
```

输出：
```
SimpleNet(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=10, bias=True)
)
```

### 18.1.3 forward 方法的调用

不要直接调用 `model.forward(x)`，而应调用 `model(x)`。后者会额外触发 PyTorch 注册的钩子（hooks），这在调试和高级功能中非常重要。

```python
x = torch.randn(32, 784)  # batch_size=32, 每张图像展平为 784 维
output = model(x)          # 等价于 model.forward(x)，但优先使用前者
print(output.shape)        # torch.Size([32, 10])
```

### 18.1.4 嵌套 Module

`nn.Module` 支持任意嵌套，子模块的参数会被自动纳入父模块管理：

```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 64)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(64, 256)
        self.layer2 = nn.Linear(256, 784)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

ae = Autoencoder()
print(ae)
# 所有子模块参数都被自动追踪
total_params = sum(p.numel() for p in ae.parameters())
print(f"总参数量: {total_params:,}")
```

---

## 18.2 常用层

### 18.2.1 线性层（nn.Linear）

全连接层是最基本的网络层，实现 `y = xW^T + b`。

```python
# nn.Linear(in_features, out_features, bias=True)
linear = nn.Linear(128, 64)
print(f"权重形状: {linear.weight.shape}")  # torch.Size([64, 128])
print(f"偏置形状: {linear.bias.shape}")    # torch.Size([64])

x = torch.randn(32, 128)
y = linear(x)
print(f"输出形状: {y.shape}")              # torch.Size([32, 64])
```

**无偏置的线性层：**

```python
linear_no_bias = nn.Linear(128, 64, bias=False)
```

### 18.2.2 卷积层（nn.Conv2d）

卷积层用于处理图像等具有空间结构的数据。

```python
# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
conv = nn.Conv2d(
    in_channels=3,    # 输入通道数（RGB图像为3）
    out_channels=32,  # 输出通道数（卷积核数量）
    kernel_size=3,    # 卷积核大小（3x3）
    stride=1,         # 步长
    padding=1         # 填充（保持尺寸不变）
)

# 输入：batch=8, 3通道, 32x32图像
x = torch.randn(8, 3, 32, 32)
y = conv(x)
print(f"输出形状: {y.shape}")  # torch.Size([8, 32, 32, 32])

# 验证输出尺寸公式：(H + 2*padding - kernel_size) / stride + 1
H_out = (32 + 2*1 - 3) // 1 + 1
print(f"预计输出高度: {H_out}")  # 32（填充保持尺寸）
```

**常用卷积配置：**

```python
# 1x1 卷积（用于通道压缩/扩展，计算高效）
conv1x1 = nn.Conv2d(64, 32, kernel_size=1)

# 深度可分离卷积（groups=in_channels，每个通道独立卷积，参数少）
depthwise = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
pointwise = nn.Conv2d(32, 64, kernel_size=1)

# 转置卷积（上采样，常用于解码器/生成模型）
deconv = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # 尺寸翻倍
```

### 18.2.3 批归一化（nn.BatchNorm2d）

批归一化（Batch Normalization）在每个 mini-batch 上对特征进行归一化，有助于加速训练、缓解梯度问题，并具有一定正则化效果。

```python
# 对卷积层输出使用 BatchNorm2d（对应4D张量: N, C, H, W）
bn = nn.BatchNorm2d(num_features=32)  # num_features = 通道数

x = torch.randn(8, 32, 16, 16)
y = bn(x)
print(f"输出形状: {y.shape}")  # torch.Size([8, 32, 16, 16])

# 对全连接层输出使用 BatchNorm1d（对应2D张量: N, C）
bn1d = nn.BatchNorm1d(num_features=256)
x_fc = torch.randn(32, 256)
y_fc = bn1d(x_fc)
```

**BatchNorm 在训练和推理时行为不同：**

```python
model.train()   # 训练模式：使用当前 batch 的均值和方差
model.eval()    # 推理模式：使用训练过程中累积的均值和方差（running_mean/var）
```

**典型用法（卷积-BN-激活 结构）：**

```python
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)  # bias=False，因为 BN 有自己的偏移参数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

### 18.2.4 Dropout

Dropout 在训练时以概率 `p` 随机将神经元置零，以防止过拟合。推理时自动关闭。

```python
# 全连接网络中的 Dropout
dropout = nn.Dropout(p=0.5)   # 每个元素有50%概率被置零

# 卷积网络中使用 Dropout2d（整个通道被置零）
dropout2d = nn.Dropout2d(p=0.2)

# 示例
x = torch.ones(4, 10)
model_with_dropout = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(10, 2)
)

model_with_dropout.train()
out_train = model_with_dropout(x)  # Dropout 生效

model_with_dropout.eval()
out_eval = model_with_dropout(x)   # Dropout 不生效，输出确定性结果
```

---

## 18.3 激活函数

激活函数引入非线性，使网络能够拟合复杂函数。

### 18.3.1 ReLU 及其变体

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.linspace(-3, 3, 100)

# ReLU: max(0, x)
relu = nn.ReLU()
print(relu(torch.tensor([-1.0, 0.0, 1.0])))  # tensor([0., 0., 1.])

# Leaky ReLU: x if x>0, else negative_slope*x（解决"死亡ReLU"问题）
leaky_relu = nn.LeakyReLU(negative_slope=0.01)

# ELU: x if x>0, else alpha*(exp(x)-1)（更平滑的负值响应）
elu = nn.ELU(alpha=1.0)

# GELU: x * Φ(x)，在 Transformer 中广泛使用
gelu = nn.GELU()

# inplace=True 节省内存（直接修改输入张量，但不可求导记录）
relu_inplace = nn.ReLU(inplace=True)

# 在模块中直接使用函数形式
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))       # 函数形式（无状态，推荐用于激活）
        return self.fc2(x)
```

### 18.3.2 Sigmoid

输出范围 (0, 1)，常用于二分类输出层或门控机制（如 LSTM）。

```python
sigmoid = nn.Sigmoid()
x = torch.tensor([-2.0, 0.0, 2.0])
print(sigmoid(x))  # tensor([0.1192, 0.5000, 0.8808])

# 注意：深层网络中不建议在隐藏层使用 Sigmoid（梯度消失问题）
# 二分类输出层示例：
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出0~1的概率
        )

    def forward(self, x):
        return self.layers(x)
```

### 18.3.3 Tanh

输出范围 (-1, 1)，零中心化，在 RNN/LSTM 中常用。

```python
tanh = nn.Tanh()
x = torch.tensor([-2.0, 0.0, 2.0])
print(tanh(x))  # tensor([-0.9640, 0.0000,  0.9640])

# Tanh 比 Sigmoid 零中心化，在隐藏层中效果略好于 Sigmoid
# 但在深层网络中同样存在梯度消失问题
```

### 18.3.4 Softmax

将一组实数转换为概率分布（所有值之和为1），用于多分类任务的输出层。

```python
softmax = nn.Softmax(dim=1)  # dim=1 对每个样本的类别维度归一化
x = torch.tensor([[1.0, 2.0, 3.0],
                  [1.0, 1.0, 1.0]])
probs = softmax(x)
print(probs)
# tensor([[0.0900, 0.2447, 0.6652],
#         [0.3333, 0.3333, 0.3333]])
print(probs.sum(dim=1))  # tensor([1., 1.])

# 实践注意：在使用 nn.CrossEntropyLoss 时，不要在最后一层加 Softmax，
# 因为 CrossEntropyLoss 内部已包含 log_softmax 操作
class MultiClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        # 不加 Softmax，搭配 nn.CrossEntropyLoss 使用

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # 返回 logits（未归一化的分数）
```

### 18.3.5 激活函数对比

| 激活函数 | 输出范围 | 优点 | 缺点 | 适用场景 |
|---------|---------|------|------|---------|
| ReLU | [0, +∞) | 计算简单，缓解梯度消失 | 可能出现"死亡神经元" | 隐藏层首选 |
| Leaky ReLU | (-∞, +∞) | 解决死亡神经元 | 负斜率需调参 | 隐藏层，GAN |
| GELU | (-∞, +∞) | 平滑，性能好 | 计算稍慢 | Transformer |
| Sigmoid | (0, 1) | 概率解释 | 梯度消失，非零中心 | 二分类输出层 |
| Tanh | (-1, 1) | 零中心 | 梯度消失 | RNN/LSTM |
| Softmax | (0, 1)，和为1 | 概率分布 | 数值不稳定 | 多分类输出层 |

---

## 18.4 容器

PyTorch 提供了多种容器来组织子模块，使网络结构更清晰、灵活。

### 18.4.1 nn.Sequential

`Sequential` 按顺序依次执行各层，适合线性堆叠的网络。

```python
# 方式一：传入有序的层列表
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

x = torch.randn(32, 784)
output = model(x)
print(output.shape)  # torch.Size([32, 10])

# 方式二：使用 OrderedDict 为每层命名（便于按名称访问）
from collections import OrderedDict

model_named = nn.Sequential(OrderedDict([
    ('fc1',    nn.Linear(784, 256)),
    ('relu1',  nn.ReLU()),
    ('drop1',  nn.Dropout(0.3)),
    ('fc2',    nn.Linear(256, 10)),
]))

print(model_named.fc1)   # 按名称访问
print(model_named[0])    # 按索引访问

# 动态构建 Sequential
layers = []
dims = [784, 512, 256, 128, 10]
for i in range(len(dims) - 1):
    layers.append(nn.Linear(dims[i], dims[i+1]))
    if i < len(dims) - 2:
        layers.append(nn.ReLU())

dynamic_model = nn.Sequential(*layers)
print(dynamic_model)
```

### 18.4.2 nn.ModuleList

`ModuleList` 以列表形式持有一组子模块，自动注册其参数，适合需要在 `forward` 中手动控制执行逻辑的场景。

```python
class ResNet(nn.Module):
    """模拟一个具有多个残差块的网络"""
    def __init__(self, num_blocks, hidden_size):
        super().__init__()
        self.input_proj = nn.Linear(784, hidden_size)
        # 使用 ModuleList 存储多个相同结构的层
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = F.relu(self.input_proj(x))
        for block in self.blocks:   # 手动遍历，灵活控制
            residual = x
            x = F.relu(block(x))
            x = x + residual        # 残差连接
        return self.output_proj(x)

model = ResNet(num_blocks=4, hidden_size=256)
x = torch.randn(16, 784)
print(model(x).shape)  # torch.Size([16, 10])

# 错误示范：普通 Python list 不会注册参数！
class WrongModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10, 10) for _ in range(3)]  # 参数不被追踪！

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

wrong = WrongModel()
print(len(list(wrong.parameters())))  # 0 ← 参数未被追踪！
```

### 18.4.3 nn.ModuleDict

`ModuleDict` 以字典形式持有子模块，适合根据条件选择不同的网络分支。

```python
class MultiTaskModel(nn.Module):
    """多任务学习模型：共享主干 + 多个任务头"""
    def __init__(self):
        super().__init__()
        # 共享特征提取主干
        self.backbone = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # 使用 ModuleDict 存储多个任务头
        self.task_heads = nn.ModuleDict({
            'classification': nn.Linear(128, 10),
            'regression':     nn.Linear(128, 1),
            'embedding':      nn.Linear(128, 32),
        })

    def forward(self, x, task: str):
        features = self.backbone(x)
        return self.task_heads[task](features)  # 根据任务名选择对应头

model = MultiTaskModel()
x = torch.randn(16, 784)

cls_out = model(x, 'classification')  # torch.Size([16, 10])
reg_out = model(x, 'regression')      # torch.Size([16, 1])
emb_out = model(x, 'embedding')       # torch.Size([16, 32])

print(cls_out.shape, reg_out.shape, emb_out.shape)
```

### 18.4.4 三种容器对比

| 容器 | 数据结构 | 参数注册 | 适用场景 |
|------|---------|---------|---------|
| `Sequential` | 有序列表 | 自动 | 线性堆叠，无分支 |
| `ModuleList` | 列表 | 自动 | 需手动控制执行顺序，循环结构 |
| `ModuleDict` | 字典 | 自动 | 条件分支，多任务，按名选择 |

---

## 18.5 参数管理

### 18.5.1 查看所有参数

```python
model = SimpleNet(784, 256, 10)

# parameters()：返回所有可学习参数的迭代器
total = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total:,}")

# 只统计可训练参数（排除 requires_grad=False 的参数）
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数量: {trainable:,}")

# named_parameters()：同时返回参数名和参数值
for name, param in model.named_parameters():
    print(f"{name:20s} | 形状: {str(param.shape):25s} | 需要梯度: {param.requires_grad}")
```

输出示例：
```
fc1.weight           | 形状: torch.Size([256, 784])    | 需要梯度: True
fc1.bias             | 形状: torch.Size([256])          | 需要梯度: True
fc2.weight           | 形状: torch.Size([10, 256])     | 需要梯度: True
fc2.bias             | 形状: torch.Size([10])           | 需要梯度: True
```

### 18.5.2 冻结参数（迁移学习常用）

```python
# 冻结特定层的参数（不参与反向传播更新）
for param in model.fc1.parameters():
    param.requires_grad = False

# 验证
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# 优化器只更新 requires_grad=True 的参数
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
```

### 18.5.3 state_dict：保存与加载模型

`state_dict` 是一个有序字典，包含所有层的参数（权重和偏置）及 BatchNorm 的运行统计量。

```python
# 查看 state_dict
state = model.state_dict()
for key, value in state.items():
    print(f"{key}: {value.shape}")

# 保存模型参数（推荐方式）
torch.save(model.state_dict(), 'model_weights.pth')

# 加载模型参数
loaded_model = SimpleNet(784, 256, 10)  # 先创建同结构模型
loaded_model.load_state_dict(torch.load('model_weights.pth'))
loaded_model.eval()  # 推理前切换到 eval 模式

# 跨设备加载（GPU 模型 → CPU）
loaded_model.load_state_dict(
    torch.load('model_weights.pth', map_location='cpu')
)
```

**保存完整模型（含结构，不推荐，依赖类定义）：**

```python
# 不推荐：序列化整个模型对象（依赖 pickle，移植性差）
torch.save(model, 'full_model.pth')
full_model = torch.load('full_model.pth')

# 推荐方式：保存 state_dict，加载时重建结构
torch.save(model.state_dict(), 'weights_only.pth')
```

### 18.5.4 自定义参数：nn.Parameter

当需要将一个张量作为可学习参数（而非子模块的参数）时，使用 `nn.Parameter`。

```python
class AttentionLayer(nn.Module):
    """带可学习温度参数的注意力层"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        # 自定义可学习参数
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        return F.softmax(scores, dim=-1)

attn = AttentionLayer(64)
print(list(attn.named_parameters()))
# temperature 会出现在参数列表中，并参与梯度更新
```

### 18.5.5 模块遍历

```python
# children()：只遍历直接子模块
for name, child in model.named_children():
    print(f"直接子模块: {name}")

# modules()：递归遍历所有子模块（含自身）
for name, module in model.named_modules():
    print(f"模块: {name} | 类型: {type(module).__name__}")

# 实用：批量替换激活函数
def replace_relu_with_gelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.GELU())
        else:
            replace_relu_with_gelu(child)

replace_relu_with_gelu(model)
```

---

## 本章小结

| 知识点 | 核心内容 | 关键 API |
|--------|---------|---------|
| nn.Module | 所有网络的基类，管理参数和子模块 | `__init__`, `forward`, `to()`, `train()`, `eval()` |
| Linear | 全连接层：y = xW^T + b | `nn.Linear(in, out)` |
| Conv2d | 二维卷积，提取空间特征 | `nn.Conv2d(in_ch, out_ch, kernel_size)` |
| BatchNorm | 批归一化，加速训练 | `nn.BatchNorm2d(C)`, `nn.BatchNorm1d(C)` |
| Dropout | 随机置零，防止过拟合 | `nn.Dropout(p)` |
| ReLU / GELU | 最常用的隐藏层激活 | `nn.ReLU()`, `nn.GELU()`, `F.relu()` |
| Sigmoid / Softmax | 输出层概率激活 | `nn.Sigmoid()`, `nn.Softmax(dim)` |
| Sequential | 线性堆叠容器 | `nn.Sequential(*layers)` |
| ModuleList | 列表容器，手动控制执行 | `nn.ModuleList([...])` |
| ModuleDict | 字典容器，条件选择 | `nn.ModuleDict({...})` |
| parameters() | 获取可学习参数迭代器 | `model.parameters()` |
| state_dict | 模型序列化 | `model.state_dict()`, `load_state_dict()` |

---

## 深度学习应用：模型架构设计

本节展示如何利用本章知识，从零搭建一个完整的 **CNN 图像分类器** 和 **MLP 分类器**，并包含完整的训练流程框架。

### 完整 CNN：CIFAR-10 图像分类

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """基础卷积模块：Conv -> BN -> ReLU（可选池化）"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class CIFAR10CNN(nn.Module):
    """
    用于 CIFAR-10 的 CNN 分类器
    输入: (N, 3, 32, 32)
    输出: (N, 10)
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__()

        # 特征提取主干
        self.features = nn.Sequential(
            # Block 1: 3 → 32, 32x32 → 16x16
            ConvBlock(3,  32, pool=True),
            # Block 2: 32 → 64, 16x16 → 8x8
            ConvBlock(32, 64, pool=True),
            # Block 3: 64 → 128, 8x8 → 4x4
            ConvBlock(64, 128, pool=True),
            # Block 4: 128 → 256, 4x4（不池化）
            ConvBlock(128, 256, pool=False),
        )

        # 自适应平均池化，输出固定 2x2 大小（解耦输入尺寸与分类头）
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 256 * 2 * 2 = 1024
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, num_classes),
        )

        # 参数初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """He 初始化（适合 ReLU 激活）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)


# --- 验证模型 ---
model = CIFAR10CNN(num_classes=10)
print(model)

x = torch.randn(8, 3, 32, 32)
out = model(x)
print(f"\n输入形状: {x.shape}")
print(f"输出形状: {out.shape}")  # (8, 10)

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")


# --- 完整训练循环框架 ---
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)          # 前向传播
        loss = criterion(outputs, labels)
        loss.backward()                  # 反向传播
        optimizer.step()                 # 参数更新

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


# 使用示例（假设已有 train_loader, val_loader）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CIFAR10CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# for epoch in range(50):
#     train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
#     val_loss, val_acc = evaluate(model, val_loader, criterion, device)
#     scheduler.step()
#     print(f"Epoch {epoch+1}: "
#           f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
#           f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
```

### 完整 MLP：通用分类器

```python
class MLP(nn.Module):
    """
    可配置的多层感知机（MLP）
    支持任意深度和宽度，带 BN 和 Dropout
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        # 激活函数映射
        activation_map = {
            'relu':  nn.ReLU,
            'gelu':  nn.GELU,
            'tanh':  nn.Tanh,
            'leaky': lambda: nn.LeakyReLU(0.01),
        }
        act_fn = activation_map[activation]

        # 动态构建层
        layers = []
        in_dim = input_dim

        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(act_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        """返回各类别概率（推理时使用）"""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return F.softmax(logits, dim=1)


# 多种配置示例
# 浅层宽网络
mlp_wide = MLP(
    input_dim=784,
    hidden_dims=[1024, 1024],
    output_dim=10,
    dropout_rate=0.5,
    use_batchnorm=True
)

# 深层窄网络
mlp_deep = MLP(
    input_dim=784,
    hidden_dims=[256, 256, 256, 256, 256],
    output_dim=10,
    activation='gelu',
    dropout_rate=0.3,
    use_batchnorm=True
)

x = torch.randn(32, 784)
print("Wide MLP output:", mlp_wide(x).shape)   # (32, 10)
print("Deep MLP output:", mlp_deep(x).shape)   # (32, 10)

# 查看参数量对比
for name, m in [('Wide', mlp_wide), ('Deep', mlp_deep)]:
    n = sum(p.numel() for p in m.parameters())
    print(f"{name} MLP 参数量: {n:,}")
```

### 模型检查点保存（Best Model）

```python
class ModelCheckpoint:
    """保存验证集上最佳模型的工具类"""
    def __init__(self, save_path: str, monitor: str = 'val_loss',
                 mode: str = 'min'):
        self.save_path = save_path
        self.monitor = monitor
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.mode = mode

    def step(self, model, current_value: float) -> bool:
        improved = (
            current_value < self.best_value if self.mode == 'min'
            else current_value > self.best_value
        )
        if improved:
            self.best_value = current_value
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_value': self.best_value,
            }, self.save_path)
            return True
        return False


# 使用示例
checkpoint = ModelCheckpoint('best_model.pth', monitor='val_acc', mode='max')
# for epoch in ...:
#     ...
#     if checkpoint.step(model, val_acc):
#         print(f"  -> 保存最佳模型 (val_acc={val_acc:.4f})")
```

---

## 练习题

### 基础题

**题目 1（基础）：** 定义一个名为 `TwoLayerNet` 的神经网络，包含两个线性层，中间使用 ReLU 激活函数。输入维度为 128，隐藏层维度为 64，输出维度为 4。要求：
- 使用 `nn.Module` 继承
- 打印模型结构
- 计算并打印总参数量

**题目 2（基础）：** 使用 `nn.Sequential` 构建一个具有如下结构的网络，并用随机输入验证输出形状：
```
输入 (batch=16, 3, 28, 28)
→ Conv2d(3, 16, 3, padding=1) → BN → ReLU → MaxPool2d(2)
→ Conv2d(16, 32, 3, padding=1) → BN → ReLU → MaxPool2d(2)
→ Flatten
→ Linear(32*7*7, 128) → ReLU → Dropout(0.5)
→ Linear(128, 10)
```

### 中级题

**题目 3（中级）：** 实现一个 `MultiScaleConv` 模块，同时使用 3×3、5×5、1×1 三种卷积核对输入进行卷积，并将三路输出在通道维度上拼接（类似 Inception 模块）。输入通道数为 `in_ch`，每路输出通道数为 `out_ch`，最终输出通道数为 `3 * out_ch`。

**题目 4（中级）：** 实现一个支持"冻结主干"的迁移学习框架。要求：
1. 创建一个包含 `backbone`（4层Linear+ReLU）和 `head`（2层Linear）的模型
2. 实现 `freeze_backbone()` 方法冻结主干参数
3. 实现 `unfreeze_backbone()` 方法解冻主干参数
4. 验证冻结/解冻后可训练参数量的变化

### 进阶题

**题目 5（进阶）：** 实现一个 **残差网络（ResNet）块** 并组建一个小型 ResNet：
1. 实现 `BasicBlock(in_ch, out_ch)`：包含两个卷积层（3×3, padding=1）+ BN + ReLU，以及当输入输出通道不同时的 `shortcut`（1×1 卷积 + BN）
2. 使用 4 个 `BasicBlock` 组建 `MiniResNet`，通道数为 [3, 16, 32, 64, 128]
3. 在 `MiniResNet` 末尾加入全局平均池化和全连接层，完成 10 类分类
4. 验证输入 `(4, 3, 32, 32)` 时输出形状正确，并打印参数量

---

## 练习答案

### 答案 1

```python
import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = TwoLayerNet()
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
# fc1: 128*64 + 64 = 8256
# fc2: 64*4 + 4   = 260
# 总计: 8516
```

### 答案 2

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),                  # 28x28 → 14x14

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),                  # 14x14 → 7x7

    nn.Flatten(),                     # 32 * 7 * 7 = 1568
    nn.Linear(32 * 7 * 7, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(128, 10),
)

x = torch.randn(16, 3, 28, 28)
out = model(x)
print(f"输出形状: {out.shape}")  # torch.Size([16, 10])
assert out.shape == (16, 10), "形状不匹配！"
print("形状验证通过！")
```

### 答案 3

```python
import torch
import torch.nn as nn

class MultiScaleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b3 = self.branch3x3(x)
        b5 = self.branch5x5(x)
        return torch.cat([b1, b3, b5], dim=1)  # 拼接通道维度

# 验证
ms = MultiScaleConv(in_ch=32, out_ch=16)
x = torch.randn(4, 32, 16, 16)
out = ms(x)
print(f"输出形状: {out.shape}")  # (4, 48, 16, 16)
assert out.shape == (4, 16*3, 16, 16)
```

### 答案 4

```python
import torch
import torch.nn as nn

class TransferModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256),       nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.head(self.backbone(x))


model = TransferModel(input_dim=784, num_classes=10)
print(f"初始可训练参数: {model.count_trainable():,}")

model.freeze_backbone()
print(f"冻结主干后可训练参数: {model.count_trainable():,}")

model.unfreeze_backbone()
print(f"解冻主干后可训练参数: {model.count_trainable():,}")

# 验证前向传播
x = torch.randn(8, 784)
out = model(x)
print(f"输出形状: {out.shape}")  # (8, 10)
```

### 答案 5

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """ResNet 基础残差块"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # 当通道数不同时，使用 1x1 卷积对齐 shortcut
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)  # 残差相加后激活


class MiniResNet(nn.Module):
    """
    小型 ResNet：用于 32x32 图像的 10 分类
    通道数: 3 → 16 → 32 → 64 → 128
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = BasicBlock(16, 16)
        self.layer2 = BasicBlock(16, 32)   # 通道数变化，触发 shortcut
        self.layer3 = BasicBlock(32, 64)
        self.layer4 = BasicBlock(64, 128)

        # 全局平均池化 + 分类头
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)  # (N, 128, 1, 1)
        x = x.view(x.size(0), -1)   # (N, 128)
        return self.fc(x)


# 验证
model = MiniResNet(num_classes=10)
x = torch.randn(4, 3, 32, 32)
out = model(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {out.shape}")  # (4, 10)
assert out.shape == (4, 10), "形状错误！"

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# 测试残差连接：确保梯度能正常流过
loss = out.sum()
loss.backward()
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"警告：{name} 没有梯度！")
        break
else:
    print("所有参数梯度正常！")
```
