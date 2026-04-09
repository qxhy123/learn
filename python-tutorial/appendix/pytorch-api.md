# 附录B：PyTorch常用API

本附录提供PyTorch核心API的快速参考手册，涵盖张量操作、自动微分、神经网络模块、损失函数、优化器和数据加载等内容。

---

## B.1 张量操作

### B.1.1 创建张量

| 函数 | 说明 | 示例 |
|------|------|------|
| `torch.tensor(data)` | 从数据创建张量 | `torch.tensor([1, 2, 3])` |
| `torch.zeros(shape)` | 创建全零张量 | `torch.zeros(3, 4)` |
| `torch.ones(shape)` | 创建全一张量 | `torch.ones(2, 3)` |
| `torch.rand(shape)` | 均匀分布随机张量 [0,1) | `torch.rand(3, 3)` |
| `torch.randn(shape)` | 标准正态分布随机张量 | `torch.randn(2, 4)` |
| `torch.arange(start, end, step)` | 创建等差序列张量 | `torch.arange(0, 10, 2)` |
| `torch.linspace(start, end, steps)` | 创建均匀间隔张量 | `torch.linspace(0, 1, 5)` |
| `torch.empty(shape)` | 创建未初始化张量 | `torch.empty(2, 3)` |
| `torch.eye(n)` | 创建单位矩阵 | `torch.eye(4)` |
| `torch.full(shape, val)` | 创建填充指定值的张量 | `torch.full((2, 3), 7)` |

```python
import torch

# 从列表创建
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 创建特定形状的零张量
zeros = torch.zeros(3, 4)          # shape: (3, 4)

# 创建随机张量
rand_t = torch.rand(2, 3)          # 均匀分布 [0, 1)
randn_t = torch.randn(2, 3)        # 标准正态分布

# 等差序列
seq = torch.arange(0, 10, 2)       # tensor([0, 2, 4, 6, 8])

# 均匀间隔
lin = torch.linspace(0.0, 1.0, 5)  # tensor([0.00, 0.25, 0.50, 0.75, 1.00])

# 类似已有张量的形状和dtype
x_like = torch.zeros_like(x)       # 与 x 形状相同的全零张量
r_like = torch.rand_like(x)        # 与 x 形状相同的随机张量
```

### B.1.2 张量属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `.shape` / `.size()` | `torch.Size` | 张量的形状（各维度大小） |
| `.dtype` | `torch.dtype` | 张量的数据类型 |
| `.device` | `torch.device` | 张量所在设备（cpu / cuda） |
| `.ndim` | `int` | 张量的维度数 |
| `.numel()` | `int` | 张量元素总数 |
| `.requires_grad` | `bool` | 是否参与梯度计算 |
| `.is_cuda` | `bool` | 是否在GPU上 |

```python
x = torch.randn(3, 4, dtype=torch.float32)

print(x.shape)        # torch.Size([3, 4])
print(x.dtype)        # torch.float32
print(x.device)       # device(type='cpu')
print(x.ndim)         # 2
print(x.numel())      # 12

# 常用数据类型
# torch.float32 / torch.float  —— 单精度浮点（默认）
# torch.float64 / torch.double —— 双精度浮点
# torch.int32   / torch.int    —— 32位整型
# torch.int64   / torch.long   —— 64位整型（索引常用）
# torch.bool                   —— 布尔型

# 类型转换
x_int = x.to(torch.int32)
x_gpu = x.to('cuda')          # 移动到GPU（需要CUDA环境）
x_cpu = x_gpu.cpu()           # 移动回CPU
```

### B.1.3 形状操作

| 函数 | 说明 | 备注 |
|------|------|------|
| `.reshape(shape)` | 改变形状，返回新张量 | 可能复制数据 |
| `.view(shape)` | 改变形状，返回视图 | 要求内存连续 |
| `.squeeze(dim)` | 移除大小为1的维度 | `dim` 可选 |
| `.unsqueeze(dim)` | 在指定位置插入大小为1的维度 | |
| `.permute(dims)` | 任意维度重排 | |
| `.transpose(dim0, dim1)` | 交换两个维度 | |
| `.contiguous()` | 使张量内存连续 | `view` 前常用 |
| `.flatten(start, end)` | 将指定维度范围展平 | |
| `torch.cat(tensors, dim)` | 沿指定维度拼接 | |
| `torch.stack(tensors, dim)` | 创建新维度并堆叠 | |

```python
x = torch.randn(2, 3, 4)

# reshape / view
y = x.reshape(6, 4)            # shape: (6, 4)
z = x.view(2, 12)              # shape: (2, 12)，要求内存连续

# squeeze / unsqueeze
a = torch.randn(1, 3, 1, 4)
a_sq = a.squeeze()             # shape: (3, 4)  移除所有大小为1的维度
a_sq1 = a.squeeze(0)           # shape: (3, 1, 4)
b = torch.randn(3, 4)
b_us = b.unsqueeze(0)          # shape: (1, 3, 4)
b_us2 = b.unsqueeze(-1)        # shape: (3, 4, 1)

# permute / transpose
x = torch.randn(2, 3, 4)
xp = x.permute(2, 0, 1)       # shape: (4, 2, 3)
xt = x.transpose(1, 2)        # shape: (2, 4, 3)，交换维度1和2

# flatten
xf = x.flatten(1)             # shape: (2, 12)，从第1维开始展平

# cat / stack
a = torch.randn(2, 3)
b = torch.randn(2, 3)
cat_0 = torch.cat([a, b], dim=0)    # shape: (4, 3)
cat_1 = torch.cat([a, b], dim=1)    # shape: (2, 6)
stk   = torch.stack([a, b], dim=0)  # shape: (2, 2, 3)
```

### B.1.4 索引与切片

```python
x = torch.arange(24).reshape(2, 3, 4)

# 基本索引
x[0]           # 第0个样本，shape: (3, 4)
x[0, 1]        # 第0个样本的第1行，shape: (4,)
x[0, 1, 2]     # 标量元素

# 切片
x[:, 1:3, :]   # 所有样本，第1到2行，所有列，shape: (2, 2, 4)
x[..., -1]     # 最后一列，shape: (2, 3)

# 布尔索引
mask = x > 10
x[mask]        # 所有大于10的元素，返回1D张量

# 高级索引（gather / index_select）
idx = torch.tensor([0, 2])
x.index_select(1, idx)        # 选取第1维的第0和第2个，shape: (2, 2, 4)

# gather：按索引收集元素
scores = torch.randn(4, 10)   # batch=4, num_classes=10
labels = torch.tensor([2, 5, 0, 9]).unsqueeze(1)  # shape: (4, 1)
selected = scores.gather(1, labels)               # shape: (4, 1)
```

---

## B.2 数学运算

### B.2.1 基本运算

| 函数 / 运算符 | 说明 |
|---------------|------|
| `torch.add(a, b)` / `a + b` | 逐元素加法 |
| `torch.sub(a, b)` / `a - b` | 逐元素减法 |
| `torch.mul(a, b)` / `a * b` | 逐元素乘法（Hadamard积） |
| `torch.div(a, b)` / `a / b` | 逐元素除法 |
| `torch.neg(a)` / `-a` | 取反 |
| `torch.abs(a)` | 绝对值 |
| `torch.clamp(a, min, max)` | 将值限制在 [min, max] 范围 |

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)                   # tensor([5., 7., 9.])
print(torch.add(a, b))         # 等价于上一行
print(a * b)                   # tensor([ 4., 10., 18.])
print(a / b)                   # tensor([0.25, 0.40, 0.50])

# 原地操作（in-place，以 _ 结尾）
a.add_(1.0)                    # a 的每个元素加1，原地修改
a.mul_(2.0)                    # a 的每个元素乘2，原地修改

# clamp 限制范围
x = torch.randn(5)
x_clamped = torch.clamp(x, min=-1.0, max=1.0)
```

### B.2.2 矩阵运算

| 函数 | 说明 | 输入形状 |
|------|------|----------|
| `torch.mm(A, B)` | 2D矩阵乘法 | `(n,k)` × `(k,m)` → `(n,m)` |
| `torch.bmm(A, B)` | 批量矩阵乘法 | `(b,n,k)` × `(b,k,m)` → `(b,n,m)` |
| `torch.matmul(A, B)` | 通用矩阵乘法（自动广播） | 支持多维 |
| `A @ B` | `matmul` 的运算符形式 | 同上 |
| `torch.dot(a, b)` | 1D向量点积 | `(n,)` × `(n,)` → 标量 |
| `A.T` / `.t()` | 转置（2D） | |
| `torch.linalg.inv(A)` | 矩阵求逆 | |
| `torch.linalg.det(A)` | 行列式 | |

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)

C = torch.mm(A, B)             # shape: (3, 5)
C = A @ B                      # 等价写法

# 批量矩阵乘法
batch_A = torch.randn(8, 3, 4)
batch_B = torch.randn(8, 4, 5)
batch_C = torch.bmm(batch_A, batch_B)   # shape: (8, 3, 5)

# matmul 自动处理广播
x = torch.randn(2, 3, 4)
w = torch.randn(4, 5)
out = torch.matmul(x, w)       # shape: (2, 3, 5)
```

### B.2.3 统计函数

| 函数 | 说明 | 备注 |
|------|------|------|
| `torch.mean(x, dim)` | 均值 | `dim` 可选，指定轴 |
| `torch.sum(x, dim)` | 求和 | |
| `torch.std(x, dim)` | 标准差 | |
| `torch.var(x, dim)` | 方差 | |
| `torch.max(x, dim)` | 最大值 | 返回值和索引 |
| `torch.min(x, dim)` | 最小值 | 返回值和索引 |
| `torch.argmax(x, dim)` | 最大值的索引 | |
| `torch.argmin(x, dim)` | 最小值的索引 | |
| `torch.median(x)` | 中位数 | |
| `torch.cumsum(x, dim)` | 累积求和 | |
| `torch.prod(x, dim)` | 连乘 | |

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

print(torch.mean(x))           # tensor(3.5000)  全局均值
print(torch.mean(x, dim=0))    # tensor([2.5, 3.5, 4.5])  列均值
print(torch.mean(x, dim=1))    # tensor([2., 5.])  行均值

# keepdim 保持维度
m = torch.mean(x, dim=1, keepdim=True)  # shape: (2, 1)

# max 返回 (values, indices)
vals, idx = torch.max(x, dim=1)
# vals: tensor([3., 6.]),  idx: tensor([2, 2])

# argmax 只返回索引
print(torch.argmax(x, dim=1))  # tensor([2, 2])
```

### B.2.4 数学函数

| 函数 | 说明 |
|------|------|
| `torch.exp(x)` | 指数函数 e^x |
| `torch.log(x)` | 自然对数 ln(x) |
| `torch.log2(x)` | 以2为底的对数 |
| `torch.log10(x)` | 以10为底的对数 |
| `torch.sqrt(x)` | 平方根 |
| `torch.pow(x, n)` / `x ** n` | 幂运算 |
| `torch.sin(x)` / `cos` / `tan` | 三角函数 |
| `torch.sigmoid(x)` | Sigmoid函数 |
| `torch.tanh(x)` | 双曲正切 |
| `torch.floor(x)` / `ceil` / `round` | 取整 |

```python
x = torch.tensor([1.0, 2.0, 4.0])

print(torch.exp(x))            # tensor([ 2.7183,  7.3891, 54.5982])
print(torch.log(x))            # tensor([0.0000, 0.6931, 1.3863])
print(torch.sqrt(x))           # tensor([1.0000, 1.4142, 2.0000])
print(torch.pow(x, 2))         # tensor([ 1.,  4., 16.])
print(x ** 2)                  # 等价写法
```

---

## B.3 自动微分

PyTorch的自动微分引擎（`autograd`）通过动态计算图实现反向传播。

### B.3.1 核心概念

| 概念 | 说明 |
|------|------|
| `requires_grad=True` | 标记张量参与梯度计算 |
| `.backward()` | 对标量张量执行反向传播 |
| `.grad` | 存储梯度的属性（首次backward后可用） |
| `torch.no_grad()` | 上下文管理器，禁用梯度追踪 |
| `.detach()` | 返回从计算图中分离的张量 |
| `.grad_fn` | 创建该张量的操作函数（叶节点为None） |
| `.is_leaf` | 是否为叶节点张量 |

```python
import torch

# 创建需要梯度的张量
x = torch.tensor(3.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 前向计算
y = w * x + b          # y = 2*3 + 1 = 7.0

# 反向传播
y.backward()

# 查看梯度
print(x.grad)          # dy/dx = w = 2.0
print(w.grad)          # dy/dw = x = 3.0
print(b.grad)          # dy/db = 1.0
```

### B.3.2 多步反向传播与梯度累积

```python
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()

y.backward()
print(x.grad)          # 2 * x

# 注意：梯度默认累积，训练时需在每次 backward 前清零
x.grad.zero_()         # 手动清零梯度
```

### B.3.3 no_grad 与 detach

```python
x = torch.randn(3, requires_grad=True)

# 方式1：上下文管理器（推理时常用）
with torch.no_grad():
    y = x * 2          # y 不追踪梯度
    print(y.requires_grad)   # False

# 方式2：detach（从计算图中分离，共享数据）
z = x.detach()
print(z.requires_grad)       # False

# 方式3：装饰器形式
@torch.no_grad()
def inference(model, x):
    return model(x)
```

### B.3.4 非标量的 backward

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2             # shape: (3,)

# 非标量需要传入 gradient 参数（与输出形状相同）
y.backward(torch.ones_like(y))   # 等价于 y.sum().backward()
print(x.grad)          # 2 * x
```

---

## B.4 神经网络模块（torch.nn）

所有网络层继承自 `nn.Module`，核心方法是 `forward()`。

### B.4.1 线性层

```python
import torch.nn as nn

# nn.Linear(in_features, out_features, bias=True)
linear = nn.Linear(128, 64)
x = torch.randn(32, 128)    # batch_size=32, input_dim=128
out = linear(x)             # shape: (32, 64)

print(linear.weight.shape)  # (64, 128)
print(linear.bias.shape)    # (64,)
```

### B.4.2 卷积层

| 层 | 说明 | 主要参数 |
|----|------|----------|
| `nn.Conv1d` | 1D卷积（序列/文本） | `in_channels, out_channels, kernel_size` |
| `nn.Conv2d` | 2D卷积（图像） | 同上，另有 `stride, padding, dilation, groups` |
| `nn.Conv3d` | 3D卷积（视频/体积） | 同上 |
| `nn.ConvTranspose2d` | 转置卷积（上采样） | 同上 |

```python
# Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
x = torch.randn(8, 3, 32, 32)   # (batch, channels, H, W)
out = conv(x)                    # shape: (8, 64, 32, 32)

# Conv1d
conv1d = nn.Conv1d(16, 32, kernel_size=3, padding=1)
x1d = torch.randn(8, 16, 100)   # (batch, channels, length)
out1d = conv1d(x1d)             # shape: (8, 32, 100)

# 输出尺寸公式（Conv2d）：
# H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
```

### B.4.3 池化层

| 层 | 说明 | 参数 |
|----|------|------|
| `nn.MaxPool2d` | 最大池化 | `kernel_size, stride, padding` |
| `nn.AvgPool2d` | 平均池化 | 同上 |
| `nn.AdaptiveMaxPool2d` | 自适应最大池化 | `output_size` |
| `nn.AdaptiveAvgPool2d` | 自适应平均池化 | `output_size` |

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.randn(8, 64, 32, 32)
out = pool(x)                   # shape: (8, 64, 16, 16)

# 自适应池化（常用于全局池化）
gap = nn.AdaptiveAvgPool2d((1, 1))
out = gap(x)                    # shape: (8, 64, 1, 1)
out = out.flatten(1)            # shape: (8, 64)
```

### B.4.4 归一化层

| 层 | 说明 | 适用场景 |
|----|------|----------|
| `nn.BatchNorm1d(num_features)` | 批归一化（1D/全连接） | 全连接层后 |
| `nn.BatchNorm2d(num_features)` | 批归一化（2D/卷积） | 卷积层后 |
| `nn.LayerNorm(normalized_shape)` | 层归一化 | Transformer |
| `nn.GroupNorm(num_groups, num_channels)` | 组归一化 | 小批量场景 |
| `nn.InstanceNorm2d(num_features)` | 实例归一化 | 风格迁移 |

```python
# BatchNorm2d（跨batch在C维归一化）
bn = nn.BatchNorm2d(64)
x = torch.randn(8, 64, 32, 32)
out = bn(x)                     # shape: (8, 64, 32, 32)

# LayerNorm（在最后N维归一化，常用于Transformer）
ln = nn.LayerNorm(512)
x = torch.randn(32, 10, 512)   # (batch, seq_len, d_model)
out = ln(x)                     # shape: (32, 10, 512)
```

### B.4.5 Dropout

```python
# nn.Dropout(p=0.5)  — 2D全连接
# nn.Dropout2d(p=0.5) — 4D特征图（整通道置零）
dropout = nn.Dropout(p=0.5)
x = torch.randn(32, 128)
out = dropout(x)   # 训练时随机置零50%，推理时自动关闭

# 注意：model.eval() 会自动禁用 Dropout 和 BatchNorm 的训练行为
```

### B.4.6 激活函数

| 函数 | 说明 | 输出范围 |
|------|------|----------|
| `nn.ReLU()` | 修正线性单元 max(0,x) | [0, +∞) |
| `nn.LeakyReLU(negative_slope)` | 带泄漏的ReLU | (-∞, +∞) |
| `nn.PReLU()` | 可学习斜率的ReLU | (-∞, +∞) |
| `nn.ELU()` | 指数线性单元 | (-1, +∞) |
| `nn.GELU()` | 高斯误差线性单元（BERT/GPT用） | 近似(-∞, +∞) |
| `nn.Sigmoid()` | Sigmoid函数 1/(1+e^-x) | (0, 1) |
| `nn.Tanh()` | 双曲正切 | (-1, 1) |
| `nn.Softmax(dim)` | 归一化指数，输出概率分布 | (0, 1)，和为1 |
| `nn.LogSoftmax(dim)` | log(Softmax)，数值更稳定 | (-∞, 0) |

```python
relu     = nn.ReLU()
sigmoid  = nn.Sigmoid()
tanh     = nn.Tanh()
softmax  = nn.Softmax(dim=-1)
gelu     = nn.GELU()

x = torch.randn(4, 10)
probs = softmax(x)             # shape: (4, 10)，每行和为1

# 函数式写法（无需实例化）
import torch.nn.functional as F
y = F.relu(x)
y = F.gelu(x)
y = F.softmax(x, dim=-1)
```

### B.4.7 自定义 nn.Module

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(128, 256, 10)
x = torch.randn(32, 128)
logits = model(x)              # shape: (32, 10)

# 查看参数数量
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数: {total:,}  可训练: {trainable:,}")
```

---

## B.5 损失函数

### B.5.1 回归损失

| 损失 | 全称 | 说明 |
|------|------|------|
| `nn.MSELoss` | 均方误差 | L2损失，对大误差敏感 |
| `nn.L1Loss` | 平均绝对误差 | L1损失，对离群点鲁棒 |
| `nn.SmoothL1Loss` | Smooth L1（Huber Loss） | 结合MSE和L1优点 |

```python
mse  = nn.MSELoss()
mae  = nn.L1Loss()
huber = nn.SmoothL1Loss(beta=1.0)

pred   = torch.randn(8, 1)
target = torch.randn(8, 1)

loss_mse   = mse(pred, target)
loss_mae   = mae(pred, target)
loss_huber = huber(pred, target)
```

### B.5.2 分类损失

| 损失 | 说明 | 输入要求 |
|------|------|----------|
| `nn.CrossEntropyLoss` | 交叉熵（含Softmax） | logits `(N,C)` + 标签 `(N,)` |
| `nn.NLLLoss` | 负对数似然 | log概率 `(N,C)` + 标签 `(N,)` |
| `nn.BCELoss` | 二元交叉熵 | 概率 `(N,)` + 标签 `(N,)` |
| `nn.BCEWithLogitsLoss` | 含Sigmoid的BCE（数值更稳定） | logits `(N,)` + 标签 `(N,)` |

```python
# 多分类（最常用）
ce_loss = nn.CrossEntropyLoss()
logits = torch.randn(32, 10)    # (batch, num_classes)
labels = torch.randint(0, 10, (32,))  # (batch,)
loss = ce_loss(logits, labels)

# 类别权重（处理不均衡数据）
weights = torch.tensor([1.0, 2.0, 1.5, ...])  # 每个类别的权重
ce_loss_w = nn.CrossEntropyLoss(weight=weights)

# NLLLoss（通常与 LogSoftmax 配合）
nll_loss   = nn.NLLLoss()
log_probs  = F.log_softmax(logits, dim=1)
loss_nll   = nll_loss(log_probs, labels)

# 二分类
bce_logits = nn.BCEWithLogitsLoss()   # 推荐：数值稳定
logits_bin = torch.randn(32)
targets    = torch.randint(0, 2, (32,)).float()
loss_bce   = bce_logits(logits_bin, targets)

# 标签平滑（CrossEntropyLoss）
ce_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## B.6 优化器（torch.optim）

### B.6.1 常用优化器

| 优化器 | 说明 | 推荐场景 |
|--------|------|----------|
| `optim.SGD` | 随机梯度下降（支持动量/权重衰减） | CV任务 |
| `optim.Adam` | 自适应矩估计 | 通用，NLP首选 |
| `optim.AdamW` | Adam + 解耦权重衰减 | Transformer模型 |
| `optim.RMSprop` | 均方根传播 | RNN任务 |
| `optim.Adagrad` | 自适应学习率 | 稀疏数据 |

```python
import torch.optim as optim

model = MLP(128, 256, 10)

# SGD（含动量和权重衰减）
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)

# Adam
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0
)

# AdamW（Transformer 推荐）
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

# 不同参数组（不同学习率）
optimizer = optim.AdamW([
    {'params': model.net[0].parameters(), 'lr': 1e-4},
    {'params': model.net[-1].parameters(), 'lr': 1e-3},
], weight_decay=0.01)
```

### B.6.2 训练循环标准写法

```python
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()          # 1. 清零梯度
        logits = model(x_batch)        # 2. 前向传播
        loss = criterion(logits, y_batch)  # 3. 计算损失
        loss.backward()                # 4. 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 可选：梯度裁剪
        optimizer.step()               # 5. 更新参数
```

### B.6.3 学习率调度器

| 调度器 | 说明 | 关键参数 |
|--------|------|----------|
| `StepLR` | 每 N 步乘以 gamma | `step_size, gamma` |
| `MultiStepLR` | 在指定milestone乘以gamma | `milestones, gamma` |
| `ExponentialLR` | 每步乘以 gamma | `gamma` |
| `CosineAnnealingLR` | 余弦退火 | `T_max, eta_min` |
| `ReduceLROnPlateau` | 监控指标停滞时降低学习率 | `mode, factor, patience` |
| `OneCycleLR` | 单周期策略（Fast.ai）| `max_lr, total_steps` |
| `CosineAnnealingWarmRestarts` | 带热重启的余弦退火 | `T_0, T_mult` |

```python
from torch.optim import lr_scheduler

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 余弦退火（最常用）
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 在 plateau 时降低学习率
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# OneCycleLR（需要提前知道总步数）
scheduler = lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3,
    total_steps=num_epochs * len(train_loader)
)

# 调度器调用位置
for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)
    scheduler.step()                        # 大多数调度器
    # scheduler.step(val_loss)              # ReduceLROnPlateau 需要传入指标

# 查看当前学习率
current_lr = optimizer.param_groups[0]['lr']
```

---

## B.7 数据加载（torch.utils.data）

### B.7.1 Dataset

```python
from torch.utils.data import Dataset, DataLoader

# 自定义 Dataset 必须实现 __len__ 和 __getitem__
class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data          # 可以是 numpy array、列表等
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# 使用示例
import numpy as np
X = np.random.randn(1000, 128).astype(np.float32)
Y = np.random.randint(0, 10, 1000)

dataset = MyDataset(X, Y)
print(len(dataset))             # 1000
x, y = dataset[0]
```

### B.7.2 DataLoader

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `dataset` | `Dataset` | 数据集对象 | 必填 |
| `batch_size` | `int` | 每批样本数 | `1` |
| `shuffle` | `bool` | 是否随机打乱（训练集用True） | `False` |
| `num_workers` | `int` | 数据加载子进程数 | `0` |
| `pin_memory` | `bool` | 锁页内存，加速GPU传输 | `False` |
| `drop_last` | `bool` | 丢弃最后不完整的batch | `False` |
| `collate_fn` | `callable` | 自定义batch组装函数 | 默认 |
| `sampler` | `Sampler` | 自定义采样策略 | `None` |
| `prefetch_factor` | `int` | 每个worker预取批数 | `2` |

```python
# 训练集加载器
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,    # GPU训练时设为True
    drop_last=True
)

# 验证/测试集加载器
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 迭代使用
for batch_idx, (x, y) in enumerate(train_loader):
    x = x.to(device)
    y = y.to(device)
    # ... 训练逻辑
```

### B.7.3 内置数据集与拆分工具

```python
from torch.utils.data import random_split, Subset

# 随机拆分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

# 按索引取子集
indices = list(range(100))
subset  = Subset(dataset, indices)

# 使用 torchvision 内置数据集（以CIFAR-10为例）
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

cifar10 = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
loader  = DataLoader(cifar10, batch_size=64, shuffle=True, num_workers=4)
```

---

## B.8 模型保存与加载

```python
# 推荐方式：只保存参数（state_dict）
torch.save(model.state_dict(), 'model.pth')

# 加载参数
model = MLP(128, 256, 10)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

# 保存完整检查点（含优化器状态，用于继续训练）
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载检查点
ckpt = torch.load('checkpoint.pth', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
start_epoch = ckpt['epoch'] + 1
```

---

## B.9 设备管理

```python
# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将模型和数据移动到设备
model = model.to(device)
x = x.to(device)

# 多GPU（数据并行）
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# 查看GPU信息
print(torch.cuda.is_available())       # True/False
print(torch.cuda.device_count())       # GPU数量
print(torch.cuda.get_device_name(0))   # 第0块GPU名称
print(torch.cuda.memory_allocated(0))  # 已分配显存（字节）
```

---

## B.10 常用工具函数速查

```python
# 设置随机种子（保证可复现）
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np, random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 计算模型参数量
def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# 模型推理模式切换
model.train()   # 启用 Dropout 和 BatchNorm 的训练行为
model.eval()    # 禁用上述行为（推理时使用）

# 冻结/解冻参数
for param in model.parameters():
    param.requires_grad = False   # 冻结所有参数

for param in model.net[-1].parameters():
    param.requires_grad = True    # 只解冻最后一层（微调常用）

# 梯度裁剪（防止梯度爆炸）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Tensor 与 NumPy 互转
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
t   = torch.from_numpy(arr)      # 共享内存
arr2 = t.numpy()                 # 共享内存（CPU张量）
arr3 = t.detach().cpu().numpy()  # 安全写法（GPU张量）
```

---

> **版本说明**：本附录基于 PyTorch 2.x 编写。建议通过 `import torch; print(torch.__version__)` 确认当前版本，部分API在旧版本中可能存在差异。
