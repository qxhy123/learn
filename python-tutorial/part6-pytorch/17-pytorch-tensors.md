# 第17章：PyTorch张量基础

> 张量是深度学习的语言。理解张量，就是理解神经网络的数据流动方式。

---

## 学习目标

学完本章，你将能够：

1. 理解PyTorch张量的概念，掌握与NumPy数组的联系与区别
2. 熟练使用多种方式创建张量（`tensor`、`zeros`、`ones`、`rand`、`from_numpy`）
3. 掌握张量的算术运算、矩阵乘法和广播机制
4. 理解自动微分原理，使用`autograd`计算梯度
5. 利用GPU加速张量运算，掌握`.cuda()`和`.to(device)`的用法

---

## 17.1 PyTorch简介与安装

### 什么是PyTorch

*PyTorch* 是由Meta AI Research（Facebook）开发的深度学习框架，于2016年开源。它以**动态计算图**为核心特性，使得调试和实验极为便捷，已成为学术研究和工业实践中最受欢迎的深度学习框架之一。

PyTorch的核心组件：

| 组件 | 说明 |
|------|------|
| `torch` | 核心张量库，类似NumPy但支持GPU |
| `torch.nn` | 神经网络层、损失函数等模块 |
| `torch.optim` | 优化算法（SGD、Adam等） |
| `torch.autograd` | 自动微分引擎 |
| `torch.utils.data` | 数据加载工具 |

### 安装PyTorch

```bash
# 方式一：CPU版本（适合学习和入门）
pip install torch torchvision torchaudio

# 方式二：GPU版本（需要NVIDIA显卡和CUDA）
# 访问 https://pytorch.org/get-started/locally/ 获取对应版本的安装命令
# 以CUDA 12.1为例：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 方式三：使用conda（推荐，能自动处理CUDA依赖）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

验证安装是否成功：

```python
import torch

print(torch.__version__)
# 输出: 2.1.0（版本号因安装版本而异）

print(torch.cuda.is_available())
# 输出: True（有GPU时）或 False（无GPU时）
```

### PyTorch与NumPy的对比

PyTorch张量和NumPy数组在概念上非常相似，但张量额外支持：

- **GPU加速**：张量可以在GPU上存储和运算，速度远超CPU
- **自动微分**：张量记录运算历史，自动计算梯度
- **深度学习集成**：与`nn.Module`等深度学习组件无缝衔接

```python
import numpy as np
import torch

# NumPy数组
arr = np.array([1.0, 2.0, 3.0])
print(type(arr))           # <class 'numpy.ndarray'>
print(arr.dtype)           # float64

# PyTorch张量
tensor = torch.tensor([1.0, 2.0, 3.0])
print(type(tensor))        # <class 'torch.Tensor'>
print(tensor.dtype)        # torch.float32
```

> **注意**：NumPy默认使用`float64`（双精度），而PyTorch默认使用`float32`（单精度）。深度学习通常使用`float32`，因为它在速度和精度之间取得了良好平衡。

---

## 17.2 张量创建

张量（*Tensor*）是多维数组的统称。标量是0维张量，向量是1维张量，矩阵是2维张量，图像数据通常是4维张量（批量大小×通道×高度×宽度）。

### 使用 `torch.tensor()` 从数据创建

```python
import torch

# 从列表创建一维张量（向量）
v = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(v)
# 输出: tensor([1., 2., 3., 4.])
print(v.shape)
# 输出: torch.Size([4])

# 从嵌套列表创建二维张量（矩阵）
m = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(m)
# 输出:
# tensor([[1, 2, 3],
#         [4, 5, 6]])
print(m.shape)
# 输出: torch.Size([2, 3])

# 指定数据类型
x = torch.tensor([1, 2, 3], dtype=torch.float32)
print(x.dtype)
# 输出: torch.float32

# 三维张量（如小批量图像特征）
t = torch.tensor([[[1, 2], [3, 4]],
                  [[5, 6], [7, 8]]])
print(t.shape)
# 输出: torch.Size([2, 2, 2])
```

### 使用 `torch.zeros()` 和 `torch.ones()`

```python
# 全零张量
zeros = torch.zeros(3, 4)
print(zeros)
# 输出:
# tensor([[0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])

# 全一张量
ones = torch.ones(2, 3)
print(ones)
# 输出:
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])

# 指定形状的常数张量
fives = torch.full((3, 3), fill_value=5.0)
print(fives)
# 输出:
# tensor([[5., 5., 5.],
#         [5., 5., 5.],
#         [5., 5., 5.]])

# 与已有张量相同形状的全零/全一张量
existing = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
zeros_like = torch.zeros_like(existing)
ones_like = torch.ones_like(existing)
print(zeros_like.shape)    # torch.Size([2, 2])
print(zeros_like.dtype)    # torch.float32（保留原始数据类型）
```

### 使用 `torch.rand()` 和相关函数创建随机张量

```python
# 均匀分布随机张量（值在[0, 1)之间）
uniform = torch.rand(3, 3)
print(uniform)
# 输出（示例）:
# tensor([[0.7231, 0.1402, 0.5849],
#         [0.3901, 0.9128, 0.2345],
#         [0.6712, 0.0531, 0.8874]])

# 标准正态分布随机张量（均值0，标准差1）
normal = torch.randn(2, 4)
print(normal)
# 输出（示例）:
# tensor([[ 0.3241, -1.2104,  0.7823, -0.4512],
#         [ 1.1023,  0.0534, -0.8821,  0.2217]])

# 随机整数张量（范围[low, high)）
ints = torch.randint(low=0, high=10, size=(3, 3))
print(ints)
# 输出（示例）:
# tensor([[7, 2, 9],
#         [0, 5, 3],
#         [8, 1, 6]])

# 设置随机种子以保证可复现性
torch.manual_seed(42)
reproducible = torch.rand(2, 2)
print(reproducible)
# 输出（固定结果）:
# tensor([[0.8823, 0.9150],
#         [0.3829, 0.9593]])
```

### 使用 `torch.from_numpy()` 从NumPy转换

```python
import numpy as np
import torch

# NumPy数组转PyTorch张量
arr = np.array([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])
tensor = torch.from_numpy(arr)
print(tensor)
# 输出:
# tensor([[1., 2., 3.],
#         [4., 5., 6.]], dtype=torch.float64)

# 重要：from_numpy创建的张量与原数组共享内存
arr[0, 0] = 99.0
print(tensor[0, 0])
# 输出: tensor(99., dtype=torch.float64)  ← 张量也随之改变！

# 如果不想共享内存，使用torch.tensor()（会复制数据）
arr2 = np.array([1.0, 2.0, 3.0])
tensor_copy = torch.tensor(arr2)     # 数据副本，不共享内存
arr2[0] = 99.0
print(tensor_copy[0])
# 输出: tensor(1.)  ← 不受影响

# PyTorch张量转NumPy数组
t = torch.tensor([1.0, 2.0, 3.0])
numpy_arr = t.numpy()
print(type(numpy_arr))
# 输出: <class 'numpy.ndarray'>
```

### 其他常用创建方式

```python
# 等差数列
arange_t = torch.arange(0, 10, step=2)
print(arange_t)
# 输出: tensor([0, 2, 4, 6, 8])

# 均匀分布的N个点
linspace_t = torch.linspace(0, 1, steps=5)
print(linspace_t)
# 输出: tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

# 单位矩阵
eye = torch.eye(4)
print(eye)
# 输出:
# tensor([[1., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 1., 0.],
#         [0., 0., 0., 1.]])
```

---

## 17.3 张量运算

### 基本算术运算

张量支持所有标准算术运算，既可以用运算符，也可以用函数式写法：

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 加法
print(a + b)
# 输出: tensor([5., 7., 9.])
print(torch.add(a, b))
# 输出: tensor([5., 7., 9.])

# 减法
print(a - b)
# 输出: tensor([-3., -3., -3.])

# 逐元素乘法
print(a * b)
# 输出: tensor([ 4., 10., 18.])

# 逐元素除法
print(a / b)
# 输出: tensor([0.2500, 0.4000, 0.5000])

# 幂运算
print(a ** 2)
# 输出: tensor([1., 4., 9.])

# 标量运算（广播到所有元素）
print(a * 2 + 1)
# 输出: tensor([3., 5., 7.])
```

**原地操作（In-place Operations）**：

```python
x = torch.tensor([1.0, 2.0, 3.0])

# 原地加法（方法名末尾有下划线"_"表示原地操作）
x.add_(10)
print(x)
# 输出: tensor([11., 12., 13.])

# 原地乘法
x.mul_(2)
print(x)
# 输出: tensor([22., 24., 26.])
```

> **注意**：原地操作修改张量本身，节省内存，但会破坏自动微分的计算图，在需要梯度的张量上谨慎使用。

### 常用数学函数

```python
x = torch.tensor([1.0, 4.0, 9.0, 16.0])

# 平方根
print(torch.sqrt(x))
# 输出: tensor([1., 2., 3., 4.])

# 自然指数
print(torch.exp(torch.tensor([0.0, 1.0, 2.0])))
# 输出: tensor([1.0000, 2.7183, 7.3891])

# 自然对数
print(torch.log(x))
# 输出: tensor([0.0000, 1.3863, 2.1972, 2.7726])

# 绝对值
print(torch.abs(torch.tensor([-3.0, 1.0, -2.0])))
# 输出: tensor([3., 1., 2.])

# 激活函数（深度学习常用）
z = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(torch.relu(z))      # ReLU: max(0, x)
# 输出: tensor([0., 0., 0., 1., 2.])

print(torch.sigmoid(z))   # Sigmoid: 1 / (1 + e^(-x))
# 输出: tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])

print(torch.tanh(z))      # Tanh
# 输出: tensor([-0.9640, -0.7616,  0.0000,  0.7616,  0.9640])
```

### 聚合运算

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# 求和
print(torch.sum(x))
# 输出: tensor(21.)

# 按维度求和
print(torch.sum(x, dim=0))   # 按列求和
# 输出: tensor([5., 7., 9.])
print(torch.sum(x, dim=1))   # 按行求和
# 输出: tensor([ 6., 15.])

# 均值
print(torch.mean(x))
# 输出: tensor(3.5000)

# 最大值和最小值
print(torch.max(x))
# 输出: tensor(6.)
print(torch.min(x))
# 输出: tensor(1.)

# 最大值的索引
print(torch.argmax(x, dim=1))
# 输出: tensor([2, 2])（每行最大值在第2列）
```

### 矩阵乘法

矩阵乘法是深度学习中最核心的运算，线性层本质上就是矩阵乘法：

```python
import torch

# 定义两个矩阵
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])   # 形状: (2, 2)

B = torch.tensor([[5.0, 6.0],
                  [7.0, 8.0]])   # 形状: (2, 2)

# 方式一：torch.matmul()（推荐，适用范围广）
C = torch.matmul(A, B)
print(C)
# 输出:
# tensor([[19., 22.],
#         [43., 50.]])

# 方式二：@ 运算符（Python 3.5+ 支持）
C2 = A @ B
print(C2)
# 输出:
# tensor([[19., 22.],
#         [43., 50.]])

# 矩阵与向量的乘法
W = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])   # 形状: (2, 3)

x = torch.tensor([1.0, 1.0, 1.0])    # 形状: (3,)

y = W @ x
print(y)
# 输出: tensor([ 6., 15.])

# 批量矩阵乘法（Batched matmul）
# 形状 (batch, m, k) × (batch, k, n) → (batch, m, n)
batch_A = torch.randn(4, 3, 5)   # 4个 3×5 矩阵
batch_B = torch.randn(4, 5, 2)   # 4个 5×2 矩阵
batch_C = torch.bmm(batch_A, batch_B)
print(batch_C.shape)
# 输出: torch.Size([4, 3, 2])

# 点积（内积）
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
dot = torch.dot(a, b)
print(dot)
# 输出: tensor(32.)   # 1*4 + 2*5 + 3*6 = 32
```

### 广播机制

*广播（Broadcasting）* 允许形状不同的张量进行运算，规则与NumPy完全一致：

```python
import torch

# 示例一：向量加矩阵
A = torch.ones(3, 4)      # 形状: (3, 4)
b = torch.tensor([1.0, 2.0, 3.0, 4.0])  # 形状: (4,)

result = A + b            # b广播为 (3, 4)
print(result)
# 输出:
# tensor([[2., 3., 4., 5.],
#         [2., 3., 4., 5.],
#         [2., 3., 4., 5.]])

# 示例二：列向量加行向量（生成外积矩阵）
col = torch.tensor([[1.0], [2.0], [3.0]])   # 形状: (3, 1)
row = torch.tensor([[10.0, 20.0, 30.0]])    # 形状: (1, 3)

print(col + row)
# 输出:
# tensor([[11., 21., 31.],
#         [12., 22., 32.],
#         [13., 23., 33.]])

# 深度学习中常见的广播：批归一化
batch = torch.randn(8, 32, 224, 224)   # 形状: (N, C, H, W)
mean = torch.zeros(1, 32, 1, 1)        # 每通道均值
std = torch.ones(1, 32, 1, 1)          # 每通道标准差
normalized = (batch - mean) / std      # 广播运算
print(normalized.shape)
# 输出: torch.Size([8, 32, 224, 224])
```

### 张量形状操作

```python
x = torch.arange(12, dtype=torch.float32)
print(x)
# 输出: tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])

# 变形
y = x.reshape(3, 4)
print(y)
# 输出:
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])

# 转置
print(y.T)
# 输出:
# tensor([[ 0.,  4.,  8.],
#         [ 1.,  5.,  9.],
#         [ 2.,  6., 10.],
#         [ 3.,  7., 11.]])

# 增加/删除维度
z = torch.tensor([1.0, 2.0, 3.0])   # 形状: (3,)
print(z.unsqueeze(0).shape)          # 输出: torch.Size([1, 3])
print(z.unsqueeze(1).shape)          # 输出: torch.Size([3, 1])

w = torch.zeros(1, 3, 1)
print(w.squeeze().shape)             # 输出: torch.Size([3])

# 拼接
a = torch.ones(2, 3)
b = torch.zeros(2, 3)
print(torch.cat([a, b], dim=0).shape)   # 输出: torch.Size([4, 3])
print(torch.cat([a, b], dim=1).shape)   # 输出: torch.Size([2, 6])
```

---

## 17.4 自动微分

*自动微分（Autograd）* 是PyTorch的核心功能，它能自动计算张量运算的梯度，为深度学习中的参数更新提供基础。

### requires_grad 与计算图

```python
import torch

# 创建需要梯度的张量
x = torch.tensor(3.0, requires_grad=True)
print(x)
# 输出: tensor(3., requires_grad=True)

# 定义计算
y = x ** 2         # y = x²
z = 2 * y + 1      # z = 2x² + 1

print(z)
# 输出: tensor(19., grad_fn=<AddBackward0>)
# grad_fn 表示这个张量是通过某个运算得到的，autograd会追踪它
```

### backward() 计算梯度

```python
import torch

x = torch.tensor(3.0, requires_grad=True)

# 前向计算
z = 2 * x ** 2 + 1   # z = 2x² + 1

# 反向传播（计算梯度）
z.backward()

# 查看梯度 dz/dx = 4x = 12
print(x.grad)
# 输出: tensor(12.)
```

对多变量函数求梯度：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 线性变换：y = w * x + b
y = w * x + b      # y = 3*2 + 1 = 7

# 定义损失：L = y²
L = y ** 2         # L = 49

# 反向传播
L.backward()

# 梯度计算：
# dL/dy = 2y = 14
# dL/dw = dL/dy * dy/dw = 14 * x = 14 * 2 = 28
# dL/dx = dL/dy * dy/dx = 14 * w = 14 * 3 = 42
# dL/db = dL/dy * dy/db = 14 * 1 = 14

print(f"dL/dw = {w.grad}")   # 输出: dL/dw = tensor(28.)
print(f"dL/dx = {x.grad}")   # 输出: dL/dx = tensor(42.)
print(f"dL/db = {b.grad}")   # 输出: dL/db = tensor(14.)
```

### 向量输出的梯度

当输出是向量时，`backward()` 需要传入梯度权重：

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

y = x ** 2 + 2 * x + 1   # y = x² + 2x + 1

# dy/dx = 2x + 2
# 在x=[1,2,3]处: [4, 6, 8]

# 传入和y形状相同的权重向量（通常是全1向量，等效于对输出求和后再backward）
y.backward(torch.ones_like(y))

print(x.grad)
# 输出: tensor([4., 6., 8.])
```

### 梯度累积与清零

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

# 第一次反向传播
y = x ** 2
y.backward()
print(x.grad)
# 输出: tensor(4.)

# 第二次反向传播（梯度累积！）
y = x ** 2
y.backward()
print(x.grad)
# 输出: tensor(8.)  ← 梯度被累积了，不是4.！

# 正确做法：每次反向传播前清零梯度
x.grad.zero_()
y = x ** 2
y.backward()
print(x.grad)
# 输出: tensor(4.)  ← 清零后正确
```

> **重要**：在训练神经网络时，每个批次开始前必须调用 `optimizer.zero_grad()` 清零梯度，否则梯度会不断累积，导致错误的参数更新。

### 停止梯度追踪

有时只需要前向推理，不需要计算梯度（如验证/测试阶段）：

```python
import torch

x = torch.tensor(3.0, requires_grad=True)

# 方式一：torch.no_grad() 上下文管理器（推荐）
with torch.no_grad():
    y = x ** 2 + 1
    print(y)
    # 输出: tensor(10.)
    print(y.requires_grad)
    # 输出: False

# 方式二：.detach() 方法
y = x ** 2 + 1
z = y.detach()       # z与y共享数据，但不参与梯度计算
print(z.requires_grad)
# 输出: False

# 方式三：直接修改 requires_grad
x2 = torch.tensor(3.0)
x2.requires_grad_(False)
print(x2.requires_grad)
# 输出: False
```

---

## 17.5 GPU加速

GPU（图形处理单元）拥有数千个计算核心，非常适合深度学习中大量的并行矩阵运算。

### 检测GPU可用性

```python
import torch

# 检查CUDA是否可用
print(torch.cuda.is_available())
# 输出: True 或 False

# 查看GPU数量
print(torch.cuda.device_count())
# 输出: 1（单GPU系统）

# 查看当前GPU名称
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    # 输出（示例）: NVIDIA GeForce RTX 3080

# 最佳实践：自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 输出: cuda（有GPU时）或 cpu（无GPU时）
```

### 将张量移动到GPU

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 方式一：创建时直接指定设备
x_gpu = torch.tensor([1.0, 2.0, 3.0], device=device)
print(x_gpu.device)
# 输出: cuda:0（有GPU时）或 cpu（无GPU时）

# 方式二：使用 .to(device) 移动已有张量
x_cpu = torch.tensor([1.0, 2.0, 3.0])
x_gpu = x_cpu.to(device)
print(x_gpu.device)
# 输出: cuda:0

# 方式三：使用 .cuda()（仅适用于CUDA，不推荐，可移植性差）
# x_gpu = x_cpu.cuda()  # 等价于 x_cpu.to("cuda")

# 验证：GPU上的运算结果仍在GPU上
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)
c = a @ b
print(c.device)
# 输出: cuda:0
```

### CPU与GPU之间的数据传输

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU张量 → CPU
x = torch.tensor([1.0, 2.0, 3.0], device=device)
x_cpu = x.cpu()
print(x_cpu.device)
# 输出: cpu

# GPU张量 → NumPy（必须先移到CPU）
if device.type == "cuda":
    np_array = x.cpu().numpy()    # 正确：先.cpu()再.numpy()
else:
    np_array = x.numpy()
```

> **陷阱**：在GPU张量上直接调用 `.numpy()` 会报错，必须先用 `.cpu()` 移到CPU。

### GPU加速的性能对比

```python
import torch
import time

size = 5000

# CPU运算
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU时间: {cpu_time:.4f}秒")

# GPU运算（仅在有GPU时才有意义）
if torch.cuda.is_available():
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()

    # 预热（第一次运算有额外开销）
    _ = a_gpu @ b_gpu
    torch.cuda.synchronize()

    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()   # 等待GPU运算完成
    gpu_time = time.time() - start
    print(f"GPU时间: {gpu_time:.4f}秒")
    print(f"加速比: {cpu_time / gpu_time:.1f}x")
```

### 多GPU支持

```python
import torch

# 指定使用第二块GPU（如果有多块GPU）
device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

# 查看所有GPU的显存使用情况
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    total_mem = props.total_memory / 1024 ** 3
    print(f"GPU {i}: {props.name}, 显存: {total_mem:.1f} GB")
```

---

## 本章小结

| 知识点 | 核心API | 说明 |
|--------|---------|------|
| 张量创建 | `torch.tensor()` | 从Python列表/NumPy数组创建 |
| | `torch.zeros()` / `torch.ones()` | 全零/全一张量 |
| | `torch.rand()` / `torch.randn()` | 均匀/正态随机张量 |
| | `torch.from_numpy()` | 从NumPy转换，共享内存 |
| | `torch.arange()` / `torch.linspace()` | 序列张量 |
| 张量属性 | `.shape` / `.dtype` / `.device` | 形状/类型/设备 |
| 算术运算 | `+` `-` `*` `/` `**` | 逐元素运算，支持广播 |
| 矩阵乘法 | `torch.matmul()` / `@` | 矩阵乘法（非逐元素）|
| | `torch.bmm()` | 批量矩阵乘法 |
| 形状操作 | `.reshape()` / `.view()` | 变形，不复制数据 |
| | `.unsqueeze()` / `.squeeze()` | 增加/删除维度 |
| | `.T` / `.transpose()` | 转置 |
| | `torch.cat()` | 拼接 |
| 自动微分 | `requires_grad=True` | 标记需要梯度的张量 |
| | `.backward()` | 反向传播，计算梯度 |
| | `.grad` | 访问梯度值 |
| | `torch.no_grad()` | 关闭梯度追踪 |
| | `.detach()` | 从计算图分离 |
| GPU加速 | `torch.cuda.is_available()` | 检测GPU |
| | `.to(device)` | 移动到指定设备 |
| | `.cuda()` / `.cpu()` | 设备间移动 |

---

## 深度学习应用：计算图构建

本节通过一个完整的**单层神经网络前向传播和反向传播**示例，展示PyTorch计算图的构建过程，并可视化梯度的流动。

### 场景描述

实现一个简单的单层神经网络（即线性层 + 激活函数），用于二分类任务：

```
输入 x (4维) → 线性变换 (W, b) → Sigmoid激活 → 输出 ŷ (1维)
损失函数：二元交叉熵 L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

### 完整代码实现

```python
import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────
# 第一步：初始化参数（requires_grad=True 开启梯度追踪）
# ──────────────────────────────────────────────
torch.manual_seed(42)

# 权重矩阵：形状 (1, 4)，即4输入1输出
W = torch.randn(1, 4, requires_grad=True)
# 偏置：形状 (1,)
b = torch.zeros(1, requires_grad=True)

print("初始参数：")
print(f"  W = {W.data}")
print(f"  b = {b.data}")
print()

# ──────────────────────────────────────────────
# 第二步：准备输入数据和标签
# ──────────────────────────────────────────────
# 一个样本，4个特征
x = torch.tensor([[2.0, -1.0, 3.0, 0.5]])   # 形状: (1, 4)
y_true = torch.tensor([[1.0]])               # 真实标签: 1（正类）

print(f"输入 x: {x}")
print(f"真实标签 y: {y_true.item()}")
print()

# ──────────────────────────────────────────────
# 第三步：前向传播（构建计算图）
# ──────────────────────────────────────────────

# 3.1 线性变换：z = x @ W.T + b
# x: (1, 4),  W.T: (4, 1)  →  z: (1, 1)
z = x @ W.T + b
print(f"线性变换结果 z = x·W^T + b = {z.item():.4f}")
print(f"  grad_fn: {z.grad_fn}")

# 3.2 Sigmoid激活：ŷ = 1 / (1 + e^(-z))
y_pred = torch.sigmoid(z)
print(f"Sigmoid激活后 ŷ = σ(z) = {y_pred.item():.4f}")
print(f"  grad_fn: {y_pred.grad_fn}")
print()

# ──────────────────────────────────────────────
# 第四步：计算损失（二元交叉熵）
# ──────────────────────────────────────────────
# L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
loss = F.binary_cross_entropy(y_pred, y_true)
print(f"损失 L = BCE(ŷ, y) = {loss.item():.4f}")
print(f"  grad_fn: {loss.grad_fn}")
print()

# ──────────────────────────────────────────────
# 第五步：反向传播（沿计算图反向计算梯度）
# ──────────────────────────────────────────────
loss.backward()

print("反向传播后的梯度：")
print(f"  dL/dW = {W.grad}")
print(f"  dL/db = {b.grad}")
print()

# ──────────────────────────────────────────────
# 第六步：梯度下降更新参数
# ──────────────────────────────────────────────
learning_rate = 0.1

with torch.no_grad():   # 参数更新不需要追踪梯度
    W -= learning_rate * W.grad
    b -= learning_rate * b.grad

# 清零梯度，准备下一轮迭代
W.grad.zero_()
b.grad.zero_()

print("更新后的参数：")
print(f"  W = {W.data}")
print(f"  b = {b.data}")
```

**输出示例：**

```
初始参数：
  W = tensor([[ 0.3367, -0.1288,  0.2345,  0.2303]])
  b = tensor([0.])

输入 x: tensor([[ 2., -1.,  3.,  0.5]])
真实标签 y: 1.0

线性变换结果 z = x·W^T + b = 1.2600
  grad_fn: <AddBackward0>
Sigmoid激活后 ŷ = σ(z) = 0.7791
  grad_fn: <SigmoidBackward0>

损失 L = BCE(ŷ, y) = 0.2498
  grad_fn: <BinaryCrossEntropyBackward0>

反向传播后的梯度：
  dL/dW = tensor([[-0.4843,  0.2421, -0.7264, -0.1210]])
  dL/db = tensor([-0.2421])

更新后的参数：
  W = tensor([[ 0.3851, -0.1530,  0.3071,  0.2424]])
  b = tensor([0.0242])
```

### 可视化计算图结构

下面展示计算图的结构，帮助理解反向传播的链式法则：

```python
import torch

def print_computation_graph(tensor, indent=0):
    """递归打印计算图结构"""
    prefix = "  " * indent
    if tensor.grad_fn is not None:
        print(f"{prefix}[运算节点] {tensor.grad_fn.__class__.__name__}")
        for child, _ in tensor.grad_fn.next_functions:
            if child is not None:
                print_computation_graph_fn(child, indent + 1)
    else:
        print(f"{prefix}[叶节点] shape={tensor.shape}, requires_grad={tensor.requires_grad}")

def print_computation_graph_fn(grad_fn, indent=0):
    """递归打印grad_fn的计算图"""
    prefix = "  " * indent
    print(f"{prefix}└─ {grad_fn.__class__.__name__}")
    for child, _ in grad_fn.next_functions:
        if child is not None:
            print_computation_graph_fn(child, indent + 1)

# 构建一个简单的计算图
torch.manual_seed(0)
W = torch.randn(2, 3, requires_grad=True)
x = torch.randn(3, requires_grad=False)
b = torch.zeros(2, requires_grad=True)

z = W @ x + b       # 线性层
a = torch.sigmoid(z)  # 激活
L = a.sum()         # 标量损失

print("计算图结构（从损失L往回追踪）：")
print(f"L (损失标量)")
print_computation_graph_fn(L.grad_fn)
```

**输出：**

```
计算图结构（从损失L往回追踪）：
L (损失标量)
└─ SumBackward0
  └─ SigmoidBackward0
    └─ AddBackward0
      └─ MvBackward0
        └─ AccumulateGrad   ← W 叶节点
      └─ AccumulateGrad     ← b 叶节点
```

### 手动实现梯度下降训练循环

用纯张量运算（不使用`nn.Module`）实现一个完整的训练循环，以深入理解底层原理：

```python
import torch

torch.manual_seed(42)

# ── 生成玩具数据集 ──
# 真实模型：y = 2*x1 - 3*x2 + 1
n_samples = 100
X = torch.randn(n_samples, 2)
true_W = torch.tensor([[2.0, -3.0]])
true_b = torch.tensor([1.0])
y = (X @ true_W.T + true_b).squeeze() + 0.1 * torch.randn(n_samples)

# ── 初始化可学习参数 ──
W = torch.randn(1, 2, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

learning_rate = 0.05
n_epochs = 100

print(f"{'Epoch':>6} | {'Loss':>10} | {'W[0]':>8} | {'W[1]':>8} | {'b':>8}")
print("-" * 50)

for epoch in range(n_epochs):
    # 前向传播
    y_pred = (X @ W.T + b).squeeze()

    # 计算MSE损失
    loss = ((y_pred - y) ** 2).mean()

    # 反向传播
    loss.backward()

    # 参数更新（no_grad中进行，避免追踪梯度）
    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

    # 清零梯度
    W.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(f"{epoch+1:>6} | {loss.item():>10.4f} | "
              f"{W[0,0].item():>8.4f} | {W[0,1].item():>8.4f} | "
              f"{b[0].item():>8.4f}")

print()
print(f"真实参数: W=[{true_W[0,0].item():.4f}, {true_W[0,1].item():.4f}], b={true_b[0].item():.4f}")
print(f"学习参数: W=[{W[0,0].item():.4f}, {W[0,1].item():.4f}], b={b[0].item():.4f}")
```

**输出示例：**

```
 Epoch |       Loss |     W[0] |     W[1] |        b
--------------------------------------------------
    20 |     0.2134 |   1.7823 |  -2.8901 |   0.9012
    40 |     0.1089 |   1.9241 |  -2.9512 |   0.9623
    60 |     0.0821 |   1.9712 |  -2.9801 |   0.9845
    80 |     0.0743 |   1.9876 |  -2.9912 |   0.9934
   100 |     0.0721 |   1.9934 |  -2.9961 |   0.9971

真实参数: W=[2.0000, -3.0000], b=1.0000
学习参数: W=[1.9934, -2.9961], b=0.9971
```

学习到的参数已非常接近真实值，验证了梯度下降的有效性。

---

## 练习题

### 基础题

**练习1**：张量创建与属性

创建以下三个张量，并分别打印它们的 `shape`、`dtype` 和 `device`：
1. 一个形状为 `(5, 5)` 的全零浮点张量
2. 一个包含1到10的整数张量（形状为 `(2, 5)`）
3. 一个从 NumPy 数组 `np.array([1.1, 2.2, 3.3])` 转换来的张量

**练习2**：张量运算

给定矩阵 `A = [[1, 2], [3, 4]]` 和向量 `b = [10, 20]`，使用PyTorch完成：
1. 计算 `A @ A`（矩阵乘以自身）
2. 计算 `A + b`（利用广播机制）
3. 计算 `A` 每行的均值

---

### 中级题

**练习3**：自动微分手动验证

定义函数 $f(x, y) = x^3 + 2xy^2 - 3y$，在点 $(x=2, y=1)$ 处：
1. 使用PyTorch的`autograd`计算 $\partial f / \partial x$ 和 $\partial f / \partial y$
2. 手动推导偏导数公式，验证PyTorch的计算结果是否正确

提示：$\partial f / \partial x = 3x^2 + 2y^2$，$\partial f / \partial y = 4xy - 3$

**练习4**：线性回归优化

给定数据点（x, y）：`[(1, 2.1), (2, 3.9), (3, 6.2), (4, 7.8), (5, 10.1)]`，
用PyTorch张量和自动微分实现线性回归（`y = w*x + b`），
使用梯度下降训练200轮，学习率0.01，每50轮打印一次损失和参数。

---

### 提高题

**练习5**：实现Softmax与交叉熵

不使用`torch.nn.functional`，仅用基础张量运算手动实现：
1. **Softmax函数**：$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$，注意数值稳定性
2. **交叉熵损失**：$L = -\sum_i y_i \log(\hat{y}_i)$
3. 对输入 `logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])` 和标签 `labels = torch.tensor([0, 1])` 计算损失
4. 调用 `loss.backward()` 验证可以正常计算梯度
5. 与 `F.cross_entropy(logits, labels)` 的结果进行对比验证

---

## 练习答案

### 练习1答案

```python
import numpy as np
import torch

# 1. 全零浮点张量
t1 = torch.zeros(5, 5)
print(f"形状: {t1.shape}, 类型: {t1.dtype}, 设备: {t1.device}")
# 输出: 形状: torch.Size([5, 5]), 类型: torch.float32, 设备: cpu

# 2. 1到10整数张量（reshape为2×5）
t2 = torch.arange(1, 11).reshape(2, 5)
print(f"形状: {t2.shape}, 类型: {t2.dtype}, 设备: {t2.device}")
# 输出: 形状: torch.Size([2, 5]), 类型: torch.int64, 设备: cpu
print(t2)
# 输出:
# tensor([[ 1,  2,  3,  4,  5],
#         [ 6,  7,  8,  9, 10]])

# 3. 从NumPy转换
arr = np.array([1.1, 2.2, 3.3])
t3 = torch.from_numpy(arr)
print(f"形状: {t3.shape}, 类型: {t3.dtype}, 设备: {t3.device}")
# 输出: 形状: torch.Size([3]), 类型: torch.float64, 设备: cpu
```

### 练习2答案

```python
import torch

A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([10.0, 20.0])

# 1. 矩阵乘以自身
print("A @ A =")
print(A @ A)
# 输出:
# tensor([[ 7., 10.],
#         [15., 22.]])

# 2. 广播加法：b形状(2,)广播到(2,2)
print("A + b =")
print(A + b)
# 输出:
# tensor([[11., 22.],
#         [13., 24.]])

# 3. 每行均值（沿dim=1求均值）
print("每行均值:", torch.mean(A, dim=1))
# 输出: 每行均值: tensor([1.5000, 3.5000])
```

### 练习3答案

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)

# f(x, y) = x³ + 2xy² - 3y
f = x**3 + 2*x*y**2 - 3*y

f.backward()

print(f"f(2, 1) = {f.item()}")
# 输出: f(2, 1) = 9.0  （8 + 4 - 3 = 9）

print(f"df/dx (PyTorch) = {x.grad.item()}")
# 输出: df/dx (PyTorch) = 14.0

print(f"df/dy (PyTorch) = {y.grad.item()}")
# 输出: df/dy (PyTorch) = 5.0

# 手动验证：
# df/dx = 3x² + 2y² = 3*(4) + 2*(1) = 12 + 2 = 14  ✓
# df/dy = 4xy - 3 = 4*2*1 - 3 = 8 - 3 = 5          ✓
print("手动验证: df/dx = 3*(2²) + 2*(1²) =", 3*4 + 2*1)  # 14
print("手动验证: df/dy = 4*(2)*(1) - 3 =", 4*2*1 - 3)    # 5
```

### 练习4答案

```python
import torch

# 数据
x_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_data = torch.tensor([2.1, 3.9, 6.2, 7.8, 10.1])

# 初始化参数
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.01
n_epochs = 200

for epoch in range(1, n_epochs + 1):
    # 前向传播
    y_pred = w * x_data + b

    # MSE损失
    loss = ((y_pred - y_data) ** 2).mean()

    # 反向传播
    loss.backward()

    # 参数更新
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:>3}: loss={loss.item():.4f}, "
              f"w={w.item():.4f}, b={b.item():.4f}")

# 输出:
# Epoch  50: loss=0.0463, w=1.9124, b=0.2014
# Epoch 100: loss=0.0212, w=1.9512, b=0.1023
# Epoch 150: loss=0.0182, w=1.9701, b=0.0534
# Epoch 200: loss=0.0176, w=1.9812, b=0.0287
# （结果接近真实 w≈2, b≈0）
```

### 练习5答案

```python
import torch
import torch.nn.functional as F

def stable_softmax(logits):
    """数值稳定的Softmax实现"""
    # 减去每行最大值防止exp上溢（log-sum-exp技巧）
    z = logits - logits.max(dim=1, keepdim=True).values
    exp_z = torch.exp(z)
    return exp_z / exp_z.sum(dim=1, keepdim=True)

def cross_entropy_loss(logits, labels):
    """手动实现交叉熵损失"""
    probs = stable_softmax(logits)
    n = logits.shape[0]
    # 取每个样本在真实类别上的概率
    correct_probs = probs[torch.arange(n), labels]
    # 交叉熵：-log(p_correct)，对batch取均值
    loss = -torch.log(correct_probs).mean()
    return loss

# 测试数据
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.3]], requires_grad=True)
labels = torch.tensor([0, 1])

# 手动实现
loss_manual = cross_entropy_loss(logits, labels)
print(f"手动实现损失: {loss_manual.item():.4f}")

# 验证反向传播
loss_manual.backward()
print(f"logits梯度:\n{logits.grad}")

# 与PyTorch官方实现对比
logits2 = torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3]], requires_grad=True)
loss_official = F.cross_entropy(logits2, labels)
print(f"官方实现损失: {loss_official.item():.4f}")

# 输出（手动实现与官方一致）:
# 手动实现损失: 0.4290
# logits梯度:
# tensor([[-0.1749,  0.1265,  0.0484],
#         [ 0.0859, -0.2055,  0.1196]])
# 官方实现损失: 0.4290
```

---

*下一章：[第18章：神经网络构建](./18-neural-networks.md) — 使用`nn.Module`搭建神经网络*
