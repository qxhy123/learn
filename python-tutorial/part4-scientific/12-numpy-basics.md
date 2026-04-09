# 第12章：NumPy基础

NumPy（Numerical Python）是Python科学计算的核心库，为多维数组操作提供了高效的数据结构和数学函数。深度学习框架如PyTorch和TensorFlow的底层设计均受NumPy影响，掌握NumPy是理解张量计算的必经之路。

---

## 学习目标

完成本章学习后，你将能够：

1. 使用多种方式创建NumPy数组，理解`ndarray`对象的核心属性
2. 熟练运用索引与切片操作访问和修改数组元素
3. 掌握元素级运算和通用函数（ufunc）的使用方法
4. 理解广播机制的规则，并将其应用于不同形状数组间的运算
5. 使用`reshape`、`transpose`、`flatten`等方法对数组进行变形操作，为深度学习中的张量变换打下基础

---

## 12.1 NumPy数组创建

NumPy的核心数据结构是`ndarray`（N维数组）。与Python原生列表相比，`ndarray`在内存布局上连续存储同类型数据，运算效率远高于列表推导式。

### 12.1.1 安装与导入

```python
# 安装（如未安装）
# pip install numpy

import numpy as np

# 查看版本
print(np.__version__)  # 例如：1.24.3
```

### 12.1.2 从列表创建数组

```python
# 从一维列表创建
a = np.array([1, 2, 3, 4, 5])
print(a)          # [1 2 3 4 5]
print(type(a))    # <class 'numpy.ndarray'>

# 从二维列表创建
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(b)
# [[1 2 3]
#  [4 5 6]]

# 指定数据类型
c = np.array([1.0, 2.0, 3.0], dtype=np.float32)
print(c.dtype)    # float32
```

### 12.1.3 ndarray的核心属性

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.ndim)    # 维度数：2
print(a.shape)   # 形状：(2, 3)
print(a.size)    # 元素总数：6
print(a.dtype)   # 数据类型：int64
print(a.itemsize)  # 每个元素字节数：8
```

### 12.1.4 zeros 和 ones

```python
# 全零数组
z = np.zeros((3, 4))
print(z)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# 全一数组
o = np.ones((2, 3), dtype=np.int32)
print(o)
# [[1 1 1]
#  [1 1 1]]

# 填充指定值
f = np.full((2, 2), 7.0)
print(f)
# [[7. 7.]
#  [7. 7.]]

# 单位矩阵（对角线为1）
eye = np.eye(3)
print(eye)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# 与已有数组形状相同的全零/全一数组
x = np.array([[1, 2], [3, 4]])
print(np.zeros_like(x))
# [[0 0]
#  [0 0]]
```

### 12.1.5 arange

`arange`类似Python内置的`range`，但返回的是`ndarray`。

```python
# arange(stop)
print(np.arange(5))          # [0 1 2 3 4]

# arange(start, stop)
print(np.arange(2, 8))       # [2 3 4 5 6 7]

# arange(start, stop, step)
print(np.arange(0, 1, 0.2))  # [0.  0.2 0.4 0.6 0.8]

# 注意：arange使用浮点步长时可能出现精度问题
print(np.arange(0, 0.3, 0.1))  # 可能输出 [0.  0.1 0.2]（不稳定）
```

### 12.1.6 linspace

`linspace`在指定区间内均匀生成固定数量的点，比`arange`更适合需要精确控制点数的场景。

```python
# linspace(start, stop, num)  包含端点
x = np.linspace(0, 1, 5)
print(x)   # [0.   0.25 0.5  0.75 1.  ]

# 不包含终点
x2 = np.linspace(0, 1, 5, endpoint=False)
print(x2)  # [0.  0.2 0.4 0.6 0.8]

# 常用场景：绘制函数曲线
import matplotlib.pyplot as plt  # 需要matplotlib

x = np.linspace(-np.pi, np.pi, 100)
y = np.sin(x)
# plt.plot(x, y)
```

### 12.1.7 随机数组

```python
# 设置随机种子（保证可复现）
np.random.seed(42)

# 均匀分布 [0, 1)
u = np.random.rand(3, 4)
print(u.shape)  # (3, 4)

# 标准正态分布
n = np.random.randn(2, 3)
print(n)

# 整数随机数
i = np.random.randint(0, 10, size=(3, 3))
print(i)

# 从已有数据中随机采样
data = np.array([10, 20, 30, 40, 50])
sample = np.random.choice(data, size=3, replace=False)
print(sample)  # 例如：[30 10 50]
```

---

## 12.2 数组索引与切片

NumPy提供了多种索引方式，灵活性远超Python列表。

### 12.2.1 基本索引

```python
a = np.array([10, 20, 30, 40, 50])

# 正向索引（从0开始）
print(a[0])   # 10
print(a[2])   # 30

# 负向索引（从末尾开始）
print(a[-1])  # 50
print(a[-2])  # 40

# 二维数组
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(b[0, 0])   # 1
print(b[1, 2])   # 6
print(b[-1, -1]) # 9

# 获取整行或整列
print(b[0])      # [1 2 3]（第0行）
print(b[:, 1])   # [2 5 8]（第1列）
```

### 12.2.2 切片操作

```python
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 基本切片 [start:stop:step]
print(a[2:7])     # [2 3 4 5 6]
print(a[::2])     # [0 2 4 6 8]（步长为2）
print(a[::-1])    # [9 8 7 6 5 4 3 2 1 0]（反转）

# 二维数组切片
b = np.arange(12).reshape(3, 4)
print(b)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

print(b[0:2, 1:3])   # 前两行，第1到2列
# [[1 2]
#  [5 6]]

print(b[:, ::2])     # 所有行，每隔一列
# [[ 0  2]
#  [ 4  6]
#  [ 8 10]]
```

### 12.2.3 切片是视图，不是副本

```python
a = np.array([1, 2, 3, 4, 5])
b = a[1:4]        # b 是 a 的视图
b[0] = 99
print(a)          # [ 1 99  3  4  5]（a 也被修改！）

# 若需要副本，使用 .copy()
c = a[1:4].copy()
c[0] = 0
print(a)          # [ 1 99  3  4  5]（a 不变）
```

### 12.2.4 花式索引（Fancy Indexing）

```python
a = np.array([10, 20, 30, 40, 50])

# 使用整数列表索引
idx = [0, 2, 4]
print(a[idx])     # [10 30 50]

# 二维花式索引
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

rows = [0, 2]
cols = [1, 2]
print(b[rows, cols])    # [2 9]（(0,1)和(2,2)处的元素）

# 用于按索引赋值
a[[1, 3]] = 0
print(a)  # [10  0 30  0 50]
```

### 12.2.5 布尔索引

布尔索引是NumPy最强大的特性之一，常用于数据过滤。

```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# 创建布尔掩码
mask = a > 3
print(mask)   # [False False  True False  True  True False  True]

# 用掩码过滤
print(a[mask])    # [4 5 9 6]

# 直接写条件
print(a[a % 2 == 0])  # [4 2 6]（偶数）

# 结合多个条件
print(a[(a > 2) & (a < 7)])  # [3 4 5 6]

# 深度学习场景：将负值置零（ReLU激活函数）
weights = np.array([-0.5, 0.3, -0.1, 0.8, -0.2])
relu_output = weights.copy()
relu_output[relu_output < 0] = 0
print(relu_output)  # [0.  0.3 0.  0.8 0. ]
```

---

## 12.3 数组运算

### 12.3.1 元素级算术运算

NumPy的运算默认是元素级（element-wise）的，无需循环。

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(a + b)    # [ 6  8 10 12]
print(a - b)    # [-4 -4 -4 -4]
print(a * b)    # [ 5 12 21 32]
print(a / b)    # [0.2        0.33333333 0.42857143 0.5       ]
print(a ** 2)   # [ 1  4  9 16]
print(a % 3)    # [1 2 0 1]

# 与标量的运算
print(a * 2)    # [2 4 6 8]
print(a + 10)   # [11 12 13 14]
```

### 12.3.2 性能对比：NumPy vs Python列表

```python
import time

size = 1_000_000

# Python列表方式
list_a = list(range(size))
list_b = list(range(size))

start = time.time()
result = [x + y for x, y in zip(list_a, list_b)]
print(f"列表耗时: {time.time() - start:.4f}s")

# NumPy方式
np_a = np.arange(size)
np_b = np.arange(size)

start = time.time()
result = np_a + np_b
print(f"NumPy耗时: {time.time() - start:.4f}s")

# 典型结果：NumPy比列表快约50-100倍
```

### 12.3.3 通用函数（ufunc）

通用函数是对`ndarray`逐元素操作的向量化函数。

```python
a = np.array([0, np.pi/4, np.pi/2, np.pi])

# 三角函数
print(np.sin(a))    # [0.000 0.707 1.000 0.000]（近似值）
print(np.cos(a))    # [1.000 0.707 0.000 -1.00]
print(np.tan(a))    # [0.000 1.000 很大  0.000]

# 指数与对数
x = np.array([1.0, 2.0, 3.0])
print(np.exp(x))    # [ 2.718  7.389 20.086]
print(np.log(x))    # [0.     0.693 1.099]
print(np.log2(x))   # [0.    1.    1.585]
print(np.log10(x))  # [0.    0.301 0.477]

# 平方根与绝对值
print(np.sqrt(np.array([1, 4, 9, 16])))  # [1. 2. 3. 4.]
print(np.abs(np.array([-3, -1, 0, 2])))  # [3 1 0 2]

# 取整函数
x = np.array([1.2, 2.7, -1.5, -2.3])
print(np.floor(x))   # [ 1.  2. -2. -3.]（向下取整）
print(np.ceil(x))    # [ 2.  3. -1. -2.]（向上取整）
print(np.round(x))   # [ 1.  3. -2. -2.]（四舍五入）
```

### 12.3.4 聚合函数

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# 全局聚合
print(np.sum(a))     # 21
print(np.mean(a))    # 3.5
print(np.max(a))     # 6
print(np.min(a))     # 1
print(np.std(a))     # 标准差

# 按轴聚合（axis=0 沿行方向，即逐列；axis=1 沿列方向，即逐行）
print(np.sum(a, axis=0))   # [5 7 9]（每列之和）
print(np.sum(a, axis=1))   # [ 6 15]（每行之和）
print(np.max(a, axis=1))   # [3 6]（每行最大值）

# 累积操作
print(np.cumsum(np.array([1, 2, 3, 4])))  # [ 1  3  6 10]
print(np.cumprod(np.array([1, 2, 3, 4]))) # [ 1  2  6 24]

# argmax / argmin（返回最大/最小值的索引）
b = np.array([3, 1, 4, 1, 5, 9])
print(np.argmax(b))   # 5
print(np.argmin(b))   # 1
```

### 12.3.5 矩阵运算

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# 元素级乘法（不是矩阵乘法）
print(A * B)
# [[ 5 12]
#  [21 32]]

# 矩阵乘法（点积）
print(np.dot(A, B))
# [[19 22]
#  [43 50]]

# Python 3.5+ 矩阵乘法运算符
print(A @ B)
# [[19 22]
#  [43 50]]

# 转置
print(A.T)
# [[1 3]
#  [2 4]]

# 行列式、逆矩阵（需要linalg模块）
print(np.linalg.det(A))    # -2.0
print(np.linalg.inv(A))
# [[-2.   1. ]
#  [ 1.5 -0.5]]
```

---

## 12.4 广播机制

广播（Broadcasting）是NumPy中形状不同的数组进行运算时的规则。理解广播对于高效编写向量化代码至关重要。

### 12.4.1 广播规则

NumPy广播遵循以下规则（从尾部维度开始对齐）：

1. 如果两个数组维度数不同，则在维度数少的数组的形状前补1
2. 如果某个维度大小为1，则沿该维度扩展以匹配另一个数组
3. 如果维度大小既不相等，也不为1，则报错

```python
# 规则图示
# 数组 A: (3, 4)
# 数组 B: (   4)  → 广播为 (1, 4) → (3, 4)
# 结果:   (3, 4)
```

### 12.4.2 标量广播

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# 标量被广播到所有元素
print(a + 10)
# [[11 12 13]
#  [14 15 16]]

print(a * 2)
# [[ 2  4  6]
#  [ 8 10 12]]
```

### 12.4.3 一维数组广播

```python
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])   # shape: (3, 3)

b = np.array([10, 20, 30])  # shape: (3,) → (1, 3) → (3, 3)

print(a + b)
# [[11 22 33]
#  [14 25 36]
#  [17 28 39]]

# 列方向广播：需要将 b 变为列向量
c = np.array([[10],
              [20],
              [30]])  # shape: (3, 1) → (3, 3)

print(a + c)
# [[11 12 13]
#  [24 25 26]
#  [37 38 39]]
```

### 12.4.4 典型应用：均值归一化

```python
# 对数据集的每个特征（列）进行归一化
data = np.array([[1.0, 200.0, 0.1],
                 [2.0, 400.0, 0.3],
                 [3.0, 600.0, 0.5]])  # shape: (3, 3)

mean = data.mean(axis=0)   # shape: (3,)，每列均值
std  = data.std(axis=0)    # shape: (3,)，每列标准差

# 广播自动将 mean 和 std 扩展为 (3, 3)
normalized = (data - mean) / std
print(normalized)
# 每列均值约为0，标准差约为1
```

### 12.4.5 广播失败示例

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape: (2, 3)

b = np.array([1, 2])        # shape: (2,)

# 尾部维度不匹配：3 ≠ 2，且都不为1
try:
    print(a + b)
except ValueError as e:
    print(f"错误：{e}")
# 错误：operands could not be broadcast together with shapes (2,3) (2,)

# 正确做法：将 b 变为列向量
b_col = b.reshape(2, 1)     # shape: (2, 1)
print(a + b_col)
# [[2 3 4]
#  [6 7 8]]
```

---

## 12.5 数组变形

数组变形操作在深度学习中频繁使用，例如将图像数据从 `(H, W, C)` 转换为 `(C, H, W)`，或将批量数据展平送入全连接层。

### 12.5.1 reshape

```python
a = np.arange(12)
print(a)        # [ 0  1  2  3  4  5  6  7  8  9 10 11]
print(a.shape)  # (12,)

# reshape 到二维
b = a.reshape(3, 4)
print(b)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# reshape 到三维
c = a.reshape(2, 2, 3)
print(c.shape)  # (2, 2, 3)

# -1 表示自动推断该维度
d = a.reshape(4, -1)   # 等同于 reshape(4, 3)
print(d.shape)  # (4, 3)

e = a.reshape(-1, 6)   # 等同于 reshape(2, 6)
print(e.shape)  # (2, 6)

# reshape 返回视图（共享内存）
b[0, 0] = 999
print(a[0])  # 999（原数组也被修改）
```

### 12.5.2 transpose（转置）

```python
a = np.arange(24).reshape(2, 3, 4)
print(a.shape)  # (2, 3, 4)

# 二维转置
b = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape: (2, 3)
print(b.T)
# [[1 4]
#  [2 5]
#  [3 6]]
print(b.T.shape)  # (3, 2)

# 多维转置：指定轴的排列顺序
# 原始轴：(0, 1, 2) → (2, 3, 4)
# 转置后：(1, 0, 2) → (3, 2, 4)
c = np.transpose(a, (1, 0, 2))
print(c.shape)  # (3, 2, 4)

# 深度学习场景：图像格式转换
# PyTorch 使用 (C, H, W)，matplotlib 使用 (H, W, C)
image_hwc = np.random.rand(64, 64, 3)  # H×W×C
image_chw = np.transpose(image_hwc, (2, 0, 1))  # C×H×W
print(image_hwc.shape)  # (64, 64, 3)
print(image_chw.shape)  # (3, 64, 64)
```

### 12.5.3 flatten 和 ravel

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# flatten：返回一维副本（不共享内存）
flat = a.flatten()
print(flat)       # [1 2 3 4 5 6]
flat[0] = 99
print(a[0, 0])    # 1（原数组不变）

# ravel：返回一维视图（尽量共享内存）
rav = a.ravel()
print(rav)        # [1 2 3 4 5 6]
rav[0] = 99
print(a[0, 0])    # 99（原数组被修改）

# 深度学习场景：将特征图展平为向量
feature_map = np.random.rand(32, 7, 7)   # 批次中单个样本：32通道×7×7
flattened = feature_map.flatten()
print(flattened.shape)  # (1568,)
```

### 12.5.4 squeeze 和 expand_dims

```python
# squeeze：删除大小为1的维度
a = np.array([[[1, 2, 3]]])
print(a.shape)          # (1, 1, 3)
print(np.squeeze(a).shape)  # (3,)

# expand_dims：插入大小为1的维度
b = np.array([1, 2, 3])     # shape: (3,)
print(np.expand_dims(b, axis=0).shape)  # (1, 3)
print(np.expand_dims(b, axis=1).shape)  # (3, 1)

# 深度学习场景：为单张图像添加批次维度
image = np.random.rand(3, 64, 64)      # C×H×W
batch = np.expand_dims(image, axis=0)  # 1×C×H×W
print(batch.shape)  # (1, 3, 64, 64)
```

### 12.5.5 concatenate 和 stack

```python
a = np.array([[1, 2], [3, 4]])   # (2, 2)
b = np.array([[5, 6], [7, 8]])   # (2, 2)

# concatenate：沿已有轴拼接
print(np.concatenate([a, b], axis=0))  # 沿行方向
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

print(np.concatenate([a, b], axis=1))  # 沿列方向
# [[1 2 5 6]
#  [3 4 7 8]]

# stack：沿新轴堆叠
print(np.stack([a, b], axis=0).shape)  # (2, 2, 2)
print(np.stack([a, b], axis=1).shape)  # (2, 2, 2)

# 深度学习场景：将多张图像组成批次
img1 = np.random.rand(3, 32, 32)
img2 = np.random.rand(3, 32, 32)
batch = np.stack([img1, img2], axis=0)
print(batch.shape)  # (2, 3, 32, 32)  → (批量, 通道, 高, 宽)
```

---

## 本章小结

| 操作类别 | 函数/方法 | 说明 |
|----------|-----------|------|
| **创建** | `np.array()` | 从列表/元组创建数组 |
| | `np.zeros()`, `np.ones()` | 全零/全一数组 |
| | `np.arange()` | 等差序列（类似`range`） |
| | `np.linspace()` | 线性间隔序列（指定点数） |
| | `np.random.rand()` | 均匀分布随机数 |
| | `np.random.randn()` | 标准正态分布随机数 |
| **索引** | `a[i]`, `a[i, j]` | 基本索引 |
| | `a[start:stop:step]` | 切片（返回视图） |
| | `a[[0, 2, 4]]` | 花式索引（返回副本） |
| | `a[a > 0]` | 布尔索引（返回副本） |
| **运算** | `+`, `-`, `*`, `/`, `**` | 元素级算术 |
| | `np.sin()`, `np.exp()` | 通用函数 |
| | `np.sum()`, `np.mean()` | 聚合函数（支持axis参数） |
| | `np.dot()`, `@` | 矩阵乘法 |
| **广播** | 自动规则 | 尾部维度对齐，大小为1则扩展 |
| **变形** | `reshape()` | 改变形状（共享内存） |
| | `transpose()`, `.T` | 轴重排 |
| | `flatten()` | 展平为一维（返回副本） |
| | `squeeze()` | 删除大小为1的维度 |
| | `expand_dims()` | 插入大小为1的维度 |
| | `concatenate()` | 沿已有轴拼接 |
| | `stack()` | 沿新轴堆叠 |

---

## 深度学习应用：张量操作基础

深度学习框架PyTorch的`Tensor`与NumPy的`ndarray`在概念和API上高度相似，掌握NumPy可以显著降低学习PyTorch的成本。

### NumPy与PyTorch张量的对应关系

```python
import numpy as np
import torch

# ============================================================
# 1. 创建操作
# ============================================================

# NumPy
np_arr = np.array([1.0, 2.0, 3.0])
np_zeros = np.zeros((3, 4))
np_ones  = np.ones((2, 3))
np_rand  = np.random.rand(2, 3)

# PyTorch（对应写法）
pt_arr   = torch.tensor([1.0, 2.0, 3.0])
pt_zeros = torch.zeros(3, 4)
pt_ones  = torch.ones(2, 3)
pt_rand  = torch.rand(2, 3)

# ============================================================
# 2. 形状与属性
# ============================================================

np_arr2d = np.array([[1, 2, 3], [4, 5, 6]])
pt_arr2d = torch.tensor([[1, 2, 3], [4, 5, 6]])

# NumPy                    PyTorch（等价）
print(np_arr2d.shape)    # print(pt_arr2d.shape)
print(np_arr2d.ndim)     # print(pt_arr2d.dim())
print(np_arr2d.dtype)    # print(pt_arr2d.dtype)

# ============================================================
# 3. 变形操作
# ============================================================

np_x = np.arange(12).reshape(3, 4)
pt_x = torch.arange(12).reshape(3, 4)  # 或 .view(3, 4)

# NumPy reshape    ↔    PyTorch view / reshape
np_y = np_x.reshape(2, 6)
pt_y = pt_x.view(2, 6)          # view 要求内存连续
pt_y = pt_x.reshape(2, 6)       # reshape 更灵活

# NumPy transpose  ↔    PyTorch permute
np_z = np.transpose(np_x, (1, 0))
pt_z = pt_x.permute(1, 0)

# ============================================================
# 4. 互相转换
# ============================================================

# NumPy → PyTorch
np_data = np.array([1.0, 2.0, 3.0])
pt_data = torch.from_numpy(np_data)  # 共享内存！

# PyTorch → NumPy
pt_tensor = torch.tensor([4.0, 5.0, 6.0])
np_back   = pt_tensor.numpy()        # 仅适用于CPU张量

# ============================================================
# 5. 一个完整的前向传播示例（纯NumPy实现）
# ============================================================

class SimpleLinearLayer:
    """用NumPy实现的线性层（全连接层）"""

    def __init__(self, in_features: int, out_features: int):
        # 使用He初始化（适合ReLU激活函数）
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: shape (batch_size, in_features)
        返回: shape (batch_size, out_features)
        """
        return x @ self.W + self.b   # 广播：b 形状 (out,) 自动扩展

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU激活：max(0, x)，用布尔索引实现"""
    return np.maximum(0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的Softmax"""
    x_shifted = x - x.max(axis=1, keepdims=True)   # 防止exp溢出
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# 模拟一个 batch：32个样本，每个样本784维（如MNIST展平图像）
np.random.seed(0)
batch_size = 32
x = np.random.randn(batch_size, 784)

# 两层神经网络
layer1 = SimpleLinearLayer(784, 128)
layer2 = SimpleLinearLayer(128, 10)

# 前向传播
h1 = relu(layer1.forward(x))    # (32, 128)
logits = layer2.forward(h1)      # (32, 10)
probs = softmax(logits)          # (32, 10)，每行加和为1

print(f"输入形状:    {x.shape}")       # (32, 784)
print(f"隐层输出:    {h1.shape}")      # (32, 128)
print(f"预测概率:    {probs.shape}")   # (32, 10)
print(f"概率和验证:  {probs[0].sum():.6f}")  # 应约等于1.0
```

### NumPy与PyTorch API速查对照表

| 操作 | NumPy | PyTorch |
|------|-------|---------|
| 创建全零 | `np.zeros((2,3))` | `torch.zeros(2,3)` |
| 创建全一 | `np.ones((2,3))` | `torch.ones(2,3)` |
| 等差序列 | `np.arange(10)` | `torch.arange(10)` |
| 随机正态 | `np.random.randn(2,3)` | `torch.randn(2,3)` |
| 改变形状 | `a.reshape(3,4)` | `a.view(3,4)` 或 `a.reshape(3,4)` |
| 转置 | `a.T` 或 `np.transpose(a,axes)` | `a.T` 或 `a.permute(dims)` |
| 矩阵乘法 | `a @ b` 或 `np.dot(a,b)` | `a @ b` 或 `torch.mm(a,b)` |
| 按轴求和 | `np.sum(a, axis=0)` | `a.sum(dim=0)` |
| 按轴均值 | `np.mean(a, axis=1)` | `a.mean(dim=1)` |
| 展平 | `a.flatten()` | `a.flatten()` 或 `a.view(-1)` |
| 插入维度 | `np.expand_dims(a,0)` | `a.unsqueeze(0)` |
| 删除维度 | `np.squeeze(a)` | `a.squeeze()` |
| 拼接 | `np.concatenate([a,b],axis=0)` | `torch.cat([a,b],dim=0)` |
| 堆叠 | `np.stack([a,b],axis=0)` | `torch.stack([a,b],dim=0)` |

---

## 练习题

### 基础题

**练习1：数组创建与属性**

创建一个形状为 `(4, 5)` 的数组，满足以下要求：
- 第一行：0到4的整数
- 第二行：5到9的整数
- 第三行：全零
- 第四行：全一

然后打印该数组的`shape`、`dtype`、`size`属性。

**练习2：切片与布尔索引**

给定数组：
```python
data = np.array([[3, 1, 4, 1, 5],
                 [9, 2, 6, 5, 3],
                 [5, 8, 9, 7, 9],
                 [3, 2, 3, 8, 4]])
```

完成以下操作：
1. 提取右下角 `2×3` 的子数组（第2-3行，第2-4列）
2. 找出所有大于6的元素，并将其替换为0
3. 计算每行的最大值

---

### 进阶题

**练习3：广播机制实现批量归一化**

给定一个形状为 `(100, 4)` 的随机矩阵（模拟100个样本，4个特征），使用广播机制完成：
1. 计算每个特征的均值和标准差（沿`axis=0`）
2. 对矩阵进行Z-score标准化：`(x - mean) / std`
3. 验证标准化后每列的均值接近0，标准差接近1

**练习4：矩阵运算实现线性回归预测**

给定权重矩阵 `W`（形状 `(3, 2)`）和偏置向量 `b`（形状 `(2,)`），以及输入数据 `X`（形状 `(5, 3)`），完成：
1. 计算线性变换 `Y = X @ W + b`，输出形状应为 `(5, 2)`
2. 对结果应用ReLU激活：`max(0, Y)`
3. 计算激活后每列的均值

```python
np.random.seed(7)
W = np.random.randn(3, 2)
b = np.array([0.5, -0.5])
X = np.random.randn(5, 3)
```

---

### 挑战题

**练习5：用NumPy实现卷积操作（2D滑动窗口）**

实现一个简单的2D卷积函数，不使用任何深度学习框架：

```python
def conv2d_naive(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    对单通道图像进行2D卷积（无填充，步长为1）

    参数：
        image:  形状 (H, W) 的输入图像
        kernel: 形状 (kH, kW) 的卷积核
    返回：
        形状 (H - kH + 1, W - kW + 1) 的输出特征图
    """
    # 你的实现
    pass
```

测试用例：
```python
image = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]], dtype=float)

# Sobel水平边缘检测核
kernel = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=float)

result = conv2d_naive(image, kernel)
print(result)
# 期望输出：
# [[40. 40. 40.]
#  [40. 40. 40.]]
```

提示：使用双重循环遍历输出的每个位置，利用切片提取对应的图像patch，再与kernel做元素级乘法后求和。

---

## 练习答案

### 答案1

```python
import numpy as np

# 方法一：逐行构建
row0 = np.arange(5)         # [0 1 2 3 4]
row1 = np.arange(5, 10)     # [5 6 7 8 9]
row2 = np.zeros(5)          # [0. 0. 0. 0. 0.]
row3 = np.ones(5)           # [1. 1. 1. 1. 1.]

# 注意类型统一
arr = np.array([row0, row1, row2, row3], dtype=float)
print(arr)
# [[ 0.  1.  2.  3.  4.]
#  [ 5.  6.  7.  8.  9.]
#  [ 0.  0.  0.  0.  0.]
#  [ 1.  1.  1.  1.  1.]]

print(f"shape: {arr.shape}")  # (4, 5)
print(f"dtype: {arr.dtype}")  # float64
print(f"size:  {arr.size}")   # 20

# 方法二：先创建再赋值
arr2 = np.zeros((4, 5))
arr2[0] = np.arange(5)
arr2[1] = np.arange(5, 10)
arr2[3] = 1
print(arr2)
```

### 答案2

```python
import numpy as np

data = np.array([[3, 1, 4, 1, 5],
                 [9, 2, 6, 5, 3],
                 [5, 8, 9, 7, 9],
                 [3, 2, 3, 8, 4]])

# 1. 提取右下角 2×3 子数组（第2-3行，第2-4列）
sub = data[2:4, 2:5]
print("右下角子数组：")
print(sub)
# [[ 9  7  9]
#  [ 3  8  4]]

# 2. 将所有大于6的元素替换为0
data_copy = data.copy()
data_copy[data_copy > 6] = 0
print("\n大于6的元素置零后：")
print(data_copy)
# [[3 1 4 1 5]
#  [0 2 6 5 3]
#  [5 0 0 0 0]
#  [3 2 3 0 4]]

# 3. 计算每行最大值
row_max = np.max(data, axis=1)
print(f"\n每行最大值：{row_max}")  # [5 9 9 8]
```

### 答案3

```python
import numpy as np

np.random.seed(42)
X = np.random.randn(100, 4)

# 1. 计算每个特征的均值和标准差
mean = X.mean(axis=0)   # shape: (4,)
std  = X.std(axis=0)    # shape: (4,)
print(f"均值: {mean.round(3)}")
print(f"标准差: {std.round(3)}")

# 2. Z-score标准化（广播自动处理）
X_norm = (X - mean) / std   # (100,4) - (4,) → 广播为 (100,4)

# 3. 验证
print(f"\n标准化后均值:    {X_norm.mean(axis=0).round(6)}")
# 应接近 [0. 0. 0. 0.]

print(f"标准化后标准差:  {X_norm.std(axis=0).round(6)}")
# 应接近 [1. 1. 1. 1.]
```

### 答案4

```python
import numpy as np

np.random.seed(7)
W = np.random.randn(3, 2)
b = np.array([0.5, -0.5])
X = np.random.randn(5, 3)

# 1. 线性变换
Y = X @ W + b       # (5,3)@(3,2) + (2,) → (5,2)
print(f"线性变换输出形状: {Y.shape}")  # (5, 2)
print(f"线性变换结果:\n{Y.round(3)}")

# 2. ReLU激活
Y_relu = np.maximum(0, Y)
print(f"\nReLU后:\n{Y_relu.round(3)}")

# 3. 每列均值
col_mean = Y_relu.mean(axis=0)
print(f"\n激活后每列均值: {col_mean.round(4)}")
```

### 答案5

```python
import numpy as np

def conv2d_naive(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    对单通道图像进行2D卷积（无填充，步长为1）
    """
    H, W   = image.shape
    kH, kW = kernel.shape

    # 输出特征图尺寸
    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            # 提取与卷积核对齐的patch
            patch = image[i:i+kH, j:j+kW]
            # 元素级乘法后求和（即卷积）
            output[i, j] = np.sum(patch * kernel)

    return output


# 测试
image = np.array([[1,  2,  3,  4],
                  [5,  6,  7,  8],
                  [9,  10, 11, 12],
                  [13, 14, 15, 16]], dtype=float)

kernel = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=float)  # Sobel水平边缘检测

result = conv2d_naive(image, kernel)
print("卷积结果:")
print(result)
# [[40. 40. 40.]
#  [40. 40. 40.]]

# 扩展：验证与scipy的结果一致
from scipy.signal import convolve2d
# scipy使用相关操作，需要翻转kernel
result_scipy = convolve2d(image, kernel[::-1, ::-1], mode='valid')
print(f"\n与scipy结果一致: {np.allclose(result, result_scipy)}")

# 扩展：高效实现（使用stride_tricks避免显式循环）
def conv2d_fast(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """使用as_strided实现的高效版本"""
    from numpy.lib.stride_tricks import as_strided

    H, W   = image.shape
    kH, kW = kernel.shape
    out_H  = H - kH + 1
    out_W  = W - kW + 1

    # 构造滑动窗口视图
    shape = (out_H, out_W, kH, kW)
    strides = image.strides + image.strides
    patches = as_strided(image, shape=shape, strides=strides)

    # 批量点积
    return np.einsum('ijkl,kl->ij', patches, kernel)

result_fast = conv2d_fast(image, kernel)
print(f"快速版本结果一致: {np.allclose(result, result_fast)}")
```

---

> **下一章预告**：第13章将介绍Pandas数据处理，学习如何使用DataFrame进行数据清洗、分析和可视化，为机器学习的数据预处理阶段做好准备。
