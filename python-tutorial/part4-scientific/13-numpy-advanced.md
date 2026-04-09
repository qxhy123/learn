# 第13章：NumPy高级应用

## 学习目标

完成本章学习后，你将能够：

1. 熟练使用布尔索引和花式索引对数组进行高级选取与修改
2. 调用NumPy线性代数模块完成矩阵乘法、方程组求解和特征值分解
3. 使用`numpy.random`模块生成各类分布的随机数，并控制随机种子
4. 运用NumPy统计函数对数据集进行描述性统计分析
5. 理解向量化计算与内存布局对性能的影响，写出高效的NumPy代码

---

## 13.1 高级索引：布尔索引与花式索引

NumPy提供了远比Python列表更强大的索引机制。除基本的切片索引外，还支持**布尔索引**（Boolean Indexing）和**花式索引**（Fancy Indexing），这两类索引在数据清洗和深度学习数据预处理中极为常用。

### 13.1.1 布尔索引

布尔索引通过一个与原数组形状相同（或可广播）的布尔数组来选取元素。

```python
import numpy as np

scores = np.array([88, 42, 95, 61, 73, 50, 99, 38])

# 创建布尔掩码
mask = scores >= 60
print("掩码:", mask)
# 掩码: [ True False  True  True  True False  True False]

# 用掩码选取元素
passing = scores[mask]
print("及格成绩:", passing)
# 及格成绩: [88 95 61 73 99]

# 也可以直接在方括号内写条件
print("高于均值的成绩:", scores[scores > scores.mean()])
# 高于均值的成绩: [88 95 73 99]
```

布尔索引可以与赋值结合，实现条件修改：

```python
data = np.array([3.5, -1.2, 0.0, 4.8, -0.5, 2.1])

# 将所有负值替换为0（ReLU激活函数的核心操作）
data[data < 0] = 0
print(data)
# [3.5 0.  0.  4.8 0.  2.1]
```

**多条件组合**使用位运算符`&`、`|`、`~`（不能用`and`/`or`）：

```python
x = np.arange(20)

# 选取10到15之间（包含端点）的元素
result = x[(x >= 10) & (x <= 15)]
print(result)
# [10 11 12 13 14 15]

# 选取小于3或大于16的元素
result2 = x[(x < 3) | (x > 16)]
print(result2)
# [ 0  1  2 17 18 19]
```

**二维数组的布尔索引**：

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 选取所有大于5的元素（返回一维数组）
print(matrix[matrix > 5])
# [6 7 8 9]

# 按行过滤：选取行和大于10的行
row_sums = matrix.sum(axis=1)
print("各行之和:", row_sums)
# 各行之和: [ 6 15 24]
print(matrix[row_sums > 10])
# [[4 5 6]
#  [7 8 9]]
```

`np.where` 是布尔索引的扩展形式，可实现三元运算：

```python
a = np.array([10, -3, 5, -8, 2])

# np.where(condition, x, y)：条件为True时取x，否则取y
result = np.where(a > 0, a, 0)
print(result)
# [10  0  5  0  2]

# 返回满足条件的下标
indices = np.where(a > 0)
print("正数的下标:", indices)
# 正数的下标: (array([0, 2, 4]),)
```

### 13.1.2 花式索引

花式索引使用整数数组或整数列表作为下标，可以按任意顺序选取元素，也可以重复选取。

```python
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80])

# 用整数列表索引
idx = [2, 5, 1, 7]
print(arr[idx])
# [30 60 20 80]

# 可以重复选取
print(arr[[0, 0, 3, 3]])
# [10 10 40 40]
```

**二维数组的花式索引**：

```python
matrix = np.arange(16).reshape(4, 4)
print(matrix)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]

# 选取第0、2、3行
print(matrix[[0, 2, 3]])
# [[ 0  1  2  3]
#  [ 8  9 10 11]
#  [12 13 14 15]]

# 同时指定行和列下标（选取对角线元素）
rows = [0, 1, 2, 3]
cols = [3, 2, 1, 0]
print(matrix[rows, cols])
# [ 3  6  9 12]
```

**花式索引与赋值**：

```python
arr = np.zeros(8, dtype=int)
arr[[1, 3, 5]] = [100, 200, 300]
print(arr)
# [  0 100   0 200   0 300   0   0]
```

**`np.ix_` 构建开放网格索引**，用于选取子矩阵：

```python
matrix = np.arange(25).reshape(5, 5)

# 选取第1、3行与第0、2、4列构成的子矩阵
rows = np.array([1, 3])
cols = np.array([0, 2, 4])
sub = matrix[np.ix_(rows, cols)]
print(sub)
# [[ 5  7  9]
#  [15 17 19]]
```

### 13.1.3 索引类型对比

| 索引类型 | 语法示例 | 返回类型 | 是否共享内存 |
|----------|----------|----------|--------------|
| 基本切片 | `a[1:4]` | 视图（View） | 是 |
| 布尔索引 | `a[a > 0]` | 副本（Copy） | 否 |
| 花式索引 | `a[[1, 3]]` | 副本（Copy） | 否 |

> **注意**：布尔索引和花式索引返回的都是**副本**，修改结果不会影响原数组。

---

## 13.2 线性代数

`numpy.linalg` 模块提供了完整的线性代数功能，是数值计算和机器学习的基础工具。

### 13.2.1 矩阵乘法：dot 与 matmul

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 方法一：np.dot
C1 = np.dot(A, B)

# 方法二：np.matmul / @ 运算符（Python 3.5+）
C2 = np.matmul(A, B)
C3 = A @ B

print(C1)
# [[19 22]
#  [43 50]]
print(np.array_equal(C1, C2), np.array_equal(C1, C3))
# True True
```

`dot` 与 `matmul` 的区别在于处理批量矩阵时：

```python
# 批量矩阵乘法（batch matrix multiplication）
batch_A = np.random.randn(10, 3, 4)  # 10个 3x4 矩阵
batch_B = np.random.randn(10, 4, 5)  # 10个 4x5 矩阵

# matmul 支持批量操作
result = np.matmul(batch_A, batch_B)
print(result.shape)  # (10, 3, 5)

# np.einsum 更灵活地表达批量操作
result2 = np.einsum('bij,bjk->bik', batch_A, batch_B)
print(result2.shape)  # (10, 3, 5)
```

### 13.2.2 求逆、行列式与秩

```python
from numpy import linalg as LA

A = np.array([[2.0, 1.0], [5.0, 3.0]])

# 矩阵求逆
A_inv = LA.inv(A)
print("逆矩阵:\n", A_inv)
# [[ 3. -1.]
#  [-5.  2.]]

# 验证：A @ A_inv ≈ 单位矩阵
print("验证:\n", np.round(A @ A_inv, decimals=10))
# [[1. 0.]
#  [0. 1.]]

# 行列式
print("行列式:", LA.det(A))
# 行列式: 1.0

# 矩阵的秩
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("矩阵的秩:", LA.matrix_rank(M))
# 矩阵的秩: 2（因为第三行 = 2*第二行 - 第一行）
```

### 13.2.3 线性方程组求解：solve

`LA.solve(A, b)` 求解线性方程组 `Ax = b`，比先求逆再相乘数值更稳定且更快。

```python
# 求解方程组：
# 2x + y  = 5
# 5x + 3y = 13
A = np.array([[2.0, 1.0],
              [5.0, 3.0]])
b = np.array([5.0, 13.0])

x = LA.solve(A, b)
print("解:", x)
# 解: [2. 1.]

# 验证
print("验证 Ax =", A @ x)
# 验证 Ax = [ 5. 13.]
```

**最小二乘解**（方程组无精确解时）：

```python
# 过定方程组（方程数 > 未知数）
A = np.array([[1.0, 1.0],
              [1.0, 2.0],
              [1.0, 3.0]])
b = np.array([6.0, 5.0, 7.0])

# lstsq 返回：(解, 残差, 秩, 奇异值)
x, residuals, rank, sv = LA.lstsq(A, b, rcond=None)
print("最小二乘解:", x)
# 最小二乘解: [5.  0.5]
```

### 13.2.4 特征值分解：eig

特征值分解在PCA（主成分分析）、谱聚类等算法中至关重要。

```python
# 对称矩阵的特征值分解
S = np.array([[4.0, 2.0],
              [2.0, 3.0]])

# eig 返回 (特征值数组, 特征向量矩阵)
eigenvalues, eigenvectors = LA.eig(S)
print("特征值:", eigenvalues)
# 特征值: [5.56155281 1.43844719]
print("特征向量（列向量）:\n", eigenvectors)

# 验证：S @ v = λ * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lv = eigenvalues[i] * v
    Sv = S @ v
    print(f"λ{i}验证:", np.allclose(Sv, lv))
    # True

# 对于对称矩阵，推荐用 eigh（更稳定，保证实数结果）
eigenvalues_h, eigenvectors_h = LA.eigh(S)
print("eigh特征值:", eigenvalues_h)
```

### 13.2.5 奇异值分解：SVD

SVD是矩阵分解的核心工具，用于降维、数据压缩和推荐系统。

```python
# 构造一个 4x3 的矩阵
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10,11,12]], dtype=float)

# full_matrices=False 返回经济型SVD
U, s, Vt = LA.svd(M, full_matrices=False)
print("U shape:", U.shape)   # (4, 3)
print("s shape:", s.shape)   # (3,)
print("Vt shape:", Vt.shape) # (3, 3)

# 重建矩阵
M_reconstructed = U @ np.diag(s) @ Vt
print("重建误差:", np.max(np.abs(M - M_reconstructed)))
# 重建误差: 约 1e-14

# 低秩近似（保留前k个奇异值）
k = 1
M_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
print("低秩近似:\n", np.round(M_approx, 2))
```

---

## 13.3 随机数生成

NumPy的`random`模块提供了丰富的随机数生成功能，在深度学习中用于权重初始化、数据增强和随机采样。

### 13.3.1 新版API：Generator

NumPy 1.17+ 推荐使用`numpy.random.default_rng()`创建`Generator`对象，比旧版全局函数更灵活：

```python
import numpy as np

# 创建随机数生成器（可选固定种子保证可重复性）
rng = np.random.default_rng(seed=42)

# 均匀分布 [0, 1)
u = rng.random(size=5)
print("均匀分布:", u)
# 均匀分布: [0.77395605 0.43887844 0.85859792 0.69736803 0.09417735]

# 指定范围的均匀分布 [low, high)
u2 = rng.uniform(low=-1.0, high=1.0, size=(3, 3))
print("[-1,1)均匀分布:\n", np.round(u2, 3))
```

### 13.3.2 常用分布

**正态分布（高斯分布）**：

```python
rng = np.random.default_rng(seed=0)

# 标准正态分布 N(0, 1)
z = rng.standard_normal(size=10000)
print(f"均值: {z.mean():.4f}, 标准差: {z.std():.4f}")
# 均值: -0.0045, 标准差: 0.9999

# 指定均值和标准差的正态分布
x = rng.normal(loc=5.0, scale=2.0, size=(100, 10))
print("shape:", x.shape)
print(f"均值: {x.mean():.2f}, 标准差: {x.std():.2f}")
```

**整数与离散分布**：

```python
rng = np.random.default_rng(seed=1)

# 随机整数 [0, 10)
ints = rng.integers(low=0, high=10, size=8)
print("随机整数:", ints)
# 随机整数: [9 6 2 6 7 4 3 3]

# 二项分布：n次试验中成功k次的概率
binomial = rng.binomial(n=10, p=0.3, size=5)
print("二项分布:", binomial)

# 泊松分布
poisson = rng.poisson(lam=3.0, size=5)
print("泊松分布:", poisson)
```

**选择与打乱**：

```python
rng = np.random.default_rng(seed=7)

arr = np.arange(10)

# 随机打乱（in-place）
rng.shuffle(arr)
print("打乱后:", arr)

# 随机抽样（不放回）
sample = rng.choice(arr, size=4, replace=False)
print("不放回抽样:", sample)

# 随机抽样（放回）
sample2 = rng.choice(arr, size=4, replace=True)
print("放回抽样:", sample2)

# 按权重抽样
weights = np.array([0.1, 0.4, 0.3, 0.2])
items = ['A', 'B', 'C', 'D']
result = rng.choice(items, size=10, p=weights)
from collections import Counter
print("加权抽样分布:", Counter(result))
```

### 13.3.3 旧版API（兼容参考）

```python
# 旧版API（仍可用，但不推荐用于新代码）
np.random.seed(42)
print(np.random.rand(3))      # 均匀分布
print(np.random.randn(3))     # 标准正态
print(np.random.randint(0, 10, size=5))  # 随机整数
```

### 13.3.4 随机种子的重要性

```python
# 固定种子保证实验可重复
def train_model_init(seed):
    rng = np.random.default_rng(seed=seed)
    weights = rng.normal(0, 0.01, size=(784, 256))
    return weights

w1 = train_model_init(seed=42)
w2 = train_model_init(seed=42)
w3 = train_model_init(seed=99)

print("相同种子结果一致:", np.array_equal(w1, w2))   # True
print("不同种子结果不同:", not np.array_equal(w1, w3)) # True
```

---

## 13.4 统计函数

NumPy提供了丰富的统计函数，支持沿指定轴进行聚合计算。

### 13.4.1 基本统计量

```python
import numpy as np

data = np.array([
    [2, 4, 6, 8],
    [1, 3, 5, 7],
    [10, 0, 3, 5]
], dtype=float)

# 均值
print("全局均值:", data.mean())
print("列均值 (axis=0):", data.mean(axis=0))
print("行均值 (axis=1):", data.mean(axis=1))

# 标准差和方差
print("全局标准差:", data.std())
print("列标准差:", data.std(axis=0))

# ddof=1 使用无偏估计（样本标准差）
print("样本标准差:", data.std(axis=0, ddof=1))

# 中位数
print("中位数:", np.median(data))
print("列中位数:", np.median(data, axis=0))
```

### 13.4.2 极值与求和

```python
# 最大值、最小值
print("最大值:", data.max())
print("列最小值:", data.min(axis=0))

# 最大/小值的下标
print("最大值位置:", np.argmax(data))         # 展开后的索引
print("列最大值位置:", np.argmax(data, axis=0)) # 每列最大值所在行

# 累积和与累积乘积
arr = np.array([1, 2, 3, 4, 5])
print("累积和:", np.cumsum(arr))   # [ 1  3  6 10 15]
print("累积积:", np.cumprod(arr))  # [  1   2   6  24 120]

# 求和、乘积
print("全局求和:", data.sum())
print("行求和:", data.sum(axis=1))
```

### 13.4.3 百分位数与分位数

```python
scores = np.array([55, 70, 85, 90, 45, 78, 92, 63, 88, 75])

# 百分位数
p25 = np.percentile(scores, 25)   # 第一四分位数
p50 = np.percentile(scores, 50)   # 中位数
p75 = np.percentile(scores, 75)   # 第三四分位数
print(f"Q1={p25}, Q2={p50}, Q3={p75}")
# Q1=67.25, Q2=76.5, Q3=88.25

# 四分位距（IQR）——用于异常值检测
iqr = p75 - p25
lower_fence = p25 - 1.5 * iqr
upper_fence = p75 + 1.5 * iqr
print(f"IQR={iqr:.2f}, 下界={lower_fence:.2f}, 上界={upper_fence:.2f}")

# np.quantile（等价于percentile但用0-1范围）
q = np.quantile(scores, [0.25, 0.5, 0.75])
print("分位数:", q)
```

### 13.4.4 相关性与协方差

```python
# 两个变量的相关系数
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

corr = np.corrcoef(x, y)
print("相关系数矩阵:\n", corr)
# [[1.         0.90453403]
#  [0.90453403 1.        ]]

# 协方差矩阵
cov = np.cov(x, y)
print("协方差矩阵:\n", cov)

# 多变量相关性（对特征矩阵的列计算）
features = np.random.randn(100, 4)
corr_matrix = np.corrcoef(features.T)  # 转置使每行代表一个特征
print("特征相关矩阵shape:", corr_matrix.shape)  # (4, 4)
```

### 13.4.5 直方图统计

```python
data = np.random.normal(0, 1, 1000)

# 计算直方图（返回频次和bin边界）
counts, bin_edges = np.histogram(data, bins=10)
print("各区间频次:", counts)
print("区间边界:", np.round(bin_edges, 2))

# 归一化为概率密度
density, edges = np.histogram(data, bins=20, density=True)
bin_centers = (edges[:-1] + edges[1:]) / 2  # 计算区间中点
```

---

## 13.5 性能优化

NumPy的高性能来自于其底层C实现和内存连续性。理解向量化和内存布局可以写出比朴素Python快几十甚至上百倍的代码。

### 13.5.1 向量化：消灭Python循环

**向量化（Vectorization）**是用NumPy数组操作替代Python显式循环的编程范式。

```python
import numpy as np
import time

n = 1_000_000

# 方法一：Python循环（慢）
def python_loop(n):
    result = []
    for i in range(n):
        result.append(i ** 2 + 2 * i + 1)
    return result

# 方法二：NumPy向量化（快）
def numpy_vectorized(n):
    x = np.arange(n)
    return x ** 2 + 2 * x + 1

# 性能对比
t0 = time.time()
r1 = python_loop(n)
print(f"Python循环: {time.time()-t0:.3f}s")

t0 = time.time()
r2 = numpy_vectorized(n)
print(f"NumPy向量化: {time.time()-t0:.3f}s")
# Python循环: 约0.300s
# NumPy向量化: 约0.005s  （快约60倍）
```

**通用函数（ufunc）**是向量化的基础，支持逐元素操作和广播：

```python
x = np.linspace(-np.pi, np.pi, 1000000)

# 以下操作均为向量化
sin_x = np.sin(x)
cos_x = np.cos(x)
exp_x = np.exp(-x**2)

# 复合表达式同样向量化
result = np.sin(x) * np.exp(-x**2 / 2)
```

**避免不必要的中间数组**：

```python
# 低效：创建多个中间数组
a, b, c = np.random.randn(1000000), np.random.randn(1000000), np.random.randn(1000000)

t0 = time.time()
result = a + b + c  # 创建两个临时数组
print(f"朴素: {time.time()-t0:.4f}s")

# 高效：使用 np.add 的 out 参数，原地操作
result2 = np.empty_like(a)
t0 = time.time()
np.add(a, b, out=result2)
np.add(result2, c, out=result2)  # 原地加
print(f"原地: {time.time()-t0:.4f}s")
```

### 13.5.2 广播（Broadcasting）

广播允许不同形状的数组之间进行运算，避免手动复制数据。

```python
# 广播规则：从尾部维度对齐，维度为1可扩展
A = np.ones((3, 4, 5))
b = np.ones((4, 5))     # 可广播到 (3, 4, 5)
c = np.ones((5,))        # 可广播到 (3, 4, 5)

print((A + b).shape)    # (3, 4, 5)
print((A + c).shape)    # (3, 4, 5)

# 实用示例：批量标准化
data = np.random.randn(100, 20)  # 100个样本，20个特征
mean = data.mean(axis=0)         # 每个特征的均值，shape (20,)
std  = data.std(axis=0)          # 每个特征的标准差，shape (20,)

# 广播实现标准化（无需循环）
normalized = (data - mean) / std
print(normalized.mean(axis=0)[:5])  # 约 [0, 0, 0, 0, 0]
print(normalized.std(axis=0)[:5])   # 约 [1, 1, 1, 1, 1]
```

### 13.5.3 内存布局：C序与Fortran序

数组在内存中的存储顺序影响缓存效率，进而影响性能。

```python
# C序（行优先，默认）vs Fortran序（列优先）
A_C = np.array([[1, 2, 3], [4, 5, 6]], order='C')
A_F = np.array([[1, 2, 3], [4, 5, 6]], order='F')

print("C序连续:", A_C.flags['C_CONTIGUOUS'])  # True
print("F序连续:", A_F.flags['F_CONTIGUOUS'])  # True
print("C序strides:", A_C.strides)             # (24, 8) 行跨度 > 列跨度
print("F序strides:", A_F.strides)             # (8, 16) 列跨度 < 行跨度

# 性能影响：沿连续内存方向遍历更快
n = 3000
M = np.random.randn(n, n)

t0 = time.time()
_ = M.sum(axis=1)  # 沿行求和，C序数组的高效方向
print(f"行求和 (C序): {time.time()-t0:.4f}s")

M_F = np.asfortranarray(M)
t0 = time.time()
_ = M_F.sum(axis=0)  # 沿列求和，F序数组的高效方向
print(f"列求和 (F序): {time.time()-t0:.4f}s")
```

**视图与副本的内存管理**：

```python
original = np.arange(12).reshape(3, 4)

# 切片返回视图（共享内存，无拷贝开销）
view = original[1:3, 1:3]
view[0, 0] = 999
print(original)   # original 也被修改了

# 花式索引返回副本
copy = original[[0, 2]]
copy[0, 0] = -1
print(original[0, 0])  # original 不受影响

# 检查是否共享内存
print(np.shares_memory(view, original))  # True
print(np.shares_memory(copy, original))  # False
```

### 13.5.4 使用 np.einsum 表达复杂张量运算

`einsum` 提供了简洁表达复杂矩阵/张量运算的方式，有时比组合多个函数更高效：

```python
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# 矩阵乘法
C1 = np.einsum('ij,jk->ik', A, B)
C2 = A @ B
print("einsum矩阵乘法正确:", np.allclose(C1, C2))

# 批量点积（每个样本的特征与权重的点积）
X = np.random.randn(100, 10)  # 100个样本，10维特征
W = np.random.randn(10)       # 权重向量
dot_products = np.einsum('ij,j->i', X, W)
print("批量点积shape:", dot_products.shape)  # (100,)

# 外积
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
outer = np.einsum('i,j->ij', a, b)
print("外积:\n", outer)
```

### 13.5.5 数据类型选择

```python
# 选择合适的dtype节省内存
x_float64 = np.random.randn(1000000)           # 默认 float64，8字节
x_float32 = x_float64.astype(np.float32)       # float32，4字节

print(f"float64内存: {x_float64.nbytes / 1e6:.1f} MB")  # 8.0 MB
print(f"float32内存: {x_float32.nbytes / 1e6:.1f} MB")  # 4.0 MB

# 深度学习中通常使用 float32（精度足够，速度更快）
weights = np.random.randn(784, 256).astype(np.float32)
print("权重dtype:", weights.dtype)
print("权重内存:", weights.nbytes / 1e3, "KB")
```

---

## 本章小结

| 主题 | 核心函数/方法 | 典型应用场景 |
|------|---------------|--------------|
| 布尔索引 | `arr[arr > 0]`、`np.where` | 数据过滤、条件赋值、ReLU |
| 花式索引 | `arr[[i, j, k]]`、`np.ix_` | 批量采样、子矩阵提取 |
| 矩阵乘法 | `np.dot`、`@`、`np.matmul` | 神经网络前向传播 |
| 线性方程组 | `LA.solve`、`LA.lstsq` | 最小二乘回归 |
| 特征值分解 | `LA.eig`、`LA.eigh` | PCA、谱聚类 |
| 奇异值分解 | `LA.svd` | 降维、矩阵近似 |
| 随机数生成 | `rng.random`、`rng.normal` | 权重初始化、数据增强 |
| 统计函数 | `mean`、`std`、`percentile` | 数据分析、批量归一化 |
| 向量化 | ufunc、广播 | 消灭循环、加速计算 |
| 内存优化 | `order`、`astype`、视图 | 内存节省、缓存友好 |

---

## 深度学习应用：神经网络权重初始化

权重初始化是深度学习中一个重要且经常被忽视的细节。不良的初始化会导致梯度消失/爆炸，使网络无法收敛。本节使用NumPy实现并对比各种初始化策略。

### 背景：为什么初始化很重要

```python
import numpy as np

# 演示：糟糕的初始化导致激活值消亡或爆炸
def simulate_forward_pass(init_scale, n_layers=10, n_neurons=256):
    """模拟多层网络的前向传播，观察激活值变化"""
    x = np.random.randn(100)  # 输入

    activation_stds = [x.std()]
    for _ in range(n_layers):
        W = np.random.randn(len(x), n_neurons) * init_scale
        x = np.tanh(W.T @ x[:W.shape[0]])  # tanh激活
        activation_stds.append(x.std())

    return activation_stds

# 太小的初始化 → 梯度消失
stds_small = simulate_forward_pass(init_scale=0.01)
print("小初始化各层std:", [f"{s:.4f}" for s in stds_small])

# 太大的初始化 → 梯度爆炸/tanh饱和
stds_large = simulate_forward_pass(init_scale=1.0)
print("大初始化各层std:", [f"{s:.4f}" for s in stds_large])
```

### 各种初始化方法的NumPy实现

```python
import numpy as np

class WeightInitializer:
    """神经网络权重初始化方法集合"""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed=seed)

    def zeros(self, shape):
        """零初始化（仅用于偏置项，不用于权重）"""
        return np.zeros(shape, dtype=np.float32)

    def random_normal(self, shape, mean=0.0, std=0.01):
        """小随机正态初始化（简单但不推荐深层网络）"""
        return self.rng.normal(loc=mean, scale=std,
                               size=shape).astype(np.float32)

    def random_uniform(self, shape, low=-0.05, high=0.05):
        """均匀分布初始化"""
        return self.rng.uniform(low=low, high=high,
                                size=shape).astype(np.float32)

    def xavier_uniform(self, shape):
        """
        Xavier / Glorot 均匀初始化（适合 sigmoid/tanh 激活函数）

        原理：保持前向传播中各层激活值的方差不变
        范围：[-limit, limit]，limit = sqrt(6 / (fan_in + fan_out))

        论文：Glorot & Bengio (2010) "Understanding the difficulty of
              training deep feedforward neural networks"
        """
        fan_in, fan_out = self._compute_fans(shape)
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(low=-limit, high=limit,
                                size=shape).astype(np.float32)

    def xavier_normal(self, shape):
        """
        Xavier / Glorot 正态初始化
        标准差：std = sqrt(2 / (fan_in + fan_out))
        """
        fan_in, fan_out = self._compute_fans(shape)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return self.rng.normal(loc=0.0, scale=std,
                               size=shape).astype(np.float32)

    def he_uniform(self, shape):
        """
        He 均匀初始化（适合 ReLU 及其变体激活函数）

        原理：考虑ReLU将一半神经元置零，调整方差
        范围：[-limit, limit]，limit = sqrt(6 / fan_in)

        论文：He et al. (2015) "Delving Deep into Rectifiers"
        """
        fan_in, _ = self._compute_fans(shape)
        limit = np.sqrt(6.0 / fan_in)
        return self.rng.uniform(low=-limit, high=limit,
                                size=shape).astype(np.float32)

    def he_normal(self, shape):
        """
        He 正态初始化（最常用于ReLU网络）
        标准差：std = sqrt(2 / fan_in)
        """
        fan_in, _ = self._compute_fans(shape)
        std = np.sqrt(2.0 / fan_in)
        return self.rng.normal(loc=0.0, scale=std,
                               size=shape).astype(np.float32)

    def lecun_normal(self, shape):
        """
        LeCun 正态初始化（适合 SELU 激活函数）
        标准差：std = sqrt(1 / fan_in)
        """
        fan_in, _ = self._compute_fans(shape)
        std = np.sqrt(1.0 / fan_in)
        return self.rng.normal(loc=0.0, scale=std,
                               size=shape).astype(np.float32)

    def orthogonal(self, shape, gain=1.0):
        """
        正交初始化（适合RNN，有助于缓解梯度消失）

        原理：随机正态矩阵做QR分解，取正交矩阵Q
        """
        flat_shape = (shape[0], int(np.prod(shape[1:])))
        a = self.rng.normal(0.0, 1.0, flat_shape)
        U, _, Vt = np.linalg.svd(a, full_matrices=False)
        q = U if U.shape == flat_shape else Vt
        q = q.reshape(shape)
        return (gain * q).astype(np.float32)

    def _compute_fans(self, shape):
        """计算 fan_in 和 fan_out"""
        if len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) >= 3:
            # 卷积层：shape = (height, width, in_channels, out_channels)
            receptive_field = int(np.prod(shape[:-2]))
            fan_in = shape[-2] * receptive_field
            fan_out = shape[-1] * receptive_field
        else:
            fan_in = fan_out = int(np.sqrt(np.prod(shape)))
        return fan_in, fan_out
```

### 实例化与验证

```python
initializer = WeightInitializer(seed=42)

# 第一层：784输入 -> 256隐藏节点
shape = (784, 256)

methods = {
    'Random Normal (std=0.01)': initializer.random_normal(shape),
    'Xavier Uniform':           initializer.xavier_uniform(shape),
    'Xavier Normal':            initializer.xavier_normal(shape),
    'He Uniform':               initializer.he_uniform(shape),
    'He Normal':                initializer.he_normal(shape),
    'LeCun Normal':             initializer.lecun_normal(shape),
    'Orthogonal':               initializer.orthogonal(shape),
}

print(f"{'方法':<30} {'均值':>8} {'标准差':>8} {'最小值':>8} {'最大值':>8}")
print("-" * 66)
for name, W in methods.items():
    print(f"{name:<30} {W.mean():>8.5f} {W.std():>8.5f} "
          f"{W.min():>8.5f} {W.max():>8.5f}")
```

输出示例：
```
方法                           均值     标准差      最小值      最大值
------------------------------------------------------------------
Random Normal (std=0.01)   0.00001  0.01001  -0.03892  0.03942
Xavier Uniform             0.00002  0.05103  -0.08839  0.08839
Xavier Normal              0.00000  0.05103  -0.20361  0.19843
He Uniform                 0.00003  0.08083  -0.14000  0.14000
He Normal                  0.00001  0.07229  -0.28780  0.28054
LeCun Normal              -0.00001  0.05102  -0.20266  0.19839
Orthogonal                -0.00000  0.05618  -0.21302  0.21116
```

### 验证激活值稳定性

```python
def relu(x):
    return np.maximum(0, x)

def check_activation_stability(W_init_fn, shape=(256, 256),
                                n_layers=20, activation=relu):
    """检验初始化方法下各层激活值的方差稳定性"""
    rng = np.random.default_rng(0)
    x = rng.normal(size=shape[0])
    stds = [x.std()]

    init = WeightInitializer(seed=0)
    for _ in range(n_layers):
        W = W_init_fn(init, shape)
        x = activation(W.T @ x)
        stds.append(x.std())

    return stds

# 对比 He Normal vs Random Normal with ReLU
stds_random = check_activation_stability(
    lambda i, s: i.random_normal(s, std=0.01))
stds_he     = check_activation_stability(
    lambda i, s: i.he_normal(s))

print("Random Normal 各层std (前5层):",
      [f"{s:.4f}" for s in stds_random[:6]])
print("He Normal     各层std (前5层):",
      [f"{s:.4f}" for s in stds_he[:6]])
# Random Normal: 激活值迅速趋向0（梯度消失）
# He Normal:     激活值保持在合理范围
```

### 初始化方法选择指南

```python
# 实践建议：根据激活函数选择初始化
INIT_GUIDE = {
    'sigmoid / tanh': 'Xavier Uniform 或 Xavier Normal',
    'ReLU':           'He Normal（最推荐）',
    'Leaky ReLU':     'He Normal（适当调整gain）',
    'SELU':           'LeCun Normal',
    'RNN / LSTM':     'Orthogonal（循环权重）+ Xavier（输入权重）',
    'Transformer':    'Xavier Normal（通常配合small std）',
}

for activation, recommendation in INIT_GUIDE.items():
    print(f"{activation:<20} → {recommendation}")
```

---

## 练习题

### 基础题

**练习 1**：给定一个形状为 `(6, 5)` 的随机整数矩阵（值域 0-99），完成以下操作：
- (a) 使用布尔索引找出所有大于50的元素，并将其替换为 -1
- (b) 使用花式索引提取第0、2、4行和第1、3列构成的子矩阵
- (c) 找出每列最大值所在的行号

**练习 2**：实现一个函数 `normalize_dataset(X)`，接受形状为 `(n_samples, n_features)` 的数组 `X`，返回经过Z-score标准化的结果。要求：
- 沿样本轴（axis=0）计算均值和标准差
- 对标准差为0的特征保持原值不变（避免除以零）
- 使用NumPy向量化操作，不使用任何Python循环

---

### 中级题

**练习 3**：使用NumPy的线性代数模块实现**主成分分析（PCA）**：
- 接受数据矩阵 `X`（形状 `(n, d)`）和目标维度 `k`
- 步骤：中心化 → 计算协方差矩阵 → 特征值分解 → 选取前k个主成分
- 返回投影后的低维数据（形状 `(n, k)`）和主成分方向
- 用随机生成的二维数据验证（投影到1维）

**练习 4**：实现一个函数 `bootstrap_confidence_interval(data, statistic, n_bootstrap, alpha)`：
- 用**自助法（Bootstrap）**估计统计量的置信区间
- `statistic` 为函数（如 `np.mean`、`np.median`）
- `n_bootstrap` 为重采样次数（如 1000）
- `alpha` 为显著性水平（如 0.05 对应95%置信区间）
- 使用 `numpy.random` 模块的放回抽样，不使用Python循环

---

### 进阶题

**练习 5**：实现一个向量化的 **K-Means 聚类算法**：
- 函数签名：`kmeans(X, k, max_iter=100, seed=42)`
- 返回：每个样本的簇标签和簇中心
- 要求完全使用NumPy向量化实现（距离计算、标签分配、中心更新均不使用Python循环）
- 提示：距离矩阵可以用广播实现：`distances = ||X - centers||^2`
- 在一个包含3个明显类别的合成数据集上验证效果

---

## 练习答案

### 答案 1

```python
import numpy as np

rng = np.random.default_rng(seed=0)
M = rng.integers(0, 100, size=(6, 5))
print("原始矩阵:\n", M)

# (a) 布尔索引替换大于50的元素
M_copy = M.copy()
M_copy[M_copy > 50] = -1
print("\n(a) 替换后:\n", M_copy)

# (b) 花式索引提取子矩阵
rows = np.array([0, 2, 4])
cols = np.array([1, 3])
sub = M[np.ix_(rows, cols)]
print("\n(b) 子矩阵:\n", sub)

# (c) 每列最大值所在的行号
row_indices = np.argmax(M, axis=0)
print("\n(c) 每列最大值所在行:", row_indices)
print("    对应的最大值:", M[row_indices, np.arange(M.shape[1])])
```

### 答案 2

```python
import numpy as np

def normalize_dataset(X):
    """
    Z-score 标准化，避免除以零。

    参数：
        X: ndarray, shape (n_samples, n_features)
    返回：
        X_norm: ndarray, 标准化后的数组
    """
    mean = X.mean(axis=0)          # shape (n_features,)
    std  = X.std(axis=0)           # shape (n_features,)

    # 对标准差为0的特征，用1替换以避免除零（结果为 (x - mean) / 1 = 0）
    std_safe = np.where(std == 0, 1.0, std)

    X_norm = (X - mean) / std_safe
    return X_norm

# 验证
rng = np.random.default_rng(42)
X = rng.normal(loc=[10, 5, 0], scale=[3, 1, 0], size=(100, 3))
# 第3列标准差为0

X_norm = normalize_dataset(X)
print("标准化后各列均值:", np.round(X_norm.mean(axis=0), 4))
# 约 [0, 0, 0]
print("标准化后各列标准差:", np.round(X_norm.std(axis=0), 4))
# 约 [1, 1, 0]（第3列仍为0）
```

### 答案 3

```python
import numpy as np

def pca(X, k):
    """
    主成分分析（PCA）

    参数：
        X: ndarray, shape (n, d)，输入数据
        k: int，目标维度
    返回：
        X_proj: ndarray, shape (n, k)，投影后的数据
        components: ndarray, shape (k, d)，主成分方向（按方差降序）
        explained_ratio: ndarray, shape (k,)，各主成分解释的方差比例
    """
    # Step 1：中心化
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Step 2：计算协方差矩阵
    n = X.shape[0]
    cov = (X_centered.T @ X_centered) / (n - 1)

    # Step 3：特征值分解（eigh 对对称矩阵更稳定）
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Step 4：按特征值降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5：取前k个主成分
    components = eigenvectors[:, :k].T      # shape (k, d)
    X_proj = X_centered @ eigenvectors[:, :k]  # shape (n, k)

    # 解释方差比例
    explained_ratio = eigenvalues[:k] / eigenvalues.sum()

    return X_proj, components, explained_ratio

# 验证：生成包含主方向的2D数据
rng = np.random.default_rng(0)
t = rng.uniform(0, 2*np.pi, 200)
X = np.column_stack([3*np.cos(t) + 0.5*rng.normal(size=200),
                      np.sin(t)  + 0.5*rng.normal(size=200)])

X_proj, components, ratio = pca(X, k=1)
print("投影后shape:", X_proj.shape)           # (200, 1)
print("主成分方向:", np.round(components, 3)) # 应接近 [1, 0] 或 [-1, 0]
print("解释方差比例:", np.round(ratio, 3))    # 约 [0.9]，第1主成分解释90%方差
```

### 答案 4

```python
import numpy as np

def bootstrap_confidence_interval(data, statistic, n_bootstrap=1000, alpha=0.05):
    """
    自助法置信区间

    参数：
        data: 1D ndarray
        statistic: 统计函数，如 np.mean
        n_bootstrap: 重采样次数
        alpha: 显著性水平
    返回：
        (lower, upper): 置信区间端点
        stat_observed: 观测到的统计量
    """
    rng = np.random.default_rng(seed=0)
    n = len(data)

    # 一次性生成所有重采样下标：shape (n_bootstrap, n)
    indices = rng.integers(0, n, size=(n_bootstrap, n))

    # 向量化计算每个Bootstrap样本的统计量
    # data[indices] shape: (n_bootstrap, n)
    bootstrap_stats = statistic(data[indices], axis=1)

    # 计算置信区间
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    stat_observed = statistic(data)

    return (lower, upper), stat_observed

# 验证
rng = np.random.default_rng(42)
data = rng.exponential(scale=2.0, size=200)   # 真实均值 = 2.0

(lo, hi), obs = bootstrap_confidence_interval(data, np.mean, n_bootstrap=2000)
print(f"样本均值: {obs:.4f}")
print(f"95% 置信区间: [{lo:.4f}, {hi:.4f}]")
# 95% 置信区间应包含真实均值 2.0

(lo2, hi2), obs2 = bootstrap_confidence_interval(data, np.median)
print(f"\n样本中位数: {obs2:.4f}")
print(f"95% 置信区间: [{lo2:.4f}, {hi2:.4f}]")
```

### 答案 5

```python
import numpy as np

def kmeans(X, k, max_iter=100, seed=42):
    """
    向量化 K-Means 聚类

    参数：
        X: ndarray, shape (n, d)
        k: int，簇数
        max_iter: 最大迭代次数
        seed: 随机种子
    返回：
        labels: ndarray, shape (n,)，每个样本的簇标签
        centers: ndarray, shape (k, d)，簇中心
    """
    rng = np.random.default_rng(seed=seed)
    n, d = X.shape

    # 随机选取k个初始中心（从数据点中选取）
    init_idx = rng.choice(n, size=k, replace=False)
    centers = X[init_idx].copy()     # shape (k, d)

    labels = np.zeros(n, dtype=int)

    for iteration in range(max_iter):
        # Step 1：计算距离矩阵（向量化）
        # X[:, np.newaxis] shape: (n, 1, d)
        # centers[np.newaxis, :] shape: (1, k, d)
        # diff shape: (n, k, d)
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        distances = (diff ** 2).sum(axis=2)   # shape (n, k)

        # Step 2：分配最近的簇
        new_labels = np.argmin(distances, axis=1)  # shape (n,)

        # Step 3：检查收敛
        if np.array_equal(new_labels, labels):
            print(f"  收敛于第 {iteration} 轮")
            break
        labels = new_labels

        # Step 4：更新中心（向量化）
        # 使用 one-hot 编码加权求均值
        one_hot = (labels[:, np.newaxis] == np.arange(k))  # (n, k)
        counts = one_hot.sum(axis=0)                         # (k,)
        centers = (one_hot.T @ X) / np.maximum(counts[:, np.newaxis], 1)

    return labels, centers

# 验证：生成3个明显分离的高斯簇
rng = np.random.default_rng(0)
cluster_centers_true = np.array([[0, 0], [5, 0], [2.5, 4.5]])
X_parts = [rng.normal(loc=c, scale=0.7, size=(100, 2))
           for c in cluster_centers_true]
X_data = np.vstack(X_parts)
true_labels = np.repeat([0, 1, 2], 100)

# 运行K-Means
labels, centers = kmeans(X_data, k=3)

# 评估（计算准确率，需处理标签置换）
from scipy.optimize import linear_sum_assignment

def clustering_accuracy(true_labels, pred_labels, k):
    """使用匈牙利算法计算聚类准确率"""
    confusion = np.zeros((k, k), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        confusion[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    return confusion[row_ind, col_ind].sum() / len(true_labels)

# 如果没有scipy，用简单方式验证
print("发现的簇中心:\n", np.round(centers, 2))
print("真实簇中心:\n", cluster_centers_true)
print("\n每个簇的样本数:", np.bincount(labels))
# 期望：每个簇约100个样本
```

---

> **本章关键点回顾**
>
> 1. 布尔索引和花式索引是NumPy数据操作的核心工具，返回副本而非视图
> 2. `numpy.linalg` 提供完整的线性代数支持，`solve` 比手动求逆更稳定
> 3. 优先使用 `numpy.random.default_rng()` 而非旧版全局函数
> 4. 向量化是NumPy高性能的关键，用广播和ufunc消灭Python循环
> 5. 权重初始化的选择对深度网络的训练稳定性至关重要，ReLU网络使用He初始化
