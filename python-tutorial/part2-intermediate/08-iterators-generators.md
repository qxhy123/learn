# 第8章：迭代器与生成器

## 学习目标

学完本章后，你将能够：

1. 理解可迭代对象与迭代器的区别，掌握迭代器协议（`__iter__`、`__next__`）的工作原理
2. 实现自定义迭代器类，控制对象的遍历行为
3. 使用 `yield` 关键字编写生成器函数，理解惰性求值的优势
4. 熟练使用生成器表达式创建轻量级数据流
5. 运用 `itertools` 模块中的工具函数处理复杂迭代场景，并在深度学习数据加载中应用生成器技术

---

## 8.1 可迭代对象与迭代器协议

### 8.1.1 什么是可迭代对象

在 Python 中，任何可以在 `for` 循环中使用的对象都是**可迭代对象（Iterable）**。列表、元组、字符串、字典、集合都是可迭代对象。

```python
# 这些都是可迭代对象
for x in [1, 2, 3]:
    print(x)

for ch in "hello":
    print(ch)

for k, v in {"a": 1, "b": 2}.items():
    print(k, v)
```

判断一个对象是否可迭代，可以使用 `hasattr` 检查是否有 `__iter__` 方法，或者使用 `collections.abc.Iterable`：

```python
from collections.abc import Iterable

print(isinstance([1, 2, 3], Iterable))   # True
print(isinstance("hello", Iterable))     # True
print(isinstance(42, Iterable))          # False
print(isinstance(range(10), Iterable))   # True
```

### 8.1.2 迭代器协议

**迭代器（Iterator）**是实现了以下两个方法的对象：

| 方法 | 说明 |
|------|------|
| `__iter__()` | 返回迭代器自身（`self`） |
| `__next__()` | 返回下一个元素；当没有元素时抛出 `StopIteration` |

可迭代对象和迭代器的区别：
- **可迭代对象**：实现了 `__iter__()` 方法，返回一个迭代器
- **迭代器**：同时实现了 `__iter__()` 和 `__next__()` 方法

```python
# 列表是可迭代对象，但不是迭代器
my_list = [1, 2, 3]
print(hasattr(my_list, '__iter__'))   # True
print(hasattr(my_list, '__next__'))   # False

# 用 iter() 从可迭代对象获取迭代器
my_iter = iter(my_list)
print(hasattr(my_iter, '__iter__'))   # True
print(hasattr(my_iter, '__next__'))   # True

# 手动驱动迭代器
print(next(my_iter))  # 1
print(next(my_iter))  # 2
print(next(my_iter))  # 3
# print(next(my_iter))  # 抛出 StopIteration
```

### 8.1.3 for 循环的底层机制

`for` 循环实际上是迭代器协议的语法糖：

```python
# for 循环
for item in [10, 20, 30]:
    print(item)

# 等价的手动实现
_iter = iter([10, 20, 30])
while True:
    try:
        item = next(_iter)
        print(item)
    except StopIteration:
        break
```

理解这一点有助于我们编写更高效的代码，以及设计自定义数据结构的遍历方式。

### 8.1.4 迭代器是一次性的

迭代器只能向前移动，不能重置。一旦耗尽，再次调用 `next()` 始终抛出 `StopIteration`：

```python
numbers = [1, 2, 3]
it = iter(numbers)

# 第一次遍历
for x in it:
    print(x)  # 输出 1 2 3

# 第二次遍历——迭代器已耗尽，不输出任何内容
for x in it:
    print(x)

# 列表本身可以重复遍历，因为每次 for 都会调用 iter() 创建新迭代器
for x in numbers:
    print(x)  # 输出 1 2 3
```

---

## 8.2 自定义迭代器

### 8.2.1 实现一个简单的范围迭代器

通过实现 `__iter__` 和 `__next__`，可以让任何类支持迭代：

```python
class MyRange:
    """模拟 range() 的自定义迭代器"""

    def __init__(self, start, stop, step=1):
        self.current = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        return self  # 迭代器返回自身

    def __next__(self):
        if self.current >= self.stop:
            raise StopIteration
        value = self.current
        self.current += self.step
        return value


# 使用自定义迭代器
for n in MyRange(0, 10, 2):
    print(n, end=' ')
# 输出: 0 2 4 6 8

print()
print(list(MyRange(1, 6)))  # [1, 2, 3, 4, 5]
```

### 8.2.2 分离可迭代对象与迭代器

更好的设计是将**可迭代容器**与**迭代状态**分离，这样同一个容器可以同时被多个迭代器独立遍历：

```python
class NumberSequence:
    """可迭代容器（不保存迭代状态）"""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        # 每次调用都返回一个新的迭代器
        return NumberIterator(self.data)


class NumberIterator:
    """迭代器（保存迭代状态）"""

    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        value = self.data[self.index]
        self.index += 1
        return value


seq = NumberSequence([10, 20, 30, 40])

# 两个迭代器独立工作
it1 = iter(seq)
it2 = iter(seq)

print(next(it1))  # 10
print(next(it1))  # 20
print(next(it2))  # 10  ← 独立的迭代状态
print(next(it1))  # 30
```

### 8.2.3 实际案例：矩阵行迭代器

```python
class Matrix:
    """支持按行迭代的矩阵类"""

    def __init__(self, data):
        self.data = data  # 二维列表
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0

    def __iter__(self):
        """默认按行迭代"""
        return iter(self.data)

    def __repr__(self):
        rows = '\n'.join(['  ' + str(row) for row in self.data])
        return f"Matrix(\n{rows}\n)"

    def rows_iter(self):
        """行迭代器"""
        return iter(self.data)

    def cols_iter(self):
        """列迭代器"""
        for col_idx in range(self.cols):
            yield [self.data[row_idx][col_idx] for row_idx in range(self.rows)]

    def elements_iter(self):
        """所有元素的迭代器（按行展开）"""
        for row in self.data:
            for elem in row:
                yield elem


m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("按行遍历:")
for row in m:
    print(row)

print("\n按列遍历:")
for col in m.cols_iter():
    print(col)

print("\n所有元素:")
print(list(m.elements_iter()))
```

---

## 8.3 生成器函数与 yield

### 8.3.1 生成器函数简介

**生成器函数（Generator Function）**是包含 `yield` 语句的函数。调用生成器函数不会立即执行，而是返回一个**生成器对象**（同时也是迭代器）。

```python
def simple_generator():
    print("开始执行")
    yield 1
    print("产出第一个值后继续")
    yield 2
    print("产出第二个值后继续")
    yield 3
    print("函数结束")


gen = simple_generator()
print(type(gen))          # <class 'generator'>

print(next(gen))          # 开始执行 → 1
print(next(gen))          # 产出第一个值后继续 → 2
print(next(gen))          # 产出第二个值后继续 → 3
# next(gen)               # 函数结束 → StopIteration
```

每次调用 `next()` 时，函数从上次暂停的 `yield` 处继续执行，直到遇到下一个 `yield`。

### 8.3.2 yield 与 return 的对比

| 特性 | `return` | `yield` |
|------|----------|---------|
| 执行后 | 函数终止 | 函数暂停，保留状态 |
| 返回值 | 单个值 | 可以产出多个值 |
| 内存占用 | 一次性计算所有结果 | 惰性计算，按需产出 |
| 重复调用 | 从头开始 | 从上次暂停处继续 |

### 8.3.3 用生成器实现无限序列

生成器非常适合表示无限序列，因为它们是惰性求值的：

```python
def fibonacci():
    """无限斐波那契数列生成器"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def take(n, iterable):
    """取前 n 个元素"""
    count = 0
    for item in iterable:
        if count >= n:
            break
        yield item
        count += 1


fib = fibonacci()
print(list(take(10, fib)))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

```python
def natural_numbers(start=1):
    """自然数生成器"""
    n = start
    while True:
        yield n
        n += 1


# 生成前20个奇数
odd_numbers = (x for x in natural_numbers() if x % 2 != 0)
print(list(take(20, odd_numbers)))
```

### 8.3.4 生成器的内存优势

对比列表与生成器处理大数据的内存占用：

```python
import sys

# 列表：一次性占用所有内存
big_list = [x ** 2 for x in range(1_000_000)]
print(f"列表内存: {sys.getsizeof(big_list):,} bytes")  # ~8,000,056 bytes

# 生成器：几乎不占额外内存
big_gen = (x ** 2 for x in range(1_000_000))
print(f"生成器内存: {sys.getsizeof(big_gen):,} bytes")  # ~104 bytes

# 对大数据求和：生成器效率远高于列表
total = sum(x ** 2 for x in range(1_000_000))
print(f"总和: {total:,}")
```

### 8.3.5 yield from 委托子生成器

`yield from` 可以将迭代委托给另一个可迭代对象，简化嵌套生成器：

```python
def flatten(nested):
    """展平嵌套列表"""
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)  # 递归委托
        else:
            yield item


data = [1, [2, 3], [4, [5, 6]], 7]
print(list(flatten(data)))  # [1, 2, 3, 4, 5, 6, 7]
```

```python
def chain_generators(*iterables):
    """连接多个可迭代对象"""
    for it in iterables:
        yield from it


result = list(chain_generators([1, 2], [3, 4], [5, 6]))
print(result)  # [1, 2, 3, 4, 5, 6]
```

### 8.3.6 生成器的 send() 方法

生成器不仅可以产出值，还可以通过 `send()` 接收外部传入的值：

```python
def accumulator():
    """接受输入并累加"""
    total = 0
    while True:
        value = yield total  # yield 既产出 total，也接收外部值
        if value is None:
            break
        total += value


acc = accumulator()
next(acc)           # 启动生成器（执行到第一个 yield）
print(acc.send(10)) # 20 不对，total先是0，send(10)后total=10，yield total=10
# 修正后：
acc2 = accumulator()
print(next(acc2))    # 0  (初始值)
print(acc2.send(5))  # 5
print(acc2.send(3))  # 8
print(acc2.send(7))  # 15
```

---

## 8.4 生成器表达式

### 8.4.1 语法与列表推导式的对比

生成器表达式与列表推导式语法几乎相同，只是用圆括号代替方括号：

```python
# 列表推导式 — 立即求值，返回列表
squares_list = [x ** 2 for x in range(10)]
print(type(squares_list))   # <class 'list'>
print(squares_list)         # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 生成器表达式 — 惰性求值，返回生成器对象
squares_gen = (x ** 2 for x in range(10))
print(type(squares_gen))    # <class 'generator'>
print(list(squares_gen))    # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### 8.4.2 何时使用生成器表达式

选择原则：

- 需要**多次遍历**或**随机访问** → 用列表推导式
- 只需**遍历一次**或数据量**非常大** → 用生成器表达式
- 作为函数参数（如 `sum()`、`max()`）时 → 优先用生成器表达式

```python
# 在函数调用中，生成器表达式不需要额外括号
data = range(1, 1001)

# 推荐：sum 只遍历一次，用生成器
total = sum(x ** 2 for x in data if x % 3 == 0)

# 不推荐（多余的列表分配）
total = sum([x ** 2 for x in data if x % 3 == 0])

# 查找第一个满足条件的值
first_large = next(x for x in data if x ** 2 > 500)
print(first_large)  # 23
```

### 8.4.3 嵌套生成器表达式

```python
# 笛卡尔积（惰性）
pairs = ((x, y) for x in range(3) for y in range(3) if x != y)
print(list(pairs))
# [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

# 矩阵展平
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = (elem for row in matrix for elem in row)
print(list(flat))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### 8.4.4 生成器管道

生成器最强大的特性是可以组合成**处理管道**，每一步都是惰性的：

```python
def read_lines(filename):
    """逐行读取文件（模拟）"""
    lines = ["  Hello World  ", "  Python 3.12  ", "  Generator  ", "  Pipeline  "]
    yield from lines

def strip_lines(lines):
    """去除空白"""
    return (line.strip() for line in lines)

def to_upper(lines):
    """转大写"""
    return (line.upper() for line in lines)

def filter_short(lines, min_len=6):
    """过滤短字符串"""
    return (line for line in lines if len(line) >= min_len)

# 组合管道 — 整个流程都是惰性的
pipeline = filter_short(to_upper(strip_lines(read_lines("data.txt"))))

for line in pipeline:
    print(line)
# HELLO WORLD
# PYTHON 3.12
# GENERATOR
# PIPELINE
```

---

## 8.5 itertools 模块

`itertools` 是 Python 标准库中专门处理迭代器的模块，提供了高效、内存友好的迭代工具。

### 8.5.1 无限迭代器

```python
import itertools

# count(start, step) — 从 start 开始，步长为 step 的无限计数
counter = itertools.count(10, 2)
print(list(itertools.islice(counter, 5)))  # [10, 12, 14, 16, 18]

# cycle(iterable) — 无限循环一个可迭代对象
cycler = itertools.cycle(['A', 'B', 'C'])
print(list(itertools.islice(cycler, 7)))   # ['A', 'B', 'C', 'A', 'B', 'C', 'A']

# repeat(value, times) — 重复一个值
print(list(itertools.repeat(0, 5)))        # [0, 0, 0, 0, 0]
```

### 8.5.2 终止于最短输入的迭代器

```python
import itertools

# chain(*iterables) — 连接多个可迭代对象
chained = itertools.chain([1, 2], [3, 4], [5])
print(list(chained))  # [1, 2, 3, 4, 5]

# chain.from_iterable — 展平一层嵌套
nested = [[1, 2], [3, 4], [5, 6]]
flat = list(itertools.chain.from_iterable(nested))
print(flat)  # [1, 2, 3, 4, 5, 6]

# islice(iterable, stop) / islice(iterable, start, stop, step)
data = range(100)
print(list(itertools.islice(data, 5)))        # [0, 1, 2, 3, 4]
print(list(itertools.islice(data, 2, 10, 2))) # [2, 4, 6, 8]

# compress(data, selectors) — 根据布尔掩码过滤
data = ['a', 'b', 'c', 'd', 'e']
mask = [1, 0, 1, 0, 1]
print(list(itertools.compress(data, mask)))   # ['a', 'c', 'e']

# takewhile / dropwhile
nums = [1, 3, 5, 2, 4, 6]
print(list(itertools.takewhile(lambda x: x < 4, nums)))  # [1, 3]
print(list(itertools.dropwhile(lambda x: x < 4, nums)))  # [2, 4, 6]

# zip_longest — 等长补齐版 zip
a = [1, 2, 3]
b = ['x', 'y']
print(list(itertools.zip_longest(a, b, fillvalue='-')))
# [(1, 'x'), (2, 'y'), (3, '-')]
```

### 8.5.3 组合迭代器

```python
import itertools

# product — 笛卡尔积
print(list(itertools.product([1, 2], ['a', 'b'])))
# [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]

# permutations — 排列（有序，不重复取）
print(list(itertools.permutations([1, 2, 3], 2)))
# [(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)]

# combinations — 组合（无序，不重复取）
print(list(itertools.combinations([1, 2, 3, 4], 2)))
# [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

# combinations_with_replacement — 组合（允许重复取）
print(list(itertools.combinations_with_replacement([1, 2, 3], 2)))
# [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)]
```

### 8.5.4 分组与累积

```python
import itertools

# groupby — 对连续相同键的元素分组（需先排序）
data = [('苹果', '水果'), ('香蕉', '水果'), ('胡萝卜', '蔬菜'), ('白菜', '蔬菜')]
data.sort(key=lambda x: x[1])  # 先按类别排序

for category, group in itertools.groupby(data, key=lambda x: x[1]):
    items = [item[0] for item in group]
    print(f"{category}: {items}")
# 水果: ['苹果', '香蕉']
# 蔬菜: ['胡萝卜', '白菜']

# accumulate — 累积计算
import operator
nums = [1, 2, 3, 4, 5]
print(list(itertools.accumulate(nums)))                          # [1, 3, 6, 10, 15] (累加)
print(list(itertools.accumulate(nums, operator.mul)))            # [1, 2, 6, 24, 120] (累乘)
print(list(itertools.accumulate(nums, max)))                     # [1, 2, 3, 4, 5] (累积最大值)
```

### 8.5.5 实用组合示例：批量处理

```python
import itertools

def batched(iterable, n):
    """将可迭代对象按 n 个一批分割（Python 3.12 内置 itertools.batched）"""
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch


data = range(1, 11)
for batch in batched(data, 3):
    print(batch)
# [1, 2, 3]
# [4, 5, 6]
# [7, 8, 9]
# [10]
```

---

## 本章小结

| 概念 | 关键点 | 适用场景 |
|------|--------|---------|
| 可迭代对象 | 实现 `__iter__()`，返回迭代器 | 容器类（列表、自定义集合等） |
| 迭代器 | 实现 `__iter__()` + `__next__()`，有状态 | 一次性遍历，节省内存 |
| 生成器函数 | 含 `yield`，惰性执行，自动实现迭代器协议 | 无限序列、大数据处理、管道 |
| 生成器表达式 | `(expr for x in iter)` 语法，内存高效 | 一次性计算、函数参数 |
| `yield from` | 委托子生成器，简化嵌套迭代 | 递归结构、连接多个生成器 |
| `itertools` | 标准库迭代工具集，高效且内存友好 | 组合、过滤、分组、无限序列 |

**选择指南：**
- 需要**状态控制**或**复杂逻辑** → 自定义迭代器类
- 需要**惰性求值**或**无限序列** → 生成器函数
- 简单**转换/过滤** → 生成器表达式
- **标准迭代模式**（分批、组合等） → `itertools`

---

## 深度学习应用：数据加载的流水线设计

在深度学习中，训练数据通常远大于内存容量。生成器和迭代器是实现高效**数据加载流水线**的核心技术。

### 应用背景

训练一个图像分类模型时，可能有数百万张图片。如果一次性加载所有图片到内存，通常会导致内存溢出（OOM）。生成器允许我们**按需加载**，每次只在内存中保留当前批次的数据。

### 完整的数据加载流水线实现

```python
import os
import random
import itertools
from pathlib import Path


# ────────────────────────────────────────
# 第1层：数据源生成器 — 枚举文件路径
# ────────────────────────────────────────
def file_path_generator(data_dir, extensions=('.jpg', '.png')):
    """
    递归枚举目录下所有图片路径。
    生成器：惰性遍历，不提前扫描全部文件。
    """
    data_dir = Path(data_dir)
    for path in data_dir.rglob('*'):
        if path.suffix.lower() in extensions:
            yield str(path)


# ────────────────────────────────────────
# 第2层：数据读取生成器 — 按路径加载数据
# ────────────────────────────────────────
def image_loader(path_gen, label_fn=None):
    """
    接收路径生成器，产出 (image_array, label) 对。
    label_fn: 从路径推断标签的函数（如按父目录名）
    """
    for path in path_gen:
        try:
            # 实际场景中使用 PIL 或 cv2 加载
            # image = np.array(Image.open(path))
            # 这里用模拟数据代替
            image = simulate_load_image(path)
            label = label_fn(path) if label_fn else 0
            yield image, label
        except Exception as e:
            print(f"警告：跳过损坏文件 {path}，原因：{e}")
            continue


def simulate_load_image(path):
    """模拟图片加载，返回随机数组"""
    import random
    return [[random.random() for _ in range(3)] for _ in range(4)]  # 4x3 模拟图像


# ────────────────────────────────────────
# 第3层：数据增强生成器 — 变换图像
# ────────────────────────────────────────
def augment_generator(data_gen, augment=True):
    """
    对图像应用随机数据增强。
    augment=False 时直接传递（用于验证集）。
    """
    for image, label in data_gen:
        if augment and random.random() > 0.5:
            image = horizontal_flip(image)
        if augment and random.random() > 0.5:
            image = random_brightness(image)
        yield image, label


def horizontal_flip(image):
    """水平翻转（模拟）"""
    return [row[::-1] for row in image]


def random_brightness(image, factor=None):
    """随机亮度调整（模拟）"""
    if factor is None:
        factor = 0.8 + random.random() * 0.4  # [0.8, 1.2]
    return [[min(1.0, v * factor) for v in row] for row in image]


# ────────────────────────────────────────
# 第4层：归一化生成器
# ────────────────────────────────────────
def normalize_generator(data_gen, mean=0.5, std=0.25):
    """标准化像素值"""
    for image, label in data_gen:
        normalized = [[(v - mean) / std for v in row] for row in image]
        yield normalized, label


# ────────────────────────────────────────
# 第5层：批处理生成器 — 将样本打包为批次
# ────────────────────────────────────────
def batch_generator(data_gen, batch_size=32, drop_last=False):
    """
    将单个样本打包为 mini-batch。
    drop_last=True 时丢弃不足一个批次的尾部数据。
    """
    batch_images = []
    batch_labels = []

    for image, label in data_gen:
        batch_images.append(image)
        batch_labels.append(label)

        if len(batch_images) == batch_size:
            yield batch_images, batch_labels
            batch_images = []
            batch_labels = []

    # 处理最后一个不完整批次
    if batch_images and not drop_last:
        yield batch_images, batch_labels


# ────────────────────────────────────────
# 第6层：洗牌缓冲生成器 — 近似随机打乱
# ────────────────────────────────────────
def shuffle_buffer_generator(data_gen, buffer_size=1000):
    """
    使用缓冲区实现近似洗牌（无法一次性加载全部数据时的折中方案）。
    缓冲区满后随机取出一个样本，并用新样本填充。
    """
    buffer = []

    for item in data_gen:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            idx = random.randrange(len(buffer))
            yield buffer[idx]
            buffer[idx] = buffer[-1]
            buffer.pop()

    # 输出缓冲区剩余元素
    random.shuffle(buffer)
    yield from buffer


# ────────────────────────────────────────
# 组装完整流水线
# ────────────────────────────────────────
def create_training_pipeline(data_dir, batch_size=32, buffer_size=500):
    """
    组装训练数据流水线。
    整个流水线是惰性的——在迭代之前不加载任何数据。
    """
    label_map = {'cat': 0, 'dog': 1, 'bird': 2}

    def get_label(path):
        # 从父目录名推断标签（如 data/cat/img001.jpg → 0）
        parent = Path(path).parent.name
        return label_map.get(parent, -1)

    # 各层生成器串联
    paths     = file_path_generator(data_dir)
    raw_data  = image_loader(paths, label_fn=get_label)
    shuffled  = shuffle_buffer_generator(raw_data, buffer_size=buffer_size)
    augmented = augment_generator(shuffled, augment=True)
    normalized = normalize_generator(augmented)
    batches   = batch_generator(normalized, batch_size=batch_size, drop_last=True)

    return batches


# ────────────────────────────────────────
# 模拟训练循环
# ────────────────────────────────────────
def simulate_training():
    """使用模拟数据演示流水线"""

    def mock_paths():
        """模拟100张图片的路径"""
        categories = ['cat', 'dog', 'bird']
        for i in range(100):
            cat = categories[i % 3]
            yield f"/data/{cat}/img_{i:04d}.jpg"

    label_map = {'cat': 0, 'dog': 1, 'bird': 2}

    def get_label(path):
        parent = path.split('/')[-2]
        return label_map.get(parent, -1)

    # 构建流水线
    raw_data   = image_loader(mock_paths(), label_fn=get_label)
    shuffled   = shuffle_buffer_generator(raw_data, buffer_size=50)
    augmented  = augment_generator(shuffled, augment=True)
    normalized = normalize_generator(augmented)
    batches    = batch_generator(normalized, batch_size=16, drop_last=True)

    total_samples = 0
    for epoch in range(3):
        batch_count = 0
        for images, labels in batches:
            # 模拟前向传播
            batch_count += 1
            total_samples += len(images)

        print(f"Epoch {epoch + 1}: 处理了 {batch_count} 个批次")

    print(f"\n流水线总处理样本数: {total_samples}")
    print("注意：生成器已耗尽，多轮训练需重新创建流水线")


simulate_training()
```

### 关键设计思想

```python
# 1. 每个生成器只做一件事（单一职责）
# 2. 输入输出类型统一（均为 (image, label) 对），便于组合
# 3. 整个流水线在迭代之前零内存占用

# 对比 PyTorch DataLoader 的设计
# PyTorch 的 Dataset + DataLoader 正是基于类似思想：
#   Dataset.__getitem__  ← 对应 image_loader
#   transforms.Compose  ← 对应 augment + normalize
#   DataLoader           ← 对应 shuffle_buffer + batch

# 生成器版的等价 PyTorch 代码（仅示意）:
"""
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = list(Path(root).rglob('*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, get_label(self.paths[idx])

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.25]),
])

dataset = ImageDataset('data/', transform=transform)
loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for images, labels in loader:
    train_step(images, labels)
"""
```

### 多轮训练的正确写法

```python
def create_pipeline_factory(data_dir, batch_size=32):
    """
    返回一个工厂函数，每次调用都创建新的流水线。
    解决生成器一次性的问题。
    """
    def make_pipeline():
        paths      = file_path_generator(data_dir)
        raw_data   = image_loader(paths)
        augmented  = augment_generator(raw_data, augment=True)
        normalized = normalize_generator(augmented)
        return batch_generator(normalized, batch_size=batch_size)

    return make_pipeline


# 使用示例
# make_batches = create_pipeline_factory('data/', batch_size=32)
# for epoch in range(num_epochs):
#     for images, labels in make_batches():  # 每轮创建新流水线
#         train_step(images, labels)
```

---

## 练习题

### 基础题

**练习 8-1**：实现一个 `CountDown` 迭代器类，从给定数字倒数到 0（包含 0）。要求实现 `__iter__` 和 `__next__` 方法。

```
示例：list(CountDown(5)) → [5, 4, 3, 2, 1, 0]
```

**练习 8-2**：编写一个生成器函数 `prime_generator()`，无限产出质数（2, 3, 5, 7, 11, ...）。使用 `itertools.islice` 获取前 20 个质数并打印。

### 中级题

**练习 8-3**：实现一个 `sliding_window(iterable, n)` 生成器，产出长度为 n 的滑动窗口：

```
示例：list(sliding_window([1,2,3,4,5], 3)) → [(1,2,3), (2,3,4), (3,4,5)]
```

提示：可以使用 `collections.deque` 维护固定长度的缓冲区。

**练习 8-4**：使用 `itertools` 模块实现一个函数 `round_robin(*iterables)`，轮流从每个可迭代对象取一个元素，直到所有元素都被取完：

```
示例：list(round_robin('ABC', 'D', 'EF')) → ['A', 'D', 'E', 'B', 'F', 'C']
```

### 进阶题

**练习 8-5（深度学习）**：在本章的数据加载流水线基础上，实现一个 `prefetch_generator(data_gen, buffer_size=2)` 生成器。它应该在消费者处理当前批次的同时，预先读取接下来 `buffer_size` 个批次到内存缓冲区，从而减少 I/O 等待时间。

提示：使用 `threading.Thread` 和 `queue.Queue` 实现后台预取。

---

## 练习答案

### 答案 8-1

```python
class CountDown:
    """从 start 倒数到 0 的迭代器"""

    def __init__(self, start):
        if start < 0:
            raise ValueError("start 必须为非负整数")
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < 0:
            raise StopIteration
        value = self.current
        self.current -= 1
        return value


# 测试
print(list(CountDown(5)))     # [5, 4, 3, 2, 1, 0]
print(list(CountDown(0)))     # [0]
print(list(CountDown(3)))     # [3, 2, 1, 0]

# 验证可在 for 循环中使用
for n in CountDown(3):
    print(f"倒计时: {n}")
```

### 答案 8-2

```python
import itertools

def prime_generator():
    """无限质数生成器（埃拉托斯特尼筛法的惰性版本）"""
    # 使用已知质数试除法
    primes = []
    candidate = 2
    while True:
        is_prime = all(candidate % p != 0 for p in primes if p * p <= candidate)
        if is_prime:
            primes.append(candidate)
            yield candidate
        candidate += 1


# 获取前 20 个质数
first_20 = list(itertools.islice(prime_generator(), 20))
print(first_20)
# [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

# 验证：获取第 100 个质数
hundredth_prime = next(itertools.islice(prime_generator(), 99, None))
print(f"第 100 个质数: {hundredth_prime}")  # 541
```

### 答案 8-3

```python
from collections import deque

def sliding_window(iterable, n):
    """
    产出长度为 n 的滑动窗口元组。
    使用 deque 作为固定大小缓冲区，O(1) 追加和弹出。
    """
    if n <= 0:
        raise ValueError("窗口大小必须为正整数")

    window = deque(maxlen=n)
    it = iter(iterable)

    # 填充初始窗口
    for _ in range(n):
        try:
            window.append(next(it))
        except StopIteration:
            return  # 序列长度不足 n，不产出任何窗口

    yield tuple(window)

    # 滑动窗口
    for item in it:
        window.append(item)
        yield tuple(window)


# 测试
print(list(sliding_window([1, 2, 3, 4, 5], 3)))
# [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

print(list(sliding_window("ABCDE", 2)))
# [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')]

print(list(sliding_window([1, 2], 5)))
# [] （序列太短，不产出任何窗口）

# 计算移动平均（金融/信号处理中常见操作）
prices = [10, 11, 12, 10, 9, 11, 13, 14]
moving_avg = [sum(w) / len(w) for w in sliding_window(prices, 3)]
print([f"{v:.2f}" for v in moving_avg])
# ['11.00', '11.00', '10.33', '10.00', '11.00', '12.67']
```

### 答案 8-4

```python
import itertools

def round_robin(*iterables):
    """
    轮流从多个可迭代对象取元素。
    当某个迭代器耗尽时将其移除，继续处理剩余的。
    """
    # 将每个可迭代对象转为迭代器
    iterators = [iter(it) for it in iterables]

    while iterators:
        next_round = []
        for it in iterators:
            try:
                yield next(it)
                next_round.append(it)  # 该迭代器还有剩余元素
            except StopIteration:
                pass  # 该迭代器已耗尽，不加入下一轮

        iterators = next_round


# 测试
print(list(round_robin('ABC', 'D', 'EF')))
# ['A', 'D', 'E', 'B', 'F', 'C']

print(list(round_robin([1, 2, 3], [4], [5, 6])))
# [1, 4, 5, 2, 6, 3]

# 使用 itertools 的等价实现（更 Pythonic）
def round_robin_itertools(*iterables):
    """使用 itertools 实现的 round_robin"""
    sentinel = object()
    iterators = itertools.cycle(iter(it) for it in iterables)
    active_count = len(iterables)

    while active_count:
        try:
            value = next(next(iterators))
            if value is not sentinel:
                yield value
        except StopIteration:
            active_count -= 1
            iterators = itertools.cycle(
                it for it in [iter(x) for x in iterables] if it is not None
            )
```

### 答案 8-5

```python
import threading
import queue


def prefetch_generator(data_gen, buffer_size=2):
    """
    预取生成器：在后台线程中提前读取数据，
    减少主线程因 I/O 等待造成的阻塞。

    参数：
        data_gen: 上游数据生成器（通常是 batch_generator）
        buffer_size: 预取缓冲区大小（批次数量）
    """
    # 使用有界队列作为缓冲区
    # maxsize=buffer_size+1 保证最多预取 buffer_size 个批次
    buf = queue.Queue(maxsize=buffer_size + 1)
    _DONE = object()  # 哨兵值，表示上游已耗尽

    def producer():
        """后台线程：不断从上游读取并放入缓冲区"""
        try:
            for item in data_gen:
                buf.put(item)   # 缓冲区满时自动阻塞
        finally:
            buf.put(_DONE)      # 上游耗尽，发送哨兵

    # 启动后台生产者线程
    t = threading.Thread(target=producer, daemon=True)
    t.start()

    # 主线程从缓冲区消费
    while True:
        item = buf.get()
        if item is _DONE:
            break
        yield item

    t.join(timeout=1.0)  # 等待后台线程结束


# ── 测试 ──────────────────────────────────────────
import time
import random

def slow_data_gen(n_batches, batch_size=4):
    """模拟慢速数据源（如磁盘 I/O）"""
    for i in range(n_batches):
        time.sleep(0.05)  # 模拟 50ms I/O 延迟
        images = [[random.random()] * 3 for _ in range(batch_size)]
        labels = [random.randint(0, 2) for _ in range(batch_size)]
        yield images, labels


def benchmark(use_prefetch):
    start = time.time()
    data = slow_data_gen(20, batch_size=8)
    pipeline = prefetch_generator(data, buffer_size=3) if use_prefetch else data

    for images, labels in pipeline:
        time.sleep(0.03)  # 模拟 30ms 训练时间

    elapsed = time.time() - start
    return elapsed


t_no_prefetch  = benchmark(use_prefetch=False)
t_with_prefetch = benchmark(use_prefetch=True)

print(f"无预取:  {t_no_prefetch:.2f}s")
print(f"有预取:  {t_with_prefetch:.2f}s")
print(f"加速比:  {t_no_prefetch / t_with_prefetch:.2f}x")
# 典型输出:
# 无预取:  1.62s
# 有预取:  0.92s
# 加速比:  1.76x
```

**说明**：`prefetch_generator` 的核心思路是**生产者-消费者模式**。后台线程（生产者）持续从上游读取数据放入有界队列；主线程（消费者）从队列取数据进行训练。由于两者并行运行，I/O 等待时间与训练计算时间得以重叠，总耗时显著减少。这正是 PyTorch `DataLoader(num_workers>0)` 的底层原理之一。

---

*本章完 | 下一章：第9章 — 上下文管理器与 with 语句*
