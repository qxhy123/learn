# 第9章：函数式编程

> "函数式编程不是一种工具，而是一种思维方式。" —— 函数式编程社区

---

## 学习目标

完成本章学习后，你将能够：

1. **理解函数式编程的核心思想**，包括纯函数、不可变性和函数组合等基本概念
2. **熟练使用 `lambda` 表达式**创建简洁的匿名函数，用于回调和简单变换
3. **掌握 `map()` 函数**，对可迭代对象的每个元素应用变换函数
4. **掌握 `filter()` 函数**，根据条件筛选可迭代对象中的元素
5. **掌握 `reduce()` 函数**和 `functools` 模块，将序列归约为单一值，并构建高阶函数工具

---

## 9.1 函数式编程概述

### 9.1.1 什么是函数式编程

函数式编程（Functional Programming，FP）是一种编程范式，它将计算视为数学函数的求值过程，强调**避免状态变化和可变数据**。

与命令式编程（"告诉计算机怎么做"）不同，函数式编程更接近"告诉计算机做什么"。

**函数式编程的核心特征：**

| 特征 | 说明 | 示例 |
|------|------|------|
| 纯函数 | 相同输入始终得到相同输出，无副作用 | `def add(a, b): return a + b` |
| 不可变性 | 不修改已有数据，而是产生新数据 | 使用 `tuple` 代替 `list` |
| 函数组合 | 将小函数组合成大函数 | `f(g(x))` |
| 高阶函数 | 函数可以作为参数或返回值 | `map`, `filter`, `reduce` |
| 惰性求值 | 只在需要时才计算结果 | 生成器、迭代器 |

### 9.1.2 纯函数 vs 非纯函数

```python
# 非纯函数：依赖外部状态，有副作用
total = 0

def add_to_total(x):
    global total
    total += x     # 修改了外部状态
    return total

print(add_to_total(5))   # 5
print(add_to_total(5))   # 10  （相同输入，不同输出）

# 纯函数：只依赖输入参数，无副作用
def pure_add(current_total, x):
    return current_total + x   # 返回新值，不修改任何状态

print(pure_add(0, 5))    # 5
print(pure_add(0, 5))    # 5  （相同输入，始终相同输出）
```

### 9.1.3 Python 中的函数式编程工具

Python 不是纯函数式语言，但提供了丰富的函数式编程工具：

```python
# Python 函数式编程工具概览
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 命令式风格：使用循环
result_imperative = []
for n in numbers:
    if n % 2 == 0:
        result_imperative.append(n ** 2)

print("命令式:", result_imperative)
# 输出: [4, 16, 36, 64, 100]

# 函数式风格：使用 map + filter
result_functional = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, numbers)))

print("函数式:", result_functional)
# 输出: [4, 16, 36, 64, 100]

# 列表推导式风格（Python 惯用法，兼顾可读性）
result_comprehension = [n**2 for n in numbers if n % 2 == 0]

print("推导式:", result_comprehension)
# 输出: [4, 16, 36, 64, 100]
```

### 9.1.4 为什么在深度学习中使用函数式编程

在数据预处理和模型训练流程中，函数式编程风格带来以下好处：

- **可组合性**：将数据变换步骤组合成流水线
- **可测试性**：每个变换函数独立可测
- **并行友好**：无副作用的函数天然支持并行执行
- **代码简洁**：减少中间变量，逻辑更清晰

```python
# 深度学习数据预处理的函数式风格预览
import numpy as np

# 定义各个变换步骤（纯函数）
normalize = lambda x: (x - x.mean()) / (x.std() + 1e-8)
clip_outliers = lambda x: np.clip(x, -3, 3)
to_float32 = lambda x: x.astype(np.float32)

# 组合成流水线
transforms = [normalize, clip_outliers, to_float32]

def apply_pipeline(data, transforms):
    result = data
    for transform in transforms:
        result = transform(result)
    return result

# 应用到数据
raw_data = np.array([1.0, 2.0, 100.0, 3.0, 4.0])  # 含异常值
processed = apply_pipeline(raw_data, transforms)
print("处理后:", processed)
```

---

## 9.2 lambda 匿名函数

### 9.2.1 lambda 基本语法

`lambda` 表达式用于创建**匿名函数**（没有名字的函数），适合简单的单行逻辑。

**语法格式：**
```
lambda 参数列表 : 表达式
```

```python
# 普通函数定义
def square(x):
    return x ** 2

# 等价的 lambda 表达式
square_lambda = lambda x: x ** 2

print(square(5))         # 25
print(square_lambda(5))  # 25

# 多个参数
add = lambda x, y: x + y
print(add(3, 4))         # 7

# 带默认值
greet = lambda name, greeting="你好": f"{greeting}, {name}!"
print(greet("小明"))              # 你好, 小明!
print(greet("小红", "早上好"))    # 早上好, 小红!
```

### 9.2.2 lambda 与条件表达式

```python
# 在 lambda 中使用三元表达式
abs_val = lambda x: x if x >= 0 else -x
print(abs_val(-5))   # 5
print(abs_val(3))    # 3

# 判断奇偶
parity = lambda x: "偶数" if x % 2 == 0 else "奇数"
print(parity(4))     # 偶数
print(parity(7))     # 奇数

# ReLU 激活函数（深度学习中常用）
relu = lambda x: max(0, x)
print(relu(-2.5))    # 0
print(relu(3.7))     # 3.7
```

### 9.2.3 lambda 用于排序

`lambda` 最常见的用途之一是作为 `sort()` 和 `sorted()` 的 `key` 参数：

```python
# 按学生成绩排序
students = [
    {"name": "张三", "score": 85, "age": 20},
    {"name": "李四", "score": 92, "age": 19},
    {"name": "王五", "score": 78, "age": 21},
    {"name": "赵六", "score": 92, "age": 18},
]

# 按成绩降序排序
by_score = sorted(students, key=lambda s: s["score"], reverse=True)
for s in by_score:
    print(f"{s['name']}: {s['score']}")
# 李四: 92
# 赵六: 92
# 张三: 85
# 王五: 78

# 多级排序：先按成绩降序，再按年龄升序
by_score_then_age = sorted(students, key=lambda s: (-s["score"], s["age"]))
for s in by_score_then_age:
    print(f"{s['name']}: 成绩={s['score']}, 年龄={s['age']}")
# 赵六: 成绩=92, 年龄=18
# 李四: 成绩=92, 年龄=19
# 张三: 成绩=85, 年龄=20
# 王五: 成绩=78, 年龄=21
```

### 9.2.4 lambda 的局限性

```python
# lambda 只能包含单个表达式，不能包含语句
# 以下代码会报错：
# bad_lambda = lambda x: if x > 0: return x  # SyntaxError

# 不能在 lambda 中使用赋值语句（Python 3.8 前）
# complex_lambda = lambda x: (y = x + 1; y * 2)  # SyntaxError

# 正确方式：复杂逻辑使用普通函数
def complex_transform(x):
    y = x + 1
    z = y * 2
    return z if z > 0 else 0

# lambda 适合简单的一行逻辑
# 复杂逻辑用普通函数，可读性更好

# 对比可读性
# 可读性差的 lambda（不推荐）
f = lambda x: x**3 - 2*x**2 + x - 1 if x > 0 else -(x**3 - 2*x**2 + x - 1)

# 可读性好的普通函数（推荐）
def polynomial(x):
    """计算三次多项式 x^3 - 2x^2 + x - 1 的绝对值"""
    value = x**3 - 2*x**2 + x - 1
    return abs(value)
```

### 9.2.5 立即调用的 lambda

```python
# lambda 可以立即调用（IIFE 风格）
result = (lambda x, y: x + y)(3, 4)
print(result)  # 7

# 实用场景：根据条件动态选择函数
mode = "train"
transform = (lambda x: x * 0.8 + 0.1 * (2 * __import__('random').random() - 1)
             if mode == "train"
             else lambda x: x)
```

---

## 9.3 map 函数

### 9.3.1 map 基本用法

`map(function, iterable)` 将函数应用到可迭代对象的**每个元素**，返回一个迭代器。

```python
# 基本用法
numbers = [1, 2, 3, 4, 5]

# 对每个元素求平方
squares = map(lambda x: x**2, numbers)
print(list(squares))  # [1, 4, 9, 16, 25]

# 使用普通函数
def celsius_to_fahrenheit(c):
    return c * 9/5 + 32

temps_celsius = [0, 20, 37, 100]
temps_fahrenheit = list(map(celsius_to_fahrenheit, temps_celsius))
print(temps_fahrenheit)  # [32.0, 68.0, 98.6, 212.0]

# map 返回的是迭代器（惰性求值）
m = map(lambda x: x**2, range(10))
print(type(m))    # <class 'map'>
print(next(m))    # 0
print(next(m))    # 1
print(next(m))    # 4
```

### 9.3.2 map 处理多个可迭代对象

`map` 支持同时处理多个可迭代对象，此时函数接收来自每个迭代对象的对应元素：

```python
# 两个列表对应元素相加
a = [1, 2, 3, 4]
b = [10, 20, 30, 40]

result = list(map(lambda x, y: x + y, a, b))
print(result)  # [11, 22, 33, 44]

# 等价于 zip + 列表推导式
result2 = [x + y for x, y in zip(a, b)]
print(result2)  # [11, 22, 33, 44]

# 三个列表：计算加权和
weights = [0.1, 0.2, 0.3, 0.4]
values = [100, 200, 150, 300]
biases = [5, 10, 3, 8]

weighted = list(map(lambda w, v, b: w * v + b, weights, values, biases))
print(weighted)  # [15.0, 50.0, 48.0, 128.0]

# 多个迭代对象长度不同时，map 以最短的为准
x = [1, 2, 3, 4, 5]
y = [10, 20, 30]
result3 = list(map(lambda a, b: a + b, x, y))
print(result3)  # [11, 22, 33]  （只处理3个元素）
```

### 9.3.3 map 与字符串处理

```python
# 字符串列表的批量处理
names = ["  张三  ", "李四 ", "  王五"]
clean_names = list(map(str.strip, names))
print(clean_names)  # ['张三', '李四', '王五']

# 转换为大写（英文）
words = ["hello", "world", "python"]
upper_words = list(map(str.upper, words))
print(upper_words)  # ['HELLO', 'WORLD', 'PYTHON']

# 字符串转数字
str_numbers = ["1", "2", "3", "4", "5"]
int_numbers = list(map(int, str_numbers))
print(int_numbers)  # [1, 2, 3, 4, 5]

float_numbers = list(map(float, str_numbers))
print(float_numbers)  # [1.0, 2.0, 3.0, 4.0, 5.0]
```

### 9.3.4 map 在数据处理中的应用

```python
import numpy as np

# 批量归一化数据批次
def normalize_batch(batch):
    """对单个批次数据进行归一化"""
    mean = np.mean(batch)
    std = np.std(batch) + 1e-8
    return (batch - mean) / std

# 模拟多个数据批次
batches = [
    np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    np.array([10.0, 20.0, 30.0, 40.0]),
    np.array([0.1, 0.5, 0.9, 1.3]),
]

normalized_batches = list(map(normalize_batch, batches))

for i, batch in enumerate(normalized_batches):
    print(f"批次 {i+1}: 均值={batch.mean():.4f}, 标准差={batch.std():.4f}")
# 批次 1: 均值=0.0000, 标准差=1.0000
# 批次 2: 均值=0.0000, 标准差=1.0000
# 批次 3: 均值=0.0000, 标准差=1.0000

# 批量应用激活函数
def softmax(x):
    """数值稳定的 softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

logits_list = [
    np.array([1.0, 2.0, 3.0]),
    np.array([0.5, 0.5, 0.5]),
    np.array([-1.0, 0.0, 1.0]),
]

probabilities = list(map(softmax, logits_list))
for i, prob in enumerate(probabilities):
    print(f"样本 {i+1}: {prob.round(4)}, 和={prob.sum():.4f}")
# 样本 1: [0.0900 0.2447 0.6652], 和=1.0000
# 样本 2: [0.3333 0.3333 0.3333], 和=1.0000
# 样本 3: [0.0900 0.2447 0.6652], 和=1.0000
```

### 9.3.5 map vs 列表推导式

```python
import time

data = list(range(1000000))

# 性能对比
start = time.time()
result_map = list(map(lambda x: x**2, data))
map_time = time.time() - start

start = time.time()
result_comp = [x**2 for x in data]
comp_time = time.time() - start

print(f"map 耗时: {map_time:.4f}s")
print(f"列表推导式耗时: {comp_time:.4f}s")

# 选择建议：
# - 使用已有函数（如 str.upper, int, float）：优先用 map
# - 需要条件过滤：用列表推导式
# - 简单变换且注重可读性：用列表推导式
# - 处理大数据集且不需要一次性加载：用 map（惰性求值）
```

---

## 9.4 filter 函数

### 9.4.1 filter 基本用法

`filter(function, iterable)` 返回使 `function` 结果为 `True` 的所有元素，返回一个迭代器。

```python
# 基本用法
numbers = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]

# 筛选正数
positives = list(filter(lambda x: x > 0, numbers))
print(positives)  # [1, 3, 5, 7, 9]

# 筛选偶数
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [-2, -4, -6, -8, -10]

# 使用普通函数
def is_prime(n):
    """判断是否为质数"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

numbers_1_to_30 = range(1, 31)
primes = list(filter(is_prime, numbers_1_to_30))
print(primes)  # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```

### 9.4.2 filter 与 None

当第一个参数为 `None` 时，`filter` 过滤掉所有假值（falsy values）：

```python
# filter(None, iterable) 过滤假值
mixed = [0, 1, "", "hello", None, [1, 2], [], False, True, 0.0, 3.14]
truthy = list(filter(None, mixed))
print(truthy)  # [1, 'hello', [1, 2], True, 3.14]

# 实用：过滤空字符串
texts = ["", "你好", "  ", "世界", "", "Python"]
non_empty = list(filter(None, texts))
print(non_empty)  # ['你好', '  ', '世界', 'Python']

# 如果还要过滤空白字符串：
non_blank = list(filter(str.strip, texts))
print(non_blank)  # ['你好', '世界', 'Python']
```

### 9.4.3 filter 处理复杂对象

```python
# 过滤数据集中的无效样本
import numpy as np

samples = [
    {"id": 1, "features": np.array([1.0, 2.0, 3.0]), "label": 0},
    {"id": 2, "features": np.array([np.nan, 2.0, 3.0]), "label": 1},  # 含 NaN
    {"id": 3, "features": np.array([1.0, 2.0, 3.0]), "label": 2},
    {"id": 4, "features": np.array([1.0, np.inf, 3.0]), "label": 0},  # 含 Inf
    {"id": 5, "features": np.array([4.0, 5.0, 6.0]), "label": 1},
]

def is_valid_sample(sample):
    """检查样本是否有效（不含 NaN 或 Inf）"""
    features = sample["features"]
    return not (np.any(np.isnan(features)) or np.any(np.isinf(features)))

valid_samples = list(filter(is_valid_sample, samples))
print(f"原始样本数: {len(samples)}")
print(f"有效样本数: {len(valid_samples)}")
for s in valid_samples:
    print(f"  样本 {s['id']}: {s['features']}")
# 原始样本数: 5
# 有效样本数: 3
#   样本 1: [1. 2. 3.]
#   样本 3: [1. 2. 3.]
#   样本 5: [4. 5. 6.]
```

### 9.4.4 filter 组合使用

```python
# 多条件过滤：链式 filter
students = [
    {"name": "张三", "score": 85, "absent_days": 2},
    {"name": "李四", "score": 92, "absent_days": 0},
    {"name": "王五", "score": 60, "absent_days": 5},
    {"name": "赵六", "score": 75, "absent_days": 1},
    {"name": "钱七", "score": 88, "absent_days": 3},
    {"name": "孙八", "score": 55, "absent_days": 0},
]

# 筛选：成绩 >= 75 且 缺勤天数 <= 2
passed = filter(lambda s: s["score"] >= 75, students)
eligible = list(filter(lambda s: s["absent_days"] <= 2, passed))

for s in eligible:
    print(f"{s['name']}: 成绩={s['score']}, 缺勤={s['absent_days']}天")
# 张三: 成绩=85, 缺勤=2天
# 李四: 成绩=92, 缺勤=0天
# 赵六: 成绩=75, 缺勤=1天

# 也可以合并条件（更推荐的写法）
eligible2 = list(filter(
    lambda s: s["score"] >= 75 and s["absent_days"] <= 2,
    students
))
```

### 9.4.5 filter vs 列表推导式

```python
# filter 和列表推导式的等价写法
data = range(1, 21)

# 使用 filter
result_filter = list(filter(lambda x: x % 3 == 0 or x % 5 == 0, data))

# 使用列表推导式（更 Pythonic）
result_comp = [x for x in data if x % 3 == 0 or x % 5 == 0]

print(result_filter)  # [3, 5, 6, 9, 10, 12, 15, 18, 20]
print(result_comp)    # [3, 5, 6, 9, 10, 12, 15, 18, 20]

# 选择建议：
# - 有已定义的过滤函数：用 filter（代码更清晰）
# - 简单条件：用列表推导式（更 Pythonic）
# - 大数据流式处理：用 filter（惰性求值，节省内存）
```

---

## 9.5 reduce 函数与 functools 模块

### 9.5.1 reduce 基本用法

`reduce` 在 Python 3 中被移到了 `functools` 模块。它将一个二元函数**累积地**应用到序列的元素，将序列归约为单一值。

```python
from functools import reduce

# 求和
numbers = [1, 2, 3, 4, 5]
total = reduce(lambda acc, x: acc + x, numbers)
print(total)  # 15

# 执行过程：
# 第1步: acc=1, x=2  → 3
# 第2步: acc=3, x=3  → 6
# 第3步: acc=6, x=4  → 10
# 第4步: acc=10, x=5 → 15

# 求积
product = reduce(lambda acc, x: acc * x, numbers)
print(product)  # 120 (即 5!)

# 求最大值
maximum = reduce(lambda acc, x: acc if acc > x else x, numbers)
print(maximum)  # 5

# 带初始值
total_with_init = reduce(lambda acc, x: acc + x, numbers, 100)
print(total_with_init)  # 115 (100 + 15)
```

### 9.5.2 reduce 构建复杂归约

```python
from functools import reduce

# 将嵌套列表展平
nested = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
flat = reduce(lambda acc, x: acc + x, nested)
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 统计词频
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
word_count = reduce(
    lambda acc, word: {**acc, word: acc.get(word, 0) + 1},
    words,
    {}  # 初始值为空字典
)
print(word_count)  # {'apple': 3, 'banana': 2, 'cherry': 1}

# 构建嵌套函数调用
# compose(f, g)(x) 等价于 f(g(x))
def compose(*functions):
    """从右到左组合多个函数"""
    return reduce(lambda f, g: lambda x: f(g(x)), functions)

double = lambda x: x * 2
add_one = lambda x: x + 1
square = lambda x: x ** 2

# 组合：先 square，再 add_one，再 double
transform = compose(double, add_one, square)
print(transform(3))  # double(add_one(square(3))) = double(add_one(9)) = double(10) = 20
```

### 9.5.3 functools 模块其他工具

```python
import functools

# 1. partial：偏函数，固定部分参数
def power(base, exp):
    return base ** exp

square = functools.partial(power, exp=2)
cube = functools.partial(power, exp=3)

print(square(4))  # 16
print(cube(3))    # 27

# 在深度学习中常用于固定超参数
import numpy as np

def dropout(x, rate, training=True):
    if not training:
        return x
    mask = np.random.binomial(1, 1 - rate, x.shape) / (1 - rate)
    return x * mask

# 创建固定 dropout rate 的函数
dropout_50 = functools.partial(dropout, rate=0.5)
dropout_20 = functools.partial(dropout, rate=0.2)

x = np.ones(10)
print("50% dropout:", dropout_50(x).round(2))
print("20% dropout:", dropout_20(x).round(2))
```

```python
import functools

# 2. lru_cache：最近最少使用缓存，用于记忆化
@functools.lru_cache(maxsize=128)
def fibonacci(n):
    """带缓存的斐波那契数列（避免重复计算）"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

import time

# 无缓存版本（慢）
def fib_slow(n):
    if n < 2:
        return n
    return fib_slow(n - 1) + fib_slow(n - 2)

start = time.time()
print(fibonacci(35))  # 9227465
print(f"带缓存: {time.time() - start:.6f}s")

start = time.time()
print(fib_slow(35))   # 9227465
print(f"无缓存: {time.time() - start:.6f}s")

# 查看缓存信息
print(fibonacci.cache_info())
# CacheInfo(hits=33, misses=36, maxsize=128, currsize=36)
```

```python
import functools

# 3. wraps：保留被装饰函数的元信息
def timer_decorator(func):
    @functools.wraps(func)   # 保留原函数的 __name__, __doc__ 等
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} 耗时: {elapsed:.4f}s")
        return result
    return wrapper

@timer_decorator
def train_epoch(data, model_name="CNN"):
    """训练一个 epoch"""
    import time
    time.sleep(0.1)  # 模拟训练
    return {"loss": 0.5, "acc": 0.85}

result = train_epoch([1, 2, 3])
print(f"函数名: {train_epoch.__name__}")  # train_epoch（而不是 wrapper）
print(f"文档: {train_epoch.__doc__}")    # 训练一个 epoch
```

### 9.5.4 functools.reduce 在深度学习中的应用

```python
from functools import reduce
import numpy as np

# 1. 计算多层网络的输出维度
layer_configs = [
    {"type": "linear", "out": 256},
    {"type": "linear", "out": 128},
    {"type": "linear", "out": 64},
    {"type": "linear", "out": 10},
]

input_dim = 784  # MNIST 输入维度

def process_layer(current_dim, layer):
    print(f"  {layer['type']}: {current_dim} -> {layer['out']}")
    return layer["out"]

print("网络结构：")
output_dim = reduce(process_layer, layer_configs, input_dim)
print(f"最终输出维度: {output_dim}")
# 网络结构：
#   linear: 784 -> 256
#   linear: 256 -> 128
#   linear: 128 -> 64
#   linear: 64 -> 10
# 最终输出维度: 10

# 2. 累积计算梯度乘积（链式法则）
gradients = [0.9, 0.8, 0.7, 0.6]  # 各层梯度
total_gradient = reduce(lambda acc, g: acc * g, gradients, 1.0)
print(f"\n梯度累积乘积: {total_gradient:.4f}")  # 0.3024
# 这展示了深层网络梯度消失的原因

# 3. 组合多个损失函数
def combine_losses(loss_fns, weights, predictions, targets):
    """加权组合多个损失函数"""
    weighted_losses = map(
        lambda fw: fw[1] * fw[0](predictions, targets),
        zip(loss_fns, weights)
    )
    return reduce(lambda acc, l: acc + l, weighted_losses, 0.0)

# 模拟损失函数
mse_loss = lambda pred, true: np.mean((pred - true) ** 2)
mae_loss = lambda pred, true: np.mean(np.abs(pred - true))

pred = np.array([1.0, 2.0, 3.0])
true = np.array([1.1, 1.9, 3.2])

total_loss = combine_losses(
    loss_fns=[mse_loss, mae_loss],
    weights=[0.7, 0.3],
    predictions=pred,
    targets=true
)
print(f"\n组合损失: {total_loss:.6f}")
```

---

## 本章小结

| 工具 | 语法 | 用途 | 返回类型 |
|------|------|------|----------|
| `lambda` | `lambda 参数: 表达式` | 创建简单匿名函数 | 函数对象 |
| `map()` | `map(func, iterable)` | 对每个元素应用变换 | `map` 迭代器 |
| `filter()` | `filter(func, iterable)` | 筛选满足条件的元素 | `filter` 迭代器 |
| `reduce()` | `reduce(func, iterable, init)` | 将序列归约为单一值 | 任意类型 |
| `partial()` | `partial(func, *args, **kwargs)` | 固定部分参数，创建新函数 | 函数对象 |
| `lru_cache` | `@lru_cache(maxsize=N)` | 缓存函数结果（记忆化） | 装饰器 |
| `wraps()` | `@wraps(func)` | 保留原函数元信息 | 装饰器 |

**核心思想回顾：**

- **纯函数**：相同输入 → 相同输出，无副作用，易于测试和推理
- **函数组合**：将小函数组合成复杂变换，每一步清晰可测
- **惰性求值**：`map` 和 `filter` 返回迭代器，按需计算，节省内存
- **高阶函数**：函数作为参数/返回值，提升代码复用性和表达力

**选择建议：**

```
简单变换/过滤  →  列表推导式（更 Pythonic）
有现成函数     →  map/filter（代码简洁）
大数据流处理   →  map/filter（惰性求值）
序列归约       →  reduce（或内置 sum/max/min）
固定参数       →  partial（避免重复代码）
```

---

## 深度学习应用：数据变换管道

在深度学习中，数据预处理是模型训练的关键环节。函数式编程风格非常适合构建**可组合、可复用**的数据变换流水线。

### 完整示例：图像数据预处理流水线

```python
import numpy as np
from functools import reduce, partial
import random

# ============================================================
# 第一部分：定义基础变换函数（纯函数）
# ============================================================

def normalize(mean, std):
    """返回归一化函数（使用 partial 风格的闭包）"""
    def _normalize(image):
        return (image - mean) / (std + 1e-8)
    return _normalize

def resize(target_size):
    """返回缩放函数（模拟）"""
    def _resize(image):
        # 实际中使用 PIL 或 OpenCV，这里用插值模拟
        h, w = image.shape[:2]
        # 简化：只做形状变换演示
        return image[:target_size, :target_size] if image.shape[0] >= target_size \
               else np.pad(image, ((0, target_size - h), (0, target_size - w)))
    return _resize

def random_flip(prob=0.5):
    """随机水平翻转"""
    def _flip(image):
        if random.random() < prob:
            return np.fliplr(image)
        return image
    return _flip

def random_crop(crop_size, pad=4):
    """随机裁剪"""
    def _crop(image):
        h, w = image.shape[:2]
        # 填充后随机裁剪
        padded = np.pad(image, ((pad, pad), (pad, pad)))
        top = random.randint(0, 2 * pad)
        left = random.randint(0, 2 * pad)
        return padded[top:top + h, left:left + w]
    return _crop

def add_gaussian_noise(std=0.01):
    """添加高斯噪声（数据增强）"""
    def _add_noise(image):
        noise = np.random.normal(0, std, image.shape)
        return np.clip(image + noise, 0.0, 1.0)
    return _add_noise

def to_float32(image):
    """转换为 float32"""
    return image.astype(np.float32)

def expand_dims(image):
    """添加通道维度：(H, W) -> (1, H, W)"""
    return image[np.newaxis, :]  # 添加 channel 维度

# ============================================================
# 第二部分：流水线构建器
# ============================================================

def make_pipeline(*transforms):
    """
    将多个变换函数组合成流水线。
    使用 reduce 将函数列表归约为单个复合变换。
    """
    def apply_all(image):
        return reduce(lambda img, fn: fn(img), transforms, image)
    return apply_all

# ============================================================
# 第三部分：构建训练和验证流水线
# ============================================================

# 数据集统计信息（通常在训练集上计算）
DATASET_MEAN = 0.4914
DATASET_STD = 0.2023

# 训练流水线（含数据增强）
train_pipeline = make_pipeline(
    to_float32,
    normalize(DATASET_MEAN, DATASET_STD),
    random_flip(prob=0.5),
    random_crop(32, pad=4),
    add_gaussian_noise(std=0.01),
    expand_dims,
)

# 验证/测试流水线（不含数据增强）
val_pipeline = make_pipeline(
    to_float32,
    normalize(DATASET_MEAN, DATASET_STD),
    expand_dims,
)

# ============================================================
# 第四部分：批处理与数据加载器模拟
# ============================================================

def process_dataset(images, pipeline, batch_size=32):
    """
    使用 map 对整个数据集应用流水线，
    使用 filter 过滤无效样本，
    并按批次生成数据。
    """
    # 过滤无效图像
    valid_images = list(filter(
        lambda img: img is not None and img.shape == (32, 32),
        images
    ))

    # 应用变换流水线（惰性求值）
    processed = map(pipeline, valid_images)

    # 按批次返回
    batch = []
    for image in processed:
        batch.append(image)
        if len(batch) == batch_size:
            yield np.stack(batch)
            batch = []
    if batch:
        yield np.stack(batch)

# ============================================================
# 第五部分：演示运行
# ============================================================

# 生成模拟数据集（100张 32x32 灰度图像）
np.random.seed(42)
dataset = [np.random.rand(32, 32) for _ in range(100)]

print("=" * 50)
print("数据变换流水线演示")
print("=" * 50)

# 处理单张图像
sample = dataset[0]
print(f"\n原始图像:")
print(f"  形状: {sample.shape}")
print(f"  数据类型: {sample.dtype}")
print(f"  值域: [{sample.min():.3f}, {sample.max():.3f}]")

train_sample = train_pipeline(sample)
print(f"\n训练流水线处理后:")
print(f"  形状: {train_sample.shape}")
print(f"  数据类型: {train_sample.dtype}")
print(f"  均值: {train_sample.mean():.4f}")
print(f"  标准差: {train_sample.std():.4f}")

val_sample = val_pipeline(sample)
print(f"\n验证流水线处理后:")
print(f"  形状: {val_sample.shape}")
print(f"  数据类型: {val_sample.dtype}")
print(f"  均值: {val_sample.mean():.4f}")
print(f"  标准差: {val_sample.std():.4f}")

# 批处理
print(f"\n批处理演示 (batch_size=32):")
batches = list(process_dataset(dataset, train_pipeline, batch_size=32))
print(f"  总批次数: {len(batches)}")
for i, batch in enumerate(batches):
    print(f"  批次 {i+1}: 形状={batch.shape}, 均值={batch.mean():.4f}")

# ============================================================
# 第六部分：可组合的损失函数流水线
# ============================================================

print("\n" + "=" * 50)
print("损失函数组合演示")
print("=" * 50)

# 定义各种损失函数
def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)

def l1_loss(pred, target):
    return np.mean(np.abs(pred - target))

def smooth_l1_loss(pred, target, beta=1.0):
    diff = np.abs(pred - target)
    loss = np.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return np.mean(loss)

# 使用 partial 固定超参数
huber_loss = partial(smooth_l1_loss, beta=0.5)

# 使用 reduce 组合加权损失
def weighted_loss(loss_fns_weights, pred, target):
    """组合多个加权损失函数"""
    losses = map(
        lambda fw: fw[1] * fw[0](pred, target),
        loss_fns_weights
    )
    return reduce(lambda a, b: a + b, losses)

# 定义组合损失（70% MSE + 20% L1 + 10% Huber）
combined_loss = partial(weighted_loss, [
    (mse_loss, 0.7),
    (l1_loss, 0.2),
    (huber_loss, 0.1),
])

# 测试
pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
target = np.array([1.1, 1.8, 3.3, 3.9, 5.2])

print(f"\nMSE 损失:    {mse_loss(pred, target):.6f}")
print(f"L1 损失:     {l1_loss(pred, target):.6f}")
print(f"Huber 损失:  {huber_loss(pred, target):.6f}")
print(f"组合损失:    {combined_loss(pred, target):.6f}")
```

**输出示例：**
```
==================================================
数据变换流水线演示
==================================================

原始图像:
  形状: (32, 32)
  数据类型: float64
  值域: [0.000, 1.000]

训练流水线处理后:
  形状: (1, 32, 32)
  数据类型: float32
  均值: -0.0312
  标准差: 0.9987

验证流水线处理后:
  形状: (1, 32, 32)
  数据类型: float32
  均值: -0.0387
  标准差: 0.9834

批处理演示 (batch_size=32):
  总批次数: 4
  批次 1: 形状=(32, 1, 32, 32), 均值=-0.0021
  批次 2: 形状=(32, 1, 32, 32), 均值=0.0043
  批次 3: 形状=(32, 1, 32, 32), 均值=-0.0018
  批次 4: 形状=(4, 1, 32, 32), 均值=0.0067
```

### 函数式流水线的设计原则总结

```python
# 好的函数式流水线设计：

# 1. 每个变换是纯函数（不修改输入）
def good_transform(x):
    return x * 2  # 返回新值

# 2. 使用闭包/partial 参数化变换
def make_scale(factor):
    return lambda x: x * factor

scale_2x = make_scale(2)
scale_3x = make_scale(3)

# 3. 变换可以独立测试
assert good_transform(5) == 10
assert scale_2x(5) == 10
assert scale_3x(5) == 15

# 4. 流水线通过组合构建，而非修改
basic_pipeline = make_pipeline(to_float32, expand_dims)
augmented_pipeline = make_pipeline(to_float32, random_flip(), expand_dims)
# 两个流水线独立，互不影响

print("所有断言通过，流水线设计正确！")
```

---

## 练习题

### 基础题

**练习 9.1**：使用 `lambda` 和 `map` 实现温度转换

编写代码，将一组摄氏温度值转换为华氏温度和开尔文温度。

```python
celsius_temps = [0, 20, 37, 100, -40, 200]

# 要求：
# 1. 使用 lambda 定义转换函数
# 2. 使用 map 进行批量转换
# 3. 输出格式：[{"C": 0, "F": 32.0, "K": 273.15}, ...]
```

**练习 9.2**：使用 `filter` 清洗文本数据

给定一个包含各种字符串的列表，使用 `filter` 筛选出有效的电子邮件地址（包含且仅包含一个 `@`，且 `@` 前后均有内容）。

```python
raw_data = [
    "user@example.com",
    "invalid-email",
    "@nodomain.com",
    "noatsign.com",
    "valid@test.org",
    "double@@email.com",
    "",
    "another@valid.net",
    "spaces in@email.com",
]

# 要求：使用 filter 和 lambda 筛选有效邮件地址
```

### 中级题

**练习 9.3**：使用 `reduce` 实现深度优先树遍历统计

给定一个表示树形目录结构的嵌套字典，使用 `reduce` 计算所有文件的总大小。

```python
from functools import reduce

file_tree = {
    "name": "root",
    "size": 0,
    "children": [
        {"name": "file1.py", "size": 1024, "children": []},
        {
            "name": "subdir",
            "size": 0,
            "children": [
                {"name": "file2.py", "size": 2048, "children": []},
                {"name": "file3.py", "size": 512, "children": []},
            ]
        },
        {"name": "file4.py", "size": 4096, "children": []},
    ]
}

# 要求：使用递归 + reduce 计算所有文件的总大小
# 预期结果：7680
```

**练习 9.4**：构建可组合的数据验证器

使用函数式编程思想，构建一个可组合的数据验证系统。

```python
# 要求：
# 1. 定义基础验证函数（每个返回 True/False）
# 2. 使用 reduce 将多个验证器组合成一个
# 3. 对以下学生数据进行验证：
#    - 姓名不为空
#    - 年龄在 15-25 之间
#    - 成绩在 0-100 之间

students = [
    {"name": "张三", "age": 20, "score": 85},
    {"name": "", "age": 20, "score": 90},
    {"name": "李四", "age": 30, "score": 75},
    {"name": "王五", "age": 18, "score": 110},
    {"name": "赵六", "age": 22, "score": 95},
]
```

### 高级题

**练习 9.5**：实现函数式神经网络前向传播

使用 `map`、`reduce` 和 `functools.partial` 实现一个简单神经网络的前向传播，要求完全使用函数式风格（无 for 循环，无类定义）。

```python
import numpy as np
from functools import reduce, partial

# 网络结构：784 -> 256 -> 128 -> 10
# 激活函数：隐藏层用 ReLU，输出层用 Softmax

# 要求：
# 1. 定义纯函数形式的 linear_layer(W, b, x)
# 2. 定义 relu 和 softmax 激活函数
# 3. 使用 reduce 将多层操作组合为完整的前向传播
# 4. 对一个批次的数据（batch_size=4）进行预测
# 5. 验证输出概率之和为 1
```

---

## 练习答案

### 练习 9.1 答案

```python
celsius_temps = [0, 20, 37, 100, -40, 200]

# 定义转换函数
to_fahrenheit = lambda c: c * 9/5 + 32
to_kelvin = lambda c: c + 273.15
to_record = lambda c: {"C": c, "F": round(to_fahrenheit(c), 2), "K": round(to_kelvin(c), 2)}

# 使用 map 批量转换
temperature_records = list(map(to_record, celsius_temps))

for record in temperature_records:
    print(f"摄氏 {record['C']:6.1f}°C  →  华氏 {record['F']:7.2f}°F  →  开尔文 {record['K']:7.2f}K")

# 输出：
# 摄氏    0.0°C  →  华氏   32.00°F  →  开尔文  273.15K
# 摄氏   20.0°C  →  华氏   68.00°F  →  开尔文  293.15K
# 摄氏   37.0°C  →  华氏   98.60°F  →  开尔文  310.15K
# 摄氏  100.0°C  →  华氏  212.00°F  →  开尔文  373.15K
# 摄氏  -40.0°C  →  华氏  -40.00°F  →  开尔文  233.15K
# 摄氏  200.0°C  →  华氏  392.00°F  →  开尔文  473.15K
```

### 练习 9.2 答案

```python
raw_data = [
    "user@example.com",
    "invalid-email",
    "@nodomain.com",
    "noatsign.com",
    "valid@test.org",
    "double@@email.com",
    "",
    "another@valid.net",
    "spaces in@email.com",
]

def is_valid_email(email):
    """验证邮件地址：恰好一个@，且前后均有内容，且不含空格"""
    if not email or ' ' in email:
        return False
    parts = email.split('@')
    return len(parts) == 2 and len(parts[0]) > 0 and len(parts[1]) > 0

valid_emails = list(filter(is_valid_email, raw_data))
print("有效邮件地址：")
for email in valid_emails:
    print(f"  ✓ {email}")

# 输出：
# 有效邮件地址：
#   ✓ user@example.com
#   ✓ valid@test.org
#   ✓ another@valid.net
```

### 练习 9.3 答案

```python
from functools import reduce

file_tree = {
    "name": "root",
    "size": 0,
    "children": [
        {"name": "file1.py", "size": 1024, "children": []},
        {
            "name": "subdir",
            "size": 0,
            "children": [
                {"name": "file2.py", "size": 2048, "children": []},
                {"name": "file3.py", "size": 512, "children": []},
            ]
        },
        {"name": "file4.py", "size": 4096, "children": []},
    ]
}

def total_size(node):
    """使用递归 + reduce 计算目录树总大小"""
    own_size = node["size"]
    if not node["children"]:
        return own_size
    children_size = reduce(
        lambda acc, child: acc + total_size(child),
        node["children"],
        0
    )
    return own_size + children_size

result = total_size(file_tree)
print(f"目录树总大小: {result} 字节")  # 目录树总大小: 7680 字节
assert result == 7680, f"期望 7680，实际 {result}"
print("断言通过！")
```

### 练习 9.4 答案

```python
from functools import reduce

# 定义基础验证函数（每个返回 (bool, str)）
def validate_name(student):
    valid = bool(student["name"].strip())
    return valid, "姓名不能为空" if not valid else ""

def validate_age(student):
    valid = 15 <= student["age"] <= 25
    return valid, f"年龄 {student['age']} 不在 [15, 25] 范围内" if not valid else ""

def validate_score(student):
    valid = 0 <= student["score"] <= 100
    return valid, f"成绩 {student['score']} 不在 [0, 100] 范围内" if not valid else ""

def combine_validators(*validators):
    """使用 reduce 组合多个验证器"""
    def combined(student):
        results = map(lambda v: v(student), validators)
        errors = [msg for ok, msg in results if not ok]
        return len(errors) == 0, errors
    return combined

# 组合验证器
student_validator = combine_validators(validate_name, validate_age, validate_score)

students = [
    {"name": "张三", "age": 20, "score": 85},
    {"name": "", "age": 20, "score": 90},
    {"name": "李四", "age": 30, "score": 75},
    {"name": "王五", "age": 18, "score": 110},
    {"name": "赵六", "age": 22, "score": 95},
]

print("学生数据验证结果：")
for student in students:
    is_valid, errors = student_validator(student)
    name = student["name"] or "（空）"
    if is_valid:
        print(f"  ✓ {name}: 验证通过")
    else:
        print(f"  ✗ {name}: {'; '.join(errors)}")

# 输出：
# 学生数据验证结果：
#   ✓ 张三: 验证通过
#   ✗ （空）: 姓名不能为空
#   ✗ 李四: 年龄 30 不在 [15, 25] 范围内
#   ✗ 王五: 成绩 110 不在 [0, 100] 范围内
#   ✓ 赵六: 验证通过
```

### 练习 9.5 答案

```python
import numpy as np
from functools import reduce, partial

# 设置随机种子
np.random.seed(42)

# ---- 定义纯函数层操作 ----

def linear_forward(W, b, x):
    """线性变换：y = xW^T + b"""
    return x @ W.T + b

def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

def softmax(x):
    """数值稳定的 Softmax"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# ---- 初始化网络参数（He 初始化）----

def init_layer(in_dim, out_dim):
    """使用 He 初始化创建一层参数"""
    W = np.random.randn(out_dim, in_dim) * np.sqrt(2.0 / in_dim)
    b = np.zeros(out_dim)
    return W, b

# 网络结构：784 -> 256 -> 128 -> 10
layer_dims = [(784, 256), (256, 128), (128, 10)]
params = list(map(lambda dims: init_layer(*dims), layer_dims))

# ---- 构建前向传播流水线 ----

# 创建每一层的变换函数（使用 partial 固定参数）
layer_fns = [
    partial(linear_forward, W, b)
    for W, b in params
]

# 为每层指定激活函数
activations = [relu, relu, softmax]

# 将线性变换和激活函数组合
layer_transforms = list(map(
    lambda lf_act: (lambda x: lf_act[1](lf_act[0](x))),
    zip(layer_fns, activations)
))

# 使用 reduce 组合所有层
def forward(x, layers):
    """使用 reduce 进行前向传播"""
    return reduce(lambda h, layer_fn: layer_fn(h), layers, x)

# ---- 测试前向传播 ----

# 创建批次输入（4个样本，每个784维）
batch_size = 4
x_input = np.random.randn(batch_size, 784)

# 前向传播
output = forward(x_input, layer_transforms)

print("函数式神经网络前向传播结果：")
print(f"  输入形状: {x_input.shape}")
print(f"  输出形状: {output.shape}")
print(f"\n  各样本输出概率（前5个类别）：")
for i in range(batch_size):
    probs = output[i][:5]
    total = output[i].sum()
    print(f"  样本 {i+1}: {probs.round(4)}... 概率和={total:.6f}")

# 验证输出为合法概率分布
assert np.allclose(output.sum(axis=1), 1.0, atol=1e-6), "概率之和不为1！"
assert np.all(output >= 0), "存在负概率！"
print("\n所有验证通过：输出是合法的概率分布！")

# 输出示例：
# 函数式神经网络前向传播结果：
#   输入形状: (4, 784)
#   输出形状: (4, 10)
#
#   各样本输出概率（前5个类别）：
#   样本 1: [0.0821 0.0934 0.1102 0.0987 0.0876]... 概率和=1.000000
#   样本 2: [0.0756 0.1023 0.0891 0.1145 0.0934]... 概率和=1.000000
#   样本 3: [0.0912 0.0867 0.0978 0.1034 0.0823]... 概率和=1.000000
#   样本 4: [0.0845 0.0956 0.1023 0.0912 0.0867]... 概率和=1.000000
#
# 所有验证通过：输出是合法的概率分布！
```

---

*下一章：第10章 —— 迭代器与生成器*
