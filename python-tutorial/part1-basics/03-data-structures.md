# 第3章：数据结构

> Python 内置了多种强大的数据结构，是编写高效程序的基础。在深度学习中，数据结构用于组织训练样本、模型配置、超参数等关键信息。

---

## 学习目标

完成本章后，你将能够：

1. 熟练使用列表（List）进行序列数据的存储、索引、切片和操作
2. 理解元组（Tuple）的不可变性，掌握解包和命名元组的用法
3. 使用字典（Dict）管理键值对数据，编写字典推导式
4. 利用集合（Set）完成去重和集合运算
5. 编写列表推导式、字典推导式和集合推导式，写出更 Pythonic 的代码

---

## 3.1 列表（List）

列表是 Python 中最常用的数据结构，用方括号 `[]` 表示，可以存储任意类型的元素，支持增删改查。

### 3.1.1 创建列表

```python
# 空列表
empty = []
empty2 = list()

# 整数列表
numbers = [1, 2, 3, 4, 5]

# 混合类型列表
mixed = [1, "hello", 3.14, True]

# 嵌套列表（二维矩阵）
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# 用 list() 将其他可迭代对象转换为列表
from_range = list(range(1, 11))   # [1, 2, 3, ..., 10]
from_str   = list("Python")       # ['P', 'y', 't', 'h', 'o', 'n']

print(numbers)       # [1, 2, 3, 4, 5]
print(from_range)    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(from_str)      # ['P', 'y', 't', 'h', 'o', 'n']
```

### 3.1.2 索引与切片

Python 使用基于 0 的正向索引，也支持从 -1 开始的负向索引。

```python
fruits = ["apple", "banana", "cherry", "date", "elderberry"]

# 正向索引
print(fruits[0])    # apple
print(fruits[2])    # cherry
print(fruits[-1])   # elderberry（最后一个）
print(fruits[-2])   # date（倒数第二个）

# 切片：list[start:stop:step]
# start 包含，stop 不包含
print(fruits[1:3])    # ['banana', 'cherry']
print(fruits[:3])     # ['apple', 'banana', 'cherry']（从头开始）
print(fruits[2:])     # ['cherry', 'date', 'elderberry']（到末尾）
print(fruits[::2])    # ['apple', 'cherry', 'elderberry']（步长为2）
print(fruits[::-1])   # 反转列表

# 二维列表索引
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(matrix[1][2])   # 6（第2行第3列）
```

**切片示意图：**

```
索引：   0       1        2        3       4
      "apple" "banana" "cherry" "date" "elderberry"
负索引：  -5      -4       -3       -2      -1
```

### 3.1.3 修改列表

```python
scores = [85, 92, 78, 90, 88]

# 修改单个元素
scores[2] = 80
print(scores)    # [85, 92, 80, 90, 88]

# 修改切片
scores[1:3] = [95, 82]
print(scores)    # [85, 95, 82, 90, 88]

# 删除元素
del scores[0]
print(scores)    # [95, 82, 90, 88]
```

### 3.1.4 常用方法

```python
nums = [3, 1, 4, 1, 5, 9, 2, 6]

# 添加元素
nums.append(5)          # 末尾添加单个元素
nums.extend([7, 8])     # 末尾添加多个元素
nums.insert(0, 0)       # 在索引0处插入0

# 删除元素
nums.remove(1)          # 删除第一个值为1的元素
popped = nums.pop()     # 删除并返回最后一个元素
popped2 = nums.pop(0)   # 删除并返回索引0处的元素

# 查找
idx = nums.index(5)     # 返回值5的索引
count = nums.count(1)   # 统计值1出现的次数

# 排序
nums.sort()             # 原地升序排序
nums.sort(reverse=True) # 原地降序排序
sorted_nums = sorted(nums)  # 返回新的排序列表，原列表不变

# 反转
nums.reverse()          # 原地反转

# 其他
length = len(nums)      # 获取长度
nums.clear()            # 清空列表
copy = nums.copy()      # 浅拷贝

print(f"长度: {length}")
```

**方法速查表：**

| 方法 | 功能 | 返回值 |
|------|------|--------|
| `append(x)` | 末尾添加元素 | None |
| `extend(iterable)` | 末尾添加多个元素 | None |
| `insert(i, x)` | 在位置 i 插入元素 | None |
| `remove(x)` | 删除第一个值为 x 的元素 | None |
| `pop(i=-1)` | 删除并返回位置 i 的元素 | 被删除的元素 |
| `index(x)` | 返回第一个 x 的索引 | int |
| `count(x)` | 统计 x 出现次数 | int |
| `sort()` | 原地排序 | None |
| `reverse()` | 原地反转 | None |
| `copy()` | 浅拷贝 | list |
| `clear()` | 清空列表 | None |

---

## 3.2 元组（Tuple）

元组用圆括号 `()` 表示，与列表的关键区别是**不可变**：创建后不能修改元素。这种不可变性使元组更安全，且性能略优于列表。

### 3.2.1 创建元组

```python
# 空元组
empty = ()
empty2 = tuple()

# 单元素元组（注意末尾的逗号）
single = (42,)      # 正确
wrong  = (42)       # 这是整数，不是元组！

# 多元素元组
point = (3, 4)
rgb   = (255, 128, 0)

# 不带括号也可以（逗号分隔）
coords = 10, 20, 30
print(type(coords))  # <class 'tuple'>

# 由列表转换
t = tuple([1, 2, 3])
```

### 3.2.2 不可变性

```python
point = (3, 4)

# 可以读取元素
print(point[0])   # 3
print(point[1])   # 4

# 不能修改元素
# point[0] = 10   # TypeError: 'tuple' object does not support item assignment

# 但如果元组内包含可变对象（如列表），该对象内部可以修改
data = ([1, 2], [3, 4])
data[0].append(99)
print(data)    # ([1, 2, 99], [3, 4])
```

### 3.2.3 元组解包

解包是元组最强大的特性之一，可以将元组的元素同时赋给多个变量。

```python
# 基本解包
point = (3, 4)
x, y = point
print(f"x={x}, y={y}")   # x=3, y=4

# 三维坐标
coords = (1, 2, 3)
x, y, z = coords

# 交换变量（Python 经典技巧）
a, b = 10, 20
a, b = b, a
print(a, b)   # 20 10

# 扩展解包（使用 * 收集剩余元素）
first, *rest = [1, 2, 3, 4, 5]
print(first)   # 1
print(rest)    # [2, 3, 4, 5]

head, *middle, last = [1, 2, 3, 4, 5]
print(head, middle, last)   # 1 [2, 3, 4] 5

# 函数返回多个值（本质上是元组）
def min_max(numbers):
    return min(numbers), max(numbers)

low, high = min_max([3, 1, 4, 1, 5, 9])
print(f"最小值: {low}, 最大值: {high}")   # 最小值: 1, 最大值: 9
```

### 3.2.4 命名元组

`collections.namedtuple` 让元组的字段拥有名字，代码更具可读性。

```python
from collections import namedtuple

# 定义命名元组类型
Point = namedtuple("Point", ["x", "y"])
Color = namedtuple("Color", ["red", "green", "blue"])

# 创建实例
p = Point(3, 4)
c = Color(255, 128, 0)

# 既可以通过名字访问，也可以通过索引访问
print(p.x, p.y)         # 3 4
print(p[0], p[1])       # 3 4
print(c.red)            # 255

# 解包依然有效
x, y = p
print(x, y)             # 3 4

# 转换为字典
print(p._asdict())      # {'x': 3, 'y': 4}

# 实际应用：深度学习训练样本
Sample = namedtuple("Sample", ["image", "label", "filename"])
sample = Sample(image=[0.1, 0.5, 0.9], label=1, filename="cat_001.jpg")
print(sample.label)     # 1
print(sample.filename)  # cat_001.jpg
```

---

## 3.3 字典（Dict）

字典是**键值对**（key-value）的集合，用花括号 `{}` 表示。键必须唯一且不可变（字符串、数字、元组均可），值可以是任意类型。字典在 Python 3.7+ 中保持插入顺序。

### 3.3.1 创建字典

```python
# 字面量方式
person = {
    "name": "Alice",
    "age": 30,
    "city": "Beijing"
}

# dict() 构造函数
config = dict(lr=0.001, batch_size=32, epochs=100)

# 从键值对列表构建
pairs = [("a", 1), ("b", 2), ("c", 3)]
d = dict(pairs)

# 空字典
empty = {}
empty2 = dict()

print(person["name"])    # Alice
print(config["lr"])      # 0.001
```

### 3.3.2 访问与修改

```python
student = {"name": "Bob", "score": 85, "grade": "B"}

# 访问（键不存在会抛出 KeyError）
print(student["name"])    # Bob

# 安全访问（键不存在返回默认值，不抛出异常）
print(student.get("age"))           # None
print(student.get("age", 18))       # 18（自定义默认值）

# 添加或修改
student["score"] = 90               # 修改已有键
student["email"] = "bob@example.com"  # 添加新键
print(student)

# 删除
del student["grade"]                # 删除指定键
removed = student.pop("email")      # 删除并返回值
print(removed)                      # bob@example.com

# 检查键是否存在
print("name" in student)            # True
print("age" in student)             # False
```

### 3.3.3 常用方法

```python
inventory = {"apple": 50, "banana": 30, "cherry": 20}

# 获取所有键、值、键值对
keys   = list(inventory.keys())     # ['apple', 'banana', 'cherry']
values = list(inventory.values())   # [50, 30, 20]
items  = list(inventory.items())    # [('apple', 50), ('banana', 30), ('cherry', 20)]

# 遍历
for fruit, count in inventory.items():
    print(f"{fruit}: {count}个")

# 更新（合并另一个字典）
extra = {"grape": 15, "apple": 60}   # apple 会被覆盖
inventory.update(extra)
print(inventory)  # {'apple': 60, 'banana': 30, 'cherry': 20, 'grape': 15}

# setdefault：键不存在时设置默认值，键存在时返回原值
inventory.setdefault("mango", 0)
print(inventory["mango"])   # 0

# Python 3.9+ 合并字典
d1 = {"a": 1, "b": 2}
d2 = {"b": 3, "c": 4}
merged = d1 | d2             # {'a': 1, 'b': 3, 'c': 4}
d1 |= d2                     # 原地合并

# 长度和清空
print(len(inventory))
inventory.clear()
```

### 3.3.4 嵌套字典

```python
# 深度学习模型配置（嵌套字典）
model_config = {
    "architecture": "ResNet50",
    "optimizer": {
        "type": "Adam",
        "lr": 0.001,
        "betas": (0.9, 0.999)
    },
    "training": {
        "batch_size": 32,
        "epochs": 100,
        "early_stopping": {
            "patience": 10,
            "monitor": "val_loss"
        }
    },
    "data": {
        "train_size": 0.8,
        "val_size": 0.1,
        "test_size": 0.1
    }
}

# 访问嵌套值
lr = model_config["optimizer"]["lr"]
patience = model_config["training"]["early_stopping"]["patience"]
print(f"学习率: {lr}")        # 学习率: 0.001
print(f"早停轮数: {patience}") # 早停轮数: 10
```

### 3.3.5 字典推导式

```python
# 基本形式：{key_expr: value_expr for item in iterable}
squares = {x: x**2 for x in range(1, 6)}
print(squares)    # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# 带条件过滤
even_squares = {x: x**2 for x in range(1, 11) if x % 2 == 0}
print(even_squares)   # {2: 4, 4: 16, 6: 36, 8: 64, 10: 100}

# 反转键值对
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
print(inverted)   # {1: 'a', 2: 'b', 3: 'c'}

# 实际应用：将类别名称映射为索引
class_names = ["cat", "dog", "bird", "fish"]
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
idx_to_class = {idx: name for name, idx in class_to_idx.items()}
print(class_to_idx)   # {'cat': 0, 'dog': 1, 'bird': 2, 'fish': 3}
print(idx_to_class)   # {0: 'cat', 1: 'dog', 2: 'bird', 3: 'fish'}
```

---

## 3.4 集合（Set）

集合是**无序**、**不重复**元素的集合，用花括号 `{}` 表示（注意：空集合必须用 `set()`，因为 `{}` 创建的是空字典）。

### 3.4.1 创建集合

```python
# 字面量方式（自动去重）
fruits = {"apple", "banana", "cherry", "apple", "banana"}
print(fruits)    # {'cherry', 'apple', 'banana'}（顺序不固定）

# 从列表创建（常用于去重）
nums_with_dup = [1, 2, 2, 3, 3, 3, 4]
unique_nums = set(nums_with_dup)
print(unique_nums)   # {1, 2, 3, 4}

# 空集合
empty = set()   # 注意：不能用 {}，那是空字典

# 集合中只能存放不可变元素
valid   = {1, "hello", (1, 2)}      # 合法
# invalid = {1, [2, 3]}            # TypeError: unhashable type: 'list'
```

### 3.4.2 集合操作

```python
a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7}

# 添加和删除
a.add(6)
a.remove(1)         # 元素不存在时抛出 KeyError
a.discard(99)       # 元素不存在时不报错
popped = a.pop()    # 随机删除并返回一个元素

# 成员检测（时间复杂度 O(1)，比列表快得多）
print(3 in a)       # True / False

# 遍历（顺序不固定）
for x in {1, 2, 3}:
    print(x)
```

### 3.4.3 集合运算

```python
a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7}

# 并集（union）：a 或 b 中的元素
print(a | b)              # {1, 2, 3, 4, 5, 6, 7}
print(a.union(b))         # 同上

# 交集（intersection）：a 和 b 共有的元素
print(a & b)              # {3, 4, 5}
print(a.intersection(b))  # 同上

# 差集（difference）：在 a 中但不在 b 中
print(a - b)              # {1, 2}
print(a.difference(b))    # 同上

# 对称差集（symmetric difference）：在 a 或 b 中，但不同时在两者中
print(a ^ b)                       # {1, 2, 6, 7}
print(a.symmetric_difference(b))   # 同上

# 子集与超集
c = {3, 4}
print(c.issubset(a))      # True（c 是 a 的子集）
print(a.issuperset(c))    # True（a 是 c 的超集）
print(a.isdisjoint({8, 9}))  # True（没有共同元素）
```

**集合运算示意图：**

```
a = {1, 2, 3, 4, 5}       b = {3, 4, 5, 6, 7}

a | b  →  {1, 2, 3, 4, 5, 6, 7}   （并集）
a & b  →  {3, 4, 5}                （交集）
a - b  →  {1, 2}                   （差集）
a ^ b  →  {1, 2, 6, 7}             （对称差集）
```

### 3.4.4 实际应用：去重与集合运算

```python
# 场景：找出同时购买了两款产品的用户
product_a_buyers = {"user1", "user2", "user3", "user5"}
product_b_buyers = {"user2", "user3", "user4", "user6"}

# 两款都买了的用户
both = product_a_buyers & product_b_buyers
print("两款都买:", both)      # {'user2', 'user3'}

# 只买了 A 的用户
only_a = product_a_buyers - product_b_buyers
print("只买 A:", only_a)      # {'user1', 'user5'}

# 至少买了一款的用户总数
total_buyers = product_a_buyers | product_b_buyers
print("总用户数:", len(total_buyers))   # 6
```

---

## 3.5 推导式

推导式（Comprehension）是 Python 的一大特色，用简洁的语法从可迭代对象构建新的数据结构。

### 3.5.1 列表推导式

```python
# 基本语法：[expression for item in iterable if condition]

# 平方数列表
squares = [x**2 for x in range(10)]
print(squares)   # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 带条件过滤：偶数的平方
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(even_squares)   # [0, 4, 16, 36, 64]

# 字符串处理
words = ["hello", "WORLD", "Python"]
lower_words = [w.lower() for w in words]
print(lower_words)   # ['hello', 'world', 'python']

# 嵌套推导式（展平二维列表）
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [x for row in matrix for x in row]
print(flat)   # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 等价的 for 循环写法（可读性对比）
flat_loop = []
for row in matrix:
    for x in row:
        flat_loop.append(x)
```

### 3.5.2 字典推导式

```python
# 基本语法：{key_expr: value_expr for item in iterable if condition}

# 单词长度映射
words = ["cat", "elephant", "dog", "butterfly"]
word_lengths = {w: len(w) for w in words}
print(word_lengths)
# {'cat': 3, 'elephant': 8, 'dog': 3, 'butterfly': 9}

# 过滤长单词
long_words = {w: len(w) for w in words if len(w) > 3}
print(long_words)   # {'elephant': 8, 'butterfly': 9}

# 批量构建归一化参数
raw_params = {"weight": 2.5, "bias": -0.3, "scale": 1.8}
normalized = {k: round(v / max(raw_params.values()), 4)
              for k, v in raw_params.items()}
print(normalized)
```

### 3.5.3 集合推导式

```python
# 基本语法：{expression for item in iterable if condition}

# 不重复的平方数
nums = [1, -1, 2, -2, 3, -3]
abs_set = {abs(x) for x in nums}
print(abs_set)   # {1, 2, 3}

# 过滤奇数
odd_set = {x for x in range(20) if x % 2 != 0}
print(odd_set)   # {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}（顺序不固定）
```

### 3.5.4 生成器表达式（Generator Expression）

生成器表达式与列表推导式语法类似，但用圆括号，**惰性求值**，节省内存，适合处理大数据。

```python
# 列表推导式：立即计算，全部存入内存
squares_list = [x**2 for x in range(1_000_000)]   # 占用大量内存

# 生成器表达式：惰性求值，按需生成
squares_gen = (x**2 for x in range(1_000_000))    # 几乎不占内存

# 使用方式相同，但只能迭代一次
total = sum(squares_gen)
print(total)

# 实际应用：流式处理大型数据集
def load_batch(paths):
    """模拟从文件路径流式加载数据"""
    return ({"path": p, "data": len(p)} for p in paths)

file_paths = [f"data/image_{i}.jpg" for i in range(100)]
batch_gen = load_batch(file_paths)
first_item = next(batch_gen)
print(first_item)
```

### 3.5.5 推导式性能对比

```python
import time

N = 10_000_000

# 方式1：for 循环
start = time.time()
result = []
for x in range(N):
    if x % 2 == 0:
        result.append(x * x)
print(f"for 循环: {time.time() - start:.3f}s")

# 方式2：列表推导式
start = time.time()
result = [x * x for x in range(N) if x % 2 == 0]
print(f"列表推导式: {time.time() - start:.3f}s")

# 列表推导式通常比 for 循环快 15%~30%，因为底层做了优化
```

---

## 本章小结

| 数据结构 | 符号 | 有序 | 可重复 | 可变 | 主要用途 |
|----------|------|------|--------|------|----------|
| 列表 List | `[]` | 是 | 是 | 是 | 序列数据、有序集合 |
| 元组 Tuple | `()` | 是 | 是 | 否 | 不变数据、函数多返回值 |
| 字典 Dict | `{}` | 是* | 键不重复 | 是 | 键值映射、配置参数 |
| 集合 Set | `{}` | 否 | 否 | 是 | 去重、集合运算 |

> *Python 3.7+ 字典保持插入顺序。

**推导式语法对比：**

| 类型 | 语法 | 示例 |
|------|------|------|
| 列表推导式 | `[expr for x in it if cond]` | `[x**2 for x in range(5)]` |
| 字典推导式 | `{k: v for x in it if cond}` | `{x: x**2 for x in range(5)}` |
| 集合推导式 | `{expr for x in it if cond}` | `{x**2 for x in range(5)}` |
| 生成器表达式 | `(expr for x in it if cond)` | `(x**2 for x in range(5))` |

---

## 深度学习应用：批量数据的组织方式

在深度学习项目中，合理使用 Python 数据结构可以清晰地组织训练数据、标签、超参数和模型配置。下面展示几种常见的数据组织模式。

### 应用 3-A：图像分类数据集组织

```python
# ============================================================
# 图像分类任务的数据组织
# ============================================================

# 1. 用列表存储训练样本
train_images = [
    "data/train/cat/cat_001.jpg",
    "data/train/cat/cat_002.jpg",
    "data/train/dog/dog_001.jpg",
    "data/train/dog/dog_002.jpg",
]

train_labels = [0, 0, 1, 1]   # 0=cat, 1=dog

# 2. 用字典建立类别映射
class_to_idx = {"cat": 0, "dog": 1}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# 3. 用命名元组封装单个样本
from collections import namedtuple
Sample = namedtuple("Sample", ["image_path", "label", "class_name"])

dataset = [
    Sample(img, lbl, idx_to_class[lbl])
    for img, lbl in zip(train_images, train_labels)
]

# 4. 查看数据集信息
print(f"样本总数: {len(dataset)}")
print(f"第一个样本: {dataset[0]}")
print(f"标签分布: { {cls: train_labels.count(idx)
                    for cls, idx in class_to_idx.items()} }")

# 输出：
# 样本总数: 4
# 第一个样本: Sample(image_path='data/train/cat/cat_001.jpg', label=0, class_name='cat')
# 标签分布: {'cat': 2, 'dog': 2}
```

### 应用 3-B：超参数配置管理

```python
# ============================================================
# 用嵌套字典管理模型超参数
# ============================================================

hyperparams = {
    "model": {
        "name": "CNN",
        "in_channels": 3,
        "num_classes": 10,
        "dropout_rate": 0.5
    },
    "optimizer": {
        "type": "Adam",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": {
            "type": "CosineAnnealing",
            "T_max": 100
        }
    },
    "training": {
        "batch_size": 64,
        "epochs": 200,
        "num_workers": 4,
        "pin_memory": True
    },
    "augmentation": {
        "horizontal_flip": 0.5,
        "random_crop": True,
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
}

# 安全读取嵌套配置（避免 KeyError）
def get_config(config, *keys, default=None):
    """从嵌套字典中安全获取值"""
    current = config
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current

lr = get_config(hyperparams, "optimizer", "lr")
t_max = get_config(hyperparams, "optimizer", "scheduler", "T_max")
dropout = get_config(hyperparams, "model", "dropout_rate")

print(f"学习率: {lr}")          # 学习率: 0.001
print(f"余弦退火周期: {t_max}") # 余弦退火周期: 100
print(f"Dropout: {dropout}")    # Dropout: 0.5
```

### 应用 3-C：Mini-Batch 数据流水线

```python
# ============================================================
# 用列表和字典构建 mini-batch 数据流水线
# ============================================================

import random

def create_dataset(num_samples: int, num_classes: int) -> list:
    """创建模拟数据集（返回字典列表）"""
    dataset = []
    for i in range(num_samples):
        sample = {
            "id": i,
            "features": [random.gauss(0, 1) for _ in range(4)],  # 4维特征
            "label": random.randint(0, num_classes - 1)
        }
        dataset.append(sample)
    return dataset

def get_batches(dataset: list, batch_size: int, shuffle: bool = True):
    """将数据集切分为 mini-batches（生成器）"""
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch = [dataset[i] for i in batch_indices]

        # 将字典列表转换为批量张量格式
        yield {
            "features": [s["features"] for s in batch],
            "labels":   [s["label"]    for s in batch],
            "ids":      [s["id"]       for s in batch]
        }

def get_label_distribution(dataset: list) -> dict:
    """统计各类别样本数量"""
    distribution = {}
    for sample in dataset:
        label = sample["label"]
        distribution[label] = distribution.get(label, 0) + 1
    return distribution

# 使用示例
random.seed(42)
dataset = create_dataset(num_samples=100, num_classes=3)

# 查看类别分布
dist = get_label_distribution(dataset)
print("类别分布:", dist)

# 遍历批次
batch_size = 16
total_batches = 0
for batch in get_batches(dataset, batch_size=batch_size):
    total_batches += 1
    # 实际训练中这里会调用 model(batch["features"])

print(f"总批次数: {total_batches}")  # ceil(100 / 16) = 7
print(f"最后一批大小: {len(batch['labels'])}")
```

### 应用 3-D：词汇表与词嵌入索引

```python
# ============================================================
# 用字典和集合构建 NLP 词汇表
# ============================================================

def build_vocab(sentences: list, min_freq: int = 2) -> dict:
    """
    从句子列表构建词汇表。
    特殊符号：<PAD>=0, <UNK>=1
    """
    # 统计词频
    word_freq = {}
    for sentence in sentences:
        for word in sentence.split():
            word_freq[word] = word_freq.get(word, 0) + 1

    # 过滤低频词
    vocab_words = {w for w, freq in word_freq.items() if freq >= min_freq}

    # 构建词 -> 索引映射（特殊符号在前）
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, word in enumerate(sorted(vocab_words), start=2):
        vocab[word] = idx

    return vocab

def encode_sentence(sentence: str, vocab: dict, max_len: int = 10) -> list:
    """将句子编码为整数序列，不足补 PAD，超长截断"""
    tokens = sentence.split()[:max_len]
    ids = [vocab.get(w, vocab["<UNK>"]) for w in tokens]
    # 补 PAD
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids

# 示例数据
corpus = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "a cat and a dog",
    "the cat and the dog are friends",
]

vocab = build_vocab(corpus, min_freq=2)
print(f"词汇表大小: {len(vocab)}")
print(f"词汇表: {vocab}")

encoded = encode_sentence("the cat ran fast", vocab, max_len=6)
print(f"编码结果: {encoded}")
# 'ran' 出现2次，'fast' 出现不足2次（被编为 <UNK>）
```

---

## 练习题

### 基础题

**练习 1：列表操作**

给定一个整数列表 `data = [5, 3, 8, 1, 9, 2, 7, 4, 6, 10]`，完成以下操作：
1. 取出前5个元素
2. 取出索引为奇数位置的元素（索引 1, 3, 5, ...）
3. 将列表升序排序（不修改原列表）
4. 找出大于 5 的元素，用列表推导式实现

**练习 2：字典操作**

已有一个学生成绩字典：
```python
grades = {
    "Alice": [85, 92, 78],
    "Bob":   [90, 88, 95],
    "Carol": [72, 65, 80]
}
```
1. 计算每个学生的平均分，用字典推导式生成 `{姓名: 平均分}` 的新字典
2. 找出平均分最高的学生姓名

---

### 进阶题

**练习 3：集合运算**

有两个班级的学生名单：
```python
class_a = {"Alice", "Bob", "Carol", "Dave", "Eve"}
class_b = {"Bob", "Dave", "Frank", "Grace", "Henry"}
```
1. 找出两个班都有的学生
2. 找出只在 A 班的学生
3. 找出参加了任意一个班的所有学生
4. 找出只在一个班（不同时在两个班）的学生

**练习 4：数据结构综合应用**

编写一个函数 `analyze_dataset(samples)`，接受如下格式的样本列表：
```python
samples = [
    {"id": 1, "label": "cat",  "confidence": 0.92},
    {"id": 2, "label": "dog",  "confidence": 0.87},
    {"id": 3, "label": "cat",  "confidence": 0.76},
    {"id": 4, "label": "bird", "confidence": 0.95},
    {"id": 5, "label": "dog",  "confidence": 0.61},
]
```
函数需要返回一个字典，包含：
- `"total"`: 样本总数
- `"label_counts"`: 每种标签的数量（字典）
- `"avg_confidence"`: 每种标签的平均置信度（字典）
- `"high_confidence_ids"`: 置信度 >= 0.85 的样本 id 列表

---

### 挑战题

**练习 5：Mini-Batch 生成器**

实现一个 `DataLoader` 类，满足以下要求：

```python
class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        """
        dataset:    字典列表，每个字典包含 "x"（特征列表）和 "y"（标签）
        batch_size: 每批大小
        shuffle:    是否在每个 epoch 开始时打乱数据
        """
        ...

    def __iter__(self):
        """返回批次迭代器，每次产出一个字典：
           {"x": [[...], ...], "y": [...]}
        """
        ...

    def __len__(self):
        """返回总批次数（向上取整）"""
        ...
```

要求：
1. 支持 `for batch in loader` 的迭代方式
2. `len(loader)` 返回正确的批次数
3. 当 `shuffle=True` 时，每次遍历顺序随机
4. 最后一批不足 `batch_size` 时，仍然返回（不丢弃）
5. 编写测试代码验证上述行为

---

## 练习答案

### 答案 1：列表操作

```python
data = [5, 3, 8, 1, 9, 2, 7, 4, 6, 10]

# 1. 前5个元素
first_five = data[:5]
print("前5个:", first_five)   # [5, 3, 8, 1, 9]

# 2. 奇数索引位置的元素
odd_indexed = data[1::2]
print("奇数索引:", odd_indexed)   # [3, 1, 2, 4, 10]

# 3. 升序排序（不修改原列表）
sorted_data = sorted(data)
print("排序后:", sorted_data)   # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("原列表:", data)          # [5, 3, 8, 1, 9, 2, 7, 4, 6, 10]（未改变）

# 4. 大于5的元素（列表推导式）
greater_than_5 = [x for x in data if x > 5]
print("大于5:", greater_than_5)   # [8, 9, 7, 6, 10]
```

### 答案 2：字典操作

```python
grades = {
    "Alice": [85, 92, 78],
    "Bob":   [90, 88, 95],
    "Carol": [72, 65, 80]
}

# 1. 计算平均分（字典推导式）
averages = {name: sum(scores) / len(scores)
            for name, scores in grades.items()}
print("平均分:", averages)
# {'Alice': 85.0, 'Bob': 91.0, 'Carol': 72.33...}

# 2. 平均分最高的学生
top_student = max(averages, key=averages.get)
print(f"最高分: {top_student}（{averages[top_student]:.2f}分）")
# 最高分: Bob（91.00分）
```

### 答案 3：集合运算

```python
class_a = {"Alice", "Bob", "Carol", "Dave", "Eve"}
class_b = {"Bob", "Dave", "Frank", "Grace", "Henry"}

# 1. 两班都有
both = class_a & class_b
print("两班都有:", both)        # {'Bob', 'Dave'}

# 2. 只在A班
only_a = class_a - class_b
print("只在A班:", only_a)       # {'Alice', 'Carol', 'Eve'}

# 3. 任意一班的所有学生
all_students = class_a | class_b
print("所有学生:", all_students)   # 共8人

# 4. 只在一个班（对称差集）
unique_students = class_a ^ class_b
print("只在一班:", unique_students)  # {'Alice', 'Carol', 'Eve', 'Frank', 'Grace', 'Henry'}
```

### 答案 4：数据结构综合应用

```python
def analyze_dataset(samples):
    """分析样本数据集，返回统计信息字典"""

    # 统计标签数量
    label_counts = {}
    for sample in samples:
        label = sample["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    # 统计每种标签的置信度总和
    confidence_sums = {}
    for sample in samples:
        label = sample["label"]
        confidence_sums[label] = (
            confidence_sums.get(label, 0.0) + sample["confidence"]
        )

    # 计算每种标签的平均置信度
    avg_confidence = {
        label: round(confidence_sums[label] / label_counts[label], 4)
        for label in label_counts
    }

    # 找出高置信度样本的 id
    high_confidence_ids = [
        s["id"] for s in samples if s["confidence"] >= 0.85
    ]

    return {
        "total":               len(samples),
        "label_counts":        label_counts,
        "avg_confidence":      avg_confidence,
        "high_confidence_ids": high_confidence_ids
    }


# 测试
samples = [
    {"id": 1, "label": "cat",  "confidence": 0.92},
    {"id": 2, "label": "dog",  "confidence": 0.87},
    {"id": 3, "label": "cat",  "confidence": 0.76},
    {"id": 4, "label": "bird", "confidence": 0.95},
    {"id": 5, "label": "dog",  "confidence": 0.61},
]

result = analyze_dataset(samples)
print("总样本数:", result["total"])             # 5
print("标签分布:", result["label_counts"])      # {'cat': 2, 'dog': 2, 'bird': 1}
print("平均置信度:", result["avg_confidence"])  # {'cat': 0.84, 'dog': 0.74, 'bird': 0.95}
print("高置信度ID:", result["high_confidence_ids"])  # [1, 2, 4]
```

### 答案 5：Mini-Batch 生成器

```python
import random
import math

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle

    def __iter__(self):
        """每次迭代产出一个 mini-batch"""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch_samples = [self.dataset[i] for i in batch_indices]

            yield {
                "x": [s["x"] for s in batch_samples],
                "y": [s["y"] for s in batch_samples]
            }

    def __len__(self):
        """向上取整，最后一批不满也算一批"""
        return math.ceil(len(self.dataset) / self.batch_size)


# 测试代码
if __name__ == "__main__":
    random.seed(0)

    # 创建模拟数据集：20个样本，每个样本有3个特征
    dataset = [
        {"x": [random.gauss(0, 1) for _ in range(3)],
         "y": random.randint(0, 1)}
        for _ in range(20)
    ]

    loader = DataLoader(dataset, batch_size=6, shuffle=True)

    print(f"总批次数: {len(loader)}")   # ceil(20/6) = 4

    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}:")
        for i, batch in enumerate(loader):
            print(f"  批次 {i+1}: x shape ({len(batch['x'])} x {len(batch['x'][0])}), "
                  f"y={batch['y']}")

    # 预期输出（顺序因 shuffle 而不同）：
    # 总批次数: 4
    # Epoch 1:
    #   批次 1: x shape (6 x 3), y=[...]
    #   批次 2: x shape (6 x 3), y=[...]
    #   批次 3: x shape (6 x 3), y=[...]
    #   批次 4: x shape (2 x 3), y=[...]   ← 最后一批只有2个
    # Epoch 2:
    #   批次 1: x shape (6 x 3), y=[...]   ← 顺序与 Epoch 1 不同
    #   ...
```

---

*下一章：[第4章：函数与模块](./04-functions-modules.md)*
