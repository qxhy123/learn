# 附录A：Python速查表

本速查表汇总了 Python 编程中最常用的语法与内置工具，适合快速查阅与复习。

---

## 1. 基本语法速查

### 1.1 变量与数据类型

| 类型 | 示例 | 说明 |
|------|------|------|
| `int` | `x = 42` | 整数 |
| `float` | `x = 3.14` | 浮点数 |
| `complex` | `x = 2 + 3j` | 复数 |
| `bool` | `x = True` | 布尔值（`True` / `False`） |
| `str` | `x = "hello"` | 字符串 |
| `bytes` | `x = b"hello"` | 字节串 |
| `list` | `x = [1, 2, 3]` | 列表（可变） |
| `tuple` | `x = (1, 2, 3)` | 元组（不可变） |
| `set` | `x = {1, 2, 3}` | 集合（无序、不重复） |
| `dict` | `x = {"a": 1}` | 字典（键值对） |
| `NoneType` | `x = None` | 空值 |

```python
# 类型检查
type(x)           # 返回类型对象
isinstance(x, int)  # 判断是否为某类型

# 多重赋值
a, b, c = 1, 2, 3
a, *rest = [1, 2, 3, 4]   # a=1, rest=[2, 3, 4]
a, b = b, a                 # 交换变量
```

---

### 1.2 运算符

**算术运算符**

| 运算符 | 说明 | 示例 |
|--------|------|------|
| `+` | 加法 | `3 + 2 = 5` |
| `-` | 减法 | `3 - 2 = 1` |
| `*` | 乘法 | `3 * 2 = 6` |
| `/` | 除法（浮点） | `7 / 2 = 3.5` |
| `//` | 整除 | `7 // 2 = 3` |
| `%` | 取余 | `7 % 2 = 1` |
| `**` | 幂运算 | `2 ** 3 = 8` |

**比较运算符**

| 运算符 | 说明 |
|--------|------|
| `==` | 等于 |
| `!=` | 不等于 |
| `<` / `>` | 小于 / 大于 |
| `<=` / `>=` | 小于等于 / 大于等于 |
| `is` | 对象同一性 |
| `is not` | 对象非同一性 |
| `in` | 成员测试 |
| `not in` | 非成员测试 |

**逻辑运算符**

| 运算符 | 说明 | 示例 |
|--------|------|------|
| `and` | 逻辑与 | `True and False` → `False` |
| `or` | 逻辑或 | `True or False` → `True` |
| `not` | 逻辑非 | `not True` → `False` |

**位运算符**

| 运算符 | 说明 |
|--------|------|
| `&` | 按位与 |
| `\|` | 按位或 |
| `^` | 按位异或 |
| `~` | 按位取反 |
| `<<` | 左移 |
| `>>` | 右移 |

---

### 1.3 字符串操作

```python
s = "Hello, Python!"

# 常用方法
s.upper()           # "HELLO, PYTHON!"
s.lower()           # "hello, python!"
s.strip()           # 去除两端空白
s.lstrip()          # 去除左端空白
s.rstrip()          # 去除右端空白
s.replace("o", "0") # "Hell0, Pyth0n!"
s.split(", ")       # ["Hello", "Python!"]
s.startswith("He")  # True
s.endswith("!")     # True
s.find("Python")    # 7（找不到返回 -1）
s.index("Python")   # 7（找不到抛出异常）
s.count("l")        # 3
s.join(["a","b","c"])  # 以 s 为分隔符连接列表
"  hello  ".strip() # "hello"
s.isdigit()         # False
s.isalpha()         # False
s.isalnum()         # False

# 切片
s[0]       # 'H'
s[-1]      # '!'
s[0:5]     # 'Hello'
s[::2]     # 每隔一个取一个字符
s[::-1]    # 反转字符串
```

---

### 1.4 格式化输出

```python
name = "Alice"
age  = 30
pi   = 3.14159

# f-string（推荐，Python 3.6+）
print(f"姓名: {name}, 年龄: {age}")
print(f"Pi = {pi:.2f}")          # 保留两位小数
print(f"{age:05d}")              # 05位，零填充 → 00030
print(f"{1000000:,}")            # 千分位分隔 → 1,000,000
print(f"{'居中':^10}")           # 居中对齐，总宽10
print(f"{'左对齐':<10}")         # 左对齐
print(f"{'右对齐':>10}")         # 右对齐

# str.format()
print("姓名: {}, 年龄: {}".format(name, age))
print("姓名: {name}, 年龄: {age}".format(name=name, age=age))

# % 格式化（旧式）
print("姓名: %s, 年龄: %d" % (name, age))
print("Pi = %.2f" % pi)
```

---

## 2. 数据结构速查

### 2.1 列表方法

```python
lst = [3, 1, 4, 1, 5, 9, 2, 6]
```

| 方法 | 说明 | 示例 |
|------|------|------|
| `append(x)` | 末尾追加元素 | `lst.append(7)` |
| `insert(i, x)` | 在索引 i 处插入 | `lst.insert(0, 0)` |
| `extend(iterable)` | 追加可迭代对象 | `lst.extend([10, 11])` |
| `remove(x)` | 删除第一个 x | `lst.remove(1)` |
| `pop(i=-1)` | 删除并返回索引 i 的元素 | `lst.pop()` |
| `index(x)` | 返回第一个 x 的索引 | `lst.index(5)` |
| `count(x)` | 统计 x 出现次数 | `lst.count(1)` |
| `sort()` | 原地排序 | `lst.sort(reverse=True)` |
| `reverse()` | 原地反转 | `lst.reverse()` |
| `copy()` | 浅拷贝 | `lst2 = lst.copy()` |
| `clear()` | 清空列表 | `lst.clear()` |

```python
# 切片操作
lst[1:4]        # 索引 1 到 3
lst[::2]        # 步长为 2
lst[::-1]       # 反转

# 列表拼接与重复
[1, 2] + [3, 4]  # [1, 2, 3, 4]
[0] * 5          # [0, 0, 0, 0, 0]

# 排序（返回新列表）
sorted(lst)
sorted(lst, key=lambda x: -x)   # 降序

# 解包
first, *middle, last = [1, 2, 3, 4, 5]
```

---

### 2.2 字典方法

```python
d = {"name": "Alice", "age": 30, "city": "Beijing"}
```

| 方法 | 说明 | 示例 |
|------|------|------|
| `d[key]` | 取值（不存在则报错） | `d["name"]` |
| `d.get(key, default)` | 取值（不存在返回默认值） | `d.get("x", 0)` |
| `d[key] = val` | 设置键值 | `d["email"] = "..."` |
| `d.update(other)` | 合并另一字典 | `d.update({"x": 1})` |
| `d.pop(key)` | 删除并返回值 | `d.pop("age")` |
| `d.setdefault(key, default)` | 键不存在时设置默认值 | `d.setdefault("x", 0)` |
| `d.keys()` | 返回所有键视图 | `list(d.keys())` |
| `d.values()` | 返回所有值视图 | `list(d.values())` |
| `d.items()` | 返回所有键值对视图 | `list(d.items())` |
| `d.copy()` | 浅拷贝 | `d2 = d.copy()` |
| `d.clear()` | 清空字典 | `d.clear()` |

```python
# 合并字典（Python 3.9+）
merged = d1 | d2

# 遍历
for key, value in d.items():
    print(f"{key}: {value}")

# 字典推导式
squared = {k: v**2 for k, v in {"a": 1, "b": 2}.items()}
```

---

### 2.3 集合方法

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
```

| 方法/运算符 | 说明 | 示例 |
|-------------|------|------|
| `add(x)` | 添加元素 | `a.add(5)` |
| `remove(x)` | 删除元素（不存在则报错） | `a.remove(1)` |
| `discard(x)` | 删除元素（不存在不报错） | `a.discard(99)` |
| `pop()` | 随机删除并返回一个元素 | `a.pop()` |
| `union` / `\|` | 并集 | `a \| b` |
| `intersection` / `&` | 交集 | `a & b` |
| `difference` / `-` | 差集 | `a - b` |
| `symmetric_difference` / `^` | 对称差集 | `a ^ b` |
| `issubset` / `<=` | 子集判断 | `a <= b` |
| `issuperset` / `>=` | 超集判断 | `a >= b` |
| `isdisjoint` | 是否不相交 | `a.isdisjoint(b)` |
| `copy()` | 浅拷贝 | `a.copy()` |

```python
# 冻结集合（不可变）
fs = frozenset([1, 2, 3])
```

---

### 2.4 推导式语法

```python
# 列表推导式
squares = [x**2 for x in range(10)]
evens   = [x for x in range(20) if x % 2 == 0]
matrix  = [[i*j for j in range(1, 4)] for i in range(1, 4)]

# 字典推导式
word_len = {word: len(word) for word in ["apple", "banana", "cherry"]}

# 集合推导式
unique_squares = {x**2 for x in [-2, -1, 0, 1, 2]}

# 生成器表达式（惰性求值，节省内存）
gen = (x**2 for x in range(1000000))
total = sum(x**2 for x in range(100))
```

---

## 3. 控制流速查

### 3.1 条件语句

```python
# if / elif / else
if x > 0:
    print("正数")
elif x < 0:
    print("负数")
else:
    print("零")

# 三元表达式
result = "偶数" if x % 2 == 0 else "奇数"

# match-case（Python 3.10+，结构化模式匹配）
match command:
    case "quit":
        quit()
    case "hello":
        print("Hello!")
    case _:
        print("未知命令")

# match 匹配数据结构
match point:
    case (0, 0):
        print("原点")
    case (x, 0):
        print(f"x 轴，x={x}")
    case (0, y):
        print(f"y 轴，y={y}")
    case (x, y):
        print(f"任意点 ({x}, {y})")
```

---

### 3.2 循环语句

```python
# for 循环
for i in range(5):          # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 10, 2):   # 2, 4, 6, 8
    print(i)

# enumerate：带索引遍历
for i, val in enumerate(["a", "b", "c"], start=1):
    print(i, val)

# zip：并行遍历
for x, y in zip([1, 2, 3], ["a", "b", "c"]):
    print(x, y)

# while 循环
n = 10
while n > 0:
    print(n)
    n -= 1

# break / continue / else
for i in range(10):
    if i == 3:
        continue       # 跳过当前迭代
    if i == 7:
        break          # 退出循环
else:
    print("循环正常结束（未触发 break）")

# 嵌套循环与标志
found = False
for row in matrix:
    for val in row:
        if val == target:
            found = True
            break
    if found:
        break
```

---

### 3.3 异常处理

```python
# 基本结构
try:
    result = 10 / 0
except ZeroDivisionError:
    print("除零错误")
except (TypeError, ValueError) as e:
    print(f"类型或值错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")
else:
    print("无异常时执行")
finally:
    print("无论如何都执行")

# 主动抛出异常
raise ValueError("无效的输入")
raise RuntimeError("运行时错误") from original_error

# 自定义异常
class MyError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code

# 上下文管理器抑制异常
from contextlib import suppress
with suppress(FileNotFoundError):
    open("不存在的文件.txt")
```

**常见内置异常**

| 异常 | 触发场景 |
|------|----------|
| `ValueError` | 值不合法 |
| `TypeError` | 类型不匹配 |
| `KeyError` | 字典键不存在 |
| `IndexError` | 列表索引越界 |
| `AttributeError` | 对象属性不存在 |
| `FileNotFoundError` | 文件不存在 |
| `ZeroDivisionError` | 除以零 |
| `ImportError` | 模块导入失败 |
| `StopIteration` | 迭代器耗尽 |
| `RuntimeError` | 通用运行时错误 |
| `NotImplementedError` | 方法未实现 |

---

## 4. 函数速查

### 4.1 函数定义

```python
# 基本定义
def greet(name):
    """文档字符串（docstring）"""
    return f"你好, {name}!"

# 带返回值的函数
def add(a, b):
    return a + b

# 返回多个值（实际返回元组）
def min_max(lst):
    return min(lst), max(lst)

low, high = min_max([3, 1, 4, 1, 5])
```

---

### 4.2 参数类型

```python
# 位置参数
def func(a, b, c):
    pass

# 默认参数（默认参数必须在位置参数之后）
def greet(name, greeting="你好"):
    return f"{greeting}, {name}!"

# 关键字参数调用
greet(name="Alice", greeting="Hi")

# 可变位置参数 *args
def total(*args):
    return sum(args)

total(1, 2, 3, 4)  # 10

# 可变关键字参数 **kwargs
def show_info(**kwargs):
    for key, val in kwargs.items():
        print(f"{key}: {val}")

show_info(name="Alice", age=30)

# 组合使用
def func(pos1, pos2, *args, kw_only, **kwargs):
    pass

# 仅限位置参数（Python 3.8+，/ 之前为仅限位置）
def func(pos_only, /, normal, *, kw_only):
    pass

# 参数解包
args   = [1, 2, 3]
kwargs = {"name": "Alice"}
func(*args)
func(**kwargs)
```

---

### 4.3 lambda 表达式

```python
# 语法：lambda 参数: 表达式
square = lambda x: x ** 2
add    = lambda x, y: x + y

# 常见用途
nums = [3, 1, 4, 1, 5, 9]
nums.sort(key=lambda x: -x)                    # 按负值排序（降序）
sorted_words = sorted(words, key=lambda w: len(w))  # 按长度排序

pairs = [(1, "b"), (2, "a"), (3, "c")]
pairs.sort(key=lambda p: p[1])                 # 按第二个元素排序

# 与 map / filter 结合
squares  = list(map(lambda x: x**2, range(5)))
evens    = list(filter(lambda x: x % 2 == 0, range(10)))
```

---

### 4.4 装饰器语法

```python
# 基本装饰器
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("调用前")
        result = func(*args, **kwargs)
        print("调用后")
        return result
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# 使用 functools.wraps 保留元信息
from functools import wraps

def log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"调用 {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# 带参数的装饰器（装饰器工厂）
def repeat(n):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def hello():
    print("Hello!")

# 常用内置装饰器
class MyClass:
    @staticmethod
    def static_method():       # 静态方法，不接收 self/cls
        pass

    @classmethod
    def class_method(cls):     # 类方法，接收 cls
        pass

    @property
    def value(self):           # 将方法作为属性访问
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
```

---

## 5. 面向对象速查

### 5.1 类定义

```python
class Animal:
    # 类变量（所有实例共享）
    species = "动物"

    def __init__(self, name, age):
        # 实例变量
        self.name = name
        self.age  = age
        self._private = "约定私有"      # 约定私有
        self.__mangled = "名称改写"     # 名称改写

    def speak(self):
        raise NotImplementedError

    def __str__(self):
        return f"{self.name} ({self.age}岁)"

    def __repr__(self):
        return f"Animal(name={self.name!r}, age={self.age!r})"
```

---

### 5.2 魔术方法（Dunder Methods）

| 方法 | 触发场景 |
|------|----------|
| `__init__(self, ...)` | 实例化时调用 |
| `__str__(self)` | `str(obj)` / `print(obj)` |
| `__repr__(self)` | `repr(obj)` / 调试器显示 |
| `__len__(self)` | `len(obj)` |
| `__getitem__(self, key)` | `obj[key]` |
| `__setitem__(self, key, val)` | `obj[key] = val` |
| `__delitem__(self, key)` | `del obj[key]` |
| `__contains__(self, item)` | `item in obj` |
| `__iter__(self)` | `iter(obj)` / for 循环 |
| `__next__(self)` | `next(obj)` |
| `__call__(self, ...)` | `obj(...)` |
| `__eq__(self, other)` | `obj == other` |
| `__lt__(self, other)` | `obj < other` |
| `__add__(self, other)` | `obj + other` |
| `__enter__(self)` | `with` 语句进入 |
| `__exit__(self, ...)` | `with` 语句退出 |
| `__hash__(self)` | `hash(obj)` |
| `__bool__(self)` | `bool(obj)` |
| `__del__(self)` | 对象被垃圾回收时 |

```python
# 示例：实现上下文管理器
class FileManager:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.file = open(self.filename)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        return False  # 不抑制异常

with FileManager("data.txt") as f:
    content = f.read()
```

---

### 5.3 继承语法

```python
# 单继承
class Dog(Animal):
    def __init__(self, name, age, breed):
        super().__init__(name, age)   # 调用父类 __init__
        self.breed = breed

    def speak(self):
        return "汪汪!"

# 多继承
class C(A, B):
    pass

# 方法解析顺序（MRO）
print(C.__mro__)      # 查看继承链

# 抽象基类
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        import math
        return math.pi * self.radius ** 2

    def perimeter(self):
        import math
        return 2 * math.pi * self.radius

# dataclass（Python 3.7+）
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float
    label: str = ""
    tags: list = field(default_factory=list)

    def distance_to_origin(self):
        return (self.x**2 + self.y**2) ** 0.5

p = Point(3.0, 4.0, label="A")
```

---

## 6. 常用内置函数

### 6.1 类型转换

| 函数 | 说明 | 示例 |
|------|------|------|
| `int(x)` | 转整数 | `int("42")` → `42` |
| `float(x)` | 转浮点数 | `float("3.14")` → `3.14` |
| `str(x)` | 转字符串 | `str(42)` → `"42"` |
| `bool(x)` | 转布尔值 | `bool(0)` → `False` |
| `list(x)` | 转列表 | `list("abc")` → `['a','b','c']` |
| `tuple(x)` | 转元组 | `tuple([1,2])` → `(1,2)` |
| `set(x)` | 转集合 | `set([1,1,2])` → `{1,2}` |
| `dict(x)` | 转字典 | `dict([("a",1)])` → `{"a":1}` |
| `bytes(x)` | 转字节串 | `bytes("hi","utf-8")` |
| `chr(n)` | 整数转字符 | `chr(65)` → `"A"` |
| `ord(c)` | 字符转整数 | `ord("A")` → `65` |
| `hex(n)` | 整数转十六进制字符串 | `hex(255)` → `"0xff"` |
| `bin(n)` | 整数转二进制字符串 | `bin(10)` → `"0b1010"` |
| `oct(n)` | 整数转八进制字符串 | `oct(8)` → `"0o10"` |

---

### 6.2 序列操作

| 函数 | 说明 | 示例 |
|------|------|------|
| `len(x)` | 返回长度 | `len([1,2,3])` → `3` |
| `range(stop)` | 生成整数序列 | `range(5)` → `0..4` |
| `range(start, stop, step)` | 带起点和步长 | `range(1,10,2)` |
| `enumerate(x)` | 带索引遍历 | `enumerate(["a","b"])` |
| `zip(*iterables)` | 并行遍历 | `zip([1,2],["a","b"])` |
| `sorted(x)` | 返回排序后的新列表 | `sorted([3,1,2])` |
| `reversed(x)` | 返回反向迭代器 | `list(reversed([1,2,3]))` |
| `min(x)` | 最小值 | `min([3,1,2])` → `1` |
| `max(x)` | 最大值 | `max([3,1,2])` → `3` |
| `sum(x)` | 求和 | `sum([1,2,3])` → `6` |
| `any(x)` | 任意为真则返回 True | `any([0,1,0])` → `True` |
| `all(x)` | 全部为真则返回 True | `all([1,1,0])` → `False` |
| `map(func, x)` | 映射函数 | `list(map(str,[1,2]))` |
| `filter(func, x)` | 过滤序列 | `list(filter(bool,[0,1,2]))` |

---

### 6.3 数学函数

```python
import math

math.sqrt(16)       # 4.0，平方根
math.ceil(3.2)      # 4，向上取整
math.floor(3.8)     # 3，向下取整
math.trunc(3.9)     # 3，截断取整
math.fabs(-3.5)     # 3.5，绝对值（返回浮点）
math.factorial(5)   # 120，阶乘
math.gcd(12, 8)     # 4，最大公约数
math.lcm(4, 6)      # 12，最小公倍数（Python 3.9+）
math.log(100, 10)   # 2.0，以 10 为底的对数
math.log2(8)        # 3.0
math.log10(1000)    # 3.0
math.exp(1)         # e ≈ 2.718
math.pow(2, 10)     # 1024.0（返回浮点）
math.pi             # π ≈ 3.14159
math.e              # e ≈ 2.71828
math.inf            # 正无穷大
math.isfinite(x)    # 判断是否有限
math.isinf(x)       # 判断是否无穷
math.isnan(x)       # 判断是否 NaN

# 内置函数
abs(-5)             # 5
round(3.14159, 2)   # 3.14
pow(2, 10)          # 1024（整数）
divmod(17, 5)       # (3, 2)，商和余数
```

---

### 6.4 迭代工具

```python
from itertools import (
    count, cycle, repeat,
    chain, islice, takewhile, dropwhile,
    product, permutations, combinations, combinations_with_replacement,
    groupby, accumulate
)

# 无限迭代器
count(10)          # 10, 11, 12, ...
cycle("ABC")       # A, B, C, A, B, C, ...
repeat(5, 3)       # 5, 5, 5（重复 3 次）

# 有限迭代器
chain([1,2], [3,4])                    # 1, 2, 3, 4
islice(range(100), 5)                  # 0, 1, 2, 3, 4
takewhile(lambda x: x < 5, range(10)) # 0, 1, 2, 3, 4
dropwhile(lambda x: x < 5, range(10)) # 5, 6, 7, 8, 9

# 组合迭代器
list(product("AB", repeat=2))          # AA, AB, BA, BB
list(permutations("ABC", 2))           # AB, AC, BA, BC, CA, CB
list(combinations("ABC", 2))           # AB, AC, BC
list(combinations_with_replacement("AB", 2))  # AA, AB, BB

# 累积
list(accumulate([1, 2, 3, 4]))        # [1, 3, 6, 10]，前缀和

# functools 工具
from functools import reduce, partial, lru_cache

reduce(lambda a, b: a + b, [1,2,3,4])  # 10
double = partial(pow, 2)                # double(10) → 1024

@lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)
```

---

## 7. 文件操作速查

### 7.1 文件读写

```python
# 打开模式
# "r"  只读（默认）
# "w"  写入（覆盖）
# "a"  追加
# "x"  独占创建（文件已存在则报错）
# "b"  二进制模式（配合 r/w/a 使用）
# "t"  文本模式（默认）
# "+"  读写模式

# 推荐使用 with 语句（自动关闭文件）
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()          # 读取全部内容为字符串
    lines   = f.readlines()     # 读取为行列表
    line    = f.readline()      # 读取一行

# 逐行读取（内存友好）
with open("data.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line.strip())

# 写入文件
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("第一行\n")
    f.writelines(["行1\n", "行2\n", "行3\n"])

# 追加写入
with open("log.txt", "a", encoding="utf-8") as f:
    f.write("新增日志\n")

# 读取二进制文件
with open("image.png", "rb") as f:
    data = f.read()

# JSON 文件
import json

with open("data.json", "r", encoding="utf-8") as f:
    obj = json.load(f)

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, indent=2)

# 序列化为字符串
json_str = json.dumps(obj, ensure_ascii=False)
obj      = json.loads(json_str)

# CSV 文件
import csv

with open("data.csv", "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)

with open("data.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "age"])
    writer.writeheader()
    writer.writerow({"name": "Alice", "age": 30})
```

---

### 7.2 路径操作

```python
from pathlib import Path

# 创建路径对象
p = Path("/Users/alice/documents")
p = Path.home()               # 用户主目录
p = Path.cwd()                # 当前工作目录

# 路径拼接
file_path = p / "data" / "file.txt"

# 路径属性
file_path.name        # "file.txt"（文件名含扩展名）
file_path.stem        # "file"（文件名不含扩展名）
file_path.suffix      # ".txt"（扩展名）
file_path.suffixes    # [".tar", ".gz"]（多个扩展名）
file_path.parent      # 父目录路径
file_path.parts       # 路径各部分的元组
file_path.anchor      # 根目录（"/" 或 "C:\\"）

# 路径判断
file_path.exists()    # 是否存在
file_path.is_file()   # 是否为文件
file_path.is_dir()    # 是否为目录
file_path.is_symlink() # 是否为符号链接

# 目录操作
p.mkdir(parents=True, exist_ok=True)   # 创建目录（含父目录）
p.rmdir()                               # 删除空目录

# 文件操作
file_path.touch()                       # 创建空文件或更新时间戳
file_path.unlink(missing_ok=True)       # 删除文件
file_path.rename(p / "new_name.txt")    # 重命名
file_path.replace(p / "other.txt")      # 移动/覆盖

# 读写快捷方法
text    = file_path.read_text(encoding="utf-8")
content = file_path.read_bytes()
file_path.write_text("内容", encoding="utf-8")
file_path.write_bytes(b"data")

# 遍历目录
for item in p.iterdir():
    print(item)

# 递归遍历（glob）
for py_file in p.rglob("*.py"):
    print(py_file)

# os.path 常用函数（兼容旧代码）
import os.path

os.path.join("/usr", "local", "bin")   # "/usr/local/bin"
os.path.basename("/usr/local/bin")     # "bin"
os.path.dirname("/usr/local/bin")      # "/usr/local"
os.path.splitext("file.txt")          # ("file", ".txt")
os.path.abspath("relative/path")      # 转为绝对路径
os.path.expanduser("~/documents")     # 展开 ~ 符号
os.path.getsize("file.txt")           # 文件大小（字节）

# shutil：高级文件操作
import shutil

shutil.copy("src.txt", "dst.txt")          # 复制文件
shutil.copy2("src.txt", "dst.txt")         # 复制文件（含元数据）
shutil.copytree("src_dir", "dst_dir")      # 复制整个目录树
shutil.rmtree("target_dir")               # 递归删除目录
shutil.move("src", "dst")                  # 移动文件或目录
shutil.make_archive("backup", "zip", ".")  # 创建压缩包
```

---

## 附：常用标准库速览

| 模块 | 用途 |
|------|------|
| `os` | 操作系统接口（环境变量、进程、目录）|
| `sys` | Python 运行时（命令行参数、退出、路径）|
| `pathlib` | 面向对象的文件路径操作 |
| `shutil` | 高级文件和目录操作 |
| `re` | 正则表达式 |
| `json` | JSON 序列化与反序列化 |
| `csv` | CSV 文件读写 |
| `datetime` | 日期和时间处理 |
| `time` | 时间相关函数 |
| `math` | 数学函数 |
| `random` | 随机数生成 |
| `collections` | 高级数据结构（Counter、defaultdict、deque）|
| `itertools` | 迭代器工具 |
| `functools` | 高阶函数（reduce、partial、lru_cache）|
| `contextlib` | 上下文管理工具 |
| `abc` | 抽象基类 |
| `dataclasses` | 数据类（自动生成 `__init__` 等）|
| `typing` | 类型提示 |
| `copy` | 浅拷贝与深拷贝 |
| `pprint` | 美化打印 |
| `logging` | 日志记录 |
| `unittest` | 单元测试框架 |
| `argparse` | 命令行参数解析 |
| `subprocess` | 子进程管理 |
| `threading` | 多线程 |
| `multiprocessing` | 多进程 |
| `asyncio` | 异步 I/O |
| `socket` | 网络套接字 |
| `http.client` | HTTP 客户端 |
| `urllib` | URL 处理 |
| `hashlib` | 加密哈希（MD5、SHA）|
| `base64` | Base64 编码 |
| `struct` | 二进制数据结构 |
| `sqlite3` | SQLite 数据库接口 |
| `pickle` | Python 对象序列化 |
| `gzip` / `zipfile` / `tarfile` | 压缩文件处理 |
| `tempfile` | 临时文件和目录 |
| `io` | I/O 流 |

---

*本速查表基于 Python 3.10+ 语法。带 "Python 3.x+" 注释的特性需对应版本支持。*
