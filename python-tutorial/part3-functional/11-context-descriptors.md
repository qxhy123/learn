# 第11章：上下文管理器与描述符

## 学习目标

学完本章后，你将能够：

1. 理解上下文管理器协议，掌握 `__enter__` 和 `__exit__` 方法的用法
2. 熟练使用 `contextlib` 模块简化上下文管理器的编写
3. 理解描述符协议，掌握 `__get__`、`__set__`、`__delete__` 的语义
4. 区分数据描述符与非数据描述符，理解属性查找顺序
5. 在深度学习场景中应用上下文管理器（如自动混合精度训练）

---

## 11.1 上下文管理器协议（`__enter__`、`__exit__`）

### 11.1.1 为什么需要上下文管理器

在编写程序时，我们经常遇到"资源获取——使用——释放"的模式：

```python
# 没有上下文管理器的写法：容易遗忘释放资源
f = open("data.txt", "r")
data = f.read()
f.close()          # 如果 read() 抛出异常，这行永远不会执行！

# 改进：用 try/finally 保证释放
f = open("data.txt", "r")
try:
    data = f.read()
finally:
    f.close()      # 无论如何都会执行
```

`try/finally` 虽然正确，但代码冗长。*上下文管理器*（Context Manager）提供了更简洁的语法——`with` 语句：

```python
# 使用 with 语句：简洁、安全
with open("data.txt", "r") as f:
    data = f.read()
# 离开 with 块后，文件自动关闭
```

### 11.1.2 上下文管理器协议

任何实现了 `__enter__` 和 `__exit__` 两个方法的对象都是上下文管理器。

| 方法 | 何时调用 | 返回值 |
|------|----------|--------|
| `__enter__(self)` | 进入 `with` 块时 | 赋给 `as` 子句的变量 |
| `__exit__(self, exc_type, exc_val, exc_tb)` | 离开 `with` 块时（正常或异常） | `True` 表示吞掉异常，`False`/`None` 表示继续传播 |

`__exit__` 的三个参数：

- `exc_type`：异常类型（无异常时为 `None`）
- `exc_val`：异常实例（无异常时为 `None`）
- `exc_tb`：追溯对象（无异常时为 `None`）

### 11.1.3 实现第一个上下文管理器

```python
class Timer:
    """计时上下文管理器，用于测量代码块的执行时间。"""
    import time

    def __init__(self, name=""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        import time
        self._start = time.perf_counter()
        return self          # 把 self 赋给 as 子句

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.elapsed = time.perf_counter() - self._start
        label = f"[{self.name}] " if self.name else ""
        print(f"{label}耗时: {self.elapsed:.4f} 秒")
        return False         # 不吞掉异常，继续传播

# 使用示例
with Timer("矩阵乘法") as t:
    result = sum(i * i for i in range(1_000_000))

print(f"计算结果: {result}")
print(f"记录的耗时: {t.elapsed:.4f} 秒")
# 输出:
# [矩阵乘法] 耗时: 0.0523 秒
# 计算结果: 333332833333500000
# 记录的耗时: 0.0523 秒
```

### 11.1.4 异常处理：吞掉还是传播？

```python
class SuppressError:
    """选择性地抑制特定类型的异常。"""

    def __init__(self, *exception_types):
        self.exception_types = exception_types

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 如果是我们想抑制的异常类型，返回 True（吞掉异常）
        if exc_type is not None and issubclass(exc_type, self.exception_types):
            print(f"已抑制异常: {exc_type.__name__}: {exc_val}")
            return True
        # 其他情况返回 False（继续传播）
        return False


# 示例 1：成功抑制 ZeroDivisionError
with SuppressError(ZeroDivisionError):
    result = 1 / 0
    print("这行不会执行")
print("程序继续运行")
# 输出:
# 已抑制异常: ZeroDivisionError: division by zero
# 程序继续运行

# 示例 2：不抑制 ValueError
try:
    with SuppressError(ZeroDivisionError):
        raise ValueError("这是一个值错误")
except ValueError as e:
    print(f"捕获到: {e}")
# 输出:
# 捕获到: 这是一个值错误
```

### 11.1.5 资源管理模式：数据库连接示例

```python
import sqlite3

class DatabaseConnection:
    """管理数据库连接的上下文管理器，自动处理事务和连接关闭。"""

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        return self.cursor      # 直接返回 cursor，方便操作

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # 无异常：提交事务
            self.conn.commit()
            print("事务已提交")
        else:
            # 有异常：回滚事务
            self.conn.rollback()
            print(f"事务已回滚，原因: {exc_val}")
        self.conn.close()
        return False            # 异常继续传播，让调用者知道出错了


# 使用示例（概念演示，无需真实数据库文件）
# with DatabaseConnection(":memory:") as cursor:
#     cursor.execute("CREATE TABLE users (id INTEGER, name TEXT)")
#     cursor.execute("INSERT INTO users VALUES (1, 'Alice')")
#     # 离开 with 块时自动提交
```

---

## 11.2 `contextlib` 模块

`contextlib` 是 Python 标准库提供的工具集，可以更简便地创建上下文管理器。

### 11.2.1 `@contextmanager` 装饰器

最常用的方式是用生成器函数配合 `@contextmanager` 装饰器。`yield` 之前的代码对应 `__enter__`，`yield` 的值赋给 `as` 子句，`yield` 之后的代码对应 `__exit__`。

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name=""):
    """用生成器实现计时上下文管理器。"""
    start = time.perf_counter()
    try:
        yield               # 这里暂停，执行 with 块内的代码
    finally:
        elapsed = time.perf_counter() - start
        label = f"[{name}] " if name else ""
        print(f"{label}耗时: {elapsed:.4f} 秒")

with timer("快速排序"):
    data = sorted(range(100_000), reverse=True)
# 输出: [快速排序] 耗时: 0.0089 秒
```

处理异常的版本：

```python
@contextmanager
def managed_resource(resource_name):
    """演示如何在 contextmanager 中处理异常。"""
    print(f"获取资源: {resource_name}")
    resource = {"name": resource_name, "data": []}
    try:
        yield resource              # 把资源传给 with 块
    except Exception as e:
        print(f"操作失败，释放资源: {e}")
        raise                       # 重新抛出异常
    else:
        print(f"操作成功，释放资源: {resource_name}")
    finally:
        print(f"清理完成: {resource_name}")

with managed_resource("GPU内存") as res:
    res["data"].append(42)
    print(f"使用资源，当前数据: {res['data']}")
# 输出:
# 获取资源: GPU内存
# 使用资源，当前数据: [42]
# 操作成功，释放资源: GPU内存
# 清理完成: GPU内存
```

### 11.2.2 `contextlib.suppress`

用于抑制指定异常，等同于我们之前手写的 `SuppressError`：

```python
from contextlib import suppress
import os

# 删除文件时，如果文件不存在就忽略
with suppress(FileNotFoundError):
    os.remove("不存在的文件.txt")
print("程序继续，没有崩溃")
# 输出: 程序继续，没有崩溃
```

### 11.2.3 `contextlib.ExitStack`

`ExitStack` 允许动态管理多个上下文管理器，特别适合文件数量不定的场景：

```python
from contextlib import ExitStack

filenames = ["file1.txt", "file2.txt", "file3.txt"]

# 动态打开多个文件
with ExitStack() as stack:
    # 这里演示概念，实际使用时文件需要存在
    files = []
    for fname in filenames:
        try:
            f = stack.enter_context(open(fname, "w"))
            files.append(f)
        except FileNotFoundError:
            pass
    # 离开 with 块时，ExitStack 自动关闭所有已打开的文件
    print(f"打开了 {len(files)} 个文件")
```

### 11.2.4 `contextlib.nullcontext`

有时需要一个"什么都不做"的上下文管理器作为占位符：

```python
from contextlib import nullcontext

def process_data(data, use_timer=False):
    """根据参数决定是否计时。"""
    ctx = timer("处理数据") if use_timer else nullcontext()
    with ctx:
        result = [x ** 2 for x in data]
    return result

data = list(range(1000))
result1 = process_data(data, use_timer=True)   # 会打印耗时
result2 = process_data(data, use_timer=False)  # 静默执行
```

### 11.2.5 `contextlib.closing`

将拥有 `close()` 方法但没有实现上下文管理器协议的对象包装成上下文管理器：

```python
from contextlib import closing
from urllib.request import urlopen

# urlopen 在旧版本 Python 中不支持 with 语句
# closing() 确保离开 with 块时调用 close()
# with closing(urlopen("https://example.com")) as page:
#     content = page.read()

# 自定义示例
class LegacyResource:
    """模拟只有 close() 方法的旧式资源。"""
    def __init__(self, name):
        self.name = name
        print(f"打开: {self.name}")

    def do_work(self):
        print(f"工作中: {self.name}")

    def close(self):
        print(f"关闭: {self.name}")

with closing(LegacyResource("旧式数据库")) as res:
    res.do_work()
# 输出:
# 打开: 旧式数据库
# 工作中: 旧式数据库
# 关闭: 旧式数据库
```

---

## 11.3 描述符协议（`__get__`、`__set__`、`__delete__`）

### 11.3.1 什么是描述符

*描述符*（Descriptor）是一种对象，它定义了另一个对象属性的访问行为。当一个类的属性是描述符时，访问该属性会触发描述符的方法，而不是直接读写字典。

描述符协议由三个方法组成：

| 方法 | 触发时机 | 签名 |
|------|----------|------|
| `__get__` | 读取属性时 | `__get__(self, obj, objtype=None)` |
| `__set__` | 设置属性时 | `__set__(self, obj, value)` |
| `__delete__` | 删除属性时 | `__delete__(self, obj)` |

其中：
- `self`：描述符实例本身
- `obj`：拥有该属性的对象实例（通过类访问时为 `None`）
- `objtype`：拥有该属性的类

### 11.3.2 第一个描述符示例

```python
class Validator:
    """验证数值范围的描述符。"""

    def __init__(self, name, min_val, max_val):
        self.name = name          # 属性名（用于错误信息）
        self.min_val = min_val
        self.max_val = max_val
        self.private_name = f"_{name}"   # 实际存储数据的属性名

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self           # 通过类访问时，返回描述符本身
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} 必须是数字，得到 {type(value).__name__}")
        if not self.min_val <= value <= self.max_val:
            raise ValueError(
                f"{self.name} 必须在 [{self.min_val}, {self.max_val}] 范围内，"
                f"得到 {value}"
            )
        setattr(obj, self.private_name, value)

    def __delete__(self, obj):
        print(f"删除属性 {self.name}")
        delattr(obj, self.private_name)


class NeuralNetworkConfig:
    """神经网络配置类，使用描述符验证超参数。"""
    learning_rate = Validator("learning_rate", 1e-6, 1.0)
    dropout_rate  = Validator("dropout_rate",  0.0,  0.9)
    batch_size    = Validator("batch_size",    1,    4096)

    def __init__(self, lr, dropout, batch):
        self.learning_rate = lr       # 触发 Validator.__set__
        self.dropout_rate  = dropout
        self.batch_size    = batch


# 正常使用
config = NeuralNetworkConfig(lr=0.001, dropout=0.5, batch=32)
print(f"学习率: {config.learning_rate}")   # 触发 __get__
print(f"批大小: {config.batch_size}")
# 输出:
# 学习率: 0.001
# 批大小: 32

# 触发验证错误
try:
    config.learning_rate = 5.0      # 超出范围
except ValueError as e:
    print(f"错误: {e}")
# 输出: 错误: learning_rate 必须在 [1e-06, 1.0] 范围内，得到 5.0

try:
    config.batch_size = "large"     # 类型错误
except TypeError as e:
    print(f"错误: {e}")
# 输出: 错误: batch_size 必须是数字，得到 str

# 通过类访问，返回描述符本身
print(type(NeuralNetworkConfig.learning_rate))
# 输出: <class '__main__.Validator'>
```

### 11.3.3 `__set_name__` 方法

Python 3.6+ 提供了 `__set_name__` 方法，在描述符被分配给类属性时自动调用，简化了名称管理：

```python
class TypedAttribute:
    """带类型检查的描述符，自动获取属性名。"""

    def __init__(self, expected_type):
        self.expected_type = expected_type
        self.name = None          # 将由 __set_name__ 填充

    def __set_name__(self, owner, name):
        """在类创建时自动调用，owner 是拥有此描述符的类。"""
        self.name = name
        self.private_name = f"_{name}"
        print(f"描述符 {name!r} 绑定到类 {owner.__name__!r}")

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"属性 {self.name!r} 期望类型 {self.expected_type.__name__}，"
                f"得到 {type(value).__name__}"
            )
        setattr(obj, self.private_name, value)


class Model:
    name         = TypedAttribute(str)
    num_layers   = TypedAttribute(int)
    hidden_size  = TypedAttribute(int)

    def __init__(self, name, num_layers, hidden_size):
        self.name        = name
        self.num_layers  = num_layers
        self.hidden_size = hidden_size

# 输出（类定义时）:
# 描述符 'name' 绑定到类 'Model'
# 描述符 'num_layers' 绑定到类 'Model'
# 描述符 'hidden_size' 绑定到类 'Model'

m = Model("ResNet", 50, 512)
print(f"{m.name}: {m.num_layers} 层, 隐藏维度 {m.hidden_size}")
# 输出: ResNet: 50 层, 隐藏维度 512

try:
    m.num_layers = "fifty"
except TypeError as e:
    print(f"类型错误: {e}")
# 输出: 类型错误: 属性 'num_layers' 期望类型 int，得到 str
```

---

## 11.4 数据描述符与非数据描述符

### 11.4.1 两种描述符的定义

Python 根据描述符实现的方法，将其分为两类：

| 类型 | 条件 | 优先级 |
|------|------|--------|
| **数据描述符** | 定义了 `__set__` 或 `__delete__` | 高于实例 `__dict__` |
| **非数据描述符** | 只定义了 `__get__` | 低于实例 `__dict__` |

这个区别决定了**属性查找时的优先级**，是理解描述符行为的关键。

### 11.4.2 非数据描述符示例：函数方法

Python 中的普通函数就是非数据描述符——它只实现了 `__get__`：

```python
# 演示函数作为描述符的行为
def greet(self):
    return f"你好，我是 {self.name}"

class Person:
    name = "Alice"
    greet = greet       # 把函数赋给类属性

p = Person()

# 通过实例访问：触发 greet.__get__(p, Person)，返回绑定方法
bound_method = p.greet
print(type(bound_method))     # <class 'method'>
print(bound_method())          # 你好，我是 Alice

# 通过类访问：触发 greet.__get__(None, Person)，返回函数本身
unbound = Person.greet
print(type(unbound))           # <class 'function'>
print(unbound(p))               # 你好，我是 Alice
```

### 11.4.3 数据描述符 vs 实例属性：优先级对比

```python
class DataDescriptor:
    """数据描述符：有 __set__，优先级高于实例 __dict__。"""

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        print(f"DataDescriptor.__get__ 被调用")
        return obj.__dict__.get("_x", "未设置")

    def __set__(self, obj, value):
        print(f"DataDescriptor.__set__ 被调用，值={value}")
        obj.__dict__["_x"] = value


class NonDataDescriptor:
    """非数据描述符：只有 __get__，优先级低于实例 __dict__。"""

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        print(f"NonDataDescriptor.__get__ 被调用")
        return "来自描述符的值"


class MyClass:
    data     = DataDescriptor()       # 数据描述符
    non_data = NonDataDescriptor()    # 非数据描述符


obj = MyClass()

# --- 测试数据描述符 ---
obj.data = 42
# 输出: DataDescriptor.__set__ 被调用，值=42

print(obj.data)
# 输出: DataDescriptor.__get__ 被调用
#       42

# 尝试直接写实例 __dict__（绕过描述符）
obj.__dict__["data"] = 999
print(obj.data)
# 输出: DataDescriptor.__get__ 被调用
#       42
# 注意：数据描述符优先级更高，实例 __dict__ 的 999 被忽略！

# --- 测试非数据描述符 ---
print(obj.non_data)
# 输出: NonDataDescriptor.__get__ 被调用
#       来自描述符的值

# 直接写实例 __dict__（覆盖描述符）
obj.__dict__["non_data"] = "实例自己的值"
print(obj.non_data)
# 输出: 实例自己的值
# 注意：非数据描述符被实例 __dict__ 中的值遮蔽！
```

### 11.4.4 `property` 是数据描述符

Python 内置的 `property` 就是数据描述符的实现：

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        """半径（米）"""
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("半径不能为负数")
        self._radius = value

    @property
    def area(self):
        """面积（只读）"""
        import math
        return math.pi * self._radius ** 2


c = Circle(5.0)
print(f"半径: {c.radius}")        # 输出: 半径: 5.0
print(f"面积: {c.area:.4f}")      # 输出: 面积: 78.5398

c.radius = 10.0
print(f"新半径: {c.radius}")      # 输出: 新半径: 10.0

try:
    c.radius = -1
except ValueError as e:
    print(f"错误: {e}")           # 输出: 错误: 半径不能为负数

# property 是数据描述符的证明
print(hasattr(property, "__set__"))    # 输出: True
print(hasattr(property, "__delete__")) # 输出: True
```

---

## 11.5 属性查找顺序与描述符应用

### 11.5.1 完整的属性查找顺序

当访问 `obj.attr` 时，Python 按以下顺序查找：

```
1. type(obj).__mro__ 中是否有【数据描述符】
   ↓ 没有
2. obj.__dict__ 中是否有该属性
   ↓ 没有
3. type(obj).__mro__ 中是否有【非数据描述符】或普通类属性
   ↓ 没有
4. 抛出 AttributeError
```

用代码验证：

```python
class DataDesc:
    def __get__(self, obj, objtype=None):
        return "数据描述符"
    def __set__(self, obj, value):
        pass

class NonDataDesc:
    def __get__(self, obj, objtype=None):
        return "非数据描述符"

class Demo:
    data_attr    = DataDesc()
    nondata_attr = NonDataDesc()

obj = Demo()

# 在实例 __dict__ 中手动放值
obj.__dict__["data_attr"]    = "实例字典"
obj.__dict__["nondata_attr"] = "实例字典"

# 验证优先级
print(obj.data_attr)        # 输出: 数据描述符（数据描述符优先）
print(obj.nondata_attr)     # 输出: 实例字典（实例字典优先）
```

### 11.5.2 用描述符实现懒加载（Lazy Loading）

```python
class lazy_property:
    """
    懒加载描述符：第一次访问时计算，之后直接从实例缓存返回。
    这是非数据描述符（只有 __get__），所以实例 __dict__ 会覆盖它。
    """

    def __init__(self, func):
        self.func = func
        self.attr_name = None

    def __set_name__(self, owner, name):
        self.attr_name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # 第一次访问：计算值，存入实例 __dict__
        value = self.func(obj)
        obj.__dict__[self.attr_name] = value    # 存入缓存
        print(f"计算并缓存了 {self.attr_name!r}")
        return value


class HeavyModel:
    """模拟需要昂贵计算才能初始化的模型属性。"""

    def __init__(self, config):
        self.config = config

    @lazy_property
    def weight_matrix(self):
        """权重矩阵：第一次访问时初始化（昂贵操作）。"""
        import time
        time.sleep(0.01)   # 模拟耗时初始化
        size = self.config.get("size", 100)
        return [[0.0] * size for _ in range(size)]

    @lazy_property
    def bias_vector(self):
        """偏置向量：第一次访问时初始化。"""
        size = self.config.get("size", 100)
        return [0.0] * size


model = HeavyModel({"size": 10})

print("创建模型完成，尚未初始化权重")
w = model.weight_matrix           # 第一次访问：触发计算
# 输出: 计算并缓存了 'weight_matrix'
w2 = model.weight_matrix          # 第二次访问：直接从 __dict__ 读取（不打印）
print(f"两次访问是同一对象: {w is w2}")   # 输出: True
```

### 11.5.3 用描述符实现单位换算

```python
class UnitConverter:
    """
    单位换算描述符，统一在内部以基础单位存储，
    通过不同描述符提供不同单位的视图。
    """

    def __init__(self, storage_attr, factor):
        """
        storage_attr: 内部存储用的属性名（基础单位）
        factor: 此单位与基础单位的换算系数（此单位 * factor = 基础单位）
        """
        self.storage_attr = storage_attr
        self.factor = factor

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        base_value = getattr(obj, self.storage_attr, 0.0)
        return base_value / self.factor

    def __set__(self, obj, value):
        setattr(obj, self.storage_attr, value * self.factor)


class Distance:
    """距离类，内部以米存储，提供多种单位的读写接口。"""
    # 1 千米 = 1000 米
    km      = UnitConverter("_meters", 1000)
    # 1 英里 = 1609.344 米
    miles   = UnitConverter("_meters", 1609.344)
    # 1 米 = 1 米（基础单位视图）
    meters  = UnitConverter("_meters", 1)

    def __init__(self, meters=0.0):
        self._meters = meters


d = Distance()
d.km = 1.0                        # 设置为 1 千米
print(f"千米:   {d.km:.4f}")      # 输出: 千米:   1.0000
print(f"米:     {d.meters:.4f}")  # 输出: 米:     1000.0000
print(f"英里:   {d.miles:.4f}")   # 输出: 英里:   0.6214

d.miles = 26.2188                 # 马拉松距离（英里）
print(f"马拉松: {d.km:.3f} 千米") # 输出: 马拉松: 42.195 千米
```

### 11.5.4 描述符应用总结

描述符在 Python 框架中无处不在：

```python
# 以下都是描述符的应用
class Examples:
    # 1. property：内置数据描述符
    @property
    def value(self):
        return self._value

    # 2. classmethod：非数据描述符，返回绑定到类的方法
    @classmethod
    def create(cls):
        return cls()

    # 3. staticmethod：非数据描述符，返回普通函数
    @staticmethod
    def helper():
        pass

# 验证它们都实现了描述符协议
print(hasattr(property, "__get__"))        # True
print(hasattr(classmethod, "__get__"))     # True
print(hasattr(staticmethod, "__get__"))    # True
```

---

## 本章小结

| 概念 | 关键点 | 典型应用 |
|------|--------|----------|
| `__enter__` | 进入 `with` 块时调用，返回值赋给 `as` | 资源获取、状态保存 |
| `__exit__` | 离开 `with` 块时调用，接收异常信息 | 资源释放、异常处理 |
| 返回 `True` vs `False` | `True` 吞掉异常，`False`/`None` 传播 | 异常策略控制 |
| `@contextmanager` | 用生成器代替类实现上下文管理器 | 简化代码、内联场景 |
| `contextlib.suppress` | 静默抑制特定异常 | 可选操作的错误忽略 |
| `contextlib.ExitStack` | 动态管理多个上下文管理器 | 动态资源列表 |
| `__get__` | 读取属性时触发 | 计算属性、懒加载 |
| `__set__` | 写入属性时触发 | 数据验证、单位换算 |
| `__delete__` | 删除属性时触发 | 清理副作用 |
| `__set_name__` | 类定义时自动绑定名称 | 避免手动传名 |
| 数据描述符 | 有 `__set__`/`__delete__`，优先于实例字典 | `property`、验证器 |
| 非数据描述符 | 只有 `__get__`，低于实例字典优先级 | 函数/方法、懒加载 |
| 查找顺序 | 数据描述符 > 实例字典 > 非数据描述符 > 报错 | 调试属性访问问题 |

---

## 深度学习应用：自动混合精度训练

### 背景

*自动混合精度*（Automatic Mixed Precision，AMP）是现代深度学习中的重要优化技术。它在前向传播中使用 float16（半精度）计算来加速并节省显存，同时使用 float32（单精度）维护数值稳定性。PyTorch 通过 `torch.cuda.amp` 模块以上下文管理器的形式暴露这一功能，是学习上下文管理器的绝佳实例。

### AMP 中的上下文管理器

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from contextlib import contextmanager
import time

# ─────────────────────────────────────────────
# 1. 简单演示 autocast 上下文管理器
# ─────────────────────────────────────────────

def demo_autocast():
    """演示 autocast 如何切换数据类型。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(4, 16, device=device)
    linear = nn.Linear(16, 8).to(device)

    print("=== autocast 关闭 ===")
    with torch.no_grad():
        out = linear(x)
        print(f"权重类型: {linear.weight.dtype}")   # torch.float32
        print(f"输出类型: {out.dtype}")              # torch.float32

    print("\n=== autocast 开启（cuda 设备）===")
    with torch.no_grad():
        with autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.float32):
            out_amp = linear(x)
            print(f"权重类型: {linear.weight.dtype}")      # torch.float32（权重不变）
            print(f"输出类型: {out_amp.dtype}")             # torch.float16（自动降精度）


# ─────────────────────────────────────────────
# 2. 完整的 AMP 训练循环
# ─────────────────────────────────────────────

class SimpleNet(nn.Module):
    """用于演示 AMP 的简单全连接网络。"""

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_with_amp(num_epochs=3, batch_size=64, use_amp=True):
    """
    使用自动混合精度的训练循环。

    关键组件：
    - autocast：前向传播中自动选择计算精度
    - GradScaler：防止 float16 梯度下溢（underflow）
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}，AMP: {use_amp}")

    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # GradScaler 用于缩放损失，防止 float16 梯度下溢
    # 如果不使用 AMP，创建禁用状态的 scaler（代码统一，无副作用）
    scaler = GradScaler(enabled=use_amp)

    total_time = 0.0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 10
        start = time.perf_counter()

        for batch_idx in range(num_batches):
            # 生成随机批次数据（模拟真实训练数据）
            inputs = torch.randn(batch_size, 128, device=device)
            labels = torch.randint(0, 10, (batch_size,), device=device)

            optimizer.zero_grad()

            # ── 关键部分：用 autocast 包裹前向传播 ──
            with autocast(device_type=device, enabled=use_amp,
                          dtype=torch.float16 if device == "cuda" else torch.bfloat16):
                outputs = model(inputs)          # float16 计算（GPU上）
                loss = criterion(outputs, labels) # 损失也在 float16 空间

            # ── 用 scaler 缩放损失并反向传播 ──
            # scaler.scale(loss) 将损失乘以一个大数，防止梯度变成 0
            scaler.scale(loss).backward()

            # ── 用 scaler 更新参数 ──
            # 内部会：1) 反缩放梯度  2) 检查是否有 inf/nan  3) 决定是否跳过本步
            scaler.step(optimizer)

            # ── 更新 scaler 的缩放因子 ──
            # 如果没有 inf/nan，缩放因子可能会增大；有则会缩小
            scaler.update()

            epoch_loss += loss.item()

        epoch_time = time.perf_counter() - start
        total_time += epoch_time
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"loss={avg_loss:.4f}, "
              f"时间={epoch_time:.3f}s, "
              f"scale={scaler.get_scale():.0f}")

    print(f"总训练时间: {total_time:.3f}s")
    return model


# ─────────────────────────────────────────────
# 3. 自定义 AMP 上下文管理器：封装更多逻辑
# ─────────────────────────────────────────────

class AMPTrainingContext:
    """
    封装 AMP 训练所需全部状态的上下文管理器。
    演示如何将多个资源组合成一个统一的上下文管理器。
    """

    def __init__(self, model, optimizer, use_amp=True, clip_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.clip_grad_norm = clip_grad_norm
        self.scaler = GradScaler(enabled=use_amp)
        self.device = next(model.parameters()).device.type
        self._autocast_ctx = None

    def __enter__(self):
        self.optimizer.zero_grad()
        # 启动 autocast 上下文
        self._autocast_ctx = autocast(
            device_type=self.device,
            enabled=self.use_amp,
            dtype=torch.float16 if self.device == "cuda" else torch.bfloat16
        )
        self._autocast_ctx.__enter__()
        return self                         # 返回自身供 as 子句使用

    def backward(self, loss):
        """执行缩放反向传播。"""
        self._autocast_ctx.__exit__(None, None, None)   # 退出 autocast
        self._autocast_ctx = None
        self.scaler.scale(loss).backward()

    def step(self):
        """执行梯度裁剪和参数更新。"""
        if self.clip_grad_norm is not None:
            # 先反缩放梯度，再裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 确保 autocast 已退出（如果 backward 未被调用）
        if self._autocast_ctx is not None:
            self._autocast_ctx.__exit__(exc_type, exc_val, exc_tb)
            self._autocast_ctx = None
        return False    # 不吞掉异常


def train_with_custom_amp_context(num_steps=5):
    """使用自定义 AMPTrainingContext 的训练循环。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\n使用自定义 AMPTrainingContext，设备: {device}")

    for step in range(num_steps):
        inputs = torch.randn(32, 128, device=device)
        labels = torch.randint(0, 10, (32,), device=device)

        with AMPTrainingContext(model, optimizer, use_amp=True) as amp_ctx:
            outputs = model(inputs)        # 在 autocast 内执行
            loss = criterion(outputs, labels)
            amp_ctx.backward(loss)         # 缩放反向传播
            amp_ctx.step()                 # 梯度裁剪 + 参数更新

        if step % 2 == 0:
            print(f"步骤 {step+1}: loss={loss.item():.4f}")


# ─────────────────────────────────────────────
# 4. 使用 @contextmanager 实现训练阶段切换
# ─────────────────────────────────────────────

@contextmanager
def training_phase(model, phase="train"):
    """
    在 with 块内自动切换模型的训练/评估模式。
    退出时恢复原始模式。
    """
    original_mode = model.training
    if phase == "train":
        model.train()
    elif phase == "eval":
        model.eval()
    else:
        raise ValueError(f"未知阶段: {phase!r}，应为 'train' 或 'eval'")

    try:
        yield model
    finally:
        # 无论发生什么，恢复原始模式
        model.train(original_mode)


def evaluate_model():
    """演示 training_phase 上下文管理器。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleNet().to(device)
    model.train()    # 初始处于训练模式

    print(f"\n初始模式: {'train' if model.training else 'eval'}")

    with training_phase(model, "eval"):
        print(f"with 块内: {'train' if model.training else 'eval'}")
        with torch.no_grad():
            dummy_input = torch.randn(4, 128, device=device)
            output = model(dummy_input)
            print(f"推理输出形状: {output.shape}")

    print(f"with 块外: {'train' if model.training else 'eval'}")
    # 输出:
    # 初始模式: train
    # with 块内: eval
    # 推理输出形状: torch.Size([4, 10])
    # with 块外: train


# ─────────────────────────────────────────────
# 主程序入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("1. autocast 数据类型演示")
    print("=" * 50)
    demo_autocast()

    print("\n" + "=" * 50)
    print("2. AMP 标准训练循环")
    print("=" * 50)
    train_with_amp(num_epochs=3, use_amp=True)

    print("\n" + "=" * 50)
    print("3. 自定义 AMPTrainingContext")
    print("=" * 50)
    train_with_custom_amp_context(num_steps=5)

    print("\n" + "=" * 50)
    print("4. 训练/评估模式切换")
    print("=" * 50)
    evaluate_model()
```

### AMP 核心概念总结

```
float32 权重（主副本）
      ↓  autocast 自动降精度
float16 前向计算（更快，更省显存）
      ↓
float16 损失值
      ↓  scaler.scale()：乘以缩放因子（如 65536）
大数值损失（防止梯度下溢）
      ↓  .backward()
大数值梯度
      ↓  scaler.step()：内部先 unscale，再检查 inf/nan，再更新
float32 参数更新
      ↓  scaler.update()：调整下一轮的缩放因子
```

> **关键洞察**：`autocast` 和 `GradScaler` 本质上都是上下文管理器。`autocast` 在进入时修改 PyTorch 的计算精度策略，离开时恢复；`GradScaler` 则管理缩放因子的状态。这正是上下文管理器"进入时设置环境、离开时恢复环境"模式的完美体现。

---

## 练习题

### 基础题

**练习 11-1**：实现一个 `Indent` 上下文管理器，进入时将全局缩进级别加 1，离开时减 1。每次调用 `Indent.print(msg)` 时，按当前缩进级别输出带前缀空格的文本。

```python
# 期望用法：
with Indent():
    Indent.print("第一级")
    with Indent():
        Indent.print("第二级")
        with Indent():
            Indent.print("第三级")
    Indent.print("回到第一级")
Indent.print("顶层")

# 期望输出：
#   第一级
#     第二级
#       第三级
#   回到第一级
# 顶层
```

**练习 11-2**：用描述符实现一个 `PositiveNumber` 描述符，确保属性值始终为正数（`> 0`）。并将其用于一个 `Rectangle` 类，要求宽和高只能设置正数，否则抛出 `ValueError`。

```python
# 期望用法：
r = Rectangle(width=5.0, height=3.0)
print(r.area)        # 15.0
r.width = 10.0       # 正常
r.height = -1.0      # 抛出 ValueError
```

### 中级题

**练习 11-3**：实现一个 `retry` 上下文管理器，可以配置最大重试次数和要捕获的异常类型。在 `with` 块中发生指定异常时自动重试，超过最大次数后才真正抛出异常。

```python
# 期望用法：
attempt = [0]
with retry(max_attempts=3, catch=ConnectionError):
    attempt[0] += 1
    if attempt[0] < 3:
        raise ConnectionError("连接失败")
    print(f"第 {attempt[0]} 次尝试成功")
# 输出: 第 3 次尝试成功
```

**练习 11-4**：实现一个 `CachedProperty` 描述符，功能类似 `lazy_property`，但额外支持：1) 通过 `cache_info()` 方法查看缓存命中次数；2) 通过 `cache_clear()` 方法清除特定实例的缓存。

```python
# 期望用法：
class HeavyComputation:
    @CachedProperty
    def result(self):
        print("正在计算...")
        return sum(range(1_000_000))

obj = HeavyComputation()
_ = obj.result    # 输出: 正在计算...
_ = obj.result    # 无输出（使用缓存）
_ = obj.result    # 无输出（使用缓存）
HeavyComputation.result.cache_clear(obj)
_ = obj.result    # 输出: 正在计算...（重新计算）
print(HeavyComputation.result.cache_info())  # hits=2, misses=2
```

### 提高题

**练习 11-5**：设计一个 `ModelCheckpoint` 上下文管理器，用于深度学习训练中的检查点管理。要求：

1. 进入时记录模型当前的参数快照（state_dict）
2. 正常离开时，比较新旧损失值，如果性能提升则保存检查点到指定路径
3. 异常离开时，自动恢复到快照状态（回滚），防止模型被损坏的训练步骤污染
4. 支持 `verbose=True` 参数，开启后打印详细日志

```python
# 期望用法：
model = SimpleNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 场景 1：性能提升，保存检查点
with ModelCheckpoint(model, best_loss=0.5, save_path="best_model.pt", verbose=True) as ckpt:
    # ... 训练步骤 ...
    ckpt.current_loss = 0.3   # 比 best_loss 低，会保存

# 场景 2：训练失败，自动回滚
try:
    with ModelCheckpoint(model, best_loss=0.3, save_path="best_model.pt", verbose=True):
        # ... 训练步骤发生错误 ...
        raise RuntimeError("梯度爆炸")
except RuntimeError:
    pass  # 模型已自动恢复到进入 with 块时的状态
```

---

## 练习答案

### 答案 11-1

```python
class Indent:
    _level = 0       # 类变量，全局共享缩进级别
    _indent = "  "   # 每级缩进两个空格

    def __enter__(self):
        Indent._level += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Indent._level -= 1
        return False

    @classmethod
    def print(cls, msg):
        print(cls._indent * cls._level + msg)


# 测试
with Indent():
    Indent.print("第一级")
    with Indent():
        Indent.print("第二级")
        with Indent():
            Indent.print("第三级")
    Indent.print("回到第一级")
Indent.print("顶层")
# 输出:
#   第一级
#     第二级
#       第三级
#   回到第一级
# 顶层
```

### 答案 11-2

```python
class PositiveNumber:
    """确保属性值为正数的数据描述符。"""

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} 必须是数字")
        if value <= 0:
            raise ValueError(f"{self.name} 必须是正数，得到 {value}")
        setattr(obj, self.private_name, float(value))


class Rectangle:
    width  = PositiveNumber()
    height = PositiveNumber()

    def __init__(self, width, height):
        self.width  = width
        self.height = height

    @property
    def area(self):
        return self.width * self.height

    def __repr__(self):
        return f"Rectangle(width={self.width}, height={self.height})"


# 测试
r = Rectangle(width=5.0, height=3.0)
print(r.area)        # 15.0
r.width = 10.0
print(r)             # Rectangle(width=10.0, height=3.0)

try:
    r.height = -1.0
except ValueError as e:
    print(f"捕获错误: {e}")
# 输出: 捕获错误: height 必须是正数，得到 -1.0

try:
    r.width = 0
except ValueError as e:
    print(f"捕获错误: {e}")
# 输出: 捕获错误: width 必须是正数，得到 0
```

### 答案 11-3

```python
from contextlib import contextmanager

@contextmanager
def retry(max_attempts=3, catch=Exception, delay=0.0):
    """
    重试上下文管理器。
    注意：因为 contextmanager 不支持直接重试 with 块，
    这里使用类实现，提供迭代接口。
    """
    # contextmanager 无法直接重试 yield，改用类实现
    pass

# 正确实现：使用类
class retry:
    """重试上下文管理器。"""

    def __init__(self, max_attempts=3, catch=Exception, delay=0.0):
        self.max_attempts = max_attempts
        self.catch = catch
        self.delay = delay
        self._attempt = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        if exc_type is not None and issubclass(exc_type, self.catch):
            self._attempt += 1
            if self._attempt < self.max_attempts:
                print(f"第 {self._attempt} 次失败: {exc_val}，正在重试...")
                if self.delay > 0:
                    time.sleep(self.delay)
                return True    # 吞掉异常，触发 with 块重新执行？
            # 超过最大次数，让异常传播
        return False

# 注意：上面的实现有个关键限制——with 语句本身不支持重试（__exit__ 返回 True
# 只是抑制了异常，但不会重新执行 with 块）。正确的重试模式应使用循环：

def retry_loop(func, max_attempts=3, catch=Exception, *args, **kwargs):
    """函数式重试辅助。"""
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except catch as e:
            if attempt == max_attempts:
                raise
            print(f"第 {attempt} 次失败: {e}，正在重试...")

# 更实用的模式：使用 for 循环
def demo_retry():
    attempt = [0]

    def unreliable_operation():
        attempt[0] += 1
        if attempt[0] < 3:
            raise ConnectionError("连接失败")
        return f"第 {attempt[0]} 次尝试成功"

    result = retry_loop(unreliable_operation, max_attempts=3, catch=ConnectionError)
    print(result)
    # 输出:
    # 第 1 次失败: 连接失败，正在重试...
    # 第 2 次失败: 连接失败，正在重试...
    # 第 3 次尝试成功

demo_retry()
```

### 答案 11-4

```python
class CachedProperty:
    """带统计信息的缓存属性描述符（非数据描述符）。"""

    def __init__(self, func):
        self.func = func
        self.attr_name = None
        self._hits = 0
        self._misses = 0

    def __set_name__(self, owner, name):
        self.attr_name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        cache_key = self.attr_name
        if cache_key in obj.__dict__:
            self._hits += 1
            return obj.__dict__[cache_key]
        # 缓存未命中：计算并存储
        self._misses += 1
        value = self.func(obj)
        obj.__dict__[cache_key] = value
        return value

    def cache_clear(self, obj):
        """清除特定实例的缓存。"""
        obj.__dict__.pop(self.attr_name, None)

    def cache_info(self):
        """返回缓存统计信息。"""
        return f"hits={self._hits}, misses={self._misses}"


# 测试
class HeavyComputation:
    @CachedProperty
    def result(self):
        print("正在计算...")
        return sum(range(1_000_000))


obj = HeavyComputation()
_ = obj.result    # 输出: 正在计算...
_ = obj.result    # 无输出
_ = obj.result    # 无输出
HeavyComputation.result.cache_clear(obj)
_ = obj.result    # 输出: 正在计算...
print(HeavyComputation.result.cache_info())  # 输出: hits=2, misses=2
```

### 答案 11-5

```python
import torch
import torch.nn as nn
import copy

class ModelCheckpoint:
    """
    深度学习训练检查点上下文管理器。

    功能：
    - 进入时保存模型快照
    - 正常离开且性能提升时保存检查点
    - 异常离开时自动回滚到快照
    """

    def __init__(self, model, best_loss=float("inf"),
                 save_path="checkpoint.pt", verbose=False):
        self.model = model
        self.best_loss = best_loss
        self.save_path = save_path
        self.verbose = verbose
        self.current_loss = None
        self._snapshot = None

    def __enter__(self):
        # 深拷贝模型参数作为快照
        self._snapshot = copy.deepcopy(self.model.state_dict())
        if self.verbose:
            print(f"[Checkpoint] 已保存快照，当前最优损失: {self.best_loss:.4f}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # 有异常：回滚到快照
            self.model.load_state_dict(self._snapshot)
            if self.verbose:
                print(f"[Checkpoint] 检测到异常 {exc_type.__name__}，已回滚模型")
            return False    # 继续传播异常

        # 无异常：检查性能是否提升
        if self.current_loss is not None and self.current_loss < self.best_loss:
            self.best_loss = self.current_loss
            torch.save(self.model.state_dict(), self.save_path)
            if self.verbose:
                print(f"[Checkpoint] 性能提升！新最优损失: {self.best_loss:.4f}，"
                      f"已保存到 {self.save_path!r}")
        else:
            if self.verbose:
                current = self.current_loss if self.current_loss is not None else "未记录"
                print(f"[Checkpoint] 性能未提升（当前: {current}，最优: {self.best_loss:.4f}），"
                      f"不保存")
        return False


# 测试
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


model = SimpleNet()

# 场景 1：性能提升，保存检查点
print("=== 场景 1：性能提升 ===")
with ModelCheckpoint(model, best_loss=0.5, save_path="/tmp/best_model.pt", verbose=True) as ckpt:
    ckpt.current_loss = 0.3

# 场景 2：性能未提升
print("\n=== 场景 2：性能未提升 ===")
with ModelCheckpoint(model, best_loss=0.3, save_path="/tmp/best_model.pt", verbose=True) as ckpt:
    ckpt.current_loss = 0.4

# 场景 3：异常回滚
print("\n=== 场景 3：异常回滚 ===")
original_weight = model.fc.weight.data.clone()
try:
    with ModelCheckpoint(model, best_loss=0.3, save_path="/tmp/best_model.pt", verbose=True) as ckpt:
        # 模拟训练中权重被修改
        with torch.no_grad():
            model.fc.weight.fill_(999.0)
        raise RuntimeError("梯度爆炸！")
except RuntimeError as e:
    print(f"捕获到异常: {e}")

# 验证模型已回滚
weight_restored = torch.allclose(model.fc.weight.data, original_weight)
print(f"权重已恢复到原始值: {weight_restored}")
# 输出:
# === 场景 1：性能提升 ===
# [Checkpoint] 已保存快照，当前最优损失: 0.5000
# [Checkpoint] 性能提升！新最优损失: 0.3000，已保存到 '/tmp/best_model.pt'
#
# === 场景 2：性能未提升 ===
# [Checkpoint] 已保存快照，当前最优损失: 0.3000
# [Checkpoint] 性能未提升（当前: 0.4，最优: 0.3000），不保存
#
# === 场景 3：异常回滚 ===
# [Checkpoint] 已保存快照，当前最优损失: 0.3000
# [Checkpoint] 检测到异常 RuntimeError，已回滚模型
# 捕获到异常: 梯度爆炸！
# 权重已恢复到原始值: True
```

---

[上一章：装饰器与闭包](./10-decorators-closures.md) ｜ [下一章：NumPy基础](../part4-scientific/12-numpy-basics.md)
