# 第10章：装饰器与闭包

> "装饰器是Python最优雅的特性之一，它让我们可以在不修改原有代码的情况下增强函数的功能。"

---

## 学习目标

完成本章学习后，你将能够：

1. 理解闭包的概念，掌握自由变量的捕获机制
2. 编写和使用函数装饰器，理解 `@` 语法糖的本质
3. 构造带参数的装饰器工厂，实现灵活的功能增强
4. 使用类实现装饰器，理解 `__call__` 协议
5. 熟练运用 `functools` 模块中的 `wraps`、`lru_cache` 和 `partial` 工具

---

## 10.1 闭包的概念与应用

### 10.1.1 什么是闭包

闭包（Closure）是指一个函数记住了它被定义时所在作用域中的变量，即使那个作用域已经不再存在。这个"记住"的变量称为**自由变量**（free variable）。

```python
def make_counter():
    count = 0  # 自由变量

    def counter():
        nonlocal count
        count += 1
        return count

    return counter

# 创建闭包
c1 = make_counter()
c2 = make_counter()

print(c1())  # 1
print(c1())  # 2
print(c1())  # 3
print(c2())  # 1  — c2 有独立的 count
print(c2())  # 2
```

每次调用 `make_counter()` 都会产生一个**独立的闭包**，各自拥有自己的 `count` 变量。

### 10.1.2 检查闭包内部

Python 提供了内省工具来查看闭包的细节：

```python
def make_multiplier(factor):
    def multiply(x):
        return x * factor
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)

# 查看自由变量名称
print(double.__code__.co_freevars)  # ('factor',)

# 查看自由变量的值
print(double.__closure__[0].cell_contents)  # 2
print(triple.__closure__[0].cell_contents)  # 3
```

### 10.1.3 nonlocal 关键字

在嵌套函数中修改外层变量必须使用 `nonlocal` 声明（读取不需要）：

```python
def make_accumulator(initial=0):
    total = initial

    def add(value):
        nonlocal total      # 声明要修改外层变量
        total += value
        return total

    def reset():
        nonlocal total
        total = initial
        return total

    return add, reset

add, reset = make_accumulator(100)
print(add(10))   # 110
print(add(20))   # 130
print(reset())   # 100
print(add(5))    # 105
```

### 10.1.4 闭包的实际应用

**场景1：生成配置化的函数**

```python
def make_validator(min_val, max_val):
    """生成一个范围验证器"""
    def validate(value):
        if not (min_val <= value <= max_val):
            raise ValueError(
                f"值 {value} 超出范围 [{min_val}, {max_val}]"
            )
        return value
    return validate

validate_probability = make_validator(0.0, 1.0)
validate_age = make_validator(0, 150)
validate_learning_rate = make_validator(1e-6, 1.0)

print(validate_probability(0.5))    # 0.5
print(validate_learning_rate(0.01)) # 0.01
# validate_probability(1.5)  # 抛出 ValueError
```

**场景2：延迟计算**

```python
def lazy_property(func):
    """只在第一次访问时计算，之后缓存结果"""
    cache = {}

    def wrapper(self):
        if self not in cache:
            cache[self] = func(self)
        return cache[self]

    return wrapper
```

**场景3：深度学习中的学习率调度器**

```python
import math

def make_cosine_scheduler(initial_lr, total_steps):
    """余弦退火学习率调度器"""
    def get_lr(step):
        progress = step / total_steps
        return initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
    return get_lr

scheduler = make_cosine_scheduler(initial_lr=0.1, total_steps=1000)

for step in [0, 250, 500, 750, 1000]:
    print(f"Step {step:4d}: lr = {scheduler(step):.6f}")
# Step    0: lr = 0.100000
# Step  250: lr = 0.085355
# Step  500: lr = 0.050000
# Step  750: lr = 0.014645
# Step 1000: lr = 0.000000
```

---

## 10.2 装饰器基础

### 10.2.1 装饰器的本质

装饰器是一个接受函数作为参数、返回新函数的可调用对象。`@decorator` 语法糖等价于：

```python
def greet(name):
    return f"Hello, {name}!"

# 使用 @ 语法
@decorator
def greet(name):
    return f"Hello, {name}!"

# 等价于
greet = decorator(greet)
```

### 10.2.2 编写第一个装饰器

```python
import time

def timer(func):
    """测量函数执行时间的装饰器"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} 耗时 {elapsed:.4f} 秒")
        return result
    return wrapper

@timer
def slow_sum(n):
    """对 0 到 n-1 求和"""
    return sum(range(n))

result = slow_sum(10_000_000)
print(f"结果: {result}")
# slow_sum 耗时 0.3412 秒
# 结果: 49999995000000
```

### 10.2.3 装饰器堆叠

多个装饰器可以叠加使用，**从下到上**依次包裹：

```python
def bold(func):
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper

def italic(func):
    def wrapper(*args, **kwargs):
        return f"<i>{func(*args, **kwargs)}</i>"
    return wrapper

@bold
@italic
def greet(name):
    return f"Hello, {name}"

# 等价于: greet = bold(italic(greet))
print(greet("Alice"))  # <b><i>Hello, Alice</i></b>
```

### 10.2.4 保留原函数元信息

未经处理的装饰器会丢失原函数的 `__name__`、`__doc__` 等属性：

```python
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"耗时: {time.perf_counter() - start:.4f}s")
        return result
    return wrapper

@timer
def compute():
    """执行复杂计算"""
    pass

print(compute.__name__)  # wrapper  ← 错误！
print(compute.__doc__)   # None     ← 丢失了！
```

解决方法：使用 `functools.wraps`（详见 10.5 节）。

### 10.2.5 实用装饰器示例集

**日志装饰器**

```python
import logging
import functools

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"调用 {func.__name__}，参数: args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logging.info(f"{func.__name__} 返回: {result}")
            return result
        except Exception as e:
            logging.error(f"{func.__name__} 抛出异常: {e}")
            raise
    return wrapper

@log_calls
def divide(a, b):
    return a / b

divide(10, 2)   # INFO: 调用 divide，参数: ...
# divide(10, 0)  # ERROR: divide 抛出异常: division by zero
```

**重试装饰器**

```python
import time
import functools

def retry(max_attempts=3, delay=1.0, exceptions=(Exception,)):
    """自动重试装饰器（这是带参数的装饰器，详见 10.3 节）"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    print(f"第 {attempt} 次失败: {e}，{delay}s 后重试...")
                    time.sleep(delay)
        return wrapper
    return decorator
```

---

## 10.3 带参数的装饰器

### 10.3.1 装饰器工厂模式

带参数的装饰器实际上是一个**返回装饰器的函数**（装饰器工厂）：

```python
# 三层嵌套结构
def decorator_factory(param1, param2):   # 层1：接收参数
    def decorator(func):                  # 层2：接收被装饰的函数
        @functools.wraps(func)
        def wrapper(*args, **kwargs):     # 层3：实际的包装逻辑
            # 使用 param1, param2, func
            return func(*args, **kwargs)
        return wrapper
    return decorator

# 使用方式
@decorator_factory(param1=..., param2=...)
def my_function():
    pass
```

### 10.3.2 完整示例：带配置的计时器

```python
import time
import functools

def timer(unit='s', verbose=True):
    """
    可配置的计时装饰器

    参数:
        unit: 时间单位，'s'（秒）、'ms'（毫秒）或 'us'（微秒）
        verbose: 是否打印输出
    """
    multipliers = {'s': 1, 'ms': 1e3, 'us': 1e6}

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * multipliers[unit]
            if verbose:
                print(f"[timer] {func.__name__}: {elapsed:.3f} {unit}")
            wrapper.last_elapsed = elapsed  # 将耗时挂载到 wrapper 上
            return result
        return wrapper
    return decorator

@timer(unit='ms')
def matrix_multiply(size):
    import random
    A = [[random.random() for _ in range(size)] for _ in range(size)]
    B = [[random.random() for _ in range(size)] for _ in range(size)]
    return [[sum(A[i][k] * B[k][j] for k in range(size))
             for j in range(size)] for i in range(size)]

matrix_multiply(50)
print(f"上次耗时: {matrix_multiply.last_elapsed:.1f} ms")
```

### 10.3.3 类型检查装饰器

```python
import functools
import inspect

def typecheck(func):
    """
    根据类型注解自动检查参数类型
    """
    hints = func.__annotations__
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            if name in hints:
                expected = hints[name]
                if not isinstance(value, expected):
                    raise TypeError(
                        f"参数 '{name}' 期望 {expected.__name__}，"
                        f"实际传入 {type(value).__name__}"
                    )
        return func(*args, **kwargs)
    return wrapper

@typecheck
def train_model(epochs: int, learning_rate: float, model_name: str):
    print(f"训练 {model_name}，{epochs} 轮，lr={learning_rate}")

train_model(10, 0.001, "ResNet")          # 正常
# train_model("10", 0.001, "ResNet")      # TypeError: 参数 'epochs' 期望 int
```

### 10.3.4 权限控制装饰器

```python
import functools

def require_permission(*required_roles):
    """
    权限控制装饰器

    用法:
        @require_permission('admin', 'superuser')
        def delete_model(model_id):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(user, *args, **kwargs):
            user_roles = getattr(user, 'roles', [])
            if not any(role in user_roles for role in required_roles):
                raise PermissionError(
                    f"用户 '{user.name}' 没有执行 '{func.__name__}' 的权限。"
                    f"需要: {required_roles}，拥有: {user_roles}"
                )
            return func(user, *args, **kwargs)
        return wrapper
    return decorator

class User:
    def __init__(self, name, roles):
        self.name = name
        self.roles = roles

@require_permission('admin')
def delete_checkpoint(user, checkpoint_id):
    print(f"{user.name} 删除了检查点 {checkpoint_id}")

admin = User("Alice", ["admin", "user"])
guest = User("Bob", ["user"])

delete_checkpoint(admin, "ckpt_epoch_50")  # 正常
# delete_checkpoint(guest, "ckpt_epoch_50")  # PermissionError
```

---

## 10.4 类装饰器

### 10.4.1 使用类实现装饰器

类可以通过实现 `__call__` 方法来充当装饰器。类装饰器的优势是可以**维护状态**，比嵌套函数更清晰：

```python
import functools
import time

class Timer:
    """计时装饰器（类实现）"""

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.call_count = 0
        self.total_time = 0.0

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        result = self.func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        self.call_count += 1
        self.total_time += elapsed
        print(f"[{self.func.__name__}] 第{self.call_count}次调用，耗时 {elapsed:.4f}s")
        return result

    @property
    def avg_time(self):
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count

@Timer
def compute(n):
    return sum(i ** 2 for i in range(n))

compute(100_000)
compute(200_000)
compute(300_000)
print(f"平均耗时: {compute.avg_time:.4f}s")
print(f"总调用次数: {compute.call_count}")
```

### 10.4.2 带参数的类装饰器

带参数的类装饰器需要调整 `__init__` 接收参数，`__call__` 接收函数：

```python
import functools

class RateLimit:
    """
    限流装饰器：限制函数每秒最多调用 N 次
    """

    def __init__(self, max_calls_per_second):
        self.max_calls = max_calls_per_second
        self.interval = 1.0 / max_calls_per_second
        self.last_called = 0.0

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            elapsed = time.time() - self.last_called
            wait = self.interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self.last_called = time.time()
            return func(*args, **kwargs)
        return wrapper

@RateLimit(max_calls_per_second=2)
def fetch_data(url):
    print(f"获取: {url}")

# 每次调用间隔至少 0.5 秒
fetch_data("https://api.example.com/data/1")
fetch_data("https://api.example.com/data/2")
fetch_data("https://api.example.com/data/3")
```

### 10.4.3 类装饰器用于类

装饰器不仅能装饰函数，也能装饰整个类：

```python
def add_repr(cls):
    """为类自动添加 __repr__ 方法"""
    def __repr__(self):
        attrs = ', '.join(
            f"{k}={v!r}"
            for k, v in self.__dict__.items()
            if not k.startswith('_')
        )
        return f"{cls.__name__}({attrs})"
    cls.__repr__ = __repr__
    return cls

def singleton(cls):
    """单例模式装饰器"""
    instances = {}
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@add_repr
@singleton
class ModelConfig:
    def __init__(self, lr=0.001, batch_size=32, epochs=10):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

cfg1 = ModelConfig(lr=0.01)
cfg2 = ModelConfig(lr=0.001)  # 返回同一个实例
print(cfg1 is cfg2)   # True
print(cfg1)           # ModelConfig(lr=0.01, batch_size=32, epochs=10)
```

### 10.4.4 Mixin 与类装饰器的比较

| 特性 | 类装饰器 | Mixin |
|------|----------|-------|
| 侵入性 | 低（不修改原类继承链） | 高（改变 MRO） |
| 动态性 | 可在运行时应用 | 编译期确定 |
| 复用性 | 高（可用于任意类） | 依赖继承 |
| 适合场景 | 横切关注点（日志、缓存） | 功能组合 |

---

## 10.5 functools 工具

### 10.5.1 functools.wraps

`wraps` 是一个装饰器，用于保留被包装函数的元信息：

```python
import functools

def my_decorator(func):
    @functools.wraps(func)   # 关键：保留 func 的元信息
    def wrapper(*args, **kwargs):
        """这是 wrapper 的文档"""
        print("前置处理")
        result = func(*args, **kwargs)
        print("后置处理")
        return result
    return wrapper

@my_decorator
def greet(name: str) -> str:
    """向用户打招呼"""
    return f"Hello, {name}!"

# 元信息得以保留
print(greet.__name__)       # greet（而非 wrapper）
print(greet.__doc__)        # 向用户打招呼
print(greet.__annotations__)# {'name': <class 'str'>, 'return': <class 'str'>}

# 可以通过 __wrapped__ 访问原函数
print(greet.__wrapped__)    # <function greet at 0x...>
```

### 10.5.2 functools.lru_cache

`lru_cache`（Least Recently Used Cache）提供**内存缓存**，自动缓存函数的返回值：

```python
import functools
import time

@functools.lru_cache(maxsize=128)
def fibonacci(n):
    """带缓存的斐波那契数列"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# 第一次调用：计算
start = time.perf_counter()
print(fibonacci(35))                              # 9227465
print(f"首次: {time.perf_counter() - start:.4f}s")

# 第二次调用：从缓存读取
start = time.perf_counter()
print(fibonacci(35))                              # 9227465
print(f"缓存: {time.perf_counter() - start:.6f}s")

# 查看缓存统计
info = fibonacci.cache_info()
print(f"命中: {info.hits}, 未命中: {info.misses}, 缓存大小: {info.currsize}")

# 清除缓存
fibonacci.cache_clear()
```

**深度学习中的应用：缓存数据集元信息**

```python
import functools
import os

@functools.lru_cache(maxsize=None)
def get_dataset_stats(dataset_path: str):
    """
    计算数据集统计信息（耗时操作，缓存结果）
    maxsize=None 表示无限缓存（相当于 @cache）
    """
    print(f"正在计算 {dataset_path} 的统计信息...")
    # 模拟耗时的文件扫描
    time.sleep(0.5)
    # 实际场景中会计算均值、标准差等
    return {"mean": 0.485, "std": 0.229, "num_samples": 50000}

# 多次调用只计算一次
stats = get_dataset_stats("/data/imagenet/train")
stats = get_dataset_stats("/data/imagenet/train")  # 从缓存读取
print(stats)
```

### 10.5.3 functools.partial

`partial` 用于**固定函数的部分参数**，生成新的可调用对象：

```python
import functools

def power(base, exponent):
    return base ** exponent

# 固定 exponent=2，生成平方函数
square = functools.partial(power, exponent=2)
# 固定 exponent=3，生成立方函数
cube = functools.partial(power, exponent=3)

print(square(5))   # 25
print(cube(3))     # 27
print(list(map(square, range(6))))  # [0, 1, 4, 9, 16, 25]
```

**深度学习中的应用：配置化损失函数**

```python
import functools

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal Loss 实现（用于处理类别不平衡）

    参数:
        alpha: 类别权重
        gamma: 聚焦参数
        reduction: 'mean'、'sum' 或 'none'
    """
    import math
    # 简化实现，仅用于演示
    pt = y_pred if y_true == 1 else 1 - y_pred
    loss = -alpha * (1 - pt) ** gamma * math.log(pt + 1e-8)
    return loss

# 针对不同任务预配置损失函数
standard_focal = functools.partial(focal_loss, alpha=0.25, gamma=2.0)
hard_example_focal = functools.partial(focal_loss, alpha=0.5, gamma=5.0)
balanced_focal = functools.partial(focal_loss, alpha=0.5, gamma=2.0)

# 调用时只需传入 y_true 和 y_pred
loss = standard_focal(y_true=1, y_pred=0.9)
print(f"Focal Loss: {loss:.4f}")
```

**与 map/filter 配合使用**

```python
import functools

def clip(value, min_val, max_val):
    return max(min_val, min(max_val, value))

# 将梯度裁剪到 [-1, 1]
clip_gradient = functools.partial(clip, min_val=-1.0, max_val=1.0)

gradients = [0.5, -2.3, 1.8, -0.1, 3.7, -0.9]
clipped = list(map(clip_gradient, gradients))
print(clipped)  # [0.5, -1.0, 1.0, -0.1, 1.0, -0.9]
```

### 10.5.4 functools.reduce

```python
import functools
import operator

# 用 reduce 计算连乘
numbers = [1, 2, 3, 4, 5]
product = functools.reduce(operator.mul, numbers, 1)
print(product)  # 120

# 用 reduce 合并字典（Python 3.9+ 也可用 | 运算符）
configs = [
    {'lr': 0.001},
    {'batch_size': 32},
    {'epochs': 10, 'lr': 0.01}  # 后面的覆盖前面的
]
merged = functools.reduce(lambda a, b: {**a, **b}, configs)
print(merged)  # {'lr': 0.01, 'batch_size': 32, 'epochs': 10}
```

---

## 本章小结

| 概念 | 关键点 | 典型用途 |
|------|--------|----------|
| **闭包** | 内层函数捕获外层作用域的自由变量 | 状态封装、工厂函数、回调配置 |
| **nonlocal** | 声明修改外层（非全局）变量 | 闭包内计数器、累加器 |
| **基础装饰器** | `wrapper` 包裹原函数，`@` 是语法糖 | 计时、日志、验证 |
| **带参数装饰器** | 三层嵌套：工厂→装饰器→包装器 | 可配置的限流、重试、权限 |
| **类装饰器** | `__call__` 实现调用，天然支持状态 | 统计调用次数、维护缓存 |
| **functools.wraps** | 保留 `__name__`、`__doc__` 等元信息 | 所有装饰器的标配 |
| **lru_cache** | LRU 策略内存缓存，`maxsize` 控制大小 | 纯函数的昂贵计算缓存 |
| **partial** | 固定部分参数，生成特化版本 | 配置化函数、函数式编程 |

---

## 深度学习应用：训练监控装饰器系统

在深度学习训练中，我们常常需要对各种操作进行监控：计时、内存追踪、梯度健康检查、自动保存检查点等。装饰器是实现这些横切关注点的理想工具。

```python
import time
import functools
import logging
from collections import defaultdict
from typing import Optional, Callable

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. 计时装饰器（带统计）
# ─────────────────────────────────────────────

class TrainingTimer:
    """
    训练计时装饰器，记录每次调用的耗时并提供统计报告。
    """
    _registry: dict = {}

    def __init__(self, phase: str = "unknown"):
        self.phase = phase

    def __call__(self, func):
        phase = self.phase
        stats = defaultdict(list)
        TrainingTimer._registry[func.__name__] = stats

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            stats[phase].append(elapsed)
            logger.debug(f"[{phase}] {func.__name__}: {elapsed*1000:.1f}ms")
            return result
        return wrapper

    @classmethod
    def report(cls):
        """打印所有被装饰函数的耗时统计"""
        print("\n" + "="*50)
        print("训练耗时统计报告")
        print("="*50)
        for func_name, phase_stats in cls._registry.items():
            print(f"\n函数: {func_name}")
            for phase, times in phase_stats.items():
                if times:
                    avg = sum(times) / len(times)
                    total = sum(times)
                    print(f"  [{phase}] 调用{len(times)}次 | "
                          f"平均 {avg*1000:.1f}ms | "
                          f"总计 {total:.2f}s")


# ─────────────────────────────────────────────
# 2. 梯度健康检查装饰器
# ─────────────────────────────────────────────

def gradient_check(
    check_nan: bool = True,
    check_inf: bool = True,
    max_norm_threshold: Optional[float] = None
):
    """
    梯度健康检查装饰器。

    检查反向传播后的梯度是否存在 NaN/Inf，
    以及梯度范数是否超过阈值（可能表示梯度爆炸）。

    参数:
        check_nan: 是否检查 NaN 梯度
        check_inf: 是否检查 Inf 梯度
        max_norm_threshold: 梯度范数警告阈值，None 表示不检查
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(model, *args, **kwargs):
            result = func(model, *args, **kwargs)

            # 模拟梯度检查（真实场景中遍历 model.parameters()）
            issues = []
            total_norm = 0.0

            # 伪代码示意，实际使用 PyTorch/TensorFlow API
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad = param.grad.data
            #         if check_nan and torch.isnan(grad).any():
            #             issues.append(f"NaN 梯度: {name}")
            #         if check_inf and torch.isinf(grad).any():
            #             issues.append(f"Inf 梯度: {name}")
            #         total_norm += grad.norm().item() ** 2
            # total_norm = total_norm ** 0.5

            if issues:
                logger.warning(f"梯度异常检测到 {len(issues)} 个问题:")
                for issue in issues:
                    logger.warning(f"  - {issue}")

            if max_norm_threshold and total_norm > max_norm_threshold:
                logger.warning(
                    f"梯度范数 {total_norm:.4f} 超过阈值 {max_norm_threshold}，"
                    f"考虑降低学习率或使用梯度裁剪。"
                )

            return result
        return wrapper
    return decorator


# ─────────────────────────────────────────────
# 3. 自动检查点装饰器
# ─────────────────────────────────────────────

def auto_checkpoint(
    save_every: int = 10,
    save_dir: str = "./checkpoints",
    metric_key: str = "val_loss"
):
    """
    自动保存检查点的装饰器。

    在每隔 save_every 个 epoch 时，以及验证损失改善时保存模型。

    参数:
        save_every: 每隔多少 epoch 保存一次
        save_dir: 检查点保存目录
        metric_key: 用于判断模型是否改善的指标键名
    """
    best_metric = [float('inf')]  # 用列表以便在闭包中修改

    def decorator(func):
        @functools.wraps(func)
        def wrapper(epoch: int, model, metrics: dict, *args, **kwargs):
            result = func(epoch, model, metrics, *args, **kwargs)

            # 定期保存
            if epoch % save_every == 0:
                path = f"{save_dir}/epoch_{epoch:04d}.pt"
                logger.info(f"定期检查点: {path}")
                # torch.save(model.state_dict(), path)

            # 最优保存
            current_metric = metrics.get(metric_key, float('inf'))
            if current_metric < best_metric[0]:
                best_metric[0] = current_metric
                path = f"{save_dir}/best_model.pt"
                logger.info(
                    f"新最优模型 ({metric_key}={current_metric:.4f}): {path}"
                )
                # torch.save(model.state_dict(), path)

            return result
        return wrapper
    return decorator


# ─────────────────────────────────────────────
# 4. 组合使用示例
# ─────────────────────────────────────────────

class MockModel:
    """模拟深度学习模型"""
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"MockModel(name={self.name!r})"


@TrainingTimer(phase="train")
def train_one_epoch(model, dataloader, optimizer):
    """训练一个 epoch"""
    time.sleep(0.01)  # 模拟训练耗时
    return {"train_loss": 0.5, "train_acc": 0.85}


@TrainingTimer(phase="val")
def validate(model, dataloader):
    """验证一个 epoch"""
    time.sleep(0.005)  # 模拟验证耗时
    return {"val_loss": 0.45, "val_acc": 0.88}


@auto_checkpoint(save_every=5, save_dir="./checkpoints")
def training_epoch(epoch, model, metrics):
    """完整的单轮训练（含自动检查点）"""
    logger.info(f"Epoch {epoch}: loss={metrics.get('val_loss', 'N/A'):.4f}")


def run_training_demo():
    """演示完整的训练监控流程"""
    model = MockModel("ResNet50")
    num_epochs = 15

    logger.info(f"开始训练 {model}，共 {num_epochs} 个 epoch")

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(model, dataloader=None, optimizer=None)
        val_metrics = validate(model, dataloader=None)
        all_metrics = {**train_metrics, **val_metrics}
        training_epoch(epoch, model, all_metrics)

    TrainingTimer.report()


if __name__ == "__main__":
    run_training_demo()
```

**运行输出示例：**

```
09:30:01 [INFO] 开始训练 MockModel(name='ResNet50')，共 15 个 epoch
09:30:01 [INFO] Epoch 1: loss=0.4500
09:30:01 [INFO] 新最优模型 (val_loss=0.4500): ./checkpoints/best_model.pt
...
09:30:01 [INFO] Epoch 5: loss=0.4500
09:30:01 [INFO] 定期检查点: ./checkpoints/epoch_0005.pt
...

==================================================
训练耗时统计报告
==================================================

函数: train_one_epoch
  [train] 调用15次 | 平均 10.2ms | 总计 0.15s

函数: validate
  [val] 调用15次 | 平均 5.1ms | 总计 0.08s
```

---

## 练习题

### 基础题

**练习 1：记忆化斐波那契**

不使用 `functools.lru_cache`，手动实现一个带缓存的斐波那契装饰器 `@memoize`，要求：
- 使用字典缓存已计算的结果
- 支持任意参数的函数（不只是单参数）
- 保留原函数的 `__name__` 和 `__doc__`

```python
# 请实现 memoize 装饰器
def memoize(func):
    # TODO: 实现此装饰器
    pass

@memoize
def fib(n):
    """计算第 n 个斐波那契数"""
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

assert fib(10) == 55
assert fib(30) == 832040
print(fib.__name__)   # 应输出 "fib"
```

---

**练习 2：调用频率统计**

编写一个 `@call_counter` 装饰器，记录每个函数被调用的次数，并提供一个类方法 `report()` 打印所有函数的调用次数。

```python
# 请实现 call_counter 装饰器
class call_counter:
    # TODO: 实现此类装饰器
    pass

@call_counter
def train():
    pass

@call_counter
def validate():
    pass

train()
train()
train()
validate()
validate()

call_counter.report()
# 预期输出:
# train: 3 次
# validate: 2 次
```

---

### 中级题

**练习 3：可撤销的装饰器**

实现一个 `@toggleable(name)` 装饰器，支持在运行时启用/禁用特定功能：

```python
# 请实现 toggleable 装饰器工厂
def toggleable(name):
    # TODO: 实现此装饰器工厂
    pass

@toggleable("logging")
def log_step(message):
    print(f"[LOG] {message}")

@toggleable("profiling")
def profile_step(step_name):
    print(f"[PROFILE] {step_name}")

log_step("训练开始")           # 打印：[LOG] 训练开始
profile_step("forward_pass")   # 打印：[PROFILE] forward_pass

toggleable.disable("logging")

log_step("这条不会被打印")     # 静默
profile_step("backward_pass")  # 打印：[PROFILE] backward_pass

toggleable.enable("logging")
log_step("日志恢复了")         # 打印：[LOG] 日志恢复了
```

---

**练习 4：链式装饰器验证器**

实现一个 `@validate` 系统，支持链式添加验证规则：

```python
# 请实现验证系统
class validate:
    # TODO: 实现此类，支持链式规则添加
    pass

@validate.positive("lr", "batch_size")
@validate.range("epochs", 1, 1000)
@validate.type_check
def create_trainer(lr: float, batch_size: int, epochs: int):
    return f"训练器: lr={lr}, batch_size={batch_size}, epochs={epochs}"

print(create_trainer(0.01, 32, 100))   # 正常
# create_trainer(-0.01, 32, 100)       # ValueError: lr 必须为正数
# create_trainer(0.01, 32, 2000)       # ValueError: epochs 超出范围 [1, 1000]
```

---

### 高级题

**练习 5：异步装饰器兼容**

实现一个 `@retry` 装饰器，同时支持**同步函数**和**异步函数（async/await）**，且支持指数退避（exponential backoff）：

```python
import asyncio

# 请实现 retry 装饰器
def retry(max_attempts=3, base_delay=1.0, backoff=2.0, exceptions=(Exception,)):
    """
    重试装饰器，同时支持同步和异步函数。

    参数:
        max_attempts: 最大重试次数
        base_delay: 初始等待时间（秒）
        backoff: 指数退避倍数（每次失败等待时间 *= backoff）
        exceptions: 需要重试的异常类型元组
    """
    # TODO: 实现同步/异步兼容的重试装饰器
    pass

# 同步函数测试
attempt_count = 0

@retry(max_attempts=3, base_delay=0.1, backoff=2.0)
def flaky_sync_function():
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ConnectionError(f"连接失败（第{attempt_count}次）")
    return "同步成功"

# 异步函数测试
async_attempt_count = 0

@retry(max_attempts=3, base_delay=0.1, backoff=2.0)
async def flaky_async_function():
    global async_attempt_count
    async_attempt_count += 1
    if async_attempt_count < 3:
        raise ConnectionError(f"连接失败（第{async_attempt_count}次）")
    return "异步成功"

print(flaky_sync_function())           # 同步成功
print(asyncio.run(flaky_async_function()))  # 异步成功
```

---

## 练习答案

### 答案 1：记忆化斐波那契

```python
import functools

def memoize(func):
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 将 kwargs 转换为可哈希的键
        key = args + tuple(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache = cache  # 暴露缓存以便检查
    wrapper.cache_clear = lambda: cache.clear()
    return wrapper

@memoize
def fib(n):
    """计算第 n 个斐波那契数"""
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

assert fib(10) == 55
assert fib(30) == 832040
assert fib.__name__ == "fib"
assert fib.__doc__ == "计算第 n 个斐波那契数"
print("练习1 通过！")
```

---

### 答案 2：调用频率统计

```python
import functools

class call_counter:
    _counts = {}  # 类变量：所有实例共享

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        call_counter._counts[func.__name__] = 0

    def __call__(self, *args, **kwargs):
        call_counter._counts[self.func.__name__] += 1
        return self.func(*args, **kwargs)

    @classmethod
    def report(cls):
        print("\n调用统计:")
        for name, count in cls._counts.items():
            print(f"  {name}: {count} 次")

    @classmethod
    def reset(cls):
        for key in cls._counts:
            cls._counts[key] = 0

@call_counter
def train():
    pass

@call_counter
def validate():
    pass

train(); train(); train()
validate(); validate()

call_counter.report()
# 调用统计:
#   train: 3 次
#   validate: 2 次
```

---

### 答案 3：可撤销的装饰器

```python
import functools

class toggleable:
    _enabled: dict = {}  # 全局开关状态

    def __new__(cls, name):
        # toggleable("logging") 调用时返回装饰器
        instance = object.__new__(cls)
        instance.name = name
        if name not in cls._enabled:
            cls._enabled[name] = True
        return instance

    def __call__(self, func):
        feature_name = self.name

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if toggleable._enabled.get(feature_name, True):
                return func(*args, **kwargs)
            # 被禁用时静默返回 None

        return wrapper

    @classmethod
    def enable(cls, name):
        cls._enabled[name] = True

    @classmethod
    def disable(cls, name):
        cls._enabled[name] = False

    @classmethod
    def status(cls):
        for name, enabled in cls._enabled.items():
            state = "ON " if enabled else "OFF"
            print(f"  [{state}] {name}")

# 测试
@toggleable("logging")
def log_step(message):
    print(f"[LOG] {message}")

@toggleable("profiling")
def profile_step(step_name):
    print(f"[PROFILE] {step_name}")

log_step("训练开始")
profile_step("forward_pass")

toggleable.disable("logging")
log_step("这条不会被打印")
profile_step("backward_pass")

toggleable.enable("logging")
log_step("日志恢复了")
```

---

### 答案 4：链式装饰器验证器

```python
import functools
import inspect

class validate:
    """验证装饰器命名空间"""

    @staticmethod
    def type_check(func):
        hints = func.__annotations__
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for name, value in bound.arguments.items():
                if name in hints and name != 'return':
                    if not isinstance(value, hints[name]):
                        raise TypeError(
                            f"参数 '{name}' 期望 {hints[name].__name__}，"
                            f"实际为 {type(value).__name__}"
                        )
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    def positive(*param_names):
        def decorator(func):
            sig = inspect.signature(func)
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                for name in param_names:
                    if name in bound.arguments:
                        if bound.arguments[name] <= 0:
                            raise ValueError(f"参数 '{name}' 必须为正数，实际为 {bound.arguments[name]}")
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def range(param_name, min_val, max_val):
        def decorator(func):
            sig = inspect.signature(func)
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                if param_name in bound.arguments:
                    v = bound.arguments[param_name]
                    if not (min_val <= v <= max_val):
                        raise ValueError(
                            f"参数 '{param_name}' 超出范围 [{min_val}, {max_val}]，实际为 {v}"
                        )
                return func(*args, **kwargs)
            return wrapper
        return decorator

# 测试
@validate.positive("lr", "batch_size")
@validate.range("epochs", 1, 1000)
@validate.type_check
def create_trainer(lr: float, batch_size: int, epochs: int):
    return f"训练器: lr={lr}, batch_size={batch_size}, epochs={epochs}"

print(create_trainer(0.01, 32, 100))

try:
    create_trainer(-0.01, 32, 100)
except ValueError as e:
    print(f"捕获异常: {e}")

try:
    create_trainer(0.01, 32, 2000)
except ValueError as e:
    print(f"捕获异常: {e}")
```

---

### 答案 5：异步装饰器兼容

```python
import asyncio
import time
import functools
import inspect

def retry(max_attempts=3, base_delay=1.0, backoff=2.0, exceptions=(Exception,)):
    def decorator(func):
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                delay = base_delay
                last_exception = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts:
                            print(f"[retry] 第{attempt}次失败: {e}，{delay:.1f}s 后重试...")
                            await asyncio.sleep(delay)
                            delay *= backoff
                        else:
                            print(f"[retry] 第{attempt}次失败: {e}，已达最大重试次数。")
                raise last_exception
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                delay = base_delay
                last_exception = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts:
                            print(f"[retry] 第{attempt}次失败: {e}，{delay:.1f}s 后重试...")
                            time.sleep(delay)
                            delay *= backoff
                        else:
                            print(f"[retry] 第{attempt}次失败: {e}，已达最大重试次数。")
                raise last_exception
            return sync_wrapper

    return decorator

# 同步测试
attempt_count = 0

@retry(max_attempts=3, base_delay=0.1, backoff=2.0)
def flaky_sync():
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ConnectionError(f"连接失败（第{attempt_count}次）")
    return "同步成功"

# 异步测试
async_count = 0

@retry(max_attempts=3, base_delay=0.1, backoff=2.0)
async def flaky_async():
    global async_count
    async_count += 1
    if async_count < 3:
        raise ConnectionError(f"连接失败（第{async_count}次）")
    return "异步成功"

print(flaky_sync())
print(asyncio.run(flaky_async()))
```

---

*下一章：[第11章：迭代器与生成器](./11-iterators-generators.md)*
