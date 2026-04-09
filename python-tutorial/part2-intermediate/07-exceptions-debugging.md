# 第7章：异常处理与调试

## 学习目标

完成本章学习后，你将能够：

1. 理解Python异常体系结构，识别并区分常见异常类型及其触发场景
2. 熟练使用 `try/except/else/finally` 语句捕获和处理异常，编写健壮的错误处理逻辑
3. 设计并实现自定义异常类，用于表达特定业务或算法错误
4. 使用 `logging` 模块替代 `print` 语句，建立分级日志系统
5. 掌握 `print` 调试、`assert` 断言以及 `pdb` 调试器等多种调试技术，快速定位和修复bug

---

## 7.1 异常的概念与常见异常类型

### 7.1.1 什么是异常

程序在运行过程中遇到无法正常执行的情况时，Python会**抛出（raise）异常**。异常是一种对象，携带了错误类型和错误信息。如果异常没有被捕获处理，程序就会终止并打印**回溯信息（traceback）**。

```python
# 未处理异常的示例
numbers = [1, 2, 3]
print(numbers[10])  # IndexError: list index out of range
```

输出的 traceback 信息：

```
Traceback (most recent call last):
  File "example.py", line 2, in <module>
    print(numbers[10])
IndexError: list index out of range
```

traceback 从下往上阅读：最底部是**错误类型和描述**，往上是调用链。

### 7.1.2 Python异常层次结构

Python所有异常都继承自 `BaseException`，用户自定义异常通常继承自 `Exception`：

```
BaseException
├── SystemExit          # sys.exit() 触发
├── KeyboardInterrupt   # Ctrl+C 触发
└── Exception           # 所有普通异常的基类
    ├── ArithmeticError
    │   ├── ZeroDivisionError
    │   └── OverflowError
    ├── LookupError
    │   ├── IndexError
    │   └── KeyError
    ├── TypeError
    ├── ValueError
    ├── AttributeError
    ├── NameError
    ├── FileNotFoundError（OSError的子类）
    ├── RuntimeError
    │   └── RecursionError
    └── StopIteration
```

### 7.1.3 常见异常类型详解

```python
# 1. TypeError - 操作或函数应用于不适当类型的对象
result = "100" + 200        # TypeError: can only concatenate str (not "int") to str
result = len(42)            # TypeError: object of type 'int' has no len()

# 2. ValueError - 类型正确但值不合法
int("abc")                  # ValueError: invalid literal for int() with base 10: 'abc'
import math
math.sqrt(-1)               # ValueError: math domain error

# 3. IndexError - 序列下标超出范围
lst = [1, 2, 3]
lst[5]                      # IndexError: list index out of range

# 4. KeyError - 字典中不存在的键
d = {"a": 1}
d["b"]                      # KeyError: 'b'

# 5. AttributeError - 对象没有该属性或方法
x = 42
x.upper()                   # AttributeError: 'int' object has no attribute 'upper'

# 6. NameError - 变量名未定义
print(undefined_variable)   # NameError: name 'undefined_variable' is not defined

# 7. ZeroDivisionError - 除以零
10 / 0                      # ZeroDivisionError: division by zero

# 8. FileNotFoundError - 文件不存在
open("nonexistent.txt")     # FileNotFoundError: [Errno 2] No such file or directory

# 9. ImportError / ModuleNotFoundError
import nonexistent_module   # ModuleNotFoundError: No module named 'nonexistent_module'

# 10. RecursionError - 超过最大递归深度
def infinite():
    return infinite()
infinite()                  # RecursionError: maximum recursion depth exceeded
```

### 7.1.4 如何读懂Traceback

```python
def divide(a, b):
    return a / b

def calculate(x):
    return divide(x, 0)

calculate(10)
```

```
Traceback (most recent call last):
  File "demo.py", line 7, in <module>    # <-- 调用链最外层
    calculate(10)
  File "demo.py", line 5, in calculate   # <-- 中间调用
    return divide(x, 0)
  File "demo.py", line 2, in divide      # <-- 最终出错位置
    return a / b
ZeroDivisionError: division by zero      # <-- 错误类型和原因
```

**阅读技巧**：从最底部的错误类型开始，沿调用栈向上追溯，找到自己代码中最靠近底部的那一行，那通常是问题根源。

---

## 7.2 try/except/else/finally语句

### 7.2.1 基本语法

```python
try:
    # 可能引发异常的代码
    result = 10 / 0
except ZeroDivisionError:
    # 捕获到指定异常时执行
    print("不能除以零！")
```

### 7.2.2 捕获多种异常

```python
def safe_parse(text, index):
    try:
        number = int(text)       # 可能 ValueError
        result = [1, 2, 3][index]  # 可能 IndexError
        return number + result
    except ValueError:
        print(f"'{text}' 不是有效的整数")
        return None
    except IndexError:
        print(f"索引 {index} 超出范围")
        return None

safe_parse("abc", 1)    # 输出：'abc' 不是有效的整数
safe_parse("5", 10)     # 输出：索引 10 超出范围
safe_parse("5", 1)      # 返回：7
```

**将多种异常合并捕获（使用元组）：**

```python
try:
    value = int(input("请输入数字："))
    result = 100 / value
except (ValueError, ZeroDivisionError) as e:
    print(f"输入错误：{e}")
```

### 7.2.3 获取异常对象

```python
try:
    x = int("not_a_number")
except ValueError as e:
    print(f"异常类型：{type(e).__name__}")  # ValueError
    print(f"异常信息：{e}")                 # invalid literal for int()...
    print(f"异常参数：{e.args}")            # ('invalid literal...',)
```

### 7.2.4 捕获所有异常

```python
try:
    risky_operation()
except Exception as e:
    # 捕获所有 Exception 子类（不包括 SystemExit、KeyboardInterrupt）
    print(f"发生错误：{type(e).__name__}: {e}")
```

> **警告**：避免使用裸 `except:` 或 `except BaseException:`，这会连 `KeyboardInterrupt`（Ctrl+C）也捕获，导致程序无法正常终止。

### 7.2.5 else子句

`else` 块在 **try块没有发生任何异常时** 执行，用于放置只有在成功情况下才运行的代码：

```python
def read_number_from_file(filename):
    try:
        f = open(filename)
    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
        return None
    else:
        # 只有文件成功打开才执行这里
        content = f.read()
        f.close()
        return int(content.strip())
```

### 7.2.6 finally子句

`finally` 块**无论是否发生异常都会执行**，通常用于释放资源：

```python
def process_file(filename):
    f = None
    try:
        f = open(filename, 'r')
        data = f.read()
        return data.upper()
    except FileNotFoundError:
        print(f"文件不存在：{filename}")
        return None
    except PermissionError:
        print(f"没有读取权限：{filename}")
        return None
    finally:
        # 即使发生异常，也确保文件被关闭
        if f is not None:
            f.close()
            print("文件已关闭")
```

### 7.2.7 完整语法示例

```python
import json

def load_config(filepath):
    """加载JSON配置文件，演示完整的异常处理结构"""
    file_handle = None
    try:
        file_handle = open(filepath, 'r', encoding='utf-8')
        config = json.load(file_handle)
        # 验证必要字段
        if 'learning_rate' not in config:
            raise ValueError("配置文件缺少 'learning_rate' 字段")
        return config

    except FileNotFoundError:
        print(f"[ERROR] 配置文件不存在：{filepath}")
        return {}
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON解析失败：{e}")
        return {}
    except ValueError as e:
        print(f"[ERROR] 配置验证失败：{e}")
        return {}
    except Exception as e:
        print(f"[ERROR] 未知错误：{type(e).__name__}: {e}")
        return {}
    else:
        print(f"[INFO] 配置加载成功，共 {len(config)} 个字段")
    finally:
        if file_handle is not None:
            file_handle.close()

# 测试
config = load_config("model_config.json")
```

### 7.2.8 主动抛出异常

使用 `raise` 语句主动抛出异常：

```python
def set_learning_rate(lr):
    if not isinstance(lr, (int, float)):
        raise TypeError(f"学习率必须是数值，得到了 {type(lr).__name__}")
    if lr <= 0 or lr >= 1:
        raise ValueError(f"学习率必须在 (0, 1) 范围内，得到了 {lr}")
    return lr

# 在 except 块中重新抛出
try:
    set_learning_rate(-0.1)
except ValueError as e:
    print(f"捕获到错误：{e}")
    raise  # 重新抛出，不改变原始异常信息
```

---

## 7.3 自定义异常

### 7.3.1 为什么需要自定义异常

内置异常类型是通用的，当需要表达特定业务逻辑或算法错误时，自定义异常能让代码更清晰、更易于处理：

- `ValueError` 太宽泛，`InvalidLearningRateError` 一看就懂
- 调用者可以针对特定错误做精确处理
- 可以在异常中携带额外的结构化信息

### 7.3.2 创建自定义异常

```python
# 最简单的自定义异常
class ModelError(Exception):
    """模型相关错误的基类"""
    pass

class InvalidLearningRateError(ModelError):
    """学习率无效时抛出"""
    pass

class GradientExplosionError(ModelError):
    """梯度爆炸时抛出"""
    pass
```

### 7.3.3 携带额外信息的异常

```python
class NaNLossError(ModelError):
    """损失值出现NaN时抛出"""

    def __init__(self, epoch, batch_idx, loss_value):
        self.epoch = epoch
        self.batch_idx = batch_idx
        self.loss_value = loss_value
        # 调用父类构造，设置str()的内容
        message = (
            f"在第 {epoch} 轮、第 {batch_idx} 批次检测到NaN损失。"
            f"最后有效损失值：{loss_value}"
        )
        super().__init__(message)

    def __str__(self):
        return (
            f"NaNLossError(epoch={self.epoch}, "
            f"batch={self.batch_idx}, "
            f"last_valid_loss={self.loss_value})"
        )


class GradientExplosionError(ModelError):
    """梯度爆炸时抛出"""

    def __init__(self, layer_name, grad_norm, threshold):
        self.layer_name = layer_name
        self.grad_norm = grad_norm
        self.threshold = threshold
        message = (
            f"层 '{layer_name}' 梯度范数 {grad_norm:.4f} "
            f"超过阈值 {threshold}"
        )
        super().__init__(message)
```

### 7.3.4 使用自定义异常

```python
import math

def check_loss(epoch, batch_idx, loss, prev_valid_loss=None):
    """检查损失值是否合法"""
    if math.isnan(loss):
        raise NaNLossError(epoch, batch_idx, prev_valid_loss)
    if math.isinf(loss):
        raise ValueError(f"损失值为无穷大：{loss}")
    return loss


def check_gradients(gradients, threshold=10.0):
    """检查梯度是否爆炸"""
    for layer_name, grad in gradients.items():
        grad_norm = sum(g**2 for g in grad) ** 0.5
        if grad_norm > threshold:
            raise GradientExplosionError(layer_name, grad_norm, threshold)


# 使用示例
try:
    loss = float('nan')
    check_loss(epoch=5, batch_idx=32, loss=loss, prev_valid_loss=0.423)
except NaNLossError as e:
    print(f"捕获到NaN损失：{e}")
    print(f"  发生在 epoch={e.epoch}, batch={e.batch_idx}")
    # 可以根据结构化信息做出决策，例如回滚到上一个检查点
except ModelError as e:
    # 捕获所有模型相关错误
    print(f"模型错误：{e}")
```

### 7.3.5 异常链

Python 3支持异常链，可以在捕获异常后抛出新异常，同时保留原始异常信息：

```python
def load_weights(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        return parse_weights(data)
    except FileNotFoundError as e:
        # raise X from Y: Y是原因，X是新异常
        raise ModelError(f"无法加载模型权重：{filepath}") from e
    except struct.error as e:
        raise ModelError(f"权重文件格式损坏：{filepath}") from e


# 捕获时可以访问原始异常
try:
    load_weights("model.bin")
except ModelError as e:
    print(f"错误：{e}")
    if e.__cause__:
        print(f"原因：{e.__cause__}")
```

---

## 7.4 日志系统（logging模块）

### 7.4.1 为什么要用logging而不是print

| 特性 | `print` | `logging` |
|------|---------|-----------|
| 日志级别 | 无 | DEBUG/INFO/WARNING/ERROR/CRITICAL |
| 输出控制 | 全有或全无 | 按级别过滤 |
| 输出目标 | 仅终端 | 终端、文件、网络等 |
| 时间戳/模块名 | 需手动添加 | 自动记录 |
| 生产环境关闭 | 需逐个删除 | 修改配置即可 |
| 线程安全 | 否 | 是 |

### 7.4.2 日志级别

```python
import logging

# 五个标准级别（数值越小越详细）
logging.debug("调试信息：变量值、函数入参等")    # DEBUG = 10
logging.info("正常流程信息：程序启动、任务完成")  # INFO = 20
logging.warning("警告：不影响运行但需注意")       # WARNING = 30（默认最低显示级别）
logging.error("错误：某功能无法完成")             # ERROR = 40
logging.critical("严重错误：程序可能无法继续")    # CRITICAL = 50
```

### 7.4.3 基本配置

```python
import logging

# basicConfig 只在第一次调用时生效
logging.basicConfig(
    level=logging.DEBUG,           # 最低显示级别
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),   # 输出到终端
        logging.FileHandler('training.log', encoding='utf-8'),  # 输出到文件
    ]
)

logger = logging.getLogger(__name__)  # 推荐：使用模块名作为logger名称

logger.info("程序启动")
logger.debug("调试：学习率 = 0.001")
logger.warning("警告：GPU内存使用率超过 90%")
logger.error("错误：数据加载失败")
```

输出示例：
```
2024-03-15 14:23:01 [INFO] __main__: 程序启动
2024-03-15 14:23:01 [DEBUG] __main__: 调试：学习率 = 0.001
2024-03-15 14:23:01 [WARNING] __main__: 警告：GPU内存使用率超过 90%
2024-03-15 14:23:01 [ERROR] __main__: 错误：数据加载失败
```

### 7.4.4 记录异常信息

```python
import logging

logger = logging.getLogger(__name__)

def process_batch(batch_data):
    try:
        result = risky_computation(batch_data)
        return result
    except ValueError as e:
        # exc_info=True 会自动附加完整的 traceback
        logger.error("批次处理失败", exc_info=True)
        return None
    except Exception as e:
        # 等价写法：logger.exception() 自动包含 exc_info
        logger.exception(f"未预期的错误：{e}")
        raise
```

### 7.4.5 深度学习训练日志配置

```python
import logging
import sys
from pathlib import Path

def setup_training_logger(log_dir="logs", experiment_name="experiment"):
    """为深度学习训练设置结构化日志"""

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 获取根logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)

    # 防止重复添加handler（重新运行脚本时）
    if logger.handlers:
        logger.handlers.clear()

    # 格式：详细格式用于文件，简洁格式用于终端
    detailed_fmt = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_fmt = logging.Formatter(
        fmt='[%(levelname)s] %(message)s'
    )

    # 文件handler：记录所有DEBUG以上的日志
    file_handler = logging.FileHandler(
        log_path / f"{experiment_name}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_fmt)

    # 终端handler：只显示INFO以上
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 使用示例
logger = setup_training_logger(experiment_name="resnet50_cifar10")

logger.info("开始训练 ResNet-50 on CIFAR-10")
logger.debug("超参数：lr=0.001, batch_size=128, epochs=100")

for epoch in range(1, 101):
    # 模拟训练循环
    train_loss = 2.0 / epoch  # 假设损失下降
    logger.info(f"Epoch {epoch:3d}/100 | Train Loss: {train_loss:.4f}")

    if train_loss < 0.01:
        logger.info("损失已收敛，提前停止训练")
        break
```

### 7.4.6 使用日志上下文记录结构化信息

```python
import logging

logger = logging.getLogger(__name__)

class TrainingLogger:
    """封装训练过程中的日志记录"""

    def __init__(self, total_epochs, total_batches):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.best_loss = float('inf')

    def log_epoch_start(self, epoch):
        logger.info("=" * 50)
        logger.info(f"Epoch {epoch}/{self.total_epochs} 开始")

    def log_batch(self, epoch, batch, loss, lr):
        if batch % 100 == 0:  # 每100个batch记录一次
            logger.debug(
                f"Epoch {epoch}, Batch {batch}/{self.total_batches} | "
                f"Loss: {loss:.6f} | LR: {lr:.2e}"
            )

    def log_epoch_end(self, epoch, train_loss, val_loss):
        improved = val_loss < self.best_loss
        if improved:
            self.best_loss = val_loss
            logger.info(
                f"Epoch {epoch} 完成 | Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | 最佳模型已保存"
            )
        else:
            logger.info(
                f"Epoch {epoch} 完成 | Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | (最佳: {self.best_loss:.4f})"
            )

    def log_warning(self, message):
        logger.warning(f"[训练警告] {message}")

    def log_error(self, message, exc_info=False):
        logger.error(f"[训练错误] {message}", exc_info=exc_info)
```

---

## 7.5 调试技巧

### 7.5.1 print调试

最简单直接的调试方式，适合快速验证假设：

```python
def compute_loss(predictions, targets):
    print(f"[DEBUG] predictions shape: {predictions.shape}")   # 检查形状
    print(f"[DEBUG] targets shape: {targets.shape}")
    print(f"[DEBUG] predictions sample: {predictions[:3]}")    # 检查数值
    print(f"[DEBUG] targets sample: {targets[:3]}")

    diff = predictions - targets
    print(f"[DEBUG] diff stats: min={diff.min():.4f}, max={diff.max():.4f}")

    loss = (diff ** 2).mean()
    print(f"[DEBUG] loss = {loss:.6f}")
    return loss
```

**增强版：带标签的调试函数**

```python
import inspect

def debug_print(label, value, show_type=True, show_shape=False):
    """增强的调试打印函数"""
    # 获取调用者的行号
    frame = inspect.currentframe().f_back
    lineno = frame.f_lineno

    type_info = f" [{type(value).__name__}]" if show_type else ""

    # 尝试获取shape（适用于numpy/torch张量）
    shape_info = ""
    if show_shape and hasattr(value, 'shape'):
        shape_info = f" shape={value.shape}"

    print(f"[L{lineno}] {label}{type_info}{shape_info}: {value}")


# 使用示例
x = [1, 2, 3, 4, 5]
debug_print("输入数组", x)
debug_print("数组长度", len(x), show_type=False)
```

**调试开关：避免生产环境中的调试输出**

```python
import os

DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

def dprint(*args, **kwargs):
    """只在DEBUG模式下打印"""
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

# 启用：在命令行设置 DEBUG=true python train.py
# 或在代码中：import os; os.environ['DEBUG'] = 'true'
dprint("这条信息只在DEBUG模式下显示")
```

### 7.5.2 断言（assert）

断言用于验证程序中的**不变量（invariant）**，即"此处的条件必须为真，否则程序逻辑有误"：

```python
def normalize(data):
    """将数据归一化到 [0, 1] 范围"""
    assert len(data) > 0, "输入数据不能为空"

    min_val = min(data)
    max_val = max(data)

    assert min_val != max_val, f"数据范围为零，无法归一化（所有值均为 {min_val}）"

    result = [(x - min_val) / (max_val - min_val) for x in data]

    # 后置条件验证
    assert abs(min(result) - 0.0) < 1e-9, "归一化后最小值应为0"
    assert abs(max(result) - 1.0) < 1e-9, "归一化后最大值应为1"

    return result


# assert 在生产环境中可以通过 python -O 关闭
# 因此不要用 assert 做输入验证（应用 raise ValueError），
# 只用它验证内部逻辑的正确性
```

**断言 vs 异常的选择原则：**

```python
# 用 assert：内部不变量，"这不可能发生，如果发生说明代码有bug"
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        assert 0 <= mid < len(arr), f"mid={mid} 超出范围，这是bug"
        # ...

# 用 raise：用户输入或外部条件，"这可能发生，需要处理"
def set_epochs(n):
    if not isinstance(n, int):
        raise TypeError(f"epochs必须是整数，得到 {type(n).__name__}")
    if n <= 0:
        raise ValueError(f"epochs必须为正整数，得到 {n}")
    return n
```

### 7.5.3 pdb调试器

`pdb`（Python Debugger）是Python内置的交互式调试器，可以在运行时检查程序状态：

**常用pdb命令：**

| 命令 | 缩写 | 说明 |
|------|------|------|
| `next` | `n` | 执行下一行（不进入函数） |
| `step` | `s` | 执行下一行（进入函数） |
| `continue` | `c` | 继续执行到下一个断点 |
| `break` | `b` | 设置断点：`b 42` 或 `b func_name` |
| `print` | `p` | 打印变量：`p variable` |
| `list` | `l` | 显示当前代码上下文 |
| `where` | `w` | 显示调用栈 |
| `up/down` | `u/d` | 在调用栈中移动 |
| `quit` | `q` | 退出调试器 |
| `help` | `h` | 显示帮助 |

**方法一：在代码中设置断点**

```python
import pdb

def train_step(model, batch, optimizer):
    inputs, labels = batch

    # 在这里暂停，进入交互式调试
    pdb.set_trace()

    outputs = model(inputs)
    loss = compute_loss(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
```

**方法二：Python 3.7+ 内置断点函数**

```python
def suspicious_function(data):
    processed = preprocess(data)
    breakpoint()  # 等同于 pdb.set_trace()，但更简洁
    result = compute(processed)
    return result

# 可通过环境变量禁用断点：PYTHONBREAKPOINT=0 python script.py
```

**方法三：从命令行启动pdb**

```bash
# 对整个脚本启动pdb
python -m pdb train.py

# 发生异常时自动进入pdb（事后调试）
python -m pdb -c continue train.py
```

**实际调试示例：**

```python
import pdb

def calculate_accuracy(predictions, labels):
    """计算分类准确率"""
    if len(predictions) != len(labels):
        raise ValueError("预测和标签数量不匹配")

    # 假设此处出现了意外结果，设置断点排查
    # breakpoint()

    correct = sum(p == l for p, l in zip(predictions, labels))
    total = len(labels)
    return correct / total

# 模拟bug：标签是字符串，预测是整数
predictions = [0, 1, 2, 1, 0]
labels = ["0", "1", "2", "1", "0"]  # 错误！应该是整数

# 运行时会发现准确率为0（因为 0 != "0"）
# 设置 breakpoint() 后可以检查 p 和 l 的类型
acc = calculate_accuracy(predictions, labels)
print(f"准确率：{acc}")  # 输出 0.0，而不是预期的 1.0
```

### 7.5.4 综合调试策略

```python
import logging
import traceback

logger = logging.getLogger(__name__)

def robust_forward_pass(model, batch):
    """展示综合调试策略的前向传播"""
    inputs, targets = batch

    # 1. 前置条件检查（assert）
    assert inputs.ndim == 4, f"期望4D输入（N,C,H,W），得到 {inputs.ndim}D"
    assert inputs.shape[0] == targets.shape[0], "批次大小不匹配"

    # 2. 调试日志
    logger.debug(f"输入形状: {inputs.shape}, 目标形状: {targets.shape}")
    logger.debug(f"输入范围: [{inputs.min():.3f}, {inputs.max():.3f}]")

    try:
        # 3. 核心计算
        outputs = model(inputs)

        # 4. 输出验证
        import math
        if math.isnan(outputs.mean()):
            logger.error("前向传播输出包含NaN！")
            logger.error(f"输入统计：mean={inputs.mean():.4f}, std={inputs.std():.4f}")
            # 5. 必要时设置断点
            # breakpoint()
            raise NaNLossError(epoch=0, batch_idx=0, loss_value=None)

        return outputs

    except RuntimeError as e:
        # 6. 捕获并记录带完整traceback的异常
        logger.error(f"前向传播失败：{e}")
        logger.debug("完整错误信息：\n" + traceback.format_exc())
        raise
```

---

## 本章小结

| 概念 | 核心要点 |
|------|---------|
| **异常层次** | `BaseException → Exception → 具体异常`，用户代码继承 `Exception` |
| **try/except** | `except ExceptionType as e:` 精确捕获，避免裸 `except:` |
| **else子句** | try无异常时执行，分离"成功路径"代码 |
| **finally子句** | 无论如何都执行，用于资源释放 |
| **raise** | 主动抛出异常；`raise X from Y` 保留异常链 |
| **自定义异常** | 继承 `Exception`，添加结构化属性，建立异常层次 |
| **logging** | 替代print；五个级别；同时输出到文件和终端 |
| **assert** | 验证内部不变量，不用于用户输入验证 |
| **breakpoint()** | 交互式调试，`PYTHONBREAKPOINT=0` 可全局禁用 |
| **调试策略** | 先加日志→复现问题→缩小范围→断点验证→修复→回归测试 |

---

## 深度学习应用：训练过程中的异常处理

在深度学习训练中，常见的运行时问题包括：NaN损失、梯度爆炸、显存不足、数据加载错误等。本节展示如何用本章所学构建一个健壮的训练循环。

```python
import math
import logging
import traceback
from pathlib import Path

# ============================================================
# 自定义异常体系
# ============================================================

class TrainingError(Exception):
    """训练相关错误的基类"""
    pass


class NaNLossError(TrainingError):
    """损失值出现NaN"""
    def __init__(self, epoch, step, last_valid_loss=None):
        self.epoch = epoch
        self.step = step
        self.last_valid_loss = last_valid_loss
        super().__init__(
            f"Epoch {epoch}, Step {step}: 检测到NaN损失"
            + (f"（上次有效损失：{last_valid_loss:.6f}）" if last_valid_loss else "")
        )


class GradientExplosionError(TrainingError):
    """梯度爆炸"""
    def __init__(self, grad_norm, threshold, epoch, step):
        self.grad_norm = grad_norm
        self.threshold = threshold
        self.epoch = epoch
        self.step = step
        super().__init__(
            f"Epoch {epoch}, Step {step}: 梯度范数 {grad_norm:.2f} 超过阈值 {threshold}"
        )


class CheckpointError(TrainingError):
    """检查点保存/加载失败"""
    pass


class EarlyStoppingSignal(Exception):
    """早停信号（非错误，用于控制流）"""
    def __init__(self, best_epoch, best_loss):
        self.best_epoch = best_epoch
        self.best_loss = best_loss
        super().__init__(f"早停于第 {best_epoch} 轮，最佳损失：{best_loss:.6f}")


# ============================================================
# 日志配置
# ============================================================

def setup_logger(name, log_dir="logs"):
    Path(log_dir).mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger  # 避免重复添加

    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%H:%M:%S'
    )
    fh = logging.FileHandler(f"{log_dir}/{name}.log", encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================
# 梯度和损失监控工具
# ============================================================

def check_loss_value(loss_val, epoch, step, last_valid_loss=None):
    """验证损失值合法性"""
    if math.isnan(loss_val):
        raise NaNLossError(epoch, step, last_valid_loss)
    if math.isinf(loss_val):
        raise TrainingError(f"Epoch {epoch}, Step {step}: 损失值为无穷大 ({loss_val})")
    if loss_val > 1e6:
        # 不抛出异常，但记录警告
        return False  # 异常大，需要关注
    return True


def check_gradients(named_params, threshold=100.0, epoch=0, step=0):
    """
    检查所有参数的梯度范数。
    named_params: model.named_parameters() 返回的迭代器
    """
    total_norm = 0.0
    for name, param in named_params:
        if param.grad is not None:
            param_norm = sum(g**2 for g in param.grad.flatten())
            total_norm += param_norm

    total_norm = total_norm ** 0.5

    if total_norm > threshold:
        raise GradientExplosionError(total_norm, threshold, epoch, step)

    return total_norm


def clip_gradients(named_params, max_norm=1.0):
    """梯度裁剪（防止梯度爆炸的替代方案）"""
    params = [p for _, p in named_params if p.grad is not None]
    total_norm = sum(p.grad.norm()**2 for p in params) ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            p.grad.mul_(clip_coef)
    return total_norm


# ============================================================
# 检查点管理
# ============================================================

import json
import os

def save_checkpoint(state, filepath):
    """安全保存检查点（先写临时文件，再重命名，防止写入中断导致文件损坏）"""
    tmp_path = filepath + ".tmp"
    try:
        with open(tmp_path, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, filepath)
    except OSError as e:
        raise CheckpointError(f"保存检查点失败：{filepath}") from e
    finally:
        # 清理可能残留的临时文件
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def load_checkpoint(filepath):
    """加载检查点，如果不存在返回None"""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise CheckpointError(f"加载检查点失败：{filepath}") from e


# ============================================================
# 健壮的训练循环
# ============================================================

class RobustTrainer:
    """
    具备完整异常处理的训练器。
    展示了如何在深度学习训练中系统地运用异常处理与日志。
    """

    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("trainer")
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0
        self.last_valid_loss = None

        # 早停配置
        self.patience = config.get('patience', 10)
        self.grad_clip = config.get('grad_clip', 1.0)
        self.checkpoint_path = config.get('checkpoint_path', 'best_model.json')

    def train_one_epoch(self, epoch, dataloader, model, optimizer):
        """训练一个epoch，包含完整的异常处理"""
        total_loss = 0.0
        num_batches = len(dataloader)

        for step, (inputs, targets) in enumerate(dataloader):
            try:
                # 前向传播
                outputs = model(inputs)
                loss_val = compute_loss_fn(outputs, targets)

                # 检查损失
                is_normal = check_loss_value(
                    loss_val, epoch, step,
                    last_valid_loss=self.last_valid_loss
                )
                if not is_normal:
                    self.logger.warning(
                        f"Epoch {epoch}, Step {step}: 损失异常大 ({loss_val:.2f})，"
                        f"继续监视..."
                    )

                # 反向传播
                backward(loss_val)

                # 梯度裁剪（防止爆炸）
                grad_norm = clip_gradients(
                    model.named_parameters(),
                    max_norm=self.grad_clip
                )
                if grad_norm > 10.0:
                    self.logger.warning(
                        f"Step {step}: 梯度被裁剪，原始范数={grad_norm:.2f}"
                    )

                # 参数更新
                optimizer_step(optimizer)

                total_loss += loss_val
                self.last_valid_loss = loss_val

                if step % 50 == 0:
                    self.logger.debug(
                        f"  Step {step:4d}/{num_batches} | "
                        f"Loss: {loss_val:.6f} | GradNorm: {grad_norm:.4f}"
                    )

            except NaNLossError as e:
                self.logger.error(f"训练中断：{e}")
                # 尝试加载最近的检查点继续训练
                self.logger.info("尝试恢复到最佳检查点...")
                self._try_restore_checkpoint(model)
                raise  # 重新抛出，让外层决定是否继续

            except KeyboardInterrupt:
                self.logger.info("用户中断训练（Ctrl+C），保存当前状态...")
                self._save_state(epoch, step, total_loss / max(step, 1))
                raise  # 不吞掉 KeyboardInterrupt

        return total_loss / num_batches

    def validate(self, epoch, dataloader, model):
        """验证阶段"""
        total_loss = 0.0
        for step, (inputs, targets) in enumerate(dataloader):
            try:
                outputs = model(inputs)
                loss_val = compute_loss_fn(outputs, targets)
                total_loss += loss_val
            except Exception as e:
                self.logger.warning(f"验证步骤 {step} 出错，已跳过：{e}")
                continue

        return total_loss / len(dataloader)

    def train(self, model, train_loader, val_loader, optimizer):
        """主训练循环"""
        self.logger.info("=" * 60)
        self.logger.info(f"开始训练，最大 {self.config['epochs']} 轮")
        self.logger.info(f"梯度裁剪阈值：{self.grad_clip}")
        self.logger.info(f"早停耐心值：{self.patience}")
        self.logger.info("=" * 60)

        # 尝试恢复已有检查点
        start_epoch = self._resume_if_available(model)

        try:
            for epoch in range(start_epoch, self.config['epochs'] + 1):
                # 训练
                try:
                    train_loss = self.train_one_epoch(
                        epoch, train_loader, model, optimizer
                    )
                except NaNLossError:
                    self.logger.error("检测到NaN，终止训练")
                    break

                # 验证
                val_loss = self.validate(epoch, val_loader, model)

                self.logger.info(
                    f"Epoch {epoch:3d}/{self.config['epochs']} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
                )

                # 检查是否改善
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_epoch = epoch
                    self.no_improve_count = 0

                    # 保存最佳检查点
                    try:
                        self._save_checkpoint(model, epoch, val_loss)
                        self.logger.info(f"  新最佳模型已保存（Val Loss: {val_loss:.4f}）")
                    except CheckpointError as e:
                        self.logger.error(f"保存检查点失败：{e}，继续训练")

                else:
                    self.no_improve_count += 1
                    if self.no_improve_count >= self.patience:
                        raise EarlyStoppingSignal(self.best_epoch, self.best_loss)

        except EarlyStoppingSignal as e:
            self.logger.info(f"早停触发：{e}")
        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
        except Exception as e:
            self.logger.error(f"训练异常终止：{e}")
            self.logger.debug(traceback.format_exc())
            raise
        finally:
            self.logger.info("=" * 60)
            self.logger.info(
                f"训练结束 | 最佳Epoch: {self.best_epoch} | "
                f"最佳Val Loss: {self.best_loss:.4f}"
            )
            self.logger.info("=" * 60)

    def _save_checkpoint(self, model, epoch, val_loss):
        state = {
            'epoch': epoch,
            'val_loss': val_loss,
            'config': self.config,
        }
        save_checkpoint(state, self.checkpoint_path)

    def _try_restore_checkpoint(self, model):
        try:
            ckpt = load_checkpoint(self.checkpoint_path)
            if ckpt:
                self.logger.info(f"已恢复到 Epoch {ckpt['epoch']}，Val Loss: {ckpt['val_loss']:.4f}")
                return True
        except CheckpointError as e:
            self.logger.warning(f"恢复检查点失败：{e}")
        return False

    def _resume_if_available(self, model):
        try:
            ckpt = load_checkpoint(self.checkpoint_path)
            if ckpt:
                self.logger.info(f"找到检查点，从 Epoch {ckpt['epoch']+1} 继续训练")
                self.best_loss = ckpt.get('val_loss', float('inf'))
                return ckpt['epoch'] + 1
        except CheckpointError:
            pass
        return 1

    def _save_state(self, epoch, step, avg_loss):
        """紧急保存当前状态"""
        state = {
            'epoch': epoch,
            'step': step,
            'avg_loss': avg_loss,
            'interrupted': True
        }
        try:
            save_checkpoint(state, 'interrupted_state.json')
            self.logger.info("紧急状态已保存到 interrupted_state.json")
        except CheckpointError:
            self.logger.error("紧急保存也失败了！")


# ============================================================
# 占位符函数（实际使用时替换为真实实现）
# ============================================================

def compute_loss_fn(outputs, targets):
    """损失函数占位符"""
    return sum((o - t)**2 for o, t in zip(outputs, targets)) / len(outputs)

def backward(loss):
    """反向传播占位符"""
    pass

def optimizer_step(optimizer):
    """优化器更新占位符"""
    pass


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    config = {
        'epochs': 100,
        'patience': 10,
        'grad_clip': 1.0,
        'checkpoint_path': 'best_model.json',
        'learning_rate': 0.001,
    }

    trainer = RobustTrainer(config)
    # trainer.train(model, train_loader, val_loader, optimizer)
    print("RobustTrainer 已初始化，可开始训练")
```

---

## 练习题

### 基础题

**练习1：文件读取的异常处理**

编写一个函数 `safe_read_csv(filepath)`，要求：
- 如果文件不存在，打印提示并返回 `None`
- 如果文件存在但不是有效的CSV（用简单规则判断：每行逗号数量不一致），抛出 `ValueError`
- 如果文件成功读取，返回二维列表（每行是一个列表）
- 使用 `finally` 确保文件一定被关闭
- 使用 `else` 子句打印成功读取的行数

**练习2：学习率验证器**

创建一个 `LearningRateScheduler` 类，包含：
- 自定义异常 `InvalidLRError`，携带属性 `value`（无效值）和 `reason`（原因字符串）
- 方法 `set_lr(lr)`：验证 `lr` 是数值类型、大于0、小于1，否则抛出 `InvalidLRError`
- 方法 `decay(factor)`：将当前 `lr` 乘以 `factor`，若结果过小（< 1e-7）记录警告日志而非抛出异常
- 使用 `logging` 记录每次学习率变化

### 中级题

**练习3：带重试的数据加载器**

编写函数 `load_data_with_retry(url, max_retries=3, delay=1.0)`，模拟从网络加载数据：
- 自定义异常 `DataLoadError`，包含 `attempts`（已尝试次数）属性
- 如果加载失败（用随机数模拟：70%失败率），等待 `delay` 秒后重试
- 达到最大重试次数后，抛出 `DataLoadError`
- 每次重试都记录 WARNING 日志，最终失败记录 ERROR 日志
- 成功时记录 INFO 日志并返回数据

**练习4：训练日志分析器**

编写函数 `analyze_training_log(log_filepath)`，解析训练日志文件（格式如 `[INFO] Epoch  5/100 | Train: 0.4231 | Val: 0.3912`）：
- 如果日志文件不存在或格式不符，抛出适当的自定义异常
- 提取每个epoch的训练/验证损失
- 找出最佳epoch（验证损失最低）
- 检测损失是否出现异常跳升（相邻epoch损失变化 > 50%）
- 返回包含统计信息的字典

### 高级题

**练习5：健壮的模型配置加载系统**

设计一个配置加载系统，包含：

1. 异常层次：`ConfigError → MissingKeyError / InvalidValueError / TypeMismatchError`
2. 函数 `validate_config(config, schema)` 根据schema验证字典，schema格式为：
   ```python
   schema = {
       'learning_rate': {'type': float, 'range': (1e-6, 1.0), 'required': True},
       'batch_size': {'type': int, 'range': (1, 2048), 'required': True},
       'optimizer': {'type': str, 'choices': ['adam', 'sgd', 'adamw'], 'required': True},
       'dropout': {'type': float, 'range': (0.0, 1.0), 'required': False, 'default': 0.5},
   }
   ```
3. 函数 `load_and_validate_config(filepath, schema)` 加载JSON文件并验证
4. 使用异常链（`raise X from Y`）保留原始错误信息
5. 为整个加载过程配置日志（包括INFO、WARNING、DEBUG级别的输出）

---

## 练习答案

### 答案1：文件读取的异常处理

```python
def safe_read_csv(filepath):
    """安全读取CSV文件"""
    f = None
    try:
        f = open(filepath, 'r', encoding='utf-8')
        lines = f.readlines()

        if not lines:
            return []

        # 验证CSV格式一致性
        rows = [line.strip().split(',') for line in lines if line.strip()]
        expected_cols = len(rows[0])
        for i, row in enumerate(rows[1:], start=2):
            if len(row) != expected_cols:
                raise ValueError(
                    f"第 {i} 行列数（{len(row)}）与第1行（{expected_cols}）不一致"
                )
        return rows

    except FileNotFoundError:
        print(f"文件不存在：{filepath}")
        return None
    except ValueError:
        raise  # 重新抛出，让调用者处理格式错误
    finally:
        if f is not None:
            f.close()
            print(f"文件已关闭：{filepath}")
    # else 在 try 没有异常时执行
    # 注意：由于上面的 return 语句，else 等效写法是在 return 之前
    # 实际上 finally 中的打印替代了 else 的作用


# 更规范的写法（使用上下文管理器 + else）：
def safe_read_csv_v2(filepath):
    try:
        f = open(filepath, 'r', encoding='utf-8')
    except FileNotFoundError:
        print(f"文件不存在：{filepath}")
        return None
    else:
        try:
            lines = f.readlines()
        finally:
            f.close()

        rows = [line.strip().split(',') for line in lines if line.strip()]
        if not rows:
            return []

        expected_cols = len(rows[0])
        for i, row in enumerate(rows[1:], start=2):
            if len(row) != expected_cols:
                raise ValueError(f"第 {i} 行格式错误")

        print(f"成功读取 {len(rows)} 行")
        return rows
```

### 答案2：学习率验证器

```python
import logging

class InvalidLRError(Exception):
    """无效学习率异常"""
    def __init__(self, value, reason):
        self.value = value
        self.reason = reason
        super().__init__(f"无效学习率 {value}：{reason}")


class LearningRateScheduler:
    def __init__(self, initial_lr=0.01):
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.DEBUG,
                            format='[%(levelname)s] %(name)s: %(message)s')
        self.lr = self.set_lr(initial_lr)

    def set_lr(self, lr):
        if not isinstance(lr, (int, float)):
            raise InvalidLRError(lr, f"必须是数值类型，得到 {type(lr).__name__}")
        if lr <= 0:
            raise InvalidLRError(lr, "必须大于0")
        if lr >= 1:
            raise InvalidLRError(lr, "必须小于1")
        self.logger.info(f"学习率设置为 {lr:.2e}")
        self.lr = lr
        return lr

    def decay(self, factor):
        if not 0 < factor <= 1:
            raise ValueError(f"衰减因子必须在 (0, 1] 范围内，得到 {factor}")
        new_lr = self.lr * factor
        if new_lr < 1e-7:
            self.logger.warning(
                f"学习率 {new_lr:.2e} 过小（< 1e-7），可能导致训练停滞"
            )
        old_lr = self.lr
        self.lr = new_lr
        self.logger.info(f"学习率衰减：{old_lr:.2e} → {new_lr:.2e}（因子：{factor}）")
        return new_lr


# 测试
scheduler = LearningRateScheduler(0.01)
scheduler.decay(0.5)      # 0.01 → 0.005
scheduler.decay(0.1)      # 0.005 → 0.0005

try:
    scheduler.set_lr(-0.001)
except InvalidLRError as e:
    print(f"捕获异常：{e}，value={e.value}，reason={e.reason}")
```

### 答案3：带重试的数据加载器

```python
import logging
import random
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')


class DataLoadError(Exception):
    """数据加载失败"""
    def __init__(self, url, attempts):
        self.url = url
        self.attempts = attempts
        super().__init__(f"从 {url} 加载数据失败，已尝试 {attempts} 次")


def _mock_load(url):
    """模拟网络请求：70%失败率"""
    if random.random() < 0.7:
        raise ConnectionError(f"网络连接失败：{url}")
    return {"data": list(range(100)), "source": url}


def load_data_with_retry(url, max_retries=3, delay=1.0):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"尝试第 {attempt}/{max_retries} 次加载：{url}")
            data = _mock_load(url)
            logger.info(f"数据加载成功（第 {attempt} 次尝试）：{url}")
            return data
        except ConnectionError as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    f"加载失败（第 {attempt} 次）：{e}，"
                    f"{delay}秒后重试..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"加载最终失败，已尝试 {max_retries} 次：{e}"
                )

    raise DataLoadError(url, max_retries) from last_error


# 测试
random.seed(42)
try:
    data = load_data_with_retry("http://example.com/dataset.csv", max_retries=3)
    print(f"加载到 {len(data['data'])} 条数据")
except DataLoadError as e:
    print(f"最终失败：{e}（尝试次数：{e.attempts}）")
```

### 答案4：训练日志分析器

```python
import re
import logging

logger = logging.getLogger(__name__)


class LogParseError(Exception):
    """日志解析错误"""
    pass


def analyze_training_log(log_filepath):
    """解析训练日志，提取统计信息"""

    pattern = re.compile(
        r'\[INFO\s*\]\s*Epoch\s+(\d+)/\d+\s*\|\s*Train:\s*([\d.]+)\s*\|\s*Val:\s*([\d.]+)'
    )

    try:
        with open(log_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError as e:
        raise LogParseError(f"日志文件不存在：{log_filepath}") from e

    epochs = []
    for lineno, line in enumerate(lines, 1):
        m = pattern.search(line)
        if m:
            epochs.append({
                'epoch': int(m.group(1)),
                'train_loss': float(m.group(2)),
                'val_loss': float(m.group(3)),
            })

    if not epochs:
        raise LogParseError(f"日志中未找到符合格式的训练记录：{log_filepath}")

    # 最佳epoch
    best = min(epochs, key=lambda x: x['val_loss'])

    # 异常跳升检测
    anomalies = []
    for i in range(1, len(epochs)):
        prev_loss = epochs[i-1]['val_loss']
        curr_loss = epochs[i]['val_loss']
        if prev_loss > 0 and (curr_loss - prev_loss) / prev_loss > 0.5:
            anomalies.append({
                'epoch': epochs[i]['epoch'],
                'from': prev_loss,
                'to': curr_loss,
                'change_pct': (curr_loss - prev_loss) / prev_loss * 100
            })
            logger.warning(
                f"Epoch {epochs[i]['epoch']}: 验证损失异常跳升 "
                f"{prev_loss:.4f} → {curr_loss:.4f} "
                f"(+{(curr_loss-prev_loss)/prev_loss*100:.1f}%)"
            )

    return {
        'total_epochs': len(epochs),
        'best_epoch': best['epoch'],
        'best_val_loss': best['val_loss'],
        'final_train_loss': epochs[-1]['train_loss'],
        'final_val_loss': epochs[-1]['val_loss'],
        'loss_anomalies': anomalies,
        'epochs': epochs,
    }


# 测试
import tempfile, os

sample_log = """
[INFO    ] Epoch   1/100 | Train: 2.4321 | Val: 2.3156
[INFO    ] Epoch   2/100 | Train: 1.8932 | Val: 1.7843
[INFO    ] Epoch   3/100 | Train: 1.4521 | Val: 1.3902
[INFO    ] Epoch   4/100 | Train: 2.9900 | Val: 2.8100
[INFO    ] Epoch   5/100 | Train: 0.9823 | Val: 0.8934
"""
with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
    f.write(sample_log)
    tmp_path = f.name

try:
    result = analyze_training_log(tmp_path)
    print(f"总轮数：{result['total_epochs']}")
    print(f"最佳Epoch：{result['best_epoch']}，Val Loss：{result['best_val_loss']:.4f}")
    print(f"异常跳升次数：{len(result['loss_anomalies'])}")
finally:
    os.unlink(tmp_path)
```

### 答案5：健壮的模型配置加载系统

```python
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# 异常层次
class ConfigError(Exception):
    """配置相关错误基类"""
    pass

class MissingKeyError(ConfigError):
    def __init__(self, key):
        self.key = key
        super().__init__(f"缺少必要配置项：'{key}'")

class InvalidValueError(ConfigError):
    def __init__(self, key, value, reason):
        self.key = key
        self.value = value
        super().__init__(f"配置项 '{key}' 的值 {value!r} 无效：{reason}")

class TypeMismatchError(ConfigError):
    def __init__(self, key, expected_type, actual_type):
        self.key = key
        super().__init__(
            f"配置项 '{key}' 类型错误：期望 {expected_type.__name__}，"
            f"得到 {actual_type.__name__}"
        )


def validate_config(config, schema):
    """根据schema验证配置字典，返回填充了默认值的完整配置"""
    result = dict(config)

    for key, rules in schema.items():
        required = rules.get('required', True)

        if key not in config:
            if required:
                raise MissingKeyError(key)
            else:
                result[key] = rules.get('default')
                logger.debug(f"配置项 '{key}' 使用默认值：{rules.get('default')}")
            continue

        value = config[key]
        expected_type = rules.get('type')

        # 类型检查
        if expected_type and not isinstance(value, expected_type):
            raise TypeMismatchError(key, expected_type, type(value))

        # 范围检查
        if 'range' in rules:
            lo, hi = rules['range']
            if not (lo <= value <= hi):
                raise InvalidValueError(
                    key, value, f"必须在 [{lo}, {hi}] 范围内"
                )

        # 枚举检查
        if 'choices' in rules:
            if value not in rules['choices']:
                raise InvalidValueError(
                    key, value,
                    f"必须是 {rules['choices']} 之一"
                )

        logger.debug(f"配置项 '{key}' = {value!r} 验证通过")

    logger.info(f"配置验证完成，共 {len(schema)} 个字段")
    return result


def load_and_validate_config(filepath, schema):
    """加载并验证配置文件"""
    logger.info(f"加载配置文件：{filepath}")

    # 加载JSON
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_config = json.load(f)
        logger.debug(f"JSON解析成功，共 {len(raw_config)} 个顶层键")
    except FileNotFoundError as e:
        raise ConfigError(f"配置文件不存在：{filepath}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"配置文件JSON格式错误：{e}") from e
    except OSError as e:
        raise ConfigError(f"读取配置文件失败：{e}") from e

    # 验证配置
    try:
        validated = validate_config(raw_config, schema)
    except ConfigError:
        raise  # 直接传播，保留原始异常

    logger.info("配置加载和验证完成")
    return validated


# 测试
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)-8s] %(message)s')

SCHEMA = {
    'learning_rate': {'type': float, 'range': (1e-6, 1.0), 'required': True},
    'batch_size':    {'type': int,   'range': (1, 2048),   'required': True},
    'optimizer':     {'type': str,   'choices': ['adam', 'sgd', 'adamw'], 'required': True},
    'dropout':       {'type': float, 'range': (0.0, 1.0), 'required': False, 'default': 0.5},
}

# 测试有效配置
valid_config = {
    'learning_rate': 0.001,
    'batch_size': 128,
    'optimizer': 'adam',
}

try:
    result = validate_config(valid_config, SCHEMA)
    print(f"\n验证通过：{result}")
except ConfigError as e:
    print(f"验证失败：{e}")

# 测试无效配置
invalid_config = {
    'learning_rate': 2.0,  # 超出范围
    'batch_size': 128,
    'optimizer': 'adam',
}
try:
    validate_config(invalid_config, SCHEMA)
except InvalidValueError as e:
    print(f"\n捕获到 InvalidValueError：{e}")
    print(f"  错误的键：{e.key}，值：{e.value}")
```

---

> **下一章预告**：第8章将介绍Python的**面向对象编程进阶**，包括魔法方法、描述符、元类等特性，并展示如何用OOP思想设计深度学习框架的核心组件。
