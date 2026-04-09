# 第2章：控制流与函数

> **系列**：Python深度学习基础教程
> **难度**：初级 → 中级
> **预计学习时间**：3-4小时

---

## 学习目标

完成本章学习后，你将能够：

1. 使用 `if/elif/else` 编写条件判断逻辑，处理多分支场景
2. 熟练运用 `for` 和 `while` 循环遍历数据，并掌握 `break`、`continue`、`else` 等循环控制语句
3. 定义和调用函数，理解函数的意义与设计原则
4. 灵活使用位置参数、关键字参数、默认值参数、`*args` 和 `**kwargs`
5. 理解 Python 的作用域规则（LEGB），避免变量遮蔽和命名冲突

---

## 2.1 条件语句（if/elif/else）

### 2.1.1 基础语法

条件语句让程序根据不同的条件执行不同的代码分支。Python 使用缩进（通常为4个空格）来界定代码块。

```python
# 最简单的 if 语句
score = 85

if score >= 60:
    print("及格")

# if/else 二分支
if score >= 60:
    print("及格")
else:
    print("不及格")

# if/elif/else 多分支
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"成绩：{score}，等级：{grade}")  # 成绩：85，等级：B
```

### 2.1.2 比较运算符与逻辑运算符

```python
x = 10
y = 20

# 比较运算符
print(x == y)   # False  等于
print(x != y)   # True   不等于
print(x < y)    # True   小于
print(x > y)    # False  大于
print(x <= 10)  # True   小于等于
print(x >= 10)  # True   大于等于

# 逻辑运算符：and、or、not
age = 25
has_id = True

if age >= 18 and has_id:
    print("允许入场")

temperature = 35
if temperature < 0 or temperature > 40:
    print("极端天气警告")

is_raining = False
if not is_raining:
    print("适合出门")

# 链式比较（Python 特有语法，非常简洁）
x = 15
if 10 < x < 20:
    print("x 在 10 到 20 之间")  # 输出此行
```

### 2.1.3 真值判断与条件表达式

```python
# Python 中以下值被视为 False（假值）
# False, None, 0, 0.0, "", [], {}, ()

# 利用真值判断简化代码
data = [1, 2, 3]
if data:           # 等价于 if len(data) > 0
    print("数据非空")

name = ""
if not name:       # 等价于 if name == ""
    print("姓名为空")

# 三元条件表达式（一行写 if/else）
# 格式：值A if 条件 else 值B
score = 75
result = "及格" if score >= 60 else "不及格"
print(result)  # 及格

# 在深度学习中常见的用法
use_gpu = True
device = "cuda" if use_gpu else "cpu"
print(f"使用设备：{device}")  # 使用设备：cuda
```

### 2.1.4 match 语句（Python 3.10+）

```python
# Python 3.10 引入了结构化模式匹配，类似其他语言的 switch/case
status_code = 404

match status_code:
    case 200:
        print("请求成功")
    case 400:
        print("请求错误")
    case 404:
        print("资源未找到")
    case 500:
        print("服务器内部错误")
    case _:
        print(f"未知状态码：{status_code}")
```

---

## 2.2 循环语句（for、while、循环控制）

### 2.2.1 for 循环

`for` 循环用于遍历任何可迭代对象（列表、元组、字符串、字典等）。

```python
# 遍历列表
fruits = ["苹果", "香蕉", "橙子"]
for fruit in fruits:
    print(fruit)

# 遍历字符串
for char in "Python":
    print(char, end=" ")  # P y t h o n
print()

# 使用 range() 生成数字序列
for i in range(5):        # 0, 1, 2, 3, 4
    print(i, end=" ")
print()

for i in range(1, 6):     # 1, 2, 3, 4, 5
    print(i, end=" ")
print()

for i in range(0, 10, 2): # 0, 2, 4, 6, 8（步长为2）
    print(i, end=" ")
print()

# 反向遍历
for i in range(5, 0, -1): # 5, 4, 3, 2, 1
    print(i, end=" ")
print()
```

### 2.2.2 enumerate 与 zip

```python
# enumerate：同时获取索引和值
animals = ["猫", "狗", "鸟"]
for index, animal in enumerate(animals):
    print(f"{index}: {animal}")
# 0: 猫
# 1: 狗
# 2: 鸟

# enumerate 可以指定起始索引
for index, animal in enumerate(animals, start=1):
    print(f"第{index}个：{animal}")

# zip：同时遍历多个序列
names = ["Alice", "Bob", "Charlie"]
scores = [92, 85, 78]
for name, score in zip(names, scores):
    print(f"{name}: {score}分")
# Alice: 92分
# Bob: 85分
# Charlie: 78分

# 遍历字典
model_params = {"learning_rate": 0.001, "batch_size": 32, "epochs": 100}
for key, value in model_params.items():
    print(f"{key} = {value}")
```

### 2.2.3 while 循环

`while` 循环在条件为真时持续执行，适合循环次数不确定的场景。

```python
# 基础 while 循环
count = 0
while count < 5:
    print(f"count = {count}")
    count += 1  # 必须有递增，否则无限循环

# 用 while 实现输入验证
# （实际运行时会等待用户输入，此处仅展示结构）
# while True:
#     user_input = input("请输入一个正整数：")
#     if user_input.isdigit() and int(user_input) > 0:
#         number = int(user_input)
#         break
#     print("输入无效，请重试")

# 深度学习中的训练终止条件
loss = 1.0
epoch = 0
tolerance = 0.01

while loss > tolerance:
    loss *= 0.8   # 模拟 loss 下降
    epoch += 1
    print(f"Epoch {epoch}: loss = {loss:.4f}")

print(f"训练完成，共 {epoch} 轮")
```

### 2.2.4 循环控制语句

```python
# break：立即退出循环
print("=== break 示例 ===")
for i in range(10):
    if i == 5:
        break          # 当 i 等于 5 时退出循环
    print(i, end=" ")  # 0 1 2 3 4
print()

# continue：跳过当前迭代，继续下一次
print("=== continue 示例 ===")
for i in range(10):
    if i % 2 == 0:
        continue       # 跳过偶数
    print(i, end=" ")  # 1 3 5 7 9
print()

# for/else 和 while/else：else 在循环正常结束（非 break）后执行
print("=== for/else 示例 ===")
target = 7
for num in [1, 3, 5, 9]:
    if num == target:
        print(f"找到目标 {target}")
        break
else:
    print(f"未找到目标 {target}")  # 输出此行，因为列表中没有 7

# 嵌套循环中的 break 只退出最内层循环
print("=== 嵌套循环 ===")
for i in range(3):
    for j in range(3):
        if j == 1:
            break        # 只退出内层循环
        print(f"({i},{j})", end=" ")
    print()
```

### 2.2.5 列表推导式（循环的简洁写法）

```python
# 传统写法
squares = []
for x in range(1, 6):
    squares.append(x ** 2)
print(squares)  # [1, 4, 9, 16, 25]

# 列表推导式（更简洁）
squares = [x ** 2 for x in range(1, 6)]
print(squares)  # [1, 4, 9, 16, 25]

# 带条件的列表推导式
even_squares = [x ** 2 for x in range(1, 11) if x % 2 == 0]
print(even_squares)  # [4, 16, 36, 64, 100]

# 字典推导式
word_lengths = {word: len(word) for word in ["python", "deep", "learning"]}
print(word_lengths)  # {'python': 6, 'deep': 4, 'learning': 8}

# 集合推导式
unique_lengths = {len(word) for word in ["cat", "dog", "fish", "bird"]}
print(unique_lengths)  # {3, 4}（集合，顺序不固定）

# 深度学习中常用：批量归一化
data = [10, 20, 30, 40, 50]
mean = sum(data) / len(data)
normalized = [(x - mean) / mean for x in data]
print(normalized)  # [-0.8, -0.4, 0.0, 0.4, 0.8]
```

---

## 2.3 函数定义与调用

### 2.3.1 为什么需要函数

函数是将一段可复用代码封装起来的机制。使用函数的好处：

- **避免重复**：相同逻辑只写一次
- **提高可读性**：给代码块起一个有意义的名字
- **便于测试**：每个函数可以独立测试
- **易于维护**：修改一处，全局生效

### 2.3.2 定义和调用函数

```python
# 最简单的函数
def greet():
    print("你好，世界！")

greet()  # 调用函数：你好，世界！

# 带参数的函数
def greet_person(name):
    print(f"你好，{name}！")

greet_person("Alice")  # 你好，Alice！

# 带返回值的函数
def add(a, b):
    return a + b

result = add(3, 5)
print(result)  # 8

# 函数可以有多个返回值（实际上返回一个元组）
def min_max(numbers):
    return min(numbers), max(numbers)

low, high = min_max([3, 1, 4, 1, 5, 9, 2, 6])
print(f"最小值：{low}，最大值：{high}")  # 最小值：1，最大值：9
```

### 2.3.3 函数文档字符串

良好的文档字符串是专业代码的标志：

```python
def calculate_accuracy(predictions, labels):
    """
    计算分类准确率。

    Args:
        predictions (list): 模型预测的类别列表
        labels (list): 真实标签列表

    Returns:
        float: 准确率，范围在 [0.0, 1.0] 之间

    Example:
        >>> calculate_accuracy([1, 0, 1, 1], [1, 0, 0, 1])
        0.75
    """
    if len(predictions) != len(labels):
        raise ValueError("预测和标签的长度必须相同")

    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels)

# 查看文档字符串
help(calculate_accuracy)
# 或者
print(calculate_accuracy.__doc__)

# 使用函数
preds = [1, 0, 1, 1, 0]
true_labels = [1, 0, 0, 1, 0]
acc = calculate_accuracy(preds, true_labels)
print(f"准确率：{acc:.2%}")  # 准确率：80.00%
```

### 2.3.4 函数是一等对象

在 Python 中，函数本身也是对象，可以赋值给变量、作为参数传递、作为返回值：

```python
# 函数赋值给变量
def square(x):
    return x ** 2

f = square          # f 和 square 指向同一个函数
print(f(5))        # 25

# 函数作为参数（高阶函数）
def apply(func, values):
    return [func(v) for v in values]

result = apply(square, [1, 2, 3, 4, 5])
print(result)  # [1, 4, 9, 16, 25]

# 使用内置高阶函数
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
print(sorted(numbers))                    # [1, 1, 2, 3, 4, 5, 6, 9]
print(sorted(numbers, reverse=True))      # [9, 6, 5, 4, 3, 2, 1, 1]

# 按字符串长度排序
words = ["banana", "apple", "kiwi", "cherry"]
print(sorted(words, key=len))             # ['kiwi', 'apple', 'banana', 'cherry']

# lambda 匿名函数：用于简单的单行函数
double = lambda x: x * 2
print(double(7))   # 14

# lambda 常与 sorted、map、filter 配合使用
students = [("Alice", 92), ("Bob", 85), ("Charlie", 95)]
sorted_students = sorted(students, key=lambda s: s[1], reverse=True)
print(sorted_students)  # [('Charlie', 95), ('Alice', 92), ('Bob', 85)]
```

---

## 2.4 函数参数

Python 的函数参数系统非常灵活，支持多种传参方式。

### 2.4.1 位置参数与关键字参数

```python
def describe_model(name, layers, learning_rate):
    print(f"模型：{name}，层数：{layers}，学习率：{learning_rate}")

# 位置参数：按顺序传入
describe_model("ResNet", 50, 0.001)
# 模型：ResNet，层数：50，学习率：0.001

# 关键字参数：按名称传入，顺序无关
describe_model(learning_rate=0.001, name="ResNet", layers=50)
# 模型：ResNet，层数：50，学习率：0.001

# 混合使用：位置参数必须在关键字参数之前
describe_model("ResNet", layers=50, learning_rate=0.001)
```

### 2.4.2 默认值参数

```python
def create_optimizer(lr=0.001, momentum=0.9, weight_decay=1e-4):
    """
    创建优化器配置（使用常见的默认超参数）。
    """
    config = {
        "learning_rate": lr,
        "momentum": momentum,
        "weight_decay": weight_decay
    }
    return config

# 全部使用默认值
opt1 = create_optimizer()
print(opt1)
# {'learning_rate': 0.001, 'momentum': 0.9, 'weight_decay': 0.0001}

# 只修改部分参数
opt2 = create_optimizer(lr=0.01)
print(opt2)
# {'learning_rate': 0.01, 'momentum': 0.9, 'weight_decay': 0.0001}

# 全部自定义
opt3 = create_optimizer(lr=0.1, momentum=0.95, weight_decay=0.0)
print(opt3)
```

> **注意**：不要使用可变对象（列表、字典）作为默认值，这是常见陷阱！

```python
# 错误写法：默认列表在所有调用之间共享！
def add_item_bad(item, container=[]):
    container.append(item)
    return container

print(add_item_bad("苹果"))   # ['苹果']
print(add_item_bad("香蕉"))   # ['苹果', '香蕉'] ← 意外！

# 正确写法：默认值使用 None，在函数内创建新对象
def add_item_good(item, container=None):
    if container is None:
        container = []
    container.append(item)
    return container

print(add_item_good("苹果"))   # ['苹果']
print(add_item_good("香蕉"))   # ['香蕉'] ← 正确
```

### 2.4.3 *args：可变位置参数

`*args` 将多余的位置参数收集为一个元组：

```python
def sum_all(*args):
    """接受任意数量的数字并求和。"""
    total = 0
    for num in args:
        total += num
    return total

print(sum_all(1, 2, 3))          # 6
print(sum_all(1, 2, 3, 4, 5))    # 15
print(sum_all())                  # 0

# *args 可以与普通参数混用，但必须放在后面
def greet_many(greeting, *names):
    for name in names:
        print(f"{greeting}，{name}！")

greet_many("你好", "Alice", "Bob", "Charlie")
# 你好，Alice！
# 你好，Bob！
# 你好，Charlie！

# 展开列表/元组传入参数（解包操作符 *）
numbers = [1, 2, 3, 4, 5]
print(sum_all(*numbers))  # 等价于 sum_all(1, 2, 3, 4, 5)，输出 15
```

### 2.4.4 **kwargs：可变关键字参数

`**kwargs` 将多余的关键字参数收集为一个字典：

```python
def print_config(**kwargs):
    """打印任意配置项。"""
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

print("训练配置：")
print_config(epochs=100, batch_size=32, learning_rate=0.001, optimizer="Adam")
# 训练配置：
#   epochs: 100
#   batch_size: 32
#   learning_rate: 0.001
#   optimizer: Adam

# **kwargs 可以与其他参数混用
def train_model(model_name, **hyperparams):
    print(f"训练模型：{model_name}")
    print("超参数：")
    for k, v in hyperparams.items():
        print(f"  {k} = {v}")

train_model("CNN", lr=0.001, batch_size=64, dropout=0.5)

# 展开字典传入关键字参数（解包操作符 **）
config = {"lr": 0.001, "batch_size": 64, "dropout": 0.5}
train_model("ResNet", **config)  # 等价于 train_model("ResNet", lr=0.001, ...)
```

### 2.4.5 参数顺序综合示例

函数参数的完整顺序规则：

```
def func(普通参数, 默认值参数, *args, 仅关键字参数, **kwargs)
```

```python
def comprehensive_example(a, b, c=10, *args, keyword_only=99, **kwargs):
    print(f"a={a}, b={b}, c={c}")
    print(f"args={args}")
    print(f"keyword_only={keyword_only}")
    print(f"kwargs={kwargs}")

comprehensive_example(1, 2, 3, 4, 5, keyword_only=42, x=100, y=200)
# a=1, b=2, c=3
# args=(4, 5)
# keyword_only=42
# kwargs={'x': 100, 'y': 200}

# 深度学习函数的典型参数设计
def build_model(
    input_size,          # 必填：输入维度
    output_size,         # 必填：输出维度
    hidden_size=128,     # 可选：隐藏层大小
    num_layers=2,        # 可选：层数
    activation="relu",   # 可选：激活函数
    dropout=0.0,         # 可选：dropout 比例
    **extra_kwargs       # 额外的自定义参数
):
    """构建神经网络模型配置。"""
    config = {
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "activation": activation,
        "dropout": dropout,
        **extra_kwargs
    }
    return config

# 简单使用
model = build_model(784, 10)
print(model)

# 详细配置
model = build_model(
    input_size=784,
    output_size=10,
    hidden_size=256,
    num_layers=3,
    activation="leaky_relu",
    dropout=0.3,
    batch_norm=True,      # 通过 **kwargs 传入
    weight_init="kaiming"
)
print(model)
```

---

## 2.5 作用域与命名空间（LEGB规则）

### 2.5.1 什么是作用域

作用域决定了变量在程序的哪些位置可以被访问。Python 使用 **LEGB** 规则来查找变量：

| 字母 | 作用域 | 说明 |
|------|--------|------|
| **L** | Local（局部） | 当前函数内部定义的变量 |
| **E** | Enclosing（闭包） | 外层函数的局部变量（嵌套函数时） |
| **G** | Global（全局） | 模块级别的变量 |
| **B** | Built-in（内置） | Python 内置的名称（如 `len`、`print`） |

```python
# 内置作用域（B）
print(len([1, 2, 3]))  # len 是内置函数

# 全局作用域（G）
global_var = "我是全局变量"

def outer_function():
    # 闭包作用域（E）
    enclosing_var = "我是外层函数的变量"

    def inner_function():
        # 局部作用域（L）
        local_var = "我是局部变量"

        # LEGB 查找顺序：先找 L，再找 E，再找 G，最后找 B
        print(local_var)       # 找到 L
        print(enclosing_var)   # 找到 E
        print(global_var)      # 找到 G

    inner_function()

outer_function()
```

### 2.5.2 局部变量与全局变量

```python
x = 100  # 全局变量

def modify_local():
    x = 200  # 创建了一个同名的局部变量，不影响全局变量
    print(f"函数内 x = {x}")  # 200

modify_local()
print(f"函数外 x = {x}")  # 100（全局变量未改变）

# 使用 global 关键字修改全局变量
count = 0

def increment():
    global count   # 声明使用全局变量
    count += 1

increment()
increment()
increment()
print(f"count = {count}")  # count = 3

# 深度学习中的实际例子：记录训练步数
training_step = 0

def train_one_batch(data, labels):
    global training_step
    # ... 实际训练代码 ...
    training_step += 1
    return training_step
```

### 2.5.3 nonlocal 关键字

在嵌套函数中，`nonlocal` 用于修改外层（非全局）函数的变量：

```python
def make_counter(start=0):
    """创建一个计数器函数（闭包示例）。"""
    count = start  # 外层函数的局部变量

    def counter():
        nonlocal count    # 声明修改外层变量
        count += 1
        return count

    return counter        # 返回内层函数

# 创建两个独立的计数器
counter_a = make_counter(0)
counter_b = make_counter(10)

print(counter_a())  # 1
print(counter_a())  # 2
print(counter_b())  # 11
print(counter_a())  # 3
print(counter_b())  # 12

# 两个计数器相互独立，各自维护自己的 count 变量
```

### 2.5.4 闭包的应用

闭包是深度学习框架（如 PyTorch）背后的重要概念之一：

```python
def make_lr_scheduler(initial_lr, decay_rate):
    """
    创建一个学习率衰减调度器（闭包模式）。

    每次调用时，学习率按 decay_rate 衰减。
    """
    lr = initial_lr
    step = 0

    def get_lr():
        nonlocal lr, step
        current_lr = lr * (decay_rate ** step)
        step += 1
        return current_lr

    return get_lr

# 创建调度器
scheduler = make_lr_scheduler(initial_lr=0.1, decay_rate=0.9)

for epoch in range(5):
    current_lr = scheduler()
    print(f"Epoch {epoch+1}: lr = {current_lr:.6f}")
# Epoch 1: lr = 0.100000
# Epoch 2: lr = 0.090000
# Epoch 3: lr = 0.081000
# Epoch 4: lr = 0.072900
# Epoch 5: lr = 0.065610
```

### 2.5.5 避免常见的作用域陷阱

```python
# 陷阱1：循环变量在闭包中的捕获问题
# 错误示例
funcs_bad = []
for i in range(3):
    funcs_bad.append(lambda: i)  # 所有 lambda 都捕获同一个 i

# 循环结束后 i = 2，所以全部输出 2
print([f() for f in funcs_bad])  # [2, 2, 2] ← 意外！

# 正确示例：使用默认参数捕获当前值
funcs_good = []
for i in range(3):
    funcs_good.append(lambda x=i: x)  # 用默认参数固定当前 i 的值

print([f() for f in funcs_good])  # [0, 1, 2] ← 正确

# 陷阱2：函数内使用全局变量前赋值会导致 UnboundLocalError
total = 100

def problematic():
    # Python 看到下面有 total = ...，就将 total 视为局部变量
    # 但在赋值前先读取，导致 UnboundLocalError
    # print(total)  # 错误！
    total = total + 1  # 这里报 UnboundLocalError
    return total

# 解决方案：
def fixed():
    global total
    total = total + 1
    return total
```

---

## 本章小结

| 概念 | 要点 | 典型用法 |
|------|------|----------|
| `if/elif/else` | 多分支判断，使用缩进界定代码块 | 验证参数、选择设备 |
| `for` 循环 | 遍历可迭代对象，搭配 `enumerate`/`zip` | 遍历数据集、批次处理 |
| `while` 循环 | 条件为真时持续执行 | 训练到收敛、等待用户输入 |
| `break/continue` | 控制循环执行流 | 早停（early stopping） |
| 列表推导式 | 简洁地创建列表 | 数据预处理、批量转换 |
| 函数定义 | `def` 关键字，支持文档字符串 | 封装复用逻辑 |
| 位置参数 | 按顺序传入 | 必填参数 |
| 关键字参数 | 按名称传入，顺序无关 | 提高可读性 |
| 默认值参数 | 提供合理默认值 | 超参数、配置项 |
| `*args` | 收集多余位置参数为元组 | 可变长参数 |
| `**kwargs` | 收集多余关键字参数为字典 | 灵活配置传递 |
| LEGB 规则 | 局部→闭包→全局→内置 | 理解变量查找顺序 |
| `global`/`nonlocal` | 修改外层作用域的变量 | 计数器、状态跟踪 |
| 闭包 | 内层函数记住外层函数的变量 | 工厂模式、调度器 |

---

## 深度学习应用：训练循环的控制流

深度学习中最核心的代码结构就是**训练循环**，它综合运用了本章所有知识点。下面展示一个典型的 epoch/batch 训练循环结构：

```python
import random

# ============================================================
# 模拟深度学习训练循环
# 使用纯 Python 演示控制流和函数的综合运用
# ============================================================

def create_dummy_data(num_samples=1000, input_dim=20, num_classes=5):
    """
    创建模拟数据集。

    Args:
        num_samples: 样本数量
        input_dim: 输入特征维度
        num_classes: 分类类别数

    Returns:
        tuple: (features, labels) 的列表
    """
    data = []
    for _ in range(num_samples):
        features = [random.gauss(0, 1) for _ in range(input_dim)]
        label = random.randint(0, num_classes - 1)
        data.append((features, label))
    return data


def create_batches(data, batch_size):
    """
    将数据集分割为批次。

    Args:
        data: 完整数据集
        batch_size: 每批的样本数

    Returns:
        生成器，每次产出一个批次
    """
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]


def simulate_forward_pass(batch):
    """
    模拟前向传播，返回 loss 和预测结果。

    在真实的 PyTorch 代码中，这里调用 model(inputs)。
    """
    batch_size = len(batch)
    # 模拟 loss（随机，逐渐减小）
    loss = random.uniform(0.5, 2.0)
    # 模拟预测结果
    predictions = [random.randint(0, 4) for _ in range(batch_size)]
    labels = [item[1] for item in batch]
    return loss, predictions, labels


def calculate_accuracy(predictions, labels):
    """计算批次准确率。"""
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels)


def train(
    num_epochs=10,
    batch_size=32,
    early_stopping_patience=3,
    min_delta=0.001,
    verbose=True
):
    """
    完整的训练循环，包含：
    - 多轮（epoch）训练
    - 批次（batch）处理
    - 早停（early stopping）
    - 训练历史记录

    Args:
        num_epochs: 最大训练轮数
        batch_size: 每批样本数
        early_stopping_patience: 早停耐心值（连续多少轮无改善则停止）
        min_delta: 判断改善的最小变化量
        verbose: 是否打印详细日志

    Returns:
        dict: 包含训练历史的字典
    """
    # 初始化数据
    train_data = create_dummy_data(num_samples=500)
    val_data = create_dummy_data(num_samples=100)

    # 训练历史记录
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # 早停状态
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"开始训练：最多 {num_epochs} 轮，批次大小 {batch_size}")
    print("=" * 60)

    # =========================================================
    # 外层循环：遍历每个 epoch
    # =========================================================
    for epoch in range(1, num_epochs + 1):

        # ---- 训练阶段 ----
        epoch_train_losses = []
        epoch_train_accs = []

        # 每个 epoch 开始前打乱数据
        random.shuffle(train_data)

        # 内层循环：遍历每个 batch
        for batch_idx, batch in enumerate(create_batches(train_data, batch_size)):
            # 前向传播 + 计算 loss
            loss, preds, labels = simulate_forward_pass(batch)

            # 模拟 loss 随训练下降
            loss = loss * (0.95 ** epoch)

            acc = calculate_accuracy(preds, labels)
            epoch_train_losses.append(loss)
            epoch_train_accs.append(acc)

        # 计算 epoch 平均指标
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        avg_train_acc = sum(epoch_train_accs) / len(epoch_train_accs)

        # ---- 验证阶段 ----
        val_losses = []
        val_accs = []

        for batch in create_batches(val_data, batch_size):
            loss, preds, labels = simulate_forward_pass(batch)
            loss = loss * (0.95 ** epoch)
            val_losses.append(loss)
            val_accs.append(calculate_accuracy(preds, labels))

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_acc = sum(val_accs) / len(val_accs)

        # ---- 记录历史 ----
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)

        # ---- 打印日志 ----
        if verbose:
            print(
                f"Epoch [{epoch:3d}/{num_epochs}]  "
                f"Train Loss: {avg_train_loss:.4f}  "
                f"Train Acc: {avg_train_acc:.2%}  "
                f"Val Loss: {avg_val_loss:.4f}  "
                f"Val Acc: {avg_val_acc:.2%}"
            )

        # =========================================================
        # 早停（Early Stopping）逻辑
        # =========================================================
        if avg_val_loss < best_val_loss - min_delta:
            # 有改善：重置耐心计数器，更新最佳 loss
            best_val_loss = avg_val_loss
            patience_counter = 0
            if verbose:
                print(f"  --> 验证 loss 改善，最佳 val_loss = {best_val_loss:.4f}")
        else:
            # 无改善：增加耐心计数器
            patience_counter += 1
            if verbose:
                print(f"  --> 无改善，耐心计数：{patience_counter}/{early_stopping_patience}")

            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发！在第 {epoch} 轮停止训练。")
                break  # 退出 epoch 循环

    print("=" * 60)
    print(f"训练结束。最终 val_loss = {history['val_loss'][-1]:.4f}")

    return history


def evaluate_history(history):
    """分析并打印训练历史摘要。"""
    train_losses = history["train_loss"]
    val_losses = history["val_loss"]

    best_epoch = val_losses.index(min(val_losses)) + 1
    best_val_loss = min(val_losses)

    print("\n训练历史摘要")
    print("-" * 40)
    print(f"总训练轮数：{len(train_losses)}")
    print(f"最佳验证 loss：{best_val_loss:.4f}（第 {best_epoch} 轮）")
    print(f"最终训练准确率：{history['train_acc'][-1]:.2%}")
    print(f"最终验证准确率：{history['val_acc'][-1]:.2%}")


# 运行训练
if __name__ == "__main__":
    random.seed(42)
    history = train(
        num_epochs=20,
        batch_size=32,
        early_stopping_patience=4,
        verbose=True
    )
    evaluate_history(history)
```

**代码结构解析：**

```
train()
├── for epoch in range(num_epochs):          # 外层循环：epoch
│   ├── for batch in create_batches(...):    # 内层循环：batch
│   │   ├── simulate_forward_pass()          # 前向传播
│   │   └── calculate_accuracy()            # 指标计算
│   ├── 验证阶段（类似的 batch 循环）
│   ├── 记录历史
│   └── if patience >= threshold: break     # 早停：提前退出 epoch 循环
```

这个结构与真实的 PyTorch/TensorFlow 训练代码高度相似，区别仅在于真实代码中需要调用框架 API 进行梯度计算和参数更新。

---

## 练习题

### 基础题

**练习 2-1**：FizzBuzz 变体

编写函数 `fizzbuzz(n)`，对 1 到 n 的每个数字：
- 如果能被 3 整除，输出 "Fizz"
- 如果能被 5 整除，输出 "Buzz"
- 如果既能被 3 又能被 5 整除，输出 "FizzBuzz"
- 否则，输出该数字本身

```python
# 调用示例
fizzbuzz(15)
# 预期输出：
# 1, 2, Fizz, 4, Buzz, Fizz, 7, 8, Fizz, Buzz, 11, Fizz, 13, 14, FizzBuzz
```

---

**练习 2-2**：统计单词频次

编写函数 `word_count(text)`，接收一段文本，返回每个单词出现次数的字典（忽略大小写）。

```python
text = "to be or not to be that is the question to be"
result = word_count(text)
print(result)
# 预期输出（顺序可以不同）：
# {'to': 3, 'be': 3, 'or': 1, 'not': 1, 'that': 1, 'is': 1, 'the': 1, 'question': 1}
```

---

### 进阶题

**练习 2-3**：计算滑动平均

在深度学习中，常用滑动平均平滑 loss 曲线。编写函数 `moving_average(data, window_size)`，计算给定窗口大小的滑动平均值。

```python
losses = [2.5, 2.1, 1.8, 1.6, 1.9, 1.4, 1.2, 1.1, 1.3, 1.0]
smoothed = moving_average(losses, window_size=3)
print(smoothed)
# 预期输出（保留2位小数）：
# [2.5, 2.3, 2.13, 1.83, 1.77, 1.63, 1.5, 1.23, 1.2, 1.13]
# （前 window_size-1 个元素使用已有数据的平均）
```

---

**练习 2-4**：灵活的超参数合并

编写函数 `merge_configs(base_config, **overrides)`，将覆盖参数合并到基础配置中，返回新的配置字典（不修改原始字典）。

```python
base = {
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "Adam"
}

# 创建一个新的配置，覆盖部分参数
new_config = merge_configs(base, lr=0.01, epochs=50, dropout=0.3)
print(new_config)
# {'lr': 0.01, 'batch_size': 32, 'epochs': 50, 'optimizer': 'Adam', 'dropout': 0.3}

# 原始配置不变
print(base["lr"])  # 0.001
```

---

### 挑战题

**练习 2-5**：实现带缓存的斐波那契函数

使用闭包实现一个带有调用缓存（记忆化）的斐波那契函数 `make_fib_cached()`，避免重复计算。同时统计缓存命中次数和总调用次数。

```python
fib = make_fib_cached()

print(fib(10))   # 55
print(fib(10))   # 55（从缓存中取）
print(fib(15))   # 610
print(fib(15))   # 610（从缓存中取）

stats = fib.get_stats()
print(stats)
# {'total_calls': 4, 'cache_hits': 2, 'hit_rate': '50.00%'}
```

提示：
1. 使用字典作为缓存存储，保存在闭包的外层变量中
2. 函数对象可以动态添加属性（`fib.get_stats = ...`）
3. 注意递归调用时也应使用缓存版本

---

## 练习答案

### 答案 2-1：FizzBuzz 变体

```python
def fizzbuzz(n):
    """
    输出 1 到 n 的 FizzBuzz 序列。

    Args:
        n: 上限（含）
    """
    results = []
    for i in range(1, n + 1):
        if i % 15 == 0:      # 先检查 15（3 和 5 的公倍数）
            results.append("FizzBuzz")
        elif i % 3 == 0:
            results.append("Fizz")
        elif i % 5 == 0:
            results.append("Buzz")
        else:
            results.append(str(i))
    print(", ".join(results))

fizzbuzz(15)
# 1, 2, Fizz, 4, Buzz, Fizz, 7, 8, Fizz, Buzz, 11, Fizz, 13, 14, FizzBuzz
```

**关键点**：必须先检查 `i % 15 == 0`，否则 `elif i % 3 == 0` 会先匹配 15，导致只输出 "Fizz"。

---

### 答案 2-2：统计单词频次

```python
def word_count(text):
    """
    统计文本中每个单词的出现次数（忽略大小写）。

    Args:
        text: 输入文本字符串

    Returns:
        dict: 单词 -> 出现次数
    """
    words = text.lower().split()  # 转小写并分割
    counts = {}
    for word in words:
        # 方式1：使用 if/else
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

        # 方式2（更简洁）：使用 dict.get()
        # counts[word] = counts.get(word, 0) + 1

    return counts

text = "to be or not to be that is the question to be"
result = word_count(text)
print(result)
# {'to': 3, 'be': 3, 'or': 1, 'not': 1, 'that': 1, 'is': 1, 'the': 1, 'question': 1}

# 进阶：找出出现最多的单词
most_common = max(result, key=lambda w: result[w])
print(f"出现最多的单词：'{most_common}'，共 {result[most_common]} 次")
```

---

### 答案 2-3：计算滑动平均

```python
def moving_average(data, window_size):
    """
    计算滑动平均（处理边界：使用已有数据）。

    Args:
        data: 数值列表
        window_size: 窗口大小

    Returns:
        list: 滑动平均值列表，长度与 data 相同
    """
    result = []
    for i in range(len(data)):
        # 取从 max(0, i-window_size+1) 到 i（含）的窗口
        start = max(0, i - window_size + 1)
        window = data[start: i + 1]
        avg = sum(window) / len(window)
        result.append(round(avg, 2))
    return result

losses = [2.5, 2.1, 1.8, 1.6, 1.9, 1.4, 1.2, 1.1, 1.3, 1.0]
smoothed = moving_average(losses, window_size=3)
print(smoothed)
# [2.5, 2.3, 2.13, 1.83, 1.77, 1.63, 1.5, 1.23, 1.2, 1.13]
```

---

### 答案 2-4：灵活的超参数合并

```python
def merge_configs(base_config, **overrides):
    """
    将覆盖参数合并到基础配置中，返回新字典（不修改原始）。

    Args:
        base_config: 基础配置字典
        **overrides: 要覆盖或新增的参数

    Returns:
        dict: 合并后的新配置字典
    """
    # 方式1：逐步复制和更新
    new_config = dict(base_config)   # 浅拷贝，不修改原始
    new_config.update(overrides)     # 更新/新增参数
    return new_config

    # 方式2（更简洁，Python 3.9+）：
    # return base_config | overrides

    # 方式3（字典解包，Python 3.5+）：
    # return {**base_config, **overrides}

base = {
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "Adam"
}

new_config = merge_configs(base, lr=0.01, epochs=50, dropout=0.3)
print(new_config)
# {'lr': 0.01, 'batch_size': 32, 'epochs': 50, 'optimizer': 'Adam', 'dropout': 0.3}

print(base["lr"])   # 0.001（原始配置未被修改）
```

---

### 答案 2-5：带缓存的斐波那契函数

```python
def make_fib_cached():
    """
    创建一个带缓存和统计功能的斐波那契函数（闭包实现）。

    Returns:
        function: 缓存版斐波那契函数，附带 get_stats 方法
    """
    cache = {}       # 缓存：{n: fib(n)}
    total_calls = 0
    cache_hits = 0

    def fib(n):
        nonlocal total_calls, cache_hits

        if n < 0:
            raise ValueError("n 必须为非负整数")

        total_calls += 1

        # 检查缓存
        if n in cache:
            cache_hits += 1
            return cache[n]

        # 基础情况
        if n <= 1:
            result = n
        else:
            # 递归：注意递归调用 fib()，会继续使用缓存
            result = fib(n - 1) + fib(n - 2)

        # 存入缓存
        cache[n] = result
        return result

    def get_stats():
        """返回调用统计信息。"""
        hit_rate = (cache_hits / total_calls * 100) if total_calls > 0 else 0
        return {
            "total_calls": total_calls,
            "cache_hits": cache_hits,
            "hit_rate": f"{hit_rate:.2f}%"
        }

    # 将 get_stats 附加到 fib 函数上
    fib.get_stats = get_stats
    return fib


# 测试
fib = make_fib_cached()

print(fib(10))   # 55
print(fib(10))   # 55（从缓存中取）
print(fib(15))   # 610
print(fib(15))   # 610（从缓存中取）

stats = fib.get_stats()
print(stats)
# {'total_calls': 4, 'cache_hits': 2, 'hit_rate': '50.00%'}

# 验证缓存效果：计算 fib(30) 并查看统计
fib2 = make_fib_cached()
result = fib2(30)
print(f"fib(30) = {result}")          # fib(30) = 832040
stats2 = fib2.get_stats()
print(f"总调用次数：{stats2['total_calls']}")
print(f"缓存命中次数：{stats2['cache_hits']}")
print(f"缓存命中率：{stats2['hit_rate']}")
```

**设计要点**：
- `cache` 字典存储在外层函数作用域，被 `fib` 闭包捕获
- 递归时调用的是 `fib(n-1)`（缓存版），而非原始递归，所以 `fib(10)` 的计算结果在计算 `fib(15)` 时可以复用
- `fib.get_stats = get_stats` 利用了 Python 函数也是对象的特性，动态添加方法

---

> **下一章预告**：第3章将深入介绍 Python 的数据结构——列表、元组、字典、集合——以及它们在处理深度学习数据时的高效用法。
