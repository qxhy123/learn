# 第1章：Python环境与基本语法

> **系列定位**：本教程面向希望将Python应用于深度学习的读者。第一部分（第1–6章）夯实Python基础，为后续NumPy、PyTorch和神经网络编程做好准备。

---

## 学习目标

完成本章学习后，你将能够：

1. 了解Python的发展历史及其在深度学习领域的地位
2. 在Windows、macOS、Linux三种平台上独立完成Python环境的安装与配置
3. 正确使用Python的基本数据类型（`int`、`float`、`bool`、`str`、`None`）并理解它们的底层特性
4. 熟练运用算术、比较、逻辑和位运算符，理解运算符优先级
5. 编写规范的Python输入输出代码，并养成良好的注释习惯

---

## 1.1 Python简介与发展历史

### 1.1.1 Python的诞生

Python由荷兰程序员**Guido van Rossum**创建，1991年发布第一个公开版本（0.9.0）。名字来源于英国喜剧团体"Monty Python"，而非蟒蛇。Guido在设计之初就将"可读性"列为最高优先级，这一哲学贯穿了整个语言的演化。

```
Python版本里程碑：

1991  Python 0.9.0  — 首个公开版本，包含函数、异常处理、列表
1994  Python 1.0    — 加入 lambda、map、filter、reduce
2000  Python 2.0    — 引入列表推导式、垃圾回收
2008  Python 3.0    — 重大重构，不向后兼容（print函数化、Unicode默认）
2020  Python 2.7    — 官方终止支持（EOL）
2023  Python 3.12   — 性能大幅提升，错误信息更清晰
2024  Python 3.13   — 实验性 JIT 编译器
```

### 1.1.2 Python的设计哲学

在Python交互式解释器中输入 `import this`，可以看到著名的"Python之禅"（The Zen of Python）：

```python
import this
```

```
The Zen of Python, by Tim Peters

Beautiful is better than ugly.          # 美丽胜于丑陋
Explicit is better than implicit.       # 明确胜于隐晦
Simple is better than complex.          # 简单胜于复杂
Readability counts.                     # 可读性很重要
```

### 1.1.3 Python在深度学习中的地位

Python已成为深度学习领域**事实上的标准语言**，原因如下：

| 优势 | 说明 |
|------|------|
| 生态丰富 | NumPy、PyTorch、TensorFlow、scikit-learn等顶级框架均以Python为主语言 |
| 开发效率高 | 动态类型、交互式REPL、Jupyter Notebook加速实验迭代 |
| 社区活跃 | 数百万开发者、大量教程和论文代码复现资源 |
| 性能可扩展 | 计算密集型部分由C/C++/CUDA实现，Python做"胶水层" |
| 学术推动 | 顶会论文（NeurIPS、ICML、CVPR）代码几乎清一色Python |

```python
# Python作为"胶水层"的典型示意
import torch  # C++/CUDA核心，Python接口

x = torch.randn(1000, 1000)   # 在GPU上创建随机张量（一行Python）
y = torch.mm(x, x.T)          # 矩阵乘法，底层CUDA并行计算
print(y.shape)                 # torch.Size([1000, 1000])
```

---

## 1.2 环境安装与配置

### 1.2.1 安装策略选择

深度学习开发推荐使用 **Miniconda/Anaconda** 而非系统Python，原因是：
- 支持多个独立环境（不同项目用不同Python版本和包版本）
- 方便安装CUDA依赖的二进制包（如PyTorch GPU版本）
- 环境可导出分享，团队协作更方便

### 1.2.2 Windows安装

**步骤一：下载Miniconda**

访问 https://docs.conda.io/en/latest/miniconda.html，下载 `Miniconda3-latest-Windows-x86_64.exe`。

**步骤二：安装**

```
安装时勾选选项（推荐）：
[x] Add Miniconda3 to my PATH environment variable
[x] Register Miniconda3 as my default Python
```

**步骤三：验证安装**

打开"Anaconda Prompt"或PowerShell：

```powershell
conda --version      # conda 23.x.x
python --version     # Python 3.11.x
```

**步骤四：创建深度学习专用环境**

```powershell
# 创建名为 dl_env 的环境，指定Python版本
conda create -n dl_env python=3.11 -y

# 激活环境
conda activate dl_env

# 安装核心包（CPU版PyTorch，入门用）
pip install torch torchvision numpy matplotlib jupyter

# 验证PyTorch
python -c "import torch; print(torch.__version__)"
```

### 1.2.3 macOS安装

**使用Homebrew + Miniconda（推荐）**

```bash
# 安装Homebrew（如未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装Miniconda
brew install --cask miniconda

# 初始化conda
conda init zsh   # 若使用bash则改为 conda init bash

# 重启终端后创建环境
conda create -n dl_env python=3.11 -y
conda activate dl_env
pip install torch torchvision numpy matplotlib jupyter
```

**Apple Silicon (M1/M2/M3) 特别说明**

```bash
# Apple Silicon原生支持（Metal Performance Shaders）
pip install torch torchvision  # 自动安装MPS支持版本

# 验证MPS可用性
python -c "import torch; print(torch.backends.mps.is_available())"  # True
```

### 1.2.4 Linux安装

```bash
# 下载并安装Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b  # -b 静默安装

# 初始化
~/miniconda3/bin/conda init bash
source ~/.bashrc

# 创建环境
conda create -n dl_env python=3.11 -y
conda activate dl_env

# 安装带CUDA支持的PyTorch（需要NVIDIA GPU，CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 验证GPU支持
python -c "import torch; print(torch.cuda.is_available())"  # True（有NVIDIA GPU时）
```

### 1.2.5 集成开发环境（IDE）推荐

| IDE | 适用场景 | 推荐程度 |
|-----|----------|----------|
| **VS Code + Python扩展** | 日常开发、脚本编写 | ★★★★★ |
| **Jupyter Notebook/Lab** | 数据探索、实验记录 | ★★★★★ |
| **PyCharm Professional** | 大型项目 | ★★★★☆ |
| **Google Colab** | 免费GPU、快速实验 | ★★★★☆ |

```bash
# 启动Jupyter Lab
jupyter lab

# 在浏览器打开后，新建Notebook即可开始编码
```

### 1.2.6 验证完整环境

```python
# 运行这段代码验证所有核心包已就绪
import sys
import torch
import numpy as np

print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"NumPy版本: {np.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 简单张量运算测试
x = torch.tensor([1.0, 2.0, 3.0])
print(f"张量: {x}")
print(f"均值: {x.mean():.4f}")
```

**期望输出示例：**
```
Python版本: 3.11.7 (main, ...)
PyTorch版本: 2.2.0
NumPy版本: 1.26.3
CUDA可用: True
张量: tensor([1., 2., 3.])
均值: 2.0000
```

---

## 1.3 变量与数据类型

### 1.3.1 变量的本质

Python中变量是**对象的引用**（标签），而非存储值的盒子。理解这一点对后续理解深拷贝/浅拷贝、张量引用非常重要。

```python
# 变量赋值：x 是指向整数对象 42 的标签
x = 42
print(type(x))   # <class 'int'>
print(id(x))     # 对象在内存中的地址，例如 140234567890

# 重新赋值：x 改为指向新对象，原对象可能被垃圾回收
x = "hello"
print(type(x))   # <class 'str'>

# 多重赋值
a = b = c = 0
print(a, b, c)   # 0 0 0

# 链式赋值（同时赋多个变量）
width, height, depth = 1920, 1080, 3
print(f"图像尺寸: {width}x{height}x{depth}")  # 图像尺寸: 1920x1080x3
```

**变量命名规范（PEP 8）：**

```python
# 正确命名示例
learning_rate = 0.001          # 蛇形命名（snake_case）：变量和函数
MAX_EPOCHS = 100               # 全大写：常量
BatchNormLayer = None          # 驼峰命名（PascalCase）：类名（后续章节）
_internal_var = "私有变量"     # 单下划线前缀：约定俗成的"内部使用"

# 错误命名（会导致SyntaxError）
# 2weight = 0.5     # 不能以数字开头
# for = 10          # 不能使用关键字
# my-var = 1        # 不能含连字符
```

### 1.3.2 整数类型（int）

Python的整数是**任意精度**的，不存在溢出问题。

```python
# 基本整数
a = 42
b = -17
c = 0

# 大整数（Python自动处理，无溢出）
big = 2 ** 1000
print(big)          # 一个极大的数，正常显示

# 不同进制字面量
binary  = 0b1010    # 二进制：10
octal   = 0o17      # 八进制：15
hex_val = 0xFF      # 十六进制：255

print(binary, octal, hex_val)   # 10 15 255

# 进制转换函数
n = 255
print(bin(n))   # 0b11111111
print(oct(n))   # 0o377
print(hex(n))   # 0xff

# 整数的常用方法
print(abs(-42))        # 42，绝对值
print(pow(2, 10))      # 1024，等同于 2**10
print(divmod(17, 5))   # (3, 2)，商和余数

# 深度学习中常见的整数用法
batch_size   = 32
num_classes  = 1000    # ImageNet类别数
image_width  = 224
num_epochs   = 50
```

### 1.3.3 浮点数类型（float）

Python的浮点数遵循IEEE 754双精度标准（64位），但存在精度问题。

```python
# 基本浮点数
pi = 3.14159265358979
e  = 2.71828182845905

# 科学计数法
tiny   = 1e-7     # 0.0000001
huge   = 6.022e23 # 阿伏伽德罗常数

print(tiny)   # 1e-07
print(huge)   # 6.022e+23

# 精度陷阱（重要！）
print(0.1 + 0.2)          # 0.30000000000000004，不是0.3！
print(0.1 + 0.2 == 0.3)   # False

# 正确的浮点数比较方式
import math
print(math.isclose(0.1 + 0.2, 0.3))          # True
print(abs(0.1 + 0.2 - 0.3) < 1e-9)           # True

# 特殊浮点值
inf      = float('inf')    # 正无穷
neg_inf  = float('-inf')   # 负无穷
nan      = float('nan')    # 非数字 (Not a Number)

print(inf > 1e308)     # True
print(nan == nan)      # False！NaN不等于自身
print(math.isnan(nan)) # True，检测NaN的正确方式

# 深度学习中常见的浮点数
learning_rate   = 0.001       # 学习率
momentum        = 0.9         # 动量参数
weight_decay    = 1e-4        # L2正则化系数
dropout_rate    = 0.5         # Dropout比例
epsilon         = 1e-8        # Adam优化器的数值稳定项
```

### 1.3.4 布尔类型（bool）

`bool`是`int`的子类，`True`等于1，`False`等于0。

```python
# 基本布尔值
flag_a = True
flag_b = False

print(type(True))   # <class 'bool'>
print(True + 1)     # 2，bool是int的子类
print(False + 5)    # 5

# 真值测试（Truthiness）
# 以下值被视为False：
falsy_values = [False, 0, 0.0, 0j, "", [], {}, (), None]
for v in falsy_values:
    print(f"bool({v!r}) = {bool(v)}")

# 输出：
# bool(False) = False
# bool(0) = False
# bool(0.0) = False
# ...全为False

# 非零非空值均为True
print(bool(42))       # True
print(bool(-1))       # True（负数也是True！）
print(bool("hello"))  # True
print(bool([0]))      # True（非空列表）

# 深度学习中的布尔用法
use_gpu        = torch.cuda.is_available()   # True/False
requires_grad  = True                         # 是否计算梯度
is_training    = True                         # 是否处于训练模式
```

### 1.3.5 字符串类型（str）

Python字符串是**不可变的Unicode序列**。

```python
# 创建字符串的多种方式
s1 = 'single quotes'
s2 = "double quotes"
s3 = '''多行
字符串'''
s4 = """也是
多行"""

# 原始字符串（raw string）：反斜杠不转义
path_win  = r"C:\Users\yang\data"    # Windows路径常用
regex_pat = r"\d+\.\d+"             # 正则表达式常用

# 字符串基本操作
s = "Hello, Python!"

print(len(s))           # 14，长度
print(s[0])             # H，索引（从0开始）
print(s[-1])            # !，负索引（从末尾计）
print(s[7:13])          # Python，切片 [start:end]
print(s[::2])           # Hlo yhn，步长为2
print(s[::-1])          # !nohtyP ,olleH，反转

# 字符串方法
text = "  Deep Learning  "
print(text.strip())          # "Deep Learning"，去除首尾空白
print(text.lower())          # "  deep learning  "
print(text.upper())          # "  DEEP LEARNING  "
print("hello".replace("l","r"))   # "herro"

# 字符串格式化（三种方式）
name = "PyTorch"
version = 2.2

# 方式1：% 格式化（旧式，不推荐）
print("框架: %s, 版本: %.1f" % (name, version))

# 方式2：str.format()（中式）
print("框架: {}, 版本: {:.1f}".format(name, version))

# 方式3：f-string（推荐，Python 3.6+）
print(f"框架: {name}, 版本: {version:.1f}")
print(f"计算: {2**10 = }")   # Python 3.8+ 调试语法

# f-string进阶用法
loss = 0.0234567
epoch = 42
print(f"Epoch {epoch:03d} | Loss: {loss:.6f}")  # Epoch 042 | Loss: 0.023457
print(f"Loss: {loss:.2e}")                        # Loss: 2.35e-02

# 字符串分割与连接
csv_line = "128,0.001,0.9,100"
parts = csv_line.split(",")
print(parts)          # ['128', '0.001', '0.9', '100']

rejoined = " | ".join(parts)
print(rejoined)       # 128 | 0.001 | 0.9 | 100
```

### 1.3.6 None类型

`None`是Python中表示"空值"或"缺失"的单例对象。

```python
# None的基本用法
x = None
print(x)           # None
print(type(x))     # <class 'NoneType'>

# 检查是否为None（始终用 is，不用 ==）
if x is None:
    print("x未初始化")

if x is not None:
    print("x有值")

# 函数默认返回None
def no_return():
    pass   # 不写return

result = no_return()
print(result)          # None
print(result is None)  # True

# 深度学习中None的典型用法
def build_model(hidden_size=None):
    if hidden_size is None:
        hidden_size = 256    # 默认值
    # ... 构建模型
    return hidden_size

print(build_model())        # 256
print(build_model(512))     # 512
```

### 1.3.7 类型检查与转换

```python
# type() 检查确切类型
print(type(42))        # <class 'int'>
print(type(3.14))      # <class 'float'>
print(type(True))      # <class 'bool'>
print(type("hi"))      # <class 'str'>
print(type(None))      # <class 'NoneType'>

# isinstance() 检查类型（推荐，支持继承关系）
print(isinstance(True, bool))   # True
print(isinstance(True, int))    # True（bool是int子类）
print(isinstance(42, (int, float)))  # True，检查多类型

# 显式类型转换
print(int(3.9))        # 3，截断（不四舍五入）
print(int("42"))       # 42
print(int("0xFF", 16)) # 255，带进制转换
print(float(42))       # 42.0
print(float("3.14"))   # 3.14
print(str(42))         # "42"
print(str(3.14))       # "3.14"
print(bool(0))         # False
print(bool(42))        # True

# 类型转换出错
try:
    int("hello")
except ValueError as e:
    print(f"错误: {e}")   # 错误: invalid literal for int()...
```

---

## 1.4 运算符

### 1.4.1 算术运算符

```python
a, b = 17, 5

print(a + b)    # 22，加法
print(a - b)    # 12，减法
print(a * b)    # 85，乘法
print(a / b)    # 3.4，真除法（结果始终为float）
print(a // b)   # 3，整除（向下取整）
print(a % b)    # 2，取模（余数）
print(a ** b)   # 1419857，幂运算

# 整除的取整方向
print(7 // 2)    # 3
print(-7 // 2)   # -4，向下取整（负数注意！）
print(7 // -2)   # -4

# 取模与负数
print(7 % 3)     # 1
print(-7 % 3)    # 2，结果符号与除数相同

# 浮点数运算
print(2.0 ** 0.5)   # 1.4142135623730951，平方根

# 运算符优先级（从高到低摘录）
# ** > 一元+ - > * / // % > + -
print(2 + 3 * 4)    # 14，先乘后加
print(2 ** 3 ** 2)  # 512，**右结合：2**(3**2)=2**9=512

# 深度学习中的典型算术
batch_size   = 32
dataset_size = 50000
steps_per_epoch = dataset_size // batch_size   # 1562
total_steps     = steps_per_epoch * 100         # 156200

# 归一化计算
pixel_value  = 128
normalized   = (pixel_value - 128) / 255.0      # -0.00392...
```

### 1.4.2 增量赋值运算符

```python
x = 10

x += 5    # 等同于 x = x + 5  -> 15
x -= 3    # 等同于 x = x - 3  -> 12
x *= 2    # 等同于 x = x * 2  -> 24
x /= 4    # 等同于 x = x / 4  -> 6.0
x //= 2   # 等同于 x = x // 2 -> 3.0
x %= 2    # 等同于 x = x % 2  -> 1.0
x **= 3   # 等同于 x = x ** 3 -> 1.0

# 在训练循环中常见的用法
total_loss = 0.0
for batch_loss in [0.5, 0.4, 0.3, 0.35]:
    total_loss += batch_loss
avg_loss = total_loss / 4
print(f"平均损失: {avg_loss:.4f}")   # 平均损失: 0.3875
```

### 1.4.3 比较运算符

```python
a, b = 10, 20

print(a == b)   # False，等于
print(a != b)   # True，不等于
print(a < b)    # True，小于
print(a > b)    # False，大于
print(a <= b)   # True，小于等于
print(a >= b)   # False，大于等于

# Python支持链式比较（独特特性）
x = 5
print(1 < x < 10)        # True，等同于 (1<x) and (x<10)
print(0 <= x <= 100)     # True，检查范围
print(1 < x > 3)         # True，x>1 且 x>3

# 对象同一性比较（is vs ==）
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)   # True，值相等
print(a is b)   # False，不是同一对象
print(a is c)   # True，同一对象（c只是a的别名）

# 小整数缓存（CPython实现细节）
x = 256
y = 256
print(x is y)   # True，CPython缓存-5到256的整数

x = 257
y = 257
print(x is y)   # False（可能），超出缓存范围

# 深度学习场景：检查精度条件
val_accuracy = 0.943
print(val_accuracy > 0.90)    # True，是否达标
print(0.85 < val_accuracy < 0.99)  # True，合理范围检查
```

### 1.4.4 逻辑运算符

```python
# and、or、not 三个逻辑运算符
print(True and True)    # True
print(True and False)   # False
print(False and True)   # False
print(False or True)    # True
print(not True)         # False
print(not False)        # True

# 短路求值（Short-circuit Evaluation）
# and：左边为False，右边不执行
# or：左边为True，右边不执行

x = 0
result = x != 0 and (10 / x > 1)  # 安全，不会除以零
print(result)   # False

# or 的默认值模式
name = ""
display_name = name or "匿名用户"    # name为空，取右边的默认值
print(display_name)   # 匿名用户

config_lr = None
lr = config_lr or 0.001   # 若未配置则用默认值
print(lr)   # 0.001

# and 的惰性取值模式
user = {"name": "Alice", "age": 30}
# 只有user存在时才取name字段
greeting = user and f"Hello, {user['name']}"
print(greeting)   # Hello, Alice

# 深度学习场景
use_gpu  = torch.cuda.is_available()
use_amp  = True   # 自动混合精度

# 只有GPU可用时才启用AMP
enable_amp = use_gpu and use_amp
print(f"启用AMP: {enable_amp}")

# 检查多个停止条件
epoch       = 95
val_loss    = 0.001
patience    = 10
no_improve  = 12

should_stop = (epoch >= 100) or (val_loss < 1e-5) or (no_improve > patience)
print(f"应停止训练: {should_stop}")   # True
```

### 1.4.5 位运算符

位运算在深度学习中常用于标志位处理、数据增强掩码等场景。

```python
a = 0b1010   # 10
b = 0b1100   # 12

print(bin(a & b))   # 0b1000 = 8，按位与（AND）
print(bin(a | b))   # 0b1110 = 14，按位或（OR）
print(bin(a ^ b))   # 0b0110 = 6，按位异或（XOR）
print(bin(~a))      # -0b1011 = -11，按位取反（NOT）

# 移位运算
n = 1
print(n << 3)   # 8，左移3位 = 1 * 2^3
print(16 >> 2)  # 4，右移2位 = 16 / 2^2

# 实用技巧：用位运算检查奇偶
def is_even(n):
    return (n & 1) == 0   # 最低位为0则为偶数

print(is_even(4))    # True
print(is_even(7))    # False

# 2的幂运算（移位比**更快）
batch_sizes = [1 << i for i in range(5, 10)]
print(batch_sizes)   # [32, 64, 128, 256, 512]

# 标志位组合（深度学习中的增强选项）
FLIP_LR      = 0b0001   # 1，水平翻转
FLIP_UD      = 0b0010   # 2，垂直翻转
ROTATE       = 0b0100   # 4，旋转
COLOR_JITTER = 0b1000   # 8，颜色扰动

# 开启多个增强
augmentation_flags = FLIP_LR | ROTATE   # 0b0101 = 5

# 检查某个增强是否启用
if augmentation_flags & FLIP_LR:
    print("水平翻转已启用")    # 打印

if augmentation_flags & FLIP_UD:
    print("垂直翻转已启用")    # 不打印
```

### 1.4.6 运算符优先级总表

```
优先级（从高到低）：
  **                    幂
  +x  -x  ~x           一元正、负、按位取反
  *  /  //  %           乘、除、整除、取模
  +  -                  加、减
  <<  >>                位移
  &                     按位与
  ^                     按位异或
  |                     按位或
  ==  !=  <  >  <=  >=  is  is not  in  not in   比较
  not                   逻辑非
  and                   逻辑与
  or                    逻辑或
  :=                    海象运算符（Python 3.8+）
```

```python
# 优先级示例
print(2 + 3 * 4)         # 14，*优先于+
print(1 < 2 + 3 < 10)   # True，+先计算
print(not True or False) # False，not先计算，再or
print(not (True or False)) # False

# 使用括号提高可读性（强烈推荐）
result = ((a + b) * c) / d   # 好习惯，意图明确
```

---

## 1.5 基本输入输出与注释

### 1.5.1 print() 函数详解

```python
# 基本打印
print("Hello, World!")      # Hello, World!
print(42)                    # 42
print(3.14)                  # 3.14
print(True)                  # True

# 打印多个值
print("a =", 1, "b =", 2)   # a = 1 b = 2（默认空格分隔）

# sep 参数：自定义分隔符
print(2024, 4, 9, sep="-")   # 2024-4-9
print("root", "home", "user", sep="/")   # root/home/user

# end 参数：自定义结束符（默认\n换行）
print("Loading", end="")
print(".", end="")
print(".", end="")
print(".", end="\n")     # 最终打印：Loading...

# 进度条模式（训练时常用）
import time
for i in range(5):
    print(f"\rEpoch {i+1}/5", end="", flush=True)
    time.sleep(0.1)
print()   # 换行

# 打印到文件（重定向）
import sys
print("这是错误信息", file=sys.stderr)    # 打印到stderr

with open("/tmp/log.txt", "w") as f:
    print("训练日志", file=f)             # 打印到文件

# 格式化输出回顾
epoch = 10
loss  = 0.0345678
acc   = 0.9234

print(f"Epoch: {epoch:3d} | Loss: {loss:.4f} | Acc: {acc:.2%}")
# 输出：Epoch:  10 | Loss: 0.0346 | Acc: 92.34%

# 格式规范速查
# {:d}   整数
# {:f}   浮点（默认6位小数）
# {:.2f} 2位小数
# {:e}   科学计数法
# {:.2%} 百分比（自动乘100）
# {:>10} 右对齐，宽度10
# {:<10} 左对齐，宽度10
# {:^10} 居中对齐，宽度10
# {:0>5} 右对齐，宽度5，用0填充
```

### 1.5.2 input() 函数

```python
# 基本输入（结果始终是字符串）
name = input("请输入你的名字: ")
print(f"你好，{name}！")

# 类型转换
age_str = input("请输入你的年龄: ")
age = int(age_str)               # 转为整数
print(f"明年你将是 {age + 1} 岁")

# 简洁写法
lr = float(input("请输入学习率: "))   # 一步完成输入和转换

# 安全输入（处理转换错误）
def get_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("无效输入，请输入整数！")

# batch_size = get_int("请输入批大小: ")

# 多值输入（用split分割）
values = input("请输入三个数（空格分隔）: ").split()
a, b, c = int(values[0]), int(values[1]), int(values[2])
# 或更简洁：
a, b, c = map(int, input("请输入三个数: ").split())
print(f"和为 {a + b + c}")
```

### 1.5.3 注释规范

```python
# 单行注释：# 号后加空格，简明说明意图而非重复代码

learning_rate = 0.001  # 初始学习率，Adam优化器推荐范围 1e-4 ~ 1e-2

# 错误注释示例（不要这样写）
x = x + 1  # 将x加1（这只是重复代码，没有说明为什么）

# 正确注释示例（说明意图和原因）
x = x + 1  # 跳过索引0（保留给padding token）

# 多行注释：用多个 # 或三引号字符串（后者实为字符串字面量）
# 预处理步骤：
# 1. 归一化到 [0, 1]
# 2. 减去训练集均值
# 3. 除以训练集标准差

"""
这是一个多行字符串，常被误用为注释。
实际上它是一个被丢弃的字符串表达式。
不推荐作为普通注释使用，但可用于临时屏蔽代码块。
"""

# 文档字符串（Docstring）：函数/类/模块的官方注释
def normalize(x, mean, std):
    """
    对输入张量进行标准化。

    参数：
        x    (float or Tensor): 输入值
        mean (float): 均值
        std  (float): 标准差，不能为0

    返回：
        float: 标准化后的值，(x - mean) / std

    示例：
        >>> normalize(128.0, 127.5, 64.0)
        0.0078125
    """
    return (x - mean) / std

# 查看文档字符串
help(normalize)
# 或
print(normalize.__doc__)

# TODO / FIXME / NOTE 标记（常用约定）
# TODO: 添加对批量输入的支持
# FIXME: 当std=0时会触发除零错误
# NOTE: 此实现假设输入已经过范围检查
# HACK: 临时方案，待重构
# NOQA: 告诉linter忽略本行

# 代码块注释：说明一段逻辑的整体目的
# === 数据增强阶段 ===
# 训练时随机应用以下变换以提高泛化能力
# 验证/测试时跳过此块
if is_training:
    image = random_flip(image)
    image = random_crop(image, size=224)
    image = color_jitter(image, brightness=0.4)
```

### 1.5.4 代码风格：PEP 8快速参考

```python
# 好的代码风格示例（遵循PEP 8）

# 1. 缩进：4个空格（不用Tab）
def train_epoch(model, dataloader, optimizer):
    total_loss = 0.0
    for batch in dataloader:
        loss = compute_loss(model, batch)
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 2. 空行：函数/类之间2个空行，方法之间1个空行

# 3. 行长：不超过79字符（可适当放宽到99）
# 长表达式换行（在括号内换行，不需要反斜杠）
result = (first_variable
          + second_variable
          - third_variable)

# 4. 导入：每行一个，顺序：标准库→第三方→本地
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from .models import ResNet
from .utils import load_data

# 5. 空格：运算符两侧加空格，逗号后加空格，冒号后加空格
x = 1 + 2
my_list = [1, 2, 3]
my_dict = {"key": "value"}
```

---

## 本章小结

| 知识点 | 核心内容 | 深度学习应用 |
|--------|----------|-------------|
| Python简介 | 1991年诞生，强调可读性，3.x为主流 | AI/ML领域的主导语言 |
| 环境配置 | Miniconda管理多环境，conda/pip安装包 | 为PyTorch/TensorFlow环境隔离 |
| `int` | 任意精度，支持多进制 | batch_size、num_classes、epoch |
| `float` | IEEE 754双精度，有精度限制 | learning_rate、loss、accuracy |
| `bool` | `int`子类，True=1 False=0 | use_gpu、is_training、requires_grad |
| `str` | 不可变Unicode，f-string格式化 | 模型名称、日志输出、路径 |
| `None` | 单例，用`is`比较，表示缺失值 | 函数默认参数、可选配置项 |
| 算术运算符 | `//`整除，`**`幂，`%`取模 | 步数计算、归一化、幂运算 |
| 比较运算符 | 链式比较，`is`比较身份 | 精度检查、范围验证 |
| 逻辑运算符 | 短路求值，`or`提供默认值 | 条件训练逻辑、参数默认值 |
| 位运算符 | `&\|^~<<>>` | 增强标志位、2的幂批大小 |
| 输入输出 | `print()`格式化，`input()`字符串 | 训练日志、进度显示 |
| 注释规范 | 单行`#`、Docstring、TODO标记 | 说明超参、记录实验意图 |

---

## 深度学习应用：张量的数据类型

### 为什么数据类型在深度学习中至关重要？

深度学习训练涉及大量数值计算，选择正确的数据类型会直接影响：
- **内存占用**：float32比float64省一半内存
- **计算速度**：float16/bfloat16在现代GPU上更快
- **数值稳定性**：精度不足可能导致梯度消失/爆炸
- **硬件兼容**：不同GPU对不同精度的支持不同

### PyTorch中的dtype概览

```python
import torch

# 整数类型
i8  = torch.tensor(127,  dtype=torch.int8)    # 8位整数，范围-128~127
i16 = torch.tensor(1000, dtype=torch.int16)   # 16位整数
i32 = torch.tensor(1000, dtype=torch.int32)   # 32位整数（默认整数）
i64 = torch.tensor(1000, dtype=torch.int64)   # 64位整数（Python int默认映射）

# 浮点类型
f16  = torch.tensor(1.0, dtype=torch.float16)    # 半精度，6~7位有效数字
bf16 = torch.tensor(1.0, dtype=torch.bfloat16)   # BF16，更大动态范围
f32  = torch.tensor(1.0, dtype=torch.float32)    # 单精度（训练默认）
f64  = torch.tensor(1.0, dtype=torch.float64)    # 双精度（科学计算）

# 布尔类型
b = torch.tensor(True, dtype=torch.bool)

# 查看dtype
print(f"i8  dtype: {i8.dtype},  大小: {i8.element_size()} 字节")
print(f"f16 dtype: {f16.dtype}, 大小: {f16.element_size()} 字节")
print(f"f32 dtype: {f32.dtype}, 大小: {f32.element_size()} 字节")
print(f"f64 dtype: {f64.dtype}, 大小: {f64.element_size()} 字节")
```

**输出：**
```
i8  dtype: torch.int8,    大小: 1 字节
f16 dtype: torch.float16, 大小: 2 字节
f32 dtype: torch.float32, 大小: 4 字节
f64 dtype: torch.float64, 大小: 8 字节
```

### 各类型详细对比

```python
import torch

# 构造比较表格
dtypes_info = [
    ("int8",    torch.int8,    "整数量化后的权重/激活值"),
    ("uint8",   torch.uint8,   "图像像素值 [0, 255]"),
    ("int32",   torch.int32,   "类别标签、索引"),
    ("int64",   torch.int64,   "长整数索引（默认整数）"),
    ("float16", torch.float16, "混合精度训练（旧GPU）"),
    ("bfloat16",torch.bfloat16,"混合精度训练（Ampere+ GPU/TPU）"),
    ("float32", torch.float32, "训练和推理标准精度"),
    ("float64", torch.float64, "科学计算/数值验证"),
    ("bool",    torch.bool,    "掩码（Mask）、注意力掩码"),
]

print(f"{'dtype':<12} {'字节':>4} {'常见用途'}")
print("-" * 55)
for name, dtype, usage in dtypes_info:
    t = torch.tensor(1, dtype=dtype)
    print(f"{name:<12} {t.element_size():>4} 字节  {usage}")
```

**输出：**
```
dtype        字节 常见用途
-------------------------------------------------------
int8            1 字节  整数量化后的权重/激活值
uint8           1 字节  图像像素值 [0, 255]
int32           4 字节  类别标签、索引
int64           8 字节  长整数索引（默认整数）
float16         2 字节  混合精度训练（旧GPU）
bfloat16        2 字节  混合精度训练（Ampere+ GPU/TPU）
float32         4 字节  训练和推理标准精度
float64         8 字节  科学计算/数值验证
bool            1 字节  掩码（Mask）、注意力掩码
```

### dtype转换与精度影响

```python
import torch

# 类型转换
x = torch.tensor([1.5, 2.7, 3.9], dtype=torch.float32)
print(f"float32: {x}")

x_int = x.to(torch.int32)
print(f"int32:   {x_int}")   # tensor([1, 2, 3])，截断！

x_f16 = x.to(torch.float16)
print(f"float16: {x_f16}")   # tensor([1.5000, 2.6992, 3.8984])，精度损失

# 精度对比：float32 vs float16 vs bfloat16
val = 0.12345678901234567890

f32  = torch.tensor(val, dtype=torch.float32)
f16  = torch.tensor(val, dtype=torch.float16)
bf16 = torch.tensor(val, dtype=torch.bfloat16)

print(f"原始值:    {val}")
print(f"float32:   {f32.item():.20f}")
print(f"float16:   {f16.item():.20f}")
print(f"bfloat16:  {bf16.item():.20f}")
```

**输出：**
```
原始值:    0.12345678901234568
float32:   0.12345679104328155518
float16:   0.12347412109375000000
bfloat16:  0.12304687500000000000
```

### 图像数据的典型dtype工作流

```python
import torch

# 模拟一批图像：batch=2, channel=3, height=4, width=4
# 像素值 [0, 255]，uint8
raw_images = torch.randint(0, 256, (2, 3, 4, 4), dtype=torch.uint8)
print(f"原始图像: dtype={raw_images.dtype}, shape={raw_images.shape}")
print(f"内存占用: {raw_images.nbytes} 字节")

# 步骤1：转为float32并归一化
images_f32 = raw_images.float() / 255.0   # .float() 等同于 .to(torch.float32)
print(f"\n归一化后: dtype={images_f32.dtype}, 值域=[{images_f32.min():.2f}, {images_f32.max():.2f}]")
print(f"内存占用: {images_f32.nbytes} 字节 (×4倍)")

# 步骤2：转为float16用于混合精度训练
images_f16 = images_f32.half()   # .half() 等同于 .to(torch.float16)
print(f"\n混合精度: dtype={images_f16.dtype}")
print(f"内存占用: {images_f16.nbytes} 字节 (×2倍于float32)")
```

### 标签与掩码的dtype

```python
import torch

# 分类标签：int64（CrossEntropyLoss要求）
batch_size  = 4
num_classes = 10
labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.int64)
print(f"标签: {labels}")          # tensor([3, 7, 1, 9])
print(f"dtype: {labels.dtype}")   # torch.int64

# 注意力掩码：bool（Transformer中常用）
seq_len = 5
# 模拟padding掩码（True=有效token，False=padding）
mask = torch.tensor([True, True, True, False, False], dtype=torch.bool)
print(f"\n注意力掩码: {mask}")
print(f"dtype: {mask.dtype}")

# 掩码的整数等价
print(f"掩码转int: {mask.int()}")   # tensor([1, 1, 1, 0, 0])

# 用掩码选择元素
scores = torch.tensor([0.9, 0.7, 0.8, 0.1, 0.2])
valid_scores = scores[mask]
print(f"有效分数: {valid_scores}")   # tensor([0.9000, 0.7000, 0.8000])
```

### 混合精度训练简介

```python
import torch

# 自动混合精度（AMP）：自动在f32和f16之间切换
# 好处：GPU内存减半，速度提升1.5-3x，精度几乎不损失

# 示例（概念演示，需要真实模型和数据）
device = "cuda" if torch.cuda.is_available() else "cpu"

# 创建GradScaler防止f16下的梯度下溢
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 训练循环片段
# for images, labels in dataloader:
#     optimizer.zero_grad()
#
#     with autocast():   # 此块内自动使用float16
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#     scaler.scale(loss).backward()   # 缩放梯度防止下溢
#     scaler.step(optimizer)
#     scaler.update()

print("混合精度训练框架代码示例展示完毕")
print(f"当前设备: {device}")
```

### dtype选择建议

```python
# dtype选择决策树（以注释形式展示）

def choose_dtype(scenario: str) -> str:
    """根据使用场景推荐dtype。"""
    recommendations = {
        "模型训练（标准）":     "float32  — 最稳定，精度够用",
        "模型训练（混合精度）":  "float16/bfloat16 + float32 — 速度与精度平衡",
        "模型推理（生产部署）":  "float16 或 int8 — 追求速度和内存",
        "科学计算/数值验证":    "float64 — 需要高精度",
        "图像像素存储":         "uint8 — 节省内存",
        "分类标签":             "int64 — PyTorch损失函数要求",
        "量化模型权重":         "int8 — 极致压缩",
        "注意力掩码":           "bool — 语义清晰，节省内存",
    }
    return recommendations.get(scenario, "float32（默认安全选择）")

for scenario in [
    "模型训练（标准）", "模型推理（生产部署）",
    "图像像素存储", "分类标签", "注意力掩码"
]:
    print(f"{scenario:<20} → {choose_dtype(scenario)}")
```

---

## 练习题

### 基础题

**练习1**（基础）：数据类型识别与转换

下面的代码片段来自一个图像分类项目。请判断每个变量的类型，并修复类型错误。

```python
# 以下代码有3处类型问题，请找出并修复
num_classes = "10"              # 问题1
learning_rate = "0.001"         # 问题2
is_training = 1                 # 问题3

# 这行代码应输出 "类别数: 10, 学习率: 0.001"
# 请用正确类型完成计算
total_params = num_classes * 512
scaled_lr = learning_rate * 0.1
print(f"每类参数: {total_params}, 缩小后学习率: {scaled_lr}")
```

**练习2**（基础）：运算符综合练习

编写一个函数 `describe_number(n)`，接收一个整数 `n`，打印以下信息：
- 是否为正数/负数/零
- 是否为偶数
- 是否在深度学习常用批大小范围内（2的幂次，且在 [8, 512] 之间）
- 其二进制表示

示例输出（n=32）：
```
数字: 32
正负: 正数
奇偶: 偶数
批大小合适: True
二进制: 0b100000
```

---

### 中级题

**练习3**（中级）：字符串格式化日志系统

实现一个训练日志格式化函数 `format_log(epoch, total_epochs, loss, accuracy, lr)`，要求：
- epoch编号右对齐，总宽度与total_epochs位数相同（如总共100轮，则"  1/100"）
- loss保留6位小数，科学计数法
- accuracy以百分比显示，保留2位小数
- lr以科学计数法显示
- 每次调用在同一行更新（使用`\r`）

示例输出：
```
Epoch   3/100 | Loss: 2.345678e-01 | Acc: 45.67% | LR: 1.00e-03
```

**练习4**（中级）：位运算实现数据增强配置

实现一个数据增强配置系统：
1. 定义增强标志位常量（水平翻转、垂直翻转、随机裁剪、颜色抖动、随机旋转），每个占一个bit位
2. 实现函数 `get_augmentation_config(flags)` 返回启用的增强列表
3. 实现函数 `add_augmentation(current_flags, new_flag)` 和 `remove_augmentation(current_flags, flag)`
4. 验证：创建"训练集增强配置"（启用所有）和"验证集增强配置"（无增强），打印各自启用的增强

---

### 进阶题

**练习5**（进阶）：PyTorch dtype内存计算器

实现一个函数 `memory_calculator(shape, dtype_name, num_tensors=1)`，功能：
1. 接受张量形状（tuple）、dtype名称（字符串，如"float32"）和张量数量
2. 计算单个张量的元素数量、每元素字节数、总内存（字节、KB、MB、GB）
3. 计算 `num_tensors` 个同类张量的总内存
4. 给出建议：若单张量超过1GB打印警告，若总内存超过可用显存（假设8GB）打印警告

扩展：计算将ResNet-50（约25M参数）分别用float32、float16、int8存储时的内存占用。

---

## 练习答案

### 答案1：数据类型识别与转换

```python
# 修复后的代码
num_classes = 10              # 修复1: str -> int
learning_rate = 0.001         # 修复2: str -> float
is_training = True            # 修复3: int -> bool（语义更清晰）

# 现在可以正确计算
total_params = num_classes * 512
scaled_lr = learning_rate * 0.1

print(f"每类参数: {total_params}, 缩小后学习率: {scaled_lr}")
# 每类参数: 5120, 缩小后学习率: 0.0001

# 类型验证
print(type(num_classes))    # <class 'int'>
print(type(learning_rate))  # <class 'float'>
print(type(is_training))    # <class 'bool'>
```

**问题分析：**
- `"10" * 512` 会产生字符串重复（`"10" * 512 = "1010101010..."512次`），而非整数乘法
- `"0.001" * 0.1` 会抛出 `TypeError: can't multiply sequence by non-int`
- `is_training = 1` 虽然功能上可工作（bool是int子类），但语义不清晰

---

### 答案2：运算符综合练习

```python
def describe_number(n):
    """打印整数的各种属性。"""
    print(f"数字: {n}")

    # 正负判断
    if n > 0:
        sign = "正数"
    elif n < 0:
        sign = "负数"
    else:
        sign = "零"
    print(f"正负: {sign}")

    # 奇偶判断（位运算）
    parity = "偶数" if (n & 1) == 0 else "奇数"
    print(f"奇偶: {parity}")

    # 检查是否为2的幂次且在[8, 512]范围内
    # 2的幂次特性：n & (n-1) == 0（且n>0）
    is_power_of_2 = n > 0 and (n & (n - 1)) == 0
    is_valid_batch = is_power_of_2 and (8 <= n <= 512)
    print(f"批大小合适: {is_valid_batch}")

    # 二进制表示
    print(f"二进制: {bin(n)}")


# 测试
describe_number(32)
print()
describe_number(-7)
print()
describe_number(100)
```

**输出：**
```
数字: 32
正负: 正数
奇偶: 偶数
批大小合适: True
二进制: 0b100000

数字: -7
正负: 负数
奇偶: 奇数
批大小合适: False
二进制: -0b111

数字: 100
正负: 正数
奇偶: 偶数
批大小合适: False
二进制: 0b1100100
```

---

### 答案3：字符串格式化日志系统

```python
import sys

def format_log(epoch, total_epochs, loss, accuracy, lr, flush=True):
    """
    格式化训练日志，在同一行原地更新。

    参数：
        epoch        (int): 当前轮次（从1开始）
        total_epochs (int): 总轮次
        loss       (float): 当前损失值
        accuracy   (float): 当前精度（0~1之间的小数）
        lr         (float): 当前学习率
        flush       (bool): 是否立即刷新输出缓冲区
    """
    # 计算epoch编号所需宽度
    width = len(str(total_epochs))

    log_str = (
        f"\rEpoch {epoch:{width}d}/{total_epochs} | "
        f"Loss: {loss:.6e} | "
        f"Acc: {accuracy:.2%} | "
        f"LR: {lr:.2e}"
    )
    print(log_str, end="", flush=flush)

    # 最后一个epoch换行
    if epoch == total_epochs:
        print()


# 测试：模拟训练过程
import time
import math

total_epochs = 100
for epoch in range(1, total_epochs + 1):
    # 模拟损失下降和精度提升
    loss     = 2.0 * math.exp(-epoch * 0.05)
    accuracy = 1.0 - loss / 2.0
    lr       = 0.001 * (0.95 ** (epoch // 10))

    format_log(epoch, total_epochs, loss, accuracy, lr)
    time.sleep(0.02)   # 模拟计算耗时
```

**关键格式化说明：**
- `{epoch:{width}d}` — 动态宽度整数格式化，`width`是变量
- `{loss:.6e}` — 科学计数法6位小数
- `{accuracy:.2%}` — 百分比自动乘100并格式化
- `{lr:.2e}` — 科学计数法2位小数
- `\r` — 回车符（不换行，回到行首），实现原地刷新

---

### 答案4：位运算实现数据增强配置

```python
# 增强标志位常量
FLIP_LR      = 1 << 0   # 0b00001 = 1
FLIP_UD      = 1 << 1   # 0b00010 = 2
RANDOM_CROP  = 1 << 2   # 0b00100 = 4
COLOR_JITTER = 1 << 3   # 0b01000 = 8
RANDOM_ROT   = 1 << 4   # 0b10000 = 16

# 标志位到名称的映射
FLAG_NAMES = {
    FLIP_LR:      "水平翻转",
    FLIP_UD:      "垂直翻转",
    RANDOM_CROP:  "随机裁剪",
    COLOR_JITTER: "颜色抖动",
    RANDOM_ROT:   "随机旋转",
}

def get_augmentation_config(flags):
    """返回启用的增强名称列表。"""
    enabled = []
    for flag, name in FLAG_NAMES.items():
        if flags & flag:
            enabled.append(name)
    return enabled

def add_augmentation(current_flags, new_flag):
    """添加增强（按位或）。"""
    return current_flags | new_flag

def remove_augmentation(current_flags, flag):
    """移除增强（按位与非）。"""
    return current_flags & ~flag

def toggle_augmentation(current_flags, flag):
    """切换增强状态（按位异或）。"""
    return current_flags ^ flag


# 创建配置
ALL_FLAGS   = FLIP_LR | FLIP_UD | RANDOM_CROP | COLOR_JITTER | RANDOM_ROT
NO_FLAGS    = 0

# 验证
train_config = ALL_FLAGS
val_config   = NO_FLAGS

print(f"训练集增强 (flags={train_config:05b}):")
for aug in get_augmentation_config(train_config):
    print(f"  + {aug}")

print(f"\n验证集增强 (flags={val_config:05b}):")
enabled = get_augmentation_config(val_config)
print(f"  {enabled if enabled else '（无增强）'}")

# 动态修改：验证集加入随机裁剪
val_config = add_augmentation(val_config, RANDOM_CROP)
print(f"\n修改后验证集 (flags={val_config:05b}):")
for aug in get_augmentation_config(val_config):
    print(f"  + {aug}")

# 训练集移除垂直翻转
train_config = remove_augmentation(train_config, FLIP_UD)
print(f"\n修改后训练集 (flags={train_config:05b}):")
for aug in get_augmentation_config(train_config):
    print(f"  + {aug}")
```

**输出：**
```
训练集增强 (flags=11111):
  + 水平翻转
  + 垂直翻转
  + 随机裁剪
  + 颜色抖动
  + 随机旋转

验证集增强 (flags=00000):
  （无增强）

修改后验证集 (flags=00100):
  + 随机裁剪

修改后训练集 (flags=11101):
  + 水平翻转
  + 随机裁剪
  + 颜色抖动
  + 随机旋转
```

---

### 答案5：PyTorch dtype内存计算器

```python
import torch
from math import prod

# dtype名称到torch.dtype的映射
DTYPE_MAP = {
    "uint8":    torch.uint8,
    "int8":     torch.int8,
    "int16":    torch.int16,
    "int32":    torch.int32,
    "int64":    torch.int64,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
    "float32":  torch.float32,
    "float64":  torch.float64,
    "bool":     torch.bool,
}

def memory_calculator(shape, dtype_name, num_tensors=1):
    """
    计算张量内存占用。

    参数：
        shape      (tuple): 张量形状，如 (32, 3, 224, 224)
        dtype_name  (str):  dtype名称，如 "float32"
        num_tensors  (int): 张量数量

    返回：
        dict: 包含各种内存统计信息
    """
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"未知dtype: {dtype_name}。可用: {list(DTYPE_MAP.keys())}")

    dtype = DTYPE_MAP[dtype_name]

    # 创建零张量获取元素大小（不占实际内存）
    dummy = torch.empty(1, dtype=dtype)
    bytes_per_element = dummy.element_size()

    num_elements  = prod(shape)
    bytes_single  = num_elements * bytes_per_element
    bytes_total   = bytes_single * num_tensors

    def fmt_bytes(b):
        if b < 1024:
            return f"{b} B"
        elif b < 1024**2:
            return f"{b/1024:.2f} KB"
        elif b < 1024**3:
            return f"{b/1024**2:.2f} MB"
        else:
            return f"{b/1024**3:.4f} GB"

    result = {
        "shape":             shape,
        "dtype":             dtype_name,
        "num_elements":      num_elements,
        "bytes_per_element": bytes_per_element,
        "bytes_single":      bytes_single,
        "bytes_total":       bytes_total,
        "size_single_fmt":   fmt_bytes(bytes_single),
        "size_total_fmt":    fmt_bytes(bytes_total),
    }

    # 打印报告
    print(f"\n{'='*50}")
    print(f"张量形状:    {shape}")
    print(f"dtype:       {dtype_name} ({bytes_per_element} 字节/元素)")
    print(f"元素数量:    {num_elements:,}")
    print(f"单张量大小:  {result['size_single_fmt']}")
    if num_tensors > 1:
        print(f"×{num_tensors} 张量总计: {result['size_total_fmt']}")

    # 警告检查
    GPU_MEMORY_LIMIT = 8 * 1024**3   # 8 GB
    if bytes_single > 1024**3:
        print(f"⚠ 警告：单张量超过 1 GB！")
    if bytes_total > GPU_MEMORY_LIMIT:
        print(f"⚠ 警告：总内存 {result['size_total_fmt']} 超过 8 GB 显存限制！")

    return result


# 测试1：标准训练批次
print("测试1：图像批次")
memory_calculator((32, 3, 224, 224), "float32")
memory_calculator((32, 3, 224, 224), "float16")
memory_calculator((32, 3, 224, 224), "uint8")

# 测试2：ResNet-50参数存储
print("\n\n测试2：ResNet-50 参数内存（~25M参数）")
resnet50_params = 25_000_000   # 约25M

for dtype_name in ["float32", "float16", "int8"]:
    dtype  = DTYPE_MAP[dtype_name]
    dummy  = torch.empty(1, dtype=dtype)
    bpe    = dummy.element_size()
    total  = resnet50_params * bpe
    mb     = total / 1024**2
    print(f"ResNet-50 ({dtype_name:>8}): {mb:.1f} MB")
```

**输出：**
```
测试1：图像批次

==================================================
张量形状:    (32, 3, 224, 224)
dtype:       float32 (4 字节/元素)
元素数量:    4,816,896
单张量大小:  18.38 MB

==================================================
张量形状:    (32, 3, 224, 224)
dtype:       float16 (2 字节/元素)
元素数量:    4,816,896
单张量大小:  9.19 MB

==================================================
张量形状:    (32, 3, 224, 224)
dtype:       uint8 (1 字节/元素)
元素数量:    4,816,896
单张量大小:  4.59 MB


测试2：ResNet-50 参数内存（~25M参数）
ResNet-50 ( float32): 95.4 MB
ResNet-50 ( float16): 47.7 MB
ResNet-50 (    int8): 23.8 MB
```

---

*本章完。下一章将介绍Python的控制流（条件语句、循环、推导式），并展示其在数据预处理和训练循环中的实际应用。*
