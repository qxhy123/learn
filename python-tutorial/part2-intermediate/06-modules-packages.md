# 第6章：模块与包

> **适用读者**：已掌握 Python 基础语法的学习者，希望了解如何组织代码、使用标准库以及管理第三方依赖。

---

## 学习目标

完成本章后，你将能够：

1. 理解模块的概念，熟练使用 `import`、`from...import`、`as` 等导入方式
2. 掌握 Python 标准库中 `os`、`sys`、`math`、`random`、`datetime` 等常用模块
3. 理解包的结构，能够创建并使用含 `__init__.py` 的自定义包
4. 搭建和管理 Python 虚拟环境（venv 与 conda）
5. 使用 `pip` 安装、升级、卸载第三方包，并理解 `requirements.txt` 的作用

---

## 6.1 模块的概念与导入方式

### 6.1.1 什么是模块

在 Python 中，**模块（Module）**就是一个以 `.py` 为扩展名的文件。模块中可以包含函数、类、变量以及可执行代码。通过模块，我们可以把功能相关的代码组织在一起，方便复用和维护。

```
项目/
├── main.py          # 主程序
├── utils.py         # 工具模块
└── model.py         # 模型模块
```

假设 `utils.py` 内容如下：

```python
# utils.py

PI = 3.14159

def circle_area(radius):
    """计算圆的面积"""
    return PI * radius ** 2

def rectangle_area(width, height):
    """计算矩形的面积"""
    return width * height

class Vector2D:
    """二维向量"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector2D({self.x}, {self.y})"

    def magnitude(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5
```

### 6.1.2 import 语句

最基本的导入方式是直接 `import` 模块名。使用模块内的内容时，需要加上模块名作为前缀：

```python
# main.py
import utils

# 访问模块中的变量
print(utils.PI)              # 3.14159

# 调用模块中的函数
area = utils.circle_area(5)
print(f"半径为5的圆面积：{area:.2f}")  # 78.54

# 使用模块中的类
v = utils.Vector2D(3, 4)
print(v.magnitude())         # 5.0
```

**优点**：命名空间清晰，不会与本地变量冲突。
**缺点**：每次访问都需要写模块名前缀，代码较冗长。

### 6.1.3 from...import 语句

使用 `from...import` 可以直接从模块中导入特定的名称，使用时无需模块名前缀：

```python
# 导入特定函数和变量
from utils import PI, circle_area, Vector2D

print(PI)                    # 3.14159
area = circle_area(5)        # 直接调用，无需前缀
print(area)

v = Vector2D(3, 4)
print(v.magnitude())
```

**导入全部内容**（不推荐）：

```python
from utils import *          # 导入所有公开名称

# 可以直接使用，但容易造成命名污染
print(PI)
print(circle_area(3))
```

> **注意**：`from module import *` 会导入模块中所有不以下划线开头的名称，可能与本地变量发生冲突，不推荐在生产代码中使用。若模块定义了 `__all__`，则只导入 `__all__` 中列出的名称。

### 6.1.4 as 别名

当模块名或函数名过长，或存在命名冲突时，可以使用 `as` 创建别名：

```python
# 为模块创建别名
import numpy as np          # 深度学习中极常见的约定
import pandas as pd
import matplotlib.pyplot as plt

# 为函数创建别名
from utils import circle_area as ca
from utils import rectangle_area as ra

print(ca(5))                 # 78.53975
print(ra(3, 4))              # 12
```

深度学习领域常见的导入约定：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
```

### 6.1.5 模块的搜索路径

当执行 `import` 时，Python 按以下顺序搜索模块：

1. 当前目录
2. 环境变量 `PYTHONPATH` 中指定的目录
3. Python 安装目录下的标准库路径
4. 第三方库的安装路径（`site-packages`）

可以通过 `sys.path` 查看和修改搜索路径：

```python
import sys

# 查看当前搜索路径
for path in sys.path:
    print(path)

# 动态添加路径（临时生效）
sys.path.append('/path/to/my/modules')
```

### 6.1.6 `__name__` 与模块独立运行

每个模块都有一个内置变量 `__name__`。当模块**直接运行**时，`__name__` 为 `'__main__'`；当模块**被导入**时，`__name__` 为模块的文件名（不含 `.py`）。

```python
# utils.py
def circle_area(radius):
    return 3.14159 * radius ** 2

# 仅在直接运行时执行，被导入时不执行
if __name__ == '__main__':
    print("模块独立运行中...")
    print(f"测试：半径为3的圆面积 = {circle_area(3):.2f}")
```

```bash
# 直接运行 utils.py
$ python utils.py
模块独立运行中...
测试：半径为3的圆面积 = 28.27

# 被其他模块导入时，if __name__ == '__main__' 块不会执行
$ python main.py
（只执行 main.py 中的代码）
```

---

## 6.2 标准库常用模块

Python 内置了大量功能强大的标准库，"自带电池"（Batteries Included）是 Python 的设计哲学之一。

### 6.2.1 os 模块：操作系统接口

`os` 模块提供与操作系统交互的接口，包括文件系统操作、环境变量、进程管理等：

```python
import os

# ---- 路径操作 ----
cwd = os.getcwd()                          # 获取当前工作目录
print(f"当前目录：{cwd}")

os.chdir('/tmp')                           # 切换工作目录

# 路径拼接（推荐使用 os.path.join，自动处理不同系统的分隔符）
data_dir = os.path.join('data', 'train', 'images')
print(data_dir)                            # data/train/images（Linux/Mac）

# 检查路径是否存在
print(os.path.exists('data'))             # True 或 False
print(os.path.isfile('README.md'))        # 是否是文件
print(os.path.isdir('data'))              # 是否是目录

# 获取文件名和目录
full_path = '/home/user/data/train.csv'
print(os.path.dirname(full_path))         # /home/user/data
print(os.path.basename(full_path))        # train.csv
name, ext = os.path.splitext('train.csv')
print(name, ext)                          # train  .csv

# ---- 目录操作 ----
os.makedirs('output/checkpoints', exist_ok=True)   # 递归创建目录

# 遍历目录树
for root, dirs, files in os.walk('data'):
    print(f"目录：{root}")
    for file in files:
        print(f"  文件：{file}")

# 列出目录内容
entries = os.listdir('.')
print(entries)

# ---- 环境变量 ----
home = os.environ.get('HOME', '/tmp')     # 获取环境变量，带默认值
print(f"Home 目录：{home}")

os.environ['MY_CONFIG'] = 'production'    # 设置环境变量

# ---- 执行系统命令 ----
ret = os.system('echo Hello from OS')    # 返回退出码
```

深度学习中 `os` 的典型用法：

```python
import os

def setup_experiment(exp_name):
    """创建实验所需的目录结构"""
    base_dir = os.path.join('experiments', exp_name)
    dirs = ['checkpoints', 'logs', 'results', 'figures']
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    print(f"实验目录已创建：{base_dir}")
    return base_dir

setup_experiment('resnet50_cifar10')
```

### 6.2.2 sys 模块：系统参数与解释器接口

```python
import sys

# Python 版本信息
print(sys.version)           # 3.11.4 (main, ...) [GCC ...]
print(sys.version_info)      # sys.version_info(major=3, minor=11, ...)

# 运行平台
print(sys.platform)          # 'linux', 'darwin', 'win32'

# 命令行参数
# 运行：python train.py --lr 0.01 --epochs 100
print(sys.argv)              # ['train.py', '--lr', '0.01', '--epochs', '100']
script_name = sys.argv[0]
args = sys.argv[1:]

# 模块搜索路径
print(sys.path)

# 标准输入/输出/错误流
sys.stdout.write("标准输出\n")
sys.stderr.write("错误输出\n")

# 退出程序
# sys.exit(0)                # 0 表示正常退出，非0表示异常

# 已导入的模块
print(list(sys.modules.keys())[:10])

# 获取对象占用内存（字节）
import numpy as np
arr = np.zeros((1000, 1000))
print(f"数组占用：{sys.getsizeof(arr)} 字节")
```

### 6.2.3 math 模块：数学函数

```python
import math

# 常用常数
print(math.pi)               # 3.141592653589793
print(math.e)                # 2.718281828459045
print(math.inf)              # inf（正无穷）
print(math.nan)              # nan（非数字）

# 基本运算
print(math.sqrt(16))         # 4.0
print(math.pow(2, 10))       # 1024.0
print(math.fabs(-3.14))      # 3.14（绝对值）

# 取整
print(math.floor(3.7))       # 3（向下取整）
print(math.ceil(3.2))        # 4（向上取整）
print(math.trunc(3.9))       # 3（截断小数部分）

# 对数与指数
print(math.log(math.e))      # 1.0（自然对数）
print(math.log(100, 10))     # 2.0（以10为底的对数）
print(math.log2(8))          # 3.0
print(math.exp(1))           # 2.718...

# 三角函数（参数为弧度）
print(math.sin(math.pi / 2)) # 1.0
print(math.cos(0))           # 1.0
print(math.degrees(math.pi)) # 180.0（弧度转角度）
print(math.radians(180))     # 3.14...（角度转弧度）

# 组合与排列
print(math.factorial(5))     # 120（5!）
print(math.comb(10, 3))      # 120（C(10,3)，组合数）
print(math.perm(10, 3))      # 720（A(10,3)，排列数）

# 特殊判断
print(math.isnan(float('nan')))   # True
print(math.isinf(float('inf')))   # True
print(math.isclose(0.1 + 0.2, 0.3, rel_tol=1e-9))  # True（浮点数比较）
```

### 6.2.4 random 模块：随机数生成

```python
import random

# 设置随机种子（保证结果可复现，深度学习中非常重要）
random.seed(42)

# 生成随机浮点数 [0.0, 1.0)
print(random.random())       # 0.6394267984578837

# 生成指定范围内的随机浮点数 [a, b]
print(random.uniform(0, 10)) # 2.0584494295802446

# 生成随机整数 [a, b]（含两端）
print(random.randint(1, 6))  # 模拟骰子

# 从序列中随机选择
colors = ['red', 'green', 'blue', 'yellow']
print(random.choice(colors))             # 随机选一个
print(random.choices(colors, k=3))      # 有放回地随机选3个
print(random.sample(colors, k=2))       # 无放回地随机选2个

# 随机打乱列表（原地修改）
deck = list(range(1, 53))
random.shuffle(deck)
print(deck[:5])

# 正态分布
mean, std = 0, 1
samples = [random.gauss(mean, std) for _ in range(5)]
print([f"{x:.3f}" for x in samples])

# 深度学习中的随机种子设置
def set_all_seeds(seed=42):
    """设置所有随机种子以保证可复现性"""
    import random
    import numpy as np
    # import torch
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    print(f"所有随机种子已设置为 {seed}")

set_all_seeds(42)
```

### 6.2.5 datetime 模块：日期与时间

```python
from datetime import datetime, date, time, timedelta
import time as time_module

# ---- 获取当前时间 ----
now = datetime.now()
print(now)                    # 2024-01-15 14:30:25.123456
print(now.year)               # 2024
print(now.month)              # 1
print(now.day)                # 15
print(now.hour)               # 14
print(now.minute)             # 30

# ---- 创建特定日期时间 ----
birthday = datetime(1990, 6, 15, 8, 30, 0)
print(birthday)               # 1990-06-15 08:30:00

# ---- 格式化输出 ----
print(now.strftime("%Y-%m-%d %H:%M:%S"))   # 2024-01-15 14:30:25
print(now.strftime("%Y年%m月%d日"))         # 2024年01月15日

# ---- 从字符串解析 ----
dt = datetime.strptime("2024-01-15 14:30:25", "%Y-%m-%d %H:%M:%S")
print(dt)

# ---- 时间差计算 ----
delta = timedelta(days=7, hours=3, minutes=30)
next_week = now + delta
print(f"一周后：{next_week.strftime('%Y-%m-%d')}")

# 计算两个日期之间的差
start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)
diff = end - start
print(f"2024年共 {diff.days} 天")

# ---- 时间戳 ----
timestamp = now.timestamp()   # Unix 时间戳（秒）
print(f"时间戳：{timestamp}")
back = datetime.fromtimestamp(timestamp)
print(f"还原：{back}")

# ---- 深度学习中的用途：记录训练时间 ----
def train_one_epoch(epoch):
    start_time = datetime.now()
    # ... 训练代码 ...
    time_module.sleep(0.1)  # 模拟训练耗时
    elapsed = datetime.now() - start_time
    print(f"Epoch {epoch} 完成，耗时：{elapsed.total_seconds():.2f}秒")
    return elapsed

train_one_epoch(1)
```

---

## 6.3 包的结构与 `__init__.py`

### 6.3.1 什么是包

**包（Package）**是包含 `__init__.py` 文件的目录，用于将多个相关模块组织在一起。包可以嵌套，形成层级结构。

```
mylib/                    # 包（顶层）
├── __init__.py           # 包的初始化文件（必须）
├── math_utils.py         # 子模块
├── string_utils.py       # 子模块
└── ml/                   # 子包
    ├── __init__.py       # 子包的初始化文件
    ├── layers.py         # 子模块
    └── losses.py         # 子模块
```

### 6.3.2 创建一个包

**步骤1**：创建目录和 `__init__.py`

```python
# mylib/__init__.py

# __init__.py 在包被导入时自动执行
# 可以在这里定义包级别的变量、导入常用内容、设置 __all__

__version__ = '1.0.0'
__author__ = 'Your Name'

# 从子模块导入，方便用户直接从包访问
from .math_utils import circle_area, Vector2D
from .string_utils import to_snake_case

# 控制 from mylib import * 的行为
__all__ = ['circle_area', 'Vector2D', 'to_snake_case']

print(f"mylib {__version__} 已加载")  # 仅用于演示，生产中通常不打印
```

```python
# mylib/math_utils.py

import math

def circle_area(radius):
    """计算圆的面积"""
    if radius < 0:
        raise ValueError(f"半径不能为负数，得到：{radius}")
    return math.pi * radius ** 2

def sphere_volume(radius):
    """计算球的体积"""
    return (4 / 3) * math.pi * radius ** 3

class Vector2D:
    """二维向量类"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return f"Vector2D({self.x}, {self.y})"

    def magnitude(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("零向量无法归一化")
        return Vector2D(self.x / mag, self.y / mag)
```

```python
# mylib/string_utils.py

import re

def to_snake_case(name):
    """将驼峰命名转换为下划线命名"""
    # 在大写字母前插入下划线
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def to_camel_case(name):
    """将下划线命名转换为驼峰命名"""
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def truncate(text, max_length=50, suffix='...'):
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
```

```python
# mylib/ml/__init__.py

from .layers import LinearLayer, ReLULayer
from .losses import MSELoss, CrossEntropyLoss

__all__ = ['LinearLayer', 'ReLULayer', 'MSELoss', 'CrossEntropyLoss']
```

```python
# mylib/ml/layers.py

class LinearLayer:
    """简单线性层（演示用）"""
    def __init__(self, in_features, out_features):
        import random
        self.in_features = in_features
        self.out_features = out_features
        # 随机初始化权重
        self.weight = [[random.gauss(0, 0.01)
                        for _ in range(in_features)]
                       for _ in range(out_features)]

    def __repr__(self):
        return f"LinearLayer({self.in_features} -> {self.out_features})"

class ReLULayer:
    """ReLU 激活层"""
    def forward(self, x):
        return [max(0, v) for v in x]

    def __repr__(self):
        return "ReLU()"
```

```python
# mylib/ml/losses.py

def MSELoss(y_pred, y_true):
    """均方误差损失"""
    n = len(y_pred)
    return sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / n

def CrossEntropyLoss(y_pred, y_true):
    """交叉熵损失（简化版）"""
    import math
    return -sum(t * math.log(p + 1e-9)
                for p, t in zip(y_pred, y_true))
```

### 6.3.3 使用包

```python
# 方式1：导入包（使用 __init__.py 中导出的内容）
import mylib

area = mylib.circle_area(5)
print(area)

v = mylib.Vector2D(3, 4)
print(v.magnitude())

# 方式2：从包中导入特定内容
from mylib import circle_area, Vector2D
from mylib.math_utils import sphere_volume

print(sphere_volume(3))

# 方式3：导入子包
from mylib.ml import LinearLayer, MSELoss

layer = LinearLayer(128, 64)
print(layer)

# 方式4：完整路径导入
from mylib.ml.layers import ReLULayer

relu = ReLULayer()
print(relu)
```

### 6.3.4 相对导入与绝对导入

在包内部的模块之间互相引用时，可以使用**相对导入**：

```python
# mylib/ml/layers.py 中导入同包的其他模块

# 相对导入（推荐在包内部使用）
from . import losses                    # 导入同级的 losses 模块
from .losses import MSELoss             # 导入同级模块中的特定内容
from .. import math_utils               # 导入上级包的模块
from ..math_utils import circle_area    # 导入上级包模块中的内容

# 绝对导入（从顶级包开始）
from mylib.ml import losses
from mylib.math_utils import circle_area
```

> **建议**：在包内部使用相对导入，在应用代码中使用绝对导入，这样代码更清晰、不易出错。

---

## 6.4 虚拟环境（venv 与 conda）

### 6.4.1 为什么需要虚拟环境

不同项目可能依赖同一个库的不同版本，例如：

- 项目 A：PyTorch 1.x + Python 3.8
- 项目 B：PyTorch 2.x + Python 3.11

如果所有项目共用一套全局 Python 环境，版本冲突将不可避免。**虚拟环境**为每个项目提供独立、隔离的 Python 运行环境，彻底解决版本冲突问题。

### 6.4.2 使用 venv（Python 内置）

`venv` 是 Python 3.3+ 内置的虚拟环境工具，无需额外安装：

```bash
# ---- 创建虚拟环境 ----
# 在当前目录创建名为 .venv 的虚拟环境
python -m venv .venv

# 指定 Python 版本（需要该版本已安装）
python3.11 -m venv .venv

# ---- 激活虚拟环境 ----
# macOS / Linux
source .venv/bin/activate

# Windows (CMD)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 激活后，命令提示符前会显示环境名称，例如：
# (.venv) $

# ---- 验证激活成功 ----
which python         # 应显示 .venv 内的 python 路径
python --version

# ---- 安装依赖 ----
pip install numpy pandas torch

# ---- 退出虚拟环境 ----
deactivate

# ---- 删除虚拟环境 ----
rm -rf .venv         # macOS/Linux
```

**venv 创建的目录结构**：

```
.venv/
├── bin/             # 可执行文件（macOS/Linux）
│   ├── python
│   ├── python3
│   └── pip
├── include/
├── lib/
│   └── python3.11/
│       └── site-packages/   # 第三方包安装在这里
└── pyvenv.cfg
```

### 6.4.3 使用 conda

`conda` 是 Anaconda/Miniconda 提供的包管理和环境管理工具，特别适合科学计算和深度学习领域。与 `venv` 相比，conda 的优势在于：

- 可以管理 Python 本身的版本
- 支持非 Python 依赖（如 CUDA、cuDNN）
- 预编译的二进制包，安装更快更稳定

```bash
# ---- 查看现有环境 ----
conda env list
# 或
conda info --envs

# ---- 创建新环境 ----
# 创建名为 dl-env 的环境，指定 Python 版本
conda create -n dl-env python=3.11

# 创建时同时安装包
conda create -n dl-env python=3.11 numpy pandas

# 从 environment.yml 文件创建
conda env create -f environment.yml

# ---- 激活/退出环境 ----
conda activate dl-env
conda deactivate

# ---- 在环境中安装包 ----
conda install numpy pandas matplotlib
conda install pytorch torchvision -c pytorch     # 从指定 channel 安装
pip install some-package                         # conda 中也可以用 pip

# ---- 管理环境 ----
conda env export > environment.yml    # 导出环境配置
conda env remove -n dl-env            # 删除环境

# ---- 克隆环境 ----
conda create -n dl-env-copy --clone dl-env
```

**environment.yml 示例**（深度学习项目）：

```yaml
name: dl-project
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy=1.24
  - pandas=2.0
  - matplotlib=3.7
  - scikit-learn=1.3
  - pip:
    - torch==2.1.0
    - torchvision==0.16.0
    - tensorboard==2.14.0
    - tqdm==4.66.0
```

### 6.4.4 venv vs conda 的选择

| 特性 | venv | conda |
|------|------|-------|
| 是否内置 | 是（Python 3.3+） | 否（需安装 Anaconda/Miniconda） |
| Python 版本管理 | 否 | 是 |
| 非 Python 依赖 | 否 | 是（CUDA、cuDNN 等） |
| 包数量 | PyPI 全量 | conda-forge + PyPI |
| 磁盘占用 | 小 | 较大 |
| 适用场景 | 普通 Web/应用开发 | 数据科学、深度学习 |

---

## 6.5 包管理与 pip

### 6.5.1 pip 基础操作

`pip` 是 Python 的标准包管理工具，从 PyPI（Python Package Index）下载并安装第三方包：

```bash
# ---- 查看 pip 版本 ----
pip --version
pip3 --version

# ---- 安装包 ----
pip install numpy                        # 安装最新版本
pip install numpy==1.24.0               # 安装指定版本
pip install "numpy>=1.20,<2.0"          # 安装满足版本范围的版本
pip install numpy pandas matplotlib     # 一次安装多个包

# ---- 升级包 ----
pip install --upgrade numpy             # 升级到最新版本
pip install -U numpy pandas             # -U 是 --upgrade 的简写

# ---- 卸载包 ----
pip uninstall numpy                     # 卸载（会询问确认）
pip uninstall -y numpy                  # 不询问直接卸载

# ---- 查看已安装的包 ----
pip list                                # 列出所有已安装的包
pip list --outdated                     # 列出有更新的包
pip show numpy                          # 查看特定包的详细信息

# ---- 搜索包 ----
pip index versions numpy               # 查看可用版本

# ---- 使用国内镜像加速 ----
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple   # 清华镜像
pip install numpy -i https://mirrors.aliyun.com/pypi/simple/    # 阿里云镜像
```

**设置默认镜像源**（一劳永逸）：

```bash
# Linux/macOS
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 或手动编辑 ~/.pip/pip.conf（Linux/macOS）
# 或 %APPDATA%\pip\pip.ini（Windows）
```

```ini
# pip.conf 内容示例
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

### 6.5.2 requirements.txt

`requirements.txt` 是记录项目依赖的文本文件，是 Python 项目的标准规范：

```bash
# 生成 requirements.txt（记录当前环境所有包）
pip freeze > requirements.txt

# 根据 requirements.txt 安装依赖（部署时使用）
pip install -r requirements.txt
```

**requirements.txt 示例**（深度学习项目）：

```
# 核心框架
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# 数据处理
numpy==1.24.3
pandas==2.0.3
Pillow==10.0.1

# 可视化
matplotlib==3.7.3
seaborn==0.12.2
tensorboard==2.14.0

# 工具
tqdm==4.66.1
scikit-learn==1.3.0
pyyaml==6.0.1
```

**分层的 requirements 结构**（更灵活的方式）：

```
requirements/
├── base.txt          # 基础依赖（所有环境共用）
├── dev.txt           # 开发环境额外依赖
├── prod.txt          # 生产环境额外依赖
└── test.txt          # 测试专用依赖
```

```
# requirements/base.txt
torch>=2.0.0
numpy>=1.20.0
pandas>=1.5.0

# requirements/dev.txt
-r base.txt
jupyter
ipython
black
flake8
pytest

# requirements/test.txt
-r base.txt
pytest
pytest-cov
```

### 6.5.3 pip 的高级用法

```bash
# 安装本地包（开发模式，修改源码立即生效）
pip install -e .                         # 在包的根目录执行
pip install -e /path/to/package

# 从 Git 仓库安装
pip install git+https://github.com/user/repo.git
pip install git+https://github.com/user/repo.git@v1.0.0  # 指定 tag

# 下载但不安装（用于离线部署）
pip download numpy -d ./packages/
pip install --no-index --find-links=./packages/ numpy

# 查看依赖树
pip install pipdeptree
pipdeptree

# 检查依赖冲突
pip check
```

---

## 本章小结

| 知识点 | 核心要点 |
|--------|----------|
| 模块导入 | `import mod`、`from mod import func`、`import mod as alias` |
| `__name__` | 直接运行时为 `'__main__'`，被导入时为模块名 |
| os 模块 | 文件系统操作、路径拼接、环境变量 |
| sys 模块 | 命令行参数、搜索路径、Python 版本信息 |
| math 模块 | 数学常数（pi/e）、三角函数、对数、组合数 |
| random 模块 | 随机数生成、随机选择、随机打乱、设置种子 |
| datetime 模块 | 日期时间的创建、格式化、计算时间差 |
| 包结构 | 目录 + `__init__.py`，支持多层嵌套 |
| `__init__.py` | 包初始化、控制公开接口、便捷导入 |
| 相对导入 | `.module`（同级）、`..module`（上级） |
| venv | Python 内置虚拟环境工具，轻量简单 |
| conda | 跨平台环境管理，支持非 Python 依赖 |
| pip | 包安装、升级、卸载，国内镜像加速 |
| requirements.txt | 记录项目依赖，使用 `pip freeze` 生成 |

---

## 深度学习应用：PyTorch 项目的组织结构

一个规范的 PyTorch 深度学习项目通常按照以下目录结构组织，这种结构清晰地分离了数据、模型、训练逻辑和配置，便于团队协作和实验管理。

### 典型项目目录布局

```
image-classifier/               # 项目根目录
│
├── README.md                   # 项目说明
├── requirements.txt            # Python 依赖
├── environment.yml             # conda 环境配置
├── setup.py                    # 可选：将项目打包为可安装包
│
├── configs/                    # 配置文件目录
│   ├── default.yaml            # 默认配置
│   ├── resnet50.yaml           # ResNet50 专用配置
│   └── mobilenet.yaml          # MobileNet 专用配置
│
├── data/                       # 数据目录（通常加入 .gitignore）
│   ├── raw/                    # 原始数据
│   ├── processed/              # 预处理后的数据
│   └── splits/                 # 数据集划分（train/val/test）
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
├── src/                        # 核心源码包
│   ├── __init__.py
│   │
│   ├── datasets/               # 数据集相关
│   │   ├── __init__.py
│   │   ├── base_dataset.py     # 基类
│   │   ├── cifar10.py          # CIFAR-10 数据集
│   │   └── imagenet.py         # ImageNet 数据集
│   │
│   ├── models/                 # 模型定义
│   │   ├── __init__.py
│   │   ├── resnet.py           # ResNet 系列
│   │   ├── mobilenet.py        # MobileNet 系列
│   │   └── layers/             # 自定义层
│   │       ├── __init__.py
│   │       ├── attention.py    # 注意力机制
│   │       └── normalization.py
│   │
│   ├── losses/                 # 损失函数
│   │   ├── __init__.py
│   │   ├── cross_entropy.py
│   │   └── focal_loss.py       # Focal Loss
│   │
│   ├── optimizers/             # 优化器（自定义）
│   │   ├── __init__.py
│   │   └── warmup_scheduler.py
│   │
│   ├── trainers/               # 训练逻辑
│   │   ├── __init__.py
│   │   ├── base_trainer.py     # 训练基类
│   │   └── classifier_trainer.py
│   │
│   └── utils/                  # 工具函数
│       ├── __init__.py
│       ├── metrics.py          # 评估指标（accuracy, F1）
│       ├── visualization.py    # 可视化工具
│       ├── checkpoint.py       # 模型保存/加载
│       └── logger.py           # 日志工具
│
├── scripts/                    # 脚本文件
│   ├── train.py                # 训练入口
│   ├── evaluate.py             # 评估入口
│   ├── predict.py              # 推理入口
│   └── export_onnx.py          # 导出 ONNX
│
├── notebooks/                  # Jupyter Notebook（探索性分析）
│   ├── 01_data_exploration.ipynb
│   └── 02_model_visualization.ipynb
│
├── experiments/                # 实验记录（通常加入 .gitignore）
│   └── resnet50_cifar10_2024-01-15/
│       ├── checkpoints/        # 模型权重
│       │   ├── epoch_10.pth
│       │   └── best.pth
│       ├── logs/               # TensorBoard 日志
│       └── config.yaml         # 本次实验的配置快照
│
└── tests/                      # 单元测试
    ├── __init__.py
    ├── test_datasets.py
    ├── test_models.py
    └── test_losses.py
```

### 关键模块代码示例

**`src/utils/checkpoint.py`** — 模型检查点管理：

```python
# src/utils/checkpoint.py

import os
import torch
from datetime import datetime


def save_checkpoint(state, save_dir, filename='checkpoint.pth', is_best=False):
    """
    保存训练检查点。

    Args:
        state (dict): 包含模型权重、优化器状态、epoch 等信息
        save_dir (str): 保存目录
        filename (str): 文件名
        is_best (bool): 是否是目前最优模型
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(save_dir, 'best.pth')
        import shutil
        shutil.copyfile(filepath, best_path)
        print(f"[Checkpoint] 最优模型已保存至：{best_path}")

    print(f"[Checkpoint] 检查点已保存：{filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """
    加载训练检查点。

    Returns:
        dict: 包含 epoch 和其他元数据
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在：{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"[Checkpoint] 已加载检查点：{checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  最优验证准确率: {checkpoint.get('best_val_acc', 'N/A')}")

    return checkpoint
```

**`src/utils/logger.py`** — 统一日志管理：

```python
# src/utils/logger.py

import os
import sys
import logging
from datetime import datetime


def setup_logger(name, log_dir=None, level=logging.INFO):
    """
    创建并配置 Logger。

    Args:
        name (str): Logger 名称
        log_dir (str): 日志文件保存目录，None 则只输出到控制台
        level: 日志级别

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（可选）
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 使用示例
# logger = setup_logger('trainer', log_dir='experiments/run1/logs')
# logger.info("开始训练...")
# logger.warning("学习率过高，可能不稳定")
# logger.error("GPU 内存不足！")
```

**`scripts/train.py`** — 训练入口：

```python
# scripts/train.py

import sys
import os
import argparse

# 将项目根目录加入搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from src.utils.checkpoint import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='图像分类训练脚本')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--exp-name', type=str, default='experiment',
                        help='实验名称')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置实验目录
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join('experiments', f'{args.exp_name}_{timestamp}')

    # 初始化日志
    logger = setup_logger('train', log_dir=os.path.join(exp_dir, 'logs'))
    logger.info(f"实验目录：{exp_dir}")
    logger.info(f"配置文件：{args.config}")

    # --- 以下是训练主循环的骨架 ---
    # config = load_config(args.config)
    # model = build_model(config)
    # optimizer = build_optimizer(model, config)
    # train_loader, val_loader = build_dataloaders(config)
    #
    # for epoch in range(config.epochs):
    #     train_one_epoch(model, optimizer, train_loader, logger)
    #     val_acc = evaluate(model, val_loader)
    #     save_checkpoint({...}, exp_dir, is_best=(val_acc > best_acc))

    logger.info("训练完成！")


if __name__ == '__main__':
    main()
```

---

## 练习题

### 基础题

**练习 1**：创建一个 `geometry.py` 模块，包含以下内容：
- 常量 `PI = 3.14159`
- 函数 `circle_perimeter(radius)`：计算圆的周长
- 函数 `triangle_area(base, height)`：计算三角形面积
- 类 `Rectangle`：包含 `width`、`height` 属性，以及 `area()` 和 `perimeter()` 方法
- 在 `if __name__ == '__main__':` 块中添加测试代码

要求：在另一个文件中分别用三种方式导入并使用该模块。

---

**练习 2**：使用标准库完成以下任务：
1. 使用 `os` 模块在当前目录下创建如下目录结构：
   ```
   project/
   ├── data/
   ├── output/
   └── logs/
   ```
2. 使用 `random` 模块生成 20 个 [1, 100] 范围内的随机整数，统计其中偶数的个数
3. 使用 `datetime` 模块计算从今天起 100 天后是哪一天（格式：`YYYY年MM月DD日`）

---

### 进阶题

**练习 3**：创建一个名为 `statslib` 的包，结构如下：

```
statslib/
├── __init__.py
├── basic.py      # 包含 mean(), median(), mode()
└── advanced.py   # 包含 variance(), std_dev(), correlation()
```

要求：
- `__init__.py` 中从两个模块导出所有函数，并定义 `__version__ = '1.0.0'`
- `basic.py` 中的函数只使用 Python 内置功能（不得使用 numpy/statistics 模块）
- `advanced.py` 可以导入 `basic.py` 中的函数
- 编写测试，验证 `mean([1, 2, 3, 4, 5])` 返回 `3.0`，`variance([2, 4, 4, 4, 5, 5, 7, 9])` 返回 `4.0`

---

**练习 4**：编写一个命令行脚本 `file_stats.py`，使用 `sys.argv` 和 `os` 模块实现以下功能：

```bash
python file_stats.py <目录路径>
```

输出内容示例：
```
目录：/home/user/data
----------------------------------------
总文件数：42
总目录数：8
各类型文件统计：
  .py 文件：15 个，共 45,678 字节
  .txt 文件：10 个，共 12,345 字节
  .csv 文件：17 个，共 1,234,567 字节
----------------------------------------
最大文件：data/train.csv (1,200,000 字节)
最小文件：data/README.txt (128 字节)
```

要求：
- 使用 `os.walk` 递归遍历目录
- 使用 `os.path.getsize` 获取文件大小
- 正确处理目录不存在的情况（打印错误信息并退出）

---

### 进阶挑战题

**练习 5**：仿照本章介绍的 PyTorch 项目结构，为一个**文本分类**项目设计目录结构，并实现以下模块（不需要真正的深度学习代码，用伪代码或占位函数即可）：

1. `src/datasets/text_dataset.py`：包含 `TextDataset` 类，能从 `.txt` 文件加载数据
2. `src/utils/text_preprocessing.py`：包含 `tokenize(text)`、`build_vocab(texts)`、`text_to_indices(text, vocab)` 函数
3. `src/utils/metrics.py`：包含 `accuracy(y_pred, y_true)`、`f1_score(y_pred, y_true, average='macro')` 函数
4. `src/__init__.py`：定义 `__version__`，并从各子模块导入关键类和函数
5. `scripts/train.py`：包含 `parse_args()` 函数（至少支持 `--data-dir`、`--epochs`、`--lr`、`--batch-size` 参数）和 `main()` 函数骨架

要求：所有模块中包含完整的 `docstring`，所有函数有明确的类型提示（Type Hints）。

---

## 练习答案

### 练习 1 答案

```python
# geometry.py

"""
几何形状工具模块

提供常见几何形状的面积、周长计算功能。
"""

PI = 3.14159


def circle_perimeter(radius: float) -> float:
    """
    计算圆的周长。

    Args:
        radius: 圆的半径，必须为非负数

    Returns:
        圆的周长
    """
    if radius < 0:
        raise ValueError(f"半径不能为负数：{radius}")
    return 2 * PI * radius


def triangle_area(base: float, height: float) -> float:
    """
    计算三角形面积。

    Args:
        base: 底边长度
        height: 高度

    Returns:
        三角形面积
    """
    if base < 0 or height < 0:
        raise ValueError("底边和高度必须为非负数")
    return 0.5 * base * height


class Rectangle:
    """矩形类"""

    def __init__(self, width: float, height: float):
        if width <= 0 or height <= 0:
            raise ValueError("宽度和高度必须为正数")
        self.width = width
        self.height = height

    def area(self) -> float:
        """返回矩形面积"""
        return self.width * self.height

    def perimeter(self) -> float:
        """返回矩形周长"""
        return 2 * (self.width + self.height)

    def __repr__(self) -> str:
        return f"Rectangle(width={self.width}, height={self.height})"


if __name__ == '__main__':
    # 测试代码
    print("=== geometry 模块测试 ===")
    print(f"PI = {PI}")

    r = 5
    print(f"半径为 {r} 的圆，周长 = {circle_perimeter(r):.4f}")

    print(f"底为 6、高为 4 的三角形面积 = {triangle_area(6, 4):.2f}")

    rect = Rectangle(3, 5)
    print(f"{rect}")
    print(f"  面积 = {rect.area()}")
    print(f"  周长 = {rect.perimeter()}")
```

```python
# 在另一个文件中使用三种方式导入

# 方式1：import 整个模块
import geometry
print(geometry.PI)
print(geometry.circle_perimeter(3))
r = geometry.Rectangle(4, 6)
print(r.area())

# 方式2：from...import 导入特定内容
from geometry import PI, triangle_area, Rectangle
print(PI)
print(triangle_area(3, 4))
rect = Rectangle(5, 8)
print(rect.perimeter())

# 方式3：使用别名
import geometry as geo
from geometry import circle_perimeter as cp

print(geo.PI)
print(cp(10))
```

---

### 练习 2 答案

```python
# exercise2.py

import os
import random
from datetime import datetime, timedelta


# --- 任务1：创建目录结构 ---
def create_project_dirs():
    dirs = ['project/data', 'project/output', 'project/logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"已创建目录：{d}")


# --- 任务2：生成随机整数并统计偶数 ---
def count_even_numbers():
    random.seed(42)  # 设置种子保证可复现
    numbers = [random.randint(1, 100) for _ in range(20)]
    even_count = sum(1 for n in numbers if n % 2 == 0)
    print(f"生成的随机数：{numbers}")
    print(f"偶数个数：{even_count}")
    return even_count


# --- 任务3：计算100天后的日期 ---
def date_after_100_days():
    today = datetime.today()
    future = today + timedelta(days=100)
    print(f"今天是：{today.strftime('%Y年%m月%d日')}")
    print(f"100天后是：{future.strftime('%Y年%m月%d日')}")
    return future


if __name__ == '__main__':
    print("=== 任务1：创建目录 ===")
    create_project_dirs()

    print("\n=== 任务2：随机数统计 ===")
    count_even_numbers()

    print("\n=== 任务3：日期计算 ===")
    date_after_100_days()
```

---

### 练习 3 答案

```python
# statslib/__init__.py

"""
statslib - 简单统计库

提供基础和进阶统计函数。
"""

__version__ = '1.0.0'
__author__ = 'Python Learner'

from .basic import mean, median, mode
from .advanced import variance, std_dev, correlation

__all__ = ['mean', 'median', 'mode', 'variance', 'std_dev', 'correlation']
```

```python
# statslib/basic.py

"""基础统计函数"""

from typing import List


def mean(data: List[float]) -> float:
    """计算算术平均值"""
    if not data:
        raise ValueError("数据列表不能为空")
    return sum(data) / len(data)


def median(data: List[float]) -> float:
    """计算中位数"""
    if not data:
        raise ValueError("数据列表不能为空")
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    return float(sorted_data[mid])


def mode(data: List[float]) -> float:
    """计算众数（返回出现次数最多的值，若有多个返回最小的）"""
    if not data:
        raise ValueError("数据列表不能为空")
    # 统计每个值出现的次数
    counts = {}
    for value in data:
        counts[value] = counts.get(value, 0) + 1
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    return min(modes)
```

```python
# statslib/advanced.py

"""进阶统计函数"""

from typing import List
from .basic import mean


def variance(data: List[float], ddof: int = 0) -> float:
    """
    计算方差。

    Args:
        data: 数据列表
        ddof: 自由度修正（0=总体方差，1=样本方差）

    Returns:
        方差值
    """
    if len(data) < 2:
        raise ValueError("至少需要2个数据点")
    m = mean(data)
    n = len(data) - ddof
    return sum((x - m) ** 2 for x in data) / n


def std_dev(data: List[float], ddof: int = 0) -> float:
    """计算标准差"""
    return variance(data, ddof) ** 0.5


def correlation(x: List[float], y: List[float]) -> float:
    """
    计算皮尔逊相关系数。

    Returns:
        相关系数，范围 [-1, 1]
    """
    if len(x) != len(y):
        raise ValueError("两组数据长度必须相同")
    if len(x) < 2:
        raise ValueError("至少需要2个数据点")

    mean_x = mean(x)
    mean_y = mean(y)
    numerator = sum((xi - mean_x) * (yi - mean_y)
                    for xi, yi in zip(x, y))
    denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if denom_x == 0 or denom_y == 0:
        raise ValueError("数据方差为0，无法计算相关系数")
    return numerator / (denom_x * denom_y)


# 验证测试
if __name__ == '__main__':
    from .basic import mean
    print(mean([1, 2, 3, 4, 5]))          # 3.0
    print(variance([2, 4, 4, 4, 5, 5, 7, 9]))  # 4.0
```

---

### 练习 4 答案

```python
# file_stats.py

"""
文件统计脚本

用法：python file_stats.py <目录路径>
"""

import sys
import os
from collections import defaultdict


def format_size(size_bytes: int) -> str:
    """将字节数格式化为带千分位的字符串"""
    return f"{size_bytes:,}"


def analyze_directory(directory: str) -> None:
    """分析目录并打印统计信息"""
    if not os.path.exists(directory):
        print(f"错误：目录不存在：{directory}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(directory):
        print(f"错误：路径不是目录：{directory}", file=sys.stderr)
        sys.exit(1)

    total_files = 0
    total_dirs = 0
    ext_stats = defaultdict(lambda: {'count': 0, 'size': 0})
    max_file = ('', 0)
    min_file = ('', float('inf'))

    for root, dirs, files in os.walk(directory):
        total_dirs += len(dirs)
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                size = os.path.getsize(filepath)
            except OSError:
                continue

            total_files += 1
            _, ext = os.path.splitext(filename)
            ext = ext.lower() if ext else '(无扩展名)'
            ext_stats[ext]['count'] += 1
            ext_stats[ext]['size'] += size

            rel_path = os.path.relpath(filepath, directory)
            if size > max_file[1]:
                max_file = (rel_path, size)
            if size < min_file[1]:
                min_file = (rel_path, size)

    # 打印结果
    sep = '-' * 40
    print(f"目录：{os.path.abspath(directory)}")
    print(sep)
    print(f"总文件数：{total_files}")
    print(f"总目录数：{total_dirs}")

    if ext_stats:
        print("各类型文件统计：")
        for ext, info in sorted(ext_stats.items()):
            print(f"  {ext} 文件：{info['count']} 个，"
                  f"共 {format_size(info['size'])} 字节")

    print(sep)
    if max_file[0]:
        print(f"最大文件：{max_file[0]} ({format_size(max_file[1])} 字节)")
    if min_file[0] and min_file[1] != float('inf'):
        print(f"最小文件：{min_file[0]} ({format_size(min_file[1])} 字节)")


def main():
    if len(sys.argv) != 2:
        print(f"用法：python {sys.argv[0]} <目录路径>")
        sys.exit(1)

    target_dir = sys.argv[1]
    analyze_directory(target_dir)


if __name__ == '__main__':
    main()
```

---

### 练习 5 答案

```python
# src/datasets/text_dataset.py

"""文本数据集模块"""

import os
from typing import List, Tuple, Optional


class TextDataset:
    """
    从文本文件加载分类数据集。

    文件格式：每行一条数据，格式为 "标签\t文本内容"

    Args:
        file_path (str): 数据文件路径
        max_length (int): 文本最大长度，超出则截断
    """

    def __init__(self, file_path: str, max_length: int = 512):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在：{file_path}")

        self.file_path = file_path
        self.max_length = max_length
        self.data: List[Tuple[str, str]] = []  # (label, text)
        self._load()

    def _load(self) -> None:
        """加载数据文件"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    print(f"警告：第 {line_no} 行格式不正确，已跳过")
                    continue
                label, text = parts
                self.data.append((label, text[:self.max_length]))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.data[idx]

    def get_labels(self) -> List[str]:
        """返回所有唯一标签"""
        return sorted(set(label for label, _ in self.data))
```

```python
# src/utils/text_preprocessing.py

"""文本预处理工具"""

import re
from typing import List, Dict, Optional


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    对文本进行简单分词（按空格和标点切分）。

    Args:
        text: 输入文本
        lowercase: 是否转换为小写

    Returns:
        词列表
    """
    if lowercase:
        text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def build_vocab(
    texts: List[str],
    min_freq: int = 1,
    max_size: Optional[int] = None,
    special_tokens: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    从文本列表构建词汇表。

    Args:
        texts: 文本列表
        min_freq: 词的最小出现频率
        max_size: 词汇表最大大小（不含特殊词元）
        special_tokens: 特殊词元列表，如 ['<PAD>', '<UNK>']

    Returns:
        词 -> 索引的字典
    """
    if special_tokens is None:
        special_tokens = ['<PAD>', '<UNK>']

    # 统计词频
    freq: Dict[str, int] = {}
    for text in texts:
        for token in tokenize(text):
            freq[token] = freq.get(token, 0) + 1

    # 过滤低频词并排序
    vocab_tokens = [w for w, c in freq.items() if c >= min_freq]
    vocab_tokens.sort(key=lambda w: -freq[w])  # 按频率降序

    if max_size is not None:
        vocab_tokens = vocab_tokens[:max_size]

    # 构建词典（特殊词元排在前面）
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for token in vocab_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def text_to_indices(
    text: str,
    vocab: Dict[str, int],
    max_length: Optional[int] = None,
    pad_token: str = '<PAD>',
    unk_token: str = '<UNK>'
) -> List[int]:
    """
    将文本转换为词索引序列。

    Args:
        text: 输入文本
        vocab: 词汇表
        max_length: 最大序列长度（不足则填充，超出则截断）
        pad_token: 填充词元
        unk_token: 未知词元

    Returns:
        词索引列表
    """
    tokens = tokenize(text)
    unk_idx = vocab.get(unk_token, 0)
    pad_idx = vocab.get(pad_token, 0)

    indices = [vocab.get(token, unk_idx) for token in tokens]

    if max_length is not None:
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices += [pad_idx] * (max_length - len(indices))

    return indices
```

```python
# src/utils/metrics.py

"""评估指标工具"""

from typing import List, Dict


def accuracy(y_pred: List[int], y_true: List[int]) -> float:
    """
    计算分类准确率。

    Args:
        y_pred: 预测标签列表
        y_true: 真实标签列表

    Returns:
        准确率，范围 [0, 1]
    """
    if len(y_pred) != len(y_true):
        raise ValueError("预测和真实标签长度必须相同")
    if not y_pred:
        raise ValueError("标签列表不能为空")
    correct = sum(p == t for p, t in zip(y_pred, y_true))
    return correct / len(y_true)


def f1_score(
    y_pred: List[int],
    y_true: List[int],
    average: str = 'macro'
) -> float:
    """
    计算 F1 分数。

    Args:
        y_pred: 预测标签列表
        y_true: 真实标签列表
        average: 平均方式，'macro'（宏平均）或 'micro'（微平均）

    Returns:
        F1 分数，范围 [0, 1]
    """
    labels = set(y_true) | set(y_pred)

    def _f1_per_class(label: int) -> float:
        tp = sum(p == t == label for p, t in zip(y_pred, y_true))
        fp = sum(p == label and t != label for p, t in zip(y_pred, y_true))
        fn = sum(p != label and t == label for p, t in zip(y_pred, y_true))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    if average == 'macro':
        scores = [_f1_per_class(label) for label in labels]
        return sum(scores) / len(scores)
    elif average == 'micro':
        tp_total = sum(p == t for p, t in zip(y_pred, y_true))
        return tp_total / len(y_true)
    else:
        raise ValueError(f"不支持的 average 参数：{average}")
```

```python
# src/__init__.py

"""
文本分类项目核心包

使用方法：
    from src.datasets.text_dataset import TextDataset
    from src.utils.text_preprocessing import tokenize, build_vocab
    from src.utils.metrics import accuracy, f1_score
"""

__version__ = '0.1.0'
__author__ = 'Deep Learning Practitioner'

from .datasets.text_dataset import TextDataset
from .utils.text_preprocessing import tokenize, build_vocab, text_to_indices
from .utils.metrics import accuracy, f1_score

__all__ = [
    'TextDataset',
    'tokenize',
    'build_vocab',
    'text_to_indices',
    'accuracy',
    'f1_score',
]
```

```python
# scripts/train.py

"""
文本分类训练脚本

用法：
    python scripts/train.py --data-dir data/ --epochs 10 --lr 0.001
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='文本分类模型训练脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据相关
    parser.add_argument('--data-dir', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--max-length', type=int, default=256,
                        help='文本最大序列长度')

    # 训练相关
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')

    # 模型相关
    parser.add_argument('--model', type=str, default='textcnn',
                        choices=['textcnn', 'lstm', 'bert'],
                        help='模型架构')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='隐藏层维度')

    # 其他
    parser.add_argument('--exp-name', type=str, default='experiment',
                        help='实验名称')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='训练设备')

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 50)
    print("文本分类训练脚本")
    print("=" * 50)
    print(f"数据目录：{args.data_dir}")
    print(f"模型：{args.model}")
    print(f"训练轮数：{args.epochs}")
    print(f"学习率：{args.lr}")
    print(f"批次大小：{args.batch_size}")
    print(f"设备：{args.device}")
    print("=" * 50)

    # 训练主循环骨架（占位）
    # 1. 设置随机种子
    # set_all_seeds(args.seed)

    # 2. 加载数据
    # train_dataset = TextDataset(os.path.join(args.data_dir, 'train.txt'))
    # val_dataset   = TextDataset(os.path.join(args.data_dir, 'val.txt'))

    # 3. 构建词汇表
    # vocab = build_vocab([text for _, text in train_dataset])

    # 4. 创建模型
    # model = build_model(args.model, len(vocab), args.hidden_size)

    # 5. 训练循环
    # for epoch in range(args.epochs):
    #     train_one_epoch(model, train_dataset, args.lr, args.batch_size)
    #     val_metrics = evaluate(model, val_dataset)
    #     print(f"Epoch {epoch+1}/{args.epochs} - "
    #           f"val_acc: {val_metrics['accuracy']:.4f}")

    print("训练完成！（当前为骨架代码，待补充实现）")


if __name__ == '__main__':
    main()
```
