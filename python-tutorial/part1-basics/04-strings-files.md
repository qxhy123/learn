# 第4章：字符串与文件操作

> **系列定位**：本章是"Python基础"模块的第4章。前三章介绍了变量、控制流与函数，本章聚焦字符串处理与文件I/O——深度学习工程中最高频的数据预处理技能。

---

## 学习目标

完成本章后，你将能够：

1. 熟练使用字符串索引、切片和常用内置方法对文本数据进行清洗与变换
2. 掌握三种字符串格式化方式（`%`、`.format()`、f-string），并选择适合场景的方式
3. 使用 `open()` 与 `with` 语句安全地读写文本和二进制文件
4. 利用 `os.path` 和 `pathlib` 构建跨平台的文件路径
5. 综合运用以上技能加载 CSV 和 JSON 格式的深度学习数据集

---

## 4.1 字符串基础

### 4.1.1 创建字符串

Python 字符串是**不可变的 Unicode 字符序列**，支持四种字面量写法：

```python
# 单引号、双引号效果相同
s1 = 'Hello, Python!'
s2 = "Hello, Python!"

# 三引号字符串：可跨行，常用于文档字符串或多行模板
s3 = """第一行
第二行
第三行"""

# 原始字符串（raw string）：反斜杠不转义，常用于正则和Windows路径
s4 = r"C:\Users\model\data"
print(s4)  # C:\Users\model\data

# 字节字符串（bytes）：深度学习读取二进制文件时会遇到
b1 = b"binary data"
print(type(b1))  # <class 'bytes'>
```

字符串长度与类型：

```python
text = "深度学习"
print(len(text))   # 4  （按字符计，每个汉字算1个字符）
print(type(text))  # <class 'str'>
```

### 4.1.2 索引与切片

字符串的每个字符都有正向（从 0 开始）和反向（从 -1 开始）两套索引：

```
字符串:  P  y  t  h  o  n
正向索引: 0  1  2  3  4  5
反向索引:-6 -5 -4 -3 -2 -1
```

```python
s = "Python"

# 单字符索引
print(s[0])   # P
print(s[-1])  # n
print(s[2])   # t

# 切片语法：s[start:stop:step]
# stop 位置的字符不包含在结果中
print(s[0:3])   # Pyt
print(s[2:])    # thon   （省略stop，到末尾）
print(s[:4])    # Pyth   （省略start，从开头）
print(s[::2])   # Pto    （步长为2，隔一个取一个）
print(s[::-1])  # nohtyP （步长为-1，倒序）

# 字符串是不可变的——不能通过索引修改
# s[0] = 'J'  →  TypeError: 'str' object does not support item assignment
```

实际应用：从文件名中提取扩展名

```python
filename = "train_data.csv"

# 方法一：手动切片
dot_pos = filename.rfind('.')    # 找最后一个 '.' 的位置
ext = filename[dot_pos:]         # '.csv'
name = filename[:dot_pos]        # 'train_data'

print(f"文件名: {name}, 扩展名: {ext}")
# 输出: 文件名: train_data, 扩展名: .csv
```

### 4.1.3 字符串的不可变性与拼接

```python
# 拼接：+ 运算符（小规模使用）
greeting = "Hello" + ", " + "World!"
print(greeting)  # Hello, World!

# 重复：* 运算符
separator = "-" * 30
print(separator)  # ------------------------------

# in 运算符：成员检测
print("py" in "python")    # True
print("Java" in "python")  # False

# 大量字符串拼接时，用 join 而非循环 +
# 低效（每次 + 都创建新对象）：
words = ["deep", "learning", "is", "fun"]
result = ""
for w in words:
    result += w + " "

# 高效（join 一次性分配内存）：
result = " ".join(words)
print(result)  # deep learning is fun
```

---

## 4.2 字符串方法

Python 字符串提供 40+ 个内置方法，以下是深度学习工程中最常用的子集。

### 4.2.1 大小写与空白处理

```python
text = "  Hello, Deep Learning!  "

# 去除首尾空白（strip 家族）
print(text.strip())        # "Hello, Deep Learning!"
print(text.lstrip())       # "Hello, Deep Learning!  "
print(text.rstrip())       # "  Hello, Deep Learning!"
print(text.strip("! "))    # "Hello, Deep Learning"  （去除指定字符）

# 大小写转换
s = "PyTorch vs TensorFlow"
print(s.lower())       # pytorch vs tensorflow
print(s.upper())       # PYTORCH VS TENSORFLOW
print(s.title())       # Pytorch Vs Tensorflow
print(s.capitalize())  # Pytorch vs tensorflow
print(s.swapcase())    # pYtORCH VS tENSORfLOW
```

### 4.2.2 搜索与替换

```python
text = "the quick brown fox jumps over the lazy dog"

# 查找子串位置
print(text.find("fox"))     # 16  （找不到返回 -1）
print(text.rfind("the"))    # 31  （从右往左查找）
print(text.index("fox"))    # 16  （找不到抛出 ValueError）
print(text.count("the"))    # 2   （统计出现次数）

# 判断开头/结尾
print(text.startswith("the"))  # True
print(text.endswith("dog"))    # True
print(text.startswith(("the", "a")))  # True（元组：任意匹配即返回True）

# 替换
print(text.replace("fox", "cat"))           # 替换所有匹配
print(text.replace("the", "a", 1))          # 只替换第1次出现
```

### 4.2.3 分割与连接

```python
# split：按分隔符分割
csv_line = "Alice,25,engineer,Beijing"
fields = csv_line.split(",")
print(fields)  # ['Alice', '25', 'engineer', 'Beijing']

# maxsplit：限制分割次数
parts = csv_line.split(",", 2)
print(parts)   # ['Alice', '25', 'engineer,Beijing']

# splitlines：按行分割（兼容 \n \r\n \r）
multiline = "line1\nline2\r\nline3\rline4"
print(multiline.splitlines())
# ['line1', 'line2', 'line3', 'line4']

# join：拼接序列
words = ["2024", "01", "15"]
print("-".join(words))   # 2024-01-15
print("/".join(words))   # 2024/01/15

# 实际场景：处理 TSV（制表符分隔）数据
tsv_line = "image001.jpg\t0\t猫"
img_path, label_id, label_name = tsv_line.split("\t")
print(img_path, label_id, label_name)
# image001.jpg 0 猫
```

### 4.2.4 检测与验证

```python
# 字符类型检测——常用于数据清洗
print("123".isdigit())      # True
print("abc".isalpha())      # True
print("abc123".isalnum())   # True
print("   ".isspace())      # True
print("Hello".istitle())    # True

# 实际应用：验证标签ID是否为纯数字
def is_valid_label(s):
    return s.strip().isdigit()

labels = ["0", "1", " 2 ", "cat", "3a"]
valid = [l for l in labels if is_valid_label(l)]
print(valid)  # ['0', '1', ' 2 ']
```

### 4.2.5 编码与解码

深度学习常需要处理多语言文本：

```python
text = "深度学习"

# 编码：str → bytes
encoded_utf8 = text.encode("utf-8")
encoded_gbk  = text.encode("gbk")
print(encoded_utf8)  # b'\xe6\xb7\xb1\xe5\xba\xa6\xe5\xad\xa6\xe4\xb9\xa0'

# 解码：bytes → str
decoded = encoded_utf8.decode("utf-8")
print(decoded)  # 深度学习

# 处理未知编码时的容错
raw = b"\xe6\xb7\xb1\xb6\xa8"  # 混乱字节
safe = raw.decode("utf-8", errors="replace")   # 用 ? 替代无效字节
print(safe)
```

---

## 4.3 字符串格式化

Python 提供三代格式化方案，推荐优先使用 **f-string**（Python 3.6+）。

### 4.3.1 % 格式化（传统风格，了解即可）

```python
name = "AlexNet"
acc  = 0.8931

# 基本占位符
print("模型: %s, 准确率: %.2f%%" % (name, acc * 100))
# 模型: AlexNet, 准确率: 89.31%

# 常用格式说明符
# %s  字符串
# %d  整数
# %f  浮点数（默认6位小数）
# %.2f 保留2位小数
# %05d 用0填充到5位宽度
epoch = 3
print("Epoch %05d" % epoch)  # Epoch 00003
```

### 4.3.2 .format() 方法（Python 3.0+）

```python
# 位置参数
print("损失: {}, 准确率: {}".format(0.234, 0.912))

# 命名参数（更易读）
template = "Epoch [{epoch}/{total}] Loss: {loss:.4f} Acc: {acc:.2%}"
log = template.format(epoch=5, total=100, loss=0.1234, acc=0.9156)
print(log)
# Epoch [5/100] Loss: 0.1234 Acc: 91.56%

# 格式规范 (Format Spec Mini-Language)
pi = 3.14159265
print("{:.2f}".format(pi))    # 3.14  （2位小数）
print("{:10.4f}".format(pi))  # '    3.1416' （宽度10，右对齐）
print("{:<10}".format("left")) # 'left      '（左对齐）
print("{:^10}".format("mid"))  # '   mid    '（居中对齐）
print("{:,}".format(1000000))  # 1,000,000   （千分位分隔符）
```

### 4.3.3 f-string（推荐，Python 3.6+）

```python
model_name = "ResNet50"
top1_acc   = 0.7612
top5_acc   = 0.9295
params_m   = 25.6   # 单位：百万参数

# 基本用法：{表达式}
print(f"模型: {model_name}")
print(f"Top-1: {top1_acc:.2%}, Top-5: {top5_acc:.2%}")
# Top-1: 76.12%, Top-5: 92.95%

# 可以在 {} 内写任意表达式
print(f"参数量: {params_m:.1f}M ({params_m * 1e6:.2e})")
# 参数量: 25.6M (2.56e+07)

# 调试用法：{变量=} 同时显示变量名和值（Python 3.8+）
loss = 0.3456
print(f"{loss=:.4f}")  # loss=0.3456

# 多行 f-string
report = (
    f"{'模型':>8}: {model_name}\n"
    f"{'Top-1':>8}: {top1_acc:.4f}\n"
    f"{'Top-5':>8}: {top5_acc:.4f}\n"
    f"{'参数量':>8}: {params_m}M"
)
print(report)
```

输出：
```
    模型: ResNet50
   Top-1: 0.7612
   Top-5: 0.9295
  参数量: 25.6M
```

### 4.3.4 格式化对比总结

| 场景 | 推荐方式 | 原因 |
|------|----------|------|
| 新代码、日常使用 | f-string | 简洁、性能好、可读性强 |
| 模板字符串（外部配置） | `.format()` | 模板可与代码分离 |
| 兼容 Python 2 的旧代码 | `%` | 历史遗留 |
| 国际化（i18n） | `.format()` | 翻译工具支持更好 |

---

## 4.4 文件读写

### 4.4.1 open() 与文件模式

```python
# 语法：open(file, mode='r', encoding=None, errors=None)
# 常用模式：
# 'r'  读取文本（默认）
# 'w'  写入文本（覆盖）
# 'a'  追加文本
# 'x'  创建新文件（文件已存在则报错）
# 'b'  二进制模式（与上述模式组合，如 'rb'、'wb'）
# '+'  读写模式（如 'r+'、'w+'）
```

### 4.4.2 with 语句（推荐写法）

`with` 语句保证文件在块结束时**自动关闭**，即使发生异常也不例外：

```python
# 写入文件
with open("/tmp/hello.txt", "w", encoding="utf-8") as f:
    f.write("第一行\n")
    f.write("第二行\n")
    f.writelines(["第三行\n", "第四行\n"])  # 写入字符串列表

# 读取整个文件
with open("/tmp/hello.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(repr(content))
    # '第一行\n第二行\n第三行\n第四行\n'

# 逐行读取（大文件友好：不一次性加载到内存）
with open("/tmp/hello.txt", "r", encoding="utf-8") as f:
    for line in f:               # 文件对象本身是迭代器
        print(line.rstrip())     # strip 去掉行尾换行符

# 读取所有行到列表
with open("/tmp/hello.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
print(lines)  # ['第一行\n', '第二行\n', '第三行\n', '第四行\n']

# 只读第一行
with open("/tmp/hello.txt", "r", encoding="utf-8") as f:
    first_line = f.readline()
```

### 4.4.3 追加与文件指针

```python
# 追加模式：不清空原有内容
with open("/tmp/hello.txt", "a", encoding="utf-8") as f:
    f.write("第五行\n")

# 文件指针操作（在 'r+' 模式下随机读写）
with open("/tmp/hello.txt", "r+", encoding="utf-8") as f:
    print(f.tell())      # 0  （当前指针位置，字节数）
    f.read(3)
    print(f.tell())      # 9  （每个汉字3字节 UTF-8）
    f.seek(0)            # 移回开头
    print(f.readline())  # 第一行
```

### 4.4.4 二进制文件读写

深度学习常用的模型权重（`.pt`、`.npy`）是二进制文件：

```python
import struct

# 写入二进制：将三个浮点数保存为二进制
weights = [0.1, 0.5, 0.9]
with open("/tmp/weights.bin", "wb") as f:
    for w in weights:
        f.write(struct.pack("f", w))  # 'f' 代表 C float（4字节）

# 读取二进制
with open("/tmp/weights.bin", "rb") as f:
    raw = f.read()
    n = len(raw) // 4   # 每个 float 4字节
    values = [struct.unpack("f", raw[i*4:(i+1)*4])[0] for i in range(n)]
    print([round(v, 4) for v in values])  # [0.1, 0.5, 0.9]
```

### 4.4.5 异常处理与健壮性

```python
def safe_read(filepath):
    """安全读取文件，返回内容或None"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"文件不存在: {filepath}")
        return None
    except PermissionError:
        print(f"权限不足: {filepath}")
        return None
    except UnicodeDecodeError:
        # 尝试 GBK 编码回退
        try:
            with open(filepath, "r", encoding="gbk") as f:
                return f.read()
        except Exception as e:
            print(f"编码错误: {e}")
            return None

content = safe_read("/tmp/hello.txt")
```

---

## 4.5 路径处理

### 4.5.1 os.path 模块（经典方式）

```python
import os

# 路径拼接：跨平台（自动使用 / 或 \）
base = "/data/datasets"
sub  = "imagenet/train"
path = os.path.join(base, sub, "n01440764")
print(path)  # /data/datasets/imagenet/train/n01440764

# 拆分路径
full = "/data/datasets/train.csv"
print(os.path.dirname(full))   # /data/datasets
print(os.path.basename(full))  # train.csv
print(os.path.split(full))     # ('/data/datasets', 'train.csv')
print(os.path.splitext(full))  # ('/data/datasets/train', '.csv')

# 路径状态查询
print(os.path.exists(full))    # True / False
print(os.path.isfile(full))    # 是否为普通文件
print(os.path.isdir(full))     # 是否为目录
print(os.path.getsize(full))   # 文件字节数

# 获取当前工作目录与脚本所在目录
cwd = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录

# 规范化路径（解析 . 和 ..）
messy = "/data/./datasets/../datasets/train.csv"
print(os.path.normpath(messy))  # /data/datasets/train.csv
```

### 4.5.2 pathlib 模块（推荐，Python 3.4+）

`pathlib` 以面向对象方式操作路径，代码更直观：

```python
from pathlib import Path

# 创建 Path 对象
data_dir = Path("/data/datasets")
train_csv = data_dir / "imagenet" / "train.csv"  # / 运算符拼接路径
print(train_csv)         # /data/datasets/imagenet/train.csv
print(type(train_csv))   # <class 'pathlib.PosixPath'>

# 路径属性
p = Path("/data/datasets/train.csv")
print(p.name)       # train.csv
print(p.stem)       # train
print(p.suffix)     # .csv
print(p.suffixes)   # ['.csv']
print(p.parent)     # /data/datasets
print(p.parts)      # ('/', 'data', 'datasets', 'train.csv')

# 路径查询
print(p.exists())
print(p.is_file())
print(p.is_dir())

# 相对路径与绝对路径
rel = Path("data/train.csv")
print(rel.resolve())    # 转为绝对路径（基于当前工作目录）

# 创建目录（递归，已存在不报错）
output_dir = Path("/tmp/experiment/logs")
output_dir.mkdir(parents=True, exist_ok=True)

# 遍历目录
data_root = Path("/tmp")
for f in data_root.iterdir():
    print(f.name, "dir" if f.is_dir() else "file")

# 通配符搜索
for csv_file in data_root.glob("*.txt"):
    print(csv_file)

for py_file in data_root.rglob("*.py"):  # 递归搜索
    print(py_file)

# 读写文件（Path 对象直接支持）
config_path = Path("/tmp/config.txt")
config_path.write_text("lr=0.001\nbatch_size=32\n", encoding="utf-8")
content = config_path.read_text(encoding="utf-8")
print(content)

# 修改后缀
train_path = Path("annotations/train.csv")
json_path = train_path.with_suffix(".json")
print(json_path)  # annotations/train.json
```

### 4.5.3 os.path vs pathlib 对比

| 操作 | os.path 方式 | pathlib 方式 |
|------|-------------|-------------|
| 拼接路径 | `os.path.join(a, b)` | `Path(a) / b` |
| 获取文件名 | `os.path.basename(p)` | `p.name` |
| 获取扩展名 | `os.path.splitext(p)[1]` | `p.suffix` |
| 判断存在 | `os.path.exists(p)` | `p.exists()` |
| 读取文本 | `open(p).read()` | `p.read_text()` |
| 创建目录 | `os.makedirs(p, exist_ok=True)` | `p.mkdir(parents=True, exist_ok=True)` |

**建议**：新代码统一使用 `pathlib`；与旧库交互时，用 `str(path)` 将 Path 对象转为字符串。

---

## 本章小结

| 知识点 | 核心语法 | 注意事项 |
|--------|----------|----------|
| 字符串创建 | `''` `""` `""" """` `r""` | 字符串不可变 |
| 索引与切片 | `s[i]` `s[a:b:step]` | `stop` 不含端点；支持负索引 |
| 常用方法 | `strip/split/join/replace/find` | `split()` 无参按任意空白分割 |
| 格式化 | f-string 优先 | f-string 需 Python 3.6+ |
| 文件读写 | `with open(...) as f` | 始终指定 `encoding="utf-8"` |
| 路径处理 | `pathlib.Path` | 用 `/` 拼接；`mkdir(parents=True)` |

---

## 深度学习应用：数据集文件的加载

真实的深度学习项目需要从磁盘批量加载样本。本节展示如何读取 **CSV** 和 **JSON** 两种常见数据集格式。

### 应用 4-1：加载 CSV 标注文件

假设 ImageNet 风格数据集的标注文件 `annotations.csv` 格式如下：

```
image_path,label_id,label_name,split
data/train/n01440764/img001.JPEG,0,tench,train
data/train/n01440764/img002.JPEG,0,tench,train
data/val/n01440764/img001.JPEG,0,tench,val
```

**方法一：纯字符串操作（不依赖第三方库）**

```python
from pathlib import Path

def load_csv_annotations(csv_path, split="train"):
    """
    加载 CSV 标注文件，返回指定 split 的样本列表。

    Args:
        csv_path: CSV 文件路径（str 或 Path）
        split:    数据集划分，'train' / 'val' / 'test'

    Returns:
        list of dict，每个元素包含 image_path, label_id, label_name
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"标注文件不存在: {csv_path}")

    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")  # 读取表头
        print(f"表头: {header}")

        for lineno, line in enumerate(f, start=2):
            line = line.strip()
            if not line:          # 跳过空行
                continue

            fields = line.split(",")
            if len(fields) != len(header):
                print(f"第{lineno}行字段数不符，跳过: {line}")
                continue

            row = dict(zip(header, fields))

            if row["split"] != split:
                continue

            samples.append({
                "image_path": row["image_path"],
                "label_id":   int(row["label_id"]),
                "label_name": row["label_name"],
            })

    print(f"加载完成：{split} 集共 {len(samples)} 条样本")
    return samples


# 演示：创建临时CSV并加载
import tempfile, os

csv_content = """\
image_path,label_id,label_name,split
data/train/cat/img001.jpg,0,cat,train
data/train/cat/img002.jpg,0,cat,train
data/train/dog/img001.jpg,1,dog,train
data/val/cat/img001.jpg,0,cat,val
data/val/dog/img001.jpg,1,dog,val
"""

with tempfile.NamedTemporaryFile(
    mode="w", suffix=".csv", delete=False, encoding="utf-8"
) as tmp:
    tmp.write(csv_content)
    tmp_path = tmp.name

train_samples = load_csv_annotations(tmp_path, split="train")
val_samples   = load_csv_annotations(tmp_path, split="val")

print("\n训练集前2条:")
for s in train_samples[:2]:
    print(f"  {s}")

os.unlink(tmp_path)  # 清理临时文件
```

输出：
```
表头: ['image_path', 'label_id', 'label_name', 'split']
加载完成：train 集共 3 条样本
表头: ['image_path', 'label_id', 'label_name', 'split']
加载完成：val 集共 2 条样本

训练集前2条:
  {'image_path': 'data/train/cat/img001.jpg', 'label_id': 0, 'label_name': 'cat'}
  {'image_path': 'data/train/cat/img002.jpg', 'label_id': 0, 'label_name': 'cat'}
```

**方法二：使用标准库 `csv` 模块（处理含引号的字段）**

```python
import csv
from pathlib import Path

def load_csv_with_module(csv_path, split="train"):
    """使用 csv 模块加载，正确处理含逗号/引号的字段"""
    samples = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)    # 自动将第一行作为字段名
        for row in reader:
            if row["split"] == split:
                samples.append({
                    "image_path": row["image_path"],
                    "label_id":   int(row["label_id"]),
                    "label_name": row["label_name"],
                })
    return samples
```

### 应用 4-2：加载 JSON 格式数据集

COCO 目标检测数据集使用 JSON 格式，结构示意：

```json
{
    "info": {"year": 2017, "version": "1.0"},
    "images": [
        {"id": 1, "file_name": "000001.jpg", "height": 480, "width": 640},
        {"id": 2, "file_name": "000002.jpg", "height": 720, "width": 1280}
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 80]},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 150, 50, 60]},
        {"id": 3, "image_id": 2, "category_id": 1, "bbox": [30, 40, 200, 150]}
    ],
    "categories": [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "car"}
    ]
}
```

```python
import json
from pathlib import Path
from collections import defaultdict

def load_coco_annotations(json_path):
    """
    加载 COCO 格式 JSON 标注文件。

    Returns:
        dict: {
            'images':      {image_id: image_info},
            'annotations': {image_id: [annotation, ...]},
            'categories':  {category_id: category_name}
        }
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构建 id → info 的查找字典
    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    # 按 image_id 聚合 annotations
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)

    print(f"图片数: {len(images)}")
    print(f"类别数: {len(categories)}, 类别: {list(categories.values())}")
    print(f"标注数: {sum(len(v) for v in annotations.values())}")

    return {
        "images":      images,
        "annotations": dict(annotations),
        "categories":  categories,
    }


def get_image_annotations(dataset, image_id):
    """获取某张图片的所有标注，附带类别名称"""
    img_info = dataset["images"].get(image_id)
    if img_info is None:
        print(f"图片 ID {image_id} 不存在")
        return []

    anns = dataset["annotations"].get(image_id, [])
    result = []
    for ann in anns:
        cat_name = dataset["categories"][ann["category_id"]]
        result.append({
            "category": cat_name,
            "bbox":     ann["bbox"],   # [x, y, width, height]
        })
    return result


# 演示
coco_data = {
    "info": {"year": 2017},
    "images": [
        {"id": 1, "file_name": "000001.jpg", "height": 480, "width": 640},
        {"id": 2, "file_name": "000002.jpg", "height": 720, "width": 1280},
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 80]},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 150, 50, 60]},
        {"id": 3, "image_id": 2, "category_id": 1, "bbox": [30, 40, 200, 150]},
    ],
    "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
}

tmp_json = Path("/tmp/coco_demo.json")
tmp_json.write_text(json.dumps(coco_data, ensure_ascii=False, indent=2), encoding="utf-8")

dataset = load_coco_annotations(tmp_json)
anns_img1 = get_image_annotations(dataset, image_id=1)
print(f"\n图片 1 的标注:")
for a in anns_img1:
    print(f"  类别: {a['category']}, BBox: {a['bbox']}")

tmp_json.unlink()
```

输出：
```
图片数: 2
类别数: 2, 类别: ['person', 'car']
标注数: 3

图片 1 的标注:
  类别: person, BBox: [10, 20, 100, 80]
  类别: car, BBox: [200, 150, 50, 60]
```

### 应用 4-3：训练日志的读写

```python
import json
from pathlib import Path
from datetime import datetime

class TrainingLogger:
    """将训练指标记录到 JSONL（每行一个 JSON 对象）格式的日志文件"""

    def __init__(self, log_dir, run_name):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{run_name}.jsonl"
        self._file = open(self.log_path, "a", encoding="utf-8")

    def log(self, epoch, **metrics):
        record = {
            "timestamp": datetime.now().isoformat(),
            "epoch":     epoch,
            **metrics,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()   # 立即刷写，防止崩溃丢失数据

    def close(self):
        self._file.close()

    @staticmethod
    def load(log_path):
        """读取 JSONL 日志，返回记录列表"""
        records = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


# 使用示例
logger = TrainingLogger("/tmp/logs", run_name="resnet50_exp1")
logger.log(1, train_loss=1.234, val_loss=1.456, val_acc=0.612)
logger.log(2, train_loss=0.876, val_loss=1.102, val_acc=0.731)
logger.log(3, train_loss=0.654, val_loss=0.923, val_acc=0.789)
logger.close()

# 回读日志
records = TrainingLogger.load("/tmp/logs/resnet50_exp1.jsonl")
print(f"共 {len(records)} 条记录")
for r in records:
    print(f"  Epoch {r['epoch']:2d} | "
          f"train_loss={r['train_loss']:.3f} | "
          f"val_acc={r['val_acc']:.3f}")
```

---

## 练习题

### 基础题

**练习 4-1**：字符串清洗

给定一批从网页抓取的类别标签，存在多余空白和大小写不一致的问题，请编写函数 `clean_labels(labels)` 将其统一处理：

```python
raw_labels = [
    "  Golden Retriever  ",
    "TABBY CAT",
    "  bald Eagle",
    "great white SHARK",
    "  African Elephant  ",
]
# 期望输出：
# ['Golden Retriever', 'Tabby Cat', 'Bald Eagle', 'Great White Shark', 'African Elephant']
```

要求：去除首尾空白，并将每个单词首字母大写、其余小写。

---

**练习 4-2**：文件行数统计

编写函数 `count_file_stats(filepath)`，统计一个文本文件的：
- 总行数
- 非空行数
- 总字符数（不含换行符）
- 最长行的字符数

函数应返回一个字典，并处理文件不存在的情况。

---

### 进阶题

**练习 4-3**：CSV 写入与格式化

编写函数 `save_results_to_csv(results, output_path)`，将模型评估结果列表保存为格式化的 CSV 文件：

```python
results = [
    {"model": "ResNet50",    "top1": 0.7612, "top5": 0.9295, "params_m": 25.6},
    {"model": "EfficientB0", "top1": 0.7732, "top5": 0.9360, "params_m": 5.3},
    {"model": "ViT-B/16",    "top1": 0.8141, "top5": 0.9592, "params_m": 86.6},
]
# CSV 格式要求：
# model,top1_pct,top5_pct,params_m
# ResNet50,76.12,92.95,25.6
# ...
```

Top-1 和 Top-5 精度应转换为百分比（保留2位小数），参数量保留1位小数。

---

**练习 4-4**：目录扫描器

编写函数 `scan_dataset_dir(root_dir)`，扫描以下结构的图像数据集目录：

```
root_dir/
    train/
        cat/   img001.jpg, img002.jpg, ...
        dog/   img001.jpg, img002.jpg, ...
    val/
        cat/   img001.jpg, ...
        dog/   img001.jpg, ...
```

函数返回字典：
```python
{
    "train": {"cat": ["路径1", "路径2"], "dog": [...]},
    "val":   {"cat": [...], "dog": [...]},
    "stats": {"train_total": 100, "val_total": 20, "num_classes": 2}
}
```

---

### 挑战题

**练习 4-5**：配置文件解析器

实现一个简单的 `.ini` 风格配置文件解析器，支持：
- `[section]` 节头
- `key = value` 键值对（`=` 两侧允许有空格）
- `#` 或 `;` 开头的注释行
- 同一节内的键名不重复，但不同节可以重名

```ini
# 训练配置
[model]
name = ResNet50
num_classes = 1000

[training]
# 学习率
lr = 0.001
batch_size = 32
epochs = 100

[data]
train_dir = /data/imagenet/train
val_dir   = /data/imagenet/val
```

函数签名：`parse_config(filepath) -> dict`，返回嵌套字典，例如：
```python
{
    "model":    {"name": "ResNet50", "num_classes": "1000"},
    "training": {"lr": "0.001", "batch_size": "32", "epochs": "100"},
    "data":     {"train_dir": "/data/imagenet/train", "val_dir": "/data/imagenet/val"},
}
```

---

## 练习答案

### 答案 4-1

```python
def clean_labels(labels):
    """去除空白并转换为 Title Case"""
    cleaned = []
    for label in labels:
        # strip() 去首尾空白，title() 转 Title Case
        cleaned.append(label.strip().title())
    return cleaned

# 更 Pythonic 的写法（列表推导式）
def clean_labels_v2(labels):
    return [label.strip().title() for label in labels]

raw_labels = [
    "  Golden Retriever  ",
    "TABBY CAT",
    "  bald Eagle",
    "great white SHARK",
    "  African Elephant  ",
]
print(clean_labels(raw_labels))
# ['Golden Retriever', 'Tabby Cat', 'Bald Eagle', 'Great White Shark', 'African Elephant']
```

---

### 答案 4-2

```python
def count_file_stats(filepath):
    """统计文本文件的行数、字符数等信息"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {"error": f"文件不存在: {filepath}"}
    except Exception as e:
        return {"error": str(e)}

    total_lines    = len(lines)
    non_empty      = sum(1 for l in lines if l.strip())
    total_chars    = sum(len(l.rstrip("\n")) for l in lines)
    max_line_len   = max((len(l.rstrip("\n")) for l in lines), default=0)

    return {
        "total_lines":  total_lines,
        "non_empty":    non_empty,
        "total_chars":  total_chars,
        "max_line_len": max_line_len,
    }

# 测试
from pathlib import Path
p = Path("/tmp/test_stats.txt")
p.write_text("hello world\n\ndeep learning\n  \npython\n", encoding="utf-8")
print(count_file_stats(str(p)))
# {'total_lines': 5, 'non_empty': 3, 'total_chars': 30, 'max_line_len': 13}
p.unlink()
```

---

### 答案 4-3

```python
def save_results_to_csv(results, output_path):
    """将模型评估结果保存为 CSV 文件"""
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = "model,top1_pct,top5_pct,params_m"
    rows = []
    for r in results:
        top1_pct  = round(r["top1"] * 100, 2)
        top5_pct  = round(r["top5"] * 100, 2)
        params_m  = round(r["params_m"], 1)
        rows.append(f"{r['model']},{top1_pct},{top5_pct},{params_m}")

    content = header + "\n" + "\n".join(rows) + "\n"
    output_path.write_text(content, encoding="utf-8")
    print(f"结果已保存到: {output_path}")
    return output_path


results = [
    {"model": "ResNet50",    "top1": 0.7612, "top5": 0.9295, "params_m": 25.6},
    {"model": "EfficientB0", "top1": 0.7732, "top5": 0.9360, "params_m": 5.3},
    {"model": "ViT-B/16",    "top1": 0.8141, "top5": 0.9592, "params_m": 86.6},
]
out = save_results_to_csv(results, "/tmp/model_results.csv")
print(out.read_text())
```

---

### 答案 4-4

```python
from pathlib import Path

def scan_dataset_dir(root_dir):
    """扫描图像数据集目录，返回文件路径字典"""
    root = Path(root_dir)
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    result = {}
    stats  = {}

    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        result[split_name] = {}
        split_total = 0

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            images = [
                str(f) for f in sorted(class_dir.iterdir())
                if f.suffix.lower() in IMAGE_EXTS
            ]
            result[split_name][class_name] = images
            split_total += len(images)

        stats[f"{split_name}_total"] = split_total

    # 统计类别数（取任意 split 的类别数）
    first_split = next(iter(result.values()), {})
    stats["num_classes"] = len(first_split)
    result["stats"] = stats

    return result


# 创建测试目录结构
import os
base = Path("/tmp/demo_dataset")
for split in ("train", "val"):
    for cls in ("cat", "dog"):
        d = base / split / cls
        d.mkdir(parents=True, exist_ok=True)
        count = 4 if split == "train" else 2
        for i in range(count):
            (d / f"img{i:03d}.jpg").write_bytes(b"")

info = scan_dataset_dir(base)
print(f"训练集: {info['stats']['train_total']} 张")
print(f"验证集: {info['stats']['val_total']} 张")
print(f"类别数: {info['stats']['num_classes']}")
print(f"train/cat 文件: {info['train']['cat']}")

# 清理
import shutil
shutil.rmtree(base)
```

---

### 答案 4-5

```python
def parse_config(filepath):
    """
    解析 .ini 风格配置文件。

    支持：[section] 节头、key = value 键值对、# ; 注释。
    返回嵌套字典 {section: {key: value}}。
    """
    from pathlib import Path

    config = {}
    current_section = None

    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            # 跳过空行和注释行
            if not line or line.startswith("#") or line.startswith(";"):
                continue

            # 节头
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                if not current_section:
                    raise ValueError(f"第{lineno}行：节名为空")
                if current_section not in config:
                    config[current_section] = {}
                continue

            # 键值对
            if "=" in line:
                key, _, value = line.partition("=")
                key   = key.strip()
                value = value.strip()
                if current_section is None:
                    raise ValueError(f"第{lineno}行：键值对出现在节之前: {line}")
                config[current_section][key] = value
                continue

            # 无法识别的行（宽松处理：警告并跳过）
            print(f"警告：第{lineno}行无法解析，已跳过: {repr(raw_line)}")

    return config


# 测试
ini_content = """\
# 训练配置
[model]
name = ResNet50
num_classes = 1000

[training]
# 学习率
lr = 0.001
batch_size = 32
epochs = 100

[data]
train_dir = /data/imagenet/train
val_dir   = /data/imagenet/val
"""

from pathlib import Path
cfg_path = Path("/tmp/train_config.ini")
cfg_path.write_text(ini_content, encoding="utf-8")

config = parse_config(cfg_path)
import json
print(json.dumps(config, indent=2, ensure_ascii=False))

cfg_path.unlink()
```

输出：
```json
{
  "model": {
    "name": "ResNet50",
    "num_classes": "1000"
  },
  "training": {
    "lr": "0.001",
    "batch_size": "32",
    "epochs": "100"
  },
  "data": {
    "train_dir": "/data/imagenet/train",
    "val_dir": "/data/imagenet/val"
  }
}
```

---

> **下一章预告**：第5章将介绍 Python 的**面向对象编程**——类、继承与魔法方法，并以构建一个小型的 `Dataset` 基类作为深度学习应用案例。
