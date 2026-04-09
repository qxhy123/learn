# 第15章：Matplotlib基础

> 数据可视化是理解数据和模型行为的最直观方式。Matplotlib 是 Python 生态中历史最悠久、功能最完整的绘图库，也是深度学习训练监控的标准工具。

---

## 学习目标

完成本章学习后，你将能够：

1. 理解 Matplotlib 的三层架构（Figure、Axes、Artist），并区分面向对象 API 与 pyplot 接口的使用场景
2. 绘制折线图、散点图、柱状图和直方图等基础图表，并对数据进行可视化呈现
3. 通过标题、坐标轴标签、图例、颜色和线型等属性定制专业图表
4. 使用 `subplot`、`subplots` 和 `GridSpec` 创建多子图复合布局
5. 将训练过程中的 loss 曲线、accuracy 曲线和学习率变化可视化，用于监控深度学习训练状态

---

## 15.1 Matplotlib 架构：Figure、Axes、Artist

### 15.1.1 三层对象模型

Matplotlib 采用层次化的对象模型，理解这一架构是高效使用该库的基础。

```
Figure（画布）
└── Axes（坐标系，可有多个）
    ├── Axis（坐标轴：XAxis / YAxis）
    │   ├── Tick（刻度）
    │   └── Label（轴标签）
    ├── Line2D（折线）
    ├── PathCollection（散点）
    ├── Text（文字注释）
    └── ... 所有可见元素均为 Artist
```

| 层级 | 类名 | 描述 |
|------|------|------|
| 顶层 | `Figure` | 整张图的画布容器，管理所有子元素 |
| 中层 | `Axes` | 单个坐标系（非复数 Axis），是绘图的主要工作区 |
| 底层 | `Artist` | 所有可见对象的基类（线、点、文字、图片等） |

### 15.1.2 两种使用风格

**风格一：pyplot 函数式接口（快速探索）**

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("正弦曲线")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()
```

pyplot 内部维护一个"当前 Figure"和"当前 Axes"的全局状态，函数调用作用于当前活跃对象。适合交互式探索，但在复杂脚本中容易引发状态混淆。

**风格二：面向对象接口（推荐用于生产代码）**

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots()          # 显式创建 Figure 和 Axes
ax.plot(x, y)                     # 在指定 ax 上绘图
ax.set_title("正弦曲线")
ax.set_xlabel("x")
ax.set_ylabel("sin(x)")
plt.show()
```

面向对象接口直接操作对象，适合多子图、函数封装和测试场景。**本章后续内容均采用面向对象接口。**

### 15.1.3 创建 Figure 的常用方式

```python
import matplotlib.pyplot as plt

# 方式 1：默认尺寸
fig, ax = plt.subplots()

# 方式 2：指定尺寸（英寸）和分辨率（DPI）
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

# 方式 3：先创建 Figure 再添加 Axes
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)          # 1行1列第1个

# 方式 4：多子图（3行2列）
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
# axes 是 shape=(3,2) 的 ndarray

plt.tight_layout()  # 自动调整子图间距
plt.show()
```

### 15.1.4 检查对象层次

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])

print(type(fig))          # <class 'matplotlib.figure.Figure'>
print(type(ax))           # <class 'matplotlib.axes._subplots.AxesSubplot'>
print(fig.get_axes())     # [<AxesSubplot: >]
print(ax.lines)           # [<matplotlib.lines.Line2D object at ...>]
```

---

## 15.2 基本图表

### 15.2.1 折线图（Line Plot）

折线图是展示连续数据趋势的最常用图表。

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 4 * np.pi, 200)

fig, ax = plt.subplots(figsize=(10, 4))

# 绘制多条线，label 用于图例
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')
ax.plot(x, np.sin(x) * np.exp(-0.1 * x), label='衰减正弦', linestyle='--')

ax.set_title("三角函数曲线", fontsize=14)
ax.set_xlabel("x（弧度）")
ax.set_ylabel("函数值")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**折线图常用参数：**

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `color` | 线条颜色 | `'red'`, `'#2196F3'`, `(0.1, 0.5, 0.9)` |
| `linestyle` | 线型 | `'-'`, `'--'`, `'-.'`, `':'` |
| `linewidth` | 线宽 | `1.0`, `2.0` |
| `marker` | 数据点标记 | `'o'`, `'s'`, `'^'`, `'*'` |
| `markersize` | 标记大小 | `6`, `10` |
| `alpha` | 透明度 | `0.0`~`1.0` |

### 15.2.2 散点图（Scatter Plot）

散点图用于展示两个变量之间的分布关系。

```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)

# 生成两类数据
n = 150
class_a = rng.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], n)
class_b = rng.multivariate_normal([4, 4], [[1, -0.3], [-0.3, 1]], n)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：基础散点图
ax = axes[0]
ax.scatter(class_a[:, 0], class_a[:, 1], label='类别 A', alpha=0.6)
ax.scatter(class_b[:, 0], class_b[:, 1], label='类别 B', alpha=0.6)
ax.set_title("二维分类数据分布")
ax.legend()
ax.grid(True, alpha=0.3)

# 右图：用颜色和大小编码额外信息
ax = axes[1]
x = rng.uniform(0, 10, 100)
y = rng.uniform(0, 10, 100)
sizes = rng.uniform(20, 300, 100)    # 点的大小
colors = rng.uniform(0, 1, 100)      # 颜色映射值

sc = ax.scatter(x, y, s=sizes, c=colors, cmap='viridis', alpha=0.7)
fig.colorbar(sc, ax=ax, label='颜色值')
ax.set_title("气泡图（大小+颜色编码）")

plt.tight_layout()
plt.show()
```

### 15.2.3 柱状图（Bar Chart）

柱状图适合对比离散类别的数值大小。

```python
import matplotlib.pyplot as plt
import numpy as np

categories = ['CNN', 'RNN', 'Transformer', 'ResNet', 'BERT']
accuracy = [0.921, 0.887, 0.953, 0.944, 0.968]
params_m = [1.2, 0.8, 3.5, 2.1, 110.0]   # 参数量（百万）

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：垂直柱状图
ax = axes[0]
x = np.arange(len(categories))
bars = ax.bar(x, accuracy, width=0.6, color='steelblue', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=15)
ax.set_ylabel("准确率")
ax.set_title("各模型准确率对比")
ax.set_ylim(0.85, 1.0)

# 在柱顶标注数值
for bar, val in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=9)

# 右图：水平分组柱状图
ax = axes[1]
x = np.arange(3)
models = ['Transformer', 'ResNet', 'BERT']
train_acc = [0.971, 0.958, 0.982]
val_acc   = [0.953, 0.944, 0.968]

width = 0.35
ax.barh(x - width/2, train_acc, height=width, label='训练集', color='#FF7043')
ax.barh(x + width/2, val_acc,   height=width, label='验证集', color='#42A5F5')
ax.set_yticks(x)
ax.set_yticklabels(models)
ax.set_xlabel("准确率")
ax.set_title("训练集 vs 验证集准确率")
ax.legend()
ax.set_xlim(0.93, 1.0)

plt.tight_layout()
plt.show()
```

### 15.2.4 直方图（Histogram）

直方图展示数值数据的频率分布。

```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
data_normal  = rng.normal(0, 1, 1000)
data_bimodal = np.concatenate([rng.normal(-2, 0.8, 500),
                                rng.normal(2, 0.8, 500)])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 左：基础直方图
axes[0].hist(data_normal, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_title("正态分布（bins=30）")
axes[0].set_xlabel("值")
axes[0].set_ylabel("频次")

# 中：密度归一化 + KDE
axes[1].hist(data_normal, bins=30, density=True,
             color='steelblue', edgecolor='white', alpha=0.5, label='直方图')
# 叠加核密度估计曲线
from scipy.stats import gaussian_kde
kde = gaussian_kde(data_normal)
x_range = np.linspace(-4, 4, 200)
axes[1].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
axes[1].set_title("密度归一化 + KDE")
axes[1].legend()

# 右：双峰分布对比
axes[2].hist(data_normal,  bins=40, alpha=0.5, label='单峰', density=True)
axes[2].hist(data_bimodal, bins=40, alpha=0.5, label='双峰', density=True)
axes[2].set_title("分布对比")
axes[2].legend()

plt.tight_layout()
plt.show()
```

---

## 15.3 图表定制

### 15.3.1 标题与轴标签

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x) * x)

# 主标题
ax.set_title("阻尼振荡", fontsize=16, fontweight='bold', pad=15)

# 坐标轴标签（支持 LaTeX 数学公式）
ax.set_xlabel(r"时间 $t$（秒）", fontsize=13)
ax.set_ylabel(r"振幅 $A(t) = \sin(t) \cdot t$", fontsize=13)

# 坐标轴范围
ax.set_xlim(0, 10)
ax.set_ylim(-12, 12)

# 自定义刻度
ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi],
              labels=['0', 'π', '2π', '3π'])
ax.set_yticks([-10, -5, 0, 5, 10])

plt.tight_layout()
plt.show()
```

### 15.3.2 图例（Legend）

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.linspace(0, 10, 200)

# 左图：图例位置和样式
ax = axes[0]
for n in [1, 2, 3]:
    ax.plot(x, np.sin(x * n) / n, label=f'n={n}')

ax.legend(
    loc='upper right',          # 位置：'best'会自动选最优
    fontsize=11,
    framealpha=0.8,             # 图例框透明度
    edgecolor='gray',
    title='参数 n',
    title_fontsize=12
)
ax.set_title("图例位置与样式")

# 右图：图例放在图外
ax = axes[1]
lines = []
for color, style, label in zip(['blue', 'red', 'green'],
                                 ['-', '--', '-.'],
                                 ['模型A', '模型B', '模型C']):
    line, = ax.plot(x, np.sin(x + np.random.rand()),
                    color=color, linestyle=style, label=label)
    lines.append(line)

# bbox_to_anchor 将图例放到坐标系外部
ax.legend(handles=lines,
          bbox_to_anchor=(1.02, 1),
          loc='upper left',
          borderaxespad=0)
ax.set_title("图例置于图外")

plt.tight_layout()
plt.show()
```

### 15.3.3 颜色与样式

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x = np.linspace(0, 2 * np.pi, 100)

# 左上：颜色指定方式
ax = axes[0, 0]
colors = ['red', '#2196F3', (0.2, 0.8, 0.4), '0.7']  # 名称/十六进制/RGB元组/灰度
labels = ['颜色名', '十六进制', 'RGB元组', '灰度值']
for i, (c, lbl) in enumerate(zip(colors, labels)):
    ax.plot(x, np.sin(x + i * 0.5), color=c, linewidth=2, label=lbl)
ax.legend()
ax.set_title("颜色指定方式")

# 右上：线型与标记
ax = axes[0, 1]
styles = [('-', 'o'), ('--', 's'), ('-.', '^'), (':', 'D')]
for i, (ls, mk) in enumerate(styles):
    ax.plot(x[::10], np.sin(x[::10] + i),
            linestyle=ls, marker=mk, markersize=8, label=f'{ls},{mk}')
ax.legend()
ax.set_title("线型与标记组合")

# 左下：颜色映射（colormap）
ax = axes[1, 0]
cmap = plt.get_cmap('plasma')
n_lines = 8
for i in range(n_lines):
    color = cmap(i / n_lines)
    ax.plot(x, np.sin(x + i * np.pi / n_lines), color=color, linewidth=2)
sm = plt.cm.ScalarMappable(cmap='plasma',
                            norm=plt.Normalize(0, n_lines))
plt.colorbar(sm, ax=ax, label='序号')
ax.set_title("颜色映射（plasma）")

# 右下：使用样式表
ax = axes[1, 1]
with plt.style.context('seaborn-v0_8-whitegrid'):
    # 注：style.context 对已创建的 ax 不起作用，仅演示用法
    ax.plot(x, np.sin(x), linewidth=2)
    ax.plot(x, np.cos(x), linewidth=2)
ax.set_title("Seaborn 网格风格（示意）")
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

### 15.3.4 网格、注释与参考线

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 200)
y = np.sin(x) * np.exp(-0.2 * x) + 0.1 * np.random.randn(200)

ax.plot(x, y, 'b-', alpha=0.7, label='信号')

# 水平/垂直参考线
ax.axhline(y=0,   color='black', linewidth=0.8, linestyle='--')
ax.axvline(x=5,   color='gray',  linewidth=0.8, linestyle=':')
ax.axhspan(-0.1, 0.1, alpha=0.1, color='yellow', label='零值区间')

# 注释箭头
peak_idx = np.argmax(y[:50])
ax.annotate(
    f'第一峰值\n({x[peak_idx]:.2f}, {y[peak_idx]:.2f})',
    xy=(x[peak_idx], y[peak_idx]),
    xytext=(x[peak_idx] + 1.5, y[peak_idx] + 0.4),
    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
    fontsize=10, color='red'
)

# 网格（主次刻度分别设置）
ax.grid(True, which='major', linestyle='-',  alpha=0.3)
ax.grid(True, which='minor', linestyle=':', alpha=0.2)
ax.minorticks_on()

ax.set_title("阻尼信号分析", fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
```

### 15.3.5 中文字体配置

在 matplotlib 中显示中文需要指定支持中文的字体：

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# 方式 1：全局设置（推荐放在脚本开头）
mpl.rcParams['font.family'] = 'SimHei'      # Windows: 黑体
# mpl.rcParams['font.family'] = 'STHeiti'   # macOS
# mpl.rcParams['font.family'] = 'WenQuanYi Micro Hei'  # Linux

mpl.rcParams['axes.unicode_minus'] = False  # 修复负号显示为方块的问题

# 方式 2：使用 matplotlib-fontja 或 mplfonts（第三方库）
# pip install mplfonts
# from mplfonts import use_font
# use_font('Noto Serif CJK SC')

# 方式 3：单次临时指定
fig, ax = plt.subplots()
ax.set_title("中文标题", fontproperties='SimHei', fontsize=14)
```

---

## 15.4 子图布局

### 15.4.1 subplot / subplots

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 4 * np.pi, 200)
functions = [
    (np.sin(x),       'sin(x)',       'blue'),
    (np.cos(x),       'cos(x)',       'orange'),
    (np.tan(x[:150]), 'tan(x)',       'green'),
    (np.sin(x)**2,    'sin²(x)',      'red'),
]

# subplots：同时创建多个 Axes
fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                          sharex=True,     # 共享 x 轴
                          sharey=False)

for ax, (y, title, color) in zip(axes.flat, functions):
    ax.plot(x[:len(y)], y, color=color, linewidth=1.5)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)

# 添加统一的 x 标签（Figure 级别）
fig.supxlabel("x（弧度）", fontsize=13)
fig.suptitle("三角函数族", fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()
```

### 15.4.2 GridSpec：非均匀网格布局

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

fig = plt.figure(figsize=(12, 8))

# 创建 3x3 网格，可以用 slice 跨越多格
gs = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.4, wspace=0.3)

ax_main  = fig.add_subplot(gs[0:2, 0:2])  # 左上 2x2 大图
ax_right = fig.add_subplot(gs[0:2, 2])    # 右侧竖长图
ax_bot0  = fig.add_subplot(gs[2, 0])      # 底部三小图
ax_bot1  = fig.add_subplot(gs[2, 1])
ax_bot2  = fig.add_subplot(gs[2, 2])

rng = np.random.default_rng(7)
x = np.linspace(0, 10, 200)

# 主图：散点 + 回归线
ax_main.scatter(rng.uniform(0, 10, 80), rng.uniform(0, 10, 80),
                alpha=0.5, s=40)
ax_main.plot(x, x * 0.8 + 1.5, 'r--', linewidth=2, label='回归线')
ax_main.set_title("散点图（主）", fontsize=13)
ax_main.legend()

# 右侧：水平条形图
labels = ['A', 'B', 'C', 'D', 'E']
vals   = rng.uniform(0.6, 1.0, 5)
ax_right.barh(labels, vals, color='steelblue')
ax_right.set_xlim(0.5, 1.05)
ax_right.set_title("类别分布", fontsize=11)

# 底部三图：不同分布
for ax, name, data in zip(
    [ax_bot0, ax_bot1, ax_bot2],
    ['正态', '均匀', '指数'],
    [rng.normal(5, 1.5, 500),
     rng.uniform(0, 10, 500),
     rng.exponential(2, 500)]
):
    ax.hist(data, bins=20, color='teal', edgecolor='white', alpha=0.8)
    ax.set_title(name, fontsize=10)
    ax.set_yticks([])

fig.suptitle("GridSpec 非均匀布局示例", fontsize=15, fontweight='bold', y=1.01)
plt.show()
```

### 15.4.3 inset_axes：图中图

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

x = np.linspace(0, 10, 500)
y = np.sin(x) * np.exp(-0.15 * x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, 'b-', linewidth=2, label='信号')
ax.set_title("主图 + 局部放大", fontsize=14)
ax.legend()

# 嵌入局部放大图
ax_inset = inset_axes(ax, width="35%", height="40%", loc='upper right')
zoom_x = (2, 4)
mask = (x >= zoom_x[0]) & (x <= zoom_x[1])
ax_inset.plot(x[mask], y[mask], 'b-', linewidth=2)
ax_inset.set_xlim(*zoom_x)
ax_inset.set_title("局部放大", fontsize=9)
ax_inset.tick_params(labelsize=8)
ax_inset.grid(True, alpha=0.3)

# 在主图上标出放大区域
rect = patches.Rectangle((zoom_x[0], y[mask].min() - 0.05),
                           zoom_x[1] - zoom_x[0],
                           y[mask].max() - y[mask].min() + 0.1,
                           linewidth=1, edgecolor='red',
                           facecolor='none', linestyle='--')
ax.add_patch(rect)

plt.tight_layout()
plt.show()
```

### 15.4.4 constrained_layout 与 tight_layout

```python
import matplotlib.pyplot as plt
import numpy as np

# tight_layout：调整子图参数以使子图填充整个图区域（最常用）
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax in axes.flat:
    ax.plot(np.random.randn(50))
    ax.set_title("子图标题示例")
plt.tight_layout(pad=1.5, h_pad=2.0, w_pad=1.0)
plt.show()

# constrained_layout：更精确，推荐搭配 colorbar 使用
fig, axes = plt.subplots(2, 3, figsize=(12, 7),
                          layout='constrained')
for ax in axes.flat:
    im = ax.imshow(np.random.rand(10, 10), cmap='viridis')
    fig.colorbar(im, ax=ax)
plt.show()
```

---

## 15.5 图表保存与显示

### 15.5.1 保存图表

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)))
ax.set_title("保存示例")

# 基础保存（根据扩展名自动选择格式）
fig.savefig("output.png")                      # PNG（默认 DPI=100）
fig.savefig("output.pdf")                      # PDF（矢量格式，适合论文）
fig.savefig("output.svg")                      # SVG（矢量格式，适合网页）

# 高质量保存配置
fig.savefig(
    "high_quality.png",
    dpi=300,                   # 分辨率（论文要求通常 300 DPI）
    bbox_inches='tight',       # 自动裁剪多余空白
    facecolor='white',         # 背景色（默认透明）
    transparent=False          # 是否透明背景
)

# 保存到内存（用于 Web 服务）
import io
buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)
image_bytes = buf.read()
print(f"图片大小：{len(image_bytes) / 1024:.1f} KB")

plt.close(fig)  # 释放内存，脚本中批量绘图时必须调用
```

### 15.5.2 支持的格式与选择建议

| 格式 | 特点 | 适用场景 |
|------|------|----------|
| PNG | 无损位图，支持透明度 | 报告、幻灯片、网页 |
| PDF | 矢量格式，文字可选中 | 学术论文（LaTeX 嵌入） |
| SVG | 矢量格式，可在浏览器缩放 | 交互式网页 |
| EPS | 矢量格式（旧标准） | 老版 LaTeX 期刊 |
| JPG | 有损压缩，不支持透明 | 照片类内容（不推荐用于图表） |

### 15.5.3 后端（Backend）配置

```python
import matplotlib
import matplotlib.pyplot as plt

# 查看当前后端
print(matplotlib.get_backend())

# 非交互环境（服务器、CI）必须使用 Agg
matplotlib.use('Agg')          # 必须在 import pyplot 之前调用

# 或通过环境变量
# export MPLBACKEND=Agg

# Jupyter Notebook 专用后端（交互式）
# %matplotlib inline          # 静态图片嵌入
# %matplotlib widget          # 交互式（需安装 ipympl）
```

### 15.5.4 批量生成图表的最佳实践

```python
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

def plot_and_save(data: np.ndarray, title: str, filename: str) -> None:
    """封装绘图与保存，确保资源释放。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)   # 关键：批量绘图必须关闭，否则内存泄漏

for i in range(10):
    data = np.sin(np.linspace(0, 10, 200) * (i + 1))
    plot_and_save(data, f"频率 {i+1}", f"freq_{i+1:02d}.png")

print("所有图表已保存。")
```

---

## 本章小结

| 知识点 | 核心 API | 注意事项 |
|--------|----------|----------|
| Figure/Axes 创建 | `plt.subplots(figsize, dpi)` | 优先使用面向对象接口 |
| 折线图 | `ax.plot(x, y, color, linestyle, marker)` | `linewidth` 调整视觉重量 |
| 散点图 | `ax.scatter(x, y, s, c, cmap, alpha)` | `s` 和 `c` 可以是数组 |
| 柱状图 | `ax.bar / ax.barh` | `width` 控制间距，记得加 edgecolor |
| 直方图 | `ax.hist(data, bins, density)` | `density=True` 做概率密度 |
| 标题/标签 | `ax.set_title/xlabel/ylabel` | 支持 LaTeX `r'$...$'` |
| 图例 | `ax.legend(loc, bbox_to_anchor)` | `'best'` 自动定位 |
| 子图布局 | `subplots(sharex/sharey)` / `GridSpec` | 复杂布局用 GridSpec |
| 保存 | `fig.savefig(path, dpi, bbox_inches)` | 批量必须 `plt.close(fig)` |
| 中文 | `mpl.rcParams['font.family']` | 同时设置 `unicode_minus` |

---

## 深度学习应用：训练曲线可视化

训练过程监控是深度学习工程实践的核心环节。本节展示如何用 Matplotlib 构建完整的训练可视化面板。

### 应用一：模拟训练数据生成

```python
import numpy as np

def simulate_training(
    n_epochs: int = 50,
    initial_lr: float = 0.01,
    seed: int = 42
) -> dict:
    """
    模拟一个带有学习率衰减的分类模型训练过程。
    返回包含各种指标的字典。
    """
    rng = np.random.default_rng(seed)
    epochs = np.arange(1, n_epochs + 1)

    # 学习率：前10轮线性 warmup，之后余弦衰减
    warmup = 10
    lr = np.where(
        epochs <= warmup,
        initial_lr * epochs / warmup,
        initial_lr * 0.5 * (1 + np.cos(np.pi * (epochs - warmup) / (n_epochs - warmup)))
    )

    # 损失：指数衰减 + 噪声
    noise_scale = 0.02
    train_loss = 2.5 * np.exp(-0.08 * epochs) + 0.3 + rng.normal(0, noise_scale, n_epochs)
    val_loss   = 2.5 * np.exp(-0.07 * epochs) + 0.4 + rng.normal(0, noise_scale * 1.5, n_epochs)

    # 准确率：从低到高收敛
    train_acc = 1 - 0.9 * np.exp(-0.09 * epochs) + rng.normal(0, 0.005, n_epochs)
    val_acc   = 1 - 0.9 * np.exp(-0.08 * epochs) + rng.normal(0, 0.008, n_epochs)
    train_acc = np.clip(train_acc, 0, 1)
    val_acc   = np.clip(val_acc, 0, 1)

    # 梯度范数（用于监控梯度爆炸/消失）
    grad_norm = 2.0 * np.exp(-0.05 * epochs) + rng.lognormal(0, 0.3, n_epochs) * 0.3

    return {
        'epochs': epochs,
        'lr': lr,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'grad_norm': grad_norm,
    }

history = simulate_training(n_epochs=80)
```

### 应用二：完整训练监控面板

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def plot_training_dashboard(history: dict, save_path: str | None = None) -> None:
    """
    绘制深度学习训练监控仪表盘，包含：
    - Loss 曲线（训练集 + 验证集）
    - Accuracy 曲线（训练集 + 验证集）
    - 学习率变化曲线
    - 梯度范数曲线
    """
    epochs    = history['epochs']
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = history['val_loss'].min()
    best_val_acc  = history['val_acc'][np.argmin(history['val_loss'])]

    # ---- 布局：2x2 + 顶部标题 ----
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
    axes = {
        'loss':      fig.add_subplot(gs[0, 0]),
        'acc':       fig.add_subplot(gs[0, 1]),
        'lr':        fig.add_subplot(gs[1, 0]),
        'grad':      fig.add_subplot(gs[1, 1]),
    }

    # ---- 通用样式 ----
    TRAIN_COLOR = '#1565C0'   # 深蓝
    VAL_COLOR   = '#C62828'   # 深红
    LR_COLOR    = '#2E7D32'   # 深绿
    GRAD_COLOR  = '#6A1B9A'   # 紫色

    def style_ax(ax, title):
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    # ---- 1. Loss 曲线 ----
    ax = axes['loss']
    ax.plot(epochs, history['train_loss'], color=TRAIN_COLOR,
            linewidth=2, alpha=0.9, label='训练 Loss')
    ax.plot(epochs, history['val_loss'],   color=VAL_COLOR,
            linewidth=2, alpha=0.9, label='验证 Loss', linestyle='--')

    # 标注最佳点
    ax.axvline(best_epoch, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax.scatter([best_epoch], [best_val_loss], s=80, color=VAL_COLOR,
               zorder=5, label=f'最优 epoch={best_epoch}')
    ax.annotate(f'  best={best_val_loss:.3f}',
                xy=(best_epoch, best_val_loss),
                fontsize=9, color=VAL_COLOR)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9, loc='upper right')
    style_ax(ax, "Loss 曲线")

    # ---- 2. Accuracy 曲线 ----
    ax = axes['acc']
    ax.plot(epochs, history['train_acc'] * 100, color=TRAIN_COLOR,
            linewidth=2, alpha=0.9, label='训练 Acc')
    ax.plot(epochs, history['val_acc']   * 100, color=VAL_COLOR,
            linewidth=2, alpha=0.9, label='验证 Acc', linestyle='--')

    # 阴影区间：填充训练/验证之间的过拟合区域
    gap = history['train_acc'] - history['val_acc']
    overfit_mask = gap > 0.02
    if overfit_mask.any():
        ax.fill_between(epochs,
                        history['train_acc'] * 100,
                        history['val_acc']   * 100,
                        where=overfit_mask,
                        alpha=0.15, color='orange', label='过拟合区间')

    ax.axvline(best_epoch, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax.scatter([best_epoch], [best_val_acc * 100], s=80, color=VAL_COLOR, zorder=5)
    ax.annotate(f'  {best_val_acc*100:.1f}%',
                xy=(best_epoch, best_val_acc * 100),
                fontsize=9, color=VAL_COLOR)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(50, 105)
    ax.legend(fontsize=9, loc='lower right')
    style_ax(ax, "Accuracy 曲线")

    # ---- 3. 学习率曲线 ----
    ax = axes['lr']
    ax.semilogy(epochs, history['lr'], color=LR_COLOR, linewidth=2)

    # 标注 warmup 阶段
    warmup_end = np.argmax(history['lr']) + 1
    ax.axvspan(1, warmup_end, alpha=0.1, color='yellow', label=f'Warmup ({warmup_end} epochs)')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate（对数轴）")
    ax.legend(fontsize=9)
    style_ax(ax, "学习率变化（Warmup + 余弦衰减）")

    # ---- 4. 梯度范数 ----
    ax = axes['grad']
    # 原始值（半透明）+ 滑动平均（实线）
    window = 5
    smooth = np.convolve(history['grad_norm'],
                         np.ones(window) / window, mode='valid')
    smooth_epochs = epochs[window - 1:]

    ax.plot(epochs, history['grad_norm'], color=GRAD_COLOR,
            alpha=0.25, linewidth=1, label='原始')
    ax.plot(smooth_epochs, smooth, color=GRAD_COLOR,
            linewidth=2, label=f'滑动均值 (w={window})')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.legend(fontsize=9)
    style_ax(ax, "梯度范数监控")

    # ---- 总标题与信息框 ----
    info = (f"总轮次: {len(epochs)}  |  "
            f"最优 epoch: {best_epoch}  |  "
            f"最优 val_loss: {best_val_loss:.4f}  |  "
            f"最优 val_acc: {best_val_acc*100:.2f}%")
    fig.suptitle(f"训练监控仪表盘\n{info}",
                 fontsize=13, fontweight='bold', y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"仪表盘已保存：{save_path}")

    plt.tight_layout()
    plt.show()


# 运行
history = simulate_training(n_epochs=80)
plot_training_dashboard(history, save_path="training_dashboard.png")
```

### 应用三：多模型对比曲线

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison(
    histories: dict[str, dict],
    metric: str = 'val_loss'
) -> None:
    """
    对比多个模型在同一指标上的训练曲线。

    参数
    ----
    histories : {模型名称: history字典}
    metric    : 要对比的指标键名
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')

    for i, (name, hist) in enumerate(histories.items()):
        y = hist[metric]
        epochs = hist['epochs']
        color  = cmap(i)

        # 绘制曲线
        ax.plot(epochs, y, color=color, linewidth=2, label=name, alpha=0.85)

        # 标注最优点
        best_idx = np.argmin(y) if 'loss' in metric else np.argmax(y)
        ax.scatter([epochs[best_idx]], [y[best_idx]],
                   color=color, s=80, zorder=5)
        ax.annotate(f'{y[best_idx]:.3f}',
                    xy=(epochs[best_idx], y[best_idx]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color=color)

    metric_label = {
        'val_loss':  '验证 Loss',
        'val_acc':   '验证 Accuracy',
        'train_loss':'训练 Loss',
    }.get(metric, metric)

    ax.set_title(f"多模型对比：{metric_label}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()


# 模拟三个不同配置的模型
model_histories = {
    'baseline (lr=0.01)':     simulate_training(n_epochs=80, initial_lr=0.01, seed=0),
    'high_lr   (lr=0.05)':    simulate_training(n_epochs=80, initial_lr=0.05, seed=1),
    'low_lr    (lr=0.001)':   simulate_training(n_epochs=80, initial_lr=0.001, seed=2),
}
plot_model_comparison(model_histories, metric='val_loss')
plot_model_comparison(model_histories, metric='val_acc')
```

### 应用四：实时训练动画（可选）

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate_training(history: dict, interval: int = 50) -> animation.FuncAnimation:
    """
    用动画方式回放训练曲线（适用于 Jupyter Notebook）。
    在脚本中运行时需要设置 plt.show(block=True)。
    """
    epochs     = history['epochs']
    train_loss = history['train_loss']
    val_loss   = history['val_loss']

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0, train_loss[0] * 1.1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练动画")
    ax.grid(True, alpha=0.3)

    line_train, = ax.plot([], [], 'b-', linewidth=2, label='训练 Loss')
    line_val,   = ax.plot([], [], 'r--', linewidth=2, label='验证 Loss')
    epoch_text  = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                          fontsize=11, verticalalignment='top')
    ax.legend()

    def update(frame):
        n = frame + 1
        line_train.set_data(epochs[:n], train_loss[:n])
        line_val.set_data(epochs[:n], val_loss[:n])
        epoch_text.set_text(f'Epoch: {n}/{len(epochs)}')
        return line_train, line_val, epoch_text

    ani = animation.FuncAnimation(fig, update, frames=len(epochs),
                                  interval=interval, blit=True)
    return ani

# Jupyter 中：
# ani = animate_training(history)
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# 保存为 GIF：
# ani.save('training.gif', writer='pillow', fps=20)
```

---

## 练习题

### 基础题

**练习 15-1**：绘制数学函数图像

使用 Matplotlib 在同一个坐标系内绘制以下三条曲线（x 范围：[0, 2π]），要求：
- 每条曲线使用不同颜色和线型
- 添加图例（使用 LaTeX 数学公式）
- 添加标题、x 轴标签、y 轴标签
- 在 y=0 处添加黑色水平参考线

函数：
1. $f_1(x) = \sin(x)$
2. $f_2(x) = \sin(2x) / 2$
3. $f_3(x) = \sin(3x) / 3$

---

**练习 15-2**：学生成绩分布可视化

给定以下数据（50名学生的语文、数学、英语成绩）：

```python
import numpy as np
rng = np.random.default_rng(2024)
chinese = rng.normal(72, 12, 50).clip(40, 100)
math    = rng.normal(68, 15, 50).clip(30, 100)
english = rng.normal(75, 10, 50).clip(50, 100)
```

创建一个 1×3 的子图布局，分别用直方图展示三科成绩的分布，要求：
- 每个直方图设置 `bins=15`，半透明填充
- 用垂直线标出每科的平均分
- 在图例中显示平均分数值

---

### 进阶题

**练习 15-3**：交互式热力图（相关性矩阵）

使用 `ax.imshow()` 绘制如下特征相关性矩阵的热力图：

```python
import numpy as np
features = ['特征A', '特征B', '特征C', '特征D', '特征E']
corr = np.array([
    [1.00,  0.82, -0.31,  0.55,  0.12],
    [0.82,  1.00, -0.25,  0.48,  0.08],
    [-0.31,-0.25,  1.00, -0.41, -0.19],
    [0.55,  0.48, -0.41,  1.00,  0.33],
    [0.12,  0.08, -0.19,  0.33,  1.00],
])
```

要求：
- 使用 `RdBu_r` 颜色映射，范围 [-1, 1]
- 在每个格子中显示相关系数值（两位小数）
- 添加 colorbar 和适当标题
- 对角线文字用粗体显示

---

**练习 15-4**：使用 GridSpec 创建综合报告图

构建一个模拟分类任务的报告图，使用 GridSpec 布局，包含：
- 顶部（占 2 行）：绘制两类样本的散点分布，使用不同颜色区分，并绘制一条决策边界直线
- 底部左（1 行）：绘制类别 A 特征的直方图
- 底部右（1 行）：绘制类别 B 特征的直方图

数据自行使用 `np.random.default_rng` 生成。

---

### 挑战题

**练习 15-5**：训练曲线可视化函数封装

封装一个通用函数 `plot_training_history(history, config)`，满足以下要求：

1. `history` 接受字典，键包含 `'train_loss'`、`'val_loss'`（必须）以及可选的 `'train_acc'`、`'val_acc'`、`'lr'`
2. `config` 字典支持以下配置项：
   - `title`：图表主标题
   - `figsize`：图表尺寸
   - `save_path`：保存路径（None 则不保存）
   - `smooth_window`：损失曲线的平滑窗口大小（默认 1，即不平滑）
3. 函数根据 history 中有哪些键，自动决定绘制 2、3 或 4 个子图
4. 最优 epoch（val_loss 最小处）需在所有子图中用垂直虚线标出
5. 函数末尾关闭 Figure 前返回 `fig` 对象

---

## 练习答案

### 答案 15-1

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 400)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x, np.sin(x),         color='steelblue', linestyle='-',
        linewidth=2, label=r'$f_1(x) = \sin(x)$')
ax.plot(x, np.sin(2 * x) / 2, color='tomato',    linestyle='--',
        linewidth=2, label=r'$f_2(x) = \sin(2x)/2$')
ax.plot(x, np.sin(3 * x) / 3, color='seagreen',  linestyle='-.',
        linewidth=2, label=r'$f_3(x) = \sin(3x)/3$')

ax.axhline(0, color='black', linewidth=0.8, linestyle=':')

ax.set_title(r"傅里叶分量：$\sin(nx)/n$", fontsize=14)
ax.set_xlabel(r"$x$（弧度）", fontsize=12)
ax.set_ylabel("函数值", fontsize=12)
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
              labels=['0', 'π/2', 'π', '3π/2', '2π'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 答案 15-2

```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(2024)
chinese = rng.normal(72, 12, 50).clip(40, 100)
math    = rng.normal(68, 15, 50).clip(30, 100)
english = rng.normal(75, 10, 50).clip(50, 100)

subjects = [('语文', chinese, '#5C85D6'),
            ('数学', math,    '#E07B54'),
            ('英语', english, '#59B077')]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, (name, scores, color) in zip(axes, subjects):
    mean_score = scores.mean()
    ax.hist(scores, bins=15, color=color, alpha=0.7,
            edgecolor='white', label=f'样本分布')
    ax.axvline(mean_score, color='darkred', linewidth=2,
               linestyle='--', label=f'均值={mean_score:.1f}')
    ax.set_title(f"{name}成绩分布", fontsize=12)
    ax.set_xlabel("分数")
    ax.set_ylabel("人数")
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

fig.suptitle("三科成绩分布对比（n=50）", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### 答案 15-3

```python
import matplotlib.pyplot as plt
import numpy as np

features = ['特征A', '特征B', '特征C', '特征D', '特征E']
corr = np.array([
    [1.00,  0.82, -0.31,  0.55,  0.12],
    [0.82,  1.00, -0.25,  0.48,  0.08],
    [-0.31,-0.25,  1.00, -0.41, -0.19],
    [0.55,  0.48, -0.41,  1.00,  0.33],
    [0.12,  0.08, -0.19,  0.33,  1.00],
])

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='相关系数')

n = len(features)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(features, fontsize=11)
ax.set_yticklabels(features, fontsize=11)

for i in range(n):
    for j in range(n):
        val = corr[i, j]
        text_color = 'white' if abs(val) > 0.6 else 'black'
        weight = 'bold' if i == j else 'normal'
        ax.text(j, i, f'{val:.2f}',
                ha='center', va='center',
                fontsize=10, color=text_color, fontweight=weight)

ax.set_title("特征相关性矩阵", fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()
```

### 答案 15-4

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

rng = np.random.default_rng(99)
class_a = rng.multivariate_normal([2, 3], [[1.2, 0.4],[0.4, 0.8]], 120)
class_b = rng.multivariate_normal([5, 6], [[0.9, -0.3],[-0.3, 1.1]], 100)

fig = plt.figure(figsize=(10, 9))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

ax_scatter = fig.add_subplot(gs[0:2, :])   # 顶部占两行全宽
ax_hist_a  = fig.add_subplot(gs[2, 0])
ax_hist_b  = fig.add_subplot(gs[2, 1])

# 散点图 + 决策边界
ax_scatter.scatter(class_a[:, 0], class_a[:, 1],
                   alpha=0.6, label='类别 A', s=40, color='#1976D2')
ax_scatter.scatter(class_b[:, 0], class_b[:, 1],
                   alpha=0.6, label='类别 B', s=40, color='#E53935')

x_line = np.linspace(0, 8, 100)
ax_scatter.plot(x_line, x_line + 0.2, 'k--', linewidth=1.5,
                label='决策边界（示意）')
ax_scatter.set_title("二维分类散点图与决策边界", fontsize=12)
ax_scatter.legend()
ax_scatter.grid(True, alpha=0.3)

# 直方图
for ax, data, name, color in [
    (ax_hist_a, class_a[:, 0], '类别 A（x特征）', '#1976D2'),
    (ax_hist_b, class_b[:, 0], '类别 B（x特征）', '#E53935'),
]:
    ax.hist(data, bins=15, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(data.mean(), color='black', linewidth=1.5,
               linestyle='--', label=f'均值={data.mean():.2f}')
    ax.set_title(name, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

fig.suptitle("分类任务综合报告", fontsize=14, fontweight='bold')
plt.show()
```

### 答案 15-5

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Any

def plot_training_history(
    history: dict[str, np.ndarray],
    config: dict[str, Any] | None = None
) -> plt.Figure:
    """
    通用训练曲线可视化函数。

    参数
    ----
    history : 包含训练指标的字典
        必须包含 'train_loss', 'val_loss'
        可选包含 'train_acc', 'val_acc', 'lr'
    config  : 可选配置字典
        title          - 图表主标题 (str)
        figsize        - (width, height) (tuple)
        save_path      - 保存路径 (str | None)
        smooth_window  - 平滑窗口 (int, 默认 1)

    返回
    ----
    fig : matplotlib.figure.Figure
    """
    if config is None:
        config = {}

    # 配置项解析
    title         = config.get('title', '训练历史曲线')
    figsize       = config.get('figsize', (12, 8))
    save_path     = config.get('save_path', None)
    smooth_window = max(1, int(config.get('smooth_window', 1)))

    # 确定要绘制哪些子图
    has_acc = 'train_acc' in history and 'val_acc' in history
    has_lr  = 'lr' in history
    n_plots = 2 + int(has_acc) + int(has_lr)   # 至少2个（loss必须）

    epochs = np.arange(1, len(history['train_loss']) + 1)
    best_epoch = int(np.argmin(history['val_loss'])) + 1

    # 计算平滑函数
    def smooth(arr: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode='valid')

    # 创建布局
    cols = 2 if n_plots > 2 else 1
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=figsize,
                              squeeze=False)
    ax_list = axes.flat

    def style(ax: plt.Axes, ylabel: str) -> None:
        ax.axvline(best_epoch, color='gray', linewidth=1,
                   linestyle=':', alpha=0.7, label=f'最优 epoch={best_epoch}')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=9)

    plot_idx = 0

    # 子图1：Loss
    ax = next(ax_list)
    sm_epochs = epochs[smooth_window - 1:]
    ax.plot(epochs, history['train_loss'],
            alpha=0.3 if smooth_window > 1 else 0.9,
            color='steelblue', linewidth=1)
    ax.plot(sm_epochs, smooth(history['train_loss'], smooth_window),
            color='steelblue', linewidth=2, label='训练 Loss')
    ax.plot(epochs, history['val_loss'],
            alpha=0.3 if smooth_window > 1 else 0.9,
            color='tomato', linewidth=1)
    ax.plot(sm_epochs, smooth(history['val_loss'], smooth_window),
            color='tomato', linewidth=2, linestyle='--', label='验证 Loss')
    ax.set_title("Loss 曲线")
    style(ax, "Loss")
    plot_idx += 1

    # 子图2（可选）：Accuracy
    if has_acc:
        ax = next(ax_list)
        ax.plot(epochs, history['train_acc'] * 100, color='steelblue',
                linewidth=2, label='训练 Acc')
        ax.plot(epochs, history['val_acc'] * 100, color='tomato',
                linewidth=2, linestyle='--', label='验证 Acc')
        ax.set_title("Accuracy 曲线")
        style(ax, "Accuracy (%)")
        plot_idx += 1

    # 子图3（可选）：Learning Rate
    if has_lr:
        ax = next(ax_list)
        ax.semilogy(epochs, history['lr'], color='seagreen', linewidth=2, label='LR')
        ax.set_title("学习率变化")
        style(ax, "Learning Rate")
        plot_idx += 1

    # 子图4（如果有剩余）：val_loss 滑动均值放大
    if plot_idx < n_plots:
        ax = next(ax_list)
        ax.plot(sm_epochs, smooth(history['val_loss'], smooth_window),
                color='tomato', linewidth=2, label='验证 Loss（平滑）')
        ax.set_title("验证 Loss 放大视图")
        style(ax, "Val Loss")

    # 隐藏多余的子图
    for ax in ax_list:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    plt.show()
    return fig


# 测试
history = simulate_training(n_epochs=60)
fig = plot_training_history(
    history,
    config={
        'title': '通用训练历史可视化',
        'figsize': (12, 8),
        'smooth_window': 5,
        'save_path': None,
    }
)
```

---

> **下一章预告**：第16章将介绍 Seaborn——基于 Matplotlib 的统计可视化库，它提供更高级的 API，能用更少的代码生成更精美的统计图表，并与 Pandas DataFrame 无缝集成。
