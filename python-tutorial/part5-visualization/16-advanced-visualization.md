# 第16章：高级可视化

## 学习目标

完成本章学习后，你将能够：

1. 使用 Seaborn 创建统计图表，包括分布图、回归图和分类图
2. 绘制热力图与关联矩阵，分析变量间的关系
3. 使用 Matplotlib 和 Plotly 创建三维可视化图形
4. 使用 `FuncAnimation` 制作动画，并用 Plotly 创建交互式图表
5. 掌握可视化最佳实践，能够针对深度学习任务可视化 CNN 特征图、注意力权重和 t-SNE 降维结果

---

## 16.1 Seaborn 统计可视化

Seaborn 是基于 Matplotlib 的高级统计可视化库，内置了丰富的主题和调色板，能用简洁的 API 绘制出美观的统计图表。

### 16.1.1 安装与基础设置

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置主题与调色板
sns.set_theme(style="whitegrid", palette="muted")

# 查看内置数据集
print(sns.get_dataset_names()[:10])
```

### 16.1.2 分布图

分布图用于展示数据的概率分布，是探索性数据分析的第一步。

```python
# 载入示例数据
tips = sns.load_dataset("tips")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 直方图 + KDE
sns.histplot(tips["total_bill"], kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("直方图 + 核密度估计")

# 箱线图
sns.boxplot(x="day", y="total_bill", data=tips, ax=axes[1], palette="Set2")
axes[1].set_title("按星期分组的消费箱线图")

# 小提琴图（结合箱线图与KDE）
sns.violinplot(x="day", y="total_bill", data=tips, ax=axes[2],
               palette="Set3", inner="box")
axes[2].set_title("小提琴图")

plt.tight_layout()
plt.savefig("distribution_plots.png", dpi=150)
plt.show()
```

### 16.1.3 回归图

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 线性回归散点图
sns.regplot(x="total_bill", y="tip", data=tips, ax=axes[0],
            scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
axes[0].set_title("消费金额 vs 小费（线性回归）")

# 按类别分组的回归图
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
           height=5, aspect=1.2)
plt.title("吸烟者 vs 非吸烟者的消费回归")
plt.tight_layout()
plt.show()
```

### 16.1.4 分类图

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 条形图（带误差线）
sns.barplot(x="day", y="total_bill", data=tips, ax=axes[0, 0],
            palette="Blues_d", capsize=0.1)
axes[0, 0].set_title("各天平均消费（含95%置信区间）")

# 点图
sns.pointplot(x="day", y="tip", hue="sex", data=tips, ax=axes[0, 1],
              dodge=True, palette="Set1")
axes[0, 1].set_title("各天小费（按性别）")

# 计数图
sns.countplot(x="day", hue="sex", data=tips, ax=axes[1, 0], palette="pastel")
axes[1, 0].set_title("各天就餐人数（按性别）")

# 蜂群图（所有数据点均可见）
sns.swarmplot(x="day", y="total_bill", data=tips, ax=axes[1, 1],
              size=3, palette="muted")
axes[1, 1].set_title("蜂群图（每笔消费都可见）")

plt.tight_layout()
plt.savefig("categorical_plots.png", dpi=150)
plt.show()
```

### 16.1.5 成对关系图（Pair Plot）

```python
iris = sns.load_dataset("iris")

# pairplot 展示所有特征两两关系
g = sns.pairplot(iris, hue="species", diag_kind="kde",
                 plot_kws={"alpha": 0.6}, palette="Set2")
g.figure.suptitle("鸢尾花数据集成对关系图", y=1.02, fontsize=14)
plt.savefig("pairplot.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## 16.2 热力图与关联图

热力图通过颜色深浅展示矩阵数据，是分析特征相关性、混淆矩阵等的利器。

### 16.2.1 相关系数热力图

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 使用鸢尾花数据集
iris = sns.load_dataset("iris")
numeric_cols = iris.select_dtypes(include=np.number)

# 计算相关系数矩阵
corr_matrix = numeric_cols.corr()

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制热力图
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,          # 在格子中显示数值
    fmt=".2f",           # 小数点后两位
    cmap="RdYlGn",       # 红-黄-绿色阶
    vmin=-1, vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    ax=ax
)
ax.set_title("鸢尾花特征相关系数矩阵（下三角）", fontsize=13)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()
```

### 16.2.2 混淆矩阵热力图

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 训练简单分类器
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=42
)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10),
    ax=ax
)
ax.set_xlabel("预测标签", fontsize=12)
ax.set_ylabel("真实标签", fontsize=12)
ax.set_title("手写数字分类混淆矩阵", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
```

### 16.2.3 聚类热力图（Clustermap）

聚类热力图在绘制热力图的同时，对行和列进行层次聚类排序。

```python
# 使用鸟类数据（这里用随机数模拟基因表达矩阵）
np.random.seed(42)
n_genes, n_samples = 30, 20

# 模拟两组样本（正常 vs 疾病）
group_a = np.random.randn(n_genes, n_samples // 2) + np.tile(
    np.sin(np.linspace(0, 3, n_genes)), (n_samples // 2, 1)).T
group_b = np.random.randn(n_genes, n_samples // 2) + np.tile(
    np.cos(np.linspace(0, 3, n_genes)), (n_samples // 2, 1)).T

data_matrix = np.hstack([group_a, group_b])
df = pd.DataFrame(
    data_matrix,
    index=[f"Gene_{i:02d}" for i in range(n_genes)],
    columns=[f"Normal_{i}" for i in range(n_samples // 2)] +
            [f"Disease_{i}" for i in range(n_samples // 2)]
)

# 列颜色标注
col_colors = ["#2196F3"] * (n_samples // 2) + ["#F44336"] * (n_samples // 2)

g = sns.clustermap(
    df,
    col_colors=col_colors,
    cmap="RdBu_r",
    center=0,
    figsize=(12, 10),
    dendrogram_ratio=0.15,
    cbar_pos=(0.02, 0.8, 0.03, 0.15)
)
g.figure.suptitle("基因表达聚类热力图", y=1.01, fontsize=14)
plt.savefig("clustermap.png", dpi=150, bbox_inches="tight")
plt.show()
```

### 16.2.4 自定义注解与颜色映射

```python
# 展示多种 colormap 的效果对比
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
colormaps = ["viridis", "plasma", "RdBu_r", "coolwarm", "YlOrRd", "Blues"]

data = np.random.randn(8, 8)

for ax, cmap in zip(axes.flat, colormaps):
    sns.heatmap(data, ax=ax, cmap=cmap, center=0,
                annot=False, cbar=True, square=True)
    ax.set_title(f"cmap='{cmap}'", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("常用颜色映射对比", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("colormap_comparison.png", dpi=150)
plt.show()
```

---

## 16.3 3D 可视化

三维可视化能展示数据的空间分布，常用于展示损失曲面、三维特征空间等场景。

### 16.3.1 Matplotlib 3D 基础

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

fig = plt.figure(figsize=(14, 5))

# ---- 3D 散点图 ----
ax1 = fig.add_subplot(131, projection="3d")
np.random.seed(0)
n = 200
x = np.random.randn(n)
y = np.random.randn(n)
z = x * 0.5 + y * 0.3 + np.random.randn(n) * 0.5
colors = z  # 用 z 值着色

sc = ax1.scatter(x, y, z, c=colors, cmap="viridis", alpha=0.7, s=20)
fig.colorbar(sc, ax=ax1, shrink=0.5)
ax1.set_title("3D 散点图")
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

# ---- 3D 曲面图 ----
ax2 = fig.add_subplot(132, projection="3d")
x_lin = np.linspace(-3, 3, 60)
y_lin = np.linspace(-3, 3, 60)
X, Y = np.meshgrid(x_lin, y_lin)
Z = np.sin(np.sqrt(X**2 + Y**2))

surf = ax2.plot_surface(X, Y, Z, cmap="coolwarm", alpha=0.85,
                         rstride=1, cstride=1)
fig.colorbar(surf, ax=ax2, shrink=0.5)
ax2.set_title("3D 曲面图：sin(√(x²+y²))")
ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

# ---- 3D 等高线图 ----
ax3 = fig.add_subplot(133, projection="3d")
ax3.contour3D(X, Y, Z, 50, cmap="plasma")
ax3.set_title("3D 等高线图")
ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")

plt.tight_layout()
plt.savefig("3d_basic.png", dpi=150)
plt.show()
```

### 16.3.2 损失曲面可视化

展示神经网络损失函数的曲面形状是深度学习可视化的重要应用。

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def loss_surface(w1, w2):
    """模拟一个带有鞍点的损失曲面"""
    return 0.5 * w1**2 - 0.5 * w2**2 + 0.1 * (w1**4 + w2**4)

w1 = np.linspace(-3, 3, 100)
w2 = np.linspace(-3, 3, 100)
W1, W2 = np.meshgrid(w1, w2)
L = loss_surface(W1, W2)

fig = plt.figure(figsize=(14, 6))

# 3D 曲面
ax1 = fig.add_subplot(121, projection="3d")
surf = ax1.plot_surface(W1, W2, L, cmap="RdYlGn_r", alpha=0.8,
                         rstride=2, cstride=2)
fig.colorbar(surf, ax=ax1, shrink=0.5, label="Loss")
ax1.set_title("损失曲面（含鞍点）")
ax1.set_xlabel("权重 w₁"); ax1.set_ylabel("权重 w₂"); ax1.set_zlabel("Loss")

# 2D 等高线俯视图
ax2 = fig.add_subplot(122)
contour = ax2.contourf(W1, W2, L, levels=50, cmap="RdYlGn_r")
fig.colorbar(contour, ax=ax2, label="Loss")
ax2.contour(W1, W2, L, levels=20, colors="black", linewidths=0.5, alpha=0.4)
ax2.set_title("损失曲面俯视图（等高线）")
ax2.set_xlabel("权重 w₁"); ax2.set_ylabel("权重 w₂")
# 标注鞍点
ax2.plot(0, 0, "r*", markersize=15, label="鞍点 (0,0)")
ax2.legend()

plt.tight_layout()
plt.savefig("loss_surface.png", dpi=150)
plt.show()
```

### 16.3.3 Plotly 交互式 3D 图表

Plotly 生成可在浏览器中旋转、缩放的交互式 3D 图表。

```python
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# 3D 曲面（Plotly）
x = np.linspace(-3, 3, 80)
y = np.linspace(-3, 3, 80)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.1 * (X**2 + Y**2))

fig = go.Figure(data=[go.Surface(
    x=X, y=Y, z=Z,
    colorscale="Viridis",
    contours={
        "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project_z": True}
    }
)])
fig.update_layout(
    title="交互式 3D 曲面（衰减正弦波）",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
    ),
    width=700, height=500
)
fig.write_html("interactive_surface.html")
fig.show()

# 3D 散点图（Plotly）—— 鸢尾花数据
from sklearn.datasets import load_iris
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris["species"] = [iris.target_names[t] for t in iris.target]

fig2 = px.scatter_3d(
    df_iris,
    x="sepal length (cm)", y="sepal width (cm)", z="petal length (cm)",
    color="species",
    symbol="species",
    title="鸢尾花数据集 3D 散点图",
    opacity=0.8
)
fig2.update_traces(marker=dict(size=5))
fig2.write_html("iris_3d_scatter.html")
fig2.show()
```

---

## 16.4 动画与交互（FuncAnimation）

### 16.4.1 FuncAnimation 基础

`matplotlib.animation.FuncAnimation` 通过重复调用更新函数来生成逐帧动画。

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, 4 * np.pi)
ax.set_ylim(-1.5, 1.5)
ax.set_title("正弦波动画")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.grid(True, alpha=0.3)

x = np.linspace(0, 4 * np.pi, 300)
line, = ax.plot([], [], lw=2, color="royalblue")
time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes, fontsize=12)

def init():
    line.set_data([], [])
    time_text.set_text("")
    return line, time_text

def update(frame):
    phase = frame * 0.1
    y = np.sin(x - phase)
    line.set_data(x, y)
    time_text.set_text(f"phase = {phase:.2f} rad")
    return line, time_text

ani = FuncAnimation(
    fig,
    update,
    frames=100,
    init_func=init,
    interval=50,      # 每帧间隔 50ms
    blit=True         # 只重绘变化部分，提升性能
)

# 保存为 GIF（需要 pillow）
ani.save("sine_wave.gif", writer="pillow", fps=20, dpi=100)
plt.show()
```

### 16.4.2 梯度下降动画

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 定义损失函数（二维 Rosenbrock 变形）
def loss(w):
    return w[0]**2 + 10 * w[1]**2

def grad(w):
    return np.array([2 * w[0], 20 * w[1]])

# 梯度下降轨迹
lr = 0.08
w = np.array([2.5, 1.5])
trajectory = [w.copy()]

for _ in range(60):
    w = w - lr * grad(w)
    trajectory.append(w.copy())

trajectory = np.array(trajectory)

# 绘制损失曲面等高线
w1_range = np.linspace(-3, 3, 200)
w2_range = np.linspace(-2, 2, 200)
W1, W2 = np.meshgrid(w1_range, w2_range)
L = W1**2 + 10 * W2**2

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(W1, W2, L, levels=40, cmap="YlOrRd_r", alpha=0.7)
ax.contour(W1, W2, L, levels=20, colors="gray", linewidths=0.5, alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-2, 2)
ax.set_title("梯度下降收敛动画")
ax.set_xlabel("w₁"); ax.set_ylabel("w₂")

path_line, = ax.plot([], [], "b-o", markersize=4, lw=1.5, alpha=0.7)
point, = ax.plot([], [], "r*", markersize=15, zorder=5)
loss_text = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                    fontsize=11, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

def init():
    path_line.set_data([], [])
    point.set_data([], [])
    loss_text.set_text("")
    return path_line, point, loss_text

def update(frame):
    t = trajectory[:frame + 1]
    path_line.set_data(t[:, 0], t[:, 1])
    point.set_data([t[-1, 0]], [t[-1, 1]])
    current_loss = loss(t[-1])
    loss_text.set_text(f"Iter: {frame:3d}  Loss: {current_loss:.4f}")
    return path_line, point, loss_text

ani = FuncAnimation(fig, update, frames=len(trajectory),
                    init_func=init, interval=100, blit=True)
ani.save("gradient_descent.gif", writer="pillow", fps=10, dpi=100)
plt.show()
```

### 16.4.3 Plotly 交互式图表

```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# 带滑块的时间序列图
np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=200, freq="D")
price = 100 + np.cumsum(np.random.randn(200) * 0.5)
volume = np.abs(np.random.randn(200) * 1e6 + 5e6)

df = pd.DataFrame({"date": dates, "price": price, "volume": volume})

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["date"], y=df["price"],
    mode="lines", name="价格",
    line=dict(color="royalblue", width=2)
))

fig.update_layout(
    title="股价时间序列（可交互）",
    xaxis=dict(
        rangeselector=dict(
            buttons=[
                dict(count=7, label="1周", step="day", stepmode="backward"),
                dict(count=1, label="1月", step="month", stepmode="backward"),
                dict(count=3, label="3月", step="month", stepmode="backward"),
                dict(step="all", label="全部")
            ]
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    yaxis_title="价格",
    height=500
)
fig.write_html("interactive_timeseries.html")
fig.show()
```

### 16.4.4 Plotly Dash 简介

对于需要持续更新的交互仪表盘，可以使用 Plotly Dash：

```python
# 注意：运行此代码需要安装 dash：pip install dash
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("实时数据仪表盘示例"),
    dcc.Slider(id="noise-slider", min=0.1, max=2.0, step=0.1, value=0.5,
               marks={i/10: str(i/10) for i in range(1, 21, 3)}),
    html.Label("噪声强度"),
    dcc.Graph(id="scatter-chart"),
])

@app.callback(
    Output("scatter-chart", "figure"),
    Input("noise-slider", "value")
)
def update_chart(noise):
    np.random.seed(0)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.randn(100) * noise
    df = pd.DataFrame({"x": x, "y": y})
    fig = px.scatter(df, x="x", y="y", title=f"噪声={noise:.1f} 的正弦波")
    fig.add_scatter(x=x, y=np.sin(x), mode="lines", name="真实值",
                    line=dict(color="red"))
    return fig

if __name__ == "__main__":
    app.run(debug=True)
```

---

## 16.5 可视化最佳实践

### 16.5.1 选择合适的图表类型

| 场景 | 推荐图表 |
|------|---------|
| 比较类别大小 | 条形图、水平条形图 |
| 展示趋势变化 | 折线图、面积图 |
| 探索两变量关系 | 散点图、回归图 |
| 展示数据分布 | 直方图、箱线图、小提琴图 |
| 展示比例 | 饼图（类别 ≤ 5）、堆叠条形图 |
| 展示相关性矩阵 | 热力图 |
| 展示地理数据 | 地图（Folium、Plotly）|
| 三维空间数据 | 3D 散点图、曲面图 |

### 16.5.2 颜色与风格规范

```python
import matplotlib.pyplot as plt
import numpy as np

# 原则1：色盲友好调色板
# matplotlib 内置 tab10 与 colorblind 调色板都具有较好的可区分性
colors_cb = plt.cm.tab10.colors

# 原则2：避免彩虹色阶（rainbow），优先使用感知均匀的色阶
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

data = np.random.rand(8, 8)

# 不推荐
sns_ax1 = axes[0]
import seaborn as sns
sns.heatmap(data, cmap="rainbow", ax=sns_ax1, cbar=True)
sns_ax1.set_title("不推荐：彩虹色阶（感知不均匀）")

# 推荐
sns_ax2 = axes[1]
sns.heatmap(data, cmap="viridis", ax=sns_ax2, cbar=True)
sns_ax2.set_title("推荐：viridis（感知均匀）")

plt.tight_layout()
plt.savefig("colormap_advice.png", dpi=150)
plt.show()
```

### 16.5.3 图表可读性提升技巧

```python
import matplotlib.pyplot as plt
import numpy as np

# 示范：一张"糟糕"的图 vs 一张"良好"的图
np.random.seed(42)
categories = ["模型A", "模型B", "模型C", "模型D"]
accuracies = [0.82, 0.91, 0.87, 0.95]
errors = [0.03, 0.02, 0.025, 0.015]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ---- 糟糕的图 ----
axes[0].bar(range(4), accuracies, color="green")
axes[0].set_title("accuracy")
# 没有轴标签、没有误差线、y轴从0开始掩盖差异

# ---- 良好的图 ----
x = np.arange(len(categories))
bars = axes[1].bar(x, accuracies, yerr=errors, capsize=5,
                   color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
                   edgecolor="black", linewidth=0.8, alpha=0.85)

# 在每个柱子上方标注数值
for bar, acc in zip(bars, accuracies):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{acc:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

axes[1].set_xticks(x)
axes[1].set_xticklabels(categories, fontsize=12)
axes[1].set_ylim(0.75, 1.0)   # 缩小 y 轴范围，突出差异
axes[1].set_ylabel("准确率 (Accuracy)", fontsize=12)
axes[1].set_title("各模型准确率对比（含标准误差）", fontsize=13)
axes[1].yaxis.grid(True, linestyle="--", alpha=0.5)
axes[1].set_axisbelow(True)   # 网格线置于柱子下方
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("good_vs_bad_chart.png", dpi=150)
plt.show()
```

### 16.5.4 保存图表的规范

```python
# 论文/出版物 → 高分辨率矢量格式
plt.savefig("figure.pdf", bbox_inches="tight")        # 矢量，适合LaTeX
plt.savefig("figure.svg", bbox_inches="tight")        # 矢量，适合网页
plt.savefig("figure.png", dpi=300, bbox_inches="tight")  # 光栅，300 dpi

# 演示/网络分享 → 适中分辨率
plt.savefig("figure_web.png", dpi=150, bbox_inches="tight")

# 设置全局字体大小（便于统一调整）
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
})
```

### 16.5.5 可视化调试清单

在发布图表之前，应检查以下事项：

- **标题**：图表是否有清晰的标题？
- **轴标签**：两轴是否都有标签和单位？
- **图例**：多条曲线是否有图例？图例位置是否遮挡数据？
- **颜色**：颜色是否对色盲友好？是否使用了感知均匀的色阶？
- **数据范围**：y 轴是否从适当的值开始，不误导读者？
- **字体大小**：文字是否足够大，缩小后仍可读？
- **分辨率**：保存的图片分辨率是否满足使用场景？

---

## 本章小结

| 主题 | 核心 API / 工具 | 典型应用场景 |
|------|----------------|-------------|
| Seaborn 统计可视化 | `histplot`, `boxplot`, `violinplot`, `regplot`, `pairplot` | EDA、分布分析、回归分析 |
| 热力图与关联图 | `sns.heatmap`, `sns.clustermap` | 相关矩阵、混淆矩阵、基因表达 |
| 3D 可视化 | `mpl_toolkits.mplot3d`, `plotly.graph_objects` | 损失曲面、三维特征空间 |
| 动画与交互 | `FuncAnimation`, `plotly`, `dash` | 优化过程、时间序列、仪表盘 |
| 可视化最佳实践 | `rcParams`, `bbox_inches`, `viridis` | 论文出图、技术报告 |

---

## 深度学习应用：特征可视化

深度学习模型通常被视为"黑盒"，但通过可视化技术可以揭示模型内部的工作机制。

### 16.A.1 CNN 卷积核与特征图可视化

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

# 加载预训练 ResNet18（仅用于演示，不需要真实图片）
model = models.resnet18(pretrained=False)
model.eval()

# ---------- 可视化第一层卷积核 ----------
first_conv_weights = model.conv1.weight.data  # shape: (64, 3, 7, 7)

fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    if i < first_conv_weights.shape[0]:
        # 取前3通道作为 RGB，归一化到 [0,1]
        kernel = first_conv_weights[i].permute(1, 2, 0).numpy()
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
        ax.imshow(kernel)
    ax.axis("off")

plt.suptitle("ResNet18 第一层卷积核（64个，7×7）", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("conv_kernels.png", dpi=150)
plt.show()

# ---------- 钩子提取特征图 ----------
def get_feature_maps(model, layer_name):
    """注册前向传播钩子，提取指定层的特征图"""
    features = {}

    def hook(module, input, output):
        features[layer_name] = output.detach()

    # 获取对应子模块并注册钩子
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook)
            break
    return features

# 创建随机输入（模拟一张 224×224 图片）
dummy_input = torch.randn(1, 3, 224, 224)
features = get_feature_maps(model, "layer1.0.conv1")

with torch.no_grad():
    _ = model(dummy_input)

# 可视化 layer1.0.conv1 的特征图
if "layer1.0.conv1" in features:
    fmaps = features["layer1.0.conv1"][0]  # shape: (C, H, W)
    n_show = min(16, fmaps.shape[0])

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < n_show:
            fmap = fmaps[i].numpy()
            ax.imshow(fmap, cmap="viridis")
            ax.set_title(f"Channel {i}", fontsize=9)
        ax.axis("off")
    plt.suptitle("layer1.0.conv1 特征图（随机输入）", fontsize=13)
    plt.tight_layout()
    plt.savefig("feature_maps.png", dpi=150)
    plt.show()
```

### 16.A.2 注意力权重可视化

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 模拟 Transformer 自注意力权重
class SimpleAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.d_head ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        return attn  # shape: (B, n_heads, T, T)

# 生成随机输入（batch=1, seq_len=10, d_model=64）
torch.manual_seed(42)
seq_len = 10
d_model = 64
n_heads = 4
tokens = ["[CLS]", "今天", "天气", "真的", "非常", "好", "，", "心情", "愉快", "[SEP]"]

attn_module = SimpleAttention(d_model, n_heads)
x = torch.randn(1, seq_len, d_model)

with torch.no_grad():
    attn_weights = attn_module(x)  # (1, 4, 10, 10)

attn_weights = attn_weights[0].numpy()  # (4, 10, 10)

# 可视化4个注意力头
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for head_idx, ax in enumerate(axes.flat):
    sns.heatmap(
        attn_weights[head_idx],
        ax=ax,
        cmap="Blues",
        vmin=0, vmax=attn_weights[head_idx].max(),
        xticklabels=tokens,
        yticklabels=tokens,
        annot=True, fmt=".2f",
        cbar_kws={"shrink": 0.8},
        square=True,
        linewidths=0.5
    )
    ax.set_title(f"注意力头 {head_idx + 1}", fontsize=12)
    ax.set_xlabel("Key 位置", fontsize=10)
    ax.set_ylabel("Query 位置", fontsize=10)
    ax.tick_params(axis="x", rotation=45)

plt.suptitle("Transformer 自注意力权重可视化（4个头）", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("attention_weights.png", dpi=150, bbox_inches="tight")
plt.show()
```

### 16.A.3 t-SNE 高维特征降维可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 使用手写数字数据集（64维特征）
digits = load_digits()
X = digits.data       # (1797, 64)
y = digits.target     # (1797,)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# t-SNE 降维到 2D
print("正在运行 t-SNE...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    random_state=42,
    learning_rate="auto",
    init="pca"
)
X_2d = tsne.fit_transform(X_scaled)
print(f"t-SNE 完成，输出形状: {X_2d.shape}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 散点图（按数字类别着色）
palette = sns.color_palette("tab10", n_colors=10)
for digit in range(10):
    mask = y == digit
    axes[0].scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        c=[palette[digit]], label=str(digit),
        alpha=0.7, s=15, edgecolors="none"
    )

axes[0].set_title("t-SNE 降维：手写数字特征分布", fontsize=13)
axes[0].set_xlabel("t-SNE 维度 1")
axes[0].set_ylabel("t-SNE 维度 2")
axes[0].legend(title="数字", bbox_to_anchor=(1.02, 1), loc="upper left",
               markerscale=2)

# KDE 密度图（仅展示几个类别）
from scipy.stats import gaussian_kde
for digit in [0, 1, 7, 9]:
    mask = y == digit
    x_pts, y_pts = X_2d[mask, 0], X_2d[mask, 1]
    # 简单用椭圆表示聚类中心
    cx, cy = x_pts.mean(), y_pts.mean()
    axes[1].scatter(x_pts, y_pts, c=[palette[digit]], alpha=0.3, s=10)
    axes[1].annotate(
        f" {digit}",
        xy=(cx, cy),
        fontsize=16,
        fontweight="bold",
        color=palette[digit]
    )

axes[1].set_title("t-SNE 聚类中心标注（部分类别）", fontsize=13)
axes[1].set_xlabel("t-SNE 维度 1")
axes[1].set_ylabel("t-SNE 维度 2")

plt.tight_layout()
plt.savefig("tsne_visualization.png", dpi=150, bbox_inches="tight")
plt.show()
```

### 16.A.4 梯度加权类激活映射（Grad-CAM 简化版）

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    """用于演示 Grad-CAM 的简单 CNN"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# 注册梯度钩子
gradients = {}
activations = {}

def save_gradient(name):
    def hook(module, grad_input, grad_output):
        gradients[name] = grad_output[0]
    return hook

def save_activation(name):
    def hook(module, input, output):
        activations[name] = output
    return hook

model = SimpleCNN()
# 注册最后一个卷积层的钩子
last_conv = model.features[3]  # 第2个 Conv2d
last_conv.register_forward_hook(save_activation("last_conv"))
last_conv.register_backward_hook(save_gradient("last_conv"))

# 前向传播
torch.manual_seed(0)
x = torch.randn(1, 1, 28, 28, requires_grad=True)
output = model(x)
target_class = output.argmax().item()

# 反向传播（对目标类别的分数求梯度）
model.zero_grad()
output[0, target_class].backward()

# 计算 Grad-CAM 热力图
grads = gradients["last_conv"][0]       # (C, H, W)
acts = activations["last_conv"][0]      # (C, H, W)
weights = grads.mean(dim=(1, 2))        # 对空间维度求均值

cam = torch.zeros(acts.shape[1:])
for i, w in enumerate(weights):
    cam += w * acts[i]

cam = torch.relu(cam)                   # 只保留正激活
cam = cam.detach().numpy()
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# 上采样到输入尺寸
from scipy.ndimage import zoom
zoom_factor = (28 / cam.shape[0], 28 / cam.shape[1])
cam_resized = zoom(cam, zoom_factor)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

input_img = x[0, 0].detach().numpy()
axes[0].imshow(input_img, cmap="gray")
axes[0].set_title("输入图像（随机）")
axes[0].axis("off")

axes[1].imshow(cam_resized, cmap="jet")
axes[1].set_title(f"Grad-CAM 热力图\n(预测类别: {target_class})")
axes[1].axis("off")

axes[2].imshow(input_img, cmap="gray")
axes[2].imshow(cam_resized, cmap="jet", alpha=0.5)
axes[2].set_title("叠加效果")
axes[2].axis("off")

plt.suptitle("Grad-CAM：CNN 决策区域可视化", fontsize=13)
plt.tight_layout()
plt.savefig("gradcam.png", dpi=150)
plt.show()
```

---

## 练习题

### 基础题

**1. Seaborn 分布分析**

使用 Seaborn 内置的 `penguins` 数据集，完成以下任务：
- 绘制企鹅嘴峰长度（`bill_length_mm`）的分布直方图（含 KDE 曲线）
- 按企鹅种类（`species`）绘制嘴峰长度的小提琴图
- 绘制嘴峰长度与嘴峰深度（`bill_depth_mm`）的回归散点图（按种类着色）

**2. 热力图绘制**

加载 `sklearn` 的 `wine` 数据集，计算所有特征的相关系数矩阵，并绘制：
- 完整的相关系数热力图（注明数值，使用 `RdBu_r` 色阶）
- 只展示下三角部分的热力图（通过 `mask` 参数）

### 进阶题

**3. 梯度下降 3D 可视化**

定义如下损失函数：
$$L(w_1, w_2) = (w_1 - 1)^2 + 5(w_2 - 2)^2$$
- 使用 Matplotlib 绘制该函数的 3D 曲面图
- 在曲面上用红色标记全局最小值点 $(1, 2, 0)$
- 在同一图上叠加从初始点 $(3.5, 0.5)$ 出发的梯度下降轨迹（使用学习率 $\alpha = 0.1$，迭代 30 步）

**4. FuncAnimation 动画**

制作一个展示"泰勒展开逼近 $\cos(x)$"的动画：
- 每一帧增加一个高阶项（从 0 阶到 8 阶）
- 背景绘制真实的 $\cos(x)$ 曲线（红色虚线）
- 当前阶次的泰勒近似用蓝色实线绘制
- 在图的标题或文本框中显示当前阶次
- 保存为 GIF 文件

### 挑战题

**5. 综合：训练过程可视化仪表盘**

使用 PyTorch 在 MNIST 数据集上训练一个简单 MLP，训练过程中记录每个 epoch 的训练损失、验证损失和验证准确率。训练完成后，创建一个包含以下内容的综合可视化图：
- 子图1：训练损失 vs 验证损失曲线（折线图，含图例）
- 子图2：验证准确率随 epoch 的变化曲线
- 子图3：对测试集中随机选取的 16 张图片进行预测，绘制带标签的图像网格（正确预测用绿色标题，错误预测用红色标题）
- 子图4：最终模型在测试集上的混淆矩阵热力图

---

## 练习答案

### 答案 1：Seaborn 分布分析

```python
import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset("penguins").dropna()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. 分布直方图 + KDE
sns.histplot(penguins["bill_length_mm"], kde=True, ax=axes[0],
             color="steelblue", bins=25)
axes[0].set_title("嘴峰长度分布直方图", fontsize=12)
axes[0].set_xlabel("嘴峰长度 (mm)")

# 2. 按种类的小提琴图
sns.violinplot(x="species", y="bill_length_mm", data=penguins,
               ax=axes[1], palette="Set2", inner="box")
axes[1].set_title("各种类嘴峰长度小提琴图", fontsize=12)
axes[1].set_xlabel("种类"); axes[1].set_ylabel("嘴峰长度 (mm)")

# 3. 回归散点图（按种类着色）
for species, color in zip(penguins["species"].unique(), ["#4C72B0", "#DD8452", "#55A868"]):
    subset = penguins[penguins["species"] == species]
    axes[2].scatter(subset["bill_length_mm"], subset["bill_depth_mm"],
                    label=species, alpha=0.6, s=40, color=color)
    # 添加线性回归线
    z = np.polyfit(subset["bill_length_mm"], subset["bill_depth_mm"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(subset["bill_length_mm"].min(),
                           subset["bill_length_mm"].max(), 100)
    axes[2].plot(x_range, p(x_range), color=color, lw=2)

axes[2].set_title("嘴峰长度 vs 嘴峰深度（按种类）", fontsize=12)
axes[2].set_xlabel("嘴峰长度 (mm)"); axes[2].set_ylabel("嘴峰深度 (mm)")
axes[2].legend(title="种类")

plt.tight_layout()
plt.savefig("penguins_analysis.png", dpi=150)
plt.show()
```

### 答案 2：热力图绘制

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import pandas as pd

wine_data = load_wine()
df_wine = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
corr = df_wine.corr()

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 完整热力图
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.3,
            ax=axes[0], cbar_kws={"shrink": 0.8})
axes[0].set_title("Wine 数据集完整相关矩阵", fontsize=12)
axes[0].tick_params(axis="x", rotation=45)

# 下三角热力图
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.3,
            ax=axes[1], cbar_kws={"shrink": 0.8})
axes[1].set_title("Wine 数据集相关矩阵（下三角）", fontsize=12)
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("wine_correlation.png", dpi=150)
plt.show()
```

### 答案 3：梯度下降 3D 可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def L(w1, w2):
    return (w1 - 1)**2 + 5 * (w2 - 2)**2

def grad_L(w):
    return np.array([2 * (w[0] - 1), 10 * (w[1] - 2)])

# 梯度下降轨迹
lr = 0.1
w = np.array([3.5, 0.5])
traj = [w.copy()]
for _ in range(30):
    w = w - lr * grad_L(w)
    traj.append(w.copy())
traj = np.array(traj)

# 绘制曲面
w1_lin = np.linspace(-1, 5, 100)
w2_lin = np.linspace(-1, 5, 100)
W1, W2 = np.meshgrid(w1_lin, w2_lin)
Z = L(W1, W2)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(W1, W2, Z, cmap="YlOrRd_r", alpha=0.7, rstride=2, cstride=2)
ax1.plot(traj[:, 0], traj[:, 1], L(traj[:, 0], traj[:, 1]),
         "b-o", markersize=3, lw=2, label="梯度下降路径")
ax1.scatter([1], [2], [0], color="red", s=100, zorder=10, label="全局最小值")
ax1.set_title("梯度下降 3D 轨迹")
ax1.set_xlabel("w₁"); ax1.set_ylabel("w₂"); ax1.set_zlabel("L")
ax1.legend()

ax2 = fig.add_subplot(122)
cf = ax2.contourf(W1, W2, Z, levels=40, cmap="YlOrRd_r", alpha=0.8)
plt.colorbar(cf, ax=ax2)
ax2.contour(W1, W2, Z, levels=20, colors="gray", linewidths=0.5, alpha=0.4)
ax2.plot(traj[:, 0], traj[:, 1], "b-o", markersize=4, lw=1.5, label="梯度下降路径")
ax2.scatter([1], [2], color="red", s=150, zorder=10, marker="*", label="全局最小值 (1,2)")
ax2.set_title("梯度下降俯视轨迹")
ax2.set_xlabel("w₁"); ax2.set_ylabel("w₂")
ax2.legend()

plt.tight_layout()
plt.savefig("gd_3d_trajectory.png", dpi=150)
plt.show()
```

### 答案 4：泰勒展开动画

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import factorial

def taylor_cos(x, n_terms):
    """计算 cos(x) 的泰勒展开，展开到第 n_terms 项（偶数阶）"""
    result = np.zeros_like(x, dtype=float)
    for k in range(n_terms + 1):
        result += ((-1)**k * x**(2*k)) / factorial(2*k)
    return result

x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
y_true = np.cos(x)
max_order = 8  # 展开到第 8 阶

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, y_true, "r--", lw=2, label="cos(x) 精确值")
ax.set_xlim(-2 * np.pi, 2 * np.pi)
ax.set_ylim(-3, 3)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.axhline(0, color="black", lw=0.5)

approx_line, = ax.plot([], [], "b-", lw=2.5, label="泰勒近似")
order_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                     fontsize=13, va="top",
                     bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

def init():
    approx_line.set_data([], [])
    order_text.set_text("")
    return approx_line, order_text

def update(frame):
    n = frame  # 当前阶次（0, 2, 4, 6, 8）
    y_approx = taylor_cos(x, n)
    y_approx = np.clip(y_approx, -3, 3)  # 防止高阶截断爆炸
    approx_line.set_data(x, y_approx)
    approx_line.set_label(f"泰勒近似（{2*n} 阶）")
    order_text.set_text(f"阶次: {2*n}  项数: {n+1}")
    ax.set_title(f"cos(x) 泰勒展开逼近（0 → {2*max_order} 阶）")
    return approx_line, order_text

ani = FuncAnimation(
    fig, update, frames=range(max_order + 1),
    init_func=init, interval=800, blit=True, repeat=True
)

ani.save("taylor_cosine.gif", writer="pillow", fps=1.5, dpi=100)
plt.show()
```

### 答案 5：训练过程可视化仪表盘

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---- 数据加载 ----
transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])

train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

# 为加快演示速度，仅使用子集
train_sub = Subset(train_set, range(5000))
val_sub = Subset(test_set, range(1000))

train_loader = DataLoader(train_sub, batch_size=64, shuffle=True)
val_loader = DataLoader(val_sub, batch_size=64, shuffle=False)

# ---- 模型定义 ----
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---- 训练循环 ----
n_epochs = 10
train_losses, val_losses, val_accs = [], [], []

for epoch in range(n_epochs):
    # 训练
    model.train()
    running_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(X)
    train_losses.append(running_loss / len(train_sub))

    # 验证
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            val_loss += criterion(out, y).item() * len(X)
            correct += (out.argmax(1) == y).sum().item()
    val_losses.append(val_loss / len(val_sub))
    val_accs.append(correct / len(val_sub))
    print(f"Epoch {epoch+1:2d} | Train Loss: {train_losses[-1]:.4f} "
          f"| Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accs[-1]:.4f}")

# ---- 测试集预测 ----
test_loader_full = DataLoader(test_set, batch_size=1000, shuffle=False)
all_preds, all_labels, sample_imgs = [], [], []

model.eval()
with torch.no_grad():
    for i, (X, y) in enumerate(test_loader_full):
        X, y = X.to(device), y.to(device)
        out = model(X)
        preds = out.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        if i == 0:
            sample_imgs = X[:16].cpu()
            sample_preds = preds[:16].cpu().numpy()
            sample_labels = y[:16].cpu().numpy()

# ---- 综合可视化 ----
fig = plt.figure(figsize=(18, 16))
fig.suptitle("MNIST MLP 训练过程综合可视化", fontsize=16, y=0.98)

epochs_range = range(1, n_epochs + 1)

# 子图1：损失曲线
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(epochs_range, train_losses, "b-o", label="训练损失", markersize=5)
ax1.plot(epochs_range, val_losses, "r-s", label="验证损失", markersize=5)
ax1.set_title("训练 vs 验证损失", fontsize=13)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(True, alpha=0.3)

# 子图2：准确率曲线
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(epochs_range, [a * 100 for a in val_accs], "g-^",
         label="验证准确率", markersize=5)
ax2.set_title("验证准确率", fontsize=13)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("准确率 (%)")
ax2.set_ylim(80, 100)
ax2.legend(); ax2.grid(True, alpha=0.3)

# 子图3：预测图像网格
ax3 = fig.add_subplot(2, 2, 3)
ax3.axis("off")
ax3.set_title("测试集预测结果（前16张）", fontsize=13)

inner_grid = fig.add_gridspec(2, 2, left=0.05, right=0.48,
                               bottom=0.05, top=0.45,
                               hspace=0.1, wspace=0.1)
# 重新布局为4x4子图
grid_ax = [fig.add_subplot(4, 8, i + 25) for i in range(16)]
for i, gax in enumerate(grid_ax):
    img = sample_imgs[i, 0].numpy()
    gax.imshow(img, cmap="gray")
    gax.axis("off")
    color = "green" if sample_preds[i] == sample_labels[i] else "red"
    gax.set_title(f"{sample_preds[i]}", fontsize=9, color=color, pad=1)

# 子图4：混淆矩阵
ax4 = fig.add_subplot(2, 2, 4)
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10),
            ax=ax4, cbar=False)
ax4.set_title("测试集混淆矩阵", fontsize=13)
ax4.set_xlabel("预测标签"); ax4.set_ylabel("真实标签")

plt.tight_layout()
plt.savefig("mnist_training_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()

final_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"\n最终测试准确率: {final_acc:.4f} ({final_acc*100:.2f}%)")
```

---

> **提示**：本章代码依赖 `matplotlib`、`seaborn`、`plotly`、`torch`、`torchvision`、`sklearn`、`scipy`、`pillow`。可通过以下命令一次性安装：
> ```bash
> pip install matplotlib seaborn plotly torch torchvision scikit-learn scipy pillow dash
> ```
