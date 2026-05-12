# 中考代数教程 图形风格规范

本规范为 `figures/src/{tikz,asy}/` 下所有源文件统一约定。新增图请遵守，已有图重渲时按规范微调。基础约定与 `chuzhong_geometry/figures/STYLE.md` 保持一致；本文档在此基础上增补代数特有约定（§5 起）。

---

## 一、命名规则

`<scope>-<topic>[-variant].{tex,asy}`

- `thm-<topic>.{tex,asy}` ——定理/性质图，如 `thm-quadratic-vertex.tex`、`thm-linear-fn-slope.tex`
- `ex-<id>-<n>.{tex,asy}` ——例题图，`<id>` 使用 `p1` 至 `p14` 对应 14 个 part，`<n>` 为同一节内的序号
  - 示例：`ex-p6-03-1.tex`（part6 第三章例题 1）、`ex-p10-07-2.asy`（part10 第七章例题 2）
- `q-<level>-NN.{tex,asy}` ——附录题库图，`<level>` 为 `c`（基础）/`d`（中档）/`e`（压轴），`NN` 为题号
  - 示例：`q-c-12.tex`、`q-d-07.asy`、`q-e-03.tex`
- `thm-`, `def-`, `model-` 前缀分别用于定理图、概念图、方法骨架图（与几何教程一致）
- 文件名全小写，用连字符分隔单词；不含空格、不含中文

---

## 二、TikZ 通用模板

```latex
\documentclass[tikz,border=4pt]{standalone}
\usepackage{ctex}                  % 中文支持（xelatex）
\usepackage{amsmath}
\usepackage{pgfplots}              % 函数图象（代数教程新增）
\pgfplotsset{compat=1.18}
\usetikzlibrary{calc, angles, quotes, decorations.markings, arrows.meta}

\begin{document}
\begin{tikzpicture}[
  scale=1.0, thick,
  % 等长边标记（与几何教程一致）
  tickone/.style={postaction={decorate, decoration={markings,
    mark=at position 0.5 with {\draw (-1.5pt,-3pt) -- (1.5pt,3pt);}}}},
  ticktwo/.style={postaction={decorate, decoration={markings,
    mark=at position 0.5 with {\draw (-2.5pt,-3pt) -- (-0.5pt,3pt);
                               \draw (0.5pt,-3pt)  -- (2.5pt,3pt);}}}},
]
  % ... 你的图 ...
\end{tikzpicture}
\end{document}
```

**用 xelatex 编译**（render.sh 自动选择）。

`\usepackage{pgfplots}` 仅在含函数图象的文件中引入；纯几何辅助线图可省略以加快编译。

---

## 三、Asymptote 通用模板

```asymptote
size(8cm);
import geometry;
// 函数图象用 graph 模块
// import graph;
// 中文文字用 label("文字", pos, dir, p=fontsize(10))
// xelatex 引擎下需在 settings 启用 xelatex（render.sh 全局注入）

// ... 你的图 ...
```

函数图象优先推荐 TikZ + pgfplots（与几何图统一风格，利于维护）；仅曲线形状复杂（如三次及以上函数）时考虑 Asymptote graph 模块。

---

## 四、颜色与线型

| 用途 | 颜色 | 线型 |
|---|---|---|
| 主图形（主要线段、坐标轴） | `black` | 实线 `thick` |
| 重点/待证线段或表达式 | `red` | 实线 `thick` |
| 函数曲线主色（一次/二次/反比例） | `blue` | 实线 `thick` |
| 变换后的副本（平移/反射/伸缩后的图象） | `blue` | 虚线 `dashed, thick` |
| 辅助线、网格线、延长线 | `gray` | 虚线 `dashed` |
| 强调区域（不等式解集、定义域着色） | `blue!15` | 填充 |
| 坐标轴箭头 | `black` | `[->]` `thick` |

> 说明：代数教程中几何图的颜色约定与几何教程相同（`black/red/blue/gray`）。代数新增"函数曲线主色 = 蓝色"，与几何"旋转副本 = 蓝色虚线"在语义上有所不同，不冲突——根据图的性质选择实线或虚线即可区分。

---

## 五、代数特有约定

### 5.1 函数图象

坐标轴绘制规范：

```latex
% 坐标轴（标准写法）
\draw[->] (-0.5,0) -- (4.5,0) node[right] {$x$};   % x 轴，箭头在右端
\draw[->] (0,-0.5) -- (0,4.5) node[above] {$y$};   % y 轴，箭头在上端
\node[below left] at (0,0) {$O$};                   % 原点标 $O$，放在左下角
```

- x 轴标签 `$x$` 置于右端箭头后（`node[right]`）
- y 轴标签 `$y$` 置于上端箭头后（`node[above]`）
- 原点标 `$O$`，位置 `node[below left]` at (0,0)
- 坐标轴刻度线（tick）用 `\draw (k, 2pt) -- (k, -2pt)` 标记整数刻度
- 图形默认不画网格；仅习题讲解类图或统计图使用网格

**关键点标记规范：**

```latex
% 函数图象上的关键点（顶点、零点、截距等）
\fill (2,3) circle (2.5pt);                             % 实心圆点
\node[above right] at (2,3) {$A(2,\,3)$};              % 坐标标签放在右上方（或不碍图的方向）

% 空心圆（开区间端点）
\draw (1,0) circle (2.5pt);
```

- **实心圆** `\fill` + 2.5pt 半径 = 闭端点 / 函数图象上的具体点
- **空心圆** `\draw circle` = 开端点（数轴、区间图中）
- 坐标标签格式 `$A(x_0,\,y_0)$`，括号内用 `\,` 加细间距

**函数曲线绘制（pgfplots 方式）：**

```latex
\begin{axis}[
  axis lines=center,         % 坐标轴过原点
  xlabel={$x$}, ylabel={$y$},
  every axis x label/.style={at={(axis cs:5,0)}, anchor=west},
  every axis y label/.style={at={(axis cs:0,5)}, anchor=south},
  xmin=-3, xmax=3, ymin=-1, ymax=5,
  thick,
]
  \addplot[blue, thick, domain=-2:2, samples=100] {x^2 + 2*x + 1};
\end{axis}
```

pgfplots 中 `axis lines=center` 使坐标轴自动过原点并带箭头，与手绘箭头风格一致。

---

### 5.2 数轴

数轴是不等式解集、实数比较等的标准图形工具：

```latex
% 数轴骨架
\draw[->] (-3.5,0) -- (3.5,0);              % 水平线 + 向右箭头
% 整数刻度
\foreach \x in {-3,-2,...,3} {
  \draw (\x, 2pt) -- (\x, -2pt) node[below] {\small$\x$};
}

% 闭端点（∈ 解集，实心）
\fill (1,0) circle (2.5pt);

% 开端点（不含端点，空心）
\draw (2,0) circle (2.5pt);
\draw[very thick] (2,0) -- (3.5,0);        % 解集区间（加粗标出范围）
```

- 用 `\bullet`（`\fill`）标记**闭端点**，用 `\circ`（`\draw circle`）标记**开端点**
- 解集区间在数轴上方用加粗线段或 `\draw[very thick]` 显著标出
- 若解集为无穷延伸（$x > a$ 或 $x < a$），线段延伸到坐标轴末端箭头处

---

### 5.3 统计图表

本教程统计图（条形图、折线图、扇形图、直方图）统一使用 TikZ 手工绘制或 pgfplots。

**颜色约定（浅色填充，至多 4 种）：**

| 序号 | TikZ 颜色写法 | 语义 |
|---|---|---|
| 第 1 类 | `cyan!30` | 第一组/A 类 |
| 第 2 类 | `orange!30` | 第二组/B 类 |
| 第 3 类 | `green!30` | 第三组/C 类 |
| 第 4 类 | `pink!30` | 第四组/D 类 |

- 避免使用深色填充（影响文字可读性）
- 超过 4 类时，优先对数据分组而非增加颜色种数
- 条形图柱体描边用 `draw=black`，配合 `fill=cyan!30` 等

**直方图示例（pgfplots）：**

```latex
\begin{axis}[
  ybar, bar width=1.5cm,
  axis lines=left,
  xlabel={分数段}, ylabel={频数},
  symbolic x coords={60--70,70--80,80--90,90--100},
  xtick=data,
]
  \addplot[fill=cyan!30, draw=black] coordinates {
    (60--70, 5) (70--80, 12) (80--90, 18) (90--100, 9)
  };
\end{axis}
```

---

### 5.4 概率树状图

概率树状图（列举法、古典概率）采用**上下分层**布局：

```latex
% 第一层：根节点
\node (root) at (0,0) {$\cdot$};
% 第二层：分支
\node (A) at (-2,-1.5) {$A$};
\node (B) at (2,-1.5)  {$B$};
% 分支箭头
\draw[->] (root) -- (A) node[midway, above left]  {$\frac{1}{2}$};
\draw[->] (root) -- (B) node[midway, above right] {$\frac{1}{2}$};
% 第三层：子分支
\node (AA) at (-3,-3) {$AA$};
\node (AB) at (-1,-3) {$AB$};
\draw[->] (A) -- (AA) node[midway, left]  {$\frac{1}{3}$};
\draw[->] (A) -- (AB) node[midway, right] {$\frac{2}{3}$};
```

- 树从上到下展开，根节点在顶部
- 分支用带箭头的线段（`->`）连接父子节点
- **概率值**标在分支边的中间位置（`node[midway]`），靠近对应分支一侧
- 叶子节点（末端事件）标出事件名称；必要时在右侧额外列出概率值 $P = \frac{m}{n}$
- 节点直径小图用 `circle (1.5pt)`，大图用文字框

---

## 六、标签习惯（与几何教程一致）

- **顶点 / 关键点**：用大写字母 $A, B, C, \dots$ 或坐标形式 $A(x_0, y_0)$，`\node[方向]` 放在点旁边
- **函数表达式**：直接标在曲线旁，如 `\node[right, blue] at (2, 3.5) {$y = x^2$};`
- **角**：用 `pic{angle=...}` 或 `\draw arc`，标 $\alpha, \beta, \theta$ 等
- **等长边**：用 `tickone` / `ticktwo` 短斜线（几何图中，代数图一般不用）

---

## 七、字号与缩放

- 整体 `scale=1.0`（常规图）或 `1.2`（小细节图）
- 函数标签用默认字号（`\normalsize`）
- 数轴刻度、辅助值用 `\small` 或 `\footnotesize`
- 概率分支上的数值用 `\footnotesize`，避免与分支线重叠

---

## 八、坐标精度（与几何教程一致）

- 关键点坐标须**数学上精确**，不能"看上去差不多"
- 函数图象：顶点、零点、截距等关键点必须符合解析式（如 $y = (x-2)^2 + 1$ 顶点严格在 $(2, 1)$）
- 不等式解集：端点位置须精确对应数值（如 $x > -3$ 的端点严格在 $-3$ 处）
- 统计图：柱高 / 线段高度须与数据完全对应，不能用目测近似

---

## 九、图风格参照（代数版起点）

代数教程的图风格以几何教程的已有 SVG 为起点，下列文件展示了本教程遵循的基础约定，可作为新图的对照参考：

- **坐标系基础风格**：`chuzhong_geometry/figures/svg/thm-slope.svg`（坐标轴+直线图，箭头、标签位置的范例）
- **关键点标记**：`chuzhong_geometry/figures/svg/thm-distance-derivation.svg`（含实心点 + 坐标标签的范例）
- **综合几何代数图**：`chuzhong_geometry/figures/svg/ex-p10-02-1.svg`、`ex-p10-02-2.svg`（坐标系中动点图，代数教程 part14 综合图参考）
- **简单直线图**：`chuzhong_geometry/figures/svg/thm-parallel-projection.svg`
- **样例整体风格**：`chuzhong_geometry/figures/svg/sample-handshake.svg`、`sample-triangle.svg`（黑色主线、红色标注、灰色辅助线的三色约定）

代数教程新增的函数图象、数轴、统计图、概率树状图无几何教程对应范例，以本文档 §5 约定为准。

---

## 十、模型图与例题图的关系（与几何教程一致）

- **方法骨架图**（`model-` 前缀，如 `model-quadratic-vertex.tex`）：不含具体数值，仅用字母和变量——突出"式子结构"或"图象形状"
- **例题图**（`ex-` 前缀）：保留题目给定的具体数值（如 $a = 2, b = -4, c = 3$），让读者可直接对照题目
- 例题图与骨架图颜色风格一致（同颜色/字体/标签习惯），便于读者从例题反认方法
