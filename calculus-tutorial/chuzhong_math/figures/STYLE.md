# 中考几何教程 图形风格规范

本规范为 `figures/src/{tikz,asy}/` 下所有源文件统一约定。新增图请遵守，已有图重渲时按规范微调。

---

## 命名

`<scope>-<topic>[-variant].{tex,asy}`

- `scope`：`model-` 模型主图、`thm-` 定理图、`ex-<partN>-<chapter>-<n>` 例题图（如 `ex-p3cong03-1`）、`def-` 概念图
- `topic`：模型名 / 主题（短横线分隔，全小写）
- 示例：
  - `model-handshake-basic.tex`
  - `ex-p3cong03-1.tex`（手拉手章节例题 1）
  - `thm-pythagoras.tex`
  - `def-vertical-angle.tex`

---

## TikZ 通用模板

```latex
\documentclass[tikz,border=4pt]{standalone}
\usepackage{ctex}                  % 中文支持（xelatex）
\usepackage{amsmath}
\usetikzlibrary{calc, angles, quotes, decorations.markings, arrows.meta}

\begin{document}
\begin{tikzpicture}[
  scale=1.0, thick,
  % 等长边标记
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

---

## Asymptote 通用模板

```asymptote
size(8cm);
import geometry;
// 中文文字用 label("文字", pos, dir, p=fontsize(10))，xelatex 引擎下需在 settings 启用 xelatex
// 推荐写法：tex(\"\\usepackage{ctex}\");  // 已在 render.sh 全局注入

// ... 你的图 ...
```

---

## 颜色与线型

| 用途 | 颜色 | 线型 |
|---|---|---|
| 主图形（原始线段） | `black` | 实线 `thick` |
| 辅助线 / 待证线段 | `red` | 实线 `thick` |
| 旋转/翻折/平移后的副本 | `blue` | 虚线 `dashed, thick` |
| 已知半径 / 隐藏的几何关系 | `gray` | 虚线 `dashed` |
| 待求的关键线段 | `red, thick` | 实线 |

---

## 标签习惯

- **顶点**：用大写字母 $A, B, C, \dots$，`\node[方向]` 放在点旁边
- **角**：用 `\angle ABC` 或 $\alpha, \beta, \theta$，配 `\draw arc` 或 `pic{angle=...}`
- **等长边**：用 `tickone` / `ticktwo` 短斜线标记
- **直角**：用 `\draw` 画小方块（边长约 0.15-0.2）：
  ```latex
  \draw (B) -- ($(B)!0.18!(A)$) -- ($(B)!0.18!(A) + 0.18*(0,1)$) -- ($(B)!0.18!(C)$);
  ```

---

## 字号 / 缩放

- 整体 `scale=1.0` 或 `1.2`（小图）
- 顶点字母用默认字号；角度用 `\footnotesize` 或 `\small`
- 长度标注（如 "2", "3"）用 `\footnotesize`

---

## 坐标精度

- 关键点的坐标要**几何上准确**，不能"看上去差不多"
- 例如手拉手：两个等腰三角形的顶角必须相等；母子相似的 D 点必须满足 $AD/AB = (\text{cathetus}/\text{hypotenuse})^2$
- 不确定时先在草稿验证（如用三边/勾股反推坐标）

---

## 引入图与例题图的关系

- **模型主图**（"一图速记"）：去掉所有数值/具体长度，仅用字母与符号——突出"图形结构"
- **例题图**：保留题目给定的具体数值（如 $AB=8, BC=6$），让读者可直接对照题目
- 例题图与主图风格保持一致（同颜色/字体/标签习惯）
