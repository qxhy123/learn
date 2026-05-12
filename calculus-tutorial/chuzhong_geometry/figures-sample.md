# 图形渲染样例（预渲染方案）

源码在 `figures/src/{tikz,asy}/`，SVG 产物在 `figures/svg/`，由 `figures/render.sh` 批量生成。markdown 直接用 `![]()` 引用 SVG —— VSCode 原生预览即可显示，无需 MPE 执行代码。

---

## 样例 1：TikZ 基础三角形

源码：[`figures/src/tikz/sample-triangle.tex`](figures/src/tikz/sample-triangle.tex)

![基础三角形](figures/svg/sample-triangle.svg)

---

## 样例 2：TikZ 手拉手模型

源码：[`figures/src/tikz/sample-handshake.tex`](figures/src/tikz/sample-handshake.tex)

![手拉手模型](figures/svg/sample-handshake.svg)

---

## 样例 3：Asymptote 圆周角

源码：[`figures/src/asy/sample-inscribed-angle.asy`](figures/src/asy/sample-inscribed-angle.asy)

![圆周角](figures/svg/sample-inscribed-angle.svg)

*（如尚未渲染，运行 `cd figures && ./render.sh` 生成；Asymptote 需要 ghostscript：`brew install ghostscript`）*

---

## 工作流回顾

1. 写 / 改源码 → `figures/src/tikz/xxx.tex` 或 `figures/src/asy/xxx.asy`
2. 渲染 → `cd figures && ./render.sh`（批量）或 `./render.sh src/tikz/xxx.tex`（单文件）
3. markdown 引用 → `![描述](figures/svg/xxx.svg)`
4. VSCode 预览 → 直接看，零外部依赖

源码与 SVG 都进 git，需要时随时改。
