# 图形渲染样例

此文件用于验证 VSCode + Markdown Preview Enhanced + LaTeX/Asymptote 的渲染环境。打开本文件预览，应能看到三张图。

---

## 样例 1：TikZ 简单图——三角形与标注

最基础的 TikZ 用法：画一个三角形，标注顶点、边、角。这种简单图将占图集的大多数。

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[scale=1.2]
  % 三角形三顶点
  \coordinate (A) at (0, 0);
  \coordinate (B) at (4, 0);
  \coordinate (C) at (1.5, 2.8);

  % 三边
  \draw[thick] (A) -- (B) -- (C) -- cycle;

  % 顶点标签
  \node[below left]  at (A) {$A$};
  \node[below right] at (B) {$B$};
  \node[above]       at (C) {$C$};

  % 角标记（小弧线）
  \draw (0.5, 0) arc (0:62:0.5);
  \node at (0.85, 0.25) {\small $\alpha$};
\end{tikzpicture}
\end{document}
```

---

## 样例 2：TikZ 中等图——手拉手模型

这是教程里典型的模型图（part3/congruence/03 章的"一图速记"）：两个共顶点等腰三角形。

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[scale=1.0]
  % 公共顶点
  \coordinate (O) at (0, 0);
  % 等腰三角形 OAB
  \coordinate (A) at (-2.5, 1.5);
  \coordinate (B) at (-2.5, -1.5);
  % 等腰三角形 OCD（OAB 绕 O 旋转某角度）
  \coordinate (C) at (1.5, 2.5);
  \coordinate (D) at (2.9, 0.2);

  % 两个等腰三角形
  \draw[thick] (O) -- (A) -- (B) -- cycle;
  \draw[thick] (O) -- (C) -- (D) -- cycle;

  % 连接两腰端点 AC、BD（手拉手）
  \draw[red, thick, dashed] (A) -- (C);
  \draw[red, thick, dashed] (B) -- (D);

  % 顶点标签
  \node[above right] at (O) {$O$};
  \node[left]        at (A) {$A$};
  \node[left]        at (B) {$B$};
  \node[above]       at (C) {$C$};
  \node[right]       at (D) {$D$};

  % 标记相等的腰（用短横线）
  \foreach \p/\q in {O/A, O/B, O/C, O/D} {
    \draw ($(\p)!0.5!(\q)$) +(-0.08,-0.08) -- +(0.08,0.08);
  }
\end{tikzpicture}
\end{document}
```

---

## 样例 3：Asymptote 复杂图——圆周角定理

更精细的几何图（part5/03 章圆周角定理）。Asymptote 在画圆和曲线时比 TikZ 表达力更强。

```asymptote
size(8cm);
import geometry;

// 圆心 O，半径 r
pair O = (0, 0);
real r = 2;
draw(circle(O, r));

// 弦端点 A, B 与圆周角顶点 C
pair A = r * dir(210);
pair B = r * dir(330);
pair C = r * dir(90);

// 弦 AB、AC、BC
draw(A -- B);
draw(A -- C);
draw(B -- C);

// 半径 OA、OB（虚线）
draw(O -- A, dashed);
draw(O -- B, dashed);

// 标记圆心角 ∠AOB 与圆周角 ∠ACB
markangle(Label("$2\theta$", Relative(0.5)), A, O, B, radius=0.6cm);
markangle(Label("$\theta$",  Relative(0.5)), A, C, B, radius=0.5cm);

// 顶点标签
label("$O$", O, NE);
label("$A$", A, SW);
label("$B$", B, SE);
label("$C$", C, N);

// 关键文字
label("圆周角 $\angle ACB$", (0, -2.6), S);
```

---

## 验证要点

打开 VSCode → 命令面板 → `Markdown Preview Enhanced: Open Preview`：

- [ ] 样例 1 显示一个三角形（带顶点字母 A/B/C 和角度 $\alpha$）
- [ ] 样例 2 显示两个共顶点等腰三角形，红色虚线连接（手拉手）
- [ ] 样例 3 显示一个圆，中心 O，圆周上三点 A、B、C，标注圆周角 $\theta$ 和圆心角 $2\theta$

如果三张都正常显示 → 环境就绪，可以开始批量补图。

如果某张报错或不显示，把错误信息告诉我，常见问题：
- TikZ 库未加载（需在 `\usepackage{tikz}` 后加 `\usetikzlibrary{...}`）
- Asymptote 未在 PATH 中
- MPE 的 LaTeX 引擎设置（默认是 `pdflatex`，可在设置里改）
