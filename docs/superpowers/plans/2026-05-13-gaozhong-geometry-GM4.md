# 高中几何教程 GM4 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development + rendering-math-figures

**Goal:** 完成 Part 4 圆与方程（4 章，含 1 ⭐ 模板 D）+ 6-8 张配图。

**前序里程碑:** GM0-GM3

---

## 文件结构

```
gaozhong_math/geometry/part4-circle-and-equation/
├── 01-circle-standard-equation.md   # 标准方程 + 一般方程
├── 02-line-circle-position.md       # ⭐ 直线与圆位置关系 + 弦长 + 切线
├── 03-circle-circle-position.md     # 圆与圆位置关系
└── 04-circle-applications.md        # 含参圆 + 综合
```

---

## Task 1: part4/01 圆的方程（模板 C，2500-3500 字）

Title: `# 圆的标准方程与一般方程`

涵盖：
- **标准方程**：$(x-a)^2 + (y-b)^2 = r^2$（圆心 $(a, b)$、半径 $r$）
- **一般方程**：$x^2 + y^2 + Dx + Ey + F = 0$（$D^2 + E^2 - 4F > 0$）
  - 圆心 $(-D/2, -E/2)$、半径 $r = \frac{1}{2}\sqrt{D^2 + E^2 - 4F}$
- 两种方程互转换
- 通过三点求圆：① 用两条中垂线交点 ② 设一般方程代入三点解 D, E, F
- 典型应用 3 例
- 易错点：$D^2 + E^2 - 4F > 0$ 是圆的必要条件（否则是点或空集）；标准方程的开方
- 自测题 4 题

**配图 2 张**：
- `geo-p4-01-1.tex`：圆心 $C(a, b)$、半径 $r$、动点 $P(x, y)$ 在圆上的几何示意
- `geo-p4-01-2.tex`：含 $D^2+E^2-4F$ 判别：圆 / 点 / 空集 三种情形（一图三栏）

---

## Task 2: part4/02 直线与圆位置 ⭐（模板 D，4000-5500 字）

Title: `# 直线与圆的位置关系`

**一例速记**：
> **3 种位置**：相离（$d > r$）/ 相切（$d = r$）/ 相交（$d < r$）  
> **$d$ = 圆心到直线距离 $= \frac{|Aa + Bb + C|}{\sqrt{A^2 + B^2}}$**  
> **弦长公式**：相交时 $|AB| = 2\sqrt{r^2 - d^2}$  
> **切线方程**：过圆 $x^2+y^2=r^2$ 上一点 $(x_0, y_0)$ 的切线为 $x_0 x + y_0 y = r^2$

**引入题**：求过 $A(2, 3)$ 与圆 $x^2 + y^2 = 4$ 相切的切线方程。

**思维路径还原** 15-20 行：判 $A$ 在圆外（$4 + 9 > 4$）→ 设切线 $y - 3 = k(x - 2)$ → 用圆心到直线距离 = 半径解 $k$，注意切线斜率不存在的情形 $x = 2$（验证 = $r$）。

**抽象成方法**：判位置（$d$ vs $r$）→ 求弦长（$2\sqrt{r^2-d^2}$）→ 切线方程（在圆上 vs 圆外两种情形）

**方法变形**：
- 圆外一点求切线（常含两条 + 斜率不存在情形）
- 含参直线与圆 → 解 $d$ vs $r$ 的不等式
- 弦中点 + 弦长 → 用半弦 $\sqrt{r^2 - d^2}$ + 中点距圆心

**思考路标 ≥8 条**：
- 看到"直线与圆" → 用 $d$ 与 $r$ 比较
- 看到"切线" → $d = r$
- 看到"弦长" → $2\sqrt{r^2-d^2}$
- 圆外一点切线必有两条（斜率不存在情形要单独考虑）
- 含参 → 转 $d$ 与 $r$ 的不等式

例题 3 例（直线与圆相交求弦长 / 圆外切线 / 含参讨论位置）+ 自测 5 题。

**配图 2 张**：
- `geo-p4-02-1.tex`：直线与圆的 3 种位置关系（相离 / 相切 / 相交）
- `geo-p4-02-2.tex`：圆外一点 $A$ 引两条切线，标切点 $T_1, T_2$

---

## Task 3: part4/03 圆与圆位置（模板 C，2500-3500 字）

Title: `# 圆与圆的位置关系`

涵盖：
- **5 种位置**（用圆心距 $d$ vs 半径 $r_1, r_2$）：
  - 外离 $d > r_1 + r_2$
  - 外切 $d = r_1 + r_2$
  - 相交 $|r_1 - r_2| < d < r_1 + r_2$
  - 内切 $d = |r_1 - r_2|$
  - 内含 $d < |r_1 - r_2|$
- **公共弦方程**：两圆相减消去 $x^2 + y^2$ 项得到的方程就是公共弦所在直线
- 公共弦长：用一圆半径 + 圆心到公共弦距离
- 典型应用 3 例（判位置 / 公共弦 / 含参讨论）
- 易错点 5 条
- 自测题 5 题

**配图 2 张**：
- `geo-p4-03-1.tex`：5 种圆与圆位置关系（一图五栏）
- `geo-p4-03-2.tex`：公共弦（两圆相交，标公共弦所在直线 + 公共弦端点）

---

## Task 4: part4/04 圆综合应用（模板 C，2500-3500 字）

Title: `# 圆的综合应用`

涵盖：
- 含参圆方程讨论（$D^2 + E^2 - 4F > 0$ 限制）
- 圆与函数 / 不等式综合
- 用圆求几何最值（如点到圆上最远 / 最近）
- 轨迹方程问题（动点轨迹是圆的识别）
- 典型应用 3 例
- 易错点 5 条
- 自测题 4 题

无配图。

---

## Task 5: GM4 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|待补' gaozhong_math/geometry/part4-circle-and-equation 2>/dev/null
ls gaozhong_math/geometry/part4-circle-and-equation/
ls gaozhong_math/geometry/figures/svg/ | grep '^geo-p4-'
printf '\n---\n**GM4 完成于：2026-05-13**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-geometry-GM4.md
git add docs/superpowers/plans/2026-05-13-gaozhong-geometry-GM4.md
git commit -m "docs(gaozhong/geometry): mark GM4 milestone complete"
git push origin master
```

---

## 调度

**2 个 sonnet subagent 并行：**
- A: 01 圆方程 + 02 ⭐ 直线与圆（+ 4 张图）
- B: 03 圆与圆 + 04 综合（+ 2 张图）

每章独立 commit。
