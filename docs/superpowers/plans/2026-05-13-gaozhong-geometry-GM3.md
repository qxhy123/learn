# 高中几何教程 GM3 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development + rendering-math-figures

**Goal:** 完成 Part 3 直线与方程（5 章，含 2 ⭐ 模板 D）+ 8-10 张配图。

**前序里程碑:** GM0-GM2

---

## 文件结构

```
gaozhong_math/geometry/part3-line-and-equation/
├── 01-slope-and-inclination.md     # 斜率与倾斜角
├── 02-line-equation-forms.md       # 5 种方程形式
├── 03-line-position-relations.md   # ⭐ 平行 / 垂直 / 重合
├── 04-distance-formulas.md         # ⭐ 点到直线 / 两平行线距离
└── 05-line-applications.md         # 含参讨论 + 综合
```

---

## Task 1: part3/01 斜率与倾斜角（模板 C，2500-3500 字）

Title: `# 直线的斜率与倾斜角`

涵盖：
- **倾斜角** $\alpha$：直线与 $x$ 轴**正方向**的夹角，$\alpha \in [0, \pi)$
- **斜率** $k = \tan\alpha$（$\alpha \neq \frac{\pi}{2}$）
- 已知两点 $(x_1, y_1), (x_2, y_2)$ 求斜率：$k = \frac{y_2 - y_1}{x_2 - x_1}$（$x_1 \neq x_2$）
- 倾斜角与斜率的对应关系表：
  - $\alpha = 0$ → $k = 0$（水平线）
  - $0 < \alpha < \frac{\pi}{2}$ → $k > 0$
  - $\alpha = \frac{\pi}{2}$ → $k$ 不存在（垂直线 $x = c$）
  - $\frac{\pi}{2} < \alpha < \pi$ → $k < 0$
- 典型应用 3 例（求斜率 / 求倾斜角 / 含参讨论）
- 易错点：$\alpha = \frac{\pi}{2}$ 时 $k$ 不存在；倾斜角范围 $[0, \pi)$ 不含 $\pi$
- 自测题 4 题

**配图 2 张**：
- `geo-p3-01-1.tex`：倾斜角与斜率的几何示意
- `geo-p3-01-2.tex`：四种倾斜角情形（水平 / 锐角 / 直角 / 钝角）对应斜率

---

## Task 2: part3/02 直线方程的 5 种形式（模板 C，2500-3500 字）

Title: `# 直线方程的五种形式`

涵盖：
- **点斜式** $y - y_0 = k(x - x_0)$（需 $k$ 存在）
- **斜截式** $y = kx + b$（$b$ 截距）
- **两点式** $\frac{y - y_1}{y_2 - y_1} = \frac{x - x_1}{x_2 - x_1}$
- **截距式** $\frac{x}{a} + \frac{y}{b} = 1$（$a, b$ 为 $x, y$ 轴截距，$a, b \neq 0$）
- **一般式** $Ax + By + C = 0$（$A, B$ 不同时为 0）
- **5 种形式的转换 + 选用场景表**
- 不能用点斜式的情形：$k$ 不存在（垂直 $x$ 轴的线 $x = x_0$）
- 不能用截距式的情形：过原点或平行轴
- 典型应用 3 例（已知斜率 / 已知两点 / 已知截距）
- 易错点 5 条
- 自测题 5 题

**配图 2 张**：
- `geo-p3-02-1.tex`：点斜式几何示意
- `geo-p3-02-2.tex`：截距式（标 $x$ 轴截距 $a$、$y$ 轴截距 $b$）

---

## Task 3: part3/03 直线位置关系 ⭐（模板 D，4000-5500 字）

Title: `# 直线的位置关系`

**一例速记**：
> **平行**：$l_1 \parallel l_2 \Leftrightarrow k_1 = k_2$ 且 $b_1 \neq b_2$（或一般式 $\frac{A_1}{A_2} = \frac{B_1}{B_2} \neq \frac{C_1}{C_2}$）  
> **垂直**：$l_1 \perp l_2 \Leftrightarrow k_1 \cdot k_2 = -1$（或一般式 $A_1 A_2 + B_1 B_2 = 0$）  
> **重合**：$k_1 = k_2$ 且 $b_1 = b_2$

**引入题**：$l_1: 2x + y - 1 = 0$，$l_2: x + ay + 2 = 0$，若 $l_1 \parallel l_2$，求 $a$。

**思维路径还原** 15-20 行：用一般式判别 $\frac{A_1}{A_2} = \frac{B_1}{B_2} \neq \frac{C_1}{C_2}$ → $\frac{2}{1} = \frac{1}{a} \neq \frac{-1}{2}$ → $a = \frac{1}{2}$ → 验证 $\frac{-1}{2} \neq \frac{1}{2}$（OK）。

**抽象成方法**：3 种位置关系的判别完整表 + 注意"垂直"含 $k$ 不存在情形（$l_1$ 平行 $x$ 轴 $\perp$ $l_2$ 垂直 $x$ 轴）。

**方法变形**：
- 含参讨论平行 / 垂直 → 含 $a$ 的参数方程
- 一般式 vs 斜截式 → 用一般式可避免讨论 $k$ 是否存在

**思考路标 ≥8 条**：
- 看到"平行" → $k_1 = k_2$（或一般式比例）
- 看到"垂直" → $k_1 k_2 = -1$（或一般式 $A_1A_2 + B_1B_2 = 0$）
- 看到"含参" → 用一般式（更稳妥）
- 注意垂直特例：$x = a$ 与 $y = b$ 垂直
- 重合 = 平行 + 同截距（或一般式三比相等）

例题 3 例（含参平行 / 垂直特例 / 一般式判别）+ 自测 5 题。

**配图 2 张**：
- `geo-p3-03-1.tex`：平行 / 垂直 / 重合三种情形
- `geo-p3-03-2.tex`：垂直时含参陷阱（$k$ 不存在的情形）

---

## Task 4: part3/04 距离公式 ⭐（模板 D，4000-5500 字）

Title: `# 点到直线的距离与两平行线距离`

**一例速记**：
> **点到直线距离公式**：点 $P(x_0, y_0)$ 到直线 $Ax + By + C = 0$ 的距离 $d = \frac{|Ax_0 + By_0 + C|}{\sqrt{A^2 + B^2}}$  
> **两平行线距离**：$Ax + By + C_1 = 0$ 与 $Ax + By + C_2 = 0$（系数相同）的距离 $d = \frac{|C_1 - C_2|}{\sqrt{A^2 + B^2}}$  
> **两点距离公式**：$d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$

**引入题**：求点 $P(3, 1)$ 到直线 $l: 4x - 3y + 1 = 0$ 的距离，并求与 $l$ 平行且过 $P$ 的直线方程。

**思维路径还原** 15-20 行：直接代距离公式 $d = \frac{|4 \cdot 3 - 3 \cdot 1 + 1|}{5} = \frac{10}{5} = 2$。平行直线斜率相同，用点斜式得 $4x - 3y - 9 = 0$。

**抽象成方法**：3 种距离公式（点-直线 / 平行线 / 两点）+ 何时该用哪个

**方法变形**：
- 含绝对值方程 → 距离条件
- 距离与圆心 / 切线的关联（铺垫 Part 4）
- 平行线间距离用同系数公式（需先化系数相同）

**思考路标 ≥8 条**：
- 点到直线距离 → 直接代公式（注意 $|\cdot|$）
- 两平行线 → 系数化相同后用 $|C_1 - C_2| / \sqrt{A^2+B^2}$
- 看到"距离 = $r$" → 圆相切（铺垫 Part 4）
- 点是否在直线上 → 代入符号判
- 含参距离条件 → 解一元方程或不等式
- 注意：两平行线必须 $A, B$ 同号同值

例题 3 例（含参点 / 平行线距离 / 距离条件解参）+ 自测 5 题。

**配图 2 张**：
- `geo-p3-04-1.tex`：点到直线距离（点 $P$、直线 $l$、垂足 $H$、距离 $d$）
- `geo-p3-04-2.tex`：两平行线距离（两线 $l_1, l_2$ + 公共垂线）

---

## Task 5: part3/05 直线综合应用（模板 C，2500-3500 字）

Title: `# 直线综合应用`

涵盖：
- 含参直线讨论
- 直线与点的位置关系
- 对称问题（点关于直线对称 / 直线关于点对称 / 直线关于直线对称）
- 与函数 / 不等式结合
- 典型应用 3 例（含参 / 对称 / 综合）
- 易错点 5 条
- 自测题 4 题

无配图。

---

## Task 6: GM3 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|待补' gaozhong_math/geometry/part3-line-and-equation 2>/dev/null
ls gaozhong_math/geometry/part3-line-and-equation/
ls gaozhong_math/geometry/figures/svg/ | grep '^geo-p3-'
printf '\n---\n**GM3 完成于：2026-05-13**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-geometry-GM3.md
git add docs/superpowers/plans/2026-05-13-gaozhong-geometry-GM3.md
git commit -m "docs(gaozhong/geometry): mark GM3 milestone complete"
git push origin master
```

---

## 调度

**3 个 sonnet subagent 并行：**
- A: 01 斜率倾斜 + 02 5 种方程形式（+ 4 张图）
- B: 03 ⭐ 位置关系 + 04 ⭐ 距离公式（+ 4 张图）
- C: 05 综合应用

每章独立 commit。

---
**GM3 完成于：2026-05-13**
