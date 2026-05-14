# 高中几何教程 GM6 里程碑实施计划（种子阶段 appendix）

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development

**Goal:** 完成 `gaozhong_math/geometry/appendix/` 子集——公式表、模型图集、高一下 + 高二上前半 80 题题库、详解。

**Architecture:** 4 个 markdown 文件，覆盖 GM0-GM5 已完成的 23 章内容（Part 1, 2, 3, 4, 8）。每文件独立提交。

**Spec 参考:** `docs/superpowers/specs/2026-05-13-gaozhong-geometry-design.md`
**前序里程碑:** GM0-GM5（已完成 23 章正文 + 12 toolkit）
**参考模式:** `gaozhong_math/algebra/appendix/` 的种子阶段子集（A + B + C-seed + F-seed）

---

## 文件结构

```
gaozhong_math/geometry/appendix/
├── A-formula-sheet.md      # 公式表（高一下 + 高二上前半覆盖范围）
├── B-model-atlas.md        # 模型图集（高一下 + 高二上前半典型套路）
├── C-exercises-seed.md     # 种子 80 题（按 Part 1, 2, 3, 4, 8 分布，按难度分层）
└── F-solutions-seed.md     # 80 题详解
```

字数：A ~ 3500-4500、B ~ 3500-4500、C ~ 2800-3800、F ~ 8000-12000（含完整解析）

---

## Task 1: appendix/A 公式表

**Files:** Create: `gaozhong_math/geometry/appendix/A-formula-sheet.md`

Title: `# 附录 A：高中几何公式表（种子）`

**结构（按 Part 顺序整理）：**

### Part 1 平面向量基础
- 向量定义、模、零向量、单位向量、相等 / 共线 / 相反
- 加减法：三角形法则、平行四边形法则、$\vec{OA} - \vec{OB} = \vec{BA}$
- 数乘：$\lambda\vec{a}$ 的模与方向；运算律 3 条；共线定理 $\vec{b} = \lambda\vec{a}$
- 平面向量基本定理 + 正交基坐标 $(x, y)$
- 坐标运算：加减、数乘、$\vec{AB} = (x_B-x_A, y_B-y_A)$、模 $\sqrt{x^2+y^2}$、共线 $x_1 y_2 - x_2 y_1 = 0$
- 中点公式、重心公式

### Part 2 平面向量应用
- 数量积 $\vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}|\cos\theta = x_1 x_2 + y_1 y_2$
- 夹角 $\cos\theta = \frac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|}$
- 垂直 $\vec{a} \perp \vec{b} \Leftrightarrow \vec{a}\cdot\vec{b} = 0$
- 投影 $= \frac{\vec{a}\cdot\vec{b}}{|\vec{a}|}$
- 模平方 $\vec{a}\cdot\vec{a} = |\vec{a}|^2$；$|\vec{a}\pm\vec{b}|^2 = |\vec{a}|^2 \pm 2\vec{a}\cdot\vec{b} + |\vec{b}|^2$
- 极化恒等式
- 三点共线 $\vec{OC} = \lambda\vec{OA} + (1-\lambda)\vec{OB}$
- 分点比公式

### Part 3 直线与方程
- 倾斜角 $\alpha \in [0, \pi)$、斜率 $k = \tan\alpha$
- 两点斜率公式 $k = \frac{y_2-y_1}{x_2-x_1}$
- 5 种方程形式（点斜 / 斜截 / 两点 / 截距 / 一般）+ 选用场景表
- 平行 / 垂直 / 重合判定（含斜截式与一般式两种）
- 距离公式：点到直线、两平行线、两点

### Part 4 圆与方程
- 标准方程 $(x-a)^2 + (y-b)^2 = r^2$
- 一般方程 $x^2+y^2+Dx+Ey+F = 0$（$D^2+E^2-4F > 0$）+ 圆心半径公式
- 直线与圆 3 种位置（$d$ vs $r$）+ 弦长 $2\sqrt{r^2-d^2}$ + 切线方程
- 圆与圆 5 种位置（$d$ vs $r_1, r_2$）+ 公共弦方程

### Part 8 立体几何初步
- 3 大公理 + 3 条推论
- 三视图三规律（长对正 / 高平齐 / 宽相等）
- 异面直线夹角 $\theta \in (0, \pi/2]$、求法（综合 / 向量）
- 线面位置 3 种、判定 + 性质（平行 / 垂直）
- 面面位置 2 种、判定 + 性质（平行 / 垂直）
- 二面角及平面角作法、范围 $[0, \pi]$
- 表面积 / 体积公式族（柱 / 锥 / 台 / 球）
- 球的内切外接经典模型表

每节末附"易错提醒"2-3 条。

Commit: `docs(gaozhong/geometry): add appendix A (formula sheet for seed phase)`

---

## Task 2: appendix/B 模型图集

**Files:** Create: `gaozhong_math/geometry/appendix/B-model-atlas.md`

Title: `# 附录 B：高中几何模型图集（种子）`

**12 个核心模型**（每个 250-400 字，覆盖种子阶段所有 Part）：

1. **向量分解模型**：用基底表示未知向量（→ Part 1/04, 2/02）
2. **数量积"三用途"**：求夹角 / 判垂直 / 求投影（→ Part 2/01）
3. **三点共线 + 分点比**：$\vec{OC} = \lambda\vec{OA} + (1-\lambda)\vec{OB}$（→ Part 2/02）
4. **直线方程选择树**：5 种形式何时选哪种（→ Part 3/02）
5. **直线位置关系判别**：含一般式 + 斜截式双判别（→ Part 3/03）
6. **距离公式速算**：点-线 / 两平行线 / 两点（→ Part 3/04）
7. **圆方程双形式**：标准 vs 一般（含 $D^2+E^2-4F > 0$ 判别）（→ Part 4/01）
8. **直线与圆"3+1"**：3 种位置 + 弦长公式（→ Part 4/02）
9. **圆与圆 5 种位置**：完整判别（→ Part 4/03）
10. **异面直线夹角"综合 vs 向量"**：两种解法对比（→ Part 8/02）
11. **线面垂直判定**："$\perp$ 两相交直线"模型（→ Part 8/03）
12. **二面角 3 步法**：找棱 → 作平面角 → 算（→ Part 8/04）

每模型包含：**触发条件** / **核心思路** / **关键步骤** / **典型题** / **关联章节**

末尾附"思维方法网"——12 toolkit 篇与 12 模型关联表。

Commit: `docs(gaozhong/geometry): add appendix B (model atlas for seed phase)`

---

## Task 3: appendix/C 种子 80 题题库

**Files:** Create: `gaozhong_math/geometry/appendix/C-exercises-seed.md`

Title: `# 附录 C：高中几何种子 80 题题库`

**分布**（按 Part 章节，三档难度）：

| Part | 基础 | 中档 | 提升 | 小计 |
|---|---|---|---|---|
| Part 1 平面向量基础 | 4 | 4 | 2 | 10 |
| Part 2 平面向量应用 | 4 | 4 | 2 | 10 |
| Part 3 直线与方程 | 6 | 6 | 3 | 15 |
| Part 4 圆与方程 | 5 | 5 | 3 | 13 |
| Part 8 立体几何初步 | 8 | 12 | 6 | 26 |
| 综合（跨 Part） | — | 4 | 2 | 6 |
| **合计** | **27** | **35** | **18** | **80** |

**题号约定：** C.01 - C.80，按 Part 顺序连续编号。

**每题包含：**
- 题号 / [难度] / Part 章节
- 题面（LaTeX 数学）

**举例：**
```
**C.15** [中档] Part 3/03  
若直线 $l_1: ax + 2y - 1 = 0$ 与 $l_2: x + (a-1)y + 1 = 0$ 平行，求 $a$。
```

Commit: `docs(gaozhong/geometry): add appendix C (80 problems for seed phase)`

---

## Task 4: appendix/F 80 题详解

**Files:** Create: `gaozhong_math/geometry/appendix/F-solutions-seed.md`

Title: `# 附录 F：种子 80 题详解`

**每题解析包含：**
1. 题号回顾（再次列出题面）
2. **思路**：1-3 行点明本题考察什么、套路是什么（→ 引用 toolkit / 章节）
3. **解答**：完整推导，每步骤的关键代数变形展示
4. **答案**：用 $\boxed{...}$ 框出
5. **总结**：1-2 句指出本题的"识题特征"

要求：
- 答案"看得懂、学得会"
- 中档以上题目展示完整思维链
- 关键步骤标注（"换元"、"分类"、"建系"等）

Commit: `docs(gaozhong/geometry): add appendix F (full solutions for 80 problems)`

---

## Task 5: GM6 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' gaozhong_math/geometry/appendix
ls gaozhong_math/geometry/appendix
printf '\n---\n**GM6 完成于：2026-05-14**\n' >> docs/superpowers/plans/2026-05-14-gaozhong-geometry-GM6.md
git add docs/superpowers/plans/2026-05-14-gaozhong-geometry-GM6.md
git commit -m "docs(gaozhong/geometry): mark GM6 milestone complete (seed phase done)"
git push origin master
```

---

## 调度

**2 个 sonnet subagent 并行：**
- A: appendix A（公式表）+ appendix B（模型图集）
- B: appendix C（80 题题库）+ appendix F（80 题详解）

每文件独立 commit。完成后报告：commit SHA、字数、题号分布。
