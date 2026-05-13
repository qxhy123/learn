# 高中几何教程 GM1 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development + rendering-math-figures

**Goal:** 完成 Part 1 平面向量基础（4 章）+ 5-8 张配图。

**Architecture:** 3 章模板 C + 1 章 ⭐ 模板 D（基底与坐标）。每章 + 配图独立提交。

**Spec 参考:** `docs/superpowers/specs/2026-05-13-gaozhong-geometry-design.md`
**前序里程碑:** GM0（已完成 README + preface + 12 toolkit + figures 基建）

---

## 文件结构

```
gaozhong_math/geometry/part1-plane-vectors-basics/
├── 01-vector-concept.md             # 向量定义、相等、共线、零向量
├── 02-vector-addition.md            # 加减法（三角形 / 平行四边形法则）
├── 03-scalar-multiplication.md      # 数乘 + 共线定理
└── 04-basis-and-coordinates.md      # ⭐ 基底、平面向量基本定理、坐标表示
```

配图：5-8 张 SVG（向量箭头 / 加法平行四边形 / 数乘示意 / 基底分解）。

---

## Task 1: part1/01 向量概念（模板 C，2500-3500 字）

Title: `# 向量的概念`

涵盖：
- **向量定义**：既有大小又有方向的量。区别 vs 数量（scalar）
- **几何表示**：有向线段 $\vec{AB}$；起点 $A$、终点 $B$
- **代数符号**：$\vec{a}, \vec{b}$；模 $|\vec{a}|$
- **特殊向量**：
  - 零向量 $\vec{0}$（模为 0，方向任意）
  - 单位向量（模为 1）
- **关系**：
  - 相等向量：模相同方向相同（**与起点无关**）
  - 共线向量（平行向量）：方向相同或相反（零向量与任何向量共线）
  - 相反向量：模相同方向相反 $\vec{a}$ 与 $-\vec{a}$
- 典型应用 3 例（向量与有向线段区分 / 共线判定 / 相等向量识别）
- 易错点：① 向量与起点无关（位置不固定）② 零向量与任何向量共线 ③ 不能说"两向量相等"先要看模和方向都相同
- 自测题 4 题

**配图**（1-2 张）：
- `geo-p1-01-1.tex`：有向线段表示向量 $\vec{AB}$（起点 $A$、终点 $B$、箭头）
- `geo-p1-01-2.tex`：相等向量（位置不同但模、方向相同）+ 共线向量示意

---

## Task 2: part1/02 向量加减法（模板 C，2500-3500 字）

Title: `# 向量的加法与减法`

涵盖：
- **三角形法则**：$\vec{AB} + \vec{BC} = \vec{AC}$（首尾相接）
- **平行四边形法则**：$\vec{AB} + \vec{AD} = \vec{AC}$（$ABCD$ 为平行四边形）
- 加法运算律：交换律 $\vec{a} + \vec{b} = \vec{b} + \vec{a}$；结合律 $(\vec{a} + \vec{b}) + \vec{c} = \vec{a} + (\vec{b} + \vec{c})$
- **减法**：$\vec{a} - \vec{b} = \vec{a} + (-\vec{b})$
- 几何意义：$\vec{OA} - \vec{OB} = \vec{BA}$（共起点的两向量相减 = 起点指向终点 - 起点指向终点 = 从减数指向被减数）
- 模的三角不等式：$||\vec{a}| - |\vec{b}|| \leq |\vec{a} + \vec{b}| \leq |\vec{a}| + |\vec{b}|$
- 典型应用 3 例（多向量首尾相接 / 减法几何意义 / 模长求范围）
- 易错点：减法方向（$\vec{OA} - \vec{OB}$ 是 $\vec{BA}$ 不是 $\vec{AB}$）；模的三角不等式取等条件
- 自测题 5 题

**配图**（2 张）：
- `geo-p1-02-1.tex`：三角形法则（首尾相接得和向量）
- `geo-p1-02-2.tex`：平行四边形法则 + 减法（$\vec{OA} - \vec{OB} = \vec{BA}$）

---

## Task 3: part1/03 数乘向量（模板 C，2500-3500 字）

Title: `# 向量的数乘`

涵盖：
- **数乘定义**：$\lambda \vec{a}$（$\lambda$ 实数）的模 $|\lambda||\vec{a}|$；方向：$\lambda > 0$ 同向，$\lambda < 0$ 反向，$\lambda = 0$ 零向量
- **运算律**：$\lambda(\mu\vec{a}) = (\lambda\mu)\vec{a}$；$(\lambda + \mu)\vec{a} = \lambda\vec{a} + \mu\vec{a}$；$\lambda(\vec{a} + \vec{b}) = \lambda\vec{a} + \lambda\vec{b}$
- **共线向量定理**（重要！）：$\vec{a}, \vec{b}$ 共线 $\Leftrightarrow$ 存在唯一实数 $\lambda$ 使 $\vec{b} = \lambda\vec{a}$（$\vec{a} \neq \vec{0}$）
- 用于判定三点共线：$A, B, C$ 三点共线 $\Leftrightarrow \vec{AB} = \lambda \vec{AC}$
- 典型应用 3 例（数乘求模 / 三点共线 / 中点公式 $\vec{OM} = \frac{1}{2}(\vec{OA} + \vec{OB})$）
- 易错点：$\lambda = 0$ 时结果是零向量；共线定理要求 $\vec{a} \neq \vec{0}$
- 自测题 5 题

**配图**（1 张）：
- `geo-p1-03-1.tex`：数乘示意（$\vec{a}, 2\vec{a}, -\vec{a}, \frac{1}{2}\vec{a}$ 四条向量）

---

## Task 4: part1/04 基底与坐标 ⭐（模板 D，4000-5500 字）

Title: `# 基底、平面向量基本定理与坐标表示`

**一例速记**：
> **平面向量基本定理**：若 $\vec{e_1}, \vec{e_2}$ 不共线，则平面内**任意**向量 $\vec{a}$ 有**唯一**实数对 $(\lambda_1, \lambda_2)$ 使 $\vec{a} = \lambda_1 \vec{e_1} + \lambda_2 \vec{e_2}$  
> 当取**正交基**（垂直且模为 1）：$\vec{e_1} = (1, 0), \vec{e_2} = (0, 1)$ → $\vec{a} = (x, y)$ 即坐标表示

**引入题**：已知 $\vec{e_1}, \vec{e_2}$ 不共线，$\vec{a} = 2\vec{e_1} - 3\vec{e_2}$，$\vec{b} = \vec{e_1} + 4\vec{e_2}$，求 $\vec{a} + \vec{b}$ 和 $2\vec{a} - \vec{b}$ 用 $\vec{e_1}, \vec{e_2}$ 表示。

**思维路径还原** 15-20 行（用 `>` 引用块）：识别"用基底表示向量" → 直接按运算律展开 → 合并 $\vec{e_1}, \vec{e_2}$ 的系数。

**抽象成方法**：
- 选定基底 → 任意向量唯一分解
- 坐标 = 基底系数（正交基底下）
- 坐标运算：$(x_1, y_1) + (x_2, y_2) = (x_1 + x_2, y_1 + y_2)$；$\lambda(x, y) = (\lambda x, \lambda y)$
- 模长公式：$|\vec{a}| = \sqrt{x^2 + y^2}$
- 向量起点终点的坐标关系：$\vec{AB} = (x_B - x_A, y_B - y_A)$

**方法变形**：
- 非正交基底下的坐标（系数表示）
- 用坐标判定共线：$x_1 y_2 - x_2 y_1 = 0$
- 中点公式坐标版：$M = (\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2})$

**思考路标 ≥8 条**：
- 看到"任意向量" → 想基底分解
- 看到正交基底 → 直接坐标
- 看到"$A, B, C$ 共线" → 用 $\vec{AB} = \lambda \vec{AC}$ 或坐标判定
- 看到"$\vec{AB}$" → 写 $(x_B - x_A, y_B - y_A)$
- 看到"模长" → $\sqrt{x^2 + y^2}$
- 选基底要不共线（不一定垂直，但题目常用正交基简化）

例题 3 例（基底表示 / 共线判定 / 三角形重心坐标）+ 自测题 5 题。

**配图**（1-2 张）：
- `geo-p1-04-1.tex`：基底分解（$\vec{a} = \lambda_1 \vec{e_1} + \lambda_2 \vec{e_2}$ 平行四边形分解）
- `geo-p1-04-2.tex`：正交基底 + 坐标表示

---

## Task 5: GM1 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' gaozhong_math/geometry/part1-plane-vectors-basics 2>/dev/null
ls gaozhong_math/geometry/part1-plane-vectors-basics/
ls gaozhong_math/geometry/figures/svg/ | grep '^geo-p1-'
printf '\n---\n**GM1 完成于：2026-05-13**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-geometry-GM1.md
git add docs/superpowers/plans/2026-05-13-gaozhong-geometry-GM1.md
git commit -m "docs(gaozhong/geometry): mark GM1 milestone complete"
git push origin master
```

---

## 调度

**2 个 sonnet subagent 并行：**
- A: Task 1 (01 概念) + Task 2 (02 加减法) + 对应 figures（3-4 张）
- B: Task 3 (03 数乘) + Task 4 (04 ⭐ 基底坐标) + 对应 figures（2-3 张）

每章 + 对应图独立 commit。
