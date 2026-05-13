# 高中几何教程 GM2 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development + rendering-math-figures

**Goal:** 完成 Part 2 平面向量应用（4 章，含 2 ⭐ 模板 D）+ 5-8 张配图。

**前序里程碑:** GM0、GM1（已完成）

---

## 文件结构

```
gaozhong_math/geometry/part2-plane-vectors-applications/
├── 01-dot-product.md             # ⭐ 数量积 + 夹角 / 投影
├── 02-vector-in-triangle.md      # ⭐ 向量解三角形 + 三点共线 + 重心
├── 03-vector-in-line-circle.md   # 向量与直线 / 圆的关系
└── 04-vector-applications.md     # 物理 / 综合应用
```

---

## Task 1: part2/01 数量积 ⭐（模板 D，4000-5500 字）

Title: `# 平面向量的数量积`

**一例速记**：
> **数量积定义** $\vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}|\cos\theta$（$\theta$ 是夹角，$0 \leq \theta \leq \pi$）  
> **坐标公式**：$\vec{a} = (x_1, y_1), \vec{b} = (x_2, y_2)$ → $\vec{a} \cdot \vec{b} = x_1 x_2 + y_1 y_2$  
> **应用三大用途**：① 求夹角 $\cos\theta = \frac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|}$ ② 判断垂直 $\vec{a} \perp \vec{b} \Leftrightarrow \vec{a}\cdot\vec{b} = 0$ ③ 求投影 $\vec{b}$ 在 $\vec{a}$ 上的投影 = $\frac{\vec{a}\cdot\vec{b}}{|\vec{a}|}$

**引入题**：$\vec{a} = (3, 4), \vec{b} = (1, -2)$，求 $\vec{a} \cdot \vec{b}$、$|\vec{a}|, |\vec{b}|$、夹角余弦。

**思维路径还原** 15-20 行：直接代公式。

**抽象成方法**：5 大公式（点积定义、坐标、模、夹角、投影）+ 性质（交换、分配、$\vec{a}\cdot\vec{a} = |\vec{a}|^2$）

**思考路标 ≥8 条**：
- 看到"求夹角" → 用 $\cos\theta = \frac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|}$
- 看到"判垂直" → 验 $\vec{a}\cdot\vec{b} = 0$
- 看到"投影" → 用 $\frac{\vec{a}\cdot\vec{b}}{|\vec{a}|}$
- 模平方 = $\vec{a}\cdot\vec{a}$
- 含参的垂直条件解参
- 数量积可正可负可零

例题 3 例 + 自测 5 题。

**配图 2 张**：
- `geo-p2-01-1.tex`：夹角 $\theta$ + 数量积几何意义（$|\vec{a}| \cdot $ "$\vec{b}$ 在 $\vec{a}$ 上的投影"）
- `geo-p2-01-2.tex`：垂直情形 $\vec{a} \perp \vec{b}$

---

## Task 2: part2/02 向量解三角形 ⭐（模板 D，4000-5500 字）

Title: `# 向量在三角形中的应用`

**一例速记**：
> **三点共线**：$A, B, C$ 共线 ⇔ $\vec{OC} = \lambda\vec{OA} + (1-\lambda)\vec{OB}$（$\lambda \in \mathbb{R}$，$O$ 不在直线上）  
> **重心** $G = \frac{1}{3}(A + B + C)$，$\vec{OG} = \frac{1}{3}(\vec{OA} + \vec{OB} + \vec{OC})$  
> **中点** $M$ 在 $BC$ 上：$\vec{OM} = \frac{1}{2}(\vec{OB} + \vec{OC})$

**引入题**：在 $\triangle ABC$ 中，$D$ 是 $BC$ 中点，求 $\vec{AD}$ 用 $\vec{AB}, \vec{AC}$ 表示。

**思维路径还原** 15-20 行：$\vec{AD} = \vec{AB} + \vec{BD} = \vec{AB} + \frac{1}{2}\vec{BC} = \vec{AB} + \frac{1}{2}(\vec{AC} - \vec{AB}) = \frac{1}{2}(\vec{AB} + \vec{AC})$

**抽象成方法**：① 用已知向量表示未知向量（基底法）② 中点公式 ③ 重心 ④ 三点共线判定

**思考路标 ≥8 条**：
- 看到三角形 + 中点 → 中点公式
- 看到重心 → $\frac{1}{3}(\vec{OA}+\vec{OB}+\vec{OC})$
- 看到三点共线 → $\vec{OC} = \lambda\vec{OA} + (1-\lambda)\vec{OB}$
- 看到"分点比" $AP : PB = m : n$ → $\vec{OP} = \frac{n}{m+n}\vec{OA} + \frac{m}{m+n}\vec{OB}$
- 用向量证几何定理：先选基底

例题 3 例（中位线 / 重心 / 含参分点）+ 自测 5 题。

**配图 2 张**：
- `geo-p2-02-1.tex`：三角形 + 中线 + 中点
- `geo-p2-02-2.tex`：三角形重心 $G$（3 条中线交点）

---

## Task 3: part2/03 向量与直线圆（模板 C，2500-3500 字）

Title: `# 向量与直线、圆的关系`

涵盖：
- 直线的向量方程：过定点 $P_0$、方向向量 $\vec{d}$ 的直线 $P = P_0 + t\vec{d}$
- 圆的向量方程：$|\vec{OP} - \vec{OC}| = r$（$C$ 圆心、$r$ 半径）
- 向量法判直线垂直 / 平行
- 向量法判直线与圆相切 / 相交
- 典型应用 3 例
- 易错点
- 自测 5 题

**配图 1 张**：
- `geo-p2-03-1.tex`：直线的向量方程示意（过 $P_0$ 沿方向 $\vec{d}$ 的直线 + 动点 $P$）

---

## Task 4: part2/04 向量应用综合（模板 C，2500-3500 字）

Title: `# 向量的物理与综合应用`

涵盖：
- 物理应用：力（合力 / 分力）、速度合成
- 综合应用：用向量证几何不等式、解几何最值
- 与代数交叉：向量 + 函数、向量 + 三角
- 典型应用 3 例
- 自测 4 题

无配图（应用题为主）。

---

## Task 5: GM2 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|待补' gaozhong_math/geometry/part2-plane-vectors-applications 2>/dev/null
ls gaozhong_math/geometry/part2-plane-vectors-applications/
ls gaozhong_math/geometry/figures/svg/ | grep '^geo-p2-'
printf '\n---\n**GM2 完成于：2026-05-13**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-geometry-GM2.md
git add docs/superpowers/plans/2026-05-13-gaozhong-geometry-GM2.md
git commit -m "docs(gaozhong/geometry): mark GM2 milestone complete"
git push origin master
```

---

## 调度

**2 个 sonnet subagent 并行：**
- A: 01 ⭐ 数量积 + 02 ⭐ 三角形（+ 4 张图）
- B: 03 直线圆 + 04 应用（+ 1 张图）

每章独立 commit。
