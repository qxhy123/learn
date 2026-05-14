# 高中几何教程 GM5 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development + rendering-math-figures

**Goal:** 完成 Part 8 立体几何初步（6 章，含 3 ⭐ 模板 D）+ 10-15 张配图（含立体图）。

**前序里程碑:** GM0-GM4（种子已完成 5 个里程碑，本里程碑是种子最后一个）

---

## 文件结构

```
gaozhong_math/geometry/part8-solid-geometry-basics/
├── 01-points-lines-planes.md          # 点线面公理 + 三视图
├── 02-spatial-line-position.md        # ⭐ 异面直线 + 夹角
├── 03-line-plane-position.md          # ⭐ 直线与平面位置
├── 04-plane-plane-position.md         # ⭐ 二面角 + 平面与平面
├── 05-solid-figures-area-volume.md    # 表面积 + 体积
└── 06-solid-geometry-applications.md  # 综合应用
```

立体图建议用 **Asymptote**（asy），可以渲染 3D；正方体 / 四面体 / 长方体等首选。

---

## Task 1: part8/01 点线面公理（模板 C，2500-3500 字）

Title: `# 点、线、面的基本关系与三视图`

涵盖：
- **三大基本公理**：
  - 公理 1：两点确定一线（线 $\subset$ 平面）
  - 公理 2：不在一直线上的三点确定一平面
  - 公理 3：两平面有公共点，公共点构成的集合是一条直线
- 三条**推论**：
  - 一条直线和这直线外一点确定一个平面
  - 两条相交直线确定一个平面
  - 两条平行直线确定一个平面
- **三视图**：正视图 / 侧视图 / 俯视图（三投影规律：长对正、高平齐、宽相等）
- 典型应用 3 例
- 易错点 5 条
- 自测题 4 题

**配图 3 张**：
- `geo-p8-01-1.tex`：3 大公理几何示意（三个 subplot）
- `geo-p8-01-2.asy`：长方体的三视图（正视 / 侧视 / 俯视 + 立体图）
- `geo-p8-01-3.asy`：正四面体（4 个等边三角形面）立体图

---

## Task 2: part8/02 异面直线 ⭐（模板 D，4000-5500 字）

Title: `# 异面直线及其夹角`

**一例速记**：
> **异面直线**：不同在任何一个平面内的两条直线（既不平行也不相交）  
> **异面直线夹角**：作平移使两直线交于一点，所成的锐角（或直角）即为异面直线夹角，$\theta \in (0, \pi/2]$  
> **求法**：① 综合法（平移找平行线）② 向量法（建系 + 用 $\cos\theta = |\cos\langle\vec{a}, \vec{b}\rangle|$）

**引入题**：正方体 $ABCD$-$A_1B_1C_1D_1$ 中，求异面直线 $AB_1$ 与 $BC_1$ 所成角。

**思维路径还原** 15-20 行（用 `>` 引用块）：
- 综合法：平移 $BC_1$ 到 $AD_1$（因 $BC_1 \parallel AD_1$）→ $AB_1$ 与 $AD_1$ 同起点，所成角为 $\triangle AB_1 D_1$ 的角
- 计算 $|AB_1| = |AD_1| = \sqrt{2}$，$|B_1 D_1| = \sqrt{2}$ → 等边三角形 → 角 = $\pi/3$
- 或用向量法：建系，$A = (0,0,0), B_1 = (1,0,1), C_1 = (1,1,1), B = (1,0,0)$，$\vec{AB_1} = (1,0,1)$，$\vec{BC_1} = (0,1,1)$ → $\cos\theta = |1|/2 = 1/2$ → $\theta = \pi/3$

**抽象成方法**：综合法（平移） + 向量法（建系）

**方法变形**：
- 异面直线夹角范围 $(0, \pi/2]$（不能 $> \pi/2$）
- 平面 vs 立体的"角"概念差异

**思考路标 ≥8 条**

例题 3 例 + 自测 5 题。

**配图 2 张**：
- `geo-p8-02-1.asy`：正方体中异面直线 $AB_1$ 与 $BC_1$
- `geo-p8-02-2.asy`：平移找平行线（$BC_1 \parallel AD_1$，所成角化为 $\triangle AB_1D_1$）

---

## Task 3: part8/03 直线与平面位置 ⭐（模板 D，4000-5500 字）

Title: `# 直线与平面的位置关系`

**一例速记**：
> **3 种位置**：① 直线在平面内 $l \subset \alpha$ ② 直线与平面平行 $l \parallel \alpha$ ③ 直线与平面相交 $l \cap \alpha = P$  
> **线面平行判定**：平面外一直线 $\parallel$ 平面内一直线 → 该线与平面平行  
> **线面垂直判定**：直线 $\perp$ 平面内两条相交直线 → 该线与平面垂直

**引入题**：在长方体 $ABCD$-$A_1B_1C_1D_1$ 中，证明 $AC_1 \perp BD$。

**思维路径还原** 15-20 行：先证 $AC_1$ 垂直于平面 $BDD_1B_1$（用线面垂直判定）→ 因 $BD \subset$ 该平面 → $AC_1 \perp BD$。

**抽象成方法**：3 种位置关系判别 + 平行 / 垂直的判定定理

**方法变形**：
- 三垂线定理（在平面内的射影垂直 → 原线垂直）
- 用向量法证明垂直
- 平行的"判定 + 性质"两组定理

**思考路标 ≥8 条**

例题 3 例（证平行 / 证垂直 / 综合）+ 自测 5 题。

**配图 2 张**：
- `geo-p8-03-1.asy`：3 种线面位置关系
- `geo-p8-03-2.asy`：线面垂直判定（直线 $\perp$ 平面内两相交直线）

---

## Task 4: part8/04 平面与平面位置 ⭐（模板 D，4000-5500 字）

Title: `# 平面与平面的位置关系及二面角`

**一例速记**：
> **2 种位置**：① 两平面平行 $\alpha \parallel \beta$ ② 两平面相交 $\alpha \cap \beta = l$  
> **面面平行判定**：$\alpha$ 内两相交直线分别 $\parallel \beta$ → $\alpha \parallel \beta$  
> **面面垂直判定**：$\alpha$ 内有一条直线 $\perp \beta$ → $\alpha \perp \beta$  
> **二面角**：$\alpha$-$l$-$\beta$（棱 $l$ + 两半平面），平面角范围 $[0, \pi]$

**引入题**：在正三棱锥 $P$-$ABC$ 中，$E$ 是 $BC$ 中点，证明 $PE \perp$ 平面 $ABC$，并求二面角 $P$-$BC$-$A$。

**思维路径还原** 15-20 行：先证 $PE$ 垂直三角形 $ABC$，用三垂线定理。求二面角：在棱 $BC$ 上取一点（如 $E$），从两半平面分别引 $\perp BC$ 的线，所成角即为二面角。

**抽象成方法**：2 种位置 + 二面角 3 种求法（综合 / 法向量 / 三垂线）

**思考路标 ≥8 条**

例题 3 例 + 自测 5 题。

**配图 3 张**：
- `geo-p8-04-1.asy`：2 种面面位置关系（平行 / 相交）
- `geo-p8-04-2.asy`：二面角 $\alpha$-$l$-$\beta$（棱 + 两半平面 + 平面角）
- `geo-p8-04-3.asy`：正三棱锥 $P$-$ABC$ 中 $PE \perp$ 平面 $ABC$（应用）

---

## Task 5: part8/05 表面积与体积（模板 C，2500-3500 字）

Title: `# 立体图形的表面积与体积`

涵盖：
- **棱柱**：底面积 × 高
- **棱锥**：$\frac{1}{3}$ × 底面积 × 高
- **球**：表面积 $4\pi r^2$，体积 $\frac{4}{3}\pi r^3$
- **圆柱 / 圆锥 / 圆台**：表面积公式 + 体积公式
- 球与正多面体内切 / 外接
- 典型应用 3 例（计算 / 含参 / 实际应用）
- 易错点 5 条
- 自测题 5 题

**配图 2 张**：
- `geo-p8-05-1.asy`：棱柱 + 棱锥 + 球 三类
- `geo-p8-05-2.asy`：圆柱 + 圆锥 + 圆台 三类

---

## Task 6: part8/06 立体几何综合（模板 C，2500-3500 字）

Title: `# 立体几何综合应用`

涵盖：
- 折叠 / 展开问题
- 截面问题（正方体 8 种截面）
- 立体几何中的最值（最短路径 / 最大角度）
- 用综合法证空间几何定理
- 典型应用 3 例
- 易错点
- 自测 4 题

**配图 1 张**：
- `geo-p8-06-1.asy`：正方体 8 种截面（典型代表 3-4 种）

---

## Task 7: GM5 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|待补' gaozhong_math/geometry/part8-solid-geometry-basics 2>/dev/null
ls gaozhong_math/geometry/part8-solid-geometry-basics/
ls gaozhong_math/geometry/figures/svg/ | grep '^geo-p8-'
printf '\n---\n**GM5 完成于：2026-05-13**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-geometry-GM5.md
git add docs/superpowers/plans/2026-05-13-gaozhong-geometry-GM5.md
git commit -m "docs(gaozhong/geometry): mark GM5 milestone complete (seed phase done)"
git push origin master
```

种子阶段完成后，整个"高中几何种子"宣告就绪：约 25 章 + 12 toolkit + ~40 SVG。

---

## 调度

**3 个 sonnet subagent 并行：**
- A: 01 点线面 + 02 ⭐ 异面直线（+ 5 张图）
- B: 03 ⭐ 线面 + 04 ⭐ 面面（+ 5 张图）
- C: 05 表面积体积 + 06 综合（+ 3 张图）

每章独立 commit。如 asy 渲染失败可降级用 TikZ。

---
**GM5 完成于：2026-05-14**
