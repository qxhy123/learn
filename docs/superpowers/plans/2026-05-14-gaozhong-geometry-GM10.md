# 高中几何教程 GM10 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development + rendering-math-figures

**Goal:** 完成 Part 10 高考几何综合（6 章，全部 ⭐ 模板 D）+ 8-12 张配图。

**前序里程碑:** GM0-GM9

---

## 文件结构

```
gaozhong_math/geometry/part10-geometry-comprehensive/
├── 01-conic-comprehensive.md       # ⭐ 圆锥曲线综合（直线 + 曲线 + 韦达）
├── 02-conic-with-parameter.md      # ⭐ 含参圆锥曲线
├── 03-trajectory-and-locus.md      # ⭐ 轨迹方程问题
├── 04-fixed-point-and-line.md      # ⭐ 定点 / 定值 / 定直线问题
├── 05-solid-comprehensive.md       # ⭐ 立体几何综合
└── 06-vector-comprehensive.md      # ⭐ 向量与代数 / 几何交叉
```

每章 4000-5500 字。配图 1-2 张。

---

## Task 1: part10/01 圆锥曲线综合 ⭐

**核心**：直线与椭圆 / 双曲线 / 抛物线的综合题，韦达定理 + 设而不求。

一例速记 + 引入题（高考真题级别）+ 思维路径还原 + 5 大套路 + 思考路标 ≥8 + 例题 3 + 自测 4。

配图 2 张：geo-p10-01-1（椭圆 + 直线 + 韦达定理示意）、geo-p10-01-2（抛物线焦点弦综合）

## Task 2: part10/02 含参圆锥曲线 ⭐

**核心**：含参方程的分类讨论。

含参椭圆 / 含参双曲线 / 含参抛物线的位置 / 弦长 / 含参讨论。

配图 1 张：geo-p10-02-1（含参直线扫描圆锥曲线）

## Task 3: part10/03 轨迹方程问题 ⭐

**核心**：动点轨迹求方程。

5 大方法：① 直接法（代入条件）② 定义法（识别圆锥曲线定义）③ 几何法（用几何性质）④ 参数法（设参数消元）⑤ 相关点法（动点依赖另一动点）

配图 2 张：geo-p10-03-1（轨迹方法对比）、geo-p10-03-2（相关点法示例）

## Task 4: part10/04 定点定值定直线 ⭐

**核心**：高考压轴最常考的"3 定"问题。

定点问题（恒过某点）/ 定值问题（某量恒定）/ 定直线问题（弦中点在某直线上）

配图 1 张：geo-p10-04-1（定点 / 定值 / 定直线对比）

## Task 5: part10/05 立体几何综合 ⭐

**核心**：综合法 + 向量法混合求解立体题。

含参立体几何 / 折叠后立体几何 / 探索性问题（存在性）

配图 2 张：geo-p10-05-1（综合 vs 向量法对比）、geo-p10-05-2（折叠立体示例）

## Task 6: part10/06 向量综合 ⭐

**核心**：向量与解析 / 三角 / 函数交叉。

向量 + 三角函数 / 向量 + 圆锥曲线 / 向量 + 不等式综合

配图 1 张：geo-p10-06-1（向量综合示意）

---

## Task 7: GM10 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|待补' gaozhong_math/geometry/part10-geometry-comprehensive 2>/dev/null
ls gaozhong_math/geometry/part10-geometry-comprehensive/
ls gaozhong_math/geometry/figures/svg/ | grep '^geo-p10-'
printf '\n---\n**GM10 完成于：2026-05-14**\n' >> docs/superpowers/plans/2026-05-14-gaozhong-geometry-GM10.md
git add docs/superpowers/plans/2026-05-14-gaozhong-geometry-GM10.md
git commit -m "docs(gaozhong/geometry): mark GM10 milestone complete"
git push origin master
```

---

## 调度

**2 个 sonnet subagent 并行：**
- A: 01 + 02 + 03（圆锥曲线综合 3 章 + 5 张图）
- B: 04 + 05 + 06（定点 + 立体 + 向量 + 4 张图）

每章独立 commit。

---
**GM10 完成于：2026-05-14**
