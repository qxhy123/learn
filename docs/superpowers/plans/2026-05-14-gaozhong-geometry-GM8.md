# 高中几何教程 GM8 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development + rendering-math-figures

**Goal:** 完成 Part 6 双曲线（4 章）+ Part 7 抛物线（4 章），共 8 章（含 4 ⭐ 模板 D）+ 15-20 张配图。

**前序里程碑:** GM0-GM7

---

## 文件结构

```
gaozhong_math/geometry/
├── part6-hyperbola/
│   ├── 01-hyperbola-definition-equation.md    # 定义 + 标准方程
│   ├── 02-hyperbola-properties.md             # ⭐ 性质 + 渐近线 + 离心率
│   ├── 03-line-hyperbola.md                   # ⭐ 直线与双曲线
│   └── 04-hyperbola-applications.md
└── part7-parabola/
    ├── 01-parabola-definition-equation.md     # 定义 + 标准方程
    ├── 02-parabola-properties.md              # ⭐ 性质 + 焦点弦 + 准线
    ├── 03-line-parabola.md                    # ⭐ 直线与抛物线 + 焦点弦性质
    └── 04-parabola-applications.md
```

---

## Part 6: 双曲线

### Task 1: part6/01 双曲线定义与方程（模板 C，2500-3500 字）

涵盖：
- 双曲线定义：到两定点距离**差的绝对值**等于常数 $2a$（$0 < 2a < |F_1F_2| = 2c$）
- 标准方程 $\dfrac{x^2}{a^2}-\dfrac{y^2}{b^2}=1$（$a > 0, b > 0$）；焦点在 $y$ 轴 $\dfrac{y^2}{a^2}-\dfrac{x^2}{b^2}=1$
- 关系 $c^2 = a^2 + b^2$
- 推导
- 应用 + 易错（$c^2 = a^2 + b^2$ 不是 $-$；定义是绝对值）+ 自测 4 题

配图 2 张：geo-p6-01-1, geo-p6-01-2（双曲线 + 焦点 + 动点；两种方位）

### Task 2: part6/02 双曲线性质 ⭐（模板 D，4000-5500 字）

一例速记：
> 顶点 $(\pm a, 0)$，焦点 $(\pm c, 0)$  
> **渐近线**：$y = \pm\dfrac{b}{a}x$（双曲线的标志性特征）  
> **离心率** $e = c/a > 1$（双曲线 $e > 1$，区别椭圆 $e < 1$）  
> **焦半径**：$|PF_1| = |a + ex_0|, |PF_2| = |a - ex_0|$（绝对值！）  
> **等轴双曲线** $a = b$ → $e = \sqrt{2}$，渐近线 $y = \pm x$

引入题：求 $\dfrac{x^2}{4}-\dfrac{y^2}{9}=1$ 的渐近线、离心率、顶点、焦点。

思维路径还原 + 抽象方法 + 思考路标 ≥8 条 + 例题 3 + 自测 5。

配图 3 张：geo-p6-02-1（双曲线 + 顶点 + 焦点 + 渐近线）、geo-p6-02-2（不同 $e$ 双曲线对比）、geo-p6-02-3（等轴双曲线）

### Task 3: part6/03 直线与双曲线 ⭐（模板 D，4000-5500 字）

一例速记：
> 与椭圆 / 抛物线**不同**：直线与双曲线**单交点**有两种情形——
> ① $\Delta = 0$（切线）② 直线平行于渐近线（必单交点）  
> 弦长公式 + 中点弦点差法 $k = \dfrac{b^2 x_0}{a^2 y_0}$（与椭圆差一个负号）

引入题 + 思维路径还原 + 抽象方法（5 大套路）+ 思考路标 + 例题 3 + 自测 5。

配图 2 张：geo-p6-03-1（位置关系图，含"平行渐近线单交点"特殊情形）、geo-p6-03-2（焦点弦）

### Task 4: part6/04 双曲线综合（模板 C，2500-3500 字）

含参 / 与圆 / 与直线综合 / 实际应用。例题 3 + 易错 5 + 自测 4。无图。

---

## Part 7: 抛物线

### Task 5: part7/01 抛物线定义与方程（模板 C，2500-3500 字）

涵盖：
- 抛物线定义：到一定点（焦点 $F$）与一定直线（准线 $l$）距离**相等**的点集
- 4 种标准方程：$y^2 = 2px$（开口右）、$y^2 = -2px$（左）、$x^2 = 2py$（上）、$x^2 = -2py$（下）
- 焦点坐标 + 准线方程（4 种对应）
- 推导：用距离相等 + 化简
- 例题 3 + 易错 5（$p$ 与焦点距离关系 = $p/2$）+ 自测 4

配图 3 张：geo-p7-01-1（开口右抛物线 + 焦点 + 准线 + 动点）、geo-p7-01-2（4 种方位对比）、geo-p7-01-3（推导示意）

### Task 6: part7/02 抛物线性质 ⭐（模板 D，4000-5500 字）

一例速记：
> 抛物线 $y^2 = 2px$（$p > 0$）：顶点 $(0, 0)$，焦点 $(p/2, 0)$，准线 $x = -p/2$  
> **焦半径**：$|PF| = x_0 + p/2$（**到焦点的距离** = $x_0$ 加上 $p/2$）  
> **离心率** $e = 1$（这是抛物线的标志）  
> **通径** = $2p$（过焦点垂直对称轴的弦）  
> **焦点弦性质**：直线过焦点交抛物线于 $A, B$，$|AB| = x_1 + x_2 + p$；$|AF| \cdot |BF| = p^2$；$\dfrac{1}{|AF|} + \dfrac{1}{|BF|} = \dfrac{2}{p}$

引入题 + 思维路径还原 + 抽象方法 + 思考路标 ≥8 + 例题 3 + 自测 5。

配图 3 张：geo-p7-02-1（抛物线 + 焦点 + 准线 + 焦半径 = 准线距离）、geo-p7-02-2（焦点弦 + 三大性质）、geo-p7-02-3（通径）

### Task 7: part7/03 直线与抛物线 ⭐（模板 D，4000-5500 字）

一例速记：
> 位置：相离 / 相切 / 相交（用 $\Delta$）；特殊：直线平行对称轴时**必单交点**（与双曲线渐近线特殊情形类比）  
> 弦长公式 + 焦点弦性质

引入题 + 抽象方法（5 大套路 + 焦点弦特技）+ 思考路标 + 例题 3 + 自测 5。

配图 2 张：geo-p7-03-1（位置关系图）、geo-p7-03-2（焦点弦 + 韦达定理 + 调和关系）

### Task 8: part7/04 抛物线综合（模板 C，2500-3500 字）

含参 / 实际应用（抛物面天线 / 抛物线弹道 / 反射性质）/ 综合。例题 3 + 易错 5 + 自测 4。无图。

---

## Task 9: GM8 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|待补' gaozhong_math/geometry/part6-hyperbola gaozhong_math/geometry/part7-parabola 2>/dev/null
ls gaozhong_math/geometry/part6-hyperbola gaozhong_math/geometry/part7-parabola
ls gaozhong_math/geometry/figures/svg/ | grep -E '^geo-p[67]-'
printf '\n---\n**GM8 完成于：2026-05-14**\n' >> docs/superpowers/plans/2026-05-14-gaozhong-geometry-GM8.md
git add docs/superpowers/plans/2026-05-14-gaozhong-geometry-GM8.md
git commit -m "docs(gaozhong/geometry): mark GM8 milestone complete"
git push origin master
```

---

## 调度

**4 个 sonnet subagent 并行：**
- A: Part 6 01+02 双曲线定义+性质（+ 5 张图）
- B: Part 6 03+04 直线双曲线+应用（+ 2 张图）
- C: Part 7 01+02 抛物线定义+性质（+ 6 张图）
- D: Part 7 03+04 直线抛物线+应用（+ 2 张图）

每章独立 commit。

---
**GM8 完成于：2026-05-14**
