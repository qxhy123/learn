# 高中几何教程 GM9 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development + rendering-math-figures

**Goal:** 完成 Part 9 空间向量与立体几何（6 章，含 4 ⭐ 模板 D）+ 8-12 张配图（含立体图）。

**前序里程碑:** GM0-GM8

---

## 文件结构

```
gaozhong_math/geometry/part9-spatial-vectors/
├── 01-spatial-vector-basics.md            # 空间向量定义、加减、数乘
├── 02-spatial-vector-dot-product.md       # 空间向量数量积
├── 03-spatial-basis-coordinates.md        # ⭐ 空间基底、坐标系建立
├── 04-spatial-line-plane-by-vector.md     # ⭐ 用向量证平行、垂直
├── 05-spatial-angles-by-vector.md         # ⭐ 用向量求线线/线面/二面角
└── 06-spatial-distance-by-vector.md       # ⭐ 用向量求距离
```

---

## Task 1: part9/01 空间向量基础（模板 C，2500-3500 字）

涵盖：
- 空间向量定义、几何表示
- 加减法（平行四边形 / 三角形法则在空间中的推广）
- 数乘
- 共面向量定理：$\vec{p}, \vec{a}, \vec{b}$ 共面 ⇔ $\vec{p} = x\vec{a} + y\vec{b}$
- 空间向量基本定理（**4 个不共面向量必线性相关**）
- 例题 3 + 易错 5 + 自测 4

配图 2 张：geo-p9-01-1（空间向量加法平行六面体）、geo-p9-01-2（共面向量定理示意）

## Task 2: part9/02 空间向量数量积（模板 C，2500-3500 字）

涵盖：
- 定义 $\vec{a}\cdot\vec{b}=|\vec{a}||\vec{b}|\cos\theta$（与平面相同）
- 坐标公式 $\vec{a}\cdot\vec{b}=x_1x_2+y_1y_2+z_1z_2$（多一项）
- 性质：$\perp\Leftrightarrow$ 点积 = 0；$|\vec{a}|=\sqrt{x^2+y^2+z^2}$；夹角 $\cos\theta=\dfrac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|}$
- 投影
- 例题 3 + 易错 5 + 自测 4

配图 1 张：geo-p9-02-1（空间夹角示意）

## Task 3: part9/03 空间基底与坐标 ⭐（模板 D，4000-5500 字）

一例速记：
> **空间向量基本定理**：3 个不共面向量构成基底，任意空间向量唯一表示  
> **正交基底（建系）**：3 条相互垂直、共原点的轴 → $\vec{i}, \vec{j}, \vec{k}$ 单位向量 → 任意点 $(x, y, z)$  
> **建系 4 步**：① 找正交三边 ② 设单位长 ③ 求关键点坐标 ④ 用向量公式

引入题：在长方体中建系 + 写各顶点坐标。

思维路径还原 + 抽象方法 + 思考路标 ≥8 + 例题 3（含长方体 / 正方体 / 正三棱锥）+ 自测 5。

配图 2 张：geo-p9-03-1（长方体建系）、geo-p9-03-2（正三棱锥建系，含中心 / 顶点坐标）

## Task 4: part9/04 用向量证平行、垂直 ⭐（模板 D，4000-5500 字）

一例速记：
> **线线平行**：$\vec{a}\parallel\vec{b}$ ⇔ $\vec{a}=\lambda\vec{b}$  
> **线线垂直**：$\vec{a}\perp\vec{b}$ ⇔ $\vec{a}\cdot\vec{b}=0$  
> **线面平行**：$\vec{a}\perp\vec{n}$（$\vec{n}$ 是面法向量）且 $\vec{a}$ 不在面内  
> **线面垂直**：$\vec{a}\parallel\vec{n}$  
> **面面平行**：$\vec{n_1}\parallel\vec{n_2}$  
> **面面垂直**：$\vec{n_1}\perp\vec{n_2}$  
> **法向量求法**：$\vec{n}\cdot\vec{a}=0, \vec{n}\cdot\vec{b}=0$（$\vec{a}, \vec{b}$ 是面内两不共线向量）解出 $\vec{n}$

引入题：用向量法证长方体某线垂直某面。

思维路径还原 + 抽象方法 + 思考路标 ≥8 + 例题 3 + 自测 5。

配图 2 张：geo-p9-04-1（线面垂直向量判定）、geo-p9-04-2（面面平行向量判定）

## Task 5: part9/05 向量法求角 ⭐（模板 D，4000-5500 字）

一例速记：
> **线线角**：$\cos\theta=\left|\dfrac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|}\right|$ —— **取绝对值**（异面线角 $\in (0, \pi/2]$）  
> **线面角**：$\sin\theta=\left|\dfrac{\vec{l}\cdot\vec{n}}{|\vec{l}||\vec{n}|}\right|$（$\vec{n}$ 是面法向量；$\theta\in [0, \pi/2]$，取绝对值）  
> **二面角**：$\cos\theta=\pm\dfrac{\vec{n_1}\cdot\vec{n_2}}{|\vec{n_1}||\vec{n_2}|}$（看图判正负，**不**总是取绝对值）

引入题：正方体二面角的法向量求法（标准建系流程）。

思维路径还原 + 抽象方法（3 类角的完整公式表）+ 思考路标 + 例题 3（线线 / 线面 / 二面）+ 自测 5。

配图 2 张：geo-p9-05-1（3 类角的几何意义对比）、geo-p9-05-2（二面角法向量法 + 看图判正负）

## Task 6: part9/06 向量法求距离 ⭐（模板 D，4000-5500 字）

一例速记：
> **点到平面距离**：$d=\dfrac{|\vec{AP}\cdot\vec{n}|}{|\vec{n}|}$（$A$ 是面内任意点）  
> **点到直线距离**：$d=\dfrac{|\vec{AP}\times\vec{l}|}{|\vec{l}|}$（用叉积；或 $d=\sqrt{|\vec{AP}|^2-\left(\dfrac{\vec{AP}\cdot\vec{l}}{|\vec{l}|}\right)^2}$ 用投影）  
> **异面直线距离**：$d=\dfrac{|\vec{AB}\cdot\vec{n}|}{|\vec{n}|}$（$\vec{n}$ 是公共法向量 $\perp\vec{a}, \vec{b}$；$A, B$ 分别在两条线上）

引入题：求长方体中某点到某面距离。

思维路径还原 + 抽象方法 + 思考路标 ≥8 + 例题 3 + 自测 5。

配图 1 张：geo-p9-06-1（3 类距离对比 + 公式）

---

## Task 7: GM9 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|待补' gaozhong_math/geometry/part9-spatial-vectors 2>/dev/null
ls gaozhong_math/geometry/part9-spatial-vectors/
ls gaozhong_math/geometry/figures/svg/ | grep '^geo-p9-'
printf '\n---\n**GM9 完成于：2026-05-14**\n' >> docs/superpowers/plans/2026-05-14-gaozhong-geometry-GM9.md
git add docs/superpowers/plans/2026-05-14-gaozhong-geometry-GM9.md
git commit -m "docs(gaozhong/geometry): mark GM9 milestone complete"
git push origin master
```

---

## 调度

**3 个 sonnet subagent 并行：**
- A: 01 基础 + 02 数量积（+ 3 张图）
- B: 03 ⭐ 基底坐标 + 04 ⭐ 平行垂直（+ 4 张图）
- C: 05 ⭐ 角 + 06 ⭐ 距离（+ 3 张图）

每章独立 commit。
