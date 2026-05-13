# 高中几何教程

**人教 A 版 2019 新课标** | 平面向量 / 解析几何 / 立体几何 / 空间向量

一份强调**几何思维四把剑切换**的高中几何完整教程，覆盖人教 A 版必修第二册（平面向量、立体几何初步）与选择性必修第一册（直线与方程、圆与方程、圆锥曲线、空间向量与立体几何）的全部几何内容。不仅讲"怎么算"，更讲"看到什么图、什么条件，应该立刻切到哪一种方法"。核心主张：**综合 / 解析 / 向量 / 三角四种方法不是孤立工具，而是同一道题的四把切入剑，看懂条件就能选对剑**。

**配套姊妹篇**：
- [`gaozhong_math/algebra/`](../algebra/)：高中代数教程（与本书平行，函数 / 导数 / 三角 / 数列 / 概率统计）
- [`chuzhong_geometry/`](../../chuzhong_geometry/)：中考几何教程（本书的直接前序，综合法与基础模型；若初中几何尚未扎实，建议先读它）

---

## 这本书写给谁

- **高一下新生**：刚学完集合、函数，第一次接触平面向量与立体几何初步，发现"原来初中的纯综合法不够用了，还要学向量这种新工具"的同学
- **备战高考的高二高三学生**：已学过课本，但遇到圆锥曲线压轴就掉进运算泥潭、遇到二面角不知用综合法还是建系、遇到含参直线弦长就不知设而不求该怎么"不求"的考生
- **自学高中几何的成年读者**：想补全高中几何基础，或工作后希望用向量、坐标这种代数化的视角重新理解几何的学习者

---

## 本书的特色

普通教辅按"定义 → 公式 → 例题 → 习题"罗列。本书额外强调：

1. **高中几何思维四把剑——综合 / 解析 / 向量 / 三角，何时选哪一把**：高中几何与初中最大的差别，是同一道题往往有四种攻法。看到"等腰三角形顶角角平分线"应该想综合法（角平分线性质）；看到"焦点弦弦长"应该想解析法（设而不求 + 韦达）；看到"二面角"应该想向量法（建系 + 法向量）；看到"$\vec{AB} \cdot \vec{AC}$"应该想三角法（余弦定理）。本书把"看到 X 立刻想用第 Y 把剑"这件事系统化、可复用化
2. **思维工具箱 12 篇，强调几何独有思维**：toolkit 不与代数重复——01 四把剑切换、02 辅助线思维、04 参数化、07 几何最值五法、08 向量两副面孔（代数面 / 几何面）、09 空间想象力、10 直线圆锥综合、11 立体向量法。这些是高中几何独有的方法论，不在代数教程里
3. **模板 D"思维路径还原"**：每个带 ⭐ 的章节，用引用块完整展示作者解题瞬间的内心独白——你读完就知道"他第一秒看到了什么图象特征、为什么直接选了向量法而不是综合法、下一步往哪条线索走"。这件事在普通教辅里通常被省略，但它恰恰是高分考生与中等考生最大的差别
4. **套路图集（40-50 个高频模型）**：附录 B 把高考几何最常见的模型（如焦点三角形、设而不求、定点定值、垂直法向量、三余弦定理、阿基米德三角形等）整理为可对照查阅的图集，每个模型配 ASCII 示意 + 触发条件 + 标准套路
5. **几何配图 100+ 张（TikZ + Asymptote）**：核心章节随章配图，圆锥曲线 / 立体 / 空间向量章节每章 3-5 张；平面图用 TikZ + pgfplots，立体图与空间向量用 Asymptote

---

## 年级标签说明

**基准教材：人教 A 版 2019 新课标**（全国多省通用）。

`[高一下]` 对应必修第二册（平面向量基础与应用、立体几何初步）；`[高二上]` 对应选择性必修第一册（直线、圆、圆锥曲线、空间向量）；`[高三复习]` 是高考综合题型，散见于各年级、高三集中训练。同一 part 内的章节可能跨学期。

> 若你的学校进度与标准课程表有差异（例如部分省份把空间向量提到必修），以学校当年教学计划为准。

---

## 完整目录

### 方法论（建议先读）

- [序言：什么是"高中几何思维"](00-preface.md) `[通用]`
- [思维工具箱](thinking-toolkit/) `[通用 / 高三复习集中读]`
  - [01. 几何思维四把剑（综合 / 解析 / 向量 / 三角）](thinking-toolkit/01-four-swords.md) ⭐
  - [02. 辅助线思维](thinking-toolkit/02-auxiliary-lines.md) ⭐
  - [03. 数形结合（几何深化版）](thinking-toolkit/03-number-shape-deep-v2.md)
  - [04. 参数化思想](thinking-toolkit/04-parametrization.md) ⭐
  - [05. 不变量与对称](thinking-toolkit/05-invariants-and-symmetry.md)
  - [06. 位置关系的分类讨论](thinking-toolkit/06-position-classification.md)
  - [07. 几何最值的 5 种方法](thinking-toolkit/07-geometric-extrema.md) ⭐
  - [08. 向量的两副面孔（代数面 / 几何面）](thinking-toolkit/08-vector-two-faces.md) ⭐
  - [09. 空间几何想象力](thinking-toolkit/09-spatial-imagination.md)
  - [10. 直线与圆锥曲线综合](thinking-toolkit/10-line-conic-synthesis.md) ⭐
  - [11. 立体几何向量法](thinking-toolkit/11-vector-in-solid.md) ⭐
  - [12. 几何开题清单](thinking-toolkit/12-checklist-before-solving.md)

### 知识章节

- **Part 1：平面向量基础** `[高一下]`
  - [01. 向量的概念（相等、共线、零向量）](part1-plane-vectors-basics/01-vector-concept.md)
  - [02. 向量加减法（三角形法 / 平行四边形法）](part1-plane-vectors-basics/02-vector-addition.md)
  - [03. 数乘运算与共线定理](part1-plane-vectors-basics/03-scalar-multiplication.md)
  - [04. 基底、平面向量基本定理、坐标表示](part1-plane-vectors-basics/04-basis-and-coordinates.md) ⭐

- **Part 2：平面向量应用** `[高一下]`
  - [01. 数量积、夹角、投影](part2-plane-vectors-applications/01-dot-product.md) ⭐
  - [02. 向量解三角形、三点共线、重心](part2-plane-vectors-applications/02-vector-in-triangle.md) ⭐
  - [03. 向量与直线、圆的关系](part2-plane-vectors-applications/03-vector-in-line-circle.md)
  - [04. 向量的物理与综合应用](part2-plane-vectors-applications/04-vector-applications.md)

- **Part 3：直线与方程** `[高二上]`
  - [01. 斜率与倾斜角](part3-line-and-equation/01-slope-and-inclination.md)
  - [02. 直线方程的 5 种形式](part3-line-and-equation/02-line-equation-forms.md)
  - [03. 平行、垂直、重合的位置关系](part3-line-and-equation/03-line-position-relations.md) ⭐
  - [04. 点到直线距离、两平行线距离](part3-line-and-equation/04-distance-formulas.md) ⭐
  - [05. 含参直线与综合](part3-line-and-equation/05-line-applications.md)

- **Part 4：圆与方程** `[高二上]`
  - [01. 圆的标准方程与一般方程](part4-circle-and-equation/01-circle-standard-equation.md)
  - [02. 直线与圆的位置关系、弦长、切线](part4-circle-and-equation/02-line-circle-position.md) ⭐
  - [03. 圆与圆的位置关系](part4-circle-and-equation/03-circle-circle-position.md)
  - [04. 含参圆与综合](part4-circle-and-equation/04-circle-applications.md)

- **Part 5：椭圆** `[高二上]`
  - [01. 椭圆的定义与标准方程](part5-ellipse/01-ellipse-definition-equation.md)
  - [02. 椭圆的几何性质（含离心率）](part5-ellipse/02-ellipse-properties.md) ⭐
  - [03. 直线与椭圆：弦长、中点弦](part5-ellipse/03-line-ellipse.md) ⭐
  - [04. 椭圆含参与综合应用](part5-ellipse/04-ellipse-applications.md)

- **Part 6：双曲线** `[高二上]`
  - [01. 双曲线的定义与标准方程](part6-hyperbola/01-hyperbola-definition-equation.md)
  - [02. 双曲线的几何性质（含渐近线与离心率）](part6-hyperbola/02-hyperbola-properties.md) ⭐
  - [03. 直线与双曲线（含 $\Delta$ 与单交点情形）](part6-hyperbola/03-line-hyperbola.md) ⭐
  - [04. 双曲线综合](part6-hyperbola/04-hyperbola-applications.md)

- **Part 7：抛物线** `[高二上]`
  - [01. 抛物线的定义与标准方程](part7-parabola/01-parabola-definition-equation.md)
  - [02. 抛物线的性质、焦点弦、准线](part7-parabola/02-parabola-properties.md) ⭐
  - [03. 直线与抛物线、焦点弦性质](part7-parabola/03-line-parabola.md) ⭐
  - [04. 抛物线综合](part7-parabola/04-parabola-applications.md)

- **Part 8：立体几何初步** `[高一下]`
  - [01. 点、线、面与三视图](part8-solid-geometry-basics/01-points-lines-planes.md)
  - [02. 异面直线及其夹角](part8-solid-geometry-basics/02-spatial-line-position.md) ⭐
  - [03. 直线与平面的位置关系（平行 / 垂直 / 相交）](part8-solid-geometry-basics/03-line-plane-position.md) ⭐
  - [04. 平面与平面的位置关系、二面角](part8-solid-geometry-basics/04-plane-plane-position.md) ⭐
  - [05. 棱柱、棱锥、球的表面积与体积](part8-solid-geometry-basics/05-solid-figures-area-volume.md)
  - [06. 立体几何综合应用](part8-solid-geometry-basics/06-solid-geometry-applications.md)

- **Part 9：空间向量与立体几何** `[高二上]`
  - [01. 空间向量的概念与加减、数乘](part9-spatial-vectors/01-spatial-vector-basics.md)
  - [02. 空间向量的数量积](part9-spatial-vectors/02-spatial-vector-dot-product.md)
  - [03. 空间基底与坐标系建立](part9-spatial-vectors/03-spatial-basis-coordinates.md) ⭐
  - [04. 用向量证明平行与垂直](part9-spatial-vectors/04-spatial-line-plane-by-vector.md) ⭐
  - [05. 用向量求线线 / 线面 / 二面角](part9-spatial-vectors/05-spatial-angles-by-vector.md) ⭐
  - [06. 用向量求距离](part9-spatial-vectors/06-spatial-distance-by-vector.md) ⭐

- **Part 10：高考几何综合** `[高三复习]`
  - [01. 圆锥曲线综合（直线 + 曲线 + 韦达）](part10-geometry-comprehensive/01-conic-comprehensive.md) ⭐
  - [02. 含参圆锥曲线](part10-geometry-comprehensive/02-conic-with-parameter.md) ⭐
  - [03. 轨迹方程问题](part10-geometry-comprehensive/03-trajectory-and-locus.md) ⭐
  - [04. 定点 / 定值 / 定直线问题](part10-geometry-comprehensive/04-fixed-point-and-line.md) ⭐
  - [05. 立体几何综合（向量 + 综合）](part10-geometry-comprehensive/05-solid-comprehensive.md) ⭐
  - [06. 向量与代数 / 几何交叉](part10-geometry-comprehensive/06-vector-comprehensive.md) ⭐

### 附录

- [A. 公式速查表](appendix/A-formula-sheet.md)
- [B. 解题套路图集（约 40-50 模型）](appendix/B-model-atlas.md)
- [C. 基础题库（80 题 ☆）](appendix/C-exercises-basic.md)
- [D. 中档题库（100 题 ☆☆）](appendix/D-exercises-medium.md)
- [E. 高考压轴题库（60 题 ☆☆☆）](appendix/E-exercises-advanced.md)
- [F1. 详解：平面向量 / 空间向量（Part 1, 2, 9）](appendix/F1-solutions-vectors.md)
- [F2. 详解：直线与圆（Part 3, 4）](appendix/F2-solutions-line-circle.md)
- [F3. 详解：圆锥曲线（Part 5, 6, 7）](appendix/F3-solutions-conics.md)
- [F4. 详解：立体几何 / 综合（Part 8, 10）](appendix/F4-solutions-solid-comprehensive.md)

---

## 怎么用这本书

**高一下新生（向量 + 立体几何初步起步路线）**：

先通读思维工具箱前三篇核心——01 四把剑 / 02 辅助线 / 08 向量两面孔，建立"高中几何不再是纯综合法"的方法论框架；然后按 Part 1（平面向量基础）→ Part 2（平面向量应用）→ Part 8（立体几何初步）顺序推进。Part 8 是高一下的难点（空间想象力），建议每章配合 figures 中的 SVG 立体图反复看。每遇到带 ⭐ 的章节，先花 10-15 分钟自己尝试引入题，再读"思维路径还原"。

**高二上学生（解析几何 + 空间向量主干路线）**：

补齐思维工具箱剩余篇——04 参数化、07 几何最值、10 直线圆锥、11 立体向量；再按 Part 3（直线）→ Part 4（圆）→ Part 5 (椭圆) → Part 6（双曲线）→ Part 7（抛物线）→ Part 9（空间向量）顺序推进。Part 5-7 圆锥曲线是高考压轴的核心区域，建议每章与附录 B 套路图集（焦点弦、设而不求、中点弦等模型）配合使用——学完一种方法就去附录 B 做"看图识套路"训练。

**高三复习（高考冲刺路线）**：

直接跳 Part 10（6 章全是高考几何压轴套路）+ 思维工具箱 12 篇完整通读 + 附录 E（60 道压轴题）。Part 10 每章覆盖一类大题套路（综合、含参、轨迹、定点定值、立体、交叉），与附录 E 的题目互为印证。遇到不熟悉的知识点再回查对应 Part。

---

## 章节内部结构

本书章节分两种模板：

**模板 C（概念/运算法则类）**，如向量加法、圆的方程、二面角定义、棱锥体积公式：
> 一、概念特征 → 二、定义与定理 → 三、推导证明 → 四、典型应用（带【思路】）→ 五、易错点 → 六、思路自测题（字数 2500-3500）

**模板 D（套路/题型类）**，即所有带 ⭐ 标记的章节，字数 4000-5500：
> **一例速记**（极简样板题）→ 一、引入 → 二、**思维路径还原**（15-20 行高手内心独白，`>` 引用块）→ 三、抽象成方法 → 四、方法变形 → 五、**思考路标**（≥8 条"看到 X → 立刻想 Y"）→ 六、应用例题（含至少 1 道含参讨论或严格证明或位置分类）→ 七、思路自测题

模板 D 的**思维路径还原**和**思考路标**是本书的核心特色——把高手解题时"看图的前几秒选哪把剑"的内部过程系统化、可复用化。

---

## 关于练习题

- **节内例题**：完整解答，附【思路】小节，说明方法选择（综合 / 解析 / 向量 / 三角）的理由
- **节末思路自测题**：仅给提示，不给完整解答，逼你自己走一遍思维路径
- **附录题库 240 题**：基础 80 题（☆）/ 中档 100 题（☆☆）/ 高考压轴 60 题（☆☆☆）；每题详解标注方法 / 模型标签（如 `[设而不求 + 韦达]`、`[向量法 + 法向量求二面角]`、`[焦点弦定理]`），便于回查对应章节

与代数教程 240 题对齐，几何题目按主题分配到 F1-F4 四个详解文件中：F1 向量、F2 直线圆、F3 圆锥曲线、F4 立体几何与综合。

---

## 关于图

本书图分两类，几何核心章节配图密集：

- **TikZ 源码**（存于 `figures/src/tikz/`）：平面向量图、直线与圆、圆锥曲线（用 `pgfplots` 绘制椭圆 / 双曲线 / 抛物线）、二维几何变换图
- **Asymptote 源码**（存于 `figures/src/asy/`）：立体几何图（棱柱、棱锥、球的三维图）、空间向量图、空间坐标系图

两类源文件均通过 [`figures/render.sh`](figures/render.sh) 编译为 SVG（xelatex 支持中文）。图形风格规范见 [`figures/STYLE.md`](figures/STYLE.md)。核心可视化章节（向量加减、圆锥曲线性质、立体图、空间向量建系）随章配图 3-5 张；纯概念章节图较少。

---

## 反馈

本教程是开源项目，欢迎指出错误或提出改进建议。如发现推导跳步、套路描述不准确、或有比书中更优雅的方法（特别是"四把剑切换"中某把剑更合适的情形），请提 issue 或 PR。
