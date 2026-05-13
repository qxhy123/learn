# 高中代数教程——人教 A 版 2019 新课标

一份强调**高级思维培养**的高中代数教程，覆盖人教 A 版必修一、必修二、选择性必修（一二三）的全部代数内容。不仅讲"怎么算"，更讲"为什么这样变形、看到 X 应该立刻想 Y"。核心主张：**结构识别先于计算，含参讨论严格分类，构造法与数形结合是高级武器**。

**配套姊妹篇**：[`chuzhong_algebra/`](../../chuzhong_algebra/) 中考代数教程（本书的直接前序，建议先完成初中部分再读本书）；未来的 `gaozhong_math/geometry/`（向量、解析几何、立体几何，与本书平行）。

---

## 这本书写给谁

- **高一新生**：刚从初中升入高中，感觉代数概念突然变抽象（函数、集合、不等式一下子全来了），需要把初中"做题直觉"升级为"严格推理能力"的同学
- **备战高考的高二高三学生**：已学过课本，但遇到含参讨论、导数压轴、抽象函数就卡壳，想把高考代数的所有常考套路系统梳理一遍的考生
- **自学高中代数的成年读者**：想补全高中数学基础，或工作后希望用更严格的数学思维重新审视代数的学习者

---

## 与普通教辅的不同

普通教辅：公式 → 例题 → 习题，以覆盖知识点为目标。本书额外强调：

1. **高级思维显式化**：把构造法、抽象函数、含参分类讨论、数形结合深化这些高考压轴必备的"高手思维"直接摊开写出来，而不是留给学生"悟"
2. **思维路径还原**（模板 D 核心）：每个带 ⭐ 的章节，用引用块完整展示高手解题瞬间的内心独白——你读完就知道"他认出了什么结构、为什么选这种方法"
3. **思维工具箱 12 篇**：特别强调 03 构造法、06 抽象函数、04 数形结合深化、11 分类讨论框架这四篇——它们是高考代数压轴题的底层操作系统
4. **套路反射弧**：每个典型套路给出"看到 X → 立即想 Y"的格式，让"灵感"变成可复用的条件反射

---

## 年级标签说明

**基准教材：人教 A 版 2019 新课标**（全国多省通用）。

`[高一上]` `[高一下]` `[高二上]` `[高二下]` `[选必三]` `[高三复习]` 以该内容在课标中首次系统学习的学期为准；同一 part 内的章节可能跨学期。`[选必三]` 对应选择性必修第三册（概率统计深化）。

> 若你的学校进度与标准课程表有差异，以学校当年教学计划为准。

---

## 目录

### 方法论（建议先读）

- [序言：什么是"高中代数思维"](00-preface.md) `[通用]`
- [思维工具箱](thinking-toolkit/) `[通用 / 高三复习集中读]`
  - [01. 高级结构识别](thinking-toolkit/01-structure-recognition.md)
  - [02. 换元与化简策略](thinking-toolkit/02-substitution-and-simplification.md)
  - [03. 构造法](thinking-toolkit/03-construction.md) ⭐
  - [04. 数形结合（高中深化版）](thinking-toolkit/04-number-shape-deep.md) ⭐
  - [05. 含参讨论与参数分离](thinking-toolkit/05-parameter-strategy.md) ⭐
  - [06. 抽象函数推理](thinking-toolkit/06-abstract-function.md) ⭐
  - [07. 复合函数与函数嵌套](thinking-toolkit/07-composite-function.md)
  - [08. 单调性与极值思想](thinking-toolkit/08-monotonicity-extrema.md)
  - [09. 对称性与周期性](thinking-toolkit/09-symmetry-and-periodicity.md)
  - [10. 极限与放缩思想](thinking-toolkit/10-limit-and-bounds.md)
  - [11. 分类讨论框架](thinking-toolkit/11-classification-and-case-split.md) ⭐
  - [12. 开题清单（高中版）](thinking-toolkit/12-checklist-before-solving.md)

### 知识章节

- **Part 1：集合与常用逻辑** `[高一上]`
  - [01. 集合基础](part1-sets-and-logic/01-sets-basics.md)
  - [02. 集合运算（交并补）](part1-sets-and-logic/02-set-operations.md)
  - [03. 子集与幂集](part1-sets-and-logic/03-subset-and-power-set.md)
  - [04. 全称量词与存在量词](part1-sets-and-logic/04-logic-quantifiers.md) ⭐
  - [05. 充分条件与必要条件](part1-sets-and-logic/05-necessary-sufficient.md) ⭐

- **Part 2：不等式** `[高一上]`
  - [01. 不等式基础](part2-inequalities/01-inequality-basics.md)
  - [02. 一元二次不等式](part2-inequalities/02-quadratic-inequality.md) ⭐
  - [03. 分式不等式](part2-inequalities/03-fractional-inequality.md) ⭐
  - [04. 基本不等式 $a+b \geq 2\sqrt{ab}$](part2-inequalities/04-basic-inequality.md) ⭐

- **Part 3：函数概念与性质** `[高一上]`
  - [01. 函数三要素](part3-function-concepts/01-function-definition.md)
  - [02. 定义域与值域](part3-function-concepts/02-domain-range.md)
  - [03. 单调性](part3-function-concepts/03-monotonicity.md) ⭐
  - [04. 奇偶性](part3-function-concepts/04-parity.md) ⭐
  - [05. 分段函数与含绝对值函数](part3-function-concepts/05-piecewise-and-abs.md)
  - [06. 复合函数](part3-function-concepts/06-composite-function.md) ⭐
  - [07. 函数应用与建模](part3-function-concepts/07-function-applications.md)

- **Part 4：指数 / 对数 / 幂函数** `[高一上]`
  - [01. 指数函数](part4-exp-log-power/01-exponential-function.md)
  - [02. 对数运算与对数函数](part4-exp-log-power/02-logarithm-and-log-function.md) ⭐
  - [03. 幂函数](part4-exp-log-power/03-power-function.md)

- **Part 5：三角函数** `[高一上 / 高一下]`
  - [01. 弧度制](part5-trigonometry/01-radian-measure.md)
  - [02. 单位圆与三角函数定义](part5-trigonometry/02-trig-on-unit-circle.md)
  - [03. 三角函数图象与性质](part5-trigonometry/03-trig-graphs-and-properties.md)
  - [04. 同角关系与诱导公式](part5-trigonometry/04-trig-identities.md) ⭐
  - [05. 和差角与倍半角公式](part5-trigonometry/05-sum-difference-formulas.md) ⭐
  - [06. 三角方程](part5-trigonometry/06-trig-equations.md)
  - [07. 解三角形：正弦定理与余弦定理](part5-trigonometry/07-solving-triangle.md)
  - [08. 三角综合应用](part5-trigonometry/08-trig-applications.md)

- **Part 6：复数** `[高一下]`
  - [01. 复数基础](part6-complex-numbers/01-complex-basics.md)
  - [02. 复数运算](part6-complex-numbers/02-complex-operations.md)

- **Part 7：统计** `[高一下]`
  - [01. 抽样方法](part7-statistics/01-sampling.md)
  - [02. 数据的数字特征](part7-statistics/02-data-features.md)
  - [03. 相关性与回归](part7-statistics/03-correlation.md)

- **Part 8：概率** `[高一下 / 选必三]`
  - [01. 古典概型](part8-probability/01-classical-probability.md)
  - [02. 几何概型](part8-probability/02-geometric-probability.md)
  - [03. 独立性](part8-probability/03-independence.md)
  - [04. 条件概率](part8-probability/04-conditional-probability.md) ⭐
  - [05. 全概率公式与贝叶斯定理](part8-probability/05-bayes-and-total-probability.md) ⭐

- **Part 9：数列** `[高二上]`
  - [01. 数列基础](part9-sequences/01-sequence-basics.md)
  - [02. 等差数列](part9-sequences/02-arithmetic-sequence.md)
  - [03. 等比数列](part9-sequences/03-geometric-sequence.md)
  - [04. 递推关系与通项](part9-sequences/04-recurrence-relations.md) ⭐
  - [05. 求和技巧：错位相减与裂项](part9-sequences/05-sum-techniques.md) ⭐
  - [06. 数列综合应用](part9-sequences/06-sequence-applications.md)

- **Part 10：导数** `[高二上 / 高二下]`
  - [01. 导数概念](part10-derivatives/01-derivative-concept.md)
  - [02. 求导法则](part10-derivatives/02-derivative-rules.md)
  - [03. 用导数求单调区间](part10-derivatives/03-monotonicity-by-derivative.md) ⭐
  - [04. 极值与最值](part10-derivatives/04-extrema-and-optimization.md) ⭐
  - [05. 切线问题](part10-derivatives/05-tangent-line.md)
  - [06. 导数与零点（含参讨论）](part10-derivatives/06-derivative-and-zeros.md) ⭐
  - [07. 导数综合应用](part10-derivatives/07-derivative-applications.md)

- **Part 11：计数原理** `[高二下 / 高三上]`
  - [01. 分类加法与分步乘法原理](part11-counting/01-fundamental-counting.md)
  - [02. 排列](part11-counting/02-permutation.md)
  - [03. 组合](part11-counting/03-combination.md)
  - [04. 二项式定理](part11-counting/04-binomial-theorem.md) ⭐

- **Part 12：随机变量** `[选必三]`
  - [01. 离散型随机变量](part12-random-variables/01-discrete-rv.md)
  - [02. 二项分布](part12-random-variables/02-binomial-distribution.md)
  - [03. 超几何分布](part12-random-variables/03-hypergeometric.md)
  - [04. 正态分布](part12-random-variables/04-normal-distribution.md)
  - [05. 期望与方差](part12-random-variables/05-expectation-variance.md) ⭐

- **Part 13：高考代数综合** `[高三复习]`
  - [01. 函数与导数综合压轴](part13-comprehensive/01-function-and-derivative.md) ⭐
  - [02. 含参导数的存在性问题](part13-comprehensive/02-derivative-existence.md) ⭐
  - [03. 数列与归纳综合](part13-comprehensive/03-sequence-with-induction.md) ⭐
  - [04. 用导数证明不等式](part13-comprehensive/04-inequality-proofs.md) ⭐
  - [05. 三角综合](part13-comprehensive/05-trig-comprehensive.md) ⭐
  - [06. 概率统计综合](part13-comprehensive/06-probability-comprehensive.md) ⭐

### 附录

- [A. 公式速查表](appendix/A-formula-sheet.md)
- [B. 解题套路图集（约 40-50 模型）](appendix/B-model-atlas.md)
- [C. 基础题库（80 题 ☆）](appendix/C-exercises-basic.md)
- [D. 中档题库（100 题 ☆☆）](appendix/D-exercises-medium.md)
- [E. 高考压轴题库（60 题 ☆☆☆）](appendix/E-exercises-advanced.md)
- [F1. 详解：集合、不等式、函数概念（Part 1-3）](appendix/F1-solutions-sets-inequalities-functions.md)
- [F2. 详解：指数对数、三角、复数（Part 4-6）](appendix/F2-solutions-exp-trig-complex.md)
- [F3. 详解：统计、概率、数列、导数（Part 7-10）](appendix/F3-solutions-stat-prob-seq-deriv.md)
- [F4. 详解：计数、随机变量、综合（Part 11-13）](appendix/F4-solutions-counting-rv-comprehensive.md)

---

## 怎么用这本书

**高一新生（Part 1-5 顺序路线）**：

先通读思维工具箱前 4 篇（01 结构识别、03 构造法、04 数形结合、11 分类讨论），建立方法论框架；然后按 Part 1 → 2 → 3 → 4 → 5 顺序推进。每遇到带 ⭐ 的章节，先花 10-15 分钟自己尝试"思维路径还原"那道引入题，再对照书中的分析。完成高一上内容后再读 Part 5（三角函数）。

**高二学生（Part 6-10 进阶路线）**：

先补齐思维工具箱剩余篇（05 含参讨论、06 抽象函数、07 复合函数、08 单调极值），再按 Part 9 数列 → Part 10 导数顺序推进。Part 10 是高考压轴的核心区域，建议每章与附录 B 套路图集配合使用——学完一个方法就去附录 B 做"看式说法"训练。

**高三复习（综合冲刺路线）**：

直接跳 Part 13（6 章全是高考压轴套路）+ 思维工具箱 12 篇完整通读 + 附录 C/D/E 三档题库。遇到不熟悉的知识点再回查对应 Part。附录 E（60 道压轴题）每题都标注"方法/模型标签"，可对照检验自己的套路识别能力。

---

## 章节内部结构

本书章节分两种模板：

**模板 C（概念/运算法则类）**，如集合运算、指数函数、等差数列：
> 一、概念特征 → 二、定义与定理 → 三、推导过程 → 四、典型应用（带【思路】）→ 五、易错点 → 六、思路自测题

**模板 D（套路/题型类）**，即所有带 ⭐ 标记的章节，字数 3500-5000：
> **一例速记**（极简样板题）→ 一、引入题 → 二、**思维路径还原**（15-20 行高手内心独白，`>` 引用块）→ 三、抽象成方法 → 四、方法变形 → 五、**思考路标**（≥8 条"看到 X → 想到 Y"）→ 六、应用例题（含至少 1 道含参讨论或严格证明）→ 七、思路自测题

模板 D 的**思维路径还原**和**思考路标**是本书核心特色——把"高手盯着题的第一秒想什么"系统化、可复用化。

---

## 关于练习题

- **节内例题**：完整解答，附【思路】小节，说明方法选择的理由
- **节末思路自测题**：仅给提示，不给完整解答，逼你自己走一遍思维路径
- **附录题库 240 题**：基础 80 题（☆）/ 中档 100 题（☆☆）/ 高考压轴 60 题（☆☆☆）；每题详解标注方法/模型标签（如 `[构造函数 + 导数判单调]`、`[参数分离 + 图象法]`），便于回查对应章节

与中考代数（180 题）相比，本书 240 题是因为高考题单题工作量更大——一道压轴题可能综合函数、导数、不等式三个知识点，需要更多专项训练。

---

## 关于图

本书图分两类：

- **TikZ 源码**（存于 `figures/src/tikz/`）：函数图象（指数/对数/三角/二次/分段）、数轴、统计图、概率树状图；用 `pgfplots` 绘制连续曲线
- **Asymptote 源码**（存于 `figures/src/asy/`）：复数几何图、极坐标图、较复杂的曲线族

两类源文件均通过 [`figures/render.sh`](figures/render.sh) 编译为 SVG（xelatex 支持中文）。图形风格规范见 [`figures/STYLE.md`](figures/STYLE.md)。

---

## 反馈

本教程是开源项目，欢迎指出错误或提出改进建议。如发现推导跳步、套路描述不准确、或有比书中更优雅的方法，请提 issue 或 PR。
