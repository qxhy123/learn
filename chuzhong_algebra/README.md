# 中考代数教程

一份强调**思路培养**的中考代数（含统计与概率）教程。不仅讲"怎么算"，更讲"为什么这么变形 / 看到 X 应该立刻想 Y"。核心主张：**结构识别先于计算，整体思想优于硬拆，算两次生方程**。

**配套姊妹篇**：[`chuzhong_geometry/`](../chuzhong_geometry/) 中考几何教程

---

## 这本书写给谁

- **零基础读者**：刚接触代数、想从整数运算到函数一步步打地基的初学者
- **中等水平读者**：已学过课本，但一遇到分式方程、含参问题、函数综合题就卡壳的学习者
- **中考冲刺读者**：准备中考，想系统掌握应用题设元策略和代数压轴题型的考生

---

## 与普通教辅的不同

普通教辅把公式列出来 → 配例题 → 配习题。本书额外强调：

1. **看到一个式子，脑子里第一秒应该想什么** —— 把解题者的"内心独白"完整写出来
2. **式子结构 → 方法 → 变形** 的反射弧训练（配方、换元、整体、算两次……）
3. **思维工具箱**：10 篇专注方法论的小册子，尤其是"算两次""结构识别""整体思想""数形结合"四篇，是可独立阅读的代数核心思维
4. **模板 D 章节**（带 ⭐️ 标记，共 32 章）含"**一例速记 + 思维路径还原 + 思考路标**"——把高手脑子里发生的事写在台面上

---

## 年级标签说明

**基准教材：北师大版 2024 新课标版**（陕西及多省通用）。

`[初一上]` `[初一下]` `[初二上]` `[初二下]` `[初三上]` `[初三下]` 对应北师大版的开课学期；`[初三复习]` 指中考综合训练阶段。以**该知识点首次系统教授**的学期为准；同一 part 的不同章节可能跨学期。

> ⚠️ 若你的教材是**人教版/苏科版/华师大版**等，章节顺序与开课学期会有差异。建议以学校当年教材目录为准。

---

## 目录

### 方法论（先读这部分）

- [序言：什么是"代数思维"](00-preface.md) `[通用]`
- [思维工具箱](thinking-toolkit/) `[通用 / 初三复习集中读]`
  - [01. 何时换元](thinking-toolkit/01-when-to-substitute.md)
  - [02. 何时配方](thinking-toolkit/02-when-to-complete-square.md)
  - [03. 判别式 + 韦达定理思想](thinking-toolkit/03-discriminant-and-vieta.md)
  - [04. 设元的艺术](thinking-toolkit/04-setting-up-equations.md)
  - [05. 整体思想](thinking-toolkit/05-integral-thinking.md)
  - [06. 算两次思想](thinking-toolkit/06-counting-twice.md) ⭐️
  - [07. 结构识别](thinking-toolkit/07-structure-recognition.md) ⭐️
  - [08. 参数分离 + 含参讨论框架](thinking-toolkit/08-parameter-strategy.md) ⭐️
  - [09. 数形结合（式 ↔ 图）](thinking-toolkit/09-number-shape-combination.md) ⭐️
  - [10. 代数题开题清单](thinking-toolkit/10-checklist-before-solving.md)

### 知识章节（按需阅读）

- **Part 1：数（含根式）** `[初一上 / 七上 第二章 + 初二上根式]`
  - [01. 有理数](part1-numbers/01-rational-numbers.md)
  - [02. 有理数运算](part1-numbers/02-rational-operations.md)
  - [03. 乘方与科学记数法](part1-numbers/03-power-and-scientific.md)
  - [04. 实数](part1-numbers/04-real-numbers.md)
  - [05. 平方根与立方根](part1-numbers/05-square-cube-roots.md)
  - [06. 根式化简](part1-numbers/06-radicals.md)
  - [07. 根式运算技巧](part1-numbers/07-radical-operations.md) ⭐️

- **Part 2：整式** `[初一下 / 七下 第三章]`
  - [01. 单项式与多项式](part2-polynomials/01-monomial-polynomial.md)
  - [02. 整式加减](part2-polynomials/02-add-subtract.md)
  - [03. 指数法则](part2-polynomials/03-exponent-rules.md)
  - [04. 整式乘法](part2-polynomials/04-multiply.md)
  - [05. 乘法公式（完全平方/平方差）](part2-polynomials/05-special-formulas.md) ⭐️
  - [06. 整式除法](part2-polynomials/06-divide.md)
  - [07. 公式逆用与变形技巧](part2-polynomials/07-special-formula-tricks.md) ⭐️

- **Part 3：因式分解** `[初二下 / 八下 第四章]`
  - [01. 提取公因式](part3-factorization/01-common-factor.md)
  - [02. 公式法](part3-factorization/02-formula-method.md)
  - [03. 十字相乘法](part3-factorization/03-cross-multiplication.md) ⭐️
  - [04. 分组分解](part3-factorization/04-grouping.md)
  - [05. 综合因式分解策略](part3-factorization/05-factor-strategy.md) ⭐️

- **Part 4：分式** `[初二下 / 八下 第五章]`
  - [01. 分式基础](part4-fractions/01-fraction-basics.md)
  - [02. 分式运算](part4-fractions/02-fraction-operations.md)
  - [03. 分式方程与增根](part4-fractions/03-fraction-equation.md) ⭐️
  - [04. 分式化简技巧（裂项/通分）](part4-fractions/04-fraction-techniques.md) ⭐️

- **Part 5：一次方程（组）** `[初一上 / 七上 第五章 + 初二上方程组]`
  - [01. 一元一次方程](part5-linear-eq/01-linear-one-variable.md)
  - [02. 二元一次方程组](part5-linear-eq/02-linear-two-variable.md)
  - [03. 加减代入消元法](part5-linear-eq/03-elimination-methods.md)
  - [04. 一次方程解题技巧](part5-linear-eq/04-linear-eq-tricks.md) ⭐️
  - [05. 含参一次方程组](part5-linear-eq/05-parametric-linear.md) ⭐️

- **Part 6：一元二次方程** `[初三上 / 九上 第二章]`
  - [01. 一元二次方程基础](part6-quadratic-eq/01-quadratic-basics.md)
  - [02. 直接开平方法与因式分解法](part6-quadratic-eq/02-direct-and-factoring.md)
  - [03. 配方法](part6-quadratic-eq/03-completing-square.md) ⭐️
  - [04. 求根公式](part6-quadratic-eq/04-quadratic-formula.md)
  - [05. 韦达定理](part6-quadratic-eq/05-vieta-theorem.md) ⭐️
  - [06. 一元二次方程应用题](part6-quadratic-eq/06-quadratic-applications.md)

- **Part 7：不等式** `[初二下 / 八下 第二章]`
  - [01. 不等式基础](part7-inequalities/01-inequality-basics.md)
  - [02. 解一元一次不等式](part7-inequalities/02-solve-linear-inequality.md)
  - [03. 不等式组](part7-inequalities/03-inequality-system.md)
  - [04. 不等式应用题](part7-inequalities/04-inequality-applications.md)
  - [05. 含参不等式的分类讨论](part7-inequalities/05-parametric-inequality.md) ⭐️

- **Part 8：一次函数** `[初二上 / 八上 第四章]`
  - [01. 函数的概念](part8-linear-functions/01-function-concept.md)
  - [02. 一次函数基础](part8-linear-functions/02-linear-function-basics.md)
  - [03. 待定系数法求解析式](part8-linear-functions/03-finding-expression.md) ⭐️
  - [04. 一次函数应用题](part8-linear-functions/04-linear-function-applications.md)
  - [05. 方程组与交点（几何意义）](part8-linear-functions/05-system-and-line.md) ⭐️
  - [06. 含参一次函数（图象法/动态）](part8-linear-functions/06-parametric-linear-fn.md) ⭐️

- **Part 9：反比例函数** `[初三上 / 九上 第三章]`
  - [01. 反比例函数基础](part9-inverse-proportional/01-inverse-basics.md)
  - [02. 反比例函数性质（k 的几何意义）](part9-inverse-proportional/02-inverse-properties.md) ⭐️
  - [03. 反比例函数应用](part9-inverse-proportional/03-inverse-applications.md)

- **Part 10：二次函数** `[初三下 / 九下 第二章]`
  - [01. 二次函数基础](part10-quadratic-functions/01-quadratic-fn-basics.md)
  - [02. 三种表达形式（一般式/顶点式/交点式）](part10-quadratic-functions/02-three-forms.md) ⭐️
  - [03. 二次函数的性质](part10-quadratic-functions/03-quadratic-fn-properties.md)
  - [04. 待定系数法求解析式](part10-quadratic-functions/04-finding-expression.md)
  - [05. 二次函数应用题](part10-quadratic-functions/05-quadratic-fn-applications.md)
  - [06. 函数与方程的关系](part10-quadratic-functions/06-fn-and-equation.md) ⭐️
  - [07. 含参二次函数最值](part10-quadratic-functions/07-parametric-quadratic-fn.md) ⭐️
  - [08. 二次函数中三角形面积（铅垂高法）](part10-quadratic-functions/08-quadratic-fn-area.md) ⭐️

- **Part 11：统计** `[初二上 / 八上 第六章 + 初三上]`
  - [01. 数据收集与描述](part11-statistics/01-data-collection.md)
  - [02. 统计图（条形/折线/扇形/直方图）](part11-statistics/02-statistical-charts.md)
  - [03. 平均数、中位数、众数](part11-statistics/03-central-tendency.md)
  - [04. 方差与样本估计](part11-statistics/04-variance.md)

- **Part 12：概率** `[初三上 / 九上 第五章]`
  - [01. 可能性与事件](part12-probability/01-possibility-and-events.md)
  - [02. 列举法与树状图求概率](part12-probability/02-counting-and-tree.md) ⭐️
  - [03. 用频率估计概率](part12-probability/03-frequency-and-probability.md)

- **Part 13：应用题专题** `[初三复习]`
  - [01. 设元的策略（贯穿全部题型）](part13-applications/01-setup-strategies.md) ⭐️
  - [02. 行程问题模型](part13-applications/02-distance-problems.md) ⭐️
  - [03. 工程问题模型](part13-applications/03-work-problems.md) ⭐️
  - [04. 利润/折扣/利率模型](part13-applications/04-profit-problems.md) ⭐️
  - [05. 方案设计与最优化](part13-applications/05-scheme-design.md) ⭐️
  - [06. 配比/浓度/调配模型](part13-applications/06-mixture-problems.md) ⭐️
  - [07. 增长率模型](part13-applications/07-rate-of-change.md) ⭐️
  - [08. 资源分配模型](part13-applications/08-allocation.md) ⭐️

- **Part 14：中考代数综合** `[初三复习]`
  - [01. 多种函数综合](part14-comprehensive/01-functions-comprehensive.md) ⭐️
  - [02. 二次函数与几何综合](part14-comprehensive/02-quadratic-fn-with-geometry.md) ⭐️
  - [03. 函数中的存在性问题](part14-comprehensive/03-existence-in-functions.md) ⭐️
  - [04. 函数图象上的动点](part14-comprehensive/04-moving-points-on-graph.md) ⭐️
  - [05. 新定义型代数题](part14-comprehensive/05-new-definition.md)
  - [06. 含绝对值/分段题型](part14-comprehensive/06-piecewise-and-absolute.md)

### 附录

- [A. 公式速查表](appendix/A-formula-sheet.md)
- [B. 解题套路图集（看式说法）](appendix/B-model-atlas.md)
- [C. 基础题库（60 题 ☆）](appendix/C-exercises-basic.md)
- [D. 中档题库（80 题 ☆☆）](appendix/D-exercises-medium.md)
- [E. 压轴题库（40 题 ☆☆☆）](appendix/E-exercises-advanced.md)
- [F1. 详解：数与整式（part1-4）](appendix/F1-solutions-numbers-and-expressions.md)
- [F2. 详解：方程与不等式（part5-7）](appendix/F2-solutions-equations-and-inequalities.md)
- [F3. 详解：函数（part8-10）](appendix/F3-solutions-functions.md)
- [F4. 详解：统计/概率/应用/综合（part11-14）](appendix/F4-solutions-stat-prob-apps-comp.md)

---

## 怎么用这本书

**零基础读者**：按 Part 1 → Part 14 顺序阅读。每读完一个 Part，回到"思维工具箱"对应章节回顾。重点先完成模板 C（概念类）章节，再挑战带 ⭐️ 的模板 D（技巧类）章节。

**已学过课本的读者**：先通读"思维工具箱"10 篇，建立方法论框架；再按主题切入——比如正在学二次方程，就直接 Part 6 + toolkit 02（配方）+ toolkit 03（韦达）组合阅读。模板 D 章节是重点，先自己尝试"思维路径还原"那道引入题，再对照书中分析。

**中考冲刺读者**：直接跳 Part 13（应用题专题）+ Part 14（综合）；遇到不熟悉的知识点再回查对应 Part。题库练习路径：附录 C → D → E，难度递进。配合几何教程，综合题往往需要"代数计算 + 几何图形识别"双线作战。

---

## 章节内部结构

本书章节分两种模板：

**模板 C（概念/运算法则类）**，如有理数运算、根式化简、统计图：
> 一、概念特征 → 二、定义与运算法则 → 三、为什么这么算（推导） → 四、典型应用（2-3 例题，带【思路】）→ 五、易错点 & 反例 → 六、思路自测题

**模板 D（技巧/套路类）**，即所有带 ⭐️ 标记的章节：
> **一例速记**（极简样板题代替几何的"一图速记"）→ 一、引入题 → 二、**思维路径还原**（解题者的内心独白，体现"看到 X → 想到 Y"的反射弧）→ 三、抽象成方法 → 四、方法变形 → 五、**思考路标**（至少 6 条"看到 X → 想到 Y"）→ 六、应用例题 → 七、思路自测题

模板 D 章节的"**思维路径还原**"和"**思考路标**"是本书的核心特色——它们把高手解题时脑中发生的事完整摊开，让"灵感"变成可复用的反射弧。

---

## 关于练习题

- **节内例题**：完整解答，附【思路】小节（包含方法选择的理由）
- **节末"思路自测题"**：仅给 💡 提示，不给完整解答——逼你自己走一遍思维路径
- **附录题库 180 题**：基础 60 题（☆）/ 中档 80 题（☆☆）/ 压轴 40 题（☆☆☆），覆盖 14 个 part 的核心题型；每题详解标注"方法/模型标签"（如 `[配方法 + 韦达]`、`[设元 + 行程模型]`），便于回查对应章节

---

## 关于图

本书图分两类：

- **TikZ 源码**（嵌入 markdown 代码块，可被 LaTeX 渲染器直接出图）——多数概念图和数轴图
- **预渲染 SVG**（复杂函数图象或统计图，存放于 `figures/svg/`）

代数教程的图以**函数图象**为主（一次/反比例/二次函数图象、含参图象），辅以数轴、统计图、概率树状图、应用题示意图。图的风格规范见 [`figures/STYLE.md`](figures/STYLE.md)。

---

## 反馈

本教程是开源项目，欢迎指出错误或提出改进建议。
