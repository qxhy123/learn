# 高中代数教程 设计文档

**创建日期**：2026-05-13
**目标目录**：`gaozhong_math/algebra/`（与 `gaozhong_math/geometry/` 平行，留给以后做）
**教材基准**：人教 A 版 2019 新课标（必修一二 + 选择性必修一二三）
**对照参考**：`chuzhong_algebra/`（已完成的中考代数教程，相同风格与基础设施）
**节奏**：**先做高一上"种子"（Part 1-5，约 25 章）**，看效果再决定是否扩到全套
**状态**：设计已确认，待生成实施计划

---

## 0. 设计要点

照搬 chuzhong_algebra 的成功模式：
- 两种章节模板 C / D（模板 D 含"思维路径还原 + 思考路标"）
- thinking-toolkit 思维方法论独立小册子
- appendix（公式表 + 模型图集 + 三档题库 + 详解）
- TikZ/Asymptote + xelatex + ctex 渲图基础设施（可复用 `rendering-math-figures` skill）
- 按里程碑分批执行，每批 subagent 并行

**与初中代数的关键差异**：
- 高中概念抽象度更高（如**抽象函数**、**复合函数**、**含参极值**、**导数应用**）
- 证明要求更严格（高考压轴常考严格推理）
- 题型套路更密集（每章 3-5 个常用解题套路是常态）
- 字数：模板 C 章节 2200-3200 字（初中是 1800-2500），模板 D 章节 3500-5000 字（初中是 3000-4000）

---

## 1. 定位与风格

- **读者**：高中学生 / 高考备考者 / 想自学高中代数的成年读者
- **核心特色**：**高中代数思路培养**——强调结构识别、构造法、数形结合、参数分离、极限思想、抽象推理等
- **基础设施**：用 `~/.claude/skills/rendering-math-figures/templates/` 中的 render.sh 和 STYLE.md 初始化

---

## 2. 范围切分（algebra vs geometry）

`gaozhong_math/algebra/` 涵盖：
- 集合与常用逻辑用语
- 函数（一般概念、指数、对数、幂、三角函数）
- 不等式
- 数列
- 导数与初等函数应用
- 复数
- 概率（古典/几何概型/条件概率/独立性）
- 统计（数据/分布/相关性）
- 计数原理（排列组合/二项式定理）
- 解三角形（含正/余弦定理）

`gaozhong_math/geometry/`（**以后做**）涵盖：
- 平面向量、空间向量
- 解析几何（直线、圆、圆锥曲线）
- 立体几何

---

## 3. 整体目录结构（13 part + thinking-toolkit + appendix）

```
gaozhong_math/algebra/
├── README.md
├── 00-preface.md
│
├── thinking-toolkit/                      # 高中代数思维工具箱（12 篇）
│   ├── 01-structure-recognition.md        # 高级结构识别
│   ├── 02-substitution-and-simplification.md  # 换元与化简策略
│   ├── 03-construction.md                 # 构造法（构造函数/不等式/数列）
│   ├── 04-number-shape-deep.md            # 数形结合（高中深化版）
│   ├── 05-parameter-strategy.md           # 含参讨论与参数分离
│   ├── 06-abstract-function.md            # 抽象函数推理
│   ├── 07-composite-function.md           # 复合函数与函数嵌套
│   ├── 08-monotonicity-extrema.md         # 单调性与极值思想（含导数前置）
│   ├── 09-symmetry-and-periodicity.md     # 对称性与周期性
│   ├── 10-limit-and-bounds.md             # 极限与放缩思想
│   ├── 11-classification-and-case-split.md  # 分类讨论框架
│   └── 12-checklist-before-solving.md     # 开题清单（高中版）
│
├── part1-sets-and-logic/                  # 集合与常用逻辑 [高一上]
│   ├── 01-sets-basics.md
│   ├── 02-set-operations.md
│   ├── 03-subset-and-power-set.md
│   ├── 04-logic-quantifiers.md            # ⭐ 全称/存在量词与否定
│   └── 05-necessary-sufficient.md         # ⭐ 充分必要条件
│
├── part2-inequalities/                    # 不等式 [高一上]
│   ├── 01-inequality-basics.md
│   ├── 02-quadratic-inequality.md         # ⭐ 一元二次不等式
│   ├── 03-fractional-inequality.md        # ⭐ 分式不等式（高中范畴）
│   └── 04-basic-inequality.md             # ⭐ 基本不等式 a+b ≥ 2√(ab)
│
├── part3-function-concepts/               # 函数概念与性质 [高一上]
│   ├── 01-function-definition.md          # 函数三要素
│   ├── 02-domain-range.md                 # 定义域、值域
│   ├── 03-monotonicity.md                 # ⭐ 单调性
│   ├── 04-parity.md                       # ⭐ 奇偶性
│   ├── 05-piecewise-and-abs.md            # 分段函数与含绝对值
│   ├── 06-composite-function.md           # ⭐ 复合函数
│   └── 07-function-applications.md        # 函数应用与建模
│
├── part4-exp-log-power/                   # 指数 / 对数 / 幂函数 [高一上]
│   ├── 01-exponential-function.md
│   ├── 02-logarithm-and-log-function.md   # ⭐ 对数运算与对数函数
│   └── 03-power-function.md
│
├── part5-trigonometry/                    # 三角函数 [高一上 / 高一下]
│   ├── 01-radian-measure.md
│   ├── 02-trig-on-unit-circle.md          # 单位圆 + 三角函数定义
│   ├── 03-trig-graphs-and-properties.md
│   ├── 04-trig-identities.md              # ⭐ 同角/诱导公式
│   ├── 05-sum-difference-formulas.md      # ⭐ 和差倍半角公式
│   ├── 06-trig-equations.md
│   ├── 07-solving-triangle.md             # 解三角形 + 正余弦定理
│   └── 08-trig-applications.md            # 三角应用与综合
│
├── part6-complex-numbers/                 # 复数 [高一下]
│   ├── 01-complex-basics.md
│   └── 02-complex-operations.md
│
├── part7-statistics/                      # 统计 [高一下]
│   ├── 01-sampling.md
│   ├── 02-data-features.md
│   └── 03-correlation.md
│
├── part8-probability/                     # 概率 [高一下 + 选必三]
│   ├── 01-classical-probability.md
│   ├── 02-geometric-probability.md
│   ├── 03-independence.md
│   ├── 04-conditional-probability.md      # ⭐ 条件概率（选必三）
│   └── 05-bayes-and-total-probability.md  # ⭐ 全概率公式 + 贝叶斯
│
├── part9-sequences/                       # 数列 [高二上]
│   ├── 01-sequence-basics.md
│   ├── 02-arithmetic-sequence.md          # 等差数列
│   ├── 03-geometric-sequence.md           # 等比数列
│   ├── 04-recurrence-relations.md         # ⭐ 由递推求通项
│   ├── 05-sum-techniques.md               # ⭐ 求和技巧（错位相减/裂项）
│   └── 06-sequence-applications.md
│
├── part10-derivatives/                    # 导数 [高二上/下]
│   ├── 01-derivative-concept.md
│   ├── 02-derivative-rules.md             # 求导法则
│   ├── 03-monotonicity-by-derivative.md   # ⭐ 用导数求单调区间
│   ├── 04-extrema-and-optimization.md     # ⭐ 极值最值
│   ├── 05-tangent-line.md                 # 切线问题
│   ├── 06-derivative-and-zeros.md         # ⭐ 导数与零点（含参讨论）
│   └── 07-derivative-applications.md      # 导数综合
│
├── part11-counting/                       # 计数原理 [高二下/高三上]
│   ├── 01-fundamental-counting.md         # 分类加法、分步乘法
│   ├── 02-permutation.md                  # 排列
│   ├── 03-combination.md                  # 组合
│   └── 04-binomial-theorem.md             # ⭐ 二项式定理
│
├── part12-random-variables/               # 随机变量 [选必三]
│   ├── 01-discrete-rv.md                  # 离散型随机变量
│   ├── 02-binomial-distribution.md
│   ├── 03-hypergeometric.md
│   ├── 04-normal-distribution.md          # 正态分布
│   └── 05-expectation-variance.md         # ⭐ 期望与方差
│
├── part13-comprehensive/                  # 高考代数综合 [高三复习]
│   ├── 01-function-and-derivative.md      # ⭐ 函数与导数综合压轴
│   ├── 02-derivative-existence.md         # ⭐ 含参导数的存在性
│   ├── 03-sequence-with-induction.md      # ⭐ 数列与归纳综合
│   ├── 04-inequality-proofs.md            # ⭐ 用导数证明不等式
│   ├── 05-trig-comprehensive.md           # ⭐ 三角综合
│   └── 06-probability-comprehensive.md    # ⭐ 概率统计综合
│
├── figures/
│   ├── render.sh
│   ├── STYLE.md
│   ├── src/{tikz,asy}/
│   └── svg/
│
└── appendix/
    ├── A-formula-sheet.md
    ├── B-model-atlas.md                   # 套路图集（约 40-50 模型）
    ├── C-exercises-basic.md               # 基础题库 80 题（☆）
    ├── D-exercises-medium.md              # 中档题库 100 题（☆☆）
    ├── E-exercises-advanced.md            # 高考压轴 60 题（☆☆☆）
    └── F1-F4-solutions.md                 # 240 题详解（按 part 分 4 文件）
```

**规模估算**：~80 章正文 + 12 篇 toolkit + 6 个 appendix 文件 = **约 100 个 md 文件**，约 **45 个 ⭐ 模板 D 章节**。

---

## 4. 章节内部模板

**完全照搬 chuzhong_algebra**：

- **模板 C（概念/运算法则类，6 节）**：概念特征 / 定义与定理 / 推导证明 / 典型应用（带【思路】）/ 易错点 / 思路自测题。**字数 2200-3200**。
- **模板 D（套路/题型类，7 节）**：一例速记 / 引入 / 思维路径还原（`>` 引用，15-20 行）/ 抽象成方法 / 方法变形 / 思考路标（≥8 条）/ 例题 + 自测。**字数 3500-5000**。

**与初中代数的关键差异**：
- 模板 D 的"思维路径还原"更长（15-20 行 vs 12-18 行），因为高中题目本身的推理链条更长
- "思考路标"至少 8 条（初中是 6 条）
- 例题中至少 1 道带"含参讨论"或"严格证明"

---

## 5. 图、公式、约定

- 完全复用 `~/.claude/skills/rendering-math-figures/` 中的 render.sh、STYLE.md
- 函数图象（指数/对数/三角/二次/分段）用 TikZ + pgfplots
- 复数 / 圆锥曲线（如涉及）用 Asymptote
- 数轴 / 统计图 / 概率树用 TikZ

---

## 6. 练习题系统

- 节内 2-4 例 + 节末 3-5 题"思路自测"
- 附录 **C 80 题 + D 100 题 + E 60 题 = 240 题**（初中是 180，高中题更难、单题工作量更大）
- F 按 part 大组拆 4 个文件

---

## 7. 里程碑路线（先种子，再决定）

### **种子阶段（高一上，Part 1-5）**

| 里程碑 | 内容 | 文件数 |
|---|---|---|
| **GM0** | README + preface + thinking-toolkit 12 篇 + figures 基础设施 | 15 |
| **GM1** | Part 1 集合与逻辑（5 章）+ Part 2 不等式（4 章） | 9 |
| **GM2** | Part 3 函数概念与性质（7 章） | 7 |
| **GM3** | Part 4 指数对数幂（3 章）+ Part 5 三角函数（8 章） | 11 |
| **GM4** | 种子阶段 appendix 子集（公式表 + 模型图集 + 高一上题库 80 题） | 4 |

**种子产出**：约 **45 个 md 文件**（含 toolkit、序言、README、4 个 appendix），覆盖**高一上完整代数内容**。

**完成种子后决策点**：
- 看效果好 → 继续 GM5-GM12 做完高二高三 + 全套 appendix + 配图
- 看效果一般 → 调整模板、字数、深度后再继续
- 暂停 → 种子也已是独立可用的"高一上代数学习指南"

### 全套阶段（种子通过后）

| 里程碑 | 内容 | 文件数 |
|---|---|---|
| GM5 | Part 6 复数 + Part 7 统计 + Part 8 概率 | 10 |
| GM6 | Part 9 数列 6 章 | 6 |
| GM7 | Part 10 导数 7 章（最重） | 7 |
| GM8 | Part 11 计数 + Part 12 随机变量 | 9 |
| GM9 | Part 13 高考综合 6 章 | 6 |
| GM10 | 全套 appendix（补 D 中档 + E 压轴 + 全 F 详解） | 5 |
| GM11 | 全教程配图 | ~150-200 张 |

**全套总计约 100 个 md + 150-200 张 SVG + 240 题 + 详解**。

---

## 8. 不在本设计范围内（YAGNI）

- 不做向量、解析几何、立体几何（留给 `gaozhong_math/geometry/`）
- 不覆盖大学数学（数学分析、线性代数等）
- 不做互动 web 版 / 视频
- 不做习题自动判分
- 不做英文版

---

## 9. 成功标准

- 每个模板 D 章节能让学生"看到 X 立刻想 Y"（如"看到 $\dfrac{a+b}{2} \geq \sqrt{ab}$ 形式立刻想基本不等式套路"）
- thinking-toolkit 12 篇能作为独立"高中代数方法论"小册子
- Part 10 导数章节能覆盖近 5 年高考压轴题的所有套路
- 240 题题库每题反查模型 / 思路标签
- 与 `chuzhong_algebra/` 配合使用，形成"初中 + 高中"完整代数体系
