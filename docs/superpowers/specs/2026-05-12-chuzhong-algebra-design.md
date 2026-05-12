# 中考代数（含统计与概率）教程 设计文档（v2）

**创建日期**：2026-05-12（v2 修订）
**目标目录**：`chuzhong_algebra/`（与 `chuzhong_geometry/` 同级，在 `/Users/yangyang/ai_projs/math/`）
**教材基准**：北师大版 2024 新课标 / 陕西适用
**对照参考**：`chuzhong_geometry/`（已完成的中考几何教程）
**状态**：v2 设计已确认，待生成实施计划

---

## 0. v1 → v2 修订要点

v1 spec 的问题与本次修订：

1. **范围漏洞**：v1 漏了统计与概率（中考四大板块之一）。v2 新增 `part11-statistics`（4 章）+ `part12-probability`（3 章），章节总数 75 → ~94。
2. **核心特色不到位**：v1 直接照搬几何模板，但代数"思路"本质是**结构识别 + 变形策略**而非图形识别。v2 修订：
   - thinking-toolkit 从 6 篇扩为 **10 篇**，新增"算两次思想 / 结构识别 / 参数分离 / 数形结合"四篇代数核心方法论
   - 模板 D 章节从 13 个增至 **30+ 个**，每个 part 至少 2 个（除概念为主的 part1 数、part11 统计外）
   - 每个 D 章节字数从 2500-3500 增至 **3000-4000**，"思维路径还原"写得更细
3. **题库规模偏小**：v1 共 100 题对 12+ 个 part 分摊太薄。v2 扩为 **180 题**（60 基础 + 80 中档 + 40 压轴），详解 F 拆 4 个文件按 part 大组组织，每题保留模型/思路标签。

---

## 1. 定位与风格

- **读者**：综合型（零基础到中考冲刺）
- **核心特色**：**代数思路培养**——强调结构识别、变形策略、整体思想、算两次、数形结合等代数本质思维；模板 D 章节含"思维路径还原 + 思考路标"
- **基础设施**：拷贝 `chuzhong_geometry/figures/render.sh` 与 `STYLE.md`（增补代数补充约定）

---

## 2. 目录结构（14 part + 10 toolkit + appendix）

```
chuzhong_algebra/
├── README.md                                  # 总入口（含年级标签）
├── 00-preface.md                              # 序言：什么是"代数思维"
│
├── thinking-toolkit/                          # 代数思维工具箱（10 篇方法论）
│   ├── 01-when-to-substitute.md               # 何时换元
│   ├── 02-when-to-complete-square.md          # 何时配方
│   ├── 03-discriminant-and-vieta.md           # 判别式 + 韦达定理思想
│   ├── 04-setting-up-equations.md             # 设元的艺术
│   ├── 05-integral-thinking.md                # 整体思想（整体代入/换元/平移）
│   ├── 06-counting-twice.md                   # ⭐NEW 算两次思想（一量两式 = 方程）
│   ├── 07-structure-recognition.md            # ⭐NEW 结构识别（看式子识方法）
│   ├── 08-parameter-strategy.md               # ⭐NEW 参数分离 + 含参讨论框架
│   ├── 09-number-shape-combination.md         # ⭐NEW 数形结合（式 ↔ 图）
│   └── 10-checklist-before-solving.md         # 代数题开题清单
│
├── part1-numbers/                             # 数 [初一上 + 初二上根式]
│   ├── 01-rational-numbers.md
│   ├── 02-rational-operations.md
│   ├── 03-power-and-scientific.md
│   ├── 04-real-numbers.md
│   ├── 05-square-cube-roots.md
│   ├── 06-radicals.md
│   └── 07-radical-operations.md
│
├── part2-polynomials/                         # 整式 [初一上 + 初一下]
│   ├── 01-monomial-polynomial.md
│   ├── 02-add-subtract.md
│   ├── 03-exponent-rules.md
│   ├── 04-multiply.md
│   ├── 05-special-formulas.md                 # ⭐ 完全平方/平方差
│   ├── 06-divide.md
│   └── 07-special-formula-tricks.md           # ⭐NEW 公式逆用与变形
│
├── part3-factorization/                       # 因式分解 [初二下]
│   ├── 01-common-factor.md
│   ├── 02-formula-method.md
│   ├── 03-cross-multiplication.md             # ⭐ 十字相乘
│   ├── 04-grouping.md
│   └── 05-factor-strategy.md                  # ⭐ 综合因式分解思路
│
├── part4-fractions/                           # 分式 [初二下]
│   ├── 01-fraction-basics.md
│   ├── 02-fraction-operations.md
│   ├── 03-fraction-equation.md                # ⭐ 分式方程及增根
│   └── 04-fraction-techniques.md              # ⭐ 分式化简技巧（裂项/通分）
│
├── part5-linear-eq/                           # 一次方程（组） [初一上 + 初二上]
│   ├── 01-linear-one-variable.md
│   ├── 02-linear-two-variable.md
│   ├── 03-elimination-methods.md
│   ├── 04-linear-eq-tricks.md                 # ⭐ 解一次方程技巧
│   └── 05-parametric-linear.md                # ⭐NEW 含参一次方程组
│
├── part6-quadratic-eq/                        # 一元二次方程 [初三上]
│   ├── 01-quadratic-basics.md
│   ├── 02-direct-and-factoring.md
│   ├── 03-completing-square.md                # ⭐ 配方法
│   ├── 04-quadratic-formula.md
│   ├── 05-vieta-theorem.md                    # ⭐ 韦达定理
│   └── 06-quadratic-applications.md
│
├── part7-inequalities/                        # 不等式 [初二下]
│   ├── 01-inequality-basics.md
│   ├── 02-solve-linear-inequality.md
│   ├── 03-inequality-system.md
│   ├── 04-inequality-applications.md
│   └── 05-parametric-inequality.md            # ⭐NEW 含参不等式分类讨论
│
├── part8-linear-functions/                    # 一次函数 [初二上]
│   ├── 01-function-concept.md
│   ├── 02-linear-function-basics.md
│   ├── 03-finding-expression.md               # ⭐ 待定系数法
│   ├── 04-linear-function-applications.md
│   ├── 05-system-and-line.md                  # ⭐ 与方程组的几何意义
│   └── 06-parametric-linear-fn.md             # ⭐NEW 含参一次函数（图象法、动态）
│
├── part9-inverse-proportional/                # 反比例函数 [初三上]
│   ├── 01-inverse-basics.md
│   ├── 02-inverse-properties.md               # ⭐ k 的几何意义
│   └── 03-inverse-applications.md
│
├── part10-quadratic-functions/                # 二次函数 [初三下]
│   ├── 01-quadratic-fn-basics.md
│   ├── 02-three-forms.md                      # ⭐ 一般式/顶点式/交点式
│   ├── 03-quadratic-fn-properties.md
│   ├── 04-finding-expression.md
│   ├── 05-quadratic-fn-applications.md
│   ├── 06-fn-and-equation.md                  # ⭐ 函数与方程的关系
│   ├── 07-parametric-quadratic-fn.md          # ⭐NEW 含参二次函数最值
│   └── 08-quadratic-fn-area.md                # ⭐NEW 二次函数中三角形面积（铅垂高法）
│
├── part11-statistics/                         # 统计 [初一上 + 初二上 + 初三上]
│   ├── 01-data-collection.md                  # 数据收集与描述
│   ├── 02-statistical-charts.md               # 条形/折线/扇形/直方图
│   ├── 03-central-tendency.md                 # 平均数/中位数/众数
│   └── 04-variance.md                         # 方差与样本估计
│
├── part12-probability/                        # 概率 [初一下 + 初三上]
│   ├── 01-possibility-and-events.md           # 可能性与事件
│   ├── 02-counting-and-tree.md                # ⭐ 列举法与树状图求概率
│   └── 03-frequency-and-probability.md        # 用频率估计概率
│
├── part13-applications/                       # 应用题专题 [初三复习]
│   ├── 01-setup-strategies.md                 # ⭐ 设元的策略（贯穿全部）
│   ├── 02-distance-problems.md                # ⭐ 行程问题模型
│   ├── 03-work-problems.md                    # ⭐ 工程问题模型
│   ├── 04-profit-problems.md                  # ⭐ 利润/折扣/利率模型
│   ├── 05-scheme-design.md                    # ⭐ 方案设计与最优化
│   ├── 06-mixture-problems.md                 # ⭐ 配比/浓度/调配模型
│   ├── 07-rate-of-change.md                   # ⭐NEW 增长率模型
│   └── 08-allocation.md                       # ⭐NEW 资源分配模型
│
├── part14-comprehensive/                      # 中考代数综合 [初三复习]
│   ├── 01-functions-comprehensive.md          # ⭐ 多种函数综合
│   ├── 02-quadratic-fn-with-geometry.md       # ⭐ 二次函数与几何综合
│   ├── 03-existence-in-functions.md           # ⭐ 函数中的存在性问题
│   ├── 04-moving-points-on-graph.md           # ⭐ 函数图象上的动点
│   ├── 05-new-definition.md                   # 新定义型代数题
│   └── 06-piecewise-and-absolute.md           # 含绝对值/分段题型
│
├── figures/                                   # 图渲染基础设施（拷贝自几何教程）
│   ├── render.sh
│   ├── STYLE.md                               # 含代数补充约定
│   ├── src/{tikz,asy}/
│   └── svg/
│
└── appendix/
    ├── A-formula-sheet.md                     # 公式速查
    ├── B-model-atlas.md                       # 解题套路图集
    ├── C-exercises-basic.md                   # 基础题库 60 题（☆）
    ├── D-exercises-medium.md                  # 中档题库 80 题（☆☆）
    ├── E-exercises-advanced.md                # 压轴题库 40 题（☆☆☆）
    ├── F1-solutions-numbers-and-expressions.md   # 详解 part1-4 部分
    ├── F2-solutions-equations-and-inequalities.md # 详解 part5-7 部分
    ├── F3-solutions-functions.md              # 详解 part8-10 部分
    └── F4-solutions-stat-prob-apps-comp.md    # 详解 part11-14 部分
```

**规模估算**：~95 章正文 + 10 篇 toolkit + 9 个 appendix 文件 + README + preface = **约 116 个 markdown 文件**。

**⭐ 标记 32 个模板 D 章节**（v1 是 13 个），覆盖代数全部核心套路。

---

## 3. 章节内部模板

### 模板 C（概念/运算法则类）—— 6 节

1. **一、概念特征**：怎么一眼认出
2. **二、定义与运算法则**
3. **三、为什么这么算（推导）**
4. **四、典型应用**（2-3 例题，每题带【思路】）
5. **五、易错点 & 反例**
6. **六、思路自测题**（3-5 题，仅给提示 `💡 提示：...`）

### 模板 D（技巧/套路类）—— 7 节（v2 加厚）

1. **一例速记**（一道极简样板题代替几何里的"一图速记"）
2. **一、引入：一道让你卡住的题**
3. **二、思维路径还原**（解题者的内心独白，用 `>` 引用块，**12-18 行**，体现"看到 X 想到 Y"的反射弧；v2 比 v1 加长）
4. **三、抽象成方法**：什么时候用 / 操作步骤 / 为什么有效
5. **四、方法变形**：一般化 / 组合应用 / 退化情形
6. **五、思考路标**（**至少 6 条**，看到 X → 想到 Y；v2 加密）
7. **六、应用例题**（2-3 题）+ **七、思路自测题**（4 题）

**关键调整**：代数里"一图速记"改为"一例速记"——样板题（如 $x^2 + 6x + 5 = (x+3)^2 - 4$ 体现配方）替代图。函数与几何综合章仍保留几何图。

**字数**：模板 D **3000-4000 字**，模板 C 1800-2500 字。

---

## 4. 图、公式、约定

### 4.1 图分类

| 类型 | 用途 | 工具 |
|---|---|---|
| 函数图象 | 一次/反比例/二次函数图象、含参动态 | Asymptote `graph` 模块 / TikZ + pgfplots |
| 数轴 | 不等式解集、实数比较 | TikZ |
| 坐标几何图 | 函数与几何综合、动点 | TikZ / Asymptote |
| 统计图表 | 条形/折线/扇形/直方图/茎叶图 | TikZ |
| 树状图 / 表格 | 列举法、概率树 | TikZ |
| 应用题示意 | 行程/工程/面积优化示意 | TikZ |
| 表达式排版 | 十字相乘、配方步骤、解题对照 | markdown 表 + LaTeX `align` |

### 4.2 基础设施

- `chuzhong_algebra/figures/render.sh` —— 拷贝自 `chuzhong_geometry/figures/render.sh`
- `chuzhong_algebra/figures/STYLE.md` —— 基于几何 STYLE.md，新增"函数图象坐标轴/网格/关键点标记约定"、"统计图表着色约定"、"概率树状图布局约定"等

### 4.3 LaTeX 约定

- 行内 `$...$`、行间 `$$...$$`
- 多行对齐 `\begin{align*}...\end{align*}`、`\begin{aligned}...\end{aligned}`
- 集合 `\{x \mid x > 0\}`、区间 `(-\infty, 2)`、`[1, 3)`
- 函数 $f(x)$、$y = ax^2 + bx + c$
- 判别式 $\Delta = b^2 - 4ac$
- 韦达定理 $x_1 + x_2 = -\frac{b}{a}$、$x_1 x_2 = \frac{c}{a}$
- 统计：$\bar{x}$（平均数）、$S^2$（方差）
- 概率：$P(A) = \frac{n_A}{n}$

---

## 5. 练习题系统（v2 扩容）

### 5.1 节内 / 节末

- **节内例题** 2-4 道：完整解答 + 【思路】
- **节末思路自测题** 3-5 道：仅给提示

### 5.2 附录题库 180 题

- C 基础（☆）**60 题**：覆盖各 part 基本运算与概念
- D 中档（☆☆）**80 题**：典型综合
- E 压轴（☆☆☆）**40 题**：函数综合 15 + 应用题压轴 10 + 代几综合 10 + 新定义/创新 5
- **共 180 题**

### 5.3 详解 F 拆分（4 个文件，避免单文件过长）

- `F1-solutions-numbers-and-expressions.md`（part1-4 范围题）
- `F2-solutions-equations-and-inequalities.md`（part5-7）
- `F3-solutions-functions.md`（part8-10）
- `F4-solutions-stat-prob-apps-comp.md`（part11-14）

每题详解结构：题号 + **模型/思路标签**（如 `[配方法 + 韦达]`、`[设元 + 行程模型]`、`[列举法 + 树状图]`）+ 解答 + 答案。

---

## 6. 实施路线（按 part 分里程碑）

| 里程碑 | 内容 | 文件数 |
|---|---|---|
| **AM0** | README + preface + thinking-toolkit 10 篇 + figures 基础设施 | 13 |
| **AM1** | part1 数（7 章） + part2 整式（7 章） | 14 |
| **AM2** | part3 因式分解（5 章） + part4 分式（4 章） | 9 |
| **AM3** | part5 一次方程（5 章） + part7 不等式（5 章） | 10 |
| **AM4** | part6 一元二次方程（6 章） | 6 |
| **AM5** | part8 一次函数（6 章） + part9 反比例（3 章） | 9 |
| **AM6** | **part10 二次函数（8 章）** 最重 | 8 |
| **AM7** | part11 统计（4 章） + part12 概率（3 章） | 7 |
| **AM8** | **part13 应用题专题（8 章）** | 8 |
| **AM9** | part14 中考代数综合（6 章） | 6 |
| **AM10** | appendix（公式 + 套路图集 + 180 题 + 4 个详解文件） | 9 |
| **AM11** | 全教程配图（分批，仿几何 F1-F9，约 250 张） | ~250 张 |

总计 **~116 个 md + ~250 张图**。每个里程碑独立实施计划与执行。

---

## 7. 不在本设计范围内（YAGNI）

- 不做交互式 web 版
- 不做视频/动画
- 不做习题自动判分
- 不覆盖高中代数内容
- 不做英文版
- 不专门做"综合与实践"项目式跨学科专题（部分场景在 part13 应用题里覆盖）
- 反证法不单独成章（穿插在分式方程"否定法"、二次方程"无实根判别"等需要处）

---

## 8. 成功标准

- 模板 D 章节能让学生看到"$x^2 + px + q$ 立刻判别用配方/十字/求根"等结构识别反射
- thinking-toolkit 10 篇能作为独立"代数方法论"小册子单独成立
- part13 应用题专题覆盖近 5 年陕西中考主流应用题型，附 8 种模型骨架
- part14 中考代数综合覆盖近 5 年陕西中考代数压轴题型
- appendix 180 题库每题都能反查到对应方法/模型标签
- 与 `chuzhong_geometry/` 形成"几何 + 代数 + 统计概率"完整组合，覆盖中考数学四大板块（综合与实践穿插）
