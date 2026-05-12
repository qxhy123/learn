# 中考代数教程 设计文档

**创建日期**：2026-05-12
**目标目录**：`chuzhong_algebra/`（与 `chuzhong_geometry/` 同级，在 `/Users/yangyang/ai_projs/math/`）
**教材基准**：北师大版 2024 新课标 / 陕西适用
**对照参考**：`chuzhong_geometry/`（已完成的中考几何教程，相同风格与基础设施）
**状态**：设计已确认，待生成实施计划

---

## 1. 定位与风格

- **读者**：综合型（零基础到中考冲刺）
- **核心特色**：与几何教程一致——**思路培养**优先，模板 D 类章节含"思维路径还原 + 思考路标"
- **风格**：纯 markdown + LaTeX，TikZ/Asymptote 图（函数图象为主、数轴 + 坐标几何 + 应用示意图为辅），节内例题 + 节末自测 + 附录题库三层结构
- **基础设施复用**：拷贝 `chuzhong_geometry/figures/render.sh` 与 `STYLE.md`，新增代数补充约定（函数图象、数轴等）

---

## 2. 整体目录结构（12 part + 工具箱 + 附录）

```
chuzhong_algebra/
├── README.md                                  # 总入口（含年级标签）
├── 00-preface.md                              # 序言：什么是"代数思维"
│
├── thinking-toolkit/                          # 代数思维工具箱（6 篇方法论）
│   ├── 01-when-to-substitute.md               # 何时换元
│   ├── 02-when-to-complete-square.md          # 何时配方
│   ├── 03-discriminant-and-vieta.md           # 判别式 + 韦达定理思想
│   ├── 04-setting-up-equations.md             # 设元的艺术
│   ├── 05-integral-thinking.md                # 整体思想
│   └── 06-checklist-before-solving.md         # 代数题开题清单
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
│   └── 06-divide.md
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
│   └── 04-fraction-techniques.md              # ⭐ 分式化简技巧
│
├── part5-linear-eq/                           # 一次方程（组） [初一上 + 初二上]
│   ├── 01-linear-one-variable.md
│   ├── 02-linear-two-variable.md
│   ├── 03-elimination-methods.md
│   └── 04-linear-eq-tricks.md                 # ⭐ 解一次方程组技巧
│
├── part6-quadratic-eq/                        # 一元二次方程 [初三上]
│   ├── 01-quadratic-basics.md
│   ├── 02-direct-and-factoring.md
│   ├── 03-completing-square.md                # ⭐ 配方法
│   ├── 04-quadratic-formula.md                # 求根公式与判别式
│   ├── 05-vieta-theorem.md                    # ⭐ 韦达定理
│   └── 06-quadratic-applications.md
│
├── part7-inequalities/                        # 不等式 [初二下]
│   ├── 01-inequality-basics.md
│   ├── 02-solve-linear-inequality.md
│   ├── 03-inequality-system.md
│   └── 04-inequality-applications.md
│
├── part8-linear-functions/                    # 一次函数 [初二上]
│   ├── 01-function-concept.md
│   ├── 02-linear-function-basics.md
│   ├── 03-finding-expression.md               # ⭐ 待定系数法
│   ├── 04-linear-function-applications.md
│   └── 05-system-and-line.md                  # ⭐ 与方程组的几何意义
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
│   └── 06-fn-and-equation.md                  # ⭐ 函数与方程的关系
│
├── part11-applications/                       # 应用题专题 [初三复习]
│   ├── 01-setup-strategies.md                 # ⭐ 设元的策略
│   ├── 02-distance-problems.md                # 行程模型
│   ├── 03-work-problems.md                    # 工程模型
│   ├── 04-profit-problems.md                  # 利润/利率
│   ├── 05-scheme-design.md                    # 方案设计与最优化
│   └── 06-mixture-problems.md                 # 配比/浓度
│
├── part12-comprehensive/                      # 中考代数综合 [初三复习]
│   ├── 01-functions-comprehensive.md
│   ├── 02-quadratic-fn-with-geometry.md       # ⭐ 二次函数与几何综合
│   ├── 03-existence-in-functions.md
│   ├── 04-moving-points-on-graph.md
│   ├── 05-new-definition.md
│   └── 06-piecewise-and-absolute.md
│
├── figures/                                   # 图渲染基础设施（拷贝自几何教程）
│   ├── render.sh
│   ├── STYLE.md                               # 含代数补充约定（函数图、数轴等）
│   ├── src/{tikz,asy}/
│   └── svg/
│
└── appendix/
    ├── A-formula-sheet.md                     # 公式速查
    ├── B-model-atlas.md                       # 解题套路图集
    ├── C-exercises-basic.md                   # 基础题库 40 题（☆）
    ├── D-exercises-medium.md                  # 中档题库 40 题（☆☆）
    ├── E-exercises-advanced.md                # 压轴题库 20 题（☆☆☆）
    └── F-solutions.md                         # 100 题详解
```

**规模估算**：~75 章正文 + 6 篇 toolkit + 6 篇 appendix = **约 87 个 markdown 文件**。⭐ 标记 13 个为模板 D（技巧/套路类章节）。

---

## 3. 章节内部模板

### 模板 C（概念/运算法则类）—— 6 节

1. **一、概念特征**：怎么一眼认出（如"分母含未知数 → 分式方程"、"$ax^2+bx+c=0\,(a\ne0)$ → 一元二次方程"）
2. **二、定义与运算法则**
3. **三、为什么这么算（推导/证明）**：突出概念的来源
4. **四、典型应用**（2-3 例题，每题带【思路】）
5. **五、易错点 & 反例**
6. **六、思路自测题**（3-5 题，仅给提示 `💡 提示：...`）

### 模板 D（技巧/套路类）—— 7 节

1. **一例速记**（用一道极简样板题代替几何里的"一图速记"）
2. **一、引入：一道让你卡住的题**
3. **二、思维路径还原**（解题者的内心独白；用 `>` 引用块，10-15 行）—— **核心特色**
4. **三、抽象成方法**：什么时候用 / 操作步骤 / 为什么有效
5. **四、方法变形**：一般化 / 组合应用 / 退化情形
6. **五、思考路标**（看到 X → 想到 Y）—— **核心特色**
7. **六、应用例题**（2-3 题）+ **七、思路自测题**

**与几何模板 D 的差异**：代数模型大多没有图（直接是代数式变形），故把"一图速记"改为"一例速记"——用样板题（如 $x^2 + 6x + 5 = (x+3)^2 - 4$）代替速记图。函数与几何综合章节仍可保留图。

---

## 4. 图、公式、约定

### 4.1 图分类

| 类型 | 用途 | 工具 |
|---|---|---|
| 函数图象 | 一次/反比例/二次函数图象、含参函数动态 | Asymptote (`graph` 模块) 或 TikZ + pgfplots |
| 数轴 | 不等式解集、实数比较 | TikZ |
| 坐标几何图 | 函数与几何综合、动点 | TikZ / Asymptote |
| 应用题示意 | 行程 / 工程 / 面积优化示意 | TikZ |
| 表达式排版 | 十字相乘、配方步骤、解题对照 | markdown 表 + LaTeX `align` |

### 4.2 基础设施复用

- `chuzhong_algebra/figures/render.sh` —— 直接拷贝 `chuzhong_geometry/figures/render.sh`
- `chuzhong_algebra/figures/STYLE.md` —— 基于几何 STYLE.md，加入代数补充约定
- 命名：`thm-<topic>.{tex,asy}`、`ex-<partN>-<chapter>-<n>.{tex,asy}`、`q-<level>-NN.{tex,asy}`

### 4.3 LaTeX 约定

- 行内 `$...$`，行间 `$$...$$`
- 多行对齐 `\begin{align*}...\end{align*}` 或 `\begin{aligned}...\end{aligned}`
- 集合 `\{x \mid x > 0\}`
- 区间 `(-\infty, 2)`, `[1, 3)`
- 函数 $f(x)$、$y = ax^2 + bx + c$
- 判别式 $\Delta = b^2 - 4ac$
- 韦达定理 $x_1 + x_2 = -\frac{b}{a}$、$x_1 x_2 = \frac{c}{a}$

---

## 5. 练习题系统

- **节内例题** 2-4 道：完整解答 + 【思路】
- **节末"思路自测题"** 3-5 道：仅给提示 `💡 提示：...`
- **附录三档题库**：
  - C 基础（☆）40 题
  - D 中档（☆☆）40 题
  - E 压轴（☆☆☆）20 题
  - F 详解 100 题
- **代数题图配比**预计：基础 10-15 张、中档 20-25 张、压轴 18-20 张。多数代数题（化简、解方程、推理）不需图。

---

## 6. 实施路线（按 part 分里程碑）

| 里程碑 | 内容 | 文件数 |
|---|---|---|
| AM0 | README + preface + thinking-toolkit 6 篇 + figures 基础设施 | 9 |
| AM1 | part1 数（7 章） + part2 整式（6 章） | 13 |
| AM2 | part3 因式分解（5 章） + part4 分式（4 章） | 9 |
| AM3 | part5 一次方程（4 章） + part7 不等式（4 章） | 8 |
| AM4 | part6 一元二次方程（6 章） | 6 |
| AM5 | part8 一次函数（5 章） + part9 反比例（3 章） | 8 |
| AM6 | **part10 二次函数（6 章）** 最难、最重 | 6 |
| AM7 | part11 应用题专题（6 章） | 6 |
| AM8 | part12 中考代数综合（6 章） | 6 |
| AM9 | appendix（公式表 + 套路图集 + 100 题 + 详解） | 6 |
| AM10 | 全教程配图（分批，仿几何 F1-F9） | ~200 张图 |

总计 **~87 个 md 文件 + ~200 张图**。每个里程碑独立实施计划与执行。

---

## 7. 不在本设计范围内（YAGNI）

- 不做交互式 web 版（纯 markdown 即可）
- 不做视频/动画
- 不做习题自动判分
- 不覆盖高中代数内容（导数、向量、解析几何深入等）
- 不做英文版

---

## 8. 成功标准

- 任意一个章节，读者读完后能复述该章核心方法的"使用时机"
- 模板 D 章节能让学生看到"$x^2 + 6x$ 这种式子立刻想到配方"等反射弧
- part11 应用题专题能覆盖近 5 年陕西中考的所有典型应用题型
- appendix 题库每题都能反查到对应方法/模型标签
- 与 `chuzhong_geometry/` 形成"几何 + 代数"对称组合，复习时可配对使用
