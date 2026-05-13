# 高中几何教程 设计文档

**创建日期**：2026-05-13
**目标目录**：`gaozhong_math/geometry/`（与 `gaozhong_math/algebra/` 平行）
**教材基准**：人教 A 版 2019 新课标（必修第二册 + 选择性必修一）
**对照参考**：
- `gaozhong_math/algebra/`（刚完成的高中代数教程，84 章 + 240 题 + 40 SVG）
- `chuzhong_geometry/`（已完成的中考几何教程）
**节奏**：**先做种子（Part 1, 2, 3, 4, 8）**，约 25-30 章，看效果再扩到全套
**状态**：设计已确认，待生成实施计划

---

## 0. 设计要点

照搬 `gaozhong_math/algebra/` 的成功模式：
- 两种章节模板 C / D（模板 D 含"思维路径还原 + 思考路标"）
- thinking-toolkit 思维方法论独立小册子（**重写为几何思维**）
- appendix（公式表 + 模型图集 + 三档题库 + 详解）
- TikZ/Asymptote + xelatex + ctex 渲图基础设施（复用 `rendering-math-figures` skill）
- 按里程碑分批执行，每批 subagent 并行

**与代数的关键差异**：
- 几何是**图密集型**：核心章节随章配图（向量 / 圆锥曲线 / 立体 / 空间向量）
- 几何思维差异大：综合法 / 解析法 / 向量法 / 三角法的切换是核心能力
- 模板字数稍高：C 2500-3500，D 4000-5500（几何推导链更长）
- thinking-toolkit 全部重写为几何思维（不复用代数版）

---

## 1. 定位与风格

- **读者**：高中学生（高一下 → 高三）、高考备考者、自学高中几何的成年读者
- **核心特色**：**几何思维四把剑（综合 / 解析 / 向量 / 三角）的切换 + 数形结合深化**
- **基础设施**：复用 `~/.claude/skills/rendering-math-figures/templates/` 中的 render.sh、STYLE.md

---

## 2. 范围切分

`gaozhong_math/geometry/` 涵盖：
- 平面向量（基础 + 应用）
- 解析几何（直线、圆、椭圆、双曲线、抛物线）
- 立体几何初步（点线面、表面积体积）
- 空间向量与立体几何（建系法）
- 高考几何综合

**不涵盖**（属代数 / 不属高中）：
- 函数 / 导数 / 数列 / 概率统计（已在 `gaozhong_math/algebra/`）
- 高等几何（黎曼 / 微分几何）
- 数学竞赛专题（IMO 几何）

---

## 3. 整体目录结构（10 part + thinking-toolkit + appendix）

```
gaozhong_math/geometry/
├── README.md
├── 00-preface.md
│
├── thinking-toolkit/                       # 几何思维工具箱（12 篇）
│   ├── 01-four-swords.md                  # 综合 / 解析 / 向量 / 三角四把剑
│   ├── 02-auxiliary-lines.md              # 辅助线思维
│   ├── 03-number-shape-deep-v2.md         # 数形结合（几何深化）
│   ├── 04-parametrization.md              # 参数化思想
│   ├── 05-invariants-and-symmetry.md      # 不变量与对称
│   ├── 06-position-classification.md      # 位置关系的分类讨论
│   ├── 07-geometric-extrema.md            # 几何最值 5 种方法
│   ├── 08-vector-two-faces.md             # 向量的两副面孔
│   ├── 09-spatial-imagination.md          # 空间几何想象力
│   ├── 10-line-conic-synthesis.md         # 直线圆锥曲线综合
│   ├── 11-vector-in-solid.md              # 立体几何向量法
│   └── 12-checklist-before-solving.md     # 几何开题清单
│
├── part1-plane-vectors-basics/             # ⭐ 种子 [高一下]
│   ├── 01-vector-concept.md                # 向量定义、相等、共线
│   ├── 02-vector-addition.md               # 加减法（三角形 / 平行四边形）
│   ├── 03-scalar-multiplication.md         # 数乘 + 共线定理
│   └── 04-basis-and-coordinates.md         # ⭐ 基底、平面向量基本定理、坐标
│
├── part2-plane-vectors-applications/       # ⭐ 种子 [高一下]
│   ├── 01-dot-product.md                   # ⭐ 数量积 + 夹角 / 投影
│   ├── 02-vector-in-triangle.md            # ⭐ 向量解三角形 + 三点共线 + 重心
│   ├── 03-vector-in-line-circle.md         # 向量与直线 / 圆的关系
│   └── 04-vector-applications.md           # 物理 / 综合应用
│
├── part3-line-and-equation/                # ⭐ 种子 [高二上]
│   ├── 01-slope-and-inclination.md         # 斜率与倾斜角
│   ├── 02-line-equation-forms.md           # 5 种方程形式（点斜 / 斜截 / 两点 / 截距 / 一般）
│   ├── 03-line-position-relations.md       # ⭐ 平行 / 垂直 / 重合
│   ├── 04-distance-formulas.md             # ⭐ 点到直线 / 两平行线距离
│   └── 05-line-applications.md             # 含参讨论 + 综合
│
├── part4-circle-and-equation/              # ⭐ 种子 [高二上]
│   ├── 01-circle-standard-equation.md      # 标准方程 + 一般方程
│   ├── 02-line-circle-position.md          # ⭐ 直线与圆位置关系 + 弦长 + 切线
│   ├── 03-circle-circle-position.md        # 圆与圆位置关系
│   └── 04-circle-applications.md           # 含参圆 + 综合
│
├── part5-ellipse/                          # [高二上 / 扩展]
│   ├── 01-ellipse-definition-equation.md   # 定义 + 标准方程
│   ├── 02-ellipse-properties.md            # ⭐ 性质 + 离心率
│   ├── 03-line-ellipse.md                  # ⭐ 直线与椭圆位置 + 弦长 + 中点弦
│   └── 04-ellipse-applications.md          # 含参 + 综合 + 实际应用
│
├── part6-hyperbola/                        # [高二上 / 扩展]
│   ├── 01-hyperbola-definition-equation.md # 定义 + 标准方程
│   ├── 02-hyperbola-properties.md          # ⭐ 性质 + 渐近线 + 离心率
│   ├── 03-line-hyperbola.md                # ⭐ 直线与双曲线位置（含 $\Delta$ 与单交点情形）
│   └── 04-hyperbola-applications.md        # 综合
│
├── part7-parabola/                         # [高二上 / 扩展]
│   ├── 01-parabola-definition-equation.md  # 定义 + 标准方程
│   ├── 02-parabola-properties.md           # ⭐ 性质 + 焦点弦 + 准线
│   ├── 03-line-parabola.md                 # ⭐ 直线与抛物线 + 焦点弦性质
│   └── 04-parabola-applications.md         # 综合
│
├── part8-solid-geometry-basics/            # ⭐ 种子 [高一下]
│   ├── 01-points-lines-planes.md           # 点线面公理 + 三视图
│   ├── 02-spatial-line-position.md         # ⭐ 异面直线 + 夹角
│   ├── 03-line-plane-position.md           # ⭐ 直线与平面位置（平行 / 垂直 / 相交）
│   ├── 04-plane-plane-position.md          # ⭐ 二面角 + 平面与平面
│   ├── 05-solid-figures-area-volume.md     # 表面积 + 体积（棱柱 / 锥 / 球）
│   └── 06-solid-geometry-applications.md   # 综合应用
│
├── part9-spatial-vectors/                  # [高二上 / 扩展]
│   ├── 01-spatial-vector-basics.md         # 空间向量定义 + 加减 + 数乘
│   ├── 02-spatial-vector-dot-product.md    # 数量积
│   ├── 03-spatial-basis-coordinates.md     # ⭐ 空间基底 + 坐标系建立
│   ├── 04-spatial-line-plane-by-vector.md  # ⭐ 用向量证平行 / 垂直
│   ├── 05-spatial-angles-by-vector.md      # ⭐ 用向量求线线 / 线面 / 二面角
│   └── 06-spatial-distance-by-vector.md    # ⭐ 用向量求距离
│
├── part10-geometry-comprehensive/          # [高三复习 / 扩展，全 ⭐]
│   ├── 01-conic-comprehensive.md           # ⭐ 圆锥曲线综合（直线 + 曲线 + 韦达）
│   ├── 02-conic-with-parameter.md          # ⭐ 含参圆锥曲线
│   ├── 03-trajectory-and-locus.md          # ⭐ 轨迹方程问题
│   ├── 04-fixed-point-and-line.md          # ⭐ 定点 / 定值 / 定直线问题
│   ├── 05-solid-comprehensive.md           # ⭐ 立体几何综合（向量 + 综合）
│   └── 06-vector-comprehensive.md          # ⭐ 向量与代数 / 几何交叉
│
├── figures/
│   ├── render.sh
│   ├── STYLE.md
│   ├── src/{tikz,asy}/
│   └── svg/
│
└── appendix/
    ├── A-formula-sheet.md
    ├── B-model-atlas.md                    # 套路图集（约 40-50 模型）
    ├── C-exercises-basic.md                # 基础题库 80 题（☆）
    ├── D-exercises-medium.md               # 中档题库 100 题（☆☆）
    ├── E-exercises-advanced.md             # 高考压轴 60 题（☆☆☆）
    └── F1-F4-solutions.md                  # 240 题详解（按主题分 4 文件）
        # F1: 向量 / F2: 直线圆 / F3: 圆锥曲线 / F4: 立体几何 / 空间向量
```

**规模估算**：~50 章正文 + 12 篇 toolkit + 6 个 appendix = **约 80 个 md 文件**；约 **30-35 个 ⭐ 模板 D 章节**。

---

## 4. 章节内部模板

**与 `gaozhong_math/algebra/` 一致，字数微调**：

- **模板 C（概念/运算法则类，6 节）**：概念特征 / 定义与定理 / 推导证明 / 典型应用（带【思路】）/ 易错点 / 思路自测题。**字数 2500-3500**。
- **模板 D（套路/题型类，7 节）**：一例速记 / 引入 / 思维路径还原（`>` 引用块，15-20 行）/ 抽象成方法 / 方法变形 / 思考路标（≥8 条）/ 例题 + 自测。**字数 4000-5500**。

**与代数的关键差异**：
- 几何题述说更长 → 模板 C / D 字数微高
- 模板 D 的"思维路径还原"含图象 / 图示推理（用文字描述 + 配图引用）
- 例题中至少 1 道带"含参讨论"或"严格证明"或"分类讨论位置关系"

---

## 5. 图、公式、约定

### 图配置策略（混合模式 C）

- **核心章节随章配图**：向量加减 / 圆锥曲线 / 立体几何 / 空间向量等可视化必需的章节，每章 3-5 张图
- **纯概念章节后置**：定义、运算法则等图可有可无的章节，由对应里程碑统一补充
- **每个里程碑包含 5-15 张 figures**（视章节内容）

### 命名约定

`geo-pX-YY-Z.{tex,asy}` 表示 `gaozhong/geometry` Part X 第 YY 章第 Z 张图。

### 渲染工具链

- 完全复用 `~/.claude/skills/rendering-math-figures/` 中的 render.sh、STYLE.md
- 平面图 / 函数图 / 圆锥曲线用 TikZ + pgfplots
- 立体图 / 空间向量用 Asymptote
- 向量图 / 几何变换用 TikZ

### 全套规模

约 100-150 张 SVG。

---

## 6. 练习题系统（与代数对齐）

- 节内 2-3 例 + 节末 3-5 题"思路自测"
- 附录 **C 80 + D 100 + E 60 = 240 题**
- F 按主题分 4 个文件：
  - F1 向量（Part 1, 2, 9 题目）
  - F2 直线圆（Part 3, 4 题目）
  - F3 圆锥曲线（Part 5, 6, 7 题目）
  - F4 立体几何 / 空间向量 / 综合（Part 8, 9, 10 题目）

---

## 7. 里程碑路线

### 种子阶段（Part 1, 2, 3, 4, 8）

| 里程碑 | 内容 | 文件数 |
|---|---|---|
| **GM0** | README + preface + thinking-toolkit 12 篇 + figures 基础设施 | 15 |
| **GM1** | Part 1 平面向量基础（4 章）+ figures 5-8 张 | 4 章 |
| **GM2** | Part 2 平面向量应用（4 章）+ figures 5-8 张 | 4 章 |
| **GM3** | Part 3 直线与方程（5 章）+ figures 8-10 张 | 5 章 |
| **GM4** | Part 4 圆与方程（4 章）+ figures 6-8 张 | 4 章 |
| **GM5** | Part 8 立体几何初步（6 章）+ figures 10-15 张 | 6 章 |
| **GM6** | 种子阶段 appendix 子集（公式表 + 模型图集 + 100 题 + 详解）| 4 |

**种子产出**：约 **40-45 个 md 文件 + 约 50 张 SVG**，覆盖"高一下 + 高二上前半"几何。

**完成种子后决策点**：
- 看效果好 → 继续 GM7-GM12 做完圆锥曲线 + 空间向量 + 综合 + 全套 appendix
- 看效果一般 → 调整模板、字数、深度后再继续
- 暂停 → 种子也已是独立可用的"高一下 + 高二上几何入门指南"

### 全套阶段（种子通过后）

| 里程碑 | 内容 | 文件数 |
|---|---|---|
| GM7 | Part 5 椭圆（4 章）+ figures 10-12 张 | 4 章 |
| GM8 | Part 6 双曲线（4 章）+ Part 7 抛物线（4 章）+ figures 15-20 张 | 8 章 |
| GM9 | Part 9 空间向量（6 章）+ figures 8-12 张 | 6 章 |
| GM10 | Part 10 高考几何综合（6 章，全 ⭐）+ figures 10 张 | 6 章 |
| GM11 | 全套 appendix（补 D 中档 + E 提升 + F 全详解 240 题）| 5 |
| GM12 | 补足 figures（视需要补到 100-150 张）| ~50 张 |

**全套总计约 80 md + 100-150 SVG + 240 题 + 详解**。

---

## 8. 不在本设计范围内（YAGNI）

- 不做函数 / 导数 / 数列 / 概率（已在 `gaozhong_math/algebra/`）
- 不覆盖大学几何（射影 / 微分 / 黎曼）
- 不做互动 web 版 / 视频
- 不做习题自动判分
- 不做英文版

---

## 9. 成功标准

- 每个模板 D 章节能让学生"看到 X 立刻想 Y"（如"看到含焦点的椭圆题立刻想'第二定义 / 准线 / 椭圆参数方程'"）
- thinking-toolkit 12 篇能作为独立"高中几何方法论"小册子
- Part 10 综合章节能覆盖近 5 年高考几何压轴题的所有套路
- 240 题题库每题反查模型 / 思路标签
- 与 `gaozhong_math/algebra/` 配合使用，形成"高中代数 + 几何"完整高考备考体系
