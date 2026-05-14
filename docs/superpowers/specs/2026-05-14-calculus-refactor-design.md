# 微积分教程改造设计文档

**创建日期**：2026-05-14
**目标目录**：`calculus-tutorial/`（保留现有目录结构）
**改造方案**：**A2（保留 + 包装）** —— 保留现有大学教材正文，加上模板 D 化前后置 + thinking-toolkit + 三档题库 + figures
**对照参考**：`gaozhong_math/algebra/`、`gaozhong_math/geometry/`（已完成的高中数学全套）
**状态**：设计已确认，待生成实施计划

---

## 0. 设计要点

参照 `gaozhong_math/algebra/` 的成功模式做"包装式改造"：
- 现有 28 章正文 **不丢弃**（含严格推导 + 深度学习应用）
- 加上**模板 D 化前后置**：一例速记 / 思维路径还原 / 思考路标 / 易错点
- 新增 thinking-toolkit 12 篇（微积分思维方法论）
- 新增 C/D/E 三档题库 240 题 + F 详解
- 新增 figures ~80 张（混合模式：核心章节随章）
- 复用 `~/.claude/skills/rendering-math-figures/` 渲图基础设施

**与初高中的关键差异**：
- 微积分章节信息密度高得多（每章原本 400-940 行），不重写正文，仅"包装"
- 章节字数范围调高：模板 C 章节包装后 ~3500-5500 字；模板 D 章节 ~5500-8000 字
- thinking-toolkit 主题完全针对微积分思维（极限 / 求导 / 积分 / 级数 / 多元 / ODE / AI）
- 第 12 篇 toolkit 直接对接 AI 应用（呼应教程 AI 工程定位）

---

## 1. 定位与风格

- **读者**：从首学微积分的高中生 / 大学新生，到备战考研 / 巩固基础的 AI 工程师
- **核心特色**：
  - **严格推导 + AI 应用并重**（保留现有强项）
  - **"看到 X 立刻想 Y"思维训练**（新增模板 D 价值）
  - **完整套路图集**（套路化求导 / 积分 / 判敛 / ODE 求解）

---

## 2. 范围切分

### `calculus-tutorial/` 涵盖（保留现有 8 part）：

- Part 1 预备（集合 / 函数 / 三角 / 对数指数 / 线性仿射）
- Part 2 极限连续
- Part 3 微分学
- Part 4 积分学
- Part 5 级数
- Part 6 多元微积分
- Part 7 常微分方程
- Part 8 AI 微积分（凸优化 / 矩阵微积分 / 概率中的微积分 / SDE）

### 不在本设计范围内（YAGNI）

- 不涉及实分析 / 测度论 / 泛函等更高数学
- 不做偏微分方程 / 复变函数
- 不做互动 web 版 / 视频
- 不做英文版

---

## 3. 章节内部模板（保留 + 包装）

### 现有结构（保留不动）

```
学习目标
N 个小节正文（含严格推导）
本章小结
深度学习应用
练习题
练习答案
```

### 包装后结构（A2 方案）

```
# 第 X 章 标题

> 一例速记（⭐ 章节必有，C 章节可选）
> 引入题 + 思维路径还原 (15-20 行 > 引用块) （仅 ⭐ 章节）

## 学习目标
## X.1 ... X.N 现有小节正文（不动）
## 本章小结
## 深度学习应用

---

## 思考路标（条件反射）
（≥8 条；包装后**新增**）

## 易错点
（5 条；包装后**新增**）

## 练习题 + 答案（保留）
```

**字数变化**：
- 模板 C 章节：原 400-700 行 → 包装后 500-900 行
- 模板 D 章节：原 600-940 行 → 包装后 800-1300 行
- 全部章节均保留现有严格推导

### ⭐ 模板 D 章节（15 个）

| Part | 章 | 标题 |
|---|---|---|
| 2 | 5 | 函数极限（$\varepsilon$-$\delta$）|
| 3 | 8 | 求导法则 |
| 3 | 9 | 导数应用（L'Hospital）|
| 3 | 10 | Taylor 级数 |
| 4 | 13 | 积分技巧 |
| 5 | 15 | 数项级数 |
| 5 | 16 | 幂级数 |
| 6 | 18 | 偏导数 |
| 6 | 19 | 重积分 |
| 6 | 22 | 向量微积分 |
| 7 | 23 | 一阶 ODE |
| 7 | 24 | 二阶 ODE |
| 8 | 25 | 凸优化 |
| 8 | 26 | 矩阵微积分 |

**其余 13 章按模板 C 包装**（仅加思考路标 + 易错点，无一例速记 / 思维路径还原）。

---

## 4. thinking-toolkit 12 篇（新增）

```
calculus-tutorial/thinking-toolkit/
├── 01-epsilon-language.md          # 极限的 ε 语言
├── 02-equivalent-infinitesimals.md # 等价无穷小与小 o
├── 03-differentiation-rules.md     # 求导套路系统化
├── 04-integration-techniques.md    # 积分技巧反 LIATE
├── 05-series-convergence.md        # 级数判敛流程图
├── 06-taylor-and-error.md          # Taylor 展开与误差
├── 07-multivar-chain.md            # 多元链式与梯度
├── 08-multivar-integration.md      # 多元积分变换
├── 09-ode-classification.md        # ODE 类型识别
├── 10-convexity-extrema.md         # 凸性、单调与极值
├── 11-inequality-techniques.md     # 微积分中的不等式
└── 12-calculus-for-ai.md           # 微积分中的 AI 思维
```

每篇 3000-4500 字。结构与初高中 toolkit 一致：
- 引言
- N 类方法 / 思路
- 演示题（`>` 引用块内心独白）
- 思考路标 ≥6 条
- 典型应用 3 例
- 自测题 4-5 题

---

## 5. 图、公式、约定

### 图配置策略（混合模式）

**目标 80 张 SVG**：

| Part | 张数 | 重点 |
|---|---|---|
| 1 预备 | 5 | 函数图象、对数指数对比 |
| 2 极限 | 8 | $\varepsilon$-$\delta$、序列收敛、间断点 |
| 3 微分 | 10 | 切线、Taylor 截断、单调极值 |
| 4 积分 | 10 | 曲边梯形、定积分几何、换元 |
| 5 级数 | 8 | 判敛流程、收敛域、Fourier 部分和 |
| 6 多元 | 15 | 3D 偏导、二重积分区域、坐标变换、Jacobian |
| 7 ODE | 8 | 方向场、解曲线、振动响应 |
| 8 AI | 10 | 梯度下降、Hessian 椭球、KL 散度、反向传播 |
| 综合 | 6 | 套路速查 |
| **合计** | **80** | |

### 命名约定

`calc-pX-YY-Z.{tex,asy}` 表示 `calculus` Part X 第 YY 章第 Z 张图。

### 渲染工具链

- 复用 `gaozhong_math/algebra/figures/render.sh`（xelatex + asy + pdf2svg）
- 2D 用 TikZ + pgfplots
- 3D 用 Asymptote（多元微积分章节高频）
- 等距投影（多元几何）TikZ 备用

---

## 6. 练习题系统（保留 + 重组）

### 保留

- **每章末习题保留**（作为学习节奏的一部分，与初高中"自测题"角色相同）
- 现有 `appendix/formula-sheet.md` 保留（扩展）
- 现有 `appendix/notation-guide.md` 保留

### 重组 / 新增

- **C 80 + D 100 + E 60 = 240 题总题库** 新建在 `appendix/`
- **F 详解** 按主题分 4 文件
- 原 `appendix/kaoyan-problems.md` 内容拆入 D / E（不再独立保留）
- 原 `appendix/answers.md` 内容拆入各章自身 + F 详解

### 240 题分布

| 分组 | 章节范围 | C | D | E | 小计 |
|---|---|---|---|---|---|
| 极限连续 | Ch.4-6 | 10 | 12 | 6 | 28 |
| 微分应用 | Ch.7-10 | 15 | 18 | 12 | 45 |
| 积分技巧 | Ch.11-14 | 15 | 18 | 10 | 43 |
| 级数 | Ch.15-17 | 10 | 12 | 8 | 30 |
| 多元微积分 | Ch.18-22 | 15 | 20 | 12 | 47 |
| ODE | Ch.23-24 | 8 | 10 | 4 | 22 |
| AI 微积分 | Ch.25-28 | 7 | 10 | 8 | 25 |
| **合计** | | **80** | **100** | **60** | **240** |

### F 详解分文件

- F1-solutions-limits-differentiation.md（极限 + 微分）
- F2-solutions-integration-series.md（积分 + 级数）
- F3-solutions-multivar-ode.md（多元 + ODE）
- F4-solutions-ai-calculus.md（AI 微积分）

---

## 7. 整体目录结构（改造后）

```
calculus-tutorial/
├── README.md                              # 扩展（加 toolkit + appendix 导航）
├── 00-preface.md                          # 保留 + 加"模板 D 价值"段
│
├── thinking-toolkit/                      # 新增（12 篇）
│   ├── 01-epsilon-language.md
│   ├── 02-equivalent-infinitesimals.md
│   ├── ... (共 12 篇)
│   └── 12-calculus-for-ai.md
│
├── part1-foundations/                     # 6 章（包装）
├── part2-limits/                          # 3 章（含 ⭐ Ch.5）
├── part3-differentiation/                 # 4 章（含 3 ⭐）
├── part4-integration/                     # 4 章（含 ⭐ Ch.13）
├── part5-series/                          # 3 章（含 2 ⭐）
├── part6-multivariable/                   # 5 章（含 3 ⭐）
├── part7-ode/                             # 2 章（全 ⭐）
├── part8-ai-calculus/                     # 4 章（含 2 ⭐）
│
├── figures/                               # 新增基建 + 80 张图
│   ├── render.sh
│   ├── STYLE.md
│   ├── src/{tikz,asy}/
│   └── svg/
│
└── appendix/
    ├── A-formula-sheet.md                 # 扩展自现有 formula-sheet.md
    ├── B-model-atlas.md                   # 新增（约 30 个套路模型）
    ├── C-exercises-basic.md               # 新增（80 题）
    ├── D-exercises-medium.md              # 新增（100 题）
    ├── E-exercises-advanced.md            # 新增（60 题）
    ├── F1-solutions-limits-differentiation.md
    ├── F2-solutions-integration-series.md
    ├── F3-solutions-multivar-ode.md
    ├── F4-solutions-ai-calculus.md
    └── notation-guide.md                  # 保留
```

**规模估算**：28 章正文 + 12 toolkit + README + preface + 10 appendix = **52 个 md 文件** + ~80 SVG + 240 题。

---

## 8. 里程碑路线（11 个 CM）

| 里程碑 | 内容 | 文件数 |
|---|---|---|
| **CM0** | README + preface + thinking-toolkit 12 篇 + figures 基建 | 15 |
| **CM1** | Part 1 预备 6 章包装 + 5 图 | 6 章 |
| **CM2** | Part 2 极限 3 章（含 ⭐ Ch.5）+ 8 图 | 3 章 |
| **CM3** | Part 3 微分 4 章（含 3 ⭐）+ 10 图 | 4 章 |
| **CM4** | Part 4 积分 4 章（含 ⭐ Ch.13）+ 10 图 | 4 章 |
| **CM5** | Part 5 级数 3 章（含 2 ⭐）+ 8 图 | 3 章 |
| **CM6** | Part 6 多元 5 章（含 3 ⭐）+ 15 图 | 5 章 |
| **CM7** | Part 7 ODE 2 章（全 ⭐）+ 8 图 | 2 章 |
| **CM8** | Part 8 AI 微积分 4 章（含 2 ⭐）+ 10 图 | 4 章 |
| **CM9** | 全套 appendix（扩 A + 新 B + C 80 + D 100 + E 60） | 5 文件 |
| **CM10** | F 详解（按主题分 4 文件，240 题详解） | 4 文件 |

**总产出**：52 md + ~80 SVG + 240 题。

---

## 9. 不在本设计范围内（YAGNI）

- 不重写现有 28 章的核心正文（仅"包装"）
- 不加偏微分方程 / 复变函数 / 测度论
- 不做英文版
- 不做互动 web 版

---

## 10. 成功标准

- 每个 ⭐ 章节加上"一例速记"和"思维路径还原"后，能让读者"看到 X 立刻想 Y"
- 12 篇 toolkit 可作为独立的"微积分方法论"小册子
- 240 题三档分明、与考研 + AI 应用场景对齐
- 与 `gaozhong_math/algebra/` `gaozhong_math/geometry/` 配合，构成"初中 → 高中 → 微积分（含 AI）" 完整数学体系
- 保留现有教程的所有严格推导 + 深度学习应用价值（不丢失）
