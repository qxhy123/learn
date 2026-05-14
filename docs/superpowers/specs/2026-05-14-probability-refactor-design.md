# 概率论与数理统计教程改造设计文档

**创建日期**：2026-05-14
**目标目录**：`probability-statistics-tutorial/`（保留现有目录结构）
**改造方案**：**A2（保留 + 包装）** —— 复用 `calculus-tutorial/` 改造模式
**对照参考**：`calculus-tutorial/`（已完成的微积分改造，11 个 CM）
**状态**：设计已确认，待生成实施计划

---

## 0. 设计要点

参照 `calculus-tutorial/` 的成功"包装式改造"模式：
- 现有 24 章正文 **不丢弃**（含严格推导 + 深度学习应用）
- 加上**模板 D 化前后置**：一例速记 / 思维路径还原 / 思考路标 / 易错点
- 新增 thinking-toolkit 12 篇（概率论思维方法论）
- 新增 C/D/E 三档题库 240 题 + F 详解
- 新增 figures ~70 张（混合模式：核心章节随章）
- 复用 `~/.claude/skills/rendering-math-figures/` 渲图基础设施

**与 calculus 改造的差异**：
- 章节数较少（24 vs 28），里程碑数同（11 PM）
- toolkit 主题完全针对概率统计（贝叶斯 / MLE / 假设检验 / 信息论）
- 第 12 篇 toolkit 直接对接生成式 AI（VAE / 扩散模型 / 贝叶斯神经网络）
- 图数略少（70 vs 80）

---

## 1. 定位与风格

- **读者**：大学本科生 / 考研学生 / ML 工程师 / 数据科学家
- **核心特色**：
  - **严格概率论 + 现代 ML 应用并重**（保留现有强项）
  - **"看到 X 立刻想 Y"思维训练**（新增模板 D 价值）
  - **完整套路图集**（贝叶斯反演 / MLE / 假设检验 / 信息论）

---

## 2. 范围切分

### `probability-statistics-tutorial/` 涵盖（保留现有 8 part）：

- Part 1 概率基础
- Part 2 随机变量
- Part 3 分布
- Part 4 极限定理
- Part 5 统计基础
- Part 6 估计
- Part 7 假设检验
- Part 8 高级专题（信息论 / Monte Carlo / 概率图模型）

### 不在本设计范围内（YAGNI）

- 不涉及随机过程 / 测度论 / 随机控制论
- 不做时间序列 / 回归分析（属统计学习专题）
- 不做互动 web 版 / 视频
- 不做英文版

---

## 3. 章节内部模板（保留 + 包装）

### 现有结构（保留不动）

```
学习目标
正文内容（N 节）
本章小结
深度学习应用（代码示例）
练习题
练习答案
```

### 包装后结构（A2 方案）

```
# 第 X 章 标题

> 一例速记（⭐ 章节必有）
> 引入题 + 思维路径还原 (15-20 行 > 引用块) （仅 ⭐ 章节）

## 学习目标
## X.1 ... X.N 现有小节正文（不动）
## 本章小结
## 深度学习应用

---

## 几何示意（如有图）

## 思考路标（≥8 条）

## 易错点（5 条）

## 练习题 + 答案（保留）
```

**字数变化**：
- 模板 C 章节：原 600-900 行 → 包装后 700-1100 行
- 模板 D 章节：原 800-1200 行 → 包装后 1000-1500 行

### ⭐ 模板 D 章节（10 个）

| Part | 章 | 标题 |
|---|---|---|
| 1 | 2 | 条件概率与独立性（贝叶斯）|
| 2 | 6 | 多元随机变量 |
| 3 | 8 | 连续分布（正态 / Gamma / Beta）|
| 4 | 10 | 大数定律 |
| 4 | 11 | 中心极限定理 |
| 6 | 16 | 点估计（MLE / MoM）|
| 6 | 18 | 贝叶斯估计 |
| 7 | 19 | 假设检验基础 |
| 8 | 22 | 信息论（KL / 熵 / 互信息）|
| 8 | 24 | 概率图模型 |

**其余 14 章按模板 C 包装**（仅加思考路标 + 易错点，无一例速记 / 思维路径还原）。

---

## 4. thinking-toolkit 12 篇（新增）

```
probability-statistics-tutorial/thinking-toolkit/
├── 01-probability-axioms.md         # 概率公理与样本空间
├── 02-conditional-bayes.md          # 条件概率与贝叶斯思维
├── 03-expectation-variance-moments.md  # 期望 / 方差 / 矩
├── 04-distributions-cheatsheet.md   # 常见分布速查表
├── 05-multivariate-covariance.md    # 多元分布与协方差
├── 06-limit-theorems.md             # 极限定理（LLN + CLT）
├── 07-estimator-evaluation.md       # 估计量评价四标尺
├── 08-mle-patterns.md               # MLE 求解套路
├── 09-hypothesis-testing-7steps.md  # 假设检验 7 步法
├── 10-information-theory.md         # 信息论核心三件套
├── 11-monte-carlo-mcmc.md           # Monte Carlo + MCMC
└── 12-probability-for-ai.md         # 概率论中的 AI 思维
```

每篇 3000-4500 字。结构与 calculus toolkit 一致：
- 引言
- N 类方法 / 思路
- 演示题（`>` 引用块内心独白）
- 思考路标 ≥6 条
- 典型应用 3 例
- 自测题 4-5 题

---

## 5. 图、公式、约定

### 图配置策略（混合模式）

**目标 70 张 SVG**：

| Part | 张数 | 重点 |
|---|---|---|
| 1 基础 | 6 | Venn 图、概率树、贝叶斯反演反差 |
| 2 随机变量 | 10 | PMF / PDF / CDF、联合分布 |
| 3 分布 | 15 | 离散分布柱状图、连续分布密度、多元正态等高线 |
| 4 极限 | 8 | LLN 收敛轨迹、CLT 直方图收敛、收敛模式对比 |
| 5 统计基础 | 6 | $\chi^2 / t / F$ 抽样分布、箱线图 |
| 6 估计 | 8 | MLE 似然曲面、置信区间、贝叶斯先验/后验 |
| 7 假设检验 | 7 | 检验决策图、$p$ 值、I/II 类错误 |
| 8 高级 | 10 | 熵曲线、KL 散度、MCMC 轨迹、VAE 流程、PGM |
| **合计** | **70** | |

### 命名约定

`prob-pX-YY-Z.{tex,asy}` 表示 `probability` Part X 第 YY 章第 Z 张图。

### 渲染工具链

- 复用 `calculus-tutorial/figures/render.sh`（xelatex + asy + pdf2svg）
- 2D 用 TikZ + pgfplots
- 3D / 多元等高线用 Asymptote / pgfplots surface

---

## 6. 练习题系统（保留 + 重组）

### 保留

- **每章末习题保留**（作为学习节奏的一部分）
- 现有 `appendix/formula-sheet.md`（扩展为 A）
- 现有 `appendix/notation-guide.md` 保留

### 重组 / 新增

- **C 80 + D 100 + E 60 = 240 题总题库** 新建在 `appendix/`
- **F 详解** 按主题分 6-7 文件
- 原 `appendix/answers.md` 内容拆入各章 + F 详解（不再独立保留）

### 240 题分布

| 分组 | 章节范围 | C | D | E | 小计 |
|---|---|---|---|---|---|
| Part 1 基础 | Ch.1-3 | 10 | 12 | 6 | 28 |
| Part 2 随机变量 | Ch.4-6 | 12 | 15 | 8 | 35 |
| Part 3 分布 | Ch.7-9 | 12 | 15 | 8 | 35 |
| Part 4 极限 | Ch.10-12 | 8 | 10 | 6 | 24 |
| Part 5 统计基础 | Ch.13-15 | 8 | 10 | 5 | 23 |
| Part 6 估计 | Ch.16-18 | 10 | 14 | 10 | 34 |
| Part 7 假设检验 | Ch.19-21 | 10 | 12 | 7 | 29 |
| Part 8 高级 | Ch.22-24 | 10 | 12 | 10 | 32 |
| **合计** | | **80** | **100** | **60** | **240** |

### F 详解分文件（视拆分需要，约 6-7 文件）

按经验（calculus F 拆为 7 个文件避免 32K 输出限制）：
- F1a 基础概率 + 条件 + 组合
- F1b 随机变量
- F2a 分布
- F2b 极限定理
- F3a 统计基础 + 估计
- F3b 假设检验
- F4 高级（信息论 / Monte Carlo / 图模型）

---

## 7. 整体目录结构（改造后）

```
probability-statistics-tutorial/
├── README.md                              # 扩展（加 toolkit + appendix 导航）
├── 00-preface.md                          # 保留 + 加"模板 D 价值"段
│
├── thinking-toolkit/                      # 新增（12 篇）
│   ├── 01-probability-axioms.md
│   ├── ... (共 12 篇)
│   └── 12-probability-for-ai.md
│
├── part1-probability-basics/              # 3 章（包装，1 ⭐）
├── part2-random-variables/                # 3 章（1 ⭐）
├── part3-distributions/                   # 3 章（1 ⭐）
├── part4-limit-theorems/                  # 3 章（2 ⭐）
├── part5-statistics-basics/               # 3 章（0 ⭐）
├── part6-estimation/                      # 3 章（2 ⭐）
├── part7-hypothesis-testing/              # 3 章（1 ⭐）
├── part8-advanced-topics/                 # 3 章（2 ⭐）
│
├── figures/                               # 新增基建 + 70 张图
│   ├── render.sh
│   ├── STYLE.md
│   ├── src/{tikz,asy}/
│   └── svg/
│
└── appendix/
    ├── A-formula-sheet.md                 # 扩展自现有 formula-sheet.md
    ├── B-model-atlas.md                   # 新增（约 25-30 个套路模型）
    ├── C-exercises-basic.md               # 新增（80 题）
    ├── D-exercises-medium.md              # 新增（100 题）
    ├── E-exercises-advanced.md            # 新增（60 题）
    ├── F1a-solutions-basics.md
    ├── F1b-solutions-random-variables.md
    ├── F2a-solutions-distributions.md
    ├── F2b-solutions-limit-theorems.md
    ├── F3a-solutions-statistics-estimation.md
    ├── F3b-solutions-hypothesis-testing.md
    ├── F4-solutions-advanced.md
    └── notation-guide.md                  # 保留
```

**规模估算**：24 章 + 12 toolkit + README + preface + 13 appendix = **51 个 md** + ~70 SVG + 240 题。

---

## 8. 里程碑路线（11 个 PM）

| 里程碑 | 内容 | 章数 / 图 |
|---|---|---|
| **PM0** | README + preface 扩展 + 12 toolkit + figures 基建 | 15 文件 |
| **PM1** | Part 1 基础 3 章（1 ⭐ Ch.2）+ 6 图 | 3 章 |
| **PM2** | Part 2 随机变量 3 章（1 ⭐ Ch.6）+ 10 图 | 3 章 |
| **PM3** | Part 3 分布 3 章（1 ⭐ Ch.8）+ 15 图 | 3 章 |
| **PM4** | Part 4 极限 3 章（2 ⭐ Ch.10/11）+ 8 图 | 3 章 |
| **PM5** | Part 5 统计基础 3 章 + 6 图 | 3 章 |
| **PM6** | Part 6 估计 3 章（2 ⭐ Ch.16/18）+ 8 图 | 3 章 |
| **PM7** | Part 7 假设检验 3 章（1 ⭐ Ch.19）+ 7 图 | 3 章 |
| **PM8** | Part 8 高级 3 章（2 ⭐ Ch.22/24）+ 10 图 | 3 章 |
| **PM9** | 全套 appendix（A 扩展 + B 新增 + C 80 + D 100 + E 60）| 5 文件 |
| **PM10** | F 详解 6-7 文件 | 6-7 文件 |

**总产出**：51 md + ~70 SVG + 240 题。

---

## 9. 不在本设计范围内（YAGNI）

- 不重写现有 24 章的核心正文（仅"包装"）
- 不加随机过程 / 测度论 / 回归分析
- 不做英文版
- 不做互动 web 版

---

## 10. 成功标准

- 每个 ⭐ 章节加上"一例速记"和"思维路径还原"后，能让读者"看到 X 立刻想 Y"（如看到"已知后验求先验"立刻想贝叶斯反演）
- 12 篇 toolkit 可作为独立的"概率论方法论"小册子
- 240 题三档分明，与考研 + ML 应用场景对齐
- 与 `calculus-tutorial/` 配合，构成"微积分 → 概率论 → AI"完整数学体系
- 保留现有教程的所有严格推导 + 深度学习应用价值（不丢失）
