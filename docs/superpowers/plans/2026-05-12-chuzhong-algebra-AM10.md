# 中考代数教程 AM10 里程碑实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** 完成 `chuzhong_algebra/appendix/` —— 公式表 + 模型图集 + 180 题三档题库 + 4 个详解文件。共 9 个文件。

**Architecture:** 两阶段执行。Phase 1 并行 4 subagent（公式+图集 / C / D / E 题库）；Phase 2 并行 2 subagent（F1+F2 详解 / F3+F4 详解）。详解必须在题库写完后才能写。

**Spec 参考:** `docs/superpowers/specs/2026-05-12-chuzhong-algebra-design.md`
**前序里程碑:** AM0–AM7、AM9（已完成）；**AM8 留待**

---

## 文件结构

```
chuzhong_algebra/appendix/
├── A-formula-sheet.md                         # 公式定理速查
├── B-model-atlas.md                           # 套路图集（30+ 模板 D 模型）
├── C-exercises-basic.md                       # 基础题库 60 题（☆）
├── D-exercises-medium.md                      # 中档题库 80 题（☆☆）
├── E-exercises-advanced.md                    # 压轴题库 40 题（☆☆☆）
├── F1-solutions-numbers-and-expressions.md   # 详解 part1-4 范围题
├── F2-solutions-equations-and-inequalities.md # 详解 part5-7 范围题
├── F3-solutions-functions.md                  # 详解 part8-10 范围题
└── F4-solutions-stat-prob-apps-comp.md        # 详解 part11-14 范围题（不含 part13）
```

---

## Task 1: A 公式定理速查表

**File:** `chuzhong_algebra/appendix/A-formula-sheet.md`（3500-5000 字）

按 part 顺序汇总全部公式与定理。每条简洁、附章节链接。

包含（按章节顺序）：
- **part1 数**：相反数、绝对值、运算法则、科学记数、平方根/立方根、二次根式性质
- **part2 整式**：单项式系数次数、整式运算法则、幂运算 6 条、整式乘法 3 类、完全平方 + 平方差
- **part3 因式分解**：4 种方法决策树
- **part4 分式**：基本性质、四则运算、分式方程三步走 + 验根
- **part5 一次方程**：等式性质、解一元一次方程 5 步、代入/加减消元、含参 3 步
- **part6 一元二次方程**：4 种解法、求根公式、判别式三情形、韦达定理 + 4 个变形公式
- **part7 不等式**：3 大性质（变号注意）、4 种情形（同大取大等）
- **part8 一次函数**：$y = kx+b$ 性质、象限判断表、待定系数法、与方程组几何意义
- **part9 反比例函数**：$y = k/x$ 性质、$|k|$ 几何意义（矩形/三角形）
- **part10 二次函数**：三种形式、性质（对称轴/顶点/最值）、$\Delta$ + 抛物线交点、含参/铅垂高
- **part11 统计**：抽样、4 种统计图、平均/中位/众数对比、方差公式
- **part12 概率**：$P(A) = m/n$、列表法、树状图、频率估计
- **part14 综合**：4 类常考综合题型概览
- **thinking-toolkit 10 篇**：10 个思维方法核心要点

Commit: `docs(chuzhong_algebra): add appendix A (formula sheet)`

---

## Task 2: B 套路图集

**File:** `chuzhong_algebra/appendix/B-model-atlas.md`（3000-4000 字）

对全部 ⭐ 模板 D 模型做一例速记式总览。每个模型：**名称 + 一例速记（公式样板） + 一句话方法描述 + 链接原章节**。

涵盖（含 30+ 模板 D 章节）：
- **整式类** (2)：完全平方/平方差、公式逆用
- **因式分解类** (2)：十字相乘、综合策略
- **分式类** (2)：分式方程及增根、分式化简技巧
- **一次方程组类** (2)：解题技巧、含参
- **一元二次方程类** (2)：配方法、韦达定理
- **不等式类** (1)：含参不等式分类讨论
- **一次函数类** (3)：待定系数法、与方程组几何意义、含参
- **反比例类** (1)：$k$ 的几何意义
- **二次函数类** (4)：三种形式、与方程关系、含参最值、铅垂高法
- **概率类** (1)：列举法 / 树状图
- **综合类** (6)：part14 全部 6 章型

每条 50-100 字。最后加"**思维方法网**"：toolkit 10 篇与模型的对应关系。

Commit: `docs(chuzhong_algebra): add appendix B (model atlas)`

---

## Task 3: C 基础题库 60 题

**File:** `chuzhong_algebra/appendix/C-exercises-basic.md`（3500-5000 字，60 题）

按章节顺序排列。每题：
- 编号 C.x（带 ☆）
- 出题点 + 模型标签 `(partN/yy)`
- 题目本文（LaTeX）
- **不附答案**（答案在 F）

分组（按 part 数量分布）：
- part1 数：8 题
- part2 整式：6 题
- part3 因式分解：5 题
- part4 分式：4 题
- part5 一次方程：5 题
- part6 一元二次方程：5 题
- part7 不等式：4 题
- part8 一次函数：5 题
- part9 反比例：3 题
- part10 二次函数：5 题
- part11 统计：4 题
- part12 概率：3 题
- part14 综合：3 题
**共 60 题**。

格式样例：
```
**C.1** ☆（出题点：有理数运算；模型：part1/02）
计算 $-3^2 + (-2)^3 \times (-\frac{1}{2})^2$。
```

Commit: `docs(chuzhong_algebra): add appendix C (basic exercises, 60)`

---

## Task 4: D 中档题库 80 题

**File:** `chuzhong_algebra/appendix/D-exercises-medium.md`（4500-6000 字，80 题）

中档 ☆☆。需要识别 1-2 个模型 + 推理。

分组：
- part1-4 数/式/因式/分式：18 题
- part5-7 方程/不等式：18 题
- part8-10 函数：22 题
- part11-12 统计概率：8 题
- part14 综合：14 题
**共 80 题**。

每题用 `**D.x** ☆☆（出题点；模型）` 格式。

Commit: `docs(chuzhong_algebra): add appendix D (medium exercises, 80)`

---

## Task 5: E 压轴题库 40 题

**File:** `chuzhong_algebra/appendix/E-exercises-advanced.md`（3500-5000 字，40 题）

压轴 ☆☆☆。综合 2-3 个模型 + 分类讨论 + 列方程 / 函数。中考真题风格，含多问 (1)(2)(3)。

分组：
- 函数综合（一次+反比例+二次）：15 题
- 应用题压轴（增长率/方案/分段）：10 题
- 二次函数与几何代数综合：10 题
- 新定义 / 创新题：5 题
**共 40 题**。

每题用 `**E.x** ☆☆☆（出题点；模型）` 格式。

Commit: `docs(chuzhong_algebra): add appendix E (advanced exercises, 40)`

---

## Task 6: F1 详解 part1-4 范围题

**File:** `chuzhong_algebra/appendix/F1-solutions-numbers-and-expressions.md`（4500-6500 字）

按题号顺序详解 C/D/E 中所有 **part1-4 范围内**的题目。

每题详解结构：
- **题号 ☆/☆☆/☆☆☆**
- **【思路】模型标签**（如"配方法 + 完全平方"、"分式方程 + 增根"）
- **解答**（完整步骤）
- **答案**

字数：基础 50-100 字/题；中档 100-200 字/题；压轴 200-300 字/题。

Commit: `docs(chuzhong_algebra): add appendix F1 (solutions for part1-4)`

---

## Task 7: F2 详解 part5-7 范围题

**File:** `chuzhong_algebra/appendix/F2-solutions-equations-and-inequalities.md`（4500-6500 字）

按题号顺序详解所有 part5-7 范围题（一次方程/一元二次方程/不等式）。

Commit: `docs(chuzhong_algebra): add appendix F2 (solutions for part5-7)`

---

## Task 8: F3 详解 part8-10 范围题

**File:** `chuzhong_algebra/appendix/F3-solutions-functions.md`（5000-7000 字）

按题号顺序详解所有 part8-10 范围题（一次函数/反比例/二次函数）。**函数压轴部分篇幅最长**。

Commit: `docs(chuzhong_algebra): add appendix F3 (solutions for part8-10)`

---

## Task 9: F4 详解 part11-12 + part14 范围题

**File:** `chuzhong_algebra/appendix/F4-solutions-stat-prob-apps-comp.md`（4000-5500 字）

按题号顺序详解所有 part11-12 范围题（统计/概率）+ part14 综合题（不含 part13 应用题，应用题暂未做）。

Commit: `docs(chuzhong_algebra): add appendix F4 (solutions for part11-14)`

---

## Task 10: AM10 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' chuzhong_algebra/appendix
ls chuzhong_algebra/appendix
printf '\n---\n**AM10 完成于：YYYY-MM-DD**\n' >> docs/superpowers/plans/2026-05-12-chuzhong-algebra-AM10.md
git add docs/superpowers/plans/2026-05-12-chuzhong-algebra-AM10.md
git commit -m "docs(chuzhong_algebra): mark AM10 milestone complete"
git push origin master
```

---

## Phase 1 / Phase 2 调度

**Phase 1（并行 4 subagent）**：
- A: Task 1 (公式表) + Task 2 (模型图集)
- B: Task 3 (C 题库 60 题)
- C: Task 4 (D 题库 80 题)
- D: Task 5 (E 题库 40 题)

**Phase 2（Phase 1 完成后并行 2 subagent）**：
- E: Task 6 (F1) + Task 7 (F2)
- F: Task 8 (F3) + Task 9 (F4)

详解 subagent 需要读 C/D/E 题库（找该 part 范围的题号），所以必须等 Phase 1 完成。
