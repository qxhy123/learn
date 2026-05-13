# 高中代数教程 GM10 里程碑实施计划（全套 appendix）

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development

**Goal:** 把种子阶段 appendix（4 文件，覆盖 Part 1-5）扩展为全套 appendix（6 文件 + F1-F4 详解，覆盖 Part 1-13 共 84 章），共 **240 题** + 完整详解。

**Architecture:** 6 个新增 / 重写文件 + 4 个详解文件，共 10 个 appendix 文件。

**前序里程碑:** GM0-GM9（13 个 Part 全部完成）

---

## 文件结构（GM10 完成后）

```
gaozhong_math/algebra/appendix/
├── A-formula-sheet.md            # （扩展：加 Part 6-13 公式）
├── B-model-atlas.md              # （扩展：加 Part 6-13 模型）
├── C-exercises-basic.md          # （重命名 / 重写，从 80 题 → 全套基础 80 题，覆盖 Part 1-13）
├── D-exercises-medium.md         # （新增：中档 100 题）
├── E-exercises-advanced.md       # （新增：提升 60 题）
└── F1-solutions-fundamentals.md  # Part 1-5 题目详解
    F2-solutions-functions.md     # Part 3 + 10 详解（函数 + 导数）
    F3-solutions-discrete.md      # Part 6 + 8 + 9 + 11 + 12 详解（数列 / 计数 / 概率 / 随机变量）
    F4-solutions-comprehensive.md # Part 7 + 13 详解（统计 / 综合）
```

实际操作：保留种子阶段 `C-exercises-seed.md` 和 `F-solutions-seed.md` 不动，新增 D / E 文件 + F1-F4 详解文件；同时扩展 A 和 B。

---

## Phase 1：扩展 A、B + 新增 D、E（4 subagent 并行）

### Task 1: 扩展 A-formula-sheet.md（subagent A 任务之一）

**操作**：在现有 A 文件末尾追加：

- Part 6 复数公式：$i^2 = -1$、四则、模、共轭、几何意义
- Part 7 统计公式：方差简化式、相关系数、最小二乘 $\hat{b}, \hat{a}$
- Part 8 概率公式：互斥 / 独立加法乘法、条件 / 全概率 / 贝叶斯
- Part 9 数列公式：等差 / 等比通项 + 求和、5 类递推、6 类求和
- Part 10 导数公式：基本公式表、四则、链式、单调 / 极值判别
- Part 11 计数公式：排列 / 组合 / 二项式
- Part 12 随机变量公式：二项 / 超几何 / 正态、$E$、$D$、$3\sigma$ 法则
- Part 13 综合：放缩砖头 ($\ln(1+x) < x$ 等)、辅助角

### Task 2: 扩展 B-model-atlas.md（subagent A 任务之二）

**操作**：在现有 B 文件末尾追加 12 个新模型（覆盖 Part 6-13）：

- 模型 13：复数共轭对称
- 模型 14：散点图 + 最小二乘
- 模型 15：独立 vs 互斥辨析
- 模型 16：贝叶斯反演（含医学检验经典）
- 模型 17：等差数列 $S_n$ 二次函数判据
- 模型 18：错位相减
- 模型 19：导数 4 步求单调
- 模型 20：构造法证不等式
- 模型 21：闭区间最值候选集
- 模型 22：二项式定理通项 $T_{k+1}$
- 模型 23：二项 / 超几何 / 正态选择树
- 模型 24：决策 $E$ vs $D$

### Task 3: 新增 D-exercises-medium.md（subagent B 任务）

**操作**：100 题中档练习，按 Part 分布：

| Part | 中档题数 | 编号 |
|---|---|---|
| Part 1-5（高一上） | 30 | D.01-D.30 |
| Part 6-8（高一下 / 概统） | 15 | D.31-D.45 |
| Part 9 数列 | 12 | D.46-D.57 |
| Part 10 导数 | 18 | D.58-D.75 |
| Part 11-12 计数 / 随机变量 | 12 | D.76-D.87 |
| Part 13 综合 | 13 | D.88-D.100 |

### Task 4: 新增 E-exercises-advanced.md（subagent C 任务）

**操作**：60 题高考压轴级，按 Part 分布：

| Part | 提升题数 | 编号 |
|---|---|---|
| Part 3 + 10 函数导数综合 | 20 | E.01-E.20（核心压轴）|
| Part 9 数列综合 | 10 | E.21-E.30 |
| Part 5 + 11 三角 / 计数 | 8 | E.31-E.38 |
| Part 8 + 12 概率综合 | 8 | E.39-E.46 |
| Part 13 高考真题难度 | 14 | E.47-E.60 |

---

## Phase 2：F1-F4 详解（4 subagent 并行）

D 100 + E 60 = 160 题 + 现有种子 C 80 题 = 总 240 题。详解按 Part 分 4 组。

### Task 5: F1-solutions-fundamentals.md（基础与逻辑/不等式/集合/三角）

覆盖 Part 1, 2, 4, 5 在 D 和 E 中的题目（约 60 题）。

### Task 6: F2-solutions-functions-and-derivatives.md（函数 + 导数）

覆盖 Part 3 和 Part 10 在 D 和 E 中的题目（约 70 题，含 E 中 20 题压轴）。

### Task 7: F3-solutions-discrete.md（数列 / 计数 / 概率 / 随机变量）

覆盖 Part 6 + 8 + 9 + 11 + 12 在 D 和 E 中的题目（约 60 题）。

### Task 8: F4-solutions-statistics-and-comprehensive.md（统计 / 综合）

覆盖 Part 7 和 Part 13 在 D 和 E 中的题目（约 30 题）。

---

## Task 9: GM10 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' gaozhong_math/algebra/appendix
ls gaozhong_math/algebra/appendix
printf '\n---\n**GM10 完成于：YYYY-MM-DD**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM10.md
git add docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM10.md
git commit -m "docs(gaozhong/algebra): mark GM10 milestone complete (full appendix)"
git push origin master
```

---

## 调度

**Phase 1（4 subagent 并行）：**
- A: 扩 A 公式表 + 扩 B 模型图集
- B: D 中档 100 题
- C: E 提升 60 题
- D: 实际不安排（Phase 2 等数据齐备）

**Phase 2（Phase 1 完成后，4 subagent 并行）：**
- E1: F1 详解
- E2: F2 详解（函数 + 导数 70 题，最多）
- E3: F3 详解
- E4: F4 详解

每文件独立 commit。
