# 高中几何教程 GM11 里程碑实施计划（全套 appendix）

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development

**Goal:** 把种子阶段 appendix（4 文件，覆盖 Part 1, 2, 3, 4, 8）扩展为全套 appendix（10 文件），覆盖全部 10 个 Part 共 **240 题**（C 80 + D 100 + E 60）+ 完整详解。

**前序里程碑:** GM0-GM10（10 个 Part 51 章正文已全部完成）

---

## 文件结构（GM11 完成后）

```
gaozhong_math/geometry/appendix/
├── A-formula-sheet.md           # （扩展：加 Part 5, 6, 7, 9, 10 公式）
├── B-model-atlas.md             # （扩展：加 12 个新模型）
├── C-exercises-seed.md          # （保留：80 题种子）
├── D-exercises-medium.md        # （新增：中档 100 题）
├── E-exercises-advanced.md      # （新增：提升 60 题）
├── F-solutions-seed.md          # （保留：80 题详解）
└── F1-F4（4 个新增详解文件）:
    F1-solutions-vectors.md           # Part 1, 2, 9 向量 D+E 详解
    F2-solutions-line-circle.md       # Part 3, 4 直线圆 D+E 详解
    F3-solutions-conics.md            # Part 5, 6, 7 圆锥曲线 D+E 详解
    F4-solutions-solid-comprehensive.md # Part 8, 10 立体 + 综合 D+E 详解
```

---

## Phase 1：扩展 A、B + 新增 D、E（4 subagent 并行）

### Task 1: 扩展 A-formula-sheet.md

追加 Part 5-7, 9, 10 公式（约 3000-4000 字）：
- Part 5 椭圆（定义 + 标准方程 + 7 大性质 + 焦半径 + 通径 + 直线椭圆 + 点差法）
- Part 6 双曲线（定义 + 方程 + 渐近线 + $e>1$ + 焦半径含绝对值 + 等轴双曲线）
- Part 7 抛物线（定义 + 4 种方程 + 焦点弦三大性质 + $e=1$）
- Part 9 空间向量（坐标公式 + 法向量求法 + 3 类角公式 + 3 类距离公式）
- Part 10 综合（5 大套路 / 3 定问题 / 5 大轨迹方法）

### Task 2: 扩展 B-model-atlas.md

追加 12 个新模型（编号 13-24）：
- 模型 13: 椭圆 7 大性质（→ Part 5/02）
- 模型 14: 椭圆中点弦点差法（→ Part 5/03）
- 模型 15: 双曲线渐近线 + 单交点 4 种位置（→ Part 6/03）
- 模型 16: 抛物线焦点弦三大性质（→ Part 7/02-03）
- 模型 17: 圆锥曲线统一定义（$e$ 与离心率）（→ Part 5-7）
- 模型 18: 空间建系 4 步（→ Part 9/03）
- 模型 19: 法向量求法（→ Part 9/04）
- 模型 20: 二面角法向量法（同指 vs 异指）（→ Part 9/05）
- 模型 21: 韦达定理 + 设而不求（→ Part 10/01）
- 模型 22: 定点问题套路（→ Part 10/04）
- 模型 23: 轨迹方程 5 大方法（→ Part 10/03）
- 模型 24: 立体几何"综合 vs 向量"切换（→ Part 10/05）

### Task 3: 新增 D-exercises-medium.md（100 题）

| 分组 | Part | 题数 | 编号 |
|---|---|---|---|
| Part 1+2 平面向量综合 | 1, 2 | 10 | D.01-D.10 |
| Part 3+4 直线圆综合 | 3, 4 | 15 | D.11-D.25 |
| Part 5 椭圆 | 5 | 12 | D.26-D.37 |
| Part 6 双曲线 | 6 | 10 | D.38-D.47 |
| Part 7 抛物线 | 7 | 10 | D.48-D.57 |
| Part 8 立体几何 | 8 | 15 | D.58-D.72 |
| Part 9 空间向量 | 9 | 15 | D.73-D.87 |
| Part 10 综合 | 10 | 13 | D.88-D.100 |
| **合计** | | **100** | |

### Task 4: 新增 E-exercises-advanced.md（60 题）

| 分组 | 主题 | 题数 | 编号 |
|---|---|---|---|
| 圆锥曲线综合（核心压轴） | Part 5, 6, 7, 10 | 25 | E.01-E.25 |
| 立体几何 + 空间向量 | Part 8, 9, 10 | 18 | E.26-E.43 |
| 定点 / 定值 / 定直线 | Part 10 | 10 | E.44-E.53 |
| 高考真题难度综合 | 全 Part | 7 | E.54-E.60 |
| **合计** | | **60** | |

---

## Phase 2：F1-F4 详解（4 subagent 并行）

D 100 + E 60 = 160 题，按主题分 4 组：

### Task 5: F1 向量详解（Part 1, 2, 9 在 D+E 中的题）
- D.01-D.10（平面向量 10 题）
- D.73-D.87（空间向量 15 题）
- E 中向量相关（含立体 / 综合）约 10 题
- 约 35-40 题

### Task 6: F2 直线圆详解（Part 3, 4）
- D.11-D.25（直线圆 15 题）
- E 中直线圆相关约 5 题
- 约 20 题

### Task 7: F3 圆锥曲线详解（Part 5, 6, 7）
- D.26-D.57（椭圆 + 双曲线 + 抛物线 共 32 题）
- E.01-E.25（圆锥压轴 25 题 + E.44-E.53 定点 10 题）
- 约 65 题

### Task 8: F4 立体 + 综合详解（Part 8, 10）
- D.58-D.72（立体 15 题）
- D.88-D.100（综合 13 题）
- E.26-E.43（立体压轴 18 题）+ E.54-E.60（综合 7 题）
- 约 53 题

---

## Task 9: GM11 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|待补' gaozhong_math/geometry/appendix
ls gaozhong_math/geometry/appendix
printf '\n---\n**GM11 完成于：2026-05-14**\n' >> docs/superpowers/plans/2026-05-14-gaozhong-geometry-GM11.md
git add docs/superpowers/plans/2026-05-14-gaozhong-geometry-GM11.md
git commit -m "docs(gaozhong/geometry): mark GM11 milestone complete (full 240-problem appendix)"
git push origin master
```

---

## 调度

**Phase 1（4 subagent 并行）：**
- A: 扩 A + 扩 B
- B: D 100 题
- C: E 60 题（独立 subagent，等 D 完成后启动）

实际：先 3 个并行（A 扩展 A+B、B 写 D、C 写 E），完成后再 Phase 2 用 4 subagent。

**Phase 2 (Phase 1 完成后，4 subagent 并行)：**
- E1: F1 向量详解
- E2: F2 直线圆详解
- E3: F3 圆锥曲线详解
- E4: F4 立体 + 综合详解

每文件独立 commit。
