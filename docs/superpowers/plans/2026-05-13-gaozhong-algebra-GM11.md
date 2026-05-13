# 高中代数教程 GM11 里程碑实施计划（全教程配图）

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development + rendering-math-figures

**Goal:** 给已完成的 84 章正文 + appendix 配置 60-80 张高优先级 SVG 图（覆盖最关键的可视化点）。

**Architecture:** 用 TikZ + pgfplots / Asymptote 写源文件 → `figures/render.sh` 一键渲染为 SVG → 章节中以 `<img>` 引用。

**前序里程碑:** GM0-GM10（已完成 84 章 + 240 题题库 + 全套 appendix）

**工具链验证：** xelatex（MacTeX 2026）、pdf2svg、asy 全部可用，render.sh 一键执行测试通过。

---

## 文件命名约定

`gz-pX-YY-Z.{tex,asy}` 表示 `gaozhong/algebra` Part X 第 YY 章第 Z 张图。

例：`gz-p4-01-1.tex` = Part 4 第 01 章第 1 张图（$y = a^x$ 与 $y = (1/2)^x$ 对比）。

---

## 图清单（按优先级）

### P1 必备（约 40 张，本里程碑核心目标）

#### Part 4 指对幂（6 张）
- `gz-p4-01-1`：$y = 2^x$ 与 $y = (1/2)^x$ 对比
- `gz-p4-02-1`：$y = \log_2 x$ 与 $y = \log_{1/2} x$ 对比
- `gz-p4-02-2`：$y = a^x$ 与 $y = \log_a x$ 关于 $y = x$ 对称
- `gz-p4-03-1`：5 个幂函数 $y = x, x^2, x^3, x^{1/2}, x^{-1}$

#### Part 5 三角函数（10 张）
- `gz-p5-01-1`：弧度定义（弧长 = 半径）
- `gz-p5-02-1`：单位圆上 $(\cos\alpha, \sin\alpha)$
- `gz-p5-02-2`：四象限三角函数符号
- `gz-p5-03-1`：$y = \sin x$ 图象 + 性质
- `gz-p5-03-2`：$y = \cos x$ 图象 + 性质
- `gz-p5-03-3`：$y = \tan x$ 图象 + 性质
- `gz-p5-03-4`：$y = A\sin(\omega x + \varphi) + b$ 变换链
- `gz-p5-04-1`：诱导公式（圆上对称变换）
- `gz-p5-05-1`：辅助角合并几何意义
- `gz-p5-07-1`：解三角形 ASS 二解判别

#### Part 10 导数（8 张）
- `gz-p10-01-1`：切线斜率 = 极限
- `gz-p10-03-1`：单调与 $f'$ 符号
- `gz-p10-04-1`：极值点 $f' = 0$ + 两侧变号
- `gz-p10-04-2`：闭区间最值（极值 + 端点）
- `gz-p10-05-1`：在 vs 过：两类切线对比
- `gz-p10-06-1`：函数 $h(x) = \frac{\ln x}{x}$ 图象 + 水平线扫
- `gz-p10-07-1`：构造法证不等式 $\ln(1+x) < x$

#### Part 7 统计（3 张）
- `gz-p7-02-1`：频率分布直方图（典型）
- `gz-p7-03-1`：散点图 + 线性回归
- `gz-p7-03-2`：相关系数 $r$ 强度示意

#### Part 8 概率（5 张）
- `gz-p8-01-1`：Venn 图：事件 $A$、$B$、$A \cup B$、$A \cap B$
- `gz-p8-02-1`：几何概型：圆内概率
- `gz-p8-04-1`：条件概率：缩小样本空间
- `gz-p8-05-1`：贝叶斯 / 全概率：树状图
- `gz-p8-05-2`：医学检验贝叶斯反差（$10\times$ 假阳性）

#### Part 12 随机变量（4 张）
- `gz-p12-02-1`：二项分布柱状图 $B(10, 0.4)$
- `gz-p12-04-1`：正态分布钟形曲线 + 3σ 区间
- `gz-p12-04-2`：不同 $\sigma$ 正态曲线对比
- `gz-p12-05-1`：决策类 $E$ 相等 $D$ 不同的柱状图对比

#### Part 1, 6, 9, 11（4 张）
- `gz-p1-01-1`：集合 Venn 图（5 集合关系）
- `gz-p6-01-1`：复平面：$z$、$\overline{z}$、$|z|$
- `gz-p9-02-1`：等差数列 vs 等比数列对比
- `gz-p11-04-1`：杨辉三角 / 帕斯卡三角

### P2 选做（额外约 20-30 张，本里程碑视进度补充）

- Part 2 一元二次不等式 $\Delta$ 三态
- Part 3 单调性 / 奇偶性图解
- Part 8 独立 vs 互斥 Venn 对比
- Part 9 错位相减直观图
- Part 13 综合压轴典型图

---

## 调度

**3 个 subagent 并行：**
- A: P1 Part 4 + Part 5（共 16 张）
- B: P1 Part 10（8 张）+ Part 7 + Part 8（共 16 张）
- C: P1 Part 12 + Part 1, 6, 9, 11（共 8 张）

每完成一批：
1. 写源文件到 `figures/src/{tikz,asy}/`
2. 运行 `cd gaozhong_math/algebra/figures && ./render.sh`
3. commit 源 + SVG

**目标：** 40+ 张 P1 图全部产出，并 push 到 GitHub。

如果 P1 完成后还有时间 / token，再追加 P2。

---

## Task 4：GM11 收尾

```bash
cd /Users/yangyang/ai_projs/math
ls gaozhong_math/algebra/figures/svg/ | wc -l
printf '\n---\n**GM11 完成于：YYYY-MM-DD**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM11.md
git add docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM11.md
git commit -m "docs(gaozhong/algebra): mark GM11 milestone complete"
git push origin master
```

---
**GM11 完成于：2026-05-13**

## 实际产出（P1 优先级）

40 张 SVG 全部渲染成功：

- Part 1 集合（1）
- Part 4 指对幂（6）
- Part 5 三角函数（10）
- Part 6 复数（1）
- Part 7 统计（3）
- Part 8 概率（5）
- Part 9 数列（1）
- Part 10 导数（8）
- Part 11 计数（1）
- Part 12 随机变量（4）

涵盖最关键的可视化点。P2 选做未执行（用户可后续按需补充）。
