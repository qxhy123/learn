# 高中代数教程 GM8 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development

**Goal:** 完成 Part 11 计数原理（4 章）+ Part 12 随机变量（5 章），共 9 章。

**Architecture:** 7 章模板 C + **2 章模板 D**（二项式定理、期望与方差）。

**前序里程碑:** GM0-GM7（已完成）

---

## 文件结构

```
gaozhong_math/algebra/
├── part11-counting/                   # [高二下/高三上]
│   ├── 01-fundamental-counting.md
│   ├── 02-permutation.md
│   ├── 03-combination.md
│   └── 04-binomial-theorem.md        # ⭐
└── part12-random-variables/           # [选必三]
    ├── 01-discrete-rv.md
    ├── 02-binomial-distribution.md
    ├── 03-hypergeometric.md
    ├── 04-normal-distribution.md
    └── 05-expectation-variance.md    # ⭐
```

---

## Part 11: 计数原理

### Task 1: part11/01 基本计数原理（模板 C）

Title: `# 分类加法与分步乘法原理`

- 分类加法 $N = n_1 + n_2 + \ldots + n_k$（互斥）
- 分步乘法 $N = n_1 \cdot n_2 \cdots n_k$
- 何时分类 / 分步：分类是"不同情形"，分步是"完成一件事的多个步骤"
- 典型应用 3 例（排路 / 涂色 / 选课）+ 自测题 4 题
- 易错点：分类必须互斥不重；分步必须连贯

### Task 2: part11/02 排列（模板 C）

Title: `# 排列`

- 排列定义：从 $n$ 个不同元素取 $m$ 个按一定顺序排列
- 排列数 $A_n^m = \frac{n!}{(n-m)!} = n(n-1)\cdots(n-m+1)$
- 全排列 $A_n^n = n!$
- **特殊位置 / 相邻 / 间隔**：先排特殊，再排其它；相邻"捆绑"；间隔"插空"
- 典型应用 3 例 + 自测题 5 题

### Task 3: part11/03 组合（模板 C，2500-3500 字）

Title: `# 组合`

- 组合定义：不考虑顺序
- 组合数 $C_n^m = \frac{A_n^m}{m!} = \frac{n!}{m!(n-m)!}$
- 性质：$C_n^m = C_n^{n-m}$、$C_n^m + C_n^{m-1} = C_{n+1}^m$（帕斯卡）
- **隔板法**：分球入盒
- **染色问题**：与排列结合
- 典型应用 3 例 + 自测题 5 题
- 易错点：排列还是组合（按"是否有序"）

### Task 4: part11/04 二项式定理 ⭐（模板 D）

Title: `# 二项式定理`

**一例速记**：
> $(a+b)^n = \sum_{k=0}^n C_n^k a^{n-k} b^k$  
> 通项 $T_{k+1} = C_n^k a^{n-k} b^k$（注意 $k$ 从 0 开始）  
> 系数和：$x = y = 1$ 代入 $(x+y)^n$

**引入题**：求 $(x - \frac{1}{x})^6$ 的展开式中的常数项。

**思维路径还原** 15-20 行：用通项 $T_{k+1}$，令 $x$ 的指数 = 0 → 解 $k$ → 代入。

**3 类常见题型**：
① 求指定项 / 指定系数
② 系数和（赋值法）
③ 不等式 / 整除问题（如 $11^{2020}$ 除以 100 余多少）

**思考路标 ≥8 条**：
- 用 $T_{k+1}$，不要漏 $C_n^k$
- 系数和 $\to$ 代入 $x = 1$
- 奇偶项系数和 $\to$ 代入 $x = -1$
- $(1+x)^n$ 中 $x^k$ 系数 = $C_n^k$

例题 3 例 + 自测题 5 题。

---

## Part 12: 随机变量

### Task 5: part12/01 离散型随机变量（模板 C）

Title: `# 离散型随机变量及其分布列`

- 离散型随机变量 $X$
- **分布列**：列出 $X$ 所有可能取值及对应概率，$\sum P_i = 1$
- 两点分布 / 二项分布介绍
- 期望 $E(X) = \sum x_i P_i$
- 典型应用 3 例 + 自测题 4 题

### Task 6: part12/02 二项分布（模板 C，2500-3500 字）

Title: `# 二项分布 B(n, p)`

- **n 次独立重复试验**，每次成功概率 $p$ → $X$ = 成功次数 ~ $B(n, p)$
- $P(X = k) = C_n^k p^k (1-p)^{n-k}$
- **期望** $E(X) = np$；**方差** $D(X) = np(1-p)$
- 与古典概率联动
- 典型应用 3 例 + 自测题 4 题

### Task 7: part12/03 超几何分布（模板 C）

Title: `# 超几何分布`

- 不放回抽样：$N$ 件中 $M$ 件次品，抽 $n$ 件，$X$ = 抽到次品数
- $P(X=k) = \frac{C_M^k C_{N-M}^{n-k}}{C_N^n}$
- 与二项分布对比（关键：放回 vs 不放回）
- 典型应用 3 例 + 自测题 4 题

### Task 8: part12/04 正态分布（模板 C，2500-3500 字）

Title: `# 正态分布`

- 连续型随机变量初步
- 正态分布 $X \sim N(\mu, \sigma^2)$
- 概率密度函数 $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
- 性质：钟形对称，$x = \mu$ 处取最大值
- **3$\sigma$ 法则**：$P(\mu-\sigma < X < \mu+\sigma) \approx 0.6826$；$P(\mu-2\sigma < X < \mu+2\sigma) \approx 0.9544$；$P(\mu-3\sigma < X < \mu+3\sigma) \approx 0.9974$
- 标准正态 $N(0, 1)$
- 典型应用 3 例 + 自测题 4 题

### Task 9: part12/05 期望与方差 ⭐（模板 D）

Title: `# 随机变量的期望与方差`

**一例速记**：
> $E(X) = \sum x_i P_i$；$D(X) = \sum (x_i - E(X))^2 P_i = E(X^2) - [E(X)]^2$  
> 线性性质：$E(aX + b) = aE(X) + b$；$D(aX + b) = a^2 D(X)$  
> 独立：$E(XY) = E(X)E(Y)$（独立时）；$D(X+Y) = D(X) + D(Y)$（独立时）

**引入题**：$X$ 服从 $B(5, 0.4)$，求 $E(X)$ 和 $D(X)$；再求 $E(3X+2)$ 和 $D(3X+2)$。

**思维路径还原** 15-20 行：直接代公式 + 线性变换。

**3 类常见题型**：
① 直接套分布公式
② 由分布列求期望方差
③ 决策问题（比较两种方案的期望 / 方差）

**思考路标 ≥8 条**：
- 二项 $E = np, D = np(1-p)$
- 超几何 $E = n\frac{M}{N}$
- 线性变换 $E$ 线性、$D$ 平方系数

例题 3 例 + 自测题 5 题。

---

## Task 10: GM8 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' gaozhong_math/algebra/part11-counting gaozhong_math/algebra/part12-random-variables
ls gaozhong_math/algebra/part11-counting gaozhong_math/algebra/part12-random-variables
printf '\n---\n**GM8 完成于：YYYY-MM-DD**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM8.md
git add docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM8.md
git commit -m "docs(gaozhong/algebra): mark GM8 milestone complete"
git push origin master
```

---

## 调度

**2 个 sonnet subagent 并行：**
- A: Part 11 计数 4 章（含 ⭐ 二项式定理）
- B: Part 12 随机变量 5 章（含 ⭐ 期望方差）

每章独立 commit。
