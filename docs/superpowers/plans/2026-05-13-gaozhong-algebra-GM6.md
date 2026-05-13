# 高中代数教程 GM6 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development

**Goal:** 完成 Part 9 数列（6 章）。

**Architecture:** 4 章模板 C + **2 章模板 D**（递推求通项、求和技巧）。每章独立提交。

**前序里程碑:** GM0-GM5（已完成）

---

## 文件结构

```
gaozhong_math/algebra/part9-sequences/   # [高二上]
├── 01-sequence-basics.md
├── 02-arithmetic-sequence.md
├── 03-geometric-sequence.md
├── 04-recurrence-relations.md   # ⭐ 递推求通项
├── 05-sum-techniques.md         # ⭐ 错位相减 / 裂项求和
└── 06-sequence-applications.md
```

---

### Task 1: part9/01 数列基础（模板 C）

Title: `# 数列的概念与表示`

涵盖：
- 数列定义：按一定次序排列的数 $\{a_n\}$
- 通项公式 $a_n = f(n)$；递推公式 $a_{n+1} = g(a_n)$
- 前 $n$ 项和 $S_n$；$a_n$ 与 $S_n$ 关系 $a_n = S_n - S_{n-1}$（$n \geq 2$），$a_1 = S_1$
- 数列性质：单调（递增 / 递减 / 摆动）、有界、周期
- 典型应用 3 例 + 自测题 4 题
- 易错点：$S_n - S_{n-1}$ 公式必须 $n \geq 2$，$n = 1$ 单独验证

### Task 2: part9/02 等差数列（模板 C，2500-3500 字）

Title: `# 等差数列`

涵盖：
- 定义 $a_{n+1} - a_n = d$（常数）
- 通项 $a_n = a_1 + (n-1) d$
- 求和 $S_n = \frac{n(a_1 + a_n)}{2} = na_1 + \frac{n(n-1)}{2} d$
- 等差中项 $a_n = \frac{a_{n-1} + a_{n+1}}{2}$
- 性质：若 $m + n = p + q$，则 $a_m + a_n = a_p + a_q$
- $S_n, S_{2n}-S_n, S_{3n}-S_{2n}$ 也成等差
- 典型应用 3 例（求和 / 公差 / 含参） + 自测题 5 题

### Task 3: part9/03 等比数列（模板 C，2500-3500 字）

Title: `# 等比数列`

涵盖：
- 定义 $\frac{a_{n+1}}{a_n} = q$（$q \neq 0$）
- 通项 $a_n = a_1 q^{n-1}$
- 求和 $S_n = \frac{a_1(1-q^n)}{1-q}$（$q \neq 1$），$S_n = na_1$（$q = 1$）
- 等比中项 $a_n^2 = a_{n-1} a_{n+1}$
- 性质：若 $m + n = p + q$，则 $a_m a_n = a_p a_q$
- 与等差的对偶（加 → 乘、和 → 积）
- 典型应用 3 例（含分类 $q=1$ / 含参） + 自测题 5 题
- 易错点：$q=1$ 单独处理

### Task 4: part9/04 递推求通项 ⭐（模板 D）

Title: `# 由递推关系求通项公式`

**一例速记**：
> 5 类常见递推：① $a_{n+1} = a_n + f(n)$ 累加 ② $a_{n+1} = a_n \cdot g(n)$ 累乘  
> ③ $a_{n+1} = p a_n + q$ 待定系数法构造等比 ④ $a_{n+1} = \frac{a_n}{a_n + c}$ 取倒数  
> ⑤ $a_{n+1} = p a_n + q^n$ 两边除 $q^{n+1}$

**引入题**：$a_1 = 1$，$a_{n+1} = 2a_n + 3$，求 $a_n$。

**思维路径还原** 15-20 行：识别"线性递推 + 常数项" → 构造 $a_n + 3 = 2(a_{n-1} + 3)$。

**思考路标 ≥8 条**：
- 看到 $a_{n+1} - a_n = $ 含 $n$ 表达式 → 累加
- 看到 $\frac{a_{n+1}}{a_n} = $ 含 $n$ 表达式 → 累乘
- 看到 $a_{n+1} = p a_n + q$ → 配凑等比
- 看到 $\frac{1}{a_{n+1}} = \frac{1}{a_n} + c$ → 取倒数
- 看到含 $q^n$ → 除以 $q^{n+1}$ 化为等差

5 类经典构造演示 + 自测题 5 题。

### Task 5: part9/05 求和技巧 ⭐（模板 D）

Title: `# 数列求和的常见方法`

**一例速记**：
> 6 类常见求和：① 公式法 ② 错位相减 ③ 裂项相消 ④ 倒序相加 ⑤ 分组求和 ⑥ 拆项

**引入题**：求 $S_n = 1 \cdot 2 + 2 \cdot 2^2 + 3 \cdot 2^3 + \ldots + n \cdot 2^n$。

**思维路径还原** 15-20 行：识别"等差 × 等比" → 错位相减。

**思考路标 ≥8 条**：
- 等差 × 等比 → 错位相减
- $\frac{1}{n(n+1)} = \frac{1}{n} - \frac{1}{n+1}$ → 裂项
- $\frac{1}{(2n-1)(2n+1)} = \frac{1}{2}(\frac{1}{2n-1} - \frac{1}{2n+1})$
- $\frac{1}{n(n+1)(n+2)} = \frac{1}{2}[\frac{1}{n(n+1)} - \frac{1}{(n+1)(n+2)}]$
- 看到 $a_{n+1} + a_n$ → 倒序相加

6 类方法的经典演示 + 自测题 5 题。

### Task 6: part9/06 数列应用（模板 C）

Title: `# 数列的实际应用与综合`

涵盖：
- 银行复利 / 等额本息
- 增长率 / 衰减问题
- 树形结构（如分形数列）
- 数列不等式（与基本不等式 / 二次函数结合）
- 典型应用 3 例 + 自测题 4 题

---

## Task 7: GM6 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' gaozhong_math/algebra/part9-sequences
ls gaozhong_math/algebra/part9-sequences
printf '\n---\n**GM6 完成于：YYYY-MM-DD**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM6.md
git add docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM6.md
git commit -m "docs(gaozhong/algebra): mark GM6 milestone complete"
git push origin master
```

---

## 调度

**2 个 sonnet subagent 并行：**
- A: 01 基础、02 等差、03 等比
- B: 04 ⭐ 递推、05 ⭐ 求和、06 应用

每章独立 commit。
