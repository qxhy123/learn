# 高中代数教程 GM7 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development

**Goal:** 完成 Part 10 导数（7 章，最重要的一个 part）。

**Architecture:** 4 章模板 C + **3 章模板 D**（用导数求单调区间、极值最值、导数与零点）。每章独立提交。

**前序里程碑:** GM0-GM6（已完成）

---

## 文件结构

```
gaozhong_math/algebra/part10-derivatives/   # [高二上 / 下]
├── 01-derivative-concept.md
├── 02-derivative-rules.md
├── 03-monotonicity-by-derivative.md   # ⭐
├── 04-extrema-and-optimization.md     # ⭐
├── 05-tangent-line.md
├── 06-derivative-and-zeros.md         # ⭐
└── 07-derivative-applications.md
```

---

### Task 1: part10/01 导数概念（模板 C，2500-3500 字）

Title: `# 导数的概念与几何意义`

涵盖：
- 平均变化率 $\frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$
- 瞬时变化率 = 极限定义：$f'(x_0) = \lim_{\Delta x \to 0} \frac{f(x_0+\Delta x) - f(x_0)}{\Delta x}$
- **几何意义**：$f'(x_0)$ = 在 $x_0$ 处切线的斜率
- **物理意义**：位移导数 = 速度；速度导数 = 加速度
- 导函数 $f'(x)$ 概念
- 典型应用 3 例（用定义求导 / 几何意义切线 / 速度问题）
- 易错点：极限定义中 $\Delta x \to 0$ 而非 $= 0$
- 自测题 4 题

### Task 2: part10/02 求导法则（模板 C，2500-3500 字）

Title: `# 求导法则与基本公式`

涵盖：
- **基本初等函数求导公式表**：
  - $C' = 0$，$(x^n)' = n x^{n-1}$
  - $(\sin x)' = \cos x$，$(\cos x)' = -\sin x$，$(\tan x)' = \sec^2 x$
  - $(a^x)' = a^x \ln a$，$(e^x)' = e^x$
  - $(\log_a x)' = \frac{1}{x \ln a}$，$(\ln x)' = \frac{1}{x}$
- **运算法则**：
  - $(u \pm v)' = u' \pm v'$
  - $(uv)' = u'v + uv'$
  - $(\frac{u}{v})' = \frac{u'v - uv'}{v^2}$
  - **链式法则**：$[f(g(x))]' = f'(g(x)) \cdot g'(x)$
- 高阶导数 $f''(x)$ 概念
- 典型应用 3 例（含复合函数 / 商 / 乘法） + 自测题 5 题
- 易错点：链式法则不要漏内层导数

### Task 3: part10/03 用导数求单调区间 ⭐（模板 D）

Title: `# 用导数判断函数的单调性`

**一例速记**：
> $f'(x) > 0$ → $f$ 增；$f'(x) < 0$ → $f$ 减  
> 求单调区间的标准 4 步：① 求定义域 ② 求 $f'(x)$ ③ 解 $f'(x) > 0$ / $f'(x) < 0$ ④ 写区间（不能用 $\cup$）

**引入题**：求 $f(x) = x^3 - 3x^2 - 9x + 1$ 的单调区间。

**思维路径还原** 15-20 行：求导 → 因式分解 → 列表分析符号 → 写区间。

**3 类常见题型**：
① 一般函数求单调区间
② **含参函数的单调性**：分类讨论 $a$
③ 反向：已知单调求参数范围

**思考路标 ≥8 条**：
- 看到"求单调区间" → 先定义域 → 求导
- 含参 → 必分类讨论
- 单调区间不能用 $\cup$ 连接
- $f'(x) \geq 0$（$f'(x) = 0$ 在有限个点处）也是增
- 含 $\ln$ → 定义域 $x > 0$

例题 3 例（含含参分类）+ 自测题 5 题。

### Task 4: part10/04 极值与最值 ⭐（模板 D）

Title: `# 函数的极值与最值问题`

**一例速记**：
> **极值**：$x_0$ 是极值点 $\Leftrightarrow f'(x_0) = 0$ 且 $f'$ 在 $x_0$ 两侧变号  
> **最值**：闭区间 $[a, b]$ 上的最值 = max/min{极值点 + 端点 $f(a), f(b)$}

**引入题**：求 $f(x) = x^3 - 3x$ 在 $[-2, 3]$ 上的最大值最小值。

**思维路径还原** 15-20 行：求 $f'$ → 解 $f' = 0$ → 判断极值 → 比较端点。

**3 类常见题型**：
① 闭区间最值（标准 4 步）
② **含参极值**（如 $f(x) = x^3 - ax$ 极值随 $a$ 变化）
③ **实际优化问题**（如箱子表面积最小 / 利润最大）

**思考路标 ≥8 条**：
- $f'(x_0) = 0$ 是极值的**必要不充分**条件
- 必须验证两侧变号
- 闭区间最值 = 极值 + 端点比较
- 实际问题：先建模型 → 求导 → 验证范围

例题 3 例（含实际应用）+ 自测题 5 题。

### Task 5: part10/05 切线问题（模板 C，2500-3500 字）

Title: `# 切线方程与切线问题`

涵盖：
- **过曲线上一点 $(x_0, f(x_0))$ 的切线**：$y - f(x_0) = f'(x_0)(x - x_0)$
- **过曲线外一点 $(a, b)$ 的切线**：设切点 $(t, f(t))$，列方程 $b - f(t) = f'(t)(a - t)$ 求 $t$
- 注意"在……处的切线"与"过……的切线"差别
- 公切线问题（两条曲线的公切线）
- 典型应用 3 例 + 自测题 5 题

### Task 6: part10/06 导数与零点 ⭐（模板 D）

Title: `# 用导数研究函数零点与方程根`

**一例速记**：
> 函数 $f(x)$ 零点个数 = $y = f(x)$ 与 $x$ 轴交点个数  
> 含参方程根的个数 = $y = $ 含 $x$ 部分与 $y = $ 含参数部分（水平线）交点

**引入题**：讨论方程 $\ln x - kx = 0$（$k > 0$）的根的个数。

**思维路径还原** 15-20 行：参分 $k = \frac{\ln x}{x}$ → 研究 $g(x) = \frac{\ln x}{x}$ → 求导 / 极值 / 单调 → 数交点。

**3 类常见题型**：
① 无参情形：用单调性 + 中间值 / IVT
② **含参参数分离**：转化为 $a = h(x)$ 求 $h(x)$ 值域
③ 含参原型不能分离：分类讨论 $a$ 取值

**思考路标 ≥8 条**：
- 能参分尽量参分
- 不能分离 → 分类讨论 $a$
- 极值的符号 = 跨零点的关键
- 看到"恰好 $k$ 个根" → 极值与 $k$ 关系
- 端点（含极限端点）行为也要分析

例题 3 例（含参分典型）+ 自测题 5 题。

### Task 7: part10/07 导数综合应用（模板 C，2500-3500 字）

Title: `# 导数综合应用`

涵盖：
- **不等式证明**（用导数）：
  - 经典：$\ln(1+x) < x$（$x > 0$），$e^x > 1 + x$（$x \neq 0$）
  - 构造法：证 $f(x) > g(x)$ → 构造 $h(x) = f(x) - g(x)$，证 $h(x) > 0$
- **恒成立 / 存在性**：参数分离思想
- 函数综合（与不等式 / 数列 / 三角结合）
- 典型应用 3 例 + 自测题 5 题

---

## Task 8: GM7 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' gaozhong_math/algebra/part10-derivatives
ls gaozhong_math/algebra/part10-derivatives
printf '\n---\n**GM7 完成于：YYYY-MM-DD**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM7.md
git add docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM7.md
git commit -m "docs(gaozhong/algebra): mark GM7 milestone complete"
git push origin master
```

---

## 调度

**3 个 sonnet subagent 并行：**
- A: 01 概念、02 法则、03 ⭐ 单调
- B: 04 ⭐ 极值、05 切线
- C: 06 ⭐ 零点、07 综合

每章独立 commit。

---
**GM7 完成于：2026-05-13**
