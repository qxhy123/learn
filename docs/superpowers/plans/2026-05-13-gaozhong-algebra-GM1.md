# 高中代数教程 GM1 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development

**Goal:** 完成 Part 1 集合与逻辑（5 章）+ Part 2 不等式（4 章），共 9 章。

**Architecture:** 4 章模板 C + **5 章模板 D**（量词否定、充要条件、二次不等式、分式不等式、基本不等式）。每章独立提交。

**Spec 参考:** `docs/superpowers/specs/2026-05-13-gaozhong-algebra-design.md`
**前序里程碑:** GM0（已完成）

---

## 文件结构

```
gaozhong_math/algebra/
├── part1-sets-and-logic/                  # 集合与常用逻辑 [高一上]
│   ├── 01-sets-basics.md
│   ├── 02-set-operations.md
│   ├── 03-subset-and-power-set.md
│   ├── 04-logic-quantifiers.md            # ⭐ 全称/存在量词与否定
│   └── 05-necessary-sufficient.md         # ⭐ 充分必要条件
└── part2-inequalities/                    # 不等式 [高一上]
    ├── 01-inequality-basics.md
    ├── 02-quadratic-inequality.md         # ⭐ 一元二次不等式
    ├── 03-fractional-inequality.md        # ⭐ 分式不等式
    └── 04-basic-inequality.md             # ⭐ 基本不等式
```

模板 C 6 节，**2200-3200 字**；模板 D 7 节（一例速记 + 引入 + 思维路径还原 15-20 行 + 抽象方法 + 变形 + 思考路标 ≥8 条 + 例题 + 自测），**3500-5000 字**。LaTeX 全数学。

---

## Task 1: part1/01 集合基础（模板 C，2200-3200 字）

Title: `# 集合的基本概念`

涵盖：
- **集合**定义（确定 / 互异 / 无序 三大特征）
- **元素与集合的关系**：$\in$ 属于、$\notin$ 不属于
- **集合的表示法**：列举法 $\{1,2,3\}$、描述法 $\{x \mid P(x)\}$、图形法（Venn 图）
- **常用数集**：$\mathbb{N}$ 自然数、$\mathbb{N}^*$ / $\mathbb{N}_+$ 正整数、$\mathbb{Z}$ 整数、$\mathbb{Q}$ 有理数、$\mathbb{R}$ 实数
- **空集**：$\varnothing$ 没有元素的集合
- **典型应用** 3 例：
  - 例 1：用描述法表示 $\{x \mid x^2 - 5x + 6 = 0\}$
  - 例 2：判断 $\{1, 2\}$ 与 $\{2, 1\}$ 是否相同（**集合无序**，相同）
  - 例 3：若 $a \in A = \{1, a, a^2\}$，求 $a$ 的可能值（注意元素**互异性**：$a \ne 1, a \ne a^2$ → $a = -1$ 或某值）
- **易错点**：元素互异性必须验证；$\varnothing$ 与 $\{\varnothing\}$ 不同；集合中元素可以是任何对象
- **自测题** 4 题

Commit: `docs(gaozhong/algebra): add part1/01 (sets basics)`

---

## Task 2: part1/02 集合运算（模板 C，2200-3200 字）

Title: `# 集合的基本运算`

涵盖：
- **交集** $A \cap B = \{x \mid x \in A \text{ 且 } x \in B\}$
- **并集** $A \cup B = \{x \mid x \in A \text{ 或 } x \in B\}$
- **补集** $\complement_U A$（在全集 $U$ 中）
- **运算律**：交换律、结合律、分配律、德摩根律 $\complement(A \cup B) = \complement A \cap \complement B$
- **Venn 图直观理解**
- **典型应用** 3 例：
  - 例 1：基础——$A = \{1,2,3\}, B = \{2,3,4\}$，求 $A \cap B$、$A \cup B$
  - 例 2：含参集合 $A = \{x \mid 1 \leq x \leq a\}$，$B = \{x \mid x \leq 3\}$，求 $A \cap B \ne \varnothing$ 时 $a$ 范围
  - 例 3：德摩根律应用
- **易错点**：$A \cap B = \varnothing$ 时不要漏写 $\varnothing$；区间表示与不等式表示要清楚
- **自测题** 4 题

Commit: `docs(gaozhong/algebra): add part1/02 (set operations)`

---

## Task 3: part1/03 子集与幂集（模板 C，2200-3200 字）

Title: `# 子集、真子集与幂集`

- **子集** $A \subseteq B$：$A$ 的每个元素都在 $B$ 中
- **真子集** $A \subsetneq B$：$A \subseteq B$ 且 $A \ne B$
- **空集是任何集合的子集**
- **集合相等** $A = B$ ⇔ $A \subseteq B$ **且** $B \subseteq A$
- **幂集** $2^A$：$A$ 的所有子集组成的集合；$|A| = n$ 时 $|2^A| = 2^n$
- **典型应用** 3 例：
  - 例 1：写出 $\{a, b, c\}$ 的所有子集和真子集
  - 例 2：含参——$A = \{x \mid x^2 - 1 = 0\}, B = \{x \mid mx = 1\}$，$B \subseteq A$ 时 $m$ 取值（注意 $B = \varnothing$ 也成立 → $m = 0$；以及 $m = 1$（$B = \{1\}$）、$m = -1$（$B = \{-1\}$））
  - 例 3：证明 $A = B$（双向包含）
- **易错点**：$\varnothing$ 是所有集合的子集（含参题中必讨论 $B = \varnothing$ 的情形）；子集 vs 真子集区别
- **自测题** 4 题

Commit: `docs(gaozhong/algebra): add part1/03 (subset and power set)`

---

## Task 4: part1/04 全称/存在量词与否定 ⭐（模板 D，3500-5000 字）

Title: `# 全称量词、存在量词与命题否定`

**一例速记**：
> $\forall x \in M, p(x)$ 的否定是 $\exists x \in M, \neg p(x)$；$\exists x \in M, p(x)$ 的否定是 $\forall x \in M, \neg p(x)$。**变更量词、否定结论、保留范围**。

### 一、引入题
> 写出"对所有 $x \in \mathbb{R}$，$x^2 + 1 > 0$"的否定。

### 二、思维路径还原（15-20 行 `>` 引用）
> "原命题：$\forall x \in \mathbb{R}, x^2 + 1 > 0$ —— 是**全称命题**  
> 否定的核心：**找一个反例**就够了 → 把"对所有"改成"存在"  
> 同时**否定结论部分**：$x^2 + 1 > 0$ 的否定是 $x^2 + 1 \leq 0$（不是 $x^2 + 1 < 0$！）  
> 最终否定：$\exists x \in \mathbb{R}, x^2 + 1 \leq 0$  
> 验证：原命题为真 → 否定为假（实际找不到 $x$ 使 $x^2 + 1 \leq 0$）✓  
> **关键反射**：见 $\forall \to \exists$，见 $\exists \to \forall$；见 $>$ 否定为 $\leq$；见 $=$ 否定为 $\ne$  
> 范围保留：原命题里"$x \in \mathbb{R}$"是约束，否定时**保留不变**  
> 易错：含'且'和'或'的否定要用德摩根：$\neg(p \wedge q) = \neg p \vee \neg q$  
> 完整链：'对所有 X 都有性质 P' 否定 = '存在某个 X 使 P 不成立'  
> 高考爱考：写出'$\forall n \in \mathbb{N}^*, n^2 + n + 1$ 为奇数'的否定 → '$\exists n \in \mathbb{N}^*, n^2 + n + 1$ 为偶数'"

### 三、抽象成方法

**3 步否定法**：
1. 变更量词：$\forall \to \exists$，$\exists \to \forall$
2. 否定结论部分：用反义符号表替换
3. 范围保留

**反义符号表**（必背）：

| 原 | 否定 |
|---|---|
| $=$ | $\ne$ |
| $>$ | $\leq$ |
| $<$ | $\geq$ |
| $\geq$ | $<$ |
| $\leq$ | $>$ |
| $\in$ | $\notin$ |
| 且（$\wedge$） | 或（$\vee$） |
| 或（$\vee$） | 且（$\wedge$） |

### 四、方法变形
- **含两个量词的命题**：$\forall x \exists y, P(x,y)$ → $\exists x \forall y, \neg P(x,y)$
- **含参全称命题**：把"对所有 $x$" 当成约束
- **不等式恒成立 / 存在性**：$\forall x \in D, f(x) > 0$ 等价于 $\min_{x \in D} f(x) > 0$

### 五、思考路标（≥8 条）
- 见 $\forall$ → 否定后变 $\exists$
- 见 $\exists$ → 否定后变 $\forall$
- 否定结论时按反义符号表精确替换
- "且"否定后变"或"（德摩根）
- "或"否定后变"且"（德摩根）
- 范围不变，只变量词和结论
- $\forall$ 命题转化 = 恒成立 → 求最值
- $\exists$ 命题转化 = 存在性 → 与最值比

### 六、应用例题 3 例
- 例 1：写出"对所有 $x \geq 0, \sqrt{x} \geq 0$"的否定
- 例 2：含两个量词 "$\exists x \in \mathbb{R}, \forall y \in \mathbb{R}, x + y > 0$" 的否定
- 例 3：含"且"的命题否定

### 七、自测题 5 题

Commit: `docs(gaozhong/algebra): add part1/04 (quantifiers and negation)`

---

## Task 5: part1/05 充分必要条件 ⭐（模板 D，3500-5000 字）

Title: `# 充分条件、必要条件与充要条件`

**一例速记**：
> $p \Rightarrow q$（$p$ 真则 $q$ 真）→ $p$ 是 $q$ 的**充分**条件、$q$ 是 $p$ 的**必要**条件  
> $p \Leftrightarrow q$（双向）→ $p$ 是 $q$ 的**充要**条件  
> **集合视角**：$\{p\} \subseteq \{q\}$ ⇔ $p \Rightarrow q$

### 一、引入题
> 设 $p: x > 1$，$q: x > 0$。问 $p$ 是 $q$ 的什么条件？

### 二、思维路径还原（15-20 行）
> "原题：$p: x > 1$，$q: x > 0$  
> **方法 1（逻辑判断）**：若 $p$ 真即 $x > 1$，则 $x > 0$ 显然 → $p \Rightarrow q$ ✓  
> 反向：若 $q$ 真即 $x > 0$，是否一定 $x > 1$？反例 $x = 0.5$ 时 $q$ 真但 $p$ 假 → $q \not\Rightarrow p$  
> 所以 $p$ 是 $q$ 的**充分不必要条件**  
> **方法 2（集合视角）**：$P = \{x \mid x > 1\}$，$Q = \{x \mid x > 0\}$  
> 显然 $P \subsetneq Q$（$P$ 是 $Q$ 的真子集）→ $p \Rightarrow q$，且 $q \not\Rightarrow p$  
> 结论：充分不必要  
> **集合视角是高考解题最快路径**——画出两个范围一比就出来。  
> 4 种情形对应：  
>   - $P \subsetneq Q$：充分不必要  
>   - $Q \subsetneq P$：必要不充分  
>   - $P = Q$：充要  
>   - $P, Q$ 不互相包含：既不充分也不必要"

### 三、抽象成方法
**4 种情形**（用表格）：

| $p$ 与 $q$ 集合关系 | $p$ 是 $q$ 的什么条件 |
|---|---|
| $P \subsetneq Q$（$P$ 真子集 $Q$）| 充分不必要 |
| $Q \subsetneq P$ | 必要不充分 |
| $P = Q$ | 充要 |
| $P, Q$ 互不包含 | 既不充分也不必要 |

**两种判断方法**：
1. **逻辑推断**：从 $p$ 推 $q$、从 $q$ 推 $p$，看哪边成立
2. **集合视角**：化为不等式 / 范围，比较集合包含关系

### 四、方法变形
- **含参题**：$p$ 是 $q$ 的充分条件 → $P \subseteq Q$ → 列范围比较得参数
- **反向使用**：题给"$p$ 是 $q$ 的必要不充分条件"，求参数 → $Q \subsetneq P$
- **等价条件证明**：证 $p \Leftrightarrow q$ 等价于证 $p \Rightarrow q$ **且** $q \Rightarrow p$（双向）

### 五、思考路标（≥8 条）
- 见"$p$ 是 $q$ 的什么条件" → 集合视角最快
- 化为 $P = \{x \mid p(x)\}, Q = \{x \mid q(x)\}$
- $P \subsetneq Q$：充分不必要
- $P = Q$：充要（注意是真等）
- 含参题 → $P \subseteq Q$ → 列范围不等式
- 充要 = 双向蕴含 = 集合相等
- 否命题与逆否命题的关系（原 ⇔ 逆否）
- 常见混淆："充分" vs "必要" 搞反——记住箭头方向：$p \Rightarrow q$ → $p$ 充分

### 六、应用例题 3 例
- 例 1：判断 $p: a > b$ 是 $q: a^2 > b^2$ 的什么条件（既不充分也不必要——反例 $a=1, b=-2$）
- 例 2：含参——$p: 0 < x < 3$，$q: 0 < x < a$，求 $p$ 是 $q$ 的充分不必要条件时 $a$ 的范围（$a > 3$）
- 例 3：充要条件证明（双向）

### 七、自测题 5 题

Commit: `docs(gaozhong/algebra): add part1/05 (necessary sufficient conditions)`

---

## Task 6: part2/01 不等式基础（模板 C，2200-3200 字）

Title: `# 不等式的性质`

涵盖：
- **大小比较**的三大方法：作差比较 / 作商比较（注意符号）/ 函数性质
- **不等式 8 大性质**：
  1. 对称性 $a > b \Leftrightarrow b < a$
  2. 传递性 $a > b, b > c \Rightarrow a > c$
  3. 同向不等式相加 $a > b, c > d \Rightarrow a + c > b + d$
  4. 异向不等式相减 $a > b, c < d \Rightarrow a - c > b - d$
  5. 同向同号不等式相乘
  6. 平方 / 开方（注意条件）
  7. 倒数：$a > b > 0 \Rightarrow \frac{1}{a} < \frac{1}{b}$
  8. 移项性
- **典型应用** 3 例：
  - 例 1：比较 $a^2 + b^2$ 与 $2ab$（作差 = $(a-b)^2 \geq 0$）
  - 例 2：已知 $-1 < a + b < 3, 2 < a - b < 4$，求 $2a + b$ 范围（用不等式性质组合）
  - 例 3：判断命题真假（如"$a > b \Rightarrow ac > bc$"——错，要求 $c > 0$）
- **易错点**：作商比较要分母符号；不等式相乘必须同向同号；倒数变号
- **自测题** 4 题

Commit: `docs(gaozhong/algebra): add part2/01 (inequality basics)`

---

## Task 7: part2/02 一元二次不等式 ⭐（模板 D，3500-5000 字）

Title: `# 一元二次不等式`

**一例速记**：
> $ax^2 + bx + c > 0$（$a > 0$）：① 求根 $x_1, x_2$（$\Delta > 0$ 时）② 写出"两根之外" $x < x_1$ 或 $x > x_2$；类似 $< 0$ 写"两根之间" $x_1 < x < x_2$。**$a < 0$ 先变号**。

### 一、引入题
> 解不等式 $x^2 - 5x + 6 > 0$。

### 二、思维路径还原（15-20 行 `>` 引用）
> "$x^2 - 5x + 6 > 0$ —— 标准二次三项式 $a > 0$  
> 求根：$x^2 - 5x + 6 = (x-2)(x-3) = 0$ → $x_1 = 2, x_2 = 3$  
> 抛物线 $y = x^2 - 5x + 6$ 开口向上，与 $x$ 轴交于 $(2, 0), (3, 0)$  
> 求 $y > 0$ 的 $x$ 范围 = 图象在 $x$ 轴**上方**部分 = 两根**之外**  
> 答：$x < 2$ 或 $x > 3$  
> **关键反射**：开口向上 + 求 $> 0$ → 两根之外；开口向上 + 求 $< 0$ → 两根之间  
> 若 $a < 0$（开口向下），**先两边乘 -1 变号**，转为 $a > 0$ 的情形  
> $\Delta < 0$（无根）：抛物线整体在 $x$ 轴一侧，要么恒 $> 0$ 要么恒 $< 0$  
> $\Delta = 0$（重根）：抛物线与 $x$ 轴相切于一点，$> 0$ 解集是除该点外全实数  
> 一切归到图象——数形结合是二次不等式的核心"

### 三、抽象成方法
**5 步标准流程**：
1. 化为 $a x^2 + bx + c > 0$（$a > 0$）形式
2. 求 $\Delta$ 判别根的情况
3. $\Delta > 0$ 求两根 $x_1 < x_2$
4. 根据不等号方向写解集（用"两根之外 / 两根之间"口诀）
5. 检验端点是否取（$> $ 不取，$\geq$ 取）

**4 种情形对照表**（$a > 0$）：

| $\Delta$ | $f(x) > 0$ 解集 | $f(x) < 0$ 解集 |
|---|---|---|
| $> 0$ | $x < x_1$ 或 $x > x_2$ | $x_1 < x < x_2$ |
| $= 0$ | $x \ne x_0$ | $\varnothing$ |
| $< 0$ | $\mathbb{R}$ | $\varnothing$ |

### 四、方法变形
- **含参二次不等式**：按 $a$、$\Delta$、根的位置分类讨论
- **恒成立问题**：$ax^2+bx+c > 0$ 对所有 $x$ 恒成立 → $a > 0$ 且 $\Delta < 0$（或 $a = b = 0, c > 0$）

### 五、思考路标（≥8 条）
- 见二次不等式 → 第一步求根 + 判别开口
- $a > 0$ + $> 0$ → 两根之外
- $a > 0$ + $< 0$ → 两根之间
- $a < 0$ → 先两边乘 $-1$ 变号
- $\Delta < 0$ → 抛物线整体在轴一侧
- $\Delta = 0$ → 一个切点
- 含参 → 按 $a, \Delta$、根的位置分类
- 恒成立 → $a$ 和 $\Delta$ 联合约束

### 六、应用例题 3 例
- 例 1：解 $-x^2 + 3x + 4 > 0$（先变号 → $x^2 - 3x - 4 < 0$ → 两根之间）
- 例 2：恒成立题 —— $kx^2 + 2kx + 4 > 0$ 对所有 $x$ 恒成立，求 $k$（$k = 0$ 或 $k > 0$ 且 $\Delta < 0$ → $0 \leq k < 4$）
- 例 3：含参 —— $x^2 - (a+1)x + a < 0$ 按 $a$ 分类

### 七、自测题 5 题

Commit: `docs(gaozhong/algebra): add part2/02 (quadratic inequality)`

---

## Task 8: part2/03 分式不等式 ⭐（模板 D，3500-5000 字）

Title: `# 分式不等式`

**一例速记**：
> $\frac{f(x)}{g(x)} > 0$ ⇔ $f(x) \cdot g(x) > 0$（且 $g(x) \ne 0$）  
> $\frac{f(x)}{g(x)} \geq 0$ ⇔ $f(x) \cdot g(x) \geq 0$ **且** $g(x) \ne 0$  
> **化为乘积不等式 + 注意分母不为零**

### 一、引入题
> 解不等式 $\frac{x-1}{x+2} \geq 0$。

### 二、思维路径还原（15-20 行）
> "$\frac{x-1}{x+2} \geq 0$ —— 分式不等式标准型  
> **错误做法**：两边乘 $(x+2)$ → 不知道 $x+2$ 正负、不等号方向不定  
> **正确做法**：化为**乘积不等式**：$(x-1)(x+2) \geq 0$ **且** $x + 2 \ne 0$  
> 求 $(x-1)(x+2) \geq 0$：两根 $-2, 1$（开口向上）→ 两根之外（取 $=$）→ $x \leq -2$ 或 $x \geq 1$  
> 再加约束 $x \ne -2$ → 最终 $x < -2$ 或 $x \geq 1$  
> **关键反射**：分式 → 化乘积，但分母不能为零（即使是 $\geq$ 也要排除）  
> 化为乘积后用二次不等式套路解  
> 注意：原分式不等式 $\geq 0$ 时，分子可以 $= 0$（值为 0 满足 $\geq$），但分母不能 $= 0$  
> 更复杂的：含 $\frac{f}{g} > k$ 形式 → 先移项化为 $\frac{f - kg}{g} > 0$ 再化乘积"

### 三、抽象成方法
**4 步流程**：
1. **化为 $\frac{f}{g}$ 与 0 比较**（不要直接乘分母）：$\frac{f}{g} > k$ → 移项化为 $\frac{f - kg}{g} > 0$
2. **化为乘积**：$\frac{f}{g} > 0$ ⇔ $f \cdot g > 0$
3. **解乘积不等式**（用 part2/02 二次不等式或符号分析）
4. **排除分母为零**

**"$>$" "$\geq$" 区别**（用表格）：

| 原 | 化为乘积 | 分母约束 |
|---|---|---|
| $\frac{f}{g} > 0$ | $fg > 0$ | $g \ne 0$（自动满足，因为 $fg > 0$ 时 $g \ne 0$）|
| $\frac{f}{g} \geq 0$ | $fg \geq 0$ | **$g \ne 0$ 必须额外加** |

### 四、方法变形
- **三因子分式** $\frac{(x-a)(x-b)}{x-c} > 0$ → 化为 $(x-a)(x-b)(x-c) > 0$ + $x \ne c$，用**穿针引线法**（高考神技）
- **含参** → 分类讨论根的大小
- **不等式右边非 0** → 先移项化为右边 0

### 五、思考路标（≥8 条）
- 见分式不等式 → **不直接乘分母**
- 化为乘积形式
- 分母 $\ne 0$ 必须显式约束
- $\geq$ 要分别考虑分子 = 0 时和分母 = 0 时（前者可，后者不可）
- 多因子用穿针引线法
- 含参 → 按根大小分类
- 右边非 0 → 先移项 + 通分
- 验证：代入特殊值检查

### 六、应用例题 3 例
- 例 1：解 $\frac{2x-1}{x+3} < 1$（先移项化为 $\frac{x-4}{x+3} < 0$ → $-3 < x < 4$）
- 例 2：穿针引线 —— $\frac{(x-1)(x+2)}{x-3} \leq 0$
- 例 3：含参 —— $\frac{x - a}{x + 1} \geq 0$ 按 $a$ 与 $-1$ 关系分类

### 七、自测题 5 题

Commit: `docs(gaozhong/algebra): add part2/03 (fractional inequality)`

---

## Task 9: part2/04 基本不等式 ⭐（模板 D，3500-5000 字）

Title: `# 基本不等式 $\frac{a+b}{2} \geq \sqrt{ab}$`

**一例速记**：
> $a, b > 0$ 时 $\frac{a+b}{2} \geq \sqrt{ab}$（即 $a + b \geq 2\sqrt{ab}$）**当且仅当 $a = b$ 时取等**  
> 算术-几何平均不等式（AM-GM）。**3 个使用条件：正数 / 求最值 / 取等条件**。

### 一、引入题
> 已知 $x > 0$，求 $f(x) = x + \frac{1}{x}$ 的最小值。

### 二、思维路径还原（15-20 行）
> "$x > 0$，求 $x + \frac{1}{x}$ 最小值  
> 见'$x + \frac{1}{x}$' → 立刻想基本不等式 $x + \frac{1}{x} \geq 2\sqrt{x \cdot \frac{1}{x}} = 2$  
> 验证 3 条件：①$x > 0, \frac{1}{x} > 0$ ✓ ②$x \cdot \frac{1}{x} = 1$ 是常数 ✓ ③ 取等 $x = \frac{1}{x}$ → $x = 1$ ✓  
> 答：最小值 $2$，在 $x = 1$ 时取到  
> **关键反射**：见 $a + b$ 形式 + 想求最值 → 检查 $ab$ 是否为常数 → 用基本不等式  
> '一正二定三相等'三原则缺一不可  
> 如果 $ab$ 非常数：先**配凑**让 $ab$ 为常数（如 $x + \frac{4}{x-1}$ → $x - 1 + \frac{4}{x-1} + 1$）  
> 反向应用：见 $ab = $ 常数，求 $a + b$ 最值 → 同样用基本不等式  
> '1' 的代换：$\frac{1}{x} + \frac{1}{y} = 1$ 求 $x + y$ 最小值 → 用 $x + y = (x+y) \cdot 1 = (x+y)(\frac{1}{x} + \frac{1}{y})$ 展开后用基本不等式  
> 取等条件**必须验证**，否则只能给出 $\geq$ 不能说"最小"是该值"

### 三、抽象成方法
**基本不等式 4 大变形**（必背）：
1. $a + b \geq 2\sqrt{ab}$（$a, b > 0$）→ 见和求积，或反向
2. $\sqrt{ab} \leq \frac{a+b}{2}$ → 同上
3. $a^2 + b^2 \geq 2ab$（无正数约束！）→ 见平方和
4. $\frac{a^2 + b^2}{2} \geq \left(\frac{a+b}{2}\right)^2$ → 平方平均 $\geq$ 算术平均

**"一正二定三相等"三原则**：
1. **一正**：变量必须为正
2. **二定**：和或积之一必须是常数
3. **三相等**：取等条件可达成

**"1" 的代换技巧**：$\frac{1}{x} + \frac{1}{y} = 1$ 时，$x + y = (x+y)(\frac{1}{x} + \frac{1}{y})$ 展开求最值。

### 四、方法变形
- **配凑法**：$x + \frac{4}{x-1}$ → 凑出 $x - 1 + \frac{4}{x-1} + 1 \geq 2\sqrt{4} + 1 = 5$（$x = 3$ 时取等）
- **平方法**：见 $\sqrt{a} + \sqrt{b}$ 求最大值 → 两边平方
- **柯西不等式**（高考拓展）：$(a^2+b^2)(c^2+d^2) \geq (ac+bd)^2$

### 五、思考路标（≥8 条）
- 见 $a + b$ 求最值 + 已知 $ab$ 为常数 → 基本不等式
- 见 $ab$ 求最值 + 已知 $a + b$ 为常数 → 基本不等式（反向）
- 见 $x + \frac{c}{x}$ → 直接套
- 见 $x + \frac{c}{x - k}$ → 配凑 $x - k$
- 见 $\frac{1}{x} + \frac{1}{y} = 常$ → '1' 代换技巧
- 见 $a^2 + b^2$ → 用 $a^2 + b^2 \geq 2ab$
- 取等条件**必须验证**
- 三原则缺一不可

### 六、应用例题 3 例
- 例 1：$x > 0$，求 $f(x) = x + \frac{4}{x}$ 最小值（4，$x = 2$ 时取）
- 例 2：'1' 代换 —— $\frac{2}{x} + \frac{1}{y} = 1$（$x, y > 0$），求 $x + 2y$ 最小值
- 例 3：配凑 —— $x > 1$，求 $f(x) = x + \frac{4}{x-1}$ 最小值

### 七、自测题 5 题

Commit: `docs(gaozhong/algebra): add part2/04 (basic inequality AM-GM)`

---

## Task 10: GM1 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' gaozhong_math/algebra/part1-sets-and-logic gaozhong_math/algebra/part2-inequalities
ls gaozhong_math/algebra/part1-sets-and-logic gaozhong_math/algebra/part2-inequalities
printf '\n---\n**GM1 完成于：YYYY-MM-DD**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM1.md
git add docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM1.md
git commit -m "docs(gaozhong/algebra): mark GM1 milestone complete"
git push origin master
```

---

## 调度

**2 个 sonnet subagent 并行：**
- A: part1/01-05（5 章，含 2 个 ⭐ 模板 D）
- B: part2/01-04（4 章，含 3 个 ⭐ 模板 D）

完成后立刻收尾推送。

---
**GM1 完成于：2026-05-13**
