# 附录 F1：基础题型详解（集合 / 逻辑 / 不等式 / 指对 / 三角 / 复数 / 统计）

> 本附录给出附录 D 中 **31 题** 中档题的完整解答，主题分布如下：
> - **Part 1 集合与含参** （D.01–D.05，共 $5$ 题）
> - **Part 1 逻辑量词与充要条件** （D.06–D.08，共 $3$ 题）
> - **Part 2 不等式与基本不等式** （D.09–D.15，共 $7$ 题）
> - **Part 4 指数对数运算** （D.22–D.24，共 $3$ 题）
> - **Part 5 三角化简、最值、解三角形** （D.25–D.30，共 $6$ 题）
> - **Part 6 复数四则与共轭** （D.31–D.33，共 $3$ 题）
> - **Part 7 统计：方差 / 相关系数 / 线性回归** （D.34–D.37，共 $4$ 题）
>
> **每题格式**：
> - **题目回顾**：再次列出题面（保证文档独立可读）；
> - **思路**：1–2 句点明套路 + 关键技巧 + toolkit / 章节引用；
> - **解答**：完整推导，关键步骤带 "← ..." 小标说明（如"← 换元"、"← 平方"）；
> - **答案**：用 $\boxed{}$ 框出最终结果；
> - **总结**：1 句"识题特征"——下次见这类题如何 1 秒识别。
>
> **引用说明**：
> - toolkit 引用：→ toolkit/01 结构识别、→ toolkit/02 换元、→ toolkit/03 构造、→ toolkit/04 数形结合、→ toolkit/05 参数策略、→ toolkit/11 分类讨论、→ toolkit/12 解题前清单；
> - 章节引用形如 → Part 2/04 基本不等式（即对应章节）。

---

## Part 1：集合与含参（D.01–D.05）

> 本组涉及"含参集合的包含关系"，核心思路：
> 1. **先解二次不等式**，定出固定集合（不含参的那个）的具体区间。
> 2. **写出含参集合的形态**，注意参数取值导致集合空 / 单 / 双元素的临界变化。
> 3. **集合等式翻译**：$A \cap B = A \Leftrightarrow A \subseteq B$、$A \cup B = A \Leftrightarrow B \subseteq A$、$A \cap B = B \Leftrightarrow B \subseteq A$。
> 4. **判 $B$ 是否可能为空**：含参区间 $[\alpha(m), \beta(m)]$ 若 $\alpha > \beta$ 即空。空集是任何集合的子集，必须单独列入分类讨论。

---

## D.01 [中档] Part 1/02

**题目回顾**：已知集合 $A = \{x \mid x^2 - 3x - 4 \leq 0\}$，$B = \{x \mid m \leq x \leq m + 3\}$。若 $B \subseteq A$，求实数 $m$ 的取值范围。

**思路**　先解二次不等式定 $A$ 的具体区间；再写出 $B$ 子集条件。注意 $B$ 长度固定为 $3$，恒非空（端点 $m \leq m+3$ 显然），无需讨论空集情形。→ toolkit/05 参数策略。

**解答**

**第一步**：解二次不等式确定 $A$。

$x^2 - 3x - 4 = (x - 4)(x + 1) \leq 0$，两根为 $-1, 4$，开口向上 ← 介于两根之间，

$\Rightarrow -1 \leq x \leq 4$，故 $A = [-1, 4]$。

**第二步**：写出 $B$ 与子集条件。

$B = [m, m+3]$（长度恒为 $3$，恒非空）。

$B \subseteq A$ $\Leftrightarrow$ $B$ 的两端点都落在 $A$ 内（端点都满足 $\Rightarrow$ 区间全部在内）：
$$\begin{cases} m \geq -1 \\ m + 3 \leq 4 \end{cases} \Rightarrow \begin{cases} m \geq -1 \\ m \leq 1 \end{cases}$$

**答案**：$\boxed{-1 \leq m \leq 1}$。

**总结**　"区间型子集"模板：$[m, m+L] \subseteq [a, b]$ $\Leftrightarrow$ $m \geq a$ 且 $m + L \leq b$。两端点同时管，长度固定时不必讨论空集。

---

## D.02 [中档] Part 1/02

**题目回顾**：设全集 $U = \mathbb{R}$，$A = \{x \mid x^2 - 5x + 4 < 0\}$，$B = \{x \mid x - a > 0\}$。若 $A \cap (\complement_U B) = A$，求 $a$ 的取值范围。

**思路**　集合等式 → 包含关系：$A \cap (\complement_U B) = A$ 表示"$A$ 与 $\complement_U B$ 取交后恢复为 $A$ 整体" $\Leftrightarrow A \subseteq \complement_U B$。再写区间，比较端点。→ toolkit/01 结构识别。

**解答**

**第一步**：定 $A$。

$x^2 - 5x + 4 = (x - 1)(x - 4) < 0 \Rightarrow 1 < x < 4$，故 $A = (1, 4)$。

**第二步**：定 $\complement_U B$。

$B = \{x \mid x - a > 0\} = (a, +\infty)$，

$\complement_U B = (-\infty, a]$（注意全集 $\mathbb{R}$ 取补，$B$ 开 → 补集闭于 $a$）。

**第三步**：化集合关系并解。

$A \cap (\complement_U B) = A \Leftrightarrow A \subseteq \complement_U B$

$\Leftrightarrow$ 对一切 $x \in (1, 4)$ 都有 $x \leq a$

$\Leftrightarrow a \geq \sup A = 4$（$A$ 是开区间，$4 \notin A$，"$x < 4 \leq a$" 中 $a$ 可取等号 $4$）。

**答案**：$\boxed{a \geq 4}$。

**总结**　集合等式标准翻译：$X \cap Y = X \Leftrightarrow X \subseteq Y$；$X \cup Y = X \Leftrightarrow Y \subseteq X$。补集端点开闭要看原集合的端点是开还是闭——反过来。

---

## D.03 [中档] Part 1/03

**题目回顾**：已知集合 $A = \{x \mid -2 \leq x \leq 5\}$，$B = \{x \mid m+1 \leq x \leq 2m - 1\}$。若 $A \cap B = B$，求 $m$ 的取值范围。

**思路**　$A \cap B = B$ 与 $A \cup B = A$ 等价 $\Leftrightarrow B \subseteq A$。$B$ 含参区间形态依赖 $m$：当 $m + 1 > 2m - 1$ 即 $m < 2$ 时 $B = \varnothing$（空集是任何集合的子集），必须分类。→ toolkit/11 分类讨论。

**解答**

集合等式化为包含关系：$A \cap B = B \Leftrightarrow B \subseteq A$。

**情况 1**（$B = \varnothing$）：

$B$ 空 $\Leftrightarrow m + 1 > 2m - 1 \Leftrightarrow m < 2$。空集是任何集合的子集 ✓。

**情况 2**（$B \neq \varnothing$）：

非空 $\Leftrightarrow m + 1 \leq 2m - 1 \Leftrightarrow m \geq 2$。

$B \subseteq A$ $\Leftrightarrow B$ 两端点都在 $A$ 内：
$$\begin{cases} m + 1 \geq -2 \\ 2m - 1 \leq 5 \end{cases} \Rightarrow \begin{cases} m \geq -3 \\ m \leq 3 \end{cases}$$

结合 $m \geq 2$，得 $2 \leq m \leq 3$。

**合并**：$m < 2$ 或 $2 \leq m \leq 3$ $\Rightarrow m \leq 3$。

**答案**：$\boxed{m \leq 3}$。

**总结**　"$B \subseteq A$" 必查 $B$ 是否可能为空。含参区间 $B = [\alpha(m), \beta(m)]$，空集判据是 $\alpha(m) > \beta(m)$。分类合并答案时端点不要遗漏。

---

## D.04 [中档] Part 1/02

**题目回顾**：设 $A = \{x \mid x^2 + 4x = 0\}$，$B = \{x \mid x^2 + 2(a+1)x + a^2 - 1 = 0\}$。若 $A \cup B = A$，求 $a$ 的取值范围。

**思路**　$A \cup B = A \Leftrightarrow B \subseteq A$（并集恢复 $A$ 等价于 $B$ 没有 $A$ 之外的元素）；$B$ 是二次方程的解集，按判别式 $\Delta$ 分空 / 单 / 双根三种情形讨论。→ toolkit/11 分类。

**解答**

**第一步**：定 $A$。$x^2 + 4x = x(x + 4) = 0 \Rightarrow A = \{0, -4\}$。

**第二步**：分析 $B$ 的判别式。方程 $x^2 + 2(a+1)x + a^2 - 1 = 0$，

$\Delta = [2(a+1)]^2 - 4(a^2 - 1) = 4(a+1)^2 - 4(a^2 - 1)$
$= 4[(a+1)^2 - (a^2 - 1)] = 4[a^2 + 2a + 1 - a^2 + 1] = 4(2a + 2) = 8(a + 1)$。

**第三步**：分类讨论。

**情况 1**（$B = \varnothing$，即 $\Delta < 0$）：$8(a + 1) < 0 \Rightarrow a < -1$。此时 $B$ 是空集，自然 $B \subseteq A$ ✓。

**情况 2**（$B$ 单元素，即 $\Delta = 0$）：$a = -1$，方程化为 $x^2 + 0 + 0 = 0 \Rightarrow x = 0$，$B = \{0\}$，$B \subseteq A$ ✓。

**情况 3**（$B$ 双元素，即 $\Delta > 0$，$a > -1$）：要求 $B$ 的两根均在 $\{0, -4\}$ 中，即 $B = \{0, -4\} = A$。

由韦达定理（设根为 $x_1, x_2$）：
$$x_1 + x_2 = -2(a+1) = 0 + (-4) = -4 \Rightarrow a + 1 = 2 \Rightarrow a = 1.$$
$$x_1 x_2 = a^2 - 1 = 0 \cdot (-4) = 0 \Rightarrow a^2 = 1 \Rightarrow a = \pm 1.$$

← 两式必须同时成立 $\Rightarrow a = 1$；验证 $a = 1$ 时 $\Delta = 16 > 0$ ✓ 且方程化为 $x^2 + 4x = 0$ 解为 $0, -4$ ✓。

**第四步**：合并三种情况，$a \leq -1$ 或 $a = 1$。

**答案**：$\boxed{a \leq -1 \text{ 或 } a = 1}$。

**总结**　"二次方程的解集 $B \subseteq$ 双元素集 $A$"标准三分：$\Delta < 0$（$B$ 空）、$\Delta = 0$（$B$ 单元素必在 $A$ 中）、$\Delta > 0$（$B = A$，韦达双校验）。一定不能漏空集！

---

## D.05 [中档] Part 1/03

**题目回顾**：已知非空集合 $A = \{x \mid 2a + 1 \leq x \leq 3a - 5\}$，$B = \{x \mid 3 \leq x \leq 22\}$。若 $A \subseteq B$，求 $a$ 的取值范围。

**思路**　题目明确声明 $A$ 非空 → 必须把"$A$ 非空"和"$A \subseteq B$"两组约束同时列入。$A$ 非空给出 $a$ 的下界、$A \subseteq B$ 给出端点不等式，三式联立。→ toolkit/05 参数策略。

**解答**

**第一步**：$A$ 非空。

$2a + 1 \leq 3a - 5 \Rightarrow -a \leq -6 \Rightarrow a \geq 6$。

**第二步**：$A \subseteq B$（$A$ 两端点都在 $B$ 内）。

$$\begin{cases} 2a + 1 \geq 3 \\ 3a - 5 \leq 22 \end{cases} \Rightarrow \begin{cases} 2a \geq 2 \\ 3a \leq 27 \end{cases} \Rightarrow \begin{cases} a \geq 1 \\ a \leq 9 \end{cases}$$

**第三步**：综合三个不等式 $a \geq 6, a \geq 1, a \leq 9$：

$\boxed{6 \leq a \leq 9}$。

**总结**　题目明示"非空"——务必把"非空条件"列入约束，不能漏。容易因为只看 $A \subseteq B$ 漏掉 $a \geq 6$ 而把答案写成 $1 \leq a \leq 9$。

---

## Part 1：逻辑量词与充要条件（D.06–D.08）

> 本组核心技巧：
> 1. **充分必要 → 集合包含**：$p$ 充分 $q$ $\Leftrightarrow P \subseteq Q$；$p$ 充分不必要 $q$ $\Leftrightarrow P \subsetneq Q$；$p$ 必要不充分 $q$ $\Leftrightarrow Q \subsetneq P$。
> 2. **逆否换形**：$\neg p \Rightarrow \neg q \Leftrightarrow q \Rightarrow p$。涉及"$\neg$"的关系都可换为"$p, q$"原命题的关系。
> 3. **量词命题真假**：$\exists$ 假 $\Leftrightarrow \forall$ 反命题真；二次型恒成立常化为开口 + 判别式条件。

---

## D.06 [中档] Part 1/05

**题目回顾**：设 $p$：$\dfrac{1}{2} \leq x \leq 1$，$q$：$(x - a)(x - a - 1) \leq 0$。若 $p$ 是 $q$ 的充分不必要条件，求 $a$ 的取值范围。

**思路**　把 $p, q$ 化为集合 $P, Q$；"$p$ 充分不必要于 $q$" $\Leftrightarrow p \Rightarrow q$ 但 $q \not\Rightarrow p$ $\Leftrightarrow P \subsetneq Q$。再列端点不等式。→ toolkit/01 结构识别。

**解答**

**第一步**：写出 $P, Q$ 集合。

$P = \left[\dfrac{1}{2}, 1\right]$（长度 $\dfrac{1}{2}$）。

$q$：$(x - a)(x - a - 1) \leq 0$。两根为 $a$ 与 $a + 1$，由 $a < a + 1$ 知 $a$ 是小根，

$\Rightarrow a \leq x \leq a + 1$，即 $Q = [a, a + 1]$（长度 $1$）。

**第二步**：化集合关系。

"$p$ 充分不必要于 $q$" $\Leftrightarrow P \subsetneq Q$。

$P \subseteq Q$ 即 $P$ 两端点都在 $Q$ 内：
$$\begin{cases} a \leq \dfrac{1}{2} \\ a + 1 \geq 1 \end{cases} \Rightarrow \begin{cases} a \leq \dfrac{1}{2} \\ a \geq 0 \end{cases} \Rightarrow 0 \leq a \leq \dfrac{1}{2}.$$

**第三步**：检查真包含。

$P$ 长 $\dfrac{1}{2}$、$Q$ 长 $1$，长度不等 $\Rightarrow P \neq Q$ 永远成立 ← 真包含自动满足，无需额外排除端点。

**答案**：$\boxed{0 \leq a \leq \dfrac{1}{2}}$。

**总结**　"充分不必要 → 真包含"是核心翻译；不等式 $(x - a)(x - a - b) \leq 0$ 直接化为区间 $[a, a + b]$（$b > 0$）。两区间长度不同时，包含自动严格。

---

## D.07 [中档] Part 1/05

**题目回顾**：已知 $p$：实数 $x$ 满足 $x^2 - 4ax + 3a^2 < 0$（$a > 0$）；$q$：$x^2 - x - 6 \leq 0$ 且 $x^2 + 2x - 8 > 0$。若 $\neg p$ 是 $\neg q$ 的充分不必要条件，求 $a$ 的取值范围。

**思路**　"$\neg p$ 充分不必要 $\neg q$" 用逆否等价转 "$q$ 充分不必要 $p$"$\Leftrightarrow Q \subsetneq P$。再化集合关系列端点不等式。→ toolkit/05 参数策略。

**解答**

**第一步**：化简 $p$ 的解集 $P$。

$x^2 - 4ax + 3a^2 = (x - a)(x - 3a) < 0$。

由 $a > 0$，$a < 3a$，故 $P = (a, 3a)$（开口向上，小于零取两根之间）。

**第二步**：化简 $q$ 的解集 $Q$。

- 子条件 1：$x^2 - x - 6 = (x - 3)(x + 2) \leq 0 \Rightarrow -2 \leq x \leq 3$；
- 子条件 2：$x^2 + 2x - 8 = (x + 4)(x - 2) > 0 \Rightarrow x < -4 \text{ 或 } x > 2$。
- 取交集（"且"）：$[-2, 3] \cap [(-\infty, -4) \cup (2, +\infty)] = (2, 3]$。

故 $Q = (2, 3]$。

**第三步**：逆否换形。

"$\neg p$ 充分不必要 $\neg q$" $\Leftrightarrow \neg p \Rightarrow \neg q$ 但 $\neg q \not\Rightarrow \neg p$

$\Leftrightarrow q \Rightarrow p$ 但 $p \not\Rightarrow q$ ← 取逆否

$\Leftrightarrow Q \subsetneq P$。

**第四步**：列端点不等式 $Q = (2, 3] \subsetneq P = (a, 3a)$。

$Q \subseteq P$ 要求 $Q$ 的左端点 $> $ $P$ 的左端点 $a$（因 $P$ 是开区间，$2$ 处 $Q$ 是开 ← 实则 $Q$ 左开），且 $Q$ 右端点 $\leq P$ 右端点（$P$ 右开，$Q$ 右闭 $3 \in Q$ 必须 $3 < 3a$）：

$$\begin{cases} a \leq 2 \\ 3a > 3 \end{cases} \Rightarrow \begin{cases} a \leq 2 \\ a > 1 \end{cases}$$

← 左端：$a = 2$ 时 $P = (2, 6) \supset (2, 3] = Q$ ✓；右端：$3a > 3$ 必须严格（$3 \in Q$ 需在 $P$ 内即 $3 < 3a$）。

$Q \neq P$：长度不同 ← 永远是真子集，无需额外排除。

**答案**：$\boxed{1 < a \leq 2}$。

**总结**　"$\neg p$ 与 $\neg q$ 关系"→ 逆否换 "$p$ 与 $q$ 关系" 再化集合包含；含参集合的端点开闭与是否能取等逐点核对。

---

## D.08 [中档] Part 1/04

**题目回顾**：写出命题"$\exists x \in \mathbb{R},\ x^2 + 2ax + 4 \leq 0$"为假命题时，实数 $a$ 的取值范围。

**思路**　存在命题为假 $\Leftrightarrow$ 其否定（全称命题）为真 $\Leftrightarrow$ 对一切 $x \in \mathbb{R}$，$x^2 + 2ax + 4 > 0$。这是二次函数恒正问题：开口向上 + $\Delta < 0$。→ toolkit/04 数形结合。

**解答**

**第一步**：取否定转全称。

原命题"$\exists x \in \mathbb{R},\ x^2 + 2ax + 4 \leq 0$"为假

$\Leftrightarrow$ 其否定"$\forall x \in \mathbb{R},\ x^2 + 2ax + 4 > 0$"为真。

**第二步**：化为 $\Delta < 0$。

设 $g(x) = x^2 + 2ax + 4$，二次系数 $1 > 0$，开口向上。

$\forall x \in \mathbb{R},\ g(x) > 0$ $\Leftrightarrow$ 抛物线整体在 $x$ 轴上方 $\Leftrightarrow \Delta < 0$。

$\Delta = (2a)^2 - 4 \cdot 1 \cdot 4 = 4a^2 - 16 < 0$

$\Rightarrow a^2 < 4 \Rightarrow -2 < a < 2$。

**答案**：$\boxed{-2 < a < 2}$。

**总结**　"$\exists$ 假" $\Leftrightarrow$ "$\forall$ 反命题真"——量词命题真假转化的核心技巧。二次恒正 / 恒负看开口 + 判别式两件事，缺一不可。

---

## Part 2：不等式与基本不等式（D.09–D.15）

> 本组分两类：
> 1. **含参二次不等式**（D.09–D.12）：核心三看：①二次项系数 $a$ 是否为零（退化为一次）；②$a$ 的正负（开口方向）；③两根大小关系。恒成立 / 解集空 → 判别式 $\Delta$ 条件。
> 2. **基本不等式应用**（D.13–D.15）：核心三招："1 的代换"（约束是 $\dfrac{m}{x} + \dfrac{n}{y} = 1$ 型）、换元让分母可加（$\dfrac{1}{a+1} + \dfrac{4}{b+1}$ 型）、"和 + 积"模型（$\alpha x + \beta y + xy = c$ 化为关于和的二次不等式）。**取等条件务必验证可达**。

---

## D.09 [中档] Part 2/02

**题目回顾**：解关于 $x$ 的不等式：$ax^2 - (a + 1)x + 1 < 0$（$a \in \mathbb{R}$）。

**思路**　含参二次不等式三看：①二次系数 $a$ 是否为零（决定是否退化为一次）；②$a$ 的正负（决定开口方向）；③两根大小关系。先因式分解再分类。→ toolkit/11 分类讨论。

**解答**

**第一步**：因式分解。$ax^2 - (a+1)x + 1 = (ax - 1)(x - 1)$ ← 验证：展开 $(ax - 1)(x - 1) = ax^2 - ax - x + 1 = ax^2 - (a+1)x + 1$ ✓。

不等式化为 $(ax - 1)(x - 1) < 0$。

**第二步**：按 $a$ 分类。

**情况 1**（$a = 0$）：原不等式退化为 $-x + 1 < 0 \Rightarrow x > 1$，解集 $(1, +\infty)$。

**情况 2**（$a > 0$）：开口向上，方程两根为 $x_1 = \dfrac{1}{a}$ 与 $x_2 = 1$。

不等式 $< 0$ 的解集 = 两根之间。但需比较 $\dfrac{1}{a}$ 与 $1$ 的大小：

- 若 $\dfrac{1}{a} < 1$（即 $a > 1$）：解集 $\left(\dfrac{1}{a}, 1\right)$；
- 若 $\dfrac{1}{a} = 1$（即 $a = 1$）：$(x - 1)^2 < 0$ 无解；
- 若 $\dfrac{1}{a} > 1$（即 $0 < a < 1$）：解集 $\left(1, \dfrac{1}{a}\right)$。

**情况 3**（$a < 0$）：开口向下，$\dfrac{1}{a} < 0 < 1$（两根 $\dfrac{1}{a}$ 必为负），

不等式 $< 0$ 的解集 = 两根之外（开口向下 + 小于零 = 抛物线在 $x$ 轴下方两段）：

$\left(-\infty, \dfrac{1}{a}\right) \cup (1, +\infty)$。

**答案**：
$$\boxed{\begin{cases}
a < 0: & \left(-\infty, \dfrac{1}{a}\right) \cup (1, +\infty) \\
a = 0: & (1, +\infty) \\
0 < a < 1: & \left(1, \dfrac{1}{a}\right) \\
a = 1: & \varnothing \\
a > 1: & \left(\dfrac{1}{a}, 1\right)
\end{cases}}$$

**总结**　"含参二次不等式 $ax^2 + \ldots$"三看：①$a$ 是否为零（退化一次）；②$a$ 的正负（开口方向）；③两根大小关系（$1/a$ 与 $1$ 比较）。漏一就丢分。

---

## D.10 [中档] Part 2/02

**题目回顾**：解关于 $x$ 的不等式：$x^2 - (a + a^2)x + a^3 > 0$（$a \in \mathbb{R}$）。

**思路**　二次系数 $1 > 0$，开口固定向上；先因式分解，按两根 $a$ 与 $a^2$ 的大小关系分类。$> 0$ 的解集是两根之外。→ toolkit/11 分类讨论。

**解答**

**第一步**：因式分解。

$x^2 - (a + a^2)x + a^3 = (x - a)(x - a^2)$ ← 验证：和 $a + a^2$ ✓，积 $a \cdot a^2 = a^3$ ✓。

**第二步**：比较两根 $a$ 与 $a^2$ 的大小。

$a^2 - a = a(a - 1)$，符号：

- $a(a - 1) > 0 \Leftrightarrow a < 0 \text{ 或 } a > 1 \Rightarrow a^2 > a$；
- $a(a - 1) = 0 \Leftrightarrow a = 0 \text{ 或 } a = 1 \Rightarrow a^2 = a$（双根重合）；
- $a(a - 1) < 0 \Leftrightarrow 0 < a < 1 \Rightarrow a^2 < a$。

**第三步**：分类写解集。$(x - a)(x - a^2) > 0$，开口向上 → 两根之外。

**情况 1**（$a < 0$ 或 $a > 1$）：小根 $a$，大根 $a^2$。解集 $(-\infty, a) \cup (a^2, +\infty)$。

**情况 2.1**（$a = 0$）：$(x - 0)(x - 0) = x^2 > 0 \Rightarrow x \neq 0$。

**情况 2.2**（$a = 1$）：$(x - 1)^2 > 0 \Rightarrow x \neq 1$。

**情况 3**（$0 < a < 1$）：小根 $a^2$，大根 $a$。解集 $(-\infty, a^2) \cup (a, +\infty)$。

**答案**：
$$\boxed{\begin{cases}
a < 0 \text{ 或 } a > 1: & (-\infty, a) \cup (a^2, +\infty) \\
a = 0: & \{x \mid x \neq 0\} \\
0 < a < 1: & (-\infty, a^2) \cup (a, +\infty) \\
a = 1: & \{x \mid x \neq 1\}
\end{cases}}$$

**总结**　"两根 $a$、$a^2$ 比大小"是核心难点；$a^2 - a = a(a-1)$ 因式定符号最快，分四种 $a$ 取值情形：负 / 零 / $(0, 1)$ / $1$ / $> 1$。

---

## D.11 [中档] Part 2/02

**题目回顾**：若不等式 $x^2 - 2ax + a + 2 \leq 0$ 的解集是 $\varnothing$，求 $a$ 的取值范围。

**思路**　二次系数 $1 > 0$，开口固定向上。"$\leq 0$ 的解集空" 意味着抛物线恒在 $x$ 轴**严格上方**（不能与 $x$ 轴相切——切点处 $= 0$ 进入解集），即 $\Delta < 0$（严格小于零）。→ toolkit/04 数形结合。

**解答**

设 $g(x) = x^2 - 2ax + a + 2$（开口向上）。

解集 $\{x \mid g(x) \leq 0\} = \varnothing$ $\Leftrightarrow$ 抛物线整体严格在 $x$ 轴上方 $\Leftrightarrow \Delta < 0$。

$\Delta = (-2a)^2 - 4 \cdot 1 \cdot (a + 2) = 4a^2 - 4a - 8$。

$\Delta < 0 \Rightarrow 4a^2 - 4a - 8 < 0 \Rightarrow a^2 - a - 2 < 0$

$\Rightarrow (a - 2)(a + 1) < 0 \Rightarrow -1 < a < 2$。

**答案**：$\boxed{-1 < a < 2}$。

**总结**　二次不等式"$\leq 0$ 解集空" $\Leftrightarrow \Delta < 0$（开口向上）；注意是**严格**小于零——$\Delta = 0$ 时抛物线与 $x$ 轴相切，切点 $x_0$ 满足 $g(x_0) = 0$ 仍属解集。开口向上 + 解集空 + 不等号取 $\leq$ 三个条件锁住 $\Delta < 0$。

---

## D.12 [中档] Part 2/02

**题目回顾**：若关于 $x$ 的不等式 $x^2 - 2x + a > 0$ 在 $[1, 2]$ 上恒成立，求 $a$ 的取值范围。

**思路**　"恒成立" 类不等式标准做法：分离参数（把 $a$ 单独移到一边），再求另一边在给定区间上的极值。本题分离得 $a > -x^2 + 2x = g(x)$，要求 $a > \max_{x \in [1,2]} g(x)$。→ toolkit/05 参数策略。

**解答**

**第一步**：分离参数。

$x^2 - 2x + a > 0 \Leftrightarrow a > -x^2 + 2x = -(x - 1)^2 + 1$。

设 $g(x) = -(x - 1)^2 + 1$。

**第二步**：求 $g(x)$ 在 $[1, 2]$ 上的最大值。

$g$ 的图像是开口向下的抛物线，顶点 $(1, 1)$。

- 顶点 $x = 1$ 正好是区间左端点：$g(1) = 1$（最大）；
- 区间右端点 $x = 2$：$g(2) = -(2 - 1)^2 + 1 = 0$。

由于 $g$ 在 $[1, 2]$ 上单调递减（顶点在左端），最大值 $g_{\max} = 1$ 在 $x = 1$ 处取到。

**第三步**：恒成立条件。

$a > g(x)$ 对一切 $x \in [1, 2]$ 成立 $\Leftrightarrow a > g_{\max} = 1$。

**答案**：$\boxed{a > 1}$。

**总结**　"$a > f(x)$ 恒成立" $\Leftrightarrow a > f_{\max}$；"$a < f(x)$ 恒成立" $\Leftrightarrow a < f_{\min}$。先分离参数、再求极值是恒成立题的金科玉律。

---

## D.13 [中档] Part 2/04

**题目回顾**：已知 $x > 0$，$y > 0$，$\dfrac{2}{x} + \dfrac{1}{y} = 1$，求 $x + 2y$ 的最小值。

**思路**　"1 的代换"经典模型：把待求量乘以 $\dfrac{2}{x} + \dfrac{1}{y} = 1$（乘 $1$ 不变值），展开后凑出形如 $\dfrac{x}{y} + \dfrac{4y}{x}$ 的对偶项，再用基本不等式。这是高考"基本不等式约束最值"的最高频套路。→ Part 2/04 基本不等式。

**解答**

**第一步**：乘"$1$"展开。

$x + 2y = (x + 2y) \cdot 1 = (x + 2y)\left(\dfrac{2}{x} + \dfrac{1}{y}\right)$ ← 乘"1"，等价于乘 $\dfrac{2}{x} + \dfrac{1}{y}$

$= x \cdot \dfrac{2}{x} + x \cdot \dfrac{1}{y} + 2y \cdot \dfrac{2}{x} + 2y \cdot \dfrac{1}{y}$
$= 2 + \dfrac{x}{y} + \dfrac{4y}{x} + 2 = 4 + \dfrac{x}{y} + \dfrac{4y}{x}$。

**第二步**：基本不等式取等。

$\dfrac{x}{y} + \dfrac{4y}{x} \geq 2\sqrt{\dfrac{x}{y} \cdot \dfrac{4y}{x}} = 2\sqrt{4} = 4$，

等号当 $\dfrac{x}{y} = \dfrac{4y}{x}$，即 $x^2 = 4y^2$，由 $x, y > 0 \Rightarrow x = 2y$。

**第三步**：验证取等可达。代回约束：$\dfrac{2}{2y} + \dfrac{1}{y} = \dfrac{1}{y} + \dfrac{1}{y} = \dfrac{2}{y} = 1 \Rightarrow y = 2, x = 4$ ✓。

故 $x + 2y \geq 4 + 4 = 8$，最小值在 $(x, y) = (4, 2)$ 处取到。

**答案**：$\boxed{8}$。

**总结**　"1 的代换"识别：约束是 $\dfrac{m}{x} + \dfrac{n}{y} = 1$ 形式，求 $\alpha x + \beta y$ 最小值 → 乘上约束等于 $1$，展开后用基本不等式。**取等必须可达**：代回约束验证取等参数是否满足条件。

---

## D.14 [中档] Part 2/04

**题目回顾**：已知 $a > 0$，$b > 0$，$a + b = 2$，求 $\dfrac{1}{a + 1} + \dfrac{4}{b + 1}$ 的最小值。

**思路**　换元 $u = a + 1$，$v = b + 1$，则 $u + v = (a + b) + 2 = 4$；目标转为 $\dfrac{1}{u} + \dfrac{4}{v}$，仍是"1 的代换"模板——把约束 $u + v = 4$ 写成 $\dfrac{u + v}{4} = 1$，乘上目标。→ toolkit/02 换元 + Part 2/04 基本不等式。

**解答**

**第一步**：换元。

令 $u = a + 1, v = b + 1$。由 $a > 0, b > 0$ 知 $u > 1, v > 1$；由 $a + b = 2$ 知 $u + v = 4$。

目标：$\dfrac{1}{u} + \dfrac{4}{v}$ 最小。

**第二步**：乘"$1$"展开。

$\dfrac{1}{u} + \dfrac{4}{v} = \left(\dfrac{1}{u} + \dfrac{4}{v}\right) \cdot \dfrac{u + v}{4}$ ← 乘 $1 = \dfrac{u + v}{4}$

$= \dfrac{1}{4}\left(\dfrac{u}{u} + \dfrac{v}{u} + \dfrac{4u}{v} + \dfrac{4v}{v}\right) = \dfrac{1}{4}\left(1 + \dfrac{v}{u} + \dfrac{4u}{v} + 4\right)$

$= \dfrac{1}{4}\left(5 + \dfrac{v}{u} + \dfrac{4u}{v}\right)$。

**第三步**：基本不等式。

$\dfrac{v}{u} + \dfrac{4u}{v} \geq 2\sqrt{\dfrac{v}{u} \cdot \dfrac{4u}{v}} = 2\sqrt{4} = 4$，

等号当 $\dfrac{v}{u} = \dfrac{4u}{v} \Rightarrow v^2 = 4u^2 \Rightarrow v = 2u$（取正）。

**第四步**：验证可达。$v = 2u$ 代入 $u + v = 4 \Rightarrow 3u = 4 \Rightarrow u = \dfrac{4}{3}, v = \dfrac{8}{3}$ → $a = \dfrac{1}{3}, b = \dfrac{5}{3}$，均 $> 0$ ✓。

故原式 $\geq \dfrac{1}{4}(5 + 4) = \dfrac{9}{4}$。

**答案**：$\boxed{\dfrac{9}{4}}$。

**总结**　遇见"$\dfrac{m}{a + p} + \dfrac{n}{b + q}$ 与 $a + b = $ 常数"，先换元让分母 $u, v$ 直接相加成定数，再"1 的代换"是高考热门套路。

---

## D.15 [中档] Part 2/04

**题目回顾**：已知正数 $x, y$ 满足 $x + 4y + xy = 12$，求 $x + 4y$ 的最小值。

**思路**　约束含"和 + 积"两类项，直接代换困难。模板：令 $s = x + 4y$，用基本不等式 $x \cdot 4y \leq \left(\dfrac{x + 4y}{2}\right)^2$ 把 $xy$ 用 $s$ 上界化，从约束反解出关于 $s$ 的二次不等式。→ toolkit/03 构造法。

**解答**

**第一步**：由约束分离 $xy$。

$x + 4y + xy = 12 \Rightarrow xy = 12 - (x + 4y)$。

**第二步**：用基本不等式给 $xy$ 上界。

由 $x > 0, 4y > 0$，基本不等式：

$x \cdot 4y \leq \left(\dfrac{x + 4y}{2}\right)^2 \Rightarrow 4xy \leq \dfrac{(x + 4y)^2}{4} \Rightarrow xy \leq \dfrac{(x + 4y)^2}{16}$，

等号当 $x = 4y$。

**第三步**：列关于 $s = x + 4y$ 的不等式。

设 $s = x + 4y > 0$。$xy = 12 - s$，又 $xy \leq \dfrac{s^2}{16}$：

$12 - s \leq \dfrac{s^2}{16}$

$\Rightarrow 16(12 - s) \leq s^2$

$\Rightarrow 192 - 16s \leq s^2$

$\Rightarrow s^2 + 16s - 192 \geq 0$。

判别式：$\Delta = 256 + 768 = 1024 = 32^2$，根 $s = \dfrac{-16 \pm 32}{2} = 8, -24$。

$\Rightarrow (s + 24)(s - 8) \geq 0$。由 $s > 0 \Rightarrow s + 24 > 0$，故 $s - 8 \geq 0 \Rightarrow s \geq 8$。

**第四步**：验证取等可达。

$s = 8$ 时：$xy = 12 - 8 = 4$，$x = 4y \Rightarrow 4y \cdot y = 4 \Rightarrow y^2 = 1 \Rightarrow y = 1, x = 4$（取正）。

验证原约束 $4 + 4 + 4 \cdot 1 = 12$ ✓。

**答案**：$\boxed{x + 4y \text{ 的最小值为 } 8}$。

**总结**　"和 + 积型约束 $\alpha x + \beta y + xy = c$"模板：用 $xy \leq \left(\dfrac{\alpha x + \beta y}{2\sqrt{\alpha\beta}}\right)^2 \cdot \dfrac{1}{\alpha\beta} \cdot \alpha\beta$ 化为关于和的二次不等式；取等条件 $\alpha x = \beta y$ 与约束联立解出具体值。

---

## Part 4：指数对数运算（D.22–D.24）

> 本组三个核心：
> 1. **指数化简**：负指数 → 取倒；分数指数 $p/q$ → 先 $p$ 次方再 $q$ 次根（或反顺序）；$\sqrt{a^2} = |a|$ 必须带绝对值。
> 2. **对数换底**：$\log_a b = \dfrac{\log_c b}{\log_c a}$（统一底）；积变和（$\log MN = \log M + \log N$）；幂变倍（$\log M^n = n\log M$）。
> 3. **对数方程**：标准三步——列定义域 → 合并 → 解后必验根（定义域是最后筛子）。

---

## D.22 [中档] Part 4/01

**题目回顾**：计算：$\left(\dfrac{1}{27}\right)^{-\frac{2}{3}} - \left(\dfrac{16}{81}\right)^{0.75} + \sqrt{(2 - \pi)^2}$。

**思路**　逐项化简：①负指数取倒数；②分数指数 $p/q$ 化为 $q$ 次根的 $p$ 次方；③$\sqrt{a^2} = |a|$ 必须带绝对值，结合 $\pi > 2$ 决定符号。→ Part 4/01 指数运算。

**解答**

**第一项**：$\left(\dfrac{1}{27}\right)^{-\frac{2}{3}}$。

负指数取倒数：$\left(\dfrac{1}{27}\right)^{-\frac{2}{3}} = 27^{\frac{2}{3}} = (3^3)^{\frac{2}{3}} = 3^{3 \cdot \frac{2}{3}} = 3^2 = 9$。

**第二项**：$\left(\dfrac{16}{81}\right)^{0.75} = \left(\dfrac{16}{81}\right)^{\frac{3}{4}}$。

分子分母独立处理：
$$\left(\dfrac{16}{81}\right)^{\frac{3}{4}} = \dfrac{16^{3/4}}{81^{3/4}} = \dfrac{(2^4)^{3/4}}{(3^4)^{3/4}} = \dfrac{2^3}{3^3} = \dfrac{8}{27}.$$

**第三项**：$\sqrt{(2 - \pi)^2} = |2 - \pi|$。

由 $\pi \approx 3.14 > 2$ → $2 - \pi < 0 \Rightarrow |2 - \pi| = -(2 - \pi) = \pi - 2$。

**合并三项**：

$9 - \dfrac{8}{27} + (\pi - 2) = (9 - 2) + \pi - \dfrac{8}{27} = 7 + \pi - \dfrac{8}{27} = \dfrac{189 - 8}{27} + \pi = \dfrac{181}{27} + \pi$。

**答案**：$\boxed{\dfrac{181}{27} + \pi}$。

**总结**　指数运算三件事：①负号 → 取倒数；②$p/q$ 次方 → 底先乘方再开方（或反顺序）；③$\sqrt{a^2} = |a|$ 而非 $a$！与 $\pi$ 等无理数比大小时务必核对符号。

---

## D.23 [中档] Part 4/02

**题目回顾**：若 $\log_2 3 = a$，$\log_3 7 = b$，用 $a, b$ 表示 $\log_{42} 56$。

**思路**　"已知两个对数表 → 求第三个对数"套路：①把已知统一换底到同一底（这里选 $\log_2$ 最方便）；②整数因数分解 $42, 56$；③对求量换底分子分母同到 $\log_2$ 即可。→ Part 4/02 换底公式。

**解答**

**第一步**：统一换底到 $\log_2$。

由 $\log_3 7 = b$ 和 $\log_2 3 = a$，用换底公式：
$$\log_3 7 = \dfrac{\log_2 7}{\log_2 3} = \dfrac{\log_2 7}{a} = b \Rightarrow \log_2 7 = ab.$$

至此用 $a, b$ 表出了 $\log_2 3 = a$ 与 $\log_2 7 = ab$。

**第二步**：整数因数分解被求量。

$42 = 2 \cdot 3 \cdot 7$，$56 = 8 \cdot 7 = 2^3 \cdot 7$。

$$\log_2 42 = \log_2 2 + \log_2 3 + \log_2 7 = 1 + a + ab,$$
$$\log_2 56 = \log_2(2^3) + \log_2 7 = 3 + ab.$$

**第三步**：对求量换底。
$$\log_{42} 56 = \dfrac{\log_2 56}{\log_2 42} = \dfrac{3 + ab}{1 + a + ab}.$$

**答案**：$\boxed{\log_{42} 56 = \dfrac{3 + ab}{1 + a + ab}}$。

**总结**　"已知两对数求第三个"标准三步：①统一换底（通常选已知中出现的较小底）；②对要求底和真数都做整数分解；③换底直接落地。$a = \log_2 3, b = \log_3 7$ 组合可推 $\log_2 7 = ab$ 是关键技巧。

---

## D.24 [中档] Part 4/02

**题目回顾**：解方程：$\log_2(x - 1) + \log_2(x + 1) = 3$。

**思路**　对数方程标准三步：①列定义域（每个真数 $> 0$）；②对数和合并为真数积（$\log M + \log N = \log MN$）；③解出后必须验根（看是否落在定义域内）。→ Part 4/02 对数方程。

**解答**

**第一步**：列定义域。

每个真数都要 $> 0$：$x - 1 > 0$ 且 $x + 1 > 0$，即 $x > 1$。

**第二步**：对数和合并。

$\log_2(x - 1) + \log_2(x + 1) = \log_2[(x - 1)(x + 1)] = 3$

$\Rightarrow (x - 1)(x + 1) = 2^3 = 8$。

**第三步**：解代数方程。

$x^2 - 1 = 8 \Rightarrow x^2 = 9 \Rightarrow x = \pm 3$。

**第四步**：验根。

$x = 3$：满足 $x > 1$ ✓；$x = -3$：不满足 $x > 1$ ✗（且 $x - 1 = -4 < 0$ 真数为负，对数无定义）。

**答案**：$\boxed{x = 3}$。

**总结**　对数方程"先列定义域、合并取对数、解完验根" 三步缺一不可。$x = -3$ 在合并步骤"挤"出来，但定义域早就把它排除——必须始终用定义域作最后筛子。

---

## Part 5：三角化简、最值、解三角形（D.25–D.30）

> 本组四个核心：
> 1. **平方互推**：$(\sin \pm \cos)^2 = 1 \pm \sin 2\alpha$，三件套互推；最后用象限定符号。
> 2. **辅助角公式**：$a\sin x + b\cos x = R\sin(x + \varphi)$，$R = \sqrt{a^2 + b^2}$；$\varphi$ 由 $\cos\varphi, \sin\varphi$ 同时定（仅靠 $\tan$ 易丢象限）。
> 3. **三角恒等变换**：$\cos\theta = \sin(\theta + \pi/2)$ 是 $\sin/\cos$ 互化的桥；合并 $\to$ 周期 $\to$ 单调区间 $\to$ 最值是模板。
> 4. **解三角形**：正弦定理（边角桥）、余弦定理（两边夹角 / 三边求角）；"边-边-角"必查两解；"已知一边求面积最大"用基本不等式 $b^2 + c^2 \geq 2bc$ 给上界。

---

## D.25 [中档] Part 5/05

**题目回顾**：已知 $\sin\alpha - \cos\alpha = \dfrac{\sqrt{2}}{2}$，$\alpha \in \left(0, \dfrac{\pi}{2}\right)$，求 $\sin 2\alpha$ 与 $\sin\alpha + \cos\alpha$。

**思路**　"$\sin + \cos$、$\sin - \cos$、$\sin 2\alpha = 2\sin\alpha\cos\alpha$"三件套靠平方互推：

$$(\sin\alpha \pm \cos\alpha)^2 = 1 \pm 2\sin\alpha\cos\alpha = 1 \pm \sin 2\alpha.$$

最后用象限定符号。→ toolkit/01 结构识别。

**解答**

**第一步**：从已知平方求 $\sin 2\alpha$。

$(\sin\alpha - \cos\alpha)^2 = \sin^2\alpha - 2\sin\alpha\cos\alpha + \cos^2\alpha = 1 - \sin 2\alpha$。

由已知 $(\sin\alpha - \cos\alpha)^2 = \left(\dfrac{\sqrt{2}}{2}\right)^2 = \dfrac{1}{2}$：

$1 - \sin 2\alpha = \dfrac{1}{2} \Rightarrow \sin 2\alpha = \dfrac{1}{2}$。

**第二步**：算 $(\sin\alpha + \cos\alpha)^2$。

$(\sin\alpha + \cos\alpha)^2 = 1 + \sin 2\alpha = 1 + \dfrac{1}{2} = \dfrac{3}{2}$

$\Rightarrow \sin\alpha + \cos\alpha = \pm\sqrt{\dfrac{3}{2}} = \pm\dfrac{\sqrt{6}}{2}$。

**第三步**：定符号。

$\alpha \in \left(0, \dfrac{\pi}{2}\right) \Rightarrow \sin\alpha > 0, \cos\alpha > 0 \Rightarrow \sin\alpha + \cos\alpha > 0$，取正。

**答案**：$\boxed{\sin 2\alpha = \dfrac{1}{2},\ \sin\alpha + \cos\alpha = \dfrac{\sqrt{6}}{2}}$。

**总结**　"$\sin + \cos$、$\sin - \cos$、$\sin 2\alpha$" 三件套平方互推：$(\sin\alpha \pm \cos\alpha)^2 = 1 \pm \sin 2\alpha$。最后开方时务必用象限定符号——别丢负号也别保留多余正负号。

---

## D.26 [中档] Part 5/05

**题目回顾**：将 $f(x) = \sin x - \sqrt{3}\cos x$ 化为 $R\sin(x + \varphi)$ 形式，并求 $f(x)$ 在 $\left[0, \dfrac{\pi}{2}\right]$ 上的最值。

**思路**　辅助角公式：$a\sin x + b\cos x = R\sin(x + \varphi)$，$R = \sqrt{a^2 + b^2}$；$\varphi$ 由 $\cos\varphi = a/R, \sin\varphi = b/R$ 同时定（仅用 $\tan$ 易丢象限）。→ Part 5/05 辅助角。

**解答**

**第一步**：求 $R$ 与 $\varphi$。

$R = \sqrt{1^2 + (-\sqrt{3})^2} = \sqrt{4} = 2$。

设 $f(x) = 2\sin(x + \varphi)$，展开比较：
$$2\sin(x + \varphi) = 2\cos\varphi \sin x + 2\sin\varphi \cos x.$$

对比系数：
- $\sin x$ 项：$2\cos\varphi = 1 \Rightarrow \cos\varphi = \dfrac{1}{2}$；
- $\cos x$ 项：$2\sin\varphi = -\sqrt{3} \Rightarrow \sin\varphi = -\dfrac{\sqrt{3}}{2}$。

$\Rightarrow \varphi$ 在第四象限，取 $\varphi = -\dfrac{\pi}{3}$。

故 $f(x) = 2\sin\left(x - \dfrac{\pi}{3}\right)$。

**第二步**：求 $\left[0, \dfrac{\pi}{2}\right]$ 上的最值。

令 $t = x - \dfrac{\pi}{3}$。$x \in \left[0, \dfrac{\pi}{2}\right] \Rightarrow t \in \left[-\dfrac{\pi}{3}, \dfrac{\pi}{6}\right]$。

该区间含于 $\left[-\dfrac{\pi}{2}, \dfrac{\pi}{2}\right]$ ← $\sin t$ 在此区间上单调递增，故：
- 最小：$t = -\dfrac{\pi}{3}$（即 $x = 0$）：$\sin\left(-\dfrac{\pi}{3}\right) = -\dfrac{\sqrt{3}}{2} \Rightarrow f_{\min} = -\sqrt{3}$；
- 最大：$t = \dfrac{\pi}{6}$（即 $x = \dfrac{\pi}{2}$）：$\sin\left(\dfrac{\pi}{6}\right) = \dfrac{1}{2} \Rightarrow f_{\max} = 1$。

**答案**：$\boxed{f(x) = 2\sin\left(x - \dfrac{\pi}{3}\right),\ f_{\max} = 1,\ f_{\min} = -\sqrt{3}}$。

**总结**　$a\sin x + b\cos x$ 化辅助角公式 $R\sin(x + \varphi)$：$R = \sqrt{a^2 + b^2}$；$\varphi$ 由 $\cos\varphi = a/R$ 与 $\sin\varphi = b/R$ **同时定**——仅靠 $\tan\varphi$ 易丢象限！

---

## D.27 [中档] Part 5/03+05

**题目回顾**：已知 $f(x) = \sin\left(2x + \dfrac{\pi}{6}\right) + \cos\left(2x - \dfrac{\pi}{3}\right)$。求 $f(x)$ 的最小正周期与单调递增区间。

**思路**　多项三角和先化归"同名同角"才能合并。本题观察到两项角度差 $\dfrac{\pi}{6} - \left(-\dfrac{\pi}{3}\right) = \dfrac{\pi}{2}$，正好可借 $\cos\theta = \sin(\theta + \pi/2)$ 把 $\cos$ 转 $\sin$ 与第一项合并。→ Part 5/05 三角化简。

**解答**

**第一步**：化第二项为 $\sin$。

$\cos\left(2x - \dfrac{\pi}{3}\right) = \sin\left(\dfrac{\pi}{2} - \left(2x - \dfrac{\pi}{3}\right)\right)$ 不利于合并，

改用 $\cos\theta = \sin\left(\theta + \dfrac{\pi}{2}\right)$：

$\cos\left(2x - \dfrac{\pi}{3}\right) = \sin\left(2x - \dfrac{\pi}{3} + \dfrac{\pi}{2}\right) = \sin\left(2x + \dfrac{\pi}{6}\right)$ ✓ 两项角度对齐。

故 $f(x) = \sin\left(2x + \dfrac{\pi}{6}\right) + \sin\left(2x + \dfrac{\pi}{6}\right) = 2\sin\left(2x + \dfrac{\pi}{6}\right)$。

**第二步**：周期。$\omega = 2$，$T = \dfrac{2\pi}{|\omega|} = \dfrac{2\pi}{2} = \pi$。

**第三步**：单调递增区间。

$\sin u$ 在 $u \in \left[-\dfrac{\pi}{2} + 2k\pi, \dfrac{\pi}{2} + 2k\pi\right]$（$k \in \mathbb{Z}$）上递增。

令 $u = 2x + \dfrac{\pi}{6}$：
$$-\dfrac{\pi}{2} + 2k\pi \leq 2x + \dfrac{\pi}{6} \leq \dfrac{\pi}{2} + 2k\pi$$

各项减 $\dfrac{\pi}{6}$：
$$-\dfrac{\pi}{2} - \dfrac{\pi}{6} + 2k\pi \leq 2x \leq \dfrac{\pi}{2} - \dfrac{\pi}{6} + 2k\pi$$
$$-\dfrac{2\pi}{3} + 2k\pi \leq 2x \leq \dfrac{\pi}{3} + 2k\pi$$

各项除 $2$：
$$-\dfrac{\pi}{3} + k\pi \leq x \leq \dfrac{\pi}{6} + k\pi.$$

**答案**：$\boxed{T = \pi;\ \text{递增区间} \left[-\dfrac{\pi}{3} + k\pi, \dfrac{\pi}{6} + k\pi\right]\ (k \in \mathbb{Z})}$。

**总结**　多项三角和先化归同名同角后必能合并；周期由 $\omega$ 决定，单调区间套 $\sin u$ 的标准递增区间反解。"角度差 $\dfrac{\pi}{2}$"是 $\sin / \cos$ 互化的标志。

---

## D.28 [中档] Part 5/07

**题目回顾**：在 $\triangle ABC$ 中，$a, b, c$ 分别为内角 $A, B, C$ 的对边，已知 $b = 2$，$c = 2\sqrt{3}$，$B = 30°$，求 $a$。

**思路**　已知"边-边-角"型——先用正弦定理求另一角 $C$，**两解陷阱**：$\sin C$ 已知时 $C$ 在锐角与钝角中各有一个候选，需结合三角形内角条件筛除。再用内角和定 $A$，正弦定理求 $a$。→ Part 5/07 解三角形。

**解答**

**第一步**：正弦定理求 $\sin C$。

$\dfrac{b}{\sin B} = \dfrac{c}{\sin C}$
$\Rightarrow \sin C = \dfrac{c \sin B}{b} = \dfrac{2\sqrt{3} \cdot \sin 30°}{2} = \dfrac{2\sqrt{3} \cdot \frac{1}{2}}{2} = \dfrac{\sqrt{3}}{2}$。

**第二步**：定 $C$（两解筛查）。

$\sin C = \dfrac{\sqrt{3}}{2}$ 的可能取值：$C = 60°$ 或 $C = 120°$。

由 $c = 2\sqrt{3} > 2 = b$，"大边对大角" $\Rightarrow C > B = 30°$；两候选 $60°, 120°$ 均满足 $> 30°$。

又需 $A + B + C < 180°$（实际是 $= 180°$）：
- $C = 60° \Rightarrow A = 90°$（合法）；
- $C = 120° \Rightarrow A = 30°$（合法）。

→ 两解都有效。

**第三步**：分别求 $a$。

由正弦定理 $\dfrac{a}{\sin A} = \dfrac{b}{\sin B} = \dfrac{2}{\sin 30°} = 4$，故 $a = 4\sin A$。

- $A = 90°$：$a = 4 \cdot 1 = 4$；
- $A = 30°$：$a = 4 \cdot \dfrac{1}{2} = 2$。

**答案**：$\boxed{a = 4 \text{ 或 } a = 2}$。

**总结**　"边-边-角"必查两解。$\sin$ 已知 → 锐 / 钝两候选；用"大边对大角"+ "内角和 $< 180°$" 双筛。两解都通过时全部保留。

---

## D.29 [中档] Part 5/07

**题目回顾**：在 $\triangle ABC$ 中，已知 $2\cos^2 \dfrac{A}{2} = \dfrac{\sqrt{3}}{3}\sin A$，$b + c = 6$，$a = 2\sqrt{3}$，求 $\triangle ABC$ 的面积。

**思路**　先用倍角公式 $2\cos^2(A/2) = 1 + \cos A$ 化简条件求 $A$；再用余弦定理结合 $b + c = 6$ 求 $bc$；最后面积 $S = \dfrac{1}{2} bc \sin A$。三件套：$A$、$bc$、$\sin A$ 凑齐即可。→ Part 5/05 倍角 + Part 5/07 解三角形。

**解答**

**第一步**：化简条件求 $A$。

由倍角公式 $\cos A = 2\cos^2\dfrac{A}{2} - 1 \Rightarrow 2\cos^2\dfrac{A}{2} = 1 + \cos A$（倍角逆用）。

代入原条件：$1 + \cos A = \dfrac{\sqrt{3}}{3} \sin A$，即 $\sqrt{3}(1 + \cos A) = \sin A$ ← 两边乘 $\sqrt{3}$。

再用 $\sin A = 2\sin\dfrac{A}{2}\cos\dfrac{A}{2}$ 与 $1 + \cos A = 2\cos^2\dfrac{A}{2}$：

$\sqrt{3} \cdot 2\cos^2\dfrac{A}{2} = 2\sin\dfrac{A}{2}\cos\dfrac{A}{2}$。

由 $A \in (0, \pi) \Rightarrow \dfrac{A}{2} \in (0, \pi/2) \Rightarrow \cos\dfrac{A}{2} > 0$，两边除 $2\cos\dfrac{A}{2}$：

$\sqrt{3} \cos\dfrac{A}{2} = \sin\dfrac{A}{2} \Rightarrow \tan\dfrac{A}{2} = \sqrt{3} \Rightarrow \dfrac{A}{2} = \dfrac{\pi}{3} \Rightarrow A = \dfrac{2\pi}{3}$。

故 $\cos A = -\dfrac{1}{2}$，$\sin A = \dfrac{\sqrt{3}}{2}$。

**第二步**：求 $bc$。

余弦定理 $a^2 = b^2 + c^2 - 2bc\cos A$：

$12 = b^2 + c^2 - 2bc \cdot \left(-\dfrac{1}{2}\right) = b^2 + c^2 + bc$。

由 $(b + c)^2 = b^2 + c^2 + 2bc = 36 \Rightarrow b^2 + c^2 = 36 - 2bc$，代入：

$12 = (36 - 2bc) + bc = 36 - bc \Rightarrow bc = 24$。

**第三步**：求面积。

$S = \dfrac{1}{2} bc \sin A = \dfrac{1}{2} \cdot 24 \cdot \dfrac{\sqrt{3}}{2} = 6\sqrt{3}$。

**答案**：$\boxed{S = 6\sqrt{3}}$。

**总结**　半角条件 → 倍角逆用 → 化为 $\tan(A/2)$（避免平方引入增根）；面积 $\dfrac{1}{2}bc\sin A$ 三件套：$A$（定 $\sin A$）、$bc$（余弦定理 + $b + c$ 联立）。

---

## D.30 [中档] Part 5/07

**题目回顾**：在 $\triangle ABC$ 中，$a, b, c$ 分别为 $A, B, C$ 的对边，已知 $a^2 - b^2 - c^2 + bc = 0$，$a = 2$，求 $\triangle ABC$ 面积的最大值。

**思路**　两步：①条件 $a^2 = b^2 + c^2 - bc$ 与余弦定理 $a^2 = b^2 + c^2 - 2bc\cos A$ 对照得 $\cos A = \dfrac{1}{2}$（定 $A$）；②面积 $S = \dfrac{1}{2}bc\sin A$ 要 $bc$ 最大，由 $a^2 = b^2 + c^2 - bc \geq 2bc - bc = bc$ 反解出 $bc$ 上界。→ Part 5/07 解三角形 + Part 2/04 基本不等式。

**解答**

**第一步**：定 $A$。

由条件 $a^2 = b^2 + c^2 - bc$，与余弦定理 $a^2 = b^2 + c^2 - 2bc\cos A$ 对比：

$-bc = -2bc\cos A \Rightarrow 2bc\cos A = bc \Rightarrow \cos A = \dfrac{1}{2}$ ← $bc \neq 0$ 可约去。

故 $A = \dfrac{\pi}{3}$，$\sin A = \dfrac{\sqrt{3}}{2}$。

**第二步**：求 $bc$ 上界。

由 $a = 2$，条件 $4 = b^2 + c^2 - bc$。

由基本不等式 $b^2 + c^2 \geq 2bc$（等号当 $b = c$）：

$4 = b^2 + c^2 - bc \geq 2bc - bc = bc$，

即 $bc \leq 4$。

取等条件：$b = c$，代入 $4 = b^2 + b^2 - b^2 = b^2 \Rightarrow b = c = 2$（等边三角形 ← 与 $A = 60°$ 一致）。

**第三步**：求面积最大值。

$S = \dfrac{1}{2} bc \sin A = \dfrac{1}{2} bc \cdot \dfrac{\sqrt{3}}{2} = \dfrac{\sqrt{3}}{4} bc \leq \dfrac{\sqrt{3}}{4} \cdot 4 = \sqrt{3}$。

取等当 $b = c = 2$（正三角形）✓。

**答案**：$\boxed{S_{\max} = \sqrt{3}}$。

**总结**　"给定一边 $a$ + 一条 $b, c$ 关系，求面积最大值"模板：①余弦定理找 $A$；②基本不等式 $b^2 + c^2 \geq 2bc$ 反解 $bc$ 上界；③面积公式 $\dfrac{1}{2}bc\sin A$。等号常在正三角形取到。

---

## Part 6：复数四则与共轭（D.31–D.33）

> 本组三个核心：
> 1. **复数除法**：分子分母同乘分母的共轭使分母实化（$(a + bi)(a - bi) = a^2 + b^2$）。
> 2. **共轭与模**：$\bar{z}$ 是虚部取负；$|z| = \sqrt{a^2 + b^2}$；恒等式 $z\bar{z} = |z|^2$ 必用。
> 3. **几何意义**：$|z - z_0|$ 是"$z$ 到 $z_0$ 的距离"；$|z - a| = |z - b|$ → 中垂线；$|z - z_0| = r$ → 圆；$|z - a| + |z - b| =$ 常 → 椭圆。

---

## D.31 [中档] Part 6/02

**题目回顾**：已知 $z = \dfrac{1 + 2i}{1 - i}$。求 $z$、$|z|$ 与 $z \cdot \bar{z}$。

**思路**　复数除法标配：分子分母同乘分母的共轭使分母实化。模 $|z|$ 用平方和开方；$z\bar z = |z|^2$ 是核心恒等式。→ Part 6/02 复数四则。

**解答**

**第一步**：化简 $z$（分母实化）。

$z = \dfrac{1 + 2i}{1 - i} = \dfrac{(1 + 2i)(1 + i)}{(1 - i)(1 + i)}$ ← 分子分母同乘 $\overline{1 - i} = 1 + i$。

分子：$(1 + 2i)(1 + i) = 1 + i + 2i + 2i^2 = 1 + 3i - 2 = -1 + 3i$。

分母：$(1 - i)(1 + i) = 1 - i^2 = 1 + 1 = 2$（实数化成功）。

$z = \dfrac{-1 + 3i}{2} = -\dfrac{1}{2} + \dfrac{3}{2}i$。

**第二步**：求 $|z|$。

$|z| = \sqrt{\left(-\dfrac{1}{2}\right)^2 + \left(\dfrac{3}{2}\right)^2} = \sqrt{\dfrac{1}{4} + \dfrac{9}{4}} = \sqrt{\dfrac{10}{4}} = \dfrac{\sqrt{10}}{2}$。

**第三步**：求 $z \cdot \bar{z}$。

由公式 $z\bar{z} = |z|^2$：

$z \cdot \bar{z} = \left(\dfrac{\sqrt{10}}{2}\right)^2 = \dfrac{10}{4} = \dfrac{5}{2}$。

**答案**：$\boxed{z = -\dfrac{1}{2} + \dfrac{3}{2}i,\ |z| = \dfrac{\sqrt{10}}{2},\ z\bar{z} = \dfrac{5}{2}}$。

**总结**　复数除法 → 分母乘共轭实化是唯一套路；$z\bar{z} = |z|^2$ 反推 $|z|^2$ 比逐个相乘快。

---

## D.32 [中档] Part 6/02

**题目回顾**：复数 $z$ 满足 $(1 + i) z = 3 - i$，求 $z$ 的共轭复数 $\bar{z}$，并指出 $\bar{z}$ 在复平面上对应的象限。

**思路**　先求 $z$（分母实化），再共轭（虚部取负），最后看 $\bar{z}$ 实部虚部符号定象限。→ Part 6/02 复数除法 + 共轭。

**解答**

**第一步**：求 $z$。

$z = \dfrac{3 - i}{1 + i} = \dfrac{(3 - i)(1 - i)}{(1 + i)(1 - i)}$ ← 分子分母同乘 $\overline{1 + i} = 1 - i$。

分子：$(3 - i)(1 - i) = 3 - 3i - i + i^2 = 3 - 4i - 1 = 2 - 4i$。

分母：$(1 + i)(1 - i) = 1 - i^2 = 2$。

$z = \dfrac{2 - 4i}{2} = 1 - 2i$。

**第二步**：求共轭。

$\bar{z} = \overline{1 - 2i} = 1 + 2i$（虚部取负）。

**第三步**：定象限。

$\bar{z} = 1 + 2i$ 对应点 $(1, 2)$：实部 $1 > 0$，虚部 $2 > 0$ → 第一象限。

**答案**：$\boxed{\bar{z} = 1 + 2i,\ \text{对应点在第一象限}}$。

**总结**　共轭复数 = 虚部取负；复平面象限按 $(\text{Re}, \text{Im})$ 的正负符号读：$(+, +)$ 第一象限、$(-, +)$ 第二象限、$(-, -)$ 第三象限、$(+, -)$ 第四象限。

---

## D.33 [中档] Part 6/02

**题目回顾**：设复数 $z$ 满足 $|z - 1| = |z - i|$，求 $z$ 对应点在复平面上的轨迹方程。

**思路**　$|z - z_0|$ 表示"$z$ 到点 $z_0$ 的距离"；本题"$z$ 到 $(1, 0)$ 与到 $(0, 1)$ 的距离相等" → 几何上是两定点连线的中垂线。代数处理：设 $z = x + yi$ 平方展开。→ toolkit/04 数形结合。

**解答**

**几何视角**：$z$ 对应点 $(x, y)$，$1$ 对应 $(1, 0)$，$i$ 对应 $(0, 1)$。条件 $|z - 1| = |z - i|$ 即"$P$ 到 $A(1, 0)$ 距离 = $P$ 到 $B(0, 1)$ 距离" → $P$ 在 $AB$ 中垂线上。

**代数推导**：设 $z = x + yi$（$x, y \in \mathbb{R}$）。

$|z - 1| = |(x - 1) + yi| = \sqrt{(x - 1)^2 + y^2}$；
$|z - i| = |x + (y - 1)i| = \sqrt{x^2 + (y - 1)^2}$。

两边平方相等：$(x - 1)^2 + y^2 = x^2 + (y - 1)^2$

$\Rightarrow x^2 - 2x + 1 + y^2 = x^2 + y^2 - 2y + 1$

$\Rightarrow -2x = -2y \Rightarrow y = x$。

验证：$A(1, 0), B(0, 1)$ 中点 $(\frac{1}{2}, \frac{1}{2})$ 显然在 $y = x$ 上 ✓；$AB$ 斜率 $\dfrac{1 - 0}{0 - 1} = -1$，中垂线斜率 $1$，方程 $y - \frac{1}{2} = 1 \cdot (x - \frac{1}{2}) \Rightarrow y = x$ ✓。

**答案**：$\boxed{y = x}$（过原点的 $45°$ 直线，即连接 $(1, 0)$ 与 $(0, 1)$ 的中垂线）。

**总结**　复数模等式 $|z - a| = |z - b|$ → 几何中垂线；代数代入 $z = x + yi$，平方差消去 $x^2, y^2$ 同类项即得直线方程。$|z - a| < r$ 是圆盘，$|z - a| = |z - b|$ 是中垂线，$|z - a| + |z - b| = $ 常数是椭圆——三大模型要分清。

---

## Part 7：统计——方差、相关系数、线性回归（D.34–D.37）

> 本组三个核心：
> 1. **基本统计量**：平均数 $\bar{x} = \dfrac{1}{n}\sum x_i$；方差 $s^2 = \dfrac{1}{n}\sum(x_i - \bar{x})^2 = \overline{x^2} - \bar{x}^2$；中位数偶数项取中间两数平均。**稳定性**比较看方差。
> 2. **相关系数 $r$**：$r \in [-1, 1]$；符号定方向（正 / 负相关），绝对值定强弱（$|r| \geq 0.75$ 高度、$0.3 \sim 0.75$ 中等、$< 0.3$ 弱）。
> 3. **线性回归**：$\hat{b} = \dfrac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$，$\hat{a} = \bar{y} - \hat{b}\bar{x}$；回归直线必过样本中心点 $(\bar{x}, \bar{y})$。

---

## D.34 [中档] Part 7/02

**题目回顾**：某次数学测验班级 6 名学生成绩为：$72, 78, 80, 85, 90, 95$。求该组数据的平均数、方差与中位数。

**思路**　三个基本统计量分别计算：①平均数 $\bar{x} = \dfrac{1}{n}\sum x_i$；②中位数：数据排序后取中间值（偶数项取中间两数均值）；③方差 $s^2 = \dfrac{1}{n}\sum(x_i - \bar{x})^2$，等价地 $\overline{x^2} - \bar{x}^2$。本题平均数非整数，建议用 $\overline{x^2} - \bar{x}^2$ 避免逐项算分数偏差。→ Part 7/02 方差。

**解答**

**第一步**：平均数。

$\bar{x} = \dfrac{72 + 78 + 80 + 85 + 90 + 95}{6} = \dfrac{500}{6} = \dfrac{250}{3} \approx 83.33$。

**第二步**：中位数。

数据 $72, 78, 80, 85, 90, 95$ 已升序排列，$n = 6$ 偶数，中位数 = 第 $3$ 与第 $4$ 个数据的均值 = $\dfrac{80 + 85}{2} = 82.5$。

**第三步**：方差（用 $\overline{x^2} - \bar{x}^2$ 公式）。

先算 $\sum x_i^2$：

$72^2 = 5184$；$78^2 = 6084$；$80^2 = 6400$；$85^2 = 7225$；$90^2 = 8100$；$95^2 = 9025$。

$\sum x_i^2 = 5184 + 6084 + 6400 + 7225 + 8100 + 9025 = 42018$。

$\overline{x^2} = \dfrac{42018}{6} = 7003$。

$\bar{x}^2 = \left(\dfrac{250}{3}\right)^2 = \dfrac{62500}{9}$。

$s^2 = 7003 - \dfrac{62500}{9} = \dfrac{63027 - 62500}{9} = \dfrac{527}{9} \approx 58.56$。

**答案**：$\boxed{\bar{x} = \dfrac{250}{3} \approx 83.33;\ \text{中位数} = 82.5;\ s^2 = \dfrac{527}{9} \approx 58.56}$。

**总结**　$s^2 = \overline{x^2} - \bar{x}^2$ 是计算方差的快捷公式（避免逐项算分数偏差）。中位数对偶数项必取中间两数平均，对奇数项直接取中间数。

---

## D.35 [中档] Part 7/02

**题目回顾**：甲、乙两人各打靶 5 次，成绩（环）分别为：甲：$8, 9, 10, 9, 9$；乙：$10, 7, 9, 10, 9$。分别计算两人的平均数与方差，并据此判断谁的成绩更稳定。

**思路**　成绩"稳定性"由方差衡量：方差小 → 数据更集中 → 更稳定。两步：①算平均数；②算偏差平方和除以 $n$。→ Part 7/02 方差。

**解答**

**甲**

平均数：$\bar{x}_甲 = \dfrac{8 + 9 + 10 + 9 + 9}{5} = \dfrac{45}{5} = 9$。

偏差 $x_i - \bar{x}_甲$：$-1, 0, 1, 0, 0$；平方：$1, 0, 1, 0, 0$；和 $= 2$。

方差：$s_甲^2 = \dfrac{2}{5} = 0.4$。

**乙**

平均数：$\bar{x}_乙 = \dfrac{10 + 7 + 9 + 10 + 9}{5} = \dfrac{45}{5} = 9$。

偏差：$1, -2, 0, 1, 0$；平方：$1, 4, 0, 1, 0$；和 $= 6$。

方差：$s_乙^2 = \dfrac{6}{5} = 1.2$。

**比较与结论**

$\bar{x}_甲 = \bar{x}_乙 = 9$（平均水平相同）；

$s_甲^2 = 0.4 < 1.2 = s_乙^2$（甲的方差更小 → 甲的成绩更集中、波动更小）。

**结论**：甲的成绩更稳定。

**答案**：$\boxed{\bar{x}_甲 = \bar{x}_乙 = 9,\ s_甲^2 = 0.4,\ s_乙^2 = 1.2;\ \text{甲更稳定}}$。

**总结**　稳定性比较 → 方差比较，方差小者更稳。当两组数据平均数相同时，方差小者优；若平均数不同时，常用变异系数 $\dfrac{s}{\bar{x}}$ 比较相对稳定性。

---

## D.36 [中档] Part 7/03

**题目回顾**：某商店调查广告投入 $x$（万元）与销售额 $y$（万元）有如下五组数据：

| $x$ | 1 | 2 | 3 | 4 | 5 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $y$ | 6 | 8 | 11 | 13 | 17 |

求 $y$ 关于 $x$ 的线性回归方程 $\hat{y} = \hat{b} x + \hat{a}$，并预测当 $x = 6$ 时的销售额。

**思路**　最小二乘公式：$\hat{b} = \dfrac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$，$\hat{a} = \bar{y} - \hat{b}\bar{x}$。先算两个均值，再列偏差表，最后代入公式。回归直线必过样本中心点 $(\bar{x}, \bar{y})$，这是验证公式的常用检查点。→ Part 7/03 线性回归。

**解答**

**第一步**：计算均值。

$\bar{x} = \dfrac{1 + 2 + 3 + 4 + 5}{5} = \dfrac{15}{5} = 3$；
$\bar{y} = \dfrac{6 + 8 + 11 + 13 + 17}{5} = \dfrac{55}{5} = 11$。

**第二步**：列偏差表。

| $x_i$ | $y_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | $(x_i - \bar{x})(y_i - \bar{y})$ | $(x_i - \bar{x})^2$ |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 6 | $-2$ | $-5$ | $10$ | $4$ |
| 2 | 8 | $-1$ | $-3$ | $3$ | $1$ |
| 3 | 11 | $0$ | $0$ | $0$ | $0$ |
| 4 | 13 | $1$ | $2$ | $2$ | $1$ |
| 5 | 17 | $2$ | $6$ | $12$ | $4$ |
| **和** |  |  |  | $27$ | $10$ |

**第三步**：计算回归系数。

$\hat{b} = \dfrac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} = \dfrac{27}{10} = 2.7$；

$\hat{a} = \bar{y} - \hat{b}\bar{x} = 11 - 2.7 \times 3 = 11 - 8.1 = 2.9$。

回归方程：$\hat{y} = 2.7 x + 2.9$。

**验证**：回归直线过 $(\bar{x}, \bar{y}) = (3, 11)$？代入：$2.7 \times 3 + 2.9 = 8.1 + 2.9 = 11$ ✓。

**第四步**：预测 $x = 6$。

$\hat{y} = 2.7 \times 6 + 2.9 = 16.2 + 2.9 = 19.1$（万元）。

**答案**：$\boxed{\hat{y} = 2.7 x + 2.9;\ x = 6 \text{ 时预测销售额 } \hat{y} = 19.1 \text{ 万元}}$。

**总结**　最小二乘三步：① 算 $\bar{x}, \bar{y}$；② 列偏差表算分子分母（$\hat{b}$）；③ $\hat{a} = \bar{y} - \hat{b}\bar{x}$。回归直线必过 $(\bar{x}, \bar{y})$ 是必查项。

---

## D.37 [中档] Part 7/03

**题目回顾**：已知两个变量 $x, y$ 的样本相关系数 $r = 0.92$，请说明 $x, y$ 之间的相关性强弱与方向。又：若进一步给出 $\bar{x} = 5$，$\bar{y} = 12$，$\hat{b} = 2$，写出线性回归方程。

**思路**　相关系数 $r$ 解读两步：①符号定方向（正 / 负相关）；②绝对值定强弱。回归方程 $\hat{a} = \bar{y} - \hat{b}\bar{x}$ 是必背公式（回归直线必过样本中心点）。→ Part 7/03 相关性 & 回归。

**解答**

**第一步**：相关性判定。

- **方向**：$r = 0.92 > 0$ → 正相关（$x$ 增大时 $y$ 也呈增大趋势）；
- **强弱**：$|r| = 0.92$ 非常接近 $1$，按经验法则（$|r| \geq 0.75$ 高度相关）：高度（强）线性相关。

**第二步**：写线性回归方程。

由 $\hat{a} = \bar{y} - \hat{b}\bar{x}$：

$\hat{a} = 12 - 2 \times 5 = 12 - 10 = 2$。

故 $\hat{y} = 2x + 2$。

**验证**：代入 $x = \bar{x} = 5 \Rightarrow \hat{y} = 10 + 2 = 12 = \bar{y}$ ✓（回归直线过中心点 $(5, 12)$）。

**答案**：$\boxed{x, y \text{ 之间为高度正相关};\ \hat{y} = 2x + 2}$。

**总结**　$r$ 解读："正/负"看符号；"强/弱"看绝对值。经验区间：$|r| \geq 0.75$ 高度；$0.3 \leq |r| < 0.75$ 中等；$|r| < 0.3$ 弱。回归直线必过 $(\bar{x}, \bar{y})$ —— 公式 $\hat{a} = \bar{y} - \hat{b}\bar{x}$ 本质上就是要求这一点。

---

> **题号索引（共 31 题，全部中档难度）**
>
> | 主题 | 题号范围 | 题数 |
> |------|---------|------|
> | 集合与含参 | D.01–D.05 | 5 |
> | 逻辑量词与充要条件 | D.06–D.08 | 3 |
> | 不等式与基本不等式 | D.09–D.15 | 7 |
> | 指数对数运算 | D.22–D.24 | 3 |
> | 三角化简、最值、解三角形 | D.25–D.30 | 6 |
> | 复数四则与共轭 | D.31–D.33 | 3 |
> | 统计：方差 / 相关系数 / 线性回归 | D.34–D.37 | 4 |
> | **合计** | | **31** |
