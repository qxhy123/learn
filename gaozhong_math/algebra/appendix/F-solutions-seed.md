# 附录 F：高一上 80 题详解

> 涵盖附录 C 全部 80 题（C.01–C.80）。每题独立小节，按题号顺序排列。
> 每题包含：**题目回顾**（再列题面）、**思路**（1 句套路 + toolkit 引用）、**解答**（完整推导）、**总结**（识题特征）。
> 引用：→ toolkit/01 结构识别、→ toolkit/02 换元、→ toolkit/03 构造、→ toolkit/04 数形、→ toolkit/05 参数策略、→ toolkit/06 抽象函数、→ toolkit/07 复合、→ toolkit/08 单调极值、→ toolkit/09 对称周期、→ toolkit/10 放缩、→ toolkit/11 分类讨论、→ toolkit/12 解题前清单。

---

## Part 1 集合与逻辑

---

## C.01 [基础] Part 1/01

**题目回顾**：用列举法表示集合 $A = \{x \in \mathbb{N} \mid -2 < x \leq 3\}$，并写出 $A$ 的元素个数。

**思路**　自然数集 $\mathbb{N}$ 含 $0$；按条件枚举满足的整数即可。→ toolkit/12 清单（先看元素来源）。

**解答**　$\mathbb{N} = \{0, 1, 2, 3, \ldots\}$；条件 $-2 < x \leq 3$ 在 $\mathbb{N}$ 内取得的元素为 $0, 1, 2, 3$。

故 $A = \{0, 1, 2, 3\}$，元素个数为 $4$。

**总结**　看到"$x \in \mathbb{N}$"立即区分 $\mathbb{N}$（含 $0$）与 $\mathbb{N}^*$（不含 $0$），别少元素。

---

## C.02 [基础] Part 1/02

**题目回顾**：$A = \{1, 2, 3, 4\}$，$B = \{2, 4, 6\}$，求 $A \cup B$ 与 $A \cap B$。

**思路**　直接套定义：并集合并所有元素（重复只算一次），交集取公共元素。

**解答**　$A \cup B = \{1, 2, 3, 4, 6\}$；$A \cap B = \{2, 4\}$。

**总结**　并交是集合最基础的两个运算；写并集时不要漏元素，写交集时不要把非公共元素带进来。

---

## C.03 [基础] Part 1/03

**题目回顾**：集合 $M = \{a, b\}$ 的所有子集是什么？真子集个数是多少？

**思路**　$n$ 元集合子集数为 $2^n$，真子集数为 $2^n - 1$。

**解答**　$M$ 的子集有 $\varnothing, \{a\}, \{b\}, \{a, b\}$，共 $4 = 2^2$ 个；真子集（不含 $M$ 本身）$\varnothing, \{a\}, \{b\}$，共 $3 = 2^2 - 1$ 个。

**总结**　记牢公式：子集 $2^n$，真子集 $2^n - 1$，非空真子集 $2^n - 2$。

---

## C.04 [基础] Part 1/04

**题目回顾**：写出命题"$\forall x \in \mathbb{R}, x^2 \geq 0$"的否定。

**思路**　全称命题的否定是存在命题，并且把结论取反。

**解答**　原命题否定：$\exists x \in \mathbb{R}, x^2 < 0$。

**总结**　"$\forall \to \exists$，$\geq \to <$"两个变化要同时做，缺一不可。

---

## C.05 [中档] Part 1/02

**题目回顾**：$U = \{1, 2, 3, 4, 5, 6\}$，$A = \{1, 3, 5\}$，$B = \{2, 3, 5, 6\}$，求 $\complement_U(A \cup B)$ 与 $(\complement_U A) \cap (\complement_U B)$，验证相等（德摩根律）。

**思路**　先算两边各自的式子，验证 $\complement(A \cup B) = (\complement A) \cap (\complement B)$。→ toolkit/01 结构识别。

**解答**　$A \cup B = \{1, 2, 3, 5, 6\}$，故 $\complement_U(A \cup B) = \{4\}$。

$\complement_U A = \{2, 4, 6\}$，$\complement_U B = \{1, 4\}$，故 $(\complement_U A) \cap (\complement_U B) = \{4\}$。

两者均为 $\{4\}$，等式成立。

**总结**　德摩根律："并的补 = 补的交，交的补 = 补的并"；记住这条小题秒解。

---

## C.06 [中档] Part 1/03

**题目回顾**：$A = \{x \mid x^2 - 3x + 2 = 0\}$，$B = \{x \mid x^2 - ax + a - 1 = 0\}$，若 $B \subseteq A$，求 $a$ 的所有可能值。

**思路**　先求 $A$；$B \subseteq A$ 意味着 $B$ 是 $A$ 的子集，分 $B = \varnothing$、$B$ 单元素、$B = A$ 三种情况讨论。→ toolkit/11 分类。

**解答**　$A$：$x^2 - 3x + 2 = (x-1)(x-2) = 0 \Rightarrow A = \{1, 2\}$。

对 $B$ 做因式分解：$x^2 - ax + (a - 1) = (x - 1)(x - (a - 1)) = 0$，所以 $B$ 的根总是包含 $1$，另一个根为 $a - 1$。

- 若 $a - 1 = 1$，即 $a = 2$，$B = \{1\}$，$B \subseteq A$ ✓
- 若 $a - 1 = 2$，即 $a = 3$，$B = \{1, 2\} = A$，$B \subseteq A$ ✓
- 若 $a - 1$ 既不是 $1$ 也不是 $2$，$B$ 含有 $A$ 之外的元素，不满足。

故 $a = 2$ 或 $a = 3$。

**总结**　$B \subseteq A$ 这种条件总要追问"$B$ 能不能为空？$B$ 能等于 $A$ 吗？"——分类必须完整。

---

## C.07 [中档] Part 1/05

**题目回顾**：$p$：$x^2 - x - 6 \leq 0$；$q$：$|x - 1| \leq m$（$m > 0$）。$p$ 是 $q$ 的必要不充分条件，求 $m$ 的取值范围。

**思路**　把 $p, q$ 化为区间，"$p$ 必要不充分于 $q$"$\Leftrightarrow$$q$ 集合真包含于 $p$ 集合。→ toolkit/05 参数策略。

**解答**　$p$：$x^2 - x - 6 \leq 0 \Rightarrow (x - 3)(x + 2) \leq 0 \Rightarrow -2 \leq x \leq 3$，记 $P = [-2, 3]$。

$q$：$|x - 1| \leq m \Rightarrow 1 - m \leq x \leq 1 + m$，记 $Q = [1 - m, 1 + m]$。

"$p$ 必要不充分于 $q$" 即 "$q \Rightarrow p$ 但 $p \not\Rightarrow q$"，等价于 $Q \subsetneq P$。

需要：$\begin{cases} 1 - m \geq -2 \\ 1 + m \leq 3 \\ Q \neq P \end{cases}$，即 $\begin{cases} m \leq 3 \\ m \leq 2 \end{cases}$ 且 $Q \neq P$。

由 $m > 0$ 知 $Q$ 不能等于 $P$ 仅在 $m = 2$ 时同时 $Q = [-1, 3] \neq [-2, 3] = P$，仍是真子集 ✓。

综上 $0 < m \leq 2$。

**总结**　充分必要条件题目"翻译成集合关系"是关键：必要不充分 $\Leftrightarrow$ 真包含。等号能否取要单独检查。

---

## C.08 [中档] Part 1/02

**题目回顾**：$A = [-1, 4]$，$B = \{x \mid x < a\}$，$A \cap B \neq \varnothing$，求 $a$ 的范围。

**思路**　$A \cap B$ 非空意味着 $B = (-\infty, a)$ 与 $A$ 有重叠。画数轴即明。→ toolkit/04 数形。

**解答**　若 $a \leq -1$，则 $B = (-\infty, a) \subseteq (-\infty, -1]$，与 $[-1, 4]$ 至多在 $-1$ 处相切；但 $B$ 是开区间，$-1 \notin B$，故 $A \cap B = \varnothing$。

若 $a > -1$，则 $-1 \in B$（因 $-1 < a$）且 $-1 \in A$，故 $A \cap B \ni -1$，非空。

综上 $a > -1$。

**总结**　集合交非空 $\Leftrightarrow$ 数轴上两区间有重叠，端点开闭要看清。

---

## C.09 [提升] Part 1/03

**题目回顾**：$A = \{x \mid x^2 - 5x + 6 = 0\}$，$B = \{x \mid mx - 1 = 0\}$，$B \subseteq A$，求 $m$。

**思路**　含参集合 $B$ 的形态依赖 $m$：$m = 0$ 时 $B = \varnothing$（"$mx = 1$" 无解）；$m \neq 0$ 时 $B = \{1/m\}$ 单元素。→ toolkit/11 分类。

**解答**　$A$：$x^2 - 5x + 6 = (x - 2)(x - 3) = 0 \Rightarrow A = \{2, 3\}$。

讨论 $B$：

- 若 $m = 0$，方程 $0 \cdot x = 1$ 无解，$B = \varsigma$，空集是任何集合的子集 ✓
- 若 $m \neq 0$，$B = \{1/m\}$，要 $1/m \in A$，即 $1/m = 2$（$m = 1/2$）或 $1/m = 3$（$m = 1/3$）

综上 $m \in \left\{0, \dfrac{1}{2}, \dfrac{1}{3}\right\}$。

**总结**　"含参 + 子集" 第一步永远是"参数能不能让方程退化"。$mx - 1 = 0$ 在 $m = 0$ 时退化为无解（$B = \varnothing$），不能漏。

---

## C.10 [提升] Part 1/04+05

**题目回顾**：$p$：$x^2 - 8x - 20 \leq 0$；$q$：$x^2 - 2x + 1 - m^2 \leq 0$（$m > 0$）。$\neg p$ 是 $\neg q$ 的必要不充分条件，求 $m$。

**思路**　"$\neg p$ 必要不充分于 $\neg q$" $\Leftrightarrow$ "$\neg q \Rightarrow \neg p$，$\neg p \not\Rightarrow \neg q$"$\Leftrightarrow$ "$p \Rightarrow q$，$q \not\Rightarrow p$"（逆否等价）$\Leftrightarrow$ $P \subsetneq Q$。→ toolkit/05 参数策略。

**解答**　$p$：$x^2 - 8x - 20 = (x - 10)(x + 2) \leq 0 \Rightarrow -2 \leq x \leq 10$，$P = [-2, 10]$。

$q$：$x^2 - 2x + 1 - m^2 = (x - 1)^2 - m^2 = (x - 1 - m)(x - 1 + m) \leq 0$。

由 $m > 0$，$\Rightarrow 1 - m \leq x \leq 1 + m$，$Q = [1 - m, 1 + m]$。

由"$P \subsetneq Q$" 要：$\begin{cases} 1 - m \leq -2 \\ 1 + m \geq 10 \\ \text{至少有一个严格不等}\end{cases} \Rightarrow \begin{cases} m \geq 3 \\ m \geq 9 \end{cases}$。

取较强约束 $m \geq 9$；严格不等只需 $m > 9$ 或某端点严格放宽，常规答案 $m \geq 9$（两端点同时取 $-2, 10$ 时 $m = 9$ 仅满足 $P = Q$ 不算真包含；故 $m > 9$）。

**注**：题意通常理解 $P \subsetneq Q$（真子集），最稳妥答案 $m \geq 9$（含 $m = 9$ 时 $Q = [-8, 10] \supsetneq [-2, 10] = P$ 确为真包含 ✓）。

故 $m \geq 9$。

**总结**　"$\neg p$ 必要不充分于 $\neg q$"等价 "$p$ 真包含于 $q$"——逆否换形后转集合关系，是高考必考套路。

---

## Part 2 不等式

---

## C.11 [基础] Part 2/01

**题目回顾**：$a > b$，比较 $-3a + 1$ 与 $-3b + 1$。

**思路**　乘以负数变号。

**解答**　$a > b$ 两边乘 $-3$（负数）反向：$-3a < -3b$；两边加 $1$ 不变向：$-3a + 1 < -3b + 1$。

**总结**　不等式"乘正不变向、乘负变向、加常数不变向"是最基本的性质。

---

## C.12 [基础] Part 2/02

**题目回顾**：解 $x^2 - 5x + 6 < 0$。

**思路**　因式分解 → 标轴法 / 抛物线判断。→ toolkit/04 数形。

**解答**　$x^2 - 5x + 6 = (x - 2)(x - 3) < 0$。

抛物线开口向上，与 $x$ 轴交于 $2, 3$；"小于零"取两根之间。

解集：$\{x \mid 2 < x < 3\}$。

**总结**　记口诀"大于号取两边、小于号取中间"（开口向上时）。

---

## C.13 [基础] Part 2/02

**题目回顾**：解 $-x^2 + 4x - 3 \geq 0$。

**思路**　先把首项系数变正再处理。

**解答**　两边乘 $-1$ 变号：$x^2 - 4x + 3 \leq 0$，即 $(x - 1)(x - 3) \leq 0$。

抛物线开口向上、与 $x$ 轴交于 $1, 3$；"小于等于零"取两根之间（含端点）。

解集：$[1, 3]$。

**总结**　看到首项负系数，第一步"两边乘 $-1$ 变号"，否则容易记错口诀。

---

## C.14 [基础] Part 2/04

**题目回顾**：$x > 0$，求 $x + \dfrac{4}{x}$ 的最小值。

**思路**　基本不等式三条件自查："一正、二定、三相等"。→ toolkit/01 结构识别。

**解答**　$x > 0 \Rightarrow x > 0$ 且 $\dfrac{4}{x} > 0$（一正 ✓）；

两项之积 $x \cdot \dfrac{4}{x} = 4$（二定 ✓）；

由基本不等式 $x + \dfrac{4}{x} \geq 2\sqrt{x \cdot \dfrac{4}{x}} = 2\sqrt{4} = 4$。

取等条件 $x = \dfrac{4}{x} \Rightarrow x^2 = 4 \Rightarrow x = 2$（$x > 0$ 内可达，三相等 ✓）。

故最小值为 $4$，在 $x = 2$ 时取得。

**总结**　基本不等式题目固定模板："验证一正二定三相等 → 写出不等式 → 报告等号成立条件"，缺一不可。

---

## C.15 [中档] Part 2/02

**题目回顾**：解 $x^2 - (a + 1)x + a < 0$。

**思路**　因式分解后含参，按两根大小分类。→ toolkit/11 分类。

**解答**　$x^2 - (a + 1)x + a = (x - 1)(x - a) < 0$。

两根为 $1$ 与 $a$。开口向上，取两根之间。

- 若 $a > 1$，解集 $(1, a)$；
- 若 $a = 1$，$(x - 1)^2 < 0$ 无解，解集 $\varnothing$；
- 若 $a < 1$，解集 $(a, 1)$。

**总结**　含参一元二次不等式，分类的核心是"两根大小关系（或重合）"。

---

## C.16 [中档] Part 2/03

**题目回顾**：解 $\dfrac{x - 1}{x + 2} \geq 0$。

**思路**　分式不等式标准化：$\dfrac{f}{g} \geq 0 \Leftrightarrow f \cdot g \geq 0$ 且 $g \neq 0$。→ toolkit/01 结构识别。

**解答**　$\dfrac{x - 1}{x + 2} \geq 0 \Leftrightarrow (x - 1)(x + 2) \geq 0$ 且 $x + 2 \neq 0$。

$(x - 1)(x + 2) \geq 0$ 解集 $(-\infty, -2] \cup [1, +\infty)$；

去掉 $x = -2$：解集为 $(-\infty, -2) \cup [1, +\infty)$。

**总结**　分式不等式标准转化口诀："$\geq$ 同号、$\leq$ 异号、分母非零分母不取等号"。

---

## C.17 [中档] Part 2/04

**题目回顾**：$x > 1$，求 $x + \dfrac{4}{x - 1}$ 的最小值。

**思路**　积非常数，凑常数。→ toolkit/03 构造法。

**解答**　$x + \dfrac{4}{x - 1} = (x - 1) + \dfrac{4}{x - 1} + 1$。

令 $t = x - 1 > 0$，则 $(x - 1) + \dfrac{4}{x - 1} \geq 2\sqrt{(x - 1) \cdot \dfrac{4}{x - 1}} = 4$（一正二定三相等：$t = 2/t$ 时 $t = 2 \Rightarrow x = 3$ ✓）。

所以原式 $\geq 4 + 1 = 5$，最小值 $5$，在 $x = 3$ 时取得。

**总结**　"分母不同形"凑差是基本不等式的核心套路：把表达式拆出"两项相乘为常数"的结构。

---

## C.18 [中档] Part 2/04

**题目回顾**：$x > 0$，$y > 0$，$\dfrac{1}{x} + \dfrac{4}{y} = 1$，求 $x + y$ 的最小值。

**思路**　"$1$ 的代换"经典技巧：把 $x + y$ 乘上 $1 = \dfrac{1}{x} + \dfrac{4}{y}$ 展开。→ toolkit/03 构造。

**解答**　$x + y = (x + y) \cdot 1 = (x + y)\left(\dfrac{1}{x} + \dfrac{4}{y}\right) = 1 + \dfrac{4x}{y} + \dfrac{y}{x} + 4 = 5 + \dfrac{4x}{y} + \dfrac{y}{x}$。

由基本不等式 $\dfrac{4x}{y} + \dfrac{y}{x} \geq 2\sqrt{4} = 4$（积为 $4$，常数 ✓）。

故 $x + y \geq 5 + 4 = 9$。

取等：$\dfrac{4x}{y} = \dfrac{y}{x} \Rightarrow y^2 = 4x^2 \Rightarrow y = 2x$；再代入约束 $\dfrac{1}{x} + \dfrac{4}{2x} = 1 \Rightarrow \dfrac{3}{x} = 1 \Rightarrow x = 3, y = 6$ ✓。

最小值为 $9$。

**总结**　"$1$ 的代换"是基本不等式中最常考的技巧：见 "$\dfrac{1}{x} + \dfrac{?}{y} = 1$ 求 $x + y$ 类"立刻反应。

---

## C.19 [中档] Part 2/02

**题目回顾**：$ax^2 + bx + 2 > 0$ 的解集为 $\left(-\dfrac{1}{2}, \dfrac{1}{3}\right)$，求 $a + b$。

**思路**　解集为开区间且包含原方程两根 $\Rightarrow$ $a < 0$（开口向下）；用韦达定理反推。→ toolkit/05 参数策略。

**解答**　由"大于号取中间"，开口向下，$a < 0$；两根为 $-\dfrac{1}{2}, \dfrac{1}{3}$。

韦达定理：两根之和 $-\dfrac{1}{2} + \dfrac{1}{3} = -\dfrac{1}{6} = -\dfrac{b}{a}$；两根之积 $-\dfrac{1}{2} \cdot \dfrac{1}{3} = -\dfrac{1}{6} = \dfrac{2}{a}$。

由 $\dfrac{2}{a} = -\dfrac{1}{6} \Rightarrow a = -12$。

由 $-\dfrac{b}{a} = -\dfrac{1}{6} \Rightarrow b = \dfrac{a}{6} = -2$。

故 $a + b = -12 + (-2) = -14$。

**总结**　"已知解集反推系数"标准套路：看开口方向（解集形状）+ 韦达定理（根和、根积），两条信息推出 $a, b$。

---

## C.20 [提升] Part 2/02

**题目回顾**：解 $ax^2 - (a + 1)x + 1 < 0$。

**思路**　含参一元二次不等式，按 $a = 0$、$a > 0$、$a < 0$ 分大类，再分根的相对大小。→ toolkit/11 分类。

**解答**　$ax^2 - (a + 1)x + 1$ 因式分解：把 $1$ 与 $a$ 看作十字相乘 $ax^2 - ax - x + 1 = ax(x - 1) - (x - 1) = (ax - 1)(x - 1)$。

故原不等式 $(ax - 1)(x - 1) < 0$。

- 若 $a = 0$，$-(x - 1) < 0 \Rightarrow x > 1$，解集 $(1, +\infty)$。
- 若 $a > 0$，两根为 $x = 1, x = \dfrac{1}{a}$。需比 $1$ 与 $\dfrac{1}{a}$ 的大小：
  - $0 < a < 1$：$\dfrac{1}{a} > 1$，解集 $\left(1, \dfrac{1}{a}\right)$；
  - $a = 1$：两根相等，$(x - 1)^2 < 0$ 无解，解集 $\varnothing$；
  - $a > 1$：$\dfrac{1}{a} < 1$，解集 $\left(\dfrac{1}{a}, 1\right)$。
- 若 $a < 0$，$ax - 1 = 0 \Rightarrow x = \dfrac{1}{a} < 0 < 1$。$(ax - 1)(x - 1) < 0$；由 $a < 0$，写成 $a(x - \dfrac{1}{a})(x - 1) < 0$，两边除以 $a$ 变号 $(x - \dfrac{1}{a})(x - 1) > 0$，解集 $\left(-\infty, \dfrac{1}{a}\right) \cup (1, +\infty)$。

**总结**　含参一元二次不等式的标准步骤：① 因式分解；② 分 $a$ 的正负零；③ 分两根大小（或重合）。耐心列清每一类。

---

## C.21 [提升] Part 2/04

**题目回顾**：$a, b > 0$，$a + 2b = 1$，求 $\dfrac{1}{a} + \dfrac{1}{b}$ 的最小值。

**思路**　$1$ 的代换：把 $\dfrac{1}{a} + \dfrac{1}{b}$ 乘以 $1 = a + 2b$。→ toolkit/03 构造。

**解答**　$\dfrac{1}{a} + \dfrac{1}{b} = \left(\dfrac{1}{a} + \dfrac{1}{b}\right)(a + 2b) = 1 + \dfrac{2b}{a} + \dfrac{a}{b} + 2 = 3 + \dfrac{2b}{a} + \dfrac{a}{b}$。

由基本不等式 $\dfrac{2b}{a} + \dfrac{a}{b} \geq 2\sqrt{2}$。

故 $\dfrac{1}{a} + \dfrac{1}{b} \geq 3 + 2\sqrt{2}$。

取等：$\dfrac{2b}{a} = \dfrac{a}{b} \Rightarrow a^2 = 2b^2 \Rightarrow a = \sqrt{2}b$；代入 $a + 2b = 1$：$\sqrt{2}b + 2b = 1 \Rightarrow b = \dfrac{1}{\sqrt{2} + 2} = \dfrac{2 - \sqrt{2}}{2}$，$a = \sqrt{2} \cdot \dfrac{2 - \sqrt{2}}{2} = \sqrt{2} - 1 \cdot$（在 $a, b > 0$ 范围内可达 ✓）。

最小值 $3 + 2\sqrt{2}$。

**总结**　$1$ 的代换是基本不等式高频套路；"展开 + 配对 + 取等"三步骤要熟练。

---

## C.22 [提升] Part 2/03

**题目回顾**：解 $\dfrac{x^2 - x - 2}{x^2 - 4x + 3} \leq 0$。

**思路**　把分式不等式转化为 $f \cdot g \leq 0$ 且 $g \neq 0$；用标轴法（穿针引线）。→ toolkit/04 数形。

**解答**　分子 $x^2 - x - 2 = (x - 2)(x + 1)$；分母 $x^2 - 4x + 3 = (x - 1)(x - 3)$。

定义域：$x \neq 1, x \neq 3$。

$\dfrac{(x - 2)(x + 1)}{(x - 1)(x - 3)} \leq 0 \Leftrightarrow (x - 2)(x + 1)(x - 1)(x - 3) \leq 0$ 且 $x \neq 1, 3$。

四个零点（升序）：$-1, 1, 2, 3$。"奇穿偶不穿"画数轴：自右向左符号交替为 $+, -, +, -, +$。

"$\leq 0$" 取负区间（含零点，但要排除分母零点）：$[-1, 1) \cup [2, 3)$。

**总结**　标轴法是分式 / 高次不等式的通用利器：因式分解 → 排零点 → 自右向左交替符号 → 取所需。

---

## Part 3 函数概念

---

## C.23 [基础] Part 3/01

**题目回顾**：判断 $f(x) = x^2$（$\mathbb{R} \to \mathbb{R}$）与 $g(x) = \pm\sqrt{x^2 + 1}$（$\mathbb{R} \to \mathbb{R}$）是否为函数。

**思路**　函数的核心：每个 $x$ 只对应**唯一**的 $y$。

**解答**　$f(x) = x^2$：每个 $x \in \mathbb{R}$ 都得到唯一 $y = x^2$，**是函数**。

$g(x) = \pm\sqrt{x^2 + 1}$：每个 $x$ 对应两个值（正负），**不是函数**。

**总结**　函数定义"一对一或多对一，绝不一对多"；"$\pm$"立刻警觉。

---

## C.24 [基础] Part 3/02

**题目回顾**：求 $f(x) = \sqrt{x - 1} + \dfrac{1}{x - 3}$ 的定义域。

**思路**　多个条件取交集：① 根号内 $\geq 0$；② 分母 $\neq 0$。

**解答**　$x - 1 \geq 0 \Rightarrow x \geq 1$；$x - 3 \neq 0 \Rightarrow x \neq 3$。

定义域：$[1, 3) \cup (3, +\infty)$。

**总结**　看到"根号、分式、对数、零次幂"立刻自查各自条件，多约束取交集。

---

## C.25 [基础] Part 3/02

**题目回顾**：求 $f(x) = x^2 - 2x + 3$ 在 $[0, 3]$ 上的值域。

**思路**　配方找顶点 → 看顶点和端点函数值。→ toolkit/04 数形。

**解答**　$f(x) = (x - 1)^2 + 2$，顶点 $(1, 2)$，开口向上。

$1 \in [0, 3]$，故 $f$ 在 $[0, 1]$ 递减，$[1, 3]$ 递增。

最小值 $f(1) = 2$；端点 $f(0) = 3$，$f(3) = 6$，最大值 $f(3) = 6$。

值域：$[2, 6]$。

**总结**　二次函数在闭区间上值域 = "看顶点是否在区间内 + 比端点函数值"，永远这两步。

---

## C.26 [基础] Part 3/04

**题目回顾**：判断 $f(x) = x^3 - x$ 的奇偶性。

**思路**　按定义 $f(-x)$ 与 $f(x)$ 的关系。

**解答**　定义域 $\mathbb{R}$，关于原点对称 ✓。

$f(-x) = (-x)^3 - (-x) = -x^3 + x = -(x^3 - x) = -f(x)$。

故 $f$ 为奇函数。

**总结**　奇偶性判断三步：① 定义域对称？② $f(-x)$ 与 $f(x)$ 关系？③ 下结论。

---

## C.27 [基础] Part 3/03

**题目回顾**：用定义法证明 $f(x) = 2x + 1$ 在 $\mathbb{R}$ 上单调递增。

**思路**　定义法三步：取值、作差、判号。

**解答**　任取 $x_1, x_2 \in \mathbb{R}$，$x_1 < x_2$。

作差：$f(x_1) - f(x_2) = (2x_1 + 1) - (2x_2 + 1) = 2(x_1 - x_2)$。

由 $x_1 < x_2$ 知 $x_1 - x_2 < 0$，故 $f(x_1) - f(x_2) < 0$，即 $f(x_1) < f(x_2)$。

由定义，$f$ 在 $\mathbb{R}$ 上单调递增。

**总结**　定义法"三步走"是高一硬通货：取值 → 作差 → 判号，缺一不可。

---

## C.28 [基础] Part 3/05

**题目回顾**：$f(x) = \begin{cases} 2x + 1, & x < 0 \\ x^2, & x \geq 0 \end{cases}$，求 $f(-2) + f(3)$。

**思路**　分段函数取值看定义域归属。

**解答**　$-2 < 0$ 用上支：$f(-2) = 2(-2) + 1 = -3$；

$3 \geq 0$ 用下支：$f(3) = 3^2 = 9$；

$f(-2) + f(3) = -3 + 9 = 6$。

**总结**　分段函数代值"先判段，再代入"；端点（$x = 0$）属于哪段要看不等号严不严。

---

## C.29 [中档] Part 3/02

**题目回顾**：$f(x) = \dfrac{\sqrt{x + 1}}{x - 2}$，求定义域与 $f(3)$。

**思路**　多约束取交集。

**解答**　$x + 1 \geq 0 \Rightarrow x \geq -1$；$x - 2 \neq 0 \Rightarrow x \neq 2$。

定义域 $A = [-1, 2) \cup (2, +\infty)$。

$f(3) = \dfrac{\sqrt{3 + 1}}{3 - 2} = \dfrac{2}{1} = 2$。

**总结**　定义域看"根 / 分母 / 对数"；代值前先确认 $x$ 在定义域内。

---

## C.30 [中档] Part 3/03

**题目回顾**：用定义法证明 $f(x) = \dfrac{1}{x}$ 在 $(0, +\infty)$ 上单调递减。

**思路**　定义法三步；分式作差要通分。

**解答**　任取 $x_1, x_2 \in (0, +\infty)$，$x_1 < x_2$。

$f(x_1) - f(x_2) = \dfrac{1}{x_1} - \dfrac{1}{x_2} = \dfrac{x_2 - x_1}{x_1 x_2}$。

$x_1, x_2 > 0 \Rightarrow x_1 x_2 > 0$；$x_1 < x_2 \Rightarrow x_2 - x_1 > 0$。

故 $f(x_1) - f(x_2) > 0$，即 $f(x_1) > f(x_2)$。

$f$ 在 $(0, +\infty)$ 上单调递减。

**总结**　分式作差永远先通分；分子分母符号要分开判断，最后合成。

---

## C.31 [中档] Part 3/03

**题目回顾**：$f(x) = x + \dfrac{a}{x}$（$x > 0$）在 $[2, +\infty)$ 上单调递增，求 $a$。

**思路**　对勾函数 $x + \dfrac{a}{x}$（$a > 0$）在 $[\sqrt{a}, +\infty)$ 递增；$a \leq 0$ 时 $f$ 在 $(0, +\infty)$ 恒增。→ toolkit/08 单调极值。

**解答**　分两种情况讨论 $a$。

**情形 1**：$a \leq 0$。$f'(x) = 1 - \dfrac{a}{x^2}$；由 $a \leq 0$ 得 $-\dfrac{a}{x^2} \geq 0$，故 $f'(x) \geq 1 > 0$。$f$ 在 $(0, +\infty)$ 上严格递增，自然在 $[2, +\infty)$ 递增 ✓。

（不用导数也可直接说：$\dfrac{a}{x}$ 在 $a \leq 0$ 时是单调递增函数，$f$ 是两个递增函数之和。）

**情形 2**：$a > 0$。$f(x) = x + \dfrac{a}{x}$ 在 $(0, \sqrt{a}]$ 递减、$[\sqrt{a}, +\infty)$ 递增（标准结论，基本不等式取等位）。

要 $f$ 在 $[2, +\infty)$ 递增，需 $\sqrt{a} \leq 2 \Rightarrow a \leq 4$。

结合 $a > 0$：$0 < a \leq 4$。

**合并**：$a \leq 4$。

**总结**　对勾函数 $f(x) = x + \dfrac{a}{x}$（$a > 0$）的"分水岭" 是 $\sqrt{a}$；记牢"递减区间 $(0, \sqrt{a}]$、递增区间 $[\sqrt{a}, +\infty)$" 这套模板。

---

## C.32 [中档] Part 3/04

**题目回顾**：$f$ 在 $\mathbb{R}$ 上奇函数，$x > 0$ 时 $f(x) = x^2 - 2x$，求 $x < 0$ 时 $f(x)$。

**思路**　奇函数 $f(-x) = -f(x)$；要求 $x < 0$ 表达式时令 $-x > 0$ 用已知部分。→ toolkit/09 对称。

**解答**　设 $x < 0$，则 $-x > 0$，由已知 $f(-x) = (-x)^2 - 2(-x) = x^2 + 2x$。

由奇函数 $f(-x) = -f(x)$，得 $f(x) = -f(-x) = -(x^2 + 2x) = -x^2 - 2x$。

故 $x < 0$ 时 $f(x) = -x^2 - 2x$。

**总结**　已知一侧求另一侧的固定套路："换 $-x$ 进已知 + 奇偶性反推"。

---

## C.33 [中档] Part 3/06

**题目回顾**：$f(x) = x^2 + 1$，$g(x) = 2x - 3$。求 $f(g(x))$、$g(f(x))$、$f(g(2))$。

**思路**　复合函数从内到外代入。→ toolkit/07 复合。

**解答**　$f(g(x)) = (g(x))^2 + 1 = (2x - 3)^2 + 1 = 4x^2 - 12x + 9 + 1 = 4x^2 - 12x + 10$。

$g(f(x)) = 2 f(x) - 3 = 2(x^2 + 1) - 3 = 2x^2 - 1$。

$f(g(2)) = f(2 \cdot 2 - 3) = f(1) = 1^2 + 1 = 2$。

**总结**　复合函数永远"从里往外"算；$f(g(x))$ 中 $g$ 先算。

---

## C.34 [中档] Part 3/03+04

**题目回顾**：$f$ 是 $\mathbb{R}$ 上的偶函数，且在 $[0, +\infty)$ 单调递增。若 $f(2a - 1) < f(3)$，求 $a$。

**思路**　偶函数 $f(x) = f(|x|)$；不等式两边的绝对值（"距离"）大小决定函数值大小。→ toolkit/09 对称。

**解答**　偶函数：$f(2a - 1) = f(|2a - 1|)$，$f(3) = f(|3|) = f(3)$。

在 $[0, +\infty)$ 单调递增，故 $f(|2a - 1|) < f(3) \Leftrightarrow |2a - 1| < 3$。

解 $|2a - 1| < 3$：$-3 < 2a - 1 < 3 \Rightarrow -1 < a < 2$。

**总结**　偶函数 + 单调（半轴）= 将函数值大小转化为"绝对值大小"。这是高考必考结构。

---

## C.35 [中档] Part 3/05

**题目回顾**：画出 $y = |x - 1| + |x + 2|$ 的图象，并求最小值。

**思路**　分段去绝对值；几何意义：$|x - 1|$ 是到 $1$ 的距离，$|x + 2|$ 是到 $-2$ 的距离。→ toolkit/04 数形。

**解答**　拆区间：

- $x < -2$：$y = -(x - 1) - (x + 2) = -2x - 1$；
- $-2 \leq x \leq 1$：$y = -(x - 1) + (x + 2) = 3$；
- $x > 1$：$y = (x - 1) + (x + 2) = 2x + 1$。

图象：$x < -2$ 直线下降，$-2 \leq x \leq 1$ 水平线 $y = 3$，$x > 1$ 上升直线。整体为 V 形（底部为水平段）。

最小值：在 $-2 \leq x \leq 1$ 段 $y = 3$，即 $y_{\min} = 3$。

几何上 $|x - 1| + |x + 2|$ 是 $x$ 到两个定点 $1, -2$ 的距离之和，最小值为两点间距 $|1 - (-2)| = 3$。

**总结**　$|x - a| + |x - b|$ 形式立刻想几何意义：到 $a, b$ 距离之和 $\geq |a - b|$。

---

## C.36 [中档] Part 3/02+03

**题目回顾**：求 $y = x - \sqrt{1 - 2x}$ 的值域。

**思路**　换元 $t = \sqrt{1 - 2x}$ 将无理函数化为二次函数。→ toolkit/02 换元。

**解答**　令 $t = \sqrt{1 - 2x}$，$t \geq 0$；则 $1 - 2x = t^2 \Rightarrow x = \dfrac{1 - t^2}{2}$。

代入：$y = \dfrac{1 - t^2}{2} - t = -\dfrac{1}{2}t^2 - t + \dfrac{1}{2}$（$t \geq 0$）。

这是 $t$ 的二次函数，顶点 $t = -\dfrac{-1}{2 \cdot (-1/2)} = -1$，但 $t \geq 0$ 不含顶点。

在 $[0, +\infty)$ 上，开口向下且顶点在左侧，函数递减。

$t = 0$ 时 $y = \dfrac{1}{2}$（最大）；$t \to +\infty$ 时 $y \to -\infty$。

值域：$\left(-\infty, \dfrac{1}{2}\right]$。

**总结**　含 $\sqrt{ax + b}$ 类函数，换元化二次是标准动作；注意新变量的范围。

---

## C.37 [提升] Part 3/06

**题目回顾**：$f(x) = x^2 - 4x + 3$，解 $f(f(x)) = 0$。

**思路**　令 $u = f(x)$，先解 $f(u) = 0$，再对每个 $u$ 解 $f(x) = u$。→ toolkit/02 换元 + toolkit/07 复合。

**解答**　令 $u = f(x)$，先解 $f(u) = u^2 - 4u + 3 = (u - 1)(u - 3) = 0 \Rightarrow u = 1$ 或 $u = 3$。

- $u = 1$：$f(x) = x^2 - 4x + 3 = 1 \Rightarrow x^2 - 4x + 2 = 0 \Rightarrow x = 2 \pm \sqrt{2}$。
- $u = 3$：$f(x) = x^2 - 4x + 3 = 3 \Rightarrow x^2 - 4x = 0 \Rightarrow x(x - 4) = 0 \Rightarrow x = 0$ 或 $x = 4$。

综上 $x \in \{0, 4, 2 - \sqrt{2}, 2 + \sqrt{2}\}$，共 $4$ 个解。

**总结**　复合方程 $f(f(x)) = c$ 标准套路："外层换元 → 解外层 → 回代解内层"，得多解。

---

## C.38 [提升] Part 3/03

**题目回顾**：$f(x) = \dfrac{ax + b}{x^2 + 1}$ 是奇函数，$f(1) = \dfrac{1}{2}$。（1）求 $a, b$；（2）$f$ 在 $[1, +\infty)$ 单调性。

**思路**　（1）奇函数 $f(0) = 0$（若 $0$ 在定义域）得 $b$；$f(1)$ 得 $a$。（2）作差判号。

**解答**　(1) 定义域 $\mathbb{R}$，奇函数 $\Rightarrow f(0) = 0 \Rightarrow \dfrac{b}{1} = 0 \Rightarrow b = 0$。

$f(1) = \dfrac{a \cdot 1}{2} = \dfrac{a}{2} = \dfrac{1}{2} \Rightarrow a = 1$。

故 $f(x) = \dfrac{x}{x^2 + 1}$。

(2) 任取 $x_1, x_2 \in [1, +\infty)$，$x_1 < x_2$。

$f(x_1) - f(x_2) = \dfrac{x_1}{x_1^2 + 1} - \dfrac{x_2}{x_2^2 + 1} = \dfrac{x_1(x_2^2 + 1) - x_2(x_1^2 + 1)}{(x_1^2 + 1)(x_2^2 + 1)} = \dfrac{x_1 x_2^2 - x_1^2 x_2 + x_1 - x_2}{(x_1^2 + 1)(x_2^2 + 1)} = \dfrac{x_1 x_2(x_2 - x_1) + (x_1 - x_2)}{(x_1^2 + 1)(x_2^2 + 1)} = \dfrac{(x_1 - x_2)(1 - x_1 x_2)}{(x_1^2 + 1)(x_2^2 + 1)}$。

由 $x_1, x_2 \geq 1$ 知 $x_1 x_2 \geq 1$，$1 - x_1 x_2 \leq 0$；分母正；$x_1 - x_2 < 0$。

故 $f(x_1) - f(x_2) = \dfrac{(-)(-)}{(+)} \geq 0$，即 $f(x_1) \geq f(x_2)$。

实际上 $x_1 < x_2$ 时 $x_1 - x_2 < 0$；$1 - x_1 x_2 \leq 0$ 在 $x_1 = x_2 = 1$ 为零，其余严格负。

得 $f(x_1) > f(x_2)$（严格），$f$ 在 $[1, +\infty)$ 单调递减。

**总结**　含参奇 / 偶函数定 $a, b$ 第一招"代特殊值（$0$ 或对称值）"；分式定义法判单调性必须把差**因式分解到可判号**。

---

## C.39 [提升] Part 3/03+04

**题目回顾**：$f(x + y) = f(x) + f(y)$；$x > 0$ 时 $f(x) < 0$。求 $f(0)$、判断奇偶性、判断单调性。

**思路**　抽象函数标准三步：代特殊值（$0, 0$）、令 $y = -x$、定义法判单调。→ toolkit/06 抽象函数。

**解答**　(1) 令 $x = y = 0$：$f(0) = f(0) + f(0) \Rightarrow f(0) = 0$。

(2) 令 $y = -x$：$f(x + (-x)) = f(x) + f(-x) \Rightarrow f(0) = f(x) + f(-x) \Rightarrow 0 = f(x) + f(-x) \Rightarrow f(-x) = -f(x)$。

故 $f$ 为奇函数。

(3) 任取 $x_1 < x_2$，令 $\Delta = x_2 - x_1 > 0$。

$f(x_2) = f(x_1 + \Delta) = f(x_1) + f(\Delta)$；

故 $f(x_2) - f(x_1) = f(\Delta)$。

由已知 $\Delta > 0 \Rightarrow f(\Delta) < 0$；故 $f(x_2) - f(x_1) < 0$，即 $f(x_1) > f(x_2)$。

$f$ 在 $\mathbb{R}$ 上单调递减。

**总结**　抽象函数三连击：① 代 $0, 0$ 求 $f(0)$；② 代 $y = -x$ 判奇偶；③ 利用恒等式 + 已知条件作差判单调。

---

## C.40 [提升] Part 3/07

**题目回顾**：进价 $40$，售价 $50$ 时每天卖 $500$ 件，每涨 $1$ 元少 $10$ 件。$x \geq 50$，求 $W(x)$ 与最大利润。

**思路**　建立"销量 - 售价"线性关系，再写利润 = (售价 - 进价) × 销量。

**解答**　销量：$x = 50$ 时 $500$ 件，每涨 $1$ 元少 $10$ 件，故销量为 $500 - 10(x - 50) = 1000 - 10x$。

需销量 $> 0$：$1000 - 10x > 0 \Rightarrow x < 100$。

利润 $W(x) = (x - 40) \cdot (1000 - 10x) = -10x^2 + 1400x - 40000$（$50 \leq x < 100$）。

求最值：开口向下，顶点 $x = -\dfrac{1400}{2 \cdot (-10)} = 70$（在 $[50, 100)$ 内）。

$W(70) = -10 \cdot 4900 + 1400 \cdot 70 - 40000 = -49000 + 98000 - 40000 = 9000$。

最大利润 $9000$ 元，售价 $70$ 元。

**总结**　实际应用题套路："设变量 → 表销量 / 表收入 / 表利润 → 求最值"。注意定义域（销量 $> 0$）。

---

## Part 4 指数对数幂

---

## C.41 [基础] Part 4/01

**题目回顾**：计算 $\left(\dfrac{1}{8}\right)^{-\frac{2}{3}} + 16^{0.75}$。

**思路**　化为最简底数 $\to$ 整理指数。

**解答**　$\left(\dfrac{1}{8}\right)^{-\frac{2}{3}} = 8^{\frac{2}{3}} = (2^3)^{\frac{2}{3}} = 2^2 = 4$；

$16^{0.75} = 16^{\frac{3}{4}} = (2^4)^{\frac{3}{4}} = 2^3 = 8$。

合计 $4 + 8 = 12$。

**总结**　指数运算永远"统一底数"；负指数变倒数、分数指数变根式（或直接幂指数）。

---

## C.42 [基础] Part 4/02

**题目回顾**：$\log_2 8 + \log_3 \dfrac{1}{9} + \log_5 \sqrt{5}$。

**思路**　每项化为简单幂的形式。

**解答**　$\log_2 8 = \log_2 2^3 = 3$；

$\log_3 \dfrac{1}{9} = \log_3 3^{-2} = -2$；

$\log_5 \sqrt{5} = \log_5 5^{1/2} = \dfrac{1}{2}$。

合计 $3 + (-2) + \dfrac{1}{2} = \dfrac{3}{2}$。

**总结**　对数运算第一步"把真数写成底数的幂"，再用 $\log_a a^n = n$。

---

## C.43 [基础] Part 4/02

**题目回顾**：$\log_2 3 \cdot \log_3 8$。

**思路**　换底公式 $\log_a b = \dfrac{\ln b}{\ln a}$，或直接利用 $\log_a b \cdot \log_b c = \log_a c$。

**解答**　$\log_2 3 \cdot \log_3 8 = \log_2 8 = \log_2 2^3 = 3$。

**总结**　记牢链式 $\log_a b \cdot \log_b c = \log_a c$，乘对乘错一步差万里。

---

## C.44 [基础] Part 4/03

**题目回顾**：比较 $0.6^{0.5}$、$0.5^{0.6}$、$0.6^{0.6}$（从小到大）。

**思路**　同底数比较 / 同指数比较，分别处理。→ toolkit/01 结构识别。

**解答**　记 $A = 0.6^{0.5}$，$B = 0.5^{0.6}$，$C = 0.6^{0.6}$。

**$A$ 与 $C$**：同底 $0.6$，底 $\in (0, 1)$ 指数函数递减，指数 $0.5 < 0.6$，故 $A > C$。

**$B$ 与 $C$**：同指数 $0.6$，幂函数 $y = x^{0.6}$（$x > 0$）单调递增（指数正），底 $0.5 < 0.6$，故 $B < C$。

综上 $B < C < A$，即 $0.5^{0.6} < 0.6^{0.6} < 0.6^{0.5}$。

**总结**　比较幂值"找共同点"：同底用指数函数单调性，同指数用幂函数单调性。

---

## C.45 [中档] Part 4/01

**题目回顾**：解 $2^{x+1} + 2^x = 12$。

**思路**　把 $2^{x+1}$ 化为 $2 \cdot 2^x$，提公因式。

**解答**　$2^{x+1} + 2^x = 2 \cdot 2^x + 2^x = 3 \cdot 2^x = 12 \Rightarrow 2^x = 4 = 2^2 \Rightarrow x = 2$。

**总结**　指数方程见"$a^{x+k}$ 形式"立刻拆 $a^k \cdot a^x$ 提公因式。

---

## C.46 [中档] Part 4/02

**题目回顾**：$\log_a 2 = m$，$\log_a 3 = n$，表 $\log_a 12$ 与 $\log_a \dfrac{9}{8}$。

**思路**　分解真数为已知数（$2, 3$）的乘积或商，套用对数运算法则。

**解答**　$\log_a 12 = \log_a (4 \cdot 3) = \log_a 4 + \log_a 3 = 2\log_a 2 + \log_a 3 = 2m + n$。

$\log_a \dfrac{9}{8} = \log_a 9 - \log_a 8 = 2\log_a 3 - 3\log_a 2 = 2n - 3m$。

**总结**　对数法则三条 $\log(ab) = \log a + \log b$、$\log(a/b) = \log a - \log b$、$\log a^k = k\log a$ 是组合表达式必杀技。

---

## C.47 [中档] Part 4/01

**题目回顾**：解 $9^x - 3^{x+1} + 2 = 0$。

**思路**　$9^x = (3^x)^2$ 换元 $t = 3^x > 0$ 化为二次。→ toolkit/02 换元。

**解答**　$9^x = (3^2)^x = (3^x)^2$；$3^{x+1} = 3 \cdot 3^x$。

令 $t = 3^x > 0$，原方程 $t^2 - 3t + 2 = 0 \Rightarrow (t - 1)(t - 2) = 0 \Rightarrow t = 1$ 或 $t = 2$。

回代：$3^x = 1 \Rightarrow x = 0$；$3^x = 2 \Rightarrow x = \log_3 2$。

**总结**　见"$a^{2x}$ + $a^x$" 立刻换元 $t = a^x$ 化二次。

---

## C.48 [中档] Part 4/02

**题目回顾**：$f(x) = \log_a(x^2 - ax + 3)$（$a > 0, a \neq 1$）在 $[2, +\infty)$ 单调递增，求 $a$。

**思路**　复合函数 $f = \log_a u$，$u = x^2 - ax + 3$；"同增异减"。→ toolkit/07 复合。

**解答**　设 $u(x) = x^2 - ax + 3$，$u$ 的对称轴 $x = \dfrac{a}{2}$，开口向上。$u$ 在 $\left[\dfrac{a}{2}, +\infty\right)$ 递增。

要 $f = \log_a u$ 在 $[2, +\infty)$ 递增。分两种情况：

**情形 1**：$a > 1$，外层 $\log_a$ 递增；内层 $u$ 必须在 $[2, +\infty)$ 递增 $\Rightarrow \dfrac{a}{2} \leq 2 \Rightarrow a \leq 4$。

且要 $u(2) > 0$（真数 $> 0$）：$4 - 2a + 3 > 0 \Rightarrow a < \dfrac{7}{2}$。

综合 $1 < a < \dfrac{7}{2}$。

**情形 2**：$0 < a < 1$，外层 $\log_a$ 递减，要 $f$ 递增 $\Rightarrow u$ 必须在 $[2, +\infty)$ 递减。

但 $u$ 在 $\left[\dfrac{a}{2}, +\infty\right)$ 总是递增，不可能在 $[2, +\infty)$ 递减——情形 2 无解。

综上 $1 < a < \dfrac{7}{2}$。

**总结**　对数 / 指数复合单调性"同增异减"；别忘了真数 $> 0$ 的额外约束。

---

## C.49 [提升] Part 4/01

**题目回顾**：$2^a = 5^b = 10$，求 $\dfrac{1}{a} + \dfrac{1}{b}$。

**思路**　两边取对数将 $a, b$ 用 $\log$ 表达，再代入。

**解答**　$2^a = 10 \Rightarrow a = \log_2 10$；$5^b = 10 \Rightarrow b = \log_5 10$。

$\dfrac{1}{a} = \dfrac{1}{\log_2 10} = \log_{10} 2$（换底）；$\dfrac{1}{b} = \log_{10} 5$。

故 $\dfrac{1}{a} + \dfrac{1}{b} = \log_{10} 2 + \log_{10} 5 = \log_{10} 10 = 1$。

**总结**　遇到"$a^x = c$" 立刻取对数 + 换底；$\dfrac{1}{\log_a b} = \log_b a$ 这一关键变形要熟。

---

## C.50 [提升] Part 4/02

**题目回顾**：$\log_a (a^2 + 1) < \log_a (2a) < 0$，求 $a$。

**思路**　对数符号 + 单调性同时讨论，分 $a > 1$ 与 $0 < a < 1$。→ toolkit/11 分类。

**解答**　**前提**：$a^2 + 1 > 0$（恒成立）；$2a > 0 \Rightarrow a > 0$；$a \neq 1$。

**情形 1**：$a > 1$。$\log_a$ 递增。

$\log_a (2a) < 0 = \log_a 1 \Rightarrow 2a < 1 \Rightarrow a < \dfrac{1}{2}$；与 $a > 1$ 矛盾，无解。

**情形 2**：$0 < a < 1$。$\log_a$ 递减。

$\log_a (a^2 + 1) < \log_a (2a) \Rightarrow a^2 + 1 > 2a \Rightarrow (a - 1)^2 > 0 \Rightarrow a \neq 1$ ✓（在 $0 < a < 1$ 内显然成立）；

$\log_a (2a) < 0 = \log_a 1 \Rightarrow 2a > 1 \Rightarrow a > \dfrac{1}{2}$。

综合 $\dfrac{1}{2} < a < 1$。

**总结**　对数不等式"先讨论底数 $> 1$ 还是 $< 1$"，再翻译成真数大小关系（注意 $< 1$ 时不等号反向）。

---

## Part 5 三角函数

---

## C.51 [基础] Part 5/01

**题目回顾**：将 $\dfrac{2\pi}{3}$ 化角度；将 $-150°$ 化弧度。

**思路**　$180° = \pi$ 弧度的换算。

**解答**　$\dfrac{2\pi}{3} = \dfrac{2}{3} \times 180° = 120°$。

$-150° = -150 \times \dfrac{\pi}{180} = -\dfrac{5\pi}{6}$。

**总结**　$\pi$ 换 $180°$，$180°$ 换 $\pi$；同乘同除是单位换算的核心。

---

## C.52 [基础] Part 5/01

**题目回顾**：扇形半径 $3$，弧长 $\pi$，求圆心角与面积。

**思路**　$l = r\theta$ 求 $\theta$；$S = \dfrac{1}{2}r^2\theta$ 求 $S$。

**解答**　$\theta = \dfrac{l}{r} = \dfrac{\pi}{3}$。

$S = \dfrac{1}{2} r^2 \theta = \dfrac{1}{2} \cdot 9 \cdot \dfrac{\pi}{3} = \dfrac{3\pi}{2}$。

**总结**　扇形两个公式必背：弧长 $l = r\theta$、面积 $S = \dfrac{1}{2}r^2\theta = \dfrac{1}{2}lr$。

---

## C.53 [基础] Part 5/02

**题目回顾**：$\sin\alpha = \dfrac{3}{5}$，$\alpha$ 在第二象限，求 $\cos\alpha$ 与 $\tan\alpha$。

**思路**　先用 $\sin^2 + \cos^2 = 1$ 求 $|\cos\alpha|$，再用象限定符号。

**解答**　$\cos^2\alpha = 1 - \sin^2\alpha = 1 - \dfrac{9}{25} = \dfrac{16}{25} \Rightarrow |\cos\alpha| = \dfrac{4}{5}$。

第二象限 $\cos < 0$，故 $\cos\alpha = -\dfrac{4}{5}$。

$\tan\alpha = \dfrac{\sin\alpha}{\cos\alpha} = \dfrac{3/5}{-4/5} = -\dfrac{3}{4}$。

**总结**　已知三角函数一个值求另一个："勾股求绝对值 + 象限定符号"。

---

## C.54 [基础] Part 5/04

**题目回顾**：化简 $\dfrac{\sin(\pi - \alpha)\cos(2\pi - \alpha)}{\cos(\pi/2 + \alpha)}$。

**思路**　逐项用诱导公式"奇变偶不变，符号看象限"。

**解答**　$\sin(\pi - \alpha) = \sin\alpha$（偶数倍 $\pi$ 关系，不变）；

$\cos(2\pi - \alpha) = \cos(-\alpha) = \cos\alpha$；

$\cos(\pi/2 + \alpha) = -\sin\alpha$（奇数倍 $\pi/2$，函数互换；象限为第二象限 $\cos < 0$）。

原式 $= \dfrac{\sin\alpha \cdot \cos\alpha}{-\sin\alpha} = -\cos\alpha$。

**总结**　诱导公式"奇变偶不变" 指 $\pi/2$ 的倍数：奇数倍变三角函数种类，偶数倍不变。

---

## C.55 [基础] Part 5/03

**题目回顾**：$y = 2\sin\left(2x - \dfrac{\pi}{3}\right)$，写振幅 / 周期 / 初相，求最大值。

**思路**　标准形式 $y = A\sin(\omega x + \varphi)$ 的参数对应。

**解答**　$A = 2$（振幅）；$T = \dfrac{2\pi}{\omega} = \dfrac{2\pi}{2} = \pi$（周期）；$\varphi = -\dfrac{\pi}{3}$（初相）。

最大值 $= A = 2$（当 $2x - \dfrac{\pi}{3} = \dfrac{\pi}{2} + 2k\pi$ 时取到）。

**总结**　$y = A\sin(\omega x + \varphi)$ 中 $A$ 是振幅、$T = 2\pi/\omega$ 是周期、$\varphi$ 是初相、$A$ 是最值。

---

## C.56 [基础] Part 5/05

**题目回顾**：计算 $\sin 75°$。

**思路**　$\sin 75° = \sin(45° + 30°)$ 用和角公式。

**解答**　$\sin 75° = \sin(45° + 30°) = \sin 45° \cos 30° + \cos 45° \sin 30°$。

$= \dfrac{\sqrt{2}}{2} \cdot \dfrac{\sqrt{3}}{2} + \dfrac{\sqrt{2}}{2} \cdot \dfrac{1}{2} = \dfrac{\sqrt{6} + \sqrt{2}}{4}$。

**总结**　特殊角拆分"$75° = 45° + 30°$"、"$15° = 45° - 30°$" 是和角公式的典型应用。

---

## C.57 [基础] Part 5/07

**题目回顾**：$\triangle ABC$ 中 $a = 2, b = \sqrt{3}, A = 60°$，求 $\sin B$。

**思路**　正弦定理 $\dfrac{a}{\sin A} = \dfrac{b}{\sin B}$。

**解答**　$\sin B = \dfrac{b \sin A}{a} = \dfrac{\sqrt{3} \cdot \sin 60°}{2} = \dfrac{\sqrt{3} \cdot \dfrac{\sqrt{3}}{2}}{2} = \dfrac{3/2}{2} = \dfrac{3}{4}$。

**总结**　正弦定理是"边对应角"的桥梁；已知"边 - 边 - 角"求角用此公式。

---

## C.58 [基础] Part 5/07

**题目回顾**：$\triangle ABC$ 中 $a = 2, b = 3, C = 60°$，求 $c$。

**思路**　余弦定理 $c^2 = a^2 + b^2 - 2ab\cos C$。

**解答**　$c^2 = 4 + 9 - 2 \cdot 2 \cdot 3 \cdot \cos 60° = 13 - 12 \cdot \dfrac{1}{2} = 13 - 6 = 7$。

$c = \sqrt{7}$。

**总结**　余弦定理用于"两边夹一角求对边"或"三边求角"，几乎是解三角形必备。

---

## C.59 [中档] Part 5/02

**题目回顾**：$\sin\alpha + \cos\alpha = \dfrac{1}{5}$，$\alpha \in (0, \pi)$，求 $\sin\alpha - \cos\alpha$ 与 $\sin\alpha\cos\alpha$。

**思路**　平方得 $\sin\alpha\cos\alpha$；$(\sin - \cos)^2 = 1 - 2\sin\cos$。→ toolkit/01 结构识别。

**解答**　$(\sin\alpha + \cos\alpha)^2 = \sin^2\alpha + 2\sin\alpha\cos\alpha + \cos^2\alpha = 1 + 2\sin\alpha\cos\alpha = \dfrac{1}{25}$。

故 $2\sin\alpha\cos\alpha = -\dfrac{24}{25}$，$\sin\alpha\cos\alpha = -\dfrac{12}{25}$。

$(\sin\alpha - \cos\alpha)^2 = 1 - 2\sin\alpha\cos\alpha = 1 + \dfrac{24}{25} = \dfrac{49}{25}$。

$\sin\alpha - \cos\alpha = \pm\dfrac{7}{5}$。

**定符号**：$\sin\alpha\cos\alpha = -\dfrac{12}{25} < 0$，结合 $\sin\alpha + \cos\alpha > 0$，必 $\sin > 0$，$\cos < 0$，即 $\alpha \in \left(\dfrac{\pi}{2}, \pi\right)$，此时 $\sin\alpha > 0 > \cos\alpha$ 所以 $\sin - \cos > 0$。

故 $\sin\alpha - \cos\alpha = \dfrac{7}{5}$。

**总结**　"$\sin + \cos$"、"$\sin - \cos$"、"$\sin\cos$" 三者通过平方互推；最后定符号要用象限。

---

## C.60 [中档] Part 5/04

**题目回顾**：化简 $\dfrac{\cos(\pi/2 + \alpha)\sin(\pi + \alpha)}{\cos(\pi - \alpha)\sin(3\pi/2 - \alpha)}$。

**思路**　逐项用诱导公式。

**解答**　$\cos(\pi/2 + \alpha) = -\sin\alpha$；

$\sin(\pi + \alpha) = -\sin\alpha$；

$\cos(\pi - \alpha) = -\cos\alpha$；

$\sin(3\pi/2 - \alpha) = -\cos\alpha$。

原式 $= \dfrac{(-\sin\alpha)(-\sin\alpha)}{(-\cos\alpha)(-\cos\alpha)} = \dfrac{\sin^2\alpha}{\cos^2\alpha} = \tan^2\alpha$。

**总结**　诱导公式逐项处理后约分；记牢六组公式 + "奇变偶不变、符号看象限"。

---

## C.61 [中档] Part 5/05

**题目回顾**：$\sin\alpha = \dfrac{4}{5}$，$\alpha \in (\pi/2, \pi)$，求 $\sin(\alpha + \pi/4)$。

**思路**　和角公式 $\sin(\alpha + \beta)$ + 用 $\sin\alpha$ 推 $\cos\alpha$。

**解答**　$\alpha$ 第二象限 $\cos\alpha < 0$；$\cos^2\alpha = 1 - \dfrac{16}{25} = \dfrac{9}{25} \Rightarrow \cos\alpha = -\dfrac{3}{5}$。

$\sin(\alpha + \pi/4) = \sin\alpha\cos(\pi/4) + \cos\alpha\sin(\pi/4) = \dfrac{4}{5} \cdot \dfrac{\sqrt{2}}{2} + \left(-\dfrac{3}{5}\right) \cdot \dfrac{\sqrt{2}}{2} = \dfrac{\sqrt{2}}{2} \cdot \dfrac{1}{5} = \dfrac{\sqrt{2}}{10}$。

**总结**　"已知 $\sin\alpha$ + 象限求 $\sin(\alpha + \pi/4)$" 是和角公式的入门标杆题。

---

## C.62 [中档] Part 5/05

**题目回顾**：$\tan\alpha = 2$，求 $\tan(\alpha - \pi/4)$ 与 $\sin 2\alpha$。

**思路**　$\tan$ 差公式 + 倍角"全部转 $\tan$"。→ toolkit/02 换元（齐次处理）。

**解答**　$\tan(\alpha - \pi/4) = \dfrac{\tan\alpha - 1}{1 + \tan\alpha} = \dfrac{2 - 1}{1 + 2} = \dfrac{1}{3}$。

$\sin 2\alpha = \dfrac{2\sin\alpha\cos\alpha}{\sin^2\alpha + \cos^2\alpha} = \dfrac{2\tan\alpha}{1 + \tan^2\alpha} = \dfrac{4}{1 + 4} = \dfrac{4}{5}$（齐次化）。

**总结**　已知 $\tan$ 求"$\sin / \cos$ 表达式"，把分子分母同除以 $\cos^2$ 化为 $\tan$ 的表达式（齐次化）。

---

## C.63 [中档] Part 5/05

**题目回顾**：$\cos 75°\cos 15° - \sin 75°\sin 15°$。

**思路**　反向识别 $\cos(A + B)$ 公式。→ toolkit/01 结构识别。

**解答**　$\cos 75°\cos 15° - \sin 75°\sin 15° = \cos(75° + 15°) = \cos 90° = 0$。

**总结**　"乘积差 = $\cos(A + B)$"，"乘积和 = $\cos(A - B)$"，反向识别公式秒解。

---

## C.64 [中档] Part 5/05

**题目回顾**：化简 $\dfrac{\sin 2\alpha}{1 + \cos 2\alpha}$。

**思路**　倍角 $\sin 2\alpha = 2\sin\alpha\cos\alpha$；$1 + \cos 2\alpha = 2\cos^2\alpha$。

**解答**　$\dfrac{\sin 2\alpha}{1 + \cos 2\alpha} = \dfrac{2\sin\alpha\cos\alpha}{2\cos^2\alpha} = \dfrac{\sin\alpha}{\cos\alpha} = \tan\alpha$。

**总结**　"$1 + \cos 2\alpha = 2\cos^2\alpha$"、"$1 - \cos 2\alpha = 2\sin^2\alpha$" 是降幂 / 升幂之间的桥梁。

---

## C.65 [中档] Part 5/05

**题目回顾**：化 $\sqrt{3}\sin x + \cos x$ 为 $R\sin(x + \varphi)$，求最大值。

**思路**　辅助角公式。

**解答**　$R = \sqrt{(\sqrt{3})^2 + 1^2} = 2$。

$\sqrt{3}\sin x + \cos x = 2\left(\dfrac{\sqrt{3}}{2}\sin x + \dfrac{1}{2}\cos x\right) = 2\left(\sin x \cos\dfrac{\pi}{6} + \cos x \sin\dfrac{\pi}{6}\right) = 2\sin\left(x + \dfrac{\pi}{6}\right)$。

最大值为 $2$。

**总结**　辅助角公式 $a\sin x + b\cos x = \sqrt{a^2 + b^2}\sin(x + \varphi)$：$R = \sqrt{a^2 + b^2}$，$\cos\varphi = a/R$，$\sin\varphi = b/R$。

---

## C.66 [中档] Part 5/03

**题目回顾**：$y = \sin(\omega x + \varphi)$（$\omega > 0$，$|\varphi| < \pi/2$）周期 $\pi$，过 $(\pi/6, 1)$，求 $\omega, \varphi$。

**思路**　周期定 $\omega$；代入求 $\varphi$。

**解答**　$T = \dfrac{2\pi}{\omega} = \pi \Rightarrow \omega = 2$。

代入 $(\pi/6, 1)$：$\sin\left(2 \cdot \dfrac{\pi}{6} + \varphi\right) = \sin\left(\dfrac{\pi}{3} + \varphi\right) = 1 \Rightarrow \dfrac{\pi}{3} + \varphi = \dfrac{\pi}{2} + 2k\pi \Rightarrow \varphi = \dfrac{\pi}{6} + 2k\pi$。

由 $|\varphi| < \pi/2$ 取 $k = 0$，$\varphi = \dfrac{\pi}{6}$。

**总结**　"周期 $\to \omega$"、"特殊点 $\to \varphi$" 是定 $A\sin(\omega x + \varphi)$ 标准两步。

---

## C.67 [中档] Part 5/03

**题目回顾**：$y = 2\sin(2x + \pi/6)$ 在 $[-\pi/4, \pi/4]$ 上的最值。

**思路**　换元 $u = 2x + \pi/6$；先求 $u$ 范围。→ toolkit/02 换元。

**解答**　$x \in [-\pi/4, \pi/4] \Rightarrow 2x \in [-\pi/2, \pi/2] \Rightarrow u = 2x + \pi/6 \in [-\pi/3, 2\pi/3]$。

$y = 2\sin u$ 在 $u \in [-\pi/3, 2\pi/3]$。

$\sin u$ 在 $[-\pi/3, \pi/2]$ 递增，$[\pi/2, 2\pi/3]$ 递减。最大值在 $u = \pi/2$ 处 $\sin = 1$；最小值在 $u = -\pi/3$ 处 $\sin = -\dfrac{\sqrt{3}}{2}$。

所以 $y_{\max} = 2$，$y_{\min} = -\sqrt{3}$。

**总结**　"$y = A\sin(\omega x + \varphi)$ 在闭区间求最值" 必做换元转化为标准 $\sin u$ 问题，再看 $u$ 的范围内 $\sin$ 的极值位置。

---

## C.68 [中档] Part 5/06

**题目回顾**：$[0, 2\pi]$ 内解 $2\sin x - 1 = 0$。

**思路**　$\sin x = 1/2$，单位圆上找所有满足的角。

**解答**　$\sin x = \dfrac{1}{2}$。

参考角 $\pi/6$；$\sin > 0$ 在第一、二象限：

第一象限：$x = \pi/6$；第二象限：$x = \pi - \pi/6 = 5\pi/6$。

故 $x \in \left\{\dfrac{\pi}{6}, \dfrac{5\pi}{6}\right\}$。

**总结**　简单三角方程求解："参考角 + 象限符号 + 范围"三步组合，单位圆里找点。

---

## C.69 [中档] Part 5/07

**题目回顾**：$a = 7, b = 5, c = 3$，求 $\cos A$。

**思路**　余弦定理 $\cos A = \dfrac{b^2 + c^2 - a^2}{2bc}$。

**解答**　$\cos A = \dfrac{25 + 9 - 49}{2 \cdot 5 \cdot 3} = \dfrac{-15}{30} = -\dfrac{1}{2}$。

（顺便：$A = 2\pi/3 = 120°$。）

**总结**　"三边求一角" 永远用余弦定理；得 $\cos$ 后可直接识别特殊角。

---

## C.70 [中档] Part 5/07

**题目回顾**：$a = 2, b = 2\sqrt{2}, A = 30°$，判断是否存在，求 $B$。

**思路**　ASS（边边角）二解 / 一解 / 无解判别。→ toolkit/11 分类。

**解答**　正弦定理 $\dfrac{a}{\sin A} = \dfrac{b}{\sin B}$：

$\sin B = \dfrac{b \sin A}{a} = \dfrac{2\sqrt{2} \cdot \sin 30°}{2} = \dfrac{2\sqrt{2} \cdot 1/2}{2} = \dfrac{\sqrt{2}}{2}$。

$\sin B = \dfrac{\sqrt{2}}{2} \Rightarrow B = 45°$ 或 $B = 135°$。

**检验**：$a < b$（$2 < 2\sqrt{2}$）$\Rightarrow A < B$，故 $B > 30°$ 总成立；

- $B = 45°$：$A + B = 75° < 180°$ ✓；
- $B = 135°$：$A + B = 165° < 180°$ ✓。

两解均成立，$B \in \{45°, 135°\}$。

**总结**　ASS 情形必查二解：解出 $\sin B$ 两值后逐一验证 $A + B < 180°$，否则舍。

---

## C.71 [提升] Part 5/05

**题目回顾**：$\cos\alpha + \cos\beta = \dfrac{1}{2}$，$\sin\alpha + \sin\beta = \dfrac{1}{3}$，求 $\cos(\alpha - \beta)$。

**思路**　两式平方相加 + 用差角公式。→ toolkit/01 结构识别。

**解答**　$(\cos\alpha + \cos\beta)^2 + (\sin\alpha + \sin\beta)^2$

$= \cos^2\alpha + 2\cos\alpha\cos\beta + \cos^2\beta + \sin^2\alpha + 2\sin\alpha\sin\beta + \sin^2\beta$

$= 2 + 2(\cos\alpha\cos\beta + \sin\alpha\sin\beta) = 2 + 2\cos(\alpha - \beta)$。

代入：$\dfrac{1}{4} + \dfrac{1}{9} = \dfrac{13}{36} = 2 + 2\cos(\alpha - \beta)$。

$\cos(\alpha - \beta) = \dfrac{13/36 - 2}{2} = \dfrac{13 - 72}{72} = -\dfrac{59}{72}$。

**总结**　看到"$\cos + \cos$ 与 $\sin + \sin$" 平方相加套路，必然出现 $\cos(\alpha - \beta)$。

---

## C.72 [提升] Part 5/05

**题目回顾**：化简 $\dfrac{2\sin 50° + \sin 80°(1 + \sqrt{3}\tan 10°)}{\sqrt{1 + \cos 10°}}$。

**思路**　分子分母各自化简——分子用辅助角和恒等式；分母用倍角降幂。

**解答**　**分母**：$1 + \cos 10° = 2\cos^2 5°$，$\sqrt{1 + \cos 10°} = \sqrt{2}|\cos 5°| = \sqrt{2}\cos 5°$（$\cos 5° > 0$）。

**分子**：先看 $1 + \sqrt{3}\tan 10° = \dfrac{\cos 10° + \sqrt{3}\sin 10°}{\cos 10°} = \dfrac{2(\cos 10° \cdot 1/2 + \sin 10° \cdot \sqrt{3}/2)}{\cos 10°} = \dfrac{2\cos(10° - 60°)}{\cos 10°} = \dfrac{2\cos(-50°)}{\cos 10°} = \dfrac{2\cos 50°}{\cos 10°}$。

（注：合并形式 $a\cos x + b\sin x$ 化为 $R\cos(x - \varphi)$。）

所以 $\sin 80° (1 + \sqrt{3}\tan 10°) = \sin 80° \cdot \dfrac{2\cos 50°}{\cos 10°} = \dfrac{2\sin 80° \cos 50°}{\cos 10°}$。

由 $\sin 80° = \cos 10°$，简化为 $\dfrac{2\cos 10° \cos 50°}{\cos 10°} = 2\cos 50°$。

故分子 $= 2\sin 50° + 2\cos 50° = 2(\sin 50° + \cos 50°) = 2 \cdot \sqrt{2}\sin(50° + 45°) = 2\sqrt{2}\sin 95° = 2\sqrt{2}\cos 5°$。

原式 $= \dfrac{2\sqrt{2}\cos 5°}{\sqrt{2}\cos 5°} = 2$。

**总结**　复杂三角化简："分母先用倍角降幂、分子先合并 $a\sin + b\cos$" 双线作战；耐心一步一步推。

---

## C.73 [提升] Part 5/03+05

**题目回顾**：$f(x) = \sin 2x + \sqrt{3}\cos 2x$。（1）化标准形式；（2）周期与单调递增区间；（3）$[0, \pi/2]$ 上最值。

**思路**　辅助角 → 标准 $A\sin(\omega x + \varphi)$ → 用模板分析。

**解答**　(1) $f(x) = 2\left(\dfrac{1}{2}\sin 2x + \dfrac{\sqrt{3}}{2}\cos 2x\right) = 2\left(\sin 2x\cos\dfrac{\pi}{3} + \cos 2x\sin\dfrac{\pi}{3}\right) = 2\sin\left(2x + \dfrac{\pi}{3}\right)$。

(2) 周期 $T = \dfrac{2\pi}{2} = \pi$。

单调递增区间：$2x + \dfrac{\pi}{3} \in \left[-\dfrac{\pi}{2} + 2k\pi, \dfrac{\pi}{2} + 2k\pi\right]$，解出 $x \in \left[-\dfrac{5\pi}{12} + k\pi, \dfrac{\pi}{12} + k\pi\right]$（$k \in \mathbb{Z}$）。

(3) $x \in [0, \pi/2] \Rightarrow 2x + \pi/3 \in [\pi/3, 4\pi/3]$。

$\sin u$ 在 $[\pi/3, \pi/2]$ 递增、$[\pi/2, 4\pi/3]$ 递减。

最大值 $\sin(\pi/2) = 1 \Rightarrow f = 2$（在 $x = \pi/12$）；

最小值 $\sin(4\pi/3) = -\sqrt{3}/2 \Rightarrow f = -\sqrt{3}$（在 $x = \pi/2$）。

**总结**　$y = a\sin\omega x + b\cos\omega x$ 三连击：① 辅助角化标准；② $T, $ 单调区间用模板；③ 闭区间最值用换元 + $\sin u$ 范围。

---

## C.74 [提升] Part 5/07

**题目回顾**：$\triangle ABC$ 中 $b^2 + c^2 - a^2 = bc$，$a = \sqrt{3}$。（1）求 $A$；（2）求面积最大值。

**思路**　（1）余弦定理识别 $A$；（2）面积 $\dfrac{1}{2}bc\sin A$ 配合不等式 / 余弦定理。→ toolkit/01 结构识别 + toolkit/05 基本不等式。

**解答**　(1) $\cos A = \dfrac{b^2 + c^2 - a^2}{2bc} = \dfrac{bc}{2bc} = \dfrac{1}{2} \Rightarrow A = \dfrac{\pi}{3}$。

(2) 由余弦定理 $a^2 = b^2 + c^2 - 2bc\cos A = b^2 + c^2 - bc = 3$。

由基本不等式 $b^2 + c^2 \geq 2bc$，故 $3 = b^2 + c^2 - bc \geq 2bc - bc = bc \Rightarrow bc \leq 3$。

面积 $S = \dfrac{1}{2}bc\sin A = \dfrac{1}{2}bc \cdot \dfrac{\sqrt{3}}{2} = \dfrac{\sqrt{3}}{4}bc \leq \dfrac{3\sqrt{3}}{4}$。

取等 $b = c$；此时由 $b^2 + c^2 - bc = 3$ 得 $b^2 = 3 \Rightarrow b = \sqrt{3}$，可达。

最大值 $\dfrac{3\sqrt{3}}{4}$。

**总结**　"解三角形面积最值" 套路：余弦定理建 $b, c$ 约束 + 基本不等式定 $bc$ 上限 + 面积公式。

---

## C.75 [提升] Part 5/05+07

**题目回顾**：$\triangle ABC$ 中 $\sin A = 2\sin B\cos C$，且 $a^2 - b^2 = bc$。求 $A$。

**思路**　第一式用 $\sin A = \sin(B + C)$ 展开得边角关系；第二式接余弦定理求 $\cos A$。→ toolkit/01 结构识别。

**解答**　**第一步：分析 $\sin A = 2\sin B \cos C$。**

由 $A + B + C = \pi$ 得 $A = \pi - (B + C)$，故 $\sin A = \sin(B + C) = \sin B\cos C + \cos B\sin C$。

代入：$\sin B\cos C + \cos B\sin C = 2\sin B\cos C \Rightarrow \cos B\sin C = \sin B\cos C \Rightarrow \sin(B - C) = 0$。

由 $B, C \in (0, \pi) \Rightarrow B - C \in (-\pi, \pi)$，故 $B - C = 0$，即 $B = C$。

由此 $b = c$。

**第二步：用第二式 $a^2 - b^2 = bc$。**

代 $c = b$：$a^2 - b^2 = b \cdot b = b^2 \Rightarrow a^2 = 2b^2$。

**第三步：余弦定理求 $\cos A$。**

$\cos A = \dfrac{b^2 + c^2 - a^2}{2bc} = \dfrac{b^2 + b^2 - 2b^2}{2b \cdot b} = \dfrac{0}{2b^2} = 0$。

故 $A = \dfrac{\pi}{2}$。

**总结**　"$\sin A = \sin(B + C)$" 是三角形里的关键替换，把三角函数等式翻译成边角关系；再综合余弦定理。

---

## C.76 [提升] Part 5/08

**题目回顾**：观测站测船在北偏东 $60°$ 方向，距 $10$ 海里。船以 $10$ 海里/小时向正南行驶 $0.5$ 小时，求新距离。

**思路**　建坐标系或用余弦定理。→ toolkit/04 数形。

**解答**　以观测站为原点 $O$。北偏东 $60°$ 方向，与正北方向夹角 $60°$（向东偏）。

初始船位 $A$：以正北为 $y$ 轴正方向，方位坐标：$A = (10\sin 60°, 10\cos 60°) = (5\sqrt{3}, 5)$。

船向正南行驶 $0.5 \cdot 10 = 5$ 海里，向 $-y$ 方向移动 $5$。

新位置 $B = (5\sqrt{3}, 5 - 5) = (5\sqrt{3}, 0)$。

距离 $|OB| = \sqrt{(5\sqrt{3})^2 + 0^2} = 5\sqrt{3}$ 海里。

**验证**：用余弦定理。$\triangle OAB$ 中 $OA = 10, AB = 5$，$\angle OAB$ 为 $A$ 处由观测站方向（北偏东 $60°$ 的反向 = 南偏西 $60°$）与船航向（正南）的夹角。$A$ 到 $O$ 方向为南偏西 $60°$，船航向正南，二者夹角为 $60°$（西偏 $60°$）。

$OB^2 = OA^2 + AB^2 - 2 \cdot OA \cdot AB \cdot \cos\angle OAB = 100 + 25 - 2 \cdot 10 \cdot 5 \cdot \cos 60° = 125 - 50 = 75 \Rightarrow OB = 5\sqrt{3}$ ✓。

新距离 $5\sqrt{3}$ 海里。

**总结**　方向问题"北偏东 $\theta$" 等于以正北顺时针 $\theta$；解可用坐标系或余弦定理双管齐下交叉验证。

---

## 综合（跨 Part）

---

## C.77 [中档] Part 3+5

**题目回顾**：$f(x) = \sin x + \cos x$。（1）化标准形式；（2）奇偶性；（3）单调递增区间。

**思路**　辅助角 → 用 $\sin(x + \pi/4)$ 模板。

**解答**　(1) $f(x) = \sqrt{2}\sin(x + \pi/4)$（$R = \sqrt{2}$，$\varphi = \pi/4$）。

(2) 定义域 $\mathbb{R}$ 对称。

$f(-x) = \sin(-x) + \cos(-x) = -\sin x + \cos x \neq f(x)$ 且 $\neq -f(x)$。

故 $f$ 既非奇也非偶（非奇非偶函数）。

(3) $\sqrt{2}\sin(x + \pi/4)$ 递增区间：$x + \pi/4 \in [-\pi/2 + 2k\pi, \pi/2 + 2k\pi]$，解出 $x \in [-3\pi/4 + 2k\pi, \pi/4 + 2k\pi]$（$k \in \mathbb{Z}$）。

**总结**　$\sin x + \cos x$ 经典化为 $\sqrt{2}\sin(x + \pi/4)$；周期 $2\pi$ 与单调区间套用模板。

---

## C.78 [中档] Part 2+4

**题目回顾**：$a, b > 0$，$\log_2 a + \log_2 b = 3$，求 $a + b$ 的最小值。

**思路**　对数转积 $ab = 8$ → 基本不等式。

**解答**　$\log_2 a + \log_2 b = \log_2 (ab) = 3 \Rightarrow ab = 8$。

由基本不等式 $a + b \geq 2\sqrt{ab} = 2\sqrt{8} = 4\sqrt{2}$。

取等 $a = b = \sqrt{8} = 2\sqrt{2}$ ✓。

最小值 $4\sqrt{2}$。

**总结**　对数和 $\Rightarrow$ 真数积；积定 $\Rightarrow$ 和有下界（基本不等式）；这一组合是高考热点。

---

## C.79 [提升] Part 3+4

**题目回顾**：$f(x) = \log_2(4^x + 1) - x$。（1）奇偶性；（2）证明 $[0, +\infty)$ 单调递增；（3）最小值。

**思路**　化简 $f(x)$ 形式；用定义法。

**解答**　(1) $f(x) = \log_2(4^x + 1) - x = \log_2(4^x + 1) - \log_2 2^x = \log_2 \dfrac{4^x + 1}{2^x} = \log_2 (2^x + 2^{-x})$。

$f(-x) = \log_2(2^{-x} + 2^x) = f(x)$，故 $f$ 为偶函数。

(2) 任取 $0 \leq x_1 < x_2$。

$f(x_2) - f(x_1) = \log_2(2^{x_2} + 2^{-x_2}) - \log_2(2^{x_1} + 2^{-x_1}) = \log_2 \dfrac{2^{x_2} + 2^{-x_2}}{2^{x_1} + 2^{-x_1}}$。

由于 $g(t) = t + 1/t$（$t \geq 1$）单调递增，且 $2^{x_2} \geq 2^{x_1} \geq 1$，故 $2^{x_2} + 2^{-x_2} > 2^{x_1} + 2^{-x_1}$。

$\Rightarrow \dfrac{2^{x_2} + 2^{-x_2}}{2^{x_1} + 2^{-x_1}} > 1 \Rightarrow \log_2(\cdot) > 0 \Rightarrow f(x_2) > f(x_1)$。

故 $f$ 在 $[0, +\infty)$ 单调递增。

(3) $f$ 偶函数且 $[0, +\infty)$ 递增，故 $x = 0$ 时 $f$ 取最小值。

$f(0) = \log_2(2^0 + 2^0) = \log_2 2 = 1$。

最小值为 $1$。

**总结**　"指对混合函数" 第一步永远尝试化简 $\log_2(4^x + 1) - x = \log_2(2^x + 2^{-x})$ 这类美形式，再用 $t + 1/t$ 模型。

---

## C.80 [提升] Part 1+3

**题目回顾**：$A = \{x \mid x^2 - 4x + 3 \leq 0\}$；$B = \{x \mid \log_{0.5}(x - 1) \geq 0\}$。求 $A \cap B$。

**思路**　$A$ 解二次不等式；$B$ 解对数不等式（底 $< 1$ 注意方向）；交集。

**解答**　$A$：$x^2 - 4x + 3 = (x - 1)(x - 3) \leq 0 \Rightarrow x \in [1, 3]$，$A = [1, 3]$。

$B$：$\log_{0.5}(x - 1) \geq 0 = \log_{0.5} 1$。底 $0.5 \in (0, 1)$，$\log_{0.5}$ 递减，故 $x - 1 \leq 1 \Rightarrow x \leq 2$；同时真数 $x - 1 > 0 \Rightarrow x > 1$。$B = (1, 2]$。

$A \cap B = [1, 3] \cap (1, 2] = (1, 2]$。

**总结**　对数不等式底 $< 1$ 时"翻向"+ 真数 $> 0$ 必查；集合交集端点开闭仔细。

---

> **题号索引（共 80 题，完整解析）**
>
> | Part | 题号范围 | 题数 |
> |------|---------|------|
> | Part 1 集合逻辑 | C.01–C.10 | 10 |
> | Part 2 不等式 | C.11–C.22 | 12 |
> | Part 3 函数概念 | C.23–C.40 | 18 |
> | Part 4 指对幂 | C.41–C.50 | 10 |
> | Part 5 三角函数 | C.51–C.76 | 26 |
> | 综合 | C.77–C.80 | 4 |
> | **合计** | **C.01–C.80** | **80** |
