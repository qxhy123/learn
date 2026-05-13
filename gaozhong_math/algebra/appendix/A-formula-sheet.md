# 附录 A：高一上代数公式表

> 覆盖 Part 1–Part 5（集合与逻辑、不等式、函数概念、指对幂、三角函数）共 42 章核心公式与定理。每条配适用条件与简短说明，方便考前 1–2 小时通览。

---

## Part 1 集合与逻辑

### 1.1 集合三要素与三性 `(part1/01)`

- **三要素**：元素、集合、属于关系（$\in$ / $\notin$）。
- **集合三性**：
  - **确定性**：元素是否在集合中必须确定，不能含糊。
  - **互异性**：集合内元素两两不同（重复算一个）。
  - **无序性**：$\{1, 2, 3\} = \{3, 2, 1\}$，排列顺序不影响集合。
- **常用数集符号**：$\mathbb{N}$（自然数）、$\mathbb{N}^*$ 或 $\mathbb{N}_+$（正整数）、$\mathbb{Z}$（整数）、$\mathbb{Q}$（有理数）、$\mathbb{R}$（实数）。
- **空集**：$\varnothing$，不含任何元素；空集是任意集合的子集，是非空集合的真子集。

### 1.2 子集 / 真子集 / 幂集 `(part1/03)`

设集合 $A$ 共有 $n$ 个元素，则：

| 概念 | 定义 | 个数 |
|------|------|------|
| **子集** | 所有 $A$ 的子集（含 $\varnothing$ 与 $A$ 自身） | $2^n$ |
| **真子集** | 子集中除去 $A$ 自身 | $2^n - 1$ |
| **非空子集** | 子集中除去 $\varnothing$ | $2^n - 1$ |
| **非空真子集** | 子集中除去 $\varnothing$ 与 $A$ | $2^n - 2$ |
| **幂集** $\mathcal{P}(A)$ | 所有子集组成的集合 | $|\mathcal{P}(A)| = 2^n$ |

**记忆**：$n = 3$ 时，$2^3 = 8$ 个子集；$2^3 - 1 = 7$ 个真子集。

### 1.3 集合三大运算 `(part1/02)`

设全集 $U$，$A, B \subseteq U$：

| 运算 | 定义 | 关键 |
|------|------|------|
| **交集** $A \cap B$ | $\{x \mid x \in A \text{ 且 } x \in B\}$ | "同时属于" |
| **并集** $A \cup B$ | $\{x \mid x \in A \text{ 或 } x \in B\}$ | "至少属于一个" |
| **补集** $\complement_U A$ | $\{x \mid x \in U \text{ 且 } x \notin A\}$ | "在 $U$ 中扣掉 $A$" |

**性质**：
- $A \cap A = A$，$A \cup A = A$，$A \cap \varnothing = \varnothing$，$A \cup \varnothing = A$；
- $A \cap U = A$，$A \cup U = U$，$A \cap \complement_U A = \varnothing$，$A \cup \complement_U A = U$；
- $A \subseteq B \Leftrightarrow A \cap B = A \Leftrightarrow A \cup B = B$。

### 1.4 德摩根律（对偶律）`(part1/02)`

$$\boxed{\complement_U(A \cup B) = (\complement_U A) \cap (\complement_U B)}$$

$$\boxed{\complement_U(A \cap B) = (\complement_U A) \cup (\complement_U B)}$$

**口诀**：补集翻并交、并交翻——"补与并/交可以互换，并/交一翻就反"。

### 1.5 全称 / 存在量词及其否定 `(part1/04)`

| 量词 | 符号 | 否定 |
|------|------|------|
| 全称量词"任意" | $\forall$ | $\exists$（变存在） |
| 存在量词"存在" | $\exists$ | $\forall$（变全称） |

**否定规则**：$\neg(\forall x, p(x)) \Leftrightarrow \exists x, \neg p(x)$；$\neg(\exists x, p(x)) \Leftrightarrow \forall x, \neg p(x)$。

**口诀**：**量词换、结论否、范围不动**。

### 1.6 充分条件与必要条件 `(part1/05)`

设命题 $p, q$，若 $p \Rightarrow q$ 成立：

| 关系 | 名称 | 等价说法 |
|------|------|---------|
| $p \Rightarrow q$ 真，$q \Rightarrow p$ 假 | $p$ 是 $q$ 的**充分不必要**条件 | $p$ 真则 $q$ 真；$q$ 真未必 $p$ 真 |
| $p \Rightarrow q$ 假，$q \Rightarrow p$ 真 | $p$ 是 $q$ 的**必要不充分**条件 | $p$ 真未必 $q$ 真；$q$ 真则 $p$ 真 |
| $p \Leftrightarrow q$（双向均真） | $p$ 是 $q$ 的**充要**条件 | 两者等价 |
| 两向均假 | **既不充分也不必要** | — |

**集合视角**：设 $p$ 对应集合 $A$，$q$ 对应集合 $B$：
- $A \subsetneq B \Leftrightarrow p$ 是 $q$ 的充分不必要条件；
- $A \supsetneq B \Leftrightarrow p$ 是 $q$ 的必要不充分条件；
- $A = B \Leftrightarrow$ 充要条件。

> **易错提醒（Part 1）**
> 1. 写集合时检查互异性：$\{1, a, a^2\}$ 要保证三者两两不等。
> 2. 空集是任意集合的子集，所以遇到"$A \subseteq B$"含参讨论时，必须考虑 $A = \varnothing$ 的特殊情形。
> 3. 量词否定时不要忘了把"$>$"否定为"$\le$"（"$\ge$"变"$<$"）。

---

## Part 2 不等式

### 2.1 不等式基本性质（7 条）`(part2/01)`

1. **对称性**：$a > b \Leftrightarrow b < a$。
2. **传递性**：$a > b, b > c \Rightarrow a > c$。
3. **加法保号**：$a > b \Rightarrow a + c > b + c$。
4. **同向相加**：$a > b, c > d \Rightarrow a + c > b + d$。
5. **乘正不变号**：$a > b, c > 0 \Rightarrow ac > bc$。
6. **乘负变号**：$a > b, c < 0 \Rightarrow ac < bc$。
7. **乘方与开方**：$a > b > 0 \Rightarrow a^n > b^n$，$\sqrt[n]{a} > \sqrt[n]{b}$（$n \in \mathbb{N}^*$）。

**两条衍生**：
- 同号倒数翻转：$a > b > 0 \Rightarrow \dfrac{1}{a} < \dfrac{1}{b}$。
- 同向异向相减：$a > b, c < d \Rightarrow a - c > b - d$（注意"异向才能相减"）。

### 2.2 一元二次不等式解集（按 $\Delta$ 与 $a$ 分类）`(part2/02)`

**前提**：先把不等式化为 $a > 0$ 的标准形式。若 $a < 0$，两边乘 $-1$ 并翻转不等号。

设 $\Delta = b^2 - 4ac$，$a > 0$：

| $\Delta$ | 根的情况 | $ax^2+bx+c > 0$ 解集 | $ax^2+bx+c < 0$ 解集 |
|---|---|---|---|
| $\Delta > 0$ | 两根 $x_1 < x_2$ | $\{x \mid x < x_1 \text{ 或 } x > x_2\}$ | $\{x \mid x_1 < x < x_2\}$ |
| $\Delta = 0$ | 重根 $x_0$ | $\{x \mid x \ne x_0\}$ | $\varnothing$ |
| $\Delta < 0$ | 无实根 | $\mathbb{R}$ | $\varnothing$ |

**图象记忆**：开口向上的抛物线"大于零取两边、小于零取中间"。

**恒成立条件**：
- $ax^2 + bx + c > 0$ 在 $\mathbb{R}$ 上恒成立 $\Leftrightarrow a > 0$ 且 $\Delta < 0$。
- $ax^2 + bx + c < 0$ 在 $\mathbb{R}$ 上恒成立 $\Leftrightarrow a < 0$ 且 $\Delta < 0$。

### 2.3 分式不等式等价转化 `(part2/03)`

$$\frac{f(x)}{g(x)} > 0 \Leftrightarrow f(x) \cdot g(x) > 0 \quad (g(x) \ne 0)$$

$$\frac{f(x)}{g(x)} \ge 0 \Leftrightarrow \begin{cases} f(x) \cdot g(x) \ge 0 \\ g(x) \ne 0 \end{cases} \Leftrightarrow \begin{cases} f(x) \cdot g(x) > 0 \\ \text{或 } f(x) = 0 \text{ 且 } g(x) \ne 0 \end{cases}$$

**穿针引线法**：高次分式不等式标根后，在数轴上从右上方"穿"过最右根，遵循"**奇穿偶不穿**"（重数为奇穿过去、为偶反弹回来）。

### 2.4 基本不等式 `(part2/04)`

**两元基本不等式**（$a, b > 0$）：

$$\boxed{\frac{a + b}{2} \ge \sqrt{ab}, \quad \text{即 } a + b \ge 2\sqrt{ab}}$$

等号成立 $\Leftrightarrow a = b$。

**变形版本**（已知与未知互换）：

$$a^2 + b^2 \ge 2ab \quad (\text{对任意实数都成立，等号当且仅当 } a = b)$$

$$\frac{a + b}{2} \ge \sqrt{ab} \ge \frac{2ab}{a + b} \quad (a, b > 0) \quad (\text{算术} \ge \text{几何} \ge \text{调和})$$

**三元基本不等式**（$a, b, c > 0$）：

$$\frac{a + b + c}{3} \ge \sqrt[3]{abc}, \quad a^3 + b^3 + c^3 \ge 3abc$$

等号成立 $\Leftrightarrow a = b = c$。

### 2.5 "和定积大、积定和小"`(part2/04)`

| 已知 | 求 | 结论 | 等号条件 |
|---|---|---|---|
| $a + b = S$（定值） | $ab$ 最大值 | $ab \le \dfrac{S^2}{4}$ | $a = b = \dfrac{S}{2}$ |
| $ab = P$（定值，且 $a, b > 0$） | $a + b$ 最小值 | $a + b \ge 2\sqrt{P}$ | $a = b = \sqrt{P}$ |

**口诀**："和定积大、积定和小"——即给定和，积有上界；给定积，和有下界。

> **易错提醒（Part 2）**
> 1. 用基本不等式必查 **"一正二定三相等"**：正数、和或积为定值、能取到等号。
> 2. 一元二次不等式若题中 $a$ 不明符号（如含参 $kx^2 + \cdots$），必须先讨论 $a = 0$ 与 $a \ne 0$。
> 3. 分式不等式不能直接两边乘分母——分母正负未知，必须先转化为乘积不等式。

---

## Part 3 函数概念

### 3.1 函数三要素 `(part3/01)`

$$y = f(x): \quad x \in A \to y \in B$$

- **定义域** $A$：自变量 $x$ 允许取值范围。
- **值域** $\{f(x) \mid x \in A\}$：函数值的集合。
- **对应关系** $f$：从 $x$ 到 $y$ 的映射规则（每个 $x$ 对应唯一 $y$）。

**两函数相同**：定义域相同 + 对应关系相同（值域随之相同）。

### 3.2 常见定义域约束 `(part3/02)`

| 表达式 | 定义域条件 |
|---|---|
| $\sqrt{f(x)}$（偶次根式） | $f(x) \ge 0$ |
| $\dfrac{1}{f(x)}$ | $f(x) \ne 0$ |
| $\log_a f(x)$ | $f(x) > 0$（同时 $a > 0, a \ne 1$） |
| $\tan f(x)$ | $f(x) \ne \dfrac{\pi}{2} + k\pi$ |
| $[f(x)]^0$ | $f(x) \ne 0$ |
| $f^{1/n}(x)$（分数指数，$n$ 偶） | $f(x) \ge 0$ |

**多重约束**：把所有条件列出，取交集即为定义域。

### 3.3 单调性定义 `(part3/03)`

设 $f(x)$ 在区间 $I$ 上有定义，对任意 $x_1, x_2 \in I$ 且 $x_1 < x_2$：

| 性质 | 定义 | 图象 |
|---|---|---|
| **增函数** | $f(x_1) < f(x_2)$ | 从左下到右上 |
| **减函数** | $f(x_1) > f(x_2)$ | 从左上到右下 |

**判别工具**：作差法 $f(x_1) - f(x_2)$；作商法 $\dfrac{f(x_1)}{f(x_2)}$（要求同号）；导数法（后续学）。

**复合函数单调性**："**同增异减**" —— 内外函数单调性相同则复合为增，不同则为减。

### 3.4 奇偶性定义与图象特征 `(part3/04)`

**前提**：定义域关于原点对称（必要条件）。

| 性质 | 定义 | 图象对称 |
|---|---|---|
| **偶函数** | $f(-x) = f(x)$ | 关于 **$y$ 轴**对称 |
| **奇函数** | $f(-x) = -f(x)$ | 关于 **原点**对称 |
| 非奇非偶 | 两者均不成立 | 无特殊对称 |
| 既奇又偶 | $f(x) \equiv 0$（定义域对称） | — |

**性质组合**（"奇 / 偶 ± × ÷ 奇 / 偶"）：
- 奇 + 奇 = 奇，偶 + 偶 = 偶，奇 + 偶 = 非奇非偶（一般）；
- 奇 × 奇 = 偶，偶 × 偶 = 偶，奇 × 偶 = 奇。

**奇函数关键性质**：若 $0$ 在定义域内，则 $f(0) = 0$。

### 3.5 抽象函数常用恒等式 `(part3/01, part3/07)`

| 函数族 | 特征方程 |
|---|---|
| 一次型 $f(x) = kx$ | $f(x + y) = f(x) + f(y)$ |
| 指数型 $f(x) = a^x$ | $f(x + y) = f(x) \cdot f(y)$ |
| 对数型 $f(x) = \log_a x$ | $f(xy) = f(x) + f(y)$ |
| 幂函数型 $f(x) = x^\alpha$ | $f(xy) = f(x) \cdot f(y)$ |

**使用方法**：见恒等式 → 识别函数族 → 用赋值法（取 $x = y = 0$、$x = 0, y = 1$ 等）求关键值。

### 3.6 函数图象的 6 类变换 `(part3/05)`

设 $y = f(x)$ 已知，$a, b > 0$：

| 变换 | 新函数 | 图象操作 |
|---|---|---|
| 平移 | $y = f(x - a)$ | 右移 $a$ 个单位 |
| 平移 | $y = f(x + a)$ | 左移 $a$ 个单位 |
| 平移 | $y = f(x) + b$ | 上移 $b$ 个单位 |
| 平移 | $y = f(x) - b$ | 下移 $b$ 个单位 |
| 翻折 | $y = -f(x)$ | 关于 $x$ 轴翻折 |
| 翻折 | $y = f(-x)$ | 关于 $y$ 轴翻折 |
| 取绝对值 | $y = |f(x)|$ | 把 $x$ 轴下方部分翻折到上方 |
| 取绝对值 | $y = f(|x|)$ | 保留 $y$ 轴右侧，把它复制翻折到左侧 |
| 伸缩（横） | $y = f(\omega x)$（$\omega > 0$） | 横坐标缩为 $1/\omega$ 倍 |
| 伸缩（纵） | $y = A f(x)$（$A > 0$） | 纵坐标拉为 $A$ 倍 |

**口诀**：**左加右减、上加下减**（对 $x$ 操作"反着来"，对 $y$ 操作"顺着来"）。

### 3.7 对称性公式 `(part3/04)`

| 条件 | 几何结论 |
|---|---|
| $f(a + x) = f(a - x)$ | 图象关于直线 $x = a$ **轴对称** |
| $f(a + x) = -f(a - x)$ | 图象关于点 $(a, 0)$ **中心对称** |
| $f(a + x) = f(b - x)$ | 对称轴为 $x = \dfrac{a + b}{2}$ |
| $f(x + T) = f(x)$ | 周期函数，$T$ 是一个周期 |

**特殊**：$f(x) = f(-x) \Leftrightarrow$ 偶函数 $\Leftrightarrow$ 关于 $y$ 轴对称（即 $a = 0$）。

> **易错提醒（Part 3）**
> 1. 先判断**定义域是否关于原点对称**，再讨论奇偶性，否则直接是非奇非偶。
> 2. 复合函数单调性中的"同增异减"是指内 / 外函数单调性的乘积；当内函数减、外函数减时，复合是增。
> 3. 图象变换 $y = f(x - a)$ 是右移，不是左移；$x$ 前面的运算"反着读"。

---

## Part 4 指数 / 对数 / 幂函数

### 4.1 指数运算五法则 `(part4/01)`

设 $a > 0, a \ne 1$，$m, n \in \mathbb{R}$：

$$\boxed{a^m \cdot a^n = a^{m+n}}$$

$$\boxed{\frac{a^m}{a^n} = a^{m-n}}$$

$$\boxed{(a^m)^n = a^{mn}}$$

$$\boxed{(ab)^n = a^n b^n \quad (a, b > 0)}$$

$$\boxed{a^{-n} = \frac{1}{a^n}, \quad a^0 = 1, \quad a^{m/n} = \sqrt[n]{a^m} \quad (a > 0)}$$

**特殊值**：$a^1 = a$，$1^x = 1$（任意实数 $x$）。

### 4.2 对数运算三法则与换底公式 `(part4/02)`

设 $a > 0, a \ne 1$，$M, N > 0$：

$$\boxed{\log_a(MN) = \log_a M + \log_a N \quad \text{（积变和）}}$$

$$\boxed{\log_a \frac{M}{N} = \log_a M - \log_a N \quad \text{（商变差）}}$$

$$\boxed{\log_a M^n = n \log_a M \quad \text{（幂变积）}}$$

**换底公式**：

$$\log_a b = \frac{\log_c b}{\log_c a} \quad (a, b, c > 0; a, c \ne 1)$$

**衍生**：$\log_a b \cdot \log_b a = 1$；$\log_{a^m} b^n = \dfrac{n}{m} \log_a b$。

**特殊值**：$\log_a 1 = 0$，$\log_a a = 1$，$\log_a a^n = n$，$a^{\log_a M} = M$。

### 4.3 指数函数 / 对数函数性质双对照表 `(part4/01-02)`

设 $a > 0, a \ne 1$，记 $f(x) = a^x$，$g(x) = \log_a x$。

| 项目 | $a > 1$ | $0 < a < 1$ |
|---|---|---|
| **$y = a^x$ 定义域** | $\mathbb{R}$ | $\mathbb{R}$ |
| **$y = a^x$ 值域** | $(0, +\infty)$ | $(0, +\infty)$ |
| **$y = a^x$ 单调性** | $\mathbb{R}$ 上单调递增 | $\mathbb{R}$ 上单调递减 |
| **$y = a^x$ 经过的定点** | $(0, 1)$ | $(0, 1)$ |
| **$y = a^x$ 渐近线** | $y = 0$（$x \to -\infty$） | $y = 0$（$x \to +\infty$） |
| **$y = \log_a x$ 定义域** | $(0, +\infty)$ | $(0, +\infty)$ |
| **$y = \log_a x$ 值域** | $\mathbb{R}$ | $\mathbb{R}$ |
| **$y = \log_a x$ 单调性** | 单调递增 | 单调递减 |
| **$y = \log_a x$ 经过的定点** | $(1, 0)$ | $(1, 0)$ |
| **$y = \log_a x$ 渐近线** | $x = 0$ | $x = 0$ |

**互为反函数**：$y = a^x$ 与 $y = \log_a x$ 关于直线 $y = x$ 对称。

### 4.4 五个常见幂函数 `(part4/03)`

幂函数：$y = x^\alpha$（$\alpha$ 为常数）。高考重点掌握五个：

| 函数 | 定义域 | 奇偶性 | 单调性（$x > 0$） | 图象特征 |
|---|---|---|---|---|
| $y = x$ | $\mathbb{R}$ | 奇 | 增 | 一三象限直线 |
| $y = x^2$ | $\mathbb{R}$ | 偶 | 增 | 开口向上抛物线 |
| $y = x^3$ | $\mathbb{R}$ | 奇 | 增 | 一三象限拐点过原点 |
| $y = x^{1/2} = \sqrt{x}$ | $[0, +\infty)$ | 非奇非偶 | 增 | 一象限抛物线右半 |
| $y = x^{-1} = \dfrac{1}{x}$ | $(-\infty, 0) \cup (0, +\infty)$ | 奇 | 减（在每个分支上） | 一三象限双曲线 |

**一般规律**（$x > 0$ 时）：
- $\alpha > 0$：单调递增，过 $(0, 0)$ 和 $(1, 1)$；
- $\alpha < 0$：单调递减，不过原点，但过 $(1, 1)$；
- $\alpha = 0$：$y = 1$（常函数，去除 $x = 0$）。

> **易错提醒（Part 4）**
> 1. 指数 / 对数函数题型几乎都需要分类讨论 $a > 1$ 还是 $0 < a < 1$（单调性方向相反）。
> 2. 对数函数定义域是 $x > 0$；含参对数 $\log_a f(x)$ 还要保证真数 $f(x) > 0$。
> 3. 换底公式选择常用底数 $e, 10$ 或目标式中已出现的底，避免引入新底。

---

## Part 5 三角函数

### 5.1 角度与弧度互化 `(part5/01)`

**核心关系**：

$$\boxed{180° = \pi \text{ rad}}$$

由此推出：

$$1° = \frac{\pi}{180} \text{ rad}, \quad 1 \text{ rad} = \frac{180°}{\pi} \approx 57.3°$$

**常用角换算**：

| 角度 | $30°$ | $45°$ | $60°$ | $90°$ | $120°$ | $135°$ | $150°$ | $180°$ | $270°$ | $360°$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 弧度 | $\frac{\pi}{6}$ | $\frac{\pi}{4}$ | $\frac{\pi}{3}$ | $\frac{\pi}{2}$ | $\frac{2\pi}{3}$ | $\frac{3\pi}{4}$ | $\frac{5\pi}{6}$ | $\pi$ | $\frac{3\pi}{2}$ | $2\pi$ |

### 5.2 弧长与扇形面积 `(part5/01)`

设圆半径 $r$，圆心角 $\theta$（**弧度制**），弧长 $l$，扇形面积 $S$：

$$\boxed{l = r\theta}$$

$$\boxed{S = \frac{1}{2}r^2\theta = \frac{1}{2}lr}$$

> 切记圆心角必须用弧度制（不能用度数）！

### 5.3 任意角三角函数定义与象限符号 `(part5/02)`

设角 $\alpha$ 终边与单位圆交于点 $P(x, y)$，则：

$$\sin\alpha = y, \quad \cos\alpha = x, \quad \tan\alpha = \frac{y}{x} \quad (x \ne 0)$$

**象限符号**（"**ASTC**" 或口诀 **"一全二正弦三切四余弦"**——指在该象限为正的函数）：

| 象限 | $\sin$ | $\cos$ | $\tan$ |
|---|---|---|---|
| I（一） | $+$ | $+$ | $+$ |
| II（二） | $+$ | $-$ | $-$ |
| III（三） | $-$ | $-$ | $+$ |
| IV（四） | $-$ | $+$ | $-$ |

### 5.4 特殊角三角函数值表 `(part5/02)`

| $\alpha$ | $0$ | $\dfrac{\pi}{6}$（30°） | $\dfrac{\pi}{4}$（45°） | $\dfrac{\pi}{3}$（60°） | $\dfrac{\pi}{2}$（90°） | $\dfrac{2\pi}{3}$（120°） | $\dfrac{3\pi}{4}$（135°） | $\dfrac{5\pi}{6}$（150°） | $\pi$（180°） | $\dfrac{3\pi}{2}$（270°） |
|---|---|---|---|---|---|---|---|---|---|---|
| $\sin\alpha$ | $0$ | $\dfrac{1}{2}$ | $\dfrac{\sqrt{2}}{2}$ | $\dfrac{\sqrt{3}}{2}$ | $1$ | $\dfrac{\sqrt{3}}{2}$ | $\dfrac{\sqrt{2}}{2}$ | $\dfrac{1}{2}$ | $0$ | $-1$ |
| $\cos\alpha$ | $1$ | $\dfrac{\sqrt{3}}{2}$ | $\dfrac{\sqrt{2}}{2}$ | $\dfrac{1}{2}$ | $0$ | $-\dfrac{1}{2}$ | $-\dfrac{\sqrt{2}}{2}$ | $-\dfrac{\sqrt{3}}{2}$ | $-1$ | $0$ |
| $\tan\alpha$ | $0$ | $\dfrac{\sqrt{3}}{3}$ | $1$ | $\sqrt{3}$ | 不存在 | $-\sqrt{3}$ | $-1$ | $-\dfrac{\sqrt{3}}{3}$ | $0$ | 不存在 |

### 5.5 同角三角恒等式 `(part5/04)`

$$\boxed{\sin^2\alpha + \cos^2\alpha = 1}$$

$$\boxed{\tan\alpha = \frac{\sin\alpha}{\cos\alpha} \quad (\cos\alpha \ne 0)}$$

$$\boxed{1 + \tan^2\alpha = \sec^2\alpha = \frac{1}{\cos^2\alpha}}$$

**常用变形**：

$$\sin^2\alpha = 1 - \cos^2\alpha, \quad \cos^2\alpha = 1 - \sin^2\alpha$$

$$(\sin\alpha \pm \cos\alpha)^2 = 1 \pm 2\sin\alpha\cos\alpha$$

由 $(\sin\alpha + \cos\alpha)^2 + (\sin\alpha - \cos\alpha)^2 = 2$ 可联立求 $\sin\alpha \pm \cos\alpha$。

### 5.6 诱导公式六组 `(part5/04)`

**口诀**：**奇变偶不变，符号看象限**。"奇 / 偶"指 $\dfrac{\pi}{2}$ 倍数中的奇偶；"变"指 $\sin \leftrightarrow \cos$ 互换；"符号"指把 $\alpha$ 当锐角时该象限原函数的符号。

| 角 | $\sin$ | $\cos$ | $\tan$ |
|---|---|---|---|
| $-\alpha$ | $-\sin\alpha$ | $\cos\alpha$ | $-\tan\alpha$ |
| $\pi - \alpha$ | $\sin\alpha$ | $-\cos\alpha$ | $-\tan\alpha$ |
| $\pi + \alpha$ | $-\sin\alpha$ | $-\cos\alpha$ | $\tan\alpha$ |
| $2\pi - \alpha$（或 $-\alpha + 2k\pi$） | $-\sin\alpha$ | $\cos\alpha$ | $-\tan\alpha$ |
| $\dfrac{\pi}{2} - \alpha$ | $\cos\alpha$ | $\sin\alpha$ | $\cot\alpha$ |
| $\dfrac{\pi}{2} + \alpha$ | $\cos\alpha$ | $-\sin\alpha$ | $-\cot\alpha$ |
| $\dfrac{3\pi}{2} - \alpha$ | $-\cos\alpha$ | $-\sin\alpha$ | $\cot\alpha$ |
| $\dfrac{3\pi}{2} + \alpha$ | $-\cos\alpha$ | $\sin\alpha$ | $-\cot\alpha$ |

**用法**：把任意角化为 $[0, 2\pi)$ 或 $[0, \dfrac{\pi}{2}]$ 范围内的角再查表。

### 5.7 和差角公式 `(part5/05)`

$$\boxed{\sin(\alpha \pm \beta) = \sin\alpha\cos\beta \pm \cos\alpha\sin\beta}$$

$$\boxed{\cos(\alpha \pm \beta) = \cos\alpha\cos\beta \mp \sin\alpha\sin\beta}$$

$$\boxed{\tan(\alpha \pm \beta) = \frac{\tan\alpha \pm \tan\beta}{1 \mp \tan\alpha\tan\beta}}$$

**记忆**：$\sin$ 同号（$+$ / $+$），$\cos$ 异号（$-$ / $+$），$\tan$ 分子同号，分母异号。

### 5.8 二倍角公式 `(part5/05)`

$$\boxed{\sin 2\alpha = 2\sin\alpha\cos\alpha}$$

$$\boxed{\cos 2\alpha = \cos^2\alpha - \sin^2\alpha = 2\cos^2\alpha - 1 = 1 - 2\sin^2\alpha}$$

$$\boxed{\tan 2\alpha = \frac{2\tan\alpha}{1 - \tan^2\alpha}}$$

**降幂公式**（由 $\cos 2\alpha$ 反推，**积化和差时常用**）：

$$\sin^2\alpha = \frac{1 - \cos 2\alpha}{2}, \quad \cos^2\alpha = \frac{1 + \cos 2\alpha}{2}$$

### 5.9 半角公式（选学）`(part5/05)`

$$\sin^2\frac{\alpha}{2} = \frac{1 - \cos\alpha}{2}, \quad \cos^2\frac{\alpha}{2} = \frac{1 + \cos\alpha}{2}$$

$$\tan\frac{\alpha}{2} = \frac{\sin\alpha}{1 + \cos\alpha} = \frac{1 - \cos\alpha}{\sin\alpha}$$

### 5.10 辅助角公式 `(part5/05)`

$$\boxed{a\sin x + b\cos x = \sqrt{a^2 + b^2}\sin(x + \varphi)}$$

其中 $\cos\varphi = \dfrac{a}{\sqrt{a^2 + b^2}}$，$\sin\varphi = \dfrac{b}{\sqrt{a^2 + b^2}}$，即 $\tan\varphi = \dfrac{b}{a}$（$\varphi$ 在 $a > 0$ 时取主值）。

**意义**：把同周期的正余弦混合式收缩为单一正弦式，最值变为 $\pm\sqrt{a^2 + b^2}$。

### 5.11 三角函数 $y = A\sin(\omega x + \varphi) + b$ `(part5/03)`

| 参数 | 几何意义 | 影响 |
|---|---|---|
| $A$（$A > 0$） | 振幅 | 纵向拉伸 $A$ 倍；最大值 $A + b$、最小值 $-A + b$ |
| $\omega$（$\omega > 0$） | 角频率 | 周期 $T = \dfrac{2\pi}{\omega}$ |
| $\varphi$ | 初相 | 水平平移：左移 $\dfrac{\varphi}{\omega}$（$\varphi > 0$）|
| $b$ | 垂直位移 | 整体上下平移 |

**频率与周期**：

$$T = \frac{2\pi}{\omega}, \quad f = \frac{1}{T} = \frac{\omega}{2\pi}$$

**对称轴 / 对称中心**：
- 对称轴 $x_k$ 满足 $\omega x_k + \varphi = \dfrac{\pi}{2} + k\pi$；
- 对称中心 $(x_k, b)$ 满足 $\omega x_k + \varphi = k\pi$。

### 5.12 三角函数三大基本图象与性质 `(part5/03)`

| 函数 | 定义域 | 值域 | 周期 | 奇偶性 | 单调增区间 |
|---|---|---|---|---|---|
| $y = \sin x$ | $\mathbb{R}$ | $[-1, 1]$ | $2\pi$ | 奇 | $\left[-\dfrac{\pi}{2} + 2k\pi, \dfrac{\pi}{2} + 2k\pi\right]$ |
| $y = \cos x$ | $\mathbb{R}$ | $[-1, 1]$ | $2\pi$ | 偶 | $[-\pi + 2k\pi, 2k\pi]$ |
| $y = \tan x$ | $\{x \mid x \ne \dfrac{\pi}{2} + k\pi\}$ | $\mathbb{R}$ | $\pi$ | 奇 | $\left(-\dfrac{\pi}{2} + k\pi, \dfrac{\pi}{2} + k\pi\right)$ |

### 5.13 正弦定理 `(part5/07)`

设 $\triangle ABC$ 三边 $a, b, c$ 对应角 $A, B, C$，外接圆半径 $R$：

$$\boxed{\frac{a}{\sin A} = \frac{b}{\sin B} = \frac{c}{\sin C} = 2R}$$

**等价用法**：
- 已知两角一边（AAS / ASA）→ 求其他边；
- 已知两边及其中一边对角（ASS）→ 求另一边对角（注意可能两解）；
- $a = 2R\sin A$，可用于"边角互化"。

### 5.14 余弦定理 `(part5/07)`

$$\boxed{c^2 = a^2 + b^2 - 2ab\cos C \quad \text{（轮换得 } a^2, b^2 \text{ 形式）}}$$

**逆用**（已知三边求角）：

$$\cos C = \frac{a^2 + b^2 - c^2}{2ab}$$

**适用场景**：
- 已知两边及夹角（SAS）→ 求第三边；
- 已知三边（SSS）→ 求任一角；
- 由 $\cos C$ 符号可判断三角形形状（正 / 零 / 负 ↔ 锐 / 直 / 钝）。

### 5.15 三角形面积公式 `(part5/07)`

$$\boxed{S_{\triangle} = \frac{1}{2}ab\sin C = \frac{1}{2}bc\sin A = \frac{1}{2}ca\sin B}$$

**结合外接圆半径**：

$$S = \frac{abc}{4R}$$

**结合内切圆半径** $r$：$S = \dfrac{1}{2}(a + b + c) \cdot r$。

> **易错提醒（Part 5）**
> 1. 弧度制下弧长公式 $l = r\theta$ 中的 $\theta$ 必须用弧度，不能用度数。
> 2. 诱导公式判断符号时把 $\alpha$ 当 **锐角**，看 $\dfrac{\pi}{2} \pm \alpha$ 等所在象限的原函数符号。
> 3. 已知三角函数值反求角时一定要分象限讨论符号，不能漏掉负值。
> 4. ASS 情形（已知两边及非夹角）的正弦定理可能出现两解、一解或无解，需用大边对大角原则筛选。
> 5. 辅助角公式中 $\varphi$ 的象限由 $a, b$ 的符号共同确定，写错象限会导致最值范围错误。

---

*本表覆盖 Part 1–5 全部 42 章核心公式与定理约 80 条。配合附录 B 模型图集与各章详细推导使用，可在高一上学期复习时快速完成知识点扫描。*
