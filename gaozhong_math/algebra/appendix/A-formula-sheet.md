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
| **幂集** $\mathcal{P}(A)$ | 所有子集组成的集合 | $\vert \mathcal{P}(A)\vert = 2^n$ |

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
| 取绝对值 | $y = \vert f(x)\vert$ | 把 $x$ 轴下方部分翻折到上方 |
| 取绝对值 | $y = f(\vert x\vert )$ | 保留 $y$ 轴右侧，把它复制翻折到左侧 |
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

## Part 6 复数

### 6.1 虚数单位与幂周期 `(part6/01)`

**虚数单位**：$i$ 满足

$$\boxed{i^2 = -1}$$

**$i$ 的幂周期为 4**：

$$i^{4k} = 1, \quad i^{4k+1} = i, \quad i^{4k+2} = -1, \quad i^{4k+3} = -i \quad (k \in \mathbb{Z})$$

**实用记忆**：$i^n$ 用 $n \bmod 4$ 查表；$1 + i + i^2 + i^3 = 0$，因此任意连续四项 $i^n$ 之和为 $0$。

### 6.2 复数的标准形式与几何表示 `(part6/01)`

**代数形式**：$z = a + bi$（$a, b \in \mathbb{R}$）。$a$ 为实部 $\operatorname{Re}(z)$，$b$ 为虚部 $\operatorname{Im}(z)$。

**复数相等**：$a + bi = c + di \Leftrightarrow a = c \text{ 且 } b = d$（"实部对实部、虚部对虚部"）。

**复数为零**：$a + bi = 0 \Leftrightarrow a = 0 \text{ 且 } b = 0$。

**几何对应**：复数 $z = a + bi \leftrightarrow$ 复平面点 $Z(a, b) \leftrightarrow$ 向量 $\overrightarrow{OZ}$。

| 类型 | 条件 | 几何位置 |
|---|---|---|
| 实数 | $b = 0$ | 实轴上 |
| 纯虚数 | $a = 0, b \ne 0$ | 虚轴上（去原点） |
| 虚数 | $b \ne 0$ | 复平面（除实轴） |

### 6.3 共轭复数与模 `(part6/01)`

**共轭复数**：$\overline{z} = a - bi$（实部不变、虚部取反）。几何上是关于实轴对称。

**模**：

$$\boxed{|z| = \sqrt{a^2 + b^2}}$$

即点 $Z(a, b)$ 到原点的距离。

**核心恒等式**：

$$\boxed{z \cdot \overline{z} = |z|^2 = a^2 + b^2}$$

**共轭运算性质**：

$$\overline{z_1 \pm z_2} = \overline{z_1} \pm \overline{z_2}, \quad \overline{z_1 z_2} = \overline{z_1} \cdot \overline{z_2}, \quad \overline{\left(\frac{z_1}{z_2}\right)} = \frac{\overline{z_1}}{\overline{z_2}}$$

$$z = \overline{z} \Leftrightarrow z \in \mathbb{R}, \quad z + \overline{z} = -\overline{z} \cdot 0 \text{ 不一定; } z = -\overline{z} \Leftrightarrow z \text{ 为纯虚数或 } 0$$

### 6.4 复数四则运算 `(part6/02)`

设 $z_1 = a + bi$，$z_2 = c + di$：

$$z_1 \pm z_2 = (a \pm c) + (b \pm d)i$$

$$z_1 \cdot z_2 = (ac - bd) + (ad + bc)i$$

$$\frac{z_1}{z_2} = \frac{z_1 \overline{z_2}}{z_2 \overline{z_2}} = \frac{(a + bi)(c - di)}{c^2 + d^2} = \frac{ac + bd}{c^2 + d^2} + \frac{bc - ad}{c^2 + d^2} i \quad (z_2 \ne 0)$$

**除法核心**：分子分母同乘**分母的共轭**。

### 6.5 模的运算性质 `(part6/02)`

$$\boxed{|z_1 z_2| = |z_1| \cdot |z_2|, \quad \left|\frac{z_1}{z_2}\right| = \frac{|z_1|}{|z_2|}}$$

$$|z^n| = |z|^n, \quad |z| = |\overline{z}|, \quad |z_1 + z_2| \le |z_1| + |z_2| \text{ (三角不等式)}$$

**几何意义**：$|z_1 - z_2|$ 表示复平面上两点 $Z_1, Z_2$ 之间的距离。

> **易错提醒（Part 6）**
> 1. **复数不可比较大小**，不能写 "$z_1 > z_2$"（除非两复数都是实数）。
> 2. 纯虚数定义 $a = 0$ 且 $b \ne 0$，不要漏掉 $b \ne 0$ 这一条件。
> 3. 复数除法不是 "分子除分母" 直接做，必须用共轭分母法转化为实数分母。

---

## Part 7 统计

### 7.1 样本数字特征 `(part7/02)`

设样本数据 $x_1, x_2, \ldots, x_n$：

**平均数**：

$$\boxed{\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i}$$

**方差**（衡量离散程度）：

$$\boxed{s^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2 = \frac{1}{n}\sum_{i=1}^{n} x_i^2 - \bar{x}^2}$$

**标准差**：$s = \sqrt{s^2}$，单位与原数据相同。

**线性变换性质**：若 $y_i = a x_i + b$，则

$$\bar{y} = a\bar{x} + b, \quad s_y^2 = a^2 s_x^2, \quad s_y = |a| s_x$$

**中位数 / 众数 / 极差**：中位数 = 排序后中间位置；众数 = 出现最多的数；极差 = $\max - \min$。

### 7.2 三种抽样方法 `(part7/01)`

| 方法 | 适用条件 | 关键公式 |
|---|---|---|
| **简单随机抽样** | 总体不大且无明显差异 | 抽签法、随机数表 |
| **系统抽样** | 总体均匀且较大 | 抽样间隔 $k = \dfrac{N}{n}$，先随机抽一个起点，再每 $k$ 个抽一个 |
| **分层抽样** | 总体有明显层次差异 | 各层抽取 $n_i = N_i \cdot \dfrac{n}{N}$ |

**核心原则**：每个个体被抽到的概率均为 $\dfrac{n}{N}$。

### 7.3 频率分布直方图 `(part7/02)`

- **纵轴**：**频率 / 组距**（**不是频率本身**）。
- **每个矩形面积** = 该组的频率。
- **所有矩形面积之和 = 1**。
- 频数 = 频率 × 样本容量。

**估计指标**：
- 中位数 $\approx$ 累积频率为 $0.5$ 时对应的 $x$；
- 众数 $\approx$ 最高矩形所对应的区间中点；
- 平均数 $\approx \sum (\text{区间中点}) \cdot (\text{对应频率})$。

### 7.4 相关系数与回归 `(part7/03)`

**相关系数**：

$$r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \cdot \sum_{i=1}^n (y_i - \bar{y})^2}}$$

**含义**：$r \in [-1, 1]$。

| $r$ 的范围 | 相关性 |
|---|---|
| $r > 0$ | 正相关 |
| $r < 0$ | 负相关 |
| $\vert r\vert \to 1$ | 线性相关性强 |
| $\vert r\vert \to 0$ | 线性相关性弱（可能无线性关系） |

**最小二乘回归直线** $\hat{y} = \hat{b}x + \hat{a}$：

$$\boxed{\hat{b} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{\sum x_i y_i - n\bar{x}\bar{y}}{\sum x_i^2 - n\bar{x}^2}}$$

$$\boxed{\hat{a} = \bar{y} - \hat{b}\bar{x}}$$

**核心性质**：回归直线**必过样本中心** $(\bar{x}, \bar{y})$。

**残差**：$e_i = y_i - \hat{y}_i$。残差平方和 $\sum e_i^2$ 越小，拟合越好。

**决定系数**：$R^2 = 1 - \dfrac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$，越接近 1 拟合越好。

> **易错提醒（Part 7）**
> 1. 频率分布直方图纵轴是 **频率 / 组距**，不是频率本身。
> 2. 方差线性变换：$y = ax + b$ 时 $s_y^2 = a^2 s_x^2$（$b$ 不影响方差），不是 $a s_x^2$。
> 3. 相关系数 $|r|$ 大只说明**线性相关性强**，不代表因果关系，也可能存在非线性关系。

---

## Part 8 概率

### 8.1 古典概型与基本运算 `(part8/01)`

**古典概型**（等可能、有限）：

$$\boxed{P(A) = \frac{n(A)}{n(\Omega)} = \frac{\text{事件 } A \text{ 包含的样本点数}}{\text{样本空间总数}}}$$

**概率公理**：$0 \le P(A) \le 1$；$P(\Omega) = 1$；$P(\varnothing) = 0$。

**加法公式**：

$$\boxed{P(A \cup B) = P(A) + P(B) - P(AB)}$$

**互斥事件**（$AB = \varnothing$）：$P(A \cup B) = P(A) + P(B)$。

**对立事件**（$A \cup \overline{A} = \Omega$，$A \cap \overline{A} = \varnothing$）：

$$\boxed{P(\overline{A}) = 1 - P(A)}$$

### 8.2 几何概型 `(part8/02)`

**度量化概率**（无限等可能）：

$$\boxed{P(A) = \frac{\mu(A)}{\mu(\Omega)} = \frac{\text{事件 } A \text{ 的度量}}{\text{样本空间度量}}}$$

度量按维度选取：一维取长度、二维取面积、三维取体积。

### 8.3 事件的独立性 `(part8/03)`

**独立定义**：

$$\boxed{P(AB) = P(A) \cdot P(B) \Leftrightarrow A, B \text{ 相互独立}}$$

**独立性等价条件**：$P(B|A) = P(B)$（已知 $A$ 不改变 $B$ 的概率）。

**多事件独立**：$A_1, A_2, \ldots, A_n$ 相互独立 $\Leftrightarrow$ 任取 $k$ 个事件的交事件概率等于各自概率之积。

**性质**：若 $A, B$ 独立，则 $A$ 与 $\overline{B}$、$\overline{A}$ 与 $B$、$\overline{A}$ 与 $\overline{B}$ 都独立。

**独立 vs 互斥**（容易混淆）：

| 项目 | 独立 | 互斥 |
|---|---|---|
| 含义 | 事件互不影响概率 | 事件不能同时发生 |
| 关系 | $P(AB) = P(A)P(B)$ | $P(AB) = 0$ |
| 兼容性 | 若 $P(A), P(B) > 0$，**独立则不互斥**，**互斥则不独立** |

### 8.4 条件概率 `(part8/04)`

$$\boxed{P(B|A) = \frac{P(AB)}{P(A)} \quad (P(A) > 0)}$$

**乘法公式**（由条件概率反求联合概率）：

$$P(AB) = P(A) \cdot P(B|A) = P(B) \cdot P(A|B)$$

### 8.5 全概率公式与贝叶斯 `(part8/05)`

设 $A_1, A_2, \ldots, A_n$ 为样本空间 $\Omega$ 的一组**划分**（两两互斥、并为 $\Omega$、每个 $P(A_i) > 0$），$B$ 为任一事件：

**全概率公式**：

$$\boxed{P(B) = \sum_{i=1}^n P(A_i) \cdot P(B|A_i)}$$

直观理解：把 $B$ "分而治之" 到每个 $A_i$ 上。

**贝叶斯公式**（已知结果 $B$ 反推原因 $A_k$）：

$$\boxed{P(A_k | B) = \frac{P(A_k) \cdot P(B|A_k)}{P(B)} = \frac{P(A_k) \cdot P(B|A_k)}{\sum_{i=1}^n P(A_i) P(B|A_i)}}$$

**记忆**：分子 = "$A_k$ 这条路径"，分母 = "全部路径之和"。

### 8.6 $n$ 重独立试验（伯努利）`(part8/03)`

每次试验只有"成功 / 失败"两种结果，概率分别为 $p, 1-p$，试验间相互独立。

恰好 $k$ 次成功的概率：

$$P(X = k) = C_n^k p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

至少一次发生的概率（用对立事件）：$P(X \ge 1) = 1 - (1-p)^n$。

> **易错提醒（Part 8）**
> 1. 互斥与独立要分清：**互斥 → 不能同时发生；独立 → 互不影响概率**。两者一般不重合。
> 2. 条件概率分母 $P(A) > 0$，写公式前先确认 $A$ 可能发生。
> 3. 贝叶斯公式中分母容易少项，要写完整的 $\sum P(A_i) P(B|A_i)$。

---

## Part 9 数列

### 9.1 数列基本概念 `(part9/01)`

数列 $\{a_n\}$：按一定顺序排列的一列数。第 $n$ 项为 $a_n$（通项）。

**前 $n$ 项和**：$S_n = a_1 + a_2 + \cdots + a_n$。

**通项与 $S_n$ 关系**：

$$\boxed{a_n = \begin{cases} S_1, & n = 1 \\ S_n - S_{n-1}, & n \ge 2 \end{cases}}$$

注意 $n = 1$ 单独验证；若 $S_n - S_{n-1}$ 在 $n = 1$ 时也成立，可统一写。

### 9.2 等差数列 `(part9/02)`

**定义**：$a_{n+1} - a_n = d$（公差）。

**通项公式**：

$$\boxed{a_n = a_1 + (n - 1)d}$$

**任意两项关系**：$a_n = a_m + (n - m)d$。

**前 $n$ 项和**：

$$\boxed{S_n = \frac{n(a_1 + a_n)}{2} = n a_1 + \frac{n(n-1)}{2} d}$$

**重要判据**：$\{a_n\}$ 等差 $\Leftrightarrow S_n = An^2 + Bn$（关于 $n$ 的**无常数项**二次式）。

**中项公式**：$2 a_n = a_{n-1} + a_{n+1}$（等差中项）。

**性质**：若 $m + n = p + q$，则 $a_m + a_n = a_p + a_q$。

### 9.3 等比数列 `(part9/03)`

**定义**：$\dfrac{a_{n+1}}{a_n} = q$（公比），$a_n \ne 0$，$q \ne 0$。

**通项公式**：

$$\boxed{a_n = a_1 q^{n-1}}$$

**任意两项关系**：$a_n = a_m q^{n-m}$。

**前 $n$ 项和**：

$$\boxed{S_n = \begin{cases} n a_1, & q = 1 \\ \dfrac{a_1(1 - q^n)}{1 - q} = \dfrac{a_1 - a_n q}{1 - q}, & q \ne 1 \end{cases}}$$

**中项公式**：$a_n^2 = a_{n-1} \cdot a_{n+1}$（等比中项；只有同号才有等比中项 $a_n = \pm\sqrt{a_{n-1} a_{n+1}}$）。

**性质**：若 $m + n = p + q$，则 $a_m \cdot a_n = a_p \cdot a_q$。

### 9.4 递推数列五类构造 `(part9/04)`

| 递推形式 | 构造方法 | 化为 |
|---|---|---|
| $a_{n+1} = a_n + f(n)$ | **累加法** | $a_n - a_1 = \sum f(k)$ |
| $a_{n+1} = a_n \cdot f(n)$ | **累乘法** | $\dfrac{a_n}{a_1} = \prod f(k)$ |
| $a_{n+1} = p a_n + q$（$p \ne 1$） | **待定系数法** | 令 $a_n + \lambda = p(a_{n-1} + \lambda)$，构造等比 |
| $a_{n+1} = \dfrac{p a_n}{q a_n + r}$ | **取倒数法** | 令 $b_n = \dfrac{1}{a_n}$，得线性递推 |
| $a_{n+1} = p a_n + q^n$ | **除幂法** | 两边除以 $q^{n+1}$ 或 $p^{n+1}$ |

### 9.5 数列求和六法 `(part9/05)`

| 类型 | 方法 | 典例 |
|---|---|---|
| 等差 / 等比 | 直接公式 | $S_n = \dfrac{n(a_1 + a_n)}{2}$ 等 |
| 等差 × 等比 | **错位相减** | $\sum n \cdot 2^n$ 类 |
| 分子常数、分母乘积 | **裂项相消** | $\dfrac{1}{n(n+1)} = \dfrac{1}{n} - \dfrac{1}{n+1}$ |
| 首尾对称 | **倒序相加** | 推导 $S_n$ 公式本身 |
| 多种类型混合 | **分组求和** | $\sum (2n + 3^n) = \sum 2n + \sum 3^n$ |
| 通项含奇偶 | **奇偶讨论 / 拆项** | $a_n = (-1)^n \cdot n$ 类 |

**常见裂项**：

$$\frac{1}{n(n+1)} = \frac{1}{n} - \frac{1}{n+1}, \quad \frac{1}{n(n+2)} = \frac{1}{2}\left(\frac{1}{n} - \frac{1}{n+2}\right)$$

$$\frac{1}{\sqrt{n} + \sqrt{n+1}} = \sqrt{n+1} - \sqrt{n}$$

> **易错提醒（Part 9）**
> 1. 用 $a_n = S_n - S_{n-1}$ 求通项时，必须验证 $n = 1$ 是否符合通式。
> 2. 等比求和必须先讨论 $q = 1$ 与 $q \ne 1$；含参时不能直接用 $\dfrac{a_1(1 - q^n)}{1 - q}$。
> 3. 等比中项 $a_n^2 = a_{n-1} a_{n+1}$ 不能直接开方；必须验证同号。

---

## Part 10 导数

### 10.1 导数定义 `(part10/01)`

$$\boxed{f'(x_0) = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}}$$

**几何意义**：$f'(x_0)$ 等于曲线 $y = f(x)$ 在点 $(x_0, f(x_0))$ 处的切线斜率。

**切线方程**：

$$y - f(x_0) = f'(x_0)(x - x_0)$$

### 10.2 基本导数公式表 `(part10/02)`

| 函数 | 导数 |
|---|---|
| $C$（常数） | $0$ |
| $x^\alpha$（$\alpha \in \mathbb{R}$） | $\alpha x^{\alpha - 1}$ |
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $e^x$ | $e^x$ |
| $a^x$（$a > 0, a \ne 1$） | $a^x \ln a$ |
| $\ln x$ | $\dfrac{1}{x}$ |
| $\log_a x$ | $\dfrac{1}{x \ln a}$ |
| $\tan x$ | $\sec^2 x = \dfrac{1}{\cos^2 x}$ |

### 10.3 导数四则与链式 `(part10/02)`

设 $u = u(x)$，$v = v(x)$ 均可导：

$$\boxed{(u \pm v)' = u' \pm v'}$$

$$\boxed{(uv)' = u'v + uv'}$$

$$\boxed{\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2} \quad (v \ne 0)}$$

**链式法则**（复合函数）：

$$\boxed{[f(g(x))]' = f'(g(x)) \cdot g'(x)}$$

口诀：**外导 × 内导**。

### 10.4 导数与单调性 `(part10/03)`

设 $f(x)$ 在区间 $I$ 上可导：

| 条件 | 结论 |
|---|---|
| $f'(x) > 0$ 在 $I$ 上恒成立 | $f(x)$ 在 $I$ 上**单调递增** |
| $f'(x) < 0$ 在 $I$ 上恒成立 | $f(x)$ 在 $I$ 上**单调递减** |
| $f'(x) = 0$ 在 $I$ 上恒成立 | $f(x)$ 在 $I$ 上为**常函数** |

**反向**：若 $f$ 在 $I$ 上单调递增，则 $f'(x) \ge 0$（不严格）。

**求单调区间四步法**：
1. 求导 $f'(x)$；
2. 解 $f'(x) > 0$ 与 $f'(x) < 0$；
3. 与定义域取交集；
4. 写出单调递增 / 递减区间（区间之间用 "和" 或 "，" 分隔，不写并集 ∪）。

### 10.5 极值与最值 `(part10/04)`

**极值必要条件**：若 $f$ 在 $x_0$ 取极值且 $f'(x_0)$ 存在，则 $f'(x_0) = 0$（**驻点**）。

**极值充分条件**（"变号"判据）：

| $f'$ 在 $x_0$ 两侧 | $x_0$ 是 |
|---|---|
| 左正右负 | 极大值点 |
| 左负右正 | 极小值点 |
| 同号 | 非极值点（仅是驻点） |

**闭区间最值**：在 $[a, b]$ 上连续的 $f(x)$ 的最大 / 最小值在以下三类点取得：
1. **驻点**（$f'(x) = 0$）；
2. **不可导点**（导数不存在）；
3. **端点** $x = a, x = b$。

把所有候选点的 $f$ 值算出来比较即可。

> **易错提醒（Part 10）**
> 1. $f'(x_0) = 0$ 是极值的**必要条件**，**不是充分条件**（如 $f(x) = x^3$ 在 $x = 0$）。
> 2. 写单调区间时不能用 $\cup$，例如 $f(x) = \dfrac{1}{x}$ 的减区间写成 $(-\infty, 0)$ 和 $(0, +\infty)$，不写并集。
> 3. 求闭区间最值不要漏端点，也不要漏不可导点。

---

## Part 11 计数原理

### 11.1 两个基本原理 `(part11/01)`

**分类加法原理**：完成一件事有 $n$ 类方法，第 $i$ 类有 $m_i$ 种，共

$$N = m_1 + m_2 + \cdots + m_n \quad \text{（各类互不影响）}$$

**分步乘法原理**：完成一件事需 $n$ 步，第 $i$ 步有 $m_i$ 种，共

$$N = m_1 \cdot m_2 \cdots m_n \quad \text{（各步必须依次完成）}$$

**口诀**："**类加步乘**"。识别：分类——只选其一；分步——必须全做。

### 11.2 排列数 `(part11/02)`

从 $n$ 个不同元素中取 $m$ 个排成一列：

$$\boxed{A_n^m = n(n-1)(n-2) \cdots (n - m + 1) = \frac{n!}{(n-m)!}}$$

**特殊**：$A_n^n = n!$（全排列），$A_n^0 = 1$，$0! = 1$。

### 11.3 组合数 `(part11/03)`

从 $n$ 个不同元素中取 $m$ 个（不计顺序）：

$$\boxed{C_n^m = \frac{A_n^m}{m!} = \frac{n!}{m!(n-m)!}}$$

**关键性质**：

$$\boxed{C_n^m = C_n^{n - m} \quad \text{（对称性）}}$$

$$\boxed{C_n^m + C_n^{m-1} = C_{n+1}^m \quad \text{（帕斯卡恒等式）}}$$

$$C_n^0 + C_n^1 + \cdots + C_n^n = 2^n \quad \text{（全集子集数）}$$

### 11.4 二项式定理 `(part11/04)`

$$\boxed{(a + b)^n = \sum_{k=0}^n C_n^k a^{n-k} b^k}$$

**通项**：

$$\boxed{T_{k+1} = C_n^k a^{n-k} b^k \quad (k = 0, 1, \ldots, n)}$$

注意：**第 $k+1$ 项**对应 $C_n^k$，下标从 $0$ 起，常错位。

**二项展开系数性质**：
- **对称性**：$C_n^k = C_n^{n-k}$；
- **二项系数和**：令 $a = b = 1$，得 $\sum C_n^k = 2^n$；
- **奇偶项之和**：令 $a = 1, b = -1$，得 $\sum_{k \text{ 偶}} C_n^k - \sum_{k \text{ 奇}} C_n^k = 0$；
- **奇偶项分别求和**：奇偶项和均为 $2^{n-1}$。

**系数 vs 二项系数**：
- **二项系数** 指 $C_n^k$（不含 $a, b$ 中的数字）；
- **项的系数** 指完整数值系数（含 $a, b$ 中的常数）；
- 求"系数最大项"必须算具体系数，不能只看 $C_n^k$。

> **易错提醒（Part 11）**
> 1. **分类 vs 分步要分清**：能独立完成的是分类（加），必须连贯才能完成的是分步（乘）。
> 2. 排列数下标从 $1$ 开始（如 $A_5^2 = 5 \cdot 4$），组合数同理。
> 3. 二项展开通项 $T_{k+1}$，要先确定要求哪一项再找 $k$，不要直接代第 $n$ 项的下标。

---

## Part 12 随机变量

### 12.1 离散型随机变量的期望与方差 `(part12/01)`

设离散型随机变量 $X$ 的分布列为

| $X$ | $x_1$ | $x_2$ | $\cdots$ | $x_n$ |
|---|---|---|---|---|
| $P$ | $p_1$ | $p_2$ | $\cdots$ | $p_n$ |

且 $\sum p_i = 1$。

**期望**（数学期望，加权平均）：

$$\boxed{E(X) = \sum_{i=1}^n x_i p_i}$$

**方差**：

$$\boxed{D(X) = \sum_{i=1}^n [x_i - E(X)]^2 p_i = E(X^2) - [E(X)]^2}$$

**标准差**：$\sigma(X) = \sqrt{D(X)}$。

**线性变换**（$a, b$ 为常数）：

$$\boxed{E(aX + b) = a E(X) + b, \quad D(aX + b) = a^2 D(X)}$$

注意：$D$ 中 $a$ 要平方，$b$ 消失（常数不影响离散度）。

### 12.2 两点分布（0-1 分布）`(part12/01)`

$X \sim B(1, p)$：$P(X = 1) = p$，$P(X = 0) = 1 - p$。

$$E(X) = p, \quad D(X) = p(1 - p)$$

### 12.3 二项分布 `(part12/02)`

$X \sim B(n, p)$，表示 $n$ 重伯努利试验中成功次数：

$$P(X = k) = C_n^k p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

$$\boxed{E(X) = np, \quad D(X) = np(1-p)}$$

### 12.4 超几何分布 `(part12/03)`

总体 $N$ 件中有 $M$ 件次品，不放回抽 $n$ 件，恰有 $k$ 件次品：

$$P(X = k) = \frac{C_M^k C_{N-M}^{n-k}}{C_N^n}, \quad \max(0, n - N + M) \le k \le \min(n, M)$$

$$\boxed{E(X) = \frac{nM}{N}}$$

**记忆**：超几何 = 不放回；二项 = 有放回。$n$ 远小于 $N$ 时，超几何 $\approx$ 二项。

### 12.5 正态分布 `(part12/04)`

$X \sim N(\mu, \sigma^2)$，概率密度

$$\varphi(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$

**几何**：钟形曲线关于 $x = \mu$ 对称，$\sigma$ 越大越扁平。

**3σ 原则**：

$$\boxed{P(\mu - \sigma < X < \mu + \sigma) \approx 0.6826}$$

$$\boxed{P(\mu - 2\sigma < X < \mu + 2\sigma) \approx 0.9544}$$

$$\boxed{P(\mu - 3\sigma < X < \mu + 3\sigma) \approx 0.9974}$$

记忆口诀：**"68 / 95 / 99.7"** 法则。

**标准化**：令 $Z = \dfrac{X - \mu}{\sigma}$，则 $Z \sim N(0, 1)$。

**期望与方差**：

$$E(X) = \mu, \quad D(X) = \sigma^2$$

> **易错提醒（Part 12）**
> 1. 方差线性变换：$D(aX + b) = a^2 D(X)$，**$a$ 要平方、$b$ 消失**。
> 2. 二项分布要求**有放回 / 独立同分布**；超几何分布是**不放回**。
> 3. 正态分布的对称性可用来简化概率：$P(X < \mu - a) = P(X > \mu + a)$。

---

## Part 13 综合（砖头不等式与综合方法）

### 13.1 三大砖头不等式 `(part13/04)`

**对数砖**：

$$\boxed{\ln(1 + x) < x \quad (x > 0; \text{ 等号当 } x = 0)}$$

更精细：$\dfrac{x}{1 + x} \le \ln(1 + x) \le x$（$x > -1$）。

**指数砖**：

$$\boxed{e^x \ge 1 + x \quad (\text{等号当 } x = 0)}$$

由 $e^x$ 在 $x = 0$ 的切线 $y = 1 + x$ 在 $e^x$ 下方得出。

**正弦砖**：

$$\boxed{\sin x < x \quad (x > 0); \quad |\sin x| \le |x| \quad (\forall x \in \mathbb{R})}$$

由 $\sin x$ 在 $x = 0$ 处切线 $y = x$ 与函数单调性结合得出。

### 13.2 切线放缩法 `(part13/04)`

利用曲线 $y = f(x)$ 在 $x = x_0$ 的切线作为放缩工具：

- 凸函数 $\Rightarrow$ 切线**在曲线下方**；
- 凹函数 $\Rightarrow$ 切线**在曲线上方**。

常用切线：
- $e^x$ 在 $x = 0$：$y = 1 + x$ → $e^x \ge 1 + x$；
- $\ln x$ 在 $x = 1$：$y = x - 1$ → $\ln x \le x - 1$；
- $\sin x$ 在 $x = 0$：$y = x$ → $\sin x \le x$（$x \ge 0$）。

### 13.3 辅助角公式（综合常用）`(part13/05)`

$$\boxed{a\sin x + b\cos x = R\sin(x + \varphi), \quad R = \sqrt{a^2 + b^2}}$$

其中 $\cos\varphi = \dfrac{a}{R}$，$\sin\varphi = \dfrac{b}{R}$（由 $a, b$ 符号定 $\varphi$ 象限）。

**用途**：三角综合题最值、值域、周期、对称轴等问题。

### 13.4 单调 + 端点 = 不等式证明 `(part13/04)`

**模板**（证 $f(x) > g(x)$ 在区间 $I$ 上恒成立）：

1. 构造 $h(x) = f(x) - g(x)$；
2. 求 $h'(x)$，判 $h$ 单调性；
3. 取端点（或极值点）值 $h(x_0)$，若 $h$ 在端点处的值非负，则结合单调性得 $h(x) \ge 0$。

**关键**：从端点入手——**单调性 + 端点值** = 不等式整段成立。

### 13.5 数列与不等式综合 `(part13/03)`

**数学归纳法三步**：

1. **奠基**：$n = 1$（或 $n = n_0$）时成立；
2. **归纳假设**：假设 $n = k$ 时成立；
3. **归纳推进**：证明 $n = k + 1$ 也成立。

**结合数列不等式**：常需在归纳推进时**放缩**（如裂项放缩、用 $\dfrac{1}{n(n+1)} < \dfrac{1}{n^2} < \dfrac{1}{(n-1)n}$）。

### 13.6 决策综合（期望 + 方差）`(part13/06)`

**决策原则**：
- **比期望**：期望大者收益高（首要考虑）；
- **比方差 / 标准差**：期望相同时，方差小者风险低（稳定性更好）；
- **风险偏好**：保守型选小 $D$，进取型选大 $E$。

**典型场景**：投资方案 / 抽奖游戏 / 试卷选择题对错策略。

> **易错提醒（Part 13）**
> 1. 三大砖头不等式的方向不要写反；$\ln(1+x) \le x$、$e^x \ge 1 + x$ 都是"切线在下"或"切线在上"。
> 2. 数学归纳法在归纳推进时要**用到归纳假设**，否则推理不成立。
> 3. 期望相同时再比方差；不要见到方差小就选——必须先看期望。

---

*本表覆盖 Part 1–13 全部 55 章核心公式与定理。配合附录 B 模型图集与各章详细推导使用，可在高三总复习时快速完成知识点扫描。*
