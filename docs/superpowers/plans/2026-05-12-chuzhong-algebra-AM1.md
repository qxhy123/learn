# 中考代数教程 AM1 里程碑实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** 完成 `chuzhong_algebra/` 的 part1（数, 7 章）+ part2（整式, 7 章），共 14 章。

**Architecture:** 12 章用模板 C（概念/运算法则类，6 节），2 章用模板 D（公式逆用 + 因式分解后置铺垫）。每章 1500-2800 字，独立提交。

**Spec 参考:** `docs/superpowers/specs/2026-05-12-chuzhong-algebra-design.md`
**前序里程碑:** AM0（已完成）

---

## 文件结构

```
chuzhong_algebra/
├── part1-numbers/                             # 数 [初一上 + 初二上根式]
│   ├── 01-rational-numbers.md                 # 有理数（含相反数/绝对值/数轴）
│   ├── 02-rational-operations.md              # 有理数加减乘除乘方
│   ├── 03-power-and-scientific.md             # 乘方深化 + 科学记数法
│   ├── 04-real-numbers.md                     # 实数与无理数
│   ├── 05-square-cube-roots.md                # 平方根 / 立方根
│   ├── 06-radicals.md                         # 二次根式
│   └── 07-radical-operations.md               # 二次根式运算
└── part2-polynomials/                         # 整式 [初一下]
    ├── 01-monomial-polynomial.md              # 单项式 / 多项式
    ├── 02-add-subtract.md                     # 整式加减
    ├── 03-exponent-rules.md                   # 幂的运算法则
    ├── 04-multiply.md                         # 整式乘法
    ├── 05-special-formulas.md                 # ⭐ 完全平方/平方差（模板 D）
    ├── 06-divide.md                           # 整式除法
    └── 07-special-formula-tricks.md           # ⭐ 公式逆用与变形（模板 D）
```

---

## 模板规范

**模板 C（12 章）—— 6 节:**
1. 概念特征
2. 定义与运算法则
3. 推导/为什么
4. 典型应用（2-3 例，带【思路】）
5. 易错点 & 反例
6. 思路自测题（3-5 题，用 `💡 提示：`）

**模板 D（2 章：part2/05, part2/07）—— 7 节:**
1. 一例速记
2. 引入题
3. 思维路径还原（`>` 引用，12-18 行）
4. 抽象成方法
5. 方法变形
6. 思考路标（≥ 6 条）
7. 应用例题 + 自测题

字数：模板 C 1800-2500，模板 D 3000-4000。LaTeX 全部数学符号。

---

## Task 1: part1/01 有理数

**File:** `chuzhong_algebra/part1-numbers/01-rational-numbers.md`（模板 C，1800-2500 字）

Title: `# 有理数`

涵盖：
- **概念**：有理数 = 整数 + 分数（含正负零）；用集合表示
- **数轴**：三要素（原点 / 正方向 / 单位长度）；任何有理数都对应数轴上唯一点
- **相反数**：定义（绝对值相等、符号相反）；表示 $-a$；性质：$a + (-a) = 0$
- **绝对值**：定义 $|a| = \begin{cases} a, & a \geq 0 \\ -a, & a < 0 \end{cases}$；几何意义：数轴上点到原点距离
- **比较大小**：用数轴 / 用绝对值（同号比绝对值，异号正大于负）
- **典型应用** 3 例：
  - 例 1：求 $|-3|, |0|, -|−5|$
  - 例 2：比较 $-3.5, -\frac{7}{2}, -\sqrt{12}$ 的大小
  - 例 3：已知 $|a-2| + |b+3| = 0$，求 $a, b$（绝对值非负，两项和为 0 各为 0）
- **易错点**：$|a|$ 总是非负；$-a$ 不一定是负数（取决于 $a$ 符号）
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part1/01 (rational numbers)`

---

## Task 2: part1/02 有理数运算

**File:** `chuzhong_algebra/part1-numbers/02-rational-operations.md`（模板 C，1800-2500 字）

Title: `# 有理数的运算`

- **加减法**：
  - 同号相加：取相同符号 + 绝对值相加
  - 异号相加：取较大绝对值的符号 + 绝对值相减
  - 减法 = 加上相反数
- **乘除法**：
  - 同号得正、异号得负；多个数相乘：负因数个数偶数为正，奇数为负
  - $0 \cdot a = 0$；$a \div 0$ 无意义
- **乘方**：$a^n = \underbrace{a \cdot a \cdots a}_{n \text{ 个}}$；$(-a)^n$ 与 $-a^n$ 区别
- **混合运算顺序**：先乘方 → 再乘除 → 后加减；同级从左到右；括号优先
- **运算律**：加法/乘法的交换律、结合律、分配律
- **典型应用** 3 例：
  - 例 1：基础混合运算 $-3^2 + (-2)^3 \times (-\frac{1}{2})^2$
  - 例 2：用分配律简便计算 $-\frac{1}{6} \times (-\frac{1}{12} + \frac{1}{4} - \frac{1}{2})$
  - 例 3：含绝对值的混合运算
- **易错点**：$-3^2 = -9$（先平方再取负），$(-3)^2 = 9$；除以 0 无意义；除法没有结合律
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part1/02 (rational operations)`

---

## Task 3: part1/03 乘方与科学记数法

**File:** `chuzhong_algebra/part1-numbers/03-power-and-scientific.md`（模板 C，1500-2200 字）

Title: `# 乘方与科学记数法`

- **乘方深化**：底数 / 指数 / 幂；正数任何次幂为正；负数偶次幂为正、奇次幂为负
- **科学记数法**：$a \times 10^n$（$1 \leq |a| < 10$，$n$ 为整数）
  - 大数：$n =$ 整数部分位数 − 1（如 $3 \times 10^8 = 3$ 亿）
  - 小数：$n =$ 负数（如 $0.000003 = 3 \times 10^{-6}$）
- **有效数字**：从左边第一个非零数字起，到末位的所有数字
- **近似数与精确度**：四舍五入 / 精确到某位
- **典型应用** 3 例：
  - 例 1：化 $3, 200, 000$ 为科学记数法 → $3.2 \times 10^6$
  - 例 2：化 $0.0000058$ 为科学记数法 → $5.8 \times 10^{-6}$
  - 例 3：$2.05 \times 10^4$ 的有效数字（3 个：2, 0, 5）
- **易错点**：$a$ 必须满足 $1 \leq |a| < 10$；小数的 $n$ 为负且 = 小数点后到第一个非零数字的位数取负
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part1/03 (power and scientific notation)`

---

## Task 4: part1/04 实数与无理数

**File:** `chuzhong_algebra/part1-numbers/04-real-numbers.md`（模板 C，1800-2500 字）

Title: `# 实数与无理数`

- **无理数的发现**：$\sqrt{2}$ 是无限不循环小数
- **常见无理数类型**：
  - 开方开不尽的数 $\sqrt{2}, \sqrt{3}, \sqrt[3]{5}$
  - 圆周率 $\pi$
  - 构造类 $0.1010010001\ldots$（无周期）
- **实数 = 有理数 ∪ 无理数**
- **实数与数轴**：实数与数轴上的点**一一对应**
- **实数的运算**：有理数运算法则在实数仍成立（含开方）
- **典型应用** 3 例：
  - 例 1：判断 $\sqrt{4}, \sqrt{5}, \pi, 0.333\ldots, \frac{22}{7}$ 哪些是无理数
  - 例 2：估算 $\sqrt{20}$ 在哪两个整数之间（$4 < \sqrt{20} < 5$，因为 $16 < 20 < 25$）
  - 例 3：实数大小比较（含根号）
- **易错点**：$0.333\ldots$ 是循环小数 = $\frac{1}{3}$ 是有理数；$\sqrt{9} = 3$ 是有理数；$\frac{22}{7}$ 是有理数（$\pi$ 才是无理数）
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part1/04 (real numbers and irrationals)`

---

## Task 5: part1/05 平方根/立方根

**File:** `chuzhong_algebra/part1-numbers/05-square-cube-roots.md`（模板 C，1800-2500 字）

Title: `# 平方根与立方根`

- **平方根**：$x^2 = a$（$a \geq 0$）的 $x$ 称为 $a$ 的平方根
  - 正数有两个平方根（互为相反数）记作 $\pm\sqrt{a}$
  - 0 的平方根是 0
  - 负数没有平方根
- **算术平方根**：正的那个平方根，记作 $\sqrt{a}$
- **立方根**：$x^3 = a$ 的 $x$ 称为 $a$ 的立方根，记作 $\sqrt[3]{a}$
  - 任何数都有立方根；正数立方根为正，负数立方根为负
- **性质**：$(\sqrt{a})^2 = a$（$a \geq 0$）；$\sqrt{a^2} = |a|$；$(\sqrt[3]{a})^3 = a$
- **典型应用** 3 例：
  - 例 1：求 $\sqrt{81}, \pm\sqrt{0.25}, \sqrt[3]{-27}$
  - 例 2：解方程 $x^2 = 9$（$x = \pm 3$）vs $\sqrt{x} = 3$（$x = 9$，仅一解）
  - 例 3：化简 $\sqrt{(-5)^2}$（注意 $= 5$ 不是 $-5$）
- **易错点**：$\sqrt{a^2} = |a|$ 不一定等于 $a$；负数无平方根但有立方根；$\sqrt{a}$ 默认是算术平方根（非负）
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part1/05 (square and cube roots)`

---

## Task 6: part1/06 二次根式

**File:** `chuzhong_algebra/part1-numbers/06-radicals.md`（模板 C，1800-2500 字）

Title: `# 二次根式`

- **概念**：形如 $\sqrt{a}$（$a \geq 0$）的式子；$a$ 为被开方数
- **有意义条件**：$a \geq 0$（含字母时需讨论范围，如 $\sqrt{x-1}$ 需 $x \geq 1$）
- **最简二次根式**：被开方数不含**平方因子**，不含**分母**
  - 反例：$\sqrt{8} = 2\sqrt{2}$（含平方因子 4）；$\sqrt{\frac{1}{2}} = \frac{\sqrt{2}}{2}$（含分母）
- **同类二次根式**：化简后被开方数相同
- **主要性质**：
  - $\sqrt{a^2} = |a|$
  - $\sqrt{ab} = \sqrt{a} \cdot \sqrt{b}$（$a, b \geq 0$）
  - $\sqrt{\frac{a}{b}} = \frac{\sqrt{a}}{\sqrt{b}}$（$a \geq 0, b > 0$）
- **典型应用** 3 例：
  - 例 1：化 $\sqrt{50}, \sqrt{48}, \sqrt{\frac{3}{4}}$ 为最简
  - 例 2：求 $\sqrt{x-1}$ 中 $x$ 的范围
  - 例 3：化简 $\sqrt{18} + \sqrt{8} - \sqrt{2}$（先化最简 + 合并同类）
- **易错点**：负数不能在根号内（除非用 $\sqrt{-a}$ 表示 $-a \geq 0$）；分母有理化要"上下同乘共轭"
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part1/06 (radicals)`

---

## Task 7: part1/07 二次根式运算

**File:** `chuzhong_algebra/part1-numbers/07-radical-operations.md`（模板 C，1800-2500 字）

Title: `# 二次根式的运算`

- **加减法**：先化最简，再合并同类二次根式
- **乘法**：$\sqrt{a} \cdot \sqrt{b} = \sqrt{ab}$
- **除法（分母有理化）**：$\frac{1}{\sqrt{a}} = \frac{\sqrt{a}}{a}$；$\frac{1}{\sqrt{a}+\sqrt{b}} = \frac{\sqrt{a}-\sqrt{b}}{a-b}$（乘共轭）
- **典型应用** 3 例：
  - 例 1：基础加减 $\sqrt{18} - \sqrt{8} + 3\sqrt{2}$
  - 例 2：分母有理化 $\frac{2}{\sqrt{3}-1}$
  - 例 3：综合运算 $(\sqrt{3} + \sqrt{2})(\sqrt{3} - \sqrt{2})$（用平方差公式）
- **易错点**：分母有理化要乘共轭，不是乘自己；含 $\sqrt{}$ 的乘法用平方差 / 完全平方公式简化
- **自测题** 5 题。

Commit: `docs(chuzhong_algebra): add part1/07 (radical operations)`

---

## Task 8: part2/01 单项式与多项式

**File:** `chuzhong_algebra/part2-polynomials/01-monomial-polynomial.md`（模板 C，1500-2200 字）

Title: `# 单项式与多项式`

- **代数式**：用运算符号和数字字母组合而成的式子（如 $3x, x^2+1, \frac{a}{b}$）
- **整式 = 单项式 + 多项式**（不含字母分母）
- **单项式**：数与字母的乘积（含单个字母、单个数字）
  - 系数：数字因数（如 $-3x^2$ 系数 $-3$）
  - 次数：字母指数之和（$-3x^2y^3$ 次数 $5$）
- **多项式**：几个单项式的和
  - 项：每个单项式叫一项
  - 次数：各项中最高次数
  - 升幂/降幂排列：按某字母的指数升序/降序
- **典型应用** 3 例：
  - 例 1：找出 $-\frac{2}{3}x^2y$ 的系数和次数
  - 例 2：把 $3x^2 - 2x^3 + 1 - x$ 按 $x$ 降幂排列
  - 例 3：判断哪些是整式（含分式 $\frac{1}{x}$ 不是）
- **易错点**：系数包含符号；常数项是次数为 0 的项；$\frac{x}{2}$ 是单项式但 $\frac{1}{x}$ 不是
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part2/01 (monomial and polynomial)`

---

## Task 9: part2/02 整式加减

**File:** `chuzhong_algebra/part2-polynomials/02-add-subtract.md`（模板 C，1500-2200 字）

Title: `# 整式的加减`

- **同类项**：所含字母相同 + 相同字母指数相同（与系数无关、与字母顺序无关）
- **合并同类项**：系数相加，字母及其指数不变
- **去括号法则**：
  - $a + (b - c) = a + b - c$
  - $a - (b - c) = a - b + c$（每项变号）
- **添括号**：与去括号相反（添到 $-$ 后每项变号）
- **典型应用** 3 例：
  - 例 1：合并 $3x^2 - 5x^2 + 2x^2$
  - 例 2：化简 $(2a - 3b) - (a - 2b + c) + (3a + b)$
  - 例 3：化简求值 $3(x^2 + 2x) - (x^2 - x + 1)$，当 $x = -1$
- **易错点**：去括号时括号前为负号要每项变号；合并同类项前先识别（字母顺序无关 $xy = yx$）
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part2/02 (add subtract integers)`

---

## Task 10: part2/03 幂的运算法则

**File:** `chuzhong_algebra/part2-polynomials/03-exponent-rules.md`（模板 C，1800-2500 字）

Title: `# 幂的运算法则`

- **同底数幂相乘**：$a^m \cdot a^n = a^{m+n}$
- **幂的乘方**：$(a^m)^n = a^{mn}$
- **积的乘方**：$(ab)^n = a^n b^n$
- **同底数幂相除**：$a^m \div a^n = a^{m-n}$（$a \neq 0$）
- **零指数**：$a^0 = 1$（$a \neq 0$）
- **负整数指数**：$a^{-n} = \frac{1}{a^n}$（$a \neq 0$，初二上）
- **推导**：每个法则用乘方定义推导（如 $a^2 \cdot a^3 = (aa)(aaa) = a^5$）
- **典型应用** 3 例：
  - 例 1：基础 $x^3 \cdot x^4 \cdot x = x^8$
  - 例 2：综合 $(2a^2 b)^3 \cdot a^2 b^2 \div (a^3 b)^2$
  - 例 3：含负指数 $(2^{-1} + 3^{-1})^{-1}$（结果 $\frac{6}{5}$）
- **易错点**：底数和指数符号；$(a^m)^n \neq a^{m^n}$；$a^0 = 1$ 但 $0^0$ 无意义
- **自测题** 5 题。

Commit: `docs(chuzhong_algebra): add part2/03 (exponent rules)`

---

## Task 11: part2/04 整式乘法

**File:** `chuzhong_algebra/part2-polynomials/04-multiply.md`（模板 C，1500-2200 字）

Title: `# 整式乘法`

- **单 × 单**：$3x^2 \cdot 5x^3 = 15 x^5$（系数相乘 + 幂相乘）
- **单 × 多**：分配律 $a(b+c) = ab + ac$
- **多 × 多**：$(a+b)(c+d) = ac+ad+bc+bd$（每项 × 每项）
- **典型应用** 3 例每带【思路】：
  - 例 1：$-2x(x^2 - 3x + 4) = -2x^3 + 6x^2 - 8x$
  - 例 2：$(x + 2)(2x - 3) = 2x^2 + x - 6$
  - 例 3：$(a + 2b)^2$（直接展开作铺垫 part2/05 公式）
- **易错点**：分配律每项都要乘；符号；不要漏项
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part2/04 (polynomial multiplication)`

---

## Task 12: part2/05 完全平方与平方差公式 ⭐（模板 D）

**File:** `chuzhong_algebra/part2-polynomials/05-special-formulas.md`（**模板 D**，3000-4000 字）

Title: `# 完全平方与平方差公式`

**一例速记**：
> 完全平方 $(a \pm b)^2 = a^2 \pm 2ab + b^2$  
> 平方差 $(a+b)(a-b) = a^2 - b^2$  
> 看到"两平方相加 ± 2 倍乘积" → 完全平方；看到"两平方相减" → 平方差

### 一、引入：一道展开题
> 化简 $(2x + 3y)^2 - (2x - 3y)(2x + 3y)$。盲算需展开两次，每次 4 项再合并；用公式 30 秒。

### 二、思维路径还原（12-18 行 `>` 引用）：
> "$(2x + 3y)^2$ —— 看到两项**和的平方** → 完全平方公式  
> = $(2x)^2 + 2 \cdot 2x \cdot 3y + (3y)^2$  
> = $4x^2 + 12xy + 9y^2$  
> $(2x - 3y)(2x + 3y)$ —— 一减一加 → **平方差公式**  
> = $(2x)^2 - (3y)^2$  
> = $4x^2 - 9y^2$  
> 原式 $= (4x^2 + 12xy + 9y^2) - (4x^2 - 9y^2)$  
> = $12xy + 18y^2$  
> 关键反射：看 $(\cdot)^2$ 立即写完全平方；看 $(\cdot+\cdot)(\cdot-\cdot)$ 立即写平方差。"

### 三、抽象成方法
**完全平方公式**：$(a \pm b)^2 = a^2 \pm 2ab + b^2$
- 三项组合：首平方 + 末平方 ± 2 倍乘积
- 中项符号 = 原式中间符号

**平方差公式**：$(a+b)(a-b) = a^2 - b^2$
- 两项组合：前平方 − 后平方
- "和差积 = 平方差"

**推导**：用多项式乘法展开得到（不是凭空背公式）。

### 四、方法变形
- **公式逆用**（part2/07 单独讲）：$a^2 - b^2$ 见到立即想 $(a+b)(a-b)$
- **复合**：$(a + b + c)^2 = a^2 + b^2 + c^2 + 2(ab + bc + ca)$（三项完全平方）
- **构造**：$98 \times 102 = (100-2)(100+2) = 10000-4 = 9996$

### 五、思考路标（至少 6 条）
- 见 $(\text{两项})^2$ → 完全平方公式
- 见 $(\cdot+\cdot)(\cdot-\cdot)$ → 平方差公式
- 见两数**和/差的平方** → 写成展开式（或反向）
- 见两数**平方的和/差** → 试写成完全平方/平方差
- 见数字接近"整百" → 用平方差简便计算
- 见 $a^2 + b^2$（无 $2ab$）→ **加减 $2ab$** 凑完全平方

### 六、应用例题 3 例
- 例 1：用平方差简便计算 $99 \times 101$（= $100^2 - 1 = 9999$）
- 例 2：已知 $a + b = 5, ab = 6$，求 $a^2 + b^2$（用 $(a+b)^2 - 2ab = 25 - 12 = 13$）
- 例 3：化简 $(x + 2y)^2 - (x - 2y)^2$（= $8xy$，差平方差再展开）

### 七、思路自测题 4 题

Commit: `docs(chuzhong_algebra): add part2/05 (complete square and difference of squares)`

---

## Task 13: part2/06 整式除法

**File:** `chuzhong_algebra/part2-polynomials/06-divide.md`（模板 C，1500-2200 字）

Title: `# 整式的除法`

- **单 ÷ 单**：系数相除 + 幂相除（同底数）
- **多 ÷ 单**：每项除以单项式后相加
- **(初中不要求多项式除多项式)**
- **典型应用** 3 例：
  - 例 1：$12x^3y^2 \div 3xy = 4x^2 y$
  - 例 2：$(6a^3 - 4a^2 + 2a) \div 2a = 3a^2 - 2a + 1$
  - 例 3：用除法验证乘法（如 $(x+2)(x-3) = x^2 - x - 6$ → 除以 $x+2$ 应得 $x-3$）
- **易错点**：系数符号；漏项；零指数
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part2/06 (polynomial division)`

---

## Task 14: part2/07 公式逆用与变形 ⭐（模板 D）

**File:** `chuzhong_algebra/part2-polynomials/07-special-formula-tricks.md`（**模板 D**，3000-4000 字）

Title: `# 完全平方与平方差公式的逆用与变形`

**一例速记**：
> $a^2 - b^2$ 立刻看作 $(a+b)(a-b)$ —— 把"展开式"逆向**还原**成"乘积式"；这是因式分解的"小种子"，也是中考填空题的"加速器"。

### 一、引入：一道"逆向"题
> 已知 $x^2 - y^2 = 12$, $x - y = 2$, 求 $x + y$。

### 二、思维路径还原（12-18 行 `>` 引用）：
> "$x^2 - y^2 = 12$，直接解 $x, y$ 不可能（两个未知一个方程）。  
> 但 $x^2 - y^2$ 看着像**平方差**：$(x+y)(x-y)$。  
> 题目给 $x - y = 2$ —— 整体代入：$(x+y) \cdot 2 = 12$ → $x + y = 6$。  
> 关键：**逆用公式**——把展开式看成乘积形式  
> 类似的：$a^2 + 2ab + b^2 = (a+b)^2$、$a^2 - 2ab + b^2 = (a-b)^2$  
> 见 $a^2 \pm 2ab + b^2$ 立即想"完全平方折叠"成 $(a \pm b)^2$  
> 这是中考填空 / 选择题最常考的"快速变形"。"

### 三、抽象成方法
**逆用三类公式**：
1. $a^2 - b^2 = (a+b)(a-b)$（平方差逆用）
2. $a^2 + 2ab + b^2 = (a+b)^2$（完全平方和）
3. $a^2 - 2ab + b^2 = (a-b)^2$（完全平方差）

**配方法**：见 $a^2 + b^2$ 但无 $2ab$ → 加减 $2ab$ 凑完全平方
- $a^2 + b^2 = (a+b)^2 - 2ab = (a-b)^2 + 2ab$

### 四、方法变形（重要！）
**已知**：$x + \frac{1}{x} = 3$ → 平方得 $x^2 + \frac{1}{x^2} = 9 - 2 = 7$（用 $(a+b)^2 - 2ab$）
**已知**：$a + b = 5, ab = 6$ → $a^2 + b^2 = 13$；$(a-b)^2 = 5^2 - 4 \cdot 6 = 1$ → $|a-b| = 1$

### 五、思考路标（至少 6 条）
- 见 $a^2 - b^2$ → 立刻写 $(a+b)(a-b)$
- 见 $a^2 + b^2$ + 题目给 $a+b, ab$ → 用 $(a+b)^2 - 2ab$
- 见 $a^2 + b^2$ + 题目给 $a-b, ab$ → 用 $(a-b)^2 + 2ab$
- 见 $(a-b)^2$ 求 $a^2 + b^2$ → $+ 2ab$
- 见 $x + \frac{1}{x}$ 求 $x^2 + \frac{1}{x^2}$ → 平方
- 见 $a^2 + b^2 + c^2$ 求 $ab + bc + ca$ → 用 $(a+b+c)^2 = a^2+b^2+c^2 + 2(ab+bc+ca)$

### 六、应用例题 3 例
- 例 1：$(x+y)^2 - (x-y)^2$（= $4xy$，展开相减）
- 例 2：已知 $x+y = 3, xy = 1$，求 $x^2 + y^2, (x-y)^2, x^3 + y^3$
- 例 3：因式分解 $a^4 - 16$（先平方差两次：$(a^2-4)(a^2+4) = (a-2)(a+2)(a^2+4)$）

### 七、思路自测题 5 题

Commit: `docs(chuzhong_algebra): add part2/07 (formula reversal and tricks)`

---

## Task 15: AM1 收尾

- [ ] **Step 1: 一致性扫描**

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' chuzhong_algebra/part1-numbers chuzhong_algebra/part2-polynomials
ls chuzhong_algebra/part1-numbers
ls chuzhong_algebra/part2-polynomials
```

期望：grep 无输出；part1 7 文件；part2 7 文件。

- [ ] **Step 2: 术语一致性检查**

- 术语与 thinking-toolkit 一致（"换元"、"配方"、"结构识别"、"整体代入"等）
- LaTeX 符号统一

- [ ] **Step 3: 标记 AM1 完成**

在本计划末尾追加 `**AM1 完成于：YYYY-MM-DD**`，提交。

---
**AM1 完成于：2026-05-12**
