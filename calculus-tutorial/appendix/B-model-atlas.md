# 附录 B：微积分套路模型图集

> 覆盖微积分 Part 1–8（全 28 章）共 30 个核心套路模型。每个模型包含触发条件、核心思路、关键步骤、典型题与关联章节。配合附录 A 公式表和 thinking-toolkit 12 篇，可快速识别题型、套路和解法。

---

## Part 1–2 预备与极限（模型 1–4）

---

## 模型 1：ε-N / ε-δ 三步证法

**触发条件**

题面要求"用 ε-N 语言证明数列极限"或"用 ε-δ 语言证明函数极限"；考试要求严格证明而非直觉描述；或题目出现"对任意 ε>0 存在 N"之类的定量要求。

**核心思路**

极限的精确定义是整个分析学的基础。ε 是对方指定的"误差容限"，而你需要找到一个 N（或 δ），使得从此之后（或此范围内）误差永远不超过 ε。三步结构固定：设 ε → 反解临界量 → 验证。

**关键步骤**

1. 设任意 $\varepsilon>0$（不要忘记这是已知量）。
2. 分析 $|a_n-A|<\varepsilon$（或 $|f(x)-L|<\varepsilon$），用代数方法**反解出 $n>N(\varepsilon)$**（或 $|x-x_0|<\delta(\varepsilon)$）。
3. 写规范验证语句："当 $n>N$ 时，$|a_n-A|\leq\cdots<\varepsilon$。故 $\lim a_n=A$。"

**注意事项**

- $N$ 只能依赖 $\varepsilon$，不得依赖 $n$（否则逻辑循环）。
- $\delta$ 取 $\min\{1,\cdots\}$ 时，各分支均需验证。
- 否命题（证极限不为 $A$）：存在 $\varepsilon_0>0$，对任何 $N$，均可找到 $n>N$ 使 $|a_n-A|\geq\varepsilon_0$。

**典型题**

用 ε-N 语言证明 $\lim_{n\to\infty}\dfrac{n}{n+1}=1$。

> 设 $\varepsilon>0$。$\left|\dfrac{n}{n+1}-1\right|=\dfrac{1}{n+1}<\dfrac{1}{n}<\varepsilon$ 当且仅当 $n>1/\varepsilon$。取 $N=\lfloor1/\varepsilon\rfloor$，则 $n>N$ 时结论成立。

**关联章节**：Ch.4（数列极限）、Ch.5（函数极限）| **关联 Toolkit**：TK-01

---

## 模型 2：等价无穷小乘除替换 + 加减陷阱

**触发条件**

极限表达式中出现 $\sin x$、$\tan x$、$e^x-1$、$\ln(1+x)$、$(1+x)^\alpha-1$ 等形式与 $x$ 的乘除运算，且极限过程是 $x\to0$；或题目明确要求"利用等价无穷小"简化计算。

**核心思路**

等价无穷小替换是简化"零比零"型极限的最快武器，但它**只在乘除结构中安全**。加减运算中不能直接替换，因为相减两项等价无穷小的"差"可能是更高阶的量，用低阶近似时丢失关键信息。

**关键步骤**

1. 识别 $x\to0$ 时各因子的等价：$\sin x\sim x$，$\tan x\sim x$，$1-\cos x\sim x^2/2$，$e^x-1\sim x$，$\ln(1+x)\sim x$，$(1+x)^\alpha-1\sim\alpha x$，$\arcsin x\sim x$，$\arctan x\sim x$。
2. 若表达式是乘除形式，直接替换，约简，求极限。
3. 若表达式含加减（如 $\sin x-x$），立刻**放弃替换，改用 Taylor 展开**。

**加减陷阱示例**

$\lim_{x\to0}\dfrac{\sin x-\tan x}{x^3}$：分子中 $\sin x\sim x$ 和 $\tan x\sim x$，若各自替换则 $x-x=0$，答案错误。正确做法：$\sin x-\tan x=\left(x-\dfrac{x^3}{6}+o(x^3)\right)-\left(x+\dfrac{x^3}{3}+o(x^3)\right)=-\dfrac{x^3}{2}+o(x^3)$，极限为 $-\dfrac{1}{2}$。

**典型题**

$\lim_{x\to0}\dfrac{\arctan(3x)}{\sin(2x)}$。

> 分子 $\sim 3x$，分母 $\sim 2x$（乘除结构，安全替换），极限 $=3/2$。

**关联章节**：Ch.5（等价无穷小）| **关联 Toolkit**：TK-02

---

## 模型 3：两个重要极限

**触发条件**

极限式中出现 $\dfrac{\sin(\cdot)}{(\cdot)}$ 或 $\dfrac{(\cdot)}{\sin(\cdot)}$ 的结构；或出现 $\left(1+(\cdot)\right)^{1/(\cdot)}$ 的指数型结构；以及这两种形式的复合和变形（如 $\lim(1+1/n)^n$、$\lim(1+x)^{1/x}$）。

**核心思路**

两个重要极限是微积分中最基本的"非初等极限"，从幂函数或有理函数无法推导，需要单独记忆。第一个极限与三角函数的几何意义直接关联；第二个极限定义了自然常数 $e$。

**两个重要极限**

$$\lim_{x\to0}\frac{\sin x}{x}=1 \quad \text{（连同变形：} \lim_{x\to0}\frac{\tan x}{x}=1, \lim_{x\to0}\frac{\arcsin x}{x}=1\text{）}$$

$$\lim_{x\to\infty}\left(1+\frac{1}{x}\right)^x=e, \quad \lim_{x\to0}(1+x)^{1/x}=e$$

**第二极限的变形技巧**

对 $\lim(1+f(x))^{g(x)}$ 型，若 $f(x)\to0$ 且 $g(x)\to\infty$，判断 $f(x)\cdot g(x)$ 的极限：$\lim f\cdot g=\lambda$，则原极限为 $e^\lambda$。

**关键步骤**（第二极限）

1. 判断基底是否 $\to1$、指数是否 $\to\infty$（即 $1^\infty$ 型不定式）。
2. 令 $u=f(x)$，将 $(1+u)^{1/u}\cdot (ug(x))$ 拆分，计算 $\lim u\cdot g(x)$。
3. 或直接取对数：$g(x)\ln(1+f(x))\approx g(x)\cdot f(x)$ 当 $f(x)\to0$，再求极限。

**典型题**

$\lim_{n\to\infty}\left(1-\dfrac{2}{n}\right)^n$。

> 令 $u=-2/n$，则 $(1+u)^{1/u}=e$，指数为 $u\cdot n=(-2/n)\cdot n=-2$，极限为 $e^{-2}$。

**关联章节**：Ch.5（两个重要极限）| **关联 Toolkit**：TK-01，TK-02

---

## 模型 4：连续 / 一致连续辨析

**触发条件**

题目给出函数定义，要求证明连续性或一致连续性；或考题要求区分二者（如"在有限闭区间上连续的函数是否一致连续？"）；或函数在无穷区间上的行为讨论。

**核心思路**

连续性是"逐点"的局部概念：对每个固定点 $x_0$，$\delta$ 可以依赖 $x_0$ 和 $\varepsilon$。一致连续性是"全局"概念：找到的 $\delta$ 对区间上所有点均有效，不依赖具体的 $x_0$。

**对比速查**

| | 连续（逐点）| 一致连续（全局）|
|---|---|---|
| **定义** | $\forall\varepsilon>0,\forall x_0,\exists\delta(x_0,\varepsilon)>0$：$|x-x_0|<\delta\Rightarrow|f(x)-f(x_0)|<\varepsilon$ | $\forall\varepsilon>0,\exists\delta(\varepsilon)>0$：$|x-y|<\delta\Rightarrow|f(x)-f(y)|<\varepsilon$（对所有 $x,y$）|
| **关键差异** | $\delta$ 可随 $x_0$ 变化 | $\delta$ 对区间上所有点均有效 |
| **典型正例** | $f(x)=x^2$（在每点连续）| $f(x)=\sin x$（全局 Lipschitz）|
| **典型反例** | $f(x)=1/x$ 在 $(0,1)$ 连续但不一致连续 | — |

**Cantor 定理**：若 $f$ 在有界闭区间 $[a,b]$ 上连续，则 $f$ 在 $[a,b]$ 上一致连续。

**关联章节**：Ch.6（连续性）| **关联 Toolkit**：TK-01

---

## Part 3 微分学（模型 5–9）

---

## 模型 5：导数 6 大规则决策树

**触发条件**

题目给出函数表达式，要求求导；表达式是多种运算的复合（和差、乘除、复合、幂型、隐函数等）；需要在动笔之前确定用哪一条规则（或哪几条规则的组合）。

**核心思路**

求导是机械的，但必须先识别函数的结构。6 大规则形成一棵决策树：看到什么结构，走哪条规则。错误往往来自于"结构识别错误"而非规则本身。

**6 大规则决策树**

```
看到函数 f：
  ├── f = c（常数）→ 常数法则：f' = 0
  ├── f = u ± v → 和差法则：f' = u' ± v'
  ├── f = c·u → 数乘法则：f' = c·u'
  ├── f = u·v → 乘积法则：f' = u'v + uv'（"前导后不动 + 前不动后导"）
  ├── f = u/v → 商法则：f' = (u'v - uv')/v²（分子相减，分母平方）
  ├── f = g(h(x)) → 链式法则：f' = g'(h(x))·h'(x)（外层导数×内层导数）
  ├── f = u^v（u,v 均含 x）→ 对数求导法
  └── F(x,y)=0 → 隐函数求导：dy/dx = -F_x/F_y
```

**对数求导法**：对 $y=u^v$，两边取对数 $\ln y=v\ln u$，再对 $x$ 求导 $y'/y=v'\ln u+v\cdot u'/u$，还原 $y'=y\cdot(v'\ln u+v\cdot u'/u)$。

**典型题**

求 $y=x^{\sin x}$ 的导数。

> 对数求导：$\ln y=\sin x\cdot\ln x$，$y'/y=\cos x\cdot\ln x+\sin x/x$，$y'=x^{\sin x}(\cos x\ln x+\sin x/x)$。

**关联章节**：Ch.7（导数定义）、Ch.8（求导法则）| **关联 Toolkit**：TK-03

---

## 模型 6：隐函数 + 对数求导法

**触发条件**

函数以方程 $F(x,y)=0$ 的形式给出（无法显式解出 $y$）；或函数是多个因子的乘积（超过三项乘积、分子分母各含多因子）；或出现 $f^g$ 型幂函数（底数和指数都含变量）。

**核心思路**

隐函数求导：把方程两边视为 $x$ 的函数，对 $x$ 求导（$y$ 视为 $x$ 的函数，用链式），然后解出 $y'$。对数求导法：先取对数（将乘除变为加减，幂变为乘），再求导，适合乘积幂型复杂结构。

**隐函数求导步骤**

1. 方程 $F(x,y)=0$ 两边对 $x$ 求导，记住 $y$ 对 $x$ 求导要加链式（如 $d(y^2)/dx=2y\cdot y'$）。
2. 从等式中解出 $y'$；公式形式：$y'=-F_x/F_y$（$F_y\neq0$）。
3. 若需二阶导，对 $y'$ 再对 $x$ 求导一次（注意 $y'$ 中含 $y$，再次用链式）。

**对数求导法步骤**

1. 设 $y=f(x)$，两边取对数：$\ln y=\ln f(x)$（利用对数把乘法变加法、幂变乘法）。
2. 两边对 $x$ 求导：$y'/y=[\ln f(x)]'$。
3. 解出 $y'=y\cdot[\ln f(x)]'$，代入 $y=f(x)$。

**典型题**

设 $x^2+y^2+xy=3$，求 $y'$。

> 两边对 $x$ 求导：$2x+2y\cdot y'+y+xy'=0$，整理 $(2y+x)y'=-(2x+y)$，$y'=-(2x+y)/(2y+x)$。

**关联章节**：Ch.8（隐函数求导）| **关联 Toolkit**：TK-03

---

## 模型 7：单调极值标准 4 步

**触发条件**

题目要求"求函数 $f(x)$ 的单调区间"、"求极值（极大值、极小值）"；或需要确定某参数使函数具有特定的单调性；或需要证明某函数在区间上是单调的。

**核心思路**

单调性由一阶导数的符号决定：$f'>0$ 递增，$f'<0$ 递减。极值点是一阶导数变号的点（必要条件是 $f'=0$ 或 $f'$ 不存在）。标准 4 步骤机械化：求导 → 找零点 → 列符号表 → 读结论。

**标准 4 步**

1. 求 $f'(x)$，化简到最简形式（分子分母均因式分解）。
2. 解 $f'(x)=0$ 和 $f'(x)$ 不存在的点（候选极值点），列出所有分界点。
3. 作符号表：以分界点为分隔，列各区间上 $f'(x)$ 的符号。
4. 读结论：$f'$ 从正变负 → 极大值；从负变正 → 极小值；不变号 → 无极值（仅为单调趋势的分界点）。

**二阶导判别（当一阶导符号表不方便时）**

驻点 $x_0$（$f'(x_0)=0$）：若 $f''(x_0)>0$ → 极小；若 $f''(x_0)<0$ → 极大；若 $f''(x_0)=0$ → 失效，用一阶导号表法。

**典型题**

求 $f(x)=x^3-3x$ 的单调区间和极值。

> $f'=3x^2-3=3(x-1)(x+1)$；$x=-1$ 和 $x=1$ 是分界点；$f'$：$(-\infty,-1)$ 正，$(-1,1)$ 负，$(1,+\infty)$ 正；故 $f$ 在 $(-\infty,-1)$ 和 $(1,+\infty)$ 上递增，在 $(-1,1)$ 上递减；$x=-1$ 为极大值点 $f(-1)=2$，$x=1$ 为极小值点 $f(1)=-2$。

**关联章节**：Ch.9（导数应用）| **关联 Toolkit**：TK-03，TK-10

---

## 模型 8：L'Hôpital 法则 + Taylor 取代

**触发条件**

极限属于不定式：$0/0$，$\infty/\infty$，$0\cdot\infty$，$\infty-\infty$，$1^\infty$，$0^0$，$\infty^0$；特别是当等价无穷小替换不适用（含加减运算）时，用 L'Hôpital 或 Taylor。

**核心思路**

L'Hôpital 法则：在满足条件的情况下，$\lim f/g = \lim f'/g'$（对分子分母各求一次导数）。Taylor 展开：将函数展开到需要的阶次，做代数约消。两种方法往往可互换，但 Taylor 在含复合函数时通常更高效（避免反复求导）。

**L'Hôpital 使用条件**

1. 必须是 $0/0$ 或 $\infty/\infty$ 型（其余不定式先化为这两种）。
2. 化型方式：$f\cdot g=f/(1/g)$；$f-g=\cdots/\cdots$；$1^\infty$ 型取对数后变为 $0/0$。
3. 若求导后仍是不定式，可再次应用（但每次均需验证条件）。
4. L'Hôpital 失效情形：极限不存在时（如 $\lim_{x\to\infty}\sin x/x$ 中分子不趋零但分母趋无穷）。

**Taylor 取代策略**

展开到"第一个不消失的幂次"：若分母是 $x^n$ 型，将分子展开到 $x^n$ 项即可约消。含复合函数时，先展内层（如 $e^{x^2}=1+x^2+x^4/2+\cdots$），再做整体计算。

**典型题**

$\lim_{x\to0}\dfrac{e^x-1-x}{x^2}$（用两种方法）。

> **L'Hôpital**：两次求导后 $\lim\dfrac{e^x}{2}=1/2$。
> **Taylor**：$e^x=1+x+x^2/2+o(x^2)$，分子 $=x^2/2+o(x^2)$，极限 $=1/2$。

**关联章节**：Ch.9（L'Hôpital）、Ch.10（Taylor 展开）| **关联 Toolkit**：TK-06

---

## 模型 9：Taylor 6 大 Maclaurin 速查

**触发条件**

需要将函数近似为多项式（求极限、估计误差、数值计算、函数近似）；或需要展开复合函数（如 $\sin(x^2)$、$e^{-x^2}$）；或幂级数展开时需要起点公式。

**核心思路**

6 大 Maclaurin 展开是所有 Taylor 应用的"基本弹药"，无条件记忆。复合型展开靠"变量替换"：将已知展开式中的 $x$ 替换为复合的内层函数，收敛域相应调整。

**6 大展开速查**（$x\to0$ 时）

| 函数 | 展开式（关键前几项） |
|:---:|:---|
| $e^x$ | $1+x+\frac{x^2}{2}+\frac{x^3}{6}+\cdots$ |
| $\sin x$ | $x-\frac{x^3}{6}+\frac{x^5}{120}-\cdots$ |
| $\cos x$ | $1-\frac{x^2}{2}+\frac{x^4}{24}-\cdots$ |
| $\ln(1+x)$ | $x-\frac{x^2}{2}+\frac{x^3}{3}-\cdots$，$x\in(-1,1]$ |
| $(1+x)^\alpha$ | $1+\alpha x+\frac{\alpha(\alpha-1)}{2}x^2+\cdots$，$|x|<1$ |
| $\arctan x$ | $x-\frac{x^3}{3}+\frac{x^5}{5}-\cdots$，$|x|\leq1$ |

**复合展开技巧**

- $e^{-x^2}$：将 $e^x$ 展开式中 $x$ 换为 $-x^2$，得 $1-x^2+x^4/2-\cdots$
- $\sin(x^2)$：$x^2-x^6/6+\cdots$
- $\ln(1+x^2)$：$x^2-x^4/2+x^6/3-\cdots$（$|x|<1$）

**Lagrange 余项**：$R_n(x)=\dfrac{f^{(n+1)}(\xi)}{(n+1)!}x^{n+1}$，用于数值误差上界估计。

**典型题**

求 $\lim_{x\to0}\dfrac{\cos x-e^{-x^2/2}}{x^4}$。

> $\cos x=1-x^2/2+x^4/24-\cdots$；$e^{-x^2/2}=1-x^2/2+x^4/8-\cdots$；分子 $=x^4/24-x^4/8+o(x^4)=-x^4/12+o(x^4)$；极限 $=-1/12$。

**关联章节**：Ch.10（Taylor 展开）| **关联 Toolkit**：TK-06

---

## Part 4 积分学（模型 10–15）

---

## 模型 10：不定积分基本公式

**触发条件**

求不定积分时，被积函数是初等函数的简单组合（无需换元或分部积分）；或作为换元后的收尾步骤（换元后的积分可以直接套公式）。

**核心思路**

不定积分基本公式是微分公式的"逆读"，必须无条件记忆。关键是记忆时要双向检验：对积分结果求导，恢复被积函数。

**核心公式速查**

| 被积函数 | 积分结果 | 被积函数 | 积分结果 |
|:---:|:---:|:---:|:---:|
| $x^n$（$n\neq-1$）| $\dfrac{x^{n+1}}{n+1}+C$ | $\dfrac{1}{x}$ | $\ln|x|+C$ |
| $e^x$ | $e^x+C$ | $a^x$ | $\dfrac{a^x}{\ln a}+C$ |
| $\sin x$ | $-\cos x+C$ | $\cos x$ | $\sin x+C$ |
| $\sec^2x$ | $\tan x+C$ | $\csc^2x$ | $-\cot x+C$ |
| $\dfrac{1}{\sqrt{1-x^2}}$ | $\arcsin x+C$ | $\dfrac{1}{1+x^2}$ | $\arctan x+C$ |
| $\dfrac{1}{a^2+x^2}$ | $\dfrac{1}{a}\arctan\dfrac{x}{a}+C$ | $\dfrac{1}{\sqrt{a^2-x^2}}$ | $\arcsin\dfrac{x}{a}+C$ |
| $\dfrac{1}{\sqrt{x^2\pm a^2}}$ | $\ln\|x+\sqrt{x^2\pm a^2}\|+C$ | $\dfrac{1}{a^2-x^2}$ | $\dfrac{1}{2a}\ln\left\|\dfrac{a+x}{a-x}\right\|+C$ |

**凑微分技巧**：对 $\int f(ax+b)\,dx$，令 $u=ax+b$，结果多一个 $1/a$ 的因子。

**典型题**

$\displaystyle\int\frac{1}{1+4x^2}\,dx$。

> 改写为 $\dfrac{1}{4}\int\dfrac{1}{(1/2)^2+x^2}\,dx$，套公式得 $\dfrac{1}{4}\cdot\dfrac{1}{1/2}\arctan(2x)+C=\dfrac{1}{2}\arctan(2x)+C$。

**关联章节**：Ch.11（不定积分）| **关联 Toolkit**：TK-04

---

## 模型 11：定积分牛顿-莱布尼茨

**触发条件**

需要计算定积分 $\int_a^b f(x)\,dx$，且被积函数的原函数可以显式求出；或题目以定积分形式给出某物理量（面积、位移、功等）要求计算数值。

**核心思路**

牛顿-莱布尼茨公式（微积分基本定理）将积分（面积计算）与求导（反过程）连接：$\int_a^b f(x)\,dx=F(b)-F(a)$，其中 $F'=f$。这是微积分最核心的定理，将原本需要"极限求和"的计算转化为代数运算。

**关键步骤**

1. 求被积函数 $f(x)$ 的一个原函数 $F(x)$（不必加常数 $C$，因为定积分时 $C$ 会消掉）。
2. 计算 $F(b)-F(a)$（写作 $[F(x)]_a^b$ 或 $F(x)\Big|_a^b$）。
3. 若被积函数在 $[a,b]$ 上有间断点（特别是无界点），该积分是反常积分，需单独处理。

**对称区间技巧**

若 $f$ 为偶函数：$\int_{-a}^a f\,dx=2\int_0^a f\,dx$；若 $f$ 为奇函数：$\int_{-a}^a f\,dx=0$。

**积分中值定理**：$\exists\xi\in(a,b)$，使 $\int_a^b f(x)\,dx=f(\xi)(b-a)$（$f$ 在 $[a,b]$ 上连续）。

**典型题**

$\displaystyle\int_0^1(2x+e^x)\,dx$。

> $=[x^2+e^x]_0^1=(1+e)-(0+1)=e$。

**关联章节**：Ch.12（定积分）| **关联 Toolkit**：TK-04

---

## 模型 12：LIATE 分部积分

**触发条件**

被积函数是**两个不同类型函数的乘积**，且无法直接套基本公式或换元化简；或积分中出现 $\ln x$、$\arctan x$、$\arcsin x$（这些只能选为 $u$，因为它们求积分很难但求导后变为代数）。

**核心思路**

分部积分公式 $\int u\,dv=uv-\int v\,du$ 将原积分转化为（希望更简单的）新积分。选 $u$ 的核心原则：$u$ 求导后应变简单，$dv$ 积分后不应变复杂。LIATE 给出了优先级顺序，是实际操作中最可靠的"选 $u$ 规则"。

**LIATE 优先级速查**：对数(L) > 反三角(I) > 代数/多项式(A) > 三角(T) > 指数(E)

**循环积分处理**

$e^x\sin x$ 型：分部两次后原积分 $I$ 再次出现，设 $I=\int e^x\sin x\,dx$，移项解方程 $2I=e^x(\sin x-\cos x)$，得 $I=e^x(\sin x-\cos x)/2+C$。

**典型题**

$\displaystyle\int x\ln x\,dx$（L > A，$u=\ln x$，$dv=x\,dx$）。

> $v=x^2/2$，$\int x\ln x\,dx=\dfrac{x^2\ln x}{2}-\int\dfrac{x^2}{2}\cdot\dfrac{1}{x}\,dx=\dfrac{x^2\ln x}{2}-\dfrac{x^2}{4}+C$。

**关联章节**：Ch.13（积分技巧）| **关联 Toolkit**：TK-04

---

## 模型 13：三角换元 3 种情形

**触发条件**

被积函数含 $\sqrt{a^2-x^2}$、$\sqrt{a^2+x^2}$ 或 $\sqrt{x^2-a^2}$ 等根号形式；或含相同形式的倒数（如 $1/\sqrt{a^2-x^2}$）；或被积函数的化简需要去掉根号。

**核心思路**

三角换元利用三角恒等式去掉根号：$\sin^2+\cos^2=1$ 消去 $\sqrt{a^2-x^2}$ 型；$1+\tan^2=\sec^2$ 消去 $\sqrt{a^2+x^2}$ 型；$\sec^2-1=\tan^2$ 消去 $\sqrt{x^2-a^2}$ 型。换元后得三角有理式，再积分或用其他技巧处理。

**3 种情形速查**

| 根号形式 | 换元 | 范围 | 关键化简 |
|:---:|:---:|:---:|:---|
| $\sqrt{a^2-x^2}$ | $x=a\sin t$ | $t\in[-\pi/2,\pi/2]$ | $\sqrt{a^2-x^2}=a\cos t$（$\cos t\geq0$）|
| $\sqrt{a^2+x^2}$ | $x=a\tan t$ | $t\in(-\pi/2,\pi/2)$ | $\sqrt{a^2+x^2}=a\sec t$（$\sec t>0$）|
| $\sqrt{x^2-a^2}$ | $x=a\sec t$ | $t\in[0,\pi/2)\cup(\pi/2,\pi]$ | $\sqrt{x^2-a^2}=a\|\tan t\|$（注意绝对值）|

**回代步骤**：换元后积分得到以 $t$ 表示的结果，需用 $\sin t=x/a$、$\cos t=\sqrt{1-x^2/a^2}$（等）将 $t$ 还原为 $x$；作辅助直角三角形有助于快速读出各三角函数的值。

**典型题**

$\displaystyle\int\frac{dx}{\sqrt{4-x^2}}$。

> 令 $x=2\sin t$，$dx=2\cos t\,dt$，$\sqrt{4-x^2}=2\cos t$；积分变为 $\int\dfrac{2\cos t\,dt}{2\cos t}=\int dt=t+C=\arcsin\dfrac{x}{2}+C$。

**关联章节**：Ch.13（积分技巧）| **关联 Toolkit**：TK-04

---

## 模型 14：部分分式分解

**触发条件**

被积函数是有理函数 $P(x)/Q(x)$（分子分母均为多项式），且 $\deg P<\deg Q$；题目要求对有理函数积分，或在解 ODE 时需要对有理函数积分。

**核心思路**

任何真分式都可以分解为若干简单分式之和（部分分式分解定理）。分解的关键是 $Q(x)$ 的因子分解：每个一次因子 $(x-a)^k$ 对应 $k$ 个待定系数，每个不可约二次因子 $(x^2+px+q)^k$ 对应 $k$ 个分子为一次式的项。

**分解步骤**

1. 确认 $\deg P<\deg Q$（若否，先做多项式长除法）。
2. 将 $Q(x)$ 分解为不可约因子的乘积（实系数范围内：一次因子和不可约二次因子）。
3. 对每种因子写待定分式（见下表），令等式恒成立，比较系数求出待定常数。
4. 对每个简单分式积分（均为基本公式范围）。

**分解形式速查**

| $Q(x)$ 的因子 | 对应部分分式 |
|:---:|:---|
| $(x-a)$（一次单因子）| $\dfrac{A}{x-a}$ |
| $(x-a)^k$（一次重因子）| $\dfrac{A_1}{x-a}+\dfrac{A_2}{(x-a)^2}+\cdots+\dfrac{A_k}{(x-a)^k}$ |
| $(x^2+px+q)$（不可约二次单因子）| $\dfrac{Ax+B}{x^2+px+q}$ |
| $(x^2+px+q)^k$ | $\dfrac{A_1x+B_1}{x^2+px+q}+\cdots+\dfrac{A_kx+B_k}{(x^2+px+q)^k}$ |

**典型题**

$\displaystyle\int\frac{dx}{x^2-1}=\int\frac{dx}{(x-1)(x+1)}$，分解为 $\dfrac{1/2}{x-1}-\dfrac{1/2}{x+1}$，积分得 $\dfrac{1}{2}\ln|x-1|-\dfrac{1}{2}\ln|x+1|+C=\dfrac{1}{2}\ln\left|\dfrac{x-1}{x+1}\right|+C$。

**关联章节**：Ch.13（积分技巧）| **关联 Toolkit**：TK-04

---

## 模型 15：反常积分 p-判别

**触发条件**

积分上限或下限为 $\pm\infty$（无穷限积分）；或被积函数在积分区间的某端点处无界（瑕积分）；需要判断反常积分的收敛性或计算其值。

**核心思路**

反常积分通过极限来定义：$\int_a^{+\infty}f\,dx=\lim_{b\to+\infty}\int_a^b f\,dx$；瑕积分 $\int_a^b f\,dx$（$f$ 在 $a$ 处无界）$=\lim_{\varepsilon\to0^+}\int_{a+\varepsilon}^b f\,dx$。$p$-判别法是最常用的比较收敛判别法。

**$p$-判别法速查**

$$\int_1^{+\infty}\frac{dx}{x^p}：\text{收敛} \Leftrightarrow p>1 \quad \int_0^1\frac{dx}{x^p}：\text{收敛} \Leftrightarrow p<1$$

**比较判别法**（反常积分版）：若 $0\leq f(x)\leq g(x)$，则 $\int g$ 收敛 $\Rightarrow$ $\int f$ 收敛；$\int f$ 发散 $\Rightarrow$ $\int g$ 发散。

**极限比较**：$\lim_{x\to+\infty}f(x)/g(x)=L\in(0,+\infty)$ $\Rightarrow$ $\int_a^{+\infty}f$ 与 $\int_a^{+\infty}g$ 同敛散。

**典型题**

判断 $\displaystyle\int_1^{+\infty}\frac{\sin^2x}{x^2}\,dx$ 的收敛性。

> $0\leq\sin^2x/x^2\leq1/x^2$，$\int_1^{+\infty}1/x^2\,dx=1$ 收敛（$p=2>1$），由比较判别法，原积分收敛。

**关联章节**：Ch.14（反常积分）| **关联 Toolkit**：TK-05

---

## Part 5 级数（模型 16–18）

---

## 模型 16：级数判敛决策树

**触发条件**

题目给出一个数项级数 $\sum a_n$，要求判断其收敛性（绝对收敛 / 条件收敛 / 发散）；或要求找出使级数收敛的参数范围。

**核心思路**

判敛没有万能公式，但有一棵固定的决策树：按照一定顺序检验各判别法，从最简单（必要条件）开始，逐步升级到更复杂的工具。见附录 A 第 17 节的完整决策树。

**关键步骤**

0. 先验必要条件：$a_n\not\to0$ 直接发散。
1. 识别类型（正项 / 交错 / 任意项）。
2. 正项级数：比值法（含 $n!$ 或 $a^n$ 时）→ 根值法（含 $(\cdot)^n$ 时）→ 比较法（$p$-级数参照）→ 积分判别（$a_n=f(n)$ 单调减）。
3. 交错级数：Leibniz 判别（$b_n$ 单调递减趋零）。
4. 绝对收敛 $\Rightarrow$ 收敛（一般性结论）。

**常见结论速查**

- $\sum n^{-p}$：$p>1$ 收敛，$p\leq1$ 发散。
- $\sum n!/(n^n)$：收敛（比值 $\to 1/e$）。
- $\sum(-1)^n/n$：条件收敛（Leibniz，但 $\sum 1/n$ 发散）。
- $\sum 1/(n\ln n)$：发散（积分判别）。

**典型题**

判断 $\displaystyle\sum_{n=1}^\infty\frac{n^2}{2^n}$ 的收敛性。

> 比值法：$\rho=\lim\dfrac{(n+1)^2/2^{n+1}}{n^2/2^n}=\lim\dfrac{(n+1)^2}{2n^2}=\dfrac{1}{2}<1$，收敛。

**关联章节**：Ch.15（数项级数）| **关联 Toolkit**：TK-05

---

## 模型 17：幂级数收敛半径 + 端点

**触发条件**

题目给出幂级数 $\sum a_n x^n$ 或 $\sum a_n(x-x_0)^n$，要求求收敛半径、收敛区间（注意端点的收敛性需单独检验）；或求 $f(x)$ 展开成幂级数后的收敛域。

**核心思路**

幂级数的收敛性由系数 $\{a_n\}$ 的"增长速度"决定，比值法或根值法给出收敛半径 $R$：在 $|x-x_0|<R$ 内绝对收敛，在 $|x-x_0|>R$ 外发散，在 $x=x_0\pm R$ 处需单独用数项级数判别法检验。

**收敛半径公式**

$$R=\frac{1}{\limsup_{n\to\infty}\sqrt[n]{|a_n|}} \quad \text{（根值公式，Cauchy-Hadamard）}$$

等价地，若 $\lim|a_{n+1}/a_n|=\rho$（$\neq0$），则 $R=1/\rho$。

**完整步骤**

1. 用比值/根值公式求 $R$（收敛半径）。
2. 写出开区间 $(x_0-R, x_0+R)$（绝对收敛）。
3. 代入 $x=x_0+R$ 和 $x=x_0-R$，分别用数项级数判别法检验（可能收敛可能发散）。
4. 写出最终收敛域（开/闭/半开区间）。

**典型题**

求 $\displaystyle\sum_{n=1}^\infty\frac{x^n}{n}$ 的收敛域。

> $\rho=\lim n/(n+1)=1$，$R=1$；$x=1$：$\sum 1/n$ 发散；$x=-1$：$\sum(-1)^n/n$ 条件收敛（Leibniz）。收敛域为 $[-1,1)$。

**关联章节**：Ch.16（幂级数）| **关联 Toolkit**：TK-05，TK-06

---

## 模型 18：Fourier 级数展开 + Dirichlet 收敛

**触发条件**

题目要求将 $2\pi$ 或 $2l$ 周期函数展开为 Fourier 级数；或给出分段函数要求求其 Fourier 展开，并说明收敛的情况；或求特殊级数的和（如 $\sum 1/n^2$）。

**核心思路**

Fourier 展开将周期函数分解为正弦、余弦的叠加，系数由正交性公式确定。Dirichlet 收敛定理告诉我们 Fourier 级数在何处收敛以及收敛到什么值。

**Fourier 系数公式**（以 $2\pi$ 为周期）

$$a_0=\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\,dx, \quad a_n=\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\cos(nx)\,dx, \quad b_n=\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\sin(nx)\,dx$$

$$f(x)\sim\frac{a_0}{2}+\sum_{n=1}^{\infty}(a_n\cos nx+b_n\sin nx)$$

**奇偶化简**

- $f$ 为偶函数：$b_n=0$，只有余弦项（偶展开）；$a_n=\dfrac{2}{\pi}\int_0^\pi f(x)\cos(nx)\,dx$。
- $f$ 为奇函数：$a_n=0$，只有正弦项（奇展开）；$b_n=\dfrac{2}{\pi}\int_0^\pi f(x)\sin(nx)\,dx$。

**Dirichlet 收敛定理**：若 $f$ 满足 Dirichlet 条件（逐段单调、逐段连续），则 Fourier 级数在连续点处收敛到 $f(x)$，在间断点 $x_0$ 处收敛到 $\dfrac{f(x_0^+)+f(x_0^-)}{2}$（左右极限的平均）。

**典型题**

$f(x)=x$（$-\pi<x<\pi$），求 Fourier 展开。

> $a_n=0$（$f$ 为奇函数）；$b_n=\dfrac{2}{\pi}\int_0^\pi x\sin(nx)\,dx=\dfrac{2(-1)^{n+1}}{n}$；$f(x)=\displaystyle\sum_{n=1}^\infty\dfrac{2(-1)^{n+1}}{n}\sin(nx)$；在 $x=\pm\pi$ 处收敛到 $0=\dfrac{\pi+(-\pi)}{2}$（左右极限平均）。

**关联章节**：Ch.17（Fourier 级数）| **关联 Toolkit**：TK-05，TK-06

---

## Part 6 多元微积分（模型 19–24）

---

## 模型 19：多元链式 + 梯度 / Jacobian

**触发条件**

题目涉及复合多元函数 $z=f(u,v)$，其中 $u,v$ 均依赖于 $x,y$；需要求偏导数 $\partial z/\partial x$ 或 $\partial z/\partial y$；或需要求梯度向量；或反向传播中需要计算损失对参数的梯度。

**核心思路**

多元链式：每条从输出到目标变量的路径，沿路相乘，所有路径相加。Jacobian 矩阵是多元链式的矩阵形式，是深度学习反向传播的理论基础。梯度向量指向函数值增加最快的方向，负梯度方向是梯度下降的基础。

**关键公式**（见附录 A 第 18 节）

- 链式基本形式：$\partial z/\partial x_j=\sum_i(\partial z/\partial u_i)(\partial u_i/\partial x_j)$
- 梯度：$\nabla f=\left(f_{x_1},\ldots,f_{x_n}\right)^T$
- 方向导数：$\partial f/\partial\mathbf{l}=\nabla f\cdot\mathbf{e}_l=|\nabla f|\cos\theta$（$\theta$ 为梯度与方向 $\mathbf{l}$ 的夹角）
- Jacobian 矩阵：$(J_f)_{ij}=\partial f_i/\partial x_j$；多元链式 $\Leftrightarrow$ Jacobian 相乘

**典型题**

设 $z=e^{u+v}$，$u=x^2$，$v=\sin y$，求 $\partial z/\partial x$。

> $\partial z/\partial x=(\partial z/\partial u)(\partial u/\partial x)+(\partial z/\partial v)(\partial v/\partial x)=e^{u+v}\cdot2x+e^{u+v}\cdot0=2xe^{x^2+\sin y}$。

**关联章节**：Ch.18（偏导数）| **关联 Toolkit**：TK-07

---

## 模型 20：二重积分极坐标 vs 直角

**触发条件**

被积函数含 $x^2+y^2$、$x/y$、$y/x$ 或其根号形式；积分区域是圆盘、圆环、扇形、圆锥截面；或直角坐标下积分顺序交换后仍然复杂，需要换坐标系。

**核心思路**

坐标系的选择决定计算难度。圆形区域 + $x^2+y^2$ 型被积函数，是极坐标的最强触发信号。换元后面积元 $dA=r\,dr\,d\theta$（不要忘记因子 $r$！）。

**极坐标换元**

$$x=r\cos\theta,\quad y=r\sin\theta,\quad dA=r\,dr\,d\theta$$

**典型积分限**

| 区域 | 极坐标积分限 |
|:---|:---|
| 圆盘 $x^2+y^2\leq R^2$ | $0\leq r\leq R$，$0\leq\theta\leq2\pi$ |
| 上半圆盘 | $0\leq r\leq R$，$0\leq\theta\leq\pi$ |
| 圆环 $a^2\leq x^2+y^2\leq b^2$ | $a\leq r\leq b$，$0\leq\theta\leq2\pi$ |
| 第一象限圆盘 | $0\leq r\leq R$，$0\leq\theta\leq\pi/2$ |

**交换积分次序**：先画积分区域 $D$，重新确定 X 型或 Y 型表示，再写出积分限。

**典型题**

$\displaystyle\iint_D\sqrt{x^2+y^2}\,dA$，$D:x^2+y^2\leq4$。

> 极坐标：$\displaystyle\int_0^{2\pi}\!\!\int_0^2 r\cdot r\,dr\,d\theta=2\pi\cdot\left[\frac{r^3}{3}\right]_0^2=\frac{16\pi}{3}$。

**关联章节**：Ch.19（重积分）| **关联 Toolkit**：TK-08

---

## 模型 21：三重积分球 / 柱坐标

**触发条件**

积分区域是球体、半球、球壳，被积函数含 $x^2+y^2+z^2$（球坐标）；或积分区域是圆柱体、圆锥，被积函数关于 $z$ 轴旋转对称（柱坐标）；直角坐标下计算极度复杂时考虑变换。

**核心思路**

三重积分的坐标选择策略：球体/球壳 → 球坐标；圆柱/圆锥/旋转体 → 柱坐标；长方体/一般多面体 → 直角坐标。换元后体积元包含 Jacobian 因子，不可遗漏。

**柱坐标**：$x=r\cos\theta$，$y=r\sin\theta$，$z=z$，$dV=r\,dr\,d\theta\,dz$

**球坐标**：$x=\rho\sin\varphi\cos\theta$，$y=\rho\sin\varphi\sin\theta$，$z=\rho\cos\varphi$，$dV=\rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$

（$\rho\geq0$，$\varphi\in[0,\pi]$，$\theta\in[0,2\pi)$）

**典型题**

$\displaystyle\iiint_V z\,dV$，$V:x^2+y^2+z^2\leq R^2$，$z\geq0$（上半球）。

> 球坐标：$z=\rho\cos\varphi$；积分 $\displaystyle\int_0^{2\pi}\!\!\int_0^{\pi/2}\!\!\int_0^R\rho\cos\varphi\cdot\rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta=2\pi\cdot\dfrac{R^4}{4}\cdot\dfrac{1}{2}=\dfrac{\pi R^4}{4}$。

**关联章节**：Ch.19（重积分）| **关联 Toolkit**：TK-08

---

## 模型 22：第一型 vs 第二型曲线积分

**触发条件**

题目给出曲线 $C$ 上的积分，需要区分"关于弧长的积分"（第一型，结果是标量）和"关于坐标的积分"（第二型，结果可含方向）；或题目给出向量场 $\mathbf{F}$ 沿曲线的功（即第二型曲线积分）。

**核心思路**

第一型对弧长积分 $\int_C f\,ds$：$ds=\sqrt{x'^2+y'^2}dt$（或 $\sqrt{1+y'^2}dx$），方向无关。第二型对坐标积分 $\int_C P\,dx+Q\,dy$：方向有关（反向则变号），物理意义是向量场做的功。

**对比速查**

| | 第一型（弧长积分）| 第二型（坐标积分）|
|---|---|---|
| **形式** | $\int_C f(x,y)\,ds$ | $\int_C P\,dx+Q\,dy$ |
| **参数化**（$x=x(t),y=y(t)$）| $\int_\alpha^\beta f(x(t),y(t))\sqrt{x'^2+y'^2}\,dt$ | $\int_\alpha^\beta\left[P\cdot x'(t)+Q\cdot y'(t)\right]dt$ |
| **方向性** | 无关（始终非负）| 有关（反向变号）|
| **物理意义** | 密度线的质量 / 函数值在曲线上的平均 | 向量场做的功 |

**典型题**

$\displaystyle\int_C(x+y)\,ds$，$C$ 是从 $(0,0)$ 到 $(1,1)$ 的线段。

> 参数化 $x=t,y=t$（$t\in[0,1]$），$ds=\sqrt{1^2+1^2}dt=\sqrt{2}dt$；积分 $=\sqrt{2}\int_0^1 2t\,dt=\sqrt{2}$。

**关联章节**：Ch.20（曲线积分）| **关联 Toolkit**：TK-07

---

## 模型 23：通量与曲面积分

**触发条件**

题目给出向量场 $\mathbf{F}=(P,Q,R)$ 和曲面 $S$，要求计算"通量"（向量场穿过曲面的总量）；或要求对曲面面积计算标量函数的积分（第一型曲面积分）；或题目中涉及散度与 Gauss 公式。

**核心思路**

第一型曲面积分 $\iint_S f\,dS$（关于面积元，方向无关）；第二型曲面积分（通量）$\iint_S\mathbf{F}\cdot d\mathbf{S}=\iint_S P\,dydz+Q\,dzdx+R\,dxdy$（方向有关，法向量指向规定的一侧）。

**面积元公式**（曲面 $z=z(x,y)$）：

$$dS=\sqrt{1+z_x^2+z_y^2}\,dA, \quad d\mathbf{S}=(-z_x,-z_y,1)\,dA\text{（法向量朝上）}$$

**典型题**

计算 $\iint_S\mathbf{F}\cdot d\mathbf{S}$，$\mathbf{F}=(x,y,z)$，$S$ 是单位球面（朝外法向）。

> 由 Gauss 公式：$\oiint_S\mathbf{F}\cdot d\mathbf{S}=\iiint_V\nabla\cdot\mathbf{F}\,dV=\iiint_V3\,dV=3\cdot\dfrac{4\pi}{3}=4\pi$。

**关联章节**：Ch.21（曲面积分）| **关联 Toolkit**：TK-08

---

## 模型 24：Green / Stokes / Gauss 三大定理

**触发条件**

计算曲线积分 $\oint_L P\,dx+Q\,dy$ 时积分路径是封闭曲线（Green）；计算空间曲线积分时曲线是某曲面的边界（Stokes）；计算向量场穿过封闭曲面的通量（Gauss）；或题目中出现"利用微积分基本定理"简化计算的暗示。

**核心思路**

三大定理是微积分基本定理的高维推广，都是"把边界上的积分转化为内部的积分"（或反过来）。掌握三大定理可以把困难的边界积分转化为更简单的体积/面积分（或反之）。

**三大定理速查**

| 定理 | 公式 | 维度 | 条件 |
|:---:|:---|:---:|:---|
| **Green**（平面）| $\oint_L P\,dx+Q\,dy=\iint_D\left(\dfrac{\partial Q}{\partial x}-\dfrac{\partial P}{\partial y}\right)dA$ | 2D | $L$ 为 $D$ 的正向边界（逆时针）|
| **Stokes**（空间）| $\oint_C\mathbf{F}\cdot d\mathbf{r}=\iint_S(\nabla\times\mathbf{F})\cdot d\mathbf{S}$ | 3D | $C$ 为 $S$ 的边界，方向相容 |
| **Gauss**（散度）| $\oiint_S\mathbf{F}\cdot d\mathbf{S}=\iiint_V(\nabla\cdot\mathbf{F})\,dV$ | 3D | $S$ 为 $V$ 的封闭外侧曲面 |

**路径无关条件**（平面）：$\dfrac{\partial Q}{\partial x}=\dfrac{\partial P}{\partial y}$ 在单连通区域 $D$ 上成立 $\Rightarrow$ 曲线积分与路径无关 $\Rightarrow$ $P\,dx+Q\,dy$ 是全微分（存在势函数）。

**典型题**

$\oint_L(2x-y)\,dx+(x+3y)\,dy$，$L$ 是单位圆正向。

> Green：$Q_x-P_y=1-(-1)=2$；积分 $=\iint_D2\,dA=2\cdot\pi\cdot1^2=2\pi$。

**关联章节**：Ch.22（向量微积分）| **关联 Toolkit**：TK-07，TK-08

---

## Part 7 常微分方程（模型 25–26）

---

## 模型 25：一阶 ODE 5 类识别

**触发条件**

题目给出一阶常微分方程 $y'=F(x,y)$ 或 $P\,dx+Q\,dy=0$，要求求其通解；或给出初始条件要求求特解。

**核心思路**

ODE 求解的关键是"类型识别先于解法"：类型认定错误，则一切方法徒劳。5 类识别有固定的优先级顺序，沿决策树走一遍必定落到某类。

**5 类识别决策树**

```
Step 1：能否分离变量？（能写成 y'=f(x)g(y)？）
  是 → 类型 1（可分离变量）：dy/g(y) = f(x)dx，两边积分

Step 2：能否写成 y'=φ(y/x)？
  是 → 类型 2（齐次方程）：令 u=y/x，化为可分离

Step 3：能否写成 y'+p(x)y=q(x)？（y 一次）
  是 → 类型 3（一阶线性）：积分因子 e^{∫p dx}

Step 4：能否写成 y'+p(x)y=q(x)y^n？（n≠0,1）
  是 → 类型 4（Bernoulli）：令 v=y^{1-n}，化为线性

Step 5：验证 P_y=Q_x？
  是 → 类型 5（恰当方程）：求势函数 u，u=C 为通解
  否 → 需要更高级方法（积分因子变换等）
```

**各类通解公式**（见附录 A 第 19 节）

**典型题**

解方程 $y'=\dfrac{y}{x}+x$。

> 改写为 $y'-\dfrac{1}{x}y=x$，为一阶线性，$p=-1/x$，积分因子 $\mu=e^{-\int1/x\,dx}=1/x$；$y/x=\int(x\cdot1/x)\,dx+C=x+C$，故 $y=x^2+Cx$。

**关联章节**：Ch.23（一阶 ODE）| **关联 Toolkit**：TK-09

---

## 模型 26：二阶 ODE 特征根 3 种

**触发条件**

题目给出二阶常系数线性 ODE $y''+py'+qy=f(x)$，要求求通解；或给出初始条件 $y(x_0)=y_0$，$y'(x_0)=y_0'$ 要求求特解；或用于描述振动、电路、弹簧等物理系统。

**核心思路**

二阶常系数线性 ODE 的通解 = 齐次通解 + 特解。齐次通解完全由特征根决定（只有 3 种情形）；非齐次特解用待定系数法（根据 $f(x)$ 的形式设形）。整体结构：先用特征方程求齐次通解，再根据 $f(x)$ 设特解，最后叠加。

**齐次通解（特征方程 $r^2+pr+q=0$）**

| 特征根类型 | 通解 | 典型物理场景 |
|:---:|:---|:---|
| $r_1\neq r_2$（两不同实根）| $C_1e^{r_1x}+C_2e^{r_2x}$ | 过阻尼振动 |
| $r=r_1=r_2$（重根）| $(C_1+C_2x)e^{rx}$ | 临界阻尼振动 |
| $r=\alpha\pm\beta i$（共轭复根）| $e^{\alpha x}(C_1\cos\beta x+C_2\sin\beta x)$ | 欠阻尼振动（振荡）|

**非齐次特解设法**（待定系数法，$f(x)=P_m(x)e^{\lambda x}$）

设 $y^*=x^k Q_m(x)e^{\lambda x}$，其中 $k$ 等于 $\lambda$ 作为特征根的重数（0、1 或 2）。

**典型题**

解 $y''-3y'+2y=e^x$。

> 特征方程 $r^2-3r+2=0$，根 $r=1,2$；齐次通解 $y_H=C_1e^x+C_2e^{2x}$；$\lambda=1$ 是单重根（$k=1$），设 $y^*=axe^x$；代入 $-ae^x=e^x$，$a=-1$；$y^*=-xe^x$；通解 $y=C_1e^x+C_2e^{2x}-xe^x$。

**关联章节**：Ch.24（二阶 ODE）| **关联 Toolkit**：TK-09

---

## Part 8 AI 微积分（模型 27–30）

---

## 模型 27：凸函数 Hessian 判定

**触发条件**

需要判断多元函数是否为凸函数（以确保优化问题的全局最优性）；或需要判断某驻点是极小值 / 极大值 / 鞍点；或验证损失函数的凸性（如逻辑回归、SVM 损失）。

**核心思路**

多变量函数的凸性由 Hessian 矩阵（二阶偏导数矩阵）的正定性决定。正定 Hessian $\Leftrightarrow$ 严格凸，半正定 $\Leftrightarrow$ 凸。凸函数的局部最小就是全局最小，这是凸优化算法（梯度下降、Newton 法等）有效性的基础。

**Hessian 矩阵**

$$H=\nabla^2f=\begin{pmatrix}f_{x_1x_1}&f_{x_1x_2}&\cdots&f_{x_1x_n}\\f_{x_2x_1}&f_{x_2x_2}&\cdots&f_{x_2x_n}\\\vdots&\vdots&\ddots&\vdots\\f_{x_nx_1}&f_{x_nx_2}&\cdots&f_{x_nx_n}\end{pmatrix}$$

**判定速查**

| Hessian 性质 | 充要 / 充分 | 结论 |
|:---:|:---:|:---:|
| $H\succ0$（正定，所有特征值 $>0$）| 充分 | $f$ 严格凸 |
| $H\succcurlyeq0$（半正定）| 充分 | $f$ 凸 |
| $H\prec0$（负定）| 充分 | $f$ 严格凹 |
| $H$ 不定（特征值有正有负）| — | $f$ 非凸非凹（鞍点候选）|

**正定判别法（二阶）**：$H=\begin{pmatrix}a&b\\b&c\end{pmatrix}$ 正定 $\Leftrightarrow$ $a>0$ 且 $ac-b^2>0$（Sylvester 准则）。

**AI 应用**：梯度下降的最优步长 $\eta^*=1/\lambda_{\max}(H)$；Newton 法用 $H^{-1}\nabla f$ 代替梯度，收敛速度从一阶加速到二阶。

**典型题**

判断 $f(x,y)=x^2+xy+y^2$ 是否凸。

> $H=\begin{pmatrix}2&1\\1&2\end{pmatrix}$，$\det H=4-1=3>0$，$f_{xx}=2>0$，$H$ 正定，$f$ 严格凸。

**关联章节**：Ch.25（凸优化）| **关联 Toolkit**：TK-10，TK-12

---

## 模型 28：矩阵微积分（向量 / 矩阵导数）

**触发条件**

需要对向量函数或矩阵函数求导（如损失函数对权重矩阵的梯度）；或题目给出矩阵形式的目标函数（如最小二乘 $\|Ax-b\|^2$）要求求最优解；或反向传播中需要求各层权重的梯度。

**核心思路**

矩阵微积分是多元链式法则的矩阵版本。对向量 $\mathbf{x}\in\mathbb{R}^n$，标量函数 $f$ 对 $\mathbf{x}$ 的梯度是一个向量；对矩阵 $A\in\mathbb{R}^{m\times n}$，标量函数 $f$ 对 $A$ 的"梯度"是同维数的矩阵（每个元素是 $\partial f/\partial A_{ij}$）。

**核心公式速查**（见附录 A 第 11 节）

| 公式 | 说明 |
|:---|:---|
| $\dfrac{\partial(Ax)}{\partial x}=A$ | 线性变换的 Jacobian |
| $\dfrac{\partial(x^Tx)}{\partial x}=2x$ | 二次型 $\|x\|^2$ 的梯度 |
| $\dfrac{\partial(x^TAx)}{\partial x}=(A+A^T)x$（$A$ 对称时 $=2Ax$）| 二次型的梯度 |
| $\dfrac{\partial\|Ax-b\|^2}{\partial x}=2A^T(Ax-b)$ | 最小二乘梯度（令其为零得 $A^TAx=A^Tb$）|
| $\dfrac{\partial\ln\|A\|}{\partial A}=A^{-T}$ | 对数行列式 |

**矩阵微积分的布局约定**：分子布局（梯度是行向量）vs 分母布局（梯度是列向量）；约定统一为分母布局，与梯度向量同维。

**典型题**

最小化 $f(\mathbf{w})=\|\mathbf{X}\mathbf{w}-\mathbf{y}\|^2$（线性回归），求最优 $\mathbf{w}^*$。

> $\nabla f=2\mathbf{X}^T(\mathbf{X}\mathbf{w}-\mathbf{y})=\mathbf{0}$，即正规方程 $\mathbf{X}^T\mathbf{X}\mathbf{w}^*=\mathbf{X}^T\mathbf{y}$，$\mathbf{w}^*=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$。

**关联章节**：Ch.26（矩阵微积分）| **关联 Toolkit**：TK-07，TK-12

---

## 模型 29：概率 PDF/CDF + KL 散度

**触发条件**

需要计算概率分布相关的积分（归一化验证、期望、方差）；需要计算两个概率分布之间的 KL 散度；或在 VAE、GAN 等模型中需要优化 KL 散度。

**核心思路**

概率 PDF 的归一化条件 $\int f(x)\,dx=1$ 和 Gamma、Beta 函数是概率分布积分的基础工具。KL 散度是衡量两分布差异的"准距离"（非对称），其非负性由 Jensen 不等式保证，其最小化是最大似然估计和 EM 算法的理论基础。

**关键公式**

$$\int_{-\infty}^{+\infty}e^{-ax^2}\,dx=\sqrt{\frac{\pi}{a}} \quad (a>0) \quad \text{（高斯积分）}$$

$$\Gamma(n)=\int_0^\infty t^{n-1}e^{-t}\,dt=(n-1)! \quad (n\in\mathbb{N}^+), \quad \Gamma(1/2)=\sqrt{\pi}$$

$$B(\alpha,\beta)=\int_0^1t^{\alpha-1}(1-t)^{\beta-1}\,dt=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$$

$$\mathrm{KL}(p\|q)=\int p(x)\ln\frac{p(x)}{q(x)}\,dx\geq0, \quad \text{等号当且仅当 } p=q \text{ a.e.}$$

**一维高斯 KL 散度**（闭合公式）：

$$\mathrm{KL}\!\left(\mathcal{N}(\mu_1,\sigma_1^2)\|\mathcal{N}(\mu_2,\sigma_2^2)\right)=\ln\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac{1}{2}$$

**典型题**

验证正态分布 $\mathcal{N}(0,1)$ 的 PDF $f(x)=\dfrac{1}{\sqrt{2\pi}}e^{-x^2/2}$ 满足归一化条件。

> $\int_{-\infty}^{+\infty}e^{-x^2/2}\,dx=\sqrt{2\pi}$（令 $a=1/2$ 代入高斯积分公式），故 $\int_{-\infty}^{+\infty}f(x)\,dx=1$。

**关联章节**：Ch.27（概率中的微积分）| **关联 Toolkit**：TK-11，TK-12

---

## 模型 30：Itô 公式 + Euler-Maruyama

**触发条件**

问题涉及随机过程、布朗运动（Wiener 过程）和随机微分方程（SDE）；或需要对随机过程的函数 $f(W_t)$ 求微分；或数值模拟 SDE 路径（Euler-Maruyama 格式）。

**核心思路**

布朗运动 $W_t$ 的路径不可微（几乎处处），不能用普通链式法则求 $df(W_t)$。Itô 公式是随机微积分中的"链式法则"，多了一个"二阶 Itô 修正项" $\dfrac{1}{2}f''dt$（因为 $dW_t^2=dt$，阶数高于普通 $dt^2$）。

**Itô 公式**（一维情形，$dX_t=\mu\,dt+\sigma\,dW_t$）

$$df(X_t)=f'(X_t)\,dX_t+\frac{1}{2}f''(X_t)\sigma^2\,dt=\left(f'(X_t)\mu+\frac{1}{2}f''(X_t)\sigma^2\right)dt+f'(X_t)\sigma\,dW_t$$

**与普通链式的区别**：多了 $\dfrac{1}{2}f''(X_t)\sigma^2\,dt$ 这一项（Itô 修正），正是因为布朗运动的二次变差 $[W]_t=t$。

**Euler-Maruyama 数值格式**（$\Delta t$ 为步长，$\xi_k\sim\mathcal{N}(0,1)$ 独立）

$$X_{t+\Delta t}\approx X_t+\mu(X_t,t)\Delta t+\sigma(X_t,t)\sqrt{\Delta t}\,\xi_k$$

这是 SDE 数值解的最基础格式（一阶强收敛，强收敛阶 $1/2$）。

**典型例子**

对几何布朗运动 $dS_t=\mu S_t\,dt+\sigma S_t\,dW_t$，令 $f(S)=\ln S$，由 Itô 公式：

$$d(\ln S_t)=\frac{1}{S_t}dS_t-\frac{1}{2S_t^2}\sigma^2S_t^2\,dt=\left(\mu-\frac{\sigma^2}{2}\right)dt+\sigma\,dW_t$$

故 $\ln S_t=\ln S_0+\left(\mu-\dfrac{\sigma^2}{2}\right)t+\sigma W_t$，即 $S_t=S_0e^{(\mu-\sigma^2/2)t+\sigma W_t}$（对数正态分布）。

**关联章节**：Ch.28（随机微分方程）| **关联 Toolkit**：TK-07，TK-12

---

## 附：思维方法网——12 Toolkit × 30 模型关联表

> 行为 12 个 toolkit（TK-01 至 TK-12），列为 30 个模型。"●" 表示主要关联，"○" 表示次要关联。

| Toolkit | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---------|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| TK-01 极限 ε 语言 | ● | ● | ● | ● | | | | | | | | | | | | | | | | | | | | | | | | | | |
| TK-02 等价无穷小 | | ● | ● | ○ | | | | ○ | ○ | | | | | | | | | | | | | | | | | | | | | |
| TK-03 求导套路 | | | | | ● | ● | ● | ○ | | | | | | | | | | | | | | | | | | | | | | |
| TK-04 积分技巧 | | | | | | | | | | ● | ● | ● | ● | ● | ○ | | | | | | | | | | | | | | | |
| TK-05 级数判敛 | | | | | | | | | | | | | | | ● | ● | ● | ● | | | | | | | | | | | | |
| TK-06 Taylor 展开 | | ○ | | | | | | ● | ● | | | | | | | ○ | ● | ○ | | | | | | | | | | | | |
| TK-07 多元链式 | | | | | | | | | | | | | | | | | | | ● | ○ | ○ | ● | ○ | ● | | | ○ | ● | | ● |
| TK-08 多元积分 | | | | | | | | | | | ○ | | | | | | | | | ● | ● | ● | ● | ● | | | | | | |
| TK-09 ODE 识别 | | | | | | | | | | | | | | | | | | | | | | | | | ● | ● | | | | |
| TK-10 凸性极值 | | | | | | | ● | | | | | | | | | | | | | | | | | | | ○ | ● | ○ | | |
| TK-11 不等式 | ○ | | | | | | | | | | | | | | ● | | | | | | | | | | | | | | ● | |
| TK-12 AI 思维 | | | | | | | | | | | | | | | | | | | ○ | | | | | | | | ● | ● | ● | ● |

### 高频关联：每个模型最常用的 Toolkit

| 模型 | 名称 | 最常用 Toolkit |
|:---:|:---|:---|
| 1 | ε-N / ε-δ 三步证法 | TK-01（极限 ε 语言）|
| 2 | 等价无穷小乘除替换 | TK-02（等价无穷小）、TK-01 |
| 3 | 两个重要极限 | TK-01、TK-02 |
| 4 | 连续 / 一致连续辨析 | TK-01（定量定义）|
| 5 | 导数 6 大规则决策树 | TK-03（求导套路）|
| 6 | 隐函数 + 对数求导法 | TK-03（求导套路）|
| 7 | 单调极值标准 4 步 | TK-03、TK-10（凸性极值）|
| 8 | L'Hôpital + Taylor 取代 | TK-06（Taylor 展开）、TK-02 |
| 9 | Taylor 6 大 Maclaurin | TK-06（Taylor 展开）|
| 10 | 不定积分基本公式 | TK-04（积分技巧）|
| 11 | 定积分牛顿-莱布尼茨 | TK-04（积分技巧）|
| 12 | LIATE 分部积分 | TK-04（积分技巧）|
| 13 | 三角换元 3 种情形 | TK-04（积分技巧）|
| 14 | 部分分式分解 | TK-04（积分技巧）|
| 15 | 反常积分 p-判别 | TK-05（级数判敛）、TK-11 |
| 16 | 级数判敛决策树 | TK-05（级数判敛）|
| 17 | 幂级数收敛半径 + 端点 | TK-05、TK-06 |
| 18 | Fourier 级数展开 | TK-05、TK-06 |
| 19 | 多元链式 + 梯度 / Jacobian | TK-07（多元链式）|
| 20 | 二重积分极坐标 vs 直角 | TK-08（多元积分）|
| 21 | 三重积分球 / 柱坐标 | TK-08（多元积分）|
| 22 | 第一型 vs 第二型曲线积分 | TK-07、TK-08 |
| 23 | 通量与曲面积分 | TK-08（多元积分）|
| 24 | Green / Stokes / Gauss | TK-07、TK-08 |
| 25 | 一阶 ODE 5 类识别 | TK-09（ODE 识别）|
| 26 | 二阶 ODE 特征根 3 种 | TK-09（ODE 识别）|
| 27 | 凸函数 Hessian 判定 | TK-10（凸性极值）、TK-12 |
| 28 | 矩阵微积分 | TK-07、TK-12 |
| 29 | 概率 PDF/CDF + KL 散度 | TK-11（不等式）、TK-12 |
| 30 | Itô 公式 + Euler-Maruyama | TK-07、TK-12 |
