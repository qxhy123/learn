# 附录 E：微积分 60 道提升题（考研真题 + AI 应用）

> 共 **60 题**，对应考研数学（数一 / 数二）压轴档难度，兼含 AI 工程应用场景。
> 覆盖 **极限连续**（Ch.4–6）、**微分应用**（Ch.7–10）、**积分技巧**（Ch.11–14）、**级数**（Ch.15–17）、**多元微积分**（Ch.18–22）、**ODE**（Ch.23–24）、**AI 微积分**（Ch.25–28）七大专题。
> 每题均标注 **关键技巧标签**，便于按方法分类训练。**不附答案**，详解请见附录 F。
> 编号 E.01–E.60，按主题分组连续编号。
>
> **使用建议**：本附录为提升档（考研压轴 + AI 工程）专题训练册。建议在熟练附录 D（中档 100 题）后再练本附录；每天精做 1–2 题，重点是 **识别题型 → 选择方法 → 严格推导** 三步逻辑。AI 应用题（E.53–E.60）需结合 Part 8 相关章节。

---

## 分组 1：极限连续（E.01–E.06，共 6 题）

> 对应 Ch.4–6，考查 $\varepsilon$-$\delta$ 严格定义、连续性判断、特殊极限精确化三大主题。
> 主要技巧：$\varepsilon$-$\delta$ 构造、Taylor 展开配合阶数比较、Squeeze/Sandwich、间断点分类。

**E.01** [提升] Ch.5 [$\varepsilon$-$\delta$ 证明]

用 $\varepsilon$-$\delta$ 语言严格证明 $\displaystyle\lim_{x\to 1}\frac{x^2-1}{x-1}=2$。

（1）写出待证命题的 $\varepsilon$-$\delta$ 表述；
（2）对给定 $\varepsilon > 0$，显式构造满足条件的 $\delta$，并验证 $0 < |x - 1| < \delta$ 时确有 $\left|\dfrac{x^2-1}{x-1} - 2\right| < \varepsilon$；
（3）将上述方法推广：用 $\varepsilon$-$\delta$ 证明一般结论 $\displaystyle\lim_{x\to a}\frac{x^n - a^n}{x - a} = na^{n-1}$（$n \in \mathbb{N}^*$，$a \neq 0$）。

**E.02** [提升] Ch.5 [Taylor 展开 + 阶比较]

求极限 $\displaystyle\lim_{x\to 0}\frac{\sqrt{1+2x}-\sqrt[3]{1+3x}}{x^2}$。

（1）将 $\sqrt{1+2x}$ 与 $\sqrt[3]{1+3x}$ 各自展开至 $x^2$ 项；
（2）计算分子的二阶展开，说明为何一阶相消；
（3）给出极限值，并用同阶无穷小的语言描述结论。

**E.03** [提升] Ch.5 [$1^\infty$ 型 + 对数技巧]

求 $\displaystyle\lim_{x\to 0}\left(\frac{1+\tan x}{1+\sin x}\right)^{1/\sin^3 x}$。

（1）说明该极限属于 $1^\infty$ 不定型；
（2）取对数后化为 $0/0$ 型，利用 $\ln(1+u)\sim u$ 与 $\tan x - \sin x = \tan x(1-\cos x)$ 算出对数极限；
（3）给出原极限，并与 $e^{1/2}$ 的近似值作数值对照。

**E.04** [提升] Ch.4–5 [Riemann 和 + 定积分]

求 $\displaystyle\lim_{n\to\infty}\frac{1}{n}\sum_{k=1}^n \sqrt{1 - \left(\frac{k}{n}\right)^2}$。

（1）将求和识别为 $f(x) = \sqrt{1-x^2}$ 在 $[0,1]$ 上的 Riemann 和；
（2）计算对应的定积分（几何意义：单位圆面积的四分之一）；
（3）推广：写出 $\displaystyle\lim_{n\to\infty}\frac{1}{n}\sum_{k=1}^n f\!\left(\frac{k}{n}\right) = \int_0^1 f(x)\,dx$ 的成立条件。

**E.05** [提升] Ch.6 [间断点分类 + 补全连续]

设 $f(x) = \dfrac{\sin x}{|x|(1-e^{1/x})}$（$x \neq 0$）。

（1）分别计算 $x \to 0^+$ 与 $x \to 0^-$ 时的极限，判断 $x=0$ 的间断点类型；
（2）讨论是否可以补充定义 $f(0)$ 使 $f$ 在 $x=0$ 处连续；
（3）证明：若 $f$ 在闭区间 $[a,b]$ 上连续且 $f(a)f(b) < 0$，则存在 $c \in (a,b)$ 使 $f(c)=0$（零点定理 / 介值定理的特殊情形），并用此证明 $x^3 - x - 1 = 0$ 在 $(1,2)$ 内有实根。

**E.06** [提升] Ch.5–6 [等价无穷小替换 + 精确化]

设 $\alpha(x) = \ln(1+x) - x + \dfrac{x^2}{2}$ 与 $\beta(x) = x^3$（$x\to 0$）。

（1）利用 Taylor 展开确定 $\alpha(x)$ 与 $x^3$ 的比值极限，说明 $\alpha(x) = O(x^3)$ 并求精确系数；
（2）计算 $\displaystyle\lim_{x\to 0}\frac{\ln(1+x) - x + x^2/2 - x^3/3}{x^4}$；
（3）讨论：在等价无穷小替换 $\ln(1+x)\sim x$ 中，**何时**替换有效，**何时**必须保留高阶项。

---

## 分组 2：微分应用（E.07–E.18，共 12 题）

> 对应 Ch.7–10，覆盖凸性与 Jensen 不等式、Newton 迭代、中值定理综合、Taylor 余项估计、最优化等核心专题。
> 主要技巧：凸性证明、构造辅助函数、Rolle / Lagrange / Cauchy 中值定理、Newton-Raphson 迭代收敛分析、Taylor 余项 Lagrange 形式。

**E.07** [提升] Ch.10 [凸性 + Jensen 不等式]

设 $f(x) = -\ln x$（$x > 0$）。

（1）证明 $f$ 是严格下凸函数（$f''(x) > 0$）；
（2）对 $a, b > 0$，由 Jensen 不等式 $f\!\left(\dfrac{a+b}{2}\right) \le \dfrac{f(a)+f(b)}{2}$ 推出 AM-GM 不等式 $\dfrac{a+b}{2} \ge \sqrt{ab}$；
（3）推广至 $n$ 元：对 $a_1, \ldots, a_n > 0$，用 Jensen 证明 $\dfrac{a_1+\cdots+a_n}{n} \ge \sqrt[n]{a_1\cdots a_n}$。

**E.08** [提升] Ch.9 [Rolle 定理 + 构造辅助函数]

设 $f \in C[0,1]$，在 $(0,1)$ 上可导，且 $\int_0^1 f(x)\,dx = 0$。证明存在 $\xi \in (0,1)$ 使 $f'(\xi) = 0$。

（1）说明 $\int_0^1 f(x)\,dx = 0$ 的几何含义；
（2）构造 $F(x) = \int_0^x f(t)\,dt$，验证 $F(0) = F(1) = 0$；
（3）对 $F$ 应用 Rolle 定理完成证明，并给出一个具体反例说明"$f(0)=f(1)=0$"不足以保证 $f'\equiv 0$。

**E.09** [提升] Ch.9 [Lagrange 中值定理 + 不等式证明]

对 $x > 0$，证明 $\dfrac{x}{1+x} < \ln(1+x) < x$。

（1）对右边不等式 $\ln(1+x) < x$：令 $g(x) = x - \ln(1+x)$，验证 $g(0) = 0$ 且 $g'(x) > 0$（$x > 0$）；
（2）对左边不等式 $\ln(1+x) > \dfrac{x}{1+x}$：构造 $h(x) = \ln(1+x) - \dfrac{x}{1+x}$，同法验证；
（3）由上述夹逼推出 $\displaystyle\lim_{n\to\infty}\left(1+\frac{1}{n}\right)^n = e$ 的严格下界：$\left(1+\dfrac{1}{n}\right)^n > e^{n/(n+1)}$。

**E.10** [提升] Ch.9–10 [Cauchy 中值定理 + L'Hôpital]

求 $\displaystyle\lim_{x\to 0}\frac{e^x - 1 - x - x^2/2}{x^3}$，并用两种方法：

（1）方法一：将分子展开至 $x^3$ 项后直接取极限；
（2）方法二：三次应用 L'Hôpital 法则，说明每步的 $0/0$ 条件；
（3）说明 Cauchy 中值定理与 L'Hôpital 的本质联系（$\dfrac{f(x)-f(0)}{g(x)-g(0)} = \dfrac{f'(\xi)}{g'(\xi)}$ 的推导思路）。

**E.11** [提升] Ch.10 [Taylor 展开 + 余项估计]

设 $f(x) = \sin x$，在 $x_0 = 0$ 处展开到 $n$ 阶。

（1）写出 Maclaurin 公式 $\sin x = x - \dfrac{x^3}{6} + \dfrac{x^5}{120} - \cdots + R_n(x)$，给出 Lagrange 余项 $R_{2m+1}(x)$ 的表达式；
（2）取 $n=5$，估计 $\left|\sin(0.1) - \left(0.1 - \dfrac{0.1^3}{6} + \dfrac{0.1^5}{120}\right)\right|$ 的上界；
（3）由余项估计说明 $\pi/4 = 1 - 1/3 + 1/5 - 1/7 + \cdots$（Leibniz 公式）的收敛速度：保留精度 $10^{-3}$ 至少需要多少项？

**E.12** [提升] Ch.9 [Newton 迭代 + 二阶收敛]

用 Newton-Raphson 迭代求方程 $x^3 - x - 1 = 0$ 的实根 $r \approx 1.3247$。

（1）写出迭代格式 $x_{n+1} = x_n - \dfrac{f(x_n)}{f'(x_n)}$，其中 $f(x) = x^3 - x - 1$；
（2）以 $x_0 = 1.5$ 为初值，手算前三步 $x_1, x_2, x_3$（保留 6 位有效数字）；
（3）证明 Newton 迭代在 $r$ 附近**二阶**收敛：$|x_{n+1} - r| \le C|x_n - r|^2$，给出常数 $C = \dfrac{|f''(r)|}{2|f'(r)|}$ 的表达式，并数值估计 $C$。

**E.13** [提升] Ch.10 [极值 + 含参讨论]

设 $f(x) = x \ln x - a(x - 1)$（$x > 0$，$a \in \mathbb{R}$）。

（1）求 $f'(x)$ 并分析驻点个数随 $a$ 的变化；
（2）当 $a = 1$ 时，确定 $f$ 的单调区间与极值；
（3）当 $a > 0$ 时，证明 $x \ln x \ge a(x-1) - (x-1)^2/2$（对所有 $x > 0$），并说明等号成立的条件。

**E.14** [提升] Ch.9–10 [凸函数 + 不等式证明]

证明：当 $x > 0$ 时，$\ln(1+x) < x - \dfrac{x^2}{2} + \dfrac{x^3}{3}$。

（1）令 $g(x) = x - \dfrac{x^2}{2} + \dfrac{x^3}{3} - \ln(1+x)$，计算 $g(0), g'(0)$；
（2）求 $g''(x)$ 并分析其符号，从而确定 $g'$ 的单调性；
（3）逐步推导 $g(x) > 0$（$x > 0$），完成不等式证明；并由此估计 $\ln 2$ 的精确范围。

**E.15** [提升] Ch.9 [Lagrange 中值 + 双变量不等式]

设 $0 < x_1 < x_2$，证明 $\dfrac{\ln x_1 - \ln x_2}{x_1 - x_2} < \dfrac{1}{\sqrt{x_1 x_2}}$。

（1）对 $\ln$ 在 $[x_1, x_2]$ 上应用 Lagrange 中值定理，得 $\ln x_1 - \ln x_2 = (x_1 - x_2)/\xi$，$\xi \in (x_1, x_2)$；
（2）比较 $\xi$ 与 $\sqrt{x_1 x_2}$：由 AM-GM 知 $\sqrt{x_1 x_2} < \dfrac{x_1 + x_2}{2}$，分析 $\xi$ 的范围以证明 $\xi > \sqrt{x_1 x_2}$；
（3）整合步骤完成证明，并讨论等号不成立的原因。

**E.16** [提升] Ch.9 [参数方程 + 弧长]

设参数曲线 $\begin{cases} x = t - \sin t \\ y = 1 - \cos t \end{cases}$（摆线，$0 \le t \le 2\pi$）。

（1）求 $\dfrac{dy}{dx}$ 与 $\dfrac{d^2y}{dx^2}$（用 $t$ 表示）；
（2）求摆线一拱（$t \in [0,2\pi]$）的弧长 $L = \int_0^{2\pi} \sqrt{x'^2 + y'^2}\,dt$；
（3）求摆线一拱与 $x$ 轴围成的面积，并指出与弧长公式的联系。

**E.17** [提升] Ch.8–9 [隐函数 + 高阶导数]

方程 $e^y + xy = e$ 在 $(0, 1)$ 附近确定隐函数 $y = y(x)$。

（1）隐函数存在性：验证 $(0,1)$ 满足方程且 $\partial(e^y + xy)/\partial y \neq 0$；
（2）求 $y'(0)$ 与 $y''(0)$；
（3）写出 $y(x)$ 在 $x = 0$ 处的二阶 Taylor 展开，并用展开值估计 $y(0.1)$。

**E.18** [提升] Ch.9–10 [最优化 + Lagrange 乘子预热]

在约束 $x + y = s$（$s > 0$，$x, y > 0$）下，求 $P = x^a y^b$（$a, b > 0$）的最大值。

（1）代入约束 $y = s - x$，化为单变量优化，求临界点 $x^* = \dfrac{as}{a+b}$；
（2）验证 $x^*$ 对应极大值，计算 $P_{\max} = \left(\dfrac{a}{a+b}\right)^a \left(\dfrac{b}{a+b}\right)^b s^{a+b}$；
（3）由此推出加权 AM-GM 不等式：$\dfrac{ax + by}{a+b} \ge x^{a/(a+b)} y^{b/(a+b)}$（对 $x, y > 0$）；说明 $a = b = 1$ 时退化为标准 AM-GM。

---

## 分组 3：积分技巧（E.19–E.28，共 10 题）

> 对应 Ch.11–14，覆盖万能代换、分部积分循环、Euler 积分、换序技巧、广义积分判敛等高频考点。
> 主要技巧：万能代换（$t=\tan(x/2)$）、LIATE 分部、循环积分、Wallis 公式、Dirichlet 判别、对称区间化简。

**E.19** [提升] Ch.13 [万能代换]

计算 $\displaystyle\int \frac{dx}{1 + \sin x + \cos x}$。

（1）令 $t = \tan(x/2)$，写出 $\sin x = \dfrac{2t}{1+t^2}$，$\cos x = \dfrac{1-t^2}{1+t^2}$，$dx = \dfrac{2\,dt}{1+t^2}$ 的代换公式；
（2）代入后化简被积函数为 $t$ 的有理式；
（3）完成积分并回代 $t = \tan(x/2)$，给出最终结果 $\ln\left|1+\tan\dfrac{x}{2}\right| + C$。

**E.20** [提升] Ch.11–12 [分部积分 + 循环]

计算 $\displaystyle\int e^{2x}\cos x\,dx$。

（1）设 $I = \int e^{2x}\cos x\,dx$，第一次分部（令 $u = \cos x$，$dv = e^{2x}dx$）；
（2）对余项再分部一次，得到含 $I$ 的等式；
（3）移项解出 $I$，并验证 $I = \dfrac{e^{2x}(2\cos x + \sin x)}{5} + C$（系数 $5 = 2^2 + 1^2$）。

**E.21** [提升] Ch.13 [根式代换 + 有理化]

计算 $\displaystyle\int \frac{dx}{\sqrt{x} + \sqrt[3]{x}}$。

（1）取 $x = t^6$（$6 = \mathrm{lcm}(2,3)$），写出 $dx, \sqrt{x}, \sqrt[3]{x}$ 用 $t$ 的表达式；
（2）化为有理函数 $\int \dfrac{6t^3}{t+1}\,dt$，做多项式除法；
（3）完成积分并回代 $t = x^{1/6}$，给出含 $\ln|x^{1/6}+1|$ 的结果。

**E.22** [提升] Ch.11–12 [Wallis 积分 + 递推]

证明 Wallis 积分递推公式：$I_n = \displaystyle\int_0^{\pi/2} \sin^n x\,dx$ 满足 $I_n = \dfrac{n-1}{n} I_{n-2}$，并计算 $I_6$。

（1）对 $I_n$ 做一次分部（令 $u = \sin^{n-1}x$，$dv = \sin x\,dx$）推导递推关系；
（2）用递推公式计算 $I_6 = \dfrac{5}{6}\cdot\dfrac{3}{4}\cdot\dfrac{1}{2}\cdot\dfrac{\pi}{2}$，给出精确值；
（3）由 Wallis 公式推出 $\displaystyle\lim_{n\to\infty}\frac{I_{2n}}{I_{2n+1}} = 1$，并从中推出 Wallis 乘积 $\dfrac{\pi}{2} = \displaystyle\prod_{k=1}^\infty \frac{4k^2}{4k^2-1}$。

**E.23** [提升] Ch.12 [含参定积分 + 对称技巧]

计算 $\displaystyle\int_0^1 \frac{\ln(1+x)}{1+x^2}\,dx$。

（1）令 $x = \tan t$，$t \in [0, \pi/4]$，将积分化为 $\displaystyle\int_0^{\pi/4}\ln(1+\tan t)\,dt$；
（2）令 $u = \pi/4 - t$ 得到另一表达式，利用 $\tan(\pi/4 - t) = \dfrac{1-\tan t}{1+\tan t}$；
（3）两式相加，利用 $(1+\tan t)\!\left(1+\tan(\pi/4-t)\right) = 2$ 得 $I = \dfrac{\pi\ln 2}{8}$。

**E.24** [提升] Ch.12 [对称区间 + $e^x$ 技巧]

计算 $\displaystyle\int_{-1}^1 \frac{x^2}{1+e^x}\,dx$。

（1）设 $I = \displaystyle\int_{-1}^1 \dfrac{x^2}{1+e^x}\,dx$，令 $x \to -x$ 得 $I = \displaystyle\int_{-1}^1 \dfrac{x^2 e^x}{1+e^x}\,dx$；
（2）两式相加，利用 $\dfrac{1}{1+e^x} + \dfrac{e^x}{1+e^x} = 1$，化为 $2I = \displaystyle\int_{-1}^1 x^2\,dx$；
（3）得出 $I = \dfrac{1}{3}$，并总结"对称区间含 $e^x$ 分母"题型的一般做法。

**E.25** [提升] Ch.14 [广义积分判敛 + $p$-积分比较]

讨论广义积分 $\displaystyle\int_0^{+\infty}\frac{dx}{x^p(1+x)}$ 的收敛性（$p \in \mathbb{R}$）。

（1）分析 $x \to 0^+$ 端：被积 $\sim x^{-p}$，收敛条件 $p < 1$；
（2）分析 $x \to +\infty$ 端：被积 $\sim x^{-p-1}$，收敛条件 $p > 0$；
（3）结论：两端同时收敛当且仅当 $0 < p < 1$；写出 $p = 1/2$ 时的积分精确值（利用 $\beta$ 函数 $B(1-p, p) = \pi/\sin(\pi p)$）。

**E.26** [提升] Ch.12–13 [换序积分 + 变限积分]

计算 $\displaystyle\int_0^1 \frac{e^x - 1}{x}\,dx$（给出级数表示）。

（1）将 $e^x - 1 = \sum_{n=1}^\infty \dfrac{x^n}{n!}$ 代入，逐项积分（需验证可以换序）；
（2）得出 $\displaystyle\int_0^1 \frac{e^x-1}{x}\,dx = \sum_{n=1}^\infty \frac{1}{n \cdot n!}$；
（3）估计此级数的数值（前四项求和与精确值的误差），并讨论它与 $\mathrm{Ei}(1)$（指数积分）的关系。

**E.27** [提升] Ch.14 [Gamma 函数 + 递推]

利用 $\Gamma(s) = \displaystyle\int_0^{+\infty} t^{s-1} e^{-t}\,dt$（$s > 0$）。

（1）分部证明递推 $\Gamma(s+1) = s\,\Gamma(s)$；
（2）由 $\Gamma(1) = 1$ 推出 $\Gamma(n) = (n-1)!$（$n \in \mathbb{N}^*$）；
（3）计算 $\Gamma\!\left(\dfrac{7}{2}\right)$，并利用 $\Gamma\!\left(\dfrac{1}{2}\right) = \sqrt{\pi}$（由高斯积分 $\int_0^\infty e^{-t^2}dt = \sqrt{\pi}/2$ 推出）写出结果。

**E.28** [提升] Ch.14 [Dirichlet 积分 + 条件收敛]

考察 $\displaystyle\int_0^{+\infty} \frac{\sin x}{x}\,dx$（Dirichlet 积分）。

（1）证明该广义积分**条件收敛**：用 Dirichlet 判别说明收敛，再证 $\displaystyle\int_0^{+\infty}\left|\dfrac{\sin x}{x}\right|dx = +\infty$；
（2）利用参数积分方法，设 $I(t) = \displaystyle\int_0^\infty e^{-tx}\dfrac{\sin x}{x}\,dx$，对 $t$ 求导得 $I'(t) = -\dfrac{1}{1+t^2}$；
（3）由 $I(\infty) = 0$ 积分反推 $I(0) = \arctan t\Big|_0^\infty = \dfrac{\pi}{2}$，完成计算。

---

## 分组 4：级数（E.29–E.36，共 8 题）

> 对应 Ch.15–17，覆盖条件收敛与绝对收敛、幂级数和函数、Fourier 级数、Abel / Dirichlet 判别等。
> 主要技巧：比值法、Leibniz 判别、逐项积分 / 求导、Abel 求和、Parseval 等式。

**E.29** [提升] Ch.15 [Leibniz 判别 + 绝对 vs 条件收敛]

对级数 $\displaystyle\sum_{n=1}^\infty \frac{(-1)^n}{\sqrt{n}}$：

（1）用 Leibniz 判别（交错级数，项单调趋零）证明级数**收敛**；
（2）讨论是否**绝对**收敛：$\sum 1/\sqrt{n}$ 是 $p$-级数（$p = 1/2 < 1$），发散；
（3）结论：该级数**条件收敛**但不绝对收敛；并由 Riemann 重排定理说明，条件收敛级数可以重排成任意实数或 $\pm\infty$。

**E.30** [提升] Ch.15–16 [Abel 判别 + 幂级数端点]

求幂级数 $\displaystyle\sum_{n=1}^\infty \frac{x^n}{n}$ 的收敛半径、收敛域与和函数。

（1）比值法得收敛半径 $R = 1$；
（2）讨论端点：$x = 1$ 时调和级数发散；$x = -1$ 时交错调和级数收敛（Leibniz）；
（3）和函数 $S(x) = -\ln(1-x)$（$x \in [-1, 1)$）：先在 $|x| < 1$ 内逐项积分推导，再讨论 $x = -1$ 端点的 Abel 定理。

**E.31** [提升] Ch.15 [比较判别 + 复合型级数]

判断 $\displaystyle\sum_{n=1}^\infty \frac{n!}{n^n}$ 的收敛性。

（1）用比值法：$\dfrac{a_{n+1}}{a_n} = \dfrac{(n+1)!}{(n+1)^{n+1}} \cdot \dfrac{n^n}{n!} = \left(\dfrac{n}{n+1}\right)^n \to \dfrac{1}{e} < 1$；
（2）由此证明级数收敛；
（3）引申：利用 Stirling 公式 $n! \approx \sqrt{2\pi n}\left(\dfrac{n}{e}\right)^n$ 给出 $a_n$ 的渐近估计，说明级数以 $1/e$ 速度收敛。

**E.32** [提升] Ch.16 [幂级数求和 + 逐项求导]

求 $\displaystyle\sum_{n=1}^\infty n x^{n-1}$（$|x| < 1$）的和函数，并计算 $\displaystyle\sum_{n=1}^\infty \frac{n}{2^n}$。

（1）注意 $\displaystyle\sum_{n=1}^\infty nx^{n-1} = \dfrac{d}{dx}\sum_{n=1}^\infty x^n = \dfrac{d}{dx}\dfrac{x}{1-x}$（$|x|<1$）；
（2）完成求导，得和函数 $S(x) = \dfrac{1}{(1-x)^2}$（$|x| < 1$）；
（3）取 $x = 1/2$ 得 $\displaystyle\sum_{n=1}^\infty \dfrac{n}{2^n} = 2$，验证计算。

**E.33** [提升] Ch.16 [Taylor 级数 + 误差控制]

将 $f(x) = \arctan x$ 展开为 Maclaurin 级数，并用它计算 $\pi$ 的近似值。

（1）由 $\dfrac{1}{1+x^2} = \sum_{n=0}^\infty (-1)^n x^{2n}$（$|x| < 1$）逐项积分得 $\arctan x = \sum_{n=0}^\infty \dfrac{(-1)^n x^{2n+1}}{2n+1}$（$|x| \le 1$）；
（2）取 $x = 1$ 得 Leibniz 公式 $\pi/4 = 1 - 1/3 + 1/5 - \cdots$，但收敛极慢；
（3）取 $x = 1/\sqrt{3}$ 得 $\pi/6 = \sum_{n=0}^\infty \dfrac{(-1)^n}{(2n+1)3^n\sqrt{3}}$，估计保留精度 $10^{-6}$ 所需项数。

**E.34** [提升] Ch.15 [根值法 + 级数敛散]

判断 $\displaystyle\sum_{n=1}^\infty \left(\frac{n}{n+1}\right)^{n^2}$ 的收敛性。

（1）用 Cauchy 根值法：$\sqrt[n]{a_n} = \left(\dfrac{n}{n+1}\right)^n = \left(1 - \dfrac{1}{n+1}\right)^n$；
（2）计算 $\left(1 - \dfrac{1}{n+1}\right)^n \to 1/e < 1$（利用 $\ln$ 展开）；
（3）结论：根值极限 $l = 1/e < 1$，级数**收敛**；对比比值法的优劣。

**E.35** [提升] Ch.17 [Fourier 级数 + Parseval 等式]

设 $f(x) = x$（$-\pi < x < \pi$），作 $2\pi$ 周期延拓后展成 Fourier 级数。

（1）由奇函数性质得 $a_n = 0$；计算 $b_n = \dfrac{2}{\pi}\displaystyle\int_0^\pi x \sin nx\,dx = \dfrac{2(-1)^{n+1}}{n}$；
（2）写出 Fourier 级数 $x = \displaystyle\sum_{n=1}^\infty \dfrac{2(-1)^{n+1}}{n}\sin nx$（在 $(-\pi,\pi)$ 内成立）；
（3）用 Parseval 等式 $\dfrac{1}{\pi}\displaystyle\int_{-\pi}^\pi |f(x)|^2\,dx = \displaystyle\sum_{n=1}^\infty b_n^2$ 推出 $\displaystyle\sum_{n=1}^\infty \dfrac{1}{n^2} = \dfrac{\pi^2}{6}$（Basel 问题）。

**E.36** [提升] Ch.15–17 [函数项级数 + 一致收敛]

证明 $\displaystyle\sum_{n=1}^\infty \frac{\sin nx}{n^2}$ 在 $(-\infty, +\infty)$ 上一致收敛，并讨论其连续性与逐项积分。

（1）用 Weierstrass M-判别：$\left|\dfrac{\sin nx}{n^2}\right| \le \dfrac{1}{n^2}$，$\displaystyle\sum \dfrac{1}{n^2}$ 收敛，故一致收敛；
（2）由一致收敛推出和函数 $S(x) = \displaystyle\sum_{n=1}^\infty \dfrac{\sin nx}{n^2}$ 在 $\mathbb{R}$ 上连续；
（3）逐项积分 $\displaystyle\int_0^\pi S(x)\,dx = \displaystyle\sum_{n=1}^\infty \dfrac{1-\cos(n\pi)}{n^3} = 2\displaystyle\sum_{k=0}^\infty \dfrac{1}{(2k+1)^3} = \dfrac{7}{4}\zeta(3)$（Apéry 常数，给出计算过程）。

---

## 分组 5：多元微积分（E.37–E.48，共 12 题）

> 对应 Ch.18–22，覆盖 Green / Gauss / Stokes 定理、重积分换序与换元、Lagrange 乘子、梯度与方向导数等核心专题。
> 主要技巧：Green 定理（曲线 → 面积）、Gauss 定理（通量 → 散度）、Stokes 定理（环量 → 旋度）、极坐标 / 球坐标换元、Jacobi 行列式。

**E.37** [提升] Ch.22 [Green 定理 + 曲线积分为零]

设 $L$ 是 $xOy$ 平面内的简单闭曲线（正向）。证明 $\displaystyle\oint_L (y\,dx + x\,dy) = 0$。

（1）用 Green 定理 $\displaystyle\oint_L(P\,dx + Q\,dy) = \displaystyle\iint_D\!\left(\dfrac{\partial Q}{\partial x} - \dfrac{\partial P}{\partial y}\right)dA$，计算 $\dfrac{\partial Q}{\partial x} - \dfrac{\partial P}{\partial y}$；
（2）说明被积区域 $D$ 上 $Q_x - P_y \equiv 0$，从而积分为零；
（3）直接验证：$y\,dx + x\,dy = d(xy)$，故积分 $= [xy]_{\text{起}}^{\text{终}} = 0$（闭曲线起终相同），与 Green 方法对照。

**E.38** [提升] Ch.22 [Green 定理 + 面积公式]

用 Green 定理计算椭圆 $\dfrac{x^2}{a^2} + \dfrac{y^2}{b^2} = 1$（正向）的曲线积分 $\displaystyle\oint_L x\,dy$，并由此得出椭圆面积公式 $S = \pi ab$。

（1）回顾面积公式 $S = \displaystyle\oint_L x\,dy = -\displaystyle\oint_L y\,dx = \dfrac{1}{2}\displaystyle\oint_L(x\,dy - y\,dx)$；
（2）参数化 $x = a\cos\theta$，$y = b\sin\theta$（$\theta: 0 \to 2\pi$），计算 $\displaystyle\oint_L x\,dy$；
（3）验证三种面积公式给出相同结果 $\pi ab$。

**E.39** [提升] Ch.19–20 [二重积分换序]

计算 $\displaystyle\int_0^1\!\int_x^1 e^{y^2}\,dy\,dx$（内层积分 $\int e^{y^2}dy$ 无初等原函数）。

（1）画出积分区域（直角三角形），写出等价的先积 $x$ 后积 $y$ 的表达式；
（2）换序后计算 $\displaystyle\int_0^1\!\int_0^y e^{y^2}\,dx\,dy = \displaystyle\int_0^1 y e^{y^2}\,dy$；
（3）完成计算得 $\dfrac{e-1}{2}$，并总结"内层无初等原函数 → 考虑换序"的判断思路。

**E.40** [提升] Ch.19 [极坐标 + 二重积分]

计算 $\displaystyle\iint_{x^2+y^2\le R^2} e^{-(x^2+y^2)}\,dA$，并由此推出高斯积分 $\displaystyle\int_{-\infty}^{+\infty} e^{-x^2}\,dx = \sqrt{\pi}$。

（1）极坐标变换 $x = r\cos\theta$，$y = r\sin\theta$，计算 $\displaystyle\iint = \displaystyle\int_0^{2\pi}\!\!\int_0^R e^{-r^2}r\,dr\,d\theta = \pi(1 - e^{-R^2})$；
（2）令 $R \to +\infty$，得 $\displaystyle\iint_{\mathbb{R}^2} e^{-(x^2+y^2)}\,dA = \pi$；
（3）由 Fubini 定理 $\pi = \left(\displaystyle\int_{-\infty}^{+\infty} e^{-x^2}\,dx\right)^2$，推出 $\displaystyle\int_{-\infty}^{+\infty} e^{-x^2}\,dx = \sqrt{\pi}$。

**E.41** [提升] Ch.20 [三重积分 + 球坐标]

计算 $\displaystyle\iiint_{\Omega}(x^2 + y^2 + z^2)\,dV$，其中 $\Omega: x^2 + y^2 + z^2 \le R^2$。

（1）球坐标 $x = r\sin\phi\cos\theta$，$y = r\sin\phi\sin\theta$，$z = r\cos\phi$，$dV = r^2\sin\phi\,dr\,d\phi\,d\theta$；
（2）积分 $= \displaystyle\int_0^{2\pi}\!\!\int_0^\pi\!\!\int_0^R r^2 \cdot r^2\sin\phi\,dr\,d\phi\,d\theta = \dfrac{4\pi R^5}{5}$；
（3）由此写出球的转动惯量（质量均匀分布）：$I = \dfrac{2MR^2}{5}$（$M$ 为总质量）。

**E.42** [提升] Ch.18–19 [Lagrange 乘子法]

在约束 $g(x,y,z) = x + y + z - 1 = 0$ 下，求 $f(x,y,z) = xyz$ 的最大值（$x, y, z > 0$）。

（1）写出 Lagrange 条件 $\nabla f = \lambda \nabla g$，即 $yz = \lambda$，$xz = \lambda$，$xy = \lambda$；
（2）由三个方程推出 $x = y = z = 1/3$，计算 $f_{\max} = 1/27$；
（3）由此推出三元 AM-GM 不等式 $\dfrac{x+y+z}{3} \ge (xyz)^{1/3}$（$x,y,z > 0$），并指出 Lagrange 方法的一般框架。

**E.43** [提升] Ch.18 [方向导数 + 梯度最速上升]

设 $f(x,y) = x^2 - y^2 + 2xy$，点 $P = (1, -1)$。

（1）计算梯度 $\nabla f(P) = (f_x, f_y)\big|_P$；
（2）求 $f$ 沿单位方向 $\mathbf{l} = (\cos\alpha, \sin\alpha)$ 的方向导数 $D_{\mathbf{l}}f(P)$，指出使方向导数最大的方向及最大值；
（3）说明梯度方向是函数值增长最快的方向，这是梯度下降算法的数学基础：写出梯度下降迭代 $(x_{n+1}, y_{n+1}) = (x_n, y_n) - \eta \nabla f(x_n, y_n)$，并讨论步长 $\eta$ 的选取。

**E.44** [提升] Ch.22 [Gauss 定理 + 通量计算]

设 $\mathbf{F}(x,y,z) = (x, y, z)$，$\Sigma$ 为单位球面 $x^2+y^2+z^2=1$ 的外侧。计算 $\displaystyle\iint_\Sigma \mathbf{F}\cdot d\mathbf{S}$。

（1）计算散度 $\mathrm{div}\,\mathbf{F} = \dfrac{\partial x}{\partial x} + \dfrac{\partial y}{\partial y} + \dfrac{\partial z}{\partial z} = 3$；
（2）用 Gauss 定理 $\displaystyle\iint_\Sigma \mathbf{F}\cdot d\mathbf{S} = \displaystyle\iiint_\Omega \mathrm{div}\,\mathbf{F}\,dV = 3V_{\text{球}} = 4\pi$；
（3）直接验证：球面上 $\mathbf{F} \cdot \mathbf{n} = (x,y,z)\cdot(x,y,z) = 1$，故通量 $= 1 \times 4\pi = 4\pi$（与 Gauss 定理一致）。

**E.45** [提升] Ch.22 [Stokes 定理]

设 $\mathbf{F} = (y, z, x)$，$\Gamma$ 是平面 $x + y + z = 1$ 与第一卦限坐标平面的交线（正向）。用 Stokes 定理计算 $\displaystyle\oint_\Gamma \mathbf{F}\cdot d\mathbf{r}$。

（1）计算旋度 $\mathrm{rot}\,\mathbf{F} = \nabla \times \mathbf{F}$；
（2）取曲面 $\Sigma$：$x+y+z=1$（第一卦限三角形，法向朝上），用 Stokes 定理化为面积分；
（3）计算面积分，得出 $\displaystyle\oint_\Gamma \mathbf{F}\cdot d\mathbf{r} = -\dfrac{3}{2}\cdot\dfrac{\sqrt{3}}{2}\cdot\dfrac{1}{2}\cdot(-1) = \cdots$（给出过程与结果）。

**E.46** [提升] Ch.18–19 [隐函数 + Jacobi 行列式]

设映射 $F: (u,v) \mapsto (x, y)$ 由 $\begin{cases}x = u\cos v - v\sin u \\ y = u\sin v + v\cos u\end{cases}$ 给出。

（1）在点 $(u,v) = (1,0)$ 处计算 Jacobi 矩阵 $\dfrac{\partial(x,y)}{\partial(u,v)}$；
（2）验证 $\det J \neq 0$，说明隐函数定理给出局部逆映射的存在性；
（3）利用 $\det J$ 给出二重积分换元公式：$\displaystyle\iint_D f(x,y)\,dx\,dy = \displaystyle\iint_{D'} f(x(u,v), y(u,v))\left|\det\dfrac{\partial(x,y)}{\partial(u,v)}\right|du\,dv$。

**E.47** [提升] Ch.21–22 [曲面积分 + 参数化]

设 $S$ 是球面 $x^2+y^2+z^2=4$（$z \ge 0$，上半球面），方向向外，$f(x,y,z) = z$。计算 $\displaystyle\iint_S f\,dS$。

（1）参数化：$x = 2\sin\phi\cos\theta$，$y = 2\sin\phi\sin\theta$，$z = 2\cos\phi$（$\phi\in[0,\pi/2]$，$\theta\in[0,2\pi]$），$dS = 4\sin\phi\,d\phi\,d\theta$；
（2）计算 $\displaystyle\iint_S z\,dS = \displaystyle\int_0^{2\pi}\!\!\int_0^{\pi/2} 2\cos\phi \cdot 4\sin\phi\,d\phi\,d\theta$；
（3）完成积分得 $4\pi$，并用"$\displaystyle\iint_S z\,dS = \bar{z} \cdot \mathrm{Area}(S)$"（质心公式）交叉验证。

**E.48** [提升] Ch.18–22 [混合型：含参曲线积分 + 凑全微分]

计算曲线积分 $\displaystyle\int_L \frac{x\,dy - y\,dx}{x^2 + y^2}$，其中 $L$ 是从 $(1,0)$ 沿逆时针圆弧到 $(-1,0)$ 的上半圆。

（1）验证 $P = -y/(x^2+y^2)$，$Q = x/(x^2+y^2)$，计算 $Q_x - P_y$；
（2）注意 $Q_x - P_y = 0$（$x^2+y^2 \neq 0$），但区域含原点故 Green 定理需补充；
（3）参数化圆弧 $x = \cos t$，$y = \sin t$（$t: 0 \to \pi$）直接计算，结果为 $\pi$；说明为何 $\dfrac{x\,dy - y\,dx}{x^2+y^2} = d(\arctan(y/x))$ 在去掉原点的区域中成立。

---

## 分组 6：ODE（E.49–E.52，共 4 题）

> 对应 Ch.23–24，覆盖二阶常系数 ODE 共振情形、常数变易法、Bernoulli 方程、系统 ODE 等核心专题。
> 主要技巧：特征方程、共振（待定系数修正）、常数变易法、Bernoulli 变换、Wronskian 行列式。

**E.49** [提升] Ch.24 [二阶常系数 ODE + 共振]

求 $y'' + 4y = \sin(2t)$ 的通解（共振情形）。

（1）特征方程 $r^2 + 4 = 0$，根 $r = \pm 2i$，齐次通解 $y_h = C_1\cos 2t + C_2 \sin 2t$；
（2）右端 $\sin(2t)$ 对应的频率 $2$ 恰好是特征频率，待定特解形式为 $y_p = t(A\cos 2t + B\sin 2t)$（乘以 $t$）；
（3）代入方程确定 $A, B$，写出完整通解；讨论共振的物理意义（振幅随时间线性增大）。

**E.50** [提升] Ch.23–24 [常数变易法 + 非标准右端]

求 $y'' + y = \sec t$（$|t| < \pi/2$）的通解。

（1）齐次通解 $y_h = C_1\cos t + C_2\sin t$，Wronskian $W = 1$；
（2）常数变易法：设 $y_p = u_1(t)\cos t + u_2(t)\sin t$，由方程组 $u_1'\cos t + u_2'\sin t = 0$，$-u_1'\sin t + u_2'\cos t = \sec t$ 解出 $u_1'$，$u_2'$；
（3）积分得 $u_1 = \ln|\cos t|$，$u_2 = t$，特解 $y_p = \cos t\ln|\cos t| + t\sin t$；写出完整通解。

**E.51** [提升] Ch.23 [Bernoulli 方程]

求 $y' + y = y^2 e^x$（Bernoulli 方程，$n=2$）的通解。

（1）令 $v = y^{1-2} = y^{-1}$，计算 $v' = -y^{-2}y'$；
（2）将方程转化为关于 $v$ 的线性方程 $v' - v = -e^x$；
（3）用积分因子 $\mu = e^{-x}$ 求解 $v$，回代 $y = 1/v$，写出通解（注意 $y \equiv 0$ 也是解）。

**E.52** [提升] Ch.23–24 [ODE 建模 + 初值问题]

人口增长的 Logistic 模型：$\dfrac{dN}{dt} = rN\!\left(1 - \dfrac{N}{K}\right)$，初始条件 $N(0) = N_0$（$0 < N_0 < K$）。

（1）分离变量：$\dfrac{dN}{N(1-N/K)} = r\,dt$，用部分分式 $\dfrac{1}{N} + \dfrac{1/K}{1-N/K}$ 积分；
（2）得通解 $N(t) = \dfrac{K N_0 e^{rt}}{K - N_0 + N_0 e^{rt}}$，验证初始条件与 $N(\infty) = K$；
（3）求 $N'(t)$ 最大值（增长最快时刻），证明发生在 $N = K/2$ 时，对应 $t^* = \dfrac{1}{r}\ln\dfrac{K-N_0}{N_0}$。

---

## 分组 7：AI 微积分（E.53–E.60，共 8 题）

> 对应 Ch.25–28，覆盖 softmax 偏导、自动微分、KL 散度、KKT 条件、Lagrange 乘子优化、Itô 公式等 AI 工程核心数学。
> 主要技巧：矩阵微积分、反向传播链式法则、凸对偶、随机微积分。

**E.53** [提升] Ch.26 [矩阵微积分 + softmax 偏导]

设 softmax 函数 $\sigma(\mathbf{z})_i = \dfrac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}$（$\mathbf{z} \in \mathbb{R}^n$）。

（1）计算 $\dfrac{\partial \sigma_i}{\partial z_k}$：分两种情况（$i = k$ 与 $i \neq k$）用商法则推导；
（2）证明 $\dfrac{\partial \sigma_i}{\partial z_k} = \sigma_i(\delta_{ik} - \sigma_k)$（Kronecker $\delta$），写成矩阵形式 $J_\sigma = \mathrm{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma}\boldsymbol{\sigma}^T$；
（3）验证 $J_\sigma$ 是半负定的（对任意 $\mathbf{v}$，$\mathbf{v}^T J_\sigma \mathbf{v} \le 0$），这对应 softmax 的"竞争性"——增大某个分量必然减小其他分量之和。

**E.54** [提升] Ch.26 [反向传播 + 链式法则]

设神经网络一层计算 $\mathbf{y} = \sigma(W\mathbf{x} + \mathbf{b})$（$W \in \mathbb{R}^{m\times n}$，$\sigma$ 逐元素激活函数）。损失 $L$ 关于 $\mathbf{y}$ 的梯度 $\partial L/\partial \mathbf{y}$ 已知。

（1）用链式法则写出 $\partial L/\partial W_{ij}$ 的表达式（用 $\partial L/\partial y_i$，$\sigma'$，$x_j$ 表示）；
（2）写成矩阵形式：$\dfrac{\partial L}{\partial W} = \mathrm{diag}(\sigma'(W\mathbf{x}+\mathbf{b})) \cdot \dfrac{\partial L}{\partial \mathbf{y}} \cdot \mathbf{x}^T$（验证维度）；
（3）给出 $\partial L/\partial \mathbf{x}$ 的表达式（用于将梯度传播到前一层），说明"反向传播"名称的由来。

**E.55** [提升] Ch.25–28 [KL 散度 $\ge 0$ + Jensen 不等式]

设 $p, q$ 是离散概率分布（$p_i, q_i > 0$，$\sum p_i = \sum q_i = 1$）。证明 KL 散度 $D_\mathrm{KL}(p\|q) = \displaystyle\sum_i p_i \ln\dfrac{p_i}{q_i} \ge 0$，等号当且仅当 $p = q$。

（1）等价地证明 $-D_\mathrm{KL}(p\|q) = \displaystyle\sum_i p_i \ln\dfrac{q_i}{p_i} \le 0$；
（2）注意 $\ln$ 是凹函数，由 Jensen 不等式 $\sum p_i \ln(q_i/p_i) \le \ln\!\left(\sum p_i \cdot q_i/p_i\right) = \ln 1 = 0$；
（3）讨论等号条件：$q_i/p_i$ 为常数 $\Rightarrow q = p$；由此说明 KL 散度不是真正的"距离"（不对称），但 $D_\mathrm{KL}(p\|q) + D_\mathrm{KL}(q\|p) \ge 0$ 恒成立。

**E.56** [提升] Ch.25 [KKT 条件 + 等式约束优化]

最小化 $f(\mathbf{x}) = \|\mathbf{x}\|^2 = \displaystyle\sum_i x_i^2$，约束 $\mathbf{a}^T\mathbf{x} = b$（$\mathbf{a} \neq \mathbf{0}$，$b \in \mathbb{R}$）。

（1）写出 Lagrange 函数 $\mathcal{L}(\mathbf{x}, \lambda) = \|\mathbf{x}\|^2 - \lambda(\mathbf{a}^T\mathbf{x} - b)$；
（2）由 KKT 条件 $\nabla_\mathbf{x}\mathcal{L} = 0$ 得 $\mathbf{x}^* = \dfrac{\lambda}{2}\mathbf{a}$，代入约束解出 $\lambda = \dfrac{2b}{\|\mathbf{a}\|^2}$；
（3）最优解 $\mathbf{x}^* = \dfrac{b}{\|\mathbf{a}\|^2}\mathbf{a}$，最小值 $f^* = \dfrac{b^2}{\|\mathbf{a}\|^2}$；这就是向量到超平面的距离公式 $d = |b|/\|\mathbf{a}\|$ 的代数证明。

**E.57** [提升] Ch.25–26 [凸函数 + 次梯度]

设 $f(\mathbf{x}) = \|\mathbf{x}\|_1 = \displaystyle\sum_i |x_i|$（$L_1$ 范数，用于 Lasso 正则化）。

（1）证明 $f$ 是凸函数（用 $L_1$ 范数的三角不等式）；
（2）$f$ 在 $x_i = 0$ 处不可微；定义次梯度（subgradient）：$g \in \partial f(\mathbf{x})$ 满足 $f(\mathbf{y}) \ge f(\mathbf{x}) + g^T(\mathbf{y}-\mathbf{x})$。求 $\partial f(\mathbf{x})$ 的表达式（$\mathrm{sign}(x_i)$ 或区间 $[-1,1]$）；
（3）Lasso 最优性条件：$\mathbf{0} \in \nabla_\mathbf{x}(\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2) + \lambda \partial\|\mathbf{w}\|_1$，解释软阈值（soft-thresholding）算子 $\mathbf{w}_i^* = \mathrm{sgn}(z_i)\max(|z_i|-\lambda/2, 0)$ 的来源。

**E.58** [提升] Ch.26 [自动微分 + 计算复杂度]

比较前向模式（forward mode）与反向模式（backward mode）自动微分的计算复杂度。

（1）设函数 $f: \mathbb{R}^n \to \mathbb{R}^m$，计算 Jacobian $J \in \mathbb{R}^{m\times n}$。前向模式：每次传播一个方向导数向量，计算一列 Jacobian 需 $O(T)$ 时间（$T$ 为计算图边数），全 Jacobian 需 $O(nT)$；
（2）反向模式：每次传播一个余切向量（行梯度），计算一行 Jacobian 需 $O(T)$，全 Jacobian 需 $O(mT)$；
（3）结论：当 $m \ll n$（如损失函数 $m=1$），反向模式（backprop）只需 $O(T)$ 即得全梯度；这是深度学习高效训练的核心原因。用具体的 2 层网络例子（$n = 10^6$，$m = 1$）数值对比两种模式的计算量。

**E.59** [提升] Ch.28 [Itô 公式 + 布朗运动]

设 $B_t$ 是标准布朗运动（$B_0 = 0$，$dB_t$ 满足 Itô 等距 $E[(dB_t)^2] = dt$）。

（1）直接用二阶 Taylor 展开"启发性推导" $d(B_t^2)$：$d(B_t^2) \approx 2B_t\,dB_t + \dfrac{1}{2}\cdot 2(dB_t)^2 = 2B_t\,dB_t + dt$（关键：$(dB_t)^2 = dt$，不可忽略）；
（2）写出 Itô 公式的一般形式：若 $X_t$ 满足 SDE，$f \in C^2$，则 $df(X_t) = f'(X_t)\,dX_t + \dfrac{1}{2}f''(X_t)\sigma^2\,dt$（$\sigma^2$ 为扩散系数）；
（3）对 $B_t^2$ 两边积分：$B_t^2 = 2\displaystyle\int_0^t B_s\,dB_s + t$，由此得 $\displaystyle\int_0^t B_s\,dB_s = \dfrac{B_t^2 - t}{2}$（Itô 随机积分的精确结果，与普通积分 $\int_0^t x\,dx = x^2/2$ 相差 $-t/2$）。

**E.60** [提升] Ch.25–28 [信息熵 + 最大熵原理]

设离散分布 $p = (p_1, \ldots, p_n)$（$p_i > 0$，$\sum p_i = 1$）。Shannon 熵 $H(p) = -\displaystyle\sum_i p_i \ln p_i$。

（1）用 Lagrange 乘子法在约束 $\sum p_i = 1$ 下最大化 $H(p)$，得最优解 $p_i^* = 1/n$，最大熵 $H^* = \ln n$；
（2）由 KL 散度非负性 $D_\mathrm{KL}(p\|q) \ge 0$（E.55），取 $q_i = 1/n$（均匀分布）直接推出 $H(p) \le \ln n$；
（3）讨论最大熵原理的 AI 含义：在已知约束（如均值、方差）下选择最大熵分布等价于"最少假设"先验；以均值约束 $\sum p_i x_i = \mu$ 为例，说明最大熵分布是指数族分布（Gibbs 分布）。

---

> **题号 / 分组分布索引**
>
> | 分组 | 主题 | 题数 | 编号范围 |
> |------|------|------|----------|
> | 分组 1 | 极限连续（$\varepsilon$-$\delta$ / Taylor / Riemann 和 / 间断点）| 6 | E.01–E.06 |
> | 分组 2 | 微分应用（凸性 / Jensen / Newton 迭代 / 中值定理 / Taylor 余项）| 12 | E.07–E.18 |
> | 分组 3 | 积分技巧（万能代换 / 分部循环 / Wallis / 广义积分 / Gamma）| 10 | E.19–E.28 |
> | 分组 4 | 级数（Leibniz / Weierstrass / Fourier / Parseval / 幂级数）| 8 | E.29–E.36 |
> | 分组 5 | 多元微积分（Green / Gauss / Stokes / Lagrange / Jacobi）| 12 | E.37–E.48 |
> | 分组 6 | ODE（共振 / 常数变易 / Bernoulli / Logistic）| 4 | E.49–E.52 |
> | 分组 7 | AI 微积分（softmax / 反向传播 / KL 散度 / KKT / Itô / 最大熵）| 8 | E.53–E.60 |
> | **合计** | | **60** | **E.01–E.60** |
>
> **难度说明：** 分组 1–6 对应考研数学压轴档（数一 / 数二），分组 7 对应 AI 工程师必备微积分应用。
> 关键技巧标签覆盖：$\varepsilon$-$\delta$ 证明、Taylor 展开、Riemann 和、凸性证、Jensen 不等式、Newton 迭代、万能代换、Wallis、Gamma 函数、Dirichlet 积分、Green / Gauss / Stokes、Lagrange 乘子、KKT、softmax 偏导、反向传播、KL 散度、Itô 公式、最大熵原理等。
