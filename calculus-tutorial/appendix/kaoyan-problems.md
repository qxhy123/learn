# 附录：历年考研真题精选 100 题

本附录精选 100 道考研数学（数一/数二/数三）风格真题，覆盖极限、连续与可导、一元微分学、微分中值定理与应用、不定积分、定积分、广义积分与积分应用、多元函数微分学、重积分与曲线曲面积分、级数、常微分方程等核心考点。**每题给出"思路"段（说明识别题型与选择方法的动机）与"解"段（完整计算）。**

> **使用建议**：先遮住「解」独立尝试，做不出时只看「思路」获得提示；做完后再核对「解」复盘。

---

## 一、极限（题 1–15）

**题 1**（数一/数二）求 $\displaystyle\lim_{x\to 0}\frac{\sqrt{1+2x}-\sqrt[3]{1+3x}}{x^2}$。

**思路**：分子是两个根式之差。在 $x\to 0$ 时如果只展到一阶（$\sqrt{1+2x}\approx 1+x$、$\sqrt[3]{1+3x}\approx 1+x$）会得到 $0/x^2$ 不定型——说明一阶项相消，需展到 $x^2$ 项。Taylor 展到二阶是最稳的做法；用 L'Hôpital 两次也行，但展开更快。

**解**：

$$
\sqrt{1+2x}=1+x-\tfrac{x^2}{2}+o(x^2),\quad
\sqrt[3]{1+3x}=1+x-x^2+o(x^2).
$$

相减：$\sqrt{1+2x}-\sqrt[3]{1+3x}=\tfrac{x^2}{2}+o(x^2)$，故极限为 $\boxed{\tfrac12}$。

---

**题 2**（数三）求 $\displaystyle\lim_{x\to 0}\frac{e^x-e^{\sin x}}{x-\sin x}$。

**思路**：分母 $x-\sin x$ 是经典三阶小量（$\sim x^3/6$），分子也是无穷小。看到 $e^a-e^b=e^b(e^{a-b}-1)$ 这一恒等技巧——它能把指数差换成 $a-b$ 的乘积，恰好凑出与分母一致的 $x-\sin x$，从而约掉。

**解**：分子 $=e^{\sin x}(e^{x-\sin x}-1)\sim e^{\sin x}\cdot(x-\sin x)$。极限 $=\displaystyle\lim_{x\to 0}e^{\sin x}=\boxed{1}$。

---

**题 3** 求 $\displaystyle\lim_{x\to 0}\left(\frac{1+\tan x}{1+\sin x}\right)^{1/\sin^3 x}$。

**思路**：$1^\infty$ 不定型——标准做法是取对数，转为 $0\cdot\infty$ 或 $0/0$。取对后分子是两个 $\ln(1+\cdot)$ 之差，自然想到 $\ln(1+u)\sim u$，从而化为 $\tan x-\sin x$；再用恒等 $\tan x-\sin x=\tan x(1-\cos x)$ 算出阶数。

**解**：取对数 $L=\lim\dfrac{\ln(1+\tan x)-\ln(1+\sin x)}{\sin^3 x}$。分子 $\sim\tan x-\sin x=\tan x(1-\cos x)\sim x\cdot\tfrac{x^2}{2}=\tfrac{x^3}{2}$；分母 $\sim x^3$。故 $L=\tfrac12$，原极限 $=\boxed{e^{1/2}}$。

---

**题 4** 求 $\displaystyle\lim_{n\to\infty}\sum_{k=1}^n\frac{1}{n+k}$。

**思路**：求和的项数随 $n$ 增长，逐项放缩没用。看到结构 $\dfrac{1}{n}\cdot f(k/n)$ 形式，立即想到 Riemann 和——把和式视为某个连续函数在 $[0,1]$ 的近似积分。

**解**：

$$
\sum_{k=1}^n\frac{1}{n+k}=\frac{1}{n}\sum_{k=1}^n\frac{1}{1+k/n}\to\int_0^1\frac{dx}{1+x}=\boxed{\ln 2}.
$$

---

**题 5** 求 $\displaystyle\lim_{x\to+\infty}\left(\sqrt{x^2+x+1}-\sqrt{x^2-x+1}\right)$。

**思路**：$\infty-\infty$ 型且含根式——条件反射是分子有理化。乘共轭后大头 $x^2$ 项相消，剩下的一次项 $2x$ 与分母 $\sim 2x$ 同阶。

**解**：

$$
=\lim\frac{(x^2+x+1)-(x^2-x+1)}{\sqrt{x^2+x+1}+\sqrt{x^2-x+1}}=\lim\frac{2x}{2x}=\boxed{1}.
$$

---

**题 6** 求 $\displaystyle\lim_{x\to 0}\frac{\ln(1+x)-x}{x^2}$。

**思路**：分子是 $\ln(1+x)$ 与一阶近似的差，所以必须展到二阶。这是 Taylor 展开最基础应用之一。

**解**：$\ln(1+x)=x-\tfrac{x^2}{2}+o(x^2)$，分子 $=-\tfrac{x^2}{2}+o(x^2)$，极限 $=\boxed{-\tfrac12}$。

---

**题 7** 求 $\displaystyle\lim_{x\to 0^+}x^x$。

**思路**：$0^0$ 不定型。指数中含变量时统一套路：写成 $e^{x\ln x}$，把指数极限单独算。指数 $x\ln x$ 是经典 $0\cdot(-\infty)$ 型，用 L'Hôpital 把它倒置为 $0/0$ 或 $\infty/\infty$。

**解**：$x^x=e^{x\ln x}$。$\lim_{x\to 0^+}x\ln x=\lim\dfrac{\ln x}{1/x}\stackrel{\text{L}}{=}\lim\dfrac{1/x}{-1/x^2}=\lim(-x)=0$。故极限 $=e^0=\boxed{1}$。

---

**题 8** 求 $\displaystyle\lim_{n\to\infty}\sqrt[n]{n!}/n$。

**思路**：直接的 $\sqrt[n]{n!}$ 难处理，但取对数后变成 $\dfrac{1}{n}\sum\ln k$，再除以 $n$ 后变成 $\dfrac{1}{n}\sum\ln(k/n)$——又是 Riemann 和形式，对应 $\int_0^1\ln x\,dx$。

**解**：令 $a_n=\sqrt[n]{n!}/n$，则 $\ln a_n=\dfrac{1}{n}\sum_{k=1}^n\ln\dfrac{k}{n}\to\int_0^1\ln x\,dx=-1$。极限 $=\boxed{1/e}$。

---

**题 9** 求 $\displaystyle\lim_{x\to 0}\frac{(1+x)^{1/x}-e}{x}$。

**思路**：$(1+x)^{1/x}\to e$ 是已知极限，本题求其趋于 $e$ 的速度（一阶修正项）。把 $(1+x)^{1/x}$ 写成 $e^{g(x)}$ 后，在 $g(x)=1$ 附近做 Taylor。先算 $g(x)=\dfrac{\ln(1+x)}{x}=1-\tfrac x2+o(x)$，于是 $(1+x)^{1/x}=e^{g(x)}=e\cdot e^{g(x)-1}=e(1+(g-1)+o(g-1))$。

**解**：设 $g(x)=\dfrac{\ln(1+x)}{x}$，$g(x)-1\sim -\tfrac x2$。故 $(1+x)^{1/x}-e=e\bigl(e^{g-1}-1\bigr)\sim e(g-1)\sim -\tfrac{ex}{2}$。极限 $=\boxed{-\tfrac e2}$。

---

**题 10** 求 $\displaystyle\lim_{x\to 0}\frac{\arctan x-x}{x^3}$。

**思路**：与题 6 同类——三角/反三角函数与其一阶近似之差，展到三阶即可。需记忆 $\arctan x=x-x^3/3+x^5/5-\cdots$。

**解**：$\arctan x=x-\tfrac{x^3}{3}+o(x^3)$，极限 $=\boxed{-\tfrac13}$。

---

**题 11** 已知 $\lim_{x\to 0}\dfrac{f(x)}{x^2}=2$，求 $\lim_{x\to 0}\dfrac{f(\sin x)}{x^2}$。

**思路**：换元题。$f(\cdot)$ 的"自变量"从 $x$ 换成 $\sin x$，由 $\sin x\sim x$ 知二者趋零同阶，因此 $f(\sin x)\sim 2\sin^2 x\sim 2x^2$。

**解**：$f(\sin x)\sim 2\sin^2 x$，$\dfrac{f(\sin x)}{x^2}\to 2\cdot 1=\boxed{2}$。

---

**题 12** 求 $\displaystyle\lim_{x\to 1}\frac{x-x^x}{1-x+\ln x}$。

**思路**：$x\to 1$ 的不定型，令 $t=x-1\to 0$ 化为常见形式。分子 $x-x^x=x(1-x^{x-1})$，再用 $x^{x-1}=e^{(x-1)\ln x}\approx 1+(x-1)\ln x$；分母直接代 $\ln(1+t)$ 的 Taylor 展。

**解**：分子 $=x\bigl(1-e^{(x-1)\ln x}\bigr)\sim -x(x-1)\ln x\sim -(x-1)^2$（用 $\ln x\sim x-1$）。分母 $1-x+\ln x=-t+(t-\tfrac{t^2}{2}+o(t^2))=-\tfrac{t^2}{2}+o(t^2)$。极限 $=\dfrac{-t^2}{-t^2/2}=\boxed{2}$。

---

**题 13** 求 $\displaystyle\lim_{n\to\infty}n\left(\sqrt[n]{a}-1\right)$（$a>0$）。

**思路**：$\sqrt[n]a=a^{1/n}=e^{(\ln a)/n}$。$1/n$ 是无穷小，用 $e^u-1\sim u$ 立刻得阶。

**解**：$\sqrt[n]a-1=e^{(\ln a)/n}-1\sim\dfrac{\ln a}{n}$。极限 $=\boxed{\ln a}$。

---

**题 14** 求 $\displaystyle\lim_{n\to\infty}\left(\frac{1}{n^2}+\frac{2}{n^2}+\cdots+\frac{n}{n^2}\right)$。

**思路**：分子直接求和（等差数列），不需要 Riemann 和那么复杂。

**解**：$=\dfrac{1+2+\cdots+n}{n^2}=\dfrac{n(n+1)}{2n^2}\to\boxed{\tfrac12}$。

---

**题 15** 设 $f(x)$ 连续且 $f(0)=0$、$f'(0)=2$，求 $\displaystyle\lim_{x\to 0}\dfrac{\int_0^{x^2}f(t)\,dt}{x^4}$。

**思路**：典型变限积分 $0/0$ 型，首选 L'Hôpital + 变限积分求导公式 $\dfrac{d}{dx}\int_0^{u(x)}f=f(u(x))u'(x)$。求导后剩下的极限再用 $f(t)/t\to f'(0)$。

**解**：

$$
\lim\frac{f(x^2)\cdot 2x}{4x^3}=\frac12\lim\frac{f(x^2)}{x^2}=\frac12 f'(0)=\boxed{1}.
$$

---

## 二、连续与可导（题 16–22）

**题 16** 设 $f(x)=\begin{cases}\dfrac{\ln(1+ax)}{x},& x>0\\ b,& x=0\\ \dfrac{e^{x}-1}{x},& x<0\end{cases}$ 在 $x=0$ 连续，求 $a,b$。

**思路**：分段函数在分段点连续 $\Leftrightarrow$ 左极限 = 右极限 = 函数值。利用 $\ln(1+ax)\sim ax$ 和 $e^x-1\sim x$ 直接算出两侧极限。

**解**：右极限 $\lim_{x\to 0^+}\dfrac{ax}{x}=a$；左极限 $\lim_{x\to 0^-}\dfrac{x}{x}=1$。连续要求 $a=b=1$。

---

**题 17** 讨论 $f(x)=\lim_{n\to\infty}\dfrac{x^{2n}-1}{x^{2n}+1}x$ 的连续性。

**思路**：定义本身是一个极限，关键是按 $|x|$ 与 $1$ 的大小分类讨论 $x^{2n}$ 的行为：$|x|>1$ 时 $x^{2n}\to\infty$，分式趋于 $1$；$|x|<1$ 时分式趋于 $-1$；$|x|=1$ 单独算。然后看分段点是否连续。

**解**：

- $|x|>1$：$f(x)=x$；
- $|x|<1$：$f(x)=-x$；
- $|x|=1$：$f(\pm1)=0$。

$x=1$ 处左极限 $-1$、右极限 $1$，不连续（跳跃间断）；$x=-1$ 同理。其余处连续。

---

**题 18** 设 $f(x)=|x-1|\cdot|x+1|$，问 $f$ 在哪些点不可导？

**思路**：$|u|$ 在 $u=0$ 处出现尖角，若 $u$ 本身在 $0$ 不变号则尖角"消失"。这里 $f=|x^2-1|$，在 $x=\pm1$ 处 $x^2-1$ 过零变号，故是尖角，不可导。

**解**：$f(x)=|x^2-1|$。$x=\pm 1$ 处左右导数为 $\mp 2,\pm 2$，不相等，故**$x=\pm 1$ 不可导**；其余点可导。

---

**题 19** 已知 $f(x)$ 在 $x=0$ 可导且 $f(0)=0,\ f'(0)=a$（$a\ne 0$）。求 $\displaystyle\lim_{x\to 0}\dfrac{f(x^2)}{x f(x)}$。

**思路**：分子分母都是 $0$。可导的最直接刻画 $f(x)\sim f'(0)x=ax$。代换得分子 $\sim ax^2$、分母 $\sim x\cdot ax=ax^2$。

**解**：极限 $=\dfrac{ax^2}{ax^2}\to\boxed{1}$。

---

**题 20** 设 $f(x)=\begin{cases}x^2\sin\tfrac1x,& x\ne 0\\ 0,& x=0\end{cases}$，求 $f'(0)$ 并讨论 $f'$ 在 $0$ 的连续性。

**思路**：这是经典例题——"在一点可导但导函数在该点不连续"。$f'(0)$ 用定义算（夹逼）；$x\ne 0$ 处直接求导后看 $x\to 0$ 时是否有极限。

**解**：$f'(0)=\lim_{h\to 0}h\sin\tfrac1h=0$（夹逼）。$x\ne 0$ 时 $f'(x)=2x\sin\tfrac1x-\cos\tfrac1x$，$\cos(1/x)$ 在 $x\to 0$ 无极限，故 $f'$ 在 $0$ **不连续**。

---

**题 21** 设 $f$ 二阶可导且 $f(0)=0,\ f'(0)=1,\ f''(0)=2$，求 $\displaystyle\lim_{x\to 0}\dfrac{f(x)-x}{x^2}$。

**思路**：分子 $f(x)-x$ 把一阶项扣掉了，剩下的就是二阶 Taylor 余项。直接展开 $f(x)=f(0)+f'(0)x+\tfrac{f''(0)}{2}x^2+o(x^2)$ 一步得结果。

**解**：$f(x)=x+x^2+o(x^2)$，故 $f(x)-x\sim x^2$，极限 $=\boxed{1}$。

---

**题 22** 证明：若 $f$ 在 $[a,b]$ 连续、$(a,b)$ 可导且 $f'$ 单调，则 $f'$ 在 $(a,b)$ 连续。

**思路**：单调函数只能有跳跃间断点。但是导函数有 Darboux 介值性（即使不连续也满足介值定理），跳跃会跳过中间值，与 Darboux 矛盾。

**解**：设 $f'$ 在 $c$ 跳跃，则左右极限 $f'(c^-)\ne f'(c^+)$。由单调性 $f'$ 在 $c$ 邻域取值跳过区间 $(f'(c^-),f'(c^+))$ 内某些值；但 Darboux 定理保证 $f'$ 取得任意介值，矛盾。故 $f'$ 连续。 $\square$

---

## 三、一元微分计算（题 23–30）

**题 23** 求 $y=(\sin x)^{\cos x}$ 的导数。

**思路**：底和指数都是变量——典型对数求导法。两边取 $\ln$ 后两边对 $x$ 求导，再乘回 $y$。

**解**：$\ln y=\cos x\ln\sin x$。求导：

$$
\frac{y'}{y}=-\sin x\ln\sin x+\cos x\cdot\cot x.
$$

$$
y'=(\sin x)^{\cos x}\bigl[\cot x\cos x-\sin x\ln\sin x\bigr].
$$

---

**题 24** 由方程 $e^y+xy=e$ 确定的隐函数 $y(x)$ 在 $x=0$ 处的 $y'$ 与 $y''$。

**思路**：隐函数求导——把 $y$ 视为 $x$ 的函数，方程两边对 $x$ 求导，解出 $y'$；再求导一次解出 $y''$。先由原方程定出 $x=0$ 时的 $y$ 值（代入得 $e^y=e$，$y=1$）。

**解**：$x=0\Rightarrow y=1$。一阶导：$e^y y'+y+xy'=0$，代入得 $y'(0)=-1/e$。

二阶导：$e^y(y')^2+e^y y''+2y'+xy''=0$。代入 $x=0,y=1,y'=-1/e$：$e\cdot 1/e^2+e y''-2/e=0\Rightarrow y''(0)=1/e^2$。

---

**题 25** 设 $\begin{cases}x=t-\sin t\\ y=1-\cos t\end{cases}$，求 $\dfrac{dy}{dx}$ 与 $\dfrac{d^2y}{dx^2}$。

**思路**：参数方程求导公式 $\dfrac{dy}{dx}=\dfrac{y'(t)}{x'(t)}$，二阶 $\dfrac{d^2y}{dx^2}=\dfrac{d}{dt}(\tfrac{dy}{dx})\big/x'(t)$。注意二阶不能简单地"分子分母分别二阶导"。

**解**：$\dfrac{dy}{dx}=\dfrac{\sin t}{1-\cos t}=\cot\dfrac{t}{2}$。

$\dfrac{d^2y}{dx^2}=\dfrac{-\frac12\csc^2(t/2)}{1-\cos t}=-\dfrac{1}{4\sin^4(t/2)}$（用 $1-\cos t=2\sin^2(t/2)$）。

---

**题 26** 求 $y=\dfrac{1}{x^2-3x+2}$ 的 $n$ 阶导数。

**思路**：有理函数高阶导首选部分分式——把它拆成 $\dfrac{1}{x-a}$ 之和，每项 $n$ 阶导有公式 $\left(\dfrac{1}{x-a}\right)^{(n)}=\dfrac{(-1)^n n!}{(x-a)^{n+1}}$。

**解**：$y=\dfrac{1}{x-2}-\dfrac{1}{x-1}$。

$$
y^{(n)}=(-1)^n n!\left[\frac{1}{(x-2)^{n+1}}-\frac{1}{(x-1)^{n+1}}\right].
$$

---

**题 27** 求 $f(x)=\ln(1+x)$ 在 $x=0$ 的 $n$ 阶 Taylor 展开（带 Lagrange 余项）。

**思路**：先归纳出 $f^{(k)}(0)$，再代入 Taylor 公式与余项公式。

**解**：$f^{(k)}(x)=\dfrac{(-1)^{k-1}(k-1)!}{(1+x)^k}$，故 $f^{(k)}(0)=(-1)^{k-1}(k-1)!$。

$$
\ln(1+x)=\sum_{k=1}^n\frac{(-1)^{k-1}x^k}{k}+\frac{(-1)^n x^{n+1}}{(n+1)(1+\xi)^{n+1}},\ \xi\in(0,x).
$$

---

**题 28** 设 $y=x\ln x$，求 $y^{(n)}$（$n\ge 2$）。

**思路**：低阶手算几次，发现规律后归纳。也可用 Leibniz：$y=x\cdot\ln x$，$x$ 的导数从 $n=2$ 起为 $0$，只剩两项。

**解**：$y'=\ln x+1$，$y''=1/x$。对 $n\ge 2$，$y^{(n)}=(1/x)^{(n-2)}=\dfrac{(-1)^{n-2}(n-2)!}{x^{n-1}}=\dfrac{(-1)^n(n-2)!}{x^{n-1}}$。

---

**题 29** 已知 $f$ 二阶可导，$g(x)=f(\sin x)$，求 $g''(x)$。

**思路**：链式法则用两次，注意第二次求导时 $f'(\sin x)$ 也是 $x$ 的复合函数。

**解**：$g'(x)=f'(\sin x)\cos x$。再求导（乘法 + 链式）：

$$
g''(x)=f''(\sin x)\cos^2 x-f'(\sin x)\sin x.
$$

---

**题 30** 设 $y=\arctan\dfrac{2x}{1-x^2}$，求 $y'$。

**思路**：直接对复合求导麻烦。注意 $\dfrac{2x}{1-x^2}=\tan(2\arctan x)$ 的形式，故 $y=2\arctan x$（在 $|x|<1$ 区间）；用恒等比硬算简单得多。

**解**：在 $|x|<1$ 时 $y=2\arctan x$，$y'=\dfrac{2}{1+x^2}$。（$|x|>1$ 时需加 $\pm\pi$ 修正，导数仍 $\dfrac{2}{1+x^2}$。）

---

## 四、微分中值定理与应用（题 31–40）

**题 31** 设 $f$ 在 $[0,1]$ 连续、$(0,1)$ 可导，$f(0)=0,\ f(1)=1$。证明 $\exists\xi\in(0,1)$ 使 $f'(\xi)=2\xi$。

**思路**：要证 $f'(\xi)-2\xi=0$，把它视为某个函数 $g(\xi)$ 的导数为零——令 $g(x)=f(x)-x^2$ 即可，$g$ 在 $0,1$ 处取相同值 $0$，套 Rolle。

**解**：$g(x)=f(x)-x^2$，$g(0)=g(1)=0$。Rolle 定理给出 $\xi$ 使 $g'(\xi)=0$，即 $f'(\xi)=2\xi$。 $\square$

---

**题 32** 设 $f\in C^2[a,b]$ 且 $f(a)=f(b)=0$。证明 $\exists\xi\in(a,b)$ 使 $f''(\xi)=\dfrac{2f(c)}{(c-a)(c-b)}$，其中 $c\in(a,b)$ 给定。

**思路**：要让 $f$ 与三个零点关联（$a,c,b$），构造一个二次多项式 $P(x)=K(x-a)(x-b)$ 使 $P(c)=f(c)$（解出 $K$），再令 $\varphi=f-P$，$\varphi$ 在 $a,c,b$ 处都为零；连用两次 Rolle 得到 $\varphi''(\xi)=0$。

**解**：取 $\varphi(x)=f(x)-f(c)\dfrac{(x-a)(x-b)}{(c-a)(c-b)}$，$\varphi(a)=\varphi(c)=\varphi(b)=0$。两次 Rolle 给出 $\varphi''(\xi)=0$，即 $f''(\xi)=\dfrac{2f(c)}{(c-a)(c-b)}$。 $\square$

---

**题 33** 证明：$x>0$ 时 $\ln(1+x)<x$。

**思路**：把不等式化为 $f(x)=x-\ln(1+x)>0$，验证 $f(0)=0$ 与 $f'>0$ 即可。这是把"不等式"变"单调性"的标准套路。

**解**：$f(x)=x-\ln(1+x)$，$f(0)=0$，$f'=1-\dfrac{1}{1+x}=\dfrac{x}{1+x}>0$（$x>0$）。故 $f$ 严格增，$f(x)>0$。 $\square$

---

**题 34** 证明：当 $0<x<\dfrac\pi2$ 时 $\dfrac{2x}{\pi}<\sin x<x$。

**思路**：拆成两个不等式。右半还是构造 $x-\sin x$ 验单调。左半 $\sin x>2x/\pi$ 等价于 $\dfrac{\sin x}{x}>\dfrac{2}{\pi}=\dfrac{\sin(\pi/2)}{\pi/2}$，提示证 $\dfrac{\sin x}{x}$ 在 $(0,\pi/2]$ 递减——求导验证。

**解**：右半：$(x-\sin x)'=1-\cos x\ge 0$，且 $x=0$ 处取 $0$，故 $x>\sin x$（$x>0$）。

左半：$g(x)=\dfrac{\sin x}{x}$，$g'=\dfrac{x\cos x-\sin x}{x^2}<0$（因 $\tan x>x$）。故 $g$ 递减，$g(\pi/2)=2/\pi<g(x)$。 $\square$

---

**题 35** 求 $f(x)=x^3-3x$ 在 $[-2,2]$ 上的最值。

**思路**：闭区间连续函数最值出现在驻点或端点。先求 $f'=0$ 找驻点。

**解**：$f'=3x^2-3=0\Rightarrow x=\pm 1$。$f(-2)=-2$，$f(-1)=2$，$f(1)=-2$，$f(2)=2$。最大值 $2$，最小值 $-2$。

---

**题 36** 求 $y=\dfrac{x}{1+x^2}$ 的凹凸区间与拐点。

**思路**：凹凸看 $y''$ 符号，拐点在 $y''$ 变号处。求导技巧：$\dfrac{x}{1+x^2}$ 求导用商法则。

**解**：$y'=\dfrac{1-x^2}{(1+x^2)^2}$，$y''=\dfrac{2x(x^2-3)}{(1+x^2)^3}$。$y''=0$ 在 $x=0,\pm\sqrt 3$。

符号：$(-\infty,-\sqrt 3)$ 负、$(-\sqrt 3,0)$ 正、$(0,\sqrt 3)$ 负、$(\sqrt 3,\infty)$ 正。对应**凹下、凹上、凹下、凹上**，三个拐点 $(0,0),(\pm\sqrt 3,\pm\sqrt 3/4)$。

---

**题 37** 求 $f(x)=e^x\sin x$ 在 $[0,2\pi]$ 上的最大、最小值。

**思路**：闭区间连续函数最值——驻点 + 端点。驻点条件 $f'=e^x(\sin x+\cos x)=0$ 即 $\tan x=-1$，落在第二、第四象限。

**解**：$x=3\pi/4$ 或 $7\pi/4$。$f(3\pi/4)=\tfrac{\sqrt 2}{2}e^{3\pi/4}>0$，$f(7\pi/4)=-\tfrac{\sqrt 2}{2}e^{7\pi/4}<0$。端点 $f(0)=f(2\pi)=0$。最大 $\tfrac{\sqrt 2}{2}e^{3\pi/4}$，最小 $-\tfrac{\sqrt 2}{2}e^{7\pi/4}$。

---

**题 38** 求 $\displaystyle\lim_{x\to 0}\dfrac{x-\sin x}{x^3}$。

**思路**：$0/0$ 型可反复 L'Hôpital 直到极限可计算；或一次性用 $\sin x$ 的 Taylor 展开。这里用 L'Hôpital 演示多次套用。

**解**：

$$
\lim\frac{1-\cos x}{3x^2}=\lim\frac{\sin x}{6x}=\boxed{\tfrac16}.
$$

---

**题 39** 设 $f$ 在 $[0,1]$ 连续、可导且 $f(0)=0,\ |f'(x)|\le M$。证明 $\int_0^1 f^2(x)\,dx\le \tfrac{M^2}{3}$。

**思路**：把 $f(x)$ 表为 $\int_0^x f'(t)\,dt$（用 $f(0)=0$），再 Cauchy-Schwarz 估计平方，最后积分得到目标上界。

**解**：$f^2(x)=\left(\int_0^x f'\,dt\right)^2\le x\int_0^x f'^2\,dt\le M^2 x^2$。积分：$\int_0^1 f^2\,dx\le M^2\int_0^1 x^2\,dx=M^2/3$。 $\square$

---

**题 40** 若 $f\in C^2[a,b]$ 且 $f(a)=f(b)=0$，证明对任意 $c\in[a,b]$ 有 $|f(c)|\le\dfrac{(b-a)^2}{8}\max|f''|$。

**思路**：直接套题 32 的结论 $f(c)=\dfrac{(c-a)(c-b)}{2}f''(\xi)$，再用 $(c-a)(c-b)$ 在区间内的极值 $\le(b-a)^2/4$ 估计。

**解**：由题 32：$|f(c)|\le\dfrac{(c-a)|c-b|}{2}\max|f''|\le\dfrac{(b-a)^2}{8}\max|f''|$（最大值在 $c=(a+b)/2$ 处取得）。 $\square$

---

## 五、不定积分（题 41–48）

**题 41** $\displaystyle\int\dfrac{dx}{x\sqrt{1-\ln^2 x}}$。

**思路**：被积函数里同时出现 $\ln x$ 与 $\dfrac{1}{x}dx=d(\ln x)$——明显是凑微分信号，令 $u=\ln x$ 把它化为反三角函数的标准形式。

**解**：$u=\ln x$，原积分 $=\int\dfrac{du}{\sqrt{1-u^2}}=\arcsin u+C=\arcsin(\ln x)+C$。

---

**题 42** $\displaystyle\int\dfrac{x}{\sqrt{x^2+2x+5}}\,dx$。

**思路**：根式下含二次三项式，先配方化为 $(x+1)^2+4$；分子的 $x$ 拆成 $(x+1)-1$，前半凑微分根式自身（$d[(x+1)^2+4]=2(x+1)dx$），后半化为标准 $\dfrac{1}{\sqrt{u^2+a^2}}$ 公式。

**解**：$x^2+2x+5=(x+1)^2+4$。$\int\dfrac{(x+1)-1}{\sqrt{(x+1)^2+4}}\,dx=\sqrt{(x+1)^2+4}-\ln|x+1+\sqrt{(x+1)^2+4}|+C$。

---

**题 43** $\displaystyle\int e^{2x}\cos x\,dx$。

**思路**：$e^{ax}$ 与 $\sin/\cos$ 组合是经典循环积分——分部两次后会回到原积分，移项即可。

**解**：设 $I=\int e^{2x}\cos x\,dx$。两次分部后得 $I=\dfrac{e^{2x}\sin x}{1}+\dfrac{e^{2x}(2\cos x)}{...}$，整理得 $I=\dfrac{e^{2x}(2\cos x+\sin x)}{5}+C$（系数 $5=2^2+1^2$）。

---

**题 44** $\displaystyle\int\dfrac{dx}{1+e^x}$。

**思路**：分子凑出"分母 - 缺项"形式：$1=\dfrac{1+e^x-e^x}{1+e^x}\cdot$。或直接令 $u=e^x$。

**解**：$=\int\dfrac{1+e^x-e^x}{1+e^x}\,dx=x-\int\dfrac{e^x}{1+e^x}\,dx=x-\ln(1+e^x)+C$。

---

**题 45** $\displaystyle\int\dfrac{\sin x}{1+\sin x}\,dx$。

**思路**：分子凑分母 $\sin x=(1+\sin x)-1$，拆开后第二个积分需要进一步技巧：$\dfrac{1}{1+\sin x}$ 用"乘共轭" $\dfrac{1-\sin x}{\cos^2 x}$。

**解**：

$$
=\int\!\left(1-\dfrac{1}{1+\sin x}\right)dx=x-\int\dfrac{1-\sin x}{\cos^2 x}\,dx=x-\tan x+\sec x+C.
$$

（用 $\int\sec^2 x\,dx=\tan x$、$\int\sec x\tan x\,dx=\sec x$。）

---

**题 46** $\displaystyle\int x\arctan x\,dx$。

**思路**：反三角 $\times$ 代数——按 LIATE 优先级选 $u=\arctan x$（更复杂的求导反而简单：$\frac{1}{1+x^2}$）。$dv=x\,dx$。

**解**：$u=\arctan x$，$v=x^2/2$。

$$
=\tfrac{x^2}{2}\arctan x-\tfrac12\int\tfrac{x^2}{1+x^2}\,dx=\tfrac{x^2}{2}\arctan x-\tfrac{x}{2}+\tfrac12\arctan x+C.
$$

中间用 $\dfrac{x^2}{1+x^2}=1-\dfrac{1}{1+x^2}$。

---

**题 47** $\displaystyle\int\dfrac{dx}{\sqrt{x}+\sqrt[3]{x}}$。

**思路**：两个不同次根式，做整体代换使两个根式都变成多项式——取最小公倍数次幂 $x=t^6$（$6=\text{lcm}(2,3)$）。

**解**：$x=t^6,\ dx=6t^5\,dt,\ \sqrt x=t^3,\ \sqrt[3]x=t^2$。

$$
\int\dfrac{6t^5}{t^3+t^2}\,dt=6\int\dfrac{t^3}{t+1}\,dt=6\int(t^2-t+1-\tfrac{1}{t+1})\,dt=2t^3-3t^2+6t-6\ln|t+1|+C,
$$

回代 $t=x^{1/6}$。

---

**题 48** $\displaystyle\int\dfrac{\ln(1+x)}{x^2}\,dx$。

**思路**：分子是对数，分母是幂——分部首选 $u=\ln(1+x)$，$dv=dx/x^2$（$v=-1/x$ 简单）。分部后剩下的有理积分用部分分式 $\dfrac{1}{x(1+x)}=\dfrac{1}{x}-\dfrac{1}{1+x}$。

**解**：

$$
=-\dfrac{\ln(1+x)}{x}+\int\dfrac{dx}{x(1+x)}=-\dfrac{\ln(1+x)}{x}+\ln\left|\dfrac{x}{1+x}\right|+C.
$$

---

## 六、定积分（题 49–58）

**题 49** 求 $\displaystyle\int_0^{\pi/2}\sin^4 x\,dx$。

**思路**：$\int_0^{\pi/2}\sin^n x\,dx$ 是 Wallis 积分，有标准递推公式。或用 $\sin^4 x=\left(\frac{1-\cos 2x}{2}\right)^2$ 展开。

**解**：Wallis 公式（$n=4$ 偶）：$=\dfrac{3}{4}\cdot\dfrac{1}{2}\cdot\dfrac{\pi}{2}=\boxed{\tfrac{3\pi}{16}}$。

---

**题 50** $\displaystyle\int_0^1\dfrac{\ln(1+x)}{1+x^2}\,dx$。

**思路**：$1+x^2$ 提示 $x=\tan t$ 代换。代换后变成 $\int_0^{\pi/4}\ln(1+\tan t)\,dt$，再用对称技巧 $t\to\pi/4-t$ 与恒等 $(1+\tan t)(1+\tan(\pi/4-t))=2$ 让两个积分相加得常数。

**解**：$x=\tan t$，$I=\int_0^{\pi/4}\ln(1+\tan t)\,dt$。令 $u=\pi/4-t$，$I=\int_0^{\pi/4}\ln(1+\tan(\pi/4-u))\,du=\int_0^{\pi/4}\ln\dfrac{2}{1+\tan u}\,du$。两式相加：$2I=\int_0^{\pi/4}\ln 2\,dt=\dfrac{\pi\ln 2}{4}$。故 $I=\boxed{\dfrac{\pi\ln 2}{8}}$。

---

**题 51** $\displaystyle\int_{-1}^1\dfrac{x^2}{1+e^x}\,dx$。

**思路**：积分区间关于原点对称，被积里有 $e^x$——这是典型"对称区间 + $e^x$"题型，用 $f(x)+f(-x)$ 化简。$\dfrac{x^2}{1+e^x}+\dfrac{x^2}{1+e^{-x}}=x^2$。

**解**：$I=\int_{-1}^1\dfrac{x^2}{1+e^x}\,dx$。由对称 $I=\int_{-1}^1\dfrac{x^2 e^x}{1+e^x}\,dx$（换元 $x\to -x$）。两式相加 $2I=\int_{-1}^1 x^2\,dx=\tfrac23$，$I=\boxed{\tfrac13}$。

---

**题 52** $\displaystyle\int_0^\pi x\sin x\,dx$。

**思路**：被积是"代数 $\times$ 三角"——分部，取 $u=x,dv=\sin x\,dx$。

**解**：$=[-x\cos x]_0^\pi+\int_0^\pi\cos x\,dx=\pi-0=\boxed{\pi}$。

---

**题 53** 讨论 $\displaystyle\int_0^1\dfrac{x\arctan x}{(1+x^2)^{3/2}}\,dx$ 的存在性。

**思路**：被积函数在 $[0,1]$ 连续（无瑕点），自然是定积分，必存在。

**解**：被积函数在闭区间连续，积分存在；数值约 $\approx 0.17$。

---

**题 54** $\displaystyle\int_0^{2\pi}\dfrac{dx}{a+b\sin x}$（$a>|b|$）。

**思路**：三角有理函数标准做法是万能代换 $t=\tan(x/2)$，但 $0\to 2\pi$ 时 $t$ 跳到 $\infty$，需分段或用周期对称。结果有现成公式。

**解**：标准结论：$\displaystyle\int_0^{2\pi}\dfrac{dx}{a+b\sin x}=\dfrac{2\pi}{\sqrt{a^2-b^2}}$。

---

**题 55** 求 $\displaystyle\int_0^{+\infty} e^{-x^2}\,dx$。

**思路**：单变量没有初等原函数，但定积分 $(-\infty,+\infty)$ 通过二维极坐标技巧可算（高斯积分）；半区间是它的一半。

**解**：$=\dfrac{\sqrt\pi}{2}$。

---

**题 56** 求 $\displaystyle\int_0^1\dfrac{x^4(1-x)^4}{1+x^2}\,dx$ 并由此说明 $\dfrac{22}{7}>\pi$。

**思路**：经典数学小品。分子是高次多项式，分母 $1+x^2$——做长除法得多项式部分 + $\dfrac{\text{余项}}{1+x^2}$，后者积出 $\arctan$。结合 $\arctan 1=\pi/4$ 与被积非负即得不等式。

**解**：展开 $x^4(1-x)^4=x^8-4x^7+6x^6-4x^5+x^4$，长除得

$$
\dfrac{x^4(1-x)^4}{1+x^2}=x^6-4x^5+5x^4-4x^2+4-\dfrac{4}{1+x^2}.
$$

积分（$0\to 1$）：$\dfrac{1}{7}-\dfrac{2}{3}+1-\dfrac{4}{3}+4-\pi=\dfrac{22}{7}-\pi$。由被积非负知 $\dfrac{22}{7}>\pi$。

---

**题 57** 求 $\displaystyle\int_0^1\dfrac{\ln x}{1-x}\,dx$。

**思路**：$\dfrac{1}{1-x}=\sum x^n$（几何级数）逐项积分；每项 $\int_0^1 x^n\ln x\,dx$ 分部得 $-\dfrac{1}{(n+1)^2}$，求和即 $-\zeta(2)$。

**解**：$\int_0^1 x^n\ln x\,dx=-\dfrac{1}{(n+1)^2}$。求和 $=-\sum_{n\ge 0}\dfrac{1}{(n+1)^2}=-\dfrac{\pi^2}{6}$。

---

**题 58** $\displaystyle\int_0^{\pi/2}\ln\sin x\,dx$。

**思路**：用对称性 $I=\int_0^{\pi/2}\ln\cos x\,dx$（$x\to\pi/2-x$）。$2I=\int_0^{\pi/2}\ln(\sin x\cos x)\,dx=\int_0^{\pi/2}\ln\tfrac{\sin 2x}{2}\,dx$。再用倍角后换元 $u=2x$ 拆出原积分本身，解方程。

**解**：

$$
2I=\int_0^{\pi/2}\ln\sin 2x\,dx-\dfrac{\pi}{2}\ln 2.
$$

令 $u=2x$：$\int_0^{\pi/2}\ln\sin 2x\,dx=\tfrac12\int_0^\pi\ln\sin u\,du=\int_0^{\pi/2}\ln\sin u\,du=I$（用 $u\to\pi-u$ 对称）。故 $2I=I-\tfrac\pi2\ln 2$，$I=-\dfrac{\pi\ln 2}{2}$。

---

## 七、广义积分与积分应用（题 59–65）

**题 59** 讨论 $\displaystyle\int_0^{+\infty}\dfrac{dx}{x^p(1+x)}$ 的收敛性。

**思路**：广义积分要分别在两个"危险点" $x\to 0^+$ 和 $x\to+\infty$ 处单独判断。每处用 $p$-积分作比较。

**解**：

- $x\to 0$：$\dfrac{1}{x^p(1+x)}\sim x^{-p}$，需 $p<1$；
- $x\to\infty$：$\sim x^{-p-1}$，需 $p+1>1\Leftrightarrow p>0$。

两端都收敛 $\Leftrightarrow 0<p<1$。

---

**题 60** 求曲线 $y=x^2$ 与 $y=\sqrt x$ 围成区域的面积。

**思路**：先求交点 $(0,0),(1,1)$，再判哪条在上（$\sqrt x>x^2$ 当 $0<x<1$），上下相减积分。

**解**：$\int_0^1(\sqrt x-x^2)\,dx=\dfrac{2}{3}-\dfrac{1}{3}=\boxed{\tfrac13}$。

---

**题 61** 求 $y=\ln x$ 在 $[1,e]$ 段绕 $x$ 轴旋转的旋转体体积。

**思路**：旋转体体积公式 $V=\pi\int y^2\,dx$。被积是 $\ln^2 x$，用分部积分（两次降幂）。

**解**：$\int\ln^2 x\,dx=x\ln^2 x-2\int\ln x\,dx=x\ln^2 x-2(x\ln x-x)+C$。代入 $1\to e$：$V=\pi[(e-2e+2e)-(0-0+2)]=\pi(e-2)$。

---

**题 62** 求 $y=\sin x$（$0\le x\le \pi$）的弧长。

**思路**：弧长 $L=\int\sqrt{1+y'^2}\,dx=\int_0^\pi\sqrt{1+\cos^2 x}\,dx$。这是椭圆积分，没有初等表达。

**解**：$L=\int_0^\pi\sqrt{1+\cos^2 x}\,dx\approx 3.820$（无闭形式）。

---

**题 63** 计算 $\Gamma(5/2)$。

**思路**：$\Gamma(s+1)=s\Gamma(s)$ 递推 + $\Gamma(1/2)=\sqrt\pi$。

**解**：$\Gamma(5/2)=\tfrac32\Gamma(3/2)=\tfrac32\cdot\tfrac12\Gamma(1/2)=\dfrac{3\sqrt\pi}{4}$。

---

**题 64** $\displaystyle\int_0^{+\infty}\dfrac{\sin x}{x}\,dx$。

**思路**：Dirichlet 积分，标准结果。证法之一：考虑参数积分 $I(t)=\int_0^\infty e^{-tx}\dfrac{\sin x}{x}\,dx$，对 $t$ 求导后变成有理函数积分。

**解**：$=\dfrac{\pi}{2}$（条件收敛）。

---

**题 65** 圆 $x^2+y^2=R^2$ 绕 $y$ 轴旋转得球，求体积。

**思路**：圆盘法 $V=\pi\int(R^2-y^2)\,dy$，从 $-R$ 到 $R$。

**解**：$V=\pi\int_{-R}^R(R^2-y^2)\,dy=\pi[R^2 y-y^3/3]_{-R}^R=\dfrac{4\pi R^3}{3}$。

---

## 八、多元函数微分（题 66–75）

**题 66** $z=x^y$，求偏导。

**思路**：$\partial_x z$ 时 $y$ 当常数，用幂法则；$\partial_y z$ 时 $x$ 当常数，用指数法则 $a^y\to a^y\ln a$。

**解**：$z_x=yx^{y-1}$，$z_y=x^y\ln x$。

---

**题 67** $u=f(x,y,z),\ z=g(x,y)$，求 $\partial u/\partial x$。

**思路**：多层复合，把 $u$ 看作 $(x,y,z(x,y))$ 的函数；$x$ 既直接出现又通过 $z$ 出现，故链式法则有两条路径。

**解**：$\dfrac{\partial u}{\partial x}=f_x+f_z\cdot g_x$。

---

**题 68** 证明 $z=\sin(x+y)+\cos(x-y)$ 满足 $z_{xx}-z_{yy}=0$。

**思路**：波动方程。直接两次求偏导验证。或观察 $\sin(x+y)$ 与 $\cos(x-y)$ 都形如 $f(x\pm y)$，本身满足 $\partial_x^2 f=\partial_y^2 f$（对 $f(x+y)$）和 $\partial_x^2 f=\partial_y^2 f$（对 $f(x-y)$）。

**解**：$z_{xx}=-\sin(x+y)-\cos(x-y)=z_{yy}$。故 $z_{xx}-z_{yy}=0$。 $\square$

---

**题 69** 求 $f(x,y)=x^2+y^2-xy+x-y$ 的极值。

**思路**：先解 $\nabla f=0$ 找驻点，再用 Hessian 二阶判定。

**解**：$f_x=2x-y+1=0,\ f_y=2y-x-1=0$ $\Rightarrow x=-1/3,y=1/3$。Hessian $\begin{pmatrix}2&-1\\-1&2\end{pmatrix}$ 正定，$f(-1/3,1/3)=-1/3$ 为极小值。

---

**题 70** 求 $f(x,y)=x^2+2y^2$ 在约束 $x+y=1$ 下的最值。

**思路**：约束极值——Lagrange 乘子法，或直接代入 $y=1-x$ 化为一元。

**解**：$\nabla f=\lambda\nabla g$：$2x=\lambda,4y=\lambda$。配合 $x+y=1$：$x=2/3,y=1/3,f=2/3$。这是最小值；当 $|x|\to\infty$ 时 $f\to\infty$，无最大值。

---

**题 71** 求方向导数 $\partial_{\mathbf l}f$，$f=x^2+y^2+z^2$，$\mathbf l=(1,2,2)/3$，点 $(1,1,1)$。

**思路**：方向导数 $=\nabla f\cdot\mathbf l$（$\mathbf l$ 须为单位向量，此处已归一）。

**解**：$\nabla f|_{(1,1,1)}=(2,2,2)$，$\partial_{\mathbf l}f=\dfrac{2+4+4}{3}=\dfrac{10}{3}$。

---

**题 72** 求 $f(x,y)=e^{x+y}$ 在 $(0,0)$ 的二阶 Taylor 展开。

**思路**：把 $e^u$ 展开后代入 $u=x+y$。

**解**：$e^{x+y}=1+(x+y)+\dfrac{(x+y)^2}{2}+\cdots=1+x+y+\tfrac{x^2+2xy+y^2}{2}+\cdots$。

---

**题 73** 由 $F(x,y,z)=x^2+y^2+z^2-3xyz=0$ 在 $(1,1,1)$ 附近确定 $z=z(x,y)$。求 $z_x,z_y$。

**思路**：隐函数定理：$z_x=-F_x/F_z$。先验证 $F_z\ne 0$。

**解**：$F_x=2x-3yz,\ F_z=2z-3xy$，在 $(1,1,1)$ 处 $F_x=F_z=-1$（同样 $F_y=-1$）。故 $z_x=-(-1)/(-1)=-1$，$z_y=-1$。

---

**题 74** $f(x,y)=x^2-y^2$ 在单位圆上的最值。

**思路**：单位圆参数化 $x=\cos\theta,y=\sin\theta$，化为一元函数。

**解**：$f=\cos 2\theta\in[-1,1]$。最大 $1$（$\theta=0,\pi$），最小 $-1$（$\theta=\pm\pi/2$）。

---

**题 75** $\mathbf F=(yz,xz,xy)$，证明保守并求势函数。

**思路**：保守场判据 $\nabla\times\mathbf F=0$。势函数 $\varphi$ 满足 $\nabla\varphi=\mathbf F$，可逐分量积分凑出。

**解**：$\nabla\times\mathbf F=(x-x,y-y,z-z)=\mathbf 0$。$\varphi_x=yz\Rightarrow\varphi=xyz+h(y,z)$；再由 $\varphi_y=xz$ 得 $h_y=0$；由 $\varphi_z=xy$ 得 $h_z=0$。故 $\varphi=xyz$。

---

## 九、重积分与曲线/曲面积分（题 76–85）

**题 76** $\displaystyle\iint_D xy\,dA$，$D=\{0\le x\le 1,0\le y\le x\}$。

**思路**：三角形区域，按"先 $y$ 后 $x$"积分更顺：$y\in[0,x],x\in[0,1]$。

**解**：$\int_0^1\!\!\int_0^x xy\,dy\,dx=\int_0^1\tfrac{x^3}{2}\,dx=\tfrac18$。

---

**题 77** 极坐标计算 $\displaystyle\iint_{x^2+y^2\le 1}e^{-(x^2+y^2)}\,dA$。

**思路**：被积函数与积分区域都对 $r$ 有对称性——直接极坐标，$dA=r\,dr\,d\theta$，$x^2+y^2=r^2$ 在指数里凑出 $d(r^2)$。

**解**：$\int_0^{2\pi}\!\!\int_0^1 e^{-r^2}r\,dr\,d\theta=2\pi\cdot\dfrac{1-e^{-1}}{2}=\pi(1-e^{-1})$。

---

**题 78** 求圆柱 $x^2+y^2\le 1,0\le z\le 2$ 内 $\iiint xyz\,dV$。

**思路**：被积函数关于 $x$ 是奇函数，区域关于 $x=0$ 对称——直接为 0。

**解**：$=0$。

---

**题 79** $\oint_L(-y\,dx+x\,dy)$，$L:x^2+y^2=1$ 逆时针。

**思路**：经典的"$\oint(-y\,dx+x\,dy)=2\cdot$面积"——Green 公式立即给结果。

**解**：Green 给 $\iint_D(1+1)\,dA=2\pi$。

---

**题 80** Green：$\oint_L(x^2y\,dx+xy^2\,dy)$，$L$ 是 $[0,1]^2$ 正向边界。

**思路**：套 Green 公式 $\oint(P\,dx+Q\,dy)=\iint(Q_x-P_y)\,dA$。

**解**：$Q_x-P_y=y^2-x^2$。$\int_0^1\!\!\int_0^1(y^2-x^2)\,dx\,dy=\tfrac13-\tfrac13=0$。

---

**题 81** $\iint_S z\,dS$，$S$ 单位上半球面。

**思路**：参数化或用对称——更简洁的是用 $dS=\dfrac{dA}{|n_z|}$，$|n_z|=z$（因球面外法向单位化），把 $z\,dS$ 化为 $dA$。

**解**：$\iint_S z\,dS=\iint_{D}1\,dA=\pi$（$D$ 单位圆盘）。

---

**题 82** Gauss：$\iint_S(x\,dy\,dz+y\,dz\,dx+z\,dx\,dy)$，$S$ 单位球外侧。

**思路**：$\mathbf F=(x,y,z)$ 散度 $=3$，球内体积 $=\tfrac{4\pi}{3}$。

**解**：$\iiint 3\,dV=4\pi$。

---

**题 83** $\iint_D\sqrt{x^2+y^2}\,dA$，$D=\{x^2+y^2\le 4\}$。

**思路**：圆域 + 含 $\sqrt{x^2+y^2}$ → 极坐标。

**解**：$\int_0^{2\pi}\!\!\int_0^2 r\cdot r\,dr\,d\theta=2\pi\cdot\tfrac{8}{3}=\dfrac{16\pi}{3}$。

---

**题 84** $z=x^2+y^2$ 与 $z=4$ 围成立体的体积。

**思路**：在 $z=4$ 处底圆 $x^2+y^2=4$；体积 $=\iint(\text{上}-\text{下})\,dA=\iint(4-(x^2+y^2))\,dA$，极坐标。

**解**：$\int_0^{2\pi}\!\!\int_0^2(4-r^2)r\,dr\,d\theta=2\pi[2r^2-r^4/4]_0^2=2\pi(8-4)=8\pi$。

---

**题 85** $\mathbf r(t)=(\cos t,\sin t,t),0\le t\le 2\pi$ 的弧长。

**思路**：$L=\int|\mathbf r'(t)|\,dt$。

**解**：$|\mathbf r'|=\sqrt{\sin^2+\cos^2+1}=\sqrt 2$，$L=2\sqrt 2\pi$。

---

## 十、级数（题 86–93）

**题 86** $\displaystyle\sum_{n=1}^\infty\dfrac{n^2}{2^n}$ 收敛性。

**思路**：分子多项式、分母指数——指数压幂，比值法立即得收敛。

**解**：$\dfrac{a_{n+1}}{a_n}=\dfrac{(n+1)^2}{2n^2}\to\tfrac12<1$，收敛。

---

**题 87** $\displaystyle\sum_{n=1}^\infty\dfrac{(-1)^{n-1}}{n}$。

**思路**：交错调和级数——Leibniz 判别收敛；求和则需要 $\ln(1+x)$ 的幂级数 $x=1$ 处的值。

**解**：$\ln(1+x)=\sum(-1)^{n-1}x^n/n$，取 $x=1$ 得 $\ln 2$。

---

**题 88** $\displaystyle\sum\dfrac{1}{n\ln n}$ 收敛性。

**思路**：项不是简单 $1/n^p$，用积分判别法对应到 $\int dx/(x\ln x)$。

**解**：$\int_2^\infty\dfrac{dx}{x\ln x}=\ln\ln x|_2^\infty=\infty$，**发散**。

---

**题 89** $\displaystyle\sum_{n=0}^\infty\dfrac{x^n}{n!}$ 收敛域与和。

**思路**：比值法判收敛半径，认出是 $e^x$ 的 Maclaurin。

**解**：$R=\infty$，$\sum=e^x$。

---

**题 90** $\displaystyle\sum_{n=1}^\infty\dfrac{x^n}{n}$ 收敛半径与和。

**思路**：比值得 $R=1$；端点单独验。和函数：项 $\dfrac{x^n}{n}=\int_0^x t^{n-1}\,dt$，调换求和与积分得 $\int_0^x\dfrac{1}{1-t}\,dt$。

**解**：$R=1$，$x\in[-1,1)$。和 $=-\ln(1-x)$。

---

**题 91** $f(x)=\dfrac{1}{1-x}$ 的 Maclaurin。

**思路**：几何级数。

**解**：$\sum_{n\ge 0}x^n$，$|x|<1$。

---

**题 92** $\arctan x$ 的 Maclaurin。

**思路**：$\arctan' x=\dfrac{1}{1+x^2}=\sum(-1)^n x^{2n}$；逐项积分。

**解**：$\arctan x=\sum_{n=0}^\infty\dfrac{(-1)^n x^{2n+1}}{2n+1}$，$|x|\le 1$。

---

**题 93** $f(x)=x$（$-\pi<x<\pi$）的 Fourier 级数。

**思路**：奇函数 → $a_n=0$。只需算 $b_n=\dfrac{2}{\pi}\int_0^\pi x\sin nx\,dx$，分部积分。

**解**：$b_n=\dfrac{2(-1)^{n+1}}{n}$。$x=\sum_{n\ge 1}\dfrac{2(-1)^{n+1}}{n}\sin nx$。

---

## 十一、常微分方程（题 94–100）

**题 94** $y'+2y=e^{-x}$ 的通解。

**思路**：一阶线性 $y'+P(x)y=Q(x)$ 的标准做法：积分因子 $\mu=e^{\int P}$，方程左边变成 $(\mu y)'$。

**解**：$\mu=e^{2x}$，$(ye^{2x})'=e^x$，$ye^{2x}=e^x+C$，$y=e^{-x}+Ce^{-2x}$。

---

**题 95** $y''-3y'+2y=0$ 通解。

**思路**：齐次常系数线性 ODE，特征方程 $r^2-3r+2=0$ 解出两个特征根，通解由对应基解组合。

**解**：$r=1,2$，通解 $y=C_1 e^x+C_2 e^{2x}$。

---

**题 96** $y''-3y'+2y=e^x$ 通解。

**思路**：非齐次 = 齐次通解 + 一个特解。右端 $e^x$ 是齐次解的一部分（$r=1$ 是特征根），故"共振"，特解形式应改为 $Axe^x$。

**解**：齐次同题 95。设 $y_p=Axe^x$，代入得 $A=-1$。通解 $y=C_1 e^x+C_2 e^{2x}-xe^x$。

---

**题 97** $y'=\dfrac{y}{x}+1$（$x>0$）通解。

**思路**：化为标准线性 $y'-\dfrac{1}{x}y=1$，积分因子 $\mu=e^{-\int dx/x}=1/x$。

**解**：$(y/x)'=1/x$，$y/x=\ln x+C$，$y=x\ln x+Cx$。

---

**题 98** $y''+y=\sec x$（$|x|<\pi/2$）通解。

**思路**：右端 $\sec x$ 不是多项式 $\times$ 指数 $\times$ 三角的标准形式，待定系数失效；用常数变易法。设 $y_p=u_1\cos x+u_2\sin x$，由 $u_1'\cos x+u_2'\sin x=0$ 与 $-u_1'\sin x+u_2'\cos x=\sec x$ 解出 $u_1',u_2'$ 再积分。

**解**：$u_1'=-\tan x\Rightarrow u_1=\ln\cos x$；$u_2'=1\Rightarrow u_2=x$。故 $y_p=\cos x\ln\cos x+x\sin x$，通解 $y=C_1\cos x+C_2\sin x+\cos x\ln\cos x+x\sin x$。

---

**题 99** $y'=\dfrac{x+y}{x-y}$ 通解。

**思路**：右端是 $y/x$ 的函数（齐次方程），令 $u=y/x$ 化为可分离变量。

**解**：$y=ux$，$y'=u+xu'$，方程化为 $xu'=\dfrac{1+u^2}{1-u}$。分离：

$$
\int\dfrac{1-u}{1+u^2}\,du=\int\dfrac{dx}{x}\Rightarrow\arctan u-\tfrac12\ln(1+u^2)=\ln|x|+C.
$$

代回 $u=y/x$ 得隐式解。

---

**题 100** 初值问题 $y'=2xy,\ y(0)=1$。

**思路**：可分离变量。或视为一阶线性 $y'-2xy=0$。

**解**：$\dfrac{dy}{y}=2x\,dx\Rightarrow\ln|y|=x^2+C$；$y(0)=1$ 定 $C=0$。$y=e^{x^2}$。

---

## 解题策略小结

1. **极限**：四个工具——基本极限、等价替换、Taylor 展开、L'Hôpital。一阶相消就展二阶、二阶相消就展三阶；遇 $1^\infty$ 必取对数。
2. **可导/连续**：分段函数看左右极限；构造反例从 $|x|,x^{1/3},x^2\sin(1/x)$ 三个原型出发。
3. **求导**：复合用链式；隐式两边对 $x$ 求导；幂指型对数求导；参数方程 $\dfrac{dy}{dx}=\dfrac{y'(t)}{x'(t)}$。
4. **中值定理**：见"$\exists\xi$ 使..."就构造 $g(x)=$ 待证式的原函数，套 Rolle。
5. **不定积分**：识别类型——有理（部分分式）、根式（三角/根式代换）、$\ln/\arctan/\arcsin$（分部）、循环型（两次分部）。
6. **定积分**：奇偶对称、$\sin$-$\cos$ 互换（$x\to\pi/2-x$）、$x\to a+b-x$、Wallis、Riemann 和；广义积分分两端单独判敛。
7. **多元微分**：偏导对其他变量当常数；隐函数 $z_x=-F_x/F_z$；约束极值用 Lagrange。
8. **重积分**：极/柱/球坐标选用看积分区域；交换次序处理复杂区域。
9. **线面积分**：看到 $\oint$ 就想 Green，闭曲面想 Gauss，闭曲线在曲面边界想 Stokes。
10. **级数**：先比值/根值判收敛，再用幂级数转已知函数（$\dfrac{1}{1-x}$、$e^x$、$\ln(1+x)$、$\arctan x$）求和。
11. **ODE**：识别类型 → 可分离 / 齐次型 / 一阶线性（积分因子）/ 恰当 / 二阶常系数（特征方程 + 待定系数 / 常数变易）。

---

## 资料与延伸阅读

- 全国硕士研究生招生考试数学历年真题（数一、数二、数三）。
- 李永乐 / 武忠祥《考研数学复习全书》。
- 张宇《考研数学基础 30 讲 / 强化 36 讲》。
- 教育部考试中心《考试大纲及考试分析》。
- 本教程其他章节：1–10 章打牢基础、11–17 章覆盖积分与级数、18–22 章覆盖多元与向量分析、23–24 章覆盖 ODE。
