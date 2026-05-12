# 附录：历年考研真题精选 100 题

本附录精选 100 道考研数学（数一/数二/数三）风格真题，覆盖极限、连续与可导、一元微分学、微分中值定理与应用、不定积分、定积分、广义积分与积分应用、多元函数微分学、重积分与曲线曲面积分、级数、常微分方程等核心考点。每题给出完整解析。

题目均按教学顺序排列，每节末附小结性提示。建议先独立完成再核对。

> **使用建议**：将每题视为 25–30 分钟限时训练；做完后对照解析复盘思路；同类题目错误率高时回到对应章节复习。

---

## 一、极限（题 1–15）

**题 1**（数一/数二）求 $\displaystyle\lim_{x\to 0}\frac{\sqrt{1+2x}-\sqrt[3]{1+3x}}{x^2}$。

**解**：用 Taylor 展开。

$$
\sqrt{1+2x}=1+x-\tfrac{x^2}{2}+o(x^2),
$$
$$
\sqrt[3]{1+3x}=1+x-x^2+o(x^2).
$$

相减：$\sqrt{1+2x}-\sqrt[3]{1+3x}=\tfrac{x^2}{2}+o(x^2)$，故极限为 $\boxed{\tfrac12}$。

---

**题 2**（数三）求 $\displaystyle\lim_{x\to 0}\frac{e^x-e^{\sin x}}{x-\sin x}$。

**解**：分子 $=e^{\sin x}(e^{x-\sin x}-1)\sim e^{\sin x}\cdot(x-\sin x)$。因此极限为 $\displaystyle\lim_{x\to 0}e^{\sin x}=\boxed{1}$。

---

**题 3** 求 $\displaystyle\lim_{x\to 0}\left(\frac{1+\tan x}{1+\sin x}\right)^{1/\sin^3 x}$。

**解**：取对数 $L=\lim\dfrac{\ln(1+\tan x)-\ln(1+\sin x)}{\sin^3 x}$。

$$
\ln(1+\tan x)-\ln(1+\sin x)=\ln\!\left(1+\frac{\tan x-\sin x}{1+\sin x}\right)\sim \tan x-\sin x.
$$

而 $\tan x-\sin x=\tan x(1-\cos x)\sim x\cdot\tfrac{x^2}{2}=\tfrac{x^3}{2}$，$\sin^3 x\sim x^3$。故 $L=\tfrac12$，极限为 $\boxed{e^{1/2}}$。

---

**题 4** 求 $\displaystyle\lim_{n\to\infty}\sum_{k=1}^n\frac{1}{n+k}$。

**解**：化为 Riemann 和

$$
\sum_{k=1}^n\frac{1}{n+k}=\sum_{k=1}^n\frac{1}{1+k/n}\cdot\frac{1}{n}\to\int_0^1\frac{1}{1+x}\,dx=\boxed{\ln 2}.
$$

---

**题 5** 求 $\displaystyle\lim_{x\to+\infty}\left(\sqrt{x^2+x+1}-\sqrt{x^2-x+1}\right)$。

**解**：分子有理化，

$$
=\lim\frac{2x}{\sqrt{x^2+x+1}+\sqrt{x^2-x+1}}=\frac{2}{1+1}=\boxed{1}.
$$

---

**题 6** 求 $\displaystyle\lim_{x\to 0}\frac{\ln(1+x)-x}{x^2}$。

**解**：$\ln(1+x)=x-\tfrac{x^2}{2}+o(x^2)$，分子 $=-\tfrac{x^2}{2}+o(x^2)$，极限为 $\boxed{-\tfrac12}$。

---

**题 7** 求 $\displaystyle\lim_{x\to 0^+}x^x$。

**解**：$x^x=e^{x\ln x}$。$\lim_{x\to 0^+}x\ln x=\lim\dfrac{\ln x}{1/x}=\lim\dfrac{1/x}{-1/x^2}=\lim(-x)=0$。极限为 $\boxed{1}$。

---

**题 8** 求 $\displaystyle\lim_{n\to\infty}\sqrt[n]{n!}/n$。

**解**：取对数 $\dfrac{1}{n}\sum_{k=1}^n\ln\dfrac{k}{n}\to\int_0^1\ln x\,dx=-1$。极限为 $\boxed{1/e}$。

---

**题 9** 求 $\displaystyle\lim_{x\to 0}\frac{(1+x)^{1/x}-e}{x}$。

**解**：$(1+x)^{1/x}=e^{\frac{\ln(1+x)}{x}}$。设 $g(x)=\dfrac{\ln(1+x)}{x}$，则 $g(0)=1,\ g'(0)=-\tfrac12$。

$$
(1+x)^{1/x}=e\cdot e^{g(x)-1}\approx e(1+g(x)-1)=e+e(g(x)-1).
$$

而 $g(x)-1\sim -\tfrac{x}{2}$。故极限为 $\boxed{-\tfrac{e}{2}}$。

---

**题 10** 求 $\displaystyle\lim_{x\to 0}\frac{\arctan x-x}{x^3}$。

**解**：$\arctan x=x-\tfrac{x^3}{3}+o(x^3)$，极限为 $\boxed{-\tfrac13}$。

---

**题 11** 已知 $\lim_{x\to 0}\dfrac{f(x)}{x^2}=2$，求 $\lim_{x\to 0}\dfrac{f(\sin x)}{x^2}$。

**解**：$\sin x\sim x$，所以 $f(\sin x)\sim 2\sin^2 x\sim 2x^2$。极限为 $\boxed{2}$。

---

**题 12** 求 $\displaystyle\lim_{x\to 1}\frac{x-x^x}{1-x+\ln x}$。

**解**：分子 $=x(1-x^{x-1})=x(1-e^{(x-1)\ln x})\sim -x(x-1)\ln x$。$x\to 1$ 时 $\ln x\sim x-1$，故分子 $\sim-(x-1)^2$。

分母：令 $t=x-1$，$\ln(1+t)=t-\tfrac{t^2}{2}+o(t^2)$，故 $1-x+\ln x=-t+t-\tfrac{t^2}{2}+o(t^2)=-\tfrac{t^2}{2}+o(t^2)$。

极限 $=\dfrac{-t^2}{-t^2/2}=\boxed{2}$。

---

**题 13** 求 $\displaystyle\lim_{n\to\infty}n\left(\sqrt[n]{a}-1\right)$（$a>0$）。

**解**：$\sqrt[n]{a}-1=e^{\ln a/n}-1\sim \dfrac{\ln a}{n}$。极限为 $\boxed{\ln a}$。

---

**题 14** 求 $\displaystyle\lim_{n\to\infty}\left(\frac{1}{n^2}+\frac{2}{n^2}+\cdots+\frac{n}{n^2}\right)$。

**解**：$=\dfrac{1+2+\cdots+n}{n^2}=\dfrac{n(n+1)}{2n^2}\to\boxed{\tfrac12}$。

---

**题 15** 设 $f(x)$ 连续且 $f(0)=0$、$f'(0)=2$，求 $\displaystyle\lim_{x\to 0}\dfrac{\int_0^{x^2}f(t)\,dt}{x^4}$。

**解**：由 L'Hôpital，$\lim\dfrac{f(x^2)\cdot 2x}{4x^3}=\lim\dfrac{f(x^2)}{2x^2}=\lim\dfrac{f(x^2)}{x^2}\cdot\tfrac12=\tfrac12\cdot f'(0)=\boxed{1}$。

---

## 二、连续与可导（题 16–22）

**题 16** 设 $f(x)=\begin{cases}\dfrac{\ln(1+ax)}{x},& x>0\\ b,& x=0\\ \dfrac{e^{x}-1}{x},& x<0\end{cases}$ 在 $x=0$ 连续，求 $a,b$。

**解**：右极限 $a$，左极限 $1$。故 $a=b=1$。$\boxed{a=b=1}$。

---

**题 17** 讨论 $f(x)=\lim_{n\to\infty}\dfrac{x^{2n}-1}{x^{2n}+1}x$ 的连续性。

**解**：

- $|x|>1$：$x^{2n}\to\infty$，$f(x)=x$；
- $|x|<1$：$x^{2n}\to 0$，$f(x)=-x$；
- $|x|=1$：$f(\pm 1)=0$。

$x=1$ 处左极限 $-1$、右极限 $1$，不连续；$x=-1$ 类似。其余处连续。

---

**题 18** 设 $f(x)=|x-1|\cdot|x+1|$，问 $f$ 在哪些点不可导？

**解**：$f(x)=|x^2-1|$，在 $x=\pm 1$ 处函数过零变号但 $f$ 仍为 $0$，左右导数差为 $\pm 2$，故 $x=\pm 1$ **不可导**。

---

**题 19** 已知 $f(x)$ 在 $x=0$ 可导且 $f(0)=0,\ f'(0)=a$。求 $\displaystyle\lim_{x\to 0}\dfrac{f(x^2)}{x f(x)}$（设 $a\ne 0$）。

**解**：分子 $\sim a x^2$，分母 $\sim x\cdot a x=a x^2$。极限为 $\boxed{1}$。

---

**题 20** 设 $f(x)=\begin{cases}x^2\sin\tfrac1x,& x\ne 0\\ 0,& x=0\end{cases}$，求 $f'(0)$ 及讨论 $f'$ 在 $0$ 是否连续。

**解**：$f'(0)=\lim_{h\to 0}\dfrac{h^2\sin(1/h)}{h}=\lim h\sin(1/h)=0$。

$x\ne 0$ 时 $f'(x)=2x\sin\tfrac1x-\cos\tfrac1x$，$x\to 0$ 时第二项无极限，故 $f'$ 在 $0$ 不连续。

---

**题 21** 设 $f$ 二阶可导且 $f(0)=0,\ f'(0)=1,\ f''(0)=2$，求 $\displaystyle\lim_{x\to 0}\dfrac{f(x)-x}{x^2}$。

**解**：由 Taylor $f(x)=x+x^2+o(x^2)$。极限 $=\boxed{1}$。

---

**题 22** 证明：若 $f$ 在 $[a,b]$ 连续、$(a,b)$ 可导且 $f'$ 单调，则 $f'$ 在 $(a,b)$ 连续。

**解**：单调函数只可能存在跳跃间断。若 $f'$ 在 $c$ 有跳跃，则由 Darboux 定理（导函数中间值性质），$f'$ 的值会跳过区间 $(f'(c-),f'(c+))$ 内某些值，但 Darboux 要求取得介值，矛盾。 $\square$

---

## 三、一元微分计算（题 23–30）

**题 23** 求 $y=(\sin x)^{\cos x}$ 的导数。

**解**：取对数 $\ln y=\cos x\ln\sin x$，

$$
\dfrac{y'}{y}=-\sin x\ln\sin x+\cos x\cdot\cot x.
$$

$$
y'=(\sin x)^{\cos x}\!\left[\cot x\cos x-\sin x\ln\sin x\right].
$$

---

**题 24** 求由方程 $e^y+xy=e$ 确定的隐函数 $y(x)$ 在 $x=0$ 处的 $y'$ 与 $y''$。

**解**：$x=0\Rightarrow y=1$。对方程两边求导：

$$
e^y y'+y+xy'=0\Rightarrow y'(0)=-\tfrac{1}{e}.
$$

再求导：$e^y(y')^2+e^y y''+2y'+xy''=0$。代入 $x=0,y=1,y'=-1/e$：

$$
e\cdot\tfrac{1}{e^2}+e y''-\tfrac{2}{e}=0\Rightarrow y''(0)=\tfrac{1}{e^2}.
$$

---

**题 25** 设 $\begin{cases}x=t-\sin t\\ y=1-\cos t\end{cases}$，求 $\dfrac{dy}{dx}$ 与 $\dfrac{d^2y}{dx^2}$。

**解**：

$$
\dfrac{dy}{dx}=\dfrac{\sin t}{1-\cos t}=\cot\dfrac{t}{2}.
$$

$$
\dfrac{d^2y}{dx^2}=\dfrac{d}{dt}\!\left(\cot\tfrac{t}{2}\right)\Big/\dfrac{dx}{dt}=\dfrac{-\tfrac12\csc^2(t/2)}{1-\cos t}=-\dfrac{1}{2(1-\cos t)^2}\cdot 2=-\dfrac{1}{(1-\cos t)^2}\cdot \tfrac12.
$$

整理：$\dfrac{d^2y}{dx^2}=-\dfrac{1}{4\sin^4(t/2)}$。

---

**题 26** 求 $y=\dfrac{1}{x^2-3x+2}$ 的 $n$ 阶导数。

**解**：部分分式 $y=\dfrac{1}{x-2}-\dfrac{1}{x-1}$。$n$ 阶导：

$$
y^{(n)}=(-1)^n n!\!\left[\dfrac{1}{(x-2)^{n+1}}-\dfrac{1}{(x-1)^{n+1}}\right].
$$

---

**题 27** 求 $f(x)=\ln(1+x)$ 在 $x=0$ 的 $n$ 阶 Taylor 展开（带 Lagrange 余项）。

**解**：$f^{(k)}(0)=(-1)^{k-1}(k-1)!$，

$$
\ln(1+x)=\sum_{k=1}^n\frac{(-1)^{k-1}x^k}{k}+\frac{(-1)^n x^{n+1}}{(n+1)(1+\xi)^{n+1}},\ \xi\in(0,x).
$$

---

**题 28** 求 $y=\arctan\dfrac{x}{1-x^2/2}$ 在 $x=0$ 的 $y^{(2025)}(0)$（仅利用奇偶性给结论）。

**解**：$y$ 为奇函数，偶数阶导数在 $0$ 为 $0$，奇数阶非零。可写 Maclaurin 展开 $y=x+\dots$（系数从展开可得）。具体：

$$
y(x)=\arctan x+\arctan(x/2)+\cdots\text{（用 }\arctan\text{ 和差公式拆分）}.
$$

实际上 $\tan(\arctan a+\arctan b)=\dfrac{a+b}{1-ab}$。令 $a=b=x/\sqrt{2}\cdot\dots$ 略；本题主要考奇偶：$y^{(2024)}(0)=0$，$y^{(2025)}(0)$ 由展开系数 $\times 2025!$。

> **要点**：考研此类题着重于奇偶判断与高阶展开系数。

---

**题 29** 设 $y=x\ln x$，求 $y^{(n)}$（$n\ge 2$）。

**解**：$y'=\ln x+1$，$y''=1/x$，$y^{(k)}=(-1)^k(k-2)!/x^{k-1}$（$k\ge 2$）。

---

**题 30** 已知 $f(x)$ 二阶可导，$g(x)=f(\sin x)$，求 $g''(x)$。

**解**：$g'=f'(\sin x)\cos x$，$g''=f''(\sin x)\cos^2 x-f'(\sin x)\sin x$。

---

## 四、微分中值定理与应用（题 31–40）

**题 31** 设 $f$ 在 $[0,1]$ 连续，$(0,1)$ 可导，$f(0)=0,\ f(1)=1$。证明存在 $\xi\in(0,1)$ 使 $f'(\xi)=2\xi$。

**解**：令 $g(x)=f(x)-x^2$，$g(0)=0,\ g(1)=0$。由 Rolle，$\exists\xi$ 使 $g'(\xi)=0$，即 $f'(\xi)=2\xi$。 $\square$

---

**题 32** 设 $f$ 在 $[a,b]$ 二阶可导，$f(a)=f(b)=0$。证明 $\exists\xi\in(a,b)$ 使 $f''(\xi)=\dfrac{2f(c)}{(c-a)(c-b)}$，其中 $c\in(a,b)$ 给定。

**解**：构造辅助函数

$$
\varphi(x)=f(x)-f(c)\cdot\frac{(x-a)(x-b)}{(c-a)(c-b)}.
$$

$\varphi(a)=\varphi(c)=\varphi(b)=0$。由 Rolle 用两次得 $\varphi''(\xi)=0$，即得证。 $\square$

---

**题 33** 证明：$x>0$ 时 $\ln(1+x)<x$。

**解**：令 $f(x)=x-\ln(1+x)$，$f(0)=0$，$f'(x)=1-\dfrac{1}{1+x}=\dfrac{x}{1+x}>0$。故 $f$ 单调增，$f(x)>0$。 $\square$

---

**题 34** 证明：当 $0<x<\dfrac\pi2$ 时 $\dfrac{2x}{\pi}<\sin x<x$。

**解**：右半：$f(x)=x-\sin x$，$f'=1-\cos x\ge 0$。

左半：$g(x)=\dfrac{\sin x}{x}$，$g'(x)=\dfrac{x\cos x-\sin x}{x^2}$。在 $(0,\pi/2)$ 中 $x\cos x<\sin x$（因 $h(x)=\tan x-x>0$），故 $g$ 递减，$g(\pi/2)=\dfrac{2}{\pi}<g(x)<g(0^+)=1$。 $\square$

---

**题 35** 求 $f(x)=x^3-3x$ 在 $[-2,2]$ 上的最值。

**解**：$f'=3x^2-3=0\Rightarrow x=\pm 1$。$f(\pm 1)=\mp 2$，$f(\pm 2)=\pm 2$。最大值 $2$，最小值 $-2$。

---

**题 36** 求 $y=\dfrac{x}{1+x^2}$ 的凹凸区间与拐点。

**解**：$y'=\dfrac{1-x^2}{(1+x^2)^2}$，$y''=\dfrac{2x(x^2-3)}{(1+x^2)^3}$。$y''=0$ 在 $x=0,\pm\sqrt 3$。

$(-\infty,-\sqrt 3)$ 凹下，$(-\sqrt 3,0)$ 凹上，$(0,\sqrt 3)$ 凹下，$(\sqrt 3,\infty)$ 凹上。拐点 $(0,0),(\pm\sqrt 3,\pm\sqrt 3/4)$。

---

**题 37** 求 $f(x)=e^x\sin x$ 在 $[0,2\pi]$ 上的最大、最小值。

**解**：$f'=e^x(\sin x+\cos x)=0$，得 $\tan x=-1$，$x=3\pi/4,7\pi/4$。

$$
f(3\pi/4)=\tfrac{\sqrt 2}{2}e^{3\pi/4},\quad f(7\pi/4)=-\tfrac{\sqrt 2}{2}e^{7\pi/4}.
$$

$f(0)=0,f(2\pi)=0$。最大值 $\tfrac{\sqrt 2}{2}e^{3\pi/4}$，最小值 $-\tfrac{\sqrt 2}{2}e^{7\pi/4}$。

---

**题 38** 用洛必达法则求 $\displaystyle\lim_{x\to 0}\dfrac{x-\sin x}{x^3}$。

**解**：$\to\lim\dfrac{1-\cos x}{3x^2}\to\lim\dfrac{\sin x}{6x}=\boxed{\tfrac16}$。

---

**题 39** 设 $f$ 在 $[0,1]$ 连续、可导且 $f(0)=0,\ |f'(x)|\le M$。证明 $\int_0^1 f^2(x)\,dx\le \tfrac{M^2}{3}$。

**解**：由 $f(x)=\int_0^x f'(t)\,dt$ 与 Cauchy-Schwarz：$f^2(x)\le x\int_0^x f'^2(t)\,dt\le M^2 x^2$。积分得 $\int_0^1 f^2\,dx\le M^2/3$。 $\square$

---

**题 40** 证明：若 $f\in C^2[a,b]$ 且 $f(a)=f(b)=0$，则 $\exists \xi$ 使 $|f(c)|\le\dfrac{(b-a)^2}{8}\max|f''|$（$c$ 为 $[a,b]$ 内任一点）。

**解**：利用题 32 公式 $f(c)=\dfrac{(c-a)(c-b)}{2}f''(\xi)$，$(c-a)(c-b)\le(b-a)^2/4$。故 $|f(c)|\le \dfrac{(b-a)^2}{8}|f''(\xi)|$。 $\square$

---

## 五、不定积分（题 41–48）

**题 41** $\displaystyle\int\dfrac{dx}{x\sqrt{1-\ln^2 x}}$。

**解**：令 $u=\ln x$，$du=dx/x$。$\int\dfrac{du}{\sqrt{1-u^2}}=\arcsin u+C=\arcsin\ln x+C$。

---

**题 42** $\displaystyle\int\dfrac{x}{\sqrt{x^2+2x+5}}\,dx$。

**解**：$x^2+2x+5=(x+1)^2+4$。$\int\dfrac{x\,dx}{\sqrt{(x+1)^2+4}}=\int\dfrac{(x+1)-1}{\sqrt{(x+1)^2+4}}\,dx$。

前部分凑微分 $d((x+1)^2+4)/2$，得 $\sqrt{(x+1)^2+4}$；后部分为 $-\ln|x+1+\sqrt{(x+1)^2+4}|$。

最终：$\sqrt{x^2+2x+5}-\ln|x+1+\sqrt{x^2+2x+5}|+C$。

---

**题 43** $\displaystyle\int e^{2x}\cos x\,dx$。

**解**：分部积分两次循环求解，得

$$
\int e^{2x}\cos x\,dx=\dfrac{e^{2x}(2\cos x+\sin x)}{5}+C.
$$

---

**题 44** $\displaystyle\int\dfrac{dx}{1+e^x}$。

**解**：$=\int\dfrac{1+e^x-e^x}{1+e^x}\,dx=x-\ln(1+e^x)+C$。

---

**题 45** $\displaystyle\int\dfrac{\sin x}{1+\sin x}\,dx$。

**解**：$=\int\!\left(1-\dfrac{1}{1+\sin x}\right)dx=x-\int\dfrac{1-\sin x}{\cos^2 x}\,dx=x-\tan x+\sec x+C$。

---

**题 46** $\displaystyle\int x\arctan x\,dx$。

**解**：分部积分 $u=\arctan x,\ dv=x\,dx$：

$$
=\tfrac{x^2}{2}\arctan x-\tfrac12\int\tfrac{x^2}{1+x^2}\,dx=\tfrac{x^2}{2}\arctan x-\tfrac{x}{2}+\tfrac12\arctan x+C.
$$

---

**题 47** $\displaystyle\int\dfrac{dx}{\sqrt{x}+\sqrt[3]{x}}$。

**解**：令 $x=t^6$，$dx=6t^5\,dt$。$\sqrt x=t^3$，$\sqrt[3]x=t^2$。

$$
\int\dfrac{6t^5}{t^3+t^2}\,dt=6\int\dfrac{t^3}{t+1}\,dt.
$$

长除：$\dfrac{t^3}{t+1}=t^2-t+1-\dfrac{1}{t+1}$。积出后回代 $t=x^{1/6}$。

---

**题 48** $\displaystyle\int\dfrac{\ln(1+x)}{x^2}\,dx$。

**解**：分部 $u=\ln(1+x),\ dv=dx/x^2$，$v=-1/x$。

$$
=-\dfrac{\ln(1+x)}{x}+\int\dfrac{dx}{x(1+x)}=-\dfrac{\ln(1+x)}{x}+\ln\dfrac{|x|}{|1+x|}+C.
$$

---

## 六、定积分（题 49–58）

**题 49** 求 $\displaystyle\int_0^{\pi/2}\sin^4 x\,dx$。

**解**：Wallis 公式：$=\dfrac{3}{4}\cdot\dfrac{1}{2}\cdot\dfrac{\pi}{2}=\boxed{\tfrac{3\pi}{16}}$。

---

**题 50** $\displaystyle\int_0^1\dfrac{\ln(1+x)}{1+x^2}\,dx$。

**解**：经典题。令 $x=\tan t$，$dx=\sec^2 t\,dt$，

$$
=\int_0^{\pi/4}\ln(1+\tan t)\,dt.
$$

用 $\ln(1+\tan t)+\ln(1+\tan(\pi/4-t))=\ln 2$（因 $(1+\tan t)(1+\tan(\pi/4-t))=2$）。故积分 $=\tfrac{1}{2}\cdot\tfrac{\pi}{4}\cdot\ln 2=\boxed{\tfrac{\pi\ln 2}{8}}$。

---

**题 51** $\displaystyle\int_{-1}^1\dfrac{x^2}{1+e^x}\,dx$。

**解**：$f(x)=\dfrac{x^2}{1+e^x}$，$f(x)+f(-x)=x^2$。故积分 $=\tfrac12\int_{-1}^1 x^2\,dx=\boxed{\tfrac13}$。

---

**题 52** 求 $\displaystyle\int_0^\pi x\sin x\,dx$。

**解**：分部 $=[-x\cos x]_0^\pi+\int_0^\pi\cos x\,dx=\pi+0=\boxed{\pi}$。

---

**题 53** 求 $\displaystyle\int_0^1\dfrac{x\arctan x}{(1+x^2)^{3/2}}\,dx$ 的存在性与近似（不要求闭形式）。

**解**：被积函数在 $[0,1]$ 连续，积分存在。数值约 $\approx 0.17$。

---

**题 54** $\displaystyle\int_0^{2\pi}\dfrac{dx}{a+b\sin x}$（$a>|b|$）。

**解**：万能代换 $t=\tan(x/2)$ 或公式：积分 $=\dfrac{2\pi}{\sqrt{a^2-b^2}}$。

---

**题 55** 求 $\displaystyle\int_0^{+\infty} e^{-x^2}\,dx$。

**解**：高斯积分 $=\dfrac{\sqrt\pi}{2}$。

---

**题 56** 求 $\displaystyle\int_0^1\dfrac{x^4(1-x)^4}{1+x^2}\,dx$ 并由此说明 $\dfrac{22}{7}>\pi$。

**解**：长除 $x^4(1-x)^4=(1+x^2)\,Q(x)+4$（具体 $Q$ 略）。积分得 $\dfrac{22}{7}-\pi$，且被积非负，故 $\dfrac{22}{7}>\pi$。

---

**题 57** 求 $\displaystyle\int_0^1 \dfrac{\ln x}{1-x}\,dx$。

**解**：展开 $\dfrac{1}{1-x}=\sum x^n$，

$$
=\sum_{n=0}^\infty\int_0^1 x^n\ln x\,dx=-\sum_{n=0}^\infty\dfrac{1}{(n+1)^2}=-\dfrac{\pi^2}{6}.
$$

---

**题 58** 求 $\displaystyle\int_0^{\pi/2}\ln\sin x\,dx$。

**解**：经典 $=-\dfrac{\pi\ln 2}{2}$（用对称性 $\int_0^{\pi/2}\ln\sin x\,dx=\int_0^{\pi/2}\ln\cos x\,dx$ 与 $\sin 2x$ 倍角即得）。

---

## 七、广义积分与积分应用（题 59–65）

**题 59** 讨论 $\displaystyle\int_0^{+\infty}\dfrac{dx}{x^p(1+x)}$ 的收敛性。

**解**：$x\to 0$ 时 $\sim x^{-p}$，需 $p<1$；$x\to\infty$ 时 $\sim x^{-p-1}$，需 $p+1>1$ 即 $p>0$。故 $0<p<1$ 时收敛。

---

**题 60** 求曲线 $y=x^2$ 与 $y=\sqrt x$ 围成区域的面积。

**解**：交点 $(0,0),(1,1)$。$\int_0^1(\sqrt x-x^2)\,dx=\tfrac{2}{3}-\tfrac{1}{3}=\boxed{\tfrac13}$。

---

**题 61** 求曲线 $y=\ln x$ 在 $[1,e]$ 段绕 $x$ 轴旋转所得旋转体体积。

**解**：$V=\pi\int_1^e\ln^2 x\,dx$。分部两次得

$$
\int\ln^2 x\,dx=x\ln^2 x-2x\ln x+2x+C.
$$

代入：$V=\pi[(e-2e+2e)-(0-0+2)]=\pi(e-2)$。

---

**题 62** 求曲线 $y=\sin x$（$0\le x\le \pi$）的弧长。

**解**：$L=\int_0^\pi\sqrt{1+\cos^2 x}\,dx$。这是椭圆积分，没有初等闭形式，数值 $\approx 3.820$。

---

**题 63** 计算 $\Gamma(5/2)$。

**解**：$\Gamma(5/2)=\dfrac{3}{2}\cdot\dfrac{1}{2}\cdot\sqrt\pi=\dfrac{3\sqrt\pi}{4}$。

---

**题 64** 求 $\displaystyle\int_0^{+\infty}\dfrac{\sin x}{x}\,dx$ 的值（条件收敛）。

**解**：经典 Dirichlet 积分 $=\dfrac{\pi}{2}$。绝对值积分发散。

---

**题 65** 圆 $x^2+y^2=R^2$ 围成圆盘绕 $y$ 轴旋转得球，求球体体积。

**解**：$V=\pi\int_{-R}^R(R^2-y^2)\,dy=\dfrac{4\pi R^3}{3}$。

---

## 八、多元函数微分（题 66–75）

**题 66** 设 $z=x^y$，求 $\dfrac{\partial z}{\partial x},\ \dfrac{\partial z}{\partial y}$。

**解**：$z_x=yx^{y-1}$，$z_y=x^y\ln x$。

---

**题 67** 设 $u=f(x,y,z)$，$z=g(x,y)$，求 $\dfrac{\partial u}{\partial x}$。

**解**：$\dfrac{\partial u}{\partial x}=f_x+f_z\cdot g_x$。

---

**题 68** 求 $z=\sin(x+y)+\cos(x-y)$ 满足 $z_{xx}-z_{yy}=0$。

**解**：直接计算两次偏导后相减得 $0$（左右两项在 $xx$ 与 $yy$ 下相等）。 $\square$

---

**题 69** 求 $f(x,y)=x^2+y^2-xy+x-y$ 的极值。

**解**：$f_x=2x-y+1=0,\ f_y=2y-x-1=0$，解得 $x=-1/3,\ y=1/3$。Hessian $\begin{pmatrix}2&-1\\-1&2\end{pmatrix}$ 正定，极小值 $f=-\tfrac13$。

---

**题 70** 求 $f(x,y)=x^2+2y^2$ 在约束 $x+y=1$ 下的最值。

**解**：Lagrange：$\nabla f=\lambda\nabla g\Rightarrow 2x=\lambda,\ 4y=\lambda$，结合 $x+y=1$ 得 $x=2/3,\ y=1/3$，$f=\tfrac{4}{9}+\tfrac{2}{9}=\tfrac{2}{3}$。最小值 $\tfrac{2}{3}$（无最大值）。

---

**题 71** 求方向导数 $\partial_{\mathbf l}f$，其中 $f=x^2+y^2+z^2$，$\mathbf l=(1,2,2)/3$，点 $(1,1,1)$。

**解**：$\nabla f=(2,2,2)$，$\partial_{\mathbf l}f=(2,2,2)\cdot(1,2,2)/3=\dfrac{10}{3}$。

---

**题 72** 求 $f(x,y)=e^{x+y}$ 在 $(0,0)$ 的二阶 Taylor 展开。

**解**：$e^{x+y}=1+(x+y)+\tfrac{(x+y)^2}{2}+\cdots$。

---

**题 73** 设 $F(x,y,z)=x^2+y^2+z^2-3xyz$，方程 $F=0$ 在 $(1,1,1)$ 附近确定 $z=z(x,y)$。求 $z_x(1,1),\ z_y(1,1)$。

**解**：$F_x=2x-3yz=-1,\ F_y=-1,\ F_z=2z-3xy=-1$。$z_x=-F_x/F_z=-1,\ z_y=-1$。

---

**题 74** 求 $f(x,y)=x^2-y^2$ 在单位圆上的最值。

**解**：参数化 $x=\cos\theta,\ y=\sin\theta$，$f=\cos 2\theta$，最大 $1$（$\theta=0$），最小 $-1$（$\theta=\pi/2$）。

---

**题 75** 设 $\mathbf F=(yz,xz,xy)$，证明 $\mathbf F$ 是保守场并求势函数。

**解**：$\nabla\times\mathbf F=\mathbf 0$，所以保守。势函数 $\varphi=xyz$，$\nabla\varphi=\mathbf F$。

---

## 九、重积分与曲线/曲面积分（题 76–85）

**题 76** 求 $\displaystyle\iint_D xy\,dA$，$D=\{0\le x\le 1,\ 0\le y\le x\}$。

**解**：$\int_0^1\int_0^x xy\,dy\,dx=\int_0^1\tfrac{x^3}{2}\,dx=\tfrac18$。

---

**题 77** 用极坐标计算 $\displaystyle\iint_{x^2+y^2\le 1}e^{-(x^2+y^2)}\,dA$。

**解**：$\int_0^{2\pi}\int_0^1 e^{-r^2}r\,dr\,d\theta=2\pi\cdot\tfrac{1-e^{-1}}{2}=\pi(1-e^{-1})$。

---

**题 78** 求圆柱 $x^2+y^2\le 1$、$0\le z\le 2$ 内 $\iiint xyz\,dV$。

**解**：$\int_0^2 z\,dz\cdot\iint xy\,dA=2\cdot 0=0$（被积奇对称）。

---

**题 79** 求曲线 $L$：$x^2+y^2=1$（逆时针一周）下 $\oint_L(-y\,dx+x\,dy)$。

**解**：参数化或用 Green 公式：$=2\iint_D 1\,dA=2\pi$。

---

**题 80** Green 公式：求 $\oint_L(x^2y\,dx+xy^2\,dy)$，$L$ 是边界 $0\le x\le 1,\ 0\le y\le 1$ 正向。

**解**：$\iint_D(y^2-x^2)\,dA=\int_0^1\int_0^1(y^2-x^2)\,dx\,dy=\tfrac13-\tfrac13=0$。

---

**题 81** 计算曲面积分 $\iint_S z\,dS$，$S$ 为单位上半球面。

**解**：$z=\sqrt{1-x^2-y^2}$，$dS=\dfrac{dA}{z}$，故 $\iint z\,dS=\iint_{D}1\,dA=\pi$。

---

**题 82** 用 Gauss 定理计算 $\iint_S(x\,dy\,dz+y\,dz\,dx+z\,dx\,dy)$，$S$ 为单位球面外侧。

**解**：$\iiint\nabla\cdot\mathbf F\,dV=\iiint 3\,dV=3\cdot\tfrac{4\pi}{3}=4\pi$。

---

**题 83** 求 $\displaystyle\iint_D\sqrt{x^2+y^2}\,dA$，$D=\{x^2+y^2\le 4\}$。

**解**：极坐标 $=\int_0^{2\pi}\int_0^2 r\cdot r\,dr\,d\theta=2\pi\cdot\tfrac{8}{3}=\dfrac{16\pi}{3}$。

---

**题 84** 求由 $z=x^2+y^2$ 与 $z=4$ 围成立体的体积。

**解**：$V=\iint_{x^2+y^2\le 4}(4-(x^2+y^2))\,dA=\int_0^{2\pi}\int_0^2(4-r^2)r\,dr\,d\theta=2\pi(8-4)=8\pi$。

---

**题 85** 求曲线 $\mathbf r(t)=(\cos t,\sin t,t)$（$0\le t\le 2\pi$）的弧长。

**解**：$|\mathbf r'(t)|=\sqrt{\sin^2 t+\cos^2 t+1}=\sqrt 2$。弧长 $=2\sqrt 2\pi$。

---

## 十、级数（题 86–93）

**题 86** 判别 $\displaystyle\sum_{n=1}^\infty\dfrac{n^2}{2^n}$ 收敛性。

**解**：根值/比值：$\dfrac{a_{n+1}}{a_n}=\dfrac{(n+1)^2}{2n^2}\to\tfrac12<1$，收敛。

---

**题 87** 求 $\displaystyle\sum_{n=1}^\infty\dfrac{(-1)^{n-1}}{n}$。

**解**：$\ln 2$（由 $\ln(1+x)$ 幂级数取 $x=1$）。

---

**题 88** 判别 $\displaystyle\sum\dfrac{1}{n\ln n}$ 收敛性。

**解**：积分判别 $\int_2^\infty\dfrac{dx}{x\ln x}=[\ln\ln x]=\infty$，发散。

---

**题 89** 求幂级数 $\displaystyle\sum_{n=0}^\infty\dfrac{x^n}{n!}$ 的收敛域与和。

**解**：收敛域 $\mathbb R$，和 $=e^x$。

---

**题 90** 求 $\displaystyle\sum_{n=1}^\infty\dfrac{x^n}{n}$ 的收敛半径与和函数。

**解**：$R=1$，收敛区间 $[-1,1)$。和 $=-\ln(1-x)$。

---

**题 91** 求 $f(x)=\dfrac{1}{1-x}$ 在 $x=0$ 的幂级数及收敛半径。

**解**：$\sum x^n$，$R=1$。

---

**题 92** 将 $f(x)=\arctan x$ 展为 Maclaurin 级数。

**解**：$\arctan x=\sum_{n=0}^\infty\dfrac{(-1)^n x^{2n+1}}{2n+1}$，$|x|\le 1$。

---

**题 93** 求 Fourier 级数 $f(x)=x$（$-\pi<x<\pi$）。

**解**：奇函数，$a_n=0$，$b_n=\dfrac{2}{\pi}\int_0^\pi x\sin nx\,dx=\dfrac{2(-1)^{n+1}}{n}$。故 $x=\sum\dfrac{2(-1)^{n+1}}{n}\sin nx$。

---

## 十一、常微分方程（题 94–100）

**题 94** 求 $y'+2y=e^{-x}$ 的通解。

**解**：一阶线性，积分因子 $e^{2x}$。$(ye^{2x})'=e^x$，$y=e^{-2x}(e^x+C)=e^{-x}+Ce^{-2x}$。

---

**题 95** 求 $y''-3y'+2y=0$ 的通解。

**解**：特征 $r^2-3r+2=0$，$r=1,2$。$y=C_1 e^x+C_2 e^{2x}$。

---

**题 96** 求 $y''-3y'+2y=e^x$ 的通解。

**解**：齐次同上。$r=1$ 是单根，特解设 $y_p=Axe^x$，代入得 $A=-1$。通解 $y=C_1 e^x+C_2 e^{2x}-xe^x$。

---

**题 97** 求 $y'=\dfrac{y}{x}+1$（$x>0$）的通解。

**解**：化为 $y'-\dfrac{1}{x}y=1$，因子 $\dfrac{1}{x}$。$(y/x)'=\dfrac{1}{x}$，$y/x=\ln x+C$，$y=x\ln x+Cx$。

---

**题 98** 求 $y''+y=\sec x$（$|x|<\pi/2$）的通解。

**解**：齐次解 $y_h=C_1\cos x+C_2\sin x$。常数变易法：$y_p=u_1\cos x+u_2\sin x$，$u_1'\cos x+u_2'\sin x=0$，$-u_1'\sin x+u_2'\cos x=\sec x$。解得 $u_1'=-\tan x$，$u_2'=1$。$u_1=\ln\cos x$，$u_2=x$。故 $y_p=\cos x\ln\cos x+x\sin x$。

---

**题 99** 求 $y'=\dfrac{x+y}{x-y}$ 的通解。

**解**：齐次型，令 $u=y/x$。$xu'+u=\dfrac{1+u}{1-u}$，$xu'=\dfrac{1+u^2}{1-u}$。分离变量

$$
\int\dfrac{1-u}{1+u^2}\,du=\int\dfrac{dx}{x}\Rightarrow \arctan u-\tfrac12\ln(1+u^2)=\ln|x|+C.
$$

代回 $u=y/x$ 得隐式通解。

---

**题 100** 求初值问题 $y'=2xy,\ y(0)=1$。

**解**：分离变量 $\dfrac{dy}{y}=2x\,dx$，$\ln|y|=x^2+C$，$y=e^{x^2}$。

---

## 解题策略小结

1. **极限**：四个工具——基本极限、等价替换、Taylor 展开、L'Hôpital；高阶问题优先用 Taylor。
2. **可导/连续**：定义、左右极限、单调或介值定理；常用反例 $|x|,x^{1/3},x^2\sin(1/x)$。
3. **求导**：复合函数、隐函数、参数方程、对数求导法；$n$ 阶导用 Leibniz 或部分分式。
4. **中值定理**：构造辅助函数 $g(x)=f(x)-\text{目标}$ 后用 Rolle / Lagrange / Cauchy。
5. **不定积分**：先看类型——有理函数（部分分式）、三角（万能代换/凑微分）、根式（三角或根式代换）、$\ln/\arctan/\arcsin$（分部）。
6. **定积分**：对称性、换元、分部、Wallis、Riemann 和；区分常义与广义。
7. **多元微分**：偏导、全微分、链式、隐函数定理、Lagrange 乘子。
8. **重积分**：选合适坐标（极/柱/球），交换积分次序。
9. **曲线/曲面积分**：参数化、Green/Stokes/Gauss 定理。
10. **级数**：先讨论收敛域，再求和（幂级数转已知函数）；Fourier 注意奇偶。
11. **ODE**：识别类型（可分离/齐次型/线性/恰当/常系数线性），齐次 + 特解结构。

---

## 资料与延伸阅读

- 全国硕士研究生招生考试数学历年真题（数一、数二、数三）。
- 李永乐 / 武忠祥《考研数学复习全书》。
- 张宇《考研数学基础 30 讲 / 强化 36 讲》。
- 教育部考试中心《考试大纲及考试分析》。
- 本教程其他章节：1–10 章打牢基础、11–17 章覆盖积分与级数、18–21 章覆盖多元与 ODE。
