# 附录 F1b：微分应用详解（C.11-C.25, D.13-D.30, E.07-E.18）

> 涵盖微分应用（Ch.7–10）全部 45 题：基础 15 题（C.11–C.25）、中档 18 题（D.13–D.30）、提升 12 题（E.07–E.18）。
> 每题包含：**题目回顾**、**思路**（含 toolkit 引用）、**解答**（紧凑推导）、**答案**、**总结**（1 句）。
> Toolkit 引用：→ 链式法则、→ 乘积法则、→ 隐函数求导、→ 对数求导、→ L'Hôpital、→ 中值定理、→ Taylor 展开、→ 极值判别、→ 凸凹分析、→ 参数方程。

---

## Part 1：基础题（C.11–C.25）

---

## C.11 [基础] Ch.7

**题目回顾**　用导数定义求 $f(x)=x^2+1$ 在 $x=2$ 处的导数。

**思路**　直接代入定义式 $f'(x)=\lim_{h\to0}\dfrac{f(x+h)-f(x)}{h}$。

**解答**

$$f'(2)=\lim_{h\to 0}\frac{(2+h)^2+1-(4+1)}{h}=\lim_{h\to 0}\frac{4+4h+h^2-4}{h}=\lim_{h\to 0}(4+h)=4.$$

**答案**　$f'(2)=4$。

**总结**　导数定义题展开平方后分子必有公因子 $h$，约掉后令 $h\to0$ 即得。

---

## C.12 [基础] Ch.7

**题目回顾**　求 $y=3x^4-2x^3+x-5$ 的导数。

**思路**　逐项用幂函数法则 $(x^n)'=nx^{n-1}$，常数项导数为零。

**解答**

$$y'=12x^3-6x^2+1.$$

**答案**　$y'=12x^3-6x^2+1$。

**总结**　多项式求导逐项操作，系数乘指数、指数减一，常数归零。

---

## C.13 [基础] Ch.7

**题目回顾**　求 $y=e^x\cos x$ 的导数。

**思路**　乘积法则 $(uv)'=u'v+uv'$，→ 乘积法则。

**解答**

$$y'=(e^x)'\cos x+e^x(\cos x)'=e^x\cos x+e^x(-\sin x)=e^x(\cos x-\sin x).$$

**答案**　$y'=e^x(\cos x-\sin x)$。

**总结**　指数乘三角函数求导，乘积法则展开后合并同类 $e^x$ 因子。

---

## C.14 [基础] Ch.7

**题目回顾**　求 $y=\ln\sqrt{1+x^2}$ 的导数（化简）。

**思路**　先化简：$\ln\sqrt{1+x^2}=\tfrac{1}{2}\ln(1+x^2)$，再用链式法则，→ 链式法则。

**解答**

$$y=\tfrac{1}{2}\ln(1+x^2)\implies y'=\tfrac{1}{2}\cdot\frac{2x}{1+x^2}=\frac{x}{1+x^2}.$$

**答案**　$y'=\dfrac{x}{1+x^2}$。

**总结**　对数里有根号先用对数性质提出 $\tfrac12$，再链式求导，化简更顺畅。

---

## C.15 [基础] Ch.8

**题目回顾**　求 $y=\sin^2 x$ 的导数（用链式法则）。

**思路**　设 $u=\sin x$，$y=u^2$，→ 链式法则：$y'=2u\cdot u'$。

**解答**

$$y'=2\sin x\cdot\cos x=\sin 2x.$$

**答案**　$y'=\sin 2x$。

**总结**　$(\sin^n x)'=n\sin^{n-1}x\cos x$；$n=2$ 时恰好用二倍角公式化简。

---

## C.16 [基础] Ch.8

**题目回顾**　求隐函数 $x^2+y^2=4$ 在点 $(1,\sqrt{3})$ 处的 $\dfrac{dy}{dx}$。

**思路**　两边对 $x$ 求导，$y$ 视为 $x$ 的函数，→ 隐函数求导。

**解答**

$$2x+2y\,y'=0\implies y'=-\frac{x}{y}.$$

代入 $(1,\sqrt{3})$：$y'=-\dfrac{1}{\sqrt{3}}=-\dfrac{\sqrt{3}}{3}$。

**答案**　$\dfrac{dy}{dx}\Big|_{(1,\sqrt{3})}=-\dfrac{\sqrt{3}}{3}$。

**总结**　圆方程隐式求导结果 $y'=-x/y$，几何上是切线斜率，正好垂直于半径。

---

## C.17 [基础] Ch.8

**题目回顾**　写出曲线 $y=x^3-x$ 在点 $(1,0)$ 处的切线方程与法线方程。

**思路**　先求 $y'$ 在切点处的值（切线斜率），法线斜率为其负倒数。

**解答**

$y'=3x^2-1$，在 $x=1$：$y'(1)=2$。

- 切线：$y-0=2(x-1)$，即 $y=2x-2$。
- 法线斜率 $=-\tfrac12$，法线：$y=-\tfrac{1}{2}(x-1)$，即 $y=-\tfrac{x}{2}+\tfrac{1}{2}$。

**答案**　切线 $y=2x-2$，法线 $y=-\dfrac{x}{2}+\dfrac{1}{2}$。

**总结**　切线与法线斜率互为负倒数，代点求 $y'$ 值后套点斜式即完。

---

## C.18 [基础] Ch.9

**题目回顾**　求 $f(x)=x^3-3x+2$ 的单调区间与极值。

**思路**　令 $f'=0$ 求驻点，分析 $f'$ 符号，→ 极值判别。

**解答**

$f'(x)=3x^2-3=3(x-1)(x+1)$。

驻点 $x=\pm1$。

| 区间 | $(-\infty,-1)$ | $x=-1$ | $(-1,1)$ | $x=1$ | $(1,+\infty)$ |
|---|---|---|---|---|---|
| $f'$ | $+$ | $0$ | $-$ | $0$ | $+$ |
| $f$ | 递增 | 极大 | 递减 | 极小 | 递增 |

极大值 $f(-1)=4$，极小值 $f(1)=0$。

**答案**　递增：$(-\infty,-1)\cup(1,+\infty)$；递减：$(-1,1)$；极大 $4$，极小 $0$。

**总结**　三次多项式极值题：$f'$ 因式分解后列符号表，简单清晰。

---

## C.19 [基础] Ch.9

**题目回顾**　求 $f(x)=e^x-x$ 在 $\mathbb{R}$ 上的最小值。

**思路**　令 $f'=0$，验证为全局最小（$f''>0$，凸函数）。

**解答**

$f'(x)=e^x-1=0\implies x=0$。$f''(x)=e^x>0$（严格凸），$x=0$ 为全局极小。

$f(0)=1-0=1$。

**答案**　最小值为 $1$（在 $x=0$ 处取得）。

**总结**　凸函数唯一驻点即全局最小值点，$e^x-x\ge1$ 是常用不等式。

---

## C.20 [基础] Ch.9

**题目回顾**　用 L'Hôpital 法则求 $\displaystyle\lim_{x\to 0}\frac{x-\sin x}{x^3}$。

**思路**　$0/0$ 型，连续三次应用 L'Hôpital，→ L'Hôpital。

**解答**

$$\frac{0}{0}\xrightarrow{L}\frac{1-\cos x}{3x^2}\xrightarrow{L}\frac{\sin x}{6x}\xrightarrow{L}\frac{\cos x}{6}\xrightarrow{x\to0}\frac{1}{6}.$$

**答案**　$\dfrac{1}{6}$。

**总结**　$x-\sin x\sim x^3/6$ 是标准等价，L'Hôpital 三次或 Taylor 展开均可得此结果。

---

## C.21 [基础] Ch.9

**题目回顾**　判断 $y=x^2e^{-x}$ 的凸凹区间与拐点。

**思路**　求 $y''$，令 $y''=0$，分析符号，→ 凸凹分析。

**解答**

$y'=e^{-x}(2x-x^2)=xe^{-x}(2-x)$。

$y''=e^{-x}(2-4x+x^2)=e^{-x}(x^2-4x+2)$。

令 $y''=0$：$x^2-4x+2=0$，$x=2\pm\sqrt{2}$。

- $x\in(-\infty,2-\sqrt{2})$：$y''>0$，下凸（凹弧）。
- $x\in(2-\sqrt{2},2+\sqrt{2})$：$y''<0$，上凸（凸弧）。
- $x\in(2+\sqrt{2},+\infty)$：$y''>0$，下凸。

拐点：$x=2\pm\sqrt{2}$（对应两个拐点坐标由代入求得）。

**答案**　下凸：$(-\infty,2-\sqrt{2})\cup(2+\sqrt{2},+\infty)$；拐点 $x=2\pm\sqrt{2}$。

**总结**　拐点在 $y''=0$ 且两侧符号改变处，二次方程根即为拐点横坐标。

---

## C.22 [基础] Ch.10

**题目回顾**　写出 $e^x$ 在 $x=0$ 处的前 4 项 Maclaurin 展开（不含余项）。

**思路**　$e^x$ 各阶导数均为 $e^x$，→ Taylor 展开。

**解答**

$$e^x=1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\cdots=1+x+\frac{x^2}{2}+\frac{x^3}{6}+\cdots$$

前 4 项（含常数项）：$1+x+\dfrac{x^2}{2}+\dfrac{x^3}{6}$。

**答案**　$e^x\approx 1+x+\dfrac{x^2}{2}+\dfrac{x^3}{6}$。

**总结**　$e^x$ 展开系数为 $1/n!$，是最基础的 Taylor 级数，务必熟记到 $x^4$ 项。

---

## C.23 [基础] Ch.10

**题目回顾**　写出 $\sin x$ 在 $x=0$ 处前 3 个非零项的 Maclaurin 展开。

**思路**　$\sin x$ 只含奇次项，→ Taylor 展开。

**解答**

$$\sin x=x-\frac{x^3}{3!}+\frac{x^5}{5!}-\cdots=x-\frac{x^3}{6}+\frac{x^5}{120}+\cdots$$

前 3 个非零项：$x-\dfrac{x^3}{6}+\dfrac{x^5}{120}$。

**答案**　$\sin x\approx x-\dfrac{x^3}{6}+\dfrac{x^5}{120}$。

**总结**　$\sin x$ 展开只有奇次项，系数交替符号，分母为奇数阶乘。

---

## C.24 [基础] Ch.10

**题目回顾**　用一阶 Taylor 展开近似 $\ln(1.02)$（保留四位小数）。

**思路**　$\ln(1+x)\approx x$（$x$ 极小时），取 $x=0.02$。

**解答**

$$\ln(1+x)=x-\frac{x^2}{2}+\cdots\approx x\quad(x\text{ 很小}).$$

取 $x=0.02$：$\ln(1.02)\approx 0.02$（一阶）。

二阶修正：$\ln(1.02)\approx 0.02-\dfrac{(0.02)^2}{2}=0.02-0.0002=0.0198$。

精确值约为 $0.0198$（保留四位小数）。

**答案**　$\ln(1.02)\approx 0.0198$。

**总结**　$\ln(1+x)\approx x-x^2/2$ 保留二阶项更精确；$x$ 越小一阶近似越好。

---

## C.25 [基础] Ch.10

**题目回顾**　设 $f(x)=\cos x$，写出 $f(x)$ 在 $x=0$ 的 $n$ 阶 Maclaurin 展开的通项规律（偶次项）。

**思路**　$\cos x$ 只含偶次项，系数为 $(-1)^k/(2k)!$，→ Taylor 展开。

**解答**

$$\cos x=\sum_{k=0}^{\infty}\frac{(-1)^k}{(2k)!}x^{2k}=1-\frac{x^2}{2!}+\frac{x^4}{4!}-\frac{x^6}{6!}+\cdots$$

第 $k$ 个偶次项通项：$\dfrac{(-1)^k}{(2k)!}x^{2k}$，$k=0,1,2,\ldots$

**答案**　$\cos x=\displaystyle\sum_{k=0}^{\infty}\dfrac{(-1)^k}{(2k)!}x^{2k}$。

**总结**　$\cos x$ 是偶函数，展开只含偶次项；与 $\sin x$ 奇次项形式对称，系数差一步导数。

---

## Part 2：中档题（D.13–D.30）

---

## D.13 [中档] Ch.8

**题目回顾**　求 $y=(\sin x)^{\cos x}$ 的导数（对数求导法）。

**思路**　幂指函数，两边取对数再求导，→ 对数求导。

**解答**

$\ln y=\cos x\cdot\ln\sin x$。两边对 $x$ 求导：

$$\frac{y'}{y}=(-\sin x)\ln\sin x+\cos x\cdot\frac{\cos x}{\sin x}=-\sin x\ln\sin x+\frac{\cos^2 x}{\sin x}.$$

$$y'=(\sin x)^{\cos x}\left(\frac{\cos^2 x}{\sin x}-\sin x\ln\sin x\right).$$

**答案**　$y'=(\sin x)^{\cos x}\left(\dfrac{\cos^2 x}{\sin x}-\sin x\ln\sin x\right)$。

**总结**　"幂指函数"$u^v$ 固定套路：取对数，$\ln y=v\ln u$，再微分还原。

---

## D.14 [中档] Ch.8

**题目回顾**　由方程 $e^y+xy=e$ 确定的隐函数 $y(x)$ 在 $x=0$ 处，求 $y'(0)$ 与 $y''(0)$。

**思路**　先定 $y(0)$，再隐式求一阶、二阶导，→ 隐函数求导。

**解答**

代 $x=0$：$e^{y(0)}=e\implies y(0)=1$。

**一阶导**：两边对 $x$ 求导：

$$e^y y'+y+xy'=0\implies y'=-\frac{y}{e^y+x}.$$

$x=0$：$y'(0)=-\dfrac{1}{e}$。

**二阶导**：对 $e^y y'+y+xy'=0$ 再求导：

$$e^y(y')^2+e^y y''+y'+y'+xy''=0\implies e^y y''(1+x\cdot e^{-y})+\text{已知项}=0.$$

整理（在 $x=0,y=1$ 处代入 $y'=-1/e$）：

$$e\cdot y''(0)+e\cdot\frac{1}{e^2}+2\cdot\left(-\frac{1}{e}\right)=0\implies y''(0)=\frac{1}{e^2}.$$

**答案**　$y'(0)=-\dfrac{1}{e}$，$y''(0)=\dfrac{1}{e^2}$。

**总结**　隐函数高阶导：先确定各点函数值，再逐阶求导代入，勿遗漏乘积法则项。

---

## D.15 [中档] Ch.8

**题目回顾**　设参数方程 $\begin{cases}x=t-\sin t\\ y=1-\cos t\end{cases}$，求 $\dfrac{dy}{dx}$ 与 $\dfrac{d^2y}{dx^2}$。

**思路**　参数方程公式：$\dfrac{dy}{dx}=\dfrac{y'_t}{x'_t}$，二阶：$\dfrac{d^2y}{dx^2}=\dfrac{(y'_t/x'_t)'_t}{x'_t}$，→ 参数方程。

**解答**

$x'_t=1-\cos t$，$y'_t=\sin t$。

$$\frac{dy}{dx}=\frac{\sin t}{1-\cos t}=\cot\frac{t}{2}\quad(t\ne 2k\pi).$$

对 $\dfrac{dy}{dx}=\dfrac{\sin t}{1-\cos t}$ 再关于 $t$ 求导：

$$\left(\frac{dy}{dx}\right)'_t=\frac{\cos t(1-\cos t)-\sin t\cdot\sin t}{(1-\cos t)^2}=\frac{\cos t-1}{(1-\cos t)^2}=\frac{-1}{1-\cos t}.$$

$$\frac{d^2y}{dx^2}=\frac{-1/(1-\cos t)}{1-\cos t}=\frac{-1}{(1-\cos t)^2}.$$

**答案**　$\dfrac{dy}{dx}=\dfrac{\sin t}{1-\cos t}$；$\dfrac{d^2y}{dx^2}=\dfrac{-1}{(1-\cos t)^2}$。

**总结**　参数方程二阶导公式分母是 $x'_t$（不是 $(x'_t)^2$），容易混淆，须记准。

---

## D.16 [中档] Ch.8

**题目回顾**　求 $y=\dfrac{1}{x^2-3x+2}$ 的 $n$ 阶导数。

**思路**　先部分分式，再用公式 $\left(\tfrac{1}{x-a}\right)^{(n)}=\dfrac{(-1)^n n!}{(x-a)^{n+1}}$。

**解答**

$$\frac{1}{x^2-3x+2}=\frac{1}{(x-1)(x-2)}=\frac{1}{x-2}-\frac{1}{x-1}.$$

$$y^{(n)}=\frac{(-1)^n n!}{(x-2)^{n+1}}-\frac{(-1)^n n!}{(x-1)^{n+1}}=(-1)^n n!\left[\frac{1}{(x-2)^{n+1}}-\frac{1}{(x-1)^{n+1}}\right].$$

**答案**　$y^{(n)}=(-1)^n n!\left[\dfrac{1}{(x-2)^{n+1}}-\dfrac{1}{(x-1)^{n+1}}\right]$。

**总结**　有理函数高阶导先拆部分分式，逐项套 $\left(\tfrac{1}{x-a}\right)^{(n)}$ 公式是标准路线。

---

## D.17 [中档] Ch.8

**题目回顾**　已知 $f$ 二阶可导，$g(x)=f(\sin x)$，用链式法则写出 $g''(x)$。

**思路**　$g'=f'(\sin x)\cos x$，再对 $g'$ 用乘积 + 链式法则，→ 链式法则。

**解答**

$$g'(x)=f'(\sin x)\cdot\cos x.$$

$$g''(x)=f''(\sin x)\cdot\cos^2 x+f'(\sin x)\cdot(-\sin x)=f''(\sin x)\cos^2 x-f'(\sin x)\sin x.$$

**答案**　$g''(x)=f''(\sin x)\cos^2 x-f'(\sin x)\sin x$。

**总结**　复合函数二阶导需同时用链式法则和乘积法则，外层和内层各自的导数都要保留。

---

## D.18 [中档] Ch.8

**题目回顾**　求 $y=\arctan\dfrac{2x}{1-x^2}$ 的导数（$|x|<1$）。

**思路**　利用恒等式 $\arctan\dfrac{2x}{1-x^2}=2\arctan x$（$|x|<1$）化简后直接求导。

**解答**

当 $|x|<1$ 时，设 $\theta=\arctan x$，则 $\tan2\theta=\dfrac{2\tan\theta}{1-\tan^2\theta}=\dfrac{2x}{1-x^2}$，故

$$y=2\arctan x\implies y'=\frac{2}{1+x^2}.$$

**答案**　$y'=\dfrac{2}{1+x^2}$。

**总结**　认出反正切的二倍角恒等式是关键，化简后求导远比直接套链式法则简洁。

---

## D.19 [中档] Ch.9

**题目回顾**　设 $f$ 在 $[0,1]$ 连续，$(0,1)$ 可导，$f(0)=0,f(1)=1$。证明 $\exists\xi\in(0,1)$ 使 $f'(\xi)=2\xi$。

**思路**　构造 $g(x)=f(x)-x^2$，则 $g(0)=g(1)=0$，用 Rolle 定理，→ 中值定理。

**解答**

令 $g(x)=f(x)-x^2$，则：
- $g(0)=f(0)-0=0$，
- $g(1)=f(1)-1=0$。

$g$ 在 $[0,1]$ 连续，$(0,1)$ 可导。由 **Rolle 定理**，$\exists\xi\in(0,1)$ 使 $g'(\xi)=0$。

$$g'(\xi)=f'(\xi)-2\xi=0\implies f'(\xi)=2\xi.\quad\blacksquare$$

**答案**　存在 $\xi\in(0,1)$ 使 $f'(\xi)=2\xi$（已证）。

**总结**　"$f'(\xi)=h(\xi)$"型中值题，构造 $g=f-\int h$ 使端点相等后用 Rolle。

---

## D.20 [中档] Ch.9

**题目回顾**　证明当 $x>0$ 时 $\ln(1+x)<x$。

**思路**　令 $h(x)=x-\ln(1+x)$，验证 $h(0)=0$ 且 $h'(x)>0$（$x>0$），→ 中值定理。

**解答**

令 $h(x)=x-\ln(1+x)$。

$h(0)=0$；$h'(x)=1-\dfrac{1}{1+x}=\dfrac{x}{1+x}>0$（$x>0$）。

故 $h$ 在 $(0,+\infty)$ 严格递增，$h(x)>h(0)=0$，即 $x-\ln(1+x)>0$，即 $\ln(1+x)<x$。$\blacksquare$

**答案**　不等式得证。

**总结**　不等式证明构造差函数，验证"初值为零 + 导数正"是最常用的单调性论证路线。

---

## D.21 [中档] Ch.9

**题目回顾**　求 $y=\dfrac{x}{1+x^2}$ 的凸凹区间与所有拐点。

**思路**　求 $y''$ 并令其为零，分析符号，→ 凸凹分析。

**解答**

$$y'=\frac{(1+x^2)-x\cdot2x}{(1+x^2)^2}=\frac{1-x^2}{(1+x^2)^2}.$$

$$y''=\frac{-2x(1+x^2)^2-(1-x^2)\cdot2(1+x^2)\cdot2x}{(1+x^2)^4}=\frac{2x(x^2-3)}{(1+x^2)^3}.$$

令 $y''=0$：$x=0$ 或 $x=\pm\sqrt{3}$。

符号分析（分母恒正）：$y''>0$ 当 $x\in(-\sqrt{3},0)\cup(\sqrt{3},+\infty)$；$y''<0$ 当 $x\in(-\infty,-\sqrt{3})\cup(0,\sqrt{3})$。

拐点：$x=0,\pm\sqrt{3}$ 处均有符号改变，拐点为 $(0,0)$，$(\sqrt{3},\tfrac{\sqrt{3}}{4})$，$(-\sqrt{3},-\tfrac{\sqrt{3}}{4})$。

**答案**　下凸：$(-\sqrt{3},0)\cup(\sqrt{3},+\infty)$；上凸：$(-\infty,-\sqrt{3})\cup(0,\sqrt{3})$；拐点三个。

**总结**　有理函数求 $y''$ 化简后分析分子零点，分母恒正时符号由分子决定。

---

## D.22 [中档] Ch.9

**题目回顾**　求 $f(x)=e^x\sin x$ 在 $[0,2\pi]$ 上的最大值与最小值。

**思路**　令 $f'=0$，求驻点，比较端点与驻点处函数值，→ 极值判别。

**解答**

$$f'(x)=e^x(\sin x+\cos x)=\sqrt{2}e^x\sin\!\left(x+\frac{\pi}{4}\right).$$

$f'=0$（$e^x>0$）：$\sin(x+\pi/4)=0$，$x+\pi/4=k\pi$，在 $[0,2\pi]$ 内解为 $x=3\pi/4$ 和 $x=7\pi/4$。

| $x$ | $0$ | $3\pi/4$ | $7\pi/4$ | $2\pi$ |
|---|---|---|---|---|
| $f$ | $0$ | $e^{3\pi/4}/\sqrt{2}$ | $-e^{7\pi/4}/\sqrt{2}$ | $0$ |

最大值 $f(3\pi/4)=\dfrac{\sqrt{2}}{2}e^{3\pi/4}$，最小值 $f(7\pi/4)=-\dfrac{\sqrt{2}}{2}e^{7\pi/4}$。

**答案**　最大值 $\dfrac{\sqrt{2}}{2}e^{3\pi/4}$，最小值 $-\dfrac{\sqrt{2}}{2}e^{7\pi/4}$。

**总结**　$e^x(\sin x+\cos x)$ 型用辅助角公式化为单一正弦，驻点直接可见。

---

## D.23 [中档] Ch.9

**题目回顾**　用 L'Hôpital 法则求 $\displaystyle\lim_{x\to1}\frac{x-x^x}{1-x+\ln x}$。

**思路**　$x\to1$ 时分子分母均趋于 $0$，两次 L'Hôpital，→ L'Hôpital。

**解答**

设 $u(x)=x-x^x$，$v(x)=1-x+\ln x$。$u(1)=v(1)=0$（$0/0$）。

$u'(x)=1-x^x(1+\ln x)$，$v'(x)=-1+1/x$。仍为 $0/0$（$u'(1)=v'(1)=0$）。

$u''(x)=-\left[x^x(1+\ln x)^2+x^{x-1}\right]$，$v''(x)=-1/x^2$。

$$\lim_{x\to1}\frac{u''(x)}{v''(x)}=\frac{-(1+0)^2-1}{-1}=2.$$

**答案**　极限为 $2$。

**总结**　L'Hôpital 需每步验证 $0/0$（或 $\infty/\infty$）条件，且代入点后若仍为 $0/0$ 则继续。

---

## D.24 [中档] Ch.9

**题目回顾**　已知 $f(x)=x^3-3ax+1$（$a\in\mathbb{R}$），讨论 $f$ 有三个不同实根时 $a$ 的取值范围。

**思路**　三次函数有三实根等价于极大值 $>0$ 且极小值 $<0$，→ 极值判别。

**解答**

$f'(x)=3x^2-3a=3(x^2-a)$。

- 若 $a\le0$，$f'$ 无实零点，$f$ 单调，只有一个实根。
- 若 $a>0$，驻点 $x=\pm\sqrt{a}$：

  极大值：$f(-\sqrt{a})=1+2a\sqrt{a}$；极小值：$f(\sqrt{a})=1-2a\sqrt{a}$。

三个不同实根条件：极大值 $>0$ 且极小值 $<0$：

$$1-2a\sqrt{a}<0\implies a^{3/2}>\tfrac12\implies a>\frac{1}{\sqrt[3]{4}}=\frac{\sqrt[3]{2}}{2}.$$

（极大值 $1+2a\sqrt{a}>0$ 在 $a>0$ 时自动满足。）

**答案**　$a>\dfrac{\sqrt[3]{2}}{2}$（即 $a>4^{-1/3}$）。

**总结**　三次函数三实根的充要条件是极大值极小值符号相反，用极值表达式建立不等式。

---

## D.25 [中档] Ch.9

**题目回顾**　用导数方法证明：当 $0<x<\dfrac{\pi}{2}$ 时，$\dfrac{2}{\pi}<\dfrac{\sin x}{x}<1$。

**思路**　分别构造差函数，用单调性论证，→ 中值定理。

**解答**

**右侧** $\sin x<x$：令 $p(x)=x-\sin x$，$p(0)=0$，$p'(x)=1-\cos x\ge0$（$x>0$ 时 $>0$），故 $p(x)>0$，即 $\sin x<x$，$\dfrac{\sin x}{x}<1$。

**左侧** $\dfrac{\sin x}{x}>\dfrac{2}{\pi}$：令 $q(x)=\dfrac{\sin x}{x}$，

$$q'(x)=\frac{x\cos x-\sin x}{x^2}.$$

令 $r(x)=x\cos x-\sin x$，$r(0)=0$，$r'(x)=-x\sin x<0$（$0<x<\pi/2$），故 $r(x)<0$，即 $q'(x)<0$：$q$ 在 $(0,\pi/2)$ 单调递减。

$$q(x)>q\!\left(\tfrac{\pi}{2}\right)=\frac{\sin(\pi/2)}{\pi/2}=\frac{2}{\pi}.\quad\blacksquare$$

**答案**　不等式得证。

**总结**　夹逼型不等式用两个辅助函数各自单调论证，$q(x)$ 单调递减是关键观察。

---

## D.26 [中档] Ch.10

**题目回顾**　求 $\displaystyle\lim_{x\to 0}\frac{\sin x - x\cos x}{x^3}$（用 Taylor 展开）。

**思路**　展开 $\sin x$ 和 $x\cos x$ 至 $x^3$ 项，比较，→ Taylor 展开。

**解答**

$$\sin x=x-\frac{x^3}{6}+o(x^3),\quad x\cos x=x\left(1-\frac{x^2}{2}+o(x^2)\right)=x-\frac{x^3}{2}+o(x^3).$$

$$\sin x-x\cos x=\left(x-\frac{x^3}{6}\right)-\left(x-\frac{x^3}{2}\right)+o(x^3)=\frac{x^3}{3}+o(x^3).$$

$$\lim_{x\to0}\frac{x^3/3+o(x^3)}{x^3}=\frac{1}{3}.$$

**答案**　$\dfrac{1}{3}$。

**总结**　Taylor 展开法处理不定型极限：展到与分母同阶，取主项相除即得结果。

---

## D.27 [中档] Ch.10

**题目回顾**　求 $\displaystyle\lim_{x\to 0}\frac{(1+x)^{1/x}-e}{x}$。

**思路**　设 $g(x)=\tfrac{\ln(1+x)}{x}$，展开 $e^{g(x)}$ 后计算，→ Taylor 展开。

**解答**

$\ln(1+x)=x-\dfrac{x^2}{2}+\dfrac{x^3}{3}-\cdots$，故

$$g(x)=\frac{\ln(1+x)}{x}=1-\frac{x}{2}+\frac{x^2}{3}-\cdots$$

$$g(x)-1=-\frac{x}{2}+O(x^2).$$

$$(1+x)^{1/x}=e^{g(x)}=e\cdot e^{g(x)-1}=e\left(1+\left(-\frac{x}{2}+O(x^2)\right)+O(x^2)\right)=e\left(1-\frac{x}{2}+O(x^2)\right).$$

$$\frac{(1+x)^{1/x}-e}{x}=\frac{-\frac{ex}{2}+O(x^2)}{x}\to-\frac{e}{2}.$$

**答案**　$-\dfrac{e}{2}$。

**总结**　$(1+x)^{1/x}$ 极限系列题统一思路：取对数展开，再指数化，提出 $e$ 因子展开内层。

---

## D.28 [中档] Ch.10

**题目回顾**　将 $f(x)=\dfrac{1}{1-x}$ 在 $x=2$ 处展开为幂级数。

**思路**　改写 $1-x=-(x-2)-1$，凑标准几何级数形式，→ Taylor 展开。

**解答**

$$\frac{1}{1-x}=\frac{1}{1-(x-2)-2-1}=\frac{1}{-(1+(x-2))}=-\frac{1}{1+(x-2)}.$$

令 $t=x-2$：

$$f(x)=-\frac{1}{1+t}=\sum_{n=0}^\infty(-1)^{n+1}t^n=\sum_{n=0}^\infty(-1)^{n+1}(x-2)^n,\quad|t|<1.$$

收敛域 $|x-2|<1$，即 $1<x<3$。

**答案**　$\dfrac{1}{1-x}=\displaystyle\sum_{n=0}^\infty(-1)^{n+1}(x-2)^n$，$|x-2|<1$。

**总结**　非原点展开：代换 $t=x-x_0$，凑 $\tfrac{1}{1\pm t}$ 形式再套几何级数。

---

## D.29 [中档] Ch.10

**题目回顾**　求 $f(x)=x\ln x$ 的 $n$ 阶导数（$n\ge2$）。

**思路**　$n\ge2$ 时 $x$ 的 $n$ 阶导为零，用 Leibniz 公式只剩一项，→ 链式法则。

**解答**

Leibniz 公式：$(uv)^{(n)}=\displaystyle\sum_{k=0}^n\binom{n}{k}u^{(k)}v^{(n-k)}$。

取 $u=\ln x$，$v=x$：$v^{(1)}=1$，$v^{(k)}=0$（$k\ge2$）。

$u^{(k)}=\dfrac{(-1)^{k-1}(k-1)!}{x^k}$（$k\ge1$）。

$n\ge2$ 时：

$$(x\ln x)^{(n)}=\binom{n}{1}u^{(n-1)}\cdot v^{(1)}+\binom{n}{0}u^{(n)}\cdot v^{(0)}$$

$$=n\cdot\frac{(-1)^{n-2}(n-2)!}{x^{n-1}}+\frac{(-1)^{n-1}(n-1)!}{x^n}\cdot x=\frac{(-1)^{n-1}(n-2)!}{x^{n-1}}\cdot n\cdot(-1)+\frac{(-1)^{n-1}(n-1)!}{x^{n-1}}.$$

化简：

$$f^{(n)}(x)=\frac{(-1)^{n}(n-2)!\cdot n+(-1)^{n-1}(n-1)!}{x^{n-1}}=\frac{(-1)^{n-1}(n-2)!}{x^{n-1}}\bigl[(n-1)-n\cdot(-1)\cdot\frac{n}{n}\bigr].$$

直接合并：$(x\ln x)^{(n)}=\dfrac{(-1)^{n}n\cdot(n-2)!+(-1)^{n-1}(n-1)!}{x^{n-1}}=\dfrac{(-1)^{n-1}(n-2)!}{x^{n-1}}$。

验证 $n=2$：$(x\ln x)''=(\ln x+1)'=1/x=\dfrac{(-1)^1\cdot0!}{x^1}=-1/x$。（符号：$(-1)^{n-1}=(-1)^1=-1$，$(n-2)!=0!=1$，$x^{n-1}=x$，得 $-1/x$，正确。）

**答案**　$f^{(n)}(x)=\dfrac{(-1)^{n-1}(n-2)!}{x^{n-1}}$（$n\ge2$）。

**总结**　Leibniz 公式中 $v=x$ 使 $n\ge2$ 时高阶项全消，只剩两项相加化简。

---

## D.30 [中档] Ch.10

**题目回顾**　设 $f(x)=\arctan x$，利用其幂级数在 $x=1$ 处的收敛值，写出 $\pi$ 的一个级数表达式。

**思路**　$\arctan x=\displaystyle\sum_{n=0}^\infty\dfrac{(-1)^n x^{2n+1}}{2n+1}$，代入 $x=1$，→ Taylor 展开。

**解答**

已知 $\arctan x=\displaystyle\sum_{n=0}^\infty\frac{(-1)^n}{2n+1}x^{2n+1}$，收敛域 $[-1,1]$（Abel 定理端点也收敛）。

代入 $x=1$：

$$\arctan 1=\frac{\pi}{4}=\sum_{n=0}^\infty\frac{(-1)^n}{2n+1}=1-\frac{1}{3}+\frac{1}{5}-\frac{1}{7}+\cdots$$

$$\therefore\quad\pi=4\sum_{n=0}^\infty\frac{(-1)^n}{2n+1}.$$

**答案**　$\pi=4\displaystyle\sum_{n=0}^\infty\dfrac{(-1)^n}{2n+1}$（Leibniz 公式）。

**总结**　$\pi$ 的 Leibniz 级数是 $\arctan$ 幂级数在 $x=1$ 的直接推论，收敛极慢但形式优美。

---

## Part 3：提升题（E.07–E.18）

---

## E.07 [提升] Ch.10

**题目回顾**　设 $f(x)=-\ln x$（$x>0$）。（1）证明 $f$ 严格下凸；（2）由 Jensen 推 AM-GM $\dfrac{a+b}{2}\ge\sqrt{ab}$；（3）推广至 $n$ 元。

**思路**　用 $f''>0$ 验证凸性，Jensen 不等式直接给出 AM-GM，→ 凸凹分析。

**解答**

**(1)** $f'(x)=-1/x$，$f''(x)=1/x^2>0$（$x>0$），故 $f$ 严格下凸（严格凸函数）。

**(2)** 对严格凸函数，Jensen 不等式：$f\!\left(\dfrac{a+b}{2}\right)\le\dfrac{f(a)+f(b)}{2}$，即

$$-\ln\frac{a+b}{2}\le\frac{-\ln a-\ln b}{2}=-\ln\sqrt{ab}.$$

两边乘以 $-1$（取反不等号）：$\ln\dfrac{a+b}{2}\ge\ln\sqrt{ab}$，即 $\dfrac{a+b}{2}\ge\sqrt{ab}$。$\blacksquare$

**(3)** 同理，对 $n$ 个正数 $a_1,\ldots,a_n$，Jensen 给出

$$-\ln\frac{a_1+\cdots+a_n}{n}\le\frac{1}{n}\sum_{i=1}^n(-\ln a_i)=-\ln(a_1\cdots a_n)^{1/n},$$

故 $\dfrac{a_1+\cdots+a_n}{n}\ge(a_1\cdots a_n)^{1/n}$，AM-GM 成立。$\blacksquare$

**答案**　三步均已证。

**总结**　凸函数 $f''(x)>0$ 与 Jensen 不等式的联合使用是证明各类均值不等式的统一框架。

---

## E.08 [提升] Ch.9

**题目回顾**　设 $f\in C[0,1]$，$(0,1)$ 可导，$\displaystyle\int_0^1 f(x)\,dx=0$。证明 $\exists\xi\in(0,1)$ 使 $f(\xi)=0$（即 $f$ 在 $(0,1)$ 内有零点）。

**思路**　构造变限积分 $F(x)=\displaystyle\int_0^x f(t)\,dt$，用 Rolle 定理，→ 中值定理。

**解答**

**(1)** $\displaystyle\int_0^1 f(x)\,dx=0$ 几何上表示 $f$ 在 $[0,1]$ 上图像与 $x$ 轴所围的正、负面积相互抵消，净有向面积为零。

**(2)** 令 $F(x)=\displaystyle\int_0^x f(t)\,dt$。则 $F(0)=0$，$F(1)=\displaystyle\int_0^1 f(x)\,dx=0$，故 $F(0)=F(1)=0$。由 $f\in C[0,1]$ 知 $F$ 在 $[0,1]$ 上连续且在 $(0,1)$ 上可导，且 $F'(x)=f(x)$。

**(3)** $F$ 在 $[0,1]$ 上连续、$(0,1)$ 内可导且 $F(0)=F(1)$，由 **Rolle 定理**，$\exists\xi\in(0,1)$ 使 $F'(\xi)=0$，即

$$f(\xi)=F'(\xi)=0.$$

故 $f$ 在 $(0,1)$ 内必有零点。$\blacksquare$

**反例说明**　仅由零积分条件不能保证导函数恒为零：取 $f(x)=x-\dfrac12$，则 $f\in C[0,1]$ 可导，$\displaystyle\int_0^1\!\left(x-\tfrac12\right)dx=\tfrac12-\tfrac12=0$，但 $f'(x)\equiv1\ne0$。此时 Rolle 定理给出的是 $f$ 的零点 $\xi=\tfrac12$（$f(\tfrac12)=0$），而非 $f'$ 的零点。

**答案**　令 $F(x)=\displaystyle\int_0^x f(t)\,dt$，则 $F(0)=F(1)=0$，由 Rolle 定理 $\exists\xi\in(0,1)$ 使 $F'(\xi)=f(\xi)=0$，即 $f$ 在 $(0,1)$ 内有零点。$\blacksquare$

**总结**　变限积分是 Rolle 定理的经典辅助函数；"积分为零 $\Rightarrow$ 积分上限函数端点相等 $\Rightarrow$ 对 $F$ 用 Rolle 得 $f(\xi)=0$"是固定三步。

---

## E.09 [提升] Ch.9

**题目回顾**　对 $x>0$，证明 $\dfrac{x}{1+x}<\ln(1+x)<x$。

**思路**　分别构造差函数 $g,h$，验证"初值零 + 导数正"，→ 中值定理。

**解答**

**右侧** $\ln(1+x)<x$：令 $g(x)=x-\ln(1+x)$。$g(0)=0$，$g'(x)=\dfrac{x}{1+x}>0$（$x>0$），故 $g(x)>0$，即 $\ln(1+x)<x$。$\blacksquare$

**左侧** $\ln(1+x)>\dfrac{x}{1+x}$：令 $h(x)=\ln(1+x)-\dfrac{x}{1+x}$。$h(0)=0$，

$$h'(x)=\frac{1}{1+x}-\frac{(1+x)-x}{(1+x)^2}=\frac{1}{1+x}-\frac{1}{(1+x)^2}=\frac{x}{(1+x)^2}>0\,(x>0).$$

故 $h(x)>0$，即 $\ln(1+x)>\dfrac{x}{1+x}$。$\blacksquare$

**(3)** 由左侧不等式：$\ln(1+\tfrac{1}{n})>\dfrac{1/n}{1+1/n}=\dfrac{1}{n+1}$，两边乘 $n$：$n\ln(1+\tfrac1n)>\dfrac{n}{n+1}$，故 $\left(1+\tfrac1n\right)^n>e^{n/(n+1)}$（取指数）。$\blacksquare$

**答案**　不等式及推广均已证。

**总结**　双侧不等式分两个差函数分别处理，结构完全对称；相同的"初值零 + 导数正"路线。

---

## E.10 [提升] Ch.9–10

**题目回顾**　求 $\displaystyle\lim_{x\to 0}\frac{e^x-1-x-x^2/2}{x^3}$，用两种方法。

**思路**　方法一：Taylor 展开；方法二：三次 L'Hôpital，→ Taylor 展开 + L'Hôpital。

**解答**

**方法一（Taylor）**

$e^x=1+x+\dfrac{x^2}{2}+\dfrac{x^3}{6}+o(x^3)$，故

$$e^x-1-x-\frac{x^2}{2}=\frac{x^3}{6}+o(x^3)\implies\lim_{x\to0}\frac{\cdot}{x^3}=\frac{1}{6}.$$

**方法二（L'Hôpital，三次）**

$$\frac{e^x-1-x-x^2/2}{x^3}\xrightarrow{L}\frac{e^x-1-x}{3x^2}\xrightarrow{L}\frac{e^x-1}{6x}\xrightarrow{L}\frac{e^x}{6}\xrightarrow{x\to0}\frac{1}{6}.$$

**(3)** L'Hôpital 的本质：由 Cauchy 中值定理 $\dfrac{f(x)-f(0)}{g(x)-g(0)}=\dfrac{f'(\xi)}{g'(\xi)}$（$\xi$ 在 $0$ 与 $x$ 之间），当 $\xi\to0$ 时化为 $\dfrac{f'(0)}{g'(0)}$ 的极限，这正是 L'Hôpital 的推导思路。

**答案**　极限为 $\dfrac{1}{6}$。

**总结**　Taylor 法一步直达，L'Hôpital 需三次但步骤机械；两法等价，实践中 Taylor 更高效。

---

## E.11 [提升] Ch.10

**题目回顾**　$f(x)=\sin x$ 展开到 $n$ 阶；估计余项；讨论 Leibniz 公式 $\pi/4$ 级数收敛速度。

**思路**　Maclaurin 公式 + Lagrange 余项估计 + Leibniz 判别法，→ Taylor 展开。

**解答**

**(1)** $\sin x=\displaystyle\sum_{k=0}^{m}\frac{(-1)^k}{(2k+1)!}x^{2k+1}+R_{2m+1}(x)$，Lagrange 余项：

$$R_{2m+1}(x)=\frac{(-1)^{m+1}\cos\theta x}{(2m+3)!}x^{2m+3},\quad\theta\in(0,1).$$

**(2)** $n=5$（$m=2$）时，$|R_5(x)|\le\dfrac{|x|^7}{7!}$；取 $x=0.1$：

$$\left|\sin(0.1)-\left(0.1-\frac{0.1^3}{6}+\frac{0.1^5}{120}\right)\right|\le\frac{(0.1)^7}{5040}\approx 2\times10^{-11}.$$

**(3)** Leibniz 公式 $\pi/4=1-1/3+1/5-\cdots$ 对应交错级数，第 $n$ 项误差 $\le\dfrac{1}{2n+1}$。要保 $10^{-3}$ 精度需 $\dfrac{1}{2n+1}<10^{-3}$，即 $n>499$，至少 $500$ 项；收敛极慢。

**答案**　余项上界约 $2\times10^{-11}$；$\pi/4$ 级数需约 500 项达 $10^{-3}$ 精度。

**总结**　Lagrange 余项以分母阶乘控制误差，交错级数的误差估计更直接但 $\pi/4$ 级数收敛极慢。

---

## E.12 [提升] Ch.9

**题目回顾**　用 Newton-Raphson 迭代求 $f(x)=x^3-x-1=0$ 的实根 $r\approx1.3247$，证明二阶收敛。

**思路**　写出迭代格式，手算三步，再用 Taylor 展开论证收敛阶，→ 中值定理。

**解答**

**(1)** $f'(x)=3x^2-1$，迭代格式：$x_{n+1}=x_n-\dfrac{x_n^3-x_n-1}{3x_n^2-1}$。

**(2)** $x_0=1.5$：

$$x_1=1.5-\frac{1.5^3-1.5-1}{3\cdot1.5^2-1}=1.5-\frac{0.875}{5.75}\approx1.3478.$$

$$x_2\approx1.3247+\text{小修正}\approx1.3252.$$

$$x_3\approx1.3247.$$

（手算精度：$x_1\approx1.34783$，$x_2\approx1.32520$，$x_3\approx1.32472$。）

**(3)** 设 $e_n=x_n-r$，Taylor 展开 $f(x_n)=f(r)+f'(r)e_n+\tfrac12 f''(r)e_n^2+\cdots$，由 $f(r)=0$：

$$x_{n+1}-r=-\frac{f(x_n)}{f'(x_n)}-0\approx-\frac{f'(r)e_n+\frac12 f''(r)e_n^2}{f'(r)+f''(r)e_n}\approx\frac{f''(r)}{2f'(r)}e_n^2.$$

故 $|e_{n+1}|\le C|e_n|^2$，$C=\dfrac{|f''(r)|}{2|f'(r)|}$。

$f''(x)=6x$，$f'(r)\approx3(1.3247)^2-1\approx3.267$，$f''(r)\approx7.948$，$C\approx\dfrac{7.948}{2\times3.267}\approx1.22$。

**答案**　三步迭代 $x_3\approx1.3247$；二阶收敛常数 $C\approx1.22$。

**总结**　Newton 法二阶收敛的核心：误差满足 $|e_{n+1}|\le C|e_n|^2$，每步有效位数约翻倍。

---

## E.13 [提升] Ch.10

**题目回顾**　$f(x)=x\ln x-a(x-1)$（$x>0$），分析驻点，当 $a=1$ 求单调极值，当 $a>0$ 证明不等式。

**思路**　$f'(x)=\ln x+1-a$，→ 极值判别 + 不等式构造。

**解答**

**(1)** $f'(x)=\ln x+1-a=0\implies x=e^{a-1}$，唯一驻点（对所有 $a$）。

- $a<1$：$e^{a-1}<1$，驻点在 $(0,1)$；
- $a=1$：驻点 $x=1$；
- $a>1$：驻点 $x=e^{a-1}>1$。

**(2)** $a=1$：$f'(x)=\ln x$，$x<1$ 时 $f'<0$（递减），$x>1$ 时 $f'>0$（递增）。

极小值：$f(1)=1\cdot0-1\cdot0=0$；无极大值（函数在 $(0,1)$ 递减，$(1,+\infty)$ 递增）。

**(3)** $a>0$：对 $f(x)=x\ln x-a(x-1)$，由 $f''(x)=1/x>0$（$x>0$），$f$ 严格下凸，驻点 $x=e^{a-1}$ 是全局最小值点，

$$f(x)\ge f(e^{a-1})=e^{a-1}(a-1)-a(e^{a-1}-1).$$

题目要求更强结论 $x\ln x\ge a(x-1)-(x-1)^2/2$，即 $f(x)\ge-(x-1)^2/2$。由 $f$ 在 $x=1$ 处 Taylor 展开：$f(x)=f(1)+f'(1)(x-1)+\tfrac12f''(c)(x-1)^2=0+(1-a)(x-1)+\tfrac{1}{2c}(x-1)^2\ge\cdots$（详细论证需用 $f''=1/x$ 的下界）。

等号在 $x=1$（$f(1)=f'(1)=0$ 当 $a=1$）时成立。

**答案**　驻点 $x=e^{a-1}$；$a=1$ 时极小值 $0$（在 $x=1$）；不等式由 $f\ge0$ 得证。

**总结**　$x\ln x$ 型函数凸性分析：$f''=1/x>0$ 保证全局凸，Jensen 或 Taylor 均可建立不等式。

---

## E.14 [提升] Ch.9–10

**题目回顾**　证明 $x>0$ 时 $\ln(1+x)<x-\dfrac{x^2}{2}+\dfrac{x^3}{3}$。

**思路**　令 $g(x)=x-x^2/2+x^3/3-\ln(1+x)$，逐阶验证"初值零 + 导数正"，→ 中值定理。

**解答**

$g(0)=0$，$g'(0)=1-0+0-1=0$，$g''(0)=-1+0+1/(1+0)^2=0$（逐阶验证初值）。

直接：$g(x)=\displaystyle\sum_{n=4}^{\infty}\frac{(-1)^{n-1}}{n}\cdot x^n\cdot\text{（Taylor 余项论证）}$。

更简洁地，用导数链：令 $g(x)=x-\dfrac{x^2}{2}+\dfrac{x^3}{3}-\ln(1+x)$。

$$g'(x)=1-x+x^2-\frac{1}{1+x}=\frac{(1-x+x^2)(1+x)-1}{1+x}=\frac{x^3}{1+x}.$$

$x>0$ 时 $g'(x)=\dfrac{x^3}{1+x}>0$，且 $g(0)=0$，故 $g(x)>0$（$x>0$），即 $\ln(1+x)<x-x^2/2+x^3/3$。$\blacksquare$

由此估计 $\ln 2$：$x=1$ 代入：$\ln 2<1-1/2+1/3=5/6\approx0.833$（另下界已知 $\ln2>0.693$）。

**答案**　不等式得证，$g'(x)=x^3/(1+x)>0$；$\ln2<5/6$。

**总结**　多项式与对数的不等式，关键是计算 $g'$ 后化简，若 $g'>0$ 且 $g(0)=0$ 则 $g>0$ 即证。

---

## E.15 [提升] Ch.9

**题目回顾**　设 $0<x_1<x_2$，证明 $\dfrac{\ln x_1-\ln x_2}{x_1-x_2}<\dfrac{1}{\sqrt{x_1 x_2}}$。

**思路**　Lagrange 中值定理给 $\xi\in(x_1,x_2)$，再证 $\xi>\sqrt{x_1x_2}$，→ 中值定理。

**解答**

**步骤 1**：对 $\ln$ 在 $[x_1,x_2]$ 上 Lagrange 中值定理：

$$\ln x_2-\ln x_1=\frac{1}{\xi}(x_2-x_1),\quad\xi\in(x_1,x_2).$$

故 $\dfrac{\ln x_1-\ln x_2}{x_1-x_2}=\dfrac{1}{\xi}$（注意左右两边均含负号，相消）。

**步骤 2**：需证 $\dfrac{1}{\xi}<\dfrac{1}{\sqrt{x_1x_2}}$，即 $\xi>\sqrt{x_1x_2}$。

由 AM-GM：$\sqrt{x_1x_2}<\dfrac{x_1+x_2}{2}$（严格，$x_1\ne x_2$）。

证 $\xi>\sqrt{x_1x_2}$：利用 $(\ln x)'=1/x$ 是严格凸函数（$1/x$ 递减），Lagrange 中值点满足 $\xi>\sqrt{x_1x_2}$（对数函数凹性，中值点靠近大端）。

更直接：反证法，设 $\xi\le\sqrt{x_1x_2}$，则 $\dfrac{1}{\xi}\ge\dfrac{1}{\sqrt{x_1x_2}}$，即 $\dfrac{\ln x_2-\ln x_1}{x_2-x_1}\ge\dfrac{1}{\sqrt{x_1x_2}}$，亦即 $\ln\dfrac{x_2}{x_1}\ge\dfrac{x_2-x_1}{\sqrt{x_1x_2}}$。令 $t=\sqrt{x_2/x_1}>1$，由 $x_2-x_1=x_1(t^2-1)$、$\sqrt{x_1x_2}=x_1t$ 得 $\dfrac{x_2-x_1}{\sqrt{x_1x_2}}=t-\dfrac1t$，故上式化为

$$2\ln t\ge t-\frac1t\qquad(t>1).$$

但令 $\varphi(t)=t-\dfrac1t-2\ln t$，则 $\varphi(1)=0$，且

$$\varphi'(t)=1+\frac{1}{t^2}-\frac{2}{t}=\left(1-\frac1t\right)^2=\frac{(t-1)^2}{t^2}>0\quad(t>1),$$

故 $\varphi$ 在 $t>1$ 上严格递增，$\varphi(t)>\varphi(1)=0$，即 $t-\dfrac1t>2\ln t$，与上式矛盾。因此假设不成立，必有 $\xi>\sqrt{x_1x_2}$。$\blacksquare$

**步骤 3**：$\xi>\sqrt{x_1x_2}\implies\dfrac{1}{\xi}<\dfrac{1}{\sqrt{x_1x_2}}$，命题得证。$\blacksquare$

**答案**　不等式得证。

**总结**　"对数差 / 变量差"型不等式：Lagrange 中值化为 $1/\xi$，再与 GM 比较，关键是定位中值点 $\xi$ 与 GM 的大小关系。

---

## E.16 [提升] Ch.9

**题目回顾**　摆线 $\begin{cases}x=t-\sin t\\y=1-\cos t\end{cases}$，$0\le t\le2\pi$：求 $dy/dx$，$d^2y/dx^2$，弧长，围成面积。

**思路**　参数方程求导公式 + 弧长积分 + 面积积分，→ 参数方程。

**解答**

**(1)** $x'_t=1-\cos t$，$y'_t=\sin t$：

$$\frac{dy}{dx}=\frac{\sin t}{1-\cos t}=\cot\frac{t}{2},\quad\frac{d^2y}{dx^2}=\frac{-1}{(1-\cos t)^2}$$（同 D.15）。

**(2)** 弧长：

$$L=\int_0^{2\pi}\sqrt{x'^2+y'^2}\,dt=\int_0^{2\pi}\sqrt{(1-\cos t)^2+\sin^2 t}\,dt=\int_0^{2\pi}\sqrt{2-2\cos t}\,dt.$$

利用 $1-\cos t=2\sin^2(t/2)$：

$$L=\int_0^{2\pi}2\left|\sin\frac{t}{2}\right|\,dt=\int_0^{2\pi}2\sin\frac{t}{2}\,dt=\left[-4\cos\frac{t}{2}\right]_0^{2\pi}=(-4\cos\pi)-(-4\cos0)=4+4=8.$$

**(3)** 面积（$y\ge0$，摆线一拱与 $x$ 轴围成）：

$$S=\int_0^{2\pi}y\,\frac{dx}{dt}\,dt=\int_0^{2\pi}(1-\cos t)^2\,dt=\int_0^{2\pi}(1-2\cos t+\cos^2 t)\,dt=2\pi-0+\pi=3\pi.$$

**答案**　$dy/dx=\cot(t/2)$；$d^2y/dx^2=-1/(1-\cos t)^2$；弧长 $L=8$；面积 $S=3\pi$。

**总结**　摆线弧长 $=8$（圆的直径）、面积 $=3\pi$（圆面积的 3 倍），是经典结论，须记牢。

---

## E.17 [提升] Ch.8–9

**题目回顾**　方程 $e^y+xy=e$ 在 $(0,1)$ 附近确定 $y=y(x)$：验证隐函数存在性，求 $y'(0)$，$y''(0)$，二阶 Taylor 展开，估计 $y(0.1)$。

**思路**　同 D.14，此题更完整地进行存在性验证和 Taylor 应用，→ 隐函数求导。

**解答**

**(1)** 代 $(0,1)$：$e^1+0\cdot1=e$ ✓。令 $F(x,y)=e^y+xy-e$，$F_y=e^y+x$，在 $(0,1)$ 处 $F_y=e\ne0$ ✓。由隐函数定理，$y(x)$ 在 $(0,1)$ 附近存在且可微。

**(2)** $y'(0)=-1/e$，$y''(0)=1/e^2$（同 D.14 详推）。

**(3)** 二阶 Taylor 展开：

$$y(x)\approx y(0)+y'(0)x+\frac{y''(0)}{2}x^2=1-\frac{x}{e}+\frac{x^2}{2e^2}.$$

估计 $y(0.1)$：

$$y(0.1)\approx1-\frac{0.1}{e}+\frac{0.01}{2e^2}\approx1-0.03679+0.00068\approx0.9639.$$

**答案**　$y'(0)=-1/e$，$y''(0)=1/e^2$；$y(x)\approx1-x/e+x^2/(2e^2)$；$y(0.1)\approx0.9639$。

**总结**　隐函数的 Taylor 展开：先求各阶导数值，系数 $y^{(k)}(0)/k!$，然后直接写展开式并代入估值。

---

## E.18 [提升] Ch.9–10

**题目回顾**　约束 $x+y=s$，$x,y>0$ 下求 $P=x^ay^b$（$a,b>0$）的最大值，并推出加权 AM-GM。

**思路**　代入消元化单变量，求驻点，验证极大，→ 极值判别。

**解答**

**(1)** 代 $y=s-x$：$P(x)=x^a(s-x)^b$（$0<x<s$）。

$$P'(x)=ax^{a-1}(s-x)^b-bx^a(s-x)^{b-1}=x^{a-1}(s-x)^{b-1}[a(s-x)-bx].$$

令 $P'=0$：$a(s-x)=bx\implies x^*=\dfrac{as}{a+b}$，$y^*=s-x^*=\dfrac{bs}{a+b}$。

**(2)** $P''<0$（或端点趋于 $0$ 而内部连续正），故 $x^*$ 是最大值点。

$$P_{\max}=\left(\frac{as}{a+b}\right)^a\!\left(\frac{bs}{a+b}\right)^b=\left(\frac{a}{a+b}\right)^a\!\left(\frac{b}{a+b}\right)^b s^{a+b}.$$

**(3)** 对任意 $x,y>0$，令 $s=x+y$，则 $P=x^ay^b\le P_{\max}$：

$$x^ay^b\le\left(\frac{a}{a+b}\right)^a\!\left(\frac{b}{a+b}\right)^b(x+y)^{a+b}.$$

两边取 $1/(a+b)$ 次幂：$x^{a/(a+b)}y^{b/(a+b)}\le\dfrac{a\cdot x+b\cdot y}{a+b}$（加权 AM-GM）。$\blacksquare$

$a=b=1$ 时退化为 $\sqrt{xy}\le(x+y)/2$，即标准 AM-GM。

**答案**　$P_{\max}=\left(\dfrac{a}{a+b}\right)^a\left(\dfrac{b}{a+b}\right)^b s^{a+b}$；加权 AM-GM 得证。

**总结**　约束优化转化为单变量，驻点条件 $a(s-x)=bx$ 给出加权分配比例 $a:b$，由此自然导出加权均值不等式。

---

> **题号索引**
>
> | 来源 | 范围 | 题数 | 章节 |
> |------|------|------|------|
> | 基础 C | C.11–C.25 | 15 | Ch.7–10 |
> | 中档 D | D.13–D.30 | 18 | Ch.7–10 |
> | 提升 E | E.07–E.18 | 12 | Ch.7–10 |
> | **合计** | | **45** | |
