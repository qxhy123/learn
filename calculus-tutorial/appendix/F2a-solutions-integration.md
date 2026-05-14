# 附录 F2a：积分技巧详解（C.26-C.40, D.31-D.48, E.19-E.28）

> 共 **43 题**，涵盖 C 组基础 15 题（C.26–C.40）、D 组中档 18 题（D.31–D.48）、E 组提升 10 题（E.19–E.28）。
> 对应教材 Ch.11–14：不定积分、定积分、广义积分、积分应用。
> 每题格式：题目回顾 / 思路 / 解答 / 答案 / 总结。

---

## C 组基础（C.26–C.40）

---

## C.26 [基础] Ch.11

**题目回顾**：求 $\displaystyle\int(3x^2 - 2x + 5)\,dx$。

**思路**　逐项用幂函数积分公式 $\int x^n\,dx = \dfrac{x^{n+1}}{n+1}+C$。

**解答**　$\displaystyle\int(3x^2-2x+5)\,dx = 3\cdot\frac{x^3}{3} - 2\cdot\frac{x^2}{2} + 5x + C = x^3 - x^2 + 5x + C$。

**答案**　$x^3 - x^2 + 5x + C$

**总结**　多项式逐项积分，次数加 1 除以新次数，常数项积分得 $x$ 项。

---

## C.27 [基础] Ch.11

**题目回顾**：求 $\displaystyle\int \frac{1}{\sqrt{x}}\,dx$。

**思路**　改写 $\tfrac{1}{\sqrt{x}}=x^{-1/2}$，再用幂函数公式。

**解答**　$\displaystyle\int x^{-1/2}\,dx = \frac{x^{1/2}}{1/2}+C = 2\sqrt{x}+C$。

**答案**　$2\sqrt{x}+C$

**总结**　根式统一化为分数指数幂，避免计算失误。

---

## C.28 [基础] Ch.11

**题目回顾**：求 $\displaystyle\int e^{3x}\,dx$（换元法）。

**思路**　令 $u=3x$，$du=3\,dx$，利用 $\int e^u\,du=e^u+C$。

**解答**　令 $u=3x$，$dx=\tfrac{du}{3}$，则

$$\int e^{3x}\,dx = \int e^u\cdot\frac{du}{3} = \frac{1}{3}e^u+C = \frac{1}{3}e^{3x}+C.$$

**答案**　$\dfrac{1}{3}e^{3x}+C$

**总结**　线性换元 $\int e^{ax}\,dx=\tfrac{1}{a}e^{ax}+C$，凑微分时系数取倒数补偿。

---

## C.29 [基础] Ch.11

**题目回顾**：求 $\displaystyle\int\sin^2 x\,dx$（利用半角公式）。

**思路**　用半角公式 $\sin^2 x=\tfrac{1-\cos 2x}{2}$ 降次，化为可积形式。

**解答**

$$\int\sin^2 x\,dx = \int\frac{1-\cos 2x}{2}\,dx = \frac{x}{2} - \frac{\sin 2x}{4}+C.$$

**答案**　$\dfrac{x}{2}-\dfrac{\sin 2x}{4}+C$

**总结**　$\sin^2 x$ 与 $\cos^2 x$ 都靠半角公式降次，$\sin^2 x$ 对应减号，$\cos^2 x$ 对应加号。

---

## C.30 [基础] Ch.11

**题目回顾**：求 $\displaystyle\int xe^x\,dx$（分部积分）。

**思路**　LIATE 原则：令 $u=x$，$dv=e^x\,dx$，分部积分一次即可。

**解答**　$u=x,\,dv=e^x\,dx \Rightarrow du=dx,\,v=e^x$。

$$\int xe^x\,dx = xe^x - \int e^x\,dx = xe^x - e^x + C = (x-1)e^x+C.$$

**答案**　$(x-1)e^x+C$

**总结**　分部公式 $\int u\,dv = uv-\int v\,du$；多项式×指数型：多项式取 $u$，指数取 $dv$，一次完成。

---

## C.31 [基础] Ch.12

**题目回顾**：计算 $\displaystyle\int_0^1(2x+1)\,dx$。

**思路**　先求不定积分，再用牛顿–莱布尼茨公式代入上下限。

**解答**　$F(x)=x^2+x$，故

$$\int_0^1(2x+1)\,dx = F(1)-F(0) = (1+1)-0 = 2.$$

**答案**　$2$

**总结**　N–L 公式是定积分计算的基础工具；注意原函数不含常数 $C$，代入上下限作差。

---

## C.32 [基础] Ch.12

**题目回顾**：计算 $\displaystyle\int_0^{\pi/2}\cos x\,dx$。

**思路**　$\cos x$ 的原函数是 $\sin x$，直接代入。

**解答**

$$\int_0^{\pi/2}\cos x\,dx = [\sin x]_0^{\pi/2} = \sin\tfrac{\pi}{2}-\sin 0 = 1-0 = 1.$$

**答案**　$1$

**总结**　$[\sin x]$ 在 $[0,\pi/2]$ 从 $0$ 增到 $1$，结果就是 $1$；几何上等于从 $0$ 到 $\pi/2$ 余弦曲线下的面积。

---

## C.33 [基础] Ch.12

**题目回顾**：计算 $\displaystyle\int_1^e\frac{\ln x}{x}\,dx$（换元令 $u=\ln x$）。

**思路**　令 $u=\ln x$，则 $du=\tfrac{1}{x}\,dx$，换元后积分变为 $\int_0^1 u\,du$。

**解答**　换元：$u=\ln x$，$x=1\Rightarrow u=0$，$x=e\Rightarrow u=1$。

$$\int_1^e\frac{\ln x}{x}\,dx = \int_0^1 u\,du = \left[\frac{u^2}{2}\right]_0^1 = \frac{1}{2}.$$

**答案**　$\dfrac{1}{2}$

**总结**　换元后上下限同步变换，避免回代；"$\ln x/x$"型题的标准换元是 $u=\ln x$。

---

## C.34 [基础] Ch.12

**题目回顾**：计算 $\displaystyle\int_{-1}^1 x^3\,dx$，说明为何结果为 $0$。

**思路**　$f(x)=x^3$ 是奇函数，在对称区间 $[-1,1]$ 上积分为零。

**解答**　由奇函数性质：若 $f$ 是奇函数且 $[-a,a]$ 对称，则 $\displaystyle\int_{-a}^a f(x)\,dx=0$。

$x^3$ 满足 $f(-x)=-f(x)$，故 $\displaystyle\int_{-1}^1 x^3\,dx=0$。

直接验证：$\left[\tfrac{x^4}{4}\right]_{-1}^1=\tfrac{1}{4}-\tfrac{1}{4}=0$ ✓

**答案**　$0$

**总结**　奇函数在对称区间积分恒为零；偶函数在对称区间积分可折半 $\int_{-a}^a f=2\int_0^a f$。

---

## C.35 [基础] Ch.12

**题目回顾**：计算 $\displaystyle\int_0^\pi\sin x\,dx$ 并给出几何解释。

**思路**　直接积分；几何上是 $[0,\pi]$ 上正弦曲线围成的面积。

**解答**

$$\int_0^\pi\sin x\,dx = [-\cos x]_0^\pi = (-\cos\pi)-(-\cos 0) = 1+1 = 2.$$

几何意义：$\sin x$ 在 $[0,\pi]$ 上非负，积分值等于曲线 $y=\sin x$ 与 $x$ 轴围成的弓形面积，其值为 $2$。

**答案**　$2$

**总结**　$[-\cos x]_0^\pi$ 易出符号错误；$-\cos\pi=-(-1)=1$，$-\cos 0=-1$，差为 $2$。

---

## C.36 [基础] Ch.11

**题目回顾**：求 $\displaystyle\int\frac{x}{1+x^2}\,dx$（凑微分）。

**思路**　观察分子 $x\,dx$ 恰好是分母 $1+x^2$ 微分的一半，凑出 $d(1+x^2)$。

**解答**

$$\int\frac{x}{1+x^2}\,dx = \frac{1}{2}\int\frac{2x\,dx}{1+x^2} = \frac{1}{2}\int\frac{d(1+x^2)}{1+x^2} = \frac{1}{2}\ln(1+x^2)+C.$$

**答案**　$\dfrac{1}{2}\ln(1+x^2)+C$

**总结**　"分子是分母导数倍数"立即凑微分：$\int\tfrac{f'(x)}{f(x)}dx=\ln|f(x)|+C$。此处分母恒正，去掉绝对值。

---

## C.37 [基础] Ch.11

**题目回顾**：求 $\displaystyle\int x\ln x\,dx$（分部积分）。

**思路**　LIATE：对数优先取 $u$，令 $u=\ln x$，$dv=x\,dx$。

**解答**　$u=\ln x,\,v=\tfrac{x^2}{2}$，故

$$\int x\ln x\,dx = \frac{x^2}{2}\ln x - \int\frac{x^2}{2}\cdot\frac{1}{x}\,dx = \frac{x^2}{2}\ln x - \frac{1}{2}\int x\,dx = \frac{x^2}{2}\ln x - \frac{x^2}{4}+C.$$

**答案**　$\dfrac{x^2}{2}\ln x - \dfrac{x^2}{4}+C$

**总结**　对数乘多项式型：对数取 $u$，分部一次降去 $\ln$；结果含 $x^2\ln x$ 与 $x^2$ 两项。

---

## C.38 [基础] Ch.13

**题目回顾**：讨论广义积分 $\displaystyle\int_1^{+\infty}\frac{1}{x^2}\,dx$ 的收敛性并求值。

**思路**　$p$-积分：$\int_1^{+\infty}x^{-p}\,dx$ 当 $p>1$ 时收敛，$p\le 1$ 时发散；此处 $p=2>1$，收敛。

**解答**

$$\int_1^{+\infty}\frac{dx}{x^2} = \lim_{b\to+\infty}\left[-\frac{1}{x}\right]_1^b = \lim_{b\to+\infty}\left(-\frac{1}{b}+1\right) = 1.$$

**答案**　收敛，值为 $1$

**总结**　$p$-积分判别：$p>1$ 收敛，$p\le 1$ 发散；$p=2$ 是收敛的典型例子。

---

## C.39 [基础] Ch.13

**题目回顾**：计算广义积分 $\displaystyle\int_0^{+\infty}\frac{dx}{1+x^2}$。

**思路**　被积函数有初等原函数 $\arctan x$，取极限即可。

**解答**

$$\int_0^{+\infty}\frac{dx}{1+x^2} = \lim_{b\to+\infty}[\arctan x]_0^b = \frac{\pi}{2}-0 = \frac{\pi}{2}.$$

**答案**　$\dfrac{\pi}{2}$

**总结**　$\arctan(+\infty)=\pi/2$，$\arctan(-\infty)=-\pi/2$；此积分是经典结论，直接记忆。

---

## C.40 [基础] Ch.14

**题目回顾**：求曲线 $y=x^2$ 与直线 $y=x$ 围成区域的面积。

**思路**　先求交点确定积分区间，再用上减下积分。

**解答**　联立 $x^2=x$，得 $x=0$ 或 $x=1$，即在 $[0,1]$ 上 $x\ge x^2$。

$$S=\int_0^1(x-x^2)\,dx = \left[\frac{x^2}{2}-\frac{x^3}{3}\right]_0^1 = \frac{1}{2}-\frac{1}{3} = \frac{1}{6}.$$

**答案**　$\dfrac{1}{6}$

**总结**　面积 = 上方曲线 - 下方曲线，积分区间由交点确定；抛物线与直线围成面积的标准套路。

---

## D 组中档（D.31–D.48）

---

## D.31 [中档] Ch.11

**题目回顾**：求 $\displaystyle\int\frac{x}{\sqrt{x^2+2x+5}}\,dx$。

**思路**　分母配方：$x^2+2x+5=(x+1)^2+4$；分子拆成 $\tfrac{1}{2}(2x+2)-1$，凑微分项加标准型。

**解答**　将分子写成 $x=\tfrac{1}{2}(2x+2)-1$，故

$$I = \frac{1}{2}\int\frac{(2x+2)\,dx}{\sqrt{(x+1)^2+4}} - \int\frac{dx}{\sqrt{(x+1)^2+4}}.$$

第一项：令 $u=(x+1)^2+4$，$du=(2x+2)dx$，得 $\tfrac{1}{2}\int u^{-1/2}\,du=\sqrt{(x+1)^2+4}$。

第二项：$\int\dfrac{d(x+1)}{\sqrt{(x+1)^2+4}}=\ln\!\left|(x+1)+\sqrt{(x+1)^2+4}\right|$。

$$I = \sqrt{x^2+2x+5} - \ln\left|x+1+\sqrt{x^2+2x+5}\right|+C.$$

**答案**　$\sqrt{x^2+2x+5}-\ln\!\left|x+1+\sqrt{x^2+2x+5}\right|+C$

**总结**　根号下二次式先配方，分子凑导数再加标准双曲型。

---

## D.32 [中档] Ch.11

**题目回顾**：求 $\displaystyle\int e^{2x}\cos x\,dx$。

**思路**　设 $I=\int e^{2x}\cos x\,dx$，分部两次后"循环"，移项解 $I$。

**解答**　第一次分部（$u=\cos x,\,dv=e^{2x}dx$）：

$$I = \frac{e^{2x}\cos x}{2}+\frac{1}{2}\int e^{2x}\sin x\,dx.$$

第二次分部（$u=\sin x,\,dv=e^{2x}dx$）：

$$\int e^{2x}\sin x\,dx = \frac{e^{2x}\sin x}{2}-\frac{1}{2}\int e^{2x}\cos x\,dx = \frac{e^{2x}\sin x}{2}-\frac{I}{2}.$$

代回：$I=\tfrac{e^{2x}\cos x}{2}+\tfrac{1}{2}\!\left(\tfrac{e^{2x}\sin x}{2}-\tfrac{I}{2}\right)$，化简得 $\tfrac{5}{4}I=\tfrac{e^{2x}(2\cos x+\sin x)}{4}$，故

$$I = \frac{e^{2x}(2\cos x+\sin x)}{5}+C.$$

**答案**　$\dfrac{e^{2x}(2\cos x+\sin x)}{5}+C$

**总结**　指数×三角型：分部两次出现 $I$，系数为 $a^2+b^2$（此处 $4+1=5$），移项即得。

---

## D.33 [中档] Ch.11

**题目回顾**：求 $\displaystyle\int\frac{dx}{1+e^x}$。

**思路**　分子写成 $(1+e^x)-e^x$，拆开后第二项凑 $\tfrac{e^x}{1+e^x}$ 的导数。

**解答**

$$\int\frac{dx}{1+e^x} = \int\left(1-\frac{e^x}{1+e^x}\right)dx = x - \ln(1+e^x)+C.$$

**答案**　$x-\ln(1+e^x)+C$

**总结**　含 $e^x$ 的分式先分子加减技巧，化为 $1-\tfrac{e^x}{1+e^x}$，后者是对数的凑微分。

---

## D.34 [中档] Ch.11

**题目回顾**：求 $\displaystyle\int\frac{\sin x}{1+\sin x}\,dx$。

**思路**　分子写成 $(1+\sin x)-1$，然后对 $\tfrac{1}{1+\sin x}$ 乘共轭 $1-\sin x$，利用 $\sec^2 x-\tan x\sec x$ 积分。

**解答**

$$\int\frac{\sin x}{1+\sin x}\,dx = \int\left(1-\frac{1}{1+\sin x}\right)dx.$$

对 $\tfrac{1}{1+\sin x}$，乘以共轭 $\dfrac{1-\sin x}{1-\sin x}$：

$$\frac{1}{1+\sin x} = \frac{1-\sin x}{\cos^2 x} = \sec^2 x-\tan x\sec x.$$

$$\int(\sec^2 x-\tan x\sec x)\,dx = \tan x-\sec x+C.$$

故

$$I = x-(\tan x-\sec x)+C = x-\tan x+\sec x+C.$$

**答案**　$x-\tan x+\sec x+C$

**总结**　"含 $\sin x$ 分式"乘共轭化为 $\sec^2,\tan\sec$ 标准型，这是三角积分的常用技巧。

---

## D.35 [中档] Ch.11

**题目回顾**：求 $\displaystyle\int x\arctan x\,dx$。

**思路**　LIATE：$u=\arctan x$（反三角优先），$dv=x\,dx$，分部一次。

**解答**　$u=\arctan x,\,v=\tfrac{x^2}{2}$，

$$\int x\arctan x\,dx = \frac{x^2}{2}\arctan x-\int\frac{x^2}{2}\cdot\frac{1}{1+x^2}\,dx.$$

注意 $\dfrac{x^2}{1+x^2}=1-\dfrac{1}{1+x^2}$，故

$$\int\frac{x^2}{2(1+x^2)}\,dx = \frac{1}{2}\left(x-\arctan x\right)+C.$$

因此

$$I = \frac{x^2}{2}\arctan x-\frac{x}{2}+\frac{\arctan x}{2}+C = \frac{x^2+1}{2}\arctan x-\frac{x}{2}+C.$$

**答案**　$\dfrac{x^2+1}{2}\arctan x-\dfrac{x}{2}+C$

**总结**　反三角取 $u$，分部后余项用"多项式÷$(1+x^2)$"处理，结果含两个 $\arctan$ 项可合并。

---

## D.36 [中档] Ch.11

**题目回顾**：求 $\displaystyle\int\frac{dx}{\sqrt{x}+\sqrt[3]{x}}$（令 $x=t^6$，化为有理函数）。

**思路**　根号 2 与根号 3 的最小公倍数为 6，令 $x=t^6$，消去根式化为有理函数。

**解答**　令 $x=t^6$（$t>0$），则 $dx=6t^5\,dt$，$\sqrt{x}=t^3$，$\sqrt[3]{x}=t^2$。

$$I=\int\frac{6t^5\,dt}{t^3+t^2}=\int\frac{6t^5}{t^2(t+1)}\,dt=6\int\frac{t^3}{t+1}\,dt.$$

多项式除法：$\dfrac{t^3}{t+1}=t^2-t+1-\dfrac{1}{t+1}$，故

$$6\int\left(t^2-t+1-\frac{1}{t+1}\right)dt = 6\left(\frac{t^3}{3}-\frac{t^2}{2}+t-\ln|t+1|\right)+C.$$

回代 $t=x^{1/6}$：

$$I = 2x^{1/2}-3x^{1/3}+6x^{1/6}-6\ln(x^{1/6}+1)+C.$$

**答案**　$2\sqrt{x}-3\sqrt[3]{x}+6x^{1/6}-6\ln(x^{1/6}+1)+C$

**总结**　混合根式取各指数的公分母幂次换元，换元后做多项式除法处理有理分式。

---

## D.37 [中档] Ch.11

**题目回顾**：求 $\displaystyle\int\frac{\ln(1+x)}{x^2}\,dx$。

**思路**　分部：$u=\ln(1+x)$，$dv=x^{-2}\,dx$，余项再用部分分式。

**解答**　$u=\ln(1+x),\,v=-\tfrac{1}{x}$，

$$I = -\frac{\ln(1+x)}{x}+\int\frac{1}{x(1+x)}\,dx.$$

部分分式：$\dfrac{1}{x(1+x)}=\dfrac{1}{x}-\dfrac{1}{1+x}$，故

$$\int\frac{dx}{x(1+x)} = \ln|x|-\ln|1+x|+C.$$

$$I = -\frac{\ln(1+x)}{x}+\ln x-\ln(1+x)+C \quad(x>0).$$

**答案**　$-\dfrac{\ln(1+x)}{x}+\ln x-\ln(1+x)+C$

**总结**　分子为对数时分部，余项常含 $\tfrac{1}{x(1+x)}$ 型，部分分式分解是标准收尾。

---

## D.38 [中档] Ch.11

**题目回顾**：求 $\displaystyle\int\frac{dx}{x\sqrt{1-\ln^2 x}}$（令 $u=\ln x$）。

**思路**　$\tfrac{1}{x}\,dx=d(\ln x)$，换元后变为标准反正弦型。

**解答**　令 $u=\ln x$，$du=\tfrac{dx}{x}$，

$$I=\int\frac{du}{\sqrt{1-u^2}} = \arcsin u+C = \arcsin(\ln x)+C.$$

**答案**　$\arcsin(\ln x)+C$

**总结**　看到 $\tfrac{1}{x\sqrt{1-\ln^2 x}}$，立即令 $u=\ln x$，被积函数化为 $\tfrac{1}{\sqrt{1-u^2}}$，原函数为 $\arcsin$。

---

## D.39 [中档] Ch.12

**题目回顾**：计算 $\displaystyle\int_0^{\pi/2}\sin^4 x\,dx$（Wallis 公式）。

**思路**　Wallis 公式：$n=4$ 为偶数，$I_4=\tfrac{3}{4}\cdot\tfrac{1}{2}\cdot\tfrac{\pi}{2}$。

**解答**　由 Wallis 公式（$n$ 偶）：

$$I_n = \frac{n-1}{n}\cdot\frac{n-3}{n-2}\cdots\frac{1}{2}\cdot\frac{\pi}{2}.$$

取 $n=4$：

$$I_4 = \frac{3}{4}\cdot\frac{1}{2}\cdot\frac{\pi}{2} = \frac{3\pi}{16}.$$

**答案**　$\dfrac{3\pi}{16}$

**总结**　$\sin^n$ 或 $\cos^n$ 在 $[0,\pi/2]$ 的 Wallis 公式：偶数结果含 $\pi/2$，奇数不含 $\pi$。

---

## D.40 [中档] Ch.12

**题目回顾**：计算 $\displaystyle\int_{-1}^1\frac{x^2}{1+e^x}\,dx$。

**思路**　对称区间：令 $x\to-x$ 得另一表达式，两式相加利用 $\tfrac{1}{1+e^x}+\tfrac{e^x}{1+e^x}=1$。

**解答**　设 $I=\displaystyle\int_{-1}^1\frac{x^2}{1+e^x}\,dx$，令 $x\to-x$：

$$I=\int_{-1}^1\frac{x^2}{1+e^{-x}}\,dx=\int_{-1}^1\frac{x^2 e^x}{1+e^x}\,dx.$$

两式相加：$2I=\displaystyle\int_{-1}^1 x^2\left(\frac{1}{1+e^x}+\frac{e^x}{1+e^x}\right)dx=\int_{-1}^1 x^2\,dx=\frac{2}{3}$。

故 $I=\dfrac{1}{3}$。

**答案**　$\dfrac{1}{3}$

**总结**　含 $e^x$ 分母的对称积分：利用 $f(x)+f(-x)$ 消去 $e^x$，是高频技巧。

---

## D.41 [中档] Ch.12

**题目回顾**：计算 $\displaystyle\int_0^\pi x\sin x\,dx$。

**思路**　分部积分：$u=x$，$dv=\sin x\,dx$。

**解答**

$$\int_0^\pi x\sin x\,dx = [-x\cos x]_0^\pi+\int_0^\pi\cos x\,dx = \pi+[\sin x]_0^\pi = \pi+0 = \pi.$$

**答案**　$\pi$

**总结**　$x$ 乘三角型分部：$u=x$，$dv=$ 三角；$[-x\cos x]_0^\pi=\pi$，$[\sin x]_0^\pi=0$。

---

## D.42 [中档] Ch.12

**题目回顾**：计算 $\displaystyle\int_0^1\frac{\ln(1+x)}{1+x^2}\,dx$。

**思路**　令 $x=\tan t$，换元后对 $t\to\tfrac{\pi}{4}-t$ 的对称变换，两式相加利用对数加法恒等式。

**解答**　令 $x=\tan t$，$t\in[0,\pi/4]$，$dx=\sec^2 t\,dt$，$1+x^2=\sec^2 t$：

$$I=\int_0^{\pi/4}\ln(1+\tan t)\,dt.$$

令 $u=\tfrac{\pi}{4}-t$，$\tan u=\dfrac{1-\tan t}{1+\tan t}$，则

$$I=\int_0^{\pi/4}\ln\!\left(1+\frac{1-\tan t}{1+\tan t}\right)dt=\int_0^{\pi/4}\ln\!\frac{2}{1+\tan t}\,dt=\frac{\pi}{4}\ln 2-I.$$

故 $2I=\dfrac{\pi\ln 2}{4}$，$I=\dfrac{\pi\ln 2}{8}$。

**答案**　$\dfrac{\pi\ln 2}{8}$

**总结**　含 $\ln(1+x)/(1+x^2)$ 换元为 $\arctan$ 型后，"折叠"对称技巧是关键。

---

## D.43 [中档] Ch.12

**题目回顾**：计算 $\displaystyle\int_0^{\pi/2}\ln\sin x\,dx$。

**思路**　利用 $\int_0^{\pi/2}\ln\sin x=\int_0^{\pi/2}\ln\cos x$，再用二倍角公式拼接。

**解答**　设 $I=\displaystyle\int_0^{\pi/2}\ln\sin x\,dx$，由 $x\to\tfrac{\pi}{2}-x$，知 $I=\int_0^{\pi/2}\ln\cos x\,dx$。

$$2I=\int_0^{\pi/2}(\ln\sin x+\ln\cos x)\,dx=\int_0^{\pi/2}\ln\frac{\sin 2x}{2}\,dx=\int_0^{\pi/2}\ln\sin 2x\,dx-\frac{\pi}{2}\ln 2.$$

令 $u=2x$，$\int_0^{\pi/2}\ln\sin 2x\,dx=\tfrac{1}{2}\int_0^\pi\ln\sin u\,du=I$（利用 $\int_0^\pi\ln\sin u\,du=2I$）。

故 $2I=I-\dfrac{\pi}{2}\ln 2$，即 $I=-\dfrac{\pi}{2}\ln 2$。

**答案**　$-\dfrac{\pi\ln 2}{2}$

**总结**　此为经典结论，推导关键在"$2I$ 等于自身 $+$ 修正"的循环等式；结果含 $\pi\ln 2$。

---

## D.44 [中档] Ch.13

**题目回顾**：讨论 $\displaystyle\int_0^{+\infty}\frac{dx}{x^p(1+x)}$ 的收敛性。

**思路**　在两端分别判断：$x\to0^+$ 看 $x^{-p}$ 的幂次，$x\to+\infty$ 看 $x^{-p-1}$ 的幂次。

**解答**　分拆为 $\int_0^1+\int_1^{+\infty}$。

- $x\to0^+$：$\tfrac{1}{x^p(1+x)}\sim x^{-p}$，收敛条件为 $p<1$。
- $x\to+\infty$：$\tfrac{1}{x^p(1+x)}\sim x^{-p-1}$，收敛条件为 $p+1>1$，即 $p>0$。

两端均收敛当且仅当 $0<p<1$。

**答案**　收敛当且仅当 $0<p<1$

**总结**　含两端奇异的广义积分必须分拆后各自判断；$p$-积分判别是基本工具。

---

## D.45 [中档] Ch.13

**题目回顾**：计算 Gauss 积分 $\displaystyle\int_0^{+\infty}e^{-x^2}\,dx$。

**思路**　设 $I=\int_0^{+\infty}e^{-x^2}\,dx$，则 $I^2=\iint_{\mathbb{R}^2_+}e^{-(x^2+y^2)}\,dA$，极坐标计算。

**解答**　$(2I)^2=\left(\displaystyle\int_{-\infty}^{+\infty}e^{-x^2}\,dx\right)^2=\iint_{\mathbb{R}^2}e^{-(x^2+y^2)}\,dA$。

极坐标：

$$\iint_{\mathbb{R}^2}e^{-r^2}r\,dr\,d\theta = \int_0^{2\pi}\,d\theta\int_0^{+\infty}e^{-r^2}r\,dr = 2\pi\cdot\frac{1}{2}=\pi.$$

故 $(2I)^2=\pi$，$I=\dfrac{\sqrt{\pi}}{2}$。

**答案**　$\dfrac{\sqrt{\pi}}{2}$

**总结**　Gauss 积分的二维极坐标法是标准推导；结论 $\int_0^\infty e^{-x^2}\,dx=\tfrac{\sqrt\pi}{2}$ 要记住。

---

## D.46 [中档] Ch.14

**题目回顾**：求 $y=x^2$ 与 $y=\sqrt{x}$ 围成区域的面积，及该区域绕 $x$ 轴旋转所得旋转体体积。

**思路**　交点 $(0,0),(1,1)$；面积用上减下积分；体积用圆盘法（上曲线的 $\pi y^2$ 减下曲线的 $\pi y^2$）。

**解答**　在 $[0,1]$ 上 $\sqrt{x}\ge x^2$。

**面积**：$S=\displaystyle\int_0^1(\sqrt{x}-x^2)\,dx=\left[\tfrac{2}{3}x^{3/2}-\tfrac{x^3}{3}\right]_0^1=\tfrac{2}{3}-\tfrac{1}{3}=\dfrac{1}{3}$。

**体积**：$V=\pi\displaystyle\int_0^1\left[(\sqrt{x})^2-(x^2)^2\right]dx=\pi\int_0^1(x-x^4)\,dx=\pi\left[\tfrac{x^2}{2}-\tfrac{x^5}{5}\right]_0^1=\dfrac{3\pi}{10}$。

**答案**　面积 $\dfrac{1}{3}$，体积 $\dfrac{3\pi}{10}$

**总结**　绕 $x$ 轴旋转体积：$V=\pi\int(y_{\text{上}}^2-y_{\text{下}}^2)\,dx$，注意两条曲线都要包括。

---

## D.47 [中档] Ch.14

**题目回顾**：求 $y=\ln x$ 在 $[1,e]$ 段绕 $x$ 轴旋转的旋转体体积。

**思路**　圆盘法：$V=\pi\displaystyle\int_1^e\ln^2 x\,dx$，分部积分两次。

**解答**

$$V=\pi\int_1^e\ln^2 x\,dx.$$

第一次分部（$u=\ln^2 x$，$dv=dx$）：

$$\int\ln^2 x\,dx=x\ln^2 x-2\int\ln x\,dx.$$

第二次（$\int\ln x\,dx=x\ln x-x$）：

$$\int\ln^2 x\,dx=x\ln^2 x-2x\ln x+2x+C.$$

代入 $[1,e]$：$[x\ln^2 x-2x\ln x+2x]_1^e=(e-2e+2e)-(0-0+2)=e-2$。

$$V=\pi(e-2).$$

**答案**　$\pi(e-2)$

**总结**　$\int\ln^2 x\,dx$ 分部两次，每次降低 $\ln$ 的次数；最终结果 $\pi(e-2)$ 是常考结论。

---

## D.48 [中档] Ch.14

**题目回顾**：用变限积分 + L'Hôpital 求 $\displaystyle\lim_{x\to 0}\frac{\int_0^{x^2}f(t)\,dt}{x^4}$，其中 $f$ 连续且 $f(0)=0,f'(0)=2$。

**思路**　分子分母均趋 $0$，对分子用变限积分求导（链式法则），再次用 L'Hôpital。

**解答**　第一次 L'Hôpital（分子对 $x$ 求导用链式法则：$\tfrac{d}{dx}\int_0^{x^2}f(t)\,dt=f(x^2)\cdot 2x$）：

$$\lim_{x\to0}\frac{2xf(x^2)}{4x^3}=\lim_{x\to0}\frac{f(x^2)}{2x^2}.$$

令 $u=x^2\to0$：$\displaystyle\lim_{u\to0}\frac{f(u)}{2u}$。再次 L'Hôpital（$f(0)=0$）：$\dfrac{f'(0)}{2}=\dfrac{2}{2}=1$。

**答案**　$1$

**总结**　变限积分对参数求导靠链式法则；若仍为 $0/0$ 型则继续 L'Hôpital 或泰勒展开。

---

## E 组提升（E.19–E.28）

---

## E.19 [提升] Ch.13 万能代换

**题目回顾**：计算 $\displaystyle\int\frac{dx}{1+\sin x+\cos x}$。

**思路**　令 $t=\tan(x/2)$（万能代换），将三角有理式化为 $t$ 的有理分式。

**解答**　令 $t=\tan(x/2)$：$\sin x=\dfrac{2t}{1+t^2}$，$\cos x=\dfrac{1-t^2}{1+t^2}$，$dx=\dfrac{2\,dt}{1+t^2}$。

分母：$1+\dfrac{2t}{1+t^2}+\dfrac{1-t^2}{1+t^2}=\dfrac{(1+t^2)+2t+(1-t^2)}{1+t^2}=\dfrac{2(1+t)}{1+t^2}$。

$$I=\int\frac{\dfrac{2}{1+t^2}}{\dfrac{2(1+t)}{1+t^2}}\,dt=\int\frac{dt}{1+t}=\ln|1+t|+C=\ln\!\left|1+\tan\frac{x}{2}\right|+C.$$

**答案**　$\ln\!\left|1+\tan\dfrac{x}{2}\right|+C$

**总结**　万能代换 $t=\tan(x/2)$ 将任意三角有理式化为代数有理式，适合分母含 $1\pm\sin x\pm\cos x$ 的不定积分。

---

## E.20 [提升] Ch.11–12 分部积分循环

**题目回顾**：计算 $\displaystyle\int e^{2x}\cos x\,dx$，验证结果系数分母为 $2^2+1^2=5$。

**思路**　设 $I=\int e^{2x}\cos x\,dx$，分部两次后含 $I$ 的等式移项求解。

**解答**　第一次（$u=\cos x$，$dv=e^{2x}dx$）：

$$I=\frac{e^{2x}\cos x}{2}+\frac{1}{2}\int e^{2x}\sin x\,dx.$$

第二次（$u=\sin x$，$dv=e^{2x}dx$）：

$$\int e^{2x}\sin x\,dx=\frac{e^{2x}\sin x}{2}-\frac{1}{2}I.$$

代回：$I=\dfrac{e^{2x}\cos x}{2}+\dfrac{e^{2x}\sin x}{4}-\dfrac{I}{4}$，整理 $\dfrac{5}{4}I=\dfrac{e^{2x}(2\cos x+\sin x)}{4}$，

$$I=\frac{e^{2x}(2\cos x+\sin x)}{5}+C.$$

分母 $5=2^2+1^2$，与 $e^{ax}\cos bx$ 型的一般公式一致。

**答案**　$\dfrac{e^{2x}(2\cos x+\sin x)}{5}+C$

**总结**　$e^{ax}\cos bx$ 型：分母为 $a^2+b^2$，分子系数为 $(a\cos bx+b\sin bx)$，可直接套用。

---

## E.21 [提升] Ch.13 根式代换有理化

**题目回顾**：计算 $\displaystyle\int\frac{dx}{\sqrt{x}+\sqrt[3]{x}}$。

**思路**　$\text{lcm}(2,3)=6$，令 $x=t^6$，化为有理函数 $\int\tfrac{6t^3}{t+1}\,dt$，多项式除法。

**解答**　$x=t^6$，$dx=6t^5\,dt$，$\sqrt{x}=t^3$，$\sqrt[3]{x}=t^2$：

$$I=\int\frac{6t^5\,dt}{t^3+t^2}=6\int\frac{t^3}{t+1}\,dt.$$

除法：$t^3=(t+1)(t^2-t+1)-1$，故 $\dfrac{t^3}{t+1}=t^2-t+1-\dfrac{1}{t+1}$。

$$I=6\left(\frac{t^3}{3}-\frac{t^2}{2}+t-\ln|t+1|\right)+C=2x^{1/2}-3x^{1/3}+6x^{1/6}-6\ln(x^{1/6}+1)+C.$$

**答案**　$2\sqrt{x}-3\sqrt[3]{x}+6x^{1/6}-6\ln(x^{1/6}+1)+C$

**总结**　混合根式换元：公分母幂次，除法化为多项式 + 真分式；每个根式分别回代。

---

## E.22 [提升] Ch.11–12 Wallis 积分递推

**题目回顾**：证明 $I_n=\int_0^{\pi/2}\sin^n x\,dx$ 满足 $I_n=\tfrac{n-1}{n}I_{n-2}$，并计算 $I_6$。

**思路**　对 $I_n$ 分部（$u=\sin^{n-1}x$，$dv=\sin x\,dx$），推出递推式。

**解答**　**递推推导**：$u=\sin^{n-1}x$，$v=-\cos x$，

$$I_n=\left[-\cos x\sin^{n-1}x\right]_0^{\pi/2}+(n-1)\int_0^{\pi/2}\cos^2 x\sin^{n-2}x\,dx.$$

边界项为零；$\cos^2 x=1-\sin^2 x$，故 $(n-1)(I_{n-2}-I_n)$，整理得 $nI_n=(n-1)I_{n-2}$，即

$$I_n=\frac{n-1}{n}I_{n-2}.$$

**计算 $I_6$**：$I_0=\dfrac{\pi}{2}$，$I_2=\dfrac{1}{2}I_0=\dfrac{\pi}{4}$，$I_4=\dfrac{3}{4}I_2=\dfrac{3\pi}{16}$，$I_6=\dfrac{5}{6}I_4=\dfrac{5\pi}{32}$。

**答案**　$I_6=\dfrac{5\pi}{32}$

**总结**　Wallis 递推：偶数步每次乘以 $\tfrac{奇}{偶}$，末尾乘 $\tfrac{\pi}{2}$；奇数步末尾为 $1$。

---

## E.23 [提升] Ch.12 含参定积分对称技巧

**题目回顾**：计算 $\displaystyle\int_0^1\frac{\ln(1+x)}{1+x^2}\,dx$。

**思路**　令 $x=\tan t$，换元后设 $I=\int_0^{\pi/4}\ln(1+\tan t)\,dt$，对 $t\to\tfrac{\pi}{4}-t$ 作对称，两式相加利用 $(1+\tan t)(1+\tan(\pi/4-t))=2$。

**解答**　令 $x=\tan t$，$I=\int_0^{\pi/4}\ln(1+\tan t)\,dt$。

令 $u=\tfrac{\pi}{4}-t$，$\tan(\tfrac{\pi}{4}-t)=\dfrac{1-\tan t}{1+\tan t}$：

$$I=\int_0^{\pi/4}\ln\!\left(1+\frac{1-\tan t}{1+\tan t}\right)dt=\int_0^{\pi/4}\ln\!\frac{2}{1+\tan t}\,dt=\frac{\pi}{4}\ln 2-I.$$

故 $2I=\dfrac{\pi\ln 2}{4}$，$I=\dfrac{\pi\ln 2}{8}$。

**答案**　$\dfrac{\pi\ln 2}{8}$

**总结**　含参定积分对称技巧："令 $u=a-x$" 后两式相加消去复杂项；关键等式 $(1+\tan t)(1+\tan(\pi/4-t))=2$。

---

## E.24 [提升] Ch.12 对称区间 $e^x$ 技巧

**题目回顾**：计算 $\displaystyle\int_{-1}^1\frac{x^2}{1+e^x}\,dx$。

**思路**　设 $I$ 并令 $x\to-x$，两式相加利用 $\tfrac{1}{1+e^x}+\tfrac{e^x}{1+e^x}=1$，化为偶函数积分。

**解答**　$I=\displaystyle\int_{-1}^1\frac{x^2}{1+e^x}\,dx$，令 $x\to-x$：$I=\displaystyle\int_{-1}^1\frac{x^2 e^x}{1+e^x}\,dx$。

$$2I=\int_{-1}^1 x^2\,dx=\frac{2}{3},\quad I=\frac{1}{3}.$$

**答案**　$\dfrac{1}{3}$

**总结**　"$e^x$ 分母 + 对称区间"型的通用公式：$f(x)+f(-x)=g(x)$（偶函数），积分化简为 $\tfrac12\int g$。

---

## E.25 [提升] Ch.14 广义积分判敛

**题目回顾**：讨论 $\displaystyle\int_0^{+\infty}\frac{dx}{x^p(1+x)}$ 的收敛性，并在 $p=1/2$ 时求精确值。

**思路**　分拆两端，分别用 $p$-积分比较；$p=1/2$ 利用 Beta 函数 $B(1/2,1/2)=\pi$。

**解答**　**收敛条件**：$0<p<1$（详见 D.44）。

**$p=1/2$ 时**：令 $x=t^2$（$dx=2t\,dt$），

$$\int_0^{+\infty}\frac{dx}{\sqrt{x}(1+x)}=\int_0^{+\infty}\frac{2t\,dt}{t(1+t^2)}=2\int_0^{+\infty}\frac{dt}{1+t^2}=2\cdot\frac{\pi}{2}=\pi.$$

（也可用 Beta 函数：$B(1-p,p)=\pi/\sin(\pi p)$，$p=1/2$ 给出 $\pi/\sin(\pi/2)=\pi$。）

**答案**　当 $0<p<1$ 时收敛；$p=1/2$ 时积分值为 $\pi$

**总结**　两端发散判断要分拆；$p=1/2$ 换元后转化为 $\arctan$ 型。Beta 函数公式可快速给出精确值。

---

## E.26 [提升] Ch.12–13 换序积分

**题目回顾**：计算 $\displaystyle\int_0^1\frac{e^x-1}{x}\,dx$（给出级数表示）。

**思路**　将 $e^x-1=\sum_{n=1}^\infty\tfrac{x^n}{n!}$ 代入，逐项积分，验证可换序。

**解答**　由 $e^x-1=\displaystyle\sum_{n=1}^\infty\frac{x^n}{n!}$，故 $\dfrac{e^x-1}{x}=\displaystyle\sum_{n=1}^\infty\frac{x^{n-1}}{n!}$，在 $[0,1]$ 上一致收敛，可逐项积分：

$$\int_0^1\frac{e^x-1}{x}\,dx=\sum_{n=1}^\infty\frac{1}{n\cdot n!}.$$

数值估计（前四项）：$1+\tfrac{1}{4}+\tfrac{1}{18}+\tfrac{1}{96}\approx1+0.25+0.0556+0.0104\approx1.316$（精确值约 $1.3179$）。

此级数与指数积分 $\mathrm{Ei}(1)=\gamma+\sum_{n=1}^\infty\tfrac{1}{n\cdot n!}$ 相差欧拉常数 $\gamma\approx0.5772$。

**答案**　$\displaystyle\sum_{n=1}^\infty\frac{1}{n\cdot n!}$（约 $1.3179$）

**总结**　"无初等原函数"积分靠级数展开逐项积分；需验证一致收敛才能换序。

---

## E.27 [提升] Ch.14 Gamma 函数递推

**题目回顾**：利用 $\Gamma(s)=\displaystyle\int_0^{+\infty}t^{s-1}e^{-t}\,dt$（$s>0$），证递推并计算 $\Gamma(7/2)$。

**思路**　分部推导 $\Gamma(s+1)=s\Gamma(s)$，由 $\Gamma(1/2)=\sqrt\pi$ 逐步计算 $\Gamma(7/2)$。

**解答**　**递推**：$u=t^{s-1}$，$dv=e^{-t}\,dt$，

$$\Gamma(s+1)=\left[-t^{s-1}e^{-t}\right]_0^\infty+(s-1)\int_0^\infty t^{s-2}e^{-t}\,dt=s\Gamma(s).$$

**$\Gamma(7/2)$**：

$$\Gamma\!\left(\frac{7}{2}\right)=\frac{5}{2}\Gamma\!\left(\frac{5}{2}\right)=\frac{5}{2}\cdot\frac{3}{2}\cdot\Gamma\!\left(\frac{3}{2}\right)=\frac{5}{2}\cdot\frac{3}{2}\cdot\frac{1}{2}\cdot\Gamma\!\left(\frac{1}{2}\right)=\frac{15}{8}\sqrt{\pi}.$$

**答案**　$\Gamma(7/2)=\dfrac{15\sqrt\pi}{8}$

**总结**　$\Gamma$ 函数递推是分部积分的典范；半整数 $\Gamma$ 值含 $\sqrt\pi$，每步乘以 $\tfrac{n-1}{2}$。

---

## E.28 [提升] Ch.14 Dirichlet 积分

**题目回顾**：考察 $\displaystyle\int_0^{+\infty}\frac{\sin x}{x}\,dx$，证明条件收敛并求值。

**思路**　Dirichlet 判别证收敛；$|\sin x/x|$ 用比较法证不绝对收敛；含参积分 $I(t)=\int_0^\infty e^{-tx}\tfrac{\sin x}{x}\,dx$ 对 $t$ 求导法求值。

**解答**　**条件收敛**：$\tfrac{1}{x}$ 单调趋零，$|\int_0^A\sin x\,dx|\le2$ 有界，Dirichlet 判别知积分收敛。

**非绝对收敛**：$\displaystyle\int_0^{+\infty}\left|\frac{\sin x}{x}\right|dx\ge\sum_{k=1}^\infty\int_{k\pi}^{(k+1)\pi}\frac{|\sin x|}{(k+1)\pi}\,dx=\frac{2}{\pi}\sum_{k=1}^\infty\frac{1}{k+1}=+\infty$。

**求值**：设 $I(t)=\displaystyle\int_0^\infty e^{-tx}\frac{\sin x}{x}\,dx$（$t>0$），对 $t$ 求导：

$$I'(t)=-\int_0^\infty e^{-tx}\sin x\,dx=-\frac{1}{1+t^2}.$$

由 $I(\infty)=0$，积分反推：$I(t)=\displaystyle\int_t^\infty\frac{du}{1+u^2}=\frac{\pi}{2}-\arctan t$。

令 $t\to0^+$：$\displaystyle\int_0^{+\infty}\frac{\sin x}{x}\,dx=\frac{\pi}{2}$。

**答案**　$\dfrac{\pi}{2}$（条件收敛）

**总结**　Dirichlet 积分的含参微分法是经典技巧：对 $t$ 求导去掉 $x$ 分母，积分后反推初值；结论 $\pi/2$ 是重要常数。

---

> **方法分类索引**
>
> | 方法 | 题号 |
> |------|------|
> | 幂函数/基本公式 | C.26, C.27 |
> | 换元法（凑微分） | C.28, C.36, D.38 |
> | 半角/三角降次 | C.29, D.39 |
> | 分部积分（一次） | C.30, C.37, D.35, D.37, D.41 |
> | 分部积分（循环） | D.32, E.20 |
> | 牛顿–莱布尼茨公式 | C.31, C.32, C.33 |
> | 奇偶函数性质 | C.34, D.40, E.24 |
> | 对称区间技巧 | D.40, E.24 |
> | 广义积分/判敛 | C.38, C.39, D.44, E.25, E.28 |
> | 面积/体积应用 | C.40, D.46, D.47 |
> | 配方 + 标准型 | D.31 |
> | 有理化/根式换元 | D.34, D.36, E.21 |
> | 变限积分 + L'Hôpital | D.48 |
> | 万能代换 | E.19 |
> | Wallis 公式/递推 | D.39, E.22 |
> | 含参定积分对称 | D.42, D.43, E.23 |
> | Gauss 积分 | D.45 |
> | 旋转体体积 | D.46, D.47 |
> | 级数换序积分 | E.26 |
> | Gamma 函数 | E.27 |
> | Dirichlet 积分 | E.28 |
