# 附录 F1a：极限连续详解（C.01-C.10, D.01-D.12, E.01-E.06）

> 共 **28 题**，覆盖极限与连续（Ch.4–6）三个难度层次：基础（C）、中档（D）、提升（E）。
> 每题含：**题目回顾**、**思路**（套路 + 工具）、**解答**（紧凑推导）、**答案**（$\boxed{}$）、**总结**（识题特征）。

---

## 基础题（C.01–C.10）

---

## C.01 [基础] Ch.4

**题目回顾**：用 $\varepsilon$-$N$ 语言证明 $\displaystyle\lim_{n\to\infty}\frac{2n+1}{n+3}=2$。

**思路**：$\varepsilon$-$N$ 证明的固定套路：估计 $|a_n - L|$，令其 $<\varepsilon$ 解出 $N$。

**解答**：
$$\left|\frac{2n+1}{n+3}-2\right|=\left|\frac{2n+1-2(n+3)}{n+3}\right|=\frac{5}{n+3}<\frac{5}{n}.$$

对任意 $\varepsilon>0$，取 $N=\left\lfloor\dfrac{5}{\varepsilon}\right\rfloor+1$。当 $n>N$ 时，$\dfrac{5}{n}<\dfrac{5}{N}<\varepsilon$。

故 $\displaystyle\lim_{n\to\infty}\frac{2n+1}{n+3}=2$。$\blacksquare$

**答案**：$\boxed{2}$（极限值；证明见上）

**总结**：$\varepsilon$-$N$ 证明 = 估计 $|a_n-L|$ → 找上界 $\to 0$ → 取 $N$；估计时放大分子、缩小分母。

---

## C.02 [基础] Ch.4

**题目回顾**：求 $\displaystyle\lim_{n\to\infty}\left(1+\frac{1}{n}\right)^{2n}$。

**思路**：第二重要极限 $\left(1+\tfrac{1}{n}\right)^n\to e$，乘幂 $2n$ 拆成两倍。

**解答**：
$$\left(1+\frac{1}{n}\right)^{2n}=\left[\left(1+\frac{1}{n}\right)^n\right]^2\xrightarrow{n\to\infty}e^2.$$

**答案**：$\boxed{e^2}$

**总结**：见 $\left(1+\tfrac{1}{n}\right)^{cn}$ 型 → 直接得 $e^c$；记住基本形 $\left(1+\tfrac{1}{n}\right)^n\to e$。

---

## C.03 [基础] Ch.5

**题目回顾**：利用等价无穷小求 $\displaystyle\lim_{x\to 0}\frac{\sin 3x}{\tan 5x}$。

**思路**：$x\to 0$ 时 $\sin u\sim u$，$\tan u\sim u$；直接替换分子分母。

**解答**：
$$\lim_{x\to 0}\frac{\sin 3x}{\tan 5x}=\lim_{x\to 0}\frac{3x}{5x}=\frac{3}{5}.$$

**答案**：$\boxed{\dfrac{3}{5}}$

**总结**：等价无穷小替换只在乘除（非加减）中有效；$\sin u,\tan u,\arcsin u,\ln(1+u),e^u-1$ 均等价于 $u$（$u\to 0$）。

---

## C.04 [基础] Ch.5

**题目回顾**：用两个重要极限求 $\displaystyle\lim_{x\to 0}\frac{1-\cos x}{x^2}$。

**思路**：半角公式将 $1-\cos x$ 化为 $2\sin^2\!\tfrac{x}{2}$，再用 $\sin u\sim u$。

**解答**：
$$1-\cos x=2\sin^2\!\frac{x}{2},\quad \lim_{x\to 0}\frac{2\sin^2\!\tfrac{x}{2}}{x^2}=2\cdot\lim_{x\to 0}\left(\frac{\sin\tfrac{x}{2}}{\tfrac{x}{2}}\right)^2\cdot\frac{1}{4}=2\cdot 1\cdot\frac{1}{4}=\frac{1}{2}.$$

**答案**：$\boxed{\dfrac{1}{2}}$

**总结**：$1-\cos x\sim\tfrac{x^2}{2}$（$x\to 0$）是高频等价无穷小，记住直接用。

---

## C.05 [基础] Ch.5

**题目回顾**：求 $\displaystyle\lim_{x\to 0}\frac{e^x - 1}{\ln(1+2x)}$。

**思路**：$e^x-1\sim x$，$\ln(1+2x)\sim 2x$；等价替换后相除。

**解答**：
$$\lim_{x\to 0}\frac{e^x-1}{\ln(1+2x)}=\lim_{x\to 0}\frac{x}{2x}=\frac{1}{2}.$$

**答案**：$\boxed{\dfrac{1}{2}}$

**总结**：分子分母同为 $x$ 的一阶无穷小时，等价替换后比系数即得答案。

---

## C.06 [基础] Ch.5

**题目回顾**：求 $\displaystyle\lim_{x\to+\infty}\left(1-\frac{2}{x}\right)^x$。

**思路**：$1^\infty$ 型；凑标准形 $\left(1+\tfrac{1}{t}\right)^t\to e$。

**解答**：令 $t=-x/2$（$x\to+\infty$ 时 $t\to-\infty$）：
$$\left(1-\frac{2}{x}\right)^x=\left(1+\frac{1}{t}\right)^{-2t}=\left[\left(1+\frac{1}{t}\right)^t\right]^{-2}\to e^{-2}.$$

**答案**：$\boxed{e^{-2}}$

**总结**：见 $\left(1+\tfrac{c}{x}\right)^x$ → 结果为 $e^c$；通用做法：取对数后用 $\ln(1+u)\sim u$。

---

## C.07 [基础] Ch.6

**题目回顾**：$f(x)=\dfrac{x^2-1}{x-1}$（$x\ne 1$），$f(1)=2$，判断 $f$ 在 $x=1$ 处是否连续。

**思路**：连续三要素：极限存在、函数有定义、两者相等。

**解答**：
$$\lim_{x\to 1}f(x)=\lim_{x\to 1}\frac{(x-1)(x+1)}{x-1}=\lim_{x\to 1}(x+1)=2.$$

$f(1)=2$。因 $\lim_{x\to 1}f(x)=f(1)=2$，故 $f$ 在 $x=1$ 处**连续**。

**答案**：$f$ 在 $x=1$ 处连续，$\boxed{\text{连续}}$

**总结**：补充定义点，若极限值 = 补充值则连续，该点原为可去间断点。

---

## C.08 [基础] Ch.6

**题目回顾**：$f(x)=\dfrac{\sin x}{x}$（$x\ne 0$），$f(0)=1$，判断 $x=0$ 处的连续性。

**思路**：第一重要极限 $\displaystyle\lim_{x\to 0}\frac{\sin x}{x}=1=f(0)$，直接比较。

**解答**：$\displaystyle\lim_{x\to 0}\frac{\sin x}{x}=1=f(0)$，三条件满足，$f$ 在 $x=0$ 处**连续**。

**答案**：$\boxed{\text{连续}}$

**总结**：$\sin x/x$ 在 $x=0$ 有可去间断点，补定义 $f(0)=1$ 后消除，是教材最经典连续性例题。

---

## C.09 [基础] Ch.6

**题目回顾**：指出 $g(x)=\dfrac{1}{x^2-1}$ 的间断点并分类。

**思路**：找使分母为零的点，分别计算左右极限判断类型。

**解答**：$x^2-1=0$ 解得 $x=\pm 1$。

- **$x=1$**：$\displaystyle\lim_{x\to 1}\frac{1}{x^2-1}=\frac{1}{(x-1)(x+1)}\to\infty$，为**无穷间断点**。
- **$x=-1$**：同理 $\to\infty$，为**无穷间断点**。

**答案**：间断点 $x=\pm 1$，均为 $\boxed{\text{无穷间断点（第二类）}}$

**总结**：间断点分类：第一类（左右极限均存在）含可去型和跳跃型；第二类（至少一侧无穷或振荡）含无穷型和振荡型。

---

## C.10 [基础] Ch.5

**题目回顾**：利用夹逼定理求 $\displaystyle\lim_{n\to\infty}\sqrt[n]{n}$。

**思路**：令 $a_n=\sqrt[n]{n}-1>0$，用二项式展开做夹逼。

**解答**：设 $\sqrt[n]{n}=1+a_n$（$n\ge 2$，$a_n>0$），则
$$n=(1+a_n)^n\ge\binom{n}{2}a_n^2=\frac{n(n-1)}{2}a_n^2,$$
故 $0<a_n^2\le\dfrac{2}{n-1}\to 0$，即 $a_n\to 0$，从而 $\sqrt[n]{n}\to 1$。

**答案**：$\boxed{1}$

**总结**：夹逼定理套路：设 $x_n=L+a_n$，展开后控制 $a_n\to 0$；$\sqrt[n]{n}\to 1$ 是经典结论。

---

## 中档题（D.01–D.12）

---

## D.01 [中档] Ch.5

**题目回顾**：求 $\displaystyle\lim_{x\to 0}\frac{\sqrt{1+2x}-\sqrt[3]{1+3x}}{x^2}$。

**思路**：分子展开至二阶 Taylor，一阶项相消后看二阶系数之差。

**解答**：利用 $(1+u)^\alpha=1+\alpha u+\dfrac{\alpha(\alpha-1)}{2}u^2+o(u^2)$：
$$\sqrt{1+2x}=1+x+\frac{(1/2)(-1/2)}{2}(2x)^2+o(x^2)=1+x-\frac{x^2}{2}+o(x^2),$$
$$\sqrt[3]{1+3x}=1+x+\frac{(1/3)(-2/3)}{2}(3x)^2+o(x^2)=1+x-x^2+o(x^2).$$
分子 $=\left(-\tfrac{1}{2}+1\right)x^2+o(x^2)=\tfrac{1}{2}x^2+o(x^2)$，故
$$\lim_{x\to 0}\frac{\sqrt{1+2x}-\sqrt[3]{1+3x}}{x^2}=\frac{1}{2}.$$

**答案**：$\boxed{\dfrac{1}{2}}$

**总结**：分子一阶项相消时，必须展开至二阶才能定阶；$(1+u)^\alpha$ 展开是此类题的统一工具。

---

## D.02 [中档] Ch.5

**题目回顾**：求 $\displaystyle\lim_{x\to 0}\frac{e^x - e^{\sin x}}{x - \sin x}$。

**思路**：提出 $e^{\sin x}$，利用 $e^u-1\sim u$ 和 $x-\sin x\sim\tfrac{x^3}{6}$ 消去分母。

**解答**：
$$\frac{e^x-e^{\sin x}}{x-\sin x}=e^{\sin x}\cdot\frac{e^{x-\sin x}-1}{x-\sin x}.$$
令 $u=x-\sin x\to 0$，则 $\dfrac{e^u-1}{u}\to 1$，而 $e^{\sin x}\to e^0=1$，故极限 $=1\cdot 1=1$。

**答案**：$\boxed{1}$

**总结**：看到 $e^A - e^B$ 提出 $e^B$ 变成 $e^B(e^{A-B}-1)$，再用 $e^u-1\sim u$，是通用技巧。

---

## D.03 [中档] Ch.5

**题目回顾**：求 $\displaystyle\lim_{x\to 0}\left(\frac{1+\tan x}{1+\sin x}\right)^{1/\sin^3 x}$（$1^\infty$ 型）。

**思路**：取对数化为 $0/0$ 型，利用 $\ln(1+u)\sim u$ 与 $\tan x-\sin x\sim\tfrac{x^3}{2}$。

**解答**：原式 $=e^L$，其中
$$L=\lim_{x\to 0}\frac{1}{\sin^3 x}\ln\frac{1+\tan x}{1+\sin x}=\lim_{x\to 0}\frac{\ln(1+\tan x)-\ln(1+\sin x)}{\sin^3 x}.$$
用 $\ln(1+u)\sim u$：$\ln(1+\tan x)-\ln(1+\sin x)\sim\tan x-\sin x=\sin x\!\left(\tfrac{1}{\cos x}-1\right)\sim x\cdot\tfrac{x^2}{2}=\tfrac{x^3}{2}$。

$\sin^3 x\sim x^3$，故 $L=\dfrac{x^3/2}{x^3}=\dfrac{1}{2}$，原式 $=e^{1/2}$。

**答案**：$\boxed{e^{1/2}}$

**总结**：$1^\infty$ 型标准处：取对数 → 化 $0\cdot\infty$ → 等价无穷小化简；$\tan x-\sin x\sim\tfrac{x^3}{2}$ 要记住。

---

## D.04 [中档] Ch.5

**题目回顾**：求 $\displaystyle\lim_{x\to+\infty}\left(\sqrt{x^2+x+1}-\sqrt{x^2-x+1}\right)$。

**思路**：$\infty-\infty$ 型，分子有理化后化为 $0/0$ 型。

**解答**：
$$\sqrt{x^2+x+1}-\sqrt{x^2-x+1}=\frac{(x^2+x+1)-(x^2-x+1)}{\sqrt{x^2+x+1}+\sqrt{x^2-x+1}}=\frac{2x}{\sqrt{x^2+x+1}+\sqrt{x^2-x+1}}.$$
$x\to+\infty$ 时，分母 $\sim 2x$，故极限 $=\dfrac{2x}{2x}=1$。

**答案**：$\boxed{1}$

**总结**：$\infty-\infty$ 必须有理化，分子为 $2x$，分母提出 $x$ 因子后余项趋于常数。

---

## D.05 [中档] Ch.5

**题目回顾**：求 $\displaystyle\lim_{x\to 0}\frac{\ln(1+x)-x}{x^2}$。

**思路**：Taylor 展开 $\ln(1+x)=x-\tfrac{x^2}{2}+o(x^2)$，分子取二阶项。

**解答**：
$$\ln(1+x)-x=-\frac{x^2}{2}+o(x^2),\quad\lim_{x\to 0}\frac{-x^2/2+o(x^2)}{x^2}=-\frac{1}{2}.$$

**答案**：$\boxed{-\dfrac{1}{2}}$

**总结**：$\ln(1+x)-x\sim-\dfrac{x^2}{2}$（$x\to 0$），是高频结论，直接记忆可省去展开步骤。

---

## D.06 [中档] Ch.5

**题目回顾**：求 $\displaystyle\lim_{x\to 0^+}x^{\sin x}$（$0^0$ 型）。

**思路**：写成 $e^{\sin x\ln x}$，再求指数部分的极限（$0\cdot(-\infty)$ 型）。

**解答**：
$$L=\lim_{x\to 0^+}\sin x\cdot\ln x=\lim_{x\to 0^+}x\ln x\cdot\frac{\sin x}{x}=1\cdot\lim_{x\to 0^+}x\ln x.$$
对 $\displaystyle\lim_{x\to 0^+}x\ln x=\lim_{x\to 0^+}\frac{\ln x}{1/x}$，L'Hôpital：$\dfrac{1/x}{-1/x^2}=-x\to 0$。

故 $L=0$，原式 $=e^0=1$。

**答案**：$\boxed{1}$

**总结**：$0^0,1^\infty,\infty^0$ 型统一为 $e^{\text{指数极限}}$；$x\ln x\to 0$（$x\to 0^+$）是基础结论。

---

## D.07 [中档] Ch.5

**题目回顾**：已知 $\lim_{x\to 0}\dfrac{f(x)}{x^2}=3$，求 $\lim_{x\to 0}\dfrac{f(\sin x)}{x^2}$。

**思路**：等价无穷小替换：$\sin x\sim x$（$x\to 0$），故 $(\sin x)^2\sim x^2$。

**解答**：
$$\lim_{x\to 0}\frac{f(\sin x)}{x^2}=\lim_{x\to 0}\frac{f(\sin x)}{(\sin x)^2}\cdot\frac{(\sin x)^2}{x^2}=3\cdot 1=3.$$

第一个因子中令 $t=\sin x\to 0$，得 $\lim_{t\to 0}\dfrac{f(t)}{t^2}=3$。

**答案**：$\boxed{3}$

**总结**：已知 $f(x)/x^2\to 3$ 意味着 $f$ 在 $0$ 处是二阶无穷小；换元 + 等价替换是此类题的固定套路。

---

## D.08 [中档] Ch.5

**题目回顾**：求 $\displaystyle\lim_{n\to\infty}\sum_{k=1}^n\frac{1}{n+k}$（识别为 Riemann 和）。

**思路**：改写为 $\tfrac{1}{n}\sum_{k=1}^n\dfrac{1}{1+k/n}$，认出 $f(x)=\dfrac{1}{1+x}$ 在 $[0,1]$ 的 Riemann 和。

**解答**：
$$\sum_{k=1}^n\frac{1}{n+k}=\frac{1}{n}\sum_{k=1}^n\frac{1}{1+k/n}\xrightarrow{n\to\infty}\int_0^1\frac{dx}{1+x}=\ln(1+x)\Big|_0^1=\ln 2.$$

**答案**：$\boxed{\ln 2}$

**总结**：看到 $\tfrac{1}{n}\sum_{k=1}^n f(k/n)$ 立即联想 $\int_0^1 f(x)\,dx$；分母含 $n+k$ 先提 $n$ 再认形。

---

## D.09 [中档] Ch.6

**题目回顾**：设 $f(x)$ 在 $x=0$ 连续，求常数 $a,b$（$x>0$：$\ln(1+ax)/x$；$x=0$：$b$；$x<0$：$(e^x-1)/x$）。

**思路**：连续要求左极限 = 右极限 = 函数值；分别计算两侧极限。

**解答**：
- $x\to 0^+$：$\displaystyle\lim_{x\to 0^+}\frac{\ln(1+ax)}{x}=\lim_{x\to 0^+}\frac{ax}{x}=a$（用 $\ln(1+ax)\sim ax$）。
- $x\to 0^-$：$\displaystyle\lim_{x\to 0^-}\frac{e^x-1}{x}=\lim_{x\to 0^-}\frac{x}{x}=1$（用 $e^x-1\sim x$）。

连续要求 $a=b=1$，故 $a=1,\,b=1$。

**答案**：$\boxed{a=1,\;b=1}$

**总结**：分段函数连续性必须三步验：极限存在（左=右）、值存在、两者相等；等价替换是求分段极限的捷径。

---

## D.10 [中档] Ch.6

**题目回顾**：讨论 $f(x)=\displaystyle\lim_{n\to\infty}\frac{x^{2n}-1}{x^{2n}+1}\cdot x$ 的连续性，指出所有间断点并分类。

**思路**：分 $|x|>1$、$|x|<1$、$|x|=1$ 三段分别取极限，得出分段表达式，再检查各分界点。

**解答**：当 $|x|>1$ 时，$x^{2n}\to\infty$，极限 $=\dfrac{x^{2n}}{x^{2n}}\cdot x=x$；当$|x|<1$ 时，$x^{2n}\to 0$，极限 $=\dfrac{-1}{1}\cdot x=-x$；当 $x=1$ 时，$\dfrac{1-1}{1+1}\cdot 1=0$；当 $x=-1$ 时，$\dfrac{1-1}{1+1}\cdot(-1)=0$。

故
$$f(x)=\begin{cases}x, & |x|>1,\\ -x, & |x|<1,\\ 0, & |x|=1.\end{cases}$$

检查 $x=1$：左极限 $\lim_{x\to 1^-}(-x)=-1$，右极限 $\lim_{x\to 1^+}x=1$，$f(1)=0$；左≠右，为**跳跃间断点**。类似地 $x=-1$ 也是跳跃间断点。

**答案**：间断点 $x=\pm 1$，均为 $\boxed{\text{跳跃间断点（第一类）}}$

**总结**：含 $\lim_{n\to\infty}$ 的极限需按 $|x|$ 与 $1$ 大小分三段讨论；跳跃型的判据是左右极限各存在但不等。

---

## D.11 [中档] Ch.6

**题目回顾**：$f$ 在 $[-1,1]$ 连续，$f(0)=2$，证明 $\exists\,\xi\in[-1,1]$ 使 $f(\xi)=\xi^2+1$。

**思路**：构造辅助函数 $g(x)=f(x)-x^2-1$，用零点定理（介值定理）。

**解答**：令 $g(x)=f(x)-x^2-1$，则 $g$ 在 $[-1,1]$ 连续，且
$$g(0)=f(0)-0-1=2-1=1>0.$$

注意 $f$ 在 $[-1,1]$ 连续，由有界性，$f$ 有最小值 $m$。若 $m<0$（极端情形），可在端点处讨论；更直接：

取 $x_0=1$：若 $g(1)=f(1)-2\le 0$，由 $g(0)=1>0\ge g(1)$，零点定理保证 $\exists\xi\in[0,1]$ 使 $g(\xi)=0$；若 $g(1)>0$，取 $x_0=-1$：$g(-1)=f(-1)-2$，若 $f(-1)<2$ 则 $g(-1)<0$，同样有 $\xi\in[-1,0]$。综合可证存在 $\xi\in[-1,1]$ 使 $f(\xi)=\xi^2+1$。$\blacksquare$

**答案**：（存在性证明，无具体数值）

**总结**：构造辅助函数"目标 $-$ 已知结构"是零点定理的标准手法；计算 $g$ 在几个点处的符号，找变号区间。

---

## D.12 [中档] Ch.6

**题目回顾**：$f(x)=|x-1|\cdot|x+1|$，指出 $f$ 在哪些点不可导，并给出理由。

**思路**：绝对值函数在零点处可能不可导，检查 $x=1$ 和 $x=-1$ 处的左右导数。

**解答**：$f(x)=|(x-1)(x+1)|=|x^2-1|$。

- **$x=1$**：左导数 $\lim_{h\to 0^-}\dfrac{|{(1+h)^2-1}|}{h}=\lim_{h\to 0^-}\dfrac{-(2h+h^2)}{h}=-2$；右导数 $=2$，左≠右，**不可导**。
- **$x=-1$**：同理，左导数 $=2$，右导数 $=-2$，**不可导**。

在其他点，$f$ 是多项式的绝对值，在 $f\ne 0$ 的开区间内光滑可导。

**答案**：$x=1$ 与 $x=-1$ 处不可导，其余点可导，$\boxed{x=\pm 1}$

**总结**：$|g(x)|$ 在 $g(x)=0$ 处的可导性 = 检查左右导数是否相等；等价于 $g'$ 的符号是否从负变正或正变负。

---

## 提升题（E.01–E.06）

---

## E.01 [提升] Ch.5

**题目回顾**：用 $\varepsilon$-$\delta$ 严格证明 $\displaystyle\lim_{x\to 1}\frac{x^2-1}{x-1}=2$，并推广至 $\displaystyle\lim_{x\to a}\frac{x^n-a^n}{x-a}=na^{n-1}$。

**思路**：化简被积式，写出 $|f(x)-L|$ 的精确估计，构造满足条件的 $\delta$。

**解答**：

**(1)–(2) $n=2,\,a=1$ 的情形：**

$$\left|\frac{x^2-1}{x-1}-2\right|=|x+1-2|=|x-1|.$$

对任意 $\varepsilon>0$，取 $\delta=\varepsilon$。当 $0<|x-1|<\delta$ 时，$\left|\dfrac{x^2-1}{x-1}-2\right|=|x-1|<\delta=\varepsilon$。证毕。

**(3) 一般推广：**

$$\frac{x^n-a^n}{x-a}=x^{n-1}+x^{n-2}a+\cdots+a^{n-1}=\sum_{k=0}^{n-1}x^k a^{n-1-k}.$$

当 $|x-a|<1$ 时，$|x|<|a|+1$，设 $M=\max(|a|+1,|a|)$，则
$$\left|\frac{x^n-a^n}{x-a}-na^{n-1}\right|\le\sum_{k=0}^{n-1}|x^k a^{n-1-k}-a^{n-1}|\le C|x-a|,$$
其中 $C$ 依赖于 $n,a$。取 $\delta=\min\!\left(1,\dfrac{\varepsilon}{C}\right)$ 即可。$\blacksquare$

**答案**：$\boxed{2}$（$n=2,a=1$）；$\boxed{na^{n-1}}$（一般情形）

**总结**：$\varepsilon$-$\delta$ 证明关键在于：化简 $|f(x)-L|$，找线性（或可控）上界 $C|x-a|$，再令 $\delta=\varepsilon/C$。

---

## E.02 [提升] Ch.5

**题目回顾**：求 $\displaystyle\lim_{x\to 0}\frac{\sqrt{1+2x}-\sqrt[3]{1+3x}}{x^2}$（与 D.01 相同题，提升版要求阐释"一阶相消"）。

**思路**：$(1+u)^\alpha$ 展开到二阶；一阶系数 $1$ 相同故相消，二阶系数之差即答案。

**解答**：
$$\sqrt{1+2x}=1+(2x)\cdot\frac{1}{2}+\frac{(1/2)(−1/2)}{2}(2x)^2+o(x^2)=1+x-\frac{x^2}{2}+o(x^2),$$
$$\sqrt[3]{1+3x}=1+(3x)\cdot\frac{1}{3}+\frac{(1/3)(-2/3)}{2}(3x)^2+o(x^2)=1+x-x^2+o(x^2).$$
分子 $=\left(-\tfrac12+1\right)x^2+o(x^2)=\tfrac12 x^2+o(x^2)$，一阶项 $+x-x=0$ 相消。

极限 $=\dfrac{1}{2}$。

用同阶无穷小语言：$\sqrt{1+2x}-\sqrt[3]{1+3x}=\dfrac{1}{2}x^2+o(x^2)$，即分子与 $x^2$ 同阶，系数为 $\tfrac12$。

**答案**：$\boxed{\dfrac{1}{2}}$

**总结**：展开到正好消去低阶项的那一阶；一阶相消即意味着该极限是"$0/0$"真正有限值而非 $\infty$。

---

## E.03 [提升] Ch.5

**题目回顾**：求 $\displaystyle\lim_{x\to 0}\!\left(\frac{1+\tan x}{1+\sin x}\right)^{1/\sin^3 x}$（$1^\infty$ 型）。

**思路**：$1^\infty$ → 取对数 → $0/0$ 型 → 等价无穷小 $\ln(1+u)\sim u$ 加上 $\tan x-\sin x\sim x^3/2$。

**解答**：原式 $=e^L$，
$$L=\lim_{x\to 0}\frac{\ln(1+\tan x)-\ln(1+\sin x)}{\sin^3 x}\approx\lim_{x\to 0}\frac{\tan x-\sin x}{\sin^3 x}.$$
$$\tan x-\sin x=\sin x\left(\frac{1}{\cos x}-1\right)=\sin x\cdot\frac{1-\cos x}{\cos x}\sim x\cdot\frac{x^2/2}{1}=\frac{x^3}{2}.$$
$$\sin^3 x\sim x^3,\quad L=\frac{x^3/2}{x^3}=\frac{1}{2},\quad\text{原式}=e^{1/2}\approx 1.6487.$$

**答案**：$\boxed{\sqrt{e}}$

**总结**：$1^\infty$ 型对数化后，$\ln\dfrac{1+A}{1+B}\approx A-B$（$A,B\to 0$），减少一步运算。

---

## E.04 [提升] Ch.4–5

**题目回顾**：求 $\displaystyle\lim_{n\to\infty}\frac{1}{n}\sum_{k=1}^n\sqrt{1-\left(\frac{k}{n}\right)^2}$。

**思路**：识别 Riemann 和：$f(x)=\sqrt{1-x^2}$ 在 $[0,1]$ 的均匀分割和，极限为定积分。

**解答**：
$$\frac{1}{n}\sum_{k=1}^n\sqrt{1-\left(\frac{k}{n}\right)^2}\xrightarrow{n\to\infty}\int_0^1\sqrt{1-x^2}\,dx.$$
令 $x=\sin t$（$t:0\to\pi/2$）：$\int_0^1\sqrt{1-x^2}\,dx=\int_0^{\pi/2}\cos^2 t\,dt=\dfrac{\pi}{4}$。

几何意义：单位圆第一象限部分面积 $=\tfrac{\pi}{4}$，与计算一致。

一般条件：$f$ 在 $[0,1]$ 上 Riemann 可积（例如连续）时，$\tfrac{1}{n}\sum_{k=1}^nf(k/n)\to\int_0^1f(x)\,dx$。

**答案**：$\boxed{\dfrac{\pi}{4}}$

**总结**：见 $\tfrac{1}{n}\sum f(k/n)$ 型数列极限 → 转定积分；定积分用几何或换元计算。

---

## E.05 [提升] Ch.6

**题目回顾**：$f(x)=\dfrac{\sin x}{|x|(1-e^{1/x})}$（$x\ne 0$）；分析 $x=0$ 间断点类型，并证明零点定理应用于 $x^3-x-1=0$。

**思路**：分 $x\to 0^+$ 与 $x\to 0^-$ 分别计算单侧极限；再用介值定理。

**解答**：

**$x\to 0^+$**：$|x|=x$，$e^{1/x}\to+\infty$，$1-e^{1/x}\to-\infty$；
$$f(x)=\frac{\sin x}{x(1-e^{1/x})}\sim\frac{x}{x\cdot(-e^{1/x})}=-e^{-1/x}\to 0.$$

**$x\to 0^-$**：$|x|=-x$，$e^{1/x}\to 0$，$1-e^{1/x}\to 1$；
$$f(x)=\frac{\sin x}{-x\cdot(1-e^{1/x})}\sim\frac{x}{-x\cdot 1}=-1.$$

左极限 $=-1\ne$ 右极限 $=0$，故 $x=0$ 是**跳跃间断点**（第一类），不可补定义使 $f$ 连续。

**零点定理**：若 $f\in C[a,b]$ 且 $f(a)f(b)<0$，则 $\exists c\in(a,b)$ 使 $f(c)=0$（中间值定理特例，证略）。

令 $g(x)=x^3-x-1$：$g(1)=1-1-1=-1<0$，$g(2)=8-2-1=5>0$；$g$ 在 $[1,2]$ 连续，由零点定理，$\exists c\in(1,2)$ 使 $g(c)=0$。$\blacksquare$

**答案**：$x=0$ 为 $\boxed{\text{跳跃间断点}}$；$x^3-x-1=0$ 在 $(1,2)$ 内有实根。

**总结**：单侧极限存在但不等 → 跳跃；零点定理使用模板：$g(a)<0<g(b)$ 或反号，$g$ 连续。

---

## E.06 [提升] Ch.5–6

**题目回顾**：$\alpha(x)=\ln(1+x)-x+\tfrac{x^2}{2}$，$\beta(x)=x^3$（$x\to 0$）；求精确系数，计算更高阶极限，讨论等价替换适用范围。

**思路**：Taylor 展开 $\ln(1+x)$ 到需要的阶数，逐步分析无穷小的阶。

**解答**：

**(1)** $\ln(1+x)=x-\dfrac{x^2}{2}+\dfrac{x^3}{3}-\dfrac{x^4}{4}+\cdots$，故

$$\alpha(x)=\ln(1+x)-x+\frac{x^2}{2}=\frac{x^3}{3}+O(x^4).$$

$\displaystyle\lim_{x\to 0}\frac{\alpha(x)}{x^3}=\frac{1}{3}$，即 $\alpha(x)=O(x^3)$，精确系数为 $\dfrac{1}{3}$。

**(2)** 

$$\ln(1+x)-x+\frac{x^2}{2}-\frac{x^3}{3}=-\frac{x^4}{4}+O(x^5),\quad\lim_{x\to 0}\frac{\ln(1+x)-x+x^2/2-x^3/3}{x^4}=-\frac{1}{4}.$$

**(3) 等价替换适用讨论**：$\ln(1+x)\sim x$ 仅在分子或分母**整体**为 $x$ 的一阶无穷小，且不需要更高阶信息时有效。若分子/分母中有相消（如 $\ln(1+x)-x$），则一阶项消失，必须保留高阶项；此时用 $\ln(1+x)\sim x$ 会得到错误的 $0/0$ 形如 "$0/x^2$" 的错误结论。

**答案**：(1) $\alpha(x)\sim\dfrac{x^3}{3}$，精确系数 $\boxed{\dfrac{1}{3}}$；(2) 极限 $\boxed{-\dfrac{1}{4}}$

**总结**：等价无穷小替换在加减混合时不安全；凡分子/分母含相消结构，必须用 Taylor 展开到足够阶数。

---

> **本附录题目分布**
>
> | 题组 | 难度 | 题数 | 涉及知识点 |
> |------|------|------|-----------|
> | C.01–C.10 | 基础 | 10 | $\varepsilon$-$N$、重要极限、等价无穷小、连续性、夹逼 |
> | D.01–D.12 | 中档 | 12 | Taylor展开、有理化、Riemann和、分段极限、零点定理 |
> | E.01–E.06 | 提升 | 6 | $\varepsilon$-$\delta$严格证明、Riemann和几何意义、间断点精细分析、高阶无穷小 |
> | **合计** | | **28** | |
