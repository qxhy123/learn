# 附录 F2b：级数详解（C.41-C.50, D.49-D.60, E.29-E.36）

> 本附录对应 Ch.15–17（数项级数、幂级数、Fourier 级数），共 **30 题**详解。  
> 题目来源：C 组（基础）10 题 + D 组（中档）12 题 + E 组（提升）8 题。  
> 格式：**题目回顾 → 思路 → 解答 → 答案 → 总结**。

---

## 第一部分：C 组基础题（C.41–C.50）

---

### C.41 比值法判断 $\sum 1/n!$ 的收敛性

**题目回顾**  
用比值法判断 $\displaystyle\sum_{n=1}^\infty \frac{1}{n!}$ 的收敛性。

**思路**  
比值判别法（d'Alembert）：若 $\lim_{n\to\infty} a_{n+1}/a_n = l < 1$，则级数收敛。取 $a_n = 1/n!$，计算相邻项之比，阶乘使分子远小于分母。

**解答**  
令 $a_n = \dfrac{1}{n!}$，则

$$\frac{a_{n+1}}{a_n} = \frac{1/(n+1)!}{1/n!} = \frac{n!}{(n+1)!} = \frac{1}{n+1}.$$

因此

$$\lim_{n\to\infty} \frac{a_{n+1}}{a_n} = \lim_{n\to\infty} \frac{1}{n+1} = 0 < 1.$$

由比值判别法，级数**收敛**。

**答案**  
级数 $\displaystyle\sum_{n=1}^\infty \frac{1}{n!}$ 收敛（其和为 $e - 1$）。

**总结**  
含阶乘的级数首选比值法；$n!$ 增长远快于指数函数，使得比值趋于 $0$，是判敛最有力的情形之一。

---

### C.42 $p$-级数 $\sum 1/n^2$ 的收敛性

**题目回顾**  
判断 $p$-级数 $\displaystyle\sum_{n=1}^\infty \frac{1}{n^2}$ 是否收敛。

**思路**  
直接套用 $p$-级数判别定理：$\sum n^{-p}$ 当 $p > 1$ 时收敛，当 $p \le 1$ 时发散。

**解答**  
这是 $p = 2 > 1$ 的 $p$-级数。由 $p$-级数判别定理，级数**收敛**。

也可用积分判别法验证：

$$\int_1^{+\infty} \frac{dx}{x^2} = \left[-\frac{1}{x}\right]_1^{+\infty} = 0 - (-1) = 1 < +\infty,$$

积分收敛，故级数收敛（其和为著名的 $\pi^2/6$，即 Basel 问题）。

**答案**  
$\displaystyle\sum_{n=1}^\infty \frac{1}{n^2}$ 收敛，和为 $\dfrac{\pi^2}{6}$。

**总结**  
$p$-级数是判别其他级数收敛性的标准参照。记住临界值 $p = 1$：$p > 1$ 收敛，$p = 1$（调和级数）发散。

---

### C.43 调和级数 $\sum 1/n$ 的发散性（积分判别）

**题目回顾**  
判断调和级数 $\displaystyle\sum_{n=1}^\infty \frac{1}{n}$ 是否收敛（用积分判别法）。

**思路**  
积分判别法：若 $f(x) = 1/x$ 在 $[1,+\infty)$ 上单调递减且非负，则 $\sum f(n)$ 与 $\int_1^{+\infty} f(x)\,dx$ 同敛散。

**解答**  
$f(x) = 1/x$ 在 $[1,+\infty)$ 上连续、正值、单调递减，可使用积分判别法：

$$\int_1^{+\infty} \frac{dx}{x} = \ln x \Big|_1^{+\infty} = +\infty.$$

广义积分发散，因此 $\displaystyle\sum_{n=1}^\infty \frac{1}{n}$ **发散**。

**答案**  
调和级数 $\displaystyle\sum_{n=1}^\infty \frac{1}{n}$ 发散（$p = 1$ 的临界情形）。

**总结**  
调和级数是经典反例：各项趋于零，但级数发散。这说明"通项趋零"只是收敛的必要条件，绝非充分条件。

---

### C.44 几何级数 $\sum (1/3)^n$ 的和

**题目回顾**  
求几何级数 $\displaystyle\sum_{n=0}^\infty \left(\frac{1}{3}\right)^n$ 的和。

**思路**  
等比级数公式：$\displaystyle\sum_{n=0}^\infty r^n = \dfrac{1}{1-r}$（当 $|r| < 1$）。直接代入 $r = 1/3$。

**解答**  
公比 $r = 1/3$，$|r| = 1/3 < 1$，几何级数收敛，且

$$\sum_{n=0}^\infty \left(\frac{1}{3}\right)^n = \frac{1}{1 - 1/3} = \frac{1}{2/3} = \frac{3}{2}.$$

**答案**  
$\displaystyle\sum_{n=0}^\infty \left(\frac{1}{3}\right)^n = \dfrac{3}{2}$。

**总结**  
几何级数是少数可精确求和的级数之一。记住公式 $1/(1-r)$ 及适用范围 $|r|<1$，它也是幂级数展开的基础。

---

### C.45 交错级数 $\sum (-1)^{n-1}/n$ 的收敛性（Leibniz 判别）

**题目回顾**  
用 Leibniz 判别法判断交错级数 $\displaystyle\sum_{n=1}^\infty \frac{(-1)^{n-1}}{n}$ 的收敛性。

**思路**  
Leibniz 判别法：若交错级数 $\sum (-1)^{n-1} b_n$ 中，$b_n > 0$ 单调递减趋零，则级数收敛。

**解答**  
令 $b_n = 1/n > 0$。验证两个条件：

1. **单调递减**：$b_{n+1} = \dfrac{1}{n+1} < \dfrac{1}{n} = b_n$，成立。  
2. **趋于零**：$\lim_{n\to\infty} b_n = \lim_{n\to\infty} \dfrac{1}{n} = 0$，成立。

由 Leibniz 判别法，级数 $\displaystyle\sum_{n=1}^\infty \frac{(-1)^{n-1}}{n}$ **收敛**。

注意：由于 $\sum 1/n$ 发散，该级数只是条件收敛，不是绝对收敛。其和为 $\ln 2$。

**答案**  
交错调和级数 $\displaystyle\sum_{n=1}^\infty \frac{(-1)^{n-1}}{n}$ 条件收敛，和为 $\ln 2$。

**总结**  
Leibniz 判别法是判断交错级数收敛性的标准工具。注意还需区分**绝对收敛**与**条件收敛**。

---

### C.46 幂级数 $\sum x^n/2^n$ 的收敛半径与收敛域

**题目回顾**  
求幂级数 $\displaystyle\sum_{n=0}^\infty \frac{x^n}{2^n}$ 的收敛半径与收敛域。

**思路**  
这是公比为 $x/2$ 的几何级数，也可用比值法求收敛半径 $R = 1/\limsup |a_n|^{1/n}$，然后逐一验证端点。

**解答**  
**收敛半径**：令 $a_n = 1/2^n$，由比值法

$$R = \lim_{n\to\infty} \frac{a_n}{a_{n+1}} = \lim_{n\to\infty} \frac{1/2^n}{1/2^{n+1}} = \lim_{n\to\infty} 2 = 2.$$

即 $R = 2$，在 $|x| < 2$ 内绝对收敛。

**端点验证**：

- $x = 2$：$\displaystyle\sum_{n=0}^\infty \frac{2^n}{2^n} = \sum_{n=0}^\infty 1$，发散。  
- $x = -2$：$\displaystyle\sum_{n=0}^\infty \frac{(-2)^n}{2^n} = \sum_{n=0}^\infty (-1)^n$，振荡，发散。

**收敛域**为 $(-2, 2)$，和函数为 $\dfrac{1}{1 - x/2} = \dfrac{2}{2-x}$（$|x| < 2$）。

**答案**  
收敛半径 $R = 2$，收敛域 $(-2, 2)$。

**总结**  
幂级数端点处必须单独验证：即使内部绝对收敛，端点处级数的敛散性需额外判断（可能收敛、可能发散）。

---

### C.47 幂级数 $\sum x^n/n$ 的收敛半径

**题目回顾**  
求幂级数 $\displaystyle\sum_{n=1}^\infty \frac{x^n}{n}$ 的收敛半径（用比值法）。

**思路**  
对幂级数 $\sum a_n x^n$，收敛半径 $R = \lim_{n\to\infty} |a_n/a_{n+1}|$（比值法形式）。

**解答**  
令 $a_n = 1/n$，则

$$R = \lim_{n\to\infty} \frac{a_n}{a_{n+1}} = \lim_{n\to\infty} \frac{1/n}{1/(n+1)} = \lim_{n\to\infty} \frac{n+1}{n} = 1.$$

故收敛半径 $R = 1$。（端点讨论详见 D.55。）

**答案**  
$R = 1$。

**总结**  
比值法求收敛半径时，计算的是系数比 $|a_n/a_{n+1}|$ 的极限，而非通项比——注意区别于数项级数中的比值法。

---

### C.48 $\dfrac{1}{1+x}$ 的幂级数展开

**题目回顾**  
利用 $\dfrac{1}{1-x} = \displaystyle\sum_{n=0}^\infty x^n$ 写出 $\dfrac{1}{1+x}$ 的幂级数展开（$|x|<1$）。

**思路**  
将已知展开式中的 $x$ 替换为 $-x$，即得所求展开。

**解答**  
由 $\dfrac{1}{1-t} = \displaystyle\sum_{n=0}^\infty t^n$（$|t|<1$），令 $t = -x$：

$$\frac{1}{1-(-x)} = \frac{1}{1+x} = \sum_{n=0}^\infty (-x)^n = \sum_{n=0}^\infty (-1)^n x^n.$$

即

$$\frac{1}{1+x} = 1 - x + x^2 - x^3 + \cdots = \sum_{n=0}^\infty (-1)^n x^n, \quad |x| < 1.$$

**答案**  
$\dfrac{1}{1+x} = \displaystyle\sum_{n=0}^\infty (-1)^n x^n$（$|x| < 1$）。

**总结**  
"以 $-x$ 代替 $x$"是推导幂级数展开的基本操作，由此出发还可逐项积分得到 $\ln(1+x)$ 的展开式。

---

### C.49 $f(x) = x$ 在 $(-\pi, \pi)$ 的 Fourier 系数

**题目回顾**  
写出 $f(x) = x$（$-\pi < x < \pi$）的 Fourier 级数系数 $a_0, a_n, b_n$。

**思路**  
$f(x) = x$ 是奇函数。Fourier 系数公式：$a_n = \frac{1}{\pi}\int_{-\pi}^\pi f(x)\cos nx\,dx$，$b_n = \frac{1}{\pi}\int_{-\pi}^\pi f(x)\sin nx\,dx$。奇函数与偶函数（$\cos$）之积为奇函数，在对称区间上积分为零。

**解答**  
由于 $f(x) = x$ 是奇函数：

- **$a_0$**：$a_0 = \dfrac{1}{\pi}\displaystyle\int_{-\pi}^\pi x\,dx = 0$（奇函数在对称区间上积分为零）。

- **$a_n$**（$n \ge 1$）：$x \cos nx$ 是奇函数，故 $a_n = 0$。

- **$b_n$**（$n \ge 1$）：$x \sin nx$ 是偶函数，故

$$b_n = \frac{1}{\pi}\int_{-\pi}^\pi x\sin nx\,dx = \frac{2}{\pi}\int_0^\pi x\sin nx\,dx.$$

分部积分（$u = x$，$dv = \sin nx\,dx$）：

$$\int_0^\pi x\sin nx\,dx = \left[-\frac{x\cos nx}{n}\right]_0^\pi + \int_0^\pi \frac{\cos nx}{n}\,dx = -\frac{\pi\cos n\pi}{n} + \frac{\sin nx}{n^2}\Big|_0^\pi = \frac{(-1)^{n+1}\pi}{n}.$$

因此 $b_n = \dfrac{2}{\pi} \cdot \dfrac{(-1)^{n+1}\pi}{n} = \dfrac{2(-1)^{n+1}}{n}$。

**答案**  
$a_0 = 0$，$a_n = 0$，$b_n = \dfrac{2(-1)^{n+1}}{n}$。  
Fourier 级数：$x = \displaystyle\sum_{n=1}^\infty \frac{2(-1)^{n+1}}{n}\sin nx$（$x \in (-\pi, \pi)$）。

**总结**  
奇函数的 Fourier 级数只含正弦项（$a_n = 0$）；偶函数只含余弦项（$b_n = 0$）。利用对称性可大幅简化计算。

---

### C.50 由 $\sum 1/n^2 = \pi^2/6$ 推导 $\sum (-1)^{n-1}/n^2$

**题目回顾**  
利用 Fourier 级数的结论 $\displaystyle\sum_{n=1}^\infty\frac{1}{n^2}=\frac{\pi^2}{6}$，写出 $\displaystyle\sum_{n=1}^\infty\frac{(-1)^{n-1}}{n^2}$ 的值。

**思路**  
将 $\sum 1/n^2$ 按奇偶项拆分，利用奇数项之和与偶数项之和之间的关系。

**解答**  
将正项级数按奇偶分组：

$$\sum_{n=1}^\infty \frac{1}{n^2} = \underbrace{\sum_{k=1}^\infty \frac{1}{(2k-1)^2}}_{S_{\text{奇}}} + \underbrace{\sum_{k=1}^\infty \frac{1}{(2k)^2}}_{S_{\text{偶}}}.$$

注意 $S_{\text{偶}} = \displaystyle\sum_{k=1}^\infty \frac{1}{4k^2} = \frac{1}{4}\sum_{k=1}^\infty \frac{1}{k^2} = \frac{\pi^2}{24}$，

故 $S_{\text{奇}} = \dfrac{\pi^2}{6} - \dfrac{\pi^2}{24} = \dfrac{3\pi^2}{24} = \dfrac{\pi^2}{8}$。

现在计算目标级数：

$$\sum_{n=1}^\infty \frac{(-1)^{n-1}}{n^2} = \frac{1}{1^2} - \frac{1}{2^2} + \frac{1}{3^2} - \cdots = S_{\text{奇}} - S_{\text{偶}} = \frac{\pi^2}{8} - \frac{\pi^2}{24} = \frac{3\pi^2}{24} - \frac{\pi^2}{24} = \frac{\pi^2}{12}.$$

**答案**  
$\displaystyle\sum_{n=1}^\infty\frac{(-1)^{n-1}}{n^2} = \dfrac{\pi^2}{12}$。

**总结**  
"奇偶分拆"是由已知常数级数推导相关交错级数的标准方法。也可利用 $f(x) = x^2$ 的 Fourier 级数在 $x = \pi$ 处代值直接得到。

---

## 第二部分：D 组中档题（D.49–D.60）

---

### D.49 判断 $\sum n!/n^n$ 的收敛性

**题目回顾**  
判断 $\displaystyle\sum_{n=1}^\infty\frac{n!}{n^n}$ 的收敛性（比值法）。

**思路**  
含 $n!$ 和 $n^n$ 的级数，优先使用比值法。关键极限 $(1+1/n)^n \to e$。

**解答**  
令 $a_n = \dfrac{n!}{n^n}$，则

$$\frac{a_{n+1}}{a_n} = \frac{(n+1)!}{(n+1)^{n+1}} \cdot \frac{n^n}{n!} = \frac{(n+1)\cdot n!}{(n+1)^{n+1}} \cdot \frac{n^n}{n!} = \frac{n^n}{(n+1)^n} = \left(\frac{n}{n+1}\right)^n = \frac{1}{\left(1+\frac{1}{n}\right)^n}.$$

由经典极限 $\displaystyle\lim_{n\to\infty}\left(1+\frac{1}{n}\right)^n = e$，得

$$\lim_{n\to\infty}\frac{a_{n+1}}{a_n} = \frac{1}{e} < 1.$$

由比值判别法，级数**收敛**。

**答案**  
$\displaystyle\sum_{n=1}^\infty\frac{n!}{n^n}$ 收敛，比值极限为 $1/e \approx 0.368$。

**总结**  
极限 $(1+1/n)^n \to e$ 在比值法中频繁出现，是熟练解题的关键。含 $n^n$ 的级数往往比含 $n!$ 的级数"更小"，因为 $n^n = n \cdot n \cdots n$ 远大于 $n! = 1 \cdot 2 \cdots n$（斯特林公式：$n! \sim \sqrt{2\pi n}(n/e)^n$）。

---

### D.50 判断 $\sum 1/(n\ln n)$ 的收敛性（积分判别）

**题目回顾**  
判断 $\displaystyle\sum_{n=2}^\infty\frac{1}{n\ln n}$ 的收敛性（积分判别法）。

**思路**  
$f(x) = 1/(x\ln x)$ 在 $[2,+\infty)$ 上单调递减且正值，可用积分判别法。积分可以计算。

**解答**  
令 $f(x) = \dfrac{1}{x\ln x}$（$x \ge 2$），$f$ 连续、正值、单调递减。

对 $\displaystyle\int_2^{+\infty}\frac{dx}{x\ln x}$，令 $u = \ln x$，$du = dx/x$：

$$\int_2^{+\infty}\frac{dx}{x\ln x} = \int_{\ln 2}^{+\infty}\frac{du}{u} = \ln u\Big|_{\ln 2}^{+\infty} = +\infty.$$

积分发散，由积分判别法，级数 $\displaystyle\sum_{n=2}^\infty\frac{1}{n\ln n}$ **发散**。

**答案**  
$\displaystyle\sum_{n=2}^\infty\frac{1}{n\ln n}$ 发散。

**总结**  
调和级数 $\sum 1/n$ 发散，$\sum 1/(n\ln n)$ 也发散，而 $\sum 1/(n\ln^2 n)$ 收敛。对数因子的幂次是调和级数的精细分界。

---

### D.51 求 $\sum n^2/2^n$ 的和

**题目回顾**  
求 $\displaystyle\sum_{n=1}^\infty\frac{n^2}{2^n}$ 的和。

**思路**  
利用幂级数逐项求导技巧。从 $\displaystyle\sum x^n = \frac{x}{1-x}$（$|x|<1$，从 $n=1$ 开始）出发，两次求导后代 $x=1/2$。

**解答**  
**第一步**：由 $\displaystyle\sum_{n=0}^\infty x^n = \frac{1}{1-x}$（$|x|<1$），对 $x$ 求导：

$$\sum_{n=1}^\infty n x^{n-1} = \frac{1}{(1-x)^2}.$$

乘以 $x$：$\displaystyle\sum_{n=1}^\infty n x^n = \frac{x}{(1-x)^2}$。

**第二步**：对上式再求导：

$$\sum_{n=1}^\infty n^2 x^{n-1} = \frac{d}{dx}\left[\frac{x}{(1-x)^2}\right] = \frac{(1-x)^2 + 2x(1-x)}{(1-x)^4} = \frac{1+x}{(1-x)^3}.$$

乘以 $x$：$\displaystyle\sum_{n=1}^\infty n^2 x^n = \frac{x(1+x)}{(1-x)^3}$（$|x|<1$）。

**第三步**：代入 $x = 1/2$：

$$\sum_{n=1}^\infty \frac{n^2}{2^n} = \frac{\frac{1}{2}\left(1+\frac{1}{2}\right)}{\left(1-\frac{1}{2}\right)^3} = \frac{\frac{1}{2}\cdot\frac{3}{2}}{\frac{1}{8}} = \frac{3/4}{1/8} = 6.$$

**答案**  
$\displaystyle\sum_{n=1}^\infty\frac{n^2}{2^n} = 6$。

**总结**  
对含多项式系数的级数，"先写幂级数 $\to$ 微分 $\to$ 代值"是标准流程。每次乘 $x$ 再求导，可以升高系数中 $n$ 的次数。

---

### D.52 $\sum (-1)^{n-1} x^n/n$ 在端点处的收敛性

**题目回顾**  
讨论 $\displaystyle\sum_{n=1}^\infty(-1)^{n-1}\frac{x^n}{n}$ 在 $x=1$ 与 $x=-1$ 处的收敛性，并写出收敛域。

**思路**  
先求收敛半径（比值法得 $R=1$），再逐一检验端点 $x=\pm 1$。

**解答**  
**收敛半径**：$R = \lim_{n\to\infty} \dfrac{1/n}{1/(n+1)} = 1$，故 $|x| < 1$ 内绝对收敛。

**端点 $x = 1$**：级数变为 $\displaystyle\sum_{n=1}^\infty\frac{(-1)^{n-1}}{n}$（交错调和级数），由 Leibniz 判别法**收敛**（和为 $\ln 2$）。

**端点 $x = -1$**：级数变为 $\displaystyle\sum_{n=1}^\infty(-1)^{n-1}\frac{(-1)^n}{n} = \sum_{n=1}^\infty\frac{(-1)^{2n-1}}{n} = -\sum_{n=1}^\infty\frac{1}{n}$，即负的调和级数，**发散**。

**答案**  
收敛域为 $(-1, 1]$。

**总结**  
幂级数在收敛圆边界上的行为需要特别处理：可能两端点均收敛、均发散，或一端收敛一端发散。Abel 定理保证端点收敛时和函数的连续性。

---

### D.53 绝对收敛 $\Rightarrow$ 收敛，反命题不成立

**题目回顾**  
证明：若 $\displaystyle\sum a_n$ 绝对收敛，则它也收敛。并举例说明反命题不成立。

**思路**  
利用 $|a_n + a_m| \le |a_n| + |a_m|$（三角不等式）和 Cauchy 收敛准则。

**解答**  
**证明**：设 $\displaystyle\sum |a_n|$ 收敛，即部分和 $S_N^* = \displaystyle\sum_{n=1}^N |a_n|$ 收敛（有界且单调）。

对任意 $\varepsilon > 0$，由 $\sum|a_n|$ 的 Cauchy 条件，存在 $N_0$ 使当 $m > n > N_0$ 时

$$\sum_{k=n+1}^m |a_k| < \varepsilon.$$

于是 $\left|\displaystyle\sum_{k=n+1}^m a_k\right| \le \displaystyle\sum_{k=n+1}^m |a_k| < \varepsilon$，

由 Cauchy 准则，$\displaystyle\sum a_n$ 收敛。$\square$

**反例**：$\displaystyle\sum_{n=1}^\infty\frac{(-1)^{n-1}}{n}$ 条件收敛（Leibniz 判别），但 $\displaystyle\sum\frac{1}{n}$ 发散，故不是绝对收敛。

**答案**  
绝对收敛 $\Rightarrow$ 收敛（已证）；逆命题不成立，反例为 $\displaystyle\sum\frac{(-1)^{n-1}}{n}$。

**总结**  
绝对收敛是比条件收敛更强的条件。绝对收敛级数可以任意重排而不改变其和（Riemann 重排定理的逆否命题）。

---

### D.54 幂级数 $\sum x^n/n!$ 的收敛半径与和函数

**题目回顾**  
求幂级数 $\displaystyle\sum_{n=0}^\infty\frac{x^n}{n!}$ 的收敛半径与和函数。

**思路**  
比值法求收敛半径；和函数从 $e^x$ 的 Taylor 展开识别。

**解答**  
**收敛半径**：令 $a_n = 1/n!$，

$$\lim_{n\to\infty}\frac{a_{n+1}}{a_n} = \lim_{n\to\infty}\frac{1/(n+1)!}{1/n!} = \lim_{n\to\infty}\frac{1}{n+1} = 0.$$

因此 $l = 0 < 1$ 对所有 $x$ 成立，收敛半径 $R = +\infty$（在 $\mathbb{R}$ 上处处收敛）。

**和函数**：$e^x$ 的 Maclaurin 展开正是 $\displaystyle\sum_{n=0}^\infty\frac{x^n}{n!}$，故和函数为 $e^x$。

**答案**  
收敛半径 $R = +\infty$，和函数 $S(x) = e^x$（$x \in \mathbb{R}$）。

**总结**  
指数函数的 Taylor 级数在整个实直线上收敛，这是因为 $n!$ 的增长速度超过任意指数函数。这一展开式是 $e^x$ 解析性的根本体现。

---

### D.55 $\sum x^n/n$ 的收敛域与和函数

**题目回顾**  
求幂级数 $\displaystyle\sum_{n=1}^\infty\frac{x^n}{n}$ 的收敛域与和函数 $S(x)$。

**思路**  
收敛半径 $R = 1$（由 C.47 已得）。对和函数，利用逐项求导后再积分：$S'(x) = \sum x^{n-1} = 1/(1-x)$。

**解答**  
**端点验证**（补充 D.52）：
- $x = 1$：$\sum 1/n$ 发散；$x = -1$：$\sum (-1)^n/n = -\sum (-1)^{n-1}/n$，Leibniz 判别收敛。

故收敛域为 $[-1, 1)$。

**和函数**：在 $|x| < 1$ 内，逐项求导：

$$S'(x) = \sum_{n=1}^\infty x^{n-1} = \frac{1}{1-x}, \quad |x| < 1.$$

又 $S(0) = 0$，对 $S'$ 积分：

$$S(x) = \int_0^x \frac{dt}{1-t} = -\ln(1-x), \quad |x| < 1.$$

当 $x = -1$ 时，由 Abel 定理（级数在 $x=-1$ 处收敛，$S$ 在 $[-1,1)$ 上连续），

$$S(-1) = \lim_{x\to -1^+} S(x) = -\ln(1-(-1)) = -\ln 2.$$

**答案**  
收敛域 $[-1, 1)$，和函数 $S(x) = -\ln(1-x)$（$x \in [-1, 1)$）。

**总结**  
"逐项求导 $\to$ 识别几何级数 $\to$ 积分还原"是求幂级数和函数的核心手法。Abel 定理是将端点收敛性和连续性联系起来的关键。

---

### D.56 $\sum (n+1)x^n$ 的和函数

**题目回顾**  
求幂级数 $\displaystyle\sum_{n=0}^\infty(n+1)x^n$ 的收敛半径，并用逐项求导方法写出其和函数（$|x|<1$）。

**思路**  
注意 $\displaystyle\sum_{n=0}^\infty(n+1)x^n = \frac{d}{dx}\sum_{n=0}^\infty x^{n+1} = \frac{d}{dx}\frac{x}{1-x}$，或者等价地是 $\frac{d}{dx}\frac{1}{1-x}$（从 $n=0$ 的 $\sum x^n$ 出发）。

**解答**  
**收敛半径**：$a_n = n+1$，

$$\frac{a_{n+1}}{a_n} = \frac{n+2}{n+1} \to 1, \quad R = 1.$$

**和函数**：注意 $\displaystyle\sum_{n=0}^\infty x^{n+1} = \frac{x}{1-x}$（$|x|<1$），对 $x$ 求导：

$$\sum_{n=0}^\infty (n+1)x^n = \frac{d}{dx}\left(\frac{x}{1-x}\right) = \frac{(1-x) + x}{(1-x)^2} = \frac{1}{(1-x)^2}.$$

亦可直接对 $\dfrac{1}{1-x} = \displaystyle\sum_{n=0}^\infty x^n$ 求导：

$$\frac{1}{(1-x)^2} = \sum_{n=0}^\infty (n+1)x^n.$$

**答案**  
收敛半径 $R = 1$，和函数 $S(x) = \dfrac{1}{(1-x)^2}$（$|x|<1$）。

**总结**  
$1/(1-x)^2$ 是幂级数理论中出现频率极高的和函数，它是 $1/(1-x)$ 对 $x$ 的导数。这一公式是计算 $\sum n/2^n$、$\sum n^2/2^n$ 等的基础。

---

### D.57 $\ln(1+x)$ 的幂级数与 $\sum (-1)^{n-1}/n = \ln 2$

**题目回顾**  
将 $f(x) = \ln(1+x)$ 在 $x=0$ 展为幂级数，并利用 $x=1$ 处的收敛值推出 $\displaystyle\sum_{n=1}^\infty\frac{(-1)^{n-1}}{n} = \ln 2$。

**思路**  
对 $\dfrac{1}{1+x} = \displaystyle\sum_{n=0}^\infty(-1)^n x^n$ 逐项积分，得 $\ln(1+x)$ 的展开；由 Abel 定理在 $x=1$ 处代值。

**解答**  
由 $\dfrac{1}{1+t} = \displaystyle\sum_{n=0}^\infty(-1)^n t^n$（$|t|<1$），从 $0$ 到 $x$ 积分：

$$\ln(1+x) = \int_0^x \frac{dt}{1+t} = \sum_{n=0}^\infty(-1)^n\int_0^x t^n\,dt = \sum_{n=0}^\infty\frac{(-1)^n x^{n+1}}{n+1} = \sum_{n=1}^\infty\frac{(-1)^{n-1}x^n}{n}.$$

该幂级数的收敛半径为 $1$；在 $x = 1$ 处，级数为 $\displaystyle\sum_{n=1}^\infty\frac{(-1)^{n-1}}{n}$，由 Leibniz 判别法收敛。由 Abel 定理，

$$\sum_{n=1}^\infty\frac{(-1)^{n-1}}{n} = \lim_{x\to 1^-}\ln(1+x) = \ln 2.$$

**答案**  
$\ln(1+x) = \displaystyle\sum_{n=1}^\infty\frac{(-1)^{n-1}x^n}{n}$（$-1 < x \le 1$）；  
特别地，$\displaystyle\sum_{n=1}^\infty\frac{(-1)^{n-1}}{n} = \ln 2$。

**总结**  
逐项积分是从已知的幂级数生成新函数展开式的利器。Abel 定理保证在端点收敛时可以"顺滑地代入"，从而得到精确级数值。

---

### D.58 $\arctan x$ 的 Maclaurin 级数与 Leibniz 公式

**题目回顾**  
求 $f(x) = \arctan x$ 的 Maclaurin 级数，收敛域，并推出 $\pi = 4\displaystyle\sum_{n=0}^\infty\frac{(-1)^n}{2n+1}$。

**思路**  
对 $\dfrac{1}{1+x^2} = \displaystyle\sum_{n=0}^\infty(-1)^n x^{2n}$ 逐项积分，得 $\arctan x$ 展开；在 $x=1$ 代值配合 $\arctan 1 = \pi/4$。

**解答**  
由 $\dfrac{1}{1+t^2} = \displaystyle\sum_{n=0}^\infty(-1)^n t^{2n}$（$|t|<1$），积分：

$$\arctan x = \int_0^x \frac{dt}{1+t^2} = \sum_{n=0}^\infty\frac{(-1)^n x^{2n+1}}{2n+1}.$$

收敛半径为 $1$。端点：$x = \pm 1$ 时级数为 $\displaystyle\sum\frac{(\pm 1)^{2n+1}(-1)^n}{2n+1}$，均由 Leibniz 判别法收敛。故收敛域为 $[-1, 1]$。

代入 $x = 1$，$\arctan 1 = \pi/4$：

$$\frac{\pi}{4} = \sum_{n=0}^\infty\frac{(-1)^n}{2n+1} = 1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \cdots$$

因此 $\pi = 4\displaystyle\sum_{n=0}^\infty\frac{(-1)^n}{2n+1}$（Leibniz 公式）。

**答案**  
$\arctan x = \displaystyle\sum_{n=0}^\infty\frac{(-1)^n x^{2n+1}}{2n+1}$（$|x|\le 1$）；$\pi = 4\displaystyle\sum_{n=0}^\infty\frac{(-1)^n}{2n+1}$。

**总结**  
Leibniz 公式虽然优美，但收敛极慢（误差约 $1/(2n+1)$）。实际计算 $\pi$ 需用更快收敛的展开，如 Machin 公式：$\pi/4 = 4\arctan(1/5) - \arctan(1/239)$。

---

### D.59 $f(x) = |x|$ 的 Fourier 级数与 $\sum 1/(2n+1)^2 = \pi^2/8$

**题目回顾**  
求 $f(x) = |x|$（$-\pi \le x \le \pi$）的 Fourier 级数，并由此推出 $\displaystyle\sum_{n=0}^\infty\frac{1}{(2n+1)^2} = \frac{\pi^2}{8}$。

**思路**  
$|x|$ 是偶函数，故 $b_n = 0$，只需计算 $a_0$ 和 $a_n$。分部积分求 $a_n$，然后代 $x = 0$。

**解答**  
由 $f(x) = |x|$ 为偶函数，$b_n = 0$，且

$$a_0 = \frac{1}{\pi}\int_{-\pi}^\pi |x|\,dx = \frac{2}{\pi}\int_0^\pi x\,dx = \frac{2}{\pi}\cdot\frac{\pi^2}{2} = \pi.$$

对 $n \ge 1$：

$$a_n = \frac{1}{\pi}\int_{-\pi}^\pi |x|\cos nx\,dx = \frac{2}{\pi}\int_0^\pi x\cos nx\,dx.$$

分部积分（$u = x$，$dv = \cos nx\,dx$）：

$$\int_0^\pi x\cos nx\,dx = \frac{x\sin nx}{n}\Big|_0^\pi - \int_0^\pi\frac{\sin nx}{n}\,dx = 0 + \frac{\cos nx}{n^2}\Big|_0^\pi = \frac{\cos n\pi - 1}{n^2} = \frac{(-1)^n - 1}{n^2}.$$

故 $a_n = \dfrac{2}{\pi}\cdot\dfrac{(-1)^n - 1}{n^2} = \begin{cases}0, & n \text{ 偶},\\ -\dfrac{4}{\pi n^2}, & n \text{ 奇}.\end{cases}$

Fourier 级数：

$$|x| = \frac{\pi}{2} - \frac{4}{\pi}\sum_{k=0}^\infty\frac{\cos(2k+1)x}{(2k+1)^2}.$$

代入 $x = 0$（$f(0) = 0$）：

$$0 = \frac{\pi}{2} - \frac{4}{\pi}\sum_{k=0}^\infty\frac{1}{(2k+1)^2} \implies \sum_{k=0}^\infty\frac{1}{(2k+1)^2} = \frac{\pi^2}{8}.$$

**答案**  
$|x| = \dfrac{\pi}{2} - \dfrac{4}{\pi}\displaystyle\sum_{n=0}^\infty\dfrac{\cos(2n+1)x}{(2n+1)^2}$（$x \in [-\pi,\pi]$）；$\displaystyle\sum_{n=0}^\infty\dfrac{1}{(2n+1)^2} = \dfrac{\pi^2}{8}$。

**总结**  
由 Fourier 展开代特殊值是推导数值级数精确和的经典方法。注意代值时需确认级数在该点的收敛性（Dirichlet 定理）。

---

### D.60 $f(x) = x^2$ 的 Fourier 级数与 Basel 问题

**题目回顾**  
求 $f(x) = x^2$（$-\pi \le x \le \pi$）的 Fourier 级数，并由此推出 $\displaystyle\sum_{n=1}^\infty\frac{1}{n^2} = \frac{\pi^2}{6}$。

**思路**  
$x^2$ 是偶函数，$b_n = 0$。计算 $a_0$ 和 $a_n$，然后代 $x = \pi$（或用 Parseval 等式）。

**解答**  
$f(x) = x^2$ 为偶函数，$b_n = 0$。

$$a_0 = \frac{1}{\pi}\int_{-\pi}^\pi x^2\,dx = \frac{2}{\pi}\cdot\frac{\pi^3}{3} = \frac{2\pi^2}{3}.$$

对 $n \ge 1$（分部积分两次）：

$$a_n = \frac{2}{\pi}\int_0^\pi x^2\cos nx\,dx.$$

第一次分部（$u=x^2$，$dv = \cos nx\,dx$）：

$$= \frac{2}{\pi}\left[\frac{x^2\sin nx}{n}\Big|_0^\pi - \frac{2}{n}\int_0^\pi x\sin nx\,dx\right] = \frac{2}{\pi}\left[0 - \frac{2}{n}\int_0^\pi x\sin nx\,dx\right].$$

第二次分部（由 C.49 知 $\int_0^\pi x\sin nx\,dx = (-1)^{n+1}\pi/n$）：

$$a_n = \frac{2}{\pi}\cdot\left(-\frac{2}{n}\right)\cdot\frac{(-1)^{n+1}\pi}{n} = \frac{4(-1)^n}{n^2}.$$

Fourier 级数：

$$x^2 = \frac{\pi^2}{3} + \sum_{n=1}^\infty\frac{4(-1)^n}{n^2}\cos nx.$$

**代入 $x = \pi$**：$f(\pi) = \pi^2$，$\cos n\pi = (-1)^n$：

$$\pi^2 = \frac{\pi^2}{3} + \sum_{n=1}^\infty\frac{4(-1)^n\cdot(-1)^n}{n^2} = \frac{\pi^2}{3} + 4\sum_{n=1}^\infty\frac{1}{n^2}.$$

$$\implies \sum_{n=1}^\infty\frac{1}{n^2} = \frac{\pi^2 - \pi^2/3}{4} = \frac{2\pi^2/3}{4} = \frac{\pi^2}{6}.$$

**答案**  
$x^2 = \dfrac{\pi^2}{3} + \displaystyle\sum_{n=1}^\infty\dfrac{4(-1)^n}{n^2}\cos nx$；$\displaystyle\sum_{n=1}^\infty\dfrac{1}{n^2} = \dfrac{\pi^2}{6}$（Basel 问题）。

**总结**  
Basel 问题（$\sum 1/n^2 = \pi^2/6$）是数学史上著名难题，由欧拉在 1735 年首次解决。Fourier 级数给出了最简洁的现代证明之一。代入端点 $x=\pi$ 的关键是保证 Fourier 级数在该点收敛——而 $x^2$ 连续，Dirichlet 条件自动满足。

---

## 第三部分：E 组提升题（E.29–E.36）

---

### E.29 $\sum (-1)^n/\sqrt{n}$ 的条件收敛性

**题目回顾**  
对级数 $\displaystyle\sum_{n=1}^\infty \frac{(-1)^n}{\sqrt{n}}$：证明收敛；讨论是否绝对收敛；结合 Riemann 重排定理给出完整结论。

**思路**  
Leibniz 判别法证明收敛；$p$-级数（$p=1/2<1$）证明不绝对收敛；然后引用 Riemann 重排定理。

**解答**  
**（1）收敛性（Leibniz 判别法）**

令 $b_n = 1/\sqrt{n} > 0$。

- 单调性：$b_{n+1} = 1/\sqrt{n+1} < 1/\sqrt{n} = b_n$，单调递减。  
- 极限：$\lim_{n\to\infty} b_n = 0$。

由 Leibniz 判别法，$\displaystyle\sum_{n=1}^\infty \dfrac{(-1)^n}{\sqrt{n}}$ **收敛**。

**（2）非绝对收敛**

$\displaystyle\sum_{n=1}^\infty \left|\frac{(-1)^n}{\sqrt{n}}\right| = \sum_{n=1}^\infty \frac{1}{\sqrt{n}} = \sum_{n=1}^\infty \frac{1}{n^{1/2}}$

是 $p = 1/2 < 1$ 的 $p$-级数，**发散**。故原级数不绝对收敛。

**（3）Riemann 重排定理**

由于级数条件收敛（收敛但不绝对收敛），由 Riemann 重排定理：对任意实数 $L$（包括 $\pm\infty$），存在该级数的一个重排，使得重排后的级数收敛到 $L$。

这一定理揭示了条件收敛级数的"不稳定性"：收敛性依赖于项的排列顺序。

**答案**  
$\displaystyle\sum_{n=1}^\infty \dfrac{(-1)^n}{\sqrt{n}}$ 条件收敛，不绝对收敛；重排后可以收敛到任意实数或发散到 $\pm\infty$。

**总结**  
绝对收敛与条件收敛的本质区别体现在重排定理上：只有绝对收敛的级数才对重排具有鲁棒性。考研常考"判断收敛类型"，需同时考察 $\sum |a_n|$。

---

### E.30 $\sum x^n/n$ 的收敛域、和函数与 Abel 定理

**题目回顾**  
求幂级数 $\displaystyle\sum_{n=1}^\infty \frac{x^n}{n}$ 的收敛半径、收敛域与和函数。

**思路**  
比值法得 $R=1$；端点逐一验证；$|x|<1$ 内逐项积分推导和函数；用 Abel 定理处理端点 $x=-1$。

**解答**  
**收敛半径**：$R = \lim_{n\to\infty}\left|\frac{1/n}{1/(n+1)}\right| = 1$。

**端点**：
- $x = 1$：$\displaystyle\sum_{n=1}^\infty \frac{1}{n}$ 调和级数，**发散**。  
- $x = -1$：$\displaystyle\sum_{n=1}^\infty \frac{(-1)^n}{n}$，由 Leibniz 判别法**收敛**。

故收敛域为 $[-1, 1)$。

**和函数**（$|x| < 1$）：令 $S(x) = \displaystyle\sum_{n=1}^\infty \dfrac{x^n}{n}$，$S(0) = 0$。逐项求导：

$$S'(x) = \sum_{n=1}^\infty x^{n-1} = \frac{1}{1-x}, \quad |x| < 1.$$

积分：$S(x) = \displaystyle\int_0^x \frac{dt}{1-t} = -\ln(1-x)$（$|x|<1$）。

在 $x = -1$ 处，级数收敛，由 Abel 定理 $S$ 在 $x=-1$ 处左连续，

$$S(-1) = \lim_{x\to -1^+}(-\ln(1-x)) = -\ln 2.$$

**答案**  
收敛域 $[-1, 1)$；和函数 $S(x) = -\ln(1-x)$（$x \in [-1, 1)$）。特别 $S(-1) = -\ln 2$。

**总结**  
本题综合了比值法、端点验证、逐项求导积分、Abel 定理四个核心工具，是幂级数部分的综合训练题。Abel 定理在端点连续性的应用中不可缺少。

---

### E.31 $\sum n!/n^n$ 的收敛性（比值法 + Stirling 估计）

**题目回顾**  
判断 $\displaystyle\sum_{n=1}^\infty \frac{n!}{n^n}$ 的收敛性，并用 Stirling 公式给出渐近估计。

**思路**  
比值法是主要工具；Stirling 公式 $n! \approx \sqrt{2\pi n}(n/e)^n$ 给出更精确的量级描述。

**解答**  
**（1）比值法**

$$\frac{a_{n+1}}{a_n} = \frac{(n+1)!}{(n+1)^{n+1}}\cdot\frac{n^n}{n!} = \frac{n^n}{(n+1)^n} = \left(\frac{n}{n+1}\right)^n = \frac{1}{\left(1+\frac{1}{n}\right)^n} \to \frac{1}{e} < 1.$$

由比值判别法，级数**收敛**。

**（2）Stirling 估计**

由 Stirling 公式 $n! \approx \sqrt{2\pi n}\left(\dfrac{n}{e}\right)^n$，

$$a_n = \frac{n!}{n^n} \approx \frac{\sqrt{2\pi n}(n/e)^n}{n^n} = \sqrt{2\pi n}\cdot\frac{1}{e^n}.$$

故 $a_n \sim C\sqrt{n}\cdot e^{-n}$（$C = \sqrt{2\pi}$），级数项以指数速度趋零，收敛速度极快（超指数）。

**（3）比值极限**

比值极限 $l = 1/e \approx 0.368$，说明相邻项之比约为 $1/e$，收敛速度与公比为 $1/e$ 的几何级数相当。

**答案**  
$\displaystyle\sum_{n=1}^\infty \dfrac{n!}{n^n}$ 收敛；通项 $a_n \approx \sqrt{2\pi n}\,e^{-n}$，以指数速度衰减。

**总结**  
比值法给出收敛性，Stirling 公式给出量级。两者配合，对含阶乘的级数能做出完整的渐近分析，是研究生数学和概率论中的常用工具。

---

### E.32 $\sum nx^{n-1}$ 的和函数与 $\sum n/2^n$

**题目回顾**  
求 $\displaystyle\sum_{n=1}^\infty n x^{n-1}$（$|x|<1$）的和函数，并计算 $\displaystyle\sum_{n=1}^\infty \dfrac{n}{2^n}$。

**思路**  
注意到 $\displaystyle\sum_{n=1}^\infty nx^{n-1} = \dfrac{d}{dx}\displaystyle\sum_{n=1}^\infty x^n$（对几何级数逐项求导）；代 $x = 1/2$ 得数值结果。

**解答**  
**和函数**：由 $\displaystyle\sum_{n=0}^\infty x^n = \dfrac{1}{1-x}$（$|x|<1$），对 $x$ 求导：

$$\frac{d}{dx}\left(\frac{1}{1-x}\right) = \frac{1}{(1-x)^2} = \frac{d}{dx}\sum_{n=0}^\infty x^n = \sum_{n=1}^\infty nx^{n-1}.$$

（注意 $n=0$ 项求导后为 $0$，可从 $n=1$ 开始。）

故 $\displaystyle\sum_{n=1}^\infty nx^{n-1} = \dfrac{1}{(1-x)^2}$（$|x|<1$）。

**计算 $\sum n/2^n$**：将 $|x|<1$ 内的公式乘以 $x$ 得 $\displaystyle\sum_{n=1}^\infty nx^n = \dfrac{x}{(1-x)^2}$，令 $x = \dfrac{1}{2}$：

$$\sum_{n=1}^\infty \frac{n}{2^n} = \frac{1/2}{(1-1/2)^2} = \frac{1/2}{1/4} = 2.$$

**验证**：前几项 $\frac{1}{2} + \frac{2}{4} + \frac{3}{8} + \frac{4}{16} + \cdots = 0.5 + 0.5 + 0.375 + 0.25 + \cdots$ 趋近于 $2$，与答案吻合。

**答案**  
$\displaystyle\sum_{n=1}^\infty nx^{n-1} = \dfrac{1}{(1-x)^2}$（$|x|<1$）；$\displaystyle\sum_{n=1}^\infty\dfrac{n}{2^n} = 2$。

**总结**  
逐项求导是幂级数的核心运算，保证在收敛域内部（开区间）成立。将"含 $n$ 系数的级数"转化为"对几何级数求导"，是处理此类问题的标准范式。

---

### E.33 $\arctan x$ 的 Maclaurin 级数与 $\pi$ 的逼近

**题目回顾**  
将 $f(x) = \arctan x$ 展开为 Maclaurin 级数，讨论收敛域，并估计 $x = 1/\sqrt{3}$ 时保精度 $10^{-6}$ 所需项数。

**思路**  
由 $1/(1+x^2)$ 逐项积分得 $\arctan x$；端点 $x = \pm 1$ 需用 Abel 定理；$x = 1/\sqrt{3}$ 对应 $\pi/6$，收敛更快。

**解答**  
**（1）展开式**

由 $\dfrac{1}{1+t^2} = \displaystyle\sum_{n=0}^\infty(-1)^n t^{2n}$（$|t|<1$），积分：

$$\arctan x = \sum_{n=0}^\infty\frac{(-1)^n x^{2n+1}}{2n+1}, \quad |x| \le 1.$$

收敛半径 $R = 1$；端点 $x = \pm 1$ 处 Leibniz 判别均收敛，故收敛域为 $[-1, 1]$。

**（2）Leibniz 公式**

$x = 1$：$\arctan 1 = \pi/4$，得 $\pi = 4\displaystyle\sum_{n=0}^\infty\dfrac{(-1)^n}{2n+1}$（收敛极慢，误差约 $\dfrac{4}{2N+3}$，需数千项达 $10^{-3}$ 精度）。

**（3）$x = 1/\sqrt{3}$ 的估计**

$\arctan(1/\sqrt{3}) = \pi/6$，故

$$\frac{\pi}{6} = \sum_{n=0}^\infty\frac{(-1)^n}{(2n+1)(\sqrt{3})^{2n+1}} = \frac{1}{\sqrt{3}}\sum_{n=0}^\infty\frac{(-1)^n}{(2n+1)3^n}.$$

这是 Leibniz 型交错级数，截断误差 $\le \dfrac{1}{\sqrt{3}(2N+3)3^{N+1}}$。  
要求误差 $\le 10^{-6}$：$\dfrac{1}{\sqrt{3}(2N+3)3^{N+1}} \le 10^{-6}$，即 $(2N+3)3^{N+1} \ge \sqrt{3}\times 10^6 \approx 1.73\times 10^6$。  
试算：$N=8$：$3^9 = 19683$，$(19)19683 \approx 3.7\times 10^5$，不够；$N=9$：$3^{10} = 59049$，$(21)59049 \approx 1.24\times 10^6$，接近；$N=10$：$3^{11}= 177147$，$(23)\times 177147 \approx 4.07\times 10^6 > 1.73\times 10^6$，满足。

故取 **$N = 10$（即 11 项）** 即可保证精度 $10^{-6}$。

**答案**  
$\arctan x = \displaystyle\sum_{n=0}^\infty\dfrac{(-1)^n x^{2n+1}}{2n+1}$（$|x| \le 1$）；取 $x = 1/\sqrt{3}$ 时约需 11 项达到 $10^{-6}$ 精度。

**总结**  
Leibniz 型交错级数的截断误差由首项略去项控制，这使得精度估计非常方便。$x=1$ 的展开收敛缓慢，工程中常用 Machin 类公式加速收敛。

---

### E.34 $\sum (n/(n+1))^{n^2}$ 的收敛性（Cauchy 根值法）

**题目回顾**  
判断 $\displaystyle\sum_{n=1}^\infty \left(\frac{n}{n+1}\right)^{n^2}$ 的收敛性（Cauchy 根值法）。

**思路**  
当 $a_n$ 含有 $n$ 次幂时，根值法（$\sqrt[n]{a_n}$）往往更简洁。计算极限 $\lim \sqrt[n]{a_n}$ 时需用对数技巧。

**解答**  
令 $a_n = \left(\dfrac{n}{n+1}\right)^{n^2}$，取 $n$ 次方根：

$$\sqrt[n]{a_n} = \left(\frac{n}{n+1}\right)^n = \left(1 - \frac{1}{n+1}\right)^n.$$

计算极限：令 $m = n+1$，当 $n\to\infty$ 时 $m\to\infty$，

$$\left(1 - \frac{1}{n+1}\right)^n = \left(1-\frac{1}{m}\right)^{m-1} = \left(1-\frac{1}{m}\right)^m \cdot \left(1-\frac{1}{m}\right)^{-1} \to e^{-1}\cdot 1 = \frac{1}{e}.$$

也可直接用对数：$n\ln\left(1-\dfrac{1}{n+1}\right) \approx n\cdot\left(-\dfrac{1}{n+1}\right) \to -1$，故极限为 $e^{-1}$。

因此 $\displaystyle\lim_{n\to\infty}\sqrt[n]{a_n} = \frac{1}{e} < 1$，由 Cauchy 根值判别法，级数**收敛**。

**与比值法对比**：若用比值法，需计算 $a_{n+1}/a_n = [(n+1)/(n+2)]^{(n+1)^2} / [n/(n+1)]^{n^2}$，较为繁杂。根值法更简洁。

**答案**  
$\displaystyle\sum_{n=1}^\infty \left(\dfrac{n}{n+1}\right)^{n^2}$ 收敛，根值极限为 $1/e < 1$。

**总结**  
当 $a_n = f(n)^{g(n)}$ 且 $g(n)$ 含有 $n$ 的高次幂时，根值法通过 $\sqrt[n]{a_n} = f(n)^{g(n)/n}$ 将问题降维。对比比值法，根值法对此类"指数型"级数更自然。

---

### E.35 $f(x) = x$ 的 Fourier 级数与 Basel 问题（Parseval 等式）

**题目回顾**  
将 $f(x) = x$（$-\pi < x < \pi$）展成 Fourier 级数，用 Parseval 等式推出 $\displaystyle\sum_{n=1}^\infty \dfrac{1}{n^2} = \dfrac{\pi^2}{6}$。

**思路**  
由 C.49 已知系数 $b_n = 2(-1)^{n+1}/n$（$a_n = 0$）；Parseval 等式 $\frac{1}{\pi}\int_{-\pi}^\pi |f|^2\,dx = \frac{a_0^2}{2} + \sum (a_n^2 + b_n^2)$ 给出数值关系。

**解答**  
**Fourier 系数**（由 C.49）：$a_0 = 0$，$a_n = 0$，$b_n = \dfrac{2(-1)^{n+1}}{n}$。

Fourier 级数：$x = \displaystyle\sum_{n=1}^\infty \frac{2(-1)^{n+1}}{n}\sin nx$（$x \in (-\pi,\pi)$）。

**Parseval 等式**（$f$ 在 $[-\pi,\pi]$ 上平方可积时成立）：

$$\frac{1}{\pi}\int_{-\pi}^\pi |f(x)|^2\,dx = \frac{a_0^2}{2} + \sum_{n=1}^\infty(a_n^2 + b_n^2).$$

左端：$\dfrac{1}{\pi}\displaystyle\int_{-\pi}^\pi x^2\,dx = \dfrac{1}{\pi}\cdot\dfrac{2\pi^3}{3} = \dfrac{2\pi^2}{3}$。

右端：$0 + \displaystyle\sum_{n=1}^\infty\left(\frac{2(-1)^{n+1}}{n}\right)^2 = \sum_{n=1}^\infty\frac{4}{n^2}$。

因此：$\dfrac{2\pi^2}{3} = 4\displaystyle\sum_{n=1}^\infty\dfrac{1}{n^2}$，即

$$\sum_{n=1}^\infty\frac{1}{n^2} = \frac{\pi^2}{6}.$$

**答案**  
$x = \displaystyle\sum_{n=1}^\infty\dfrac{2(-1)^{n+1}}{n}\sin nx$（$x\in(-\pi,\pi)$）；Parseval 等式给出 $\displaystyle\sum_{n=1}^\infty\dfrac{1}{n^2} = \dfrac{\pi^2}{6}$。

**总结**  
Parseval 等式是 Fourier 分析中的"勾股定理"，建立了函数的 $L^2$ 范数与 Fourier 系数的 $\ell^2$ 范数之间的等距关系。它是求数项级数精确和的有力工具，在量子力学、信号处理等领域也有重要应用。

---

### E.36 $\sum \sin(nx)/n^2$ 的一致收敛性与逐项积分

**题目回顾**  
证明 $\displaystyle\sum_{n=1}^\infty \dfrac{\sin nx}{n^2}$ 在 $(-\infty,+\infty)$ 上一致收敛，讨论其连续性，并计算 $\displaystyle\int_0^\pi S(x)\,dx$。

**思路**  
Weierstrass M-判别（控制函数 $M_n = 1/n^2$）证明一致收敛；然后由一致收敛推连续性和逐项积分。

**解答**  
**（1）一致收敛**

对所有 $x \in \mathbb{R}$ 和所有 $n \ge 1$：

$$\left|\frac{\sin nx}{n^2}\right| \le \frac{|\sin nx|}{n^2} \le \frac{1}{n^2} =: M_n.$$

由于 $\displaystyle\sum_{n=1}^\infty M_n = \sum_{n=1}^\infty\dfrac{1}{n^2} = \dfrac{\pi^2}{6} < +\infty$，

由 Weierstrass M-判别法，$\displaystyle\sum_{n=1}^\infty\dfrac{\sin nx}{n^2}$ 在 $\mathbb{R}$ 上**一致收敛**。

**（2）连续性**

每个函数 $f_n(x) = \dfrac{\sin nx}{n^2}$ 在 $\mathbb{R}$ 上连续，且级数一致收敛，故和函数

$$S(x) = \sum_{n=1}^\infty\frac{\sin nx}{n^2}$$

在 $\mathbb{R}$ 上**连续**。

**（3）逐项积分**

由一致收敛，可以逐项积分：

$$\int_0^\pi S(x)\,dx = \sum_{n=1}^\infty\int_0^\pi\frac{\sin nx}{n^2}\,dx = \sum_{n=1}^\infty\frac{1}{n^2}\left[-\frac{\cos nx}{n}\right]_0^\pi = \sum_{n=1}^\infty\frac{1-\cos n\pi}{n^3} = \sum_{n=1}^\infty\frac{1-(-1)^n}{n^3}.$$

当 $n$ 为偶数时 $1 - (-1)^n = 0$；当 $n = 2k-1$（奇数）时 $1-(-1)^n = 2$，故

$$\int_0^\pi S(x)\,dx = \sum_{k=1}^\infty\frac{2}{(2k-1)^3} = 2\sum_{k=1}^\infty\frac{1}{(2k-1)^3}.$$

利用已知结论 $\displaystyle\sum_{k=1}^\infty\dfrac{1}{(2k-1)^3} = \dfrac{7}{8}\zeta(3) = \dfrac{7}{8}\cdot 1.202...$（Apéry 常数），故

$$\int_0^\pi S(x)\,dx = 2\cdot\frac{7\zeta(3)}{8} = \frac{7\zeta(3)}{4} \approx \frac{7 \times 1.20206}{4} \approx 2.104.$$

（注：$\zeta(3) = \displaystyle\sum_{n=1}^\infty 1/n^3 \approx 1.202$，奇数次级数与 $\zeta$ 函数的关系为 $\displaystyle\sum_{k=1}^\infty\dfrac{1}{(2k-1)^3} = \dfrac{7\zeta(3)}{8}$，由 $\zeta(3) = \displaystyle\sum_{\text{奇}} + \displaystyle\sum_{\text{偶}} = \displaystyle\sum_{\text{奇}} + \dfrac{\zeta(3)}{8}$。）

**答案**  
$\displaystyle\sum_{n=1}^\infty\dfrac{\sin nx}{n^2}$ 在 $\mathbb{R}$ 上一致收敛，和函数连续；$\displaystyle\int_0^\pi S(x)\,dx = \dfrac{7\zeta(3)}{4}$，其中 $\zeta(3) \approx 1.202$。

**总结**  
Weierstrass M-判别是证明函数项级数一致收敛的"万能钥匙"，核心是找到一个与 $x$ 无关的绝对优控制列 $M_n$。一致收敛的三个主要应用：①和函数继承连续性；②可以逐项求导（需附加条件）；③可以逐项积分（最常用）。

---

## 附录：核心方法速查

| 方法 | 适用场景 | 关键判据 |
|------|----------|----------|
| 比值法（d'Alembert）| 含阶乘、指数 | $l = \lim |a_{n+1}/a_n|$；$l<1$ 收敛，$l>1$ 发散 |
| 根值法（Cauchy）| 含 $n$ 次幂 | $l = \lim \sqrt[n]{|a_n|}$；同上 |
| 积分判别法 | 单调递减正项级数 | $\sum f(n)$ 与 $\int f$ 同敛散 |
| 比较判别法 | 与已知级数比较 | $a_n \le b_n$ 且 $\sum b_n$ 收敛 $\Rightarrow \sum a_n$ 收敛 |
| Leibniz 判别 | 交错级数 | $b_n \searrow 0$ $\Rightarrow \sum(-1)^{n-1}b_n$ 收敛 |
| 幂级数求和 | 求和函数 | 逐项求导/积分 $\leftrightarrow$ 几何级数 |
| Fourier 展开 | 周期函数 | 计算 $a_n, b_n$ 后用 Dirichlet / Parseval |
| Abel 定理 | 端点处和函数 | 端点收敛 $\Rightarrow$ 和函数连续到端点 |
| Weierstrass M | 一致收敛 | $|f_n(x)| \le M_n$ 且 $\sum M_n < \infty$ |

---

> **本附录覆盖**：Ch.15 数项级数（C.41–C.45, D.49–D.53, E.29, E.31, E.34）、Ch.16 幂级数（C.46–C.48, D.54–D.58, E.30, E.32, E.33）、Ch.17 Fourier 级数（C.49–C.50, D.59–D.60, E.35, E.36）。
