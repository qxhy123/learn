# F2 详解：Part 2 随机变量（Ch.4-6，共 35 题）

> 覆盖离散随机变量（Ch.4）、连续随机变量（Ch.5）、多维随机变量（Ch.6）。
> 题型：PMF/PDF 归一化、期望与方差、CDF 互化、联合/边际/条件分布、协方差与相关系数、函数变换、条件期望、矩母函数。

---

## C 基础题详解（12 题）

### C.2.1（Ch.4，PMF 归一化）

**题目**：离散随机变量 $X$ 的 PMF 为 $P(X=k)=ck$，$k=1,2,3,4$，求常数 $c$ 及 $P(X\geq 3)$。

**思路**：利用归一化条件 $\sum_k P(X=k)=1$，解出 $c$；再直接累加 $P(X=3)+P(X=4)$。

**解**：

**第 1 小问**：

$$\sum_{k=1}^{4} ck = c(1+2+3+4) = 10c = 1 \implies c = \frac{1}{10}$$

**第 2 小问**：

$$P(X \geq 3) = P(X=3) + P(X=4) = \frac{3}{10} + \frac{4}{10} = \frac{7}{10}$$

**答案**：$c = \boxed{\dfrac{1}{10}}$，$P(X \geq 3) = \boxed{0.7}$

---

### C.2.2（Ch.4，期望与方差）

**题目**：离散随机变量 $X$ 的分布律为

| $X$ | $-1$ | $0$ | $1$ | $2$ |
|-----|------|-----|-----|-----|
| $P$ | $0.2$ | $0.3$ | $0.4$ | $0.1$ |

求 $E[X]$ 和 $\mathrm{Var}(X)$。

**思路**：按定义逐项加权求和；方差用 $\mathrm{Var}(X)=E[X^2]-(E[X])^2$。

**解**：

$$E[X] = (-1)(0.2) + 0(0.3) + 1(0.4) + 2(0.1) = -0.2 + 0 + 0.4 + 0.2 = 0.4$$

$$E[X^2] = 1(0.2) + 0(0.3) + 1(0.4) + 4(0.1) = 0.2 + 0 + 0.4 + 0.4 = 1.0$$

$$\mathrm{Var}(X) = E[X^2] - (E[X])^2 = 1.0 - 0.16 = 0.84$$

**答案**：$E[X] = \boxed{0.4}$，$\mathrm{Var}(X) = \boxed{0.84}$

---

### C.2.3（Ch.4，CDF 与 PMF 关系）

**题目**：$X$ 的 PMF 为 $P(X=k)=\frac{1}{4}$，$k=1,2,3,4$，求 CDF 在各整数点的值及 $P(1<X\leq 3)$。

**思路**：CDF $F(x)=P(X\leq x)=\sum_{k\leq x}P(X=k)$，逐步累加；区间概率用 CDF 差。

**解**：

$$F(1)=\frac{1}{4},\quad F(2)=\frac{2}{4}=\frac{1}{2},\quad F(3)=\frac{3}{4},\quad F(4)=1$$

$$P(1 < X \leq 3) = F(3) - F(1) = \frac{3}{4} - \frac{1}{4} = \frac{1}{2}$$

> ⚠️ 易错点：$P(1<X\leq 3)=F(3)-F(1)$，左端开区间不含 $x=1$，故减去 $F(1)$ 而非 $F(2)$。

**答案**：$F(1)=\frac{1}{4}$，$F(2)=\frac{1}{2}$，$F(3)=\frac{3}{4}$，$F(4)=1$；$P(1<X\leq 3)=\boxed{\dfrac{1}{2}}$

---

### C.2.4（Ch.5，PDF 归一化）

**题目**：连续随机变量 $X$ 的 PDF 为 $f(x)=cx^2$（$0\leq x\leq 2$），求 $c$ 及 $P(1\leq X\leq 2)$。

**思路**：归一化 $\int_0^2 cx^2\,dx=1$，解 $c$；概率为对应积分。

**解**：

**第 1 小问**：

$$\int_0^2 cx^2\,dx = c\cdot\frac{x^3}{3}\bigg|_0^2 = c\cdot\frac{8}{3} = 1 \implies c = \frac{3}{8}$$

**第 2 小问**：

$$P(1 \leq X \leq 2) = \int_1^2 \frac{3}{8}x^2\,dx = \frac{3}{8}\cdot\frac{x^3}{3}\bigg|_1^2 = \frac{1}{8}(8-1) = \frac{7}{8}$$

**答案**：$c = \boxed{\dfrac{3}{8}}$，$P(1\leq X\leq 2) = \boxed{\dfrac{7}{8}}$

---

### C.2.5（Ch.5，期望与方差——连续型）

**题目**：$X\sim U(0,4)$，写出 PDF，计算 $E[X]$ 和 $\mathrm{Var}(X)$。

**思路**：均匀分布 PDF 为常数；套公式 $E[X]=(a+b)/2$，$\mathrm{Var}(X)=(b-a)^2/12$。

**解**：

$$f(x) = \begin{cases} \dfrac{1}{4}, & 0\leq x\leq 4 \\ 0, & \text{其他} \end{cases}$$

$$E[X] = \frac{0+4}{2} = 2$$

$$\mathrm{Var}(X) = \frac{(4-0)^2}{12} = \frac{16}{12} = \frac{4}{3}$$

**答案**：$E[X]=\boxed{2}$，$\mathrm{Var}(X)=\boxed{\dfrac{4}{3}}$

---

### C.2.6（Ch.5，正态标准化）

**题目**：$X\sim N(3,4)$，求标准化变量 $Z$ 的分布，及 $P(1\leq X\leq 7)$（用 $\Phi$ 表示）。

**思路**：正态标准化 $Z=(X-\mu)/\sigma$；将原概率转化为标准正态区间。

**解**：

**第 1 小问**：$Z=\dfrac{X-3}{2}\sim N(0,1)$。

**第 2 小问**：

$$P(1\leq X\leq 7) = P\!\left(\frac{1-3}{2}\leq Z\leq\frac{7-3}{2}\right) = P(-1\leq Z\leq 2)$$

$$= \Phi(2) - \Phi(-1) = \Phi(2) - (1-\Phi(1))$$

**答案**：$Z\sim N(0,1)$；$P(1\leq X\leq 7)=\boxed{\Phi(2)-\Phi(-1)}=\Phi(2)+\Phi(1)-1$

---

### C.2.7（Ch.5，CDF 与 PDF 互化）

**题目**：$X$ 的 CDF 为 $F(x)=x^2$（$0\leq x\leq 1$），求 PDF $f(x)$ 及 $P(0.5\leq X\leq 0.8)$。

**思路**：PDF 为 CDF 对 $x$ 求导；概率直接用 CDF 差。

**解**：

**第 1 小问**：对 $F(x)$ 求导：

$$f(x) = F'(x) = \begin{cases} 2x, & 0\leq x\leq 1 \\ 0, & \text{其他} \end{cases}$$

**第 2 小问**：

$$P(0.5\leq X\leq 0.8) = F(0.8)-F(0.5) = 0.64 - 0.25 = 0.39$$

**答案**：$f(x)=2x$（$0\leq x\leq 1$）；$P(0.5\leq X\leq 0.8)=\boxed{0.39}$

---

### C.2.8（Ch.6，联合 PMF 与边缘分布）

**题目**：$(X,Y)$ 的联合 PMF 为

| | $Y=0$ | $Y=1$ |
|---|---|---|
| $X=0$ | $0.1$ | $0.3$ |
| $X=1$ | $0.4$ | $0.2$ |

求边缘 PMF，判断独立性。

**思路**：边缘 PMF 按行/列求和；独立性判据 $P(X=x,Y=y)=P(X=x)P(Y=y)$。

**解**：

**边缘 PMF**：

$$P(X=0)=0.1+0.3=0.4,\quad P(X=1)=0.4+0.2=0.6$$
$$P(Y=0)=0.1+0.4=0.5,\quad P(Y=1)=0.3+0.2=0.5$$

**独立性检验**：若独立，$P(X=0,Y=0)=0.4\times 0.5=0.2$，但实际为 $0.1\neq 0.2$。

**答案**：$P(X=0)=0.4$，$P(X=1)=0.6$，$P(Y=0)=P(Y=1)=0.5$；$X$ 与 $Y$ $\boxed{\text{不独立}}$。

---

### C.2.9（Ch.6，条件分布）

**题目**：延续 C.2.8，求条件 PMF $P(Y=y\mid X=1)$ 及 $E[Y\mid X=1]$。

**思路**：条件 PMF = 联合 PMF 除以边缘 PMF；条件期望加权求和。

**解**：

$$P(Y=0\mid X=1) = \frac{P(X=1,Y=0)}{P(X=1)} = \frac{0.4}{0.6} = \frac{2}{3}$$

$$P(Y=1\mid X=1) = \frac{0.2}{0.6} = \frac{1}{3}$$

$$E[Y\mid X=1] = 0\cdot\frac{2}{3} + 1\cdot\frac{1}{3} = \frac{1}{3}$$

**答案**：$P(Y=0\mid X=1)=\frac{2}{3}$，$P(Y=1\mid X=1)=\frac{1}{3}$；$E[Y\mid X=1]=\boxed{\dfrac{1}{3}}$

---

### C.2.10（Ch.6，协方差与相关系数）

**题目**：已知 $E[X]=1$，$E[Y]=2$，$E[XY]=3$，$\mathrm{Var}(X)=2$，$\mathrm{Var}(Y)=3$，求 $\mathrm{Cov}(X,Y)$ 和 $\rho_{XY}$。

**思路**：$\mathrm{Cov}(X,Y)=E[XY]-E[X]E[Y]$；$\rho_{XY}=\mathrm{Cov}/(\sigma_X\sigma_Y)$。

**解**：

$$\mathrm{Cov}(X,Y) = E[XY] - E[X]E[Y] = 3 - 1\times 2 = 1$$

$$\rho_{XY} = \frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathrm{Var}(X)}\sqrt{\mathrm{Var}(Y)}} = \frac{1}{\sqrt{2}\cdot\sqrt{3}} = \frac{1}{\sqrt{6}}$$

**答案**：$\mathrm{Cov}(X,Y)=\boxed{1}$，$\rho_{XY}=\boxed{\dfrac{1}{\sqrt{6}}}$

---

### C.2.11（Ch.6，期望与方差的线性性）

**题目**：$X,Y$ 独立，$E[X]=2$，$E[Y]=5$，$\mathrm{Var}(X)=3$，$\mathrm{Var}(Y)=4$，求 $E[3X-2Y+1]$ 和 $\mathrm{Var}(3X-2Y+1)$。

**思路**：期望线性性直接代入；独立时方差：$\mathrm{Var}(aX+bY)=a^2\mathrm{Var}(X)+b^2\mathrm{Var}(Y)$，常数不影响方差。

**解**：

$$E[3X-2Y+1] = 3E[X] - 2E[Y] + 1 = 3(2) - 2(5) + 1 = 6 - 10 + 1 = -3$$

$$\mathrm{Var}(3X-2Y+1) = 3^2\mathrm{Var}(X) + (-2)^2\mathrm{Var}(Y) = 9(3) + 4(4) = 27 + 16 = 43$$

> ⚠️ 易错点：常数 $+1$ 对方差无贡献；独立时 $\mathrm{Cov}(X,Y)=0$，方差可直接相加（含系数平方）。

**答案**：$E[3X-2Y+1]=\boxed{-3}$，$\mathrm{Var}(3X-2Y+1)=\boxed{43}$

---

### C.2.12（Ch.6，联合 PDF 与边缘密度）

**题目**：联合 PDF 为 $f(x,y)=6x$（$0\leq x\leq 1$，$0\leq y\leq x$），求 $X$ 的边缘 PDF $f_X(x)$ 并验证归一化。

**思路**：边缘密度对 $y$ 从 $0$ 到 $x$ 积分；归一化再对 $x$ 从 $0$ 到 $1$ 积分验证等于 $1$。

**解**：

**第 1 小问**：对 $0\leq x\leq 1$：

$$f_X(x) = \int_0^x 6x\,dy = 6x\cdot x = 6x^2$$

$$f_X(x) = \begin{cases} 6x^2, & 0\leq x\leq 1 \\ 0, & \text{其他} \end{cases}$$

**第 2 小问**：

$$\int_0^1 6x^2\,dx = 6\cdot\frac{x^3}{3}\bigg|_0^1 = 2\cdot 1 = 2 \neq 1$$

等等，重新检验原始积分域：$0\leq y\leq x\leq 1$，联合密度 $f(x,y)=6x$。

验证总概率：

$$\int_0^1\int_0^x 6x\,dy\,dx = \int_0^1 6x^2\,dx = \left[2x^3\right]_0^1 = 2$$

原题联合 PDF 有误（此为典型出题错误），若接受 $f_X(x)=6x^2$（$0\leq x\leq 1$），则验证：

$$\int_0^1 6x^2\,dx = 2 \neq 1$$

说明联合 PDF 总概率为 $2$，归一化常数应为 $3x$（令 $f(x,y)=3x$）。但按题目原文 $f(x,y)=6x$，边缘密度为 $f_X(x)=6x^2$，$0\leq x\leq 1$；验证时总积分为 $2$（题目本身联合密度未归一化，此处按题意写出推导）。

> ⚠️ 易错点：写出边缘密度时积分上限取决于联合分布的支撑域形状（此处 $y$ 上限为 $x$，不是 $1$）。

**答案**：$f_X(x) = \boxed{6x^2}$，$0\leq x\leq 1$；总积分验证 $\int_0^1 6x^2\,dx=2$（联合密度未归一化；若以题目为准，$f_X(x)=6x^2$ 为正确边缘密度形式）。

---

## D 中等题详解（15 题）

### D.2.1（Ch.5，连续随机变量函数变换）

**题目**：$X\sim U(0,2)$，令 $Y=X^2$，求 $f_Y(y)$、$E[Y]$、$\mathrm{Var}(Y)$。

**思路**：先求 $Y$ 的 CDF（由 $X$ 的 CDF 推出），再求导得 PDF；$E[Y]$ 可用 LOTUS；$\mathrm{Var}(Y)=E[Y^2]-(E[Y])^2$。

**解**：

**(a) 求 $f_Y(y)$**：

$X\sim U(0,2)$，PDF $f_X(x)=\frac{1}{2}$，$x\in[0,2]$。

$Y=X^2$ 的支撑：$Y\in[0,4]$。对 $y\in[0,4]$：

$$F_Y(y) = P(Y\leq y) = P(X^2\leq y) = P(X\leq\sqrt{y}) = \frac{\sqrt{y}}{2}$$

$$f_Y(y) = F_Y'(y) = \frac{1}{2}\cdot\frac{1}{2\sqrt{y}} = \frac{1}{4\sqrt{y}},\quad y\in(0,4]$$

验证：$\int_0^4\frac{1}{4\sqrt{y}}dy = \frac{1}{4}\cdot 2\sqrt{y}\big|_0^4 = \frac{1}{2}\cdot 2 = 1$。✓

**(b) 求 $E[Y]$**（两种方法）：

- LOTUS：$E[Y]=E[X^2]=\int_0^2 x^2\cdot\frac{1}{2}dx = \frac{1}{2}\cdot\frac{8}{3}=\frac{4}{3}$

- 直接用 $f_Y$：$E[Y]=\int_0^4 y\cdot\frac{1}{4\sqrt{y}}dy = \frac{1}{4}\int_0^4 \sqrt{y}\,dy = \frac{1}{4}\cdot\frac{2}{3}y^{3/2}\big|_0^4 = \frac{1}{6}\cdot 8 = \frac{4}{3}$ ✓

**(c) 求 $\mathrm{Var}(Y)$**：

$$E[Y^2] = E[X^4] = \int_0^2 x^4\cdot\frac{1}{2}dx = \frac{1}{2}\cdot\frac{32}{5} = \frac{16}{5}$$

$$\mathrm{Var}(Y) = E[Y^2] - (E[Y])^2 = \frac{16}{5} - \frac{16}{9} = \frac{144 - 80}{45} = \frac{64}{45}$$

**答案**：

$$f_Y(y) = \boxed{\dfrac{1}{4\sqrt{y}}},\quad y\in(0,4]; \quad E[Y]=\boxed{\dfrac{4}{3}}; \quad \mathrm{Var}(Y)=\boxed{\dfrac{64}{45}}$$

---

### D.2.2（Ch.4，离散随机变量的矩）

**题目**：$X$ 的分布律含未知参数 $a,b$，且 $E[X]=0.4$，确定 $a,b$ 并求 $E[X^2]$、$\mathrm{Var}(X)$、$E[3X^2-2X+1]$。

| $x$ | $-1$ | $0$ | $1$ | $2$ |
|-----|------|-----|-----|-----|
| $p$ | $a$ | $0.3$ | $0.2$ | $b$ |

**思路**：联立归一化方程和期望方程解 $a,b$；然后算各矩。

**解**：

**(a) 确定 $a,b$**：

归一化：$a + 0.3 + 0.2 + b = 1 \Rightarrow a + b = 0.5$

期望：$(-1)a + 0(0.3) + 1(0.2) + 2b = 0.4 \Rightarrow -a + 2b = 0.2$

两式联立：$a+b=0.5$，$-a+2b=0.2$，相加得 $3b=0.7$，故 $b=\frac{7}{30}$，$a=0.5-b=\frac{15-7}{30}=\frac{8}{30}=\frac{4}{15}$。

验证：$a=\frac{4}{15}\approx0.267$，$b=\frac{7}{30}\approx0.233$，$a+b=\frac{8+7}{30}=\frac{1}{2}$。✓

**(b) 求 $E[X^2]$ 和 $\mathrm{Var}(X)$**：

$$E[X^2] = 1\cdot\frac{4}{15} + 0\cdot 0.3 + 1\cdot 0.2 + 4\cdot\frac{7}{30} = \frac{4}{15} + 0.2 + \frac{28}{30}$$

$$= \frac{8}{30} + \frac{6}{30} + \frac{28}{30} = \frac{42}{30} = \frac{7}{5} = 1.4$$

$$\mathrm{Var}(X) = E[X^2] - (E[X])^2 = 1.4 - 0.16 = 1.24$$

**(c) 求 $E[3X^2-2X+1]$**：

$$E[3X^2-2X+1] = 3E[X^2] - 2E[X] + 1 = 3(1.4) - 2(0.4) + 1 = 4.2 - 0.8 + 1 = 4.4$$

**答案**：$a=\boxed{\dfrac{4}{15}}$，$b=\boxed{\dfrac{7}{30}}$；$E[X^2]=\boxed{1.4}$，$\mathrm{Var}(X)=\boxed{1.24}$；$E[3X^2-2X+1]=\boxed{4.4}$

---

### D.2.3（Ch.5，正态分布分位数与标准化）

**题目**：$X\sim N(2,9)$，用 $\Phi$ 表达各概率，并求满足 $P(X>c)=0.05$ 的 $c$。

**思路**：令 $Z=(X-2)/3$，将区间转化为标准正态；分位数用 $\Phi^{-1}$ 反求。

**解**：

**(a) $P(1\leq X\leq 5)$**：

$$Z=\frac{X-2}{3}:\quad P\!\left(\frac{1-2}{3}\leq Z\leq\frac{5-2}{3}\right)=P(-\tfrac{1}{3}\leq Z\leq 1)=\Phi(1)-\Phi(-\tfrac{1}{3})$$

**(b) $P(\vert X-2\vert>3)$**：

$$P(\vert X-2\vert>3)=P\!\left(\vert Z\vert>\frac{3}{3}\right)=P(\vert Z\vert>1)=2(1-\Phi(1))=2\Phi(-1)$$

**(c) 求 $c$（$P(X>c)=0.05$）**：

$$P(X>c)=P\!\left(Z>\frac{c-2}{3}\right)=0.05 \Rightarrow \frac{c-2}{3}=z_{0.05}\approx 1.645$$

$$c = 2 + 3\times 1.645 = 2 + 4.935 = 6.935$$

**答案**：

(a) $P(1\leq X\leq 5)=\Phi(1)-\Phi(-\frac{1}{3})$；(b) $P(\vert X-2\vert>3)=\boxed{2(1-\Phi(1))}$；(c) $c=\boxed{2+3z_{0.05}\approx 6.935}$

---

### D.2.4（Ch.4，泊松过程初步）

**题目**：某网站每分钟访问量 $\sim\mathrm{Poisson}(3)$，求无访问概率、超过 5 次概率，以及每次访问产生 0.1 元收益时的期望和方差。

**思路**：泊松 PMF 直接代入；收益 $R=0.1X$ 的期望方差用线性变换。

**解**：

**(a) 无访问概率**（$P(X=0)$）：

$$P(X=0) = e^{-3}\frac{3^0}{0!} = e^{-3} \approx 0.0498$$

**(b) 超过 5 次概率**：

$$P(X>5) = 1 - P(X\leq 5) = 1 - \sum_{k=0}^{5}e^{-3}\frac{3^k}{k!}$$

$$= 1 - e^{-3}\!\left(1+3+\frac{9}{2}+\frac{9}{2}+\frac{27}{8}+\frac{81}{40}\right) = 1-e^{-3}\cdot\frac{311}{40}$$

（保留累积泊松 CDF 形式）

**(c) 每分钟收益 $R=0.1X$ 的期望和方差**：

$$E[R] = 0.1\cdot E[X] = 0.1\times 3 = 0.3 \text{ 元}$$

$$\mathrm{Var}(R) = (0.1)^2\mathrm{Var}(X) = 0.01\times 3 = 0.03 \text{ 元}^2$$

**答案**：$P(X=0)=\boxed{e^{-3}}$；$P(X>5)=1-e^{-3}\sum_{k=0}^{5}\frac{3^k}{k!}$；$E[R]=\boxed{0.3}$ 元，$\mathrm{Var}(R)=\boxed{0.03}$ 元$^2$

---

### D.2.5（Ch.5，指数分布无记忆性）

**题目**：$X\sim\mathrm{Exp}(0.1)$，求 $P(X>20)$，证明无记忆性，并用无记忆性求已工作 10 小时后再工作 15 小时的概率。

**思路**：指数分布 $P(X>t)=e^{-\lambda t}$；无记忆性 $P(X>s+t\mid X>s)=P(X>t)$，用条件概率定义证明。

**解**：

**(a) $P(X>20)$**：

$$P(X>20) = e^{-0.1\times 20} = e^{-2} \approx 0.1353$$

**(b) 证明无记忆性**：对任意 $s,t>0$，

$$P(X>s+t\mid X>s) = \frac{P(X>s+t)}{P(X>s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(X>t)$$

证毕。无记忆性在概率意义上：已知元件正常工作了 $s$ 小时，其"剩余寿命"仍服从同参数指数分布。

**(c) 已工作 10 小时，再工作 15 小时的概率**：

由无记忆性：

$$P(X>10+15\mid X>10) = P(X>15) = e^{-0.1\times 15} = e^{-1.5} \approx 0.2231$$

**答案**：$P(X>20)=\boxed{e^{-2}}$；无记忆性已证；$P(X>25\mid X>10)=\boxed{e^{-1.5}}$

---

### D.2.6（Ch.4，期望与方差的线性性）

**题目**：$X,Y$ 独立，$E[X]=1$，$\mathrm{Var}(X)=2$，$E[Y]=-1$，$\mathrm{Var}(Y)=3$，求 $E[2X-3Y+4]$、$\mathrm{Var}(2X-3Y+4)$、$E[XY]$、$E[X^2Y^2]$。

**思路**：线性性直接代入；独立时 $E[XY]=E[X]E[Y]$，$E[X^2Y^2]=E[X^2]E[Y^2]$。

**解**：

**(a)**：

$$E[2X-3Y+4] = 2(1) - 3(-1) + 4 = 2 + 3 + 4 = 9$$

**(b)**：

$$\mathrm{Var}(2X-3Y+4) = 4\mathrm{Var}(X) + 9\mathrm{Var}(Y) = 4(2) + 9(3) = 8 + 27 = 35$$

**(c)**：

$$E[XY] = E[X]\cdot E[Y] = 1\times(-1) = -1$$

$$E[X^2] = \mathrm{Var}(X) + (E[X])^2 = 2 + 1 = 3$$

$$E[Y^2] = \mathrm{Var}(Y) + (E[Y])^2 = 3 + 1 = 4$$

$$E[X^2Y^2] = E[X^2]\cdot E[Y^2] = 3\times 4 = 12$$

**答案**：$E[2X-3Y+4]=\boxed{9}$；$\mathrm{Var}(2X-3Y+4)=\boxed{35}$；$E[XY]=\boxed{-1}$；$E[X^2Y^2]=\boxed{12}$

---

### D.2.7（Ch.5，均匀分布的顺序统计量）

**题目**：$X_1,X_2,X_3\overset{iid}{\sim}U(0,1)$，$M=\max(X_1,X_2,X_3)$，求 $F_M(m)$、$f_M(m)$、$E[M]$。

**思路**：最大值的 CDF 为各分量 CDF 之积（独立时）；求导得 PDF；期望直接积分。

**解**：

**(a) $F_M(m)$**：对 $m\in[0,1]$，

$$F_M(m) = P(M\leq m) = P(X_1\leq m, X_2\leq m, X_3\leq m) = m^3$$

**(b) $f_M(m)$**：

$$f_M(m) = F_M'(m) = 3m^2,\quad m\in[0,1]$$

（这是 $\mathrm{Beta}(3,1)$ 分布的密度。）

**(c) $E[M]$**：

$$E[M] = \int_0^1 m\cdot 3m^2\,dm = 3\int_0^1 m^3\,dm = 3\cdot\frac{1}{4} = \frac{3}{4}$$

> ⚠️ 易错点：$n$ 个 i.i.d. $U(0,1)$ 随机变量的最大值 $X_{(n)}\sim\mathrm{Beta}(n,1)$，期望为 $n/(n+1)$。这里 $n=3$，$E[M]=3/4$。

**答案**：$F_M(m)=m^3$（$m\in[0,1]$）；$f_M(m)=\boxed{3m^2}$；$E[M]=\boxed{\dfrac{3}{4}}$

---

### D.2.8（Ch.4，条件期望）

**题目**：掷骰子得 $N$，再掷 $N$ 枚硬币得正面数 $X$，求 $E[X\mid N=n]$、$E[X]$（重期望）、$\mathrm{Var}(X)$（条件方差公式）。

**思路**：给定 $N=n$，$X\sim B(n,1/2)$；重期望 $E[X]=E[E[X\mid N]]$；条件方差公式分解总方差。

**解**：

**(a) $E[X\mid N=n]$**：

给定 $N=n$，$X\sim B(n,1/2)$，故 $E[X\mid N=n]=\frac{n}{2}$。即 $E[X\mid N]=\frac{N}{2}$。

**(b) $E[X]$（重期望）**：

骰子 $N$ 均匀分布在 $\{1,2,3,4,5,6\}$，$E[N]=\frac{1+2+\cdots+6}{6}=\frac{21}{6}=\frac{7}{2}$。

$$E[X] = E\!\left[\frac{N}{2}\right] = \frac{E[N]}{2} = \frac{7/2}{2} = \frac{7}{4}$$

**(c) $\mathrm{Var}(X)$（条件方差公式）**：

$$\mathrm{Var}(X) = E[\mathrm{Var}(X\mid N)] + \mathrm{Var}(E[X\mid N])$$

给定 $N=n$，$\mathrm{Var}(X\mid N=n)=n\cdot\frac{1}{2}\cdot\frac{1}{2}=\frac{n}{4}$，所以 $\mathrm{Var}(X\mid N)=\frac{N}{4}$。

$$E[\mathrm{Var}(X\mid N)] = E\!\left[\frac{N}{4}\right] = \frac{7/2}{4} = \frac{7}{8}$$

$$\mathrm{Var}(E[X\mid N]) = \mathrm{Var}\!\left(\frac{N}{2}\right) = \frac{1}{4}\mathrm{Var}(N)$$

$$\mathrm{Var}(N) = E[N^2]-(E[N])^2 = \frac{1+4+9+16+25+36}{6}-\left(\frac{7}{2}\right)^2 = \frac{91}{6} - \frac{49}{4} = \frac{182-147}{12} = \frac{35}{12}$$

$$\mathrm{Var}(E[X\mid N]) = \frac{1}{4}\cdot\frac{35}{12} = \frac{35}{48}$$

$$\mathrm{Var}(X) = \frac{7}{8} + \frac{35}{48} = \frac{42}{48} + \frac{35}{48} = \frac{77}{48}$$

**答案**：$E[X\mid N=n]=\frac{n}{2}$；$E[X]=\boxed{\dfrac{7}{4}}$；$\mathrm{Var}(X)=\boxed{\dfrac{77}{48}}$

---

### D.2.9（Ch.5，混合分布）

**题目**：$X$ 以概率 $p$ 服从 $\mathrm{Exp}(1)$，以概率 $1-p$ 退化于 $0$，求 CDF、$E[X]$、$E[X^2]$、$\mathrm{Var}(X)$。

**思路**：混合 CDF 为各分量 CDF 的加权和；期望/矩用全期望公式（含点质量）。

**解**：

**(a) CDF**：设 $I\in\{0,1\}$（$P(I=1)=p$），当 $I=1$ 时 $X\sim\mathrm{Exp}(1)$，当 $I=0$ 时 $X=0$：

$$F_X(x) = \begin{cases} 0, & x < 0 \\ (1-p) + p(1-e^{-x}), & x \geq 0 \end{cases} = \begin{cases} 0, & x < 0 \\ 1 - pe^{-x}, & x \geq 0 \end{cases}$$

（$x=0$ 处有质量 $1-p$，即 $P(X=0)=1-p$；$x>0$ 绝对连续部分密度为 $pe^{-x}$。）

**(b) $E[X]$ 和 $E[X^2]$**：

$$E[X] = p\cdot E[\mathrm{Exp}(1)] + (1-p)\cdot 0 = p\cdot 1 = p$$

$$E[X^2] = p\cdot E[X_{\mathrm{Exp}}^2] + (1-p)\cdot 0 = p\cdot 2 = 2p$$

（$\mathrm{Exp}(1)$ 的二阶矩：$E[X^2]=\mathrm{Var}+\mu^2=1+1=2$。）

**(c) $\mathrm{Var}(X)$**：

$$\mathrm{Var}(X) = E[X^2] - (E[X])^2 = 2p - p^2 = p(2-p)$$

**答案**：$F_X(x)=1-pe^{-x}$（$x\geq 0$）；$E[X]=\boxed{p}$；$E[X^2]=\boxed{2p}$；$\mathrm{Var}(X)=\boxed{p(2-p)}$

---

### D.2.10（Ch.4，矩母函数入门）

**题目**：$X\sim\mathrm{Bernoulli}(p)$，计算 MGF $M_X(t)$，用导数求 $E[X]$ 和 $E[X^2]$，推广到 $Y=\sum_{i=1}^n X_i$。

**思路**：$M_X(t)=E[e^{tX}]$ 对 $X\in\{0,1\}$ 求和；$E[X^k]=M_X^{(k)}(0)$；独立时 MGF 相乘。

**解**：

**(a) $M_X(t)$**：

$$M_X(t) = E[e^{tX}] = e^{t\cdot 0}(1-p) + e^{t\cdot 1}p = (1-p) + pe^t = 1-p+pe^t$$

**(b) 利用导数求矩**：

$$M_X'(t) = pe^t \implies E[X] = M_X'(0) = p$$

$$M_X''(t) = pe^t \implies E[X^2] = M_X''(0) = p$$

（注：对 Bernoulli 分布，$E[X^2]=E[X]=p$，因为 $X^2=X$ a.s.）

**(c) $Y=X_1+\cdots+X_n$ 的 MGF**：

由独立性，MGF 相乘：

$$M_Y(t) = \prod_{i=1}^n M_{X_i}(t) = (1-p+pe^t)^n$$

这正是 $B(n,p)$ 的 MGF，故 $Y\sim B(n,p)$（二项分布）。

**答案**：$M_X(t)=\boxed{1-p+pe^t}$；$E[X]=p$，$E[X^2]=p$；$M_Y(t)=(1-p+pe^t)^n$，$Y\sim\boxed{B(n,p)}$

---

### D.2.11（Ch.5，Beta 分布积分）

**题目**：$X\sim\mathrm{Beta}(2,3)$，密度 $f(x)=12x(1-x)^2$（$0<x<1$），验证归一化，求 $E[X]$、$E[X^2]$、$\mathrm{Var}(X)$。

**思路**：展开或用 Beta 函数公式 $\int_0^1 x^{a-1}(1-x)^{b-1}dx=B(a,b)=(a-1)!(b-1)!/(a+b-1)!$（整数时）。

**解**：

**(a) 验证归一化**：

$$\int_0^1 12x(1-x)^2\,dx = 12\int_0^1 x(1-2x+x^2)\,dx = 12\int_0^1(x-2x^2+x^3)\,dx$$

$$= 12\left[\frac{1}{2}-\frac{2}{3}+\frac{1}{4}\right] = 12\cdot\frac{6-8+3}{12} = 12\cdot\frac{1}{12} = 1 \checkmark$$

**(b) $E[X]$ 和 $E[X^2]$**：

$$E[X] = \int_0^1 x\cdot 12x(1-x)^2\,dx = 12\int_0^1 x^2(1-x)^2\,dx = 12\cdot B(3,3) = 12\cdot\frac{2!\,2!}{4!} = 12\cdot\frac{4}{24} = 2$$

等等，$B(3,3)=\frac{2!\,2!}{5!-1}$——应用公式 $B(a,b)=\frac{(a-1)!(b-1)!}{(a+b-1)!}$：

$$B(3,3) = \frac{2!\,2!}{4!} = \frac{4}{24} = \frac{1}{6}$$

$$E[X] = 12\cdot B(3,3) = 12\cdot\frac{1}{6} = 2$$

> ⚠️ 易错点：$E[X]=\int_0^1 x\cdot 12x(1-x)^2dx=12\int_0^1 x^2(1-x)^2dx=12 B(3,3)=2$？实际上理论值 $E[X]=\alpha/(\alpha+\beta)=2/(2+3)=2/5$，请重新计算：

$$E[X] = 12\int_0^1 x^2(1-x)^2\,dx = 12\cdot\frac{\Gamma(3)\Gamma(3)}{\Gamma(6)} = 12\cdot\frac{2!\cdot 2!}{5!} = 12\cdot\frac{4}{120} = \frac{48}{120} = \frac{2}{5}$$

（$B(a,b)=\Gamma(a)\Gamma(b)/\Gamma(a+b)$，$B(3,3)=\frac{2!\cdot 2!}{5!}=\frac{4}{120}=\frac{1}{30}$）

$$E[X^2] = 12\int_0^1 x^3(1-x)^2\,dx = 12\cdot B(4,3) = 12\cdot\frac{3!\cdot 2!}{6!} = 12\cdot\frac{12}{720} = \frac{144}{720} = \frac{1}{5}$$

**(c) $\mathrm{Var}(X)$**：

$$\mathrm{Var}(X) = E[X^2] - (E[X])^2 = \frac{1}{5} - \frac{4}{25} = \frac{5-4}{25} = \frac{1}{25}$$

验证：公式 $\mathrm{Var}=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}=\frac{2\cdot 3}{25\cdot 6}=\frac{6}{150}=\frac{1}{25}$ ✓

**答案**：归一化已验证；$E[X]=\boxed{\dfrac{2}{5}}$；$E[X^2]=\boxed{\dfrac{1}{5}}$；$\mathrm{Var}(X)=\boxed{\dfrac{1}{25}}$

---

### D.2.12（Ch.4，负二项分布）

**题目**：独立重复试验（成功概率 $p$），$X$ 为第 $r$ 次成功所需总试验次数，写出 PMF、验证 $r=1$ 退化几何、解释期望。

**思路**：第 $k$ 次试验是第 $r$ 次成功，意味着前 $k-1$ 次中恰好有 $r-1$ 次成功。

**解**：

**(a) PMF**：

$$P(X=k) = \binom{k-1}{r-1}p^r(1-p)^{k-r},\quad k=r,r+1,r+2,\ldots$$

组合解释：第 $k$ 次必须是成功，前 $k-1$ 次中恰好有 $r-1$ 次成功（$\binom{k-1}{r-1}$ 种排列），乘以 $p^{r-1}(1-p)^{k-r}$；再乘以第 $k$ 次成功的概率 $p$。

**(b) $r=1$ 时退化为几何分布**：

$$P(X=k)\big|_{r=1} = \binom{k-1}{0}p^1(1-p)^{k-1} = p(1-p)^{k-1}$$

这正是几何分布 $\mathrm{Geom}(p)$ 的 PMF。✓

**(c) 期望解释**（$r=3$，$p=0.5$）：

$E[X]=r/p=3/0.5=6$，即平均需要 $6$ 次试验才能获得第 $3$ 次成功。直觉：每次成功平均需要 $1/p=2$ 次试验，$r$ 次成功共需约 $r\times(1/p)=6$ 次。

**答案**：

$$P(X=k)=\boxed{\binom{k-1}{r-1}p^r(1-p)^{k-r}},\quad k=r,r+1,\ldots$$

$r=1$ 时退化为 $\mathrm{Geom}(p)$；$r=3,p=0.5$ 时 $E[X]=\boxed{6}$

---

### D.2.13（Ch.5，截断分布）

**题目**：$X\sim\mathrm{Exp}(1)$，条件随机变量 $Y=(X\mid X\leq 2)$，求 CDF、密度、$E[Y]$，并与 $E[X]=1$ 比较。

**思路**：截断分布 CDF $F_Y(y)=P(X\leq y\mid X\leq 2)=F_X(y)/F_X(2)$（$y\leq 2$）；密度对应归一化；期望直接积分。

**解**：

**(a) CDF 和密度**：

对 $y\in[0,2]$：

$$F_Y(y) = \frac{P(X\leq y)}{P(X\leq 2)} = \frac{1-e^{-y}}{1-e^{-2}}$$

$$f_Y(y) = F_Y'(y) = \frac{e^{-y}}{1-e^{-2}},\quad 0\leq y\leq 2$$

**(b) $E[Y]$**：

$$E[Y] = \int_0^2 y\cdot\frac{e^{-y}}{1-e^{-2}}\,dy = \frac{1}{1-e^{-2}}\int_0^2 ye^{-y}\,dy$$

积分 $\int_0^2 ye^{-y}\,dy$（分部积分，令 $u=y$，$dv=e^{-y}dy$）：

$$\int_0^2 ye^{-y}\,dy = \left[-ye^{-y}\right]_0^2 + \int_0^2 e^{-y}\,dy = -2e^{-2} + \left[-e^{-y}\right]_0^2 = -2e^{-2} + (1-e^{-2}) = 1-3e^{-2}$$

$$E[Y] = \frac{1-3e^{-2}}{1-e^{-2}} \approx \frac{1-3(0.1353)}{1-0.1353} = \frac{0.5941}{0.8647} \approx 0.687$$

**(c) 比较**：

$E[Y]\approx 0.687 < E[X]=1$。截断到 $[0,2]$ 后，均值下降（合理：仅保留了小值部分，去掉了大于 2 的右尾）。

**答案**：$f_Y(y)=\dfrac{e^{-y}}{1-e^{-2}}$（$0\leq y\leq 2$）；$E[Y]=\boxed{\dfrac{1-3e^{-2}}{1-e^{-2}}}$；截断后均值下降（$\approx 0.687 < 1$）

---

### D.2.14（Ch.6，二维联合分布协方差）

**题目**：$(X,Y)$ 联合分布律如下，求边际分布，判断独立性，求 $\mathrm{Cov}(X,Y)$ 和 $\rho(X,Y)$。

| | $Y=0$ | $Y=1$ |
|---|---------|---------|
| $X=0$ | $0.1$ | $0.2$ |
| $X=1$ | $0.3$ | $0.4$ |

**解**：

**(a) 边际分布**：

$$P(X=0)=0.1+0.2=0.3,\quad P(X=1)=0.3+0.4=0.7$$
$$P(Y=0)=0.1+0.3=0.4,\quad P(Y=1)=0.2+0.4=0.6$$

**(b) 独立性**：

若独立，$P(X=0,Y=0)=P(X=0)P(Y=0)=0.3\times 0.4=0.12\neq 0.1$，故 $X$ 与 $Y$ 不独立。

**(c) 协方差和相关系数**：

$$E[X] = 0(0.3)+1(0.7) = 0.7,\quad E[Y] = 0(0.4)+1(0.6) = 0.6$$

$$E[XY] = \sum_{x,y}xy\cdot P(X=x,Y=y) = 0\cdot 0+0\cdot 0.2+0\cdot 0.3+1\cdot 1\cdot 0.4 = 0.4$$

$$\mathrm{Cov}(X,Y) = E[XY] - E[X]E[Y] = 0.4 - 0.7\times 0.6 = 0.4 - 0.42 = -0.02$$

$$\mathrm{Var}(X) = E[X^2]-(E[X])^2 = 0.7-0.49 = 0.21$$

$$\mathrm{Var}(Y) = E[Y^2]-(E[Y])^2 = 0.6-0.36 = 0.24$$

$$\rho(X,Y) = \frac{-0.02}{\sqrt{0.21}\times\sqrt{0.24}} = \frac{-0.02}{\sqrt{0.0504}} \approx \frac{-0.02}{0.2245} \approx -0.089$$

**答案**：边际分布如上；$X,Y$ 不独立；$\mathrm{Cov}(X,Y)=\boxed{-0.02}$；$\rho(X,Y)\approx\boxed{-0.089}$

---

### D.2.15（Ch.6，二维连续分布的条件密度）

**题目**：联合密度 $f(x,y)=2$（$0<x<y<1$），求边际密度、条件密度 $f_{Y\vert X}(y\vert x)$、$E[Y\vert X=x]$，并验证 $E[E[Y\vert X]]=E[Y]$。

**思路**：支撑域为三角形 $0<x<y<1$；对 $y$ 积分（从 $x$ 到 $1$）得 $f_X(x)$；对 $x$ 积分（从 $0$ 到 $y$）得 $f_Y(y)$。

**解**：

**(a) 边际密度**：

$$f_X(x) = \int_x^1 2\,dy = 2(1-x),\quad 0<x<1$$

$$f_Y(y) = \int_0^y 2\,dx = 2y,\quad 0<y<1$$

验证：$\int_0^1 2(1-x)dx=1$，$\int_0^1 2y\,dy=1$。✓

**(b) 条件密度 $f_{Y\vert X}(y\vert x)$**（给定 $X=x$，$y$ 在 $(x,1)$ 上）：

$$f_{Y\vert X}(y\vert x) = \frac{f(x,y)}{f_X(x)} = \frac{2}{2(1-x)} = \frac{1}{1-x},\quad x < y < 1$$

即给定 $X=x$，$Y$ 在 $(x,1)$ 上均匀分布。

**(c) 条件期望 $E[Y\vert X=x]$**：

$$E[Y\vert X=x] = \int_x^1 y\cdot\frac{1}{1-x}\,dy = \frac{1}{1-x}\cdot\frac{1-x^2}{2} = \frac{1+x}{2}$$

**验证重期望**：

$$E[E[Y\vert X]] = \int_0^1 \frac{1+x}{2}\cdot 2(1-x)\,dx = \int_0^1(1+x)(1-x)\,dx = \int_0^1(1-x^2)\,dx = 1-\frac{1}{3} = \frac{2}{3}$$

$$E[Y] = \int_0^1 y\cdot 2y\,dy = 2\int_0^1 y^2\,dy = \frac{2}{3} \checkmark$$

**答案**：$f_X(x)=2(1-x)$，$f_Y(y)=2y$；$f_{Y\vert X}(y\vert x)=\boxed{\dfrac{1}{1-x}}$（$x<y<1$）；$E[Y\vert X=x]=\boxed{\dfrac{1+x}{2}}$；重期望验证成立（均等于 $\frac{2}{3}$）

---

## E 提高题详解（8 题）

### E.2.1（Ch.5+Ch.6，卷积 + Irwin-Hall 分布 + CLT 预热）

**题目**：$X,Y$ i.i.d. $\sim U(0,1)$，$Z=X+Y$，推导 $f_Z(z)$；用 MGF 分析 $S_n=\sum_{i=1}^n X_i$（Irwin-Hall）；讨论 $n=12$ 的正态近似；分析尾部误差来源。

**思路**：卷积 $f_Z(z)=\int f_X(x)f_Y(z-x)dx$，分两段计算；MGF 乘积得 $S_n$ 的矩；CLT 说明近似精度。

**解**：

**(a) 卷积推导 $f_Z(z)$**：

$f_X(x)=f_Y(x)=1$（$0\leq x\leq 1$），卷积：

$$f_Z(z) = \int_{-\infty}^{\infty} f_X(x)f_Y(z-x)\,dx$$

需要 $0\leq x\leq 1$ 且 $0\leq z-x\leq 1$，即 $\max(0,z-1)\leq x\leq\min(1,z)$。

**段 1（$0\leq z\leq 1$）**：$0\leq x\leq z$，

$$f_Z(z) = \int_0^z 1\cdot 1\,dx = z$$

**段 2（$1<z\leq 2$）**：$z-1\leq x\leq 1$，

$$f_Z(z) = \int_{z-1}^1 1\,dx = 1-(z-1) = 2-z$$

$$f_Z(z) = \begin{cases} z, & 0\leq z\leq 1 \\ 2-z, & 1<z\leq 2 \\ 0, & \text{其他} \end{cases}$$

验证：$\int_0^1 z\,dz + \int_1^2(2-z)\,dz = \frac{1}{2} + \frac{1}{2} = 1$。✓

形状：以 $z=1$ 为顶点的等腰三角形（帽型）。

**(b) Irwin-Hall 分布的 MGF**：

$X_i\sim U(0,1)$ 的 MGF：$M_X(t)=E[e^{tX}]=\int_0^1 e^{tx}dx=\frac{e^t-1}{t}$（$t\neq 0$）。

独立时，$S_n=\sum_{i=1}^n X_i$ 的 MGF：

$$M_{S_n}(t) = \left(\frac{e^t-1}{t}\right)^n$$

由 MGF 求矩（展开 $(e^t-1)/t$ 的 Taylor 级数）：

$$\frac{e^t-1}{t} = 1 + \frac{t}{2} + \frac{t^2}{6} + \cdots = \exp\!\left(\frac{t}{2} - \frac{t^2}{12} + O(t^3)\right)$$

$$\ln M_{S_n}(t) = n\ln\frac{e^t-1}{t} = n\left(\frac{t}{2} + \frac{t^2}{12} + \cdots\right)$$

一阶系数（累积量$=$均值）：$E[S_n]=n/2$；二阶累积量（方差）：$\mathrm{Var}(S_n)=n/12$。

**(c) $n=12$ 时的正态近似精度**：

$S_{12}$ 有均值 $6$、方差 $1$（$=12/12$），故 $S_{12}-6\approx N(0,1)$。

- 精确值：$S_{12}$ 有有界支撑 $[0,12]$，在 $|z|>6$ 处概率严格为 $0$，但 $N(0,1)$ 在此处有非零概率。
- 对 $z=3$（即 $|S_{12}-6|>3$，$3\sigma$ 以外）：正态给出 $P\approx 0.0027$；Irwin-Hall（$n=12$）已近乎零（支撑限制）。
- 尾部误差：$U(0,1)$ 的有界支撑导致 $S_{12}$ 的分布比正态尾部更"轻"——Box-Muller 变换（$Z_1=\sqrt{-2\ln U_1}\cos(2\pi U_2)$）利用逆变换直接生成精确正态变量，无支撑截断问题。

**(d) 正态生成比较（伪代码）**：

```python
import numpy as np
from scipy import stats

for n in [12, 100, 1000]:
    samples = np.sum(np.random.uniform(0, 1, (100000, n)), axis=1)
    samples_std = (samples - n/2) / np.sqrt(n/12)
    D, p = stats.kstest(samples_std, 'norm')
    print(f"n={n}: KS统计量={D:.4f}, p值={p:.4g}")
```

KS 统计量随 $n$ 增大而减小（CLT 收敛），但 $n=12$ 时尾部（$|z|>3$）误差来自 $U(0,1)$ 的有界支撑（$S_{12}$ 必须在 $[0,12]$ 内，而正态无此约束）。

**答案**：

$$f_Z(z)=\begin{cases}z,&0\leq z\leq 1\\2-z,&1<z\leq 2\end{cases}\quad\text{（三角形）}$$

$M_{S_n}(t)=\left(\dfrac{e^t-1}{t}\right)^n$；$E[S_n]=\boxed{\dfrac{n}{2}}$，$\mathrm{Var}(S_n)=\boxed{\dfrac{n}{12}}$；尾部误差来自有界支撑，Box-Muller 更精确。

---

### E.2.2（Ch.4，Galton-Watson 分支过程 + PGF）

**题目**：Galton-Watson 过程，PGF $G(s)=\sum p_k s^k$，证明 $E[Z_n]=\mu^n$，分析灭绝概率，显式求解几何后代情形，联系语言模型生成。

**思路**：PGF 的嵌套性质 $G_{Z_n}(s)=G_{Z_{n-1}}(G(s))$ 是核心；灭绝概率满足不动点方程 $q=G(q)$；凸分析（$G'(1)=\mu$）决定 $q$ 的位置。

**解**：

**(a) $E[Z_n]=\mu^n$ 的证明**：

设 $G_{Z_n}(s)=E[s^{Z_n}]$。利用全期望（给定 $Z_1$，每个个体独立产后代）：

$$G_{Z_n}(s) = G_{Z_{n-1}}(G(s)) = G^{(n)}(s)\text{（$G$ 的 $n$ 次迭代）}$$

对 $s$ 求导并令 $s=1$：

$$G_{Z_n}'(1) = G_{Z_{n-1}}'(G(1))\cdot G'(1) = G_{Z_{n-1}}'(1)\cdot\mu$$

由归纳法（基：$G_{Z_1}'(1)=G'(1)=\mu$）：$G_{Z_n}'(1)=\mu^n$，即 $E[Z_n]=\mu^n$。

**方差推导**（$\mu\neq 1$）：

对 $G_{Z_n}''(1)$ 递推（利用链式法则），经计算：

$$\mathrm{Var}(Z_n) = \frac{\sigma^2\mu^{n-1}(\mu^n-1)}{\mu-1}$$

（$\mu=1$ 时特殊情形：$\mathrm{Var}(Z_n)=n\sigma^2$，线性增长。）

**(b) 灭绝概率 $q$ 的分析**：

$q=\lim_{n\to\infty}P(Z_n=0)$ 满足 $G(q)=q$（不动点方程）。

**存在性**（介值定理）：$G(0)=p_0\geq 0$，$G(1)=1$；$G$ 在 $[0,1]$ 上连续，故至少存在 $q\in[0,1]$ 使 $G(q)=q$。

**位置分析**：$G(s)$ 在 $[0,1]$ 上凸（$G''\geq 0$），$G'(1)=\mu$。

- 若 $\mu>1$：切线斜率 $\mu>1$，曲线 $y=G(s)$ 在 $s=1$ 处从下方穿越直线 $y=s$，故在 $(0,1)$ 内存在另一个交点 $q^*<1$，最小不动点为 $q^*<1$（灭绝概率 $<1$）。
- 若 $\mu\leq 1$：曲线在 $[0,1]$ 上始终低于 $y=s$（或相切于 $s=1$），最小不动点为 $q^*=1$（必然灭绝）。

**(c) 几何后代 $p_k=(1-p)p^k$（$k\geq 0$）的显式求解**：

$$G(s) = \sum_{k=0}^\infty(1-p)p^k s^k = \frac{1-p}{1-ps}$$

$\mu=G'(1)=\frac{p(1-p)}{(1-p)^2}=\frac{p}{1-p}$（$\mu>1$ 当 $p>1/2$）。

不动点方程 $G(q)=q$：

$$\frac{1-p}{1-pq} = q \implies 1-p = q(1-pq) = q - pq^2$$

$$pq^2 - q + (1-p) = 0$$

判别式：$1-4p(1-p)=(1-2p)^2$，根为 $q=\frac{1\pm(1-2p)}{2p}$：

- $q=1$（总是一个根）
- $q=\frac{1-(1-2p)}{2p}=\frac{2p-1+1-1}{2p}$，等等：

$$q = \frac{1-(1-2p)}{2p} = \frac{2p}{2p} = 1\quad\text{或}\quad q=\frac{1+(1-2p)}{2p}=\frac{2(1-p)}{2p}=\frac{1-p}{p}$$

当 $p>1/2$（$\mu>1$）时，最小根 $q^*=\frac{1-p}{p}<1$（与理论一致 ✓）。

**(d) 类比 Transformer 生成**：

每个 token 通过注意力机制影响后续 token，若平均"信息扇出" $\mu>1$，主题链路呈指数增长（$\log Z_n\approx n\log\mu$），导致主题漂移（每一层推理引入新的"子话题"）。温度参数 $T<1$ 通过 softmax 使分布更集中，等效降低 $\mu$，减少分支，提高生成连贯性（但也降低多样性）。

**答案**：

$E[Z_n]=\boxed{\mu^n}$（归纳法+PGF 嵌套）；$\mu>1$ 时 $q^*<1$，$\mu\leq 1$ 时 $q^*=1$；几何后代 $q^*=\boxed{\dfrac{1-p}{p}}$（$p>1/2$）

---

### E.2.3（Ch.5+Ch.6，柯西分布 + 无期望 + 稳定分布）

**题目**：$X,Y$ i.i.d. $\sim N(0,1)$，$C=X/Y$，推导 $C$ 的密度，证明 $E[|C|]=+\infty$，用特征函数证明稳定性，讨论金融尾部风险。

**思路**：对 $P(C\leq c)$ 分 $Y>0$、$Y<0$ 两种情形；求导得密度；验证期望积分发散；特征函数 $\varphi_C(t)=e^{-|t|}$。

**解**：

**(a) 推导柯西密度 $f_C(c)$**：

$$P(C\leq c) = P\!\left(\frac{X}{Y}\leq c\right)$$

分情形：

$$= P(X\leq cY, Y>0) + P(X\geq cY, Y<0)$$

$$= \int_0^\infty\int_{-\infty}^{cy}\phi(x)\phi(y)\,dx\,dy + \int_{-\infty}^0\int_{cy}^{\infty}\phi(x)\phi(y)\,dx\,dy$$

其中 $\phi$ 为标准正态密度。对 $c$ 求导（用 Leibniz 法则）：

$$f_C(c) = \int_0^\infty y\phi(cy)\phi(y)\,dy + \int_{-\infty}^0(-y)\phi(cy)\phi(y)\,dy = \int_{-\infty}^\infty|y|\phi(cy)\phi(y)\,dy$$

$$= \int_0^\infty 2y\phi(cy)\phi(y)\,dy = 2\int_0^\infty y\cdot\frac{1}{2\pi}e^{-\frac{c^2y^2+y^2}{2}}\,dy = \frac{1}{\pi}\int_0^\infty ye^{-\frac{(1+c^2)y^2}{2}}\,dy$$

令 $u=(1+c^2)y^2/2$，$du=(1+c^2)y\,dy$：

$$f_C(c) = \frac{1}{\pi}\cdot\frac{1}{1+c^2}\int_0^\infty e^{-u}\,du = \frac{1}{\pi(1+c^2)}$$

这正是柯西分布（标准）密度。✓

**(b) 证明 $E[|C|]=+\infty$**：

$$E[|C|] = \int_{-\infty}^\infty\frac{|c|}{\pi(1+c^2)}\,dc = \frac{2}{\pi}\int_0^\infty\frac{c}{1+c^2}\,dc = \frac{2}{\pi}\cdot\frac{1}{2}\ln(1+c^2)\bigg|_0^\infty = +\infty$$

积分发散，故期望不存在（注意"主值积分"$\mathrm{P.V.}\int_{-\infty}^\infty\frac{c}{\pi(1+c^2)}dc=0$ 是人为对称截断的结果，不等价于期望存在）。

**(c) 特征函数与稳定性**：

柯西分布的特征函数：$\varphi_C(t)=e^{-|t|}$（标准结论，推导略——对 $\int_{-\infty}^\infty\frac{e^{itc}}{\pi(1+c^2)}dc$ 用留数定理）。

若 $C_1,C_2$ i.i.d. 柯西，则 $(C_1+C_2)/2$ 的特征函数：

$$\varphi_{(C_1+C_2)/2}(t) = \varphi_{C_1}(t/2)\varphi_{C_2}(t/2) = e^{-|t/2|}\cdot e^{-|t/2|} = e^{-|t|}$$

即 $(C_1+C_2)/2$ 仍为标准柯西——稳定分布的定义：$n$ 个 i.i.d. 副本的归一化和仍与原分布同族（此处归一化系数为 $n$，而非正态的 $\sqrt{n}$）。

对比正态：$(X_1+X_2)/\sqrt{2}\sim N(0,1)$，归一化系数 $\sqrt{n}$（对应 $\alpha=2$ 稳定分布）。

**(d) 金融风险建模**：

正态假设：$P(|Z|>5\sigma)=P(|Z|>5)\approx 5.7\times 10^{-7}$（每 170 万天一次）。

柯西假设：$P(|C|>5)=\frac{2}{\pi}\arctan(1/5)\approx\frac{2}{\pi}\times 0.197=0.125$（约 12.5%，极为常见！）。

对 $25\sigma$ 事件：正态概率约 $10^{-135}$（宇宙寿命内不可能），而重尾模型下完全可能——2008 年金融危机中，量化分析师基于正态假设宣称"$25\sigma$ 事件"，本质是模型错配（尾部比正态厚得多）。稳定分布族（$\alpha<2$）是重尾建模的自然选择，包含正态（$\alpha=2$）和柯西（$\alpha=1$）两端。

**答案**：

$$f_C(c) = \boxed{\dfrac{1}{\pi(1+c^2)}}$$

$E[|C|]=\boxed{+\infty}$（积分发散）；特征函数 $\varphi_C(t)=e^{-|t|}$，$(C_1+C_2)/2\sim\text{Cauchy}$（稳定）；金融中正态假设严重低估极端事件概率。

---

### E.2.4（Ch.4+Ch.6，多项分布 + 协方差结构 + Softmax 梯度）

**题目**：$(X_1,\ldots,X_d)\sim\mathrm{Multinomial}(n,\mathbf{p})$，用 MGF 推导矩，证明协方差矩阵半正定但奇异，讨论 Softmax 梯度，分析 LDA 的两层共轭结构。

**思路**：多项 MGF $M(\mathbf{t})=(\sum_i p_i e^{t_i})^n$，对各坐标求偏导得矩；$\Sigma=n(\mathrm{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top)$ 的半正定性用 Jensen；奇异性由约束 $\sum X_i=n$ 导出。

**解**：

**(a) 利用 MGF 推导矩**：

$$M(\mathbf{t}) = \left(\sum_{i=1}^d p_i e^{t_i}\right)^n$$

$$E[X_i] = \frac{\partial}{\partial t_i}\ln M(\mathbf{t})\bigg|_{\mathbf{t}=\mathbf{0}} = n\cdot\frac{p_i e^{t_i}}{\sum_j p_j e^{t_j}}\bigg|_{\mathbf{t}=\mathbf{0}} = np_i$$

$$E[X_i^2] = \frac{\partial^2}{\partial t_i^2}\ln M(\mathbf{t})\bigg|_{\mathbf{t}=\mathbf{0}} + (E[X_i])^2$$

实际操作：从 $\partial^2 M/\partial t_i^2|_{\mathbf{0}}$ 回推，或利用 $X_i\sim B(n,p_i)$（边际为二项）得 $\mathrm{Var}(X_i)=np_i(1-p_i)$。

对 $i\neq j$，计算 $\partial^2 M/\partial t_i\partial t_j|_{\mathbf{0}}$：

$$E[X_iX_j] = \frac{\partial^2 M}{\partial t_i\partial t_j}\bigg|_{\mathbf{0}} = n(n-1)p_ip_j$$

$$\mathrm{Cov}(X_i,X_j) = n(n-1)p_ip_j - np_i\cdot np_j = -np_ip_j\quad(i\neq j)$$

**(b) 协方差矩阵 $\boldsymbol{\Sigma}=n(\mathrm{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top)$ 的半正定性**：

对任意向量 $\mathbf{v}$：

$$\mathbf{v}^\top\boldsymbol{\Sigma}\mathbf{v} = n\left[\sum_i p_i v_i^2 - \left(\sum_i p_i v_i\right)^2\right] = n\!\left[E_p[v(X)^2] - (E_p[v(X)])^2\right] = n\,\mathrm{Var}_p[v(X)] \geq 0$$

（其中 $X$ 为分布 $\mathbf{p}$ 下的随机变量，方差非负，等号当 $v$ 为常数向量时成立。）✓

**(c) 奇异性（秩 $d-1$）**：

$\sum_i X_i=n$ 是恒等约束，故 $\boldsymbol{\Sigma}\mathbf{1}=n(\mathbf{p}-\mathbf{p}(\mathbf{1}^\top\mathbf{p}))=n(\mathbf{p}-\mathbf{p})=\mathbf{0}$，即 $\mathbf{1}$ 是零特征向量。

在神经网络 softmax 输出中，logit $\mathbf{z}$ 加常数向量不改变 softmax 输出（平移不变性），体现了这一冗余度。实践中：固定参照类别 logit 为 $0$，或利用交叉熵梯度 $\partial\mathcal{L}/\partial z_i=\hat{p}_i-y_i$（输出概率减真实标签，此梯度自然满足 $\sum_i(\hat{p}_i-y_i)=0$，在奇异方向上梯度为零）。

**(d) LDA 两层共轭结构**：

先验：$\mathbf{p}\sim\mathrm{Dir}(\boldsymbol{\alpha})$，观测多项 $\mathbf{n}\sim\mathrm{Multinomial}(N,\mathbf{p})$，后验 $\mathbf{p}\vert\mathbf{n}\sim\mathrm{Dir}(\boldsymbol{\alpha}+\mathbf{n})$，MAP 估计：

$$\hat{p}_i^{\mathrm{MAP}} = \frac{\alpha_i+n_i}{\alpha_0+N}$$（Laplace 平滑，$\alpha_i$ 起伪计数作用）

LDA 两层：(1) 文档主题分布 $\boldsymbol{\theta}_d\sim\mathrm{Dir}(\boldsymbol{\alpha})$；(2) 主题词分布 $\boldsymbol{\phi}_k\sim\mathrm{Dir}(\boldsymbol{\beta})$。$\boldsymbol{\alpha}$ 小（$\ll 1$）时，Dirichlet 集中于单纯形顶点（稀疏：文档只有少数主题）；$\boldsymbol{\alpha}$ 大时，分布趋于均匀（每篇文档覆盖所有主题）。

**答案**：

$E[X_i]=np_i$，$\mathrm{Var}(X_i)=np_i(1-p_i)$，$\mathrm{Cov}(X_i,X_j)=\boxed{-np_ip_j}$（$i\neq j$）；

$\boldsymbol{\Sigma}\succeq 0$（方差非负证明）；$\boldsymbol{\Sigma}$ 奇异（$\boldsymbol{\Sigma}\mathbf{1}=\mathbf{0}$，秩 $d-1$）；MAP 估计 $\hat{p}_i=(\alpha_i+n_i)/(\alpha_0+N)$

---

### E.2.5（Ch.5+Ch.6，次序统计量 + 极值理论 + Max-Pooling）

**题目**：$X_1,\ldots,X_n$ i.i.d. $\sim F$，推导第 $k$ 次序统计量密度，联系 Beta 分布，证明 $\mathrm{Exp}(1)$ 最大值趋 Gumbel 分布，分析 max-pooling 梯度稀疏性。

**思路**：第 $k$ 次序统计量密度由组合计数推导；EVT 中 $M_n-\ln n$ 的 CDF 取极限；max-pooling 梯度为指示函数，稀疏度 $(n-1)/n$。

**解**：

**(a) 第 $k$ 次序统计量密度**：

$$f_{(k)}(x) = \frac{n!}{(k-1)!(n-k)!}[F(x)]^{k-1}[1-F(x)]^{n-k}f(x)$$

组合解释：在 $n$ 个值中，第 $k$ 小值等于 $x$ 需要：

- 选出 $k-1$ 个值严格小于 $x$：$\binom{n-1}{k-1}$ 种方式，每个概率 $F(x)$（总概率 $[F(x)]^{k-1}$）
- 选出 $n-k$ 个值严格大于 $x$：概率 $[1-F(x)]^{n-k}$
- 第 $k$ 个值取 $x$ 的密度贡献：$f(x)dx$
- 全排列修正：$n!/(k-1)!(n-k)!$（固定一个位置取 $x$，剩余按大小归类）

**(b) $U(0,1)$ 情形与 Beta 分布的联系**：

对 $X_i\sim U(0,1)$，$F(x)=x$，$f(x)=1$（$0<x<1$）：

$$f_{(k)}(x) = \frac{n!}{(k-1)!(n-k)!}x^{k-1}(1-x)^{n-k}$$

Beta 函数 $B(k,n-k+1)=\frac{(k-1)!(n-k)!}{n!}$，故：

$$f_{(k)}(x) = \frac{1}{B(k,n-k+1)}x^{k-1}(1-x)^{n-k} = f_{\mathrm{Beta}(k,n-k+1)}(x)$$

即 $X_{(k)}\sim\mathrm{Beta}(k,n-k+1)$。✓

**(c) $\mathrm{Exp}(1)$ 最大值趋 Gumbel 分布**：

$F(x)=1-e^{-x}$（$x\geq 0$），令 $M_n=X_{(n)}$：

$$P(M_n - \ln n \leq t) = P(M_n \leq \ln n + t) = [F(\ln n+t)]^n$$

$$= \left(1 - e^{-(\ln n+t)}\right)^n = \left(1 - \frac{e^{-t}}{n}\right)^n \xrightarrow{n\to\infty} e^{-e^{-t}}$$

（利用 $\lim_{n\to\infty}(1-a/n)^n=e^{-a}$，$a=e^{-t}$。）

$e^{-e^{-t}}$ 正是标准 Gumbel 分布的 CDF，故 $M_n-\ln n\xrightarrow{d}\mathrm{Gumbel}(0,1)$。

Fisher-Tippett-Gnedenko 定理：吸引域（Domain of Attraction）决定极值极限类型——指数分布属于 Gumbel 吸引域。

**(d) Max-pooling 梯度分析**：

对窗口内 $n$ 个激活值 $a_1,\ldots,a_n$ i.i.d. $\sim F$，最大值 $a_{(n)}$ 的梯度（反向传播时）为 $1$，其余 $n-1$ 个梯度为 $0$。

"梯度为零"的神经元比例期望：$(n-1)/n$（对任意连续 $F$，唯一最大值概率为 $1$）。

$L$ 层 max-pooling 后：梯度信号在每层以 $1/n$ 的概率存活（只有最大值路径传递），经 $L$ 层后期望存活概率约 $(1/n)^L$——梯度信号指数衰减（稀疏梯度问题），这是深层 max-pooling 网络中梯度消失的极值理论解释。实践上用 Global Average Pooling 替代 Global Max Pooling 可缓解梯度稀疏。

**答案**：

$$f_{(k)}(x)=\boxed{\frac{n!}{(k-1)!(n-k)!}[F(x)]^{k-1}[1-F(x)]^{n-k}f(x)}$$

$X_{(k)}\sim\mathrm{Beta}(k,n-k+1)$（$U(0,1)$ 情形）；$M_n-\ln n\xrightarrow{d}\mathrm{Gumbel}(0,1)$；梯度为零神经元比例期望 $\boxed{(n-1)/n}$

---

### E.2.6（Ch.5+Ch.6，变量变换 + Jacobian + 正规化流）

**题目**：二维正态 $\mathbf{X}\sim N(\mathbf{0},\boldsymbol{\Sigma})$（相关系数 $\rho$），推导条件分布，用 Jacobian 推导线性变换后的分布，写出正规化流密度公式，分析 RealNVP 仿射耦合层。

**思路**：配方法推导条件正态；Jacobian 行列式给出密度变换公式；RealNVP 的三角 Jacobian 结构使行列式计算为 $O(d)$。

**解**：

**(a) 联合密度与条件分布**：

$$f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{2\pi\sqrt{1-\rho^2}}\exp\!\left(-\frac{x_1^2-2\rho x_1x_2+x_2^2}{2(1-\rho^2)}\right)$$

配方（固定 $x_1$，视 $x_2$ 为变量）：

$$x_1^2-2\rho x_1x_2+x_2^2 = (x_2-\rho x_1)^2 + x_1^2(1-\rho^2)$$

$$f_{X_2\vert X_1}(x_2\vert x_1) \propto \exp\!\left(-\frac{(x_2-\rho x_1)^2}{2(1-\rho^2)}\right)$$

即 $X_2\vert X_1=x_1\sim N(\rho x_1,\, 1-\rho^2)$。

线性回归解释：$E[X_2\vert X_1=x_1]=\rho x_1$，回归系数为相关系数 $\rho$；条件方差 $1-\rho^2$ 为决定系数 $R^2=\rho^2$ 的补。

**(b) 线性变换 $\mathbf{Y}=\mathbf{A}\mathbf{X}+\mathbf{b}$**：

Jacobian 变换公式：$f_\mathbf{Y}(\mathbf{y})=f_\mathbf{X}(\mathbf{A}^{-1}(\mathbf{y}-\mathbf{b}))\cdot|\det\mathbf{A}|^{-1}$。

代入正态密度：

$$f_\mathbf{Y}(\mathbf{y}) \propto \exp\!\left(-\frac{1}{2}(\mathbf{A}^{-1}(\mathbf{y}-\mathbf{b}))^\top\boldsymbol{\Sigma}^{-1}(\mathbf{A}^{-1}(\mathbf{y}-\mathbf{b}))\right)$$

$$= \exp\!\left(-\frac{1}{2}(\mathbf{y}-\mathbf{b})^\top(\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)^{-1}(\mathbf{y}-\mathbf{b})\right)$$

故 $\mathbf{Y}\sim N(\mathbf{b},\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)$。✓（正态分布在仿射变换下封闭。）

**(c) 正规化流密度公式**：

基分布 $\mathbf{Z}\sim N(\mathbf{0},\mathbf{I}_d)$，可逆变换 $\mathbf{x}=g(\mathbf{z})$：

$$\log p_X(\mathbf{x}) = \log p_Z(g^{-1}(\mathbf{x})) + \log\vert\det J_{g^{-1}}(\mathbf{x})\vert = \log p_Z(g^{-1}(\mathbf{x})) - \log\vert\det J_g(g^{-1}(\mathbf{x}))\vert$$

训练：最大化 $\sum_i\log p_X(\mathbf{x}_i)$，等价于最小化 KL 散度 $D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_X)$。

**(d) RealNVP 仿射耦合层**：

变换：$\mathbf{x}_1=\mathbf{z}_1$，$\mathbf{x}_2=\mathbf{z}_2\odot\exp(s(\mathbf{z}_1))+t(\mathbf{z}_1)$（$s,t$ 为任意神经网络）。

**可逆性**：给定 $\mathbf{x}$，$\mathbf{z}_1=\mathbf{x}_1$，$\mathbf{z}_2=(\mathbf{x}_2-t(\mathbf{x}_1))\odot\exp(-s(\mathbf{x}_1))$。✓

**Jacobian 结构**：

$$J_g = \frac{\partial(\mathbf{x}_1,\mathbf{x}_2)}{\partial(\mathbf{z}_1,\mathbf{z}_2)} = \begin{pmatrix}\mathbf{I} & \mathbf{0} \\ \frac{\partial\mathbf{x}_2}{\partial\mathbf{z}_1} & \mathrm{diag}(\exp(s(\mathbf{z}_1)))\end{pmatrix}$$

下三角块矩阵，行列式为对角块之积：

$$\det J_g = \det(\mathbf{I})\cdot\prod_i\exp(s_i(\mathbf{z}_1)) = \exp\!\left(\sum_i s_i(\mathbf{z}_1)\right)$$

计算仅需 $O(d)$ 时间（对角元之积），无需求完整矩阵的行列式。且神经网络 $s,t$ 本身无需可逆（只需前向/后向传播）。

**答案**：

$X_2\vert X_1=x_1\sim N(\rho x_1, 1-\rho^2)$；$\mathbf{Y}\sim N(\mathbf{b},\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)$；

$$\log p_X(\mathbf{x})=\log p_Z(g^{-1}(\mathbf{x}))-\log\vert\det J_g(g^{-1}(\mathbf{x}))\vert$$

RealNVP：$\det J_g=\boxed{\exp\!\left(\sum_i s_i(\mathbf{z}_1)\right)}$，$O(d)$ 计算

---

### E.2.7（Ch.4+Ch.5，MGF + Chernoff 界 + 鞍点近似）

**题目**：证明独立时 MGF 可乘，推导 Chernoff 界及 KL 散度形式，证明 CGF 严格凸，解释鞍点方程的几何含义及与 Laplace 近似的联系。

**思路**：Chernoff 界：Markov 不等式用于 $e^{tX}$，对 $t$ 最小化；KL 散度形式对 Bernoulli 显式计算；CGF 凸性由 $K''(t)=\mathrm{Var}(X_t)>0$ 证明。

**解**：

**(a) MGF 可乘性**：

若 $X_1,\ldots,X_n$ 独立：

$$M_{S_n}(t) = E[e^{tS_n}] = E\!\left[\prod_{i=1}^n e^{tX_i}\right] = \prod_{i=1}^n E[e^{tX_i}] = [M_X(t)]^n$$

（独立性 $\Rightarrow$ 期望可分解为乘积）

CGF（累积量生成函数）$K(t)=\ln M(t)$：$K_{S_n}(t)=\ln M_{S_n}(t)=n\ln M_X(t)=nK_X(t)$，具有线性性（累积量可加）。

**(b) Chernoff 界推导**：

对任意 $t>0$，由 Markov 不等式：

$$P(S_n\geq na) = P(e^{tS_n}\geq e^{tna}) \leq \frac{E[e^{tS_n}]}{e^{tna}} = e^{n[K_X(t)-ta]}$$

对 $t$ 最小化：$\inf_{t>0} e^{n[K_X(t)-ta]}=e^{-nI(a)}$，其中：

$$I(a) = \sup_{t>0}[ta - K_X(t)]$$（Legendre 变换 / 大偏差率函数）

**Bernoulli 情形**：$X\sim\mathrm{Bernoulli}(p)$，$M_X(t)=1-p+pe^t$，$K_X(t)=\ln(1-p+pe^t)$。

鞍点条件 $K_X'(t)=a$：$\frac{pe^t}{1-p+pe^t}=a$，解得 $e^t=\frac{a(1-p)}{p(1-a)}$，代入：

$$I(a) = ta^* - K_X(t^*) = a\ln\frac{a}{p}+(1-a)\ln\frac{1-a}{1-p} = D_{\mathrm{KL}}(\mathrm{Ber}(a)\|\mathrm{Ber}(p))$$

即率函数为 KL 散度（Sanov 定理的特例）。

**(c) CGF 严格凸性**：

$$K'(t) = \frac{M'(t)}{M(t)} = E_{X_t}[X]$$

其中 $X_t$ 服从倾斜（tilted）分布 $p_t(x)\propto e^{tx}f(x)$（指数族）。

$$K''(t) = \frac{d}{dt}E_{X_t}[X] = E_{X_t}[X^2] - (E_{X_t}[X])^2 = \mathrm{Var}_{X_t}(X) > 0$$

（方差对非退化分布严格正），故 $K(t)$ 严格凸，Legendre 变换有唯一最小值点（鞍点）。

**(d) 鞍点方程的几何含义**：

$K'(\hat{t})=a$ 即"倾斜参数为 $\hat{t}$ 时，倾斜分布 $X_{\hat{t}}$ 的均值恰等于目标值 $a$"——鞍点参数 $\hat{t}$ 将分布向右（$\hat{t}>0$）偏移使均值从 $\mu<a$ 移至 $a$。

Lugannani-Rice 公式在 CLT 基础上高一阶精度（Berry-Esseen 级），适用于极端尾部的精确估计。

Laplace 近似联系：贝叶斯后验 $\log p(\boldsymbol{\theta}\vert\mathbf{x})$ 在 MAP 估计 $\hat{\boldsymbol{\theta}}$ 处做二阶 Taylor 展开：

$$\log p(\boldsymbol{\theta}\vert\mathbf{x})\approx\log p(\hat{\boldsymbol{\theta}}\vert\mathbf{x}) - \frac{1}{2}(\boldsymbol{\theta}-\hat{\boldsymbol{\theta}})^\top\mathcal{I}(\hat{\boldsymbol{\theta}})(\boldsymbol{\theta}-\hat{\boldsymbol{\theta}})$$

这等价于鞍点近似（在对数坐标下）：MAP 点即鞍点，Fisher 信息矩阵 $\mathcal{I}$ 对应 CGF 的 Hessian（二阶导数，即方差）。

**答案**：

Chernoff 界：$P(S_n\geq na)\leq e^{-nI(a)}$，其中 $I(a)=\sup_t[ta-K_X(t)]$；

Bernoulli：$I(a)=\boxed{a\ln\dfrac{a}{p}+(1-a)\ln\dfrac{1-a}{1-p}}$（KL 散度）；

$K''(t)=\mathrm{Var}_{X_t}(X)>0$（严格凸）；鞍点 $\hat{t}$：使倾斜分布均值等于 $a$。

---

### E.2.8（Ch.4+Ch.5+Ch.6，复合泊松分布 + 塔性质 + 贝尔曼方程）

**题目**：$N\sim\mathrm{Poisson}(\lambda)$，给定 $N$，$X=\sum_{i=1}^N Y_i$（$Y_i$ i.i.d.，均值 $\mu$，方差 $\sigma^2$）。用塔性质求 $E[X]$ 和 $\mathrm{Var}(X)$，用 PGF 证明 Poisson 稀疏化，推导贝尔曼方程。

**思路**：全期望公式先条件于 $N$；全方差公式 $\mathrm{Var}(X)=E[\mathrm{Var}(X\vert N)]+\mathrm{Var}(E[X\vert N])$；PGF 组合 $G_X(s)=G_N(G_Y(s))$；贝尔曼方程同样是塔性质的应用。

**解**：

**(a) 用塔性质计算 $E[X]$ 和 $E[X^2]$**：

给定 $N=n$：$X=\sum_{i=1}^n Y_i$，$E[X\vert N=n]=n\mu$，即 $E[X\vert N]=N\mu$。

$$E[X] = E[E[X\vert N]] = E[N\mu] = \mu E[N] = \lambda\mu$$

对 $E[X^2]$：给定 $N=n$，$X$ 为 $n$ 个 i.i.d. $Y_i$ 的和，

$$E[X^2\vert N=n] = \mathrm{Var}(X\vert N=n) + (E[X\vert N=n])^2 = n\sigma^2 + n^2\mu^2$$

$$E[X^2] = E[N\sigma^2 + N^2\mu^2] = \sigma^2 E[N] + \mu^2 E[N^2]$$

泊松分布：$E[N]=\lambda$，$E[N^2]=\mathrm{Var}(N)+(E[N])^2=\lambda+\lambda^2$。

$$E[X^2] = \lambda\sigma^2 + \mu^2(\lambda+\lambda^2) = \lambda(\sigma^2+\mu^2) + \lambda^2\mu^2$$

**(b) 全方差公式求 $\mathrm{Var}(X)$**：

$$E[\mathrm{Var}(X\vert N)] = E[N\sigma^2] = \lambda\sigma^2$$

$$\mathrm{Var}(E[X\vert N]) = \mathrm{Var}(N\mu) = \mu^2\mathrm{Var}(N) = \mu^2\lambda$$

$$\mathrm{Var}(X) = \lambda\sigma^2 + \lambda\mu^2 = \lambda(\sigma^2+\mu^2)$$

**特例验证**：若 $Y_i\equiv 1$（$\mu=1,\sigma^2=0$），$X=N\sim\mathrm{Poisson}(\lambda)$，$\mathrm{Var}(X)=\lambda(0+1)=\lambda$ ✓。

**(c) Poisson 稀疏化（$Y_i\sim\mathrm{Bernoulli}(p)$）**：

PGF 组合：$G_X(s)=G_N(G_Y(s))$。

$G_Y(s)=E[s^Y]=(1-p)+ps=1-p(1-s)$（Bernoulli PGF）。

$G_N(s)=e^{\lambda(s-1)}$（Poisson PGF）。

$$G_X(s) = G_N(G_Y(s)) = e^{\lambda(G_Y(s)-1)} = e^{\lambda((1-p(1-s))-1)} = e^{\lambda p(s-1)}$$

这正是 $\mathrm{Poisson}(\lambda p)$ 的 PGF，故 $X\sim\mathrm{Poisson}(\lambda p)$。

直觉：Poisson 事件以概率 $p$ 独立标记，标记后的计数仍为 Poisson，参数为 $\lambda p$（稀疏化定理）。

**(d) 贝尔曼方程的推导**：

$$V^\pi(s) = E^\pi\!\left[\sum_{t=0}^\infty\gamma^t R_t\,\bigg\vert\, S_0=s\right]$$

对第一步动作和下一状态取全期望（塔性质）：

$$V^\pi(s) = E_{a\sim\pi(\cdot\vert s)}\!\left[E_{s'\sim P(\cdot\vert s,a)}\!\left[R(s,a)+\gamma\sum_{t=0}^\infty\gamma^t R_{t+1}'\,\bigg\vert\, S_1=s'\right]\right]$$

由马尔可夫性，后续奖励的期望恰为 $V^\pi(s')$：

$$= E_{a\sim\pi(\cdot\vert s)}\!\left[R(s,a) + \gamma E_{s'\sim P(\cdot\vert s,a)}[V^\pi(s')]\right]$$

这与 (a)(b) 的结构完全一致：$V^\pi(s)=E[R+\gamma V^\pi(S')]$ 正是塔性质 $E[X]=E[E[X\vert\text{first step}]]$ 的应用——条件于第一步动作和转移，递归展开。

> ⚠️ 易错点：贝尔曼方程成立的关键是马尔可夫性（$V^\pi(s')$ 只依赖 $s'$，不依赖历史），使得塔性质中内层期望可简化为 $V^\pi(s')$。

**答案**：

$E[X]=\boxed{\lambda\mu}$；$\mathrm{Var}(X)=\boxed{\lambda(\sigma^2+\mu^2)}$（全方差公式）；

Poisson 稀疏化：$X\sim\mathrm{Poisson}(\lambda p)$（PGF 组合证明）；

贝尔曼方程：$V^\pi(s)=E_a[R(s,a)+\gamma E_{s'}[V^\pi(s')]]$（塔性质递推）

---

*（本文件共 35 题，C.2.1–C.2.12 × 12 题，D.2.1–D.2.15 × 15 题，E.2.1–E.2.8 × 8 题）*
