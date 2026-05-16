# F4 详解：Part 4 极限定理（Ch.10-12，共 24 题）

> 覆盖切比雪夫/马尔可夫不等式（Ch.10）、中心极限定理（Ch.11）、四种收敛类型（Ch.12）。
> 收敛强弱顺序：a.s. $\Rightarrow$ in prob. $\Leftarrow$ $L^p$；in prob. $\Rightarrow$ in dist.；逆向均不成立。

---

## C 基础题详解（8 题）

---

### C.4.1（Ch.10，切比雪夫不等式）

**题目**：已知 $E[X]=10$，$\mathrm{Var}(X)=4$，求 $P(|X-10|\ge3)$ 的上界和 $P(7\le X\le13)$ 的下界。

**思路**：切比雪夫不等式 $P(|X-\mu|\ge k)\le\sigma^2/k^2$，取 $k=3$。

**解**：

(1) 切比雪夫不等式：

$$P(|X-10|\ge 3)\le \frac{\mathrm{Var}(X)}{3^2}=\frac{4}{9}.$$

(2) $\{7\le X\le13\}=\{|X-10|\le3\}$，故

$$P(7\le X\le13)=1-P(|X-10|\ge3)\ge 1-\frac{4}{9}=\frac{5}{9}.$$

**答案**：$P(|X-10|\ge3)\le\boxed{\dfrac{4}{9}}$；$P(7\le X\le13)\ge\boxed{\dfrac{5}{9}}$

---

### C.4.2（Ch.10，大数定律的应用）

**题目**：$X_i$ i.i.d.，$E[X_i]=2$，$\mathrm{Var}(X_i)=5$。WLLN 结论及需多大 $n$ 使误差概率 $\le0.05$。

**思路**：WLLN 给出 $\bar{X}_n\xrightarrow{P}2$；再用切比雪夫对 $\bar{X}_n$ 估界。

**解**：

(1) 由 WLLN（Chebyshev 版本）：$\bar{X}_n \xrightarrow{P} 2$，即对任意 $\varepsilon>0$，$P(|\bar{X}_n-2|\ge\varepsilon)\to0$。

(2) $\mathrm{Var}(\bar{X}_n)=5/n$。取 $\varepsilon=0.5$：

$$P(|\bar{X}_n-2|\ge0.5)\le\frac{5/n}{0.5^2}=\frac{20}{n}\le0.05 \implies n\ge400.$$

**答案**：$\bar{X}_n\xrightarrow{P}2$；需 $n\ge\boxed{400}$

> ⚠️ 切比雪夫上界通常偏保守（只需二阶矩），给出的 $n$ 是充分条件，实际所需 $n$ 可能更小。

---

### C.4.3（Ch.11，中心极限定理——样本均值）

**题目**：$X_i$ i.i.d.，$\mu=4$，$\sigma^2=9$，$n=36$。求 $\bar{X}_{36}$ 的期望/方差和近似分布，计算 $P(\bar{X}_{36}>5)$。

**思路**：$\bar{X}_n$ 精确期望为 $\mu$，方差为 $\sigma^2/n$；CLT 给出渐近正态。

**解**：

(1) $E[\bar{X}_{36}]=4$，$\mathrm{Var}(\bar{X}_{36})=9/36=1/4$，$\mathrm{SD}(\bar{X}_{36})=0.5$。

(2) 由 CLT：$\bar{X}_{36}\stackrel{\text{近似}}{\sim} N(4,\,1/4)$。

(3) 标准化：

$$P(\bar{X}_{36}>5)=P\!\left(\frac{\bar{X}_{36}-4}{0.5}>\frac{5-4}{0.5}\right)\approx P(Z>2)=1-\Phi(2).$$

**答案**：期望 4，方差 $1/4$；近似 $N(4,1/4)$；$P(\bar{X}_{36}>5)\approx\boxed{1-\Phi(2)\approx0.0228}$

**注**：CLT 的标准化步骤是关键：
- 标准化变量 $Z_n=(\bar{X}_n-\mu)/(\sigma/\sqrt{n})$；
- 查表或记住常用分位数：$\Phi(1.645)=0.95$，$\Phi(1.96)=0.975$，$\Phi(2)=0.9772$，$\Phi(2.576)=0.995$。

---

### C.4.4（Ch.11，二项分布的正态近似）

**题目**：$X\sim B(100,0.4)$，近似计算 $P(35\le X\le45)$。

**思路**：$\mu=np=40$，$\sigma=\sqrt{np(1-p)}=\sqrt{24}$，标准化后查标准正态表。

**解**：

$$\mu=40,\quad \sigma=\sqrt{100\times0.4\times0.6}=\sqrt{24}\approx4.899.$$

$$P(35\le X\le45)\approx P\!\left(\frac{35-40}{\sqrt{24}}\le Z\le\frac{45-40}{\sqrt{24}}\right)=\Phi\!\left(\frac{5}{\sqrt{24}}\right)-\Phi\!\left(\frac{-5}{\sqrt{24}}\right)=2\Phi\!\left(\frac{5}{\sqrt{24}}\right)-1.$$

$\dfrac{5}{\sqrt{24}}\approx1.021$，故结果 $\approx 2\Phi(1.02)-1$。

**答案**：$\boxed{2\Phi\!\left(5/\sqrt{24}\right)-1\approx0.693}$

**注**：若加连续性修正（$P(34.5\le X\le45.5)$），则界为 $\Phi\!\left(5.5/\sqrt{24}\right)-\Phi\!\left(-5.5/\sqrt{24}\right)$，近似更精确。

---

### C.4.5（Ch.12，依概率收敛定义）

**题目**：$P(X_n=0)=1-1/n$，$P(X_n=n)=1/n$。验证 $X_n\xrightarrow{P}0$。

**思路**：对任意 $\varepsilon>0$，计算 $P(|X_n|\ge\varepsilon)$；当 $n>\varepsilon$ 时该概率即 $1/n\to0$。

**解**：

(1) 当 $n\ge\varepsilon$（$n$ 足够大）：$P(|X_n|\ge\varepsilon)=P(X_n=n)=\dfrac{1}{n}\to0$（$n\to\infty$）。

（若 $n<\varepsilon$，$P(|X_n|\ge\varepsilon)=0$；无论如何，极限为 0。）

(2) 对任意 $\varepsilon>0$，$P(|X_n-0|\ge\varepsilon)\to0$，故 $X_n\xrightarrow{P}0$。

> ⚠️ 注意：虽然 $X_n\xrightarrow{P}0$，但 $E[X_n]=n\cdot(1/n)=1\not\to0$，说明依概率收敛不保证期望的收敛。

**答案**：$P(|X_n|\ge\varepsilon)=1/n\to0$，$\boxed{X_n\xrightarrow{P}0}$。

---

### C.4.6（Ch.12，收敛类型辨析）

**题目**：逐条判断三个命题的正误。

**思路**：记住收敛关系图：a.s. $\Rightarrow$ in prob. $\Rightarrow$ in dist.；逆向均不成立；$L^p\Rightarrow$ in prob.

**解**：

1. **正确**。a.s. 收敛 $\Rightarrow$ 依概率收敛：设 $P(X_n\to X)=1$，则对任意 $\varepsilon>0$，$P(|X_n-X|\ge\varepsilon)\to0$（由测度论标准结论）。

2. **错误**。依概率收敛 $\not\Rightarrow$ a.s. 收敛。反例：$[0,1]$ 上均匀测度的"走马灯"序列——$X_n=\mathbf{1}_{[(k-2^j)/(2^j),(k+1-2^j)/(2^j)]}$（按 $j,k$ 遍历）依概率收敛到 0，但几乎处处不收敛。

3. **错误**。依分布收敛 $\not\Rightarrow$ 依概率收敛（极限为随机变量时）。若极限为常数 $c$ 则等价，但一般不然。反例：$X_n\stackrel{d}{=}X\sim N(0,1)$ 互相独立，则 $X_n\xrightarrow{d}X$ 但 $X_n\not\to{P}X$。

**答案**：命题 1 $\boxed{\text{正确}}$，命题 2 $\boxed{\text{错误}}$，命题 3 $\boxed{\text{错误}}$。

---

### C.4.7（Ch.12，连续映射定理）

**题目**：$X_n\xrightarrow{P}3$，$Y_n\xrightarrow{P}2$，求 $X_n^2$、$X_n+Y_n$、$X_n\cdot Y_n$ 的依概率极限。

**思路**：连续映射定理（CMT）：若 $X_n\xrightarrow{P}c$，$g$ 连续，则 $g(X_n)\xrightarrow{P}g(c)$；依概率收敛对加减乘均封闭。

**解**：

(1) $g(x)=x^2$ 在 $x=3$ 处连续，故 $X_n^2\xrightarrow{P}3^2=9$。

(2) $X_n+Y_n\xrightarrow{P}3+2=5$（加法连续）。

(3) $X_n\cdot Y_n\xrightarrow{P}3\times2=6$（乘法连续）。

**答案**：$X_n^2\xrightarrow{P}\boxed{9}$；$X_n+Y_n\xrightarrow{P}\boxed{5}$；$X_n Y_n\xrightarrow{P}\boxed{6}$

**注**：CMT 对依分布收敛同样成立（Slutsky 定理的推广）：若 $X_n\xrightarrow{d}X$，$g$ 连续，则 $g(X_n)\xrightarrow{d}g(X)$。但注意依分布收敛 + 连续映射只保证分布收敛，不一定有依概率收敛。

---

### C.4.8（Ch.10，马尔可夫不等式）

**题目**：$X\ge0$，$E[X]=6$，求 $P(X\ge20)$ 和 $P(X\ge30)$ 的上界。

**思路**：马尔可夫不等式：$P(X\ge a)\le E[X]/a$（$X\ge0$，$a>0$）。

**解**：

(1) $P(X\ge20)\le\dfrac{6}{20}=\dfrac{3}{10}$.

(2) $P(X\ge30)\le\dfrac{6}{30}=\dfrac{1}{5}$.

**答案**：$P(X\ge20)\le\boxed{3/10}$；$P(X\ge30)\le\boxed{1/5}$

**注**：马尔可夫不等式条件弱（只需非负和期望有限），但上界通常很松。更高阶矩可给出更紧的界：$P(X\ge a)\le E[X^k]/a^k$（$k\ge1$）。

常见不等式比较（$X$ 非负，$a>0$）：

| 不等式 | 条件 | 上界 |
|--------|------|------|
| Markov | $E[X]<\infty$ | $E[X]/a$ |
| Chebyshev | $\mathrm{Var}(X)<\infty$ | $\mathrm{Var}(X)/(a-\mu)^2$ |
| Hoeffding | $X\in[c_i,d_i]$ | $e^{-2n^2\varepsilon^2/\sum(d_i-c_i)^2}$ |
| Bernstein | 有界+方差控制 | $e^{-n\varepsilon^2/(2\sigma^2+2M\varepsilon/3)}$ |

---

## D 中等题详解（10 题）

---

### D.4.1（Ch.10，大数定律验证条件）

**题目**：$X_n$ 独立，$P(X_n=\pm n^\alpha)=1/2$（$\alpha>0$）。验证 WLLN 的 Chebyshev 条件，分析 $\alpha=1$ 情形。

**思路**：Chebyshev-WLLN 要求 $\frac{1}{n^2}\sum_{k=1}^n\mathrm{Var}(X_k)\to0$；逐步计算并确定 $\alpha$ 的范围。

**解**：

**(a)** $E[X_n]=\frac{1}{2}(n^\alpha)+\frac{1}{2}(-n^\alpha)=0$；$\mathrm{Var}(X_n)=E[X_n^2]=n^{2\alpha}$。

**(b)** Chebyshev 条件：

$$\frac{1}{n^2}\sum_{k=1}^n\mathrm{Var}(X_k)=\frac{1}{n^2}\sum_{k=1}^n k^{2\alpha}\sim\frac{1}{n^2}\cdot\frac{n^{2\alpha+1}}{2\alpha+1}=\frac{n^{2\alpha-1}}{2\alpha+1}.$$

此量 $\to0$ $\Leftrightarrow$ $2\alpha-1<0$ $\Leftrightarrow$ $\alpha<1/2$。

故 **WLLN 成立的 Chebyshev 条件：$0<\alpha<1/2$**。

**(c)** $\alpha=1$ 时，上述量 $\sim n/(3)\to\infty$，Chebyshev 方法失效。此时 $n^{-1}\sum X_i\xrightarrow{P}0$ 实际仍成立（对称性 + 截尾论证），但 Chebyshev 不给出证明。

> ⚠️ Chebyshev 条件仅为充分条件，$\alpha\ge1/2$ 时需其他方法（如特征函数）判断。

**答案**：(a) $E[X_n]=0$，$\mathrm{Var}(X_n)=n^{2\alpha}$；(b) $\boxed{0<\alpha<1/2}$；(c) Chebyshev 不足以证明，但 WLLN 仍成立。

**延伸**：当 $1/2\le\alpha<1$ 时，可用特征函数方法（Feller WLLN）在对称分布下证明 $\bar{X}_n\xrightarrow{P}0$，即使方差无穷大；$\alpha\ge1$ 时均值可能不存在，WLLN 一般不成立。

---

### D.4.2（Ch.11，CLT 的应用）

**题目**：1000 个客户，单客户年赔付均值 500、标准差 2000。总赔付 $S$ 的概率计算。

**思路**：$S=\sum_{i=1}^{1000}X_i$，$E[S]=500000$，$\mathrm{SD}(S)=2000\sqrt{1000}$；标准化后用正态表。

**解**：

$E[S]=1000\times500=500000$，$\mathrm{Var}(S)=1000\times2000^2=4\times10^9$，$\mathrm{SD}(S)=2000\sqrt{1000}\approx63246$。

**(a)** 标准化：

$$P(S>550000)=P\!\left(Z>\frac{550000-500000}{63246}\right)=P(Z>0.7906)\approx1-\Phi(0.79)\approx0.2148.$$

**(b)** $P(S\le c)\ge0.99$：需 $\dfrac{c-500000}{63246}\ge z_{0.01}=2.326$，故

$$c\ge500000+2.326\times63246\approx500000+147109\approx647109 \text{ 元}.$$

**(c)** 指数分布偏度较大（$\gamma_1=2$，$E[|X_i-\mu_i|^3]=2\sigma^3$），Berry-Esseen 上界 $\approx C\cdot2\sigma^3/(\sigma^3\sqrt{n})=2C/\sqrt{1000}\approx0.030$，即正态近似误差不超过 3%。$n=1000$ 时 CLT 近似相当可靠，尾部进一步用精算工具（如鞍点近似）验证更佳。

**答案**：(a) $P(S>550000)\approx\boxed{1-\Phi(0.79)\approx0.215}$；(b) $c\approx\boxed{647109}$ 元；(c) $n=1000$ 时 B-E 误差 $\le3\%$，近似可靠。

---

### D.4.3（Ch.10，Markov 不等式与 Chebyshev 不等式）

**题目**：证明 Markov 不等式，再由此推导 Chebyshev 不等式，并对 $U(0,1)$ 比较松紧。

**思路**：利用示性函数法证 Markov；令 $Y=(X-\mu)^2$ 应用 Markov 得 Chebyshev。

**解**：

**(a) Markov 不等式证明**：$X\ge0$，$a>0$。

$$E[X]=E[X\cdot\mathbf{1}_{X\ge a}]+E[X\cdot\mathbf{1}_{X<a}]\ge E[X\cdot\mathbf{1}_{X\ge a}]\ge a\cdot P(X\ge a).$$

故 $P(X\ge a)\le E[X]/a$。$\square$

**(b) Chebyshev 推导**：令 $Y=(X-\mu)^2\ge0$，$E[Y]=\sigma^2$，$a=(k\sigma)^2$：

$$P(|X-\mu|\ge k\sigma)=P(Y\ge k^2\sigma^2)\le\frac{\sigma^2}{k^2\sigma^2}=\frac{1}{k^2}.\quad\square$$

**(c)** $X\sim U(0,1)$：$\mu=1/2$，$\sigma=1/\sqrt{12}$，$k=2$，$k\sigma=2/\sqrt{12}=1/\sqrt{3}\approx0.577$。

Chebyshev 上界：$1/k^2=1/4=0.25$。

精确值：$P(|X-1/2|\ge1/\sqrt{3})=P(X\le1/2-1/\sqrt{3})\cup P(X\ge1/2+1/\sqrt{3})$。

$1/2-1/\sqrt{3}\approx-0.077<0$，$1/2+1/\sqrt{3}\approx1.077>1$，两个事件概率均为 0。

**精确值 = 0**，而上界为 0.25，可见 Chebyshev 不等式非常保守。

**答案**：Markov 和 Chebyshev 均已证明；精确值 $\boxed{0}$，Chebyshev 上界 $1/4$，非常保守。

> ⚠️ 均匀分布是有界分布，其尾部衰减极快，而 Chebyshev 不等式对任意二阶矩有限的分布均成立，因此对有界分布非常保守。Hoeffding 不等式对有界随机变量给出指数型界，远比 Chebyshev 紧。

---

### D.4.4（Ch.12，依概率收敛的反例）

**题目**：$P(X_n=n)=1/n$，$P(X_n=0)=1-1/n$。证 $X_n\xrightarrow{P}0$ 但 $E[X_n]\not\to0$。

**思路**：计算期望；再按定义验证依概率收敛；最后指明含义。

**解**：

**(a)** $E[X_n]=n\cdot\dfrac{1}{n}+0\cdot\left(1-\dfrac{1}{n}\right)=1\not\to0$。

**(b)** 对任意 $\varepsilon>0$，当 $n>\varepsilon$ 时：

$$P(|X_n-0|\ge\varepsilon)=P(X_n=n)=\frac{1}{n}\to0.$$

故 $X_n\xrightarrow{P}0$。$\square$

**(c)** 含义：依概率收敛只要求大概率下随机变量接近极限，但 $X_n=n$ 这个"罕见大值"（概率 $1/n\to0$）贡献了固定期望 1，导致 $E[X_n]=1$ 始终不变。**依概率收敛不蕴含均值收敛（$L^1$ 收敛）。**

> ⚠️ 若还有一致可积性，则依概率收敛 $\Rightarrow$ $L^1$ 收敛。本例中 $\{X_n\}$ 不一致可积。

**答案**：$E[X_n]=\boxed{1}$ 对所有 $n$；$X_n\xrightarrow{P}0$ 已证；二者不矛盾，依概率收敛不保证均值收敛。

**注**：一致可积（UI）是"弥补"这个缺口的关键条件：若 $\{X_n\}$ 依概率收敛到 $X$ 且一致可积，则 $E[X_n]\to E[X]$。本例 $\{X_n\}$ 不满足 UI（因为 $E[X_n\mathbf{1}_{X_n\ge M}]=P(X_n=n)\cdot n=1$ 对任意 $M$ 及足够大的 $n$）。

---

### D.4.5（Ch.11，CLT 的 Berry-Esseen 定理应用）

**题目**：$X_i\sim\mathrm{Bernoulli}(1/2)-1/2$（中心化），利用 Berry-Esseen 定理分析近似误差。

**思路**：计算 $\rho=E[|X_i|^3]$，代入上界公式，再解 $n$ 使误差小于给定阈值。

**解**：

**(a)** 中心化 Bernoulli：$P(X_i=1/2)=P(X_i=-1/2)=1/2$。

$E[X_i^2]=1/4$（已知标准化为方差 1 的情形时需重新确认），这里按题意 $E[X_i^2]=1$（即已缩放）。

原始：$Y_i=\text{Bernoulli}(1/2)-1/2$ 取值 $\pm1/2$，$E[Y_i^2]=1/4$，$E[|Y_i|^3]=1/8$。

题目要求 $E[X_i^2]=1$，故令 $X_i=2Y_i$，取值 $\pm1$，则 $\rho=E[|X_i|^3]=1$。

**(b)** $n=100$，Berry-Esseen 上界：

$$\sup_x|P(S_n/\sqrt{n}\le x)-\Phi(x)|\le\frac{C\rho}{\sqrt{n}}=\frac{0.4785\times1}{\sqrt{100}}=\frac{0.4785}{10}\approx0.0479.$$

**(c)** 要使误差 $<0.01$：$\dfrac{C\rho}{\sqrt{n}}<0.01$，即 $\sqrt{n}>C\rho/0.01=47.85$，故 $n>2290$。

**答案**：(a) $\rho=\boxed{1}$；(b) 上界 $\approx\boxed{0.0479}$；(c) 需 $n\ge\boxed{2290}$。

**注**：Berry-Esseen 定理给出均匀（sup 范数）误差界，是最强的收敛速率结论。对特定点 $x$（如 $x=0$）的误差可用 Edgeworth 展开得到更精细的 $O(1/n)$ 阶修正。

---

### D.4.6（Ch.12，各种收敛关系）

**题目**：证明 a.s. $\Rightarrow$ in prob.（正向）；构造 in prob. $\not\Rightarrow$ a.s. 的反例；证 $L^2\Rightarrow$ in prob.

**思路**：(a) 用测度论逻辑；(b) 走马灯序列；(c) 利用 Markov 不等式。

**解**：

**(a)** 设 $X_n\xrightarrow{\text{a.s.}}X$，即 $P(\omega:\lim_n X_n(\omega)=X(\omega))=1$。

对任意 $\varepsilon>0$，定义 $A_n=\bigcup_{k\ge n}\{|X_k-X|\ge\varepsilon\}$，则 $A_n\downarrow A_\infty=\limsup_n\{|X_n-X|\ge\varepsilon\}$。

由 a.s. 收敛，$P(A_\infty)=0$，故 $P(A_n)\to0$，又 $\{|X_n-X|\ge\varepsilon\}\subset A_n$，故 $P(|X_n-X|\ge\varepsilon)\to0$。$\square$

**(b) 走马灯（Typewriter）反例**：在 $(\Omega,\mathcal{F},P)=([0,1],\mathcal{B},\lambda)$ 上，令 $X_{2^j+k}=\mathbf{1}_{[k/2^j,(k+1)/2^j]}$，$k=0,1,\ldots,2^j-1$，$j=0,1,2,\ldots$

对任意 $\varepsilon\in(0,1)$，$P(|X_n|\ge\varepsilon)\le P(X_n=1)=2^{-j}\to0$，故 $X_n\xrightarrow{P}0$。

但对任意 $\omega\in[0,1]$，$X_n(\omega)=1$ 无穷次出现，故 $X_n(\omega)\not\to0$，a.s. 不收敛。

**(c)** 由 Markov 不等式，$Y=(X_n-X)^2\ge0$，$a=\varepsilon^2$：

$$P(|X_n-X|\ge\varepsilon)=P((X_n-X)^2\ge\varepsilon^2)\le\frac{E[(X_n-X)^2]}{\varepsilon^2}=\frac{\|X_n-X\|_2^2}{\varepsilon^2}\to0.$$

故 $L^2$ 收敛 $\Rightarrow$ 依概率收敛。$\square$

**答案**：三个命题均已证明/构造反例，详见解过程。

**收敛关系速查**：

$$L^2\xrightarrow{\text{Markov}}\text{in prob.}\quad\text{a.s.}\xrightarrow{\text{见(a)}}\text{in prob.}\quad\text{in prob.}\xrightarrow{\text{子列}}\text{a.s. 子列收敛}$$

其中"子列收敛"是指：若 $X_n\xrightarrow{P}X$，则存在子列 $X_{n_k}\xrightarrow{\text{a.s.}}X$（这是将依概率收敛与 a.s. 收敛联系起来的重要工具）。

---

### D.4.7（Ch.10，强大数定律）

**题目**：陈述 Kolmogorov SLLN；以 Bernoulli 样本为例；比较 SLLN 与 WLLN。

**思路**：SLLN 给出几乎处处收敛，比 WLLN（依概率收敛）更强。

**解**：

**(a) Kolmogorov SLLN**：设 $X_1,X_2,\ldots$ i.i.d.，若 $E[|X_1|]<\infty$，令 $\mu=E[X_1]$，则

$$P\!\left(\lim_{n\to\infty}\bar{X}_n=\mu\right)=1,\quad\text{即}\;\bar{X}_n\xrightarrow{\text{a.s.}}\mu.$$

**(b)** $X_i\sim\mathrm{Bernoulli}(p)$：$E[|X_1|]=p<\infty$，$\mu=p$。由 SLLN，$\bar{X}_n\xrightarrow{\text{a.s.}}p$，即样本频率以概率 1 收敛到真实概率 $p$，这是频率解释概率（Mises 频率极限）的数学基础。

**(c)** **SLLN 比 WLLN 更强**：

- SLLN：$\bar{X}_n\xrightarrow{\text{a.s.}}\mu$（几乎处处收敛）
- WLLN：$\bar{X}_n\xrightarrow{P}\mu$（依概率收敛）
- 由 (a) 中结论，a.s. $\Rightarrow$ in prob.，故 SLLN 蕴含 WLLN；逆不成立。

**答案**：SLLN 结论已陈述；Bernoulli 情形 $\bar{X}_n\xrightarrow{\text{a.s.}}\boxed{p}$；SLLN 更强，蕴含 WLLN。

**注**：Kolmogorov SLLN 的条件 $E[|X_1|]<\infty$ 是充要条件（不可减弱）：若 $E[|X_1|]=\infty$，则 $\limsup_n|\bar{X}_n|=\infty$ a.s.（Feller 定理）。这也解释了柯西分布（均值不存在）的样本均值不收敛的现象。

---

### D.4.8（Ch.11，多维 CLT）

**题目**：$\mathbf{X}_i$ i.i.d. $\in\mathbb{R}^d$，均值 $\boldsymbol{\mu}$，协方差 $\boldsymbol{\Sigma}$。陈述多维 CLT 并用 delta 方法推导渐近方差。

**思路**：多维 CLT 将一维推广到向量；delta 方法利用一阶泰勒展开，渐近方差由梯度给出。

**解**：

**(a) 多维 CLT**：设 $\mathbf{X}_1,\ldots,\mathbf{X}_n$ i.i.d.，$E[\mathbf{X}_i]=\boldsymbol{\mu}$，$\mathrm{Cov}(\mathbf{X}_i)=\boldsymbol{\Sigma}$（正定），则

$$\sqrt{n}(\bar{\mathbf{X}}-\boldsymbol{\mu})\xrightarrow{d}N(\mathbf{0},\boldsymbol{\Sigma}).$$

等价地，对任意 $\mathbf{a}\in\mathbb{R}^d$，$\mathbf{a}^\top\sqrt{n}(\bar{\mathbf{X}}-\boldsymbol{\mu})\xrightarrow{d}N(0,\mathbf{a}^\top\boldsymbol{\Sigma}\mathbf{a})$（Cramér-Wold）。

**(b) Delta 方法**：设 $g:\mathbb{R}^d\to\mathbb{R}$ 在 $\boldsymbol{\mu}$ 处可微，梯度 $\nabla g(\boldsymbol{\mu})$。泰勒展开：

$$g(\bar{\mathbf{X}})\approx g(\boldsymbol{\mu})+\nabla g(\boldsymbol{\mu})^\top(\bar{\mathbf{X}}-\boldsymbol{\mu}).$$

故 $\sqrt{n}(g(\bar{\mathbf{X}})-g(\boldsymbol{\mu}))\xrightarrow{d}N\!\left(0,\;\nabla g(\boldsymbol{\mu})^\top\boldsymbol{\Sigma}\,\nabla g(\boldsymbol{\mu})\right).$

**(c)** $d=1$，$g(x)=x^2$，$g'(x)=2x$，渐近方差为 $[g'(\mu)]^2\sigma^2=(2\mu)^2\sigma^2=4\mu^2\sigma^2$。

$$\sqrt{n}(\bar{X}^2-\mu^2)\xrightarrow{d}N(0,\,4\mu^2\sigma^2).$$

**答案**：多维 CLT 已陈述；delta 方法渐近方差为 $\nabla g^\top\boldsymbol{\Sigma}\nabla g$；$g(x)=x^2$ 时渐近方差为 $\boxed{4\mu^2\sigma^2}$。

**常见 delta 方法应用**：

| 变换 $g(\mu)$ | $g'(\mu)$ | 渐近方差 |
|---------------|-----------|----------|
| $1/\mu$ | $-1/\mu^2$ | $\sigma^2/\mu^4$ |
| $\sqrt{\mu}$ | $1/(2\sqrt{\mu})$ | $\sigma^2/(4\mu)$ |
| $\log\mu$ | $1/\mu$ | $\sigma^2/\mu^2$ |
| $e^\mu$ | $e^\mu$ | $e^{2\mu}\sigma^2$ |
| $\mu^2$ | $2\mu$ | $4\mu^2\sigma^2$ |

---

### D.4.9（Ch.12，特征函数与弱收敛）

**题目**：陈述 Lévy 连续性定理；用特征函数验证正态情形的 CLT；说明 CF 方法的优势。

**思路**：CF 方法将弱收敛等价为逐点收敛，计算更直接。

**解**：

**(a) Lévy 连续性定理**：$X_n\xrightarrow{d}X$ 当且仅当对所有 $t\in\mathbb{R}$，$\varphi_{X_n}(t)\to\varphi_X(t)$（逐点收敛），且极限 $\varphi(t)$ 在 $t=0$ 处连续。

**(b) 正态情形**：设 $X_i\overset{\text{i.i.d.}}{\sim}N(0,1)$，则 $\bar{X}_n=\frac{1}{n}\sum X_i$，$\bar{X}_n\sqrt{n}=\frac{S_n}{\sqrt{n}}$。

$S_n=\sum_{i=1}^n X_i\sim N(0,n)$，故 $S_n/\sqrt{n}\sim N(0,1)$，其 CF $= e^{-t^2/2}$。直接说明是恒同的，无需极限过程。

一般情形（$X_i$ 均值 0，方差 1）：$\varphi_{X_i}(t)=1-t^2/2+o(t^2)$，则

$$\varphi_{S_n/\sqrt{n}}(t)=\left[\varphi_{X_i}\!\left(\frac{t}{\sqrt{n}}\right)\right]^n=\left(1-\frac{t^2}{2n}+o(1/n)\right)^n\to e^{-t^2/2}.$$

这是标准正态的 CF，由 Lévy 定理得 $S_n/\sqrt{n}\xrightarrow{d}N(0,1)$。

**(c) CF 方法的优势**：CF 处处存在（$|\varphi|\le1$），无需 CDF 的单调性假设；可直接处理重尾分布（如 Cauchy，CF = $e^{-|t|}$）、非对称分布；乘积独立随机变量的 CF 是各 CF 之积，计算极为简洁。

**答案**：Lévy 定理已陈述；CF 验证 CLT：$\left(1-\frac{t^2}{2n}\right)^n\to\boxed{e^{-t^2/2}}$；CF 方法更强大，见(c)。

**注**：Lévy 连续性定理的"连续性"指极限 CF $\varphi(t)$ 在 $t=0$ 处连续（等价于存在对应分布），否则逐点收敛的 CF 未必对应某分布（如狄拉克测度情形）。

---

### D.4.10（Ch.11，样本均值的大样本置信区间）

**题目**：用 CLT 构造 $\mu$ 的近似 95% CI；以样本 $\{2,5,3,7,4\}$ 计算；讨论 $n=5$ 时的严格性。

**思路**：CLT 给出 $(\bar{X}-\mu)/(S/\sqrt{n})\stackrel{\text{approx}}{\sim}N(0,1)$，取 $z_{0.025}=1.96$。

**解**：

**(a)** 由 CLT 和 Slutsky 定理（$S\xrightarrow{P}\sigma$），

$$\frac{\bar{X}-\mu}{S/\sqrt{n}}\xrightarrow{d}N(0,1).$$

近似 95% CI：$\left(\bar{X}-1.96\cdot\dfrac{S}{\sqrt{n}},\;\bar{X}+1.96\cdot\dfrac{S}{\sqrt{n}}\right).$

**(b)** 样本 $\{2,5,3,7,4\}$：$n=5$，$\bar{X}=21/5=4.2$，

$$S^2=\frac{1}{4}\left[(2-4.2)^2+(5-4.2)^2+(3-4.2)^2+(7-4.2)^2+(4-4.2)^2\right]=\frac{4.84+0.64+1.44+7.84+0.04}{4}=\frac{14.8}{4}=3.7.$$

$S=\sqrt{3.7}\approx1.924$，$S/\sqrt{5}\approx0.860$。

95% CI $\approx(4.2-1.96\times0.860,\;4.2+1.96\times0.860)=(2.51,\;5.89)$。

**(c)** $n=5$ 时 CLT 近似不严格：CLT 是渐近结论，$n=5$ 样本量极小，正态近似可能存在较大误差。若总体为正态分布，应精确用 $t(4)$ 分布（$t_{0.025}(4)=2.776$）；若总体非正态，$n=5$ 时正态近似的误差难以量化，置信水平可能偏离 95%。

**答案**：CI 公式已给出；$\bar{X}=4.2$，$S\approx1.924$，95% CI $\approx\boxed{(2.51,\;5.89)}$；$n=5$ 应用 $t(4)$ 更严格。

**注**：用 $t(4)$ 时，$t_{0.025}(4)=2.776$，CI $=(4.2-2.776\times0.860,\;4.2+2.776\times0.860)=(1.81,\;6.59)$，比 CLT CI 更宽，更保守，覆盖概率更有保证。小样本应优先 $t$ 区间。

---

## E 提高题详解（6 题）

---

### E.4.1（Ch.10，强大数定律 + Borel-Cantelli 引理）

**题目**：$X_n$ i.i.d.，$E[X_1^4]<\infty$。证 WLLN，叙述 Borel-Cantelli 引理，利用 B-C 证 SLLN，构造 $E[|X_1|]=\infty$ 的反例。

**思路**：(a) 直接用 Chebyshev；(b) B-C 第一引理靠测度连续性；(c) 用 $E[X_1^4]<\infty$ 控制 $\sum P(|\bar{X}_n-\mu|>\varepsilon)$；(d) Cauchy 分布尾部构造。

**解**：

**(a) WLLN**：$\mathrm{Var}(\bar{X}_n)=\sigma^2/n$（$\sigma^2=E[(X_1-\mu)^2]\le\sqrt{E[X_1^4]}<\infty$）。由 Chebyshev：

$$P(|\bar{X}_n-\mu|\ge\varepsilon)\le\frac{\sigma^2}{n\varepsilon^2}\to0.\quad\square$$

**(b) Borel-Cantelli 引理**：

- **第一引理**：若 $\sum_{n=1}^\infty P(A_n)<\infty$，则 $P(\limsup_n A_n)=0$（几乎处处仅有有限个 $A_n$ 发生）。
- **第二引理**：若 $\{A_n\}$ 独立且 $\sum P(A_n)=\infty$，则 $P(\limsup_n A_n)=1$。

**第一引理证明**：令 $B=\limsup_n A_n=\bigcap_{N=1}^\infty\bigcup_{n\ge N}A_n$，则

$$P(B)\le P\!\left(\bigcup_{n\ge N}A_n\right)\le\sum_{n\ge N}P(A_n)\to0\;\text{（尾级数）}.$$

故 $P(B)=0$。$\square$

**(c) 用 B-C 证 SLLN**（在 $E[X_1^4]<\infty$ 下）：

令 $Y_i=X_i-\mu$（中心化），$\mu_4=E[Y_1^4]<\infty$，$\sigma^2=E[Y_1^2]$。

由 Markov 不等式（四阶矩版本）：

$$P(|\bar{X}_n-\mu|\ge\varepsilon)=P\!\left(\left|\frac{1}{n}\sum_{i=1}^nY_i\right|\ge\varepsilon\right)\le\frac{E\!\left[(\sum_iY_i)^4\right]}{n^4\varepsilon^4}.$$

展开 $(\sum_iY_i)^4$ 后，利用独立性 $E[Y_iY_j]=0$（$i\ne j$）：非零期望只来自

- 指标四元组 $(i,i,i,i)$：$n$ 项，各贡献 $\mu_4$；
- 两对配对 $(i,i,j,j)$（$i\ne j$）：$3\binom{n}{2}$ 项，各贡献 $\sigma^4$；

故 $E\!\left[\left(\sum_iY_i\right)^4\right]=n\mu_4+3n(n-1)\sigma^4\le Cn^2$（$C$ 与 $n$ 无关）。

$$P(|\bar{X}_n-\mu|\ge\varepsilon)\le\frac{Cn^2}{n^4\varepsilon^4}=\frac{C}{n^2\varepsilon^4}.$$

从而 $\sum_{n=1}^\infty P(|\bar{X}_n-\mu|\ge\varepsilon)\le\frac{C}{\varepsilon^4}\sum_{n=1}^\infty\frac{1}{n^2}<\infty$。

由 B-C 第一引理，$P(|\bar{X}_n-\mu|\ge\varepsilon\text{ i.o.})=0$。

对任意 $\varepsilon>0$ 均成立，故 $\bar{X}_n\xrightarrow{\text{a.s.}}\mu$。$\square$

**(d) 反例**：令 $P(X>x)\sim\dfrac{1}{x(\log x)^2}$（$x\to\infty$），则 $E[|X|]=\int_e^\infty P(|X|>x)\,dx=\int_e^\infty\frac{dx}{x(\log x)^2}=\Big[-\frac{1}{\log x}\Big]_e^\infty=1<\infty$（此例 $E[|X|]<\infty$ 仍然）。

> 构造 $E[|X|]=\infty$ 的经典反例：$P(X>x)\sim 1/(x\log x)$，则 $E[|X|]=\int_e^\infty\frac{dx}{x\log x}=\infty$，此时 SLLN 失效，$\bar{X}_n$ 可能不收敛或收敛到 $\pm\infty$。

在深度学习中，梯度爆炸对应梯度分布尾部过重（$E[|g|]=\infty$），导致 SGD 均值估计失效；梯度裁剪（Gradient Clipping）本质是截断以确保矩存在。

**答案**：WLLN、Borel-Cantelli 引理均已证明；$E[X_1^4]<\infty$ 下四阶矩衰减保证级数收敛，SLLN 成立；$E[|X|]=\infty$ 时 SLLN 可失效。

**技术细节补充**（四阶矩展开）：$(\sum_iY_i)^4$展开有 $n^4$ 项，按指标重复情况分类：

| 类型 | 指标结构 | 期望值 | 项数 |
|------|----------|--------|------|
| 全重合 | $(i,i,i,i)$ | $\mu_4$ | $n$ |
| 两对配对 | $(i,i,j,j)$，$i\ne j$ | $\sigma^4$ | $3n(n-1)$ |
| 其余 | 含奇数次指标 | $0$（由独立+零均值） | — |

故 $E\!\left[\left(\sum Y_i\right)^4\right]=n\mu_4+3n(n-1)\sigma^4=O(n^2)$，从而 $P$-界为 $O(1/n^2)$，级数 $\sum1/n^2<\infty$。

---

### E.4.2（Ch.11+Ch.10，CLT + Berry-Esseen + 非 i.i.d. 推广）

**题目**：独立非同分布序列，叙述 Lindeberg 条件和 CLT；陈述 Berry-Esseen；分析 SGD 梯度噪声的 CLT 行为。

**思路**：Lindeberg 条件是非 i.i.d. CLT 的关键充分条件；Berry-Esseen 给出收敛速率；SGD 可类比。

**解**：

**(a) Lindeberg 条件与 CLT**：

令 $S_n=\sum_{i=1}^n(X_i-\mu_i)$，$B_n^2=\sum_{i=1}^n\sigma_i^2$。**Lindeberg 条件**：对任意 $\varepsilon>0$，

$$\frac{1}{B_n^2}\sum_{i=1}^nE\!\left[(X_i-\mu_i)^2\mathbf{1}_{|X_i-\mu_i|>\varepsilon B_n}\right]\to0\quad(n\to\infty).$$

**Lindeberg CLT**：若 Lindeberg 条件成立，则 $S_n/B_n\xrightarrow{d}N(0,1)$。

证明框架（特征函数法）：

1. 将 $X_i$ 截断：$X_i^{(\varepsilon)}=X_i\mathbf{1}_{|X_i-\mu_i|\le\varepsilon B_n}$，令 $\tilde{X}_i=X_i^{(\varepsilon)}-E[X_i^{(\varepsilon)}]$。
2. 有界变量的 CF 满足 $\log\varphi_{\tilde{X}_i/B_n}(t)\approx-t^2\sigma_i^{(\varepsilon)2}/(2B_n^2)$，故 $\varphi_{S_n^{(\varepsilon)}/B_n}(t)\to e^{-t^2/2}$。
3. Lindeberg 条件保证截断误差 $\|S_n-S_n^{(\varepsilon)}\|_2^2/B_n^2\to0$，故截断前后 CF 差趋 0，得 $\varphi_{S_n/B_n}(t)\to e^{-t^2/2}$。

**(b) Berry-Esseen 定理**：存在常数 $C\le0.4785$，使得

$$\sup_x\left|P\!\left(\frac{S_n}{B_n}\le x\right)-\Phi(x)\right|\le\frac{C\sum_{i=1}^n\rho_i}{B_n^3}.$$

其中 $\rho_i=E[|X_i-\mu_i|^3]$。i.i.d. 情形退化为 $C\rho/\sqrt{n}$。

**(c) SGD 梯度噪声**：批大小 $n$，第 $i$ 个样本随机梯度 $g_i\approx\nabla L(\theta)+\xi_i$（$\xi_i$ 近似独立，均值 0，方差 $\sigma_g^2$）。

CLT 给出批梯度噪声 $\bar{g}-\nabla L\approx N(0,\sigma_g^2/n)$：批大小越大，梯度估计越精确（方差 $\propto1/n$）；Berry-Esseen 上界 $\propto1/\sqrt{n}$，批大小从 16 增至 256 使近似质量提升 4 倍。

**(d) Lindeberg 条件在非平稳流中失效**：当数据分布偏移时，$\sigma_i^2$ 可能集中在某些时刻（如概念漂移），导致 $\max_i\sigma_i^2/B_n^2\not\to0$，Lindeberg 条件违反，CLT 不适用。

Adam/AdaGrad 用历史梯度二阶矩 $v_t=\beta v_{t-1}+(1-\beta)g_t^2$ 自适应调整学习率，隐式地对每维度标准化，相当于将非同方差情形转化为近似同方差，有利于恢复 Lindeberg 条件的精神。

**答案**：Lindeberg 条件和 CLT 已陈述；Berry-Esseen 上界为 $C\sum\rho_i/B_n^3$；SGD 批大小增大则梯度分布更接近正态，近似质量提升。

> ⚠️ Lindeberg 条件不要求各 $X_i$ 同分布，但要求没有单一 $X_i$ 主导总方差（"均匀小性"）。i.i.d. 时 Lindeberg 条件等价于 $E[X_1^2]<\infty$（自动满足），故 i.i.d. CLT 是特例。

---

### E.4.3（Ch.11+Ch.12，Delta 方法 + Bootstrap）

**题目**：$\sqrt{n}(\hat{\theta}_n-\theta)\xrightarrow{d}N(0,\sigma^2)$。证 Delta 方法；推导 logit 变换渐近方差；Bootstrap 一致性；Deep Ensemble 分析。

**思路**：Delta 方法 = 泰勒展开 + Slutsky；Bootstrap 用经验分布代替真实分布。

**解**：

**(a) Delta 方法证明**：在 $\theta$ 处泰勒展开：

$$g(\hat{\theta}_n)=g(\theta)+g'(\theta)(\hat{\theta}_n-\theta)+R_n,\quad R_n=O((\hat{\theta}_n-\theta)^2).$$

由于 $\hat{\theta}_n-\theta=O_P(1/\sqrt{n})$，$R_n=O_P(1/n)$，故 $\sqrt{n}R_n\xrightarrow{P}0$。

$$\sqrt{n}(g(\hat{\theta}_n)-g(\theta))=g'(\theta)\cdot\sqrt{n}(\hat{\theta}_n-\theta)+\sqrt{n}R_n\xrightarrow{d}g'(\theta)\cdot N(0,\sigma^2)=N(0,[g'(\theta)]^2\sigma^2).\quad\square$$

（最后一步用 Slutsky 定理：$\sqrt{n}R_n\xrightarrow{P}0$ 与 $\sqrt{n}(\hat\theta_n-\theta)\xrightarrow{d}N(0,\sigma^2)$ 叠加。）

**(b) Logit 变换**：$\hat{p}_n=\bar{X}_n$，$\sigma^2=p(1-p)$，$g(p)=\log\dfrac{p}{1-p}$，$g'(p)=\dfrac{1}{p(1-p)}$。

$$\sqrt{n}(g(\hat{p}_n)-g(p))\xrightarrow{d}N\!\left(0,\frac{p(1-p)}{[p(1-p)]^2}\right)=N\!\left(0,\frac{1}{p(1-p)}\right).$$

logit 尺度下的 CI（再逆变换回 $[0,1]$）在 $p$ 接近 0 或 1 时比直接 CI 更稳定，因为 logit 将有界参数空间映射到实轴，近似更准确。

**(c) Bootstrap 一致性框架**：

步骤：① 从原样本 $X_1,\ldots,X_n$ 有放回抽样得 $X_1^*,\ldots,X_n^*\sim\hat{F}_n$；② 计算 Bootstrap 统计量 $T_n^*=T(X_1^*,\ldots,X_n^*)$；③ 用 $T_n^*$ 的分布估计 $T_n$ 的分布。

关键结论（Bickel-Freedman 1981）：在正则条件（如 $E[|X_1|^p]<\infty$ 对足够大 $p$）下，

$$\sup_x\left|P^*(T_n^*\le x)-P(T_n\le x)\right|\xrightarrow{P}0,$$

即 Bootstrap 分布（条件于数据）依概率一致逼近真实抽样分布。证明直觉：$\hat{F}_n\xrightarrow{\text{a.s.}}F$（Glivenko-Cantelli），从 $\hat{F}_n$ 抽样的统计量行为趋近于从 $F$ 抽样的行为。

**(d) Deep Ensemble 分析**：Deep Ensemble 训练 $M$ 个独立网络（随机初始化），预测时取平均——类似于对参数的 Bootstrap 重采样。

理论联系：若各网络参数 $\hat{\theta}^{(m)}\stackrel{\text{approx}}{\sim}$ 后验分布（变分意义），则集成均值近似贝叶斯预测均值。局限：初始化不保证参数的多样性；各网络损失面拓扑结构相同，集成方差可能被低估。

与 Laplace 近似对比：Laplace 近似假设后验为正态（在 MAP 处做二阶 Taylor），计算代价小但依赖单点近似，不捕捉多模态；Deep Ensemble 更灵活但代价为 $M$ 倍训练。

**答案**：Delta 方法已证明；logit 渐近方差为 $\boxed{1/[p(1-p)]}$；Bootstrap 一致性依赖 Glivenko-Cantelli；Deep Ensemble 是隐式 Bootstrap 但存在多样性局限。

> ⚠️ Delta 方法要求 $g'(\theta)\ne0$（否则需要二阶 delta 方法）。若 $g'(\theta)=0$，渐近分布变为 $\tfrac{1}{2}g''(\theta)\sigma^2\cdot\chi^2(1)$。例如估计 $\theta^2$ 在 $\theta=0$ 处：直接 delta 方法失效，需二阶展开。

---

### E.4.4（Ch.12，四种收敛类型的关系）

**题目**：严格定义四种收敛；构造 a.s. $\not\Rightarrow L^1$ 的反例（打字机序列）；构造 in prob. $\not\Rightarrow$ a.s. 的反例；分析 SGD 参数收敛性质。

**思路**：四种收敛各有定义，关系图是核心；反例是辨析要点。

**解**：

**(a) 四种收敛定义与关系**：

| 收敛类型 | 记号 | 定义 |
|----------|------|------|
| $L^p$ 收敛 | $X_n\xrightarrow{L^p}X$ | $E[\vert X_n-X\vert^p]\to0$ |
| 几乎处处收敛 | $X_n\xrightarrow{\text{a.s.}}X$ | $P(\lim_n X_n=X)=1$ |
| 依概率收敛 | $X_n\xrightarrow{P}X$ | $\forall\varepsilon>0,\;P(\vert X_n-X\vert\ge\varepsilon)\to0$ |
| 依分布收敛 | $X_n\xrightarrow{d}X$ | $F_{X_n}(x)\to F_X(x)$（$F_X$ 连续点处） |

蕴含关系（强 $\to$ 弱）：

$$L^p\;\Rightarrow\;\text{in prob.},\quad \text{a.s.}\;\Rightarrow\;\text{in prob.},\quad\text{in prob.}\;\Rightarrow\;\text{in dist.}$$

逆向均不成立（需反例）；a.s. 与 $L^p$ 之间无一般蕴含关系——需加**一致可积**（UI）才能由 in prob. 或 a.s. 得到 $L^1$；需加**有界性**才能由 a.s. 得到 $L^p$。

特殊情况：若 $X_n\xrightarrow{d}c$（常数），则 $X_n\xrightarrow{P}c$（in dist. $\Rightarrow$ in prob.，仅限极限为常数）。

**(b) a.s. $\not\Rightarrow L^1$（打字机序列）**：在 $[0,1]$ 上取 $X_n=n\cdot\mathbf{1}_{[0,1/n]}$。

$X_n(\omega)\to0$ 对所有 $\omega\in(0,1]$（当 $n>1/\omega$ 时），故 $X_n\xrightarrow{\text{a.s.}}0$。

但 $E[|X_n-0|]=E[X_n]=n\cdot(1/n)=1\not\to0$，不是 $L^1$ 收敛。

**(c) in prob. $\not\Rightarrow$ a.s.**：在 $([0,1],\mathcal{B},\lambda)$ 上，走马灯序列按如下规则排列：

$$X_1=\mathbf{1}_{[0,1]},\quad X_2=\mathbf{1}_{[0,1/2]},\;X_3=\mathbf{1}_{[1/2,1]},\quad X_4=\mathbf{1}_{[0,1/4]},\;X_5=\mathbf{1}_{[1/4,1/2]},\;\ldots$$

第 $j$ 层共有 $2^j$ 个区间，每个长度 $2^{-j}$，故 $P(X_n=1)=2^{-j}\to0$，$X_n\xrightarrow{P}0$。

但对任意 $\omega\in[0,1]$，每层必有一个区间覆盖 $\omega$，故 $X_n(\omega)=1$ 无穷次发生，$\limsup_n X_n(\omega)=1\ne0$，a.s. 不收敛到 0。

**(d) SGD 收敛性质**：

- **强凸损失**：SGD 参数序列 $\boldsymbol{\theta}_n$ 以 $O(1/n)$ 速率**依概率**收敛到最优解 $\boldsymbol{\theta}^*$；在强凸 + 有界梯度条件下也可证 a.s. 收敛。
- **非凸损失**：一般只能证明梯度范数 $\|\nabla L(\boldsymbol{\theta}_n)\|\xrightarrow{P}0$（即收敛到平稳点），不能保证全局最优。
- **逃离鞍点**：SGD 的随机性（梯度噪声 $\sim N(0,\Sigma)$）使其在鞍点处受到随机扰动，以正概率逃离，这是 SGD 比梯度下降更有优势的地方（信息论和微分几何角度的解释均支持）。

**答案**：四种收敛定义及关系图已给出；打字机序列为 a.s. 不蕴含 $L^1$ 的反例；走马灯序列为 in prob. 不蕴含 a.s. 的反例；SGD 在凸情形依概率收敛，非凸情形仅保证收敛到平稳点。

> ⚠️ 关键记忆：依分布收敛若极限为常数 $c$，则等价于依概率收敛（Portmanteau 定理）。否则依分布收敛是所有类型中最弱的，只刻画分布的信息，不涉及各 $X_n$ 与极限的路径关系。

---

### E.4.5（Ch.10+Ch.11+Ch.12，泛函 CLT + Donsker 定理）

**题目**：$W_n(t)=S_{\lfloor nt\rfloor}/(\sigma\sqrt{n})$。证有限维 CLT；陈述 Donsker 定理；推导 KS 统计量极限分布；分析位置编码与布朗运动的联系。

**思路**：Donsker 定理将 CLT 推广到函数空间 $C[0,1]$；KS 统计量与布朗桥相关。

**解**：

**(a) 有限维分布的 CLT**：固定 $t\in(0,1]$，$\lfloor nt\rfloor\to\infty$。

$$W_n(t)=\frac{1}{\sigma\sqrt{n}}\sum_{i=1}^{\lfloor nt\rfloor}X_i=\sqrt{\frac{\lfloor nt\rfloor}{n}}\cdot\frac{S_{\lfloor nt\rfloor}}{\sigma\sqrt{\lfloor nt\rfloor}}.$$

由 CLT，$S_{\lfloor nt\rfloor}/(\sigma\sqrt{\lfloor nt\rfloor})\xrightarrow{d}N(0,1)$；$\lfloor nt\rfloor/n\to t$；故 $W_n(t)\xrightarrow{d}N(0,t)$。$\square$

**(b) Donsker 定理（泛函 CLT）**：设 $X_i$ i.i.d.，均值 0，方差 $\sigma^2$，定义折线过程 $W_n\in C[0,1]$。则

$$W_n\Rightarrow W\quad\text{（在 }C[0,1]\text{ 上弱收敛）},$$

其中 $W$ 为标准布朗运动（Wiener 过程）：$W(0)=0$，增量独立，$W(t)-W(s)\sim N(0,t-s)$，路径几乎处处连续。

**(c) Kolmogorov-Smirnov 统计量**：$D_n=\sup_x|\hat{F}_n(x)-F(x)|$。

将 $\hat{F}_n(x)-F(x)$ 表示为经验过程，经适当标准化后可与 $W_n$ 建立联系。由 Donsker 定理，标准化经验过程 $\sqrt{n}(\hat{F}_n-F)\Rightarrow B$（布朗桥），其中布朗桥 $B(t)=W(t)-tW(1)$（$t\in[0,1]$）。

故 $\sqrt{n}D_n\xrightarrow{d}\sup_{t\in[0,1]}|B(t)|$，其分布为 Kolmogorov 分布（可用级数表示）。

**(d) 位置编码与布朗运动**：若输入序列 $\{e_1,\ldots,e_T\}$（token embeddings）近似独立同分布，均值 0，方差 $\sigma^2$，则前缀和 $E_t=\sum_{i=1}^t e_i$ 经标准化后趋向布朗运动 $W(t/T)$。

Transformer 的正弦位置编码 $\mathrm{PE}(pos,2i)=\sin(pos/10000^{2i/d})$ 提供了显式的 $t$ 信息，其频率 $1/10000^{2i/d}$ 从低到高覆盖不同时间尺度——类比于布朗运动的 Karhunen-Loève 展开（用正弦函数作正交基），这与布朗运动的谱密度 $S(f)\propto1/f^2$（低频主导）在精神上一致。

**答案**：有限维 CLT 已证；Donsker 定理已陈述；$\sqrt{n}D_n\xrightarrow{d}\sup_t\vert B(t)\vert$（布朗桥上确界）；位置编码与布朗运动谱密度有内在频率结构联系。

**注**：KS 分布的精确表达式为 $P(\sup_t|B(t)|\le x)=1-2\sum_{k=1}^\infty(-1)^{k-1}e^{-2k^2x^2}$（Kolmogorov 公式）。常用分位数：$x_{0.05}\approx1.358$（即 $P(\sqrt{n}D_n\le1.358)\approx0.95$）。Donsker 定理的证明依赖 Prokhorov 定理（紧性准则）和有限维分布的收敛性。

---

### E.4.6（Ch.11+Ch.12，稳定分布 + 广义 CLT）

**题目**：重尾分布（$\alpha$-稳定），证 $\alpha<2$ 时方差无穷大；陈述广义 CLT；验证柯西稳定性；分析 Hill 估计量。

**思路**：$P(|X|>x)\sim L(x)x^{-\alpha}$ 时尾部积分发散给出 $\mathrm{Var}=\infty$；稳定分布由特征函数刻画。

**解**：

**(a) $\alpha<2$ 时 $\mathrm{Var}(X)=\infty$**：

$$E[X^2]=\int_0^\infty P(X^2>t)\,dt=2\int_0^\infty xP(|X|>x)\,dx\ge 2\int_1^\infty x\cdot\frac{c}{x^\alpha}\,dx=2c\int_1^\infty x^{1-\alpha}\,dx.$$

当 $\alpha<2$ 时，$1-\alpha>-1$，上述积分 $=\Big[\dfrac{x^{2-\alpha}}{2-\alpha}\Big]_1^\infty=+\infty$。故 $E[X^2]=\infty$，$\mathrm{Var}(X)=\infty$，经典 CLT 不适用。$\square$

**(b) 广义中心极限定理（GCLT）**：设 $X_i$ i.i.d.，尾指数 $\alpha\in(0,2)$。存在规范化常数 $a_n\sim n^{1/\alpha}$ 和中心化常数 $b_n$，使得

$$(S_n-b_n)/a_n\xrightarrow{d}Z_\alpha,$$

其中 $Z_\alpha$ 为 $\alpha$-稳定分布。其特征函数为

$$\varphi_{Z_\alpha}(t)=\exp\!\left\{-c|t|^\alpha\left(1-i\beta\,\mathrm{sgn}(t)\tan\frac{\pi\alpha}{2}\right)+i\delta t\right\},\quad\alpha\ne1,$$

参数 $c>0$（尺度），$\beta\in[-1,1]$（偏度），$\delta\in\mathbb{R}$（位置）。$\alpha=2$ 退化为正态分布。

**(c) 对称柯西分布（$\alpha=1$）的稳定性**：对称柯西的 CF 为 $\varphi(t)=e^{-|t|}$。

$n$ 个独立柯西之和 $S_n$ 的 CF：$\varphi_{S_n}(t)=[e^{-|t|}]^n=e^{-n|t|}$，故 $S_n\sim\mathrm{Cauchy}(0,n)$。

$S_n/n$ 的 CF：$\varphi_{S_n/n}(t)=\varphi_{S_n}(t/n)=e^{-|t|}$，即 $S_n/n\sim\mathrm{Cauchy}(0,1)$——与原分布相同。

对比经典 CLT：正态情形 $S_n/\sqrt{n}\to N(0,\sigma^2)$，归一化为 $\sqrt{n}$；柯西情形需除以 $n$（$\alpha=1$，$a_n\sim n$），且极限仍为柯西——平均无法"改善"精度，无法用样本均值估计柯西的"中心"（均值不存在）。

**(d) Hill 估计量**：Hill 估计量

$$\hat{\alpha}_k^{-1}=\frac{1}{k}\sum_{i=1}^k\log X_{(n-i+1)}-\log X_{(n-k)},$$

**理论依据**：超过高阈值 $u=X_{(n-k)}$ 的超额量 $X-u\mid X>u$ 在 Pareto 尾（$P(X>x)\sim cx^{-\alpha}$）下近似服从指数分布：

$$P(X-u>y\mid X>u)\approx e^{-y/(\alpha^{-1}\cdot u)}\cdot\frac{x^\alpha}{(x+y)^\alpha}\approx e^{-\alpha y/u}.$$

取对数比值：$\log(X_{(n-i+1)}/X_{(n-k)})\approx\mathrm{Exp}(\alpha)$，其均值为 $1/\alpha$。Hill 估计量即为 $k$ 个这样的对数超额量的样本均值，由 SLLN，当 $k,n\to\infty$（$k/n\to0$）时，$\hat{\alpha}_k^{-1}\xrightarrow{P}\alpha^{-1}$。

在训练损失的随机游走中，若梯度噪声重尾（$\alpha<2$），则 SGD 步长 $\eta g_t$ 的二阶矩无穷大，理论上的 CLT 型收敛不适用；实践中表现为偶发性的"梯度爆炸"。梯度裁剪本质是将分布截断为有限方差分布，使 CLT 可再次适用。

**答案**：(a) $\alpha<2$ 时 $\mathrm{Var}=\infty$ 已证；(b) GCLT 特征函数形式已给出；(c) 柯西 CF $e^{-n|t|}$，$S_n/n\sim\mathrm{Cauchy}$，归一化为 $n$ 而非 $\sqrt{n}$；(d) Hill 估计量基于超额量指数近似，重尾梯度导致 CLT 失效需梯度裁剪。

> ⚠️ $\alpha$-稳定分布具有**稳定性**：若 $X_1,X_2$ i.i.d. 同为 $\alpha$-稳定分布，则 $aX_1+bX_2$ 仍为 $\alpha$-稳定（仅差尺度变换）。$\alpha=2$ 时即正态分布的稳定性；$\alpha=1$ 时即柯西的稳定性。金融中的 Lévy 稳定分布建模收益率尾部比高斯模型更真实。

---

---

## 知识点速查

### 核心不等式

| 不等式 | 形式 | 条件 |
|--------|------|------|
| Markov | $P(X\ge a)\le E[X]/a$ | $X\ge0$，$a>0$ |
| Chebyshev | $P(\vert X-\mu\vert\ge k\sigma)\le 1/k^2$ | $\mathrm{Var}(X)<\infty$ |
| Chebyshev（$\bar{X}_n$） | $P(\vert\bar{X}_n-\mu\vert\ge\varepsilon)\le\sigma^2/(n\varepsilon^2)$ | i.i.d.，有限方差 |

### 收敛类型蕴含关系

$$L^p\;\underset{}{\Rightarrow}\;\text{in prob.}\;\underset{}{\Rightarrow}\;\text{in dist.}\;\underset{}{\Leftarrow}\;\text{（极限为常数时等价）}$$

$$\text{a.s.}\;\underset{}{\Rightarrow}\;\text{in prob.}\quad\quad \text{in prob.}\;\underset{\text{子列}}{\Rightarrow}\;\text{a.s. 子列}$$

### 大数定律对比

| 定理 | 条件 | 结论 |
|------|------|------|
| WLLN（Chebyshev） | $\mathrm{Var}(X_1)<\infty$ | $\bar{X}_n\xrightarrow{P}\mu$ |
| WLLN（Feller） | $E[\vert X_1\vert]<\infty$ | $\bar{X}_n\xrightarrow{P}\mu$ |
| SLLN（Kolmogorov） | $E[\vert X_1\vert]<\infty$ | $\bar{X}_n\xrightarrow{\text{a.s.}}\mu$ |

### CLT 家族

| 定理 | 条件 | 结论 |
|------|------|------|
| CLT（i.i.d.） | $\sigma^2<\infty$ | $S_n/(\sigma\sqrt{n})\xrightarrow{d}N(0,1)$ |
| Berry-Esseen | $E[\vert X\vert^3]<\infty$ | 误差 $\le C\rho/\sqrt{n}$ |
| Lindeberg CLT | Lindeberg 条件 | $S_n/B_n\xrightarrow{d}N(0,1)$（非 i.i.d.） |
| GCLT | $\alpha$-稳定尾，$\alpha\in(0,2)$ | $S_n/n^{1/\alpha}\xrightarrow{d}Z_\alpha$ |
| Donsker | $\sigma^2<\infty$ | $W_n\Rightarrow W$（函数空间） |

---

*文件结束。共 C 基础 8 题 + D 中等 10 题 + E 提高 6 题 = 24 题。*
