# 第12章：收敛性理论

## 学习目标

学完本章后，你将能够：

- 掌握四种收敛概念（依概率、依分布、几乎必然、$L^p$）的严格定义，并理解各自的直觉含义
- 理解"几乎必然收敛"（强收敛）与"依概率收敛"（弱收敛）的本质区别，能用反例说明二者不等价
- 熟记并证明四种收敛之间的蕴含关系图，掌握各方向成立或不成立的条件
- 运用连续映射定理、Slutsky 定理和 $\delta$ 方法处理依分布收敛的复合运算
- 将收敛理论应用于深度学习模型的训练收敛性分析与 PAC 学习理论的样本复杂度推导

---

## 12.1 依概率收敛

### 直觉引入

想象你用一把有随机误差的尺子反复量同一根棍子的长度。每次测量结果 $X_n$ 都有波动，但随着测量技术改进，出现"大误差"的可能性越来越小。这种"犯大错的概率趋于零"的收敛方式，就是**依概率收敛**。

### 严格定义

**定义 12.1（依概率收敛）**
设 $X_1, X_2, \ldots$ 和 $X$ 是定义在同一概率空间 $(\Omega, \mathcal{F}, P)$ 上的随机变量。若对任意 $\varepsilon > 0$，

$$
\boxed{P\!\left(|X_n - X| > \varepsilon\right) \to 0 \quad (n \to \infty)}
$$

则称 $X_n$ **依概率收敛**（converge in probability）到 $X$，记作

$$
X_n \xrightarrow{P} X \quad \text{或} \quad \text{plim}_{n\to\infty} X_n = X
$$

### 等价刻画

以下三条陈述等价：

1. $X_n \xrightarrow{P} X$
2. 对任意 $\varepsilon > 0$，$P(|X_n - X| \leq \varepsilon) \to 1$
3. 对任意 $\varepsilon > 0$，$\delta > 0$，存在 $N$ 使得 $n > N$ 时 $P(|X_n - X| > \varepsilon) < \delta$

**注意**："依概率收敛到 $X$"不要求 $X_n(\omega) \to X(\omega)$ 对每个样本点 $\omega$ 成立，只要求违反此收敛的样本点集合的概率趋于零。

### 基本性质

**命题 12.1**（依概率收敛的运算规则）若 $X_n \xrightarrow{P} X$，$Y_n \xrightarrow{P} Y$，则：

1. $aX_n + bY_n \xrightarrow{P} aX + bY$（线性性）
2. $X_n Y_n \xrightarrow{P} XY$（乘积性）
3. 若 $g$ 连续，则 $g(X_n) \xrightarrow{P} g(X)$（连续映射定理）
4. 若 $P(Y = 0) = 0$，则 $X_n / Y_n \xrightarrow{P} X/Y$

**证明（性质 1）**：对任意 $\varepsilon > 0$，

$$
\{|aX_n + bY_n - aX - bY| > \varepsilon\} \subseteq \left\{|X_n - X| > \frac{\varepsilon}{2|a|}\right\} \cup \left\{|Y_n - Y| > \frac{\varepsilon}{2|b|}\right\}
$$

（当 $a, b \neq 0$ 时，$a = 0$ 时更简单）由次可加性，

$$
P(|aX_n + bY_n - aX - bY| > \varepsilon) \leq P\!\left(|X_n - X| > \frac{\varepsilon}{2|a|}\right) + P\!\left(|Y_n - Y| > \frac{\varepsilon}{2|b|}\right) \to 0
$$

$\blacksquare$

### Markov 不等式与 Chebyshev 不等式

依概率收敛常通过以下工具建立：

**Markov 不等式**：对非负随机变量 $Y$ 和 $a > 0$，

$$
P(Y \geq a) \leq \frac{\mathbb{E}[Y]}{a}
$$

**Chebyshev 不等式**：对任意随机变量 $X$ 和 $\varepsilon > 0$，

$$
P(|X - \mu| \geq \varepsilon) \leq \frac{\operatorname{Var}(X)}{\varepsilon^2}
$$

**示例**：设 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$（i.i.d.，均值 $\mu$，方差 $\sigma^2 < \infty$），则

$$
P\!\left(|\bar{X}_n - \mu| \geq \varepsilon\right) \leq \frac{\sigma^2}{n\varepsilon^2} \to 0
$$

这正是**弱大数定律**的一种证明：$\bar{X}_n \xrightarrow{P} \mu$。

### 依概率收敛的局限性

依概率收敛**不保证**：
- 每条样本路径都收敛（可能有"游荡"行为）
- 期望的收敛：$X_n \xrightarrow{P} 0$ 不能推出 $\mathbb{E}[X_n] \to 0$

**反例**（期望不收敛）：令 $P(X_n = n) = 1/n$，$P(X_n = 0) = 1 - 1/n$。则 $X_n \xrightarrow{P} 0$，但 $\mathbb{E}[X_n] = 1$ 不趋于零。

---

## 12.2 依分布收敛（弱收敛）

### 定义

**定义 12.2（依分布收敛）**
设 $X_1, X_2, \ldots$ 和 $X$ 是随机变量，$F_n$ 和 $F$ 分别为其分布函数。若在 $F$ 的每个**连续点** $x$ 处均有

$$
\boxed{F_n(x) \to F(x) \quad (n \to \infty)}
$$

则称 $X_n$ **依分布收敛**（converge in distribution）到 $X$，记作

$$
X_n \xrightarrow{d} X \quad \text{或} \quad X_n \rightsquigarrow X
$$

也称为**弱收敛**（weak convergence）。

**为何只要求连续点处收敛**：分布函数 $F$ 可能在某些点有跳跃（如离散分布），在跳跃点处 $F_n(x) \to F(x)$ 可能不成立即使整体分布收敛，因此只在连续点处要求收敛是自然的选择。

### 与特征函数的等价刻画

**定理 12.1（Lévy 连续性定理）**
$X_n \xrightarrow{d} X$ 当且仅当对每个 $t \in \mathbb{R}$，$X_n$ 的特征函数 $\varphi_n(t) \to \varphi(t)$，且 $\varphi$ 在 $t = 0$ 处连续。

这是中心极限定理证明的核心工具（见第 11 章）。

### 依分布收敛的等价刻画（Portmanteau 定理）

**定理 12.2（Portmanteau 定理）**
以下陈述等价：

1. $X_n \xrightarrow{d} X$
2. 对所有有界连续函数 $f$：$\mathbb{E}[f(X_n)] \to \mathbb{E}[f(X)]$
3. 对所有闭集 $F$：$\limsup_{n} P(X_n \in F) \leq P(X \in F)$
4. 对所有开集 $G$：$\liminf_{n} P(X_n \in G) \geq P(X \in G)$
5. 对所有 Borel 集 $B$ 满足 $P(X \in \partial B) = 0$：$P(X_n \in B) \to P(X \in B)$

**直觉**：条件 2 说明依分布收敛等价于所有"有界连续测试函数"的期望收敛，这是弱收敛名称的由来——它是函数空间中的弱拓扑意义下的收敛。

### 依分布收敛的重要性质

**定理 12.3（连续映射定理，CMT）**
若 $X_n \xrightarrow{d} X$，$g$ 是连续函数（或更一般地，$P(X \in \text{Disc}(g)) = 0$），则

$$
g(X_n) \xrightarrow{d} g(X)
$$

**定理 12.4（Slutsky 定理）**
若 $X_n \xrightarrow{d} X$，$Y_n \xrightarrow{P} c$（常数），则：

$$
X_n + Y_n \xrightarrow{d} X + c, \qquad X_n Y_n \xrightarrow{d} cX
$$

**注意**：若 $Y_n \xrightarrow{d} Y$（$Y$ 非常数），则一般不能得出 $X_n + Y_n \xrightarrow{d} X + Y$（联合分布未必收敛）。

**定理 12.5（$\delta$ 方法）**
设 $\sqrt{n}(X_n - \theta) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$，$g$ 在 $\theta$ 处可微且 $g'(\theta) \neq 0$，则

$$
\boxed{\sqrt{n}\bigl(g(X_n) - g(\theta)\bigr) \xrightarrow{d} \mathcal{N}\!\left(0,\, [g'(\theta)]^2 \sigma^2\right)}
$$

**证明思路**：Taylor 展开 $g(X_n) \approx g(\theta) + g'(\theta)(X_n - \theta)$，再用 Slutsky 定理。$\blacksquare$

**$\delta$ 方法示例**：设 $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$，取 $g(x) = e^x$，则

$$
\sqrt{n}\bigl(e^{\bar{X}_n} - e^\mu\bigr) \xrightarrow{d} \mathcal{N}\!\left(0,\, e^{2\mu}\sigma^2\right)
$$

### 依分布收敛只是"分布层面"的收敛

依分布收敛是最弱的一种随机收敛，它**不要求** $X_n$ 和 $X$ 定义在同一概率空间上，也不要求 $X_n(\omega) - X(\omega) \to 0$。

**极端例子**：设 $X_n \equiv X \sim \mathcal{N}(0,1)$，但令 $Y_n = -X$（与 $X$ 反号）。则 $Y_n \xrightarrow{d} \mathcal{N}(0,1)$（依分布等于 $X$），但 $Y_n - X_n = -2X$ 并不趋于零。这说明"依分布相同"不等于"轨道上接近"。

---

## 12.3 几乎必然收敛（强收敛）

### 定义

**定义 12.3（几乎必然收敛）**
若

$$
\boxed{P\!\left(\lim_{n \to \infty} X_n = X\right) = P\!\left(\{\omega : X_n(\omega) \to X(\omega)\}\right) = 1}
$$

则称 $X_n$ **几乎必然收敛**（converge almost surely）到 $X$，记作

$$
X_n \xrightarrow{a.s.} X \quad \text{或} \quad X_n \to X \quad \text{a.s.}
$$

也称**以概率 1 收敛**（converge with probability one）或**强收敛**。

### 几乎必然收敛的直觉

定义在概率空间 $\Omega$ 上，几乎必然收敛要求：**除了一个零概率集 $N$ 以外**，对每个样本点 $\omega \notin N$，数列 $X_1(\omega), X_2(\omega), \ldots$ 像普通实数数列一样收敛到 $X(\omega)$。

用 $\limsup$ 和 $\liminf$ 改写：

$$
X_n \xrightarrow{a.s.} X \iff P\!\left(\limsup_{n\to\infty} |X_n - X| > 0\right) = 0
$$

等价地（用上极限集合）：对任意 $\varepsilon > 0$，

$$
X_n \xrightarrow{a.s.} X \iff P\!\left(\limsup_{n\to\infty} \{|X_n - X| > \varepsilon\}\right) = 0
$$

即

$$
P\!\left(\bigcap_{N=1}^\infty \bigcup_{n=N}^\infty \{|X_n - X| > \varepsilon\}\right) = 0
$$

### 几乎必然收敛的等价条件

**命题 12.2**
$X_n \xrightarrow{a.s.} X$ 当且仅当对任意 $\varepsilon > 0$，

$$
P\!\left(\sup_{k \geq n} |X_k - X| > \varepsilon\right) \to 0 \quad (n \to \infty)
$$

这个条件说明：$X_n$ 之后的所有项都接近 $X$ 的概率趋于 1——这比依概率收敛更强，后者只要求第 $n$ 项接近 $X$。

### Borel-Cantelli 引理与几乎必然收敛

**引理 12.1（Borel-Cantelli 第一引理）**
若 $\sum_{n=1}^\infty P(A_n) < \infty$，则 $P(\limsup_{n\to\infty} A_n) = 0$（即无穷多个 $A_n$ 同时发生的概率为零）。

**推论（充分条件）**：若对每个 $\varepsilon > 0$，$\sum_{n=1}^\infty P(|X_n - X| > \varepsilon) < \infty$，则 $X_n \xrightarrow{a.s.} X$。

**引理 12.2（Borel-Cantelli 第二引理）**
若事件 $\{A_n\}$ 相互独立，且 $\sum_{n=1}^\infty P(A_n) = \infty$，则 $P(\limsup_{n\to\infty} A_n) = 1$。

**示例**：强大数定律（SLLN）的核心结论即 $\bar{X}_n \xrightarrow{a.s.} \mu$，通常用 Borel-Cantelli 引理或鞅理论证明（见第 10 章）。

### 几乎必然收敛的逐路径性质

**命题 12.3**（运算封闭性）若 $X_n \xrightarrow{a.s.} X$，$Y_n \xrightarrow{a.s.} Y$，$g$ 连续，则

$$
aX_n + bY_n \xrightarrow{a.s.} aX + bY, \qquad g(X_n) \xrightarrow{a.s.} g(X)
$$

这是因为在 a.s. 的路径集合上，极限运算与普通实数极限的运算规则完全相同。

### 典型反例：依概率收敛但不几乎必然收敛

**"打字机序列"（Typewriter Sequence）**

设 $\Omega = [0, 1]$，$P$ 为 Lebesgue 测度。构造如下随机变量序列（按行排列指示函数）：

$$
X_1 = \mathbf{1}_{[0,1]}, \quad X_2 = \mathbf{1}_{[0,1/2]}, \quad X_3 = \mathbf{1}_{[1/2,1]},
$$
$$
X_4 = \mathbf{1}_{[0,1/3]}, \quad X_5 = \mathbf{1}_{[1/3,2/3]}, \quad X_6 = \mathbf{1}_{[2/3,1]}, \quad \ldots
$$

一般地，第 $2^k + j$（$0 \leq j < 2^k$）个随机变量为 $\mathbf{1}_{[j/2^k, (j+1)/2^k]}$。

- **依概率**：$P(|X_n - 0| > \varepsilon) = P(X_n = 1) = $ 相应区间长度 $\to 0$，故 $X_n \xrightarrow{P} 0$。
- **几乎必然**：对任意 $\omega \in [0,1]$，$X_n(\omega)$ 无穷次等于 $1$（每次 $\omega$ 被某个区间覆盖），故数列 $\{X_n(\omega)\}$ 不收敛到 $0$，$X_n \not\xrightarrow{a.s.} 0$。

这个例子清晰地展示：**依概率收敛不蕴含几乎必然收敛**。

---

## 12.4 $L^p$ 收敛与均方收敛

### $L^p$ 空间与范数

**定义 12.4（$L^p$ 范数）**
对 $p \geq 1$，随机变量 $X$ 的 $L^p$ 范数为

$$
\|X\|_p = \left(\mathbb{E}[|X|^p]\right)^{1/p}
$$

$L^p$ 空间为所有 $p$ 阶矩有限的随机变量的集合：$L^p = \{X : \mathbb{E}[|X|^p] < \infty\}$。

**定义 12.5（$L^p$ 收敛）**
若

$$
\boxed{\mathbb{E}[|X_n - X|^p] \to 0 \quad (n \to \infty)}
$$

则称 $X_n$ **在 $L^p$ 意义下收敛**（converge in $L^p$）到 $X$，记作

$$
X_n \xrightarrow{L^p} X \quad \text{或} \quad X_n \xrightarrow{p} X \text{（$p$ 阶矩收敛）}
$$

### 均方收敛（$L^2$ 收敛）

最重要的特殊情形是 $p = 2$：

**定义 12.6（均方收敛）**
若

$$
\boxed{\mathbb{E}[(X_n - X)^2] \to 0 \quad (n \to \infty)}
$$

则称 $X_n$ **均方收敛**（converge in mean square）到 $X$，记作 $X_n \xrightarrow{m.s.} X$ 或 $X_n \xrightarrow{L^2} X$。

**均方收敛与期望、方差的联系**：

$$
\mathbb{E}[(X_n - X)^2] = \operatorname{Var}(X_n - X) + (\mathbb{E}[X_n] - \mathbb{E}[X])^2
$$

因此均方收敛要求均值收敛且方差趋于零。

### $L^p$ 收敛的基本不等式

**Jensen 不等式**（$\phi$ 凸）：$\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)]$

**Hölder 不等式**：$\mathbb{E}[|XY|] \leq \|X\|_p \|Y\|_q$（$1/p + 1/q = 1$）

**Minkowski 不等式**：$\|X + Y\|_p \leq \|X\|_p + \|Y\|_p$（三角不等式）

**$L^p$ 空间的嵌套**（由 Jensen）：若 $p \geq q \geq 1$，则

$$
\|X\|_q \leq \|X\|_p \cdot (\text{vol}(\Omega))^{1/q - 1/p}
$$

在有限测度空间（概率空间）中：$L^p \subseteq L^q$（$p \geq q$），即**高阶矩有限蕴含低阶矩有限**。

**推论**：$X_n \xrightarrow{L^p} X$（$p \geq q$）$\Rightarrow$ $X_n \xrightarrow{L^q} X$。

### $L^p$ 收敛与期望的关系

**命题 12.4**：若 $X_n \xrightarrow{L^p} X$（$p \geq 1$），则：

1. $\mathbb{E}[|X_n|^p] \to \mathbb{E}[|X|^p]$（$p$ 阶矩收敛）
2. 若 $p \geq 1$：$\mathbb{E}[X_n] \to \mathbb{E}[X]$（可以在积分号下取极限）

**证明（2）**：由 Hölder 不等式（取 $q = 1, p = 1$ 或直接用 Markov）：

$$
|\mathbb{E}[X_n] - \mathbb{E}[X]| \leq \mathbb{E}[|X_n - X|] = \|X_n - X\|_1 \leq \|X_n - X\|_p \to 0
$$

$\blacksquare$

### 均方收敛的判别准则

**充分条件（二阶矩准则）**：若

$$
\mathbb{E}[X_n^2] \to c < \infty \quad \text{且} \quad \mathbb{E}[X_n X_m] \to c \quad (n, m \to \infty)
$$

则 $\{X_n\}$ 是 $L^2$ 中的 Cauchy 序列，从而存在 $L^2$ 极限 $X$（$L^2$ 空间完备）。

**示例（WLLN 的均方版本）**：设 $X_i$ i.i.d.，均值 $\mu$，方差 $\sigma^2 < \infty$，则

$$
\mathbb{E}\!\left[(\bar{X}_n - \mu)^2\right] = \frac{\sigma^2}{n} \to 0
$$

故 $\bar{X}_n \xrightarrow{L^2} \mu$，这比依概率收敛（WLLN）的结论更强。

### 一致可积与 $L^1$ 收敛

**定义 12.7（一致可积，UI）**
随机变量族 $\{X_n\}$ 称为**一致可积**，若

$$
\lim_{M \to \infty} \sup_n \mathbb{E}\!\left[|X_n| \cdot \mathbf{1}_{\{|X_n| > M\}}\right] = 0
$$

**定理 12.6**：$X_n \xrightarrow{L^1} X$ 当且仅当 $X_n \xrightarrow{P} X$ 且 $\{X_n\}$ 一致可积。

这是联系依概率收敛与 $L^1$ 收敛的关键桥梁。

---

## 12.5 收敛性之间的关系

### 蕴含关系总图

四种收敛之间的关系如下（$\Rightarrow$ 表示蕴含，$\not\Rightarrow$ 表示一般不蕴含）：

$$
\underbrace{X_n \xrightarrow{a.s.}}_{\text{几乎必然}} \;\Rightarrow\; \underbrace{X_n \xrightarrow{P}}_{\text{依概率}} \;\Rightarrow\; \underbrace{X_n \xrightarrow{d}}_{\text{依分布}}
$$

$$
\underbrace{X_n \xrightarrow{L^p}}_{\text{$L^p$ 收敛}} \;\Rightarrow\; \underbrace{X_n \xrightarrow{P}}_{\text{依概率}}
$$

$$
X_n \xrightarrow{L^p} \;\Rightarrow\; X_n \xrightarrow{L^q} \quad (p \geq q \geq 1)
$$

**关键不等式方向（不成立）**：

$$
X_n \xrightarrow{P} \;\not\Rightarrow\; X_n \xrightarrow{a.s.} \qquad (\text{打字机序列反例})
$$

$$
X_n \xrightarrow{P} \;\not\Rightarrow\; X_n \xrightarrow{L^p} \qquad (\text{需要矩条件})
$$

$$
X_n \xrightarrow{d} \;\not\Rightarrow\; X_n \xrightarrow{P} \qquad (\text{极限须为常数时例外})
$$

$$
X_n \xrightarrow{a.s.} \;\not\Rightarrow\; X_n \xrightarrow{L^p} \qquad (\text{需要一致可积性})
$$

### 定理：a.s. 收敛蕴含依概率收敛

**定理 12.7**：若 $X_n \xrightarrow{a.s.} X$，则 $X_n \xrightarrow{P} X$。

**证明**：对任意 $\varepsilon > 0$，

$$
\{|X_n - X| > \varepsilon\} \subseteq \bigcup_{k=n}^\infty \{|X_k - X| > \varepsilon\}
$$

故

$$
P(|X_n - X| > \varepsilon) \leq P\!\left(\bigcup_{k=n}^\infty \{|X_k - X| > \varepsilon\}\right)
$$

由 a.s. 收敛，$P\!\left(\bigcap_{n=1}^\infty \bigcup_{k=n}^\infty \{|X_k - X| > \varepsilon\}\right) = 0$，即

$$
P\!\left(\bigcup_{k=n}^\infty \{|X_k - X| > \varepsilon\}\right) \to 0 \quad (n \to \infty)
$$

（单调集合列，极限为零集的概率）因此 $P(|X_n - X| > \varepsilon) \to 0$。$\blacksquare$

### 定理：$L^p$ 收敛蕴含依概率收敛

**定理 12.8**：若 $X_n \xrightarrow{L^p} X$（$p \geq 1$），则 $X_n \xrightarrow{P} X$。

**证明**：由 Markov 不等式（对 $|X_n - X|^p$）：

$$
P(|X_n - X| > \varepsilon) = P(|X_n - X|^p > \varepsilon^p) \leq \frac{\mathbb{E}[|X_n - X|^p]}{\varepsilon^p} \to 0
$$

$\blacksquare$

### 定理：依概率收敛蕴含依分布收敛

**定理 12.9**：若 $X_n \xrightarrow{P} X$，则 $X_n \xrightarrow{d} X$。

**证明**：对任意连续有界函数 $f$，因 $f$ 一致连续（紧集上），对 $\varepsilon > 0$ 存在 $\delta > 0$ 使得 $|x - y| < \delta \Rightarrow |f(x) - f(y)| < \varepsilon$。

$$
|\mathbb{E}[f(X_n)] - \mathbb{E}[f(X)]| \leq \mathbb{E}[|f(X_n) - f(X)|]
$$

$$
\leq \varepsilon + 2\|f\|_\infty \cdot P(|X_n - X| \geq \delta) \to \varepsilon
$$

由 $\varepsilon$ 的任意性结论成立。$\blacksquare$

### 特殊情形：依分布收敛到常数等价于依概率收敛

**定理 12.10**：$X_n \xrightarrow{d} c$（常数）当且仅当 $X_n \xrightarrow{P} c$。

**证明**（$\Rightarrow$ 方向）：$P(|X_n - c| > \varepsilon) = P(X_n > c + \varepsilon) + P(X_n < c - \varepsilon)$。

$$
P(X_n > c + \varepsilon) = 1 - F_n(c + \varepsilon) \to 1 - F(c + \varepsilon) = 1 - 1 = 0
$$

$$
P(X_n < c - \varepsilon) = F_n((c-\varepsilon)^-) \to F((c-\varepsilon)^-) = 0
$$

（常数 $c$ 的分布函数 $F(x) = \mathbf{1}_{[c, \infty)}(x)$，在 $c \pm \varepsilon$ 处均连续）$\blacksquare$

**重要推论（Slutsky 定理的基础）**：大数定律给出的 $\bar{X}_n \xrightarrow{P} \mu$，结合 CLT 的 $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$，正是这两种收敛在统计中的典型组合。

### Skorokhod 表示定理

**定理 12.11（Skorokhod 表示定理）**
若 $X_n \xrightarrow{d} X$（取值于可分度量空间），则存在定义在同一概率空间上的随机变量 $\widetilde{X}_n$ 和 $\widetilde{X}$，使得：

$$
\widetilde{X}_n \overset{d}{=} X_n, \quad \widetilde{X} \overset{d}{=} X, \quad \text{且} \quad \widetilde{X}_n \xrightarrow{a.s.} \widetilde{X}
$$

**意义**：定理允许我们"把依分布收敛提升为几乎必然收敛"（通过重新选择概率空间）。这是很多极限定理证明的重要技巧：先在 Skorokhod 空间里用 a.s. 收敛做运算，再"翻译"回分布结论。

### 收敛关系的反例汇总

| 反例类型 | 构造 | 说明哪个蕴含不成立 |
|---------|------|-----------------|
| 打字机序列 | $X_n = \mathbf{1}_{[j/2^k,(j+1)/2^k]}$ | $\xrightarrow{P}$ 不蕴含 $\xrightarrow{a.s.}$ |
| 尖峰序列 | $X_n = n \cdot \mathbf{1}_{[0,1/n]}$ | $\xrightarrow{a.s.}$ 不蕴含 $\xrightarrow{L^1}$ |
| 分布收敛到非常数 | $X_n \sim \mathcal{N}(0,1)$（独立） | $\xrightarrow{d}$ 不蕴含 $\xrightarrow{P}$ |
| 期望不收敛 | $P(X_n = n) = 1/n$，$P(X_n=0) = 1-1/n$ | $\xrightarrow{P}$ 不蕴含 $\xrightarrow{L^1}$ |

---

## 本章小结

### 四种收敛的比较

| 收敛类型 | 符号 | 定义核心 | 强弱排序 | 典型应用 |
|---------|------|---------|---------|---------|
| 几乎必然收敛（a.s.） | $X_n \xrightarrow{a.s.} X$ | $P(\omega: X_n(\omega) \to X(\omega)) = 1$ | 最强（路径层面） | 强大数定律 |
| $L^p$ 收敛 | $X_n \xrightarrow{L^p} X$ | $\mathbb{E}[|X_n-X|^p] \to 0$ | 较强（矩层面） | 均方误差收敛、矩估计 |
| 依概率收敛 | $X_n \xrightarrow{P} X$ | $P(|X_n-X|>\varepsilon) \to 0$ | 中等（概率层面） | 弱大数定律、相合估计 |
| 依分布收敛 | $X_n \xrightarrow{d} X$ | $F_n(x) \to F(x)$ 在连续点 | 最弱（分布层面） | 中心极限定理、渐近理论 |

### 蕴含关系总结

$$
\xrightarrow{a.s.} \;\Longrightarrow\; \xrightarrow{P} \;\Longrightarrow\; \xrightarrow{d}
$$

$$
\xrightarrow{L^p} \;\Longrightarrow\; \xrightarrow{L^q} \;(p \geq q)\;\Longrightarrow\; \xrightarrow{P} \;\Longrightarrow\; \xrightarrow{d}
$$

**附加条件下的逆方向**：

| 逆蕴含 | 所需附加条件 |
|--------|------------|
| $\xrightarrow{P} \Rightarrow \xrightarrow{a.s.}$ | 可取子列（必存在几乎必然收敛子列） |
| $\xrightarrow{P} \Rightarrow \xrightarrow{L^p}$ | 一致可积性（$p = 1$）或有界性 |
| $\xrightarrow{d} \Rightarrow \xrightarrow{P}$ | 极限为常数 |
| $\xrightarrow{a.s.} \Rightarrow \xrightarrow{L^p}$ | 一致可积（$p \geq 1$）或有界（$|X_n| \leq Y \in L^p$） |

**子列原理**（常用工具）：$X_n \xrightarrow{P} X$ 当且仅当对 $X_n$ 的任意子列，存在进一步的子列几乎必然收敛到 $X$。

---

## 深度学习应用：模型收敛性分析与 PAC 学习理论

### 背景：为什么收敛理论在深度学习中至关重要

深度学习的理论基础在很大程度上依赖于本章所讨论的各种收敛概念：

- **模型训练**：损失函数是否收敛？以何种方式收敛（a.s.、依概率还是均方）？
- **泛化理论**：训练误差与测试误差之差是否依概率趋于零？
- **PAC 学习**：样本量多大才能以高概率近似正确地学习到目标概念？

这些问题的回答都需要精确的收敛性语言。

### 12.6.1 随机梯度下降（SGD）的收敛性

考虑最小化期望风险 $L(\theta) = \mathbb{E}[\ell(\theta; Z)]$，其中 $Z$ 为训练数据。

**SGD 更新规则**（第 $t$ 步）：

$$
\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \ell(\theta_t; Z_t)
$$

其中 $Z_t$ 为第 $t$ 步随机抽取的样本（或 mini-batch），$\eta_t > 0$ 为学习率。

**梯度估计量的性质**：设 $g_t = \nabla_\theta \ell(\theta_t; Z_t)$，由于 $Z_t$ 随机，

$$
\mathbb{E}[g_t | \theta_t] = \mathbb{E}_{Z}[\nabla_\theta \ell(\theta_t; Z)] = \nabla_\theta L(\theta_t) \quad \text{（无偏）}
$$

$$
\operatorname{Var}(g_t) = \mathbb{E}[\|g_t - \nabla L(\theta_t)\|^2] \triangleq \sigma_t^2 \quad \text{（方差随批大小减小）}
$$

**定理（SGD 依概率收敛）**：在 Lipschitz 梯度（$L$-smooth）和有界方差假设下，若学习率满足 Robbins-Monro 条件

$$
\sum_{t=1}^\infty \eta_t = \infty, \qquad \sum_{t=1}^\infty \eta_t^2 < \infty \quad \text{（如 } \eta_t = c/t\text{）}
$$

则 $\min_{t \leq T} \mathbb{E}[\|\nabla L(\theta_t)\|^2] \to 0$（找到近似稳定点），对非凸问题这是依概率意义下的收敛。

**与四种收敛的对应关系**：

| 收敛类型 | 在 SGD 中对应 | 所需条件 |
|---------|-------------|---------|
| 依分布收敛 | 参数分布趋向平稳分布 | 学习率衰减 |
| 依概率收敛 | $\theta_t$ 以高概率接近最优点 | 凸或PL条件，方差有界 |
| 均方收敛 | $\mathbb{E}[\|\theta_t - \theta^*\|^2] \to 0$ | 强凸，常数学习率（有偏差） |
| 几乎必然收敛 | 几乎所有训练路径都收敛 | 较强假设，如Polyak步长 |

### 12.6.2 PAC 学习理论中的收敛性

**PAC（Probably Approximately Correct）学习框架**由 Leslie Valiant 在 1984 年提出，是机器学习泛化理论的基石。

**基本设置**：
- 输入空间 $\mathcal{X}$，标签空间 $\mathcal{Y} = \{0, 1\}$
- 未知数据分布 $\mathcal{D}$ 定义在 $\mathcal{X} \times \mathcal{Y}$ 上
- 假设类 $\mathcal{H}$（模型族）
- 训练集 $S = \{(x_1, y_1), \ldots, (x_n, y_n)\}$，i.i.d. 来自 $\mathcal{D}$

**风险定义**：
- **真实风险**（泛化误差）：$R(h) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\mathbf{1}(h(x) \neq y)]$
- **经验风险**（训练误差）：$\hat{R}_n(h) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(h(x_i) \neq y_i)$

**核心问题**：何时 $\hat{R}_n(h) \xrightarrow{P} R(h)$（一致地对所有 $h \in \mathcal{H}$）？

**定理 12.12（均匀大数定律，ULLN）**
若 $\mathcal{H}$ 的 Rademacher 复杂度 $\mathfrak{R}_n(\mathcal{H}) \to 0$，则

$$
\sup_{h \in \mathcal{H}} |R(h) - \hat{R}_n(h)| \xrightarrow{P} 0
$$

**定理 12.13（有限假设类的 PAC 界）**
设 $|\mathcal{H}| < \infty$，则以概率至少 $1 - \delta$，对所有 $h \in \mathcal{H}$，

$$
\boxed{R(h) \leq \hat{R}_n(h) + \sqrt{\frac{\ln|\mathcal{H}| + \ln(1/\delta)}{2n}}}
$$

**证明思路**：
1. 对固定 $h$，由 Hoeffding 不等式（i.i.d. 有界随机变量）：

$$
P(R(h) - \hat{R}_n(h) > \varepsilon) \leq \exp(-2n\varepsilon^2)
$$

2. Union bound（联合界）对所有 $h \in \mathcal{H}$：

$$
P\!\left(\sup_{h \in \mathcal{H}} (R(h) - \hat{R}_n(h)) > \varepsilon\right) \leq |\mathcal{H}| \cdot \exp(-2n\varepsilon^2)
$$

3. 令右端 $= \delta$，解出 $\varepsilon = \sqrt{\frac{\ln|\mathcal{H}| + \ln(1/\delta)}{2n}}$。$\blacksquare$

**PAC 界的收敛语言解读**：该界说明 $\sup_h |R(h) - \hat{R}_n(h)| = O_P(1/\sqrt{n})$，即**依概率以 $O(1/\sqrt{n})$ 速率收敛到零**，这正是依概率收敛的量化版本。

### PyTorch 代码示例：模型收敛性监测与 PAC 界验证

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)


# ─── 1. 合成二分类数据 ────────────────────────────────────────────────────────
def make_dataset(n: int, d: int = 20):
    """
    生成 n 个 d 维样本的线性可分二分类数据。
    真实分类器：w* = e_1（仅依赖第一个特征）。
    """
    X = torch.randn(n, d)
    y = (X[:, 0] > 0).long()   # 真实标签
    return X, y


# ─── 2. 简单线性分类器 ────────────────────────────────────────────────────────
class LinearClassifier(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x):
        return self.fc(x).squeeze()

    def predict(self, x):
        return (self.forward(x) > 0).long()


# ─── 3. 训练并记录收敛过程 ────────────────────────────────────────────────────
def train_and_track_convergence(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_epochs: int = 200,
    lr: float = 0.05,
    batch_size: int = 32,
) -> dict:
    """
    训练模型，逐 epoch 记录训练误差、测试误差和泛化差距（gap）。
    从收敛理论角度：
      - 训练误差（经验风险）应单调下降
      - 泛化差距应依概率趋于零（PAC 界保证）
    """
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(X_train, y_train.float())
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {
        'train_loss': [], 'test_loss': [],
        'train_err':  [], 'test_err':  [],
        'gap':        [],  # 泛化差距 = test_err - train_err
    }

    for epoch in range(n_epochs):
        # ── 训练阶段 ──────────────────────────────────────────────────────────
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        # ── 评估阶段 ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            # 训练集指标
            train_logits = model(X_train)
            train_loss   = criterion(train_logits, y_train.float()).item()
            train_err    = (model.predict(X_train) != y_train).float().mean().item()

            # 测试集指标
            test_logits = model(X_test)
            test_loss   = criterion(test_logits, y_test.float()).item()
            test_err    = (model.predict(X_test) != y_test).float().mean().item()

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_err'].append(train_err)
        history['test_err'].append(test_err)
        history['gap'].append(test_err - train_err)

    return history


# ─── 4. 验证 PAC 界：泛化差距与样本量的关系 ──────────────────────────────────
def verify_pac_bound(d: int = 20, n_trials: int = 30) -> None:
    """
    对不同训练样本量 n，重复 n_trials 次实验，
    验证泛化差距（test_err - train_err）是否与 PAC 界 O(1/√n) 吻合。

    从依概率收敛角度：随 n 增大，sup_h |R(h) - R̂_n(h)| → 0（依概率），
    这里用单个模型的泛化差距作为近似替代。
    """
    train_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    n_test      = 10000  # 固定大测试集近似真实风险

    X_test, y_test = make_dataset(n_test, d)

    mean_gaps = []
    std_gaps  = []

    for n in train_sizes:
        gaps = []
        for _ in range(n_trials):
            X_tr, y_tr = make_dataset(n, d)
            model = LinearClassifier(d)

            # 快速训练（固定 epoch 数）
            opt = optim.SGD(model.parameters(), lr=0.1)
            crit = nn.BCEWithLogitsLoss()
            dataset = TensorDataset(X_tr, y_tr.float())
            loader  = DataLoader(dataset, batch_size=min(32, n), shuffle=True)

            for _ in range(100):
                for Xb, yb in loader:
                    opt.zero_grad()
                    crit(model(Xb), yb).backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                train_err = (model.predict(X_tr) != y_tr).float().mean().item()
                test_err  = (model.predict(X_test) != y_test).float().mean().item()
                gaps.append(test_err - train_err)

        mean_gaps.append(np.mean(gaps))
        std_gaps.append(np.std(gaps))

        print(f"n={n:5d}: 平均泛化差距 = {np.mean(gaps):.4f} ± {np.std(gaps):.4f}")

    # ── 可视化：验证 O(1/√n) 收敛速率 ────────────────────────────────────────
    n_arr = np.array(train_sizes, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：泛化差距均值与 PAC 界理论曲线
    delta    = 0.05    # 置信度参数
    H_size   = 2 ** d  # 对线性分类器：VC 维 ≈ d+1，简化估计 |H| ≈ 2^d
    pac_bound = np.sqrt((d * np.log(2) + np.log(1 / delta)) / (2 * n_arr))

    axes[0].errorbar(train_sizes, mean_gaps, yerr=std_gaps,
                     fmt='o-', capsize=5, label='实验泛化差距（均值±标准差）', color='steelblue')
    axes[0].plot(n_arr, pac_bound, 'r--', label=f'PAC 界 $\\sqrt{{(d\\ln 2 + \\ln(1/\\delta))/2n}}$', linewidth=2)
    axes[0].set_xlabel('训练样本量 $n$')
    axes[0].set_ylabel('泛化差距（测试误差 − 训练误差）')
    axes[0].set_title('PAC 界验证：泛化差距随样本量的衰减')
    axes[0].legend(fontsize=9)
    axes[0].set_xscale('log')

    # 右图：对数坐标验证 O(1/√n) 速率
    # 如果斜率约为 -0.5，说明收敛速率确为 O(n^{-1/2})
    log_n    = np.log(n_arr)
    log_gap  = np.log(np.array(mean_gaps) + 1e-8)
    slope, intercept = np.polyfit(log_n, log_gap, 1)

    axes[1].scatter(log_n, log_gap, color='steelblue', zorder=5, label='实验数据（对数坐标）')
    axes[1].plot(log_n, slope * log_n + intercept, 'r-',
                 label=f'拟合斜率 = {slope:.3f}（理论值 ≈ −0.5）', linewidth=2)
    axes[1].set_xlabel('$\\ln(n)$')
    axes[1].set_ylabel('$\\ln$(泛化差距)')
    axes[1].set_title('对数坐标验证收敛速率 $O(n^{-1/2})$')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('pac_bound_convergence.png', dpi=150)
    plt.show()

    print(f"\n拟合收敛速率指数: {slope:.3f}（PAC 理论预测：−0.5）")


# ─── 5. 收敛类型可视化：四种收敛方式对比 ─────────────────────────────────────
def visualize_convergence_types() -> None:
    """
    可视化四种收敛类型的典型行为，帮助直观理解其差异。
    """
    n_paths = 50    # 样本路径数量
    T = 200         # 时间步数
    ts = np.arange(1, T + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('四种收敛类型的直观比较（随机过程轨迹）', fontsize=14)

    # ── (1) 几乎必然收敛：每条路径都趋向 0 ──────────────────────────────────
    ax = axes[0, 0]
    for _ in range(n_paths):
        path = np.cumsum(np.random.randn(T)) / ts  # SLLN：样本均值趋向 0
        ax.plot(ts, path, alpha=0.3, linewidth=0.8, color='steelblue')
    ax.axhline(0, color='red', linewidth=2, label='极限值 0')
    ax.set_title('几乎必然收敛\n$\\bar{X}_n \\xrightarrow{a.s.} 0$（SLLN）', fontsize=11)
    ax.set_xlabel('$n$'); ax.set_ylabel('$\\bar{X}_n$')
    ax.legend(); ax.set_ylim(-1.5, 1.5)

    # ── (2) 依概率收敛但不几乎必然（打字机序列模拟）────────────────────────
    ax = axes[0, 1]
    # 模拟：P(X_n=1) = 1/n，P(X_n=0) = 1-1/n
    for _ in range(n_paths):
        path = np.array([
            1.0 if np.random.rand() < 1 / t else 0.0
            for t in ts
        ])
        ax.plot(ts, path, alpha=0.2, linewidth=0.8, color='orange')
    # 叠加"依概率"收敛的期望路径
    ax.plot(ts, 1.0 / ts, color='red', linewidth=2.5, label='$P(X_n=1)=1/n \\to 0$')
    ax.set_title('依概率收敛\n$X_n \\xrightarrow{P} 0$（但各路径持续跳动）', fontsize=11)
    ax.set_xlabel('$n$'); ax.set_ylabel('$X_n$')
    ax.legend(); ax.set_ylim(-0.1, 1.3)

    # ── (3) 均方收敛（L²）：MSE 趋向 0 ──────────────────────────────────────
    ax = axes[1, 0]
    mse_values = []
    for _ in range(500):  # 大量模拟估计 MSE
        path = np.cumsum(np.random.randn(T)) / ts
        mse_values.append(path ** 2)
    mse_mean = np.mean(mse_values, axis=0)
    theoretical_mse = 1.0 / ts  # E[X̄_n²] = σ²/n = 1/n

    ax.semilogy(ts, mse_mean, color='steelblue', linewidth=2, label='实验 MSE $\\mathbb{E}[\\bar{X}_n^2]$')
    ax.semilogy(ts, theoretical_mse, 'r--', linewidth=2, label='理论 $1/n$')
    ax.set_title('$L^2$（均方）收敛\n$\\mathbb{E}[\\bar{X}_n^2] = 1/n \\to 0$', fontsize=11)
    ax.set_xlabel('$n$'); ax.set_ylabel('MSE（对数尺度）')
    ax.legend()

    # ── (4) 依分布收敛：分布函数逐渐趋向正态 CDF ────────────────────────────
    ax = axes[1, 1]
    from scipy import stats
    x_grid = np.linspace(-3, 3, 300)
    colors = plt.cm.viridis(np.linspace(0.2, 1.0, 5))

    for idx, n_sample in enumerate([2, 5, 20, 100, 1000]):
        # Z_n = (S_n - 0) / sqrt(n)，来自均匀分布 U(-1,1)
        z_vals = np.array([
            np.random.uniform(-1, 1, n_sample).sum() / np.sqrt(n_sample / 3)
            for _ in range(5000)
        ])
        empirical_cdf = np.array([np.mean(z_vals <= x) for x in x_grid])
        ax.plot(x_grid, empirical_cdf, color=colors[idx], linewidth=1.5,
                label=f'$n={n_sample}$', alpha=0.9)

    # 标准正态 CDF（极限分布）
    ax.plot(x_grid, stats.norm.cdf(x_grid), 'k-', linewidth=2.5, label='$\\mathcal{N}(0,1)$ CDF（极限）')
    ax.set_title('依分布收敛\n$Z_n \\xrightarrow{d} \\mathcal{N}(0,1)$（CLT）', fontsize=11)
    ax.set_xlabel('$x$'); ax.set_ylabel('$F_n(x)$')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('convergence_types_visualization.png', dpi=150)
    plt.show()


# ─── 主程序 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("实验 1：模型收敛性监测")
    print("=" * 60)
    d = 20
    X_train, y_train = make_dataset(1000, d)
    X_test,  y_test  = make_dataset(5000, d)

    model   = LinearClassifier(d)
    history = train_and_track_convergence(model, X_train, y_train, X_test, y_test,
                                           n_epochs=300, lr=0.05, batch_size=64)

    final_train = history['train_err'][-1]
    final_test  = history['test_err'][-1]
    final_gap   = history['gap'][-1]
    print(f"\n训练结束：训练误差={final_train:.4f}, 测试误差={final_test:.4f}, "
          f"泛化差距={final_gap:.4f}")

    # 绘制训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(len(history['train_err']))

    ax1.plot(epochs, history['train_err'], label='训练误差（经验风险）', linewidth=2)
    ax1.plot(epochs, history['test_err'],  label='测试误差（真实风险近似）', linewidth=2)
    ax1.set_title('训练/测试误差随训练轮数的变化'); ax1.set_xlabel('Epoch')
    ax1.set_ylabel('0-1 误差'); ax1.legend()

    ax2.plot(epochs, history['gap'], color='darkorange', linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_title('泛化差距（测试误差 − 训练误差）\n应依概率趋于 0（PAC 理论保证）')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('泛化差距')

    plt.tight_layout()
    plt.savefig('model_convergence_monitoring.png', dpi=150)
    plt.show()

    print("\n" + "=" * 60)
    print("实验 2：PAC 界验证（需要几分钟）")
    print("=" * 60)
    verify_pac_bound(d=20, n_trials=20)

    print("\n" + "=" * 60)
    print("实验 3：四种收敛类型可视化")
    print("=" * 60)
    visualize_convergence_types()
```

### 理论要点总结

**从收敛理论视角理解深度学习**：

| 深度学习概念 | 对应收敛理论 | 数学表达 |
|------------|------------|---------|
| 训练损失趋向零 | $L^2$ 或依概率收敛 | $\mathbb{E}[\ell(\theta_t)] \to 0$ 或 $\ell(\theta_t) \xrightarrow{P} 0$ |
| 泛化误差收敛 | 一致大数定律（ULLN） | $\sup_h \|R(h) - \hat{R}_n(h)\| \xrightarrow{P} 0$ |
| SGD 找到稳定点 | 依概率收敛 | $\|\nabla L(\theta_t)\| \xrightarrow{P} 0$ |
| PAC 样本复杂度 | 依概率收敛速率 | $O_P(1/\sqrt{n})$ 泛化界 |
| 批归一化（BN） | CLT（依分布收敛） | 批均值 $\xrightarrow{d} \mathcal{N}(\mu, \sigma^2/m)$ |
| Dropout 正则化 | 依概率收敛（随机近似） | 期望网络 $\approx$ 集成均值 |

**PAC 学习的核心洞察**：随着样本量 $n \to \infty$：

$$
\sup_{h \in \mathcal{H}} |R(h) - \hat{R}_n(h)| = O_P\!\left(\sqrt{\frac{\text{complexity}(\mathcal{H}) + \ln(1/\delta)}{n}}\right) \xrightarrow{P} 0
$$

这正是"依概率收敛"的量化表达——泛化差距**依概率**以 $O(1/\sqrt{n})$ 的速率趋于零，其中"complexity$(\mathcal{H})$"可以是 $\ln|\mathcal{H}|$（有限假设类）、VC 维，或 Rademacher 复杂度的函数。

---

## 练习题

**题 1（依概率收敛基础）** 设 $X_1, X_2, \ldots$ 为独立随机变量，$X_n$ 的分布为

$$
P(X_n = n^2) = \frac{1}{n^2}, \quad P(X_n = 0) = 1 - \frac{1}{n^2}
$$

（a）证明 $X_n \xrightarrow{P} 0$。

（b）计算 $\mathbb{E}[X_n]$ 和 $\operatorname{Var}(X_n)$，并由此说明 $X_n$ **不**以均方收敛到 $0$。

（c）利用 Borel-Cantelli 第一引理证明 $X_n \xrightarrow{a.s.} 0$。

---

**题 2（$\delta$ 方法应用）** 设 $X_1, X_2, \ldots$ i.i.d.，$\mathbb{E}[X_i] = \mu > 0$，$\operatorname{Var}(X_i) = \sigma^2 < \infty$，且 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$。

（a）写出 $\sqrt{n}(\bar{X}_n - \mu)$ 的渐近分布（CLT）。

（b）利用 $\delta$ 方法，求 $\sqrt{n}(1/\bar{X}_n - 1/\mu)$ 的渐近分布。

（c）利用 $\delta$ 方法，求 $\sqrt{n}(\ln\bar{X}_n - \ln\mu)$ 的渐近分布。

（d）若 $\mu = 2$，$\sigma^2 = 4$，当 $n = 100$ 时，用渐近分布近似 $P(\ln\bar{X}_{100} > 0.8)$。

---

**题 3（收敛类型的判别）** 对以下每个序列，判断其对哪些类型的收敛成立（a.s.、$L^2$、依概率、依分布），并给出理由或反例。设 $\Omega = [0, 1]$，$P$ 为 Lebesgue 测度。

（a）$X_n(\omega) = \omega^n$

（b）$X_n(\omega) = n \cdot \mathbf{1}_{[0, 1/n]}(\omega)$

（c）$X_n(\omega) = \sin(2\pi n \omega)$

（d）$X_n(\omega) = \mathbf{1}_{[0, 1/n]}(\omega)$

---

**题 4（Slutsky 定理与渐近分布）** 设 $\hat{\sigma}_n^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X}_n)^2$ 为样本方差（有偏版本），$X_i$ i.i.d.，均值 $\mu$，四阶矩有限。

（a）证明 $\hat{\sigma}_n^2 \xrightarrow{P} \sigma^2$（提示：用 WLLN 和连续映射定理）。

（b）由 CLT 知 $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$。利用 Slutsky 定理，证明

$$
T_n = \frac{\sqrt{n}(\bar{X}_n - \mu)}{\hat{\sigma}_n} \xrightarrow{d} \mathcal{N}(0, 1)
$$

（c）这个结论为什么比直接用已知方差 $\sigma$ 更有实际价值？

---

**题 5（PAC 界与样本复杂度）** 考虑在 $\mathbb{R}^d$ 中用轴对齐矩形（axis-aligned rectangles）进行二分类：假设类 $\mathcal{H}_{\text{rect}}$ 由所有形如 $h_{a_1,b_1,\ldots,a_d,b_d}(x) = \mathbf{1}[a_k \leq x_k \leq b_k, \forall k]$ 的分类器组成（将矩形内的点分为正类）。

（a）$\mathcal{H}_{\text{rect}}$ 是无限假设类，直接用有限类的 PAC 界（定理 12.13）不适用。但已知 $\mathcal{H}_{\text{rect}}$ 的 VC 维为 $d_{\text{VC}} = 2d$。利用 VC 维的 PAC 界（Vapnik-Chervonenkis 定理）：

$$
P\!\left(\sup_{h \in \mathcal{H}} |R(h) - \hat{R}_n(h)| > \varepsilon\right) \leq 8 \cdot \left(\frac{en}{d_{\text{VC}}}\right)^{d_{\text{VC}}} \cdot e^{-n\varepsilon^2/8}
$$

写出此概率 $\leq \delta$ 时的样本复杂度（$n$ 关于 $\varepsilon, \delta, d$ 的表达式）。

（b）当 $d = 10$，$\varepsilon = 0.05$，$\delta = 0.05$ 时，至少需要多大的样本量 $n$？（给出数量级估计即可）

（c）上述结果说明样本复杂度关于维度 $d$ 是**线性**的（而非指数级）。从依概率收敛的角度解释：这意味着随样本量增大，$n$ 个样本足以保证泛化差距的依概率收敛速率不受维度灾难的影响——为什么这在实践中很重要？

---

## 练习答案

<details>
<summary>题 1 详细解答</summary>

**（a）$X_n \xrightarrow{P} 0$**

对任意 $\varepsilon > 0$（不妨设 $\varepsilon < 1$，否则更简单）：

$$
P(|X_n - 0| > \varepsilon) = P(X_n = n^2) = \frac{1}{n^2} \to 0
$$

故 $X_n \xrightarrow{P} 0$。$\blacksquare$

**（b）期望与方差**

$$
\mathbb{E}[X_n] = n^2 \cdot \frac{1}{n^2} + 0 \cdot \left(1 - \frac{1}{n^2}\right) = 1 \quad \forall n
$$

$$
\mathbb{E}[X_n^2] = (n^2)^2 \cdot \frac{1}{n^2} = n^2 \to \infty
$$

$$
\mathbb{E}[X_n^2] = n^2 \to \infty \neq 0 = \mathbb{E}[0^2]
$$

故 $\mathbb{E}[(X_n - 0)^2] = \mathbb{E}[X_n^2] = n^2 \to \infty$，$X_n$ 不均方收敛到 $0$（也不 $L^1$ 收敛：$\mathbb{E}[|X_n|] = 1 \not\to 0$）。

**（c）Borel-Cantelli 推导 a.s. 收敛**

对任意 $\varepsilon > 0$，令 $A_n = \{|X_n| > \varepsilon\} = \{X_n = n^2\}$。

$$
\sum_{n=1}^\infty P(A_n) = \sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6} < \infty
$$

由 Borel-Cantelli 第一引理，$P(\limsup_{n\to\infty} A_n) = 0$，即以概率 1 只有有限个 $n$ 满足 $|X_n| > \varepsilon$，故对几乎所有 $\omega$，当 $n$ 足够大时 $|X_n(\omega)| \leq \varepsilon$，即 $X_n \xrightarrow{a.s.} 0$。$\blacksquare$

**注**：本题展示了一个反直觉的现象：$\mathbb{E}[X_n] = 1 \not\to 0$（即 $L^1$ 不收敛），但 $X_n \xrightarrow{a.s.} 0$（强收敛）。这正是"a.s. 收敛不蕴含 $L^1$ 收敛"的典型例子，原因是 $\{X_n\}$ 不一致可积。

</details>

<details>
<summary>题 2 详细解答</summary>

**（a）CLT**

$$
\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)
$$

**（b）$g(x) = 1/x$，$g'(x) = -1/x^2$**

由 $\delta$ 方法（在 $\theta = \mu$ 处）：

$$
\sqrt{n}\!\left(\frac{1}{\bar{X}_n} - \frac{1}{\mu}\right) \xrightarrow{d} \mathcal{N}\!\left(0,\, \left(\frac{-1}{\mu^2}\right)^2 \sigma^2\right) = \mathcal{N}\!\left(0,\, \frac{\sigma^2}{\mu^4}\right)
$$

**（c）$g(x) = \ln x$，$g'(x) = 1/x$**

$$
\sqrt{n}(\ln\bar{X}_n - \ln\mu) \xrightarrow{d} \mathcal{N}\!\left(0,\, \frac{\sigma^2}{\mu^2}\right)
$$

**（d）数值近似**

$\mu = 2$，$\sigma^2 = 4$，$n = 100$。渐近标准差为 $\frac{\sigma}{\mu \sqrt{n}} = \frac{2}{2 \times 10} = 0.1$。

$$
P(\ln\bar{X}_{100} > 0.8) \approx P\!\left(\mathcal{N}(\ln 2, 0.01) > 0.8\right) = P\!\left(Z > \frac{0.8 - \ln 2}{0.1}\right)
$$

$$
= P\!\left(Z > \frac{0.8 - 0.6931}{0.1}\right) = P(Z > 1.069) = 1 - \Phi(1.069) \approx 1 - 0.8574 = \mathbf{0.1426}
$$

</details>

<details>
<summary>题 3 详细解答</summary>

**（a）$X_n(\omega) = \omega^n$**

极限：$\lim_{n\to\infty} \omega^n = 0$ 对 $\omega \in [0,1)$，$= 1$ 对 $\omega = 1$。

- **a.s. 收敛**：$P(\omega = 1) = 0$（单点集），故 $X_n \xrightarrow{a.s.} 0$。
- **依概率收敛**：由 a.s. 收敛蕴含，$X_n \xrightarrow{P} 0$。
- **依分布收敛**：$X_n \xrightarrow{d} 0$（由依概率蕴含）。
- **$L^2$ 收敛**：$\mathbb{E}[X_n^2] = \int_0^1 \omega^{2n} d\omega = \frac{1}{2n+1} \to 0$，故 $X_n \xrightarrow{L^2} 0$。

结论：**四种收敛均成立**。

**（b）$X_n(\omega) = n \cdot \mathbf{1}_{[0,1/n]}(\omega)$**

- **a.s. 收敛**：对 $\omega \in (0,1]$，当 $n > 1/\omega$ 时 $X_n(\omega) = 0$，故 $X_n(\omega) \to 0$。在 $\omega = 0$ 处 $X_n(0) = n \to \infty$，但 $P(\omega = 0) = 0$。故 $X_n \xrightarrow{a.s.} 0$。
- **依概率收敛**：$P(|X_n| > \varepsilon) = P(X_n = n) = 1/n \to 0$（对 $\varepsilon < n$ 时），故 $X_n \xrightarrow{P} 0$。
- **$L^2$ 收敛**：$\mathbb{E}[X_n^2] = n^2 \cdot \frac{1}{n} = n \to \infty$，故 $X_n \not\xrightarrow{L^2} 0$。
- **$L^1$ 收敛**：$\mathbb{E}[|X_n|] = n \cdot \frac{1}{n} = 1 \not\to 0$，故 $X_n \not\xrightarrow{L^1} 0$。

结论：**a.s. 和依概率收敛成立；$L^p$（$p \geq 1$）收敛不成立**。这是 a.s. 收敛不蕴含 $L^1$ 收敛的典型例子（不满足一致可积）。

**（c）$X_n(\omega) = \sin(2\pi n \omega)$**

- **依分布收敛**：$P(X_n \leq x) \to \frac{1}{\pi} \arcsin(x) + \frac{1}{2}$（$x \in [-1,1]$），即收敛到弧正弦分布。
- **a.s. 收敛**：由 Weyl 等分定理，对几乎所有 $\omega$，$\{n\omega\}$（小数部分）在 $[0,1]$ 上均匀分布，故 $\sin(2\pi n\omega)$ **不**收敛（在 $[-1,1]$ 内稠密振荡）。
- **依概率收敛**：若依概率收敛到某极限 $X$，则 $X_n \xrightarrow{d} X$，但 $X_n$ 依分布收敛到弧正弦分布（非常数），故 $X$ 只能是弧正弦分布，而 $X_n \not\xrightarrow{P} X$（路径层面不收敛）。实际上 $X_n \not\xrightarrow{P}$ 任何极限（反证：若收到常数 $c$，则依分布极限为 $c$，矛盾）。
- **$L^2$ 收敛**：$\mathbb{E}[\sin^2(2\pi n\omega)] = 1/2$ 不趋于零，故不 $L^2$ 收敛到 $0$。

结论：**仅依分布收敛成立（到弧正弦分布）；a.s.、依概率、$L^p$ 均不收敛**。

**（d）$X_n(\omega) = \mathbf{1}_{[0,1/n]}(\omega)$**

- **a.s. 收敛**：对 $\omega \in (0,1]$，当 $n > 1/\omega$ 时 $X_n(\omega) = 0$，故 $X_n \xrightarrow{a.s.} 0$（$\omega = 0$ 为零测集）。
- **$L^p$ 收敛**：$\mathbb{E}[|X_n|^p] = \int_0^{1/n} 1\, d\omega = 1/n \to 0$，故 $X_n \xrightarrow{L^p} 0$ 对所有 $p \geq 1$。
- 由以上两种均蕴含依概率和依分布收敛。

结论：**四种收敛均成立**（与 (b) 对比：(b) 中振幅随 $n$ 增长导致 $L^p$ 失败，而 (d) 中振幅为 $1$，矩有界）。

</details>

<details>
<summary>题 4 详细解答</summary>

**（a）$\hat{\sigma}_n^2 \xrightarrow{P} \sigma^2$**

分解：

$$
\hat{\sigma}_n^2 = \frac{1}{n}\sum_{i=1}^n X_i^2 - \bar{X}_n^2
$$

- 由 WLLN（对 $X_i^2$，需 $\mathbb{E}[X_i^4] < \infty$，即四阶矩有限）：$\frac{1}{n}\sum X_i^2 \xrightarrow{P} \mathbb{E}[X^2] = \sigma^2 + \mu^2$
- 由 WLLN（对 $X_i$）：$\bar{X}_n \xrightarrow{P} \mu$
- 由连续映射定理（$g(x) = x^2$ 连续）：$\bar{X}_n^2 \xrightarrow{P} \mu^2$

由运算规则：

$$
\hat{\sigma}_n^2 = \frac{1}{n}\sum X_i^2 - \bar{X}_n^2 \xrightarrow{P} (\sigma^2 + \mu^2) - \mu^2 = \sigma^2
$$

$\blacksquare$

**（b）Student 化统计量渐近正态**

由（a），$\hat{\sigma}_n \xrightarrow{P} \sigma$（由 CMT，$g(x) = \sqrt{x}$ 在 $\sigma^2 > 0$ 处连续）。

由 Slutsky 定理（$X_n \xrightarrow{d} X$，$Y_n \xrightarrow{P} c \Rightarrow X_n/Y_n \xrightarrow{d} X/c$）：

$$
T_n = \frac{\sqrt{n}(\bar{X}_n - \mu)}{\hat{\sigma}_n} = \underbrace{\sqrt{n}(\bar{X}_n - \mu)}_{\xrightarrow{d}\,\mathcal{N}(0,\sigma^2)} \cdot \underbrace{\frac{1}{\hat{\sigma}_n}}_{\xrightarrow{P}\,1/\sigma}
$$

由 Slutsky 定理：

$$
T_n \xrightarrow{d} \mathcal{N}(0, \sigma^2) \cdot \frac{1}{\sigma} = \mathcal{N}(0, 1)
$$

$\blacksquare$

**（c）实际价值**

在实际应用中，$\sigma$ 通常未知（若已知则可直接使用正态分布）。$T_n$ 的渐近正态性意味着：即使不知道总体方差，用样本标准差 $\hat{\sigma}_n$ 代替后，统计量仍渐近地服从标准正态分布。这直接支撑了大样本下 $z$ 检验和置信区间构造的理论基础，是统计推断中最重要的渐近结果之一。

</details>

<details>
<summary>题 5 详细解答</summary>

**（a）VC 维 PAC 样本复杂度**

令右端 $\leq \delta$：

$$
8 \cdot \left(\frac{en}{d_{\text{VC}}}\right)^{d_{\text{VC}}} \cdot e^{-n\varepsilon^2/8} \leq \delta
$$

取对数：

$$
\ln 8 + d_{\text{VC}} \ln\!\left(\frac{en}{d_{\text{VC}}}\right) - \frac{n\varepsilon^2}{8} \leq \ln\delta
$$

整理得：

$$
\frac{n\varepsilon^2}{8} \geq \ln 8 + d_{\text{VC}} \ln\!\left(\frac{en}{d_{\text{VC}}}\right) - \ln\delta = \ln\!\frac{8}{\delta} + d_{\text{VC}} \ln\!\left(\frac{en}{d_{\text{VC}}}\right)
$$

由于右端含 $n$，需迭代求解。数量级估计：

$$
\boxed{n = O\!\left(\frac{d_{\text{VC}} \ln(d_{\text{VC}}/\varepsilon) + \ln(1/\delta)}{\varepsilon^2}\right)}
$$

**（b）数值估计（$d=10, \varepsilon=0.05, \delta=0.05$）**

$d_{\text{VC}} = 2d = 20$。代入：

$$
n \gtrsim \frac{8}{\varepsilon^2}\left(\ln\frac{8}{\delta} + d_{\text{VC}} \ln\!\left(\frac{e \cdot n}{d_{\text{VC}}}\right)\right)
$$

初始估计取 $n = 10000$：

$$
\frac{8}{0.0025}\left(\ln 160 + 20 \ln\!\left(\frac{e \times 10000}{20}\right)\right) = 3200 \times (5.075 + 20 \times 6.908) \approx 3200 \times 143.2 \approx 458240
$$

精确迭代给出 $n \approx$ 数万量级（约 $10^4 \sim 10^5$）。关键是 $n$ 关于 $d$ 是**线性**的（$O(d/\varepsilon^2)$ 级别），而非指数级。

**（c）维度线性性的意义**

样本复杂度 $O(d/\varepsilon^2)$ 表明：即使特征维度 $d$ 增大，所需样本量也只是**线性增长**，而非指数增长（"维度灾难"通常指非参数方法中的 $O((1/\varepsilon)^d)$ 增长）。

从依概率收敛角度：这说明轴对齐矩形分类器类是 PAC 可学习的——随样本量 $n$ 的增大，经验风险以 $O_P(\sqrt{d/n})$ 的速率一致地（对整个假设类）依概率收敛到真实风险。在实践中，VC 维有限的假设类（包括大多数参数化模型）的泛化差距都具有这种多项式收敛速率，这正是机器学习理论保证神经网络泛化性能的数学基础。

</details>

---

*下一章预告*：[第13章：统计量与抽样分布](../part5-statistics-basics/13-sampling-distributions.md) — 我们将从概率论过渡到数理统计，学习总体、样本、统计量等核心概念，以及三大抽样分布（卡方、t、F分布）的性质与应用。
