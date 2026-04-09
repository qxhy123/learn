# 第4章 数列极限

## 学习目标

通过本章学习，你将能够：

- 理解数列的概念，掌握通项公式和递推公式两种表示方法
- 深刻理解数列极限的 $\varepsilon$-$N$ 定义，能用定义证明简单极限
- 掌握极限的唯一性、有界性、保号性等基本性质
- 熟练运用极限的四则运算法则和夹逼定理
- 理解单调有界定理，并用它证明数 $e$ 的存在性

---

## 4.1 数列的概念

### 4.1.1 数列的定义

**数列**是按照一定顺序排列的一列数。严格地说，数列是定义在正整数集 $\mathbb{N}^+$ 上的函数。

设 $f: \mathbb{N}^+ \to \mathbb{R}$，则 $f(1), f(2), f(3), \ldots$ 构成一个数列。我们通常记 $a_n = f(n)$，并把数列记为 $\{a_n\}$ 或 $\{a_n\}_{n=1}^{\infty}$。

- $a_n$ 称为数列的**通项**或**第 $n$ 项**
- $n$ 称为**项数**或**下标**

> **例题 4.1** 写出下列数列的前五项：
> (1) $a_n = \dfrac{1}{n}$
> (2) $a_n = (-1)^n$
> (3) $a_n = \dfrac{n}{n+1}$

**解**：
(1) $1, \dfrac{1}{2}, \dfrac{1}{3}, \dfrac{1}{4}, \dfrac{1}{5}, \ldots$

(2) $-1, 1, -1, 1, -1, \ldots$

(3) $\dfrac{1}{2}, \dfrac{2}{3}, \dfrac{3}{4}, \dfrac{4}{5}, \dfrac{5}{6}, \ldots$

### 4.1.2 数列的通项公式

如果能用一个关于 $n$ 的表达式直接表示 $a_n$，这个表达式称为数列的**通项公式**。

常见的数列类型：

- **等差数列**：$a_n = a_1 + (n-1)d$，其中 $d$ 是公差
- **等比数列**：$a_n = a_1 \cdot q^{n-1}$，其中 $q$ 是公比
- **调和数列**：$a_n = \dfrac{1}{n}$

### 4.1.3 递推数列

有些数列难以写出通项公式，但可以用前面的项来定义后面的项，这称为**递推公式**。

> **例题 4.2** 斐波那契数列定义如下：
> $$a_1 = 1, \quad a_2 = 1, \quad a_{n+2} = a_{n+1} + a_n \quad (n \geq 1)$$
> 写出该数列的前八项。

**解**：根据递推公式依次计算：
$$1, 1, 2, 3, 5, 8, 13, 21, \ldots$$

---

## 4.2 数列极限的定义

### 4.2.1 直观理解

观察数列 $a_n = \dfrac{1}{n}$：
$$1, \frac{1}{2}, \frac{1}{3}, \frac{1}{4}, \frac{1}{5}, \ldots$$

随着 $n$ 越来越大，$a_n$ 越来越接近 $0$。我们说这个数列"趋向于" $0$，或者说 $0$ 是这个数列的**极限**。

直观上，数列 $\{a_n\}$ 的极限是 $L$，意味着：当 $n$ 足够大时，$a_n$ 可以任意接近 $L$。

但"任意接近"是一个模糊的说法。到底多近才算"接近"？多大才算"足够大"？这就需要严格的数学定义。

### 4.2.2 $\varepsilon$-$N$ 定义（严格定义）

**定义**（数列极限）：设 $\{a_n\}$ 是一个数列，$L$ 是一个实数。如果对于**任意给定的** $\varepsilon > 0$，**都存在**正整数 $N$，使得当 $n > N$ 时，有
$$|a_n - L| < \varepsilon$$
则称数列 $\{a_n\}$ **收敛**于 $L$，记作
$$\lim_{n \to \infty} a_n = L \quad \text{或} \quad a_n \to L \;(n \to \infty)$$

若数列不收敛，则称它**发散**。

**用逻辑符号表述**：
$$\lim_{n \to \infty} a_n = L \iff \forall \varepsilon > 0, \exists N \in \mathbb{N}^+, \forall n > N: |a_n - L| < \varepsilon$$

**理解这个定义的要点**：

1. **$\varepsilon$ 是任意的**：不管你提出多么苛刻的精度要求（$\varepsilon$ 多么小），
2. **$N$ 是存在的**：总能找到一个"起点" $N$，
3. **从 $N$ 往后全部满足**：使得第 $N$ 项之后的所有项都落在 $(L-\varepsilon, L+\varepsilon)$ 这个区间内。

> **注意**：$N$ 通常依赖于 $\varepsilon$。$\varepsilon$ 越小，所需的 $N$ 通常越大。

### 4.2.3 几何解释

在数轴上，$|a_n - L| < \varepsilon$ 意味着 $a_n$ 落在以 $L$ 为中心、半径为 $\varepsilon$ 的邻域 $(L-\varepsilon, L+\varepsilon)$ 内。

极限的定义说的是：无论这个邻域多小，数列从某一项开始，所有的项都会落在这个邻域内。邻域外至多只有有限项。

> **例题 4.3** 用 $\varepsilon$-$N$ 定义证明：$\lim_{n \to \infty} \dfrac{1}{n} = 0$。

**解**：设 $\varepsilon > 0$ 是任意给定的正数。我们需要找到 $N$，使得当 $n > N$ 时，
$$\left|\frac{1}{n} - 0\right| = \frac{1}{n} < \varepsilon$$

由 $\dfrac{1}{n} < \varepsilon$ 得 $n > \dfrac{1}{\varepsilon}$。

因此，取 $N = \left\lfloor \dfrac{1}{\varepsilon} \right\rfloor$（取整），则当 $n > N$ 时，$n > \dfrac{1}{\varepsilon}$，从而 $\dfrac{1}{n} < \varepsilon$。

这就证明了 $\lim_{n \to \infty} \dfrac{1}{n} = 0$。 $\square$

> **例题 4.4** 用 $\varepsilon$-$N$ 定义证明：$\lim_{n \to \infty} \dfrac{n+1}{n} = 1$。

**解**：设 $\varepsilon > 0$ 是任意给定的。我们需要证明存在 $N$，使得当 $n > N$ 时，
$$\left|\frac{n+1}{n} - 1\right| < \varepsilon$$

计算：
$$\left|\frac{n+1}{n} - 1\right| = \left|\frac{n+1-n}{n}\right| = \frac{1}{n}$$

要使 $\dfrac{1}{n} < \varepsilon$，只需 $n > \dfrac{1}{\varepsilon}$。

取 $N = \left\lfloor \dfrac{1}{\varepsilon} \right\rfloor$，则当 $n > N$ 时，不等式成立。

因此 $\lim_{n \to \infty} \dfrac{n+1}{n} = 1$。 $\square$

---

## 4.3 极限的性质

### 4.3.1 唯一性

**定理**（极限的唯一性）：若数列 $\{a_n\}$ 收敛，则其极限唯一。

**证明**：反证法。假设 $\lim_{n \to \infty} a_n = L_1$ 且 $\lim_{n \to \infty} a_n = L_2$，其中 $L_1 \neq L_2$。

取 $\varepsilon = \dfrac{|L_1 - L_2|}{2} > 0$。

由极限定义，存在 $N_1$，当 $n > N_1$ 时，$|a_n - L_1| < \varepsilon$；
存在 $N_2$，当 $n > N_2$ 时，$|a_n - L_2| < \varepsilon$。

取 $N = \max\{N_1, N_2\}$，当 $n > N$ 时，由三角不等式：
$$|L_1 - L_2| = |L_1 - a_n + a_n - L_2| \leq |a_n - L_1| + |a_n - L_2| < \varepsilon + \varepsilon = 2\varepsilon = |L_1 - L_2|$$

这导出 $|L_1 - L_2| < |L_1 - L_2|$，矛盾。

因此极限必唯一。 $\square$

### 4.3.2 有界性

**定理**（收敛数列的有界性）：若数列 $\{a_n\}$ 收敛，则 $\{a_n\}$ 有界。

**证明**：设 $\lim_{n \to \infty} a_n = L$。取 $\varepsilon = 1$，则存在 $N$，当 $n > N$ 时，$|a_n - L| < 1$，即
$$L - 1 < a_n < L + 1$$

令 $M = \max\{|a_1|, |a_2|, \ldots, |a_N|, |L-1|, |L+1|\}$，则对所有 $n \in \mathbb{N}^+$，有 $|a_n| \leq M$。 $\square$

> **注意**：有界是收敛的**必要条件**，但不是充分条件。例如 $a_n = (-1)^n$ 有界但不收敛。

### 4.3.3 保号性

**定理**（保号性）：设 $\lim_{n \to \infty} a_n = L$。

(1) 若 $L > 0$，则存在 $N$，当 $n > N$ 时，$a_n > 0$。

(2) 若 $L < 0$，则存在 $N$，当 $n > N$ 时，$a_n < 0$。

**证明**：只证 (1)。设 $L > 0$，取 $\varepsilon = \dfrac{L}{2} > 0$。

由极限定义，存在 $N$，当 $n > N$ 时，$|a_n - L| < \dfrac{L}{2}$，即
$$L - \frac{L}{2} < a_n < L + \frac{L}{2}$$

因此 $a_n > \dfrac{L}{2} > 0$。 $\square$

**推论**（保号性的逆命题）：若从某项起 $a_n \geq 0$（或 $a_n \leq 0$），且 $\lim_{n \to \infty} a_n = L$，则 $L \geq 0$（或 $L \leq 0$）。

---

## 4.4 极限的运算法则

### 4.4.1 四则运算

**定理**（极限的四则运算）：设 $\lim_{n \to \infty} a_n = A$，$\lim_{n \to \infty} b_n = B$，则：

1. $\lim_{n \to \infty} (a_n \pm b_n) = A \pm B$
2. $\lim_{n \to \infty} (a_n \cdot b_n) = A \cdot B$
3. $\lim_{n \to \infty} \dfrac{a_n}{b_n} = \dfrac{A}{B}$（要求 $B \neq 0$ 且 $b_n \neq 0$）

此外，若 $c$ 是常数，则 $\lim_{n \to \infty} c \cdot a_n = c \cdot A$。

> **例题 4.5** 求 $\lim_{n \to \infty} \dfrac{3n^2 + 2n - 1}{2n^2 - n + 5}$。

**解**：分子分母同除以 $n^2$：
$$\lim_{n \to \infty} \frac{3n^2 + 2n - 1}{2n^2 - n + 5} = \lim_{n \to \infty} \frac{3 + \frac{2}{n} - \frac{1}{n^2}}{2 - \frac{1}{n} + \frac{5}{n^2}}$$

由于 $\lim_{n \to \infty} \dfrac{1}{n} = 0$，$\lim_{n \to \infty} \dfrac{1}{n^2} = 0$，运用四则运算法则：
$$= \frac{3 + 0 - 0}{2 - 0 + 0} = \frac{3}{2}$$

> **例题 4.6** 求 $\lim_{n \to \infty} \dfrac{2^n + 3^n}{3^n + 4^n}$。

**解**：分子分母同除以 $4^n$（最大的指数底）：
$$\lim_{n \to \infty} \frac{2^n + 3^n}{3^n + 4^n} = \lim_{n \to \infty} \frac{\left(\frac{2}{4}\right)^n + \left(\frac{3}{4}\right)^n}{\left(\frac{3}{4}\right)^n + 1} = \lim_{n \to \infty} \frac{\left(\frac{1}{2}\right)^n + \left(\frac{3}{4}\right)^n}{\left(\frac{3}{4}\right)^n + 1}$$

由于 $\left(\dfrac{1}{2}\right)^n \to 0$，$\left(\dfrac{3}{4}\right)^n \to 0$：
$$= \frac{0 + 0}{0 + 1} = 0$$

### 4.4.2 夹逼定理

**定理**（夹逼定理 / 三明治定理）：设数列 $\{a_n\}$、$\{b_n\}$、$\{c_n\}$ 满足：

1. 从某项起，$a_n \leq b_n \leq c_n$
2. $\lim_{n \to \infty} a_n = \lim_{n \to \infty} c_n = L$

则 $\lim_{n \to \infty} b_n = L$。

**证明**：设对 $n > N_0$，有 $a_n \leq b_n \leq c_n$。

对任意 $\varepsilon > 0$，由于 $\lim a_n = L$，存在 $N_1$，当 $n > N_1$ 时，$|a_n - L| < \varepsilon$，即 $L - \varepsilon < a_n$。

同理，存在 $N_2$，当 $n > N_2$ 时，$c_n < L + \varepsilon$。

取 $N = \max\{N_0, N_1, N_2\}$，当 $n > N$ 时：
$$L - \varepsilon < a_n \leq b_n \leq c_n < L + \varepsilon$$

即 $|b_n - L| < \varepsilon$。 $\square$

> **例题 4.7** 求 $\lim_{n \to \infty} \dfrac{\sin n}{n}$。

**解**：由于 $-1 \leq \sin n \leq 1$，有
$$-\frac{1}{n} \leq \frac{\sin n}{n} \leq \frac{1}{n}$$

而 $\lim_{n \to \infty} \left(-\dfrac{1}{n}\right) = 0$，$\lim_{n \to \infty} \dfrac{1}{n} = 0$。

由夹逼定理，$\lim_{n \to \infty} \dfrac{\sin n}{n} = 0$。

> **例题 4.8** 求 $\lim_{n \to \infty} \sqrt[n]{n}$。

**解**：设 $a_n = \sqrt[n]{n} - 1$，则 $\sqrt[n]{n} = 1 + a_n$，且当 $n \geq 2$ 时 $a_n > 0$。

由二项式展开：
$$n = (1 + a_n)^n \geq 1 + \binom{n}{2} a_n^2 = 1 + \frac{n(n-1)}{2} a_n^2$$

因此 $\dfrac{n(n-1)}{2} a_n^2 \leq n - 1$，即 $a_n^2 \leq \dfrac{2}{n}$。

由 $0 \leq a_n \leq \sqrt{\dfrac{2}{n}} \to 0$，利用夹逼定理得 $a_n \to 0$。

因此 $\lim_{n \to \infty} \sqrt[n]{n} = 1 + 0 = 1$。

---

## 4.5 单调有界定理

### 4.5.1 定理内容

**定理**（单调有界定理）：单调有界数列必收敛。

具体地说：
- 单调递增且有上界的数列必收敛
- 单调递减且有下界的数列必收敛

**证明思路**：以单调递增有上界为例。

设 $\{a_n\}$ 单调递增且有上界。由确界原理，集合 $\{a_n : n \in \mathbb{N}^+\}$ 有上确界，设为 $L = \sup\{a_n\}$。

对任意 $\varepsilon > 0$，由上确界定义，存在某个 $a_N$ 使得 $a_N > L - \varepsilon$。

由于数列单调递增，当 $n > N$ 时，$a_n \geq a_N > L - \varepsilon$。

又因为 $L$ 是上界，$a_n \leq L < L + \varepsilon$。

因此当 $n > N$ 时，$L - \varepsilon < a_n < L + \varepsilon$，即 $|a_n - L| < \varepsilon$。 $\square$

> **注意**：这个定理的意义在于，它可以判定极限的**存在性**，而不需要事先知道极限值。

### 4.5.2 重要应用：数 $e$ 的定义

**例题 4.9** 证明数列 $a_n = \left(1 + \dfrac{1}{n}\right)^n$ 收敛，并将其极限定义为 $e$。

**证明**：我们分两步进行。

**第一步：证明 $\{a_n\}$ 单调递增。**

利用算术-几何平均不等式。考虑 $n+1$ 个正数：
$$\underbrace{1 + \frac{1}{n}, 1 + \frac{1}{n}, \ldots, 1 + \frac{1}{n}}_{n \text{ 个}}, \quad 1$$

它们的算术平均值为：
$$\frac{n \cdot \left(1 + \frac{1}{n}\right) + 1}{n+1} = \frac{n + 1 + 1}{n+1} = \frac{n+2}{n+1} = 1 + \frac{1}{n+1}$$

几何平均值为：
$$\sqrt[n+1]{\left(1 + \frac{1}{n}\right)^n \cdot 1} = \sqrt[n+1]{a_n}$$

由算术-几何平均不等式（几何平均 $\leq$ 算术平均）：
$$\sqrt[n+1]{a_n} < 1 + \frac{1}{n+1}$$

即 $a_n < \left(1 + \dfrac{1}{n+1}\right)^{n+1} = a_{n+1}$。因此 $\{a_n\}$ 单调递增。

**第二步：证明 $\{a_n\}$ 有上界。**

由二项式定理：
$$a_n = \left(1 + \frac{1}{n}\right)^n = \sum_{k=0}^{n} \binom{n}{k} \frac{1}{n^k}$$

展开后：
$$a_n = 1 + 1 + \frac{1}{2!}\left(1 - \frac{1}{n}\right) + \frac{1}{3!}\left(1 - \frac{1}{n}\right)\left(1 - \frac{2}{n}\right) + \cdots$$

由于每个括号中的因子都小于 $1$：
$$a_n < 1 + 1 + \frac{1}{2!} + \frac{1}{3!} + \cdots + \frac{1}{n!}$$

又因为 $k! \geq 2^{k-1}$（当 $k \geq 1$），所以：
$$a_n < 1 + 1 + \frac{1}{2} + \frac{1}{4} + \cdots + \frac{1}{2^{n-1}} < 1 + \frac{1}{1 - \frac{1}{2}} = 3$$

因此 $a_n < 3$，数列有上界。

**结论**：由单调有界定理，$\{a_n\}$ 收敛。我们将其极限定义为自然常数 $e$：
$$e = \lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n \approx 2.71828\ldots$$

数 $e$ 是微积分中最重要的常数之一，它将在指数函数、对数函数的导数以及微分方程中反复出现。

---

## 本章小结

1. **数列**是定义在正整数集上的函数，可以用通项公式或递推公式表示。

2. **数列极限的 $\varepsilon$-$N$ 定义**是理解极限的核心：对任意 $\varepsilon > 0$，存在 $N$，当 $n > N$ 时 $|a_n - L| < \varepsilon$。这个定义精确刻画了"任意接近"的含义。

3. **极限的基本性质**包括：
   - 唯一性：收敛数列的极限唯一
   - 有界性：收敛数列必有界（反之不然）
   - 保号性：极限的符号决定了数列最终的符号

4. **极限运算法则**：收敛数列的和、差、积、商（分母极限非零）的极限等于极限的和、差、积、商。

5. **夹逼定理**：若 $a_n \leq b_n \leq c_n$ 且 $\lim a_n = \lim c_n = L$，则 $\lim b_n = L$。

6. **单调有界定理**：单调有界数列必收敛。这是判定极限存在性的重要工具，数 $e$ 正是通过这个定理定义的。

---

## 深度学习应用

数列极限的概念在深度学习中有直接且重要的应用。神经网络的训练本质上是一个迭代优化过程，每次迭代产生一个新的参数状态，形成一个序列。理解这些序列的收敛行为，是分析训练过程稳定性的数学基础。

### 训练过程的收敛性

神经网络训练中，每轮（epoch）计算一次损失值，形成**损失序列**：
$$\{L_1, L_2, \ldots, L_n, \ldots\}$$

训练收敛意味着损失序列有极限：
$$\lim_{n \to \infty} L_n = L^*$$

其中 $L^*$ 是模型在当前优化路径上所能达到的最优损失值。

在实践中，我们无法等待无穷步，因此用以下条件判断**近似收敛**：
$$|L_{n+1} - L_n| < \varepsilon$$

即相邻两次损失的变化量小于阈值 $\varepsilon$。这正是 $\varepsilon$-$N$ 定义中"任意接近"思想的工程化实现。

**早停（Early Stopping）策略**是基于此思想的正则化技术：监控验证集损失序列，当连续 $p$（patience）轮内损失改善量小于 $\delta$（min\_delta）时，判定序列已收敛并停止训练，以防止过拟合。

### 学习率调度

梯度下降的更新步长由学习率 $\eta_n$ 控制，其本身也构成一个序列。**指数衰减**是最常见的调度方案：
$$\eta_n = \eta_0 \cdot \gamma^n \quad (0 < \gamma < 1)$$

这是一个公比为 $\gamma$ 的等比数列，满足 $\lim_{n \to \infty} \eta_n = 0$。

随机梯度下降（SGD）理论中，参数序列收敛的充分条件是学习率序列满足 **Robbins-Monro 条件**：
$$\sum_{n=1}^{\infty} \eta_n = \infty, \qquad \sum_{n=1}^{\infty} \eta_n^2 < \infty$$

第一个条件保证学习率的累积步长足够大，使优化能到达任意目标点；第二个条件保证随机噪声的累积影响有限，从而序列不会因噪声而发散。指数衰减序列满足第二个条件（几何级数收敛），但不满足第一个条件（求和趋于有限值），因此常与动量等技巧配合使用。

### 指数移动平均（EMA）

**指数移动平均**是深度学习中广泛使用的平滑技术，其递推公式为：
$$v_n = \beta v_{n-1} + (1 - \beta)\,\theta_n \quad (0 < \beta < 1)$$

其中 $\theta_n$ 是第 $n$ 步的原始参数（或梯度），$\beta$ 是衰减系数。将递推展开：
$$v_n = (1-\beta)\sum_{k=1}^{n} \beta^{n-k}\,\theta_k$$

这是一个加权求和，权重 $\beta^{n-k}$ 随时间差指数衰减，越近的观测权重越大。

**极限行为分析**：设 $\theta_n \to \theta^*$，则 $v_n$ 也收敛于 $\theta^*$。直观上，由于 $\sum_{k=0}^{\infty}(1-\beta)\beta^k = 1$（几何级数），EMA 是对历史值的凸组合，极限与原序列一致。Adam、RMSProp 等优化器内部均使用 EMA 来估计梯度的一阶和二阶矩。

### 代码示例

```python
import torch
import torch.nn as nn

# 学习率调度器示例
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 指数衰减：eta_n = eta_0 * gamma^n
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# 训练收敛判断
losses = []
patience, min_delta = 10, 1e-4

for epoch in range(100):
    loss = torch.rand(1)  # 模拟损失
    losses.append(loss.item())

    # 早停判断：损失序列是否收敛
    if len(losses) > patience:
        recent = losses[-patience:]
        if max(recent) - min(recent) < min_delta:
            print(f"收敛于 epoch {epoch}")
            break

    scheduler.step()
```

> **数学联系**：上述代码中，`max(recent) - min(recent) < min_delta` 正是在验证：损失序列在最近 `patience` 项组成的子列中，振幅小于阈值。这与 Cauchy 收敛准则（数列收敛当且仅当它是 Cauchy 列）的工程近似直接对应。

---

## 练习题

**1.** ⭐ 用 $\varepsilon$-$N$ 定义证明：$\lim_{n \to \infty} \dfrac{2n-1}{n+1} = 2$。

**2.** ⭐ 求极限：$\lim_{n \to \infty} \dfrac{n^2 - 2n + 3}{2n^2 + n - 1}$。

**3.** ⭐⭐ 利用夹逼定理求：$\lim_{n \to \infty} \left(\dfrac{1}{n^2+1} + \dfrac{1}{n^2+2} + \cdots + \dfrac{1}{n^2+n}\right)$。

**4.** ⭐⭐ 设 $a_1 = \sqrt{2}$，$a_{n+1} = \sqrt{2 + a_n}$（$n \geq 1$）。证明数列 $\{a_n\}$ 收敛，并求其极限。

**5.** ⭐⭐⭐ 设 $a_1 > 0$，$a_{n+1} = \dfrac{1}{2}\left(a_n + \dfrac{2}{a_n}\right)$。证明 $\{a_n\}$ 从第二项起单调递减且有下界，并求 $\lim_{n \to \infty} a_n$。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.** 设 $\varepsilon > 0$。计算：
$$\left|\frac{2n-1}{n+1} - 2\right| = \left|\frac{2n-1 - 2(n+1)}{n+1}\right| = \left|\frac{-3}{n+1}\right| = \frac{3}{n+1}$$

要使 $\dfrac{3}{n+1} < \varepsilon$，需要 $n > \dfrac{3}{\varepsilon} - 1$。

取 $N = \left\lfloor \dfrac{3}{\varepsilon} \right\rfloor$，则当 $n > N$ 时，$\left|\dfrac{2n-1}{n+1} - 2\right| < \varepsilon$。

因此 $\lim_{n \to \infty} \dfrac{2n-1}{n+1} = 2$。 $\square$

---

**2.** 分子分母同除以 $n^2$：
$$\lim_{n \to \infty} \frac{n^2 - 2n + 3}{2n^2 + n - 1} = \lim_{n \to \infty} \frac{1 - \frac{2}{n} + \frac{3}{n^2}}{2 + \frac{1}{n} - \frac{1}{n^2}} = \frac{1 - 0 + 0}{2 + 0 - 0} = \frac{1}{2}$$

---

**3.** 设 $S_n = \dfrac{1}{n^2+1} + \dfrac{1}{n^2+2} + \cdots + \dfrac{1}{n^2+n}$。

由于 $n^2 + 1 \leq n^2 + k \leq n^2 + n$ 对 $k = 1, 2, \ldots, n$ 成立，有：
$$\frac{n}{n^2+n} \leq S_n \leq \frac{n}{n^2+1}$$

即：
$$\frac{1}{n+1} \leq S_n \leq \frac{n}{n^2+1}$$

由于 $\lim_{n \to \infty} \dfrac{1}{n+1} = 0$，$\lim_{n \to \infty} \dfrac{n}{n^2+1} = \lim_{n \to \infty} \dfrac{1}{n + \frac{1}{n}} = 0$。

由夹逼定理，$\lim_{n \to \infty} S_n = 0$。

---

**4.** **第一步**：证明 $\{a_n\}$ 有界。

用数学归纳法证明 $a_n < 2$。$a_1 = \sqrt{2} < 2$。设 $a_n < 2$，则：
$$a_{n+1} = \sqrt{2 + a_n} < \sqrt{2 + 2} = 2$$

因此对所有 $n$，$a_n < 2$。

**第二步**：证明 $\{a_n\}$ 单调递增。

$$a_{n+1}^2 - a_n^2 = (2 + a_n) - a_n^2 = -(a_n^2 - a_n - 2) = -(a_n - 2)(a_n + 1)$$

由于 $a_n < 2$ 且 $a_n > 0$，有 $(a_n - 2) < 0$，$(a_n + 1) > 0$。

因此 $a_{n+1}^2 - a_n^2 > 0$，即 $a_{n+1} > a_n$。

**第三步**：由单调有界定理，$\{a_n\}$ 收敛。设 $\lim a_n = L$，则在递推公式 $a_{n+1} = \sqrt{2 + a_n}$ 两边取极限：
$$L = \sqrt{2 + L}$$

解方程：$L^2 = 2 + L$，即 $L^2 - L - 2 = 0$，$(L-2)(L+1) = 0$。

由于 $L > 0$，得 $L = 2$。 $\square$

---

**5.** **第一步**：证明 $a_n > 0$ 且 $a_n \geq \sqrt{2}$（$n \geq 2$）。

由递推公式，若 $a_n > 0$，则 $a_{n+1} > 0$。由 $a_1 > 0$，归纳得 $a_n > 0$。

对于 $n \geq 1$，由算术-几何平均不等式：
$$a_{n+1} = \frac{1}{2}\left(a_n + \frac{2}{a_n}\right) \geq \sqrt{a_n \cdot \frac{2}{a_n}} = \sqrt{2}$$

**第二步**：证明从第二项起单调递减。

对 $n \geq 2$（此时 $a_n \geq \sqrt{2}$）：
$$a_{n+1} - a_n = \frac{1}{2}\left(a_n + \frac{2}{a_n}\right) - a_n = \frac{1}{2}\left(\frac{2}{a_n} - a_n\right) = \frac{2 - a_n^2}{2a_n}$$

由于 $a_n \geq \sqrt{2}$，有 $a_n^2 \geq 2$，因此 $a_{n+1} - a_n \leq 0$。

**第三步**：由单调有界定理，$\{a_n\}$ 收敛。设 $\lim a_n = L$，在递推公式两边取极限：
$$L = \frac{1}{2}\left(L + \frac{2}{L}\right)$$

解得 $2L = L + \dfrac{2}{L}$，即 $L = \dfrac{2}{L}$，$L^2 = 2$。

由于 $L > 0$，得 $L = \sqrt{2}$。 $\square$

</details>
