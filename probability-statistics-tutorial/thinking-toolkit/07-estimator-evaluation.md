# 估计量评价四标尺

> **一例速记**：设 $X_1,\ldots,X_n$ i.i.d. $\sim N(\mu, \sigma^2)$，考察样本均值 $\bar{X}$ 估计 $\mu$ 的质量。
> 无偏性：$E\bar{X} = \mu$ ✓；一致性：$\text{Var}(\bar{X}) = \sigma^2/n \to 0$ ✓；
> 有效性：能达到 Fisher 信息给出的方差下界吗？正态情形 $I(\mu) = n/\sigma^2$，$1/I = \sigma^2/n = \text{Var}(\bar{X})$ ✓。
> **$\bar{X}$ 对正态均值是无偏、一致、有效的估计量——它是最优估计的标杆。**

---

## 一、为什么需要评价标准

拿到样本后，我们可以构造形形色色的估计量：$\bar{X}$、中位数、最小值、第一个观测值 $X_1$……它们都是 $\mu$ 的某种猜测。凭什么说某个估计量"好"？

判断估计量好坏需要**客观标准**，而且不同场景侧重不同。"无偏"关注估计的系统偏差；"一致"关注数据多了以后能不能收敛；"有效"关注精度的绝对上限；Cramér-Rao 下界（CRB）则给出了任何无偏估计量都无法突破的精度天花板。

---

## 二、四大评价标准

### 2.1 无偏性（Unbiasedness）

**定义**：若 $E_\theta(\hat\theta) = \theta$（对所有 $\theta$），称 $\hat\theta$ 为 $\theta$ 的**无偏估计量**。

**偏差**（Bias）：$\text{Bias}(\hat\theta) = E(\hat\theta) - \theta$。无偏即偏差为零。

**直觉**：在大量重复实验中，估计值围绕真值波动，平均而言不高估也不低估。

**经典例子**：

- 样本均值 $\bar{X} = \frac{1}{n}\sum X_i$：$E\bar{X} = \mu$，无偏。
- 样本方差 $S^2 = \frac{1}{n-1}\sum(X_i-\bar{X})^2$：$E(S^2) = \sigma^2$，无偏。
- 但 $\hat\sigma^2 = \frac{1}{n}\sum(X_i-\bar{X})^2$：$E(\hat\sigma^2) = \frac{n-1}{n}\sigma^2 \neq \sigma^2$，有偏（系统低估）。

**注意**：无偏并不等于"好"。$X_1$（只取第一个观测值）是 $\mu$ 的无偏估计，但方差为 $\sigma^2$，远大于 $\bar{X}$ 的方差 $\sigma^2/n$。

**渐近无偏**（Asymptotically Unbiased）：若 $\text{Bias}(\hat\theta_n) \to 0$（$n\to\infty$），称为渐近无偏。$\hat\sigma^2$ 是渐近无偏的例子。

### 2.2 一致性（Consistency）

**定义**：若 $\hat\theta_n \xrightarrow{P} \theta$（依概率收敛），称 $\hat\theta_n$ 为 $\theta$ 的**一致（相合）估计量**。

等价条件（常用于证明）：若同时满足
- $\text{Bias}(\hat\theta_n) \to 0$，以及
- $\text{Var}(\hat\theta_n) \to 0$，

则 $\hat\theta_n$ 一致（因 MSE $= \text{Bias}^2 + \text{Var} \to 0$，结合 Chebyshev）。

**直觉**：数据越多，估计越精准，最终完全逼近真值。这是统计估计的最基本要求——如果样本无限多时估计量还不收敛到真值，这个估计方法就根本不可靠。

**例子**：

- $\bar{X}_n$：$\text{Var}(\bar{X}_n) = \sigma^2/n \to 0$，故一致。
- $\hat\sigma^2 = \frac{1}{n}\sum(X_i-\bar{X})^2$：虽然有偏，但 $\text{Bias} \to 0$、$\text{Var} \to 0$，故仍是 $\sigma^2$ 的一致估计。

### 2.3 有效性（Efficiency）

**定义**（在无偏估计类中）：若 $\hat\theta_1, \hat\theta_2$ 均为 $\theta$ 的无偏估计量，且 $\text{Var}(\hat\theta_1) \leq \text{Var}(\hat\theta_2)$（对所有 $\theta$），称 $\hat\theta_1$ 比 $\hat\theta_2$ **更有效**。

**相对效率**：$e(\hat\theta_1, \hat\theta_2) = \dfrac{\text{Var}(\hat\theta_2)}{\text{Var}(\hat\theta_1)}$。$e > 1$ 表示 $\hat\theta_1$ 更有效。

**均方误差（MSE）**：若比较的估计量不都是无偏的，改用 MSE 比较更公平：

$$\text{MSE}(\hat\theta) = E(\hat\theta - \theta)^2 = \text{Var}(\hat\theta) + [\text{Bias}(\hat\theta)]^2.$$

**均匀最小方差无偏估计量（UMVUE）**：在所有无偏估计量中，方差最小且对所有 $\theta$ 都成立的估计量。

### 2.4 小结对比

| 标准 | 数学定义 | 核心问题 |
|---|---|---|
| 无偏性 | $E(\hat\theta) = \theta$ | 系统是否偏移？ |
| 一致性 | $\hat\theta_n \xrightarrow{P} \theta$ | 数据越多越准吗？ |
| 有效性 | $\text{Var}(\hat\theta)$ 最小（无偏类中）| 精度是否最优？ |
| CRB 可达 | $\text{Var}(\hat\theta) = 1/I(\theta)$ | 是否达到理论极限？ |

---

## 三、Cramér-Rao 下界与 Fisher 信息

### 3.1 Fisher 信息

**Fisher 信息量** $I(\theta)$ 度量了样本数据关于参数 $\theta$ 所携带的"信息量"：

$$I(\theta) = E_\theta\!\left[\left(\frac{\partial}{\partial\theta}\ln f(X;\theta)\right)^2\right] = -E_\theta\!\left[\frac{\partial^2}{\partial\theta^2}\ln f(X;\theta)\right].$$

（两个公式等价，后者有时更易计算；需要正则条件：对数密度对 $\theta$ 的二阶导数与期望可以交换。）

对 $n$ 个 i.i.d. 样本：$I_n(\theta) = n\cdot I_1(\theta)$，Fisher 信息随样本量线性增长。

**计分函数**（Score function）：$s(\theta; x) = \dfrac{\partial}{\partial\theta}\ln f(x;\theta)$，满足 $E_\theta[s(\theta;X)] = 0$，$\text{Var}(s) = I(\theta)$。

### 3.2 Cramér-Rao 下界（CRB）

**定理**：设 $\hat\theta$ 是 $\theta$ 的无偏估计量，在正则条件下：

$$\text{Var}_\theta(\hat\theta) \geq \frac{1}{I_n(\theta)} = \frac{1}{n \cdot I_1(\theta)}.$$

**解读**：这是方差的"物理极限"——无论多聪明的无偏估计方法，方差都不可能低于 $1/I_n(\theta)$。若某估计量的方差恰好等于下界，称其为**有效估计量**（efficient estimator）。

**达到 CRB 的充要条件**：计分函数与 $\hat\theta - \theta$ 成线性关系，即

$$\frac{\partial}{\partial\theta}\ln f(\mathbf{x};\theta) = A(\theta)\left[\hat\theta(\mathbf{x}) - \theta\right],$$

此时 $\hat\theta$ 是有效估计量，且 $\text{Var}(\hat\theta) = 1/(n\cdot I_1(\theta))$。

### 3.3 常见分布的 Fisher 信息

| 分布 | 参数 $\theta$ | $I_1(\theta)$ | $1/I_n(\theta) = \sigma^2/n$ |
|---|---|---|---|
| $N(\mu, \sigma^2)$（$\sigma^2$ 已知）| $\mu$ | $1/\sigma^2$ | $\sigma^2/n$ |
| $N(\mu, \sigma^2)$（$\mu$ 已知）| $\sigma^2$ | $1/(2\sigma^4)$ | $2\sigma^4/n$ |
| $\text{Bernoulli}(p)$ | $p$ | $1/(p(1-p))$ | $p(1-p)/n$ |
| $\text{Poisson}(\lambda)$ | $\lambda$ | $1/\lambda$ | $\lambda/n$ |
| $\text{Exp}(\lambda)$ | $\lambda$ | $1/\lambda^2$ | $\lambda^2/n$ |

---

## 四、演示题：样本均值 vs 样本中位数估计正态均值

**题目**：设 $X_1,\ldots,X_n$ i.i.d. $\sim N(\mu, \sigma^2)$，比较样本均值 $\bar{X}$ 和样本中位数 $\tilde{X}$ 估计 $\mu$ 的有效性。

> **分析框架。** 两个估计量都是无偏的（$N(\mu,\sigma^2)$ 关于 $\mu$ 对称，中位数的期望等于对称中心 $\mu$）。比较有效性就是比较方差。
>
> **$\bar{X}$ 的方差。**
>
> $$\text{Var}(\bar{X}) = \frac{\sigma^2}{n}.$$
>
> **$\tilde{X}$ 的方差（大样本近似）。** 利用顺序统计量理论，样本中位数（$n$ 为奇数时取中间值）的渐近方差为
>
> $$\text{Var}(\tilde{X}) \approx \frac{1}{4n[f(\mu)]^2},$$
>
> 其中 $f(\mu)$ 是总体在 $\mu$ 处的密度值。对 $N(\mu,\sigma^2)$：
>
> $$f(\mu) = \frac{1}{\sqrt{2\pi}\,\sigma}, \quad [f(\mu)]^2 = \frac{1}{2\pi\sigma^2}.$$
>
> $$\text{Var}(\tilde{X}) \approx \frac{1}{4n \cdot \frac{1}{2\pi\sigma^2}} = \frac{\pi\sigma^2}{2n}.$$
>
> **相对效率。**
>
> $$e(\bar{X}, \tilde{X}) = \frac{\text{Var}(\tilde{X})}{\text{Var}(\bar{X})} = \frac{\pi\sigma^2/(2n)}{\sigma^2/n} = \frac{\pi}{2} \approx 1.571.$$
>
> **结论。** 样本中位数的方差约是样本均值方差的 $\pi/2 \approx 1.57$ 倍。用中位数估计正态均值，需要约多用 $57\%$ 的样本量才能达到与均值相同的精度。
>
> **为何均值更优？** 对正态分布，均值就是 UMVUE，其方差 $\sigma^2/n$ 恰好等于 Cramér-Rao 下界（$1/I_n(\mu) = \sigma^2/n$），因此没有无偏估计量能比它更有效。中位数不利用分布的参数结构，"浪费"了数据中的分布信息。
>
> **权衡。** 然而在重尾分布（如 Cauchy）或数据有离群值时，中位数可能远比均值稳健（robust）——此时有效性与稳健性之间需要权衡。

---

## 五、思考路标

**路标 1**：验证无偏性时，不要只看形式，要**计算期望**：$E(\hat\theta) = ?$。常见陷阱：$\frac{1}{n}\sum(X_i - \bar{X})^2$ 的期望是 $\frac{n-1}{n}\sigma^2$，不是 $\sigma^2$。

**路标 2**：证明一致性的最快路线：计算 $\text{Bias}^2 + \text{Var}$（即 MSE），若 MSE $\to 0$，则依 Chebyshev 得 $\hat\theta_n \xrightarrow{P} \theta$。

**路标 3**：比较两个无偏估计量的有效性 → 直接比较方差，方差小的更有效。若不都是无偏的，改用 MSE 比较。

**路标 4**：计算 Fisher 信息 $I(\theta)$ 时，记住两个等价公式，选算起来更容易的那个。对指数族分布，通常用第二个公式（对数密度的二阶偏导取期望再取负）更简单。

**路标 5**：若问"某估计量是否有效"，先算 CRB $= 1/I_n(\theta)$，再与估计量的方差比较。相等 → 有效；更大 → 有改进空间但不一定找得到更好的（可能 CRB 不可达）。

**路标 6**：注意无偏、一致、有效三者的逻辑关系：无偏不蕴含一致（$X_1$ 无偏但不一致），一致不蕴含无偏（$\hat\sigma^2$ 有偏但一致），有效蕴含无偏（有效的定义是在无偏类中方差最小）。

---

## 六、典型应用 3 例

### 例 1：验证无偏性与一致性

**题目**：设 $X_1,\ldots,X_n$ i.i.d. $\sim \text{Exp}(\lambda)$（均值 $1/\lambda$），验证 $\hat\lambda = 1/\bar{X}$ 的无偏性和一致性。

**无偏性**：$E(1/\bar{X}) \neq 1/E(\bar{X})$（Jensen 不等式，$1/x$ 是凸函数），$\hat\lambda = 1/\bar{X}$ 是**有偏**的。具体地，可以证明（$n \geq 2$ 时）$E(1/\bar{X}) = n\lambda/(n-1) \neq \lambda$，存在正偏差 $\lambda/(n-1)$。

**一致性**：$\bar{X} \xrightarrow{P} 1/\lambda$（LLN），再由连续映射定理，$1/\bar{X} \xrightarrow{P} \lambda$，故 $\hat\lambda$ 是一致的。

**结论**：$\hat\lambda$ 有偏但一致，这是"有偏但一致"的典型例子，强调了无偏与一致的独立性。

---

### 例 2：计算 Fisher 信息与 CRB

**题目**：设 $X_1,\ldots,X_n$ i.i.d. $\sim \text{Poisson}(\lambda)$，计算 Fisher 信息 $I_1(\lambda)$，并给出 $\lambda$ 的无偏估计量方差的下界。

**Fisher 信息**：$\ln f(x;\lambda) = x\ln\lambda - \lambda - \ln(x!)$。

$$\frac{\partial^2}{\partial\lambda^2}\ln f = -\frac{x}{\lambda^2}, \quad -E\!\left[-\frac{X}{\lambda^2}\right] = \frac{E(X)}{\lambda^2} = \frac{\lambda}{\lambda^2} = \frac{1}{\lambda}.$$

故 $I_1(\lambda) = 1/\lambda$，$I_n(\lambda) = n/\lambda$，CRB $= \lambda/n$。

$\bar{X}$ 是 $\lambda$ 的无偏估计，$\text{Var}(\bar{X}) = \lambda/n$，**恰好达到 CRB**，故 $\bar{X}$ 是 Poisson 均值的有效估计量。

---

### 例 3：MSE 比较有偏与无偏估计

**题目**：已知 $X_1,\ldots,X_n$ i.i.d. $\sim N(\mu, \sigma^2)$，比较 $S^2 = \frac{1}{n-1}\sum(X_i-\bar{X})^2$ 和 $\hat\sigma^2 = \frac{1}{n}\sum(X_i-\bar{X})^2$ 估计 $\sigma^2$ 的 MSE。

设 $Q = \sum(X_i - \bar{X})^2$，则 $Q/\sigma^2 \sim \chi^2(n-1)$，$E(Q) = (n-1)\sigma^2$，$\text{Var}(Q) = 2(n-1)\sigma^4$。

**对 $S^2 = Q/(n-1)$**（无偏）：$\text{Var}(S^2) = \dfrac{2\sigma^4}{n-1}$，$\text{Bias}=0$，MSE $= \dfrac{2\sigma^4}{n-1}$。

**对 $\hat\sigma^2 = Q/n$**（有偏）：$E(\hat\sigma^2) = \dfrac{n-1}{n}\sigma^2$，Bias $= -\sigma^2/n$，$\text{Var}(\hat\sigma^2) = \dfrac{2(n-1)\sigma^4}{n^2}$。

$$\text{MSE}(\hat\sigma^2) = \frac{2(n-1)\sigma^4}{n^2} + \frac{\sigma^4}{n^2} = \frac{(2n-1)\sigma^4}{n^2}.$$

比较：$\text{MSE}(S^2) = \dfrac{2\sigma^4}{n-1} \approx \dfrac{2\sigma^4}{n}$；$\text{MSE}(\hat\sigma^2) = \dfrac{(2n-1)\sigma^4}{n^2} \approx \dfrac{2\sigma^4}{n}$。

精确比较：$\text{MSE}(\hat\sigma^2) - \text{MSE}(S^2) = \sigma^4\!\left[\dfrac{2n-1}{n^2} - \dfrac{2}{n-1}\right] = \sigma^4 \cdot \dfrac{(2n-1)(n-1) - 2n^2}{n^2(n-1)} = \sigma^4 \cdot \dfrac{-1}{n^2(n-1)} < 0$。

故 $\text{MSE}(\hat\sigma^2) < \text{MSE}(S^2)$：有偏估计量 $\hat\sigma^2$ 的 MSE 反而更小！这说明，若以 MSE 为准绳，有偏估计量有时优于无偏的。

---

## 七、自测题

**第 1 题**：设 $X_1,\ldots,X_n$ i.i.d.，$E(X_i) = \mu$，$E(X_i^2) = \mu^2 + \sigma^2$。验证 $\hat\sigma^2 = \frac{1}{n}\sum X_i^2 - \bar{X}^2$ 是 $\sigma^2$ 的无偏估计量还是有偏估计量（写出计算过程）。

> 提示：$E(\hat\sigma^2) = E(X_i^2) - E(\bar{X}^2) = (\mu^2 + \sigma^2) - (\mu^2 + \sigma^2/n) = \sigma^2(1 - 1/n) = \sigma^2(n-1)/n$，有偏，偏差为 $-\sigma^2/n$。

**第 2 题**：设 $X_1,\ldots,X_n$ i.i.d. $\sim U(0, \theta)$（均匀分布），$\hat\theta_1 = 2\bar{X}$，$\hat\theta_2 = \frac{n+1}{n}X_{(n)}$（$X_{(n)}$ 是最大顺序统计量）。验证两者都无偏，并比较方差。

> 提示：$\text{Var}(\hat\theta_1) = \theta^2/(3n)$；$E(X_{(n)}) = n\theta/(n+1)$，$E(X_{(n)}^2) = n\theta^2/(n+2)$，$\text{Var}(\hat\theta_2) = \theta^2/[n(n+2)]$。由于 $1/[n(n+2)] < 1/(3n)$（对 $n \geq 1$），$\hat\theta_2$ 更有效。

**第 3 题**：设 $X \sim N(\mu, \sigma^2)$，$\sigma^2$ 已知，计算 Fisher 信息 $I_1(\mu)$，并写出基于 $n$ 个样本的 $\mu$ 的无偏估计量方差的 Cramér-Rao 下界。

> 提示：$\ln f = -\frac{(x-\mu)^2}{2\sigma^2} + \text{const}$，$\partial^2\ln f/\partial\mu^2 = -1/\sigma^2$，故 $I_1(\mu) = 1/\sigma^2$，CRB $= \sigma^2/n$。

**第 4 题**：为什么说"有偏估计量一定不是最优的"这句话是错误的？请举例说明。

> 提示：参见例 3 中 $\hat\sigma^2$ 与 $S^2$ 的 MSE 比较，有偏的 $\hat\sigma^2$ MSE 更小。此外，James-Stein 估计量是有偏的但在 MSE 意义下优于样本均值（当维数 $\geq 3$ 时）。"无偏性"只是评价标准之一，不是唯一准绳。

**第 5 题（综合）**：设 $X_1,\ldots,X_n$ i.i.d. $\sim \text{Bernoulli}(p)$，$\hat p = \bar{X}$。（a）验证无偏性；（b）计算 Fisher 信息并验证 $\hat p$ 是有效估计量；（c）用 CRB 确认 $\hat p$ 方差的下界与 $\text{Var}(\hat p)$ 相等。

> 提示：(a) $E\bar{X} = p$；(b) $\ln f = x\ln p + (1-x)\ln(1-p)$，$\partial^2\ln f/\partial p^2 = -x/p^2 - (1-x)/(1-p)^2$，$I_1(p) = E[-\partial^2\ln f/\partial p^2] = 1/(p(1-p))$；(c) CRB $= p(1-p)/n = \text{Var}(\hat p)$，故 $\hat p$ 有效。
