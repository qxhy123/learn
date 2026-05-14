# MLE 求解套路

> **一例速记**：$X_1,\ldots,X_n$ i.i.d. $\sim \text{Poisson}(\lambda)$，求 $\lambda$ 的 MLE。
> ① 似然：$L(\lambda) = \prod \frac{e^{-\lambda}\lambda^{x_i}}{x_i!}$；
> ② 对数：$\ell = -n\lambda + (\sum x_i)\ln\lambda - \sum\ln(x_i!)$；
> ③ 求导：$\ell' = -n + (\sum x_i)/\lambda = 0$；
> ④ 解：$\hat\lambda_{\text{MLE}} = \bar{x}$；
> ⑤ 二阶验证：$\ell'' = -\sum x_i/\lambda^2 < 0$ ✓。
> **"写似然 → 取对数 → 求导令零 → 解参数 → 验极大"——五步走完，结果几乎总是样本矩。**

---

## 一、为什么 MLE 是参数估计的首选方法

最大似然估计（Maximum Likelihood Estimation, MLE）的核心思想极为直觉：**选择使观测数据"最可能出现"的参数值作为估计**。

这个思想具有深刻的数学性质：在正则条件下，MLE 是相合估计（$n\to\infty$ 时收敛到真值）、渐近有效（方差达到 Cramér-Rao 下界）、渐近正态。换句话说，在大样本下，MLE 是"最聪明"的估计方法——没有其他估计量能在渐近意义下全面超越它。

而且，MLE 的求解流程高度机械化，适合复杂模型的参数估计。机器学习中的逻辑回归、高斯混合模型（EM 算法）、神经网络的交叉熵损失函数，本质上都是 MLE 的变体。

---

## 二、五步求解流程

### 步骤 ① 写似然函数 $L(\theta)$

设 $x_1, x_2, \ldots, x_n$ 是观测样本。**似然函数**是参数 $\theta$ 的函数（样本 $x_i$ 视为固定）：

$$L(\theta) = \prod_{i=1}^n f(x_i;\,\theta).$$

- 连续型：$f$ 是密度函数（PDF）。
- 离散型：$f$ 是概率质量函数（PMF）。

**关键：把 $f(x_i;\theta)$ 里的每个 $x_i$ 代入具体观测值，$\theta$ 保持符号形式。**

### 步骤 ② 取对数得对数似然 $\ell(\theta)$

$$\ell(\theta) = \ln L(\theta) = \sum_{i=1}^n \ln f(x_i;\,\theta).$$

取对数有两个好处：将乘积转为求和（便于求导），且由于 $\ln$ 单调，$L$ 和 $\ell$ 的极大值点相同。

### 步骤 ③ 求导，令 $\partial\ell/\partial\theta = 0$

$$\frac{\partial\ell}{\partial\theta} = 0.$$

多参数时对每个分量分别求偏导，组成方程组：

$$\frac{\partial\ell}{\partial\theta_1} = 0, \quad \frac{\partial\ell}{\partial\theta_2} = 0, \quad \ldots$$

### 步骤 ④ 解方程，得 $\hat\theta_{\text{MLE}}$

从方程（组）中解出参数，即为 MLE 候选点。

**注意**：若似然在边界取最大值（如均匀分布 $U(0,\theta)$ 的 MLE 是最大顺序统计量 $X_{(n)}$，由边界分析而非求导得到），则"求导令零"这一步不适用，需要直接分析似然函数的单调性。

### 步骤 ⑤ 验证为极大值（二阶导数检验）

确认 $\hat\theta$ 处为极大值（而非极小值或鞍点）：

$$\frac{\partial^2\ell}{\partial\theta^2}\bigg|_{\hat\theta} < 0.$$

多参数时验证 Hessian 矩阵 $H = \left(\partial^2\ell/\partial\theta_i\partial\theta_j\right)$ 在 $\hat{\boldsymbol\theta}$ 处负定。

**流程概览**：

| 步骤 | 操作 | 目的 |
|---|---|---|
| ① 写似然 | $L(\theta) = \prod f(x_i;\theta)$ | 构造目标函数 |
| ② 取对数 | $\ell(\theta) = \sum\ln f(x_i;\theta)$ | 乘积 → 求和，便于微分 |
| ③ 求导令零 | $\partial\ell/\partial\theta = 0$ | 找候选极值点 |
| ④ 解参数 | 得 $\hat\theta_{\text{MLE}}$ | 求解方程 |
| ⑤ 验极大 | $\partial^2\ell/\partial\theta^2 < 0$ | 确认是极大 |

---

## 三、不变性原理

**MLE 的不变性**：若 $\hat\theta_{\text{MLE}}$ 是 $\theta$ 的 MLE，$g$ 是任意（可测）函数，则

$$\widehat{g(\theta)}_{\text{MLE}} = g(\hat\theta_{\text{MLE}}).$$

**例子**：

- 估计 $\sigma^2$（正态，均值已知）的 MLE 是 $\hat\sigma^2$；则估计 $\sigma = \sqrt{\sigma^2}$ 的 MLE 是 $\hat\sigma = \sqrt{\hat\sigma^2}$，直接开根号。
- 若 $\hat\lambda$ 是 Poisson 参数 $\lambda$ 的 MLE，则 $e^{-\lambda}$（无事件的概率）的 MLE 是 $e^{-\hat\lambda}$，无需重新推导。
- 若 $\hat p$ 是成功概率的 MLE，则 $\text{odds} = p/(1-p)$ 的 MLE 是 $\hat p/(1-\hat p)$。

**不变性原理的重要性**：当目标参数是原始参数的非线性变换时，不必对 $g(\theta)$ 重新建似然和求导——只需先求出 $\theta$ 的 MLE，再做变换。

---

## 四、演示题：正态分布双参数 MLE

**题目**：设 $X_1,\ldots,X_n$ i.i.d. $\sim N(\mu, \sigma^2)$（$\mu, \sigma^2$ 均未知），求 $(\mu, \sigma^2)$ 的 MLE。

> **步骤 ①：写似然函数。**
>
> $$L(\mu,\sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right) = \left(2\pi\sigma^2\right)^{-n/2}\exp\!\left(-\frac{\sum_{i=1}^n(x_i-\mu)^2}{2\sigma^2}\right).$$
>
> **步骤 ②：取对数。**
>
> $$\ell(\mu,\sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i - \mu)^2.$$
>
> **步骤 ③：对 $\mu$ 求偏导，令其为零。**
>
> $$\frac{\partial\ell}{\partial\mu} = \frac{1}{\sigma^2}\sum_{i=1}^n(x_i - \mu) = 0 \implies \sum_{i=1}^n(x_i - \mu) = 0.$$
>
> **步骤 ④（$\mu$ 分量）：解 $\hat\mu$。**
>
> $$\sum x_i = n\mu \implies \hat\mu_{\text{MLE}} = \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i.$$
>
> **步骤 ③：对 $\sigma^2$ 求偏导。** 令 $v = \sigma^2$（记号简化）：
>
> $$\frac{\partial\ell}{\partial v} = -\frac{n}{2v} + \frac{1}{2v^2}\sum_{i=1}^n(x_i-\mu)^2 = 0.$$
>
> **步骤 ④（$\sigma^2$ 分量）：代入 $\hat\mu = \bar{x}$，解 $\hat\sigma^2$。**
>
> $$\frac{n}{2v} = \frac{\sum(x_i-\bar{x})^2}{2v^2} \implies v = \frac{\sum(x_i-\bar{x})^2}{n}.$$
>
> $$\hat\sigma^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n(x_i - \bar{x})^2.$$
>
> **步骤 ⑤：验证极大值。** 对 $\mu$ 的二阶偏导：$\partial^2\ell/\partial\mu^2 = -n/\sigma^2 < 0$ ✓。对 $\sigma^2$ 的二阶偏导：$\partial^2\ell/\partial v^2 = n/(2v^2) - \sum(x_i-\mu)^2/v^3$，代入 $\hat v = \sum(x_i-\bar x)^2/n$ 后得 $< 0$ ✓。
>
> **关键注意：$\hat\sigma^2_{\text{MLE}}$ 是有偏的！**
>
> $$E\!\left(\hat\sigma^2_{\text{MLE}}\right) = \frac{n-1}{n}\sigma^2 \neq \sigma^2.$$
>
> 这是 MLE 有偏的经典例子。无偏修正版为 $S^2 = \frac{1}{n-1}\sum(X_i-\bar{X})^2$（在第 07 篇已经讨论过）。但按不变性原理，$\hat\sigma_{\text{MLE}} = \sqrt{\hat\sigma^2_{\text{MLE}}}$ 是 $\sigma$ 的 MLE。
>
> **完整结论**：
>
> $$\hat\mu_{\text{MLE}} = \bar{X}, \qquad \hat\sigma^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n(X_i - \bar{X})^2.$$

---

## 五、思考路标

**路标 1**：构造对数似然时，**所有不含参数的常数项可以直接丢弃**（它们不影响极值点）。例如，$N(\mu,\sigma^2)$ 的对数似然中，$-\frac{n}{2}\ln(2\pi)$ 与 $\mu, \sigma^2$ 无关，求导时自动消失，可以从一开始就忽略。

**路标 2**：看到对数似然中出现 $\sum x_i$ 或 $\sum x_i^2$ 等统计量 → 最终 MLE 往往是样本均值、样本二阶矩或它们的函数。这是"充分统计量"的体现。

**路标 3**：若 MLE 方程解出来的 $\hat\theta$ 不含 $x_i$ 的个别值，只含 $\sum x_i$、$\bar x$、$\sum x_i^2$ 等聚合量 → 这些聚合量是**充分统计量**，它们捕捉了样本中关于 $\theta$ 的全部信息。

**路标 4**：遇到估计 $g(\theta)$（如 $e^{-\lambda}$、$\theta^2$、$1/\lambda$）→ 先用五步流程求出 $\theta$ 的 MLE $\hat\theta$，再用**不变性原理**直接给出 $g(\hat\theta)$，无需重新建立和求解对应似然。

**路标 5**：**正态 MLE 有偏**这一事实是考试高频考点。$\hat\sigma^2_{\text{MLE}} = \frac{1}{n}\sum(X_i-\bar{X})^2$，乘以 $\frac{n}{n-1}$ 才无偏。两个公式形状相同，分母不同（$n$ vs $n-1$）。

**路标 6**：如果似然函数在参数空间内部无极值点（如 $U(0,\theta)$ 的似然关于 $\theta$ 单调递减），MLE 在**边界**取得，此时需要直接分析似然的单调性或图形，而非求导。

**路标 7**：多参数 MLE 时，若各参数的方程解耦（如正态的 $\hat\mu$ 与 $\hat\sigma^2$ 方程分别求解），则逐个解；若方程组耦合（参数互相牵制），则需联立求解。

---

## 六、典型应用 3 例

### 例 1：指数分布的 MLE

**题目**：$X_1,\ldots,X_n$ i.i.d. $\sim \text{Exp}(\lambda)$（密度 $f(x;\lambda) = \lambda e^{-\lambda x}$，$x > 0$），求 $\lambda$ 的 MLE 及其均值 $1/\lambda$ 的 MLE。

**五步求解**：

① $L(\lambda) = \lambda^n \exp\!\left(-\lambda\sum x_i\right)$

② $\ell(\lambda) = n\ln\lambda - \lambda\sum x_i$

③ $\ell'(\lambda) = n/\lambda - \sum x_i = 0$

④ $\hat\lambda_{\text{MLE}} = n/\sum x_i = 1/\bar{x}$

⑤ $\ell''(\lambda) = -n/\lambda^2 < 0$ ✓

由不变性原理，$1/\lambda$ 的 MLE $= 1/\hat\lambda_{\text{MLE}} = \bar{X}$（样本均值）。

**注意**：$\hat\lambda = 1/\bar{X}$ 是有偏的（$E(1/\bar{X}) \neq 1/E(\bar{X}) = \lambda$，因为 $1/x$ 是凸函数），但 $\bar{X}$ 是 $1/\lambda$ 的无偏估计。

---

### 例 2：Bernoulli 总体的 MLE 与不变性原理

**题目**：$X_1,\ldots,X_n$ i.i.d. $\sim \text{Bernoulli}(p)$，$n = 100$，观测到 $\sum x_i = 40$。求 $p$ 的 MLE，并用不变性原理求"两次伯努利试验都成功"的概率 $p^2$ 的 MLE。

**求解**：

① $L(p) = p^{\sum x_i}(1-p)^{n - \sum x_i}$

② $\ell(p) = (\sum x_i)\ln p + (n - \sum x_i)\ln(1-p)$

③ $\ell'(p) = \frac{\sum x_i}{p} - \frac{n-\sum x_i}{1-p} = 0$

④ $\hat p_{\text{MLE}} = \frac{\sum x_i}{n} = \frac{40}{100} = 0.4$

⑤ $\ell''(p) = -(\sum x_i)/p^2 - (n-\sum x_i)/(1-p)^2 < 0$ ✓

由不变性原理，$p^2$ 的 MLE $= \hat p^2 = 0.16$。

---

### 例 3：均匀分布的 MLE（边界极值）

**题目**：$X_1,\ldots,X_n$ i.i.d. $\sim U(0, \theta)$（$\theta > 0$），求 $\theta$ 的 MLE。

**分析**：密度 $f(x;\theta) = 1/\theta$（$0 \leq x \leq \theta$），否则为 $0$。

① 似然：$L(\theta) = \theta^{-n} \cdot \mathbf{1}[\theta \geq \max(x_1,\ldots,x_n)] = \theta^{-n} \cdot \mathbf{1}[\theta \geq x_{(n)}]$

其中 $x_{(n)} = \max_i x_i$ 是最大顺序统计量。

② $\ell(\theta) = -n\ln\theta$（当 $\theta \geq x_{(n)}$ 时）

③ $\ell'(\theta) = -n/\theta < 0$：$\ell$ 关于 $\theta$ 单调递减！

④ 在 $\theta \geq x_{(n)}$ 的约束下，$\ell(\theta)$ 在 $\theta = x_{(n)}$ 时最大（越小越好，但不能小于 $x_{(n)}$）。

$$\hat\theta_{\text{MLE}} = X_{(n)} = \max(X_1,\ldots,X_n).$$

**此题不能用"令导数为零"解出 MLE**，必须分析似然的单调性并结合参数约束。

**注意**：$\hat\theta_{\text{MLE}} = X_{(n)}$ 是有偏的（$E(X_{(n)}) = n\theta/(n+1) < \theta$），但渐近无偏（$E(X_{(n)}) \to \theta$）且一致。无偏修正为 $\frac{n+1}{n}X_{(n)}$。

---

## 七、自测题

**第 1 题**：$X_1,\ldots,X_n$ i.i.d. $\sim N(\mu, 1)$（方差已知为 $1$），求 $\mu$ 的 MLE 并验证其无偏性与有效性。

> 提示：$\ell(\mu) = -\frac{1}{2}\sum(x_i-\mu)^2 + \text{const}$，$\ell'(\mu) = \sum(x_i-\mu) = 0$，得 $\hat\mu = \bar{x}$。$E\bar{X} = \mu$（无偏）；$\text{Var}(\bar{X}) = 1/n = \text{CRB}$（有效）。

**第 2 题**：$X_1,\ldots,X_n$ i.i.d.，$f(x;\theta) = \theta x^{\theta-1}$（$0 < x < 1$，$\theta > 0$），求 $\theta$ 的 MLE。

> 提示：$\ell(\theta) = n\ln\theta + (\theta-1)\sum\ln x_i$，$\ell'(\theta) = n/\theta + \sum\ln x_i = 0$，得 $\hat\theta = -n/\sum\ln x_i = -1/\overline{\ln x}$。

**第 3 题**：在第 2 题的设定下，利用不变性原理求 $P(X > 0.5) = 1 - 0.5^\theta$ 的 MLE。

> 提示：直接代入 $\hat\theta$：$\widehat{P(X>0.5)} = 1 - 0.5^{\hat\theta} = 1 - 2^{-\hat\theta}$，其中 $\hat\theta = -1/\overline{\ln x}$。

**第 4 题**：设 $X_1,\ldots,X_n$ i.i.d. $\sim \text{Gamma}(\alpha, \lambda)$（形状参数 $\alpha$ 已知，率参数 $\lambda$ 未知），密度 $f(x;\lambda) = \frac{\lambda^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\lambda x}$（$x > 0$），求 $\lambda$ 的 MLE。

> 提示：$\ell(\lambda) = n\alpha\ln\lambda - \lambda\sum x_i + \text{const}$，$\ell'(\lambda) = n\alpha/\lambda - \sum x_i = 0$，得 $\hat\lambda = n\alpha/\sum x_i = \alpha/\bar{x}$（当 $\alpha = 1$ 时退化为指数分布的结果 $1/\bar{x}$，验证一致性）。

**第 5 题（双参数）**：$X_1,\ldots,X_n$ i.i.d. $\sim N(\mu,\sigma^2)$，$n=5$，观测值为 $\{2, 3, 5, 7, 8\}$。计算 $\hat\mu_{\text{MLE}}$ 和 $\hat\sigma^2_{\text{MLE}}$ 的具体数值，并判断 $\hat\sigma^2_{\text{MLE}}$ 是否无偏。

> 提示：$\bar{x} = 25/5 = 5$，$\hat\sigma^2_{\text{MLE}} = \frac{1}{5}[(2-5)^2+(3-5)^2+(5-5)^2+(7-5)^2+(8-5)^2] = \frac{1}{5}[9+4+0+4+9] = 26/5 = 5.2$。$E(\hat\sigma^2_{\text{MLE}}) = \frac{n-1}{n}\sigma^2 = \frac{4}{5}\sigma^2 \neq \sigma^2$，有偏。无偏修正：$S^2 = \frac{n}{n-1}\hat\sigma^2_{\text{MLE}} = \frac{26}{4} = 6.5$。
