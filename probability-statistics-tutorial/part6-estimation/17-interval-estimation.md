# 第17章：区间估计

## 学习目标

学完本章后，你将能够：

- 理解置信区间的严格定义，正确解释"置信水平 $1-\alpha$"的频率学派含义，避免常见误解
- 掌握**枢轴量法**的构造思路，将样本统计量变换为分布已知的随机变量，从而反解出参数的区间范围
- 熟练推导正态总体在四种情形下（均值/方差已知或未知）的置信区间，并理解 $z$、$t$、$\chi^2$、$F$ 分布的适用条件
- 运用中心极限定理构造大样本区间估计，覆盖均值、比例、泊松速率等常见参数
- 区分双侧置信区间与单侧置信限，根据实际问题的安全约束选择合适的区间形式

---

## 17.1 置信区间的概念

### 17.1.1 点估计的局限性

第16章介绍的点估计给出参数的单一数值，如 $\hat{\mu} = \bar{X}$。但点估计本身没有提供**估计精度**的信息——我们需要知道这个估计量"差多少"才算合理。

**例17.1** 用 $n=25$ 个样本估计总体均值 $\mu$，得 $\bar{x} = 10.3$，$s = 2.0$。

- 这个估计有多可靠？真实 $\mu$ 是否一定接近 $10.3$？
- 如何给出一个"区间"，使我们有把握认为 $\mu$ 落在其中？

**区间估计**正是为了回答这一问题：不给出一个数，而是给出一个**随机区间** $[L(X), U(X)]$，使该区间以指定的高概率覆盖真参数。

### 17.1.2 置信区间的定义

**定义 17.1（置信区间）** 设总体分布含未知参数 $\theta$，$X_1, \ldots, X_n$ 为来自该总体的样本。若统计量 $L = L(X_1, \ldots, X_n)$ 和 $U = U(X_1, \ldots, X_n)$ 满足

$$
P_\theta(L \leq \theta \leq U) = 1 - \alpha, \quad \forall \theta \in \Theta
$$

则称随机区间 $[L, U]$ 为参数 $\theta$ 的**置信水平**（置信度）为 $1-\alpha$ 的**置信区间**，$L$ 和 $U$ 分别称为**置信下限**和**置信上限**，$\alpha$ 称为**显著性水平**。

常用置信水平：$1-\alpha = 90\%$、$95\%$、$99\%$。

### 17.1.3 置信区间的正确解读

置信区间是频率学派的概念，其含义容易被误解：

**正确理解**：若对同一总体重复抽样无限多次，每次构造一个置信区间，则 $1-\alpha$ 比例的区间会覆盖真参数 $\theta$。

**常见误解（错误！）**：某次观测得到 $[2.1, 3.5]$，则参数 $\theta$ 以 95% 的概率落在 $[2.1, 3.5]$ 内。

关键点在于：参数 $\theta$ 是**固定常数**，不是随机变量；样本观测后，区间 $[l, u]$ 也变成了固定区间——此时 $\theta$ 要么在其中，要么不在，不存在"落入概率"。**随机性来自区间本身**（在抽样之前，$L$ 和 $U$ 是随机变量）。

**可视化理解**：想象重复抽样 100 次，每次构造一个 95% 置信区间，大约有 95 个区间会包含真值 $\theta$，约 5 个不包含。但你无法事先知道当前这次构造的区间是否属于那 95% 之列。

### 17.1.4 置信区间的长度与精度

置信区间的长度 $U - L$ 度量了估计的精度：

- **区间越短**：估计越精确
- **区间越长**：估计越粗糙

影响区间长度的因素：

| 因素 | 对区间长度的影响 | 原因 |
|------|----------------|------|
| 样本量 $n$ 增大 | 变短（$\propto 1/\sqrt{n}$）| 估计更稳定 |
| 置信水平 $1-\alpha$ 提高 | 变长 | 更"保守"才能覆盖更多情况 |
| 总体方差 $\sigma^2$ 增大 | 变长 | 数据更散乱 |

存在**精度与置信度的权衡**：不可能同时要求区间很短且置信度很高，除非增加样本量。

---

## 17.2 枢轴量法

### 17.2.1 枢轴量的定义

**定义 17.2（枢轴量）** 设 $X_1, \ldots, X_n$ 来自含参数 $\theta$ 的总体。若统计量 $G = G(X_1, \ldots, X_n, \theta)$ 的分布**完全已知**（不依赖任何未知参数），则称 $G$ 为参数 $\theta$ 的**枢轴量**（pivot）。

注意：枢轴量 $G$ 同时含有**样本**和**待估参数** $\theta$，这是它与一般统计量的区别。

### 17.2.2 枢轴量法的步骤

枢轴量法（Pivotal Method）是构造置信区间的通用方法，步骤如下：

**第一步：构造枢轴量**

找到含有 $\theta$ 且分布已知的量 $G(X_1, \ldots, X_n, \theta)$。

**第二步：确定分位数**

对给定的置信水平 $1-\alpha$，找到常数 $a, b$ 使得

$$
P(a \leq G \leq b) = 1 - \alpha
$$

通常取**等尾**形式：$P(G < a) = \alpha/2$ 和 $P(G > b) = \alpha/2$。

**第三步：反解参数**

将不等式 $a \leq G(X_1, \ldots, X_n, \theta) \leq b$ 对 $\theta$ 求解，得到

$$
P(L(X) \leq \theta \leq U(X)) = 1 - \alpha
$$

其中 $L(X), U(X)$ 即为置信下限和上限。

### 17.2.3 一个完整示例

**例17.2** 设 $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$，$\sigma^2$ 已知，求 $\mu$ 的 $1-\alpha$ 置信区间。

**第一步**：由于

$$
\bar{X} \sim \mathcal{N}\!\left(\mu, \frac{\sigma^2}{n}\right)
$$

构造枢轴量：

$$
G = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim \mathcal{N}(0, 1)
$$

$G$ 的分布是标准正态，不依赖 $\mu$，故 $G$ 是枢轴量。

**第二步**：设 $z_{\alpha/2}$ 为标准正态分布的上 $\alpha/2$ 分位数，即 $P(Z > z_{\alpha/2}) = \alpha/2$，则

$$
P\!\left(-z_{\alpha/2} \leq \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \leq z_{\alpha/2}\right) = 1 - \alpha
$$

**第三步**：对中间不等式关于 $\mu$ 求解：

$$
-z_{\alpha/2} \leq \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \leq z_{\alpha/2}
\quad \Longleftrightarrow \quad
\bar{X} - z_{\alpha/2} \frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{X} + z_{\alpha/2} \frac{\sigma}{\sqrt{n}}
$$

故 $\mu$ 的 $1-\alpha$ 置信区间为：

$$
\boxed{\left[\bar{X} - z_{\alpha/2} \frac{\sigma}{\sqrt{n}},\quad \bar{X} + z_{\alpha/2} \frac{\sigma}{\sqrt{n}}\right]}
$$

### 17.2.4 常用分布的分位数

| 分布 | 记号 | $\alpha=0.10$ | $\alpha=0.05$ | $\alpha=0.01$ |
|------|------|--------------|--------------|--------------|
| 标准正态 $\mathcal{N}(0,1)$ | $z_{\alpha/2}$ | 1.645 | 1.960 | 2.576 |
| $t(n-1)$ | $t_{\alpha/2}(n-1)$ | 随自由度变化 | 随自由度变化 | 随自由度变化 |
| $\chi^2(n-1)$ | $\chi^2_{\alpha/2}(n-1)$ | 查表 | 查表 | 查表 |

对于 $t$ 分布，自由度 $\nu$ 增大时 $t_{\alpha/2}(\nu) \to z_{\alpha/2}$；当 $\nu \geq 30$ 时，$t$ 分布与标准正态已十分接近。

---

## 17.3 正态总体的区间估计

正态总体是区间估计的核心场景。设 $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$，样本均值 $\bar{X}$ 和样本方差 $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$。

回顾正态总体的抽样分布（第13章结论）：

$$
\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim \mathcal{N}(0,1), \qquad
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1), \qquad
\frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t(n-1)
$$

其中后两个量相互独立。

### 17.3.1 均值 $\mu$ 的置信区间（$\sigma^2$ 已知）

**枢轴量**：$G = \dfrac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim \mathcal{N}(0,1)$

**$1-\alpha$ 置信区间**：

$$
\boxed{\mu \in \left[\bar{X} - z_{\alpha/2} \frac{\sigma}{\sqrt{n}},\quad \bar{X} + z_{\alpha/2} \frac{\sigma}{\sqrt{n}}\right]}
$$

**区间半宽（误差限）**：$\delta = z_{\alpha/2} \cdot \dfrac{\sigma}{\sqrt{n}}$

**所需样本量**：若要求误差限不超过 $\delta_0$，则

$$
n \geq \left(\frac{z_{\alpha/2} \cdot \sigma}{\delta_0}\right)^2
$$

**例17.3** 某厂生产灯泡，寿命 $X \sim \mathcal{N}(\mu, 100^2)$（小时）。随机抽取 25 个，得 $\bar{x} = 1500$，求 $\mu$ 的 95% 置信区间。

$$
\left[1500 - 1.96 \times \frac{100}{\sqrt{25}},\quad 1500 + 1.96 \times \frac{100}{\sqrt{25}}\right] = [1500 - 39.2,\; 1500 + 39.2] = [1460.8,\; 1539.2]
$$

### 17.3.2 均值 $\mu$ 的置信区间（$\sigma^2$ 未知）

实践中 $\sigma^2$ 通常未知，此时用样本标准差 $S$ 代替 $\sigma$。

**枢轴量**：$G = \dfrac{\bar{X} - \mu}{S/\sqrt{n}} \sim t(n-1)$

**$1-\alpha$ 置信区间**：

$$
\boxed{\mu \in \left[\bar{X} - t_{\alpha/2}(n-1) \frac{S}{\sqrt{n}},\quad \bar{X} + t_{\alpha/2}(n-1) \frac{S}{\sqrt{n}}\right]}
$$

与 $\sigma^2$ 已知情形相比，$z_{\alpha/2}$ 被 $t_{\alpha/2}(n-1)$ 替换，后者更大，因此区间**更宽**——这反映了 $\sigma^2$ 未知带来的额外不确定性。

**例17.4** 随机测量某金属零件直径（毫米）10 次，得：$\bar{x} = 50.02$，$s = 0.04$。设直径服从正态分布，求 $\mu$ 的 95% 置信区间。

查 $t$ 分布表：$t_{0.025}(9) = 2.262$。

$$
\left[50.02 - 2.262 \times \frac{0.04}{\sqrt{10}},\quad 50.02 + 2.262 \times \frac{0.04}{\sqrt{10}}\right] = [50.02 - 0.0286,\; 50.02 + 0.0286] = [49.991,\; 50.049]
$$

### 17.3.3 方差 $\sigma^2$ 的置信区间

**枢轴量**：$G = \dfrac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)$

注意 $\chi^2$ 分布不对称，等尾区间的上下限分别对应两个不同的分位数。

设 $\chi^2_{1-\alpha/2}(n-1)$ 和 $\chi^2_{\alpha/2}(n-1)$ 分别是 $\chi^2(n-1)$ 分布的下 $\alpha/2$ 和上 $\alpha/2$ 分位数，则

$$
P\!\left(\chi^2_{1-\alpha/2}(n-1) \leq \frac{(n-1)S^2}{\sigma^2} \leq \chi^2_{\alpha/2}(n-1)\right) = 1 - \alpha
$$

对 $\sigma^2$ 求解：

$$
\boxed{\sigma^2 \in \left[\frac{(n-1)S^2}{\chi^2_{\alpha/2}(n-1)},\quad \frac{(n-1)S^2}{\chi^2_{1-\alpha/2}(n-1)}\right]}
$$

**例17.5** 对例17.4 的直径数据，求 $\sigma^2$ 的 95% 置信区间。

$n=10$，$s^2 = 0.04^2 = 0.0016$，$(n-1)s^2 = 9 \times 0.0016 = 0.0144$。

查表：$\chi^2_{0.025}(9) = 19.023$，$\chi^2_{0.975}(9) = 2.700$。

$$
\sigma^2 \in \left[\frac{0.0144}{19.023},\; \frac{0.0144}{2.700}\right] = [0.000757,\; 0.00533]
$$

标准差 $\sigma \in [0.0275, 0.0730]$（毫米）。

### 17.3.4 两正态总体均值差的置信区间

设两个独立正态总体 $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$，$Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$，分别抽取样本量 $m, n$，统计量为 $\bar{X}, S_1^2$ 和 $\bar{Y}, S_2^2$。

**情形一：$\sigma_1^2, \sigma_2^2$ 已知**

枢轴量：

$$
G = \frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{\sqrt{\sigma_1^2/m + \sigma_2^2/n}} \sim \mathcal{N}(0,1)
$$

$\mu_1 - \mu_2$ 的 $1-\alpha$ 置信区间：

$$
(\bar{X} - \bar{Y}) \pm z_{\alpha/2}\sqrt{\frac{\sigma_1^2}{m} + \frac{\sigma_2^2}{n}}
$$

**情形二：$\sigma_1^2 = \sigma_2^2 = \sigma^2$（未知）**

合并样本方差：

$$
S_p^2 = \frac{(m-1)S_1^2 + (n-1)S_2^2}{m+n-2}
$$

枢轴量：

$$
G = \frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{S_p\sqrt{1/m + 1/n}} \sim t(m+n-2)
$$

$\mu_1 - \mu_2$ 的 $1-\alpha$ 置信区间：

$$
\boxed{(\bar{X} - \bar{Y}) \pm t_{\alpha/2}(m+n-2) \cdot S_p\sqrt{\frac{1}{m} + \frac{1}{n}}}
$$

### 17.3.5 两正态总体方差比的置信区间

**枢轴量**：

$$
G = \frac{S_1^2 / \sigma_1^2}{S_2^2 / \sigma_2^2} = \frac{S_1^2}{S_2^2} \cdot \frac{\sigma_2^2}{\sigma_1^2} \sim F(m-1, n-1)
$$

设 $F_{\alpha/2}(m-1, n-1)$ 和 $F_{1-\alpha/2}(m-1, n-1)$ 分别是 $F(m-1,n-1)$ 分布的上、下 $\alpha/2$ 分位数，利用 $F$ 分布的对称性 $F_{1-\alpha/2}(m-1,n-1) = 1/F_{\alpha/2}(n-1,m-1)$，得

$$
\boxed{\frac{\sigma_1^2}{\sigma_2^2} \in \left[\frac{S_1^2}{S_2^2} \cdot \frac{1}{F_{\alpha/2}(m-1, n-1)},\quad \frac{S_1^2}{S_2^2} \cdot F_{\alpha/2}(n-1, m-1)\right]}
$$

---

## 17.4 大样本区间估计

当总体分布未知或非正态时，若样本量足够大（通常 $n \geq 30$），可利用**中心极限定理**（CLT）构造近似区间估计。

### 17.4.1 总体均值的大样本区间估计

设 $X_1, \ldots, X_n$ 为来自均值 $\mu$、方差 $\sigma^2$ 的总体的 i.i.d. 样本。

由 CLT：

$$
\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1) \quad (n \to \infty)
$$

**$\sigma^2$ 已知**：直接使用 $z$ 区间：

$$
\mu \in \left[\bar{X} - z_{\alpha/2} \frac{\sigma}{\sqrt{n}},\quad \bar{X} + z_{\alpha/2} \frac{\sigma}{\sqrt{n}}\right]
$$

**$\sigma^2$ 未知**：将 $\sigma$ 替换为 $S$（一致估计量），仍用 $z$ 分位数（大样本时 $t_{n-1} \approx z$）：

$$
\boxed{\mu \in \left[\bar{X} - z_{\alpha/2} \frac{S}{\sqrt{n}},\quad \bar{X} + z_{\alpha/2} \frac{S}{\sqrt{n}}\right]}
$$

这是**大样本**下的近似置信区间，$n$ 越大近似越精确。

### 17.4.2 比例 $p$ 的大样本置信区间

设总体服从 Bernoulli 分布，成功概率为 $p$。样本量 $n$，成功次数 $k$，样本比例 $\hat{p} = k/n$。

由 CLT（$np \geq 5$ 且 $n(1-p) \geq 5$ 时效果好）：

$$
\frac{\hat{p} - p}{\sqrt{p(1-p)/n}} \xrightarrow{d} \mathcal{N}(0,1)
$$

用 $\hat{p}$ 估计分母中的 $p$，得 $p$ 的 $1-\alpha$ 近似置信区间：

$$
\boxed{p \in \left[\hat{p} - z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}},\quad \hat{p} + z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}\right]}
$$

**例17.6** 随机调查 400 人，其中 120 人支持某政策，$\hat{p} = 120/400 = 0.3$。求支持率 $p$ 的 95% 置信区间。

$$
\sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = \sqrt{\frac{0.3 \times 0.7}{400}} = \sqrt{0.000525} \approx 0.0229
$$

$$
p \in [0.3 - 1.96 \times 0.0229,\; 0.3 + 1.96 \times 0.0229] = [0.255,\; 0.345]
$$

**保守估计**：由于 $p(1-p) \leq 1/4$（最大值在 $p=0.5$ 时取到），取 $\hat{p}(1-\hat{p}) = 1/4$ 可得**保守区间**：

$$
p \in \left[\hat{p} \pm \frac{z_{\alpha/2}}{2\sqrt{n}}\right]
$$

对 95% 置信度，$n=400$ 时误差限为 $1.96/(2\times 20) = 0.049$，即"±4.9%"。

### 17.4.3 泊松参数的大样本置信区间

设 $X_1, \ldots, X_n \overset{iid}{\sim} \text{Pois}(\lambda)$，则 $E[X] = \text{Var}[X] = \lambda$。

由 CLT，大样本下：

$$
\frac{\bar{X} - \lambda}{\sqrt{\lambda/n}} \xrightarrow{d} \mathcal{N}(0,1)
$$

用 $\hat{\lambda} = \bar{X}$ 估计分母中的 $\lambda$，得：

$$
\lambda \in \left[\bar{X} - z_{\alpha/2}\sqrt{\frac{\bar{X}}{n}},\quad \bar{X} + z_{\alpha/2}\sqrt{\frac{\bar{X}}{n}}\right]
$$

**例17.7** 某服务台 100 分钟内共接到 200 个呼叫（$n=100$ 分钟，$\bar{x}=2$），估计每分钟呼叫率 $\lambda$ 的 95% 置信区间。

$$
\lambda \in \left[2 - 1.96\sqrt{\frac{2}{100}},\; 2 + 1.96\sqrt{\frac{2}{100}}\right] = [2 - 0.277,\; 2 + 0.277] = [1.723,\; 2.277]
$$

### 17.4.4 Delta 方法：参数变换的区间估计

设 $\hat{\theta}$ 是 $\theta$ 的估计量，满足 $\sqrt{n}(\hat{\theta} - \theta) \xrightarrow{d} \mathcal{N}(0, V(\theta))$。

若 $g(\cdot)$ 是可微函数，则由 **Delta 方法**：

$$
\sqrt{n}(g(\hat{\theta}) - g(\theta)) \xrightarrow{d} \mathcal{N}\!\left(0,\; [g'(\theta)]^2 V(\theta)\right)
$$

用 $\hat{\theta}$ 代入，得 $g(\theta)$ 的 $1-\alpha$ 置信区间：

$$
g(\theta) \in \left[g(\hat{\theta}) \pm z_{\alpha/2} \frac{|g'(\hat{\theta})|\sqrt{V(\hat{\theta})}}{\sqrt{n}}\right]
$$

**例17.8** 设 $\hat{p}$ 为比例的 MLE，估计 $\log\!\left(\dfrac{p}{1-p}\right)$（对数优势比，log-odds）的置信区间。

令 $g(p) = \log\!\left(\dfrac{p}{1-p}\right)$，则 $g'(p) = \dfrac{1}{p(1-p)}$，$V(p) = p(1-p)$，故

$$
\text{标准误} = \frac{|g'(\hat{p})|\sqrt{V(\hat{p})}}{\sqrt{n}} = \frac{1}{\hat{p}(1-\hat{p})} \cdot \frac{\sqrt{\hat{p}(1-\hat{p})}}{\sqrt{n}} = \frac{1}{\sqrt{n\hat{p}(1-\hat{p})}}
$$

---

## 17.5 单侧置信限

### 17.5.1 单侧置信限的定义

在许多工程和安全问题中，我们只关心参数的一个方向。例如：

- 安全问题：某化学品的毒性**上限**（越低越安全）
- 质量控制：产品次品率的**上限**（不希望超标）
- 可靠性工程：零件寿命的**下限**（不希望太短）

**定义 17.3（单侧置信限）**

若统计量 $L = L(X_1, \ldots, X_n)$ 满足

$$
P_\theta(\theta \geq L) = 1 - \alpha, \quad \forall \theta \in \Theta
$$

则称 $L$ 为 $\theta$ 的**置信水平为 $1-\alpha$ 的单侧置信下限**，相应的区间 $[L, +\infty)$ 称为**单侧置信区间**。

类似地，若 $U = U(X_1, \ldots, X_n)$ 满足

$$
P_\theta(\theta \leq U) = 1 - \alpha, \quad \forall \theta \in \Theta
$$

则称 $U$ 为**单侧置信上限**，区间 $(-\infty, U]$ 为单侧置信区间。

### 17.5.2 单侧置信限与双侧置信区间的关系

从双侧置信区间推导单侧置信限非常简单：将双侧区间一端"推至无穷"。

具体地，若 $\mu$ 的双侧 $1-2\alpha$ 区间为 $[L, U]$，则：

- $L$ 是 $\mu$ 的（单侧）置信水平 $1-\alpha$ 的**下限**
- $U$ 是 $\mu$ 的（单侧）置信水平 $1-\alpha$ 的**上限**

| 类型 | 表达式 | 置信水平 |
|------|--------|---------|
| 双侧区间 | $[\bar{X} - z_{\alpha/2}\frac{\sigma}{\sqrt{n}},\; \bar{X} + z_{\alpha/2}\frac{\sigma}{\sqrt{n}}]$ | $1-\alpha$ |
| 单侧下限 | $\bar{X} - z_{\alpha}\frac{\sigma}{\sqrt{n}}$ | $1-\alpha$ |
| 单侧上限 | $\bar{X} + z_{\alpha}\frac{\sigma}{\sqrt{n}}$ | $1-\alpha$ |

注意单侧时使用 $z_\alpha$（不是 $z_{\alpha/2}$）：分位数更小，因为只需保护一侧。

### 17.5.3 正态总体各参数的单侧置信限

**均值 $\mu$ 的单侧置信限（$\sigma^2$ 未知）**

单侧置信下限（保证 $\mu$ 不太小）：

$$
L = \bar{X} - t_{\alpha}(n-1) \frac{S}{\sqrt{n}}
$$

单侧置信上限（保证 $\mu$ 不太大）：

$$
U = \bar{X} + t_{\alpha}(n-1) \frac{S}{\sqrt{n}}
$$

**方差 $\sigma^2$ 的单侧置信限**

单侧置信上限（常用于质量控制）：

$$
U = \frac{(n-1)S^2}{\chi^2_{1-\alpha}(n-1)}
$$

单侧置信下限：

$$
L = \frac{(n-1)S^2}{\chi^2_{\alpha}(n-1)}
$$

### 17.5.4 示例

**例17.9** 对例17.4 的零件直径，求 $\mu$ 的 95% 单侧置信下限（即保证平均直径不低于某值）。

$n=10$，$\bar{x}=50.02$，$s=0.04$，查表 $t_{0.05}(9) = 1.833$。

$$
L = 50.02 - 1.833 \times \frac{0.04}{\sqrt{10}} = 50.02 - 0.0232 = 49.997 \text{（毫米）}
$$

含义：有 95% 的把握认为，该批零件的平均直径不低于 49.997 毫米。

**例17.10** 某型号电池寿命 $X$（小时），测试 16 块，得 $\bar{x} = 350$，$s = 20$。求寿命均值 $\mu$ 的 95% 单侧置信下限。

$t_{0.05}(15) = 1.753$。

$$
L = 350 - 1.753 \times \frac{20}{\sqrt{16}} = 350 - 8.765 = 341.235 \text{（小时）}
$$

---

## 本章小结

**置信区间的本质**：置信区间是频率学派对参数不确定性的量化工具，$1-\alpha$ 置信水平的含义是"重复抽样中 $1-\alpha$ 比例的区间覆盖真参数"，而非"参数以 $1-\alpha$ 概率落在区间内"。

**枢轴量法的三步骤**：

1. 构造含参数 $\theta$ 且分布已知的枢轴量 $G$
2. 利用分位数建立 $P(a \leq G \leq b) = 1-\alpha$
3. 反解不等式得到 $[L, U]$

**正态总体的四种标准情形**：

| 估计目标 | 已知条件 | 枢轴量 | 区间（$1-\alpha$） |
|---------|---------|--------|-------------------|
| 均值 $\mu$ | $\sigma^2$ 已知 | $\frac{\bar{X}-\mu}{\sigma/\sqrt{n}} \sim \mathcal{N}(0,1)$ | $\bar{X} \pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}$ |
| 均值 $\mu$ | $\sigma^2$ 未知 | $\frac{\bar{X}-\mu}{S/\sqrt{n}} \sim t(n-1)$ | $\bar{X} \pm t_{\alpha/2}(n-1)\frac{S}{\sqrt{n}}$ |
| 方差 $\sigma^2$ | $\mu$ 未知 | $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)$ | $\left[\frac{(n-1)S^2}{\chi^2_{\alpha/2}},\frac{(n-1)S^2}{\chi^2_{1-\alpha/2}}\right]$ |
| 方差比 $\sigma_1^2/\sigma_2^2$ | $\mu_1,\mu_2$ 未知 | $\frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2} \sim F(m-1,n-1)$ | 见 17.3.5 |

**大样本区间估计**：当 $n$ 较大时，CLT 保证 $\dfrac{\bar{X}-\mu}{S/\sqrt{n}} \approx \mathcal{N}(0,1)$，可用 $z$ 分位数替代 $t$ 分位数，适用于未知分布总体、比例估计、泊松参数等场景。

**单侧置信限**：当仅关心参数的上界或下界时使用，用 $z_\alpha$（或 $t_\alpha$）替代 $z_{\alpha/2}$（或 $t_{\alpha/2}$），区间一端延伸至无穷。

**精度-置信度权衡**：在总体方差固定的条件下，缩短区间的唯一办法是增大样本量 $n$（误差限 $\propto 1/\sqrt{n}$）；提高置信水平必然导致区间变宽。

---

## 深度学习应用：不确定性量化与预测区间

在深度学习的部署场景中，模型不仅要给出预测值，还要量化预测的**不确定性**——这与置信区间/预测区间的思想密切相关。

### 认知不确定性与偶然不确定性

深度学习中的不确定性可分为两类：

| 类型 | 来源 | 能否通过增加数据减少 | 对应统计概念 |
|------|------|--------------------|-----------|
| **偶然不确定性**（Aleatoric） | 数据本身的噪声，不可约 | 否 | 随机误差 $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ |
| **认知不确定性**（Epistemic） | 模型参数的不确定性 | 是（更多数据） | 参数的置信区间 |

标准神经网络只给出点预测，无法区分这两类不确定性。以下介绍三种实用方案。

### 方案一：预测区间（直接回归法）

最简单的方案：让网络直接输出**预测区间的上下界**，损失函数使用**分位数损失**（Pinball Loss）。

设置信水平 $1-\alpha$，分别训练两个分位数水平 $\tau_L = \alpha/2$ 和 $\tau_U = 1-\alpha/2$ 的分位数回归模型。

分位数损失：

$$
\mathcal{L}_\tau(y, \hat{q}) = \begin{cases} \tau (y - \hat{q}) & \text{若 } y \geq \hat{q} \\ (1-\tau)(\hat{q} - y) & \text{若 } y < \hat{q} \end{cases}
$$

```python
import torch
import torch.nn as nn

# ============================================================
# 方案一：分位数回归——直接预测置信区间上下界
# ============================================================

class QuantileNet(nn.Module):
    """
    双头网络：同时预测第 alpha/2 和 1-alpha/2 分位数
    输出形状：(batch_size, 2)  ->  [下界, 上界]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 两个独立输出头：下分位数 q_lo，上分位数 q_hi
        self.head_lo = nn.Linear(hidden_dim, 1)
        self.head_hi = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        q_lo = self.head_lo(h).squeeze(-1)
        q_hi = self.head_hi(h).squeeze(-1)
        return q_lo, q_hi


def pinball_loss(y: torch.Tensor, q_hat: torch.Tensor, tau: float) -> torch.Tensor:
    """
    分位数损失（Pinball Loss）
    tau=0.025 -> 预测第 2.5 百分位（下界）
    tau=0.975 -> 预测第 97.5 百分位（上界）
    """
    residual = y - q_hat
    loss = torch.where(residual >= 0, tau * residual, (tau - 1) * residual)
    return loss.mean()


def train_quantile_net(model: QuantileNet, X_train: torch.Tensor,
                       y_train: torch.Tensor, alpha: float = 0.05,
                       epochs: int = 200, lr: float = 1e-3):
    """训练分位数回归网络，输出 1-alpha 预测区间"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tau_lo, tau_hi = alpha / 2, 1 - alpha / 2  # 例：0.025, 0.975

    for epoch in range(epochs):
        optimizer.zero_grad()
        q_lo, q_hi = model(X_train)

        # 下分位数损失 + 上分位数损失
        loss = pinball_loss(y_train, q_lo, tau_lo) + \
               pinball_loss(y_train, q_hi, tau_hi)
        loss.backward()
        optimizer.step()

    return model


def predict_interval(model: QuantileNet, X_test: torch.Tensor):
    """推理：返回预测区间 [下界, 上界] 及区间宽度"""
    model.eval()
    with torch.no_grad():
        q_lo, q_hi = model(X_test)
    interval_width = q_hi - q_lo          # 区间宽度 = 不确定性指标
    return q_lo, q_hi, interval_width
```

### 方案二：MC Dropout——蒙特卡洛近似贝叶斯推断

Dropout 通常只在训练时使用。**MC Dropout** 的关键思想：在**预测阶段也保持 Dropout 开启**，对同一输入进行 $T$ 次随机前向传播，以样本均值作为预测，以样本方差作为不确定性度量。

理论依据（Gal & Ghahramani, 2016）：带 Dropout 的神经网络可视为对深度高斯过程的**变分贝叶斯近似**，每次前向传播等价于从近似后验 $q(\mathbf{w})$ 中采样一组权重。

```python
import torch
import torch.nn as nn
import numpy as np

# ============================================================
# 方案二：MC Dropout——测试阶段保持 Dropout，多次采样
# ============================================================

class MCDropoutNet(nn.Module):
    """
    MC Dropout 网络：预测时也启用 Dropout（通过 self.train() 模式）
    dropout_p：每个神经元被随机置零的概率
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 output_dim: int = 1, dropout_p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),          # 关键：Dropout 层
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_with_uncertainty(
        self, x: torch.Tensor, T: int = 100
    ) -> dict:
        """
        MC Dropout 预测：T 次随机前向传播，统计预测分布。

        返回字典：
          mean     - 预测均值（点估计）
          variance - 预测方差（不确定性）
          std      - 预测标准差
          ci_lower - 95% 置信下界（均值 ± 1.96 * std）
          ci_upper - 95% 置信上界
        """
        # 必须调用 self.train()，否则 Dropout 在 eval 模式下被关闭
        self.train()

        with torch.no_grad():
            # 堆叠 T 次前向传播结果：形状 (T, batch_size)
            preds = torch.stack([self.net(x).squeeze(-1) for _ in range(T)])

        mean = preds.mean(dim=0)       # 预测均值：近似后验预测均值
        variance = preds.var(dim=0)    # 预测方差：近似认知不确定性
        std = variance.sqrt()

        # 基于正态近似构造 95% 预测区间
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std

        return {
            "mean": mean,
            "variance": variance,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }


def evaluate_coverage(model: MCDropoutNet, X_test: torch.Tensor,
                      y_test: torch.Tensor, T: int = 100) -> float:
    """
    经验覆盖率：计算预测区间实际包含真实值的比例。
    理想情况下，95% 置信区间的覆盖率应接近 0.95。
    """
    result = model.predict_with_uncertainty(X_test, T=T)
    ci_lo = result["ci_lower"]
    ci_hi = result["ci_upper"]

    # 检查每个样本的真实值是否落在预测区间内
    covered = ((y_test >= ci_lo) & (y_test <= ci_hi)).float()
    coverage = covered.mean().item()
    return coverage


# ============================================================
# 使用示例：合成数据
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    n_train, n_test = 500, 100
    input_dim = 5

    # 生成合成回归数据
    X_train = torch.randn(n_train, input_dim)
    y_train = X_train[:, 0] * 2 + X_train[:, 1] - 0.5 + \
              torch.randn(n_train) * 0.5  # 真实噪声 sigma=0.5

    X_test = torch.randn(n_test, input_dim)
    y_test = X_test[:, 0] * 2 + X_test[:, 1] - 0.5 + \
             torch.randn(n_test) * 0.5

    # 训练模型
    model = MCDropoutNet(input_dim=input_dim, dropout_p=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    # 预测与不确定性量化
    result = model.predict_with_uncertainty(X_test, T=200)

    # 评估覆盖率
    coverage = evaluate_coverage(model, X_test, y_test, T=200)
    print(f"MC Dropout 95% 预测区间覆盖率：{coverage:.3f}")
    # 理想值接近 0.95

    # 分析高不确定性样本
    high_unc_idx = result["variance"].topk(5).indices
    print(f"\n不确定性最高的 5 个测试样本（索引）：{high_unc_idx.tolist()}")
    print(f"对应预测方差：{result['variance'][high_unc_idx].tolist()}")
    print(f"对应真实值：{y_test[high_unc_idx].tolist()}")
```

### 方案三：共形预测（Conformal Prediction）

共形预测是一种**无分布假设**的预测区间方法，提供严格的有限样本覆盖率保证，无需假设数据的生成分布。

核心思想：利用**校准集**（calibration set）的残差分布，确定预测区间的宽度，使得预测区间对新样本的经验覆盖率恰好达到 $1-\alpha$。

```python
import torch
import numpy as np

# ============================================================
# 方案三：分割共形预测（Split Conformal Prediction）
# 保证有限样本覆盖率，无需分布假设
# ============================================================

class SplitConformalPredictor:
    """
    分割共形预测器（Split Conformal Prediction）。

    原理：
      1. 将标记数据分为训练集和校准集
      2. 在训练集上训练模型，得到点预测 f(x)
      3. 在校准集上计算非一致性分数（residual）：
         s_i = |y_i - f(x_i)|
      4. 找到校准集分数的 (1-alpha)(1+1/n) 分位数 q_hat
      5. 对测试点，预测区间为 [f(x) - q_hat, f(x) + q_hat]

    理论保证：
      P(y_test ∈ C(x_test)) >= 1 - alpha  （有限样本，无分布假设）
    """

    def __init__(self, base_model: nn.Module):
        self.model = base_model
        self.q_hat = None   # 校准后的分位数阈值

    def calibrate(self, X_cal: torch.Tensor, y_cal: torch.Tensor,
                  alpha: float = 0.05):
        """
        校准步骤：用校准集估计非一致性分数的分位数。

        参数：
          X_cal, y_cal：校准集（独立于训练集，但有标签）
          alpha：误覆盖率，预测区间覆盖率目标为 1-alpha
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_cal)
            if y_pred.dim() > 1:
                y_pred = y_pred.squeeze(-1)

        # 非一致性分数：绝对残差
        scores = (y_cal - y_pred).abs().numpy()  # 形状：(n_cal,)

        # 取校准集分数的 ceil((n+1)(1-alpha)/n) 分位数
        n_cal = len(scores)
        level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        level = min(level, 1.0)             # 不超过 1

        self.q_hat = float(np.quantile(scores, level))
        print(f"校准完成：alpha={alpha}, q_hat={self.q_hat:.4f}")
        return self

    def predict_interval(self, X_test: torch.Tensor):
        """
        推理：为每个测试点生成宽度固定的预测区间。

        返回：
          y_pred   - 点预测值
          ci_lower - 区间下界 = y_pred - q_hat
          ci_upper - 区间上界 = y_pred + q_hat
        """
        if self.q_hat is None:
            raise RuntimeError("请先调用 calibrate() 进行校准。")

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test)
            if y_pred.dim() > 1:
                y_pred = y_pred.squeeze(-1)

        ci_lower = y_pred - self.q_hat
        ci_upper = y_pred + self.q_hat
        return y_pred, ci_lower, ci_upper


# 三种方案的覆盖率对比函数
def compare_coverage(
    y_true: torch.Tensor,
    ci_lower: torch.Tensor,
    ci_upper: torch.Tensor,
    method_name: str = "方法"
) -> None:
    """计算并打印预测区间的经验覆盖率和平均区间宽度。"""
    covered = ((y_true >= ci_lower) & (y_true <= ci_upper)).float()
    coverage = covered.mean().item()
    avg_width = (ci_upper - ci_lower).mean().item()
    print(f"[{method_name}] 覆盖率={coverage:.3f}  平均区间宽度={avg_width:.3f}")
```

### 三种方案对比

| 方法 | 分布假设 | 计算开销 | 覆盖率保证 | 适用场景 |
|------|---------|---------|-----------|---------|
| 分位数回归 | 无 | 低（训练两个头） | 渐近 | 大数据、快速部署 |
| MC Dropout | 近似贝叶斯 | 中（$T$ 次前向） | 近似 | 需要认知不确定性分解 |
| 共形预测 | 无（最弱） | 低（需校准集） | 有限样本严格 | 高可靠性要求场景 |

---

## 练习题

**练习 17.1**（枢轴量法基础）

设 $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$，$\sigma^2 = 4$ 已知，$n=16$，样本均值 $\bar{x} = 12.5$。

(1) 构造 $\mu$ 的枢轴量，写出其分布；

(2) 求 $\mu$ 的 95% 双侧置信区间；

(3) 若要将区间半宽压缩至 0.5，需要多大的样本量？

---

**练习 17.2**（$t$ 置信区间）

某工厂随机抽取 10 件产品，测得其质量（克）为：

$$
98, 102, 99, 101, 100, 103, 97, 104, 99, 101
$$

设质量服从正态分布。

(1) 计算样本均值 $\bar{x}$ 和样本标准差 $s$；

(2) 求总体均值 $\mu$ 的 95% 置信区间；

(3) 求总体方差 $\sigma^2$ 的 90% 置信区间。

---

**练习 17.3**（大样本比例区间）

某选举民调对 1000 人进行调查，其中 480 人表示支持候选人甲。

(1) 求支持率 $p$ 的 95% 置信区间；

(2) 若希望误差限（半宽）不超过 2%，至少需要调查多少人（使用保守估计 $p=0.5$）？

(3) 若实际支持率约为 48%，用更精确的方式重新计算所需样本量。

---

**练习 17.4**（两样本区间估计）

两台机器加工相同零件，分别抽取样本：

- 机器 A：$m=12$，$\bar{x}_A = 50.05$，$s_A = 0.05$（毫米）
- 机器 B：$n=10$，$\bar{x}_B = 50.03$，$s_B = 0.04$（毫米）

设两总体均服从正态分布，且方差相等。

(1) 计算合并样本标准差 $s_p$；

(2) 求均值差 $\mu_A - \mu_B$ 的 95% 置信区间；

(3) 根据置信区间，能否认为两台机器的加工精度（均值）无显著差异？

---

**练习 17.5**（单侧置信限与预测区间）

某品牌灯泡寿命 $X \sim \mathcal{N}(\mu, \sigma^2)$，随机抽取 9 只，测得寿命（小时）：

$$
1200, 1350, 1280, 1420, 1300, 1250, 1380, 1320, 1290
$$

(1) 计算 $\bar{x}$ 和 $s$；

(2) 求 $\mu$ 的 95% 单侧置信下限（保证平均寿命不低于某值）；

(3) 求 $\sigma^2$ 的 95% 单侧置信上限（控制寿命波动的上界）；

(4) 构造单个新灯泡寿命 $X_{\text{new}}$ 的 95% 预测区间：

$$
\bar{X} \pm t_{\alpha/2}(n-1) \cdot S \sqrt{1 + \frac{1}{n}}
$$

比较预测区间与置信区间的宽度，解释差异来源。

---

## 练习答案

### 答案 17.1

**(1) 枢轴量**

$$
G = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} = \frac{\bar{X} - \mu}{2/\sqrt{16}} = \frac{\bar{X} - \mu}{0.5} \sim \mathcal{N}(0,1)
$$

**(2) 95% 置信区间**

$z_{0.025} = 1.96$，误差限 $= 1.96 \times 0.5 = 0.98$。

$$
\mu \in [12.5 - 0.98,\; 12.5 + 0.98] = [11.52,\; 13.48]
$$

**(3) 所需样本量**

要求 $z_{0.025} \cdot \dfrac{\sigma}{\sqrt{n}} \leq 0.5$，即

$$
\sqrt{n} \geq \frac{1.96 \times 2}{0.5} = 7.84 \implies n \geq 7.84^2 = 61.5
$$

故至少需要 $n = 62$ 个样本。

---

### 答案 17.2

**(1) 样本均值和标准差**

$$
\bar{x} = \frac{98+102+99+101+100+103+97+104+99+101}{10} = \frac{1004}{10} = 100.4
$$

各偏差的平方：$(-2.4)^2, (1.6)^2, (-1.4)^2, (0.6)^2, (-0.4)^2, (2.6)^2, (-3.4)^2, (3.6)^2, (-1.4)^2, (0.6)^2$

$= 5.76, 2.56, 1.96, 0.36, 0.16, 6.76, 11.56, 12.96, 1.96, 0.36$

$$
\sum_{i=1}^{10}(x_i - \bar{x})^2 = 44.4
$$

$$
s^2 = \frac{44.4}{9} \approx 4.933, \quad s \approx 2.221
$$

**(2) 均值的 95% 置信区间**

$t_{0.025}(9) = 2.262$，误差限 $= 2.262 \times \dfrac{2.221}{\sqrt{10}} = 2.262 \times 0.702 = 1.588$。

$$
\mu \in [100.4 - 1.588,\; 100.4 + 1.588] = [98.81,\; 101.99] \text{（克）}
$$

**(3) 方差的 90% 置信区间**

$\chi^2_{0.05}(9) = 16.919$，$\chi^2_{0.95}(9) = 3.325$，$(n-1)s^2 = 9 \times 4.933 = 44.4$。

$$
\sigma^2 \in \left[\frac{44.4}{16.919},\; \frac{44.4}{3.325}\right] = [2.624,\; 13.353] \text{（克}^2\text{）}
$$

---

### 答案 17.3

**(1) 支持率的 95% 置信区间**

$\hat{p} = 480/1000 = 0.48$，$z_{0.025} = 1.96$。

$$
\sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = \sqrt{\frac{0.48 \times 0.52}{1000}} = \sqrt{0.0002496} \approx 0.01580
$$

$$
p \in [0.48 - 1.96 \times 0.01580,\; 0.48 + 1.96 \times 0.01580] = [0.449,\; 0.511]
$$

**(2) 保守估计所需样本量**（$p = 0.5$）

$$
n \geq \left(\frac{z_{\alpha/2}}{2\delta_0}\right)^2 = \left(\frac{1.96}{2 \times 0.02}\right)^2 = \left(\frac{1.96}{0.04}\right)^2 = 49^2 = 2401
$$

至少需要 2401 人。

**(3) 精确估计所需样本量**（$p \approx 0.48$）

$$
n \geq \left(\frac{z_{\alpha/2}}{\delta_0}\right)^2 p(1-p) = \left(\frac{1.96}{0.02}\right)^2 \times 0.48 \times 0.52 = 9604 \times 0.2496 \approx 2397
$$

约需 2397 人（与保守估计接近，因 $p$ 接近 0.5）。

---

### 答案 17.4

**(1) 合并标准差**

$$
s_p^2 = \frac{(m-1)s_A^2 + (n-1)s_B^2}{m+n-2} = \frac{11 \times 0.0025 + 9 \times 0.0016}{20} = \frac{0.0275 + 0.0144}{20} = \frac{0.0419}{20} = 0.002095
$$

$$
s_p = \sqrt{0.002095} \approx 0.04577 \text{（毫米）}
$$

**(2) 均值差的 95% 置信区间**

$t_{0.025}(20) = 2.086$，$\bar{x}_A - \bar{x}_B = 50.05 - 50.03 = 0.02$。

$$
s_p\sqrt{\frac{1}{m} + \frac{1}{n}} = 0.04577\sqrt{\frac{1}{12} + \frac{1}{10}} = 0.04577 \times \sqrt{0.1833} = 0.04577 \times 0.4282 = 0.01960
$$

误差限 $= 2.086 \times 0.01960 = 0.04089$。

$$
\mu_A - \mu_B \in [0.02 - 0.04089,\; 0.02 + 0.04089] = [-0.021,\; 0.061] \text{（毫米）}
$$

**(3) 结论**

区间包含 0，故在 95% 置信水平下，**不能认为两台机器的平均加工尺寸有显著差异**。

---

### 答案 17.5

**(1) 样本均值和标准差**

$$
\bar{x} = \frac{1200+1350+1280+1420+1300+1250+1380+1320+1290}{9} = \frac{11790}{9} = 1310
$$

各偏差：$-110, 40, -30, 110, -10, -60, 70, 10, -20$

$$
\sum (x_i - \bar{x})^2 = 12100 + 1600 + 900 + 12100 + 100 + 3600 + 4900 + 100 + 400 = 35800
$$

$$
s^2 = \frac{35800}{8} = 4475, \quad s = \sqrt{4475} \approx 66.9 \text{（小时）}
$$

**(2) 均值的 95% 单侧置信下限**

$t_{0.05}(8) = 1.860$。

$$
L = \bar{x} - t_{0.05}(8) \cdot \frac{s}{\sqrt{n}} = 1310 - 1.860 \times \frac{66.9}{\sqrt{9}} = 1310 - 1.860 \times 22.3 = 1310 - 41.5 = 1268.5
$$

有 95% 把握认为平均寿命不低于 1268.5 小时。

**(3) 方差的 95% 单侧置信上限**

$\chi^2_{0.05}(8) = 2.733$（下侧 5% 分位数），$(n-1)s^2 = 8 \times 4475 = 35800$。

$$
U = \frac{(n-1)s^2}{\chi^2_{1-\alpha}(n-1)} = \frac{35800}{\chi^2_{0.05}(8)} = \frac{35800}{2.733} \approx 13101
$$

即 $\sigma^2 \leq 13101$，$\sigma \leq 114.5$ 小时（95% 置信）。

**(4) 单个灯泡寿命的 95% 预测区间**

$t_{0.025}(8) = 2.306$。

$$
X_{\text{new}} \in \left[1310 \pm 2.306 \times 66.9 \times \sqrt{1 + \frac{1}{9}}\right] = \left[1310 \pm 2.306 \times 66.9 \times 1.054\right]
$$

$$
= \left[1310 \pm 162.8\right] = [1147.2,\; 1472.8] \text{（小时）}
$$

**对比**：置信区间（均值）半宽 $= 2.306 \times 66.9/3 = 51.4$；预测区间（单个）半宽 $= 162.8$，**约为置信区间的 3 倍宽**。

**差异来源**：置信区间描述**均值 $\mu$ 的不确定性**（来源于抽样误差，$\propto 1/\sqrt{n}$）；预测区间还需加入**单个观测值相对均值的随机波动**（$\sigma$），因此公式中有 $\sqrt{1+1/n}$ 而非 $\sqrt{1/n}$，两者之比约为 $\sqrt{1+1/n}/(1/\sqrt{n}) = \sqrt{n+1}$。
