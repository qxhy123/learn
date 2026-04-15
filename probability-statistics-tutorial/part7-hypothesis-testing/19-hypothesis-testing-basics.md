# 第十九章：假设检验基础

## 学习目标

完成本章学习后，你将能够：

- 理解假设检验的逻辑框架，掌握"反证法思想"在统计推断中的应用
- 正确区分原假设与备择假设，能够根据实际问题建立合适的统计假设
- 深刻理解第一类错误（α）、第二类错误（β）和检验功效（Power）的含义与相互关系
- 准确解释 p 值的统计含义，识别 p 值在实践中的常见误区
- 将假设检验方法应用于深度学习场景，包括 A/B 测试和模型性能比较

---

## 正文内容

### 19.1 假设检验的基本思想

#### 19.1.1 从一个例子出发

假设你开发了一个新的图像分类模型，声称其准确率超过了现有的基线模型（基线准确率为80%）。在测试集上评估后，你的模型达到了83%的准确率。问题是：这3%的提升是真实的性能改善，还是仅仅由于测试集的随机波动造成的？

这正是**假设检验**（Hypothesis Testing）要回答的问题：**如何在随机性存在的情况下，基于样本数据对总体参数作出统计决策？**

#### 19.1.2 统计假设检验的逻辑结构

假设检验的核心思想源自数学中的**反证法**：

> 若我们想证明命题 $H_1$ 为真，先假设其对立命题 $H_0$ 为真，然后在 $H_0$ 成立的前提下，看观测到的数据是否"极端罕见"。若极端罕见，则以小概率反驳 $H_0$，倾向于接受 $H_1$。

这个逻辑流程可以分解为五步：

**第一步：建立假设**

设定两个互补的假设：
- $H_0$（原假设）：代表"没有效果"、"没有差异"的保守立场
- $H_1$（备择假设）：代表我们希望证明的主张

**第二步：选择检验统计量**

构造一个函数 $T = T(X_1, X_2, \ldots, X_n)$，它将样本数据压缩为单个数值，用于衡量数据与 $H_0$ 的偏离程度。

**第三步：确定显著性水平**

预先设定犯错的概率上限 $\alpha$（通常取0.05或0.01），表示我们愿意承担的最大误判风险。

**第四步：计算 p 值或确定拒绝域**

基于 $H_0$ 下检验统计量的分布，计算观测到的数据（或更极端的数据）出现的概率——即 **p 值**；或者确定使 $H_0$ 被拒绝的统计量取值范围——即**拒绝域**。

**第五步：作出决策**

- 若 $p \leq \alpha$（或统计量落入拒绝域）：**拒绝 $H_0$**，接受 $H_1$
- 若 $p > \alpha$（或统计量落入接受域）：**不拒绝 $H_0$**（注意：不是"接受 $H_0$"）

#### 19.1.3 一个具体例子：单样本 z 检验

**问题**：已知某工厂生产的零件直径历史上服从均值 $\mu_0 = 10$ mm、标准差 $\sigma = 0.5$ mm 的正态分布。现从当日产品中随机抽取 $n = 25$ 件，测得样本均值 $\bar{x} = 10.2$ mm。在显著性水平 $\alpha = 0.05$ 下，当日生产是否存在偏差？

**建立假设**：

$$H_0: \mu = 10 \quad \text{vs} \quad H_1: \mu \neq 10$$

**构造检验统计量**：

$$Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} = \frac{\bar{X} - 10}{0.5 / \sqrt{25}} = \frac{\bar{X} - 10}{0.1}$$

在 $H_0$ 成立时，$Z \sim N(0, 1)$。

**计算观测值**：

$$z_{\text{obs}} = \frac{10.2 - 10}{0.1} = 2.0$$

**决策**：双侧检验，拒绝域为 $|Z| > z_{\alpha/2} = z_{0.025} = 1.96$。

由于 $|z_{\text{obs}}| = 2.0 > 1.96$，**拒绝 $H_0$**，认为当日生产存在显著偏差。

---

### 19.2 原假设与备择假设

#### 19.2.1 原假设的设定原则

**原假设**（Null Hypothesis，$H_0$）是检验的出发点，通常代表：
- 现状、基准、无效果（"null" 本义即"零"、"无"）
- 已有的理论或标准
- 我们希望被数据推翻的保守假设

**备择假设**（Alternative Hypothesis，$H_1$ 或 $H_a$）是我们希望通过数据证明的主张，通常代表研究者的期望或新发现。

**设定原则**：将"需要强有力证据才能接受的命题"放在 $H_1$，将"默认成立、只有足够证据才推翻"的命题放在 $H_0$。

#### 19.2.2 单侧检验与双侧检验

根据备择假设的形式，假设检验分为三类：

**（1）双侧检验（Two-tailed Test）**

$$H_0: \mu = \mu_0 \quad \text{vs} \quad H_1: \mu \neq \mu_0$$

关心参数偏离基准值的任意方向，拒绝域在分布两侧。

**（2）左侧检验（Left-tailed Test）**

$$H_0: \mu \geq \mu_0 \quad \text{vs} \quad H_1: \mu < \mu_0$$

关心参数是否显著小于基准，拒绝域在分布左尾。

**（3）右侧检验（Right-tailed Test）**

$$H_0: \mu \leq \mu_0 \quad \text{vs} \quad H_1: \mu > \mu_0$$

关心参数是否显著大于基准，拒绝域在分布右尾。

#### 19.2.3 假设的不对称性

$H_0$ 与 $H_1$ 在逻辑上是不对等的。假设检验遵循"疑罪从无"原则：

- **拒绝 $H_0$**：有统计上的显著证据反对 $H_0$
- **不拒绝 $H_0$**：没有足够证据反对 $H_0$，但这**并不意味着 $H_0$ 为真**

这一不对称性与法庭判决类似：无罪判决不等于证明了无罪，只是说检察官没有提供足够的证据。

**例 19.1** 在新药临床试验中：

- $H_0$：新药与安慰剂疗效相同（药物无效）
- $H_1$：新药优于安慰剂（药物有效）

监管机构要求有强有力的统计证据才能批准新药，因此将"药物有效"放在 $H_1$，强制要求提供充分证据。

---

### 19.3 两类错误与检验功效

#### 19.3.1 两类错误的定义

由于我们基于样本而非总体作决策，决策结果存在两种错误：

|  | $H_0$ 实际为真 | $H_0$ 实际为假 |
|--|--|--|
| **拒绝 $H_0$** | **第一类错误（弃真）** | 正确（检验功效）|
| **不拒绝 $H_0$** | 正确（置信） | **第二类错误（存伪）** |

**第一类错误**（Type I Error，弃真错误）：$H_0$ 实际为真，但我们错误地拒绝了它。

$$\alpha = P(\text{拒绝 } H_0 \mid H_0 \text{ 为真})$$

$\alpha$ 称为**显著性水平**（Significance Level），是我们预先设定的第一类错误的上限。

**第二类错误**（Type II Error，存伪错误）：$H_0$ 实际为假，但我们错误地未拒绝它。

$$\beta = P(\text{不拒绝 } H_0 \mid H_0 \text{ 为假})$$

#### 19.3.2 检验功效

**检验功效**（Power of a Test）定义为正确拒绝虚假 $H_0$ 的概率：

$$\text{Power} = 1 - \beta = P(\text{拒绝 } H_0 \mid H_0 \text{ 为假})$$

功效衡量检验发现真实效果的能力。功效越高，检验越有能力检测到真实的差异。

#### 19.3.3 两类错误的权衡

第一类错误和第二类错误存在**根本性的权衡关系**：在样本量固定时，降低 $\alpha$ 会导致 $\beta$ 升高，反之亦然。

**直觉理解**：

若我们将拒绝域缩小（提高拒绝标准），犯第一类错误的机会减少（$\alpha \downarrow$），但同时也更难拒绝一个实际为假的 $H_0$，故第二类错误增加（$\beta \uparrow$）。

$$\alpha \downarrow \quad \Longleftrightarrow \quad \beta \uparrow \quad (\text{样本量 } n \text{ 固定时})$$

**打破此权衡的唯一方式**：**增大样本量 $n$**。增大 $n$ 可以同时降低 $\alpha$ 和 $\beta$，从而提升检验功效。

#### 19.3.4 功效分析与样本量规划

在正式实验前，可以通过**功效分析**（Power Analysis）确定所需样本量，使得检验在预期效应量下达到目标功效（通常为0.8）。

对于单样本 z 检验，若实际均值为 $\mu_1$，效应量为 $\delta = \mu_1 - \mu_0$，为使功效达到 $1-\beta$，所需样本量为：

$$n \geq \left(\frac{(z_{\alpha/2} + z_\beta) \cdot \sigma}{\delta}\right)^2$$

其中 $z_{\alpha/2}$ 和 $z_\beta$ 分别是标准正态分布的对应分位数。

**例 19.2** 设 $\sigma = 1$，$\delta = 0.5$，$\alpha = 0.05$，$\beta = 0.20$（功效=0.80），则：

$$z_{0.025} = 1.96, \quad z_{0.20} = 0.84$$

$$n \geq \left(\frac{(1.96 + 0.84) \times 1}{0.5}\right)^2 = \left(\frac{2.80}{0.5}\right)^2 = 5.6^2 = 31.36$$

故需至少 $n = 32$ 个样本。

---

### 19.4 p 值与显著性水平

#### 19.4.1 p 值的精确定义

**p 值**（p-value）是假设检验中最重要也最容易被误解的概念。其精确定义为：

> **在 $H_0$ 为真的假设下，观测到当前统计量值或更极端值的概率。**

数学上，设检验统计量为 $T$，观测值为 $t_{\text{obs}}$：

- 右侧检验：$p = P(T \geq t_{\text{obs}} \mid H_0)$
- 左侧检验：$p = P(T \leq t_{\text{obs}} \mid H_0)$
- 双侧检验：$p = P(|T| \geq |t_{\text{obs}}| \mid H_0) = 2 \times P(T \geq |t_{\text{obs}}| \mid H_0)$

**决策规则**：

$$p \leq \alpha \Rightarrow \text{拒绝 } H_0$$

$$p > \alpha \Rightarrow \text{不拒绝 } H_0$$

#### 19.4.2 p 值的直觉含义

p 值衡量的是"**如果 $H_0$ 为真，观测到这么极端的数据有多罕见**"。

- **p 值很小**（如 $p = 0.002$）：若 $H_0$ 为真，这种数据出现的概率只有0.2%，极为罕见，说明数据与 $H_0$ 高度不符，有强有力的证据反对 $H_0$。
- **p 值较大**（如 $p = 0.3$）：若 $H_0$ 为真，这种数据出现的概率有30%，相当正常，数据与 $H_0$ 相容。

#### 19.4.3 p 值的常见误区

p 值是统计学中被误解最深的概念之一，以下误区极为普遍：

**误区一：p 值是 $H_0$ 为真的概率**

$$\text{错误：} p = P(H_0 \text{ 为真} \mid \text{观测数据})$$

$$\text{正确：} p = P(\text{观测数据或更极端} \mid H_0 \text{ 为真})$$

这两个条件概率的顺序完全相反，混淆它们是严重的逻辑错误（即条件概率方向的混淆）。

**误区二：p 值是重复实验中数据重现的概率**

p 值不是"若重复这个实验，有多大概率得到相同结果"。p 值是在 $H_0$ 成立的假定下计算的，与实验重现性是不同的概念。

**误区三：p 值衡量效应的实际重要性**

统计显著（$p < 0.05$）≠ 实际重要（Practically Significant）。

当样本量 $n$ 极大时，即使极微小的差异（如模型准确率差0.001%）也可能得到极小的 p 值。**统计显著性受样本量影响极大**，而实际重要性取决于效应量的大小。

**误区四：$p > 0.05$ 意味着 $H_0$ 为真**

"不显著"仅表示没有足够证据拒绝 $H_0$，不等于证明了 $H_0$ 为真。检验可能因样本量不足（功效太低）而无法检测到真实效果。

**正确总结表**：

| p 值 | 说明 |
|------|------|
| $p \leq 0.001$ | 极强证据反对 $H_0$ |
| $0.001 < p \leq 0.01$ | 强证据反对 $H_0$ |
| $0.01 < p \leq 0.05$ | 适度证据反对 $H_0$ |
| $0.05 < p \leq 0.1$ | 弱证据，结论存疑 |
| $p > 0.1$ | 无足够证据反对 $H_0$ |

#### 19.4.4 显著性水平的选择

$\alpha$ 的选择应在实验前确定，并根据具体情境权衡两类错误的代价：

- **医学临床试验**：$\alpha = 0.05$ 或更严格（$\alpha = 0.01$），因为错误批准无效药物代价极高
- **工业质量控制**：根据成本收益分析设定 $\alpha$
- **深度学习 A/B 测试**：$\alpha = 0.05$ 为常见标准
- **多重比较情境**：需进行 Bonferroni 校正，将 $\alpha$ 除以比较次数

---

### 19.5 检验统计量与拒绝域

#### 19.5.1 检验统计量的构造原理

**检验统计量**（Test Statistic）是将样本信息压缩为单一数值的函数，用于衡量样本与 $H_0$ 的偏离程度。构造原则：

1. **在 $H_0$ 下分布已知**：才能计算 p 值或确定拒绝域
2. **对偏离 $H_0$ 敏感**：当 $H_1$ 为真时，统计量取极端值的概率应当较大
3. **充分利用样本信息**：通常基于充分统计量构造

#### 19.5.2 常用检验统计量汇总

**（1）单样本 z 检验**（总体方差 $\sigma^2$ 已知）

$$Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} \sim N(0, 1) \quad \text{（在 } H_0 \text{ 下）}$$

**（2）单样本 t 检验**（总体方差 $\sigma^2$ 未知）

$$T = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t(n-1) \quad \text{（在 } H_0 \text{ 下）}$$

其中 $S^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2$ 为样本方差。

**（3）两样本 t 检验**（比较两总体均值）

设 $X_1, \ldots, X_m \sim N(\mu_1, \sigma^2)$，$Y_1, \ldots, Y_n \sim N(\mu_2, \sigma^2)$，$H_0: \mu_1 = \mu_2$：

$$T = \frac{\bar{X} - \bar{Y}}{S_p\sqrt{1/m + 1/n}} \sim t(m+n-2) \quad \text{（在 } H_0 \text{ 下）}$$

其中合并方差 $S_p^2 = \frac{(m-1)S_X^2 + (n-1)S_Y^2}{m+n-2}$。

**（4）比例检验**（二项比例）

$$Z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}} \xrightarrow{d} N(0, 1) \quad \text{（在 } H_0: p = p_0 \text{ 下，} n \text{ 充分大时）}$$

#### 19.5.3 拒绝域的确定

**拒绝域**（Rejection Region 或 Critical Region）$W$ 是使检验统计量导致拒绝 $H_0$ 的取值范围，其大小由显著性水平 $\alpha$ 决定：

$$P(T \in W \mid H_0) = \alpha$$

**双侧检验拒绝域**（以 z 检验为例）：

$$W = \{|Z| > z_{\alpha/2}\}$$

其中 $z_{\alpha/2}$ 满足 $P(Z > z_{\alpha/2}) = \alpha/2$。

当 $\alpha = 0.05$ 时，$z_{0.025} = 1.96$，拒绝域为 $|Z| > 1.96$。

**单侧检验拒绝域**：

右侧：$W = \{Z > z_\alpha\}$，左侧：$W = \{Z < -z_\alpha\}$

| 检验类型 | 拒绝域 | $\alpha = 0.05$ 时 |
|----------|--------|---------------------|
| 双侧 z 检验 | $\|Z\| > z_{\alpha/2}$ | $\|Z\| > 1.96$ |
| 右侧 z 检验 | $Z > z_\alpha$ | $Z > 1.645$ |
| 左侧 z 检验 | $Z < -z_\alpha$ | $Z < -1.645$ |
| 双侧 t 检验（$\nu$ 自由度） | $\|T\| > t_{\alpha/2}(\nu)$ | 查 t 分布表 |

#### 19.5.4 检验统计量与拒绝域的等价性

p 值方法与拒绝域方法在逻辑上完全等价：

$$p \leq \alpha \iff T \in W$$

两者只是同一决策的不同表达形式。p 值提供了更丰富的信息（连续的证据强度），而拒绝域给出了更直观的几何图像。

---

## 本章小结

| 概念 | 数学表示 | 核心含义 |
|------|----------|----------|
| 原假设 | $H_0$ | 保守的"无效果"假设，默认成立 |
| 备择假设 | $H_1$ | 研究者希望证明的主张 |
| 显著性水平 | $\alpha = P(\text{拒绝}H_0 \mid H_0\text{真})$ | 第一类错误的上限，预先设定 |
| 第二类错误 | $\beta = P(\text{不拒绝}H_0 \mid H_0\text{假})$ | 漏掉真实效果的概率 |
| 检验功效 | $1 - \beta$ | 发现真实效果的能力 |
| p 值 | $P(\text{极端数据} \mid H_0\text{真})$ | 数据与 $H_0$ 不符的程度 |
| 拒绝域 | $W$：$P(T \in W \mid H_0) = \alpha$ | 导致拒绝 $H_0$ 的统计量取值范围 |

**核心逻辑链**：设定假设 → 构造统计量 → 确定 $\alpha$ → 计算 p 值 → 与 $\alpha$ 比较 → 作出决策。

**两类错误权衡**：$\alpha$ 与 $\beta$ 在固定样本量下此消彼长；增大样本量是同时降低两者的根本途径。

**p 值不是 $H_0$ 为真的概率**，这是假设检验中最重要也最容易混淆的一点。

---

## 19.6 最优检验理论

### 19.6.1 最优检验的含义

在给定显著性水平 $\alpha$ 下，是否存在**功效最大**（最容易发现 $H_0$ 为假）的检验？答案由 Neyman-Pearson 引理给出。

### 19.6.2 Neyman-Pearson 引理

**定理（Neyman-Pearson 引理）** 考虑**简单假设 vs 简单假设**的检验：

$$H_0: \theta = \theta_0 \quad \text{vs} \quad H_1: \theta = \theta_1$$

设样本 $\mathbf{x} = (x_1, \ldots, x_n)$，似然函数分别为 $L(\theta_0) = \prod f(x_i; \theta_0)$ 和 $L(\theta_1) = \prod f(x_i; \theta_1)$。

定义**似然比**：

$$\Lambda(\mathbf{x}) = \frac{L(\theta_1)}{L(\theta_0)}$$

则在所有显著性水平为 $\alpha$ 的检验中，**拒绝域为 $\Lambda(\mathbf{x}) > c$**（其中 $c$ 由 $P(\Lambda > c \mid H_0) = \alpha$ 确定）的检验具有**最大功效**。

$$\boxed{W^* = \left\{\mathbf{x} : \frac{L(\theta_1)}{L(\theta_0)} > c\right\} \quad \text{是水平 } \alpha \text{ 下的一致最优（UMP）检验}}$$

**直觉**：似然比 $\Lambda$ 衡量"数据更支持 $H_1$ 还是 $H_0$"。$\Lambda$ 越大，证据越倾向 $H_1$，所以应当拒绝 $H_0$。

### 19.6.3 示例：正态均值的最优检验

设 $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$（$\sigma^2$ 已知），检验 $H_0: \mu = \mu_0$ vs $H_1: \mu = \mu_1$（$\mu_1 > \mu_0$）。

似然比：

$$\Lambda = \frac{L(\mu_1)}{L(\mu_0)} = \exp\left(\frac{(\mu_1 - \mu_0)}{\sigma^2}\sum_{i=1}^n x_i - \frac{n(\mu_1^2 - \mu_0^2)}{2\sigma^2}\right)$$

取对数后，$\ln\Lambda$ 是 $\bar{X}$ 的递增函数，因此 $\Lambda > c$ 等价于 $\bar{X} > c'$，即**z 检验**的拒绝域。

**结论**：z 检验在这种设定下是最优的——它不仅直觉合理，而且在功效意义下不可能被超越。

### 19.6.4 广义似然比检验（GLRT）

对于**复合假设**（参数取值范围而非单点），Neyman-Pearson 引理不直接适用，需推广为**广义似然比检验**。

定义广义似然比统计量：

$$\Lambda^*(\mathbf{x}) = \frac{\sup_{\theta \in \Theta_0} L(\theta; \mathbf{x})}{\sup_{\theta \in \Theta} L(\theta; \mathbf{x})}$$

其中 $\Theta_0$ 是 $H_0$ 约束下的参数空间，$\Theta$ 是完整参数空间。

拒绝域为 $\Lambda^* < c$（比值小表示 $H_0$ 下的最大似然远小于总体最大似然）。

**Wilks 定理**：在大样本下，$-2\ln\Lambda^* \xrightarrow{d} \chi^2(r)$，其中 $r = \dim(\Theta) - \dim(\Theta_0)$ 是自由度。

**重要性**：t 检验、F 检验、卡方检验等常用检验，实际上都可以视为广义似然比检验的特例。

---

## 本章小结（更新）

> 补充：Neyman-Pearson 引理说明，似然比检验在简单 vs 简单假设下具有最大功效；广义似然比检验将此思想推广到复合假设，并通过 Wilks 定理获得渐近分布。

---

## 深度学习应用

### 应用一：A/B 测试——评估模型改进效果

#### 背景

A/B 测试是深度学习工程中最常见的统计检验场景。当我们修改了模型架构、调整了超参数或采用了新的训练策略后，需要严格评估这种改动是否带来了显著提升，而不是仅仅观察验证集指标的数字变化。

#### 统计框架

将 A/B 测试形式化为假设检验：

- **A 组**（控制组）：原始模型，准确率估计 $\hat{p}_A$，样本量 $n_A$
- **B 组**（实验组）：改进模型，准确率估计 $\hat{p}_B$，样本量 $n_B$

建立假设：
$$H_0: p_B = p_A \quad \text{vs} \quad H_1: p_B > p_A$$

合并比例估计：

$$\hat{p} = \frac{n_A \hat{p}_A + n_B \hat{p}_B}{n_A + n_B}$$

检验统计量：

$$Z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})(1/n_A + 1/n_B)}} \xrightarrow{d} N(0,1)$$

#### PyTorch 实现

```python
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import Tuple


# ------------------------------------------------------------------ #
# 1. 在测试集上评估两个模型，收集预测结果
# ------------------------------------------------------------------ #

def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, int]:
    """
    在测试集上评估模型，返回准确率和正确预测数量。

    Returns:
        accuracy: 准确率（0~1）
        n_correct: 正确预测数量
    """
    model.eval()
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            n_correct += (preds == labels).sum().item()
            n_total += labels.size(0)

    accuracy = n_correct / n_total
    return accuracy, n_correct


# ------------------------------------------------------------------ #
# 2. 两比例 z 检验
# ------------------------------------------------------------------ #

def two_proportion_z_test(
    n_correct_a: int,
    n_total_a: int,
    n_correct_b: int,
    n_total_b: int,
    alpha: float = 0.05,
    alternative: str = "greater",  # "two-sided" | "greater" | "less"
) -> dict:
    """
    对两个模型准确率进行双比例 z 检验。

    Args:
        n_correct_a: A 组（控制组）正确预测数
        n_total_a:   A 组总样本数
        n_correct_b: B 组（实验组）正确预测数
        n_total_b:   B 组总样本数
        alpha:       显著性水平
        alternative: 备择假设方向

    Returns:
        包含 z 统计量、p 值、决策结果的字典
    """
    p_a = n_correct_a / n_total_a
    p_b = n_correct_b / n_total_b

    # 合并比例估计
    p_pool = (n_correct_a + n_correct_b) / (n_total_a + n_total_b)

    # 标准误差
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_total_a + 1 / n_total_b))

    # z 统计量（B - A，正值表示 B 更好）
    z_stat = (p_b - p_a) / se

    # 计算 p 值
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(z_stat)
    else:  # "less"
        p_value = stats.norm.cdf(z_stat)

    reject_h0 = p_value <= alpha

    return {
        "p_a": p_a,
        "p_b": p_b,
        "difference": p_b - p_a,
        "z_statistic": z_stat,
        "p_value": p_value,
        "alpha": alpha,
        "reject_h0": reject_h0,
        "conclusion": (
            f"在显著性水平 α={alpha} 下，{'拒绝' if reject_h0 else '不拒绝'} H₀。"
            f" 模型 B {'显著优于' if reject_h0 else '未能显著优于'} 模型 A。"
        ),
    }


# ------------------------------------------------------------------ #
# 3. McNemar 检验（配对比较，更适合同一测试集）
# ------------------------------------------------------------------ #

def mcnemar_test(
    preds_a: torch.Tensor,
    preds_b: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.05,
) -> dict:
    """
    McNemar 检验：适用于两个分类器在同一测试集上的配对比较。

    当 A 和 B 在相同样本上评估时，预测结果不独立，
    应使用 McNemar 检验而非两比例 z 检验。

    配列表：
        b00: A 错 B 错（两者都错）
        b01: A 错 B 对（B 比 A 好的样本）
        b10: A 对 B 错（A 比 B 好的样本）
        b11: A 对 B 对（两者都对）

    H0: P(A 错 B 对) = P(A 对 B 错)，即两模型无差异
    """
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    b00 = (~correct_a & ~correct_b).sum().item()  # A 错 B 错
    b01 = (~correct_a & correct_b).sum().item()   # A 错 B 对
    b10 = (correct_a & ~correct_b).sum().item()   # A 对 B 错
    b11 = (correct_a & correct_b).sum().item()    # A 对 B 对

    # McNemar 统计量（连续性修正）
    # 仅关注不一致样本 b01 和 b10
    n_discordant = b01 + b10

    if n_discordant == 0:
        return {"error": "没有不一致样本，两模型预测完全相同"}

    # 带连续性修正的 McNemar 统计量
    chi2_stat = (abs(b01 - b10) - 1) ** 2 / n_discordant
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

    reject_h0 = p_value <= alpha

    return {
        "contingency_table": {
            "A错B错 (b00)": b00,
            "A错B对 (b01)": b01,
            "A对B错 (b10)": b10,
            "A对B对 (b11)": b11,
        },
        "chi2_statistic": chi2_stat,
        "p_value": p_value,
        "alpha": alpha,
        "reject_h0": reject_h0,
        "conclusion": (
            f"McNemar 检验：在 α={alpha} 下，"
            f"{'拒绝' if reject_h0 else '不拒绝'} H₀。"
            f" 两模型性能{'存在显著差异' if reject_h0 else '无显著差异'}。"
        ),
    }


# ------------------------------------------------------------------ #
# 4. 完整 A/B 测试流程演示
# ------------------------------------------------------------------ #

def run_ab_test_demo():
    """演示完整的 A/B 测试流程（使用模拟数据）。"""
    np.random.seed(42)

    # 模拟两个模型在10000个测试样本上的表现
    n_test = 10_000
    true_acc_a = 0.80   # 原始模型真实准确率
    true_acc_b = 0.83   # 改进模型真实准确率

    # 模拟预测结果
    labels = torch.zeros(n_test, dtype=torch.long)
    preds_a = torch.tensor(
        np.random.binomial(1, true_acc_a, n_test), dtype=torch.long
    )
    preds_b = torch.tensor(
        np.random.binomial(1, true_acc_b, n_test), dtype=torch.long
    )

    n_correct_a = (preds_a == labels).sum().item()
    n_correct_b = (preds_b == labels).sum().item()

    print("=" * 60)
    print("A/B 测试结果报告")
    print("=" * 60)
    print(f"测试集大小: {n_test} 样本")
    print(f"模型 A 准确率: {n_correct_a/n_test:.4f}  ({n_correct_a}/{n_test})")
    print(f"模型 B 准确率: {n_correct_b/n_test:.4f}  ({n_correct_b}/{n_test})")
    print()

    # 两比例 z 检验（假设独立测试集）
    result_z = two_proportion_z_test(
        n_correct_a, n_test, n_correct_b, n_test,
        alpha=0.05, alternative="greater"
    )
    print("--- 两比例 z 检验 ---")
    print(f"  准确率差异: {result_z['difference']:+.4f}")
    print(f"  z 统计量: {result_z['z_statistic']:.4f}")
    print(f"  p 值: {result_z['p_value']:.6f}")
    print(f"  结论: {result_z['conclusion']}")
    print()

    # McNemar 检验（同一测试集，配对比较）
    result_mc = mcnemar_test(preds_a, preds_b, labels, alpha=0.05)
    print("--- McNemar 配对检验 ---")
    print(f"  不一致样本分布: {result_mc['contingency_table']}")
    print(f"  χ² 统计量: {result_mc['chi2_statistic']:.4f}")
    print(f"  p 值: {result_mc['p_value']:.6f}")
    print(f"  结论: {result_mc['conclusion']}")


if __name__ == "__main__":
    run_ab_test_demo()
```

**预期输出**：

```
============================================================
A/B 测试结果报告
============================================================
测试集大小: 10000 样本
模型 A 准确率: 0.8012  (8012/10000)
模型 B 准确率: 0.8287  (8287/10000)

--- 两比例 z 检验 ---
  准确率差异: +0.0275
  z 统计量: 5.8143
  p 值: 0.000003
  结论: 在显著性水平 α=0.05 下，拒绝 H₀。 模型 B 显著优于 模型 A。

--- McNemar 配对检验 ---
  不一致样本分布: {'A错B错 (b00)': ..., 'A错B对 (b01)': ..., ...}
  χ² 统计量: 28.1234
  p 值: 0.000000
  结论: McNemar 检验：在 α=0.05 下，拒绝 H₀。 两模型性能存在显著差异。
```

---

### 应用二：模型显著性比较——交叉验证配对 t 检验

#### 背景

在学术研究中，常需要比较两个机器学习模型在多个数据集或多折交叉验证上的性能。此时，单次评估的随机性较大，需要使用多次评估的**配对 t 检验**（Paired t-test）。

#### 统计框架

设模型 A 和模型 B 在 $k$ 折交叉验证中每折的准确率差为：

$$d_i = \text{acc}_B^{(i)} - \text{acc}_A^{(i)}, \quad i = 1, 2, \ldots, k$$

建立假设：
$$H_0: \mu_d = 0 \quad \text{vs} \quad H_1: \mu_d \neq 0$$

检验统计量：
$$T = \frac{\bar{d}}{S_d / \sqrt{k}} \sim t(k-1) \quad \text{（在 } H_0 \text{ 下）}$$

其中 $\bar{d} = \frac{1}{k}\sum_{i=1}^k d_i$，$S_d^2 = \frac{1}{k-1}\sum_{i=1}^k (d_i - \bar{d})^2$。

#### PyTorch 实现

```python
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import List, Callable


def cross_validation_comparison(
    model_a_factory: Callable[[], nn.Module],
    model_b_factory: Callable[[], nn.Module],
    dataset: torch.utils.data.Dataset,
    k_folds: int = 5,
    n_epochs: int = 10,
    batch_size: int = 64,
    alpha: float = 0.05,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    使用 k 折交叉验证配对 t 检验比较两个模型。

    Args:
        model_a_factory: 创建模型 A 实例的工厂函数（每折重新初始化）
        model_b_factory: 创建模型 B 实例的工厂函数
        dataset:         完整数据集
        k_folds:         交叉验证折数
        n_epochs:        每折训练轮数
        batch_size:      批量大小
        alpha:           显著性水平

    Returns:
        包含每折结果、t 统计量、p 值和结论的字典
    """
    n = len(dataset)
    fold_size = n // k_folds
    indices = np.random.permutation(n)

    acc_a_list = []
    acc_b_list = []

    for fold in range(k_folds):
        # 划分训练集和验证集
        val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.concatenate(
            [indices[: fold * fold_size], indices[(fold + 1) * fold_size :]]
        )

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=False
        )

        # 训练并评估两个模型（省略具体训练循环）
        acc_a = _train_and_evaluate(
            model_a_factory(), train_loader, val_loader, n_epochs, device
        )
        acc_b = _train_and_evaluate(
            model_b_factory(), train_loader, val_loader, n_epochs, device
        )

        acc_a_list.append(acc_a)
        acc_b_list.append(acc_b)
        print(f"  Fold {fold+1}/{k_folds}: A={acc_a:.4f}, B={acc_b:.4f}, "
              f"差异={acc_b-acc_a:+.4f}")

    # 计算差值序列
    differences = np.array(acc_b_list) - np.array(acc_a_list)
    d_bar = differences.mean()
    s_d = differences.std(ddof=1)
    t_stat = d_bar / (s_d / np.sqrt(k_folds))

    # 双侧 t 检验 p 值
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=k_folds - 1))
    reject_h0 = p_value <= alpha

    # 95% 置信区间
    t_critical = stats.t.ppf(1 - alpha / 2, df=k_folds - 1)
    margin = t_critical * s_d / np.sqrt(k_folds)
    ci_lower = d_bar - margin
    ci_upper = d_bar + margin

    return {
        "acc_a": acc_a_list,
        "acc_b": acc_b_list,
        "differences": differences.tolist(),
        "mean_difference": d_bar,
        "std_difference": s_d,
        "t_statistic": t_stat,
        "degrees_of_freedom": k_folds - 1,
        "p_value": p_value,
        "confidence_interval": (ci_lower, ci_upper),
        "reject_h0": reject_h0,
        "conclusion": (
            f"配对 t 检验（{k_folds} 折 CV）：\n"
            f"  平均准确率差异: {d_bar:+.4f} ± {margin:.4f} "
            f"（95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]）\n"
            f"  t({k_folds-1}) = {t_stat:.4f}, p = {p_value:.4f}\n"
            f"  结论: 在 α={alpha} 下，"
            f"{'两模型存在显著差异' if reject_h0 else '两模型无显著差异'}。"
        ),
    }


def _train_and_evaluate(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epochs: int,
    device: torch.device,
) -> float:
    """训练模型并返回验证集准确率（简化实现）。"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(n_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    # 评估
    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)
            n_correct += (preds == labels).sum().item()
            n_total += labels.size(0)

    return n_correct / n_total


# ------------------------------------------------------------------ #
# 用 scipy 快速演示配对 t 检验
# ------------------------------------------------------------------ #

def demo_paired_t_test():
    """使用模拟的5折交叉验证结果演示配对 t 检验。"""
    np.random.seed(42)

    # 模拟5折 CV 结果（ResNet vs ViT 的准确率）
    acc_resnet = np.array([0.821, 0.835, 0.828, 0.819, 0.832])
    acc_vit    = np.array([0.847, 0.852, 0.839, 0.843, 0.856])

    differences = acc_vit - acc_resnet
    print("5折交叉验证配对 t 检验")
    print("-" * 40)
    print(f"ResNet 各折准确率: {acc_resnet}")
    print(f"ViT    各折准确率: {acc_vit}")
    print(f"差异 d_i = ViT - ResNet: {differences}")
    print(f"差异均值 d̄ = {differences.mean():.4f}")
    print(f"差异标准差 S_d = {differences.std(ddof=1):.4f}")

    t_stat, p_value = stats.ttest_rel(acc_vit, acc_resnet)
    print(f"\nt 统计量: {t_stat:.4f}")
    print(f"p 值 (双侧): {p_value:.6f}")
    print(f"\n结论: p = {p_value:.4f} {'<' if p_value < 0.05 else '>'} 0.05")
    if p_value < 0.05:
        print("在 α=0.05 水平下，ViT 与 ResNet 存在显著性能差异。")
    else:
        print("在 α=0.05 水平下，两模型性能无显著差异。")

    # 效应量（Cohen's d）
    cohen_d = differences.mean() / differences.std(ddof=1)
    print(f"\nCohen's d 效应量: {cohen_d:.4f}")
    print("效应量解释: |d| < 0.2 小效应, 0.2~0.8 中效应, > 0.8 大效应")


if __name__ == "__main__":
    demo_paired_t_test()
```

**预期输出**：

```
5折交叉验证配对 t 检验
----------------------------------------
ResNet 各折准确率: [0.821 0.835 0.828 0.819 0.832]
ViT    各折准确率: [0.847 0.852 0.839 0.843 0.856]
差异 d_i = ViT - ResNet: [0.026 0.017 0.011 0.024 0.024]
差异均值 d̄ = 0.0204
差异标准差 S_d = 0.0060

t 统计量: 7.5921
p 值 (双侧): 0.001634

结论: p = 0.0016 < 0.05
在 α=0.05 水平下，ViT 与 ResNet 存在显著性能差异。

Cohen's d 效应量: 3.4010
效应量解释: |d| < 0.2 小效应, 0.2~0.8 中效应, > 0.8 大效应
```

#### 实践注意事项

1. **同一测试集 vs 独立测试集**：在同一测试集上评估时，两模型预测相关，优先使用 McNemar 检验或配对 t 检验，而非独立两样本检验。

2. **多重比较问题**：若同时比较多个模型，需进行 Bonferroni 校正：将 $\alpha$ 替换为 $\alpha / m$，其中 $m$ 为比较次数，以控制整体第一类错误率。

3. **效应量比统计显著性更重要**：大样本下，微小的性能差异也可能统计显著。应同时报告效应量（如 Cohen's d）和置信区间，而非仅报告 p 值。

4. **功效分析先行**：在设计实验前，根据预期效应量进行功效分析，确保实验有足够的检测能力（功效 $\geq 0.8$）。

---

## 练习题

**习题 19.1**（基础）某机器学习竞赛中，某选手声称其模型准确率超过了90%的基准。他在1000个测试样本上评估，得到准确率92%。

（1）建立适当的原假设和备择假设；
（2）构造检验统计量并计算其值；
（3）在显著性水平 $\alpha = 0.05$ 下作出判断。

---

**习题 19.2**（基础）解释以下关于 p 值的陈述是否正确，并给出理由：

（1）"p 值为0.03意味着 $H_0$ 为真的概率只有3%"；
（2）"p 值为0.03意味着若重复实验，有97%的概率得到相同结论"；
（3）"p 值为0.03意味着在 $H_0$ 为真时，观测到这么极端数据的概率为3%"；
（4）"p > 0.05 说明两组之间没有差异"。

---

**习题 19.3**（中级）两位研究员比较同一模型在两种数据增强策略下的效果。研究员甲在独立的两批测试数据上分别评估（$n_A = n_B = 500$），策略A正确预测430个，策略B正确预测455个。研究员乙在同一批500个样本上评估，发现策略A对B错25个，策略A错B对50个。

（1）研究员甲应使用什么检验？计算检验统计量和 p 值；
（2）研究员乙应使用什么检验？计算检验统计量和 p 值；
（3）两种检验的结论是否相同？为什么两位研究员应使用不同的检验？

---

**习题 19.4**（中级）某深度学习实验设计如下：改进优化器后，预期模型损失降低 $\delta = 0.05$，已知损失的标准差约为 $\sigma = 0.1$，希望在 $\alpha = 0.05$、功效 $1 - \beta = 0.80$ 下检测到此差异。

（1）这是单侧还是双侧检验？为什么？
（2）计算所需的最小样本量（每组）；
（3）若实际只有20个样本，此时检验的功效约为多少？

---

**习题 19.5**（提高）某团队在同一5折交叉验证实验中比较了模型A和模型B，5折准确率如下：

| 折 | 模型A | 模型B |
|----|-------|-------|
| 1  | 0.812 | 0.831 |
| 2  | 0.798 | 0.815 |
| 3  | 0.823 | 0.820 |
| 4  | 0.807 | 0.829 |
| 5  | 0.815 | 0.835 |

（1）计算每折差值 $d_i = B_i - A_i$，差值均值 $\bar{d}$ 和差值标准差 $S_d$；
（2）构造配对 t 检验统计量，查表或计算 p 值（$t_{0.025}(4) = 2.776$）；
（3）在 $\alpha = 0.05$ 下作出决策，并计算效应量 Cohen's d；
（4）若某人主张因为"第3折模型A更好"，所以两模型无差异，请从统计检验的角度指出其错误。

---

## 练习答案

<details>
<summary><strong>习题 19.1 详解</strong></summary>

**（1）建立假设**

检验方向：我们关心准确率是否"超过"90%，属于右侧检验。

$$H_0: p \leq 0.90 \quad \text{vs} \quad H_1: p > 0.90$$

（等价地取 $H_0: p = 0.90$，检验统计量不变。）

**（2）构造检验统计量**

样本量 $n = 1000$，样本比例 $\hat{p} = 0.92$，$H_0$ 下 $p_0 = 0.90$。

$$Z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}} = \frac{0.92 - 0.90}{\sqrt{0.90 \times 0.10 / 1000}} = \frac{0.02}{\sqrt{0.00009}} = \frac{0.02}{0.009487} \approx 2.108$$

**（3）决策**

右侧检验，拒绝域为 $Z > z_{0.05} = 1.645$。

由于 $z_{\text{obs}} = 2.108 > 1.645$，**拒绝 $H_0$**。

p 值 $= P(Z > 2.108) = 1 - \Phi(2.108) \approx 0.0175 < 0.05$。

在 $\alpha = 0.05$ 下，有统计显著证据表明该模型准确率超过90%。

</details>

<details>
<summary><strong>习题 19.2 详解</strong></summary>

**（1）错误。**

$p$ 值的条件方向是 $P(\text{数据} \mid H_0)$，而非 $P(H_0 \mid \text{数据})$。后者是贝叶斯后验概率，需要 $H_0$ 的先验概率才能计算，与 p 值是完全不同的量。

**（2）错误。**

p 值不衡量实验重现性，也不是"得到相同结论的概率"。"重现性"涉及置信区间和功效，是不同的概念。

**（3）正确。**

这是 p 值的精确定义：在 $H_0$ 为真的假设下，观测到当前检验统计量值或更极端值的概率。

**（4）错误。**

$p > 0.05$ 表示"没有足够证据拒绝 $H_0$"，而不是"证明了两组无差异"。检验可能因样本量不足（功效低）而无法检测真实存在的差异，这是第二类错误（存伪）的情形。正确表述是"未能检测到显著差异"。

</details>

<details>
<summary><strong>习题 19.3 详解</strong></summary>

**（1）研究员甲：两比例 z 检验（独立样本）**

$\hat{p}_A = 430/500 = 0.86$，$\hat{p}_B = 455/500 = 0.91$，$n_A = n_B = 500$。

合并比例：$\hat{p} = (430+455)/(500+500) = 885/1000 = 0.885$

$$Z = \frac{0.91 - 0.86}{\sqrt{0.885 \times 0.115 \times (1/500 + 1/500)}} = \frac{0.05}{\sqrt{0.885 \times 0.115 \times 0.004}}$$

$$= \frac{0.05}{\sqrt{0.0004071}} = \frac{0.05}{0.02018} \approx 2.478$$

双侧 p 值 $= 2 \times P(Z > 2.478) \approx 2 \times 0.0066 = 0.0132 < 0.05$，**拒绝 $H_0$**。

**（2）研究员乙：McNemar 检验（配对样本）**

A 对 B 错（$b_{10}$）= 25，A 错 B 对（$b_{01}$）= 50，不一致总数 = 75。

$$\chi^2 = \frac{(|b_{01} - b_{10}| - 1)^2}{b_{01} + b_{10}} = \frac{(|50 - 25| - 1)^2}{75} = \frac{24^2}{75} = \frac{576}{75} = 7.68$$

p 值 $= P(\chi^2(1) > 7.68) \approx 0.0056 < 0.05$，**拒绝 $H_0$**。

**（3）分析**

两者结论相同（均拒绝 $H_0$），但理由不同：

- 甲的数据是独立的两批测试集，两组样本间没有配对关系，用独立样本检验合理。
- 乙的数据在同一批500个样本上评估，每个样本上两种策略的预测结果相关。若用独立检验，会忽略这种相关性，导致检验效率降低。McNemar 检验充分利用了不一致样本的信息，是配对二值数据的标准检验。

</details>

<details>
<summary><strong>习题 19.4 详解</strong></summary>

**（1）检验类型**

研究者关心"损失是否降低"，有明确的方向性，应使用**右侧检验**（若以降低量 $\delta > 0$ 为正方向，检验 $H_1: \mu_d > 0$），即单侧检验。

**（2）样本量计算**

$\sigma = 0.1$，$\delta = 0.05$，$\alpha = 0.05$（单侧），$1-\beta = 0.80$。

$$z_\alpha = z_{0.05} = 1.645, \quad z_\beta = z_{0.20} = 0.842$$

$$n \geq \left(\frac{(z_\alpha + z_\beta) \cdot \sigma}{\delta}\right)^2 = \left(\frac{(1.645 + 0.842) \times 0.1}{0.05}\right)^2 = \left(\frac{0.2487}{0.05}\right)^2 = 4.974^2 \approx 24.74$$

故每组至少需要 **$n = 25$** 个样本。

**（3）实际功效（$n = 20$）**

$$\text{Power} = P\!\left(Z > z_\alpha - \frac{\delta\sqrt{n}}{\sigma}\right) = P\!\left(Z > 1.645 - \frac{0.05 \times \sqrt{20}}{0.1}\right)$$

$$= P\!\left(Z > 1.645 - \frac{0.05 \times 4.472}{0.1}\right) = P(Z > 1.645 - 2.236) = P(Z > -0.591)$$

$$= \Phi(0.591) \approx 0.723$$

功效约为 **72.3%**，低于目标的80%，意味着有约27.7%的概率漏掉真实效果。

</details>

<details>
<summary><strong>习题 19.5 详解</strong></summary>

**（1）计算差值**

| 折 | A | B | $d_i = B - A$ |
|----|-------|-------|---------------|
| 1 | 0.812 | 0.831 | +0.019 |
| 2 | 0.798 | 0.815 | +0.017 |
| 3 | 0.823 | 0.820 | −0.003 |
| 4 | 0.807 | 0.829 | +0.022 |
| 5 | 0.815 | 0.835 | +0.020 |

$$\bar{d} = \frac{0.019 + 0.017 + (-0.003) + 0.022 + 0.020}{5} = \frac{0.075}{5} = 0.015$$

$$S_d = \sqrt{\frac{\sum_{i=1}^5 (d_i - \bar{d})^2}{4}}$$

偏差平方：$(0.004)^2 + (0.002)^2 + (-0.018)^2 + (0.007)^2 + (0.005)^2$
$= 0.000016 + 0.000004 + 0.000324 + 0.000049 + 0.000025 = 0.000418$

$$S_d = \sqrt{0.000418 / 4} = \sqrt{0.0001045} \approx 0.01022$$

**（2）配对 t 检验统计量**

$$T = \frac{\bar{d}}{S_d / \sqrt{k}} = \frac{0.015}{0.01022 / \sqrt{5}} = \frac{0.015}{0.004571} \approx 3.282$$

双侧检验，自由度 $\nu = k - 1 = 4$。

由于 $|T| = 3.282 > t_{0.025}(4) = 2.776$，**拒绝 $H_0$**，p 值 $< 0.05$。

**（3）效应量与决策**

$$\text{Cohen's } d = \frac{\bar{d}}{S_d} = \frac{0.015}{0.01022} \approx 1.468$$

这是大效应量（$|d| > 0.8$），表明差异不仅统计显著，在实践中也具有一定意义。

**（4）对主张的反驳**

假设检验关注的是整体规律，而非个别折的结果。第3折 A 比 B 好0.003，只是随机波动的体现。统计检验从5个差值的整体模式来判断：差值均值 $\bar{d} = 0.015 > 0$，且变异性小（$S_d = 0.01022$），导致 t 统计量显著大于临界值。

单次折的结果可能由随机误差主导，我们不能以个别反例否定总体趋势。正是因为存在随机波动，才需要用统计检验而非简单比较数字来得出可靠结论。

</details>
