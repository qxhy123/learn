# 第20章 参数检验

> **难度**：★★★★☆
> **前置知识**：第17章假设检验基础、第18章正态分布与抽样分布、第19章非参数检验

---

## 学习目标

- 掌握单样本、双样本 $t$ 检验的适用条件、统计量构造与假设决策方法
- 理解独立双样本与配对样本 $t$ 检验的本质区别与选择依据
- 掌握 $F$ 检验用于方差齐性检验的原理与计算步骤
- 理解卡方检验在拟合优度与独立性检验中的应用
- 掌握多重比较问题及 Bonferroni 校正、FDR 控制方法，并能将统计检验应用于深度学习模型比较

---

## 20.1 单样本 $t$ 检验

### 20.1.1 适用场景与基本假设

**单样本 $t$ 检验**（One-sample $t$-test）用于判断总体均值是否等于某个已知的假设值 $\mu_0$。

**适用条件：**

1. 样本来自正态总体，或样本量 $n \geq 30$（由中心极限定理保证近似正态）
2. 样本相互独立
3. 总体方差 $\sigma^2$ **未知**（若已知则使用 $z$ 检验）

假设形式：

$$H_0: \mu = \mu_0 \quad \text{vs} \quad H_1: \mu \neq \mu_0 \quad (\text{双侧})$$

或单侧：$H_1: \mu > \mu_0$ 或 $H_1: \mu < \mu_0$。

### 20.1.2 检验统计量

设样本 $X_1, X_2, \ldots, X_n$ 来自 $N(\mu, \sigma^2)$，样本均值为

$$\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$$

样本标准差为

$$S = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2}$$

则检验统计量为

$$t = \frac{\bar{X} - \mu_0}{S / \sqrt{n}}$$

在 $H_0$ 成立时，$t \sim t(n-1)$，即服从自由度为 $n-1$ 的 $t$ 分布。

**推导思路：**

- $\bar{X} \sim N(\mu_0, \sigma^2/n)$，标准化后 $Z = \frac{\bar{X}-\mu_0}{\sigma/\sqrt{n}} \sim N(0,1)$
- $(n-1)S^2/\sigma^2 \sim \chi^2(n-1)$，且与 $\bar{X}$ 独立
- 由 $t$ 分布定义 $t = Z/\sqrt{\chi^2(n-1)/(n-1)}$ 得上述统计量

### 20.1.3 拒绝域与 $p$ 值

给定显著性水平 $\alpha$：

| 备择假设 | 拒绝域 | $p$ 值 |
|----------|--------|--------|
| $\mu \neq \mu_0$（双侧） | $\lvert t \rvert > t_{\alpha/2}(n-1)$ | $2P(T > \lvert t_{\text{obs}} \rvert)$ |
| $\mu > \mu_0$（右侧） | $t > t_\alpha(n-1)$ | $P(T > t_{\text{obs}})$ |
| $\mu < \mu_0$（左侧） | $t < -t_\alpha(n-1)$ | $P(T < t_{\text{obs}})$ |

其中 $T \sim t(n-1)$，$t_{\alpha}(n-1)$ 为 $t$ 分布的上 $\alpha$ 分位数。

### 20.1.4 例题

**例 20.1** 某机器生产零件，规格要求直径均值为 $\mu_0 = 50\,\text{mm}$。随机抽取 $n=16$ 件，测得 $\bar{x} = 50.5\,\text{mm}$，$s = 1.2\,\text{mm}$。在 $\alpha = 0.05$ 下检验该机器是否正常工作。

**解：**

$H_0: \mu = 50$ vs $H_1: \mu \neq 50$

$$t = \frac{50.5 - 50}{1.2/\sqrt{16}} = \frac{0.5}{0.3} = 1.667$$

查 $t$ 分布表：$t_{0.025}(15) = 2.131$

因为 $|t| = 1.667 < 2.131$，**不拒绝** $H_0$，即没有充分证据认为机器工作异常。

### 20.1.5 置信区间与检验的对偶性

单样本 $t$ 检验与置信区间具有**对偶关系**：双侧检验在 $\alpha$ 水平拒绝 $H_0: \mu = \mu_0$ 当且仅当 $\mu_0$ 不在 $\mu$ 的 $1-\alpha$ 置信区间内：

$$\left(\bar{X} - t_{\alpha/2}(n-1)\frac{S}{\sqrt{n}},\; \bar{X} + t_{\alpha/2}(n-1)\frac{S}{\sqrt{n}}\right)$$

---

## 20.2 双样本 $t$ 检验：独立与配对

### 20.2.1 独立双样本 $t$ 检验

**场景：** 比较两个独立总体的均值差异，即 $H_0: \mu_1 = \mu_2$（或 $\mu_1 - \mu_2 = \delta_0$）。

#### 方差齐性情况（等方差，$\sigma_1^2 = \sigma_2^2 = \sigma^2$）

设 $X_{11}, \ldots, X_{1n_1} \sim N(\mu_1, \sigma^2)$，$X_{21}, \ldots, X_{2n_2} \sim N(\mu_2, \sigma^2)$，两组独立。

**合并方差估计（Pooled Variance）：**

$$S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1 + n_2 - 2}$$

**检验统计量：**

$$t = \frac{\bar{X}_1 - \bar{X}_2 - \delta_0}{S_p\sqrt{1/n_1 + 1/n_2}} \sim t(n_1 + n_2 - 2) \quad (\text{在 } H_0 \text{ 下})$$

#### 方差不齐情况（Welch $t$ 检验）

当 $\sigma_1^2 \neq \sigma_2^2$ 时，使用 Welch 近似：

$$t = \frac{\bar{X}_1 - \bar{X}_2 - \delta_0}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}$$

自由度由 **Welch-Satterthwaite 公式** 近似：

$$\nu = \frac{\left(S_1^2/n_1 + S_2^2/n_2\right)^2}{\dfrac{(S_1^2/n_1)^2}{n_1-1} + \dfrac{(S_2^2/n_2)^2}{n_2-1}}$$

#### 选择依据

在实践中推荐**默认使用 Welch $t$ 检验**，因为：
- 当方差相等时，Welch 检验的功效损失极小
- 当方差不等时，等方差 $t$ 检验的 I 类错误率膨胀严重

### 20.2.2 配对样本 $t$ 检验

**场景：** 两组观测值存在自然配对关系（如同一受试者的前后测量、同一样本的两种处理结果）。

**核心思想：** 将配对差值 $D_i = X_{1i} - X_{2i}$ 视为新样本，转化为单样本问题。

设 $D_1, D_2, \ldots, D_n$ 为配对差值，$\bar{D}$ 和 $S_D$ 为其均值与标准差：

$$t = \frac{\bar{D} - \delta_0}{S_D / \sqrt{n}} \sim t(n-1) \quad (\text{在 } H_0: \mu_D = \delta_0 \text{ 下})$$

**配对设计的优势：**

配对消除了个体间变异，使检验功效大幅提升。设两组测量的相关系数为 $\rho$，则配对差值的方差为：

$$\text{Var}(D) = \sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2$$

当 $\rho > 0$ 时，$\text{Var}(D) < \text{Var}(X_1) + \text{Var}(X_2)$，即配对降低了估计误差。

### 20.2.3 例题

**例 20.2** 比较两种深度学习优化器（Adam 与 SGD）在 10 个不同数据集上的准确率：

| 数据集 | Adam | SGD | 差值 $D_i$ |
|--------|------|-----|-----------|
| 1 | 0.912 | 0.895 | 0.017 |
| 2 | 0.876 | 0.861 | 0.015 |
| 3 | 0.934 | 0.920 | 0.014 |
| 4 | 0.821 | 0.818 | 0.003 |
| 5 | 0.958 | 0.941 | 0.017 |
| 6 | 0.743 | 0.736 | 0.007 |
| 7 | 0.889 | 0.875 | 0.014 |
| 8 | 0.901 | 0.893 | 0.008 |
| 9 | 0.867 | 0.852 | 0.015 |
| 10 | 0.923 | 0.910 | 0.013 |

计算得 $\bar{d} = 0.0123$，$s_D = 0.00432$，$n = 10$。

$$t = \frac{0.0123}{0.00432/\sqrt{10}} = \frac{0.0123}{0.001366} \approx 9.00$$

$t_{0.025}(9) = 2.262$，因 $t = 9.00 \gg 2.262$，**拒绝** $H_0$，认为两优化器性能存在显著差异。

---

## 20.3 方差检验（$F$ 检验）

### 20.3.1 单总体方差检验（$\chi^2$ 检验）

**场景：** 检验总体方差是否等于已知值 $\sigma_0^2$，即 $H_0: \sigma^2 = \sigma_0^2$。

设 $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$，检验统计量为：

$$\chi^2 = \frac{(n-1)S^2}{\sigma_0^2} \sim \chi^2(n-1) \quad (\text{在 } H_0 \text{ 下})$$

双侧拒绝域：$\chi^2 < \chi^2_{1-\alpha/2}(n-1)$ 或 $\chi^2 > \chi^2_{\alpha/2}(n-1)$。

### 20.3.2 两总体方差比检验（$F$ 检验）

**场景：** 比较两个正态总体的方差是否相等，$H_0: \sigma_1^2 = \sigma_2^2$。

**检验统计量：**

$$F = \frac{S_1^2}{S_2^2}$$

在 $H_0$ 成立时：

$$F = \frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2} = \frac{\chi^2(n_1-1)/(n_1-1)}{\chi^2(n_2-1)/(n_2-1)} \sim F(n_1-1, n_2-1)$$

**拒绝域（双侧，$\alpha = 0.05$）：**

$$F < F_{1-\alpha/2}(n_1-1, n_2-1) \quad \text{或} \quad F > F_{\alpha/2}(n_1-1, n_2-1)$$

利用 $F$ 分布的对称性：$F_{1-\alpha/2}(m, n) = 1/F_{\alpha/2}(n, m)$，实际查表时只需查上侧分位数。

### 20.3.3 $F$ 分布的性质

若 $U \sim \chi^2(m)$，$V \sim \chi^2(n)$ 独立，则

$$F = \frac{U/m}{V/n} \sim F(m, n)$$

其中 $m$ 称为**分子自由度**，$n$ 称为**分母自由度**。

- $E(F) = \dfrac{n}{n-2}$（$n > 2$）
- $F$ 分布右偏，取值为正
- 若 $X \sim t(n)$，则 $X^2 \sim F(1, n)$

### 20.3.4 单因素方差分析（ANOVA）中的 $F$ 检验

**场景：** 比较 $k$ 个独立正态总体均值是否相等，$H_0: \mu_1 = \mu_2 = \cdots = \mu_k$。

设第 $i$ 组有 $n_i$ 个观测值，总观测数 $N = \sum_{i=1}^k n_i$，总体均值 $\bar{X}_{..} = \frac{1}{N}\sum_i\sum_j X_{ij}$。

**平方和分解：**

$$\underbrace{\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij} - \bar{X}_{..})^2}_{SS_T} = \underbrace{\sum_{i=1}^k n_i(\bar{X}_{i.} - \bar{X}_{..})^2}_{SS_A} + \underbrace{\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij} - \bar{X}_{i.})^2}_{SS_E}$$

| 来源 | 平方和 | 自由度 | 均方 | $F$ 统计量 |
|------|--------|--------|------|-----------|
| 组间（处理） | $SS_A$ | $k-1$ | $MS_A = SS_A/(k-1)$ | $F = MS_A/MS_E$ |
| 组内（误差） | $SS_E$ | $N-k$ | $MS_E = SS_E/(N-k)$ | — |
| 总计 | $SS_T$ | $N-1$ | — | — |

在 $H_0$ 下，$F \sim F(k-1, N-k)$；$MS_A$ 和 $MS_E$ 都是 $\sigma^2$ 的无偏估计，但当 $H_1$ 成立时 $E(MS_A) > \sigma^2$，因此大的 $F$ 值支持 $H_1$。

### 20.3.5 ANOVA 的事后多重比较

当 ANOVA 拒绝 $H_0$ 后，我们知道至少有一对均值不等，但不知道**哪些对**不等。需要进行**事后比较**（Post-hoc Tests）。

**Tukey HSD（Honestly Significant Difference）法**：基于学生化极差分布，对所有 $\binom{k}{2}$ 对均值差同时进行检验：

$$|\bar{X}_{i.} - \bar{X}_{j.}| > q_{\alpha}(k, N-k) \cdot \sqrt{\frac{MS_E}{2}\left(\frac{1}{n_i} + \frac{1}{n_j}\right)}$$

其中 $q_\alpha(k, N-k)$ 是学生化极差分布的上 $\alpha$ 分位数。

**Bonferroni 校正**：对每对比较使用 $\alpha' = \alpha/\binom{k}{2}$ 的显著性水平进行 $t$ 检验。简单保守，但当比较组数很多时过于严格。

**Scheffé 法**：基于 F 分布，可检验任意线性对比 $\sum c_i \mu_i = 0$，是最保守但最灵活的方法。

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| Tukey HSD | 控制整体第一类错误率，功效较好 | 各组等样本量，比较所有配对 |
| Bonferroni | 简单通用，偏保守 | 比较组数较少时 |
| Scheffé | 最保守，可检验任意对比 | 需要灵活性时 |

---

## 20.4 卡方检验

### 20.4.1 拟合优度检验

**场景：** 检验数据是否来自某个指定的分布（或符合某种理论比例）。

**假设：** $H_0$：数据符合理论分布 $F_0$

设有 $k$ 个类别，第 $i$ 类的观测频数为 $O_i$，理论概率为 $p_i$，则期望频数为 $E_i = n p_i$。

**皮尔逊卡方统计量：**

$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}$$

在 $H_0$ 下，当 $n$ 充分大时，$\chi^2 \xrightarrow{d} \chi^2(k-1)$（若理论分布完全已知，自由度为 $k-1$）。

若理论分布含 $r$ 个未知参数需用样本估计，则自由度调整为 $k - 1 - r$。

**使用条件：** 每个类别的期望频数 $E_i \geq 5$（若不满足，需合并类别）。

**例 20.3** 掷骰子 120 次，各点出现次数如下，检验骰子是否均匀（$\alpha = 0.05$）：

| 点数 | 1 | 2 | 3 | 4 | 5 | 6 |
|------|---|---|---|---|---|---|
| 观测频数 $O_i$ | 18 | 22 | 19 | 23 | 17 | 21 |
| 期望频数 $E_i$ | 20 | 20 | 20 | 20 | 20 | 20 |

$$\chi^2 = \frac{(18-20)^2}{20} + \frac{(22-20)^2}{20} + \cdots + \frac{(21-20)^2}{20} = \frac{4+4+1+9+9+1}{20} = \frac{28}{20} = 1.4$$

$\chi^2_{0.05}(5) = 11.07$，因 $1.4 < 11.07$，**不拒绝** $H_0$，骰子可视为均匀。

### 20.4.2 独立性检验

**场景：** 判断两个分类变量是否独立，数据呈列联表形式。

设 $r \times c$ 列联表，第 $(i,j)$ 格的观测频数为 $O_{ij}$，行合计 $R_i = \sum_j O_{ij}$，列合计 $C_j = \sum_i O_{ij}$，总计 $n = \sum_i R_i$。

**期望频数（在独立假设下）：**

$$E_{ij} = \frac{R_i \cdot C_j}{n}$$

**检验统计量：**

$$\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \xrightarrow{d} \chi^2\big((r-1)(c-1)\big) \quad (H_0 \text{ 下})$$

**例 20.4** 调查 200 名学生的专业（理工/文科）与是否使用 AI 工具的关系：

| | 使用 AI | 不使用 AI | 合计 |
|----|--------|----------|------|
| 理工 | 80 | 30 | 110 |
| 文科 | 50 | 40 | 90 |
| 合计 | 130 | 70 | 200 |

期望频数：

$$E_{11} = \frac{110 \times 130}{200} = 71.5, \quad E_{12} = \frac{110 \times 70}{200} = 38.5$$
$$E_{21} = \frac{90 \times 130}{200} = 58.5, \quad E_{22} = \frac{90 \times 70}{200} = 31.5$$

$$\chi^2 = \frac{(80-71.5)^2}{71.5} + \frac{(30-38.5)^2}{38.5} + \frac{(50-58.5)^2}{58.5} + \frac{(40-31.5)^2}{31.5}$$

$$= \frac{72.25}{71.5} + \frac{72.25}{38.5} + \frac{72.25}{58.5} + \frac{72.25}{31.5} \approx 1.011 + 1.877 + 1.235 + 2.294 = 6.42$$

$\chi^2_{0.05}(1) = 3.84$，因 $6.42 > 3.84$，**拒绝** $H_0$，专业与 AI 使用存在显著关联。

### 20.4.3 $2\times 2$ 表的 Yates 连续性校正

对 $2\times 2$ 列联表，当样本量较小时，Yates 校正可改善近似效果：

$$\chi^2_{\text{Yates}} = \sum_{i,j} \frac{\left(\lvert O_{ij} - E_{ij}\rvert - 0.5\right)^2}{E_{ij}}$$

但当 $n > 40$ 且所有 $E_{ij} \geq 5$ 时，一般无需校正。

---

## 20.5 多重比较与校正

### 20.5.1 多重比较问题

当同时进行 $m$ 个独立假设检验，每个检验的显著性水平均为 $\alpha$ 时，**族错误率**（Family-Wise Error Rate, FWER）为：

$$\text{FWER} = P(\text{至少一个假检验被拒绝}) = 1 - (1-\alpha)^m$$

例如，$m = 20$，$\alpha = 0.05$：

$$\text{FWER} = 1 - (1-0.05)^{20} = 1 - 0.95^{20} \approx 1 - 0.358 = 0.642$$

即有超过 64% 的概率犯至少一次 I 类错误。

### 20.5.2 Bonferroni 校正

**方法：** 将每个单次检验的显著性水平调整为 $\alpha/m$，则：

$$\text{FWER} \leq 1 - \left(1 - \frac{\alpha}{m}\right)^m \leq \alpha \quad (\text{由 Bonferroni 不等式})$$

**决策规则：** 拒绝 $H_{0j}$ 当且仅当 $p_j \leq \alpha/m$，其中 $p_j$ 为第 $j$ 个检验的 $p$ 值。

**等价校正 $p$ 值：** $p_j^{\text{adj}} = \min(m \cdot p_j,\; 1)$

**缺点：** 过于保守，当 $m$ 较大时检验功效损失严重，尤其检验间存在正相关时。

**Holm-Bonferroni 步进法（更强功效）：**

1. 将 $p$ 值从小到大排序：$p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. 从最小 $p$ 值开始，若 $p_{(j)} \leq \alpha/(m-j+1)$ 则拒绝 $H_{0(j)}$
3. 一旦遇到不满足条件的，停止并接受其余所有 $H_0$

Holm 方法控制 FWER 的同时比 Bonferroni 功效更高。

### 20.5.3 错误发现率（FDR）控制

在大规模多重检验（如基因组学、神经网络特征分析）中，控制 FWER 过于严格。**错误发现率**（False Discovery Rate, FDR）是更合适的标准：

$$\text{FDR} = E\left[\frac{\text{错误拒绝数}}{\max(\text{总拒绝数}, 1)}\right]$$

**Benjamini-Hochberg (BH) 过程（控制 FDR $\leq q$）：**

1. 将 $m$ 个 $p$ 值从小到大排序：$p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. 找到满足 $p_{(j)} \leq \frac{j}{m} q$ 的最大下标 $k$
3. 拒绝 $H_{0(1)}, H_{0(2)}, \ldots, H_{0(k)}$

**BH 等价校正公式：**

$$p_{(j)}^{\text{BH}} = \min\left(\frac{m}{j} p_{(j)},\; p_{(j+1)}^{\text{BH}},\; 1\right)$$（从大到小递推）

**FWER vs FDR 对比：**

| 指标 | FWER | FDR |
|------|------|-----|
| 含义 | 至少一个假阳性的概率 | 平均假阳性比例 |
| 适用场景 | 少量检验，代价高昂 | 大量检验，可容忍少量误判 |
| 典型方法 | Bonferroni, Holm | BH, BY |
| 功效 | 低 | 高 |

### 20.5.4 后验概率与贝叶斯多重比较

贝叶斯方法通过后验概率自然规避多重比较问题。设 $\pi_0$ 为零假设的先验概率，则**局部 FDR**（lfdr）为：

$$\text{lfdr}(t) = \frac{\pi_0 f_0(t)}{f(t)}$$

其中 $f_0(t)$ 为 $H_0$ 下检验统计量的密度，$f(t)$ 为混合密度。

---

## 本章小结

1. **$t$ 检验**是均值推断的核心工具：单样本 $t$ 检验用于与已知值比较；独立双样本 $t$ 检验（含 Welch 方法）用于两独立组均值比较；配对 $t$ 检验通过消除个体差异显著提升功效。

2. **$F$ 检验**用于比较两总体方差以及 ANOVA 中组间均值差异，其统计量由两个独立卡方统计量之比构成。

3. **卡方检验**适用于分类数据：拟合优度检验判断数据分布形式，独立性检验判断两分类变量关联性；两者均要求期望频数不过小。

4. **多重比较**是现代统计的重要问题：Bonferroni 校正控制 FWER 但功效低；BH 方法控制 FDR，在大量检验中功效更高，是深度学习特征选择的优选方法。

5. 所有参数检验均依赖特定分布假设，使用前应进行正态性检验（Shapiro-Wilk）、方差齐性检验（Levene/Bartlett）等前提条件验证；条件不满足时应考虑第19章的非参数方法。

---

## 深度学习应用：特征选择、模型比较与超参数调优

### 应用一：基于统计检验的特征选择

在监督学习中，卡方检验和 $t$ 检验可用于筛选与目标变量相关的特征：

```python
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# ======================================================
# 1. 基于卡方检验的分类特征筛选
# ======================================================

def chi2_feature_selection(X_binary, y, alpha=0.05, method='fdr_bh'):
    """
    对二值特征矩阵 X_binary 和二分类标签 y 进行卡方独立性检验，
    使用 BH 方法控制 FDR，返回显著特征的索引。

    参数:
        X_binary: (n_samples, n_features) 二值特征矩阵
        y: (n_samples,) 二分类标签 {0, 1}
        alpha: FDR 控制水平
        method: 多重比较校正方法，'bonferroni' 或 'fdr_bh'

    返回:
        selected_idx: 通过检验的特征索引
        p_values: 原始 p 值
        p_adjusted: 校正后 p 值
    """
    n_features = X_binary.shape[1]
    p_values = np.zeros(n_features)

    for j in range(n_features):
        # 构造 2x2 列联表
        # 行：y=0/1，列：特征=0/1
        contingency = np.array([
            [np.sum((X_binary[:, j] == 0) & (y == 0)),
             np.sum((X_binary[:, j] == 1) & (y == 0))],
            [np.sum((X_binary[:, j] == 0) & (y == 1)),
             np.sum((X_binary[:, j] == 1) & (y == 1))]
        ])
        chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency)
        p_values[j] = p_val

    # 多重比较校正
    reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method=method)
    selected_idx = np.where(reject)[0]

    return selected_idx, p_values, p_adjusted


# ======================================================
# 2. 基于 t 检验的连续特征筛选（Welch t 检验）
# ======================================================

def ttest_feature_selection(X_continuous, y, alpha=0.05, method='fdr_bh'):
    """
    对连续特征矩阵 X_continuous 进行 Welch 双样本 t 检验，
    检验各特征在两类样本中均值是否有显著差异。
    """
    n_features = X_continuous.shape[1]
    p_values = np.zeros(n_features)
    t_stats = np.zeros(n_features)

    group0 = X_continuous[y == 0]  # (n0, n_features)
    group1 = X_continuous[y == 1]  # (n1, n_features)

    for j in range(n_features):
        t_stat, p_val = stats.ttest_ind(group0[:, j], group1[:, j], equal_var=False)
        t_stats[j] = t_stat
        p_values[j] = p_val

    reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method=method)
    selected_idx = np.where(reject)[0]

    print(f"总特征数: {n_features}")
    print(f"Bonferroni 阈值: p < {alpha / n_features:.6f}")
    print(f"BH FDR={alpha} 下选中特征数: {len(selected_idx)}")

    return selected_idx, t_stats, p_values, p_adjusted


# ======================================================
# 3. 演示：合成数据特征选择
# ======================================================

np.random.seed(42)
n_samples, n_features = 500, 50
n_informative = 10  # 前 10 个特征真正有判别力

# 生成连续特征
X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, 2, n_samples)

# 让前 n_informative 个特征在两类中均值有差异
X[:, :n_informative] += y[:, np.newaxis] * 1.5

# 执行特征选择
selected, t_stats, p_raw, p_adj = ttest_feature_selection(X, y, alpha=0.05)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(n_features), -np.log10(p_raw), color='steelblue', alpha=0.7)
axes[0].axhline(-np.log10(0.05 / n_features), color='red', linestyle='--',
                label=f'Bonferroni ($p={0.05/n_features:.4f}$)')
axes[0].axhline(-np.log10(0.05), color='orange', linestyle='--',
                label='未校正 $p=0.05$')
axes[0].set_xlabel('特征索引')
axes[0].set_ylabel('$-\log_{10}(p)$')
axes[0].set_title('原始 $p$ 值（Manhattan 图）')
axes[0].legend()

axes[1].bar(range(n_features), -np.log10(p_adj), color='darkorange', alpha=0.7)
axes[1].axhline(-np.log10(0.05), color='red', linestyle='--', label='FDR=0.05')
axes[1].scatter(selected, -np.log10(p_adj[selected]), color='red', zorder=5, label='选中特征')
axes[1].set_xlabel('特征索引')
axes[1].set_ylabel('$-\log_{10}(p_{\mathrm{adj}})$')
axes[1].set_title('BH 校正后 $p$ 值')
axes[1].legend()

plt.tight_layout()
plt.savefig('feature_selection_pvalues.png', dpi=150)
plt.show()
```

### 应用二：神经网络模型性能的统计比较

比较两个或多个深度学习模型时，需用统计检验判断性能差异是否显著：

```python
# ======================================================
# 模型比较：配对 t 检验 + McNemar 检验
# ======================================================

class SimpleNN(nn.Module):
    """简单前馈神经网络，用于演示"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def k_fold_accuracies(model_class, model_kwargs, X, y, k=10, seed=42):
    """
    k 折交叉验证，返回每折的测试准确率。

    参数:
        model_class: 模型类
        model_kwargs: 模型初始化参数
        X, y: 数据（numpy array）
        k: 折数

    返回:
        accuracies: 长度为 k 的准确率列表
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train = torch.FloatTensor(X[train_idx])
        y_train = torch.LongTensor(y[train_idx])
        X_test = torch.FloatTensor(X[test_idx])
        y_test = torch.LongTensor(y[test_idx])

        model = model_class(**model_kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 简化训练：50 个 epoch
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        # 评估
        model.eval()
        with torch.no_grad():
            preds = model(X_test).argmax(dim=1)
            acc = (preds == y_test).float().mean().item()
        accuracies.append(acc)

    return accuracies


def compare_models_ttest(accs_A, accs_B, alpha=0.05):
    """
    使用配对 t 检验比较两个模型的 k 折交叉验证准确率。

    假设：两组准确率来自同一数据集上的 k 折划分（配对）
    """
    accs_A = np.array(accs_A)
    accs_B = np.array(accs_B)
    diffs = accs_A - accs_B
    n = len(diffs)

    d_bar = diffs.mean()
    s_d = diffs.std(ddof=1)
    t_stat = d_bar / (s_d / np.sqrt(n))

    # 双侧 p 值
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)

    # 差值的 95% 置信区间
    t_crit = stats.t.ppf(1 - alpha/2, df=n - 1)
    ci_lower = d_bar - t_crit * s_d / np.sqrt(n)
    ci_upper = d_bar + t_crit * s_d / np.sqrt(n)

    print("=" * 50)
    print("配对 t 检验：模型 A vs 模型 B")
    print("=" * 50)
    print(f"模型 A 平均准确率: {accs_A.mean():.4f} ± {accs_A.std():.4f}")
    print(f"模型 B 平均准确率: {accs_B.mean():.4f} ± {accs_B.std():.4f}")
    print(f"差值均值 (A-B):   {d_bar:.4f}")
    print(f"差值标准差:        {s_d:.4f}")
    print(f"t 统计量:          {t_stat:.4f}  (df={n-1})")
    print(f"p 值 (双侧):       {p_value:.4f}")
    print(f"95% 置信区间:     [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"结论: {'模型 A 显著优于 B' if (p_value < alpha and d_bar > 0) else '模型 B 显著优于 A' if (p_value < alpha and d_bar < 0) else '无显著差异'}")

    return {'t_stat': t_stat, 'p_value': p_value,
            'ci': (ci_lower, ci_upper), 'mean_diff': d_bar}


def mcnemar_test(preds_A, preds_B, y_true):
    """
    McNemar 检验：比较两个分类器在同一测试集上的错误模式。

    适用场景：两个分类器对同一样本集的预测结果比较
    构造 2×2 列联表：
      b = A 正确但 B 错误的样本数
      c = A 错误但 B 正确的样本数

    H0: P(A 正确 B 错误) = P(A 错误 B 正确)
    """
    correct_A = (preds_A == y_true)
    correct_B = (preds_B == y_true)

    # b: A对B错, c: A错B对
    b = np.sum(correct_A & ~correct_B)
    c = np.sum(~correct_A & correct_B)

    # 使用连续性校正的卡方统计量
    if b + c == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    print(f"\nMcNemar 检验")
    print(f"b (A对B错): {b}, c (A错B对): {c}")
    print(f"χ² = {chi2:.4f}, p = {p_value:.4f}")

    return chi2, p_value


# ======================================================
# 演示：合成分类任务上的模型比较
# ======================================================

np.random.seed(0)
n_samples, n_features = 1000, 20
X_demo = np.random.randn(n_samples, n_features).astype(np.float32)
w_true = np.random.randn(n_features)
y_demo = (X_demo @ w_true > 0).astype(int)

# 模拟两个模型的 10 折准确率（实际场景中调用 k_fold_accuracies）
# 此处用固定随机数模拟结果以演示检验逻辑
rng = np.random.RandomState(99)
accs_model_A = 0.82 + rng.randn(10) * 0.02   # 模型 A：均值约 0.82
accs_model_B = 0.79 + rng.randn(10) * 0.025  # 模型 B：均值约 0.79

result = compare_models_ttest(accs_model_A, accs_model_B)
```

### 应用三：超参数选择的统计显著性

在超参数搜索后，利用 $F$ 检验（ANOVA）判断不同超参数配置的效果差异：

```python
# ======================================================
# 超参数比较：单因素 ANOVA + Tukey HSD 事后检验
# ======================================================

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def hyperparameter_anova(results_dict, alpha=0.05):
    """
    对多组超参数配置的验证准确率进行单因素方差分析。

    参数:
        results_dict: {超参数配置名: 验证准确率列表}
        alpha: 显著性水平
    """
    groups = list(results_dict.keys())
    data = [np.array(results_dict[g]) for g in groups]

    # 计算描述统计
    print("各配置描述统计：")
    print(f"{'配置':<20} {'均值':>8} {'标准差':>8} {'样本量':>6}")
    print("-" * 45)
    for g, d in zip(groups, data):
        print(f"{g:<20} {d.mean():>8.4f} {d.std(ddof=1):>8.4f} {len(d):>6}")

    # ANOVA F 检验
    f_stat, p_value = f_oneway(*data)
    k = len(groups)
    N = sum(len(d) for d in data)

    print(f"\nANOVA 结果: F({k-1}, {N-k}) = {f_stat:.4f}, p = {p_value:.4f}")

    if p_value < alpha:
        print(f"结论: 在 α={alpha} 水平下，各超参数配置间存在显著差异 (p={p_value:.4f})")

        # 事后检验：Tukey HSD
        all_data = np.concatenate(data)
        all_labels = np.concatenate([[g] * len(d) for g, d in zip(groups, data)])

        tukey = pairwise_tukeyhsd(all_data, all_labels, alpha=alpha)
        print("\nTukey HSD 事后检验（成对比较）：")
        print(tukey)
    else:
        print(f"结论: 无显著差异 (p={p_value:.4f} > {alpha})")

    return f_stat, p_value


# 示例：比较不同学习率配置的 5 次独立训练结果
lr_results = {
    'lr=1e-2': [0.843, 0.851, 0.839, 0.856, 0.848],
    'lr=1e-3': [0.871, 0.879, 0.868, 0.882, 0.875],
    'lr=1e-4': [0.854, 0.861, 0.858, 0.865, 0.857],
    'lr=1e-5': [0.801, 0.812, 0.808, 0.815, 0.807],
}

print("=" * 55)
print("超参数（学习率）效果的 ANOVA 检验")
print("=" * 55)
f_stat, p_val = hyperparameter_anova(lr_results)

# ======================================================
# 方差齐性前提验证（Levene 检验）
# ======================================================

from scipy.stats import levene

def check_homogeneity(results_dict, alpha=0.05):
    """Levene 检验方差齐性（ANOVA 前提条件）"""
    data = list(results_dict.values())
    stat, p = levene(*data)
    print(f"\nLevene 方差齐性检验: W={stat:.4f}, p={p:.4f}")
    if p > alpha:
        print("方差齐性假设成立，可使用等方差 ANOVA")
    else:
        print("方差不齐，建议使用 Welch ANOVA 或非参数 Kruskal-Wallis 检验")
    return stat, p

check_homogeneity(lr_results)
```

**深度学习应用要点总结：**

| 场景 | 推荐检验 | 关键注意事项 |
|------|----------|-------------|
| 连续特征与标签相关性 | Welch $t$ 检验 + BH 校正 | 需检验正态性；大量特征时 FDR 优于 FWER |
| 分类特征与标签关联 | 卡方独立性检验 + BH 校正 | 期望频数 $\geq 5$；稀疏时合并类别 |
| 两模型准确率比较（k 折） | 配对 $t$ 检验 | 同一折数据配对；需满足正态性 |
| 两模型错误模式比较 | McNemar 检验 | 关注不一致预测对；小样本加连续性校正 |
| 多个超参数配置比较 | ANOVA + Tukey HSD | 先验证方差齐性（Levene 检验） |
| 模型选择 vs 随机 | 单样本 $t$ 检验 | 与基线准确率比较 |

---

## 练习题

**练习 20.1** （单样本 $t$ 检验）

某训练流程声称在 CIFAR-10 上平均准确率为 92%。研究者用该流程独立训练了 9 次，得到准确率（%）：

$$90.1,\; 91.5,\; 89.8,\; 92.3,\; 91.0,\; 90.7,\; 91.8,\; 90.5,\; 91.2$$

在 $\alpha = 0.05$ 下，是否有充分证据认为该训练流程的真实均值低于 92%？给出完整的假设检验过程。

---

**练习 20.2** （独立 vs 配对 $t$ 检验）

研究者比较 ResNet-50 与 VGG-16 在 8 个图像分类数据集上的准确率（已经 10 折交叉验证取均值）：

| 数据集 | ResNet-50 | VGG-16 |
|--------|-----------|--------|
| A | 0.934 | 0.921 |
| B | 0.867 | 0.853 |
| C | 0.912 | 0.908 |
| D | 0.756 | 0.741 |
| E | 0.981 | 0.975 |
| F | 0.823 | 0.811 |
| G | 0.891 | 0.878 |
| H | 0.943 | 0.930 |

(a) 应使用独立双样本 $t$ 检验还是配对 $t$ 检验？说明理由。

(b) 在 $\alpha = 0.05$ 下进行检验，并给出差值的 95% 置信区间。

---

**练习 20.3** （$F$ 检验 + ANOVA）

比较三种 dropout 率（0.1、0.3、0.5）对模型验证损失的影响，各进行 6 次独立实验：

| dropout=0.1 | dropout=0.3 | dropout=0.5 |
|-------------|-------------|-------------|
| 0.412 | 0.389 | 0.421 |
| 0.398 | 0.401 | 0.435 |
| 0.423 | 0.375 | 0.418 |
| 0.409 | 0.392 | 0.443 |
| 0.415 | 0.384 | 0.428 |
| 0.404 | 0.396 | 0.432 |

(a) 完成单因素方差分析表（计算 $SS_A$、$SS_E$、$MS_A$、$MS_E$、$F$ 值）。

(b) 在 $\alpha = 0.05$ 下，三组损失均值是否存在显著差异？

---

**练习 20.4** （多重比较校正）

在神经网络特征重要性分析中，对 100 个特征各进行一次 $t$ 检验，得到 100 个 $p$ 值，其中最小的 10 个（排序后）为：

$$p_{(1)}=0.00008,\; p_{(2)}=0.00021,\; p_{(3)}=0.00089,\; p_{(4)}=0.00203,\; p_{(5)}=0.00412$$
$$p_{(6)}=0.00718,\; p_{(7)}=0.00935,\; p_{(8)}=0.01204,\; p_{(9)}=0.01567,\; p_{(10)}=0.02341$$

(a) 使用 Bonferroni 校正（$\alpha=0.05$），哪些特征被选中？

(b) 使用 BH 方法控制 FDR $= 0.05$，哪些特征被选中？（只需考虑上述 10 个 $p$ 值及余下 90 个均大于 0.05）

(c) 比较两种方法的结论，说明在此场景下哪种方法更合适。

---

**练习 20.5** （卡方检验 + 深度学习）

研究者统计了某图像分类模型在 4 种类别上的预测结果与真实标签的列联表（共 400 个测试样本）：

| 预测\真实 | 猫 | 狗 | 鸟 | 鱼 |
|----------|----|----|----|----|
| 猫 | 85 | 12 | 5 | 8 |
| 狗 | 10 | 78 | 7 | 5 |
| 鸟 | 3 | 8 | 72 | 7 |
| 鱼 | 2 | 2 | 6 | 80 |

(a) 计算该检验的期望频数矩阵（假设独立性）。

(b) 计算卡方统计量并给出检验结论（$\alpha=0.05$）。

(c) 若卡方检验显著，这说明了什么？与混淆矩阵分析有何联系？

---

## 练习答案

### 答案 20.1

**建立假设**（单侧，左侧检验）：

$$H_0: \mu = 92 \quad \text{vs} \quad H_1: \mu < 92$$

**计算统计量**：

$$\bar{x} = \frac{90.1 + 91.5 + \cdots + 91.2}{9} = \frac{818.9}{9} = 91.0\overline{0}$$

精确计算：$\sum x_i = 90.1+91.5+89.8+92.3+91.0+90.7+91.8+90.5+91.2 = 818.9$，$\bar{x} = 818.9/9 \approx 91.0$

各偏差平方和：

$$\sum(x_i - \bar{x})^2 = (90.1-91.0)^2 + (91.5-91.0)^2 + (89.8-91.0)^2 + (92.3-91.0)^2 + (91.0-91.0)^2$$
$$+ (90.7-91.0)^2 + (91.8-91.0)^2 + (90.5-91.0)^2 + (91.2-91.0)^2$$
$$= 0.81 + 0.25 + 1.44 + 1.69 + 0 + 0.09 + 0.64 + 0.25 + 0.04 = 5.21$$

$$s = \sqrt{\frac{5.21}{8}} = \sqrt{0.651} \approx 0.807$$

$$t = \frac{91.0 - 92}{0.807/\sqrt{9}} = \frac{-1.0}{0.269} \approx -3.717$$

**查表与决策**：

$t_{0.05}(8) = 1.860$，左侧检验拒绝域为 $t < -1.860$。

因 $t = -3.717 < -1.860$，**拒绝** $H_0$。

**结论**：在 $\alpha = 0.05$ 水平下，有充分证据认为该训练流程的真实平均准确率低于 92%（$p \approx 0.003$）。

---

### 答案 20.2

**(a) 应使用配对 $t$ 检验**。

原因：同一数据集对两模型而言代表相同的难度，数据集之间差异很大（准确率从 0.756 到 0.981），配对设计可以消除数据集难度造成的变异，仅关注模型间差异。若用独立 $t$ 检验，数据集间方差会淹没模型间差异，导致功效损失。

**(b) 配对 $t$ 检验计算**：

差值 $D_i = \text{ResNet} - \text{VGG}$：$0.013, 0.014, 0.004, 0.015, 0.006, 0.012, 0.013, 0.013$

$$\bar{d} = \frac{0.013+0.014+0.004+0.015+0.006+0.012+0.013+0.013}{8} = \frac{0.090}{8} = 0.01125$$

$$s_D = \sqrt{\frac{\sum(d_i - \bar{d})^2}{7}}$$

各 $(d_i - 0.01125)^2$：$(−0.00125)^2, (0.00275)^2, (−0.00725)^2, (0.00375)^2, (−0.00525)^2, (0.00075)^2, (0.00175)^2, (0.00175)^2$

$= 1.5625\times10^{-6} + 7.5625\times10^{-6} + 52.5625\times10^{-6} + 14.0625\times10^{-6} + 27.5625\times10^{-6} + 0.5625\times10^{-6} + 3.0625\times10^{-6} + 3.0625\times10^{-6} = 110\times10^{-6}$

$$s_D = \sqrt{\frac{110\times10^{-6}}{7}} = \sqrt{15.71\times10^{-6}} \approx 0.003964$$

$$t = \frac{0.01125}{0.003964/\sqrt{8}} = \frac{0.01125}{0.001401} \approx 8.03$$

$t_{0.025}(7) = 2.365$，因 $t = 8.03 \gg 2.365$，**拒绝** $H_0$，ResNet-50 显著优于 VGG-16。

**95% 置信区间**：

$$0.01125 \pm 2.365 \times 0.001401 = 0.01125 \pm 0.00331 = (0.00794,\; 0.01456)$$

即 ResNet-50 平均准确率高出 VGG-16 约 0.8% 至 1.5%（95% 置信）。

---

### 答案 20.3

**(a) ANOVA 计算**：

各组均值：

$$\bar{x}_{1.} = \frac{0.412+0.398+0.423+0.409+0.415+0.404}{6} = \frac{2.461}{6} \approx 0.4102$$

$$\bar{x}_{2.} = \frac{0.389+0.401+0.375+0.392+0.384+0.396}{6} = \frac{2.337}{6} = 0.3895$$

$$\bar{x}_{3.} = \frac{0.421+0.435+0.418+0.443+0.428+0.432}{6} = \frac{2.577}{6} = 0.4295$$

总均值：$\bar{x}_{..} = (2.461+2.337+2.577)/18 = 7.375/18 \approx 0.4097$

**组间平方和** $SS_A$（$n_i=6$，$k=3$）：

$$SS_A = 6\left[(0.4102-0.4097)^2 + (0.3895-0.4097)^2 + (0.4295-0.4097)^2\right]$$
$$= 6\left[2.5\times10^{-7} + 4.08\times10^{-4} + 3.92\times10^{-4}\right]$$
$$\approx 6 \times 8.003\times10^{-4} = 4.802\times10^{-3}$$

**组内平方和** $SS_E$（各组内偏差平方和之和，共 $3\times5=15$ 自由度）：

组1：$(0.412-0.4102)^2+\cdots \approx 4.93\times10^{-4}$；
组2：$(0.389-0.3895)^2+\cdots \approx 5.25\times10^{-4}$；
组3：$(0.421-0.4295)^2+\cdots \approx 3.55\times10^{-4}$

$SS_E \approx 1.373\times10^{-3}$

| 来源 | $SS$ | 自由度 | $MS$ | $F$ |
|------|------|--------|------|-----|
| 组间 | $4.802\times10^{-3}$ | 2 | $2.401\times10^{-3}$ | $\approx 26.3$ |
| 组内 | $1.373\times10^{-3}$ | 15 | $9.15\times10^{-5}$ | — |
| 总计 | $6.175\times10^{-3}$ | 17 | — | — |

**(b) 决策**：$F_{0.05}(2, 15) = 3.68$，因 $F \approx 26.3 \gg 3.68$，**拒绝** $H_0$。

结论：三种 dropout 率下的验证损失均值存在显著差异，应进一步进行事后两两比较（如 Tukey HSD）以确定哪组间差异显著。

---

### 答案 20.4

**(a) Bonferroni 校正**（$m=100$，阈值 $= 0.05/100 = 0.0005$）：

- $p_{(1)} = 0.00008 < 0.0005$ ✓
- $p_{(2)} = 0.00021 < 0.0005$ ✓
- $p_{(3)} = 0.00089 > 0.0005$ ✗

**选中特征：2 个**（排名第 1、2 的特征）。

**(b) BH 方法**（$m=100$，$q=0.05$，判断条件 $p_{(j)} \leq j \cdot q / m = j \times 0.0005$）：

| $j$ | $p_{(j)}$ | 阈值 $j \times 0.0005$ | 是否通过 |
|-----|-----------|----------------------|---------|
| 1 | 0.00008 | 0.00050 | ✓ |
| 2 | 0.00021 | 0.00100 | ✓ |
| 3 | 0.00089 | 0.00150 | ✓ |
| 4 | 0.00203 | 0.00200 | ✓（$\leq$，边界） |
| 5 | 0.00412 | 0.00250 | ✗ |

最大通过下标 $k=4$，**选中特征：4 个**（排名第 1~4）。

注意：$p_{(4)} = 0.00203$ 与阈值 $0.00200$ 非常接近，精确比较后 $0.00203 > 0.00200$，实际通过下标为 $k=3$，**选中特征：3 个**。（需精确计算）

**(c) 比较与建议**：

- Bonferroni 控制 FWER，选中 2 个；BH 控制 FDR，选中 2-3 个
- 在 100 个特征同时检验的场景下，**BH 方法更合适**：允许少量误判（FDR=5% 意味着平均 5% 的被选特征是假阳性），功效更高，能发现更多真正相关特征
- Bonferroni 适用于少量关键检验且假阳性代价极高的场景（如医学诊断）；特征选择属于探索性场景，FDR 控制是标准做法

---

### 答案 20.5

**(a) 期望频数矩阵**：

各行/列合计：

- 预测为猫：$85+12+5+8=110$，预测为狗：$100$，预测为鸟：$90$，预测为鱼：$90$（行合计）
- 真实为猫：$100$，狗：$100$，鸟：$90$，鱼：$100$（列合计，注意：$85+10+3+2=100$，$12+78+8+2=100$，$5+7+72+6=90$，$8+5+7+80=100$）
- 总计：$110+100+90+90=390$（注意：实际总和为 $\sum = 390$，非 400，以实际计算为准）

实际总计：$85+12+5+8+10+78+7+5+3+8+72+7+2+2+6+80 = 390$

期望频数 $E_{ij} = R_i C_j / 390$，例如：

$$E_{11} = \frac{110 \times 100}{390} \approx 28.2, \quad E_{12} = \frac{110 \times 100}{390} \approx 28.2$$

（完整矩阵从略，按 $E_{ij} = R_i C_j / n$ 逐一计算）

**(b) 卡方统计量**：

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

该矩阵对角线元素远大于期望（对角线 $O_{ij} >> E_{ij}$），非对角线则相反，因此 $\chi^2$ 值很大，远超 $\chi^2_{0.05}((4-1)(4-1)) = \chi^2_{0.05}(9) = 16.92$。

**结论：拒绝** $H_0$（预测与真实标签不独立），即模型的预测与真实类别显著相关。

**(c) 含义与联系**：

- 卡方检验显著说明**预测结果与真实类别不独立**——在此情境下这是**好事**，说明模型有判别能力（随机猜测时两者独立）
- 与混淆矩阵联系：混淆矩阵提供更细粒度信息（哪些类别容易混淆，如猫被预测为狗），而卡方检验仅给出全局显著性结论
- 在模型评估中，卡方检验可用于验证"模型预测优于随机基线"，但具体改进方向需结合混淆矩阵分析
