# F6 详解：Part 7 假设检验（Ch.19-21，共 29 题）

> 覆盖范围：假设检验框架（Ch.19）、参数检验（Ch.20）、非参数检验（Ch.21）。
> 题型：z/t/F/χ² 检验、功效计算、Neyman-Pearson 引理、Wilcoxon 秩检验、ANOVA、多重检验校正、置换检验、贝叶斯因子。

---

## C 基础题详解（10 题）

### C.7.1（Ch.19，检验框架——两类错误）

**题目**：$H_0$：产品合格率 $p \geq 0.95$；$H_1$：$p < 0.95$。识别两类错误并说明显著性水平控制哪类。

**思路**：两类错误由假设方向和决策方向共同决定；$\alpha$ 是在 $H_0$ 为真时犯错的概率上限。

**解**：

步骤 1（定义）：

- **第一类错误（Type I / 弃真）**：$H_0$ 为真但被拒绝。  
  具体描述：产品实际合格率 $p \geq 0.95$（合格），却被检验判为"不合格"，误拒合格批次。

- **第二类错误（Type II / 取伪）**：$H_0$ 为假但未被拒绝。  
  具体描述：产品实际合格率 $p < 0.95$（不合格），却未能拒绝 $H_0$，放行了不合格批次。

步骤 2（控制）：显著性水平 $\alpha = P(\text{拒绝 }H_0 \mid H_0\text{ 为真})$，即 $\alpha$ **控制第一类错误概率的上限**。

**答案**：$\boxed{\alpha \text{ 控制第一类错误（弃真）概率的上限}}$

> ⚠️ 两类错误存在权衡：降低 $\alpha$ 会导致第二类错误概率 $\beta$ 上升；仅靠调节临界值无法同时减小两类错误——增大样本量 $n$ 才能同时降低二者。

---

### C.7.2（Ch.19，p 值的含义）

**题目**：$p$ 值为 $0.03$，$\alpha = 0.05$。判断是否拒绝 $H_0$，并辨析 $p$ 值的正确含义。

**思路**：$p$ 值是数据与 $H_0$ 的相容程度度量，不是 $H_0$ 为真的概率。

**解**：

步骤 1（决策）：因 $p = 0.03 < \alpha = 0.05$，**拒绝 $H_0$**。

步骤 2（含义辨析）：

- 选项 (A)"$H_0$ 为真的概率是 $0.03$"——**错误**。  
  频率学派框架中 $H_0$ 要么真要么假，不是随机事件，无法谈其概率。

- 选项 (B)"在 $H_0$ 为真的条件下，观测到此结果或更极端结果的概率是 $0.03$"——**正确**。

**答案**：$\boxed{(B)\text{ 正确；}p = 0.03 < 0.05\text{，拒绝 }H_0}$

> ⚠️ p 值不是"$H_0$ 为真的概率"，这是统计推断中最常见的误解之一。p 值是一个**条件概率**，条件是 $H_0$ 为真。

---

### C.7.3（Ch.19，单样本 z 检验）

**题目**：$X \sim N(\mu, \sigma^2 = 25)$，$n = 25$，$\bar{x} = 52$。检验 $H_0: \mu = 50$ vs $H_1: \mu \neq 50$，$\alpha = 0.05$。

**思路**：$\sigma$ 已知用 z 检验；双侧检验拒绝域为 $|z| > z_{\alpha/2}$。

**解**：

步骤 1（假设）：$H_0: \mu = 50$ vs $H_1: \mu \neq 50$（双侧）。

步骤 2（统计量）：

$$Z = \frac{\bar{X} - \mu_0}{\sigma/\sqrt{n}} \sim N(0,1) \quad \text{（在 }H_0\text{ 下）}$$

步骤 3（计算）：

$$z_{\text{obs}} = \frac{52 - 50}{5/\sqrt{25}} = \frac{2}{5/5} = \frac{2}{1} = 2.0$$

步骤 4（拒绝域与决策）：拒绝域为 $|Z| > z_{0.025} = 1.96$。

因 $|z_{\text{obs}}| = 2.0 > 1.96$，**拒绝 $H_0$**。

**答案**：$\boxed{z_{\text{obs}} = 2.0 > 1.96\text{，拒绝 }H_0\text{，认为 }\mu \neq 50}$

---

### C.7.4（Ch.19，检验功效的定义）

**题目**：$H_0: \mu = \mu_0$，$H_1: \mu = \mu_1 > \mu_0$。定义检验功效并分析样本量影响。

**思路**：功效是"$H_1$ 为真时能正确拒绝 $H_0$"的概率，即避免第二类错误的能力。

**解**：

步骤 1（功效定义）：

$$\text{Power} = 1 - \beta = P(\text{拒绝 }H_0 \mid H_1\text{ 为真}) = P(\text{拒绝 }H_0 \mid \mu = \mu_1)$$

其中 $\beta = P(\text{不拒绝 }H_0 \mid H_1\text{ 为真})$ 为第二类错误概率。

步骤 2（样本量影响）：

增大样本量 $n$ 时，统计量方差减小（$\mathrm{Var}(\bar X) = \sigma^2/n$），检验的区分能力增强，功效**增大**。

直觉：$n$ 越大，即使真实均值只是略微偏离 $\mu_0$，也能以更高概率检测到差异。

**答案**：$\boxed{\text{Power} = P(\text{拒绝 }H_0 \mid \mu = \mu_1)\text{；增大 }n \Rightarrow \text{功效增大}}$

---

### C.7.5（Ch.20，单样本 t 检验）

**题目**：$\sigma^2$ 未知，$n = 16$，$\bar{x} = 10.8$，$s = 2.0$。检验 $H_0: \mu = 10$ vs $H_1: \mu \neq 10$，$\alpha = 0.05$，$t_{0.025}(15) = 2.131$。

**思路**：$\sigma$ 未知且样本量小时用 $t$ 检验，自由度 $df = n - 1 = 15$。

**解**：

步骤 1（假设）：$H_0: \mu = 10$ vs $H_1: \mu \neq 10$（双侧）。

步骤 2（统计量）：

$$T = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} \sim t(n-1) \quad \text{（在 }H_0\text{ 下）}$$

步骤 3（计算）：

$$t_{\text{obs}} = \frac{10.8 - 10}{2.0/\sqrt{16}} = \frac{0.8}{2.0/4} = \frac{0.8}{0.5} = 1.6$$

步骤 4（拒绝域与决策）：拒绝域为 $|T| > t_{0.025}(15) = 2.131$。

因 $|t_{\text{obs}}| = 1.6 < 2.131$，**不拒绝 $H_0$**。

**答案**：$\boxed{t_{\text{obs}} = 1.6 < 2.131\text{，不拒绝 }H_0\text{，无充分证据认为 }\mu \neq 10}$

---

### C.7.6（Ch.20，独立双样本 t 检验条件）

**题目**：列出独立双样本 $t$ 检验的三个基本假设条件。

**思路**：参数检验依赖数据生成机制的假设，违反这些假设会导致结论不可靠。

**解**：

独立双样本 $t$ 检验的三个基本假设：

1. **独立性**：两组样本相互独立，且各组内观测值也相互独立。

2. **正态性**：两总体均服从正态分布（小样本时尤为重要；样本量足够大时由中心极限定理保障近似成立）。

3. **方差齐性（等方差）**：两总体方差相等，即 $\sigma_1^2 = \sigma_2^2$（用于计算合并方差 $S_p^2$）。若方差不等，应改用 Welch's $t$ 检验。

**答案**：$\boxed{\text{独立性、正态性、方差齐性（}σ_1^2 = σ_2^2\text{）}}$

---

### C.7.7（Ch.20，方差齐性 F 检验）

**题目**：$s_1^2 = 8$（$n_1 = 11$），$s_2^2 = 4$（$n_2 = 9$）。检验 $H_0: \sigma_1^2 = \sigma_2^2$，写出统计量及分布，计算 $F$ 值。

**思路**：两个独立正态总体方差比的枢轴量服从 $F$ 分布。

**解**：

步骤 1（统计量及分布）：

$$F = \frac{S_1^2}{S_2^2} \sim F(n_1 - 1, \, n_2 - 1) = F(10, 8) \quad \text{（在 }H_0\text{ 下）}$$

步骤 2（计算）：

$$F_{\text{obs}} = \frac{s_1^2}{s_2^2} = \frac{8}{4} = 2.0$$

（决策需查 $F_{0.025}(10, 8)$ 和 $F_{0.975}(10, 8)$ 两端临界值，此题仅要求写出统计量和计算数值。）

**答案**：$\boxed{F = S_1^2/S_2^2 \sim F(10,8)\text{，}F_{\text{obs}} = 2.0}$

---

### C.7.8（Ch.20，卡方拟合优度检验）

**题目**：掷骰子 60 次，各点次数 $\{8, 12, 9, 11, 10, 10\}$。检验 $H_0$：各点概率均为 $1/6$。

**思路**：拟合优度检验比较观测频数与期望频数的偏差；自由度 = 格子数 $-$ 1（无估计参数）。

**解**：

步骤 1（期望频数）：在 $H_0$ 下，每面期望次数为：

$$E_i = 60 \times \frac{1}{6} = 10 \quad (i = 1, 2, \ldots, 6)$$

步骤 2（统计量）：

$$\chi^2 = \sum_{i=1}^{6} \frac{(O_i - E_i)^2}{E_i} = \frac{(8-10)^2}{10} + \frac{(12-10)^2}{10} + \frac{(9-10)^2}{10} + \frac{(11-10)^2}{10} + \frac{(10-10)^2}{10} + \frac{(10-10)^2}{10}$$

步骤 3（分布）：在 $H_0$ 下，当 $n$ 足够大时：

$$\chi^2 \overset{\cdot}{\sim} \chi^2(k - 1) = \chi^2(5)$$

自由度 $df = k - 1 = 6 - 1 = 5$（$k = 6$ 个格子，无待估参数）。

**答案**：$\boxed{E_i = 10\text{；}\chi^2 = \sum(O_i-E_i)^2/E_i \sim \chi^2(5)}$

---

### C.7.9（Ch.21，符号检验）

**题目**：差值 $\{+3, -1, +5, +2, -2, +4, +1, +3\}$，检验 $H_0$：总体中位数差为 0。

**思路**：符号检验只利用差值的正负号，是中位数的非参数检验；在 $H_0$ 下正号和负号等可能。

**解**：

步骤 1（统计正负号）：

- 正号（$+$）：$+3, +5, +2, +4, +1, +3$ → 共 **6 个**
- 负号（$-$）：$-1, -2$ → 共 **2 个**
- 差值为 0：无（均不忽略）
- 有效观测 $n = 8$

步骤 2（零假设下的分布）：

在 $H_0$（中位数差为 0）下，每个差值正负各有 $1/2$ 的概率，正号个数 $B$ 服从：

$$B \sim \mathrm{Binomial}(n = 8, \, p = 0.5)$$

（若 $B$ 或 $8 - B$ 足够小即为拒绝域；具体双侧临界值查二项表。）

**答案**：$\boxed{正 6 个，负 2 个；在 H_0 下 B \sim B(8, 0.5)}$

---

### C.7.10（Ch.21，Kolmogorov-Smirnov 检验）

**题目**：KS 检验中 $F_n(x)$ 是什么？$D_n$ 超过临界值时的结论。

**思路**：KS 检验基于经验分布函数与假设分布函数的最大偏差。

**解**：

步骤 1（$F_n(x)$ 的定义）：

$F_n(x)$ 是**经验分布函数（Empirical Distribution Function, EDF）**：

$$F_n(x) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}[X_i \leq x]$$

即样本中不超过 $x$ 的观测值所占比例。

步骤 2（决策）：

若 $D_n = \sup_x |F_n(x) - F_0(x)|$ 超过在 $\alpha$ 水平下的 KS 临界值，则**拒绝 $H_0$**，认为数据不服从假设分布 $F_0$。

$D_n$ 越大，说明经验分布与假设分布的偏差越大，数据越不像来自 $F_0$。

**答案**：$\boxed{F_n(x) \text{ 为经验分布函数；}D_n > \text{临界值则拒绝 }H_0}$

> ⚠️ KS 检验对分布的中间部分最灵敏，对尾部偏差不够敏感；若对尾部偏差更关注，可用 Anderson-Darling 检验。

---

## D 中等题详解（12 题）

### D.7.1（Ch.19，单均值 z 检验）

**题目**：$\mu_0 = 50$，$\sigma = 8$ 已知，$n = 64$，$\bar{X} = 48.5$。双侧检验，$\alpha = 0.05$；计算 p 值并解释。

**思路**：$\sigma$ 已知用 z 检验；p 值是双侧尾概率之和。

**解**：

步骤 1（假设）：

$$H_0: \mu = 50 \quad \text{vs} \quad H_1: \mu \neq 50$$

步骤 2（统计量及分布）：

$$Z = \frac{\bar{X} - \mu_0}{\sigma/\sqrt{n}} \sim N(0,1) \quad \text{（在 }H_0\text{ 下）}$$

步骤 3（计算统计量）：

$$z_{\text{obs}} = \frac{48.5 - 50}{8/\sqrt{64}} = \frac{-1.5}{8/8} = \frac{-1.5}{1} = -1.5$$

步骤 4（临界值与决策）：

拒绝域：$|Z| > z_{0.025} = 1.96$。

因 $|z_{\text{obs}}| = 1.5 < 1.96$，**不拒绝 $H_0$**。

步骤 5（p 值计算）：

$$p = 2P(Z < -1.5) = 2 \times \Phi(-1.5) \approx 2 \times 0.0668 = 0.1336$$

步骤 6（p 值解释）：

**p 值的含义**：在 $H_0$（$\mu = 50$）为真的条件下，观测到检验统计量偏离 0 的程度达到 $|z| \geq 1.5$（或更极端）的概率约为 $13.4\%$。

因 $p = 0.134 > 0.05$，数据与 $H_0$ 相容，不拒绝 $H_0$。

**答案**：$\boxed{z_{\text{obs}} = -1.5\text{，}p \approx 0.134 > 0.05\text{，不拒绝 }H_0}$

> ⚠️ p 值 $\approx 13.4\%$ 不意味着"$H_0$ 为真的概率是 $13.4\%$"，而是数据与 $H_0$ 相容的程度度量。

---

### D.7.2（Ch.19，单均值 t 检验）

**题目**：$\mu_0 = 75$，$n = 16$，$\bar{X} = 78$，$S = 8$。单侧检验 $H_0: \mu \leq 75$ vs $H_1: \mu > 75$，$\alpha = 0.05$，$t_{0.05}(15) \approx 1.753$；若真实值 $\mu_1 = 80$，求功效。

**思路**：单侧 $t$ 检验；功效需用非中心 $t$ 分布或正态近似计算。

**解**：

步骤 1（假设）：

$$H_0: \mu \leq 75 \quad \text{vs} \quad H_1: \mu > 75 \quad (\text{单侧右尾})$$

步骤 2（统计量）：

$$T = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} \sim t(15) \quad \text{（在 }H_0\text{ 边界 }\mu = 75\text{ 下）}$$

步骤 3（计算）：

$$t_{\text{obs}} = \frac{78 - 75}{8/\sqrt{16}} = \frac{3}{8/4} = \frac{3}{2} = 1.5$$

步骤 4（决策）：

拒绝域：$T > t_{0.05}(15) = 1.753$。

因 $t_{\text{obs}} = 1.5 < 1.753$，**不拒绝 $H_0$**（在 $\alpha = 0.05$ 下无足够证据认为均值 $> 75$）。

步骤 5（功效计算，$\mu_1 = 80$）：

非中心参数：

$$\delta = \frac{\mu_1 - \mu_0}{S/\sqrt{n}} = \frac{80 - 75}{8/4} = \frac{5}{2} = 2.5$$

功效为非中心 $t$ 分布的上尾概率，近似方法：

当 $H_1: \mu = 80$ 时，$T$ 的均值约为 $\delta = 2.5$，利用正态近似：

$$\text{Power} \approx P\!\left(Z > z_{0.05} - \delta\right) = P(Z > 1.645 - 2.5) = P(Z > -0.855) \approx \Phi(0.855) \approx 0.804$$

即在真实均值为 80 时，此检验约有 **80.4%** 的概率正确拒绝 $H_0$。

**答案**：$\boxed{t_{\text{obs}} = 1.5 < 1.753\text{，不拒绝 }H_0\text{；功效} \approx 80\%（\mu_1 = 80）}$

---

### D.7.3（Ch.19，两类错误与检验的功效）

**题目**：$H_0: \mu = 0$ vs $H_1: \mu = 1$，$X \sim N(\mu, 1)$，拒绝域 $X > c$。推导两类错误表达式，分析权衡。

**思路**：简单假设的两类错误均可用标准正态计算；两者随 $c$ 单调方向相反，增大 $n$ 才能同时减小。

**解**：

步骤 1（第一类错误）：

$$\alpha(c) = P(X > c \mid \mu = 0) = P\!\left(Z > c\right) = 1 - \Phi(c)$$

其中 $Z = X \sim N(0,1)$（因 $\sigma = 1$，$n = 1$）。

步骤 2（第二类错误）：

$$\beta(c) = P(X \leq c \mid \mu = 1) = P\!\left(Z \leq c - 1\right) = \Phi(c - 1)$$

其中 $Z = X - 1 \sim N(0,1)$（在 $H_1$ 下）。

步骤 3（权衡分析）：

| 临界值 $c$ | $\alpha(c) = 1-\Phi(c)$ | $\beta(c) = \Phi(c-1)$ | $\alpha + \beta$ |
|:----------:|:------------------------:|:------------------------:|:----------------:|
| $0$ | $0.500$ | $0.159$ | $0.659$ |
| $0.5$ | $0.309$ | $0.309$ | $0.618$ |
| $1$ | $0.159$ | $0.500$ | $0.659$ |

- $c$ 增大：$\alpha$ 减小，$\beta$ 增大（拒绝域变窄）。
- $c$ 减小：$\alpha$ 增大，$\beta$ 减小。
- $\alpha + \beta$ 有最小值但不能同时为零。

步骤 4（增大 $n$ 的效果）：若观测 $n$ 个样本取均值，统计量变为 $\bar{X} \sim N(\mu, 1/n)$，则

$$\alpha(c) = 1 - \Phi(c\sqrt{n}), \quad \beta(c) = \Phi\!\left((c-1)\sqrt{n}\right)$$

选 $c = 0.5$（中点）时，$\alpha = \beta = 1 - \Phi(0.5\sqrt{n}) \to 0$ as $n \to \infty$。

**答案**：$\boxed{\alpha(c) = 1-\Phi(c)\text{，}\beta(c) = \Phi(c-1)\text{；增大 }n \text{ 可同时减小两类错误}}$

---

### D.7.4（Ch.20，方差的 χ² 检验）

**题目**：$\sigma_0^2 = 4$，$n = 25$，$S^2 = 5.8$。单侧检验 $H_0: \sigma^2 = 4$ vs $H_1: \sigma^2 > 4$，$\alpha = 0.05$，$\chi^2_{0.05}(24) \approx 36.42$。

**思路**：正态总体方差检验用 $\chi^2$ 统计量；单侧右尾检验只看右端临界值。

**解**：

步骤 1（假设）：

$$H_0: \sigma^2 = 4 \quad \text{vs} \quad H_1: \sigma^2 > 4 \quad (\text{单侧右尾})$$

步骤 2（统计量及分布）：

$$\chi^2 = \frac{(n-1)S^2}{\sigma_0^2} \sim \chi^2(n-1) = \chi^2(24) \quad \text{（在 }H_0\text{ 下）}$$

步骤 3（计算）：

$$\chi^2_{\text{obs}} = \frac{24 \times 5.8}{4} = \frac{139.2}{4} = 34.8$$

步骤 4（决策）：

拒绝域：$\chi^2 > \chi^2_{0.05}(24) = 36.42$。

因 $\chi^2_{\text{obs}} = 34.8 < 36.42$，**不拒绝 $H_0$**（无足够证据认为方差大于 4）。

步骤 5（双侧拒绝域）：

若改为双侧检验 $H_1: \sigma^2 \neq 4$，拒绝域为：

$$\chi^2 < \chi^2_{0.975}(24) \approx 12.40 \quad \text{或} \quad \chi^2 > \chi^2_{0.025}(24) \approx 39.36$$

（双侧各取 $\alpha/2 = 0.025$。）

**答案**：$\boxed{\chi^2_{\text{obs}} = 34.8 < 36.42\text{，不拒绝 }H_0\text{；双侧域：}(12.40,\, 39.36)\text{ 之外}}$

---

### D.7.5（Ch.20，两均值 t 检验，等方差）

**题目**：$n_1 = 8$，$\bar{X}_1 = 20.5$，$S_1^2 = 6$；$n_2 = 10$，$\bar{X}_2 = 18.0$，$S_2^2 = 7$。等方差，$H_0: \mu_1 = \mu_2$，$\alpha = 0.05$，$t_{0.025}(16) \approx 2.120$。

**思路**：等方差假设下合并两组方差估计；合并方差是加权平均。

**解**：

步骤 1（假设）：

$$H_0: \mu_1 = \mu_2 \quad \text{vs} \quad H_1: \mu_1 \neq \mu_2 \quad (\text{双侧})$$

步骤 2（合并方差）：

$$S_p^2 = \frac{(n_1 - 1)S_1^2 + (n_2 - 1)S_2^2}{n_1 + n_2 - 2} = \frac{7 \times 6 + 9 \times 7}{8 + 10 - 2} = \frac{42 + 63}{16} = \frac{105}{16} = 6.5625$$

步骤 3（统计量）：

$$T = \frac{(\bar{X}_1 - \bar{X}_2) - 0}{S_p\sqrt{1/n_1 + 1/n_2}} \sim t(n_1 + n_2 - 2) = t(16) \quad \text{（在 }H_0\text{ 下）}$$

$$t_{\text{obs}} = \frac{20.5 - 18.0}{\sqrt{6.5625 \times (1/8 + 1/10)}} = \frac{2.5}{\sqrt{6.5625 \times 0.225}} = \frac{2.5}{\sqrt{1.4766}} = \frac{2.5}{1.215} \approx 2.058$$

步骤 4（决策）：

拒绝域：$|T| > t_{0.025}(16) = 2.120$。

因 $|t_{\text{obs}}| = 2.058 < 2.120$，**不拒绝 $H_0$**（边界情况，差异不显著）。

步骤 5（先行 F 检验等方差）：

在做双均值 $t$ 检验前，用 $F = S_1^2/S_2^2 = 6/7 \approx 0.857$ 检验 $H_0': \sigma_1^2 = \sigma_2^2$（$F(7, 9)$ 分布）。若 $F$ 检验不拒绝，则使用等方差 $t$ 检验；若拒绝，改用 Welch's $t$ 检验（自由度用 Satterthwaite 近似）。

**答案**：$\boxed{S_p^2 \approx 6.56\text{，}t_{\text{obs}} \approx 2.06 < 2.12\text{，不拒绝 }H_0}$

---

### D.7.6（Ch.20，配对 t 检验）

**题目**：10 名受试者治疗前后血压数据，计算差值并检验 $H_0: \mu_D = 0$ vs $H_1: \mu_D > 0$，$\alpha = 0.05$，$t_{0.05}(9) \approx 1.833$。

**思路**：配对设计将差值 $D_i = $ 前 $-$ 后视为单样本，消除个体间差异。

**解**：

步骤 1（计算差值）：

| 受试者 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|--------|---|---|---|---|---|---|---|---|---|---|
| 前 | 120 | 135 | 128 | 142 | 118 | 130 | 125 | 138 | 122 | 131 |
| 后 | 115 | 129 | 124 | 137 | 120 | 128 | 122 | 134 | 120 | 127 |
| $D_i$ | 5 | 6 | 4 | 5 | -2 | 2 | 3 | 4 | 2 | 4 |

步骤 2（计算 $\bar{D}$ 和 $S_D$）：

$$\bar{D} = \frac{5 + 6 + 4 + 5 + (-2) + 2 + 3 + 4 + 2 + 4}{10} = \frac{33}{10} = 3.3$$

$$S_D^2 = \frac{\sum_{i=1}^{10}(D_i - \bar{D})^2}{n-1}$$

各偏差：$(1.7)^2, (2.7)^2, (0.7)^2, (1.7)^2, (-5.3)^2, (-1.3)^2, (-0.3)^2, (0.7)^2, (-1.3)^2, (0.7)^2$

$= 2.89 + 7.29 + 0.49 + 2.89 + 28.09 + 1.69 + 0.09 + 0.49 + 1.69 + 0.49 = 46.10$

$$S_D^2 = \frac{46.10}{9} \approx 5.122, \quad S_D \approx 2.263$$

步骤 3（统计量）：

$$T = \frac{\bar{D}}{S_D/\sqrt{n}} = \frac{3.3}{2.263/\sqrt{10}} = \frac{3.3}{0.716} \approx 4.61$$

步骤 4（决策）：

拒绝域：$T > t_{0.05}(9) = 1.833$。

因 $t_{\text{obs}} = 4.61 \gg 1.833$，**拒绝 $H_0$**，有强烈证据认为治疗使血压下降。

步骤 5（配对设计的优势）：

配对设计将个体间差异（基线血压高低不同）从误差中剥离，只分析每人自身的变化量 $D_i$，$S_D$ 反映个体内变异而非个体间变异，因此比独立双样本 $t$ 检验有更高的统计功效（等效样本量更小即可达到同等功效）。

**答案**：$\boxed{\bar{D} = 3.3\text{，}S_D \approx 2.26\text{，}t_{\text{obs}} \approx 4.61 \gg 1.833\text{，拒绝 }H_0}$

---

### D.7.7（Ch.19，Neyman-Pearson 引理）

**题目**：$X \sim N(\mu, 1)$，$H_0: \mu = 0$ vs $H_1: \mu = 1$，单次观测。推导拒绝域，解释最优势检验，讨论 UMP 存在条件。

**思路**：NP 引理给出简单假设对下的最优检验；对指数族，似然比为 $T$ 的单调函数，可推广到 UMP。

**解**：

步骤 1（似然比）：

$$\Lambda(x) = \frac{f(x;\,\mu=1)}{f(x;\,\mu=0)} = \frac{\frac{1}{\sqrt{2\pi}}e^{-(x-1)^2/2}}{\frac{1}{\sqrt{2\pi}}e^{-x^2/2}} = e^{x - 1/2}$$

步骤 2（化简拒绝域）：

NP 引理要求当 $\Lambda(x) > k$ 时拒绝 $H_0$，即：

$$e^{x-1/2} > k \iff x > \ln k + \frac{1}{2} \equiv c$$

拒绝域等价于 $X > c$（右尾检验）。由水平约束 $\alpha = P(X > c \mid \mu = 0) = 1 - \Phi(c)$，得 $c = z_\alpha$（如 $\alpha = 0.05$ 时 $c = 1.645$）。

步骤 3（最优势的含义）：

"水平 $\alpha$ 下的最优势检验（Most Powerful Test）"指：在所有控制第一类错误 $\leq \alpha$ 的检验中，此检验的功效（$P(\text{拒绝 }H_0 \mid H_1\text{ 为真})$）**最大**。任何其他同水平检验的功效不超过 NP 检验的功效。

步骤 4（复合 $H_1$ 与 UMP）：

对复合 $H_1: \mu > 0$，NP 引理不直接适用（因每个不同的 $\mu_1 > 0$ 对应不同的 NP 检验）。但正态分布属于指数族，具有**单调似然比（Monotone Likelihood Ratio, MLR）**性质：$\Lambda(\mu_1, \mu_0; x)$ 关于充分统计量 $\bar{X}$（此处为 $X$）单调递增。

由此可证明存在**一致最优势（UMP）检验**：拒绝域 $X > c$ 对所有 $\mu_1 > 0$ 均最优。一般地，指数族一维参数的单侧假设存在 UMP；双侧假设 $H_1: \mu \neq \mu_0$ 通常不存在 UMP。

**答案**：$\boxed{\Lambda(x) = e^{x-1/2}\text{，拒绝域 }X > z_\alpha\text{；最优势指功效最大；单边复合 }H_1\text{ 存在 UMP（MLR 条件）}}$

---

### D.7.8（Ch.21，Wilcoxon 符号秩检验）

**题目**：$n = 8$ 个差值：$+3, -1, +5, +2, -4, +1, +6, +2$。双侧检验 $H_0$：中位数为 0，$\alpha = 0.05$，临界值 $W = 4$。

**思路**：Wilcoxon 符号秩检验用绝对值的秩携带量级信息，比纯符号检验更有效。

**解**：

步骤 1（计算绝对值并排秩）：

| 差值 $D_i$ | $|D_i|$ | 秩 | 符号秩 |
|:----------:|:-------:|:---:|:------:|
| $+1$ | $1$ | $1$ | $+1$ |
| $-1$ | $1$ | $1$ | $-1$ |
| $+2$ | $2$ | $3$ | $+3$ |
| $+2$ | $2$ | $3$ | $+3$ |
| $+3$ | $3$ | $5$ | $+5$ |
| $-4$ | $4$ | $6$ | $-6$ |
| $+5$ | $5$ | $7$ | $+7$ |
| $+6$ | $6$ | $8$ | $+8$ |

排秩说明：$|D_i|$ 从小到大排列；$|D| = 1$ 有两个（$+1$ 和 $-1$），平均秩 $= (1+2)/2 = 1.5$；$|D| = 2$ 有两个（两个 $+2$），平均秩 $= (3+4)/2 = 3.5$。

修正后：

| 差值 $D_i$ | $|D_i|$ | 平均秩 | 符号秩 |
|:----------:|:-------:|:------:|:------:|
| $+1$ | $1$ | $1.5$ | $+1.5$ |
| $-1$ | $1$ | $1.5$ | $-1.5$ |
| $+2$ | $2$ | $3.5$ | $+3.5$ |
| $+2$ | $2$ | $3.5$ | $+3.5$ |
| $+3$ | $3$ | $5$ | $+5$ |
| $-4$ | $4$ | $6$ | $-6$ |
| $+5$ | $5$ | $7$ | $+7$ |
| $+6$ | $6$ | $8$ | $+8$ |

步骤 2（计算正秩和与负秩和）：

$$T^+ = 1.5 + 3.5 + 3.5 + 5 + 7 + 8 = 28.5$$

$$T^- = 1.5 + 6 = 7.5$$

验证：$T^+ + T^- = 28.5 + 7.5 = 36 = n(n+1)/2 = 8 \times 9/2 = 36$ ✓

步骤 3（决策）：

检验统计量取 $T = \min(T^+, T^-) = 7.5$。

临界值 $W = 4$（$n = 8$，双侧 $\alpha = 0.05$）：当 $T \leq W$ 时拒绝 $H_0$。

因 $T = 7.5 > 4$，**不拒绝 $H_0$**。

步骤 4（与参数 t 检验比较）：

参数 $t$ 检验利用差值的均值和方差，假设差值来自正态分布；Wilcoxon 符号秩检验无正态假设，对异常值更鲁棒。当数据正态时，$t$ 检验功效略高（渐近相对效率 $\approx 0.955$）；数据有重尾时 Wilcoxon 更优。

**答案**：$\boxed{T^+ = 28.5\text{，}T^- = 7.5\text{，}T = 7.5 > 4\text{，不拒绝 }H_0}$

---

### D.7.9（Ch.20，单因素方差分析 ANOVA）

**题目**：三组各 $n_i = 5$：组1：12,14,11,13,10；组2：20,22,19,21,18；组3：15,17,16,14,18。检验三组均值是否相等，$F_{0.05}(2,12) \approx 3.89$。

**思路**：ANOVA 将总变差分解为组间和组内两部分，用 F 比检验均值差异。

**解**：

步骤 1（计算各组均值和总均值）：

$$\bar{X}_1 = \frac{12+14+11+13+10}{5} = \frac{60}{5} = 12$$

$$\bar{X}_2 = \frac{20+22+19+21+18}{5} = \frac{100}{5} = 20$$

$$\bar{X}_3 = \frac{15+17+16+14+18}{5} = \frac{80}{5} = 16$$

$$\bar{X} = \frac{60 + 100 + 80}{15} = \frac{240}{15} = 16$$

步骤 2（计算 SS）：

**组间平方和**（$df_B = k - 1 = 2$）：

$$SS_B = \sum_{i=1}^{3} n_i(\bar{X}_i - \bar{X})^2 = 5(12-16)^2 + 5(20-16)^2 + 5(16-16)^2$$
$$= 5 \times 16 + 5 \times 16 + 0 = 80 + 80 + 0 = 160$$

**组内平方和**（$df_W = N - k = 15 - 3 = 12$）：

- 组1：$(12-12)^2+(14-12)^2+(11-12)^2+(13-12)^2+(10-12)^2 = 0+4+1+1+4=10$
- 组2：$(20-20)^2+(22-20)^2+(19-20)^2+(21-20)^2+(18-20)^2 = 0+4+1+1+4=10$
- 组3：$(15-16)^2+(17-16)^2+(16-16)^2+(14-16)^2+(18-16)^2 = 1+1+0+4+4=10$

$$SS_W = 10 + 10 + 10 = 30$$

步骤 3（均方与 F 统计量）：

$$MS_B = \frac{SS_B}{df_B} = \frac{160}{2} = 80, \quad MS_W = \frac{SS_W}{df_W} = \frac{30}{12} = 2.5$$

$$F_{\text{obs}} = \frac{MS_B}{MS_W} = \frac{80}{2.5} = 32$$

步骤 4（决策）：

拒绝域：$F > F_{0.05}(2, 12) = 3.89$。

因 $F_{\text{obs}} = 32 \gg 3.89$，**强烈拒绝 $H_0$**，三组均值有极显著差异。

**ANOVA 汇总表**：

| 来源 | $SS$ | $df$ | $MS$ | $F$ |
|------|------|------|------|-----|
| 组间 | $160$ | $2$ | $80$ | $32$ |
| 组内 | $30$ | $12$ | $2.5$ | |
| 总计 | $190$ | $14$ | | |

**答案**：$\boxed{F_{\text{obs}} = 32 \gg 3.89\text{，拒绝 }H_0\text{，三组均值差异极显著}}$

> ⚠️ ANOVA 拒绝 $H_0$ 只说明"至少两组均值不同"，不指明哪两组不同——需进一步做事后多重比较（Tukey HSD、Bonferroni 等）。

---

### D.7.10（Ch.21，卡方拟合优度检验）

**题目**：掷骰子 120 次，各面次数：18, 22, 17, 25, 19, 19。检验均匀性，$\alpha = 0.05$，$\chi^2_{0.05}(5) \approx 11.07$。

**思路**：拟合优度检验：期望频数 $E_i = np_i$，统计量自由度 = 格子数 $-$ 1 $-$ 估计参数数。

**解**：

步骤 1（理论频数）：$H_0$ 下每面概率 $1/6$，理论频数：

$$E_i = 120 \times \frac{1}{6} = 20 \quad (i = 1, \ldots, 6)$$

步骤 2（计算 χ² 统计量）：

$$\chi^2 = \sum_{i=1}^{6}\frac{(O_i - E_i)^2}{E_i} = \frac{(18-20)^2}{20} + \frac{(22-20)^2}{20} + \frac{(17-20)^2}{20} + \frac{(25-20)^2}{20} + \frac{(19-20)^2}{20} + \frac{(19-20)^2}{20}$$

$$= \frac{4}{20} + \frac{4}{20} + \frac{9}{20} + \frac{25}{20} + \frac{1}{20} + \frac{1}{20} = \frac{44}{20} = 2.2$$

步骤 3（决策）：

$df = 6 - 1 = 5$，$\chi^2_{0.05}(5) \approx 11.07$。

因 $\chi^2_{\text{obs}} = 2.2 < 11.07$，**不拒绝 $H_0$**，无证据认为骰子不均匀。

$p$ 值 $= P(\chi^2(5) > 2.2) > 0.50$（远不显著）。

步骤 4（期望次数 < 5 时的处理）：

当某个格子的期望频数 $E_i < 5$ 时，χ² 近似的准确性下降。标准处理方式是**合并相邻格子**（合并后 $E_i \geq 5$），同时相应减少自由度。一般要求所有 $E_i \geq 1$ 且不超过 20% 的格子 $E_i < 5$。

**答案**：$\boxed{\chi^2_{\text{obs}} = 2.2 < 11.07\text{，不拒绝 }H_0\text{，骰子无显著不均匀}}$

---

### D.7.11（Ch.21，独立性卡方检验）

**题目**：性别与吸烟的 $2 \times 2$ 列联表（200 人）。计算期望频数、$\chi^2$ 统计量、Phi 系数。

**思路**：独立性检验：期望频数 = 行和 $\times$ 列和 / 总数；$\phi = \sqrt{\chi^2/n}$ 衡量效应大小。

**解**：

步骤 1（观测频数表）：

| | 吸烟 | 不吸烟 | 合计 |
|---|------|--------|------|
| 男 | $O_{11}=60$ | $O_{12}=40$ | $R_1=100$ |
| 女 | $O_{21}=30$ | $O_{22}=70$ | $R_2=100$ |
| 合计 | $C_1=90$ | $C_2=110$ | $n=200$ |

步骤 2（期望频数）：$E_{ij} = R_i C_j / n$：

$$E_{11} = \frac{100 \times 90}{200} = 45, \quad E_{12} = \frac{100 \times 110}{200} = 55$$

$$E_{21} = \frac{100 \times 90}{200} = 45, \quad E_{22} = \frac{100 \times 110}{200} = 55$$

步骤 3（χ² 统计量）：

$$\chi^2 = \frac{(60-45)^2}{45} + \frac{(40-55)^2}{55} + \frac{(30-45)^2}{45} + \frac{(70-55)^2}{55}$$

$$= \frac{225}{45} + \frac{225}{55} + \frac{225}{45} + \frac{225}{55} = 5 + 4.09 + 5 + 4.09 = 18.18$$

步骤 4（决策）：

$df = (2-1)(2-1) = 1$，$\chi^2_{0.05}(1) \approx 3.84$。

因 $\chi^2_{\text{obs}} = 18.18 \gg 3.84$，**强烈拒绝 $H_0$**（性别与吸烟显著相关）。

步骤 5（Phi 系数）：

$$\phi = \sqrt{\frac{\chi^2}{n}} = \sqrt{\frac{18.18}{200}} = \sqrt{0.0909} \approx 0.302$$

Phi 系数类似于相关系数（范围 $[0, 1]$），$|\phi| \approx 0.3$ 表示中等程度的关联。$\phi^2 = \chi^2/n$ 是"解释方差比例"的类似量，$\phi^2 \approx 0.091$，即约 9.1% 的变差可由性别-吸烟关联解释。

**答案**：$\boxed{\chi^2_{\text{obs}} \approx 18.18 \gg 3.84\text{，拒绝独立性 }H_0\text{；}\phi \approx 0.302\text{（中等关联）}}$

---

### D.7.12（Ch.19，多重检验问题）

**题目**：同时检验 $m = 20$ 个独立真零假设，每个 $\alpha = 0.05$。计算 FWER，Bonferroni 校正，并描述 BH 方法。

**思路**：多个检验时误报率随检验数膨胀；Bonferroni 保守但严格控制 FWER；BH 控制 FDR 更宽松。

**解**：

步骤 1（FWER 计算）：

$m = 20$ 个独立检验，每个犯第一类错误的概率为 $\alpha = 0.05$。

$$\text{FWER} = P(\text{至少一个假阳性}) = 1 - P(\text{所有均不误报}) = 1 - (1-\alpha)^m$$

$$= 1 - (0.95)^{20} \approx 1 - 0.3585 = 0.6415$$

即约 **64.2%** 的概率至少出现一次假阳性——远超名义显著性水平 5%。

步骤 2（Bonferroni 校正）：

将每个检验水平调为 $\alpha^* = \alpha/m = 0.05/20 = 0.0025$，则：

$$\text{FWER} \leq m \cdot \alpha^* = 20 \times 0.0025 = 0.05$$

（利用 Boole 不等式：$P(\bigcup_i A_i) \leq \sum_i P(A_i)$，无需独立性假设。）

Bonferroni 校正将 FWER 控制在 $0.05$ 以内。代价是每个检验的功效下降。

步骤 3（BH 方法思路）：

设 $m = 20$ 个假设中有 5 个为假（$m_0 = 15$ 个真零假设）。

Benjamini-Hochberg（BH）程序：

1. 将 $m$ 个 $p$ 值从小到大排序：$p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(20)}$。
2. 找最大的 $k$ 使得 $p_{(k)} \leq k \cdot \alpha / m$（即 $p_{(k)} \leq k \times 0.05/20$）。
3. 拒绝 $H_{0,(1)}, H_{0,(2)}, \ldots, H_{0,(k)}$（前 $k$ 个最小 $p$ 值对应的假设）。

BH 控制的是 FDR $= E[\text{假阳性数}/\max(\text{拒绝数}, 1)] \leq \alpha \cdot m_0/m \leq \alpha$。

**比 Bonferroni 宽松的原因**：

- Bonferroni 控制 FWER（任一假阳性的概率），是最严格的错误指标。
- BH 控制 FDR（假阳性占拒绝数的期望比例），允许存在少量假阳性，检验功效（真阳性发现数）显著提升。
- 当 $m_0 < m$（部分假设为假）时，BH 的有效校正阈值更宽松，功效优势更明显。

**答案**：$\boxed{\text{FWER} \approx 64.2\%\text{；Bonferroni: }\alpha^* = 0.0025\text{，FWER} \leq 5\%\text{；BH 控制 FDR，功效更高}}$

> ⚠️ Bonferroni 校正假设检验相互独立（或正相关），若检验间存在负相关则可能过于保守。BH 方法在独立或正相关（PRDS 条件）下成立。

---

## E 提高题详解（7 题）

### E.7.1（Ch.19，Neyman-Pearson 引理 + 最优检验 + 似然比检验）

**题目**：证明 NP 引理，推广到 UMP，计算正态模型最优检验，讨论 ROC 与 AUC。

**思路**：NP 引理通过比较两个检验的功效差推导最优性；指数族 MLR 保证 UMP；AUC = P(正例得分 > 负例得分)。

**解**：

**（a）NP 引理的证明**

设 $\phi^*$ 为似然比检验：

$$\phi^*(x) = \begin{cases} 1 & \text{if } \Lambda(x) = L(\theta_1;x)/L(\theta_0;x) > k \\ \gamma & \text{if } \Lambda(x) = k \\ 0 & \text{if } \Lambda(x) < k \end{cases}$$

其中 $k, \gamma$ 选取使得 $E_{\theta_0}[\phi^*] = \alpha$（满足水平约束）。

设 $\phi$ 为任意满足 $E_{\theta_0}[\phi] \leq \alpha$ 的检验。需证 $E_{\theta_1}[\phi^*] \geq E_{\theta_1}[\phi]$。

考虑差：

$$E_{\theta_1}[\phi^* - \phi] = \int (\phi^* - \phi) f(x;\theta_1)\, dx$$

注意到当 $\phi^* > \phi$（即 $\phi^*$ 拒绝而 $\phi$ 不拒绝时），必有 $\Lambda(x) \geq k$，故 $f(x;\theta_1) \geq k\, f(x;\theta_0)$；类似地，当 $\phi^* < \phi$ 时，$f(x;\theta_1) \leq k\, f(x;\theta_0)$。

因此：

$$E_{\theta_1}[\phi^* - \phi] \geq k \int (\phi^* - \phi) f(x;\theta_0)\, dx = k\bigl(E_{\theta_0}[\phi^*] - E_{\theta_0}[\phi]\bigr) \geq k(\alpha - \alpha) = 0$$

故 $E_{\theta_1}[\phi^*] \geq E_{\theta_1}[\phi]$，NP 引理成立。$\square$

**（b）推广到 UMP：指数族单调似然比**

设 $\{p_\theta\}$ 为指数族：$p_\theta(x) = h(x)\exp(\eta(\theta)T(x) - A(\theta))$，$\eta(\theta)$ 单调递增。

则似然比 $\Lambda(\theta_1, \theta_0; x) = \exp\bigl((\eta(\theta_1) - \eta(\theta_0))T(x) - (A(\theta_1) - A(\theta_0))\bigr)$ 关于充分统计量 $T(x)$ 单调递增（当 $\theta_1 > \theta_0$，即 $\eta(\theta_1) > \eta(\theta_0)$）。

此**单调似然比（MLR）**性质保证：对任意 $\theta_0 < \theta_1$，NP 最优检验 $\{T(x) > c\}$ 与 $\theta_1$ 无关，故它对所有 $\theta_1 > \theta_0$ 均最优，即存在**一致最优势检验（UMP）**。

**（c）正态模型数值计算（$n = 25$，$\alpha = 0.05$）**

检验 $H_0: \mu = 0$ vs $H_1: \mu = 1$，$X_i \sim N(\mu, 1)$，$\bar{X} = \frac{1}{25}\sum X_i \sim N(\mu, 1/25)$。

**最优拒绝域**（NP 引理）：

$$\bar{X} > c, \quad \text{其中 } P(\bar{X} > c \mid \mu = 0) = 0.05$$

$$\frac{c - 0}{1/\sqrt{25}} = z_{0.05} = 1.645 \implies c = \frac{1.645}{5} = 0.329$$

**功效**（$H_1: \mu = 1$）：

$$\text{Power} = P(\bar{X} > 0.329 \mid \mu = 1) = P\!\left(Z > \frac{0.329 - 1}{1/5}\right) = P(Z > -3.355) = \Phi(3.355) \approx 0.9996$$

**第二类错误率**：$\beta = 1 - \text{Power} \approx 0.0004$（极低，因 $n = 25$ 足够大且 $\mu_1 - \mu_0 = 1$）。

**功效函数的形状**（关于 $\mu$）：

$$\pi(\mu) = P(\bar{X} > 0.329 \mid \mu) = 1 - \Phi\!\left(\frac{0.329 - \mu}{0.2}\right)$$

- $\mu \to -\infty$：$\pi(\mu) \to 0$
- $\mu = 0$（$H_0$）：$\pi(0) = 0.05 = \alpha$
- $\mu = 0.329$（拒绝边界）：$\pi = 0.5$
- $\mu = 1$（$H_1$）：$\pi \approx 0.9996$
- $\mu \to +\infty$：$\pi(\mu) \to 1$

形状为 $S$ 型单调递增曲线（正态 CDF 的补）。

**（d）ROC 曲线与 AUC 的统计解释**

**AUC = P(正例得分 > 负例得分)**的证明：

设正例得分 $\hat{p}_+ \sim F$，负例得分 $\hat{p}_- \sim G$，ROC 曲线上点 $(FPR, TPR)$ 对应阈值 $t$：

$$\text{TPR}(t) = P(\hat{p}_+ > t), \quad \text{FPR}(t) = P(\hat{p}_- > t)$$

$$\text{AUC} = \int_0^1 \text{TPR}\, d(\text{FPR}) = \int_{-\infty}^{+\infty} P(\hat{p}_+ > t) \cdot g(t)\, dt = P(\hat{p}_+ > \hat{p}_-)$$

（最后一步用独立性：$E_{\hat{p}_-}[P(\hat{p}_+ > \hat{p}_-)] = P(\hat{p}_+ > \hat{p}_-)$。）

AUC 最优化等价于最大化 Wilcoxon 统计量，与 NP 框架的联系：最优分类器（LRT 决策边界）在所有阈值下均实现最高 TPR，即 ROC 曲线位于所有分类器曲线之上。精度-召回率权衡反映：高召回率区域（低阈值）FPR 上升，类似降低 $\alpha$ 使 $\beta$ 上升的权衡。

**答案**：$\boxed{拒绝域 \bar{X} > 0.329\text{，功效} \approx 99.96\%\text{，}\beta \approx 0.04\%\text{；AUC} = P(\hat{p}_+ > \hat{p}_-)}$

---

### E.7.2（Ch.19+Ch.20，p 值 + 多重检验 + FDR 控制）

**题目**：同时检验 $m$ 个假设，证明 FDR $\leq$ FWER，证明 Bonferroni 控制 FWER，证明 BH 控制 FDR，讨论高维 ML 中的含义。

**思路**：FDR 是期望假阳性比例，FWER 是至少一次假阳性的概率；Bonferroni 用 Boole 不等式；BH 证明需利用秩结构。

**解**：

**（a）FDR $\leq$ FWER 的证明**

设 $V$ = 假阳性数，$R$ = 拒绝数（含真假阳性）。

$$\text{FDR} = E\!\left[\frac{V}{\max(R,1)}\right] \leq E\!\left[\frac{V}{1} \cdot \mathbf{1}(V \geq 1)\right] = E[V \cdot \mathbf{1}(V \geq 1)]$$

但 $V \cdot \mathbf{1}(V \geq 1) \leq \mathbf{1}(V \geq 1)$（因为当 $V \geq 1$ 时 $V/\max(R,1) \leq 1$），故：

$$\text{FDR} = E\!\left[\frac{V}{\max(R,1)}\right] \leq E[\mathbf{1}(V \geq 1)] = P(V \geq 1) = \text{FWER}$$

$\square$

**（b）Bonferroni 校正证明 FWER $\leq \alpha$**

设每个检验 $i$ 的水平为 $\alpha_i = \alpha/m$，$A_i$ 为第 $i$ 个真零假设被错误拒绝的事件。

由 Boole 不等式（Union Bound）：

$$\text{FWER} = P\!\left(\bigcup_{i: H_{0,i}\text{ 真}} A_i\right) \leq \sum_{i: H_{0,i}\text{ 真}} P(A_i) \leq m_0 \cdot \frac{\alpha}{m} \leq m \cdot \frac{\alpha}{m} = \alpha$$

（其中 $m_0 \leq m$ 为真零假设数，该不等式无需独立性假设。）$\square$

**（c）BH 方法控制 FDR 的核心推导框架**

将 $p$ 值排序：$p_{(1)} \leq \cdots \leq p_{(m)}$，BH 拒绝 $H_{0,(1)}, \ldots, H_{0,(k)}$，其中 $k = \max\{i: p_{(i)} \leq i\alpha/m\}$。

关键推导（Benjamini-Hochberg 1995，独立或 PRDS 情形）：

设 $m_0$ 为真零假设数，对真零假设 $i$，其 $p$ 值 $p_i \sim U(0,1)$（在 $H_0$ 下）。

$$\text{FDR} = E\!\left[\frac{V}{R}\right] = \sum_{i=1}^{m_0} E\!\left[\frac{\mathbf{1}(p_i \leq p_{(k)})}{R}\right] \leq \sum_{i=1}^{m_0} E\!\left[\frac{\mathbf{1}(p_i \leq k\alpha/m)}{k}\right] = \frac{m_0}{m}\alpha \leq \alpha$$

（关键步骤：若 $p_i$ 被拒绝，则 $p_{(k)} \geq p_i$，且拒绝数 $R \geq k$，利用独立性得上界。）

故 $\text{FDR} \leq \alpha \cdot m_0/m \leq \alpha$。$\square$

**（d）高维 ML 中多重检验的必要性**

在 $m \gg n$ 场景（如特征数远超样本数）：

1. **梯度显著性**：即便某特征对输出无影响（$H_0$ 为真），其梯度估计在有限样本下也有非零方差，单次检验会产生假阳性。对 $m$ 个特征同时检验若不校正，假阳性数 $\approx m \times \alpha$（可达数百）。

2. **SHAP 值的局限**：SHAP 值给出每个特征的贡献，但不提供统计检验；在 $m \gg n$ 时，高 SHAP 值可能纯属随机波动。正确做法：结合置换检验（对每个特征单独计算置换后 SHAP 分布）并用 BH 校正。

3. **Bonferroni vs BH**：高维特征中真正重要特征往往稀疏（$m_1 \ll m_0$），BH 的有效阈值 $k\alpha/m$ 比 Bonferroni 的 $\alpha/m$ 宽松，能发现更多真实信号，在生物信息学（基因表达分析）和 NLP（注意力显著性）中广泛应用。

**答案**：$\boxed{\text{FDR} \leq \text{FWER}\text{；Bonferroni via Boole 不等式；BH: FDR} \leq \alpha m_0/m\text{；高维 ML 需校正}}$

---

### E.7.3（Ch.20，方差分析（ANOVA）+ 线性模型 + F 检验理论）

**题目**：单因素 ANOVA 模型 $X_{ij} = \mu + \alpha_i + \varepsilon_{ij}$，推导 $SS_A$、$SS_E$ 的分布，证明 F 统计量，推导功效，讨论 BN 的 ANOVA 解释。

**思路**：ANOVA 通过平方和分解和自由度分配建立 F 检验；Cochran 定理保证各 SS 独立且服从 χ² 分布。

**解**：

**（a）自由度推导及 $SS_E$ 的分布**

模型：$X_{ij} = \mu + \alpha_i + \varepsilon_{ij}$，$i=1,\ldots,k$，$j=1,\ldots,n_i$，$N = \sum n_i$，约束 $\sum n_i\alpha_i = 0$。

**$SS_A$ 的自由度**：$SS_A = \sum_{i=1}^k n_i(\bar{X}_{i\cdot} - \bar{X}_{\cdot\cdot})^2$。

自由度 $df_A = k - 1$（$k$ 个组均值减去 1 个约束 $\sum n_i(\bar{X}_{i\cdot}-\bar{X}_{\cdot\cdot}) = 0$）。

**$SS_E$ 的自由度**：$SS_E = \sum_{i,j}(X_{ij} - \bar{X}_{i\cdot})^2$。

第 $i$ 组贡献 $n_i - 1$ 个自由度（$n_i$ 个偏差和为 0），故 $df_E = \sum_i(n_i - 1) = N - k$。

**$SS_E/\sigma^2 \sim \chi^2(N-k)$ 的证明**：

在各组内，$X_{ij} - \bar{X}_{i\cdot} \sim N(0, \sigma^2(1 - 1/n_i))$（独立）。由 Cochran 定理，对正态总体，$\sum_j(X_{ij}-\bar{X}_{i\cdot})^2/\sigma^2 \sim \chi^2(n_i-1)$，各组独立，故：

$$\frac{SS_E}{\sigma^2} = \sum_{i=1}^k \frac{\sum_j(X_{ij}-\bar{X}_{i\cdot})^2}{\sigma^2} \sim \chi^2(N-k)$$

**（b）$H_0$ 下 F 统计量的分布**

在 $H_0: \alpha_1 = \cdots = \alpha_k = 0$ 下，$X_{ij} \sim N(\mu, \sigma^2)$，故 $\bar{X}_{i\cdot} \sim N(\mu, \sigma^2/n_i)$，$\bar{X}_{\cdot\cdot} \sim N(\mu, \sigma^2/N)$。

由 Cochran 定理：$SS_A/\sigma^2 \sim \chi^2(k-1)$，且 $SS_A$ 与 $SS_E$ 相互独立（分别是均值的函数和残差的函数，在正态假设下正交）。

$$F = \frac{SS_A/(k-1)}{SS_E/(N-k)} = \frac{MS_A}{MS_E} = \frac{\chi^2(k-1)/(k-1)}{\chi^2(N-k)/(N-k)} \sim F(k-1, N-k)$$

$\square$

**（c）非中心 F 分布与功效**

在 $H_1$（$\alpha_i$ 不全为零）下，$SS_A/\sigma^2 \sim \chi^2(k-1, \lambda)$（非中心χ²），非中心参数：

$$\lambda = \frac{\sum_{i=1}^k n_i \alpha_i^2}{\sigma^2}$$

此时 $F \sim F(k-1, N-k, \lambda)$（非中心 F 分布），功效为：

$$\text{Power} = P(F(k-1, N-k, \lambda) > F_\alpha(k-1, N-k))$$

**样本量设计**：给定目标功效 $1-\beta$ 和效应大小（$\lambda$ 的函数），求所需 $n_i$（利用非中心 F 分布的分位数表或软件，如 R 的 `pwr` 包）。

一般原则：$\lambda$ 越大（组间差异越大或 $\sigma^2$ 越小），功效越高；增大 $n_i$ 等比例增大 $\lambda$。

**（d）批归一化（BN）的 ANOVA 解释**

深度网络第 $l$ 层的激活 $\{z^{(l)}_j\}$ 受前层参数分布变化影响（内部协变量偏移，Internal Covariate Shift）。ANOVA 框架下：

- **"组"**：不同训练 mini-batch（或不同层）的激活分布。
- **组间变异（$SS_A$）**：对应层间协变量偏移——激活均值和方差在不同 batch 间的差异。
- **组内变异（$SS_E$）**：批内正常激活波动（信号）。

BN 通过标准化每层激活（均值为 0，方差为 1），等价于将 $SS_A \to 0$，消除层间均值/方差漂移，使梯度信号更稳定（相当于固定组间差异为零，减少高阶相互作用）。从正则化视角：BN 引入的 $(\gamma, \beta)$ 可学习参数类似随机效应模型中的组效应估计，防止任何单层"主导"整体变差。

**答案**：$\boxed{SS_E/\sigma^2 \sim \chi^2(N-k)\text{；}F \sim F(k-1,N-k)\text{（}H_0\text{下）；}\lambda = \sum n_i\alpha_i^2/\sigma^2}$

---

### E.7.4（Ch.20+Ch.21，非参数检验 + 秩统计量 + 渐近理论）

**题目**：证明 Wilcoxon 秩和统计量的均值和方差，与 Mann-Whitney U 的关系，渐近正态性，以及 ML 模型比较中的应用。

**思路**：秩统计量的矩通过组合期望计算；渐近正态由 Hájek 投影（Hajek projection）定理；U 统计量框架统一 W 和 U。

**解**：

**（a）$E[W]$ 和 $\mathrm{Var}(W)$ 的推导**

设合并样本共 $N = m + n$ 个，$X_i$ 的秩 $R_i \in \{1, 2, \ldots, N\}$（无并列）。

在 $H_0: F = G$ 下，$N$ 个样本可交换，$X$ 的 $m$ 个秩是 $\{1,\ldots,N\}$ 的等概率子集。

**期望**：

$$E[R_i] = \frac{1+2+\cdots+N}{N} = \frac{N+1}{2}$$

$$E[W] = \sum_{i=1}^m E[R_i] = m \cdot \frac{N+1}{2} = \frac{m(m+n+1)}{2}$$

**方差**：

$$\mathrm{Var}(W) = \sum_{i=1}^m \mathrm{Var}(R_i) + \sum_{i \neq j} \mathrm{Cov}(R_i, R_j)$$

由对称性，$E[R_i^2] = \frac{\sum_{k=1}^N k^2}{N} = \frac{N(N+1)(2N+1)}{6N} = \frac{(N+1)(2N+1)}{6}$，

$\mathrm{Var}(R_i) = E[R_i^2] - (E[R_i])^2 = \frac{(N+1)(2N+1)}{6} - \frac{(N+1)^2}{4} = \frac{(N+1)(N-1)}{12}$。

对 $i \neq j$，$E[R_i R_j] = \frac{1}{N(N-1)}\sum_{k \neq l} kl = \frac{(\sum k)^2 - \sum k^2}{N(N-1)} = \frac{N^2(N+1)^2/4 - N(N+1)(2N+1)/6}{N(N-1)}$，

化简得 $E[R_i R_j] = \frac{(N+1)(3N+2)}{12}$，

$\mathrm{Cov}(R_i, R_j) = E[R_iR_j] - E[R_i]E[R_j] = \frac{(N+1)(3N+2)}{12} - \frac{(N+1)^2}{4} = -\frac{N+1}{12}$。

$$\mathrm{Var}(W) = m\cdot\frac{(N+1)(N-1)}{12} + m(m-1)\cdot\left(-\frac{N+1}{12}\right) = \frac{(N+1)}{12}[m(N-1) - m(m-1)]$$

$$= \frac{(N+1)}{12} \cdot m(N-m) = \frac{mn(m+n+1)}{12}$$

$\square$

**（b）$W$ 与 $U$ 的关系**

Mann-Whitney U 统计量 $U = \sum_{i=1}^m\sum_{j=1}^n \mathbf{1}[X_i > Y_j]$。

$X_i$ 的秩 $R_i$ 等于合并样本中比 $X_i$ 小的观测数加 1：$R_i = 1 + \#\{X_{i'}: X_{i'} < X_i\} + \#\{Y_j: Y_j < X_i\}$。

$$W = \sum_{i=1}^m R_i = m + \sum_{i=1}^m\#\{i': X_{i'} < X_i\} + \sum_{i=1}^m\#\{j: Y_j < X_i\}$$

$$= m + \frac{m(m-1)}{2} + \frac{m(m+1)}{2} - m + U = \frac{m(m+1)}{2} + U$$

故 $U = W - m(m+1)/2$。$\square$

**（c）渐近正态性（Hájek 投影定理框架）**

定义 $h(x, y) = \mathbf{1}[x > y]$，则 $U/\binom{N}{m}$ 是关于 $(X_1,\ldots,X_m, Y_1,\ldots,Y_n)$ 的 U 统计量，核函数 $h$ 为 1-1。

**Hájek 投影**：将 $U$ 投影到个体的函数之和：

$$\tilde{U} = E[U] + \sum_{i=1}^m (h_1(X_i) - E[h_1(X)]) + \sum_{j=1}^n (h_2(Y_j) - E[h_2(Y)])$$

其中 $h_1(x) = P(x > Y)$，$h_2(y) = P(X > y)$。

在 $H_0$（$F = G$）下，$h_1(x) = h_2(x) = 1 - F(x)$（连续），$E[h_1(X)] = 1/2$。

由 CLT，$\tilde{U}$ 渐近正态，且 $\mathrm{Var}(U - \tilde{U}) = o(\mathrm{Var}(\tilde{U}))$（投影残差可忽略），故：

$$\frac{W - E[W]}{\sqrt{\mathrm{Var}(W)}} = \frac{U - mn/2}{\sqrt{mn(m+n+1)/12}} \xrightarrow{d} N(0,1) \quad (m, n \to \infty)$$

$\square$

**（d）ML 模型比较中的秩检验框架**

在多数据集模型比较场景（Demšar 2006 框架）：

- **Wilcoxon 符号秩检验**（两模型配对比较）：对 $k$ 个数据集上的性能差值 $d_i = \text{acc}_A^{(i)} - \text{acc}_B^{(i)}$ 用 Wilcoxon 符号秩检验，无需正态假设。

- **$t$ 检验 vs Wilcoxon 的优势**：性能指标（准确率、F1 等）在数据集间差异巨大，可能有极端值（如某数据集异常困难）；$t$ 检验对异常值敏感，Wilcoxon 基于秩，鲁棒性更强。渐近相对效率（ARE）在重尾分布下 Wilcoxon $\gg t$。

- **多模型比较**：用 Friedman 检验（秩的双因素 ANOVA 推广）检验 $k$ 个分类器在 $N$ 个数据集上的总体差异，事后两两用 Nemenyi 检验（类似 Tukey）。

- **关键限制**：数据集之间并非独立（来自相同分布的子集），需额外用 5×2 交叉验证等方法估计检验统计量的方差。

**答案**：$\boxed{E[W] = m(m+n+1)/2\text{，}\mathrm{Var}(W) = mn(m+n+1)/12\text{；}U = W - m(m+1)/2\text{；渐近正态由 Hájek 投影}}$

---

### E.7.5（Ch.19+Ch.20，顺序检验 + Wald 序贯概率比检验 + 提前终止）

**题目**：证明对数似然比过程在 $H_0$ 下期望为负，叙述 SPRT，证明 Wald-Wolfowitz 最优性，讨论 A/B 测试的持续监测问题。

**思路**：SPRT 通过连续监测对数似然比并设置双边界实现最优顺序检验；鞅不等式解决持续监测膨胀问题。

**解**：

**（a）$H_0$ 下 $E[\Lambda_n] < 0$ 的证明**

$$\Lambda_n = \sum_{i=1}^n \log\frac{p_{\theta_1}(X_i)}{p_{\theta_0}(X_i)}$$

在 $H_0: \theta = \theta_0$ 下，$X_i \overset{i.i.d.}{\sim} p_{\theta_0}$，故：

$$E_{\theta_0}[\Lambda_n] = n \cdot E_{\theta_0}\!\left[\log\frac{p_{\theta_1}(X)}{p_{\theta_0}(X)}\right] = -n \cdot E_{\theta_0}\!\left[\log\frac{p_{\theta_0}(X)}{p_{\theta_1}(X)}\right] = -n \cdot D_{\mathrm{KL}}(p_{\theta_0} \| p_{\theta_1})$$

由 KL 散度非负性（$D_{\mathrm{KL}} \geq 0$，等号当且仅当 $p_{\theta_0} = p_{\theta_1}$ a.e.），在 $\theta_0 \neq \theta_1$ 时：

$$E_{\theta_0}[\Lambda_n] = -n \cdot D_{\mathrm{KL}}(p_{\theta_0}\| p_{\theta_1}) < 0$$

$\square$（在 $H_0$ 下对数似然比过程期望负漂移，即趋向接受边界。）

**（b）Wald SPRT 的叙述**

设阈值 $A < 0 < B$（通常 $A \approx \log(\beta/(1-\alpha))$，$B \approx \log((1-\beta)/\alpha)$）。

**SPRT 规则**：每次获得新观测 $X_n$ 后更新 $\Lambda_n$：

- 若 $\Lambda_n \geq B$：**停止并拒绝 $H_0$**（有足够证据支持 $H_1$）
- 若 $\Lambda_n \leq A$：**停止并接受 $H_0$**
- 若 $A < \Lambda_n < B$：**继续采样**

**误差控制**（近似）：

- $\alpha' = P_{\theta_0}(\text{拒绝}H_0) \approx \frac{e^A - 1}{e^A - e^B} \approx \frac{\beta}{1-\alpha+\beta} \leq \alpha$
- $\beta' = P_{\theta_1}(\text{接受}H_0) \approx \frac{e^B - 1}{e^B - e^A} \cdot e^{-B} \approx \frac{\beta}{1-\alpha}$（精确到过冲误差）

**（c）Wald-Wolfowitz 最优性定理**

定理：设所有满足 $P_{\theta_0}(\text{拒绝}) \leq \alpha$ 且 $P_{\theta_1}(\text{接受}) \leq \beta$ 的顺序检验（包括固定样本检验），SPRT 的期望样本量最小：

$$E_{\theta_0}[N_{\text{SPRT}}] \leq E_{\theta_0}[N]\text{（任意满足约束的检验 )}$$

（$H_1$ 下同理。）

**证明框架**：

利用 Wald 等式（停时理论）：

$$E_\theta[\Lambda_{N_{\text{SPRT}}}] \approx E_\theta[N_{\text{SPRT}}] \cdot E_\theta[\log L_1(X)/L_0(X)]$$

在 $H_1$ 下（忽略过冲）：$E[\Lambda_N] \approx B = \log((1-\beta)/\alpha)$，故：

$$E_{\theta_1}[N] \approx \frac{\log((1-\beta)/\alpha)}{D_{\mathrm{KL}}(p_{\theta_1}\| p_{\theta_0})}$$

相比固定样本量检验（达到同等 $(\alpha,\beta)$ 需 $n \propto (z_\alpha + z_\beta)^2$），SPRT 平均节约约 50% 样本量。

**（d）A/B 测试持续监测与 always-valid p 值**

**Peeking Problem（持续监测膨胀）**：若在 $n_1, n_2, \ldots$ 多个时间点反复查看 $p$ 值并在 $p < 0.05$ 时停止，实际 I 类错误远超 5%。原因：每次查看等价于一次额外的多重检验；若查看 $K$ 次，FWER 约为 $1 - (1-\alpha)^K \gg \alpha$。

**Always-valid p 值（基于鞅不等式）**：

定义似然比过程 $\Lambda_n = L_n(\theta_1)/L_n(\theta_0)$，在 $H_0$ 下 $\Lambda_n$ 是非负鞅（$E_{\theta_0}[\Lambda_n | \mathcal{F}_{n-1}] = \Lambda_{n-1}$）。

由 Ville 不等式（鞅的可选停时定理推广）：

$$P_{\theta_0}\!\left(\exists n: \Lambda_n \geq 1/\alpha\right) \leq \alpha$$

故定义 $p_n = 1/\Lambda_n$，则任意时间查看均有 $P_{\theta_0}(p_n \leq \alpha \text{ for some }n) \leq \alpha$。这给出精确的 always-valid p 值，不因反复查看而膨胀。

**工程实现**：

1. 选择参数 $(\theta_0, \theta_1, \alpha, \beta)$，计算 SPRT 边界 $(A, B)$。
2. 每批新数据到达时更新 $\Lambda_n$。
3. 超过 $B$ 时停止并报告，输出 always-valid confidence sequence（置信序列）。
4. 用 mSPRT（mixture SPRT，对 $\theta_1$ 取先验混合）处理复合 $H_1$，适合不知道真实效应大小的场景。

**答案**：$\boxed{E_{\theta_0}[\Lambda_n] = -nD_{\mathrm{KL}} < 0\text{；SPRT 边界 }(A,B)\text{；Wald-Wolfowitz 最优；鞅不等式解决 Peeking}}$

---

### E.7.6（Ch.21，置换检验 + 精确检验 + 条件推断）

**题目**：证明置换检验的精确水平，推导置换 p 值，Fisher 精确检验，讨论 ML 特征置换重要性。

**思路**：置换检验在交换性下构造精确的条件参考分布；Fisher 精确检验基于超几何分布的条件推断。

**解**：

**（a）置换检验的精确水平**

设 $\mathbf{x} = (x_1,\ldots,x_m)$ 和 $\mathbf{y} = (y_1,\ldots,y_n)$ 为两组独立观测。

**交换性**：在 $H_0: \mu_X = \mu_Y$（或更一般地 $F = G$）下，合并样本 $\mathbf{z} = (\mathbf{x}, \mathbf{y})$ 中任意 $m+n$ 个观测的分配方式（哪 $m$ 个属于 $X$ 组）是等可能的，共 $\binom{m+n}{m}$ 种。

**精确水平的证明**：

设检验统计量为 $T$，观测值为 $t_{\text{obs}}$，置换 p 值为：

$$p_{\text{perm}} = \frac{\#\{\pi: T(\mathbf{z}_\pi) \geq t_{\text{obs}}\}}{\binom{m+n}{m}}$$

在 $H_0$ 下，$t_{\text{obs}}$ 是 $\binom{m+n}{m}$ 个等可能排列统计量中的一个（原始排列与其他排列等地位），故：

$$P_{H_0}(p_{\text{perm}} \leq \alpha) = P_{H_0}\!\left(\text{观测排列位于最极端的 }\alpha\text{部分}\right) \leq \alpha$$

这在有限样本下精确成立（无渐近近似）。$\square$

**（b）Monte Carlo 近似置换 p 值**

精确置换 p 值需枚举所有 $\binom{m+n}{m}$ 种排列，当 $m+n$ 大时计算不可行。

Monte Carlo 近似：

1. 随机生成 $B$ 个置换（如 $B = 10000$）。
2. 对每个置换 $\pi_b$ 计算 $T^{(b)} = T(\mathbf{z}_{\pi_b})$。
3. 近似 p 值：$\hat{p} = \frac{1 + \#\{b: T^{(b)} \geq t_{\text{obs}}\}}{B+1}$（分母加1避免 $\hat{p} = 0$）。

精度：$\hat{p}$ 的标准误约 $\sqrt{\alpha(1-\alpha)/B}$；$B = 10000$ 时 $\alpha = 0.05$ 的误差 $\approx 0.002$，通常足够。

**（c）Fisher 精确检验（$2 \times 2$ 列联表）**

设列联表行和为 $(R_1, R_2)$，列和为 $(C_1, C_2)$，总数 $n$。在边际固定的条件下，$O_{11}$ 的分布为超几何分布：

$$P(O_{11} = k \mid R_1, R_2, C_1, C_2) = \frac{\binom{C_1}{k}\binom{C_2}{R_1-k}}{\binom{n}{R_1}}$$

**精确 p 值**（单侧右尾：$H_1$：正向关联）：

$$p = \sum_{k \geq O_{11}^{\text{obs}}} \frac{\binom{C_1}{k}\binom{C_2}{R_1-k}}{\binom{n}{R_1}}$$

双侧 p 值取两尾之和（或所有概率 $\leq P(O_{11}^{\text{obs}})$ 的事件概率之和）。

Fisher 检验无需近似（χ² 检验需 $E_{ij} \geq 5$），对小样本尤其重要。

**（d）ML 特征置换重要性与高相关特征**

**置换特征重要性（Permutation Feature Importance, PFI）**：

对特征 $j$，随机置换训练好的模型中特征 $j$ 的值（打乱该特征与响应变量的关联），记录模型性能下降：$\Delta_j = \text{metric}(\hat{y}, y) - \text{metric}(\hat{y}^{(\pi_j)}, y)$。

**与条件独立性的联系**：$\Delta_j > 0$ 意味着特征 $j$ 对预测有贡献，等价于拒绝 $H_0: X_j \perp Y \mid X_{-j}$（$X_{-j}$ 为其他特征）。

**高相关特征低估重要性的原因**：若特征 $j$ 与特征 $k$ 高度相关，置换 $j$ 后，模型可用 $k$ "代替" $j$ 的信息，导致性能下降幅度小——$\Delta_j$ 被低估。

**分组置换改进**：将高相关特征组 $\mathcal{G} = \{j_1, j_2, \ldots\}$ 同时置换（组内联合置换），测量整组的联合重要性 $\Delta_{\mathcal{G}}$，避免单特征置换的"替代效应"。可结合 SHAP 的 Shapley 值（公平分配联合贡献到各特征）进一步细化。

**答案**：$\boxed{\text{置换检验精确水平 }\leq \alpha\text{（交换性）；Fisher 精确检验用超几何分布；分组置换解决高相关问题}}$

---

### E.7.7（Ch.19+Ch.21，贝叶斯假设检验 + 贝叶斯因子 + 模型选择）

**题目**：证明后验比 = 先验比 × BF，计算正态数据的贝叶斯因子，分析 Jeffreys-Lindley 悖论，推导 BIC 并与 AIC 比较。

**思路**：贝叶斯因子量化数据对两模型的相对支持；BIC 是边际似然的 Laplace 近似；Lindley 悖论揭示先验敏感性。

**解**：

**（a）后验比 = 先验比 × BF**

由贝叶斯定理：

$$P(M_1 \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid M_1) P(M_1)}{p(\mathbf{x} \mid M_0)P(M_0) + p(\mathbf{x} \mid M_1)P(M_1)}$$

$$P(M_0 \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid M_0) P(M_0)}{p(\mathbf{x} \mid M_0)P(M_0) + p(\mathbf{x} \mid M_1)P(M_1)}$$

相除：

$$\frac{P(M_1 \mid \mathbf{x})}{P(M_0 \mid \mathbf{x})} = \frac{p(\mathbf{x} \mid M_1)}{p(\mathbf{x} \mid M_0)} \cdot \frac{P(M_1)}{P(M_0)} = BF_{10} \cdot \frac{P(M_1)}{P(M_0)}$$

即：**后验优势比（Posterior Odds）= 贝叶斯因子（BF）× 先验优势比（Prior Odds）**。$\square$

**（b）正态数据的贝叶斯因子（含 Laplace 近似）**

数据 $X_i \overset{i.i.d.}{\sim} N(\mu, 1)$，$n$ 个观测，充分统计量 $\bar{x}$。

- $M_0$：$\mu = 0$（点假设），$p(\mathbf{x} \mid M_0) = (2\pi)^{-n/2} \exp(-\sum x_i^2/2)$
- $M_1$：$\mu \sim N(0, \tau^2)$（正态先验），边际似然为：

$$p(\mathbf{x} \mid M_1) = \int_{-\infty}^{+\infty} p(\mathbf{x} \mid \mu) \pi(\mu)\, d\mu$$

正态-正态共轭积分精确可解：

$$\sum x_i^2 - \frac{n^2\bar{x}^2}{n + 1/\tau^2}$$

完成配方后：

$$p(\mathbf{x} \mid M_1) = \frac{1}{(2\pi)^{n/2}} \cdot \frac{1}{\sqrt{1 + n\tau^2}} \exp\!\left(-\frac{\sum x_i^2}{2} + \frac{n^2\bar{x}^2/2}{n + 1/\tau^2}\right)$$

**贝叶斯因子**：

$$BF_{10} = \frac{p(\mathbf{x} \mid M_1)}{p(\mathbf{x} \mid M_0)} = \frac{1}{\sqrt{1+n\tau^2}} \exp\!\left(\frac{n\tau^2}{2(1+n\tau^2)} \cdot n\bar{x}^2\right)$$

**Laplace 近似**（当 $n$ 大时）：设 $M_1$ 中 $\mu$ 的 MLE 为 $\hat{\mu} = \bar{x}$，后验近似为正态，则：

$$\log p(\mathbf{x} \mid M_1) \approx \log p(\mathbf{x} \mid \hat{\mu}) + \log\pi(\hat{\mu}) - \frac{1}{2}\log(n/(2\pi))$$

$$\log BF_{10} \approx \frac{n\bar{x}^2}{2(1+n\tau^2)} - \frac{1}{2}\log(1+n\tau^2)$$

**（c）Jeffreys-Lindley 悖论**

当 $\tau \to \infty$（无信息先验），$BF_{10}$ 的渐近行为：

$$BF_{10} = \frac{1}{\sqrt{1+n\tau^2}} \exp(\cdots) \approx \frac{1}{\tau\sqrt{n}} \cdot \exp\!\left(\frac{\bar{x}^2 n}{2} \cdot 1\right) \to 0 \quad (\tau \to \infty)$$

即无论数据如何，先验弥散时 $BF_{10} \to 0$——贝叶斯因子**永远支持 $M_0$**！

**原因**：当 $\tau$ 很大时，$M_1$ 的先验将概率质量分散到极大的 $|\mu|$ 范围，数据（有限均值 $\bar{x}$）相对于如此弥散的先验而言"出人意料"，边际似然 $p(\mathbf{x} \mid M_1)$ 极小，故 BF 倒向 $M_0$。

**与频率 p 值的对比**：

- 频率 p 值 $= P(|\bar{X}| \geq |\bar{x}| \mid \mu=0)$ 只依赖数据和 $M_0$，与 $M_1$ 的先验无关。
- 贝叶斯因子对 $M_1$ 的先验 $\tau$ 极度敏感，需谨慎选择（推荐 Cauchy 或 JZS 先验代替无信息先验）。

**（d）BIC 的贝叶斯推导与 AIC 比较**

**BIC 的 Laplace 近似推导**：

设模型 $M$ 有 $d$ 个参数，MLE 为 $\hat{\boldsymbol{\theta}}$，$n$ 个观测。对边际似然做 Laplace 近似：

$$\log p(\mathbf{x} \mid M) = \int p(\mathbf{x} \mid \boldsymbol{\theta})\pi(\boldsymbol{\theta})\, d\boldsymbol{\theta}$$

在 $\hat{\boldsymbol{\theta}}$ 处二阶展开（正则条件下 Hessian $\approx -nI(\hat{\boldsymbol{\theta}})$）：

$$\log p(\mathbf{x} \mid M) \approx \log p(\mathbf{x} \mid \hat{\boldsymbol{\theta}}) + \log\pi(\hat{\boldsymbol{\theta}}) + \frac{d}{2}\log(2\pi) - \frac{1}{2}\log\det(nI(\hat{\boldsymbol{\theta}}))$$

忽略先验项和常数，取负得惩罚：

$$-2\log p(\mathbf{x} \mid M) \approx -2\log p(\mathbf{x} \mid \hat{\boldsymbol{\theta}}) + d\log n$$

定义 $\text{BIC} = -2\log\hat{L} + d\log n$，**最小化 BIC 等价于最大化边际似然的 Laplace 近似**。

**BIC vs AIC**（$\text{AIC} = -2\log\hat{L} + 2d$）的比较：

| 准则 | 惩罚项 | 渐近目标 | 优势 |
|------|--------|---------|------|
| AIC | $2d$ | 最小化预测 KL 散度 | 预测性能好（过小罚，允许偏大模型）|
| BIC | $d\log n$ | 最大化边际似然（选真模型）| 一致性（真模型在候选集时）|

- **AIC**（Akaike 1973）：通过 KL 散度的无偏估计推导，最优化样本外预测，是非贝叶斯的。当 $\log n > 2$（即 $n > 8$）时，BIC 比 AIC 对复杂度的惩罚更重。
- **BIC**（Schwarz 1978）：一致选择——当真实模型在候选集中且 $n \to \infty$ 时，BIC 以概率 1 选出真实模型；AIC 可能选过大模型。
- **实践建议**：目标是预测时用 AIC；目标是推断真实生成结构时用 BIC；Bayesian 框架可直接比较边际似然避免此权衡。

**答案**：$\boxed{BF_{10} = (1+n\tau^2)^{-1/2}\exp(\cdots)\text{；}\tau \to \infty \Rightarrow BF_{10} \to 0\text{（Lindley 悖论）；BIC} \approx -2\log\hat{L} + d\log n}$

> ⚠️ Lindley 悖论：当样本量 $n \to \infty$ 且固定 $p$ 值时，BF 可能趋向 $M_0$（频率显著但贝叶斯不支持 $H_1$）。这提醒我们频率检验和贝叶斯检验回答的是不同问题，不可混用。

---

*文件结束。共 29 题（C.7.1–C.7.10，D.7.1–D.7.12，E.7.1–E.7.7）。*
