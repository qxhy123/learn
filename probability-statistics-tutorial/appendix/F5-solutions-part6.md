# F5 详解：Part 6 估计（Ch.16-18，共 34 题）

## C 基础题详解（10 题）

### C.6.1（Ch.16，矩估计）

**题目**：$X \sim U(0,\theta)$，$E[X]=\theta/2$，用矩估计法求 $\hat{\theta}_{\mathrm{MOM}}$，并当 $\bar{x}=4.2$ 时给出数值。

**思路**：令样本一阶矩等于总体一阶矩，解出参数。

**解**：

令 $\bar{X} = E[X] = \theta/2$，解得

$$\hat{\theta}_{\mathrm{MOM}} = 2\bar{X}$$

当 $\bar{x} = 4.2$ 时：

$$\hat{\theta} = 2 \times 4.2 = 8.4$$

**答案**：$\boxed{\hat{\theta}_{\mathrm{MOM}} = 2\bar{X} = 8.4}$

---

### C.6.2（Ch.16，MLE——伯努利分布）

**题目**：$X_i \overset{iid}{\sim} \mathrm{Bernoulli}(p)$，共 $n$ 个样本，$k$ 次成功，求 $\hat{p}_{\mathrm{MLE}}$。

**思路**：写似然 → 取对数 → 对 $p$ 求导令为 0 → 解 → 验证极大。

**解**：

**第一步：写似然函数**

$$L(p) = p^k (1-p)^{n-k}$$

**第二步：取对数似然**

$$\ell(p) = k\log p + (n-k)\log(1-p)$$

**第三步：对 $p$ 求导令为 0**

$$\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0$$

$$k(1-p) = (n-k)p \implies k = np$$

**第四步：解并验证极大**

$$\hat{p}_{\mathrm{MLE}} = \frac{k}{n}$$

二阶导 $d^2\ell/dp^2 = -k/p^2 - (n-k)/(1-p)^2 < 0$，确为极大值。

**答案**：$\boxed{\hat{p}_{\mathrm{MLE}} = k/n}$（即样本成功比例）

---

### C.6.3（Ch.16，MLE——指数分布）

**题目**：$X_i \overset{iid}{\sim} \mathrm{Exp}(\lambda)$，求 $\hat{\lambda}_{\mathrm{MLE}}$。

**思路**：写对数似然 → 求导 → 解方程 → 验证极大。

**解**：

**第一步：写对数似然**

$$\ell(\lambda) = \sum_{i=1}^n \log(\lambda e^{-\lambda x_i}) = n\log\lambda - \lambda\sum_{i=1}^n x_i$$

**第二步：对 $\lambda$ 求导令为 0**

$$\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0$$

**第三步：解**

$$\hat{\lambda}_{\mathrm{MLE}} = \frac{n}{\sum x_i} = \frac{1}{\bar{x}}$$

**第四步：验证极大**：$d^2\ell/d\lambda^2 = -n/\lambda^2 < 0$，确为极大值。

**答案**：$\boxed{\hat{\lambda}_{\mathrm{MLE}} = 1/\bar{X}}$

---

### C.6.4（Ch.16，无偏性验证）

**题目**：$X_1,\ldots,X_n$ i.i.d.，$E[X_i]=\mu$，$\mathrm{Var}(X_i)=\sigma^2$。证明 $\bar{X}$ 是 $\mu$ 的无偏估计，并计算 $\mathrm{Var}(\bar{X})$。

**思路**：利用期望和方差的线性性质。

**解**：

**（1）无偏性**：

$$E[\bar{X}] = E\!\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \frac{1}{n}\cdot n\mu = \mu$$

故 $\bar{X}$ 是 $\mu$ 的无偏估计量。

**（2）方差**：

由于 $X_1,\ldots,X_n$ 独立：

$$\mathrm{Var}(\bar{X}) = \mathrm{Var}\!\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \mathrm{Var}(X_i) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}$$

**答案**：$E[\bar{X}]=\mu$（无偏），$\mathrm{Var}(\bar{X})=\boxed{\sigma^2/n}$

---

### C.6.5（Ch.17，正态均值置信区间——方差已知）

**题目**：$X\sim N(\mu,4)$，$n=16$，$\bar{x}=12.5$，$\sigma^2=4$ 已知，构造 95% CI。

**思路**：方差已知用 $z$ 分布，枢轴量 $(\bar{X}-\mu)/(\sigma/\sqrt{n})\sim N(0,1)$。

**解**：

**（1）枢轴量**：

$$Z = \frac{\bar{X}-\mu}{\sigma/\sqrt{n}} = \frac{\bar{X}-\mu}{2/4} = \frac{\bar{X}-\mu}{0.5} \sim N(0,1)$$

由 $P(-1.96 \leq Z \leq 1.96) = 0.95$，得置信区间

$$\bar{X} \pm z_{0.025}\cdot\frac{\sigma}{\sqrt{n}}$$

**（2）代入数值**：$\sigma/\sqrt{n}=2/4=0.5$

$$[12.5 - 1.96\times0.5,\; 12.5+1.96\times0.5] = [12.5-0.98,\; 12.5+0.98]$$

**答案**：$\boxed{[11.52,\ 13.48]}$

---

### C.6.6（Ch.17，正态均值置信区间——方差未知）

**题目**：$X\sim N(\mu,\sigma^2)$，$\sigma^2$ 未知，$n=10$，$\bar{x}=5.0$，$s=2.0$，求 95% CI。

**思路**：方差未知用 $t$ 分布，枢轴量服从 $t(n-1)$。

**解**：

**（1）枢轴量及其分布**：

$$T = \frac{\bar{X}-\mu}{S/\sqrt{n}} \sim t(n-1) = t(9)$$

**（2）置信区间**：

$$\bar{x} \pm t_{0.025}(9)\cdot\frac{s}{\sqrt{n}} = 5.0 \pm 2.262\times\frac{2.0}{\sqrt{10}} = 5.0 \pm 2.262\times0.6325$$

$$= 5.0 \pm 1.430$$

**答案**：$\boxed{[3.57,\ 6.43]}$

> ⚠️ 方差未知时必须用 $t$ 分布，自由度为 $n-1=9$，不能用 $z_{0.025}=1.96$。

---

### C.6.7（Ch.17，样本量与置信区间宽度）

**题目**：$\sigma^2=100$，希望均值 95% CI 半宽 $\leq E_0=2$，求最小样本量 $n$。

**思路**：半宽公式 $E = z_{0.025}\cdot\sigma/\sqrt{n}$，令 $E \leq 2$ 解 $n$。

**解**：

**（1）不等式**：

$$z_{0.025}\cdot\frac{\sigma}{\sqrt{n}} \leq 2 \implies 1.96\times\frac{10}{\sqrt{n}} \leq 2$$

**（2）解 $n$**：

$$\sqrt{n} \geq \frac{1.96\times10}{2} = 9.8 \implies n \geq 9.8^2 = 96.04$$

取整数，$n_{\min} = 97$。

**答案**：$\boxed{n_{\min} = 97}$

---

### C.6.8（Ch.18，贝叶斯更新——Beta-二项共轭）

**题目**：先验 $p\sim\mathrm{Beta}(2,2)$，观测 10 次试验 7 次正面，求后验分布参数及后验均值。

**思路**：直接应用 Beta-Binomial 共轭公式：后验参数 $= $ 先验参数 $+$ 数据计数。

**解**：

**（1）后验参数**：

先验 $\mathrm{Beta}(\alpha,\beta)=\mathrm{Beta}(2,2)$，$k=7$，$n-k=3$：

$$\alpha' = \alpha+k = 2+7 = 9, \quad \beta' = \beta+(n-k) = 2+3 = 5$$

后验：$p\mid\text{数据} \sim \mathrm{Beta}(9,5)$

**（2）后验均值**：

$$E[p\mid\text{数据}] = \frac{\alpha'}{\alpha'+\beta'} = \frac{9}{9+5} = \frac{9}{14} \approx 0.643$$

**答案**：后验 $\mathrm{Beta}(9,5)$，后验均值 $\boxed{9/14 \approx 0.643}$

---

### C.6.9（Ch.18，MAP 估计 vs MLE）

**题目**：延续 C.6.8 设定（先验 $\mathrm{Beta}(2,2)$，10 次试验 7 次正面），比较 MLE 与 MAP。

**思路**：MLE 只看数据；MAP 利用后验众数公式。

**解**：

**（1）MLE**：

$$\hat{p}_{\mathrm{MLE}} = \frac{k}{n} = \frac{7}{10} = 0.7$$

**（2）MAP**：

后验 $\mathrm{Beta}(9,5)$，众数公式：

$$\hat{p}_{\mathrm{MAP}} = \frac{\alpha'-1}{\alpha'+\beta'-2} = \frac{9-1}{9+5-2} = \frac{8}{12} = \frac{2}{3} \approx 0.667$$

**（3）比较**：MAP 更接近 0.5，因为先验 $\mathrm{Beta}(2,2)$ 以 0.5 为中心，对估计施加了向均值收缩的正则化效果。

**答案**：$\hat{p}_{\mathrm{MLE}}=0.7$，$\hat{p}_{\mathrm{MAP}}=\boxed{2/3\approx0.667}$，MAP 更接近 0.5。

---

### C.6.10（Ch.18，可信区间 vs 置信区间）

**题目**：辨析贝叶斯可信区间与频率置信区间的含义。

**思路**：两者在概率的主体上不同：前者对参数，后者对区间（随机量）。

**解**：

**（1）贝叶斯 95% 可信区间**：给定观测数据，参数 $\theta$ 落在该区间内的后验概率为 95%，即 $P(\theta\in[l,u]\mid\text{data})=0.95$。

**（2）频率 95% 置信区间**：若重复进行大量相同实验并每次计算置信区间，约 95% 的区间会覆盖真参数；对单次实验，参数要么在该区间内要么不在，不存在"概率 95%"一说。

**（3）正确说法**：选 **（B）**——"若重复实验，95% 的区间会覆盖真参数"。

说法（A）混淆了贝叶斯和频率框架，在频率论中参数是固定常数，不能赋予概率。

**答案**：$\boxed{(B)}$

---

## D 中等题详解（14 题）

### D.6.1（Ch.16，矩估计法）

**题目**：$X_i\overset{iid}{\sim}\mathrm{Gamma}(\alpha,\beta)$，均值 $\alpha/\beta$，方差 $\alpha/\beta^2$，分别用两种矩方程组求 $\hat\alpha,\hat\beta$。

**思路**：(a) 用一、二阶矩；(b) 用均值和样本方差，更简洁。

**解**：

**(a) 用一阶矩和二阶矩**

总体矩：$\mu_1 = \alpha/\beta$，$\mu_2 = E[X^2] = \mathrm{Var}(X)+(\mu_1)^2 = \alpha/\beta^2 + \alpha^2/\beta^2 = \alpha(\alpha+1)/\beta^2$

令样本矩等于总体矩：

$$\bar{X} = \frac{\alpha}{\beta}, \quad \overline{X^2} = \frac{\alpha(\alpha+1)}{\beta^2}$$

由第一式 $\beta = \alpha/\bar{X}$，代入第二式：

$$\overline{X^2} = \frac{\alpha(\alpha+1)\bar{X}^2}{\alpha^2} = \frac{(\alpha+1)\bar{X}^2}{\alpha}$$

解得：

$$\hat{\alpha} = \frac{\bar{X}^2}{\overline{X^2}-\bar{X}^2}, \quad \hat{\beta} = \frac{\bar{X}}{\overline{X^2}-\bar{X}^2}$$

**(b) 用均值和样本方差**

令 $\bar{X}=\alpha/\beta$，$S^2=\alpha/\beta^2$：

由 $S^2/\bar{X} = ({\alpha}/{\beta^2})\cdot({\beta}/{\alpha}) = 1/\beta$，故

$$\hat{\beta} = \frac{\bar{X}}{S^2}, \quad \hat{\alpha} = \bar{X}\cdot\hat{\beta} = \frac{\bar{X}^2}{S^2}$$

**(c) 比较两种矩估计**

注意 $\overline{X^2}-\bar{X}^2$ 并非无偏的样本方差（分母为 $n$ 而非 $n-1$），故 (a) 与 (b) 略有差异。矩估计对矩条件选取不唯一，选用不同的总体矩会导致不同的估计量，这体现了矩估计的非唯一性。通常选用方差（有简洁形式）为宜。

**答案**：

$$\boxed{\hat{\alpha} = \frac{\bar{X}^2}{S^2},\quad \hat{\beta} = \frac{\bar{X}}{S^2}}$$（方法 b 更简洁）

---

### D.6.2（Ch.16，最大似然估计）

**题目**：$X_i\overset{iid}{\sim}\mathrm{Uniform}(0,\theta)$，求 $\hat\theta_{\mathrm{MLE}}$，分析偏差并构造无偏修正量。

**思路**：均匀分布 MLE 不能靠求导（似然函数在约束边界取极值），须直接分析似然函数单调性。

**解**：

**(a) 似然函数**

$$L(\theta) = \prod_{i=1}^n \frac{1}{\theta}\cdot\mathbf{1}(0\leq x_i\leq\theta) = \frac{1}{\theta^n}\cdot\mathbf{1}(\theta\geq x_{(n)})$$

其中 $x_{(n)}=\max_i x_i$ 是最大顺序统计量。

**(b) 求 MLE**

$L(\theta)=1/\theta^n$ 关于 $\theta$ 单调递减，故在约束 $\theta\geq x_{(n)}$ 下，$L(\theta)$ 在 $\theta=x_{(n)}$ 处取得最大值：

$$\hat{\theta}_{\mathrm{MLE}} = X_{(n)}$$

**(c) 偏差与无偏修正**

$X_{(n)}$ 的分布：$F_{X_{(n)}}(t)=(t/\theta)^n$（$0\leq t\leq\theta$），密度 $f_{X_{(n)}}(t)=nt^{n-1}/\theta^n$。

期望：

$$E[X_{(n)}] = \int_0^\theta t\cdot\frac{nt^{n-1}}{\theta^n}dt = \frac{n}{\theta^n}\cdot\frac{\theta^{n+1}}{n+1} = \frac{n}{n+1}\theta$$

偏差：$E[X_{(n)}]-\theta = -\theta/(n+1) < 0$，$X_{(n)}$ 系统偏小。

无偏修正：

$$\hat{\theta}^* = \frac{n+1}{n}X_{(n)}$$

验证：$E[\hat\theta^*]= \frac{n+1}{n}\cdot\frac{n}{n+1}\theta=\theta$。✓

**答案**：$\hat\theta_{\mathrm{MLE}}=X_{(n)}$（有偏），无偏修正量 $\boxed{\hat\theta^*=\dfrac{n+1}{n}X_{(n)}}$

---

### D.6.3（Ch.16，MLE 的不变性）

**题目**：$X_i\overset{iid}{\sim}N(\mu,\sigma^2)$，求 MLE，利用不变性求 $P(X\leq c)$ 的 MLE，分析 $\sigma$ 的 MLE 渐近分布。

**思路**：正态分布 MLE 标准结果；不变性：$\hat{g(\theta)}=g(\hat\theta)$。

**解**：

**(a) $\mu,\sigma^2$ 的 MLE**

对数似然：

$$\ell(\mu,\sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum(X_i-\mu)^2$$

对 $\mu$ 求偏导令零：$\hat\mu = \bar{X}$

对 $\sigma^2$ 求偏导令零：$\hat\sigma^2 = \frac{1}{n}\sum(X_i-\bar{X})^2$

> ⚠️ $\hat\sigma^2$ 的分母是 $n$ 不是 $n-1$，故 MLE 是有偏的（低估总体方差）。

**(b) $P(X\leq c)$ 的 MLE**

由 MLE 不变性：

$$\widehat{P(X\leq c)} = \Phi\!\left(\frac{c-\hat\mu}{\hat\sigma}\right) = \Phi\!\left(\frac{c-\bar{X}}{\hat\sigma}\right)$$

其中 $\hat\sigma=\sqrt{\hat\sigma^2}$。

**(c) $\sigma^2$ 的 MLE 渐近分布与 CRB**

Fisher 信息量（单个样本，关于 $\sigma^2$）：

$$I_1(\sigma^2) = \frac{1}{2\sigma^4}$$

故 $n$ 个样本时 $I(\sigma^2)=n/(2\sigma^4)$，CRB 为 $2\sigma^4/n$。

由 MLE 渐近理论：

$$\sqrt{n}(\hat\sigma^2-\sigma^2) \xrightarrow{d} N(0,2\sigma^4)$$

即 $\hat\sigma^2$ 渐近有效（渐近方差达到 CRB）。

**答案**：$\hat\mu=\bar{X}$，$\hat\sigma^2=\frac{1}{n}\sum(X_i-\bar{X})^2$，$\widehat{P(X\leq c)}=\Phi\!\left(\frac{c-\bar{X}}{\hat\sigma}\right)$，渐近方差 $\boxed{2\sigma^4/n}$

---

### D.6.4（Ch.16，Cramér-Rao 下界）

**题目**：$X_i\overset{iid}{\sim}\mathrm{Poisson}(\lambda)$，验证 $\bar{X}$ 达到 CRB，故为有效估计量。

**思路**：计算单样本 Fisher 信息 → CRB → 与 $\mathrm{Var}(\bar{X})$ 比较。

**解**：

**(a) 单个样本 Fisher 信息量**

$$\log f(x;\lambda) = x\log\lambda - \lambda - \log(x!)$$

$$\frac{\partial\log f}{\partial\lambda} = \frac{x}{\lambda}-1, \quad \frac{\partial^2\log f}{\partial\lambda^2} = -\frac{x}{\lambda^2}$$

$$I_1(\lambda) = -E\!\left[\frac{\partial^2\log f}{\partial\lambda^2}\right] = \frac{E[X]}{\lambda^2} = \frac{\lambda}{\lambda^2} = \frac{1}{\lambda}$$

**(b) CRB**

$n$ 个独立样本，Fisher 信息 $I(\lambda)=n/\lambda$。$\lambda$ 的任意无偏估计量满足：

$$\mathrm{Var}(\hat\lambda) \geq \frac{1}{I(\lambda)} = \frac{\lambda}{n}$$

**(c) $\bar{X}$ 是有效估计量**

$\bar{X}$ 是 $\lambda$ 的无偏估计（$E[\bar{X}]=\lambda$），且由独立性：

$$\mathrm{Var}(\bar{X}) = \frac{\mathrm{Var}(X_1)}{n} = \frac{\lambda}{n}$$

恰好等于 CRB，因此 $\bar{X}$ 是有效估计量（UMVUE）。

**答案**：CRB $=\lambda/n$，$\mathrm{Var}(\bar{X})=\lambda/n$，两者相等，故 $\bar{X}$ 有效。$\boxed{\bar{X}\ \text{达到 CRB，为有效估计量}}$

---

### D.6.5（Ch.17，单正态均值的区间估计）

**题目**：$n=25$，$\bar{X}=12.3$，$S=2.4$，$\sigma$ 未知。(a) $t$ 分布 CI；(b) 已知 $\sigma=2.5$ 时 $z$ 分布 CI；(c) 讨论宽度随 $n$ 的变化。

**思路**：$\sigma$ 未知用 $t(n-1)$；$\sigma$ 已知用 $z$；宽度正比于 $1/\sqrt{n}$。

**解**：

**(a) $\sigma$ 未知，$t(24)$ CI**

$$\bar{X} \pm t_{0.025}(24)\cdot\frac{S}{\sqrt{n}} = 12.3 \pm 2.064\times\frac{2.4}{\sqrt{25}} = 12.3 \pm 2.064\times0.48 = 12.3 \pm 0.991$$

$$\Rightarrow [11.309,\ 13.291]$$

**(b) $\sigma=2.5$ 已知，$z$ CI**

$$\bar{X} \pm z_{0.025}\cdot\frac{\sigma}{\sqrt{n}} = 12.3 \pm 1.96\times\frac{2.5}{5} = 12.3 \pm 1.96\times0.5 = 12.3 \pm 0.98$$

$$\Rightarrow [11.32,\ 13.28]$$

$z$ CI 宽度 $1.96$，$t$ CI 宽度 $1.982$，$\sigma$ 已知时略窄（因为 $t$ 分布比标准正态有更厚的尾部）。

**(c) 宽度随 $n$ 的变化**

CI 宽度 $\propto 1/\sqrt{n}$。若要宽度减半，需使 $1/\sqrt{n}$ 缩减为原来的 $1/2$，即 $n$ 增加至原来的 $\mathbf{4}$ 倍。

**答案**：(a) $[11.31, 13.29]$；(b) $[11.32, 13.28]$；(c) 宽度 $\propto 1/\sqrt{n}$，宽度减半需 $\boxed{4}$ 倍样本量。

---

### D.6.6（Ch.17，方差的区间估计）

**题目**：$n=10$，$S^2=4.5$，均值未知，构造 $\sigma^2$ 和 $\sigma$ 的 95% CI，并解释为何不对称。

**思路**：枢轴量 $(n-1)S^2/\sigma^2\sim\chi^2(n-1)$，$\chi^2$ 分布不对称。

**解**：

**(a) $\sigma^2$ 的 95% CI**

枢轴量：$Q=(n-1)S^2/\sigma^2\sim\chi^2(9)$

$$P\!\left(\chi^2_{0.975}(9)\leq Q\leq\chi^2_{0.025}(9)\right)=0.95$$

注意：$\chi^2_{0.975}(9)=2.70$（左分位数），$\chi^2_{0.025}(9)=19.02$（右分位数）。

解出 $\sigma^2$：

$$\left[\frac{(n-1)S^2}{\chi^2_{0.025}(9)},\ \frac{(n-1)S^2}{\chi^2_{0.975}(9)}\right] = \left[\frac{9\times4.5}{19.02},\ \frac{9\times4.5}{2.70}\right] = \left[\frac{40.5}{19.02},\ \frac{40.5}{2.70}\right]$$

$$= [2.13,\ 15.00]$$

**(b) $\sigma$ 的 95% CI**

对两端开根号：$[\sqrt{2.13},\ \sqrt{15.00}] = [1.46,\ 3.87]$

**(c) 不对称原因**

$\chi^2$ 分布本身是非对称的（偏右），其两个分位数关于均值不对称，故方差 CI 两侧到点估计的距离不同。相比之下，均值 CI 的枢轴量服从（对称的）$t$ 分布，故 CI 关于 $\bar{X}$ 对称。

**答案**：$\sigma^2$ 的 95% CI $=\boxed{[2.13,\ 15.00]}$，$\sigma$ 的 CI $=[1.46,\ 3.87]$。

---

### D.6.7（Ch.17，两正态均值差的区间估计）

**题目**：两独立样本（等方差），构造 $\mu_1-\mu_2$ 的 95% CI。

**思路**：合并方差 $S_p^2$ → 枢轴量服从 $t(n_1+n_2-2)$。

**解**：

**(a) 合并方差**

$$S_p^2 = \frac{(n_1-1)S_1^2+(n_2-1)S_2^2}{n_1+n_2-2} = \frac{9\times4+11\times5}{20} = \frac{36+55}{20} = \frac{91}{20} = 4.55$$

**(b) 95% CI**

枢轴量：

$$T = \frac{(\bar{X}_1-\bar{X}_2)-(\mu_1-\mu_2)}{S_p\sqrt{1/n_1+1/n_2}} \sim t(20)$$

$$S_p = \sqrt{4.55} \approx 2.133, \quad S_p\sqrt{1/10+1/12} = 2.133\sqrt{0.1+0.0833} = 2.133\times0.4282 \approx 0.913$$

CI：

$$(\bar{X}_1-\bar{X}_2)\pm t_{0.025}(20)\cdot S_p\sqrt{\frac{1}{n_1}+\frac{1}{n_2}} = (15-13)\pm2.086\times0.913 = 2\pm1.904$$

$$= [0.096,\ 3.904]$$

**(c) Welch-Satterthwaite 近似（方差不等时）**

不假设方差相等时，近似自由度：

$$\nu = \frac{\left(\frac{S_1^2}{n_1}+\frac{S_2^2}{n_2}\right)^2}{\frac{(S_1^2/n_1)^2}{n_1-1}+\frac{(S_2^2/n_2)^2}{n_2-1}}$$

统计量 $T_W=\frac{(\bar{X}_1-\bar{X}_2)-\delta}{\sqrt{S_1^2/n_1+S_2^2/n_2}}\approx t(\nu)$（不计算具体值）。

**答案**：$S_p^2=4.55$，$\mu_1-\mu_2$ 的 95% CI $=\boxed{[0.10,\ 3.90]}$（约）

---

### D.6.8（Ch.18，先验与后验）

**题目**：$\theta\sim\mathrm{Beta}(2,2)$，$n=10$，$k=7$，求后验均值、MAP、95% 可信区间端点。

**思路**：Beta-Binomial 共轭 → 后验 $\mathrm{Beta}(9,5)$ → 各统计量。

**解**：

**(a) 似然**

$$L(\theta)\propto\theta^7(1-\theta)^3$$

**(b) 后验**

先验 $\mathrm{Beta}(2,2)$，似然贡献 $\theta^7(1-\theta)^3$：

$$\text{后验} \propto \theta^{2+7-1}(1-\theta)^{2+3-1} = \theta^8(1-\theta)^4$$

即后验为 $\mathrm{Beta}(9,5)$。

**(c) 各统计量**

| 量 | 公式 | 数值 |
|---|---|---|
| 后验均值 | $\alpha'/(\alpha'+\beta')$ | $9/14\approx0.643$ |
| 后验众数（MAP） | $(\alpha'-1)/(\alpha'+\beta'-2)$ | $8/12=2/3\approx0.667$ |
| 95% 可信区间 | $[B^{-1}(0.025;9,5),\,B^{-1}(0.975;9,5)]$ | （查 Beta 分位数表） |

其中 $B^{-1}(p;\alpha,\beta)$ 为 $\mathrm{Beta}(\alpha,\beta)$ 的 $p$ 分位数。

**答案**：后验均值 $9/14$，MAP $=\boxed{2/3}$，95% 可信区间端点为 $\mathrm{Beta}(9,5)$ 的 2.5% 和 97.5% 分位数。

---

### D.6.9（Ch.18，共轭先验族——Gamma-Poisson）

**题目**：$X_i\overset{iid}{\sim}\mathrm{Poisson}(\lambda)$，先验 $\lambda\sim\mathrm{Gamma}(\alpha,\beta)$，推导后验并解释后验均值的加权平均含义。

**思路**：直接相乘，识别核为 Gamma 密度形式。

**解**：

**(a) 似然**

$$L(\lambda\mid\mathbf{x}) \propto \prod_{i=1}^n \frac{e^{-\lambda}\lambda^{x_i}}{x_i!} \propto \lambda^{\sum x_i}e^{-n\lambda}$$

**(b) 后验推导**

先验密度 $\pi(\lambda)\propto\lambda^{\alpha-1}e^{-\beta\lambda}$，与似然相乘：

$$p(\lambda\mid\mathbf{x}) \propto \lambda^{\sum x_i}e^{-n\lambda}\cdot\lambda^{\alpha-1}e^{-\beta\lambda} = \lambda^{(\alpha+\sum x_i)-1}e^{-(\beta+n)\lambda}$$

识别为 $\mathrm{Gamma}(\alpha+\sum x_i,\;\beta+n)$。

**(c) 后验均值及加权平均解释**

后验均值：

$$E[\lambda\mid\mathbf{x}] = \frac{\alpha+\sum x_i}{\beta+n}$$

改写为加权平均形式，令 $w=n/(\beta+n)$：

$$E[\lambda\mid\mathbf{x}] = \frac{\beta}{\beta+n}\cdot\frac{\alpha}{\beta} + \frac{n}{\beta+n}\cdot\frac{\sum x_i}{n} = (1-w)\cdot\underbrace{\frac{\alpha}{\beta}}_{\text{先验均值}} + w\cdot\underbrace{\bar{X}}_{\text{MLE}}$$

当 $n\to\infty$ 时 $w\to1$，后验均值 $\to$ MLE；当 $\beta\to\infty$（先验极强）时 $w\to0$，后验均值 $\to$ 先验均值。

**答案**：后验 $\lambda\mid\mathbf{x}\sim\mathrm{Gamma}(\alpha+\sum x_i,\,\beta+n)$，后验均值 $=\boxed{(\alpha+\sum x_i)/(\beta+n)}$，为先验均值与 MLE 的加权平均。

---

### D.6.10（Ch.16，正则化与 MAP 的联系）

**题目**：线性回归中，先验 $\boldsymbol\beta\sim N(\mathbf{0},\tau^2\mathbf{I})$，噪声 $\sigma^2$ 已知，推导 MAP 等价于 Ridge，并比较 Ridge 与 Lasso 的先验。

**思路**：取 MAP 的对数，整理成正则化最小二乘形式。

**解**：

**(a) MAP 等价于 Ridge 回归**

设 $\mathbf{y}=X\boldsymbol\beta+\boldsymbol\varepsilon$，$\boldsymbol\varepsilon\sim N(\mathbf{0},\sigma^2\mathbf{I})$，则：

$$\log L(\boldsymbol\beta) = -\frac{1}{2\sigma^2}\|\mathbf{y}-X\boldsymbol\beta\|^2 + \text{const}$$

$$\log\pi(\boldsymbol\beta) = -\frac{1}{2\tau^2}\|\boldsymbol\beta\|^2 + \text{const}$$

MAP：最大化 $\log L+\log\pi$，等价于最小化：

$$\frac{1}{2\sigma^2}\|\mathbf{y}-X\boldsymbol\beta\|^2 + \frac{1}{2\tau^2}\|\boldsymbol\beta\|^2$$

即等价于 Ridge 回归（正则化参数 $\lambda=\sigma^2/\tau^2$）。

**(b) 贝叶斯解释**

$\tau^2\to\infty$（先验极弱/无信息）$\Rightarrow$ $\lambda=\sigma^2/\tau^2\to0$（无正则化），退化为 OLS。

$\tau^2\to0$（先验极强）$\Rightarrow$ $\lambda\to\infty$（强正则化），估计量收缩至 $\mathbf{0}$。

**(c) Ridge vs Lasso 的先验**

| 方法 | 正则项 | 对应先验 |
|---|---|---|
| Ridge（L2） | $\lambda\|\boldsymbol\beta\|^2$ | $\boldsymbol\beta\sim N(\mathbf{0},\tau^2\mathbf{I})$（正态先验） |
| Lasso（L1） | $\lambda\|\boldsymbol\beta\|_1$ | $\beta_j\overset{iid}{\sim}\mathrm{Laplace}(0,b)$（拉普拉斯先验）|

Laplace 先验在零点有尖峰，促进稀疏解（恰好为零的系数），这是 Lasso 产生稀疏性的贝叶斯解释。

**答案**：MAP 等价于 Ridge，$\lambda=\boxed{\sigma^2/\tau^2}$；Lasso 对应 Laplace 先验。

---

### D.6.11（Ch.17，比例的区间估计）

**题目**：$n=400$，$\hat{p}=0.6$，构造 95% CI（Wald 和 Wilson），并确定最小样本量。

**思路**：Wald 用正态近似；Wilson 用 score 统计量，边界情形更稳健。

**解**：

**(a) Wald 95% CI**

标准误 $SE=\sqrt{\hat{p}(1-\hat{p})/n}=\sqrt{0.6\times0.4/400}=\sqrt{0.0006}=0.02449$

$$\hat{p}\pm z_{0.025}\cdot SE = 0.6\pm1.96\times0.02449 = 0.6\pm0.04800$$

$$= [0.552,\ 0.648]$$

**(b) 最小样本量**（最保守估计 $p=0.5$）

CI 宽度 $\leq0.04$，即半宽 $\leq0.02$：

$$z_{0.025}\sqrt{\frac{p(1-p)}{n}} \leq 0.02 \implies 1.96\sqrt{\frac{0.25}{n}} \leq 0.02$$

$$\sqrt{n}\geq\frac{1.96\times0.5}{0.02}=49 \implies n\geq2401$$

最小样本量 $n^*=2401$。

**(c) Wilson 区间的优势**

Wilson 区间：

$$\frac{\hat{p}+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n}+\frac{z^2}{4n^2}}}{1+z^2/n}$$

当 $\hat{p}$ 接近 0 或 1 时，Wald 区间可能越出 $[0,1]$ 且覆盖率严重不足。Wilson 区间始终在 $[0,1]$ 内，且在小样本和极端比例下仍有近似正确的覆盖率。

**答案**：Wald CI $=[0.552, 0.648]$，最小样本量 $\boxed{n^*=2401}$，Wilson 区间在极端比例时更准确。

---

### D.6.12（Ch.18，贝叶斯预测分布）

**题目**：$X|\theta\sim N(\theta,1)$，先验 $\theta\sim N(0,\tau^2)$，观测 $X=x_0$，求后验和预测分布。

**思路**：Normal-Normal 共轭 → 后验均值为收缩估计 → 预测分布方差 = 后验方差 + 采样方差。

**解**：

**(a) 后验（Normal-Normal 共轭）**

似然：$X|\theta\sim N(\theta,1)$，先验：$\theta\sim N(0,\tau^2)$。

精度（方差倒数）相加：

$$\frac{1}{\sigma_{\text{post}}^2} = \frac{1}{1}+\frac{1}{\tau^2} = 1+\frac{1}{\tau^2} = \frac{\tau^2+1}{\tau^2}$$

$$\sigma_{\text{post}}^2 = \frac{\tau^2}{\tau^2+1}$$

后验均值（精度加权平均）：

$$\mu_{\text{post}} = \sigma_{\text{post}}^2\!\left(\frac{x_0}{1}+\frac{0}{\tau^2}\right) = \frac{\tau^2}{\tau^2+1}\cdot x_0 = \frac{\tau^2 x_0}{\tau^2+1}$$

故后验 $\theta|X=x_0\sim N\!\left(\dfrac{\tau^2 x_0}{\tau^2+1},\,\dfrac{\tau^2}{\tau^2+1}\right)$。

**(b) 预测分布**

$\tilde{X}|\theta\sim N(\theta,1)$，用全期望：

$$E[\tilde{X}|X=x_0] = E[E[\tilde{X}|\theta]|X=x_0] = E[\theta|X=x_0] = \frac{\tau^2 x_0}{\tau^2+1}$$

$$\mathrm{Var}(\tilde{X}|X=x_0) = E[\mathrm{Var}(\tilde{X}|\theta)|X=x_0]+\mathrm{Var}(E[\tilde{X}|\theta]|X=x_0)$$

$$= E[1|X=x_0]+\mathrm{Var}(\theta|X=x_0) = 1+\frac{\tau^2}{\tau^2+1} = \frac{2\tau^2+1}{\tau^2+1}$$

预测分布：$\tilde{X}|X=x_0\sim N\!\left(\dfrac{\tau^2 x_0}{\tau^2+1},\,\dfrac{2\tau^2+1}{\tau^2+1}\right)$。

**(c) 预测方差 > 后验方差**

后验方差 $=\tau^2/(\tau^2+1)$，预测方差 $=(2\tau^2+1)/(\tau^2+1)$，差值恰为 $1$（即 $\tilde{X}$ 的采样方差）。预测方差更大，因为预测 $\tilde{X}$ 时除了参数 $\theta$ 的不确定性（后验方差），还有额外的观测噪声（采样不确定性）。

**答案**：后验 $\theta|x_0\sim N\!\left(\frac{\tau^2 x_0}{\tau^2+1},\frac{\tau^2}{\tau^2+1}\right)$，预测分布方差 $=\boxed{\dfrac{2\tau^2+1}{\tau^2+1}}$。

---

### D.6.13（Ch.16，EM 算法思想）

**题目**：混合高斯模型 $\pi N(\mu_1,1)+(1-\pi)N(\mu_2,1)$，推导 EM 算法的 E 步和 M 步。

**思路**：引入潜变量 $Z_i$ 表示样本所属组分，E 步计算软分配，M 步加权最大化。

**解**：

**(a) 完整数据对数似然**

引入 $Z_i\in\{1,2\}$（$Z_i=k$ 表示 $x_i$ 来自第 $k$ 个分量），令 $\gamma_{ik}=\mathbf{1}(Z_i=k)$：

$$\ell_c(\theta) = \sum_{i=1}^n\sum_{k=1}^2 \gamma_{ik}\left[\log\pi_k + \log\phi(x_i;\mu_k,1)\right]$$

其中 $\pi_1=\pi$，$\pi_2=1-\pi$，$\phi$ 为正态密度，$\theta=(\pi,\mu_1,\mu_2)$。

**(b) E 步：计算软分配责任**

$$r_{ik} = P(Z_i=k\mid x_i,\theta^{(t)}) = \frac{\pi_k^{(t)}\phi(x_i;\mu_k^{(t)},1)}{\sum_{j=1}^2\pi_j^{(t)}\phi(x_i;\mu_j^{(t)},1)}$$

$r_{ik}$ 是"样本 $i$ 属于第 $k$ 组分"的后验概率（软分配）。

**(c) M 步：最大化 $Q(\theta|\theta^{(t)})$**

$$Q = E[\ell_c\mid\mathbf{x},\theta^{(t)}] = \sum_{i,k}r_{ik}\left[\log\pi_k+\log\phi(x_i;\mu_k,1)\right]$$

对 $\mu_k$ 求偏导令零：

$$\frac{\partial Q}{\partial\mu_k} = \sum_i r_{ik}(x_i-\mu_k) = 0 \implies \hat\mu_k^{(t+1)} = \frac{\sum_i r_{ik}x_i}{\sum_i r_{ik}}$$

对 $\pi$ 用 Lagrange 乘子（约束 $\pi_1+\pi_2=1$）：

$$\hat\pi^{(t+1)} = \frac{1}{n}\sum_i r_{i1}$$

**答案**：E 步软分配 $r_{ik}=\pi_k\phi(x_i;\mu_k)/\sum_j\pi_j\phi(x_i;\mu_j)$，M 步更新 $\boxed{\hat\mu_k=\sum_i r_{ik}x_i/\sum_i r_{ik}}$（加权均值）。

---

### D.6.14（Ch.18，Jeffreys 先验）

**题目**：$X\sim\mathrm{Bernoulli}(\theta)$，推导 Jeffreys 先验，并说明重参数不变性。

**思路**：Fisher 信息 $I(\theta)=1/(\theta(1-\theta))$，Jeffreys 先验 $\propto\sqrt{I(\theta)}$。

**解**：

**(a) 计算 Fisher 信息，推导 Jeffreys 先验**

$$\log f(x;\theta) = x\log\theta+(1-x)\log(1-\theta)$$

$$\frac{\partial^2\log f}{\partial\theta^2} = -\frac{x}{\theta^2}-\frac{1-x}{(1-\theta)^2}$$

$$I(\theta) = -E\!\left[\frac{\partial^2\log f}{\partial\theta^2}\right] = \frac{E[X]}{\theta^2}+\frac{1-E[X]}{(1-\theta)^2} = \frac{\theta}{\theta^2}+\frac{1-\theta}{(1-\theta)^2} = \frac{1}{\theta}+\frac{1}{1-\theta} = \frac{1}{\theta(1-\theta)}$$

Jeffreys 先验：

$$\pi_J(\theta) \propto \sqrt{I(\theta)} = \frac{1}{\sqrt{\theta(1-\theta)}} = \theta^{-1/2}(1-\theta)^{-1/2}$$

**(b) 识别为 $\mathrm{Beta}(1/2,1/2)$**

对比 $\mathrm{Beta}(\alpha,\beta)$ 密度 $\propto\theta^{\alpha-1}(1-\theta)^{\beta-1}$：

$$\pi_J(\theta)\propto\theta^{1/2-1}(1-\theta)^{1/2-1}$$

故 $\pi_J=\mathrm{Beta}(1/2,1/2)$，而均匀先验为 $\mathrm{Beta}(1,1)$。$\mathrm{Beta}(1/2,1/2)$ 在 0 和 1 两端有奇点（U 形），对极端概率赋予更多权重，反映对 $\theta$ 近于 0 或 1 时的额外不确定性。

**(c) 重参数不变性**

设 $\phi=g(\theta)$ 是单调变换，由变量变换公式，$\phi$ 的 Fisher 信息为 $I_\phi(\phi)=I(\theta)/[g'(\theta)]^2$，相应的 Jeffreys 先验 $\pi_J(\phi)\propto\sqrt{I_\phi(\phi)}$。

可以验证：通过 $\theta$ 的 Jeffreys 先验变换到 $\phi$ 所得到的密度，与直接在 $\phi$ 的模型上应用 Jeffreys 公式所得结果完全一致。这保证了 Jeffreys 先验在参数变换下具有一致性（参数化方式不影响分析结论）。

**答案**：$\pi_J(\theta)\propto[\theta(1-\theta)]^{-1/2}=\mathrm{Beta}(1/2,1/2)$，$\boxed{\text{Jeffreys 先验具有重参数不变性}}$。

---

## E 提高题详解（10 题）

### E.6.1（Ch.16，MLE + Fisher 信息 + 渐近理论 + 完整证明）

**题目**：$X_i\overset{iid}{\sim}p(x;\theta)$（指数族），证明 MLE 的一致性、渐近正态性、渐近有效性；以指数分布为例验证。

**思路**：得分方程 → Taylor 展开 → CLT + LLN → 建立渐近分布；具体计算指数分布。

**解**：

**(a) MLE 满足得分方程**

设对数似然 $\ell_n(\theta)=\sum_{i=1}^n\log p(X_i;\theta)$。MLE $\hat\theta_n$ 是 $\ell_n$ 的极大值点，在正则条件（内点极大、似然可微）下，必满足一阶条件：

$$\frac{\partial\ell_n}{\partial\theta}\bigg|_{\hat\theta_n} = \sum_{i=1}^n\frac{\partial\log p(X_i;\hat\theta_n)}{\partial\theta} = 0$$

这即为得分方程（score equation）。

**(b) MLE 渐近正态性的 Taylor 展开证明**

将得分方程在真值 $\theta_0$ 处 Taylor 展开：

$$0 = \underbrace{\ell_n'(\theta_0)}_{\sum_i s(X_i;\theta_0)} + \ell_n''(\theta_0)(\hat\theta_n-\theta_0) + O\!\left((\hat\theta_n-\theta_0)^2\right)$$

其中 $s(X_i;\theta)=\partial\log p(X_i;\theta)/\partial\theta$ 为得分函数。

由 LLN：$n^{-1}\ell_n''(\theta_0)\xrightarrow{p}-\mathcal{I}(\theta_0)$（$\mathcal{I}$ 为 Fisher 信息）。

由 CLT：$n^{-1/2}\ell_n'(\theta_0)=n^{-1/2}\sum_i s(X_i;\theta_0)\xrightarrow{d}N(0,\mathcal{I}(\theta_0))$（因 $E[s]=0$，$\mathrm{Var}(s)=\mathcal{I}(\theta_0)$）。

整理：

$$\sqrt{n}(\hat\theta_n-\theta_0) \approx \frac{n^{-1/2}\ell_n'(\theta_0)}{-n^{-1}\ell_n''(\theta_0)} \xrightarrow{d} \frac{N(0,\mathcal{I}(\theta_0))}{\mathcal{I}(\theta_0)} = N(0,\mathcal{I}(\theta_0)^{-1})$$

**(c) MLE 渐近有效（CRB 紧）**

Cramér-Rao 下界：任意无偏估计量 $T_n$ 满足 $\mathrm{Var}(\sqrt{n}\,T_n)\geq\mathcal{I}(\theta_0)^{-1}$。

MLE 的渐近方差 $=\mathcal{I}(\theta_0)^{-1}$，恰好达到下界，故 MLE 是渐近有效估计量。

**(d) 指数分布验证**

$p(x;\theta)=\theta e^{-\theta x}$（$x>0$），设样本量 $n$。

**MLE**：$\ell(\theta)=n\log\theta-\theta\sum x_i$，令 $\ell'=n/\theta-\sum x_i=0$，得 $\hat\theta=1/\bar{X}$。

**Fisher 信息**（单样本）：$\ell''(\theta)=-n/\theta^2$，故 $I_1(\theta)=1/\theta^2$；$n$ 个样本 $\mathcal{I}(\theta)=n/\theta^2$。

**渐近分布**：$\sqrt{n}(\hat\theta-\theta)\xrightarrow{d}N(0,\theta^2)$（即渐近方差 $=\theta^2/n$）。

**渐近置信区间**（$1-\alpha$ 水平）：

$$\hat\theta\pm z_{\alpha/2}\cdot\frac{\hat\theta}{\sqrt{n}}$$

验证：$\mathrm{Var}(\hat\theta)\approx\theta^2/n=1/\mathcal{I}(\theta)$，与 CRB 吻合。

**答案**：MLE $\hat\theta=1/\bar{X}$，渐近方差 $=\boxed{\theta^2/n}$，达到 CRB，渐近有效。

---

### E.6.2（Ch.16+Ch.18，正则化 MLE + MAP + 惩罚似然）

**题目**：(a) 正态先验 → Ridge；(b) Laplace 先验 → Lasso；(c) Dropout → 隐式先验；(d) $n\to\infty$ 时 MAP → MLE。

**思路**：MAP = 最大化对数似然 + 对数先验；各先验形式对应不同正则项。

**解**：

**(a) 正态先验 → L2 正则化（Ridge）**

先验 $\pi(\boldsymbol\theta)=N(\mathbf{0},\tau^2\mathbf{I})$，对数先验：

$$\log\pi(\boldsymbol\theta) = -\frac{1}{2\tau^2}\|\boldsymbol\theta\|^2 + \text{const}$$

MAP：

$$\hat{\boldsymbol\theta}_{MAP} = \arg\max_{\boldsymbol\theta}\left[\log L(\boldsymbol\theta)-\frac{1}{2\tau^2}\|\boldsymbol\theta\|^2\right]$$

等价于最小化：

$$-\log L(\boldsymbol\theta) + \frac{1}{2\tau^2}\|\boldsymbol\theta\|^2 = -\log L(\boldsymbol\theta) + \frac{\lambda}{2}\|\boldsymbol\theta\|^2, \quad\lambda=\frac{1}{\tau^2}$$

此即 L2 正则化（Ridge 回归中为 $\|\mathbf{y}-X\boldsymbol\theta\|^2+\lambda\|\boldsymbol\theta\|^2$）。$\square$

**(b) Laplace 先验 → L1 正则化（Lasso）**

先验 $\pi(\theta_j)=\frac{1}{2b}\exp(-|\theta_j|/b)$（Laplace），对数先验：

$$\log\pi(\boldsymbol\theta) = -\frac{1}{b}\|\boldsymbol\theta\|_1 + \text{const}$$

MAP 等价于最小化 $-\log L(\boldsymbol\theta)+\lambda\|\boldsymbol\theta\|_1$（$\lambda=1/b$），即 L1 正则化（Lasso）。Laplace 分布在零点的尖峰促进系数恰好为零（稀疏解）。$\square$

**(c) Dropout ≈ 隐式正则化（直觉推导）**

Dropout 以概率 $p$ 随机将权重置零，训练时对 $2^d$ 个子网络（$d$ 为权重维数）求期望。Srivastava 等人（2014）及 Gal & Ghahramani（2016）证明，对某些网络结构，Dropout 等价于在权重上施加 Bernoulli 先验（每个权重独立地有概率 $p$ 被"去除"），从而 MAP 估计等价于带 Bernoulli 先验的正则化。这给出 Dropout 防止过拟合的贝叶斯解释。

**(d) 大样本时 MAP → MLE（后验收缩）**

MAP 目标：$\hat{\boldsymbol\theta}_{MAP}=\arg\max[\ell_n(\boldsymbol\theta)+\log\pi(\boldsymbol\theta)]$。

当 $n\to\infty$ 时，$\ell_n(\boldsymbol\theta)=O(n)$ 而 $\log\pi(\boldsymbol\theta)=O(1)$（固定先验），故先验贡献相对可忽略：

$$\hat{\boldsymbol\theta}_{MAP} = \arg\max\!\left[\ell_n(\boldsymbol\theta)+\underbrace{\log\pi(\boldsymbol\theta)}_{\text{相对}O(1)}\right] \xrightarrow{n\to\infty} \arg\max\,\ell_n(\boldsymbol\theta) = \hat{\boldsymbol\theta}_{MLE}$$

小样本时先验相对贡献大（正则化效果显著）；大样本时数据主导，先验被"冲淡"。

**答案**：正态先验 $\leftrightarrow$ Ridge，Laplace 先验 $\leftrightarrow$ Lasso，$n\to\infty$ 时 $\hat{\boldsymbol\theta}_{MAP}\to\hat{\boldsymbol\theta}_{MLE}$，$\boxed{\text{先验在大样本时影响消失}}$。

---

### E.6.3（Ch.16，矩估计 + GMM + 弱工具变量）

**题目**：GMM 框架，$m$ 个矩条件，$k$ 个参数，讨论恰好识别、过识别、最优权重及 J 检验。

**思路**：GMM 统一了矩估计（$m=k$）和过识别系统（$m>k$），最优权重使渐近方差最小。

**解**：

**(a) $m=k$ 时退化为矩估计——相合性与渐近正态性**

当 $m=k$，矩估计 $\hat\theta_n$ 满足 $\hat{g}_n(\hat\theta_n)=\mathbf{0}$（精确矩方程）。

**相合性**：由 LLN，$\hat{g}_n(\theta)\xrightarrow{p}g(\theta)=E[g(X;\theta)]$；在正则条件下 $g(\theta_0)=\mathbf{0}$ 且 $g$ 在 $\theta_0$ 邻域可识别，由连续性 $\hat\theta_n\xrightarrow{p}\theta_0$。

**渐近正态性**：在 $\theta_0$ 处 Taylor 展开：$\hat{g}_n(\hat\theta_n)\approx\hat{g}_n(\theta_0)+G_n(\hat\theta_n-\theta_0)=\mathbf{0}$，其中 $G_n=\partial\hat{g}_n/\partial\theta^\top\xrightarrow{p}G=\partial g(\theta_0)/\partial\theta^\top$。

由 CLT：$\sqrt{n}\,\hat{g}_n(\theta_0)\xrightarrow{d}N(\mathbf{0},\Omega)$，$\Omega=\mathrm{Var}(g(X;\theta_0))$。

故：

$$\sqrt{n}(\hat\theta_n-\theta_0) \xrightarrow{d} N\!\left(\mathbf{0},\;(G^{-1})\Omega(G^{-1})^\top\right)$$

**(b) $m>k$ 时最优权重 GMM（Hansen-Sargan 效率定理）**

GMM 目标：$Q_n(\theta)=\hat{g}_n(\theta)^\top\mathbf{W}_n\hat{g}_n(\theta)$。

一阶条件：$G^\top\mathbf{W}G\sqrt{n}(\hat\theta-\theta_0)=-G^\top\mathbf{W}\sqrt{n}\,\hat{g}_n(\theta_0)$。

渐近方差：$V(\mathbf{W})=(G^\top\mathbf{W}G)^{-1}G^\top\mathbf{W}\Omega\mathbf{W}G(G^\top\mathbf{W}G)^{-1}$。

取 $\mathbf{W}^*=\Omega^{-1}$（最优权重），由 Cauchy-Schwarz 型不等式，$V(\mathbf{W}^*)$ 最小：

$$V^* = (G^\top\Omega^{-1}G)^{-1}$$

这是最优 GMM 的渐近方差下界（Hansen 1982）。

**(c) 过识别 J 检验**

在最优 GMM 下，过识别检验统计量：

$$J = n\hat{g}_n^\top\mathbf{W}^*\hat{g}_n = n\hat{g}_n^\top\Omega^{-1}\hat{g}_n\xrightarrow{d}\chi^2(m-k)$$

自由度为 $m-k$（超额矩条件数）。若 $J$ 值过大，拒绝矩条件的联合有效性，说明模型设定（工具变量或矩条件）存在问题。

**(d) 与对比学习的联系（直觉）**

InfoNCE 损失：$\mathcal{L}=-\log\frac{e^{f(x_i,x_i^+)/\tau}}{\sum_{j}e^{f(x_i,x_j^-)/\tau}}$，可视为对"正样本对应高相似度"的矩条件的 GMM 估计。批量中负样本数量越多，矩条件 $\hat{g}_n$ 的估计越精确（类似 $n$ 增大），渐近方差减小，估计效率提升。这解释了大批量训练在 SimCLR 中提升效果的统计机制。

**答案**：最优 GMM 渐近方差 $=(G^\top\Omega^{-1}G)^{-1}$，J 检验统计量 $\sim\chi^2(m-k)$，$\boxed{\text{大批量}=\text{更多矩条件信息，提升 GMM 效率}}$。

---

### E.6.4（Ch.17，置信区间 + 枢轴量 + Bootstrap 置信区间）

**题目**：$X_i\overset{iid}{\sim}N(\mu,\sigma^2)$，均值与方差均未知。构造精确 $\mu$ CI 和 $\sigma^2$ CI；Bootstrap；Conformal Prediction。

**思路**：利用充分统计量构造精确枢轴量；Bootstrap 无需分布假设；Conformal 提供有限样本保证。

**解**：

**(a) $\mu$ 的精确 $t$ CI**

枢轴量推导：$\bar{X}\sim N(\mu,\sigma^2/n)$，$(n-1)S^2/\sigma^2\sim\chi^2(n-1)$，两者独立。

$$T = \frac{\bar{X}-\mu}{S/\sqrt{n}} = \frac{(\bar{X}-\mu)/(\sigma/\sqrt{n})}{\sqrt{(n-1)S^2/[\sigma^2(n-1)]}}\sim t(n-1)$$

95% CI：$\bar{X}\pm t_{0.025}(n-1)\cdot S/\sqrt{n}$。

**(b) $\sigma^2$ 的精确 $\chi^2$ CI**

枢轴量：$Q=(n-1)S^2/\sigma^2\sim\chi^2(n-1)$（不依赖 $\mu$，因为 $S^2$ 与 $\bar{X}$ 独立）。

$$P\!\left(\chi^2_{1-\alpha/2}(n-1)\leq\frac{(n-1)S^2}{\sigma^2}\leq\chi^2_{\alpha/2}(n-1)\right)=1-\alpha$$

解出 $\sigma^2$：$\left[\frac{(n-1)S^2}{\chi^2_{\alpha/2}},\;\frac{(n-1)S^2}{\chi^2_{1-\alpha/2}}\right]$。

不对称原因：$\chi^2$ 分布右偏，两端分位数关于中心不对称，故 CI 两侧宽度不等。

**(c) Bootstrap 置信区间**

**百分位数 Bootstrap**：从 $\{X_1,\ldots,X_n\}$ 有放回地重抽样 $B$ 次，每次计算 $\hat\theta^{*(b)}$，取 $[\hat\theta^{*(\alpha/2)},\hat\theta^{*(1-\alpha/2)}]$ 为 CI。二阶渐近正确：覆盖误差为 $O(n^{-1})$。

**BCa Bootstrap**：对百分位数进行偏差（$\hat{z}_0$）和加速（$\hat{a}$）校正：

$$\alpha_1 = \Phi\!\left(\hat{z}_0+\frac{\hat{z}_0+z_{\alpha/2}}{1-\hat{a}(\hat{z}_0+z_{\alpha/2})}\right), \quad \alpha_2 = \Phi\!\left(\hat{z}_0+\frac{\hat{z}_0+z_{1-\alpha/2}}{1-\hat{a}(\hat{z}_0+z_{1-\alpha/2})}\right)$$

BCa 是二阶渐近正确的（覆盖误差 $O(n^{-3/2})$），优于简单百分位数法，在偏斜或参数化有约束时尤为重要。

**(d) Conformal Prediction（分裂式，有限样本保证）**

**构造步骤**：

1. 将 $n$ 个训练样本分为训练集 $\mathcal{D}_1$（$n_1$ 个）和校准集 $\mathcal{D}_2$（$n_2$ 个）。
2. 在 $\mathcal{D}_1$ 上训练模型 $\hat{f}$，对每个校准样本计算非一致性分数 $s_i=|Y_i-\hat{f}(X_i)|$（$i\in\mathcal{D}_2$）。
3. 设 $q=\lceil(1-\alpha)(n_2+1)\rceil/n_2$ 分位数 $\hat{q}$。
4. 预测区间：$C(X_{n+1})=\{y:|y-\hat{f}(X_{n+1})|\leq\hat{q}\}$。

**有限样本保证**：若 $(X_i,Y_i)$ 可交换，则

$$P(Y_{n+1}\in C(X_{n+1})) \geq 1-\alpha$$

严格成立（有限样本，不依赖分布假设）。

与 Bootstrap 对比：Bootstrap 是渐近正确（$n\to\infty$），Conformal 是有限样本精确（任意 $n$），但 Conformal 不依赖模型分布假设，适用范围更广。

**答案**：$\mu$ 的精确 CI 用 $t(n-1)$ 枢轴量；$\sigma^2$ 的 CI 用 $\chi^2(n-1)$ 枢轴量（不对称）；$\boxed{\text{Conformal 具有有限样本覆盖保证}}$，Bootstrap 仅渐近正确。

---

### E.6.5（Ch.17+Ch.18，贝叶斯可信区间 + 频率置信区间 + 概率解释对比）

**题目**：$X|θ\sim B(n,\theta)$，先验 $\theta\sim\mathrm{Beta}(\alpha_0,\beta_0)$，$X=k$。比较后验 HDR、Wald/Wilson CI，深入对比两种框架。

**思路**：后验 Beta → HDR；Wald 在边界失效；频率/贝叶斯框架的哲学对比。

**解**：

**(a) 后验 HDR 可信区间**

后验 $\theta|X=k\sim\mathrm{Beta}(\alpha_0+k,\beta_0+n-k)$。

HDR（最高密度区间）是满足以下条件的最短区间 $[l^*,u^*]$：$P(\theta\in[l^*,u^*]|X=k)=0.95$。

对单峰对称分布，HDR 退化为等尾区间 $[B^{-1}(0.025),B^{-1}(0.975)]$；对偏斜 Beta 分布，HDR 通过数值优化（最小化 $u-l$，约束积分 $=0.95$）得到，通常是非对称区间。

**(b) Wald vs Wilson 区间**

**Wald 区间**：$\hat{p}\pm z_{\alpha/2}\sqrt{\hat{p}(1-\hat{p})/n}$。

**缺陷**：当 $\hat{p}=0$ 或 $1$ 时，区间退化为零宽度 $[0,0]$ 或 $[1,1]$，完全无法覆盖真参数（覆盖率 $=0$）。即使 $\hat{p}$ 只是接近边界，Wald 区间覆盖率也系统性低于名义水平 $1-\alpha$。

**Wilson Score 区间**：基于 $z$ 统计量 $z=({\hat{p}-\theta})/\sqrt{{\theta(1-\theta)/n}}$，解不等式 $|z|\leq z_{\alpha/2}$：

$$\frac{\hat{p}+\frac{z^2}{2n}\pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}}$$

Wilson 区间始终在 $[0,1]$ 内，在 $\hat{p}$ 接近边界及小样本下均有接近名义水平的覆盖率。

**(c) 频率 vs 贝叶斯的哲学含义**

| | 贝叶斯可信区间 | 频率置信区间 |
|---|---|---|
| 概率主体 | 参数 $\theta$（随机变量） | 区间端点（随机量） |
| 解释 | $P(\theta\in[\ell,u]\|X)=0.95$：给定数据，参数在区间内的概率 | $P_\theta(\theta\in[\ell(X),u(X)])=0.95$：重复实验，区间覆盖参数的频率 |
| 对"参数以95%概率落在CI内"的看法 | 对可信区间正确 | 对单次置信区间无意义（$\theta$ 是固定常数）|

对频率论者，"参数在置信区间内的概率"要么是 0 要么是 1（因 $\theta$ 固定），说这个概率是 95% 混淆了参数与估计量的随机性。

**(d) 工程实践中的权衡（A/B 测试场景）**

**场景**：电商 A/B 测试点击率，实验组 $\hat{p}_A=0.05$（$n_A=1000$），对照组 $\hat{p}_B=0.04$（$n_B=1000$）。

- **频率置信区间**：提供严格的频率覆盖保证，适合监管合规；但不能直接给出"实验组更好的概率"。
- **贝叶斯可信区间**：可以直接计算 $P(\theta_A>\theta_B|\text{数据})$，为决策提供直观的概率依据，例如"有 97% 的把握实验组更好"。

工程权衡：对需要严格误报控制的场景（医药试验）用频率方法；对快速迭代的产品决策，贝叶斯可信区间给出更直接的业务语言。

**答案**：后验 HDR 给出 $P(\theta\in[\ell,u]|\text{数据})=0.95$；Wilson CI 优于 Wald CI；$\boxed{\text{贝叶斯框架提供直接决策概率，频率框架提供频率覆盖保证}}$。

---

### E.6.6（Ch.18，分层贝叶斯 + 超先验 + 部分池化）

**题目**：$J$ 组分层模型，推导部分池化估计量，分析极端情形，联系联邦学习。

**思路**：精度加权平均 → 组内精度 $n_j/\sigma^2$ vs 组间精度 $1/\tau^2$ 决定池化程度。

**解**：

**(a) 完整联合分布**

$$p(\{X_{ij}\},\{\theta_j\},\mu,\tau) = \prod_{j=1}^J\prod_{i=1}^{n_j}N(X_{ij};\theta_j,\sigma^2)\cdot\prod_{j=1}^J N(\theta_j;\mu,\tau^2)\cdot N(\mu;0,\Sigma_0)\cdot p(\tau)$$

**(b) 部分池化估计量推导**

给定 $\mu,\tau$，$\theta_j$ 的条件后验：

$$p(\theta_j|\{X_{ij}\},\mu,\tau)\propto\prod_i N(X_{ij};\theta_j,\sigma^2)\cdot N(\theta_j;\mu,\tau^2)$$

精度（方差倒数）相加：$\frac{1}{\sigma_j^2}=\frac{n_j}{\sigma^2}+\frac{1}{\tau^2}$，即 $\sigma_j^2=\frac{\sigma^2\tau^2}{n_j\tau^2+\sigma^2}$。

后验均值（精度加权）：

$$\hat\theta_j = \sigma_j^2\!\left(\frac{n_j\bar{X}_j}{\sigma^2}+\frac{\mu}{\tau^2}\right) = \frac{n_j\tau^2}{n_j\tau^2+\sigma^2}\bar{X}_j+\frac{\sigma^2}{n_j\tau^2+\sigma^2}\mu$$

令 $\lambda_j=\frac{n_j\tau^2}{n_j\tau^2+\sigma^2}$，则

$$\hat\theta_j = \lambda_j\bar{X}_j+(1-\lambda_j)\hat\mu \quad\text{（部分池化）}$$

**(c) 极端情形分析**

- **$\tau\to0$**（组间方差趋零，各组参数相同）：$\lambda_j\to0$，$\hat\theta_j\to\hat\mu$，即完全池化——所有组共享全局均值。
- **$\tau\to\infty$**（各组参数完全独立）：$\lambda_j\to1$，$\hat\theta_j\to\bar{X}_j$，即无池化——各组独立估计。

部分池化在两极端之间自适应，根据数据决定借鉴全局信息的程度。

**(d) 联邦学习联系**

在个性化联邦学习中，客户端 $j$ 的本地参数 $\boldsymbol\theta_j$ 对应 $\theta_j$，全局超先验 $\mu$ 对应服务器维护的全局模型。

- **pFedMe**（Dinh et al. 2020）：每个客户端最小化 $\ell_j(\boldsymbol\theta_j)+\lambda\|\boldsymbol\theta_j-\mathbf{w}\|^2$，正是 Normal-Normal 分层贝叶斯的 MAP 估计（$\lambda=\sigma^2/\tau^2$）。
- **MAML**（Finn et al. 2017）：元学习的初始化参数对应超先验均值 $\mu$，每个客户端的快速适应对应从先验到后验的更新。

超先验的方差 $\tau^2$ 编码了客户端间差异程度：$\tau^2$ 大则允许客户端差异大（个性化强），$\tau^2$ 小则强迫客户端接近全局模型（泛化强）。

**答案**：部分池化 $\hat\theta_j=\lambda_j\bar{X}_j+(1-\lambda_j)\hat\mu$，$\lambda_j=\frac{n_j\tau^2}{n_j\tau^2+\sigma^2}$；$\tau\to0$ 完全池化，$\tau\to\infty$ 无池化；$\boxed{\text{联邦学习个性化}=\text{分层贝叶斯部分池化}}$。

---

### E.6.7（Ch.16+Ch.18，变分推断 + ELBO + VAE）

**题目**：推导 ELBO，证明 gap = KL，推导重参数化技巧，分析后验坍缩。

**思路**：$\log p(\mathbf{x})$ 分解为 ELBO + KL；重参数化使梯度通过采样可微分。

**解**：

**(a) ELBO 推导**

对任意分布 $q_\phi(\mathbf{z}|\mathbf{x})$：

$$\log p_\theta(\mathbf{x}) = \log\int p_\theta(\mathbf{x},\mathbf{z})d\mathbf{z} = \log E_{q_\phi}\!\left[\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]$$

由 Jensen 不等式（$\log$ 凹函数）：

$$\log p_\theta(\mathbf{x}) \geq E_{q_\phi}\!\left[\log\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right] = E_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})]-D_{KL}(q_\phi\|p(\mathbf{z})) =: \mathcal{L}(\theta,\phi;\mathbf{x})$$

**(b) gap = KL**

$$\log p_\theta(\mathbf{x})-\mathcal{L} = \log p_\theta(\mathbf{x})-E_{q_\phi}\!\left[\log\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]$$

$$= E_{q_\phi}\!\left[\log\frac{q_\phi(\mathbf{z}|\mathbf{x})p_\theta(\mathbf{x})}{p_\theta(\mathbf{x},\mathbf{z})}\right] = E_{q_\phi}\!\left[\log\frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})}\right] = D_{KL}(q_\phi\|p_\theta(\mathbf{z}|\mathbf{x}))\geq0$$

因此 $\mathcal{L}$ 确为 $\log p_\theta(\mathbf{x})$ 的下界，当且仅当 $q_\phi=p_\theta(\mathbf{z}|\mathbf{x})$ 时等号成立。

**(c) KL 项解析式与重参数化**

设 $q_\phi(\mathbf{z}|\mathbf{x})=N(\boldsymbol\mu_\phi,\mathrm{diag}(\boldsymbol\sigma_\phi^2))$，$p(\mathbf{z})=N(\mathbf{0},\mathbf{I})$，维度 $d$：

$$D_{KL}(q_\phi\|p) = \frac{1}{2}\sum_{k=1}^d\left[\mu_{\phi,k}^2+\sigma_{\phi,k}^2-\log\sigma_{\phi,k}^2-1\right]$$

**重参数化技巧**：直接采样 $\mathbf{z}\sim q_\phi(\mathbf{z}|\mathbf{x})$ 不可微（采样操作阻断梯度）。令 $\mathbf{z}=\boldsymbol\mu_\phi(\mathbf{x})+\boldsymbol\sigma_\phi(\mathbf{x})\odot\boldsymbol\varepsilon$，$\boldsymbol\varepsilon\sim N(\mathbf{0},\mathbf{I})$，将随机性转移至与 $\phi$ 无关的 $\boldsymbol\varepsilon$，梯度 $\partial\mathbf{z}/\partial\phi$ 可直接反向传播。

**(d) 后验坍缩（Posterior Collapse）**

**现象**：训练时 KL 项被驱动至 0（$q_\phi\approx p(\mathbf{z})$），隐变量 $\mathbf{z}$ 失去信息，解码器直接忽略 $\mathbf{z}$。

**原因**：强大的自回归解码器（如 LSTM）可以不依赖 $\mathbf{z}$ 而实现高重建质量，而消除 KL 项（令 $q=p$）还能移除一个正则化惩罚，形成"双赢"的局部最优。

**缓解方法**：

1. **$\beta$-VAE**（Higgins et al. 2017）：令 $\mathcal{L}_\beta=E_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})]-\beta D_{KL}$，$\beta>1$ 时强迫更高的 KL，防止坍缩；从信息瓶颈看，$\beta$ 控制压缩率与重建精度的权衡。
2. **KL 退火（KL Annealing）**：训练初期令 $\beta$ 从 0 逐渐增大至 1，先让解码器学习依赖 $\mathbf{z}$，再引入 KL 惩罚，避免解码器过早绕过隐变量。

**答案**：ELBO $=E_q[\log p(\mathbf{x}|\mathbf{z})]-D_{KL}(q\|p(\mathbf{z}))$，gap $=D_{KL}(q\|p(\mathbf{z}|\mathbf{x}))\geq0$；KL 项解析式为 $\frac{1}{2}\sum_k[\mu_k^2+\sigma_k^2-\log\sigma_k^2-1]$；$\boxed{\text{重参数化将随机性转移至}\boldsymbol\varepsilon\text{，使梯度可反传}}$。

---

### E.6.8（Ch.17+Ch.18，EM 算法收敛 + 不完全数据 + 多峰后验）

**题目**：证明 EM 单调性，分析局部最优，讨论 MCEM 和变分 EM。

**思路**：EM 单调性 = Jensen 不等式；局部最优 = 多峰似然；变分族的限制导致近似误差。

**解**：

**(a) EM 单调性证明**

构造辅助量 $H(\boldsymbol\theta|\boldsymbol\theta^{(t)})=E_{\mathbf{Z}|\mathbf{Y},\boldsymbol\theta^{(t)}}[\log p(\mathbf{Z}|\mathbf{Y},\boldsymbol\theta)]$，由联合分布分解：

$$\log p(\mathbf{Y}|\boldsymbol\theta) = Q(\boldsymbol\theta|\boldsymbol\theta^{(t)})-H(\boldsymbol\theta|\boldsymbol\theta^{(t)})$$

M 步保证 $Q(\boldsymbol\theta^{(t+1)}|\boldsymbol\theta^{(t)})\geq Q(\boldsymbol\theta^{(t)}|\boldsymbol\theta^{(t)})$（最大化 $Q$）。

由 Jensen 不等式，$H(\boldsymbol\theta|\boldsymbol\theta^{(t)})\leq H(\boldsymbol\theta^{(t)}|\boldsymbol\theta^{(t)})$（Gibbs 不等式：KL 非负）：

$$H(\boldsymbol\theta|\boldsymbol\theta^{(t)}) = -D_{KL}(p(\mathbf{Z}|\mathbf{Y},\boldsymbol\theta^{(t)})\|p(\mathbf{Z}|\mathbf{Y},\boldsymbol\theta))+H(\boldsymbol\theta^{(t)}|\boldsymbol\theta^{(t)}) \leq H(\boldsymbol\theta^{(t)}|\boldsymbol\theta^{(t)})$$

故：

$$\log p(\mathbf{Y}|\boldsymbol\theta^{(t+1)}) = Q(\boldsymbol\theta^{(t+1)}|\boldsymbol\theta^{(t)})-H(\boldsymbol\theta^{(t+1)}|\boldsymbol\theta^{(t)}) \geq Q(\boldsymbol\theta^{(t)}|\boldsymbol\theta^{(t)})-H(\boldsymbol\theta^{(t)}|\boldsymbol\theta^{(t)}) = \log p(\mathbf{Y}|\boldsymbol\theta^{(t)})$$

$\square$

**(b) 局部最优——双峰混合高斯例子**

设 $n=100$ 个观测来自 $0.5N(-3,1)+0.5N(3,1)$，参数 $\theta=(\pi,\mu_1,\mu_2)$。

若初始化 $\mu_1^{(0)}=\mu_2^{(0)}=0$（两组分重叠），E 步给出 $r_{i1}\approx r_{i2}\approx0.5$，M 步更新后两个均值均趋向 $\bar{X}\approx0$，EM 收敛到 $\mu_1=\mu_2=0$——这是鞍点（局部最优），而非全局最大值（真实 $\mu_1=-3,\mu_2=3$）。不同初始化会导致不同的局部最优，因此混合模型常用多次随机重启或谱方法初始化。

**(c) 蒙特卡洛 EM（MCEM）**

当 E 步的条件期望 $Q(\boldsymbol\theta|\boldsymbol\theta^{(t)})=E[\ell_c|\mathbf{Y},\boldsymbol\theta^{(t)}]$ 无解析式时，用 MCMC（如 Gibbs 抽样）生成 $M$ 个样本 $\mathbf{Z}^{(1)},\ldots,\mathbf{Z}^{(M)}$ 近似：

$$\hat{Q}_M(\boldsymbol\theta|\boldsymbol\theta^{(t)}) = \frac{1}{M}\sum_{m=1}^M\log p(\mathbf{Y},\mathbf{Z}^{(m)}|\boldsymbol\theta)$$

近似误差来源：MCMC 混合不充分（链未收敛），$M$ 有限导致 Monte Carlo 误差 $O(M^{-1/2})$。随着 EM 迭代推进（$\boldsymbol\theta^{(t)}$ 趋近最优），需要更精确的 $Q$ 估计（误差更小），因此应逐步增大 $M$，以保证整体收敛。

**(d) 变分 EM vs 标准 EM**

标准 EM 的 E 步取 $q=p(\mathbf{Z}|\mathbf{Y},\boldsymbol\theta^{(t)})$（精确后验），ELBO gap 归零。

变分 EM 限制 $q\in\mathcal{Q}$（变分族），当真实后验 $\notin\mathcal{Q}$ 时，ELBO gap $=D_{KL}(q^*\|p)>0$，导致次优性。

**LDA 中的平均场假设**：令 $q(\mathbf{Z})=\prod_i q(Z_i)$（各隐变量独立），忽略了变量间的相关性。若主题分布与词的分配相关（实际上总是相关的），平均场近似系统性低估后验方差，导致主题估计过于尖锐（置信度虚高）。代价是推断从指数级（真实后验）降至多项式级复杂度。

**答案**：EM 单调性由 $Q$ 增大 + $H$ 减小（KL 非负）保证；局部最优在多峰似然时出现；变分 EM 因 $\boxed{D_{KL}(q\|p)>0}$ 而次优。

---

### E.6.9（Ch.16+Ch.17+Ch.18，贝叶斯非参数估计 + Dirichlet 过程 + 无限混合）

**题目**：DP 的存在性、均值、CRP 与边际化、DPMM 聚类。

**思路**：DP 是有限 Dirichlet 的一致极限；CRP 是 DP 混合模型隐变量的边际分布；$\alpha$ 控制新桌的期望数。

**解**：

**(a) DP 的存在性（有限 Dirichlet 极限）**

对 $\mathcal{X}$ 的任意有限可测分割 $(A_1,\ldots,A_K)$，定义：

$$(G(A_1),\ldots,G(A_K))\sim\mathrm{Dir}(\alpha G_0(A_1),\ldots,\alpha G_0(A_K))$$

此定义对所有分割 $K$ 和所有选取方式一致（由 Kolmogorov 相容条件验证），故由 Kolmogorov 扩展定理存在唯一的随机测度 $G$，满足上述有限维边际，称 $G\sim DP(\alpha,G_0)$。

**(b) DP 的均值与方差**

对分割 $(A,A^c)$，$(G(A),G(A^c))\sim\mathrm{Dir}(\alpha G_0(A),\alpha G_0(A^c))$。

Dirichlet 均值：$E[G(A)]=G_0(A)$，即 $E[G]=G_0$（DP 的均值测度等于基础测度）。

方差：$\mathrm{Var}(G(A))=\frac{G_0(A)(1-G_0(A))}{\alpha+1}$，$\alpha\to\infty$ 时方差 $\to0$，$G\to G_0$（DP 集中于 $G_0$）。

**(c) CRP 是 DP 混合模型的边际**

DP 混合：$G\sim DP(\alpha,G_0)$，$\theta_i\overset{iid}{\sim}G$，$X_i|\theta_i\sim F(\cdot|\theta_i)$。

对 $\theta_i$ 边际化（积分掉 $G$），利用 DP 的 Pólya urn 性质：

$$\theta_{n+1}|\theta_1,\ldots,\theta_n \sim \frac{1}{\alpha+n}\!\left(\alpha G_0+\sum_{i=1}^n\delta_{\theta_i}\right)$$

若将不同取值 $\theta$ 视为"桌"，则 $\theta_{n+1}$ 以概率 $\frac{n_k}{\alpha+n}$ 加入已有第 $k$ 桌（取值 $\theta_k^*$），以概率 $\frac{\alpha}{\alpha+n}$ 新开一桌（取 $\theta\sim G_0$）——这正是 CRP。因此 CRP 是 DP 混合模型将 $G$ 积分掉后的 $(\theta_1,\ldots,\theta_n)$ 的联合边际分布。

**(d) DPMM 聚类与 BIC 对比**

DPMM 自动推断簇数：期望簇数 $\approx\alpha\log(1+n/\alpha)$（随 $n$ 增长缓慢），$\alpha$ 大则期望更多簇。

**$\alpha$ 的推断**：置先验 $\alpha\sim\mathrm{Gamma}(a,b)$，利用 Gibbs 抽样（Escobar & West 1995）或 Variational Bayes 推断后验 $p(\alpha|\text{数据})$，从数据中学习"多少簇是合适的"。

**与有限 GMM + BIC 对比**：

| 方法 | 簇数选择 | 优点 | 缺点 |
|---|---|---|---|
| GMM + BIC | 枚举 $K$，选 BIC 最小 | 计算快、可解释 | 需穷举 $K$，BIC 近似 |
| DPMM | 自动（$\alpha$ 推断） | 无需指定 $K$，贝叶斯不确定性 | 计算慢，MCMC 收敛问题 |

在深度学习表示学习中，DPMM 可用于对嵌入空间进行在线聚类（如无监督分类），无需预设簇数，灵活适应数据复杂度。

**答案**：DP 均值 $E[G]=G_0$；CRP 是 DP 混合的隐变量边际；DPMM 期望簇数 $\approx\boxed{\alpha\log(1+n/\alpha)}$，自动适应数据规模。

---

### E.6.10（Ch.16+Ch.18，最大熵估计 + 凸对偶 + 特征匹配）

**题目**：最大熵原理的对偶推导，最优解为指数族，联系逻辑回归，分析 RLHF 最优策略。

**思路**：原问题（最大化熵，约束特征均值）→ Lagrange 对偶 → 凸对偶等价 → 指数族；KL 约束下的奖励最大化 → 软 Q 值策略。

**解**：

**(a) 对偶问题推导（Legendre-Fenchel 变换）**

原问题：$\max_p H(p) = -\sum_x p(x)\log p(x)$，约束 $E_p[\phi_k(x)]=\hat\mu_k$，$k=1,\ldots,m$，$\sum_x p(x)=1$。

Lagrangian：$\mathcal{L}=H(p)-\sum_k\eta_k(E_p[\phi_k]-\hat\mu_k)-\nu(\sum_x p(x)-1)$

对 $p(x)$ 求偏导令零：$-\log p(x)-1-\sum_k\eta_k\phi_k(x)-\nu=0$

解出：$p^*(x)=\exp(\sum_k\eta_k\phi_k(x)-A(\boldsymbol\eta))$（指数族）

对偶目标（对 $\boldsymbol\eta$ 最小化）：

$$g(\boldsymbol\eta) = -H(p^*) + \sum_k\eta_k(E_{p^*}[\phi_k]-\hat\mu_k) = A(\boldsymbol\eta)-\boldsymbol\eta^\top\hat{\boldsymbol\mu}$$

强对偶性：在约束可行时，$\max_p\min_{\boldsymbol\eta}\mathcal{L}=\min_{\boldsymbol\eta}\max_p\mathcal{L}$（凸优化 + Slater 条件），对偶间隙为零。

**(b) 最优解为指数族**

由 (a) 的推导，最优 $p^*(x)=\exp(\boldsymbol\eta^{*\top}\phi(x)-A(\boldsymbol\eta^*))$，其中 $A(\boldsymbol\eta)=\log\sum_x\exp(\boldsymbol\eta^\top\phi(x))$ 是对数配分函数（凸函数）。最大熵分布必为指数族，$\boldsymbol\eta^*$ 通过最小化凸对偶 $A(\boldsymbol\eta)-\boldsymbol\eta^\top\hat{\boldsymbol\mu}$ 确定。

**(c) 最大熵分类器 = 逻辑回归**

对多分类（$K$ 类），特征 $\phi_k(x,y)=x\cdot\mathbf{1}(y=k)$，最大熵模型：

$$p^*(y|x) = \frac{\exp(\boldsymbol\eta_y^\top x)}{\sum_{y'}\exp(\boldsymbol\eta_{y'}^\top x)} = \mathrm{softmax}(\boldsymbol\eta^\top x)_y$$

权重 $\boldsymbol\eta$ 的估计：最大化 $\sum_i\log p^*(y_i|x_i)$（对数似然），等价于最小化经验分布 $\hat p$ 与模型 $p^*$ 之间的 KL 散度：$\min_{\boldsymbol\eta}D_{KL}(\hat{p}\|p^*(\cdot;\boldsymbol\eta))$。这与逻辑回归的最大似然估计完全等价。$\square$

**(d) RLHF 最优策略推导**

RLHF 问题：在 KL 散度约束下最大化期望奖励：

$$\max_\pi E_\pi[r(x)]-\beta D_{KL}(\pi\|\pi_{\text{ref}})$$

写出 Lagrangian（对每个 $x$）：

$$\max_{\pi(x)}\left[\pi(x)r(x)-\beta\pi(x)\log\frac{\pi(x)}{\pi_{\text{ref}}(x)}\right]$$

对 $\pi(x)$ 求导令零：$r(x)-\beta\log\frac{\pi^*(x)}{\pi_{\text{ref}}(x)}-\beta=0$

解出：

$$\pi^*(x)\propto\pi_{\text{ref}}(x)\exp(r(x)/\beta)$$

归一化：$\pi^*(x)=\frac{\pi_{\text{ref}}(x)\exp(r(x)/\beta)}{Z}$，$Z=\sum_{x'}\pi_{\text{ref}}(x')\exp(r(x')/\beta)$。

**$\beta$ 的作用**：

- $\beta\to0$：$\pi^*(x)\propto\exp(r(x)/\beta)\to$ 贪婪（确定性地选奖励最高的输出），多样性极低但奖励对齐强；
- $\beta\to\infty$：$\pi^*\to\pi_{\text{ref}}$（参考策略），完全保留多样性但忽略奖励；
- 工程中 $\beta$ 平衡"奖励对齐"与"分布保真"，防止奖励黑客（模型找到高奖励但质量差的输出）。

**答案**：最大熵分布为指数族；最大熵分类器 = 逻辑回归；RLHF 最优策略 $\pi^*(x)\propto\pi_{\text{ref}}(x)\exp(r(x)/\beta)$，$\beta$ 控制 $\boxed{\text{奖励对齐与生成多样性的权衡}}$。
