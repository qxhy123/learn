# 高级题库 E（60 题）

> 难度：★★★★★ — 完整证明 / 综合应用 / 跨章融合 / ML 实战。每题 15-30 分钟。
> 适合考研冲刺、工程师面试、深入研究。

---

## Part 1 概率基础（Ch.1-3，共 6 题）

**E.1.1**（Ch.2，贝叶斯多次更新 + 复制性危机）
某医学诊断试验灵敏度（sensitivity）$P(+ \vert D) = 95\%$、特异度（specificity）$P(- \vert \bar{D}) = 90\%$，疾病基础发病率（先验）$P(D) = 0.5\%$。将阳性检测结果记为事件 $+$，患病记为事件 $D$。

(a) 一次阳性结果后，用完整贝叶斯公式展开写出 $P(D \vert +)$，注意分母 $P(+) = P(+ \vert D)P(D) + P(+ \vert \bar{D})P(\bar{D})$ 各项的含义；

(b) 以 (a) 的后验作为新的先验，患者再次独立接受同一检测并再次为阳性，用链式贝叶斯更新求二次阳性后的患病概率 $P(D \vert ++$）；

(c) 固定灵敏度 $s = 0.95$、特异度 $t = 0.90$，令先验 $\pi \to 0$，证明 $P(D \vert +) \to 0$，并以 $\pi = 0.001$ 数值说明假阳性率（FPR in PPV sense）的严重程度；

(d) 解释为何"p 值 $< 0.05$"**不**等价于"$H_0$ 为真的概率 $< 5\%$"——用本题结构类比：p 值是 $P(\text{数据} \vert H_0)$，而研究者想要的是 $P(H_0 \vert \text{数据})$，说明两者差距的决定因素（先验、功效、发表偏倚）。

---

**E.1.2**（Ch.1+Ch.3，容斥原理 + 概率的包含-排除 + 组合证明）
设事件 $A_1, A_2, \ldots, A_n$ 满足 $P(A_i) = p$（对所有 $i$），且任意两两独立（pairwise independent）。将容斥公式与指数生成函数相结合，可以精确分析"至少一个事件发生"的概率。

(a) 用容斥原理写出 $P\!\left(\bigcup_{i=1}^n A_i\right)$ 的精确展开式（含 $\binom{n}{k}p^k(1-p)^{n-k}$ 型项），并化简为 $1-(1-p)^n$；

(b) 证明：当 $n \to \infty$、$np \to \lambda$ 时，$1-(1-p)^n \to 1 - e^{-\lambda}$，并给出收敛速率的上界（利用不等式 $\vert 1-(1-p)^n - (1-e^{-\lambda})\vert \leq \lambda^2/n$ 的推导框架）；

(c) 生日问题：$n$ 人中至少两人同天生日的概率 $P_n = 1 - \frac{365!}{365^n(365-n)!}$——推导泊松近似 $P_n \approx 1 - e^{-\binom{n}{2}/365}$，计算 $n=23$ 时精确值与近似值，量化误差（精确到小数点后三位）；

(d) 推广至"至少三人同天生日"：定义事件 $B_{ijk}$（第 $i,j,k$ 人同天），推导 $P(\exists \text{ 三人同天}) = 1 - P(\text{至多两人同天})$ 的递推结构，建立方程并分析临界人数（给出不等式框架，结论约为 $n \approx 88$）。

---

**E.1.3**（Ch.2，条件独立性 + Simpson 悖论 + 因果推断）
某临床试验按病情轻重分层，数据如下（Simpson 悖论的经典数值示例）：

| 分组 | 治疗组康复 | 治疗组总数 | 对照组康复 | 对照组总数 |
|------|-----------|-----------|-----------|-----------|
| 轻症 | 81 | 87 | 234 | 270 |
| 重症 | 192 | 263 | 55 | 80 |
| 合计 | 273 | 350 | 289 | 350 |

（注：数字来自 Charig et al., 1986 肾结石手术研究。）

(a) 分别计算轻症组、重症组、合计三种情形下治疗组与对照组的康复率（保留三位有效数字），并对比方向；

(b) 用不等式符号明确表述 Simpson 悖论：$P(R \vert T, Z=\text{轻}) > P(R \vert \bar{T}, Z=\text{轻})$ 且 $P(R \vert T, Z=\text{重}) > P(R \vert \bar{T}, Z=\text{重})$，但 $P(R \vert T) < P(R \vert \bar{T})$，证明三个不等式均成立；

(c) 用全概率公式 $P(R \vert T) = P(R \vert T, Z=\text{轻})P(Z=\text{轻} \vert T) + P(R \vert T, Z=\text{重})P(Z=\text{重} \vert T)$ 推导：悖论的根源在于分层样本比例 $P(Z \vert T) \neq P(Z \vert \bar{T})$（混淆变量 $Z$ 与处理 $T$ 相关），量化各项数值验证等式；

(d) 从 Pearl 的 do-calculus 因果推断角度分析：$P(R \vert T)$（观测）与 $P(R \vert do(T))$（干预）的区别是什么？为什么后者才是"治疗效果"的正确定义？说明后门准则（Backdoor Criterion）如何通过 $Z$ 的调整给出无偏因果效应估计。

---

**E.1.4**（Ch.3，Borel 悖论 + 条件概率的测度论本质）
设 $(X,Y)$ 在单位圆盘 $\mathcal{D} = \{(x,y) : x^2 + y^2 \leq 1\}$ 上均匀分布，其联合密度 $f(x,y) = 1/\pi$。同一几何点可用直角坐标 $(X,Y)$ 或极坐标 $(R,\Theta)$ 描述，但对**零测度集**的条件化结果依赖坐标选取，这正是 Borel 悖论的核心。

(a) 写出直角坐标联合密度 $f_{X,Y}(x,y)$ 与极坐标联合密度 $f_{R,\Theta}(r,\theta)$（用 Jacobian $r$ 推导），验证两者积分均为 1；

(b) 在直角坐标下，对零测度集 $\{X = 0\}$ 的条件化：推导条件密度 $f_{Y \vert X}(y \vert 0)$（在 $[-1,1]$ 上均匀分布，写出推导过程）；

(c) 在极坐标下，对零测度集 $\{\Theta = \pi/2\}$ 的条件化（即"点在正 $y$ 轴方向的射线上"）：推导条件密度 $f_{R \vert \Theta}(r \vert \pi/2)$（写出 $f_{R \vert \Theta} \propto r$，即在 $[0,1]$ 上不均匀）；

(d) 集合 $\{X=0\} \cap \mathcal{D}$ 与 $\{\Theta=\pi/2\} \cap \mathcal{D}$ 在几何上是**同一条线段**，但条件分布不同——精确计算两个条件分布并对比；解释 Borel 悖论的实质（条件化方式不同 $\Rightarrow$ 极限过程不同），并说明 Kolmogorov 正则条件概率（Regular Conditional Probability）通过 $\sigma$-代数的概念如何给出唯一一致的定义。

---

**E.1.5**（Ch.2+Ch.3，全概率公式 + 组合计数 + 递推）
Pólya 瓮模型（Pólya Urn Scheme）：坛子初始含 $r$ 个红球和 $b$ 个蓝球，每次随机取出一球观察颜色后**放回并额外加入** $c$ 个同色球（$c \geq 1$）。这是强化学习（preferential attachment）和贝叶斯非参数统计的基础模型。

(a) 设 $p_n = P(\text{第}n\text{次取红})$，对 $n=1$ 显然有 $p_1 = r/(r+b)$；用全概率公式对 $n=2$ 显式计算 $p_2$，然后用数学归纳法证明 $p_n = r/(r+b)$ 对所有 $n \geq 1$ 成立（关键步骤：条件于第 $n-1$ 次结果）；

(b) 推导前 $n$ 次中恰好取到 $k$ 次红球的 Pólya 分布 PMF：利用对称性，每种包含 $k$ 次红、$n-k$ 次蓝的序列概率相同，写出公式 $P(S_n = k) = \binom{n}{k}\frac{r^{(k)}b^{(n-k)}}{(r+b)^{(n)}}$，其中 $a^{(m)} = a(a+c)(a+2c)\cdots(a+(m-1)c)$ 为上升阶乘；

(c) 设 $\hat{p}_n = S_n/n$ 为红球比例，用鞅收敛定理（$\hat{p}_n$ 是有界鞅）证明 $\hat{p}_n \xrightarrow{a.s.} \hat{p}_\infty$，并证明 $\hat{p}_\infty \sim \mathrm{Beta}(r/c, b/c)$（提示：计算矩 $E[\hat{p}_n^m]$ 并取极限）；

(d) 将 Pólya 瓮推广到 $K$ 色球（对应 Dirichlet 过程）：解释"中国餐馆过程"与 Pólya 瓮的等价性，以及在 NLP 的 n-gram 语言模型中 Pólya 瓮如何描述词频的幂律（Zipf 定律）分布。

---

**E.1.6**（Ch.1+Ch.2+Ch.3，概率不等式 + 生日碰撞 + 密码学应用）
"生日攻击"（Birthday Attack）是密码学中的经典问题：在输出空间大小为 $m$ 的 Hash 函数中，攻击者随机查询 $n$ 次，寻找两次输出相同的碰撞。设各次查询独立均匀地落在 $\{1,\ldots,m\}$ 中。

(a) 精确写出 $n$ 次查询无碰撞的概率：$P(\text{无碰撞}) = \prod_{k=0}^{n-1}\left(1 - \frac{k}{m}\right) = \frac{m(m-1)\cdots(m-n+1)}{m^n}$，并推导近似 $P(\text{无碰撞}) \approx e^{-n(n-1)/(2m)}$（利用 $\ln(1-x) \approx -x$）；

(b) 用 Union Bound 给出碰撞概率上界 $P(\text{至少一次碰撞}) \leq \binom{n}{2}/m$，分析何时该上界与精确值接近（$n \ll m$ 时）；

(c) 令每对 $(i,j)$（$i<j$）碰撞的事件为 $B_{ij}$，设 $\lambda = \binom{n}{2}/m$，用 Poisson 近似说明碰撞次数近似 $\mathrm{Poisson}(\lambda)$，从而 $P(\text{至少一次碰撞}) \approx 1 - e^{-\lambda}$；推导使 $P \geq 0.5$ 的 $n$ 约为 $\sqrt{2m\ln 2}$；

(d) 密码学应用：对 SHA-256（$m = 2^{256}$），代入 $n \approx \sqrt{2 \cdot 2^{256} \cdot \ln 2} \approx 2^{128}$——推导此数量级，说明为何 256 位哈希安全性等价于 128 位的碰撞抵抗（生日界），并讨论量子计算（Grover 算法）对该界的影响（$n_{\text{quantum}} \approx 2^{85.3}$）。

---

## Part 2 随机变量（Ch.4-6，共 8 题）

**E.2.1**（Ch.5+Ch.6，函数变换 + 卷积 + CLT 预热）
设 $X, Y$ i.i.d. $\sim U(0,1)$，令 $Z = X + Y$。卷积公式 $f_Z(z) = \int f_X(x) f_Y(z-x)\,dx$ 是求和分布的基本工具，本题从二元推广到 $n$ 元并联系中心极限定理。

(a) 用卷积公式推导 $f_Z(z)$ 的分段表达式（$z \in [0,1]$ 和 $z \in (1,2]$ 两段），验证积分为 1，画出（描述）密度函数的三角形形状；

(b) Irwin-Hall 分布：$S_n = \sum_{i=1}^n X_i$，密度 $f_{S_n}(x) = \frac{1}{(n-1)!}\sum_{k=0}^{\lfloor x \rfloor}(-1)^k\binom{n}{k}(x-k)^{n-1}$——用矩母函数 $M_{X}(t) = (e^t-1)/t$ 证明 $M_{S_n}(t) = [(e^t-1)/t]^n$，由此推导 $E[S_n] = n/2$ 和 $\mathrm{Var}(S_n) = n/12$；

(c) 对 $n=12$：$S_{12} - 6$ 的均值为 $0$、方差为 $1$，且由 CLT 近似服从 $N(0,1)$——计算实际与正态分布在 $\pm 3$ 处的 CDF 误差，解释为何历史上（Irwin, 1927）用此方法生成"近似正态"随机数，以及 Box-Muller 方法为何更精确；

(d) 写出伪代码（Python 风格）：对 $n=12, 100, 1000$，各生成 $10^5$ 个样本，用 Kolmogorov-Smirnov 统计量量化与标准正态的距离，并作图（描述直方图形态），说明尾部（$\vert z \vert > 3$）的拟合误差来源于 $U(0,1)$ 的有界支撑。

---

**E.2.2**（Ch.4，生成函数 + 概率母函数 + 分支过程）
Galton-Watson 分支过程是研究种群增长、病毒传播和语言模型生成的经典模型：$Z_0 = 1$，第 $n$ 代每个个体独立产生后代数，分布为 $\{p_k\}_{k \geq 0}$，均值 $\mu$，方差 $\sigma^2$，概率母函数（PGF）$G(s) = \sum_{k=0}^\infty p_k s^k$（$s \in [0,1]$）。

(a) 利用 PGF 的嵌套性质 $G_{Z_n}(s) = G_{Z_{n-1}}(G(s))$，用数学归纳法证明 $E[Z_n] = \mu^n$；再对 PGF 求二阶导，推导 $\mathrm{Var}(Z_n) = \sigma^2\mu^{n-1}(\mu^n-1)/(\mu-1)$（$\mu\neq 1$，含 $\mu=1$ 的特殊情形 $\mathrm{Var}(Z_n)=n\sigma^2$）；

(b) 灭绝概率 $q = \lim_{n\to\infty} P(Z_n = 0)$ 满足不动点方程 $q = G(q)$——证明：（i）$G(q) = q$ 在 $[0,1]$ 内至少有一个根（介值定理）；（ii）$\mu > 1$ 时最小根 $q^* < 1$，$\mu \leq 1$ 时 $q^* = 1$（分析 $G'(1) = \mu$ 与曲线 $y=G(s)$ 和 $y=s$ 的切点关系）；

(c) 几何分布后代 $p_k = (1-p)p^k$（$k \geq 0$）时，$G(s) = (1-p)/(1-ps)$，$\mu = p/(1-p)$——显式解方程 $G(q) = q$，求出 $q = (1-p)/p$（当 $p > 1/2$，即 $\mu > 1$ 时），验证与理论一致；

(d) 类比到 Transformer 自回归生成：每个 token 通过注意力机制"影响"后续 token 的条件分布，可视为信息的"分支传播"——若平均信息影响扇出 $\mu > 1$，用信息论中的 $\log Z_n \approx n\log\mu$ 增长速率解释"主题漂移"（话题不断衍生子话题），并讨论温度参数 $T < 1$（降低 $\mu$）如何在生成质量与多样性之间权衡。

---

**E.2.3**（Ch.5+Ch.6，柯西分布 + 无期望 + 稳定分布）
柯西分布是概率论中最重要的"反例"之一：它有对称密度、峰值明确，却没有均值和方差，不满足大数定律，是 $\alpha$-稳定分布（$\alpha=1$）的代表。设 $X, Y$ i.i.d. $\sim N(0,1)$，令 $C = X/Y$。

(a) 推导 $C$ 的密度函数：对固定 $c$，计算 $P(C \leq c) = P(X/Y \leq c)$，分 $Y > 0$ 和 $Y < 0$ 两种情形积分，对 $c$ 求导得 $f_C(c) = \frac{1}{\pi(1+c^2)}$（完整积分过程，含分情况合并）；

(b) 证明 $E[\vert C \vert] = \frac{1}{\pi}\int_{-\infty}^\infty \frac{\vert c \vert}{1+c^2}\,dc = +\infty$（积分发散），从而均值 $E[C]$ 不存在（注意"主值积分 $= 0$"与"期望存在"的区别）；

(c) 用特征函数证明稳定性：$C$ 的特征函数 $\varphi_C(t) = e^{-\vert t \vert}$（推导过程）；若 $C_1, C_2$ i.i.d. 柯西，则 $(C_1+C_2)/2$ 的特征函数为 $e^{-\vert t \vert}$，故仍为柯西——对比正态情形 $(X_1+X_2)/\sqrt{2}$ 仍为标准正态，解释"稳定分布"的定义；

(d) 在金融风险建模中，股票日收益率的尾部往往比正态更"肥"（fat-tailed）：正态假设下 $5\sigma$ 事件概率约 $10^{-7}$，而柯西假设下约 $10^{-4}$（差 3 个数量级）——定量计算并解释 2008 年金融危机中"$25\sigma$ 事件"的荒谬性，以及为何稳定分布族（含正态与柯西）是重尾建模的自然选择。

---

**E.2.4**（Ch.4+Ch.6，多项分布 + 协方差结构 + Softmax 梯度 + LDA）
设 $(X_1, \ldots, X_d) \sim \mathrm{Multinomial}(n, \mathbf{p})$，其中 $\mathbf{p} = (p_1,\ldots,p_d)$，$\sum_i p_i = 1$，$\sum_i X_i = n$。多项分布是 softmax 分类器输出的自然概率模型。

(a) 用矩母函数 $M(\mathbf{t}) = \left(\sum_i p_i e^{t_i}\right)^n$ 推导 $E[X_i] = np_i$，$\mathrm{Var}(X_i) = np_i(1-p_i)$，以及 $\mathrm{Cov}(X_i, X_j) = -np_ip_j$（$i \neq j$）；

(b) 协方差矩阵 $\boldsymbol{\Sigma} = n(\mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top)$：证明其半正定性（对任意向量 $\mathbf{v}$，$\mathbf{v}^\top\boldsymbol{\Sigma}\mathbf{v} = n[\sum_i p_i v_i^2 - (\sum_i p_i v_i)^2] \geq 0$，用 Jensen 不等式）；

(c) 证明 $\boldsymbol{\Sigma}$ 奇异（秩为 $d-1$）：因 $\mathbf{1}^\top \mathbf{X} = n$ 为常数，$\boldsymbol{\Sigma}\mathbf{1} = \mathbf{0}$——在神经网络 softmax 输出层中，这意味着 logit 的梯度有一个自由度冗余（加常数不改变 softmax），解释为何实践中常固定一个 logit 为 0（参照类别）或用 cross-entropy 梯度 $\partial \mathcal{L}/\partial z_i = \hat{p}_i - y_i$（其中 $\sum y_i = 1$）自动处理此奇异性；

(d) 贝叶斯分类：先验 $\mathbf{p} \sim \mathrm{Dir}(\boldsymbol{\alpha})$，观测 $\mathbf{n} = (n_1,\ldots,n_d)$ 个各类样本，后验 $\mathbf{p} \vert \mathbf{n} \sim \mathrm{Dir}(\boldsymbol{\alpha} + \mathbf{n})$——写出后验 MAP 估计（拉普拉斯平滑），解释 LDA 中每个文档的主题分布 $\boldsymbol{\theta}_d \sim \mathrm{Dir}(\boldsymbol{\alpha})$ 与每个主题的词分布 $\boldsymbol{\phi}_k \sim \mathrm{Dir}(\boldsymbol{\beta})$ 如何形成两层共轭结构，以及 Dirichlet 参数 $\boldsymbol{\alpha}$ 如何控制主题分布的稀疏程度。

---

**E.2.5**（Ch.5+Ch.6，次序统计量 + 极值理论 + Max-Pooling 梯度）
设 $X_1, \ldots, X_n$ i.i.d. $\sim F(x)$（连续分布，密度 $f(x)$），次序统计量 $X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}$。极值理论（EVT）是可靠性工程、气候科学和深度学习分析的共同数学基础。

(a) 推导第 $k$ 次序统计量的精确密度：$f_{(k)}(x) = \frac{n!}{(k-1)!(n-k)!}[F(x)]^{k-1}[1-F(x)]^{n-k}f(x)$（组合解释：有 $\binom{n}{1}\binom{n-1}{k-1}$ 种方式让恰好 $k-1$ 个值小于 $x$，$n-k$ 个值大于 $x$，第 $k$ 个值等于 $x$）；

(b) 对 $X_i \sim U(0,1)$，将 (a) 的密度与 $\mathrm{Beta}(k, n-k+1)$ 的密度对比，证明两者一致（利用 Beta 函数 $B(a,b) = (a-1)!(b-1)!/(a+b-1)!$ 的整数情形）；

(c) Fisher-Tippett-Gnedenko 定理（极值定理）：对 $X_i \sim \mathrm{Exp}(1)$，最大值 $M_n = X_{(n)}$ 满足 $(M_n - \ln n) \xrightarrow{d} \mathrm{Gumbel}(0,1)$——推导：首先计算 $P(M_n - \ln n \leq t) = [F(\ln n + t)]^n = (1-e^{-\ln n - t})^n = (1-e^{-t}/n)^n \to e^{-e^{-t}}$；

(d) Max-pooling 梯度分析：设池化窗口内有 $n$ 个激活值 $a_1,\ldots,a_n$ i.i.d. $\sim F$，最大值 $a_{(n)}$ 的梯度为 1，其余 $n-1$ 个梯度为 0——精确计算在反向传播中"梯度为零"的神经元比例期望 $(n-1)/n$，并用极值分布分析当网络层数 $L$ 增加时，梯度信号通过 $L$ 层 max-pooling 后的衰减规律（稀疏梯度问题）。

---

**E.2.6**（Ch.5+Ch.6，变量变换 + Jacobian + 正规化流）
设二维随机向量 $\mathbf{X} = (X_1, X_2)^\top \sim N(\boldsymbol{0}, \boldsymbol{\Sigma})$，$\boldsymbol{\Sigma} = \begin{pmatrix}1 & \rho \\ \rho & 1\end{pmatrix}$，$\vert\rho\vert < 1$。变量变换（Jacobian 公式）是推导复杂分布与理解生成模型的核心工具。

(a) 写出 $\mathbf{X}$ 的联合密度 $f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{2\pi\sqrt{1-\rho^2}}\exp\!\left(-\frac{x_1^2 - 2\rho x_1 x_2 + x_2^2}{2(1-\rho^2)}\right)$，配方后推导条件密度 $f_{X_2 \vert X_1}(x_2 \vert x_1)$，证明其为 $N(\rho x_1, 1-\rho^2)$（线性回归的概率解释：$E[X_2 \vert X_1 = x_1] = \rho x_1$，回归系数即相关系数）；

(b) 令 $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$（$\mathbf{A} \in \mathbb{R}^{2\times 2}$ 可逆）：用 Jacobian 公式 $f_{\mathbf{Y}}(\mathbf{y}) = f_{\mathbf{X}}(\mathbf{A}^{-1}(\mathbf{y}-\mathbf{b})) \cdot \vert\det \mathbf{A}\vert^{-1}$ 推导 $\mathbf{Y}$ 的分布，验证 $\mathbf{Y} \sim N(\mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)$；

(c) 正规化流：基分布 $\mathbf{Z} \sim N(\boldsymbol{0}, \mathbf{I}_d)$，可逆变换 $\mathbf{x} = g(\mathbf{z})$（$g$ 可微且 $J_g = \partial g/\partial \mathbf{z}$ 可逆），推导变换后分布的对数密度：$\log p_X(\mathbf{x}) = \log p_Z(g^{-1}(\mathbf{x})) - \log\vert\det J_g(g^{-1}(\mathbf{x}))\vert$，并解释此公式如何用于流模型的训练（最大化 $\sum_i \log p_X(\mathbf{x}_i)$）；

(d) RealNVP 仿射耦合层：将 $\mathbf{z}$ 分为两半 $(\mathbf{z}_1, \mathbf{z}_2)$，变换为 $\mathbf{x}_1 = \mathbf{z}_1$，$\mathbf{x}_2 = \mathbf{z}_2 \odot \exp(s(\mathbf{z}_1)) + t(\mathbf{z}_1)$——证明此变换可逆，计算 Jacobian 行列式（上三角结构，行列式为对角元之积），解释为何 $\det J = \prod_i \exp(s_i(\mathbf{z}_1))$ 的计算仅需 $O(d)$ 时间，且神经网络 $s, t$ 无需可逆。

---

**E.2.7**（Ch.4+Ch.5，矩生成函数 + Chernoff 界 + 鞍点近似）
矩生成函数（MGF）$M_X(t) = E[e^{tX}]$（当存在时）是导出精确尾概率界的关键工具。Chernoff 界是机器学习理论（PAC 学习、泛化界）的基础。

(a) 证明独立性 $\Rightarrow$ MGF 可乘：若 $X_1, \ldots, X_n$ i.i.d.，$M_{S_n}(t) = [M_X(t)]^n$（用独立性的期望可分解性），并推导累积量生成函数（CGF）$K(t) = n K_X(t)$ 的线性性；

(b) 推导 Chernoff 界：对任意 $t > 0$，$P(S_n \geq na) = P(e^{tS_n} \geq e^{tna}) \leq e^{-tna} M_{S_n}(t) = e^{n[K_X(t)-ta]}$，对 $t$ 最小化得 $P(S_n \geq na) \leq e^{-nI(a)}$，其中 $I(a) = \sup_t[ta - K_X(t)]$ 为 Legendre 变换；对 $X \sim \mathrm{Bernoulli}(p)$，$M_X(t) = 1-p+pe^t$，显式计算 $I(a) = a\ln(a/p) + (1-a)\ln((1-a)/(1-p))$（KL 散度形式）；

(c) 证明 $K''(t) = \mathrm{Var}(X_t) > 0$（其中 $X_t$ 服从倾斜分布 $\propto e^{tx}f(x)$），因此 $K(t)$ 是严格凸函数，确保 Legendre 变换 $I(a)$ 的唯一鞍点；

(d) 鞍点方程 $K'(\hat{t}) = a$ 的几何含义：$K'(t) = E_{X_t}[X]$ 是倾斜分布均值等于 $a$ 时的倾斜参数——Lugannani-Rice 公式 $P(S_n/n \geq a) \approx \Phi(-w) + \phi(w)(1/w - 1/u)/n^{1/2}$（其中 $w, u$ 为鞍点的函数）比 CLT 精度高一阶；解释 Laplace 近似（对后验 $\log p(\boldsymbol{\theta} \vert \mathbf{x}) \approx$ 二阶 Taylor 展开）与鞍点近似的联系（鞍点 $= \hat{\boldsymbol{\theta}}_{MAP}$，Hessian $= \mathcal{I}(\hat{\boldsymbol{\theta}})$）。

---

**E.2.8**（Ch.4+Ch.5+Ch.6，复合分布 + 塔性质 + 全方差公式 + 贝尔曼方程）
设 $N \sim \mathrm{Poisson}(\lambda)$，在给定 $N = n$ 时，$X = \sum_{i=1}^n Y_i$，其中 $Y_i$ i.i.d.（与 $N$ 独立），$E[Y] = \mu$，$\mathrm{Var}(Y) = \sigma^2$。此类**复合泊松分布**在保险精算、神经元放电建模、强化学习中均有应用。

(a) 用全期望塔性质（Law of Total Expectation）计算 $E[X]$：先写 $E[X \vert N] = N\mu$（对固定 $N$ 的线性期望），再取外层期望 $E[E[X \vert N]] = \mu E[N] = \lambda\mu$；类似地，用 $E[X^2 \vert N] = N\sigma^2 + N^2\mu^2$ 推导 $E[X^2]$；

(b) 全方差公式：$\mathrm{Var}(X) = E[\mathrm{Var}(X \vert N)] + \mathrm{Var}(E[X \vert N])$——分别计算 $E[\mathrm{Var}(X \vert N)] = E[N\sigma^2] = \lambda\sigma^2$ 和 $\mathrm{Var}(E[X \vert N]) = \mathrm{Var}(N\mu) = \lambda\mu^2$，得 $\mathrm{Var}(X) = \lambda(\sigma^2 + \mu^2)$；验证特例 $Y_i \equiv 1$（$\mu=1, \sigma^2=0$）退化为泊松；

(c) Poisson 稀疏化：令 $Y_i \sim \mathrm{Bernoulli}(p)$，则 $X = \sum_{i=1}^N Y_i$ 计数 $N$ 次试验中成功次数——用 PGF $G_X(s) = G_N(G_Y(s)) = e^{\lambda(G_Y(s)-1)} = e^{\lambda p(s-1)}$ 证明 $X \sim \mathrm{Poisson}(\lambda p)$（超泊松/欠泊松的讨论）；

(d) 强化学习贝尔曼方程：设状态 $s$，动作策略 $\pi$，奖励 $R(s, a)$，转移核 $P(s' \vert s, a)$，折扣 $\gamma \in [0,1)$。值函数 $V^\pi(s) = E^\pi\!\left[\sum_{t=0}^\infty \gamma^t R_t \,\Big\vert\, S_0 = s\right]$——用全期望公式（对第一步动作和下一状态取期望）严格推导 $V^\pi(s) = E_{a \sim \pi(\cdot \vert s)}\!\left[R(s,a) + \gamma E_{s' \sim P(\cdot \vert s,a)}[V^\pi(s')]\right]$（贝尔曼期望方程），并解释此递推与 (a)(b) 中塔性质的形式一致性。

---

## Part 3 分布（Ch.7-9，共 8 题）

**E.3.1**（Ch.7+Ch.8，指数族 + 充分统计量 + 自然参数）
指数族分布的一般形式为 $p(x \vert \boldsymbol{\eta}) = h(x)\exp\!\left(\boldsymbol{\eta}^\top \mathbf{T}(x) - A(\boldsymbol{\eta})\right)$，其中 $A(\boldsymbol{\eta})$ 为对数配分函数。

(a) 证明泊松分布 $\mathrm{Poisson}(\lambda)$、正态分布 $N(\mu, \sigma^2)$（双参数）、伽马分布 $\mathrm{Gamma}(\alpha, \beta)$ 均属于指数族，写出各自的自然参数与充分统计量；

(b) 证明 $\nabla A(\boldsymbol{\eta}) = E_\eta[\mathbf{T}(X)]$，$\nabla^2 A(\boldsymbol{\eta}) = \mathrm{Cov}_\eta[\mathbf{T}(X)]$（即 Hessian 为协方差矩阵，半正定）；

(c) 证明 Fisher 信息矩阵 $\mathcal{I}(\boldsymbol{\eta}) = \nabla^2 A(\boldsymbol{\eta})$，从而对指数族，Fisher 信息等于充分统计量的协方差；

(d) 在神经网络的自然梯度（Natural Gradient）下降中，参数更新方向为 $\mathcal{I}(\boldsymbol{\theta})^{-1} \nabla_\theta \mathcal{L}$——解释为何自然梯度比普通梯度在参数空间的曲率信息方面更优越，以及 K-FAC 近似如何降低 $\mathcal{I}^{-1}$ 的计算成本。

---

**E.3.2**（Ch.8+Ch.9，多元正态 + 边际化 + 条件分布 + 高斯过程）
设 $\mathbf{X} = \begin{pmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{pmatrix} \sim N\!\left(\begin{pmatrix}\boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2\end{pmatrix},\, \begin{pmatrix}\boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22}\end{pmatrix}\right)$。

(a) 证明边际分布 $\mathbf{X}_1 \sim N(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_{11})$；

(b) 推导条件分布 $\mathbf{X}_1 \vert \mathbf{X}_2 = \mathbf{x}_2$ 的均值与协方差（Schur 补公式）：$\boldsymbol{\mu}_{1 \vert 2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$，$\boldsymbol{\Sigma}_{1 \vert 2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$；

(c) 说明高斯过程（GP）回归的预测公式与 (b) 的联系——给定训练集 $(\mathbf{X}_{\text{train}}, \mathbf{y}_{\text{train}})$，写出测试点的后验均值与后验方差；

(d) 分析 GP 回归的计算瓶颈（$O(n^3)$ 矩阵求逆），介绍至少两种稀疏 GP 近似方法（诱导点法、随机特征近似），并说明其数学原理。

---

**E.3.3**（Ch.7，负二项分布 + 过度离散 + GLM）
设 $Y \sim \mathrm{NegBin}(r, p)$，即 $Y$ 为第 $r$ 次成功前失败次数，$p$ 为单次成功概率。

(a) 写出 PMF，并证明 $E[Y] = r(1-p)/p$，$\mathrm{Var}(Y) = r(1-p)/p^2$；

(b) 证明负二项分布可以写成泊松-伽马混合：若 $Y \vert \Lambda \sim \mathrm{Poisson}(\Lambda)$，$\Lambda \sim \mathrm{Gamma}(r, p/(1-p))$，则 $Y$ 的边际分布为 $\mathrm{NegBin}(r, p)$；

(c) 计数数据中泊松模型假设均值等于方差，但实际数据常有**过度离散**（$\mathrm{Var} > E$）——从 (b) 的混合分布视角解释过度离散的统计根源；

(d) 在广义线性模型（GLM）中，负二项回归如何建模过度离散？对比泊松回归，写出负对数似然损失，说明 ML 框架下（如 PyTorch）实现负二项损失的注意事项（参数化方式）。

---

**E.3.4**（Ch.8+Ch.9，$\chi^2$ 分布 + 卡方检验 + 独立性检验理论）
设 $Z_1, \ldots, Z_k$ i.i.d. $\sim N(0,1)$，令 $Q = \sum_{i=1}^k Z_i^2 \sim \chi^2(k)$。

(a) 从伽马分布推导 $\chi^2(k)$ 的密度、均值与方差；

(b) 设 $\mathbf{X} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$（$\boldsymbol{\Sigma}$ 正定），证明 $(\mathbf{X} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu}) \sim \chi^2(d)$（马氏距离的分布）；

(c) 在列联表独立性检验中，Pearson 统计量 $\chi^2 = \sum_{i,j} (O_{ij} - E_{ij})^2 / E_{ij}$ 在 $H_0$ 下渐近 $\chi^2((r-1)(c-1))$——用多项式分布的 CLT 给出严格推导框架（向量化版本）；

(d) 大样本量时卡方检验几乎一定显著（即使效应量极小），Cramér's $V$ 如何定量效应量？讨论统计显著性与实践显著性的区别，并在 ML 特征选择场景中说明如何正确使用卡方检验。

---

**E.3.5**（Ch.8，Beta 分布 + 共轭先验 + 汤普森采样）
设 $\theta \sim \mathrm{Beta}(\alpha, \beta)$，观测数据中成功 $s$ 次、失败 $f$ 次。

(a) 证明后验 $\theta \vert s,f \sim \mathrm{Beta}(\alpha + s, \beta + f)$（Beta-Bernoulli 共轭）；

(b) 证明 $\mathrm{Beta}(\alpha,\beta)$ 的均值为 $\alpha/(\alpha+\beta)$，众数为 $(\alpha-1)/(\alpha+\beta-2)$（$\alpha,\beta > 1$），方差为 $\alpha\beta/[(\alpha+\beta)^2(\alpha+\beta+1)]$；

(c) 在多臂老虎机（Multi-Armed Bandit）中，汤普森采样（Thompson Sampling）为每个臂维护一个 $\mathrm{Beta}(\alpha_i, \beta_i)$ 后验——每轮从每个后验中采样 $\hat{\theta}_i$，选择 $\arg\max_i \hat{\theta}_i$。证明此策略的贝叶斯最优性（后验最优臂选择概率）；

(d) 推广到高斯奖励（$r \sim N(\mu_i, \sigma^2)$）：设计相应的共轭先验-后验对，并与 UCB 算法比较两种策略在探索-利用权衡上的本质差异。

---

**E.3.6**（Ch.9，Dirichlet 分布 + 多项式后验 + LDA）
Dirichlet 分布 $\mathrm{Dir}(\boldsymbol{\alpha})$（$\boldsymbol{\alpha} = (\alpha_1,\ldots,\alpha_K)$）定义在概率单纯形 $\Delta^{K-1}$ 上。

(a) 写出密度函数（含 Beta 函数归一化因子），计算均值向量 $E[\mathbf{p}] = \boldsymbol{\alpha}/\alpha_0$（$\alpha_0 = \sum \alpha_k$）与方差 $\mathrm{Var}(p_k) = \alpha_k(\alpha_0-\alpha_k)/[\alpha_0^2(\alpha_0+1)]$；

(b) 证明 Dirichlet 分布具有聚集性（Aggregation）：若将若干类合并，合并后的边际仍服从低维 Dirichlet；

(c) 在隐含狄利克雷分配（LDA）中，文档 $d$ 的主题分布 $\boldsymbol{\theta}_d \sim \mathrm{Dir}(\boldsymbol{\alpha})$，主题 $k$ 的词分布 $\boldsymbol{\phi}_k \sim \mathrm{Dir}(\boldsymbol{\beta})$——写出 LDA 的完整联合分布，并给出变分推断的 ELBO 表达式；

(d) 分析 $\alpha_k < 1$ 与 $\alpha_k > 1$ 分别对文档主题分布形状（稀疏 vs. 均匀）的影响，并解释这在大语言模型 token 分布中的类比意义。

---

**E.3.7**（Ch.8+Ch.9，Wishart 分布 + 多元样本协方差 + 矩阵分布）
设 $\mathbf{X}_1, \ldots, \mathbf{X}_n$ i.i.d. $\sim N_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，样本协方差矩阵 $\mathbf{S} = \frac{1}{n-1}\sum_{i=1}^n (\mathbf{X}_i - \bar{\mathbf{X}})(\mathbf{X}_i - \bar{\mathbf{X}})^\top$。

(a) 证明 $(n-1)\mathbf{S} \sim W_p(n-1, \boldsymbol{\Sigma})$（Wishart 分布），其中 $W_p(m, \boldsymbol{\Sigma})$ 的定义为 $m$ 个 $N_p(\boldsymbol{0}, \boldsymbol{\Sigma})$ 样本的外积之和；

(b) 计算 $E[\mathbf{S}] = \boldsymbol{\Sigma}$（无偏性）与 $E[\mathbf{S}^{-1}] = \boldsymbol{\Sigma}^{-1}/(n-p-2)$（逆的期望，要求 $n > p+2$）；

(c) 当 $p \to \infty$，$p/n \to \gamma \in (0,1)$ 时，样本特征值分布趋向 Marchenko-Pastur 定律——写出该定律的密度公式，解释为何高维小样本场景中样本协方差矩阵的特征值严重膨胀；

(d) 在深度学习的协方差估计（如批归一化）中，当批大小 $B$ 远小于特征维度 $d$ 时，会遇到奇异协方差矩阵——分析解决方案（正则化、对角近似、低秩近似）的数学依据。

---

**E.3.8**（Ch.7+Ch.8，混合分布 + EM 算法 + 模式坍塌）
设观测 $\mathbf{x}_1, \ldots, \mathbf{x}_n$ 服从高斯混合模型（GMM）：$p(\mathbf{x}) = \sum_{k=1}^K \pi_k N(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$，$\sum \pi_k = 1$。

(a) 写出完整数据对数似然 $\log p(\mathbf{X}, \mathbf{Z} \vert \boldsymbol{\theta})$（$Z_i$ 为隐变量——属于哪个分量）；

(b) 推导 EM 算法的 E 步（后验责任 $r_{ik} = P(Z_i = k \vert \mathbf{x}_i, \boldsymbol{\theta}^{\text{old}})$）与 M 步（更新 $\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k$）的完整公式；

(c) 证明 EM 算法单调不降：$\log p(\mathbf{X} \vert \boldsymbol{\theta}^{\text{new}}) \geq \log p(\mathbf{X} \vert \boldsymbol{\theta}^{\text{old}})$（利用 Jensen 不等式与 KL 散度非负性）；

(d) GAN 中的**模式坍塌**（Mode Collapse）与 GMM 的 EM 算法有深刻类比——解释当某个分量 $\pi_k \to 0$ 时 EM 的退化行为，并说明 GAN 如何从生成模型角度遭受类似问题，以及 WGAN 如何通过 Wasserstein 距离缓解之。

---

## Part 4 极限定理（Ch.10-12，共 6 题）

**E.4.1**（Ch.10，强大数定律 + Borel-Cantelli 引理 + 几乎处处收敛）
设 $\{X_n\}$ i.i.d.，$E[X_1] = \mu$，$E[X_1^4] < \infty$。

(a) 利用 Chebyshev 不等式证明弱大数定律 $\bar{X}_n \xrightarrow{P} \mu$；

(b) 叙述 Borel-Cantelli 引理（第一、第二引理），并证明第一引理（$\sum P(A_n) < \infty \Rightarrow P(\limsup A_n) = 0$）；

(c) 利用 Borel-Cantelli 引理证明：在 $E[\vert X_1 \vert^4] < \infty$ 条件下，$P(\vert \bar{X}_n - \mu \vert > \varepsilon \text{ i.o.}) = 0$（即强大数定律）；

(d) 构造反例说明：若 $E[\vert X_1 \vert] = \infty$，强大数定律可以失效——用正尾概率以 $1/(k(\log k)^2)$ 速率衰减的分布构造反例，并分析深度学习中梯度消失/爆炸与矩存在性的关联。

---

**E.4.2**（Ch.11+Ch.10，CLT + Berry-Esseen + 非 i.i.d. 推广）
设 $\{X_n\}$ 独立（非同分布），$E[X_i] = \mu_i$，$\mathrm{Var}(X_i) = \sigma_i^2$，$E[\vert X_i - \mu_i \vert^3] = \rho_i < \infty$。令 $B_n^2 = \sum_{i=1}^n \sigma_i^2$。

(a) 叙述 Lindeberg 条件，证明其蕴含 Lindeberg CLT：$(S_n - \sum \mu_i) / B_n \xrightarrow{d} N(0,1)$；

(b) 叙述 Berry-Esseen 定理，给出收敛速率的上界 $C \sum \rho_i / B_n^3$；

(c) 对批量随机梯度下降（SGD）：令 $X_i$ 为第 $i$ 个样本的随机梯度（近似独立），$n$ 为批大小——分析 SGD 梯度噪声的中心极限行为，以及批大小如何影响梯度近似的质量（Berry-Esseen 视角）；

(d) 在非平稳数据流（如在线学习）中，Lindeberg 条件可能失效——举出具体场景（如分布偏移），并分析自适应学习率（Adam、AdaGrad）如何隐式地应对梯度分布的变化。

---

**E.4.3**（Ch.11+Ch.12，Delta 方法 + 函数的渐近分布 + Bootstrap）
设 $\hat{\theta}_n$ 为参数 $\theta$ 的估计量，$\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{d} N(0, \sigma^2)$。

(a) 叙述并证明 Delta 方法：若 $g$ 在 $\theta$ 处可微且 $g'(\theta) \neq 0$，则 $\sqrt{n}(g(\hat{\theta}_n) - g(\theta)) \xrightarrow{d} N(0, [g'(\theta)]^2 \sigma^2)$；

(b) 设 $\hat{p}_n = \bar{X}_n$ 为 Bernoulli 样本的频率，$\sigma^2 = p(1-p)$——用 Delta 方法推导 $g(p) = \log[p/(1-p)]$（logit 变换）的渐近方差，并给出置信区间（更优于直接区间的理由）；

(c) Bootstrap 原理：用经验分布 $\hat{F}_n$ 代替真实 $F$ 进行重采样——证明在正则条件下 Bootstrap 分位数估计的一致性（给出直觉性证明框架）；

(d) 在深度学习不确定性估计中，Deep Ensemble 方法等价于对模型参数的 Bootstrap——分析此类方法的理论基础与局限性，并与 Laplace 近似进行对比。

---

**E.4.4**（Ch.12，依概率收敛 + 依分布收敛 + 几乎处处收敛的关系）
设 $\{X_n\}$ 与 $\{Y_n\}$ 为随机变量序列，$X$ 为极限随机变量。

(a) 严格定义四种收敛：$L^p$ 收敛、依概率收敛、几乎处处收敛、依分布收敛，并给出它们之间所有蕴含关系（画关系图）；

(b) 构造反例：几乎处处收敛**不**蕴含 $L^1$ 收敛（提示：打字机序列）；

(c) 构造反例：依概率收敛**不**蕴含几乎处处收敛（提示：在 $[0,1]$ 上的游走示性函数）；

(d) 在深度学习中，$n$ 次随机梯度步后参数 $\boldsymbol{\theta}_n$ 的收敛性是哪种意义下的收敛？分析 SGD 在凸/非凸损失面上的收敛性结论（依概率还是几乎处处），并说明随机性对逃离鞍点的帮助。

---

**E.4.5**（Ch.10+Ch.11+Ch.12，随机过程收敛 + 泛函 CLT + Donsker 定理）
设 $S_n = \sum_{i=1}^n X_i$（$X_i$ i.i.d.，均值 $0$，方差 $\sigma^2$），定义连续化折线过程 $W_n(t) = S_{\lfloor nt \rfloor}/(\sigma\sqrt{n})$（$t \in [0,1]$）。

(a) 对固定 $t$，证明 $W_n(t) \xrightarrow{d} N(0, t)$（有限维分布的 CLT）；

(b) 叙述 Donsker 定理（泛函 CLT）：$W_n \Rightarrow W$（弱收敛到标准 Brownian Motion）；

(c) 利用 Donsker 定理推导 Kolmogorov-Smirnov 统计量 $D_n = \sup_t \vert \hat{F}_n(t) - F(t) \vert$ 的极限分布（以 Brownian Bridge 的上确界表达）；

(d) 在时序深度学习（RNN、Transformer 处理长序列）中，Donsker 定理的意义：若输入 token embedding 之和经适当标准化趋向布朗运动，分析位置编码（Positional Encoding）如何提供 $t$ 的信息，以及为何正弦位置编码的频率设计与布朗运动的谱密度有内在联系。

---

**E.4.6**（Ch.11+Ch.12，矩法 + 特征函数方法 + 稳定分布 + 广义 CLT）
设 $\{X_n\}$ i.i.d.，尾概率 $P(\vert X \vert > x) \sim L(x) x^{-\alpha}$（$\alpha \in (0,2)$，$L$ 为缓变函数），即重尾分布。

(a) 证明当 $\alpha < 2$ 时，$\mathrm{Var}(X) = +\infty$，故经典 CLT 不适用；

(b) 叙述广义中心极限定理（GCLT）：适当标准化的部分和收敛到 $\alpha$-稳定分布，写出稳定分布特征函数的一般形式；

(c) 对称柯西分布（$\alpha = 1$）与 Lévy 分布（$\alpha = 0.5$）均为稳定分布——对称柯西情形，验证 $n$ 个独立柯西之和除以 $n$ 仍为柯西，与经典 CLT 的对比；

(d) 在金融风险管理中，股票日收益率的尾指数估计（Hill 估计量）为 $\hat{\alpha}_k = [k^{-1}\sum_{i=1}^k \log X_{(n-i+1)} - \log X_{(n-k)}]^{-1}$——推导此估计量的理论依据（极值指数的矩方法），并分析在深度学习训练损失的随机游走中重尾噪声的影响。

---

## Part 5 统计基础（Ch.13-15，共 5 题）

**E.5.1**（Ch.13+Ch.15，完备充分统计量 + Lehmann-Scheffé 定理 + UMVUE）
设 $X_1, \ldots, X_n$ i.i.d. $\sim \mathrm{Poisson}(\lambda)$，$\lambda > 0$ 未知。

(a) 证明 $T = \sum_{i=1}^n X_i$ 是 $\lambda$ 的充分统计量（Fisher 分解定理）；

(b) 证明 $T$ 完备（对所有有界函数 $g$，$E_\lambda[g(T)] = 0$ 对所有 $\lambda$ 成立 $\Rightarrow g \equiv 0$ a.e.）；

(c) 构造 $e^{-\lambda}$ 的 UMVUE（基于 Lehmann-Scheffé 定理：完备充分统计量的无偏函数是 UMVUE）——提示：$P(X_1 = 0 \vert T = t)$；

(d) 将结论推广：对估计 $\lambda^k e^{-\lambda}/k!$（即 $P(X = k)$ 的 UMVUE），写出一般公式，并解释完备性对于 UMVUE 唯一性的关键作用。

---

**E.5.2**（Ch.14+Ch.13，经验分布函数 + Glivenko-Cantelli + 非参数估计）
设 $X_1, \ldots, X_n$ i.i.d. $\sim F$，经验分布函数 $\hat{F}_n(x) = n^{-1}\sum_{i=1}^n \mathbf{1}[X_i \leq x]$。

(a) 证明 $E[\hat{F}_n(x)] = F(x)$，$\mathrm{Var}(\hat{F}_n(x)) = F(x)(1-F(x))/n$；

(b) 证明 Glivenko-Cantelli 定理：$\sup_x \vert \hat{F}_n(x) - F(x) \vert \xrightarrow{a.s.} 0$（先证有限点集，再推广至全局）；

(c) 核密度估计（KDE）$\hat{f}_h(x) = \frac{1}{nh}\sum_{i=1}^n K\!\left(\frac{x - X_i}{h}\right)$：推导最优带宽 $h^* \sim n^{-1/5}$（均方误差最小化，含偏差-方差权衡）；

(d) 在生成模型中，GAN 的判别器隐式地估计真实分布与生成分布之间的差距——将此与非参数密度估计联系，分析为何训练前期 KDE 比 GAN 更稳定，以及 GAN 的优势何在（高维数据的维数诅咒）。

---

**E.5.3**（Ch.15+Ch.16，充分统计量 + Fisher 信息 + Cramér-Rao 下界）
设 $X_1, \ldots, X_n$ i.i.d. $\sim N(\mu, \sigma^2)$，$\sigma^2$ 已知，$\mu$ 未知。

(a) 计算 Fisher 信息量 $\mathcal{I}_n(\mu) = n/\sigma^2$，并验证 Cramér-Rao 下界 $\mathrm{Var}(\hat{\mu}) \geq 1/\mathcal{I}_n(\mu)$；

(b) 证明样本均值 $\bar{X}_n$ 达到 C-R 下界（有效估计量），即 $\mathrm{Var}(\bar{X}_n) = \sigma^2/n$；

(c) 设参数化变换 $\phi = g(\mu) = e^\mu$——用 Delta 方法推导 $\hat{\phi} = e^{\bar{X}}$ 的渐近分布，并给出 $\phi$ 的 Cramér-Rao 下界；

(d) 在深度学习中，神经网络参数 $\boldsymbol{\theta}$ 的 Fisher 信息矩阵 $\mathcal{I}(\boldsymbol{\theta})$ 指导自然梯度下降——解释为何标准梯度下降在参数空间的曲率不均匀时效率低下，以及 Fisher 信息矩阵的对角近似（Adagrad 类）与完整矩阵（K-FAC）的权衡。

---

**E.5.4**（Ch.13+Ch.15，指数族的充分统计量 + 最大熵原理 + 对偶性）
在所有满足约束 $E[T_k(X)] = \mu_k$（$k=1,\ldots,m$）的分布中，最大熵分布具有指数族形式：$p^*(x) = h(x)\exp\!\left(\sum_k \eta_k T_k(x) - A(\boldsymbol{\eta})\right)$。

(a) 证明以上结论（变分推导：最大化 $H(p) = -\int p \log p$ 在约束 $\int p T_k = \mu_k$ 下，用 Lagrange 乘数法）；

(b) 利用 (a) 证明：在均值与方差约束下，最大熵分布为正态分布；

(c) 证明：在 $[0,1]$ 上仅约束均值 $E[X] = \mu$ 时，最大熵分布为 Beta 分布（特例：$\mu = 0.5$ 时为均匀分布）；

(d) 在大语言模型的解码策略中，温度采样 $p_T(x) \propto p(x)^{1/T}$ 与最大熵原理的联系：分析不同温度 $T$ 下输出分布的熵变化，并解释 $T \to 0$（贪心解码）和 $T \to \infty$（均匀采样）的统计含义。

---

**E.5.5**（Ch.13+Ch.14+Ch.15，次序统计量 + 分位数估计 + 鲁棒统计）
设 $X_1, \ldots, X_n$ i.i.d. $\sim F$，令 $\xi_p = F^{-1}(p)$ 为总体 $p$-分位数，样本 $p$-分位数 $\hat{\xi}_p = X_{(\lceil np \rceil)}$。

(a) 利用次序统计量理论证明：$\sqrt{n}(\hat{\xi}_p - \xi_p) \xrightarrow{d} N\!\left(0, \frac{p(1-p)}{[f(\xi_p)]^2}\right)$（分位数的渐近正态性）；

(b) 证明样本中位数 $\hat{m} = X_{(n/2)}$ 的渐近方差为 $\frac{1}{4[f(m)]^2 n}$，而样本均值（正态数据下）渐近方差为 $\sigma^2/n$——当 $F = N(\mu, \sigma^2)$ 时比较两者，解释中位数的渐近相对效率（ARE）为 $2/\pi \approx 0.637$；

(c) 影响函数（Influence Function）$\mathrm{IF}(x; T, F) = \lim_{\varepsilon \to 0} [T((1-\varepsilon)F + \varepsilon\delta_x) - T(F)]/\varepsilon$ 度量异常点的影响——分别计算均值、中位数、M 估计量的影响函数；

(d) 在联邦学习（Federated Learning）中，若某些客户端数据被污染（Byzantine 攻击），中位数聚合（Coordinate-wise Median）比均值聚合（FedAvg）更鲁棒——用影响函数理论给出理论解释，分析攻击者能操纵的最大比例（breakdown point）。

---

## Part 6 参数估计（Ch.16-18，共 10 题）

**E.6.1**（Ch.16，MLE + Fisher 信息 + 渐近理论 + 完整证明）
设 $X_1, \ldots, X_n$ i.i.d. $\sim p(x; \theta)$（指数族），$\theta \in \mathbb{R}$，MLE 为 $\hat{\theta}_n$。

(a) 证明在正则条件（可识别性、Fisher 信息有限正定、似然光滑性）下，MLE 满足得分方程 $\sum_{i=1}^n \partial \log p(X_i; \hat{\theta})/\partial\theta = 0$；

(b) 利用 Taylor 展开证明：$\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} N(0, \mathcal{I}(\theta_0)^{-1})$（MLE 的渐近正态性与有效性）；

(c) 证明 Cramér-Rao 下界对 MLE 的渐近方差是紧的（MLE 渐近有效）；

(d) 设 $p(x;\theta) = \theta e^{-\theta x}$（$x > 0$，指数分布），计算 $\hat{\theta}_{MLE}$、Fisher 信息、渐近置信区间，并验证以上理论结论。

---

**E.6.2**（Ch.16+Ch.18，正则化 MLE + MAP + 惩罚似然）
设似然为 $L(\boldsymbol{\theta}) = \prod_{i=1}^n p(\mathbf{x}_i; \boldsymbol{\theta})$，引入先验 $\pi(\boldsymbol{\theta})$，MAP 估计量为 $\hat{\boldsymbol{\theta}}_{MAP} = \arg\max_{\boldsymbol{\theta}} \log L(\boldsymbol{\theta}) + \log \pi(\boldsymbol{\theta})$。

(a) 证明当 $\pi(\boldsymbol{\theta}) = N(\boldsymbol{0}, \tau^2 \mathbf{I})$ 时，MAP 等价于 L2 正则化 MLE（岭回归）；

(b) 证明当 $\pi(\boldsymbol{\theta}) = \prod_j \mathrm{Laplace}(0, b)$ 时，MAP 等价于 L1 正则化 MLE（Lasso）；

(c) 证明：在深度神经网络中，Dropout 训练等价于对网络权重的隐式 Bernoulli 先验下的 MAP 估计（给出直觉推导）；

(d) 随着数据量 $n \to \infty$，MAP 与 MLE 的差异如何变化？证明后验收缩（Posterior Contraction）：$\hat{\boldsymbol{\theta}}_{MAP} \to \hat{\boldsymbol{\theta}}_{MLE}$，并分析先验在小样本与大样本场景中的相对贡献。

---

**E.6.3**（Ch.16，矩估计 + GMM 估计 + 弱工具变量）
设 $X_1, \ldots, X_n$ i.i.d. $\sim F_\theta$，$\theta \in \mathbb{R}^k$，$k$ 个矩条件 $E[g_j(X; \theta)] = 0$（$j = 1,\ldots,m$，$m \geq k$）。广义矩估计（GMM）最小化 $Q_n(\theta) = \hat{g}_n(\theta)^\top \mathbf{W}_n \hat{g}_n(\theta)$，其中 $\hat{g}_n(\theta) = n^{-1}\sum g(X_i; \theta)$。

(a) 当 $m = k$ 时，GMM 退化为矩估计——证明矩估计的相合性与渐近正态性；

(b) 当 $m > k$ 时（过识别），最优权重矩阵 $\mathbf{W}^* = [\mathrm{Var}(g(X;\theta_0))]^{-1}$——推导最优 GMM 估计量的渐近方差（Hansen-Sargan 效率定理）；

(c) 推导过识别检验统计量 $J = n \hat{g}_n^\top \mathbf{W}^* \hat{g}_n \xrightarrow{d} \chi^2(m-k)$（Sargan-Hansen J 检验），并解释其经济学/统计学含义；

(d) 在深度学习的对比学习（Contrastive Learning）中，InfoNCE 损失可视为矩条件的 GMM 估计框架——分析 SimCLR 中的正负样本对比损失与 GMM 的联系，解释大批量训练（更多负样本）如何提高估计效率。

---

**E.6.4**（Ch.17，置信区间 + 枢轴量 + Bootstrap 置信区间）
设 $X_1, \ldots, X_n$ i.i.d. $\sim N(\mu, \sigma^2)$，均值和方差均未知。

(a) 构造 $\mu$ 的精确 $95\%$ 置信区间（利用 $t$ 分布，推导枢轴量 $T = \sqrt{n}(\bar{X} - \mu)/S \sim t(n-1)$）；

(b) 构造 $\sigma^2$ 的精确置信区间（利用 $\chi^2$ 分布，推导枢轴量 $Q = (n-1)S^2/\sigma^2 \sim \chi^2(n-1)$），并说明为何此区间不对称；

(c) Bootstrap 置信区间：叙述百分位数 Bootstrap 与 BCa（偏差校正加速）Bootstrap 方法，证明两者在正则条件下的二阶渐近正确性；

(d) 在深度学习的预测区间（Prediction Interval）估计中，Conformal Prediction 方法不依赖任何参数假设——写出分裂 Conformal 方法的构造步骤，证明其有限样本覆盖保证 $P(Y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha$，并与 Bootstrap 方法的理论保证对比。

---

**E.6.5**（Ch.17+Ch.18，贝叶斯可信区间 + 频率置信区间 + 概率解释对比）
设数据 $X \vert \theta \sim B(n, \theta)$（二项），先验 $\theta \sim \mathrm{Beta}(\alpha_0, \beta_0)$，观测 $X = k$。

(a) 写出后验 $\theta \vert X = k \sim \mathrm{Beta}(\alpha_0 + k, \beta_0 + n - k)$，构造 HDR（最高密度区间）作为可信区间；

(b) Wald 置信区间（频率方法）$\hat{p} \pm z_{\alpha/2}\sqrt{\hat{p}(1-\hat{p})/n}$ 与 Wilson Score 区间的比较——证明 Wald 区间在 $\hat{p}$ 接近 0 或 1 时覆盖率不达标，推导 Wilson 区间的正确公式；

(c) 比较：$P(\theta \in [\ell, u] \vert X)$ 与 $P_\theta([\ell(X), u(X)] \ni \theta)$——解释两者的哲学含义，说明"参数在可信区间内的概率为 $95\%$"对频率论者意味着什么（实际上无意义）；

(d) 在 A/B 测试（如点击率实验）中，公司实际上默认使用哪种方法？构造一个具体场景说明：贝叶斯可信区间给出更直接的决策依据，但频率置信区间具有频率覆盖保证——如何在实际工程中权衡选择。

---

**E.6.6**（Ch.18，分层贝叶斯 + 超先验 + 部分池化）
设 $J$ 个组，第 $j$ 组有 $n_j$ 个观测：$X_{ij} \vert \theta_j \sim N(\theta_j, \sigma^2)$，$\theta_j \vert \mu, \tau \sim N(\mu, \tau^2)$，超先验 $\mu \sim N(0, \Sigma_0)$，$\tau \sim \text{HalfNormal}(\sigma_\tau)$。

(a) 写出完整的联合分布 $p(\{X_{ij}\}, \{\theta_j\}, \mu, \tau)$；

(b) 推导 $\theta_j$ 的后验均值（部分池化估计量）：$\hat{\theta}_j = \lambda_j \bar{X}_j + (1-\lambda_j)\hat{\mu}$，其中 $\lambda_j = n_j\tau^2/(n_j\tau^2 + \sigma^2)$（组内信息量的权重）；

(c) 证明：当 $\tau \to 0$ 时，部分池化退化为完全池化（$\hat{\theta}_j \to \hat{\mu}$）；当 $\tau \to \infty$ 时，退化为无池化（$\hat{\theta}_j \to \bar{X}_j$）；

(d) 在联邦学习中，客户端 $j$ 的本地模型参数 $\boldsymbol{\theta}_j$ 可视为来自全局超先验的层级贝叶斯——分析 pFedMe 和 MAML 等个性化联邦学习算法与分层贝叶斯的联系，以及超先验如何编码跨客户端的先验知识。

---

**E.6.7**（Ch.16+Ch.18，变分推断 + ELBO + VAE）
设观测 $\mathbf{x}$，隐变量 $\mathbf{z}$，生成模型 $p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} \vert \mathbf{z}) p(\mathbf{z})$，变分族 $q_\phi(\mathbf{z} \vert \mathbf{x})$。

(a) 推导 ELBO（Evidence Lower Bound）：$\log p_\theta(\mathbf{x}) \geq \mathcal{L}(\theta, \phi; \mathbf{x}) = E_{q_\phi}[\log p_\theta(\mathbf{x} \vert \mathbf{z})] - D_{KL}(q_\phi(\mathbf{z} \vert \mathbf{x}) \| p(\mathbf{z}))$；

(b) 证明 gap 恰为 KL 散度：$\log p_\theta(\mathbf{x}) - \mathcal{L} = D_{KL}(q_\phi \| p_\theta(\mathbf{z} \vert \mathbf{x})) \geq 0$；

(c) 当 $p(\mathbf{z}) = N(\mathbf{0}, \mathbf{I})$，$q_\phi(\mathbf{z} \vert \mathbf{x}) = N(\boldsymbol{\mu}_\phi(\mathbf{x}), \mathrm{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{x})))$ 时，推导 KL 项的解析表达式，并推导重参数化技巧 $\mathbf{z} = \boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\varepsilon}$（$\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \mathbf{I})$）如何实现梯度的反向传播；

(d) 分析 VAE 中"后验坍缩"（Posterior Collapse）问题：为何 KL 项容易被模型驱动至零（$q_\phi \approx p(\mathbf{z})$）？提出至少两种缓解方法（$\beta$-VAE、KL 退火），并从信息瓶颈原理解释各自的统计含义。

---

**E.6.8**（Ch.17+Ch.18，EM 算法收敛 + 不完全数据 + 多峰后验）
设不完全数据 $\mathbf{Y}$，完整数据 $(\mathbf{Y}, \mathbf{Z})$，参数 $\boldsymbol{\theta}$。EM 算法交替执行：
E 步：$Q(\boldsymbol{\theta} \vert \boldsymbol{\theta}^{(t)}) = E_{\mathbf{Z} \vert \mathbf{Y}, \boldsymbol{\theta}^{(t)}}[\log p(\mathbf{Y}, \mathbf{Z} \vert \boldsymbol{\theta})]$；
M 步：$\boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta}} Q(\boldsymbol{\theta} \vert \boldsymbol{\theta}^{(t)})$。

(a) 证明 EM 的单调性：$\log p(\mathbf{Y} \vert \boldsymbol{\theta}^{(t+1)}) \geq \log p(\mathbf{Y} \vert \boldsymbol{\theta}^{(t)})$（用 Jensen 不等式，建立 $\log p(\mathbf{Y} \vert \boldsymbol{\theta})$ 与 $Q$ 的关系）；

(b) 分析 EM 收敛到局部极大值而非全局最优的情形——构造一个双峰混合高斯的例子说明局部最优的产生；

(c) 蒙特卡洛 EM（MCEM）：当 E 步不可解析时，用 MCMC 近似 $Q$——分析近似误差的来源，并说明随着迭代进行为何需要增大 MCMC 样本量；

(d) 变分 EM 与标准 EM 的关系：当变分族 $q$ 取完全后验 $p(\mathbf{Z} \vert \mathbf{Y}, \boldsymbol{\theta})$ 时，变分 EM 退化为标准 EM——解释变分族的限制如何导致变分 EM 的次优性，以及在 LDA 中平均场假设 $q(\mathbf{Z}) = \prod_i q(Z_i)$ 的代价。

---

**E.6.9**（Ch.16+Ch.17+Ch.18，贝叶斯非参数估计 + Dirichlet 过程 + 无限混合）
Dirichlet 过程（DP）是分布的分布：$G \sim DP(\alpha, G_0)$，其中 $\alpha > 0$ 为集中参数，$G_0$ 为基础分布。

(a) 通过有限维 Dirichlet 分布的极限推导 DP 的存在性——对任意可测分割 $(A_1,\ldots,A_K)$，$(G(A_1),\ldots,G(A_K)) \sim \mathrm{Dir}(\alpha G_0(A_1),\ldots,\alpha G_0(A_K))$；

(b) 证明 DP 的均值 $E[G] = G_0$，且 $\alpha$ 越大，$G$ 越集中在 $G_0$ 附近；

(c) 中国餐馆过程（CRP）：$n$ 个顾客依次入座，第 $n+1$ 个顾客加入第 $k$ 桌的概率为 $n_k/(\alpha + n)$，新开一桌的概率为 $\alpha/(\alpha + n)$——证明 CRP 是 DP 混合模型的隐变量边际化；

(d) Dirichlet 过程混合模型（DPMM）在聚类问题中自动确定簇数——分析 $\alpha$ 的贝叶斯推断（通常先验 $\alpha \sim \mathrm{Gamma}(a,b)$），并与有限 GMM 的 BIC 准则选择模型阶数进行对比，解释 DPMM 在深度学习表示学习中的潜在应用。

---

**E.6.10**（Ch.16+Ch.18，最大熵估计 + 凸对偶 + 特征匹配）
在约束 $E_p[\phi_k(x)] = \hat{\mu}_k$（$k=1,\ldots,m$）下，最大熵估计等价于对偶问题：最小化凸函数 $A(\boldsymbol{\eta}) - \boldsymbol{\eta}^\top \hat{\boldsymbol{\mu}}$（相对于对偶变量 $\boldsymbol{\eta}$）。

(a) 推导对偶问题（Legendre-Fenchel 变换：$A(\boldsymbol{\eta}) = \sup_p [\boldsymbol{\eta}^\top E_p[\phi] - H(p)]$），证明强对偶性；

(b) 证明最优解 $p^*$ 必为指数族 $p^*(x) = \exp(\boldsymbol{\eta}^{*\top}\phi(x) - A(\boldsymbol{\eta}^*))$（复现 E.5.4 (a) 的结论，现从对偶角度推导）；

(c) 对数线性模型（最大熵分类器）等价于逻辑回归——证明对多分类问题（$K$ 类），最大熵模型的决策函数为 softmax，权重 $\boldsymbol{\eta}$ 通过极大化对数似然（负 KL 散度最小化）估计；

(d) 在大语言模型的 RLHF（人类反馈强化学习）中，KL 散度约束 $D_{KL}(\pi \| \pi_{\text{ref}}) \leq \delta$ 下的奖励最大化问题正是最大熵框架——推导最优策略 $\pi^*(x) \propto \pi_{\text{ref}}(x) \exp(r(x)/\beta)$（软 Q 值策略），并解释 $\beta$ 在控制生成多样性与奖励对齐上的作用。

---

## Part 7 假设检验（Ch.19-21，共 7 题）

**E.7.1**（Ch.19，Neyman-Pearson 引理 + 最优检验 + 似然比检验）
设简单假设 $H_0: \theta = \theta_0$ vs. $H_1: \theta = \theta_1$，观测 $X_1, \ldots, X_n$ i.i.d.。

(a) 叙述并严格证明 Neyman-Pearson 引理：水平 $\alpha$ 下最优检验（最大功效）为似然比检验 $\Lambda = L(\theta_1)/L(\theta_0) \gtrless k$；

(b) 将 NP 引理推广到复合备择假设（单边检验）——对指数族分布，证明一致最优检验（UMP）的存在性（单调似然比条件）；

(c) 设 $X_i \sim N(\mu, 1)$，检验 $H_0: \mu = 0$ vs. $H_1: \mu = 1$，$n = 25$，$\alpha = 0.05$——计算最优拒绝域、功效、第二类错误率 $\beta$，并绘制（描述）功效函数的形状；

(d) 在机器学习中，分类器的决策边界可视为假设检验——ROC 曲线与 Neyman-Pearson 框架的关系：证明 AUC 等于随机选取正例和负例时正例得分高于负例的概率 $P(\hat{p}_+ > \hat{p}_-)$，并分析 AUC 优化与精度-召回率权衡的统计含义。

---

**E.7.2**（Ch.19+Ch.20，p 值 + 多重检验 + FDR 控制）
在高通量实验（如基因组学）中，同时检验 $m$ 个假设 $H_{0,1}, \ldots, H_{0,m}$，各假设的 p 值为 $p_1, \ldots, p_m$。

(a) 定义族错误率（FWER）和错误发现率（FDR），证明 $FDR \leq FWER$；

(b) Bonferroni 校正控制 FWER：若每个检验用水平 $\alpha/m$，证明 $\text{FWER} \leq \alpha$；

(c) Benjamini-Hochberg（BH）方法：将 p 值排序 $p_{(1)} \leq \cdots \leq p_{(m)}$，令 $k = \max\{i: p_{(i)} \leq i\alpha/m\}$，拒绝 $H_{0,(1)}, \ldots, H_{0,(k)}$——叙述并证明（或给出核心推导框架）$\text{FDR} \leq \alpha \cdot m_0/m \leq \alpha$，其中 $m_0$ 为真零假设数；

(d) 在深度学习神经网络的特征显著性分析（如基于梯度的特征重要性）中，每个输入特征对应一个假设检验——分析在高维场景（$m \gg n$）中多重检验校正的必要性，以及为何 SHAP 值等方法需要配合统计检验才能给出可靠结论。

---

**E.7.3**（Ch.20，方差分析（ANOVA）+ 线性模型 + F 检验理论）
设单因素方差分析模型：$X_{ij} = \mu + \alpha_i + \varepsilon_{ij}$（$i=1,\ldots,k$，$j=1,\ldots,n_i$），$\varepsilon_{ij} \sim N(0,\sigma^2)$，$\sum n_i \alpha_i = 0$。

(a) 推导组间平方和 $SS_A = \sum n_i (\bar{X}_{i\cdot} - \bar{X}_{\cdot\cdot})^2$ 和组内平方和 $SS_E = \sum_{i,j}(X_{ij} - \bar{X}_{i\cdot})^2$ 的自由度，以及 $SS_E/\sigma^2 \sim \chi^2(N-k)$（$N = \sum n_i$）；

(b) 在 $H_0: \alpha_1 = \cdots = \alpha_k = 0$ 下，证明 $F = (SS_A/(k-1))/(SS_E/(N-k)) \sim F(k-1, N-k)$；

(c) 推导非中心 F 分布下功效的计算（非中心参数 $\lambda = \sum n_i\alpha_i^2/\sigma^2$），并分析样本量设计问题；

(d) 深度学习中批归一化（Batch Normalization）的统计解释：将不同层的激活视为 ANOVA 中的"组"——说明 BN 如何通过控制组间差异（层间协变量偏移）加速训练，并用方差分析框架分析 BN 的正则化效果。

---

**E.7.4**（Ch.20+Ch.21，非参数检验 + 秩统计量 + 渐近理论）
设 $X_1, \ldots, X_m$ 与 $Y_1, \ldots, Y_n$ 独立，来自连续分布 $F$ 和 $G$，检验 $H_0: F = G$。

(a) 定义 Wilcoxon 秩和统计量 $W = \sum_{i=1}^m R_i$（$R_i$ 为 $X_i$ 在合并样本中的秩）——在 $H_0$ 下计算 $E[W] = m(m+n+1)/2$，$\mathrm{Var}(W) = mn(m+n+1)/12$；

(b) 证明 $W$ 与 Mann-Whitney U 统计量 $U = \sum_{i=1}^m \sum_{j=1}^n \mathbf{1}[X_i > Y_j]$ 的关系 $U = W - m(m+1)/2$；

(c) 证明在 $H_0$ 下，$(W - E[W])/\sqrt{\mathrm{Var}(W)} \xrightarrow{d} N(0,1)$（Hájek 投影定理框架）；

(d) 在 ML 模型性能比较中，两个分类器在多个数据集上的性能可用 Wilcoxon 符号秩检验（配对数据）——分析此方法相比 $t$ 检验的优势（对异常值鲁棒），以及 Demšar 推荐的使用秩统计量进行算法比较的完整框架。

---

**E.7.5**（Ch.19+Ch.20，顺序检验 + Wald 序贯概率比检验 + 提前终止）
设 $X_1, X_2, \ldots$ i.i.d. $\sim p_\theta$，顺序地检验 $H_0: \theta = \theta_0$ vs. $H_1: \theta = \theta_1$。

(a) 定义对数似然比过程 $\Lambda_n = \sum_{i=1}^n \log[p_{\theta_1}(X_i)/p_{\theta_0}(X_i)]$，证明在 $H_0$ 下 $E[\Lambda_n] < 0$；

(b) 叙述 Wald 序贯概率比检验（SPRT）：当 $\Lambda_n \geq B$ 时拒绝 $H_0$，当 $\Lambda_n \leq A$ 时接受 $H_0$，否则继续——用 $A \approx \log(\beta/(1-\alpha))$ 和 $B \approx \log((1-\beta)/\alpha)$ 控制双侧错误；

(c) 证明 SPRT 的最优性（Wald-Wolfowitz 定理）：在所有满足错误率约束 $(\alpha, \beta)$ 的顺序检验中，SPRT 期望样本量最小；

(d) 在线上 A/B 测试中，持续监测（Peeking Problem）导致 I 类错误膨胀——分析原因（多重检验），并说明 always-valid p-values（基于鞅不等式）和顺序检验如何解决这一问题，给出实际工程实现思路。

---

**E.7.6**（Ch.21，置换检验 + 精确检验 + 条件推断）
设检验双样本均值之差 $H_0: \mu_X = \mu_Y$，无分布假设。

(a) 叙述置换检验（Permutation Test）的完整步骤，证明在 $H_0$ 下交换性成立时，检验的精确水平为 $\alpha$（有限样本精确性）；

(b) 推导置换 p 值：$p_{\text{perm}} = \#\{(\mathbf{x}, \mathbf{y}) \text{ 的置换}: T \geq t_{\text{obs}}\} / \binom{m+n}{m}$，说明计算时如何用 Monte Carlo 近似（随机置换检验）；

(c) Fisher 精确检验用于 $2 \times 2$ 列联表——证明在边际固定的条件下，检验统计量服从超几何分布，写出精确 p 值公式；

(d) 在机器学习的特征重要性评估中，置换特征重要性（permutation feature importance）基于相同思想——分析置换后模型性能下降与条件独立性检验的联系，并说明为何高相关特征集合中单特征置换会低估重要性（提出分组置换的改进）。

---

**E.7.7**（Ch.19+Ch.21，贝叶斯假设检验 + 贝叶斯因子 + 模型选择）
设两个竞争模型 $M_0$ 和 $M_1$，数据 $\mathbf{x}$，贝叶斯因子 $BF_{10} = p(\mathbf{x} \vert M_1)/p(\mathbf{x} \vert M_0)$。

(a) 证明后验比 = 先验比 × 贝叶斯因子：$P(M_1 \vert \mathbf{x})/P(M_0 \vert \mathbf{x}) = [P(M_1)/P(M_0)] \cdot BF_{10}$；

(b) 计算正态数据（$X_i \sim N(\mu, 1)$）下，$M_0: \mu = 0$ vs. $M_1: \mu \sim N(0, \tau^2)$（Zellner-Siow 先验）的贝叶斯因子（含 Laplace 近似的近似计算）；

(c) Jeffreys-Lindley 悖论：当 $\tau \to \infty$（无信息先验），$BF_{10} \to 0$——解释为何贝叶斯因子对备择假设的先验极度敏感，而频率 p 值不然；

(d) 在深度学习模型选择中，BIC 近似边际似然 $\log p(\mathbf{x} \vert M) \approx \log p(\mathbf{x} \vert \hat{\boldsymbol{\theta}}) - (d/2)\log n$——推导 BIC 的贝叶斯解释（Laplace 近似），并与 AIC 比较：AIC 优化预测性能，BIC 一致选择真实模型（当真实模型在候选集中时）。

---

## Part 8 高级主题（Ch.22-24，共 10 题）

**E.8.1**（Ch.22，KL 散度 + 互信息 + 信息瓶颈原理）
设 $X \to Z \to Y$ 为马尔可夫链（$Z$ 为 $X$ 的压缩表示，$Y$ 为预测目标）。信息瓶颈（IB）目标：最大化 $I(Z;Y) - \beta I(Z;X)$。

(a) 用 KL 散度写出互信息 $I(Z;X) = E_{p(x,z)}[\log p(z \vert x)/p(z)]$，并证明 $I(Z;X) = H(Z) - H(Z \vert X) = H(X) - H(X \vert Z)$；

(b) 证明数据处理不等式：若 $X \to Z \to Y$ 为马尔可夫链，则 $I(X;Y) \geq I(Z;Y)$；

(c) 推导信息瓶颈的 Lagrangian 最优解满足：$p^*(z \vert x) \propto p(z) \exp(-\beta D_{KL}(p(y \vert x) \| p(y \vert z)))$，并解释在 $\beta \to 0$ 和 $\beta \to \infty$ 两种极端情形下压缩表示 $Z$ 的形式；

(d) 在深度神经网络的训练过程中，Tishby 等人提出信息平面假说（Information Plane Hypothesis）：网络各层的 $(I(X;T_l), I(T_l;Y))$ 轨迹先增大 $I(T_l;Y)$（拟合阶段）再减小 $I(X;T_l)$（压缩阶段）——分析此假说的支持证据与批评（Saxe 等人的反例），以及随机梯度噪声在信息压缩中的作用。

---

**E.8.2**（Ch.22+Ch.23，最大熵谱分析 + 随机过程 + MCMC 诊断）
设平稳过程 $\{X_t\}$，自相关函数 $R(k) = E[X_t X_{t+k}]$，功率谱密度 $S(f) = \sum_{k=-\infty}^\infty R(k) e^{-2\pi i fk}$。

(a) 证明 Wiener-Khintchine 定理：$S(f) = \mathcal{F}[R(k)](f)$，并由 Bochner 定理说明 $S(f) \geq 0$；

(b) 在已知 $R(0), R(1), \ldots, R(p)$ 的条件下，最大熵谱估计（Burg 算法）最大化微分熵 $h = \int \log S(f) df$——证明最优谱为有理谱（自回归 AR(p) 过程的谱）；

(c) 在 MCMC 中，链的自相关函数影响有效样本量（ESS）$n_{\text{eff}} = n/(1 + 2\sum_{k=1}^\infty \rho(k))$——推导此公式，并解释为何高自相关（如随机游走 Metropolis 在高维问题中）导致 ESS 极低；

(d) Hamiltonian Monte Carlo（HMC）通过引入动量变量消除随机游走行为——写出 HMC 的 Hamiltonian $H(\mathbf{q}, \mathbf{p}) = -\log p(\mathbf{q}) + \mathbf{p}^\top\mathbf{p}/2$，解释 Leapfrog 积分器的保体积性（Liouville 定理），以及 NUTS（No-U-Turn Sampler）如何自适应调整步长和路径长度。

---

**E.8.3**（Ch.23，重要性采样 + 自归一化 + 粒子滤波）
设目标分布 $\pi(x) \propto \tilde{\pi}(x)$（归一化常数未知），提议分布 $q(x)$，重要性权重 $w(x) = \tilde{\pi}(x)/q(x)$。

(a) 证明自归一化重要性采样估计量 $\hat{\mu}_{SNIS} = \sum w(X_i) f(X_i)/\sum w(X_i)$ 是有偏但相合的估计量；

(b) 推导有效样本量（ESS）$\hat{n}_{\text{eff}} = (\sum w_i)^2/\sum w_i^2$，并证明当 $q = \pi$ 时 $n_{\text{eff}} = n$，当权重退化时 $n_{\text{eff}} \to 1$；

(c) 序列重要性重采样（SIR）/ 粒子滤波：设隐状态马尔可夫链 $X_t \sim p(x_t \vert x_{t-1})$，观测 $Y_t \sim p(y_t \vert x_t)$——推导过滤分布 $p(x_t \vert y_{1:t})$ 的粒子近似递推公式（含权重更新与重采样步骤）；

(d) 在扩散模型（Diffusion Model）中，DDPM 的去噪过程可视为时间反转的重要性采样——分析前向过程 $q(x_t \vert x_0)$（高斯核），以及去噪网络 $p_\theta(x_{t-1} \vert x_t)$ 学习时间反转分布的统计原理，并与粒子滤波的预测-校正框架对应。

---

**E.8.4**（Ch.24+Ch.22，变分推断 + 均场近似 + 自由能）
在概率图模型中，变分推断用变分族 $q(\mathbf{z}) \in \mathcal{Q}$ 近似后验 $p(\mathbf{z} \vert \mathbf{x})$，最小化 $D_{KL}(q \| p(\cdot \vert \mathbf{x}))$。

(a) 证明最小化 $D_{KL}(q \| p(\cdot \vert \mathbf{x}))$ 等价于最大化 ELBO $\mathcal{L}(q) = E_q[\log p(\mathbf{x}, \mathbf{z})] - E_q[\log q(\mathbf{z})]$；

(b) 均场近似假设 $q(\mathbf{z}) = \prod_i q_i(z_i)$——推导坐标上升变分推断（CAVI）的更新方程：$\log q_i^*(z_i) = E_{-i}[\log p(\mathbf{x}, \mathbf{z})] + \text{const}$；

(c) 将 (b) 应用于高斯混合模型（GMM）：写出隐类别 $Z_i$ 和分量参数 $(\boldsymbol{\mu}_k, \boldsymbol{\Lambda}_k)$ 的 CAVI 更新公式（均场 VI for GMM）；

(d) 均场近似的代价：$D_{KL}(q \| p)$ 倾向于找低方差的近似（零覆盖），而 $D_{KL}(p \| q)$（期望传播）倾向于找高方差的近似（均值覆盖）——对比两种散度在多模态后验近似中的行为，并分析 Normalizing Flow 变分族如何突破均场的因子化限制。

---

**E.8.5**（Ch.23+Ch.24，MCMC + 马尔可夫链平稳分布 + 混合时间）
设有限状态马尔可夫链转移矩阵 $\mathbf{P}$，目标分布 $\pi$（满足细致平衡 $\pi_i P_{ij} = \pi_j P_{ji}$）。

(a) 证明满足细致平衡的 $\pi$ 是 $\mathbf{P}$ 的平稳分布（$\pi \mathbf{P} = \pi$）；

(b) 证明 Metropolis-Hastings 算法的接受概率 $\alpha(x,y) = \min\!\left(1, \frac{\pi(y)q(y,x)}{\pi(x)q(x,y)}\right)$ 使得细致平衡成立；

(c) 定义混合时间 $t_{\text{mix}}(\varepsilon) = \min\{t: \max_x \|P^t(x,\cdot) - \pi\|_{TV} \leq \varepsilon\}$，证明 $t_{\text{mix}} \leq \log(1/\varepsilon\pi_{\text{min}})/\text{Gap}$，其中 $\text{Gap} = 1 - \lambda_2$ 为谱隙；

(d) 在大语言模型的自回归采样（Ancestral Sampling）中，token 序列生成可视为马尔可夫链——分析温度参数 $T$ 对谱隙（混合速度）的影响：$T \to 0$ 使链混合变慢（卡在局部极值），$T \to \infty$ 使链混合加快但输出随机——推导最优温度的权衡准则（最大化有效信息率）。

---

**E.8.6**（Ch.22+Ch.24，概率图模型 + 信念传播 + 近似推断）
设因子图 $G = (V, F, E)$，变量节点 $v \in V$，因子节点 $f \in F$，因子 $f(\mathbf{x}_f)$ 为局部势函数。联合分布 $p(\mathbf{x}) \propto \prod_f f(\mathbf{x}_f)$。

(a) 在树结构因子图上，导出精确信念传播（BP）的消息传递方程：$\mu_{f\to v}(x_v) = \sum_{\sim x_v} f(\mathbf{x}_f) \prod_{u \in \text{ne}(f)\setminus v} \mu_{u\to f}(x_u)$；

(b) 证明树结构图上 BP 的精确性（有限步内收敛到精确边际），并分析循环图（Loopy BP）的不精确性来源；

(c) Bethe 自由能近似：$F_{\text{Bethe}} = -\sum_f E_{\hat{b}_f}[\log f] + \sum_f H(\hat{b}_f) - \sum_v (d_v - 1) H(\hat{b}_v)$——证明树图上 $F_{\text{Bethe}}$ 等于精确自由能，并解释循环图上 BP 不动点对应 Bethe 自由能的驻点；

(d) 在深度学习的结构化预测（如 CRF + Neural Network）中，Loopy BP 用于序列标注和图像分割——分析神经 CRF（如 CRFasRNN、DeepLab v2 的 DenseCRF）中均场推断的迭代方程，说明均场推断为何能以 RNN 形式嵌入端到端训练。

---

**E.8.7**（Ch.22，Rényi 散度 + $\alpha$-散度 + 鲁棒推断）
$\alpha$-Rényi 散度定义为 $D_\alpha(p \| q) = \frac{1}{\alpha - 1}\log \int p(x)^\alpha q(x)^{1-\alpha} dx$（$\alpha \neq 1$）。

(a) 证明 $\lim_{\alpha \to 1} D_\alpha(p \| q) = D_{KL}(p \| q)$（L'Hôpital 法则）；

(b) 证明 $D_\alpha(p \| q) \geq 0$（对数求和不等式）以及 Rényi 散度关于 $\alpha$ 的单调性；

(c) $\alpha = 0.5$ 时给出 Bhattacharyya 距离 $D_B = -\log \int \sqrt{p(x)q(x)} dx$——计算两个正态分布 $N(\mu_1, \sigma^2)$ 和 $N(\mu_2, \sigma^2)$ 的 Bhattacharyya 距离；

(d) 在鲁棒贝叶斯推断中，$\alpha$-散度提供比 KL 散度更灵活的近似框架——分析 Power EP（期望传播的幂散度版本）在重尾似然（柯西分布数据）中的优势，以及为何 VAE 中用 $\alpha < 1$ 的 Rényi 散度替换 KL 散度能缓解后验坍塌。

---

**E.8.8**（Ch.23+Ch.24，随机变分推断 + 黑盒 VI + 策略梯度联系）
设 ELBO $\mathcal{L}(\phi) = E_{q_\phi(\mathbf{z})}[\log p(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z})]$，需要对 $\phi$ 求梯度。

(a) 推导 REINFORCE 梯度估计量（打分函数估计量）：$\nabla_\phi \mathcal{L} = E_{q_\phi}[(\log p(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z})) \nabla_\phi \log q_\phi(\mathbf{z})]$；

(b) 证明基线（baseline/control variate）$b$（不依赖 $\mathbf{z}$）不改变梯度估计量的期望，但能降低方差：$E_{q_\phi}[(f(\mathbf{z}) - b)\nabla_\phi \log q_\phi] = E_{q_\phi}[f(\mathbf{z})\nabla_\phi \log q_\phi]$；

(c) 推导最优基线 $b^* = E[f(\mathbf{z})\|\nabla_\phi \log q_\phi\|^2]/E[\|\nabla_\phi \log q_\phi\|^2]$（最小化方差），并与强化学习中策略梯度定理的 baseline（值函数）做类比；

(d) 重参数化梯度 vs. 打分函数梯度：对于 Gaussian 变分族，前者方差远低于后者——分析两者在离散变量（如图神经网络的离散结构）情形下的适用性，以及 Gumbel-Softmax / Straight-Through Estimator 如何为离散变量提供低方差梯度估计。

---

**E.8.9**（Ch.24，贝叶斯优化 + 高斯过程代理模型 + 采集函数）
设目标函数 $f(\mathbf{x})$ 黑盒且昂贵，用高斯过程代理模型 $f \sim GP(\mu_0, k(\cdot,\cdot))$，观测集 $\mathcal{D}_n = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$（$y_i = f(\mathbf{x}_i) + \varepsilon_i$，$\varepsilon_i \sim N(0, \sigma_n^2)$）。

(a) 推导 GP 后验：均值 $\mu_n(\mathbf{x}) = \mu_0(\mathbf{x}) + \mathbf{k}_n(\mathbf{x})^\top (\mathbf{K}_n + \sigma_n^2 \mathbf{I})^{-1}(\mathbf{y} - \mu_0(\mathbf{X}))$，方差 $\sigma_n^2(\mathbf{x}) = k(\mathbf{x},\mathbf{x}) - \mathbf{k}_n(\mathbf{x})^\top(\mathbf{K}_n + \sigma_n^2\mathbf{I})^{-1}\mathbf{k}_n(\mathbf{x})$；

(b) 定义置信上界（UCB）采集函数 $\alpha_{UCB}(\mathbf{x}) = \mu_n(\mathbf{x}) + \kappa \sigma_n(\mathbf{x})$，推导其期望改进（EI）的等价性，以及 $\kappa$ 如何控制探索-利用权衡（连接到 Thompson Sampling）；

(c) 证明：对 UCB 采集函数，贝叶斯遗憾（Bayesian Regret）在 $T$ 步后有上界 $O(\sqrt{T \gamma_T})$，其中 $\gamma_T$ 为核矩阵的最大信息增益；

(d) 在深度学习的超参数优化（HPO）中，贝叶斯优化（如 Optuna、SMAC）相比随机搜索和网格搜索为何更高效——分析在高维超参数空间中 GP 的维数诅咒问题，以及 BORE（贝叶斯优化通过密度比估计）和 Tree-structured Parzen Estimator（TPE）如何绕开 GP 的计算瓶颈。

---

**E.8.10**（Ch.22+Ch.23+Ch.24，Wasserstein 距离 + 最优传输 + 生成模型）
设 $\mu, \nu$ 为 $\mathbb{R}^d$ 上的概率分布，$\mathcal{P}(\mu, \nu)$ 为所有以 $\mu, \nu$ 为边际的联合分布（耦合）集合。$p$-Wasserstein 距离：$W_p(\mu, \nu) = \left(\inf_{\gamma \in \mathcal{P}(\mu,\nu)} \int \|\mathbf{x} - \mathbf{y}\|^p \, d\gamma(\mathbf{x}, \mathbf{y})\right)^{1/p}$。

(a) 证明 $W_1$ 距离的对偶表示（Kantorovich-Rubinstein 定理）：$W_1(\mu,\nu) = \sup_{\|f\|_L \leq 1} \int f \, d\mu - \int f \, d\nu$（在 1-Lipschitz 函数上取上确界）；

(b) 对一维分布，证明 $W_1(\mu,\nu) = \int_{-\infty}^\infty \vert F_\mu(t) - F_\nu(t) \vert \, dt$（CDF 的 $L^1$ 距离），并计算 $N(0,1)$ 与 $N(\delta, 1)$ 的 $W_1$ 距离；

(c) 在 WGAN（Wasserstein GAN）中，判别器（Critic）近似 Kantorovich 对偶中的 1-Lipschitz 函数——解释梯度惩罚（GP）如何强制 Lipschitz 约束，并推导 WGAN-GP 损失：$\mathcal{L} = E_{\tilde{x}}[D(\tilde{x})] - E_x[D(x)] + \lambda E_{\hat{x}}[(\|\nabla D(\hat{x})\|_2 - 1)^2]$；

(d) Schrödinger 桥（SB）问题是对最优传输的随机化推广——设正向过程为布朗运动 $dX_t = dW_t$（$X_0 \sim \mu$，$X_1 \sim \nu$），推导 SB 的 IPFP（迭代比例拟合法）算法框架，并说明扩散模型（DDPM/Score Matching）与 Schrödinger 桥的关系（I2SB 等方法的理论依据）。

---

*本题库包含 60 题，覆盖 Part 1–8 全部章节，强调跨章融合与 ML/深度学习实战应用。*
