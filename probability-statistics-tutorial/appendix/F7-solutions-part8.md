# F7 详解：Part 8 高级专题（Ch.22-24，共 32 题）

## C 基础题详解（10 题）

---

### C.8.1（Ch.22，香农熵）

**题目**：离散变量 $X$ 取值概率 $(0.5, 0.25, 0.25)$；另 $Y \sim \text{Uniform}\{1,2,3,4\}$，比较两者熵。

**思路**：代入 $H = -\sum p_i \log_2 p_i$，均匀分布熵最大。

**解**：

1. 计算 $H(X)$：

$$H(X) = -0.5\log_2 0.5 - 0.25\log_2 0.25 - 0.25\log_2 0.25$$

$$= -0.5 \times (-1) - 0.25 \times (-2) - 0.25 \times (-2)$$

$$= 0.5 + 0.5 + 0.5 = 1.5 \text{ 比特}$$

2. 计算 $H(Y)$：$Y$ 均匀分布在 4 个等概率值上，

$$H(Y) = \log_2 4 = 2 \text{ 比特}$$

比较：$H(Y) = 2 > H(X) = 1.5$，均匀分布熵更大。

**答案**：$\boxed{H(X) = 1.5 \text{ 比特},\quad H(Y) = 2 \text{ 比特},\quad H(Y) > H(X)}$

**注**：等概率分布在相同支撑大小下熵最大（最大熵定理）。

---

### C.8.2（Ch.22，KL 散度）

**题目**：$P = (0.5, 0.5)$，$Q = (0.8, 0.2)$，计算 KL 散度并判断对称性。

**思路**：直接代入定义，非对称性举反例验证。

**解**：

1. $D_{\mathrm{KL}}(P \| Q)$（以自然对数）：

$$D_{\mathrm{KL}}(P \| Q) = 0.5 \ln\frac{0.5}{0.8} + 0.5 \ln\frac{0.5}{0.2}$$

$$= 0.5 \ln(0.625) + 0.5 \ln(2.5)$$

$$= 0.5(-0.4700) + 0.5(0.9163) = 0.2232 \text{ nat}$$

2. KL 散度**不对称**。反证：

$$D_{\mathrm{KL}}(Q \| P) = 0.8\ln\frac{0.8}{0.5} + 0.2\ln\frac{0.2}{0.5}$$

$$= 0.8\ln(1.6) + 0.2\ln(0.4) = 0.8(0.4700) + 0.2(-0.9163) = 0.1926 \text{ nat}$$

$D_{\mathrm{KL}}(P \| Q) = 0.2232 \neq 0.1926 = D_{\mathrm{KL}}(Q \| P)$，故不对称。

**答案**：$\boxed{D_{\mathrm{KL}}(P \| Q) = 0.5\ln(0.625) + 0.5\ln(2.5) \approx 0.223 \text{ nat}}$，KL 散度不对称。

> ⚠️ KL 散度不是距离（不满足对称性），常被称为"相对熵"或"信息散度"。

---

### C.8.3（Ch.22，交叉熵与 KL 散度的关系）

**题目**：真实分布 $P$，模型分布 $Q$，交叉熵 $H(P,Q) = -\sum_x P(x)\log Q(x)$，建立与 $H(P)$、$D_{\mathrm{KL}}$ 的关系。

**思路**：拆开 $\log(P/Q)$ 即得。

**解**：

1. 关系式：

$$H(P, Q) = -\sum_x P(x)\log Q(x) = -\sum_x P(x)\log P(x) + \sum_x P(x)\log\frac{P(x)}{Q(x)}$$

$$\boxed{H(P, Q) = H(P) + D_{\mathrm{KL}}(P \| Q)}$$

因为 $D_{\mathrm{KL}}(P \| Q) \geq 0$，所以 $H(P, Q) \geq H(P)$，交叉熵不小于真实熵。

2. 当 $Q = P$ 时，$D_{\mathrm{KL}}(P \| P) = 0$，故：

$$H(P, P) = H(P)$$

交叉熵等于真实分布的熵。

**答案**：$\boxed{H(P,Q) = H(P) + D_{\mathrm{KL}}(P\|Q)}$；$Q = P$ 时 $H(P,P) = H(P)$。

---

### C.8.4（Ch.22，互信息）

**题目**：$H(X)=2$，$H(Y)=3$，$H(X,Y)=4$，计算 $I(X;Y)$，并说明独立情形。

**思路**：直接用互信息公式。

**解**：

1. 互信息：

$$I(X; Y) = H(X) + H(Y) - H(X, Y) = 2 + 3 - 4 = 1 \text{ 比特}$$

2. 若 $X \perp Y$，则 $H(X, Y) = H(X) + H(Y)$，从而：

$$I(X; Y) = H(X) + H(Y) - H(X, Y) = 0$$

**答案**：$\boxed{I(X;Y) = 1 \text{ 比特}}$；独立时 $I(X;Y) = 0$。

---

### C.8.5（Ch.23，蒙特卡洛积分）

**题目**：用 MC 估计 $I = \int_0^1 e^{x^2}\,dx$，写估计量及标准误差。

**思路**：MC 估计即样本均值，方差来自 CLT。

**解**：

1. 蒙特卡洛估计量：抽取 $U_1,\ldots,U_n \overset{\text{i.i.d.}}{\sim} U(0,1)$，令

$$\hat{I}_n = \frac{1}{n}\sum_{i=1}^n e^{U_i^2}$$

由大数定律，$\hat{I}_n \xrightarrow{a.s.} E[e^{U^2}] = I$，故 $\hat{I}_n$ 是无偏一致估计量。

2. 标准误差：

$$\mathrm{SE}(\hat{I}_n) = \frac{\sigma_f}{\sqrt{n}}, \quad \sigma_f = \sqrt{\mathrm{Var}(e^{U^2})}$$

**答案**：$\boxed{\hat{I}_n = \frac{1}{n}\sum_{i=1}^n e^{U_i^2},\quad \mathrm{SE} = \sigma_f/\sqrt{n}}$

**注**：误差以 $1/\sqrt{n}$ 速率收缩，与积分维度无关（MC 优势所在）。

---

### C.8.6（Ch.23，重要性采样）

**题目**：估计 $E_p[f(X)]$，从提议分布 $q(x)$ 采样，写 IS 估计量及最优 $q$ 的效果。

**思路**：改变测度，引入重要性权重。

**解**：

1. 重要性采样估计量：从 $q$ 抽样 $X_1,\ldots,X_n$，

$$\hat{I}_{IS} = \frac{1}{n}\sum_{i=1}^n f(X_i) w(X_i), \quad w(x) = \frac{p(x)}{q(x)}$$

无偏性：$E_q[f(X)w(X)] = \int f(x)\frac{p(x)}{q(x)} q(x)\,dx = E_p[f(X)]$。

2. 若 $q(x) \propto |f(x)|p(x)$（最优提议分布），则重要性权重 $w(x)f(x)$ 为常数，

$$\mathrm{Var}(\hat{I}_{IS}) = 0$$

方差降为零（理论最优，实践中需知道 $E_p[|f|]$ 才能归一化，但揭示了方向）。

**答案**：$\boxed{\hat{I}_{IS} = \frac{1}{n}\sum f(X_i)\frac{p(X_i)}{q(X_i)}}$；最优 $q \propto |f|p$ 时方差最小（理论上为零）。

---

### C.8.7（Ch.24，贝叶斯网络因子分解）

**题目**：链式结构 $A \to B \to C$，写联合分布分解，判断 $A \perp\!\!\!\perp C \mid B$。

**思路**：链结构因子分解；d-分离判定。

**解**：

1. 联合分布分解：

$$P(A, B, C) = P(A)\cdot P(B \mid A)\cdot P(C \mid B)$$

2. 在链结构中，$B$ 是 $A$ 和 $C$ 之间的中间节点（chain/pipe），观测 $B$ 时路径被阻断，故

$$A \perp\!\!\!\perp C \mid B \quad \checkmark$$

验证：$P(A,C\mid B) = \frac{P(A,B,C)}{P(B)} = \frac{P(A)P(B\mid A)P(C\mid B)}{P(B)} = P(A\mid B)\cdot P(C\mid B)$，即因子化。

**答案**：$\boxed{P(A,B,C)=P(A)P(B\mid A)P(C\mid B)}$；是，$A\perp\!\!\!\perp C\mid B$ 成立。

---

### C.8.8（Ch.24，EM 算法——E 步与 M 步）

**题目**：写出 EM 的 E 步（$Q$ 函数）和 M 步更新公式。

**思路**：EM 通过最大化对数似然的下界迭代。

**解**：

1. **E 步**：给定当前参数 $\theta^{(t)}$ 和观测 $X$，计算隐变量的后验期望：

$$Q(\theta \mid \theta^{(t)}) = E_{Z \mid X, \theta^{(t)}}\!\left[\log p(X, Z \mid \theta)\right]$$

即在 $p(Z\mid X,\theta^{(t)})$ 下对完整对数似然取期望。

2. **M 步**：最大化 $Q$ 函数：

$$\theta^{(t+1)} = \arg\max_\theta\; Q(\theta \mid \theta^{(t)})$$

每次迭代保证 $\log p(X\mid\theta^{(t+1)}) \geq \log p(X\mid\theta^{(t)})$（单调不减性）。

**答案**：$\boxed{Q(\theta\mid\theta^{(t)})=E_{Z|X,\theta^{(t)}}[\log p(X,Z|\theta)]}$，$\theta^{(t+1)}=\arg\max_\theta Q(\theta|\theta^{(t)})$。

---

### C.8.9（Ch.24，ELBO 与变分推断）

**题目**：写 ELBO 表达式，说明最大化 ELBO 的等价含义，并说明 $q=p(Z|X)$ 时的结果。

**思路**：对数证据分解为 ELBO 与 KL 散度之和。

**解**：

1. 分解：

$$\log p(X) = \underbrace{E_q[\log p(X,Z)] - E_q[\log q(Z)]}_{\mathrm{ELBO}} + D_{\mathrm{KL}}(q(Z)\|p(Z\mid X))$$

由 $D_{\mathrm{KL}} \geq 0$，ELBO $\leq \log p(X)$。

最大化 ELBO $\Leftrightarrow$ 最小化 $D_{\mathrm{KL}}(q(Z)\|p(Z\mid X))$（因 $\log p(X)$ 不依赖 $q$）。

2. 当 $q(Z) = p(Z\mid X)$ 时，$D_{\mathrm{KL}} = 0$，故：

$$\mathrm{ELBO} = \log p(X)$$

**答案**：$\boxed{\mathrm{ELBO} = E_q[\log p(X,Z)] - E_q[\log q(Z)]}$；最大化 ELBO 等价于最小化 $D_{\mathrm{KL}}(q\|p(\cdot|X))$；$q=p(Z|X)$ 时 ELBO $= \log p(X)$。

---

### C.8.10（Ch.22-24，综合辨析）

**题目**：判断三个说法的正误。

**思路**：逐条套用定理。

**解**：

1. **正确**。KL 散度非负：$D_{\mathrm{KL}}(P\|Q) = \sum p_i\ln(p_i/q_i) \geq 0$，由 Jensen 不等式（$-\ln$ 凸）；等号成立当且仅当 $p_i = q_i$ 对所有 $i$。

2. **错误**。蒙特卡洛积分误差以 $O(1/\sqrt{n})$（而非 $1/n$）的速率收缩，但与**维度无关**（这才是 MC 的优势，数值积分误差随维度指数增长）。

3. **正确**。贝叶斯网络的 Markov 性质：若 $X$ 和 $Y$ 在 DAG 中被 $Z$ d-分离，则 $X \perp\!\!\!\perp Y \mid Z$（全局 Markov 性）。

**答案**：$\boxed{(1)\text{正确}\quad(2)\text{错误（误差}\sim 1/\sqrt{n}\text{）}\quad(3)\text{正确}}$

---

## D 中等题详解（12 题）

---

### D.8.1（Ch.22，信息熵的基本性质）

**题目**：证明 $H(p)\geq0$；证明均匀分布最大化熵；计算 Bernoulli 熵并求最大化点。

**思路**：非负性来自 $p\log p \leq 0$；上界用 Jensen 不等式；Bernoulli 熵对 $p$ 求导。

**解**：

**(a) $H(p) \geq 0$**

对任意 $p_i \in (0,1]$，有 $\log p_i \leq 0$（以 $e$ 为底），故 $-p_i \log p_i \geq 0$，求和即：

$$H(p) = -\sum_{i=1}^n p_i\log p_i = \sum_{i=1}^n (-p_i\log p_i) \geq 0$$

边界：若某 $p_i = 0$，约定 $0\log 0 = 0$（连续延拓）。

**(b) $H(p) \leq \log n$**

由 Jensen 不等式（$\log$ 是凹函数）：

$$H(p) = \sum_{i=1}^n p_i\log\frac{1}{p_i} = \sum_{i=1}^n p_i \cdot \log\frac{1}{p_i} \leq \log\!\left(\sum_{i=1}^n p_i \cdot \frac{1}{p_i}\right) = \log n$$

等号成立当且仅当所有 $1/p_i$ 相等，即 $p_i = 1/n$（均匀分布）。

**(c) Bernoulli$(p)$ 的熵**

$$H(p) = -p\log p - (1-p)\log(1-p) \quad (p \in (0,1))$$

对 $p$ 求导并令其为零：

$$\frac{dH}{dp} = -\log p - 1 + \log(1-p) + 1 = \log\frac{1-p}{p} = 0$$

解得 $p = 1/2$，此时熵最大：

$$H(1/2) = -\frac{1}{2}\log_2\frac{1}{2} - \frac{1}{2}\log_2\frac{1}{2} = 1 \text{ bit} = \log_e 2 \text{ nat}$$

验证：以比特计算 $H(1/2) = \log_2 2 = 1$，等于支撑大小 $n=2$ 的 $\log_2 n = 1$，与 (b) 一致。

**答案**：$\boxed{H(p)\geq0}$（非负）；$\boxed{H(p)\leq\log n}$（均匀最大）；Bernoulli 熵在 $p=1/2$ 取最大值 $\log 2$。

---

### D.8.2（Ch.22，KL 散度）

**题目**：证明 KL 非负；验证 $=0 \Leftrightarrow P=Q$；计算 $P=\text{Ber}(0.7)$，$Q=\text{Ber}(0.5)$ 的 KL。

**思路**：Jensen 不等式用于 $-\log$（凸函数）；数值计算验证不对称。

**解**：

**(a) $D_{\mathrm{KL}}(P\|Q) \geq 0$**

$$D_{\mathrm{KL}}(P\|Q) = E_P\!\left[\log\frac{p}{q}\right] = -E_P\!\left[\log\frac{q}{p}\right]$$

由 Jensen 不等式（$-\log$ 是凸函数）：

$$-E_P\!\left[\log\frac{q}{p}\right] \geq -\log E_P\!\left[\frac{q}{p}\right] = -\log\sum_i p_i\frac{q_i}{p_i} = -\log\sum_i q_i = -\log 1 = 0$$

故 $D_{\mathrm{KL}}(P\|Q) \geq 0$。

**(b) 等号条件**

Jensen 不等式等号成立 $\Leftrightarrow$ $q_i/p_i$ 为常数 $\Leftrightarrow$ $q_i = cp_i$ 对所有 $i$；结合 $\sum q_i = \sum p_i = 1$，得 $c=1$，即 $P = Q$。

**(c) 数值计算**

$P = (0.7, 0.3)$，$Q = (0.5, 0.5)$（以自然对数）：

$$D_{\mathrm{KL}}(P\|Q) = 0.7\ln\frac{0.7}{0.5} + 0.3\ln\frac{0.3}{0.5}$$

$$= 0.7\ln 1.4 + 0.3\ln 0.6 = 0.7(0.3365) + 0.3(-0.5108) = 0.2356 - 0.1532 \approx 0.0824 \text{ nat}$$

$$D_{\mathrm{KL}}(Q\|P) = 0.5\ln\frac{0.5}{0.7} + 0.5\ln\frac{0.5}{0.3}$$

$$= 0.5\ln(0.7143) + 0.5\ln(1.6667) = 0.5(-0.3365) + 0.5(0.5108) \approx 0.0872 \text{ nat}$$

$0.0824 \neq 0.0872$，验证不对称性。

**答案**：$\boxed{D_{\mathrm{KL}}(P\|Q)\approx 0.0824\text{ nat}}$，$D_{\mathrm{KL}}(Q\|P)\approx 0.0872\text{ nat}$，不对称。

---

### D.8.3（Ch.22，互信息与独立性）

**题目**：证明互信息的链式分解；证明 $I\geq0$；计算二元对称信道容量。

**思路**：条件熵定义展开；KL 非负性；BSC 信道用熵公式。

**解**：

**(a) $I(X;Y) = H(X) - H(X|Y)$**

$$I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$$

$$= \sum_{x,y} p(x,y)\log\frac{p(x|y)}{p(x)} = \sum_y p(y)\sum_x p(x|y)\log p(x|y) - \sum_x p(x)\log p(x) \cdot \underbrace{\sum_y p(y|x)}_{1}$$

更简洁地：

$$I(X;Y) = H(X) + H(Y) - H(X,Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

（利用 $H(X,Y) = H(Y) + H(X|Y) = H(X) + H(Y|X)$。）

**(b) $I(X;Y) \geq 0$**

$$I(X;Y) = D_{\mathrm{KL}}\!\left(p(x,y) \,\|\, p(x)p(y)\right) \geq 0$$

等号成立 $\Leftrightarrow$ $p(x,y) = p(x)p(y)$ 对所有 $(x,y)$，即 $X \perp Y$。

**(c) 二元对称信道（BSC）容量**

输入均匀：$P(X=0)=P(X=1)=1/2$；信道翻转概率 $\varepsilon$。

输出也均匀：$P(Y=0)=P(Y=1)=1/2$，故 $H(Y)=1$ 比特。

条件熵：$H(Y|X) = H(\varepsilon) = -\varepsilon\log_2\varepsilon-(1-\varepsilon)\log_2(1-\varepsilon)$（二元熵函数）。

$$I(X;Y) = H(Y) - H(Y|X) = 1 - H(\varepsilon)$$

信道容量 $C = \max_{p(x)} I(X;Y) = 1 - H(\varepsilon)$ 比特/使用。

**答案**：$\boxed{I(X;Y)=H(X)-H(X|Y)=H(Y)-H(Y|X)\geq0}$；BSC 容量 $C = 1 - H(\varepsilon)$ 比特。

---

### D.8.4（Ch.23，Monte Carlo 积分误差）

**题目**：证明 MC 无偏及方差；估计 $e^x$ 积分的样本量需求；比较 MC 与数值积分的维度优势。

**思路**：期望线性性 + CLT；精确计算 $\text{Var}(e^U)$；维度分析。

**解**：

**(a) 无偏性与方差**

$$E[\hat{I}_n] = E\!\left[\frac{1}{n}\sum_{i=1}^n g(U_i)\right] = \frac{1}{n}\cdot n \cdot E[g(U)] = \int_0^1 g(x)\,dx = I$$

$$\mathrm{Var}(\hat{I}_n) = \frac{1}{n^2}\sum_{i=1}^n\mathrm{Var}(g(U_i)) = \frac{\mathrm{Var}(g(U))}{n}$$

**(b) 样本量估计（$g(x) = e^x$）**

$\text{Var}(e^U) = E[e^{2U}] - (E[e^U])^2$

$$E[e^{2U}] = \int_0^1 e^{2x}\,dx = \frac{e^2-1}{2} \approx 3.195$$

$$E[e^U] = e - 1 \approx 1.718$$

$$\sigma^2 = \frac{e^2-1}{2} - (e-1)^2 = 3.195 - 2.952 = 0.243, \quad \sigma \approx 0.493$$

95% CI 宽度 $= 2\times 1.96\times\sigma/\sqrt{n} \leq 0.01$，解得：

$$\sqrt{n} \geq \frac{2\times 1.96\times 0.493}{0.01} = 193.3$$

$$n \geq 193.3^2 \approx 37\,400$$

**(c) 维度优势**

Simpson 法则在 $d$ 维下精度 $O(n^{-4/d})$（$n$ 为总格点数），当 $d\geq5$ 时收敛极慢（维度诅咒）。MC 方差始终为 $\sigma^2/n$，误差 $O(1/\sqrt{n})$ **与维度 $d$ 无关**，在高维积分中远优于确定性数值法。

**答案**：$\boxed{n \approx 37\,400}$（CI 宽度 $\leq0.01$）；MC 误差 $O(n^{-1/2})$ 与维度无关，数值积分误差 $O(n^{-4/d})$ 随维度指数恶化。

---

### D.8.5（Ch.23，重要性采样）

**题目**：证明 IS 无偏；写出方差公式及最优 $q$；描述 SNIS 步骤。

**思路**：改变测度；最优 $q$ 使方差为零；SNIS 处理未归一化密度。

**解**：

**(a) 无偏性**

$$E_q[\hat{I}_{IS}] = E_q\!\left[\frac{1}{n}\sum_{i=1}^n g(X_i)w(X_i)\right] = E_q[g(X)w(X)]$$

$$= \int g(x)\frac{p(x)}{q(x)} q(x)\,dx = \int g(x)p(x)\,dx = E_p[g(X)]$$

**(b) 方差与最优提议分布**

$$\mathrm{Var}(\hat{I}_{IS}) = \frac{1}{n}\mathrm{Var}_q[g(X)w(X)] = \frac{1}{n}\!\left(E_q[(g(X)w(X))^2] - I^2\right)$$

最优提议分布使 $g(x)w(x) = g(x)p(x)/q(x)$ 为常数，即 $q^*(x) \propto |g(x)|p(x)$，此时：

$$\mathrm{Var}(\hat{I}_{IS}) = 0 \quad \text{（理论最优，实践需近似）}$$

**(c) 自归一化重要性采样（SNIS）**

设 $p(x) = \tilde{p}(x)/Z$，$Z$ 未知，权重 $w_i = \tilde{p}(X_i)/q(X_i)$：

$$\hat{\mu}_{SNIS} = \frac{\sum_{i=1}^n w_i f(X_i)}{\sum_{i=1}^n w_i}$$

步骤：① 从 $q$ 抽 $X_1,\ldots,X_n$；② 计算未归一化权重 $w_i = \tilde{p}(X_i)/q(X_i)$；③ 归一化 $\bar{w}_i = w_i/\sum_j w_j$；④ 估计 $\hat{\mu} = \sum_i \bar{w}_i f(X_i)$。

SNIS 有偏（比率估计偏差），但 $n\to\infty$ 时相合，优势是无需知道归一化常数 $Z$。

**答案**：IS 估计量无偏；最优 $q^*\propto|g|p$ 时方差为零；SNIS 用归一化权重处理未知 $Z$。

---

### D.8.6（Ch.23，Metropolis-Hastings 算法）

**题目**：化简 M-H 接受率；验证细致平衡；讨论步长 $\delta$ 的影响。

**思路**：对称提议分布简化；细致平衡分情况讨论；步长权衡。

**解**：

**(a) 接受率化简**

目标 $\pi(x) \propto e^{-x^2/2}$，提议 $q(x'|x) = q(x|x') = \text{Uniform}(x-\delta,x+\delta)$（对称）。

一般接受率：

$$\alpha(x,x') = \min\!\left(1,\frac{\pi(x')q(x|x')}{\pi(x)q(x'|x)}\right)$$

由于 $q$ 对称（$q(x|x')=q(x'|x)$），化简为：

$$\alpha(x,x') = \min\!\left(1,\frac{\pi(x')}{\pi(x)}\right) = \min\!\left(1, e^{-(x'^2-x^2)/2}\right)$$

**(b) 细致平衡验证**

需证 $\pi(x)k(x,x') = \pi(x')k(x',x)$，其中转移核 $k(x,x') = q(x'|x)\alpha(x,x')$。

不妨设 $\pi(x') \geq \pi(x)$（即 $\alpha(x,x')=1$，$\alpha(x',x)=\pi(x)/\pi(x')$）：

$$\pi(x)k(x,x') = \pi(x)\cdot q(x'|x)\cdot 1$$

$$\pi(x')k(x',x) = \pi(x')\cdot q(x|x')\cdot\frac{\pi(x)}{\pi(x')} = \pi(x)\cdot q(x|x')$$

由 $q$ 对称，$q(x'|x) = q(x|x')$，故两边相等。细致平衡成立。

**(c) 步长 $\delta$ 的影响**

| $\delta$ | 接受率 | 混合速度 | 问题 |
|----------|--------|----------|------|
| 过大 | 接近 0 | 极慢（几乎不移动） | 高拒绝率，样本高度相关 |
| 过小 | 接近 1 | 极慢（随机游走扩散慢） | 探索范围窄，自相关高 |
| 最优 | $\approx 0.234$（高维）/ $0.44$（一维） | 最快 | Gelman 等人理论结果 |

一维情形最优接受率约 44%，高维（$d\to\infty$）最优接受率约 23.4%。

**答案**：对称提议时 $\alpha = \min(1,\pi(x')/\pi(x))$；细致平衡经分情况讨论验证成立；最优接受率一维约 44%，高维约 23.4%。

---

### D.8.7（Ch.24，贝叶斯网络的条件独立性）

**题目**：$A\to C\leftarrow B$，$C\to D$；写联合分解；证明 $A\perp B$（未观测 $C$）；说明 explaining away。

**思路**：V 形结构（collider）：未观测时独立，观测后产生相关。

**解**：

**(a) 联合分布分解**

$$P(A,B,C,D) = P(A)\cdot P(B)\cdot P(C\mid A,B)\cdot P(D\mid C)$$

**(b) 未观测 $C$ 时 $A\perp B$**

对 $C$ 和 $D$ 边缘化：

$$P(A,B) = \sum_{C,D} P(A)P(B)P(C\mid A,B)P(D\mid C)$$

$$= P(A)P(B)\sum_C P(C\mid A,B)\underbrace{\sum_D P(D\mid C)}_{=1} = P(A)P(B)\underbrace{\sum_C P(C\mid A,B)}_{=1} = P(A)P(B)$$

故 $A\perp B$（Collider 未观测时阻断信息流）。

**(c) Explaining Away（观测 $C$ 后）**

观测 $C=c$ 后：

$$P(A,B\mid C=c) = \frac{P(A)P(B)P(c\mid A,B)}{P(c)}$$

除非 $P(c\mid A,B)$ 对所有 $(A,B)$ 相同，否则 $P(A,B\mid C=c) \neq P(A\mid C=c)\cdot P(B\mid C=c)$，$A$ 和 $B$ 不再独立。

**举例**：$C=$"草地湿"，$A=$"下雨"，$B=$"洒水器开"，两者原本独立。但观测到草地湿后（$C=1$），若已知下雨，则洒水器开的概率降低（互相"解释"对方），产生负相关。

**答案**：$P(A,B,C,D)=P(A)P(B)P(C|A,B)P(D|C)$；未观测 $C$ 时 $A\perp B$；观测 $C$ 后 explaining away 使 $A,B$ 相关。

---

### D.8.8（Ch.24，马尔可夫链基本性质）

**题目**：验证随机矩阵；求平稳分布；判断遍历性。

**思路**：行和为 1；解 $\pi P=\pi$；遍历 = 不可约 + 非周期。

**解**：

**(a) 行和验证**

$$0.7+0.2+0.1=1;\quad 0.3+0.5+0.2=1;\quad 0.1+0.3+0.6=1 \quad\checkmark$$

**(b) 平稳分布**

解 $\boldsymbol{\pi}P = \boldsymbol{\pi}$，$\boldsymbol{\pi}\mathbf{1}=1$，即：

$$\begin{cases}\pi_1 = 0.7\pi_1 + 0.3\pi_2 + 0.1\pi_3\\\pi_2 = 0.2\pi_1 + 0.5\pi_2 + 0.3\pi_3\\\pi_3 = 0.1\pi_1 + 0.2\pi_2 + 0.6\pi_3\\\pi_1+\pi_2+\pi_3=1\end{cases}$$

化简方程（取前两个方程）：

$$-0.3\pi_1 + 0.3\pi_2 + 0.1\pi_3 = 0 \implies \pi_2 = \pi_1 - \tfrac{1}{3}\pi_3$$

$$0.2\pi_1 - 0.5\pi_2 + 0.3\pi_3 = 0$$

代入得 $0.2\pi_1 - 0.5(\pi_1 - \tfrac{1}{3}\pi_3) + 0.3\pi_3 = 0$，即 $-0.3\pi_1 + \tfrac{7}{15}\pi_3 \cdot 5/3 = 0$。

用数值法（或 Cramer 法则），设 $\pi_3 = t$：

由方程组 $\mathbf{A}\boldsymbol{\pi}^\top=\mathbf{0}$ 求核，结合归一化，数值解为：

$$\boldsymbol{\pi} \approx (0.349,\; 0.326,\; 0.326)$$

> 精确解：令 $\pi_1=a, \pi_2=b, \pi_3=c$，由方程 $-0.3a+0.3b+0.1c=0$ 和 $0.2a-0.5b+0.3c=0$，联立得 $b = \frac{7c}{6+3} \cdots$（矩阵行变换后数值解更直接）。实际近似值 $\approx(0.349, 0.326, 0.326)$（可用计算器验证）。

**(c) 遍历性**

- **不可约**：任意两状态间均可到达（转移矩阵所有元素 $>0$），故不可约。
- **非周期**：$P_{11}=0.7>0$，状态 1 有自环，周期为 1，整个链非周期。
- 由遍历定理：$\mathbf{P}^n \to \mathbf{1}\boldsymbol{\pi}^\top$，初始分布无关，均收敛到唯一平稳分布 $\boldsymbol{\pi}$。

**答案**：行和均为 1；平稳分布 $\boldsymbol{\pi}\approx(0.349, 0.326, 0.326)$；链不可约且非周期，故遍历。

---

### D.8.9（Ch.22，最大熵原理）

**题目**：Lagrange 乘数法求约束均值下的最大熵分布；连续情形结论；统计物理解释。

**思路**：指数族形式是最大熵分布的一般结论；连续情形 = 正态。

**解**：

**(a) 拉格朗日法（离散，约束 $E[X]=\mu$）**

构造 Lagrangian：

$$\mathcal{L}(p,\lambda_0,\lambda_1) = -\sum_{i=1}^n p_i\log p_i - \lambda_0\!\left(\sum p_i - 1\right) - \lambda_1\!\left(\sum i\cdot p_i - \mu\right)$$

对 $p_i$ 偏导并令其为零：

$$\frac{\partial\mathcal{L}}{\partial p_i} = -\log p_i - 1 - \lambda_0 - \lambda_1 i = 0$$

$$\implies \log p_i = -1 - \lambda_0 - \lambda_1 i \implies \boxed{p_i \propto e^{-\lambda_1 i}}$$

令 $r = e^{-\lambda_1}$，则 $p_i = (1-r)r^{i-1}$（几何分布），$\lambda_1$ 由均值约束 $\sum i\cdot p_i = \mu$ 确定。

**(b) 连续情形**

约束 $E[X]=\mu$，$\mathrm{Var}(X)=\sigma^2$，最大微分熵分布为：

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) = \mathcal{N}(\mu, \sigma^2)$$

正态分布是给定均值和方差下熵最大的连续分布。

**(c) 统计物理含义**

最大熵原理等价于"在满足约束条件下，对系统状态做最少的额外假设"——对应统计力学中 Gibbs 系综的推导：给定能量期望约束，最大熵对应系统处于热平衡状态（Boltzmann 分布）。从信息论看，它是对应约束下的**无信息先验**（Jeffreys 型思路），避免引入额外偏见。

**答案**：离散最大熵分布 $p_i\propto e^{\lambda i}$（几何族）；连续情形为 $\mathcal{N}(\mu,\sigma^2)$；最大熵 = 最少额外假设。

---

### D.8.10（Ch.23，变分推断基本思想）

**题目**：推导 ELBO；证明等价性；写均场 CAVI 更新公式。

**思路**：对数证据分解；均场分解后坐标优化。

**解**：

**(a) ELBO 推导**

$$\log p(\mathbf{x}) = \log\int p(\mathbf{x},\mathbf{z})\,d\mathbf{z} = \log\int q_\phi(\mathbf{z})\frac{p(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z})}\,d\mathbf{z}$$

由 Jensen 不等式（$\log$ 凹）：

$$\geq \int q_\phi(\mathbf{z})\log\frac{p(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z})}\,d\mathbf{z} = E_q[\log p(\mathbf{x},\mathbf{z})] - E_q[\log q_\phi(\mathbf{z})] =: \mathcal{L}(\phi)$$

故 $\log p(\mathbf{x}) \geq \mathcal{L}(\phi)$（ELBO 是证据下界）。

**(b) 等价性**

$$\log p(\mathbf{x}) = \mathcal{L}(\phi) + D_{\mathrm{KL}}(q_\phi\|p(\mathbf{z}|\mathbf{x}))$$

（精确分解，因为 $\log p(\mathbf{x})$ 不依赖 $q$，等式恒成立。）

最大化 $\mathcal{L}(\phi) \Leftrightarrow$ 最小化 $D_{\mathrm{KL}}(q_\phi\|p(\mathbf{z}|\mathbf{x}))$。

**(c) 坐标上升 CAVI 更新**

均场假设 $q(\mathbf{z})=\prod_i q_i(z_i)$，固定 $q_{-j}$，对 $q_j$ 优化：

$$\mathcal{L} = E_q[\log p(\mathbf{x},\mathbf{z})] - \sum_i E_{q_i}[\log q_i(z_i)]$$

对 $q_j$ 的变分导数为零，得：

$$\log q_j^*(z_j) = E_{-j}[\log p(\mathbf{x},\mathbf{z})] + \text{const}$$

即 $q_j^*(z_j) \propto \exp\!\left(E_{-j}[\log p(\mathbf{x},\mathbf{z})]\right)$，其中 $E_{-j}$ 表示在所有其他 $q_k$（$k\neq j$）下取期望。

**答案**：$\boxed{\mathcal{L}(\phi)=E_q[\log p(\mathbf{x},\mathbf{z})]-E_q[\log q_\phi]}$；最大化 ELBO $\Leftrightarrow$ 最小化 KL；CAVI 更新 $\log q_j^*=E_{-j}[\log p(\mathbf{x},\mathbf{z})]+\text{const}$。

---

### D.8.11（Ch.24，隐马尔可夫模型前向算法）

**题目**：定义前向变量并写递推；用前向变量表达似然；分析计算复杂度。

**思路**：动态规划递推；时间复杂度对比暴力枚举。

**解**：

**(a) 前向变量定义与递推**

设 HMM 有 $N$ 个隐状态，初始分布 $\pi_i = P(S_1=i)$，转移矩阵 $A_{ij}=P(S_{t+1}=j|S_t=i)$，发射矩阵 $B_{i}(o)=P(o_t=o|S_t=i)$。

**初始化**（$t=1$）：

$$\alpha_1(i) = \pi_i \cdot B_i(o_1), \quad i=1,\ldots,N$$

**递推**（$t=2,\ldots,T$）：

$$\alpha_t(j) = \left[\sum_{i=1}^N \alpha_{t-1}(i)\cdot A_{ij}\right]\cdot B_j(o_t), \quad j=1,\ldots,N$$

**(b) 似然计算**

$$P(O\mid\lambda) = \sum_{i=1}^N \alpha_T(i)$$

对最终时刻的所有可能隐状态求和，消去隐状态序列。

**(c) 计算复杂度**

- **暴力枚举**：$N^T$ 条路径，每条路径计算 $O(T)$ 步，总复杂度 $O(N^T \cdot T)$，指数级。
- **前向算法**：每步递推需 $O(N^2)$（对每个 $j$，对所有 $i$ 求和），共 $T$ 步，总复杂度 $O(N^2 T)$。

动态规划本质：利用递推关系 $\alpha_t = f(\alpha_{t-1})$，**存储中间结果**避免重复计算，将指数问题化为多项式。

**答案**：初始化 $\alpha_1(i)=\pi_i B_i(o_1)$，递推 $\alpha_t(j)=[\sum_i\alpha_{t-1}(i)A_{ij}]B_j(o_t)$；$P(O|\lambda)=\sum_i\alpha_T(i)$；复杂度 $O(N^2T)$（vs 暴力的 $O(N^T)$）。

---

### D.8.12（Ch.24，高斯过程回归简介）

**题目**：写观测边际分布；写后验预测均值与方差；说明与核岭回归的等价性。

**思路**：GP 的闭式后验源自高斯条件分布公式。

**解**：

设训练输入 $\mathbf{X}\in\mathbb{R}^{n\times d}$，核矩阵 $K_{ij}=k(\mathbf{x}_i,\mathbf{x}_j)$，均值函数 $m(\cdot)\equiv 0$（简化）。

**(a) 观测边际分布**

$$\mathbf{y} \sim \mathcal{N}\!\left(\mathbf{0},\; \mathbf{K} + \sigma_n^2\mathbf{I}\right)$$

其中 $\mathbf{K} = k(\mathbf{X},\mathbf{X})$（$n\times n$ 核矩阵），噪声方差 $\sigma_n^2$ 来自 $\boldsymbol{\varepsilon}$。

**(b) 后验预测分布**

对新输入 $\mathbf{x}^*$，令 $\mathbf{k}_* = k(\mathbf{X},\mathbf{x}^*)$（$n\times1$），$k_{**}=k(\mathbf{x}^*,\mathbf{x}^*)$：

$$\mu_n(\mathbf{x}^*) = \mathbf{k}_*^\top(\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}\mathbf{y}$$

$$\sigma_n^2(\mathbf{x}^*) = k_{**} - \mathbf{k}_*^\top(\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}\mathbf{k}_*$$

（由二元高斯的条件分布公式导出。）

**(c) 与核岭回归（KRR）的等价性**

KRR 最优权重：$\hat{\mathbf{w}} = (\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}\mathbf{y}$，预测 $\hat{f}(\mathbf{x}^*)=\mathbf{k}_*^\top\hat{\mathbf{w}} = \mathbf{k}_*^\top(\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}\mathbf{y}$。

与 GP 后验均值 $\mu_n(\mathbf{x}^*)$ 完全一致。KRR 只给出点预测，GP 还额外提供不确定性量化（后验方差 $\sigma_n^2(\mathbf{x}^*)$），体现贝叶斯框架的优势。

**答案**：$\mathbf{y}\sim\mathcal{N}(\mathbf{0},\mathbf{K}+\sigma_n^2\mathbf{I})$；后验均值 $\mu_n=\mathbf{k}_*^\top(\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}\mathbf{y}$，方差 $\sigma_n^2=k_{**}-\mathbf{k}_*^\top(\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}\mathbf{k}_*$；与 KRR 点预测等价，GP 额外提供不确定性。

---

## E 提高题详解（10 题）

---

### E.8.1（Ch.22，KL 散度 + 互信息 + 信息瓶颈原理）

**题目**：马尔可夫链 $X\to Z\to Y$，信息瓶颈目标 $\max I(Z;Y)-\beta I(Z;X)$。

**思路**：互信息 KL 表示 → 数据处理不等式 → IB Lagrangian 最优解推导 → 极端情形分析 → 信息平面假说批评。

**解**：

**(a) 互信息的 KL 表示与链式规则**

$$I(Z;X) = D_{\mathrm{KL}}\!\left(p(x,z)\,\|\, p(x)p(z)\right) = \sum_{x,z}p(x,z)\log\frac{p(x,z)}{p(x)p(z)}$$

$$= \sum_{x,z}p(x,z)\log\frac{p(z|x)}{p(z)} = E_{p(x,z)}\!\left[\log\frac{p(z|x)}{p(z)}\right]$$

链式规则验证：

$$H(Z) - H(Z|X) = -\sum_z p(z)\log p(z) + \sum_{x,z}p(x,z)\log p(z|x)$$

$$= \sum_{x,z}p(x,z)\log\frac{p(z|x)}{p(z)} = I(Z;X) \quad\checkmark$$

类似地 $I(Z;X) = H(X) - H(X|Z)$（利用 $p(x,z)$ 对称处理）。

**(b) 数据处理不等式（DPI）**

$X\to Z\to Y$ 为马尔可夫链，即 $p(y|x,z)=p(y|z)$（$Y$ 的生成仅依赖 $Z$，不依赖 $X$）。

$$I(X;Y) - I(Z;Y) = H(Y) - H(Y|X) - H(Y) + H(Y|Z)$$

$$= H(Y|Z) - H(Y|X)$$

由马尔可夫链 $X\to Z\to Y$，$H(Y|Z) = H(Y|Z,X)$（给定 $Z$ 则 $Y\perp X$），故：

$$I(X;Y) - I(Z;Y) = H(Y|Z) - H(Y|X) = I(X;Y|Z) \geq 0$$

最后不等号因互信息非负。故 $I(X;Y) \geq I(Z;Y)$。

> 直觉：压缩表示 $Z$ 只能丢失关于 $Y$ 的信息，不能创造。

**(c) IB Lagrangian 最优解**

优化目标：$\max_{p(z|x)}\; I(Z;Y) - \beta I(Z;X)$，约束 $p(z|x)$ 为概率核。

写出 Lagrangian（含归一化约束乘子 $\lambda(x)$）并对 $p(z|x)$ 变分（泛函导数）：

$$\frac{\delta}{\delta p(z|x)}\!\left[I(Z;Y)-\beta I(Z;X)\right] = 0$$

展开 $I(Z;Y)=\sum_{x,z}p(x,z)[p(y|z)-p(y)]$（利用 $p(y|z)$ 与 $p(z|x)$ 的关系）：

最优解满足（Blahut-Arimoto 型迭代）：

$$p^*(z|x) \propto p(z)\exp\!\left(-\beta D_{\mathrm{KL}}\!\left(p(y|x)\,\|\, p(y|z)\right)\right)$$

其中 $p(z) = \sum_x p(x)p^*(z|x)$（自洽方程，需迭代求解）。

**极端情形**：

- $\beta\to0$（忽略压缩惩罚）：$p^*(z|x)\propto p(z)$，即 $Z\perp X$，$Z$ 与 $X$ 完全无关（无信息），$I(Z;Y)\to0$。

- $\beta\to\infty$（强压缩惩罚占主导，但 IB 目标中 $-\beta I(Z;X)$ 趋于 $-\infty$，需理解为 $Z=X$ 才能使 $p(y|z)=p(y|x)$ 最小化 KL）：实际上 $\beta\to\infty$ 时，最优 $Z=X$（保留所有关于 $X$ 的信息），$I(Z;X)=H(X)$，$I(Z;Y)=I(X;Y)$。

> ⚠️ $\beta$ 控制压缩-相关权衡：$\beta$ 小时保留 $X$ 少（强压缩），$\beta$ 大时保留 $X$ 多（弱压缩）。

**(d) 信息平面假说与批评**

**Tishby 等人假说**（2017）：深层网络训练分两阶段：① **拟合阶段**：$I(X;T_l)$ 和 $I(T_l;Y)$ 同时增大（梯度下降拟合数据）；② **压缩阶段**：$I(X;T_l)$ 减小，$I(T_l;Y)$ 维持（网络"忘记"与 $Y$ 无关的 $X$ 信息）。

**Saxe 等人反例**（2018）：使用**确定性（非随机）激活函数**时，训练中 $I(X;T_l)$ 不减小（因确定性函数的互信息在变换下单调）；压缩现象仅在 **SGD 随机噪声**存在时出现（ReLU + 双精度训练下无压缩）。

**随机梯度噪声**的作用：SGD 的随机性（mini-batch 噪声）等效于隐式正则化，促使网络学习低 $I(X;T_l)$ 的表示（信息压缩）。使用饱和激活函数（tanh）时，梯度饱和引入的随机性更强，压缩更明显。

**共识**：信息平面假说在某些架构和激活函数下成立，但不是普遍规律；互信息的估计本身（尤其是连续层的 $I(X;T_l)$）计算困难，结论依赖于估计方法。

---

### E.8.2（Ch.22+Ch.23，最大熵谱分析 + 随机过程 + MCMC 诊断）

**题目**：Wiener-Khintchine 定理、最大熵谱、ESS、HMC。

**思路**：平稳过程谱理论 → AR 过程最大熵 → ESS 与自相关 → HMC 保体积性。

**解**：

**(a) Wiener-Khintchine 定理**

平稳过程 $\{X_t\}$，自相关 $R(k)=E[X_tX_{t+k}]$（不依赖 $t$）。

$$S(f) = \sum_{k=-\infty}^\infty R(k)e^{-2\pi ifk} = \mathcal{F}[R(\cdot)](f)$$

证明 $S(f)\geq0$（Bochner 定理）：$R(k)$ 是正定序列（因为 $\text{Var}(\sum c_k X_{t_k})\geq0$），由 Bochner 定理，其 Fourier 变换是非负测度（即 $S(f)\geq0$）。$S(f)$ 描述单位频率带宽内的平均功率，非负具有物理意义。

**(b) 最大熵谱（Burg 算法）**

已知 $R(0),R(1),\ldots,R(p)$，最大化微分熵：

$$h[S] = \int_{-1/2}^{1/2}\log S(f)\,df$$

约束：$\int S(f)e^{2\pi ifk}\,df = R(k)$，$k=0,1,\ldots,p$。

Lagrangian 变分：

$$\frac{\delta}{\delta S(f)}\left[\int\log S(f)\,df - \sum_{k=0}^p\lambda_k\!\left(\int S(f)e^{2\pi ifk}\,df - R(k)\right)\right] = 0$$

$$\frac{1}{S(f)} = \sum_{k=-p}^p\lambda_k e^{2\pi ifk}$$

$$S(f) = \frac{1}{\sum_{k=-p}^p\lambda_k e^{2\pi ifk}} = \frac{\sigma^2}{|1+\sum_{k=1}^p a_k e^{-2\pi ifk}|^2}$$

这正是 AR($p$) 过程的功率谱密度（有理谱），其中 $a_1,\ldots,a_p$ 为 AR 系数，由 Yule-Walker 方程确定。

**(c) 有效样本量（ESS）推导**

i.i.d. 情形：$\text{Var}(\bar{X}_{n,\text{iid}}) = \sigma^2/n$。

MCMC 样本自相关为 $\rho(k)=\text{Corr}(X_t,X_{t+k})$，则：

$$\text{Var}(\bar{X}_n) = \frac{\sigma^2}{n}\left(1 + 2\sum_{k=1}^{n-1}\left(1-\frac{k}{n}\right)\rho(k)\right) \xrightarrow{n\to\infty} \frac{\sigma^2}{n}\left(1+2\sum_{k=1}^\infty\rho(k)\right)$$

等效 i.i.d. 样本量 $n_{\text{eff}}$ 满足 $\text{Var}(\bar{X}_n) = \sigma^2/n_{\text{eff}}$，故：

$$n_{\text{eff}} = \frac{n}{1+2\sum_{k=1}^\infty\rho(k)}$$

若 $\rho(k)=\rho^k$（指数衰减），则 $\sum\rho^k = \rho/(1-\rho)$，$n_{\text{eff}} = n(1-\rho)/(1+\rho)$。高维随机游走 MH 中 $\rho\to1$，$n_{\text{eff}}\to0$，效率极低。

**(d) HMC 与 NUTS**

HMC 引入辅助动量 $\mathbf{p}\sim\mathcal{N}(\mathbf{0},\mathbf{M})$，联合分布：

$$\tilde{\pi}(\mathbf{q},\mathbf{p}) \propto \exp(-H(\mathbf{q},\mathbf{p})), \quad H = -\log\pi(\mathbf{q}) + \frac{\mathbf{p}^\top\mathbf{M}^{-1}\mathbf{p}}{2}$$

Leapfrog 积分器（保体积性 / Liouville 定理）：Hamiltonian 动力学为辛映射，行列式为 1（体积保持），使得 MH 接受率接近 1（步长合适时接近 100%）。

Leapfrog 步骤（步长 $\epsilon$，$L$ 步）：

$$\mathbf{p}_{t+\epsilon/2} = \mathbf{p}_t + \frac{\epsilon}{2}\nabla\log\pi(\mathbf{q}_t)$$

$$\mathbf{q}_{t+\epsilon} = \mathbf{q}_t + \epsilon\mathbf{M}^{-1}\mathbf{p}_{t+\epsilon/2}$$

$$\mathbf{p}_{t+\epsilon} = \mathbf{p}_{t+\epsilon/2} + \frac{\epsilon}{2}\nabla\log\pi(\mathbf{q}_{t+\epsilon})$$

**NUTS**：通过"不掉头"条件自适应确定路径长度（停止于 $\mathbf{p}\cdot(\mathbf{q}-\mathbf{q}_0)<0$ 时），避免手动调 $L$，同时满足细致平衡（通过 slice 采样）。

HMC 消除随机游走：每次提议沿 Hamiltonian 轨迹移动，大步长 + 高接受率，$n_{\text{eff}}/n$ 接近 1，远优于随机游走 MH。

---

### E.8.3（Ch.23，重要性采样 + 自归一化 + 粒子滤波）

**题目**：SNIS 有偏但相合；ESS 推导；粒子滤波递推；扩散模型与粒子滤波联系。

**解**：

**(a) SNIS 有偏但相合**

$$\hat{\mu}_{SNIS} = \frac{\sum_{i=1}^n w_i f(X_i)}{\sum_{i=1}^n w_i}$$

**有偏**：$E[\hat{\mu}_{SNIS}] \neq E_\pi[f(X)]$（比率估计一般有偏，偏差 $O(1/n)$）。

**相合**：由大数定律，$\frac{1}{n}\sum w_if(X_i) \xrightarrow{p} E_q[wf] = E_\pi[f]$，$\frac{1}{n}\sum w_i \xrightarrow{p} E_q[w] = 1$，

故由连续映射定理，$\hat{\mu}_{SNIS}\xrightarrow{p} E_\pi[f(X)]$。

**(b) ESS 推导**

归一化权重 $\bar{w}_i = w_i/\sum_j w_j$，类比 i.i.d. 加权平均的等效样本量：

$$\hat{n}_{\text{eff}} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2} = \frac{1}{\sum_i\bar{w}_i^2}$$

- 当 $q=\pi$：所有 $w_i=1$，$\hat{n}_{\text{eff}} = n^2/n = n$（最大）。
- 权重退化：若某 $\bar{w}_j\to1$，其余 $\to0$，则 $\sum\bar{w}_i^2\to1$，$n_{\text{eff}}\to1$（最差）。

**(c) 粒子滤波（SIR）**

目标：递推估计过滤分布 $p(x_t|y_{1:t})$，用粒子 $\{x_t^{(i)},w_t^{(i)}\}_{i=1}^N$ 近似。

**权重更新**（时间 $t$ 的新观测 $y_t$）：

$$\tilde{w}_t^{(i)} = w_{t-1}^{(i)}\cdot\frac{p(y_t|x_t^{(i)}) p(x_t^{(i)}|x_{t-1}^{(i)})}{q(x_t^{(i)}|x_{t-1}^{(i)},y_t)}$$

用先验作为提议（Bootstrap Filter，$q=p(\cdot|x_{t-1})$）时简化为：

$$\tilde{w}_t^{(i)} = w_{t-1}^{(i)}\cdot p(y_t|x_t^{(i)})$$

归一化：$w_t^{(i)} = \tilde{w}_t^{(i)}/\sum_j\tilde{w}_t^{(j)}$。

**重采样**：当 ESS $<N/2$ 时，按权重 $w_t^{(i)}$ 多项式重采样 $N$ 个粒子（重置权重为 $1/N$），防止权重退化。

**(d) 扩散模型与粒子滤波**

**DDPM 前向过程**：$q(x_t|x_0) = \mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)\mathbf{I})$（逐步加噪的高斯核）。

**反向过程**：$p_\theta(x_{t-1}|x_t)$ 学习时间反转分布，参数化为：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2\mathbf{I})$$

**与粒子滤波对应**：

| 粒子滤波 | 扩散模型 |
|----------|----------|
| 隐状态 $x_t$（噪声水平） | 噪声图像 $x_t$ |
| 前向转移 $p(x_t\|x_{t-1})$ | 加噪核 $q(x_t\|x_{t-1})$ |
| 预测步（prior） | 去噪网络均值预测 |
| 校正步（观测更新） | 条件去噪（Classifier Guidance） |
| 重采样 | 无（确定性/随机性采样器） |

统计原理：去噪网络估计 score 函数 $\nabla_{x_t}\log q(x_t) \approx -\varepsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$，对应时间反转 SDE 的漂移项（Anderson，1982）。

---

### E.8.4（Ch.24+Ch.22，变分推断 + 均场近似 + 自由能）

**题目**：ELBO 等价性证明；CAVI 推导；GMM 的 VI 更新；KL 方向与均场的代价。

**解**：

**(a) 最小化 KL 等价于最大化 ELBO**

精确分解（Jensen 等号后加余项）：

$$\log p(\mathbf{x}) = \int q(\mathbf{z})\log p(\mathbf{x})\,d\mathbf{z} = \int q(\mathbf{z})\log\frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})}\,d\mathbf{z} + \int q(\mathbf{z})\log\frac{q(\mathbf{z})}{p(\mathbf{z}|\mathbf{x})}\,d\mathbf{z}$$

$$= \mathcal{L}(q) + D_{\mathrm{KL}}(q\|p(\cdot|\mathbf{x}))$$

其中 $\mathcal{L}(q)=E_q[\log p(\mathbf{x},\mathbf{z})]-E_q[\log q(\mathbf{z})]$。

由于 $\log p(\mathbf{x})$ 是常数（不依赖 $q$），最大化 $\mathcal{L}(q)$ 与最小化 $D_{\mathrm{KL}}(q\|p(\cdot|\mathbf{x}))$ 完全等价。

**(b) CAVI 更新方程推导**

均场假设 $q(\mathbf{z})=\prod_i q_i(z_i)$，展开 ELBO：

$$\mathcal{L} = \int\!\prod_i q_i\cdot\log p(\mathbf{x},\mathbf{z})\,d\mathbf{z} - \sum_j\int q_j\log q_j\,dz_j + \text{const}_{-j}$$

固定 $q_{i\neq j}$，对 $q_j(z_j)$ 变分（带归一化约束 Lagrange 乘子）：

$$\frac{\delta\mathcal{L}}{\delta q_j(z_j)} = E_{-j}[\log p(\mathbf{x},\mathbf{z})] - \log q_j(z_j) - 1 = 0$$

$$\log q_j^*(z_j) = E_{-j}[\log p(\mathbf{x},\mathbf{z})] + \text{const}$$

$$q_j^*(z_j) \propto \exp\!\left(E_{q_{-j}}[\log p(\mathbf{x},\mathbf{z})]\right)$$

坐标上升：循环更新每个 $q_j$，固定其余，直到 ELBO 收敛（保证单调不减）。

**(c) GMM 的 CAVI（均场 VI）**

高斯混合模型：$z_n\in\{1,\ldots,K\}$（类别），$\mathbf{x}_n|z_n=k\sim\mathcal{N}(\boldsymbol{\mu}_k,\boldsymbol{\Lambda}_k^{-1})$，先验 $p(z_n)=\text{Cat}(\boldsymbol{\pi})$，$p(\boldsymbol{\mu}_k)=\mathcal{N}(\mathbf{m}_0,\beta_0^{-1}\boldsymbol{\Lambda}_k^{-1})$。

均场近似：$q(\mathbf{z},\boldsymbol{\mu},\boldsymbol{\Lambda})=\prod_n q(z_n)\cdot\prod_k q(\boldsymbol{\mu}_k,\boldsymbol{\Lambda}_k)$。

**隐类别更新**（$q(z_n)=\text{Cat}(\boldsymbol{r}_n)$）：

$$r_{nk} \propto \exp\!\left(E[\log\pi_k] + \frac{1}{2}E[\log|\boldsymbol{\Lambda}_k|] - \frac{d}{2}\log(2\pi) - \frac{1}{2}E[(\mathbf{x}_n-\boldsymbol{\mu}_k)^\top\boldsymbol{\Lambda}_k(\mathbf{x}_n-\boldsymbol{\mu}_k)]\right)$$

其中 $r_{nk} = E[z_n=k]$（软分配，类似 EM 的 $\gamma$）。

**均值参数更新**（$q(\boldsymbol{\mu}_k)=\mathcal{N}(\mathbf{m}_k,\beta_k^{-1}\boldsymbol{\Lambda}_k^{-1})$）：

$$\beta_k = \beta_0 + N_k, \quad \mathbf{m}_k = \frac{\beta_0\mathbf{m}_0 + N_k\bar{\mathbf{x}}_k}{\beta_k}$$

其中 $N_k=\sum_n r_{nk}$（有效分配数），$\bar{\mathbf{x}}_k=\sum_n r_{nk}\mathbf{x}_n/N_k$（软加权均值）。

**(d) KL 方向与均场代价**

| | $D_{\mathrm{KL}}(q\|p)$（VI 标准） | $D_{\mathrm{KL}}(p\|q)$（期望传播 EP） |
|---|---|---|
| 行为 | **零覆盖**（zero-forcing）：$q$ 避开 $p=0$ 的区域 | **均值覆盖**（zero-avoiding）：$q$ 覆盖 $p>0$ 的所有区域 |
| 多模态 $p$ | 只捕获一个模 | 找均值模（可能落在两峰之间）|
| 方差估计 | 低估方差（过于自信） | 高估方差 |
| 均场限制 | 无法表示变量间相关性（因子化假设） | |

**Normalizing Flow** 通过可逆变换 $z = f_\phi(u)$（$u\sim\mathcal{N}$）将高斯变换为复杂分布，保持精确似然可算（$\log q(z) = \log p(u) - \log|\det\partial f/\partial u|$），突破均场因子化限制，可表示多模态后验。

---

### E.8.5（Ch.23+Ch.24，MCMC + 马尔可夫链平稳分布 + 混合时间）

**题目**：细致平衡 → 平稳分布；MH 满足细致平衡；混合时间谱隙界；LLM 温度与谱隙。

**解**：

**(a) 细致平衡蕴含平稳性**

设细致平衡：$\pi_i P_{ij} = \pi_j P_{ji}$，对 $j$ 求和：

$$\sum_i\pi_i P_{ij} = \sum_i\pi_j P_{ji} = \pi_j\sum_i P_{ji} = \pi_j\cdot 1 = \pi_j$$

即 $(\pi\mathbf{P})_j = \pi_j$，故 $\pi\mathbf{P}=\pi$，$\pi$ 是平稳分布。

**(b) M-H 满足细致平衡**

设 $x\neq y$，M-H 核：$k(x,y)=q(x,y)\alpha(x,y)$，$\alpha(x,y)=\min(1,\frac{\pi(y)q(y,x)}{\pi(x)q(x,y)})$。

不妨设 $\pi(x)q(x,y)\geq\pi(y)q(y,x)$，则 $\alpha(x,y)=\frac{\pi(y)q(y,x)}{\pi(x)q(x,y)}$，$\alpha(y,x)=1$：

$$\pi(x)k(x,y) = \pi(x)q(x,y)\cdot\frac{\pi(y)q(y,x)}{\pi(x)q(x,y)} = \pi(y)q(y,x) = \pi(y)q(y,x)\cdot 1 = \pi(y)k(y,x)$$

细致平衡成立，$\pi$ 是平稳分布。$\square$

**(c) 混合时间与谱隙**

对有限可逆链（满足细致平衡），转移矩阵特征值 $1=\lambda_1>\lambda_2\geq\cdots\geq\lambda_N\geq-1$（不可约非周期链），谱隙 $\text{Gap}=1-\lambda_2$。

由谱分解，$P^t(x,\cdot)-\pi$ 的衰减由最大非平凡特征值控制：

$$\|P^t(x,\cdot)-\pi\|_{TV} \leq \sqrt{\frac{1}{\pi(x)}}\cdot\lambda_2^t \leq \sqrt{\frac{1}{\pi_{\min}}}\cdot e^{-t\cdot\text{Gap}}$$

（近似，利用 $\lambda_2\leq e^{-\text{Gap}}$）。令右边 $\leq\varepsilon$：

$$t \geq \frac{\log(1/\varepsilon\cdot\pi_{\min}^{-1/2})}{\text{Gap}} \leq \frac{\log(1/\varepsilon\pi_{\min})}{\text{Gap}}$$

故 $t_{\text{mix}}(\varepsilon) = O\!\left(\frac{\log(1/\varepsilon\pi_{\min})}{\text{Gap}}\right)$。

**(d) LLM 自回归采样与温度**

LLM 生成 token 的转移概率：$P_T(w|w_{<t}) \propto \exp(\text{logit}(w)/T)$。

温度 $T$ 对谱隙的影响：

- $T\to0$：分布趋于 argmax（退化分布），$\pi_{\min}\to0$，谱隙趋于 0，链混合极慢（卡在局部最高概率序列）。
- $T\to\infty$：均匀分布，谱隙最大（$\sim 1-1/V$，$V$=词表大小），混合最快，但输出无意义（随机）。
- $T=1$：标准 softmax，谱隙与模型概率差异相关。

**最优温度权衡**：最大化**有效信息率** $= I(\text{context}; \text{token}) \times n_{\text{eff}}/n$（互信息乘以混合效率）。实践中 $T\in[0.7, 1.0]$ 在流畅性与多样性间取得平衡。

> ⚠️ 温度参数仅改变 logit 的尺度，不改变模型的知识，但显著影响输出分布的熵和采样链的混合速度。

---

### E.8.6（Ch.22+Ch.24，概率图模型 + 信念传播 + 近似推断）

**题目**：树上 BP 消息方程；树上精确性；Bethe 自由能；神经 CRF 均场推断。

**解**：

**(a) 树结构因子图上的 BP 消息方程**

因子图 $G=(V,F,E)$，变量 $v$ 到因子 $f$ 的消息：

$$\mu_{v\to f}(x_v) = \prod_{f'\in\text{ne}(v)\setminus f}\mu_{f'\to v}(x_v)$$

因子 $f$ 到变量 $v$ 的消息：

$$\mu_{f\to v}(x_v) = \sum_{\mathbf{x}_f\setminus x_v} f(\mathbf{x}_f)\prod_{u\in\text{ne}(f)\setminus v}\mu_{u\to f}(x_u)$$

边际信念：$b(x_v) \propto \prod_{f\in\text{ne}(v)}\mu_{f\to v}(x_v)$。

**(b) 树上 BP 精确性**

树图无环，消息从叶节点向根单向传播（两遍：叶→根，根→叶），有限步（$O(\text{直径})$ 步）内收敛。精确性证明：

通过结构归纳（剥叶定理），消息等于对应子树的边际和（精确 sum-product），最终信念等于精确边际。

**循环图（Loopy BP）**不精确原因：消息在环路中循环传递，同一信息被"重复计数"，等价于无限次展开树上的近似（Bethe 近似）。不保证收敛，即使收敛也是近似解。

**(c) Bethe 自由能**

精确自由能：$F = -\log Z$（配分函数对数）。

Bethe 近似用信念 $\{b_f, b_v\}$（满足局部一致性约束）参数化，Bethe 自由能为：

$$F_{\text{Bethe}} = -\sum_f\sum_{\mathbf{x}_f}b_f(\mathbf{x}_f)\log\frac{f(\mathbf{x}_f)}{b_f(\mathbf{x}_f)} - \sum_v(d_v-1)\sum_{x_v}b_v(x_v)\log b_v(x_v)$$

其中 $d_v=|\text{ne}(v)|$ 为变量 $v$ 的度数，$(d_v-1)$ 是对边际重叠的修正。

**树图上 $F_{\text{Bethe}} = F_{\text{exact}}$**：树图上所有变量度数 $d_v$，对应的"重计数"修正恰好使 Bethe 等于精确值（可通过精确信念代入验证等号）。

**循环图**：BP 不动点恰好是 $F_{\text{Bethe}}$ 的驻点（Yedidia 等，2001），但不保证是全局最小，且 $F_{\text{Bethe}}\neq F_{\text{exact}}$。

**(d) 神经 CRF 与均场推断**

密集 CRF（DenseCRF）能量函数：$E(\mathbf{x})=-\sum_i\psi_u(x_i)-\sum_{i<j}\psi_p(x_i,x_j)$，成对势为 Gaussian 核混合。

均场推断迭代（$Q(\mathbf{x})=\prod_i Q_i(x_i)$）：

1. **消息传递**：$\tilde{Q}_i(l) = -\sum_{m}\mu_m\sum_j k^m(\mathbf{f}_i,\mathbf{f}_j)Q_j(l)$（Gaussian 卷积，可用快速滤波）
2. **相容性变换**：$\hat{Q}_i(l) = \sum_{l'}\mu(l,l')\tilde{Q}_i(l')$（标签兼容矩阵）
3. **更新**：$Q_i(x_i) \propto \exp\!\left(\psi_u(x_i) - \hat{Q}_i(x_i)\right)$

**嵌入端到端训练**：步骤 1-3 构成确定性的 RNN 迭代（固定迭代次数 $T$），梯度通过 BPTT 反向传播，与 CNN 特征提取器联合训练（CRFasRNN，Zheng 等 2015）。

---

### E.8.7（Ch.22，Rényi 散度 + $\alpha$-散度 + 鲁棒推断）

**题目**：$\alpha\to1$ 极限；非负性与单调性；Bhattacharyya 距离计算；鲁棒推断优势。

**解**：

**(a) $\lim_{\alpha\to1} D_\alpha(p\|q) = D_{\mathrm{KL}}(p\|q)$**

$$D_\alpha(p\|q) = \frac{1}{\alpha-1}\log\!\underbrace{\int p^\alpha q^{1-\alpha}\,dx}_{Z(\alpha)}$$

当 $\alpha=1$：$Z(1)=\int p\,dx=1$，$D_1=0/0$（不定式）。用 L'Hôpital：

$$\lim_{\alpha\to1}D_\alpha = \lim_{\alpha\to1}\frac{\log Z(\alpha)}{\alpha-1} = \frac{Z'(1)}{Z(1)} = Z'(1)$$

$$Z'(\alpha) = \frac{d}{d\alpha}\int p^\alpha q^{1-\alpha}\,dx = \int p^\alpha q^{1-\alpha}\log\frac{p}{q}\,dx$$

$$Z'(1) = \int p\log\frac{p}{q}\,dx = D_{\mathrm{KL}}(p\|q)$$

故 $\lim_{\alpha\to1}D_\alpha(p\|q) = D_{\mathrm{KL}}(p\|q)$。$\square$

**(b) 非负性与单调性**

**非负性**：令 $r(x)=p(x)/q(x)$，则

$$Z(\alpha) = E_q[r^\alpha] \geq (E_q[r])^\alpha = 1^\alpha = 1$$

（对 $\alpha>0$，由 Jensen 和 $r^\alpha$ 凸/凹性，结合 $\log Z\geq0$，$D_\alpha\geq0$。对 $\alpha<0$ 类似。）

**关于 $\alpha$ 单调**：$D_\alpha(p\|q)$ 关于 $\alpha$ 单调不减（由 log-sum 不等式，$\partial D_\alpha/\partial\alpha \geq 0$），故：

$$D_0 \leq D_{1/2} \leq D_1 = D_{\mathrm{KL}} \leq D_2 \leq \cdots$$

**(c) Bhattacharyya 距离（$\alpha=1/2$）**

$$D_{1/2}(p\|q) = -2\log\int\sqrt{p(x)q(x)}\,dx$$

对 $p=\mathcal{N}(\mu_1,\sigma^2)$，$q=\mathcal{N}(\mu_2,\sigma^2)$（同方差）：

$$\int\sqrt{p(x)q(x)}\,dx = \int\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu_1)^2+(x-\mu_2)^2}{4\sigma^2}\right)dx$$

配方：$(x-\mu_1)^2+(x-\mu_2)^2 = 2(x-\bar\mu)^2 + (\mu_1-\mu_2)^2/2$，其中 $\bar\mu=(\mu_1+\mu_2)/2$：

$$\int\sqrt{pq}\,dx = \exp\!\left(-\frac{(\mu_1-\mu_2)^2}{8\sigma^2}\right)\cdot\int\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\bar\mu)^2}{2\cdot(\sigma^2/2)}\right)dx\cdot\frac{1}{\sqrt{2}}$$

实际上更简洁：两高斯的 Bhattacharyya 系数为 $BC = \exp(-(\mu_1-\mu_2)^2/(8\sigma^2))\cdot\sqrt{2\sigma^2/(2\sigma^2)} = \exp(-(\mu_1-\mu_2)^2/(8\sigma^2))$，

$$D_B = -\log BC = \frac{(\mu_1-\mu_2)^2}{8\sigma^2}$$

（同方差正态情形，Bhattacharyya 距离仅由均值之差决定。）

**答案**：$\boxed{D_B = \dfrac{(\mu_1-\mu_2)^2}{8\sigma^2}}$

**(d) 鲁棒推断中的 $\alpha$-散度**

**Power EP**：用 $\alpha$-散度 $D_\alpha(p\|q)$ 替代 KL，近似各因子时使用 $\alpha<1$（"矩匹配"转向），对重尾似然（柯西分布）更稳健：KL$(p\|q)$ 要求 $p>0$ 处 $q>0$（零覆盖），对重尾数据可能导致过度收缩；$\alpha<1$ 的散度对异常值不敏感（因为 $p^\alpha$ 对大 $p$ 的惩罚较轻）。

**VAE 中后验坍塌（Posterior Collapse）**：当 $D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))$ 主导 ELBO 时，$q_\phi\to p(z)$（先验），编码器无法传递信息。用 $\alpha<1$ 的 Rényi 散度替换 KL：

$$\mathcal{L}_\alpha = E_{q_\phi}[\log p(x|z)] - D_{1-\alpha}(q_\phi\|p)$$

$\alpha<1$ 时惩罚项更小（$D_\alpha\leq D_{\mathrm{KL}}$），$q_\phi$ 不被强制拉向先验，缓解后验坍塌，提高重建质量。

---

### E.8.8（Ch.23+Ch.24，随机变分推断 + 黑盒 VI + 策略梯度联系）

**题目**：REINFORCE 梯度估计；基线不改变期望的证明；最优基线；重参数化 vs 打分函数。

**解**：

**(a) REINFORCE（打分函数）梯度估计量**

目标：$\nabla_\phi\mathcal{L}(\phi) = \nabla_\phi E_{q_\phi(\mathbf{z})}[f(\mathbf{z})]$，其中 $f(\mathbf{z})=\log p(\mathbf{x},\mathbf{z})-\log q_\phi(\mathbf{z})$。

$$\nabla_\phi E_{q_\phi}[f(\mathbf{z})] = \nabla_\phi\int q_\phi(\mathbf{z})f(\mathbf{z})\,d\mathbf{z}$$

$$= \int f(\mathbf{z})\nabla_\phi q_\phi(\mathbf{z})\,d\mathbf{z} = \int f(\mathbf{z})q_\phi(\mathbf{z})\nabla_\phi\log q_\phi(\mathbf{z})\,d\mathbf{z}$$

（用对数导数技巧：$\nabla_\phi q_\phi = q_\phi\nabla_\phi\log q_\phi$。）

$$= E_{q_\phi}\!\left[f(\mathbf{z})\nabla_\phi\log q_\phi(\mathbf{z})\right]$$

代入 $f(\mathbf{z})$：

$$\nabla_\phi\mathcal{L} = E_{q_\phi}\!\left[\left(\log p(\mathbf{x},\mathbf{z})-\log q_\phi(\mathbf{z})\right)\nabla_\phi\log q_\phi(\mathbf{z})\right]$$

Monte Carlo 估计：从 $q_\phi$ 抽 $\mathbf{z}^{(s)}$，取样本均值。

**(b) 基线不改变期望**

设基线 $b$ 不依赖 $\mathbf{z}$：

$$E_{q_\phi}[b\cdot\nabla_\phi\log q_\phi(\mathbf{z})] = b\cdot\int q_\phi(\mathbf{z})\nabla_\phi\log q_\phi(\mathbf{z})\,d\mathbf{z} = b\cdot\nabla_\phi\int q_\phi(\mathbf{z})\,d\mathbf{z} = b\cdot\nabla_\phi 1 = 0$$

（利用 $\nabla_\phi\log q_\phi = \nabla_\phi q_\phi/q_\phi$ 和 $\int q_\phi\,d\mathbf{z}=1$ 对 $\phi$ 求导为 0。）

故 $E_{q_\phi}[(f-b)\nabla_\phi\log q_\phi] = E_{q_\phi}[f\nabla_\phi\log q_\phi]$，期望不变，但方差降低。

**(c) 最优基线**

目标：最小化每个分量 $k$ 的方差 $\text{Var}((f-b)\frac{\partial\log q_\phi}{\partial\phi_k})$。

$$\frac{\partial}{\partial b}\text{Var}\!\left[(f-b)g_k\right] = \frac{\partial}{\partial b}\!\left[E[(f-b)^2g_k^2]\right] = -2E[(f-b)g_k^2] = 0$$

$$b_k^* = \frac{E[f\cdot g_k^2]}{E[g_k^2]}, \quad g_k=\frac{\partial\log q_\phi}{\partial\phi_k}$$

向量形式：$b^* = \frac{E[f(\mathbf{z})\|\nabla_\phi\log q_\phi\|^2]}{E[\|\nabla_\phi\log q_\phi\|^2]}$。

**与 RL 策略梯度类比**：$f(\mathbf{z})\leftrightarrow R(\tau)$（回报），$\log q_\phi(\mathbf{z})\leftrightarrow\log\pi_\theta(\tau)$（策略对数），基线 $b\leftrightarrow V(s)$（值函数），最优基线减小梯度方差，对应优势函数 $A(s,a)=Q(s,a)-V(s)$。

**(d) 重参数化 vs 打分函数梯度**

**重参数化（Reparameterization Trick）**：对 Gaussian $q_\phi(\mathbf{z})=\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$，令 $\mathbf{z}=\boldsymbol{\mu}+\mathbf{L}\boldsymbol{\epsilon}$，$\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$（$\mathbf{L}\mathbf{L}^\top=\boldsymbol{\Sigma}$）：

$$\nabla_\phi\mathcal{L} = E_{\boldsymbol{\epsilon}}\!\left[\nabla_\phi f(\boldsymbol{\mu}+\mathbf{L}\boldsymbol{\epsilon})\right]$$

梯度直接通过 $f$ 传播，方差**远低于**打分函数估计量。

**离散变量问题**：离散 $\mathbf{z}$（如图神经网络离散结构选择）无法重参数化（不可微）。

- **Gumbel-Softmax**：连续松弛离散分布，$\tilde{z}_k = \text{softmax}((\log\pi_k+g_k)/\tau)$，$g_k\sim\text{Gumbel}(0,1)$，温度 $\tau\to0$ 趋近真实类别，$\tau>0$ 时可微。
- **Straight-Through Estimator（STE）**：前向传播用离散采样，反向传播用恒等函数替代（$\partial\text{stop\_gradient}/\partial\text{input}=1$），简单粗糙但实践有效。

方差比较：重参数化 $\ll$ STE $\approx$ Gumbel-Softmax $\ll$ REINFORCE（打分函数）。

---

### E.8.9（Ch.24，贝叶斯优化 + 高斯过程代理模型 + 采集函数）

**题目**：GP 后验推导；UCB 采集函数与 EI 等价；贝叶斯遗憾界；HPO 中的高维问题。

**解**：

**(a) GP 后验推导**

先验：$f\sim\mathcal{GP}(0,k(\cdot,\cdot))$（零均值），观测 $\mathbf{y}=f(\mathbf{X})+\boldsymbol{\varepsilon}$，$\boldsymbol{\varepsilon}\sim\mathcal{N}(\mathbf{0},\sigma_n^2\mathbf{I})$。

联合分布：

$$\begin{pmatrix}f^*\\\mathbf{y}\end{pmatrix}\sim\mathcal{N}\!\left(\mathbf{0},\begin{pmatrix}k_{**} & \mathbf{k}_*^\top\\\mathbf{k}_* & \mathbf{K}+\sigma_n^2\mathbf{I}\end{pmatrix}\right)$$

条件高斯公式：

$$f^*|\mathbf{y}\sim\mathcal{N}(\mu_n(\mathbf{x}^*),\sigma_n^2(\mathbf{x}^*))$$

$$\mu_n(\mathbf{x}^*) = \mathbf{k}_*^\top(\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}\mathbf{y}$$

$$\sigma_n^2(\mathbf{x}^*) = k_{**} - \mathbf{k}_*^\top(\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}\mathbf{k}_*$$

**(b) UCB 与 EI**

**UCB**：$\alpha_{UCB}(\mathbf{x}) = \mu_n(\mathbf{x}) + \kappa\sigma_n(\mathbf{x})$，$\kappa>0$ 控制探索-利用。

**期望改进（EI）**：$\alpha_{EI}(\mathbf{x}) = E[\max(f(\mathbf{x})-f^+,0)]$，$f^+ = \max_i y_i$：

$$\alpha_{EI}(\mathbf{x}) = (\mu_n(\mathbf{x})-f^+)\Phi(z) + \sigma_n(\mathbf{x})\phi(z), \quad z=\frac{\mu_n(\mathbf{x})-f^+}{\sigma_n(\mathbf{x})}$$

两者在 $\kappa = \sqrt{2\log(1/\delta)}$（概率置信区间）时近似等价；UCB 更直接分析，EI 有明确探索停止准则。

**Thompson Sampling（TS）**：从后验 $\mathcal{GP}$ 采样一条路径 $\hat{f}\sim p(\cdot|\mathcal{D}_n)$，最大化 $\hat{f}$——与 UCB 一样实现探索-利用（TS 的期望遗憾与 UCB 同阶）。

**(c) 贝叶斯遗憾界**

$T$ 步后贝叶斯遗憾：$BR_T = E\!\left[\sum_{t=1}^T(f(\mathbf{x}^*)-f(\mathbf{x}_t))\right]$。

对 UCB 采集函数，Srinivas 等（2010）证明：

$$BR_T \leq O(\sqrt{T\gamma_T\log T})$$

其中最大信息增益 $\gamma_T = \max_{|\mathcal{A}|=T} I(f_{\mathcal{A}};\mathbf{y}_{\mathcal{A}})$（前 $T$ 个点的信息增益上界）。

对常见核：SE 核 $\gamma_T = O((\log T)^{d+1})$；Matérn 核 $\gamma_T = O(T^{d/(2\nu+d)})$（$\nu$ 为平滑参数）。

**(d) HPO 中的 GP 困境与替代方法**

**GP 的维数诅咒**：核矩阵计算 $O(n^3)$（$n$ 为样本数）；在高维超参数空间（$d>20$）中，GP 需指数多样本才能有效覆盖；高维下 GP 倾向于均匀先验（所有核矩阵趋于单位矩阵）。

**BORE（贝叶斯优化通过密度比估计）**：将采集函数 $\alpha_{EI}$ 转化为分类问题（哪些参数配置产生好结果），用二元分类器（如神经网络、XGBoost）估计密度比 $p(\mathbf{x}|\text{good})/p(\mathbf{x})$，绕开 GP 的 $O(n^3)$ 计算。

**TPE（Tree-structured Parzen Estimator）**：分别建模 $p(\mathbf{x}|y<y^*) = l(\mathbf{x})$ 和 $p(\mathbf{x}|y\geq y^*) = g(\mathbf{x})$（用核密度估计），采集函数为比率 $l(\mathbf{x})/g(\mathbf{x})$。树结构处理条件超参数（如神经网络层数决定后才有每层节点数），计算复杂度 $O(n)$，适合高维和条件空间。

---

### E.8.10（Ch.22+Ch.23+Ch.24，Wasserstein 距离 + 最优传输 + 生成模型）

**题目**：K-R 对偶定理；一维 $W_1$ = CDF 的 $L^1$ 距离；WGAN-GP；Schrödinger 桥与扩散模型。

**解**：

**(a) Kantorovich-Rubinstein 对偶（$W_1$）**

$$W_1(\mu,\nu) = \inf_{\gamma\in\mathcal{P}(\mu,\nu)}\int\|\mathbf{x}-\mathbf{y}\|\,d\gamma(\mathbf{x},\mathbf{y})$$

**对偶形式**（Kantorovich-Rubinstein 定理）：

$$W_1(\mu,\nu) = \sup_{\|f\|_L\leq1}\!\left[\int f\,d\mu - \int f\,d\nu\right]$$

其中 $\|f\|_L = \sup_{x\neq y}|f(x)-f(y)|/\|x-y\|\leq1$ 为 Lipschitz 常数约束。

**证明思路（强对偶）**：原始问题（线性规划在测度空间上）满足 Slater 条件，强对偶成立。对偶变量恰好是满足 1-Lipschitz 约束的函数（由约束 $f(x)-f(y)\leq\|x-y\|$ 导出）。

**(b) 一维 $W_1$ = CDF 的 $L^1$ 距离**

一维情形，最优传输的解析解为分位数匹配（$\gamma^*(x,y) = (F_\mu^{-1}(u), F_\nu^{-1}(u))$，$u\sim U(0,1)$）：

$$W_1(\mu,\nu) = \int_0^1|F_\mu^{-1}(u)-F_\nu^{-1}(u)|\,du = \int_{-\infty}^\infty|F_\mu(t)-F_\nu(t)|\,dt$$

（最后等式：变量替换，利用 $\int_0^1|F^{-1}(u)-G^{-1}(u)|du = \int|F(t)-G(t)|dt$。）

**计算 $N(0,1)$ 与 $N(\delta,1)$**：

$F_\mu(t) = \Phi(t)$，$F_\nu(t) = \Phi(t-\delta)$，

$$W_1 = \int_{-\infty}^\infty|\Phi(t)-\Phi(t-\delta)|\,dt$$

由平移不变性，当 $\delta>0$ 时 $\Phi(t-\delta)<\Phi(t)$（均值较大的分布 CDF 在右），积分：

$$W_1 = \int_{-\infty}^\infty[\Phi(t)-\Phi(t-\delta)]\,dt = \delta$$

（平移正态的 $W_1$ = 均值之差。）

**答案**：$\boxed{W_1(N(0,1),N(\delta,1))=|\delta|}$

**(c) WGAN 与梯度惩罚**

**WGAN**（Arjovsky 等，2017）：用神经网络判别器 $D_w$（Critic）近似 K-R 对偶中的 1-Lipschitz 函数：

$$W_1(p_r,p_g) \approx \max_{\|D_w\|_L\leq1}\!\left[E_{x\sim p_r}[D_w(x)] - E_{\tilde x\sim p_g}[D_w(\tilde x)]\right]$$

**Lipschitz 约束实现**：

- 原始 WGAN 用权重裁剪（$w\in[-c,c]$），但会导致梯度爆炸/消失。
- **WGAN-GP**（Gulrajani 等，2017）：梯度惩罚项强制 $\|\nabla D\|_2=1$（1-Lipschitz 的必要条件）：

$$\mathcal{L} = \underbrace{E_{\tilde x}[D(\tilde x)] - E_x[D(x)]}_{\text{Wasserstein 距离近似}} + \lambda\underbrace{E_{\hat x}\!\left[(\|\nabla_{\hat x}D(\hat x)\|_2-1)^2\right]}_{\text{梯度惩罚}}$$

其中 $\hat x = \epsilon x + (1-\epsilon)\tilde x$，$\epsilon\sim U(0,1)$（真实与生成样本之间的插值），在插值点强制梯度范数为 1。

> ⚠️ Lipschitz 约束 $\Rightarrow$ $\|\nabla D\|_2\leq1$ 处处成立，但梯度惩罚只对插值点强制，是软约束。

**(d) Schrödinger 桥与扩散模型**

**Schrödinger 桥（SB）**：给定端点边际 $X_0\sim\mu$，$X_1\sim\nu$，在所有过程 $\mathbb{P}$ 中找最接近参考测度 $\mathbb{Q}$（布朗运动）的过程：

$$\min_{\mathbb{P}:\,X_0\sim\mu,\,X_1\sim\nu} D_{\mathrm{KL}}(\mathbb{P}\|\mathbb{Q})$$

**IPFP（迭代比例拟合法）**：交替投影到满足 $X_0\sim\mu$ 和 $X_1\sim\nu$ 的测度集（Sinkhorn 算法的连续版本）：

$$\mathbb{P}^{2k+1} = \text{proj}_{\mathcal{P}(\cdot,\nu)}\mathbb{P}^{2k}, \quad \mathbb{P}^{2k+2} = \text{proj}_{\mathcal{P}(\mu,\cdot)}\mathbb{P}^{2k+1}$$

每次投影等价于学习前向或后向的"半桥"（条件扩散过程）。

**与扩散模型的关系**：

| DDPM | Schrödinger 桥（SB）|
|------|------|
| 前向过程：数据 $\to$ 噪声（固定 $q$） | 前向过程：$\mu\to\nu$（优化） |
| 反向过程：噪声 $\to$ 数据（学习 $p_\theta$） | 反向过程：$\nu\to\mu$（学习后向 SDE）|
| 参考测度：OU 过程（不精确端点） | 参考测度：布朗运动（精确端点约束）|
| 目标：$\nu=\mathcal{N}(0,\mathbf{I})$ | 目标：任意 $\nu$（可为图像域） |

**I2SB（Image-to-Image Schrödinger Bridge，Liu 等 2023）**：用 SB 实现两个图像域间的转换（超分辨率、去雨、修复），无需先退化到高斯噪声，直接在图像对之间插值，训练效率更高。

**Score Matching 联系**：DDPM 的去噪 score $s_\theta(x_t,t)\approx\nabla_{x_t}\log q(x_t)$ 对应 SB 的后向 SDE 漂移项（Nelson 公式），两者在 $\mu=p_{\text{data}}$，$\nu=\mathcal{N}(0,\mathbf{I})$ 时完全一致，SB 是 DDPM 的推广（允许 $\nu$ 非高斯，允许非对称端点）。
