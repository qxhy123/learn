# 信息论核心三件套

> **一例速记**：为什么均匀分布的"不确定性"最大？
> 掷一枚均匀硬币（$p = 0.5$）比掷一枚 99% 朝正面的硬币，结果更"出人意料"——前者熵 $H = 1$ bit，后者熵 $\approx 0.08$ bit。熵就是"平均惊讶度"。
> **熵 = 对未来结果平均有多吃惊（bit 数）。**

---

## 一、为什么信息论是 AI 的底层语言

现代 AI 训练的核心操作——最小化损失——在概率视角下就是**让模型分布尽量接近数据分布**。信息论提供了度量"两个分布有多不同"的精确工具：

- **交叉熵**（cross-entropy）是分类任务的标准损失函数；
- **KL 散度**出现在 VAE 的正则化项、RLHF 的策略约束、知识蒸馏的目标函数；
- **互信息**是特征选择、信息瓶颈、对比学习的理论基础；
- **熵**衡量模型预测的不确定性，用于主动学习（选择最不确定的样本标注）。

理解这三件套及其关系，就掌握了深度学习损失函数的底层逻辑。

---

## 二、熵：不确定性的度量

### 2.1 定义

设离散随机变量 $X$ 的概率分布为 $P(X = x_i) = p_i$，$i = 1, \ldots, n$，则 $X$ 的**香农熵**（Shannon entropy）定义为：

$$H(X) = -\sum_{i=1}^n p_i \log_2 p_i$$

约定 $0 \log 0 = 0$（极限值）。单位：以 $\log_2$ 计为 bit，以 $\ln$ 计为 nat。

**直觉**：$-\log_2 p_i$ 是事件 $x_i$ 的"惊讶度"（information content）——越罕见的事件发生时越惊讶。熵是惊讶度的概率期望（加权平均）。

### 2.2 基本性质

**性质 1（非负性）**：$H(X) \geq 0$，等号当且仅当 $X$ 为确定性变量（某 $p_i = 1$）。

**性质 2（最大熵定理）**：在 $n$ 点分布中，**均匀分布的熵最大**：

$$H(X) \leq \log_2 n$$

等号当且仅当 $p_1 = p_2 = \cdots = p_n = 1/n$（均匀分布）。

**证明**（Jensen 不等式）：$-\log$ 是凸函数，

$$H(X) = \sum p_i (-\log p_i) = \sum p_i \log\frac{1}{p_i} \leq \log\left(\sum p_i \cdot \frac{1}{p_i}\right) = \log n$$

（最后一步用 Jensen 不等式：$\sum p_i f(1/p_i) \leq f(\sum p_i/p_i) = f(n)$，其中 $f = \log$ 是凹函数，故用 $\leq$）。$\blacksquare$

**性质 3（链式法则）**：$H(X, Y) = H(X) + H(Y|X)$，其中条件熵 $H(Y|X) = \sum_x p(x) H(Y|X=x)$。

### 2.3 熵在机器学习中的意义

| 场景 | 熵的角色 |
|---|---|
| 决策树分裂 | 选择使条件熵 $H(Y\vert X)$ 最小（信息增益最大）的特征 |
| 主动学习 | 选择预测熵最大的样本（不确定性最高，标注价值最大） |
| 语言模型评估 | 困惑度（perplexity）$= 2^{H}$，衡量模型对测试集的"不确定性" |
| 最大熵模型 | 在满足约束的分布中选熵最大的（最少主观假设）|

---

## 三、KL 散度：分布差异的度量

### 3.1 定义

**Kullback-Leibler 散度**（KL divergence，也称相对熵）衡量分布 $p$ 与参考分布 $q$ 的差异：

$$D_{\mathrm{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

（连续版：$D_{\mathrm{KL}}(p \| q) = \int p(x) \ln \frac{p(x)}{q(x)}\,dx$）

约定：若 $p(x) > 0$ 但 $q(x) = 0$，则 $D_{\mathrm{KL}} = +\infty$（$q$ 不能为 $p$ 赋予零概率的点分配零质量）。

### 3.2 核心性质

**性质 1（非负性）**：$D_{\mathrm{KL}}(p \| q) \geq 0$，等号当且仅当 $p = q$（几乎处处）。

**性质 2（不对称性）**：一般地，$D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p)$。

例：$p = \mathcal{N}(0, 1)$，$q = \mathcal{N}(3, 1)$：
$$D_{\mathrm{KL}}(p\|q) = \frac{9}{2}, \quad D_{\mathrm{KL}}(q\|p) = \frac{9}{2}$$
（此例对称是因为高斯族的特殊性；一般不对称，见下文 VAE 分析。）

**性质 3（不是距离）**：KL 散度不满足三角不等式，也不对称，故严格来说不是度量（metric）。

### 3.3 用 Jensen 不等式证明 $D_{\mathrm{KL}} \geq 0$

这是一条贯穿信息论与凸分析的核心证明，值得完整理解。

**关键事实**：$-\ln(x)$ 是凸函数（$(-\ln)'' = 1/x^2 > 0$，$x > 0$）。

**证明**：

$$-D_{\mathrm{KL}}(p\|q) = -\sum_x p(x)\log\frac{p(x)}{q(x)} = \sum_x p(x)\log\frac{q(x)}{p(x)} = \mathbb{E}_p\!\left[\log\frac{q(X)}{p(X)}\right]$$

由于 $\log$ 是凹函数，Jensen 不等式给出（对凹函数 $f$，$\mathbb{E}[f(Z)] \leq f(\mathbb{E}[Z])$）：

$$\mathbb{E}_p\!\left[\log\frac{q(X)}{p(X)}\right] \leq \log\mathbb{E}_p\!\left[\frac{q(X)}{p(X)}\right]$$

计算期望：

$$\mathbb{E}_p\!\left[\frac{q(X)}{p(X)}\right] = \sum_x p(x)\cdot\frac{q(x)}{p(x)} = \sum_x q(x) = 1$$

故：$-D_{\mathrm{KL}}(p\|q) \leq \log 1 = 0$，即 $D_{\mathrm{KL}}(p\|q) \geq 0$。$\blacksquare$

等号成立（$D_{\mathrm{KL}} = 0$）当且仅当 Jensen 不等式等号成立，即 $q(x)/p(x) = $ 常数，结合两者均为概率分布，得 $p = q$。

### 3.4 正向 KL vs 反向 KL

KL 散度的不对称性在变分推断中有深刻含义：

- **正向 KL**（inclusive，mean-seeking）：$D_{\mathrm{KL}}(p\|q)$ 最小化时，$q$ 倾向于"覆盖"$p$ 的所有支撑（若 $p(x) > 0$ 但 $q(x) \approx 0$，则代价极大）。
- **反向 KL**（exclusive，mode-seeking）：$D_{\mathrm{KL}}(q\|p)$ 最小化时，$q$ 倾向于"集中在 $p$ 的某个众数"（若 $q(x) > 0$ 但 $p(x) \approx 0$，则代价极大，故 $q$ 会避开 $p$ 密度低的区域）。

VAE 中优化 $D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))$（反向 KL），这是变分推断的标准做法——对编码器分布 $q_\phi$ 可以直接采样，便于计算梯度。

---

## 四、交叉熵：分类损失的本质

### 4.1 定义与分解

**交叉熵**（cross-entropy）定义为：

$$H(p, q) = -\sum_x p(x) \log q(x)$$

**关键分解**：

$$H(p, q) = H(p) + D_{\mathrm{KL}}(p \| q)$$

**证明**：

$$H(p, q) = -\sum p\log q = -\sum p\log p + \sum p\log\frac{p}{q} = H(p) + D_{\mathrm{KL}}(p\|q)$$

由于 $D_{\mathrm{KL}} \geq 0$，故 $H(p, q) \geq H(p)$——用 $q$ 编码 $p$ 产生的数据，平均码长不小于用 $p$ 自身编码。等号当且仅当 $p = q$。

### 4.2 分类损失 = 交叉熵

设真实标签 $y \in \{1, \ldots, K\}$，真实分布为 one-hot 向量 $p = e_y$（第 $y$ 个位置为 1，其余为 0），模型预测概率为 $q = \hat{y} = \text{softmax}(f_\theta(x))$。

$$H(p, q) = -\sum_{k=1}^K p_k \log q_k = -\log q_y = -\log \hat{y}_y$$

这正是多分类的**负对数似然损失**（negative log-likelihood），也叫**交叉熵损失**（cross-entropy loss）。

**最小化交叉熵 $\Leftrightarrow$ 最大化对数似然 $\Leftrightarrow$ 最小化 KL 散度**（因为真实分布 $p$ 的熵 $H(p)$ 与模型无关，固定为常数）。

### 4.3 二元交叉熵（Binary Cross-Entropy, BCE）

对二分类（$K = 2$），$p \in \{0, 1\}$（真实标签），$q = \sigma(f_\theta(x)) \in (0,1)$（预测概率）：

$$\text{BCE} = -[y\log q + (1-y)\log(1-q)]$$

这是逻辑回归的损失函数，结合 sigmoid 激活使梯度形式极为简洁（见 Toolkit 12）。

---

## 五、互信息：两个变量的共享信息

### 5.1 定义

**互信息**（mutual information）$I(X;Y)$ 度量随机变量 $X$ 和 $Y$ 之间的统计依赖程度：

$$I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$$

### 5.2 三种等价表示

$$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$$

以及：

$$I(X;Y) = D_{\mathrm{KL}}(p_{XY} \| p_X p_Y)$$

最后一个表示最深刻：互信息 = 联合分布与独立分布（乘积分布）之间的 KL 散度。$I(X;Y) = 0$ 当且仅当 $X$ 和 $Y$ 独立（$p_{XY} = p_X p_Y$）。

### 5.3 信息论四量关系图

$$H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$

用 Venn 图表示：
- 左圆 $= H(X)$；右圆 $= H(Y)$；
- 交集 $= I(X;Y)$（共享信息）；
- 左圆去交集 $= H(X|Y)$；右圆去交集 $= H(Y|X)$；
- 并集 $= H(X,Y)$（联合熵）。

**数据处理不等式**（DPI）：若 $X \to Y \to Z$ 形成 Markov 链，则 $I(X;Z) \leq I(X;Y)$——中间处理步骤不能增加信息量。这是信息瓶颈理论的基础。

### 5.4 互信息在 AI 中的应用

| 应用 | 互信息的角色 |
|---|---|
| 决策树（互信息增益） | $I(Y; X_j) = H(Y) - H(Y\vert X_j)$，选最大的特征分裂 |
| 信息瓶颈（IB） | 压缩 $X$：最小化 $I(Z;X)$，同时最大化 $I(Z;Y)$（$Z$ 是压缩表示） |
| 对比学习（SimCLR 等） | 最大化 $I(z_1;z_2)$（同一图像两个增强视角的互信息） |
| 独立成分分析（ICA） | 最小化成分之间的互信息，使提取的特征尽量独立 |

---

## 六、演示题：证明 $D_{\mathrm{KL}}(p \| q) \geq 0$

**题目**：设 $p, q$ 是同一离散集合 $\mathcal{X}$ 上的两个概率分布，$p(x) > 0$ 和 $q(x) > 0$ 对所有 $x \in \mathcal{X}$ 成立，用 Jensen 不等式证明 $D_{\mathrm{KL}}(p \| q) \geq 0$。

> **第 1 步：识别凸函数结构。**
>
> $D_{\mathrm{KL}}(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = -\sum_x p(x)\log\frac{q(x)}{p(x)}$
>
> 将 $\frac{q(x)}{p(x)}$ 视为关于分布 $p$ 的随机变量 $Z = q(X)/p(X)$，则：
>
> $$D_{\mathrm{KL}}(p\|q) = -\mathbb{E}_p[\log Z] = \mathbb{E}_p[-\log Z]$$

> **第 2 步：利用 $-\log$ 的凸性应用 Jensen 不等式。**
>
> 函数 $\varphi(t) = -\log t$ 在 $t > 0$ 上是**严格凸函数**：
> $$\varphi''(t) = \frac{1}{t^2} > 0$$
>
> Jensen 不等式对凸函数 $\varphi$ 成立：
> $$\mathbb{E}_p[\varphi(Z)] \geq \varphi(\mathbb{E}_p[Z])$$
>
> 即：
> $$\mathbb{E}_p\!\left[-\log\frac{q(X)}{p(X)}\right] \geq -\log\,\mathbb{E}_p\!\left[\frac{q(X)}{p(X)}\right]$$

> **第 3 步：计算右侧的期望。**
>
> $$\mathbb{E}_p\!\left[\frac{q(X)}{p(X)}\right] = \sum_x p(x)\cdot\frac{q(x)}{p(x)} = \sum_x q(x) = 1$$
>
> 故右侧 $= -\log 1 = 0$。

> **第 4 步：合并结论。**
>
> $$D_{\mathrm{KL}}(p\|q) = \mathbb{E}_p\!\left[-\log\frac{q(X)}{p(X)}\right] \geq -\log 1 = 0$$
>
> 等号成立当且仅当 $\varphi$ 的 Jensen 等号条件满足：$Z = q(X)/p(X)$ 为常数（即 $q(x)/p(x) = c$ 对所有 $x$ 成立），结合 $\sum q(x) = \sum p(x) = 1$，得 $c = 1$，即 $p = q$。$\blacksquare$
>
> **证明的结构总结**：非负性来自三步推理——① 写成期望形式；② 凸函数 Jensen；③ $q$ 是概率分布使期望 $= 1$，$\log 1 = 0$。步骤 ③ 是关键，"$q$ 的概率之和为 1"直接给出下界是 $0$ 而非其他值。

---

## 七、思考路标

**路标 1**：熵 $H(X)$ 有三种等价直觉——**信息量（平均编码长度）**、**不确定性**、**平均惊讶度**。香农证明：若要用二进制编码 $X$ 的输出，每个符号平均最少需要 $H(X)$ bits（信源编码定理）。这三种直觉相互印证，选最适合当前语境的那种来理解。

**路标 2**：KL 散度的不对称性不是缺陷，而是信息。$D_{\mathrm{KL}}(p\|q)$ 是"用 $q$ 编码 $p$ 的数据时额外浪费的 bits"；$D_{\mathrm{KL}}(q\|p)$ 含义相反。在变分推断中，选哪个方向由**哪个分布可以采样**决定——可采样的放在前面（被期望的分布）。

**路标 3**：交叉熵损失 $H(p,q) = H(p) + D_{\mathrm{KL}}(p\|q)$ 的分解告诉我们：**最小化交叉熵本质上是最小化 KL 散度**（因为 $H(p)$ 与模型参数无关，是常数）。这统一了最大似然估计（MLE）和 KL 最小化这两种看似不同的训练目标。

**路标 4**：互信息 $I(X;Y) = D_{\mathrm{KL}}(p_{XY} \| p_X p_Y)$ 揭示了"独立性"的信息论含义：两个变量独立 $\Leftrightarrow$ 其联合分布 = 边缘分布之积 $\Leftrightarrow$ $D_{\mathrm{KL}} = 0$ $\Leftrightarrow$ $I(X;Y) = 0$。**互信息是最自然的非线性相关性度量**，不像 Pearson 相关系数只能检测线性依赖。

**路标 5**：信息处理不等式（DPI）$I(X;Z) \leq I(X;Y)$（$X\to Y \to Z$ Markov 链）是深度学习中**信息瓶颈**（Information Bottleneck）理论的核心。每一层神经网络是一个信息处理步骤，深层网络逐层"压缩"输入信息，保留与任务相关的部分。DPI 保证信息只会减少，不会凭空增加。

**路标 6**：**均匀分布的最大熵性质**有深刻的推论：最大熵原理（Maximum Entropy Principle）指出，在满足已知约束（如均值）的所有分布中，选熵最大的那个——这相当于"最少主观臆断"。指数族分布（包括正态分布、泊松分布）都是特定约束下的最大熵分布。这也是正则化的信息论解释：先验对应约束，正则化损失对应 KL 散度。

---

## 八、典型应用例题

### 例 1：二分类交叉熵的梯度计算

**场景**：逻辑回归，$\hat{p} = \sigma(z)$，损失 $L = -[y\log\hat{p} + (1-y)\log(1-\hat{p})]$，计算 $\partial L/\partial z$。

**推导**：

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{p}} \cdot \frac{\partial \hat{p}}{\partial z}$$

$$\frac{\partial L}{\partial \hat{p}} = -\frac{y}{\hat{p}} + \frac{1-y}{1-\hat{p}} = \frac{\hat{p} - y}{\hat{p}(1-\hat{p})}$$

$$\frac{\partial \hat{p}}{\partial z} = \sigma(z)(1-\sigma(z)) = \hat{p}(1-\hat{p})$$

$$\frac{\partial L}{\partial z} = \frac{\hat{p}-y}{\hat{p}(1-\hat{p})} \cdot \hat{p}(1-\hat{p}) = \hat{p} - y$$

**结论**：交叉熵损失 + sigmoid 激活的组合，使梯度形式极为简洁 $= \hat{p} - y$（预测值与真实标签之差）。这也是为什么不使用 MSE 损失配合 sigmoid——MSE 会引入 $\hat{p}(1-\hat{p})$ 项，在饱和区梯度消失。

### 例 2：两个高斯分布之间的 KL 散度

**场景**：VAE 中，编码器输出 $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$，先验 $p(z) = \mathcal{N}(0, I)$，计算 $D_{\mathrm{KL}}(q_\phi \| p)$（一维情况）。

**推导**（直接代入定义）：

$$D_{\mathrm{KL}}(\mathcal{N}(\mu,\sigma^2) \| \mathcal{N}(0,1)) = \int \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(z-\mu)^2}{2\sigma^2}} \ln\frac{\frac{1}{\sigma}e^{-\frac{(z-\mu)^2}{2\sigma^2}}}{e^{-z^2/2}}\,dz$$

展开对数：$\ln\frac{1}{\sigma} - \frac{(z-\mu)^2}{2\sigma^2} + \frac{z^2}{2}$

利用 $\mathbb{E}[(Z-\mu)^2] = \sigma^2$，$\mathbb{E}[Z^2] = \sigma^2 + \mu^2$（$Z \sim \mathcal{N}(\mu, \sigma^2)$）：

$$D_{\mathrm{KL}} = -\ln\sigma - \frac{1}{2} + \frac{\sigma^2 + \mu^2}{2} = \frac{1}{2}\left(\mu^2 + \sigma^2 - \ln\sigma^2 - 1\right)$$

**AI 意义**：这是 VAE 损失中 KL 正则化项的闭合公式，可以直接反向传播。$\mu \to 0$，$\sigma \to 1$ 时取最小值 $0$（编码器分布与先验重合）。

### 例 3：特征选择中的互信息增益

**场景**：文本分类中选择最具判别力的词，计算每个词 $X_j$ 与标签 $Y$ 的互信息：

$$I(Y; X_j) = H(Y) - H(Y|X_j)$$

设 $Y \in \{+1,-1\}$，$P(Y=+1) = 0.6$，$H(Y) = -0.6\log_2 0.6 - 0.4\log_2 0.4 \approx 0.971$ bits。

设词 $X_j$ 出现（$X_j = 1$）时 $P(Y=+1|X_j=1) = 0.9$，不出现（$X_j = 0$）时 $P(Y=+1|X_j=0) = 0.5$，且 $P(X_j=1) = 0.2$。

$$H(Y|X_j = 1) = -0.9\log_2 0.9 - 0.1\log_2 0.1 \approx 0.469 \text{ bits}$$
$$H(Y|X_j = 0) = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 1.0 \text{ bits}$$
$$H(Y|X_j) = 0.2 \times 0.469 + 0.8 \times 1.0 = 0.894 \text{ bits}$$
$$I(Y;X_j) = 0.971 - 0.894 = 0.077 \text{ bits}$$

该词每次出现平均提供 $0.077$ bits 的标签信息。选取互信息最大的 $k$ 个词作为特征——这是 TF-IDF 之外更有理论依据的特征选择方法。

---

## 九、自测题

**第 1 题**：计算以下分布的熵（以 $\log_2$ 为底，单位 bit）：
(1) $P(X=1) = 1$（确定性分布）；
(2) $P(X=1) = P(X=2) = 0.5$（均匀二点分布）；
(3) $P(X=1) = 0.9$，$P(X=2) = 0.1$。

> 提示：(1) $H = 0$；(2) $H = 1$ bit；(3) $H = -0.9\log_2 0.9 - 0.1\log_2 0.1 \approx 0.469$ bits。可验证 (2) > (3) > (1)，符合"均匀分布熵最大"。

**第 2 题**：设 $p = (0.3, 0.7)$，$q = (0.5, 0.5)$，计算 $D_{\mathrm{KL}}(p\|q)$ 和 $D_{\mathrm{KL}}(q\|p)$，验证不对称性。

> 提示：$D_{\mathrm{KL}}(p\|q) = 0.3\ln(0.3/0.5) + 0.7\ln(0.7/0.5) = 0.3\ln 0.6 + 0.7\ln 1.4 \approx 0.3(-0.511) + 0.7(0.336) \approx -0.153 + 0.235 = 0.082$。$D_{\mathrm{KL}}(q\|p) = 0.5\ln(0.5/0.3) + 0.5\ln(0.5/0.7) \approx 0.5(0.511) + 0.5(-0.336) \approx 0.088$。两者均 $\geq 0$，但不相等。

**第 3 题**：证明互信息的对称性：$I(X;Y) = I(Y;X)$。

> 提示：由定义 $I(X;Y) = D_{\mathrm{KL}}(p_{XY} \| p_X p_Y)$，因为 $p_{XY} = p_{YX}$（联合分布对称）且 $p_X p_Y = p_Y p_X$，故 $D_{\mathrm{KL}}(p_{XY}\|p_Xp_Y) = D_{\mathrm{KL}}(p_{YX}\|p_Yp_X)$，即 $I(X;Y) = I(Y;X)$。或用 $I(X;Y) = H(X)+H(Y)-H(X,Y)$ 直接验证对称性。

**第 4 题**：在多分类问题中，为什么使用交叉熵损失而不是均方误差（MSE）？从信息论和梯度两个角度分析。

> 提示：信息论角度：交叉熵是最大似然估计的等价形式，有坚实的概率论基础；MSE 对概率分布没有明确的概率含义。梯度角度（见例 1）：交叉熵 + softmax 的梯度 $= \hat{p} - y$（简洁），MSE + softmax 的梯度含 softmax 的 Jacobian，在饱和区接近零（梯度消失）。

**第 5 题**：信息瓶颈目标：$\min_{p(z|x)} [I(Z;X) - \beta I(Z;Y)]$（$\beta > 0$）。解释该目标的含义，以及 $\beta$ 的作用。

> 提示：最小化 $I(Z;X)$ 意味着 $Z$ 尽量不保留 $X$ 的信息（压缩）；最大化 $I(Z;Y)$ 意味着 $Z$ 尽量保留与任务标签 $Y$ 相关的信息（预测能力）。$\beta$ 控制两者的权衡：$\beta$ 大时更重视预测（保留更多信息），$\beta$ 小时更重视压缩（如 $\beta \to 0$，$Z$ 趋向常数）。这与 VAE 中的 $\beta$-VAE 框架对应（$\beta > 1$ 时加强 KL 正则，鼓励隐变量解耦）。
