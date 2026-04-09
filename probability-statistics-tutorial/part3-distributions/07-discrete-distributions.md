# 第7章 离散分布族

> **难度**：★★★☆☆
> **前置知识**：第4章离散随机变量、第5章连续随机变量、第6章多维随机变量

---

## 学习目标

- 掌握伯努利分布、二项分布、泊松分布、几何分布、负二项分布、超几何分布的PMF、期望与方差
- 理解各离散分布之间的内在联系（如泊松分布是二项分布的极限）
- 能够根据实际问题的特征正确选择合适的离散分布模型
- 深刻理解二元交叉熵损失与伯努利分布的概率论本质
- 掌握离散分布在深度学习中的应用：分类、计数建模与序列生成

---

## 7.1 伯努利分布与二项分布

### 7.1.1 伯努利分布

**伯努利试验**（Bernoulli Trial）是只有两种结果的随机试验：成功（记为1）或失败（记为0）。

**定义**：若随机变量 $X$ 满足

$$P(X = 1) = p, \quad P(X = 0) = 1 - p, \quad 0 < p < 1$$

则称 $X$ 服从参数为 $p$ 的**伯努利分布**，记作 $X \sim \text{Bernoulli}(p)$。

**PMF的统一写法**：

$$p(x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

**期望与方差**：

$$E[X] = p$$

$$\text{Var}(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1-p)$$

方差在 $p = 1/2$ 时取最大值 $1/4$，即不确定性最大。

**例7.1**：某神经元以概率 $p = 0.7$ 被激活，$X$ 表示该神经元是否激活，则 $X \sim \text{Bernoulli}(0.7)$，$E[X] = 0.7$，$\text{Var}(X) = 0.21$。

---

### 7.1.2 二项分布

将伯努利试验**独立重复** $n$ 次，记成功次数为 $X$，则 $X$ 服从**二项分布**。

**定义**：若 $X$ 表示 $n$ 次独立伯努利试验中成功的次数，每次成功概率为 $p$，则

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

记作 $X \sim B(n, p)$ 或 $X \sim \text{Binomial}(n, p)$。

**组合数** $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ 表示从 $n$ 次试验中选取 $k$ 次成功的方案数。

**归一化验证**：由二项式定理，

$$\sum_{k=0}^{n} \binom{n}{k} p^k (1-p)^{n-k} = (p + (1-p))^n = 1 \checkmark$$

**期望**（利用线性性：$X = X_1 + X_2 + \cdots + X_n$，$X_i \sim \text{Bernoulli}(p)$）：

$$E[X] = np$$

**方差**（各 $X_i$ 独立，方差可加）：

$$\text{Var}(X) = np(1-p)$$

**例7.2**：某图像分类器对每张图片的预测准确率为 $p = 0.9$，对 $n = 20$ 张图片进行预测，正确预测数 $X \sim B(20, 0.9)$。

$$E[X] = 18, \quad \text{Var}(X) = 20 \times 0.9 \times 0.1 = 1.8$$

$P(X = 20) = 0.9^{20} \approx 0.1216$，即全部预测正确的概率约为 $12.16\%$。

**二项分布的形状**：
- 当 $p = 0.5$ 时，分布关于 $n/2$ 对称
- 当 $p < 0.5$ 时，分布右偏；$p > 0.5$ 时左偏
- 随 $n$ 增大，形状趋近于正态分布（中心极限定理）

---

## 7.2 泊松分布

### 7.2.1 定义与PMF

**泊松分布**用于描述在**固定时间或空间区间内**，某稀有事件发生次数的分布。

**定义**：若随机变量 $X$ 的PMF为

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

其中 $\lambda > 0$ 为参数，则称 $X$ 服从参数为 $\lambda$ 的**泊松分布**，记作 $X \sim \text{Poisson}(\lambda)$ 或 $X \sim P(\lambda)$。

**归一化验证**：由指数函数的泰勒展开，

$$\sum_{k=0}^{\infty} \frac{\lambda^k e^{-\lambda}}{k!} = e^{-\lambda} \sum_{k=0}^{\infty} \frac{\lambda^k}{k!} = e^{-\lambda} \cdot e^{\lambda} = 1 \checkmark$$

**期望与方差**：

$$E[X] = \lambda, \quad \text{Var}(X) = \lambda$$

泊松分布的一个显著特征是**期望等于方差**，都等于参数 $\lambda$。

### 7.2.2 泊松分布是二项分布的极限

**定理（泊松极限定理）**：设 $X_n \sim B(n, p_n)$，当 $n \to \infty$，$p_n \to 0$，且 $np_n \to \lambda$（常数）时，

$$P(X_n = k) \to \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

**直观理解**：试验次数很多（$n$ 大），但每次成功概率极小（$p$ 小），平均成功次数 $\lambda = np$ 为常数。这正是"稀有事件"的特征。

**证明要点**：

$$\binom{n}{k} p^k (1-p)^{n-k} = \frac{n(n-1)\cdots(n-k+1)}{k!} \cdot \left(\frac{\lambda}{n}\right)^k \cdot \left(1 - \frac{\lambda}{n}\right)^{n-k}$$

当 $n \to \infty$ 时：
- $\frac{n(n-1)\cdots(n-k+1)}{n^k} \to 1$
- $\left(1 - \frac{\lambda}{n}\right)^n \to e^{-\lambda}$
- $\left(1 - \frac{\lambda}{n}\right)^{-k} \to 1$

因此极限为 $\frac{\lambda^k e^{-\lambda}}{k!}$。

**实用准则**：当 $n \geq 20$，$p \leq 0.05$（或 $n \geq 100$，$p \leq 0.1$）时，可用泊松分布近似二项分布，令 $\lambda = np$。

### 7.2.3 泊松分布的典型应用

| 应用场景 | $\lambda$ 的含义 |
|----------|-----------------|
| 每小时到达服务台的顾客数 | 单位时间平均到达率 |
| 某网页每天收到的点击次数 | 日均点击量 |
| 文本中某罕见词的出现次数 | 每千词平均出现次数 |
| 放射性衰变计数 | 单位时间平均衰变数 |

**例7.3**：某服务器每分钟平均接收 $\lambda = 3$ 个请求，$X \sim \text{Poisson}(3)$。

$$P(X = 0) = e^{-3} \approx 0.0498, \quad P(X = 5) = \frac{3^5 e^{-3}}{5!} = \frac{243 e^{-3}}{120} \approx 0.1008$$

### 7.2.4 泊松过程

泊松分布与**泊松过程**密切相关。泊松过程是描述随时间随机发生的事件的数学模型，满足：
1. 不相交时间区间内的事件数**独立**
2. 事件发生率为常数 $\lambda$（单位时间平均发生数）
3. 极短时间内同时发生两个事件的概率可忽略

在时间区间 $[0, t]$ 内发生的事件数 $N(t) \sim \text{Poisson}(\lambda t)$。

---

## 7.3 几何分布与负二项分布

### 7.3.1 几何分布

在独立重复的伯努利试验中，**首次成功**所需的试验次数服从几何分布。

**定义**：设每次试验成功概率为 $p$，$X$ 为首次成功时的试验次数，则

$$P(X = k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \ldots$$

记作 $X \sim \text{Geom}(p)$。

**另一种定义**（首次成功前的失败次数 $Y = X - 1$）：

$$P(Y = k) = (1-p)^k p, \quad k = 0, 1, 2, \ldots$$

**归一化验证**：

$$\sum_{k=1}^{\infty} (1-p)^{k-1} p = p \cdot \frac{1}{1-(1-p)} = 1 \checkmark$$

**期望与方差**：

$$E[X] = \frac{1}{p}, \quad \text{Var}(X) = \frac{1-p}{p^2}$$

直观地，成功概率越小，平均需要等待越久。

### 7.3.2 无记忆性

几何分布具有**无记忆性**（Memoryless Property），即过去的失败不影响未来的预期：

$$P(X > m + n \mid X > m) = P(X > n), \quad m, n \geq 0$$

**证明**：

$$P(X > m + n \mid X > m) = \frac{P(X > m + n)}{P(X > m)} = \frac{(1-p)^{m+n}}{(1-p)^m} = (1-p)^n = P(X > n)$$

**几何分布是离散分布中唯一具有无记忆性的分布**（类比连续分布中的指数分布）。

**例7.4**：某算法每次迭代有 $p = 0.2$ 的概率收敛，则收敛所需迭代次数 $X \sim \text{Geom}(0.2)$。$E[X] = 5$，即平均需要 $5$ 次迭代。

### 7.3.3 负二项分布

将几何分布推广：在独立重复伯努利试验中，**第 $r$ 次成功**所需的试验次数服从负二项分布。

**定义**：若 $X$ 为第 $r$ 次成功时的总试验次数，则

$$P(X = k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}, \quad k = r, r+1, r+2, \ldots$$

记作 $X \sim \text{NB}(r, p)$。

理解：第 $k$ 次试验恰好是第 $r$ 次成功，意味着前 $k-1$ 次中恰好有 $r-1$ 次成功，第 $k$ 次必须成功。

**另一种等价参数化**（前 $r$ 次成功前的失败次数 $Y = X - r$）：

$$P(Y = k) = \binom{k+r-1}{k} p^r (1-p)^k, \quad k = 0, 1, 2, \ldots$$

**期望与方差**（总试验次数 $X$ 的）：

$$E[X] = \frac{r}{p}, \quad \text{Var}(X) = \frac{r(1-p)}{p^2}$$

**关系**：当 $r = 1$ 时，负二项分布退化为几何分布。若 $X_1, X_2, \ldots, X_r$ 独立同分布 $\text{Geom}(p)$，则 $X_1 + X_2 + \cdots + X_r \sim \text{NB}(r, p)$。

**例7.5**：某机器学习模型训练时，每个 epoch 有 $p = 0.3$ 的概率使验证集性能提升，求第 $3$ 次性能提升时的期望 epoch 数。

$$X \sim \text{NB}(3, 0.3), \quad E[X] = \frac{3}{0.3} = 10$$

---

## 7.4 超几何分布

### 7.4.1 定义

超几何分布描述**有限总体中不放回抽样**的分布，与二项分布（放回抽样或无穷总体）形成对比。

**设置**：总体中有 $N$ 个元素，其中 $K$ 个为"成功"，$N-K$ 个为"失败"。从中**不放回**地随机抽取 $n$ 个，$X$ 为抽到的成功数。

**PMF**：

$$P(X = k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}}, \quad \max(0, n-(N-K)) \leq k \leq \min(n, K)$$

记作 $X \sim \text{Hypergeometric}(N, K, n)$。

**直观理解**：从 $K$ 个成功中选 $k$ 个，从 $N-K$ 个失败中选 $n-k$ 个，占总选法 $\binom{N}{n}$ 的比例。

**期望与方差**：

$$E[X] = n \cdot \frac{K}{N}$$

$$\text{Var}(X) = n \cdot \frac{K}{N} \cdot \frac{N-K}{N} \cdot \frac{N-n}{N-1}$$

其中 $\frac{N-n}{N-1}$ 称为**有限总体修正因子**（Finite Population Correction Factor）。当 $N \to \infty$ 时，修正因子趋近于1，超几何分布趋近于二项分布 $B(n, K/N)$。

### 7.4.2 超几何分布与二项分布的比较

| 特征 | 二项分布 $B(n,p)$ | 超几何分布 $\text{Hyp}(N,K,n)$ |
|------|-------------------|-------------------------------|
| 抽样方式 | 放回抽样（或无限总体） | 不放回抽样 |
| 每次试验独立性 | 独立 | 不独立 |
| 成功概率 | 每次均为 $p$ | 随已抽情况变化 |
| 方差 | $np(1-p)$ | $np(1-p) \cdot \frac{N-n}{N-1}$ |

当 $n/N$ 较小（抽样比例小于5%）时，超几何分布可用二项分布近似。

**例7.6**：某数据集包含 $N = 1000$ 条样本，其中 $K = 300$ 条标注为正类。随机（不放回）抽取 $n = 50$ 条，$X$ 为正类样本数。

$$X \sim \text{Hyp}(1000, 300, 50)$$

$$E[X] = 50 \times \frac{300}{1000} = 15$$

$$\text{Var}(X) = 50 \times \frac{300}{1000} \times \frac{700}{1000} \times \frac{950}{999} \approx 10.36$$

若改用二项分布近似（$p = 0.3$）：$\text{Var}(X) = 50 \times 0.3 \times 0.7 = 10.5$，误差很小。

---

## 7.5 离散分布族的统一视角

### 7.5.1 指数族框架

许多常见离散分布（包括伯努利、二项、泊松、几何、负二项）都属于**指数族**（Exponential Family），其PMF可以写成统一形式：

$$p(x; \theta) = h(x) \exp\left(\eta(\theta)^T T(x) - A(\theta)\right)$$

其中：
- $\eta(\theta)$：自然参数（Natural Parameter）
- $T(x)$：充分统计量（Sufficient Statistic）
- $A(\theta)$：对数配分函数（Log-partition Function），保证归一化
- $h(x)$：基础测度

**伯努利分布的指数族形式**（以 $\text{Bernoulli}(p)$ 为例）：

$$p(x; p) = p^x(1-p)^{1-x} = \exp\left(x \log\frac{p}{1-p} + \log(1-p)\right)$$

其中 $\eta = \log\frac{p}{1-p}$（log-odds，即logit函数），$T(x) = x$，$A(\eta) = \log(1 + e^\eta)$。

指数族框架的优势：自动保证存在充分统计量，最大似然估计有封闭解，梯度下降计算简洁。

### 7.5.2 分布之间的关系图谱

```
Bernoulli(p)
    ↓ n次独立叠加
Binomial(n, p)
    ↓ n→∞, p→0, np=λ (泊松极限)
Poisson(λ)

Bernoulli(p)
    ↓ 等待首次成功
Geometric(p)
    ↓ 等待第r次成功
NegBinomial(r, p)

Binomial(n, K/N) ← N→∞近似
    Hypergeometric(N, K, n) ← 有限总体，不放回
```

### 7.5.3 选择分布的决策框架

在实际建模中，根据问题特征选择合适的分布：

| 问题特征 | 推荐分布 |
|----------|----------|
| 单次试验，二元结果 | 伯努利分布 |
| $n$ 次独立试验，统计成功次数 | 二项分布 |
| 固定时间/空间内，稀有事件计数 | 泊松分布 |
| 等待首次成功的试验次数 | 几何分布 |
| 等待第 $r$ 次成功的试验次数 | 负二项分布 |
| 有限总体，不放回抽样 | 超几何分布 |

### 7.5.4 最大似然估计（MLE）总结

给定 $n$ 个独立同分布观测值 $x_1, x_2, \ldots, x_n$：

- **Bernoulli$(p)$**：$\hat{p} = \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$
- **Poisson$(\lambda)$**：$\hat{\lambda} = \bar{x}$
- **Geometric$(p)$**：$\hat{p} = \frac{1}{\bar{x}}$

MLE的直观性：参数估计值等于样本均值（或其函数），体现了"用样本统计量估计总体参数"的核心思想。

---

## 本章小结

| 分布 | 记号 | PMF $p(k)$ | 期望 $E[X]$ | 方差 $\text{Var}(X)$ | 典型场景 |
|------|------|-----------|------------|---------------------|----------|
| 伯努利 | $\text{Bernoulli}(p)$ | $p^k(1-p)^{1-k}$，$k\in\{0,1\}$ | $p$ | $p(1-p)$ | 单次二元结果 |
| 二项 | $B(n,p)$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | $n$次独立试验成功数 |
| 泊松 | $P(\lambda)$ | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | 单位时间内事件计数 |
| 几何 | $\text{Geom}(p)$ | $(1-p)^{k-1}p$ | $\frac{1}{p}$ | $\frac{1-p}{p^2}$ | 首次成功等待次数 |
| 负二项 | $\text{NB}(r,p)$ | $\binom{k-1}{r-1}p^r(1-p)^{k-r}$ | $\frac{r}{p}$ | $\frac{r(1-p)}{p^2}$ | 第$r$次成功等待次数 |
| 超几何 | $\text{Hyp}(N,K,n)$ | $\frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}$ | $n\frac{K}{N}$ | $n\frac{K}{N}\frac{N-K}{N}\frac{N-n}{N-1}$ | 不放回抽样 |

**核心关系**：
- 伯努利 $\xrightarrow{n次叠加}$ 二项 $\xrightarrow{n\to\infty,p\to0,np=\lambda}$ 泊松
- 几何 $\xrightarrow{r次叠加}$ 负二项
- 超几何 $\xrightarrow{N\to\infty}$ 二项

---

## 深度学习应用

### 应用一：伯努利分布与二元交叉熵损失（BCE）

深度学习中的二分类问题（如图像分类、情感分析）本质上是参数为 $p = \sigma(\mathbf{w}^T \mathbf{x})$ 的伯努利分布建模，其中 $\sigma$ 为 sigmoid 函数。

**概率论推导**：给定 $n$ 个样本 $(x_i, y_i)$，$y_i \in \{0, 1\}$，模型预测 $\hat{p}_i = P(Y=1 \mid x_i)$，对数似然为

$$\log L = \sum_{i=1}^n \left[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \right]$$

最大化对数似然等价于最小化**二元交叉熵损失**（Binary Cross-Entropy Loss）：

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \right]$$

这正是伯努利分布负对数似然的均值形式。

```python
import torch
import torch.nn as nn
import numpy as np

# ——— 示例1：BCE损失与伯努利分布的等价性 ———
torch.manual_seed(42)
n = 100
# 真实标签（伯努利分布采样）
p_true = 0.7
y_true = torch.bernoulli(torch.full((n,), p_true))

# 模型预测概率（这里用固定值演示）
p_pred = torch.full((n,), 0.65)

# PyTorch BCE损失
bce_loss = nn.BCELoss()
loss_pytorch = bce_loss(p_pred, y_true)

# 手动计算负对数似然（等价形式）
loss_manual = -(y_true * torch.log(p_pred) + (1 - y_true) * torch.log(1 - p_pred)).mean()

print(f"PyTorch BCE Loss: {loss_pytorch:.4f}")
print(f"Manual NLL Loss:  {loss_manual:.4f}")
print(f"两者差异: {abs(loss_pytorch - loss_manual):.8f}")

# 输出示例：
# PyTorch BCE Loss: 0.6841
# Manual NLL Loss:  0.6841
# 两者差异: 0.00000000
```

### 应用二：泊松分布与事件计数建模

泊松回归（Poisson Regression）用于预测计数型输出（如用户点击次数、单词出现频率），模型输出 $\hat{\lambda} = \exp(\mathbf{w}^T \mathbf{x})$（保证非负）。

**泊松负对数似然损失**：

$$\mathcal{L}_{\text{Poisson}} = \frac{1}{n} \sum_{i=1}^n \left[ \hat{\lambda}_i - y_i \log \hat{\lambda}_i \right] + \text{const}$$

```python
import torch
import torch.nn as nn

# ——— 示例2：泊松分布用于计数建模 ———
class PoissonRegressor(nn.Module):
    """泊松回归模型：输出事件发生率 λ"""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # 使用 softplus 或 exp 保证 λ > 0
        return torch.nn.functional.softplus(self.linear(x)).squeeze(-1)

# 泊松负对数似然损失
def poisson_nll_loss(y_pred_lambda, y_true):
    """
    泊松负对数似然：-log P(Y=y|λ) = λ - y*log(λ) + log(y!)
    PyTorch内置: nn.PoissonNLLLoss
    """
    return (y_pred_lambda - y_true * torch.log(y_pred_lambda + 1e-8)).mean()

torch.manual_seed(0)
input_dim = 5
batch_size = 32

# 模拟数据：特征 x，真实计数 y（泊松分布）
x = torch.randn(batch_size, input_dim)
true_lambda = 3.0
y = torch.poisson(torch.full((batch_size,), true_lambda))

model = PoissonRegressor(input_dim)
y_pred = model(x)

loss = poisson_nll_loss(y_pred, y)
print(f"Poisson NLL Loss: {loss.item():.4f}")
print(f"预测 λ 均值: {y_pred.mean().item():.4f}")
print(f"真实计数均值: {y.mean().item():.4f}")

# PyTorch内置Poisson损失（等价）
criterion = nn.PoissonNLLLoss(log_input=False, full=False)
loss_builtin = criterion(y_pred, y)
print(f"PyTorch 内置 Poisson Loss: {loss_builtin.item():.4f}")
```

### 应用三：几何分布与序列建模中的停止概率

在自回归语言模型（如 GPT）的序列生成中，每个时间步生成终止符（`<EOS>`）的概率隐含了几何分布假设。若模型在每步以概率 $p$ 生成 `<EOS>`，则序列长度 $L \sim \text{Geom}(p)$，期望长度为 $1/p$。

```python
import torch
import torch.nn.functional as F

# ——— 示例3：序列长度的几何分布建模 ———
def simulate_sequence_lengths(p_stop, num_sequences=10000, max_len=200):
    """
    模拟自回归模型生成序列长度分布。
    每步以概率 p_stop 停止，序列长度服从 Geom(p_stop)。
    """
    lengths = []
    for _ in range(num_sequences):
        length = 1
        while length < max_len:
            if torch.bernoulli(torch.tensor(p_stop)).item() == 1:
                break
            length += 1
        lengths.append(length)
    return torch.tensor(lengths, dtype=torch.float)

# 理论期望：E[L] = 1/p
p_stop = 0.1
lengths = simulate_sequence_lengths(p_stop, num_sequences=5000)

theoretical_mean = 1 / p_stop
theoretical_std = (1 - p_stop) ** 0.5 / p_stop

print(f"停止概率 p = {p_stop}")
print(f"理论期望长度: {theoretical_mean:.1f}")
print(f"模拟均值:     {lengths.mean().item():.2f}")
print(f"理论标准差:   {theoretical_std:.2f}")
print(f"模拟标准差:   {lengths.std().item():.2f}")

# 负对数似然：在序列建模中鼓励适当长度
# 若想让模型学习在特定长度停止，可在损失中加入长度惩罚项
def length_penalty_loss(log_probs_eos, target_length, p_stop_target=0.1):
    """
    基于几何分布的序列长度正则化损失。
    log_probs_eos: 每步生成EOS的对数概率, shape (batch, seq_len)
    """
    seq_len = log_probs_eos.shape[1]
    # 期望停止步数服从 Geom(p_stop_target)
    # 简化：用MSE鼓励平均停止概率接近目标
    mean_p_stop = log_probs_eos.exp().mean()
    target_p = torch.tensor(p_stop_target)
    return F.mse_loss(mean_p_stop, target_p)
```

### 综合示例：离散分布的可视化比较

```python
import torch
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('离散分布族 PMF 比较', fontsize=16)

k_range = np.arange(0, 20)

# 1. 伯努利分布
ax = axes[0, 0]
p_vals = [0.3, 0.5, 0.7]
for p in p_vals:
    pmf = [p if k == 1 else (1-p) if k == 0 else 0 for k in [0, 1]]
    ax.bar([0, 1], pmf, alpha=0.6, label=f'p={p}', width=0.2,
           align='center')
ax.set_title('伯努利分布 Bernoulli(p)')
ax.set_xlabel('k')
ax.set_ylabel('P(X=k)')
ax.legend()

# 2. 二项分布
ax = axes[0, 1]
n = 20
for p in [0.3, 0.5, 0.7]:
    pmf = stats.binom.pmf(k_range[:21], n, p)
    ax.plot(k_range[:21], pmf, 'o-', label=f'n={n}, p={p}', markersize=4)
ax.set_title('二项分布 Binomial(n, p)')
ax.set_xlabel('k')
ax.set_ylabel('P(X=k)')
ax.legend()

# 3. 泊松分布
ax = axes[0, 2]
for lam in [1, 3, 7]:
    pmf = stats.poisson.pmf(k_range, lam)
    ax.plot(k_range, pmf, 'o-', label=f'λ={lam}', markersize=4)
ax.set_title('泊松分布 Poisson(λ)')
ax.set_xlabel('k')
ax.set_ylabel('P(X=k)')
ax.legend()

# 4. 几何分布
ax = axes[1, 0]
k_geom = np.arange(1, 21)
for p in [0.2, 0.5, 0.8]:
    pmf = stats.geom.pmf(k_geom, p)
    ax.plot(k_geom, pmf, 'o-', label=f'p={p}', markersize=4)
ax.set_title('几何分布 Geom(p)')
ax.set_xlabel('k（首次成功的试验次数）')
ax.set_ylabel('P(X=k)')
ax.legend()

# 5. 负二项分布（失败次数的PMF）
ax = axes[1, 1]
k_nb = np.arange(0, 20)
r = 5
for p in [0.3, 0.5, 0.7]:
    pmf = stats.nbinom.pmf(k_nb, r, p)
    ax.plot(k_nb, pmf, 'o-', label=f'r={r}, p={p}', markersize=4)
ax.set_title('负二项分布 NB(r, p)（失败次数）')
ax.set_xlabel('k（成功前的失败次数）')
ax.set_ylabel('P(Y=k)')
ax.legend()

# 6. 超几何分布
ax = axes[1, 2]
N, n_draw = 50, 10
for K in [10, 25, 40]:
    pmf = stats.hypergeom.pmf(k_range[:11], N, K, n_draw)
    ax.plot(k_range[:11], pmf, 'o-', label=f'N={N}, K={K}, n={n_draw}',
            markersize=4)
ax.set_title('超几何分布 Hyp(N, K, n)')
ax.set_xlabel('k（抽到的成功数）')
ax.set_ylabel('P(X=k)')
ax.legend()

plt.tight_layout()
plt.savefig('discrete_distributions.png', dpi=150)
print("图像已保存为 discrete_distributions.png")
```

---

## 练习题

**练习7.1**（基础）

某深度学习模型对每张图片独立地以概率 $p = 0.85$ 正确分类。

(1) 设 $X$ 为10张图片中正确分类的数量，写出 $X$ 的分布并计算 $P(X \geq 9)$。

(2) 计算 $E[X]$ 和 $\text{Var}(X)$。

(3) 若要使至少 $95\%$ 的概率保证10张图片全部分类正确，需要准确率 $p$ 至少为多少？

---

**练习7.2**（中等）

某网站每小时平均接收 $\lambda = 4$ 次异常请求（泊松分布）。

(1) 求某小时内恰好收到 $0$ 次异常请求的概率。

(2) 求某小时内收到不超过 $2$ 次异常请求的概率。

(3) 设安全系统每 $30$ 分钟检查一次，求两次检查之间恰好有 $3$ 次异常请求的概率。

(4) 若 $n = 1000$，$p = 0.004$，用泊松近似计算 $B(1000, 0.004)$ 中 $P(X = 3)$，并与精确值比较。

---

**练习7.3**（中等）

在强化学习中，智能体在某状态下每次行动有 $p = 0.15$ 的概率找到奖励。

(1) 设 $X$ 为第一次获得奖励所需的行动次数，求 $E[X]$、$\text{Var}(X)$ 及 $P(X > 10)$。

(2) 利用无记忆性，若智能体已经行动了 $5$ 次仍未获奖，求再至少行动 $5$ 次才能获奖的概率。

(3) 设 $Y$ 为获得第 $3$ 次奖励所需的总行动次数，$Y \sim \text{NB}(3, 0.15)$，求 $E[Y]$。

---

**练习7.4**（中等）

某数据集有 $N = 200$ 条样本，其中 $K = 60$ 条为正类样本。不放回地随机抽取 $n = 20$ 条。

(1) 设 $X$ 为抽到的正类样本数，写出 $X$ 的分布，计算 $E[X]$ 和 $\text{Var}(X)$。

(2) 计算 $P(X = 6)$。

(3) 若改用二项分布近似（$p = 60/200 = 0.3$），计算近似的 $\text{Var}(X)$ 并与精确值比较，计算相对误差。

---

**练习7.5**（深度学习应用）

设二分类神经网络对样本 $x$ 的输出为 $\hat{p} = \sigma(z)$，其中 $z \in \mathbb{R}$ 为 logit，$\sigma(z) = \frac{1}{1+e^{-z}}$。

(1) 写出单个样本 $(x, y)$，$y \in \{0,1\}$ 的伯努利对数似然 $\log P(Y=y \mid x)$，并说明它与BCE损失的关系。

(2) 对伯努利对数似然关于 $z$ 求导（链式法则），结合 $\sigma'(z) = \sigma(z)(1-\sigma(z))$，证明梯度为 $\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = \hat{p} - y$（注意符号：对负对数似然求导）。

(3) 解释为什么使用交叉熵损失（最大似然原则）比使用均方误差损失（MSE）更适合分类问题（从概率模型和梯度行为两方面分析）。

---

## 练习答案

**答案7.1**

(1) $X \sim B(10, 0.85)$。

$$P(X \geq 9) = P(X=9) + P(X=10)$$
$$= \binom{10}{9}(0.85)^9(0.15)^1 + \binom{10}{10}(0.85)^{10}(0.15)^0$$
$$= 10 \times 0.85^9 \times 0.15 + 0.85^{10}$$
$$\approx 10 \times 0.2316 \times 0.15 + 0.1969$$
$$\approx 0.3474 + 0.1969 = 0.5443$$

(2) $E[X] = np = 10 \times 0.85 = 8.5$，$\text{Var}(X) = np(1-p) = 10 \times 0.85 \times 0.15 = 1.275$。

(3) 要求 $P(X=10) \geq 0.95$，即 $p^{10} \geq 0.95$，因此 $p \geq 0.95^{1/10} = 0.95^{0.1} \approx 0.9949$。即准确率至少约需 $99.49\%$。

---

**答案7.2**

参数 $\lambda = 4$（每小时），$X \sim \text{Poisson}(4)$。

(1) $P(X = 0) = \frac{4^0 e^{-4}}{0!} = e^{-4} \approx 0.0183$。

(2) $P(X \leq 2) = P(X=0) + P(X=1) + P(X=2)$
$$= e^{-4}\left(1 + 4 + \frac{16}{2}\right) = 13e^{-4} \approx 13 \times 0.0183 \approx 0.2381$$

(3) 30分钟内，$\lambda' = 4 \times 0.5 = 2$，$Y \sim \text{Poisson}(2)$。

$$P(Y = 3) = \frac{2^3 e^{-2}}{3!} = \frac{8e^{-2}}{6} \approx \frac{8 \times 0.1353}{6} \approx 0.1804$$

(4) 泊松近似：$\lambda = np = 1000 \times 0.004 = 4$。

$$P_{\text{Poisson}}(X=3) = \frac{4^3 e^{-4}}{3!} = \frac{64 e^{-4}}{6} \approx \frac{64 \times 0.0183}{6} \approx 0.1954$$

精确二项值：$P_{\text{Binom}}(X=3) = \binom{1000}{3}(0.004)^3(0.996)^{997}$。

计算得 $\approx 0.1954$，两者高度吻合（相对误差 $< 0.1\%$）。

---

**答案7.3**

$X \sim \text{Geom}(0.15)$。

(1) $E[X] = \frac{1}{0.15} \approx 6.67$ 次，$\text{Var}(X) = \frac{1-0.15}{0.15^2} = \frac{0.85}{0.0225} \approx 37.78$。

$$P(X > 10) = (1-0.15)^{10} = 0.85^{10} \approx 0.1969$$

(2) 由无记忆性：$P(X > 10 \mid X > 5) = P(X > 5) = 0.85^5 \approx 0.4437$。

即使已经失败了5次，再至少失败5次（即总失败数超过10次）的概率仍为 $\approx 44.37\%$，与初始情况相同——这正是无记忆性的体现。

(3) $Y \sim \text{NB}(3, 0.15)$，$E[Y] = \frac{r}{p} = \frac{3}{0.15} = 20$ 次行动。

---

**答案7.4**

$X \sim \text{Hyp}(200, 60, 20)$。

(1) $E[X] = n \cdot \frac{K}{N} = 20 \times \frac{60}{200} = 6$。

$$\text{Var}(X) = 20 \times \frac{60}{200} \times \frac{140}{200} \times \frac{180}{199} = 20 \times 0.3 \times 0.7 \times \frac{180}{199} \approx 4.2 \times 0.9045 \approx 3.799$$

(2)

$$P(X=6) = \frac{\binom{60}{6}\binom{140}{14}}{\binom{200}{20}}$$

数值计算：$P(X=6) \approx 0.1651$（利用统计软件或递推公式）。

(3) 二项近似方差：$np(1-p) = 20 \times 0.3 \times 0.7 = 4.2$。

精确值 $\approx 3.799$，相对误差 $= \frac{4.2 - 3.799}{3.799} \approx 10.56\%$。

有限总体修正因子 $\frac{N-n}{N-1} = \frac{180}{199} \approx 0.9045$ 使方差缩小，因为不放回抽样降低了不确定性（已抽出的样本不再影响后续）。

---

**答案7.5**

(1) 伯努利对数似然为：

$$\log P(Y=y \mid x) = y \log \hat{p} + (1-y)\log(1-\hat{p})$$

BCE损失是负对数似然的均值：$\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^n \log P(Y=y_i \mid x_i)$，即**最小化BCE损失等价于最大化伯努利对数似然**。

(2) 对负对数似然 $\ell = -[y \log\hat{p} + (1-y)\log(1-\hat{p})]$ 求关于 $z$ 的导数：

$$\frac{\partial \ell}{\partial z} = \frac{\partial \ell}{\partial \hat{p}} \cdot \frac{\partial \hat{p}}{\partial z}$$

其中：
$$\frac{\partial \ell}{\partial \hat{p}} = -\frac{y}{\hat{p}} + \frac{1-y}{1-\hat{p}} = \frac{\hat{p} - y}{\hat{p}(1-\hat{p})}$$

$$\frac{\partial \hat{p}}{\partial z} = \sigma'(z) = \hat{p}(1-\hat{p})$$

因此：
$$\frac{\partial \ell}{\partial z} = \frac{\hat{p} - y}{\hat{p}(1-\hat{p})} \cdot \hat{p}(1-\hat{p}) = \hat{p} - y \quad \checkmark$$

这个**简洁的梯度形式**（预测值与真实值之差）是BCE损失相比MSE的重要优势之一。

(3) **从概率模型角度**：分类问题的目标变量服从伯努利分布，BCE损失是正确的最大似然目标。MSE对应高斯分布假设，与二元输出的实际分布不匹配。

**从梯度行为角度**：MSE损失为 $\frac{1}{2}(\hat{p}-y)^2$，关于 $z$ 的梯度为 $(\hat{p}-y)\hat{p}(1-\hat{p})$。当 $\hat{p} \approx 0$ 或 $\hat{p} \approx 1$ 时，$\hat{p}(1-\hat{p}) \approx 0$，即使预测严重错误（如 $\hat{p}=0.01, y=1$），梯度也接近零，导致**梯度消失**，训练极慢。BCE损失的梯度为 $\hat{p} - y$，当预测错误时梯度较大，训练更高效。

---

*本章完*
