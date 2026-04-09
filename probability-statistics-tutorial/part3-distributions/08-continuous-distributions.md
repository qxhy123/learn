# 第8章：连续分布族

## 学习目标

学完本章后，你将能够：

1. 掌握均匀分布、正态分布、指数分布、Gamma分布和Beta分布的概率密度函数（PDF）与累积分布函数（CDF）
2. 理解正态分布的性质——包括对称性、68-95-99.7法则与标准化——并能熟练进行正态概率计算
3. 深刻理解指数分布的**无记忆性**，并能将其与泊松过程联系起来
4. 掌握Gamma分布和Beta分布的参数含义，理解它们作为其他分布的推广与联系
5. 将各连续分布应用于深度学习场景，包括权重初始化、批归一化分析和贝叶斯先验设定

---

## 8.1 均匀分布

### 定义与概率密度函数

若随机变量 $X$ 在区间 $[a, b]$ 上**等可能**地取值，则称 $X$ 服从**均匀分布**，记作 $X \sim \text{Uniform}(a, b)$ 或 $X \sim \mathcal{U}(a, b)$。

**概率密度函数（PDF）：**

$$f(x) = \begin{cases} \dfrac{1}{b-a}, & a \le x \le b \\ 0, & \text{其他} \end{cases}$$

**累积分布函数（CDF）：**

$$F(x) = \begin{cases} 0, & x < a \\ \dfrac{x-a}{b-a}, & a \le x \le b \\ 1, & x > b \end{cases}$$

### 期望与方差

$$E[X] = \frac{a+b}{2}$$

$$\text{Var}(X) = \frac{(b-a)^2}{12}$$

**推导方差：**

$$E[X^2] = \int_a^b x^2 \cdot \frac{1}{b-a}\, dx = \frac{1}{b-a} \cdot \frac{b^3 - a^3}{3} = \frac{a^2 + ab + b^2}{3}$$

$$\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{a^2+ab+b^2}{3} - \frac{(a+b)^2}{4} = \frac{(b-a)^2}{12}$$

### 标准均匀分布

当 $a=0, b=1$ 时，得到**标准均匀分布** $\mathcal{U}(0,1)$：

$$f(x) = 1, \quad 0 \le x \le 1$$

$$E[X] = \frac{1}{2}, \quad \text{Var}(X) = \frac{1}{12}$$

标准均匀分布是随机数生成的基础。通过**逆变换法**，可以从 $\mathcal{U}(0,1)$ 生成任意分布的随机样本：若 $U \sim \mathcal{U}(0,1)$，则 $X = F^{-1}(U)$ 服从 CDF 为 $F$ 的分布。

### 直觉理解

均匀分布代表**完全无偏的先验**——在没有任何额外信息时，区间内所有点的可能性相同。这在贝叶斯统计中称为**无信息先验**（noninformative prior）。

---

## 8.2 正态分布（高斯分布）

### 概率密度函数

若随机变量 $X$ 服从**正态分布**（Normal distribution），记作 $X \sim \mathcal{N}(\mu, \sigma^2)$，其 PDF 为：

$$f(x) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \quad x \in (-\infty, +\infty)$$

其中：
- $\mu$：均值（分布中心）
- $\sigma^2$：方差（控制分布宽窄）
- $\sigma$：标准差

### 标准正态分布

令 $Z = \dfrac{X - \mu}{\sigma}$，则 $Z \sim \mathcal{N}(0,1)$，称为**标准正态分布**。其 PDF 记作：

$$\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$$

其 CDF 记作 $\Phi(z)$，无闭合解析式，通常查表或用 `scipy.stats.norm.cdf` 计算。

**标准化变换的意义：** 将任意正态分布转化为标准正态，统一了计算框架。

### 期望与方差

$$E[X] = \mu, \quad \text{Var}(X) = \sigma^2$$

**验证期望**（利用奇函数对称性）：

$$E[X] = \int_{-\infty}^{+\infty} x \cdot \frac{1}{\sqrt{2\pi}\sigma} e^{-(x-\mu)^2/(2\sigma^2)} dx$$

令 $u = (x-\mu)/\sigma$，则：

$$E[X] = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{+\infty} (\sigma u + \mu) e^{-u^2/2} du = \mu$$

其中 $\int u e^{-u^2/2} du = 0$（奇函数在对称区间上积分为零）。

### 正态分布的关键性质

**1. 对称性**

$f(\mu + x) = f(\mu - x)$，分布关于均值 $\mu$ 完全对称，因此**偏度为 0，峰度为 3**（超额峰度为 0）。

**2. 68-95-99.7 法则**

$$P(\mu - \sigma \le X \le \mu + \sigma) \approx 68.27\%$$

$$P(\mu - 2\sigma \le X \le \mu + 2\sigma) \approx 95.45\%$$

$$P(\mu - 3\sigma \le X \le \mu + 3\sigma) \approx 99.73\%$$

**3. 线性组合的封闭性**

若 $X_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$ 相互独立，则：

$$\sum_{i=1}^n a_i X_i \sim \mathcal{N}\!\left(\sum_i a_i \mu_i,\; \sum_i a_i^2 \sigma_i^2\right)$$

**4. 矩母函数**

$$M_X(t) = E[e^{tX}] = \exp\!\left(\mu t + \frac{\sigma^2 t^2}{2}\right)$$

### 中心极限定理（CLT）预告

正态分布之所以在自然界中无处不在，根本原因是**中心极限定理**：大量独立同分布随机变量之和，在适当标准化后趋向正态分布。这将在第10章详细讨论。

### 正态概率计算示例

**例：** 若某班考试成绩 $X \sim \mathcal{N}(75, 10^2)$，求成绩在 85 分以上的概率。

$$P(X > 85) = P\!\left(Z > \frac{85-75}{10}\right) = P(Z > 1) = 1 - \Phi(1) \approx 1 - 0.8413 = 15.87\%$$

---

## 8.3 指数分布与无记忆性

### 定义与概率密度函数

**指数分布**描述"等待时间"——在泊松过程中，两次事件之间的等待时间服从指数分布。

若 $X \sim \text{Exp}(\lambda)$（$\lambda > 0$ 为速率参数），则：

$$f(x) = \lambda e^{-\lambda x}, \quad x \ge 0$$

$$F(x) = 1 - e^{-\lambda x}, \quad x \ge 0$$

### 期望与方差

$$E[X] = \frac{1}{\lambda}, \quad \text{Var}(X) = \frac{1}{\lambda^2}$$

**推导期望**（分部积分）：

$$E[X] = \int_0^\infty x \lambda e^{-\lambda x}\, dx = \left[-x e^{-\lambda x}\right]_0^\infty + \int_0^\infty e^{-\lambda x}\, dx = \frac{1}{\lambda}$$

### 无记忆性：指数分布的核心特征

**无记忆性（Memoryless Property）** 是指数分布最重要的性质：

$$P(X > s + t \mid X > s) = P(X > t), \quad \forall s, t \ge 0$$

**证明：**

$$P(X > s+t \mid X > s) = \frac{P(X > s+t)}{P(X > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(X > t)$$

**直觉解释：** "已经等了 $s$ 分钟"这一信息对未来等待时间没有任何影响——系统没有"记忆"。这就像一枚完美的硬币：无论已经抛了多少次正面，下一次正面的概率仍然是 $1/2$。

> **重要定理：** 在连续分布中，**指数分布是唯一具有无记忆性的分布**。

### 与泊松过程的联系

若事件以速率 $\lambda$ 按泊松过程发生（单位时间内事件数 $\sim \text{Poisson}(\lambda)$），则：
- 相邻事件的等待时间 $\sim \text{Exp}(\lambda)$
- 等待第 $k$ 个事件的时间 $\sim \text{Gamma}(k, \lambda)$（见 8.4 节）

### 参数化方式

指数分布有两种常见参数化：
- **速率参数**（rate）：$f(x) = \lambda e^{-\lambda x}$，均值为 $1/\lambda$
- **尺度参数**（scale）：$f(x) = \frac{1}{\theta} e^{-x/\theta}$，均值为 $\theta$

其中 $\theta = 1/\lambda$。Python 的 `scipy.stats.expon` 使用尺度参数。

---

## 8.4 Gamma 分布与 Beta 分布

### Gamma 函数

Gamma 分布的定义基于 **Gamma 函数**：

$$\Gamma(\alpha) = \int_0^\infty x^{\alpha-1} e^{-x}\, dx, \quad \alpha > 0$$

**关键性质：**

$$\Gamma(\alpha+1) = \alpha \cdot \Gamma(\alpha) \quad \text{（递推关系）}$$

$$\Gamma(n) = (n-1)! \quad \text{（正整数时退化为阶乘）}$$

$$\Gamma\!\left(\frac{1}{2}\right) = \sqrt{\pi}$$

### Gamma 分布

**定义：** 若 $X \sim \text{Gamma}(\alpha, \beta)$（形状参数 $\alpha > 0$，速率参数 $\beta > 0$），则：

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0$$

**期望与方差：**

$$E[X] = \frac{\alpha}{\beta}, \quad \text{Var}(X) = \frac{\alpha}{\beta^2}$$

**特殊情形：**

| 参数取值 | 退化分布 |
|----------|----------|
| $\alpha = 1$ | $\text{Exp}(\beta)$（指数分布） |
| $\alpha = n/2,\ \beta = 1/2$ | $\chi^2(n)$（卡方分布） |
| $\alpha \in \mathbb{Z}^+$ | Erlang 分布（等待 $\alpha$ 个事件的时间） |

**直觉：** Gamma 分布是"等待第 $\alpha$ 个泊松事件所需时间"的分布，是指数分布的自然推广。

### Beta 分布

**定义：** 若 $X \sim \text{Beta}(\alpha, \beta)$（$\alpha, \beta > 0$），则：

$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 < x < 1$$

其中 **Beta 函数**：

$$B(\alpha, \beta) = \int_0^1 x^{\alpha-1}(1-x)^{\beta-1}\, dx = \frac{\Gamma(\alpha)\,\Gamma(\beta)}{\Gamma(\alpha+\beta)}$$

**期望与方差：**

$$E[X] = \frac{\alpha}{\alpha+\beta}, \quad \text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

**形状变化：**

| 参数关系 | 形状特征 |
|----------|----------|
| $\alpha = \beta = 1$ | 均匀分布 $\mathcal{U}(0,1)$ |
| $\alpha = \beta > 1$ | 对称单峰，峰在 $x=0.5$ |
| $\alpha = \beta < 1$ | U 形，两端密度高 |
| $\alpha > \beta$ | 右偏，峰靠近 1 |
| $\alpha < \beta$ | 左偏，峰靠近 0 |
| $\alpha, \beta \gg 1$ | 趋近于正态分布 |

**为何 Beta 分布在贝叶斯统计中极为重要？**

Beta 分布是**二项分布成功概率 $p$ 的共轭先验**：

$$p \sim \text{Beta}(\alpha, \beta) \;\Longrightarrow\; p \mid \text{data} \sim \text{Beta}(\alpha + k,\ \beta + n - k)$$

其中 $n$ 次试验中观察到 $k$ 次成功。这种"更新只需修改参数"的性质极大简化了贝叶斯推断。

---

## 8.5 分布之间的关系

连续分布族并非相互孤立，它们之间存在丰富的联系。

### 层级关系图

```
均匀分布 U(0,1)
    │
    │ 逆变换法可生成任意分布
    ▼
正态分布 N(μ, σ²)
    │                    │
    │ 平方              │ 线性组合
    ▼                    ▼
卡方分布 χ²(n)        正态混合
    │
    │ 推广（非中心 χ²/加速率参数）
    ▼
Gamma(α, β)
    ├── α=1 → 指数分布 Exp(β)
    └── α=n/2, β=1/2 → χ²(n)

Beta(α, β)
    ├── α=β=1 → U(0,1)
    └── 与 Gamma 的比值关系（见下）
```

### 关键转换关系

**1. 正态 → 卡方**

若 $Z_i \sim \mathcal{N}(0,1)$ 独立，则：

$$\sum_{i=1}^n Z_i^2 \sim \chi^2(n) = \text{Gamma}\!\left(\frac{n}{2}, \frac{1}{2}\right)$$

**2. Gamma → Beta**

若 $X \sim \text{Gamma}(\alpha, \theta)$ 与 $Y \sim \text{Gamma}(\beta, \theta)$ 独立，则：

$$\frac{X}{X+Y} \sim \text{Beta}(\alpha, \beta)$$

**3. 指数的最小值**

若 $X_i \sim \text{Exp}(\lambda_i)$ 相互独立，则：

$$\min(X_1, \ldots, X_n) \sim \text{Exp}(\lambda_1 + \cdots + \lambda_n)$$

**4. 正态的绝对值**

若 $Z \sim \mathcal{N}(0,1)$，则 $|Z|$ 服从**半正态分布（Half-Normal）**，$Z^2 \sim \chi^2(1) = \text{Gamma}(1/2, 1/2)$。

**5. 均匀 → 指数**

若 $U \sim \mathcal{U}(0,1)$，则 $-\frac{1}{\lambda}\ln U \sim \text{Exp}(\lambda)$（逆变换法的具体应用）。

### 分布比较汇总

| 分布 | 支撑 | 参数 | 均值 | 方差 | 特征 |
|------|------|------|------|------|------|
| $\mathcal{U}(a,b)$ | $[a,b]$ | $a,b$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | 等可能 |
| $\mathcal{N}(\mu,\sigma^2)$ | $\mathbb{R}$ | $\mu,\sigma^2$ | $\mu$ | $\sigma^2$ | 对称钟形 |
| $\text{Exp}(\lambda)$ | $[0,\infty)$ | $\lambda$ | $1/\lambda$ | $1/\lambda^2$ | 无记忆性 |
| $\text{Gamma}(\alpha,\beta)$ | $(0,\infty)$ | $\alpha,\beta$ | $\alpha/\beta$ | $\alpha/\beta^2$ | 推广指数 |
| $\text{Beta}(\alpha,\beta)$ | $(0,1)$ | $\alpha,\beta$ | $\frac{\alpha}{\alpha+\beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ | 概率先验 |

---

## 本章小结

| 概念 | 核心公式 | 关键意义 |
|------|----------|----------|
| 均匀分布 $\mathcal{U}(a,b)$ | $f(x)=\frac{1}{b-a}$ | 无信息先验；随机数基础 |
| 正态分布 $\mathcal{N}(\mu,\sigma^2)$ | $f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | CLT 极限；自然界普遍存在 |
| 指数分布 $\text{Exp}(\lambda)$ | $f(x)=\lambda e^{-\lambda x}$ | 等待时间；唯一连续无记忆分布 |
| Gamma 分布 $\text{Gamma}(\alpha,\beta)$ | $f(x)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$ | 多个指数之和；等待 $k$ 次事件 |
| Beta 分布 $\text{Beta}(\alpha,\beta)$ | $f(x)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$ | 概率的先验；共轭先验 |
| 无记忆性 | $P(X>s+t\mid X>s)=P(X>t)$ | 指数分布的唯一性定理 |
| 标准化 | $Z=(X-\mu)/\sigma$ | 正态计算的统一框架 |
| Gamma-Beta 关系 | $\frac{X}{X+Y}\sim\text{Beta}(\alpha,\beta)$ | 分布族的内在统一性 |

---

## 深度学习应用：权重初始化、归一化与贝叶斯先验

### 背景

概率分布在深度学习中无处不在：
- **初始化**：权重的初始分布影响训练稳定性
- **归一化**：批归一化使激活值趋近正态分布
- **贝叶斯深度学习**：用概率分布表示权重的不确定性

### 代码实现

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. 权重初始化策略与分布分析
# ============================================================

def analyze_initialization(init_fn, name, n_weights=10000):
    """分析不同初始化策略产生的权重分布"""
    layer = nn.Linear(100, 100)
    init_fn(layer.weight)
    weights = layer.weight.detach().numpy().flatten()

    mean, std = weights.mean(), weights.std()
    print(f"{name}:")
    print(f"  均值 = {mean:.4f}, 标准差 = {std:.4f}")
    print(f"  范围 = [{weights.min():.4f}, {weights.max():.4f}]")
    return weights

# Xavier 均匀初始化（均匀分布）
# 目标：使每层输入输出的方差一致
# 范围：[-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out))]
w_xavier_uniform = analyze_initialization(
    nn.init.xavier_uniform_, "Xavier 均匀 (Uniform)")

# Xavier 正态初始化（截断正态）
# std = √(2/(fan_in+fan_out))
w_xavier_normal = analyze_initialization(
    nn.init.xavier_normal_, "Xavier 正态 (Normal)")

# Kaiming 均匀初始化（ReLU 激活专用）
# 考虑 ReLU 只保留正半部分，方差补偿为 2
w_kaiming = analyze_initialization(
    lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu'),
    "Kaiming 正态 (He, ReLU)")

# ============================================================
# 2. 可视化权重初始化分布
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
configs = [
    (w_xavier_uniform, "Xavier 均匀初始化\n$\\mathcal{U}(-a, a)$", "steelblue"),
    (w_xavier_normal,  "Xavier 正态初始化\n$\\mathcal{N}(0, \\sigma^2)$", "darkorange"),
    (w_kaiming,        "Kaiming 正态初始化\n$\\mathcal{N}(0, 2/\\text{fan\\_in})$", "seagreen"),
]

for ax, (weights, title, color) in zip(axes, configs):
    ax.hist(weights, bins=60, density=True, color=color, alpha=0.7)
    x = np.linspace(weights.min(), weights.max(), 200)
    # 叠加正态拟合曲线
    mu, sigma = weights.mean(), weights.std()
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'k--', linewidth=2, label='正态拟合')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("权重值")
    ax.set_ylabel("概率密度")
    ax.legend()

plt.suptitle("深度学习权重初始化分布比较", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("weight_initialization.png", dpi=150, bbox_inches='tight')
print("\n图像已保存：weight_initialization.png")

# ============================================================
# 3. 批归一化（Batch Normalization）与正态分布
# ============================================================

print("\n" + "="*50)
print("批归一化分析")
print("="*50)

class SimpleNetWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)  # 归一化层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward_with_stats(self, x):
        """返回 BN 前后的激活统计信息"""
        z_before_bn = self.fc1(x)

        # BatchNorm 核心操作：
        # 1. 计算 mini-batch 的均值和方差
        # 2. 标准化：z_hat = (z - μ_batch) / √(σ²_batch + ε)
        # 3. 缩放平移：y = γ·z_hat + β（可学习参数）
        z_after_bn = self.bn1(z_before_bn)

        return z_before_bn.detach(), z_after_bn.detach()

model = SimpleNetWithBN()
model.eval()

# 用随机输入测试
x_test = torch.randn(128, 784)  # batch_size=128
with torch.no_grad():
    before_bn, after_bn = model.forward_with_stats(x_test)

print(f"BN 前激活：均值={before_bn.mean():.4f}, 方差={before_bn.var():.4f}")
print(f"BN 后激活：均值={after_bn.mean():.4f}, 方差={after_bn.var():.4f}")
print(f"BN 效果：将激活强制归一化为近似 N(0,1) 分布")

# ============================================================
# 4. 贝叶斯先验：Beta 分布在概率估计中的应用
# ============================================================

print("\n" + "="*50)
print("贝叶斯先验：Beta 分布更新")
print("="*50)

# 场景：估计模型在测试集上的准确率（二分类：对/错）
# 先验：Beta(α₀, β₀)，代表"初始信念"
# 似然：观察到 k 次正确，n-k 次错误
# 后验：Beta(α₀+k, β₀+n-k)（Beta 是二项分布的共轭先验）

def beta_bayesian_update(alpha_prior, beta_prior, n_correct, n_total):
    """
    贝叶斯更新：Beta 先验 + 二项似然 → Beta 后验

    参数：
        alpha_prior, beta_prior: 先验参数
        n_correct: 正确预测次数
        n_total: 总预测次数
    """
    n_wrong = n_total - n_correct
    alpha_post = alpha_prior + n_correct
    beta_post = beta_prior + n_wrong

    prior_mean = alpha_prior / (alpha_prior + beta_prior)
    post_mean = alpha_post / (alpha_post + beta_post)
    post_var = (alpha_post * beta_post) / \
               ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))
    post_std = np.sqrt(post_var)

    print(f"先验 Beta({alpha_prior}, {beta_prior})：期望准确率 = {prior_mean:.3f}")
    print(f"观测数据：{n_correct}/{n_total} 正确")
    print(f"后验 Beta({alpha_post}, {beta_post})：")
    print(f"  期望准确率 = {post_mean:.3f}")
    print(f"  标准差 = {post_std:.4f}")
    print(f"  95% 置信区间 ≈ [{post_mean-2*post_std:.3f}, {post_mean+2*post_std:.3f}]")
    return alpha_post, beta_post

# 情景1：无信息先验（均匀分布），观测100次
print("情景1：无信息先验 Beta(1,1)")
alpha_p, beta_p = beta_bayesian_update(1, 1, 87, 100)

print()
# 情景2：弱先验（认为准确率约 70%）
print("情景2：弱信息先验 Beta(7,3)，均值=0.7")
alpha_p, beta_p = beta_bayesian_update(7, 3, 87, 100)

print()
# 情景3：持续更新——顺序贝叶斯学习
print("情景3：顺序贝叶斯更新（每 20 次观测更新一次）")
alpha, beta_val = 1, 1  # 无信息先验
batches = [(16,20), (19,20), (14,20), (17,20), (21,25)]  # 每批 (正确, 总数)
for i, (k, n) in enumerate(batches):
    alpha = alpha + k
    beta_val = beta_val + (n - k)
    mean = alpha / (alpha + beta_val)
    print(f"  第{i+1}批 ({k}/{n}) → Beta({alpha}, {beta_val})，当前估计 = {mean:.3f}")

# ============================================================
# 5. 指数分布在神经网络中：Dropout 保留概率的几何分布连接
# ============================================================

print("\n" + "="*50)
print("Dropout 与几何/指数分布的联系")
print("="*50)

# Dropout：每个神经元以概率 p 被保留
# 一个神经元在连续 k 次前向传播中首次被 Dropout 关闭
# 等待次数近似服从几何分布（离散），其连续类比是指数分布

p_keep = 0.5  # 每次保留概率

# 模拟：某神经元需要多少次前向传播才首次被关闭
def simulate_first_dropout(p_keep, n_simulations=100000):
    """模拟首次被 dropout 的等待次数"""
    waiting_times = []
    for _ in range(n_simulations):
        count = 0
        while np.random.random() < p_keep:
            count += 1
        waiting_times.append(count)
    return np.array(waiting_times)

waiting = simulate_first_dropout(p_keep)
theoretical_mean = p_keep / (1 - p_keep)

print(f"Dropout 保留率 p = {p_keep}")
print(f"首次被关闭的等待次数：")
print(f"  模拟均值 = {waiting.mean():.3f}（理论均值 = {theoretical_mean:.3f}）")
print(f"  模拟方差 = {waiting.var():.3f}（理论方差 = {theoretical_mean/(1-p_keep):.3f}）")
print(f"  连续极限：等待时间 ~ Exp(λ = {1-p_keep})")
```

### 关键公式速查

| 应用场景 | 所用分布 | 公式/规则 |
|----------|----------|-----------|
| Xavier 均匀初始化 | $\mathcal{U}(-a, a)$ | $a = \sqrt{6/(n_\text{in}+n_\text{out})}$ |
| Xavier 正态初始化 | $\mathcal{N}(0, \sigma^2)$ | $\sigma = \sqrt{2/(n_\text{in}+n_\text{out})}$ |
| Kaiming 初始化（ReLU）| $\mathcal{N}(0, \sigma^2)$ | $\sigma = \sqrt{2/n_\text{in}}$ |
| 批归一化 | $\mathcal{N}(0, 1)$ | $\hat{z} = (z-\mu_B)/\sqrt{\sigma_B^2+\varepsilon}$ |
| 准确率贝叶斯估计 | $\text{Beta}(\alpha, \beta)$ | 后验：$\text{Beta}(\alpha+k, \beta+n-k)$ |
| Dropout 等待分布 | $\text{Exp}(\lambda)$ | $\lambda = 1 - p_\text{keep}$ |

### 设计启示

1. **初始化至关重要**：错误的初始化（如全零、方差过大）会导致梯度消失或爆炸。Xavier/Kaiming 初始化通过匹配各层的方差，显著提升训练稳定性。

2. **批归一化的概率意义**：BN 将每层激活强制拉回近似 $\mathcal{N}(0,1)$，避免"内部协变量偏移"（Internal Covariate Shift），相当于在每层输入上做标准化变换。

3. **贝叶斯视角的优势**：用 Beta 分布表示准确率的不确定性，比点估计更诚实——不仅给出"最佳猜测"，还量化了置信程度，尤其在小样本场景下价值巨大。

---

## 练习题

**练习 1（均匀分布）**

设 $X \sim \mathcal{U}(2, 8)$。

(a) 写出 $X$ 的 PDF 和 CDF。

(b) 计算 $P(3 \le X \le 6)$。

(c) 计算 $E[X]$ 和 $\text{Var}(X)$。

(d) 若 $Y = 2X - 1$，求 $Y$ 的分布及其期望和方差。

---

**练习 2（正态分布）**

某深度学习模型的某层激活值近似服从 $\mathcal{N}(0.5, 0.04)$（均值 $0.5$，方差 $0.04$）。

(a) 求激活值落在 $[0.3, 0.7]$ 之间的概率（标准化后查表或计算）。

(b) 若对该层做批归一化，新激活值的分布是什么？

(c) 利用 68-95-99.7 法则，求原始激活值超过 $0.9$ 的概率。

(d) 若该层有 1000 个神经元，期望有多少个激活值超过 $0.9$？

---

**练习 3（指数分布与无记忆性）**

某 Web 服务器处理请求的等待时间服从 $\text{Exp}(\lambda = 2)$（单位：分钟）。

(a) 求期望等待时间和标准差。

(b) 求等待时间超过 1 分钟的概率。

(c) 已知某请求已等待了 0.5 分钟，求还需再等超过 1 分钟的概率。

(d) (c) 的结果与 (b) 有何关系？这体现了什么性质？请写出严格的数学表达。

(e) 若现在有 3 台服务器并行处理，每台的等待时间独立且均服从 $\text{Exp}(2)$，求第一个请求被处理的等待时间分布及其期望。

---

**练习 4（Gamma 分布与 Beta 分布）**

(a) 若 $X \sim \text{Gamma}(3, 2)$，求 $E[X]$、$\text{Var}(X)$，并说明 $X$ 可以解释为什么随机变量之和。

(b) 验证：当 $\alpha = 1$ 时，$\text{Gamma}(1, \beta)$ 退化为 $\text{Exp}(\beta)$（写出 PDF 对比）。

(c) 若 $X \sim \text{Beta}(3, 7)$，求 $E[X]$ 和 $\text{Var}(X)$。直觉上，这个 Beta 分布描述的是一个成功率大约多少的二项试验的先验？

(d) 观察到 60 次试验中 12 次成功，以 $\text{Beta}(3, 7)$ 为先验，写出后验分布，并与先验的期望进行比较。

---

**练习 5（综合：分布之间的关系）**

(a) 若 $Z_1, Z_2 \sim \mathcal{N}(0,1)$ 独立，令 $W = Z_1^2 + Z_2^2$。求 $W$ 的分布（用 Gamma 分布表示）。

(b) 在 (a) 的基础上，令 $R = \sqrt{W}$。$R$ 称为**瑞利分布（Rayleigh distribution）**，其 PDF 为 $f_R(r) = r e^{-r^2/2}$（$r \ge 0$）。验证：$E[R] = \sqrt{\pi/2}$。

(c) 利用逆变换法，从 $U \sim \mathcal{U}(0,1)$ 出发，写出生成 $\text{Exp}(\lambda)$ 样本的公式，并说明为何取 $-\frac{1}{\lambda}\ln U$ 而非 $-\frac{1}{\lambda}\ln(1-U)$。

(d) 若 $X \sim \text{Gamma}(\alpha, \theta)$ 与 $Y \sim \text{Gamma}(\beta, \theta)$ 独立，证明 $\dfrac{X}{X+Y} \sim \text{Beta}(\alpha, \beta)$（提示：令 $U = X/(X+Y)$，$V = X+Y$，用雅可比行列式做变量替换）。

---

## 练习答案

### 答案 1

**(a)** $X \sim \mathcal{U}(2, 8)$，区间长度为 $b - a = 6$。

$$f(x) = \frac{1}{6}, \quad 2 \le x \le 8; \qquad f(x) = 0, \text{ 其他}$$

$$F(x) = \begin{cases} 0, & x < 2 \\ \dfrac{x-2}{6}, & 2 \le x \le 8 \\ 1, & x > 8 \end{cases}$$

**(b)** $P(3 \le X \le 6) = F(6) - F(3) = \dfrac{6-2}{6} - \dfrac{3-2}{6} = \dfrac{4}{6} - \dfrac{1}{6} = \dfrac{1}{2}$

**(c)** $E[X] = \dfrac{2+8}{2} = 5$，$\text{Var}(X) = \dfrac{(8-2)^2}{12} = \dfrac{36}{12} = 3$

**(d)** 由线性变换：$Y = 2X - 1 \sim \mathcal{U}(2\cdot2-1, 2\cdot8-1) = \mathcal{U}(3, 15)$

$$E[Y] = 2E[X] - 1 = 9, \quad \text{Var}(Y) = 4\,\text{Var}(X) = 12$$

---

### 答案 2

激活值 $X \sim \mathcal{N}(0.5, 0.04)$，即 $\mu = 0.5$，$\sigma = \sqrt{0.04} = 0.2$。

**(a)**

$$P(0.3 \le X \le 0.7) = P\!\left(\frac{0.3-0.5}{0.2} \le Z \le \frac{0.7-0.5}{0.2}\right) = P(-1 \le Z \le 1) \approx 68.27\%$$

**(b)** 批归一化将激活值标准化：$\hat{X} = \dfrac{X - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$，其中 $\mu_B \approx 0.5$，$\sigma_B^2 \approx 0.04$（若整批数据服从该分布），故 $\hat{X} \approx \mathcal{N}(0, 1)$。

**(c)**

$$0.9 = 0.5 + 2 \times 0.2 = \mu + 2\sigma$$

由 68-95-99.7 法则：$P(|X - \mu| \le 2\sigma) \approx 95.45\%$，故 $P(X > \mu + 2\sigma) \approx (1 - 0.9545)/2 \approx 2.275\%$。

**(d)** 期望超过 0.9 的神经元数 $\approx 1000 \times 2.275\% \approx 22.75 \approx 23$ 个。

---

### 答案 3

$X \sim \text{Exp}(2)$，即速率 $\lambda = 2$，均值 $= 1/\lambda = 0.5$ 分钟。

**(a)** $E[X] = 1/2 = 0.5$ 分钟，$\text{Std}(X) = 1/\lambda = 0.5$ 分钟（指数分布均值=标准差）。

**(b)** $P(X > 1) = e^{-2 \times 1} = e^{-2} \approx 0.1353$（约 13.5%）。

**(c)** 由无记忆性直接得：$P(X > 0.5 + 1 \mid X > 0.5) = P(X > 1) = e^{-2} \approx 0.1353$。

**验证：**

$$P(X > 1.5 \mid X > 0.5) = \frac{P(X > 1.5)}{P(X > 0.5)} = \frac{e^{-3}}{e^{-1}} = e^{-2} = P(X > 1) \checkmark$$

**(d)** (c) 的结果与 (b) **完全相同**，体现了**无记忆性**：

$$P(X > s+t \mid X > s) = P(X > t), \quad \forall s,t \ge 0$$

数学上等价于：$\bar{F}(s+t) = \bar{F}(s) \cdot \bar{F}(t)$，其中 $\bar{F}(x) = e^{-\lambda x}$。

**(e)** 设三台服务器等待时间 $X_1, X_2, X_3 \overset{\text{iid}}{\sim} \text{Exp}(2)$，第一个被处理意味着等待 $M = \min(X_1, X_2, X_3)$。

$$M \sim \text{Exp}(2+2+2) = \text{Exp}(6)$$

$$E[M] = \frac{1}{6} \approx 0.167 \text{ 分钟}$$

---

### 答案 4

**(a)** $X \sim \text{Gamma}(3, 2)$：

$$E[X] = \frac{\alpha}{\beta} = \frac{3}{2} = 1.5, \quad \text{Var}(X) = \frac{\alpha}{\beta^2} = \frac{3}{4} = 0.75$$

**解释：** $X$ 可解释为 3 个独立的 $\text{Exp}(2)$ 随机变量之和，即等待第 3 个速率为 2 的泊松事件所需的时间。

**(b)** 当 $\alpha = 1$ 时：

$$f_{\text{Gamma}(1,\beta)}(x) = \frac{\beta^1}{\Gamma(1)} x^{1-1} e^{-\beta x} = \beta e^{-\beta x} = f_{\text{Exp}(\beta)}(x) \checkmark$$

（因为 $\Gamma(1) = 0! = 1$）

**(c)** $X \sim \text{Beta}(3, 7)$（$\alpha=3, \beta=7, \alpha+\beta=10$）：

$$E[X] = \frac{3}{10} = 0.3, \quad \text{Var}(X) = \frac{3 \times 7}{10^2 \times 11} = \frac{21}{1100} \approx 0.019$$

直觉上，这描述了**成功率约为 30% 的二项试验**的先验信念，等价于先前已观察过 $3+7-2 = 8$ 次试验（其中 2 次成功的等效先验）。

**(d)** 观测：60 次中 12 次成功，先验 $\text{Beta}(3,7)$。

后验：$\text{Beta}(3+12,\ 7+(60-12)) = \text{Beta}(15,\ 55)$

$$E[X_\text{后验}] = \frac{15}{70} \approx 0.214$$

先验期望 $= 0.3$，观测频率 $= 12/60 = 0.2$，后验期望 $0.214$ 是两者的**加权平均**，向观测数据方向移动，体现了贝叶斯更新的"先验 + 数据 → 后验"逻辑。

---

### 答案 5

**(a)** $Z_1^2 \sim \chi^2(1) = \text{Gamma}(1/2, 1/2)$。由 Gamma 分布的可加性（形状参数相加）：

$$W = Z_1^2 + Z_2^2 \sim \chi^2(2) = \text{Gamma}(1, 1/2) = \text{Exp}(1/2)$$

（这是一个特例：$\chi^2(2)$ 就是均值为 2 的指数分布。）

**(b)**

$$E[R] = E[\sqrt{W}] = \int_0^\infty \sqrt{w} \cdot \frac{1}{2} e^{-w/2}\, dw$$

令 $u = w/2$，$w = 2u$，$dw = 2\,du$：

$$E[R] = \int_0^\infty \sqrt{2u} \cdot \frac{1}{2} e^{-u} \cdot 2\,du = \sqrt{2} \int_0^\infty u^{1/2} e^{-u}\, du = \sqrt{2}\,\Gamma\!\left(\frac{3}{2}\right) = \sqrt{2} \cdot \frac{\sqrt{\pi}}{2} = \sqrt{\frac{\pi}{2}} \checkmark$$

**(c)** 对 $X \sim \text{Exp}(\lambda)$，CDF 为 $F(x) = 1 - e^{-\lambda x}$，反函数为 $F^{-1}(u) = -\frac{1}{\lambda}\ln(1-u)$。

逆变换法：若 $U \sim \mathcal{U}(0,1)$，则 $X = F^{-1}(U) = -\frac{1}{\lambda}\ln(1-U) \sim \text{Exp}(\lambda)$。

**为何 $-\frac{1}{\lambda}\ln U$ 也正确：** 若 $U \sim \mathcal{U}(0,1)$，则 $1-U$ 也服从 $\mathcal{U}(0,1)$（对称性），故 $-\frac{1}{\lambda}\ln U \overset{d}{=} -\frac{1}{\lambda}\ln(1-U)$，两个公式在分布意义上等价。实践中直接用 $-\frac{1}{\lambda}\ln U$ 更简洁。

**(d)** **证明 $X/(X+Y) \sim \text{Beta}(\alpha, \beta)$：**

令 $U = X/(X+Y)$，$V = X+Y$。则 $X = UV$，$Y = (1-U)V$，$0 < U < 1$，$V > 0$。

**雅可比行列式：**

$$\frac{\partial(X,Y)}{\partial(U,V)} = \begin{vmatrix} V & U \\ -V & 1-U \end{vmatrix} = V(1-U) + UV = V$$

$(X,Y)$ 的联合 PDF：

$$f_{X,Y}(x,y) = \frac{\theta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\theta x} \cdot \frac{\theta^\beta}{\Gamma(\beta)} y^{\beta-1} e^{-\theta y}$$

变量替换后：

$$f_{U,V}(u,v) = \frac{\theta^{\alpha+\beta}}{\Gamma(\alpha)\Gamma(\beta)} (uv)^{\alpha-1}((1-u)v)^{\beta-1} e^{-\theta v} \cdot v$$

$$= \frac{\theta^{\alpha+\beta}}{\Gamma(\alpha)\Gamma(\beta)} u^{\alpha-1}(1-u)^{\beta-1} v^{\alpha+\beta-1} e^{-\theta v}$$

对 $v$ 从 $0$ 到 $\infty$ 积分（利用 Gamma 积分 $\int_0^\infty v^{\alpha+\beta-1} e^{-\theta v}\, dv = \Gamma(\alpha+\beta)/\theta^{\alpha+\beta}$）：

$$f_U(u) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} u^{\alpha-1}(1-u)^{\beta-1} = \frac{u^{\alpha-1}(1-u)^{\beta-1}}{B(\alpha,\beta)}$$

这正是 $\text{Beta}(\alpha, \beta)$ 的 PDF。$\blacksquare$
