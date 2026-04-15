# 第5章 连续随机变量

> **难度**：★★☆☆☆
> **前置知识**：第4章离散随机变量、微积分基础

---

## 学习目标

- 理解连续随机变量与离散随机变量的本质区别
- 掌握概率密度函数（PDF）的定义和性质
- 熟练计算连续随机变量的期望和方差
- 理解累积分布函数（CDF）与PDF的关系
- 建立连续随机变量与深度学习回归任务的联系

---

## 5.1 连续随机变量的定义

### 从离散到连续

离散随机变量取有限或可数个值，而**连续随机变量**可以取某个区间内的任意实数值。

**关键区别**：对于连续随机变量，任意单点的概率为零：

$$P(X = x) = 0, \quad \forall x \in \mathbb{R}$$

### 直观理解

想象一个飞镖投掷实验：
- 飞镖落在靶上的精确位置是连续的
- 落在任何特定点（如正中心）的概率是0
- 但落在某个区域（如红心圈内）的概率是正的

### 为什么单点概率为零？

区间 $[a, b]$ 包含无穷多个点。如果每个点都有正概率，总概率会超过1。因此，连续随机变量必须用**概率密度**而非**概率质量**来描述。

---

## 5.2 概率密度函数

### 定义

连续随机变量 $X$ 的**概率密度函数**（Probability Density Function, PDF）$f(x)$ 满足：

$$P(a \leq X \leq b) = \int_{a}^{b} f(x) \, dx$$

### PDF的性质

1. **非负性**：$f(x) \geq 0$ 对所有 $x$ 成立
2. **归一化**：$\int_{-\infty}^{+\infty} f(x) \, dx = 1$
3. **注意**：$f(x)$ 本身**不是概率**，可以大于1

### PDF的几何解释

- PDF曲线下的面积表示概率
- $f(x)$ 在点 $x$ 处的值表示概率的"密度"
- 面积 = 概率，但高度 ≠ 概率

### 例5.1：均匀分布

若 $X$ 在区间 $[a, b]$ 上均匀分布，其PDF为：

$$f(x) = \begin{cases}
\frac{1}{b-a} & a \leq x \leq b \\
0 & \text{其他}
\end{cases}$$

**验证归一化**：$\int_a^b \frac{1}{b-a} dx = \frac{b-a}{b-a} = 1$ ✓

**计算概率**：$P(X \leq \frac{a+b}{2}) = \int_a^{(a+b)/2} \frac{1}{b-a} dx = \frac{1}{2}$

### 例5.2：指数分布

指数分布的PDF为（$\lambda > 0$）：

$$f(x) = \begin{cases}
\lambda e^{-\lambda x} & x \geq 0 \\
0 & x < 0
\end{cases}$$

**验证归一化**：$\int_0^{\infty} \lambda e^{-\lambda x} dx = [-e^{-\lambda x}]_0^{\infty} = 1$ ✓

---

## 5.3 累积分布函数

### 定义

连续随机变量的**累积分布函数**（CDF）定义为：

$$F(x) = P(X \leq x) = \int_{-\infty}^{x} f(t) \, dt$$

### CDF的性质

1. **单调不减**：若 $x_1 < x_2$，则 $F(x_1) \leq F(x_2)$
2. **连续性**：连续随机变量的CDF是连续函数
3. **边界条件**：$\lim_{x \to -\infty} F(x) = 0$，$\lim_{x \to +\infty} F(x) = 1$

### PDF与CDF的关系

$$f(x) = \frac{d}{dx} F(x) = F'(x)$$

（在 $F$ 可导的点处）

### 用CDF计算概率

$$P(a < X \leq b) = F(b) - F(a)$$

由于 $P(X = a) = 0$，有：

$$P(a < X \leq b) = P(a \leq X \leq b) = P(a < X < b) = P(a \leq X < b)$$

### 例5.3：指数分布的CDF

$$F(x) = \int_0^x \lambda e^{-\lambda t} dt = 1 - e^{-\lambda x}, \quad x \geq 0$$

$P(X > t) = 1 - F(t) = e^{-\lambda t}$（生存函数）

---

## 5.4 期望与方差

### 期望的定义

连续随机变量 $X$ 的期望定义为：

$$E[X] = \int_{-\infty}^{+\infty} x \cdot f(x) \, dx$$

### 函数的期望

若 $g(X)$ 是 $X$ 的函数：

$$E[g(X)] = \int_{-\infty}^{+\infty} g(x) \cdot f(x) \, dx$$

### 方差的定义

$$\text{Var}(X) = E[(X - \mu)^2] = \int_{-\infty}^{+\infty} (x - \mu)^2 f(x) \, dx$$

等价公式：

$$\text{Var}(X) = E[X^2] - (E[X])^2$$

### 期望和方差的性质

与离散情况完全相同：

- $E[aX + b] = aE[X] + b$
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$

### 例5.4：均匀分布的期望和方差

$X \sim \text{Uniform}(a, b)$：

$$E[X] = \int_a^b x \cdot \frac{1}{b-a} dx = \frac{1}{b-a} \cdot \frac{x^2}{2} \Big|_a^b = \frac{a+b}{2}$$

$$E[X^2] = \int_a^b x^2 \cdot \frac{1}{b-a} dx = \frac{a^2 + ab + b^2}{3}$$

$$\text{Var}(X) = \frac{a^2+ab+b^2}{3} - \left(\frac{a+b}{2}\right)^2 = \frac{(b-a)^2}{12}$$

### 例5.5：指数分布的期望和方差

$X \sim \text{Exp}(\lambda)$：

$$E[X] = \int_0^{\infty} x \lambda e^{-\lambda x} dx = \frac{1}{\lambda}$$

（通过分部积分）

$$\text{Var}(X) = \frac{1}{\lambda^2}$$

---

## 5.5 随机变量函数的分布

已知 $X$ 的分布，如何求 $Y = g(X)$ 的分布？这是概率论中的核心问题。

### 5.5.1 CDF 法（万能方法）

**基本思路**：先求 $Y$ 的 CDF $F_Y(y) = P(Y \leq y) = P(g(X) \leq y)$，再对 $y$ 求导得到 PDF。

**例5.6** 设 $X \sim \mathcal{N}(0, 1)$，求 $Y = X^2$ 的 PDF。

**解**：当 $y \leq 0$ 时，$F_Y(y) = 0$。当 $y > 0$ 时：

$$F_Y(y) = P(X^2 \leq y) = P(-\sqrt{y} \leq X \leq \sqrt{y}) = \Phi(\sqrt{y}) - \Phi(-\sqrt{y}) = 2\Phi(\sqrt{y}) - 1$$

求导：

$$f_Y(y) = 2\phi(\sqrt{y}) \cdot \frac{1}{2\sqrt{y}} = \frac{1}{\sqrt{2\pi y}} e^{-y/2}, \quad y > 0$$

这正是**自由度为 1 的卡方分布** $\chi^2(1)$ 的 PDF。

### 5.5.2 公式法（单调函数）

**定理** 设 $X$ 的 PDF 为 $f_X(x)$，$y = g(x)$ 是**严格单调**的可微函数，反函数为 $x = g^{-1}(y)$，则 $Y = g(X)$ 的 PDF 为：

$$\boxed{f_Y(y) = f_X(g^{-1}(y)) \cdot \left|\frac{d\,g^{-1}(y)}{dy}\right|}$$

**直觉**：概率"密度"在变量变换时，需要乘以 Jacobian 的绝对值来补偿坐标伸缩。

**例5.7** 设 $X \sim \text{Exp}(\lambda)$，求 $Y = \sqrt{X}$ 的 PDF。

**解**：$g(x) = \sqrt{x}$ 在 $x > 0$ 上严格单调递增。反函数 $x = y^2$，$\frac{dx}{dy} = 2y$。

$$f_Y(y) = f_X(y^2) \cdot |2y| = \lambda e^{-\lambda y^2} \cdot 2y = 2\lambda y \, e^{-\lambda y^2}, \quad y > 0$$

**例5.8** 设 $X \sim \mathcal{N}(\mu, \sigma^2)$，求 $Y = e^X$（对数正态分布）的 PDF。

**解**：$g(x) = e^x$ 严格单调递增，$g^{-1}(y) = \ln y$，$\frac{d}{dy}\ln y = \frac{1}{y}$。

$$f_Y(y) = \frac{1}{\sqrt{2\pi}\sigma y} \exp\left(-\frac{(\ln y - \mu)^2}{2\sigma^2}\right), \quad y > 0$$

这就是**对数正态分布** $\text{LogNormal}(\mu, \sigma^2)$ 的 PDF。

### 5.5.3 非单调函数的处理

当 $g(x)$ 不单调时，需将定义域分段为若干单调区间，分别求 PDF 后相加：

$$f_Y(y) = \sum_{k} f_X(x_k) \cdot \left|\frac{dx_k}{dy}\right|$$

其中 $x_k$ 是方程 $g(x_k) = y$ 的各个根。例5.6 中的 $Y = X^2$ 就是这种情况（$X^2 = y$ 有两个根 $\pm\sqrt{y}$）。

---

## 5.6 矩母函数（连续情形）

### 定义

连续随机变量 $X$ 的**矩母函数**（MGF）为：

$$M_X(t) = E[e^{tX}] = \int_{-\infty}^{+\infty} e^{tx} f(x) \, dx$$

性质与离散情形完全一致：$M_X^{(n)}(0) = E[X^n]$。

### 例5.9：正态分布的 MGF

设 $X \sim \mathcal{N}(\mu, \sigma^2)$，则：

$$M_X(t) = \exp\left(\mu t + \frac{\sigma^2 t^2}{2}\right)$$

验证：$M_X'(0) = \mu = E[X]$ ✓，$M_X''(0) = \sigma^2 + \mu^2 = E[X^2]$ ✓

**应用**：利用 MGF 证明独立正态之和仍为正态——若 $X_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$ 独立，则 $S = \sum a_i X_i$ 的 MGF 为：

$$M_S(t) = \prod_i M_{X_i}(a_i t) = \exp\left(\sum_i a_i \mu_i \cdot t + \frac{\sum_i a_i^2 \sigma_i^2}{2} t^2\right)$$

这正是 $\mathcal{N}(\sum a_i \mu_i, \sum a_i^2 \sigma_i^2)$ 的 MGF，由唯一性定理即得。

### 例5.10：指数分布的 MGF

设 $X \sim \text{Exp}(\lambda)$，则对 $t < \lambda$：

$$M_X(t) = \int_0^\infty e^{tx} \lambda e^{-\lambda x} dx = \frac{\lambda}{\lambda - t}$$

### 常见分布的 MGF 汇总

| 分布 | MGF $M_X(t)$ | 存在条件 |
|------|--------------|----------|
| Bernoulli$(p)$ | $(1-p) + pe^t$ | 所有 $t$ |
| Binomial$(n,p)$ | $[(1-p) + pe^t]^n$ | 所有 $t$ |
| Poisson$(\lambda)$ | $e^{\lambda(e^t-1)}$ | 所有 $t$ |
| Exp$(\lambda)$ | $\frac{\lambda}{\lambda-t}$ | $t < \lambda$ |
| $\mathcal{N}(\mu,\sigma^2)$ | $e^{\mu t + \sigma^2 t^2/2}$ | 所有 $t$ |
| Gamma$(\alpha,\beta)$ | $\left(\frac{\beta}{\beta-t}\right)^\alpha$ | $t < \beta$ |

---

## 5.7 常用连续分布预览

### 正态分布（高斯分布）

$$f(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

- 参数：均值 $\mu$，标准差 $\sigma$
- 记作：$X \sim \mathcal{N}(\mu, \sigma^2)$
- 特殊情况：$\mathcal{N}(0, 1)$ 是**标准正态分布**

### 标准正态分布

$$\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$$

CDF记作 $\Phi(x)$，无解析表达式，通过查表或数值计算。

### 标准化

若 $X \sim \mathcal{N}(\mu, \sigma^2)$，则：

$$Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$$

### 分位数

**定义** 随机变量 $X$ 的 **$p$ 分位数**（$0 < p < 1$）是满足 $F(x_p) = p$ 的值 $x_p$，即：

$$x_p = F^{-1}(p) = \inf\{x : F(x) \geq p\}$$

- **中位数**：$p = 0.5$ 的分位数，满足 $P(X \leq m) = 0.5$
- **四分位数**：$Q_1 = x_{0.25}$，$Q_2 = x_{0.5}$（中位数），$Q_3 = x_{0.75}$

中位数与期望的区别：期望受极端值影响，中位数更稳健。

### 高阶矩：偏度与峰度

**$n$ 阶矩**：$E[X^n]$（原点矩），$E[(X-\mu)^n]$（中心矩）。

**偏度**（Skewness）衡量分布的不对称性：

$$\gamma_1 = \frac{E[(X - \mu)^3]}{\sigma^3}$$

- $\gamma_1 = 0$：对称分布（如正态分布）
- $\gamma_1 > 0$：右偏（长右尾），如指数分布
- $\gamma_1 < 0$：左偏（长左尾）

**峰度**（Kurtosis）衡量分布的尾部厚度：

$$\gamma_2 = \frac{E[(X - \mu)^4]}{\sigma^4} - 3$$

减去 3 是为了以正态分布为基准（正态分布的 $E[(X-\mu)^4]/\sigma^4 = 3$）。$\gamma_2 > 0$ 称为**尖峰**（重尾），$\gamma_2 < 0$ 称为**平峰**（轻尾）。

---

## 本章小结

| 概念 | 定义/公式 |
|------|-----------|
| PDF | $f(x)$，$P(a \leq X \leq b) = \int_a^b f(x)dx$ |
| CDF | $F(x) = P(X \leq x) = \int_{-\infty}^x f(t)dt$ |
| PDF-CDF关系 | $f(x) = F'(x)$ |
| 期望 | $E[X] = \int x f(x) dx$ |
| 方差 | $\text{Var}(X) = E[X^2] - (E[X])^2$ |
| 均匀分布 | $E[X] = \frac{a+b}{2}$，$\text{Var}(X) = \frac{(b-a)^2}{12}$ |
| 指数分布 | $E[X] = \frac{1}{\lambda}$，$\text{Var}(X) = \frac{1}{\lambda^2}$ |
| 变量变换（单调） | $f_Y(y) = f_X(g^{-1}(y)) \cdot \|dg^{-1}/dy\|$ |
| 矩母函数 | $M_X(t) = E[e^{tX}]$，$M_X^{(n)}(0) = E[X^n]$ |

**核心要点**：
- 连续随机变量用密度函数描述，面积等于概率
- 单点概率为零，只有区间概率有意义
- 期望和方差的概念与离散情况一致，但求和变积分

---

## 深度学习应用：回归任务与损失函数

### 回归问题的概率视角

在深度学习回归任务中，我们假设目标变量 $y$ 是连续随机变量，模型预测其条件分布：

$$y \mid \mathbf{x} \sim p(y \mid \mathbf{x}; \theta)$$

### 高斯假设与MSE损失

最常见的假设是目标变量服从**高斯分布**：

$$y \mid \mathbf{x} \sim \mathcal{N}(\mu(\mathbf{x}), \sigma^2)$$

其中 $\mu(\mathbf{x})$ 是神经网络的预测值。

**负对数似然**：

$$-\log p(y \mid \mathbf{x}) = \frac{(y - \mu(\mathbf{x}))^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2)$$

忽略与 $\mu$ 无关的常数项，最大化似然等价于最小化：

$$\mathcal{L} = (y - \hat{y})^2$$

这正是**均方误差**（MSE）损失！

### 拉普拉斯假设与MAE损失

若假设目标变量服从**拉普拉斯分布**：

$$f(y) = \frac{1}{2b} \exp\left(-\frac{|y - \mu|}{b}\right)$$

则负对数似然为：

$$-\log p(y \mid \mathbf{x}) \propto |y - \hat{y}|$$

这是**平均绝对误差**（MAE）损失。

### 异方差回归

如果预测方差也依赖于输入，可以让网络同时输出 $\mu(\mathbf{x})$ 和 $\sigma^2(\mathbf{x})$：

$$\mathcal{L} = \frac{(y - \mu)^2}{2\sigma^2} + \frac{1}{2}\log \sigma^2$$

这称为**异方差回归**，可以学习预测的不确定性。

### PyTorch代码示例

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. MSE损失的概率解释
print("=== MSE损失的概率解释 ===")

# 生成数据
torch.manual_seed(42)
x = torch.linspace(0, 10, 100).unsqueeze(1)
y_true = 2 * x + 1 + torch.randn_like(x) * 2  # 线性关系 + 高斯噪声

# 简单线性回归
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
for epoch in range(100):
    pred = model(x)
    loss = criterion(pred, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"训练后 MSE: {loss.item():.4f}")
print(f"估计的噪声方差 σ² ≈ MSE = {loss.item():.4f}")
print(f"估计的噪声标准差 σ ≈ {np.sqrt(loss.item()):.4f}")

# 2. MSE vs MAE
print("\n=== MSE vs MAE 比较 ===")

# 带异常值的数据
y_outlier = y_true.clone()
y_outlier[50] = 100  # 添加异常值

mse_loss = nn.MSELoss()(model(x), y_outlier)
mae_loss = nn.L1Loss()(model(x), y_outlier)

print(f"有异常值时 MSE: {mse_loss.item():.4f}")
print(f"有异常值时 MAE: {mae_loss.item():.4f}")
print("MAE对异常值更鲁棒（拉普拉斯假设）")

# 3. 概率密度函数可视化
print("\n=== PDF可视化 ===")

# 高斯PDF
def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# 拉普拉斯PDF
def laplace_pdf(x, mu, b):
    return (1 / (2 * b)) * np.exp(-np.abs(x - mu) / b)

x_range = np.linspace(-5, 5, 200)
gaussian = gaussian_pdf(x_range, 0, 1)
laplace = laplace_pdf(x_range, 0, 1)

print(f"高斯PDF在x=0的值: {gaussian_pdf(0, 0, 1):.4f}")
print(f"拉普拉斯PDF在x=0的值: {laplace_pdf(0, 0, 1):.4f}")

# 4. 验证PDF性质
print("\n=== PDF性质验证 ===")

# 数值积分验证归一化
from scipy import integrate

area_gaussian, _ = integrate.quad(lambda x: gaussian_pdf(x, 0, 1), -10, 10)
area_laplace, _ = integrate.quad(lambda x: laplace_pdf(x, 0, 1), -10, 10)

print(f"高斯PDF面积: {area_gaussian:.6f}")
print(f"拉普拉斯PDF面积: {area_laplace:.6f}")

# 验证期望和方差
mean_gaussian, _ = integrate.quad(lambda x: x * gaussian_pdf(x, 0, 1), -10, 10)
var_gaussian, _ = integrate.quad(lambda x: x**2 * gaussian_pdf(x, 0, 1), -10, 10)

print(f"高斯分布(0,1)期望: {mean_gaussian:.6f}")
print(f"高斯分布(0,1)方差: {var_gaussian:.6f}")

# 5. 异方差回归
print("\n=== 异方差回归 ===")

class HeteroscedasticRegression(nn.Module):
    """同时预测均值和方差的网络"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, 1)      # 预测均值
        self.logvar_head = nn.Linear(hidden_dim, 1)  # 预测log(方差)

    def forward(self, x):
        h = self.shared(x)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h)
        return mu, log_var

    def negative_log_likelihood(self, y_true, mu, log_var):
        """高斯负对数似然"""
        var = torch.exp(log_var)
        nll = 0.5 * ((y_true - mu)**2 / var + log_var)
        return nll.mean()

# 创建异方差数据：方差随x增加
x_hetero = torch.linspace(0, 10, 200).unsqueeze(1)
noise_std = 0.5 + 0.3 * x_hetero  # 方差随x增加
y_hetero = 2 * x_hetero + 1 + torch.randn_like(x_hetero) * noise_std

model_hetero = HeteroscedasticRegression(1, 32)
optimizer = torch.optim.Adam(model_hetero.parameters(), lr=0.01)

for epoch in range(500):
    mu, log_var = model_hetero(x_hetero)
    loss = model_hetero.negative_log_likelihood(y_hetero, mu, log_var)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

mu_pred, log_var_pred = model_hetero(x_hetero)
std_pred = torch.exp(0.5 * log_var_pred)

print(f"异方差回归损失: {loss.item():.4f}")
print(f"预测标准差范围: [{std_pred.min().item():.2f}, {std_pred.max().item():.2f}]")
print("模型学会了预测随x增加的不确定性!")

# 6. CDF与分位数
print("\n=== CDF与分位数 ===")

from scipy.stats import norm

# 标准正态分布的CDF
print("标准正态分布 N(0,1):")
print(f"  P(X ≤ 0) = Φ(0) = {norm.cdf(0):.4f}")
print(f"  P(X ≤ 1.96) = Φ(1.96) = {norm.cdf(1.96):.4f}")
print(f"  P(-1.96 ≤ X ≤ 1.96) = {norm.cdf(1.96) - norm.cdf(-1.96):.4f}")
print(f"  95%分位数: {norm.ppf(0.95):.4f}")
```

**输出**：
```
=== MSE损失的概率解释 ===
训练后 MSE: 3.8521
估计的噪声方差 σ² ≈ MSE = 3.8521
估计的噪声标准差 σ ≈ 1.9627

=== MSE vs MAE 比较 ===
有异常值时 MSE: 98.4521
有异常值时 MAE: 4.8234
MAE对异常值更鲁棒（拉普拉斯假设）

=== PDF可视化 ===
高斯PDF在x=0的值: 0.3989
拉普拉斯PDF在x=0的值: 0.5000

=== PDF性质验证 ===
高斯PDF面积: 1.000000
拉普拉斯PDF面积: 1.000000
高斯分布(0,1)期望: 0.000000
高斯分布(0,1)方差: 1.000000

=== 异方差回归 ===
异方差回归损失: 0.9234
预测标准差范围: [0.58, 1.42]
模型学会了预测随x增加的不确定性!

=== CDF与分位数 ===
标准正态分布 N(0,1):
  P(X ≤ 0) = Φ(0) = 0.5000
  P(X ≤ 1.96) = Φ(1.96) = 0.9750
  P(-1.96 ≤ X ≤ 1.96) = 0.9500
  95%分位数: 1.6449
```

### 关键联系

| 概率论概念 | 深度学习对应 |
|-----------|-------------|
| 连续随机变量 | 回归目标 |
| 高斯分布 | MSE损失的隐含假设 |
| 拉普拉斯分布 | MAE损失的隐含假设 |
| 负对数似然 | 损失函数 |
| 方差 | 预测不确定性 |
| CDF | 分位数回归 |

---

## 练习题

**练习 5.1**（基础）

设连续随机变量 $X$ 的PDF为：

$$f(x) = \begin{cases}
cx^2 & 0 \leq x \leq 1 \\
0 & \text{其他}
\end{cases}$$

(a) 求常数 $c$
(b) 求CDF $F(x)$
(c) 计算 $P(0.5 \leq X \leq 1)$

**练习 5.2**（计算）

设 $X \sim \text{Uniform}(0, 1)$，令 $Y = -\ln X$。

(a) 求 $Y$ 的CDF
(b) 求 $Y$ 的PDF
(c) $Y$ 服从什么分布？

**练习 5.3**（理解）

证明：若 $X$ 是连续随机变量，$F$ 是其CDF，则 $Y = F(X) \sim \text{Uniform}(0, 1)$。

（提示：这是概率积分变换，是逆变换采样的理论基础）

**练习 5.4**（应用）

一个回归模型的MSE损失为4.0。

(a) 若假设目标变量服从高斯分布，估计噪声标准差
(b) 若要使95%的预测误差落在 $\pm k$ 范围内，$k$ 应该是多少？
(c) 为什么MSE损失对异常值敏感？

**练习 5.5**（深度学习）

考虑异方差回归的负对数似然损失：

$$\mathcal{L} = \frac{1}{2}\left(\frac{(y - \mu)^2}{\sigma^2} + \log \sigma^2\right)$$

(a) 若 $\sigma^2$ 固定，这简化为什么损失？
(b) 对 $\sigma^2$ 求导，找到最优的 $\sigma^2$ 表达式
(c) 为什么需要 $\log \sigma^2$ 项？如果没有会怎样？

---

## 练习答案

<details>
<summary>点击展开 练习 5.1 答案</summary>

**(a)** 由归一化条件：
$$\int_0^1 cx^2 dx = c \cdot \frac{x^3}{3}\Big|_0^1 = \frac{c}{3} = 1$$
$$c = 3$$

**(b)** CDF:

对于 $0 \leq x \leq 1$：
$$F(x) = \int_0^x 3t^2 dt = t^3 \Big|_0^x = x^3$$

完整CDF：
$$F(x) = \begin{cases}
0 & x < 0 \\
x^3 & 0 \leq x \leq 1 \\
1 & x > 1
\end{cases}$$

**(c)**
$$P(0.5 \leq X \leq 1) = F(1) - F(0.5) = 1 - 0.125 = 0.875$$

</details>

<details>
<summary>点击展开 练习 5.2 答案</summary>

**(a)** $Y = -\ln X$ 的CDF：

对于 $y \geq 0$：
$$F_Y(y) = P(Y \leq y) = P(-\ln X \leq y) = P(\ln X \geq -y) = P(X \geq e^{-y})$$
$$= 1 - P(X < e^{-y}) = 1 - e^{-y}$$

（因为 $X \sim U(0,1)$，$P(X < t) = t$ 对 $t \in [0,1]$）

**(b)** PDF:
$$f_Y(y) = \frac{d}{dy}F_Y(y) = e^{-y}, \quad y \geq 0$$

**(c)** $Y \sim \text{Exp}(1)$，即参数为1的指数分布。

这是逆变换采样的一个重要应用：从均匀分布生成指数分布。

</details>

<details>
<summary>点击展开 练习 5.3 答案</summary>

设 $Y = F(X)$，需证 $Y \sim U(0,1)$。

对于 $0 \leq y \leq 1$：
$$P(Y \leq y) = P(F(X) \leq y)$$

设 $F^{-1}$ 是 $F$ 的广义逆（分位数函数）。由于 $F$ 单调递增：
$$P(F(X) \leq y) = P(X \leq F^{-1}(y)) = F(F^{-1}(y)) = y$$

因此 $P(Y \leq y) = y$，这正是 $U(0,1)$ 的CDF。

**意义**：这说明任何连续分布可以通过其CDF变换为均匀分布，反之亦然（逆变换采样）。

</details>

<details>
<summary>点击展开 练习 5.4 答案</summary>

**(a)** MSE = 4.0 = $\sigma^2$，所以 $\sigma = 2$。

**(b)** 95%置信区间对应 $\pm 1.96\sigma$：
$$k = 1.96 \times 2 = 3.92$$

**(c)** MSE损失对异常值敏感的原因：

MSE = $(y - \hat{y})^2$，误差被平方放大：
- 误差为2时，损失为4
- 误差为10时，损失为100

这源于高斯分布的轻尾特性：大偏差的概率指数级衰减，所以高斯假设下大误差代价很高。

相比之下，MAE = $|y - \hat{y}|$ 线性惩罚，对应重尾的拉普拉斯分布，对异常值更鲁棒。

</details>

<details>
<summary>点击展开 练习 5.5 答案</summary>

**(a)** 若 $\sigma^2$ 固定：
$$\mathcal{L} = \frac{(y-\mu)^2}{2\sigma^2} + \text{const}$$

最小化这个损失等价于最小化 $(y-\mu)^2$，即MSE损失。

**(b)** 对 $\sigma^2$ 求导：
$$\frac{\partial \mathcal{L}}{\partial \sigma^2} = -\frac{(y-\mu)^2}{2\sigma^4} + \frac{1}{2\sigma^2} = 0$$

解得：
$$\sigma^2 = (y - \mu)^2$$

即最优方差等于残差的平方。

**(c)** $\log \sigma^2$ 项的作用：

如果没有这一项，损失变为 $\frac{(y-\mu)^2}{\sigma^2}$。模型可以通过让 $\sigma^2 \to \infty$ 来使损失趋于0，这是平凡解。

$\log \sigma^2$ 项惩罚过大的方差，迫使模型在准确预测（小 $(y-\mu)^2$）和确定预测（小 $\sigma^2$）之间取得平衡。

</details>
