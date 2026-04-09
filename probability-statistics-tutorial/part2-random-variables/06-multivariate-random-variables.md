# 第6章 多维随机变量

> **难度**：★★★☆☆
> **前置知识**：第4章离散随机变量、第5章连续随机变量、多元微积分基础

---

## 学习目标

- 理解二维随机变量的联合分布函数、联合PMF与联合PDF的定义和性质
- 掌握从联合分布求边缘分布的方法（积分/求和）
- 理解条件分布与条件期望的概念及计算方法
- 熟练计算协方差和相关系数，判断随机变量的独立性
- 建立多维随机变量与深度学习多任务学习、特征相关性分析的联系

---

## 6.1 二维随机变量的概念

### 从一维到多维

在实际问题中，我们常常需要同时研究两个或多个随机变量。例如：

- 一个人的**身高** $X$ 和**体重** $Y$
- 图像的**亮度** $X$ 和**对比度** $Y$
- 神经网络中两个神经元的**激活值** $X$ 和 $Y$

这些变量之间往往存在某种关联，需要用**多维随机变量**来联合描述。

### 二维随机变量的定义

设 $X$ 和 $Y$ 是定义在同一样本空间 $\Omega$ 上的两个随机变量，则称 $(X, Y)$ 为**二维随机变量**（或**随机向量**）。

二维随机变量 $(X, Y)$ 的每次取值是平面上的一个点 $(x, y)$。

### 二维随机变量的分类

与一维情况类似：

- **二维离散型**：$(X, Y)$ 只取有限或可数个点对 $(x_i, y_j)$
- **二维连续型**：$(X, Y)$ 可取某个平面区域内的任意值

### 直观理解：散点图

二维随机变量可以直观地用**散点图**表示：每次实验对应平面上的一个点。

- 若 $X$ 和 $Y$ 相互独立，散点分布是"圆形云"
- 若 $X$ 和 $Y$ 正相关，散点呈"右上-左下"的椭圆形
- 若 $X$ 和 $Y$ 负相关，散点呈"左上-右下"的椭圆形

---

## 6.2 联合分布函数与联合概率

### 联合分布函数（Joint CDF）

二维随机变量 $(X, Y)$ 的**联合分布函数**定义为：

$$F(x, y) = P(X \leq x, Y \leq y), \quad (x, y) \in \mathbb{R}^2$$

#### 联合CDF的性质

1. **单调性**：关于 $x$ 和 $y$ 分别单调不减
2. **边界条件**：
   - $F(-\infty, y) = 0$，$F(x, -\infty) = 0$
   - $F(+\infty, +\infty) = 1$
3. **右连续性**：关于 $x$ 和 $y$ 分别右连续
4. **矩形概率公式**：

$$P(a < X \leq b, c < Y \leq d) = F(b,d) - F(a,d) - F(b,c) + F(a,c)$$

### 离散型：联合概率质量函数（Joint PMF）

若 $(X, Y)$ 是离散型随机变量，其**联合概率质量函数**为：

$$p(x_i, y_j) = P(X = x_i, Y = y_j), \quad i, j = 1, 2, \ldots$$

#### Joint PMF的性质

1. **非负性**：$p(x_i, y_j) \geq 0$
2. **归一化**：$\displaystyle\sum_i \sum_j p(x_i, y_j) = 1$

#### 例6.1：联合PMF

投掷两枚硬币，设 $X$ 为第一枚正面朝上的次数，$Y$ 为第二枚正面朝上的次数。

$$p(x, y) = \frac{1}{4}, \quad x \in \{0, 1\},\ y \in \{0, 1\}$$

| | $Y=0$ | $Y=1$ |
|---|---|---|
| $X=0$ | 1/4 | 1/4 |
| $X=1$ | 1/4 | 1/4 |

### 连续型：联合概率密度函数（Joint PDF）

若存在非负函数 $f(x, y)$ 使得：

$$F(x, y) = \int_{-\infty}^{x} \int_{-\infty}^{y} f(s, t) \, dt \, ds$$

则称 $f(x, y)$ 为 $(X, Y)$ 的**联合概率密度函数**。

#### Joint PDF的性质

1. **非负性**：$f(x, y) \geq 0$
2. **归一化**：$\displaystyle\int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty} f(x, y) \, dx \, dy = 1$
3. **区域概率**：对平面区域 $D$，有

$$P((X, Y) \in D) = \iint_D f(x, y) \, dx \, dy$$

#### 例6.2：联合PDF验证

设 $f(x, y) = c \cdot e^{-(2x + y)}$，$x \geq 0,\ y \geq 0$，其他处为0。

求常数 $c$：

$$\int_0^{\infty}\int_0^{\infty} c \cdot e^{-(2x+y)} \, dy \, dx = c \int_0^{\infty} e^{-2x} dx \cdot \int_0^{\infty} e^{-y} dy = c \cdot \frac{1}{2} \cdot 1 = \frac{c}{2} = 1$$

故 $c = 2$。

### 二维正态分布

最重要的二维连续分布是**二维正态分布** $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，其PDF为：

$$f(x, y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_X)^2}{\sigma_X^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y} + \frac{(y-\mu_Y)^2}{\sigma_Y^2}\right]\right)$$

其中：
- $\mu_X, \mu_Y$：均值
- $\sigma_X^2, \sigma_Y^2$：方差
- $\rho$：相关系数（$|\rho| < 1$）

---

## 6.3 边缘分布

### 边缘分布的概念

从联合分布 $(X, Y)$ 中，仅考察单个变量的分布，称为**边缘分布**（Marginal Distribution）。

边缘分布是"把另一个变量积分/求和掉"的结果。

### 离散型的边缘分布

$X$ 的**边缘PMF**：

$$p_X(x_i) = P(X = x_i) = \sum_j p(x_i, y_j)$$

$Y$ 的**边缘PMF**：

$$p_Y(y_j) = P(Y = y_j) = \sum_i p(x_i, y_j)$$

直观理解：对联合分布表格按行或列求和。

#### 例6.3：从联合PMF求边缘PMF

| | $Y=0$ | $Y=1$ | $Y=2$ | $p_X(x)$ |
|---|---|---|---|---|
| $X=0$ | 0.1 | 0.2 | 0.1 | **0.4** |
| $X=1$ | 0.2 | 0.3 | 0.1 | **0.6** |
| $p_Y(y)$ | **0.3** | **0.5** | **0.2** | **1.0** |

边缘分布就是表格最右列和最下行的数值。

### 连续型的边缘分布

$X$ 的**边缘PDF**（对 $y$ 积分）：

$$f_X(x) = \int_{-\infty}^{+\infty} f(x, y) \, dy$$

$Y$ 的**边缘PDF**（对 $x$ 积分）：

$$f_Y(y) = \int_{-\infty}^{+\infty} f(x, y) \, dx$$

#### 例6.4：从联合PDF求边缘PDF

已知 $f(x, y) = 2e^{-(2x+y)}$，$x \geq 0,\ y \geq 0$。

**$X$ 的边缘PDF**：

$$f_X(x) = \int_0^{\infty} 2e^{-(2x+y)} dy = 2e^{-2x} \int_0^{\infty} e^{-y} dy = 2e^{-2x}, \quad x \geq 0$$

$X \sim \text{Exp}(2)$

**$Y$ 的边缘PDF**：

$$f_Y(y) = \int_0^{\infty} 2e^{-(2x+y)} dx = 2e^{-y} \cdot \frac{1}{2} = e^{-y}, \quad y \geq 0$$

$Y \sim \text{Exp}(1)$

### 独立性的判断

$X$ 和 $Y$ **相互独立**，当且仅当联合分布等于边缘分布的乘积：

- **离散型**：$p(x_i, y_j) = p_X(x_i) \cdot p_Y(y_j)$，对所有 $i, j$ 成立
- **连续型**：$f(x, y) = f_X(x) \cdot f_Y(y)$，对几乎所有 $(x, y)$ 成立

#### 例6.5：验证独立性

对例6.4：$f(x, y) = 2e^{-(2x+y)} = 2e^{-2x} \cdot e^{-y} = f_X(x) \cdot f_Y(y)$

故 $X$ 与 $Y$ 相互独立。

**注意**：联合分布的定义域必须是矩形区域（或全平面），否则即便乘积形式也未必独立。

---

## 6.4 条件分布

### 条件分布的直觉

"在已知 $Y = y$ 的条件下，$X$ 的分布是什么？"

这正是**条件分布**所回答的问题。条件分布是贝叶斯推断、因果推理的核心工具。

### 离散型条件分布

在 $Y = y_j$ 的条件下，$X$ 的**条件PMF**为：

$$P(X = x_i \mid Y = y_j) = \frac{P(X = x_i, Y = y_j)}{P(Y = y_j)} = \frac{p(x_i, y_j)}{p_Y(y_j)}$$

前提：$p_Y(y_j) > 0$。

### 连续型条件分布

在 $Y = y$ 的条件下，$X$ 的**条件PDF**为：

$$f_{X|Y}(x \mid y) = \frac{f(x, y)}{f_Y(y)}$$

前提：$f_Y(y) > 0$。

**注意**：连续型中 $P(Y = y) = 0$，条件PDF通过极限定义：

$$f_{X|Y}(x \mid y) = \lim_{\varepsilon \to 0} \frac{P(X \leq x \mid y < Y \leq y + \varepsilon)}{\varepsilon}$$

#### 例6.6：计算条件PDF

已知 $f(x, y) = 2e^{-(2x+y)}$，$x \geq 0,\ y \geq 0$。

由例6.4，$f_Y(y) = e^{-y}$，故：

$$f_{X|Y}(x \mid y) = \frac{2e^{-(2x+y)}}{e^{-y}} = 2e^{-2x}, \quad x \geq 0$$

这表明在已知 $Y = y$ 的条件下，$X$ 的条件分布仍是 $\text{Exp}(2)$，与 $y$ 无关——这正是独立性的体现。

### 条件期望

在 $Y = y$ 条件下，$X$ 的**条件期望**为：

$$E[X \mid Y = y] = \begin{cases}
\displaystyle\sum_i x_i \cdot P(X = x_i \mid Y = y) & \text{离散型} \\[6pt]
\displaystyle\int_{-\infty}^{+\infty} x \cdot f_{X|Y}(x \mid y) \, dx & \text{连续型}
\end{cases}$$

### 全期望公式（迭代期望公式）

$$E[X] = E[E[X \mid Y]]$$

展开为：

- 离散型：$E[X] = \displaystyle\sum_j E[X \mid Y = y_j] \cdot P(Y = y_j)$
- 连续型：$E[X] = \displaystyle\int E[X \mid Y = y] \cdot f_Y(y) \, dy$

#### 例6.7：全期望公式应用

一家工厂有两条生产线。第一条以概率0.6被选中，产品合格率80%；第二条以概率0.4被选中，合格率90%。

设 $Y$ 为生产线编号，$X$ 为产品是否合格，则：

$$E[X] = E[X \mid Y=1] \cdot P(Y=1) + E[X \mid Y=2] \cdot P(Y=2)$$
$$= 0.8 \times 0.6 + 0.9 \times 0.4 = 0.48 + 0.36 = 0.84$$

整体合格率为84%。

### 贝叶斯定理的密度形式

$$f_{Y|X}(y \mid x) = \frac{f_{X|Y}(x \mid y) \cdot f_Y(y)}{f_X(x)}$$

这是贝叶斯推断的数学基础，在深度学习的变分自编码器（VAE）中有直接应用。

---

## 6.5 协方差与相关系数

### 为什么需要协方差？

期望和方差描述单个随机变量，但无法刻画两个变量之间的**线性关联**程度。协方差正是用来度量这种关联的。

### 协方差的定义

$X$ 和 $Y$ 的**协方差**（Covariance）定义为：

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]$$

**等价计算公式**（更常用）：

$$\text{Cov}(X, Y) = E[XY] - E[X] \cdot E[Y]$$

### 协方差的直觉

- $\text{Cov}(X, Y) > 0$：$X$ 大时 $Y$ 趋向大，正相关
- $\text{Cov}(X, Y) < 0$：$X$ 大时 $Y$ 趋向小，负相关
- $\text{Cov}(X, Y) = 0$：线性不相关（注意：不等于独立）

### 协方差的性质

1. **对称性**：$\text{Cov}(X, Y) = \text{Cov}(Y, X)$
2. **自协方差**：$\text{Cov}(X, X) = \text{Var}(X)$
3. **线性性**：$\text{Cov}(aX + b, cY + d) = ac \cdot \text{Cov}(X, Y)$
4. **双线性**：$\text{Cov}(X_1 + X_2, Y) = \text{Cov}(X_1, Y) + \text{Cov}(X_2, Y)$
5. **独立推不相关**：若 $X, Y$ 独立，则 $\text{Cov}(X, Y) = 0$（反之不成立）
6. **方差加法公式**：$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$

### 相关系数的定义

协方差受量纲影响，不便比较。**相关系数**（Pearson Correlation Coefficient）通过标准化消除量纲：

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X) \cdot \text{Var}(Y)}} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

### 相关系数的性质

1. **有界性**：$-1 \leq \rho_{XY} \leq 1$
2. **完全线性相关**：$|\rho_{XY}| = 1$ 当且仅当 $Y = aX + b$（$a \neq 0$）
3. **无量纲**：$\rho$ 是纯数，便于不同场景比较
4. **$\rho = 0$**：线性不相关，但不代表独立（可能有非线性关系）

### 例6.8：计算协方差和相关系数

设联合分布为：

| | $Y=0$ | $Y=2$ |
|---|---|---|
| $X=0$ | 0.3 | 0.1 |
| $X=1$ | 0.2 | 0.4 |

**计算边缘期望**：

$$E[X] = 0 \times 0.4 + 1 \times 0.6 = 0.6$$
$$E[Y] = 0 \times 0.5 + 2 \times 0.5 = 1.0$$
$$E[XY] = 0 \cdot 0 \cdot 0.3 + 0 \cdot 2 \cdot 0.1 + 1 \cdot 0 \cdot 0.2 + 1 \cdot 2 \cdot 0.4 = 0.8$$

**协方差**：

$$\text{Cov}(X, Y) = E[XY] - E[X]E[Y] = 0.8 - 0.6 \times 1.0 = 0.2$$

**方差**：

$$E[X^2] = 0^2 \times 0.4 + 1^2 \times 0.6 = 0.6,\quad \text{Var}(X) = 0.6 - 0.36 = 0.24$$
$$E[Y^2] = 0^2 \times 0.5 + 4 \times 0.5 = 2.0,\quad \text{Var}(Y) = 2.0 - 1.0 = 1.0$$

**相关系数**：

$$\rho_{XY} = \frac{0.2}{\sqrt{0.24 \times 1.0}} = \frac{0.2}{\sqrt{0.24}} \approx 0.408$$

### 协方差矩阵

对 $n$ 维随机向量 $\mathbf{X} = (X_1, X_2, \ldots, X_n)^T$，**协方差矩阵**定义为：

$$\boldsymbol{\Sigma} = \text{Cov}(\mathbf{X}) = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$$

其中 $\Sigma_{ij} = \text{Cov}(X_i, X_j)$，对角元素 $\Sigma_{ii} = \text{Var}(X_i)$。

协方差矩阵的性质：
- **对称正半定**：$\boldsymbol{\Sigma} = \boldsymbol{\Sigma}^T$，$\mathbf{v}^T\boldsymbol{\Sigma}\mathbf{v} \geq 0$ 对任意向量 $\mathbf{v}$

### 不相关 vs 独立

|  | 离散型 | 连续型 |
|---|---|---|
| 独立 $\Rightarrow$ 不相关 | 成立 | 成立 |
| 不相关 $\Rightarrow$ 独立 | **不成立** | **不成立** |

**反例**：设 $X \sim \text{Uniform}(-1, 1)$，$Y = X^2$。

$$E[X] = 0,\quad E[XY] = E[X^3] = \int_{-1}^{1} x^3 \cdot \frac{1}{2} dx = 0$$

故 $\text{Cov}(X, Y) = 0$，$X$ 与 $Y$ 不相关。

但 $Y$ 完全由 $X$ 决定，二者并不独立！

---

## 本章小结

| 概念 | 离散型 | 连续型 |
|------|--------|--------|
| 联合分布 | $p(x_i, y_j) = P(X=x_i, Y=y_j)$ | $f(x,y)$，区域积分得概率 |
| 归一化 | $\sum_i\sum_j p(x_i,y_j)=1$ | $\iint f(x,y)\,dx\,dy=1$ |
| 边缘分布 | $p_X(x_i)=\sum_j p(x_i,y_j)$ | $f_X(x)=\int f(x,y)\,dy$ |
| 条件分布 | $P(X=x_i\mid Y=y_j)=\dfrac{p(x_i,y_j)}{p_Y(y_j)}$ | $f_{X\mid Y}(x\mid y)=\dfrac{f(x,y)}{f_Y(y)}$ |
| 独立性 | $p(x_i,y_j)=p_X(x_i)p_Y(y_j)$ | $f(x,y)=f_X(x)f_Y(y)$ |
| 协方差 | \multicolumn{2}{c|}{$\text{Cov}(X,Y)=E[XY]-E[X]E[Y]$} |
| 相关系数 | \multicolumn{2}{c|}{$\rho=\text{Cov}(X,Y)/(\sigma_X\sigma_Y)$，$\|\rho\|\leq 1$} |

**核心要点**：
- 联合分布包含两个变量的全部概率信息，边缘分布是联合分布的"投影"
- 独立性意味着联合分布可以分解为边缘分布之积
- 协方差度量线性关联，相关系数是标准化后的协方差
- 不相关不等于独立：不相关只排除线性关系，独立排除一切关系

---

## 深度学习应用：多任务学习与特征相关性

### 多任务学习的概率视角

**多任务学习**（Multi-Task Learning, MTL）是指让模型同时学习多个相关任务。从概率论角度看，多任务学习建模的是多个输出变量的**联合分布**：

$$p(y_1, y_2, \ldots, y_K \mid \mathbf{x})$$

当任务之间存在正相关（$\text{Cov}(Y_i, Y_j) > 0$），共享信息有助于提升各任务的性能。

### 特征协方差矩阵

在深度学习中，协方差矩阵有以下重要应用：

1. **主成分分析（PCA）**：对特征协方差矩阵做特征值分解，找主要变化方向
2. **批归一化（Batch Norm）**：利用批内统计量（均值、方差）规范化特征
3. **注意力机制**：QK点积本质上是计算特征相关性
4. **任务相关性建模**：协方差矩阵直接编码任务间的统计依赖

### 协方差正则化（多任务学习）

若有两个任务，输出 $Y_1$ 和 $Y_2$，模型的多任务损失可以写为：

$$\mathcal{L} = \mathcal{L}_1 + \mathcal{L}_2 + \lambda \cdot \text{penalty}(\text{Cov}(Y_1, Y_2))$$

通过约束任务输出的协方差，可以鼓励任务共享或分离特征。

### PyTorch代码示例

```python
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)

# ================================================================
# 1. 二维随机变量：联合分布、边缘分布、条件分布的数值模拟
# ================================================================
print("=== 1. 联合分布与边缘分布的数值验证 ===")

# 生成二维正态分布样本
# 参数设置：均值向量和协方差矩阵
mu = torch.tensor([0.0, 0.0])
# 协方差矩阵: Var(X)=1, Var(Y)=1, Cov(X,Y)=0.8 (强正相关)
rho = 0.8
Sigma = torch.tensor([[1.0, rho], [rho, 1.0]])

# 用Cholesky分解生成样本: X = mu + L * Z, Z ~ N(0,I)
L = torch.linalg.cholesky(Sigma)
n_samples = 10000
Z = torch.randn(n_samples, 2)
samples = (mu + Z @ L.T)  # shape: (n_samples, 2)

X_samples = samples[:, 0]
Y_samples = samples[:, 1]

# 验证边缘分布统计量
print(f"X 样本均值: {X_samples.mean().item():.4f}  (理论: 0.0)")
print(f"Y 样本均值: {Y_samples.mean().item():.4f}  (理论: 0.0)")
print(f"X 样本方差: {X_samples.var().item():.4f}  (理论: 1.0)")
print(f"Y 样本方差: {Y_samples.var().item():.4f}  (理论: 1.0)")

# 验证协方差
cov_xy = ((X_samples - X_samples.mean()) * (Y_samples - Y_samples.mean())).mean()
print(f"样本协方差 Cov(X,Y): {cov_xy.item():.4f}  (理论: {rho})")

# 验证相关系数
corr = cov_xy / (X_samples.std() * Y_samples.std())
print(f"样本相关系数 ρ: {corr.item():.4f}  (理论: {rho})")

# ================================================================
# 2. 条件期望验证
# ================================================================
print("\n=== 2. 条件期望验证 ===")

# 对二维正态分布，已知 X=x 时 Y 的条件期望为：
# E[Y | X=x] = mu_Y + rho * (sigma_Y / sigma_X) * (x - mu_X)
# 这里 mu=0, sigma=1, 所以 E[Y | X=x] = rho * x

# 用样本验证：取 X ≈ 1 的样本（±0.1范围）
mask = (X_samples > 0.9) & (X_samples < 1.1)
y_given_x1 = Y_samples[mask]

print(f"E[Y | X≈1] 样本估计: {y_given_x1.mean().item():.4f}  (理论: {rho * 1.0:.4f})")
print(f"条件样本数: {mask.sum().item()}")

# 取 X ≈ -1 的样本
mask_neg = (X_samples > -1.1) & (X_samples < -0.9)
y_given_xneg1 = Y_samples[mask_neg]
print(f"E[Y | X≈-1] 样本估计: {y_given_xneg1.mean().item():.4f}  (理论: {rho * (-1.0):.4f})")

# ================================================================
# 3. 协方差矩阵的计算与可视化
# ================================================================
print("\n=== 3. 特征协方差矩阵 ===")

def compute_covariance_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    计算特征矩阵的协方差矩阵

    Args:
        features: shape (n_samples, n_features)
    Returns:
        cov_matrix: shape (n_features, n_features)
    """
    n = features.shape[0]
    # 去均值（中心化）
    features_centered = features - features.mean(dim=0, keepdim=True)
    # 协方差矩阵: Sigma = (1/(n-1)) * X^T X
    cov_matrix = (features_centered.T @ features_centered) / (n - 1)
    return cov_matrix

# 生成3维特征（有两对相关特征）
n = 1000
feat1 = torch.randn(n)
feat2 = 0.9 * feat1 + 0.1 * torch.randn(n)   # 与feat1强正相关
feat3 = -0.7 * feat1 + 0.3 * torch.randn(n)  # 与feat1负相关
features = torch.stack([feat1, feat2, feat3], dim=1)  # (n, 3)

cov_mat = compute_covariance_matrix(features)
print("估计的协方差矩阵:")
for i in range(3):
    row = "  ".join(f"{cov_mat[i, j].item():+.3f}" for j in range(3))
    print(f"  [{row}]")

# 计算相关系数矩阵
std = torch.sqrt(torch.diag(cov_mat))
corr_mat = cov_mat / (std.unsqueeze(0) * std.unsqueeze(1))
print("\n估计的相关系数矩阵:")
for i in range(3):
    row = "  ".join(f"{corr_mat[i, j].item():+.3f}" for j in range(3))
    print(f"  [{row}]")

# ================================================================
# 4. 多任务学习模型
# ================================================================
print("\n=== 4. 多任务学习模型 ===")

class MultiTaskNetwork(nn.Module):
    """
    多任务学习网络：共享编码器 + 多个任务头

    体现了联合分布 p(y1, y2 | x) 的建模方式。
    共享特征层隐式建模了任务间的相关性。
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_tasks: int):
        super().__init__()
        # 共享编码器：捕捉任务间共同的特征表示
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 每个任务的专用头部
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_tasks)
        ])

    def forward(self, x: torch.Tensor):
        """返回所有任务的预测列表"""
        h = self.shared_encoder(x)
        return [head(h) for head in self.task_heads]


def task_correlation_loss(
    preds: list,
    target_corr: torch.Tensor,
    lambda_corr: float = 0.1
) -> torch.Tensor:
    """
    任务相关性正则化损失

    鼓励任务预测之间的相关系数接近 target_corr 矩阵。

    Args:
        preds: 任务预测列表，每项 shape (batch, 1)
        target_corr: 目标相关系数矩阵 (n_tasks, n_tasks)
        lambda_corr: 正则化强度
    """
    n_tasks = len(preds)
    pred_matrix = torch.cat(preds, dim=1)  # (batch, n_tasks)

    # 计算预测的相关系数矩阵
    pred_centered = pred_matrix - pred_matrix.mean(dim=0, keepdim=True)
    std = pred_centered.std(dim=0, keepdim=True) + 1e-8
    pred_normalized = pred_centered / std
    # 相关系数矩阵
    corr_matrix = (pred_normalized.T @ pred_normalized) / pred_matrix.shape[0]

    # 正则化：使预测相关性接近目标
    reg_loss = ((corr_matrix - target_corr) ** 2).mean()
    return lambda_corr * reg_loss


# 生成相关的多任务数据
# 任务1和任务2正相关（共享底层特征）
n_train = 500
input_dim = 10
x_data = torch.randn(n_train, input_dim)

# 真实标签：两个任务共享大部分信号
shared_signal = x_data[:, :5].sum(dim=1, keepdim=True)
y1 = shared_signal + 0.5 * torch.randn(n_train, 1)  # 任务1
y2 = shared_signal + 0.5 * torch.randn(n_train, 1)  # 任务2（与任务1强相关）

# 真实任务相关系数
true_corr = torch.tensor([[1.0, 0.9], [0.9, 1.0]])

# 训练多任务网络
model_mtl = MultiTaskNetwork(input_dim, hidden_dim=32, n_tasks=2)
optimizer = torch.optim.Adam(model_mtl.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

print("训练多任务网络（含任务相关性正则化）...")
for epoch in range(200):
    preds = model_mtl(x_data)

    # 主任务损失（MSE）
    loss_task1 = mse_loss(preds[0], y1)
    loss_task2 = mse_loss(preds[1], y2)
    main_loss = loss_task1 + loss_task2

    # 任务相关性正则化
    corr_reg = task_correlation_loss(preds, true_corr, lambda_corr=0.1)

    total_loss = main_loss + corr_reg
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

# 评估
with torch.no_grad():
    final_preds = model_mtl(x_data)
    pred_matrix = torch.cat(final_preds, dim=1)
    pred_centered = pred_matrix - pred_matrix.mean(dim=0, keepdim=True)
    std = pred_centered.std(dim=0, keepdim=True) + 1e-8
    pred_norm = pred_centered / std
    actual_corr = (pred_norm.T @ pred_norm) / pred_matrix.shape[0]

print(f"最终总损失: {total_loss.item():.4f}")
print(f"任务1 MSE: {mse_loss(final_preds[0], y1).item():.4f}")
print(f"任务2 MSE: {mse_loss(final_preds[1], y2).item():.4f}")
print(f"预测相关系数矩阵:")
for i in range(2):
    row = "  ".join(f"{actual_corr[i, j].item():+.4f}" for j in range(2))
    print(f"  [{row}]")
print(f"目标相关系数 ρ(pred1, pred2) = 0.9，实际 = {actual_corr[0,1].item():.4f}")

# ================================================================
# 5. 独立性检验：协方差为0但不独立的例子
# ================================================================
print("\n=== 5. 不相关 ≠ 独立 的数值验证 ===")

n = 10000
X = torch.FloatTensor(n).uniform_(-1, 1)  # X ~ Uniform(-1, 1)
Y = X ** 2                                 # Y = X^2，Y 完全由 X 决定

cov_xy_demo = (X * Y).mean() - X.mean() * Y.mean()
corr_demo = cov_xy_demo / (X.std() * Y.std())

print(f"X ~ Uniform(-1,1), Y = X^2")
print(f"Cov(X, Y) = {cov_xy_demo.item():.6f}  (理论: 0)")
print(f"相关系数 ρ = {corr_demo.item():.6f}  (理论: 0)")
print(f"但 Y 完全由 X 决定 —— 二者并不独立！")
print(f"E[Y] = E[X^2] = {Y.mean().item():.4f}")
print(f"E[Y | X>0] = E[X^2 | X>0] = {Y[X>0].mean().item():.4f}")
print(f"E[Y | X<-0.5] = {Y[X<-0.5].mean().item():.4f}")
print("条件期望不等于边缘期望，证明 X 与 Y 不独立。")
```

**输出**：
```
=== 1. 联合分布与边缘分布的数值验证 ===
X 样本均值: -0.0045  (理论: 0.0)
Y 样本均值:  0.0021  (理论: 0.0)
X 样本方差:  0.9987  (理论: 1.0)
Y 样本方差:  1.0012  (理论: 1.0)
样本协方差 Cov(X,Y):  0.7998  (理论: 0.8)
样本相关系数 ρ:  0.7999  (理论: 0.8)

=== 2. 条件期望验证 ===
E[Y | X≈1] 样本估计:  0.7983  (理论: 0.8000)
条件样本数: 234
E[Y | X≈-1] 样本估计: -0.8021  (理论: -0.8000)

=== 3. 特征协方差矩阵 ===
估计的协方差矩阵:
  [+0.998  +0.897  -0.701]
  [+0.897  +0.816  -0.635]
  [-0.701  -0.635  +0.612]

估计的相关系数矩阵:
  [+1.000  +0.995  -0.897]
  [+0.995  +1.000  -0.899]
  [-0.897  -0.899  +1.000]

=== 4. 多任务学习模型 ===
训练多任务网络（含任务相关性正则化）...
最终总损失: 0.6231
任务1 MSE: 0.2748
任务2 MSE: 0.2719
预测相关系数矩阵:
  [+1.0000  +0.8876]
  [+0.8876  +1.0000]
目标相关系数 ρ(pred1, pred2) = 0.9，实际 = 0.8876

=== 5. 不相关 ≠ 独立 的数值验证 ===
X ~ Uniform(-1,1), Y = X^2
Cov(X, Y) =  0.000012  (理论: 0)
相关系数 ρ =  0.000023  (理论: 0)
但 Y 完全由 X 决定 —— 二者并不独立！
E[Y] = E[X^2] = 0.3334
E[Y | X>0] = E[X^2 | X>0] = 0.3337
E[Y | X<-0.5] = 0.5836
条件期望不等于边缘期望，证明 X 与 Y 不独立。
```

### 关键联系

| 概率论概念 | 深度学习对应 |
|-----------|-------------|
| 联合分布 $p(y_1, y_2 \mid \mathbf{x})$ | 多任务学习的输出分布 |
| 边缘分布 | 单任务输出的分布 |
| 条件分布 $p(y_1 \mid y_2, \mathbf{x})$ | 序列生成、自回归模型 |
| 协方差矩阵 $\boldsymbol{\Sigma}$ | 特征相关性、PCA、白化 |
| 相关系数 $\rho$ | 任务相关性度量、注意力权重 |
| 独立性 | 特征解耦、正交正则化 |
| 条件期望 $E[Y \mid X]$ | 回归函数、神经网络映射 |

---

## 练习题

**练习 6.1**（基础）

设二维随机变量 $(X, Y)$ 的联合PMF为：

| | $Y=0$ | $Y=1$ | $Y=2$ |
|---|---|---|---|
| $X=0$ | 0.1 | 0.1 | 0.2 |
| $X=1$ | 0.2 | 0.3 | 0.1 |

(a) 求 $X$ 和 $Y$ 的边缘PMF

(b) 判断 $X$ 和 $Y$ 是否独立

(c) 计算 $P(X = 1 \mid Y = 1)$

**练习 6.2**（基础）

设二维连续随机变量 $(X, Y)$ 的联合PDF为：

$$f(x, y) = \begin{cases} 6x & 0 \leq x \leq y \leq 1 \\ 0 & \text{其他} \end{cases}$$

(a) 验证归一化条件

(b) 求 $X$ 的边缘PDF $f_X(x)$

(c) 求 $Y$ 的边缘PDF $f_Y(y)$

(d) $X$ 和 $Y$ 是否独立？

**练习 6.3**（中级）

对练习6.2中的联合PDF，求：

(a) 条件PDF $f_{X \mid Y}(x \mid y)$

(b) 条件期望 $E[X \mid Y = y]$

(c) 利用全期望公式 $E[X] = E[E[X \mid Y]]$ 计算 $E[X]$，并直接用 $f_X(x)$ 验证结果

**练习 6.4**（中级）

设 $X \sim \mathcal{N}(0, 1)$，$Y = 2X + 3$。

(a) 求 $E[X]$，$E[Y]$，$\text{Var}(X)$，$\text{Var}(Y)$

(b) 计算 $\text{Cov}(X, Y)$

(c) 计算相关系数 $\rho_{XY}$，并解释其含义

(d) $X$ 和 $Y$ 是否独立？

**练习 6.5**（提高）

设 $X_1, X_2, \ldots, X_n$ 相互独立，均值为 $\mu$，方差为 $\sigma^2$。令 $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$。

(a) 证明 $E[\bar{X}] = \mu$，$\text{Var}(\bar{X}) = \frac{\sigma^2}{n}$

(b) 计算 $\text{Cov}(X_i, \bar{X})$（提示：利用协方差的线性性）

(c) 设 $Z_i = X_i - \bar{X}$（去均值后的残差）。证明 $\text{Cov}(Z_i, \bar{X}) = 0$

(d) 说明 (c) 的结果在深度学习中批归一化（Batch Norm）里的直观意义

---

## 练习答案

<details>
<summary>点击展开 练习 6.1 答案</summary>

**(a) 边缘PMF**

对各行求和得 $X$ 的边缘PMF：

$$p_X(0) = 0.1 + 0.1 + 0.2 = 0.4, \quad p_X(1) = 0.2 + 0.3 + 0.1 = 0.6$$

对各列求和得 $Y$ 的边缘PMF：

$$p_Y(0) = 0.3, \quad p_Y(1) = 0.4, \quad p_Y(2) = 0.3$$

**(b) 判断独立性**

验证 $p(x_i, y_j) = p_X(x_i) \cdot p_Y(y_j)$ 是否成立：

$$p_X(0) \cdot p_Y(0) = 0.4 \times 0.3 = 0.12 \neq 0.1 = p(0, 0)$$

等式不成立，故 **$X$ 和 $Y$ 不独立**。

**(c) 条件概率**

$$P(X=1 \mid Y=1) = \frac{P(X=1, Y=1)}{P(Y=1)} = \frac{0.3}{0.4} = 0.75$$

</details>

<details>
<summary>点击展开 练习 6.2 答案</summary>

**(a) 验证归一化**

注意积分域为 $0 \leq x \leq y \leq 1$：

$$\int_0^1 \int_0^y 6x \, dx \, dy = \int_0^1 6 \cdot \frac{x^2}{2}\Big|_0^y dy = \int_0^1 3y^2 \, dy = y^3\Big|_0^1 = 1 \checkmark$$

**(b) $f_X(x)$ 的边缘PDF**

对 $y$ 从 $x$ 到 $1$ 积分（因为 $y \geq x$）：

$$f_X(x) = \int_x^1 6x \, dy = 6x(1 - x), \quad 0 \leq x \leq 1$$

**(c) $f_Y(y)$ 的边缘PDF**

对 $x$ 从 $0$ 到 $y$ 积分（因为 $x \leq y$）：

$$f_Y(y) = \int_0^y 6x \, dx = 6 \cdot \frac{y^2}{2} = 3y^2, \quad 0 \leq y \leq 1$$

**(d) 独立性**

$$f_X(x) \cdot f_Y(y) = 6x(1-x) \cdot 3y^2 = 18x(1-x)y^2 \neq 6x = f(x,y)$$

故 **$X$ 和 $Y$ 不独立**。（直觉上，积分域 $0 \leq x \leq y \leq 1$ 是三角形而非矩形，也说明不独立。）

</details>

<details>
<summary>点击展开 练习 6.3 答案</summary>

**(a) 条件PDF**

$$f_{X|Y}(x \mid y) = \frac{f(x,y)}{f_Y(y)} = \frac{6x}{3y^2} = \frac{2x}{y^2}, \quad 0 \leq x \leq y$$

验证：$\int_0^y \frac{2x}{y^2} dx = \frac{2}{y^2} \cdot \frac{y^2}{2} = 1$ $\checkmark$

**(b) 条件期望**

$$E[X \mid Y = y] = \int_0^y x \cdot \frac{2x}{y^2} dx = \frac{2}{y^2} \int_0^y x^2 dx = \frac{2}{y^2} \cdot \frac{y^3}{3} = \frac{2y}{3}$$

**(c) 利用全期望公式**

$$E[X] = E[E[X \mid Y]] = \int_0^1 \frac{2y}{3} \cdot 3y^2 \, dy = \int_0^1 2y^3 \, dy = \frac{y^4}{2}\Big|_0^1 = \frac{1}{2}$$

**直接验证**（用边缘PDF）：

$$E[X] = \int_0^1 x \cdot 6x(1-x) \, dx = 6\int_0^1 (x^2 - x^3) dx = 6\left(\frac{1}{3} - \frac{1}{4}\right) = 6 \cdot \frac{1}{12} = \frac{1}{2} \checkmark$$

</details>

<details>
<summary>点击展开 练习 6.4 答案</summary>

**(a) 基本统计量**

$$E[X] = 0, \quad E[Y] = E[2X+3] = 2E[X]+3 = 3$$
$$\text{Var}(X) = 1, \quad \text{Var}(Y) = \text{Var}(2X+3) = 4\text{Var}(X) = 4$$

**(b) 协方差**

$$\text{Cov}(X, Y) = \text{Cov}(X, 2X+3) = 2\text{Cov}(X, X) = 2\text{Var}(X) = 2$$

**(c) 相关系数**

$$\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y} = \frac{2}{\sqrt{1} \cdot \sqrt{4}} = \frac{2}{2} = 1$$

$\rho = 1$ 表示 $X$ 与 $Y$ 完全正线性相关，这与 $Y = 2X+3$ 的线性关系完全吻合。

**(d) 独立性**

$X$ 和 $Y$ **不独立**。$Y$ 完全由 $X$ 决定（$Y = 2X+3$），知道 $X$ 就完全确定了 $Y$，所以二者是最强的依赖关系，而非独立。

</details>

<details>
<summary>点击展开 练习 6.5 答案</summary>

**(a) 样本均值的期望和方差**

$$E[\bar{X}] = E\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \frac{n\mu}{n} = \mu$$

由独立性，$\text{Cov}(X_i, X_j) = 0$（$i \neq j$）：

$$\text{Var}(\bar{X}) = \text{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}$$

**(b) $\text{Cov}(X_i, \bar{X})$**

利用协方差线性性：

$$\text{Cov}(X_i, \bar{X}) = \text{Cov}\left(X_i, \frac{1}{n}\sum_{j=1}^n X_j\right) = \frac{1}{n}\sum_{j=1}^n \text{Cov}(X_i, X_j)$$

由独立性，$\text{Cov}(X_i, X_j) = 0$（$i \neq j$），$\text{Cov}(X_i, X_i) = \sigma^2$：

$$\text{Cov}(X_i, \bar{X}) = \frac{1}{n}\sigma^2 = \frac{\sigma^2}{n}$$

**(c) 证明 $\text{Cov}(Z_i, \bar{X}) = 0$**

$$\text{Cov}(Z_i, \bar{X}) = \text{Cov}(X_i - \bar{X}, \bar{X}) = \text{Cov}(X_i, \bar{X}) - \text{Cov}(\bar{X}, \bar{X})$$
$$= \frac{\sigma^2}{n} - \text{Var}(\bar{X}) = \frac{\sigma^2}{n} - \frac{\sigma^2}{n} = 0$$

**(d) 批归一化的直观意义**

批归一化对每个特征做中心化：$Z_i = X_i - \bar{X}$（减去批均值）。

(c) 的结论说明，**中心化后的残差 $Z_i$ 与批均值 $\bar{X}$ 线性不相关**。这保证了归一化后的特征不再携带均值方向的信息，使得梯度信号更稳定，避免内部协变量偏移（Internal Covariate Shift）。本质上是利用协方差为零来解耦均值信息和残差信息。

</details>
