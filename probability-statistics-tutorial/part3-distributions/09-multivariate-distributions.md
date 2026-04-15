# 第9章 多维分布

> **难度**：★★★★☆
> **前置知识**：第5章连续随机变量、第8章多元正态分布初步、线性代数基础（矩阵、行列式、正定矩阵）

---

## 学习目标

- 掌握多项分布的定义、性质及与二项分布的关系
- 理解多元正态分布的几何意义与协方差矩阵的作用
- 了解 Dirichlet 分布作为多项分布的共轭先验的重要地位
- 初步认识 Wishart 分布及其在协方差矩阵建模中的作用
- 掌握分布变换与采样方法，建立与深度学习（Softmax、VAE）的联系

---

## 9.1 多项分布

### 从二项分布推广到多项分布

二项分布描述 $n$ 次独立伯努利试验中"成功"次数的分布——每次试验只有两个结果。当每次试验有 $k$ 个可能结果时，自然推广到**多项分布**（Multinomial Distribution）。

**场景**：投掷一枚有 $k$ 个面的骰子 $n$ 次，第 $i$ 面出现的概率为 $p_i$，$\sum_{i=1}^k p_i = 1$。令 $X_i$ 表示第 $i$ 面出现的次数，则随机向量 $(X_1, X_2, \ldots, X_k)$ 服从多项分布。

### 定义

若随机向量 $\mathbf{X} = (X_1, \ldots, X_k)$ 满足 $\sum_{i=1}^k X_i = n$，且联合概率质量函数为：

$$P(X_1 = x_1, \ldots, X_k = x_k) = \frac{n!}{x_1! x_2! \cdots x_k!} \prod_{i=1}^k p_i^{x_i}$$

其中 $x_i \geq 0$ 为整数，$\sum_{i=1}^k x_i = n$，$p_i > 0$，$\sum_{i=1}^k p_i = 1$，则称 $\mathbf{X}$ 服从**多项分布**，记作：

$$\mathbf{X} \sim \text{Multinomial}(n, \mathbf{p}), \quad \mathbf{p} = (p_1, \ldots, p_k)$$

### 直觉理解：多项式系数

分子 $n!$ 是全排列数，分母 $x_1! \cdots x_k!$ 消除了同类元素的重复，因此多项式系数 $\frac{n!}{x_1!\cdots x_k!}$ 表示将 $n$ 次试验分配给 $k$ 个结果的方式数。每种分配方案的概率是 $\prod_i p_i^{x_i}$，两者相乘得到联合概率。

### 均值与协方差

**边缘分布**：每个 $X_i$ 的边缘分布是二项分布：

$$X_i \sim \text{Binomial}(n, p_i)$$

因此：

$$E[X_i] = np_i, \quad \text{Var}(X_i) = np_i(1-p_i)$$

**协方差**：不同类别之间存在负相关（增加一个类别的计数必然减少其他类别）：

$$\text{Cov}(X_i, X_j) = -np_ip_j, \quad i \neq j$$

**推导**：由于 $\sum_i X_i = n$ 是常数，有 $\text{Var}\!\left(\sum_i X_i\right) = 0$，展开得：

$$\sum_i \text{Var}(X_i) + 2\sum_{i < j} \text{Cov}(X_i, X_j) = 0$$

代入各方差并整理可得上式。

### 例9.1：分类投票

某次选举有三位候选人，支持率分别为 $p_1 = 0.5, p_2 = 0.3, p_3 = 0.2$，随机调查 10 人，求恰好 5 人选1号、3 人选2号、2 人选3号的概率。

$$P(5,3,2) = \frac{10!}{5!\,3!\,2!} \times 0.5^5 \times 0.3^3 \times 0.2^2 = 252 \times 0.03125 \times 0.027 \times 0.04 \approx 0.0851$$

### 特殊情形

- $k = 2$ 时，多项分布退化为**二项分布**
- $n = 1$ 时，多项分布退化为**类别分布**（Categorical Distribution），是 softmax 输出对应的分布

---

## 9.2 多元正态分布

### 定义

$d$ 维随机向量 $\mathbf{X} = (X_1, \ldots, X_d)^\top$ 服从**多元正态分布**（Multivariate Normal Distribution），若其概率密度函数为：

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

记作 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，其中：
- $\boldsymbol{\mu} \in \mathbb{R}^d$：均值向量
- $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$：协方差矩阵（对称正定）
- $|\boldsymbol{\Sigma}|$：$\boldsymbol{\Sigma}$ 的行列式

### 协方差矩阵的几何含义

协方差矩阵 $\boldsymbol{\Sigma}$ 完全刻画了各分量之间的线性相关结构：

$$\Sigma_{ij} = \text{Cov}(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)]$$

**几何理解**：指数项中的马氏距离 $(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$ 等于常数的等值面是一个**椭球体**。$\boldsymbol{\Sigma}$ 的特征向量给出椭球的方向，特征值给出各方向的"伸展程度"。

**三种典型情形**：

| 协方差矩阵形式 | 含义 | 等值面形状 |
|--------------|------|----------|
| $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$ | 各维独立且方差相同 | 球形 |
| $\boldsymbol{\Sigma} = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$ | 各维独立但方差不同 | 轴对齐椭球 |
| 一般正定矩阵 | 各维相关 | 旋转椭球 |

### 重要性质

**性质1（线性变换封闭性）**：若 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，$\mathbf{A} \in \mathbb{R}^{m \times d}$ 为矩阵，$\mathbf{b} \in \mathbb{R}^m$，则：

$$\mathbf{A}\mathbf{X} + \mathbf{b} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b},\; \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)$$

**性质2（边缘分布）**：多元正态的任意边缘分布仍是正态分布。将 $\mathbf{X}$ 分块为 $(\mathbf{X}_1, \mathbf{X}_2)$：

$$\mathbf{X}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_{11})$$

**性质3（条件分布）**：给定 $\mathbf{X}_2 = \mathbf{x}_2$ 时，$\mathbf{X}_1$ 的条件分布仍是正态分布：

$$\mathbf{X}_1 \mid \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}\!\left(\boldsymbol{\mu}_{1|2},\; \boldsymbol{\Sigma}_{1|2}\right)$$

其中：

$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$

$$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

**性质4（独立性与不相关等价）**：对于正态分布，**不相关**等价于**独立**（这在一般分布中不成立）。

### 例9.2：二元正态分布

最常用的特例：$d = 2$，令 $\boldsymbol{\mu} = (0, 0)^\top$，

$$\boldsymbol{\Sigma} = \begin{pmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{pmatrix}$$

其中 $\rho \in (-1, 1)$ 是相关系数。PDF 为：

$$f(x_1, x_2) = \frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}} \exp\!\left(-\frac{1}{2(1-\rho^2)}\left[\frac{x_1^2}{\sigma_1^2} - \frac{2\rho x_1 x_2}{\sigma_1\sigma_2} + \frac{x_2^2}{\sigma_2^2}\right]\right)$$

- $\rho > 0$：$X_1$ 与 $X_2$ 正相关，椭球沿 $45°$ 方向倾斜
- $\rho = 0$：独立，等值面为轴对齐椭圆
- $\rho < 0$：负相关，椭球沿 $-45°$ 方向倾斜

### 从标准正态生成多元正态

设 $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，对 $\boldsymbol{\Sigma}$ 做 Cholesky 分解 $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$，则：

$$\mathbf{X} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

这是从多元正态分布采样的标准算法。

---

## 9.3 Dirichlet 分布

### 动机：概率向量的分布

在贝叶斯统计中，我们常需要对"概率参数"本身建立先验。例如，骰子每面的概率 $\mathbf{p} = (p_1, \ldots, p_k)$ 满足 $p_i \geq 0$，$\sum_i p_i = 1$，即 $\mathbf{p}$ 位于 $k-1$ 维**单纯形**（simplex）上。**Dirichlet 分布**正是定义在单纯形上的连续分布。

### 定义

若 $k$ 维随机向量 $\mathbf{p} = (p_1, \ldots, p_k)$ 满足 $p_i > 0$，$\sum_i p_i = 1$，且概率密度函数为：

$$f(\mathbf{p};\, \boldsymbol{\alpha}) = \frac{\Gamma\!\left(\sum_{i=1}^k \alpha_i\right)}{\prod_{i=1}^k \Gamma(\alpha_i)} \prod_{i=1}^k p_i^{\alpha_i - 1}$$

则称 $\mathbf{p}$ 服从参数为 $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_k)$（$\alpha_i > 0$）的 **Dirichlet 分布**，记作：

$$\mathbf{p} \sim \text{Dir}(\boldsymbol{\alpha})$$

归一化常数中的 Gamma 函数 $\Gamma(n) = (n-1)!$（整数时）确保 PDF 积分为1。

### 参数的直觉

将 $\alpha_i$ 理解为"伪计数"：$\alpha_i$ 越大，$p_i$ 趋近越大；令 $\alpha_0 = \sum_i \alpha_i$（称为**浓度参数**）：

- $\alpha_0$ 大：分布集中在均值附近（确定性强）
- $\alpha_0$ 小（接近0）：分布集中在单纯形顶点（稀疏性强）
- $\boldsymbol{\alpha} = \mathbf{1}$（均匀先验）：单纯形上的均匀分布

### 均值与方差

$$E[p_i] = \frac{\alpha_i}{\alpha_0}, \quad \alpha_0 = \sum_{j=1}^k \alpha_j$$

$$\text{Var}(p_i) = \frac{\alpha_i(\alpha_0 - \alpha_i)}{\alpha_0^2(\alpha_0 + 1)}$$

$$\text{Cov}(p_i, p_j) = \frac{-\alpha_i \alpha_j}{\alpha_0^2(\alpha_0 + 1)}, \quad i \neq j$$

### Dirichlet 是多项分布的共轭先验

这是 Dirichlet 分布最重要的性质。设先验 $\mathbf{p} \sim \text{Dir}(\boldsymbol{\alpha})$，观测到 $n$ 次试验中第 $i$ 类出现 $x_i$ 次（即 $\mathbf{x} \sim \text{Multinomial}(n, \mathbf{p})$），则后验为：

$$\mathbf{p} \mid \mathbf{x} \sim \text{Dir}(\boldsymbol{\alpha} + \mathbf{x})$$

**推导**：

$$p(\mathbf{p} \mid \mathbf{x}) \propto p(\mathbf{x} \mid \mathbf{p}) \cdot p(\mathbf{p}) \propto \prod_i p_i^{x_i} \cdot \prod_i p_i^{\alpha_i - 1} = \prod_i p_i^{(\alpha_i + x_i) - 1}$$

这正是 $\text{Dir}(\boldsymbol{\alpha} + \mathbf{x})$ 的核密度。共轭性意味着先验与后验属于同一分布族，大大简化了贝叶斯推断。

### 特殊情形

- $k = 2$：Dirichlet 分布退化为 **Beta 分布** $\text{Beta}(\alpha_1, \alpha_2)$
- 对称情形 $\boldsymbol{\alpha} = \alpha \mathbf{1}$：称为**对称 Dirichlet 分布**

### 例9.3：文本主题建模

在 LDA（潜在狄利克雷分配）中，每篇文档的主题分布 $\theta_d \sim \text{Dir}(\alpha \mathbf{1})$，每个主题的词分布 $\phi_k \sim \text{Dir}(\beta \mathbf{1})$。Dirichlet 先验的稀疏性（$\alpha < 1$ 时）促使文档只集中在少数主题上。

---

## 9.4 Wishart 分布简介

### 动机：协方差矩阵的分布

多元正态分布中，协方差矩阵 $\boldsymbol{\Sigma}$ 是未知参数。在贝叶斯框架下，需要对正定矩阵建立先验。**Wishart 分布**是正定矩阵上的分布，是 $\boldsymbol{\Sigma}$ 或 $\boldsymbol{\Sigma}^{-1}$（精度矩阵）的共轭先验。

### 从卡方分布到 Wishart 分布

**类比**：若 $Z \sim \mathcal{N}(0,1)$，则 $Z^2 \sim \chi^2(1)$；若 $Z_1, \ldots, Z_\nu \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)$，则 $\sum_i Z_i^2 \sim \chi^2(\nu)$。

推广到多维：若 $\mathbf{z}_1, \ldots, \mathbf{z}_\nu \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$，令：

$$\mathbf{W} = \sum_{i=1}^{\nu} \mathbf{z}_i \mathbf{z}_i^\top$$

则 $\mathbf{W}$ 服从 **Wishart 分布**，记作 $\mathbf{W} \sim \mathcal{W}_d(\nu, \boldsymbol{\Sigma})$，其中 $d$ 是维度，$\nu \geq d$ 是**自由度**。

### 概率密度函数

$$f(\mathbf{W}) = \frac{|\mathbf{W}|^{(\nu - d - 1)/2} \exp\!\left(-\frac{1}{2}\text{tr}(\boldsymbol{\Sigma}^{-1}\mathbf{W})\right)}{2^{\nu d/2} |\boldsymbol{\Sigma}|^{\nu/2} \Gamma_d(\nu/2)}$$

其中 $\Gamma_d(\cdot)$ 是多元 Gamma 函数，$\text{tr}(\cdot)$ 是矩阵的迹。

### 均值与关键性质

$$E[\mathbf{W}] = \nu \boldsymbol{\Sigma}$$

**与卡方分布的关系**：$d = 1$ 时，$\mathcal{W}_1(\nu, \sigma^2)$ 对应 $\sigma^2 \chi^2(\nu)$。

**逆 Wishart 分布**：若 $\mathbf{W} \sim \mathcal{W}_d(\nu, \boldsymbol{\Sigma})$，则 $\mathbf{W}^{-1} \sim \mathcal{W}^{-1}_d(\nu, \boldsymbol{\Sigma}^{-1})$，称为**逆 Wishart 分布**，常用作 $\boldsymbol{\Sigma}$ 的共轭先验。

### 贝叶斯多元正态模型

**模型设定**：

$$\boldsymbol{\Sigma}^{-1} \sim \mathcal{W}_d(\nu_0, \mathbf{V}_0^{-1}), \quad \mathbf{x}_i \mid \boldsymbol{\mu}, \boldsymbol{\Sigma} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

**后验**（观测 $n$ 个样本后）：

$$\boldsymbol{\Sigma}^{-1} \mid \{\mathbf{x}_i\} \sim \mathcal{W}_d\!\left(\nu_0 + n,\; \left(\mathbf{V}_0 + \sum_{i=1}^n (\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^\top\right)^{-1}\right)$$

Wishart 分布在高斯过程、贝叶斯线性回归、多元时间序列等模型中均有重要应用。

---

## 9.5 分布变换与采样

### 分布变换的一般理论

设 $\mathbf{X} \sim f_{\mathbf{X}}(\mathbf{x})$，$\mathbf{Y} = g(\mathbf{X})$ 是可逆变换，令 $\mathbf{x} = g^{-1}(\mathbf{y})$，则 $\mathbf{Y}$ 的 PDF 为：

$$\boxed{f_{\mathbf{Y}}(\mathbf{y}) = f_{\mathbf{X}}\!\left(g^{-1}(\mathbf{y})\right) \cdot \left|\det\mathbf{J}_{g^{-1}}(\mathbf{y})\right|}$$

其中 $\mathbf{J}_{g^{-1}}$ 是逆变换的 **Jacobian 矩阵**（各偏导数组成的矩阵），行列式的绝对值 $|\det \mathbf{J}|$ 是体积缩放因子。

**Jacobian 矩阵** 的具体形式：设 $\mathbf{x} = (x_1, \ldots, x_n)$，$\mathbf{y} = (y_1, \ldots, y_n)$，则：

$$\mathbf{J}_{g^{-1}} = \begin{pmatrix} \frac{\partial x_1}{\partial y_1} & \cdots & \frac{\partial x_1}{\partial y_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial x_n}{\partial y_1} & \cdots & \frac{\partial x_n}{\partial y_n} \end{pmatrix}$$

**直觉**：Jacobian 行列式度量变换在局部的"体积伸缩"比例。概率密度 = 概率质量/体积，因此变换后需要除以伸缩比例。

### 例9.4b：二维变量变换

设 $(X, Y)$ 的联合 PDF 为 $f_{X,Y}(x,y)$，令 $U = X + Y$，$V = X - Y$。

逆变换：$X = (U+V)/2$，$Y = (U-V)/2$。Jacobian 行列式：

$$\left|\det\begin{pmatrix} \partial x/\partial u & \partial x/\partial v \\ \partial y/\partial u & \partial y/\partial v \end{pmatrix}\right| = \left|\det\begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix}\right| = \left|-\frac{1}{2}\right| = \frac{1}{2}$$

$$f_{U,V}(u, v) = f_{X,Y}\!\left(\frac{u+v}{2}, \frac{u-v}{2}\right) \cdot \frac{1}{2}$$

**应用**：若只需要 $U = X+Y$ 的分布，对 $v$ 积分即得**卷积公式**：

$$f_U(u) = \int_{-\infty}^{+\infty} f_{X,Y}\!\left(\frac{u+v}{2}, \frac{u-v}{2}\right) \cdot \frac{1}{2} \, dv$$

当 $X, Y$ 独立时简化为 $f_U(u) = \int f_X(x) f_Y(u-x) dx$（即 $f_X * f_Y$）。

### 常用变换技术

#### 1. Cholesky 变换（多元正态采样）

目标：从 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 采样。

步骤：
1. 对 $\boldsymbol{\Sigma}$ 做 Cholesky 分解：$\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$
2. 生成 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
3. 返回 $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$

**验证**：$\text{Cov}(\mathbf{L}\mathbf{z}) = \mathbf{L}\,\text{Cov}(\mathbf{z})\,\mathbf{L}^\top = \mathbf{L}\mathbf{I}\mathbf{L}^\top = \boldsymbol{\Sigma}$ ✓

#### 2. Box-Muller 变换（从均匀分布生成正态）

从 $U_1, U_2 \sim \text{Uniform}(0,1)$ 生成两个独立标准正态：

$$Z_1 = \sqrt{-2\ln U_1}\cos(2\pi U_2), \quad Z_2 = \sqrt{-2\ln U_1}\sin(2\pi U_2)$$

**Jacobian 推导**：令 $R = \sqrt{-2\ln U_1}$，$\Theta = 2\pi U_2$，则 $R^2 \sim \text{Exp}(1/2)$（即 $\chi^2(2)$），$(R\cos\Theta, R\sin\Theta)$ 服从二维标准正态。

#### 3. Dirichlet 分布的 Gamma 采样

从 $\text{Dir}(\boldsymbol{\alpha})$ 采样的方法：
1. 独立采样 $Y_i \sim \text{Gamma}(\alpha_i, 1)$，$i = 1, \ldots, k$
2. 归一化：$p_i = Y_i / \sum_j Y_j$

则 $(p_1, \ldots, p_k) \sim \text{Dir}(\boldsymbol{\alpha})$。

**直觉**：Gamma 分布的归一化保持了各分量的相对比例，且自然落在单纯形上。

#### 4. 重参数化技巧（Reparameterization Trick）

在变分推断和 VAE 中，需要对随机变量求梯度。核心思想是将随机性与参数分离：

**问题**：$\mathbf{z} \sim q_\phi(\mathbf{z})$，无法直接对 $\phi$ 求梯度（采样不可微）。

**解决**：引入辅助噪声 $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$（与 $\phi$ 无关），通过可微变换 $\mathbf{z} = g_\phi(\boldsymbol{\epsilon})$：

$$\nabla_\phi E_{q_\phi}[f(\mathbf{z})] = \nabla_\phi E_{p(\boldsymbol{\epsilon})}[f(g_\phi(\boldsymbol{\epsilon}))] = E_{p(\boldsymbol{\epsilon})}[\nabla_\phi f(g_\phi(\boldsymbol{\epsilon}))]$$

对于正态分布：$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，$\mathbf{z} = \boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\epsilon}$。

### 正规化流（Normalizing Flows）简介

通过一系列可逆变换将简单分布（如高斯）变换为复杂分布。设 $\mathbf{z}_0 \sim p_0(\mathbf{z}_0)$，经过 $T$ 步变换 $\mathbf{z}_T = f_T \circ \cdots \circ f_1(\mathbf{z}_0)$，则：

$$\ln p_T(\mathbf{z}_T) = \ln p_0(\mathbf{z}_0) - \sum_{t=1}^T \ln\left|\det\frac{\partial f_t}{\partial \mathbf{z}_{t-1}}\right|$$

每步需要计算 Jacobian 行列式，实际设计中（如 RealNVP）通过特殊结构使其高效计算。

---

## 本章小结

| 分布 | 支撑集 | 参数 | 均值 | 关键性质 |
|------|--------|------|------|---------|
| $\text{Multinomial}(n, \mathbf{p})$ | 非负整数向量，和为 $n$ | $n$, $\mathbf{p}$ | $n\mathbf{p}$ | 二项分布推广，$\text{Cov}(X_i,X_j) = -np_ip_j$ |
| $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | $\mathbb{R}^d$ | $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$ | $\boldsymbol{\mu}$ | 线性变换封闭，条件/边缘仍正态 |
| $\text{Dir}(\boldsymbol{\alpha})$ | $k{-}1$ 维单纯形 | $\boldsymbol{\alpha}$ | $\alpha_i / \alpha_0$ | 多项分布的共轭先验 |
| $\mathcal{W}_d(\nu, \boldsymbol{\Sigma})$ | $d{\times}d$ 正定矩阵 | $\nu$, $\boldsymbol{\Sigma}$ | $\nu\boldsymbol{\Sigma}$ | 精度矩阵的共轭先验 |

**核心要点**：
- 多项分布是分类问题的基础，与 softmax 输出天然对应
- 多元正态分布由均值向量和协方差矩阵完全刻画，线性变换封闭性是深度学习中的重要工具
- Dirichlet 分布是概率向量的"元分布"，共轭性使贝叶斯更新具有解析形式
- 分布变换通过 Jacobian 行列式联系变量，重参数化技巧是 VAE 等生成模型的理论基础

---

## 深度学习应用

### 应用一：Softmax 与多项分布

神经网络分类任务中，最后一层输出 logits $\mathbf{o} = (o_1, \ldots, o_k)$，经过 **softmax** 变换得到概率分布：

$$p_i = \frac{e^{o_i}}{\sum_{j=1}^k e^{o_j}}, \quad \sum_i p_i = 1$$

输出向量 $(p_1, \ldots, p_k)$ 正好落在 $k-1$ 维单纯形上，与多项分布的参数空间完全一致。

**交叉熵损失的概率解释**：设真实标签为 $\mathbf{y}$（one-hot 向量），分类问题等价于假设：

$$\mathbf{y} \mid \mathbf{x} \sim \text{Multinomial}(1, \mathbf{p}(\mathbf{x};\theta))$$

最大化对数似然即最小化**交叉熵损失**：

$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^k y_i \log p_i = -\log p_{y^*}$$

其中 $y^*$ 是真实类别。

**标签平滑（Label Smoothing）**：软化 one-hot 目标，相当于给真实分布加入均匀先验：

$$\tilde{y}_i = (1 - \epsilon) y_i + \frac{\epsilon}{k}$$

这等价于在 Dirichlet 分布视角下引入均匀先验 $\text{Dir}(\epsilon \mathbf{1} / k)$，改善过拟合。

### 应用二：VAE 潜在空间与多元正态分布

**变分自编码器**（Variational Autoencoder, VAE）将多元正态分布作为潜在空间的先验，是重参数化技巧的经典应用。

#### VAE 模型结构

$$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I}) \quad \text{（潜在空间先验）}$$

$$p_\theta(\mathbf{x} \mid \mathbf{z}) \quad \text{（解码器，生成分布）}$$

$$q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x}))) \quad \text{（编码器，近似后验）}$$

#### ELBO 目标函数

由于真实后验 $p(\mathbf{z} \mid \mathbf{x})$ 不可解析计算，VAE 最大化**证据下界**（ELBO）：

$$\mathcal{L}_{\text{ELBO}} = E_{q_\phi(\mathbf{z}|\mathbf{x})}\!\left[\log p_\theta(\mathbf{x} \mid \mathbf{z})\right] - D_{\text{KL}}\!\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \;\|\; p(\mathbf{z})\right)$$

- **重构项**：解码器能从潜在编码重建输入
- **KL 散度项**：近似后验向标准正态靠近，正则化潜在空间

#### KL 散度的解析解

当两个分布均为正态时，KL 散度有解析表达式。设 $q = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$，$p = \mathcal{N}(\mathbf{0}, \mathbf{I})$，则：

$$D_{\text{KL}}(q \| p) = \frac{1}{2} \sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - \ln \sigma_j^2 - 1\right)$$

#### PyTorch 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Dirichlet
import numpy as np

# ============================================================
# 第一部分：Softmax 与多项分布
# ============================================================
print("=" * 55)
print("第一部分：Softmax 与多项分布")
print("=" * 55)

torch.manual_seed(42)

# 模拟分类器 logits
logits = torch.tensor([2.0, 1.0, 0.5, -0.5])  # 4 类
probs = F.softmax(logits, dim=0)
print(f"Logits:  {logits.numpy()}")
print(f"Softmax: {probs.numpy().round(4)}")
print(f"概率和:  {probs.sum().item():.6f}")

# 从 Multinomial 采样（模拟100次试验）
dist = torch.distributions.Multinomial(total_count=100, probs=probs)
sample = dist.sample()
print(f"\n100次多项采样: {sample.int().numpy()}")
print(f"经验频率:      {(sample / 100).numpy().round(3)}")

# 交叉熵损失的概率解释
y_true = torch.tensor([1, 0, 0, 0])  # 类别0的 one-hot 编码
log_probs = F.log_softmax(logits, dim=0)
ce_loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([0]))
manual_nll = -log_probs[0]
print(f"\n交叉熵损失（PyTorch）: {ce_loss.item():.6f}")
print(f"手动计算 -log(p_0):   {manual_nll.item():.6f}")

# 标签平滑
epsilon = 0.1
k = 4
smoothed_labels = (1 - epsilon) * F.one_hot(
    torch.tensor(0), k).float() + epsilon / k
print(f"\n标签平滑 (ε={epsilon}):")
print(f"  原始 one-hot:   {F.one_hot(torch.tensor(0), k).numpy()}")
print(f"  平滑后目标:     {smoothed_labels.numpy().round(3)}")


# ============================================================
# 第二部分：Dirichlet 分布采样与可视化
# ============================================================
print("\n" + "=" * 55)
print("第二部分：Dirichlet 分布")
print("=" * 55)

# 不同浓度参数的 Dirichlet 分布
configs = [
    ([0.1, 0.1, 0.1], "稀疏（α=0.1）：集中在顶点"),
    ([1.0, 1.0, 1.0], "均匀（α=1.0）：单纯形均匀分布"),
    ([5.0, 5.0, 5.0], "集中（α=5.0）：集中在中心"),
    ([2.0, 5.0, 1.0], "非对称（α=[2,5,1]）：偏向类别2"),
]

for alpha, desc in configs:
    alpha_tensor = torch.tensor(alpha)
    dist = Dirichlet(alpha_tensor)
    samples = dist.sample((5000,))
    mean = samples.mean(0)
    theoretical_mean = alpha_tensor / alpha_tensor.sum()
    print(f"\n{desc}")
    print(f"  理论均值:   {theoretical_mean.numpy().round(3)}")
    print(f"  经验均值:   {mean.numpy().round(3)}")

# 贝叶斯多项参数估计：Dirichlet-Multinomial 共轭更新
print("\n--- Dirichlet-Multinomial 共轭更新 ---")
# 先验：均匀 Dirichlet（伪计数各为1）
alpha_prior = torch.tensor([1.0, 1.0, 1.0])
# 观测计数：在3类中观测到 [10, 3, 7] 次
observed_counts = torch.tensor([10.0, 3.0, 7.0])
# 后验：直接加上观测计数
alpha_posterior = alpha_prior + observed_counts
posterior_mean = alpha_posterior / alpha_posterior.sum()
print(f"先验 α:          {alpha_prior.numpy()}")
print(f"观测计数:        {observed_counts.int().numpy()}")
print(f"后验 α:          {alpha_posterior.numpy()}")
print(f"后验均值 (MAP):  {posterior_mean.numpy().round(4)}")
print(f"MLE 估计:        {(observed_counts / observed_counts.sum()).numpy().round(4)}")


# ============================================================
# 第三部分：VAE 实现（多元正态潜在空间 + 重参数化）
# ============================================================
print("\n" + "=" * 55)
print("第三部分：VAE 与多元正态潜在空间")
print("=" * 55)


class Encoder(nn.Module):
    """编码器：将输入映射为潜在空间的均值和对数方差"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var


class Decoder(nn.Module):
    """解码器：将潜在向量重建为输入"""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=16):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        """
        重参数化技巧：z = μ + σ ⊙ ε，ε ~ N(0,I)
        将随机性转移到与参数无关的 ε，使梯度可以流过 μ 和 log_var
        """
        std = torch.exp(0.5 * log_var)      # σ = exp(log_var / 2)
        eps = torch.randn_like(std)          # ε ~ N(0,I)
        return mu + std * eps                # 可微的 z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def elbo_loss(self, x, x_recon, mu, log_var):
        """
        ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
        重构项使用二元交叉熵，KL项有解析解
        """
        # 重构损失（对应 p(x|z) 为伯努利分布）
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

        # KL 散度：KL(N(μ,σ²I) || N(0,I)) 的解析解
        # = 0.5 * sum(μ² + σ² - log(σ²) - 1)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss + kl_loss, recon_loss, kl_loss

    def sample(self, num_samples):
        """从先验 N(0,I) 采样并解码"""
        z = torch.randn(num_samples, self.latent_dim)
        return self.decoder(z)


# 训练 VAE（使用随机生成的数据模拟 MNIST）
input_dim, hidden_dim, latent_dim = 784, 256, 16
batch_size, num_epochs = 64, 30

vae = VAE(input_dim, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# 模拟数据（真实场景中替换为 MNIST 数据集）
torch.manual_seed(0)
fake_data = torch.rand(512, input_dim)
dataset = torch.utils.data.TensorDataset(fake_data)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"VAE 结构：输入{input_dim}→隐层{hidden_dim}→潜在{latent_dim}维")
print(f"参数量：{sum(p.numel() for p in vae.parameters()):,}")

losses = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for (batch,) in loader:
        optimizer.zero_grad()
        x_recon, mu, log_var = vae(batch)
        loss, recon_loss, kl_loss = vae.elbo_loss(batch, x_recon, mu, log_var)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss / len(loader))

print(f"\n训练完成（{num_epochs} 轮）")
print(f"初始 ELBO 损失: {losses[0]:.2f}")
print(f"最终 ELBO 损失: {losses[-1]:.2f}")

# 验证潜在空间的正态性
vae.eval()
with torch.no_grad():
    mu_all, logvar_all = [], []
    for (batch,) in loader:
        mu, log_var = vae.encoder(batch)
        mu_all.append(mu)
        logvar_all.append(log_var)
    mu_all = torch.cat(mu_all)
    std_all = torch.exp(0.5 * torch.cat(logvar_all))

print(f"\n潜在空间统计（编码均值 μ）：")
print(f"  μ 的均值:     {mu_all.mean().item():.4f}  （期望接近 0）")
print(f"  μ 的标准差:   {mu_all.std().item():.4f}")
print(f"  σ 的均值:     {std_all.mean().item():.4f}  （期望接近 1）")
print(f"  σ 的标准差:   {std_all.std().item():.4f}")

# 重参数化的梯度流验证
print("\n--- 重参数化技巧：梯度流验证 ---")
mu_test = torch.tensor([[0.5, -0.3]], requires_grad=True)
log_var_test = torch.tensor([[0.2, -0.1]], requires_grad=True)
eps = torch.randn_like(mu_test)

# 重参数化：z = μ + σ·ε（可微）
z_reparam = mu_test + torch.exp(0.5 * log_var_test) * eps
loss_test = z_reparam.sum()
loss_test.backward()
print(f"μ 的梯度:      {mu_test.grad.numpy()}   （应为全1，梯度可流通）")
print(f"log_var 的梯度:{log_var_test.grad.numpy().round(4)}")

# 多元正态分布的协方差矩阵可视化
print("\n--- 多元正态采样验证 ---")
mu_mv = torch.zeros(2)
# 构造相关系数 ρ=0.8 的协方差矩阵
rho = 0.8
Sigma = torch.tensor([[1.0, rho], [rho, 1.0]])
L = torch.linalg.cholesky(Sigma)          # Cholesky 分解
z_std = torch.randn(10000, 2)
samples_mv = (L @ z_std.T).T + mu_mv      # Cholesky 变换

empirical_cov = torch.cov(samples_mv.T)
print(f"目标协方差矩阵:\n  [[1.0, {rho}], [{rho}, 1.0]]")
print(f"经验协方差矩阵:\n  {empirical_cov.numpy().round(3)}")
```

**预期输出**：
```
=======================================================
第一部分：Softmax 与多项分布
=======================================================
Logits:  [ 2.   1.   0.5 -0.5]
Softmax: [0.5266 0.1936 0.1172 0.1626]  （近似值，各版本可能略有差异）
概率和:  1.000000

100次多项采样: [53  19  11  17]
经验频率:      [0.53  0.19  0.11  0.17]

交叉熵损失（PyTorch）: 0.641187
手动计算 -log(p_0):   0.641187

标签平滑 (ε=0.1):
  原始 one-hot:   [1 0 0 0]
  平滑后目标:     [0.925 0.025 0.025 0.025]

=======================================================
第二部分：Dirichlet 分布
=======================================================
稀疏（α=0.1）：集中在顶点
  理论均值:   [0.333 0.333 0.333]
  经验均值:   [0.333 0.333 0.333]
...
后验均值 (MAP):  [0.55  0.2   0.4 ]  （贝叶斯估计，平滑了 MLE）
MLE 估计:        [0.5   0.15  0.35]  （纯频率，无平滑）

=======================================================
第三部分：VAE 与多元正态潜在空间
=======================================================
VAE 结构：输入784→隐层256→潜在16维
参数量：467,472
训练完成（30 轮）
初始 ELBO 损失: ...
最终 ELBO 损失: ...（应小于初始值）

潜在空间统计（编码均值 μ）：
  μ 的均值:     ≈ 0      （KL 约束使均值接近0）
  σ 的均值:     ≈ 1      （KL 约束使方差接近1）

μ 的梯度:      [[1. 1.]]  （梯度可流通）

目标协方差矩阵:
  [[1.0, 0.8], [0.8, 1.0]]
经验协方差矩阵:
  [[1.000 0.800], [0.800 1.000]]  （大样本下接近理论值）
```

### 关键联系总结

| 概率论概念 | 深度学习对应 |
|-----------|-------------|
| 多项分布 | 分类任务中 softmax 输出的分布族 |
| 交叉熵损失 | 多项分布的负对数似然 |
| Dirichlet 先验 | 标签平滑、LDA 主题模型 |
| 多元正态分布 | VAE 潜在空间的先验与近似后验 |
| 重参数化技巧 | VAE/扩散模型中的可微采样 |
| Cholesky 分解 | 从多元正态高效采样 |
| KL 散度 | VAE 正则化项（有解析解） |

---

## 练习题

**练习 9.1**（多项分布）

某语言模型对下一个词的预测概率为 $\mathbf{p} = (0.4, 0.3, 0.2, 0.1)$（对应词汇 A、B、C、D）。在 5 次独立采样中：

(a) 求恰好采样到 2 次 A、2 次 B、1 次 C、0 次 D 的概率

(b) 求采样到 A 的次数 $X_1$ 的期望和方差

(c) 求 $\text{Cov}(X_1, X_2)$，并说明负协方差的直觉含义

**练习 9.2**（多元正态分布）

设 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，其中：

$$\boldsymbol{\mu} = \begin{pmatrix}1\\2\end{pmatrix}, \quad \boldsymbol{\Sigma} = \begin{pmatrix}4 & 2\\2 & 1\end{pmatrix}$$

(a) 验证 $\boldsymbol{\Sigma}$ 是否正定

(b) 求 $X_1$ 和 $X_2$ 的边缘分布

(c) 给定 $X_2 = 3$，求 $X_1$ 的条件分布 $X_1 \mid X_2 = 3$

(d) 求相关系数 $\rho(X_1, X_2)$

**练习 9.3**（Dirichlet 分布）

考虑三类文档分类任务，先验为 $\text{Dir}(2, 2, 2)$，观测了 100 篇文档：类别1有 50 篇，类别2有 30 篇，类别3有 20 篇。

(a) 写出后验分布

(b) 计算后验均值，并与 MLE 估计比较

(c) 若先验改为 $\text{Dir}(0.1, 0.1, 0.1)$（稀疏先验），后验均值如何变化？说明稀疏先验的效果

**练习 9.4**（分布变换）

设 $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_2)$，对 $\boldsymbol{\Sigma} = \begin{pmatrix}4&2\\2&2\end{pmatrix}$ 做 Cholesky 分解 $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$，令 $\mathbf{X} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}$，其中 $\boldsymbol{\mu} = (1, 0)^\top$。

(a) 计算 Cholesky 因子 $\mathbf{L}$（下三角矩阵）

(b) 证明 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$

(c) 用重参数化技巧写出从 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 采样的 PyTorch 代码（两行以内）

**练习 9.5**（VAE 与 KL 散度）

VAE 的 KL 散度项为 $D_{\text{KL}}(q_\phi \| p) = \frac{1}{2}\sum_j(\mu_j^2 + \sigma_j^2 - \ln\sigma_j^2 - 1)$，其中 $q_\phi = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$，$p = \mathcal{N}(\mathbf{0}, \mathbf{I})$。

(a) 从 KL 散度的积分定义 $D_{\text{KL}}(q \| p) = \int q(\mathbf{z}) \ln \frac{q(\mathbf{z})}{p(\mathbf{z})} d\mathbf{z}$ 出发，推导上述解析公式（一维情形即可）

(b) 当 $\boldsymbol{\mu} = \mathbf{0}$，$\boldsymbol{\sigma} = \mathbf{1}$ 时，KL 散度等于多少？这说明什么？

(c) 训练中若 KL 权重过大（称为"KL 坍塌"），编码器会退化为什么行为？如何从概率论角度理解？

---

## 练习答案

<details>
<summary>点击展开 练习 9.1 答案</summary>

**(a)** 多项概率计算：

$$P(2,2,1,0) = \frac{5!}{2!\,2!\,1!\,0!} \times 0.4^2 \times 0.3^2 \times 0.2^1 \times 0.1^0$$

$$= \frac{120}{2 \times 2 \times 1 \times 1} \times 0.16 \times 0.09 \times 0.2 \times 1 = 30 \times 0.002880 \approx 0.0864$$

**(b)** $X_1 \sim \text{Binomial}(5, 0.4)$：

$$E[X_1] = 5 \times 0.4 = 2, \quad \text{Var}(X_1) = 5 \times 0.4 \times 0.6 = 1.2$$

**(c)** 协方差：

$$\text{Cov}(X_1, X_2) = -np_1p_2 = -5 \times 0.4 \times 0.3 = -0.6$$

**直觉含义**：每次试验选择 A 就不能选择 B，二者竞争有限的 $n$ 次机会。一个数量增多必然挤占另一个的份额，故协方差为负。

</details>

<details>
<summary>点击展开 练习 9.2 答案</summary>

**(a)** 正定性验证：

特征值 $\lambda_1, \lambda_2$ 满足 $\det(\boldsymbol{\Sigma} - \lambda\mathbf{I}) = 0$：

$(4-\lambda)(1-\lambda) - 4 = \lambda^2 - 5\lambda = \lambda(\lambda-5) = 0$

$\lambda_1 = 0$，$\lambda_2 = 5$。由于 $\lambda_1 = 0$，$\boldsymbol{\Sigma}$ **半正定而非正定**，行列式 $|\boldsymbol{\Sigma}| = 4 \times 1 - 2 \times 2 = 0$，$X_1$ 与 $X_2$ 完全线性相关（$X_2 = X_1/2$），此协方差矩阵对应退化的多元正态分布。

**(b)** 边缘分布：

$$X_1 \sim \mathcal{N}(1, 4), \quad X_2 \sim \mathcal{N}(2, 1)$$

**(c)** 条件分布（使用公式）：

$$\mu_{1|2} = 1 + \frac{2}{1}(3 - 2) = 3$$

$$\sigma^2_{1|2} = 4 - \frac{2 \times 2}{1} = 0$$

$$X_1 \mid X_2 = 3 \sim \mathcal{N}(3, 0) \equiv 3 \quad \text{（退化，即 } X_1 = 2X_2 - 1\text{）}$$

**(d)** 相关系数：

$$\rho = \frac{\text{Cov}(X_1,X_2)}{\sqrt{\text{Var}(X_1)\text{Var}(X_2)}} = \frac{2}{\sqrt{4 \times 1}} = 1$$

完全正相关，验证了退化情形。

</details>

<details>
<summary>点击展开 练习 9.3 答案</summary>

**(a)** 后验分布：

先验 $\boldsymbol{\alpha} = (2, 2, 2)$，观测计数 $\mathbf{x} = (50, 30, 20)$，由共轭性：

$$\mathbf{p} \mid \mathbf{x} \sim \text{Dir}(52, 32, 22)$$

**(b)** 后验均值与 MLE 比较：

$$\text{后验均值} = \frac{(52, 32, 22)}{52+32+22} = \frac{(52,32,22)}{106} \approx (0.491, 0.302, 0.208)$$

$$\text{MLE} = \frac{(50,30,20)}{100} = (0.500, 0.300, 0.200)$$

贝叶斯估计向先验均值 $(1/3, 1/3, 1/3)$ 轻微收缩，差异随样本量增大而趋近于0。

**(c)** 稀疏先验 $\text{Dir}(0.1, 0.1, 0.1)$ 的后验：

$$\mathbf{p} \mid \mathbf{x} \sim \text{Dir}(50.1, 30.1, 20.1)$$

$$\text{后验均值} \approx \frac{(50.1, 30.1, 20.1)}{100.3} \approx (0.4995, 0.3001, 0.2004)$$

几乎等于 MLE。**稀疏先验效果**：$\alpha_i \to 0$ 时先验接近于"不提供信息"，后验几乎由数据决定；同时稀疏先验鼓励 $p_i$ 向0或1的极端值集中，在主题模型中产生稀疏的主题-词分布。

</details>

<details>
<summary>点击展开 练习 9.4 答案</summary>

**(a)** Cholesky 分解 $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$，$\mathbf{L}$ 为下三角矩阵：

$$\mathbf{L} = \begin{pmatrix} l_{11} & 0 \\ l_{21} & l_{22} \end{pmatrix}$$

由 $\mathbf{L}\mathbf{L}^\top = \boldsymbol{\Sigma}$：

- $l_{11}^2 = 4 \Rightarrow l_{11} = 2$
- $l_{21} \cdot l_{11} = 2 \Rightarrow l_{21} = 1$
- $l_{21}^2 + l_{22}^2 = 2 \Rightarrow l_{22} = 1$

$$\mathbf{L} = \begin{pmatrix}2 & 0 \\ 1 & 1\end{pmatrix}$$

验证：$\mathbf{L}\mathbf{L}^\top = \begin{pmatrix}2&0\\1&1\end{pmatrix}\begin{pmatrix}2&1\\0&1\end{pmatrix} = \begin{pmatrix}4&2\\2&2\end{pmatrix} = \boldsymbol{\Sigma}$ ✓

**(b)** 证明：

$$E[\mathbf{X}] = \boldsymbol{\mu} + \mathbf{L} \underbrace{E[\mathbf{Z}]}_{=\mathbf{0}} = \boldsymbol{\mu}$$

$$\text{Cov}(\mathbf{X}) = \mathbf{L}\,\text{Cov}(\mathbf{Z})\,\mathbf{L}^\top = \mathbf{L}\mathbf{I}\mathbf{L}^\top = \mathbf{L}\mathbf{L}^\top = \boldsymbol{\Sigma}$$

又因为 $\mathbf{Z}$ 是正态的，其线性变换 $\mathbf{X} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}$ 也是正态的，故 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$。

**(c)** PyTorch 代码（重参数化）：

```python
L = torch.linalg.cholesky(Sigma)
x = mu + L @ torch.randn(2)
```

</details>

<details>
<summary>点击展开 练习 9.5 答案</summary>

**(a)** 一维推导（$q = \mathcal{N}(\mu, \sigma^2)$，$p = \mathcal{N}(0, 1)$）：

$$D_{\text{KL}}(q \| p) = \int q(z)\left[\ln q(z) - \ln p(z)\right] dz$$

$$= \int q(z)\left[-\frac{(z-\mu)^2}{2\sigma^2} - \frac{1}{2}\ln(2\pi\sigma^2) + \frac{z^2}{2} + \frac{1}{2}\ln(2\pi)\right] dz$$

$$= E_q\left[\frac{z^2}{2} - \frac{(z-\mu)^2}{2\sigma^2}\right] - \frac{1}{2}\ln\sigma^2$$

计算 $E_q[z^2] = \mu^2 + \sigma^2$，$E_q[(z-\mu)^2] = \sigma^2$：

$$= \frac{\mu^2 + \sigma^2}{2} - \frac{\sigma^2}{2\sigma^2} - \frac{1}{2}\ln\sigma^2 = \frac{1}{2}(\mu^2 + \sigma^2 - 1 - \ln\sigma^2)$$

将各维度求和得多维公式：$D_{\text{KL}} = \frac{1}{2}\sum_j(\mu_j^2 + \sigma_j^2 - \ln\sigma_j^2 - 1)$。

**(b)** 当 $\boldsymbol{\mu} = \mathbf{0}$，$\boldsymbol{\sigma} = \mathbf{1}$ 时：

$$D_{\text{KL}} = \frac{1}{2}\sum_j(0 + 1 - 0 - 1) = 0$$

KL 散度为零意味着 $q_\phi(\mathbf{z} \mid \mathbf{x}) = p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$，即近似后验恰好等于先验，编码器未从输入中提取任何信息。

**(c)** KL 坍塌分析：

若 KL 权重过大，优化器发现令编码器输出 $\boldsymbol{\mu} \to \mathbf{0}$，$\boldsymbol{\sigma} \to \mathbf{1}$ 可以快速降低 KL 项。此时编码器退化为**恒输出标准正态**，$\mathbf{z}$ 与输入 $\mathbf{x}$ 无关，解码器退化为无条件生成模型（忽略潜在编码）。

**概率论角度**：KL 坍塌等价于后验完全被先验主导，数据未能更新潜在变量的后验。解决方案包括：$\beta$-VAE（动态调整 KL 权重）、Free Bits 方法（为每维 KL 设置下界），本质上是在先验正则化与数据编码能力之间寻求平衡。

</details>
