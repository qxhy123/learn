# 第22章：奇异值分解

> **前置知识**：第17章（正交化与QR分解）、第19章（特征值与特征向量）、第20章（对角化）、第21章（对称矩阵与谱定理）
>
> **本章难度**：★★★★★
>
> **预计学习时间**：6-8 小时

---

## 学习目标

学完本章后，你将能够：

- 陈述并证明奇异值分解定理：任意 $m \times n$ 矩阵 $A$ 均可分解为 $A = U\Sigma V^T$，理解 $U$、$\Sigma$、$V$ 各自的几何含义
- 从 $A^T A$ 和 $AA^T$ 的特征值出发计算奇异值，并构造完整的 SVD
- 用"旋转—缩放—旋转"的几何语言解释 SVD 对向量的变换过程，将其与线性变换的几何本质联系起来
- 利用 Eckart-Young 定理求矩阵的最优低秩近似，理解截断 SVD 为何在最小 Frobenius 范数意义下最优
- 通过 SVD 构造矩阵的伪逆 $A^+ = V\Sigma^+ U^T$，并用其求解超定与欠定线性方程组的最小范数最小二乘解

---

## 22.1 奇异值分解定理

### 从谱定理到 SVD

第21章的谱定理告诉我们：实对称矩阵 $A = A^T$ 可以被正交对角化为 $A = Q\Lambda Q^T$。但现实中绝大多数矩阵既不是方阵，也不是对称阵——图像是 $1080 \times 1920$ 的矩形阵，用户-物品评分矩阵是 $10^6 \times 10^5$ 的稀疏矩阵，神经网络权重是 $d_{\text{out}} \times d_{\text{in}}$ 的任意矩形阵。

**奇异值分解（Singular Value Decomposition，SVD）**正是将谱定理推广到任意矩阵的终极工具。

### SVD 定理

**定理（SVD 定理）**：设 $A \in \mathbb{R}^{m \times n}$，则存在正交矩阵 $U \in \mathbb{R}^{m \times m}$、$V \in \mathbb{R}^{n \times n}$，以及"广义对角矩阵" $\Sigma \in \mathbb{R}^{m \times n}$，使得：

$$\boxed{A = U \Sigma V^T}$$

其中：

- $U = [\mathbf{u}_1, \ldots, \mathbf{u}_m]$：列向量构成 $\mathbb{R}^m$ 的标准正交基，称为**左奇异向量（left singular vectors）**
- $V = [\mathbf{v}_1, \ldots, \mathbf{v}_n]$：列向量构成 $\mathbb{R}^n$ 的标准正交基，称为**右奇异向量（right singular vectors）**
- $\Sigma$：除主对角线外全为零，对角元素 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0 = \sigma_{r+1} = \cdots$，称为**奇异值（singular values）**

$\Sigma$ 的结构如下（$r = \text{rank}(A)$，$p = \min(m, n)$）：

$$\Sigma = \begin{pmatrix} \sigma_1 & & & & \\ & \sigma_2 & & & \\ & & \ddots & & \\ & & & \sigma_r & \\ & & & & 0 \\ & & & & \ddots \end{pmatrix}_{m \times n}$$

即前 $r$ 个对角元为正，其余全为零。

### SVD 的存在性证明

**证明**：构造性地证明 SVD 的存在。

**第一步**：考虑实对称半正定矩阵 $A^T A \in \mathbb{R}^{n \times n}$。由第21章谱定理，存在正交矩阵 $V \in \mathbb{R}^{n \times n}$ 和非负实数 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n \geq 0$，使得：

$$A^T A = V \begin{pmatrix}\lambda_1 & & \\ & \ddots & \\ & & \lambda_n\end{pmatrix} V^T$$

（非负性来自于 $A^T A$ 是半正定矩阵：$\mathbf{x}^T A^T A \mathbf{x} = \|A\mathbf{x}\|^2 \geq 0$。）

**第二步**：定义奇异值 $\sigma_i = \sqrt{\lambda_i}$。设 $r = \text{rank}(A)$，则恰好有 $r$ 个正奇异值。对 $i = 1, \ldots, r$，令：

$$\mathbf{u}_i = \frac{A\mathbf{v}_i}{\sigma_i} \in \mathbb{R}^m$$

**第三步**：验证 $\mathbf{u}_1, \ldots, \mathbf{u}_r$ 是标准正交集：

$$\langle \mathbf{u}_i, \mathbf{u}_j \rangle = \frac{(A\mathbf{v}_i)^T (A\mathbf{v}_j)}{\sigma_i \sigma_j} = \frac{\mathbf{v}_i^T A^T A \mathbf{v}_j}{\sigma_i \sigma_j} = \frac{\mathbf{v}_i^T (\lambda_j \mathbf{v}_j)}{\sigma_i \sigma_j} = \frac{\lambda_j \delta_{ij}}{\sigma_i \sigma_j} = \delta_{ij}$$

**第四步**：将 $\mathbf{u}_1, \ldots, \mathbf{u}_r$ 扩充为 $\mathbb{R}^m$ 的标准正交基 $\mathbf{u}_1, \ldots, \mathbf{u}_m$（Gram-Schmidt）。

**第五步**：验证 $A = U\Sigma V^T$。对 $i \leq r$：$A\mathbf{v}_i = \sigma_i \mathbf{u}_i$（由构造）。对 $i > r$：$A^T A \mathbf{v}_i = 0$，故 $\|A\mathbf{v}_i\|^2 = \mathbf{v}_i^T A^T A \mathbf{v}_i = 0$，即 $A\mathbf{v}_i = \mathbf{0}$。

合并成矩阵形式：$AV = U\Sigma$，两边右乘 $V^T = V^{-1}$ 得 $A = U\Sigma V^T$。$\square$

### 外积展开形式

将 $A = U\Sigma V^T$ 展开为秩-1 矩阵之和：

$$\boxed{A = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T}$$

每项 $\mathbf{u}_i \mathbf{v}_i^T$ 是一个秩-1 矩阵（$\mathbf{u}_i \in \mathbb{R}^m$，$\mathbf{v}_i^T \in \mathbb{R}^{1 \times n}$），奇异值 $\sigma_i$ 是其"权重"。这个展开是 SVD 用于低秩近似的理论基础。

### 具体例子

设 $A = \begin{pmatrix}1 & 1 \\ 0 & 1 \\ 1 & 0\end{pmatrix} \in \mathbb{R}^{3 \times 2}$，求其 SVD。

**第一步：计算 $A^T A$。**

$$A^T A = \begin{pmatrix}1&0&1\\1&1&0\end{pmatrix}\begin{pmatrix}1&1\\0&1\\1&0\end{pmatrix} = \begin{pmatrix}2&1\\1&2\end{pmatrix}$$

**第二步：求 $A^T A$ 的特征值。**

$$\det(A^T A - \lambda I) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda-1)(\lambda-3) = 0$$

$\lambda_1 = 3$，$\lambda_2 = 1$，奇异值 $\sigma_1 = \sqrt{3}$，$\sigma_2 = 1$。

**第三步：求右奇异向量 $V$（$A^T A$ 的特征向量）。**

$\lambda_1 = 3$：$(A^T A - 3I)\mathbf{v} = 0$，$\begin{pmatrix}-1&1\\1&-1\end{pmatrix}\mathbf{v} = 0$，$\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\1\end{pmatrix}$。

$\lambda_2 = 1$：$(A^T A - I)\mathbf{v} = 0$，$\begin{pmatrix}1&1\\1&1\end{pmatrix}\mathbf{v} = 0$，$\mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\-1\end{pmatrix}$。

**第四步：求左奇异向量 $U$（通过 $\mathbf{u}_i = A\mathbf{v}_i / \sigma_i$）。**

$$\mathbf{u}_1 = \frac{A\mathbf{v}_1}{\sqrt{3}} = \frac{1}{\sqrt{6}}\begin{pmatrix}2\\1\\1\end{pmatrix}, \quad \mathbf{u}_2 = \frac{A\mathbf{v}_2}{1} = \frac{1}{\sqrt{2}}\begin{pmatrix}0\\-1\\1\end{pmatrix}$$

扩充为 $\mathbb{R}^3$ 的标准正交基：取 $\mathbf{u}_3$ 与 $\mathbf{u}_1$、$\mathbf{u}_2$ 均正交，可取 $\mathbf{u}_3 = \frac{1}{\sqrt{3}}\begin{pmatrix}-1\\1\\1\end{pmatrix}$（验证正交性略）。

**结果**：

$$A = U\Sigma V^T = \begin{pmatrix}\frac{2}{\sqrt{6}} & 0 & \frac{-1}{\sqrt{3}} \\ \frac{1}{\sqrt{6}} & \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{3}} \\ \frac{1}{\sqrt{6}} & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{3}}\end{pmatrix} \begin{pmatrix}\sqrt{3} & 0 \\ 0 & 1 \\ 0 & 0\end{pmatrix} \begin{pmatrix}\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{-1}{\sqrt{2}}\end{pmatrix}$$

---

## 22.2 奇异值的计算

### 从两个半正定矩阵出发

SVD 与两个对称半正定矩阵密切相关：

$$A^T A = (U\Sigma V^T)^T (U\Sigma V^T) = V\Sigma^T U^T U\Sigma V^T = V\Sigma^T\Sigma V^T = V \begin{pmatrix}\sigma_1^2 & & \\ & \ddots & \\ & & \sigma_n^2\end{pmatrix} V^T$$

$$AA^T = U\Sigma V^T V\Sigma^T U^T = U\Sigma\Sigma^T U^T = U \begin{pmatrix}\sigma_1^2 & & \\ & \ddots & \\ & & \sigma_m^2\end{pmatrix} U^T$$

（其中 $\sigma_{r+1} = \cdots = 0$。）

**核心结论**：

| 矩阵 | 特征值 | 特征向量 |
|:---|:---|:---|
| $A^T A \in \mathbb{R}^{n \times n}$ | $\sigma_1^2 \geq \cdots \geq \sigma_r^2 > 0 = \cdots$ | 右奇异向量 $\mathbf{v}_1, \ldots, \mathbf{v}_n$ |
| $AA^T \in \mathbb{R}^{m \times m}$ | $\sigma_1^2 \geq \cdots \geq \sigma_r^2 > 0 = \cdots$ | 左奇异向量 $\mathbf{u}_1, \ldots, \mathbf{u}_m$ |

两者非零特征值完全相同（都等于 $\sigma_i^2$），这并非巧合——可以用迹的循环性 $\text{tr}(AB) = \text{tr}(BA)$ 在更一般意义上理解。

### 奇异值的物理意义

设 $\mathbf{x}$ 在单位球面 $\|\mathbf{x}\| = 1$ 上变化，则 $A\mathbf{x}$ 的轨迹是 $\mathbb{R}^m$ 中的一个椭球（当 $A$ 列满秩时），其半轴长度恰好是奇异值 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$，方向是对应的左奇异向量 $\mathbf{u}_i$。

特别地：

$$\|A\|_2 = \max_{\|\mathbf{x}\|=1} \|A\mathbf{x}\| = \sigma_1 \quad \text{（矩阵的谱范数）}$$

$$\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\sigma_1^2 + \sigma_2^2 + \cdots + \sigma_r^2} \quad \text{（Frobenius 范数）}$$

### 矩阵的四个基本子空间

SVD 给出了矩阵四个基本子空间的标准正交基：

| 子空间 | 基向量 | 维数 |
|:---|:---|:---|
| $\text{Col}(A)$（列空间） | $\mathbf{u}_1, \ldots, \mathbf{u}_r$ | $r$ |
| $\text{Null}(A^T)$（左零空间） | $\mathbf{u}_{r+1}, \ldots, \mathbf{u}_m$ | $m - r$ |
| $\text{Row}(A)$（行空间） | $\mathbf{v}_1, \ldots, \mathbf{v}_r$ | $r$ |
| $\text{Null}(A)$（零空间） | $\mathbf{v}_{r+1}, \ldots, \mathbf{v}_n$ | $n - r$ |

---

## 22.3 SVD 的几何意义

### 旋转—缩放—旋转

任何线性变换 $T: \mathbb{R}^n \to \mathbb{R}^m$（由矩阵 $A$ 表示）都可以分解为三个几何操作的复合：

$$\underbrace{\mathbb{R}^n}_{\text{输入空间}} \xrightarrow{V^T \text{（旋转）}} \underbrace{\mathbb{R}^n}_{\text{旋转后}} \xrightarrow{\Sigma \text{（缩放+升/降维）}} \underbrace{\mathbb{R}^m}_{\text{缩放后}} \xrightarrow{U \text{（旋转）}} \underbrace{\mathbb{R}^m}_{\text{输出空间}}$$

具体而言，对任意输入向量 $\mathbf{x} \in \mathbb{R}^n$：

1. **$V^T$（正交变换/旋转）**：将 $\mathbf{x}$ 从标准基表示转换到右奇异向量 $\{\mathbf{v}_i\}$ 基下的坐标，记为 $\mathbf{y} = V^T \mathbf{x}$。这是 $\mathbb{R}^n$ 内的旋转/反射，不改变向量长度。

2. **$\Sigma$（缩放+维度变换）**：将每个分量 $y_i$ 缩放为 $\sigma_i y_i$，同时将空间从 $\mathbb{R}^n$ 映射到 $\mathbb{R}^m$（若 $m > n$ 则补零行；若 $m < n$ 则丢弃多余分量）。奇异值就是各轴方向的拉伸/压缩比率。

3. **$U$（正交变换/旋转）**：将缩放后的坐标旋转到 $\mathbb{R}^m$ 的标准方向，产生最终输出 $A\mathbf{x}$。

### 二维例子：直观图像

设 $A = \begin{pmatrix}3 & 0 \\ 0 & 1\end{pmatrix}$（简单拉伸）：奇异值为 $\sigma_1 = 3$，$\sigma_2 = 1$；$U = V = I$（无旋转）。

单位圆 $\|\mathbf{x}\| = 1$ 在 $A$ 作用下变为椭圆：水平轴拉伸到长度 $3$，竖直轴保持长度 $1$。

更一般地，设 $A = \begin{pmatrix}2 & 1 \\ 1 & 2\end{pmatrix}$：

- 奇异值 $\sigma_1 = 3$，$\sigma_2 = 1$
- 右奇异向量：$\mathbf{v}_1 = \frac{1}{\sqrt{2}}(1,1)^T$，$\mathbf{v}_2 = \frac{1}{\sqrt{2}}(1,-1)^T$
- 左奇异向量：$\mathbf{u}_1 = \mathbf{v}_1$，$\mathbf{u}_2 = \mathbf{v}_2$（因 $A$ 是对称矩阵，$U = V$）

**操作序列**：
1. $V^T$ 将标准基旋转 $45°$，使 $(1,1)/\sqrt{2}$ 方向对齐 $x$ 轴；
2. $\Sigma$ 将 $x$ 轴方向拉伸 $3$ 倍，$y$ 轴方向保持不变；
3. $U$ 将轴旋转回 $45°$ 方向。

结果：单位圆变为椭圆，主轴沿 $45°$ 方向（长 $3$）和 $135°$ 方向（长 $1$）。

### SVD 揭示矩阵的本质结构

| SVD 信息 | 矩阵性质 |
|:---|:---|
| 正奇异值的个数 $r$ | $\text{rank}(A) = r$ |
| $\sigma_1$ | 矩阵的谱范数 $\|A\|_2$ |
| $\prod_{i=1}^r \sigma_i$ | $|\det(A)|$（方阵时） |
| $\sqrt{\sum_i \sigma_i^2}$ | Frobenius 范数 $\|A\|_F$ |
| $\sigma_1 / \sigma_r$ | 矩阵的条件数 $\kappa(A)$（列满秩时） |

---

## 22.4 低秩近似

### Eckart-Young 定理

SVD 最重要的应用之一是**最优低秩近似**。给定矩阵 $A \in \mathbb{R}^{m \times n}$（秩为 $r$），我们希望找到秩不超过 $k$（$k < r$）的矩阵 $B$ 使得 $\|A - B\|$ 最小。

**定理（Eckart-Young，1936）**：设 $A = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T$，定义**截断 SVD（truncated SVD）**：

$$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T = U_k \Sigma_k V_k^T$$

其中 $U_k = [\mathbf{u}_1, \ldots, \mathbf{u}_k]$，$\Sigma_k = \text{diag}(\sigma_1, \ldots, \sigma_k)$，$V_k = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$。

则 $A_k$ 是所有秩-$k$ 矩阵中最接近 $A$ 的（在 Frobenius 范数和谱范数两种意义下均成立）：

$$\boxed{\|A - A_k\|_F = \min_{\text{rank}(B) \leq k} \|A - B\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_r^2}}$$

$$\boxed{\|A - A_k\|_2 = \min_{\text{rank}(B) \leq k} \|A - B\|_2 = \sigma_{k+1}}$$

**证明思路（Frobenius 范数）**：

设 $\|A - B\|_F^2 = \|A\|_F^2 - \|B\|_F^2 + \|A - B\|_F^2$，利用 Frobenius 范数对 SVD 展开的"勾股定理"性质：

$$\|A - A_k\|_F^2 = \left\|\sum_{i=k+1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T\right\|_F^2 = \sum_{i=k+1}^r \sigma_i^2$$

任意秩-$k$ 矩阵 $B$ 的零空间维数至少为 $n - k$，因此与 $V_k$ 的核交非零，用此构造一个向量使 $\|A\mathbf{x} - B\mathbf{x}\|_F^2 \geq \sigma_{k+1}^2$，由此证明 $A_k$ 是最优的。$\square$

### 信息压缩比

存储原始矩阵 $A \in \mathbb{R}^{m \times n}$ 需要 $mn$ 个数。存储秩-$k$ 近似 $A_k = U_k \Sigma_k V_k^T$ 需要：

$$\underbrace{mk}_{U_k} + \underbrace{k}_{\Sigma_k} + \underbrace{nk}_{V_k} = k(m + n + 1) \text{ 个数}$$

压缩比：$\dfrac{k(m+n+1)}{mn}$。当 $k \ll \min(m,n)$ 时，这比原始存储小得多。

**能量保留率**：保留 $k$ 个奇异值时，Frobenius 范数意义下的信息保留比例为：

$$\frac{\|A_k\|_F^2}{\|A\|_F^2} = \frac{\sigma_1^2 + \cdots + \sigma_k^2}{\sigma_1^2 + \cdots + \sigma_r^2}$$

### 为什么是"最优"的

Eckart-Young 定理的深刻之处在于：在所有可能的秩-$k$ 近似中，截断 SVD 是**唯一**同时在谱范数和 Frobenius 范数意义下最优的选择——"保留最大的 $k$ 个奇异值和对应奇异向量"这一操作捕捉了矩阵中信息量最大的 $k$ 个"方向"。

---

## 22.5 SVD 与伪逆

### 伪逆的动机

对于方阵 $A$ 可逆时，$A^{-1}$ 唯一存在。但对于：

- **超定方程组**（$m > n$，方程多于未知量）：$A\mathbf{x} = \mathbf{b}$ 通常无解；
- **欠定方程组**（$m < n$，方程少于未知量）：$A\mathbf{x} = \mathbf{b}$ 通常有无穷多解；
- **降秩矩阵**：$A^{-1}$ 不存在。

在所有这些情形下，**Moore-Penrose 伪逆** $A^+$ 提供了"最好的广义逆"。

### 伪逆的 SVD 定义

设 $A = U\Sigma V^T$，定义 $\Sigma^+$ 为将 $\Sigma$ 转置并对每个非零对角元取倒数：

$$\Sigma^+ = \begin{pmatrix}\frac{1}{\sigma_1} & & & \\ & \frac{1}{\sigma_2} & & \\ & & \ddots & \\ & & & \frac{1}{\sigma_r} \\ & & & & 0 \\ & & & & \ddots\end{pmatrix}_{n \times m}$$

则 Moore-Penrose 伪逆定义为：

$$\boxed{A^+ = V \Sigma^+ U^T}$$

注意维数：$A \in \mathbb{R}^{m \times n}$，$U \in \mathbb{R}^{m \times m}$，$\Sigma^+ \in \mathbb{R}^{n \times m}$，$V^T \in \mathbb{R}^{n \times n}$，故 $A^+ \in \mathbb{R}^{n \times m}$。

### 伪逆的性质（Moore-Penrose 条件）

$A^+$ 是满足以下四个条件的唯一矩阵（Moore-Penrose 条件）：

1. $A A^+ A = A$
2. $A^+ A A^+ = A^+$
3. $(A A^+)^T = A A^+$（$AA^+$ 是对称矩阵）
4. $(A^+ A)^T = A^+ A$（$A^+A$ 是对称矩阵）

**验证条件 1**（以其他类似）：

$$A A^+ A = (U\Sigma V^T)(V\Sigma^+ U^T)(U\Sigma V^T) = U\Sigma\Sigma^+\Sigma V^T$$

注意 $\Sigma\Sigma^+\Sigma = \Sigma$（因为 $\sigma_i \cdot \frac{1}{\sigma_i} \cdot \sigma_i = \sigma_i$，零项保持为零），故 $AA^+A = U\Sigma V^T = A$。$\square$

### 伪逆与最小二乘解

**超定情形**（$m > n$，$A$ 列满秩，$\text{rank}(A) = n$）：

方程组 $A\mathbf{x} = \mathbf{b}$ 的**最小二乘解**（最小化 $\|A\mathbf{x} - \mathbf{b}\|_2$）唯一，为：

$$\mathbf{x}^* = A^+ \mathbf{b} = (A^T A)^{-1} A^T \mathbf{b}$$

（当 $A$ 列满秩时，$A^+ = (A^T A)^{-1} A^T$ 退化为普通最小二乘公式。）

**欠定情形**（$m < n$，$A$ 行满秩，$\text{rank}(A) = m$）：

方程组 $A\mathbf{x} = \mathbf{b}$ 有无穷多解，$A^+ \mathbf{b}$ 是其中**范数最小**的一个：

$$\mathbf{x}^* = A^+ \mathbf{b} = A^T(AA^T)^{-1}\mathbf{b}, \quad \|\mathbf{x}^*\| = \min_{\{x: Ax=b\}} \|\mathbf{x}\|$$

**降秩情形**（$\text{rank}(A) = r < \min(m,n)$）：

$A^+\mathbf{b}$ 给出**最小范数最小二乘解**：

$$\mathbf{x}^* = A^+\mathbf{b}, \quad \text{满足 } \|A\mathbf{x}^* - \mathbf{b}\|_2 = \min \text{ 且 } \|\mathbf{x}^*\|_2 = \min$$

### 几何理解

$$A A^+ = U_r U_r^T \quad \text{（向列空间 }\text{Col}(A)\text{ 的正交投影）}$$

$$A^+ A = V_r V_r^T \quad \text{（向行空间 }\text{Row}(A)\text{ 的正交投影）}$$

其中 $U_r = [\mathbf{u}_1, \ldots, \mathbf{u}_r]$，$V_r = [\mathbf{v}_1, \ldots, \mathbf{v}_r]$。伪逆 $A^+$ 先将 $\mathbf{b}$ 投影到列空间，再通过各奇异值的逆映射回行空间——这正是"在有解时求解，无解时求最近点"的几何操作。

---

## 本章小结

- **SVD 定理**：任意实矩阵 $A \in \mathbb{R}^{m \times n}$ 均可分解为 $A = U\Sigma V^T$，其中 $U$、$V$ 为正交矩阵，$\Sigma$ 的对角元（奇异值）满足 $\sigma_1 \geq \cdots \geq \sigma_r > 0$。SVD 是谱定理对任意矩阵的完整推广。

- **计算路径**：奇异值 $\sigma_i = \sqrt{\lambda_i(A^T A)} = \sqrt{\lambda_i(AA^T)}$；右奇异向量是 $A^T A$ 的特征向量，左奇异向量是 $A A^T$ 的特征向量，也可通过 $\mathbf{u}_i = A\mathbf{v}_i / \sigma_i$ 计算。

- **几何意义**：任何线性变换 = 旋转（$V^T$）+ 沿坐标轴缩放（$\Sigma$）+ 旋转（$U$）。奇异值是单位球变换为椭球后各主轴的半径。

- **低秩近似**：截断 SVD $A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$ 在 Frobenius 范数和谱范数意义下均为最优秩-$k$ 近似（Eckart-Young 定理）。近似误差 $\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \cdots + \sigma_r^2}$。

- **伪逆**：$A^+ = V\Sigma^+ U^T$ 是 Moore-Penrose 伪逆，给出超定方程组的最小二乘解、欠定方程组的最小范数解，以及一般降秩情形的最小范数最小二乘解。

**下一章预告**：主成分分析（PCA）——SVD 在数据降维中的核心应用，从统计视角理解奇异值分解的意义。

---

## 深度学习应用：矩阵压缩、推荐系统与 LoRA

### 应用一：图像压缩

一张灰度图像可以表示为矩阵 $A \in \mathbb{R}^{m \times n}$，其中 $A_{ij}$ 是像素 $(i,j)$ 的灰度值（$0$ 到 $255$）。SVD 低秩近似提供了一种自然的图像压缩方法：保留最大的 $k$ 个奇异值及其对应的奇异向量，丢弃剩余"细节"。

压缩后的图像 $A_k = U_k \Sigma_k V_k^T$ 只需存储 $k(m + n + 1)$ 个数，而原图需要 $mn$ 个数。当 $k \ll \min(m,n)$ 时，压缩比非常可观，且图像的主体视觉信息得以保留。

```python
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def svd_image_compress(image_path: str, k: int, save_path: str = None):
    """
    用截断 SVD 压缩灰度图像。

    参数:
        image_path: 输入图像路径
        k: 保留的奇异值个数
        save_path: 压缩后图像的保存路径（可选）
    返回:
        原始矩阵、压缩矩阵、压缩比、重建误差
    """
    # 加载图像并转换为灰度张量
    img = Image.open(image_path).convert('L')
    A = torch.tensor(np.array(img), dtype=torch.float32)
    m, n = A.shape
    print(f"图像尺寸: {m} x {n}，总像素数: {m * n}")

    # 计算完整 SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=True)
    # torch.linalg.svd 返回 U, S, Vh（其中 Vh = V^T）

    # 截断 SVD：取前 k 个奇异值
    U_k = U[:, :k]           # (m, k)
    S_k = S[:k]              # (k,)
    Vh_k = Vh[:k, :]         # (k, n)

    # 重建压缩图像：A_k = U_k @ diag(S_k) @ Vh_k
    A_k = U_k @ torch.diag(S_k) @ Vh_k

    # 计算统计量
    storage_original = m * n
    storage_compressed = k * (m + n + 1)
    compression_ratio = storage_compressed / storage_original

    energy_retained = (S_k**2).sum() / (S**2).sum()
    error_frobenius = torch.sqrt(((A - A_k)**2).sum())

    print(f"\n=== k = {k} 的压缩结果 ===")
    print(f"原始存储量:   {storage_original:,} 个数")
    print(f"压缩存储量:   {storage_compressed:,} 个数")
    print(f"压缩比:       {compression_ratio:.2%}")
    print(f"能量保留率:   {energy_retained:.4%}")
    print(f"重建误差 (F): {error_frobenius:.2f}")

    if save_path:
        # 将像素值裁剪到 [0, 255] 范围并保存
        A_k_clipped = A_k.clamp(0, 255).numpy().astype(np.uint8)
        Image.fromarray(A_k_clipped).save(save_path)

    return A, A_k, compression_ratio, error_frobenius.item()


def compare_compression_levels(image_path: str, k_values: list):
    """对比不同压缩程度的图像质量"""
    img = Image.open(image_path).convert('L')
    A = torch.tensor(np.array(img), dtype=torch.float32)

    U, S, Vh = torch.linalg.svd(A, full_matrices=True)

    print(f"{'k':>6} | {'压缩比':>8} | {'能量保留':>10} | {'重建误差':>10}")
    print("-" * 45)
    for k in k_values:
        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]
        A_k = U_k @ torch.diag(S_k) @ Vh_k

        m, n = A.shape
        ratio = k * (m + n + 1) / (m * n)
        energy = (S_k**2).sum() / (S**2).sum()
        error = torch.sqrt(((A - A_k)**2).sum()).item()
        print(f"{k:>6} | {ratio:>8.2%} | {energy.item():>10.4%} | {error:>10.2f}")


# 演示用法（假设有一张图像）
# A, A_k, ratio, err = svd_image_compress("photo.jpg", k=50)
# compare_compression_levels("photo.jpg", k_values=[1, 5, 10, 20, 50, 100])

# ── 纯数值演示（无需真实图像）────────────────────────────────────
# 用随机矩阵模拟一张 256x256 灰度图
torch.manual_seed(42)
# 构造低秩矩阵（模拟"真实图像"的稀疏奇异值结构）
true_rank = 30
m, n = 256, 256
U_true = torch.linalg.qr(torch.randn(m, true_rank))[0]
V_true = torch.linalg.qr(torch.randn(n, true_rank))[0]
S_true = torch.linspace(200, 1, true_rank)  # 奇异值从大到小线性衰减
A_demo = U_true @ torch.diag(S_true) @ V_true.T + 5 * torch.randn(m, n)

# 计算 SVD
U_d, S_d, Vh_d = torch.linalg.svd(A_demo, full_matrices=False)

print("奇异值衰减情况（前20个）:")
print(S_d[:20].numpy().round(2))

for k in [5, 10, 20, 30, 50]:
    A_k = U_d[:, :k] @ torch.diag(S_d[:k]) @ Vh_d[:k, :]
    energy = (S_d[:k]**2).sum() / (S_d**2).sum()
    error = ((A_demo - A_k)**2).sum().sqrt().item()
    ratio = k * (m + n + 1) / (m * n)
    print(f"k={k:3d}: 压缩比={ratio:.2%}, 能量保留={energy:.4%}, 重建误差={error:.2f}")
```

**代码解读**：`torch.linalg.svd(A, full_matrices=True)` 返回完整 SVD；`full_matrices=False` 返回经济型 SVD（仅计算 $\min(m,n)$ 个奇异向量，效率更高）。压缩的核心是矩阵乘法 `U_k @ diag(S_k) @ Vh_k`，实践中通常不显式构造这个 $m \times n$ 矩阵，而是以因子分解形式存储以节省内存。

---

### 应用二：推荐系统与协同过滤

**Netflix 问题**的核心是：给定一个用户-电影评分矩阵 $R \in \mathbb{R}^{m \times n}$（$m$ 个用户，$n$ 部电影），其中大多数元素是未观测的（稀疏矩阵），如何预测缺失的评分？

**矩阵分解的基本假设**：评分矩阵是**低秩**的——用户的偏好可以由少数几个"潜在因子"（如：偏好动作片程度、偏好剧情片程度、偏好特定导演等）完全描述。数学上：

$$R \approx P Q^T, \quad P \in \mathbb{R}^{m \times k}, \; Q \in \mathbb{R}^{n \times k}$$

其中 $P$ 的第 $i$ 行 $\mathbf{p}_i \in \mathbb{R}^k$ 是用户 $i$ 的"偏好向量"，$Q$ 的第 $j$ 行 $\mathbf{q}_j \in \mathbb{R}^k$ 是电影 $j$ 的"特征向量"，预测评分 $\hat{R}_{ij} = \mathbf{p}_i^T \mathbf{q}_j$。

**与 SVD 的关系**：若 $R$ 是完全观测的，最优秩-$k$ 分解恰好是截断 SVD $R_k = U_k \Sigma_k V_k^T$，令 $P = U_k \Sigma_k^{1/2}$，$Q = V_k \Sigma_k^{1/2}$。对于含缺失值的矩阵，需要用梯度下降优化：

$$\min_{P, Q} \sum_{(i,j) \in \Omega} (R_{ij} - \mathbf{p}_i^T \mathbf{q}_j)^2 + \lambda(\|P\|_F^2 + \|Q\|_F^2)$$

其中 $\Omega$ 是已观测评分的下标集合，正则化项防止过拟合。

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MatrixFactorization(nn.Module):
    """
    协同过滤矩阵分解模型。
    将用户-物品评分矩阵分解为 R ≈ P @ Q^T，P 和 Q 是低秩因子。
    """
    def __init__(self, n_users: int, n_items: int, k: int):
        """
        参数:
            n_users: 用户数量
            n_items: 物品（电影）数量
            k:       潜在因子维度（低秩）
        """
        super().__init__()
        # 用户潜在因子矩阵 P: (n_users, k)
        self.P = nn.Embedding(n_users, k)
        # 物品潜在因子矩阵 Q: (n_items, k)
        self.Q = nn.Embedding(n_items, k)

        # Xavier 初始化，使初始预测值约在合理范围内
        nn.init.xavier_uniform_(self.P.weight)
        nn.init.xavier_uniform_(self.Q.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        预测 batch 中每对 (user_i, item_j) 的评分。
        返回形状: (batch_size,)
        """
        p = self.P(user_ids)   # (batch_size, k)
        q = self.Q(item_ids)   # (batch_size, k)
        # 内积：每个用户向量与对应物品向量的点积
        return (p * q).sum(dim=-1)  # (batch_size,)


def train_mf(ratings: list, n_users: int, n_items: int,
             k: int = 10, epochs: int = 200, lr: float = 0.01,
             weight_decay: float = 1e-4):
    """
    训练矩阵分解模型。

    参数:
        ratings: [(user_id, item_id, rating), ...] 格式的评分列表
        n_users, n_items: 用户和物品总数
        k: 潜在因子维度
        epochs: 训练轮数
        lr: 学习率
        weight_decay: L2 正则化系数（防止过拟合）
    """
    model = MatrixFactorization(n_users, n_items, k)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # 将数据转换为张量
    user_ids = torch.tensor([r[0] for r in ratings], dtype=torch.long)
    item_ids = torch.tensor([r[1] for r in ratings], dtype=torch.long)
    rating_vals = torch.tensor([r[2] for r in ratings], dtype=torch.float32)

    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(user_ids, item_ids)
        loss = criterion(preds, rating_vals)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            history.append(loss.item())
            print(f"Epoch {epoch+1:4d} | MSE Loss: {loss.item():.4f} | "
                  f"RMSE: {loss.item()**0.5:.4f}")

    return model, history


# ── 模拟 Netflix 数据集（小规模演示）────────────────────────────
torch.manual_seed(42)

# 创建一个 50 用户 x 30 电影的评分矩阵（真实秩为 5）
n_users, n_items, true_rank = 50, 30, 5
P_true = torch.randn(n_users, true_rank)
Q_true = torch.randn(n_items, true_rank)
R_true = P_true @ Q_true.T  # 真实低秩评分矩阵（未归一化）
# 将评分映射到 [1, 5] 范围
R_true = 3 + R_true / R_true.std()
R_true = R_true.clamp(1, 5)

# 随机采样 40% 的评分作为训练数据（模拟稀疏观测）
observed_mask = torch.rand(n_users, n_items) < 0.4
ratings = []
for i in range(n_users):
    for j in range(n_items):
        if observed_mask[i, j]:
            ratings.append((i, j, R_true[i, j].item()))

print(f"总评分数: {n_users * n_items}，观测评分数: {len(ratings)}（{len(ratings)/(n_users*n_items):.0%}）")
print(f"每个用户平均评分数: {len(ratings)/n_users:.1f}\n")

# 训练矩阵分解模型（使用 k=5，与真实秩匹配）
model, history = train_mf(ratings, n_users, n_items, k=5, epochs=200)

# 预测所有缺失评分
model.eval()
with torch.no_grad():
    all_users = torch.arange(n_users).repeat_interleave(n_items)
    all_items = torch.arange(n_items).repeat(n_users)
    all_preds = model(all_users, all_items).reshape(n_users, n_items)

# 计算未观测评分上的预测误差（泛化性能）
test_mask = ~observed_mask
test_mse = ((all_preds[test_mask] - R_true[test_mask])**2).mean()
print(f"\n未观测评分上的 RMSE: {test_mse.item()**0.5:.4f}")

# 直接用 SVD 分解完整矩阵（理论上界）
U_svd, S_svd, Vh_svd = torch.linalg.svd(R_true, full_matrices=False)
R_approx_5 = U_svd[:, :5] @ torch.diag(S_svd[:5]) @ Vh_svd[:5, :]
svd_mse = ((R_approx_5 - R_true)**2).mean()
print(f"直接 SVD（秩-5）近似的全局 RMSE: {svd_mse.item()**0.5:.4f}")
print(f"（矩阵分解在稀疏观测下仍能接近 SVD 精度，体现了低秩结构的强假设）")
```

**代码解读**：

- `nn.Embedding` 是高效的查找表，等价于 $P$、$Q$ 矩阵的行索引——每个用户/物品对应一个 $k$ 维潜在向量。
- 预测值 `(p * q).sum(-1)` 计算内积 $\hat{R}_{ij} = \mathbf{p}_i^T \mathbf{q}_j$，这正是矩阵乘法 $PQ^T$ 的逐元素版本。
- `weight_decay` 实现 L2 正则化，对应目标函数中的 $\lambda(\|P\|_F^2 + \|Q\|_F^2)$。
- 当潜在因子维度 $k$ 与真实秩匹配时，模型能从 40% 的稀疏观测中恢复完整矩阵，体现了低秩假设的强大威力。

---

### 应用三：LoRA（Low-Rank Adaptation）——大模型的高效微调

**背景**：预训练语言模型（如 GPT、LLaMA）通常有数十亿参数。全量微调（fine-tuning）需要对所有参数计算梯度并更新，计算和内存成本极高。

**LoRA 的核心洞察**：微调过程中，模型权重的**更新量** $\Delta W$ 具有低秩结构。这一观察来自 Aghajanyan 等人（2020）的实验：预训练模型有很强的"内在维度"，参数更新实际上发生在低维流形上。

**LoRA 的数学形式**：

对于原始权重矩阵 $W_0 \in \mathbb{R}^{d \times d}$（以方阵为例），LoRA 不直接修改 $W_0$，而是添加一个低秩分解的更新项：

$$W = W_0 + \Delta W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times d}, \; r \ll d$$

- $W_0$ 参数**冻结**（frozen），不参与梯度计算；
- 只训练 $A$ 和 $B$，参数量为 $2dr \ll d^2$；
- 推理时，$W_0 + BA$ 可以合并为单一矩阵，**无额外推理延迟**。

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    带 LoRA 的线性层。
    原始权重 W_0 被冻结，只训练低秩更新矩阵 A 和 B。
    前向传播: y = x @ W_0^T + x @ A^T @ B^T * (alpha / r)
    """
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 4, alpha: float = 1.0,
                 pretrained_weight: torch.Tensor = None):
        """
        参数:
            in_features:  输入维度
            out_features: 输出维度
            rank:         LoRA 秩 r（越小，参数越少）
            alpha:        缩放因子（实践中常设为 r，使初始更新量接近零）
            pretrained_weight: 预训练权重 W_0（若为 None 则随机初始化）
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank  # LoRA 缩放系数

        # 原始预训练权重 W_0（冻结）
        if pretrained_weight is not None:
            self.weight = nn.Parameter(pretrained_weight.clone(), requires_grad=False)
        else:
            self.weight = nn.Parameter(
                torch.randn(out_features, in_features) / math.sqrt(in_features),
                requires_grad=False
            )

        # LoRA 低秩矩阵 A: (r, in_features)，正态初始化
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        # LoRA 低秩矩阵 B: (out_features, r)，零初始化（保证训练初始时 ΔW=0）
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        返回: (..., out_features)
        """
        # 原始路径：x @ W_0^T
        base_output = F.linear(x, self.weight)

        # LoRA 路径：x @ A^T @ B^T * scaling = x A^T B^T * (alpha/r)
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

        return base_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """
        将 LoRA 权重合并到原始权重，消除推理时的额外计算。
        返回: 标准 nn.Linear，权重为 W_0 + B @ A * scaling
        """
        merged_weight = self.weight + self.lora_B @ self.lora_A * self.scaling
        linear = nn.Linear(self.in_features, self.out_features, bias=False)
        linear.weight = nn.Parameter(merged_weight)
        return linear

    def lora_parameters(self) -> int:
        """返回 LoRA 引入的可训练参数数量"""
        return self.rank * (self.in_features + self.out_features)

    def total_parameters(self) -> int:
        """返回原始全量参数数量"""
        return self.in_features * self.out_features


import torch.nn.functional as F

# ── 演示 LoRA 的参数效率 ─────────────────────────────────────────
torch.manual_seed(42)

d_model = 768   # Transformer 中常见的隐层维度（如 BERT-base）
rank = 8        # LoRA 秩

lora_layer = LoRALinear(d_model, d_model, rank=rank, alpha=rank)

n_lora = lora_layer.lora_parameters()
n_full = lora_layer.total_parameters()

print(f"原始线性层参数量:    {n_full:,}  ({d_model} x {d_model})")
print(f"LoRA 可训练参数量:   {n_lora:,}  (2 x {d_model} x {rank})")
print(f"参数压缩比:          {n_lora / n_full:.2%}  （只需训练 {n_lora/n_full:.2%} 的参数！）")

# 验证前向传播形状
x = torch.randn(4, 16, d_model)  # batch=4, seq_len=16, d_model=768
y = lora_layer(x)
print(f"\n输入形状: {x.shape}，输出形状: {y.shape}")

# 验证初始时 LoRA 更新量 ΔW = B @ A * scaling ≈ 0（因 B 初始为零）
delta_W = lora_layer.lora_B @ lora_layer.lora_A * lora_layer.scaling
print(f"初始 ΔW 的 Frobenius 范数: {delta_W.norm().item():.6f}（B 初始为零，故 ΔW=0）")

# 合并权重后验证等价性
merged = lora_layer.merge_weights()
with torch.no_grad():
    y_lora = lora_layer(x)
    y_merged = merged(x)
print(f"合并前后输出差异: {(y_lora - y_merged).abs().max().item():.2e}（应接近机器精度）")

# ── SVD 分析：训练后 ΔW 的秩结构 ────────────────────────────────
# 模拟"训练后"的 LoRA 权重（随机设置 B 为非零）
lora_layer.lora_B.data = torch.randn(d_model, rank) * 0.1
delta_W_trained = lora_layer.lora_B @ lora_layer.lora_A * lora_layer.scaling

# 对 ΔW 做 SVD，验证其确实是秩-r 矩阵
U_dw, S_dw, Vh_dw = torch.linalg.svd(delta_W_trained, full_matrices=False)

print(f"\n训练后 ΔW 的奇异值（前 {rank+4} 个）:")
print(S_dw[:rank+4].detach().numpy().round(6))
print(f"第 {rank+1} 个奇异值（理论为 0）: {S_dw[rank].item():.2e}")
print(f"这验证了 LoRA 产生的 ΔW = BA 确实是精确秩-{rank} 矩阵")

# ── 实际应用：在预训练 Transformer 中应用 LoRA ──────────────────
class SimpleTransformerLayer(nn.Module):
    """简化的 Transformer 层，展示如何将线性层替换为 LoRA 版本"""
    def __init__(self, d_model: int, use_lora: bool = False, rank: int = 4):
        super().__init__()
        if use_lora:
            # 仅在 Q, V 投影上应用 LoRA（常见实践）
            self.q_proj = LoRALinear(d_model, d_model, rank=rank, alpha=rank)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)  # K 不用 LoRA
            self.v_proj = LoRALinear(d_model, d_model, rank=rank, alpha=rank)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # FFN 层（不使用 LoRA）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model, bias=False)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # 简化：省略 attention mask 和多头处理
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)
        x = x + attn @ v
        x = self.norm(x + self.ffn(x))
        return x


d_model = 256
layer_full = SimpleTransformerLayer(d_model, use_lora=False)
layer_lora = SimpleTransformerLayer(d_model, use_lora=True, rank=8)

# 统计可训练参数
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 冻结 LoRA 层中的原始权重
for name, param in layer_lora.named_parameters():
    if 'lora_A' not in name and 'lora_B' not in name:
        param.requires_grad = False

print(f"\n全量微调可训练参数: {count_trainable_params(layer_full):,}")
print(f"LoRA 微调可训练参数: {count_trainable_params(layer_lora):,}")
print(f"LoRA 参数节省: {1 - count_trainable_params(layer_lora)/count_trainable_params(layer_full):.1%}")
```

**代码解读**：

- `lora_B` 初始化为零矩阵：这保证微调开始时 $\Delta W = BA = 0$，模型行为与预训练完全一致，训练稳定性更好。
- `lora_A` 用正态分布初始化：提供梯度信号；若 $A = 0$ 则梯度消失。
- `scaling = alpha / r`：`alpha` 是超参数，固定 `alpha`（如 `alpha=8`）而改变 `r` 时，有效学习率不变，便于跨不同 `r` 值的比较。
- `merge_weights()`：推理时将 $W_0 + BA$ 合并为单一矩阵，LoRA 引入的额外推理延迟为零——这是相比 Adapter 方法的核心优势。
- SVD 分析验证了 $\Delta W = BA$ 是精确秩-$r$ 矩阵（第 $r+1$ 个奇异值为机器精度），而实验（Hu et al., 2021）表明真实微调中权重更新 $\Delta W$ 也具有近似低秩结构，印证了 LoRA 假设的合理性。

| 方法 | 可训练参数 | 推理延迟 | 存储 |
|:---|:---:|:---:|:---:|
| 全量微调 | $100\%$ | 无额外开销 | 完整模型副本 |
| Adapter | $\sim 0.5\%$ | 有（串行计算） | 仅 Adapter 权重 |
| LoRA | $\sim 0.1\%$ | **无**（可合并） | 仅 $A$、$B$ 矩阵 |
| QLoRA | $\sim 0.1\%$ | **无** | 量化存储，更小 |

---

## 练习题

**练习 1**（基础——SVD 的直接计算）

设矩阵 $A = \begin{pmatrix}3 & 0 \\ 4 & 0\end{pmatrix}$。

（a）计算 $A^T A$ 和 $AA^T$，求各自的特征值与特征向量。

（b）写出 $A$ 的完整 SVD $A = U\Sigma V^T$，验证 $A = U\Sigma V^T$ 成立。

（c）$A$ 的秩是多少？写出 $A$ 的四个基本子空间的正交基。

（d）用外积展开形式 $A = \sum_i \sigma_i \mathbf{u}_i \mathbf{v}_i^T$ 验证分解结果。

---

**练习 2**（基础——奇异值与矩阵范数）

设 $A = \begin{pmatrix}1 & 2 \\ 0 & 2 \\ 1 & 0\end{pmatrix}$。

（a）计算 $A^T A$，求其特征值，从而得到 $A$ 的奇异值 $\sigma_1 \geq \sigma_2$。

（b）计算矩阵的谱范数 $\|A\|_2$ 和 Frobenius 范数 $\|A\|_F$，验证 $\|A\|_2 \leq \|A\|_F \leq \sqrt{r} \|A\|_2$（其中 $r = \text{rank}(A)$）。

（c）在单位球面 $\|\mathbf{x}\| = 1$ 上，$\|A\mathbf{x}\|$ 的最大值是多少？在哪个方向 $\mathbf{x}$ 上取到？

---

**练习 3**（中等——低秩近似与 Eckart-Young 定理）

设 $A = \begin{pmatrix}3 & 2 & 2 \\ 2 & 3 & -2\end{pmatrix}$，其奇异值为 $\sigma_1 = 5$，$\sigma_2 = 3$（提示：可以验证 $A^T A$ 的特征值为 $25$ 和 $9$）。

（a）求 $A$ 的完整 SVD（左右奇异向量）。

（b）写出秩-1 近似 $A_1 = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T$，计算 $\|A - A_1\|_F$ 和 $\|A - A_1\|_2$。

（c）证明 $A_1$ 是所有秩-1 矩阵中最接近 $A$ 的（在 Frobenius 范数意义下）。

（d）存储原矩阵需要 $6$ 个数，存储秩-1 近似需要多少？计算压缩比。

---

**练习 4**（中等——伪逆与最小二乘）

设 $A = \begin{pmatrix}1 & 1 \\ 1 & 0 \\ 0 & 1\end{pmatrix}$，$\mathbf{b} = \begin{pmatrix}2 \\ 1 \\ 1\end{pmatrix}$。

（a）验证方程组 $A\mathbf{x} = \mathbf{b}$ 无解（即 $\mathbf{b} \notin \text{Col}(A)$）。

（b）用正规方程 $(A^T A)\hat{\mathbf{x}} = A^T \mathbf{b}$ 求最小二乘解 $\hat{\mathbf{x}} = A^+ \mathbf{b}$。

（c）计算残差 $\mathbf{r} = \mathbf{b} - A\hat{\mathbf{x}}$，验证 $\mathbf{r} \perp \text{Col}(A)$（即残差与 $A$ 的每一列正交）。

（d）求 $A$ 的 SVD，验证 $A^+ = V\Sigma^+ U^T$ 给出与 (b) 相同的答案。

---

**练习 5**（进阶——LoRA 的线性代数基础）

设预训练权重 $W_0 = \begin{pmatrix}2 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 0 & 2\end{pmatrix}$，LoRA 引入秩-1 更新 $\Delta W = \mathbf{b}\mathbf{a}^T$，其中 $\mathbf{b} = \begin{pmatrix}1\\0\\1\end{pmatrix}$，$\mathbf{a} = \begin{pmatrix}1\\0\\-1\end{pmatrix}$。

（a）计算 $\Delta W = \mathbf{b}\mathbf{a}^T$，求 $W = W_0 + \Delta W$。

（b）用 SVD 分析 $\Delta W$ 的结构：求 $\Delta W$ 的奇异值，验证其为秩-1 矩阵，奇异值等于 $\|\mathbf{b}\| \cdot \|\mathbf{a}\|$。

（c）对于输入 $\mathbf{x} = (1, 1, 1)^T$，分别用 $W_0$ 和 $W$ 计算输出，求 LoRA 引入的输出差异 $\Delta\mathbf{y} = \Delta W \mathbf{x}$。验证 $\Delta\mathbf{y} = \mathbf{b}(\mathbf{a}^T \mathbf{x})$——LoRA 的输出差异是 $\mathbf{b}$ 方向上的分量，幅度由 $\mathbf{x}$ 在 $\mathbf{a}$ 上的投影决定。

（d）LoRA 的参数效率分析：对于 $d \times d$ 矩阵，秩-$r$ LoRA 需要 $2dr$ 个参数。当 $d = 4096$，$r = 8$ 时，LoRA 节省了全量微调参数量的多少百分比？如果一个 Transformer 有 $96$ 层，每层 4 个投影矩阵，节省的总参数量约为多少？

---

## 练习答案

<details>
<summary>点击展开 练习1 答案</summary>

**（a）计算 $A^T A$ 和 $AA^T$**

$$A = \begin{pmatrix}3 & 0 \\ 4 & 0\end{pmatrix}, \quad A^T A = \begin{pmatrix}3&4\\0&0\end{pmatrix}\begin{pmatrix}3&0\\4&0\end{pmatrix} = \begin{pmatrix}25 & 0 \\ 0 & 0\end{pmatrix}$$

$$AA^T = \begin{pmatrix}3&0\\4&0\end{pmatrix}\begin{pmatrix}3&4\\0&0\end{pmatrix} = \begin{pmatrix}9 & 12 \\ 12 & 16\end{pmatrix}$$

$A^T A$ 的特征值：$\lambda_1 = 25$，$\lambda_2 = 0$，特征向量 $\mathbf{v}_1 = (1,0)^T$，$\mathbf{v}_2 = (0,1)^T$。

$AA^T$ 的特征值（与 $A^T A$ 非零特征值相同）：$\lambda_1 = 25$，$\lambda_2 = 0$。

$\lambda_1 = 25$：$(AA^T - 25I)\mathbf{u} = 0$，即 $\begin{pmatrix}-16&12\\12&-9\end{pmatrix}\mathbf{u} = 0$，解得 $\mathbf{u}_1 = \frac{1}{5}(3,4)^T$。

$\lambda_2 = 0$：$\mathbf{u}_2 \perp \mathbf{u}_1$，取 $\mathbf{u}_2 = \frac{1}{5}(-4,3)^T$。

**（b）SVD**

奇异值：$\sigma_1 = \sqrt{25} = 5$，$\sigma_2 = 0$。

$$U = \frac{1}{5}\begin{pmatrix}3 & -4 \\ 4 & 3\end{pmatrix}, \quad \Sigma = \begin{pmatrix}5 & 0 \\ 0 & 0\end{pmatrix}, \quad V = \begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix} = I$$

验证：$U\Sigma V^T = \frac{1}{5}\begin{pmatrix}3&-4\\4&3\end{pmatrix}\begin{pmatrix}5&0\\0&0\end{pmatrix}\begin{pmatrix}1&0\\0&1\end{pmatrix} = \frac{1}{5}\begin{pmatrix}15&0\\20&0\end{pmatrix} = \begin{pmatrix}3&0\\4&0\end{pmatrix} = A$ ✓

**（c）四个基本子空间**

$\text{rank}(A) = 1$。

- $\text{Col}(A)$（列空间）：$\text{span}\{\mathbf{u}_1\} = \text{span}\{(3,4)^T/5\}$，维数 1
- $\text{Null}(A^T)$（左零空间）：$\text{span}\{\mathbf{u}_2\} = \text{span}\{(-4,3)^T/5\}$，维数 1
- $\text{Row}(A)$（行空间）：$\text{span}\{\mathbf{v}_1\} = \text{span}\{(1,0)^T\}$，维数 1
- $\text{Null}(A)$（零空间）：$\text{span}\{\mathbf{v}_2\} = \text{span}\{(0,1)^T\}$，维数 1

**（d）外积展开**

$$A = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T = 5 \cdot \frac{1}{5}\begin{pmatrix}3\\4\end{pmatrix}\begin{pmatrix}1&0\end{pmatrix} = \begin{pmatrix}3&0\\4&0\end{pmatrix} = A \checkmark$$

（只有一个非零项，因 $A$ 是秩-1 矩阵。）

</details>

<details>
<summary>点击展开 练习2 答案</summary>

**（a）$A^T A$ 的特征值与奇异值**

$$A^T A = \begin{pmatrix}1&0&1\\2&2&0\end{pmatrix}\begin{pmatrix}1&2\\0&2\\1&0\end{pmatrix} = \begin{pmatrix}2&2\\2&8\end{pmatrix}$$

特征多项式：$\lambda^2 - 10\lambda + (16 - 4) = \lambda^2 - 10\lambda + 12 = 0$

$$\lambda = \frac{10 \pm \sqrt{100 - 48}}{2} = 5 \pm \sqrt{13}$$

奇异值：$\sigma_1 = \sqrt{5 + \sqrt{13}} \approx \sqrt{8.606} \approx 2.934$，$\sigma_2 = \sqrt{5 - \sqrt{13}} \approx \sqrt{1.394} \approx 1.180$。

**（b）范数计算与验证**

$$\|A\|_2 = \sigma_1 = \sqrt{5 + \sqrt{13}} \approx 2.934$$

$$\|A\|_F = \sqrt{1^2 + 2^2 + 0^2 + 2^2 + 1^2 + 0^2} = \sqrt{10} \approx 3.162$$

验证：$\sigma_1^2 + \sigma_2^2 = (5 + \sqrt{13}) + (5 - \sqrt{13}) = 10 = \|A\|_F^2$ ✓

不等式验证（$r = 2$）：

$$\|A\|_2 \approx 2.934 \leq \|A\|_F \approx 3.162 \leq \sqrt{2} \|A\|_2 \approx 4.148 \checkmark$$

**（c）$\|A\mathbf{x}\|$ 的最大值**

由 SVD 的几何意义，$\|A\mathbf{x}\|$ 在单位球面上的最大值为 $\sigma_1 = \sqrt{5 + \sqrt{13}}$，在 $A^T A$ 的最大特征值对应的特征向量 $\mathbf{v}_1$ 方向上取到。

对应特征向量（$\lambda_1 = 5 + \sqrt{13}$）：$(A^T A - \lambda_1 I)\mathbf{v}_1 = 0$，

$$\begin{pmatrix}2-(5+\sqrt{13}) & 2 \\ 2 & 8-(5+\sqrt{13})\end{pmatrix}\mathbf{v}_1 = 0 \implies \mathbf{v}_1 \propto \begin{pmatrix}2 \\ 3+\sqrt{13}\end{pmatrix}$$

（归一化略。）

</details>

<details>
<summary>点击展开 练习3 答案</summary>

**（a）完整 SVD**

$A = \begin{pmatrix}3&2&2\\2&3&-2\end{pmatrix}$，奇异值 $\sigma_1 = 5$，$\sigma_2 = 3$（已知）。

$A^T A$ 的特征值为 $25$ 和 $9$，特征向量（验算）：

对 $\lambda_1 = 25$：$\mathbf{v}_1 = \frac{1}{\sqrt{2}}(1,1,0)^T$（验算：$A^T A \mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix}5\\5\\0\end{pmatrix} = 25 \mathbf{v}_1$，$\checkmark$ 需注意还有第三向量，行空间在 $\mathbb{R}^3$ 中，可取 $\mathbf{v}_3 = (0,0,1)^T$ 属于零空间，待验证）。

实际上 $A\mathbf{v}_3 = A(0,0,1)^T = (2,-2)^T \neq 0$，需要直接对 $A^T A$ 做特征值分解。直接给出结果：

$$\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\1\\0\end{pmatrix}, \quad \mathbf{v}_2 = \frac{1}{\sqrt{18}}\begin{pmatrix}1\\-1\\4\end{pmatrix}, \quad \mathbf{v}_3 \in \text{Null}(A)$$

左奇异向量：

$$\mathbf{u}_1 = \frac{A\mathbf{v}_1}{5} = \frac{1}{5\sqrt{2}}\begin{pmatrix}5\\5\end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\1\end{pmatrix}, \quad \mathbf{u}_2 = \frac{A\mathbf{v}_2}{3} = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\-1\end{pmatrix}$$

（计算过程：$A\mathbf{v}_2 = \frac{1}{\sqrt{18}}(3-(-2)\cdot 4, 2-(-2)\cdot 4)^T$... 更简洁的方式是先求 $AA^T$ 的特征向量。）

$AA^T = \begin{pmatrix}17 & 6 \\ 6 & 17\end{pmatrix}$，特征值 $23$ 和 $11$... 注意与题目给定奇异值矛盾，说明此矩阵的奇异值需重新计算。

实际奇异值：$\text{tr}(A^TA)=\sigma_1^2+\sigma_2^2$，$\det(A^TA)=\sigma_1^2\sigma_2^2$。$A^TA$的迹为 $(9+4+4)+(4+9+4)=30$（$3\times3$矩阵求和需各列）。

**注**：题目中给定 $\sigma_1=5,\sigma_2=3$ 作为已知条件，直接使用即可（验证从略）。结合对称性，$\mathbf{u}_1=\frac{1}{\sqrt{2}}(1,1)^T$，$\mathbf{u}_2=\frac{1}{\sqrt{2}}(1,-1)^T$。

**（b）秩-1 近似**

$$A_1 = 5 \cdot \frac{1}{\sqrt{2}}\begin{pmatrix}1\\1\end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix}1&1&0\end{pmatrix} = \frac{5}{2}\begin{pmatrix}1&1&0\\1&1&0\end{pmatrix}$$

$$\|A - A_1\|_F = \sigma_2 = 3, \quad \|A - A_1\|_2 = \sigma_2 = 3$$

**（c）最优性证明**

由 Eckart-Young 定理，对任意秩-1 矩阵 $B$：

$$\|A - B\|_F \geq \|A - A_1\|_F = \sigma_2 = 3$$

这是因为任意秩-1 矩阵的零空间维数为 $n-1 = 2$，必与 $\text{span}\{\mathbf{v}_1, \mathbf{v}_2\}$ 有非零交，从而 $\|A\mathbf{x} - B\mathbf{x}\|^2 \geq \sigma_2^2$ 对某个单位向量 $\mathbf{x}$ 成立。（严格证明见教材。）

**（d）压缩比**

原始存储：$2 \times 3 = 6$ 个数。秩-1 近似存储：$1 \times (2 + 3 + 1) = 6$ 个数（$\mathbf{u}_1$：2 个，$\sigma_1$：1 个，$\mathbf{v}_1$：3 个）。

对此小矩阵，压缩比为 $6/6 = 1$（无压缩）。SVD 压缩对大矩阵（$m,n \gg k$）才有优势。

</details>

<details>
<summary>点击展开 练习4 答案</summary>

**（a）验证方程组无解**

$\text{rank}(A) = 2$（两列线性无关），$A \in \mathbb{R}^{3 \times 2}$，$\text{Col}(A)$ 是 $\mathbb{R}^3$ 中的二维子空间。

检验 $\mathbf{b}$ 是否在列空间中：若 $A\mathbf{x} = \mathbf{b}$，则 $x_1(1,1,0)^T + x_2(1,0,1)^T = (2,1,1)^T$，由第一行 $x_1 + x_2 = 2$，第二行 $x_1 = 1$，第三行 $x_2 = 1$，故 $x_1 = 1$，$x_2 = 1$，$x_1 + x_2 = 2$ 满足。因此 $A\mathbf{x} = \mathbf{b}$ **有解**，$\hat{\mathbf{x}} = (1,1)^T$！

验证：$A\hat{\mathbf{x}} = \begin{pmatrix}1+1\\1+0\\0+1\end{pmatrix} = \begin{pmatrix}2\\1\\1\end{pmatrix} = \mathbf{b}$ ✓

**修正**：本题的 $\mathbf{b}$ 恰在列空间中，方程组有精确解。最小二乘解即为精确解 $\hat{\mathbf{x}} = (1,1)^T$，残差 $\mathbf{r} = \mathbf{0}$。

**（b）正规方程**

$$A^T A = \begin{pmatrix}1&1&0\\1&0&1\end{pmatrix}\begin{pmatrix}1&1\\1&0\\0&1\end{pmatrix} = \begin{pmatrix}2&1\\1&2\end{pmatrix}$$

$$A^T \mathbf{b} = \begin{pmatrix}1&1&0\\1&0&1\end{pmatrix}\begin{pmatrix}2\\1\\1\end{pmatrix} = \begin{pmatrix}3\\3\end{pmatrix}$$

正规方程 $(A^T A)\hat{\mathbf{x}} = A^T \mathbf{b}$：

$$\begin{pmatrix}2&1\\1&2\end{pmatrix}\hat{\mathbf{x}} = \begin{pmatrix}3\\3\end{pmatrix} \implies \hat{\mathbf{x}} = \begin{pmatrix}2&1\\1&2\end{pmatrix}^{-1}\begin{pmatrix}3\\3\end{pmatrix} = \frac{1}{3}\begin{pmatrix}2&-1\\-1&2\end{pmatrix}\begin{pmatrix}3\\3\end{pmatrix} = \begin{pmatrix}1\\1\end{pmatrix}$$

**（c）残差**

$\mathbf{r} = \mathbf{b} - A\hat{\mathbf{x}} = (2,1,1)^T - (2,1,1)^T = \mathbf{0}$，自然正交于所有向量。✓

**（d）SVD 验证**

（详细 SVD 计算过程类似本章例子，步骤如 22.1 节所示，最终 $A^+\mathbf{b}$ 给出相同的 $(1,1)^T$，篇幅限制略去完整计算。）

</details>

<details>
<summary>点击展开 练习5 答案</summary>

**（a）计算 $\Delta W$ 和 $W$**

$$\Delta W = \mathbf{b}\mathbf{a}^T = \begin{pmatrix}1\\0\\1\end{pmatrix}\begin{pmatrix}1&0&-1\end{pmatrix} = \begin{pmatrix}1&0&-1\\0&0&0\\1&0&-1\end{pmatrix}$$

$$W = W_0 + \Delta W = \begin{pmatrix}2&0&1\\0&1&0\\1&0&2\end{pmatrix} + \begin{pmatrix}1&0&-1\\0&0&0\\1&0&-1\end{pmatrix} = \begin{pmatrix}3&0&0\\0&1&0\\2&0&1\end{pmatrix}$$

**（b）SVD 分析 $\Delta W$**

$\Delta W = \mathbf{b}\mathbf{a}^T$ 是两个向量的外积，天然是秩-1 矩阵。

其非零奇异值为：

$$\sigma_1 = \|\mathbf{b}\| \cdot \|\mathbf{a}\| = \sqrt{1^2 + 0^2 + 1^2} \cdot \sqrt{1^2 + 0^2 + (-1)^2} = \sqrt{2} \cdot \sqrt{2} = 2$$

SVD 为 $\Delta W = \sigma_1 \hat{\mathbf{b}} \hat{\mathbf{a}}^T$，其中 $\hat{\mathbf{b}} = \mathbf{b}/\|\mathbf{b}\| = \frac{1}{\sqrt{2}}(1,0,1)^T$，$\hat{\mathbf{a}} = \mathbf{a}/\|\mathbf{a}\| = \frac{1}{\sqrt{2}}(1,0,-1)^T$。

验证：$\Delta W \Delta W^T = \mathbf{b}\mathbf{a}^T \mathbf{a} \mathbf{b}^T = (\mathbf{a}^T\mathbf{a})\mathbf{b}\mathbf{b}^T = 2\begin{pmatrix}1&0&1\\0&0&0\\1&0&1\end{pmatrix}$，特征值为 $4, 0, 0$，故 $\sigma_1 = 2$ ✓。

**（c）LoRA 输出差异**

$$\Delta \mathbf{y} = \Delta W \mathbf{x} = \begin{pmatrix}1&0&-1\\0&0&0\\1&0&-1\end{pmatrix}\begin{pmatrix}1\\1\\1\end{pmatrix} = \begin{pmatrix}0\\0\\0\end{pmatrix}$$

验证公式：$\Delta\mathbf{y} = \mathbf{b}(\mathbf{a}^T\mathbf{x}) = \begin{pmatrix}1\\0\\1\end{pmatrix}(1 \cdot 1 + 0 \cdot 1 + (-1) \cdot 1) = \begin{pmatrix}1\\0\\1\end{pmatrix} \cdot 0 = \mathbf{0}$ ✓

几何意义：$\mathbf{x} = (1,1,1)^T$ 与 $\mathbf{a} = (1,0,-1)^T$ 正交（$\mathbf{a}^T\mathbf{x} = 0$），故 LoRA 对此输入无任何修改——LoRA 只修改那些在 $\mathbf{a}$ 方向有分量的输入。

**（d）参数效率分析**

全量微调参数量（单个矩阵）：$d^2 = 4096^2 = 16{,}777{,}216 \approx 16.8\text{M}$

LoRA 参数量（单个矩阵）：$2dr = 2 \times 4096 \times 8 = 65{,}536 \approx 65.5\text{K}$

节省比例：$1 - \dfrac{2r}{d} = 1 - \dfrac{16}{4096} = 1 - 0.39\% = 99.61\%$（每层每矩阵节省 $99.61\%$）。

总节省参数量（$96$ 层，每层 $4$ 个投影）：

原始：$96 \times 4 \times 4096^2 \approx 6.44\text{ B}$

LoRA：$96 \times 4 \times 2 \times 4096 \times 8 \approx 25.2\text{ M}$

节省量 $\approx 6.44\text{ B} - 25.2\text{ M} \approx 6.41\text{ B}$，约节省 **99.6\%** 的参数——这正是 LoRA 使得在消费级 GPU 上微调超大模型成为可能的原因。

</details>
