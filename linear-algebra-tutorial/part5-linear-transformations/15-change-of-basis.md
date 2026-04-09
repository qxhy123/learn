# 第15章：基变换

> 同一个向量，用不同的"尺子"（基）来量，会得到不同的坐标；同一个线性变换，在不同的基下，有不同的矩阵表示。基变换理论回答了这样的核心问题：**当我们换一把尺子时，坐标和矩阵如何随之变化？**

---

## 学习目标

完成本章学习后，你将能够：

- 理解坐标变换公式，掌握如何在两组基之间转换向量的坐标表示
- 构造并应用过渡矩阵（Change of Basis Matrix），实现系统化的基变换
- 理解相似矩阵的定义 $A' = P^{-1}AP$，以及其在基变换中的几何意义
- 掌握相似矩阵的核心性质：行列式、迹、特征值、秩均相同
- 了解对角化的基本思想：选择特征向量基，使矩阵表示最简

---

## 15.1 坐标变换

### 15.1.1 问题的提出

在第11章，我们学习了向量在给定基下的坐标。现在自然地提出：**若已知向量在某组基下的坐标，如何求它在另一组基下的坐标？**

设 $V$ 是 $n$ 维向量空间，有两组基：

$$\mathcal{B} = \{b_1, b_2, \ldots, b_n\}, \quad \mathcal{C} = \{c_1, c_2, \ldots, c_n\}$$

向量 $x \in V$ 在两组基下的坐标分别记为 $[x]_\mathcal{B}$ 和 $[x]_\mathcal{C}$。

**问题：** 已知 $[x]_\mathcal{B}$，如何求 $[x]_\mathcal{C}$？

### 15.1.2 几何直觉

想象平面上的一个点 $P$，用标准坐标系（东-北方向）表示为 $(3, 2)$，换一个旋转了 $45°$ 的坐标系，同一个点 $P$ 的坐标就变成了不同的数字。点本身没有移动，只是描述它的"语言"换了。

坐标变换的本质：**空间中的对象不变，改变的只是描述它所用的参考系**。

### 15.1.3 坐标变换的推导

在 $\mathbb{R}^n$ 中，设标准基为 $\mathcal{E} = \{e_1, \ldots, e_n\}$。

任意向量 $x$ 在标准基下的坐标就是其分量本身：$[x]_\mathcal{E} = x$。

设基 $\mathcal{B} = \{b_1, \ldots, b_n\}$，构造**基矩阵**（将基向量排列为列）：

$$B = [b_1 \mid b_2 \mid \cdots \mid b_n]$$

则 $x$ 在基 $\mathcal{B}$ 下的坐标满足：

$$x = B \cdot [x]_\mathcal{B} \quad \Longleftrightarrow \quad [x]_\mathcal{B} = B^{-1} x$$

类似地，若基 $\mathcal{C}$ 的基矩阵为 $C = [c_1 \mid \cdots \mid c_n]$，则：

$$[x]_\mathcal{C} = C^{-1} x = C^{-1} B \cdot [x]_\mathcal{B}$$

这就是**坐标变换公式**。

---

## 15.2 过渡矩阵

### 15.2.1 定义

**定义（过渡矩阵）：** 设 $\mathcal{B}$ 和 $\mathcal{C}$ 是向量空间 $V$ 的两组基，从 $\mathcal{B}$ 到 $\mathcal{C}$ 的**过渡矩阵**（Change of Basis Matrix）定义为：

$$P_{\mathcal{C} \leftarrow \mathcal{B}} = C^{-1} B$$

其中 $B, C$ 分别是 $\mathcal{B}, \mathcal{C}$ 的基矩阵。坐标变换公式为：

$$[x]_\mathcal{C} = P_{\mathcal{C} \leftarrow \mathcal{B}} \cdot [x]_\mathcal{B}$$

**逆变换：** 从 $\mathcal{C}$ 回到 $\mathcal{B}$ 的过渡矩阵是其逆：

$$P_{\mathcal{B} \leftarrow \mathcal{C}} = \left(P_{\mathcal{C} \leftarrow \mathcal{B}}\right)^{-1} = B^{-1} C$$

### 15.2.2 过渡矩阵的列的含义

$P_{\mathcal{C} \leftarrow \mathcal{B}}$ 的第 $j$ 列正是 $\mathcal{B}$ 的第 $j$ 个基向量 $b_j$ 在基 $\mathcal{C}$ 下的坐标：

$$P_{\mathcal{C} \leftarrow \mathcal{B}} = \bigl[[b_1]_\mathcal{C} \mid [b_2]_\mathcal{C} \mid \cdots \mid [b_n]_\mathcal{C}\bigr]$$

这给出了**构造过渡矩阵的实际方法**：将每个旧基向量用新基表示，所得坐标向量排列为列。

### 15.2.3 计算方法：增广矩阵消元

在 $\mathbb{R}^n$ 中，求从 $\mathcal{B}$ 到 $\mathcal{C}$ 的过渡矩阵 $P = C^{-1}B$ 的标准步骤：

$$[C \mid B] \xrightarrow{\text{行化简}} [I \mid C^{-1}B] = [I \mid P]$$

即对增广矩阵 $[C \mid B]$ 做行变换，将左侧化为单位矩阵，右侧自然得到过渡矩阵 $P$。

**例 15.1：** 设 $\mathbb{R}^2$ 中两组基：

$$\mathcal{B} = \left\{b_1 = \begin{pmatrix}1\\1\end{pmatrix},\ b_2 = \begin{pmatrix}1\\-1\end{pmatrix}\right\}, \quad \mathcal{C} = \left\{c_1 = \begin{pmatrix}2\\0\end{pmatrix},\ c_2 = \begin{pmatrix}0\\3\end{pmatrix}\right\}$$

求从 $\mathcal{B}$ 到 $\mathcal{C}$ 的过渡矩阵，并将 $[x]_\mathcal{B} = \begin{pmatrix}2\\1\end{pmatrix}$ 转换到 $\mathcal{C}$ 下的坐标。

**解：** 构造增广矩阵 $[C \mid B]$ 并行化简：

$$\left[\begin{array}{cc|cc}2&0&1&1\\0&3&1&-1\end{array}\right] \xrightarrow{R_1/2,\; R_2/3} \left[\begin{array}{cc|cc}1&0&\frac{1}{2}&\frac{1}{2}\\0&1&\frac{1}{3}&-\frac{1}{3}\end{array}\right]$$

故过渡矩阵为：

$$P_{\mathcal{C} \leftarrow \mathcal{B}} = \begin{pmatrix}\frac{1}{2} & \frac{1}{2}\\[4pt]\frac{1}{3} & -\frac{1}{3}\end{pmatrix}$$

坐标转换：

$$[x]_\mathcal{C} = P \cdot [x]_\mathcal{B} = \begin{pmatrix}\frac{1}{2} & \frac{1}{2}\\[4pt]\frac{1}{3} & -\frac{1}{3}\end{pmatrix}\begin{pmatrix}2\\1\end{pmatrix} = \begin{pmatrix}\frac{3}{2}\\[4pt]\frac{1}{3}\end{pmatrix}$$

**验证：** $x = 2b_1 + b_2 = \begin{pmatrix}3\\1\end{pmatrix}$，而 $\frac{3}{2}c_1 + \frac{1}{3}c_2 = \begin{pmatrix}3\\0\end{pmatrix} + \begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}3\\1\end{pmatrix}$ ✓

### 15.2.4 线性变换在不同基下的矩阵

设线性变换 $T: V \to V$，在基 $\mathcal{B}$ 下的矩阵为 $A$（即 $[Tv]_\mathcal{B} = A[v]_\mathcal{B}$），在基 $\mathcal{C}$ 下的矩阵为 $A'$。

从 $\mathcal{B}$ 到 $\mathcal{C}$ 的过渡矩阵为 $P = P_{\mathcal{C} \leftarrow \mathcal{B}}$，则：

$$A' = P^{-1} A P$$

这就是下一节要深入讨论的**相似矩阵**关系。

---

## 15.3 相似矩阵

### 15.3.1 定义

**定义（相似矩阵）：** 设 $A, A'$ 是 $n \times n$ 矩阵，若存在**可逆矩阵** $P$ 使得：

$$A' = P^{-1} A P$$

则称 $A'$ 与 $A$ **相似**（Similar），记作 $A' \sim A$，称 $P$ 为**相似变换矩阵**。

### 15.3.2 几何意义

相似性的几何本质：**同一线性变换，在不同基下的矩阵表示**。

具体地，若 $P$ 的列是新基向量（在旧基坐标系下的表示），则 $A' = P^{-1}AP$ 的计算路径是：

$$\underbrace{P}_{\text{新→旧坐标}} \xrightarrow{\text{施加变换 }A} \underbrace{A}_{\text{旧坐标下的变换}} \xrightarrow{} \underbrace{P^{-1}}_{\text{旧→新坐标}}$$

即：在新坐标系中施加变换 $A'$，等价于先切换到旧坐标系，施加 $A$，再切回新坐标系。

```
新坐标系输入 [v]_C  --(P)--> 旧坐标系 [v]_B
                                 |
                                (A)
                                 |
              新坐标系输出 <--(P^{-1})-- 旧坐标系 [Tv]_B
```

### 15.3.3 相似是等价关系

**定理：** 相似关系满足等价关系的三条性质：

1. **自反性：** $A \sim A$（取 $P = I$）
2. **对称性：** 若 $A' \sim A$，则 $A \sim A'$（取变换矩阵 $P^{-1}$）
3. **传递性：** 若 $A'' \sim A'$ 且 $A' \sim A$，则 $A'' \sim A$

**例 15.2：** 设 $A = \begin{pmatrix}4&1\\2&3\end{pmatrix}$，$P = \begin{pmatrix}1&1\\1&-2\end{pmatrix}$，求 $A' = P^{-1}AP$。

**解：** 先求 $P^{-1}$（行列式 $= -3$）：

$$P^{-1} = \frac{1}{-3}\begin{pmatrix}-2&-1\\-1&1\end{pmatrix} = \begin{pmatrix}\frac{2}{3}&\frac{1}{3}\\[4pt]\frac{1}{3}&-\frac{1}{3}\end{pmatrix}$$

计算 $AP$：

$$AP = \begin{pmatrix}4&1\\2&3\end{pmatrix}\begin{pmatrix}1&1\\1&-2\end{pmatrix} = \begin{pmatrix}5&2\\5&-4\end{pmatrix}$$

计算 $P^{-1}AP$：

$$A' = \begin{pmatrix}\frac{2}{3}&\frac{1}{3}\\[4pt]\frac{1}{3}&-\frac{1}{3}\end{pmatrix}\begin{pmatrix}5&2\\5&-4\end{pmatrix} = \begin{pmatrix}5&0\\0&2\end{pmatrix}$$

$A'$ 是对角矩阵！这正是对角化——选择了合适的 $P$（特征向量矩阵），使变换在新基下表示最简。

---

## 15.4 相似矩阵的性质

相似矩阵虽然在形式上不同，但它们描述的是**同一个线性变换**，因此共享所有与变换本身相关的"内禀"量。

**定理（相似不变量）：** 若 $A' = P^{-1}AP$，则：

### 15.4.1 行列式相同

$$\det(A') = \det(P^{-1}AP) = \det(P^{-1})\det(A)\det(P) = \frac{1}{\det(P)}\det(A)\det(P) = \det(A)$$

几何意义：行列式刻画线性变换对体积的缩放比，与坐标系选取无关。

### 15.4.2 迹相同

**定义（迹）：** 方阵 $A$ 的**迹**（Trace）为主对角元素之和：$\text{tr}(A) = \sum_{i=1}^n a_{ii}$。

**证明：** 利用迹的循环性质 $\text{tr}(XY) = \text{tr}(YX)$：

$$\text{tr}(A') = \text{tr}(P^{-1}AP) = \text{tr}(P^{-1}(AP)) = \text{tr}((AP)P^{-1}) = \text{tr}(A(PP^{-1})) = \text{tr}(A)$$

### 15.4.3 特征多项式和特征值相同

**证明：**

$$\det(\lambda I - A') = \det(\lambda I - P^{-1}AP) = \det(P^{-1}(\lambda I - A)P) = \det(\lambda I - A)$$

因此 $A$ 和 $A'$ 有**完全相同的特征多项式**，从而特征值（含重数）完全相同。

这进一步给出：

- $\det(A) =$ 所有特征值之积（特征值相同 $\Rightarrow$ 行列式相同）
- $\text{tr}(A) =$ 所有特征值之和（特征值相同 $\Rightarrow$ 迹相同）

### 15.4.4 秩相同

$$\text{rank}(A') = \text{rank}(P^{-1}AP) = \text{rank}(A)$$

因为左乘/右乘可逆矩阵不改变矩阵的秩（可逆矩阵是满秩的行/列变换）。

### 15.4.5 总结表

| 不变量 | 表达式 | 几何含义 |
|---|---|---|
| **行列式** | $\det(A)$ | 体积缩放比 |
| **迹** | $\text{tr}(A)$ | 特征值之和 |
| **特征值** | $\lambda_1, \ldots, \lambda_n$ | 本征拉伸/旋转 |
| **特征多项式** | $\det(\lambda I - A)$ | 代数结构完整描述 |
| **秩** | $\text{rank}(A)$ | 像空间维数 |
| **零化度** | $\text{nullity}(A)$ | 核空间维数 |

> **注意：** 矩阵元素本身不是相似不变量。相似但不相等的矩阵，在外观上可以截然不同。

---

## 15.5 对角化初步

### 15.5.1 什么是对角化

**定义（可对角化）：** 若方阵 $A$ 相似于某个对角矩阵 $\Lambda$，即存在可逆矩阵 $P$ 使得：

$$P^{-1}AP = \Lambda = \begin{pmatrix}\lambda_1 & & \\ & \ddots & \\ & & \lambda_n\end{pmatrix}$$

则称 $A$ **可对角化**（Diagonalizable），$P$ 称为**对角化矩阵**，$\Lambda$ 称为 $A$ 的**对角形式**。

### 15.5.2 对角化的条件

**定理（对角化充要条件）：** $n$ 阶方阵 $A$ 可对角化，当且仅当 $A$ 有 $n$ 个线性无关的特征向量。

此时，$P$ 的列恰好是这 $n$ 个特征向量，$\Lambda$ 的对角元素是对应的特征值。

**充分条件：** 若 $A$ 有 $n$ 个**不同**的特征值，则 $A$ 一定可对角化（不同特征值对应的特征向量线性无关）。

**反例：** $A = \begin{pmatrix}1&1\\0&1\end{pmatrix}$（Jordan 块），特征值为 $\lambda = 1$（重数 2），但只有 1 个线性无关的特征向量，因此**不可对角化**。

### 15.5.3 对角化的好处

若 $A = P\Lambda P^{-1}$，则计算变得极为简便：

**矩阵幂次：**

$$A^k = (P\Lambda P^{-1})^k = P\Lambda^k P^{-1} = P\begin{pmatrix}\lambda_1^k & & \\ & \ddots & \\ & & \lambda_n^k\end{pmatrix}P^{-1}$$

计算 $A^{100}$ 只需计算 $\lambda_i^{100}$，无需做 100 次矩阵乘法！

**矩阵指数、解微分方程等**也因此大为简化（将在后续章节展开）。

### 15.5.4 对角化步骤

求 $A$ 的对角化 $A = P\Lambda P^{-1}$：

1. 求特征多项式 $\det(\lambda I - A)$，解出特征值 $\lambda_1, \ldots, \lambda_k$。
2. 对每个特征值 $\lambda_i$，求特征空间（零空间 of $(\lambda_i I - A)$），得到特征向量。
3. 若共得到 $n$ 个线性无关的特征向量，则令 $P = [v_1 \mid v_2 \mid \cdots \mid v_n]$，$\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$。
4. 验证：$P^{-1}AP = \Lambda$。

**例 15.3：** 对 $A = \begin{pmatrix}4&1\\2&3\end{pmatrix}$ 求对角化。

**解：**

**步骤 1：** 特征多项式：

$$\det(\lambda I - A) = \det\begin{pmatrix}\lambda-4&-1\\-2&\lambda-3\end{pmatrix} = (\lambda-4)(\lambda-3) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda-5)(\lambda-2)$$

特征值：$\lambda_1 = 5,\ \lambda_2 = 2$。

**步骤 2：** 求特征向量：

- $\lambda_1 = 5$：$(\lambda_1 I - A) = \begin{pmatrix}1&-1\\-2&2\end{pmatrix}$，零空间为 $\text{span}\left\{\begin{pmatrix}1\\1\end{pmatrix}\right\}$。

- $\lambda_2 = 2$：$(\lambda_2 I - A) = \begin{pmatrix}-2&-1\\-2&-1\end{pmatrix}$，零空间为 $\text{span}\left\{\begin{pmatrix}1\\-2\end{pmatrix}\right\}$。

**步骤 3：** 构造 $P$ 和 $\Lambda$：

$$P = \begin{pmatrix}1&1\\1&-2\end{pmatrix}, \quad \Lambda = \begin{pmatrix}5&0\\0&2\end{pmatrix}$$

（这与例 15.2 中的结果一致——$P$ 的列正是特征向量！）

---

## 本章小结

- **坐标变换：** 已知 $[x]_\mathcal{B}$，用过渡矩阵 $P = C^{-1}B$ 计算 $[x]_\mathcal{C} = P[x]_\mathcal{B}$

- **过渡矩阵的构造：** 对增广矩阵 $[C \mid B]$ 行化简，左侧变为 $I$，右侧得 $P$；或直接将每个旧基向量用新基表示，得到 $P$ 的列

- **相似矩阵：** $A' = P^{-1}AP$ 表示同一线性变换在不同基下的矩阵，$P$ 的列是新基向量

- **相似不变量：** 行列式、迹、特征多项式、特征值、秩——这些量刻画变换本身，与基的选取无关

- **对角化：** 若选特征向量为基，线性变换的矩阵变为对角矩阵；对角化条件是存在 $n$ 个线性无关的特征向量

- **核心公式速查：**

$$[x]_\mathcal{C} = P^{-1}_\mathcal{C} P_\mathcal{B} \cdot [x]_\mathcal{B}, \quad A_\mathcal{C} = P^{-1} A_\mathcal{B} P, \quad A = P\Lambda P^{-1}$$

---

## 深度学习应用：Batch Normalization 与白化

### 背景：为什么需要"规范化坐标系"

在深度学习中，神经网络各层的输入分布会随着训练不断漂移（称为 **Internal Covariate Shift**），这导致训练不稳定、学习率难以设定。

从线性代数的视角来看，这等价于：数据所处的"坐标系"持续变化，使得梯度计算困难。解决方案是主动将数据变换到一个"更好"的坐标系。

### 15.A.1 白化变换：最彻底的坐标规范化

**白化（Whitening）** 将数据变换为**零均值、单位协方差**的分布——即各特征之间不相关且方差为 1。

**数学表述：** 设数据 $X \in \mathbb{R}^{m \times d}$（$m$ 个样本，$d$ 维特征），协方差矩阵为 $\Sigma$。

白化变换矩阵 $W = \Sigma^{-1/2}$，白化后的数据：

$$X_{\text{white}} = X_{\text{centered}} \cdot W^T$$

满足 $\text{Cov}(X_{\text{white}}) = I$。

**基变换视角：** 白化正是将数据从"原始坐标系"变换到"特征向量坐标系 + 归一化缩放"的过渡矩阵操作：

$$\Sigma = Q \Lambda Q^T \quad \Rightarrow \quad W = Q \Lambda^{-1/2} Q^T$$

其中 $Q$ 是特征向量矩阵（新基），$\Lambda^{-1/2}$ 将每个方向上的方差缩放为 1。

白化可分两步理解：
1. **去相关（旋转）：** 用特征向量矩阵 $Q^T$ 旋转坐标系，使各维度对齐方差最大方向（类 PCA）
2. **归一化（缩放）：** 用 $\Lambda^{-1/2}$ 将各方向方差压缩为 1

### 15.A.2 Batch Normalization：实用的近似白化

完整白化计算协方差矩阵并求逆，代价较高，且在小 batch 下不稳定。

**Batch Normalization（BN，Ioffe & Szegedy, 2015）** 是一种轻量的近似：只做**各特征独立的均值-方差归一化**，不去除特征间的相关性。

**BN 的变换步骤：**

设某层输入 $z \in \mathbb{R}^d$，在一个 batch 上计算：

$$\mu_B = \frac{1}{m}\sum_{i=1}^m z^{(i)}, \quad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (z^{(i)} - \mu_B)^2$$

归一化：

$$\hat{z}^{(i)} = \frac{z^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$$

线性变换（可学习参数 $\gamma, \beta$）：

$$y^{(i)} = \gamma \odot \hat{z}^{(i)} + \beta$$

**基变换视角：** 这整个过程等价于将每个特征维度的坐标系**平移**（去均值）再**缩放**（除以标准差），即一个对角矩阵形式的过渡变换：

$$z \mapsto \hat{z} = D^{-1}(z - \mu), \quad D = \text{diag}(\sigma_1, \ldots, \sigma_d)$$

然后 $\gamma, \beta$ 学习一个新的仿射变换，将 $\hat{z}$ 变换到网络实际需要的"工作坐标系"。

**与完整白化的对比：**

| 方法 | 去均值 | 归一化方差 | 去相关 | 计算复杂度 |
|---|---|---|---|---|
| 白化 | ✓ | ✓ | ✓ | $O(d^3)$（求逆）|
| Batch Norm | ✓ | ✓ | ✗ | $O(d)$（逐维） |

BN 放弃了去相关步骤（忽略特征间协方差），换来了 $O(d)$ 的高效性和训练稳定性。

### 15.A.3 代码示例

```python
import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# 第一部分：手动实现白化变换（基变换视角）
# ─────────────────────────────────────────────

np.random.seed(42)

# 生成有相关性的 2D 数据
mean = np.array([2.0, 3.0])
cov  = np.array([[3.0, 2.0],
                 [2.0, 2.0]])
X = np.random.multivariate_normal(mean, cov, size=500)  # (500, 2)

# 步骤 1：中心化（去均值）
X_centered = X - X.mean(axis=0)

# 步骤 2：计算协方差矩阵
Sigma = np.cov(X_centered.T)   # (2, 2)
print("协方差矩阵 Sigma:\n", np.round(Sigma, 4))

# 步骤 3：特征分解 Sigma = Q @ Lambda @ Q^T
eigenvalues, Q = np.linalg.eigh(Sigma)     # 升序
# 过渡矩阵 W = Q @ diag(lambda^{-1/2}) @ Q^T
W = Q @ np.diag(eigenvalues ** -0.5) @ Q.T  # 白化矩阵

# 步骤 4：应用白化（坐标变换）
X_white = X_centered @ W.T   # (500, 2)

# 验证：白化后协方差应接近单位矩阵
Sigma_white = np.cov(X_white.T)
print("\n白化后协方差矩阵（应接近 I）:\n", np.round(Sigma_white, 4))

# ─────────────────────────────────────────────
# 第二部分：PyTorch Batch Normalization
# ─────────────────────────────────────────────

torch.manual_seed(42)

# 构造一个有明显均值和方差差异的特征矩阵
# shape: (batch=32, features=4)
batch_size, d = 32, 4
raw = torch.randn(batch_size, d)
# 人为给各特征设置不同均值和方差
scale  = torch.tensor([1.0, 5.0, 0.1, 20.0])
offset = torch.tensor([0.0, 10.0, -5.0, 100.0])
z = raw * scale + offset

print("\n归一化前各特征均值:", z.mean(dim=0).detach().numpy().round(3))
print("归一化前各特征标准差:", z.std(dim=0).detach().numpy().round(3))

# 应用 Batch Normalization（affine=False 关闭可学习参数，纯归一化）
bn = nn.BatchNorm1d(num_features=d, affine=False)
bn.eval()   # 训练模式下 BN 使用 batch 统计量

# 手动模拟训练时的 BN（eval 模式需要先跑一次 forward 更新 running stats）
bn.train()
z_bn = bn(z)

print("\n归一化后各特征均值:", z_bn.mean(dim=0).detach().numpy().round(3))
print("归一化后各特征标准差:", z_bn.std(dim=0).detach().numpy().round(3))

# ─────────────────────────────────────────────
# 第三部分：对比手动 BN 与 PyTorch BN
# ─────────────────────────────────────────────

eps = 1e-5
mu    = z.mean(dim=0, keepdim=True)          # (1, d)
sigma = z.std(dim=0, keepdim=True, unbiased=False) + eps  # (1, d)

z_manual = (z - mu) / sigma   # 手动逐特征归一化

# 过渡矩阵视角：等价于 z_manual = (z - mu) @ D^{-1}
# 其中 D = diag(sigma_1, ..., sigma_d)
D_inv = torch.diag(1.0 / (z.std(dim=0, unbiased=False) + eps))
z_matrix = (z - mu) @ D_inv   # 矩阵形式的坐标变换

max_diff = (z_manual - z_matrix).abs().max().item()
print(f"\n手动 BN 与矩阵变换结果最大差值: {max_diff:.2e}")  # 应为 0（等价）

print("\n[基变换视角] D_inv（过渡矩阵，对角形式）:\n", D_inv.detach().numpy().round(4))
```

**代码解读：**

- **白化部分：** `W = Q @ diag(λ^{-1/2}) @ Q^T` 就是从原始坐标系到"标准正交坐标系"的过渡矩阵，应用后协方差矩阵变为单位矩阵
- **BN 部分：** `nn.BatchNorm1d` 对每个特征维度独立做均值-方差归一化
- **矩阵视角：** BN 等价于右乘对角过渡矩阵 $D^{-1} = \text{diag}(1/\sigma_1, \ldots, 1/\sigma_d)$，这是完整白化矩阵的对角近似

**直觉总结：**

> Batch Normalization 是把每层的输入强制"对齐"到一个标准坐标系（零均值、单位方差），消除因参数更新引起的坐标系漂移，从而让梯度下降在一个稳定、良条件的空间中进行。可学习的 $\gamma, \beta$ 则允许网络在此基础上选择它真正需要的"工作坐标系"。

---

## 练习题

**练习 1（基础：过渡矩阵构造）**

设 $\mathbb{R}^2$ 中两组基：

$$\mathcal{B} = \left\{b_1 = \begin{pmatrix}3\\1\end{pmatrix}, b_2 = \begin{pmatrix}1\\2\end{pmatrix}\right\}, \quad \mathcal{E} = \left\{e_1 = \begin{pmatrix}1\\0\end{pmatrix}, e_2 = \begin{pmatrix}0\\1\end{pmatrix}\right\}$$

(a) 求从 $\mathcal{B}$ 到标准基 $\mathcal{E}$ 的过渡矩阵 $P_{\mathcal{E} \leftarrow \mathcal{B}}$。

(b) 向量 $x$ 在基 $\mathcal{B}$ 下的坐标为 $[x]_\mathcal{B} = \begin{pmatrix}2\\-1\end{pmatrix}$，求 $[x]_\mathcal{E}$（即 $x$ 的标准坐标）。

**练习 2（坐标变换）**

设 $\mathbb{R}^3$ 中两组基：

$$\mathcal{B} = \left\{\begin{pmatrix}1\\0\\0\end{pmatrix}, \begin{pmatrix}0\\1\\0\end{pmatrix}, \begin{pmatrix}0\\0\\1\end{pmatrix}\right\} = \mathcal{E}, \quad \mathcal{C} = \left\{\begin{pmatrix}1\\1\\0\end{pmatrix}, \begin{pmatrix}0\\1\\1\end{pmatrix}, \begin{pmatrix}1\\0\\1\end{pmatrix}\right\}$$

向量 $x = \begin{pmatrix}3\\2\\1\end{pmatrix}$ 在标准基下的坐标已知，求 $[x]_\mathcal{C}$（$x$ 在基 $\mathcal{C}$ 下的坐标）。

**练习 3（相似不变量验证）**

设 $A = \begin{pmatrix}2&3\\0&-1\end{pmatrix}$，$P = \begin{pmatrix}1&1\\1&0\end{pmatrix}$，令 $A' = P^{-1}AP$。

(a) 计算 $A'$。

(b) 不通过直接计算 $A'$ 的特征值，说明 $A$ 与 $A'$ 的特征值相同，并验证 $\text{tr}(A) = \text{tr}(A')$，$\det(A) = \det(A')$。

**练习 4（对角化）**

对矩阵 $A = \begin{pmatrix}1&2\\2&1\end{pmatrix}$：

(a) 求特征值和特征向量。

(b) 找到可逆矩阵 $P$，使 $P^{-1}AP$ 为对角矩阵，写出对角矩阵 $\Lambda$。

(c) 利用对角化计算 $A^{10}$。

**练习 5（综合思考）**

设 $A$ 是 $n \times n$ 矩阵，证明：

(a) 若 $A$ 可对角化且 $A = P\Lambda P^{-1}$，则 $A^k = P\Lambda^k P^{-1}$（对任意正整数 $k$）。

(b) 若 $A \sim B$（$A$ 与 $B$ 相似），则 $A^2 \sim B^2$。

(c) 给出一个例子，说明 $A + B \sim C + D$ 不能由 $A \sim C$ 和 $B \sim D$ 推出（即相似性对加法不保持）。

---

## 练习答案

<details>
<summary><strong>练习 1 答案</strong></summary>

**(a)** 从 $\mathcal{B}$ 到标准基 $\mathcal{E}$ 的过渡矩阵：

由于标准基的基矩阵是单位矩阵 $E = I$，所以：

$$P_{\mathcal{E} \leftarrow \mathcal{B}} = E^{-1} B = I^{-1} \begin{pmatrix}3&1\\1&2\end{pmatrix} = \begin{pmatrix}3&1\\1&2\end{pmatrix} = B$$

即从 $\mathcal{B}$ 到标准基的过渡矩阵恰好就是 $\mathcal{B}$ 的基矩阵本身。这符合直觉：$[x]_\mathcal{E} = x = B \cdot [x]_\mathcal{B}$。

**(b)** 代入计算：

$$[x]_\mathcal{E} = P_{\mathcal{E} \leftarrow \mathcal{B}} \cdot [x]_\mathcal{B} = \begin{pmatrix}3&1\\1&2\end{pmatrix}\begin{pmatrix}2\\-1\end{pmatrix} = \begin{pmatrix}6-1\\2-2\end{pmatrix} = \begin{pmatrix}5\\0\end{pmatrix}$$

验证：$x = 2b_1 + (-1)b_2 = 2\begin{pmatrix}3\\1\end{pmatrix} - \begin{pmatrix}1\\2\end{pmatrix} = \begin{pmatrix}5\\0\end{pmatrix}$ ✓

</details>

<details>
<summary><strong>练习 2 答案</strong></summary>

求 $[x]_\mathcal{C}$ 需要求解线性方程组 $C \cdot [x]_\mathcal{C} = x$：

$$C = \begin{pmatrix}1&0&1\\1&1&0\\0&1&1\end{pmatrix}, \quad x = \begin{pmatrix}3\\2\\1\end{pmatrix}$$

构造增广矩阵 $[C \mid x]$ 并行化简：

$$\left[\begin{array}{ccc|c}1&0&1&3\\1&1&0&2\\0&1&1&1\end{array}\right] \xrightarrow{R_2-R_1} \left[\begin{array}{ccc|c}1&0&1&3\\0&1&-1&-1\\0&1&1&1\end{array}\right] \xrightarrow{R_3-R_2} \left[\begin{array}{ccc|c}1&0&1&3\\0&1&-1&-1\\0&0&2&2\end{array}\right]$$

$$\xrightarrow{R_3/2} \left[\begin{array}{ccc|c}1&0&1&3\\0&1&-1&-1\\0&0&1&1\end{array}\right] \xrightarrow{R_1-R_3,\; R_2+R_3} \left[\begin{array}{ccc|c}1&0&0&2\\0&1&0&0\\0&0&1&1\end{array}\right]$$

故 $[x]_\mathcal{C} = \begin{pmatrix}2\\0\\1\end{pmatrix}$。

验证：$2\begin{pmatrix}1\\1\\0\end{pmatrix} + 0\begin{pmatrix}0\\1\\1\end{pmatrix} + 1\begin{pmatrix}1\\0\\1\end{pmatrix} = \begin{pmatrix}3\\2\\1\end{pmatrix}$ ✓

</details>

<details>
<summary><strong>练习 3 答案</strong></summary>

**(a)** 计算 $P^{-1}$（行列式 $\det(P) = 0 \cdot 1 - 1 \cdot 1 = -1$）：

$$P^{-1} = \frac{1}{-1}\begin{pmatrix}0&-1\\-1&1\end{pmatrix} = \begin{pmatrix}0&1\\1&-1\end{pmatrix}$$

计算 $AP$：

$$AP = \begin{pmatrix}2&3\\0&-1\end{pmatrix}\begin{pmatrix}1&1\\1&0\end{pmatrix} = \begin{pmatrix}5&2\\-1&0\end{pmatrix}$$

计算 $A' = P^{-1}AP$：

$$A' = \begin{pmatrix}0&1\\1&-1\end{pmatrix}\begin{pmatrix}5&2\\-1&0\end{pmatrix} = \begin{pmatrix}-1&0\\6&2\end{pmatrix}$$

**(b)** 由于 $A' = P^{-1}AP$，$A$ 与 $A'$ 相似，它们有相同的特征多项式，从而特征值相同。

验证迹：
$$\text{tr}(A) = 2 + (-1) = 1, \quad \text{tr}(A') = -1 + 2 = 1 \quad \checkmark$$

验证行列式：
$$\det(A) = 2 \cdot (-1) - 3 \cdot 0 = -2, \quad \det(A') = (-1)(2) - (0)(6) = -2 \quad \checkmark$$

$A$ 的特征值满足 $\lambda_1 + \lambda_2 = 1$，$\lambda_1 \lambda_2 = -2$，解得 $\lambda_1 = 2, \lambda_2 = -1$。$A'$ 的特征值相同。

</details>

<details>
<summary><strong>练习 4 答案</strong></summary>

**(a)** 特征多项式：

$$\det(\lambda I - A) = \det\begin{pmatrix}\lambda-1&-2\\-2&\lambda-1\end{pmatrix} = (\lambda-1)^2 - 4 = \lambda^2 - 2\lambda - 3 = (\lambda-3)(\lambda+1)$$

特征值：$\lambda_1 = 3$，$\lambda_2 = -1$。

特征向量：
- $\lambda_1 = 3$：$(3I - A) = \begin{pmatrix}2&-2\\-2&2\end{pmatrix}$，零空间为 $\text{span}\left\{\begin{pmatrix}1\\1\end{pmatrix}\right\}$
- $\lambda_2 = -1$：$(-I - A) = \begin{pmatrix}-2&-2\\-2&-2\end{pmatrix}$，零空间为 $\text{span}\left\{\begin{pmatrix}1\\-1\end{pmatrix}\right\}$

**(b)** 对角化矩阵：

$$P = \begin{pmatrix}1&1\\1&-1\end{pmatrix}, \quad \Lambda = \begin{pmatrix}3&0\\0&-1\end{pmatrix}$$

验证：$\det(P) = -2 \neq 0$，$P$ 可逆。$A = P\Lambda P^{-1}$。

**(c)** 利用 $A^{10} = P\Lambda^{10}P^{-1}$：

$$\Lambda^{10} = \begin{pmatrix}3^{10}&0\\0&(-1)^{10}\end{pmatrix} = \begin{pmatrix}59049&0\\0&1\end{pmatrix}$$

$P^{-1} = \frac{1}{-2}\begin{pmatrix}-1&-1\\-1&1\end{pmatrix} = \begin{pmatrix}\frac{1}{2}&\frac{1}{2}\\[4pt]\frac{1}{2}&-\frac{1}{2}\end{pmatrix}$

$$A^{10} = \begin{pmatrix}1&1\\1&-1\end{pmatrix}\begin{pmatrix}59049&0\\0&1\end{pmatrix}\begin{pmatrix}\frac{1}{2}&\frac{1}{2}\\[4pt]\frac{1}{2}&-\frac{1}{2}\end{pmatrix}$$

$$= \begin{pmatrix}59049&1\\59049&-1\end{pmatrix}\begin{pmatrix}\frac{1}{2}&\frac{1}{2}\\[4pt]\frac{1}{2}&-\frac{1}{2}\end{pmatrix} = \begin{pmatrix}\frac{59050}{2}&\frac{59048}{2}\\[4pt]\frac{59048}{2}&\frac{59050}{2}\end{pmatrix} = \begin{pmatrix}29525&29524\\29524&29525\end{pmatrix}$$

</details>

<details>
<summary><strong>练习 5 答案</strong></summary>

**(a)** 对正整数 $k$ 用归纳法（或直接展开）：

$$A^k = (P\Lambda P^{-1})^k = \underbrace{(P\Lambda P^{-1})(P\Lambda P^{-1})\cdots(P\Lambda P^{-1})}_{k \text{ 个}}$$

每对相邻的 $P^{-1}P = I$ 消去，得：

$$A^k = P \underbrace{\Lambda \Lambda \cdots \Lambda}_{k \text{ 个}} P^{-1} = P\Lambda^k P^{-1}$$

由于 $\Lambda$ 是对角矩阵，$\Lambda^k = \text{diag}(\lambda_1^k, \ldots, \lambda_n^k)$，计算仅需对每个对角元取幂。$\square$

**(b)** 设 $A = P^{-1}BP$（即 $A \sim B$），则：

$$A^2 = (P^{-1}BP)^2 = P^{-1}BP \cdot P^{-1}BP = P^{-1}B^2P$$

故 $A^2 \sim B^2$，相似变换矩阵仍为同一个 $P$。$\square$

**(c)** 取：

$$A = \begin{pmatrix}1&0\\0&0\end{pmatrix}, \quad C = \begin{pmatrix}0&0\\0&1\end{pmatrix}, \quad B = D = \begin{pmatrix}0&0\\0&0\end{pmatrix}$$

$A \sim C$（均只有特征值 $1$ 和 $0$，都是秩 1 矩阵；取 $P = \begin{pmatrix}0&1\\1&0\end{pmatrix}$，则 $P^{-1}AP = C$）。

$B = D = 0$，显然 $B \sim D$。

但 $A + B = A = \begin{pmatrix}1&0\\0&0\end{pmatrix}$，$C + D = C = \begin{pmatrix}0&0\\0&1\end{pmatrix}$，而 $A + B \sim C + D$ 确实成立（此例不够好）。

更好的反例：取 $A = I, C = I$（$A \sim C$），$B = \begin{pmatrix}0&1\\0&0\end{pmatrix}$，$D = \begin{pmatrix}0&0\\1&0\end{pmatrix}$（$B \sim D$，同为秩 1 幂零矩阵）。

则 $A + B = \begin{pmatrix}1&1\\0&1\end{pmatrix}$（Jordan 块，不可对角化），$C + D = \begin{pmatrix}1&0\\1&1\end{pmatrix}$（Jordan 块，不可对角化）。

两者特征值相同（均为 $\lambda = 1$ 重数 2），且均为 Jordan 块，事实上它们**确实相似**（都与同一 Jordan 标准形相似）。

**真正的反例：** 取 $A = \begin{pmatrix}1&0\\0&0\end{pmatrix}$，$C = \begin{pmatrix}1&0\\0&0\end{pmatrix}$（$A = C$，当然相似），$B = \begin{pmatrix}1&0\\0&0\end{pmatrix}$，$D = \begin{pmatrix}0&0\\0&1\end{pmatrix}$（$B \sim D$）。

则 $A + B = \begin{pmatrix}2&0\\0&0\end{pmatrix}$（特征值 $2, 0$），$C + D = \begin{pmatrix}1&0\\0&1\end{pmatrix} = I$（特征值 $1, 1$）。

特征值不同，$A+B$ 与 $C+D$ 不相似。这说明**相似性对矩阵加法不具有"分配性"**：$A \sim C$ 且 $B \sim D$ 不能推出 $A+B \sim C+D$。$\square$

</details>
