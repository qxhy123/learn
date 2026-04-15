# 第6章：逆矩阵

> 逆矩阵是线性代数中"撤销"一次线性变换的工具——但并非所有变换都能被撤销。

---

## 学习目标

完成本章学习后，你将能够：

- 理解逆矩阵的定义和它在线性变换中的几何意义
- 掌握判断矩阵可逆的充要条件
- 熟练计算 $2 \times 2$ 矩阵的逆，并能用初等行变换法求一般方阵的逆
- 掌握逆矩阵的代数性质并能灵活运用
- 理解逆矩阵在求解线性方程组中的作用，以及为何实际中避免显式求逆

---

## 6.1 逆矩阵的定义

### 定义

设 $A$ 为 $n \times n$ 的**方阵**。若存在一个 $n \times n$ 矩阵 $B$，使得

$$AB = BA = I_n$$

则称 $A$ 是**可逆矩阵**（invertible matrix），$B$ 是 $A$ 的**逆矩阵**，记作 $B = A^{-1}$。

**几何直觉：** 若矩阵 $A$ 对应一个线性变换（如旋转、缩放、切变），那么 $A^{-1}$ 对应的是"将该变换还原"的操作。例如，旋转 $\theta$ 角的矩阵，其逆矩阵对应旋转 $-\theta$ 角。

**注意：** 逆矩阵仅对**方阵**有定义。非方阵（行数不等于列数）没有（双侧）逆矩阵。

### 可逆矩阵与奇异矩阵

- **可逆矩阵**（非奇异矩阵，non-singular matrix）：存在逆矩阵的方阵。
- **不可逆矩阵**（奇异矩阵，singular matrix）：不存在逆矩阵的方阵。

**奇异矩阵的直觉：** 奇异矩阵对应的线性变换会将空间"压缩"到更低维度（例如把平面压成一条线），这一过程是**不可逆**的——压缩后的信息已经丢失，无法还原。

**经典的奇异矩阵示例：**

$$A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$$

第二行是第一行的 2 倍，矩阵将整个平面压缩到一条直线上，因此不可逆。

### 逆矩阵的唯一性

**定理：** 若矩阵 $A$ 可逆，则其逆矩阵是唯一的。

**证明：** 假设 $B$ 和 $C$ 都是 $A$ 的逆矩阵，即 $AB = BA = I$ 且 $AC = CA = I$，则

$$B = BI = B(AC) = (BA)C = IC = C$$

因此 $B = C$，逆矩阵唯一。$\square$

---

## 6.2 逆矩阵的计算

### 2×2 矩阵的逆公式

对于 $2 \times 2$ 矩阵

$$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$$

若 $ad - bc \ne 0$，则 $A$ 可逆，其逆为

$$A^{-1} = \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

其中 $\det(A) = ad - bc$ 称为 $A$ 的**行列式**（determinant），将在第7章详细介绍。当 $\det(A) = 0$ 时，$A$ 不可逆。

**计算示例：**

$$A = \begin{pmatrix} 3 & 1 \\ 5 & 2 \end{pmatrix}$$

$\det(A) = 3 \cdot 2 - 1 \cdot 5 = 6 - 5 = 1$，故

$$A^{-1} = \frac{1}{1} \begin{pmatrix} 2 & -1 \\ -5 & 3 \end{pmatrix} = \begin{pmatrix} 2 & -1 \\ -5 & 3 \end{pmatrix}$$

**验证：**

$$AA^{-1} = \begin{pmatrix} 3 & 1 \\ 5 & 2 \end{pmatrix}\begin{pmatrix} 2 & -1 \\ -5 & 3 \end{pmatrix} = \begin{pmatrix} 6-5 & -3+3 \\ 10-10 & -5+6 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I \quad \checkmark$$

### 初等行变换法

对于更大的矩阵，常用**增广矩阵**方法：将 $A$ 与单位矩阵 $I$ 并排构成增广矩阵 $[A \mid I]$，然后对其施行初等行变换，将左侧化为单位矩阵，此时右侧自动变为 $A^{-1}$：

$$[A \mid I] \xrightarrow{\text{初等行变换}} [I \mid A^{-1}]$$

**原理：** 每一步初等行变换等价于左乘一个初等矩阵 $E$。经过一系列变换 $E_k \cdots E_2 E_1$ 后，若左侧变为 $I$，则

$$E_k \cdots E_2 E_1 \cdot A = I \implies E_k \cdots E_2 E_1 = A^{-1}$$

同时右侧从 $I$ 变为 $E_k \cdots E_2 E_1 \cdot I = A^{-1}$。

**计算示例：** 求

$$A = \begin{pmatrix} 1 & 2 & 1 \\ 2 & 5 & 3 \\ 0 & 1 & 2 \end{pmatrix}$$

的逆矩阵。

构造增广矩阵并进行初等行变换：

$$\left(\begin{array}{ccc|ccc} 1 & 2 & 1 & 1 & 0 & 0 \\ 2 & 5 & 3 & 0 & 1 & 0 \\ 0 & 1 & 2 & 0 & 0 & 1 \end{array}\right)$$

$r_2 \leftarrow r_2 - 2r_1$：

$$\left(\begin{array}{ccc|ccc} 1 & 2 & 1 & 1 & 0 & 0 \\ 0 & 1 & 1 & -2 & 1 & 0 \\ 0 & 1 & 2 & 0 & 0 & 1 \end{array}\right)$$

$r_3 \leftarrow r_3 - r_2$：

$$\left(\begin{array}{ccc|ccc} 1 & 2 & 1 & 1 & 0 & 0 \\ 0 & 1 & 1 & -2 & 1 & 0 \\ 0 & 0 & 1 & 2 & -1 & 1 \end{array}\right)$$

$r_2 \leftarrow r_2 - r_3$，$r_1 \leftarrow r_1 - r_3$：

$$\left(\begin{array}{ccc|ccc} 1 & 2 & 0 & -1 & 1 & -1 \\ 0 & 1 & 0 & -4 & 2 & -1 \\ 0 & 0 & 1 & 2 & -1 & 1 \end{array}\right)$$

$r_1 \leftarrow r_1 - 2r_2$：

$$\left(\begin{array}{ccc|ccc} 1 & 0 & 0 & 7 & -3 & 1 \\ 0 & 1 & 0 & -4 & 2 & -1 \\ 0 & 0 & 1 & 2 & -1 & 1 \end{array}\right)$$

左侧已化为单位矩阵，因此

$$A^{-1} = \begin{pmatrix} 7 & -3 & 1 \\ -4 & 2 & -1 \\ 2 & -1 & 1 \end{pmatrix}$$

**判断不可逆：** 若化简过程中左侧出现全零行，则 $A$ 不可逆（奇异矩阵），停止计算。

### 伴随矩阵法（简介）

另一种方法是利用**伴随矩阵**（adjugate matrix）：

$$A^{-1} = \frac{1}{\det(A)} \cdot \text{adj}(A)$$

其中 $\text{adj}(A)$ 是 $A$ 的各代数余子式构成的矩阵的转置。这一方法在 $2 \times 2$ 情形下即为前述公式，对高阶矩阵计算量很大（需计算 $n^2$ 个行列式），一般仅用于理论推导，实际计算仍推荐初等行变换法。

---

## 6.3 逆矩阵的性质

### 基本性质

设 $A$、$B$ 均为可逆的 $n \times n$ 矩阵，$c \ne 0$ 为标量，则：

| 性质 | 公式 | 说明 |
|------|------|------|
| 双重逆 | $(A^{-1})^{-1} = A$ | 逆的逆还原 |
| 乘积逆 | $(AB)^{-1} = B^{-1}A^{-1}$ | 顺序反转 |
| 转置逆 | $(A^T)^{-1} = (A^{-1})^T$ | 逆与转置可交换 |
| 标量逆 | $(cA)^{-1} = \dfrac{1}{c}A^{-1}$ | 标量提出取倒数 |
| 幂次逆 | $(A^k)^{-1} = (A^{-1})^k$ | 记作 $A^{-k}$ |

**乘积逆的证明：**

$$(AB)(B^{-1}A^{-1}) = A(BB^{-1})A^{-1} = AIA^{-1} = AA^{-1} = I$$

同理 $(B^{-1}A^{-1})(AB) = I$，因此 $(AB)^{-1} = B^{-1}A^{-1}$。$\square$

**记忆口诀：** "穿衣服和脱衣服定律"——先穿袜子后穿鞋（$AB$），脱的时候要先脱鞋后脱袜子（$B^{-1}A^{-1}$）。这一规律对三个及更多矩阵同样成立：$(ABC)^{-1} = C^{-1}B^{-1}A^{-1}$。

**转置逆的证明：**

$$(A^{-1})^T \cdot A^T = (A \cdot A^{-1})^T = I^T = I$$

同理 $A^T \cdot (A^{-1})^T = I$，故 $(A^T)^{-1} = (A^{-1})^T$。$\square$

### 可逆矩阵的等价条件

对于 $n \times n$ 方阵 $A$，以下命题**互相等价**（任一成立则全部成立）：

1. $A$ 可逆（存在 $A^{-1}$）
2. $\det(A) \ne 0$（行列式不为零）
3. $A$ 的各行线性无关（行满秩）
4. $A$ 的各列线性无关（列满秩）
5. $\text{rank}(A) = n$（满秩）
6. $A\mathbf{x} = \mathbf{0}$ 只有零解 $\mathbf{x} = \mathbf{0}$（零空间平凡）
7. 对任意 $\mathbf{b}$，方程组 $A\mathbf{x} = \mathbf{b}$ 有唯一解
8. $A$ 可以通过初等行变换化为单位矩阵 $I$
9. $A$ 可以表示为若干初等矩阵的乘积

这些等价条件构成线性代数中最核心的"**可逆矩阵定理**"（Invertible Matrix Theorem），在后续章节中将逐步建立完整的理解。

---

## 6.4 利用逆矩阵求解方程组

### 形式解：$\mathbf{x} = A^{-1}\mathbf{b}$

若 $A$ 是可逆的 $n \times n$ 矩阵，则线性方程组 $A\mathbf{x} = \mathbf{b}$ 有唯一解：

$$A\mathbf{x} = \mathbf{b} \implies A^{-1}(A\mathbf{x}) = A^{-1}\mathbf{b} \implies (A^{-1}A)\mathbf{x} = A^{-1}\mathbf{b} \implies I\mathbf{x} = A^{-1}\mathbf{b} \implies \mathbf{x} = A^{-1}\mathbf{b}$$

**示例：** 利用前面求得的 $A^{-1}$ 求解 $A\mathbf{x} = \mathbf{b}$，其中

$$A = \begin{pmatrix} 1 & 2 & 1 \\ 2 & 5 & 3 \\ 0 & 1 & 2 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} 1 \\ 3 \\ 2 \end{pmatrix}$$

$$\mathbf{x} = A^{-1}\mathbf{b} = \begin{pmatrix} 7 & -3 & 1 \\ -4 & 2 & -1 \\ 2 & -1 & 1 \end{pmatrix}\begin{pmatrix} 1 \\ 3 \\ 2 \end{pmatrix} = \begin{pmatrix} 7 - 9 + 2 \\ -4 + 6 - 2 \\ 2 - 3 + 2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$$

### 为何实际中不常用逆矩阵求解方程组

尽管 $\mathbf{x} = A^{-1}\mathbf{b}$ 形式简洁，**实际数值计算中几乎不这样做**，原因如下：

**1. 计算量更大**

计算 $A^{-1}$ 的复杂度约为 $O(n^3)$，直接用高斯消元求解 $A\mathbf{x} = \mathbf{b}$ 同样是 $O(n^3)$，但高斯消元的常数因子更小（计算逆矩阵相当于对 $n$ 个右端向量分别做高斯消元）。

**2. 数值精度更差**

显式计算逆矩阵会引入更多浮点误差。对于接近奇异（条件数很大）的矩阵，直接求逆会使误差急剧放大，而 LU 分解等方法对同样的矩阵能给出更稳定的结果。

**3. 稀疏性被破坏**

许多工程问题（如偏微分方程离散化）产生的矩阵是**稀疏矩阵**（大多数元素为零），可以高效存储和运算。但稀疏矩阵的逆通常是**稠密矩阵**，存储开销从 $O(n)$ 变为 $O(n^2)$，完全失去了稀疏性的优势。

> **工程准则：** "永远不要显式计算逆矩阵来求解线性方程组。" 遇到 $A\mathbf{x} = \mathbf{b}$，应使用 `numpy.linalg.solve(A, b)` 或 `torch.linalg.solve(A, b)`，而非 `numpy.linalg.inv(A) @ b`。

---

## 6.5 矩阵求逆引理

### Sherman-Morrison 公式

若 $A$ 可逆，$\mathbf{u}, \mathbf{v}$ 为列向量，且 $1 + \mathbf{v}^T A^{-1} \mathbf{u} \neq 0$，则：

$$\boxed{(A + \mathbf{u}\mathbf{v}^T)^{-1} = A^{-1} - \frac{A^{-1}\mathbf{u}\mathbf{v}^T A^{-1}}{1 + \mathbf{v}^T A^{-1}\mathbf{u}}}$$

**意义**：当矩阵发生**秩1更新**（$A \to A + \mathbf{u}\mathbf{v}^T$）时，逆矩阵可以**增量更新**，无需重新计算，复杂度从 $O(n^3)$ 降为 $O(n^2)$。

### Woodbury 公式（一般化）

将秩1更新推广为秩 $k$ 更新。若 $A \in \mathbb{R}^{n \times n}$ 可逆，$U \in \mathbb{R}^{n \times k}$，$C \in \mathbb{R}^{k \times k}$ 可逆，$V \in \mathbb{R}^{k \times n}$，则：

$$(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}$$

**关键**：右侧只需求逆一个 $k \times k$ 的小矩阵 $C^{-1} + VA^{-1}U$（而非 $n \times n$）。

### 应用场景

- **在线学习**：每收到一个新样本，协方差矩阵发生秩1更新，用 Sherman-Morrison 增量更新逆
- **Kalman 滤波**：状态估计中的协方差矩阵更新
- **自然梯度**：Fisher 信息矩阵的近似求逆
- **岭回归**：$(X^TX + \lambda I)^{-1}$ 在 $\lambda$ 变化时的高效更新

---

## 本章小结

| 概念 | 核心内容 |
|------|---------|
| 逆矩阵定义 | $AB = BA = I$ 则 $B = A^{-1}$；仅对方阵有定义 |
| 可逆条件 | $\det(A) \ne 0$，即满秩（等价条件众多） |
| 2×2 公式 | $A^{-1} = \frac{1}{ad-bc}\begin{pmatrix}d & -b \\ -c & a\end{pmatrix}$ |
| 行变换法 | $[A \mid I] \to [I \mid A^{-1}]$ |
| 乘积逆 | $(AB)^{-1} = B^{-1}A^{-1}$（顺序反转） |
| 转置逆 | $(A^T)^{-1} = (A^{-1})^T$ |
| 方程组求解 | 形式解 $\mathbf{x} = A^{-1}\mathbf{b}$，但实际应用 LU 分解等方法 |

**核心要点回顾：**

1. 逆矩阵对应线性变换的"撤销"，奇异矩阵因降维而无法撤销。
2. 可逆的充要条件是 $\det(A) \ne 0$，与满秩、行列线性无关等条件等价。
3. 求逆的实用方法是初等行变换法：$[A \mid I] \to [I \mid A^{-1}]$。
4. 乘积的逆需要反转顺序：$(AB)^{-1} = B^{-1}A^{-1}$。
5. 实际中求解 $A\mathbf{x} = \mathbf{b}$ 应使用高斯消元或 LU 分解，而非显式求逆。

---

## 深度学习应用

### 背景：为何逆矩阵在深度学习中"既重要又危险"

逆矩阵在优化理论中扮演核心角色，但在大规模深度学习中几乎从不被显式计算。理解这一矛盾，能帮助我们深刻认识到数学理论与工程实践之间的距离。

### Newton 法中的 Hessian 逆

在经典优化中，**Newton 法**（牛顿法）利用目标函数的二阶信息加速收敛。对于函数 $f(\boldsymbol{\theta})$，Newton 更新步为：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H^{-1} \nabla f(\boldsymbol{\theta}_t)$$

其中 $H = \nabla^2 f(\boldsymbol{\theta}_t)$ 是**Hessian 矩阵**（目标函数的二阶偏导数矩阵）。$H^{-1}$ 将梯度方向按曲率进行缩放，使得更新步长在曲率大的方向小、在曲率小的方向大，从而比梯度下降收敛更快。

**问题：** 深度学习模型的参数量动辄百万到千亿，Hessian 矩阵的大小为参数数量的平方。一个仅有 $10^6$ 参数的小模型，其 Hessian 矩阵就有 $10^{12}$ 个元素，存储和求逆完全不现实。

### 自然梯度下降与 Fisher 信息矩阵

**自然梯度下降**（Natural Gradient Descent）是对普通梯度下降的改进，考虑了参数空间的几何结构：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \, F^{-1} \nabla \mathcal{L}(\boldsymbol{\theta}_t)$$

其中 $F$ 是**Fisher 信息矩阵**：

$$F = \mathbb{E}_{p(\mathbf{x}|\boldsymbol{\theta})}\!\left[\nabla \log p(\mathbf{x}|\boldsymbol{\theta}) \cdot \nabla \log p(\mathbf{x}|\boldsymbol{\theta})^T\right]$$

Fisher 矩阵衡量模型输出分布对参数变化的敏感程度，$F^{-1}$ 将梯度从欧式空间转换到"信息几何"空间，使更新与参数化方式无关。自然梯度是 K-FAC（Kronecker-Factored Approximate Curvature）等现代优化算法的理论基础。

### 为什么深度学习避免显式求逆

在实际深度学习中，涉及逆矩阵的操作通常被替换为以下方法：

**1. 共轭梯度法（Conjugate Gradient）**

不显式计算 $H^{-1}\mathbf{v}$，而是将 $H\mathbf{x} = \mathbf{v}$ 作为线性方程组迭代求解，只需要 Hessian-向量积（Hessian-vector product，HVP），可通过两次反向传播高效计算，无需存储完整 Hessian。

**2. 低秩近似（Low-Rank Approximation）**

K-FAC 将 Fisher 矩阵近似为 Kronecker 乘积之和，将 $O(n^2)$ 的存储降至 $O(n\sqrt{n})$，并利用 Kronecker 乘积的结构快速求逆。

**3. 批归一化的隐式效果**

批归一化（Batch Normalization）通过对激活值进行归一化，隐式地改善了 Hessian 的条件数，使得普通 Adam 优化器也能取得接近二阶方法的效果，从而绕开了直接求逆的必要性。

**4. 伪逆（Moore-Penrose Pseudoinverse）**

对于非方阵或奇异矩阵，可以使用**伪逆** $A^+$（通过奇异值分解实现），在某些迁移学习和神经切线核（NTK）理论中有应用。

### 代码示例（Python/NumPy & PyTorch）

```python
import numpy as np
import torch

# ── 示例1：逆矩阵的计算与验证 ─────────────────────────────────
A = np.array([[1, 2, 1],
              [2, 5, 3],
              [0, 1, 2]], dtype=float)

A_inv = np.linalg.inv(A)
print("A_inv =")
print(A_inv)
# [[ 7. -3.  1.]
#  [-4.  2. -1.]
#  [ 2. -1.  1.]]

# 验证 A @ A_inv ≈ I
print("A @ A_inv =")
print(np.round(A @ A_inv, decimals=10))  # 单位矩阵

# ── 示例2：正确做法 vs 错误做法 ──────────────────────────────
b = np.array([1, 3, 2], dtype=float)

# 错误做法：显式求逆（数值精度差，效率低）
x_bad = np.linalg.inv(A) @ b

# 正确做法：直接求解线性方程组（LU 分解，更快更稳定）
x_good = np.linalg.solve(A, b)

print(f"两种方法结果一致: {np.allclose(x_bad, x_good)}")  # True
print(f"解: x = {x_good}")  # [0. 0. 1.]

# ── 示例3：Newton 法更新步（小规模演示）────────────────────────
def newton_step(f, grad_f, hess_f, theta):
    """
    Newton 法一步更新：theta_new = theta - H^{-1} @ grad
    仅用于演示，实际中不应显式求 H^{-1}
    """
    g = grad_f(theta)
    H = hess_f(theta)
    # 正确做法：用 solve 代替 inv
    delta = np.linalg.solve(H, g)
    return theta - delta

# 最小化 f(x,y) = x^2 + 2y^2（最小值在原点）
def f(t):       return t[0]**2 + 2*t[1]**2
def grad_f(t):  return np.array([2*t[0], 4*t[1]])
def hess_f(t):  return np.array([[2, 0], [0, 4]])  # 常数 Hessian

theta = np.array([3.0, 3.0])
for i in range(5):
    theta = newton_step(f, grad_f, hess_f, theta)
    print(f"第{i+1}步: theta = {theta}, f = {f(theta):.6f}")
# Newton 法一步即收敛（二次函数的特殊性质）

# ── 示例4：条件数与数值稳定性 ─────────────────────────────────
# 良态矩阵（条件数小）
A_good = np.array([[2, 1], [1, 2]], dtype=float)
print(f"良态矩阵条件数: {np.linalg.cond(A_good):.2f}")  # 约 3

# 病态矩阵（条件数大，接近奇异）
A_bad = np.array([[1, 1], [1, 1+1e-10]], dtype=float)
print(f"病态矩阵条件数: {np.linalg.cond(A_bad):.2e}")  # 约 4e10

b_bad = np.array([2, 2+1e-10], dtype=float)
x_via_inv = np.linalg.inv(A_bad) @ b_bad
x_via_solve = np.linalg.solve(A_bad, b_bad)
print(f"inv 方法结果: {x_via_inv}")
print(f"solve 方法结果: {x_via_solve}")
# solve 比 inv 给出更可靠的数值结果

# ── 示例5：PyTorch 中的逆矩阵（GPU 加速） ──────────────────────
A_torch = torch.tensor([[3., 1.], [5., 2.]])

# 求逆
A_inv_torch = torch.linalg.inv(A_torch)
print(f"PyTorch 逆矩阵:\n{A_inv_torch}")

# 求解方程组（推荐方式）
b_torch = torch.tensor([1., 2.])
x_torch = torch.linalg.solve(A_torch, b_torch)
print(f"方程组的解: {x_torch}")
```

### 延伸阅读

- **《Deep Learning》**（Goodfellow et al.）第2章：线性代数，逆矩阵与伪逆的理论介绍
- **《Numerical Linear Algebra》**（Trefethen & Bau）：数值求逆的稳定性分析
- **K-FAC 论文**（Martens & Grosse, 2015）：Kronecker 因子化曲率近似，自然梯度的工程实现
- **「为什么你不应该对矩阵求逆」**（gregoire.xyz）：数值线性代数的工程实践建议

---

## 练习题

**练习 6.1（2×2 逆矩阵公式）**

对以下两个矩阵，判断是否可逆，若可逆则利用公式求其逆矩阵，并验证 $AA^{-1} = I$：

$$A = \begin{pmatrix} 4 & 7 \\ 2 & 6 \end{pmatrix}, \quad B = \begin{pmatrix} 2 & 4 \\ 1 & 2 \end{pmatrix}$$

---

**练习 6.2（初等行变换法）**

用初等行变换法求矩阵

$$C = \begin{pmatrix} 2 & 1 & 0 \\ 1 & 3 & 1 \\ 0 & 1 & 2 \end{pmatrix}$$

的逆矩阵，并验证结果。

---

**练习 6.3（逆矩阵性质）**

设 $A$ 和 $B$ 均为可逆的 $n \times n$ 矩阵，证明以下结论：

(a) $(A^{-1})^{-1} = A$

(b) $(AB)^{-1} = B^{-1}A^{-1}$

(c) 若 $AB = AC$，且 $A$ 可逆，则 $B = C$（可逆矩阵的"消去律"）

---

**练习 6.4（方程组与逆矩阵）**

已知

$$A = \begin{pmatrix} 1 & 0 \\ 2 & 1 \end{pmatrix}, \quad A^{-1} = \begin{pmatrix} 1 & 0 \\ -2 & 1 \end{pmatrix}$$

利用 $A^{-1}$ 求解以下方程组：

$$\begin{cases} x_1 = 3 \\ 2x_1 + x_2 = 5 \end{cases}$$

并说明为什么对于此类问题（每次只有一个右端向量），使用 $A^{-1}$ 与直接代入求解的计算量相同。

---

**练习 6.5（可逆性判断与综合应用）**

设

$$M = \begin{pmatrix} a & 0 & 0 \\ 1 & b & 0 \\ 2 & 3 & c \end{pmatrix}$$

为下三角矩阵。

(a) 求 $\det(M)$（下三角矩阵的行列式为主对角线元素之积）。

(b) 给出 $M$ 可逆的充要条件。

(c) 当 $a = 1, b = 2, c = 3$ 时，用初等行变换法求 $M^{-1}$。

---

## 练习答案

<details>
<summary>练习 6.1 答案</summary>

**矩阵 $A$：**

$$\det(A) = 4 \cdot 6 - 7 \cdot 2 = 24 - 14 = 10 \ne 0$$

$A$ 可逆：

$$A^{-1} = \frac{1}{10}\begin{pmatrix} 6 & -7 \\ -2 & 4 \end{pmatrix} = \begin{pmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{pmatrix}$$

验证：

$$AA^{-1} = \begin{pmatrix} 4 & 7 \\ 2 & 6 \end{pmatrix}\begin{pmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{pmatrix} = \begin{pmatrix} 2.4-1.4 & -2.8+2.8 \\ 1.2-1.2 & -1.4+2.4 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \checkmark$$

**矩阵 $B$：**

$$\det(B) = 2 \cdot 2 - 4 \cdot 1 = 4 - 4 = 0$$

$B$ 不可逆（奇异矩阵）。几何意义：$B$ 的第二行是第一行的 $\frac{1}{2}$，矩阵将平面压缩到一条直线上，无法逆变换。

</details>

<details>
<summary>练习 6.2 答案</summary>

构造增广矩阵：

$$\left(\begin{array}{ccc|ccc} 2 & 1 & 0 & 1 & 0 & 0 \\ 1 & 3 & 1 & 0 & 1 & 0 \\ 0 & 1 & 2 & 0 & 0 & 1 \end{array}\right)$$

交换 $r_1$ 和 $r_2$（使主元为1，减少分数）：

$$\left(\begin{array}{ccc|ccc} 1 & 3 & 1 & 0 & 1 & 0 \\ 2 & 1 & 0 & 1 & 0 & 0 \\ 0 & 1 & 2 & 0 & 0 & 1 \end{array}\right)$$

$r_2 \leftarrow r_2 - 2r_1$：

$$\left(\begin{array}{ccc|ccc} 1 & 3 & 1 & 0 & 1 & 0 \\ 0 & -5 & -2 & 1 & -2 & 0 \\ 0 & 1 & 2 & 0 & 0 & 1 \end{array}\right)$$

$r_2 \leftarrow r_2 + 5r_3$（消去 $r_2$ 的第2个主元，同时保持整数）：

$$\left(\begin{array}{ccc|ccc} 1 & 3 & 1 & 0 & 1 & 0 \\ 0 & 0 & 8 & 1 & -2 & 5 \\ 0 & 1 & 2 & 0 & 0 & 1 \end{array}\right)$$

交换 $r_2$ 和 $r_3$：

$$\left(\begin{array}{ccc|ccc} 1 & 3 & 1 & 0 & 1 & 0 \\ 0 & 1 & 2 & 0 & 0 & 1 \\ 0 & 0 & 8 & 1 & -2 & 5 \end{array}\right)$$

$r_3 \leftarrow r_3 / 8$：

$$\left(\begin{array}{ccc|ccc} 1 & 3 & 1 & 0 & 1 & 0 \\ 0 & 1 & 2 & 0 & 0 & 1 \\ 0 & 0 & 1 & 1/8 & -1/4 & 5/8 \end{array}\right)$$

$r_2 \leftarrow r_2 - 2r_3$，$r_1 \leftarrow r_1 - r_3$：

$$\left(\begin{array}{ccc|ccc} 1 & 3 & 0 & -1/8 & 5/4 & -5/8 \\ 0 & 1 & 0 & -1/4 & 1/2 & -1/4 \\ 0 & 0 & 1 & 1/8 & -1/4 & 5/8 \end{array}\right)$$

$r_1 \leftarrow r_1 - 3r_2$：

$$\left(\begin{array}{ccc|ccc} 1 & 0 & 0 & 5/8 & -1/4 & 1/8 \\ 0 & 1 & 0 & -1/4 & 1/2 & -1/4 \\ 0 & 0 & 1 & 1/8 & -1/4 & 5/8 \end{array}\right)$$

$$C^{-1} = \frac{1}{8}\begin{pmatrix} 5 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 5 \end{pmatrix}$$

验证（可用 NumPy）：`np.allclose(C @ C_inv, np.eye(3))` 返回 `True`。

</details>

<details>
<summary>练习 6.3 答案</summary>

**(a)** $(A^{-1})^{-1} = A$

由逆矩阵定义，$A \cdot A^{-1} = A^{-1} \cdot A = I$。

这说明 $A$ 满足"$A^{-1}$ 的逆矩阵"的定义条件，因此 $(A^{-1})^{-1} = A$。$\square$

**(b)** $(AB)^{-1} = B^{-1}A^{-1}$

验证等式右侧确实是 $AB$ 的逆：

$$(AB)(B^{-1}A^{-1}) = A(BB^{-1})A^{-1} = AIA^{-1} = AA^{-1} = I$$

$$(B^{-1}A^{-1})(AB) = B^{-1}(A^{-1}A)B = B^{-1}IB = B^{-1}B = I$$

由逆矩阵的唯一性，$(AB)^{-1} = B^{-1}A^{-1}$。$\square$

**(c)** 若 $AB = AC$ 且 $A$ 可逆，则 $B = C$

左乘 $A^{-1}$：

$$A^{-1}(AB) = A^{-1}(AC) \implies (A^{-1}A)B = (A^{-1}A)C \implies IB = IC \implies B = C \quad \square$$

**注意：** 若 $A$ 不可逆，此消去律不成立。例如取 $A = \begin{pmatrix}0 & 0 \\ 0 & 0\end{pmatrix}$，$B = \begin{pmatrix}1 & 0 \\ 0 & 0\end{pmatrix}$，$C = \begin{pmatrix}0 & 0 \\ 0 & 1\end{pmatrix}$，则 $AB = AC = O$，但 $B \ne C$。

</details>

<details>
<summary>练习 6.4 答案</summary>

方程组写成矩阵形式 $A\mathbf{x} = \mathbf{b}$：

$$\begin{pmatrix} 1 & 0 \\ 2 & 1 \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 3 \\ 5 \end{pmatrix}$$

利用 $\mathbf{x} = A^{-1}\mathbf{b}$：

$$\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ -2 & 1 \end{pmatrix}\begin{pmatrix} 3 \\ 5 \end{pmatrix} = \begin{pmatrix} 1 \cdot 3 + 0 \cdot 5 \\ -2 \cdot 3 + 1 \cdot 5 \end{pmatrix} = \begin{pmatrix} 3 \\ -1 \end{pmatrix}$$

**计算量对比：**

- 直接代入（前向替代）：先由第一个方程得 $x_1 = 3$，代入第二个方程得 $x_2 = 5 - 2 \times 3 = -1$，共 2 次乘法和 1 次减法。
- 使用 $A^{-1}$（矩阵-向量乘）：同样需要 $n$ 次内积，即 $n^2$ 次乘法。

当只有一个右端向量时，用 $A^{-1}$ 并无效率优势。$A^{-1}$ 的优势在于**多个右端向量复用**：提前花 $O(n^3)$ 计算好 $A^{-1}$，之后每个新的 $\mathbf{b}$ 只需 $O(n^2)$ 即可得解。即便如此，LU 分解也能以更低的数值误差达到同样效果。

</details>

<details>
<summary>练习 6.5 答案</summary>

**(a)** 下三角矩阵的行列式等于主对角线元素之积：

$$\det(M) = a \cdot b \cdot c$$

**(b)** $M$ 可逆的充要条件为 $\det(M) \ne 0$，即

$$a \ne 0 \quad \text{且} \quad b \ne 0 \quad \text{且} \quad c \ne 0$$

三个主对角线元素均不为零。

**(c)** 当 $a=1, b=2, c=3$ 时：

$$M = \begin{pmatrix} 1 & 0 & 0 \\ 1 & 2 & 0 \\ 2 & 3 & 3 \end{pmatrix}$$

构造增广矩阵：

$$\left(\begin{array}{ccc|ccc} 1 & 0 & 0 & 1 & 0 & 0 \\ 1 & 2 & 0 & 0 & 1 & 0 \\ 2 & 3 & 3 & 0 & 0 & 1 \end{array}\right)$$

$r_2 \leftarrow r_2 - r_1$，$r_3 \leftarrow r_3 - 2r_1$：

$$\left(\begin{array}{ccc|ccc} 1 & 0 & 0 & 1 & 0 & 0 \\ 0 & 2 & 0 & -1 & 1 & 0 \\ 0 & 3 & 3 & -2 & 0 & 1 \end{array}\right)$$

$r_2 \leftarrow r_2 / 2$：

$$\left(\begin{array}{ccc|ccc} 1 & 0 & 0 & 1 & 0 & 0 \\ 0 & 1 & 0 & -1/2 & 1/2 & 0 \\ 0 & 3 & 3 & -2 & 0 & 1 \end{array}\right)$$

$r_3 \leftarrow r_3 - 3r_2$：

$$\left(\begin{array}{ccc|ccc} 1 & 0 & 0 & 1 & 0 & 0 \\ 0 & 1 & 0 & -1/2 & 1/2 & 0 \\ 0 & 0 & 3 & -1/2 & -3/2 & 1 \end{array}\right)$$

$r_3 \leftarrow r_3 / 3$：

$$\left(\begin{array}{ccc|ccc} 1 & 0 & 0 & 1 & 0 & 0 \\ 0 & 1 & 0 & -1/2 & 1/2 & 0 \\ 0 & 0 & 1 & -1/6 & -1/2 & 1/3 \end{array}\right)$$

$$M^{-1} = \begin{pmatrix} 1 & 0 & 0 \\ -1/2 & 1/2 & 0 \\ -1/6 & -1/2 & 1/3 \end{pmatrix}$$

**规律：** 下三角矩阵的逆仍为下三角矩阵，主对角线元素为原对角线元素的倒数。

</details>
