# 第17章：正交化与QR分解

> 正交基是向量空间中最"舒适"的坐标系——每个方向都相互垂直，计算简洁，数值稳定。Gram-Schmidt正交化算法将任意基变换为正交基，而QR分解将这一过程编码成矩阵语言，成为数值线性代数的核心工具之一。

---

## 学习目标

完成本章学习后，你将能够：

- 理解正交投影的定义，计算向量在向量或子空间上的投影
- 掌握Gram-Schmidt正交化算法的步骤与几何直觉，将任意线性无关组化为标准正交基
- 理解QR分解的构造方式、存在唯一性及其与Gram-Schmidt的关系
- 利用QR分解求解线性方程组与最小二乘问题
- 了解修正Gram-Schmidt和Householder反射的数值稳定性优势

---

## 17.1 正交投影

### 17.1.1 向量在向量上的投影

设 $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$，$\mathbf{b} \neq \mathbf{0}$。我们希望找到与 $\mathbf{b}$ 同方向的向量，使其与 $\mathbf{a}$ 的差垂直于 $\mathbf{b}$。

**几何直觉：** 想象 $\mathbf{a}$ 是空间中一个点，$\mathbf{b}$ 确定了一条直线方向。把 $\mathbf{a}$ "投影到"这条直线上，就是从 $\mathbf{a}$ 向直线作垂线，垂足对应的向量就是投影。

**推导：** 设投影为 $\text{proj}_{\mathbf{b}} \mathbf{a} = c\mathbf{b}$（与 $\mathbf{b}$ 共线），要求残差 $\mathbf{a} - c\mathbf{b}$ 垂直于 $\mathbf{b}$：

$$(\mathbf{a} - c\mathbf{b}) \cdot \mathbf{b} = 0 \implies \mathbf{a} \cdot \mathbf{b} - c(\mathbf{b} \cdot \mathbf{b}) = 0 \implies c = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}}$$

**定义（向量在向量上的正交投影）：**

$$\boxed{\text{proj}_{\mathbf{b}} \mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}} \mathbf{b} = \frac{\langle \mathbf{a}, \mathbf{b} \rangle}{\langle \mathbf{b}, \mathbf{b} \rangle} \mathbf{b}}$$

**残差（正交分量）：**

$$\mathbf{a}^{\perp} = \mathbf{a} - \text{proj}_{\mathbf{b}} \mathbf{a}$$

满足 $\mathbf{a}^{\perp} \perp \mathbf{b}$，且 $\mathbf{a} = \text{proj}_{\mathbf{b}} \mathbf{a} + \mathbf{a}^{\perp}$（正交分解）。

若 $\mathbf{b}$ 已是单位向量 $\hat{\mathbf{b}}$（即 $\|\mathbf{b}\| = 1$），公式化简为：

$$\text{proj}_{\hat{\mathbf{b}}} \mathbf{a} = (\mathbf{a} \cdot \hat{\mathbf{b}}) \hat{\mathbf{b}}$$

标量 $\mathbf{a} \cdot \hat{\mathbf{b}}$ 称为 $\mathbf{a}$ 在 $\hat{\mathbf{b}}$ 方向上的**标量投影**（scalar projection），也叫分量（component）。

**例子：** 设 $\mathbf{a} = (3, 1)^T$，$\mathbf{b} = (1, 1)^T$，求 $\text{proj}_{\mathbf{b}} \mathbf{a}$。

$$c = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}} = \frac{3 \cdot 1 + 1 \cdot 1}{1^2 + 1^2} = \frac{4}{2} = 2$$

$$\text{proj}_{\mathbf{b}} \mathbf{a} = 2\begin{pmatrix}1\\1\end{pmatrix} = \begin{pmatrix}2\\2\end{pmatrix}, \quad \mathbf{a}^{\perp} = \begin{pmatrix}3\\1\end{pmatrix} - \begin{pmatrix}2\\2\end{pmatrix} = \begin{pmatrix}1\\-1\end{pmatrix}$$

验证：$\mathbf{a}^{\perp} \cdot \mathbf{b} = 1 \cdot 1 + (-1) \cdot 1 = 0$。✓

### 17.1.2 投影矩阵

投影 $\text{proj}_{\mathbf{b}} \mathbf{a}$ 对 $\mathbf{a}$ 是线性的，因此可以用矩阵表示：

$$\text{proj}_{\mathbf{b}} \mathbf{a} = \frac{\mathbf{b} \mathbf{b}^T}{\mathbf{b}^T \mathbf{b}} \mathbf{a} = P \mathbf{a}$$

其中**投影矩阵**（projection matrix）为：

$$P = \frac{\mathbf{b} \mathbf{b}^T}{\mathbf{b}^T \mathbf{b}}$$

投影矩阵的两个重要性质：
- **对称性**：$P^T = P$
- **幂等性**：$P^2 = P$（投影两次等于投影一次——已经在直线上了，再投影不变）

### 17.1.3 向量在子空间上的投影

设 $W \subseteq \mathbb{R}^n$ 是一个子空间，有正交基 $\{q_1, q_2, \ldots, q_k\}$（两两正交的单位向量）。向量 $\mathbf{a}$ 在 $W$ 上的正交投影为：

$$\text{proj}_{W} \mathbf{a} = \sum_{i=1}^{k} (\mathbf{a} \cdot \mathbf{q}_i) \mathbf{q}_i$$

**关键性质：** 残差 $\mathbf{a} - \text{proj}_{W} \mathbf{a}$ 垂直于 $W$ 中每一个向量，即垂直于整个子空间 $W$。

以正交基列组成的矩阵 $Q = [q_1 \mid q_2 \mid \cdots \mid q_k]$（$n \times k$ 矩阵），投影可以写成矩阵形式：

$$\text{proj}_{W} \mathbf{a} = QQ^T \mathbf{a}$$

$P_W = QQ^T$ 是投影到 $W$ 的投影矩阵，同样满足 $P_W^T = P_W$ 和 $P_W^2 = P_W$。

---

## 17.2 Gram-Schmidt正交化

### 17.2.1 问题的提出

给定 $\mathbb{R}^n$ 中 $k$ 个线性无关的向量 $\{a_1, a_2, \ldots, a_k\}$，我们希望找到同一子空间的一组**标准正交基**（orthonormal basis）$\{q_1, q_2, \ldots, q_k\}$，使得：

$$\langle q_i, q_j \rangle = \delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

（两两正交且每个向量都是单位向量。）

**为什么需要标准正交基？**

在标准正交基下，所有坐标计算只需做内积，不用解方程组。投影公式最简，计算最稳定。

### 17.2.2 Gram-Schmidt算法

**核心思想：** 每次处理一个新向量时，先减去它在已有正交向量方向上的分量（消除"重叠"），然后归一化。

**算法步骤（产生标准正交基）：**

**第1步：** 取 $\mathbf{v}_1 = a_1$，归一化得 $q_1 = \dfrac{\mathbf{v}_1}{\|\mathbf{v}_1\|}$。

**第2步：** 从 $a_2$ 中减去其在 $q_1$ 方向上的分量：

$$\mathbf{v}_2 = a_2 - (a_2 \cdot q_1)q_1$$

归一化：$q_2 = \dfrac{\mathbf{v}_2}{\|\mathbf{v}_2\|}$。

**第3步：** 从 $a_3$ 中减去其在 $q_1$、$q_2$ 方向上的分量：

$$\mathbf{v}_3 = a_3 - (a_3 \cdot q_1)q_1 - (a_3 \cdot q_2)q_2$$

归一化：$q_3 = \dfrac{\mathbf{v}_3}{\|\mathbf{v}_3\|}$。

**一般第 $j$ 步：**

$$\mathbf{v}_j = a_j - \sum_{i=1}^{j-1} (a_j \cdot q_i) q_i, \qquad q_j = \frac{\mathbf{v}_j}{\|\mathbf{v}_j\|}$$

**几何直觉：** 第 $j$ 步是把 $a_j$ 中所有能被已有正交方向 $q_1, \ldots, q_{j-1}$ "解释"的部分全部去掉，剩下的 $\mathbf{v}_j$ 就是 $a_j$ 中"全新的、与已有方向无关的"分量，代表了 $a_j$ 对子空间的独特贡献。

```
二维示例（几何图示）：

  a₂
   ↑   \
   |    \  v₂ = a₂ - proj_{q₁}(a₂)
   |     \       ↑
   |      \    （垂直于q₁）
   |    proj↗
   +-------→ q₁ (归一化后的 a₁ 方向)
```

### 17.2.3 完整例子

设 $a_1 = (1, 1, 0)^T$，$a_2 = (1, 0, 1)^T$，$a_3 = (0, 1, 1)^T$（$\mathbb{R}^3$ 中线性无关的三个向量）。

**第1步：**

$$\mathbf{v}_1 = a_1 = \begin{pmatrix}1\\1\\0\end{pmatrix}, \quad \|\mathbf{v}_1\| = \sqrt{2}, \quad q_1 = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\1\\0\end{pmatrix}$$

**第2步：**

$$a_2 \cdot q_1 = \frac{1}{\sqrt{2}}(1 \cdot 1 + 0 \cdot 1 + 1 \cdot 0) = \frac{1}{\sqrt{2}}$$

$$\mathbf{v}_2 = a_2 - \frac{1}{\sqrt{2}} q_1 = \begin{pmatrix}1\\0\\1\end{pmatrix} - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix}1\\1\\0\end{pmatrix} = \begin{pmatrix}1\\0\\1\end{pmatrix} - \begin{pmatrix}1/2\\1/2\\0\end{pmatrix} = \begin{pmatrix}1/2\\-1/2\\1\end{pmatrix}$$

$$\|\mathbf{v}_2\| = \sqrt{1/4 + 1/4 + 1} = \sqrt{3/2} = \frac{\sqrt{6}}{2}$$

$$q_2 = \frac{2}{\sqrt{6}}\begin{pmatrix}1/2\\-1/2\\1\end{pmatrix} = \frac{1}{\sqrt{6}}\begin{pmatrix}1\\-1\\2\end{pmatrix}$$

**第3步：**

$$a_3 \cdot q_1 = \frac{1}{\sqrt{2}}(0+1+0) = \frac{1}{\sqrt{2}}, \quad a_3 \cdot q_2 = \frac{1}{\sqrt{6}}(0-1+2) = \frac{1}{\sqrt{6}}$$

$$\mathbf{v}_3 = a_3 - \frac{1}{\sqrt{2}} q_1 - \frac{1}{\sqrt{6}} q_2 = \begin{pmatrix}0\\1\\1\end{pmatrix} - \frac{1}{2}\begin{pmatrix}1\\1\\0\end{pmatrix} - \frac{1}{6}\begin{pmatrix}1\\-1\\2\end{pmatrix} = \begin{pmatrix}-2/3\\2/3\\2/3\end{pmatrix}$$

$$\|\mathbf{v}_3\| = \sqrt{4/9+4/9+4/9} = \frac{2\sqrt{3}}{3} = \frac{2}{\sqrt{3}}$$

$$q_3 = \frac{\sqrt{3}}{2} \cdot \frac{1}{3}\begin{pmatrix}-2\\2\\2\end{pmatrix} = \frac{1}{\sqrt{3}}\begin{pmatrix}-1\\1\\1\end{pmatrix}$$

**验证正交性：**

$$q_1 \cdot q_2 = \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{6}}(1 \cdot 1 + 1 \cdot (-1) + 0 \cdot 2) = 0 \checkmark$$

$$q_1 \cdot q_3 = \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{3}}(-1 + 1 + 0) = 0 \checkmark$$

$$q_2 \cdot q_3 = \frac{1}{\sqrt{6}} \cdot \frac{1}{\sqrt{3}}(-1 - 1 + 2) = 0 \checkmark$$

---

## 17.3 QR分解

### 17.3.1 从Gram-Schmidt到QR

仔细观察Gram-Schmidt过程，每个 $a_j$ 可以用已构造的 $q_1, \ldots, q_j$ 来表示：

$$a_j = (a_j \cdot q_1) q_1 + (a_j \cdot q_2) q_2 + \cdots + (a_j \cdot q_{j-1}) q_{j-1} + \|\mathbf{v}_j\| q_j$$

写成内积系数：令 $r_{ij} = a_j \cdot q_i$（$i < j$），$r_{jj} = \|\mathbf{v}_j\| > 0$，则：

$$a_j = r_{1j} q_1 + r_{2j} q_2 + \cdots + r_{jj} q_j = \sum_{i=1}^{j} r_{ij} q_i$$

用矩阵形式表示，设 $A = [a_1 \mid a_2 \mid \cdots \mid a_k]$（$n \times k$ 矩阵），$Q = [q_1 \mid q_2 \mid \cdots \mid q_k]$（$n \times k$ 矩阵），则：

$$A = QR$$

其中 $R$ 是 $k \times k$ **上三角矩阵**（upper triangular matrix）：

$$R = \begin{pmatrix}
r_{11} & r_{12} & r_{13} & \cdots & r_{1k} \\
0      & r_{22} & r_{23} & \cdots & r_{2k} \\
0      & 0      & r_{33} & \cdots & r_{3k} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0      & 0      & 0      & \cdots & r_{kk}
\end{pmatrix}$$

对角元 $r_{jj} = \|\mathbf{v}_j\| > 0$（因为 $a_j$ 线性无关，所以 $\mathbf{v}_j \neq 0$）。

### 17.3.2 QR分解定理

**定理：** 设 $A$ 是 $m \times n$（$m \geq n$）的矩阵，且 $A$ 的列向量线性无关，则 $A$ 有唯一的分解：

$$A = QR$$

其中：
- $Q$ 是 $m \times n$ 矩阵，满足 $Q^T Q = I_n$（列向量构成标准正交组）
- $R$ 是 $n \times n$ 上三角矩阵，对角元素均为**正数**

**唯一性说明：** 若不要求 $R$ 对角元素为正，则分解不唯一（可以对每列乘以 $-1$ 并同时改变 $R$ 的符号）。加上正对角元约束后，分解唯一。

**方形矩阵的情形：** 若 $A$ 是 $n \times n$ 可逆矩阵，则 $Q$ 是正交矩阵（$Q^T Q = QQ^T = I_n$），$R$ 是正对角元的可逆上三角矩阵。

### 17.3.3 QR分解示例

用17.2.3节的结果。设 $A = [a_1 \mid a_2 \mid a_3]$ 是 $3 \times 3$ 矩阵，已得标准正交基 $\{q_1, q_2, q_3\}$，则：

$$Q = \begin{pmatrix} 1/\sqrt{2} & 1/\sqrt{6} & -1/\sqrt{3} \\ 1/\sqrt{2} & -1/\sqrt{6} & 1/\sqrt{3} \\ 0 & 2/\sqrt{6} & 1/\sqrt{3} \end{pmatrix}$$

$$R = \begin{pmatrix} r_{11} & r_{12} & r_{13} \\ 0 & r_{22} & r_{23} \\ 0 & 0 & r_{33} \end{pmatrix} = Q^T A$$

由Gram-Schmidt过程直接读出（或计算 $R = Q^T A$）：

$$r_{11} = \|\mathbf{v}_1\| = \sqrt{2}, \quad r_{12} = a_2 \cdot q_1 = \frac{1}{\sqrt{2}}, \quad r_{13} = a_3 \cdot q_1 = \frac{1}{\sqrt{2}}$$

$$r_{22} = \|\mathbf{v}_2\| = \frac{\sqrt{6}}{2}, \quad r_{23} = a_3 \cdot q_2 = \frac{1}{\sqrt{6}}$$

$$r_{33} = \|\mathbf{v}_3\| = \frac{2}{\sqrt{3}}$$

**验证：** $Q^T Q = I_3$（因 $q_1, q_2, q_3$ 标准正交），$QR = A$（由Gram-Schmidt的展开式）。

### 17.3.4 实用计算：$R = Q^T A$

已知 $A = QR$ 且 $Q^T Q = I$，两边左乘 $Q^T$：

$$Q^T A = Q^T(QR) = (Q^T Q)R = IR = R$$

因此 **$R$ 可以直接由 $R = Q^T A$ 计算**，这避免了手动跟踪Gram-Schmidt系数。

---

## 17.4 QR分解的应用

### 17.4.1 求解线性方程组

设 $A = QR$（$A$ 是 $n \times n$ 可逆矩阵），求解 $A\mathbf{x} = \mathbf{b}$：

$$QR\mathbf{x} = \mathbf{b} \implies R\mathbf{x} = Q^T\mathbf{b}$$

（利用 $Q^{-1} = Q^T$，左乘 $Q^T$。）

此时 $R\mathbf{x} = Q^T\mathbf{b}$ 是上三角方程组，可用**回代法（back substitution）** 在 $O(n^2)$ 时间内求解。

**与LU分解相比：** QR分解在数值上更稳定（尤其对病态矩阵），代价是计算量略大（大约是LU的两倍）。

### 17.4.2 最小二乘问题

**问题：** 方程组 $A\mathbf{x} = \mathbf{b}$ 无解（$A$ 是 $m \times n$，$m > n$，超定方程组），求最小化 $\|A\mathbf{x} - \mathbf{b}\|^2$ 的解 $\hat{\mathbf{x}}$。

**法方程（normal equations）：** $A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$。

**用QR分解求解：** 设 $A = QR$（$Q$ 是 $m \times n$，$Q^TQ = I_n$），代入法方程：

$$(QR)^T (QR) \hat{\mathbf{x}} = (QR)^T \mathbf{b}$$

$$R^T Q^T Q R \hat{\mathbf{x}} = R^T Q^T \mathbf{b}$$

$$R^T R \hat{\mathbf{x}} = R^T Q^T \mathbf{b}$$

因 $R$ 可逆（对角元均为正数），$R^T$ 也可逆，两边左乘 $(R^T)^{-1}$：

$$\boxed{R \hat{\mathbf{x}} = Q^T \mathbf{b}}$$

同样是上三角方程组，用回代法求解。

**算法流程：**

1. 对 $A$ 做QR分解：$A = QR$
2. 计算右端：$\mathbf{c} = Q^T \mathbf{b}$（仅取前 $n$ 行）
3. 回代求解：$R\hat{\mathbf{x}} = \mathbf{c}$

**优势：** 避免了直接计算 $A^T A$（可能引入较大的条件数），数值稳定性更好。

### 17.4.3 例子：线性回归

设有3个数据点 $(0, 1), (1, 2), (2, 2.5)$，拟合直线 $y = \beta_0 + \beta_1 x$。

设计矩阵和观测向量：

$$A = \begin{pmatrix}1 & 0\\1 & 1\\1 & 2\end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix}1\\2\\2.5\end{pmatrix}$$

对 $A$ 做QR分解，$Q^T \mathbf{b}$ 给出右端项，再通过回代解 $R\hat{\mathbf{x}} = Q^T\mathbf{b}$，即得最优系数 $(\hat{\beta}_0, \hat{\beta}_1)^T$。

---

## 17.5 数值稳定性

### 17.5.1 经典Gram-Schmidt的问题

在有限精度浮点运算中，经典Gram-Schmidt算法（Classical Gram-Schmidt, CGS）存在**舍入误差累积**的问题：

每步减去投影时，浮点数运算会引入微小误差，这些误差会传播并放大。当矩阵的列向量接近线性相关（即矩阵条件数很大）时，最终得到的 $q_j$ 彼此之间可能不再精确正交，$\|Q^TQ - I\|$ 可能显著偏离零。

### 17.5.2 修正Gram-Schmidt（Modified Gram-Schmidt, MGS）

**改进思想：** 不是用原始 $a_j$ 一次性减去所有投影，而是在每步减去一个投影后**立即更新**向量，再进行下一步。

**算法差异：**

经典 Gram-Schmidt（CGS）第 $j$ 步：
$$\mathbf{v}_j = a_j - \sum_{i=1}^{j-1} (a_j \cdot q_i) q_i \quad \text{（一次性减去所有分量）}$$

修正 Gram-Schmidt（MGS）第 $j$ 步：
$$\mathbf{v}_j^{(0)} = a_j$$
$$\mathbf{v}_j^{(i)} = \mathbf{v}_j^{(i-1)} - (\mathbf{v}_j^{(i-1)} \cdot q_i) q_i, \quad i = 1, 2, \ldots, j-1$$
$$\mathbf{v}_j = \mathbf{v}_j^{(j-1)}$$

**数学等价，数值不等价：** 在精确算术下，CGS和MGS产生完全相同的结果。但在浮点运算中，MGS通过逐步正交化，使每次减去的投影更准确，正交性误差从 $O(\epsilon_{\text{mach}} \cdot \kappa(A)^2)$（CGS）降低到 $O(\epsilon_{\text{mach}} \cdot \kappa(A))$（MGS）。

### 17.5.3 Householder反射

对于要求更高数值精度的场景（如大规模科学计算），实践中更常用 **Householder反射**（Householder reflections）来实现QR分解。

**基本思想：** 利用Householder反射矩阵（也称镜面反射矩阵）：

$$H = I - 2\mathbf{u}\mathbf{u}^T, \quad \|\mathbf{u}\| = 1$$

$H$ 满足 $H^T = H$，$H^T H = I$（$H$ 是正交矩阵，也是对称矩阵）。可以选取适当的单位向量 $\mathbf{u}$，使得 $H\mathbf{a}$ 的结果只有第一个分量非零——从而将向量"反射"成正方向的倍数。

通过依次应用 $n$ 个Householder反射，将 $A$ 化为上三角矩阵：

$$H_n \cdots H_2 H_1 A = R \implies A = H_1^T H_2^T \cdots H_n^T R = QR$$

**Householder vs MGS：**

| 方法 | 正交性误差阶 | 计算复杂度 | 适用场景 |
|:---|:---:|:---:|:---|
| 经典GS (CGS) | $O(\epsilon \kappa^2)$ | $O(mn^2)$ | 不推荐用于实际计算 |
| 修正GS (MGS) | $O(\epsilon \kappa)$ | $O(mn^2)$ | 中等精度要求，迭代算法 |
| Householder | $O(\epsilon)$ | $O(mn^2)$ | 高精度要求，LAPACK默认 |

其中 $\epsilon = \epsilon_{\text{mach}}$ 为机器精度，$\kappa = \kappa(A)$ 为矩阵的条件数。

LAPACK（以及NumPy/SciPy/PyTorch底层）默认使用Householder反射实现QR分解，因为它在数值上是后向稳定（backward stable）的。

---

## 本章小结

- **正交投影** $\text{proj}_{\mathbf{b}} \mathbf{a} = \dfrac{\langle \mathbf{a}, \mathbf{b} \rangle}{\langle \mathbf{b}, \mathbf{b} \rangle} \mathbf{b}$ 将向量分解为"与 $\mathbf{b}$ 平行"和"与 $\mathbf{b}$ 垂直"两部分；推广到子空间时，投影矩阵为 $P_W = QQ^T$（$Q$ 的列是 $W$ 的标准正交基）。

- **Gram-Schmidt算法** 每步从新向量 $a_j$ 中减去其在已有标准正交向量 $q_1, \ldots, q_{j-1}$ 上的投影，得到与它们正交的分量，归一化后得到 $q_j$；产生的 $\{q_1, \ldots, q_k\}$ 与原向量张成同一子空间。

- **QR分解** $A = QR$ 是Gram-Schmidt过程的矩阵编码：$Q$ 的列是标准正交基，$R$ 是正对角元的上三角矩阵；对列满秩矩阵，在要求 $R$ 对角元正的条件下，分解唯一。

- **QR的应用**：求解方程组 $A\mathbf{x} = \mathbf{b}$ 等价于回代求解 $R\mathbf{x} = Q^T\mathbf{b}$；最小二乘问题 $\min\|A\mathbf{x} - \mathbf{b}\|$ 同样归结为 $R\hat{\mathbf{x}} = Q^T\mathbf{b}$，两者均比直接方法数值更稳定。

- **数值稳定性**：修正Gram-Schmidt（MGS）将正交性误差从 $O(\epsilon\kappa^2)$ 降至 $O(\epsilon\kappa)$；Householder反射实现的QR分解误差仅 $O(\epsilon)$，是实践中的首选方法，被LAPACK等数值库采用。

---

## 深度学习应用：正交权重与谱归一化

### 背景

在深度学习中，矩阵的"形状"对训练稳定性有深刻影响。正交矩阵和近正交矩阵拥有特殊的几何性质，使其在神经网络设计中发挥重要作用。

### 17.6.1 正交权重矩阵的优势

设权重矩阵 $W \in \mathbb{R}^{n \times n}$ 是正交矩阵（$W^TW = I$），对任意向量 $\mathbf{x}$：

$$\|W\mathbf{x}\| = \|Q\mathbf{x}\| = \sqrt{(Q\mathbf{x})^T(Q\mathbf{x})} = \sqrt{\mathbf{x}^T Q^T Q \mathbf{x}} = \sqrt{\mathbf{x}^T \mathbf{x}} = \|\mathbf{x}\|$$

**正交矩阵是等距变换（isometry）**：它保持向量的长度不变，只做旋转和反射。

这对神经网络训练有两大好处：

**1. 梯度流稳定（Gradient Flow）**

反向传播时，梯度通过权重矩阵传播：

$$\nabla_{\mathbf{x}} \mathcal{L} = W^T \nabla_{\mathbf{z}} \mathcal{L}$$

若 $W$ 是正交矩阵，则 $\|W^T\mathbf{g}\| = \|\mathbf{g}\|$，梯度的范数在每层传播时保持不变——既不爆炸，也不消失。这是深层网络（如100层以上）稳定训练的关键。

对于非正交的 $W$，设其奇异值为 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_n$：
- 若 $\sigma_1 \gg 1$：梯度爆炸
- 若 $\sigma_n \ll 1$：梯度消失
- 若所有 $\sigma_i = 1$（即 $W$ 正交）：梯度既不放大也不缩小

**2. Lipschitz约束**

函数 $f: \mathbb{R}^n \to \mathbb{R}^m$ 满足Lipschitz条件（Lipschitz constant = $L$）当且仅当：

$$\|f(\mathbf{x}) - f(\mathbf{y})\| \leq L \|\mathbf{x} - \mathbf{y}\|, \quad \forall \mathbf{x}, \mathbf{y}$$

对线性映射 $f(\mathbf{x}) = W\mathbf{x}$，其Lipschitz常数等于 $W$ 的**谱范数**（spectral norm，即最大奇异值）：

$$\text{Lip}(f) = \|W\|_2 = \sigma_{\max}(W)$$

若 $W$ 正交，$\sigma_{\max} = 1$，故 Lipschitz 常数为 1——映射不放大距离，保证了函数的"规律性"。

### 17.6.2 谱归一化（Spectral Normalization）

**核心思想：** 将权重矩阵 $W$ 除以其谱范数，强制 Lipschitz 常数等于1：

$$\hat{W} = \frac{W}{\sigma_{\max}(W)} = \frac{W}{\|W\|_2}$$

这样 $\|\hat{W}\|_2 = 1$，整个网络的 Lipschitz 常数有界。

**高效计算谱范数——幂迭代法（Power Iteration）：**

精确计算 $\sigma_{\max}(W)$ 需要完整的SVD，代价为 $O(\min(m,n) \cdot mn)$。谱归一化使用**幂迭代**近似：

初始化 $\tilde{\mathbf{v}} \in \mathbb{R}^n$（随机单位向量），反复执行：

$$\tilde{\mathbf{u}} \leftarrow \frac{W \tilde{\mathbf{v}}}{\|W \tilde{\mathbf{v}}\|}, \qquad \tilde{\mathbf{v}} \leftarrow \frac{W^T \tilde{\mathbf{u}}}{\|W^T \tilde{\mathbf{u}}\|}$$

经过若干次迭代，$\tilde{\mathbf{u}} \approx \mathbf{u}_1$（最大左奇异向量），$\tilde{\mathbf{v}} \approx \mathbf{v}_1$（最大右奇异向量），谱范数估计为：

$$\sigma_{\max}(W) \approx \tilde{\mathbf{u}}^T W \tilde{\mathbf{v}}$$

在实践中，每次前向传播只做**一步幂迭代**更新，计算开销极小（$O(mn)$），且在训练过程中 $\tilde{\mathbf{u}}, \tilde{\mathbf{v}}$ 的值会从上一轮的结果继续迭代，收敛速度非常快。

### 17.6.3 与GAN稳定性的关联：SNGAN

**GAN（生成对抗网络）** 的训练不稳定问题由来已久。判别器（Discriminator）的Lipschitz约束是稳定GAN训练的关键：

从Wasserstein GAN（WGAN）的理论分析可知，若判别器 $D$ 是1-Lipschitz函数，则Wasserstein距离有更好的梯度性质：

$$W_1(P_r, P_g) = \sup_{\|D\|_L \leq 1} \mathbb{E}_{x \sim P_r}[D(x)] - \mathbb{E}_{x \sim P_g}[D(x)]$$

**SNGAN（Spectral Normalization GAN，Miyato et al., 2018）** 的核心贡献：对判别器每一层的权重矩阵施加谱归一化，精确控制每层的Lipschitz常数为1，从而：

$$\text{Lip}(D) \leq \prod_{\ell=1}^{L} \|W_\ell\|_2 = \prod_{\ell=1}^{L} 1 = 1$$

（若激活函数也是1-Lipschitz的，如ReLU、LeakyReLU。）

**效果：**
- 训练更稳定，不需要梯度裁剪（gradient clipping）
- 不像WGAN-GP那样需要额外的梯度惩罚项
- 计算开销极小，每层只需一步幂迭代

### 17.6.4 PyTorch代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 1. 手动实现谱归一化层 ──────────────────────────────────────
class SpectralNormLinear(nn.Module):
    """带谱归一化的线性层（手动实现，便于理解原理）"""
    def __init__(self, in_features: int, out_features: int, n_power_iter: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.n_power_iter = n_power_iter

        # 幂迭代的状态向量（不参与梯度计算）
        self.register_buffer("u", F.normalize(torch.randn(out_features), dim=0))
        self.register_buffer("v", F.normalize(torch.randn(in_features), dim=0))

    def _spectral_norm(self) -> torch.Tensor:
        """幂迭代估计谱范数"""
        u, v = self.u, self.v
        W = self.weight
        for _ in range(self.n_power_iter):
            # v ← normalize(Wᵀu), u ← normalize(Wv)
            v_new = F.normalize(W.T @ u, dim=0)
            u_new = F.normalize(W @ v_new, dim=0)
            u, v = u_new, v_new
        if self.training:
            # 更新缓存（detach 避免反向传播时把状态纳入计算图）
            self.u.copy_(u.detach())
            self.v.copy_(v.detach())
        sigma = u @ W @ v     # 谱范数的估计值
        return sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self._spectral_norm()
        W_hat = self.weight / sigma    # 谱归一化
        return F.linear(x, W_hat, self.bias)


# ── 2. 使用 PyTorch 内置谱归一化（推荐） ─────────────────────
def build_discriminator_with_sn(in_dim: int = 784, hidden: int = 256) -> nn.Sequential:
    """SNGAN风格的判别器：每个Linear层都施加谱归一化"""
    return nn.Sequential(
        nn.utils.spectral_norm(nn.Linear(in_dim, hidden)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.utils.spectral_norm(nn.Linear(hidden, hidden)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.utils.spectral_norm(nn.Linear(hidden, 1)),
    )


# ── 3. 验证：谱归一化后权重矩阵的谱范数 ≈ 1 ─────────────────
torch.manual_seed(0)
D = build_discriminator_with_sn()

# 做一次前向传播（触发幂迭代更新 u, v）
dummy_input = torch.randn(32, 784)
_ = D(dummy_input)

print("验证各层谱范数：")
for name, module in D.named_modules():
    if isinstance(module, nn.Linear) and hasattr(module, "weight_orig"):
        # PyTorch spectral_norm 将原始权重存为 weight_orig
        W = module.weight_orig.detach()
        # 用 SVD 精确计算谱范数
        sigma_exact = torch.linalg.svdvals(W).max().item()
        print(f"  层 {name}: 精确谱范数 = {sigma_exact:.4f}")

# ── 4. 正交权重初始化（Orthogonal Initialization） ─────────────
class OrthogonalLinear(nn.Module):
    """正交初始化的线性层，适合深层网络训练"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        # 用正交初始化替代默认的 Kaiming 初始化
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ── 5. 演示：正交初始化对梯度范数的影响 ─────────────────────
def check_gradient_norm(model: nn.Sequential, depth: int = 10) -> float:
    """前向+反向一次，统计最后一层梯度的范数"""
    x = torch.randn(1, 64)
    for _ in range(depth):
        x = model(x)
    loss = x.sum()
    loss.backward()
    grad_norm = model[-1].weight.grad.norm().item()
    return grad_norm

torch.manual_seed(42)
# 标准随机初始化（xavier）
random_net = nn.Sequential(*[nn.Linear(64, 64) for _ in range(5)])

# 正交初始化
ortho_net = nn.Sequential(*[nn.Linear(64, 64) for _ in range(5)])
for layer in ortho_net:
    nn.init.orthogonal_(layer.weight)

grad_random = check_gradient_norm(random_net)
grad_ortho = check_gradient_norm(ortho_net)

print(f"\n随机初始化 - 第1层梯度范数: {grad_random:.4f}")
print(f"正交初始化 - 第1层梯度范数: {grad_ortho:.4f}")
print("（正交初始化的梯度范数更接近1，说明梯度传播更稳定）")
```

**代码解读：**

- `SpectralNormLinear` 展示了谱归一化的内部机制：每次前向传播时用幂迭代估计谱范数，然后用 $\hat{W} = W / \sigma$ 做线性变换。
- `nn.utils.spectral_norm` 是 PyTorch 的内置实现，原理相同但工程优化更完善，推荐实际使用。
- `nn.init.orthogonal_` 用QR分解生成正交矩阵：对随机矩阵做QR分解，取 $Q$ 作为初始权重。
- 正交初始化对信号和梯度的范数保持不变，在深层网络（ResNet等）的初始训练阶段尤其有帮助。

**谱归一化与QR分解的关联：**

谱归一化是"软性"的等距约束——权重矩阵的谱范数被限制为1，但不要求它是正交矩阵（列不必两两正交）。而正交初始化则是"硬性"的等距约束——初始权重就是正交矩阵。两者都利用了正交性保持向量范数这一核心性质，是QR分解理论在深度学习中的直接应用。

---

## 练习题

**练习 17.1（基础）** 设 $\mathbf{a} = (2, 1, -1)^T$，$\mathbf{b} = (1, 1, 1)^T$。

(a) 计算 $\text{proj}_{\mathbf{b}} \mathbf{a}$。
(b) 计算残差 $\mathbf{a}^{\perp} = \mathbf{a} - \text{proj}_{\mathbf{b}} \mathbf{a}$，并验证 $\mathbf{a}^{\perp} \perp \mathbf{b}$。
(c) 写出向量 $\mathbf{b}$ 方向上的投影矩阵 $P = \dfrac{\mathbf{b}\mathbf{b}^T}{\mathbf{b}^T\mathbf{b}}$，验证 $P^2 = P$。

---

**练习 17.2（中等）** 对以下两个向量做Gram-Schmidt正交化，求标准正交基 $\{q_1, q_2\}$：

$$a_1 = \begin{pmatrix}3\\4\end{pmatrix}, \quad a_2 = \begin{pmatrix}1\\0\end{pmatrix}$$

并写出相应的QR分解 $A = QR$，其中 $A = [a_1 \mid a_2]$。

---

**练习 17.3（中等）** 设矩阵

$$A = \begin{pmatrix}1 & 2\\0 & 1\\0 & 0\end{pmatrix}$$

(a) 对 $A$ 的列向量做Gram-Schmidt正交化，求标准正交基。
(b) 写出 $A = QR$ 分解（$Q$ 是 $3 \times 2$，$R$ 是 $2 \times 2$）。
(c) 验证 $Q^T Q = I_2$。

---

**练习 17.4（较难）** 设超定方程组 $A\mathbf{x} = \mathbf{b}$，其中

$$A = \begin{pmatrix}1 & 0\\1 & 1\\1 & 2\end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix}6\\5\\7\end{pmatrix}$$

(a) 对 $A$ 做QR分解。
(b) 利用 $R\hat{\mathbf{x}} = Q^T\mathbf{b}$ 求最小二乘解 $\hat{\mathbf{x}}$。
(c) 计算残差 $\mathbf{r} = \mathbf{b} - A\hat{\mathbf{x}}$，验证 $\mathbf{r}$ 与 $A$ 的列向量正交。

---

**练习 17.5（挑战）** 设 $W \in \mathbb{R}^{n \times n}$ 是正交矩阵。

(a) 证明：$W$ 的所有奇异值均为1，即 $\sigma_1 = \sigma_2 = \cdots = \sigma_n = 1$。（提示：利用 $W^TW = I$ 和奇异值的定义。）
(b) 由 (a) 推导：$\|W\mathbf{x}\|^2 = \|\mathbf{x}\|^2$ 对所有 $\mathbf{x}$ 成立。
(c) 设 $f(\mathbf{x}) = W\mathbf{x}$，证明 $f$ 的Lipschitz常数为1，即 $\|f(\mathbf{x}) - f(\mathbf{y})\| \leq \|\mathbf{x} - \mathbf{y}\|$ 且等号可取到。
(d) 若将神经网络每一层的权重矩阵都初始化为正交矩阵，并在训练中保持 $\|W_\ell\|_2 = 1$，请解释为什么这有助于解决梯度消失/爆炸问题（用奇异值语言描述）。

---

## 练习答案

<details>
<summary>练习 17.1 答案</summary>

**(a) 投影：**

$$\mathbf{a} \cdot \mathbf{b} = 2 \cdot 1 + 1 \cdot 1 + (-1) \cdot 1 = 2, \quad \mathbf{b} \cdot \mathbf{b} = 1 + 1 + 1 = 3$$

$$\text{proj}_{\mathbf{b}} \mathbf{a} = \frac{2}{3}\begin{pmatrix}1\\1\\1\end{pmatrix} = \begin{pmatrix}2/3\\2/3\\2/3\end{pmatrix}$$

**(b) 残差：**

$$\mathbf{a}^{\perp} = \begin{pmatrix}2\\1\\-1\end{pmatrix} - \begin{pmatrix}2/3\\2/3\\2/3\end{pmatrix} = \begin{pmatrix}4/3\\1/3\\-5/3\end{pmatrix}$$

**验证正交性：**

$$\mathbf{a}^{\perp} \cdot \mathbf{b} = \frac{4}{3} \cdot 1 + \frac{1}{3} \cdot 1 + \left(-\frac{5}{3}\right) \cdot 1 = \frac{4 + 1 - 5}{3} = 0 \checkmark$$

**(c) 投影矩阵：**

$$P = \frac{1}{3}\begin{pmatrix}1\\1\\1\end{pmatrix}\begin{pmatrix}1&1&1\end{pmatrix} = \frac{1}{3}\begin{pmatrix}1&1&1\\1&1&1\\1&1&1\end{pmatrix}$$

**验证幂等性 $P^2 = P$：**

$$P^2 = \frac{1}{9}\begin{pmatrix}1&1&1\\1&1&1\\1&1&1\end{pmatrix}^2 = \frac{1}{9}\begin{pmatrix}3&3&3\\3&3&3\\3&3&3\end{pmatrix} = \frac{1}{3}\begin{pmatrix}1&1&1\\1&1&1\\1&1&1\end{pmatrix} = P \checkmark$$

</details>

---

<details>
<summary>练习 17.2 答案</summary>

**第1步：处理 $a_1$**

$$\mathbf{v}_1 = a_1 = \begin{pmatrix}3\\4\end{pmatrix}, \quad \|\mathbf{v}_1\| = \sqrt{9+16} = 5$$

$$q_1 = \frac{1}{5}\begin{pmatrix}3\\4\end{pmatrix}$$

**第2步：处理 $a_2$**

$$a_2 \cdot q_1 = \frac{1}{5}(1 \cdot 3 + 0 \cdot 4) = \frac{3}{5}$$

$$\mathbf{v}_2 = a_2 - \frac{3}{5} q_1 = \begin{pmatrix}1\\0\end{pmatrix} - \frac{3}{5} \cdot \frac{1}{5}\begin{pmatrix}3\\4\end{pmatrix} = \begin{pmatrix}1\\0\end{pmatrix} - \begin{pmatrix}9/25\\12/25\end{pmatrix} = \begin{pmatrix}16/25\\-12/25\end{pmatrix}$$

$$\|\mathbf{v}_2\| = \frac{1}{25}\sqrt{16^2 + 12^2} = \frac{1}{25}\sqrt{256+144} = \frac{20}{25} = \frac{4}{5}$$

$$q_2 = \frac{5}{4} \cdot \begin{pmatrix}16/25\\-12/25\end{pmatrix} = \begin{pmatrix}4/5\\-3/5\end{pmatrix}$$

**标准正交基：** $q_1 = \dfrac{1}{5}(3, 4)^T$，$q_2 = \dfrac{1}{5}(4, -3)^T$。

**验证正交性：** $q_1 \cdot q_2 = \dfrac{1}{25}(3 \cdot 4 + 4 \cdot (-3)) = \dfrac{12 - 12}{25} = 0$。✓

**QR分解：**

$$Q = \begin{pmatrix}3/5 & 4/5 \\ 4/5 & -3/5\end{pmatrix}$$

$$R = Q^T A = \begin{pmatrix}3/5 & 4/5 \\ 4/5 & -3/5\end{pmatrix}\begin{pmatrix}3 & 1\\4 & 0\end{pmatrix} = \begin{pmatrix}5 & 3/5 \\ 0 & 4/5\end{pmatrix}$$

对角元 $r_{11} = \|\mathbf{v}_1\| = 5 > 0$，$r_{22} = \|\mathbf{v}_2\| = 4/5 > 0$。✓

</details>

---

<details>
<summary>练习 17.3 答案</summary>

设 $a_1 = (1, 0, 0)^T$，$a_2 = (2, 1, 0)^T$。

**(a) Gram-Schmidt正交化：**

**第1步：**

$$\mathbf{v}_1 = a_1 = \begin{pmatrix}1\\0\\0\end{pmatrix}, \quad \|\mathbf{v}_1\| = 1, \quad q_1 = \begin{pmatrix}1\\0\\0\end{pmatrix}$$

**第2步：**

$$a_2 \cdot q_1 = 2 \cdot 1 + 1 \cdot 0 + 0 \cdot 0 = 2$$

$$\mathbf{v}_2 = a_2 - 2q_1 = \begin{pmatrix}2\\1\\0\end{pmatrix} - \begin{pmatrix}2\\0\\0\end{pmatrix} = \begin{pmatrix}0\\1\\0\end{pmatrix}$$

$$\|\mathbf{v}_2\| = 1, \quad q_2 = \begin{pmatrix}0\\1\\0\end{pmatrix}$$

**(b) QR分解：**

$$Q = \begin{pmatrix}1 & 0 \\ 0 & 1 \\ 0 & 0\end{pmatrix}, \quad R = Q^T A = \begin{pmatrix}1&0&0\\0&1&0\end{pmatrix}\begin{pmatrix}1&2\\0&1\\0&0\end{pmatrix} = \begin{pmatrix}1&2\\0&1\end{pmatrix}$$

**(c) 验证 $Q^T Q = I_2$：**

$$Q^T Q = \begin{pmatrix}1&0&0\\0&1&0\end{pmatrix}\begin{pmatrix}1&0\\0&1\\0&0\end{pmatrix} = \begin{pmatrix}1&0\\0&1\end{pmatrix} = I_2 \checkmark$$

</details>

---

<details>
<summary>练习 17.4 答案</summary>

**(a) 对 $A$ 做QR分解：**

$A$ 的列向量：$a_1 = (1,1,1)^T$，$a_2 = (0,1,2)^T$。

**第1步：**

$$\|\mathbf{v}_1\| = \|a_1\| = \sqrt{3}, \quad q_1 = \frac{1}{\sqrt{3}}\begin{pmatrix}1\\1\\1\end{pmatrix}$$

**第2步：**

$$a_2 \cdot q_1 = \frac{1}{\sqrt{3}}(0 + 1 + 2) = \frac{3}{\sqrt{3}} = \sqrt{3}$$

$$\mathbf{v}_2 = a_2 - \sqrt{3} q_1 = \begin{pmatrix}0\\1\\2\end{pmatrix} - \begin{pmatrix}1\\1\\1\end{pmatrix} = \begin{pmatrix}-1\\0\\1\end{pmatrix}$$

$$\|\mathbf{v}_2\| = \sqrt{2}, \quad q_2 = \frac{1}{\sqrt{2}}\begin{pmatrix}-1\\0\\1\end{pmatrix}$$

$$Q = \begin{pmatrix}1/\sqrt{3} & -1/\sqrt{2} \\ 1/\sqrt{3} & 0 \\ 1/\sqrt{3} & 1/\sqrt{2}\end{pmatrix}, \quad R = \begin{pmatrix}\sqrt{3} & \sqrt{3} \\ 0 & \sqrt{2}\end{pmatrix}$$

**(b) 求最小二乘解：**

计算 $Q^T \mathbf{b}$：

$$Q^T \mathbf{b} = \begin{pmatrix}\frac{6+5+7}{\sqrt{3}} \\ \frac{-6+0+7}{\sqrt{2}}\end{pmatrix} = \begin{pmatrix}\frac{18}{\sqrt{3}} \\ \frac{1}{\sqrt{2}}\end{pmatrix} = \begin{pmatrix}6\sqrt{3} \\ \frac{1}{\sqrt{2}}\end{pmatrix}$$

求解 $R\hat{\mathbf{x}} = Q^T\mathbf{b}$（回代）：

第2行：$\sqrt{2}\,\hat{x}_2 = \dfrac{1}{\sqrt{2}} \implies \hat{x}_2 = \dfrac{1}{2}$

第1行：$\sqrt{3}\,\hat{x}_1 + \sqrt{3}\,\hat{x}_2 = 6\sqrt{3} \implies \hat{x}_1 + \hat{x}_2 = 6 \implies \hat{x}_1 = 6 - \frac{1}{2} = \frac{11}{2}$

故 $\hat{\mathbf{x}} = \left(\dfrac{11}{2},\, \dfrac{1}{2}\right)^T$。

**(c) 验证残差正交性：**

$$A\hat{\mathbf{x}} = \begin{pmatrix}1&0\\1&1\\1&2\end{pmatrix}\begin{pmatrix}11/2\\1/2\end{pmatrix} = \begin{pmatrix}11/2\\6\\13/2\end{pmatrix}$$

$$\mathbf{r} = \mathbf{b} - A\hat{\mathbf{x}} = \begin{pmatrix}6\\5\\7\end{pmatrix} - \begin{pmatrix}11/2\\6\\13/2\end{pmatrix} = \begin{pmatrix}1/2\\-1\\1/2\end{pmatrix}$$

验证 $\mathbf{r} \perp a_1$：$a_1 \cdot \mathbf{r} = 1/2 - 1 + 1/2 = 0$。✓

验证 $\mathbf{r} \perp a_2$：$a_2 \cdot \mathbf{r} = 0 \cdot (1/2) + 1 \cdot (-1) + 2 \cdot (1/2) = -1 + 1 = 0$。✓

</details>

---

<details>
<summary>练习 17.5 答案</summary>

**(a) 证明正交矩阵的奇异值均为1：**

奇异值 $\sigma_i$ 是矩阵 $W^TW$ 的特征值的算术平方根（即 $\sigma_i = \sqrt{\lambda_i(W^TW)}$）。

因 $W^TW = I$，$W^TW$ 的特征值全为1，故：

$$\sigma_i = \sqrt{\lambda_i(W^TW)} = \sqrt{1} = 1, \quad i = 1, 2, \ldots, n \quad \square$$

**(b) 保持向量范数：**

利用 (a) 的结论，对任意 $\mathbf{x} \in \mathbb{R}^n$：

$$\|W\mathbf{x}\|^2 = (W\mathbf{x})^T(W\mathbf{x}) = \mathbf{x}^T W^T W \mathbf{x} = \mathbf{x}^T I \mathbf{x} = \mathbf{x}^T \mathbf{x} = \|\mathbf{x}\|^2 \quad \square$$

**(c) Lipschitz常数为1：**

对任意 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$：

$$\|f(\mathbf{x}) - f(\mathbf{y})\| = \|W\mathbf{x} - W\mathbf{y}\| = \|W(\mathbf{x} - \mathbf{y})\| = \|\mathbf{x} - \mathbf{y}\|$$

（最后一步用了 (b)。）

故 $\|f(\mathbf{x}) - f(\mathbf{y})\| = \|\mathbf{x} - \mathbf{y}\| \leq \|\mathbf{x} - \mathbf{y}\|$，Lipschitz常数 $= 1$，且等号恒成立。$\square$

**(d) 解决梯度消失/爆炸的原因：**

设神经网络第 $\ell$ 层权重矩阵 $W_\ell$ 的奇异值为 $\sigma_1^{(\ell)} \geq \cdots \geq \sigma_n^{(\ell)}$。反向传播时，梯度经过 $L$ 层传播：

$$\left\|\nabla_{\mathbf{x}^{(0)}} \mathcal{L}\right\| \leq \prod_{\ell=1}^{L} \|W_\ell\|_2 \cdot \left\|\nabla_{\mathbf{x}^{(L)}} \mathcal{L}\right\| = \prod_{\ell=1}^{L} \sigma_1^{(\ell)} \cdot \left\|\nabla_{\mathbf{x}^{(L)}} \mathcal{L}\right\|$$

- 若各层 $\sigma_1^{(\ell)} > 1$：乘积指数增长，梯度爆炸（$\sigma^L \to \infty$）
- 若各层 $\sigma_1^{(\ell)} < 1$：乘积指数衰减，梯度消失（$\sigma^L \to 0$）
- 若各层 $\sigma_1^{(\ell)} = 1$（如保持正交初始化）：乘积始终为1，梯度范数在各层之间保持稳定，不发生消失或爆炸

因此，用正交矩阵初始化（或在训练中约束 $\|W_\ell\|_2 = 1$）能将每层的谱范数固定为1，使得梯度在 $L$ 层网络中传播时范数不放大也不缩小，从根本上解决梯度消失/爆炸问题。$\square$

</details>
