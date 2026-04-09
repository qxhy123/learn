# 第5章：高斯消元法

> "数学的本质在于它的自由。" —— 格奥尔格·康托尔

---

## 学习目标

完成本章学习后，你将能够：

- 掌握三种初等行变换，并理解它们对方程组解集的影响
- 熟练运用高斯消元法求解线性方程组，包括无解和无穷多解的情形
- 区分并构造行阶梯形矩阵与简化行阶梯形矩阵
- 理解 LU 分解的基本思想及其在计算中的优势

---

## 5.1 初等行变换

### 5.1.1 三种初等行变换

在求解线性方程组时，我们可以对方程组施以某些操作而不改变其解集。这些操作对应到增广矩阵上，就是**初等行变换**（Elementary Row Operations）。

**定义：** 以下三种对矩阵行的操作称为初等行变换：

1. **交换两行**（Row Swap）：将第 $i$ 行与第 $j$ 行互换，记作 $R_i \leftrightarrow R_j$。

2. **数乘一行**（Row Scaling）：用非零常数 $c \neq 0$ 乘以某一行，记作 $R_i \leftarrow c \cdot R_i$。

3. **行的倍加**（Row Replacement）：将某行的 $c$ 倍加到另一行，记作 $R_j \leftarrow R_j + c \cdot R_i$。

**重要性质：** 这三种变换均为**可逆**操作，即每种变换都有对应的逆变换，因此不会改变方程组的解集。两个矩阵若可经初等行变换相互转化，则称它们**行等价**（Row Equivalent）。

### 5.1.2 初等矩阵

每种初等行变换都对应一个**初等矩阵**（Elementary Matrix）$E$，它由对单位矩阵 $I$ 施以对应初等行变换得到。

对矩阵 $A$ 施以初等行变换，等价于用对应的初等矩阵从左乘 $A$：

$$E \cdot A = B$$

其中 $B$ 是 $A$ 经该初等行变换后的结果。

**示例：** 对 $3 \times 3$ 矩阵，将第1行的 $-2$ 倍加到第3行（$R_3 \leftarrow R_3 - 2R_1$），对应初等矩阵为：

$$E = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ -2 & 0 & 1 \end{pmatrix}$$

初等矩阵均可逆，其逆矩阵仍是初等矩阵，对应逆变换。

---

## 5.2 高斯消元法

**高斯消元法**（Gaussian Elimination）是求解线性方程组的系统化算法。其核心思想是：通过初等行变换，将增广矩阵化为**行阶梯形**，再进行**回代**求解。

### 5.2.1 行阶梯形矩阵

**定义：** 矩阵 $A$ 称为**行阶梯形矩阵**（Row Echelon Form, REF），若满足：

1. 所有全零行（若有）位于矩阵底部。
2. 每个非零行的第一个非零元素（称为**主元**，Pivot）严格位于上一行主元的右侧。

$$\text{行阶梯形示例：}\quad \begin{pmatrix} \boxed{2} & 1 & -1 & 8 \\ 0 & \boxed{3} & 2 & 5 \\ 0 & 0 & \boxed{-1} & 3 \end{pmatrix}$$

其中方框标记的元素为主元。

### 5.2.2 消元步骤

高斯消元法的前向消元过程如下：

1. 找到最左边的非零列，选取该列中的非零元素作为主元（若需要，先交换行）。
2. 用行倍加操作，将主元**下方**的所有元素化为零。
3. 对剩余子矩阵重复步骤 1–2，直至整个矩阵化为行阶梯形。

### 5.2.3 回代求解

将增广矩阵化为行阶梯形后，从最后一个非零行开始，依次向上代入求解各未知量——这一过程称为**回代**（Back Substitution）。

### 5.2.4 详细例题

**例题 5.1** 求解线性方程组：

$$\begin{cases} 2x_1 + x_2 - x_3 = 8 \\ -3x_1 - x_2 + 2x_3 = -11 \\ -2x_1 + x_2 + 2x_3 = -3 \end{cases}$$

**解：** 写出增广矩阵并逐步消元。

**初始增广矩阵：**

$$\left(\begin{array}{ccc|c} 2 & 1 & -1 & 8 \\ -3 & -1 & 2 & -11 \\ -2 & 1 & 2 & -3 \end{array}\right)$$

**步骤 1：** 消去第1列第2、3行的元素。

$R_2 \leftarrow R_2 + \dfrac{3}{2} R_1$，$R_3 \leftarrow R_3 + R_1$：

$$\left(\begin{array}{ccc|c} 2 & 1 & -1 & 8 \\ 0 & \tfrac{1}{2} & \tfrac{1}{2} & 1 \\ 0 & 2 & 1 & 5 \end{array}\right)$$

**步骤 2：** 消去第2列第3行的元素。

$R_3 \leftarrow R_3 - 4 R_2$：

$$\left(\begin{array}{ccc|c} 2 & 1 & -1 & 8 \\ 0 & \tfrac{1}{2} & \tfrac{1}{2} & 1 \\ 0 & 0 & -1 & 1 \end{array}\right)$$

矩阵已化为行阶梯形，三个主元均存在，方程组有唯一解。

**回代：**

由第3行：$-x_3 = 1 \Rightarrow x_3 = -1$

由第2行：$\dfrac{1}{2}x_2 + \dfrac{1}{2}(-1) = 1 \Rightarrow \dfrac{1}{2}x_2 = \dfrac{3}{2} \Rightarrow x_2 = 3$

由第1行：$2x_1 + 3 - (-1) = 8 \Rightarrow 2x_1 = 4 \Rightarrow x_1 = 2$

**解为：** $x_1 = 2,\; x_2 = 3,\; x_3 = -1$。

**验证：** 代入原方程：$2(2) + 3 - (-1) = 8$ ✓，$-3(2) - 3 + 2(-1) = -11$ ✓，$-2(2) + 3 + 2(-1) = -3$ ✓。

---

**例题 5.2（无穷多解）** 求解：

$$\left(\begin{array}{ccc|c} 1 & -2 & 1 & 0 \\ 2 & -3 & 1 & -1 \\ 0 & 1 & -1 & 1 \end{array}\right)$$

经消元化简后可得行阶梯形含一个自由变量，方程组有无穷多解（参数解）。其过程与上例类似，读者可作为练习完成（参见练习题第2题）。

---

## 5.3 简化行阶梯形（Gauss-Jordan 消元）

### 5.3.1 简化行阶梯形的定义

**定义：** 矩阵 $A$ 称为**简化行阶梯形矩阵**（Reduced Row Echelon Form, RREF），若它是行阶梯形，且还满足：

1. 每个主元为 $1$（称为**首一**）。
2. 每个主元所在列的其他元素均为 $0$（主元既消下方，也消上方）。

$$\text{RREF 示例：}\quad \begin{pmatrix} 1 & 0 & 0 & 2 \\ 0 & 1 & 0 & -1 \\ 0 & 0 & 1 & 3 \end{pmatrix}$$

**唯一性定理：** 任何矩阵的简化行阶梯形是唯一的。

### 5.3.2 Gauss-Jordan 消元步骤

**Gauss-Jordan 消元法**在高斯消元（前向消元）的基础上，继续**向上消元**，直接得到 RREF，从而无需回代即可读出解。

步骤如下：

1. 完成高斯消元，得到行阶梯形。
2. 从最后一个主元开始，依次向上消去每个主元所在列的其他元素。
3. 将每个主元所在行除以主元值，使主元化为 $1$。

**例题 5.3** 对例题 5.1 的增广矩阵继续进行 Gauss-Jordan 消元。

从已得到的行阶梯形出发：

$$\left(\begin{array}{ccc|c} 2 & 1 & -1 & 8 \\ 0 & \tfrac{1}{2} & \tfrac{1}{2} & 1 \\ 0 & 0 & -1 & 1 \end{array}\right)$$

**步骤 3：** 将第3行乘以 $-1$，使主元化为 $1$：

$$\left(\begin{array}{ccc|c} 2 & 1 & -1 & 8 \\ 0 & \tfrac{1}{2} & \tfrac{1}{2} & 1 \\ 0 & 0 & 1 & -1 \end{array}\right)$$

**步骤 4：** 向上消第3列：$R_1 \leftarrow R_1 + R_3$，$R_2 \leftarrow R_2 - \tfrac{1}{2}R_3$：

$$\left(\begin{array}{ccc|c} 2 & 1 & 0 & 7 \\ 0 & \tfrac{1}{2} & 0 & \tfrac{3}{2} \\ 0 & 0 & 1 & -1 \end{array}\right)$$

**步骤 5：** 将第2行乘以 $2$：$R_2 \leftarrow 2R_2$：

$$\left(\begin{array}{ccc|c} 2 & 1 & 0 & 7 \\ 0 & 1 & 0 & 3 \\ 0 & 0 & 1 & -1 \end{array}\right)$$

**步骤 6：** 向上消第2列：$R_1 \leftarrow R_1 - R_2$：

$$\left(\begin{array}{ccc|c} 2 & 0 & 0 & 4 \\ 0 & 1 & 0 & 3 \\ 0 & 0 & 1 & -1 \end{array}\right)$$

**步骤 7：** 将第1行除以 $2$：$R_1 \leftarrow \tfrac{1}{2}R_1$：

$$\left(\begin{array}{ccc|c} 1 & 0 & 0 & 2 \\ 0 & 1 & 0 & 3 \\ 0 & 0 & 1 & -1 \end{array}\right)$$

直接读出：$x_1 = 2,\; x_2 = 3,\; x_3 = -1$，与例题 5.1 结果一致。

---

## 5.4 LU 分解简介

### 5.4.1 LU 分解的概念

**LU 分解**（LU Decomposition / LU Factorization）将矩阵 $A$ 分解为一个**下三角矩阵** $L$ 与一个**上三角矩阵** $U$ 的乘积：

$$A = LU$$

其中：
- $L$（Lower triangular）：主对角线元素为 $1$，对角线以上元素为 $0$。
- $U$（Upper triangular）：对角线以下元素为 $0$（即高斯消元后的行阶梯形）。

**构造方法：** 高斯消元过程中，每次用 $R_j \leftarrow R_j - m_{ij} \cdot R_i$ 消去元素时，乘数 $m_{ij}$ 即为 $L$ 中对应位置的元素。

**示例：** 对矩阵 $A = \begin{pmatrix} 2 & 1 & -1 \\ -3 & -1 & 2 \\ -2 & 1 & 2 \end{pmatrix}$，

高斯消元的乘数为：$m_{21} = \tfrac{3}{2}$，$m_{31} = 1$，$m_{32} = 4$，故：

$$L = \begin{pmatrix} 1 & 0 & 0 \\ -\tfrac{3}{2} & 1 & 0 \\ -1 & 4 & 1 \end{pmatrix}, \quad U = \begin{pmatrix} 2 & 1 & -1 \\ 0 & \tfrac{1}{2} & \tfrac{1}{2} \\ 0 & 0 & -1 \end{pmatrix}$$

注意：$L$ 中乘数填入时取**相反数**（因消元时使用 $-m_{ij}$），可验证 $LU = A$。

### 5.4.2 LU 分解的优势

当需要对**同一矩阵 $A$** 求解多个不同右端项 $b_1, b_2, \ldots, b_k$ 时，LU 分解的优势十分显著：

1. **一次分解，多次求解：** 只需对 $A$ 进行一次 LU 分解（代价约为 $\dfrac{2n^3}{3}$ 次运算），之后每次求解 $Ax = b$ 转化为两步：
   - 前代（Forward Substitution）：$Ly = b$，求 $y$
   - 回代（Back Substitution）：$Ux = y$，求 $x$

   每步仅需 $O(n^2)$ 次运算，远少于重新消元的 $O(n^3)$。

2. **计算行列式：** $\det(A) = \det(L) \cdot \det(U) = \prod_{i=1}^n u_{ii}$（因 $\det(L)=1$），只需将 $U$ 的对角元素连乘。

3. **求逆矩阵：** 通过求解 $n$ 个系统 $Ax_i = e_i$（$e_i$ 为标准基向量）高效得到 $A^{-1}$。

**实际中的改进——部分选主元（Partial Pivoting）：** 为保证数值稳定性，实际计算中通常在每步选取绝对值最大的元素作主元，将 LU 分解改写为带置换矩阵 $P$ 的形式：

$$PA = LU$$

其中 $P$ 是置换矩阵，记录了行交换操作。

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 初等行变换 | 三种操作（交换、数乘、倍加），不改变解集，均可逆 |
| 行阶梯形（REF） | 主元逐行右移，全零行在底部 |
| 高斯消元 | 前向消元→行阶梯形→回代，时间复杂度 $O(n^3)$ |
| 简化行阶梯形（RREF） | 主元为 1 且所在列其余元素为 0，唯一确定 |
| Gauss-Jordan 消元 | 前向+后向消元，直接得 RREF，无需回代 |
| LU 分解 | $A = LU$，分解一次可多次求解，复杂度优于重复消元 |

**解的判定规则（对 $n$ 元方程组）：**

- 主元数 $= n$（无自由变量）且相容 → **唯一解**
- 主元数 $< n$（有自由变量）且相容 → **无穷多解**
- 增广矩阵含矛盾行 $[0 \cdots 0 \mid b],\; b \neq 0$ → **无解**

---

## 深度学习应用

### 数值稳定性问题

深度学习中，求解线性系统（如正规方程、优化步骤）时常面临**数值稳定性**挑战。高斯消元在主元接近零时会产生**数值放大误差**（floating-point amplification）。

**部分选主元**（Partial Pivoting）是标准工程实践：在每列消元前，将绝对值最大的行交换到主元位置，使乘数 $|m_{ij}| \leq 1$，有效遏制误差增长。NumPy、LAPACK 等工具默认采用此策略。

### 求解器在优化中的应用

在优化算法（如牛顿法）的每步迭代中，需求解：

$$H \Delta \theta = -g$$

其中 $H$ 为 Hessian 矩阵，$g$ 为梯度。对中小规模问题（如二阶优化器 K-FAC），LU 分解比迭代法（共轭梯度等）更精确，常用于：

- **自然梯度法**中求费舍尔信息矩阵的逆
- **二阶优化器**中求解局部曲率方程
- **最小二乘问题** $\min \|Ax - b\|^2$ 的正规方程 $A^\top A x = A^\top b$

### LU 分解在大规模计算中的作用

大规模神经网络训练常需反复求解结构相似的线性系统。LU 分解的"一次分解，多次使用"特性在以下场景尤为重要：

- **批量正则化**：当多个 mini-batch 共享同一矩阵结构时
- **稀疏线性系统**：图神经网络中的图拉普拉斯矩阵求解
- **结构化矩阵求逆**：注意力机制中的协方差矩阵操作

### 代码示例

```python
import numpy as np
from scipy.linalg import lu

# ========================================
# 1. 手动实现高斯消元法（带部分选主元）
# ========================================
def gaussian_elimination(A, b):
    """
    用高斯消元法（部分选主元）求解 Ax = b
    返回解向量 x
    """
    n = len(b)
    # 构建增广矩阵
    Ab = np.array(np.column_stack([A, b]), dtype=float)

    for col in range(n):
        # 部分选主元：找到第 col 列绝对值最大的行
        max_row = np.argmax(np.abs(Ab[col:, col])) + col
        Ab[[col, max_row]] = Ab[[max_row, col]]  # 交换行

        pivot = Ab[col, col]
        if abs(pivot) < 1e-12:
            raise ValueError(f"矩阵奇异或接近奇异（列 {col}）")

        # 消去主元下方的所有元素
        for row in range(col + 1, n):
            factor = Ab[row, col] / pivot
            Ab[row, col:] -= factor * Ab[col, col:]

    # 回代
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]

    return x

# 测试：求解例题 5.1
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

x = gaussian_elimination(A, b)
print("高斯消元解:", x)
# 输出：高斯消元解: [ 2.  3. -1.]

# ========================================
# 2. LU 分解（使用 scipy）
# ========================================
P, L, U = lu(A)
print("\nLU 分解结果：")
print("置换矩阵 P =\n", P)
print("下三角矩阵 L =\n", np.round(L, 4))
print("上三角矩阵 U =\n", np.round(U, 4))

# 利用 LU 分解求解多个右端项
from scipy.linalg import lu_factor, lu_solve

lu_factored = lu_factor(A)

b1 = np.array([8, -11, -3], dtype=float)
b2 = np.array([1, 0, 0], dtype=float)
b3 = np.array([0, 1, 0], dtype=float)

x1 = lu_solve(lu_factored, b1)
x2 = lu_solve(lu_factored, b2)
x3 = lu_solve(lu_factored, b3)

print("\n求解多个右端项（LU 分解仅执行一次）：")
print(f"b1 的解：{np.round(x1, 4)}")
print(f"b2 的解：{np.round(x2, 4)}")
print(f"b3 的解：{np.round(x3, 4)}")

# ========================================
# 3. 将矩阵化为 RREF（手动实现）
# ========================================
def rref(A):
    """将矩阵化为简化行阶梯形（RREF）"""
    M = A.astype(float).copy()
    rows, cols = M.shape
    pivot_row = 0

    for col in range(cols):
        # 找主元
        non_zero = np.where(np.abs(M[pivot_row:, col]) > 1e-10)[0]
        if len(non_zero) == 0:
            continue
        # 将主元行移到当前位置
        swap = non_zero[0] + pivot_row
        M[[pivot_row, swap]] = M[[swap, pivot_row]]
        # 主元归一
        M[pivot_row] /= M[pivot_row, col]
        # 消去该列其他元素（包括上方）
        for r in range(rows):
            if r != pivot_row and abs(M[r, col]) > 1e-10:
                M[r] -= M[r, col] * M[pivot_row]
        pivot_row += 1
        if pivot_row >= rows:
            break

    return np.round(M, 10)

# 测试 RREF
Ab_augmented = np.column_stack([A, b])
print("\n增广矩阵的 RREF：")
print(rref(Ab_augmented))
# 输出：[[1. 0. 0. 2.], [0. 1. 0. 3.], [0. 0. 1. -1.]]
```

---

## 练习题

**练习 5.1（基础）** 对下列矩阵施以初等行变换 $R_2 \leftarrow R_2 - 2R_1$，写出变换后的矩阵，并写出对应的初等矩阵 $E$：

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 5 & 4 \\ 0 & 1 & 2 \end{pmatrix}$$

---

**练习 5.2（中等）** 用高斯消元法求解下列方程组，判断解的情况（唯一解/无穷多解/无解），并在有解时求出完整解：

$$\begin{cases} x_1 - 2x_2 + x_3 = 0 \\ 2x_1 - 3x_2 + x_3 = -1 \\ x_2 - x_3 = 1 \end{cases}$$

---

**练习 5.3（中等）** 将下列矩阵化为简化行阶梯形（RREF），并指出主元列和自由变量：

$$B = \begin{pmatrix} 0 & 1 & -4 & 8 \\ 2 & -3 & 2 & 1 \\ 5 & -8 & 7 & 1 \end{pmatrix}$$

---

**练习 5.4（较难）** 求矩阵 $A = \begin{pmatrix} 1 & 1 & 0 \\ 2 & 1 & -1 \\ 3 & -1 & -1 \end{pmatrix}$ 的 LU 分解（不使用置换矩阵，即 $A = LU$），并利用 LU 分解求解 $Ax = \begin{pmatrix} 2 \\ 0 \\ 1 \end{pmatrix}$。

---

**练习 5.5（综合）** 考虑含参数 $k$ 的线性方程组：

$$\begin{cases} kx_1 + x_2 = 1 \\ x_1 + kx_2 = 1 \end{cases}$$

讨论当 $k$ 取不同值时，方程组解的情况（唯一解、无穷多解、无解），并在有唯一解时用 Cramer 法则或消元法给出解。

---

## 练习答案

<details>
<summary>点击展开 练习 5.1 答案</summary>

施以 $R_2 \leftarrow R_2 - 2R_1$：

$$\begin{pmatrix} 1 & 2 & 3 \\ 2-2(1) & 5-2(2) & 4-2(3) \\ 0 & 1 & 2 \end{pmatrix} = \begin{pmatrix} 1 & 2 & 3 \\ 0 & 1 & -2 \\ 0 & 1 & 2 \end{pmatrix}$$

对应初等矩阵（对 $I$ 施以同样操作）：

$$E = \begin{pmatrix} 1 & 0 & 0 \\ -2 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

验证：$EA = \begin{pmatrix} 1 & 0 & 0 \\ -2 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}\begin{pmatrix} 1 & 2 & 3 \\ 2 & 5 & 4 \\ 0 & 1 & 2 \end{pmatrix} = \begin{pmatrix} 1 & 2 & 3 \\ 0 & 1 & -2 \\ 0 & 1 & 2 \end{pmatrix}$ ✓

</details>

<details>
<summary>点击展开 练习 5.2 答案</summary>

写出增广矩阵：

$$\left(\begin{array}{ccc|c} 1 & -2 & 1 & 0 \\ 2 & -3 & 1 & -1 \\ 0 & 1 & -1 & 1 \end{array}\right)$$

$R_2 \leftarrow R_2 - 2R_1$：

$$\left(\begin{array}{ccc|c} 1 & -2 & 1 & 0 \\ 0 & 1 & -1 & -1 \\ 0 & 1 & -1 & 1 \end{array}\right)$$

$R_3 \leftarrow R_3 - R_2$：

$$\left(\begin{array}{ccc|c} 1 & -2 & 1 & 0 \\ 0 & 1 & -1 & -1 \\ 0 & 0 & 0 & 2 \end{array}\right)$$

第3行对应方程 $0 = 2$，矛盾。**方程组无解。**

</details>

<details>
<summary>点击展开 练习 5.3 答案</summary>

初始矩阵（交换 $R_1$ 与 $R_2$，使第1列有非零主元）：

$$\left(\begin{array}{cccc} 2 & -3 & 2 & 1 \\ 0 & 1 & -4 & 8 \\ 5 & -8 & 7 & 1 \end{array}\right)$$

$R_3 \leftarrow R_3 - \tfrac{5}{2}R_1$：

$$\left(\begin{array}{cccc} 2 & -3 & 2 & 1 \\ 0 & 1 & -4 & 8 \\ 0 & -\tfrac{1}{2} & 2 & -\tfrac{3}{2} \end{array}\right)$$

$R_3 \leftarrow R_3 + \tfrac{1}{2}R_2$：

$$\left(\begin{array}{cccc} 2 & -3 & 2 & 1 \\ 0 & 1 & -4 & 8 \\ 0 & 0 & 0 & \tfrac{5}{2} \end{array}\right)$$

注意第3行 $[0\;0\;0\;\tfrac{5}{2}]$ 表示 $0 = \tfrac{5}{2}$，矛盾，方程组**无解**。

（若原题矩阵不带右端项，仅化系数矩阵：则第3行全零，主元列为第1、2列，$x_3$ 为自由变量。请结合具体情况判断。）

</details>

<details>
<summary>点击展开 练习 5.4 答案</summary>

**LU 分解：**

从 $A = \begin{pmatrix} 1 & 1 & 0 \\ 2 & 1 & -1 \\ 3 & -1 & -1 \end{pmatrix}$ 出发：

消元乘数：$m_{21} = 2$，$m_{31} = 3$：

$R_2 \leftarrow R_2 - 2R_1$，$R_3 \leftarrow R_3 - 3R_1$：

$$\begin{pmatrix} 1 & 1 & 0 \\ 0 & -1 & -1 \\ 0 & -4 & -1 \end{pmatrix}$$

消元乘数：$m_{32} = 4$：

$R_3 \leftarrow R_3 - 4R_2$：

$$U = \begin{pmatrix} 1 & 1 & 0 \\ 0 & -1 & -1 \\ 0 & 0 & 3 \end{pmatrix}$$

$$L = \begin{pmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 3 & 4 & 1 \end{pmatrix}$$

验证：$LU = \begin{pmatrix}1&0&0\\2&1&0\\3&4&1\end{pmatrix}\begin{pmatrix}1&1&0\\0&-1&-1\\0&0&3\end{pmatrix} = \begin{pmatrix}1&1&0\\2&1&-1\\3&-1&-1\end{pmatrix} = A$ ✓

**利用 LU 分解求解 $Ax = b$（$b = (2,0,1)^\top$）：**

**前代** $Ly = b$：

$$\begin{cases} y_1 = 2 \\ 2y_1 + y_2 = 0 \Rightarrow y_2 = -4 \\ 3y_1 + 4y_2 + y_3 = 1 \Rightarrow y_3 = 1 - 6 + 16 = 11 \end{cases}$$

**回代** $Ux = y$：

$$\begin{cases} 3x_3 = 11 \Rightarrow x_3 = \tfrac{11}{3} \\ -x_2 - x_3 = -4 \Rightarrow x_2 = 4 - \tfrac{11}{3} = \tfrac{1}{3} \\ x_1 + x_2 = 2 \Rightarrow x_1 = 2 - \tfrac{1}{3} = \tfrac{5}{3} \end{cases}$$

**解为：** $x = \left(\dfrac{5}{3},\; \dfrac{1}{3},\; \dfrac{11}{3}\right)^\top$

</details>

<details>
<summary>点击展开 练习 5.5 答案</summary>

写出增广矩阵：

$$\left(\begin{array}{cc|c} k & 1 & 1 \\ 1 & k & 1 \end{array}\right)$$

$R_1 \leftrightarrow R_2$（便于讨论）：

$$\left(\begin{array}{cc|c} 1 & k & 1 \\ k & 1 & 1 \end{array}\right)$$

$R_2 \leftarrow R_2 - k R_1$：

$$\left(\begin{array}{cc|c} 1 & k & 1 \\ 0 & 1-k^2 & 1-k \end{array}\right)$$

注意 $1 - k^2 = (1-k)(1+k)$，分情况讨论：

**情形 1：$k \neq 1$ 且 $k \neq -1$（即 $1 - k^2 \neq 0$）**

第2行可化简为：$x_2 = \dfrac{1-k}{(1-k)(1+k)} = \dfrac{1}{1+k}$

代回第1行：$x_1 = 1 - k \cdot \dfrac{1}{1+k} = \dfrac{1+k-k}{1+k} = \dfrac{1}{1+k}$

**唯一解：** $x_1 = x_2 = \dfrac{1}{1+k}$

**情形 2：$k = 1$**

增广矩阵化为 $\left(\begin{array}{cc|c} 1 & 1 & 1 \\ 0 & 0 & 0 \end{array}\right)$

第2行全零，$x_2$ 为自由变量。令 $x_2 = t$，则 $x_1 = 1 - t$。

**无穷多解：** $\{(1-t,\; t) : t \in \mathbb{R}\}$

**情形 3：$k = -1$**

增广矩阵化为 $\left(\begin{array}{cc|c} 1 & -1 & 1 \\ 0 & 0 & 2 \end{array}\right)$

第2行对应 $0 = 2$，矛盾。**无解。**

</details>
