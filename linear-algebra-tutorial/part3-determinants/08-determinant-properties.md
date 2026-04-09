# 第8章：行列式的性质与应用

> 行列式不只是一个数——它是矩阵"压缩空间"程度的精确度量，是线性代数中最深刻的不变量之一。

---

## 学习目标

完成本章学习后，你将能够：

- 掌握行列式的七条基本性质并理解其几何含义
- 利用行列式性质简化高阶行列式的计算
- 理解并计算范德蒙德行列式
- 掌握 Cramer 法则及其适用条件与局限
- 用行列式判断矩阵可逆性，并通过伴随矩阵法求逆
- 理解行列式与特征值的关系
- 掌握行列式在变换体积缩放、概率密度变换和 Normalizing Flows 中的应用

---

## 8.1 行列式的性质

行列式的强大之处在于一组系统的性质，它们既是计算工具，也是几何直觉的来源。以下以 $n$ 阶方阵 $A$ 为对象，逐一介绍这些性质。

### 性质一：转置不变性

$$\det(A^T) = \det(A)$$

**证明思路：** 对行列式的 Leibniz 公式，置换 $\sigma$ 与其逆 $\sigma^{-1}$ 一一对应，两者的符号相同（$\text{sgn}(\sigma) = \text{sgn}(\sigma^{-1})$），且遍历所有置换的乘积之和不变。$\square$

**几何意义：** 行和列的地位是对称的——对行成立的一切性质，对列也同样成立。这一性质使我们可以将所有行性质自动推广到列。

### 性质二：行（列）交换变号

交换矩阵的任意两行（或两列），行列式变号：

$$\det(\ldots, r_i, \ldots, r_j, \ldots) = -\det(\ldots, r_j, \ldots, r_i, \ldots)$$

**推论：** 若矩阵有两行（列）完全相同，则 $\det(A) = 0$。

**证明：** 设两行相同，交换这两行后行列式变号，但矩阵本身未变，故 $\det(A) = -\det(A)$，即 $\det(A) = 0$。$\square$

**示例：**

$$\det\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = -2, \quad \det\begin{pmatrix} 3 & 4 \\ 1 & 2 \end{pmatrix} = 3 \cdot 2 - 4 \cdot 1 = 2 = -(-2) \quad \checkmark$$

### 性质三：公因子提取

若某一行（列）的所有元素有公因子 $k$，则可将 $k$ 提到行列式外：

$$\det(\ldots, k \cdot r_i, \ldots) = k \cdot \det(\ldots, r_i, \ldots)$$

**推论：** $\det(kA) = k^n \det(A)$（对 $n$ 阶方阵，每一行都提出 $k$，共提 $n$ 次）。

**示例：**

$$\det\begin{pmatrix} 2 & 6 \\ 1 & 4 \end{pmatrix} = 2 \cdot \det\begin{pmatrix} 1 & 3 \\ 1 & 4 \end{pmatrix} = 2 \cdot (4 - 3) = 2$$

**注意：** $\det(A + B) \ne \det(A) + \det(B)$，行列式对整个矩阵**不**是线性的，仅对单独一行（列）是线性的。

### 性质四：行（列）加减不变

将某一行（列）的倍数加到另一行（列）上，行列式**不变**：

$$\det(\ldots, r_i + c \cdot r_j, \ldots, r_j, \ldots) = \det(\ldots, r_i, \ldots, r_j, \ldots)$$

**证明：** 利用行列式对行的线性性：

$$\det(\ldots, r_i + c \cdot r_j, \ldots) = \det(\ldots, r_i, \ldots) + c \cdot \det(\ldots, r_j, \ldots, r_j, \ldots)$$

由于最后一项中有两行相同，其行列式为 $0$，故等于原行列式。$\square$

**计算意义：** 这正是高斯消元的理论基础——对矩阵做初等行变换（倍加型），行列式不变。因此只要追踪交换操作，就能在化简过程中精确计算行列式。

### 性质五：若某行（列）全为零则行列式为零

$$\det(\ldots, \mathbf{0}, \ldots) = 0$$

**证明：** 将全零行提出公因子 $0$，得 $\det = 0 \cdot \det(\ldots) = 0$。$\square$

### 性质六：行列式的乘积公式

$$\det(AB) = \det(A) \cdot \det(B)$$

**这是行列式最重要的性质之一。**

**证明思路（简略）：** 若 $A$ 可逆，可将 $A$ 分解为初等矩阵之积 $A = E_1 E_2 \cdots E_k$，每个初等矩阵对行列式的作用已知（交换乘 $-1$，倍加不变，数乘乘 $c$），利用归纳可得等式成立。若 $A$ 奇异，则 $AB$ 也奇异，两侧均为 $0$。$\square$

**推论：**

$$\det(A^{-1}) = \frac{1}{\det(A)} \quad (\text{当} A \text{可逆时})$$

**证明：** 由 $AA^{-1} = I$，得 $\det(A)\det(A^{-1}) = \det(I) = 1$，因此 $\det(A^{-1}) = 1/\det(A)$。$\square$

### 性质七：三角矩阵的行列式

上三角矩阵（或下三角矩阵）的行列式等于**主对角线元素之积**：

$$\det\begin{pmatrix} a_{11} & * & \cdots & * \\ 0 & a_{22} & \cdots & * \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & a_{nn} \end{pmatrix} = a_{11} \cdot a_{22} \cdots a_{nn}$$

**计算意义：** 结合性质四（行加减不变），这给出了一种高效的行列式计算路线：通过高斯消元将矩阵化为上三角形，则行列式等于对角线元素之积（需乘上交换次数带来的 $(-1)^k$ 因子）。

---

## 8.2 行列式的计算技巧

### 化为三角形矩阵

**一般步骤：**

1. 对矩阵做初等行变换（允许行交换和倍加），化为上三角矩阵；
2. 记录行交换次数 $k$（每次交换贡献一个 $-1$）；
3. 行列式 = $(-1)^k \times$ 对角线元素之积。

**示例：** 计算

$$\det\begin{pmatrix} 2 & 1 & -1 \\ 0 & 3 & 2 \\ 4 & -1 & 1 \end{pmatrix}$$

$r_3 \leftarrow r_3 - 2r_1$（无交换，$k=0$）：

$$\begin{pmatrix} 2 & 1 & -1 \\ 0 & 3 & 2 \\ 0 & -3 & 3 \end{pmatrix}$$

$r_3 \leftarrow r_3 + r_2$：

$$\begin{pmatrix} 2 & 1 & -1 \\ 0 & 3 & 2 \\ 0 & 0 & 5 \end{pmatrix}$$

$$\det = (-1)^0 \times 2 \times 3 \times 5 = 30$$

### 利用行列式性质简化计算

**技巧一：提取公因子**

若某行（列）有明显公因子，先提出，降低计算复杂度。

**技巧二：制造零元素**

利用倍加变换，使某行（列）有尽可能多的零，然后按该行（列）展开（Laplace 展开）。

**示例：**

$$\det\begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & 2 & 1 & 1 \\ 1 & 1 & 3 & 1 \\ 1 & 1 & 1 & 4 \end{pmatrix}$$

$c_2 \leftarrow c_2 - c_1$，$c_3 \leftarrow c_3 - c_1$，$c_4 \leftarrow c_4 - c_1$：

$$\det\begin{pmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 0 & 2 & 0 \\ 1 & 0 & 0 & 3 \end{pmatrix}$$

按第一行展开，得 $\det = 1 \times \det\begin{pmatrix}1&0&0\\0&2&0\\0&0&3\end{pmatrix} = 1 \times 1 \times 2 \times 3 = 6$。

### 范德蒙德行列式

**定义：** $n$ 阶范德蒙德（Vandermonde）行列式定义为：

$$V_n = \det\begin{pmatrix} 1 & x_1 & x_1^2 & \cdots & x_1^{n-1} \\ 1 & x_2 & x_2^2 & \cdots & x_2^{n-1} \\ \vdots & \vdots & \vdots & & \vdots \\ 1 & x_n & x_n^2 & \cdots & x_n^{n-1} \end{pmatrix}$$

**结论：**

$$V_n = \prod_{1 \le i < j \le n} (x_j - x_i)$$

即所有 $x_j - x_i$（$j > i$）的乘积。

**推导思路（以 $n=3$ 为例）：**

$$V_3 = \det\begin{pmatrix} 1 & x_1 & x_1^2 \\ 1 & x_2 & x_2^2 \\ 1 & x_3 & x_3^2 \end{pmatrix}$$

$r_2 \leftarrow r_2 - r_1$，$r_3 \leftarrow r_3 - r_1$：

$$V_3 = \det\begin{pmatrix} 1 & x_1 & x_1^2 \\ 0 & x_2 - x_1 & x_2^2 - x_1^2 \\ 0 & x_3 - x_1 & x_3^2 - x_1^2 \end{pmatrix}$$

注意到 $x_k^2 - x_1^2 = (x_k - x_1)(x_k + x_1)$，从第 $2$、$3$ 行分别提出公因子 $(x_2 - x_1)$ 和 $(x_3 - x_1)$：

$$V_3 = (x_2 - x_1)(x_3 - x_1) \det\begin{pmatrix} 1 & x_1 & x_1^2 \\ 0 & 1 & x_2 + x_1 \\ 0 & 1 & x_3 + x_1 \end{pmatrix}$$

$r_3 \leftarrow r_3 - r_2$：

$$= (x_2 - x_1)(x_3 - x_1) \det\begin{pmatrix} 1 & x_1 & x_1^2 \\ 0 & 1 & x_2 + x_1 \\ 0 & 0 & x_3 - x_2 \end{pmatrix} = (x_2 - x_1)(x_3 - x_1)(x_3 - x_2)$$

**应用：** $V_n \ne 0$ 当且仅当所有 $x_i$ 互不相同。这在多项式插值（Lagrange 插值的唯一性）和编码理论（Reed-Solomon 码）中有核心应用。

---

## 8.3 Cramer 法则

### 内容

设 $A$ 是 $n \times n$ 可逆矩阵（即 $\det(A) \ne 0$），线性方程组 $A\mathbf{x} = \mathbf{b}$ 有唯一解，其第 $i$ 个分量为：

$$x_i = \frac{\det(A_i)}{\det(A)}, \quad i = 1, 2, \ldots, n$$

其中 $A_i$ 是将矩阵 $A$ 的第 $i$ 列替换为 $\mathbf{b}$ 所得到的矩阵：

$$A_i = \begin{pmatrix} | & & | & | & | & & | \\ \mathbf{a}_1 & \cdots & \mathbf{a}_{i-1} & \mathbf{b} & \mathbf{a}_{i+1} & \cdots & \mathbf{a}_n \\ | & & | & | & | & & | \end{pmatrix}$$

### 证明

设 $\mathbf{x} = (x_1, \ldots, x_n)^T$ 是方程组的解，则 $A\mathbf{x} = \mathbf{b}$，即

$$\mathbf{b} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n$$

将 $\mathbf{b}$ 代入 $A_i$ 的第 $i$ 列，按第 $i$ 列展开行列式，利用行列式的线性性：

$$\det(A_i) = \det(\ldots, \mathbf{a}_{i-1}, x_1\mathbf{a}_1 + \cdots + x_n\mathbf{a}_n, \mathbf{a}_{i+1}, \ldots)$$

由线性性展开，只有 $x_i \mathbf{a}_i$ 项（其余项因有两列相同而为零）：

$$= x_i \det(\mathbf{a}_1, \ldots, \mathbf{a}_{i-1}, \mathbf{a}_i, \mathbf{a}_{i+1}, \ldots, \mathbf{a}_n) = x_i \det(A)$$

因此 $x_i = \det(A_i) / \det(A)$。$\square$

### 计算示例

求解方程组

$$\begin{cases} 2x_1 + x_2 = 5 \\ x_1 + 3x_2 = 10 \end{cases}$$

$A = \begin{pmatrix}2 & 1 \\ 1 & 3\end{pmatrix}$，$\mathbf{b} = \begin{pmatrix}5 \\ 10\end{pmatrix}$，$\det(A) = 6 - 1 = 5$。

$$x_1 = \frac{\det\begin{pmatrix}5 & 1 \\ 10 & 3\end{pmatrix}}{\det(A)} = \frac{15 - 10}{5} = 1, \quad x_2 = \frac{\det\begin{pmatrix}2 & 5 \\ 1 & 10\end{pmatrix}}{\det(A)} = \frac{20 - 5}{5} = 3$$

### 使用条件与局限

**使用条件：** $A$ 必须是可逆方阵（$\det(A) \ne 0$）。对欠定或过定方程组，Cramer 法则不适用。

**局限性：**

| 方面 | 说明 |
|------|------|
| 计算复杂度 | 需计算 $n+1$ 个 $n$ 阶行列式，复杂度 $O(n \cdot n!) \sim O(n^{n+1})$，远高于高斯消元的 $O(n^3)$ |
| 实用性 | 对 $n \ge 4$，计算量急剧增长，工程中从不使用 |
| 理论价值 | 给出解的显式表达式，便于理论分析（如参数方程组的解析性、控制理论等） |

> **工程准则：** Cramer 法则是一个理论工具，实际求解线性方程组请使用高斯消元或 LU 分解。

---

## 8.4 行列式的应用

### 判断矩阵可逆性

**核心定理：** $n$ 阶方阵 $A$ 可逆当且仅当 $\det(A) \ne 0$。

这一结论是第6章"可逆矩阵定理"的重要一条，它将可逆性从"存在逆矩阵"这一定义转化为一个可以直接计算检验的数值条件。

**几何直觉：** $\det(A) = 0$ 意味着 $A$ 将 $n$ 维空间压缩到低维超平面，体积变为零，这一压缩操作不可逆。$\det(A) \ne 0$ 意味着变换保持体积（可能缩放但不归零），因此可逆。

### 计算逆矩阵（伴随矩阵法）

**代数余子式与余子矩阵：**

矩阵 $A$ 的 $(i,j)$ **代数余子式**（cofactor）定义为：

$$C_{ij} = (-1)^{i+j} M_{ij}$$

其中 $M_{ij}$ 是删去第 $i$ 行第 $j$ 列后剩余的 $(n-1)$ 阶子矩阵的行列式（称为**余子式**）。

**Laplace 展开：** 行列式可按任意一行（或列）展开：

$$\det(A) = \sum_{j=1}^n a_{ij} C_{ij} \quad (\text{按第} i \text{行展开})$$

**伴随矩阵（经典伴随矩阵，adjugate）：**

$$\text{adj}(A) = (C_{ij})^T$$

即代数余子式矩阵的转置，第 $(i,j)$ 元素为 $C_{ji}$（注意下标顺序）。

**逆矩阵公式：**

$$A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$$

**$2 \times 2$ 特例：**

$$\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

**$3 \times 3$ 示例：** 求

$$A = \begin{pmatrix} 1 & 2 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 2 \end{pmatrix}$$

的逆矩阵。

先计算 $\det(A)$（按第一列展开）：

$$\det(A) = 1 \cdot \det\begin{pmatrix}1&1\\0&2\end{pmatrix} - 0 + 1 \cdot \det\begin{pmatrix}2&0\\1&1\end{pmatrix} = 1 \cdot 2 + 1 \cdot 2 = 4$$

计算各代数余子式（以 $C_{11}, C_{12}, C_{13}$ 为例）：

$$C_{11} = (+1)\det\begin{pmatrix}1&1\\0&2\end{pmatrix} = 2, \quad C_{21} = (-1)\det\begin{pmatrix}2&0\\0&2\end{pmatrix} = -4, \quad C_{31} = (+1)\det\begin{pmatrix}2&0\\1&1\end{pmatrix} = 2$$

$$C_{12} = (-1)\det\begin{pmatrix}0&1\\1&2\end{pmatrix} = 1, \quad C_{22} = (+1)\det\begin{pmatrix}1&0\\1&2\end{pmatrix} = 2, \quad C_{32} = (-1)\det\begin{pmatrix}1&0\\0&1\end{pmatrix} = -1$$

$$C_{13} = (+1)\det\begin{pmatrix}0&1\\1&0\end{pmatrix} = -1, \quad C_{23} = (-1)\det\begin{pmatrix}1&2\\1&0\end{pmatrix} = 2, \quad C_{33} = (+1)\det\begin{pmatrix}1&2\\0&1\end{pmatrix} = 1$$

$$\text{adj}(A) = \begin{pmatrix} C_{11} & C_{21} & C_{31} \\ C_{12} & C_{22} & C_{32} \\ C_{13} & C_{23} & C_{33} \end{pmatrix} = \begin{pmatrix} 2 & -4 & 2 \\ 1 & 2 & -1 \\ -1 & 2 & 1 \end{pmatrix}$$

$$A^{-1} = \frac{1}{4}\begin{pmatrix} 2 & -4 & 2 \\ 1 & 2 & -1 \\ -1 & 2 & 1 \end{pmatrix}$$

**注意：** 伴随矩阵法理论上很优雅，但计算量为 $O(n^3)$（需计算 $n^2$ 个 $(n-1)$ 阶行列式），不如初等行变换法高效，实际中仅用于理论推导和 $2 \times 2$、$3 \times 3$ 的手算。

### 特征值与行列式

设 $A$ 是 $n \times n$ 方阵，$\lambda_1, \lambda_2, \ldots, \lambda_n$ 是 $A$ 的（含重数的）全部特征值，则：

$$\det(A) = \prod_{i=1}^n \lambda_i = \lambda_1 \lambda_2 \cdots \lambda_n$$

**证明：** 特征值是特征多项式 $\det(\lambda I - A) = 0$ 的根。将 $\det(\lambda I - A)$ 分解为 $(\lambda - \lambda_1)(\lambda - \lambda_2)\cdots(\lambda - \lambda_n)$，令 $\lambda = 0$，得 $\det(-A) = (-1)^n \det(A) = (-1)^n \lambda_1 \lambda_2 \cdots \lambda_n$，化简即得结论。$\square$

**推论：** $A$ 不可逆（奇异）当且仅当 $A$ 至少有一个特征值为零。

**迹与行列式：**

$$\text{tr}(A) = \sum_{i=1}^n a_{ii} = \sum_{i=1}^n \lambda_i$$

行列式和迹分别是特征多项式的常数项（取绝对值）和一次项系数，是矩阵最基本的两个不变量。

---

## 本章小结

| 性质/结论 | 内容 |
|-----------|------|
| 转置不变 | $\det(A^T) = \det(A)$ |
| 行交换变号 | 交换两行，行列式乘 $-1$ |
| 公因子提取 | $\det(kA) = k^n \det(A)$ |
| 倍加不变 | 行的倍数加到另一行，$\det$ 不变 |
| 乘积公式 | $\det(AB) = \det(A)\det(B)$ |
| 三角矩阵 | $\det$ = 主对角线之积 |
| 范德蒙德 | $V_n = \prod_{j>i}(x_j - x_i)$ |
| 可逆判据 | $A$ 可逆 $\Leftrightarrow$ $\det(A) \ne 0$ |
| 逆矩阵公式 | $A^{-1} = \dfrac{1}{\det(A)}\text{adj}(A)$ |
| Cramer 法则 | $x_i = \det(A_i)/\det(A)$，仅理论适用 |
| 特征值之积 | $\det(A) = \prod_i \lambda_i$ |

**核心要点回顾：**

1. 行列式的七条性质构成完整的计算体系：转置不变让行列对称，交换变号让重复行为零，倍加不变让高斯消元保持行列式值。
2. 范德蒙德行列式 $V_n = \prod_{j>i}(x_j - x_i)$ 是多项式插值唯一性的代数保证。
3. Cramer 法则理论优美，但计算复杂度 $O(n^{n+1})$ 远超高斯消元的 $O(n^3)$，仅供理论分析。
4. 伴随矩阵法给出逆矩阵的显式公式 $A^{-1} = \text{adj}(A)/\det(A)$，实用性受限于计算量。
5. 行列式是特征值之积，零特征值等价于矩阵奇异。

---

## 深度学习应用

### 背景：行列式作为"体积缩放因子"

行列式最本质的几何含义是：矩阵 $A$ 对应的线性变换将 $n$ 维单位超立方体变换后，新体积是原体积的 $|\det(A)|$ 倍。这一性质在深度学习中有三个重要应用场景。

### 变换的体积缩放因子

设线性变换 $\mathbf{y} = A\mathbf{x}$，若 $\mathbf{x}$ 服从均匀分布（在某区域 $\Omega$ 上），则 $\mathbf{y}$ 的分布区域体积是原区域的 $|\det(A)|$ 倍。

更一般地，对于可微变换 $\mathbf{y} = f(\mathbf{x})$，**Jacobi 矩阵**定义为：

$$J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{pmatrix} \frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial y_n}{\partial x_1} & \cdots & \frac{\partial y_n}{\partial x_n} \end{pmatrix}$$

局部体积缩放因子为 $|\det(J)|$（称为 Jacobi 行列式）。这是多变量微积分换元公式的核心：

$$\int_{f(\Omega)} g(\mathbf{y})\, d\mathbf{y} = \int_\Omega g(f(\mathbf{x})) \left|\det\frac{\partial f}{\partial \mathbf{x}}\right| d\mathbf{x}$$

### 概率密度的变换公式

若随机变量 $\mathbf{x}$ 的概率密度函数为 $p_X(\mathbf{x})$，通过可逆变换 $\mathbf{y} = f(\mathbf{x})$ 得到 $\mathbf{y}$，则 $\mathbf{y}$ 的密度为：

$$p_Y(\mathbf{y}) = p_X(f^{-1}(\mathbf{y})) \cdot \left|\det\frac{\partial f^{-1}}{\partial \mathbf{y}}\right|$$

等价地写为：

$$p_Y(\mathbf{y}) = p_X(\mathbf{x}) \cdot \left|\det\frac{\partial f}{\partial \mathbf{x}}\right|^{-1}$$

其中分母中的 Jacobi 行列式起到"体积补偿"的作用——变换"拉伸"了空间（$|\det J| > 1$），密度就要相应减小。

### Normalizing Flows 中的对数行列式

**Normalizing Flows** 是一类生成模型，通过一系列可逆变换将简单分布（如标准高斯）映射到复杂数据分布。设 $\mathbf{z} \sim p_Z(\mathbf{z})$（简单先验），经过 $K$ 个可逆变换：

$$\mathbf{x} = f_K \circ f_{K-1} \circ \cdots \circ f_1(\mathbf{z})$$

由概率密度变换公式，对数似然为：

$$\log p_X(\mathbf{x}) = \log p_Z(\mathbf{z}) - \sum_{k=1}^K \log \left|\det J_{f_k}(\mathbf{z}_{k-1})\right|$$

**关键挑战：** 每个变换 $f_k$ 必须满足两个条件：
1. **可逆**：使得前向和逆向传播均可计算；
2. **行列式高效**：$\det J_{f_k}$ 能在 $O(n)$ 或 $O(n\log n)$ 而非 $O(n^3)$ 时间内计算。

**两类主流设计：**

**（1）三角 Jacobi（耦合层，如 RealNVP）**

将变量分为两组 $(\mathbf{x}_A, \mathbf{x}_B)$，令

$$\mathbf{y}_A = \mathbf{x}_A, \quad \mathbf{y}_B = \mathbf{x}_B \odot \exp(s(\mathbf{x}_A)) + t(\mathbf{x}_A)$$

其 Jacobi 矩阵是块三角形，行列式等于对角线元素之积：

$$\log|\det J| = \sum_i s_i(\mathbf{x}_A)$$

只需 $O(n)$ 时间。

**（2）自回归变换（如 IAF、MAF）**

$$x_i = \mu_i(x_{1:i-1}) + \sigma_i(x_{1:i-1}) \cdot z_i$$

Jacobi 矩阵为下三角，$\log|\det J| = \sum_i \log \sigma_i$，同样 $O(n)$。

### 代码示例

```python
import numpy as np
import torch
import torch.nn as nn

# ── 示例1：行列式性质验证 ────────────────────────────────────────
A = np.array([[1., 2., 0.],
              [0., 1., 1.],
              [1., 0., 2.]])
B = np.array([[2., -1., 0.],
              [1.,  3., 1.],
              [0.,  1., 2.]])

# 性质六：det(AB) = det(A) * det(B)
det_A  = np.linalg.det(A)
det_B  = np.linalg.det(B)
det_AB = np.linalg.det(A @ B)
print(f"det(A) = {det_A:.4f}")
print(f"det(B) = {det_B:.4f}")
print(f"det(A)*det(B) = {det_A * det_B:.4f}")
print(f"det(AB) = {det_AB:.4f}")
print(f"乘积公式成立: {np.isclose(det_AB, det_A * det_B)}")

# 转置不变性：det(A^T) = det(A)
print(f"det(A^T) = {np.linalg.det(A.T):.4f}, det(A) = {det_A:.4f}")
print(f"转置不变性成立: {np.isclose(np.linalg.det(A.T), det_A)}")

# 行交换变号
A_swap = A[[1, 0, 2], :]  # 交换第0和第1行
print(f"行交换后 det = {np.linalg.det(A_swap):.4f}，应等于 {-det_A:.4f}")

# ── 示例2：通过行化简计算行列式 ───────────────────────────────────
def det_by_elimination(M):
    """通过高斯消元计算行列式（演示用途）"""
    A = M.copy().astype(float)
    n = len(A)
    sign = 1
    for col in range(n):
        # 寻找主元
        pivot_row = np.argmax(np.abs(A[col:, col])) + col
        if abs(A[pivot_row, col]) < 1e-12:
            return 0.0
        if pivot_row != col:
            A[[col, pivot_row]] = A[[pivot_row, col]]
            sign *= -1
        for row in range(col + 1, n):
            factor = A[row, col] / A[col, col]
            A[row] -= factor * A[col]
    diag_product = np.prod(np.diag(A))
    return sign * diag_product

M = np.array([[2., 1., -1.],
              [0., 3.,  2.],
              [4., -1., 1.]])
print(f"\n手动消元: det = {det_by_elimination(M):.4f}")
print(f"numpy 验证: det = {np.linalg.det(M):.4f}")

# ── 示例3：范德蒙德行列式 ────────────────────────────────────────
def vandermonde_det_formula(xs):
    """用公式计算范德蒙德行列式"""
    n = len(xs)
    result = 1.0
    for j in range(n):
        for i in range(j):
            result *= (xs[j] - xs[i])
    return result

def vandermonde_matrix(xs):
    n = len(xs)
    return np.array([[x**k for k in range(n)] for x in xs], dtype=float)

xs = [1.0, 2.0, 3.0, 4.0]
V = vandermonde_matrix(xs)
det_formula = vandermonde_det_formula(xs)
det_numpy   = np.linalg.det(V)
print(f"\n范德蒙德行列式（公式）: {det_formula:.4f}")
print(f"范德蒙德行列式（numpy）: {det_numpy:.4f}")
print(f"两者一致: {np.isclose(det_formula, det_numpy)}")

# ── 示例4：Normalizing Flow（耦合层）中的对数行列式 ───────────────
class AffineCouplingLayer(nn.Module):
    """
    RealNVP 风格的仿射耦合层
    将输入 x 分为两半，下半变换依赖上半：
        y[:d] = x[:d]
        y[d:] = x[d:] * exp(s(x[:d])) + t(x[:d])
    log|det J| = sum(s(x[:d]))，复杂度 O(d)
    """
    def __init__(self, dim, hidden=32):
        super().__init__()
        d = dim // 2
        self.d = d
        # 网络预测缩放 s 和平移 t
        self.scale_net = nn.Sequential(
            nn.Linear(d, hidden), nn.Tanh(),
            nn.Linear(hidden, dim - d)
        )
        self.translate_net = nn.Sequential(
            nn.Linear(d, hidden), nn.Tanh(),
            nn.Linear(hidden, dim - d)
        )

    def forward(self, x):
        """前向变换，返回 (y, log_det_J)"""
        x_A = x[:, :self.d]            # 保持不变的部分
        x_B = x[:, self.d:]            # 被变换的部分
        s = self.scale_net(x_A)        # 缩放参数
        t = self.translate_net(x_A)    # 平移参数
        y_A = x_A
        y_B = x_B * torch.exp(s) + t
        y = torch.cat([y_A, y_B], dim=1)
        # Jacobi 是块三角，log|det J| = sum(s)，复杂度 O(d)
        log_det_J = s.sum(dim=1)
        return y, log_det_J

    def inverse(self, y):
        """逆变换"""
        y_A = y[:, :self.d]
        y_B = y[:, self.d:]
        s = self.scale_net(y_A)
        t = self.translate_net(y_A)
        x_A = y_A
        x_B = (y_B - t) * torch.exp(-s)
        return torch.cat([x_A, x_B], dim=1)

# 演示：验证对数行列式计算
torch.manual_seed(42)
dim = 4
layer = AffineCouplingLayer(dim)

x = torch.randn(3, dim)             # 3 个样本，4 维
y, log_det = layer(x)

# 用 autograd 数值验证 log|det J|
def numerical_log_det(f, x_single):
    """数值计算单样本的 log|det J|"""
    x_var = x_single.unsqueeze(0).requires_grad_(True)
    y_var = f(x_var)[0]
    J = torch.zeros(dim, dim)
    for i in range(dim):
        grad = torch.autograd.grad(y_var[0, i], x_var,
                                   retain_graph=True)[0]
        J[i] = grad.squeeze()
    return torch.log(torch.abs(torch.det(J)))

for i in range(3):
    numerical = numerical_log_det(layer, x[i])
    analytic  = log_det[i]
    print(f"样本 {i}: 解析值 = {analytic:.4f}, 数值验证 = {numerical:.4f}, "
          f"一致: {torch.isclose(analytic, numerical, atol=1e-4).item()}")
```

### 延伸阅读

- **《Deep Learning》**（Goodfellow et al.）第3章：概率与信息论，变量变换公式
- **「Density estimation using Real-valued Non-Volume Preserving (Real NVP) transformations」**（Dinh et al., 2017）：耦合层设计
- **「Normalizing Flows: An Introduction and Review of Current Methods」**（Kobyzev et al., 2020）：综述
- **《Matrix Analysis》**（Horn & Johnson）第0-1章：行列式的严格理论
- **「The Matrix Cookbook」**（Petersen & Pedersen）：行列式公式速查

---

## 练习题

**练习 8.1（行列式性质应用）**

设 $A$ 是 $4 \times 4$ 矩阵，$\det(A) = 3$。计算以下各值：

(a) $\det(2A)$

(b) $\det(A^T A)$

(c) $\det(A^{-1})$

(d) $\det(-A)$

---

**练习 8.2（化三角形法计算行列式）**

利用行变换将矩阵化为上三角形，计算下列行列式：

$$D = \det\begin{pmatrix} 0 & 2 & 1 \\ 1 & -1 & 3 \\ 2 & 1 & 0 \end{pmatrix}$$

要求写出每步变换过程，并说明符号的变化。

---

**练习 8.3（范德蒙德行列式）**

(a) 直接展开计算

$$V = \det\begin{pmatrix} 1 & 1 & 1 \\ a & b & c \\ a^2 & b^2 & c^2 \end{pmatrix}$$

验证结果等于 $(b-a)(c-a)(c-b)$。

(b) 设 $x_1, x_2, x_3, x_4$ 是互不相同的实数，写出 $V_4$ 的乘积公式（无需展开计算），并说明：当 $x_i = i-1$（即 $0,1,2,3$）时，$V_4$ 的值是多少？

---

**练习 8.4（Cramer 法则）**

用 Cramer 法则求解方程组：

$$\begin{cases} x_1 - x_2 + 2x_3 = 1 \\ 2x_1 + x_2 - x_3 = 2 \\ x_1 + 3x_2 - x_3 = 4 \end{cases}$$

写出每个行列式的计算过程。

---

**练习 8.5（伴随矩阵法与行列式应用）**

设矩阵

$$A = \begin{pmatrix} 2 & 0 & 1 \\ 1 & 1 & 0 \\ 0 & 1 & 2 \end{pmatrix}$$

(a) 计算 $\det(A)$，判断 $A$ 是否可逆。

(b) 计算所有代数余子式，构造 $\text{adj}(A)$，并用公式求 $A^{-1}$。

(c) 验证 $A \cdot A^{-1} = I$。

(d) 若 $A$ 的特征值之积等于 $\det(A)$，求 $A$ 的特征值之积，并简述其意义。

---

## 练习答案

<details>
<summary>练习 8.1 答案</summary>

**(a)** $\det(2A) = 2^4 \det(A) = 16 \times 3 = 48$

（$n=4$，$\det(kA) = k^n \det(A)$）

**(b)** $\det(A^T A) = \det(A^T)\det(A) = \det(A)^2 = 3^2 = 9$

（利用乘积公式和转置不变性）

**(c)** $\det(A^{-1}) = \dfrac{1}{\det(A)} = \dfrac{1}{3}$

（由 $AA^{-1} = I$ 得 $\det(A)\det(A^{-1}) = 1$）

**(d)** $\det(-A) = (-1)^4 \det(A) = 1 \times 3 = 3$

（$-A = (-1)A$，$n=4$，$(-1)^4 = 1$）

</details>

<details>
<summary>练习 8.2 答案</summary>

原矩阵：

$$\begin{pmatrix} 0 & 2 & 1 \\ 1 & -1 & 3 \\ 2 & 1 & 0 \end{pmatrix}$$

**步骤1：** $r_1 \leftrightarrow r_2$（一次行交换，符号变号，$k=1$）：

$$\begin{pmatrix} 1 & -1 & 3 \\ 0 & 2 & 1 \\ 2 & 1 & 0 \end{pmatrix}$$

**步骤2：** $r_3 \leftarrow r_3 - 2r_1$（倍加，不改变行列式）：

$$\begin{pmatrix} 1 & -1 & 3 \\ 0 & 2 & 1 \\ 0 & 3 & -6 \end{pmatrix}$$

**步骤3：** $r_3 \leftarrow r_3 - \dfrac{3}{2} r_2$（倍加，不改变行列式）：

$$\begin{pmatrix} 1 & -1 & 3 \\ 0 & 2 & 1 \\ 0 & 0 & -\frac{15}{2} \end{pmatrix}$$

对角线元素之积：$1 \times 2 \times \left(-\dfrac{15}{2}\right) = -15$

$$D = (-1)^1 \times (-15) = 15$$

**验证：** 直接按第一列展开原矩阵：

$$D = 0 \cdot (\cdots) - 1 \cdot \det\begin{pmatrix}2&1\\1&0\end{pmatrix} + 2 \cdot \det\begin{pmatrix}2&1\\-1&3\end{pmatrix}$$

$$= 0 - 1 \cdot (0-1) + 2 \cdot (6+1) = 1 + 14 = 15 \quad \checkmark$$

</details>

<details>
<summary>练习 8.3 答案</summary>

**(a)** $r_2 \leftarrow r_2 - a \cdot r_1$，$r_3 \leftarrow r_3 - a^2 \cdot r_1$：

$$\det\begin{pmatrix} 1 & 1 & 1 \\ 0 & b-a & c-a \\ 0 & b^2-a^2 & c^2-a^2 \end{pmatrix}$$

提出公因子：$r_2$ 无公因子，$r_3$ 注意 $b^2-a^2=(b-a)(b+a)$，$c^2-a^2=(c-a)(c+a)$，无法直接提。

展开（按第一列）：

$$= \det\begin{pmatrix} b-a & c-a \\ b^2-a^2 & c^2-a^2 \end{pmatrix}$$

$$= (b-a)(c^2-a^2) - (c-a)(b^2-a^2)$$

$$= (b-a)(c-a)(c+a) - (c-a)(b-a)(b+a)$$

$$= (b-a)(c-a)[(c+a) - (b+a)]$$

$$= (b-a)(c-a)(c-b) \quad \checkmark$$

**(b)** $V_4 = \prod_{1 \le i < j \le 4}(x_j - x_i)$，共 $\binom{4}{2}=6$ 个因子：

$$V_4 = (x_2-x_1)(x_3-x_1)(x_4-x_1)(x_3-x_2)(x_4-x_2)(x_4-x_3)$$

代入 $x_1=0, x_2=1, x_3=2, x_4=3$：

$$V_4 = (1-0)(2-0)(3-0)(2-1)(3-1)(3-2) = 1 \cdot 2 \cdot 3 \cdot 1 \cdot 2 \cdot 1 = 12$$

</details>

<details>
<summary>练习 8.4 答案</summary>

系数矩阵

$$A = \begin{pmatrix}1&-1&2\\2&1&-1\\1&3&-1\end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix}1\\2\\4\end{pmatrix}$$

**计算 $\det(A)$（按第一行展开）：**

$$\det(A) = 1 \cdot \det\begin{pmatrix}1&-1\\3&-1\end{pmatrix} - (-1)\det\begin{pmatrix}2&-1\\1&-1\end{pmatrix} + 2\det\begin{pmatrix}2&1\\1&3\end{pmatrix}$$

$$= 1 \cdot (-1+3) + 1 \cdot (-2+1) + 2 \cdot (6-1) = 2 - 1 + 10 = 11$$

**计算 $\det(A_1)$（用 $\mathbf{b}$ 替换第1列）：**

$$A_1 = \begin{pmatrix}1&-1&2\\2&1&-1\\4&3&-1\end{pmatrix}$$

$$\det(A_1) = 1\cdot\det\begin{pmatrix}1&-1\\3&-1\end{pmatrix} - (-1)\det\begin{pmatrix}2&-1\\4&-1\end{pmatrix} + 2\det\begin{pmatrix}2&1\\4&3\end{pmatrix}$$

$$= 1\cdot(-1+3) + 1\cdot(-2+4) + 2\cdot(6-4) = 2 + 2 + 4 = 8 \quad \Rightarrow \quad x_1 = \frac{8}{11}$$

**计算 $\det(A_2)$（用 $\mathbf{b}$ 替换第2列）：**

$$A_2 = \begin{pmatrix}1&1&2\\2&2&-1\\1&4&-1\end{pmatrix}$$

$$\det(A_2) = 1\cdot\det\begin{pmatrix}2&-1\\4&-1\end{pmatrix} - 1\cdot\det\begin{pmatrix}2&-1\\1&-1\end{pmatrix} + 2\cdot\det\begin{pmatrix}2&2\\1&4\end{pmatrix}$$

$$= 1\cdot(-2+4) - 1\cdot(-2+1) + 2\cdot(8-2) = 2 + 1 + 12 = 15 \quad \Rightarrow \quad x_2 = \frac{15}{11}$$

**计算 $\det(A_3)$（用 $\mathbf{b}$ 替换第3列）：**

$$A_3 = \begin{pmatrix}1&-1&1\\2&1&2\\1&3&4\end{pmatrix}$$

$$\det(A_3) = 1\cdot\det\begin{pmatrix}1&2\\3&4\end{pmatrix} - (-1)\det\begin{pmatrix}2&2\\1&4\end{pmatrix} + 1\cdot\det\begin{pmatrix}2&1\\1&3\end{pmatrix}$$

$$= 1\cdot(4-6) + 1\cdot(8-2) + 1\cdot(6-1) = -2 + 6 + 5 = 9 \quad \Rightarrow \quad x_3 = \frac{9}{11}$$

$$\boxed{x_1 = \frac{8}{11}, \quad x_2 = \frac{15}{11}, \quad x_3 = \frac{9}{11}}$$

</details>

<details>
<summary>练习 8.5 答案</summary>

**(a)** 按第一行展开：

$$\det(A) = 2\det\begin{pmatrix}1&0\\1&2\end{pmatrix} - 0 + 1\det\begin{pmatrix}1&1\\0&1\end{pmatrix} = 2\cdot 2 + 1\cdot 1 = 5$$

$\det(A) = 5 \ne 0$，$A$ 可逆。

**(b)** 计算全部 9 个代数余子式：

$$C_{11} = (+1)\det\begin{pmatrix}1&0\\1&2\end{pmatrix} = 2$$

$$C_{12} = (-1)\det\begin{pmatrix}1&0\\0&2\end{pmatrix} = -2$$

$$C_{13} = (+1)\det\begin{pmatrix}1&1\\0&1\end{pmatrix} = 1$$

$$C_{21} = (-1)\det\begin{pmatrix}0&1\\1&2\end{pmatrix} = -(-1) = 1$$

$$C_{22} = (+1)\det\begin{pmatrix}2&1\\0&2\end{pmatrix} = 4$$

$$C_{23} = (-1)\det\begin{pmatrix}2&0\\0&1\end{pmatrix} = -2$$

$$C_{31} = (+1)\det\begin{pmatrix}0&1\\1&0\end{pmatrix} = -1$$

$$C_{32} = (-1)\det\begin{pmatrix}2&1\\1&0\end{pmatrix} = -(-1) = 1$$

$$C_{33} = (+1)\det\begin{pmatrix}2&0\\1&1\end{pmatrix} = 2$$

伴随矩阵（代数余子式矩阵的转置）：

$$\text{adj}(A) = \begin{pmatrix} C_{11} & C_{21} & C_{31} \\ C_{12} & C_{22} & C_{32} \\ C_{13} & C_{23} & C_{33} \end{pmatrix} = \begin{pmatrix} 2 & 1 & -1 \\ -2 & 4 & 1 \\ 1 & -2 & 2 \end{pmatrix}$$

$$A^{-1} = \frac{1}{5}\begin{pmatrix} 2 & 1 & -1 \\ -2 & 4 & 1 \\ 1 & -2 & 2 \end{pmatrix}$$

**(c)** 验证 $A \cdot A^{-1} = I$：

$$\begin{pmatrix}2&0&1\\1&1&0\\0&1&2\end{pmatrix} \cdot \frac{1}{5}\begin{pmatrix}2&1&-1\\-2&4&1\\1&-2&2\end{pmatrix}$$

$$= \frac{1}{5}\begin{pmatrix} 4+0+1 & 2+0-2 & -2+0+2 \\ 2-2+0 & 1+4+0 & -1+1+0 \\ 0-2+2 & 0+4-4 & 0+1+4 \end{pmatrix} = \frac{1}{5}\begin{pmatrix}5&0&0\\0&5&0\\0&0&5\end{pmatrix} = I \quad \checkmark$$

**(d)** 特征值之积 $= \det(A) = 5$。

含义：$A$ 的三个特征值（计重数）之积为 $5$，反映了矩阵对应的线性变换将单位超立方体的体积放大为 $5$ 倍（有向体积）。由于 $\det(A) > 0$，变换保持方向性（不翻转空间）。

</details>
