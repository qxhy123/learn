# 公式速查表

本文件汇总了线性代数中的常用公式，供学习和复习时快速查阅。

---

## 一、向量运算

### 向量加法与标量乘法

$$\mathbf{u} + \mathbf{v} = (u_1 + v_1, u_2 + v_2, \ldots, u_n + v_n)$$

$$c\mathbf{v} = (cv_1, cv_2, \ldots, cv_n)$$

### 内积（点积）

$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1v_1 + u_2v_2 + \cdots + u_nv_n$$

**矩阵形式**：$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v}$

### 向量范数

**2-范数（欧几里得范数）**：
$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{\mathbf{v}^T \mathbf{v}}$$

**1-范数**：
$$\|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|$$

**无穷范数**：
$$\|\mathbf{v}\|_\infty = \max_{1 \leq i \leq n} |v_i|$$

**p-范数**：
$$\|\mathbf{v}\|_p = \left(\sum_{i=1}^{n} |v_i|^p\right)^{1/p}$$

### 向量夹角

$$\cos\theta = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

### 叉积（三维）

$$\mathbf{u} \times \mathbf{v} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ u_1 & u_2 & u_3 \\ v_1 & v_2 & v_3 \end{vmatrix} = (u_2v_3 - u_3v_2, u_3v_1 - u_1v_3, u_1v_2 - u_2v_1)$$

### 投影

**$\mathbf{u}$ 在 $\mathbf{v}$ 上的投影**：
$$\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{v}\|^2} \mathbf{v}$$

---

## 二、矩阵运算

### 矩阵乘法

$$(AB)_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

**性质**：
- $(AB)C = A(BC)$（结合律）
- $A(B + C) = AB + AC$（分配律）
- $(AB)^T = B^T A^T$
- $(AB)^{-1} = B^{-1} A^{-1}$（若可逆）

### 转置

$$(A^T)_{ij} = a_{ji}$$

**性质**：
- $(A^T)^T = A$
- $(A + B)^T = A^T + B^T$
- $(cA)^T = cA^T$
- $(AB)^T = B^T A^T$

### 迹

$$\text{tr}(A) = \sum_{i=1}^{n} a_{ii}$$

**性质**：
- $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$
- $\text{tr}(cA) = c \cdot \text{tr}(A)$
- $\text{tr}(AB) = \text{tr}(BA)$
- $\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)$（循环性质）
- $\text{tr}(A) = \sum_{i=1}^{n} \lambda_i$（特征值之和）

### 逆矩阵

**定义**：$AA^{-1} = A^{-1}A = I$

**2×2矩阵的逆**：
$$\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad-bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

**性质**：
- $(A^{-1})^{-1} = A$
- $(AB)^{-1} = B^{-1}A^{-1}$
- $(A^T)^{-1} = (A^{-1})^T$
- $(cA)^{-1} = \frac{1}{c}A^{-1}$

---

## 三、行列式

### 2×2行列式

$$\begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc$$

### 3×3行列式（Sarrus法则）

$$\begin{vmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{vmatrix} = a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32} - a_{13}a_{22}a_{31} - a_{12}a_{21}a_{33} - a_{11}a_{23}a_{32}$$

### 行列式的性质

- $\det(A^T) = \det(A)$
- $\det(AB) = \det(A)\det(B)$
- $\det(A^{-1}) = \frac{1}{\det(A)}$
- $\det(cA) = c^n \det(A)$（$n$ 为矩阵阶数）
- 交换两行（列），行列式变号
- 一行（列）乘以常数加到另一行（列），行列式不变
- 行列式等于特征值的乘积：$\det(A) = \prod_{i=1}^{n} \lambda_i$

### 余子式与代数余子式

**余子式** $M_{ij}$：删除第 $i$ 行第 $j$ 列后的 $(n-1)$ 阶行列式

**代数余子式**：$A_{ij} = (-1)^{i+j} M_{ij}$

### 按行（列）展开

$$\det(A) = \sum_{j=1}^{n} a_{ij} A_{ij} = \sum_{i=1}^{n} a_{ij} A_{ij}$$

### 伴随矩阵

$$\text{adj}(A) = (A_{ij})^T$$

$$A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$$

---

## 四、线性方程组

### 矩阵形式

$$A\mathbf{x} = \mathbf{b}$$

### 解的情况

设 $A$ 为 $m \times n$ 矩阵，增广矩阵为 $(A|\mathbf{b})$：

- **有解条件**：$\text{rank}(A) = \text{rank}(A|\mathbf{b})$
- **唯一解**：$\text{rank}(A) = n$
- **无穷多解**：$\text{rank}(A) < n$
- **无解**：$\text{rank}(A) < \text{rank}(A|\mathbf{b})$

### Cramer法则

若 $\det(A) \neq 0$，则 $A\mathbf{x} = \mathbf{b}$ 的唯一解为：

$$x_i = \frac{\det(A_i)}{\det(A)}$$

其中 $A_i$ 是将 $A$ 的第 $i$ 列替换为 $\mathbf{b}$ 所得的矩阵。

---

## 五、向量空间

### 线性组合

$$\mathbf{v} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$$

### 线性无关

向量组 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ 线性无关当且仅当：
$$c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \Rightarrow c_1 = \cdots = c_k = 0$$

### 维数公式

$$\dim(V) = \text{rank}(A)$$

### 秩-零化度定理

$$\dim(\ker(A)) + \dim(\text{Im}(A)) = n$$

即：$\text{nullity}(A) + \text{rank}(A) = n$

### 四个基本子空间

对于 $m \times n$ 矩阵 $A$：

| 子空间 | 维数 |
|--------|------|
| 列空间 $\text{col}(A)$ | $\text{rank}(A) = r$ |
| 行空间 $\text{row}(A)$ | $\text{rank}(A) = r$ |
| 零空间 $\text{null}(A)$ | $n - r$ |
| 左零空间 $\text{null}(A^T)$ | $m - r$ |

---

## 六、内积空间

### 内积的性质

1. $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$（对称性）
2. $\langle c\mathbf{u}, \mathbf{v} \rangle = c\langle \mathbf{u}, \mathbf{v} \rangle$（齐次性）
3. $\langle \mathbf{u} + \mathbf{v}, \mathbf{w} \rangle = \langle \mathbf{u}, \mathbf{w} \rangle + \langle \mathbf{v}, \mathbf{w} \rangle$（可加性）
4. $\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$，等号当且仅当 $\mathbf{v} = \mathbf{0}$（正定性）

### Cauchy-Schwarz 不等式

$$|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\| \|\mathbf{v}\|$$

### 三角不等式

$$\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$$

### Gram-Schmidt 正交化

给定线性无关向量组 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$，构造正交向量组 $\{\mathbf{u}_1, \ldots, \mathbf{u}_k\}$：

$$\mathbf{u}_1 = \mathbf{v}_1$$

$$\mathbf{u}_j = \mathbf{v}_j - \sum_{i=1}^{j-1} \frac{\langle \mathbf{v}_j, \mathbf{u}_i \rangle}{\langle \mathbf{u}_i, \mathbf{u}_i \rangle} \mathbf{u}_i, \quad j = 2, \ldots, k$$

单位化：$\mathbf{e}_i = \frac{\mathbf{u}_i}{\|\mathbf{u}_i\|}$

### QR分解

$$A = QR$$

其中 $Q$ 是正交矩阵（$Q^T Q = I$），$R$ 是上三角矩阵。

---

## 七、特征值与特征向量

### 特征方程

$$\det(A - \lambda I) = 0$$

### 特征多项式

$$p(\lambda) = \det(A - \lambda I) = (-1)^n(\lambda^n - \text{tr}(A)\lambda^{n-1} + \cdots + (-1)^n\det(A))$$

### 特征值性质

- $\sum_{i=1}^{n} \lambda_i = \text{tr}(A)$
- $\prod_{i=1}^{n} \lambda_i = \det(A)$
- $A$ 可逆 $\Leftrightarrow$ 所有特征值非零
- $A^k$ 的特征值是 $\lambda^k$
- $A^{-1}$ 的特征值是 $\frac{1}{\lambda}$

### 对角化

若 $A$ 可对角化，存在可逆矩阵 $P$ 使得：
$$P^{-1}AP = D = \text{diag}(\lambda_1, \ldots, \lambda_n)$$

**可对角化条件**：$A$ 有 $n$ 个线性无关的特征向量。

### 矩阵幂

若 $A = PDP^{-1}$，则：
$$A^k = PD^kP^{-1}$$

### 实对称矩阵的谱定理

实对称矩阵 $A$ 可正交对角化：
$$A = Q\Lambda Q^T$$

其中 $Q$ 是正交矩阵，$\Lambda$ 是特征值对角矩阵。

---

## 八、奇异值分解 (SVD)

### SVD 分解

任意 $m \times n$ 矩阵 $A$ 可分解为：
$$A = U\Sigma V^T$$

- $U$：$m \times m$ 正交矩阵（左奇异向量）
- $\Sigma$：$m \times n$ 对角矩阵（奇异值）
- $V$：$n \times n$ 正交矩阵（右奇异向量）

### 奇异值

$$\sigma_i = \sqrt{\lambda_i(A^T A)} = \sqrt{\lambda_i(AA^T)}$$

### 紧凑形式

$$A = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

其中 $r = \text{rank}(A)$。

### 低秩近似

截断 SVD 给出最佳低秩近似（Eckart-Young定理）：
$$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

$$\min_{\text{rank}(B) \leq k} \|A - B\|_F = \|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \cdots + \sigma_r^2}$$

### 伪逆

$$A^+ = V\Sigma^+ U^T$$

其中 $\Sigma^+$ 是将 $\Sigma$ 中非零奇异值取倒数后转置。

---

## 九、二次型

### 二次型定义

$$q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i,j} a_{ij} x_i x_j$$

### 正定性判别

设 $A$ 是实对称矩阵：

| 性质 | 条件 |
|------|------|
| 正定 | 所有特征值 $> 0$，或所有顺序主子式 $> 0$ |
| 半正定 | 所有特征值 $\geq 0$ |
| 负定 | 所有特征值 $< 0$ |
| 半负定 | 所有特征值 $\leq 0$ |
| 不定 | 特征值有正有负 |

### 惯性定理

实对称矩阵的正特征值个数、负特征值个数、零特征值个数在合同变换下不变。

---

## 十、矩阵范数

### Frobenius 范数

$$\|A\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2} = \sqrt{\text{tr}(A^T A)} = \sqrt{\sum_{i=1}^{r} \sigma_i^2}$$

### 谱范数（2-范数）

$$\|A\|_2 = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^T A)}$$

### 核范数

$$\|A\|_* = \sum_{i=1}^{r} \sigma_i$$

---

## 十一、矩阵微积分

### 标量对向量求导

$$\frac{\partial f}{\partial \mathbf{x}} = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

### 常用梯度公式

| 函数 | 梯度 |
|------|------|
| $f(\mathbf{x}) = \mathbf{a}^T \mathbf{x}$ | $\nabla f = \mathbf{a}$ |
| $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ | $\nabla f = (A + A^T)\mathbf{x}$（若 $A$ 对称则为 $2A\mathbf{x}$） |
| $f(\mathbf{x}) = \|\mathbf{x}\|^2$ | $\nabla f = 2\mathbf{x}$ |
| $f(\mathbf{x}) = \|A\mathbf{x} - \mathbf{b}\|^2$ | $\nabla f = 2A^T(A\mathbf{x} - \mathbf{b})$ |

### 标量对矩阵求导

$$\frac{\partial f}{\partial A} = \begin{pmatrix} \frac{\partial f}{\partial a_{11}} & \cdots & \frac{\partial f}{\partial a_{1n}} \\ \vdots & \ddots & \vdots \\ \frac{\partial f}{\partial a_{m1}} & \cdots & \frac{\partial f}{\partial a_{mn}} \end{pmatrix}$$

### 常用矩阵导数

| 函数 | 导数 |
|------|------|
| $f(A) = \text{tr}(A)$ | $\frac{\partial f}{\partial A} = I$ |
| $f(A) = \text{tr}(AB)$ | $\frac{\partial f}{\partial A} = B^T$ |
| $f(A) = \text{tr}(A^T B)$ | $\frac{\partial f}{\partial A} = B$ |
| $f(A) = \det(A)$ | $\frac{\partial f}{\partial A} = \det(A) \cdot A^{-T}$ |
| $f(A) = \ln\det(A)$ | $\frac{\partial f}{\partial A} = A^{-T}$ |

### 链式法则

$$\frac{\partial f}{\partial \mathbf{x}} = \frac{\partial f}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = J^T \nabla_{\mathbf{y}} f$$

其中 $J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ 是 Jacobian 矩阵。
