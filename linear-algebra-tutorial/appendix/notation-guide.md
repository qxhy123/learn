# 符号说明

本文件汇总了全书使用的数学符号及其含义，方便读者随时查阅。

---

## 集合与数系

| 符号 | 含义 | 示例 |
|------|------|------|
| $\mathbb{N}$ | 自然数集 | $\{0, 1, 2, 3, \ldots\}$ |
| $\mathbb{Z}$ | 整数集 | $\{\ldots, -2, -1, 0, 1, 2, \ldots\}$ |
| $\mathbb{Q}$ | 有理数集 | $\{p/q : p, q \in \mathbb{Z}, q \neq 0\}$ |
| $\mathbb{R}$ | 实数集 | 所有实数 |
| $\mathbb{C}$ | 复数集 | $\{a + bi : a, b \in \mathbb{R}\}$ |
| $\mathbb{R}^n$ | $n$ 维实向量空间 | $\mathbb{R}^3$ 是三维空间 |
| $\mathbb{R}^{m \times n}$ | $m \times n$ 实矩阵空间 | $\mathbb{R}^{2 \times 3}$ 是2行3列矩阵 |
| $\in$ | 属于 | $x \in \mathbb{R}$ |
| $\subset$ | 真子集 | $\mathbb{Q} \subset \mathbb{R}$ |
| $\subseteq$ | 子集 | $A \subseteq B$ |

---

## 向量

| 符号 | 含义 | 说明 |
|------|------|------|
| $\mathbf{v}, \mathbf{u}, \mathbf{w}$ | 向量 | 粗体小写字母表示向量 |
| $\vec{v}$ | 向量（箭头记法） | 等同于 $\mathbf{v}$ |
| $\mathbf{0}$ | 零向量 | 所有分量为0的向量 |
| $\mathbf{e}_i$ | 标准基向量 | 第 $i$ 个分量为1，其余为0 |
| $v_i$ 或 $[\mathbf{v}]_i$ | 向量的第 $i$ 个分量 | $\mathbf{v} = (v_1, v_2, \ldots, v_n)$ |
| $\|\mathbf{v}\|$ | 向量的范数（长度） | 默认为2-范数 $\sqrt{\sum v_i^2}$ |
| $\|\mathbf{v}\|_1$ | 1-范数 | $\sum |v_i|$ |
| $\|\mathbf{v}\|_2$ | 2-范数（欧几里得范数） | $\sqrt{\sum v_i^2}$ |
| $\|\mathbf{v}\|_\infty$ | 无穷范数 | $\max |v_i|$ |
| $\langle \mathbf{u}, \mathbf{v} \rangle$ | 内积 | $\sum u_i v_i$ |
| $\mathbf{u} \cdot \mathbf{v}$ | 点积 | 等同于 $\langle \mathbf{u}, \mathbf{v} \rangle$ |
| $\mathbf{u} \times \mathbf{v}$ | 叉积（仅限三维） | 结果是向量 |
| $\mathbf{u} \perp \mathbf{v}$ | 正交 | $\langle \mathbf{u}, \mathbf{v} \rangle = 0$ |

---

## 矩阵

| 符号 | 含义 | 说明 |
|------|------|------|
| $A, B, C$ | 矩阵 | 大写字母表示矩阵 |
| $a_{ij}$ 或 $[A]_{ij}$ | 矩阵元素 | 第 $i$ 行第 $j$ 列的元素 |
| $I$ 或 $I_n$ | 单位矩阵 | $n \times n$ 对角线为1的矩阵 |
| $O$ | 零矩阵 | 所有元素为0的矩阵 |
| $A^T$ | 转置 | $(A^T)_{ij} = a_{ji}$ |
| $A^{-1}$ | 逆矩阵 | $AA^{-1} = A^{-1}A = I$ |
| $A^*$ 或 $A^H$ | 共轭转置（Hermitian转置） | $(A^*)_{ij} = \overline{a_{ji}}$ |
| $A^+$ | Moore-Penrose伪逆 | 最小二乘意义下的逆 |
| $\det(A)$ 或 $|A|$ | 行列式 | 方阵的行列式值 |
| $\text{tr}(A)$ | 迹 | 对角元素之和 $\sum a_{ii}$ |
| $\text{rank}(A)$ | 秩 | 线性无关行（列）的最大数目 |
| $\text{null}(A)$ 或 $\ker(A)$ | 零空间（核） | $\{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$ |
| $\text{col}(A)$ 或 $\text{Im}(A)$ | 列空间（像） | $A$ 的列向量张成的空间 |
| $\text{row}(A)$ | 行空间 | $A$ 的行向量张成的空间 |

---

## 特殊矩阵

| 符号 | 名称 | 定义 |
|------|------|------|
| 对角矩阵 | $\text{diag}(d_1, \ldots, d_n)$ | 非对角元素为0 |
| 对称矩阵 | $A = A^T$ | 关于主对角线对称 |
| 反对称矩阵 | $A = -A^T$ | $a_{ij} = -a_{ji}$ |
| 正交矩阵 | $Q^T Q = I$ | 列向量两两正交且单位长度 |
| 酉矩阵 | $U^* U = I$ | 复数域上的正交矩阵 |
| 正定矩阵 | $\mathbf{x}^T A \mathbf{x} > 0$ | 对所有非零向量 $\mathbf{x}$ |
| 半正定矩阵 | $\mathbf{x}^T A \mathbf{x} \geq 0$ | 对所有向量 $\mathbf{x}$ |
| 上三角矩阵 | | 主对角线以下元素为0 |
| 下三角矩阵 | | 主对角线以上元素为0 |

---

## 特征值与特征向量

| 符号 | 含义 |
|------|------|
| $\lambda$ | 特征值 |
| $\mathbf{v}$ | 特征向量 |
| $\sigma(A)$ | $A$ 的谱（所有特征值的集合） |
| $\rho(A)$ | 谱半径，$\max |\lambda_i|$ |
| $\lambda_{\max}(A)$ | 最大特征值 |
| $\lambda_{\min}(A)$ | 最小特征值 |

---

## 奇异值分解

| 符号 | 含义 |
|------|------|
| $\sigma_i$ | 第 $i$ 个奇异值 |
| $U$ | 左奇异向量矩阵 |
| $V$ | 右奇异向量矩阵 |
| $\Sigma$ | 奇异值对角矩阵 |
| $A = U \Sigma V^T$ | SVD分解 |

---

## 向量空间

| 符号 | 含义 |
|------|------|
| $V, W$ | 向量空间 |
| $\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k)$ | 向量的张成 |
| $\dim(V)$ | 向量空间的维数 |
| $V \oplus W$ | 直和 |
| $V^\perp$ | 正交补 |

---

## 线性变换

| 符号 | 含义 |
|------|------|
| $T: V \to W$ | 从 $V$ 到 $W$ 的线性变换 |
| $\ker(T)$ | $T$ 的核（零空间） |
| $\text{Im}(T)$ 或 $\text{range}(T)$ | $T$ 的像（值域） |
| $[T]_{\mathcal{B}}$ | $T$ 在基 $\mathcal{B}$ 下的矩阵表示 |

---

## 其他常用符号

| 符号 | 含义 |
|------|------|
| $:=$ 或 $\triangleq$ | 定义为 |
| $\forall$ | 对于所有 |
| $\exists$ | 存在 |
| $\Rightarrow$ | 蕴含 |
| $\Leftrightarrow$ | 当且仅当 |
| $\sum$ | 求和 |
| $\prod$ | 求积 |
| $\arg\min$ | 使目标函数最小的参数 |
| $\arg\max$ | 使目标函数最大的参数 |

---

## 深度学习相关符号

| 符号 | 含义 | 说明 |
|------|------|------|
| $W$ | 权重矩阵 | 神经网络层的参数 |
| $\mathbf{b}$ | 偏置向量 | 神经网络层的参数 |
| $\mathbf{x}$ | 输入向量 | 网络的输入 |
| $\mathbf{h}$ | 隐藏层激活 | 中间层的输出 |
| $\mathbf{y}$ | 输出向量 | 网络的输出 |
| $\nabla_\theta \mathcal{L}$ | 损失函数关于参数的梯度 | 反向传播计算 |
| $H$ 或 $\nabla^2 \mathcal{L}$ | Hessian矩阵 | 二阶导数矩阵 |
| $J$ | Jacobian矩阵 | 一阶导数矩阵 |
| $\odot$ | 逐元素乘法（Hadamard积） | $[A \odot B]_{ij} = a_{ij} b_{ij}$ |
| $\otimes$ | 克罗内克积（Kronecker积） | 矩阵的张量积 |
