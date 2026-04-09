# 第14章：线性变换的矩阵表示

> 矩阵不仅仅是数字的表格——它是线性变换的"身份证"。选定基之后，抽象的线性变换与具体的矩阵之间存在完美的一一对应。这一章揭示两者之间深刻的等价关系。

---

## 学习目标

完成本章学习后，你将能够：

- 理解坐标向量的概念：在给定基下将抽象向量"编码"为数字列向量
- 掌握线性变换的矩阵表示的构造方法：只需知道基向量的像
- 理解线性变换空间与矩阵空间之间的同构关系
- 计算复合变换的矩阵：复合对应矩阵乘法
- 判断线性变换的可逆性，并利用逆矩阵求逆变换

---

## 14.1 坐标向量

### 14.1.1 基下的坐标

设 $V$ 是 $n$ 维向量空间，$\mathcal{B} = \{b_1, b_2, \ldots, b_n\}$ 是 $V$ 的一个**有序基**（ordered basis，向量的排列顺序固定）。

由基的定义，$V$ 中每个向量 $v$ 都能唯一地写成：

$$v = c_1 b_1 + c_2 b_2 + \cdots + c_n b_n$$

**定义（坐标向量）：** 标量 $c_1, c_2, \ldots, c_n$ 称为 $v$ 在基 $\mathcal{B}$ 下的**坐标**，列向量

$$[v]_{\mathcal{B}} = \begin{pmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \end{pmatrix} \in \mathbb{R}^n$$

称为 $v$ 在基 $\mathcal{B}$ 下的**坐标向量**（coordinate vector）。

**直觉：** 坐标向量就像一个"翻译"——它把空间 $V$ 中的抽象向量翻译为 $\mathbb{R}^n$ 中可以计算的列向量。不同的基给出不同的翻译方案，但每种方案都是完整且无歧义的。

### 14.1.2 例子：$\mathbb{R}^2$ 中的非标准基

设 $\mathbb{R}^2$ 有两个基：

$$\mathcal{E} = \left\{e_1 = \begin{pmatrix}1\\0\end{pmatrix},\; e_2 = \begin{pmatrix}0\\1\end{pmatrix}\right\} \quad \text{（标准基）}$$

$$\mathcal{B} = \left\{b_1 = \begin{pmatrix}1\\1\end{pmatrix},\; b_2 = \begin{pmatrix}1\\-1\end{pmatrix}\right\} \quad \text{（非标准基）}$$

取向量 $v = \begin{pmatrix}3\\1\end{pmatrix}$。

**在标准基下：** 显然 $[v]_{\mathcal{E}} = \begin{pmatrix}3\\1\end{pmatrix}$。

**在基 $\mathcal{B}$ 下：** 需要解方程 $c_1 b_1 + c_2 b_2 = v$：

$$c_1 \begin{pmatrix}1\\1\end{pmatrix} + c_2 \begin{pmatrix}1\\-1\end{pmatrix} = \begin{pmatrix}3\\1\end{pmatrix}$$

解得 $c_1 = 2,\; c_2 = 1$，故 $[v]_{\mathcal{B}} = \begin{pmatrix}2\\1\end{pmatrix}$。

验证：$2 \cdot \begin{pmatrix}1\\1\end{pmatrix} + 1 \cdot \begin{pmatrix}1\\-1\end{pmatrix} = \begin{pmatrix}3\\1\end{pmatrix}$。✓

### 14.1.3 坐标映射是同构

**命题：** 坐标映射 $\phi_{\mathcal{B}}: V \to \mathbb{R}^n$，$v \mapsto [v]_{\mathcal{B}}$ 是线性同构，即：

$$[u + v]_{\mathcal{B}} = [u]_{\mathcal{B}} + [v]_{\mathcal{B}}, \qquad [cv]_{\mathcal{B}} = c[v]_{\mathcal{B}}$$

这意味着：在基 $\mathcal{B}$ 下，$V$ 中的线性运算与 $\mathbb{R}^n$ 中的向量运算**完全一致**。$n$ 维向量空间 $V$ 在结构上与 $\mathbb{R}^n$ 没有区别。

---

## 14.2 线性变换的矩阵

### 14.2.1 核心思想

设 $T: V \to W$ 是线性变换，$\dim V = n$，$\dim W = m$。固定：

- $V$ 的有序基 $\mathcal{B} = \{b_1, \ldots, b_n\}$
- $W$ 的有序基 $\mathcal{C} = \{c_1, \ldots, c_m\}$

**关键观察：** 由于 $T$ 是线性的，只要知道 $T$ 对每个基向量 $b_j$ 的作用，就完全确定了 $T$。对任意 $v = \sum_{j=1}^n x_j b_j$，

$$T(v) = \sum_{j=1}^n x_j T(b_j)$$

因此，我们只需记录 $T(b_1), T(b_2), \ldots, T(b_n)$ 在基 $\mathcal{C}$ 下的坐标。

### 14.2.2 矩阵表示的构造

每个 $T(b_j) \in W$ 在基 $\mathcal{C}$ 下有唯一的坐标向量，设

$$[T(b_j)]_{\mathcal{C}} = \begin{pmatrix} a_{1j} \\ a_{2j} \\ \vdots \\ a_{mj} \end{pmatrix}$$

**定义（变换矩阵）：** $T$ 关于基 $\mathcal{B}$（定义域）和基 $\mathcal{C}$（值域）的**矩阵表示**（matrix representation）是 $m \times n$ 矩阵：

$$[T]_{\mathcal{B}}^{\mathcal{C}} = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}$$

即：**第 $j$ 列是 $T(b_j)$ 在基 $\mathcal{C}$ 下的坐标向量**。

**核心公式：** 对任意 $v \in V$，

$$[T(v)]_{\mathcal{C}} = [T]_{\mathcal{B}}^{\mathcal{C}} \cdot [v]_{\mathcal{B}}$$

这就是矩阵的精髓：**矩阵-向量乘法 = 线性变换在坐标下的计算**。

### 14.2.3 完整例子：旋转变换

设 $T: \mathbb{R}^2 \to \mathbb{R}^2$ 是逆时针旋转 $\theta$ 角的变换：

$$T\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}x\cos\theta - y\sin\theta \\ x\sin\theta + y\cos\theta\end{pmatrix}$$

取标准基 $\mathcal{E} = \{e_1, e_2\}$，计算基向量的像：

$$T(e_1) = T\begin{pmatrix}1\\0\end{pmatrix} = \begin{pmatrix}\cos\theta\\\sin\theta\end{pmatrix}, \qquad T(e_2) = T\begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}-\sin\theta\\\cos\theta\end{pmatrix}$$

将这两个像写成矩阵的两列，得旋转矩阵：

$$[T]_{\mathcal{E}}^{\mathcal{E}} = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}$$

**验证：** $[T]_{\mathcal{E}}^{\mathcal{E}} \begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}x\cos\theta - y\sin\theta \\ x\sin\theta + y\cos\theta\end{pmatrix}$。✓

### 14.2.4 例子：微分变换

设 $V = P_3$（次数 $\leq 3$ 的实系数多项式空间），$T = \frac{d}{dx}$ 是微分算子。

取标准基 $\mathcal{B} = \{1, x, x^2, x^3\}$，$W = P_2$，基 $\mathcal{C} = \{1, x, x^2\}$。

计算基向量的像：

| $b_j$ | $T(b_j) = b_j'$ | $[T(b_j)]_{\mathcal{C}}$ |
|:---:|:---:|:---:|
| $1$ | $0$ | $(0,0,0)^T$ |
| $x$ | $1$ | $(1,0,0)^T$ |
| $x^2$ | $2x$ | $(0,2,0)^T$ |
| $x^3$ | $3x^2$ | $(0,0,3)^T$ |

因此微分算子的矩阵为：

$$[T]_{\mathcal{B}}^{\mathcal{C}} = \begin{pmatrix}0 & 1 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 & 0 & 3\end{pmatrix}$$

验证：对 $p(x) = 2 + 3x - x^2 + 4x^3$，坐标向量 $[p]_{\mathcal{B}} = (2,3,-1,4)^T$，

$$[T]_{\mathcal{B}}^{\mathcal{C}} \begin{pmatrix}2\\3\\-1\\4\end{pmatrix} = \begin{pmatrix}3\\-2\\12\end{pmatrix}$$

对应多项式 $3 - 2x + 12x^2$，确实等于 $p'(x) = 3 - 2x + 12x^2$。✓

---

## 14.3 矩阵与线性变换的对应

### 14.3.1 同构定理

固定 $V$ 的基 $\mathcal{B}$（$n$ 维）和 $W$ 的基 $\mathcal{C}$（$m$ 维），记

$$\mathcal{L}(V, W) = \{T: V \to W \mid T \text{ 是线性变换}\}$$

**定理（同构）：** 映射 $\Phi: \mathcal{L}(V, W) \to \mathbb{R}^{m \times n}$，$T \mapsto [T]_{\mathcal{B}}^{\mathcal{C}}$ 是线性空间的同构，即：

$$[S + T]_{\mathcal{B}}^{\mathcal{C}} = [S]_{\mathcal{B}}^{\mathcal{C}} + [T]_{\mathcal{B}}^{\mathcal{C}}, \qquad [cT]_{\mathcal{B}}^{\mathcal{C}} = c[T]_{\mathcal{B}}^{\mathcal{C}}$$

**推论：** $\dim \mathcal{L}(V, W) = mn$。线性变换的结构完全由矩阵捕获。

### 14.3.2 基的选择对矩阵的影响

**同一个线性变换，不同的基给出不同的矩阵。** 设 $T: V \to V$ 是自变换（$V=W$），若将基从 $\mathcal{B}$ 换为 $\mathcal{B}'$，设过渡矩阵为 $P = [\text{id}]_{\mathcal{B}'}^{\mathcal{B}}$（第 $j$ 列是 $b_j'$ 在 $\mathcal{B}$ 下的坐标），则

$$[T]_{\mathcal{B}'} = P^{-1} [T]_{\mathcal{B}} P$$

这就是**相似变换**：同一线性变换在不同基下的矩阵互相相似。这一关系是第15章（特征值分解）的基础。

### 14.3.3 几何直觉

```
抽象空间 V          坐标空间 ℝⁿ
    v  ─────[·]_B────▶  [v]_B
    │                      │
    T                 [T]_B^C  (矩阵乘法)
    │                      │
    ▼                      ▼
   T(v) ─────[·]_C────▶  [T(v)]_C
```

上图说明：在坐标化之后，抽象的线性变换 $T$ 被矩阵乘法完全替代。选定基 = 选定坐标系，坐标系不同则矩阵不同，但描述的几何对象（变换本身）完全相同。

---

## 14.4 复合变换的矩阵表示

### 14.4.1 主定理

设 $S: U \to V$，$T: V \to W$ 是线性变换，基分别为 $\mathcal{A}$（$U$）、$\mathcal{B}$（$V$）、$\mathcal{C}$（$W$），则复合变换 $T \circ S: U \to W$ 满足：

$$[T \circ S]_{\mathcal{A}}^{\mathcal{C}} = [T]_{\mathcal{B}}^{\mathcal{C}} \cdot [S]_{\mathcal{A}}^{\mathcal{B}}$$

**结论：复合变换的矩阵 = 各变换矩阵的乘积（注意顺序：先作用的在右边）。**

**证明思路：** 对任意 $u \in U$，

$$[T(S(u))]_{\mathcal{C}} = [T]_{\mathcal{B}}^{\mathcal{C}} \cdot [S(u)]_{\mathcal{B}} = [T]_{\mathcal{B}}^{\mathcal{C}} \cdot [S]_{\mathcal{A}}^{\mathcal{B}} \cdot [u]_{\mathcal{A}}$$

故 $[T \circ S]_{\mathcal{A}}^{\mathcal{C}} = [T]_{\mathcal{B}}^{\mathcal{C}} \cdot [S]_{\mathcal{A}}^{\mathcal{B}}$。$\square$

### 14.4.2 矩阵乘法的几何意义

这个定理揭示了矩阵乘法的**几何本质**：矩阵乘法不是凑巧定义的行列计算规则，而是复合线性变换的自然产物。

**例：** 先旋转 $\alpha$ 角，再旋转 $\beta$ 角，等于旋转 $\alpha + \beta$ 角：

$$\begin{pmatrix}\cos\beta & -\sin\beta \\ \sin\beta & \cos\beta\end{pmatrix} \begin{pmatrix}\cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha\end{pmatrix} = \begin{pmatrix}\cos(\alpha+\beta) & -\sin(\alpha+\beta) \\ \sin(\alpha+\beta) & \cos(\alpha+\beta)\end{pmatrix}$$

展开左边即得三角恒等式 $\cos(\alpha+\beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$。线性代数给了三角学一个优雅的证明。

### 14.4.3 恒等变换的矩阵

恒等变换 $\text{id}: V \to V$，$v \mapsto v$ 在任意有序基 $\mathcal{B}$ 下的矩阵是单位矩阵 $I_n$。

---

## 14.5 可逆变换与可逆矩阵

### 14.5.1 等价关系

**定理：** 设 $T: V \to V$ 是线性变换，$n = \dim V$，$\mathcal{B}$ 是 $V$ 的有序基，则：

$$T \text{ 是同构（可逆线性变换）} \iff [T]_{\mathcal{B}} \text{ 是可逆矩阵}$$

且此时 $[T^{-1}]_{\mathcal{B}} = ([T]_{\mathcal{B}})^{-1}$。

**证明（$\Rightarrow$）：** 若 $T$ 可逆，则 $T \circ T^{-1} = \text{id}$，由复合定理：

$$[T]_{\mathcal{B}} \cdot [T^{-1}]_{\mathcal{B}} = [\text{id}]_{\mathcal{B}} = I_n$$

故 $[T]_{\mathcal{B}}$ 可逆，逆矩阵为 $[T^{-1}]_{\mathcal{B}}$。

**证明（$\Leftarrow$）：** 若 $[T]_{\mathcal{B}}$ 可逆，定义线性变换 $S$，使得 $[S]_{\mathcal{B}} = ([T]_{\mathcal{B}})^{-1}$，则

$$[T \circ S]_{\mathcal{B}} = [T]_{\mathcal{B}} \cdot [S]_{\mathcal{B}} = I_n = [\text{id}]_{\mathcal{B}}$$

故 $T \circ S = \text{id}$，即 $S = T^{-1}$，$T$ 可逆。$\square$

### 14.5.2 可逆性的刻画

结合前几章的结论，对线性算子 $T: V \to V$（$\dim V = n$）和矩阵 $A = [T]_{\mathcal{B}}$，以下命题等价：

| 线性变换语言 | 矩阵语言 |
|:---:|:---:|
| $T$ 是单射 | $\ker(T) = \{0\}$，即 $\text{nullity}(T) = 0$ |
| $T$ 是满射 | $\text{im}(T) = V$，即 $\text{rank}(T) = n$ |
| $T$ 是双射（同构） | $A$ 可逆，$\det(A) \neq 0$ |

### 14.5.3 例子：反射变换

$\mathbb{R}^2$ 中关于 $x$ 轴的反射：$T\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}x\\-y\end{pmatrix}$。

矩阵：$A = \begin{pmatrix}1 & 0 \\ 0 & -1\end{pmatrix}$，$\det(A) = -1 \neq 0$，故可逆。

逆变换：$T^{-1} = T$（反射自身即逆），矩阵 $A^{-1} = A$（$A$ 是对合矩阵）。

验证：$A^2 = \begin{pmatrix}1&0\\0&-1\end{pmatrix}^2 = \begin{pmatrix}1&0\\0&1\end{pmatrix} = I$。✓

---

## 本章小结

- **坐标向量** $[v]_{\mathcal{B}}$ 将抽象向量用基 $\mathcal{B}$ 的线性组合系数编码为 $\mathbb{R}^n$ 中的列向量；坐标映射 $v \mapsto [v]_{\mathcal{B}}$ 是线性同构。

- **线性变换的矩阵** $[T]_{\mathcal{B}}^{\mathcal{C}}$ 由"将每个基向量的像坐标化，依次排列为列"构造而成；核心公式 $[T(v)]_{\mathcal{C}} = [T]_{\mathcal{B}}^{\mathcal{C}} [v]_{\mathcal{B}}$ 将抽象变换化为矩阵乘法。

- **同构定理**：$\mathcal{L}(V,W) \cong \mathbb{R}^{m \times n}$，线性变换与矩阵之间存在保运算的一一对应；基不同，矩阵不同，但变换本质相同（相似矩阵描述同一变换）。

- **复合对应乘积**：$[T \circ S]_{\mathcal{A}}^{\mathcal{C}} = [T]_{\mathcal{B}}^{\mathcal{C}} \cdot [S]_{\mathcal{A}}^{\mathcal{B}}$；矩阵乘法的几何含义是线性变换的复合。

- **可逆对应可逆**：$T$ 是同构 $\iff$ $[T]_{\mathcal{B}}$ 是可逆矩阵，且 $[T^{-1}]_{\mathcal{B}} = ([T]_{\mathcal{B}})^{-1}$。

---

## 深度学习应用：卷积的矩阵形式

### 背景

卷积神经网络（CNN）是深度学习的核心组件。从线性代数角度看，**卷积运算本质上是一种线性变换**，因此可以表示为矩阵乘法。理解这一点有助于深入理解 CNN 的计算效率、梯度传播和硬件加速原理。

### 14.6.1 一维卷积与 Toeplitz 矩阵

设输入信号 $x = (x_0, x_1, x_2, x_3, x_4)^T \in \mathbb{R}^5$，卷积核 $k = (k_0, k_1, k_2)^T \in \mathbb{R}^3$（步长为1，无填充）。

离散卷积定义为：

$$y_i = \sum_{j=0}^{2} k_j \cdot x_{i+j}, \quad i = 0, 1, 2$$

输出 $y = (y_0, y_1, y_2)^T \in \mathbb{R}^3$。写成矩阵形式：

$$y = \begin{pmatrix} y_0 \\ y_1 \\ y_2 \end{pmatrix} = \begin{pmatrix} k_0 & k_1 & k_2 & 0 & 0 \\ 0 & k_0 & k_1 & k_2 & 0 \\ 0 & 0 & k_0 & k_1 & k_2 \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \\ x_4 \end{pmatrix} = K x$$

矩阵 $K$ 就是**Toeplitz 矩阵**（沿对角线元素相同的矩阵），每一行是卷积核沿输入滑动一步的结果。

**关键性质：**
- $K$ 的每一行只是上一行向右移动一格（平移等变性的矩阵体现）
- 卷积对应的线性变换维度：$\mathbb{R}^5 \to \mathbb{R}^3$，矩阵为 $3 \times 5$
- 参数量：卷积核 3 个参数，矩阵 15 个元素，但大量元素共享——这是**权重共享**（weight sharing）的本质

### 14.6.2 二维卷积与 im2col

在图像处理中，卷积是二维的：输入为 $H \times W$ 图像，卷积核为 $k \times k$，输出为 $(H-k+1) \times (W-k+1)$ 的特征图。

直接表示为矩阵较为复杂，实践中使用 **im2col**（image to column）技术：

**im2col 的思想：**

1. 对输出的每个位置，提取与卷积核对齐的输入局部块（$k \times k$ 个元素）
2. 将该局部块**展平为列向量**
3. 所有位置的列向量拼成矩阵 $X_{\text{col}}$（形状：$(k^2) \times (H_{\text{out}} \cdot W_{\text{out}})$）
4. 卷积核展平为行向量 $K_{\text{row}}$（形状：$1 \times k^2$）
5. 卷积结果 = $K_{\text{row}} \cdot X_{\text{col}}$（矩阵乘法）

对于多通道（$C$ 输入通道，$F$ 个卷积核）的情形：

$$Y = W_{\text{col}} \cdot X_{\text{col}}$$

其中 $W_{\text{col}}$ 的形状为 $F \times (C \cdot k^2)$，$X_{\text{col}}$ 的形状为 $(C \cdot k^2) \times (H_{\text{out}} \cdot W_{\text{out}})$，输出 $Y$ 的形状为 $F \times (H_{\text{out}} \cdot W_{\text{out}})$。

这样，原本复杂的二维卷积变成了一次**矩阵乘法**，可以直接调用高度优化的 BLAS（基本线性代数子程序）库，在 GPU 上获得极高的并行效率。

### 14.6.3 PyTorch 代码示例

```python
import torch
import torch.nn.functional as F

# ── 1. 用 PyTorch 的卷积计算结果 ──────────────────────────────
torch.manual_seed(42)
x = torch.randn(1, 1, 5)          # batch=1, channels=1, length=5
kernel = torch.randn(1, 1, 3)     # out_channels=1, in_channels=1, kernel_size=3

y_conv = F.conv1d(x, kernel)      # 标准卷积，shape: (1, 1, 3)

# ── 2. 手动构造 Toeplitz 矩阵，验证等价性 ────────────────────
k = kernel.squeeze()               # shape: (3,)
x_vec = x.squeeze()               # shape: (5,)

# 构造 3×5 的 Toeplitz 矩阵
K = torch.zeros(3, 5)
for i in range(3):
    K[i, i:i+3] = k

y_matrix = K @ x_vec              # 矩阵乘法，shape: (3,)

# ── 3. 对比结果 ──────────────────────────────────────────────
print("conv1d 结果:  ", y_conv.squeeze().detach().numpy().round(4))
print("矩阵乘法结果:", y_matrix.detach().numpy().round(4))
print("最大误差:    ", (y_conv.squeeze() - y_matrix).abs().max().item())
# 输出：最大误差接近机器精度（约 1e-7），两种方法完全等价

# ── 4. im2col 示意（2D 卷积）─────────────────────────────────
x2d = torch.randn(1, 3, 6, 6)     # batch=1, C=3, H=6, W=6
kernel2d = torch.randn(8, 3, 3, 3) # F=8 个卷积核，C=3，k=3

# PyTorch unfold 实现 im2col
# 输出形状: (batch, C*k*k, H_out*W_out)
x_col = torch.nn.functional.unfold(x2d, kernel_size=3)
# x_col.shape: (1, 27, 16)  [27 = 3*3*3, 16 = 4*4]

w_col = kernel2d.view(8, -1)      # (8, 27)

# 矩阵乘法替代卷积
y_col = w_col @ x_col.squeeze(0)  # (8, 16)

# 等价的标准卷积
y_ref = F.conv2d(x2d, kernel2d)   # (1, 8, 4, 4)

print("\nim2col 误差:", (y_col - y_ref.squeeze().view(8, -1)).abs().max().item())
# 输出：误差接近机器精度
```

**运行结果解读：**
- 一维卷积与 Toeplitz 矩阵乘法的结果完全一致，误差仅为浮点舍入误差（约 $10^{-7}$）
- 二维 im2col + 矩阵乘法与 `F.conv2d` 结果完全一致
- 实际 cuDNN 内部也采用类似策略（或 FFT 方法），将卷积转化为高效的矩阵运算

### 14.6.4 为什么这很重要

| 视角 | 卷积的本质 |
|:---|:---|
| 线性代数 | 具有平移等变性约束的线性变换（Toeplitz 矩阵） |
| 参数效率 | 权重共享 = 矩阵中大量元素共享同一值，大幅减少参数 |
| 计算效率 | im2col 将卷积转为 GEMM（通用矩阵乘），可利用 BLAS/cuBLAS 加速 |
| 反向传播 | 梯度计算 = 转置矩阵的乘法，对应转置卷积（transposed convolution） |

---

## 练习题

**练习 14.1（基础）** 设 $\mathbb{R}^2$ 的有序基为 $\mathcal{B} = \left\{b_1 = \begin{pmatrix}2\\1\end{pmatrix}, b_2 = \begin{pmatrix}1\\3\end{pmatrix}\right\}$，求向量 $v = \begin{pmatrix}5\\7\end{pmatrix}$ 在基 $\mathcal{B}$ 下的坐标向量 $[v]_{\mathcal{B}}$。

---

**练习 14.2（中等）** 设 $T: \mathbb{R}^2 \to \mathbb{R}^2$ 是关于直线 $y = x$ 的反射变换，即 $T\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}y\\x\end{pmatrix}$。求 $T$ 关于标准基 $\mathcal{E}$ 的矩阵 $[T]_{\mathcal{E}}$，并验证 $T^2 = \text{id}$（即 $([T]_{\mathcal{E}})^2 = I$）。

---

**练习 14.3（中等）** 设 $P_2$ 是次数不超过 2 的多项式空间，有序基为 $\mathcal{B} = \{1, x, x^2\}$。定义线性变换 $T: P_2 \to P_2$，$T(p) = p(x+1)$（将 $p(x)$ 中的 $x$ 替换为 $x+1$）。求 $[T]_{\mathcal{B}}$。

---

**练习 14.4（较难）** 设 $S, T: \mathbb{R}^3 \to \mathbb{R}^3$ 由矩阵给出：

$$A = [S]_{\mathcal{E}} = \begin{pmatrix}1&0&1\\0&1&0\\1&0&1\end{pmatrix}, \quad B = [T]_{\mathcal{E}} = \begin{pmatrix}2&0&0\\0&1&0\\0&0&3\end{pmatrix}$$

(a) 求 $[T \circ S]_{\mathcal{E}}$。
(b) 求 $[S \circ T]_{\mathcal{E}}$。
(c) 比较 (a) 和 (b)，说明复合变换一般不满足交换律。
(d) 判断 $S$ 是否可逆。

---

**练习 14.5（挑战）** 设 $V = \mathbb{R}^2$，有两个有序基：

$$\mathcal{E} = \left\{\begin{pmatrix}1\\0\end{pmatrix}, \begin{pmatrix}0\\1\end{pmatrix}\right\}, \quad \mathcal{B} = \left\{\begin{pmatrix}1\\1\end{pmatrix}, \begin{pmatrix}1\\-1\end{pmatrix}\right\}$$

设 $T: V \to V$ 是逆时针旋转 $90°$ 的变换，即 $[T]_{\mathcal{E}} = \begin{pmatrix}0&-1\\1&0\end{pmatrix}$。

(a) 求过渡矩阵 $P = [\text{id}]_{\mathcal{E}}^{\mathcal{B}}$（第 $j$ 列是 $b_j$ 在 $\mathcal{E}$ 下的坐标）。
(b) 求 $[T]_{\mathcal{B}} = P^{-1}[T]_{\mathcal{E}} P$。
(c) 直接计算 $T(b_1)$ 和 $T(b_2)$ 在基 $\mathcal{B}$ 下的坐标，验证 (b) 的结果。

---

## 练习答案

<details>
<summary>练习 14.1 答案</summary>

需要解方程 $c_1 b_1 + c_2 b_2 = v$：

$$c_1 \begin{pmatrix}2\\1\end{pmatrix} + c_2 \begin{pmatrix}1\\3\end{pmatrix} = \begin{pmatrix}5\\7\end{pmatrix}$$

即线性方程组：

$$\begin{cases} 2c_1 + c_2 = 5 \\ c_1 + 3c_2 = 7 \end{cases}$$

用增广矩阵做行化简：

$$\begin{pmatrix}2&1&5\\1&3&7\end{pmatrix} \xrightarrow{R_1 \leftrightarrow R_2} \begin{pmatrix}1&3&7\\2&1&5\end{pmatrix} \xrightarrow{R_2 - 2R_1} \begin{pmatrix}1&3&7\\0&-5&-9\end{pmatrix} \xrightarrow{R_2 / (-5)} \begin{pmatrix}1&3&7\\0&1&\frac{9}{5}\end{pmatrix}$$

$$\xrightarrow{R_1 - 3R_2} \begin{pmatrix}1&0&\frac{8}{5}\\0&1&\frac{9}{5}\end{pmatrix}$$

故 $c_1 = \dfrac{8}{5}$，$c_2 = \dfrac{9}{5}$，即 $[v]_{\mathcal{B}} = \begin{pmatrix}8/5\\9/5\end{pmatrix}$。

**验证：** $\dfrac{8}{5}\begin{pmatrix}2\\1\end{pmatrix} + \dfrac{9}{5}\begin{pmatrix}1\\3\end{pmatrix} = \begin{pmatrix}16/5 + 9/5\\8/5 + 27/5\end{pmatrix} = \begin{pmatrix}5\\7\end{pmatrix}$。✓

</details>

---

<details>
<summary>练习 14.2 答案</summary>

计算标准基向量的像：

$$T(e_1) = T\begin{pmatrix}1\\0\end{pmatrix} = \begin{pmatrix}0\\1\end{pmatrix}, \quad T(e_2) = T\begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}1\\0\end{pmatrix}$$

以像在标准基下的坐标作为矩阵的列：

$$[T]_{\mathcal{E}} = \begin{pmatrix}0&1\\1&0\end{pmatrix}$$

**验证 $T^2 = \text{id}$：**

$$([T]_{\mathcal{E}})^2 = \begin{pmatrix}0&1\\1&0\end{pmatrix}\begin{pmatrix}0&1\\1&0\end{pmatrix} = \begin{pmatrix}0 \cdot 0 + 1 \cdot 1 & 0 \cdot 1 + 1 \cdot 0\\ 1 \cdot 0 + 0 \cdot 1 & 1 \cdot 1 + 0 \cdot 0\end{pmatrix} = \begin{pmatrix}1&0\\0&1\end{pmatrix} = I$$

✓ 反射两次回到原位，符合几何直觉。

</details>

---

<details>
<summary>练习 14.3 答案</summary>

计算基向量在 $T$ 下的像：

- $T(1) = 1$ （常数多项式，代入 $x+1$ 后仍为 $1$）
- $T(x) = x + 1 = 1 \cdot 1 + 1 \cdot x + 0 \cdot x^2$
- $T(x^2) = (x+1)^2 = 1 + 2x + x^2 = 1 \cdot 1 + 2 \cdot x + 1 \cdot x^2$

在基 $\mathcal{B} = \{1, x, x^2\}$ 下的坐标向量分别为：

$$[T(1)]_{\mathcal{B}} = \begin{pmatrix}1\\0\\0\end{pmatrix}, \quad [T(x)]_{\mathcal{B}} = \begin{pmatrix}1\\1\\0\end{pmatrix}, \quad [T(x^2)]_{\mathcal{B}} = \begin{pmatrix}1\\2\\1\end{pmatrix}$$

故变换矩阵（各坐标向量作为列）：

$$[T]_{\mathcal{B}} = \begin{pmatrix}1&1&1\\0&1&2\\0&0&1\end{pmatrix}$$

**验证：** 取 $p(x) = 3 + x - 2x^2$，$[p]_{\mathcal{B}} = (3,1,-2)^T$。

$$[T]_{\mathcal{B}} \begin{pmatrix}3\\1\\-2\end{pmatrix} = \begin{pmatrix}1\cdot3+1\cdot1+1\cdot(-2)\\0\cdot3+1\cdot1+2\cdot(-2)\\0+0+1\cdot(-2)\end{pmatrix} = \begin{pmatrix}2\\-3\\-2\end{pmatrix}$$

对应多项式 $2 - 3x - 2x^2$。直接计算：$T(p) = p(x+1) = 3 + (x+1) - 2(x+1)^2 = 3 + x + 1 - 2(x^2 + 2x + 1) = 2 - 3x - 2x^2$。✓

</details>

---

<details>
<summary>练习 14.4 答案</summary>

**(a) $[T \circ S]_{\mathcal{E}} = B \cdot A$（先作用 $S$，再作用 $T$，右边是先用的）：**

$$BA = \begin{pmatrix}2&0&0\\0&1&0\\0&0&3\end{pmatrix}\begin{pmatrix}1&0&1\\0&1&0\\1&0&1\end{pmatrix} = \begin{pmatrix}2&0&2\\0&1&0\\3&0&3\end{pmatrix}$$

**(b) $[S \circ T]_{\mathcal{E}} = A \cdot B$：**

$$AB = \begin{pmatrix}1&0&1\\0&1&0\\1&0&1\end{pmatrix}\begin{pmatrix}2&0&0\\0&1&0\\0&0&3\end{pmatrix} = \begin{pmatrix}2&0&3\\0&1&0\\2&0&3\end{pmatrix}$$

**(c) 比较：**

$$BA = \begin{pmatrix}2&0&2\\0&1&0\\3&0&3\end{pmatrix} \neq \begin{pmatrix}2&0&3\\0&1&0\\2&0&3\end{pmatrix} = AB$$

两者不等，说明复合变换一般不满足交换律（$T \circ S \neq S \circ T$）。

**(d) 判断 $S$ 的可逆性：**

$$\det(A) = \det\begin{pmatrix}1&0&1\\0&1&0\\1&0&1\end{pmatrix}$$

按第二行展开：$\det(A) = 1 \cdot \det\begin{pmatrix}0&1\\1&1\end{pmatrix} \cdot (-1)^{2+2} = 1 \cdot (0 - 1) = -1$...

让我重新计算（按第一行展开）：

$$\det(A) = 1 \cdot \det\begin{pmatrix}1&0\\0&1\end{pmatrix} - 0 + 1 \cdot \det\begin{pmatrix}0&1\\1&0\end{pmatrix} = 1 \cdot 1 + 1 \cdot (0-1) = 1 - 1 = 0$$

$\det(A) = 0$，故 $S$ **不可逆**。

几何上，$A$ 的第一列和第三列相同，即 $v_1 = v_3$，矩阵的列向量线性相关，$S$ 不是单射（将两个不同向量映射到同一像）。

</details>

---

<details>
<summary>练习 14.5 答案</summary>

**(a) 过渡矩阵 $P$：**

$P$ 的第 $j$ 列是 $b_j$ 在基 $\mathcal{E}$ 下的坐标（即 $b_j$ 本身，因为 $\mathcal{E}$ 是标准基）：

$$P = \begin{pmatrix}1&1\\1&-1\end{pmatrix}$$

**(b) 计算 $[T]_{\mathcal{B}} = P^{-1}[T]_{\mathcal{E}} P$：**

先求 $P^{-1}$：$\det(P) = -1 - 1 = -2$，故

$$P^{-1} = \frac{1}{-2}\begin{pmatrix}-1&-1\\-1&1\end{pmatrix} = \begin{pmatrix}1/2&1/2\\1/2&-1/2\end{pmatrix}$$

计算 $[T]_{\mathcal{E}} P$：

$$\begin{pmatrix}0&-1\\1&0\end{pmatrix}\begin{pmatrix}1&1\\1&-1\end{pmatrix} = \begin{pmatrix}-1&1\\1&1\end{pmatrix}$$

再乘以 $P^{-1}$：

$$[T]_{\mathcal{B}} = P^{-1}[T]_{\mathcal{E}} P = \begin{pmatrix}1/2&1/2\\1/2&-1/2\end{pmatrix}\begin{pmatrix}-1&1\\1&1\end{pmatrix} = \begin{pmatrix}0&1\\-1&0\end{pmatrix}$$

**(c) 直接验证：**

$$T(b_1) = \begin{pmatrix}0&-1\\1&0\end{pmatrix}\begin{pmatrix}1\\1\end{pmatrix} = \begin{pmatrix}-1\\1\end{pmatrix}$$

求 $\begin{pmatrix}-1\\1\end{pmatrix}$ 在基 $\mathcal{B}$ 下的坐标：$c_1\begin{pmatrix}1\\1\end{pmatrix} + c_2\begin{pmatrix}1\\-1\end{pmatrix} = \begin{pmatrix}-1\\1\end{pmatrix}$，解得 $c_1 = 0, c_2 = -1$，即 $[T(b_1)]_{\mathcal{B}} = \begin{pmatrix}0\\-1\end{pmatrix}$。

$$T(b_2) = \begin{pmatrix}0&-1\\1&0\end{pmatrix}\begin{pmatrix}1\\-1\end{pmatrix} = \begin{pmatrix}1\\1\end{pmatrix}$$

求 $\begin{pmatrix}1\\1\end{pmatrix}$ 在基 $\mathcal{B}$ 下的坐标：$c_1 = 1, c_2 = 0$，即 $[T(b_2)]_{\mathcal{B}} = \begin{pmatrix}1\\0\end{pmatrix}$。

以两个坐标向量为列：

$$[T]_{\mathcal{B}} = \begin{pmatrix}0&1\\-1&0\end{pmatrix}$$

与 (b) 的结果一致。✓

**几何含义：** 旋转 $90°$ 在非标准基 $\mathcal{B}$ 下的矩阵仍是旋转矩阵（反对称结构 $\begin{pmatrix}0&1\\-1&0\end{pmatrix}$），体现了旋转变换的内在几何结构与坐标选取无关。

</details>
