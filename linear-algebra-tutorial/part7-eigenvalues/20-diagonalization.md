# 第20章：矩阵对角化

> **前置知识**：第19章（特征值与特征向量）、第11章（基与维数）、第15章（基变换）
>
> **本章难度**：★★★★☆
>
> **预计学习时间**：4-5 小时

---

## 学习目标

学完本章后，你将能够：

- 理解对角化的含义：将矩阵 $A$ 分解为 $A = PDP^{-1}$，并判断矩阵是否可对角化
- 掌握对角化的完整算法：求特征值、求特征向量、构造矩阵 $P$ 和 $D$
- 利用 $A^n = PD^nP^{-1}$ 高效计算矩阵的幂，避免逐步矩阵乘法
- 识别不可对角化（亏损）矩阵，了解 Jordan 标准形的基本思想
- 将对角化技术应用于微分方程组、Markov 链和 Fibonacci 数列等实际问题

---

## 20.1 对角化的条件

### 核心问题：能否选一组好的基？

在第15章中，我们学到：同一个线性变换在不同基下有不同的矩阵表示。**对角化的核心思想**是：能否找到一组特殊的基，使得矩阵在这组基下的表示是对角矩阵？

对角矩阵是最"简单"的矩阵——它只沿各坐标轴方向缩放，不产生任何旋转或剪切。如果我们能把一般矩阵"变成"对角矩阵，很多计算将大幅简化。

### 对角化的定义

**定义**：若存在可逆矩阵 $P$ 和对角矩阵 $D$，使得

$$\boxed{A = PDP^{-1}}$$

则称矩阵 $A \in \mathbb{R}^{n \times n}$ **可对角化（diagonalizable）**。

等价地，$A$ 可对角化当且仅当 $P^{-1}AP = D$，即 $A$ 与对角矩阵**相似**。

### 特征向量的关键角色

$P$ 和 $D$ 究竟是什么？让我们从特征方程出发推导。

设 $A$ 有特征值 $\lambda_1, \ldots, \lambda_n$（可重复），对应特征向量 $\mathbf{p}_1, \ldots, \mathbf{p}_n$，即 $A\mathbf{p}_i = \lambda_i \mathbf{p}_i$。将 $n$ 个方程合并成矩阵形式：

$$A \underbrace{\begin{pmatrix} | & | & & | \\ \mathbf{p}_1 & \mathbf{p}_2 & \cdots & \mathbf{p}_n \\ | & | & & | \end{pmatrix}}_{P} = \underbrace{\begin{pmatrix} | & | & & | \\ \mathbf{p}_1 & \mathbf{p}_2 & \cdots & \mathbf{p}_n \\ | & | & & | \end{pmatrix}}_{P} \underbrace{\begin{pmatrix}\lambda_1 & & \\ & \ddots & \\ & & \lambda_n\end{pmatrix}}_{D}$$

即 $AP = PD$。若 $P$ 可逆（即 $\mathbf{p}_1, \ldots, \mathbf{p}_n$ 线性无关），则 $A = PDP^{-1}$。

**结论**：
- $P$ 的各列是 $A$ 的 $n$ 个线性无关特征向量
- $D$ 的对角线元素是对应的特征值（顺序与 $P$ 的列对应）

### 可对角化的充要条件

**定理**：$n \times n$ 矩阵 $A$ 可对角化，当且仅当 $A$ 有 $n$ 个**线性无关**的特征向量。

**充分条件（更易验证）**：若 $A$ 的 $n$ 个特征值**互不相同**，则 $A$ 一定可对角化。

**证明思路**：属于不同特征值的特征向量线性无关（可用归纳法证明）。若 $n$ 个特征值各不相同，则 $n$ 个特征向量线性无关，故 $P$ 可逆。$\square$

**注意**：此充分条件不是必要条件。有重特征值的矩阵也可能可对角化（如数量矩阵 $\lambda I$），关键在于每个特征值的**几何重数**是否等于**代数重数**。

| 特征值情形 | 可对角化条件 |
|:---|:---|
| $n$ 个不同特征值 | 一定可对角化 |
| 有重特征值 $\lambda_k$（代数重数 $= m_k$） | 当且仅当 $\dim(\ker(A - \lambda_k I)) = m_k$（每个特征值的几何重数 = 代数重数） |
| 特征多项式有不可约实因子 | 在实数域不可对角化（但在复数域可能可以） |

---

## 20.2 对角化的步骤

### 算法：四步流程

**输入**：$n \times n$ 矩阵 $A$

**输出**：可逆矩阵 $P$ 和对角矩阵 $D$，满足 $A = PDP^{-1}$（或判断不可对角化）

**步骤一**：求特征多项式 $p(\lambda) = \det(\lambda I - A)$，解方程 $p(\lambda) = 0$，得所有特征值 $\lambda_1, \ldots, \lambda_k$（注意代数重数）。

**步骤二**：对每个特征值 $\lambda_i$，求解 $(A - \lambda_i I)\mathbf{x} = \mathbf{0}$，得特征空间 $\ker(A - \lambda_i I)$ 的基。

**步骤三**：若所有特征空间的基向量总数恰好为 $n$，则将它们合并为 $P$ 的列，对应特征值排列为 $D$ 的对角线。否则矩阵不可对角化。

**步骤四（可选）**：计算 $P^{-1}$，验证 $P^{-1}AP = D$。

### 完整计算示例

**例**：对矩阵 $A = \begin{pmatrix}4&1\\2&3\end{pmatrix}$ 进行对角化。

**步骤一**：特征多项式

$$\det(\lambda I - A) = \det\begin{pmatrix}\lambda-4&-1\\-2&\lambda-3\end{pmatrix} = (\lambda-4)(\lambda-3) - (-1)(-2) = \lambda^2 - 7\lambda + 10 = (\lambda-5)(\lambda-2)$$

特征值：$\lambda_1 = 5$，$\lambda_2 = 2$（两个不同实特征值，一定可对角化）。

**步骤二**：求特征向量

对 $\lambda_1 = 5$：$(A - 5I)\mathbf{x} = \mathbf{0}$

$$A - 5I = \begin{pmatrix}-1&1\\2&-2\end{pmatrix} \xrightarrow{\text{行化简}} \begin{pmatrix}1&-1\\0&0\end{pmatrix}$$

解为 $x_1 = x_2$，取 $\mathbf{p}_1 = \begin{pmatrix}1\\1\end{pmatrix}$。

对 $\lambda_2 = 2$：$(A - 2I)\mathbf{x} = \mathbf{0}$

$$A - 2I = \begin{pmatrix}2&1\\2&1\end{pmatrix} \xrightarrow{\text{行化简}} \begin{pmatrix}2&1\\0&0\end{pmatrix}$$

解为 $2x_1 + x_2 = 0$，取 $\mathbf{p}_2 = \begin{pmatrix}1\\-2\end{pmatrix}$。

**步骤三**：构造 $P$ 和 $D$

$$P = \begin{pmatrix}1&1\\1&-2\end{pmatrix}, \quad D = \begin{pmatrix}5&0\\0&2\end{pmatrix}$$

**步骤四**：计算 $P^{-1}$

$$P^{-1} = \frac{1}{-2-1}\begin{pmatrix}-2&-1\\-1&1\end{pmatrix} = \frac{1}{3}\begin{pmatrix}2&1\\1&-1\end{pmatrix}$$

验证：

$$P^{-1}AP = \frac{1}{3}\begin{pmatrix}2&1\\1&-1\end{pmatrix}\begin{pmatrix}4&1\\2&3\end{pmatrix}\begin{pmatrix}1&1\\1&-2\end{pmatrix} = \begin{pmatrix}5&0\\0&2\end{pmatrix} = D \quad \checkmark$$

### 几何直觉

对角化 $A = PDP^{-1}$ 可以理解为三步复合变换：

```
标准坐标系                特征向量坐标系
    x ──────────────────────────────→ Ax
          "A 的作用"（复杂）

    x ──P^{-1}──→ P^{-1}x ──D──→ D(P^{-1}x) ──P──→ Ax
        坐标变换     对角缩放（简单）   回到标准坐标
```

$P^{-1}$：将标准坐标转换为特征向量基下的坐标（"看懂"矩阵的"自然语言"）
$D$：在特征方向上分别缩放（沿 $\mathbf{p}_i$ 方向乘以 $\lambda_i$）
$P$：将特征坐标转换回标准坐标

**本质**：对角化将"复杂的线性变换"分解为"换个坐标系 + 简单缩放 + 换回来"。

---

## 20.3 矩阵幂的计算

### 公式推导

对角化最重要的应用之一是高效计算矩阵的幂。

由 $A = PDP^{-1}$ 出发：

$$A^2 = (PDP^{-1})(PDP^{-1}) = PD\underbrace{(P^{-1}P)}_{I}DP^{-1} = PD^2P^{-1}$$

归纳可得：

$$\boxed{A^n = PD^nP^{-1}}$$

其中对角矩阵的幂极易计算：

$$D^n = \begin{pmatrix}\lambda_1^n & & \\ & \ddots & \\ & & \lambda_k^n\end{pmatrix}$$

**计算代价对比**：
- 直接矩阵乘法求 $A^n$：需要 $O(n^3 \log n)$ 次运算（快速幂），且大 $n$ 时数值误差积累
- 对角化方法求 $A^n$：只需计算 $\lambda_i^n$，代价 $O(n)$（求幂后代入公式），**快得多**

### 计算示例：矩阵的高次幂

**例**：利用上例中 $A = \begin{pmatrix}4&1\\2&3\end{pmatrix}$ 的对角化，计算 $A^{10}$。

由 $A = PDP^{-1}$，

$$A^{10} = PD^{10}P^{-1} = \begin{pmatrix}1&1\\1&-2\end{pmatrix}\begin{pmatrix}5^{10}&0\\0&2^{10}\end{pmatrix}\frac{1}{3}\begin{pmatrix}2&1\\1&-1\end{pmatrix}$$

$$= \frac{1}{3}\begin{pmatrix}5^{10}&2^{10}\\5^{10}&-2\cdot2^{10}\end{pmatrix}\begin{pmatrix}2&1\\1&-1\end{pmatrix}$$

$$= \frac{1}{3}\begin{pmatrix}2\cdot5^{10}+2^{10} & 5^{10}-2^{10} \\ 2\cdot5^{10}-2\cdot2^{10} & 5^{10}+2\cdot2^{10}\end{pmatrix}$$

代入 $5^{10} = 9765625$，$2^{10} = 1024$：

$$A^{10} = \frac{1}{3}\begin{pmatrix}19532274 & 9764601 \\ 19529202 & 9767673\end{pmatrix}$$

若用直接矩阵乘法，需要 9 次 $2\times2$ 矩阵乘法（或更多），对角化方法大幅节省计算量。

### 矩阵指数函数

对角化也使矩阵指数 $e^A$ 的定义与计算变得自然：

$$e^A \triangleq \sum_{k=0}^{\infty}\frac{A^k}{k!} = P\left(\sum_{k=0}^{\infty}\frac{D^k}{k!}\right)P^{-1} = P \begin{pmatrix}e^{\lambda_1}&&\\&\ddots&\\&&e^{\lambda_n}\end{pmatrix} P^{-1}$$

矩阵指数在求解线性微分方程组 $\dot{\mathbf{x}} = A\mathbf{x}$ 时有直接应用（见 20.5 节）。

---

## 20.4 不可对角化的矩阵

### 亏损矩阵

若 $n \times n$ 矩阵 $A$ 没有 $n$ 个线性无关的特征向量，则称其为**亏损矩阵（defective matrix）**，不可对角化。

亏损矩阵的必要条件：**至少有一个特征值的几何重数小于其代数重数**。

**例**：矩阵 $A = \begin{pmatrix}2&1\\0&2\end{pmatrix}$（Jordan 块）。

特征多项式：$(\lambda-2)^2 = 0$，特征值 $\lambda = 2$（代数重数 = 2）。

特征空间：$(A - 2I) = \begin{pmatrix}0&1\\0&0\end{pmatrix}$，解空间为 $\{(x_1, 0)^T : x_1 \in \mathbb{R}\}$，维数 = 1（几何重数 = 1）。

几何重数（1）< 代数重数（2），故 $A$ **不可对角化**。

**几何直觉**：这个矩阵在二维空间中"只有一个特征方向"（$x$ 轴方向），它在 $x$ 轴上只是缩放，但在 $y$ 方向还有一个"剪切"分量，无法通过换基消除。

```
y │       ↗ 剪切：(0,1) → (1,2)
  │      /
  │─────────→ x
  │ 缩放 2 倍（特征方向）
```

### Jordan 标准形（简介）

当矩阵不可对角化时，最接近的"标准形式"是 **Jordan 标准形（Jordan Normal Form）**：

$$A = PJP^{-1}, \quad J = \begin{pmatrix}J_1 & & \\ & \ddots & \\ & & J_k\end{pmatrix}$$

每个 **Jordan 块** $J_i$ 形如：

$$J_i = \begin{pmatrix}\lambda_i & 1 & & \\ & \lambda_i & \ddots & \\ & & \ddots & 1\\ & & & \lambda_i\end{pmatrix}$$

对角线是特征值 $\lambda_i$，超对角线是 1（或 0 对大小为 1 的块）。

**要点**：
- 若 $A$ 可对角化，Jordan 标准形就是对角矩阵 $D$（所有 Jordan 块大小为 1）
- Jordan 块大小 > 1 是"不可对角化"的精确刻画
- Jordan 标准形对矩阵的幂和指数计算仍然有效，但公式更复杂（涉及 $n\lambda^{n-1}$ 等项）
- 数值计算中，Jordan 标准形对微小扰动极敏感，通常改用 Schur 分解或 SVD 替代

---

## 20.5 对角化的应用

### 应用一：线性微分方程组

考虑常系数线性微分方程组：

$$\dot{\mathbf{x}}(t) = A\mathbf{x}(t), \quad \mathbf{x}(0) = \mathbf{x}_0$$

**解法**：利用坐标变换 $\mathbf{y}(t) = P^{-1}\mathbf{x}(t)$，代入方程得：

$$P\dot{\mathbf{y}} = AP\mathbf{y} = PDP^{-1}P\mathbf{y} = PD\mathbf{y}$$

$$\dot{\mathbf{y}} = D\mathbf{y}$$

对角系统 $\dot{y}_i = \lambda_i y_i$ 是 $n$ 个**独立的标量方程**，解为 $y_i(t) = y_i(0)e^{\lambda_i t}$。回代：

$$\boxed{\mathbf{x}(t) = P e^{Dt} P^{-1} \mathbf{x}_0 = \sum_{i=1}^n c_i e^{\lambda_i t}\mathbf{p}_i}$$

其中 $\mathbf{c} = P^{-1}\mathbf{x}_0$ 是初始条件在特征向量基下的坐标。

**稳定性判断**：
- 所有 $\text{Re}(\lambda_i) < 0$：解趋于零（**渐近稳定**）
- 存在 $\text{Re}(\lambda_i) > 0$：解发散（**不稳定**）
- 所有 $\text{Re}(\lambda_i) \leq 0$：解有界（**稳定**，可能振荡）

### 应用二：Markov 链的长期行为

设 $\mathbf{x}_k$ 是第 $k$ 步的状态分布向量，Markov 链的转移方程为：

$$\mathbf{x}_{k+1} = M\mathbf{x}_k \implies \mathbf{x}_k = M^k \mathbf{x}_0$$

对 Markov 转移矩阵 $M$，由 Perron-Frobenius 定理：主特征值 $\lambda_1 = 1$，其余 $|\lambda_i| < 1$（若 $M$ 是正则 Markov 矩阵）。

利用对角化 $M^k = PD^kP^{-1}$，当 $k \to \infty$ 时：

$$D^k = \begin{pmatrix}1^k & & & \\ & \lambda_2^k & & \\ & & \ddots & \\ & & & \lambda_n^k\end{pmatrix} \to \begin{pmatrix}1 & & \\ & 0 & \\ & & \ddots\end{pmatrix}$$

非主特征值项以指数速度衰减，链以速率 $|\lambda_2|^k$ 收敛到**平稳分布** $\boldsymbol{\pi}$（对应 $\lambda_1 = 1$ 的特征向量）。

**混合时间**正比于 $\dfrac{1}{|\log \lambda_2|}$，第二大特征值 $\lambda_2$ 越小，收敛越快。

### 应用三：Fibonacci 数列

Fibonacci 数列 $F_0=0, F_1=1, F_{n}=F_{n-1}+F_{n-2}$ 可以用矩阵幂表达：

$$\begin{pmatrix}F_{n+1}\\F_n\end{pmatrix} = \underbrace{\begin{pmatrix}1&1\\1&0\end{pmatrix}}_{A}\begin{pmatrix}F_n\\F_{n-1}\end{pmatrix} \implies \begin{pmatrix}F_{n+1}\\F_n\end{pmatrix} = A^n\begin{pmatrix}1\\0\end{pmatrix}$$

矩阵 $A$ 的特征值（黄金比例！）：

$$\lambda_{1,2} = \frac{1 \pm \sqrt{5}}{2}, \quad \phi = \frac{1+\sqrt{5}}{2} \approx 1.618, \quad \psi = \frac{1-\sqrt{5}}{2} \approx -0.618$$

对角化后得到著名的 **Binet 公式**：

$$\boxed{F_n = \frac{\phi^n - \psi^n}{\sqrt{5}}}$$

由于 $|\psi| < 1$，当 $n$ 很大时 $\psi^n \to 0$，故 $F_n \approx \dfrac{\phi^n}{\sqrt{5}}$（取最近整数）。相邻 Fibonacci 数之比趋向黄金比例：$\dfrac{F_{n+1}}{F_n} \to \phi$。

---

## 本章小结

- **对角化定义**：$A = PDP^{-1}$，$P$ 的列为线性无关的特征向量，$D$ 的对角线为对应特征值

- **可对角化条件**：$A$ 有 $n$ 个线性无关特征向量；充分条件是 $n$ 个不同特征值；一般条件是每个特征值的几何重数等于代数重数

- **对角化算法**：求特征值 → 求各特征空间的基 → 合并为 $P$ 的列 → 验证 $P$ 可逆

- **矩阵幂**：$A^n = PD^nP^{-1}$，将矩阵幂转化为标量幂，计算高效

- **亏损矩阵**：几何重数 < 代数重数时不可对角化；最一般的标准形是 Jordan 标准形

- **应用**：微分方程（特征值决定稳定性）、Markov 链（谱间隙决定收敛速度）、Fibonacci 数列（Binet 公式）

| 概念 | 公式 | 要点 |
|:---|:---|:---|
| 对角化 | $A = PDP^{-1}$ | $P$ 的列：$n$ 个线性无关特征向量 |
| 矩阵幂 | $A^n = PD^nP^{-1}$ | $D^n$ 只需标量幂 |
| 矩阵指数 | $e^A = Pe^DP^{-1}$ | $e^D = \text{diag}(e^{\lambda_i})$ |
| 可对角化充分条件 | $n$ 个不同特征值 | 不是必要条件 |
| Jordan 标准形 | $A = PJP^{-1}$ | 超对角线为 1 的块 |

---

## 深度学习应用：RNN 的梯度分析

### 背景：循环神经网络的梯度问题

**循环神经网络（RNN）**通过共享权重矩阵 $W$ 在时间步上递归处理序列。对于长度为 $T$ 的序列，隐层状态的递推关系为：

$$\mathbf{h}_t = \tanh(W\mathbf{h}_{t-1} + U\mathbf{x}_t + \mathbf{b})$$

训练时，通过反向传播（BPTT）计算梯度。在线性化近似下（忽略 $\tanh$ 的非线性），梯度沿时间步反传：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = W^T \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}}$$

梯度从时刻 $T$ 反传到时刻 $1$，经过 $T-1$ 步，涉及 $W^T$ 的 $(T-1)$ 次幂：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = (W^T)^{T-1} \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} = (W^{T-1})^T \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T}$$

### 通过特征值分析梯度行为

设 $W$ 可对角化（复数域），$W = PDP^{-1}$，特征值为 $\lambda_1, \ldots, \lambda_n$。则：

$$W^{T-1} = PD^{T-1}P^{-1}, \quad D^{T-1} = \text{diag}(\lambda_1^{T-1}, \ldots, \lambda_n^{T-1})$$

梯度沿特征方向的分量以 $\lambda_i^{T-1}$ 的速率缩放：

$$\text{梯度第 }i\text{ 个特征分量} \propto \lambda_i^{T-1}$$

这直接导致两个经典问题：

**梯度消失（Vanishing Gradient）**：若 $|\lambda_i| < 1$，则 $|\lambda_i|^{T-1} \to 0$（指数衰减）。当 $T$ 很大时，早期时刻的梯度趋近于零，网络无法学习长程依赖。

**梯度爆炸（Exploding Gradient）**：若 $|\lambda_i| > 1$，则 $|\lambda_i|^{T-1} \to \infty$（指数增长）。梯度快速增大，训练发散。

```
梯度幅值（对数坐标）
  ^
  |  |\lambda|>1: 爆炸 ↗↗↗↗
  |
  |─────────────────────── |\lambda|=1: 稳定（理想情况）
  |
  |  |\lambda|<1: 消失 ↘↘↘↘
  +────────────────────────────────→ 时间步 T
```

### 谱半径与稳定性

矩阵 $W$ 的**谱半径（spectral radius）**定义为：

$$\rho(W) = \max_i |\lambda_i|$$

谱半径是判断 RNN 梯度行为的关键量：

- $\rho(W) < 1$：梯度消失，信息无法跨越长时间步传播
- $\rho(W) > 1$：梯度爆炸，训练不稳定
- $\rho(W) \approx 1$：梯度近似保持，是 RNN 最理想的初始化状态

**正交初始化**：若 $W$ 是正交矩阵（$W^TW = I$），则所有奇异值为 1，$\rho(W) = 1$，梯度既不爆炸也不消失——这就是 RNN 中**正交初始化（orthogonal initialization）**的理论依据。

**梯度裁剪（Gradient Clipping）**：梯度爆炸的实用解决方案。若梯度范数超过阈值 $\theta$，则按比例缩放：$\mathbf{g} \leftarrow \mathbf{g} \cdot \theta / \|\mathbf{g}\|$。

**LSTM/GRU 的本质**：长短期记忆网络（LSTM）通过引入"遗忘门"和"输入门"，将简单的矩阵幂传递改为自适应的加性传递，从架构层面缓解梯度消失问题。

### PyTorch 代码示例：分析 RNN 权重矩阵的特征值

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

# ============================================================
# 1. 创建并训练一个简单 RNN，分析权重矩阵的谱
# ============================================================

hidden_size = 64
seq_len = 50       # 序列长度（梯度需反传 50 步）

rnn = nn.RNN(input_size=1, hidden_size=hidden_size,
             batch_first=True, nonlinearity='tanh')

# 提取隐层到隐层的权重矩阵 W_hh（即递推权重 W）
W = rnn.weight_hh_l0.detach().numpy()   # shape: (hidden_size, hidden_size)
print(f"权重矩阵 W_hh 形状：{W.shape}")

# ============================================================
# 2. 计算特征值（复数域）
# ============================================================

eigenvalues = np.linalg.eigvals(W)       # 复数特征值
magnitudes = np.abs(eigenvalues)         # 模
spectral_radius = magnitudes.max()

print(f"\n=== 特征值分析（默认 PyTorch 初始化）===")
print(f"特征值个数：{len(eigenvalues)}")
print(f"谱半径 ρ(W) = {spectral_radius:.4f}")
print(f"最大模 = {magnitudes.max():.4f}")
print(f"最小模 = {magnitudes.min():.4f}")
print(f"模均值 = {magnitudes.mean():.4f}")

# ============================================================
# 3. 模拟梯度传播：计算 ||W^k|| 随时间步的变化
# ============================================================

W_tensor = torch.tensor(W, dtype=torch.float32)
W_power = torch.eye(hidden_size)    # W^0 = I
norms = [1.0]                       # ||W^0|| = 1

for k in range(1, seq_len + 1):
    W_power = W_power @ W_tensor
    norms.append(W_power.norm().item())

# ============================================================
# 4. 正交初始化的对比
# ============================================================

# 正交初始化：随机生成正交矩阵
W_orth_np = np.linalg.qr(np.random.randn(hidden_size, hidden_size))[0]
W_orth = torch.tensor(W_orth_np, dtype=torch.float32)
eigs_orth = np.linalg.eigvals(W_orth_np)
rho_orth = np.abs(eigs_orth).max()

W_orth_power = torch.eye(hidden_size)
norms_orth = [1.0]
for k in range(1, seq_len + 1):
    W_orth_power = W_orth_power @ W_orth
    norms_orth.append(W_orth_power.norm().item())

print(f"\n=== 正交初始化 ===")
print(f"谱半径 ρ(W_orth) = {rho_orth:.4f}（理论值应≈1.0）")

# ============================================================
# 5. 可视化：特征值分布 + 矩阵幂范数
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 图1：默认初始化的特征值分布（复平面）
ax = axes[0]
theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, alpha=0.5,
        label='单位圆 |λ|=1')
ax.scatter(eigenvalues.real, eigenvalues.imag,
           c=magnitudes, cmap='RdYlGn_r', s=30, zorder=5)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('实部 Re(λ)')
ax.set_ylabel('虚部 Im(λ)')
ax.set_title(f'默认初始化特征值分布\nρ(W) = {spectral_radius:.3f}')
ax.set_aspect('equal')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 图2：正交初始化的特征值分布
ax = axes[1]
ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, alpha=0.5,
        label='单位圆 |λ|=1')
ax.scatter(eigs_orth.real, eigs_orth.imag,
           c=np.abs(eigs_orth), cmap='RdYlGn_r', s=30, zorder=5,
           vmin=0.8, vmax=1.2)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('实部 Re(λ)')
ax.set_ylabel('虚部 Im(λ)')
ax.set_title(f'正交初始化特征值分布\nρ(W) = {rho_orth:.3f}')
ax.set_aspect('equal')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 图3：矩阵幂范数 ||W^k|| 随时间步的变化
ax = axes[2]
steps = list(range(seq_len + 1))
ax.semilogy(steps, norms, 'b-o', markersize=3, linewidth=1.5,
            label=f'默认初始化 (ρ={spectral_radius:.2f})')
ax.semilogy(steps, norms_orth, 'g-s', markersize=3, linewidth=1.5,
            label=f'正交初始化 (ρ≈{rho_orth:.2f})')
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7,
           label='基准线 ||W^k||=1')
ax.set_xlabel('时间步 k（梯度反传步数）')
ax.set_ylabel('||W^k||（对数坐标）')
ax.set_title('矩阵幂范数：梯度传播能力的代理指标')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('rnn_eigenvalue_analysis.png', dpi=150, bbox_inches='tight')
print("\n图像已保存为 rnn_eigenvalue_analysis.png")

# ============================================================
# 6. 对角化视角：分解 W 并验证矩阵幂
# ============================================================

print("\n=== 对角化视角 ===")
# 对实矩阵用 Schur 分解（比特征分解更数值稳定）
# 这里演示复数域的特征分解
eigs_vals, eigs_vecs = np.linalg.eig(W)           # W = P D P^{-1}
P = eigs_vecs
D_diag = np.diag(eigs_vals)
P_inv = np.linalg.inv(P)

# 验证对角化：W ≈ P D P^{-1}
W_reconstructed = (P @ D_diag @ P_inv).real
reconstruction_error = np.abs(W - W_reconstructed).max()
print(f"对角化重构误差（最大绝对值）：{reconstruction_error:.2e}")

# 用对角化计算 W^5，与直接乘法对比
k_test = 5
D_k = np.diag(eigs_vals ** k_test)                # D^5 = diag(λ_i^5)
W5_diag = (P @ D_k @ P_inv).real                  # W^5 via 对角化
W5_direct = np.linalg.matrix_power(W, k_test)     # W^5 via 直接乘法
error_k = np.abs(W5_diag - W5_direct).max()
print(f"W^{k_test} 对角化 vs 直接乘法误差：{error_k:.2e}")

# 谱半径预测矩阵幂的渐近增长率
print(f"\n谱半径 ρ(W) = {spectral_radius:.4f}")
print(f"预测 ||W^{seq_len}|| ≈ ρ(W)^{seq_len} = {spectral_radius**seq_len:.2e}")
print(f"实际 ||W^{seq_len}|| = {norms[-1]:.2e}")
print("（实际值与预测值之比由特征向量矩阵的条件数决定）")
```

**代码解读**：

- **第1-2部分**：提取 PyTorch RNN 的隐层权重矩阵 `W_hh`，用 `numpy.linalg.eigvals` 计算复数特征值，观察谱半径与特征值的分布
- **第3部分**：通过迭代矩阵乘法 $W^k$ 计算范数，以此代理梯度反传 $k$ 步后的幅值变化
- **第4部分**：对比正交初始化（$Q$ 来自 QR 分解），验证正交矩阵的谱半径恒为 1
- **第5部分**：三图可视化——复平面特征值分布图（单位圆是分界线）、矩阵幂范数的指数增长/衰减曲线
- **第6部分**：用 `numpy.linalg.eig` 做特征分解 $W = PDP^{-1}$，验证重构误差，并对比 $D^k$ 与直接矩阵幂的计算精度

### 实践建议

| 问题 | 解决方案 | 数学原理 |
|:---|:---|:---|
| 梯度消失 | 使用 LSTM/GRU；正交初始化 | 避免 $\rho(W) \ll 1$ |
| 梯度爆炸 | 梯度裁剪（`clip_grad_norm_`） | 限制 $\|\mathbf{g}\|$ 上界 |
| 谱半径调整 | 初始化时缩放：$W \leftarrow W / \rho(W)$ | 强制 $\rho(W) = 1$ |
| 长程依赖 | 注意力机制（Transformer）| 直接建立任意时刻间联系 |

**深层启示**：Transformer 的自注意力机制本质上绕过了矩阵幂累积的问题——每个位置都直接与所有其他位置交互，梯度路径长度始终为常数，不随序列长度增长，从根本上消除了梯度消失/爆炸的谱半径瓶颈。

---

## 练习题

**练习 1**（基础——对角化判断与计算）

对矩阵 $A = \begin{pmatrix}3&-2\\1&0\end{pmatrix}$：

（a）求特征多项式和特征值

（b）求每个特征值对应的特征向量

（c）写出对角化分解 $A = PDP^{-1}$，明确给出 $P$、$D$、$P^{-1}$

---

**练习 2**（基础——矩阵幂）

设 $B = \begin{pmatrix}2&1\\0&3\end{pmatrix}$。

（a）求 $B$ 的特征值和特征向量，写出对角化 $B = PDP^{-1}$

（b）利用对角化计算 $B^5$

（c）不用对角化，直接验证 $B^2 = \begin{pmatrix}4&5\\0&9\end{pmatrix}$，并说明为何对角化方法在计算 $B^{100}$ 时更有优势

---

**练习 3**（中等——不可对角化的识别）

判断下列矩阵是否可对角化，若可以则给出 $P$ 和 $D$，若不可以则说明原因：

（a）$C_1 = \begin{pmatrix}5&0\\0&5\end{pmatrix}$

（b）$C_2 = \begin{pmatrix}5&1\\0&5\end{pmatrix}$

（c）$C_3 = \begin{pmatrix}1&0&0\\0&2&1\\0&0&2\end{pmatrix}$

---

**练习 4**（中等——Fibonacci 数列）

Fibonacci 矩阵 $A = \begin{pmatrix}1&1\\1&0\end{pmatrix}$ 的特征值为 $\phi = \dfrac{1+\sqrt{5}}{2}$ 和 $\psi = \dfrac{1-\sqrt{5}}{2}$。

（a）求对应的特征向量，写出对角化 $A = PDP^{-1}$

（b）由 $A^n \begin{pmatrix}1\\0\end{pmatrix} = \begin{pmatrix}F_{n+1}\\F_n\end{pmatrix}$，推导 Binet 公式 $F_n = \dfrac{\phi^n - \psi^n}{\sqrt{5}}$

（c）计算 $F_{10}$，与递推结果验证一致

---

**练习 5**（进阶——微分方程组与谱半径）

考虑线性微分方程组：

$$\dot{\mathbf{x}} = A\mathbf{x}, \quad A = \begin{pmatrix}-3&1\\2&-2\end{pmatrix}, \quad \mathbf{x}(0) = \begin{pmatrix}2\\1\end{pmatrix}$$

（a）求 $A$ 的特征值和特征向量，写出对角化 $A = PDP^{-1}$

（b）写出方程组的通解 $\mathbf{x}(t) = c_1 e^{\lambda_1 t}\mathbf{p}_1 + c_2 e^{\lambda_2 t}\mathbf{p}_2$，确定常数 $c_1, c_2$

（c）判断 $\mathbf{x}(t)$ 当 $t \to \infty$ 时的行为（趋于零？趋于无穷？振荡？）并给出直觉解释

（d）若将 $A$ 视为一个 RNN 权重矩阵（离散化步长 $\Delta t = 0.1$，$W = I + 0.1A$），计算 $W$ 的谱半径，并判断对应的 RNN 是否存在梯度消失问题

---

## 练习答案

<details>
<summary>点击展开 练习1 答案</summary>

**（a）特征多项式和特征值**

$$\det(\lambda I - A) = \det\begin{pmatrix}\lambda-3 & 2 \\ -1 & \lambda\end{pmatrix} = \lambda(\lambda-3)+2 = \lambda^2-3\lambda+2 = (\lambda-1)(\lambda-2)$$

特征值：$\lambda_1 = 1$，$\lambda_2 = 2$（两个不同实特征值，一定可对角化）。

**（b）特征向量**

对 $\lambda_1 = 1$：$(A - I)\mathbf{x} = \mathbf{0}$

$$A - I = \begin{pmatrix}2&-2\\1&-1\end{pmatrix} \to \begin{pmatrix}1&-1\\0&0\end{pmatrix}$$

解：$x_1 = x_2$，取 $\mathbf{p}_1 = \begin{pmatrix}1\\1\end{pmatrix}$。

对 $\lambda_2 = 2$：$(A - 2I)\mathbf{x} = \mathbf{0}$

$$A - 2I = \begin{pmatrix}1&-2\\1&-2\end{pmatrix} \to \begin{pmatrix}1&-2\\0&0\end{pmatrix}$$

解：$x_1 = 2x_2$，取 $\mathbf{p}_2 = \begin{pmatrix}2\\1\end{pmatrix}$。

**（c）对角化分解**

$$P = \begin{pmatrix}1&2\\1&1\end{pmatrix}, \quad D = \begin{pmatrix}1&0\\0&2\end{pmatrix}$$

$$P^{-1} = \frac{1}{1\cdot1 - 2\cdot1}\begin{pmatrix}1&-2\\-1&1\end{pmatrix} = \begin{pmatrix}-1&2\\1&-1\end{pmatrix}$$

验证：$PDP^{-1} = \begin{pmatrix}1&2\\1&1\end{pmatrix}\begin{pmatrix}1&0\\0&2\end{pmatrix}\begin{pmatrix}-1&2\\1&-1\end{pmatrix} = \begin{pmatrix}1&4\\1&2\end{pmatrix}\begin{pmatrix}-1&2\\1&-1\end{pmatrix} = \begin{pmatrix}3&-2\\1&0\end{pmatrix} = A$ $\checkmark$

</details>

<details>
<summary>点击展开 练习2 答案</summary>

**（a）特征值和特征向量**

$$\det(\lambda I - B) = (\lambda-2)(\lambda-3) = 0 \implies \lambda_1 = 2, \lambda_2 = 3$$

对 $\lambda_1 = 2$：$(B-2I) = \begin{pmatrix}0&1\\0&1\end{pmatrix} \to \begin{pmatrix}0&1\\0&0\end{pmatrix}$，解 $x_2 = 0$，取 $\mathbf{p}_1 = \begin{pmatrix}1\\0\end{pmatrix}$。

对 $\lambda_2 = 3$：$(B-3I) = \begin{pmatrix}-1&1\\0&0\end{pmatrix} \to \begin{pmatrix}1&-1\\0&0\end{pmatrix}$，解 $x_1 = x_2$，取 $\mathbf{p}_2 = \begin{pmatrix}1\\1\end{pmatrix}$。

$$P = \begin{pmatrix}1&1\\0&1\end{pmatrix}, \quad D = \begin{pmatrix}2&0\\0&3\end{pmatrix}, \quad P^{-1} = \begin{pmatrix}1&-1\\0&1\end{pmatrix}$$

**（b）计算 $B^5$**

$$B^5 = PD^5P^{-1} = \begin{pmatrix}1&1\\0&1\end{pmatrix}\begin{pmatrix}32&0\\0&243\end{pmatrix}\begin{pmatrix}1&-1\\0&1\end{pmatrix}$$

$$= \begin{pmatrix}32&243\\0&243\end{pmatrix}\begin{pmatrix}1&-1\\0&1\end{pmatrix} = \begin{pmatrix}32&211\\0&243\end{pmatrix}$$

**（c）验证 $B^2$ 及对角化优势**

直接计算：$B^2 = \begin{pmatrix}2&1\\0&3\end{pmatrix}\begin{pmatrix}2&1\\0&3\end{pmatrix} = \begin{pmatrix}4&2+3\\0&9\end{pmatrix} = \begin{pmatrix}4&5\\0&9\end{pmatrix}$ $\checkmark$

计算 $B^{100}$ 时，直接乘法需要 $O(\log 100)$ 次矩阵乘法，每次代价 $O(n^3)$；而对角化方法只需计算 $2^{100}$ 和 $3^{100}$（标量幂），代价 $O(n^2)$（矩阵乘法），且结果更精确。对于大 $n$ 或大幂次，差异极为显著。

</details>

<details>
<summary>点击展开 练习3 答案</summary>

**（a）$C_1 = \begin{pmatrix}5&0\\0&5\end{pmatrix} = 5I$**

特征值 $\lambda = 5$，代数重数 = 2。特征空间：$(C_1 - 5I) = 0$，故 $\ker(C_1 - 5I) = \mathbb{R}^2$，几何重数 = 2 = 代数重数。

**可对角化**。实际上 $C_1$ 已经是对角矩阵：$P = I$，$D = 5I$。

**（b）$C_2 = \begin{pmatrix}5&1\\0&5\end{pmatrix}$**

特征值 $\lambda = 5$，代数重数 = 2。特征空间：$(C_2 - 5I) = \begin{pmatrix}0&1\\0&0\end{pmatrix}$，解空间为 $\{(x_1, 0)^T\}$，几何重数 = 1 < 2 = 代数重数。

**不可对角化**。$C_2$ 是 $2 \times 2$ Jordan 块，只有一个线性无关的特征向量 $\begin{pmatrix}1\\0\end{pmatrix}$，不足以构成 $\mathbb{R}^2$ 的基。

**（c）$C_3 = \begin{pmatrix}1&0&0\\0&2&1\\0&0&2\end{pmatrix}$**

特征值：$\lambda_1 = 1$（代数重数 1），$\lambda_2 = 2$（代数重数 2）。

对 $\lambda_1 = 1$：$(C_3 - I) = \begin{pmatrix}0&0&0\\0&1&1\\0&0&1\end{pmatrix}$，化简得特征向量 $\mathbf{p}_1 = \begin{pmatrix}1\\0\\0\end{pmatrix}$（几何重数 = 1 = 代数重数，$\checkmark$）。

对 $\lambda_2 = 2$：$(C_3 - 2I) = \begin{pmatrix}-1&0&0\\0&0&1\\0&0&0\end{pmatrix}$，化简解：$x_1 = 0$，$x_3 = 0$，$x_2$ 自由，特征向量 $\mathbf{p}_2 = \begin{pmatrix}0\\1\\0\end{pmatrix}$（几何重数 = 1 < 2 = 代数重数，**不满足条件**）。

**不可对角化**。$\lambda_2 = 2$ 的特征空间只有一维，而代数重数为 2，缺少一个特征向量。

</details>

<details>
<summary>点击展开 练习4 答案</summary>

**（a）特征向量与对角化**

对 $\phi = \dfrac{1+\sqrt{5}}{2}$：$(A - \phi I)\mathbf{x} = \mathbf{0}$

$$A - \phi I = \begin{pmatrix}1-\phi & 1\\ 1 & -\phi\end{pmatrix}$$

由 $\phi^2 = \phi + 1$（黄金比例的性质），$(1-\phi)(-\phi) = \phi^2 - \phi = 1$，故两行共线。解：$x_1 = \phi x_2$，取 $\mathbf{p}_1 = \begin{pmatrix}\phi\\1\end{pmatrix}$。

同理对 $\psi = \dfrac{1-\sqrt{5}}{2}$，取 $\mathbf{p}_2 = \begin{pmatrix}\psi\\1\end{pmatrix}$。

$$P = \begin{pmatrix}\phi & \psi \\ 1 & 1\end{pmatrix}, \quad D = \begin{pmatrix}\phi & 0 \\ 0 & \psi\end{pmatrix}$$

$P^{-1}$：$\det(P) = \phi - \psi = \sqrt{5}$，故

$$P^{-1} = \frac{1}{\sqrt{5}}\begin{pmatrix}1 & -\psi \\ -1 & \phi\end{pmatrix}$$

**（b）推导 Binet 公式**

$$\begin{pmatrix}F_{n+1}\\F_n\end{pmatrix} = A^n\begin{pmatrix}1\\0\end{pmatrix} = PD^nP^{-1}\begin{pmatrix}1\\0\end{pmatrix}$$

计算 $P^{-1}\begin{pmatrix}1\\0\end{pmatrix} = \dfrac{1}{\sqrt{5}}\begin{pmatrix}1\\-1\end{pmatrix}$

$$D^n\cdot\frac{1}{\sqrt{5}}\begin{pmatrix}1\\-1\end{pmatrix} = \frac{1}{\sqrt{5}}\begin{pmatrix}\phi^n\\-\psi^n\end{pmatrix}$$

$$\begin{pmatrix}F_{n+1}\\F_n\end{pmatrix} = \frac{1}{\sqrt{5}}\begin{pmatrix}\phi&\psi\\1&1\end{pmatrix}\begin{pmatrix}\phi^n\\-\psi^n\end{pmatrix} = \frac{1}{\sqrt{5}}\begin{pmatrix}\phi^{n+1}-\psi^{n+1}\\\phi^n - \psi^n\end{pmatrix}$$

因此 $F_n = \dfrac{\phi^n - \psi^n}{\sqrt{5}}$。$\square$

**（c）计算 $F_{10}$**

$\phi^{10} = \left(\dfrac{1+\sqrt{5}}{2}\right)^{10} \approx 122.992$，$\psi^{10} = \left(\dfrac{1-\sqrt{5}}{2}\right)^{10} \approx 0.00820$

$$F_{10} = \frac{122.992 - 0.00820}{\sqrt{5}} \approx \frac{122.984}{2.2361} \approx 55.000$$

取整得 $F_{10} = 55$，与递推 $0,1,1,2,3,5,8,13,21,34,55$ 完全一致。$\checkmark$

</details>

<details>
<summary>点击展开 练习5 答案</summary>

**（a）特征值和特征向量**

$$\det(\lambda I - A) = (\lambda+3)(\lambda+2) - 2 = \lambda^2 + 5\lambda + 4 = (\lambda+1)(\lambda+4)$$

特征值：$\lambda_1 = -1$，$\lambda_2 = -4$。

对 $\lambda_1 = -1$：$(A+I) = \begin{pmatrix}-2&1\\2&-1\end{pmatrix} \to \begin{pmatrix}2&-1\\0&0\end{pmatrix}$，解 $2x_1 = x_2$，取 $\mathbf{p}_1 = \begin{pmatrix}1\\2\end{pmatrix}$。

对 $\lambda_2 = -4$：$(A+4I) = \begin{pmatrix}1&1\\2&2\end{pmatrix} \to \begin{pmatrix}1&1\\0&0\end{pmatrix}$，解 $x_1 = -x_2$，取 $\mathbf{p}_2 = \begin{pmatrix}1\\-1\end{pmatrix}$。

$$P = \begin{pmatrix}1&1\\2&-1\end{pmatrix}, \quad D = \begin{pmatrix}-1&0\\0&-4\end{pmatrix}, \quad P^{-1} = \frac{1}{-3}\begin{pmatrix}-1&-1\\-2&1\end{pmatrix} = \frac{1}{3}\begin{pmatrix}1&1\\2&-1\end{pmatrix}$$

**（b）确定通解与常数**

通解为 $\mathbf{x}(t) = c_1 e^{-t}\begin{pmatrix}1\\2\end{pmatrix} + c_2 e^{-4t}\begin{pmatrix}1\\-1\end{pmatrix}$

由初始条件 $\mathbf{x}(0) = \begin{pmatrix}2\\1\end{pmatrix}$：

$$\begin{pmatrix}c_1\\c_2\end{pmatrix} = P^{-1}\mathbf{x}(0) = \frac{1}{3}\begin{pmatrix}1&1\\2&-1\end{pmatrix}\begin{pmatrix}2\\1\end{pmatrix} = \frac{1}{3}\begin{pmatrix}3\\3\end{pmatrix} = \begin{pmatrix}1\\1\end{pmatrix}$$

**特解**：$\mathbf{x}(t) = e^{-t}\begin{pmatrix}1\\2\end{pmatrix} + e^{-4t}\begin{pmatrix}1\\-1\end{pmatrix}$

**（c）长时行为分析**

由于 $\lambda_1 = -1 < 0$ 且 $\lambda_2 = -4 < 0$，两个特征值均为负数，所以 $e^{-t} \to 0$ 且 $e^{-4t} \to 0$（$t \to \infty$）。

因此 $\mathbf{x}(t) \to \mathbf{0}$：系统是**渐近稳定**的，从任意初始状态出发，解都指数衰减到原点。$e^{-4t}$ 项衰减更快（速率 4），长时间后由 $e^{-t}\mathbf{p}_1$ 主导，解沿特征向量 $\begin{pmatrix}1\\2\end{pmatrix}$ 方向趋向零点。

**（d）离散化后 RNN 的谱半径**

$W = I + 0.1A = \begin{pmatrix}1&0\\0&1\end{pmatrix} + \begin{pmatrix}-0.3&0.1\\0.2&-0.2\end{pmatrix} = \begin{pmatrix}0.7&0.1\\0.2&0.8\end{pmatrix}$

$W$ 的特征值：由 $A$ 的特征值离散化，$\mu_i = 1 + 0.1\lambda_i$，故：

$$\mu_1 = 1 + 0.1 \times (-1) = 0.9, \quad \mu_2 = 1 + 0.1 \times (-4) = 0.6$$

谱半径 $\rho(W) = \max(0.9, 0.6) = 0.9 < 1$。

所有特征值模均小于 1，梯度以 $0.9^k$ 的速率指数衰减。对长序列（如 $k=50$ 步），梯度幅值衰减为 $0.9^{50} \approx 0.005$，**存在明显的梯度消失问题**。这与微分方程的稳定性对应：稳定系统（所有 $\text{Re}(\lambda_i) < 0$）对应离散化后谱半径小于 1 的收缩映射。

</details>
