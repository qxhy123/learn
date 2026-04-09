# 第15章 数项级数

## 学习目标

通过本章学习，你将能够：

- 理解无穷级数的概念，掌握部分和与收敛的定义
- 掌握级数收敛的必要条件及其应用
- 熟练运用正项级数的各种判别法：比较判别法、比值判别法、根值判别法
- 理解交错级数的莱布尼茨判别法及余项估计
- 区分绝对收敛与条件收敛，了解级数重排定理
- 掌握级数的线性运算和柯西乘积

---

## 15.1 级数的概念

### 15.1.1 无穷级数的定义

设 $\{a_n\}$ 是一个数列，称形式和

$$\sum_{n=1}^{\infty} a_n = a_1 + a_2 + a_3 + \cdots + a_n + \cdots$$

为**无穷级数**，简称**级数**。其中 $a_n$ 称为级数的**通项**或**一般项**。

### 15.1.2 部分和与收敛

**部分和**：级数 $\sum_{n=1}^{\infty} a_n$ 的前 $n$ 项之和

$$S_n = \sum_{k=1}^{n} a_k = a_1 + a_2 + \cdots + a_n$$

称为级数的第 $n$ 个**部分和**。

**收敛与发散**：若部分和数列 $\{S_n\}$ 收敛，即

$$\lim_{n \to \infty} S_n = S$$

存在且有限，则称级数 $\sum_{n=1}^{\infty} a_n$ **收敛**，$S$ 称为级数的**和**，记作

$$\sum_{n=1}^{\infty} a_n = S$$

若 $\{S_n\}$ 发散，则称级数 $\sum_{n=1}^{\infty} a_n$ **发散**。

> **例题 15.1** 讨论几何级数 $\sum_{n=0}^{\infty} q^n = 1 + q + q^2 + \cdots$ 的收敛性。

**解**：部分和为

$$S_n = \sum_{k=0}^{n-1} q^k = \begin{cases} \dfrac{1 - q^n}{1 - q}, & q \neq 1 \\ n, & q = 1 \end{cases}$$

- 当 $|q| < 1$ 时，$\lim_{n \to \infty} q^n = 0$，故 $\lim_{n \to \infty} S_n = \dfrac{1}{1-q}$，级数收敛。
- 当 $|q| \geq 1$ 时，$\{S_n\}$ 发散，级数发散。

因此，几何级数在 $|q| < 1$ 时收敛于 $\dfrac{1}{1-q}$，在 $|q| \geq 1$ 时发散。

### 15.1.3 级数收敛的必要条件

**定理（收敛的必要条件）**：若级数 $\sum_{n=1}^{\infty} a_n$ 收敛，则

$$\lim_{n \to \infty} a_n = 0$$

**证明**：设级数收敛于 $S$，即 $\lim_{n \to \infty} S_n = S$。

由于 $a_n = S_n - S_{n-1}$，故

$$\lim_{n \to \infty} a_n = \lim_{n \to \infty} (S_n - S_{n-1}) = S - S = 0 \quad \square$$

> **注意**：此条件仅是必要条件，不是充分条件。通项趋于零的级数不一定收敛。

> **例题 15.2** 证明调和级数 $\sum_{n=1}^{\infty} \dfrac{1}{n} = 1 + \dfrac{1}{2} + \dfrac{1}{3} + \cdots$ 发散。

**解**：虽然 $\lim_{n \to \infty} \dfrac{1}{n} = 0$，但我们证明部分和无界。

考虑部分和的分组：

$$S_{2^n} = 1 + \frac{1}{2} + \left(\frac{1}{3} + \frac{1}{4}\right) + \left(\frac{1}{5} + \frac{1}{6} + \frac{1}{7} + \frac{1}{8}\right) + \cdots$$

每组的和满足：

$$\frac{1}{3} + \frac{1}{4} > \frac{1}{4} + \frac{1}{4} = \frac{1}{2}$$

$$\frac{1}{5} + \frac{1}{6} + \frac{1}{7} + \frac{1}{8} > 4 \cdot \frac{1}{8} = \frac{1}{2}$$

因此 $S_{2^n} > 1 + \dfrac{n}{2} \to \infty$，调和级数发散。

---

## 15.2 正项级数

设级数 $\sum_{n=1}^{\infty} a_n$ 满足 $a_n \geq 0$，称为**正项级数**。

正项级数的部分和 $\{S_n\}$ 单调递增，故正项级数收敛当且仅当 $\{S_n\}$ 有上界。

### 15.2.1 比较判别法

**定理（比较判别法）**：设 $\sum a_n$ 和 $\sum b_n$ 是正项级数，且存在 $N$，当 $n > N$ 时 $a_n \leq b_n$，则：

1. 若 $\sum b_n$ 收敛，则 $\sum a_n$ 收敛
2. 若 $\sum a_n$ 发散，则 $\sum b_n$ 发散

**比较判别法的极限形式**：设 $\sum a_n$ 和 $\sum b_n$ 是正项级数，$b_n > 0$，且

$$\lim_{n \to \infty} \frac{a_n}{b_n} = l$$

- 若 $0 < l < +\infty$，则 $\sum a_n$ 与 $\sum b_n$ 同敛散
- 若 $l = 0$ 且 $\sum b_n$ 收敛，则 $\sum a_n$ 收敛
- 若 $l = +\infty$ 且 $\sum b_n$ 发散，则 $\sum a_n$ 发散

### 15.2.2 比值判别法（达朗贝尔判别法）

**定理**：设 $\sum a_n$ 是正项级数，若

$$\lim_{n \to \infty} \frac{a_{n+1}}{a_n} = \rho$$

则：
- $\rho < 1$ 时，级数收敛
- $\rho > 1$ 时，级数发散
- $\rho = 1$ 时，判别法失效

> **例题 15.3** 判断级数 $\sum_{n=1}^{\infty} \dfrac{n!}{n^n}$ 的收敛性。

**解**：设 $a_n = \dfrac{n!}{n^n}$，则

$$\frac{a_{n+1}}{a_n} = \frac{(n+1)!}{(n+1)^{n+1}} \cdot \frac{n^n}{n!} = \frac{n+1}{(n+1)^{n+1}} \cdot n^n = \frac{n^n}{(n+1)^n} = \left(\frac{n}{n+1}\right)^n$$

$$\lim_{n \to \infty} \left(\frac{n}{n+1}\right)^n = \lim_{n \to \infty} \frac{1}{\left(1 + \frac{1}{n}\right)^n} = \frac{1}{e} < 1$$

由比值判别法，级数收敛。

### 15.2.3 根值判别法（柯西判别法）

**定理**：设 $\sum a_n$ 是正项级数，若

$$\lim_{n \to \infty} \sqrt[n]{a_n} = \rho$$

则：
- $\rho < 1$ 时，级数收敛
- $\rho > 1$ 时，级数发散
- $\rho = 1$ 时，判别法失效

> **例题 15.4** 判断级数 $\sum_{n=1}^{\infty} \left(\dfrac{n}{2n+1}\right)^n$ 的收敛性。

**解**：设 $a_n = \left(\dfrac{n}{2n+1}\right)^n$，则

$$\sqrt[n]{a_n} = \frac{n}{2n+1} = \frac{1}{2 + \frac{1}{n}} \to \frac{1}{2} < 1$$

由根值判别法，级数收敛。

### 15.2.4 p级数与几何级数

**p级数**：$\sum_{n=1}^{\infty} \dfrac{1}{n^p}$ 在 $p > 1$ 时收敛，在 $p \leq 1$ 时发散。

**几何级数**：$\sum_{n=0}^{\infty} q^n$ 在 $|q| < 1$ 时收敛于 $\dfrac{1}{1-q}$，在 $|q| \geq 1$ 时发散。

这两类级数常作为比较判别法的基准。

---

## 15.3 交错级数

形如

$$\sum_{n=1}^{\infty} (-1)^{n-1} a_n = a_1 - a_2 + a_3 - a_4 + \cdots \quad (a_n > 0)$$

的级数称为**交错级数**。

### 15.3.1 莱布尼茨判别法

**定理（莱布尼茨判别法）**：若交错级数 $\sum_{n=1}^{\infty} (-1)^{n-1} a_n$（$a_n > 0$）满足：

1. $\{a_n\}$ 单调递减：$a_n \geq a_{n+1}$
2. $\lim_{n \to \infty} a_n = 0$

则级数收敛，且其和 $S$ 满足 $0 < S \leq a_1$。

> **例题 15.5** 证明交错调和级数 $\sum_{n=1}^{\infty} \dfrac{(-1)^{n-1}}{n} = 1 - \dfrac{1}{2} + \dfrac{1}{3} - \dfrac{1}{4} + \cdots$ 收敛。

**解**：设 $a_n = \dfrac{1}{n}$，则：

1. $a_n = \dfrac{1}{n} > \dfrac{1}{n+1} = a_{n+1}$，即 $\{a_n\}$ 单调递减
2. $\lim_{n \to \infty} a_n = \lim_{n \to \infty} \dfrac{1}{n} = 0$

由莱布尼茨判别法，交错调和级数收敛。

### 15.3.2 交错级数的余项估计

若交错级数 $\sum_{n=1}^{\infty} (-1)^{n-1} a_n$ 满足莱布尼茨条件，其和为 $S$，则余项

$$R_n = S - S_n = \sum_{k=n+1}^{\infty} (-1)^{k-1} a_k$$

满足 $|R_n| \leq a_{n+1}$，即用 $S_n$ 近似 $S$ 的误差不超过第一个舍去项的绝对值。

---

## 15.4 绝对收敛与条件收敛

### 15.4.1 绝对收敛的定义

**定义**：若级数 $\sum_{n=1}^{\infty} |a_n|$ 收敛，则称级数 $\sum_{n=1}^{\infty} a_n$ **绝对收敛**。

若级数 $\sum_{n=1}^{\infty} a_n$ 收敛但 $\sum_{n=1}^{\infty} |a_n|$ 发散，则称级数 $\sum_{n=1}^{\infty} a_n$ **条件收敛**。

### 15.4.2 绝对收敛与条件收敛的关系

**定理**：绝对收敛的级数必定收敛。

**证明**：设 $\sum |a_n|$ 收敛。令 $b_n = a_n + |a_n|$，则 $0 \leq b_n \leq 2|a_n|$。

由比较判别法，$\sum b_n$ 收敛。

因此 $\sum a_n = \sum b_n - \sum |a_n|$ 收敛。 $\square$

> **例题 15.6** 判断级数 $\sum_{n=1}^{\infty} \dfrac{(-1)^{n-1}}{n}$ 和 $\sum_{n=1}^{\infty} \dfrac{(-1)^{n-1}}{n^2}$ 的收敛类型。

**解**：

对于 $\sum_{n=1}^{\infty} \dfrac{(-1)^{n-1}}{n}$：
- 由莱布尼茨判别法，级数收敛
- $\sum_{n=1}^{\infty} \dfrac{1}{n}$ 是调和级数，发散

故交错调和级数**条件收敛**。

对于 $\sum_{n=1}^{\infty} \dfrac{(-1)^{n-1}}{n^2}$：
- $\sum_{n=1}^{\infty} \dfrac{1}{n^2}$ 是 $p = 2 > 1$ 的 p级数，收敛

故此级数**绝对收敛**。

### 15.4.3 级数重排定理（黎曼定理）

**绝对收敛级数的重排不变性**：若级数 $\sum a_n$ 绝对收敛，则其任意重排后的级数仍收敛于同一个和。

**黎曼定理**：若级数 $\sum a_n$ 条件收敛，则对任意给定的实数 $S$（包括 $\pm\infty$），都存在一种重排方式，使重排后的级数收敛于 $S$。

> **注**：这个定理说明条件收敛级数的和依赖于求和顺序，改变顺序可能改变和的值。

---

## 15.5 级数的运算

### 15.5.1 级数的线性运算

**定理**：若 $\sum a_n = A$，$\sum b_n = B$，$c$ 为常数，则：

1. $\sum (a_n + b_n) = A + B$
2. $\sum c \cdot a_n = c \cdot A$

> **注**：若 $\sum a_n$ 收敛而 $\sum b_n$ 发散，则 $\sum (a_n + b_n)$ 必发散。

### 15.5.2 柯西乘积

设级数 $\sum_{n=0}^{\infty} a_n$ 和 $\sum_{n=0}^{\infty} b_n$，定义它们的**柯西乘积**为级数

$$\sum_{n=0}^{\infty} c_n, \quad \text{其中 } c_n = \sum_{k=0}^{n} a_k b_{n-k} = a_0 b_n + a_1 b_{n-1} + \cdots + a_n b_0$$

**定理（Mertens）**：若 $\sum a_n$ 绝对收敛于 $A$，$\sum b_n$ 收敛于 $B$，则其柯西乘积收敛于 $AB$。

> **例题 15.7** 利用柯西乘积计算 $\left(\sum_{n=0}^{\infty} x^n\right)^2$（$|x| < 1$）。

**解**：设 $a_n = b_n = x^n$，则

$$c_n = \sum_{k=0}^{n} a_k b_{n-k} = \sum_{k=0}^{n} x^k \cdot x^{n-k} = \sum_{k=0}^{n} x^n = (n+1)x^n$$

因此

$$\left(\sum_{n=0}^{\infty} x^n\right)^2 = \sum_{n=0}^{\infty} (n+1)x^n = \frac{1}{(1-x)^2}$$

这与 $\left(\dfrac{1}{1-x}\right)^2 = \dfrac{1}{(1-x)^2}$ 相符。

---

## 本章小结

1. **级数的收敛**通过部分和数列的极限来定义。级数收敛的**必要条件**是通项趋于零，但这不是充分条件。

2. **正项级数**的判别法：
   - **比较判别法**：与已知敛散性的级数比较
   - **比值判别法**：$\lim \dfrac{a_{n+1}}{a_n} < 1$ 收敛，$> 1$ 发散
   - **根值判别法**：$\lim \sqrt[n]{a_n} < 1$ 收敛，$> 1$ 发散

3. **交错级数**满足莱布尼茨条件（单调递减趋于零）时收敛，余项估计 $|R_n| \leq a_{n+1}$。

4. **绝对收敛**级数必收敛，其和与求和顺序无关；**条件收敛**级数的和依赖于求和顺序（黎曼定理）。

5. 收敛级数可进行线性运算，绝对收敛级数可进行柯西乘积运算。

---

## 15.6 深度学习应用

数项级数的理论在现代深度学习中有着深刻的应用，从网络结构设计到训练算法分析，级数的思想无处不在。

### 15.6.1 无限宽度网络与级数

**无限宽度神经网络的级数表示**

当神经网络的隐藏层宽度趋于无穷时，网络的行为可以用级数来精确描述。设网络输出为

$$f(x) = \sum_{n=0}^{\infty} c_n \phi_n(x)$$

其中 $\{\phi_n\}$ 是特征函数族，$c_n$ 是系数。此级数收敛要求 $\sum_{n=0}^{\infty} |c_n|^2 < \infty$，即系数构成平方可和级数。

**Neural Tangent Kernel（NTK）理论**

NTK 理论（Jacot et al., 2018）表明，无限宽度网络在梯度下降训练时等价于一个核回归问题。核函数可展开为级数：

$$K(x, x') = \sum_{n=0}^{\infty} \lambda_n \phi_n(x) \phi_n(x')$$

其中 $\lambda_n \geq 0$ 是核的特征值。级数的收敛性（$\sum \lambda_n < \infty$）保证了核函数的有界性，进而保证训练过程的稳定性。

### 15.6.2 梯度累积与级数求和

**多步梯度累积**

在显存受限的大模型训练中，通常将一个大批量拆分为 $n$ 个小批量，累积梯度：

$$g = \sum_{i=1}^{n} g_i$$

其中 $g_i$ 是第 $i$ 个小批量的梯度。这本质上是一个有限级数求和。若各小批量梯度独立同分布，则 $\|g\| \leq \sum_{i=1}^{n} \|g_i\|$，累积梯度的范数有界。

**动量方法的级数展开**

动量优化算法（Momentum SGD）的更新规则为

$$m_t = \beta m_{t-1} + g_t$$

递推展开得到

$$m_t = \sum_{k=0}^{t} \beta^k g_{t-k}$$

这是一个以 $\beta$ 为公比的指数加权级数。当 $|\beta| < 1$ 时，级数绝对收敛，历史梯度的贡献随时间指数衰减。有效学习步数约为 $\dfrac{1}{1-\beta}$，对应几何级数之和。

### 15.6.3 残差连接的级数视角

**ResNet 的级数结构**

He et al.（2016）提出的残差网络（ResNet）将每一层定义为

$$x_l = x_{l-1} + F_l(x_{l-1})$$

递推展开后，第 $L$ 层的输出为

$$x_L = x_0 + \sum_{l=1}^{L} F_l(x_{l-1})$$

这与 Taylor 级数的结构高度类似：$x_0$ 对应函数值，$\sum F_l$ 对应各阶修正项。残差连接保证了信息的直接传递，类似于级数的"零阶项"始终存在，避免了深层网络的梯度消失问题。

**与 Taylor 级数的类比**

$$f(x) = f(x_0) + f'(x_0)(x - x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \cdots$$

ResNet 中的每个残差块 $F_l$ 可理解为对前一层输出的"高阶修正"，网络深度对应展开阶数，而级数的收敛性对应网络输出的稳定性。

### 15.6.4 级数收敛与训练稳定性

**梯度级数的有界性**

在反向传播过程中，梯度通过链式法则逐层传播。设第 $l$ 层的梯度为 $\delta_l$，则

$$\delta_1 = \prod_{l=1}^{L} W_l^{\top} \cdot \delta_L$$

若每层权重矩阵的谱范数 $\|W_l\|_2 = r_l$，则梯度范数满足

$$\|\delta_1\| \leq \prod_{l=1}^{L} r_l \cdot \|\delta_L\|$$

当 $r_l > 1$ 时，乘积 $\prod r_l$ 如同发散级数；当 $r_l < 1$ 时，乘积趋于零，导致梯度消失。这与级数收敛的条件直接对应。

**梯度裁剪保证收敛**

梯度裁剪（Gradient Clipping）通过限制梯度的 $L^2$ 范数来控制更新步长：

$$\tilde{g} = \begin{cases} g, & \|g\|_2 \leq c \\ \dfrac{c}{\|g\|_2} g, & \|g\|_2 > c \end{cases}$$

这保证了参数更新量构成的级数 $\sum_{t=1}^{\infty} \|\Delta\theta_t\|$ 有上界，从而保证训练过程中参数序列的收敛性。

### 15.6.5 代码示例

以下代码展示了 ResNet 的级数结构，以及梯度裁剪在训练中的应用：

```python
import torch
import torch.nn as nn

# 残差网络的级数结构
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.fc(x)  # x + F(x)

# ResNet = x_0 + Σ F_l(x)
class ResNet(nn.Module):
    def __init__(self, dim, n_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([ResBlock(dim) for _ in range(n_blocks)])

    def forward(self, x):
        # 级数求和: x_L = x_0 + F_1 + F_2 + ... + F_L
        for block in self.blocks:
            x = block(x)
        return x

# 梯度裁剪：保证级数收敛
def train_with_gradient_clipping(model, optimizer, loss_fn, x, y, max_norm=1.0):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()

    # 梯度裁剪：限制梯度范数
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    return loss.item()
```

**代码说明**：
- `ResBlock.forward` 中的 `x + self.fc(x)` 对应级数中的一项 $x_{l-1} + F_l(x_{l-1})$
- `ResNet.forward` 中的循环对应级数累加 $x_L = x_0 + \sum_{l=1}^{L} F_l(x_{l-1})$
- `clip_grad_norm_` 将梯度投影到半径为 `max_norm` 的球内，保证每步更新有界

---

## 练习题

**1.** ⭐ 判断级数 $\sum_{n=1}^{\infty} \dfrac{1}{n(n+1)}$ 的收敛性，若收敛求其和。

**2.** ⭐⭐ 用比值判别法判断级数 $\sum_{n=1}^{\infty} \dfrac{n^2}{2^n}$ 的收敛性。

**3.** ⭐⭐ 判断级数 $\sum_{n=1}^{\infty} \dfrac{(-1)^{n-1}}{\sqrt{n}}$ 是绝对收敛、条件收敛还是发散。

**4.** ⭐⭐ 用根值判别法判断级数 $\sum_{n=1}^{\infty} \left(\dfrac{n+1}{3n}\right)^n$ 的收敛性。

**5.** ⭐⭐⭐ 设 $\sum_{n=1}^{\infty} a_n$ 绝对收敛，证明 $\sum_{n=1}^{\infty} a_n^2$ 也收敛。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.** 利用部分分式分解：

$$\frac{1}{n(n+1)} = \frac{1}{n} - \frac{1}{n+1}$$

部分和为

$$S_n = \sum_{k=1}^{n} \left(\frac{1}{k} - \frac{1}{k+1}\right) = 1 - \frac{1}{n+1}$$

因此 $\lim_{n \to \infty} S_n = 1$，级数收敛，和为 $1$。

---

**2.** 设 $a_n = \dfrac{n^2}{2^n}$，则

$$\frac{a_{n+1}}{a_n} = \frac{(n+1)^2}{2^{n+1}} \cdot \frac{2^n}{n^2} = \frac{1}{2} \cdot \frac{(n+1)^2}{n^2} = \frac{1}{2}\left(1 + \frac{1}{n}\right)^2$$

$$\lim_{n \to \infty} \frac{a_{n+1}}{a_n} = \frac{1}{2} \cdot 1 = \frac{1}{2} < 1$$

由比值判别法，级数收敛。

---

**3.** 设 $a_n = \dfrac{1}{\sqrt{n}}$，原级数为交错级数 $\sum_{n=1}^{\infty} (-1)^{n-1} a_n$。

- $a_n = \dfrac{1}{\sqrt{n}}$ 单调递减且 $\lim_{n \to \infty} a_n = 0$，由莱布尼茨判别法，级数收敛。

- $\sum_{n=1}^{\infty} \dfrac{1}{\sqrt{n}} = \sum_{n=1}^{\infty} \dfrac{1}{n^{1/2}}$ 是 $p = \dfrac{1}{2} < 1$ 的 p级数，发散。

因此，原级数**条件收敛**。

---

**4.** 设 $a_n = \left(\dfrac{n+1}{3n}\right)^n$，则

$$\sqrt[n]{a_n} = \frac{n+1}{3n} = \frac{1}{3}\left(1 + \frac{1}{n}\right) \to \frac{1}{3} < 1$$

由根值判别法，级数收敛。

---

**5.** **证明**：由于 $\sum a_n$ 绝对收敛，即 $\sum |a_n|$ 收敛。

由收敛的必要条件，$\lim_{n \to \infty} |a_n| = 0$。

因此存在 $N$，当 $n > N$ 时，$|a_n| < 1$，从而 $a_n^2 = |a_n|^2 < |a_n|$。

由比较判别法，$\sum_{n=N+1}^{\infty} a_n^2$ 收敛。

加上有限项 $\sum_{n=1}^{N} a_n^2$ 不影响收敛性，故 $\sum_{n=1}^{\infty} a_n^2$ 收敛。 $\square$

</details>
