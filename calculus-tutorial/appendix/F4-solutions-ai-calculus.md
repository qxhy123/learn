# 附录 F4：AI 微积分详解（C.74-C.80, D.91-D.100, E.53-E.60）

> 本附录对应教材 **Ch.25–28（AI 微积分）** 的全部 25 道习题（C 级 7 题 + D 级 10 题 + E 级 8 题），涵盖凸优化、矩阵微积分、概率统计、随机过程四大模块。  
> 格式：**题目回顾 → 思路 → 详细解答 → 答案框 → 总结**。  
> 难度标记：[基础] C 级 · [中档] D 级 · [提升] E 级。

---

## 第一部分：C 级基础题（C.74–C.80）

### C.74 [基础] Ch.25 — 二次函数的凸性

**题目回顾**  
判断 $f(x)=x^2$ 的凸性：计算 $f''(x)$，说明 $f$ 是凸函数（$f''\ge 0$）。

**思路**  
凸函数的二阶条件：对一元函数，若 $f''(x)\ge 0$ 恒成立，则 $f$ 是凸函数。

**解答**  
计算导数：
$$f'(x)=2x,\quad f''(x)=2.$$

因为 $f''(x)=2>0$ 对所有 $x\in\mathbb{R}$ 成立，所以 $f$ 是**严格凸函数**。

几何上，$y=x^2$ 是向上开口的抛物线，任意两点间的弦位于曲线上方，符合凸函数定义 $f(\lambda x+(1-\lambda)y)\le\lambda f(x)+(1-\lambda)f(y)$（$\lambda\in[0,1]$）。

> **答案：** $f''(x)=2>0$，$f(x)=x^2$ 是严格凸函数。

**总结**  
二阶条件是判断凸性最直接的方法。在机器学习中，损失函数的凸性保证梯度下降能收敛到全局最小值。

---

### C.75 [基础] Ch.25 — 梯度下降迭代公式

**题目回顾**  
写出梯度下降的一步迭代公式 $\theta\leftarrow\theta-\eta\,\nabla_\theta L(\theta)$，并解释各符号含义。

**思路**  
梯度下降是最基础的一阶优化算法，沿负梯度方向更新参数。

**解答**  
迭代公式：
$$\theta^{(t+1)}=\theta^{(t)}-\eta\,\nabla_\theta L(\theta^{(t)}).$$

各符号含义：
- $\theta\in\mathbb{R}^n$：模型参数向量（权重、偏置等）；
- $L(\theta)$：损失函数（如均方误差、交叉熵），衡量预测与真实标签的偏差；
- $\nabla_\theta L(\theta)$：$L$ 关于 $\theta$ 的梯度，指向损失函数增长最快的方向；
- $\eta>0$：**学习率**（步长），控制每次更新的幅度。

**收敛保证**（简要）：若 $L$ 是 $\beta$-光滑凸函数，取 $\eta\le1/\beta$，则梯度下降以 $O(1/T)$ 速度收敛到最优值；若 $L$ 还是 $\mu$-强凸，则以线性（指数）速度收敛。

> **答案：** $\theta^{(t+1)}=\theta^{(t)}-\eta\,\nabla_\theta L(\theta^{(t)})$，$\eta$ 为学习率，$L$ 为损失函数，$\nabla_\theta L$ 为参数梯度。

**总结**  
梯度下降是深度学习优化的核心。学习率 $\eta$ 的选取至关重要：过大导致发散，过小收敛缓慢。Adam、RMSProp 等自适应方法本质上是对梯度的自适应缩放。

---

### C.76 [基础] Ch.26 — 线性函数的梯度

**题目回顾**  
设 $f(\mathbf{x})=\mathbf{a}^\top\mathbf{x}$（$\mathbf{a},\mathbf{x}\in\mathbb{R}^n$），写出 $\dfrac{\partial f}{\partial \mathbf{x}}$。

**思路**  
对向量求导，逐分量分析。$f=\sum_{i=1}^n a_i x_i$，对 $x_j$ 求偏导得 $a_j$。

**解答**  
展开写出 $f=a_1x_1+a_2x_2+\cdots+a_nx_n$。

对第 $j$ 个分量求偏导：
$$\frac{\partial f}{\partial x_j}=a_j,\quad j=1,2,\ldots,n.$$

整理成向量形式：
$$\frac{\partial f}{\partial \mathbf{x}}=\mathbf{a}.$$

> **答案：** $\dfrac{\partial(\mathbf{a}^\top\mathbf{x})}{\partial\mathbf{x}}=\mathbf{a}$。

**总结**  
线性函数 $\mathbf{a}^\top\mathbf{x}$ 的梯度就是系数向量 $\mathbf{a}$ 本身，这是矩阵微积分最基础的公式，广泛用于线性回归、感知机等模型的梯度推导。

---

### C.77 [基础] Ch.26 — 二次型的梯度

**题目回顾**  
设 $f(\mathbf{x})=\mathbf{x}^\top A\mathbf{x}$（$A$ 为 $n\times n$ 对称矩阵），写出 $\dfrac{\partial f}{\partial\mathbf{x}}$。

**思路**  
展开二次型，逐分量求偏导，利用 $A$ 的对称性化简。

**解答**  
展开：$f=\sum_{i,j}a_{ij}x_ix_j$。

对 $x_k$ 求偏导：
$$\frac{\partial f}{\partial x_k}=\sum_j a_{kj}x_j+\sum_i a_{ik}x_i.$$

由 $A$ 对称（$a_{ij}=a_{ji}$），两项合并：
$$\frac{\partial f}{\partial x_k}=2\sum_j a_{kj}x_j=2(A\mathbf{x})_k.$$

向量形式：
$$\boxed{\frac{\partial(\mathbf{x}^\top A\mathbf{x})}{\partial\mathbf{x}}=2A\mathbf{x}.}$$

> **答案：** $\dfrac{\partial f}{\partial\mathbf{x}}=2A\mathbf{x}$（$A$ 对称时）。

**总结**  
二次型梯度公式 $\nabla(\mathbf{x}^\top A\mathbf{x})=2A\mathbf{x}$ 是正规方程、PCA 等的基础。若 $A$ 不对称，结果为 $(A+A^\top)\mathbf{x}$。

---

### C.78 [基础] Ch.27 — 正态分布的矩

**题目回顾**  
设 $X\sim\mathcal{N}(\mu,\sigma^2)$，写出 $E[X]$、$\mathrm{Var}(X)$ 与 $E[X^2]$。

**思路**  
利用正态分布的定义及方差与期望的关系 $E[X^2]=\mathrm{Var}(X)+(E[X])^2$。

**解答**  
由正态分布定义：
$$E[X]=\mu,\quad\mathrm{Var}(X)=\sigma^2.$$

利用方差公式 $\mathrm{Var}(X)=E[X^2]-(E[X])^2$，解出：
$$E[X^2]=\mathrm{Var}(X)+(E[X])^2=\sigma^2+\mu^2.$$

> **答案：** $E[X]=\mu$，$\mathrm{Var}(X)=\sigma^2$，$E[X^2]=\mu^2+\sigma^2$。

**总结**  
正态分布的矩公式在贝叶斯推断、变分自编码器（VAE）等模型中频繁使用。$E[X^2]=\mu^2+\sigma^2$ 体现了方差的"能量分解"含义。

---

### C.79 [基础] Ch.27 — 独立随机变量的期望与方差

**题目回顾**  
设 $X,Y$ 独立，$E[X]=\mu_X$，$E[Y]=\mu_Y$，写出 $E[XY]$ 与 $\mathrm{Var}(X+Y)$。

**思路**  
独立性意味着期望可乘、协方差为零。

**解答**  
**期望相乘：** 由独立性，$X$ 与 $Y$ 的联合分布等于各自分布之积，故
$$E[XY]=E[X]\cdot E[Y]=\mu_X\mu_Y.$$

**方差可加：** 独立时 $\mathrm{Cov}(X,Y)=0$，故
$$\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y).$$

> **答案：** $E[XY]=\mu_X\mu_Y$，$\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)$。

**总结**  
方差可加性是中心极限定理成立的重要前提。在 dropout 正则化分析中，独立掩码的方差可加性帮助分析期望激活值。

---

### C.80 [基础] Ch.28 — 标准布朗运动的基本性质

**题目回顾**  
写出标准布朗运动 $W_t$ 的三条基本性质，并写出 $E[W_t^2]$。

**思路**  
布朗运动的定义性质，直接列举并计算。

**解答**  
标准布朗运动（Wiener 过程）$\{W_t\}_{t\ge0}$ 满足：

1. **初始条件：** $W_0=0$（a.s.）；
2. **增量分布：** 对 $t>s\ge0$，$W_t-W_s\sim\mathcal{N}(0,t-s)$（增量服从正态分布，方差等于时间间隔）；
3. **独立增量：** 对任意 $0\le t_0<t_1<\cdots<t_k$，$W_{t_1}-W_{t_0},\,W_{t_2}-W_{t_1},\,\ldots,\,W_{t_k}-W_{t_{k-1}}$ 相互独立。

由性质 2（取 $s=0$）：$W_t\sim\mathcal{N}(0,t)$，故
$$E[W_t^2]=\mathrm{Var}(W_t)+(E[W_t])^2=t+0=t.$$

> **答案：** 三条性质如上，$E[W_t^2]=t$。

**总结**  
布朗运动是随机微积分的基础。性质"增量方差等于时间间隔"是金融数学中波动率建模的起点；$E[W_t^2]=t$ 也是 Itô 等距公式的直接体现。

---

## 第二部分：D 级中档题（D.91–D.100）

### D.91 [中档] Ch.25 — 凸函数全局最小值点的证明

**题目回顾**  
设 $f:\mathbb{R}^n\to\mathbb{R}$ 是凸函数，证明：若 $\nabla f(\mathbf{x}^*)=\mathbf{0}$，则 $\mathbf{x}^*$ 是全局最小值点。

**思路**  
利用凸函数的一阶充要条件：$f(\mathbf{y})\ge f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})$，将 $\nabla f(\mathbf{x}^*)=\mathbf{0}$ 代入即得。

**解答**  
由凸函数的一阶条件（该条件等价于凸性），对任意 $\mathbf{y}\in\mathbb{R}^n$：
$$f(\mathbf{y})\ge f(\mathbf{x}^*)+\nabla f(\mathbf{x}^*)^\top(\mathbf{y}-\mathbf{x}^*).$$

代入 $\nabla f(\mathbf{x}^*)=\mathbf{0}$：
$$f(\mathbf{y})\ge f(\mathbf{x}^*)+\mathbf{0}^\top(\mathbf{y}-\mathbf{x}^*)=f(\mathbf{x}^*).$$

由 $\mathbf{y}$ 的任意性，$\mathbf{x}^*$ 是 $f$ 的全局最小值点。$\square$

> **答案：** 由凸函数一阶条件，梯度为零的点满足 $f(\mathbf{y})\ge f(\mathbf{x}^*)$ 对所有 $\mathbf{y}$ 成立，故为全局最优。

**总结**  
这是凸优化最核心的定理：对凸函数，局部最优 = 全局最优；梯度为零是充要条件（注意非凸时这仅是必要条件）。神经网络损失函数的非凸性使得寻找全局最优变得极难。

---

### D.92 [中档] Ch.25 — 二元函数的强凸性

**题目回顾**  
判断 $f(x,y)=x^2+xy+y^2$ 的凸性：计算 Hessian 矩阵的特征值，说明 $f$ 是强凸函数。

**思路**  
计算 Hessian，验证正定性（所有特征值 $>0$）。

**解答**  
计算各阶偏导：
$$f_{xx}=2,\quad f_{xy}=f_{yx}=1,\quad f_{yy}=2.$$

Hessian 矩阵：
$$H=\begin{pmatrix}2&1\\1&2\end{pmatrix}.$$

计算特征值：$\det(H-\lambda I)=(2-\lambda)^2-1=0$，即 $\lambda^2-4\lambda+3=0$，解得
$$\lambda_1=1,\quad\lambda_2=3.$$

由于 $\lambda_1=1>0$，$\lambda_2=3>0$，所以 $H\succ 0$（正定）。

这意味着 $f$ 是**强凸函数**，强凸模数 $\mu=\lambda_{\min}(H)=1$（即满足 $f(\mathbf{y})\ge f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})+\frac{1}{2}\|\mathbf{y}-\mathbf{x}\|^2$）。

**验证（主子式判别）：** $f_{xx}=2>0$，$\det H=4-1=3>0$，Sylvester 准则亦确认正定。

> **答案：** $H$ 特征值为 $1$ 和 $3$，均为正，$f$ 是强凸函数，强凸模数 $\mu=1$。

**总结**  
强凸保证梯度下降线性收敛（指数速度），条件数 $\kappa=\lambda_{\max}/\lambda_{\min}=3$ 决定收敛速度。Lasso、Ridge 正则化常利用强凸性来保证唯一解。

---

### D.93 [中档] Ch.26 — 最小二乘的梯度与正规方程

**题目回顾**  
设 $f(\mathbf{x})=\|\mathbf{A}\mathbf{x}-\mathbf{b}\|^2$，计算 $\nabla_\mathbf{x}f$，写出最优性条件（正规方程）。

**思路**  
展开平方范数为二次型，逐步利用矩阵微积分公式。

**解答**  
展开：
$$f(\mathbf{x})=(\mathbf{A}\mathbf{x}-\mathbf{b})^\top(\mathbf{A}\mathbf{x}-\mathbf{b})=\mathbf{x}^\top A^\top A\mathbf{x}-2\mathbf{b}^\top A\mathbf{x}+\mathbf{b}^\top\mathbf{b}.$$

逐项对 $\mathbf{x}$ 求梯度：
- $\nabla_\mathbf{x}(\mathbf{x}^\top A^\top A\mathbf{x})=2A^\top A\mathbf{x}$（$A^\top A$ 对称，用二次型公式）；
- $\nabla_\mathbf{x}(-2\mathbf{b}^\top A\mathbf{x})=-2A^\top\mathbf{b}$（线性型公式）；
- $\nabla_\mathbf{x}(\mathbf{b}^\top\mathbf{b})=\mathbf{0}$（常数）。

合并：
$$\nabla_\mathbf{x}f=2A^\top A\mathbf{x}-2A^\top\mathbf{b}=2A^\top(A\mathbf{x}-\mathbf{b}).$$

令梯度为零，得**正规方程**：
$$\boxed{A^\top A\mathbf{x}=A^\top\mathbf{b}.}$$

若 $A^\top A$ 可逆（即 $A$ 列满秩），唯一解为 $\mathbf{x}^*=(A^\top A)^{-1}A^\top\mathbf{b}$（最小二乘解）。

> **答案：** $\nabla f=2A^\top(A\mathbf{x}-\mathbf{b})$，最优性条件为 $A^\top A\mathbf{x}=A^\top\mathbf{b}$。

**总结**  
正规方程是线性回归闭合解的来源。当 $A$ 列数远大于行数时，$A^\top A$ 奇异，需用 Moore-Penrose 伪逆或正则化（Ridge：$(A^\top A+\lambda I)\mathbf{x}=A^\top\mathbf{b}$）。

---

### D.94 [中档] Ch.26 — Sigmoid 函数的矩阵梯度

**题目回顾**  
设 $f(\mathbf{w})=\sigma(\mathbf{w}^\top\mathbf{x})$，$\sigma(t)=\dfrac{1}{1+e^{-t}}$。计算 $\dfrac{\partial f}{\partial\mathbf{w}}$。

**思路**  
链式法则分两步：先对 $t=\mathbf{w}^\top\mathbf{x}$ 求导，再对 $\mathbf{w}$ 求导。利用 $\sigma'(t)=\sigma(t)(1-\sigma(t))$。

**解答**  
令 $t=\mathbf{w}^\top\mathbf{x}$，则 $f=\sigma(t)$。

**第一步：** $\sigma$ 对 $t$ 的导数。

$$\sigma'(t)=\frac{e^{-t}}{(1+e^{-t})^2}=\sigma(t)(1-\sigma(t)).$$

验证：$\sigma'(t)=\frac{d}{dt}\frac{1}{1+e^{-t}}=\frac{e^{-t}}{(1+e^{-t})^2}=\frac{1}{1+e^{-t}}\cdot\frac{e^{-t}}{1+e^{-t}}=\sigma(t)(1-\sigma(t))$。

**第二步：** $t=\mathbf{w}^\top\mathbf{x}$ 对 $\mathbf{w}$ 的梯度为 $\mathbf{x}$（由 C.76）。

**链式法则合并：**
$$\frac{\partial f}{\partial\mathbf{w}}=\sigma'(t)\cdot\frac{\partial t}{\partial\mathbf{w}}=\sigma(\mathbf{w}^\top\mathbf{x})\bigl(1-\sigma(\mathbf{w}^\top\mathbf{x})\bigr)\mathbf{x}.$$

> **答案：** $\dfrac{\partial f}{\partial\mathbf{w}}=\sigma(\mathbf{w}^\top\mathbf{x})(1-\sigma(\mathbf{w}^\top\mathbf{x}))\,\mathbf{x}$。

**总结**  
$\sigma'=\sigma(1-\sigma)$ 是 sigmoid 最重要的性质，使得反向传播只需存储前向计算值 $\sigma$ 即可还原梯度。注意当 $|\mathbf{w}^\top\mathbf{x}|$ 很大时 $\sigma'\approx0$，即"梯度消失"问题。

---

### D.95 [中档] Ch.26 — 交叉熵对 logit 的梯度

**题目回顾**  
设损失 $L=-\mathbf{y}^\top\log(\mathbf{p})$，$\mathbf{p}=\mathrm{softmax}(\mathbf{z})$，计算 $\dfrac{\partial L}{\partial z_i}$。

**思路**  
复合函数：先对 $\mathbf{p}$ 求偏导，再对 $z_i$ 求偏导（利用 softmax 的 Jacobian），两者相乘。

**解答**  
**第一步：** $L$ 对 $p_k$ 的偏导。
$$\frac{\partial L}{\partial p_k}=-\frac{y_k}{p_k}.$$

**第二步：** softmax 的 Jacobian（E.53 中详细推导）。
$$\frac{\partial p_k}{\partial z_i}=p_k(\delta_{ki}-p_i).$$

**第三步：** 链式法则。
$$\frac{\partial L}{\partial z_i}=\sum_k\frac{\partial L}{\partial p_k}\cdot\frac{\partial p_k}{\partial z_i}=\sum_k\left(-\frac{y_k}{p_k}\right)p_k(\delta_{ki}-p_i).$$

展开：
$$=\sum_k(-y_k)(\delta_{ki}-p_i)=-y_i+p_i\sum_k y_k.$$

由于 $\mathbf{y}$ 是 one-hot 向量（$\sum_k y_k=1$），故
$$\frac{\partial L}{\partial z_i}=p_i-y_i.$$

> **答案：** $\dfrac{\partial L}{\partial z_i}=p_i-y_i$，即预测概率与真实标签的差。

**总结**  
$\nabla_\mathbf{z}L=\mathbf{p}-\mathbf{y}$ 是分类任务反向传播中最优美的结果之一——梯度正好是预测值与真实值之差，形式简洁且数值稳定（避免了直接对 $\log p_i$ 求导时的数值问题）。

---

### D.96 [中档] Ch.27 — 标准正态分布的四阶矩

**题目回顾**  
设 $X\sim\mathcal{N}(0,1)$，计算 $E[X^4]$。

**思路**  
利用正态分布高阶矩公式 $E[X^{2k}]=(2k-1)!!$，或用矩母函数 / 分部积分推导。

**解答**  
**方法一（公式法）：** 标准正态的偶数阶矩满足 $E[X^{2k}]=(2k-1)!!=(2k-1)(2k-3)\cdots3\cdot1$。取 $k=2$：
$$E[X^4]=(2\times2-1)!!=3!!=3\times1=3.$$

**方法二（分部积分验证）：** 利用递推关系 $E[X^{2k}]=(2k-1)E[X^{2k-2}]$（通过分部积分对密度函数积分得到）。

已知 $E[X^2]=\mathrm{Var}(X)=1$，故
$$E[X^4]=3\cdot E[X^2]=3\times1=3.$$

**方法三（矩母函数）：** $M_X(t)=e^{t^2/2}$，展开：
$$M_X(t)=\sum_{k=0}^\infty\frac{E[X^k]}{k!}t^k=1+\frac{t^2}{2}+\frac{t^4}{8}+\cdots$$

$E[X^4]/4!=1/8$，故 $E[X^4]=3$。

> **答案：** $E[X^4]=3$。

**总结**  
正态分布的矩公式 $(2k-1)!!$ 来源于其"充分统计量"结构。在深度学习的批归一化（Batch Normalization）分析中，四阶矩控制激活值分布的"峰度"，是稳定训练的重要指标。

---

### D.97 [中档] Ch.27 — 正态分布的卷积性（矩母函数法）

**题目回顾**  
设 $X\sim\mathcal{N}(\mu_1,\sigma_1^2)$，$Y\sim\mathcal{N}(\mu_2,\sigma_2^2)$，$X,Y$ 独立。证明 $Z=X+Y\sim\mathcal{N}(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$。

**思路**  
使用矩母函数（MGF）法：独立时 $M_{X+Y}(t)=M_X(t)\cdot M_Y(t)$，验证乘积仍是正态的 MGF。

**解答**  
**正态分布的矩母函数：** 设 $X\sim\mathcal{N}(\mu,\sigma^2)$，则
$$M_X(t)=E[e^{tX}]=\exp\!\left(\mu t+\frac{\sigma^2t^2}{2}\right).$$

（推导：$E[e^{tX}]=\int_{-\infty}^{+\infty}e^{tx}\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/(2\sigma^2)}dx$，配方后积分。）

**利用独立性：**
$$M_Z(t)=M_{X+Y}(t)=M_X(t)\cdot M_Y(t)=\exp\!\left(\mu_1t+\frac{\sigma_1^2t^2}{2}\right)\cdot\exp\!\left(\mu_2t+\frac{\sigma_2^2t^2}{2}\right).$$

合并指数：
$$M_Z(t)=\exp\!\left((\mu_1+\mu_2)t+\frac{(\sigma_1^2+\sigma_2^2)t^2}{2}\right).$$

这正是 $\mathcal{N}(\mu_1+\mu_2,\,\sigma_1^2+\sigma_2^2)$ 的矩母函数。由矩母函数唯一确定分布，故
$$Z=X+Y\sim\mathcal{N}(\mu_1+\mu_2,\,\sigma_1^2+\sigma_2^2).\quad\square$$

> **答案：** $X+Y\sim\mathcal{N}(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$，由矩母函数相乘验证。

**总结**  
正态分布对加法的封闭性（卷积稳定性）是中心极限定理的直接推论。在神经网络中，多层线性变换的输出若每层加高斯噪声，最终输出仍是高斯分布。

---

### D.98 [中档] Ch.27 — 同方差正态分布的 KL 散度

**题目回顾**  
计算 $D_\mathrm{KL}(p\|q)$，其中 $p=\mathcal{N}(\mu_1,\sigma^2)$，$q=\mathcal{N}(\mu_2,\sigma^2)$（同方差）。

**思路**  
代入 KL 散度定义，利用正态分布密度的对数展开，化简积分。

**解答**  
两者密度分别为：
$$p(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\!\left(-\frac{(x-\mu_1)^2}{2\sigma^2}\right),\quad q(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\!\left(-\frac{(x-\mu_2)^2}{2\sigma^2}\right).$$

KL 散度：
$$D_\mathrm{KL}(p\|q)=E_p\!\left[\ln\frac{p(X)}{q(X)}\right]=E_p\!\left[\ln p(X)-\ln q(X)\right].$$

计算 $\ln p(x)-\ln q(x)$：
$$\ln p-\ln q=-\frac{(x-\mu_1)^2}{2\sigma^2}+\frac{(x-\mu_2)^2}{2\sigma^2}=\frac{(x-\mu_2)^2-(x-\mu_1)^2}{2\sigma^2}.$$

展开分子：$(x-\mu_2)^2-(x-\mu_1)^2=2x(\mu_1-\mu_2)+(\mu_2^2-\mu_1^2)=2(\mu_1-\mu_2)(x-\tfrac{\mu_1+\mu_2}{2})$。

取期望（$X\sim p$，$E_p[X]=\mu_1$）：
$$D_\mathrm{KL}(p\|q)=E_p\!\left[\frac{(x-\mu_2)^2-(x-\mu_1)^2}{2\sigma^2}\right]=\frac{(\mu_1-\mu_2)^2}{2\sigma^2}.$$

（详细展开：$E_p[(X-\mu_2)^2]=\mathrm{Var}_p(X)+(E_p[X]-\mu_2)^2=\sigma^2+(\mu_1-\mu_2)^2$，$E_p[(X-\mu_1)^2]=\sigma^2$，差为 $(\mu_1-\mu_2)^2$。）

> **答案：** $D_\mathrm{KL}(\mathcal{N}(\mu_1,\sigma^2)\|\mathcal{N}(\mu_2,\sigma^2))=\dfrac{(\mu_1-\mu_2)^2}{2\sigma^2}$。

**总结**  
同方差情形下，KL 散度只依赖均值之差，与马氏距离一致。这是变分推断（VAE）中 KL 正则项的特例：KL 散度量化了近似后验与先验之间的"距离"。

---

### D.99 [中档] Ch.28 — 布朗运动的二阶矩与协方差

**题目回顾**  
计算 $E[W_t^2]$，$E[W_sW_t]$（$s<t$），$\mathrm{Var}(W_t-W_s)$。

**思路**  
利用布朗运动性质：增量独立、$W_t-W_s\sim\mathcal{N}(0,t-s)$，以及 $W_s$ 与 $W_t-W_s$ 独立。

**解答**  
**① $E[W_t^2]$：** 由 $W_t\sim\mathcal{N}(0,t)$，
$$E[W_t^2]=\mathrm{Var}(W_t)=t.$$

**② $E[W_sW_t]$（$s<t$）：** 分解 $W_t=W_s+(W_t-W_s)$，其中 $W_s$ 与增量 $W_t-W_s$ 独立，且 $E[W_t-W_s]=0$：
$$E[W_sW_t]=E[W_s(W_s+(W_t-W_s))]=E[W_s^2]+E[W_s]\cdot E[W_t-W_s]=s+0=s.$$

（一般情况：$E[W_sW_t]=\min(s,t)$，这是布朗运动的协方差函数。）

**③ $\mathrm{Var}(W_t-W_s)$：** 由 $W_t-W_s\sim\mathcal{N}(0,t-s)$，
$$\mathrm{Var}(W_t-W_s)=t-s.$$

> **答案：** $E[W_t^2]=t$，$E[W_sW_t]=s$（$s<t$），$\mathrm{Var}(W_t-W_s)=t-s$。

**总结**  
协方差 $E[W_sW_t]=\min(s,t)$ 完全刻画了布朗运动的二阶统计性质。这是高斯过程（GP）核函数设计的原型，在时间序列建模和贝叶斯非参数中广泛使用。

---

### D.100 [中档] Ch.28 — Itô 过程的二阶矩

**题目回顾**  
设 $dX_t=\mu\,dt+\sigma\,dW_t$（$X_0=0$，$\mu,\sigma$ 为常数）。用 Itô 公式计算 $d(X_t^2)$，并求 $E[X_t^2]$。

**思路**  
对 $f(x)=x^2$ 应用 Itô 公式 $df=f'\,dX+\tfrac12f''\sigma^2\,dt$，积分后取期望（随机积分期望为零）。

**解答**  
**第一步：应用 Itô 公式。** 取 $f(x)=x^2$，则 $f'(x)=2x$，$f''(x)=2$，扩散系数为 $\sigma$：
$$d(X_t^2)=2X_t\,dX_t+\frac{1}{2}\cdot 2\cdot\sigma^2\,dt=2X_t(\mu\,dt+\sigma\,dW_t)+\sigma^2\,dt.$$

整理：
$$d(X_t^2)=(2\mu X_t+\sigma^2)\,dt+2\sigma X_t\,dW_t.$$

关键：经典微积分中 $d(x^2)=2x\,dx$，Itô 公式多出 $\sigma^2\,dt$ 项（来自 $(dW_t)^2=dt$）。

**第二步：对两边积分。**
$$X_t^2=\int_0^t(2\mu X_s+\sigma^2)\,ds+2\sigma\int_0^tX_s\,dW_s.$$

**第三步：取期望。** 随机积分 $\int_0^tX_s\,dW_s$ 是鞅，期望为零。又 $X_t=\mu t+\sigma W_t$，故 $E[X_t]=\mu t$：
$$E[X_t^2]=\int_0^t(2\mu E[X_s]+\sigma^2)\,ds=\int_0^t(2\mu\cdot\mu s+\sigma^2)\,ds=\int_0^t(2\mu^2 s+\sigma^2)\,ds.$$

$$=\mu^2t^2+\sigma^2t.$$

**验证（直接法）：** $X_t=\mu t+\sigma W_t$，故 $E[X_t^2]=(\mu t)^2+\sigma^2E[W_t^2]+2\mu t\sigma E[W_t]=\mu^2t^2+\sigma^2t$（与上述一致）。

> **答案：** $d(X_t^2)=(2\mu X_t+\sigma^2)\,dt+2\sigma X_t\,dW_t$，$E[X_t^2]=\mu^2t^2+\sigma^2t$。

**总结**  
Itô 公式是随机微积分中的"链式法则"，核心修正项 $\tfrac12f''\sigma^2\,dt$ 来自 $(dW_t)^2=dt\ne0$。在金融数学中，$dX_t=\mu X_t\,dt+\sigma X_t\,dW_t$ 给出 Black-Scholes 模型的基础。

---

## 第三部分：E 级提升题（E.53–E.60）

### E.53 [提升] Ch.26 — Softmax Jacobian 矩阵

**题目回顾**  
设 $\sigma_i=\dfrac{e^{z_i}}{\sum_j e^{z_j}}$。推导 $\dfrac{\partial\sigma_i}{\partial z_k}=\sigma_i(\delta_{ik}-\sigma_k)$，写成矩阵形式并验证半负定性。

**思路**  
商法则分两情形（$i=k$ 和 $i\ne k$），将结果统一写成含 Kronecker $\delta$ 的形式，再验证 Jacobian 的正定性结构。

**解答**  
**第一步：商法则推导偏导。** 记 $S=\sum_j e^{z_j}$，则 $\sigma_i=e^{z_i}/S$。

**情形 1（$i=k$）：**
$$\frac{\partial\sigma_i}{\partial z_i}=\frac{e^{z_i}\cdot S-e^{z_i}\cdot e^{z_i}}{S^2}=\frac{e^{z_i}}{S}\left(1-\frac{e^{z_i}}{S}\right)=\sigma_i(1-\sigma_i).$$

**情形 2（$i\ne k$）：**
$$\frac{\partial\sigma_i}{\partial z_k}=\frac{0\cdot S-e^{z_i}\cdot e^{z_k}}{S^2}=-\frac{e^{z_i}}{S}\cdot\frac{e^{z_k}}{S}=-\sigma_i\sigma_k.$$

**统一写法：** 引入 Kronecker $\delta_{ik}$（$i=k$ 时为 $1$，否则为 $0$）：
$$\frac{\partial\sigma_i}{\partial z_k}=\sigma_i(\delta_{ik}-\sigma_k).$$

**第二步：矩阵形式。** Jacobian $J_\sigma\in\mathbb{R}^{n\times n}$，$(J_\sigma)_{ik}=\sigma_i(\delta_{ik}-\sigma_k)$，可写为：
$$J_\sigma=\mathrm{diag}(\boldsymbol{\sigma})-\boldsymbol{\sigma}\boldsymbol{\sigma}^\top.$$

其中 $\mathrm{diag}(\boldsymbol{\sigma})$ 为以 $\sigma_i$ 为对角元素的对角矩阵。

**第三步：验证 $J_\sigma$ 是半负定的。** 对任意向量 $\mathbf{v}\in\mathbb{R}^n$：
$$\mathbf{v}^\top J_\sigma\mathbf{v}=\sum_i\sigma_iv_i^2-\left(\sum_i\sigma_iv_i\right)^2.$$

由 Cauchy-Schwarz 不等式（权重为 $\sigma_i>0$，$\sum\sigma_i=1$）：
$$\left(\sum_i\sigma_iv_i\right)^2\le\sum_i\sigma_i\cdot\sum_i\sigma_iv_i^2=\sum_i\sigma_iv_i^2.$$

故 $\mathbf{v}^\top J_\sigma\mathbf{v}\le 0$，$J_\sigma\preceq 0$（半负定）。等号当且仅当所有 $v_i$ 相等时成立。

> **答案：** $\dfrac{\partial\sigma_i}{\partial z_k}=\sigma_i(\delta_{ik}-\sigma_k)$，矩阵形式 $J_\sigma=\mathrm{diag}(\boldsymbol{\sigma})-\boldsymbol{\sigma}\boldsymbol{\sigma}^\top\preceq 0$。

**总结**  
Softmax Jacobian 的半负定性体现了"竞争性"：增大任一 $z_i$ 会增大 $\sigma_i$ 同时减小所有其他 $\sigma_j$（总和始终为 1）。在分类任务反向传播中，D.95 的简洁结果 $\nabla_\mathbf{z}L=\mathbf{p}-\mathbf{y}$ 正是利用了此 Jacobian 与交叉熵梯度的消除。

---

### E.54 [提升] Ch.26 — 神经网络反向传播

**题目回顾**  
设 $\mathbf{y}=\sigma(W\mathbf{x}+\mathbf{b})$，$\partial L/\partial\mathbf{y}$ 已知。推导 $\partial L/\partial W_{ij}$，写成矩阵形式，并给出 $\partial L/\partial\mathbf{x}$。

**思路**  
逐步应用链式法则：$\partial L/\partial W_{ij}=(\partial L/\partial\mathbf{y})(\partial\mathbf{y}/\partial W_{ij})$，注意 $\mathbf{y}$ 是向量，需用 Jacobian。

**解答**  
**记号：** 令 $\mathbf{z}=W\mathbf{x}+\mathbf{b}\in\mathbb{R}^m$，$\mathbf{y}=\sigma(\mathbf{z})$（逐元素作用），$\delta_i=\dfrac{\partial L}{\partial y_i}\cdot\sigma'(z_i)$（反传误差信号）。

**第一步：$\partial L/\partial W_{ij}$（标量 $W_{ij}$ 对应第 $i$ 行第 $j$ 列）。**

$z_i=\sum_k W_{ik}x_k+b_i$，故 $\dfrac{\partial z_i}{\partial W_{ij}}=x_j$，其余 $\dfrac{\partial z_l}{\partial W_{ij}}=0$（$l\ne i$）。

链式法则：
$$\frac{\partial L}{\partial W_{ij}}=\sum_l\frac{\partial L}{\partial y_l}\cdot\sigma'(z_l)\cdot\frac{\partial z_l}{\partial W_{ij}}=\frac{\partial L}{\partial y_i}\cdot\sigma'(z_i)\cdot x_j=\delta_i x_j.$$

**第二步：矩阵形式。** 注意 $(\partial L/\partial W)_{ij}=\delta_i x_j$，即外积结构：
$$\frac{\partial L}{\partial W}=\boldsymbol{\delta}\mathbf{x}^\top,$$

其中 $\boldsymbol{\delta}=\mathrm{diag}(\sigma'(\mathbf{z}))\cdot\dfrac{\partial L}{\partial\mathbf{y}}\in\mathbb{R}^m$（逐元素乘积），$\mathbf{x}^\top\in\mathbb{R}^{1\times n}$。

**维度验证：** $\boldsymbol{\delta}\mathbf{x}^\top\in\mathbb{R}^{m\times n}$，与 $W\in\mathbb{R}^{m\times n}$ 一致。

**第三步：$\partial L/\partial\mathbf{x}$（用于传播到前一层）。**

$z_i=\sum_k W_{ik}x_k+b_i$，故 $\dfrac{\partial z_i}{\partial x_j}=W_{ij}$。

$$\frac{\partial L}{\partial x_j}=\sum_i\frac{\partial L}{\partial y_i}\sigma'(z_i)W_{ij}=\sum_i\delta_i W_{ij}=(W^\top\boldsymbol{\delta})_j.$$

矩阵形式：
$$\frac{\partial L}{\partial\mathbf{x}}=W^\top\boldsymbol{\delta}.$$

> **答案：** $\dfrac{\partial L}{\partial W}=\boldsymbol{\delta}\mathbf{x}^\top$，$\dfrac{\partial L}{\partial\mathbf{x}}=W^\top\boldsymbol{\delta}$，其中 $\boldsymbol{\delta}=\sigma'(\mathbf{z})\odot\dfrac{\partial L}{\partial\mathbf{y}}$。

**总结**  
反向传播的数学本质是链式法则 + 转置矩阵。名称"反向传播"来源于误差信号 $\boldsymbol{\delta}$ 从输出层向输入层逐层传递（乘以 $W^\top$），方向与前向传播（乘以 $W$）相反。这一结构使得 $O(mT)$ 的代价（见 E.58）可以计算全梯度。

---

### E.55 [提升] Ch.25-28 — KL 散度非负性（Jensen 不等式）

**题目回顾**  
设 $p,q$ 是离散概率分布（$p_i,q_i>0$，$\sum p_i=\sum q_i=1$）。证明 $D_\mathrm{KL}(p\|q)\ge 0$，等号当且仅当 $p=q$。

**思路**  
用 $\ln$ 的凹性和 Jensen 不等式：$\sum p_i\ln(q_i/p_i)\le\ln(\sum p_i\cdot q_i/p_i)=\ln 1=0$。

**解答**  
等价形式：需证 $-D_\mathrm{KL}(p\|q)=\sum_i p_i\ln\dfrac{q_i}{p_i}\le 0$。

**核心论证（Jensen 不等式）：** $\ln$ 是严格凹函数（$(\ln x)''=-1/x^2<0$）。由凹函数的 Jensen 不等式：对权重 $p_i>0$，$\sum p_i=1$，以及正数 $t_i=q_i/p_i$：
$$\sum_i p_i\ln\frac{q_i}{p_i}\le\ln\!\left(\sum_i p_i\cdot\frac{q_i}{p_i}\right)=\ln\!\left(\sum_i q_i\right)=\ln 1=0.$$

故 $D_\mathrm{KL}(p\|q)=-\sum_i p_i\ln(q_i/p_i)\ge 0$。

**等号条件：** Jensen 不等式取等 $\Leftrightarrow$ 所有 $t_i=q_i/p_i$ 相等（设为常数 $c$）。由 $\sum q_i=c\sum p_i=c\cdot1=1$，故 $c=1$，即 $q_i=p_i$ 对所有 $i$ 成立。$\square$

**补充：** KL 散度不是真正的距离（度量），因为一般 $D_\mathrm{KL}(p\|q)\ne D_\mathrm{KL}(q\|p)$（不满足对称性）；也不满足三角不等式。对称化版本 $D_\mathrm{JS}(p\|q)=\tfrac12 D_\mathrm{KL}(p\|M)+\tfrac12 D_\mathrm{KL}(q\|M)$（$M=\tfrac{p+q}{2}$）称为 JS 散度，满足对称性且有界（$\in[0,\ln 2]$），被 GAN 的原始理论采用。

> **答案：** 由 $\ln$ 的凹性和 Jensen 不等式，$D_\mathrm{KL}(p\|q)\ge 0$；等号当且仅当 $p=q$。

**总结**  
KL 散度非负性是信息论和机器学习中最基础的不等式之一。变分自编码器（VAE）的 ELBO 下界、EM 算法的单调性、最大似然估计的合理性都依赖于这一结论。

---

### E.56 [提升] Ch.25 — KKT 条件与超平面距离

**题目回顾**  
最小化 $f(\mathbf{x})=\|\mathbf{x}\|^2$，约束 $\mathbf{a}^\top\mathbf{x}=b$。求最优解并给出几何解释。

**思路**  
写出 Lagrangian，令梯度为零，代入约束解出 Lagrange 乘子，得到最优点和最小值。

**解答**  
**第一步：Lagrangian。**
$$\mathcal{L}(\mathbf{x},\lambda)=\|\mathbf{x}\|^2-\lambda(\mathbf{a}^\top\mathbf{x}-b).$$

**第二步：KKT（等式约束时为驻点条件）。**
$$\nabla_\mathbf{x}\mathcal{L}=2\mathbf{x}-\lambda\mathbf{a}=\mathbf{0}\implies\mathbf{x}=\frac{\lambda}{2}\mathbf{a}.$$

**第三步：代入约束求 $\lambda$。**
$$\mathbf{a}^\top\left(\frac{\lambda}{2}\mathbf{a}\right)=b\implies\frac{\lambda}{2}\|\mathbf{a}\|^2=b\implies\lambda=\frac{2b}{\|\mathbf{a}\|^2}.$$

**第四步：最优解与最小值。**
$$\mathbf{x}^*=\frac{b}{\|\mathbf{a}\|^2}\mathbf{a},\quad f^*=\|\mathbf{x}^*\|^2=\frac{b^2}{\|\mathbf{a}\|^4}\|\mathbf{a}\|^2=\frac{b^2}{\|\mathbf{a}\|^2}.$$

**几何解释：** 超平面 $\mathbf{a}^\top\mathbf{x}=b$ 到原点的距离为
$$d=\frac{|b|}{\|\mathbf{a}\|},\quad d^2=f^*=\frac{b^2}{\|\mathbf{a}\|^2}.$$

最优点 $\mathbf{x}^*=\dfrac{b}{\|\mathbf{a}\|^2}\mathbf{a}$ 是原点在超平面上的正交投影，方向与法向量 $\mathbf{a}$ 一致。

**凸性验证：** $f(\mathbf{x})=\|\mathbf{x}\|^2$ 是严格凸函数（Hessian $= 2I\succ0$），约束集为仿射子空间（凸集），故最优解唯一。

> **答案：** $\mathbf{x}^*=\dfrac{b}{\|\mathbf{a}\|^2}\mathbf{a}$，最小值 $f^*=\dfrac{b^2}{\|\mathbf{a}\|^2}$；几何意义为原点到超平面的最短距离的平方。

**总结**  
此题将 KKT 条件与经典几何（点到平面距离）统一。支持向量机（SVM）的核心数学——最大化间隔等价于最小化 $\|{\bf w}\|^2$ 的约束优化——正是这一结构的直接应用。

---

### E.57 [提升] Ch.25-26 — L1 范数与 Lasso 次梯度

**题目回顾**  
证明 $f(\mathbf{x})=\|\mathbf{x}\|_1$ 是凸函数，求次梯度 $\partial f(\mathbf{x})$，解释软阈值算子的来源。

**思路**  
凸性通过三角不等式直接验证；次梯度用定义逐分量分析；软阈值从 Lasso 最优性条件推出。

**解答**  
**第一步：$\|\cdot\|_1$ 的凸性。** 对任意 $\mathbf{x},\mathbf{y}\in\mathbb{R}^n$，$\lambda\in[0,1]$：
$$\|\lambda\mathbf{x}+(1-\lambda)\mathbf{y}\|_1\le\lambda\|\mathbf{x}\|_1+(1-\lambda)\|\mathbf{y}\|_1.$$

由绝对值三角不等式 $|a+b|\le|a|+|b|$ 逐分量求和即得。$\square$

**第二步：次梯度 $\partial f(\mathbf{x})$。** 次梯度集合由定义：$g\in\partial f(\mathbf{x})$ 满足 $f(\mathbf{y})\ge f(\mathbf{x})+g^\top(\mathbf{y}-\mathbf{x})$ 对所有 $\mathbf{y}$ 成立。

因为 $f(\mathbf{x})=\sum_i|x_i|$，各分量独立，次梯度逐分量给出：
$$(\partial f)_i=\begin{cases}+1 & \text{若 } x_i>0,\\ -1 & \text{若 } x_i<0,\\ [-1,+1] & \text{若 } x_i=0.\end{cases}$$

即 $\partial f(\mathbf{x})=\{\mathbf{g}:g_i=\mathrm{sign}(x_i)\text{（当 }x_i\ne0\text{），}g_i\in[-1,1]\text{（当 }x_i=0\text{）}\}$。

**第三步：软阈值算子的推导。** Lasso 问题：$\min_\mathbf{w}\|\mathbf{X}\mathbf{w}-\mathbf{y}\|^2+\lambda\|\mathbf{w}\|_1$。

最优性条件（次梯度为零）：
$$\mathbf{0}\in\nabla_\mathbf{w}\|\mathbf{X}\mathbf{w}-\mathbf{y}\|^2+\lambda\partial\|\mathbf{w}\|_1.$$

对坐标下降，固定其他分量，对第 $i$ 个分量 $w_i$ 求解（设 $z_i$ 为去掉 $w_i$ 后的残差相关量）：
$$0=2z_i-2\|X_i\|^2w_i+\lambda g_i,\quad g_i\in\partial|w_i|.$$

整理并令 $\hat{z}_i=z_i/\|X_i\|^2$（简化版）：
$$w_i^*=\mathrm{sign}(\hat{z}_i)\max(|\hat{z}_i|-\lambda/(2\|X_i\|^2),\,0).$$

这正是**软阈值（soft-thresholding）算子**：将 $|\hat{z}_i|$ 减小 $\lambda$ 并截断到非负，由 $L_1$ 的非光滑性产生自然的稀疏性（若 $|\hat{z}_i|\le\lambda$，则 $w_i^*=0$）。

> **答案：** $\|\cdot\|_1$ 是凸函数；$\partial\|\mathbf{x}\|_1=\prod_i\partial|x_i|$，逐分量为 $\mathrm{sign}(x_i)$（$x_i\ne0$）或 $[-1,1]$（$x_i=0$）；软阈值算子 $w_i^*=\mathrm{sign}(z_i)(|z_i|-\lambda/2)_+$ 来自最优性次梯度条件。

**总结**  
$L_1$ 正则化导致稀疏解的根本原因是：最优性条件在 $x_i=0$ 处允许次梯度取任意值（$[-1,1]$），形成一个"吸收区间"使解精确为零。这是 Lasso 的稀疏诱导能力的数学核心。

---

### E.58 [提升] Ch.26 — 自动微分模式的计算复杂度

**题目回顾**  
比较前向模式与反向模式自动微分的计算复杂度，以 $f:\mathbb{R}^n\to\mathbb{R}^m$ 为例，结合深度网络实例说明反向模式的优势。

**思路**  
明确 Jacobian 的维度，分析两种模式分别需要多少次"方向传播"，然后数值对比。

**解答**  
**背景：** 自动微分（AD）利用计算图，通过链式法则精确（非数值）计算导数。设计算图有 $T$ 条边（基本运算），$f:\mathbb{R}^n\to\mathbb{R}^m$，Jacobian $J\in\mathbb{R}^{m\times n}$。

**前向模式（Forward Mode / Tangent Mode）：**
- 每次传播一个"方向向量" $\mathbf{v}\in\mathbb{R}^n$，计算方向导数 $J\mathbf{v}\in\mathbb{R}^m$（Jacobian 乘以向量）；
- 计算一次方向传播的代价：$O(T)$（与函数求值同量级）；
- 计算完整 Jacobian（$n$ 列）需传播 $n$ 个标准基向量：**总代价 $O(nT)$**。

**反向模式（Backward Mode / Adjoint Mode，即反向传播）：**
- 每次传播一个"余切向量"（cotangent / 行梯度）$\mathbf{w}^\top\in\mathbb{R}^{1\times m}$，计算 $\mathbf{w}^\top J\in\mathbb{R}^{1\times n}$（转置 Jacobian 乘以向量）；
- 计算一次反向传播的代价：$O(T)$（一般为前向的 2–5 倍常数）；
- 计算完整 Jacobian（$m$ 行）需传播 $m$ 个标准基向量：**总代价 $O(mT)$**。

**结论对比：**

| 模式 | 计算完整 Jacobian | 适用场景 |
|---|---|---|
| 前向 | $O(nT)$ | $m\gg n$（输出多于输入） |
| 反向 | $O(mT)$ | $n\gg m$（输入多于输出） |

**深度网络实例（$n=10^6,m=1$）：**

典型神经网络：参数数量 $n=10^6$（输入维度）；损失函数 $m=1$（标量输出）。

- 前向模式：需要 $10^6$ 次方向传播，代价为 $10^6\times T$ 次操作——**不可行**；
- 反向模式：仅需 $1$ 次反向传播（$m=1$），代价约为 $T$——**高效**，仅比前向计算多 3–5 倍常数代价。

这解释了为什么现代深度学习框架（PyTorch、TensorFlow）均默认使用反向模式自动微分（即 backprop）：对 $m=1$ 的损失函数，单次反向传播即得所有 $10^6$ 个参数的梯度。

> **答案：** 前向模式复杂度 $O(nT)$，反向模式 $O(mT)$；$m=1$（标量损失）时反向模式以 $O(T)$ 得全梯度，远优于前向的 $O(nT)$。

**总结**  
选择自动微分模式的原则：$m<n$ 用反向（深度学习的标准情形），$m>n$ 用前向（科学计算中少见但存在）。混合模式（前向 + 反向）可进一步优化。

---

### E.59 [提升] Ch.28 — Itô 公式与随机积分

**题目回顾**  
对布朗运动 $B_t$ 推导 $d(B_t^2)$，写出 Itô 公式一般形式，并计算随机积分 $\int_0^tB_s\,dB_s$。

**思路**  
用二阶 Taylor 展开启发性推导，利用 $(dB_t)^2=dt$（Itô 等距），再积分。

**解答**  
**第一步：启发性推导 $d(B_t^2)$。**

对 $f(x)=x^2$ 做 Taylor 展开（保留到二阶）：
$$d(B_t^2)\approx 2B_t\,dB_t+\frac{1}{2}\cdot 2\cdot(dB_t)^2.$$

关键：在随机微积分中，$(dB_t)^2=dt$（而非普通微积分中的 $o(dt)$）。这来源于布朗运动的二次变差：
$$\langle B\rangle_t=\lim_{n\to\infty}\sum_{k=0}^{n-1}(B_{t_{k+1}}-B_{t_k})^2=t\quad\text{（依概率收敛）}.$$

因此：
$$d(B_t^2)=2B_t\,dB_t+dt.$$

**第二步：Itô 公式一般形式。** 设 $X_t$ 满足 SDE $dX_t=\mu_t\,dt+\sigma_t\,dB_t$，$f\in C^2$，则：
$$df(X_t)=f'(X_t)\,dX_t+\frac{1}{2}f''(X_t)\sigma_t^2\,dt.$$

展开完整形式：
$$df(X_t)=\left(\mu_tf'(X_t)+\frac{\sigma_t^2}{2}f''(X_t)\right)dt+\sigma_tf'(X_t)\,dB_t.$$

与普通微积分的差异：多出 $\dfrac{\sigma_t^2}{2}f''(X_t)\,dt$（二阶修正项，"Itô 修正"）。

**第三步：计算 $\int_0^tB_s\,dB_s$。** 对 $d(B_t^2)=2B_t\,dB_t+dt$ 两边从 $0$ 到 $t$ 积分：
$$B_t^2-B_0^2=2\int_0^tB_s\,dB_s+\int_0^t ds.$$

由 $B_0=0$，$\int_0^t ds=t$：
$$B_t^2=2\int_0^tB_s\,dB_s+t.$$

解出：
$$\boxed{\int_0^tB_s\,dB_s=\frac{B_t^2-t}{2}.}$$

对比普通微积分：$\int_0^tx\,dx=\dfrac{x^2}{2}\Big|_0^t=\dfrac{t^2}{2}$，而随机版本多了 $-t/2$（Itô 修正项）。

**验证期望：** $E\!\left[\int_0^tB_s\,dB_s\right]=\dfrac{E[B_t^2]-t}{2}=\dfrac{t-t}{2}=0$，符合 Itô 随机积分是鞅（期望为零）的性质。

> **答案：** $d(B_t^2)=2B_t\,dB_t+dt$（含 Itô 修正项 $dt$）；$\displaystyle\int_0^tB_s\,dB_s=\dfrac{B_t^2-t}{2}$。

**总结**  
Itô 公式的核心洞见：布朗运动轨迹的"粗糙性"（二次变差非零）使得经典链式法则失效，必须加入二阶修正项。在金融数学中，这一修正导致期权定价的 Black-Scholes 方程中出现 $\frac{1}{2}\sigma^2S^2\frac{\partial^2V}{\partial S^2}$ 项。

---

### E.60 [提升] Ch.25-28 — 信息熵最大化与最大熵原理

**题目回顾**  
用 Lagrange 乘子法在 $\sum p_i=1$ 约束下最大化 Shannon 熵 $H(p)=-\sum p_i\ln p_i$，并由 KL 散度非负性直接推出上界，讨论最大熵原理的 AI 含义。

**思路**  
分两种方法：① Lagrange 乘子法直接求解；② 利用 KL 散度非负性（E.55）给出简洁证明。

**解答**  
**方法一：Lagrange 乘子法。**

目标：$\max_{p}\left(-\sum_i p_i\ln p_i\right)$，约束：$\sum_i p_i=1$，$p_i>0$。

Lagrangian：
$$\mathcal{L}(p,\lambda)=-\sum_i p_i\ln p_i-\lambda\!\left(\sum_i p_i-1\right).$$

对 $p_k$ 求偏导并令其为零：
$$\frac{\partial\mathcal{L}}{\partial p_k}=-\ln p_k-1-\lambda=0\implies\ln p_k=-1-\lambda\implies p_k=e^{-1-\lambda}.$$

由约束 $\sum_i p_i=n\cdot e^{-1-\lambda}=1$，解得 $e^{-1-\lambda}=1/n$，即
$$p_k^*=\frac{1}{n},\quad\forall k.$$

最大熵值：
$$H^*=H(p^*)=-\sum_{i=1}^n\frac{1}{n}\ln\frac{1}{n}=-n\cdot\frac{1}{n}\cdot(-\ln n)=\ln n.$$

**方法二：KL 散度非负性的直接推导。**

取 $q_i=1/n$（均匀分布），由 E.55，$D_\mathrm{KL}(p\|q)\ge 0$，即：
$$\sum_i p_i\ln\frac{p_i}{1/n}\ge 0\implies\sum_i p_i\ln p_i\ge-\ln n\implies H(p)\le\ln n.$$

等号当且仅当 $p=q$（均匀分布）时成立，与方法一一致。

**最大熵原理的 AI 含义。**

**带约束（均值 $\sum p_ix_i=\mu$）的最大熵：** 在额外约束 $\sum_i p_ix_i=\mu$，$\sum_i p_i=1$ 下，Lagrangian：
$$\mathcal{L}=-\sum_i p_i\ln p_i-\lambda_0(\sum_i p_i-1)-\lambda_1(\sum_i p_ix_i-\mu).$$

最优性条件：$-\ln p_i-1-\lambda_0-\lambda_1x_i=0$，故
$$p_i^*\propto e^{-\lambda_1x_i}.$$

这是**指数族分布**（Gibbs 分布）的形式。约束越多（均值、方差、高阶矩），指数族越复杂（正态分布 = 同时约束一阶和二阶矩时的最大熵分布）。

**"最少假设"先验的含义：** 在已知信息（约束）下选择最大熵分布，等价于"对未知信息不作任何额外假设"。这是贝叶斯学派中无信息先验的原则，也是语言模型温度参数（temperature $T$）的数学基础：Softmax 中 $p_i=e^{z_i/T}/\sum_j e^{z_j/T}$ 在 $T\to\infty$ 时趋近均匀（最大熵），在 $T\to0$ 时趋近 argmax（最小熵）。

> **答案：** 最大熵在均匀分布 $p_i^*=1/n$ 时取到，最大值为 $\ln n$；最大熵原理下，均值约束导出指数族（Gibbs）分布。

**总结**  
最大熵原理将概率论（凸优化）、信息论（KL 散度）与统计物理（Gibbs 分布）统一。在大语言模型中，温度参数 $T$ 直接控制输出分布的熵，$T>1$ 增加多样性（高熵），$T<1$ 增加确定性（低熵），是生成多样化文本与避免重复之间权衡的数学核心。

---

## 总览与学习指引

| 编号 | 主题 | 核心公式 / 结论 | 对应 ML 应用 |
|------|------|----------------|-------------|
| C.74 | 凸函数 ($f''$) | $f''\ge0\Rightarrow$凸 | 损失函数凸性 |
| C.75 | 梯度下降 | $\theta\leftarrow\theta-\eta\nabla L$ | 参数优化 |
| C.76 | 线性函数梯度 | $\nabla(\mathbf{a}^\top\mathbf{x})=\mathbf{a}$ | 感知机 |
| C.77 | 二次型梯度 | $\nabla(\mathbf{x}^\top A\mathbf{x})=2A\mathbf{x}$ | PCA / Ridge |
| C.78 | 正态矩 | $E[X^2]=\mu^2+\sigma^2$ | VAE / BN |
| C.79 | 独立性 | $E[XY]=\mu_X\mu_Y$ | Dropout 分析 |
| C.80 | 布朗运动 | $E[W_t^2]=t$ | 随机过程建模 |
| D.91 | 全局最优 | 凸 + $\nabla f=0\Rightarrow$全局最小 | 凸优化基础 |
| D.92 | 强凸性 | $H\succ0$，强凸模数 $\mu=\lambda_{\min}$ | 正则化理论 |
| D.93 | 正规方程 | $A^\top A\mathbf{x}=A^\top\mathbf{b}$ | 线性回归 |
| D.94 | Sigmoid 梯度 | $\nabla\sigma=\sigma(1-\sigma)\mathbf{x}$ | logistic 回归 |
| D.95 | 交叉熵梯度 | $\nabla_\mathbf{z}L=\mathbf{p}-\mathbf{y}$ | 分类反传 |
| D.96 | 四阶矩 | $E[X^4]=3$ (标准正态) | 峰度 / BN |
| D.97 | 正态卷积性 | 独立正态之和仍正态 | 噪声叠加 |
| D.98 | 同方差 KL | $(\mu_1-\mu_2)^2/(2\sigma^2)$ | VAE 正则 |
| D.99 | 布朗协方差 | $E[W_sW_t]=\min(s,t)$ | GP 核函数 |
| D.100 | Itô 过程二阶矩 | $E[X_t^2]=\mu^2t^2+\sigma^2t$ | 随机控制 |
| E.53 | Softmax Jacobian | $J_\sigma=\mathrm{diag}(\boldsymbol{\sigma})-\boldsymbol{\sigma}\boldsymbol{\sigma}^\top$ | 分类反传 |
| E.54 | 反向传播 | $\partial L/\partial W=\boldsymbol{\delta}\mathbf{x}^\top$ | 神经网络训练 |
| E.55 | KL 散度 $\ge0$ | Jensen 不等式 | ELBO / EM |
| E.56 | KKT 优化 | $\mathbf{x}^*=b\mathbf{a}/\|\mathbf{a}\|^2$ | SVM 间隔 |
| E.57 | Lasso 次梯度 | 软阈值算子 | 稀疏学习 |
| E.58 | 自动微分 | 反向 $O(T)$ vs 前向 $O(nT)$ | 深度学习训练 |
| E.59 | Itô 公式 | $\int_0^tB_s\,dB_s=(B_t^2-t)/2$ | 金融数学 |
| E.60 | 最大熵 | $H(p)\le\ln n$，Gibbs 分布 | 温度参数 / 先验 |

> **学习建议：**  
> - C 级题目建议先独立完成，重点掌握矩阵微积分的基本运算规则（C.76、C.77）和概率矩的计算（C.78、C.79）；  
> - D 级题目需要综合运用凸优化理论（D.91–D.92）和矩阵求导（D.93–D.95），概率部分（D.96–D.98）着重理解矩生成函数和 KL 散度的几何意义；  
> - E 级题目的重点是系统性地将微积分工具应用于 AI 场景：E.53–E.54 打通矩阵微积分与反向传播，E.55–E.57 连接凸优化与机器学习正则化，E.59–E.60 展示随机微积分与信息熵的深层联系。
