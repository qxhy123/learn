# 第25章 凸优化基础

## 学习目标

通过本章学习，你将能够：

- 判断一个集合或函数是否具有凸性（convexity）
- 理解凸函数的一阶/二阶判别条件与几何直觉
- 掌握 Jensen 不等式、对偶理论与 KKT 条件的基本用法
- 将 SVM、正则化、ELBO 等 AI 问题写成凸优化语言

> **依赖章节**：第 9 章（导数的应用）、第 18 章（偏导数、Hessian、Lagrange 乘子）
>
> **阅读提示**：本章默认你已具备基本线性代数知识，例如向量内积、矩阵转置与正定矩阵的直观概念。

---

## 25.1 凸集与凸函数

### 25.1.1 凸集的定义

在一维中，区间最大的特点是：只要取区间内两点，它们之间的整段线段仍然留在区间里。这个性质在高维空间中的推广，就是**凸集**。

**定义**：集合 $S \subseteq \mathbb{R}^n$ 称为凸集，若对任意 $x, y \in S$ 与任意 $\lambda \in [0,1]$，都有

$$
\lambda x + (1-\lambda)y \in S.
$$

也就是说，连接 $x$ 和 $y$ 的线段完全落在集合内部。

**常见凸集**：

- 区间、半空间、线性不等式组定义的多面体
- 欧氏球 $\{x \mid \|x\|_2 \leq r\}$
- 椭球 $\{x \mid (x-c)^\top A (x-c) \leq 1,\ A \succ 0\}$
- 范数球 $\{x \mid \|x\| \leq r\}$

**重要性质**：任意多个凸集的交集仍为凸集。这一点非常重要，因为优化中的可行域往往由多个约束同时定义。

> **例题 25.1** 证明集合 $S = \{x \in \mathbb{R}^n \mid Ax \leq b\}$ 是凸集。

**解**：任取 $x,y \in S$，则有 $Ax \leq b$ 与 $Ay \leq b$。对任意 $\lambda \in [0,1]$，

$$
A(\lambda x + (1-\lambda)y)
= \lambda Ax + (1-\lambda)Ay
\leq \lambda b + (1-\lambda)b = b.
$$

故 $\lambda x + (1-\lambda)y \in S$，因此 $S$ 是凸集。$\square$

### 25.1.2 凸函数的定义与几何直觉

**定义**：函数 $f: \mathbb{R}^n \to \mathbb{R}$ 在凸集 $C$ 上称为凸函数，若对任意 $x,y \in C$ 与任意 $\lambda \in [0,1]$，有

$$
f(\lambda x + (1-\lambda)y)
\leq \lambda f(x) + (1-\lambda)f(y).
$$

这意味着：函数图像上任意两点连成的弦线，始终在图像上方。

若上式在 $x \neq y$ 且 $\lambda \in (0,1)$ 时严格成立，则称为**严格凸函数**。

一维时，$f''(x) \geq 0$ 常被用来判断凸性；多维时，对应工具就是 Hessian 矩阵。

### 25.1.3 一阶与二阶判别条件

若 $f$ 可微，则它在凸集上为凸函数，当且仅当对任意 $x,y$，都有

$$
f(y) \geq f(x) + \nabla f(x)^\top (y-x).
$$

这就是**一阶条件**：切平面始终位于函数图像下方，因此是全局下界。

若 $f$ 二阶可导，则一个充分条件是

$$
\nabla^2 f(x) \succeq 0
$$

即 Hessian 半正定（positive semidefinite, PSD）。当 Hessian 处处正定时，函数是严格凸的。

> **例题 25.2** 判断 $f(x) = \|x\|_2^2 = x^\top x$ 的凸性。

**解**：

$$
\nabla f(x) = 2x,\qquad \nabla^2 f(x) = 2I.
$$

矩阵 $2I$ 的所有特征值都是 $2>0$，因此 $\nabla^2 f(x)$ 正定，故 $f$ 是严格凸函数。$\square$

> **例题 25.3** 判断 $f(x)=x\ln x$（$x>0$）的凸性。

**解**：

$$
f'(x)=\ln x+1,\qquad f''(x)=\frac{1}{x}>0 \quad (x>0).
$$

因此 $f(x)=x\ln x$ 在 $(0,+\infty)$ 上严格凸。这个函数在信息论和熵的推导中反复出现。$\square$

### 25.1.4 常见凸函数与保凸运算

**常见凸函数**：

- 线性函数与仿射函数
- 二次函数 $x^\top A x + b^\top x + c$（当 $A \succeq 0$）
- 范数 $\|x\|_1,\|x\|_2$
- `log-sum-exp`：$\log \sum_i e^{x_i}$
- 负熵函数 $-\sum_i p_i \log p_i$

**保凸运算**：

- 非负加权和
- 与仿射变换复合
- 逐点最大值
- 透视函数（perspective transform）

**非凸反例**：

- $\sin x$
- 深度网络的整体损失函数
- 带多个局部极小值的多峰函数

> ⚠️ **常见陷阱**
> 梯度为零并不 automatically 意味着已经找到最优解。对非凸函数，$\nabla f(x)=0$ 可能对应局部极值，也可能只是鞍点。凸性提供的是“驻点 = 全局最优”的额外保证。

---

## 25.2 Jensen 不等式与信息论

### 25.2.1 Jensen 不等式

对于凸函数 $f$，如果 $\lambda_i \geq 0$ 且 $\sum_i \lambda_i = 1$，则

$$
f\left(\sum_i \lambda_i x_i\right)
\leq \sum_i \lambda_i f(x_i).
$$

这就是**离散形式**的 Jensen 不等式。

若 $X$ 是随机变量，则连续形式写为

$$
f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)].
$$

它告诉我们：先平均再过凸函数，通常比先过凸函数再平均更小。

> **例题 25.4** 用 Jensen 不等式证明 AM-GM 不等式。

**解**：对正数 $a,b$，取凸函数 $f(x)=-\ln x$。则

$$
-\ln \left(\frac{a+b}{2}\right)
\leq \frac{-\ln a - \ln b}{2}
= -\ln \sqrt{ab}.
$$

两边同时乘以 $-1$ 并指数化，得

$$
\frac{a+b}{2} \geq \sqrt{ab}.
$$

这就是 AM-GM 不等式。$\square$

### 25.2.2 KL 散度非负性

设 $p(x)$ 和 $q(x)$ 是两个概率密度，则

$$
\mathrm{KL}(p\|q)
= \int p(x)\ln \frac{p(x)}{q(x)}\,dx.
$$

由于 $-\ln t$ 是凸函数，对随机变量 $Z = \dfrac{q(X)}{p(X)}$ 应用 Jensen，不难得到

$$
\mathrm{KL}(p\|q)\geq 0.
$$

等号仅在 $p=q$（几乎处处）时成立。

> **例题 25.5** 用 Jensen 不等式证明 $\mathrm{KL}(p\|q)\geq 0$。

**解**：令 $f(t)=-\ln t$，它是凸函数。又因为在 $X\sim p$ 下，

$$
\mathbb{E}_p\left[\frac{q(X)}{p(X)}\right]
= \int p(x)\frac{q(x)}{p(x)}\,dx
= \int q(x)\,dx = 1.
$$

由 Jensen 不等式，

$$
-\ln \mathbb{E}_p\left[\frac{q(X)}{p(X)}\right]
\leq
\mathbb{E}_p\left[-\ln \frac{q(X)}{p(X)}\right].
$$

左边为 $-\ln 1 = 0$，右边正是 $\mathrm{KL}(p\|q)$，故

$$
0 \leq \mathrm{KL}(p\|q).
$$

$\square$

### 25.2.3 ELBO 的凸性视角

变分推断中，一个核心难点是边缘似然

$$
\log p(x) = \log \int p(x,z)\,dz
$$

难以直接计算。若引入近似分布 $q(z)$，则

$$
\log p(x)
= \log \int q(z)\frac{p(x,z)}{q(z)}\,dz
= \log \mathbb{E}_{q}\left[\frac{p(x,z)}{q(z)}\right].
$$

由于 $\log$ 是凹函数，由 Jensen 不等式得到

$$
\log p(x)
\geq \mathbb{E}_q[\log p(x,z)] - \mathbb{E}_q[\log q(z)].
$$

右边就是 ELBO（evidence lower bound）。这个推导说明：Jensen 不等式不是抽象技巧，而是 VAE 等现代生成模型的数学地基。

---

## 25.3 凸优化问题与 KKT 条件

### 25.3.1 标准形式与全局最优性

一个典型的约束优化问题写作

$$
\begin{aligned}
\min_x \quad & f(x) \\
\text{s.t.}\quad & g_i(x)\leq 0,\quad i=1,\dots,m, \\
& h_j(x)=0,\quad j=1,\dots,p.
\end{aligned}
$$

若目标函数 $f$ 和不等式约束 $g_i$ 都是凸函数，而等式约束 $h_j$ 是仿射函数，那么这就是一个**凸优化问题**。

凸优化的最重要性质是：

> 任何局部最优解都是全局最优解。

这也是为什么机器学习里即便最终问题常常是非凸的，人们仍然喜欢先寻找凸近似、凸松弛或局部凸子问题。

> **例题 25.6** 说明为什么在线性回归加 $L_2$ 正则化时，目标函数是凸的。

**解**：Ridge 回归的目标函数为

$$
f(w)=\frac{1}{2}\|Xw-y\|_2^2 + \frac{\lambda}{2}\|w\|_2^2.
$$

第一项是二次函数，其 Hessian 为 $X^\top X \succeq 0$；第二项 Hessian 为 $\lambda I \succeq 0$。两者之和仍为半正定，因此该目标函数是凸的；若 $\lambda>0$，则常常还是严格凸的。$\square$

### 25.3.2 Lagrange 对偶

定义 Lagrange 函数

$$
L(x,\lambda,\nu)
= f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x),
$$

其中 $\lambda_i \geq 0$。

对偶函数定义为

$$
g(\lambda,\nu)=\inf_x L(x,\lambda,\nu).
$$

它给出了原问题最优值 $p^\star$ 的下界：

$$
g(\lambda,\nu)\leq p^\star.
$$

这叫做**弱对偶性**。当满足 Slater 条件等正则性假设时，常有

$$
d^\star = p^\star,
$$

这叫做**强对偶性**。

对偶问题的意义并不只在“换个写法”。在 SVM 中，对偶问题自然引出核技巧；在正则化问题中，对偶变量常对应约束的“影子价格”。

### 25.3.3 KKT 条件

在凸问题并满足适当约束资格条件时，最优解 $x^\star$ 与对偶变量 $(\lambda^\star,\nu^\star)$ 满足 KKT 条件：

1. 原始可行性（primal feasibility）
2. 对偶可行性（dual feasibility）：
   $$
   \lambda_i^\star \geq 0
   $$
3. 互补松弛（complementary slackness）：
   $$
   \lambda_i^\star g_i(x^\star)=0
   $$
4. 驻点条件（stationarity）：
   $$
   \nabla_x L(x^\star,\lambda^\star,\nu^\star)=0
   $$

> **例题 25.7** 求解
> $$
> \min_x \ x^2 \quad \text{s.t.}\ x\geq 1.
> $$

**解**：把约束写成 $g(x)=1-x\leq 0$。Lagrange 函数为

$$
L(x,\lambda)=x^2+\lambda(1-x),\qquad \lambda\geq 0.
$$

驻点条件：

$$
\frac{\partial L}{\partial x}=2x-\lambda=0
\quad \Rightarrow \quad \lambda=2x.
$$

原始可行要求 $x\geq 1$；互补松弛要求

$$
\lambda(1-x)=0.
$$

由于目标函数会把 $x$ 拉向 $0$，但约束强制 $x\geq 1$，故最优点应在边界 $x^\star=1$。此时 $\lambda^\star=2$，且所有 KKT 条件都满足。

所以最优解为 $x^\star=1$，最优值为 $1$。$\square$

> **例题 25.8** 在约束 $x+y=1$ 下最小化 $x^2+y^2$。

**解**：构造

$$
L(x,y,\nu)=x^2+y^2+\nu(x+y-1).
$$

驻点条件给出

$$
2x+\nu=0,\qquad 2y+\nu=0
\Rightarrow x=y.
$$

与约束 $x+y=1$ 联立得 $x=y=\frac12$，最优值为

$$
\left(\frac12\right)^2 + \left(\frac12\right)^2 = \frac12.
$$

$\square$

---

## 25.4 凸优化在 AI 中的应用

### 25.4.1 支持向量机（SVM）

软间隔 SVM 的原始问题可写成

$$
\begin{aligned}
\min_{w,b,\xi}\quad & \frac12\|w\|_2^2 + C\sum_{i=1}^n \xi_i \\
\text{s.t.}\quad & y_i(w^\top x_i + b)\geq 1-\xi_i,\quad \xi_i\geq 0.
\end{aligned}
$$

这是一个带线性约束的凸二次规划问题。它的几个关键结论都来自凸优化：

- 目标函数是凸的，因而不存在糟糕的局部极小值
- 对偶问题只依赖样本内积，因而可以使用核函数
- KKT 条件说明：只有位于间隔边界或违背间隔的样本会有非零对偶变量，这些样本就是支持向量

### 25.4.2 正则化的凸优化解释

许多正则化都可以看成“惩罚形式”和“约束形式”的互相转换：

$$
\min_w \ L(w)+\lambda\|w\|_2^2
\qquad \Longleftrightarrow \qquad
\min_w \ L(w)\ \text{s.t.}\ \|w\|_2^2 \leq r.
$$

从这个角度看：

- $L_2$ 正则化偏好小范数参数
- $L_1$ 正则化偏好稀疏解
- Elastic Net 是两种凸惩罚的加权和，因此仍然保持凸性

> **例题 25.9** 为什么说 Ridge 回归的惩罚形式与“限制参数范数”的约束形式本质等价？

**解**：考虑两个问题

$$
\min_w \ L(w)+\lambda\|w\|_2^2
\qquad\text{和}\qquad
\min_w \ L(w)\ \text{s.t.}\ \|w\|_2^2\le r.
$$

若约束问题在最优解处恰好落在边界 $\|w^\star\|_2^2=r$，则它的 KKT 条件可写成

$$
\nabla L(w^\star)+\lambda^\star \nabla \|w^\star\|_2^2=0,\qquad
\lambda^\star\ge 0.
$$

这与惩罚形式的一阶最优性条件完全同型。也就是说，对某个合适的半径 $r$，约束形式对应某个拉格朗日乘子 $\lambda^\star$；反过来，给定惩罚系数 $\lambda$，也能找到相应的有效约束半径。二者差别主要在“超参数是直接控制惩罚强度，还是间接控制可行域大小”。$\square$

### 25.4.3 深度学习里的“非凸现实”

神经网络训练问题整体上通常不是凸的，因为：

- 多层非线性组合破坏凸性
- 参数之间存在对称性和重参数化
- 损失面中大量临界点是鞍点而非局部最小值

但凸优化依然重要，原因至少有三点：

1. 局部近似：在当前点附近，Taylor 展开给出二次模型，是经典二阶方法的基础。
2. 子问题求解：如投影梯度、近端算法、约束优化子问题常是凸的。
3. 理论参照：理解凸问题，才能理解深度学习为何偏离凸世界、又如何借用凸工具。

### 25.4.4 代码示例：凸与非凸优化轨迹

```python
import numpy as np

def gd(f_grad, x0, lr=0.1, steps=20):
    x = float(x0)
    path = [x]
    for _ in range(steps):
        x = x - lr * f_grad(x)
        path.append(x)
    return path

# 凸函数 f(x) = x^2
convex_path = gd(lambda x: 2 * x, x0=4.0, lr=0.2)

# 非凸函数 g(x) = x^4 - 3x^2
nonconvex_path = gd(lambda x: 4 * x**3 - 6 * x, x0=0.4, lr=0.05)

print("convex:", np.round(convex_path, 4))
print("nonconvex:", np.round(nonconvex_path, 4))
```

**解释**：在凸函数上，梯度下降通常稳定地朝全局最优前进；而在非凸函数上，初值不同可能收敛到不同区域，甚至在鞍点附近徘徊。

> **例题 25.10** 使用上面的代码时，为什么同样从固定初值出发，凸函数与非凸函数的优化轨迹会表现出本质差异？

**解**：对凸函数 $f(x)=x^2$，梯度始终把参数拉向唯一全局最优点 $x=0$，因此轨迹单调、稳定，且不会被“局部地形”误导。对非凸函数 $g(x)=x^4-3x^2$，梯度场中同时存在多个驻点：两个局部极小值和一个位于原点附近的鞍点。于是更新方向会强烈依赖当前位置，稍微不同的初值就可能落入不同吸引域。这正是“凸优化里局部最优就是全局最优，而非凸优化则必须面对地形复杂性”的数值直观。$\square$

---

## 本章小结

1. **凸集**要求线段封闭，**凸函数**要求弦线在图像上方。
2. 对可微凸函数，切平面是全局下界；对二阶可导函数，Hessian 半正定是常用判据。
3. Jensen 不等式把凸性与概率、积分、信息论连接起来，是 KL 非负性和 ELBO 推导的核心工具。
4. 凸优化最大的礼物是：局部最优就是全局最优。
5. KKT 条件把原始约束、对偶变量和最优性条件统一到一套框架里。
6. SVM、正则化、变分推断都可以从凸优化视角获得更清晰的解释。

---

## 练习题

**1.** ⭐ 判断集合 $S=\{x\in\mathbb{R}^2 \mid x_1+x_2\leq 1,\ x_1\geq 0,\ x_2\geq 0\}$ 是否为凸集。

**2.** ⭐ 证明 `log-sum-exp` 函数
$$
f(x)=\log\left(\sum_{i=1}^n e^{x_i}\right)
$$
是凸函数。

**3.** ⭐ 设 $f(x)=\max(0,1-yw^\top x)$，其中 $y\in\{-1,1\}$。证明它关于 $w$ 是凸函数。

**4.** ⭐⭐ 写出 Ridge 回归
$$
\min_w \frac12\|Xw-y\|_2^2 + \frac{\lambda}{2}\|w\|_2^2
$$
的一阶最优性条件。

**5.** ⭐⭐ 用 Jensen 不等式证明
$$
\mathbb{E}[X^2]\geq (\mathbb{E}[X])^2.
$$

**6.** ⭐⭐ 求解
$$
\min_x\ (x-2)^2 \quad \text{s.t.}\ x\geq 3.
$$

**7.** ⭐⭐⭐ 推导 Lasso 问题
$$
\min_w \frac12\|Xw-y\|_2^2 + \lambda \|w\|_1
$$
为何通常只能保证凸性，而不能像 Ridge 一样直接得到光滑闭式解。

**8.** ⭐⭐⭐ 编程题：实现投影梯度下降求解约束问题
$$
\min_x\ \|x-c\|_2^2 \quad \text{s.t.}\ \|x\|_2\leq 1
$$
并比较“每步投影”和“最后一次性裁剪”的效果差异。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.** 这是三个半空间的交集，而半空间都是凸集，凸集的交集仍为凸集，因此 $S$ 是凸集。

---

**2.** 令
$$
p_i = \frac{e^{x_i}}{\sum_j e^{x_j}},
$$
则
$$
\frac{\partial f}{\partial x_i}=p_i.
$$
进一步可得 Hessian
$$
\nabla^2 f(x)=\mathrm{diag}(p)-pp^\top.
$$
对任意向量 $v$，
$$
v^\top \nabla^2 f(x) v
= \sum_i p_i v_i^2 - \left(\sum_i p_i v_i\right)^2
= \mathrm{Var}_p(v_i)\geq 0.
$$
故 Hessian 半正定，$f$ 为凸函数。

---

**3.** 函数 $1-yw^\top x$ 关于 $w$ 是仿射函数，常数函数 $0$ 也是凸函数。两个凸函数的逐点最大值仍是凸函数，因此 hinge loss 关于 $w$ 凸。

---

**4.** 对目标函数求梯度：
$$
\nabla_w f(w)=X^\top(Xw-y)+\lambda w.
$$
一阶最优性条件为
$$
X^\top(Xw-y)+\lambda w=0.
$$
若 $X^\top X+\lambda I$ 可逆，则
$$
w^\star = (X^\top X + \lambda I)^{-1}X^\top y.
$$

---

**5.** 对凸函数 $f(x)=x^2$ 应用 Jensen 不等式：
$$
f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)].
$$
即
$$
(\mathbb{E}[X])^2 \leq \mathbb{E}[X^2].
$$

---

**6.** 可行域是 $x\geq 3$。无约束最优点在 $x=2$，不可行，因此最优解落在边界 $x^\star=3$。最优值为
$$
(3-2)^2=1.
$$

---

**7.** Lasso 目标函数是“光滑二次项 + 非光滑 $L_1$ 项”的和，因此仍然是凸函数。但 $|w_i|$ 在 $w_i=0$ 处不可导，所以不能像 Ridge 那样直接通过把梯度设为零得到闭式解。通常需要次梯度、坐标下降、近端梯度等方法求解。

---

**8.** 约束 $\|x\|_2\leq 1$ 的投影公式为
$$
\Pi(x)=
\begin{cases}
x, & \|x\|_2\leq 1,\\
\dfrac{x}{\|x\|_2}, & \|x\|_2>1.
\end{cases}
$$
投影梯度下降每一步都先做梯度更新再投影，能保证迭代点始终可行；“最后一次性裁剪”则中间会离开可行域，因此并不等价，也不能保证沿着约束问题的真实下降方向行进。

</details>
