# 第28章 随机微分方程入门

## 学习目标

通过本章学习，你将能够：

- 区分确定性微分方程与随机微分方程（stochastic differential equation, SDE）
- 理解 Brownian 运动、Itô 积分与 Itô 公式的基本直觉
- 认识 Fokker-Planck 方程如何描述分布而非单条轨迹
- 理解扩散模型中前向 SDE、反向 SDE 与概率流 ODE 的关系

> **依赖章节**：第 23-24 章（ODE）、第 27 章（概率论中的微积分）
>
> **阅读提示**：本章重在建立直觉和工程连接，不追求测度论层面的严格构造。

---

## 28.1 随机过程基础

### 28.1.1 从 ODE 到 SDE

经典常微分方程写成

$$
\frac{dx}{dt}=f(x,t),
$$

它描述的是一条完全确定的轨迹：给定初值后，未来路径被唯一决定。

现实系统往往含有噪声，因此更合理的模型是

$$
dX_t = f(X_t,t)\,dt + g(X_t,t)\,dW_t,
$$

其中：

- $f$ 是漂移项（drift），描述平均趋势
- $g$ 是扩散项（diffusion），描述噪声强度
- $W_t$ 是 Brownian 运动（Wiener process）

### 28.1.2 Brownian 运动

Brownian 运动 $W_t$ 具有以下性质：

1. $W_0=0$
2. 对 $0\leq s<t$，增量 $W_t-W_s \sim \mathcal N(0,t-s)$
3. 不相交区间上的增量相互独立
4. 路径连续，但几乎处处不可微

它可以看作“随机游走在连续时间极限下的版本”。

### 28.1.3 二次变分与“$(dW)^2 = dt$”

Brownian 路径太粗糙，以至于经典微分法则不再直接适用。随机微积分里一个核心经验法则是：

$$
(dW_t)^2 = dt,\qquad dW_t\,dt=0,\qquad (dt)^2=0.
$$

它不是普通代数恒等式，而是对 Brownian 增量量级的浓缩表达。正因为有这条规则，Itô 公式会比普通链式法则多出一项二阶修正。

> **例题 28.1** 为什么说 Brownian 运动“处处连续但处处不可微”？

**解**：在极短时间步长 $\Delta t$ 上，Brownian 增量的标准差是 $\sqrt{\Delta t}$。若试图形成“导数”

$$
\frac{\Delta W}{\Delta t},
$$

其典型大小约为

$$
\frac{\sqrt{\Delta t}}{\Delta t}=\frac{1}{\sqrt{\Delta t}}\to \infty.
$$

这说明 Brownian 轨迹虽然连续，但局部波动太剧烈，几乎不可能拥有通常意义下的导数。$\square$

---

## 28.2 Itô 积分与 Itô 公式

### 28.2.1 为什么经典积分不够

对光滑曲线 $\gamma(t)$，积分

$$
\int f(t)\,d\gamma(t)
$$

可以按 Riemann-Stieltjes 方式理解。但对 Brownian 运动，由于路径总变差无限，经典构造会失效。因此需要专门定义 **Itô 积分**。

工程上可以把它理解为：在每个极小时间间隔里，用区间左端点的函数值乘以随机增量，再取极限。

### 28.2.2 Itô 公式

若过程满足

$$
dX_t = \mu(X_t,t)\,dt + \sigma(X_t,t)\,dW_t,
$$

且 $Y_t=f(X_t,t)$ 足够光滑，则

$$
dY_t
= \left(
\frac{\partial f}{\partial t}
+ \mu \frac{\partial f}{\partial x}
+ \frac12 \sigma^2 \frac{\partial^2 f}{\partial x^2}
\right)dt
+ \sigma \frac{\partial f}{\partial x}dW_t.
$$

与经典链式法则相比，多出来的就是

$$
\frac12 \sigma^2 f_{xx}\,dt.
$$

### 28.2.3 一个最经典的例子：$d(W_t^2)$

取 $f(x)=x^2$，则 $f'(x)=2x$，$f''(x)=2$。对 $X_t=W_t$ 应用 Itô 公式，且此时 $\mu=0,\sigma=1$，得到

$$
d(W_t^2)=2W_t\,dW_t + dt.
$$

这正是随机微积分中最著名的一条公式之一。

> **例题 28.2** 用 Itô 公式推导 $d(W_t^2)$。

**解**：由上面的代入，直接得到

$$
d(W_t^2)=2W_t\,dW_t + \frac12 \cdot 2 \cdot dt
=2W_t\,dW_t+dt.
$$

对两边取期望，并注意 Itô 积分期望为 0，可得

$$
\mathbb E[W_t^2]=t.
$$

这也与 Brownian 运动的方差定义一致。$\square$

### 28.2.4 常见 SDE：几何 Brownian 与 OU 过程

**几何 Brownian 运动**

$$
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t
$$

的显式解为

$$
S_t=S_0 \exp\left[\left(\mu-\frac{\sigma^2}{2}\right)t+\sigma W_t\right].
$$

它保证 $S_t>0$，常用于金融建模。

**Ornstein-Uhlenbeck（OU）过程**

$$
dX_t = -\theta X_t\,dt + \sigma\,dW_t
$$

则刻画“带噪声的均值回归”。在机器学习里，它经常作为扩散、噪声注入和连续时间稳定过程的基本原型。

> **例题 28.3** 求几何 Brownian 运动
> $$
> dS_t=\mu S_t\,dt+\sigma S_t\,dW_t
> $$
> 的显式解。

**解**：对 $S_t$ 取对数，令 $Y_t=\ln S_t$。由 Itô 公式，

$$
dY_t
=\frac{1}{S_t}dS_t-\frac12\frac{1}{S_t^2}(dS_t)^2
=\left(\mu-\frac{\sigma^2}{2}\right)dt+\sigma\,dW_t.
$$

两边从 $0$ 积到 $t$：

$$
Y_t-Y_0
=\left(\mu-\frac{\sigma^2}{2}\right)t+\sigma W_t.
$$

再指数化得到

$$
S_t
=S_0\exp\left[\left(\mu-\frac{\sigma^2}{2}\right)t+\sigma W_t\right].
$$

这也解释了为什么几何 Brownian 运动天然保持正值。$\square$

---

## 28.3 Fokker-Planck 方程：从轨迹到分布

SDE 描述的是单条随机轨迹，但我们很多时候更关心“所有粒子的分布如何演化”。这时对应的 PDE 就是 Fokker-Planck 方程。

对一维 SDE

$$
dX_t = \mu(X_t,t)\,dt + \sigma(X_t,t)\,dW_t,
$$

其密度 $p(x,t)$ 的演化满足

$$
\frac{\partial p}{\partial t}
= -\frac{\partial}{\partial x}(\mu p)
+ \frac12 \frac{\partial^2}{\partial x^2}(\sigma^2 p).
$$

它的含义是：

- 第一项由漂移引起，像“整体搬运”
- 第二项由扩散引起，像“热扩散”

### 28.3.1 平稳分布

若时间足够长后分布不再变化，则有

$$
\frac{\partial p}{\partial t}=0.
$$

这时 Fokker-Planck 方程退化为常微分方程，可用来寻找平稳分布。

对 OU 过程，平稳分布恰好是高斯分布，这使它成为理解扩散模型的一个理想玩具模型。

> **例题 28.4** 为什么 OU 过程会自然收敛到高斯型平稳分布？

**解**：漂移项 $-\theta X_t$ 会把状态拉回原点，扩散项 $\sigma dW_t$ 则不断注入噪声。两者平衡后，分布既不会无限扩张，也不会塌缩成点，而会稳定在一个以 0 为中心、方差有限的高斯分布上。严格推导可由 Fokker-Planck 方程完成。$\square$

---

## 28.4 扩散模型的数学框架

### 28.4.1 前向 SDE：不断加噪

扩散模型的前向过程可写成

$$
dx = f(x,t)\,dt + g(t)\,dW_t.
$$

在 DDPM 的常见设定中，

$$
f(x,t)=-\frac12 \beta(t)x,\qquad g(t)=\sqrt{\beta(t)}.
$$

它的作用是：随着时间推进，把真实数据分布逐渐扰动成接近标准高斯的噪声分布。

> **例题 28.5** 写出离散 DDPM 前向过程的单步闭式
> $$
> q(x_t\mid x_0).
> $$

**解**：离散扩散的单步更新常写成

$$
q(x_t\mid x_{t-1})
=\mathcal N\!\left(\sqrt{\alpha_t}\,x_{t-1},\ (1-\alpha_t)I\right),
\qquad \alpha_t=1-\beta_t.
$$

把这些高斯线性变换连乘后，可得

$$
q(x_t\mid x_0)
=\mathcal N\!\left(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\right),
\qquad
\bar\alpha_t=\prod_{s=1}^t \alpha_s.
$$

等价地，可以直接写成采样形式

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I).
$$

这条闭式是扩散模型训练能高效随机抽取任意时刻噪声样本的关键。$\square$

### 28.4.2 反向 SDE：从噪声中恢复数据

Anderson 反向时间理论告诉我们，若已知每个时刻的 score function

$$
\nabla_x \log p_t(x),
$$

则可以写出反向 SDE：

$$
dx = \left[f(x,t)-g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d\bar W_t.
$$

这就是“去噪生成”的核心。问题在于真实的 score 不知道，于是用神经网络 $s_\theta(x,t)$ 去拟合。

> **例题 28.6** 为什么反向 SDE 的漂移项里会出现
> $$
> -g(t)^2\nabla_x\log p_t(x) ?
> $$

**解**：前向扩散会把样本从高密度区域不断推向更模糊、更接近高斯噪声的分布。若要在时间反向时把轨迹“拉回来”，就必须知道当前点朝哪个方向更可能回到数据高密度区域。这个方向恰好由 score function

$$
\nabla_x\log p_t(x)
$$

给出，它总是指向对数密度上升最快的方向。系数 $g(t)^2$ 则反映了噪声强度越大，反向时需要补偿的漂移也越强。于是反向漂移并不是任意加上的修正项，而是由“抵消前向扩散 + 指回高密度区域”这两件事共同决定的。$\square$

### 28.4.3 Denoising Score Matching

在实践中，人们通常不直接训练网络输出 $\nabla_x\log p_t(x)$，而是训练它预测噪声 $\varepsilon$。这两者在高斯加噪设定下是等价的。

因此扩散模型训练的许多公式，看起来像“预测噪声”，本质上是在学习 score function。

### 28.4.4 概率流 ODE 与 DDIM

与反向 SDE 配对的，还有一个不含随机项的确定性 ODE：

$$
dx = \left[f(x,t)-\frac12 g(t)^2 \nabla_x \log p_t(x)\right]dt.
$$

它与原 SDE 拥有相同的边际分布，因此叫做**概率流 ODE（probability flow ODE）**。

DDIM 采样器正可以理解为对这条 ODE 的数值求解。这样一来：

- DDPM 更像随机采样
- DDIM 更像确定性积分

这也解释了“采样步数与质量之间的权衡”。

> **例题 28.7** 为什么 DDIM 可以被理解为 ODE 采样器？

**解**：因为在扩散模型的连续极限里，存在一条与原随机扩散过程共享同一边际分布的确定性 ODE。DDIM 通过在离散时间上选择特定更新路径，相当于对这条概率流 ODE 做数值积分，因此它是一个“确定性去噪采样器”。$\square$

---

## 28.5 SDE 与 AI 的前沿连接

### 28.5.1 Flow Matching

Flow Matching 直接学习一条从噪声分布到数据分布的连续向量场，绕开了部分 SDE 训练细节，但其思想仍与概率流 ODE 紧密相关。

### 28.5.2 Consistency Models

Consistency Models 的目标是减少多步积分的成本，让模型更接近“一步或少步直达终点”。这可以看作对 ODE/SDE 采样路径的进一步压缩。

### 28.5.3 Classifier-Free Guidance

条件生成里常用的 guidance，本质上是在 score function 上叠加一个方向修正，使采样过程更偏向指定条件。它并不是“拍脑袋加个系数”，而是对分布梯度的重新组合。

### 28.5.4 代码示例：模拟 Brownian 运动

```python
import numpy as np

def brownian_path(T=1.0, steps=1000):
    dt = T / steps
    dW = np.sqrt(dt) * np.random.randn(steps)
    W = np.concatenate([[0.0], np.cumsum(dW)])
    return W

path = brownian_path()
print("W(T) =", path[-1])
print("sample variance proxy =", np.var(np.diff(path)))
```

这个示例展示了 Brownian 增量的尺度大约是 $\sqrt{dt}$，而不是 $dt$。这是 Itô 公式中二阶修正项出现的根源。

> **例题 28.8** 如果把这个思路扩展成一个极简 1D 扩散实验，采样过程通常会呈现怎样的阶段性行为？

**解**：在训练完成后，从纯噪声开始反向迭代时，前几步主要负责把样本从“几乎各向同性的高斯噪声”拉回到大致正确的支撑区域，因此变化幅度大、结构粗；后几步则更多是在已有粗结构上细化局部位置和形状，因此看起来像“先搭轮廓，再补细节”。这也是为什么减少采样步数往往先损失细节，而不是立刻让样本完全失真。$\square$

---

## 本章小结

1. SDE 是“ODE + 噪声”。
2. Brownian 运动连续但不可微，导致经典链式法则失效。
3. Itô 公式比普通链式法则多出二阶修正项。
4. Fokker-Planck 方程描述的是分布的时间演化，而不是单条样本轨迹。
5. 扩散模型的前向/反向过程、本质上是 SDE 与 score function 的组合。
6. DDIM、概率流 ODE、Flow Matching 让扩散模型与数值微分方程紧密连接起来。

---

## 练习题

**1.** ⭐ 通过模拟验证 Brownian 运动在时间长度 $\Delta t$ 上的增量方差约为 $\Delta t$。

**2.** ⭐ 用 Itô 公式求 $d(e^{W_t})$。

**3.** ⭐ 写出 OU 过程
$$
dX_t=-\theta X_t\,dt+\sigma\,dW_t
$$
对应的 Fokker-Planck 方程。

**4.** ⭐⭐ 解释 DDPM 前向过程为什么在很多小步下可以写成单步高斯叠加。

**5.** ⭐⭐ 为什么扩散模型训练需要学习 score function，而不是直接学习数据分布本身？

**6.** ⭐⭐ 编程题：比较 Euler-Maruyama 与更小步长离散化对同一个一维 SDE 的采样误差。

**7.** ⭐⭐⭐ 解释 Classifier-Free Guidance 为什么可以看作对条件/无条件 score 的线性组合。

**8.** ⭐⭐⭐ 编程题：实现一个极简 1D 扩散过程，从高斯混合数据出发完成“加噪-去噪”实验。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.** 若用离散近似
$$
\Delta W_k = \sqrt{\Delta t}\,\varepsilon_k,\qquad \varepsilon_k\sim\mathcal N(0,1),
$$
则
$$
\mathrm{Var}(\Delta W_k)=\Delta t.
$$
用程序重复采样即可验证样本方差会接近这个理论值。

---

**2.** 对 $f(x)=e^x$，有 $f'(x)=e^x$，$f''(x)=e^x$。Itô 公式给出
$$
d(e^{W_t})
= e^{W_t}dW_t + \frac12 e^{W_t}dt.
$$

---

**3.** 对 OU 过程，$\mu(x,t)=-\theta x$，$\sigma(x,t)=\sigma$，因此
$$
\frac{\partial p}{\partial t}
= -\frac{\partial}{\partial x}(-\theta x p)
+ \frac{\sigma^2}{2}\frac{\partial^2 p}{\partial x^2}.
$$

---

**4.** 因为每一步加噪都是线性变换加高斯噪声，而高斯分布在线性变换与加法下封闭，所以多步叠加后仍是高斯。只不过其均值和方差会按照时间步累计变化。

---

**5.** 直接学习高维数据分布 $p_t(x)$ 很困难，而 score
$$
\nabla_x \log p_t(x)
$$
提供了“在当前位置朝哪里移动能到更高概率区域”的局部方向信息。反向 SDE/ODE 正是依赖这个方向场来逐步去噪。

---

**6.** Euler-Maruyama 是 SDE 最常见的离散方法：
$$
X_{n+1}=X_n+\mu(X_n,t_n)\Delta t+\sigma(X_n,t_n)\sqrt{\Delta t}\,\varepsilon_n.
$$
减小 $\Delta t$ 会降低离散误差，但会增加计算成本。SDE 采样器的“步数-质量”权衡，与 ODE 数值积分的思想一脉相承。

---

**7.** Classifier-Free Guidance 常写成
$$
s_{\text{guided}}
= (1+w)s_{\text{cond}} - w s_{\text{uncond}}.
$$
这本质上是在无条件 score 的基础上，沿着“条件信息带来的额外梯度方向”做放大，因此可以理解为对条件分布梯度的控制性修正。

---

**8.** 极简 1D 扩散实验通常包括：
1. 从简单的一维数据分布采样，例如双峰高斯混合；
2. 按时间逐步加噪，得到 $(x_t,t)$；
3. 训练一个小网络预测噪声或 score；
4. 从纯噪声反向积分回去，观察能否恢复双峰结构。

这个实验的重点不是生成质量，而是帮助理解“加噪-学方向-反向去噪”这条完整链路。

</details>
