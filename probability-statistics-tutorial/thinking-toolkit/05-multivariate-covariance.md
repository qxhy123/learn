# 多元分布与协方差

> **一例速记**：设 $(X, Y)$ 的联合密度为 $f(x,y) = 6x$（$0 < x < 1,\; 0 < y < x$），求 $P(X > 1/2)$。
> 看到"联合密度 + 求边缘概率"→ 先积掉 $y$，得边缘密度 $f_X(x) = \int_0^x 6x\,dy = 6x^2$；
> 再对 $x$ 积分：$P(X > 1/2) = \int_{1/2}^1 6x^2\,dx = \left[2x^3\right]_{1/2}^1 = 2 - 1/4 = 7/8$。
> **"联合 → 边缘"靠对另一变量积分（求和），是多元分布一切计算的基础操作。**

---

## 一、为什么多元分布是概率论的"立体化"

单一随机变量只能描述一维不确定性，但现实问题往往涉及多个相互影响的量：股价与成交量、身高与体重、特征向量的各分量……**联合分布**把多个变量的整体行为打包描述，协方差则量化它们之间的线性关联强度。

掌握多元分布的三个核心操作——边缘化、条件化、独立判别——就能把任何多维问题拆解成可计算的结构。而协方差矩阵（以及它的推广：多元正态分布）是机器学习中处理高维数据的基石：主成分分析（PCA）的本质是对协方差矩阵做特征值分解。

---

## 二、联合分布：PMF / PDF / CDF

### 2.1 离散情形：联合 PMF

若 $X, Y$ 均为离散型，**联合概率质量函数**（PMF）定义为：

$$p(x, y) = P(X = x,\; Y = y)$$

满足：$p(x,y) \geq 0$，且 $\displaystyle\sum_x \sum_y p(x,y) = 1$。

### 2.2 连续情形：联合 PDF

若存在非负函数 $f(x,y)$ 使得对任意区域 $A$ 有

$$P((X,Y) \in A) = \iint_A f(x,y)\,dx\,dy,$$

则称 $f(x,y)$ 为 $(X,Y)$ 的**联合概率密度函数**（PDF）。

归一化条件：$\displaystyle\iint_{\mathbb{R}^2} f(x,y)\,dx\,dy = 1$。

### 2.3 联合 CDF

不论离散还是连续，**联合累积分布函数**（CDF）统一定义为：

$$F(x, y) = P(X \leq x,\; Y \leq y)$$

连续情形下 $f(x,y) = \dfrac{\partial^2 F}{\partial x \,\partial y}$（在 $F$ 的光滑点处）。

### 2.4 常见联合分布举例

| 类型 | 分布 | 核心特征 |
|---|---|---|
| 离散 | 多项分布 $\text{Multinomial}(n, p_1,\ldots,p_k)$ | $\sum X_i = n$，各分量为计数 |
| 连续 | 二元均匀分布 | 密度在某区域上为常数 |
| 连续 | 二元正态 $N(\boldsymbol\mu, \boldsymbol\Sigma)$ | 椭圆等高线，由均值向量与协方差矩阵完全确定 |

---

## 三、边缘分布：对另一变量"积掉"

从联合分布恢复单个变量的分布，称为**边缘化**（marginalization）。

**连续情形**：

$$f_X(x) = \int_{-\infty}^{+\infty} f(x,y)\,dy, \qquad f_Y(y) = \int_{-\infty}^{+\infty} f(x,y)\,dx.$$

**离散情形**：

$$p_X(x) = \sum_y p(x,y), \qquad p_Y(y) = \sum_x p(x,y).$$

**操作要点**：

1. 确认积分（求和）的**范围**——$y$ 的取值范围可能依赖 $x$（如上面速记例中 $0 < y < x$），此时上下限不是常数。
2. 积分后结果只是 $x$ 的函数，代入归一化检验：$\int f_X(x)\,dx = 1$。

**例**：设 $f(x,y) = e^{-y}$（$0 < x < 1,\; y > 0$）。

$$f_X(x) = \int_0^{+\infty} e^{-y}\,dy = 1 \quad (0 < x < 1).$$

$$f_Y(y) = \int_0^1 e^{-y}\,dx = e^{-y} \quad (y > 0).$$

$X \sim U(0,1)$，$Y \sim \text{Exp}(1)$，二者恰好独立（联合密度 $= f_X \cdot f_Y$）。

---

## 四、条件分布：在已知某变量值后更新信念

**条件密度**定义为：

$$f_{Y|X}(y \mid x) = \frac{f(x,y)}{f_X(x)}, \quad f_X(x) > 0.$$

直觉：把联合密度在竖线 $X = x$ 处"切一刀"，再归一化（除以 $f_X(x)$），得到 $Y$ 在已知 $X = x$ 时的分布。

**全概率公式的连续版本**：

$$f_Y(y) = \int f_{Y|X}(y \mid x)\, f_X(x)\,dx.$$

**条件期望**：

$$E(Y \mid X = x) = \int y\, f_{Y|X}(y \mid x)\,dy.$$

注意：$E(Y \mid X)$ 是 $X$ 的函数，本身也是随机变量。重期望公式：$E[E(Y \mid X)] = E(Y)$。

| 操作 | 公式 | 含义 |
|---|---|---|
| 边缘密度 | $f_X(x) = \int f(x,y)\,dy$ | 忽略 $Y$，只看 $X$ 的分布 |
| 条件密度 | $f_{Y\mid X}(y\mid x) = f(x,y)/f_X(x)$ | 在已知 $X=x$ 后，$Y$ 的分布 |
| 还原联合 | $f(x,y) = f_{Y\mid X}(y\mid x)\cdot f_X(x)$ | 条件 × 边缘 = 联合 |

---

## 五、独立性判别

$X$ 与 $Y$ **独立**当且仅当：

$$f(x,y) = f_X(x)\cdot f_Y(y) \quad \text{（对所有 } (x,y) \text{）.}$$

**实用判别技巧**：

1. **因式分解法**：若 $f(x,y)$ 能分解为只含 $x$ 的因子乘以只含 $y$ 的因子，**且支撑集（$f > 0$ 的区域）是矩形**，则独立。
   - 例：$f(x,y) = e^{-x-y}$（$x, y > 0$）$= e^{-x} \cdot e^{-y}$，支撑是矩形 $(0,\infty)^2$，故独立。
2. **支撑非矩形则必不独立**：若支撑集依赖两个变量（如 $0 < y < x < 1$），则 $X, Y$ 不独立。
3. **协方差为零 $\neq$ 独立**：协方差只度量线性关系，$\text{Cov}(X,Y) = 0$ 不能推出独立（反例：$Y = X^2$）。

---

## 六、协方差与协方差矩阵

### 6.1 协方差

$$\text{Cov}(X, Y) = E[(X - E X)(Y - E Y)] = E(XY) - E(X)\, E(Y).$$

**性质**：

- $\text{Cov}(X, X) = \text{Var}(X)$
- 对称性：$\text{Cov}(X,Y) = \text{Cov}(Y,X)$
- 双线性：$\text{Cov}(aX + b, cY + d) = ac\,\text{Cov}(X,Y)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$

**相关系数**（无量纲化的协方差）：

$$\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)}\sqrt{\text{Var}(Y)}}, \quad |\rho_{XY}| \leq 1.$$

$\rho = 1$ 表示完全正线性相关，$\rho = 0$ 表示线性不相关（不一定独立），$\rho = -1$ 表示完全负线性相关。

### 6.2 协方差矩阵

对随机向量 $\mathbf{X} = (X_1, X_2, \ldots, X_p)^\top$，**协方差矩阵**（也称方差-协方差矩阵）为 $p \times p$ 矩阵：

$$\boldsymbol\Sigma = \text{Cov}(\mathbf{X}) = E\!\left[(\mathbf{X} - \boldsymbol\mu)(\mathbf{X} - \boldsymbol\mu)^\top\right],$$

其中 $\boldsymbol\mu = E\mathbf{X}$。矩阵元素 $\Sigma_{ij} = \text{Cov}(X_i, X_j)$，对角元 $\Sigma_{ii} = \text{Var}(X_i)$。

**基本性质**：
- 对称正半定（PSD）：$\boldsymbol\Sigma = \boldsymbol\Sigma^\top$，且 $\mathbf{v}^\top \boldsymbol\Sigma\, \mathbf{v} \geq 0$ 对所有 $\mathbf{v}$。
- 线性变换：若 $\mathbf{Y} = A\mathbf{X} + \mathbf{b}$，则 $\text{Cov}(\mathbf{Y}) = A\boldsymbol\Sigma A^\top$。
- 正定时可逆（各分量不存在严格线性相关）。

---

## 七、多元正态分布 $N(\boldsymbol\mu, \boldsymbol\Sigma)$

### 7.1 定义

$p$ 维随机向量 $\mathbf{X}$ 服从**多元正态分布** $N(\boldsymbol\mu, \boldsymbol\Sigma)$（$\boldsymbol\Sigma$ 正定），其联合 PDF 为：

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2}|\boldsymbol\Sigma|^{1/2}}\exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol\mu)^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\right).$$

指数部分 $(\mathbf{x}-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)$ 称为**马氏距离**（Mahalanobis distance）的平方，等高线为以 $\boldsymbol\mu$ 为中心的椭球面。

### 7.2 二元正态的特殊情形

设 $(X,Y) \sim N(\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho)$，联合 PDF 为：

$$f(x,y) = \frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}} \exp\!\left\{-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_1)^2}{\sigma_1^2} - \frac{2\rho(x-\mu_1)(y-\mu_2)}{\sigma_1\sigma_2} + \frac{(y-\mu_2)^2}{\sigma_2^2}\right]\right\}.$$

**关键性质**：

| 性质 | 结论 |
|---|---|
| 边缘分布 | $X \sim N(\mu_1, \sigma_1^2)$，$Y \sim N(\mu_2, \sigma_2^2)$ |
| $\rho = 0$ 时 | $X, Y$ 不相关 $\Leftrightarrow$ $X, Y$ 独立（仅多元正态成立！）|
| 条件分布 | $Y \mid X=x \sim N\!\left(\mu_2 + \rho\frac{\sigma_2}{\sigma_1}(x-\mu_1),\; \sigma_2^2(1-\rho^2)\right)$ |
| 线性组合 | $aX + bY \sim N(a\mu_1 + b\mu_2,\; a^2\sigma_1^2 + 2ab\rho\sigma_1\sigma_2 + b^2\sigma_2^2)$ |

---

## 八、演示题：证明二元正态的条件分布仍是正态

**题目**：设 $(X,Y) \sim N(\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho)$，证明 $Y \mid X = x$ 服从正态分布，并给出均值与方差。

> **目标分析。** 条件密度 $= f(x,y)/f_X(x)$。$f_X(x)$ 是一元正态密度，只需把联合密度除以它，再整理成正态密度的标准形式。
>
> **第一步：写出 $f_X(x)$。**
>
> $$f_X(x) = \frac{1}{\sqrt{2\pi}\,\sigma_1}\exp\!\left(-\frac{(x-\mu_1)^2}{2\sigma_1^2}\right).$$
>
> **第二步：计算条件密度。**
>
> $$f_{Y|X}(y \mid x) = \frac{f(x,y)}{f_X(x)}.$$
>
> 分子 $f(x,y)$ 的指数部分展开为：
>
> $$-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_1)^2}{\sigma_1^2} - \frac{2\rho(x-\mu_1)(y-\mu_2)}{\sigma_1\sigma_2} + \frac{(y-\mu_2)^2}{\sigma_2^2}\right].$$
>
> 除以 $f_X(x)$ 等价于在指数里减去 $-\dfrac{(x-\mu_1)^2}{2\sigma_1^2}$。整理剩余的指数：
>
> $$-\frac{1}{2(1-\rho^2)}\left[\frac{(y-\mu_2)^2}{\sigma_2^2} - \frac{2\rho(x-\mu_1)(y-\mu_2)}{\sigma_1\sigma_2}\right] + \frac{(x-\mu_1)^2}{2\sigma_1^2}\cdot\frac{\rho^2}{1-\rho^2} - \frac{(x-\mu_1)^2}{2\sigma_1^2(1-\rho^2)}.$$
>
> 注意带 $(x-\mu_1)^2$ 的项只依赖 $x$，不含 $y$，它们在对 $y$ 积分时是常数（最终被归一化吸收）。
>
> **第三步：配方。** 关键在含 $y$ 的项。令 $\mu_{y|x} = \mu_2 + \rho\dfrac{\sigma_2}{\sigma_1}(x - \mu_1)$，$\sigma_{y|x}^2 = \sigma_2^2(1-\rho^2)$。计算：
>
> $$\frac{(y-\mu_2)^2}{\sigma_2^2} - \frac{2\rho(x-\mu_1)(y-\mu_2)}{\sigma_1\sigma_2} = \frac{1}{\sigma_2^2}\left(y - \mu_2 - \rho\frac{\sigma_2}{\sigma_1}(x-\mu_1)\right)^2 - \frac{\rho^2(x-\mu_1)^2}{\sigma_1^2}.$$
>
> 最后一项 $-\rho^2(x-\mu_1)^2/\sigma_1^2$ 和前面的 $(x-\mu_1)^2$ 相关项合并，与 $y$ 无关，进入归一化常数。
>
> **第四步：识别正态形式。** 含 $y$ 的部分为：
>
> $$-\frac{1}{2(1-\rho^2)\sigma_2^2}\left(y - \mu_{y|x}\right)^2 = -\frac{(y-\mu_{y|x})^2}{2\sigma_{y|x}^2}.$$
>
> 这正是正态密度的指数形式。
>
> **结论**：
>
> $$Y \mid X = x \;\sim\; N\!\left(\mu_2 + \rho\frac{\sigma_2}{\sigma_1}(x-\mu_1),\;\; \sigma_2^2(1-\rho^2)\right).$$
>
> **回顾。** 条件均值 $\mu_{y|x}$ 关于 $x$ 是线性的（斜率 $\rho\sigma_2/\sigma_1$），这正是线性回归的理论基础。条件方差 $\sigma_2^2(1-\rho^2) < \sigma_2^2$，相关性越强（$|\rho|$ 越大），知道 $X$ 后对 $Y$ 的不确定性减少越多。

---

## 九、思考路标

**路标 1**：看到"联合密度 + 求某单一变量的概率或分布"→ 先对另一个变量**积分（求和）**，得到边缘密度，再做后续运算。

**路标 2**：看到"已知 $X=x$，求 $Y$ 的分布"→ 条件密度 $f_{Y|X}(y|x) = f(x,y)/f_X(x)$；分子是联合密度，分母是边缘密度。

**路标 3**：判断独立性时，第一步看支撑集是否是矩形。支撑集含交叉约束（如 $0 < y < x$）→ 直接判定不独立，无需计算协方差。

**路标 4**：协方差为零不等于独立。只有在**多元正态**分布中，不相关才等价于独立。其他情形下，$\text{Cov}(X,Y)=0$ 不能推出 $X, Y$ 独立。

**路标 5**：看到协方差矩阵 $\boldsymbol\Sigma$ 和线性变换 $\mathbf{Y} = A\mathbf{X}$→ 公式 $\text{Cov}(\mathbf{Y}) = A\boldsymbol\Sigma A^\top$。这是 PCA、白化（whitening）等数据处理操作的数学核心。

**路标 6**：多元正态的条件分布、边缘分布仍是正态。见到"二元正态 + 条件 / 边缘"→ 直接套对应公式，无需从头积分。

**路标 7**：二元正态密度的"配方"步骤是研究生考试的高频考点。核心是把含 $y$ 的二次型配成 $(y - \text{线性函数}(x))^2/\text{常数}$，剩下与 $y$ 无关的项进入归一化常数。

---

## 十、典型应用 3 例

### 例 1：求边缘密度与独立性判别

**题目**：设 $(X, Y)$ 的联合密度为 $f(x,y) = \begin{cases} 2e^{-x-2y}, & x > 0,\; y > 0\\ 0, & \text{其他} \end{cases}$，判断 $X, Y$ 是否独立。

**分析**：

$$f_X(x) = \int_0^{+\infty} 2e^{-x-2y}\,dy = 2e^{-x}\cdot\frac{1}{2} = e^{-x} \quad (x>0).$$

$$f_Y(y) = \int_0^{+\infty} 2e^{-x-2y}\,dx = 2e^{-2y} \quad (y>0).$$

$$f_X(x)\cdot f_Y(y) = e^{-x}\cdot 2e^{-2y} = 2e^{-x-2y} = f(x,y).$$

支撑集为矩形 $(0,+\infty)^2$，且联合密度等于边缘密度之积，故 $X, Y$ **独立**。

---

### 例 2：条件期望与线性回归

**题目**：$(X, Y)$ 服从二元正态 $N(0, 0, 1, 4, 0.8)$（$\mu_1=\mu_2=0$，$\sigma_1=1$，$\sigma_2=2$，$\rho=0.8$），求 $E(Y \mid X = 1)$ 和 $\text{Var}(Y \mid X = 1)$。

**分析**：

$$E(Y \mid X=1) = \mu_2 + \rho\frac{\sigma_2}{\sigma_1}(1 - \mu_1) = 0 + 0.8 \times \frac{2}{1} \times 1 = 1.6.$$

$$\text{Var}(Y \mid X=1) = \sigma_2^2(1 - \rho^2) = 4(1 - 0.64) = 4 \times 0.36 = 1.44.$$

解读：知道 $X=1$ 后，对 $Y$ 的最佳预测是 $1.6$，不确定性从原来的 $\sigma_2^2 = 4$ 减少到 $1.44$，减少了 $64\%$（正好是 $\rho^2 = 0.64$）。

---

### 例 3：协方差矩阵与线性变换（PCA 预热）

**题目**：设 $\mathbf{X} = (X_1, X_2)^\top$，协方差矩阵 $\boldsymbol\Sigma = \begin{pmatrix} 4 & 2 \\ 2 & 1 \end{pmatrix}$。令 $Y = X_1 - 2X_2$（即 $A = (1, -2)$），求 $\text{Var}(Y)$。

**分析**：

$$\text{Var}(Y) = A\boldsymbol\Sigma A^\top = (1, -2)\begin{pmatrix} 4 & 2 \\ 2 & 1 \end{pmatrix}\begin{pmatrix} 1 \\ -2 \end{pmatrix}.$$

先算 $A\boldsymbol\Sigma = (1\cdot4 + (-2)\cdot2,\; 1\cdot2 + (-2)\cdot1) = (0, 0)$，因此 $\text{Var}(Y) = 0$。

这意味着 $Y = X_1 - 2X_2$ 几乎必然为常数——$X_1$ 与 $X_2$ 之间存在严格的线性依赖关系（$X_1 = 2X_2$ a.s.），协方差矩阵奇异（行列式为零：$4\times1 - 2\times2 = 0$）。

---

## 十一、自测题

**第 1 题**：设联合密度 $f(x,y) = c(x+y)$（$0 < x < 1,\; 0 < y < 1$），求常数 $c$，以及 $f_X(x)$ 和 $f_Y(y)$。

> 提示：由 $\int_0^1\int_0^1 c(x+y)\,dx\,dy = c\cdot 1 = 1$ 得 $c = 1$；$f_X(x) = \int_0^1(x+y)\,dy = x + 1/2$；$f_Y(y) = y + 1/2$。验证：$f(x,y) \neq f_X(x)\cdot f_Y(y)$，故 $X,Y$ 不独立。

**第 2 题**：设 $(X,Y)$ 的联合 PMF 为 $p(x,y) = \dfrac{e^{-2}}{x!\,(y-x)!}$（$x = 0,1,\ldots,y$；$y = 0,1,2,\ldots$）。求 $X$ 和 $Y$ 的边缘分布。

> 提示：固定 $x$，对 $y$ 从 $x$ 到 $\infty$ 求和；固定 $y$，对 $x$ 从 $0$ 到 $y$ 求和。最终 $X \sim \text{Poisson}(1)$，$Y \sim \text{Poisson}(2)$。

**第 3 题**：设 $X \sim N(0,1)$，$Y = X^2$。计算 $\text{Cov}(X, Y)$，并说明 $X, Y$ 是否独立。

> 提示：$\text{Cov}(X,Y) = E(X\cdot X^2) - E(X)\cdot E(X^2) = E(X^3) - 0 = 0$（奇函数）。但 $Y$ 完全由 $X$ 确定，显然不独立。这是协方差为零不等于独立的经典例子。

**第 4 题**：$(X,Y) \sim N(1, -1, 4, 9, -0.5)$，求 $P(Y < -1 + X/2)$。

> 提示：令 $Z = Y - (-1 + X/2) = Y + 1 - X/2$。$Z$ 是二元正态的线性组合，故 $Z \sim N$；计算 $E(Z) = -1+1 - (-0.5)\cdot(3/2\cdot 1)\cdot \ldots$ 先用 $E(Z) = E(Y) + 1 - E(X)/2 = -1+1-1/2 = -1/2$，$\text{Var}(Z) = \text{Var}(Y) + \text{Var}(X)/4 - \text{Cov}(X,Y) = 9 + 1 - (-3) = 13$。$P(Z < 0) = \Phi(1/2/\sqrt{13})$。

**第 5 题**：设随机向量 $\mathbf{X}$ 的协方差矩阵为 $\boldsymbol\Sigma$，令 $\mathbf{Y} = \boldsymbol\Sigma^{-1/2}(\mathbf{X} - \boldsymbol\mu)$（白化变换），证明 $\text{Cov}(\mathbf{Y}) = I$（单位矩阵）。

> 提示：$\text{Cov}(\mathbf{Y}) = \boldsymbol\Sigma^{-1/2}\boldsymbol\Sigma(\boldsymbol\Sigma^{-1/2})^\top = \boldsymbol\Sigma^{-1/2}\boldsymbol\Sigma\boldsymbol\Sigma^{-1/2} = I$（利用 $\boldsymbol\Sigma^{1/2}$ 对称正定）。
