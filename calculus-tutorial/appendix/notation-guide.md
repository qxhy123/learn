# 符号说明

本附录收录微积分及相关数学分支中常用的符号，配以中文解释与 LaTeX 写法，供学习参考。

---

## 一、集合符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $\in$ | `\in` | 属于；$x \in A$ 表示元素 $x$ 属于集合 $A$ |
| $\notin$ | `\notin` | 不属于；$x \notin A$ 表示 $x$ 不是 $A$ 的元素 |
| $\subset$ | `\subset` | 真子集（有时也用作子集）；$A \subset B$ 表示 $A$ 是 $B$ 的子集 |
| $\subseteq$ | `\subseteq` | 子集（含相等）；$A \subseteq B$ 表示 $A$ 是 $B$ 的子集或等于 $B$ |
| $\supset$ | `\supset` | 真超集 |
| $\supseteq$ | `\supseteq` | 超集（含相等） |
| $\cup$ | `\cup` | 并集；$A \cup B$ 包含 $A$ 或 $B$ 中的所有元素 |
| $\cap$ | `\cap` | 交集；$A \cap B$ 包含同时属于 $A$ 和 $B$ 的元素 |
| $\setminus$ | `\setminus` | 差集；$A \setminus B$ 包含属于 $A$ 但不属于 $B$ 的元素 |
| $A^c$ 或 $\complement A$ | `A^c` | 补集；全集中不属于 $A$ 的元素构成的集合 |
| $\emptyset$ | `\emptyset` | 空集；不含任何元素的集合 |
| $\vert A \vert$ 或 $\#A$ | `|A|` | 集合的基数（元素个数） |
| $\mathbb{N}$ | `\mathbb{N}` | 自然数集 $\{0, 1, 2, 3, \ldots\}$（部分文献从 $1$ 开始） |
| $\mathbb{Z}$ | `\mathbb{Z}` | 整数集 $\{\ldots, -2, -1, 0, 1, 2, \ldots\}$ |
| $\mathbb{Q}$ | `\mathbb{Q}` | 有理数集 |
| $\mathbb{R}$ | `\mathbb{R}` | 实数集 |
| $\mathbb{C}$ | `\mathbb{C}` | 复数集 |
| $[a, b]$ | `[a, b]` | 闭区间；$a \le x \le b$ |
| $(a, b)$ | `(a, b)` | 开区间；$a < x < b$ |
| $[a, b)$ | `[a, b)` | 半开区间；$a \le x < b$ |
| $(a, b]$ | `(a, b]` | 半开区间；$a < x \le b$ |
| $(-\infty, a]$ | `(-\infty, a]` | 无穷区间 |

---

## 二、逻辑符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $\forall$ | `\forall` | 全称量词，"对所有……"；$\forall x \in \mathbb{R}$ 表示"对所有实数 $x$" |
| $\exists$ | `\exists` | 存在量词，"存在……"；$\exists x$ 表示"存在某个 $x$" |
| $\exists!$ | `\exists!` | 唯一存在，"恰好存在一个" |
| $\neg$ 或 $\lnot$ | `\neg` | 否定，"非" |
| $\land$ | `\land` | 合取，"且"（逻辑与） |
| $\lor$ | `\lor` | 析取，"或"（逻辑或） |
| $\Rightarrow$ 或 $\implies$ | `\Rightarrow` | 蕴含，"若……则……"；$P \Rightarrow Q$ 表示 $P$ 成立则 $Q$ 成立 |
| $\Leftrightarrow$ 或 $\iff$ | `\Leftrightarrow` | 等价，"当且仅当"；$P \Leftrightarrow Q$ |
| $\therefore$ | `\therefore` | 所以，表示结论 |
| $\because$ | `\because` | 因为，表示原因 |
| $\square$ 或 $\blacksquare$ | `\square` | 证毕（QED） |

---

## 三、极限符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $\lim_{x \to a} f(x)$ | `\lim_{x \to a} f(x)` | $x$ 趋向 $a$ 时 $f(x)$ 的极限 |
| $\lim_{x \to a^+} f(x)$ | `\lim_{x \to a^+} f(x)` | $x$ 从右侧趋向 $a$ 的右极限 |
| $\lim_{x \to a^-} f(x)$ | `\lim_{x \to a^-} f(x)` | $x$ 从左侧趋向 $a$ 的左极限 |
| $\lim_{x \to +\infty} f(x)$ | `\lim_{x \to +\infty} f(x)` | $x$ 趋向正无穷时的极限 |
| $\lim_{x \to -\infty} f(x)$ | `\lim_{x \to -\infty} f(x)` | $x$ 趋向负无穷时的极限 |
| $\to$ | `\to` 或 `\rightarrow` | 趋向；$x \to a$ 表示 $x$ 趋近于 $a$ |
| $\infty$ | `\infty` | 无穷大 |
| $+\infty$ | `+\infty` | 正无穷大 |
| $-\infty$ | `-\infty` | 负无穷大 |
| $\limsup_{n \to \infty} a_n$ | `\limsup_{n \to \infty} a_n` | 上极限（上确界极限） |
| $\liminf_{n \to \infty} a_n$ | `\liminf_{n \to \infty} a_n` | 下极限（下确界极限） |
| $o(g)$ | `o(g)` | 小 $o$ 符号；$f = o(g)$ 表示 $f/g \to 0$ |
| $O(g)$ | `O(g)` | 大 $O$ 符号；$f = O(g)$ 表示 $\vert f/g \vert$ 有界 |
| $\Theta(g)$ | `\Theta(g)` | $\Theta$ 符号；$f$ 与 $g$ 同阶 |
| $f \sim g$ | `f \sim g` | 等价无穷小/大；$f/g \to 1$ |

---

## 四、微分符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $\dfrac{dy}{dx}$ | `\dfrac{dy}{dx}` | $y$ 关于 $x$ 的导数（Leibniz 记号） |
| $f'(x)$ | `f'(x)` | $f$ 在 $x$ 处的导数（Lagrange 记号） |
| $\dot{x}$ | `\dot{x}` | 关于时间 $t$ 的导数（Newton 记号），$\dot{x} = dx/dt$ |
| $f''(x)$，$f^{(n)}(x)$ | `f''(x)`，`f^{(n)}(x)` | 二阶导数、$n$ 阶导数 |
| $\dfrac{d^n y}{dx^n}$ | `\dfrac{d^n y}{dx^n}` | $n$ 阶导数（Leibniz 记号） |
| $\partial$ | `\partial` | 偏微分符号；$\dfrac{\partial f}{\partial x}$ 为 $f$ 对 $x$ 的偏导数 |
| $\dfrac{\partial^2 f}{\partial x \partial y}$ | `\dfrac{\partial^2 f}{\partial x \partial y}` | 混合偏导数 |
| $\nabla f$ | `\nabla f` | 梯度；$\nabla f = \left(\dfrac{\partial f}{\partial x_1}, \ldots, \dfrac{\partial f}{\partial x_n}\right)$ |
| $\nabla \cdot \mathbf{F}$ | `\nabla \cdot \mathbf{F}` | 散度；$\operatorname{div} \mathbf{F}$ |
| $\nabla \times \mathbf{F}$ | `\nabla \times \mathbf{F}` | 旋度；$\operatorname{curl} \mathbf{F}$ |
| $\nabla^2 f$ 或 $\Delta f$ | `\nabla^2 f`，`\Delta f` | Laplace 算子；$\Delta f = \sum \dfrac{\partial^2 f}{\partial x_i^2}$ |
| $\Delta x$ | `\Delta x` | 增量；$x$ 的变化量 $\Delta x = x_2 - x_1$ |
| $dx$ | `dx` | 微分；无穷小增量 |
| $Df$，$D_x f$ | `Df`，`D_x f` | 导数算子记号 |
| $\mathbf{J}_f$ 或 $J_f$ | `\mathbf{J}_f` | Jacobi 矩阵（雅可比矩阵） |
| $\mathbf{H}_f$ 或 $H_f$ | `\mathbf{H}_f` | Hesse 矩阵（黑塞矩阵） |

---

## 五、积分符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $\int f(x)\, dx$ | `\int f(x)\, dx` | 不定积分（原函数） |
| $\int_a^b f(x)\, dx$ | `\int_a^b f(x)\, dx` | 定积分；$f$ 在区间 $[a,b]$ 上的积分 |
| $\iint_D f(x,y)\, dA$ | `\iint_D f(x,y)\, dA` | 二重积分；$f$ 在区域 $D$ 上的积分 |
| $\iiint_V f\, dV$ | `\iiint_V f\, dV` | 三重积分；$f$ 在空间区域 $V$ 上的积分 |
| $\oint_C \mathbf{F} \cdot d\mathbf{r}$ | `\oint_C \mathbf{F} \cdot d\mathbf{r}` | 曲线积分（沿封闭曲线 $C$） |
| $\int_C f\, ds$ | `\int_C f\, ds` | 第一类曲线积分（对弧长的曲线积分） |
| $\int_C \mathbf{F} \cdot d\mathbf{r}$ | `\int_C \mathbf{F} \cdot d\mathbf{r}` | 第二类曲线积分（对坐标的曲线积分） |
| $\iint_S \mathbf{F} \cdot d\mathbf{S}$ | `\iint_S \mathbf{F} \cdot d\mathbf{S}` | 曲面积分（通量积分） |
| $\oiint_S$ | `\oiint_S` | 封闭曲面上的积分 |
| $F(x)\Big|_a^b$ | `F(x)\Big|_a^b` | 定积分求值记号；$= F(b) - F(a)$ |

---

## 六、向量符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $\mathbf{v}$ 或 $\vec{v}$ | `\mathbf{v}`，`\vec{v}` | 向量 $v$ |
| $\vert \mathbf{v} \vert$ 或 $\Vert \mathbf{v} \Vert$ | `|\mathbf{v}|`，`\|\mathbf{v}\|` | 向量的模（长度） |
| $\hat{\mathbf{u}}$ | `\hat{\mathbf{u}}` | 单位向量（$\vert\hat{\mathbf{u}}\vert = 1$） |
| $\mathbf{u} \cdot \mathbf{v}$ | `\mathbf{u} \cdot \mathbf{v}` | 点积（内积）；$\mathbf{u} \cdot \mathbf{v} = \vert\mathbf{u}\vert\vert\mathbf{v}\vert\cos\theta$ |
| $\mathbf{u} \times \mathbf{v}$ | `\mathbf{u} \times \mathbf{v}` | 叉积（外积）；结果为垂直于两向量的向量 |
| $\mathbf{i}, \mathbf{j}, \mathbf{k}$ | `\mathbf{i}, \mathbf{j}, \mathbf{k}` | 三维直角坐标系的标准基向量 |
| $\mathbf{e}_1, \mathbf{e}_2, \ldots$ | `\mathbf{e}_1, \mathbf{e}_2` | 一般基向量 |
| $\langle u, v \rangle$ 或 $(u, v)$ | `\langle u, v \rangle` | 内积（抽象内积空间） |
| $\mathbf{u} \otimes \mathbf{v}$ | `\mathbf{u} \otimes \mathbf{v}` | 张量积（外积） |
| $\operatorname{proj}_{\mathbf{b}} \mathbf{a}$ | `\operatorname{proj}_{\mathbf{b}} \mathbf{a}` | $\mathbf{a}$ 在 $\mathbf{b}$ 方向上的投影 |

---

## 七、其他常用符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $\approx$ | `\approx` | 约等于 |
| $\equiv$ | `\equiv` | 恒等于（或同余） |
| $\ne$ 或 $\neq$ | `\neq` | 不等于 |
| $\le$，$\ge$ | `\le`，`\ge` | 小于等于、大于等于 |
| $\ll$，$\gg$ | `\ll`，`\gg` | 远小于、远大于 |
| $\vert x \vert$ | `|x|` | 绝对值 |
| $\lfloor x \rfloor$ | `\lfloor x \rfloor` | 下取整（不超过 $x$ 的最大整数） |
| $\lceil x \rceil$ | `\lceil x \rceil` | 上取整（不小于 $x$ 的最小整数） |
| $\binom{n}{k}$ | `\binom{n}{k}` | 二项式系数，$\dfrac{n!}{k!(n-k)!}$ |
| $\sum_{i=1}^{n} a_i$ | `\sum_{i=1}^{n} a_i` | 求和符号 |
| $\prod_{i=1}^{n} a_i$ | `\prod_{i=1}^{n} a_i` | 连乘积符号 |
| $n!$ | `n!` | 阶乘；$n! = 1 \cdot 2 \cdots n$ |
| $e$ | `e` | 自然常数，$e \approx 2.71828$ |
| $\pi$ | `\pi` | 圆周率，$\pi \approx 3.14159$ |
| $i$ | `i` | 虚数单位，$i^2 = -1$ |
| $\log x$ | `\log x` | 对数（底数视语境而定，常用底 $e$ 或 $10$） |
| $\ln x$ | `\ln x` | 自然对数，以 $e$ 为底 |
| $\exp(x)$ 或 $e^x$ | `\exp(x)` | 指数函数 |
| $\sin, \cos, \tan$ | `\sin, \cos, \tan` | 三角函数（正弦、余弦、正切） |
| $\arcsin, \arccos, \arctan$ | `\arcsin, \arccos, \arctan` | 反三角函数 |
| $\max, \min$ | `\max, \min` | 最大值、最小值 |
| $\sup, \inf$ | `\sup, \inf` | 上确界、下确界 |
| $\operatorname{sgn}(x)$ | `\operatorname{sgn}(x)` | 符号函数；$x>0$ 时为 $1$，$x<0$ 时为 $-1$，$x=0$ 时为 $0$ |
| $f \circ g$ | `f \circ g` | 复合函数；$(f \circ g)(x) = f(g(x))$ |
| $f^{-1}$ | `f^{-1}` | 反函数或矩阵的逆 |
| $\mathbf{A}^\top$ 或 $\mathbf{A}^T$ | `\mathbf{A}^\top` | 矩阵转置 |
| $\det(\mathbf{A})$ 或 $\vert \mathbf{A} \vert$ | `\det(\mathbf{A})` | 矩阵行列式 |
| $\operatorname{tr}(\mathbf{A})$ | `\operatorname{tr}(\mathbf{A})` | 矩阵的迹（对角元素之和） |
| $\mathbb{1}_{A}$ | `\mathbb{1}_{A}` | 指示函数；$x \in A$ 时为 $1$，否则为 $0$ |

---

> 本附录所列符号以标准数学惯例为准。不同教材或领域偶有差异，阅读时请结合上下文判断具体含义。
