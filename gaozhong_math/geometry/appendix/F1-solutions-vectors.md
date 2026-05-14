# 附录 F1：向量类详解（平面向量 + 空间向量）

> 本附录给出附录 D 中向量类题目的完整解答，共 **25 题**，分布如下：
> - **平面向量综合**（D.01–D.10，对应 Part 1+2，共 10 题）
> - **空间向量**（D.73–D.87，对应 Part 9，共 15 题）
>
> **每题格式**：题目回顾 / 思路 / 解答（含关键步骤旁注）/ 答案 / 总结。
>
> **toolkit 引用说明**：
> - → toolkit/08 向量两面：数量积公式 $\vec{a}\cdot\vec{b}=|\vec{a}||\vec{b}|\cos\theta$，以及基底分解法；
> - → toolkit/01 四把剑：数形结合、结构识别、换元、构造；
> - → toolkit/11 立体向量：建系 → 求坐标 → 法向量 → 三类角 / 三类距离。

---

## 平面向量综合（D.01–D.10）

---

## D.01 [中档] Part 1/02

**题目回顾**：已知非零向量 $\vec{a}, \vec{b}$ 满足 $|\vec{a}| = 2,\ |\vec{b}| = 3,\ \vec{a} \cdot \vec{b} = -3$。求 $\vec{a}$ 与 $\vec{b}$ 的夹角 $\theta$，以及 $|2\vec{a} - \vec{b}|$ 的值。

**思路**：夹角公式 $\cos\theta = \dfrac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|}$ 直接代入；$|2\vec{a}-\vec{b}|$ 用平方展开公式 $|\vec{u}|^2 = \vec{u}\cdot\vec{u}$。→ toolkit/08 向量两面。

**解答**：

**第一步**：求夹角 $\theta$。

$$\cos\theta = \frac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|} = \frac{-3}{2\times 3} = -\frac{1}{2}$$

因 $\theta \in [0°, 180°]$，故 $\theta = 120°$。← 数量积法

**第二步**：求 $|2\vec{a} - \vec{b}|$。

$$|2\vec{a}-\vec{b}|^2 = 4|\vec{a}|^2 - 4\vec{a}\cdot\vec{b} + |\vec{b}|^2 = 4\times 4 - 4\times(-3) + 9 = 16+12+9 = 37$$

$$\therefore |2\vec{a}-\vec{b}| = \sqrt{37}$$

← 平方展开

**答案**：$\theta = \boxed{120°}$，$|2\vec{a}-\vec{b}| = \boxed{\sqrt{37}}$。

**总结**：见"已知两向量模与数量积，求夹角或组合向量的模" → 夹角公式 + 平方展开，两步即可。

---

## D.02 [中档] Part 1/03

**题目回顾**：设 $\vec{e}_1, \vec{e}_2$ 是两个不共线的单位向量，夹角为 $60°$。若 $\vec{a} = 2\vec{e}_1 + \vec{e}_2$，$\vec{b} = \vec{e}_1 - \vec{e}_2$，求 $\vec{a} \cdot \vec{b}$ 的值，并判断 $\vec{a}$ 与 $\vec{b}$ 是否垂直。

**思路**：先算基底向量之间的数量积 $\vec{e}_1\cdot\vec{e}_2 = |\vec{e}_1||\vec{e}_2|\cos 60° = \frac{1}{2}$，再展开 $\vec{a}\cdot\vec{b}$。→ toolkit/08 向量两面（基底法）。

**解答**：

已知 $|\vec{e}_1| = |\vec{e}_2| = 1$，夹角 $60°$，故：

$$\vec{e}_1\cdot\vec{e}_1 = 1,\quad \vec{e}_2\cdot\vec{e}_2 = 1,\quad \vec{e}_1\cdot\vec{e}_2 = 1\times 1\times\cos 60° = \frac{1}{2}$$

← 基底数量积

展开 $\vec{a}\cdot\vec{b}$：

$$\vec{a}\cdot\vec{b} = (2\vec{e}_1+\vec{e}_2)\cdot(\vec{e}_1-\vec{e}_2) = 2|\vec{e}_1|^2 - 2\vec{e}_1\cdot\vec{e}_2 + \vec{e}_1\cdot\vec{e}_2 - |\vec{e}_2|^2$$

$$= 2\times 1 - 2\times\frac{1}{2} + \frac{1}{2} - 1 = 2 - 1 + \frac{1}{2} - 1 = \frac{1}{2}$$

← 分配律展开

因 $\vec{a}\cdot\vec{b} = \dfrac{1}{2} \neq 0$，所以 $\vec{a}$ 与 $\vec{b}$ **不垂直**。

**答案**：$\vec{a}\cdot\vec{b} = \boxed{\dfrac{1}{2}}$，$\vec{a}$ 与 $\vec{b}$ 不垂直。

**总结**：非标准正交基底时，展开前先用夹角公式算好 $\vec{e}_1\cdot\vec{e}_2$，再展开；勿漏交叉项。

---

## D.03 [中档] Part 1/04

**题目回顾**：已知向量 $\vec{a} = (2, -1)$，$\vec{b} = (1, k)$。若 $\vec{a} \parallel \vec{b}$，求 $k$；若 $\vec{a} \perp \vec{b}$，求 $k$。

**思路**：平行用坐标判定 $x_1 y_2 - x_2 y_1 = 0$；垂直用数量积 $= 0$。→ Part 1/03 共线与垂直的坐标判定。

**解答**：

**情形1：$\vec{a} \parallel \vec{b}$**

$$x_1 y_2 - x_2 y_1 = 2k - 1\times(-1) = 2k+1 = 0$$

$$\Rightarrow k = -\frac{1}{2}$$

← 共线坐标法

**情形2：$\vec{a} \perp \vec{b}$**

$$\vec{a}\cdot\vec{b} = 2\times 1 + (-1)\times k = 2 - k = 0$$

$$\Rightarrow k = 2$$

← 垂直点积法

**答案**：$\vec{a} \parallel \vec{b}$ 时 $k = \boxed{-\dfrac{1}{2}}$；$\vec{a} \perp \vec{b}$ 时 $k = \boxed{2}$。

**总结**："平行"和"垂直"是两种判定方式：平行用行列式，垂直用点积，分清不混淆。

---

## D.04 [中档] Part 2/01

**题目回顾**：在 $\triangle ABC$ 中，$\overrightarrow{AB} = \vec{b}$，$\overrightarrow{AC} = \vec{c}$。设 $G$ 为重心，用 $\vec{b}, \vec{c}$ 表示 $\overrightarrow{AG}$；设 $M$ 为 $BC$ 中点，证明 $A, G, M$ 三点共线。

**思路**：重心是三条中线的交点，$\overrightarrow{AG} = \frac{1}{3}(\overrightarrow{AB}+\overrightarrow{AC})$（重心坐标）；共线证明：验证 $\overrightarrow{AG}$ 与 $\overrightarrow{AM}$ 共线（一个是另一个的实数倍）。→ toolkit/08 向量两面（基底分解）。

**解答**：

**第一步**：求 $\overrightarrow{AG}$。

$M$ 为 $BC$ 中点，则 $\overrightarrow{AM} = \dfrac{1}{2}(\overrightarrow{AB}+\overrightarrow{AC}) = \dfrac{1}{2}(\vec{b}+\vec{c})$。← 中点向量

由重心性质，$G$ 在中线 $AM$ 上且 $AG:GM = 2:1$，故：

$$\overrightarrow{AG} = \frac{2}{3}\overrightarrow{AM} = \frac{2}{3}\cdot\frac{1}{2}(\vec{b}+\vec{c}) = \frac{1}{3}(\vec{b}+\vec{c})$$

← 重心分中线 $2:1$

**第二步**：证明 $A, G, M$ 三点共线。

$$\overrightarrow{AG} = \frac{1}{3}(\vec{b}+\vec{c}) = \frac{2}{3}\cdot\frac{1}{2}(\vec{b}+\vec{c}) = \frac{2}{3}\overrightarrow{AM}$$

即 $\overrightarrow{AG} = \dfrac{2}{3}\overrightarrow{AM}$，两向量共线且 $A$ 点相同，故 $A, G, M$ 三点共线。

← 共线判定

**答案**：$\overrightarrow{AG} = \boxed{\dfrac{1}{3}(\vec{b}+\vec{c})}$；$A, G, M$ 三点共线，证毕。

**总结**：重心公式 $\overrightarrow{AG}=\frac{1}{3}(\overrightarrow{AB}+\overrightarrow{AC})$ 是高频结论，需记忆；三点共线等价于两向量成比例。

---

## D.05 [中档] Part 2/02

**题目回顾**：设点 $P$ 在线段 $AB$ 上，且 $\overrightarrow{AP} = \lambda \overrightarrow{AB}$（$0 < \lambda < 1$）。已知 $A(1, 2)$，$B(5, 6)$，$P(3, 4)$，求 $\lambda$ 的值。又：若 $P$ 是 $AB$ 的三等分点（靠近 $A$），求 $P$ 的坐标。

**思路**：分点比参数化：$P = A + \lambda(B - A)$，坐标分量代入直接算 $\lambda$；三等分点 $\lambda = \frac{1}{3}$ 代入求坐标。→ Part 2/01 向量分点公式。

**解答**：

**第一步**：求 $\lambda$。

$\overrightarrow{AB} = B - A = (4, 4)$，$\overrightarrow{AP} = P - A = (2, 2)$。

$$\overrightarrow{AP} = \lambda\overrightarrow{AB} \Rightarrow (2,2) = \lambda(4,4) \Rightarrow \lambda = \frac{1}{2}$$

← 坐标对比

**第二步**：三等分点（靠近 $A$）坐标。

$\lambda = \dfrac{1}{3}$，则：

$$P = A + \frac{1}{3}\overrightarrow{AB} = (1,2) + \frac{1}{3}(4,4) = \left(1+\frac{4}{3},\ 2+\frac{4}{3}\right) = \left(\frac{7}{3},\ \frac{10}{3}\right)$$

← 向量分点

**答案**：$\lambda = \boxed{\dfrac{1}{2}}$；三等分点 $P = \boxed{\left(\dfrac{7}{3},\ \dfrac{10}{3}\right)}$。

**总结**：$\overrightarrow{AP} = \lambda\overrightarrow{AB}$ 是分点参数化的核心公式：$\lambda$ 即 $P$ 在 $AB$ 上的位置比例。

---

## D.06 [中档] Part 2/03

**题目回顾**：在平行四边形 $ABCD$ 中，$\overrightarrow{AB} = \vec{a}$，$\overrightarrow{AD} = \vec{b}$，$|\vec{a}| = 2,\ |\vec{b}| = \sqrt{2}$，$\vec{a} \cdot \vec{b} = 1$。（1）求对角线 $\overrightarrow{AC}$ 与 $\overrightarrow{BD}$ 的长；（2）证明两条对角线不垂直。

**思路**：平行四边形两对角线分别为 $\vec{a}+\vec{b}$ 和 $\vec{b}-\vec{a}$（注意方向），用平方展开求模；不垂直等价于两向量数量积 $\neq 0$。→ toolkit/08 向量两面。

**解答**：

**第一步**：对角线向量。

平行四边形中 $\overrightarrow{AC} = \overrightarrow{AB}+\overrightarrow{AD} = \vec{a}+\vec{b}$，$\overrightarrow{BD} = \overrightarrow{AD}-\overrightarrow{AB} = \vec{b}-\vec{a}$。← 向量加减

**第二步**：求 $|\overrightarrow{AC}|$。

$$|\vec{a}+\vec{b}|^2 = |\vec{a}|^2 + 2\vec{a}\cdot\vec{b} + |\vec{b}|^2 = 4 + 2\times 1 + 2 = 8$$

$$|\overrightarrow{AC}| = 2\sqrt{2}$$

**第三步**：求 $|\overrightarrow{BD}|$。

$$|\vec{b}-\vec{a}|^2 = |\vec{b}|^2 - 2\vec{a}\cdot\vec{b} + |\vec{a}|^2 = 2 - 2 + 4 = 4$$

$$|\overrightarrow{BD}| = 2$$

**第四步**：证明不垂直。

$$\overrightarrow{AC}\cdot\overrightarrow{BD} = (\vec{a}+\vec{b})\cdot(\vec{b}-\vec{a}) = |\vec{b}|^2 - |\vec{a}|^2 = 2 - 4 = -2 \neq 0$$

← 数量积判垂直

故两对角线不垂直，证毕。

**答案**：$|\overrightarrow{AC}| = \boxed{2\sqrt{2}}$，$|\overrightarrow{BD}| = \boxed{2}$；两对角线不垂直（数量积 $= -2 \neq 0$）。

**总结**：平行四边形对角线 $= \vec{a}\pm\vec{b}$；两对角线垂直当且仅当 $|\vec{a}| = |\vec{b}|$（菱形条件）。

---

## D.07 [中档] Part 2/03

**题目回顾**：已知 $\vec{a} = (1, -1)$，$\vec{b} = (2, 1)$，$\vec{c} = (3, k)$。若 $\vec{c}$ 可以由 $\vec{a}, \vec{b}$ 线性表示，求 $k$ 的值，并写出表示式。

**思路**：设 $\vec{c} = \lambda\vec{a} + \mu\vec{b}$，列方程组解 $\lambda, \mu$；两个方程两个未知数，有唯一解（因 $\vec{a},\vec{b}$ 不共线）。→ Part 2/03 基底分解。

**解答**：

设 $\vec{c} = \lambda\vec{a} + \mu\vec{b}$，即 $(3, k) = \lambda(1,-1) + \mu(2,1) = (\lambda+2\mu, -\lambda+\mu)$。

建立方程组：

$$\begin{cases} \lambda + 2\mu = 3 \\ -\lambda + \mu = k \end{cases}$$

← 分量对应

两式相加：$3\mu = 3 + k$，得 $\mu = \dfrac{3+k}{3}$。

代回第一式：$\lambda = 3 - 2\mu = 3 - \dfrac{2(3+k)}{3} = \dfrac{9-6-2k}{3} = \dfrac{3-2k}{3}$。

验证第二式：$-\lambda + \mu = -\dfrac{3-2k}{3} + \dfrac{3+k}{3} = \dfrac{-(3-2k)+(3+k)}{3} = \dfrac{3k}{3} = k$ ✓

注意：$\vec{a}, \vec{b}$ 不共线（行列式 $1\times 1 - 2\times(-1) = 3 \neq 0$），故对**任意** $k$，$\vec{c}$ 均可由 $\vec{a}, \vec{b}$ 线性表示，$k$ 为任意实数。

← 基底不共线 → 平面内任意向量唯一分解

**答案**：$k$ 可取任意实数；表示式为 $\vec{c} = \dfrac{3-2k}{3}\vec{a} + \dfrac{3+k}{3}\vec{b}$，即 $\boxed{\vec{c} = \dfrac{3-2k}{3}\vec{a} + \dfrac{3+k}{3}\vec{b}}$。

**总结**：两不共线向量构成平面基底，平面内任意向量都能唯一线性表示；列分量方程组，解行列式不为零时必有唯一解。

---

## D.08 [中档] Part 2/04

**题目回顾**：在 $\triangle ABC$ 中，$\overrightarrow{CA} = \vec{a}$，$\overrightarrow{CB} = \vec{b}$，$|\vec{a}| = 2,\ |\vec{b}| = 3$，$\vec{a} \cdot \vec{b} = 3$。求 $\angle ACB$ 的大小，以及边 $AB$ 的长。

**思路**：$\angle ACB$ 即 $\vec{a}$ 与 $\vec{b}$ 的夹角，用数量积公式求；$AB = |\overrightarrow{AB}| = |\overrightarrow{CB}-\overrightarrow{CA}| = |\vec{b}-\vec{a}|$，平方展开。→ toolkit/08 向量两面（数量积法）。

**解答**：

**第一步**：求 $\angle ACB$。

$$\cos\angle ACB = \frac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|} = \frac{3}{2\times 3} = \frac{1}{2}$$

← 数量积法

$\angle ACB \in (0°, 180°)$，故 $\angle ACB = 60°$。

**第二步**：求 $|AB|$。

$$\overrightarrow{AB} = \overrightarrow{CB} - \overrightarrow{CA} = \vec{b} - \vec{a}$$

$$|\vec{b}-\vec{a}|^2 = |\vec{b}|^2 - 2\vec{a}\cdot\vec{b} + |\vec{a}|^2 = 9 - 6 + 4 = 7$$

$$|AB| = \sqrt{7}$$

← 平方展开

**答案**：$\angle ACB = \boxed{60°}$，$|AB| = \boxed{\sqrt{7}}$。

**总结**：三角形中两边向量已知 → 夹角用数量积，对边长用平方展开；与余弦定理本质一致。

---

## D.09 [中档] Part 2/04

**题目回顾**：已知向量 $\vec{a} = (m, 1)$，$\vec{b} = (1, m - 1)$（$m \in \mathbb{R}$）。（1）若 $|\vec{a} + \vec{b}| = |\vec{a} - \vec{b}|$，求 $m$；（2）若 $|\vec{a}| = |\vec{b}|$，求 $m$。

**思路**：（1）平方展开 $|\vec{a}+\vec{b}|^2 = |\vec{a}-\vec{b}|^2$ 化简，等价于 $\vec{a}\cdot\vec{b}=0$；（2）模相等则 $|\vec{a}|^2 = |\vec{b}|^2$。→ Part 2/04 向量模的计算。

**解答**：

**第一步**：计算 $\vec{a}\cdot\vec{b}$。

$$\vec{a}\cdot\vec{b} = m\times 1 + 1\times(m-1) = m + m - 1 = 2m - 1$$

**(1)** $|\vec{a}+\vec{b}|^2 = |\vec{a}|^2 + 2\vec{a}\cdot\vec{b} + |\vec{b}|^2$，$|\vec{a}-\vec{b}|^2 = |\vec{a}|^2 - 2\vec{a}\cdot\vec{b} + |\vec{b}|^2$。

两式相等 $\Leftrightarrow 4\vec{a}\cdot\vec{b} = 0 \Leftrightarrow \vec{a}\cdot\vec{b} = 0$：

$$2m - 1 = 0 \Rightarrow m = \frac{1}{2}$$

← 模相等 $\Rightarrow$ 点积为零

**(2)** $|\vec{a}|^2 = m^2 + 1$，$|\vec{b}|^2 = 1 + (m-1)^2 = m^2 - 2m + 2$。

$$m^2 + 1 = m^2 - 2m + 2 \Rightarrow 2m = 1 \Rightarrow m = \frac{1}{2}$$

← 模相等列方程

**答案**：(1) $m = \boxed{\dfrac{1}{2}}$；(2) $m = \boxed{\dfrac{1}{2}}$。

**总结**：$|\vec{a}+\vec{b}|=|\vec{a}-\vec{b}|$ 是 $\vec{a}\perp\vec{b}$ 的等价条件（菱形对角线等长 $\Leftrightarrow$ 对角线垂直的矩形特征）。

---

## D.10 [中档] Part 2/05

**题目回顾**：设 $\vec{a}, \vec{b}$ 均为非零向量，且 $|\vec{a} + \vec{b}| = |\vec{a} - \vec{b}|$。证明 $\vec{a} \perp \vec{b}$，并举出坐标下的具体例子验证。

**思路**：平方两边，展开后发现 $\vec{a}\cdot\vec{b} = 0$，即垂直。→ toolkit/01 四把剑（结构识别：模等式 → 平方）。

**解答**：

**证明**：

$$|\vec{a}+\vec{b}|^2 = |\vec{a}|^2 + 2\vec{a}\cdot\vec{b} + |\vec{b}|^2$$

$$|\vec{a}-\vec{b}|^2 = |\vec{a}|^2 - 2\vec{a}\cdot\vec{b} + |\vec{b}|^2$$

由 $|\vec{a}+\vec{b}| = |\vec{a}-\vec{b}|$，两边平方得两式相等：

$$|\vec{a}|^2 + 2\vec{a}\cdot\vec{b} + |\vec{b}|^2 = |\vec{a}|^2 - 2\vec{a}\cdot\vec{b} + |\vec{b}|^2$$

$$\Rightarrow 4\vec{a}\cdot\vec{b} = 0 \Rightarrow \vec{a}\cdot\vec{b} = 0$$

← 平方展开消元

故 $\vec{a} \perp \vec{b}$，证毕。

**坐标例子**：取 $\vec{a} = (1, 0)$，$\vec{b} = (0, 1)$，则：

$$|\vec{a}+\vec{b}| = |(1,1)| = \sqrt{2},\quad |\vec{a}-\vec{b}| = |(1,-1)| = \sqrt{2}$$

两者相等，且 $\vec{a}\cdot\vec{b} = 0$ 确认垂直。← 坐标验证

**答案**：证毕，$\vec{a} \perp \vec{b}$；例：$\vec{a}=(1,0),\ \vec{b}=(0,1)$，$|\vec{a}\pm\vec{b}|=\boxed{\sqrt{2}}$。

**总结**：模等式 → 立即平方展开，$4\vec{a}\cdot\vec{b}=0$ 便是垂直的充要条件；几何意义：平行四边形两对角线等长 $\Leftrightarrow$ 矩形 $\Leftrightarrow$ 邻边垂直。

---

## 空间向量（D.73–D.87）

---

## D.73 [中档] Part 9/01

**题目回顾**：在空间直角坐标系中，已知 $A(1, 0, 2)$，$B(3, -1, 4)$，$C(0, 2, 1)$。（1）求 $\overrightarrow{AB}$ 与 $\overrightarrow{AC}$；（2）求 $|\overrightarrow{AB}|$ 与 $\overrightarrow{AB} \cdot \overrightarrow{AC}$；（3）求 $\angle BAC$ 的余弦值。

**思路**：空间向量坐标运算：终点减起点；模用三维勾股定理；夹角用数量积公式。→ toolkit/11 立体向量（基础坐标运算）。

**解答**：

**(1)** 向量坐标（终减始）：

$$\overrightarrow{AB} = B - A = (3-1,\ -1-0,\ 4-2) = (2,\ -1,\ 2)$$

$$\overrightarrow{AC} = C - A = (0-1,\ 2-0,\ 1-2) = (-1,\ 2,\ -1)$$

← 终减始

**(2)** 模与数量积：

$$|\overrightarrow{AB}| = \sqrt{4+1+4} = \sqrt{9} = 3$$

$$\overrightarrow{AB}\cdot\overrightarrow{AC} = 2\times(-1) + (-1)\times 2 + 2\times(-1) = -2-2-2 = -6$$

← 三维点积

**(3)** 夹角余弦：

$$|\overrightarrow{AC}| = \sqrt{1+4+1} = \sqrt{6}$$

$$\cos\angle BAC = \frac{\overrightarrow{AB}\cdot\overrightarrow{AC}}{|\overrightarrow{AB}||\overrightarrow{AC}|} = \frac{-6}{3\sqrt{6}} = \frac{-2}{\sqrt{6}} = -\frac{\sqrt{6}}{3}$$

← 夹角公式

**答案**：$\overrightarrow{AB}=\boxed{(2,-1,2)}$，$\overrightarrow{AC}=\boxed{(-1,2,-1)}$；$|\overrightarrow{AB}|=3$，$\overrightarrow{AB}\cdot\overrightarrow{AC}=\boxed{-6}$；$\cos\angle BAC = \boxed{-\dfrac{\sqrt{6}}{3}}$。

**总结**：空间向量三步模板：①终减始求坐标 → ②求模（三维勾股）→ ③点积/模求夹角。

---

## D.74 [中档] Part 9/02

**题目回顾**：正方体 $ABCD-A_1B_1C_1D_1$ 棱长为 $1$，建立空间直角坐标系（以 $A$ 为原点，$AB$ 方向为 $x$ 轴，$AD$ 方向为 $y$ 轴，$AA_1$ 方向为 $z$ 轴）。（1）写出各顶点坐标；（2）用坐标法证明 $A_1C \perp BD_1$。

**思路**：正方体建系后坐标规律：底面正方形 + 高度方向；垂直用向量数量积 $= 0$ 证明。→ toolkit/11 立体向量（建系）。

**解答**：

**(1)** 各顶点坐标：

底面：$A(0,0,0)$，$B(1,0,0)$，$C(1,1,0)$，$D(0,1,0)$；

顶面：$A_1(0,0,1)$，$B_1(1,0,1)$，$C_1(1,1,1)$，$D_1(0,1,1)$。

← 建系

**(2)** 证明 $A_1C \perp BD_1$：

$$\overrightarrow{A_1C} = C - A_1 = (1-0,\ 1-0,\ 0-1) = (1,1,-1)$$

$$\overrightarrow{BD_1} = D_1 - B = (0-1,\ 1-0,\ 1-0) = (-1,1,1)$$

$$\overrightarrow{A_1C}\cdot\overrightarrow{BD_1} = 1\times(-1) + 1\times 1 + (-1)\times 1 = -1+1-1 = -1 \neq 0$$

**注意**：题目要求证明 $A_1C \perp BD_1$，但计算结果 $\overrightarrow{A_1C}\cdot\overrightarrow{BD_1}=-1\neq 0$。

实际上，正方体中体对角线 $A_1C$ 与 $BD_1$ 所成角由数量积计算：$\cos\theta = \dfrac{-1}{|\overrightarrow{A_1C}||\overrightarrow{BD_1}|} = \dfrac{-1}{\sqrt{3}\cdot\sqrt{3}} = -\dfrac{1}{3}$，夹角约为 $109.5°$（非直角）。

正确的垂直关系为：$A_1C \perp BD$（面对角线）。验证：$\overrightarrow{BD} = (-1,1,0)$，$\overrightarrow{A_1C}\cdot\overrightarrow{BD} = -1+1+0 = 0$ ✓

← 坐标法验证

**答案**：各顶点坐标见解答；题目应为证 $A_1C \perp BD$，$\overrightarrow{A_1C}\cdot\overrightarrow{BD} = \boxed{0}$，垂直证毕。

**总结**：坐标法证垂直只需算数量积；建系后所有几何问题化为代数计算，不依赖直觉判断。

---

## D.75 [中档] Part 9/03

**题目回顾**：如题 D.74 的正方体（棱长为 $1$），求空间向量 $\overrightarrow{A_1B}$ 与 $\overrightarrow{A_1D}$ 的数量积，以及两向量的夹角。

**思路**：沿用 D.74 的坐标系，直接计算；夹角公式 $\cos\theta = \dfrac{\vec{u}\cdot\vec{v}}{|\vec{u}||\vec{v}|}$。→ toolkit/11 立体向量。

**解答**：

沿用坐标：$A_1(0,0,1)$，$B(1,0,0)$，$D(0,1,0)$。

$$\overrightarrow{A_1B} = B - A_1 = (1,0,-1),\quad \overrightarrow{A_1D} = D - A_1 = (0,1,-1)$$

数量积：

$$\overrightarrow{A_1B}\cdot\overrightarrow{A_1D} = 1\times 0 + 0\times 1 + (-1)\times(-1) = 0 + 0 + 1 = 1$$

← 坐标点积

模：$|\overrightarrow{A_1B}| = \sqrt{1+0+1} = \sqrt{2}$，$|\overrightarrow{A_1D}| = \sqrt{0+1+1} = \sqrt{2}$。

夹角：

$$\cos\theta = \frac{1}{\sqrt{2}\times\sqrt{2}} = \frac{1}{2} \Rightarrow \theta = 60°$$

← 夹角公式

**答案**：$\overrightarrow{A_1B}\cdot\overrightarrow{A_1D} = \boxed{1}$，夹角 $\theta = \boxed{60°}$。

**总结**：正方体中从同一顶点出发的两条面对角线夹角为 $60°$，利用坐标验算是标准方法。

---

## D.76 [中档] Part 9/04

**题目回顾**：在正三棱柱 $ABC-A_1B_1C_1$ 中，底面边长为 $2$，高为 $2$，以 $A$ 为原点建立空间直角坐标系。（1）写出 $B, C, A_1, B_1, C_1$ 的坐标；（2）用向量法证明 $BB_1 \parallel$ 平面 $AA_1C_1C$。

**思路**：正三棱柱建系：$x$ 轴沿 $AB$，$y$ 轴在底面内垂直 $AB$，$z$ 轴沿高；$C$ 的 $y$ 坐标需用等边三角形高 $\sqrt{3}$；$BB_1 \parallel$ 平面等价于 $BB_1$ 的方向向量在平面内但 $B$ 不在平面内。→ toolkit/11 立体向量（建系 + 法向量）。

**解答**：

**(1)** 正三棱柱底面等边三角形边长 $2$，以 $A$ 为原点，$AB$ 方向为 $x$ 轴，底面内 $\perp AB$ 方向为 $y$ 轴，$AA_1$ 为 $z$ 轴：

- $A(0,0,0)$，$B(2,0,0)$，$C(1,\sqrt{3},0)$（等边三角形顶点）← 等边三角形高 $\sqrt{3}$
- $A_1(0,0,2)$，$B_1(2,0,2)$，$C_1(1,\sqrt{3},2)$

**(2)** 证明 $BB_1 \parallel$ 平面 $AA_1C_1C$：

平面 $AA_1C_1C$ 由 $A(0,0,0)$，$A_1(0,0,2)$，$C(1,\sqrt{3},0)$，$C_1(1,\sqrt{3},2)$ 确定。

平面内两向量：$\overrightarrow{AA_1} = (0,0,2)$，$\overrightarrow{AC} = (1,\sqrt{3},0)$。

求平面法向量 $\vec{n} = (n_1,n_2,n_3)$ 满足：

$$\vec{n}\cdot\overrightarrow{AA_1} = 0 \Rightarrow 2n_3 = 0 \Rightarrow n_3 = 0$$

$$\vec{n}\cdot\overrightarrow{AC} = 0 \Rightarrow n_1 + \sqrt{3}n_2 = 0$$

取 $n_2 = 1$，则 $n_1 = -\sqrt{3}$，法向量 $\vec{n} = (-\sqrt{3}, 1, 0)$。← 法向量

方向向量 $\overrightarrow{BB_1} = (0, 0, 2)$。

$$\vec{n}\cdot\overrightarrow{BB_1} = (-\sqrt{3})\times 0 + 1\times 0 + 0\times 2 = 0$$

故 $\overrightarrow{BB_1}$ 与法向量垂直，即 $BB_1 \parallel$ 平面（或在平面内）。

验证 $B(2,0,0)$ 是否在平面内：平面过 $A(0,0,0)$ 且法向量 $(-\sqrt{3},1,0)$，方程为 $-\sqrt{3}x + y = 0$，代入 $B$：$-2\sqrt{3} + 0 = -2\sqrt{3} \neq 0$。

故 $B \notin$ 平面，$BB_1 \parallel$ 平面 $AA_1C_1C$，证毕。← 平行判定

**答案**：坐标见解答；$BB_1 \parallel$ 平面 $AA_1C_1C$，证毕。$\boxed{\vec{n}\cdot\overrightarrow{BB_1}=0,\ B\notin\text{平面}}$。

**总结**：直线平行于平面 $\Leftrightarrow$ 方向向量 $\perp$ 平面法向量，且直线上的点不在平面内。

---

## D.77 [中档] Part 9/05

**题目回顾**：设平面 $\alpha$ 的法向量为 $\vec{n} = (1, -1, 2)$，平面 $\beta$ 的法向量为 $\vec{m} = (2, 1, -1)$。（1）求两平面所成二面角的余弦；（2）判断 $\alpha$ 与 $\beta$ 是否垂直。

**思路**：两平面所成角的余弦等于法向量夹角余弦的绝对值（取锐角）；$\alpha \perp \beta \Leftrightarrow \vec{n}\cdot\vec{m} = 0$。→ toolkit/11 立体向量（法向量法求二面角）。

**解答**：

**(1)** 计算数量积与模：

$$\vec{n}\cdot\vec{m} = 1\times 2 + (-1)\times 1 + 2\times(-1) = 2 - 1 - 2 = -1$$

$$|\vec{n}| = \sqrt{1+1+4} = \sqrt{6},\quad |\vec{m}| = \sqrt{4+1+1} = \sqrt{6}$$

两平面所成角：

$$\cos\theta = \left|\frac{\vec{n}\cdot\vec{m}}{|\vec{n}||\vec{m}|}\right| = \left|\frac{-1}{\sqrt{6}\times\sqrt{6}}\right| = \frac{1}{6}$$

← 取绝对值（二面角为锐角或直角）

**(2)** 判断垂直：

$\vec{n}\cdot\vec{m} = -1 \neq 0$，故 $\alpha$ 与 $\beta$ **不垂直**。

**答案**：两平面所成二面角 $\cos\theta = \boxed{\dfrac{1}{6}}$；$\alpha$ 与 $\beta$ 不垂直。

**总结**：两平面夹角 $=$ 法向量夹角（或其补角），取锐角即取绝对值；垂直当且仅当法向量点积为零。

---

## D.78 [中档] Part 9/06

**题目回顾**：空间中 $A(1, 2, 3)$，$B(4, 0, 1)$，$C(2, -1, 2)$，$D(5, 1, 0)$。（1）求直线 $AB$ 的方向向量；（2）求直线 $CD$ 的方向向量；（3）求直线 $AB$ 与 $CD$ 所成角的余弦值。

**思路**：方向向量即沿直线的向量（终减始）；两直线所成角用两方向向量夹角，取锐角（绝对值）。→ toolkit/11 立体向量（异面直线所成角）。

**解答**：

**(1)** $\overrightarrow{AB} = B - A = (3, -2, -2)$，即方向向量 $\vec{d}_1 = (3,-2,-2)$。

**(2)** $\overrightarrow{CD} = D - C = (3, 2, -2)$，即方向向量 $\vec{d}_2 = (3,2,-2)$。

**(3)** 所成角：

$$\vec{d}_1\cdot\vec{d}_2 = 9 + (-4) + 4 = 9$$

$$|\vec{d}_1| = \sqrt{9+4+4} = \sqrt{17},\quad |\vec{d}_2| = \sqrt{9+4+4} = \sqrt{17}$$

$$\cos\theta = \left|\frac{9}{\sqrt{17}\times\sqrt{17}}\right| = \frac{9}{17}$$

← 取绝对值（异面直线所成角为锐角）

**答案**：$\vec{d}_1 = \boxed{(3,-2,-2)}$，$\vec{d}_2 = \boxed{(3,2,-2)}$；所成角 $\cos\theta = \boxed{\dfrac{9}{17}}$。

**总结**：异面直线（或直线间）所成角：方向向量点积除以两模之积，取绝对值得锐角余弦。

---

## D.79 [中档] Part 9/07

**题目回顾**：含参问题：空间中三点 $A(1, 0, 0)$，$B(0, 1, 0)$，$C(0, 0, t)$（$t \neq 0$）。（1）用坐标法求平面 $ABC$ 的法向量（用 $t$ 表示）；（2）当 $t = 1$ 时，求平面 $ABC$ 的方程。

**思路**：平面内两向量取数量积 = 0 的方程组求法向量；平面方程用点法式。→ toolkit/11 立体向量（法向量 + 平面方程）。

**解答**：

**(1)** 求法向量：

$$\overrightarrow{AB} = B - A = (-1, 1, 0),\quad \overrightarrow{AC} = C - A = (-1, 0, t)$$

设法向量 $\vec{n} = (a, b, c)$，则：

$$\vec{n}\cdot\overrightarrow{AB} = -a + b = 0 \Rightarrow b = a$$

$$\vec{n}\cdot\overrightarrow{AC} = -a + ct = 0 \Rightarrow c = \frac{a}{t}$$

取 $a = t$，则 $b = t$，$c = 1$，法向量 $\vec{n} = (t, t, 1)$。← 法向量

**(2)** 当 $t = 1$：$\vec{n} = (1, 1, 1)$，过 $A(1, 0, 0)$，平面方程为：

$$1(x-1) + 1(y-0) + 1(z-0) = 0 \Rightarrow x + y + z = 1$$

← 点法式

验证：$A(1,0,0)$：$1+0+0=1$ ✓；$B(0,1,0)$：$0+1+0=1$ ✓；$C(0,0,1)$：$0+0+1=1$ ✓。

**答案**：法向量 $\vec{n} = \boxed{(t, t, 1)}$（$t\neq 0$）；$t=1$ 时平面方程 $\boxed{x+y+z=1}$。

**总结**：含参法向量：先建方程组，参数自然保留；平面过三截距轴点 $(a,0,0),(0,b,0),(0,0,c)$ 时方程为 $\frac{x}{a}+\frac{y}{b}+\frac{z}{c}=1$（截距式）。

---

## D.80 [中档] Part 9/08

**题目回顾**：设四棱锥 $P-ABCD$，$ABCD$ 为正方形，$P$ 在 $ABCD$ 正上方，$PA \perp$ 底面 $ABCD$，$PA = AB = 2$。以 $A$ 为原点建系。（1）写出 $P, B, C, D$ 的坐标；（2）求平面 $PBC$ 的法向量；（3）求二面角 $P-BC-A$ 的余弦值。

**思路**：$PA\perp$ 底面 → $z$ 轴沿 $PA$，底面铺 $xOy$；法向量用平面内两向量联立方程组；二面角用两平面法向量夹角（注意方向取舍）。→ toolkit/11 立体向量（建系 + 法向量 + 二面角）。

**解答**：

**(1)** 建系：$A$ 为原点，$AB$ 方向为 $x$ 轴，$AD$ 方向为 $y$ 轴，$AA_1$（即 $AP$）为 $z$ 轴。

$$A(0,0,0),\ B(2,0,0),\ C(2,2,0),\ D(0,2,0),\ P(0,0,2)$$

← 建系

**(2)** 求平面 $PBC$ 的法向量：

$$\overrightarrow{BP} = P - B = (-2, 0, 2),\quad \overrightarrow{BC} = C - B = (0, 2, 0)$$

设 $\vec{n} = (a,b,c)$：

$$\vec{n}\cdot\overrightarrow{BP} = -2a + 2c = 0 \Rightarrow a = c$$

$$\vec{n}\cdot\overrightarrow{BC} = 2b = 0 \Rightarrow b = 0$$

取 $a = 1$，法向量 $\vec{n}_1 = (1, 0, 1)$。← 法向量

**(3)** 求二面角 $P-BC-A$：

底面 $ABCD$ 的法向量 $\vec{n}_2 = (0, 0, 1)$（$z$ 轴方向）。

$$\cos\theta = \frac{\vec{n}_1\cdot\vec{n}_2}{|\vec{n}_1||\vec{n}_2|} = \frac{0+0+1}{\sqrt{2}\times 1} = \frac{1}{\sqrt{2}} = \frac{\sqrt{2}}{2}$$

← 二面角 = 法向量夹角

但需确认方向：$\vec{n}_1 = (1,0,1)$ 指向平面 $PBC$ 的外侧（与 $A$ 同侧），$\vec{n}_2 = (0,0,1)$ 指向底面上方，故二面角为锐角，$\cos = \dfrac{\sqrt{2}}{2}$。

**答案**：坐标见解答；$\vec{n}_1 = \boxed{(1,0,1)}$；二面角余弦 $\boxed{\dfrac{\sqrt{2}}{2}}$，即 $45°$。

**总结**：$PA\perp$ 底面 → 建系最简；二面角用法向量，注意取锐角还是钝角须由几何判断（法向量方向与棱两侧的点的位置关系）。

---

## D.81 [中档] Part 9/08

**题目回顾**：在上题 D.80 的条件下：（1）求点 $A$ 到平面 $PBC$ 的距离；（2）求直线 $PA$ 与平面 $BCD$ 所成角的正弦值。

**思路**：点到平面距离用法向量公式 $d = \dfrac{|\vec{n}\cdot\overrightarrow{MA}|}{|\vec{n}|}$；直线与平面所成角的正弦 $=$ 方向向量与法向量夹角的余弦（即 $\sin\phi = |\cos\angle(\vec{d},\vec{n})|$）。→ toolkit/11 立体向量（点面距 + 线面角）。

**解答**：

沿用 D.80 坐标：$A(0,0,0)$，$B(2,0,0)$，$C(2,2,0)$，$P(0,0,2)$，平面 $PBC$ 法向量 $\vec{n}_1 = (1,0,1)$。

**(1)** 点 $A$ 到平面 $PBC$ 的距离：

取平面上一点 $B(2,0,0)$，$\overrightarrow{BA} = A - B = (-2,0,0)$。

$$d = \frac{|\vec{n}_1\cdot\overrightarrow{BA}|}{|\vec{n}_1|} = \frac{|1\times(-2)+0+1\times 0|}{\sqrt{2}} = \frac{2}{\sqrt{2}} = \sqrt{2}$$

← 点面距公式

**(2)** 直线 $PA$ 与平面 $BCD$ 所成角：

$\overrightarrow{PA} = A - P = (0,0,-2)$，方向向量 $\vec{d} = (0,0,-2)$（或取 $(0,0,1)$）。

底面 $BCD$ 在 $z=0$ 平面（$B,C,D$ 的 $z$ 坐标均为 $0$），法向量 $\vec{n}_2 = (0,0,1)$。

$$\sin\phi = \left|\frac{\vec{d}\cdot\vec{n}_2}{|\vec{d}||\vec{n}_2|}\right| = \left|\frac{0+0+(-2)}{2\times 1}\right| = 1$$

← 线面角正弦公式

故 $PA \perp$ 底面 $BCD$，与题设 $PA\perp$ 底面一致，所成角为 $90°$，正弦值为 $1$。

**答案**：$A$ 到平面 $PBC$ 距离 $= \boxed{\sqrt{2}}$；$PA$ 与平面 $BCD$ 所成角正弦值 $= \boxed{1}$（即垂直）。

**总结**：点面距 $=\dfrac{|\vec{n}\cdot\overrightarrow{MP}|}{|\vec{n}|}$（$M$ 为平面上任一点）；线面角 $\phi$ 满足 $\sin\phi = |\cos\angle(\vec{d},\vec{n})|$，当线垂直面时正弦为 $1$。

---

## D.82 [中档] Part 9/09

**题目回顾**：正三棱柱 $ABC-A_1B_1C_1$，底面边长 $a = 2$，高 $h = 2\sqrt{3}$，以 $A$ 为原点建系（$x$ 轴沿 $AB$，$y$ 轴在底面内垂直 $AB$，$z$ 轴沿 $AA_1$）。（1）写出六个顶点的坐标；（2）求平面 $AB_1C$ 的法向量；（3）求二面角 $A_1-AB_1-B$ 的余弦值。

**思路**：正三棱柱建系；平面法向量用联立方程；二面角先确定棱 $AB_1$ 两侧的两个平面，分别求法向量再求夹角。→ toolkit/11 立体向量（二面角法向量法）。

**解答**：

**(1)** 坐标（等边三角形底面，边长 $2$，高 $2\sqrt{3}$）：

$$A(0,0,0),\ B(2,0,0),\ C(1,\sqrt{3},0)$$

$$A_1(0,0,2\sqrt{3}),\ B_1(2,0,2\sqrt{3}),\ C_1(1,\sqrt{3},2\sqrt{3})$$

← 等边三角形高 $=\sqrt{3}$

**(2)** 平面 $AB_1C$ 的法向量：

$$\overrightarrow{AB_1} = B_1 - A = (2, 0, 2\sqrt{3}),\quad \overrightarrow{AC} = (1, \sqrt{3}, 0)$$

设 $\vec{n} = (a,b,c)$：

$$\vec{n}\cdot\overrightarrow{AB_1}: 2a + 2\sqrt{3}c = 0 \Rightarrow a = -\sqrt{3}c$$

$$\vec{n}\cdot\overrightarrow{AC}: a + \sqrt{3}b = 0 \Rightarrow b = -\frac{a}{\sqrt{3}} = c$$

取 $c = 1$，则 $a = -\sqrt{3}$，$b = 1$，法向量 $\vec{n}_1 = (-\sqrt{3}, 1, 1)$。← 法向量

**(3)** 二面角 $A_1-AB_1-B$：

棱为 $AB_1$，一侧面为平面 $A_1AB_1$，另一侧面为平面 $BAB_1$。

平面 $A_1AB_1$ 中：$\overrightarrow{AA_1} = (0,0,2\sqrt{3})$，$\overrightarrow{AB_1} = (2,0,2\sqrt{3})$。

设法向量 $\vec{n}_2 = (p,q,r)$：

$$\vec{n}_2\cdot\overrightarrow{AA_1}: 2\sqrt{3}r = 0 \Rightarrow r = 0$$

$$\vec{n}_2\cdot\overrightarrow{AB_1}: 2p + 2\sqrt{3}r = 2p = 0 \Rightarrow p = 0$$

取 $q = 1$，$\vec{n}_2 = (0, 1, 0)$（即 $y$ 轴方向）。

平面 $BAB_1$ 中：$\overrightarrow{AB} = (2,0,0)$，$\overrightarrow{AB_1} = (2,0,2\sqrt{3})$。

设法向量 $\vec{n}_3 = (s,t,u)$：

$$\vec{n}_3\cdot\overrightarrow{AB}: 2s = 0 \Rightarrow s = 0$$

$$\vec{n}_3\cdot\overrightarrow{AB_1}: 2s + 2\sqrt{3}u = 2\sqrt{3}u = 0 \Rightarrow u = 0$$

取 $t = 1$，$\vec{n}_3 = (0, 1, 0)$。

两平面同一法向量 → 两平面平行？不对，需重新确认。平面 $A_1AB_1$ 包含 $A(0,0,0)$，$A_1(0,0,2\sqrt{3})$，$B_1(2,0,2\sqrt{3})$，三点 $y$ 坐标均为 $0$，故平面 $A_1AB_1$ 即为 $y = 0$（$xOz$ 平面）。

平面 $BAB_1$ 包含 $B(2,0,0)$，$A(0,0,0)$，$B_1(2,0,2\sqrt{3})$，三点 $y$ 坐标均为 $0$，亦在平面 $y=0$ 内。

两平面实为同一平面（$y=0$），故二面角 $A_1-AB_1-B$ 退化。

重新理解题意：应为过 $AB_1$ 的两个半平面，含 $A_1$ 的半平面和含 $B$ 的半平面，实际三点均在 $y=0$，说明 $A_1, A, B, B_1$ 共面（均在 $y=0$ 平面），二面角不构成。

**修正**：实际有意义的二面角为 $A_1-BC_1-B$（正三棱柱中常见二面角类型）。针对原题，法向量方法计算如下（合理理解"二面角 $A_1-AB_1-B$"为含 $A_1$ 的面与含 $B$ 的面沿 $AB_1$ 形成的角）：

过 $AB_1$ 的两侧：含 $A_1$ 的半平面为 $A_1AB_1$（在 $xOz$ 平面），含 $B$ 的半平面需独立判断 $C$ 在哪侧。取包含 $C$ 或 $B$ 等点的过 $AB_1$ 的截面。

实际上题目中 $A_1, A, B_1$ 在 $y=0$ 平面，$B$ 也在 $y=0$ 平面，所以四点共面，二面角为 $180°$ 或 $0°$，题目出题有误。

正确做法应计算平面 $AB_1C$ 与平面 $A_1AB_1B$ 所成二面角（沿 $AB_1$），即 $\vec{n}_1 = (-\sqrt{3},1,1)$ 与 $\vec{n}_2 = (0,1,0)$：

$$\cos\theta = \frac{\vec{n}_1\cdot\vec{n}_2}{|\vec{n}_1||\vec{n}_2|} = \frac{0+1+0}{\sqrt{3+1+1}\times 1} = \frac{1}{\sqrt{5}}$$

← 二面角计算

**答案**：法向量 $\vec{n}_1 = \boxed{(-\sqrt{3},1,1)}$；二面角（平面 $AB_1C$ 与 $xOz$ 面沿 $AB_1$）余弦值 $\boxed{\dfrac{1}{\sqrt{5}} = \dfrac{\sqrt{5}}{5}}$。

**总结**：正三棱柱建系务必先确定等边三角形顶点坐标；若所求二面角退化（棱两侧的点共面），需重新审题。

---

## D.83 [中档] Part 9/09

**题目回顾**：已知空间中向量 $\vec{a} = (1, 2, -2)$，$\vec{b} = (2, -1, 0)$，$\vec{c} = (-1, 1, 1)$。（1）求 $\vec{a}\cdot\vec{b}$，$\vec{a}\cdot\vec{c}$；（2）求 $\vec{a}$ 与 $\vec{b}$ 的夹角；（3）判断 $\vec{a}, \vec{b}, \vec{c}$ 是否线性相关。

**思路**：数量积坐标公式；夹角公式；线性相关判断：设 $\lambda\vec{a}+\mu\vec{b}+\nu\vec{c}=\vec{0}$ 判断是否有非零解，或用 Gram 行列式（高中范围：逐一验证）。→ toolkit/11 立体向量（空间向量运算）。

**解答**：

**(1)** 数量积：

$$\vec{a}\cdot\vec{b} = 1\times 2 + 2\times(-1) + (-2)\times 0 = 2-2+0 = 0$$

$$\vec{a}\cdot\vec{c} = 1\times(-1) + 2\times 1 + (-2)\times 1 = -1+2-2 = -1$$

← 坐标点积

**(2)** $\vec{a}$ 与 $\vec{b}$ 的夹角：

$\vec{a}\cdot\vec{b} = 0$，故 $\vec{a} \perp \vec{b}$，夹角 $\theta = 90°$。← 垂直判定

**(3)** 判断线性相关（设 $\lambda\vec{a}+\mu\vec{b} = \vec{c}$，看是否有解）：

$$\lambda(1,2,-2) + \mu(2,-1,0) = (-1,1,1)$$

$$\begin{cases}\lambda + 2\mu = -1 \\ 2\lambda - \mu = 1 \\ -2\lambda = 1\end{cases}$$

第三式：$\lambda = -\dfrac{1}{2}$；代入第一式：$-\dfrac{1}{2}+2\mu = -1 \Rightarrow \mu = -\dfrac{1}{4}$；

验证第二式：$2\times(-\dfrac{1}{2}) - (-\dfrac{1}{4}) = -1+\dfrac{1}{4} = -\dfrac{3}{4} \neq 1$。

方程组无解，$\vec{c}$ 不能由 $\vec{a}, \vec{b}$ 线性表示，故 $\vec{a}, \vec{b}, \vec{c}$ **线性无关**（不共面）。← 线性相关判断

**答案**：$\vec{a}\cdot\vec{b}=\boxed{0}$（$\vec{a}\perp\vec{b}$），$\vec{a}\cdot\vec{c}=\boxed{-1}$；夹角 $\theta=\boxed{90°}$；$\vec{a},\vec{b},\vec{c}$ 线性无关（不共面）。

**总结**：空间三向量线性相关 $\Leftrightarrow$ 其中一个可由另两个线性表示 $\Leftrightarrow$ 三向量共面；线性无关 $\Leftrightarrow$ 三向量构成空间基底。

---

## D.84 [中档] Part 9/10

**题目回顾**：在正四棱锥 $P-ABCD$（底面边长 $2$，高 $2$）中，以 $A$ 为原点建坐标系（$x$ 轴沿 $AB$，$y$ 轴沿 $AD$，$z$ 轴向上）。（1）写出各顶点坐标；（2）求 $PA$ 与底面 $ABCD$ 所成角；（3）求 $PB$ 与平面 $PAD$ 所成角的正弦值。

**思路**：正四棱锥底面中心 $O'$ 在 $(1,1,0)$，顶点 $P$ 在 $(1,1,2)$；$PA$ 与底面所成角用斜线在底面投影法（$PA$ 投影为 $O'A$）；平面 $PAD$ 法向量用联立方程。→ toolkit/11 立体向量（线面角）。

**解答**：

**(1)** 底面正方形边长 $2$，$A$ 为原点：

$$A(0,0,0),\ B(2,0,0),\ C(2,2,0),\ D(0,2,0)$$

底面中心 $O' = (1,1,0)$，高 $= 2$，故 $P(1,1,2)$。← 建系

**(2)** $PA$ 与底面所成角：

$PA$ 在底面的投影为 $O'A$（从 $A$ 到底面中心 $O'$，反向即 $AO'$）。

$$\overrightarrow{AP} = (1,1,2),\quad \overrightarrow{AO'} = (1,1,0)$$

$$\sin\angle = \frac{|\text{PA 的竖直分量}|}{|PA|} = \frac{2}{\sqrt{1+1+4}} = \frac{2}{\sqrt{6}} = \frac{\sqrt{6}}{3}$$

← 线面角 = 斜线与其在面上投影的夹角

也可用法向量：底面法向量 $\vec{n} = (0,0,1)$，方向向量 $\overrightarrow{AP} = (1,1,2)$：

$$\sin\phi = \frac{|\overrightarrow{AP}\cdot\vec{n}|}{|\overrightarrow{AP}||\vec{n}|} = \frac{|2|}{\sqrt{6}} = \frac{2}{\sqrt{6}} = \frac{\sqrt{6}}{3}$$

**(3)** $PB$ 与平面 $PAD$ 所成角：

平面 $PAD$ 包含 $P(1,1,2)$，$A(0,0,0)$，$D(0,2,0)$：

$$\overrightarrow{AP} = (1,1,2),\quad \overrightarrow{AD} = (0,2,0)$$

设法向量 $\vec{n}_1 = (a,b,c)$：

$$a + b + 2c = 0,\quad 2b = 0 \Rightarrow b = 0,\quad a = -2c$$

取 $c = 1$，$\vec{n}_1 = (-2, 0, 1)$。

$\overrightarrow{PB} = B - P = (2-1, 0-1, 0-2) = (1,-1,-2)$。

$$\sin\phi = \frac{|\overrightarrow{PB}\cdot\vec{n}_1|}{|\overrightarrow{PB}||\vec{n}_1|} = \frac{|1\times(-2)+(-1)\times 0+(-2)\times 1|}{\sqrt{1+1+4}\times\sqrt{4+0+1}} = \frac{|-2-2|}{\sqrt{6}\times\sqrt{5}} = \frac{4}{\sqrt{30}} = \frac{4\sqrt{30}}{30} = \frac{2\sqrt{30}}{15}$$

← 线面角正弦

**答案**：坐标见解答；$PA$ 与底面所成角正弦值 $= \boxed{\dfrac{\sqrt{6}}{3}}$，角度 $= \arcsin\dfrac{\sqrt{6}}{3}$；$PB$ 与平面 $PAD$ 所成角正弦值 $= \boxed{\dfrac{2\sqrt{30}}{15}}$。

**总结**：正四棱锥顶点坐标 $=$ 底面中心正上方，注意中心坐标为底面正方形中心。

---

## D.85 [中档] Part 9/11

**题目回顾**：已知平面 $\alpha$ 过三点 $A(2, 0, 0)$，$B(0, 3, 0)$，$C(0, 0, 4)$，求：（1）平面 $\alpha$ 的方程（截距式转一般式）；（2）原点 $O$ 到平面 $\alpha$ 的距离；（3）点 $D(1, 1, 1)$ 到平面 $\alpha$ 的距离。

**思路**：三坐标截距直接写截距式 $\frac{x}{a}+\frac{y}{b}+\frac{z}{c}=1$；点到平面距离公式 $d = \dfrac{|Ax_0+By_0+Cz_0+D|}{\sqrt{A^2+B^2+C^2}}$。→ toolkit/11 立体向量（点面距）。

**解答**：

**(1)** 截距式：$A, B, C$ 分别在 $x, y, z$ 轴截距为 $2, 3, 4$，故：

$$\frac{x}{2} + \frac{y}{3} + \frac{z}{4} = 1$$

化为一般式（乘以 $12$）：

$$6x + 4y + 3z = 12 \Rightarrow 6x + 4y + 3z - 12 = 0$$

← 截距式

**(2)** 原点 $O(0,0,0)$ 到平面 $6x+4y+3z-12=0$ 的距离：

$$d = \frac{|6\times 0+4\times 0+3\times 0-12|}{\sqrt{36+16+9}} = \frac{12}{\sqrt{61}}= \frac{12\sqrt{61}}{61}$$

← 点面距公式

**(3)** 点 $D(1,1,1)$ 到平面的距离：

$$d = \frac{|6\times 1+4\times 1+3\times 1-12|}{\sqrt{61}} = \frac{|6+4+3-12|}{\sqrt{61}} = \frac{1}{\sqrt{61}} = \frac{\sqrt{61}}{61}$$

**答案**：平面方程 $\boxed{6x+4y+3z-12=0}$；$O$ 到平面距离 $\boxed{\dfrac{12\sqrt{61}}{61}}$；$D$ 到平面距离 $\boxed{\dfrac{\sqrt{61}}{61}}$。

**总结**：三截距轴点 → 直接截距式，乘通分得一般式；点面距公式是核心工具，需背熟。

---

## D.86 [中档] Part 9/12

**题目回顾**：在正三棱柱 $ABC-A_1B_1C_1$ 中，底面边长为 $2$，$AA_1 = \sqrt{6}$，以 $A$ 为原点建系。（1）求平面 $A_1BC$ 的法向量；（2）求点 $A_1$ 到平面 $ABC$ 的距离（坐标法验证，结果应等于 $\sqrt{6}$）；（3）求二面角 $A_1-BC-A$ 的余弦值。

**思路**：正三棱柱建系，$C$ 的坐标含 $\sqrt{3}$；平面 $A_1BC$ 法向量用联立；二面角 $A_1-BC-A$：棱为 $BC$，两侧平面为 $A_1BC$ 和 $ABC$（底面），求两法向量夹角。→ toolkit/11 立体向量（二面角）。

**解答**：

**(1)** 建系：$A(0,0,0)$，$B(2,0,0)$，$C(1,\sqrt{3},0)$，$A_1(0,0,\sqrt{6})$，$B_1(2,0,\sqrt{6})$，$C_1(1,\sqrt{3},\sqrt{6})$。← 建系

求平面 $A_1BC$ 法向量（$A_1(0,0,\sqrt{6})$，$B(2,0,0)$，$C(1,\sqrt{3},0)$）：

$$\overrightarrow{BA_1} = (-2, 0, \sqrt{6}),\quad \overrightarrow{BC} = (-1, \sqrt{3}, 0)$$

设 $\vec{n} = (a,b,c)$：

$$\vec{n}\cdot\overrightarrow{BA_1}: -2a + \sqrt{6}c = 0 \Rightarrow a = \frac{\sqrt{6}}{2}c$$

$$\vec{n}\cdot\overrightarrow{BC}: -a + \sqrt{3}b = 0 \Rightarrow b = \frac{a}{\sqrt{3}} = \frac{\sqrt{6}}{2\sqrt{3}}c = \frac{\sqrt{2}}{2}c$$

取 $c = 2$：$a = \sqrt{6}$，$b = \sqrt{2}$，法向量 $\vec{n}_1 = (\sqrt{6}, \sqrt{2}, 2)$。← 法向量

**(2)** $A_1(0,0,\sqrt{6})$ 到底面 $ABC$（即 $z=0$ 平面）的距离：

底面 $z=0$，$A_1$ 的 $z$ 坐标为 $\sqrt{6}$，距离 $= \sqrt{6}$。← 直接读 $z$ 坐标

坐标法验证：底面法向量 $\vec{n}_2 = (0,0,1)$，取底面上点 $A(0,0,0)$，$\overrightarrow{AA_1} = (0,0,\sqrt{6})$，$d = \dfrac{|(0,0,\sqrt{6})\cdot(0,0,1)|}{|(0,0,1)|} = \sqrt{6}$ ✓

**(3)** 二面角 $A_1-BC-A$：

底面 $ABC$ 法向量 $\vec{n}_2 = (0,0,1)$，平面 $A_1BC$ 法向量 $\vec{n}_1 = (\sqrt{6},\sqrt{2},2)$。

$$\cos\theta = \frac{\vec{n}_1\cdot\vec{n}_2}{|\vec{n}_1||\vec{n}_2|} = \frac{0+0+2}{\sqrt{6+2+4}\times 1} = \frac{2}{\sqrt{12}} = \frac{2}{2\sqrt{3}} = \frac{1}{\sqrt{3}} = \frac{\sqrt{3}}{3}$$

← 二面角余弦

注意：须判断符号（是锐角还是钝角），由几何知 $A_1$ 在底面上方，二面角为锐角（余弦正），正确。

**答案**：法向量 $\vec{n}_1 = \boxed{(\sqrt{6},\sqrt{2},2)}$；$A_1$ 到底面距离 $= \boxed{\sqrt{6}}$；二面角余弦值 $= \boxed{\dfrac{\sqrt{3}}{3}}$。

**总结**：正三棱柱中，高等于棱柱高（$AA_1$），直接读 $z$ 坐标即是到底面的距离；二面角的符号由几何位置确定（内角则余弦可正可负）。

---

## D.87 [中档] Part 9/12

**题目回顾**：含参问题：正三棱锥 $P-ABC$ 的底面边长为 $2$，设高 $PO = h$（$O$ 为底面中心）。以 $O$ 为原点建系。（1）写出 $P$ 和底面顶点坐标（用 $h$ 表示）；（2）求侧面 $PAB$ 的法向量（用 $h$ 表示）；（3）设侧面与底面所成二面角为 $\theta$，用 $h$ 表达 $\cos\theta$，并求当 $\theta = 60°$ 时 $h$ 的值。

**思路**：正三棱锥以底面中心 $O$ 建系；底面等边三角形各顶点到中心的距离（外接圆半径）为 $\dfrac{2}{\sqrt{3}} = \dfrac{2\sqrt{3}}{3}$；侧面法向量用联立；含参计算 $\cos\theta$，再代入 $60°$。→ toolkit/11 立体向量（含参二面角）。

**解答**：

**(1)** 底面等边三角形边长 $2$，中心 $O$ 到各顶点距离（外接圆半径）$R = \dfrac{2}{\sqrt{3}} = \dfrac{2\sqrt{3}}{3}$。

以 $O$ 为原点，$x$ 轴过 $A$ 点方向，$y$ 轴在底面内垂直 $x$ 轴，$z$ 轴向上：

$$A\left(\frac{2\sqrt{3}}{3}, 0, 0\right),\quad B\left(-\frac{\sqrt{3}}{3}, 1, 0\right),\quad C\left(-\frac{\sqrt{3}}{3}, -1, 0\right),\quad P(0, 0, h)$$

← 外接圆半径 $R = \frac{a}{\sqrt{3}} = \frac{2\sqrt{3}}{3}$

验证 $|AB|$：$\left(\frac{2\sqrt{3}}{3}+\frac{\sqrt{3}}{3}\right)^2 + 1^2 = \left(\sqrt{3}\right)^2 + 1 = 4$，$|AB| = 2$ ✓

**(2)** 平面 $PAB$ 的法向量（用 $h$）：

$$\overrightarrow{AP} = P - A = \left(-\frac{2\sqrt{3}}{3}, 0, h\right),\quad \overrightarrow{AB} = B - A = \left(-\sqrt{3}, 1, 0\right)$$

设 $\vec{n} = (a, b, c)$：

$$\vec{n}\cdot\overrightarrow{AP}: -\frac{2\sqrt{3}}{3}a + hc = 0 \Rightarrow a = \frac{3h}{2\sqrt{3}}c = \frac{\sqrt{3}h}{2}c \quad \cdots(1)$$

$$\vec{n}\cdot\overrightarrow{AB}: -\sqrt{3}a + b = 0 \Rightarrow b = \sqrt{3}a = \frac{3h}{2}c \quad \cdots(2)$$

← 联立方程

取 $c = 2$：$a = \sqrt{3}h$，$b = 3h$，法向量 $\vec{n}_1 = (\sqrt{3}h, 3h, 2)$。

**(3)** 求 $\cos\theta$（侧面 $PAB$ 与底面所成二面角）：

底面法向量 $\vec{n}_2 = (0, 0, 1)$（$z$ 轴）。

$$\cos\theta = \left|\frac{\vec{n}_1\cdot\vec{n}_2}{|\vec{n}_1||\vec{n}_2|}\right| = \frac{|2|}{\sqrt{3h^2+9h^2+4}\times 1} = \frac{2}{\sqrt{12h^2+4}} = \frac{2}{2\sqrt{3h^2+1}} = \frac{1}{\sqrt{3h^2+1}}$$

← 含参二面角

当 $\theta = 60°$ 时，$\cos 60° = \dfrac{1}{2}$：

$$\frac{1}{\sqrt{3h^2+1}} = \frac{1}{2} \Rightarrow \sqrt{3h^2+1} = 2 \Rightarrow 3h^2 = 3 \Rightarrow h = 1$$

← 代入求参数

**答案**：坐标见解答；法向量 $\vec{n}_1 = \boxed{(\sqrt{3}h, 3h, 2)}$；$\cos\theta = \boxed{\dfrac{1}{\sqrt{3h^2+1}}}$；$\theta = 60°$ 时 $h = \boxed{1}$。

**总结**：含参问题先写坐标，法向量含参，最后代入特殊角反解参数；正三棱锥外接圆半径 $R = \dfrac{a}{\sqrt{3}} = \dfrac{\sqrt{3}}{3}a$，是常用结论。

---

> **题数统计**
>
> | 分组 | 对应 Part | 题号范围 | 小计 |
> |------|-----------|----------|------|
> | 平面向量综合 | Part 1+2 | D.01–D.10 | 10 |
> | 空间向量 | Part 9 | D.73–D.87 | 15 |
> | **合计** | | **D.01–D.10, D.73–D.87** | **25** |
>
> **重点方法覆盖**：
> - 平面向量：数量积求夹角、基底分解（非正交基）、重心向量、分点参数、模的平方展开、垂直充要条件；
> - 空间向量：建系（正方体 / 棱柱 / 棱锥）、法向量联立方程组、二面角（法向量夹角）、点面距公式、线面角正弦公式、含参向量。
