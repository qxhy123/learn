# 三角函数的定义（单位圆视角）

> **一例速记**：  
> 以原点为圆心、$1$ 为半径的圆叫**单位圆**。角 $\alpha$ 的终边与单位圆的交点记为 $P(x, y)$，则：  
> $$\sin\alpha = y,\quad \cos\alpha = x,\quad \tan\alpha = \frac{y}{x}\ (x \ne 0)$$  
> **象限口诀**：一全二正弦三切四余弦（第几象限哪个函数为正）  
> **特殊值牢记**：$\sin 0 = 0, \sin\dfrac{\pi}{6} = \dfrac{1}{2}, \sin\dfrac{\pi}{4} = \dfrac{\sqrt{2}}{2}, \sin\dfrac{\pi}{3} = \dfrac{\sqrt{3}}{2}, \sin\dfrac{\pi}{2} = 1$

---

## 一、概念引入：从几何比值到函数

初中定义的正弦、余弦、正切都是**直角三角形中**的比值，只对 $0°$ 到 $90°$ 之间的锐角有意义。但现实中很多问题——比如描述物体的振动、旋转运动——涉及到钝角乃至任意角。

我们需要一种**对任意角都成立**的三角函数定义。单位圆提供了完美的框架。

**直观想法**：把角放在坐标系里，让终边与圆相交，用交点坐标来定义三角函数。为了让坐标直接就是"值"（不需要除以半径），选用**半径为 $1$ 的圆**，即单位圆。

---

## 二、单位圆定义三角函数

### 2.1 基本定义

在直角坐标系中，以原点 $O$ 为圆心、$1$ 为半径的圆叫做**单位圆**，方程为 $x^2 + y^2 = 1$。

设 $\alpha$ 是任意一个角，把它的顶点放在原点，初始边与 $x$ 轴正半轴重合，终边与单位圆的交点为 $P(x, y)$。

$$\boxed{\sin\alpha = y,\quad \cos\alpha = x,\quad \tan\alpha = \frac{y}{x}\ (x \ne 0)}$$

**几何意义**：
- $\sin\alpha$ 是终边上单位圆交点的**纵坐标**
- $\cos\alpha$ 是终边上单位圆交点的**横坐标**
- $\tan\alpha$ 是终边的**斜率**（斜率 $= \dfrac{y}{x}$）

这个定义对一切角（正角、负角、零角、超过 $2\pi$ 的角）都成立。

### 2.2 定义域与特殊情况

由定义，$\sin\alpha$ 和 $\cos\alpha$ 对所有 $\alpha \in \mathbb{R}$ 都有意义（单位圆上的点 $(x,y)$ 总是存在的）。

$\tan\alpha = \dfrac{y}{x}$ 要求 $x \ne 0$，即终边不能落在 $y$ 轴上。$x = 0$ 当且仅当终边在 $y$ 轴上，即 $\alpha = \dfrac{\pi}{2} + k\pi$（$k \in \mathbb{Z}$）。

$$\text{正切函数定义域：} \alpha \ne \frac{\pi}{2} + k\pi,\ k \in \mathbb{Z}$$

### 2.3 余切函数（了解）

类似地，余切函数 $\cot\alpha = \dfrac{x}{y}$（$y \ne 0$），高中阶段以 $\sin, \cos, \tan$ 为主。

---

## 三、各象限三角函数的符号

### 3.1 符号规律

单位圆上的点 $(x, y)$ 在不同象限中，$x, y$ 的正负不同，从而三角函数值的符号也不同：

| 象限 | $x$（$\cos\alpha$）| $y$（$\sin\alpha$）| $y/x$（$\tan\alpha$）|
|------|--------|--------|--------|
| 第一象限 | $+$ | $+$ | $+$ |
| 第二象限 | $-$ | $+$ | $-$ |
| 第三象限 | $-$ | $-$ | $+$ |
| 第四象限 | $+$ | $-$ | $-$ |

### 3.2 口诀记忆

**"一全二正弦三切四余弦"**：

- **第一**象限：**全**部（$\sin, \cos, \tan$）均为正
- **第二**象限：仅 **$\sin$**（正弦）为正
- **第三**象限：仅 **$\tan$**（正切）为正
- **第四**象限：仅 **$\cos$**（余弦）为正

**验证直觉**：第二象限中 $x < 0, y > 0$，$\sin\alpha = y > 0$（正），$\cos\alpha = x < 0$（负），$\tan\alpha = y/x < 0$（负）。完全吻合。

---

## 四、特殊角的三角函数值

### 4.1 单位圆上的特殊点

利用几何关系，可以精确求出以下特殊角对应的单位圆交点坐标：

| $\alpha$ | $0$ | $\dfrac{\pi}{6}$ | $\dfrac{\pi}{4}$ | $\dfrac{\pi}{3}$ | $\dfrac{\pi}{2}$ |
|----------|-----|-----|-----|-----|-----|
| 交点 $P(x,y)$ | $(1, 0)$ | $\left(\dfrac{\sqrt{3}}{2}, \dfrac{1}{2}\right)$ | $\left(\dfrac{\sqrt{2}}{2}, \dfrac{\sqrt{2}}{2}\right)$ | $\left(\dfrac{1}{2}, \dfrac{\sqrt{3}}{2}\right)$ | $(0, 1)$ |

### 4.2 特殊角三角函数值完整表

| $\alpha$ | $0$ | $\dfrac{\pi}{6}$ | $\dfrac{\pi}{4}$ | $\dfrac{\pi}{3}$ | $\dfrac{\pi}{2}$ | $\pi$ | $\dfrac{3\pi}{2}$ | $2\pi$ |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $\sin\alpha$ | $0$ | $\dfrac{1}{2}$ | $\dfrac{\sqrt{2}}{2}$ | $\dfrac{\sqrt{3}}{2}$ | $1$ | $0$ | $-1$ | $0$ |
| $\cos\alpha$ | $1$ | $\dfrac{\sqrt{3}}{2}$ | $\dfrac{\sqrt{2}}{2}$ | $\dfrac{1}{2}$ | $0$ | $-1$ | $0$ | $1$ |
| $\tan\alpha$ | $0$ | $\dfrac{\sqrt{3}}{3}$ | $1$ | $\sqrt{3}$ | 不存在 | $0$ | 不存在 | $0$ |

**记忆技巧**：

- $\sin$ 在 $\left[0, \dfrac{\pi}{2}\right]$ 上依次为 $0, \dfrac{1}{2}, \dfrac{\sqrt{2}}{2}, \dfrac{\sqrt{3}}{2}, 1$，可以记成"$0, 1, 2, 3, 4$"然后开根号再除以 $2$（$\dfrac{\sqrt{0}}{2}, \dfrac{\sqrt{1}}{2}, \dfrac{\sqrt{2}}{2}, \dfrac{\sqrt{3}}{2}, \dfrac{\sqrt{4}}{2}$）。
- $\cos$ 的顺序恰好相反，从 $1$ 到 $0$。

---

## 五、三角函数的基本性质

### 5.1 奇偶性

**正弦函数 $\sin\alpha$**：

角 $\alpha$ 与角 $-\alpha$ 的终边关于 $x$ 轴（即余弦轴）对称。若 $\alpha$ 的终边交点为 $(x, y)$，则 $-\alpha$ 的终边交点为 $(x, -y)$。

$$\sin(-\alpha) = -y = -\sin\alpha$$

因此 $\sin\alpha$ 是**奇函数**，图象关于原点对称。

**余弦函数 $\cos\alpha$**：

$$\cos(-\alpha) = x = \cos\alpha$$

因此 $\cos\alpha$ 是**偶函数**，图象关于 $y$ 轴对称。

**正切函数 $\tan\alpha$**：

$$\tan(-\alpha) = \frac{-y}{x} = -\frac{y}{x} = -\tan\alpha$$

因此 $\tan\alpha$ 是**奇函数**，图象关于原点对称。

### 5.2 周期性

终边旋转一整圈（$2\pi$）回到原位，交点坐标不变，所以三角函数值不变：

$$\sin(\alpha + 2\pi) = \sin\alpha,\quad \cos(\alpha + 2\pi) = \cos\alpha$$

因此 $\sin$ 和 $\cos$ 都是以 $2\pi$ 为周期的**周期函数**。

对于正切函数，旋转半圈（$\pi$）后终边方向相同（过原点的直线方向不变），斜率不变：

$$\tan(\alpha + \pi) = \tan\alpha$$

因此 $\tan$ 是以 $\pi$ 为周期的周期函数。

### 5.3 有界性

由于单位圆上点的坐标满足 $x^2 + y^2 = 1$，所以 $-1 \leq x \leq 1$ 且 $-1 \leq y \leq 1$，即：

$$-1 \leq \sin\alpha \leq 1,\quad -1 \leq \cos\alpha \leq 1$$

$\tan\alpha$ 的值域为 $\mathbb{R}$（无界）。

---

## 六、典型应用例题

### 例 1：已知三角函数值求坐标

**题目**：设角 $\alpha$ 的终边上（除原点外）有一点 $P(-3, 4)$，求 $\sin\alpha, \cos\alpha, \tan\alpha$。

**【解答】**

这里 $P$ 不在单位圆上，但可以用比例关系。终边方向确定，到单位圆的交点只是等比缩放。

$|OP| = \sqrt{(-3)^2 + 4^2} = \sqrt{9 + 16} = 5$

在半径为 $r = |OP| = 5$ 的圆上，同样的定义给出：

$$\sin\alpha = \frac{y}{r} = \frac{4}{5},\quad \cos\alpha = \frac{x}{r} = \frac{-3}{5},\quad \tan\alpha = \frac{y}{x} = \frac{4}{-3} = -\frac{4}{3}$$

（注意 $P$ 在第二象限：$x < 0, y > 0$，$\sin\alpha > 0$，$\cos\alpha < 0$，$\tan\alpha < 0$，与象限口诀吻合。）

---

### 例 2：利用周期和奇偶性化简

**题目**：求 $\sin\left(-\dfrac{13\pi}{6}\right)$ 的值。

**【解答】**

**方法一**：先用奇函数性：

$$\sin\left(-\frac{13\pi}{6}\right) = -\sin\left(\frac{13\pi}{6}\right)$$

再利用周期性：

$$\frac{13\pi}{6} = 2\pi + \frac{\pi}{6}$$

$$\sin\left(\frac{13\pi}{6}\right) = \sin\left(2\pi + \frac{\pi}{6}\right) = \sin\frac{\pi}{6} = \frac{1}{2}$$

所以 $\sin\left(-\dfrac{13\pi}{6}\right) = -\dfrac{1}{2}$。

---

### 例 3：判断象限并求三角函数值

**题目**：已知 $\sin\alpha = -\dfrac{5}{13}$，且 $\alpha$ 在第三象限，求 $\cos\alpha$ 和 $\tan\alpha$。

**【解答】**

$\alpha$ 在第三象限，$\cos\alpha < 0$（由象限口诀，第三象限只有 $\tan$ 为正）。

由 $\sin^2\alpha + \cos^2\alpha = 1$：

$$\cos^2\alpha = 1 - \sin^2\alpha = 1 - \left(-\frac{5}{13}\right)^2 = 1 - \frac{25}{169} = \frac{144}{169}$$

因 $\cos\alpha < 0$，取负值：$\cos\alpha = -\dfrac{12}{13}$

$$\tan\alpha = \frac{\sin\alpha}{\cos\alpha} = \frac{-5/13}{-12/13} = \frac{5}{12}$$

---

## 七、思路自测题

**自测 1**　判断：$\sin\alpha > 0$ 能否确定 $\alpha$ 所在象限？

> 💡 提示：$\sin\alpha > 0$ 只能确定终边在 $x$ 轴上方（$y > 0$），即 $\alpha$ 在第一或第二象限（或 $y$ 轴正半轴上）。不能唯一确定象限。

**自测 2**　已知 $\cos\alpha = -\dfrac{3}{5}$，$\tan\alpha > 0$，求 $\sin\alpha$ 的值。

> 💡 提示：$\cos\alpha < 0$（第二或三象限），$\tan\alpha > 0$（第一或三象限），两者取交集 → 第三象限。$\sin\alpha < 0$。$\sin^2\alpha = 1 - \cos^2\alpha = 1 - \dfrac{9}{25} = \dfrac{16}{25}$，$\sin\alpha = -\dfrac{4}{5}$。

**自测 3**　化简 $\cos\left(\dfrac{11\pi}{4}\right)$。

> 💡 提示：$\dfrac{11\pi}{4} = 2\pi + \dfrac{3\pi}{4}$，所以 $\cos\dfrac{11\pi}{4} = \cos\dfrac{3\pi}{4}$。$\dfrac{3\pi}{4}$ 在第二象限，$\cos\dfrac{3\pi}{4} = -\cos\dfrac{\pi}{4} = -\dfrac{\sqrt{2}}{2}$（用后面学的诱导公式，或直接查特殊角表）。

**自测 4**　若 $f(\alpha) = \dfrac{\sin(\pi + \alpha)\cos(2\pi - \alpha)}{\tan(-\alpha - \pi)}$，化简此式。

> 💡 提示：$\sin(\pi + \alpha) = -\sin\alpha$；$\cos(2\pi - \alpha) = \cos(-\alpha) = \cos\alpha$；$\tan(-\alpha - \pi) = \tan(-\alpha) = -\tan\alpha$。所以 $f(\alpha) = \dfrac{(-\sin\alpha)\cos\alpha}{-\tan\alpha} = \dfrac{-\sin\alpha \cos\alpha}{-\sin\alpha/\cos\alpha} = \cos^2\alpha$。

---

**回头看一眼"一例速记"**：

> 单位圆上终边交点 $(x, y)$ 满足 $\cos\alpha = x$，$\sin\alpha = y$，$\tan\alpha = y/x$。口诀：一全二正弦三切四余弦。$\sin$ 是奇函数，$\cos$ 是偶函数，周期均为 $2\pi$；$\tan$ 是奇函数，周期为 $\pi$。

如果你现在能直接写出 $\sin\dfrac{2\pi}{3}$ 的值，并说明理由——本章，你拿下了。
