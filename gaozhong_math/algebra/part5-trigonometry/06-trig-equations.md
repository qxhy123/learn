# 简单三角方程的求解

> **一例速记**：  
> 三类基本三角方程的通解：  
> $\sin x = a$（$|a| \leq 1$）：$x = (-1)^k \arcsin a + k\pi$（$k \in \mathbb{Z}$）  
> $\cos x = a$（$|a| \leq 1$）：$x = \pm\arccos a + 2k\pi$（$k \in \mathbb{Z}$）  
> $\tan x = a$：$x = \arctan a + k\pi$（$k \in \mathbb{Z}$）  
> **高中常用**：当 $a$ 是标准值时，直接写特殊角而非 $\arcsin/\arccos/\arctan$。

---

## 一、基本三角方程的通解

### 1.1 $\sin x = a$

当 $|a| > 1$ 时，方程无实数解（正弦函数值域为 $[-1, 1]$）。

当 $|a| \leq 1$ 时，设 $x_0 = \arcsin a \in \left[-\dfrac{\pi}{2}, \dfrac{\pi}{2}\right]$ 是一个特解，通解为：

$$x = (-1)^k \arcsin a + k\pi, \quad k \in \mathbb{Z}$$

**理解**：正弦函数的"对称性"——在 $[0, \pi]$ 内，$\sin x = a$ 的两根关于 $x = \dfrac{\pi}{2}$ 对称（若一根为 $x_0$，另一根为 $\pi - x_0$）；再加上 $2\pi$ 的周期性，得到全部解。

**高中常用形式**（$a$ 为特殊值）：

| $\sin x = a$ | 通解 |
|---|---|
| $\sin x = 0$ | $x = k\pi$ |
| $\sin x = 1$ | $x = \dfrac{\pi}{2} + 2k\pi$ |
| $\sin x = -1$ | $x = -\dfrac{\pi}{2} + 2k\pi$ |
| $\sin x = \dfrac{1}{2}$ | $x = \dfrac{\pi}{6} + 2k\pi$ 或 $x = \dfrac{5\pi}{6} + 2k\pi$ |
| $\sin x = \dfrac{\sqrt{2}}{2}$ | $x = \dfrac{\pi}{4} + 2k\pi$ 或 $x = \dfrac{3\pi}{4} + 2k\pi$ |
| $\sin x = \dfrac{\sqrt{3}}{2}$ | $x = \dfrac{\pi}{3} + 2k\pi$ 或 $x = \dfrac{2\pi}{3} + 2k\pi$ |

### 1.2 $\cos x = a$

当 $|a| \leq 1$ 时，通解为：

$$x = \pm\arccos a + 2k\pi, \quad k \in \mathbb{Z}$$

**理解**：余弦函数关于 $y$ 轴对称——$\cos x = \cos(-x)$，若一根为 $x_0$，则 $-x_0$ 也是根；加上 $2\pi$ 周期性。

**高中常用形式**：

| $\cos x = a$ | 通解 |
|---|---|
| $\cos x = 0$ | $x = \dfrac{\pi}{2} + k\pi$ |
| $\cos x = 1$ | $x = 2k\pi$ |
| $\cos x = -1$ | $x = \pi + 2k\pi$ |
| $\cos x = \dfrac{\sqrt{2}}{2}$ | $x = \pm\dfrac{\pi}{4} + 2k\pi$ |
| $\cos x = \dfrac{1}{2}$ | $x = \pm\dfrac{\pi}{3} + 2k\pi$ |
| $\cos x = -\dfrac{\sqrt{3}}{2}$ | $x = \pm\dfrac{5\pi}{6} + 2k\pi$ |

### 1.3 $\tan x = a$

正切函数周期为 $\pi$（比正弦余弦短），通解最简洁：

$$x = \arctan a + k\pi, \quad k \in \mathbb{Z}$$

**高中常用形式**：

| $\tan x = a$ | 通解 |
|---|---|
| $\tan x = 0$ | $x = k\pi$ |
| $\tan x = 1$ | $x = \dfrac{\pi}{4} + k\pi$ |
| $\tan x = \sqrt{3}$ | $x = \dfrac{\pi}{3} + k\pi$ |
| $\tan x = -1$ | $x = -\dfrac{\pi}{4} + k\pi$ |

---

## 二、常见类型的解法

### 2.1 换元化归为基本方程

**场景**：方程中含 $\omega x + \varphi$ 这样的"复合角"，令 $u = \omega x + \varphi$ 先求 $u$，再解 $x$。

**步骤**：
1. 令 $u = \omega x + \varphi$，方程化为 $\sin u = a$（或 $\cos u = a$，$\tan u = a$）
2. 用基本方程的通解求 $u$
3. 由 $u = \omega x + \varphi$ 解出 $x$，再结合题目给定的范围筛选

**典型示例**：解 $\cos\!\left(2x + \dfrac{\pi}{3}\right) = -\dfrac{1}{2}$（$x \in [0, \pi]$）

令 $u = 2x + \dfrac{\pi}{3}$，则 $\cos u = -\dfrac{1}{2}$，通解 $u = \pm\dfrac{2\pi}{3} + 2k\pi$。

因 $x \in [0, \pi]$，所以 $2x + \dfrac{\pi}{3} \in \left[\dfrac{\pi}{3}, \dfrac{7\pi}{3}\right]$。

- $u = \dfrac{2\pi}{3}$：$2x = \dfrac{\pi}{3}$，$x = \dfrac{\pi}{6}$ ✓（在 $[0,\pi]$ 内）
- $u = -\dfrac{2\pi}{3} + 2\pi = \dfrac{4\pi}{3}$：$2x = \pi$，$x = \dfrac{\pi}{2}$ ✓
- $u = \dfrac{2\pi}{3} + 2\pi = \dfrac{8\pi}{3}$：超出范围，舍去

答：$x = \dfrac{\pi}{6}$ 或 $x = \dfrac{\pi}{2}$。

### 2.2 可化为关于同一三角函数的方程

**场景**：方程中含不同三角函数，但可通过同角恒等式（如 $\cos^2 x = 1 - \sin^2 x$）化为同一三角函数的方程。

**步骤**：
1. 用 $\sin^2 x + \cos^2 x = 1$ 统一成 $\sin x$ 或 $\cos x$ 的一元方程
2. 令 $t = \sin x$（或 $\cos x$），化为关于 $t$ 的代数方程（常是一元二次方程）
3. 解 $t$，注意约束 $|t| \leq 1$（超出范围的 $t$ 值舍去）
4. 再解基本三角方程

**典型示例**：解 $2\sin^2 x - 3\sin x + 1 = 0$，$x \in [0, 2\pi)$

令 $t = \sin x$，则 $2t^2 - 3t + 1 = 0$，因式分解：$(2t-1)(t-1) = 0$。

解得 $t = \dfrac{1}{2}$ 或 $t = 1$。

- $\sin x = \dfrac{1}{2}$：$x = \dfrac{\pi}{6}$ 或 $x = \dfrac{5\pi}{6}$
- $\sin x = 1$：$x = \dfrac{\pi}{2}$

答：$x = \dfrac{\pi}{6}, \dfrac{\pi}{2}, \dfrac{5\pi}{6}$。

### 2.3 含 $\cos 2x$：降幂后化为一元二次

**场景**：见到 $\cos 2x$ 与 $\sin x$ 或 $\cos x$ 同时出现，用降幂公式把 $\cos 2x$ 换成单角的平方。

**典型示例**：解 $\cos 2x + 3\sin x = 2$，$x \in [0, 2\pi)$

用 $\cos 2x = 1 - 2\sin^2 x$：

$$1 - 2\sin^2 x + 3\sin x - 2 = 0 \implies 2\sin^2 x - 3\sin x + 1 = 0$$

这与 2.2 的例子完全相同，解法一致：$x = \dfrac{\pi}{6}, \dfrac{\pi}{2}, \dfrac{5\pi}{6}$。

> 关键一步：$\cos 2x$ 出现时，根据方程中其他项含 $\sin x$ 还是 $\cos x$ 来选对应的降幂公式（$\cos 2x = 1 - 2\sin^2 x$ 或 $\cos 2x = 2\cos^2 x - 1$），使整个方程统一成单一三角函数。

### 2.4 利用积化和差 / 和差化积

**场景**：方程含 $\sin mx \pm \sin nx$ 形式（$m \neq n$），用和差化积因式分解。

**典型示例**：解 $\sin 3x + \sin x = 0$，$x \in [0, 2\pi)$

$$\sin 3x + \sin x = 2\sin\frac{3x+x}{2}\cos\frac{3x-x}{2} = 2\sin 2x\cos x = 0$$

→ $\sin 2x = 0$ 或 $\cos x = 0$

- $\sin 2x = 0$：$2x = k\pi$，$x = \dfrac{k\pi}{2}$，在 $[0, 2\pi)$ 中：$x = 0, \dfrac{\pi}{2}, \pi, \dfrac{3\pi}{2}$
- $\cos x = 0$：$x = \dfrac{\pi}{2} + k\pi$，在 $[0, 2\pi)$ 中：$x = \dfrac{\pi}{2}, \dfrac{3\pi}{2}$（已包含在上组中）

答：$x = 0, \dfrac{\pi}{2}, \pi, \dfrac{3\pi}{2}$。

---

## 三、易错点

1. **$\sin x = a$ 的通解不要写成 $\pm\arcsin a + 2k\pi$**（那是余弦的形式）。正弦用 $(-1)^k$，余弦用 $\pm$。
2. **换元后忘记约束范围**：令 $u = \omega x + \varphi$ 后，先确定 $u$ 的范围，再筛选通解。漏写范围、忘记筛选是最常见的失分点。
3. **令 $t = \sin x$ 后忘记约束 $-1 \leq t \leq 1$**：若方程解出 $t = 2$ 之类的值，必须舍去，否则 $\sin x = 2$ 无解。
4. **$\cos 2x$ 的降幂方向要与其他项一致**：如果方程含 $\sin x$，用 $\cos 2x = 1 - 2\sin^2 x$；若含 $\cos x$，用 $\cos 2x = 2\cos^2 x - 1$。混用则方程不统一，无法化为一元。
5. **漏解**：对于 $\sin x = \dfrac{1}{2}$，在 $[0, 2\pi)$ 中有两个解（$\dfrac{\pi}{6}$ 和 $\dfrac{5\pi}{6}$），初学者容易只写一个。

---

## 四、典型例题

### 例 1：基本方程的通解

**题目**：求方程 $\sin\!\left(2x - \dfrac{\pi}{6}\right) = \dfrac{\sqrt{3}}{2}$ 的通解。

**【解答】**

令 $u = 2x - \dfrac{\pi}{6}$，则 $\sin u = \dfrac{\sqrt{3}}{2}$。

$\dfrac{\sqrt{3}}{2}$ 对应的特殊角：$\arcsin\dfrac{\sqrt{3}}{2} = \dfrac{\pi}{3}$。

通解：$u = \dfrac{\pi}{3} + 2k\pi$ 或 $u = \pi - \dfrac{\pi}{3} + 2k\pi = \dfrac{2\pi}{3} + 2k\pi$（$k \in \mathbb{Z}$）

回代 $u = 2x - \dfrac{\pi}{6}$：

**情形 1**：$2x - \dfrac{\pi}{6} = \dfrac{\pi}{3} + 2k\pi$，解得 $x = \dfrac{\pi}{4} + k\pi$

**情形 2**：$2x - \dfrac{\pi}{6} = \dfrac{2\pi}{3} + 2k\pi$，解得 $x = \dfrac{5\pi}{12} + k\pi$

通解：$x = \dfrac{\pi}{4} + k\pi$ 或 $x = \dfrac{5\pi}{12} + k\pi$（$k \in \mathbb{Z}$）。

---

### 例 2：含参数的三角方程

**题目**：解 $\cos^2 x - \sin x - 1 = 0$，$x \in [0, 2\pi)$。

**【解答】**

用 $\cos^2 x = 1 - \sin^2 x$ 统一：

$$1 - \sin^2 x - \sin x - 1 = 0 \implies \sin^2 x + \sin x = 0 \implies \sin x(\sin x + 1) = 0$$

故 $\sin x = 0$ 或 $\sin x = -1$。

- $\sin x = 0$：$x = 0, \pi$（在 $[0, 2\pi)$ 中）
- $\sin x = -1$：$x = \dfrac{3\pi}{2}$（在 $[0, 2\pi)$ 中）

答：$x = 0, \pi, \dfrac{3\pi}{2}$。

---

### 例 3：方程解的个数讨论

**题目**：若方程 $\cos x = m$ 在 $[0, 2\pi]$ 上有两个不同的实数解，求 $m$ 的取值范围。

**【解答】**

$y = \cos x$ 在 $[0, 2\pi]$ 上的图象：从 $\cos 0 = 1$ 递减到 $\cos\pi = -1$，再递增到 $\cos 2\pi = 1$。

水平线 $y = m$ 与图象的交点个数：
- $m = 1$：交点 $x = 0$ 和 $x = 2\pi$，共 2 个解
- $-1 < m < 1$：图象在下降段和上升段各经过一次，共 2 个解
- $m = -1$：交点只有 $x = \pi$，共 1 个解
- $m > 1$ 或 $m < -1$：无解

因此，方程在 $[0, 2\pi]$ 上有两个不同实数解的条件是：

$$\boxed{-1 < m \leq 1}$$

> 注意：$m = 1$ 时，$x = 0$ 和 $x = 2\pi$ 是两个不同的值，所以 $m = 1$ 包括在内；$m = -1$ 只有一个解，不包括。

---

## 五、思路自测题

**自测 1**　求 $\tan\!\left(x + \dfrac{\pi}{4}\right) = 1$ 的通解。

> 💡 提示：令 $u = x + \dfrac{\pi}{4}$，$\tan u = 1$，通解 $u = \dfrac{\pi}{4} + k\pi$。回代得 $x = k\pi$（$k \in \mathbb{Z}$）。

**自测 2**　解方程 $2\cos^2 x + \cos x - 1 = 0$，$x \in [0, 2\pi)$。

> 💡 提示：令 $t = \cos x$，$2t^2 + t - 1 = (2t-1)(t+1) = 0$，$t = \dfrac{1}{2}$ 或 $t = -1$。$\cos x = \dfrac{1}{2}$：$x = \dfrac{\pi}{3}$ 或 $x = \dfrac{5\pi}{3}$；$\cos x = -1$：$x = \pi$。答：$x = \dfrac{\pi}{3}, \pi, \dfrac{5\pi}{3}$。

**自测 3**　解 $\sin 2x = \cos x$，$x \in [0, 2\pi)$。

> 💡 提示：$\sin 2x = 2\sin x\cos x$，方程变为 $2\sin x\cos x - \cos x = 0$，即 $\cos x(2\sin x - 1) = 0$。$\cos x = 0$：$x = \dfrac{\pi}{2}, \dfrac{3\pi}{2}$；$\sin x = \dfrac{1}{2}$：$x = \dfrac{\pi}{6}, \dfrac{5\pi}{6}$。答：$x = \dfrac{\pi}{6}, \dfrac{\pi}{2}, \dfrac{5\pi}{6}, \dfrac{3\pi}{2}$。

**自测 4**　解 $\cos 2x - \cos x = 0$，求通解。

> 💡 提示：和差化积 $\cos 2x - \cos x = -2\sin\dfrac{3x}{2}\sin\dfrac{x}{2} = 0$。则 $\sin\dfrac{3x}{2} = 0$ 或 $\sin\dfrac{x}{2} = 0$。前者 $\dfrac{3x}{2} = k\pi$，$x = \dfrac{2k\pi}{3}$；后者 $\dfrac{x}{2} = k\pi$，$x = 2k\pi$。合并（$x = 2k\pi$ 已包含在 $x = \dfrac{2k\pi}{3}$ 中当 $k$ 是 $3$ 的倍数时），通解：$x = \dfrac{2k\pi}{3}$（$k \in \mathbb{Z}$）。

---

**回头看一眼"一例速记"**：

> $\sin x = a$：通解 $(-1)^k\arcsin a + k\pi$；$\cos x = a$：通解 $\pm\arccos a + 2k\pi$；$\tan x = a$：通解 $\arctan a + k\pi$。

如果现在你能不查公式，直接写出 $\sin\!\left(2x - \dfrac{\pi}{3}\right) = \dfrac{1}{2}$ 的通解——本章，你拿下了。
