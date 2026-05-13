# 三角函数的图象与性质

> **一例速记**：  
> $y = \sin x$ 和 $y = \cos x$ 是最基本的三角函数图象，周期为 $2\pi$，值域 $[-1, 1]$。  
> 一般化形式：$y = A\sin(\omega x + \varphi) + b$  
> - $|A|$：振幅（图象上下幅度的一半）  
> - $\dfrac{2\pi}{|\omega|}$：周期  
> - $-\dfrac{\varphi}{\omega}$：相位偏移（图象左右移动量）  
> - $b$：竖直平移量（图象上下移动量）

---

## 一、基本图象的由来：用单位圆追踪

### 1.1 正弦曲线的生成

想象角 $x$ 从 $0$ 开始逐渐增大，终边在单位圆上逆时针转动，对应的点 $P(x) = (\cos x, \sin x)$ 也随之移动。

把纵坐标 $y = \sin x$ 随 $x$ 变化的轨迹"展开"到以 $x$ 为横轴的坐标系中，就得到正弦曲线。

关键"节点"（记住这 5 个点就能画出一个完整周期）：

| $x$ | $0$ | $\dfrac{\pi}{2}$ | $\pi$ | $\dfrac{3\pi}{2}$ | $2\pi$ |
|-----|-----|------|------|------|------|
| $\sin x$ | $0$ | $1$ | $0$ | $-1$ | $0$ |

形状描述：从原点出发，先升至波峰 $\left(\dfrac{\pi}{2}, 1\right)$，回落过 $(\pi, 0)$，降至波谷 $\left(\dfrac{3\pi}{2}, -1\right)$，再回到 $(2\pi, 0)$，之后无限周期重复。

### 1.2 余弦曲线的生成

类似地，$y = \cos x$ 的关键节点：

| $x$ | $0$ | $\dfrac{\pi}{2}$ | $\pi$ | $\dfrac{3\pi}{2}$ | $2\pi$ |
|-----|-----|------|------|------|------|
| $\cos x$ | $1$ | $0$ | $-1$ | $0$ | $1$ |

从 $(0, 1)$ 出发，降至 $\left(\dfrac{\pi}{2}, 0\right)$，到达波谷 $(\pi, -1)$，回升过 $\left(\dfrac{3\pi}{2}, 0\right)$，回到 $(2\pi, 1)$。

**观察**：$y = \cos x$ 的图象可以通过把 $y = \sin x$ 的图象向左平移 $\dfrac{\pi}{2}$ 个单位得到，因为 $\cos x = \sin\!\left(x + \dfrac{\pi}{2}\right)$。

### 1.3 正切曲线

$y = \tan x$ 的定义域为 $x \ne \dfrac{\pi}{2} + k\pi$（$k \in \mathbb{Z}$），在每个开区间 $\left(-\dfrac{\pi}{2} + k\pi,\, \dfrac{\pi}{2} + k\pi\right)$ 内单调递增，值域为 $\mathbb{R}$。

关键节点（一个基本周期内）：

| $x$ | $-\dfrac{\pi}{3}$ | $-\dfrac{\pi}{4}$ | $0$ | $\dfrac{\pi}{4}$ | $\dfrac{\pi}{3}$ |
|-----|------|------|-----|------|------|
| $\tan x$ | $-\sqrt{3}$ | $-1$ | $0$ | $1$ | $\sqrt{3}$ |

图象在 $x = \dfrac{\pi}{2} + k\pi$ 处有**铅直渐近线**（函数值趋向 $\pm\infty$）。

---

## 二、三大函数性质对比表

| 性质 | $y = \sin x$ | $y = \cos x$ | $y = \tan x$ |
|------|------|------|------|
| **定义域** | $\mathbb{R}$ | $\mathbb{R}$ | $x \ne \dfrac{\pi}{2} + k\pi$ |
| **值域** | $[-1, 1]$ | $[-1, 1]$ | $\mathbb{R}$ |
| **周期** | $2\pi$ | $2\pi$ | $\pi$ |
| **奇偶性** | 奇函数 | 偶函数 | 奇函数 |
| **单调递增区间** | $\left[-\dfrac{\pi}{2}+2k\pi,\, \dfrac{\pi}{2}+2k\pi\right]$ | $[-\pi+2k\pi,\, 2k\pi]$ | $\left(-\dfrac{\pi}{2}+k\pi,\, \dfrac{\pi}{2}+k\pi\right)$ |
| **单调递减区间** | $\left[\dfrac{\pi}{2}+2k\pi,\, \dfrac{3\pi}{2}+2k\pi\right]$ | $[2k\pi,\, \pi+2k\pi]$ | 无减区间 |
| **对称轴** | $x = \dfrac{\pi}{2} + k\pi$ | $x = k\pi$ | 无对称轴 |
| **对称中心** | $(k\pi, 0)$ | $\left(\dfrac{\pi}{2}+k\pi, 0\right)$ | $\left(\dfrac{k\pi}{2}, 0\right)$ |
| **最大值** | $1$（在 $x = \dfrac{\pi}{2}+2k\pi$） | $1$（在 $x = 2k\pi$） | 无最大值 |
| **最小值** | $-1$（在 $x = -\dfrac{\pi}{2}+2k\pi$） | $-1$（在 $x = \pi+2k\pi$） | 无最小值 |

（表中 $k \in \mathbb{Z}$）

**记忆重点**：
- $\sin$ 在 $\left[-\dfrac{\pi}{2}, \dfrac{\pi}{2}\right]$ 上递增，包含 $0$，"$\sin$ 从 $-1$ 到 $1$ 上坡经过原点"
- $\cos$ 在 $[0, \pi]$ 上递减，"$\cos$ 从 $1$ 到 $-1$ 下坡"

---

## 三、$y = A\sin(\omega x + \varphi) + b$ 的图象变换

### 3.1 四个参数的意义

$$y = A\sin(\omega x + \varphi) + b \quad (A \ne 0,\; \omega \ne 0)$$

**振幅 $|A|$**：图象在竖直方向上的伸缩幅度。值域从 $[-1,1]$ 变为 $[b-|A|, b+|A|]$。

**角频率 $\omega$ 与周期 $T$**：$\omega x + \varphi$ 增加 $2\pi$ 所需的 $x$ 变化量就是周期：

$$T = \frac{2\pi}{|\omega|}$$

**初相 $\varphi$**：影响图象的左右位移。当 $\omega > 0$ 时，图象相比 $y = A\sin(\omega x) + b$ 向左移动 $\dfrac{\varphi}{\omega}$（$\varphi > 0$ 时）或向右移动 $\dfrac{|\varphi|}{\omega}$（$\varphi < 0$ 时）。

**竖直位移 $b$**：整条曲线上移 $b$ 个单位（$b > 0$）或下移 $|b|$ 个单位（$b < 0$）。

### 3.2 从 $y = \sin x$ 到 $y = A\sin(\omega x + \varphi) + b$ 的变换步骤

**标准变换顺序**（对 $x$ 先做水平变换，再做竖直变换）：

**步骤 1** 水平伸缩：$x \to \dfrac{x}{\omega}$，即把图象沿 $x$ 轴伸缩（$|\omega| > 1$ 压缩，$0 < |\omega| < 1$ 拉伸）。

**步骤 2** 水平平移：$x \to x + \dfrac{\varphi}{\omega}$，即平移 $\dfrac{\varphi}{\omega}$ 个单位（$\varphi > 0$ 时左移 $\dfrac{\varphi}{\omega}$）。

等价地，两步合并：把 $y = \sin x$ 的每个点的横坐标 $x_0$ 变为 $\dfrac{x_0 - \varphi}{\omega}$。

**步骤 3** 竖直伸缩：$y \to Ay$（若 $A < 0$ 还需关于 $x$ 轴翻转）。

**步骤 4** 竖直平移：$y \to y + b$。

### 3.3 平移规律总结

从 $y = A\sin(\omega x) + b$ 变为 $y = A\sin(\omega x + \varphi) + b$（$\varphi \ne 0$）：

- $\varphi > 0$：图象**向左**移动 $\dfrac{\varphi}{\omega}$ 个单位
- $\varphi < 0$：图象**向右**移动 $\dfrac{|\varphi|}{\omega}$ 个单位

**口诀**："加左减右"——$\varphi$ 加正数，图象向左移；减正数（即 $\varphi$ 为负），图象向右移。

**理解方式**：$A\sin(\omega x + \varphi)$ 在 $x = x_0$ 处的值等于 $A\sin(\omega x)$ 在 $x = x_0 + \dfrac{\varphi}{\omega}$ 处的值，所以原来在 $x_0 + \dfrac{\varphi}{\omega}$ 处发生的事情，现在提前在 $x_0$ 处就发生了——也就是图象**整体左移** $\dfrac{\varphi}{\omega}$（$\varphi > 0$ 时）。

### 3.4 从图象读出参数

**逆向问题**：已知图象，求 $A, \omega, \varphi, b$。

1. **读振幅**：$|A| = \dfrac{\text{最大值} - \text{最小值}}{2}$，$b = \dfrac{\text{最大值} + \text{最小值}}{2}$。

2. **读周期**：相邻两个同相位点（如两个波峰）之间的距离即周期 $T$，$\omega = \dfrac{2\pi}{T}$。

3. **读初相**：找图象上的某个"易识别"点（如过零点），由 $\omega x + \varphi = 0$（对应 $\sin = 0$ 的升零点）解出 $\varphi$；注意 $|\varphi| \leq \dfrac{\pi}{2}$ 的限制（高考通常加此约束）。

---

## 四、典型应用例题

### 例 1：求函数的振幅、周期、相位

**题目**：函数 $y = 3\sin\left(2x - \dfrac{\pi}{3}\right) + 1$，求振幅、周期、相位（初相）和竖直平移量，并写出递增区间。

**【解答】**

对照 $y = A\sin(\omega x + \varphi) + b$：$A = 3$，$\omega = 2$，$\varphi = -\dfrac{\pi}{3}$，$b = 1$。

**振幅**：$|A| = 3$，值域为 $[1-3, 1+3] = [-2, 4]$。

**周期**：$T = \dfrac{2\pi}{|\omega|} = \dfrac{2\pi}{2} = \pi$。

**图象平移**：与 $y = 3\sin(2x) + 1$ 相比，$\varphi = -\dfrac{\pi}{3} < 0$，图象**向右**平移 $\dfrac{\pi/3}{2} = \dfrac{\pi}{6}$ 个单位。

**递增区间**：$y = \sin u$ 在 $\left[-\dfrac{\pi}{2} + 2k\pi,\, \dfrac{\pi}{2} + 2k\pi\right]$ 上递增，令 $u = 2x - \dfrac{\pi}{3}$：

$$-\frac{\pi}{2} + 2k\pi \leq 2x - \frac{\pi}{3} \leq \frac{\pi}{2} + 2k\pi$$

$$-\frac{\pi}{2} + \frac{\pi}{3} + 2k\pi \leq 2x \leq \frac{\pi}{2} + \frac{\pi}{3} + 2k\pi$$

$$-\frac{\pi}{12} + k\pi \leq x \leq \frac{5\pi}{12} + k\pi$$

递增区间为 $\left[-\dfrac{\pi}{12} + k\pi,\, \dfrac{5\pi}{12} + k\pi\right]$（$k \in \mathbb{Z}$）。

---

### 例 2：由图象确定解析式

**题目**：已知 $f(x) = A\sin(\omega x + \varphi)$（$A > 0$，$\omega > 0$，$|\varphi| < \dfrac{\pi}{2}$），图象经过点 $(0, 1)$，且最高点为 $\left(\dfrac{\pi}{3}, 2\right)$，求 $f(x)$ 的解析式。

**【解答】**

由最高点 $y$ 坐标为 $2$：振幅 $A = 2$。

最高点时内层 $\omega x + \varphi = \dfrac{\pi}{2}$（正弦取最大值的条件），代入 $x = \dfrac{\pi}{3}$：

$$\omega \cdot \frac{\pi}{3} + \varphi = \frac{\pi}{2} \quad \cdots (1)$$

代入点 $(0, 1)$：$A\sin\varphi = 1$，即 $2\sin\varphi = 1$，故 $\sin\varphi = \dfrac{1}{2}$。

由 $|\varphi| < \dfrac{\pi}{2}$，得 $\varphi = \dfrac{\pi}{6}$。

代入 (1)：$\dfrac{\omega\pi}{3} = \dfrac{\pi}{2} - \dfrac{\pi}{6} = \dfrac{\pi}{3}$，故 $\omega = 1$。

$$\boxed{f(x) = 2\sin\left(x + \frac{\pi}{6}\right)}$$

---

### 例 3：求最值与单调区间

**题目**：求 $f(x) = 1 - 2\sin^2 x + 2\sin x\cos x$ 的最大值和最小值。

**【解答】**

利用二倍角公式化简（呼应后续章节）：

$$f(x) = 1 - (1 - \cos 2x) + \sin 2x = \cos 2x + \sin 2x$$

用辅助角公式：$a\sin\theta + b\cos\theta = \sqrt{a^2+b^2}\sin(\theta + \varphi)$，其中 $\tan\varphi = \dfrac{b}{a}$：

$$f(x) = \sin 2x + \cos 2x = \sqrt{2}\sin\!\left(2x + \frac{\pi}{4}\right)$$

所以 $A = \sqrt{2}$，最大值为 $\sqrt{2}$，最小值为 $-\sqrt{2}$。

---

## 五、易错点梳理

**易错 1**：周期公式写错。$y = \sin(\omega x)$ 的周期是 $\dfrac{2\pi}{|\omega|}$，不是 $\dfrac{2\pi}{\omega}$（$\omega$ 可能为负数）。

**易错 2**：搞混"左移/右移"方向。$y = \sin(x + \varphi)$ 中，$\varphi > 0$ 是**左移**。很多同学直觉上以为"$+\varphi$"是"向右"，这是典型的反直觉错误。**验证方法**：取 $\varphi = \dfrac{\pi}{2}$，$\sin(x + \dfrac{\pi}{2}) = \cos x$，$\cos x$ 的图象确实是 $\sin x$ 图象**左移** $\dfrac{\pi}{2}$。

**易错 3**：求单调区间时，直接写 $\sin x$ 的单调区间，忘记换元回 $x$。求 $y = \sin(2x - 1)$ 的单调区间时，要设 $u = 2x - 1$，对 $u$ 解不等式，再换回 $x$。

**易错 4**：混淆振幅 $A$ 与最大值。最大值是 $A + b$，不是 $A$（要加上竖直位移 $b$）。

**易错 5**：$y = \tan x$ 的单调区间写成 $\mathbb{R}$。正切函数在**每个**区间 $\left(-\dfrac{\pi}{2}+k\pi, \dfrac{\pi}{2}+k\pi\right)$ 内单调递增，但不能跨越渐近线把多个区间合并成一个。

---

## 六、自测题

**自测 1**　写出 $y = \cos x$ 在 $[0, 2\pi]$ 上的单调递减区间，并指出在 $[0, 2\pi]$ 上所有对称轴和对称中心。

> 💡 提示：$y = \cos x$ 在 $[0, \pi]$ 上递减。对称轴为 $x = 0, \pi, 2\pi$（$y = \pm 1$ 处）；对称中心为 $\left(\dfrac{\pi}{2}, 0\right), \left(\dfrac{3\pi}{2}, 0\right)$。

**自测 2**　函数 $y = 2\sin\left(3x + \dfrac{\pi}{6}\right) - 1$，求：(1) 振幅和周期；(2) 值域；(3) 最小正相位偏移。

> 💡 提示：(1) $A = 2$，$T = \dfrac{2\pi}{3}$。(2) 值域 $= [-2-1, 2-1] = [-3, 1]$。(3) 相比 $y = 2\sin(3x) - 1$，图象向左移 $\dfrac{\pi/6}{3} = \dfrac{\pi}{18}$。

**自测 3**　已知函数图象相邻两个零点之间距离为 $\pi$（且在该区间内函数先正后负），最大值为 $4$，试写出一个满足条件的 $y = A\sin(\omega x + \varphi)$ 的表达式（$\omega > 0$，$A > 0$，$|\varphi| \leq \dfrac{\pi}{2}$）。

> 💡 提示：零点间距为 $\dfrac{T}{2} = \pi$，所以 $T = 2\pi$，$\omega = 1$。最大值为 $A = 4$。先正后负意味着从上升零点到下降零点，即最大值在两零点中间，可取 $\varphi = 0$（或其他满足条件的值）。一种答案：$y = 4\sin x$。

**自测 4**　比较 $f(x) = \sin 2x$ 和 $g(x) = \sin\left(2x + \dfrac{\pi}{6}\right)$ 的关系：$g(x)$ 的图象是怎么由 $f(x)$ 变换得到的？

> 💡 提示：$g(x) = \sin\left(2x + \dfrac{\pi}{6}\right) = f\!\left(x + \dfrac{\pi}{12}\right)$，所以 $g$ 的图象是 $f$ 的图象**向左平移** $\dfrac{\pi}{12}$ 个单位得到的。

**自测 5**　已知 $y = A\sin(\omega x + \varphi)$，图象最高点坐标为 $\left(\dfrac{\pi}{4}, 3\right)$，相邻最高点与最低点之间的水平距离为 $\pi$，且 $\omega > 0$，$A > 0$，$0 \leq \varphi \leq \pi$，求解析式。

> 💡 提示：最高点与最低点水平距离 $= \dfrac{T}{2} = \pi$，故 $T = 2\pi$，$\omega = 1$。$A = 3$。最高点时 $x + \varphi = \dfrac{\pi}{2}$，代入 $x = \dfrac{\pi}{4}$：$\varphi = \dfrac{\pi}{4}$（在 $[0, \pi]$ 内）。答案：$y = 3\sin\!\left(x + \dfrac{\pi}{4}\right)$。

---

**回头看一眼"一例速记"**：

> $y = A\sin(\omega x + \varphi) + b$：振幅 $|A|$，周期 $\dfrac{2\pi}{|\omega|}$，左右移 $-\dfrac{\varphi}{\omega}$，上下移 $b$。求单调区间必须换元。$\sin$ 是奇函数，$\cos$ 是偶函数，$\tan$ 无界无极值。

如果你现在能说出 $y = -2\sin(3x - \pi)$ 的振幅、周期，并判断其图象相对 $y = \sin(3x)$ 的变换——本章，你拿下了。
