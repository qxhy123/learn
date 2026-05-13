# 同角三角恒等式与诱导公式

> **一例速记**：  
> **三大同角恒等式**（$\alpha$ 相同，变换函数种类）：  
> $$\sin^2\alpha + \cos^2\alpha = 1$$  
> $$\tan\alpha = \frac{\sin\alpha}{\cos\alpha} \quad (\cos\alpha \ne 0)$$  
> $$1 + \tan^2\alpha = \sec^2\alpha \quad (\cos\alpha \ne 0)$$  
>
> **诱导公式口诀**：**奇变偶不变，符号看象限**  
> "奇偶"指 $\dfrac{\pi}{2}$ 的倍数是奇数倍还是偶数倍；"变"指正弦、余弦互换；"符号"指把 $\alpha$ 当锐角，看所在象限的原函数符号。

---

## 一、引入：一道看似简单却暗藏玄机的题

> **题目**：已知 $\sin\alpha = \dfrac{3}{5}$，且 $\alpha$ 在第二象限，求 $\cos\alpha$ 和 $\tan\alpha$。

请先停下来，自己试着做一做。

很多同学会直接写 $\cos\alpha = \dfrac{4}{5}$——这是**错的**，因为忽略了"第二象限"这个条件。另一种错误是：$\cos\alpha = \pm\dfrac{4}{5}$，然后停在这里，不知道取哪个。

正确的解法需要同时用到三个东西：**同角恒等式**（把 $\cos\alpha$ 与已知量联系起来）、**象限信息**（确定符号）、**规范步骤**（先算绝对值，再定符号）。

下面我们来还原一个真正掌握这道题的人，脑子里是怎么运转的。

---

## 二、思维路径还原（解题者的内心独白）

> "题目给了 $\sin\alpha = \dfrac{3}{5}$，$\alpha$ 在第二象限，让我求 $\cos\alpha$ 和 $\tan\alpha$。
>
> **第一反应：整理已知信息，明确目标。**
>
> 已知：$\sin\alpha = \dfrac{3}{5}$，$\alpha \in$ 第二象限。
>
> 目标：$\cos\alpha = ?$，$\tan\alpha = ?$
>
> **联系已知与未知——找公式。**
>
> $\sin\alpha$ 和 $\cos\alpha$ 之间有什么关系？最基本的就是勾股恒等式：$\sin^2\alpha + \cos^2\alpha = 1$。这条式子把两者绑在一起，知道一个可以求另一个——但只能求到绝对值（因为开根号有正负两解），符号需要额外条件。
>
> **先算 $|\cos\alpha|$（绝对值阶段）：**
>
> $$\cos^2\alpha = 1 - \sin^2\alpha = 1 - \left(\frac{3}{5}\right)^2 = 1 - \frac{9}{25} = \frac{16}{25}$$
>
> $$|\cos\alpha| = \frac{4}{5}$$
>
> **再定符号（象限阶段）：**
>
> $\alpha$ 在第二象限，$x < 0$，根据单位圆定义 $\cos\alpha = x$，所以 $\cos\alpha < 0$。
>
> 因此 $\cos\alpha = -\dfrac{4}{5}$（取负值）。
>
> **求 $\tan\alpha$（用商式恒等式）：**
>
> $$\tan\alpha = \frac{\sin\alpha}{\cos\alpha} = \frac{3/5}{-4/5} = \frac{3}{5} \times \left(-\frac{5}{4}\right) = -\frac{3}{4}$$
>
> **符号检验（用象限口诀）：**
>
> 第二象限：$\sin > 0$（正弦为正，符合 $\sin\alpha = \dfrac{3}{5} > 0$），$\cos < 0$（余弦为负，符合），$\tan < 0$（正切为负，符合 $\tan\alpha = -\dfrac{3}{4} < 0$）。全部自洽！
>
> **反思：这道题的核心结构。**
>
> 一共三步：① 用 $\sin^2 + \cos^2 = 1$ 求 $|\cos\alpha|$；② 用象限信息定符号；③ 用 $\tan = \sin/\cos$ 求正切。
>
> 每一步的依据分别是：同角恒等式（勾股型）、单位圆的坐标符号规律、同角恒等式（商型）。
>
> **什么地方最容易出错？**
>
> 只用 $\sin^2 + \cos^2 = 1$ 得到 $\cos^2\alpha = \dfrac{16}{25}$，然后写 $\cos\alpha = \dfrac{4}{5}$——忘了取负根。题目说"第二象限"就是在提示 $\cos < 0$，这不是废话，是关键条件。
>
> 另一个坑：$\tan\alpha$ 用 $\dfrac{\sin\alpha}{\cos\alpha}$ 时，分母 $\cos\alpha = -\dfrac{4}{5}$ 不为零，没问题；但如果 $\cos\alpha = 0$（即 $\alpha = \dfrac{\pi}{2} + k\pi$），$\tan\alpha$ 不存在，不能用这个公式。
>
> **把这道题泛化成一般流程：**
>
> 已知 $\sin\alpha$（或 $\cos\alpha$）和象限 → 用勾股恒等式求另一个函数的绝对值 → 用象限定符号 → 用商式恒等式求 $\tan\alpha$。"

把这段内心独白读两到三遍，把"先求绝对值，再定符号，最后用商式"的节奏刻进记忆中。

---

## 三、三大同角恒等式

### 3.1 勾股型恒等式（最重要）

$$\boxed{\sin^2\alpha + \cos^2\alpha = 1}$$

**证明**（用单位圆）：单位圆上终边交点 $P(\cos\alpha, \sin\alpha)$，由单位圆方程 $x^2 + y^2 = 1$ 代入即得。

**变形形式**（考试频繁用到）：

$$\sin^2\alpha = 1 - \cos^2\alpha$$

$$\cos^2\alpha = 1 - \sin^2\alpha$$

$$(\sin\alpha + \cos\alpha)^2 = 1 + 2\sin\alpha\cos\alpha$$

$$(\sin\alpha - \cos\alpha)^2 = 1 - 2\sin\alpha\cos\alpha$$

### 3.2 商式恒等式

$$\boxed{\tan\alpha = \frac{\sin\alpha}{\cos\alpha}} \quad (\cos\alpha \ne 0)$$

**证明**：单位圆上 $\tan\alpha = \dfrac{y}{x} = \dfrac{\sin\alpha}{\cos\alpha}$，直接由定义得到。

### 3.3 平方和型恒等式

$$\boxed{1 + \tan^2\alpha = \frac{1}{\cos^2\alpha} = \sec^2\alpha} \quad (\cos\alpha \ne 0)$$

**证明**：$\sin^2\alpha + \cos^2\alpha = 1$ 两边除以 $\cos^2\alpha$：

$$\frac{\sin^2\alpha}{\cos^2\alpha} + 1 = \frac{1}{\cos^2\alpha}$$

$$\tan^2\alpha + 1 = \sec^2\alpha$$

**高中使用频率**：前两个恒等式是核心，第三个在需要关联 $\tan\alpha$ 与 $\cos\alpha$ 时才用到。

---

## 四、诱导公式：将任意角化为锐角

### 4.1 什么是诱导公式？

诱导公式的功能是：把 $\dfrac{k\pi}{2} \pm \alpha$（$k \in \mathbb{Z}$）的三角函数值，化为 $\alpha$ 的三角函数值（通常取 $\alpha$ 为锐角，方便查特殊值表）。

有了诱导公式，无论碰到 $\sin\dfrac{5\pi}{3}$、$\cos(-\dfrac{7\pi}{4})$ 这样的"奇怪角"，都能化为我们熟悉的锐角三角函数值。

### 4.2 六组诱导公式

设 $\alpha$ 为任意角，列出最常用的六组公式：

**第一组（$-\alpha$）：**

$$\sin(-\alpha) = -\sin\alpha, \quad \cos(-\alpha) = \cos\alpha, \quad \tan(-\alpha) = -\tan\alpha$$

（正弦、正切是奇函数；余弦是偶函数。）

**第二组（$\pi + \alpha$）：**

$$\sin(\pi + \alpha) = -\sin\alpha, \quad \cos(\pi + \alpha) = -\cos\alpha, \quad \tan(\pi + \alpha) = \tan\alpha$$

（旋转半圈，终边关于原点对称，$x, y$ 均取反，故正弦、余弦均变符号；而 $\tan = y/x$ 符号抵消，不变。）

**第三组（$\pi - \alpha$）：**

$$\sin(\pi - \alpha) = \sin\alpha, \quad \cos(\pi - \alpha) = -\cos\alpha, \quad \tan(\pi - \alpha) = -\tan\alpha$$

**第四组（$2\pi - \alpha$）：**

$$\sin(2\pi - \alpha) = -\sin\alpha, \quad \cos(2\pi - \alpha) = \cos\alpha, \quad \tan(2\pi - \alpha) = -\tan\alpha$$

（等价于 $\sin(-\alpha)$ 等，因为 $2\pi - \alpha \equiv -\alpha \pmod{2\pi}$。）

**第五组（$\dfrac{\pi}{2} - \alpha$）：**

$$\sin\!\left(\frac{\pi}{2} - \alpha\right) = \cos\alpha, \quad \cos\!\left(\frac{\pi}{2} - \alpha\right) = \sin\alpha, \quad \tan\!\left(\frac{\pi}{2} - \alpha\right) = \frac{1}{\tan\alpha}$$

（正弦、余弦互换！这是"余角关系"：$\sin$ 和 $\cos$ 名字里的"余"正是"余角"之意。）

**第六组（$\dfrac{\pi}{2} + \alpha$）：**

$$\sin\!\left(\frac{\pi}{2} + \alpha\right) = \cos\alpha, \quad \cos\!\left(\frac{\pi}{2} + \alpha\right) = -\sin\alpha, \quad \tan\!\left(\frac{\pi}{2} + \alpha\right) = -\frac{1}{\tan\alpha}$$

### 4.3 口诀"奇变偶不变，符号看象限"详解

**"奇"和"偶"指的是什么？**

观察 $\dfrac{k\pi}{2}$ 中的系数 $k$：
- $k$ 为**奇数**（$1, 3, 5, \ldots$）：$\dfrac{\pi}{2}, \dfrac{3\pi}{2}$……称为"奇数倍 $\dfrac{\pi}{2}$"
- $k$ 为**偶数**（$0, 2, 4, \ldots$）：$0, \pi, 2\pi$……称为"偶数倍 $\dfrac{\pi}{2}$"

**"变"和"不变"指的是函数名称：**

- $k$ 为奇数 → $\sin$ 和 $\cos$ 互换（$\sin \leftrightarrow \cos$），简称"**奇变**"
- $k$ 为偶数 → 函数名称不变，$\sin$ 还是 $\sin$，$\cos$ 还是 $\cos$，简称"**偶不变**"

**"符号看象限"：**

把 $\alpha$ 视为第一象限角（锐角），看 $\dfrac{k\pi}{2} \pm \alpha$ 落在哪个象限，然后用该象限的原函数正负来确定符号。

**举例说明**：

- $\sin\!\left(\pi + \alpha\right)$（$k = 2$，偶数）→ 不变，仍为 $\sin$；$\pi + \alpha$ 在第三象限（$\alpha$ 是锐角时），第三象限 $\sin < 0$，所以 $\sin(\pi + \alpha) = -\sin\alpha$。
- $\cos\!\left(\dfrac{\pi}{2} + \alpha\right)$（$k = 1$，奇数）→ 变，$\cos$ 变为 $\sin$；$\dfrac{\pi}{2} + \alpha$ 在第二象限，第二象限 $\cos < 0$，所以 $\cos\!\left(\dfrac{\pi}{2}+\alpha\right) = -\sin\alpha$。

这就是"奇变偶不变，符号看象限"的完整用法。

### 4.4 所有诱导公式的统一记忆表

| 变换 | 是否"变"函数名 | $\sin$ 的结果 | $\cos$ 的结果 |
|------|------|------|------|
| $-\alpha$ | 偶（否） | $-\sin\alpha$ | $\cos\alpha$ |
| $\pi - \alpha$ | 偶（否） | $\sin\alpha$ | $-\cos\alpha$ |
| $\pi + \alpha$ | 偶（否） | $-\sin\alpha$ | $-\cos\alpha$ |
| $2\pi - \alpha$ | 偶（否） | $-\sin\alpha$ | $\cos\alpha$ |
| $\dfrac{\pi}{2} - \alpha$ | 奇（是） | $\cos\alpha$ | $\sin\alpha$ |
| $\dfrac{\pi}{2} + \alpha$ | 奇（是） | $\cos\alpha$ | $-\sin\alpha$ |

只需背住口诀，配合"象限符号"就能推导出所有结果，不需要死记每一行。

---

## 五、思考路标（条件反射训练）

以下每条都要反复内化，直到触发条件就知道下一步动作：

1. **见"已知 $\sin\alpha$ 求 $\cos\alpha$"** → 立刻想到 $\sin^2\alpha + \cos^2\alpha = 1$，先算 $\cos^2\alpha$，再用**象限符号**定正负。切勿直接写 $\cos\alpha = \pm\sqrt{1 - \sin^2\alpha}$ 不定符号。

2. **见"求 $\tan\alpha$"** → 优先用 $\tan\alpha = \dfrac{\sin\alpha}{\cos\alpha}$，需要先知道 $\sin\alpha$ 和 $\cos\alpha$，或者其中一个加上另一个的符号。

3. **见"$\sin^2\alpha + \cos^2\alpha$"** → 直接等于 $1$，任何情况下（无论 $\alpha$ 取何值）都成立，这是最强的化简工具。

4. **见"诱导公式"类题目** → 先数 $\dfrac{k\pi}{2}$ 的 $k$ 值，奇数变函数名，偶数不变；然后把 $\alpha$ 当锐角，判断 $\dfrac{k\pi}{2} \pm \alpha$ 在哪个象限，确定符号。不要死记公式，用口诀现推。

5. **化简含多个不同角的三角式** → 尝试用诱导公式把所有角化为同一个 $\alpha$（或其某个倍角/一半）的三角函数，统一变量。

6. **"$\sin(\pi - \alpha) = \sin\alpha$"的记忆根源** → $\pi - \alpha$ 与 $\alpha$ 关于 $y$ 轴对称（终边关于 $y$ 轴对称时，$y$ 坐标相同），所以 $\sin$ 值相同，$\cos$ 值变号。

7. **$\alpha$ 是钝角时** → $\sin\alpha > 0$（第二象限），$\cos\alpha < 0$，$\tan\alpha < 0$。这是高考中"已知 $\alpha$ 是钝角"类题目的符号起点。

8. **验证答案** → 计算完 $\sin\alpha, \cos\alpha, \tan\alpha$ 后，检查：① $\sin^2\alpha + \cos^2\alpha \overset{?}{=} 1$（勾股恒等式）；② $\dfrac{\sin\alpha}{\cos\alpha} \overset{?}{=} \tan\alpha$（商式恒等式）；③ 各符号与所给象限一致。三个检验都过才算做对。

9. **"化简三角式"的一般策略** → 优先把 $\tan$ 换成 $\dfrac{\sin}{\cos}$，统一为 $\sin, \cos$；然后寻找 $\sin^2 + \cos^2 = 1$ 的结构做代换；最后尝试因式分解或通分。

10. **周期折叠** → 在用诱导公式之前，先用周期性把角折叠到 $[0, 2\pi)$（或 $(-\pi, \pi]$）内，再用诱导公式化为第一象限角。顺序：周期折叠 → 诱导公式 → 查特殊值表。

---

## 六、方法变形与拓展

### 6.1 同角恒等式的各种变形应用

**已知 $\sin\alpha + \cos\alpha = k$，求 $\sin\alpha\cos\alpha$：**

$$(\sin\alpha + \cos\alpha)^2 = \sin^2\alpha + 2\sin\alpha\cos\alpha + \cos^2\alpha = 1 + 2\sin\alpha\cos\alpha$$

所以 $\sin\alpha\cos\alpha = \dfrac{k^2 - 1}{2}$。

**已知 $\tan\alpha = t$，用 $\sin\alpha, \cos\alpha$ 表示：**

由 $\tan\alpha = \dfrac{\sin\alpha}{\cos\alpha} = t$ 和 $\sin^2\alpha + \cos^2\alpha = 1$，解方程组：

$$\cos^2\alpha = \frac{1}{1 + \tan^2\alpha} = \frac{1}{1 + t^2}, \quad \sin^2\alpha = \frac{t^2}{1 + t^2}$$

符号由象限决定。

### 6.2 诱导公式的化简练习路径

**化简步骤标准化**：

1. 用周期性把角化到 $[0, 2\pi)$
2. 观察是 $\dfrac{k\pi}{2} + \alpha$ 还是 $\dfrac{k\pi}{2} - \alpha$ 的形式
3. 判断 $k$ 奇偶，确定是否变函数名
4. 判断整体角所在象限，确定符号
5. 写出化简结果

### 6.3 同角恒等式与诱导公式联合使用

很多题目需要两者结合：先用诱导公式把所有角化为同一个 $\alpha$，再用同角恒等式化简。

**例题（综合型）**：化简 $\dfrac{\sin(\pi+\alpha)\cos(2\pi-\alpha)}{\tan(-\alpha-\pi)\sin(\pi-\alpha)}$。

**解答**：

- $\sin(\pi+\alpha) = -\sin\alpha$（第二组诱导，偶，第三象限 $\sin < 0$）
- $\cos(2\pi-\alpha) = \cos(-\alpha) = \cos\alpha$（周期性 + 余弦偶函数）
- $\tan(-\alpha-\pi) = \tan(-\alpha) = -\tan\alpha$（第一组，正切奇函数）
- $\sin(\pi-\alpha) = \sin\alpha$（第三组诱导，偶，第二象限 $\sin > 0$）

代入：

$$\frac{(-\sin\alpha)(\cos\alpha)}{(-\tan\alpha)(\sin\alpha)} = \frac{-\sin\alpha\cos\alpha}{-\dfrac{\sin\alpha}{\cos\alpha} \cdot \sin\alpha} = \frac{-\sin\alpha\cos\alpha}{-\dfrac{\sin^2\alpha}{\cos\alpha}} = \frac{\sin\alpha\cos\alpha \cdot \cos\alpha}{\sin^2\alpha} = \frac{\cos^2\alpha}{\sin\alpha} = \frac{\cos^2\alpha}{\sin\alpha}$$

最终结果：$\dfrac{\cos^2\alpha}{\sin\alpha}$（不能进一步化简，除非还有附加条件）。

---

## 七、应用例题

### 例 1：基础——由一个三角函数值求其余两个

**题目**：已知 $\cos\theta = -\dfrac{\sqrt{5}}{5}$，且 $\theta$ 在第二象限，求 $\sin\theta$ 和 $\tan\theta$。

**【解答】**

**求 $\sin\theta$**：

$$\sin^2\theta = 1 - \cos^2\theta = 1 - \frac{5}{25} = 1 - \frac{1}{5} = \frac{4}{5}$$

$\theta$ 在第二象限，$\sin\theta > 0$，故

$$\sin\theta = \frac{2}{\sqrt{5}} = \frac{2\sqrt{5}}{5}$$

**求 $\tan\theta$**：

$$\tan\theta = \frac{\sin\theta}{\cos\theta} = \frac{2\sqrt{5}/5}{-\sqrt{5}/5} = \frac{2\sqrt{5}}{5} \times \left(-\frac{5}{\sqrt{5}}\right) = -2$$

**验证**：$\sin^2\theta + \cos^2\theta = \dfrac{4}{5} + \dfrac{1}{5} = 1$ ✓；$\tan\theta = \dfrac{\sin\theta}{\cos\theta} = \dfrac{2\sqrt{5}/5}{-\sqrt{5}/5} = -2$ ✓。

$$\boxed{\sin\theta = \frac{2\sqrt{5}}{5},\quad \tan\theta = -2}$$

---

### 例 2：诱导公式化简

**题目**：化简 $\sin\!\left(\dfrac{3\pi}{2} - \alpha\right) + \cos\!\left(\pi + \alpha\right)$。

**【解答】**

**化简第一项** $\sin\!\left(\dfrac{3\pi}{2} - \alpha\right)$：

$\dfrac{3\pi}{2} = \dfrac{3 \times \pi}{2}$，$k = 3$（奇数），函数名**变**（$\sin \to \cos$）；$\dfrac{3\pi}{2} - \alpha$ 在第三象限（$\alpha$ 为锐角时），$\sin < 0$，所以……

等等，函数名已经变成 $\cos$ 了，这里的符号判断看的是变换后的 $\cos\alpha$ 在第三象限的情况。$\cos < 0$，故 $\sin\!\left(\dfrac{3\pi}{2} - \alpha\right) = -\cos\alpha$。

**化简第二项** $\cos(\pi + \alpha)$：

$k = 2$（偶数），不变（仍为 $\cos$）；$\pi + \alpha$ 在第三象限，$\cos < 0$，故 $\cos(\pi + \alpha) = -\cos\alpha$。

合并：

$$\sin\!\left(\frac{3\pi}{2} - \alpha\right) + \cos(\pi + \alpha) = -\cos\alpha + (-\cos\alpha) = -2\cos\alpha$$

$$\boxed{-2\cos\alpha}$$

---

### 例 3：综合——已知条件涉及多个三角函数

**题目**：已知 $\sin\alpha + \cos\alpha = \dfrac{\sqrt{2}}{2}$，求 $\tan\alpha + \dfrac{1}{\tan\alpha}$ 的值。

**【解答】**

**步骤 1**：由 $\sin\alpha + \cos\alpha = \dfrac{\sqrt{2}}{2}$ 两边平方：

$$\sin^2\alpha + 2\sin\alpha\cos\alpha + \cos^2\alpha = \frac{1}{2}$$

$$1 + 2\sin\alpha\cos\alpha = \frac{1}{2}$$

$$\sin\alpha\cos\alpha = -\frac{1}{4}$$

**步骤 2**：化简目标式。

$$\tan\alpha + \frac{1}{\tan\alpha} = \frac{\sin\alpha}{\cos\alpha} + \frac{\cos\alpha}{\sin\alpha} = \frac{\sin^2\alpha + \cos^2\alpha}{\sin\alpha\cos\alpha} = \frac{1}{\sin\alpha\cos\alpha}$$

**步骤 3**：代入。

$$\tan\alpha + \frac{1}{\tan\alpha} = \frac{1}{-1/4} = -4$$

$$\boxed{-4}$$

---

## 八、思路自测题

**自测 1**　已知 $\sin\alpha = \dfrac{5}{13}$，且 $\alpha$ 在第一象限，求 $\cos\alpha$ 和 $\tan\alpha$。

> 💡 提示：$\cos^2\alpha = 1 - \dfrac{25}{169} = \dfrac{144}{169}$，第一象限 $\cos > 0$，故 $\cos\alpha = \dfrac{12}{13}$。$\tan\alpha = \dfrac{5/13}{12/13} = \dfrac{5}{12}$。

**自测 2**　化简 $\sin\!\left(\dfrac{\pi}{2} + \alpha\right) - \sin(\pi - \alpha)$。

> 💡 提示：$\sin\!\left(\dfrac{\pi}{2} + \alpha\right) = \cos\alpha$（奇变偶不变，$k=1$ 奇，变为 $\cos$，第二象限 $\sin > 0$，即 $\cos\alpha$）；$\sin(\pi - \alpha) = \sin\alpha$（$k=2$ 偶，不变，第二象限 $\sin > 0$）。结果：$\cos\alpha - \sin\alpha$。

**自测 3**　已知 $\tan\alpha = 2$，求 $\dfrac{\sin\alpha - 2\cos\alpha}{\sin\alpha + \cos\alpha}$ 的值。

> 💡 提示：分子分母同除以 $\cos\alpha$（$\cos\alpha \ne 0$）：$\dfrac{\tan\alpha - 2}{\tan\alpha + 1} = \dfrac{2 - 2}{2 + 1} = 0$。

**自测 4**　证明恒等式：$(\sin\alpha + \cos\alpha)^2 + (\sin\alpha - \cos\alpha)^2 = 2$。

> 💡 提示：左边 $= (\sin^2\alpha + 2\sin\alpha\cos\alpha + \cos^2\alpha) + (\sin^2\alpha - 2\sin\alpha\cos\alpha + \cos^2\alpha) = 1 + 2\sin\alpha\cos\alpha + 1 - 2\sin\alpha\cos\alpha = 2$。

**自测 5**　化简 $\dfrac{\cos(2\pi - \alpha)\sin(\pi + \alpha)}{\sin(-\alpha - \pi)\cos(\pi/2 + \alpha)}$，并判断其符号（不依赖具体 $\alpha$ 值）。

> 💡 提示：分子：$\cos(2\pi-\alpha) = \cos\alpha$，$\sin(\pi+\alpha) = -\sin\alpha$，乘积 $= -\sin\alpha\cos\alpha$。分母：$\sin(-\alpha-\pi) = \sin(-(\alpha+\pi)) = -\sin(\alpha+\pi) = -(-\sin\alpha) = \sin\alpha$，$\cos(\pi/2 + \alpha) = -\sin\alpha$，乘积 $= -\sin^2\alpha$。整式 $= \dfrac{-\sin\alpha\cos\alpha}{-\sin^2\alpha} = \dfrac{\cos\alpha}{\sin\alpha} = \dfrac{1}{\tan\alpha}$（即 $\cot\alpha$）。

---

**回头看一眼"一例速记"**：

> 三大同角恒等式：$\sin^2\alpha + \cos^2\alpha = 1$（勾股型），$\tan\alpha = \sin\alpha/\cos\alpha$（商型），$1 + \tan^2\alpha = \sec^2\alpha$（平方和型）。诱导口诀：奇变偶不变，符号看象限。已知一个三角函数值+象限，就能求出其余两个：先绝对值（勾股），再定符号（象限），再商式（求正切）。

如果你现在能不看笔记，完整做完引入题（$\sin\alpha = \dfrac{3}{5}$，第二象限，求 $\cos\alpha$ 和 $\tan\alpha$），包括验证步骤——本章，你拿下了。
