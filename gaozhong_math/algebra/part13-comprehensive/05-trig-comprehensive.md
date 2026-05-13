# 三角函数综合应用

> **一例速记**：  
> 三角综合套路：先**化简**（用诱导 / 恒等 / 倍角 / 辅助角）→ 再用**函数性质**（单调 / 周期 / 最值）。  
> **辅助角公式**：$a\sin x + b\cos x = \sqrt{a^2+b^2}\sin(x+\varphi)$，其中 $\tan\varphi = \dfrac{b}{a}$（或 $\cos\varphi = \dfrac{a}{\sqrt{a^2+b^2}}$，$\sin\varphi = \dfrac{b}{\sqrt{a^2+b^2}}$）。

---

## 一、引入：求 $f(x) = \sin^2 x + 2\sin x \cos x + 3\cos^2 x$ 的最大值与单调递增区间

> **题目**：已知 $f(x) = \sin^2 x + 2\sin x \cos x + 3\cos^2 x$，求：
> (1) $f$ 的最大值；
> (2) $f$ 的单调递增区间。

请先停下来想一想：这个 $f(x)$ 是 $\sin^2$、$\sin\cos$、$\cos^2$ 的线性组合，每一项的次数都是 $2$。直接对 $f$ 求导能做，但烦琐。

**最聪明的开局**：观察到所有项都是**二次**——回想 $\sin^2$、$\cos^2$、$\sin\cos$ 的"降幂公式"：

$$\sin^2 x = \frac{1 - \cos 2x}{2}, \quad \cos^2 x = \frac{1 + \cos 2x}{2}, \quad 2\sin x \cos x = \sin 2x$$

降幂后 $f$ 变成"$\sin 2x$、$\cos 2x$、常数"的线性组合——一次形式，再用辅助角合并为单个三角函数，最值和单调区间立刻可见。

这是三角综合题的**第一招**：**降幂 → 合并 → 辅助角**。下面把内心独白完整还原。

---

## 二、思维路径还原（解题者的内心独白）

> "看到 $f(x) = \sin^2 x + 2\sin x \cos x + 3\cos^2 x$，第一反应：**全是二次**，立刻**降幂**！
>
> **第一步：降幂。**
>
> $$\sin^2 x = \frac{1 - \cos 2x}{2}, \quad \cos^2 x = \frac{1 + \cos 2x}{2}, \quad 2\sin x \cos x = \sin 2x$$
>
> 代入：
>
> $$f(x) = \frac{1 - \cos 2x}{2} + \sin 2x + 3 \cdot \frac{1 + \cos 2x}{2}$$
>
> **第二步：合并同类项。**
>
> $$f(x) = \frac{1}{2} - \frac{\cos 2x}{2} + \sin 2x + \frac{3}{2} + \frac{3\cos 2x}{2}$$
>
> $$= \left(\frac{1}{2} + \frac{3}{2}\right) + \left(-\frac{1}{2} + \frac{3}{2}\right)\cos 2x + \sin 2x$$
>
> $$= 2 + \cos 2x + \sin 2x$$
>
> 漂亮，$f$ 化成了 $2 + \sin 2x + \cos 2x$，已经很简单了。
>
> **第三步：辅助角公式合并 $\sin 2x + \cos 2x$。**
>
> $$\sin 2x + \cos 2x = \sqrt{1^2 + 1^2}\sin\left(2x + \frac{\pi}{4}\right) = \sqrt{2}\sin\left(2x + \frac{\pi}{4}\right)$$
>
> （这里 $a = 1, b = 1$，$\tan\varphi = b/a = 1$，$\varphi = \pi/4$。）
>
> 故
>
> $$\boxed{f(x) = 2 + \sqrt{2}\sin\left(2x + \frac{\pi}{4}\right)}$$
>
> **第四步：读最值与单调区间。**
>
> (1) **最大值**：$\sin\left(2x + \dfrac{\pi}{4}\right) \in [-1, 1]$，最大值为 $1$，故 $f_{\max} = 2 + \sqrt{2}$，取得时需 $2x + \dfrac{\pi}{4} = \dfrac{\pi}{2} + 2k\pi$，即 $x = \dfrac{\pi}{8} + k\pi$（$k \in \mathbb{Z}$）。
>
> (2) **单调递增区间**：$\sin$ 在 $\left[-\dfrac{\pi}{2} + 2k\pi, \dfrac{\pi}{2} + 2k\pi\right]$ 上单调增。设 $u = 2x + \dfrac{\pi}{4}$，要 $u \in \left[-\dfrac{\pi}{2} + 2k\pi, \dfrac{\pi}{2} + 2k\pi\right]$，即
>
> $$-\frac{\pi}{2} + 2k\pi \leq 2x + \frac{\pi}{4} \leq \frac{\pi}{2} + 2k\pi$$
>
> 解出 $x$：
>
> $$-\frac{3\pi}{8} + k\pi \leq x \leq \frac{\pi}{8} + k\pi$$
>
> 故单调递增区间为 $\left[-\dfrac{3\pi}{8} + k\pi, \dfrac{\pi}{8} + k\pi\right]$（$k \in \mathbb{Z}$）。
>
> **关键反射**：降幂把 '二次' 转为 '一次'，辅助角把 '$a\sin + b\cos$' 转为 '单个 $\sin$'。三角综合的所有题都是这两招的反复组合。"

把这段独白读两遍，记住"降幂 → 合并 → 辅助角"的标准流程。这是高考三角综合最高频的开局。

---

## 三、抽象成方法：四步法

把三角综合题标准化为**四步法**：

### 3.1 第一步：化简为单一三角函数

化简手段：
1. **降幂**（出现 $\sin^2, \cos^2, \sin\cos$）：用 $\sin^2 = \dfrac{1 - \cos 2x}{2}$、$\cos^2 = \dfrac{1 + \cos 2x}{2}$、$2\sin\cos = \sin 2x$。这一步把"二次"压成"一次"，是化简的核心动作。
2. **倍角**（出现 $\sin 2x, \cos 2x$）：根据需要展开为 $\sin x, \cos x$，或反向合并。注意 $\cos 2x$ 有三种形式（$\cos^2 - \sin^2$、$1 - 2\sin^2$、$2\cos^2 - 1$），选择能与其他项合并的那种。
3. **诱导公式**（出现 $\sin(\pi - x), \cos(\pi/2 + x)$ 等）：化为 $\sin x, \cos x$ 标准形式。口诀"奇变偶不变，符号看象限"——把 $\dfrac{\pi}{2} \pm x$、$\pi \pm x$、$\dfrac{3\pi}{2} \pm x$ 都化掉。
4. **恒等变换**（用 $\sin^2 + \cos^2 = 1$ 替换、配方等）。这是"万能补漏"的工具。
5. **辅助角**（出现 $a\sin x + b\cos x$）：化为 $\sqrt{a^2+b^2}\sin(x + \varphi)$ 或 $\sqrt{a^2+b^2}\cos(x - \psi)$。这是化"两项"为"一项"的关键。

目标：把 $f(x)$ 化成 $A\sin(\omega x + \varphi) + B$ 形式（或 $A\cos(\omega x + \varphi) + B$）。一旦达到此形，所有性质都"显式可读"。

### 3.2 第二步：识别性质

化为 $A\sin(\omega x + \varphi) + B$ 后：
- **振幅**：$|A|$
- **周期**：$T = \dfrac{2\pi}{|\omega|}$
- **相位**：$\varphi$
- **平衡线**：$y = B$
- **最值**：$f_{\max} = |A| + B$，$f_{\min} = -|A| + B$

### 3.3 第三步：解决具体问题

| 问题类型 | 套路 |
|---------|------|
| 最值 | 用 $\sin \in [-1, 1]$，乘 $A$ 加 $B$ |
| 单调区间 | 套 $\sin$ 单调区间公式，令 $u = \omega x + \varphi$ 解 $x$ |
| 周期 | $T = 2\pi / |\omega|$ |
| 对称轴 | $\omega x + \varphi = \dfrac{\pi}{2} + k\pi$，解 $x$ |
| 对称中心 | $\omega x + \varphi = k\pi$，解 $x$（横坐标），纵坐标 $= B$ |
| 零点 | $A\sin(\omega x + \varphi) + B = 0$，即 $\sin = -B/A$ |

### 3.4 第四步：综合（与方程 / 不等式 / 解三角形结合）

三角综合压轴常与**解三角形**（正余弦定理 + 面积公式）、**向量**（高考偶尔出现）、**不等式**（基本不等式求最值）综合。综合时先各自独立分析，再用条件串起来。

| 综合类型 | 关键工具 | 典型问法 |
|---------|---------|---------|
| 三角 + 函数性质 | 降幂、辅助角、单调性公式 | 求 $f$ 最值 / 单调区间 / 对称轴 |
| 三角 + 解三角形 | 正余弦定理、面积公式、投影定理 | 求边长 / 角 / 面积 |
| 三角 + 不等式 | 基本不等式、$\|\sin\| \leq 1$、换元 | 求范围 / 最值 |
| 三角 + 向量 | $\vec{a} \cdot \vec{b}$、模长 | 求夹角 / 模 / 投影 |

---

## 四、方法变形：技巧汇总

### 4.1 辅助角公式的两种形式

$$a\sin x + b\cos x = R\sin(x + \varphi)$$

其中 $R = \sqrt{a^2 + b^2}$，$\cos\varphi = \dfrac{a}{R}$，$\sin\varphi = \dfrac{b}{R}$（即 $\tan\varphi = b/a$）。

**或写成余弦形式**：

$$a\sin x + b\cos x = R\cos(x - \psi)$$

其中 $\cos\psi = \dfrac{b}{R}$，$\sin\psi = \dfrac{a}{R}$。两种形式互通。

**常用特例**（必背）：
- $\sin x + \cos x = \sqrt{2}\sin\left(x + \dfrac{\pi}{4}\right)$
- $\sin x - \cos x = \sqrt{2}\sin\left(x - \dfrac{\pi}{4}\right)$
- $\sqrt{3}\sin x + \cos x = 2\sin\left(x + \dfrac{\pi}{6}\right)$
- $\sin x + \sqrt{3}\cos x = 2\sin\left(x + \dfrac{\pi}{3}\right)$
- $\sqrt{3}\sin x - \cos x = 2\sin\left(x - \dfrac{\pi}{6}\right)$
- $\sin x - \sqrt{3}\cos x = 2\sin\left(x - \dfrac{\pi}{3}\right)$

记住这六个，覆盖 90% 的辅助角题。**推导口诀**：振幅 $R = \sqrt{a^2 + b^2}$，相位 $\varphi$ 由 $\sin\varphi : \cos\varphi = b : a$ 决定（按系数对应）。

### 4.2 万能公式（高级技巧）

设 $t = \tan\dfrac{x}{2}$，则：

$$\sin x = \frac{2t}{1+t^2}, \quad \cos x = \frac{1-t^2}{1+t^2}, \quad \tan x = \frac{2t}{1-t^2}$$

把三角问题变成**有理函数**问题。适用于含多种三角函数混合且无法用其他化简的题。

### 4.3 解三角形：正余弦定理 + 面积

**正弦定理**：$\dfrac{a}{\sin A} = \dfrac{b}{\sin B} = \dfrac{c}{\sin C} = 2R$（$R$ 为外接圆半径）。

**余弦定理**：$c^2 = a^2 + b^2 - 2ab\cos C$（及其循环式）。

**面积公式**：$S = \dfrac{1}{2}ab\sin C = \dfrac{abc}{4R} = \sqrt{p(p-a)(p-b)(p-c)}$（海伦公式，$p$ 半周长）。

**选用原则**：
- 已知**两角一边** → 正弦定理求其他边。
- 已知**两边一夹角** → 余弦定理求第三边。
- 已知**三边** → 余弦定理求角（或用海伦求面积）。
- 已知**两边一对角** → 正弦定理（注意可能多解）。

### 4.4 三角与不等式

求 $\sin x + 2\cos x$ 的最值：辅助角 $\sqrt{5}\sin(x + \varphi)$，最值 $\pm\sqrt{5}$。这是最基础的"辅助角求最值"。

更进阶：求 $\dfrac{\sin x + 1}{\cos x + 2}$ 的最值——**线性分式**型，可设 $y = \dfrac{\sin x + 1}{\cos x + 2}$，化为 $y \cos x - \sin x = 1 - 2y$，左侧用辅助角 $\sqrt{y^2 + 1}\sin(\cdots)$，由 $\left|\sin\right| \leq 1$ 得 $\left|1 - 2y\right| \leq \sqrt{y^2 + 1}$，平方化为 $y$ 的二次不等式即可。

### 4.5 三角与向量

向量 $\vec{a} = (\cos\alpha, \sin\alpha)$、$\vec{b} = (\cos\beta, \sin\beta)$ 是单位向量。

- $\vec{a} \cdot \vec{b} = \cos\alpha\cos\beta + \sin\alpha\sin\beta = \cos(\alpha - \beta)$（**两角差余弦公式**）。
- $|\vec{a} - \vec{b}|^2 = 2 - 2\cos(\alpha - \beta)$。

向量化思考能简化推导。

### 4.6 三角恒等式速查

- 两角和差：$\sin(A \pm B) = \sin A\cos B \pm \cos A\sin B$，$\cos(A \pm B) = \cos A\cos B \mp \sin A\sin B$。
- 倍角：$\sin 2x = 2\sin x\cos x$，$\cos 2x = \cos^2 x - \sin^2 x = 1 - 2\sin^2 x = 2\cos^2 x - 1$。
- 半角：$\sin\dfrac{x}{2} = \pm\sqrt{\dfrac{1 - \cos x}{2}}$，$\cos\dfrac{x}{2} = \pm\sqrt{\dfrac{1 + \cos x}{2}}$。
- 积化和差：$\sin A\cos B = \dfrac{1}{2}[\sin(A+B) + \sin(A-B)]$ 等。
- 和差化积：$\sin A + \sin B = 2\sin\dfrac{A+B}{2}\cos\dfrac{A-B}{2}$ 等。

### 4.7 三角形里的"两个隐藏条件"

凡是 $\triangle ABC$，永远默认：
- $A + B + C = \pi$（三内角和为 $\pi$）。
- $a, b, c > 0$（边长为正），$A, B, C \in (0, \pi)$。

由 $A + B + C = \pi$ 派生：
- $\sin(A + B) = \sin C$；$\cos(A + B) = -\cos C$；$\tan(A + B) = -\tan C$。
- $\sin\dfrac{A + B}{2} = \cos\dfrac{C}{2}$；$\cos\dfrac{A + B}{2} = \sin\dfrac{C}{2}$。

这些"隐含公式"出现在解三角形的化简中频率极高。看到 $\sin C$ 或 $\cos C$ 时，立刻想"能否用 $A + B$ 替代"。

### 4.8 形如 $a\sin^2 x + b\sin x\cos x + c\cos^2 x$ 的通用降幂

最通用的处理：分别降幂。

$$a\sin^2 x = a \cdot \frac{1 - \cos 2x}{2} = \frac{a}{2} - \frac{a}{2}\cos 2x$$

$$c\cos^2 x = c \cdot \frac{1 + \cos 2x}{2} = \frac{c}{2} + \frac{c}{2}\cos 2x$$

$$b\sin x\cos x = \frac{b}{2}\sin 2x$$

合并：

$$a\sin^2 x + b\sin x\cos x + c\cos^2 x = \frac{a + c}{2} + \frac{c - a}{2}\cos 2x + \frac{b}{2}\sin 2x$$

这是引入题的"模板形式"。任何"$\sin, \cos$ 二次齐次组合"都可这样化为"$2x$ 的一次组合 + 常数"。

---

## 五、思考路标（条件反射训练）

下面每条都要反复内化，遇到对应场景立刻触发：

1. **看到 $\sin^2 x$、$\cos^2 x$、$\sin\cos$** → **降幂**！化二次为一次。

2. **看到 $a\sin x + b\cos x$** → **辅助角**！化为单个 $\sin$ 或 $\cos$。

3. **综合最值** → 化为 $A\sin(\omega x + \varphi) + B$ 后，用 $\sin \in [-1, 1]$ 直接读最值。

4. **周期** → $T = \dfrac{2\pi}{|\omega|}$。注意 $|\sin x|$ 周期是 $\pi$，不是 $2\pi$。

5. **单调区间** → 套 $\sin$ 在 $\left[-\dfrac{\pi}{2} + 2k\pi, \dfrac{\pi}{2} + 2k\pi\right]$ 递增的公式，令 $u = \omega x + \varphi$ 解出 $x$。

6. **解三角形先选定理** → 已知信息是"边边边" / "边边角" / "边角边" / "角角边"？正弦 vs 余弦定理 vs 面积公式，按形选择。

7. **三角与不等式** → 用基本不等式（$ab \leq \dfrac{a^2 + b^2}{2}$）或转化为辅助角后用 $|\sin| \leq 1$。

8. **诱导公式记口诀** → "奇变偶不变，符号看象限"。$\sin\left(\dfrac{\pi}{2} + x\right) = \cos x$（奇数倍 $\pi/2$ 改名）。

9. **倍角与半角是互逆的** → 看到 $\cos 2x$ 既可降幂（变 $\cos^2 x$），也可不变（保留 $2x$）——根据题目需要选择方向。

10. **解三角形里 $A + B + C = \pi$ 永远好用** → 这是隐藏条件，常用 $\sin(A+B) = \sin C$、$\cos(A+B) = -\cos C$。

11. **三角形中的"$a > b \iff A > B$"** → 这是正弦定理的推论（大边对大角），用于判断三角形形状或排除多解。

12. **遇到 $\sin x \cdot \cos x$** → 想 $\sin 2x / 2$，把乘积变 $\sin 2x$ 减少元。

13. **遇到 $\sin x + \cos x$ 与 $\sin x\cos x$ 同时出现** → 换元 $t = \sin x + \cos x$，则 $\sin x\cos x = \dfrac{t^2 - 1}{2}$，变成关于 $t$ 的多项式。$t \in [-\sqrt{2}, \sqrt{2}]$。

14. **看到题目要求"求 $\omega$"** → 通常由周期 $T = \dfrac{2\pi}{|\omega|}$ 或对称轴 / 对称中心给定一个方程，解出 $\omega$。注意 $\omega$ 的正负条件。

15. **化简时永远先看"能不能凑出 $\sin 2x$ / $\cos 2x$"** → $2x$ 形式的三角函数好操作，是化简的"目标形态"。

16. **遇到 $\tan x$** → 通常化为 $\dfrac{\sin x}{\cos x}$ 处理，注意 $\cos x \ne 0$ 的条件。

17. **解三角形里"判断三角形形状"** → 化简成 $\sin / \cos$ 等式后，看是否得到 $A = B$（等腰）、$A + B = \dfrac{\pi}{2}$（直角，$C = \pi/2$）等。

---

## 六、应用例题

### 例 1：化简与最值（含降幂与辅助角）

**题目**：已知函数 $f(x) = 2\sin^2\left(\dfrac{\pi}{4} + x\right) - \sqrt{3}\cos 2x$。求 $f$ 的最大值，并求 $f$ 取得最大值时 $x$ 的集合。

**【解答】**

**第一步：化简。**

用降幂公式处理 $\sin^2\left(\dfrac{\pi}{4} + x\right)$：

$$2\sin^2\left(\frac{\pi}{4} + x\right) = 1 - \cos\left(\frac{\pi}{2} + 2x\right) = 1 - (-\sin 2x) = 1 + \sin 2x$$

代入：

$$f(x) = (1 + \sin 2x) - \sqrt{3}\cos 2x = 1 + \sin 2x - \sqrt{3}\cos 2x$$

**第二步：辅助角合并。**

$\sin 2x - \sqrt{3}\cos 2x = 2\sin\left(2x - \dfrac{\pi}{3}\right)$（$a = 1, b = -\sqrt{3}$，$R = \sqrt{1 + 3} = 2$，$\tan\varphi = b/a = -\sqrt{3}$，结合 $\cos\varphi = 1/2 > 0$、$\sin\varphi = -\sqrt{3}/2 < 0$，$\varphi$ 在第四象限，取 $\varphi = -\dfrac{\pi}{3}$）。

故

$$f(x) = 1 + 2\sin\left(2x - \frac{\pi}{3}\right)$$

**第三步：读最值。**

$\sin\left(2x - \dfrac{\pi}{3}\right) \in [-1, 1]$，故 $f \in [-1, 3]$，**最大值 $f_{\max} = 3$**。

取得时需 $\sin\left(2x - \dfrac{\pi}{3}\right) = 1$，即 $2x - \dfrac{\pi}{3} = \dfrac{\pi}{2} + 2k\pi$，解出

$$x = \frac{5\pi}{12} + k\pi\quad (k \in \mathbb{Z})$$

**结论**：$f$ 的最大值为 $3$，取得时 $x \in \left\{\dfrac{5\pi}{12} + k\pi \,\middle|\, k \in \mathbb{Z}\right\}$。

> 解题要点：① $2\sin^2(\cdot) = 1 - \cos(2 \cdot)$ 是降幂的"巧用形式"。② 注意 $\cos\left(\dfrac{\pi}{2} + 2x\right) = -\sin 2x$（诱导公式）。③ 辅助角中 $b < 0$ 时 $\varphi$ 为负，结果是减号形式。

---

### 例 2：单调区间与对称轴

**题目**：已知 $f(x) = \sqrt{3}\sin\left(\omega x + \dfrac{\pi}{6}\right) + \cos\left(\omega x + \dfrac{\pi}{6}\right)$（$\omega > 0$）的最小正周期为 $\pi$。

(1) 求 $\omega$；
(2) 求 $f$ 在区间 $\left[-\dfrac{\pi}{4}, \dfrac{\pi}{4}\right]$ 上的单调递增区间；
(3) 求 $f$ 的对称轴方程。

**【解答】**

**第一步：化简 $f(x)$。**

$\sqrt{3}\sin\theta + \cos\theta = 2\sin\left(\theta + \dfrac{\pi}{6}\right)$（辅助角，$a = \sqrt{3}, b = 1$，$R = 2$，$\tan\varphi = \dfrac{1}{\sqrt{3}}$，$\varphi = \dfrac{\pi}{6}$）。

设 $\theta = \omega x + \dfrac{\pi}{6}$，得

$$f(x) = 2\sin\left(\omega x + \frac{\pi}{6} + \frac{\pi}{6}\right) = 2\sin\left(\omega x + \frac{\pi}{3}\right)$$

**第二步：由周期求 $\omega$。**

$T = \dfrac{2\pi}{\omega} = \pi \implies \omega = 2$。故 $f(x) = 2\sin\left(2x + \dfrac{\pi}{3}\right)$。

**第三步：求单调递增区间。**

$\sin$ 在 $\left[-\dfrac{\pi}{2} + 2k\pi, \dfrac{\pi}{2} + 2k\pi\right]$ 上单调增。设 $u = 2x + \dfrac{\pi}{3}$：

$$-\frac{\pi}{2} + 2k\pi \leq 2x + \frac{\pi}{3} \leq \frac{\pi}{2} + 2k\pi$$

$$-\frac{5\pi}{12} + k\pi \leq x \leq \frac{\pi}{12} + k\pi$$

故 $f$ 的单调递增区间为 $\left[-\dfrac{5\pi}{12} + k\pi, \dfrac{\pi}{12} + k\pi\right]$（$k \in \mathbb{Z}$）。

限制在 $\left[-\dfrac{\pi}{4}, \dfrac{\pi}{4}\right]$ 内：取 $k = 0$ 得 $\left[-\dfrac{5\pi}{12}, \dfrac{\pi}{12}\right]$，与 $\left[-\dfrac{\pi}{4}, \dfrac{\pi}{4}\right]$ 的交集为 $\left[-\dfrac{\pi}{4}, \dfrac{\pi}{12}\right]$。

**第四步：对称轴。**

$\sin$ 的对称轴为 $u = \dfrac{\pi}{2} + k\pi$，故 $2x + \dfrac{\pi}{3} = \dfrac{\pi}{2} + k\pi$，解出

$$x = \frac{\pi}{12} + \frac{k\pi}{2}\quad (k \in \mathbb{Z})$$

> 解题要点：① 化简时注意"嵌套相位"——$\omega x + \dfrac{\pi}{6}$ 的"$\dfrac{\pi}{6}$"加上辅助角的"$\dfrac{\pi}{6}$"，得到 $\dfrac{\pi}{3}$。② 单调递增区间要与给定区间求交。③ 对称轴公式 $u = \dfrac{\pi}{2} + k\pi$ 与 $\sin$ 的最大值 / 最小值点重合。

---

### 例 3：解三角形（正余弦综合）

**题目**：在 $\triangle ABC$ 中，内角 $A, B, C$ 的对边分别为 $a, b, c$，已知 $a = 2$，$b\cos A - a\cos B = c$，$\cos C = \dfrac{1}{3}$。求：

(1) $c$ 的值；
(2) $\triangle ABC$ 的面积。

**【解答】**

**第一步：处理条件 $b\cos A - a\cos B = c$。**

由**投影定理**（或称射影定理）：$c = a\cos B + b\cos A$。所以条件 $b\cos A - a\cos B = c = a\cos B + b\cos A$，化简：

$$b\cos A - a\cos B = a\cos B + b\cos A$$

$$-a\cos B = a\cos B \implies a\cos B = 0$$

由 $a = 2 \ne 0$，得 $\cos B = 0$，即 $B = \dfrac{\pi}{2}$（直角三角形！）。

**第二步：求 $c$。**

$B = \dfrac{\pi}{2}$，$\cos C = \dfrac{1}{3}$，$\sin C = \sqrt{1 - 1/9} = \dfrac{2\sqrt{2}}{3}$（$C$ 是锐角，$\sin > 0$）。

由 $A + B + C = \pi$，$A = \pi - \dfrac{\pi}{2} - C = \dfrac{\pi}{2} - C$。$\sin A = \sin\left(\dfrac{\pi}{2} - C\right) = \cos C = \dfrac{1}{3}$。

由正弦定理 $\dfrac{a}{\sin A} = \dfrac{c}{\sin C}$：

$$\frac{2}{1/3} = \frac{c}{2\sqrt{2}/3} \implies 6 = \frac{3c}{2\sqrt{2}} \implies c = 4\sqrt{2}$$

**第三步：求面积。**

由 $B = \dfrac{\pi}{2}$，$b$ 是斜边。$\sin B = 1$，正弦定理：$\dfrac{b}{1} = 6$，$b = 6$。

面积 $S = \dfrac{1}{2}ac\sin B = \dfrac{1}{2} \cdot 2 \cdot 4\sqrt{2} \cdot 1 = 4\sqrt{2}$。

或用 $S = \dfrac{1}{2}ab \cdot \dfrac{c}{b}$……简单点直接 $S = \dfrac{1}{2}|AB| \cdot |BC| = \dfrac{1}{2} \cdot a \cdot c = \dfrac{1}{2} \cdot 2 \cdot 4\sqrt{2} = 4\sqrt{2}$（直角三角形面积，两条直角边相乘除 2）。

**结论**：$c = 4\sqrt{2}$，$S_{\triangle ABC} = 4\sqrt{2}$。

> 解题要点：① 投影定理（射影定理）是解三角形里常用的隐藏工具：$c = a\cos B + b\cos A$、$a = b\cos C + c\cos B$、$b = c\cos A + a\cos C$。② 条件 $b\cos A - a\cos B = c$ 与投影定理结合，立刻得 $B = \dfrac{\pi}{2}$。③ 直角三角形面积可直接用 $\dfrac{1}{2}$ 两直角边相乘。

---

### 例 4：综合最值（含约束）

**题目**：在 $\triangle ABC$ 中，$a, b, c$ 分别是角 $A, B, C$ 的对边，已知 $A = \dfrac{\pi}{3}$，$b + c = 4$。求 $a$ 的最小值。

**【解答】**

由余弦定理：

$$a^2 = b^2 + c^2 - 2bc\cos A = b^2 + c^2 - 2bc \cdot \frac{1}{2} = b^2 + c^2 - bc$$

利用 $b + c = 4$，配方：

$$b^2 + c^2 = (b + c)^2 - 2bc = 16 - 2bc$$

故

$$a^2 = 16 - 2bc - bc = 16 - 3bc$$

要 $a^2$ 最小，需 $bc$ 最大。由基本不等式 $bc \leq \left(\dfrac{b + c}{2}\right)^2 = 4$，等号当且仅当 $b = c = 2$。

故 $a^2_{\min} = 16 - 12 = 4$，**$a_{\min} = 2$**（当 $b = c = 2$，三角形为等边三角形）。

> 解题要点：① 余弦定理结合"和定积有最大"的基本不等式。② 等号取得时三角形退化为等边（因 $A = \dfrac{\pi}{3}$ 且 $b = c$）。③ 这是"约束下的最值"的典范——用代数变形把目标量与已知量挂钩。

---

## 七、思路自测题

**自测 1**　化简 $f(x) = \cos^2 x + \sqrt{3}\sin x\cos x + 1$，并求最小正周期。

> 💡 提示：$\cos^2 x = \dfrac{1 + \cos 2x}{2}$，$\sqrt{3}\sin x\cos x = \dfrac{\sqrt{3}}{2}\sin 2x$。$f(x) = \dfrac{1 + \cos 2x}{2} + \dfrac{\sqrt{3}}{2}\sin 2x + 1 = \dfrac{3}{2} + \dfrac{1}{2}\cos 2x + \dfrac{\sqrt{3}}{2}\sin 2x = \dfrac{3}{2} + \sin\left(2x + \dfrac{\pi}{6}\right)$。最小正周期 $T = \dfrac{2\pi}{2} = \pi$。

**自测 2**　求 $f(x) = \sin x + \cos x + \sin x\cos x$ 的最大值。

> 💡 提示：令 $t = \sin x + \cos x = \sqrt{2}\sin\left(x + \dfrac{\pi}{4}\right) \in [-\sqrt{2}, \sqrt{2}]$。由 $t^2 = 1 + 2\sin x\cos x$，得 $\sin x\cos x = \dfrac{t^2 - 1}{2}$。$f = t + \dfrac{t^2 - 1}{2} = \dfrac{1}{2}(t + 1)^2 - 1$。$t \in [-\sqrt{2}, \sqrt{2}]$ 时 $(t+1)^2$ 最大值在 $t = \sqrt{2}$ 处，$(t+1)^2 = (\sqrt{2}+1)^2 = 3 + 2\sqrt{2}$，$f_{\max} = \dfrac{3 + 2\sqrt{2}}{2} - 1 = \dfrac{1 + 2\sqrt{2}}{2} = \dfrac{1}{2} + \sqrt{2}$。

**自测 3**　在 $\triangle ABC$ 中，$a = 7$，$b = 8$，$\cos B = -\dfrac{1}{7}$，求 $A$ 与 $\triangle ABC$ 的面积。

> 💡 提示：$\sin B = \sqrt{1 - 1/49} = \dfrac{4\sqrt{3}}{7}$（$B$ 为钝角，$\sin > 0$）。由正弦定理 $\dfrac{a}{\sin A} = \dfrac{b}{\sin B}$：$\sin A = \dfrac{a\sin B}{b} = \dfrac{7 \cdot 4\sqrt{3}/7}{8} = \dfrac{\sqrt{3}}{2}$。$A = \dfrac{\pi}{3}$ 或 $\dfrac{2\pi}{3}$；因 $B$ 钝角，$A$ 必锐角，故 $A = \dfrac{\pi}{3}$。$C = \pi - A - B$，$\sin C = \sin(A + B) = \sin A\cos B + \cos A\sin B = \dfrac{\sqrt{3}}{2} \cdot \left(-\dfrac{1}{7}\right) + \dfrac{1}{2} \cdot \dfrac{4\sqrt{3}}{7} = \dfrac{-\sqrt{3} + 4\sqrt{3}}{14} = \dfrac{3\sqrt{3}}{14}$。面积 $S = \dfrac{1}{2}ab\sin C = \dfrac{1}{2} \cdot 7 \cdot 8 \cdot \dfrac{3\sqrt{3}}{14} = 6\sqrt{3}$。

**自测 4**　求函数 $f(x) = \dfrac{\sin x}{2 + \cos x}$ 的值域。

> 💡 提示：令 $y = \dfrac{\sin x}{2 + \cos x}$，化为 $\sin x - y\cos x = 2y$。左侧用辅助角：$\sqrt{1 + y^2}\sin(x - \varphi) = 2y$（$\tan\varphi = y$）。由 $|\sin| \leq 1$，$|2y| \leq \sqrt{1 + y^2}$，平方：$4y^2 \leq 1 + y^2 \implies 3y^2 \leq 1 \implies |y| \leq \dfrac{1}{\sqrt{3}} = \dfrac{\sqrt{3}}{3}$。值域 $\left[-\dfrac{\sqrt{3}}{3}, \dfrac{\sqrt{3}}{3}\right]$。

**自测 5**　在 $\triangle ABC$ 中，$2\cos C(a\cos B + b\cos A) = c$。求角 $C$。

> 💡 提示：由投影定理 $a\cos B + b\cos A = c$，代入：$2\cos C \cdot c = c$，得 $\cos C = \dfrac{1}{2}$，$C = \dfrac{\pi}{3}$。这是"投影定理 + 简单代数"的典范。

---

**回头看一眼"一例速记"**：

> 化简（降幂 / 倍角 / 诱导 / 辅助角）→ 单一 $\sin$ → 读性质（周期 / 单调 / 最值 / 对称）。  
> 辅助角 $a\sin x + b\cos x = \sqrt{a^2+b^2}\sin(x + \varphi)$。  
> 解三角形：正弦定理 / 余弦定理 / 面积公式 / 投影定理，按条件选。

如果现在不看提示，能独立完成例 1、例 2 和自测 3——本章，你拿下了。
