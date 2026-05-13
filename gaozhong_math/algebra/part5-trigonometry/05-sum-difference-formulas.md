# 三角函数的和差倍半角公式

> **一例速记**：  
> 和角公式（必背）：  
> $\sin(\alpha \pm \beta) = \sin\alpha\cos\beta \pm \cos\alpha\sin\beta$  
> $\cos(\alpha \pm \beta) = \cos\alpha\cos\beta \mp \sin\alpha\sin\beta$  
> $\tan(\alpha \pm \beta) = \dfrac{\tan\alpha \pm \tan\beta}{1 \mp \tan\alpha\tan\beta}$  
> **倍角公式**：$\sin 2\alpha = 2\sin\alpha\cos\alpha$，$\cos 2\alpha = \cos^2\alpha - \sin^2\alpha = 2\cos^2\alpha - 1 = 1 - 2\sin^2\alpha$  
> **半角公式**：$\sin^2\dfrac{\alpha}{2} = \dfrac{1-\cos\alpha}{2}$，$\cos^2\dfrac{\alpha}{2} = \dfrac{1+\cos\alpha}{2}$，$\tan\dfrac{\alpha}{2} = \dfrac{\sin\alpha}{1+\cos\alpha} = \dfrac{1-\cos\alpha}{\sin\alpha}$

---

## 一、引入：计算 $\sin 75°$

> **题目**：化简计算 $\sin 75°$，结果用根式表示。

请先停下来，自己想想：$75°$ 不是标准的 $30°, 45°, 60°, 90°$，单角三角值表里没有。怎么办？

直接查计算器得出近似值，不是解题思路。考试要精确值。这道题的突破口在于：**$75° = 45° + 30°$**——两个我们熟悉的角相加！这就用得上和角公式了。

---

## 二、思维路径还原（解题者的内心独白）

> "看到 $\sin 75°$，脑子里第一反应：$75°$ 不是标准角，无法直接查表。
>
> **第一步：拆角。** 想 $75°$ 能不能分解成两个标准角的和或差？$75 = 45 + 30$，这两个都是标准角，$\sin 45° = \dfrac{\sqrt{2}}{2}$，$\cos 45° = \dfrac{\sqrt{2}}{2}$，$\sin 30° = \dfrac{1}{2}$，$\cos 30° = \dfrac{\sqrt{3}}{2}$——很好，全是已知量。
>
> **第二步：套公式。** 用和角公式 $\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$，令 $\alpha = 45°$，$\beta = 30°$：
>
> $$\sin 75° = \sin(45° + 30°) = \sin 45°\cos 30° + \cos 45°\sin 30°$$
>
> **第三步：代值。** 把标准角三角值代入：
>
> $$= \frac{\sqrt{2}}{2} \cdot \frac{\sqrt{3}}{2} + \frac{\sqrt{2}}{2} \cdot \frac{1}{2}$$
>
> $$= \frac{\sqrt{6}}{4} + \frac{\sqrt{2}}{4} = \frac{\sqrt{6} + \sqrt{2}}{4}$$
>
> **验证直觉。** $\sin 60° = \dfrac{\sqrt{3}}{2} \approx 0.866$，$\sin 90° = 1$，$\sin 75°$ 应在这两者之间。$\dfrac{\sqrt{6}+\sqrt{2}}{4} \approx \dfrac{2.449 + 1.414}{4} \approx \dfrac{3.863}{4} \approx 0.966$，而实际 $\sin 75° \approx 0.966$ ✓。
>
> **关键反射：** 遇到非标准角，第一步永远是"拆成两个标准角的和或差"。拆的方式不唯一：$75° = 45° + 30° = 135° - 60°$，两种拆法都可行（后者用差角公式，结果相同）。
>
> **延伸思考：** 同理可算 $\cos 75°$：
>
> $$\cos 75° = \cos(45° + 30°) = \cos 45°\cos 30° - \sin 45°\sin 30° = \frac{\sqrt{6}}{4} - \frac{\sqrt{2}}{4} = \frac{\sqrt{6} - \sqrt{2}}{4}$$
>
> 验证：$\sin^2 75° + \cos^2 75° = \left(\dfrac{\sqrt{6}+\sqrt{2}}{4}\right)^2 + \left(\dfrac{\sqrt{6}-\sqrt{2}}{4}\right)^2 = \dfrac{6+2\sqrt{12}+2 + 6-2\sqrt{12}+2}{16} = \dfrac{16}{16} = 1$ ✓。
>
> **方法迁移：** 如果要算 $\sin 15°$，可用 $15° = 45° - 30°$，套差角公式得 $\sin 15° = \dfrac{\sqrt{6}-\sqrt{2}}{4}$——正好等于 $\cos 75°$，符合余角关系 $\sin 15° = \cos 75°$ ✓。"

把这段内心独白读两遍，感受"拆角 $\to$ 套公式 $\to$ 代值 $\to$ 验证"的节奏。

---

## 三、公式全览与记忆策略

### 3.1 和差角公式

$$\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$$

$$\sin(\alpha - \beta) = \sin\alpha\cos\beta - \cos\alpha\sin\beta$$

$$\cos(\alpha + \beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$$

$$\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta$$

$$\tan(\alpha + \beta) = \frac{\tan\alpha + \tan\beta}{1 - \tan\alpha\tan\beta}$$

$$\tan(\alpha - \beta) = \frac{\tan\alpha - \tan\beta}{1 + \tan\alpha\tan\beta}$$

**记忆口诀**（正弦和差）：  
- $\sin$ 和角：**正正加正正**（正弦正加正弦的反）→ $\sin\alpha\cos\beta + \cos\alpha\sin\beta$  
- $\sin$ 差角：把"$+$"改成"$-$"  
- $\cos$ 和角：**余余减正正**（余弦余弦减正弦正弦）→ $\cos\alpha\cos\beta - \sin\alpha\sin\beta$  
- $\cos$ 差角：把"$-$"改成"$+$"  

**正弦与余弦的符号规律对比**：$\sin(\alpha\pm\beta)$ 中 $\pm$ 与 $\pm$ 同向；$\cos(\alpha\pm\beta)$ 中 $\pm$ 与 $\mp$ 相反（交叉）。

### 3.2 倍角公式

$$\sin 2\alpha = 2\sin\alpha\cos\alpha$$

$$\cos 2\alpha = \cos^2\alpha - \sin^2\alpha = 2\cos^2\alpha - 1 = 1 - 2\sin^2\alpha$$

$$\tan 2\alpha = \frac{2\tan\alpha}{1 - \tan^2\alpha}$$

$\cos 2\alpha$ 有三种等价形式，灵活选用：
- 见到 $\cos^2\alpha - \sin^2\alpha$（两个平方之差）→ 凑成 $\cos 2\alpha$
- 见到 $2\cos^2\alpha - 1$ 或 $1 - 2\sin^2\alpha$（单一平方）→ 凑成 $\cos 2\alpha$

### 3.3 半角公式

由 $\cos 2\alpha = 1 - 2\sin^2\alpha$ 令 $2\alpha \to \alpha$（即 $\alpha \to \dfrac{\alpha}{2}$）：

$$\sin^2\frac{\alpha}{2} = \frac{1 - \cos\alpha}{2}, \qquad \cos^2\frac{\alpha}{2} = \frac{1 + \cos\alpha}{2}$$

开根号后（正负号由 $\dfrac{\alpha}{2}$ 所在象限决定）：

$$\sin\frac{\alpha}{2} = \pm\sqrt{\frac{1-\cos\alpha}{2}}, \qquad \cos\frac{\alpha}{2} = \pm\sqrt{\frac{1+\cos\alpha}{2}}$$

$$\tan\frac{\alpha}{2} = \pm\sqrt{\frac{1-\cos\alpha}{1+\cos\alpha}} = \frac{\sin\alpha}{1+\cos\alpha} = \frac{1-\cos\alpha}{\sin\alpha}$$

后两种 $\tan\dfrac{\alpha}{2}$ 的表达式（万能公式前身）不需要讨论正负号，更常用。

### 3.4 积化和差 / 和差化积（选记）

**积化和差**（把乘积化为和差，便于积分或化简）：

$$\sin\alpha\cos\beta = \frac{1}{2}[\sin(\alpha+\beta) + \sin(\alpha-\beta)]$$

$$\cos\alpha\sin\beta = \frac{1}{2}[\sin(\alpha+\beta) - \sin(\alpha-\beta)]$$

$$\cos\alpha\cos\beta = \frac{1}{2}[\cos(\alpha+\beta) + \cos(\alpha-\beta)]$$

$$\sin\alpha\sin\beta = -\frac{1}{2}[\cos(\alpha+\beta) - \cos(\alpha-\beta)]$$

**和差化积**（把和差化为乘积，便于因式分解或方程求解）：

$$\sin\alpha + \sin\beta = 2\sin\frac{\alpha+\beta}{2}\cos\frac{\alpha-\beta}{2}$$

$$\sin\alpha - \sin\beta = 2\cos\frac{\alpha+\beta}{2}\sin\frac{\alpha-\beta}{2}$$

$$\cos\alpha + \cos\beta = 2\cos\frac{\alpha+\beta}{2}\cos\frac{\alpha-\beta}{2}$$

$$\cos\alpha - \cos\beta = -2\sin\frac{\alpha+\beta}{2}\sin\frac{\alpha-\beta}{2}$$

---

## 四、方法拓展：5 类常见技巧

### 4.1 非标准角拆分（万能技巧）

**场景**：遇到 $15°, 75°, 105°, 22.5°$ 等非标准角。  
**策略**：拆成两个标准角（$0°, 30°, 45°, 60°, 90°$）的和或差。

| 目标角 | 拆法 | 公式 |
|--------|------|------|
| $\sin 15°$ | $45° - 30°$ | 差角公式 |
| $\sin 75°$ | $45° + 30°$ | 和角公式 |
| $\cos 105°$ | $60° + 45°$ 或 $150° - 45°$ | 和角/差角公式 |
| $\tan 75°$ | $45° + 30°$ | 正切和角公式 |

### 4.2 辅助角公式（$a\sin x + b\cos x$ 的统一形式）

这是高考中的高频考点。

$$a\sin x + b\cos x = \sqrt{a^2 + b^2}\,\sin(x + \varphi)$$

其中 $\varphi$ 满足：$\cos\varphi = \dfrac{a}{\sqrt{a^2+b^2}}$，$\sin\varphi = \dfrac{b}{\sqrt{a^2+b^2}}$（即 $\tan\varphi = \dfrac{b}{a}$）。

**推导过程**（理解记忆）：

$$a\sin x + b\cos x = \sqrt{a^2+b^2}\left(\frac{a}{\sqrt{a^2+b^2}}\sin x + \frac{b}{\sqrt{a^2+b^2}}\cos x\right)$$

令 $\cos\varphi = \dfrac{a}{\sqrt{a^2+b^2}}$，$\sin\varphi = \dfrac{b}{\sqrt{a^2+b^2}}$，则括号内恰好是 $\sin x\cos\varphi + \cos x\sin\varphi = \sin(x+\varphi)$。

**典型应用**：
- 求 $f(x) = \sin x + \sqrt{3}\cos x$ 的最大值和最小值  
  → $f(x) = 2\sin\left(x + \dfrac{\pi}{3}\right)$，最大值 $2$，最小值 $-2$  
- 求最小正周期、单调区间，步骤完全相同

**注意**：$\varphi$ 的确定要结合 $\sin\varphi$ 和 $\cos\varphi$ 的符号（即判断 $\varphi$ 在哪个象限），不能只用 $\arctan\dfrac{b}{a}$。

### 4.3 降幂公式（半角公式的逆用）

**场景**：见到 $\sin^2\alpha$、$\cos^2\alpha$，想"降次"（把平方降为一次）：

$$\sin^2\alpha = \frac{1 - \cos 2\alpha}{2}, \qquad \cos^2\alpha = \frac{1 + \cos 2\alpha}{2}$$

**典型场景**：
- 化简 $\sin^2\alpha + \cos^4\alpha$（先降次再合并）
- 高次三角式的积分（先用降幂再处理）

### 4.4 升幂与凑角（凑成倍角/半角）

**场景**：见到 $\sin\alpha\cos\alpha$（乘积形式），想到倍角公式逆用：

$$\sin\alpha\cos\alpha = \frac{1}{2}\sin 2\alpha$$

**场景**：见到 $1 + \cos\alpha$ 或 $1 - \cos\alpha$，想到半角：

$$1 - \cos\alpha = 2\sin^2\frac{\alpha}{2}, \qquad 1 + \cos\alpha = 2\cos^2\frac{\alpha}{2}$$

这是"凑半角"的标准套路，常用于化简含 $1 \pm \cos\alpha$ 的表达式。

### 4.5 和差化积的因式分解应用

**场景**：三角方程或三角不等式中出现 $\sin\alpha \pm \sin\beta$ 的形式，用和差化积因式分解。

**典型**：解方程 $\sin 3x + \sin x = 0$：

$$\sin 3x + \sin x = 2\sin\frac{3x+x}{2}\cos\frac{3x-x}{2} = 2\sin 2x\cos x = 0$$

→ $\sin 2x = 0$ 或 $\cos x = 0$，分别求解。

---

## 五、思考路标（条件反射训练）

下面每条都要反复内化，遇到对应场景立刻触发：

1. **见到非标准角**（$15°, 75°, 22.5°$ 等）→ 立即想"能否拆成两个标准角的和或差？"$45° \pm 30°$、$60° \pm 45°$ 是最常用的拆法。

2. **见到 $\sin(\alpha + \beta)$ 展开** → 正弦和角：**两正交叉相加**（$\sin\cos + \cos\sin$）；正弦差角：**交叉相减**。永远是"正弦乘余弦 $\pm$ 余弦乘正弦"的形式。

3. **见到 $\cos(\alpha + \beta)$ 展开** → 余弦和角：**同名相乘再相减**（$\cos\cos - \sin\sin$）；余弦差角：**同名相乘再相加**（$\cos\cos + \sin\sin$）。注意：和角用减，差角用加——与正弦相反！

4. **见到 $\sin\alpha\cos\alpha$（乘积）** → 立即联想倍角 $\sin 2\alpha = 2\sin\alpha\cos\alpha$，等价于 $\sin\alpha\cos\alpha = \dfrac{1}{2}\sin 2\alpha$。

5. **见到 $\cos^2\alpha$ 或 $\sin^2\alpha$（平方）** → 想"降幂"：$\cos^2\alpha = \dfrac{1+\cos 2\alpha}{2}$，$\sin^2\alpha = \dfrac{1-\cos 2\alpha}{2}$。先降次，再化简。

6. **见到 $a\sin x + b\cos x$ 的线性组合** → 辅助角公式，提因子 $\sqrt{a^2+b^2}$，凑成 $\sin(x+\varphi)$ 的形式，先确定 $\varphi$ 的象限（用 $\sin\varphi$ 和 $\cos\varphi$ 的符号），再写 $\arctan$ 值。

7. **见到 $1 \pm \cos\alpha$** → 立即联想半角：$1-\cos\alpha = 2\sin^2\dfrac{\alpha}{2}$，$1+\cos\alpha = 2\cos^2\dfrac{\alpha}{2}$。特别是在分母或根号下出现 $1 \pm \cos\alpha$ 时。

8. **见到 $\cos 2\alpha$ 化简** → 根据其他项含 $\sin^2$ 还是 $\cos^2$ 来选三种形式：若已知 $\sin\alpha$ 用 $1-2\sin^2\alpha$；若已知 $\cos\alpha$ 用 $2\cos^2\alpha-1$；若两者均有用 $\cos^2\alpha-\sin^2\alpha$。

9. **见到 $\sin A + \sin B$ 的和** → 想和差化积：$2\sin\dfrac{A+B}{2}\cos\dfrac{A-B}{2}$，常用于三角方程的因式分解。

10. **倍角公式的"反推"** → 看到 $\dfrac{\tan 2\alpha}{1+\tan^2\alpha}$ 这类形式时，想到 $\sin 2\alpha = \dfrac{2\tan\alpha}{1+\tan^2\alpha}$（万能公式），统一用 $\tan\alpha$ 表示所有三角函数。

---

## 六、应用例题

### 例 1：已知条件求值

**题目**：已知 $\sin\alpha = \dfrac{1}{3}$，$\cos\beta = -\dfrac{\sqrt{5}}{5}$，$\alpha \in \left(0, \dfrac{\pi}{2}\right)$，$\beta \in \left(\dfrac{\pi}{2}, \pi\right)$，求 $\sin(\alpha + \beta)$。

**【解答】**

**第一步：求其余三角值。**

由 $\sin\alpha = \dfrac{1}{3}$，$\alpha \in \left(0, \dfrac{\pi}{2}\right)$（第一象限，余弦为正）：

$$\cos\alpha = \sqrt{1 - \sin^2\alpha} = \sqrt{1 - \frac{1}{9}} = \sqrt{\frac{8}{9}} = \frac{2\sqrt{2}}{3}$$

由 $\cos\beta = -\dfrac{\sqrt{5}}{5} = -\dfrac{1}{\sqrt{5}}$，$\beta \in \left(\dfrac{\pi}{2}, \pi\right)$（第二象限，正弦为正）：

$$\sin\beta = \sqrt{1 - \cos^2\beta} = \sqrt{1 - \frac{1}{5}} = \sqrt{\frac{4}{5}} = \frac{2}{\sqrt{5}} = \frac{2\sqrt{5}}{5}$$

**第二步：套和角公式。**

$$\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$$

$$= \frac{1}{3} \cdot \left(-\frac{\sqrt{5}}{5}\right) + \frac{2\sqrt{2}}{3} \cdot \frac{2\sqrt{5}}{5}$$

$$= -\frac{\sqrt{5}}{15} + \frac{4\sqrt{10}}{15} = \frac{4\sqrt{10} - \sqrt{5}}{15}$$

$$\boxed{\sin(\alpha+\beta) = \frac{4\sqrt{10}-\sqrt{5}}{15}}$$

> 解题要点：先用同角恒等式和象限信息把 $\sin\alpha, \cos\alpha, \sin\beta, \cos\beta$ 全部求出，再代入和角公式。象限信息决定正负号，绝对不能忽视。

---

### 例 2：倍角化简

**题目**：化简 $\dfrac{1 - \cos 2\alpha}{\sin 2\alpha}$。

**【解答】**

利用降幂公式和倍角公式：

$$\frac{1 - \cos 2\alpha}{\sin 2\alpha} = \frac{2\sin^2\alpha}{2\sin\alpha\cos\alpha} = \frac{\sin\alpha}{\cos\alpha} = \tan\alpha$$

$$\boxed{\frac{1-\cos 2\alpha}{\sin 2\alpha} = \tan\alpha}$$

> 解题要点：分子 $1-\cos 2\alpha$ 用降幂公式 $= 2\sin^2\alpha$；分母 $\sin 2\alpha$ 用倍角公式 $= 2\sin\alpha\cos\alpha$；约分得 $\tan\alpha$。"一升一降，约分见 $\tan$" 是经典套路。

---

### 例 3：辅助角公式求最值

**题目**：求 $f(x) = \sqrt{3}\sin x + \cos x$ 的最大值、最小值，并求最大值处的 $x$ 值（$x \in [0, 2\pi]$）。

**【解答】**

**第一步：辅助角变形。**

$$f(x) = \sqrt{3}\sin x + \cos x = 2\left(\frac{\sqrt{3}}{2}\sin x + \frac{1}{2}\cos x\right)$$

注意 $\dfrac{\sqrt{3}}{2} = \cos\dfrac{\pi}{6}$，$\dfrac{1}{2} = \sin\dfrac{\pi}{6}$，所以：

$$f(x) = 2\left(\sin x\cos\frac{\pi}{6} + \cos x\sin\frac{\pi}{6}\right) = 2\sin\left(x + \frac{\pi}{6}\right)$$

**第二步：求最值。**

$\sin\left(x + \dfrac{\pi}{6}\right) \in [-1, 1]$，故：
- 最大值为 $2$，取到时 $x + \dfrac{\pi}{6} = \dfrac{\pi}{2}$，即 $x = \dfrac{\pi}{3}$。
- 最小值为 $-2$，取到时 $x + \dfrac{\pi}{6} = \dfrac{3\pi}{2}$，即 $x = \dfrac{4\pi}{3}$。

$$\boxed{f(x)_{\max} = 2\;(x = \tfrac{\pi}{3}),\quad f(x)_{\min} = -2\;(x = \tfrac{4\pi}{3})}$$

> 解题要点：提公因子 $2 = \sqrt{(\sqrt{3})^2 + 1^2}$；辨认括号内 $\cos\dfrac{\pi}{6}\sin x + \sin\dfrac{\pi}{6}\cos x$ 的结构，对应 $\sin\!\left(x+\dfrac{\pi}{6}\right)$。

---

### 例 4：半角公式化简

**题目**：化简 $\sqrt{\dfrac{1 - \cos\theta}{2}}$（$\theta \in (0, \pi)$）。

**【解答】**

由半角公式 $\sin^2\dfrac{\theta}{2} = \dfrac{1-\cos\theta}{2}$，所以：

$$\sqrt{\frac{1-\cos\theta}{2}} = \sqrt{\sin^2\frac{\theta}{2}} = \left|\sin\frac{\theta}{2}\right|$$

因 $\theta \in (0, \pi)$，所以 $\dfrac{\theta}{2} \in \left(0, \dfrac{\pi}{2}\right)$，在第一象限，$\sin\dfrac{\theta}{2} > 0$，故绝对值符号可去：

$$= \sin\frac{\theta}{2}$$

$$\boxed{\sqrt{\frac{1-\cos\theta}{2}} = \sin\frac{\theta}{2}}$$

> 解题要点：半角公式开根号后要讨论正负号，必须结合 $\dfrac{\theta}{2}$ 所在象限来判断。很多学生漏掉正负号的讨论，失分。

---

### 例 5：综合化简——多步变换

**题目**：化简 $\sin^4\alpha + \cos^4\alpha$。

**【解答】**

**方法一（降幂）**：

$$\sin^4\alpha + \cos^4\alpha = (\sin^2\alpha)^2 + (\cos^2\alpha)^2$$

$$= (\sin^2\alpha + \cos^2\alpha)^2 - 2\sin^2\alpha\cos^2\alpha$$

$$= 1 - 2\sin^2\alpha\cos^2\alpha = 1 - \frac{1}{2}\sin^2 2\alpha$$

再用 $\sin^2 2\alpha = \dfrac{1-\cos 4\alpha}{2}$：

$$= 1 - \frac{1}{2} \cdot \frac{1-\cos 4\alpha}{2} = 1 - \frac{1-\cos 4\alpha}{4} = \frac{3 + \cos 4\alpha}{4}$$

$$\boxed{\sin^4\alpha + \cos^4\alpha = \frac{3+\cos 4\alpha}{4}}$$

> 解题要点：用恒等式 $a^2+b^2 = (a+b)^2 - 2ab$ 把四次降为二次，再用倍角和降幂处理 $\sin^2\alpha\cos^2\alpha$。

---

## 七、思路自测题

**自测 1**　不查表，用和差角公式求 $\cos 105°$ 的精确值。

> 💡 提示：$105° = 60° + 45°$，用余弦和角公式 $\cos(\alpha+\beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$，代入 $\cos 60° = \dfrac{1}{2}$，$\sin 60° = \dfrac{\sqrt{3}}{2}$，$\cos 45° = \sin 45° = \dfrac{\sqrt{2}}{2}$，得 $\cos 105° = \dfrac{\sqrt{2}}{4} - \dfrac{\sqrt{6}}{4} = \dfrac{\sqrt{2}-\sqrt{6}}{4}$。

**自测 2**　化简 $\dfrac{\sin 2\alpha}{1 + \cos 2\alpha}$。

> 💡 提示：分子 $\sin 2\alpha = 2\sin\alpha\cos\alpha$，分母 $1 + \cos 2\alpha = 2\cos^2\alpha$，约分得 $\dfrac{2\sin\alpha\cos\alpha}{2\cos^2\alpha} = \tan\alpha$。

**自测 3**　已知 $\cos\theta = -\dfrac{3}{5}$，$\theta \in (\pi, 2\pi)$，求 $\sin\dfrac{\theta}{2}$ 的值。

> 💡 提示：$\theta \in (\pi, 2\pi)$ 故 $\dfrac{\theta}{2} \in \left(\dfrac{\pi}{2}, \pi\right)$（第二象限），$\sin\dfrac{\theta}{2} > 0$。用 $\sin^2\dfrac{\theta}{2} = \dfrac{1-\cos\theta}{2} = \dfrac{1-(-3/5)}{2} = \dfrac{4}{5}$，故 $\sin\dfrac{\theta}{2} = \dfrac{2}{\sqrt{5}} = \dfrac{2\sqrt{5}}{5}$。

**自测 4**　将 $f(x) = \sin x - \cos x$ 化为辅助角形式，并求最大值和对应的 $x$（$x \in [0, 2\pi]$）。

> 💡 提示：$\sin x - \cos x = \sqrt{2}\left(\dfrac{1}{\sqrt{2}}\sin x - \dfrac{1}{\sqrt{2}}\cos x\right) = \sqrt{2}\sin\!\left(x - \dfrac{\pi}{4}\right)$。最大值 $\sqrt{2}$，当 $x - \dfrac{\pi}{4} = \dfrac{\pi}{2}$，即 $x = \dfrac{3\pi}{4}$ 时取到。

**自测 5**　证明：$\cos 3\alpha = 4\cos^3\alpha - 3\cos\alpha$。

> 💡 提示：$\cos 3\alpha = \cos(2\alpha+\alpha) = \cos 2\alpha\cos\alpha - \sin 2\alpha\sin\alpha$。代入 $\cos 2\alpha = 2\cos^2\alpha-1$，$\sin 2\alpha = 2\sin\alpha\cos\alpha$：$= (2\cos^2\alpha-1)\cos\alpha - 2\sin\alpha\cos\alpha\cdot\sin\alpha = 2\cos^3\alpha - \cos\alpha - 2\sin^2\alpha\cos\alpha$。再用 $\sin^2\alpha = 1-\cos^2\alpha$：$= 2\cos^3\alpha - \cos\alpha - 2(1-\cos^2\alpha)\cos\alpha = 4\cos^3\alpha - 3\cos\alpha$。

---

**回头看一眼"一例速记"**：

> 和角：$\sin(\alpha+\beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$；$\cos(\alpha+\beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$。  
> 倍角：$\sin 2\alpha = 2\sin\alpha\cos\alpha$；$\cos 2\alpha = 2\cos^2\alpha - 1 = 1 - 2\sin^2\alpha$。  
> 辅助角：$a\sin x + b\cos x = \sqrt{a^2+b^2}\sin(x+\varphi)$。

如果现在你能不看笔记，独立推导 $\sin(45°+30°)$ 并算出 $\sin 75°$ 的精确值——本章，你拿下了。
