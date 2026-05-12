# 锐角三角函数：正弦、余弦、正切

## 图形特征

直角三角形中，除了直角之外还有两个锐角。当我们盯住其中一个锐角 $\alpha$（$0° < \alpha < 90°$）时，三条边就有了"身份"：

- **斜边** $c$：直角所对的边（最长的那条，固定不变）
- **对边** $a$：锐角 $\alpha$ 所对的边
- **邻边** $b$：锐角 $\alpha$ 所夹的非斜边

> 提示：对边、邻边是相对**所选锐角**而言的——同一个直角三角形，看锐角 $A$ 时和看锐角 $B$ 时，对边与邻边正好互换。

锐角三角函数就是用这三条边的**比值**来度量锐角 $\alpha$ 的"形状"。

---

## 核心概念：三种比值的定义

设直角三角形的锐角 $\alpha$ 的对边为 $a$、邻边为 $b$、斜边为 $c$，定义：

- **正弦** $\sin \alpha = \dfrac{\text{对边}}{\text{斜边}} = \dfrac{a}{c}$
- **余弦** $\cos \alpha = \dfrac{\text{邻边}}{\text{斜边}} = \dfrac{b}{c}$
- **正切** $\tan \alpha = \dfrac{\text{对边}}{\text{邻边}} = \dfrac{a}{b}$

**口诀（强烈推荐）**：

> 正弦 = **对** / 斜；余弦 = **邻** / 斜；正切 = **对** / **邻**。
>
> 也可以记成"**S** 对斜、**C** 邻斜、**T** 对邻"，按字母顺序。

**为什么这样定义是合理的？** —— 因为这三个比值**只与角 $\alpha$ 的大小有关，与三角形大小无关**。

设 $\triangle ABC$ 与 $\triangle A'B'C'$ 都是直角三角形，$\angle C = \angle C' = 90°$，$\angle A = \angle A' = \alpha$。由 AA 相似（part5），两三角形相似，对应边成比例：

$$\frac{a}{a'} = \frac{b}{b'} = \frac{c}{c'} \;\Longrightarrow\; \frac{a}{c} = \frac{a'}{c'},\; \frac{b}{c} = \frac{b'}{c'},\; \frac{a}{b} = \frac{a'}{b'}.$$

所以无论直角三角形画得多大多小，只要锐角 $\alpha$ 相同，$\sin\alpha,\cos\alpha,\tan\alpha$ 都是同一个值。这才让"用比值度量角"的想法立得住。

---

## 性质清单

记 $\alpha$ 为锐角（$0° < \alpha < 90°$）。

### 1. 取值范围

$$0 < \sin\alpha < 1,\quad 0 < \cos\alpha < 1,\quad \tan\alpha > 0.$$

理由：对边、邻边都小于斜边（直角三角形中斜边最长），故 $\sin,\cos$ 都是真分数；$\tan$ 是两正数比，故为正。

### 2. 单调性

在 $0° < \alpha < 90°$ 范围内：

- $\sin\alpha$ **随 $\alpha$ 增大而增大**
- $\tan\alpha$ **随 $\alpha$ 增大而增大**
- $\cos\alpha$ **随 $\alpha$ 增大而减小**

直观理解：把斜边长度固定，让锐角 $\alpha$ 慢慢张大——对边越来越长（$\sin$ 增大）、邻边越来越短（$\cos$ 减小）；$\tan$ 是"对/邻"，分子增分母减，自然增大得更快。

### 3. 平方和恒等式

$$\boxed{\sin^2\alpha + \cos^2\alpha = 1.}$$

**证明**：由勾股 $a^2 + b^2 = c^2$，两边同除以 $c^2$：

$$\frac{a^2}{c^2} + \frac{b^2}{c^2} = 1 \;\Longrightarrow\; \sin^2\alpha + \cos^2\alpha = 1.$$

这是勾股定理在三角函数语言下的"翻译"，是初中最常用的三角恒等式。

### 4. 商关系

$$\boxed{\tan\alpha = \frac{\sin\alpha}{\cos\alpha}.}$$

证明：$\dfrac{\sin\alpha}{\cos\alpha} = \dfrac{a/c}{b/c} = \dfrac{a}{b} = \tan\alpha.$

### 5. 互余关系

若 $\alpha + \beta = 90°$（即 $\beta = 90° - \alpha$），则在同一直角三角形中，$\alpha$ 的对边正好是 $\beta$ 的邻边，反之亦然。于是：

$$\boxed{\sin(90° - \alpha) = \cos\alpha,\quad \cos(90° - \alpha) = \sin\alpha,\quad \tan(90°-\alpha) = \frac{1}{\tan\alpha}.}$$

**口诀**：互余角的正弦 = 余弦、余弦 = 正弦，"sin 与 cos 互换身份"。例如 $\sin 30° = \cos 60°$。

---

## 典型应用

### 例 1：基础——三边求三角函数

> 直角三角形 $ABC$ 中，$\angle C = 90°$，$AC = 3$，$BC = 4$。求 $\sin A,\cos A,\tan A$ 与 $\sin B,\cos B,\tan B$。

**思路**：先由勾股求斜边，再对照"对、邻、斜"代入。

**解**：$AB = \sqrt{AC^2 + BC^2} = \sqrt{9 + 16} = 5$。

对锐角 $A$：对边 $= BC = 4$，邻边 $= AC = 3$，斜边 $= AB = 5$，所以

$$\sin A = \frac{4}{5},\quad \cos A = \frac{3}{5},\quad \tan A = \frac{4}{3}.$$

对锐角 $B$：对边 $= AC = 3$，邻边 $= BC = 4$，斜边 $= AB = 5$，所以

$$\sin B = \frac{3}{5},\quad \cos B = \frac{4}{5},\quad \tan B = \frac{3}{4}.$$

**观察**：$\sin A = \cos B$、$\cos A = \sin B$、$\tan A \cdot \tan B = 1$ —— 正是互余关系（$A + B = 90°$）的验证。

---

### 例 2：已知一个三角函数值，求另外两个

> 锐角 $\alpha$ 满足 $\sin\alpha = \dfrac{3}{5}$。求 $\cos\alpha$ 与 $\tan\alpha$。

**思路 1（用恒等式）**：$\sin^2\alpha + \cos^2\alpha = 1 \Rightarrow \cos^2\alpha = 1 - \dfrac{9}{25} = \dfrac{16}{25}$。因 $\alpha$ 锐角，$\cos\alpha > 0$，故

$$\cos\alpha = \frac{4}{5},\quad \tan\alpha = \frac{\sin\alpha}{\cos\alpha} = \frac{3/5}{4/5} = \frac{3}{4}.$$

**思路 2（构造直角三角形——更直观）**：既然 $\sin\alpha = \dfrac{3}{5} = \dfrac{\text{对}}{\text{斜}}$，不妨设对边 $= 3$，斜边 $= 5$。由勾股，邻边 $= \sqrt{25 - 9} = 4$。读出 $\cos\alpha = \dfrac{4}{5}$，$\tan\alpha = \dfrac{3}{4}$。

> **小结**：已知一个三角函数值求其他时，"构造一个直角三角形 + 勾股"几乎总能避开恒等式的繁琐计算，是中考的首选方法。

---

### 例 3：三角函数与全等——视角统一

> 在 $\triangle ABC$ 与 $\triangle A'B'C'$ 中，$\angle C = \angle C' = 90°$，斜边 $AB = A'B'$，且 $\sin A = \sin A'$。证明 $\triangle ABC \cong \triangle A'B'C'$。

**思路**：$\sin A = \dfrac{BC}{AB}$，$\sin A' = \dfrac{B'C'}{A'B'}$。由 $AB = A'B'$ 与 $\sin A = \sin A'$，立刻得 $BC = B'C'$。再用 HL（斜边 + 一直角边）即得全等。

**这道题的意义**：它揭示了 HL 全等判定的"本质"—— 给定斜边，正弦值唯一决定一条直角边；正弦相等 $\Leftrightarrow$ 直角边相等。三角函数在初中阶段不仅是"工具"，它还把先前学过的相似、全等、勾股串成了一张网。

---

## 易错点

1. **三角函数只在直角三角形中定义**（初中阶段）。非直角三角形不能直接套用 $\sin = \text{对}/\text{斜}$，要先作高把它分成两个直角三角形。
2. **对边、邻边相对所选锐角**——同一个三角形看不同锐角，对边与邻边互换。计算前先确认是对哪一个角。
3. **$\sin\alpha < 1$ 不等于 $\sin\alpha = \frac{1}{\text{某数}}$**。例如 $\sin\alpha = \dfrac{3}{5}$ 不能写成 $\dfrac{1}{5/3}$。
4. **平方和等式中的角必须相同**。$\sin^2 30° + \cos^2 60° \ne 1$（因为角不同），但 $\sin^2 30° + \cos^2 30° = 1$。
5. **$\tan\alpha$ 没有上界**：$\alpha$ 越接近 $90°$，邻边越短，$\tan$ 越大，可以任意大；但 $\sin,\cos$ 始终在 $0$ 到 $1$ 之间。

---

## 自测题

1. 在 $\text{Rt}\triangle ABC$ 中，$\angle C = 90°$，$AC = 5$，$AB = 13$。求 $\sin A,\cos A,\tan A$。

2. 锐角 $\alpha$ 满足 $\cos\alpha = \dfrac{2}{3}$，求 $\sin\alpha$ 与 $\tan\alpha$。

3. 在 $\text{Rt}\triangle ABC$ 中，$\angle C = 90°$，若 $\tan A = \dfrac{3}{4}$，$AB = 10$。求 $AC$ 与 $BC$。

4. 不计算 $\alpha$ 的具体度数，比较 $\sin 20°,\sin 50°,\sin 70°$ 的大小；比较 $\cos 20°,\cos 50°,\cos 70°$ 的大小。

5. 已知 $\alpha$ 为锐角且 $\sin\alpha = \cos\alpha$。求 $\tan\alpha$ 与 $\alpha$ 的可能取值（提示：用商关系）。
