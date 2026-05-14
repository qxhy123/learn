# 积分技巧：LIATE 与换元

> **一例速记**：求 $\int x e^x\, dx$。
> 结构：乘积型，含多项式 $x$（A 类）和指数 $e^x$（E 类）。
> LIATE 口诀：A 优先于 E → 令 $u = x$（A 类先取对数求导），$dv = e^x\, dx$。
> $\int x e^x\, dx = x e^x - \int e^x\, dx = x e^x - e^x + C = e^x(x-1)+C$。
> **选 $u$ 的原则：LIATE 顺序越靠前，越优先作 $u$（求导后更简单）。**

---

## 一、为什么积分比求导难

求导是机械的：给定规则（链式、乘积、商法则），按结构一步步走，结果唯一。

积分没有通用算法：同样一个被积函数，可能需要换元、可能需要分部、可能需要先展开再逐项积分——哪种方法奏效，取决于**识别被积函数的结构类型**。

系统化积分的核心思路：**先判断被积函数的"形状"，再查对应的套路。** 本篇把最常用的两大技术——分部积分（LIATE 口诀）和换元法（5 大类型）——整理成可操作的决策流程。

---

## 二、分部积分与 LIATE

### 2.1 分部积分公式

$$\int u\, dv = uv - \int v\, du.$$

等价形式：$\displaystyle\int u(x)v'(x)\, dx = u(x)v(x) - \int v(x)u'(x)\, dx$。

**目标**：将原来的积分 $\int u\, dv$ 转化为更容易计算的 $\int v\, du$。转化成功的前提是新积分比原积分简单——这取决于如何选 $u$ 和 $dv$。

### 2.2 LIATE 口诀

选 $u$ 的优先级（越靠前越优先选为 $u$）：

| 字母 | 类型 | 典型函数 | 原因 |
|---|---|---|---|
| **L** | 对数型 | $\ln x$，$\log_a x$ | 求导后消去对数，变成代数 |
| **I** | 反三角型 | $\arcsin x$，$\arctan x$ | 求导后变成代数，积分困难 |
| **A** | 代数（多项式）型 | $x^n$，$\sqrt{x}$，$x^{-1}$ | 求导后降次，越来越简单 |
| **T** | 三角型 | $\sin x$，$\cos x$ | 求导后仍为三角，可循环 |
| **E** | 指数型 | $e^x$，$a^x$ | 求导后不变，最"稳定" |

**使用方法**：在乘积型被积函数 $f(x)\cdot g(x)$ 中，哪类字母排在 LIATE 前面，就选那个作 $u$（另一个作 $dv$）。

**直觉**：$u$ 经过求导应变简单，$dv$ 经过积分应不复杂。L 和 I 型积分起来很麻烦（$\int\ln x\, dx$ 难以直接处理），所以优先令它们做 $u$（求导后反而变简单）。E 型积分后不变，所以放在 $dv$ 里不麻烦。

### 2.3 LIATE 的常见配对

| 被积函数类型 | 选 $u$ | 选 $dv$ |
|---|---|---|
| $x^n e^x$ | $u = x^n$（A 优于 E） | $dv = e^x\, dx$ |
| $x^n \sin x$ 或 $x^n\cos x$ | $u = x^n$（A 优于 T） | $dv = \sin x\, dx$ 等 |
| $\ln x \cdot x^n$ | $u = \ln x$（L 优于 A） | $dv = x^n\, dx$ |
| $\arctan x \cdot x^n$ | $u = \arctan x$（I 优于 A） | $dv = x^n\, dx$ |
| $e^x \sin x$ 或 $e^x\cos x$ | $u = \sin x$ 或 $e^x$（两者均可，因为 T 与 E 会循环） | 另一个 |

### 2.4 循环积分处理

当 $u = e^x\sin x$ 类型，分部两次后原积分再次出现：

$$\int e^x\sin x\, dx = e^x\sin x - \int e^x\cos x\, dx = e^x\sin x - e^x\cos x - \int e^x\sin x\, dx.$$

设 $I = \int e^x\sin x\, dx$，则 $I = e^x\sin x - e^x\cos x - I$，故 $2I = e^x(\sin x - \cos x)$，$I = \dfrac{e^x(\sin x - \cos x)}{2} + C$。

---

## 三、换元法（5 大类型）

### 3.1 类型 1：简单换元（凑微分）

$$u = g(x),\quad du = g'(x)\, dx \implies \int f(g(x))g'(x)\, dx = \int f(u)\, du.$$

**触发信号**：被积函数含复合函数，且能凑出内层函数的微分。

**例**：$\displaystyle\int \frac{2x}{x^2+1}\, dx$。令 $u = x^2+1$，$du = 2x\, dx$：$\displaystyle\int \frac{du}{u} = \ln|u| + C = \ln(x^2+1)+C$。

**凑微分技巧**：识别分子是分母导数（或其倍数）→ 直接凑 $d(\text{分母})$。

### 3.2 类型 2：三角换元

用于消去根号 $\sqrt{a^2 \pm x^2}$ 或 $\sqrt{x^2 - a^2}$：

| 被积函数含 | 令 $x = $ | 消去后 |
|---|---|---|
| $\sqrt{a^2 - x^2}$ | $a\sin\theta$，$\theta\in[-\pi/2, \pi/2]$ | $\sqrt{a^2\cos^2\theta} = a\cos\theta$ |
| $\sqrt{a^2 + x^2}$ | $a\tan\theta$，$\theta\in(-\pi/2, \pi/2)$ | $\sqrt{a^2\sec^2\theta} = a\sec\theta$ |
| $\sqrt{x^2 - a^2}$ | $a\sec\theta$，$\theta\in[0,\pi/2)$ | $\sqrt{a^2\tan^2\theta} = a\tan\theta$ |

换元后还需将 $\theta$ 表达式换回 $x$（用反三角函数），并注意 $dx$ 的转换：
- $x=a\sin\theta$：$dx = a\cos\theta\, d\theta$
- $x=a\tan\theta$：$dx = a\sec^2\theta\, d\theta$
- $x=a\sec\theta$：$dx = a\sec\theta\tan\theta\, d\theta$

### 3.3 类型 3：倒代换

令 $x = \dfrac{1}{t}$（$dx = -\dfrac{1}{t^2}\, dt$），用于消去分母中的高次 $x^n$，或当 $x\to\infty$ 需要换到原点。

**例**：$\displaystyle\int\frac{dx}{x^2\sqrt{x^2-1}}$。令 $x=\dfrac{1}{t}$（$t>0$ 时 $x>1$），$dx = -\dfrac{1}{t^2}\, dt$：

$$\int\frac{-dt/t^2}{(1/t^2)\sqrt{1/t^2-1}} = \int\frac{-t^2\, dt}{t^2 \cdot (1/t)\sqrt{1-t^2}} = -\int\frac{t\, dt}{\sqrt{1-t^2}} = \sqrt{1-t^2}+C = \sqrt{1-1/x^2}+C.$$

### 3.4 类型 4：万能代换（Weierstrass 代换）

令 $t = \tan\dfrac{x}{2}$，则：

$$\sin x = \frac{2t}{1+t^2},\quad \cos x = \frac{1-t^2}{1+t^2},\quad dx = \frac{2}{1+t^2}\, dt.$$

用于有理三角式 $\displaystyle\int R(\sin x,\cos x)\, dx$（$R$ 是有理函数），换元后变为普通有理函数积分。

**缺点**：运算量较大。若被积函数有特殊对称性（如只含 $\sin^2 x$、$\cos^2 x$ 等偶次幂），优先用半角公式或降幂公式，避免万能代换。

### 3.5 类型 5：部分分式（有理函数分解）

用于 $\displaystyle\int\frac{P(x)}{Q(x)}\, dx$（$\deg P < \deg Q$）。将 $\dfrac{P(x)}{Q(x)}$ 分解为简单分式之和，逐项积分。

**步骤**：
1. 将 $Q(x)$ 在 $\mathbb{R}$ 上因式分解（实一次因子 $(x-a)$ 和不可约实二次因子 $(x^2+bx+c)$）。
2. 对每个 $(x-a)^k$ 因子，贡献 $k$ 个分式：$\dfrac{A_1}{x-a}+\dfrac{A_2}{(x-a)^2}+\cdots+\dfrac{A_k}{(x-a)^k}$。
3. 对每个 $(x^2+bx+c)^k$，贡献 $k$ 个分式：$\dfrac{B_j x+C_j}{(x^2+bx+c)^j}$（$j=1,\ldots,k$）。
4. 待定系数法（代特殊值或比较系数）确定 $A_i$、$B_j$、$C_j$。
5. 逐项积分（$\dfrac{A}{x-a}$ 积分得 $A\ln|x-a|$；$\dfrac{Bx+C}{x^2+bx+c}$ 配方后积分）。

---

## 四、换元法决策表

| 被积函数特征 | 换元类型 | 换元方式 |
|---|---|---|
| $f(g(x))g'(x)$ | 简单换元 | $u = g(x)$ |
| 含 $\sqrt{a^2-x^2}$ | 三角换元 $\sin$ | $x = a\sin\theta$ |
| 含 $\sqrt{a^2+x^2}$ | 三角换元 $\tan$ | $x = a\tan\theta$ |
| 含 $\sqrt{x^2-a^2}$ | 三角换元 $\sec$ | $x = a\sec\theta$ |
| 分母高次 $x^n$（$n\geq 2$）或无穷区间化 | 倒代换 | $x = 1/t$ |
| $R(\sin x, \cos x)$（有理三角） | 万能代换 | $t = \tan(x/2)$ |
| $P(x)/Q(x)$（有理函数） | 部分分式 | 分解 $Q(x)$ |

---

## 五、演示题：$\int x e^x\, dx$（分部积分 + LIATE）

> 拿到 $\int x e^x\, dx$，首先判断结构：乘积型，两个因子分别是 $x$（多项式，A 类）和 $e^x$（指数，E 类）。
>
> **LIATE 决策**：A 比 E 优先 → 令 $u = x$（A 类，求导后变常数 $1$，更简单），$dv = e^x\, dx$（E 类，积分后仍为 $e^x$，不复杂）。
>
> **计算 $v$**：$v = \int e^x\, dx = e^x$。
>
> **计算 $du$**：$u = x$，$du = dx$。
>
> **代入分部公式**：
> $$\int x e^x\, dx = uv - \int v\, du = x e^x - \int e^x\, dx = x e^x - e^x + C.$$
>
> **化简**：$= e^x(x-1) + C$。
>
> **验证（对结果求导）**：$\dfrac{d}{dx}[e^x(x-1)+C] = e^x(x-1) + e^x\cdot 1 = e^x(x-1+1) = xe^x$ ✓。
>
> **回顾决策链**：见乘积 → 分部积分 → LIATE 选 $u$ → 套公式 → 化简 → 导数验证。整个过程中，LIATE 帮我在"选哪个做 $u$"这一步节省了思考时间。

---

## 六、思考路标

1. **见乘积型 $f(x)\cdot g(x)$** → 第一直觉：分部积分。用 LIATE 决定 $u$：L 对数 > I 反三角 > A 多项式 > T 三角 > E 指数。令较前者为 $u$，较后者为 $dv$。

2. **见 $\int\ln x\, dx$ 或 $\int\arctan x\, dx$** → 这是"单因子分部"：把 $1$ 作为隐藏的 $dv = dx$，令 $u = \ln x$ 或 $u = \arctan x$（L 或 I 类永远优先为 $u$）。

3. **见分子是分母导数（或倍数）** → 简单换元（凑微分）：$\int\dfrac{f'(x)}{f(x)}\, dx = \ln|f(x)|+C$。

4. **见根号 $\sqrt{a^2 \pm x^2}$** → 三角换元（$\sin$ 或 $\tan$）；见 $\sqrt{x^2-a^2}$ → $\sec$ 换元。换元后记得回代，最后用反三角函数表示。

5. **见有理函数 $P(x)/Q(x)$** → 先检查 $\deg P \geq \deg Q$ 时多项式除法；再做部分分式分解，注意不可约二次因子对应 $Bx+C$ 型分子。

6. **见循环积分** → 设原积分为 $I$，分部两次后 $I$ 再次出现，建立关于 $I$ 的方程，解 $I$ 即可。

7. **定积分换元时注意上下限同步变换** → 令 $u = g(x)$，下限 $x=a$ 变 $u=g(a)$，上限 $x=b$ 变 $u=g(b)$，不需要再换回 $x$（不定积分则需要换回）。

8. **验证结果** → 对积分结果求导，应等于被积函数。这是最快的自检方式，养成习惯可减少不必要失误。

---

## 七、典型应用 3 例

### 例 1：分部积分 + 换元混合——$\int x\sin^2 x\, dx$

**题目**：求 $\displaystyle\int x\sin^2 x\, dx$。

**思路**：

先用半角公式降幂（不直接用 LIATE，因为 $\sin^2 x$ 不是线性三角）：

$$\sin^2 x = \frac{1-\cos 2x}{2}.$$

$$\int x\sin^2 x\, dx = \int x\cdot\frac{1-\cos 2x}{2}\, dx = \frac{1}{2}\int x\, dx - \frac{1}{2}\int x\cos 2x\, dx.$$

第一项：$\dfrac{1}{2}\int x\, dx = \dfrac{x^2}{4}$。

第二项：$\dfrac{1}{2}\int x\cos 2x\, dx$，用分部（A 优于 T）：$u=x$，$dv=\cos 2x\, dx$，$v=\dfrac{\sin 2x}{2}$：

$$\frac{1}{2}\int x\cos 2x\, dx = \frac{1}{2}\left[\frac{x\sin 2x}{2} - \int\frac{\sin 2x}{2}\, dx\right] = \frac{x\sin 2x}{4} + \frac{\cos 2x}{8}.$$

合并：

$$\int x\sin^2 x\, dx = \frac{x^2}{4} - \frac{x\sin 2x}{4} - \frac{\cos 2x}{8} + C.$$

---

### 例 2：三角换元——$\int\dfrac{dx}{\sqrt{x^2+4}}$

**题目**：求 $\displaystyle\int\frac{dx}{\sqrt{x^2+4}}$。

**思路**：

含 $\sqrt{x^2+a^2}$（$a=2$），令 $x = 2\tan\theta$，$dx = 2\sec^2\theta\, d\theta$：

$$\sqrt{x^2+4} = \sqrt{4\tan^2\theta+4} = 2\sec\theta.$$

$$\int\frac{2\sec^2\theta\, d\theta}{2\sec\theta} = \int\sec\theta\, d\theta = \ln|\sec\theta + \tan\theta| + C.$$

回代：$\tan\theta = \dfrac{x}{2}$，$\sec\theta = \dfrac{\sqrt{x^2+4}}{2}$：

$$\int\frac{dx}{\sqrt{x^2+4}} = \ln\left|\frac{\sqrt{x^2+4}}{2} + \frac{x}{2}\right| + C = \ln|x + \sqrt{x^2+4}| + C.$$

（常数 $\ln 2$ 并入 $C$）

---

### 例 3：部分分式——$\int\dfrac{2x+1}{(x-1)(x^2+1)}\, dx$

**题目**：求 $\displaystyle\int\frac{2x+1}{(x-1)(x^2+1)}\, dx$。

**思路**：

分解：$\dfrac{2x+1}{(x-1)(x^2+1)} = \dfrac{A}{x-1} + \dfrac{Bx+C}{x^2+1}$。

两边乘 $(x-1)(x^2+1)$：$2x+1 = A(x^2+1) + (Bx+C)(x-1)$。

令 $x=1$：$3 = 2A$，故 $A = \dfrac{3}{2}$。

展开比较系数（或令 $x=0,-1$ 等）：

$$x^2 \text{ 项}: 0 = A + B \implies B = -\frac{3}{2}.$$
$$x^0 \text{ 项}: 1 = A - C \implies C = A - 1 = \frac{1}{2}.$$

故：

$$\int\frac{2x+1}{(x-1)(x^2+1)}\, dx = \int\frac{3/2}{x-1}\, dx + \int\frac{-\frac{3}{2}x + \frac{1}{2}}{x^2+1}\, dx.$$

$$= \frac{3}{2}\ln|x-1| - \frac{3}{4}\ln(x^2+1) + \frac{1}{2}\arctan x + C.$$

（对 $\dfrac{x}{x^2+1}$ 积分：$\dfrac{1}{2}\ln(x^2+1)$；对 $\dfrac{1}{x^2+1}$ 积分：$\arctan x$）

---

## 八、自测题

**第 1 题**：求 $\displaystyle\int x^2\ln x\, dx$。

提示：L 优先于 A，$u=\ln x$，$dv = x^2\, dx$，$v = \dfrac{x^3}{3}$。$\displaystyle\int x^2\ln x\, dx = \frac{x^3\ln x}{3} - \int\frac{x^3}{3}\cdot\frac{1}{x}\, dx = \frac{x^3\ln x}{3} - \frac{x^3}{9} + C$。

---

**第 2 题**：求 $\displaystyle\int\frac{x^2}{\sqrt{1-x^2}}\, dx$。

提示：令 $x = \sin\theta$，$dx = \cos\theta\, d\theta$，$\sqrt{1-x^2}=\cos\theta$。$\displaystyle\int\frac{\sin^2\theta}{\cos\theta}\cos\theta\, d\theta = \int\sin^2\theta\, d\theta = \frac{\theta}{2} - \frac{\sin 2\theta}{4} + C = \frac{\arcsin x}{2} - \frac{x\sqrt{1-x^2}}{2} + C$。

---

**第 3 题**：求 $\displaystyle\int e^x\cos x\, dx$（循环分部）。

提示：$u=\cos x$，$dv=e^x\, dx$，分部后再分部一次，设 $I=\int e^x\cos x\, dx$，解方程得 $I = \dfrac{e^x(\cos x + \sin x)}{2}+C$。

---

**第 4 题**：求 $\displaystyle\int\frac{dx}{x^2-4}$（部分分式）。

提示：$\dfrac{1}{x^2-4} = \dfrac{1}{(x-2)(x+2)} = \dfrac{A}{x-2}+\dfrac{B}{x+2}$，$A=B'=\dfrac{1}{4}$，$B=-\dfrac{1}{4}$。$\displaystyle\int\frac{dx}{x^2-4} = \frac{1}{4}\ln|x-2| - \frac{1}{4}\ln|x+2| + C = \frac{1}{4}\ln\left|\frac{x-2}{x+2}\right|+C$。

---

**第 5 题**：求 $\displaystyle\int\frac{\sqrt{x-1}}{x}\, dx$（换元法）。

提示：令 $t = \sqrt{x-1}$，$x = t^2+1$，$dx = 2t\, dt$。$\displaystyle\int\frac{t\cdot 2t\, dt}{t^2+1} = 2\int\frac{t^2}{t^2+1}\, dt = 2\int\left(1-\frac{1}{t^2+1}\right)dt = 2t - 2\arctan t + C = 2\sqrt{x-1} - 2\arctan\sqrt{x-1}+C$。
