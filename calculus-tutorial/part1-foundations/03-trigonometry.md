# 第3章 三角函数

## 学习目标

通过本章学习，你将能够：

- 理解三角函数的单位圆定义，掌握任意角的三角函数
- 熟练运用弧度制进行角度计算
- 掌握三角函数的基本性质：周期性、奇偶性、单调性、有界性
- 熟练运用各类三角恒等式进行化简与证明
- 理解反三角函数的定义、定义域、值域及基本性质

---

## 3.1 三角函数的定义

### 3.1.1 弧度制

在微积分中，我们统一使用**弧度制**来度量角度。

**定义**：在单位圆中，弧长等于半径的弧所对的圆心角定义为 **1 弧度**（rad）。

弧度与角度的换算关系：

$$\pi \text{ rad} = 180°$$

常用换算：

| 角度 | $0°$ | $30°$ | $45°$ | $60°$ | $90°$ | $180°$ | $360°$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 弧度 | $0$ | $\frac{\pi}{6}$ | $\frac{\pi}{4}$ | $\frac{\pi}{3}$ | $\frac{\pi}{2}$ | $\pi$ | $2\pi$ |

**弧长公式**：半径为 $r$ 的圆，圆心角为 $\theta$（弧度）的弧长为

$$l = r\theta$$

**扇形面积**：

$$S = \frac{1}{2}r^2\theta = \frac{1}{2}lr$$

### 3.1.2 单位圆定义

**单位圆**是以原点为圆心、半径为 1 的圆，其方程为 $x^2 + y^2 = 1$。

设角 $\theta$ 的终边与单位圆交于点 $P(x, y)$，则定义：

$$\sin\theta = y, \quad \cos\theta = x, \quad \tan\theta = \frac{y}{x} \, (x \neq 0)$$

$$\csc\theta = \frac{1}{y} \, (y \neq 0), \quad \sec\theta = \frac{1}{x} \, (x \neq 0), \quad \cot\theta = \frac{x}{y} \, (y \neq 0)$$

### 3.1.3 任意角的三角函数

单位圆定义自然地将三角函数推广到任意实数。角 $\theta$ 可以是正角（逆时针旋转）、负角（顺时针旋转）或零角。

**特殊角的三角函数值**：

| $\theta$ | $0$ | $\frac{\pi}{6}$ | $\frac{\pi}{4}$ | $\frac{\pi}{3}$ | $\frac{\pi}{2}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $\sin\theta$ | $0$ | $\frac{1}{2}$ | $\frac{\sqrt{2}}{2}$ | $\frac{\sqrt{3}}{2}$ | $1$ |
| $\cos\theta$ | $1$ | $\frac{\sqrt{3}}{2}$ | $\frac{\sqrt{2}}{2}$ | $\frac{1}{2}$ | $0$ |
| $\tan\theta$ | $0$ | $\frac{\sqrt{3}}{3}$ | $1$ | $\sqrt{3}$ | 无定义 |

> **例题 3.1** 求 $\sin\frac{5\pi}{4}$ 和 $\cos\left(-\frac{2\pi}{3}\right)$ 的值。

**解**：

$\frac{5\pi}{4} = \pi + \frac{\pi}{4}$，终边在第三象限，故

$$\sin\frac{5\pi}{4} = -\sin\frac{\pi}{4} = -\frac{\sqrt{2}}{2}$$

$-\frac{2\pi}{3}$ 与 $\frac{4\pi}{3}$ 终边相同，在第三象限，故

$$\cos\left(-\frac{2\pi}{3}\right) = \cos\frac{4\pi}{3} = -\cos\frac{\pi}{3} = -\frac{1}{2}$$

---

## 3.2 三角函数的基本性质

### 3.2.1 定义域与值域

| 函数 | 定义域 | 值域 |
|:---:|:---:|:---:|
| $\sin x$ | $\mathbb{R}$ | $[-1, 1]$ |
| $\cos x$ | $\mathbb{R}$ | $[-1, 1]$ |
| $\tan x$ | $\{x \mid x \neq \frac{\pi}{2} + k\pi, k \in \mathbb{Z}\}$ | $\mathbb{R}$ |

### 3.2.2 周期性

若存在非零常数 $T$，使得对定义域内的所有 $x$ 都有 $f(x + T) = f(x)$，则称 $f(x)$ 是周期函数，满足条件的最小正数 $T$ 称为**最小正周期**。

- $\sin x$ 和 $\cos x$ 的最小正周期为 $2\pi$
- $\tan x$ 和 $\cot x$ 的最小正周期为 $\pi$

### 3.2.3 奇偶性

**奇函数**：$f(-x) = -f(x)$，图像关于原点对称

**偶函数**：$f(-x) = f(x)$，图像关于 $y$ 轴对称

- $\sin x$、$\tan x$、$\cot x$、$\csc x$ 是**奇函数**
- $\cos x$、$\sec x$ 是**偶函数**

### 3.2.4 单调性

**正弦函数** $y = \sin x$：
- 在 $\left[-\frac{\pi}{2} + 2k\pi, \frac{\pi}{2} + 2k\pi\right]$ 上单调递增
- 在 $\left[\frac{\pi}{2} + 2k\pi, \frac{3\pi}{2} + 2k\pi\right]$ 上单调递减

**余弦函数** $y = \cos x$：
- 在 $[-\pi + 2k\pi, 2k\pi]$ 上单调递增
- 在 $[2k\pi, \pi + 2k\pi]$ 上单调递减

**正切函数** $y = \tan x$：
- 在每个区间 $\left(-\frac{\pi}{2} + k\pi, \frac{\pi}{2} + k\pi\right)$ 上单调递增

### 3.2.5 有界性

$\sin x$ 和 $\cos x$ 是有界函数：

$$-1 \leq \sin x \leq 1, \quad -1 \leq \cos x \leq 1$$

$\tan x$ 在其定义域上无界。

### 3.2.6 函数图像

**正弦曲线** $y = \sin x$：从原点出发，向上到最高点 $({\pi}/{2}, 1)$，下降经过 $(\pi, 0)$，到最低点 $({3\pi}/{2}, -1)$，再回到 $(2\pi, 0)$，周而复始。

**余弦曲线** $y = \cos x$：可看作正弦曲线向左平移 $\frac{\pi}{2}$，即 $\cos x = \sin\left(x + \frac{\pi}{2}\right)$。

**正切曲线** $y = \tan x$：以 $x = \frac{\pi}{2} + k\pi$ 为渐近线，在每个周期内从 $-\infty$ 单调递增到 $+\infty$。

---

## 3.3 三角恒等式

### 3.3.1 基本恒等式

**平方关系**：

$$\sin^2 x + \cos^2 x = 1$$

$$1 + \tan^2 x = \sec^2 x$$

$$1 + \cot^2 x = \csc^2 x$$

**商数关系**：

$$\tan x = \frac{\sin x}{\cos x}, \quad \cot x = \frac{\cos x}{\sin x}$$

**倒数关系**：

$$\sin x \cdot \csc x = 1, \quad \cos x \cdot \sec x = 1, \quad \tan x \cdot \cot x = 1$$

### 3.3.2 诱导公式

设 $k \in \mathbb{Z}$，有：

$$\sin(x + 2k\pi) = \sin x, \quad \cos(x + 2k\pi) = \cos x$$

$$\sin(-x) = -\sin x, \quad \cos(-x) = \cos x$$

$$\sin(\pi - x) = \sin x, \quad \cos(\pi - x) = -\cos x$$

$$\sin\left(\frac{\pi}{2} - x\right) = \cos x, \quad \cos\left(\frac{\pi}{2} - x\right) = \sin x$$

**口诀**："奇变偶不变，符号看象限"

### 3.3.3 和差公式

$$\sin(\alpha \pm \beta) = \sin\alpha\cos\beta \pm \cos\alpha\sin\beta$$

$$\cos(\alpha \pm \beta) = \cos\alpha\cos\beta \mp \sin\alpha\sin\beta$$

$$\tan(\alpha \pm \beta) = \frac{\tan\alpha \pm \tan\beta}{1 \mp \tan\alpha\tan\beta}$$

> **例题 3.2** 求 $\cos 75°$ 的精确值。

**解**：$75° = 45° + 30°$，由余弦和角公式：

$$\cos 75° = \cos(45° + 30°) = \cos 45°\cos 30° - \sin 45°\sin 30°$$

$$= \frac{\sqrt{2}}{2} \cdot \frac{\sqrt{3}}{2} - \frac{\sqrt{2}}{2} \cdot \frac{1}{2} = \frac{\sqrt{6} - \sqrt{2}}{4}$$

### 3.3.4 倍角公式

$$\sin 2\alpha = 2\sin\alpha\cos\alpha$$

$$\cos 2\alpha = \cos^2\alpha - \sin^2\alpha = 2\cos^2\alpha - 1 = 1 - 2\sin^2\alpha$$

$$\tan 2\alpha = \frac{2\tan\alpha}{1 - \tan^2\alpha}$$

### 3.3.5 半角公式

$$\sin\frac{\alpha}{2} = \pm\sqrt{\frac{1 - \cos\alpha}{2}}$$

$$\cos\frac{\alpha}{2} = \pm\sqrt{\frac{1 + \cos\alpha}{2}}$$

$$\tan\frac{\alpha}{2} = \pm\sqrt{\frac{1 - \cos\alpha}{1 + \cos\alpha}} = \frac{\sin\alpha}{1 + \cos\alpha} = \frac{1 - \cos\alpha}{\sin\alpha}$$

（符号由 $\frac{\alpha}{2}$ 所在象限决定）

### 3.3.6 积化和差与和差化积

**积化和差**：

$$\sin\alpha\cos\beta = \frac{1}{2}[\sin(\alpha + \beta) + \sin(\alpha - \beta)]$$

$$\cos\alpha\sin\beta = \frac{1}{2}[\sin(\alpha + \beta) - \sin(\alpha - \beta)]$$

$$\cos\alpha\cos\beta = \frac{1}{2}[\cos(\alpha + \beta) + \cos(\alpha - \beta)]$$

$$\sin\alpha\sin\beta = -\frac{1}{2}[\cos(\alpha + \beta) - \cos(\alpha - \beta)]$$

**和差化积**：

$$\sin A + \sin B = 2\sin\frac{A + B}{2}\cos\frac{A - B}{2}$$

$$\sin A - \sin B = 2\cos\frac{A + B}{2}\sin\frac{A - B}{2}$$

$$\cos A + \cos B = 2\cos\frac{A + B}{2}\cos\frac{A - B}{2}$$

$$\cos A - \cos B = -2\sin\frac{A + B}{2}\sin\frac{A - B}{2}$$

---

## 3.4 反三角函数

由于三角函数不是一一映射，为定义反函数需要限制定义域。

### 3.4.1 反正弦函数

**定义**：$y = \arcsin x$ 是 $y = \sin x$（$x \in \left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$）的反函数。

- **定义域**：$[-1, 1]$
- **值域**：$\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$
- **性质**：奇函数，在定义域上单调递增

**基本关系**：

$$\sin(\arcsin x) = x, \quad x \in [-1, 1]$$

$$\arcsin(\sin x) = x, \quad x \in \left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$$

### 3.4.2 反余弦函数

**定义**：$y = \arccos x$ 是 $y = \cos x$（$x \in [0, \pi]$）的反函数。

- **定义域**：$[-1, 1]$
- **值域**：$[0, \pi]$
- **性质**：非奇非偶，在定义域上单调递减

**重要恒等式**：

$$\arcsin x + \arccos x = \frac{\pi}{2}, \quad x \in [-1, 1]$$

### 3.4.3 反正切函数

**定义**：$y = \arctan x$ 是 $y = \tan x$（$x \in \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$）的反函数。

- **定义域**：$\mathbb{R}$
- **值域**：$\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$
- **性质**：奇函数，在定义域上单调递增
- **渐近线**：$\lim_{x \to +\infty}\arctan x = \frac{\pi}{2}$，$\lim_{x \to -\infty}\arctan x = -\frac{\pi}{2}$

**重要恒等式**：

$$\arctan x + \arctan\frac{1}{x} = \begin{cases} \frac{\pi}{2}, & x > 0 \\ -\frac{\pi}{2}, & x < 0 \end{cases}$$

> **例题 3.3** 求 $\arcsin\left(-\frac{\sqrt{3}}{2}\right)$ 和 $\arctan(-1)$ 的值。

**解**：

由于 $\sin\left(-\frac{\pi}{3}\right) = -\frac{\sqrt{3}}{2}$ 且 $-\frac{\pi}{3} \in \left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$，故

$$\arcsin\left(-\frac{\sqrt{3}}{2}\right) = -\frac{\pi}{3}$$

由于 $\tan\left(-\frac{\pi}{4}\right) = -1$ 且 $-\frac{\pi}{4} \in \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$，故

$$\arctan(-1) = -\frac{\pi}{4}$$

---

## 本章小结

1. **三角函数的定义**：通过单位圆将三角函数推广到任意实数，弧度制是微积分的标准角度度量。

2. **基本性质**：三角函数具有周期性、奇偶性、单调性和有界性，这些性质在分析中有重要应用。

3. **三角恒等式**：包括基本恒等式、和差公式、倍角公式、半角公式、积化和差与和差化积，是化简和证明的有力工具。

4. **反三角函数**：通过限制三角函数的定义域得到一一映射，从而定义反函数。$\arcsin$、$\arccos$、$\arctan$ 是最常用的三个反三角函数。

---

## 深度学习应用

三角函数在现代深度学习中有多处核心应用，以下介绍三个重要场景。

### 3.5.1 Transformer 中的位置编码

Transformer 模型处理序列时，自注意力机制本身不含位置信息，需要额外的**位置编码**（Positional Encoding）来注入序列顺序。Vaswani 等人（2017）选择正弦/余弦函数：

$$PE_{(pos,\ 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos,\ 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

其中 $pos$ 是词在序列中的位置，$i$ 是维度索引，$d$ 是模型的嵌入维度。

**为什么使用三角函数？**

关键优势在于**相对位置可以用线性变换表示**。对于固定的偏移量 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数——这直接来自正弦/余弦的和差公式：

$$\sin(A + B) = \sin A\cos B + \cos A\sin B$$

因此模型可以通过线性运算学习到"某个位置比另一个位置早 $k$ 步"这一相对关系，而这对绝对位置编码方案（如可学习的嵌入）并不成立。

### 3.5.2 傅里叶特征（Fourier Features）

神经网络难以直接拟合高频函数（谱偏差/频率偏置）。**随机傅里叶特征**（Random Fourier Features，Rahimi & Recht 2007）将低维输入 $\mathbf{x} \in \mathbb{R}^n$ 映射到高维特征空间：

$$\gamma(\mathbf{x}) = \left[\cos(2\pi \mathbf{B}\mathbf{x}),\ \sin(2\pi \mathbf{B}\mathbf{x})\right]$$

其中 $\mathbf{B}$ 的每一行从高斯分布中随机采样。这一映射将核函数的内积近似为欧氏空间的内积，同时帮助网络突破谱偏差，更好地学习高频细节——在 NeRF（神经辐射场）等任务中效果显著。

### 3.5.3 周期性数据建模

现实中大量数据具有周期性：日销售量（7 天周期）、电力负荷（24 小时周期）、气温（365 天周期）。将原始时间特征 $t$ 转化为三角函数特征，能让模型无需学习即可感知周期性：

$$\left(\sin\frac{2\pi t}{T},\ \cos\frac{2\pi t}{T}\right)$$

使用**一对** $(\sin, \cos)$ 而非单个函数的原因：单个 $\sin$ 或 $\cos$ 在一个周期内不是单射（同一函数值对应两个时间点），而二维向量 $(\sin\theta, \cos\theta)$ 在单位圆上唯一确定角度 $\theta$，从而无歧义地表示周期内的相对位置。

### 代码示例：Transformer 位置编码实现

```python
import torch
import math

def positional_encoding(seq_len, d_model):
    """Transformer位置编码"""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()

    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

    return pe

# 示例
pe = positional_encoding(seq_len=100, d_model=512)
print(f"位置编码形状: {pe.shape}")  # torch.Size([100, 512])
```

`div_term` 的计算利用了指数与对数的等价形式：

$$\frac{1}{10000^{2i/d}} = e^{-\frac{2i}{d}\ln 10000}$$

用指数运算代替幂运算，在数值上更稳定。

---

## 练习题

**1.** ⭐ 将下列角度化为弧度，或将弧度化为角度：
   (a) $150°$　　(b) $-45°$　　(c) $\frac{5\pi}{6}$　　(d) $-\frac{3\pi}{4}$

**2.** ⭐ 求下列三角函数值：
   (a) $\sin\frac{7\pi}{6}$　　(b) $\cos\left(-\frac{5\pi}{3}\right)$　　(c) $\tan\frac{3\pi}{4}$

**3.** ⭐⭐ 已知 $\sin\alpha = \frac{3}{5}$，$\alpha \in \left(\frac{\pi}{2}, \pi\right)$，求 $\cos\alpha$、$\tan\alpha$ 和 $\sin 2\alpha$ 的值。

**4.** ⭐⭐ 证明恒等式：$\frac{1 - \cos 2x}{\sin 2x} = \tan x$

**5.** ⭐⭐⭐ 求 $\arcsin\left(\sin\frac{5\pi}{6}\right)$ 和 $\arccos\left(\cos\frac{5\pi}{3}\right)$ 的值。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.**
(a) $150° = 150 \times \frac{\pi}{180} = \frac{5\pi}{6}$

(b) $-45° = -45 \times \frac{\pi}{180} = -\frac{\pi}{4}$

(c) $\frac{5\pi}{6} = \frac{5\pi}{6} \times \frac{180}{\pi} = 150°$

(d) $-\frac{3\pi}{4} = -\frac{3\pi}{4} \times \frac{180}{\pi} = -135°$

---

**2.**
(a) $\frac{7\pi}{6} = \pi + \frac{\pi}{6}$，在第三象限，$\sin\frac{7\pi}{6} = -\sin\frac{\pi}{6} = -\frac{1}{2}$

(b) $-\frac{5\pi}{3}$ 与 $\frac{\pi}{3}$ 终边相同，$\cos\left(-\frac{5\pi}{3}\right) = \cos\frac{\pi}{3} = \frac{1}{2}$

(c) $\frac{3\pi}{4} = \pi - \frac{\pi}{4}$，在第二象限，$\tan\frac{3\pi}{4} = -\tan\frac{\pi}{4} = -1$

---

**3.** 由于 $\alpha \in \left(\frac{\pi}{2}, \pi\right)$（第二象限），$\cos\alpha < 0$。

由 $\sin^2\alpha + \cos^2\alpha = 1$：

$$\cos\alpha = -\sqrt{1 - \sin^2\alpha} = -\sqrt{1 - \frac{9}{25}} = -\frac{4}{5}$$

$$\tan\alpha = \frac{\sin\alpha}{\cos\alpha} = \frac{3/5}{-4/5} = -\frac{3}{4}$$

$$\sin 2\alpha = 2\sin\alpha\cos\alpha = 2 \times \frac{3}{5} \times \left(-\frac{4}{5}\right) = -\frac{24}{25}$$

---

**4.** **证明**：利用倍角公式

$$\frac{1 - \cos 2x}{\sin 2x} = \frac{1 - (1 - 2\sin^2 x)}{2\sin x\cos x} = \frac{2\sin^2 x}{2\sin x\cos x} = \frac{\sin x}{\cos x} = \tan x$$

证毕。 $\square$

---

**5.**
对于 $\arcsin\left(\sin\frac{5\pi}{6}\right)$：

$\sin\frac{5\pi}{6} = \sin\left(\pi - \frac{\pi}{6}\right) = \sin\frac{\pi}{6} = \frac{1}{2}$

由于 $\arcsin$ 的值域是 $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$，而 $\frac{\pi}{6} \in \left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$，故

$$\arcsin\left(\sin\frac{5\pi}{6}\right) = \arcsin\frac{1}{2} = \frac{\pi}{6}$$

对于 $\arccos\left(\cos\frac{5\pi}{3}\right)$：

$\cos\frac{5\pi}{3} = \cos\left(2\pi - \frac{\pi}{3}\right) = \cos\frac{\pi}{3} = \frac{1}{2}$

由于 $\arccos$ 的值域是 $[0, \pi]$，而 $\frac{\pi}{3} \in [0, \pi]$，故

$$\arccos\left(\cos\frac{5\pi}{3}\right) = \arccos\frac{1}{2} = \frac{\pi}{3}$$

</details>
