# 第1章 集合与实数

## 学习目标

通过本章学习，你将能够：

- 理解集合的基本概念，掌握集合的两种表示方法
- 熟练运用集合的基本运算：并集、交集、补集、差集
- 了解实数系统的构成及其基本性质
- 掌握区间和邻域的概念，为后续极限学习做准备
- 理解确界原理，认识实数的完备性

---

## 1.1 集合的基本概念

### 1.1.1 集合的定义

**集合**是数学中最基本的概念之一。直观地说，集合是由一些确定的、互不相同的对象组成的整体。组成集合的对象称为该集合的**元素**。

若元素 $a$ 属于集合 $A$，记作 $a \in A$；若 $a$ 不属于 $A$，记作 $a \notin A$。

### 1.1.2 集合的表示方法

**列举法**：将集合的元素一一列出，用花括号括起来。

$$A = \{1, 2, 3, 4, 5\}$$

**描述法**：用元素的特征性质来描述集合。

$$B = \{x \mid x \text{ 是正偶数}\} = \{x \in \mathbb{Z}^+ \mid x \equiv 0 \pmod{2}\}$$

### 1.1.3 常用数集符号

- $\mathbb{N}$：自然数集 $\{0, 1, 2, 3, \ldots\}$
- $\mathbb{Z}$：整数集 $\{\ldots, -2, -1, 0, 1, 2, \ldots\}$
- $\mathbb{Q}$：有理数集
- $\mathbb{R}$：实数集
- $\emptyset$：空集（不含任何元素的集合）

### 1.1.4 集合间的关系

**子集**：若集合 $A$ 的每个元素都是集合 $B$ 的元素，则称 $A$ 是 $B$ 的子集，记作 $A \subseteq B$。

**真子集**：若 $A \subseteq B$ 且 $A \neq B$，则称 $A$ 是 $B$ 的真子集，记作 $A \subsetneq B$。

**集合相等**：若 $A \subseteq B$ 且 $B \subseteq A$，则 $A = B$。

### 1.1.5 集合的基本运算

设 $A$、$B$ 是两个集合，$U$ 是全集。

**并集**：$A \cup B = \{x \mid x \in A \text{ 或 } x \in B\}$

**交集**：$A \cap B = \{x \mid x \in A \text{ 且 } x \in B\}$

**补集**：$A^c = \complement_U A = \{x \in U \mid x \notin A\}$

**差集**：$A \setminus B = \{x \mid x \in A \text{ 且 } x \notin B\}$

> **例题 1.1** 设 $A = \{1, 2, 3, 4\}$，$B = \{3, 4, 5, 6\}$，$U = \{1, 2, 3, 4, 5, 6, 7\}$，求 $A \cup B$、$A \cap B$、$A^c$、$A \setminus B$。

**解**：
- $A \cup B = \{1, 2, 3, 4, 5, 6\}$
- $A \cap B = \{3, 4\}$
- $A^c = \{5, 6, 7\}$
- $A \setminus B = \{1, 2\}$

---

## 1.2 实数系统

### 1.2.1 实数的构成

实数系统按照历史发展和逻辑结构，可以分为以下层次：

$$\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R}$$

- **自然数** $\mathbb{N}$：用于计数的数 $0, 1, 2, 3, \ldots$
- **整数** $\mathbb{Z}$：自然数扩充负整数
- **有理数** $\mathbb{Q}$：可表示为 $\frac{p}{q}$（$p, q \in \mathbb{Z}$，$q \neq 0$）的数
- **无理数**：不能表示为有理数的实数，如 $\sqrt{2}$、$\pi$、$e$

### 1.2.2 实数的基本性质

**有序性**：对于任意两个实数 $a$ 和 $b$，下列三种关系有且仅有一种成立：
$$a < b, \quad a = b, \quad a > b$$

**稠密性**：在任意两个不相等的实数之间，必存在无穷多个有理数和无穷多个无理数。

**阿基米德性质**：对于任意正实数 $a$ 和 $b$，存在正整数 $n$，使得 $na > b$。

### 1.2.3 绝对值

实数 $a$ 的**绝对值**定义为：

$$|a| = \begin{cases} a, & \text{若 } a \geq 0 \\ -a, & \text{若 } a < 0 \end{cases}$$

**绝对值的基本性质**：

1. $|a| \geq 0$，等号成立当且仅当 $a = 0$
2. $|ab| = |a| \cdot |b|$
3. $\left|\frac{a}{b}\right| = \frac{|a|}{|b|}$（$b \neq 0$）
4. **三角不等式**：$|a + b| \leq |a| + |b|$
5. **逆三角不等式**：$||a| - |b|| \leq |a - b|$

> **例题 1.2** 解不等式 $|2x - 1| < 3$。

**解**：由绝对值的定义，$|2x - 1| < 3$ 等价于
$$-3 < 2x - 1 < 3$$
即 $-2 < 2x < 4$，解得 $-1 < x < 2$。

因此，解集为 $\{x \mid -1 < x < 2\} = (-1, 2)$。

---

## 1.3 区间与邻域

### 1.3.1 区间

设 $a, b \in \mathbb{R}$ 且 $a < b$。

**有限区间**：

| 名称 | 符号 | 集合表示 |
|:---:|:---:|:---:|
| 开区间 | $(a, b)$ | $\{x \mid a < x < b\}$ |
| 闭区间 | $[a, b]$ | $\{x \mid a \leq x \leq b\}$ |
| 左开右闭 | $(a, b]$ | $\{x \mid a < x \leq b\}$ |
| 左闭右开 | $[a, b)$ | $\{x \mid a \leq x < b\}$ |

**无限区间**：

- $(a, +\infty) = \{x \mid x > a\}$
- $[a, +\infty) = \{x \mid x \geq a\}$
- $(-\infty, b) = \{x \mid x < b\}$
- $(-\infty, b] = \{x \mid x \leq b\}$
- $(-\infty, +\infty) = \mathbb{R}$

### 1.3.2 邻域

**邻域**是极限理论的重要工具。

**$\delta$ 邻域**：设 $a \in \mathbb{R}$，$\delta > 0$，称开区间 $(a - \delta, a + \delta)$ 为点 $a$ 的 $\delta$ 邻域，记作 $U(a, \delta)$ 或 $U_\delta(a)$。

$$U(a, \delta) = \{x \mid |x - a| < \delta\}$$

**去心邻域**：从邻域中去掉中心点 $a$，记作 $\mathring{U}(a, \delta)$。

$$\mathring{U}(a, \delta) = \{x \mid 0 < |x - a| < \delta\}$$

**单侧邻域**：

- 左邻域：$U^-(a, \delta) = (a - \delta, a)$
- 右邻域：$U^+(a, \delta) = (a, a + \delta)$

> **例题 1.3** 写出点 $x_0 = 2$ 的 $0.5$ 邻域和去心邻域。

**解**：
- $U(2, 0.5) = (1.5, 2.5) = \{x \mid |x - 2| < 0.5\}$
- $\mathring{U}(2, 0.5) = (1.5, 2) \cup (2, 2.5) = \{x \mid 0 < |x - 2| < 0.5\}$

---

## 1.4 确界原理

确界原理是实数完备性的重要体现，是微积分理论的基石之一。

### 1.4.1 上界与下界

设 $S \subseteq \mathbb{R}$ 是非空集合。

**上界**：若存在 $M \in \mathbb{R}$，使得对一切 $x \in S$ 都有 $x \leq M$，则称 $M$ 是 $S$ 的一个上界，称 $S$ **有上界**。

**下界**：若存在 $m \in \mathbb{R}$，使得对一切 $x \in S$ 都有 $x \geq m$，则称 $m$ 是 $S$ 的一个下界，称 $S$ **有下界**。

**有界集**：既有上界又有下界的集合称为有界集。

### 1.4.2 上确界与下确界

**上确界**（最小上界）：设 $S$ 有上界，若 $\beta$ 满足：
1. $\beta$ 是 $S$ 的上界：$\forall x \in S, x \leq \beta$
2. $\beta$ 是最小的上界：$\forall \varepsilon > 0, \exists x_0 \in S$，使得 $x_0 > \beta - \varepsilon$

则称 $\beta$ 为 $S$ 的上确界，记作 $\beta = \sup S$。

**下确界**（最大下界）：设 $S$ 有下界，若 $\alpha$ 满足：
1. $\alpha$ 是 $S$ 的下界：$\forall x \in S, x \geq \alpha$
2. $\alpha$ 是最大的下界：$\forall \varepsilon > 0, \exists x_0 \in S$，使得 $x_0 < \alpha + \varepsilon$

则称 $\alpha$ 为 $S$ 的下确界，记作 $\alpha = \inf S$。

### 1.4.3 确界原理

**确界原理**：设 $S$ 是 $\mathbb{R}$ 的非空子集。

- 若 $S$ 有上界，则 $S$ 必有上确界。
- 若 $S$ 有下界，则 $S$ 必有下确界。

> **注**：确界原理是实数完备性的等价表述之一。有理数集不满足确界原理，例如 $\{x \in \mathbb{Q} \mid x^2 < 2\}$ 在有理数中没有上确界。

> **例题 1.4** 求集合 $S = \left\{\frac{n}{n+1} \mid n \in \mathbb{N}^+\right\}$ 的上确界和下确界。

**解**：$S = \left\{\frac{1}{2}, \frac{2}{3}, \frac{3}{4}, \frac{4}{5}, \ldots\right\}$

对于任意 $n \in \mathbb{N}^+$，有 $\frac{n}{n+1} = 1 - \frac{1}{n+1}$。

- 当 $n = 1$ 时，$\frac{n}{n+1} = \frac{1}{2}$ 是最小值，故 $\inf S = \frac{1}{2}$。
- 当 $n \to \infty$ 时，$\frac{n}{n+1} \to 1$，但 $\frac{n}{n+1} < 1$ 对所有 $n$ 成立。

因此，$\sup S = 1$（上确界不属于 $S$），$\inf S = \frac{1}{2}$（下确界属于 $S$，即为最小值）。

---

## 本章小结

1. **集合**是数学的基础语言，集合运算（并、交、补、差）是基本工具。

2. **实数系统**具有有序性、稠密性和完备性，是微积分的数学基础。

3. **区间**是实数集的重要子集，**邻域**概念是定义极限的关键工具。

4. **确界原理**保证了有界集必有确界，这是实数完备性的核心体现，也是极限存在性定理的理论基础。

---

## 深度学习应用

本章介绍的集合与实数概念是深度学习的数学基石。

### 数据集与特征空间

在深度学习中，数据集可以看作样本的集合：
- 训练集、验证集、测试集的集合划分
- 特征空间 $\mathbb{R}^n$ 的概念
- 样本作为高维空间中的点

### 超参数的取值范围

学习率、正则化系数等超参数的取值范围：
- 学习率 $\eta \in (0, 1)$
- 区间与邻域在参数搜索中的应用

### 确界与模型容量

上确界、下确界的概念在深度学习中有直接对应：
- 损失函数的下界
- 模型容量的上界

### 代码示例

```python
import torch
from torch.utils.data import Dataset, random_split

# 数据集的集合表示
class SimpleDataset(Dataset):
    def __init__(self, n_samples=1000, n_features=10):
        # 特征空间 R^n 中的样本点
        self.X = torch.randn(n_samples, n_features)
        self.y = torch.randint(0, 2, (n_samples,))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 集合的划分：训练集、验证集、测试集
dataset = SimpleDataset(1000, 10)
train_set, val_set, test_set = random_split(dataset, [0.7, 0.15, 0.15])
print(f"训练集大小: {len(train_set)}, 验证集: {len(val_set)}, 测试集: {len(test_set)}")
```

### 延伸阅读

- 《Deep Learning》(Goodfellow) 第5章：机器学习基础
- 特征空间与表示学习的关系

---

## 练习题

**1.** ⭐ 设 $A = \{x \mid x^2 - 3x + 2 = 0\}$，$B = \{x \mid x^2 - 2x = 0\}$，求 $A \cup B$ 和 $A \cap B$。

**2.** ⭐ 解不等式 $|3x + 2| \leq 5$，并用区间表示解集。

**3.** ⭐⭐ 证明三角不等式：对任意 $a, b \in \mathbb{R}$，有 $|a + b| \leq |a| + |b|$。

**4.** ⭐⭐ 求集合 $S = \left\{\frac{1}{n} \mid n \in \mathbb{N}^+\right\}$ 的上确界和下确界，并说明它们是否属于 $S$。

**5.** ⭐⭐⭐ 设 $A$ 和 $B$ 是 $\mathbb{R}$ 的非空有界子集，定义 $A + B = \{a + b \mid a \in A, b \in B\}$。证明：$\sup(A + B) = \sup A + \sup B$。

---

## 练习答案

<details>
<summary>点击展开答案</summary>

**1.** 解方程得 $A = \{1, 2\}$，$B = \{0, 2\}$。

$A \cup B = \{0, 1, 2\}$，$A \cap B = \{2\}$。

---

**2.** $|3x + 2| \leq 5$ 等价于 $-5 \leq 3x + 2 \leq 5$。

解得 $-7 \leq 3x \leq 3$，即 $-\frac{7}{3} \leq x \leq 1$。

解集为 $\left[-\frac{7}{3}, 1\right]$。

---

**3.** **证明**：对任意 $a, b \in \mathbb{R}$，由绝对值定义有
$$-|a| \leq a \leq |a|, \quad -|b| \leq b \leq |b|$$

两式相加得
$$-(|a| + |b|) \leq a + b \leq |a| + |b|$$

由绝对值定义，这等价于 $|a + b| \leq |a| + |b|$。 $\square$

---

**4.** $S = \left\{1, \frac{1}{2}, \frac{1}{3}, \frac{1}{4}, \ldots\right\}$

- $\sup S = 1$，属于 $S$（当 $n = 1$ 时取得）。
- $\inf S = 0$，不属于 $S$（因为 $\frac{1}{n} > 0$ 对所有 $n$ 成立，但 $\frac{1}{n}$ 可以任意接近 $0$）。

---

**5.** **证明**：设 $\alpha = \sup A$，$\beta = \sup B$。

**Step 1**：证明 $\alpha + \beta$ 是 $A + B$ 的上界。

对任意 $a \in A$，$b \in B$，有 $a \leq \alpha$，$b \leq \beta$，故 $a + b \leq \alpha + \beta$。

**Step 2**：证明 $\alpha + \beta$ 是最小上界。

对任意 $\varepsilon > 0$，由上确界定义，存在 $a_0 \in A$ 使得 $a_0 > \alpha - \frac{\varepsilon}{2}$，存在 $b_0 \in B$ 使得 $b_0 > \beta - \frac{\varepsilon}{2}$。

则 $a_0 + b_0 > \alpha + \beta - \varepsilon$，而 $a_0 + b_0 \in A + B$。

由上确界定义，$\sup(A + B) = \alpha + \beta = \sup A + \sup B$。 $\square$

</details>
