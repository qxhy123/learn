# 集合的基本运算

单个集合的语言已经足够表达"哪些对象在一起"，但现实问题往往需要把两个集合**组合**或**比较**：两个班共同喜欢某项目的人是谁？至少满足两个条件之一的学生有哪些？某全体中不满足某条件的对象是什么？这就是交集、并集、补集三种运算的动机。本章系统建立这三种运算，并掌握它们的法则与高考常见题型。

## 一、概念特征：怎么一眼认出

- **交集**：两个集合的"共同部分"，元素必须**同时属于两个集合**。关键词：且、同时、公共。
- **并集**：两个集合的"合并"，元素**属于两者之一即可**。关键词：或、至少一个、合并。
- **补集**：在全集中，**不属于**某集合的全部元素。关键词：全集、除去、不满足。

看到题目中的"公共元素"就想交集；看到"合并所有"就想并集；看到"全集中排除某部分"就想补集。

## 二、定义与核心工具

### 1. 交集

设 $A$、$B$ 是两个集合，**交集** $A \cap B$ 定义为

$$A \cap B = \{x \mid x \in A \text{ 且 } x \in B\}.$$

即同时属于 $A$ 和 $B$ 的所有元素组成的集合。

**基本性质**：
- $A \cap A = A$（幂等律）；
- $A \cap \varnothing = \varnothing$；
- $A \cap B = B \cap A$（交换律）；
- $(A \cap B) \cap C = A \cap (B \cap C)$（结合律）；
- $A \cap B \subseteq A$，$A \cap B \subseteq B$。

### 2. 并集

设 $A$、$B$ 是两个集合，**并集** $A \cup B$ 定义为

$$A \cup B = \{x \mid x \in A \text{ 或 } x \in B\}.$$

即属于 $A$ **或**属于 $B$（含两者都属于）的所有元素组成的集合。这里"或"是数学意义上的**inclusive or**（相容或），即两者同时满足也算在内。

**基本性质**：
- $A \cup A = A$（幂等律）；
- $A \cup \varnothing = A$；
- $A \cup B = B \cup A$（交换律）；
- $(A \cup B) \cup C = A \cup (B \cup C)$（结合律）；
- $A \subseteq A \cup B$，$B \subseteq A \cup B$。

### 3. 补集

设 $U$ 是**全集**（universal set，问题讨论范围内最大的集合），$A \subseteq U$，则 $A$ 在 $U$ 中的**补集**（complement）定义为

$$\complement_U A = \{x \mid x \in U \text{ 且 } x \notin A\}.$$

即全集中所有不属于 $A$ 的元素组成的集合。

**基本性质**：
- $\complement_U(\complement_U A) = A$（双补律）；
- $A \cup \complement_U A = U$；
- $A \cap \complement_U A = \varnothing$；
- $\complement_U U = \varnothing$，$\complement_U \varnothing = U$。

### 4. 运算律（含德摩根律）

**交换律**：$A \cap B = B \cap A$，$A \cup B = B \cup A$。

**结合律**：$(A \cap B) \cap C = A \cap (B \cap C)$，$(A \cup B) \cup C = A \cup (B \cup C)$。

**分配律**：
$$A \cap (B \cup C) = (A \cap B) \cup (A \cap C),$$
$$A \cup (B \cap C) = (A \cup B) \cap (A \cup C).$$

**德摩根律**（De Morgan's laws，高考重点）：
$$\complement_U(A \cup B) = \complement_U A \cap \complement_U B,$$
$$\complement_U(A \cap B) = \complement_U A \cup \complement_U B.$$

口诀：**补集分配时，并变交，交变并**。

**Venn 图理解**：用 Venn 图可以直观验证这些法则。$A \cup B$ 是两个圆圈的全部区域；$A \cap B$ 是两圆重叠部分；$\complement_U A$ 是大框（全集）减去圆 $A$ 的部分。德摩根律用 Venn 图观察尤其清晰：$A \cup B$ 的补集（大框外面的部分）恰好等于"$A$ 的补集"与"$B$ 的补集"的交集。

## 三、推导：为什么这样定义

**交集为什么用"且"？** 数学中"且"（$\wedge$）表示两个条件都必须成立。$x \in A$ 且 $x \in B$，说的是 $x$ 同时满足两个集合的成员条件，这正是"公共元素"的含义。

**并集为什么用"或"（含两者）？** 现实中"A 班或 B 班的学生"通常包括同时参加两个班的人。数学里的"或"默认是相容或，避免人为排除两者都满足的情形，使得运算更自然（如果需要"恰好一个"，则用对称差 $A \triangle B$，高中不作要求）。

**为什么需要全集的概念？** 补集是"不属于 $A$ 的元素"，但"不属于 $A$"本身范围无穷大（所有不是 $A$ 元素的对象）。在数学上，我们必须明确讨论的范围，才能使补集成为一个确定的集合。全集就是这个"讨论范围"的约束框架。

**德摩根律的直觉**："既不在 $A$ 也不在 $B$ 里"等价于"不在 $A \cup B$ 里"——这就是 $\complement(A \cup B) = \complement A \cap \complement B$ 的直觉。反向：$A \cup B$ 的补集里的元素，对 $A$ 不属于、对 $B$ 也不属于，所以分别属于两个补集，交集即两条件同时满足。

## 四、典型应用

### 例 1　基础运算

**题目**：设 $A = \{1, 2, 3\}$，$B = \{2, 3, 4\}$，$U = \{1, 2, 3, 4, 5\}$，求 $A \cap B$、$A \cup B$、$\complement_U A$、$\complement_U(A \cap B)$。

**【思路】** 逐步定义代入。交集取公共元素，并集取全部元素（不重复），补集取全集中不属于该集合的元素。

**解**：

$$A \cap B = \{2, 3\}.$$

$$A \cup B = \{1, 2, 3, 4\}.$$

$$\complement_U A = \{4, 5\}.$$

$$\complement_U(A \cap B) = \complement_U\{2, 3\} = \{1, 4, 5\}.$$

验证德摩根律：$\complement_U A \cup \complement_U B = \{4, 5\} \cup \{1, 5\} = \{1, 4, 5\}$，与 $\complement_U(A \cap B) = \{1, 4, 5\}$ 吻合。$\checkmark$

### 例 2　含参集合的交集条件

**题目**：设 $A = \{x \mid 1 \leq x \leq a\}$，$B = \{x \mid x \leq 3\}$（$a > 1$），若 $A \cap B \neq \varnothing$，求 $a$ 的范围。

**【思路】** 用数轴表示两个集合，$A$ 是区间 $[1, a]$，$B$ 是区间 $(-\infty, 3]$。两集合有公共部分当且仅当它们在数轴上有交叠。

**解**：$A = [1, a]$，$B = (-\infty, 3]$。

由于 $A$ 最小值为 $1$，$B$ 最大值无穷，只要 $A$ 与 $B$ 有重叠，需要 $A$ 的左端点 $1 \leq 3$（显然），并且 $A$ 的区间存在（$a \geq 1$，题目已给 $a > 1$）。

事实上，$A \cap B = [1, \min(a, 3)]$。若 $a > 1$，则 $\min(a,3) \geq 1$（因为 $a > 1 > 0$），所以 $A \cap B$ 包含 $[1, \min(a,3)]$，非空。

因此只要 $a > 1$，就有 $A \cap B \neq \varnothing$。即 $a$ 的范围为 $a > 1$。

**补充**：若题改为 $A = [a, 2a-1]$，$B = [1, 3]$，$A \cap B \neq \varnothing$，则需要讨论区间端点关系，步骤更复杂——这类题型详见后续练习。

### 例 3　德摩根律应用

**题目**：设全集 $U = \mathbb{R}$，$A = \{x \mid x < -1 \text{ 或 } x > 2\}$，$B = \{x \mid -2 < x < 3\}$，求 $\complement_U(A \cup B)$。

**【思路】** 直接求 $A \cup B$，再取补集。也可以先分别求 $\complement_U A$ 和 $\complement_U B$，再用德摩根律取交集。两种方法应得相同答案，可互相验证。

**解（方法一：直接法）**：

$A = (-\infty, -1) \cup (2, +\infty)$，$B = (-2, 3)$。

$A \cup B$ = $A$ 的范围加上 $B$ 的范围（取并集）：

- $A \cup B = (-\infty, -1) \cup (-2, 3) \cup (2, +\infty) = (-\infty, 3)$？

更仔细：$(-\infty,-1)$ 覆盖到 $-1$ 以左；$(-2,3)$ 从 $-2$ 到 $3$；$(2,+\infty)$ 从 $2$ 以右到无穷。

合并：从 $-\infty$ 到 $3$，再加 $2$ 到 $+\infty$，即 $(-\infty, 3) \cup (2, +\infty) = (-\infty, +\infty) = \mathbb{R}$。

故 $A \cup B = \mathbb{R}$，$\complement_U(A \cup B) = \varnothing$。

**解（方法二：德摩根律）**：

$\complement_U A = [-1, 2]$，$\complement_U B = (-\infty, -2] \cup [3, +\infty)$。

$\complement_U(A \cup B) = \complement_U A \cap \complement_U B = [-1, 2] \cap ((-\infty, -2] \cup [3, +\infty)) = \varnothing$。

（$[-1,2]$ 与 $(-\infty,-2] \cup [3,+\infty)$ 无公共部分。）两种方法结果一致。$\checkmark$

## 五、易错点 & 反例

1. **$A \cap B = \varnothing$ 时不要漏写 $\varnothing$。**
   若 $A = \{1,3,5\}$，$B = \{2,4,6\}$，则 $A \cap B = \varnothing$，不能写成"无"或留空。结果必须明确写出。

2. **并集中元素不重复列举。**
   $A = \{1,2,3\}$，$B = \{2,3,4\}$，$A \cup B = \{1,2,3,4\}$，不能写 $\{1,2,3,2,3,4\}$——集合元素必须互异。

3. **补集必须指明全集。**
   没有全集，补集无意义。题目说"$A$ 的补集"时，必须先明确全集 $U$ 是什么，才能写出 $\complement_U A$。

4. **区间表示与不等式表示要清楚转换。**
   $A = \{x \mid -1 \leq x < 3\}$ 用区间表示是 $[-1, 3)$（含左端点，不含右端点）。区间端点的开闭直接影响交集、并集和补集的结果，不能粗心。

5. **德摩根律方向不能颠倒。**
   $\complement(A \cup B) = \complement A \cap \complement B$（并的补 = 补的交）；$\complement(A \cap B) = \complement A \cup \complement B$（交的补 = 补的并）。两者不能混用。

6. **$A \subseteq A \cup B$ 但 $A \not\supseteq A \cup B$（除非 $B \subseteq A$）。**
   新手常把 $A \cup B$ 误以为等于 $A$（漏掉 $B$ 中不属于 $A$ 的元素）。

## 六、思路自测题

**自测 1**　已知 $A = \{x \mid x^2 - 3x + 2 = 0\}$，$B = \{x \mid x^2 - 2x - 3 = 0\}$，$U = \mathbb{Z}$，分别求 $A \cap B$、$A \cup B$、$\complement_U A$、$\complement_U(A \cap B)$。

💡 提示：先解两个方程求出 $A$、$B$ 的元素，再逐步计算各集合运算。注意全集是整数集 $\mathbb{Z}$，"补集"在整数范围内有无穷多元素，通常用描述法表示。

**自测 2**　已知全集 $U = \{1,2,3,4,5,6\}$，$\complement_U A = \{2,4,6\}$，$\complement_U B = \{1,3,5\}$，求 $A \cap B$、$A \cup B$。

💡 提示：先由补集还原 $A$ 和 $B$：$A = U \setminus \complement_U A$，$B = U \setminus \complement_U B$。再计算交、并集。也可以用德摩根律：$A \cap B = \complement_U(\complement_U A \cup \complement_U B)$。

**自测 3**　设 $A = \{x \mid a \leq x \leq a+2\}$，$B = \{x \mid 1 \leq x \leq 4\}$，若 $A \cap B = A$，求实数 $a$ 的范围。

💡 提示：$A \cap B = A$ 等价于 $A \subseteq B$。即 $A$ 的左端点 $\geq B$ 的左端点，且 $A$ 的右端点 $\leq B$ 的右端点。注意还需考虑 $A = \varnothing$ 的情形（但这里 $a \leq a+2$ 恒成立，$A$ 不为空）。建立不等式组 $a \geq 1$ 且 $a + 2 \leq 4$，求解。

**自测 4**　设全集 $U = \mathbb{R}$，$A = \{x \mid x \leq -1 \text{ 或 } x \geq 3\}$，用 $\complement_U$ 和集合运算符号化简 $\complement_U(\complement_U A)$，并写出 $\complement_U A$ 的区间形式。

💡 提示：双补律 $\complement_U(\complement_U A) = A$，直接化简。$\complement_U A$ 是实数中不满足"$x \leq -1$ 或 $x \geq 3$"的部分，即 $-1 < x < 3$，区间形式为 $(-1, 3)$。
