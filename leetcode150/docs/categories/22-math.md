# 22 — Math（融合版）

> **难度**：★★☆☆☆
> **题数**：6
> **核心套路**：快速幂（分治）、整数平方根（二分）、数字特性分析（回文 / 尾零 / 共线）、进制模拟（Plus One）
> **本文件**：覆盖 math 6 题的算法套路总结 + 典型题精讲 + 自测

---

## 一例速记

> **快速幂（50 Pow(x,n)）**：$x^n = x^{n/2} \times x^{n/2}$，指数折半，$O(\log n)$ 次乘法；注意 $n < 0$ 时取倒数，$n$ 为奇数时额外乘一次 $x$
> **整数平方根（69 Sqrt(x)）**：二分查找 $[0, x]$ 内最大满足 $m^2 \le x$ 的 $m$；注意用 $m \le x/m$ 而非 $m^2 \le x$ 防止整数溢出（Python 不溢出可直接比较）
> **回文数（9 Palindrome Number）**：负数和末尾为 0 的数（0 除外）直接返回 False；反转数字的后半段，与前半段比较；避免转字符串
> **Plus One（66）**：从末位向前逐位进位，全 9 的情况退出循环后在首部追加 1
> **阶乘尾零（172 Trailing Zeroes）**：尾零 = 10 的因子数 = min(2 的因子数, 5 的因子数)；因 2 的因子远多于 5，只需统计 5 的因子：$\lfloor n/5 \rfloor + \lfloor n/25 \rfloor + \lfloor n/125 \rfloor + \cdots$
> **直线上最多点数（149 Max Points on a Line）**：枚举每对点确定斜率，斜率相同的点共线；用 `(dy/gcd, dx/gcd)` 的分数形式作 key 精确表示斜率，避免浮点误差
> **AI 关联**：快速幂用于模幂运算（密码学 / 同余运算）；数值稳定性（浮点精度）是深度学习训练中损失函数计算的核心问题；Softmax 减去 max 即为数值稳定性优化

---

## 思维路径还原

> "看到 **'50 Pow(x, n)'** → 快速幂（分治）：
> 若 `n == 0` 返回 1.0；若 `n < 0` 则 `x = 1/x, n = -n`；
> 若 `n % 2 == 0`：`half = myPow(x, n//2); return half * half`；
> 若 `n % 2 == 1`：`return x * myPow(x, n-1)`。
> 递归深度 $O(\log n)$，总乘法次数 $O(\log n)$。
> 迭代版：从低位到高位检查 n 的每一位，若当前位为 1 则累乘当前 x，x 每轮平方。
>
> 看到 **'69 Sqrt(x)'** → 二分：
> `lo, hi = 0, x`，`while lo <= hi: mid = (lo+hi)//2; if mid*mid <= x: ans=mid; lo=mid+1; else: hi=mid-1`。
> 最终 `ans` 即为答案。Python 中 int 不溢出，可以直接 `mid*mid`；其他语言用 `mid <= x // mid` 防溢出。
>
> 看到 **'9 Palindrome Number'** → 反转后半数字：
> 负数或（`x % 10 == 0 and x != 0`）→ False；
> 将 x 后半数字逐位放入 `rev`（`rev = rev*10 + x%10; x //= 10`），
> 直到 `x <= rev`（反转了一半）；
> 比较：偶数位 `x == rev`，奇数位 `x == rev // 10`（去掉中间数字）。
>
> 看到 **'172 Factorial Trailing Zeroes'** → 统计 5 的因子：
> $n!$ 中 5 的因子个数 = $\lfloor n/5 \rfloor + \lfloor n/25 \rfloor + \lfloor n/125 \rfloor + \cdots$；
> 循环：`result = 0; while n >= 5: n //= 5; result += n`。
>
> 看到 **'149 Max Points on a Line'** → 枚举 + 哈希斜率：
> 对每个点 i，用哈希表统计其他点与 i 的斜率频次，
> 斜率用约分后的 `(dy/gcd, dx/gcd)` 表示（处理垂直线 dx=0 时特殊处理），
> 每轮结果 = 哈希表中最大频次 + 1（+1 是点 i 本身）；
> 取所有轮次中的最大值，时间 $O(n^2)$。"

---

## 学习目标

- 掌握快速幂（50）：分治递归版和迭代位运算版，$O(\log n)$ 乘法
- 熟练二分搜索整数平方根（69），注意溢出保护
- 理解 172 尾零计数的数学原理（只需统计 5 的因子）
- 掌握 9 的回文数判断：反转后半段，避免转字符串
- 熟练 149 的斜率哈希技巧：用最简分数避免浮点误差
- 能对简单进位模拟（66）快速实现并处理全 9 边界

---

## 抽象成方法（标准模板代码）

### 套路 1：快速幂（分治递归 + 迭代）

适用题：50

```python
def my_pow_recursive(x: float, n: int) -> float:
    """
    50: 快速幂，时间 O(log n)，空间 O(log n)（递归栈）。
    """
    if n == 0:
        return 1.0
    if n < 0:
        return my_pow_recursive(1 / x, -n)
    half = my_pow_recursive(x, n // 2)
    if n % 2 == 0:
        return half * half
    else:
        return half * half * x


def my_pow_iterative(x: float, n: int) -> float:
    """
    50: 快速幂迭代版，时间 O(log n)，空间 O(1)。
    从 n 的二进制低位向高位逐位处理。
    """
    if n < 0:
        x, n = 1 / x, -n
    result = 1.0
    while n:
        if n & 1:           # 当前位为 1，将当前 x 累乘到结果
            result *= x
        x *= x              # x 平方，对应指数右移一位
        n >>= 1
    return result
```

> 迭代版：$n$ 的二进制从低位到高位，每位对应一个 $x^{2^k}$，若该位为 1 则累乘。$O(\log n)$ 次乘法，$O(1)$ 空间。

---

### 套路 2：二分搜索整数平方根

适用题：69

```python
def my_sqrt(x: int) -> int:
    """
    69: 计算 x 的整数平方根（向下取整）。时间 O(log x)，空间 O(1)。
    """
    if x < 2:
        return x
    lo, hi = 1, x // 2    # sqrt(x) <= x/2（x >= 2 时）
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid <= x // mid:        # mid*mid <= x（防溢出写法）
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


# 牛顿迭代法（更快收敛，面试进阶）
def my_sqrt_newton(x: int) -> int:
    """牛顿迭代：r = (r + x/r) / 2，收敛到 sqrt(x)。"""
    if x < 2:
        return x
    r = x
    while r * r > x:
        r = (r + x // r) // 2
    return r
```

---

### 套路 3：回文数（反转后半段）

适用题：9

```python
def is_palindrome(x: int) -> bool:
    """
    9: 判断整数是否为回文数，不转字符串。时间 O(log x)，空间 O(1)。
    """
    # 特判：负数不是回文；非零且末位为 0 不是回文
    if x < 0 or (x % 10 == 0 and x != 0):
        return False
    rev = 0
    while x > rev:         # 只反转后半段，直到 x <= rev
        rev = rev * 10 + x % 10
        x //= 10
    # 偶数位：x == rev；奇数位：x == rev // 10（跳过中间数字）
    return x == rev or x == rev // 10
```

---

### 套路 4：加一（进位模拟）

适用题：66

```python
from typing import List


def plus_one(digits: List[int]) -> List[int]:
    """
    66: 数组表示的十进制整数加 1。时间 O(n)，空间 O(1)（除全 9 情况）。
    """
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0
    # 走到这里说明全是 9（如 [9,9,9]），在最前补 1
    return [1] + digits
```

---

### 套路 5：阶乘尾零

适用题：172

```python
def trailing_zeroes(n: int) -> int:
    """
    172: n! 的尾零个数 = 5 的因子个数。时间 O(log n)，空间 O(1)。
    公式：floor(n/5) + floor(n/25) + floor(n/125) + ...
    """
    result = 0
    while n >= 5:
        n //= 5
        result += n
    return result
```

> 原理：尾零 = 因子 10 = min(因子 2 数, 因子 5 数)。因子 2 的数量远多于因子 5，所以只统计因子 5。$n!$ 中 5 的因子：每隔 5 个数贡献 1 个 5，每隔 25 个数再贡献 1 个（因为 25 = 5²），依此类推。

---

### 套路 6：直线上最多点数（斜率哈希）

适用题：149

```python
from math import gcd
from collections import defaultdict


def max_points(points: List[List[int]]) -> int:
    """
    149: 枚举每对点，用约分斜率作哈希 key。时间 O(n²)，空间 O(n)。
    """
    n = len(points)
    if n <= 2:
        return n
    ans = 2
    for i in range(n):
        slope_count: dict[tuple, int] = defaultdict(int)
        for j in range(i + 1, n):
            dy = points[j][1] - points[i][1]
            dx = points[j][0] - points[i][0]
            if dx == 0:
                key = (1, 0)       # 垂直线
            else:
                g = gcd(abs(dy), abs(dx))
                # 规范化：确保分母 dx 为正（统一斜率表示方向）
                if dx < 0:
                    dy, dx = -dy, -dx
                key = (dy // g, dx // g)
            slope_count[key] += 1
            ans = max(ans, slope_count[key] + 1)   # +1 是点 i 本身
    return ans
```

---

### 速查表

| 题型特征 | 套路 | 时间 | 空间 |
|---|---|---|---|
| $x^n$（n 可负）| 快速幂（分治/迭代）| $O(\log n)$ | $O(1)$（迭代）|
| 整数平方根（向下取整）| 二分搜索 / 牛顿迭代 | $O(\log x)$ | $O(1)$ |
| 判断回文数（不转字符串）| 反转后半段比较 | $O(\log x)$ | $O(1)$ |
| 数组表示的整数 +1 | 从末位进位模拟 | $O(n)$ | $O(1)$ |
| $n!$ 的尾零个数 | 统计 5 的因子 | $O(\log n)$ | $O(1)$ |
| 直线上最多点数 | 枚举 + 约分斜率哈希 | $O(n^2)$ | $O(n)$ |

---

## 方法变形（3 类）

### 变形 1：快速幂扩展

- **50**（实数幂）→ **基础模板**。
- **模幂**（`pow(x, n, mod)`）：在每次乘法后取模，用于密码学（RSA 解密）。
- **矩阵快速幂**：将标量乘法换成矩阵乘法，用于加速线性递推（如斐波那契 $O(\log n)$）。
- Python 内置 `pow(x, n, mod)` 即为快速幂 + 模运算，面试可直接用。

### 变形 2：整数平方根 vs 二分搜索

- **69**（Sqrt(x)）：经典二分，右边界可初始化为 `x//2`（因为 $\sqrt{x} \le x/2$ 对 $x \ge 4$ 成立）。
- **374**（Guess Number Higher or Lower，非本 category）：同样是二分框架，只是比较函数换为 `guess(mid)` API。
- **溢出保护**：C++/Java 中 `mid * mid` 可能溢出 int，改为 `mid <= x / mid`；Python 无溢出，直接写 `mid * mid <= x`。

### 变形 3：数论 / 精度

- **149 斜率精确表示**：浮点除法会有精度损失（如 `1/3` vs `2/6` 不等），用 `(dy//gcd, dx//gcd)` 的整数元组作 key 避免浮点误差。
- **172 质因数法推广**：$n!$ 中质数 $p$ 的因子个数 = $\lfloor n/p \rfloor + \lfloor n/p^2 \rfloor + \cdots$（勒让德公式）。
- **数值稳定性（AI 关联）**：Softmax 计算中先减去最大值（`x - max(x)`），避免 $e^x$ 溢出，是数值稳定性的典型应用；与本 category 的整数精度控制思路一脉相承。

---

## 思考路标（条件反射）

1. 看到 **"计算 x^n / 快速幂"** → 分治 $O(\log n)$；迭代版更省空间
2. 看到 **"整数平方根 / floor(sqrt)"** → 二分，右边界 `x//2`，用 `mid <= x//mid` 防溢出
3. 看到 **"回文数 / 不能转字符串"** → 反转后半段，与前半段比较
4. 看到 **"数组表示整数 +1"** → 从末位向前进位，全 9 情况在最前加 1
5. 看到 **"阶乘尾零 / 因子 5"** → 勒让德公式，循环除 5 累加
6. 看到 **"斜率 / 共线"** → 枚举 + 约分斜率哈希（`gcd` 化简），避免浮点
7. 看到 **"负指数"** → 快速幂：`n < 0` 时 `x = 1/x, n = -n`
8. 看到 **"溢出风险 / C++/Java"** → 改为除法比较 `mid <= x // mid`（Python 无此问题）
9. 看到 **"模幂 / 密码学"** → `pow(x, n, mod)`（Python 内置快速幂）

---

## 易错点

1. **50 负指数**：`n = -2147483648`（Python 最小整数，但无溢出）；`-n` 直接可得，无需担心；但 Java/C++ 中 INT_MIN 取反会溢出，需先转 long。
2. **50 n 为奇数的处理**：`half * half * x` 而非 `half * half + x`，乘法不是加法。
3. **69 搜索区间**：`hi = x // 2`（而非 `x`）可减少一半搜索范围；`x=1` 时 `x//2 = 0`，需特判 `x < 2` 直接返回 `x`。
4. **9 末尾为 0 的判断**：`x == 0` 是特例（回文），`x % 10 == 0 and x != 0` 才返回 False；否则 100、10 等非回文数无法被过滤。
5. **9 奇数位 vs 偶数位**：循环结束条件是 `x <= rev`；偶数位时 `x == rev`；奇数位时多转移了中间数字，应比较 `x == rev // 10`。
6. **66 原地修改**：Python 的 `list` 原地修改返回原数组（最常见情况）；全 9 时创建新数组 `[1] + digits`，注意不是 `digits.insert(0, 1)`（也可以，但 insert 是 $O(n)$）。
7. **149 斜率规范化方向**：`gcd` 只保证约分，但 `(-1, 2)` 和 `(1, -2)` 表示同一斜率；规范化方案：统一让分母 dx 为正（若 dx < 0 则 dy 和 dx 同时取反）。
8. **149 重复点**：若两点坐标完全相同（dy=dx=0），需单独计数（每对重复点都在任意直线上，应加到所有斜率的计数中）；题目保证点各异时此情况不存在。

---

## 典型应用例题

### 例 1：50. Pow(x, n)

**题目**：实现 `pow(x, n)` 即 $x^n$，`x` 为浮点数，`n` 为整数（可为负）。

**思路**：快速幂。$x^n = (x^{n/2})^2$（n 为偶数）或 $x \cdot (x^{(n-1)/2})^2$（n 为奇数）。$n < 0$ 时转化为 $(1/x)^{-n}$。迭代版枚举 n 的二进制位，从低位到高位，每次平方 x，若当前位为 1 则累乘。

**解**：

```python
# 参考：solutions/math/p050_powx_n.py
def myPow(x: float, n: int) -> float:
    if n < 0:
        x, n = 1 / x, -n
    result = 1.0
    while n:
        if n & 1:
            result *= x
        x *= x
        n >>= 1
    return result
```

**分析**：$O(\log n)$ 次乘法，$O(1)$ 空间。暴力循环 $O(n)$ 次乘法在 `n = 2^31` 时超时。

---

### 例 2：172. Factorial Trailing Zeroes

**题目**：给定整数 `n`，返回 `n!` 末尾 0 的个数。要求 $O(\log n)$ 时间。

**思路**：尾零 = 因子 10 的个数 = min(因子 2 的个数, 因子 5 的个数)。因子 2 远多于因子 5，故只统计 5 的因子：$\lfloor n/5 \rfloor$ 个数倍数含 1 个 5，$\lfloor n/25 \rfloor$ 个数倍数额外含 1 个 5（共 2 个），依此类推。

**解**：

```python
# 参考：solutions/math/p172_factorial_trailing_zeroes.py
def trailingZeroes(n: int) -> int:
    result = 0
    while n >= 5:
        n //= 5
        result += n
    return result
```

**分析**：循环执行 $O(\log_5 n)$ 次，时间 $O(\log n)$，空间 $O(1)$。例：$100! $ 的尾零 = $\lfloor 100/5 \rfloor + \lfloor 100/25 \rfloor = 20 + 4 = 24$。

---

### 例 3：149. Max Points on a Line

**题目**：给定平面上 `n` 个点，求最多有多少个点在同一条直线上。

**思路**：枚举基准点 i，对其余每个点 j 计算与 i 的斜率（用约分分数表示），用哈希表统计各斜率的点数。取最大值即为过 i 的直线上的最多点数。对所有 i 取最大值。

**解**：

```python
# 参考：solutions/math/p149_max_points_on_a_line.py
def maxPoints(points: List[List[int]]) -> int:
    n = len(points)
    if n <= 2:
        return n
    ans = 2
    for i in range(n):
        count: dict[tuple, int] = defaultdict(int)
        for j in range(i + 1, n):
            dy = points[j][1] - points[i][1]
            dx = points[j][0] - points[i][0]
            if dx == 0:
                key = (1, 0)
            else:
                g = gcd(abs(dy), abs(dx))
                if dx < 0:
                    dy, dx = -dy, -dx
                key = (dy // g, dx // g)
            count[key] += 1
            ans = max(ans, count[key] + 1)
    return ans
```

**分析**：双重循环 $O(n^2)$，每次计算 gcd $O(\log(\max |dy|, |dx|))$，整体 $O(n^2 \log C)$。`n <= 300` 的约束下可通过。

---

## 自测题

**自测 1**（9 Palindrome Number）—— `x=121` 返回 True，`x=-121` 返回 False，`x=10` 返回 False，`x=0` 返回 True。提示：负数直接 False；末尾为 0 且非 0 直接 False；反转后半段，循环终止条件 `x <= rev`；偶数位 `x==rev`，奇数位 `x==rev//10`。参考 `solutions/math/p009_palindrome_number.py`。

**自测 2**（50 Pow(x,n)）—— `x=2.0, n=10` 返回 1024.0；`x=2.1, n=3` 返回约 9.261；`x=2.0, n=-2` 返回 0.25。提示：迭代版，`n<0` 时 `x=1/x, n=-n`；while n 循环，奇数位乘 x，每轮 x 平方，n 右移。参考 `solutions/math/p050_powx_n.py`。

**自测 3**（66 Plus One）—— `digits=[1,2,3]` 返回 `[1,2,4]`；`digits=[4,3,2,1]` 返回 `[4,3,2,2]`；`digits=[9,9,9]` 返回 `[1,0,0,0]`。提示：从末位逐位检查，`< 9` 则 +1 直接返回，`== 9` 则置 0 继续，循环结束后前置 1。参考 `solutions/math/p066_plus_one.py`。

**自测 4**（69 Sqrt(x)）—— `x=4` 返回 2，`x=8` 返回 2，`x=0` 返回 0，`x=1` 返回 1。提示：二分 `[1, x//2]`，`mid <= x//mid` 时更新 ans 并收缩左边界，否则收缩右边界；`x<2` 时直接返回 x。参考 `solutions/math/p069_sqrtx.py`。

**自测 5**（172 Trailing Zeroes）—— `n=3` 返回 0，`n=5` 返回 1，`n=25` 返回 6。提示：`while n >= 5: n //= 5; result += n`；25 对应 5 + 1 = 6（`25//5=5`，`5//5=1`）。参考 `solutions/math/p172_factorial_trailing_zeroes.py`。

**自测 6**（149 Max Points on a Line）—— `points=[[1,1],[2,2],[3,3]]` 返回 3；`points=[[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]` 返回 4。提示：枚举基准点，约分斜率作 key，哈希表统计，结果加 1（基准点自身）。参考 `solutions/math/p149_max_points_on_a_line.py`。

---

## 题目全览（6 题）

| # | 题目 | 套路分类 | 难度 |
|---|---|---|---|
| 9 | Palindrome Number | 反转后半段比较 | Easy |
| 50 | Pow(x, n) | 快速幂（分治 / 迭代）| Medium |
| 66 | Plus One | 从末位进位模拟 | Easy |
| 69 | Sqrt(x) | 二分搜索 / 牛顿迭代 | Easy |
| 149 | Max Points on a Line | 枚举 + 约分斜率哈希 | Hard |
| 172 | Factorial Trailing Zeroes | 勒让德公式（统计 5 因子）| Medium |

---

## 融合版说明

| 段 | 来源 | 价值 |
|---|---|---|
| 一例速记 | 本文件 | 6 题套路一览 + AI（数值稳定性）关联 |
| 思维路径还原 | 本文件 | 6 道题的解题独白，含关键公式 |
| 抽象成方法 | 本文件 | 6 个标准模板（快速幂 / 二分平方根 / 回文数 / Plus One / 尾零 / 斜率哈希）+ 速查表 |
| 方法变形 | 本文件 | 3 类变体（快速幂扩展 / 二分变体 / 精度控制） |
| 思考路标 | 本文件 | 9 条题型识别条件反射 |
| 易错点 | 本文件 | 8 条高频踩坑（负指数 / 奇偶位 / 斜率方向 / 溢出） |
| 典型应用例题 | solutions/ | 3 道精讲（50、172、149），代码 + 分析 |
| 自测题 | leetcode | 6 题带提示，链接 solutions 文件 |
| 题目全览 | 本文件 | 6 题完整列表 |

---

> **跨 category 导航**：
> - 整数平方根 = 二分搜索变体 → `04-binary-search.md`
> - 位运算版快速幂 → `21-bit-manipulation.md`（迭代版快速幂逐位检测指数）
> - 回文字符串 → `01-array-string.md`（151 / 5 等）
> - Softmax 数值稳定性（减去 max）是深度学习框架（PyTorch `F.softmax`）的标准实现，与本 category 的数值精度控制一脉相承
