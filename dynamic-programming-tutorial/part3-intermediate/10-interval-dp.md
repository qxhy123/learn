# 第10章：区间DP

## 10.1 什么是区间DP

**区间DP** 是一类以"区间"为状态的动态规划，适用于问题可以被分解为对区间的操作：

- `dp[i][j]` 表示处理区间 `[i, j]` 的最优解
- 大区间由小区间合并而来
- 填表顺序：**按区间长度从小到大**

**适用问题特征**：
- 合并问题（石子合并、矩阵链乘）
- 分割问题（戳气球、括号问题）
- 回文问题（见第9章）

---

## 10.2 经典：矩阵链乘法

### 问题描述

```
有 n 个矩阵 A1, A2, ..., An 连续相乘。
矩阵 Ai 的大小为 p[i-1] × p[i]。
两个矩阵 (m×k) × (k×n) 的乘法需要 m×k×n 次运算。
求最少需要多少次乘法运算？

p = [10, 30, 5, 60]  → 矩阵：A1(10×30), A2(30×5), A3(5×60)
最优：(A1×A2)×A3 = 10*30*5 + 10*5*60 = 1500+3000 = 4500
```

### 状态定义

`dp[i][j]` = 矩阵 `Ai` 到 `Aj` 相乘所需的最少运算次数

### 转移方程

枚举最后一次分割点 k（在 k 处断开，先算左边再算右边，最后合并）：

```
dp[i][j] = min(dp[i][k] + dp[k+1][j] + p[i-1]*p[k]*p[j])
            for k in [i, j-1]
```

### 代码实现

```python
def matrix_chain_order(p):
    n = len(p) - 1  # 矩阵数量
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    
    # 按区间长度从小到大填表
    for length in range(2, n + 1):          # 区间长度
        for i in range(1, n - length + 2):  # 起点
            j = i + length - 1              # 终点
            dp[i][j] = float('inf')
            for k in range(i, j):           # 分割点
                cost = dp[i][k] + dp[k+1][j] + p[i-1] * p[k] * p[j]
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[1][n]

print(matrix_chain_order([10, 30, 5, 60]))   # 4500
print(matrix_chain_order([1, 2, 3, 4, 3]))   # 30
```

---

## 10.3 经典：石子合并

### 问题描述

```
n 堆石子排成一排，每次合并相邻两堆，代价为两堆石子的总重量。
求合并所有石子的最小总代价。

stones = [3, 2, 4, 1]
最优合并顺序：
  合并 [2,4] → [3, 6, 1]，代价6
  合并 [3,6] → [9, 1]，代价9
  合并 [9,1] → [10]，代价10
  总代价：25
```

### 前缀和优化

合并 `[i, j]` 的代价 = `sum(stones[i..j])` = `prefix[j+1] - prefix[i]`

```python
def stone_merge(stones):
    n = len(stones)
    
    # 前缀和
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + stones[i]
    
    # dp[i][j] = 合并 stones[i..j] 的最小代价
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            cost_to_merge = prefix[j + 1] - prefix[i]  # 本次合并的代价
            for k in range(i, j):
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + cost_to_merge)
    
    return dp[0][n-1]

print(stone_merge([3, 2, 4, 1]))  # 26（注：具体答案取决于合并方式）
```

---

## 10.4 进阶：戳气球（LeetCode 312）

### 问题描述

```
n 个气球，第 i 个气球有分值 nums[i]。
戳破气球 i 可以获得 nums[i-1] * nums[i] * nums[i+1] 的分数。
（边界外视为 nums[-1] = nums[n] = 1）
求戳破所有气球的最大分数。

nums = [3, 1, 5, 8]
最优：戳 1 → 3*1*5=15，戳 5 → 3*5*8=120，戳 3 → 1*3*8=24，戳 8 → 1*8*1=8
总分 = 167
```

### 关键思路：逆向思维

正向思维（先戳哪个）很难定义状态，因为戳破后邻居会变化。

**逆向思维**：考虑区间 `[i, j]` 中**最后一个被戳破的气球** k。

- 戳破 k 时，`[i, k-1]` 和 `[k+1, j]` 中的气球都已被戳破
- 所以戳破 k 得到的分数是 `nums[i-1] * nums[k] * nums[j+1]`

```python
def max_coins(nums):
    # 在两端加入边界值 1
    nums = [1] + nums + [1]
    n = len(nums)
    
    # dp[i][j] = 戳破所有 nums(i,j) 区间内的气球（不含端点i和j）的最大分数
    dp = [[0] * n for _ in range(n)]
    
    # 枚举区间长度（至少有1个气球在区间内，即j >= i+2）
    for length in range(2, n):        # 区间长度（j-i）
        for i in range(n - length):
            j = i + length
            for k in range(i + 1, j):  # k 是最后被戳破的气球
                dp[i][j] = max(
                    dp[i][j],
                    dp[i][k] + nums[i] * nums[k] * nums[j] + dp[k][j]
                )
    
    return dp[0][n-1]

print(max_coins([3, 1, 5, 8]))  # 167
print(max_coins([1, 5]))        # 10
```

**关键理解**：
- `dp[i][j]` 的区间是**开区间** `(i, j)`，不包含端点
- 因为 i 和 j 作为"边界"参与计算，而它们自己不在这个子问题里被戳

---

## 10.5 合法括号的数量（卡特兰数）

**题目**：n 对括号能组成多少种合法括号序列？

这不是区间DP，但用DP推导卡特兰数非常优雅：

```python
def catalan(n):
    # C(n) = C(0)*C(n-1) + C(1)*C(n-2) + ... + C(n-1)*C(0)
    # 含义：第一个左括号匹配第 k 个右括号，
    # 它把序列分成两个独立的子问题
    dp = [0] * (n + 1)
    dp[0] = 1  # 0对括号：1种（空序列）
    
    for i in range(1, n + 1):
        for k in range(i):
            dp[i] += dp[k] * dp[i-1-k]
    
    return dp[n]

for i in range(8):
    print(f"C({i}) = {catalan(i)}")
# 1, 1, 2, 5, 14, 42, 132, 429
```

---

## 10.6 奇怪的打印机（LeetCode 664）

```
打印机每次可以打印一段连续的相同字符（覆盖已打印内容）。
求打印给定字符串的最少打印次数。

s = "aaabbb" → 2
s = "aba"    → 2
```

```python
def strange_printer(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    
    # base case：单个字符打印1次
    for i in range(n):
        dp[i][i] = 1
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            # 先只打印 s[i..j-1]，再单独打印 s[j]
            dp[i][j] = dp[i][j-1] + 1
            # 如果 s[j] 与某个位置 k 相同，可以将 s[j] 的打印合并到 k
            for k in range(i, j):
                if s[k] == s[j]:
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j-1] if k+1 <= j-1 else dp[i][k])
    
    return dp[0][n-1]
```

---

## 10.7 区间DP总结

**通用模板**：

```python
def interval_dp(arr):
    n = len(arr)
    dp = [[0] * n for _ in range(n)]
    
    # base case：长度为1的区间
    for i in range(n):
        dp[i][i] = base_value(i)
    
    # 按区间长度从小到大填表（这是关键！）
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            # 枚举分割点
            for k in range(i, j):
                dp[i][j] = min(dp[i][j],
                               combine(dp[i][k], dp[k+1][j], cost(i, k, j)))
    
    return dp[0][n-1]
```

**三要素**：
1. 区间如何定义（包含端点？开区间？）
2. 分割点如何枚举
3. 合并的代价如何计算

---

## LeetCode 推荐题目

- [516. 最长回文子序列](https://leetcode.cn/problems/longest-palindromic-subsequence/) ⭐⭐
- [312. 戳气球](https://leetcode.cn/problems/burst-balloons/) ⭐⭐⭐
- [664. 奇怪的打印机](https://leetcode.cn/problems/strange-printer/) ⭐⭐⭐
- [375. 猜数字大小 II](https://leetcode.cn/problems/guess-number-higher-or-lower-ii/) ⭐⭐
- [1039. 多边形三角剖分的最低得分](https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/) ⭐⭐⭐
