# 第5章：线性DP——从斐波那契到复杂的线性模型

## 5.1 什么是线性DP

**线性DP** 是指状态按照线性顺序（一维或二维数组）排列，转移方程只涉及相邻或附近的状态。

特征：
- 状态通常是 `dp[i]` 或 `dp[i][j]`
- 转移方向单一（如从左到右、从上到下）
- 是所有DP的基础

---

## 5.2 斐波那契数列家族

### 基础版：斐波那契（LeetCode 509）

```python
def fib(n):
    if n <= 1: return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

### 变体1：爬楼梯（LeetCode 70）

每次可以走1步或2步，爬 n 级台阶有多少种方式？

```python
def climb_stairs(n):
    if n <= 2: return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b

# 本质上就是 fib(n+1)
```

### 变体2：泰波那契（LeetCode 1137）

```
T(0) = 0, T(1) = 1, T(2) = 1
T(n) = T(n-1) + T(n-2) + T(n-3)
```

```python
def tribonacci(n):
    if n == 0: return 0
    if n <= 2: return 1
    a, b, c = 0, 1, 1
    for _ in range(3, n + 1):
        a, b, c = b, c, a + b + c
    return c
```

### 变体3：爬楼梯加权（LeetCode 746）

```
使用最小花费爬到楼顶，cost[i] 是从台阶 i 起跳的花费。
可以从台阶 0 或 1 出发，每次可以爬 1 或 2 个台阶。
```

```python
def min_cost_climbing_stairs(cost):
    n = len(cost)
    dp = [0] * (n + 1)  # dp[i] = 到达台阶i的最小花费
    
    # base case: 前两级台阶可以免费到达
    dp[0] = dp[1] = 0
    
    for i in range(2, n + 1):
        dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])
    
    return dp[n]

print(min_cost_climbing_stairs([10, 15, 20]))  # 15
print(min_cost_climbing_stairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]))  # 6
```

---

## 5.3 打家劫舍系列

### 基础版（LeetCode 198）

```python
def rob(nums):
    n = len(nums)
    if n == 1: return nums[0]
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, n):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    return dp[-1]
```

### 变体1：环形街道（LeetCode 213）

```
房间排成环形，首尾相邻，不能同时偷首尾两间房。
```

**关键思路**：环形 = 两次线性

- 情形1：不偷第一间 → 在 `nums[1:]` 上做打家劫舍
- 情形2：不偷最后一间 → 在 `nums[:-1]` 上做打家劫舍

```python
def rob_circle(nums):
    def rob_linear(arr):
        if not arr: return 0
        if len(arr) == 1: return arr[0]
        dp = [0] * len(arr)
        dp[0] = arr[0]
        dp[1] = max(arr[0], arr[1])
        for i in range(2, len(arr)):
            dp[i] = max(dp[i-1], dp[i-2] + arr[i])
        return dp[-1]
    
    if len(nums) == 1: return nums[0]
    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))

print(rob_circle([2, 3, 2]))   # 3
print(rob_circle([1, 2, 3, 1]))  # 4
```

### 变体2：二叉树上偷（LeetCode 337）

```
房间排成二叉树，不能同时偷父子节点。
```

```python
from functools import lru_cache

def rob_tree(root):
    @lru_cache(maxsize=None)
    def dp(node):
        if not node:
            return (0, 0)  # (不偷该节点的最大值, 偷该节点的最大值)
        
        left_no, left_yes   = dp(node.left)
        right_no, right_yes = dp(node.right)
        
        # 不偷当前节点：左右子节点可偷可不偷
        no_rob  = max(left_no, left_yes) + max(right_no, right_yes)
        # 偷当前节点：左右子节点都不能偷
        yes_rob = node.val + left_no + right_no
        
        return (no_rob, yes_rob)
    
    return max(dp(root))
```

---

## 5.4 最大正方形（二维线性DP）

**题目**（LeetCode 221）：
```
在矩阵中找出只包含 '1' 的最大正方形。
```

**状态**：`dp[i][j]` = 以 `(i, j)` 为**右下角**的最大正方形的边长

**转移方程**（最精妙的DP之一）：
```
if matrix[i][j] == '1':
    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
```

**为什么取 min？**

```
2 2
2 dp[i][j] = ?

dp[i-1][j] = 2  → 上方有 2×2 的正方形
dp[i][j-1] = 2  → 左方有 2×2 的正方形
dp[i-1][j-1] = 2 → 左上有 2×2 的正方形

以 (i,j) 为右下角的正方形，受制于三个方向的瓶颈
min(2,2,2) + 1 = 3  → 可以形成 3×3 的正方形
```

```python
def maximal_square(matrix):
    if not matrix: return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    max_side = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])
    
    return max_side * max_side
```

---

## 5.5 接雨水（线性DP思维）

**题目**（LeetCode 42）：
```
给定柱状图，计算能接多少雨水。
height = [0,1,0,2,1,0,1,3,2,1,2,1]  → 6
```

**DP思路**：对每个位置 i，能接的水量 = `min(左侧最高, 右侧最高) - height[i]`

```python
def trap(height):
    n = len(height)
    if n == 0: return 0
    
    # 预计算每个位置左侧和右侧的最高柱子
    left_max  = [0] * n
    right_max = [0] * n
    
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i-1], height[i])
    
    right_max[n-1] = height[n-1]
    for i in range(n-2, -1, -1):
        right_max[i] = max(right_max[i+1], height[i])
    
    # 计算每个位置能接的水
    water = 0
    for i in range(n):
        water += min(left_max[i], right_max[i]) - height[i]
    
    return water

print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))  # 6
```

---

## 5.6 解码方法（字符串线性DP）

**题目**（LeetCode 91）：
```
数字编码：'A'→1, 'B'→2, ..., 'Z'→26
给定数字字符串，有多少种解码方式？
"226" → 3（"BZ", "VF", "BBF"）
```

```python
def num_decodings(s):
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1  # 空字符串，1种解码方式
    dp[1] = 0 if s[0] == '0' else 1
    
    for i in range(2, n + 1):
        # 单独解码 s[i-1]
        one_digit = int(s[i-1])
        if one_digit != 0:
            dp[i] += dp[i-1]
        
        # 与前一位一起解码 s[i-2:i]
        two_digits = int(s[i-2:i])
        if 10 <= two_digits <= 26:
            dp[i] += dp[i-2]
    
    return dp[n]

print(num_decodings("226"))  # 3
print(num_decodings("06"))   # 0
```

---

## 5.7 本章总结：线性DP的模式识别

| 题目类型 | 状态定义技巧 | 代表题目 |
|----------|-------------|----------|
| 计数/方案数 | `dp[i]` = 到达 i 的方案数 | 爬楼梯、解码方法 |
| 最大/最小值 | `dp[i]` = 到达 i 的最优值 | 最小花费爬楼梯 |
| 以某元素结尾 | `dp[i]` = 以第 i 个元素结尾的... | 最大子数组 |
| 选或不选 | `dp[i][0/1]` = 选/不选第 i 个的最优 | 打家劫舍 |
| 二维网格 | `dp[i][j]` = 到达 `(i,j)` 的... | 最小路径和 |

---

## LeetCode 推荐题目

**基础**
- [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/) ⭐
- [198. 打家劫舍](https://leetcode.cn/problems/house-robber/) ⭐
- [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/) ⭐

**进阶**
- [213. 打家劫舍 II](https://leetcode.cn/problems/house-robber-ii/) ⭐⭐
- [221. 最大正方形](https://leetcode.cn/problems/maximal-square/) ⭐⭐
- [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/) ⭐⭐⭐
- [91. 解码方法](https://leetcode.cn/problems/decode-ways/) ⭐⭐
