# 第3章：自底向上的DP（制表法）

## 3.1 从记忆化到制表

上一章的记忆化搜索是**从问题出发，递归向下**。制表法反其道而行之：**从最小子问题出发，迭代向上**。

以斐波那契为例，比较两种思路：

```
记忆化搜索（Top-Down）：
  fib(5) → fib(4) → fib(3) → fib(2) → fib(1), fib(0)
  （从大到小，遇到 base case 后回溯）

制表法（Bottom-Up）：
  fib(0)=0, fib(1)=1 → fib(2)=1 → fib(3)=2 → fib(4)=3 → fib(5)=5
  （从小到大，逐步填充）
```

制表法实现：

```python
def fib(n):
    if n <= 1:
        return n
    
    # 创建 DP 表
    dp = [0] * (n + 1)
    
    # 填入 base case
    dp[0] = 0
    dp[1] = 1
    
    # 按正确顺序填表（确保依赖已计算）
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
```

---

## 3.2 制表法的四个步骤

**第一步：定义状态**
> `dp[i]` 或 `dp[i][j]` 精确表达什么含义？

**第二步：确定 base case**
> 哪些状态是已知的（无需计算）？

**第三步：确定状态转移方程**
> `dp[i]` 如何由更小的状态推导？

**第四步：确定填表顺序**
> 计算 `dp[i]` 时，它依赖的状态必须已经填好。

---

## 3.3 实战：最大子数组和（Kadane 算法）

**题目**（LeetCode 53）：
```
给定整数数组 nums，找出和最大的子数组，返回最大和。
nums = [-2,1,-3,4,-1,2,1,-5,4]  → 6（子数组 [4,-1,2,1]）
```

**状态定义**：`dp[i]` = 以 `nums[i]` **结尾**的最大子数组和

注意"以...结尾"的定义技巧——这保证了**无后效性**！

**转移方程**：
- 如果 `dp[i-1] > 0`：将前面的子数组接上来更好 → `dp[i] = dp[i-1] + nums[i]`
- 如果 `dp[i-1] <= 0`：前面是累赘，从 `nums[i]` 重新开始 → `dp[i] = nums[i]`

即：`dp[i] = max(dp[i-1] + nums[i], nums[i]) = max(dp[i-1], 0) + nums[i]`

```python
def max_subarray(nums):
    n = len(nums)
    dp = [0] * n
    
    # base case
    dp[0] = nums[0]
    
    # 填表
    for i in range(1, n):
        dp[i] = max(dp[i-1] + nums[i], nums[i])
    
    return max(dp)

# 测试
print(max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))  # 6
```

**时间复杂度**：O(n)  
**空间复杂度**：O(n)，可进一步优化到 O(1)（见第14章）

---

## 3.4 实战：打家劫舍

**题目**（LeetCode 198）：
```
街上有 n 间房，第 i 间有 nums[i] 元钱。
不能连续偷相邻的房间。求最多能偷多少钱？
nums = [2, 7, 9, 3, 1]  → 12（偷第1、3、5间：2+9+1=12）
```

**状态定义**：`dp[i]` = 前 i 间房子能偷到的最大金额

**转移方程**：
- 偷第 i 间：`dp[i-2] + nums[i]`（不能偷第 i-1 间）
- 不偷第 i 间：`dp[i-1]`
- 取最大值：`dp[i] = max(dp[i-1], dp[i-2] + nums[i])`

```python
def rob(nums):
    n = len(nums)
    if n == 1:
        return nums[0]
    
    dp = [0] * n
    
    # base case
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    # 填表
    for i in range(2, n):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    
    return dp[n-1]

print(rob([2, 7, 9, 3, 1]))  # 12
print(rob([1, 2, 3, 1]))     # 4
```

---

## 3.5 实战：最小路径和（二维DP）

**题目**（LeetCode 64）：
```
m×n 网格，每格有非负整数。
从左上角走到右下角，只能向右或向下。
找使路径上数字之和最小的路径。

1 3 1
1 5 1
4 2 1
最小路径和 = 7（1→3→1→1→1）
```

**状态定义**：`dp[i][j]` = 从 `(0,0)` 到 `(i,j)` 的最小路径和

**转移方程**：
```
dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
```

**base case**：
- 第一行：`dp[0][j] = dp[0][j-1] + grid[0][j]`（只能从左来）
- 第一列：`dp[i][0] = dp[i-1][0] + grid[i][0]`（只能从上来）

```python
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    
    # base case: 左上角
    dp[0][0] = grid[0][0]
    
    # 第一行
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    # 第一列
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    
    # 填表（从左到右，从上到下）
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
    
    return dp[m-1][n-1]

grid = [[1,3,1],[1,5,1],[4,2,1]]
print(min_path_sum(grid))  # 7
```

---

## 3.6 填表顺序的重要性

**错误示范**：如果从右下到左上填表会怎样？

```python
# 错误！dp[i][j] 依赖 dp[i-1][j] 和 dp[i][j-1]
# 从右下到左上时，这些依赖还没被计算
for i in range(m-1, -1, -1):  # 从下到上
    for j in range(n-1, -1, -1):  # 从右到左
        dp[i][j] = ...  # 此时 dp[i-1][j] 还没算！
```

**正确原则**：计算 `dp[i][j]` 之前，**所有它依赖的状态**必须已经计算完毕。

对于常见的二维DP，常见的合法填表顺序：
- 从左到右，从上到下（最常见）
- 从右到左，从下到上（某些区间DP）
- 按对角线填（区间DP）

---

## 3.7 状态转移方程的推导技巧

**技巧1："最后一步"思维**

> 考虑最终状态是如何转移过来的。

对于 `dp[i]`，思考：**dp[i] 由哪些前驱状态转移而来？**

```
打家劫舍中的 dp[i]：
- 偷第 i 间房  → 由 dp[i-2] 转移来
- 不偷第 i 间房 → 由 dp[i-1] 转移来
因此 dp[i] = max(dp[i-1], dp[i-2] + nums[i])
```

**技巧2："增加一个元素"思维**

> dp[i] 在 dp[i-1] 的基础上，加入第 i 个元素后会发生什么？

```
最大子数组和中：
- dp[i-1] 是以 nums[i-1] 结尾的最大子数组和
- 现在加入 nums[i]：
  - 如果 dp[i-1] > 0，把 nums[i] 接上去更大
  - 否则，从 nums[i] 重新开始
```

---

## 3.8 本章小结

制表法的通用模板：

```python
def dp_template(problem_input):
    # 1. 定义 DP 数组
    dp = [initial_value] * (size)  # 或二维数组
    
    # 2. 填入 base case
    dp[base_index] = base_value
    
    # 3. 按正确顺序填表
    for i in range(start, end):
        dp[i] = transition(dp[i-1], dp[i-2], ...)
    
    # 4. 返回答案
    return dp[answer_index]
```

**下一章：状态设计——DP 成功的关键**

---

## LeetCode 推荐题目

- [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/) ⭐
- [198. 打家劫舍](https://leetcode.cn/problems/house-robber/) ⭐
- [64. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/) ⭐⭐
- [221. 最大正方形](https://leetcode.cn/problems/maximal-square/) ⭐⭐
