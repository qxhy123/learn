# 第14章：空间优化——滚动数组与降维

## 14.1 为什么要优化空间

许多DP解法的时间复杂度已经是最优，但空间复杂度往往可以大幅压缩。

**空间优化的核心观察**：
> 如果 `dp[i]` 只依赖 `dp[i-1]`（或有限的前几行），那么不需要保存所有历史状态。

---

## 14.2 一维滚动数组

**场景**：`dp[i]` 只依赖 `dp[i-1]`。

```python
# 优化前：O(n) 空间
dp = [0] * (n + 1)
dp[0] = base
for i in range(1, n + 1):
    dp[i] = f(dp[i-1])
return dp[n]

# 优化后：O(1) 空间
prev = base
for i in range(1, n + 1):
    curr = f(prev)
    prev = curr
return prev
```

**示例：斐波那契**

```python
# O(n) → O(1)
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

**示例：打家劫舍**

```python
def rob(nums):
    prev2, prev1 = 0, 0
    for num in nums:
        curr = max(prev1, prev2 + num)
        prev2, prev1 = prev1, curr
    return prev1
```

---

## 14.3 二维DP的行压缩

**场景**：`dp[i][j]` 只依赖 `dp[i-1][...]`（上一行）。

```python
# 优化前：O(mn) 空间
dp = [[0] * (n + 1) for _ in range(m + 1)]
for i in range(1, m + 1):
    for j in range(1, n + 1):
        dp[i][j] = f(dp[i-1][j], dp[i][j-1])
return dp[m][n]

# 优化后：O(n) 空间（滚动行）
dp = [0] * (n + 1)
for i in range(1, m + 1):
    new_dp = [0] * (n + 1)
    for j in range(1, n + 1):
        new_dp[j] = f(dp[j], new_dp[j-1])  # dp[j]=上一行, new_dp[j-1]=本行左边
    dp = new_dp
return dp[n]
```

**示例：最长公共子序列**

```python
def lcs_space_optimized(text1, text2):
    m, n = len(text1), len(text2)
    # 只保留上一行
    prev = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    
    return prev[n]
```

---

## 14.4 原地更新（就地滚动）

有时可以不创建新数组，直接在原数组上滚动，但要**注意遍历方向**：

**0/1背包（从右到左）**：

```python
def knapsack_01(weights, values, W):
    dp = [0] * (W + 1)
    for w, v in zip(weights, values):
        # 从右到左：保证每个物品只用一次
        for j in range(W, w - 1, -1):
            dp[j] = max(dp[j], dp[j - w] + v)
    return dp[W]
```

**完全背包（从左到右）**：

```python
def knapsack_complete(weights, values, W):
    dp = [0] * (W + 1)
    for w, v in zip(weights, values):
        # 从左到右：允许同一物品使用多次
        for j in range(w, W + 1):
            dp[j] = max(dp[j], dp[j - w] + v)
    return dp[W]
```

---

## 14.5 LCS 的对角线滚动（只需 O(1) 额外空间）

当 LCS 需要进一步压缩时，可以用一个变量保存左上角值：

```python
def lcs_o1_space(text1, text2):
    m, n = len(text1), len(text2)
    if m < n:
        text1, text2 = text2, text1
        m, n = n, m
    
    dp = list(range(n + 1))  # 不对，LCS 的 base case 是全 0
    dp = [0] * (n + 1)
    
    for i in range(1, m + 1):
        prev = 0  # 保存 dp[i-1][j-1]（左上角）
        for j in range(1, n + 1):
            temp = dp[j]  # 保存更新前的 dp[j]（即下一轮的左上角）
            if text1[i-1] == text2[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = temp
    
    return dp[n]
```

---

## 14.6 编辑距离空间优化

```python
def edit_distance_optimized(word1, word2):
    m, n = len(word1), len(word2)
    # 保证 word2 是较短的（节省空间）
    if m < n:
        word1, word2 = word2, word1
        m, n = n, m
    
    dp = list(range(n + 1))  # dp[j] = edit_distance("", word2[:j])
    
    for i in range(1, m + 1):
        prev = dp[0]      # 保存左上角 dp[i-1][j-1]
        dp[0] = i         # dp[i][0] = i（前i字符转为空串需i次删除）
        
        for j in range(1, n + 1):
            temp = dp[j]  # 暂存，作为下一轮的左上角
            if word1[i-1] == word2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    
    return dp[n]

print(edit_distance_optimized("horse", "ros"))  # 3
```

**空间复杂度**：O(min(m, n))，从 O(mn) 压缩到 O(n)。

---

## 14.7 矩阵DP的对角线遍历

某些DP的转移同时依赖左方、上方和左上方，标准行压缩时需要注意保存左上角值（上面已演示）。

某些区间DP（如LCS、编辑距离）可以用**对角线遍历**来将空间压缩到 O(n)：

```python
def lcs_diagonal(s1, s2):
    m, n = len(s1), len(s2)
    # 沿对角线方向遍历，每次只需要当前对角线和上一条对角线
    # 细节较复杂，竞赛中少用，了解思路即可
    pass
```

---

## 14.8 空间优化的通用原则

**何时可以优化空间**：

| 依赖关系 | 优化方法 | 空间 |
|---------|---------|------|
| `dp[i]` 只依赖 `dp[i-1]` | 两个变量滚动 | O(1) |
| `dp[i][j]` 只依赖上一行 | 一维数组 + 遍历方向 | O(n) |
| `dp[i][j]` 依赖 `dp[i-1][j-1]`（左上） | 额外变量保存左上角 | O(n) |

**何时不能优化空间**：

- 需要回溯路径（如还原LCS、还原编辑操作）
- 状态依赖不规则（如树形DP）
- 答案不在边界位置

---

## 14.9 本章小结

空间优化是"锦上添花"，在**确认算法正确性**后再考虑。

常见陷阱：
- 0/1背包忘记从右到左 → 变成完全背包
- 保存左上角时用错变量 → 结果错误
- 需要回溯时过早优化了空间 → 无法还原路径

---

## LeetCode 推荐题目

- [72. 编辑距离](https://leetcode.cn/problems/edit-distance/) ⭐⭐⭐ （尝试用O(n)空间实现）
- [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/) ⭐⭐ （尝试用O(n)空间实现）
- [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/) ⭐⭐ （空间优化背包）
