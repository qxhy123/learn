# 第15章：单调队列优化DP

## 15.1 问题引入

考虑如下DP：

```
dp[i] = max(dp[j]) + cost(i)，其中 j ∈ [i-k, i-1]
```

朴素做法：对每个 i，枚举 j，时间复杂度 O(n²)。

**能否更快？**

注意到随着 i 增大，窗口 `[i-k, i-1]` 是一个**滑动窗口**——右端右移，左端也右移。如果能 O(1) 得到窗口内的最大值，总复杂度就是 O(n)。

**工具：单调队列（Monotone Deque）**

---

## 15.2 单调队列回顾

单调队列维护一个**单调递减**（或递增）的队列，支持：
- O(1) 查询窗口内最大值（队头）
- O(1) 滑出窗口的元素（从队头弹出）
- O(1) 加入新元素（从队尾维护单调性）

```python
from collections import deque

def sliding_window_max(nums, k):
    dq = deque()  # 存索引，队头是当前窗口最大值的索引
    result = []
    
    for i, num in enumerate(nums):
        # 弹出超出窗口的队头
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # 维护单调递减：弹出比 num 小的队尾
        while dq and nums[dq[-1]] < num:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

---

## 15.3 经典：跳跃游戏 VIII（单调队列优化DP）

**题目**（LeetCode 1696）：
```
给定数组 nums，从索引 0 开始跳跃。
从 i 跳到 j（i < j）的条件：
  - nums[i] <= nums[j] 且 j - i <= k，或
  - nums[i] > nums[j] 且 j - i > k
跳跃代价为 nums[j]，求到达最后一个位置的最小代价。
```

这类问题可以用单调栈/队列优化，此处展示更基础的例子。

---

## 15.4 经典：滑动窗口最大值DP

**题目**：

```
给定长度为 n 的数组，每次可以向前跳 1~k 步。
从位置 0 出发，到达位置 n-1 的最小代价（代价=当前位置的值）。
```

**状态**：`dp[i]` = 到达位置 i 的最小代价

**转移**：`dp[i] = min(dp[i-k], ..., dp[i-1]) + cost[i]`

**单调队列优化**：

```python
from collections import deque

def min_cost_jump(cost, k):
    n = len(cost)
    dp = [float('inf')] * n
    dp[0] = cost[0]
    
    dq = deque([0])  # 单调队列，存索引（单调递增dp值）
    
    for i in range(1, n):
        # 弹出超出窗口的队头
        while dq and dq[0] < i - k:
            dq.popleft()
        
        # 当前最小值
        dp[i] = dp[dq[0]] + cost[i]
        
        # 维护单调队列（递增）
        while dq and dp[dq[-1]] >= dp[i]:
            dq.pop()
        dq.append(i)
    
    return dp[n-1]
```

---

## 15.5 经典：分割数组最大值（LeetCode 410）

**题目**：将数组分成 m 个非空连续子数组，使得各子数组和的最大值最小。

这是一道**二分+贪心**的题，但也可以用DP（为单调队列优化铺垫）：

```python
def split_array(nums, m):
    # dp[i][j] = 将 nums[:i] 分成 j 段的最大子数组和的最小值
    # dp[i][j] = min(max(dp[k][j-1], sum(nums[k:i])))  for k in [j-1, i)
    
    n = len(nums)
    prefix = [0] * (n + 1)
    for i, x in enumerate(nums):
        prefix[i+1] = prefix[i] + x
    
    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    
    for j in range(1, m + 1):
        for i in range(j, n + 1):
            for k in range(j-1, i):
                segment_sum = prefix[i] - prefix[k]
                dp[i][j] = min(dp[i][j], max(dp[k][j-1], segment_sum))
    
    return dp[n][m]
```

---

## 15.6 最大矩形（LeetCode 85）——单调栈辅助DP

**题目**：在0/1矩阵中找最大全1矩形的面积。

```python
def maximal_rectangle(matrix):
    if not matrix: return 0
    m, n = len(matrix), len(matrix[0])
    
    # heights[j] = 以当前行为底，第j列连续1的高度
    heights = [0] * n
    max_area = 0
    
    for i in range(m):
        for j in range(n):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
        
        # 对每一行的 heights 数组，求柱状图中最大矩形（单调栈）
        max_area = max(max_area, largest_rectangle(heights))
    
    return max_area

def largest_rectangle(heights):
    stack = [-1]
    max_area = 0
    heights = heights + [0]  # 哨兵
    
    for i, h in enumerate(heights):
        while stack[-1] != -1 and heights[stack[-1]] >= h:
            height = heights[stack.pop()]
            width = i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    
    return max_area
```

---

## 15.7 单调队列优化多重背包

在第6章中，多重背包的单调队列优化是最难的部分：

**题目**：每件物品最多用 cnt[i] 次，用单调队列将 O(nW·Σcnt) 优化到 O(nW)。

**关键思路**：对于每种物品（重量 w，价值 v，次数 c），按模 w 的余数分组，在每组内用单调队列求滑动窗口最大值。

```python
from collections import deque

def knapsack_bounded_monotone(weights, values, counts, W):
    dp = [0] * (W + 1)
    
    for w, v, c in zip(weights, values, counts):
        new_dp = dp[:]
        
        # 按余数分组：余数 r = 0, 1, ..., w-1
        for r in range(w):
            # 处理余数为 r 的所有容量：r, r+w, r+2w, ...
            indices = list(range(r, W + 1, w))
            
            # 将问题转化为滑动窗口（窗口大小 c+1）
            # 辅助数组：aux[k] = dp[r + k*w] - k*v
            dq = deque()  # 存 (辅助值, 原始索引)
            
            for k, cap in enumerate(indices):
                # 加入当前元素
                aux_val = dp[cap] - k * v
                while dq and dq[-1][0] <= aux_val:
                    dq.pop()
                dq.append((aux_val, k))
                
                # 弹出超出窗口的元素（窗口大小 c）
                while dq[0][1] < k - c:
                    dq.popleft()
                
                # 更新 new_dp
                new_dp[cap] = max(new_dp[cap], dq[0][0] + k * v)
        
        dp = new_dp
    
    return dp[W]
```

---

## 15.8 本章小结

**单调队列优化DP的适用条件**：

1. 转移形式为 `dp[i] = opt(dp[j]) + cost(i)`，其中 j 在一个**滑动窗口**内
2. `opt` 是 min 或 max
3. 窗口的左右边界都单调右移

**优化效果**：O(n²) → O(n)

---

## LeetCode 推荐题目

- [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/) ⭐⭐（基础）
- [1696. 跳跃游戏 VI](https://leetcode.cn/problems/jump-game-vi/) ⭐⭐⭐
- [85. 最大矩形](https://leetcode.cn/problems/maximal-rectangle/) ⭐⭐⭐
- [918. 环形子数组的最大和](https://leetcode.cn/problems/maximum-sum-circular-subarray/) ⭐⭐⭐
