# 第8章：组合总和（进阶变体）

## 8.1 组合总和系列总览

| 题目 | 特点 | 关键技巧 |
|------|------|---------|
| 组合总和 I (LC39) | 可重复选，无重复元素 | `backtrack(i, ...)` |
| 组合总和 II (LC40) | 不可重复，含重复元素 | 排序+`i>start`去重 |
| 组合总和 III (LC216) | 1-9，固定大小k | 双重约束剪枝 |
| 组合总和 IV (LC377) | 顺序不同算不同组合 | DP（非回溯） |

## 8.2 组合总和 IV：顺序有关

**问题**：给定数组和目标值，顺序不同的组合算不同结果，求组合数。（LeetCode 377）

```
输入：nums=[1,2,3], target=4
输出：7
解释：(1,1,1,1)(1,1,2)(1,2,1)(1,3)(2,1,1)(2,2)(3,1)
```

**这不应该用回溯！** 有重叠子问题，应用 DP：

```python
def combination_sum4(nums, target):
    """
    DP：dp[i] = 总和为 i 的组合数
    时间 O(target × n)，空间 O(target)
    """
    dp = [0] * (target + 1)
    dp[0] = 1
    
    for i in range(1, target + 1):
        for num in nums:
            if i >= num:
                dp[i] += dp[i - num]
    
    return dp[target]

# 对比：回溯解法（TLE，有重叠子问题）
def combination_sum4_backtrack(nums, target):
    count = [0]
    def bt(remaining):
        if remaining == 0:
            count[0] += 1
            return
        for num in nums:
            if num <= remaining:
                bt(remaining - num)  # 每次从所有 nums 选，因为顺序有关
    bt(target)
    return count[0]
```

## 8.3 分组背包：将数组分成 k 组等和子集

**问题**：是否可以将数组分成 k 个等和的子集？（LeetCode 698）

```
输入：nums=[4,3,2,3,5,2,1], k=4
输出：True（[5],[1,4],[2,3],[2,3]）
```

```python
def can_partition_k_subsets(nums, k):
    """
    回溯：将每个元素分配到 k 个桶中
    剪枝：
    1. 总和不能被 k 整除 → False
    2. 最大值 > 目标值 → False
    3. 桶已满 → 跳过
    4. 相同大小的桶只尝试一个（去重）
    """
    total = sum(nums)
    if total % k != 0:
        return False
    
    target = total // k
    if max(nums) > target:
        return False
    
    nums.sort(reverse=True)  # 从大到小，提前剪枝
    buckets = [0] * k
    
    def backtrack(idx):
        if idx == len(nums):
            return all(b == target for b in buckets)
        
        seen = set()  # 相同容量的桶只尝试一次
        for j in range(k):
            if buckets[j] + nums[idx] > target:
                continue
            if buckets[j] in seen:
                continue
            seen.add(buckets[j])
            
            buckets[j] += nums[idx]
            if backtrack(idx + 1):
                return True
            buckets[j] -= nums[idx]
            
            if buckets[j] == 0:
                break  # 空桶都一样，只试一次
        
        return False
    
    return backtrack(0)
```

## 8.4 零钱兑换：最少硬币数（回溯 vs DP）

**问题**：给定硬币面值，凑出目标金额的最少硬币数。（LeetCode 322）

### 回溯（TLE，展示思路）

```python
def coin_change_backtrack(coins, amount):
    """回溯解法：超时，但直观展示搜索过程"""
    coins.sort(reverse=True)
    min_count = [float('inf')]
    
    def backtrack(remaining, count):
        if remaining == 0:
            min_count[0] = min(min_count[0], count)
            return
        if count >= min_count[0]:  # 剪枝：已不可能更优
            return
        
        for coin in coins:
            if coin <= remaining:
                backtrack(remaining - coin, count + 1)
    
    backtrack(amount, 0)
    return min_count[0] if min_count[0] != float('inf') else -1
```

### DP（正确解法）

```python
def coin_change_dp(coins, amount):
    """DP：O(amount × n)"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

## 8.5 回溯 + 贪心：跳跃游戏

```python
def can_jump(nums):
    """
    是否能从第0位跳到最后（贪心，非回溯）
    """
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)
    return True

def jump_game_all_paths(nums):
    """
    找出所有能到达最后的跳跃路径（回溯）
    """
    n = len(nums)
    result = []
    
    def backtrack(pos, path):
        if pos == n - 1:
            result.append(path[:])
            return
        
        for step in range(1, nums[pos] + 1):
            next_pos = pos + step
            if next_pos < n:
                path.append(next_pos)
                backtrack(next_pos, path)
                path.pop()
    
    backtrack(0, [0])
    return result
```

## 8.6 多目标优化：凑零钱的所有方案

```python
def coin_change_all(coins, amount):
    """
    找出所有凑法（硬币可重复使用）
    等价于：组合总和（coins作candidates, amount作target）
    """
    coins.sort()
    result = []
    
    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        
        for i in range(start, len(coins)):
            if coins[i] > remaining:
                break
            path.append(coins[i])
            backtrack(i, path, remaining - coins[i])
            path.pop()
    
    backtrack(0, [], amount)
    return result

print(coin_change_all([1, 2, 5], 5))
# [[1,1,1,1,1],[1,1,1,2],[1,2,2],[5]]
```

## 8.7 完整性测试

```python
def test_combination_sum():
    # 基础测试
    assert sorted(map(sorted, combination_sum([2,3,6,7], 7))) == [[2,2,3],[7]]
    assert sorted(map(sorted, combination_sum([2,3,5], 8))) == [[2,2,2,2],[2,3,3],[3,5]]
    
    # 边界测试
    assert combination_sum([1], 1) == [[1]]
    assert combination_sum([2], 1) == []
    
    # 含重复元素
    from collections import Counter
    res = combination_sum2([10,1,2,7,6,1,5], 8)
    # 验证无重复
    res_sorted = sorted([sorted(r) for r in res])
    assert len(res_sorted) == len(set(map(tuple, res_sorted)))
    
    print("All tests passed!")

test_combination_sum()
```

## 小结

| 问题特征 | 使用方法 |
|---------|---------|
| 求所有方案 | 回溯 |
| 顺序有关的计数 | DP（完全背包） |
| 求最少/最多 | DP（回溯会TLE）|
| 方案数 + 约束 | 回溯 + 剪枝 |

**判断用回溯还是 DP 的关键**：
- 需要**枚举所有方案** → 回溯
- 只需要**计数或最优值** → 考虑 DP
- 有**重叠子问题** → 一定要 DP 或记忆化

## 练习

1. 实现"完全背包"的回溯解法，并用记忆化优化（与 DP 等价）
2. 解决"路径总和 II"（LeetCode 113）：找出所有从根到叶路径和等于目标的路径
3. 对比 `combination_sum4` 的回溯（含记忆化）与 DP 的性能

---

**上一章：** [排列](07-permutations.md) | **下一章（Part 3）：** [N 皇后](../part3-board-problems/09-n-queens.md)
