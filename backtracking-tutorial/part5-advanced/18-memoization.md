# 第18章：记忆化搜索

## 18.1 什么时候回溯需要记忆化

**记忆化**：缓存已计算过的子问题结果，避免重复计算。

```
回溯需要记忆化的信号：
✓ 搜索过程中出现相同的子状态（重叠子问题）
✓ 状态可以被唯一描述（可哈希）
✓ 状态的结果不依赖于到达它的路径（无后效性）

不需要记忆化：
✗ 每个路径都是唯一的（全排列、子集枚举）
✗ 需要记录具体路径（只需计数/判断时可用）
✗ 状态空间太大，缓存本身占用太多内存
```

## 18.2 对比：纯回溯 vs 记忆化回溯

```python
# 问题：单词拆分（LeetCode 139）
# 判断字符串 s 是否可以由字典中的单词拼成

# ---- 版本1：纯回溯（超时）----
def wordBreak_backtrack(s, wordDict):
    word_set = set(wordDict)
    
    def backtrack(start):
        if start == len(s):
            return True
        for end in range(start + 1, len(s) + 1):
            if s[start:end] in word_set:
                if backtrack(end):
                    return True
        return False
    
    return backtrack(0)

# ---- 版本2：记忆化回溯（AC）----
def wordBreak_memo(s, wordDict):
    word_set = set(wordDict)
    memo = {}  # start → 是否可拆分
    
    def backtrack(start):
        if start in memo:
            return memo[start]  # 直接返回缓存结果
        if start == len(s):
            return True
        
        result = False
        for end in range(start + 1, len(s) + 1):
            if s[start:end] in word_set and backtrack(end):
                result = True
                break
        
        memo[start] = result
        return result
    
    return backtrack(0)

# ---- 版本3：等价的 DP ----
def wordBreak_dp(s, wordDict):
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]
```

## 18.3 记忆化 vs DP 的选择

```python
# 问题：零钱兑换（LeetCode 322）

# 记忆化版本（自顶向下）
from functools import lru_cache

def coinChange_memo(coins, amount):
    @lru_cache(maxsize=None)
    def dp(remaining):
        if remaining == 0:
            return 0
        if remaining < 0:
            return float('inf')
        
        min_coins = float('inf')
        for coin in coins:
            res = dp(remaining - coin)
            if res != float('inf'):
                min_coins = min(min_coins, res + 1)
        return min_coins
    
    result = dp(amount)
    return result if result != float('inf') else -1

# DP 版本（自底向上）
def coinChange_dp(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

# 两者等价，但记忆化更直观，DP 空间局部性更好
```

## 18.4 记忆化的状态设计

状态必须能**唯一描述**当前子问题：

```python
# 问题：不同路径（LeetCode 62）——经典记忆化

def uniquePaths_memo(m, n):
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def dp(r, c):
        if r == 0 or c == 0:
            return 1
        return dp(r-1, c) + dp(r, c-1)
    
    return dp(m-1, n-1)

# 问题：戳气球（LeetCode 312）——区间 DP + 记忆化

def maxCoins_memo(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def dp(left, right):
        """戳破 (left, right) 开区间内所有气球的最大金币"""
        if left + 1 == right:  # 开区间为空
            return 0
        
        max_coins = 0
        for i in range(left + 1, right):  # i 是最后一个被戳的气球
            coins = nums[left] * nums[i] * nums[right]
            max_coins = max(max_coins, dp(left, i) + coins + dp(i, right))
        return max_coins
    
    return dp(0, n-1)
```

## 18.5 含路径收集的记忆化

当需要枚举所有路径时，需要特殊处理（记忆化更复杂）：

```python
# 问题：单词拆分 II（LeetCode 140）——枚举所有拆分方案

def wordBreak_all(s, wordDict):
    """
    枚举所有方案时，记忆化缓存 start→[路径列表]
    注意：最坏情况下路径数指数增长，不一定能降低复杂度
    """
    word_set = set(wordDict)
    memo = {}
    
    def backtrack(start):
        if start in memo:
            return memo[start]
        if start == len(s):
            return ['']
        
        result = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set:
                for rest in backtrack(end):
                    if rest:
                        result.append(word + ' ' + rest)
                    else:
                        result.append(word)
        
        memo[start] = result
        return result
    
    return backtrack(0)

print(wordBreak_all("catsanddog", ["cat","cats","and","sand","dog"]))
# ['cats and dog', 'cat sand dog']
```

## 18.6 记忆化的复杂度分析

```python
"""
记忆化回溯的时间复杂度 = 不同状态数 × 每个状态的计算时间

例1：wordBreak
  状态：start（0~n），共 n+1 个
  每个状态计算：O(n²)（遍历所有切割点）
  总复杂度：O(n³)

例2：coinChange
  状态：remaining（0~amount），共 amount+1 个
  每个状态计算：O(len(coins))
  总复杂度：O(amount × len(coins))

例3：戳气球
  状态：(left, right)，共 O(n²) 个
  每个状态计算：O(n)
  总复杂度：O(n³)
"""

# 验证记忆化命中率
def wordBreak_with_stats(s, wordDict):
    word_set = set(wordDict)
    memo = {}
    hits = [0]
    misses = [0]
    
    def backtrack(start):
        if start in memo:
            hits[0] += 1
            return memo[start]
        misses[0] += 1
        
        if start == len(s):
            return True
        
        result = False
        for end in range(start + 1, len(s) + 1):
            if s[start:end] in word_set and backtrack(end):
                result = True
                break
        
        memo[start] = result
        return result
    
    result = backtrack(0)
    print(f"缓存命中：{hits[0]}，未命中：{misses[0]}，命中率：{hits[0]/(hits[0]+misses[0]+1e-9):.1%}")
    return result
```

## 18.7 Python lru_cache 技巧

```python
from functools import lru_cache

# 技巧1：对列表参数转换为元组（可哈希）
def solve_with_list_state(nums, target):
    @lru_cache(maxsize=None)
    def dp(idx, remaining, used_tuple):  # 用 tuple 代替 list
        if remaining == 0:
            return True
        if idx == len(nums):
            return False
        used = list(used_tuple)
        # ...
    
    return dp(0, target, tuple([False] * len(nums)))

# 技巧2：清除缓存（避免测试间污染）
def solve_multiple_cases(cases):
    @lru_cache(maxsize=None)
    def dp(n):
        if n <= 1:
            return n
        return dp(n-1) + dp(n-2)
    
    results = [dp(case) for case in cases]
    dp.cache_clear()  # 清除缓存
    return results

# 技巧3：使用字典手动实现（更灵活）
def dp_with_dict(n):
    memo = {}
    def solve(state):
        if state in memo:
            return memo[state]
        # ... 计算 ...
        memo[state] = result
        return result
    return solve(n)
```

## 小结

| 场景 | 是否用记忆化 | 原因 |
|------|------------|------|
| 枚举所有路径 | 不一定 | 路径数可能指数增长 |
| 判断是否存在解 | 推荐 | 状态数有限，有重叠子问题 |
| 计数（组合数）| 推荐 | 典型 DP 场景 |
| 求最优解 | 推荐 | 有最优子结构 |

## 练习

1. 为"骑士巡游"（马的哈密顿路径）添加记忆化，分析是否有效
2. 实现"目标和"（LeetCode 494）的三种解法：回溯、记忆化、DP，比较性能
3. 解释为什么全排列枚举不能用记忆化优化

---

**上一章：** [剪枝优化专题](17-pruning-optimization.md) | **下一章：** [回溯 vs 动态规划](19-backtracking-vs-dp.md)
