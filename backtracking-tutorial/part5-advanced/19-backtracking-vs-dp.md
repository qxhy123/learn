# 第19章：回溯 vs 动态规划

## 19.1 两者的本质区别

```
回溯（Backtracking）：
  - 搜索所有可能的路径
  - 遇到无效状态立刻放弃（剪枝）
  - 目标：枚举解/找到解/验证解
  - 时间复杂度通常为指数级（但剪枝后实际很快）

动态规划（Dynamic Programming）：
  - 利用最优子结构，避免重复计算
  - 每个状态只计算一次
  - 目标：求最值/计数
  - 时间复杂度通常为多项式级

关键判断：
  问题是否有"重叠子问题"？
    是 → 考虑 DP 或记忆化
    否 → 回溯（每条路径都是独立的）
  问题是否要求枚举所有解？
    是 → 只能回溯
    否 → 两者都可能适用，DP 更高效
```

## 19.2 用同一问题演示两种方法

### 案例一：硬币找零

```python
# 回溯：找出所有凑法（枚举）
def coin_change_all_ways(coins, amount):
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
            backtrack(i, path, remaining - coins[i])  # i 允许重复选
            path.pop()
    
    backtrack(0, [], amount)
    return result

# DP：最少硬币数（最优值）
def coin_change_min(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

# DP：凑法总数（计数）
def coin_change_count(coins, amount):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    return dp[amount]

# 对比
print(coin_change_all_ways([1, 2, 5], 5))
# [[1,1,1,1,1],[1,1,1,2],[1,2,2],[5]]  ← 4种
print(coin_change_count([1, 2, 5], 5))  # 4
print(coin_change_min([1, 2, 5], 5))    # 1（一枚5元）
```

### 案例二：目标和（LeetCode 494）

```python
# 回溯：枚举所有方案（+ / -）
def findTargetSumWays_backtrack(nums, target):
    count = [0]
    
    def backtrack(idx, current_sum):
        if idx == len(nums):
            if current_sum == target:
                count[0] += 1
            return
        backtrack(idx + 1, current_sum + nums[idx])
        backtrack(idx + 1, current_sum - nums[idx])
    
    backtrack(0, 0)
    return count[0]

# 记忆化：O(n × sum) 时间
from functools import lru_cache
def findTargetSumWays_memo(nums, target):
    @lru_cache(maxsize=None)
    def dp(idx, current_sum):
        if idx == len(nums):
            return 1 if current_sum == target else 0
        return dp(idx+1, current_sum+nums[idx]) + dp(idx+1, current_sum-nums[idx])
    
    return dp(0, 0)

# DP（背包转化）：O(n × sum) 时间，O(sum) 空间
def findTargetSumWays_dp(nums, target):
    total = sum(nums)
    if abs(target) > total or (total + target) % 2 != 0:
        return 0
    
    # 转化：选 P 个加，剩余减，则 P - (total-P) = target，P = (total+target)/2
    pos_sum = (total + target) // 2
    dp = [0] * (pos_sum + 1)
    dp[0] = 1
    
    for num in nums:
        for j in range(pos_sum, num - 1, -1):
            dp[j] += dp[j - num]
    
    return dp[pos_sum]

# 性能对比
import time
nums = [1] * 20
target = 0

for name, func in [("回溯", findTargetSumWays_backtrack),
                    ("记忆化", findTargetSumWays_memo),
                    ("DP", findTargetSumWays_dp)]:
    t = time.time()
    result = func(nums[:], target)
    print(f"{name}: {result}, 用时 {time.time()-t:.4f}s")
```

## 19.3 判断使用哪种方法的决策树

```
               问题需要什么？
              /              \
        枚举所有解          只需最值/计数
            |                    |
          回溯             有重叠子问题？
                           /          \
                          是            否
                          |             |
                    DP/记忆化          回溯
                    
特殊情况：
- 问题规模小（n≤20）：回溯足够
- 有重叠子问题但需要所有路径：记忆化+回溯
- 问题有最优子结构：优先 DP
```

## 19.4 典型问题归类

```python
"""
只能用回溯（必须枚举所有解）：
  - 全排列 (LC 46/47)
  - 子集枚举 (LC 78/90)
  - N 皇后 (LC 51)
  - 括号生成 (LC 22)
  - 单词搜索 (LC 79/212)

只能用 DP（最优子结构 + 计数）：
  - 最长公共子序列 (LC 1143)
  - 编辑距离 (LC 72)
  - 最长递增子序列 (LC 300)
  - 矩阵链乘法

两者都可以（DP 更高效）：
  - 零钱兑换 (LC 322) — 最少数量用 DP
  - 爬楼梯 (LC 70) — 计数用 DP
  - 单词拆分 (LC 139) — 判断用 DP

容易混淆的情况：
  - 组合总和 IV (LC 377) — 顺序有关，用 DP（完全背包）
  - 解码方法 (LC 91) — 计数 + 重叠子问题，用 DP
  - 目标和 (LC 494) — 计数，DP 优化
"""
```

## 19.5 回溯转 DP 的通用方法

```python
# 回溯框架
def backtrack_template(state):
    if is_terminal(state):
        return base_value(state)
    
    result = initial_value
    for choice in get_choices(state):
        result = combine(result, backtrack_template(next_state(state, choice)))
    return result

# 对应的 DP（记忆化）框架
from functools import lru_cache

def dp_template(initial_state):
    @lru_cache(maxsize=None)
    def dp(state):  # state 必须可哈希
        if is_terminal(state):
            return base_value(state)
        
        result = initial_value
        for choice in get_choices(state):
            result = combine(result, dp(next_state(state, choice)))
        return result
    
    return dp(initial_state)

# 示例：爬楼梯（LeetCode 70）
def climbStairs_backtrack(n):
    """回溯版（TLE）"""
    if n <= 1:
        return 1
    return climbStairs_backtrack(n-1) + climbStairs_backtrack(n-2)

@lru_cache(maxsize=None)
def climbStairs_memo(n):
    """记忆化版（AC）"""
    if n <= 1:
        return 1
    return climbStairs_memo(n-1) + climbStairs_memo(n-2)

def climbStairs_dp(n):
    """DP 版（最优）"""
    if n <= 1:
        return 1
    a, b = 1, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b
```

## 19.6 思维误区解析

```python
"""
误区1："有回溯就用回溯，有 DP 就用 DP"
  正解：根据问题特征选择，而不是凭感觉

误区2："回溯一定比 DP 慢"
  反例：N 皇后需要枚举所有解，DP 根本无法表达
  
误区3："记忆化 = DP"
  区别：记忆化是自顶向下，DP 是自底向上
        记忆化只计算被调用的状态，DP 计算所有状态
        记忆化调用栈更深，但代码更直观

误区4："能用 DP 就一定要用 DP"
  反例：当问题规模小时，回溯代码更简单，且性能足够

误区5："回溯一定有'回退'操作"
  澄清：当状态是不可变的（如传参而非修改全局），
        无需显式回退（如 backtrack(state + choice)）
"""
```

## 小结

| 维度 | 回溯 | 动态规划 |
|------|------|---------|
| 核心思想 | 搜索 + 剪枝 | 最优子结构 + 无后效性 |
| 目标 | 枚举解/找解 | 最优值/计数 |
| 时间复杂度 | 指数级（剪枝后更好）| 多项式级 |
| 代码复杂度 | 较简单，直观 | 状态定义需技巧 |
| 适用规模 | n ≤ 20-30 | n 可以很大 |

## 练习

1. 给定 `wordBreak_all`（返回所有拆分方案），分析其时间复杂度与记忆化的收益
2. 将"0-1 背包问题"用回溯和 DP 分别实现，对比 n=30 时的性能
3. 找到一个"表面上看起来像 DP，实际上只能用回溯"的问题并解释原因

---

**上一章：** [记忆化搜索](18-memoization.md) | **下一章：** [竞赛题精选](20-competitive-programming.md)
