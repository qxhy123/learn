# 第6章：背包问题全家桶

## 6.1 背包问题的地位

背包问题是动态规划中最重要的模型之一，几乎涵盖了所有"选择与约束"类型的DP。掌握背包系列，就掌握了大量DP问题的本质。

背包家族成员：
- **0/1背包**：每件物品最多选1次
- **完全背包**：每件物品可选无限次
- **多重背包**：每件物品最多选 k 次
- **分组背包**：物品分组，每组最多选1件

---

## 6.2 0/1 背包

### 问题定义

```
有 n 件物品，背包容量为 W。
第 i 件物品重量 weight[i]，价值 value[i]。
每件物品只能选0次或1次，求能装入的最大价值。
```

### 状态定义

`dp[i][j]` = 考虑前 i 件物品，背包容量为 j 时的最大价值

### 转移方程

对于第 i 件物品，有两种选择：
- **不选**：`dp[i][j] = dp[i-1][j]`
- **选**（前提：`j >= weight[i]`）：`dp[i][j] = dp[i-1][j - weight[i]] + value[i]`

取最大值：
```
dp[i][j] = max(dp[i-1][j], dp[i-1][j - weight[i]] + value[i])
```

### 代码实现

```python
def knapsack_01(weights, values, W):
    n = len(weights)
    # dp[i][j]: 前i件物品，容量j的最大价值
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        w, v = weights[i-1], values[i-1]
        for j in range(W + 1):
            # 不选第i件
            dp[i][j] = dp[i-1][j]
            # 选第i件（容量够的话）
            if j >= w:
                dp[i][j] = max(dp[i][j], dp[i-1][j - w] + v)
    
    return dp[n][W]

weights = [2, 3, 4, 5]
values  = [3, 4, 5, 6]
W = 8
print(knapsack_01(weights, values, W))  # 10
```

### 空间优化：滚动数组（重要！）

注意 `dp[i][j]` 只依赖 `dp[i-1][...]`，所以可以用**一维数组**：

```python
def knapsack_01_optimized(weights, values, W):
    n = len(weights)
    dp = [0] * (W + 1)  # dp[j]: 容量j时的最大价值
    
    for i in range(n):
        # ⚠️ 必须从右向左遍历！
        # 原因：确保每件物品只被选一次
        # 若从左到右，dp[j-w] 可能已用本轮更新的值（相当于选了多次）
        for j in range(W, weights[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    
    return dp[W]

print(knapsack_01_optimized([2,3,4,5], [3,4,5,6], 8))  # 10
```

**为什么0/1背包要从右向左？**

```
设 weight[i] = 2，处理前 dp = [0, 0, 3, 3, 5, ...]

从左到右（错误）：
j=2: dp[2] = max(dp[2], dp[0]+3) = 3  ← dp[0] 是本轮未被修改的
j=4: dp[4] = max(dp[4], dp[2]+3) = 6  ← dp[2]=3 是本轮刚修改的！相当于选了两次

从右到左（正确）：
j=4: dp[4] = max(dp[4], dp[2]+3) = 5  ← dp[2] 还是上一轮的值
j=2: dp[2] = max(dp[2], dp[0]+3) = 3  ← 正确
```

---

## 6.3 完全背包

### 问题定义

```
与0/1背包相同，但每件物品可以选任意多次。
```

### 转移方程

```
dp[i][j] = max(dp[i-1][j - k*weight[i]] + k*value[i])
           for k = 0, 1, 2, ...
```

### 一维数组的关键区别

完全背包 = **从左到右**遍历容量（允许重复选）

```python
def knapsack_complete(weights, values, W):
    n = len(weights)
    dp = [0] * (W + 1)
    
    for i in range(n):
        # ✅ 从左到右！允许同一物品被多次选择
        for j in range(weights[i], W + 1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    
    return dp[W]
```

**对比记忆**：
| 背包类型 | 遍历方向 | 原因 |
|---------|---------|------|
| 0/1背包 | 从右到左 | 每件物品只用一次，防止重复 |
| 完全背包 | 从左到右 | 允许重复使用，需要叠加效果 |

### 实战：零钱兑换（LeetCode 322）

```python
def coin_change(coins, amount):
    # dp[j] = 凑出金额j所需的最少硬币数
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # base case：凑出0元需要0枚
    
    for coin in coins:
        for j in range(coin, amount + 1):  # 从左到右（完全背包）
            dp[j] = min(dp[j], dp[j - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

print(coin_change([1, 5, 10, 25], 41))  # 4 (25+10+5+1)
print(coin_change([2], 3))              # -1
```

### 实战：零钱兑换 II——计数版（LeetCode 518）

```python
def change(amount, coins):
    # dp[j] = 凑出金额j的组合数（不计顺序）
    dp = [0] * (amount + 1)
    dp[0] = 1  # base case：凑出0元有1种方式（什么都不选）
    
    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] += dp[j - coin]
    
    return dp[amount]

print(change(5, [1, 2, 5]))  # 4
```

**⚠️ 组合数 vs 排列数**

```python
# 组合数（不计顺序）：外层遍历物品，内层遍历容量
for coin in coins:
    for j in range(coin, amount + 1):
        dp[j] += dp[j - coin]

# 排列数（计顺序，不同顺序算不同方案）：外层遍历容量，内层遍历物品
for j in range(1, amount + 1):
    for coin in coins:
        if j >= coin:
            dp[j] += dp[j - coin]
```

---

## 6.4 多重背包

### 问题定义

```
第 i 件物品最多可选 k[i] 次。
```

### 朴素方法：拆成0/1背包

```python
def knapsack_bounded_naive(weights, values, counts, W):
    # 将多重背包展开成0/1背包
    new_weights, new_values = [], []
    for w, v, k in zip(weights, values, counts):
        for _ in range(k):
            new_weights.append(w)
            new_values.append(v)
    
    return knapsack_01_optimized(new_weights, new_values, W)
```

**时间复杂度**：O(W × Σk_i)，当 k 很大时效率低。

### 优化：二进制拆分

将 k 个物品拆成 1, 2, 4, ..., 2^m, r 组（r 为余量），将问题转化为规模更小的0/1背包：

```python
def knapsack_bounded(weights, values, counts, W):
    new_weights, new_values = [], []
    
    for w, v, k in zip(weights, values, counts):
        # 二进制分组：1, 2, 4, ..., 2^m, r
        multiplier = 1
        remaining = k
        while multiplier <= remaining:
            new_weights.append(w * multiplier)
            new_values.append(v * multiplier)
            remaining -= multiplier
            multiplier *= 2
        if remaining > 0:
            new_weights.append(w * remaining)
            new_values.append(v * remaining)
    
    return knapsack_01_optimized(new_weights, new_values, W)
```

**优化效果**：每件物品从 O(k) 变为 O(log k) 组，效率大幅提升。

---

## 6.5 分组背包

### 问题定义

```
物品分成 m 组，每组最多选1件（也可以不选），求最大价值。
```

```python
def knapsack_grouped(groups, W):
    # groups[i] = [(weight, value), ...] 第i组的物品列表
    dp = [0] * (W + 1)
    
    for group in groups:
        # ⚠️ 从右到左（类似0/1背包，每组最多选1件）
        for j in range(W, -1, -1):
            for w, v in group:
                if j >= w:
                    dp[j] = max(dp[j], dp[j - w] + v)
    
    return dp[W]
```

---

## 6.6 背包问题的变形：恰好装满

有时题目要求"**恰好**装满背包"而非"**最多**装 W"，只需修改初始化：

```python
# 求最大价值（容量不超过W）
dp = [0] * (W + 1)  # 初始全0：不装任何物品也合法

# 求恰好装满W时的最大价值
dp = [-inf] * (W + 1)
dp[0] = 0  # 只有容量0时"恰好装满"是合法的（装0件物品）
```

---

## 6.7 背包问题总结

```
背包家族速查表

类型        每轮遍历容量方向    核心约束
0/1背包     从右到左          每件物品最多用1次
完全背包    从左到右          每件物品可用无数次
多重背包    从右到左（拆分后）  每件物品最多用k次
分组背包    从右到左          每组最多选1件
```

---

## LeetCode 推荐题目

**0/1背包**
- [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/) ⭐⭐
- [494. 目标和](https://leetcode.cn/problems/target-sum/) ⭐⭐

**完全背包**
- [322. 零钱兑换](https://leetcode.cn/problems/coin-change/) ⭐⭐
- [518. 零钱兑换 II](https://leetcode.cn/problems/coin-change-ii/) ⭐⭐
- [139. 单词拆分](https://leetcode.cn/problems/word-break/) ⭐⭐

**综合**
- [474. 一和零](https://leetcode.cn/problems/ones-and-zeroes/) ⭐⭐⭐（二维背包）
- [879. 盈利计划](https://leetcode.cn/problems/profitable-schemes/) ⭐⭐⭐
