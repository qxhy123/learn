# 第12章：状压DP（Bitmask DP）

## 12.1 核心思想：用整数表示集合

**状压DP** 的核心是用一个整数的二进制位来表示一个集合的状态，然后在这些集合状态上做动态规划。

**为什么需要状压DP？**

某些问题的状态是一个子集（如"哪些城市已经访问过"），如果用数组或集合表示，状态转移会很复杂；用整数的二进制位表示则简洁高效。

**适用条件**：集合大小 n ≤ 20（因为 2^20 ≈ 10^6，可接受）

---

## 12.2 二进制操作速查

```python
# 设 mask 是一个整数，表示一个集合，i 是元素编号（0-indexed）

# 判断第 i 位是否为 1（元素 i 是否在集合中）
mask & (1 << i) != 0

# 将第 i 位置 1（将元素 i 加入集合）
mask | (1 << i)

# 将第 i 位置 0（从集合中移除元素 i）
mask & ~(1 << i)
mask ^ (1 << i)  # 仅当第i位为1时

# 全集（n个元素）
(1 << n) - 1

# 枚举 mask 的所有子集（重要技巧！）
sub = mask
while sub > 0:
    # 处理子集 sub
    sub = (sub - 1) & mask

# 统计 mask 中 1 的个数
bin(mask).count('1')
# 或：
import math
math.popcount(mask)  # Python 3.10+
```

---

## 12.3 经典：旅行商问题（TSP）

### 问题描述

```
n 个城市，从城市 0 出发，访问所有城市恰好一次，回到城市 0。
求最短路径。

dist[i][j] = 从城市 i 到城市 j 的距离。
```

**暴力复杂度**：O(n!) ，n=20 时完全不可行。

**状压DP复杂度**：O(n² × 2^n)，n=20 时约 4×10^8，勉强可行。

### 状态定义

`dp[mask][i]` = 已访问城市的集合为 `mask`，当前在城市 `i`，到达此状态的最短路径长度

### 转移方程

```
dp[mask | (1 << j)][j] = min(dp[mask | (1 << j)][j],
                              dp[mask][i] + dist[i][j])

其中 j 不在 mask 中（即还没访问 j）
```

### 代码实现

```python
def tsp(dist):
    n = len(dist)
    INF = float('inf')
    full = (1 << n) - 1  # 全集：所有城市都访问
    
    # dp[mask][i] = 从城市0出发，已访问集合为mask，当前在城市i的最短距离
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 初始：只访问了城市0（mask=0b1），在城市0
    
    for mask in range(1 << n):
        for i in range(n):
            if dp[mask][i] == INF:
                continue
            if not (mask & (1 << i)):  # i 不在 mask 中，跳过
                continue
            
            # 从城市 i 移动到未访问的城市 j
            for j in range(n):
                if mask & (1 << j):  # j 已访问，跳过
                    continue
                new_mask = mask | (1 << j)
                new_dist = dp[mask][i] + dist[i][j]
                dp[new_mask][j] = min(dp[new_mask][j], new_dist)
    
    # 从最后一个城市回到城市 0
    return min(dp[full][i] + dist[i][0] for i in range(n))

# 测试
dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
print(tsp(dist))  # 80
```

---

## 12.4 分配问题（LeetCode 526）

**题目**：优美的排列
```
n 个整数 1 到 n，构造满足以下条件的排列的数量：
第 i 个位置上的数字能整除 i 或 i 能整除该数字。
```

```python
def count_arrangement(n):
    # dp[mask] = 用 mask 中的数字填满前 popcount(mask) 个位置的方案数
    dp = [0] * (1 << n)
    dp[0] = 1  # 空排列有1种
    
    for mask in range(1 << n):
        pos = bin(mask).count('1') + 1  # 下一个要填的位置（1-indexed）
        for i in range(n):
            if mask & (1 << i):  # 数字 i+1 已用
                continue
            num = i + 1
            if num % pos == 0 or pos % num == 0:  # 满足条件
                dp[mask | (1 << i)] += dp[mask]
    
    return dp[(1 << n) - 1]

print(count_arrangement(3))   # 3
print(count_arrangement(2))   # 2
```

---

## 12.5 最短超级串（LeetCode 943）

**题目**：给定一组字符串，找一个最短的字符串使所有给定字符串都是它的子串。

```python
def shortest_superstring(words):
    n = len(words)
    
    # 预计算 overlap[i][j] = words[i] 末尾与 words[j] 开头的最长公共前缀
    overlap = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            max_ov = min(len(words[i]), len(words[j]))
            for k in range(max_ov, 0, -1):
                if words[i].endswith(words[j][:k]):
                    overlap[i][j] = k
                    break
    
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    
    # 初始：只包含单个词
    for i in range(n):
        dp[1 << i][i] = len(words[i])
    
    for mask in range(1 << n):
        for last in range(n):
            if not (mask & (1 << last)): continue
            if dp[mask][last] == INF: continue
            
            for nxt in range(n):
                if mask & (1 << nxt): continue
                new_mask = mask | (1 << nxt)
                new_len = dp[mask][last] + len(words[nxt]) - overlap[last][nxt]
                if new_len < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = new_len
                    parent[new_mask][nxt] = last
    
    full = (1 << n) - 1
    last = min(range(n), key=lambda i: dp[full][i])
    
    # 回溯构造结果
    result = []
    mask = full
    while last != -1:
        prev = parent[mask][last]
        if prev == -1:
            result.append(words[last])
        else:
            result.append(words[last][overlap[prev][last]:])
        mask ^= (1 << last)
        last = prev
    
    return ''.join(reversed(result))
```

---

## 12.6 枚举子集的技巧

### 枚举所有大小为 k 的子集

```python
from itertools import combinations

def subsets_of_size_k(n, k):
    for bits in combinations(range(n), k):
        mask = sum(1 << b for b in bits)
        yield mask
```

### SOS DP（Sum over Subsets）

**问题**：对每个集合 S，求所有 S 的子集的函数值之和。

朴素做法：O(3^n)（枚举每个集合的所有子集）。

SOS DP 优化到 O(n × 2^n)：

```python
def sos_dp(f, n):
    # f[mask] = mask 的某个值
    # 计算 g[mask] = sum(f[sub] for all sub ⊆ mask)
    g = f[:]
    
    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                g[mask] += g[mask ^ (1 << i)]
    
    return g
```

---

## 12.7 状态压缩的边界与优化

**可行范围**：
- n ≤ 20：`2^20 = 1,048,576 ≈ 10^6`，状态数可接受
- n ≤ 30：可能需要折半搜索（Meet in the Middle）
- n > 30：通常不适用状压DP

**常见优化**：
```python
# 预处理每个状态的 popcount（1的个数）
popcount = [bin(i).count('1') for i in range(1 << n)]

# 预处理每个状态的最低位1的位置（lowbit）
def lowbit(x): return x & (-x)
```

---

## 12.8 本章小结

**状压DP的思维框架**：

1. 确认状态可以用集合表示（n ≤ 20）
2. 定义 `dp[mask][...]` 表示当前已处理集合 mask 下的最优值
3. 转移：从 mask 出发，加入/移除某个元素，得到新状态
4. 用二进制操作高效实现集合运算

---

## LeetCode 推荐题目

- [526. 优美的排列](https://leetcode.cn/problems/beautiful-arrangement/) ⭐⭐
- [464. 我能赢吗](https://leetcode.cn/problems/can-i-win/) ⭐⭐
- [943. 最短超级串](https://leetcode.cn/problems/find-the-shortest-superstring/) ⭐⭐⭐
- [847. 访问所有节点的最短路径](https://leetcode.cn/problems/shortest-path-visiting-all-nodes/) ⭐⭐⭐
- [1125. 最小的必要团队](https://leetcode.cn/problems/smallest-sufficient-team/) ⭐⭐⭐
