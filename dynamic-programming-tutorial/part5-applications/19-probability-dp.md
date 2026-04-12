# 第19章：概率与期望DP

## 19.1 概率DP基础

**概率DP** 用于计算随机过程中某事件的概率，或随机变量的期望值。

**核心区别**：
- 普通DP：`dp[state]` = 最优值（确定性）
- 概率DP：`dp[state]` = 从该状态出发的概率/期望（随机性）

**两种计算方向**：
- **顺推（forward）**：`P(到达状态 s') += P(状态 s) * P(s→s')`
- **逆推（backward）**：`E(状态 s) = Σ P(s→s') * (cost + E(s'))`

---

## 19.2 基础：骰子滚动

**题目**：掷一个6面骰子，问掷到点数之和恰好为 n 的概率。

```python
def dice_probability(n, faces=6):
    # dp[i] = 点数之和恰好为 i 的概率
    dp = [0.0] * (n + 1)
    dp[0] = 1.0  # 初始概率为1
    
    for i in range(1, n + 1):
        for face in range(1, min(faces, i) + 1):
            dp[i] += dp[i - face] / faces
    
    return dp[n]

print(f"{dice_probability(7):.4f}")   # 掷和为7的概率
print(f"{dice_probability(12):.4f}")
```

**LeetCode 1155 变体**：n 个 k 面骰子，掷出目标点数的方式数。

```python
def num_rolls_to_target(n, k, target):
    MOD = 10**9 + 7
    dp = [[0] * (target + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    
    for i in range(1, n + 1):
        for j in range(i, target + 1):
            for face in range(1, min(k, j) + 1):
                dp[i][j] = (dp[i][j] + dp[i-1][j-face]) % MOD
    
    return dp[n][target]

print(num_rolls_to_target(2, 6, 7))  # 6
```

---

## 19.3 期望DP

**期望值**是概率DP中最重要的概念之一。

**期望的线性性**：`E[X + Y] = E[X] + E[Y]`（无论X和Y是否独立）

### 经典：期望步数（随机游走）

**题目**：一个棋子在数轴上，从位置 k 出发。每步以 p 概率右移1格，以 (1-p) 概率左移1格。到达位置 n 的期望步数？

```python
# 设 E[i] = 从位置 i 到达 n 的期望步数
# E[i] = p*(1 + E[i+1]) + (1-p)*(1 + E[i-1])
# 这是线性方程组，可以用"扫描法"解
# 对于均匀随机游走 (p=0.5)：E[i] = n² - i²（1D随机游走经典结论）
def expected_steps_random_walk(start, end):
    # p=0.5 的对称随机游走：E[i] = (end-i)*(end+i) 的某个形式
    # 实际上需要解方程组
    n = end - start + 1
    # E[i] 满足 E[i] = 0.5*(1+E[i+1]) + 0.5*(1+E[i-1])
    # 即 E[i+1] - 2*E[i] + E[i-1] = -2
    # 解为 E[i] = (end-i)*(i-start+end) 对于 p=0.5（吸收边界）
    # 简化：E[start] = (end-start)² （对称随机游走）
    d = end - start
    return d * d  # 期望步数
```

---

## 19.4 实战：骑士的概率（LeetCode 688）

**题目**：
```
n×n 棋盘，骑士从 (r, c) 出发，随机走 k 步。
每步从8个可能的马步方向中均匀随机选一个。
求 k 步后骑士仍在棋盘上的概率。
```

```python
def knight_probability(n, k, row, column):
    directions = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
    
    # dp[r][c] = 当前在 (r, c) 的概率
    dp = [[0.0] * n for _ in range(n)]
    dp[row][column] = 1.0
    
    for _ in range(k):
        new_dp = [[0.0] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                if dp[r][c] == 0:
                    continue
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        new_dp[nr][nc] += dp[r][c] / 8.0
        dp = new_dp
    
    return sum(dp[r][c] for r in range(n) for c in range(n))

print(f"{knight_probability(3, 2, 0, 0):.5f}")  # 0.06250
```

---

## 19.5 实战：新21点（LeetCode 837）

**题目**：
```
从1开始不断抽取 [1, maxPts] 的随机数相加。
当总和 >= k 时停止。求最终总和 <= n 的概率。
```

**状态**：`dp[i]` = 当前总和为 i 时，最终总和 ≤ n 的概率

```python
def new21game(n, k, maxPts):
    if k == 0 or n >= k + maxPts:
        return 1.0
    
    dp = [0.0] * (n + 1)
    dp[0] = 1.0
    
    window_sum = 1.0  # dp[0] 到 dp[min(maxPts-1, n)] 的滑动窗口和
    
    for i in range(1, n + 1):
        # dp[i] = (dp[i-1] + dp[i-2] + ... + dp[i-maxPts]) / maxPts
        dp[i] = window_sum / maxPts
        
        # 维护滑动窗口
        if i < k:
            window_sum += dp[i]  # i 在 [0, k-1] 范围内，会继续抽卡
        if i >= maxPts:
            window_sum -= dp[i - maxPts]  # 移出窗口
    
    return sum(dp[k:n+1])

print(f"{new21game(10, 1, 10):.5f}")   # 1.00000
print(f"{new21game(6, 1, 10):.5f}")    # 0.60000
print(f"{new21game(21, 17, 10):.5f}")  # 0.73278
```

---

## 19.6 期望的逆推：从终态到初态

有时期望计算更容易从终态往初态逆推：

**题目**：随机打乱 n 张牌，期望多少次翻牌能找到 k 张特定的牌？

```python
def expected_flips(n, k):
    # 经典超几何分布期望
    # E = n * k / (k + 1) + ... 
    # 简单结论：期望翻 (n+1)/(k+1) 张后找到第一张
    # 找到 k 张的期望：k * (n+1) / (k+1)
    
    # DP方法：
    # dp[i] = 还剩 i 张未知牌中有 j 张目标牌时的期望步数（用另一参数）
    # 此处用封闭形式
    return k * (n + 1) / (k + 1)
```

**更复杂的期望DP：糖果（LeetCode 1269）**

```python
def num_ways(steps, arrLen):
    # dp[i][j] = 走了 i 步，当前在位置 j 的方案数
    MOD = 10**9 + 7
    max_pos = min(arrLen - 1, steps // 2)  # 最远只能到 steps//2
    
    dp = {0: 1}  # 初始：位置0，1种方案
    
    for _ in range(steps):
        new_dp = {}
        for pos, ways in dp.items():
            for d in [-1, 0, 1]:
                npos = pos + d
                if 0 <= npos <= max_pos:
                    new_dp[npos] = (new_dp.get(npos, 0) + ways) % MOD
        dp = new_dp
    
    return dp.get(0, 0)
```

---

## 19.7 马尔可夫链与DP

马尔可夫链（Markov Chain）是概率DP的数学基础：

**定义**：状态转移只依赖当前状态（无后效性）

```
P(X_{n+1} = j | X_n = i, X_{n-1}, ...) = P(X_{n+1} = j | X_n = i)
```

这与DP的"无后效性"完全对应！

**平稳分布**：长期停留在各状态的概率，满足 `π = π · P`（用矩阵快速幂可以快速计算）

---

## 19.8 本章小结

| 问题类型 | DP方向 | 状态含义 |
|---------|--------|---------|
| 到达某状态的概率 | 顺推 | `dp[s]` = 到达状态 s 的概率 |
| 从某状态出发的期望 | 逆推 | `dp[s]` = 从状态 s 出发的期望代价 |
| 随机游走 | 逆推 | `dp[i]` = 从位置 i 到终点的期望步数 |

**注意事项**：
- 确认状态满足无后效性（马尔可夫性质）
- 滑动窗口优化求和（避免 O(n²)）
- 注意浮点精度问题（有时用分数或模运算）

---

## LeetCode 推荐题目

- [688. 骑士在棋盘上的概率](https://leetcode.cn/problems/knight-probability-in-chessboard/) ⭐⭐
- [837. 新21点](https://leetcode.cn/problems/new-21-game/) ⭐⭐⭐
- [808. 分汤](https://leetcode.cn/problems/soup-servings/) ⭐⭐⭐
- [1227. 飞机座位分配概率](https://leetcode.cn/problems/airplane-seat-assignment-probability/) ⭐⭐
- [1269. 停在原地的方案数](https://leetcode.cn/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/) ⭐⭐⭐
