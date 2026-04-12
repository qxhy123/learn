# 第18章：博弈论DP

## 18.1 博弈论基础

**组合博弈（Combinatorial Game）** 的特征：
- 两个玩家交替操作
- 完全信息（双方都知道当前状态）
- 无随机因素
- 最后无法操作的人输（或赢，取决于规则）

**核心定义**：
- **P-position（Previous player wins）**：先手**必败**的局面（即轮到你操作时，你必输）
- **N-position（Next player wins）**：先手**必胜**的局面（即轮到你操作时，你必赢）

**关键性质**：
- 终止状态（无法操作）是 **P-position**（标准规则下）
- 从 N-position 出发，存在至少一步可以到达 P-position
- 从 P-position 出发，所有一步都到达 N-position

---

## 18.2 Nim 游戏

**题目**（LeetCode 292 变体）：
```
n 堆石子，每次可以从某一堆中取任意数量（至少1个）。
取走最后一个石子的人赢。
判断先手是否必胜。
```

**Nim 定理（Sprague-Grundy 的特例）**：

先手必胜 当且仅当 `a₁ XOR a₂ XOR ... XOR aₙ ≠ 0`

```python
def can_win_nim(piles):
    xor_sum = 0
    for pile in piles:
        xor_sum ^= pile
    return xor_sum != 0

print(can_win_nim([1, 2, 3]))  # False（1^2^3=0，先手必败）
print(can_win_nim([1, 2, 4]))  # True
```

**为什么 XOR？**

用动态规划理解：
```
状态 (a₁, a₂, ..., aₙ)：
- 若全为0，是 P-position（无法操作，当前玩家输）
- 若 XOR != 0，是 N-position（存在一步使XOR变为0）
- 若 XOR == 0 且不全为0，是 P-position（任何一步都使XOR不为0）
```

---

## 18.3 Sprague-Grundy 定理

**Grundy 值（nimber）**的定义：

```
g(state) = mex{g(s') | s' 是 state 的后继状态}
```

其中 `mex` = **最小非负整数，不在集合中**（Minimum EXcluding）

```python
def compute_grundy(state, moves, memo={}):
    if state in memo:
        return memo[state]
    
    reachable = set()
    for move in moves(state):
        reachable.add(compute_grundy(move, moves, memo))
    
    # mex
    g = 0
    while g in reachable:
        g += 1
    
    memo[state] = g
    return g
```

**S-G 定理**：多个独立游戏的组合，总 Grundy 值 = 各部分 Grundy 值的 XOR。

---

## 18.4 经典：取石子游戏 II（LeetCode 1140）

**题目**：
```
一排石子，两人轮流取。
第一次可以取任意数量（至少1个），之后每次最多取上一次数量的2倍。
取走最后石子的人赢，双方都采用最优策略。
```

**DP解法**：

```python
from functools import lru_cache

def stone_game_ii(piles):
    n = len(piles)
    
    # 后缀和
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + piles[i]
    
    @lru_cache(maxsize=None)
    def dp(i, m):
        # 从位置 i 开始，当前玩家可以取 1~2m 堆
        # 返回当前玩家能获得的最多石子数
        if i + 2 * m >= n:
            return suffix[i]  # 可以取完剩余所有
        
        best = 0
        for x in range(1, 2 * m + 1):
            # 取 x 堆，对手从 i+x 开始，m 变为 max(m, x)
            opponent = dp(i + x, max(m, x))
            my_score = suffix[i] - opponent  # 剩余 - 对手得分
            best = max(best, my_score)
        
        return best
    
    return dp(0, 1)

print(stone_game_ii([2, 7, 9, 4, 4]))  # 10
```

---

## 18.5 石子游戏系列

### 石子游戏 I（LeetCode 877）

```
偶数堆石子，双方交替取首或尾一堆，总数更多的人赢。
先手是否必胜？
```

**数学结论**：先手必胜（奇数位的石子总和与偶数位总和，先手可以始终控制选哪组）。

```python
def stone_game(piles):
    return True  # 先手必胜

# DP验证（区间DP）
def stone_game_dp(piles):
    n = len(piles)
    # dp[i][j] = 先手在 piles[i..j] 上能比后手多拿的最多数
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = piles[i]
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(
                piles[i] - dp[i+1][j],  # 取左端
                piles[j] - dp[i][j-1]   # 取右端
            )
    
    return dp[0][n-1] > 0
```

### 石子游戏 V（LeetCode 1563）

```
每轮将一行石子分成左右两组，取较小组的分值，删除较大组继续。
求最大得分。
```

```python
from functools import lru_cache

def stone_game_v(stoneValue):
    n = len(stoneValue)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + stoneValue[i]
    
    @lru_cache(maxsize=None)
    def dp(i, j):
        if i == j:
            return 0
        best = 0
        for k in range(i, j):
            left  = prefix[k+1] - prefix[i]
            right = prefix[j+1] - prefix[k+1]
            if left < right:
                best = max(best, left + dp(i, k))
            elif left > right:
                best = max(best, right + dp(k+1, j))
            else:
                best = max(best, left + max(dp(i, k), dp(k+1, j)))
        return best
    
    return dp(0, n-1)
```

---

## 18.6 翻转游戏与记忆化（LeetCode 294）

```
字符串含 '+' 和 '-'，每次将相邻两个 '+' 翻转为 '--'。
无法操作的人输。判断先手是否必胜。
```

```python
from functools import lru_cache

def can_win(s):
    @lru_cache(maxsize=None)
    def wins(state):
        for i in range(len(state) - 1):
            if state[i] == '+' and state[i+1] == '+':
                next_state = state[:i] + '--' + state[i+2:]
                if not wins(next_state):  # 对手必败
                    return True
        return False  # 无法操作，当前玩家输
    
    return wins(s)
```

---

## 18.7 本章小结

**博弈论DP的核心框架**：

```python
@lru_cache
def dp(state):
    if is_terminal(state):
        return False  # 当前玩家输（标准规则）
    
    for next_state in successors(state):
        if not dp(next_state):  # 存在使对手必败的走法
            return True
    
    return False  # 所有走法都导致对手必胜
```

**关键工具**：

| 工具 | 适用场景 |
|------|---------|
| Nim定理（XOR） | 多堆取石子，可任意取 |
| Sprague-Grundy | 复杂单堆游戏，多游戏组合 |
| 区间DP | 从两端取，得分最大化 |
| 记忆化搜索 | 状态可枚举的任意博弈 |

---

## LeetCode 推荐题目

- [292. Nim 游戏](https://leetcode.cn/problems/nim-game/) ⭐
- [877. 石子游戏](https://leetcode.cn/problems/stone-game/) ⭐⭐
- [1140. 石子游戏 II](https://leetcode.cn/problems/stone-game-ii/) ⭐⭐⭐
- [1563. 石子游戏 V](https://leetcode.cn/problems/stone-game-v/) ⭐⭐⭐
- [375. 猜数字大小 II](https://leetcode.cn/problems/guess-number-higher-or-lower-ii/) ⭐⭐
