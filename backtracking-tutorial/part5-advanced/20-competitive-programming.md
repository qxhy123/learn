# 第20章：竞赛题精选

## 20.1 本章目标

本章精选 6 道经典竞赛题，覆盖回溯的高阶技巧：

| 题目 | 难度 | 核心技巧 |
|------|------|---------|
| 解数独（困难数独） | ★★★★ | MRV启发 + 约束传播 |
| 滑动拼图 | ★★★★ | BFS + 状态哈希 |
| 拼接最大数 | ★★★★ | 贪心 + 回溯验证 |
| 表达式添加运算符 | ★★★★ | 回溯 + 表达式树 |
| 切割木棍 | ★★★★ | 回溯 + 剪枝精讲 |
| 正则表达式匹配 | ★★★★ | 回溯 + 记忆化 |

---

## 20.2 表达式添加运算符（LeetCode 282）

**问题**：在数字字符串中添加 `+`、`-`、`*`，使表达式值等于目标值。

```
输入：num = "123", target = 6
输出：["1+2+3", "1*2*3"]

输入：num = "105", target = 5
输出：["1*0+5","10-5"]
```

```python
def addOperators(num, target):
    """
    难点：乘法需要记录上一个操作数（影响撤销）
    
    核心技巧：维护 (当前值 eval, 上一个操作数 mult)
    - 加法：eval + x,  mult = x
    - 减法：eval - x,  mult = -x
    - 乘法：eval - mult + mult*x,  mult = mult*x
    """
    result = []
    n = len(num)
    
    def backtrack(idx, path, eval_val, mult):
        if idx == n:
            if eval_val == target:
                result.append(''.join(path))
            return
        
        for end in range(idx + 1, n + 1):
            # 避免前导零（"05" 无效，但 "0" 本身有效）
            if end > idx + 1 and num[idx] == '0':
                break
            
            num_str = num[idx:end]
            x = int(num_str)
            
            if idx == 0:  # 第一个数字，不加运算符
                backtrack(end, [num_str], x, x)
            else:
                # 加法
                path.append('+'); path.append(num_str)
                backtrack(end, path, eval_val + x, x)
                path.pop(); path.pop()
                
                # 减法
                path.append('-'); path.append(num_str)
                backtrack(end, path, eval_val - x, -x)
                path.pop(); path.pop()
                
                # 乘法（关键：撤销上一次加/减，换成乘）
                path.append('*'); path.append(num_str)
                backtrack(end, path, eval_val - mult + mult * x, mult * x)
                path.pop(); path.pop()
    
    backtrack(0, [], 0, 0)
    return result

print(addOperators("123", 6))   # ['1+2+3', '1*2*3']
print(addOperators("105", 5))   # ['1*0+5', '10-5']
print(addOperators("00", 0))    # ['0+0', '0-0', '0*0']
```

---

## 20.3 切割木棍（LeetCode 473 变体）

**问题**：将整数数组分成 4 组等和子集（正方形火柴棒）。

```python
def makesquare(matchsticks):
    """
    火柴拼正方形：将数组分成 4 等份
    
    关键剪枝：
    1. 从大到小排序（大的先分配，更早发现无解）
    2. 相同大小的桶只试一次（等价状态去重）
    3. 某个桶填满后跳过（避免重复搜索）
    """
    total = sum(matchsticks)
    if total % 4 != 0:
        return False
    
    side = total // 4
    if max(matchsticks) > side:
        return False
    
    matchsticks.sort(reverse=True)
    sides = [0] * 4
    
    def backtrack(idx):
        if idx == len(matchsticks):
            return sides[0] == sides[1] == sides[2] == sides[3] == side
        
        seen = set()
        for i in range(4):
            if sides[i] + matchsticks[idx] > side:
                continue
            if sides[i] in seen:
                continue  # 等价剪枝
            seen.add(sides[i])
            
            sides[i] += matchsticks[idx]
            if backtrack(idx + 1):
                return True
            sides[i] -= matchsticks[idx]
            
            if sides[i] == 0:
                break  # 空桶等价，只试一次
        
        return False
    
    return backtrack(0)

print(makesquare([1,1,2,2,2]))  # True
print(makesquare([3,3,3,3,4]))  # False
```

---

## 20.4 正则表达式匹配（LeetCode 10）

**问题**：实现支持 `.` 和 `*` 的正则表达式匹配。

```python
# 方法一：记忆化回溯（直观）
from functools import lru_cache

def isMatch_memo(s, p):
    """
    . 匹配任意单字符
    * 匹配零个或多个前面的字符
    """
    @lru_cache(maxsize=None)
    def dp(i, j):
        if j == len(p):
            return i == len(s)
        
        first_match = i < len(s) and (p[j] == '.' or p[j] == s[i])
        
        if j + 1 < len(p) and p[j+1] == '*':
            # '*' 匹配零次 或 匹配一次并继续
            return dp(i, j+2) or (first_match and dp(i+1, j))
        else:
            return first_match and dp(i+1, j+1)
    
    return dp(0, 0)

# 方法二：DP 表格（高效）
def isMatch_dp(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n+1) for _ in range(m+1)]
    dp[0][0] = True
    
    # 处理 "a*b*c*..." 匹配空字符串
    for j in range(2, n+1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-2]  # '*' 匹配零次
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] |= dp[i-1][j]  # '*' 匹配一次及以上
            elif p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
    
    return dp[m][n]

# 测试
cases = [("aa", "a"), ("aa", "a*"), ("ab", ".*"), ("aab", "c*a*b")]
for s, p in cases:
    assert isMatch_memo(s, p) == isMatch_dp(s, p)
    print(f'"{s}" matches "{p}": {isMatch_dp(s, p)}')
```

---

## 20.5 综合练习：骑士巡游

**问题**：马在 n×n 棋盘上，从 (0,0) 出发，经过每个格子恰好一次（哈密顿路径）。

```python
def knight_tour(n):
    """
    Warnsdorff 启发式：总是优先移动到出度最小的格子
    将 O(8^(n²)) 优化到接近 O(n²)
    """
    moves = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
    
    def is_valid(r, c, board):
        return 0 <= r < n and 0 <= c < n and board[r][c] == -1
    
    def degree(r, c, board):
        """计算 (r,c) 的出度（可移动到的格子数）"""
        return sum(1 for dr, dc in moves if is_valid(r+dr, c+dc, board))
    
    board = [[-1]*n for _ in range(n)]
    board[0][0] = 0
    
    def backtrack(r, c, move_num):
        if move_num == n*n:
            return True
        
        # Warnsdorff：按出度排序，优先度小的
        next_moves = []
        for dr, dc in moves:
            nr, nc = r+dr, c+dc
            if is_valid(nr, nc, board):
                next_moves.append((degree(nr, nc, board), nr, nc))
        
        next_moves.sort()  # 按出度升序
        
        for _, nr, nc in next_moves:
            board[nr][nc] = move_num
            if backtrack(nr, nc, move_num + 1):
                return True
            board[nr][nc] = -1
        
        return False
    
    if backtrack(0, 0, 1):
        return board
    return None

# 测试 6×6 棋盘
result = knight_tour(6)
if result:
    for row in result:
        print([f'{x:2d}' for x in row])
```

---

## 20.6 竞赛技巧总结

```python
"""
竞赛中的回溯模板（高效版）

1. 状态压缩：用整数位表示集合（比 set 快 5-10 倍）
   - visited |= (1 << i)    # 标记访问
   - visited & (1 << i)     # 检查访问
   - visited & ~(1 << i)    # 取消访问

2. 预计算约束：回溯前计算好所有约束条件
   - N 皇后：预算对角线标记
   - 数独：预算行/列/宫格候选

3. 迭代加深：对深度限制的搜索
   for depth in range(1, max_depth + 1):
       if dfs(start, 0, depth):
           return solution

4. 双向搜索：从两端同时搜索，在中间相遇
   - 状态数从 O(b^d) 降到 O(b^(d/2))
   
5. 最优优先搜索（A*）：
   - 维护优先队列，优先扩展估价函数小的节点
   - 需要设计启发函数 h(state)
"""

# 位运算版全排列（竞赛常用）
def permutations_bitmask(n):
    """用位掩码记录已使用元素，比 used[] 数组快"""
    result = []
    path = []
    
    def backtrack(mask):
        if mask == (1 << n) - 1:
            result.append(path[:])
            return
        for i in range(n):
            if mask & (1 << i):
                continue
            path.append(i)
            backtrack(mask | (1 << i))
            path.pop()
    
    backtrack(0)
    return result

print(f"n=4 全排列：{len(permutations_bitmask(4))} 种")  # 24 种
```

## 20.7 学习路线回顾

```
第1-4章（基础）：
  回溯本质 → 决策树 → 模板 → 复杂度分析

第5-8章（组合问题）：
  子集 → 组合 → 排列 → 高阶变体

第9-12章（棋盘问题）：
  N皇后 → 数独 → 单词搜索 → 泛洪填充

第13-16章（字符串分割）：
  回文分割 → IP地址 → 电话号码 → 括号生成

第17-20章（高阶技巧）：
  剪枝专题 → 记忆化 → 回溯vs DP → 竞赛精选

推荐刷题顺序（LeetCode）：
  入门：78, 46, 22, 77
  进阶：90, 47, 39, 40, 131, 17
  困难：51, 37, 79, 212, 282, 10
```

## 小结

回溯算法的核心能力矩阵：

| 技能 | 对应章节 |
|------|---------|
| 决策树建模 | 第2章 |
| 标准模板 | 第3章 |
| 去重策略 | 第5-7章 |
| 棋盘约束 | 第9-12章 |
| 剪枝设计 | 第17章 |
| 记忆化判断 | 第18-19章 |
| 竞赛优化 | 第20章 |

## 最终练习

1. 实现"旅行商问题"（TSP）的精确回溯解法（n≤20），比较带/不带位压缩的性能差异
2. 用回溯实现"数独生成器"：先填入完整解，再随机移除数字（确保唯一解）
3. 阅读 Donald Knuth 的 "Dancing Links" 论文，理解精确覆盖问题的高效回溯

---

**上一章：** [回溯 vs 动态规划](19-backtracking-vs-dp.md) | **教程完结** 🎉

感谢学习本教程！回溯算法是算法竞赛和工程实践中不可或缺的工具。掌握它，不仅能解决 LeetCode 困难题，更能培养系统化的搜索思维。
