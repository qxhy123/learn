# 第2章：记忆化搜索（自顶向下）

## 2.1 核心思想：缓存已计算的结果

记忆化搜索（Memoization）= 递归 + 哈希表缓存

**核心公式**：
```
memo[state] = 第一次计算时存入
              之后遇到直接返回
```

---

## 2.2 手动实现记忆化

以斐波那契为例，对比三个版本：

**版本1：朴素递归（指数复杂度）**
```python
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)
# 时间复杂度: O(2^n)，灾难性的
```

**版本2：手动记忆化**
```python
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]   # 命中缓存，直接返回
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]
# 时间复杂度: O(n)，每个状态只计算一次
```

**版本3：使用 Python 装饰器（推荐写法）**
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
# 等价于版本2，但代码更简洁
```

---

## 2.3 记忆化搜索的执行过程

以 `fib(5)` 为例，追踪执行过程：

```
第一次调用 fib(5):
  → 需要 fib(4)
    → 需要 fib(3)
      → 需要 fib(2)
        → 需要 fib(1) = 1 ✓ (base case)
        → 需要 fib(0) = 0 ✓ (base case)
        → fib(2) = 1，存入缓存
      → 需要 fib(1) = 1 ✓ (命中缓存！)
      → fib(3) = 2，存入缓存
    → 需要 fib(2) = 1 ✓ (命中缓存！)
    → fib(4) = 3，存入缓存
  → 需要 fib(3) = 2 ✓ (命中缓存！)
  → fib(5) = 5，存入缓存
```

每个子问题只被**真正计算一次**，之后全部命中缓存。

---

## 2.4 实战：三角形路径最小和

**题目**（LeetCode 120）：
```
给定一个三角形数组，找从顶到底的最小路径和。
每步只能移动到下一行相邻的元素。

    2
   3 4
  6 5 7
 4 1 8 3

最小路径：2 + 3 + 5 + 1 = 11
```

**递归思路**：`f(i, j)` = 从位置 `(i, j)` 到底部的最小路径和

```python
from functools import lru_cache

def minimum_total(triangle):
    n = len(triangle)
    
    @lru_cache(maxsize=None)
    def dp(row, col):
        # base case: 最后一行
        if row == n - 1:
            return triangle[row][col]
        
        # 可以向左下或右下移动
        go_left  = dp(row + 1, col)
        go_right = dp(row + 1, col + 1)
        
        return triangle[row][col] + min(go_left, go_right)
    
    return dp(0, 0)

# 测试
triangle = [[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]
print(minimum_total(triangle))  # 11
```

**状态数量**：O(n²)，每个状态计算一次，总时间复杂度 O(n²)。

---

## 2.5 实战：单词拆分

**题目**（LeetCode 139）：
```
给定字符串 s 和单词字典 wordDict，
判断 s 是否可以被拆分成字典中的单词。

s = "leetcode", wordDict = ["leet", "code"]  → True
s = "applepenapple", wordDict = ["apple", "pen"]  → True
s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]  → False
```

**递归思路**：`can_break(i)` = `s[i:]` 能否被拆分

```python
from functools import lru_cache

def word_break(s, wordDict):
    word_set = set(wordDict)
    n = len(s)
    
    @lru_cache(maxsize=None)
    def can_break(start):
        # base case: 已处理完整个字符串
        if start == n:
            return True
        
        # 枚举以 start 开头的所有可能单词
        for end in range(start + 1, n + 1):
            if s[start:end] in word_set and can_break(end):
                return True
        
        return False
    
    return can_break(0)

print(word_break("leetcode", ["leet", "code"]))       # True
print(word_break("applepenapple", ["apple", "pen"]))  # True
print(word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]))  # False
```

---

## 2.6 多维状态的记忆化搜索

有时状态不只一个维度。以"不同路径"为例：

**题目**（LeetCode 62）：
```
m×n 网格，从左上角走到右下角，只能向右或向下。
有多少条不同的路径？
```

```python
from functools import lru_cache

def unique_paths(m, n):
    @lru_cache(maxsize=None)
    def dp(i, j):
        # base case: 第一行或第一列只有一种走法
        if i == 0 or j == 0:
            return 1
        return dp(i-1, j) + dp(i, j-1)
    
    return dp(m-1, n-1)

print(unique_paths(3, 7))  # 28
```

**状态**：`(i, j)` 二维状态，总数 O(m×n)
**转移**：每个状态依赖左边和上边两个状态

---

## 2.7 记忆化搜索 vs 制表法的选择

| 维度 | 记忆化搜索 | 制表法（Bottom-up） |
|------|-----------|-------------------|
| 代码结构 | 递归，直观 | 迭代，显式填表 |
| 子问题计算 | 按需（Lazy） | 全部计算 |
| 递归栈开销 | 有（O(深度)） | 无 |
| 状态顺序 | 自动处理 | 需要手动确定填表顺序 |
| 适合场景 | 状态空间稀疏、依赖关系复杂 | 状态空间密集、顺序明确 |

**实际建议**：
- 竞赛中：记忆化搜索往往能更快写出来
- 面试中：两种都要会，视情况选择
- 生产代码中：制表法通常性能更好

---

## 2.8 避免常见陷阱

### 陷阱1：Python 默认参数可变陷阱
```python
# 错误！memo 在多次调用之间共享
def fib(n, memo={}):  
    ...

# 正确写法
def fib(n, memo=None):
    if memo is None:
        memo = {}
    ...
```

### 陷阱2：递归深度限制
```python
import sys
sys.setrecursionlimit(10000)  # Python 默认递归深度约 1000

# 更好的做法：用制表法避免深递归
```

### 陷阱3：unhashable 的状态
```python
# 列表不能作为 lru_cache 的参数（unhashable）
@lru_cache
def dp(arr):  # 错误！list 不可哈希
    ...

# 解决方案：转换为 tuple
@lru_cache
def dp(arr_tuple):  # 正确
    ...
```

---

## 2.9 本章小结

记忆化搜索的通用模板：

```python
from functools import lru_cache

def solve(problem_input):
    @lru_cache(maxsize=None)
    def dp(state):
        # 1. base case
        if is_base_case(state):
            return base_value(state)
        
        # 2. 状态转移
        result = combine(
            dp(sub_state_1),
            dp(sub_state_2),
            ...
        )
        
        return result
    
    return dp(initial_state)
```

**下一章：制表法——自底向上填写DP表格**

---

## LeetCode 推荐题目

- [120. 三角形最小路径和](https://leetcode.cn/problems/triangle/) ⭐⭐
- [139. 单词拆分](https://leetcode.cn/problems/word-break/) ⭐⭐
- [62. 不同路径](https://leetcode.cn/problems/unique-paths/) ⭐
- [329. 矩阵中的最长递增路径](https://leetcode.cn/problems/longest-increasing-path-in-a-matrix/) ⭐⭐⭐
