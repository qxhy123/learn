# 第16章：括号生成

## 16.1 问题描述

**括号生成**：给定 n 对括号，生成所有有效的括号组合。（LeetCode 22）

```
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
共 5 种（第 3 个卡特兰数 C_3 = 5）
```

## 16.2 核心约束

```
有效括号的条件：
1. 总共使用 n 个 '(' 和 n 个 ')'
2. 任意前缀中，'(' 的数量 >= ')' 的数量

等价约束（便于回溯剪枝）：
- open  = 已放置 '(' 的数量，合法范围 [0, n]
- close = 已放置 ')' 的数量，合法范围 [0, open]
```

## 16.3 标准回溯解法

```python
def generateParenthesis(n):
    """
    每步选择放 '(' 或 ')'
    约束：open <= n，close <= open
    时间 O(4^n / √n)（卡特兰数），空间 O(n)
    """
    result = []
    
    def backtrack(path, open_count, close_count):
        if len(path) == 2 * n:
            result.append(''.join(path))
            return
        
        # 可以放 '('：已放的左括号数 < n
        if open_count < n:
            path.append('(')
            backtrack(path, open_count + 1, close_count)
            path.pop()
        
        # 可以放 ')'：已放的右括号数 < 左括号数
        if close_count < open_count:
            path.append(')')
            backtrack(path, open_count, close_count + 1)
            path.pop()
    
    backtrack([], 0, 0)
    return result

# 测试
for n in range(1, 5):
    res = generateParenthesis(n)
    print(f"n={n}: {len(res)} 种, {res}")
# n=1: 1 种, ['()']
# n=2: 2 种, ['(())', '()()']
# n=3: 5 种, ['((()))', '(()())', '(())()', '()(())', '()()()']
# n=4: 14 种
```

## 16.4 卡特兰数与括号计数

```python
def catalan_number(n):
    """第 n 个卡特兰数 = C(2n,n) / (n+1)"""
    from math import comb
    return comb(2*n, n) // (n+1)

# 括号对数 vs 方案数
for n in range(1, 8):
    print(f"n={n}: {catalan_number(n)} 种")
# n=1: 1   n=2: 2   n=3: 5   n=4: 14
# n=5: 42  n=6: 132  n=7: 429

# 卡特兰数递推：C(n+1) = sum(C(i) * C(n-i)) for i in 0..n
def catalan_dp(n):
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i-1-j]
    return dp[n]
```

## 16.5 验证有效括号

```python
def isValid(s):
    """验证括号字符串是否有效（LeetCode 20）"""
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            stack.append(char)
    
    return not stack

def minAddToMakeValid(s):
    """最少添加多少括号使字符串有效（LeetCode 921）"""
    open_needed = 0   # 需要配对的 '(' 数量
    close_needed = 0  # 需要配对的 ')' 数量
    
    for c in s:
        if c == '(':
            close_needed += 1
        else:
            if close_needed > 0:
                close_needed -= 1
            else:
                open_needed += 1
    
    return open_needed + close_needed
```

## 16.6 多种括号类型生成

```python
def generateAllBrackets(n):
    """
    生成含 ()、[]、{} 的所有有效括号组合
    每种括号各用 n 对
    """
    pairs = [('(', ')'), ('[', ']'), ('{', '}')]
    result = []
    counts = {c: 0 for pair in pairs for c in pair}
    
    def backtrack(path, open_counts, close_counts):
        if len(path) == 2 * n * len(pairs):
            result.append(''.join(path))
            return
        
        # 尝试放每种左括号
        for open_c, close_c in pairs:
            # 放左括号
            if open_counts[open_c] < n:
                path.append(open_c)
                open_counts[open_c] += 1
                backtrack(path, open_counts, close_counts)
                path.pop()
                open_counts[open_c] -= 1
            
            # 放右括号
            if close_counts[close_c] < open_counts[open_c]:
                path.append(close_c)
                close_counts[close_c] += 1
                backtrack(path, open_counts, close_counts)
                path.pop()
                close_counts[close_c] -= 1
    
    open_counts = {'(': 0, '[': 0, '{': 0}
    close_counts = {')': 0, ']': 0, '}': 0}
    backtrack([], open_counts, close_counts)
    return result
```

## 16.7 括号生成的动态规划方案

```python
def generateParenthesis_dp(n):
    """
    DP：dp[i] = 所有 i 对括号的有效组合
    递推：dp[n] = "("+dp[i]+")" + dp[n-1-i]，i 在 0..n-1
    
    思路：第一个 '(' 总是与某个 ')' 配对，
    其内部有 i 对括号，其右侧有 n-1-i 对括号
    """
    dp = [[] for _ in range(n + 1)]
    dp[0] = ['']
    
    for i in range(1, n + 1):
        for j in range(i):
            for inner in dp[j]:
                for right in dp[i - 1 - j]:
                    dp[i].append(f'({inner}){right}')
    
    return dp[n]

# 验证两种方法结果相同（排序后比较）
n = 4
bt_result = sorted(generateParenthesis(n))
dp_result = sorted(generateParenthesis_dp(n))
assert bt_result == dp_result, "两种方法结果不一致"
print(f"n={n}: 两种方法均得到 {len(bt_result)} 种，结果一致")
```

## 16.8 不同括号数量生成

```python
def generateParenthesisUnequal(n_open, n_close):
    """
    生成 n_open 个 '(' 和 n_close 个 ')' 的有效序列
    注意：如果 n_close < n_open，不可能有有效序列
    """
    result = []
    
    def backtrack(path, open_left, close_left, balance):
        # balance = '(' 数量 - ')' 数量，必须 >= 0
        if balance < 0:
            return
        if open_left == 0 and close_left == 0:
            result.append(''.join(path))
            return
        
        if open_left > 0:
            path.append('(')
            backtrack(path, open_left-1, close_left, balance+1)
            path.pop()
        
        if close_left > 0 and balance > 0:
            path.append(')')
            backtrack(path, open_left, close_left-1, balance-1)
            path.pop()
    
    backtrack([], n_open, n_close, 0)
    return result
```

## 小结

| 知识点 | 核心内容 |
|--------|---------|
| 回溯约束 | `open <= n`，`close <= open` |
| 计数公式 | 卡特兰数 C(n) = C(2n,n)/(n+1) |
| DP 等价 | dp[n] 用 dp[j] 和 dp[n-1-j] 构造 |
| 验证技巧 | 栈计数，保证任意前缀左 >= 右 |

## 练习

1. 求第 n 种有效括号组合（字典序排列，LeetCode 1111 变体）
2. 实现"删除无效括号"（LeetCode 301）：删除最少括号使字符串有效
3. 证明为什么 n 对括号的有效组合数恰好是卡特兰数

---

**上一章：** [电话号码字母组合](15-letter-combinations.md) | **下一章（Part 5）：** [剪枝优化专题](../part5-advanced/17-pruning-optimization.md)
