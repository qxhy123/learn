# 第15章：电话号码字母组合

## 15.1 问题描述

**电话号码字母组合**：给定包含 2-9 的字符串，返回所有可能的字母组合。（LeetCode 17）

```
数字到字母的映射（与手机键盘相同）：
2→abc, 3→def, 4→ghi, 5→jkl
6→mno, 7→pqrs, 8→tuv, 9→wxyz

输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

## 15.2 标准回溯解法

```python
def letterCombinations(digits):
    """
    逐位选择字母，组合所有可能
    时间 O(4^n × n)，n = digits 长度（7和9有4个字母）
    """
    if not digits:
        return []
    
    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi',
        '5': 'jkl', '6': 'mno', '7': 'pqrs',
        '8': 'tuv', '9': 'wxyz'
    }
    
    result = []
    
    def backtrack(idx, path):
        if idx == len(digits):
            result.append(''.join(path))
            return
        
        for letter in phone_map[digits[idx]]:
            path.append(letter)
            backtrack(idx + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

# 测试
print(letterCombinations("23"))
# ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
print(letterCombinations(""))
# []
print(letterCombinations("2"))
# ['a', 'b', 'c']
```

## 15.3 迭代解法（BFS 层扩展）

```python
def letterCombinations_bfs(digits):
    """
    每次处理一个数字，将所有当前结果扩展
    类似 BFS 逐层展开
    """
    if not digits:
        return []
    
    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi',
        '5': 'jkl', '6': 'mno', '7': 'pqrs',
        '8': 'tuv', '9': 'wxyz'
    }
    
    result = ['']
    for digit in digits:
        letters = phone_map[digit]
        result = [prev + letter for prev in result for letter in letters]
    
    return result

# 两种方法结果相同，迭代版代码更简洁
print(letterCombinations_bfs("23"))
# ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
```

## 15.4 扩展：统计满足条件的组合数

```python
def countCombinations(digits, predicate):
    """
    统计满足某个条件的字母组合数量
    predicate(combo) → bool
    """
    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi',
        '5': 'jkl', '6': 'mno', '7': 'pqrs',
        '8': 'tuv', '9': 'wxyz'
    }
    
    if not digits:
        return 0
    
    count = [0]
    
    def backtrack(idx, path):
        if idx == len(digits):
            if predicate(''.join(path)):
                count[0] += 1
            return
        for letter in phone_map[digits[idx]]:
            path.append(letter)
            backtrack(idx + 1, path)
            path.pop()
    
    backtrack(0, [])
    return count[0]

# 示例：统计包含元音的组合数
vowels = set('aeiou')
print(countCombinations("23", lambda s: any(c in vowels for c in s)))
```

## 15.5 字符映射的变体问题

### 解码方法（LeetCode 91）

```python
def numDecodings(s):
    """
    '1'→'A', '2'→'B', ..., '26'→'Z'
    计算解码方法数（DP，非回溯）
    
    解释：有重叠子问题，用 DP
    """
    n = len(s)
    if not s or s[0] == '0':
        return 0
    
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    
    for i in range(2, n + 1):
        # 单字符解码
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        # 双字符解码
        two = int(s[i-2:i])
        if 10 <= two <= 26:
            dp[i] += dp[i-2]
    
    return dp[n]

# 对比：回溯解法（超时，但展示结构）
def numDecodings_backtrack(s):
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def bt(idx):
        if idx == len(s):
            return 1
        if s[idx] == '0':
            return 0
        
        result = bt(idx + 1)  # 单字符
        if idx + 1 < len(s) and int(s[idx:idx+2]) <= 26:
            result += bt(idx + 2)  # 双字符
        return result
    
    return bt(0)
```

### 解码方法 II（含 '*' 通配符，LeetCode 639）

```python
def numDecodings_wildcard(s):
    """
    '*' 可以代表 1-9 中的任意字符
    """
    MOD = 10**9 + 7
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 9 if s[0] == '*' else (0 if s[0] == '0' else 1)
    
    for i in range(2, n + 1):
        c, p = s[i-1], s[i-2]
        
        # 单字符
        if c == '*':
            dp[i] += 9 * dp[i-1]  # '*' 可以是 1-9
        elif c != '0':
            dp[i] += dp[i-1]
        
        # 双字符
        if p == '*' and c == '*':
            dp[i] += 15 * dp[i-2]  # 11-19(9) + 21-26(6)
        elif p == '*':
            dp[i] += (2 if '1' <= c <= '6' else 1) * dp[i-2]
        elif c == '*':
            if p == '1':
                dp[i] += 9 * dp[i-2]
            elif p == '2':
                dp[i] += 6 * dp[i-2]
        else:
            two = int(s[i-2:i])
            if 10 <= two <= 26:
                dp[i] += dp[i-2]
        
        dp[i] %= MOD
    
    return dp[n]
```

## 15.6 多键盘布局

```python
def custom_keyboard_combinations(digits, layout):
    """
    自定义键盘布局的字母组合
    layout: {'0': 'aeiou', '1': 'bcdf', ...}
    """
    if not digits:
        return []
    
    result = []
    
    def backtrack(idx, path):
        if idx == len(digits):
            result.append(''.join(path))
            return
        
        digit = digits[idx]
        for letter in layout.get(digit, ''):
            path.append(letter)
            backtrack(idx + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

# 测试：九宫格输入法
nine_grid = {
    '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
    '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
}
print(len(custom_keyboard_combinations("234", nine_grid)))  # 3×3×3 = 27
```

## 小结

| 方法 | 时间复杂度 | 适用场景 |
|------|----------|---------|
| 回溯 | O(4^n × n) | 需要枚举所有组合 |
| 迭代 BFS | O(4^n × n) | 代码更简洁 |
| DP（解码变体）| O(n) | 只需计数（有重叠子问题）|

**核心认知**：电话号码组合与 N 皇后、数独不同，**没有约束需要回溯**（不会放错再撤销），只是纯粹的笛卡尔积枚举。可以用迭代 BFS 替代递归回溯，两者等价。

## 练习

1. 实现"九宫格输入法预测"：给定按键序列，返回前 k 个最可能的单词（结合字典）
2. 解决"按键持续时间最长的键"（LeetCode 1629）
3. 分析为什么"解码方法"不能用简单回溯，必须用记忆化或 DP？

---

**上一章：** [复原 IP 地址](14-restore-ip.md) | **下一章：** [括号生成](16-generate-parentheses.md)
