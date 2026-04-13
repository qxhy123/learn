# 第13章：回文分割

## 13.1 问题描述

**分割回文串**：将字符串 `s` 分割，使得每个子串都是回文串，返回所有可能的分割方案。（LeetCode 131）

```
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]

输入：s = "a"
输出：[["a"]]
```

## 13.2 基础回溯解法

```python
def partition(s):
    """
    从每个位置切割，检查左侧是否为回文
    时间 O(n × 2^n)，空间 O(n)
    """
    result = []
    
    def is_palindrome(sub):
        return sub == sub[::-1]
    
    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])
            return
        
        for end in range(start + 1, len(s) + 1):
            sub = s[start:end]
            if is_palindrome(sub):
                path.append(sub)
                backtrack(end, path)
                path.pop()
    
    backtrack(0, [])
    return result
```

## 13.3 优化：预计算回文表

每次调用 `is_palindrome` 是 O(n)，用动态规划预计算所有子串的回文性：

```python
def partition_dp(s):
    """
    预计算 is_pal[i][j] = s[i..j] 是否为回文
    回溯时 O(1) 查询
    """
    n = len(s)
    
    # DP 预计算
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True  # 单字符
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                is_pal[i][j] = (length == 2) or is_pal[i+1][j-1]
    
    result = []
    
    def backtrack(start, path):
        if start == n:
            result.append(path[:])
            return
        
        for end in range(start, n):
            if is_pal[start][end]:
                path.append(s[start:end+1])
                backtrack(end + 1, path)
                path.pop()
    
    backtrack(0, [])
    return result

# 测试
print(partition_dp("aab"))  # [['a', 'a', 'b'], ['aa', 'b']]
print(partition_dp("aabb")) # [['a', 'a', 'b', 'b'], ['a', 'a', 'bb'], ['aa', 'b', 'b'], ['aa', 'bb']]
```

## 13.4 只求最少切割次数（LeetCode 132）

**问题**：将字符串分割为若干回文子串，求最少切割次数。

```python
def minCut(s):
    """
    DP：min_cut[i] = s[0..i] 的最少切割次数
    时间 O(n²)
    """
    n = len(s)
    
    # 预计算回文表
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                is_pal[i][j] = (length == 2) or is_pal[i+1][j-1]
    
    # dp[i] = s[0..i-1] 的最少切割次数
    dp = list(range(n))  # 最多切 i 刀
    
    for i in range(n):
        if is_pal[0][i]:
            dp[i] = 0  # 整段是回文，不需要切
            continue
        for j in range(1, i + 1):
            if is_pal[j][i]:
                dp[i] = min(dp[i], dp[j-1] + 1)
    
    return dp[n-1]

print(minCut("aab"))    # 1（"aa" | "b"）
print(minCut("a"))      # 0
print(minCut("ab"))     # 1
```

## 13.5 Manacher 算法优化（O(n) 预计算）

```python
def partition_manacher(s):
    """用 Manacher 算法 O(n) 预计算所有回文半径"""
    
    # Manacher 预处理：在字符间插入 '#'
    t = '#' + '#'.join(s) + '#'
    m = len(t)
    p = [0] * m  # p[i] = 以 t[i] 为中心的最大回文半径
    
    center = right = 0
    for i in range(m):
        if i < right:
            mirror = 2 * center - i
            p[i] = min(right - i, p[mirror])
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < m and t[i-p[i]-1] == t[i+p[i]+1]:
            p[i] += 1
        if i + p[i] > right:
            center, right = i, i + p[i]
    
    n = len(s)
    # 从 Manacher 结果转回原字符串的回文判断
    is_pal = [[False] * n for _ in range(n)]
    for i in range(m):
        # t[i] 对应原字符串的区间
        radius = p[i]
        # t 中 [i-radius, i+radius] 对应原字符串的范围
        left_t = i - radius
        right_t = i + radius
        # 去掉 '#'，转回原下标
        left_s = left_t // 2
        right_s = right_t // 2 - (1 if right_t % 2 == 0 else 0)
        if left_s >= 0 and right_s < n:
            is_pal[left_s][right_s] = True
    
    result = []
    def backtrack(start, path):
        if start == n:
            result.append(path[:])
            return
        for end in range(start, n):
            if is_pal[start][end]:
                path.append(s[start:end+1])
                backtrack(end + 1, path)
                path.pop()
    
    backtrack(0, [])
    return result
```

## 13.6 回文分割的扩展

```python
def partition_k(s, k):
    """
    将字符串分成恰好 k 个回文子串，返回是否可行
    （LeetCode 1278 变体）
    """
    n = len(s)
    if k > n:
        return False
    
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            is_pal[i][j] = (s[i] == s[j]) and ((l == 2) or is_pal[i+1][j-1])
    
    def backtrack(start, remaining):
        if remaining == 0:
            return start == n
        if start == n:
            return False
        # 剪枝：剩余字符不足以分成 remaining 段
        if n - start < remaining:
            return False
        
        for end in range(start, n - remaining + 1):
            if is_pal[start][end]:
                if backtrack(end + 1, remaining - 1):
                    return True
        return False
    
    return backtrack(0, k)
```

## 小结

| 方法 | 回文检查时间 | 总时间复杂度 |
|------|------------|------------|
| 暴力（每次 O(n)）| O(n) | O(n² × 2^n) |
| DP 预计算 | O(1) | O(n × 2^n) |
| Manacher 预计算 | O(1) | O(n × 2^n) |
| 只求最少切割 | - | O(n²) DP |

## 练习

1. 实现"最长回文子序列"（LeetCode 516）
2. 解决"让字符串成为回文串的最少插入次数"（LeetCode 1312）
3. 分析回文分割为什么不能用贪心（给出反例）

---

**上一章（Part 3）：** [泛洪填充](../part3-board-problems/12-flood-fill.md) | **下一章：** [复原 IP 地址](14-restore-ip.md)
