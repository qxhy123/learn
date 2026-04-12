# 第9章：编辑距离与字符串DP

## 9.1 编辑距离——字符串DP的皇冠

**题目**（LeetCode 72）：

```
给定两个字符串 word1 和 word2，
用最少的操作次数将 word1 转换成 word2。
允许的操作：插入、删除、替换（每次操作一个字符）。

word1 = "horse", word2 = "ros"
操作：horse → rorse（替换 h→r）
        → rose（删除 r）
        → ros（删除 e）
答案：3
```

这就是著名的 **Levenshtein 距离**，在拼写检查、DNA序列比对、版本控制差异算法中广泛应用。

---

## 9.2 状态设计

**状态**：`dp[i][j]` = 将 `word1[0..i-1]` 转换为 `word2[0..j-1]` 所需的最少操作数

**base case**：
- `dp[i][0] = i`：word1 前 i 字符转换为空串，需要 i 次删除
- `dp[0][j] = j`：空串转换为 word2 前 j 字符，需要 j 次插入

**转移方程**：

考虑 `word1[i-1]` 和 `word2[j-1]`：

```
情形1：word1[i-1] == word2[j-1]（字符相同，无需操作）
  dp[i][j] = dp[i-1][j-1]

情形2：word1[i-1] != word2[j-1]（字符不同，有三种操作）
  替换：dp[i-1][j-1] + 1  （将 word1[i-1] 替换成 word2[j-1]）
  删除：dp[i-1][j]   + 1  （删除 word1[i-1]，问题变为 word1[:i-1] → word2[:j]）
  插入：dp[i][j-1]   + 1  （在 word1 末尾插入 word2[j-1]）
  
  dp[i][j] = min(替换, 删除, 插入)
```

---

## 9.3 图示理解

```
word1 = "horse", word2 = "ros"

     ""  r  o  s
""    0  1  2  3
h     1  1  2  3
o     2  2  1  2
r     3  2  2  2
s     4  3  3  2
e     5  4  4  3  ← 答案 dp[5][3] = 3
```

---

## 9.4 代码实现

```python
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # base case
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j-1],  # 替换
                    dp[i-1][j],    # 删除
                    dp[i][j-1]     # 插入
                )
    
    return dp[m][n]

print(edit_distance("horse", "ros"))    # 3
print(edit_distance("intention", "execution"))  # 5
```

---

## 9.5 编辑距离的实际操作还原

```python
def edit_distance_trace(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    
    # 回溯操作序列
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and word1[i-1] == word2[j-1]:
            i -= 1; j -= 1  # 无操作
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(f"替换 word1[{i-1}]='{word1[i-1]}' → '{word2[j-1]}'")
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(f"删除 word1[{i-1}]='{word1[i-1]}'")
            i -= 1
        else:
            ops.append(f"插入 '{word2[j-1]}'")
            j -= 1
    
    return dp[m][n], list(reversed(ops))

dist, ops = edit_distance_trace("horse", "ros")
print(f"编辑距离: {dist}")
for op in ops:
    print(f"  {op}")
```

---

## 9.6 字符串DP综合：回文系列

### 最长回文子序列（LeetCode 516）

```
给定字符串 s，找出最长回文子序列的长度。
s = "bbbab"  → 4（"bbbb"）
```

**思路1**：LCS(s, reverse(s))

```python
def longest_palindrome_subseq(s):
    return longest_common_subsequence(s, s[::-1])
```

**思路2**：区间DP

```python
def longest_palindrome_subseq_dp(s):
    n = len(s)
    # dp[i][j] = s[i..j] 的最长回文子序列长度
    dp = [[0] * n for _ in range(n)]
    
    # base case: 单个字符
    for i in range(n):
        dp[i][i] = 1
    
    # 按长度递增填表
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    
    return dp[0][n-1]
```

### 最少插入次数使字符串成为回文（LeetCode 1312）

```python
def min_insertions_palindrome(s):
    n = len(s)
    # dp[i][j] = 使 s[i..j] 成为回文所需的最少插入次数
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1]
            else:
                dp[i][j] = min(dp[i+1][j], dp[i][j-1]) + 1
    
    return dp[0][n-1]
```

### 最少分割次数使每段都是回文（LeetCode 132）

```python
def min_cut(s):
    n = len(s)
    
    # 预计算所有回文子串
    is_palindrome = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i <= 2 or is_palindrome[i+1][j-1]):
                is_palindrome[i][j] = True
    
    # dp[i] = s[0..i] 的最少分割次数
    dp = list(range(n))  # 最多切 i 次（每个字符单独一段）
    
    for i in range(1, n):
        if is_palindrome[0][i]:  # s[0..i] 本身是回文
            dp[i] = 0
            continue
        for j in range(1, i + 1):
            if is_palindrome[j][i]:
                dp[i] = min(dp[i], dp[j-1] + 1)
    
    return dp[n-1]
```

---

## 9.7 字符串通配符匹配（LeetCode 44）

```
'?' 匹配任意单个字符，'*' 匹配任意字符序列（包括空序列）。
s = "adceb", p = "*a*b"  → True
```

```python
def is_match_wildcard(s, p):
    m, n = len(s), len(p)
    # dp[i][j] = s[:i] 与 p[:j] 是否匹配
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # 处理 p 以 * 开头的情况
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                # * 匹配空：dp[i][j-1]
                # * 匹配一个字符并继续：dp[i-1][j]
                dp[i][j] = dp[i][j-1] or dp[i-1][j]
            elif p[j-1] == '?' or s[i-1] == p[j-1]:
                dp[i][j] = dp[i-1][j-1]
    
    return dp[m][n]
```

---

## 9.8 正则表达式匹配（LeetCode 10）

```
'.' 匹配任意单个字符，'*' 匹配零个或多个前面的元素。
s = "aab", p = "c*a*b"  → True
```

```python
def is_match_regex(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # 处理 p 中形如 a* 或 .* 的前缀
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]  # a* 可以匹配空
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                # x* 匹配零次：dp[i][j-2]
                # x* 匹配一次以上：前一个字符匹配且 dp[i-1][j]
                zero_match = dp[i][j-2]
                one_plus_match = dp[i-1][j] and (p[j-2] == '.' or p[j-2] == s[i-1])
                dp[i][j] = zero_match or one_plus_match
            elif p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
    
    return dp[m][n]
```

---

## 9.9 本章小结

| 问题 | 状态定义 | 核心转移 |
|------|---------|---------|
| 编辑距离 | `dp[i][j]` = word1[:i]→word2[:j] 的最少操作 | 替换/删除/插入取min |
| 最长公共子串 | `dp[i][j]` = 以i,j结尾的公共子串长度 | 相等则+1，否则0 |
| 最长回文子序列 | `dp[i][j]` = s[i..j] 的最长回文子序列 | 首尾相等则+2 |
| 通配符匹配 | `dp[i][j]` = s[:i]与p[:j]是否匹配 | *号分情况讨论 |

---

## LeetCode 推荐题目

- [72. 编辑距离](https://leetcode.cn/problems/edit-distance/) ⭐⭐⭐
- [516. 最长回文子序列](https://leetcode.cn/problems/longest-palindromic-subsequence/) ⭐⭐
- [132. 分割回文串 II](https://leetcode.cn/problems/palindrome-partitioning-ii/) ⭐⭐⭐
- [44. 通配符匹配](https://leetcode.cn/problems/wildcard-matching/) ⭐⭐⭐
- [10. 正则表达式匹配](https://leetcode.cn/problems/regular-expression-matching/) ⭐⭐⭐
