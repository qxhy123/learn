# 第7章：最长公共子序列（LCS）

## 7.1 问题定义

**最长公共子序列**（Longest Common Subsequence，LCS）：

```
给定两个字符串 text1 和 text2，
找出它们的最长公共子序列的长度。

子序列：不要求连续，但要求相对顺序不变。

text1 = "abcde"
text2 = "ace"
LCS = "ace"，长度为 3
```

**子序列 vs 子串**：
- 子序列：字符可以不连续（"ace" 是 "abcde" 的子序列）
- 子串：字符必须连续（"abc" 是 "abcde" 的子串）

---

## 7.2 状态设计

**状态**：`dp[i][j]` = `text1[0..i-1]` 和 `text2[0..j-1]` 的最长公共子序列长度

**转移方程**：

考虑 `text1[i-1]` 和 `text2[j-1]` 这两个字符：

```
情形1：text1[i-1] == text2[j-1]
  两个字符相同，可以同时纳入LCS
  dp[i][j] = dp[i-1][j-1] + 1

情形2：text1[i-1] != text2[j-1]
  两个字符不同，至少有一个不在LCS中
  dp[i][j] = max(dp[i-1][j],   # 不用 text1[i-1]
                 dp[i][j-1])    # 不用 text2[j-1]
```

**base case**：`dp[0][j] = dp[i][0] = 0`（空字符串与任何字符串的LCS为0）

---

## 7.3 图示理解

```
text1 = "abcde"
text2 = "ace"

    ""  a  c  e
""   0  0  0  0
a    0  1  1  1
b    0  1  1  1
c    0  1  2  2
d    0  1  2  2
e    0  1  2  3   ← 答案

填表规则：
- 如果 text1[i-1] == text2[j-1]：dp[i][j] = dp[i-1][j-1] + 1  （斜向+1）
- 否则：dp[i][j] = max(dp[i-1][j], dp[i][j-1])               （取上或左的最大）
```

---

## 7.4 代码实现

```python
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

print(longest_common_subsequence("abcde", "ace"))   # 3
print(longest_common_subsequence("abc", "abc"))     # 3
print(longest_common_subsequence("abc", "def"))     # 0
```

**时间复杂度**：O(mn)  
**空间复杂度**：O(mn)，可优化到 O(min(m,n))

---

## 7.5 还原实际的LCS

仅返回长度不够，如何还原实际的公共子序列？

**方法：回溯DP表**

```python
def lcs_with_trace(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # 回溯还原LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs.append(text1[i-1])  # 该字符在LCS中
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1  # 来自上方
        else:
            j -= 1  # 来自左方
    
    return dp[m][n], ''.join(reversed(lcs))

length, seq = lcs_with_trace("abcde", "ace")
print(f"LCS长度: {length}, LCS: {seq}")  # 3, ace
```

---

## 7.6 LCS 的变形题

### 变形1：最长公共子串（连续版）

```python
def longest_common_substring(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1  # 只有字符相等才能延续
                max_len = max(max_len, dp[i][j])
            # 字符不等时 dp[i][j] = 0（不继承，子串必须连续）
    
    return max_len
```

区别：子串版中字符不等时 `dp[i][j] = 0`（不可继承），子序列版中取 max。

### 变形2：删除操作使两字符串相同（LeetCode 583）

```
最少删除多少字符，使得两个字符串相同？
answer = len(text1) + len(text2) - 2 * LCS(text1, text2)
```

```python
def min_distance_delete(word1, word2):
    lcs_len = longest_common_subsequence(word1, word2)
    return len(word1) + len(word2) - 2 * lcs_len
```

### 变形3：不同的子序列（LeetCode 115）

```
s 中有多少个子序列等于 t？
s = "rabbbit", t = "rabbit"  → 3
```

```python
def num_distinct(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # base case：t 为空时，s 的任意前缀都有1个子序列等于空串
    for i in range(m + 1):
        dp[i][0] = 1
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j]  # 不用 s[i-1]
            if s[i-1] == t[j-1]:
                dp[i][j] += dp[i-1][j-1]  # 用 s[i-1] 匹配 t[j-1]
    
    return dp[m][n]

print(num_distinct("rabbbit", "rabbit"))  # 3
```

### 变形4：交错字符串（LeetCode 97）

```
s3 是否由 s1 和 s2 交错组成？
s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac" → True
```

```python
def is_interleave(s1, s2, s3):
    m, n = len(s1), len(s2)
    if m + n != len(s3):
        return False
    
    # dp[i][j] = s1[:i] 和 s2[:j] 能否交错组成 s3[:i+j]
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or \
                       (dp[i][j-1] and s2[j-1] == s3[i+j-1])
    
    return dp[m][n]
```

---

## 7.7 最短公共超序列（LCS的逆向应用）

**题目**（LeetCode 1092）：找最短的字符串，使得 s1 和 s2 都是它的子序列。

**公式**：最短公共超序列长度 = `len(s1) + len(s2) - LCS(s1, s2)`

```python
def shortest_common_supersequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # 回溯构造超序列
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            result.append(s1[i-1])
            i -= 1; j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            result.append(s1[i-1]); i -= 1
        else:
            result.append(s2[j-1]); j -= 1
    result.extend(reversed(s1[:i]))
    result.extend(reversed(s2[:j]))
    return ''.join(reversed(result))
```

---

## 7.8 本章小结

LCS 是序列DP的核心模型，记住：

```
text1[i-1] == text2[j-1]:  dp[i][j] = dp[i-1][j-1] + 1
text1[i-1] != text2[j-1]:  dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

**下一章：最长递增子序列（LIS）——另一个序列DP经典**

---

## LeetCode 推荐题目

- [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/) ⭐⭐
- [583. 两个字符串的删除操作](https://leetcode.cn/problems/delete-operation-for-two-strings/) ⭐⭐
- [115. 不同的子序列](https://leetcode.cn/problems/distinct-subsequences/) ⭐⭐⭐
- [97. 交错字符串](https://leetcode.cn/problems/interleaving-string/) ⭐⭐⭐
- [1092. 最短公共超序列](https://leetcode.cn/problems/shortest-common-supersequence/) ⭐⭐⭐
