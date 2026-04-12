# 第19章：最小覆盖子串与复杂字符串

## 19.1 本章定位

第12章介绍了字符串滑动窗口的基础框架。本章深入探讨更复杂的字符串问题，包括：
- 多约束条件的窗口（需要同时满足多个字符条件）
- 不等长模式匹配
- 字符串变换类滑动窗口
- 经典的 need/have 模型的极限应用

---

## 19.2 最小覆盖子串：深度剖析

**LeetCode 76**：给定字符串 `s` 和 `t`，找 `s` 中覆盖 `t` 所有字符（含重复）的最短子串。

上一章给出了实现，这里深入分析边界情况和优化。

```python
from collections import Counter

def min_window(s, t):
    need = Counter(t)
    missing = len(t)   # 还需要满足的字符总数（含重复计数）
    # 注意：missing 和 have/required 的区别
    # missing 计总量，更简洁

    start = 0
    result = (float('inf'), 0, 0)  # (长度, 起点, 终点)

    j = 0  # 窗口左端（收缩指针）

    for i, c in enumerate(s, 1):  # i 是右端+1（1-indexed 更方便切片）
        if need[c] > 0:
            missing -= 1   # 还差的字符减少（只有 need[c]>0 时才是真正需要的）
        need[c] -= 1

        if missing == 0:   # 窗口满足条件
            # 收缩左端：跳过不需要的字符
            while need[s[j]] < 0:
                need[s[j]] += 1
                j += 1

            # 更新答案
            if i - j < result[0]:
                result = (i - j, j, i)

            # 移出左端（为下次扩展做准备）
            need[s[j]] += 1
            missing += 1
            j += 1

    return s[result[1]:result[2]] if result[0] != float('inf') else ""

# 测试
print(min_window("ADOBECODEBANC", "ABC"))   # "BANC"
print(min_window("a", "a"))                 # "a"
print(min_window("a", "aa"))                # ""
print(min_window("aa", "aa"))               # "aa"
```

**missing 版 vs have/required 版的对比**：

```
missing 版：
  - missing = len(t)（总字符数，含重复）
  - 加入字符时：若 need[c] > 0，missing--
  - 条件：missing == 0

have/required 版：
  - required = len(need)（不同字符种类数）
  - 加入字符时：若 window[c] == need[c]，have++
  - 条件：have == required
```

`missing` 版代码更简洁；`have/required` 版逻辑更清晰。选择取决于个人偏好。

---

## 19.3 最长字符串链

**问题（LeetCode 1048）**：给定单词列表，找最长的单词链（每次可以在任意位置添加一个字母）。

```python
def longest_str_chain(words):
    """DP + 双指针思想：枚举每个单词的所有前身"""
    words.sort(key=len)
    dp = {}  # word -> 以该单词结尾的最长链

    for word in words:
        best = 1
        # 枚举删除一个字母后的所有可能前身
        for i in range(len(word)):
            prev = word[:i] + word[i+1:]
            if prev in dp:
                best = max(best, dp[prev] + 1)
        dp[word] = best

    return max(dp.values())

# 测试
print(longest_str_chain(["a","b","ba","bca","bda","bdca"]))  # 4
print(longest_str_chain(["xbc","pcxbcf","xb","cxbc","pcxbc"]))  # 5
```

---

## 19.4 字符串的所有回文分割

**问题（LeetCode 131）**：将字符串分割成若干回文子串，返回所有可能的分割方案。

```python
def partition_palindrome(s):
    n = len(s)

    # 预处理：中心扩展，标记所有回文子串
    is_pal = [[False] * n for _ in range(n)]
    for center in range(2 * n - 1):
        left = center // 2
        right = left + center % 2
        while left >= 0 and right < n and s[left] == s[right]:
            is_pal[left][right] = True
            left -= 1
            right += 1

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
print(partition_palindrome("aab"))
# [['a', 'a', 'b'], ['aa', 'b']]
```

**双指针在预处理阶段**：中心扩展法用双指针标记所有回文子串，是后续回溯的基础。

---

## 19.5 最长有效括号

**问题（LeetCode 32）**：找最长有效括号子串。

```python
def longest_valid_parentheses(s):
    """
    方法：双向扫描（双指针思想）
    从左到右扫描，统计 left/right 计数
    从右到左再扫描一遍，处理 left 更多的情况
    """
    def scan(direction):
        left = right = max_len = 0
        chars = s if direction == 'left' else reversed(s)
        open_c, close_c = ('(', ')') if direction == 'left' else (')', '(')

        for c in chars:
            if c == open_c:
                left += 1
            else:
                right += 1

            if left == right:
                max_len = max(max_len, 2 * right)
            elif right > left:
                left = right = 0  # 重置（右括号太多，前面无论如何都不合法）

        return max_len

    return max(scan('left'), scan('right'))

# 测试
print(longest_valid_parentheses("(()"))      # 2
print(longest_valid_parentheses(")()())"))   # 4
print(longest_valid_parentheses(""))         # 0
print(longest_valid_parentheses("()(())))")) # 6
```

**为什么需要两次扫描**：
- 左→右扫描能处理形如 `(()` 的情况（多余的左括号）
- 右→左扫描能处理形如 `())` 的情况（多余的右括号）
- 取两次结果的最大值

---

## 19.6 单词搜索（双指针+回溯）

**问题（LeetCode 79）**：在矩阵中找是否存在某个单词的路径。

```python
def exist(board, word):
    rows, cols = len(board), len(board[0])
    n = len(word)

    def dfs(r, c, idx):
        if idx == n:
            return True
        if (r < 0 or r >= rows or c < 0 or c >= cols
                or board[r][c] != word[idx]):
            return False

        temp = board[r][c]
        board[r][c] = '#'   # 标记已使用（原地修改）

        found = (dfs(r+1, c, idx+1) or dfs(r-1, c, idx+1)
              or dfs(r, c+1, idx+1) or dfs(r, c-1, idx+1))

        board[r][c] = temp  # 恢复
        return found

    return any(dfs(r, c, 0) for r in range(rows) for c in range(cols))

# 测试
board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
print(exist(board, "ABCCED"))  # True
print(exist(board, "SEE"))     # True
print(exist(board, "ABCB"))    # False
```

---

## 19.7 字符串匹配：KMP 与双指针

KMP 算法本质上是双指针在字符串匹配中的应用：`i` 扫描文本，`j` 跟踪模式串的匹配进度。

```python
def kmp_search(text, pattern):
    """KMP 字符串匹配：O(n+m)"""
    if not pattern:
        return [0]

    # 构建部分匹配表（failure function）
    m = len(pattern)
    fail = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = fail[j-1]
        if pattern[i] == pattern[j]:
            j += 1
        fail[i] = j

    # 搜索
    result = []
    j = 0
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = fail[j-1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            result.append(i - m + 1)
            j = fail[j-1]

    return result

# 测试
print(kmp_search("ABABDABACDABABCABAB", "ABABCABAB"))  # [10]
print(kmp_search("AAAAABAAABA", "AAAA"))               # [0, 1]
```

**双指针视角**：`i` 是文本指针（只前进），`j` 是模式指针（可以通过 fail 表跳跃，但不后退超过已有位置）。整体 O(n+m) 是因为 `j` 的跳跃次数被 `i` 的前进次数限制。

---

## 19.8 本章小结

复杂字符串问题的双指针模式：

| 问题类型 | 核心技术 | 复杂度 |
|----------|----------|--------|
| 最小覆盖子串 | need/have + 滑动窗口 | O(|s| + |t|) |
| 回文预处理 | 中心扩展双指针 | O(n²) |
| 最长有效括号 | 双向扫描 | O(n) |
| 字符串匹配 | KMP 双指针 | O(n+m) |

**下一章：竞赛级综合题——双指针与其他算法的深度融合**

---

## LeetCode 推荐题目

- [76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/) ⭐⭐⭐
- [32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/) ⭐⭐⭐
- [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/) ⭐⭐
- [79. 单词搜索](https://leetcode.cn/problems/word-search/) ⭐⭐
- [28. 找出字符串中第一个匹配项的下标](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/) ⭐⭐
- [1048. 最长字符串链](https://leetcode.cn/problems/longest-string-chain/) ⭐⭐⭐
