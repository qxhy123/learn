# 第8章：回文串与字符串双指针

## 8.1 什么是回文？

回文是正读和反读都相同的序列：`"racecar"`、`[1,2,1]`、`"A man a plan a canal Panama"`。

判断回文的本质是：**对称性验证**——第 i 个字符与第 n-1-i 个字符相同。

双指针天然契合这种对称结构：左指针从头，右指针从尾，向中心逼近。

---

## 8.2 最长回文子串：中心扩展法

**问题**：给定字符串 `s`，找出最长的回文子串。

**暴力 O(n³)**：枚举所有子串，逐个验证。

**DP O(n²) 时间 + O(n²) 空间**：`dp[i][j]` 表示 `s[i..j]` 是否是回文。

**中心扩展法 O(n²) 时间 + O(1) 空间**：以每个字符（或相邻字符间隙）为中心，向两侧扩展。

```python
def longest_palindrome(s):
    def expand_around_center(left, right):
        """从中心向外扩展，返回最长回文的左右边界"""
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        # 循环结束时 s[left] != s[right]（或越界）
        # 实际回文是 s[left+1..right-1]
        return left + 1, right - 1

    start, end = 0, 0

    for i in range(len(s)):
        # 奇数长度：以 s[i] 为中心
        l1, r1 = expand_around_center(i, i)
        # 偶数长度：以 s[i] 和 s[i+1] 之间为中心
        l2, r2 = expand_around_center(i, i + 1)

        if r1 - l1 > end - start:
            start, end = l1, r1
        if r2 - l2 > end - start:
            start, end = l2, r2

    return s[start:end + 1]

# 测试
print(longest_palindrome("babad"))   # "bab" 或 "aba"
print(longest_palindrome("cbbd"))    # "bb"
print(longest_palindrome("racecar")) # "racecar"
```

**为什么要分奇偶两种中心？**

```
奇数回文："aba"，中心是 'b'（单字符）
偶数回文："abba"，中心是 'bb' 之间的间隙（两字符）
```

每个位置扩展两次，共 2n 次扩展，每次最多扩展 n/2 步，总体 O(n²)。

---

## 8.3 中心扩展的通用模板

```python
def expand_center(s, left, right):
    """向外扩展直到不满足回文条件，返回最长回文范围"""
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return left + 1, right - 1  # 回退一步是实际的回文范围

# 使用
for i in range(len(s)):
    # 奇数中心
    l, r = expand_center(s, i, i)
    # 偶数中心
    l, r = expand_center(s, i, i + 1)
```

**边界细节**：扩展失败后 `left` 和 `right` 各越界一步，所以实际范围是 `[left+1, right-1]`。

---

## 8.4 Manacher 算法：O(n) 最长回文（进阶）

中心扩展是 O(n²)，存在大量重复计算。Manacher 算法利用已知回文的对称性避免重复扩展，达到 O(n)。

**核心思想**：

1. 预处理：在字符间插入分隔符，统一处理奇偶（`"abc"` → `"#a#b#c#"`）
2. 维护一个"当前已知最右回文"的右边界 `R` 和中心 `C`
3. 利用对称性：位置 `i` 关于 `C` 的镜像 `i' = 2*C - i`，`P[i]` 可以从 `P[i']` 初始化

```python
def manacher(s):
    # 预处理：插入分隔符
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n  # p[i] = 以 t[i] 为中心的回文半径

    center = right = 0  # 当前最右回文的中心和右边界

    for i in range(n):
        if i < right:
            mirror = 2 * center - i
            p[i] = min(right - i, p[mirror])  # 利用对称性初始化

        # 尝试扩展
        l, r = i - p[i] - 1, i + p[i] + 1
        while l >= 0 and r < n and t[l] == t[r]:
            p[i] += 1
            l -= 1
            r += 1

        # 更新最右回文
        if i + p[i] > right:
            center = i
            right = i + p[i]

    # 找最大值
    max_len, max_center = max((p[i], i) for i in range(n))
    start = (max_center - max_len) // 2  # 映射回原字符串
    return s[start:start + max_len]

# 测试
print(manacher("babad"))    # "bab"
print(manacher("cbbd"))     # "bb"
print(manacher("racecar"))  # "racecar"
```

Manacher 是字符串算法中的精华，但面试中通常接受中心扩展法（O(n²)）。

---

## 8.5 回文分割

**问题**：将字符串分割成若干回文子串，求最少分割次数。

```python
def min_cut(s):
    n = len(s)

    # 预处理：is_palindrome[i][j] 表示 s[i..j] 是否是回文
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i <= 2 or is_pal[i+1][j-1]):
                is_pal[i][j] = True

    # dp[i] = s[0..i] 的最少分割次数
    dp = list(range(n))  # 最坏情况：每个字符单独分割

    for i in range(1, n):
        if is_pal[0][i]:
            dp[i] = 0
            continue
        for j in range(1, i + 1):
            if is_pal[j][i]:
                dp[i] = min(dp[i], dp[j-1] + 1)

    return dp[n-1]

# 测试
print(min_cut("aab"))    # 1（"aa" + "b"）
print(min_cut("a"))      # 0
print(min_cut("ab"))     # 1
```

---

## 8.6 字符串双指针：反转操作

**反转单词顺序**（保持每个单词内部顺序不变）：

```python
def reverse_words(s):
    """先全局反转，再逐词反转"""
    chars = list(s.strip().split())
    # 方法：内置split处理多余空格，join后翻转
    return ' '.join(reversed(chars))

def reverse_words_inplace(s):
    """真正原地操作（C语言风格）"""
    chars = list(s)
    n = len(chars)

    def reverse(l, r):
        while l < r:
            chars[l], chars[r] = chars[r], chars[l]
            l += 1
            r -= 1

    # 1. 全局反转
    reverse(0, n - 1)

    # 2. 逐词反转
    start = 0
    for i in range(n + 1):
        if i == n or chars[i] == ' ':
            reverse(start, i - 1)
            start = i + 1

    # 3. 处理多余空格（实际题目可能需要）
    return ''.join(chars).strip()

print(reverse_words("the sky is blue"))  # "blue is sky the"
```

**反转字符串中的每个单词**（单词内部反转，顺序不变）：

```python
def reverse_each_word(s):
    return ' '.join(word[::-1] for word in s.split())

print(reverse_each_word("Let's take LeetCode contest"))
# "s'teL ekat edoCteeL tsetnoc"
```

---

## 8.7 双指针在字符串比较中的应用

**最长公共前缀**：

```python
def longest_common_prefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        # 对撞指针：同步扫描两个字符串
        i = 0
        while i < len(prefix) and i < len(s) and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            return ""
    return prefix

print(longest_common_prefix(["flower","flow","flight"]))  # "fl"
```

**字符串压缩（双指针计数）**：

```python
def compress(chars):
    """原地压缩字符串，如 ['a','a','b'] → ['a','2','b']"""
    write = 0    # 写指针
    read = 0     # 读指针

    while read < len(chars):
        char = chars[read]
        count = 0

        # 计数连续相同字符
        while read < len(chars) and chars[read] == char:
            read += 1
            count += 1

        # 写字符
        chars[write] = char
        write += 1

        # 写数量（若大于1）
        if count > 1:
            for digit in str(count):
                chars[write] = digit
                write += 1

    return write

# 测试
chars = ['a','a','b','b','c','c','c']
print(compress(chars), chars[:compress(chars)])  # 6 ['a','2','b','2','c','3']
```

---

## 8.8 本章小结

字符串双指针的核心模式：

| 模式 | 典型题目 | 关键技术 |
|------|----------|----------|
| 中心扩展 | 最长回文子串 | 奇偶两种中心，向外扩展 |
| 对称验证 | 验证回文串 | 从两端向中间 |
| 翻转重组 | 反转单词 | 全局翻转+局部翻转 |
| 写指针 | 字符串压缩 | slow/fast 双指针 |

**下一章：滑动窗口核心框架——连续子数组问题的统一解法**

---

## LeetCode 推荐题目

- [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/) ⭐⭐
- [647. 回文子串](https://leetcode.cn/problems/palindromic-substrings/) ⭐⭐
- [151. 反转字符串中的单词](https://leetcode.cn/problems/reverse-words-in-a-string/) ⭐⭐
- [14. 最长公共前缀](https://leetcode.cn/problems/longest-common-prefix/) ⭐
- [443. 压缩字符串](https://leetcode.cn/problems/string-compression/) ⭐⭐
- [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/) ⭐⭐
