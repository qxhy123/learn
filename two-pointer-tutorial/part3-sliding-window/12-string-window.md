# 第12章：字符串滑动窗口

## 12.1 字符串窗口的特殊性

字符串滑动窗口的核心挑战是**字符频率的维护**：不仅需要知道窗口内有哪些字符，还需要知道它们出现的次数，以判断是否满足条件。

常见问题类型：
1. **覆盖问题**：窗口内的字符是否包含目标字符集（含频率）
2. **排列问题**：窗口内的字符频率是否恰好等于目标
3. **变形问题**：允许有限次替换/删除后满足某条件的最长窗口

---

## 12.2 核心工具：need/have 模型

处理字符频率问题的经典框架：

```python
need  = Counter(target)   # 目标：每种字符需要多少个
have  = 0                 # 当前：已"满足需求"的字符种类数
required = len(need)      # 需要满足的字符种类总数
window = {}               # 当前窗口的字符频率
```

**关键判断**：字符 `c` 被"满足"的条件是 `window[c] >= need[c]`。

当 `have == required` 时，窗口包含了目标的所有字符（含频率）。

```python
# 加入字符 c 时
window[c] = window.get(c, 0) + 1
if c in need and window[c] == need[c]:
    have += 1   # 这种字符恰好满足（从不足到满足）

# 移出字符 c 时
if c in need and window[c] == need[c]:
    have -= 1   # 这种字符从满足变为不足
window[c] -= 1
```

---

## 12.3 最小覆盖子串

**问题（LeetCode 76）**：给定字符串 `s` 和 `t`，找 `s` 中涵盖 `t` 所有字符的最短子串。

```python
from collections import Counter

def min_window(s, t):
    if not t or not s:
        return ""

    need = Counter(t)
    required = len(need)

    left = 0
    have = 0
    window = {}
    min_len = float('inf')
    result = ""

    for right in range(len(s)):
        # 扩张：加入 s[right]
        c = s[right]
        window[c] = window.get(c, 0) + 1
        if c in need and window[c] == need[c]:
            have += 1

        # 收缩：当窗口满足条件时，尽量缩短
        while have == required:
            # 更新最短答案（收缩前记录）
            if right - left + 1 < min_len:
                min_len = right - left + 1
                result = s[left:right + 1]

            # 移出左端
            left_c = s[left]
            if left_c in need and window[left_c] == need[left_c]:
                have -= 1
            window[left_c] -= 1
            left += 1

    return result

# 测试
print(min_window("ADOBECODEBANC", "ABC"))  # "BANC"
print(min_window("a", "a"))               # "a"
print(min_window("a", "aa"))              # ""（s中只有一个a，不够）
```

**复杂度**：O(|s| + |t|)，left 和 right 各扫描一次。

---

## 12.4 字符串排列（字母异位词）

**问题（LeetCode 567）**：判断 `s2` 中是否存在 `s1` 的排列（字母异位词）。

```python
def check_inclusion(s1, s2):
    """使用 need/have 模型"""
    if len(s1) > len(s2):
        return False

    need = Counter(s1)
    required = len(need)
    window = {}
    have = 0
    left = 0

    for right in range(len(s2)):
        c = s2[right]
        window[c] = window.get(c, 0) + 1
        if c in need and window[c] == need[c]:
            have += 1

        # 定长窗口：保持大小为 len(s1)
        if right - left + 1 > len(s1):
            old_c = s2[left]
            if old_c in need and window[old_c] == need[old_c]:
                have -= 1
            window[old_c] -= 1
            left += 1

        if have == required:
            return True

    return False

# 测试
print(check_inclusion("ab", "eidbaooo"))  # True
print(check_inclusion("ab", "eidboaoo"))  # False
```

**另一种写法**（直接比较 Counter）：

```python
def check_inclusion_simple(s1, s2):
    k = len(s1)
    target = Counter(s1)
    window = Counter(s2[:k])

    if window == target:
        return True

    for right in range(k, len(s2)):
        window[s2[right]] += 1
        window[s2[right - k]] -= 1
        if window[s2[right - k]] == 0:
            del window[s2[right - k]]
        if window == target:
            return True

    return False
```

---

## 12.5 替换后最长重复字符

**问题（LeetCode 424）**：字符串中可以替换最多 k 个字符，求替换后最长的只含同一字母的子串。

```python
def character_replacement(s, k):
    """
    窗口合法条件：窗口大小 - 窗口内最多字符的频率 <= k
    即：需要替换的字符数 <= k
    """
    left = 0
    count = {}
    max_count = 0   # 窗口内最多字符的频率（只需保存最大值）
    max_len = 0

    for right in range(len(s)):
        c = s[right]
        count[c] = count.get(c, 0) + 1
        max_count = max(max_count, count[c])

        # 违规条件：需要替换的字符数 > k
        window_size = right - left + 1
        if window_size - max_count > k:
            # 收缩左端
            count[s[left]] -= 1
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len

# 测试
print(character_replacement("ABAB", 2))   # 4（全替换为A或B）
print(character_replacement("AABABBA", 1)) # 4
```

**关键洞察**：`max_count` 只需维护历史最大值，不需要在收缩时更新。

**证明**：如果 `max_count` 减小了，答案也会减小（因为 `window_size - max_count` 增大），所以收缩后的窗口不会比收缩前更优。这使得我们可以安全地跳过 max_count 的更新。

---

## 12.6 最长含有效括号子串

**用滑动窗口思想处理括号**：

```python
def longest_valid_parentheses(s):
    """
    对撞指针：分别从左到右和从右到左扫描
    """
    def count_valid(direction):
        left_count = right_count = max_len = 0
        if direction == 'left':
            iterable = s
        else:
            iterable = reversed(s)
            # 方向反转时，'(' 和 ')' 的角色互换

        open_char = '(' if direction == 'left' else ')'
        close_char = ')' if direction == 'left' else '('

        for c in iterable:
            if c == open_char:
                left_count += 1
            else:
                right_count += 1

            if left_count == right_count:
                max_len = max(max_len, 2 * right_count)
            elif right_count > left_count:
                left_count = right_count = 0

        return max_len

    return max(count_valid('left'), count_valid('right'))

# 测试
print(longest_valid_parentheses("(()"))   # 2
print(longest_valid_parentheses(")()())")) # 4
print(longest_valid_parentheses(""))       # 0
```

---

## 12.7 多模式字符串窗口

**问题**：找字符串 `s` 中包含列表 `words` 中所有单词（每个单词恰好出现一次）的所有起始下标。每个单词长度相同。

```python
def find_substring(s, words):
    if not s or not words:
        return []

    word_len = len(words[0])
    word_count = len(words)
    total_len = word_len * word_count
    need = Counter(words)
    result = []

    # 对每个可能的起始偏移（0 到 word_len-1）分别做滑动窗口
    for offset in range(word_len):
        left = offset
        window = {}
        matched = 0

        for right in range(offset, len(s) - word_len + 1, word_len):
            word = s[right:right + word_len]

            if word in need:
                window[word] = window.get(word, 0) + 1
                if window[word] == need[word]:
                    matched += 1

                # 超出需要：收缩左端
                while window[word] > need[word]:
                    left_word = s[left:left + word_len]
                    if window[left_word] == need[left_word]:
                        matched -= 1
                    window[left_word] -= 1
                    left += word_len
            else:
                # 当前单词不在 words 中，窗口重置
                window.clear()
                matched = 0
                left = right + word_len

            if matched == len(need):
                result.append(left)

    return result

# 测试
print(find_substring("barfoothefoobarman", ["foo","bar"]))  # [0, 9]
print(find_substring("wordgoodgoodgoodbestword", ["word","good","best","word"]))  # []
```

---

## 12.8 本章小结

字符串滑动窗口的核心模型：

| 模型 | 适用场景 | 关键数据结构 |
|------|----------|--------------|
| need/have | 覆盖、包含 | Counter + have 计数 |
| 定长比较 | 异位词、排列 | Counter 比较 |
| 替换窗口 | 允许 k 次替换 | 最大频率计数 |
| 多偏移窗口 | 固定长单词匹配 | 分偏移处理 |

**下一章：链表双指针高级操作——Floyd 算法的深度应用**

---

## LeetCode 推荐题目

- [76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/) ⭐⭐⭐
- [567. 字符串的排列](https://leetcode.cn/problems/permutation-in-string/) ⭐⭐
- [438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/) ⭐⭐
- [424. 替换后的最长重复字符](https://leetcode.cn/problems/longest-repeating-character-replacement/) ⭐⭐⭐
- [30. 串联所有单词的子串](https://leetcode.cn/problems/substring-with-concatenation-of-all-words/) ⭐⭐⭐
