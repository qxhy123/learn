# 第9章：滑动窗口核心框架

## 9.1 什么是滑动窗口？

滑动窗口（Sliding Window）是同向双指针的一种特殊形式，专门处理**连续子数组/子字符串**问题。

"窗口"由左右两个指针 `[left, right]` 界定，代表当前考察的连续区间：

```
数组：[1, 3, -1, -3, 5, 3, 6, 7]
       ←  窗口  →
      left    right
```

**扩张**：`right += 1`，窗口向右延伸，纳入新元素
**收缩**：`left += 1`，窗口从左缩短，移出旧元素

---

## 9.2 为什么滑动窗口是 O(n)？

关键在于：**right 和 left 都只向右移动，每个元素最多被加入窗口一次，也最多被移出窗口一次**。

```
right 移动次数 ≤ n
left  移动次数 ≤ n
总操作次数    ≤ 2n → O(n)
```

对比暴力枚举所有子数组：O(n²) 个子数组，每个验证 O(n)，总体 O(n³)。

---

## 9.3 核心框架：扩张-检查-收缩

所有滑动窗口问题都符合以下模型：

```python
def sliding_window_template(nums):
    left = 0
    window_state = ...  # 窗口内的统计信息（和、计数、频率表等）
    result = ...        # 最终结果

    for right in range(len(nums)):
        # === 扩张：将 nums[right] 加入窗口 ===
        window_state = update_with(window_state, nums[right])

        # === 检查并收缩：维护窗口不变量 ===
        while window_violates_constraint(window_state):
            # 将 nums[left] 移出窗口
            window_state = remove_from(window_state, nums[left])
            left += 1

        # === 更新答案 ===
        result = update_result(result, right - left + 1)

    return result
```

**三个关键决策**：
1. **窗口状态**：用什么数据结构维护窗口内的信息？
2. **约束条件**：什么情况下窗口"违规"，需要收缩？
3. **答案更新**：在哪个时机更新答案（收缩前/后）？

---

## 9.4 框架变体：最小窗口 vs 最大窗口

**最大窗口**（找满足条件的最长子数组）：

```python
def max_window_template(nums):
    left = 0
    state = initial_state()
    max_len = 0

    for right in range(len(nums)):
        state = add(state, nums[right])          # 扩张

        while not valid(state):                  # 违规则收缩
            state = remove(state, nums[left])
            left += 1

        max_len = max(max_len, right - left + 1) # 收缩后窗口合法，更新最大值

    return max_len
```

**最小窗口**（找满足条件的最短子数组）：

```python
def min_window_template(nums):
    left = 0
    state = initial_state()
    min_len = float('inf')

    for right in range(len(nums)):
        state = add(state, nums[right])          # 扩张

        while valid(state):                      # 满足条件则尝试收缩
            min_len = min(min_len, right - left + 1)  # 更新最小值
            state = remove(state, nums[left])
            left += 1

    return min_len if min_len != float('inf') else 0
```

**关键区别**：
- 最大窗口：收缩**后**更新答案（保证窗口合法）
- 最小窗口：收缩**前**更新答案（满足条件时才记录）

---

## 9.5 窗口状态的数据结构选择

| 问题类型 | 窗口状态 | 数据结构 |
|----------|----------|----------|
| 子数组之和 | 累积和 | 整数 |
| 不超过K个不同元素 | 元素频率 | 哈希表/Counter |
| 无重复字符 | 字符出现情况 | 集合/哈希表 |
| 最大/最小值 | 极值 | 单调队列（见第15章） |
| 乘积限制 | 累积积 | 整数 |

---

## 9.6 第一个完整例子：无重复字符的最长子串

```python
def length_of_longest_substring(s):
    left = 0
    char_count = {}   # 窗口内字符频率
    max_len = 0

    for right in range(len(s)):
        # 扩张：加入 s[right]
        char_count[s[right]] = char_count.get(s[right], 0) + 1

        # 收缩：有重复字符时，移出 left 直到无重复
        while char_count[s[right]] > 1:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1

        # 更新最大长度（此时窗口内无重复）
        max_len = max(max_len, right - left + 1)

    return max_len

# 测试
print(length_of_longest_substring("abcabcbb"))  # 3（"abc"）
print(length_of_longest_substring("bbbbb"))     # 1（"b"）
print(length_of_longest_substring("pwwkew"))    # 3（"wke"）
```

**优化版**（用哈希表直接跳跃，而不是逐步移动 left）：

```python
def length_of_longest_substring_opt(s):
    last_seen = {}  # 字符 -> 最后出现的位置
    left = 0
    max_len = 0

    for right, char in enumerate(s):
        if char in last_seen and last_seen[char] >= left:
            left = last_seen[char] + 1  # 直接跳到重复字符的下一位
        last_seen[char] = right
        max_len = max(max_len, right - left + 1)

    return max_len
```

**注意**：优化版中必须检查 `last_seen[char] >= left`，否则会错误地将已经移出窗口的字符位置当作有效位置。

---

## 9.7 框架正确性：为什么 while 收缩是安全的？

关键在于**窗口不变量的单调性**：

> 当 right 增大时，使窗口满足条件所需的 left 最小值是单调不减的。

**证明**：设 right=r 时，最小合法 left 为 `f(r)`。当 right=r+1 时，最小合法 left 为 `f(r+1)` 且 `f(r+1) >= f(r)`。

这保证了 left 永远不需要向左回退，使得整体是 O(n) 的。

如果 left 需要回退，则不能用滑动窗口，需要其他方法。

---

## 9.8 常见错误与调试

```python
# 错误1：忘记收缩时更新窗口状态
while not valid(state):
    left += 1  # 错！忘了 remove state
    # 正确：
    state = remove(state, nums[left])
    left += 1

# 错误2：最小窗口更新时机错误
for right in ...:
    state = add(state, nums[right])
    while valid(state):
        left += 1                          # 错！先移动再更新
        min_len = min(min_len, right - left + 1)
    # 正确：先记录，再收缩
    while valid(state):
        min_len = min(min_len, right - left + 1)
        state = remove(state, nums[left])
        left += 1

# 错误3：窗口长度计算
length = right - left + 1  # ✓（包含 right 和 left 本身）
length = right - left      # ✗（漏掉了一个元素）
```

---

## 9.9 本章小结

滑动窗口的核心框架：

```
for right in range(n):
    加入 nums[right] 到窗口
    while 窗口违规:
        移出 nums[left]
        left += 1
    更新答案
```

两个变体：
- **最大窗口**：收缩后更新（窗口合法时才是候选答案）
- **最小窗口**：收缩前更新（满足条件的每个状态都是候选答案）

**下一章：定长滑动窗口——差量更新的艺术**

---

## LeetCode 推荐题目

- [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/) ⭐⭐
- [219. 存在重复元素 II](https://leetcode.cn/problems/contains-duplicate-ii/) ⭐
- [643. 子数组最大平均数 I](https://leetcode.cn/problems/maximum-average-subarray-i/) ⭐（定长，预习）
