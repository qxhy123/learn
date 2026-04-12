# 第11章：变长滑动窗口

## 11.1 变长窗口的两类目标

变长滑动窗口（Variable-size Sliding Window）的窗口大小随条件动态变化。有两类目标：

| 目标 | 描述 | 收缩时机 |
|------|------|----------|
| **最长子数组** | 找满足条件的最长连续子数组 | 窗口违规时收缩，收缩后记录长度 |
| **最短子数组** | 找满足条件的最短连续子数组 | 窗口满足条件时收缩，收缩前记录长度 |

这两类的框架完全对称，但收缩时机相反——这是变长窗口最容易混淆的地方。

---

## 11.2 最长子数组：和不超过 k

**问题**：给定正整数数组 `nums`，找和不超过 `k` 的最长连续子数组长度。

```python
def max_length_subarray_sum_le_k(nums, k):
    left = 0
    window_sum = 0
    max_len = 0

    for right in range(len(nums)):
        window_sum += nums[right]             # 扩张

        while window_sum > k:                 # 违规：和超过 k
            window_sum -= nums[left]
            left += 1

        max_len = max(max_len, right - left + 1)  # 收缩后窗口合法，更新答案

    return max_len

# 测试
print(max_length_subarray_sum_le_k([3, 1, 2, 7, 4, 2, 1, 1, 5], 8))  # 4
```

**注意**：本题中 `nums` 必须是正整数，否则单调性不成立（加入负数可能使和减小，left 可能需要回退）。

---

## 11.3 最短子数组：和至少为 k

**问题**：给定正整数数组 `nums`，找和至少为 `k` 的最短连续子数组长度。

```python
def min_length_subarray_sum_ge_k(nums, k):
    left = 0
    window_sum = 0
    min_len = float('inf')

    for right in range(len(nums)):
        window_sum += nums[right]             # 扩张

        while window_sum >= k:                # 满足条件：尝试缩短
            min_len = min(min_len, right - left + 1)  # 先记录，再收缩
            window_sum -= nums[left]
            left += 1

    return min_len if min_len != float('inf') else 0

# 测试
print(min_length_subarray_sum_ge_k([2, 3, 1, 2, 4, 3], 7))  # 2（[4,3]）
print(min_length_subarray_sum_ge_k([1, 4, 4], 4))            # 1（[4]）
print(min_length_subarray_sum_ge_k([1, 1, 1, 1, 1, 1, 1, 1], 11))  # 0（无解）
```

**错误陷阱**：如果先收缩再记录，会漏掉最优解：

```python
# 错误版本
while window_sum >= k:
    window_sum -= nums[left]   # 先收缩
    left += 1
    min_len = min(min_len, right - left + 1)  # 此时窗口已经太小了！
```

---

## 11.4 最长无重复字符子串（综合版）

```python
def length_of_longest_substring(s):
    left = 0
    char_set = set()
    max_len = 0

    for right in range(len(s)):
        # 扩张
        while s[right] in char_set:   # 违规：有重复
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])

        max_len = max(max_len, right - left + 1)

    return max_len
```

**另一种写法**（先加入后检查）：

```python
def length_of_longest_substring_v2(s):
    left = 0
    count = {}
    max_len = 0

    for right in range(len(s)):
        count[s[right]] = count.get(s[right], 0) + 1  # 加入

        while count[s[right]] > 1:   # 出现重复
            count[s[left]] -= 1
            if count[s[left]] == 0:
                del count[s[left]]
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len
```

---

## 11.5 至多 k 个不同字符的最长子串

```python
def length_of_longest_substring_k_distinct(s, k):
    left = 0
    count = {}
    max_len = 0

    for right in range(len(s)):
        count[s[right]] = count.get(s[right], 0) + 1

        while len(count) > k:   # 超过 k 种字符，收缩
            count[s[left]] -= 1
            if count[s[left]] == 0:
                del count[s[left]]
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len

# 测试
print(length_of_longest_substring_k_distinct("eceba", 2))   # 3（"ece"）
print(length_of_longest_substring_k_distinct("aa", 1))       # 2
```

---

## 11.6 乘积小于 k 的子数组数目

```python
def num_subarray_product_less_than_k(nums, k):
    if k <= 1:
        return 0

    left = 0
    product = 1
    count = 0

    for right in range(len(nums)):
        product *= nums[right]

        while product >= k:
            product //= nums[left]
            left += 1

        # 以 right 结尾且满足条件的子数组数 = right - left + 1
        # （即 [left..right], [left+1..right], ..., [right..right]）
        count += right - left + 1

    return count

# 测试
print(num_subarray_product_less_than_k([10, 5, 2, 6], 100))  # 8
# [10], [5], [2], [6], [10,5], [5,2], [2,6], [5,2,6]
```

**计数技巧**：当窗口 `[left, right]` 合法时，以 `right` 结尾的所有合法子数组数 = `right - left + 1`。这是因为 `[left..right], [left+1..right], ..., [right..right]` 都满足条件（乘积单调递减）。

---

## 11.7 包含负数的情况：前缀和 + 单调队列

当数组包含负数时，滑动窗口的单调性被破坏，不能直接使用。

**例**：和至少为 k 的最短子数组（含负数）

```python
from collections import deque

def shortest_subarray_sum_ge_k(nums, k):
    """含负数的最短子数组，使用前缀和 + 单调队列"""
    n = len(nums)
    # 构建前缀和
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]

    # prefix[j] - prefix[i] >= k 即 prefix[j] >= prefix[i] + k
    # 找最小的 j - i，使得 prefix[j] - prefix[i] >= k
    dq = deque()  # 单调递增队列，存前缀和的下标
    min_len = float('inf')

    for j in range(n + 1):
        # 弹出队头：若满足条件，更新答案并缩短
        while dq and prefix[j] - prefix[dq[0]] >= k:
            min_len = min(min_len, j - dq.popleft())

        # 维护单调递增：弹出比当前大的队尾（它们不可能是更优的左端点）
        while dq and prefix[dq[-1]] >= prefix[j]:
            dq.pop()

        dq.append(j)

    return min_len if min_len != float('inf') else -1

# 测试
print(shortest_subarray_sum_ge_k([2, -1, 2], 3))  # 3
print(shortest_subarray_sum_ge_k([1, 2], 4))       # -1
```

这道题是滑动窗口与单调队列结合的经典例子，将在第15章深入讲解。

---

## 11.8 变长窗口的收缩条件选择

```
问题：找满足 f(window) 成立的子数组
            ↓
f(window) 随窗口扩大单调变化吗？
    ↓ 是                    ↓ 否
滑动窗口可用            需要其他方法（前缀和、DP）
    ↓
找最长还是最短？
  ↓ 最长                ↓ 最短
收缩条件：              收缩条件：
f 为假时收缩            f 为真时收缩
收缩后更新答案          收缩前更新答案
```

---

## 11.9 本章小结

变长滑动窗口的两个关键：

| 类型 | while 条件 | 答案更新时机 |
|------|------------|--------------|
| 最长子数组 | `while 违规` | while 循环**后** |
| 最短子数组 | `while 满足条件` | while 循环**内，收缩前** |

前提：窗口的评估函数对于窗口扩大具有单调性（正整数数组的和、集合大小、乘积等）。

**下一章：字符串滑动窗口——字符频率映射与覆盖模型**

---

## LeetCode 推荐题目

- [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/) ⭐⭐
- [209. 长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/) ⭐⭐
- [713. 乘积小于 K 的子数组](https://leetcode.cn/problems/subarray-product-less-than-k/) ⭐⭐
- [340. 至多包含 K 个不同字符的最长子串](https://leetcode.cn/problems/longest-substring-with-at-most-k-distinct-characters/) ⭐⭐
- [862. 和至少为 K 的最短子数组](https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/) ⭐⭐⭐
