# 第10章：定长滑动窗口

## 10.1 定长窗口的特点

定长滑动窗口（Fixed-size Sliding Window）：窗口大小 k 固定，每次向右滑动一格。

```
数组：[1, 3, -1, -3, 5, 3, 6, 7]，k=3

窗口位置：
[1, 3, -1]  -3  5  3  6  7   → 和=3
 1 [3, -1, -3]  5  3  6  7   → 和=-1
 1  3 [-1, -3, 5]  3  6  7   → 和=1
...
```

**核心技巧——差量更新**：每次滑动时，不重新计算整个窗口，只处理：
- 加入右端新元素 `nums[right]`
- 移出左端旧元素 `nums[right - k]`

这使得每次滑动操作为 O(1)，总体 O(n)（而暴力重算每个窗口是 O(nk)）。

---

## 10.2 差量更新的模板

```python
def fixed_window_template(nums, k):
    n = len(nums)
    if n < k:
        return []

    # 初始化第一个窗口
    window_val = compute(nums[:k])  # 例如：sum(nums[:k])
    result = [window_val]

    # 滑动窗口
    for right in range(k, n):
        # 差量更新：加入右端，移出左端
        window_val = update(window_val, nums[right], nums[right - k])
        result.append(window_val)

    return result
```

---

## 10.3 定长窗口最大/最小和

```python
def max_sum_subarray_k(nums, k):
    """大小为 k 的子数组中，最大的和"""
    n = len(nums)
    if n < k:
        return 0

    # 第一个窗口
    window_sum = sum(nums[:k])
    max_sum = window_sum

    for right in range(k, n):
        window_sum += nums[right] - nums[right - k]  # 差量更新
        max_sum = max(max_sum, window_sum)

    return max_sum

def max_avg_subarray_k(nums, k):
    """大小为 k 的子数组中，最大平均值"""
    return max_sum_subarray_k(nums, k) / k

# 测试
print(max_sum_subarray_k([1, 4, 2, 10, 23, 3, 1, 0, 20], 4))  # 39
print(max_avg_subarray_k([1, 12, -5, -6, 50, 3], 4))           # 12.75
```

---

## 10.4 定长窗口中的最大值：单调队列

**问题**：滑动窗口最大值——每个大小为 k 的窗口中的最大值。

差量更新只对"和"类型有效。对于"最大值"，移出元素时可能移出了最大值，需要重新找最大值。

**暴力**：每个窗口取最大值，O(nk)。

**最优**：单调队列，O(n)。（详见第15章，这里先用暴力演示框架）

```python
from collections import deque

def max_sliding_window(nums, k):
    """单调递减队列维护窗口最大值"""
    dq = deque()  # 存下标，队头是窗口最大值的下标
    result = []

    for right in range(len(nums)):
        # 移出不在窗口内的队头
        while dq and dq[0] <= right - k:
            dq.popleft()

        # 维护单调递减：移出所有比当前元素小的队尾
        while dq and nums[dq[-1]] < nums[right]:
            dq.pop()

        dq.append(right)

        # 窗口已满，记录最大值
        if right >= k - 1:
            result.append(nums[dq[0]])

    return result

# 测试
print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))
# [3, 3, 5, 5, 6, 7]
```

**单调队列原理**：队列中只保留"有可能成为后续窗口最大值"的元素。当新元素 `x` 进入时，所有比 `x` 小的元素永远不可能成为最大值（因为 `x` 出现得更晚，且更大），可以直接丢弃。

---

## 10.5 定长窗口中的频率统计

**问题**：长度为 k 的子数组中，最多有多少个不同元素？

```python
from collections import Counter

def max_distinct_in_k(nums, k):
    count = Counter(nums[:k])
    max_distinct = len(count)

    for right in range(k, len(nums)):
        # 加入右端
        new_elem = nums[right]
        count[new_elem] += 1

        # 移出左端
        old_elem = nums[right - k]
        count[old_elem] -= 1
        if count[old_elem] == 0:
            del count[old_elem]

        max_distinct = max(max_distinct, len(count))

    return max_distinct
```

**差量更新哈希表**：加入/移出各一次操作，O(1) 摊销。

---

## 10.6 字符串的定长窗口：字母异位词

**问题**：找字符串 `s` 中所有 `p` 的字母异位词的起始下标。

```python
def find_anagrams(s, p):
    """定长滑动窗口 + 字符频率差量"""
    k = len(p)
    if len(s) < k:
        return []

    # 目标频率
    target = Counter(p)
    # 当前窗口频率
    window = Counter(s[:k])

    result = []
    if window == target:
        result.append(0)

    for right in range(k, len(s)):
        # 加入右端
        new_char = s[right]
        window[new_char] += 1

        # 移出左端
        old_char = s[right - k]
        window[old_char] -= 1
        if window[old_char] == 0:
            del window[old_char]

        if window == target:
            result.append(right - k + 1)

    return result

# 测试
print(find_anagrams("cbaebabacd", "abc"))  # [0, 6]
print(find_anagrams("abab", "ab"))         # [0, 1, 2]
```

**优化**：比较两个 Counter 的代价是 O(字母表大小)，可以用"差值计数器"进一步优化到真正 O(1) 的更新检查：

```python
def find_anagrams_opt(s, p):
    k = len(p)
    if len(s) < k:
        return []

    need = Counter(p)
    diff = {}  # diff[c] = window[c] - need[c]
    for c in s[:k]:
        diff[c] = diff.get(c, 0) + 1
    for c in need:
        diff[c] = diff.get(c, 0) - need[c]

    # 不平衡计数：diff中非零值的数量
    imbalance = sum(1 for v in diff.values() if v != 0)
    result = [0] if imbalance == 0 else []

    for right in range(k, len(s)):
        def update(char, delta):
            nonlocal imbalance
            old = diff.get(char, 0)
            new = old + delta
            diff[char] = new
            if old != 0 and new == 0:
                imbalance -= 1
            elif old == 0 and new != 0:
                imbalance += 1

        update(s[right], 1)
        update(s[right - k], -1)

        if imbalance == 0:
            result.append(right - k + 1)

    return result
```

---

## 10.7 定长窗口的边界处理

```python
# 处理 k > n 的情况
if len(nums) < k:
    return 0  # 或 return []，根据题意

# 初始化第一个窗口
window_sum = sum(nums[:k])  # 用切片初始化

# 滑动起点：right 从 k 开始（第 k+1 个元素是第二个窗口的右端）
for right in range(k, n):
    window_sum += nums[right] - nums[right - k]
    # ...

# 结果数组大小：n - k + 1 个窗口
```

---

## 10.8 本章小结

定长滑动窗口的精华：

1. **差量更新**：每次只处理进入和离开的元素，O(1) 更新
2. **窗口大小**：始终保持 `right - left + 1 == k`（或等价地，`left = right - k + 1`）
3. **数据结构**：和用整数、频率用 Counter、最值用单调队列

**下一章：变长滑动窗口——最短/最长子数组的精确控制**

---

## LeetCode 推荐题目

- [643. 子数组最大平均数 I](https://leetcode.cn/problems/maximum-average-subarray-i/) ⭐
- [1343. 大小为 K 且平均值大于等于阈值的子数组数目](https://leetcode.cn/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/) ⭐⭐
- [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/) ⭐⭐⭐
- [438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/) ⭐⭐
- [567. 字符串的排列](https://leetcode.cn/problems/permutation-in-string/) ⭐⭐
