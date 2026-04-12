# 第8章：最长递增子序列（LIS）

## 8.1 问题定义

**最长递增子序列**（Longest Increasing Subsequence，LIS）：

```
给定整数数组 nums，找出其中最长严格递增子序列的长度。

nums = [10, 9, 2, 5, 3, 7, 101, 18]
LIS = [2, 3, 7, 101] 或 [2, 5, 7, 101]，长度为 4
```

---

## 8.2 O(n²) DP 解法

**状态**：`dp[i]` = 以 `nums[i]` 结尾的最长递增子序列长度

**转移**：枚举 i 之前所有 j，如果 `nums[j] < nums[i]`，则 `dp[i]` 可以由 `dp[j] + 1` 转移

```python
def length_of_lis(nums):
    n = len(nums)
    dp = [1] * n  # 每个元素本身至少是长度1的LIS
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

print(length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]))  # 4
print(length_of_lis([0, 1, 0, 3, 2, 3]))            # 4
```

**时间复杂度**：O(n²)  
对 n ≤ 5000 足够，但 n ≤ 10⁵ 时需要优化。

---

## 8.3 O(n log n) 耐心排序（Patience Sorting）

### 核心思想

维护一个"牌堆"数组 `tails`，其中 `tails[k]` = 所有长度为 k+1 的递增子序列中，**末尾元素最小的那个**。

**为什么要维护末尾最小值？**
> 末尾越小，后续越容易接上更多元素，LIS 越可能更长。这是一种贪心思想。

### 算法过程

```
nums = [10, 9, 2, 5, 3, 7, 101, 18]

tails = []

处理 10: tails = [10]
处理  9: 9 < 10，替换 tails[0]: tails = [9]
处理  2: 2 < 9，替换 tails[0]: tails = [2]
处理  5: 5 > 2，追加: tails = [2, 5]
处理  3: 3 < 5，替换 tails[1]: tails = [2, 3]
处理  7: 7 > 3，追加: tails = [2, 3, 7]
处理101: 101 > 7，追加: tails = [2, 3, 7, 101]
处理 18: 18 < 101，替换 tails[3]: tails = [2, 3, 7, 18]

最终 len(tails) = 4 → LIS 长度为 4
```

**关键性质**：`tails` 数组始终是严格递增的，因此可以用**二分查找**定位插入位置。

```python
import bisect

def length_of_lis_nlogn(nums):
    tails = []  # tails[i] = 长度为 i+1 的 IS 末尾最小值
    
    for num in nums:
        # 找到第一个 >= num 的位置（严格递增用 bisect_left）
        pos = bisect.bisect_left(tails, num)
        
        if pos == len(tails):
            tails.append(num)   # num 比所有末尾都大，延伸最长序列
        else:
            tails[pos] = num    # 用 num 替换，维护末尾最小值
    
    return len(tails)

print(length_of_lis_nlogn([10, 9, 2, 5, 3, 7, 101, 18]))  # 4
```

**时间复杂度**：O(n log n)  
**空间复杂度**：O(n)

### ⚠️ 重要说明

`tails` 数组**不是**一个合法的递增子序列，它只是用来辅助计算长度的！

```python
# nums = [10, 9, 2, 5, 3, 7, 101, 18]
# 最终 tails = [2, 3, 7, 18]
# 但 [2, 3, 7, 18] 不一定是原数组的子序列（18 在 101 后面）
# 真正的 LIS 是 [2, 5, 7, 101] 或 [2, 3, 7, 101]
```

如果需要还原实际的LIS，需要额外记录每个元素的"前驱"。

---

## 8.4 还原实际的 LIS

```python
def lis_with_trace(nums):
    n = len(nums)
    dp = [1] * n
    parent = [-1] * n  # parent[i] = LIS中i的前驱索引
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    
    # 找到LIS的最后一个元素
    max_len = max(dp)
    last = dp.index(max_len)
    
    # 回溯构造LIS
    lis = []
    cur = last
    while cur != -1:
        lis.append(nums[cur])
        cur = parent[cur]
    
    return max_len, list(reversed(lis))

length, seq = lis_with_trace([10, 9, 2, 5, 3, 7, 101, 18])
print(f"LIS长度: {length}, LIS: {seq}")  # 4, [2, 5, 7, 101]
```

---

## 8.5 变形题一览

### 变形1：非严格递增（允许相等）

只需将 `bisect_left` 改为 `bisect_right`：

```python
def length_of_lnis(nums):  # Longest Non-decreasing Subsequence
    tails = []
    for num in nums:
        pos = bisect.bisect_right(tails, num)  # 允许相等
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)
```

### 变形2：最长递减子序列

反转数组，转化为LIS：

```python
def length_of_lds(nums):
    return length_of_lis_nlogn([-x for x in nums])
```

### 变形3：最长摆动子序列（LeetCode 376）

已在第4章介绍。

### 变形4：俄罗斯套娃信封（LeetCode 354）

```
每个信封 [w, h]，当且仅当 w 和 h 都严格更大时才能套入。
求最多能套几个信封？
```

**思路**：二维LIS
- 先按宽度升序排序
- 宽度相同时，按高度**降序**（防止同宽度信封相互套）
- 对高度数组求LIS

```python
def max_envelopes(envelopes):
    # 宽度升序，宽度相同时高度降序
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    heights = [e[1] for e in envelopes]
    return length_of_lis_nlogn(heights)

print(max_envelopes([[5,4],[6,4],[6,7],[2,3]]))  # 3 ([2,3],[5,4],[6,7])
```

**为什么宽度相同时高度要降序？**

```
例：宽度都是6：[(6,4), (6,7)]
- 若升序：heights = [4, 7]，LIS = 2（但两个都是宽6，不能套！）
- 若降序：heights = [7, 4]，LIS = 1（正确，宽度相同的只取1个）
```

---

## 8.6 二维 LIS：最长链（LeetCode 646）

```
数对 [a, b]，当且仅当 b_prev < a_next 时才能链接。
求最长链的长度。
```

```python
def find_longest_chain(pairs):
    pairs.sort(key=lambda x: x[1])  # 按右端点排序（贪心）
    
    tails = []
    for a, b in pairs:
        # 找第一个右端点 >= a 的位置
        pos = bisect.bisect_left(tails, a)
        if pos == len(tails):
            tails.append(b)
        else:
            tails[pos] = b
    
    return len(tails)
```

---

## 8.7 本章小结

| 方法 | 时间复杂度 | 适用场景 |
|------|-----------|---------|
| O(n²) DP | O(n²) | n ≤ 5000，需要还原序列 |
| 耐心排序 | O(n log n) | n ≤ 10⁵，只需长度 |

**核心思想**：
- O(n²)：`dp[i]` = 以 `nums[i]` 结尾的最长递增子序列
- O(n log n)：贪心维护末尾最小值 + 二分查找

**下一章：编辑距离——字符串DP的皇冠**

---

## LeetCode 推荐题目

- [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/) ⭐⭐
- [354. 俄罗斯套娃信封问题](https://leetcode.cn/problems/russian-doll-envelopes/) ⭐⭐⭐
- [646. 最长数对链](https://leetcode.cn/problems/maximum-length-of-pair-chain/) ⭐⭐
- [673. 最长递增子序列的个数](https://leetcode.cn/problems/number-of-longest-increasing-subsequence/) ⭐⭐⭐
