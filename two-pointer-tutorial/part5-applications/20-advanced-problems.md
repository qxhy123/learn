# 第20章：竞赛级综合题与思维拓展

## 20.1 本章目标

本章通过精选的高难度题目，展示双指针与其他算法深度融合的思维：
- 双指针 + 贪心
- 双指针 + 动态规划
- 双指针 + 前缀和
- 双指针 + 单调栈

每道题都是多种思维的交汇点，是竞赛和高级面试的核心考点。

---

## 20.2 双指针 + 贪心：跳跃游戏系列

**LeetCode 45**：给定数组 `nums`，每个元素代表从该位置能跳的最大步数，求到达最后一个下标的最少跳跃次数。

```python
def jump(nums):
    """
    贪心：每次在当前能到达的范围内，选跳跃后覆盖最远的位置
    双指针维护：[cur_start, cur_end] 是当前跳跃能覆盖的范围
    """
    n = len(nums)
    if n == 1:
        return 0

    jumps = 0
    cur_end = 0     # 当前跳跃能到达的最右位置
    farthest = 0    # 从 [0, cur_end] 中任意一步能到的最右位置

    for i in range(n - 1):
        farthest = max(farthest, i + nums[i])

        if i == cur_end:   # 到达当前跳跃的边界，必须再跳一次
            jumps += 1
            cur_end = farthest

            if cur_end >= n - 1:
                break

    return jumps

# 测试
print(jump([2,3,1,1,4]))  # 2（0→1→4）
print(jump([2,3,0,1,4]))  # 2
print(jump([1,2,3]))       # 2
```

**双指针视角**：`i` 是遍历指针，`cur_end` 是"当前轮次"的右边界，`farthest` 是"下一轮次"的右边界。当 `i` 追上 `cur_end` 时，触发跳跃，边界更新。

---

## 20.3 双指针 + 前缀和：子数组和问题

**LeetCode 560**：子数组和等于 k 的子数组数目（含负数）。

```python
def subarray_sum(nums, k):
    """
    前缀和 + 哈希表：
    sum[i..j] = prefix[j+1] - prefix[i] = k
    → prefix[i] = prefix[j+1] - k
    
    遍历时，对每个 prefix[j+1]，查表找有多少个 prefix[i] 满足条件
    """
    count = 0
    prefix_sum = 0
    seen = {0: 1}  # prefix_sum -> 出现次数

    for num in nums:
        prefix_sum += num
        count += seen.get(prefix_sum - k, 0)
        seen[prefix_sum] = seen.get(prefix_sum, 0) + 1

    return count

# 测试
print(subarray_sum([1,1,1], 2))     # 2
print(subarray_sum([1,2,3], 3))     # 2
print(subarray_sum([-1,-1,1], 0))   # 1
```

**进阶**：含负数时滑动窗口失效，前缀和+哈希表是标准解法。

---

## 20.4 双指针 + DP：最长湍流子数组

**LeetCode 978**：最长的"湍流"子数组，即相邻元素大小交替变化。

```python
def max_turbulence_size(arr):
    """
    滑动窗口 + DP思想：
    维护以 right 结尾的最长湍流子数组的长度
    """
    n = len(arr)
    if n < 2:
        return n

    left = 0
    result = 1

    for right in range(1, n):
        if right == 1:
            if arr[right] != arr[right-1]:
                result = max(result, 2)
            continue

        prev_cmp = (arr[right-1] > arr[right-2]) - (arr[right-1] < arr[right-2])
        curr_cmp = (arr[right] > arr[right-1]) - (arr[right] < arr[right-1])

        if curr_cmp == 0:
            left = right   # 相等，窗口重置
        elif prev_cmp * curr_cmp >= 0:  # 同号（连续递增或连续递减）
            left = right - 1  # 窗口收缩到最后两个元素

        result = max(result, right - left + 1)

    return result

# 测试
print(max_turbulence_size([9,4,2,10,7,8,8,1,9]))  # 5 ([4,2,10,7,8])
print(max_turbulence_size([4,8,12,16]))            # 2
print(max_turbulence_size([100]))                  # 1
```

---

## 20.5 双指针 + 单调栈：最大宽度坡

**LeetCode 962**：给定数组 `nums`，找最大的 `j - i` 使得 `i < j` 且 `nums[i] <= nums[j]`。

```python
def max_width_ramp(nums):
    """
    两步双指针：
    1. 从左构建单调递减栈（候选左端点）
    2. 从右向左扫描，贪心匹配
    """
    n = len(nums)
    stack = []  # 单调递减栈，存下标

    # 步骤1：构建左端点候选（单调递减，后面的元素不可能成为更优左端点）
    for i in range(n):
        if not stack or nums[i] < nums[stack[-1]]:
            stack.append(i)

    # 步骤2：从右向左扫描，匹配最远的左端点
    result = 0
    j = n - 1
    while j >= 0 and stack:
        while stack and nums[stack[-1]] <= nums[j]:
            result = max(result, j - stack[-1])
            stack.pop()
        j -= 1

    return result

# 测试
print(max_width_ramp([6,0,8,2,1,5]))  # 4 (0→4, nums[1]=0 <= nums[5]=5)
print(max_width_ramp([9,8,1,0,1,9,4,0,4,1]))  # 7
```

---

## 20.6 困难题：最多 K 次替换后最长子数组

**LeetCode 2401**：给定数组，可以将最多 k 个元素替换为任意值，找使得整个子数组和可被 p 整除的最长子数组。

```python
def longest_subarray_k_replacements(nums, k):
    """
    变长滑动窗口 + 贪心：
    窗口内可以替换最多 k 个元素，使得整个子数组中所有相邻元素的差的绝对值都 <= 限制
    
    本题以"元素为1和0，最多k个0替换"为例
    """
    left = 0
    max_ones = 0  # 窗口内1的最大数量（只增不减的技巧）
    result = 0
    count = [0, 0]  # count[0]和count[1]分别记录0和1的数量

    for right in range(len(nums)):
        count[nums[right]] += 1
        max_ones = max(max_ones, count[1])

        # 窗口大小 - 最多1的数量 = 需要替换的0的数量
        if right - left + 1 - max_ones > k:
            count[nums[left]] -= 1
            left += 1

        result = max(result, right - left + 1)

    return result
```

---

## 20.7 竞赛题：统计完全子数组

**LeetCode 2799**：子数组中不同元素数等于整个数组中不同元素数的子数组数目。

```python
def count_complete_subarrays(nums):
    """
    双指针：找所有"完整"子数组（包含所有不同元素）
    
    对于每个右端点 right，找最小的左端点 left 使得窗口包含所有元素
    对于固定 right，[0..left] 都是合法的左端点，贡献 left+1 个子数组
    """
    from collections import Counter

    total_distinct = len(set(nums))
    count = Counter()
    distinct = 0
    left = 0
    result = 0

    for right in range(len(nums)):
        if count[nums[right]] == 0:
            distinct += 1
        count[nums[right]] += 1

        # 尽量收缩左端
        while distinct == total_distinct:
            result += len(nums) - right  # [left..right], [left..right+1], ..., [left..n-1] 都合法
            count[nums[left]] -= 1
            if count[nums[left]] == 0:
                distinct -= 1
            left += 1

    return result

# 测试
print(count_complete_subarrays([1,3,1,2,2]))  # 4
print(count_complete_subarrays([5,5,5,5]))    # 10
```

**批量计数技巧**：当窗口 `[left, right]` 满足条件时，以 `left` 为左端点、任何 `>= right` 的右端点都满足条件，贡献 `n - right` 个子数组。

---

## 20.8 双指针的极限：O(n) 中位数维护

用两个堆维护动态中位数（大根堆存左半，小根堆存右半）：

```python
import heapq

class MedianFinder:
    """数据流中的中位数（LeetCode 295）"""

    def __init__(self):
        self.small = []   # 大根堆（存负数模拟）：左半部分
        self.large = []   # 小根堆：右半部分

    def add_num(self, num):
        # 加入小堆（先放大根堆，弹出最大值给小根堆）
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))

        # 保持 small 的大小 >= large 的大小
        if len(self.small) < len(self.large):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def find_median(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2

# 测试
mf = MedianFinder()
for num in [1, 2, 3, 4, 5]:
    mf.add_num(num)
    print(mf.find_median())  # 1, 1.5, 2, 2.5, 3
```

---

## 20.9 双指针知识体系全览

```
双指针
├── 对撞指针（Opposite Direction）
│   ├── 两数之和（有序数组）
│   ├── 三数/N数之和
│   ├── 盛最多水
│   ├── 接雨水（单调性分析）
│   └── 矩阵搜索（行列均有序）
│
├── 同向双指针（Same Direction）
│   ├── 写指针模式（移除、去重）
│   ├── 三路分区（荷兰国旗）
│   └── 滑动窗口
│       ├── 定长窗口（差量更新）
│       ├── 变长窗口（最长/最短）
│       └── 字符串窗口（need/have）
│
├── 快慢指针（Fast & Slow）
│   ├── Floyd 判圈
│   ├── 链表中点
│   └── 倒数第 K 个
│
└── 多指针
    ├── 三指针（low/mid/high）
    ├── 四指针（螺旋遍历）
    └── K指针（区间交集等）

与其他算法结合
├── 双指针 + 二分（搜索空间压缩）
├── 双指针 + 贪心（跳跃游戏）
├── 双指针 + DP（湍流子数组）
├── 双指针 + 前缀和（子数组和）
└── 双指针 + 单调栈（最大宽度坡）
```

---

## 20.10 解题策略总结

遇到需要处理**连续子数组/子串**的问题时，按以下顺序思考：

```
1. 是否需要"所有"组合？
   是 → 暴力枚举 O(n²)，看是否能优化
   否 → 寻找单调性

2. 数组是否有序（或可以排序）？
   有序 → 考虑对撞指针
   无序但可排序 → 先排序再双指针
   无序不可排序 → 哈希表 / 前缀和

3. 连续子数组问题，窗口评估函数是否单调？
   单调 → 滑动窗口
   非单调（含负数等）→ 前缀和 + 哈希表

4. 链表问题，是否需要同时知道多个位置？
   是 → 快慢指针 / 多指针

5. 需要组合多种技术？
   → 双指针 + 二分 / 贪心 / DP / 单调栈
```

---

## 20.11 本教程小结

本教程覆盖了双指针从入门到专家级的完整知识体系：

| 部分 | 核心内容 |
|------|----------|
| 第一部分：基础 | 三种形态、单调性前提、O(n) 本质 |
| 第二部分：经典 | N数之和、原地操作、回文、有序数组 |
| 第三部分：滑动窗口 | 框架、定长/变长、字符串窗口 |
| 第四部分：高阶 | 链表高级、接雨水、区间、多分区 |
| 第五部分：综合 | 二分结合、矩阵、字符串、竞赛题 |

**双指针不是一种算法，而是一种思维方式**：利用单调性，让多个指针协同工作，将搜索空间从 O(n²) 压缩到 O(n)。

---

## LeetCode 推荐题目（竞赛级）

- [45. 跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/) ⭐⭐⭐
- [962. 最大宽度坡](https://leetcode.cn/problems/maximum-width-ramp/) ⭐⭐⭐
- [295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/) ⭐⭐⭐
- [560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/) ⭐⭐
- [978. 最长湍流子数组](https://leetcode.cn/problems/longest-turbulent-subarray/) ⭐⭐
- [2799. 统计完全子数组的数目](https://leetcode.cn/problems/count-complete-subarrays-in-an-array/) ⭐⭐⭐
