# 第16章：多指针与分区问题

## 16.1 从双指针到多指针

双指针处理"两类"问题——满足条件/不满足条件。当需要将数组分成**三类或更多类**时，就需要多指针协同工作。

**经典案例**：荷兰国旗问题——将只含 0、1、2 的数组排序，要求**一趟**扫描完成。

---

## 16.2 荷兰国旗问题（三路分区）

```python
def sort_colors(nums):
    """
    三个指针：
      low: 下一个 0 的写入位置（[0, low) 全是 0）
      mid: 当前扫描位置（[low, mid) 全是 1）
      high: 下一个 2 的写入位置（(high, n-1] 全是 2）
    
    不变量：
      nums[0..low-1] = 0
      nums[low..mid-1] = 1
      nums[mid..high] = 待处理
      nums[high+1..n-1] = 2
    """
    low = mid = 0
    high = len(nums) - 1

    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1       # low 左侧都是 0，被换过来的一定是 1，可以直接推进
        elif nums[mid] == 1:
            mid += 1       # 1 在正确位置，直接推进
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1      # 注意：mid 不前进！被换来的数还未检查

# 测试
nums = [2, 0, 2, 1, 1, 0]
sort_colors(nums)
print(nums)  # [0, 0, 1, 1, 2, 2]

nums = [2, 0, 1]
sort_colors(nums)
print(nums)  # [0, 1, 2]
```

**为什么 `mid` 遇到 2 不前进？**

当 `nums[mid] == 2` 时，将其与 `nums[high]` 交换，`nums[high]` 换到了 `mid` 位置。这个新来的元素还没有被检查，可能是 0、1 或 2，所以 `mid` 必须停在原位重新判断。

而当 `nums[mid] == 0` 时，与 `nums[low]` 交换，换来的一定是 1（因为 `[low, mid)` 全是 1），所以 `mid` 可以安全前进。

---

## 16.3 三路分区的不变量追踪

以 `[2,0,1,0,2,1]` 为例追踪：

```
初始：low=0, mid=0, high=5
  [2, 0, 1, 0, 2, 1]
   ^              ^
  low,mid        high

mid=2: 与high交换，high--
  [1, 0, 1, 0, 2, 2]  → high=4
   ^
  low,mid

mid=1: 前进，mid++
  [1, 0, 1, 0, 2, 2]  → mid=1

mid=0: 与low交换，low++, mid++
  [0, 1, 1, 0, 2, 2]  → low=1, mid=2

mid=1: 前进，mid++
  [0, 1, 1, 0, 2, 2]  → mid=3

mid=0: 与low交换，low++, mid++
  [0, 0, 1, 1, 2, 2]  → low=2, mid=4

mid(4) > high(4)，结束
结果：[0, 0, 1, 1, 2, 2] ✓
```

---

## 16.4 推广：K 路分区

将荷兰国旗的三路推广到 K 路：

```python
def k_way_partition(nums, k):
    """
    将 nums 中的 0,1,...,k-1 排序
    用计数排序思想，适合值域小的情况
    """
    count = [0] * k
    for x in nums:
        count[x] += 1

    idx = 0
    for val in range(k):
        for _ in range(count[val]):
            nums[idx] = val
            idx += 1

    return nums
```

当 k=3 时，荷兰国旗算法更优（单次扫描，原地），但对于 k>3，计数排序更简洁。

---

## 16.5 按奇偶分区（稳定版）

```python
def partition_array_by_parity(nums):
    """不稳定版（对撞指针，O(n) 时间）"""
    left, right = 0, len(nums) - 1
    while left < right:
        while left < right and nums[left] % 2 == 0:
            left += 1
        while left < right and nums[right] % 2 == 1:
            right -= 1
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1
    return nums

def partition_array_by_parity_stable(nums):
    """稳定版（双数组合并，O(n) 时间，O(n) 空间）"""
    evens = [x for x in nums if x % 2 == 0]
    odds = [x for x in nums if x % 2 == 1]
    nums[:] = evens + odds
    return nums
```

---

## 16.6 按颜色/类别分区：颜色排序II

**问题**：给定颜色数组，将相同颜色分组，保持颜色的相对出现顺序。

```python
def relative_sort_array(arr1, arr2):
    """arr2 定义顺序，arr1 中不在 arr2 里的元素升序放最后"""
    order = {v: i for i, v in enumerate(arr2)}
    return sorted(arr1, key=lambda x: (order.get(x, len(arr2)), x))

# 测试
print(relative_sort_array([2,3,1,3,2,4,6,7,9,2,19], [2,1,4,3,9,6]))
# [2, 2, 2, 1, 4, 3, 3, 9, 6, 7, 19]
```

---

## 16.7 多指针的经典：归并 K 个有序数组

```python
import heapq

def merge_k_sorted_arrays(arrays):
    """用最小堆维护 K 个数组的当前最小元素"""
    result = []
    # (值, 数组下标, 元素下标)
    heap = [(arrays[i][0], i, 0)
            for i in range(len(arrays)) if arrays[i]]
    heapq.heapify(heap)

    while heap:
        val, arr_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, arr_idx, elem_idx + 1))

    return result

# 测试
print(merge_k_sorted_arrays([[1,4,7],[2,5,8],[3,6,9]]))
# [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

---

## 16.8 三指针合并有序数组

**问题（LeetCode 88）**：将两个有序数组合并，结果存入第一个数组（nums1 有足够空间）。

```python
def merge_sorted_arrays(nums1, m, nums2, n):
    """从后往前合并，避免覆盖问题"""
    p1 = m - 1      # nums1 的最后一个有效元素
    p2 = n - 1      # nums2 的最后一个元素
    p = m + n - 1   # 写入位置（从后往前）

    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1

    # nums2 还有剩余（nums1 的剩余已在正确位置）
    nums1[:p2+1] = nums2[:p2+1]

# 测试
nums1 = [1,2,3,0,0,0]
merge_sorted_arrays(nums1, 3, [2,5,6], 3)
print(nums1)  # [1, 2, 2, 3, 5, 6]
```

**从后往前的必要性**：从前往后合并会覆盖 nums1 中还未处理的元素。从后往前时，写入位置总在两个读指针之后，不会覆盖有效数据。

---

## 16.9 多指针的统一思维模型

多指针问题的设计步骤：

```
1. 定义每个指针的语义（维护什么不变量？）
   - low: [0, low) 全是第一类
   - mid: [low, mid) 全是第二类
   - high: (high, n-1] 全是第三类

2. 确定每种情况的操作（移动哪个指针？是否需要交换？）
   - 当前元素属于第一类：移动 low 和 mid
   - 当前元素属于第二类：只移动 mid
   - 当前元素属于第三类：移动 high，mid 不动（换来的需要重新检查）

3. 确定循环终止条件
   - 通常是 mid > high（所有元素已分类）

4. 验证边界情况
   - 全是一类、全是三类、只有两个元素等
```

---

## 16.10 本章小结

多指针分区的核心：

| 问题 | 指针数 | 关键不变量 |
|------|--------|------------|
| 两路分区 | 2（对撞） | 左侧≤pivot，右侧>pivot |
| 三路分区 | 3（low/mid/high） | 三段各自类别纯净 |
| K路分区 | 计数 | 各段计数准确 |
| K有序合并 | K+1（堆辅助） | 堆顶是全局最小 |

**下一章：双指针与二分搜索的结合——跨维度的搜索空间压缩**

---

## LeetCode 推荐题目

- [75. 颜色分类](https://leetcode.cn/problems/sort-colors/) ⭐⭐（荷兰国旗）
- [88. 合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/) ⭐
- [905. 按奇偶排序数组](https://leetcode.cn/problems/sort-array-by-parity/) ⭐
- [922. 按奇偶排序数组 II](https://leetcode.cn/problems/sort-array-by-parity-ii/) ⭐⭐
- [23. 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/) ⭐⭐⭐
- [1356. 根据数字二进制下 1 的数目排序](https://leetcode.cn/problems/sort-integers-by-the-number-of-1-bits/) ⭐
