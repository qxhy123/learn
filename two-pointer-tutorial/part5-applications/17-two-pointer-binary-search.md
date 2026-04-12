# 第17章：双指针与二分搜索的结合

## 17.1 两种技术的互补性

双指针和二分搜索都是利用有序性的技术，但切入角度不同：

| 技术 | 思路 | 时间 | 适用场景 |
|------|------|------|----------|
| 双指针 | 两端或同向逼近，线性扫描 | O(n) | 配对、子数组、连续区间 |
| 二分搜索 | 折半查找，对数缩小 | O(log n) 单次 | 有序序列中查找单个目标 |
| 双指针 + 二分 | 外层线性枚举，内层二分 | O(n log n) | 枚举一端，二分确定另一端 |

**组合的价值**：当对撞指针无法直接使用（单调性不够强），但可以固定一个变量后对另一个用二分时，两者结合。

---

## 17.2 经典组合：找两数之和（无序数组）

```python
def two_sum_sorted_binary(nums, target):
    """有序数组：枚举左端，二分查找右端"""
    nums_sorted = sorted(nums)

    for i in range(len(nums_sorted)):
        complement = target - nums_sorted[i]

        # 在 i+1..n-1 范围内二分查找 complement
        left, right = i + 1, len(nums_sorted) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums_sorted[mid] == complement:
                return [nums_sorted[i], complement]
            elif nums_sorted[mid] < complement:
                left = mid + 1
            else:
                right = mid - 1

    return []

# 注：有序数组时，对撞指针 O(n) 更优，此处只演示组合思路
# 实际用途：当需要统计对数时，二分可以批量计数
def count_pairs_with_sum_le_target(nums, target):
    """统计有序数组中 nums[i]+nums[j]<=target 的对数"""
    import bisect
    nums.sort()
    count = 0
    for i in range(len(nums)):
        # 找最大的 j 使得 nums[i] + nums[j] <= target
        upper = bisect.bisect_right(nums, target - nums[i]) - 1
        if upper > i:
            count += upper - i
    return count
```

---

## 17.3 矩阵中的双指针：搜索二维有序矩阵

**问题（LeetCode 240）**：在行列均升序的矩阵中搜索目标值。

```python
def search_matrix(matrix, target):
    """
    从右上角出发的对撞指针：
      - 当前值 > target：左移（减小值）
      - 当前值 < target：下移（增大值）
      - 当前值 == target：找到
    
    为什么从右上角？
    右上角是行的最大值，也是列的最小值。
    这使得每次比较都能排除一整行或一整列。
    """
    if not matrix or not matrix[0]:
        return False

    row, col = 0, len(matrix[0]) - 1

    while row < len(matrix) and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1   # 排除当前列（所有行的这一列都太大）
        else:
            row += 1   # 排除当前行（当前行的所有列都太小）

    return False

# 测试
matrix = [
    [1,   4,  7, 11, 15],
    [2,   5,  8, 12, 19],
    [3,   6,  9, 16, 22],
    [10, 13, 14, 17, 24],
    [18, 21, 23, 26, 30]
]
print(search_matrix(matrix, 5))   # True
print(search_matrix(matrix, 20))  # False
```

**复杂度**：O(m + n)，每次移动排除一行或一列，最多 m + n 次。

---

## 17.4 双指针+二分：统计小于乘积的对数

**问题**：有序数组中，统计 `nums[i] * nums[j] < k` 的对数（i < j）。

```python
import bisect

def count_pairs_product_lt_k(nums, k):
    """枚举左端，二分确定右端范围"""
    count = 0
    n = len(nums)

    for i in range(n - 1):
        if nums[i] <= 0:
            continue
        # nums[i] * nums[j] < k → nums[j] < k / nums[i]
        limit = k / nums[i]
        # 在 i+1..n-1 中找最大的 j 使 nums[j] < limit
        j = bisect.bisect_left(nums, limit, i + 1) - 1
        if j > i:
            count += j - i

    return count
```

---

## 17.5 旋转数组中的搜索（双指针思维 + 二分）

**问题（LeetCode 33）**：在旋转有序数组中搜索目标值。

```python
def search_rotated(nums, target):
    """
    二分搜索：每次判断哪一半是有序的
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        # 左半有序
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # 右半有序
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1

# 测试
print(search_rotated([4,5,6,7,0,1,2], 0))   # 4
print(search_rotated([4,5,6,7,0,1,2], 3))   # -1
print(search_rotated([1], 0))               # -1
```

---

## 17.6 二分答案 + 双指针验证

**问题（LeetCode 1631）**：矩阵中从左上到右下的路径，最小化路径上的最大差值。

```python
def minimum_effort_path(heights):
    """二分答案：对 effort 二分，验证该 effort 下是否存在路径"""
    import collections

    rows, cols = len(heights), len(heights[0])

    def can_reach(max_effort):
        """BFS/DFS 验证：在最大代价 max_effort 下能否到达终点"""
        visited = [[False] * cols for _ in range(rows)]
        queue = collections.deque([(0, 0)])
        visited[0][0] = True

        while queue:
            r, c = queue.popleft()
            if r == rows - 1 and c == cols - 1:
                return True
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and not visited[nr][nc]
                        and abs(heights[nr][nc] - heights[r][c]) <= max_effort):
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        return False

    # 二分答案
    left, right = 0, max(max(row) for row in heights)
    while left < right:
        mid = (left + right) // 2
        if can_reach(mid):
            right = mid
        else:
            left = mid + 1

    return left

# 测试
print(minimum_effort_path([[1,2,2],[3,8,2],[5,3,5]]))  # 2
print(minimum_effort_path([[1,2,3],[3,8,4],[5,3,5]]))  # 1
```

**二分答案的通用框架**：

```python
# 当 check(x) 具有单调性（满足的 x 连续）时，用二分找边界
left, right = min_possible_answer, max_possible_answer
while left < right:
    mid = (left + right) // 2
    if check(mid):
        right = mid      # mid 满足，答案可能更小
    else:
        left = mid + 1   # mid 不满足，答案在右侧
return left
```

---

## 17.7 找两个有序数组的中位数

```python
def find_median_sorted_arrays(nums1, nums2):
    """
    二分分割：确保左半部分恰好包含总数的一半
    O(log(min(m, n)))
    """
    # 确保 nums1 是较短的数组
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    half = (m + n + 1) // 2

    left, right = 0, m
    while left <= right:
        i = (left + right) // 2  # nums1 的分割位置
        j = half - i             # nums2 的分割位置

        nums1_left_max = nums1[i-1] if i > 0 else float('-inf')
        nums1_right_min = nums1[i] if i < m else float('inf')
        nums2_left_max = nums2[j-1] if j > 0 else float('-inf')
        nums2_right_min = nums2[j] if j < n else float('inf')

        if nums1_left_max <= nums2_right_min and nums2_left_max <= nums1_right_min:
            # 找到正确分割
            left_max = max(nums1_left_max, nums2_left_max)
            if (m + n) % 2 == 1:
                return float(left_max)
            right_min = min(nums1_right_min, nums2_right_min)
            return (left_max + right_min) / 2
        elif nums1_left_max > nums2_right_min:
            right = i - 1
        else:
            left = i + 1

    return 0.0

# 测试
print(find_median_sorted_arrays([1,3], [2]))       # 2.0
print(find_median_sorted_arrays([1,2], [3,4]))     # 2.5
```

---

## 17.8 本章小结

双指针与二分的结合模式：

| 模式 | 外层 | 内层 | 总复杂度 |
|------|------|------|----------|
| 枚举一端 + 二分另一端 | O(n) 枚举 | O(log n) 二分 | O(n log n) |
| 矩阵对撞指针 | O(m+n) | — | O(m+n) |
| 二分答案 + 验证 | O(log V) | O(n) 验证 | O(n log V) |

**下一章：矩阵双指针——二维空间的双指针思维**

---

## LeetCode 推荐题目

- [240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/) ⭐⭐
- [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/) ⭐⭐
- [1631. 最小体力消耗路径](https://leetcode.cn/problems/path-with-minimum-effort/) ⭐⭐⭐
- [4. 寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/) ⭐⭐⭐⭐
- [378. 有序矩阵中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-sorted-matrix/) ⭐⭐⭐
