# 第5章：有序数组与两数之和

## 5.1 问题全景

"两数之和"是双指针最经典的入口题，但它有多个版本，考察的是不同的核心能力：

| 版本 | 数据 | 方法 | 返回 |
|------|------|------|------|
| 无序数组 | 任意 | 哈希表 | 下标 |
| 有序数组 | 升序 | 对撞指针 | 下标或值 |
| 有序数组（所有解） | 升序 | 对撞指针+去重 | 所有配对 |
| 差值为 k | 有序 | 同向双指针 | 计数/配对 |

本章深入讨论有序数组场景，并建立一套可复用的"搜索空间裁剪"思维。

---

## 5.2 核心：搜索空间裁剪

有序数组 `nums`，找 `nums[i] + nums[j] == target`（`i < j`）。

将搜索空间画成网格（行=left，列=right）：

```
nums = [1, 2, 3, 4, 6]，target = 6

       j=0  j=1  j=2  j=3  j=4
       [1]  [2]  [3]  [4]  [6]
i=0[1]  ×    3    4    5    7
i=1[2]       ×    5    6    8   ← (1,3)=2+4=6 ✓
i=2[3]            ×    7    9
i=3[4]                 ×   10
i=4[6]                      ×
```

对撞指针从右上角 `(0, 4)` 出发：
- `(0,4)`: 1+6=7 > 6，right-- → `(0,3)`
- `(0,3)`: 1+4=5 < 6，left++ → `(1,3)`
- `(1,3)`: 2+4=6 = 6，找到！

每次移动**裁剪一整行或一整列**，共最多 n+n 步。

---

## 5.3 标准实现

```python
def two_sum_sorted(numbers, target):
    """LeetCode 167：有序数组两数之和（保证唯一解）"""
    left, right = 0, len(numbers) - 1

    while left < right:
        s = numbers[left] + numbers[right]
        if s == target:
            return [left + 1, right + 1]  # 1-indexed
        elif s < target:
            left += 1
        else:
            right -= 1

    return [-1, -1]  # 无解（题目保证有解时不会到这里）


def two_sum_all_pairs(nums, target):
    """找出所有满足条件的配对（无重复）"""
    left, right = 0, len(nums) - 1
    result = []

    while left < right:
        s = nums[left] + nums[right]
        if s == target:
            result.append([nums[left], nums[right]])
            # 去重：跳过相同的 left 和 right
            while left < right and nums[left] == nums[left + 1]:
                left += 1
            while left < right and nums[right] == nums[right - 1]:
                right -= 1
            left += 1
            right -= 1
        elif s < target:
            left += 1
        else:
            right -= 1

    return result

# 测试
print(two_sum_all_pairs([-2, -1, 0, 0, 1, 2, 3], 0))
# [[-2, 2], [-1, 1], [0, 0]]
```

**去重的关键**：找到答案后，先移动到不同值的位置，再同时推进两个指针。去重操作在 while 循环内，不增加整体复杂度。

---

## 5.4 变体一：两数之差为 k

找有序数组中差为 k 的所有配对 `nums[j] - nums[i] == k`（`i < j`）。

```python
def two_sum_difference(nums, k):
    """同向双指针，因为差值问题用对撞指针不单调"""
    i, j = 0, 1
    result = []

    while j < len(nums):
        diff = nums[j] - nums[i]
        if diff == k:
            result.append([nums[i], nums[j]])
            i += 1
            j += 1
        elif diff < k:
            j += 1  # 差值不足，增大右边
        else:
            i += 1  # 差值过大，增大左边（缩小差值）

    return result

# 测试
print(two_sum_difference([1, 2, 3, 4, 5], 2))  # [[1,3],[2,4],[3,5]]
```

**为什么用同向指针而非对撞指针**：对于差值问题，当 `diff > k` 时应该增大 `i`（而不是减小 `j`），两个指针都只向右移动，是同向双指针的场景。

---

## 5.5 变体二：统计满足条件的对数

给定有序数组，统计满足 `nums[i] + nums[j] <= target` 的对数（`i < j`）。

```python
def count_pairs_le_target(nums, target):
    """O(n) 对撞指针计数"""
    left, right = 0, len(nums) - 1
    count = 0

    while left < right:
        if nums[left] + nums[right] <= target:
            # nums[left] 与 right 到 left+1 的所有元素都满足
            # 因为更小的 right 更容易满足
            count += right - left  # 固定 left，right 可以是 left+1 到当前 right
            left += 1
        else:
            right -= 1

    return count

# 测试
print(count_pairs_le_target([1, 1, 2, 3], 4))  # 4
# (1,1)=2≤4, (1,2)=3≤4, (1,3)=4≤4, (1,3)=4≤4
```

**批量计数的思维**：当 `nums[left] + nums[right] <= target` 时，由于数组有序，`nums[left] + nums[j]` 对所有 `j <= right` 也满足条件，一次性加 `right - left` 个。

---

## 5.6 变体三：最接近 target 的两数之和

```python
def two_sum_closest(nums, target):
    """有序数组，找两数之和最接近 target"""
    left, right = 0, len(nums) - 1
    best = float('inf')
    best_pair = []

    while left < right:
        s = nums[left] + nums[right]
        if abs(s - target) < abs(best - target):
            best = s
            best_pair = [nums[left], nums[right]]

        if s < target:
            left += 1
        elif s > target:
            right -= 1
        else:
            return [nums[left], nums[right]]  # 完全等于，直接返回

    return best_pair

# 测试
print(two_sum_closest([1, 2, 3, 4, 6], 5))  # [1, 4] 或 [2, 3]
```

---

## 5.7 完整框架：有序数组双指针决策树

```
有序数组，找两数满足某条件 f(a, b)
          ↓
f(a, b) 具有单调性吗？（a 增大时 f 单调变化）
    ↓ 是                    ↓ 否
对撞/同向双指针           排序后再用双指针
    ↓                       或哈希表
f(a,b) 是和还是差？
  ↓ 和                 ↓ 差
对撞指针           同向双指针
（从两端）         （同向推进）
```

---

## 5.8 边界条件陷阱

```python
# 陷阱1：left < right vs left <= right
# 要找两个不同位置的数，必须是 left < right
while left < right:  # ✓
while left <= right:  # ✗ 可能使用同一个元素两次

# 陷阱2：去重时的越界
while left < right and nums[left] == nums[left + 1]:
    left += 1
# 注意这里没有 +1 后的越界问题，因为 while 条件保证了 left < right

# 陷阱3：先去重再移动，还是先移动再去重？
# 标准做法：找到答案后，先去重，再移动
if s == target:
    result.append(...)
    while left < right and nums[left] == nums[left + 1]: left += 1  # 去重
    while left < right and nums[right] == nums[right - 1]: right -= 1  # 去重
    left += 1   # 移动
    right -= 1  # 移动
```

---

## 5.9 复杂度分析

| 操作 | 时间 | 空间 |
|------|------|------|
| 找唯一解 | O(n) | O(1) |
| 找所有解 | O(n) | O(1)（不计输出） |
| 差值问题 | O(n) | O(1) |
| 统计对数 | O(n) | O(1) |
| 最接近值 | O(n) | O(1) |

如果数组无序，需先排序：O(n log n)。

---

## 5.10 本章小结

两数之和是对撞指针的**标准模板题**：

1. **有序** → 直接对撞指针
2. **无序** → 排序后对撞指针（若空间限制严格），或哈希表
3. **去重** → 找到答案后跳过相同元素再推进
4. **批量计数** → 利用单调性一次性累加多个

**下一章：三数之和——枚举降维，对撞指针的嵌套应用**

---

## LeetCode 推荐题目

- [167. 两数之和 II](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/) ⭐
- [633. 平方数之和](https://leetcode.cn/problems/sum-of-square-numbers/) ⭐⭐
- [1099. 小于 K 的两数之和](https://leetcode.cn/problems/two-sum-less-than-k/) ⭐⭐
- [1679. K 和数对的最大数目](https://leetcode.cn/problems/max-number-of-k-sum-pairs/) ⭐⭐
- [16. 最接近的三数之和](https://leetcode.cn/problems/3sum-closest/) ⭐⭐（预习下章）
