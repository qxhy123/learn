# 第7章：原地修改数组

## 7.1 原地操作的意义

"原地"（in-place）意味着只用 O(1) 额外空间，直接在输入数组上操作。这在以下场景至关重要：

- 内存受限的嵌入式系统
- 大数据流处理（无法复制整个数组）
- 面试中对空间复杂度的要求

核心工具：**写指针模式**（Write Pointer Pattern）——slow 指针标记下一个写入位置，fast 指针扫描所有元素。

---

## 7.2 写指针模式的不变量

```
不变量：nums[0..slow-1] 是已处理好的"结果区"
约束：fast >= slow 始终成立
      结果区的元素满足某种条件（不含目标值、无重复等）
```

每次循环：
1. fast 判断当前元素是否"有价值"
2. 若有价值 → 写入 `nums[slow]`，`slow += 1`
3. 若无价值 → 跳过，`slow` 不变

结束时，`slow` 就是新数组的长度。

---

## 7.3 核心题目：移除元素的变体集合

**变体一：移除所有等于 val 的元素**

```python
def remove_element(nums, val):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
    return slow
```

**变体二：移除重复元素（每个元素只保留一次）**

```python
def remove_duplicates(nums):
    if not nums:
        return 0
    slow = 1
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow - 1]:
            nums[slow] = nums[fast]
            slow += 1
    return slow
```

**变体三：保留重复元素最多 k 次**

```python
def remove_duplicates_k(nums, k):
    slow = 0
    for fast in range(len(nums)):
        # 结果区长度 < k，或者当前元素与结果区倒数第k个不同
        if slow < k or nums[fast] != nums[slow - k]:
            nums[slow] = nums[fast]
            slow += 1
    return slow

# 测试
nums = [1,1,1,2,2,3]
print(remove_duplicates_k(nums, 2), nums)  # 5 [1,1,2,2,3,3]

nums = [0,0,1,1,1,1,2,3,3]
print(remove_duplicates_k(nums, 2), nums)  # 7 [0,0,1,1,2,3,3,...]
```

**统一公式 `slow < k or nums[fast] != nums[slow - k]` 的理解**：
- `slow < k`：结果区还没有 k 个元素，当前元素无论如何都写入
- `nums[fast] != nums[slow - k]`：当前元素与结果区倒数第 k 个不同，说明加入后不会超过 k 次

---

## 7.4 分区操作：按条件将数组分为两部分

**问题**：将数组中所有偶数放前面，奇数放后面（相对顺序可不保证）。

```python
def partition_even_odd(nums):
    """对撞指针分区（不稳定）"""
    left, right = 0, len(nums) - 1
    while left < right:
        while left < right and nums[left] % 2 == 0:
            left += 1
        while left < right and nums[right] % 2 == 1:
            right -= 1
        if left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
    return nums

def partition_even_odd_stable(nums):
    """写指针分区（稳定，保持相对顺序）"""
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] % 2 == 0:  # 偶数移到前面
            nums[slow] = nums[fast]
            slow += 1
    # 问题：原来的偶数位被覆盖，奇数未正确放置
    # 稳定分区需要额外空间，或使用插入排序思想
```

**注意**：稳定的原地分区不能简单地用写指针，因为会覆盖原数据。真正的稳定原地分区需要 O(n log n) 时间（类似归并排序）。若允许不稳定，对撞指针是最优的。

---

## 7.5 三色排序（荷兰国旗问题预告）

将只含 0、1、2 的数组排序，要求一趟扫描完成。

```python
def sort_colors(nums):
    """三指针：low/mid/high"""
    low = mid = 0
    high = len(nums) - 1

    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
            # 注意：mid 不前进，因为交换来的数还未检查

# 测试
nums = [2, 0, 2, 1, 1, 0]
sort_colors(nums)
print(nums)  # [0, 0, 1, 1, 2, 2]
```

**不变量**：
- `nums[0..low-1]` 都是 0
- `nums[low..mid-1]` 都是 1
- `nums[mid..high]` 未处理
- `nums[high+1..n-1]` 都是 2

`mid` 遇到 2 时交换到 `high`，不前进——这是与写指针模式的关键区别：**被换来的元素还未检查**，需要在原地再次判断。

---

## 7.6 原地删除有序链表中的重复节点

```python
def delete_duplicates(head):
    """链表版的原地去重"""
    curr = head
    while curr and curr.next:
        if curr.val == curr.next.val:
            curr.next = curr.next.next  # 删除重复节点
        else:
            curr = curr.next  # 只在不重复时前进
    return head

def delete_duplicates_all(head):
    """删除所有出现超过一次的节点"""
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy

    while prev.next:
        curr = prev.next
        # 检查是否有重复
        if curr.next and curr.val == curr.next.val:
            # 跳过所有相同值的节点
            while curr.next and curr.val == curr.next.val:
                curr = curr.next
            prev.next = curr.next  # 直接跳过所有重复节点
        else:
            prev = prev.next

    return dummy.next
```

---

## 7.7 原地旋转数组

将数组向右旋转 k 步，不使用额外空间。

```python
def rotate(nums, k):
    """三次翻转法：O(n) 时间，O(1) 空间"""
    n = len(nums)
    k = k % n  # 处理 k >= n 的情况

    def reverse(left, right):
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    # 翻转整个数组
    reverse(0, n - 1)
    # 翻转前 k 个
    reverse(0, k - 1)
    # 翻转后 n-k 个
    reverse(k, n - 1)

# 测试
nums = [1, 2, 3, 4, 5, 6, 7]
rotate(nums, 3)
print(nums)  # [5, 6, 7, 1, 2, 3, 4]
```

**推导**：
```
原始：  1 2 3 4 5 6 7
全翻转：7 6 5 4 3 2 1
翻k个：5 6 7 4 3 2 1
翻其余：5 6 7 1 2 3 4 ✓
```

---

## 7.8 写指针 vs 三指针：选择指南

| 场景 | 模式 | 指针数 |
|------|------|--------|
| 过滤（保留满足条件的） | 写指针（slow/fast） | 2 |
| 两类分区 | 对撞指针或写指针 | 2 |
| 三类分区（如颜色排序） | 三指针 (low/mid/high) | 3 |
| 原地旋转 | 多次翻转 | 1（逻辑上） |

---

## 7.9 本章小结

原地修改数组的核心工具：

1. **写指针模式**：`nums[0..slow-1]` 是结果区，fast 扫描全部
2. **不变量设计**：明确 slow 左侧元素的性质，是正确性的保证
3. **三指针分区**：用于三类元素的分区（荷兰国旗）
4. **翻转法**：原地旋转的经典技巧

**下一章：回文串与字符串双指针——中心扩展与双端验证**

---

## LeetCode 推荐题目

- [27. 移除元素](https://leetcode.cn/problems/remove-element/) ⭐
- [26. 删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/) ⭐
- [80. 删除有序数组中的重复项 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/) ⭐⭐
- [75. 颜色分类](https://leetcode.cn/problems/sort-colors/) ⭐⭐
- [189. 轮转数组](https://leetcode.cn/problems/rotate-array/) ⭐⭐
- [83. 删除排序链表中的重复元素](https://leetcode.cn/problems/remove-duplicates-from-sorted-list/) ⭐
- [82. 删除排序链表中的重复元素 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/) ⭐⭐
