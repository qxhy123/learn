# 第3章：同向双指针

## 3.1 问题引入：移除元素

给定数组 `nums` 和值 `val`，原地移除所有等于 `val` 的元素，返回移除后数组的新长度。

```python
# 暴力做法：用额外数组
def remove_element_extra(nums, val):
    result = [x for x in nums if x != val]
    # 但题目要求原地操作，不能用额外空间
```

**约束**：不能使用额外数组，必须原地操作，O(1) 空间。

如何在原地操作？我们需要两个角色：
- 一个指针**读**（扫描所有元素）
- 一个指针**写**（记录有效元素的写入位置）

这就是同向双指针的本质。

---

## 3.2 核心思维模型：读写分离

```
nums = [3, 2, 2, 3]，val = 3

读指针 read 扫描每个元素：
  - 遇到不等于 val 的元素 → 写入 write 位置，write 前进
  - 遇到等于 val 的元素 → 跳过，write 不动

过程：
read=0: nums[0]=3，等于val，跳过
  [3, 2, 2, 3]
   r
   w

read=1: nums[1]=2，不等于val，写入 write=0
  [2, 2, 2, 3]  ← nums[0] 被覆盖为 2
      r
   w→

read=2: nums[2]=2，不等于val，写入 write=1
  [2, 2, 2, 3]  ← nums[1] 被覆盖为 2
         r
      w→

read=3: nums[3]=3，等于val，跳过
  [2, 2, 2, 3]
            r
         w

结束：write=2，有效长度为 2
```

---

## 3.3 代码实现

```python
def remove_element(nums, val):
    write = 0  # 慢指针：下一个写入位置

    for read in range(len(nums)):  # 快指针：扫描所有元素
        if nums[read] != val:
            nums[write] = nums[read]
            write += 1

    return write

# 测试
nums = [3, 2, 2, 3]
k = remove_element(nums, 3)
print(k, nums[:k])  # 2 [2, 2]

nums = [0, 1, 2, 2, 3, 0, 4, 2]
k = remove_element(nums, 2)
print(k, nums[:k])  # 5 [0, 1, 3, 0, 4]
```

**不变量（Invariant）**：`nums[0..write-1]` 始终存储有效元素。这个不变量是同向双指针正确性的核心。

---

## 3.4 同向双指针的通用模板

```python
def same_direction_template(arr):
    slow = 0  # 慢指针：维护"结果区"的边界

    for fast in range(len(arr)):
        if condition(arr[fast]):        # 快指针发现"有价值"的元素
            arr[slow] = arr[fast]       # 写入慢指针位置
            slow += 1                   # 慢指针前进

    return slow  # 结果区大小
```

**慢指针的语义**：`arr[0..slow-1]` 是已处理好的区域（满足某种条件的元素集合）。

---

## 3.5 经典变体一：有序数组去重

```python
def remove_duplicates(nums):
    if not nums:
        return 0

    slow = 1  # 第一个元素永远保留，slow 从 1 开始

    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow - 1]:  # 与结果区最后一个元素比较
            nums[slow] = nums[fast]
            slow += 1

    return slow

# 测试
nums = [1, 1, 2]
print(remove_duplicates(nums), nums[:2])  # 2 [1, 2]

nums = [0,0,1,1,1,2,2,3,3,4]
print(remove_duplicates(nums), nums[:5])  # 5 [0,1,2,3,4]
```

**进阶**：允许重复最多 k 次

```python
def remove_duplicates_k(nums, k=2):
    slow = 0

    for fast in range(len(nums)):
        # 结果区少于 k 个元素，或当前元素与结果区倒数第 k 个不同
        if slow < k or nums[fast] != nums[slow - k]:
            nums[slow] = nums[fast]
            slow += 1

    return slow

# 测试（k=2）
nums = [1,1,1,2,2,3]
print(remove_duplicates_k(nums, 2), nums[:5])  # 5 [1,1,2,2,3]
```

**妙处**：`nums[fast] != nums[slow - k]` 这个条件统一处理了所有情况，是同向双指针不变量设计的精华。

---

## 3.6 经典变体二：移动零

将数组中所有 0 移动到末尾，保持非零元素的相对顺序。

```python
def move_zeroes(nums):
    slow = 0  # 下一个非零元素的写入位置

    # 第一步：把所有非零元素移到前面
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1

    # 第二步：后面全填 0
    for i in range(slow, len(nums)):
        nums[i] = 0

# 测试
nums = [0, 1, 0, 3, 12]
move_zeroes(nums)
print(nums)  # [1, 3, 12, 0, 0]
```

**优化版（减少写操作，使用交换）**：

```python
def move_zeroes_swap(nums):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
```

两种方式的区别：
- 前者：`slow` 位置先被覆盖后补零，写操作约 n 次
- 后者：交换，写操作约 2×非零数量 次

当非零元素较少时，交换版更优；当非零元素较多时，覆盖版更优。

---

## 3.7 经典变体三：比较含退格的字符串

给定两个字符串 `s` 和 `t`，`#` 表示退格键，判断它们是否相等。

```python
# 方法一：模拟栈，O(n) 空间
def backspace_compare_stack(s, t):
    def process(string):
        stack = []
        for c in string:
            if c != '#':
                stack.append(c)
            elif stack:
                stack.pop()
        return stack
    return process(s) == process(t)

# 方法二：从后向前的同向双指针，O(1) 空间
def backspace_compare(s, t):
    i, j = len(s) - 1, len(t) - 1
    skip_s = skip_t = 0

    while i >= 0 or j >= 0:
        # 找到 s 中下一个有效字符
        while i >= 0:
            if s[i] == '#':
                skip_s += 1
                i -= 1
            elif skip_s > 0:
                skip_s -= 1
                i -= 1
            else:
                break

        # 找到 t 中下一个有效字符
        while j >= 0:
            if t[j] == '#':
                skip_t += 1
                j -= 1
            elif skip_t > 0:
                skip_t -= 1
                j -= 1
            else:
                break

        # 比较两个有效字符
        if i >= 0 and j >= 0 and s[i] != t[j]:
            return False
        if (i >= 0) != (j >= 0):  # 一个耗尽，另一个没有
            return False

        i -= 1
        j -= 1

    return True

# 测试
print(backspace_compare("ab#c", "ad#c"))   # True
print(backspace_compare("ab##", "c#d#"))   # True
print(backspace_compare("a##c", "#a#c"))   # True
print(backspace_compare("a#c", "b"))       # False
```

**为什么从后向前**：退格影响前面的字符，从后向前扫描时可以计数退格数量，抵消前面的字符。

---

## 3.8 同向 vs 对撞：如何选择？

| 特征 | 对撞指针 | 同向双指针 |
|------|----------|------------|
| 指针方向 | 相向 | 同向 |
| 初始位置 | 两端 | 同侧（通常从左） |
| 适用场景 | 有序数组，找配对 | 过滤/修改数组，滑动窗口 |
| 不变量 | 答案在 [left, right] 区间内 | slow 左侧是已处理好的区域 |
| 典型例子 | 两数之和、回文 | 去重、移除元素、滑动窗口 |

**判断依据**：
- 需要找**两端的关系**（对称、配对）→ 对撞指针
- 需要**维护一个区域**（过滤、子数组）→ 同向双指针

---

## 3.9 本章小结

同向双指针的核心：
1. **slow 维护不变量**：左侧是已处理好的有效区域
2. **fast 负责探索**：扫描每个元素，判断是否有价值
3. **写操作**：仅当 fast 找到有价值的元素时，写入 slow 并推进 slow

**下一章：快慢指针——链表中的步差魔法**

---

## LeetCode 推荐题目

- [27. 移除元素](https://leetcode.cn/problems/remove-element/) ⭐
- [283. 移动零](https://leetcode.cn/problems/move-zeroes/) ⭐
- [26. 删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/) ⭐
- [80. 删除有序数组中的重复项 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/) ⭐⭐
- [844. 比较含退格的字符串](https://leetcode.cn/problems/backspace-string-compare/) ⭐⭐
