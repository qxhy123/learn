# 第2章：对撞指针

## 2.1 问题引入：有序数组中的两数之和

给定一个**升序排列**的数组 `numbers`，找到两个数使它们之和等于 `target`，返回它们的下标（1-indexed）。

```python
# 暴力 O(n²)
def two_sum_brute(numbers, target):
    n = len(numbers)
    for i in range(n):
        for j in range(i+1, n):
            if numbers[i] + numbers[j] == target:
                return [i+1, j+1]
```

**观察**：数组是有序的。如果 `numbers[i] + numbers[j] > target`，那么任何 `j' > j` 的配对 `(i, j')` 也一定过大——这是一个强烈的单调性信号！

---

## 2.2 核心思维模型：搜索空间的折叠

将所有可能的 `(left, right)` 对排列成一个三角形：

```
        right
         ←
↓  (0,n-1)(0,n-2)...
   (1,n-1)(1,n-2)...
   ...
left
```

对撞指针从右上角 `(0, n-1)` 出发：

- **当前和偏大** (`sum > target`)：`right--`，整列舍弃（因为 `left` 固定时更大的 `right` 只会更大）
- **当前和偏小** (`sum < target`)：`left++`，整行舍弃（因为 `right` 固定时更小的 `left` 只会更小）
- **相等**：找到答案

每次移动至少排除一行或一列，共 n 行 n 列，所以最多移动 2n 次。

```
搜索空间可视化（n=5, target=9, 数组=[2,3,4,6,8]）：

right→  0  1  2  3  4
        2  3  4  6  8
left↓
  0(2)  ×  4  ×  ×  ✓(10>9,right--)
                      ↓
                   (0,3): 2+6=8<9, left++
                      ↓
                   (1,3): 3+6=9=target ✓
```

---

## 2.3 代码实现

```python
def two_sum(numbers, target):
    left, right = 0, len(numbers) - 1

    while left < right:
        current_sum = numbers[left] + numbers[right]

        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            left += 1   # 当前和太小，增大左边
        else:
            right -= 1  # 当前和太大，减小右边

    return []  # 题目保证有解，理论上不到这里
```

**为什么 `left < right` 而不是 `left <= right`？**

如果 `left == right`，表示两个指针指向同一个元素。题目要求找两个**不同**位置的数，所以当指针相遇时，搜索结束。

---

## 2.4 对撞指针的通用模板

```python
def opposite_pointers_template(arr, condition):
    left, right = 0, len(arr) - 1

    while left < right:
        # 1. 计算当前状态
        state = compute(arr[left], arr[right])

        # 2. 根据状态决定移动哪个指针
        if state == TARGET:
            # 处理答案
            process(left, right)
            # 根据问题决定是否继续
            left += 1  # 或 right -= 1，或两者
        elif state < TARGET:
            left += 1   # 需要更大的值
        else:
            right -= 1  # 需要更小的值

    return result
```

**关键决策**：找到答案后如何继续？
- **找所有答案**：`left += 1` 和 `right -= 1` 同时移动（避免重复）
- **找唯一答案**：直接返回
- **计数/求和**：更新计数器后继续移动

---

## 2.5 经典变体一：有序数组平方

给定有序数组（可能含负数），返回每个元素平方后的有序数组。

```python
def sorted_squares(nums):
    n = len(nums)
    result = [0] * n
    left, right = 0, n - 1
    pos = n - 1  # 从后往前填充

    while left <= right:
        left_sq = nums[left] ** 2
        right_sq = nums[right] ** 2

        if left_sq > right_sq:
            result[pos] = left_sq
            left += 1
        else:
            result[pos] = right_sq
            right -= 1

        pos -= 1

    return result

# 测试
print(sorted_squares([-4, -1, 0, 3, 10]))  # [0, 1, 9, 16, 100]
print(sorted_squares([-7, -3, 2, 3, 11]))  # [4, 9, 9, 49, 121]
```

**思维分析**：原数组有序，平方后最大值一定在两端。用对撞指针从两端取最大值，从后往前填充结果数组。时间 O(n)，空间 O(n)（输出数组）。

---

## 2.6 经典变体二：验证回文串

给定字符串，只考虑字母和数字，判断是否回文。

```python
def is_palindrome(s):
    left, right = 0, len(s) - 1

    while left < right:
        # 跳过非字母数字字符
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True

# 测试
print(is_palindrome("A man, a plan, a canal: Panama"))  # True
print(is_palindrome("race a car"))                      # False
```

**陷阱**：内层 while 循环中必须保留 `left < right` 的边界检查，否则当整个字符串都是非字母数字时会越界。

---

## 2.7 经典变体三：盛最多水的容器

给定高度数组，找两条线使得它们与 x 轴形成的容器能盛最多水。

```python
def max_area(height):
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        # 水量 = 宽度 × 较短的高度
        water = (right - left) * min(height[left], height[right])
        max_water = max(max_water, water)

        # 移动较短的那一侧——关键决策
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water

# 测试
print(max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]))  # 49
```

**为什么移动较短的一侧是正确的？**

设当前 `height[left] < height[right]`：
- 固定 `left`，移动 `right` 向左：宽度减小，高度受限于 `height[left]`，水量只会减少
- 固定 `right`，移动 `left` 向右：宽度减小，但高度**有可能增大**，水量可能增大

因此移动**较短**的一侧是唯一可能找到更优解的策略。

**这是对撞指针正确性的典型证明方式：通过排除法证明被丢弃的情况不可能是答案。**

---

## 2.8 对撞指针的正确性证明框架

对撞指针的难点不在于实现，而在于**证明不会漏掉答案**。通用证明思路：

```
反证法：假设答案是 (i*, j*)，证明算法一定会访问到它。

当 left < i* 时：
  - 若 sum(left, j*) >= target：right 会向左移动
    - 能到达 right = j* 吗？
    - 能，因为 sum(left, right) 单调递减当 right 减小
  - 若 sum(left, j*) < target：left 会向右移动（缩短到 i*）
    - 此时 right 仍然 >= j*（因为还没到达 j*）

最终，当 left = i* 时，right >= j*。
而 sum(i*, right) 相对 sum(i*, j*) >= target（因为 right >= j* 且数组有序）。
所以 right 会单调减小直到等于 j*。
```

---

## 2.9 复杂度分析

| 场景 | 时间 | 空间 | 说明 |
|------|------|------|------|
| 基本对撞 | O(n) | O(1) | 每次移动一个指针 |
| 含内层跳过 | O(n) | O(1) | 所有字符仍只被访问一次 |
| 平方排序 | O(n) | O(n) | 需要输出数组 |

---

## 2.10 本章小结

对撞指针的核心：
1. **从两端出发**，相向移动
2. **移动决策**依赖于当前状态与目标的关系
3. **正确性保证**：被丢弃的配对在数学上不可能是答案
4. **适用前提**：数组有序，或状态函数具备单调性

**下一章：同向双指针——维护窗口不变量的艺术**

---

## LeetCode 推荐题目

- [167. 两数之和 II](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/) ⭐
- [977. 有序数组的平方](https://leetcode.cn/problems/squares-of-a-sorted-array/) ⭐
- [125. 验证回文串](https://leetcode.cn/problems/valid-palindrome/) ⭐
- [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/) ⭐⭐
- [680. 验证回文串 II](https://leetcode.cn/problems/valid-palindrome-ii/) ⭐⭐
