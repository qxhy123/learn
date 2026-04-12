# 第14章：接雨水与单调性分析

## 14.1 问题引入

**接雨水（LeetCode 42）**：给定 n 个宽度为 1 的柱子，高度由数组 `height` 给出，计算下雨之后能接多少雨水。

```
height = [0,1,0,2,1,0,1,3,2,1,2,1]

       #
   #   ##  #
   ## ###  ###
_________________
能接的雨水 = 6
```

这道题有多种解法，每种都揭示了不同的思维方式。我们从暴力到最优，逐步分析。

---

## 14.2 解法一：暴力（O(n²)）

对于每个位置 `i`，能接的水量 = `min(max_left[i], max_right[i]) - height[i]`，其中：
- `max_left[i]` = `height[0..i]` 的最大值
- `max_right[i]` = `height[i..n-1]` 的最大值

```python
def trap_brute(height):
    n = len(height)
    total = 0

    for i in range(1, n - 1):
        max_left = max(height[:i+1])
        max_right = max(height[i:])
        water = min(max_left, max_right) - height[i]
        total += max(0, water)

    return total
```

O(n²) 的原因：对每个位置都重新计算前后缀最大值。

---

## 14.3 解法二：预处理前后缀最大值（O(n) 时间，O(n) 空间）

```python
def trap_prefix(height):
    n = len(height)
    if n == 0:
        return 0

    # 预处理
    max_left = [0] * n
    max_right = [0] * n

    max_left[0] = height[0]
    for i in range(1, n):
        max_left[i] = max(max_left[i-1], height[i])

    max_right[n-1] = height[n-1]
    for i in range(n-2, -1, -1):
        max_right[i] = max(max_right[i+1], height[i])

    # 计算水量
    total = 0
    for i in range(1, n - 1):
        water = min(max_left[i], max_right[i]) - height[i]
        total += max(0, water)

    return total

# 测试
print(trap_prefix([0,1,0,2,1,0,1,3,2,1,2,1]))  # 6
print(trap_prefix([4,2,0,3,2,5]))               # 9
```

这个解法清晰正确，但需要 O(n) 额外空间。

---

## 14.4 解法三：对撞指针（O(n) 时间，O(1) 空间）

**核心观察**：位置 `i` 的蓄水量取决于 `min(max_left[i], max_right[i])`，即**较小的那一侧**。

```
对撞指针维护：
  left_max = height[0..left] 的最大值（左侧已知最大）
  right_max = height[right..n-1] 的最大值（右侧已知最大）

当 left_max <= right_max 时：
  - 对于 left 位置，left_max 是左侧最大值
  - 右侧最大值 >= right_max >= left_max
  - 因此 water[left] = left_max - height[left]（由左侧决定）
  - 处理 left，left += 1

当 left_max > right_max 时：
  - 对称地处理 right，right -= 1
```

```python
def trap(height):
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    total = 0

    while left < right:
        left_max = max(left_max, height[left])
        right_max = max(right_max, height[right])

        if left_max <= right_max:
            # 左侧是瓶颈，left 位置的水量由 left_max 决定
            total += left_max - height[left]
            left += 1
        else:
            # 右侧是瓶颈，right 位置的水量由 right_max 决定
            total += right_max - height[right]
            right -= 1

    return total

# 测试
print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))  # 6
print(trap([4,2,0,3,2,5]))               # 9
```

---

## 14.5 对撞指针解法的正确性证明

**命题**：当 `left_max <= right_max` 时，`water[left] = left_max - height[left]`。

**证明**：
- 真实 `water[left]` = `min(真实_max_left[left], 真实_max_right[left]) - height[left]`
- `真实_max_left[left]` = `max(height[0..left])` = `left_max`（因为我们已经更新了 left_max）
- `真实_max_right[left]` = `max(height[left..n-1])` ≥ `max(height[right..n-1])` = `right_max` ≥ `left_max`
- 因此 `min(left_max, 真实_max_right[left])` = `left_max`
- 所以 `water[left]` = `left_max - height[left]` ✓

关键步骤：`真实_max_right[left] >= right_max >= left_max`，所以左侧是瓶颈，水量由 `left_max` 决定。

---

## 14.6 单调性分析：何时选择对撞指针

接雨水是对撞指针的一个深刻案例，它揭示了一个更一般的原则：

**原则**：当一个量由两侧某种信息的"较小值"或"较大值"决定时，可以用对撞指针，每次处理由"确定"侧决定的那个位置。

类似的问题：

```
盛最多水：min(height[left], height[right]) × 宽度
  → 移动较小的一侧（因为较大的一侧已经是瓶颈，扩展它无意义）

接雨水：min(max_left, max_right) - height[i]
  → 处理较小max的那一侧（另一侧的真实max只会更大）
```

---

## 14.7 变体：直方图中最大矩形面积

**问题（LeetCode 84）**：在直方图中找面积最大的矩形。

```python
def largest_rectangle_area(heights):
    """单调栈：O(n) 时间，O(n) 空间"""
    stack = []  # 单调递增栈，存下标
    max_area = 0
    heights = heights + [0]  # 哨兵：强制处理所有剩余元素

    for right in range(len(heights)):
        while stack and heights[stack[-1]] > heights[right]:
            height = heights[stack.pop()]
            width = right if not stack else right - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(right)

    return max_area

# 测试
print(largest_rectangle_area([2,1,5,6,2,3]))  # 10（5×2 或 6×1 对应的矩形是10）
print(largest_rectangle_area([2,4]))           # 4
```

**思维连接**：最大矩形与接雨水是"互补"问题：
- 接雨水：被夹住的凹陷区域
- 最大矩形：向两侧延伸的凸出区域

---

## 14.8 变体：柱状图中的最大正方形（二维接雨水）

```python
def maximal_rectangle(matrix):
    """将每行转化为直方图问题"""
    if not matrix:
        return 0

    n_cols = len(matrix[0])
    heights = [0] * n_cols
    max_area = 0

    for row in matrix:
        # 更新直方图高度
        for j in range(n_cols):
            heights[j] = heights[j] + 1 if row[j] == '1' else 0

        # 对当前直方图求最大矩形
        max_area = max(max_area, largest_rectangle_area(heights))

    return max_area
```

---

## 14.9 三种解法对比

| 解法 | 时间 | 空间 | 思维难度 |
|------|------|------|----------|
| 暴力 | O(n²) | O(1) | 低 |
| 前后缀预处理 | O(n) | O(n) | 低 |
| 对撞指针 | O(n) | O(1) | 高 |

**面试建议**：先写前后缀预处理版（容易理解且正确），再优化为对撞指针版。能完整证明对撞指针的正确性，会给面试官留下深刻印象。

---

## 14.10 本章小结

接雨水问题揭示的双指针哲学：

1. **由较小值决定的量** → 对撞指针，处理"确定侧"
2. **对撞指针正确性** = 证明被跳过的情况不影响结果
3. **单调栈与双指针** 在"利用单调性"这一点上有深刻联系

**下一章：区间问题与双指针——排序后的区间合并与覆盖**

---

## LeetCode 推荐题目

- [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/) ⭐⭐⭐
- [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/) ⭐⭐
- [84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/) ⭐⭐⭐
- [85. 最大矩形](https://leetcode.cn/problems/maximal-rectangle/) ⭐⭐⭐
