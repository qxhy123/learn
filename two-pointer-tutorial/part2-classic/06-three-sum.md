# 第6章：三数之和与N数之和

## 6.1 问题引入

给定数组 `nums`，找出所有满足 `a + b + c = 0` 的不重复三元组。

```python
# 暴力 O(n³)
def three_sum_brute(nums):
    n = len(nums)
    result = set()
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if nums[i] + nums[j] + nums[k] == 0:
                    result.add(tuple(sorted([nums[i], nums[j], nums[k]])))
    return [list(t) for t in result]
```

三层循环 + 用集合去重——丑陋且低效。

**核心思路**：枚举降维。固定一个数 `nums[i]`，问题转化为在剩余数组中找两数之和为 `-nums[i]`——这正是上一章的对撞指针！

---

## 6.2 排序 + 枚举 + 对撞指针

```
总策略：
1. 排序（使去重和对撞指针成立）
2. 枚举第一个数 nums[i]
3. 在 i+1..n-1 范围内用对撞指针找两数之和 = -nums[i]
4. 所有层面都做去重
```

```python
def three_sum(nums):
    nums.sort()
    result = []
    n = len(nums)

    for i in range(n - 2):
        # 剪枝1：最小三数之和已大于0，后面不可能有解
        if nums[i] > 0:
            break

        # 去重：跳过重复的第一个数
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, n - 1

        while left < right:
            s = nums[i] + nums[left] + nums[right]

            if s == 0:
                result.append([nums[i], nums[left], nums[right]])
                # 去重：跳过重复的 left 和 right
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1

    return result

# 测试
print(three_sum([-1, 0, 1, 2, -1, -4]))
# [[-1, -1, 2], [-1, 0, 1]]
print(three_sum([0, 0, 0]))
# [[0, 0, 0]]
print(three_sum([]))
# []
```

**复杂度**：O(n²)，其中排序 O(n log n)，外层枚举 O(n)，内层对撞 O(n)，总体 O(n²)。

---

## 6.3 去重的精细化处理

去重是三数之和最容易出错的地方，逐个分析：

**第一层去重（枚举 i）**：

```python
if i > 0 and nums[i] == nums[i - 1]:
    continue
```

为什么是 `i > 0` 而不是 `i >= 0`？
- `i = 0` 时不跳过，因为是第一个元素
- `i > 0` 且 `nums[i] == nums[i-1]`：以 `nums[i]` 开头的三元组已经在 `i-1` 时处理过了

**第二层去重（找到答案后）**：

```python
while left < right and nums[left] == nums[left + 1]:
    left += 1
while left < right and nums[right] == nums[right - 1]:
    right -= 1
left += 1
right -= 1
```

必须先跳过重复，**再**移动到下一个不重复位置。

**常见错误**：

```python
# 错误：先移动后去重
left += 1
right -= 1
while left < right and nums[left] == nums[left - 1]:  # 错！应该比较 left+1
    left += 1
```

---

## 6.4 剪枝优化

```python
for i in range(n - 2):
    # 剪枝1：最小值已大于0
    if nums[i] > 0:
        break

    # 剪枝2：即使取最大的三个数也不够
    if nums[i] + nums[n-1] + nums[n-2] < 0:
        continue

    # 去重
    if i > 0 and nums[i] == nums[i - 1]:
        continue

    # ... 对撞指针 ...
```

剪枝2 的效果：当 `nums[i]` 太小，即使搭配最大的两个数也凑不到 0，直接跳过。

---

## 6.5 最接近的三数之和

```python
def three_sum_closest(nums, target):
    nums.sort()
    n = len(nums)
    closest = float('inf')

    for i in range(n - 2):
        left, right = i + 1, n - 1

        while left < right:
            s = nums[i] + nums[left] + nums[right]

            if abs(s - target) < abs(closest - target):
                closest = s

            if s < target:
                left += 1
            elif s > target:
                right -= 1
            else:
                return s  # 完全等于

    return closest

# 测试
print(three_sum_closest([-1, 2, 1, -4], 1))  # 2
```

---

## 6.6 四数之和：N数之和的通用框架

```python
def four_sum(nums, target):
    nums.sort()
    return k_sum(nums, target, 4, 0)

def k_sum(nums, target, k, start):
    result = []
    n = len(nums)

    if k == 2:
        # Base case：对撞指针
        left, right = start, n - 1
        while left < right:
            s = nums[left] + nums[right]
            if s == target:
                result.append([nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]: left += 1
                while left < right and nums[right] == nums[right - 1]: right -= 1
                left += 1
                right -= 1
            elif s < target:
                left += 1
            else:
                right -= 1
        return result

    # 递归：枚举第一个数，转化为 (k-1) 数之和
    for i in range(start, n - k + 1):
        # 剪枝
        if nums[i] * k > target: break
        if nums[i + k - 1] * k < target: continue  # 即使取最小k个也不够
        # 去重
        if i > start and nums[i] == nums[i - 1]: continue

        for sub in k_sum(nums, target - nums[i], k - 1, i + 1):
            result.append([nums[i]] + sub)

    return result

# 测试
print(four_sum([1, 0, -1, 0, -2, 2], 0))
# [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
```

**复杂度**：
- 两数之和：O(n)
- 三数之和：O(n²)
- 四数之和：O(n³)
- K数之和：O(n^(k-1))

递归框架使代码对任意 K 都适用，是一个非常优雅的设计。

---

## 6.7 N数之和的思维模型

```
K数之和 = 枚举 + (K-1)数之和
              = 枚举 + 枚举 + ... + 2数之和（对撞指针）
              = O(n^(K-1)) 时间，O(K) 递归栈空间
```

这是一种**递归降维**策略，每次枚举将问题规模减少 1 维。

**关键设计点**：
1. 排序是前提
2. 每层去重（`if i > start and nums[i] == nums[i-1]`）
3. 剪枝提前终止（最小值过大/最大值过小）
4. Base case 使用对撞指针

---

## 6.8 本章小结

| 问题 | 方法 | 时间 |
|------|------|------|
| 两数之和（有序） | 对撞指针 | O(n) |
| 三数之和 | 排序 + 枚举 + 对撞 | O(n²) |
| 四数之和 | 排序 + 枚举 + 枚举 + 对撞 | O(n³) |
| K数之和 | 递归降维 + 对撞 | O(n^(K-1)) |

**下一章：原地修改数组——同向双指针的写指针模式**

---

## LeetCode 推荐题目

- [15. 三数之和](https://leetcode.cn/problems/3sum/) ⭐⭐
- [16. 最接近的三数之和](https://leetcode.cn/problems/3sum-closest/) ⭐⭐
- [18. 四数之和](https://leetcode.cn/problems/4sum/) ⭐⭐
- [923. 三数之和的多种可能](https://leetcode.cn/problems/3sum-with-multiplicity/) ⭐⭐⭐
