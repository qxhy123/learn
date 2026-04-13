# 第5章：子集问题

## 5.1 基础子集（无重复元素）

**问题**：给定一个不含重复元素的整数数组，返回所有可能的子集（幂集）。（LeetCode 78）

```
输入：nums = [1, 2, 3]
输出：[[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]
共 2^3 = 8 个子集
```

### 方法一：回溯（推荐）

```python
def subsets(nums):
    """
    回溯法：每个节点都收集
    时间 O(n × 2^n)，空间 O(n)
    """
    result = []
    
    def backtrack(start, path):
        result.append(path[:])  # 每个状态都是一个有效子集
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result
```

### 方法二：位运算

```python
def subsets_bit(nums):
    """
    位运算：n 位二进制数的每个值对应一个子集
    1 = 选，0 = 不选
    """
    n = len(nums)
    result = []
    
    for mask in range(1 << n):  # 0 到 2^n - 1
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)
    
    return result

# 示例：nums=[1,2,3], mask=5=101b → 选第0位和第2位 → [1,3]
```

### 方法三：迭代（逐个添加）

```python
def subsets_iterative(nums):
    """
    迭代法：每添加一个新元素，将它加入所有现有子集
    """
    result = [[]]
    
    for num in nums:
        # 将 num 添加到现有每个子集，生成新子集
        result += [subset + [num] for subset in result]
    
    return result

# 过程：
# 初始：[[]]
# 加入1：[[], [1]]
# 加入2：[[], [1], [2], [1,2]]
# 加入3：[[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]
```

## 5.2 含重复元素的子集

**问题**：给定可能含重复元素的整数数组，返回所有不重复的子集。（LeetCode 90）

```
输入：nums = [1, 2, 2]
输出：[[], [1], [2], [1,2], [2,2], [1,2,2]]
注意：[2] 只出现一次，[2,2] 只出现一次
```

**关键**：排序 + 同层去重

```python
def subsets_with_dup(nums):
    """
    含重复元素的子集
    关键：排序后，同层跳过相同元素
    """
    nums.sort()  # 必须排序
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        
        for i in range(start, len(nums)):
            # 同层去重：i > start（不是 i > 0）
            if i > start and nums[i] == nums[i - 1]:
                continue
            
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result
```

**为什么是 `i > start` 而不是 `i > 0`？**

```
nums = [1, 2, 2], 已排序

同层（start=0）:
  i=0: 选 1 → [1]
  i=1: 选 2 → [2]
  i=2: nums[2]==nums[1] 且 i>start(2>0) → 跳过 ✓

不同层（path=[1], start=1）:
  i=1: 选 2 → [1,2]
  i=2: nums[2]==nums[1] 且 i>start(2>1) → 跳过 ✓
  
若改为 i>0:
  不同层（path=[], start=0）i=0时，下一层 start=1
  不同层（path=[1], start=1）i=1时，是合法的！但 i>0 成立会错误跳过
```

## 5.3 子集的大小限制

**问题**：返回所有大小为 k 的子集（即组合问题，见下章）。

```python
def subsets_of_size_k(nums, k):
    result = []
    
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        # 剪枝：剩余元素不足
        if len(nums) - start < k - len(path):
            return
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result
```

## 5.4 子集的变体：幂集的字典序

```python
def subsets_sorted(nums):
    """返回按字典序排列的所有子集"""
    nums.sort()
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    # 由于从小到大遍历，结果天然按字典序
    return result
```

## 5.5 子集的应用：最大 XOR 子集

```python
def max_subset_xor(nums):
    """
    找出异或值最大的子集
    思路：线性基（高斯消元）
    这里用回溯求解（适合小规模）
    """
    max_xor = [0]
    
    def backtrack(start, current_xor):
        max_xor[0] = max(max_xor[0], current_xor)
        
        for i in range(start, len(nums)):
            backtrack(i + 1, current_xor ^ nums[i])
    
    backtrack(0, 0)
    return max_xor[0]
```

## 5.6 子集和问题

**问题**：是否存在子集，其元素之和等于给定值 target？

```python
def subset_sum_exists(nums, target):
    """
    子集和：是否存在和为 target 的子集
    回溯 + 剪枝（也可以 DP 解决）
    """
    nums.sort(reverse=True)  # 降序排列，便于剪枝
    
    def backtrack(start, remaining):
        if remaining == 0:
            return True
        if remaining < 0:
            return False
        
        for i in range(start, len(nums)):
            if nums[i] > remaining:
                continue
            if backtrack(i + 1, remaining - nums[i]):
                return True
        
        return False
    
    return backtrack(0, target)

def all_subset_sums(nums, target):
    """返回所有和为 target 的子集"""
    nums.sort()
    result = []
    
    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        
        for i in range(start, len(nums)):
            if nums[i] > remaining:
                break  # 排序后可剪枝
            # 去重
            if i > start and nums[i] == nums[i-1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path, remaining - nums[i])
            path.pop()
    
    backtrack(0, [], target)
    return result
```

## 5.7 三种方法对比

| 方法 | 时间 | 空间 | 适用场景 |
|------|------|------|---------|
| 回溯 | O(n×2^n) | O(n) | 通用，可添加剪枝 |
| 位运算 | O(n×2^n) | O(1) | n ≤ 20，代码简洁 |
| 迭代 | O(n×2^n) | O(2^n) | 理解子集构建过程 |

## 小结

- 子集回溯：每个节点收集（不等到叶节点）
- 含重复元素：排序 + `i > start` 去重
- 位运算：简洁，但 n 不能太大（≤ 20）

## 练习

1. 实现"返回所有子集中元素之和为偶数的子集"
2. 用回溯计算幂集的元素总数（验证 = 2^n）
3. 解决"划分等和子集"（LeetCode 416）：是否能把数组分为两个等和子集

---

**上一章（Part 1）：** [复杂度分析](../part1-foundations/04-complexity-analysis.md) | **下一章：** [组合问题](06-combinations.md)
