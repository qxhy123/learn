# 第7章：排列问题

## 7.1 全排列（无重复元素）

**问题**：给定不含重复数字的数组，返回所有可能的全排列。（LeetCode 46）

```
输入：nums = [1, 2, 3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
共 3! = 6 种
```

### 方法一：used 数组标记

```python
def permute(nums):
    """
    used 数组标记已使用元素
    时间 O(n × n!)，空间 O(n)
    """
    n = len(nums)
    result = []
    used = [False] * n
    
    def backtrack(path):
        if len(path) == n:
            result.append(path[:])
            return
        
        for i in range(n):
            if used[i]:
                continue
            
            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False
    
    backtrack([])
    return result
```

### 方法二：原地交换

```python
def permute_swap(nums):
    """
    原地交换：将未使用的元素换到前面
    更节省空间（无需 used 数组）
    """
    result = []
    n = len(nums)
    
    def backtrack(start):
        if start == n:
            result.append(nums[:])
            return
        
        for i in range(start, n):
            nums[start], nums[i] = nums[i], nums[start]  # 交换
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]  # 还原
    
    backtrack(0)
    return result
```

**注意**：原地交换法产生的结果顺序与 used 数组法不同（不是字典序）。

## 7.2 含重复元素的全排列

**问题**：给定可能含重复数字的数组，返回所有不重复的全排列。（LeetCode 47）

```
输入：nums = [1, 1, 2]
输出：[[1,1,2],[1,2,1],[2,1,1]]
（[1,1,2] 只出现一次，不是两次）
```

**关键**：排序 + 去重条件 `used[i-1] == False`

```python
def permute_unique(nums):
    """
    含重复元素的全排列
    
    去重条件：
    - nums 已排序
    - nums[i] == nums[i-1]（当前和前一个相同）
    - not used[i-1]（前一个在本树层被撤销过，而非树枝上使用）
    
    等价于：同一树层，不重复使用相同值
    """
    nums.sort()
    n = len(nums)
    result = []
    used = [False] * n
    
    def backtrack(path):
        if len(path) == n:
            result.append(path[:])
            return
        
        for i in range(n):
            if used[i]:
                continue
            
            # 关键去重条件
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue
            
            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False
    
    backtrack([])
    return result
```

**理解去重条件**：

```
nums = [1, 1, 2]，排序后 [1a, 1b, 2]（下标区分两个1）

树层去重（正确）:
  第一层选 1a → 路径 [1a]
  第一层选 1b？1a 已撤销（used[0]=False），1b==1a → 跳过！
  第一层选 2 → 路径 [2]

树枝不去重（正确）:
  路径 [1a, ?, ?] 中第二层：
  1a 已使用（used[0]=True），1b 虽然 1b==1a，但 used[0]=True → 不跳过，允许！
  → 路径 [1a, 1b, 2] 合法
```

## 7.3 下一个排列

**问题**：找到数组的下一个字典序排列（原地修改）。（LeetCode 31）

```
输入：[1,2,3] → [1,3,2]
输入：[3,2,1] → [1,2,3]（已是最大，返回最小）
```

```python
def next_permutation(nums):
    """
    算法：
    1. 从右向左找第一个下降点 i（nums[i] < nums[i+1]）
    2. 从右向左找第一个大于 nums[i] 的位置 j
    3. 交换 nums[i] 和 nums[j]
    4. 反转 i+1 到末尾
    """
    n = len(nums)
    i = n - 2
    
    # 步骤1：找下降点
    while i >= 0 and nums[i] >= nums[i+1]:
        i -= 1
    
    if i >= 0:
        # 步骤2：找右侧第一个大于 nums[i] 的
        j = n - 1
        while nums[j] <= nums[i]:
            j -= 1
        # 步骤3：交换
        nums[i], nums[j] = nums[j], nums[i]
    
    # 步骤4：反转 i+1 到末尾
    left, right = i + 1, n - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1; right -= 1

# 用下一个排列枚举所有排列：
def all_permutations_via_next(nums):
    nums.sort()
    result = [nums[:]]
    while True:
        next_permutation(nums)
        if nums == sorted(nums) and nums == result[0]:  # 回到初始
            break
        result.append(nums[:])
    return result
```

## 7.4 排列序数（第 k 个排列）

**问题**：给定 n 和 k，返回 1..n 的全排列中第 k 个排列（1-indexed）。（LeetCode 60）

```
输入：n=3, k=3
输出："213"（全排列的字典序：123, 132, 213, 231, 312, 321）
```

```python
def get_permutation(n, k):
    """
    数学法：O(n²)，无需枚举所有排列
    
    思路：逐位确定，利用阶乘数系统
    每组有 (n-1)! 个排列
    第 k 个排列的第一位 = candidates[(k-1) // (n-1)!]
    """
    from math import factorial
    
    candidates = list(range(1, n + 1))
    k -= 1  # 转为 0-indexed
    result = []
    
    for i in range(n, 0, -1):
        fact = factorial(i - 1)
        idx = k // fact
        result.append(str(candidates[idx]))
        candidates.pop(idx)
        k %= fact
    
    return ''.join(result)

# 回溯解法（仅供理解，效率低）
def get_permutation_backtrack(n, k):
    result = []
    count = [0]
    found = [False]
    used = [False] * (n + 1)
    
    def backtrack(path):
        if found[0]:
            return
        if len(path) == n:
            count[0] += 1
            if count[0] == k:
                result.extend(path)
                found[0] = True
            return
        
        for i in range(1, n + 1):
            if not used[i]:
                used[i] = True
                path.append(str(i))
                backtrack(path)
                path.pop()
                used[i] = False
    
    backtrack([])
    return ''.join(result)
```

## 7.5 字符串的全排列

```python
def permutations_string(s):
    """字符串的全排列（含重复字符去重）"""
    chars = sorted(s)
    result = []
    used = [False] * len(chars)
    
    def backtrack(path):
        if len(path) == len(chars):
            result.append(''.join(path))
            return
        
        for i in range(len(chars)):
            if used[i]:
                continue
            if i > 0 and chars[i] == chars[i-1] and not used[i-1]:
                continue
            used[i] = True
            path.append(chars[i])
            backtrack(path)
            path.pop()
            used[i] = False
    
    backtrack([])
    return result

print(permutations_string("aab"))
# ['aab', 'aba', 'baa']
```

## 7.6 三类问题的完整对比

```python
# 子集：每个节点收集，i+1
def subsets(nums):
    result = []
    def bt(start, path):
        result.append(path[:])          # 每层都收集
        for i in range(start, len(nums)):
            path.append(nums[i])
            bt(i + 1, path)
            path.pop()
    bt(0, [])
    return result

# 组合：叶节点收集，i+1
def combine(nums, k):
    result = []
    def bt(start, path):
        if len(path) == k:
            result.append(path[:])      # 只在叶节点收集
            return
        for i in range(start, len(nums)):
            path.append(nums[i])
            bt(i + 1, path)
            path.pop()
    bt(0, [])
    return result

# 排列：叶节点收集，从 0 开始，used 标记
def permute(nums):
    result = []
    used = [False] * len(nums)
    def bt(path):
        if len(path) == len(nums):
            result.append(path[:])      # 只在叶节点收集
            return
        for i in range(len(nums)):      # 从 0 开始（不是 start）
            if used[i]: continue
            used[i] = True
            path.append(nums[i])
            bt(path)
            path.pop()
            used[i] = False
    bt([])
    return result
```

## 小结

| 问题 | 去重策略 | 核心代码 |
|------|---------|---------|
| 无重复全排列 | used 数组 | `if used[i]: continue` |
| 含重复全排列 | 排序+去重 | `if nums[i]==nums[i-1] and not used[i-1]: continue` |
| 下一个排列 | 数学 | 找下降点+交换+翻转 |
| 第 k 个排列 | 数学 | 阶乘数系统 |

## 练习

1. 实现"判断 s2 是否是 s1 某个排列的子串"（LeetCode 567）
2. 手动推导 `permute_unique([1,1,2])` 的决策树，标出被剪枝的分支
3. 实现"前一个排列"（与下一个排列相反）

---

**上一章：** [组合](06-combinations.md) | **下一章：** [组合总和（进阶）](08-combination-sum.md)
