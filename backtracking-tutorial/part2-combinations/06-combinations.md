# 第6章：组合问题

## 6.1 基础组合

**问题**：从 1 到 n 中选出 k 个数的所有组合。（LeetCode 77）

```
输入：n=4, k=2
输出：[[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
```

```python
def combine(n, k):
    """
    标准组合回溯
    时间 O(k × C(n,k))，空间 O(k)
    """
    result = []
    
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        
        # 剪枝：从 start 到 n-(k-len(path))+1（含）
        # 剩余元素数 >= 还需要的元素数
        for i in range(start, n - (k - len(path)) + 2):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(1, [])
    return result
```

**剪枝推导**：
```
需要的元素数 = k - len(path)
从 i 开始到 n 共有 n - i + 1 个元素
条件：n - i + 1 >= k - len(path)
解出：i <= n - (k - len(path)) + 1
所以循环上界为 n - (k - len(path)) + 1 + 1 = n - k + len(path) + 2
```

## 6.2 组合总和（可重复选）

**问题**：候选数组 `candidates`（无重复），找出所有和为 `target` 的组合，数字可重复使用。（LeetCode 39）

```
输入：candidates=[2,3,6,7], target=7
输出：[[2,2,3],[7]]
```

```python
def combination_sum(candidates, target):
    """
    可重复选：递归时 start = i（不是 i+1）
    """
    candidates.sort()
    result = []
    
    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break  # 排序后，后续更大，全部剪枝
            
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])  # i 不加1，允许重复
            path.pop()
    
    backtrack(0, [], target)
    return result
```

## 6.3 组合总和（不可重复选，含重复元素）

**问题**：候选数组 `candidates` 可能含重复元素，每个数字只能使用一次，找出所有不重复的和为 `target` 的组合。（LeetCode 40）

```
输入：candidates=[10,1,2,7,6,1,5], target=8
输出：[[1,1,6],[1,2,5],[1,7],[2,6]]
```

```python
def combination_sum2(candidates, target):
    """
    含重复元素，不可重复选
    关键：排序 + 同层去重
    """
    candidates.sort()
    result = []
    
    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break
            
            # 同层去重
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            
            path.append(candidates[i])
            backtrack(i + 1, path, remaining - candidates[i])
            path.pop()
    
    backtrack(0, [], target)
    return result
```

## 6.4 组合总和 III（固定大小）

**问题**：找出所有相加之和为 `n` 的 `k` 个数的组合，只使用 1-9，每个数字最多使用一次。（LeetCode 216）

```
输入：k=3, n=7
输出：[[1,2,4]]

输入：k=3, n=9
输出：[[1,2,6],[1,3,5],[2,3,4]]
```

```python
def combination_sum3(k, n):
    result = []
    
    def backtrack(start, path, remaining):
        if len(path) == k:
            if remaining == 0:
                result.append(path[:])
            return
        
        # 双重剪枝：
        # 1. 剩余元素数量
        # 2. 即使选最小值也超过 remaining
        need = k - len(path)
        for i in range(start, 10 - need + 1):
            if i > remaining:
                break
            path.append(i)
            backtrack(i + 1, path, remaining - i)
            path.pop()
    
    backtrack(1, [], n)
    return result
```

## 6.5 组合的去重技巧深析

三种去重方式对比：

### 方式一：排序 + 跳过（推荐）

```python
# 适用：candidates 有重复，不可重复选
if i > start and candidates[i] == candidates[i-1]:
    continue
```

### 方式二：used 数组

```python
# 区分"树层"去重和"树枝"去重
used = [False] * len(candidates)

def backtrack(start, path):
    for i in range(start, len(candidates)):
        # used[i-1] == True：同一树枝（不去重）
        # used[i-1] == False：同一树层（去重）
        if i > 0 and candidates[i] == candidates[i-1] and not used[i-1]:
            continue
        used[i] = True
        path.append(candidates[i])
        backtrack(i + 1, path)
        path.pop()
        used[i] = False
```

### 方式三：集合去重

```python
def backtrack(start, path):
    seen = set()  # 记录本层已使用的值
    for i in range(start, len(candidates)):
        if candidates[i] in seen:
            continue
        seen.add(candidates[i])
        path.append(candidates[i])
        backtrack(i + 1, path)
        path.pop()
```

**推荐用方式一**：不需要额外数组，代码简洁，效率高。

## 6.6 组合的迭代写法

```python
from itertools import combinations

# Python 内置
list(combinations([1,2,3,4], 2))
# [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

# 组合总和（不用内置）
from itertools import combinations_with_replacement
list(combinations_with_replacement([2,3,6,7], 2))
# 每个数字可以重复选取
```

## 6.7 综合对比：子集 vs 组合 vs 排列

```
三类问题的代码差异：

子集：    for i in range(start, n):
              bt(i+1, ...)        # 下一层从 i+1 开始

组合：    for i in range(start, n):
              bt(i+1, ...)        # 同上，多了 len==k 的终止条件

可重复组合：for i in range(start, n):
              bt(i, ...)          # 下一层从 i 开始（允许重复）

排列：    for i in range(0, n):   # 从 0 开始，不是 start
              if used[i]: continue
              bt(...)
```

## 小结

| 问题 | 核心区别 | 关键代码 |
|------|---------|---------|
| 基础组合 | 固定大小 k | `start` 递增，剪枝上界 |
| 组合总和（可重复）| 递归传 `i` | `backtrack(i, ...)` |
| 组合总和（含重复）| 不可重复选 | `i > start` 去重 |
| 组合总和 III | 固定大小+目标和 | 双重约束 |

## 练习

1. 实现"因子组合"：将整数 n 分解为所有因子的乘积组合（如 12 = 2×2×3 = 2×6 = 3×4）
2. 解决"电话号码字母组合"（LeetCode 17，见第15章预习）
3. 思考：`combination_sum` 如果不排序会怎样？能否不排序而去重？

---

**上一章：** [子集](05-subsets.md) | **下一章：** [排列问题](07-permutations.md)
