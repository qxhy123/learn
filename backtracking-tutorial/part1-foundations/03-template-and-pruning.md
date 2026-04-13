# 第3章：回溯通用模板与剪枝技术

## 3.1 万能回溯模板

```python
def backtrack(参数):
    if 终止条件:
        存放结果
        return
    
    for 选择 in 本层选择列表:
        if 不合法:            # 剪枝
            continue
        
        处理节点（做选择）
        backtrack(下一层参数)
        回溯（撤销选择）
```

这个模板涵盖了所有回溯问题的骨架，变化的只是：
1. 参数列表
2. 终止条件
3. 选择列表的范围
4. 剪枝条件

## 3.2 模板的三种变体

### 变体一：收集路径型（求所有解）

```python
def solve(nums):
    result = []
    
    def backtrack(path, start):
        # 满足条件就收集（可在任意层收集）
        if 满足条件:
            result.append(path[:])  # 必须拷贝！
            # return  # 如果找到即停止，加 return
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(path, i + 1)
            path.pop()
    
    backtrack([], 0)
    return result
```

> **关键点**：收集时必须用 `path[:]` 或 `list(path)` 拷贝，否则后续的修改会影响已收集的结果。

### 变体二：判断是否有解型

```python
def exists(board, word):
    def backtrack(pos, index):
        if index == len(word):
            return True  # 找到解，提前退出
        
        if 越界 or 不匹配:
            return False  # 剪枝
        
        board[pos] = '#'  # 标记已访问
        
        for next_pos in 相邻位置:
            if backtrack(next_pos, index + 1):
                board[pos] = word[index]  # 恢复（可选，找到解后不需要）
                return True
        
        board[pos] = word[index]  # 恢复
        return False
    
    for start in 所有起点:
        if backtrack(start, 0):
            return True
    return False
```

### 变体三：计数型（求解的数量）

```python
def count_solutions(n):
    count = [0]
    
    def backtrack(state):
        if 满足终止条件:
            count[0] += 1
            return
        
        for 选择 in 选择列表:
            if 合法(选择):
                做选择
                backtrack(新状态)
                撤销选择
    
    backtrack(初始状态)
    return count[0]
```

## 3.3 五种剪枝策略

### 策略一：约束剪枝

根据问题约束直接排除非法状态：

```python
# N 皇后：检查同行、同列、同对角线
def is_valid(board, row, col):
    n = len(board)
    # 同列
    for i in range(row):
        if board[i][col] == 'Q':
            return False
    # 左上对角线
    i, j = row - 1, col - 1
    while i >= 0 and j >= 0:
        if board[i][j] == 'Q':
            return False
        i -= 1; j -= 1
    # 右上对角线
    i, j = row - 1, col + 1
    while i >= 0 and j < n:
        if board[i][j] == 'Q':
            return False
        i -= 1; j += 1
    return True
```

### 策略二：边界剪枝

当剩余元素不足以完成目标时提前终止：

```python
def combine(n, k):
    result = []
    
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        
        # 剪枝：n - i + 1 是从 i 开始剩余的元素数
        # 需要 k - len(path) 个元素
        # 若剩余 < 需要，无法凑满
        for i in range(start, n - (k - len(path)) + 2):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(1, [])
    return result
```

### 策略三：排序 + 超限剪枝

对候选集排序后，一旦超过目标就 break（后面更大，都不符合）：

```python
def combination_sum(candidates, target):
    candidates.sort()  # 必须先排序
    result = []
    
    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break  # ← break 而非 continue！后面的更大，全部剪掉
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])
            path.pop()
    
    backtrack(0, [], target)
    return result
```

### 策略四：去重剪枝

含重复元素时，同层不重复选择相同的值：

```python
def subsets_with_dup(nums):
    nums.sort()  # 排序，将相同元素相邻
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        
        for i in range(start, len(nums)):
            # 同层去重：跳过与上一个相同的元素
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

# 关键：i > start（不是 i > 0）
# i > 0 会错误地在不同层也去重
# i > start 只在同一层去重
```

> **理解 `i > start` vs `i > 0`**：
> - `i > 0`：防止跟全局第一个重复，会误删不同层的合法情况
> - `i > start`：只防止跟当前层第一个重复，正确

### 策略五：记忆化剪枝

对已计算过的状态缓存结果（适合有重叠子问题的情况）：

```python
from functools import lru_cache

def word_break(s, word_dict):
    word_set = set(word_dict)
    
    @lru_cache(maxsize=None)
    def backtrack(start):
        if start == len(s):
            return True
        for end in range(start + 1, len(s) + 1):
            if s[start:end] in word_set:
                if backtrack(end):
                    return True
        return False
    
    return backtrack(0)
```

## 3.4 模板应用：一题三解

**问题**：给定 `n` 和 `k`，从 `1..n` 中选 `k` 个数的所有组合。

### 解法一：无剪枝

```python
def combine_v1(n, k):
    result = []
    def bt(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, n + 1):
            path.append(i)
            bt(i + 1, path)
            path.pop()
    bt(1, [])
    return result
```

### 解法二：边界剪枝

```python
def combine_v2(n, k):
    result = []
    def bt(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        # 最多从 n-(k-len(path))+1 开始循环
        for i in range(start, n - (k - len(path)) + 2):
            path.append(i)
            bt(i + 1, path)
            path.pop()
    bt(1, [])
    return result
```

### 解法三：递推剪枝（最优）

```python
def combine_v3(n, k):
    result = []
    def bt(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        need = k - len(path)
        # start 到 n-need+1（含），才有足够元素
        for i in range(start, n - need + 2):
            path.append(i)
            bt(i + 1, path)
            path.pop()
    bt(1, [])
    return result
```

对比三解的搜索节点数（n=5, k=3）：
- v1：15 个节点
- v2/v3：9 个节点（剪掉约 40%）

## 3.5 常见错误及修复

### 错误1：忘记拷贝路径

```python
# ❌ 错误：添加了路径的引用，后续修改会影响结果
result.append(path)

# ✓ 正确：添加路径的拷贝
result.append(path[:])
result.append(list(path))
```

### 错误2：忘记撤销选择

```python
# ❌ 错误：没有撤销，path 越来越长
for i in range(start, n):
    path.append(nums[i])
    backtrack(i + 1, path)
    # 忘了 path.pop()

# ✓ 正确：成对操作
for i in range(start, n):
    path.append(nums[i])     # 做选择
    backtrack(i + 1, path)
    path.pop()               # 撤销选择
```

### 错误3：修改了不该修改的全局状态

```python
# ❌ 错误：直接修改原数组
def backtrack(nums, start):
    nums.remove(some_val)    # 修改了原数组
    backtrack(nums, start)
    nums.append(some_val)    # 可能插入位置不对

# ✓ 正确：用 used 数组或临时变量标记
used = [False] * n
```

## 小结

| 剪枝类型 | 触发条件 | 典型场景 |
|---------|---------|---------|
| 约束剪枝 | 违反问题约束 | N皇后、数独 |
| 边界剪枝 | 剩余元素不足 | 组合问题 |
| 排序+超限 | 当前值>目标 | 组合总和 |
| 去重剪枝 | 同层同值 | 含重复元素的子集/排列 |
| 记忆化 | 重复子问题 | 单词拆分 |

## 练习

1. 实现"含重复元素的全排列"（需要哪种剪枝？）
2. 对比有无边界剪枝时 `combine(10, 5)` 的节点访问数
3. 找出以下代码的 bug 并修复：
   ```python
   def subsets(nums):
       result = [[]]
       def bt(start, path):
           for i in range(start, len(nums)):
               path.append(nums[i])
               result.append(path)  # bug!
               bt(i+1, path)
       bt(0, [])
       return result
   ```

---

**上一章：** [决策树](02-decision-tree.md) | **下一章：** [复杂度分析](04-complexity-analysis.md)
