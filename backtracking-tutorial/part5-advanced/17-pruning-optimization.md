# 第17章：剪枝优化专题

## 17.1 为什么剪枝是回溯的核心

回溯的本质是搜索决策树，**剪枝**是减少搜索节点的唯一手段。

```
没有剪枝的回溯 = 暴力枚举
有了剪枝的回溯 = 智能搜索

搜索树节点数对比：
  n=20 全排列：20! ≈ 2.4 × 10^18 节点（无法完成）
  n=20 子集：2^20 ≈ 100 万节点（可接受）
  加剪枝后往往只搜索理论上界的 1/100 ~ 1/1000
```

## 17.2 五大剪枝策略

### 策略一：约束剪枝（Constraint Pruning）

当当前状态已经违反约束，立即剪枝：

```python
# 示例：组合总和，当前路径和已超出目标
def combination_sum_pruned(candidates, target):
    candidates.sort()  # 排序是约束剪枝的基础
    result = []
    
    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        
        for i in range(start, len(candidates)):
            # 约束剪枝：后续元素更大，直接跳过整个分支
            if candidates[i] > remaining:
                break  # 不是 continue，是 break！
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])
            path.pop()
    
    backtrack(0, [], target)
    return result
```

### 策略二：可行性剪枝（Feasibility Pruning）

当已知当前状态无论如何扩展都不可能得到解时剪枝：

```python
# 示例：组合总和 III，提前检查是否还有足够元素
def combination_sum3_pruned(k, n):
    result = []
    
    def backtrack(start, path, remaining):
        need = k - len(path)  # 还需要几个数
        available = 9 - start + 1  # 还有几个数可选
        
        # 可行性剪枝：可选数量不足
        if available < need:
            return
        # 可行性剪枝：即使选最小的几个数也超过 remaining
        min_sum = sum(range(start, start + need))
        if min_sum > remaining:
            return
        # 可行性剪枝：即使选最大的几个数也不够 remaining
        max_sum = sum(range(10 - need, 10))
        if max_sum < remaining:
            return
        
        if len(path) == k:
            if remaining == 0:
                result.append(path[:])
            return
        
        for i in range(start, 10):
            if i > remaining:
                break
            path.append(i)
            backtrack(i + 1, path, remaining - i)
            path.pop()
    
    backtrack(1, [], n)
    return result
```

### 策略三：去重剪枝（Deduplication Pruning）

在含重复元素的搜索中，避免同层重复选择：

```python
# 同层去重的三种等价写法
def subsets_with_dup(nums):
    nums.sort()
    result = []
    
    # 写法一：i > start 判断（推荐）
    def bt1(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:
                continue  # 同层去重
            path.append(nums[i])
            bt1(i + 1, path)
            path.pop()
    
    # 写法二：seen 集合
    def bt2(start, path):
        result.append(path[:])
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue
            seen.add(nums[i])
            path.append(nums[i])
            bt2(i + 1, path)
            path.pop()
    
    bt1(0, [])
    return result
```

### 策略四：顺序剪枝（Order Pruning）

通过对候选元素排序，使得可以提前终止循环：

```python
# 背包分配问题：相同大小的桶只尝试一次
def can_partition_k_subsets(nums, k):
    total = sum(nums)
    if total % k != 0:
        return False
    target = total // k
    if max(nums) > target:
        return False
    
    nums.sort(reverse=True)  # 从大到小排序，提前发现无效分配
    buckets = [0] * k
    
    def backtrack(idx):
        if idx == len(nums):
            return all(b == target for b in buckets)
        
        seen = set()  # 顺序剪枝：相同大小的桶只试一次
        for j in range(k):
            if buckets[j] + nums[idx] > target:
                continue
            if buckets[j] in seen:
                continue  # 关键剪枝！
            seen.add(buckets[j])
            
            buckets[j] += nums[idx]
            if backtrack(idx + 1):
                return True
            buckets[j] -= nums[idx]
            
            if buckets[j] == 0:
                break  # 空桶都等价，只试一次
        
        return False
    
    return backtrack(0)
```

### 策略五：对称剪枝（Symmetry Pruning）

利用问题的对称性减少搜索量：

```python
# N 皇后：第一行只搜索左半，结果乘以 2
def total_n_queens_symmetric(n):
    cols = set(); diag1 = set(); diag2 = set()
    
    def backtrack(row):
        if row == n:
            return 1
        total = 0
        for col in range(n):
            if col in cols or (row-col) in diag1 or (row+col) in diag2:
                continue
            cols.add(col); diag1.add(row-col); diag2.add(row+col)
            total += backtrack(row + 1)
            cols.discard(col); diag1.discard(row-col); diag2.discard(row+col)
        return total
    
    # 只搜索第一行左半，结果乘以 2
    total = 0
    cols = set(); diag1 = set(); diag2 = set()
    for col in range(n // 2):
        cols.add(col); diag1.add(-col); diag2.add(col)
        total += backtrack(1)
        cols.discard(col); diag1.discard(-col); diag2.discard(col)
    
    total *= 2  # 左右对称
    
    if n % 2 == 1:  # 奇数时中间列单独处理
        mid = n // 2
        cols = {mid}; diag1 = {-mid}; diag2 = {mid}
        total += backtrack(1)
    
    return total
```

## 17.3 剪枝效果量化

```python
import time

def benchmark_pruning():
    """量化不同剪枝策略的效果"""
    
    # 测试：求 [1..20] 中和为 50 的 5 元素组合数
    nums = list(range(1, 21))
    target = 50
    k = 5
    
    # 无剪枝版本
    count_no_prune = [0]
    nodes_no_prune = [0]
    
    def bt_no_prune(start, path, remaining):
        nodes_no_prune[0] += 1
        if len(path) == k:
            if remaining == 0:
                count_no_prune[0] += 1
            return
        for i in range(start, len(nums)):
            path.append(nums[i])
            bt_no_prune(i + 1, path, remaining - nums[i])
            path.pop()
    
    # 有剪枝版本
    count_pruned = [0]
    nodes_pruned = [0]
    
    def bt_pruned(start, path, remaining):
        nodes_pruned[0] += 1
        if len(path) == k:
            if remaining == 0:
                count_pruned[0] += 1
            return
        
        need = k - len(path)
        available = len(nums) - start
        if available < need:  # 可行性剪枝
            return
        
        for i in range(start, len(nums)):
            if nums[i] > remaining:  # 约束剪枝
                break
            # 可行性剪枝：剩余数不可能凑够
            min_remaining = sum(nums[i:i+need])
            if min_remaining > remaining:
                break
            
            path.append(nums[i])
            bt_pruned(i + 1, path, remaining - nums[i])
            path.pop()
    
    t = time.time()
    bt_no_prune(0, [], target)
    t1 = time.time() - t
    
    t = time.time()
    bt_pruned(0, [], target)
    t2 = time.time() - t
    
    print(f"无剪枝：{nodes_no_prune[0]} 节点，{t1:.4f}s")
    print(f"有剪枝：{nodes_pruned[0]} 节点，{t2:.4f}s")
    print(f"节点减少：{nodes_no_prune[0]/nodes_pruned[0]:.1f}x")
    assert count_no_prune[0] == count_pruned[0]

benchmark_pruning()
```

## 17.4 剪枝选择指南

```
问题特征 → 推荐剪枝
─────────────────────────────────────────────
候选已排序 + 总和约束 → break（不是 continue）
含重复元素 → i > start 去重
分组/桶问题 → 相同大小桶只试一次
棋盘问题 → 约束集合 O(1) 检查
树/图搜索 → 访问标记
NP 问题 → 上下界剪枝
```

## 小结

五大剪枝策略按效果排序（通常）：
1. **约束剪枝**：违反约束立即停止（最常用）
2. **可行性剪枝**：无法达到目标立即停止
3. **去重剪枝**：避免同层重复分支
4. **顺序剪枝**：相同状态的等价枝只试一次
5. **对称剪枝**：利用全局对称性减半搜索

## 练习

1. 为"全排列"问题设计并测量可行性剪枝（提前判断剩余数能否配对）
2. 在"单词搜索"中实现字符频率剪枝并量化效果
3. 阅读"启发式搜索（A*）"论文，理解其与回溯剪枝的关系

---

**上一章（Part 4）：** [括号生成](../part4-string-partition/16-generate-parentheses.md) | **下一章：** [记忆化搜索](18-memoization.md)
