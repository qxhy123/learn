# 第15章：线段树（Segment Tree）

## 15.1 问题引入

**区间查询问题**：给定数组，支持以下操作：
- 查询区间 [l, r] 的最大值/最小值/求和
- 单点修改或区间修改

| 方法 | 单点更新 | 区间查询 |
|------|---------|---------|
| 暴力 | O(1) | O(n) |
| 前缀和 | O(n) | O(1) |
| **线段树** | **O(log n)** | **O(log n)** |

线段树是区间查询+修改问题的标准解法。

## 15.2 线段树结构

线段树是一棵**完全二叉树**，每个节点维护一个区间的聚合信息：

```
数组 a = [1, 3, 5, 7, 9, 11]（下标 0-5）

线段树（区间和）：
             [0,5]=36
            /         \
       [0,2]=9        [3,5]=27
       /    \          /     \
   [0,1]=4 [2,2]=5 [3,4]=16 [5,5]=11
   /    \            /    \
[0,0]=1 [1,1]=3  [3,3]=7 [4,4]=9
```

## 15.3 基于数组的线段树（推荐）

```python
class SegmentTree:
    """
    基于数组的线段树（1-indexed）
    数组大小为 4n，足以覆盖所有节点
    """
    
    def __init__(self, nums, func=sum):
        """
        func: 聚合函数，支持 sum, min, max 等
        """
        self.n = len(nums)
        self.func = func
        # 默认值：sum→0, min→inf, max→-inf
        if func == sum:
            self.default = 0
        elif func == min:
            self.default = float('inf')
        else:
            self.default = float('-inf')
        
        self.tree = [self.default] * (4 * self.n)
        self._build(nums, 1, 0, self.n - 1)
    
    def _build(self, nums, node, start, end):
        """递归建树，O(n)"""
        if start == end:
            self.tree[node] = nums[start]
        else:
            mid = (start + end) // 2
            self._build(nums, 2 * node, start, mid)
            self._build(nums, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.func([self.tree[2 * node],
                                         self.tree[2 * node + 1]])
    
    def update(self, idx, val, node=1, start=0, end=None):
        """单点更新 a[idx] = val，O(log n)"""
        if end is None:
            end = self.n - 1
        
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self.update(idx, val, 2 * node, start, mid)
            else:
                self.update(idx, val, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.func([self.tree[2 * node],
                                         self.tree[2 * node + 1]])
    
    def query(self, l, r, node=1, start=0, end=None):
        """区间查询 [l, r]，O(log n)"""
        if end is None:
            end = self.n - 1
        
        if r < start or end < l:
            return self.default  # 无交集
        
        if l <= start and end <= r:
            return self.tree[node]  # 完全覆盖
        
        mid = (start + end) // 2
        left_val = self.query(l, r, 2 * node, start, mid)
        right_val = self.query(l, r, 2 * node + 1, mid + 1, end)
        return self.func([left_val, right_val])

# 测试
nums = [1, 3, 5, 7, 9, 11]

# 区间求和
st_sum = SegmentTree(nums, sum)
print(st_sum.query(0, 2))   # 1+3+5 = 9
print(st_sum.query(1, 4))   # 3+5+7+9 = 24
st_sum.update(1, 10)         # a[1] = 10
print(st_sum.query(0, 2))   # 1+10+5 = 16

# 区间最小值
st_min = SegmentTree(nums, min)
print(st_min.query(1, 4))   # min(3,5,7,9) = 3
```

## 15.4 懒惰标记（区间修改）

当需要**区间修改**（如区间 [l,r] 所有元素加 k），使用**懒惰标记**避免逐节点更新：

```python
class LazySegmentTree:
    """支持区间加法修改和区间求和查询"""
    
    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)  # 懒惰标记
        self._build(nums, 1, 0, self.n - 1)
    
    def _build(self, nums, node, start, end):
        if start == end:
            self.tree[node] = nums[start]
        else:
            mid = (start + end) // 2
            self._build(nums, 2 * node, start, mid)
            self._build(nums, 2 * node + 1, mid + 1, end)
            self.tree[node] = self.tree[2*node] + self.tree[2*node+1]
    
    def _push_down(self, node, start, end):
        """下推懒惰标记到子节点"""
        if self.lazy[node] != 0:
            mid = (start + end) // 2
            left_len = mid - start + 1
            right_len = end - mid
            
            self.tree[2*node] += self.lazy[node] * left_len
            self.tree[2*node+1] += self.lazy[node] * right_len
            self.lazy[2*node] += self.lazy[node]
            self.lazy[2*node+1] += self.lazy[node]
            self.lazy[node] = 0  # 清除当前标记
    
    def range_update(self, l, r, delta, node=1, start=0, end=None):
        """区间 [l,r] 所有元素加 delta，O(log n)"""
        if end is None:
            end = self.n - 1
        
        if r < start or end < l:
            return
        
        if l <= start and end <= r:
            # 完全覆盖：更新当前节点，打上懒惰标记
            self.tree[node] += delta * (end - start + 1)
            self.lazy[node] += delta
            return
        
        self._push_down(node, start, end)  # 先下推
        mid = (start + end) // 2
        self.range_update(l, r, delta, 2*node, start, mid)
        self.range_update(l, r, delta, 2*node+1, mid+1, end)
        self.tree[node] = self.tree[2*node] + self.tree[2*node+1]
    
    def range_query(self, l, r, node=1, start=0, end=None):
        """区间求和查询，O(log n)"""
        if end is None:
            end = self.n - 1
        
        if r < start or end < l:
            return 0
        
        if l <= start and end <= r:
            return self.tree[node]
        
        self._push_down(node, start, end)
        mid = (start + end) // 2
        return self.range_query(l, r, 2*node, start, mid) + \
               self.range_query(l, r, 2*node+1, mid+1, end)

# 测试
lst = LazySegmentTree([1, 2, 3, 4, 5])
print(lst.range_query(0, 4))  # 15
lst.range_update(1, 3, 10)    # [1,12,13,14,5]
print(lst.range_query(0, 4))  # 45
print(lst.range_query(1, 3))  # 39
```

## 15.5 经典应用

### 统计逆序对（归并排序/线段树）

```python
def count_inversions(nums):
    """
    统计逆序对数量：i < j 且 a[i] > a[j]
    用线段树（值域线段树）：从右到左扫描，查询比当前值小的已扫描数量
    """
    max_val = max(nums)
    bit = [0] * (max_val + 2)  # 用 BIT 更简单（见下一章）
    
    def update(i):
        while i <= max_val:
            bit[i] += 1
            i += i & (-i)
    
    def query(i):
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & (-i)
        return s
    
    result = 0
    for x in reversed(nums):
        result += query(x - 1)  # 比 x 小的数的个数
        update(x)
    
    return result
```

## 15.6 线段树合并（进阶）

将两棵权值线段树合并，用于树上问题：

```python
def merge_trees(node1, node2):
    """合并两棵线段树（动态开点）"""
    if node1 is None:
        return node2
    if node2 is None:
        return node1
    
    node1.val += node2.val  # 合并操作
    node1.left = merge_trees(node1.left, node2.left)
    node1.right = merge_trees(node1.right, node2.right)
    return node1
```

## 15.7 复杂度总结

| 操作 | 时间 | 空间 |
|------|------|------|
| 建树 | O(n) | O(n) |
| 单点更新 | O(log n) | - |
| 区间更新（懒惰）| O(log n) | - |
| 区间查询 | O(log n) | - |

## 小结

- 线段树解决区间查询+修改，均为 O(log n)
- 懒惰标记支持区间修改
- 数组实现（4n 空间）比链式简洁高效
- 可支持和、最大值、最小值、GCD 等聚合操作

## 练习

1. 实现区间最大值线段树，支持区间赋值（懒惰标记）
2. 用线段树解决"数组中区间内第一个大于 x 的数"
3. 实现二维线段树（矩阵区间求和）

---

**上一章（Part 4）：** [B 树](../part4-balanced-trees/03-btree.md) | **下一章：** [树状数组（BIT）](02-fenwick-tree.md)
