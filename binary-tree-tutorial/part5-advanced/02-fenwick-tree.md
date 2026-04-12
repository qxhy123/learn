# 第16章：树状数组（Fenwick Tree / BIT）

## 16.1 什么是树状数组

树状数组（Binary Indexed Tree，BIT）由 Peter Fenwick 在1994年提出，是解决**前缀和查询+单点更新**的极简数据结构。

**对比线段树**：

| | 树状数组 | 线段树 |
|-|---------|--------|
| 代码量 | ~10 行 | ~40 行 |
| 常数因子 | 小 | 大 |
| 功能 | 前缀和/区间和 | 更通用 |
| 空间 | O(n) | O(n) |
| 时间 | O(log n) | O(log n) |

若只需要前缀和查询，**优先使用树状数组**。

## 16.2 lowbit 操作

树状数组的核心是 **lowbit(i)** = i 的二进制最低有效位：

```python
def lowbit(i):
    return i & (-i)

# lowbit(6) = lowbit(0b110) = 0b010 = 2
# lowbit(8) = lowbit(0b1000) = 0b1000 = 8
# lowbit(12) = lowbit(0b1100) = 0b100 = 4
```

**含义**：`tree[i]` 负责维护区间 `[i - lowbit(i) + 1, i]` 的前缀和。

```
下标:  1    2    3    4    5    6    7    8
       |    |    |    |    |    |    |    |
tree[1] 管 [1,1]
tree[2] 管 [1,2]
tree[3] 管 [3,3]
tree[4] 管 [1,4]
tree[5] 管 [5,5]
tree[6] 管 [5,6]
tree[7] 管 [7,7]
tree[8] 管 [1,8]
```

## 16.3 基本实现

```python
class BIT:
    """树状数组（1-indexed，支持单点更新+前缀和查询）"""
    
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, i, delta):
        """将 a[i] 加 delta，O(log n)"""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # 向上更新
    
    def query(self, i):
        """查询前缀和 a[1] + a[2] + ... + a[i]，O(log n)"""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)  # 向上累加
        return s
    
    def range_query(self, l, r):
        """区间和查询 a[l] + ... + a[r]"""
        return self.query(r) - self.query(l - 1)
    
    @classmethod
    def from_array(cls, nums):
        """O(n) 建树（比逐个 update 更快）"""
        n = len(nums)
        bit = cls(n)
        bit.tree[1:n+1] = nums[:]  # 复制数组
        for i in range(1, n + 1):
            j = i + (i & -i)  # 父节点
            if j <= n:
                bit.tree[j] += bit.tree[i]
        return bit

# 测试
nums = [3, 2, -1, 6, 5, 4, -3, 3, 7, 2, 3]
bit = BIT(len(nums))
for i, x in enumerate(nums, 1):
    bit.update(i, x)

print(bit.query(4))          # 3+2-1+6 = 10
print(bit.range_query(2, 7)) # 2-1+6+5+4-3 = 13
bit.update(3, 5)             # a[3] += 5，a[3] 变为 4
print(bit.query(4))          # 3+2+4+6 = 15
```

## 16.4 区间更新 + 单点查询

利用**差分数组**技巧，将区间更新转化为两次单点更新：

```python
class BIT_RangeUpdate:
    """支持区间更新+单点查询"""
    
    def __init__(self, n):
        self.n = n
        self.diff_bit = BIT(n)  # 维护差分数组的前缀和
    
    def range_update(self, l, r, delta):
        """区间 [l, r] 所有元素加 delta，O(log n)"""
        self.diff_bit.update(l, delta)
        self.diff_bit.update(r + 1, -delta)
    
    def point_query(self, i):
        """查询 a[i] 的当前值，O(log n)"""
        return self.diff_bit.query(i)  # 差分前缀和 = 原值

# 测试
rbu = BIT_RangeUpdate(5)
rbu.range_update(1, 3, 5)   # [5,5,5,0,0]
rbu.range_update(2, 5, 3)   # [5,8,8,3,3]
print(rbu.point_query(1))   # 5
print(rbu.point_query(2))   # 8
print(rbu.point_query(4))   # 3
```

## 16.5 区间更新 + 区间查询

需要两个 BIT（利用数学推导）：

```python
class BIT_Full:
    """支持区间更新 + 区间查询"""
    
    def __init__(self, n):
        self.n = n
        self.b1 = BIT(n)  # 维护 d[i]
        self.b2 = BIT(n)  # 维护 d[i] * i
    
    def range_update(self, l, r, delta):
        """区间 [l, r] 所有元素加 delta"""
        self.b1.update(l, delta)
        self.b1.update(r + 1, -delta)
        self.b2.update(l, delta * (l - 1))
        self.b2.update(r + 1, -delta * r)
    
    def prefix_sum(self, i):
        """前缀和 a[1]+...+a[i]"""
        return self.b1.query(i) * i - self.b2.query(i)
    
    def range_sum(self, l, r):
        """区间和"""
        return self.prefix_sum(r) - self.prefix_sum(l - 1)

# 测试
bf = BIT_Full(5)
bf.range_update(1, 5, 1)    # [1,1,1,1,1]
bf.range_update(2, 4, 2)    # [1,3,3,3,1]
print(bf.range_sum(1, 5))   # 11
print(bf.range_sum(2, 4))   # 9
```

## 16.6 经典应用

### 统计逆序对

```python
def count_inversions_bit(nums):
    """用 BIT 统计逆序对，O(n log n)"""
    # 离散化
    sorted_vals = sorted(set(nums))
    rank = {v: i+1 for i, v in enumerate(sorted_vals)}
    
    bit = BIT(len(sorted_vals))
    result = 0
    
    for x in reversed(nums):
        r = rank[x]
        result += bit.query(r - 1)  # 比 x 小的已处理元素数量
        bit.update(r, 1)
    
    return result

print(count_inversions_bit([5, 2, 6, 1]))  # 4
```

### 二维树状数组

```python
class BIT2D:
    """二维树状数组：矩阵单点更新 + 子矩阵求和"""
    
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.tree = [[0] * (n + 1) for _ in range(m + 1)]
    
    def update(self, x, y, delta):
        i = x
        while i <= self.m:
            j = y
            while j <= self.n:
                self.tree[i][j] += delta
                j += j & (-j)
            i += i & (-i)
    
    def query(self, x, y):
        s = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                s += self.tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return s
    
    def range_query(self, x1, y1, x2, y2):
        return (self.query(x2, y2) - self.query(x1-1, y2)
                - self.query(x2, y1-1) + self.query(x1-1, y1-1))
```

## 16.7 选择线段树还是树状数组

```
决策流程：
├── 只需前缀和/单点更新 → BIT（代码简洁，常数小）
├── 区间更新+区间查询（同一聚合函数）→ BIT（差分技巧）
├── 区间最大/最小值查询 → 线段树（BIT 不支持）
├── 多种操作组合/区间赋值 → 线段树（懒惰标记）
└── 区间查询 + 一次性构建（不修改）→ 稀疏表（O(1) 查询）
```

## 小结

- BIT 核心：lowbit = `i & (-i)`，向上更新，向下查询
- 10 行代码实现 O(log n) 前缀和
- 差分 BIT 支持区间更新
- 双 BIT 支持区间更新+区间查询

## 练习

1. 实现"数组中区间内小于 k 的元素个数"（BIT + 离散化）
2. 用 BIT 实现动态中位数
3. 解决二维问题：矩阵中以某点为右下角的矩形最大和

---

**上一章：** [线段树](01-segment-tree.md) | **下一章：** [字典树（Trie）](03-trie.md)
