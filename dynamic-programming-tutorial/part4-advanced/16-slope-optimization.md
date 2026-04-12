# 第16章：斜率优化DP（CHT）

## 16.1 问题引入

某类DP的转移方程形如：

```
dp[i] = min(dp[j] + cost(i, j))，j < i
```

展开后，`cost(i, j)` 是关于 i 和 j 的某个函数。如果能把转移方程变形成：

```
dp[i] = min(b[j] + m[j] * x[i])
```

其中 `b[j]` 和 `m[j]` 只与 j 有关，`x[i]` 只与 i 有关，就可以用**斜率优化**（Convex Hull Trick，CHT）将 O(n²) 优化到 O(n)。

---

## 16.2 几何直觉

将每个 j 看作一条直线：`y = m[j] * x + b[j]`

对于固定的 i（即固定的 `x = x[i]`），问题变成：**在所有直线中，找在 x[i] 处 y 值最小的那条线**。

这等价于**下凸包**问题：维护直线的下凸包，查询 x[i] 处的最小值。

```
y
|    /  /  /
|   /  /  /
|  / /  /
| //  /
|/___________  x
     x[i]
```

---

## 16.3 经典问题：工厂生产调度

**DP方程**：

```
dp[i] = min over j<i { dp[j] + (prefix[i] - prefix[j])² }
```

其中 `prefix[i]` 是前缀和。展开：

```
dp[i] = dp[j] + prefix[i]² - 2*prefix[i]*prefix[j] + prefix[j]²
       = (dp[j] + prefix[j]²) - 2*prefix[i]*prefix[j] + prefix[i]²
```

对固定 i，令 `x = prefix[i]`，每个 j 对应一条直线：
- 斜率：`m[j] = -2 * prefix[j]`
- 截距：`b[j] = dp[j] + prefix[j]²`
- 查询：`dp[i] = min(m[j] * x + b[j]) + prefix[i]²`

---

## 16.4 CHT 的实现

### 静态CHT（斜率单调，查询单调）

若斜率 m[j] 单调，查询 x[i] 单调，可以用指针 O(n) 实现：

```python
def convex_hull_trick_linear(m_list, b_list, queries):
    """
    最小化 m[j] * x + b[j]
    假设：斜率 m[j] 递减，查询 x 递增
    """
    lines = list(zip(m_list, b_list))  # (slope, intercept)
    
    def bad(l1, l2, l3):
        # l2 被 l1 和 l3 的交点"支配"，可以删除
        # 几何意义：l2 永远不会是最小值
        m1, b1 = l1
        m2, b2 = l2
        m3, b3 = l3
        # l2 的加入使 l1∩l3 的 x 坐标 <= l1∩l2 的 x 坐标
        return (b3 - b1) * (m1 - m2) <= (b2 - b1) * (m1 - m3)
    
    # 构建下凸包
    hull = []
    for line in lines:
        while len(hull) >= 2 and bad(hull[-2], hull[-1], line):
            hull.pop()
        hull.append(line)
    
    # 用双指针查询（查询单调时）
    ptr = 0
    results = []
    for x in queries:
        while ptr + 1 < len(hull):
            m1, b1 = hull[ptr]
            m2, b2 = hull[ptr + 1]
            if m1 * x + b1 >= m2 * x + b2:
                ptr += 1
            else:
                break
        m, b = hull[ptr]
        results.append(m * x + b)
    
    return results
```

### 动态CHT（查询不单调，用二分查找）

```python
import bisect

class CHT:
    """
    最小化 m*x + b 的动态凸包
    """
    def __init__(self):
        self.hull = []  # list of (slope, intercept)
    
    def _bad(self, l1, l2, l3):
        m1, b1 = l1; m2, b2 = l2; m3, b3 = l3
        return (b3 - b1) * (m1 - m2) <= (b2 - b1) * (m1 - m3)
    
    def add_line(self, m, b):
        line = (m, b)
        while len(self.hull) >= 2 and self._bad(self.hull[-2], self.hull[-1], line):
            self.hull.pop()
        self.hull.append(line)
    
    def query(self, x):
        # 二分查找最优直线
        lo, hi = 0, len(self.hull) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            m1, b1 = self.hull[mid]
            m2, b2 = self.hull[mid + 1]
            if m1 * x + b1 <= m2 * x + b2:
                hi = mid
            else:
                lo = mid + 1
        m, b = self.hull[lo]
        return m * x + b
```

---

## 16.5 实战：任务调度（NOI 2007 货币兑换）

**DP方程**（简化版）：

```python
def task_schedule_dp(a, b, t):
    """
    dp[i] = min over j<=i { dp[j] + (a[i] - a[j])^2 + t }
    a 为升序排列的时间点
    """
    n = len(a)
    dp = [float('inf')] * n
    dp[0] = 0
    
    cht = CHT()
    # j=0 对应直线：slope = -2*a[0], intercept = dp[0] + a[0]^2
    cht.add_line(-2 * a[0], dp[0] + a[0] ** 2)
    
    for i in range(1, n):
        # dp[i] = query(a[i]) + a[i]^2 + t
        dp[i] = cht.query(a[i]) + a[i] ** 2 + t
        # 将 i 加入 CHT：slope = -2*a[i], intercept = dp[i] + a[i]^2
        cht.add_line(-2 * a[i], dp[i] + a[i] ** 2)
    
    return dp[n-1]
```

---

## 16.6 李超树（Li Chao Tree）——更通用的解法

当斜率无序时，CHT 需要用 Treap 等数据结构（较复杂）。**李超树**是另一种解法，基于线段树，O(n log V) 处理动态直线查询。

```python
class LiChaoTree:
    """
    支持动态加入直线，查询最小值
    """
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.line = None  # (slope, intercept)
        self.left = self.right = None
    
    def eval(self, x):
        if self.line is None:
            return float('inf')
        m, b = self.line
        return m * x + b
    
    def add_line(self, new_line, lo=None, hi=None):
        if lo is None: lo, hi = self.lo, self.hi
        mid = (lo + hi) // 2
        
        if self.line is None:
            self.line = new_line
            return
        
        m1, b1 = self.line
        m2, b2 = new_line
        
        left_better  = m2 * lo  + b2 < m1 * lo  + b1
        mid_better   = m2 * mid + b2 < m1 * mid + b1
        
        if mid_better:
            self.line, new_line = new_line, self.line
            m1, b1 = self.line
            m2, b2 = new_line
        
        if lo == hi:
            return
        
        if left_better != mid_better:
            if self.left is None:
                self.left = LiChaoTree(lo, mid)
            self.left.add_line((m2, b2), lo, mid)
        else:
            if self.right is None:
                self.right = LiChaoTree(mid + 1, hi)
            self.right.add_line((m2, b2), mid + 1, hi)
    
    def query(self, x, lo=None, hi=None):
        if lo is None: lo, hi = self.lo, self.hi
        res = self.eval(x)
        if lo == hi:
            return res
        mid = (lo + hi) // 2
        if x <= mid:
            if self.left:
                res = min(res, self.left.query(x, lo, mid))
        else:
            if self.right:
                res = min(res, self.right.query(x, mid + 1, hi))
        return res
```

---

## 16.7 本章小结

**斜率优化适用条件**：

DP方程可以变形为 `dp[i] = min_j(m[j] * x[i] + b[j])`，其中：
- `m[j]`, `b[j]` 只与 j 有关
- `x[i]` 只与 i 有关

**复杂度对比**：

| 方法 | 时间 | 条件 |
|------|------|------|
| 朴素 | O(n²) | 无 |
| CHT（单调）| O(n) | 斜率单调 + 查询单调 |
| CHT（半单调）| O(n log n) | 斜率单调，查询任意 |
| 李超树 | O(n log V) | 斜率任意 |

---

## 练习题

- 任何含 `dp[i] = min(dp[j] + f(i,j))` 形式的DP，尝试展开 f(i,j) 并识别是否可以斜率优化
- [CF - Fence Repair](https://codeforces.com/contest/311/problem/E)（斜率优化入门）
- 洛谷 P3195 玩具装箱（斜率优化经典）
