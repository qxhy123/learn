# 第13章：红黑树

## 13.1 红黑树的五条性质

红黑树是每个节点附带**颜色**（红/黑）的 BST，通过颜色约束保证近似平衡。

**五条性质**（缺一不可）：

1. 每个节点是**红色**或**黑色**
2. **根节点**是黑色
3. 每个**叶节点**（NIL 哨兵节点）是黑色
4. **红节点的子节点**必须是黑色（不能有连续红节点）
5. 从任意节点到其所有后代叶节点，经过的**黑色节点数相同**（黑高相等）

```
合法红黑树：
         8(B)
        /    \
      4(R)   12(B)
      / \    /   \
    2(B) 6(B) 10(R) 14(R)

（叶节点 NIL 均为黑色，省略）
```

## 13.2 为什么这五条能保证平衡

**黑高**（black-height）：从节点 v 到叶节点路径上的黑色节点数（不含 v 本身）。

由性质4和性质5推导：
- 最短路径：全黑节点，长度 = 黑高 h
- 最长路径：红黑交替，长度 = 2h

因此：**最长路径 ≤ 2 × 最短路径**

对于 n 个节点的红黑树：树高 $h \leq 2\log_2(n+1)$，即 **O(log n)**。

## 13.3 节点结构与 NIL 哨兵

```python
class RBColor:
    RED = 0
    BLACK = 1

class RBNode:
    def __init__(self, key, color=RBColor.RED):
        self.key = key
        self.color = color
        self.left = None
        self.right = None
        self.parent = None  # 红黑树通常需要父指针

# NIL 哨兵节点（全局唯一，所有叶节点指向它）
NIL = RBNode(None, RBColor.BLACK)
NIL.left = NIL
NIL.right = NIL
NIL.parent = NIL
```

## 13.4 插入操作

插入步骤：
1. 像普通 BST 一样插入，**新节点染红**
2. 修复可能违反的性质4（连续红节点）

```python
class RedBlackTree:
    def __init__(self):
        self.NIL = RBNode(None, RBColor.BLACK)
        self.root = self.NIL
    
    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent == self.NIL:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
    
    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.NIL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent == self.NIL:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
    
    def insert(self, key):
        z = RBNode(key, RBColor.RED)
        z.left = z.right = z.parent = self.NIL
        
        # BST 插入
        y = self.NIL
        x = self.root
        while x != self.NIL:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        
        z.parent = y
        if y == self.NIL:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        
        # 修复红黑性质
        self._insert_fixup(z)
    
    def _insert_fixup(self, z):
        """修复插入后的红黑性质违反"""
        while z.parent.color == RBColor.RED:
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right  # 叔父节点
                
                if y.color == RBColor.RED:
                    # Case 1: 叔父是红色 → 变色
                    z.parent.color = RBColor.BLACK
                    y.color = RBColor.BLACK
                    z.parent.parent.color = RBColor.RED
                    z = z.parent.parent  # 上移
                else:
                    if z == z.parent.right:
                        # Case 2: z 是右子节点 → 左旋转为 Case 3
                        z = z.parent
                        self._rotate_left(z)
                    
                    # Case 3: z 是左子节点 → 变色 + 右旋
                    z.parent.color = RBColor.BLACK
                    z.parent.parent.color = RBColor.RED
                    self._rotate_right(z.parent.parent)
            else:
                # 对称情况（父节点是祖父的右子节点）
                y = z.parent.parent.left
                
                if y.color == RBColor.RED:
                    z.parent.color = RBColor.BLACK
                    y.color = RBColor.BLACK
                    z.parent.parent.color = RBColor.RED
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self._rotate_right(z)
                    
                    z.parent.color = RBColor.BLACK
                    z.parent.parent.color = RBColor.RED
                    self._rotate_left(z.parent.parent)
        
        self.root.color = RBColor.BLACK  # 性质2：根为黑色
```

## 13.5 插入修复的三种情况

```
设 z 是新插入的红节点，p = z.parent（红），g = p.parent（黑）

Case 1: 叔父节点 u 是红色
  g(B)              g(R) ← 继续向上修复
  /  \     →        /  \
p(R)  u(R)         p(B) u(B)
z(R)               z(R)

Case 2: u 是黑色，z 是右子节点
  g(B)               g(B)
  /       →           /
p(R)               z(R)      → 转为 Case 3
  \                /
  z(R)          p(R)

Case 3: u 是黑色，z 是左子节点
  g(B)               p(B)
  /       →          /  \
p(R)               z(R)  g(R)
z(R)
```

## 13.6 删除操作（简要）

红黑树删除是所有平衡 BST 中最复杂的操作，涉及 6 种修复情况。这里给出概念框架：

```python
def delete(self, key):
    """
    删除步骤：
    1. 按 BST 找到节点 z
    2. 确定实际被删除的节点 y（z 或 z 的后继）
    3. 用 y 的子节点 x 替换 y
    4. 若 y 是黑色，调用 delete_fixup(x) 修复
    """
    # 完整实现约 80 行，涉及 6 种 Case
    # 此处省略，建议参考 CLRS 算法导论第13章
    pass
```

删除修复的关键：若被删节点是黑色，需要给替代节点补充"一重黑"，再通过旋转/变色消除"双重黑"。

## 13.7 红黑树 vs AVL 树（实际性能）

```
操作次数（n=10^6）：

查找：AVL 略优（更严格平衡，树更矮）
插入：相近
删除：红黑树优（删除旋转次数有上界 3，AVL 最多 O(log n)）

实际应用：
- Linux 内核（进程调度、内存管理）→ 红黑树
- C++ STL map/set → 红黑树
- Java TreeMap/TreeSet → 红黑树
- 数据库索引（读多写少）→ B树/B+树
```

## 13.8 Python 中使用平衡 BST

```python
# Python 没有内置红黑树，使用 sortedcontainers
from sortedcontainers import SortedList, SortedDict, SortedSet

# SortedList：有序列表，O(log n) 插入删除
sl = SortedList([3, 1, 4, 1, 5, 9, 2, 6])
sl.add(7)
sl.remove(1)  # 删除第一个 1
print(sl.count(1))  # 计数
print(sl[3])        # O(log n) 按索引访问

# SortedDict：有序字典
sd = SortedDict({'banana': 2, 'apple': 5, 'cherry': 1})
for key, val in sd.items():
    print(key, val)  # 按键有序输出

# 实际使用 SortedList 的场景：滑动窗口中位数
class MedianFinder:
    def __init__(self):
        self.data = SortedList()
    
    def add_num(self, num):
        self.data.add(num)
    
    def find_median(self):
        n = len(self.data)
        if n % 2 == 1:
            return float(self.data[n // 2])
        return (self.data[n // 2 - 1] + self.data[n // 2]) / 2
```

## 小结

| 性质 | 作用 |
|------|------|
| 根为黑色 | 防止根为红导致问题 |
| 红节点子为黑 | 防止连续红节点 |
| 黑高相等 | 保证近似平衡 |
| 插入修复 3 种情况 | 最多 2 次旋转 |
| 删除修复 6 种情况 | 最多 3 次旋转 |

## 练习

1. 手动验证上图红黑树满足所有 5 条性质
2. 在红黑树中插入 `[1,2,3,4,5]`，画出每步的树结构
3. 用 SortedList 实现"数据流中第 K 大的元素"（LeetCode 703）

---

**上一章：** [AVL 树](01-avl.md) | **下一章：** [B 树与 B+ 树](03-btree.md)
