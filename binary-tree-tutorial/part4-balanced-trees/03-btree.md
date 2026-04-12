# 第14章：B 树与 B+ 树

## 14.1 从二叉树到多叉树

红黑树和 AVL 树在内存中表现优秀，但面对**磁盘存储**时有致命缺陷：

> 每次访问节点 = 一次磁盘 I/O（极慢！）
> 树高 O(log₂ n) → 对于 n=10^9，需要约 30 次 I/O

B 树解决方案：**增加每个节点的键数量**（扇出更大），使树更"矮胖"：

```
二叉树（高度30）:          B树（阶=1000，高度3）:
    ○                      [k1, k2, ..., k999]
   / \                    /    |    ...    \
  ○   ○                  ○    ○          ○
 ...                    [千个键]  [千个键]
```

高度从 O(log₂ n) 降至 O(log_t n)，t 是最小分支数。

## 14.2 B 树定义（阶为 m）

**m 阶 B 树**满足：

1. 每个节点最多有 **m 个子节点**
2. 非根内部节点至少有 **⌈m/2⌉** 个子节点
3. 根节点（非叶）至少有 **2 个子节点**
4. 所有叶节点在**同一层**
5. 有 k 个子节点的内部节点有 **k-1 个键**

```
3阶 B 树（每节点最多2个键，最多3个子节点）：
              [15, 25]
            /    |    \
       [5,10] [17,20] [30,35]
```

## 14.3 B 树的 Python 实现

```python
class BTreeNode:
    def __init__(self, leaf=True):
        self.keys = []       # 键列表
        self.children = []   # 子节点列表（非叶节点）
        self.leaf = leaf     # 是否为叶节点

class BTree:
    def __init__(self, t):
        """
        t: 最小度数（minimum degree）
        每个非根节点至少 t-1 个键，最多 2t-1 个键
        """
        self.t = t
        self.root = BTreeNode(leaf=True)
    
    def search(self, node, key):
        """在以 node 为根的子树中搜索 key"""
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        
        if i < len(node.keys) and key == node.keys[i]:
            return (node, i)  # 找到
        
        if node.leaf:
            return None  # 未找到
        
        return self.search(node.children[i], key)
    
    def insert(self, key):
        root = self.root
        
        if len(root.keys) == 2 * self.t - 1:
            # 根节点已满，需要分裂
            new_root = BTreeNode(leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        
        self._insert_non_full(self.root, key)
    
    def _split_child(self, parent, i):
        """分裂 parent.children[i]（已满节点）"""
        t = self.t
        full_child = parent.children[i]
        new_child = BTreeNode(leaf=full_child.leaf)
        
        # 中间键提升到父节点
        mid_key = full_child.keys[t - 1]
        parent.keys.insert(i, mid_key)
        parent.children.insert(i + 1, new_child)
        
        # 分配键
        new_child.keys = full_child.keys[t:]
        full_child.keys = full_child.keys[:t - 1]
        
        # 分配子节点（非叶）
        if not full_child.leaf:
            new_child.children = full_child.children[t:]
            full_child.children = full_child.children[:t]
    
    def _insert_non_full(self, node, key):
        """向非满节点中插入键"""
        i = len(node.keys) - 1
        
        if node.leaf:
            # 叶节点：直接插入
            node.keys.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
        else:
            # 内部节点：找到正确子节点
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            
            if len(node.children[i].keys) == 2 * self.t - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            
            self._insert_non_full(node.children[i], key)
    
    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        print("  " * level + str(node.keys))
        for child in node.children:
            self.print_tree(child, level + 1)

# 测试
bt = BTree(t=2)  # 最小度数=2 → 每节点最多3个键
for x in [10, 20, 5, 6, 12, 30, 7, 17]:
    bt.insert(x)
bt.print_tree()
```

## 14.4 B+ 树

B+ 树是 B 树的变体，是数据库索引（InnoDB、PostgreSQL）的标准结构：

### B+ 树 vs B 树

| 特性 | B 树 | B+ 树 |
|------|------|-------|
| 数据存储 | 所有节点 | 仅叶节点 |
| 叶节点链接 | 否 | 是（双向链表） |
| 范围查询 | 需回溯 | 链表顺序扫描 |
| 内部节点 | 存数据，扇出小 | 只存键，扇出大 |
| 树高 | 略高 | 略低（因扇出更大）|

```
B+ 树结构：
         [15, 25]          ← 内部节点（只存键，不存数据）
        /    |    \
    [5,10] [15,20] [25,30]  ← 叶节点（存完整数据）
       ↔              ↔      ← 叶节点通过链表连接
```

### B+ 树的范围查询优势

```python
def range_query_bplus(leaf_start, lo, hi):
    """
    B+ 树范围查询：
    1. 找到包含 lo 的叶节点
    2. 沿链表向右扫描，直到 key > hi
    
    时间：O(log n + k)，k 是结果数量
    """
    results = []
    node = leaf_start
    
    while node:
        for key in node.keys:
            if lo <= key <= hi:
                results.append(key)
            elif key > hi:
                return results
        node = node.next  # 链表指针
    
    return results
```

## 14.5 为什么 B+ 树适合数据库

1. **磁盘页对齐**：节点大小 = 磁盘页大小（通常 4KB 或 16KB）
   - t = 200 时，树高 ≤ 4（10^12 条记录只需 4 次 I/O！）

2. **顺序访问优化**：叶节点链表支持高效全表扫描

3. **缓存友好**：内部节点只存键，可完全缓存在内存中

```python
# MySQL InnoDB 的参数
PAGE_SIZE = 16 * 1024  # 16KB
KEY_SIZE = 8            # bigint 键
POINTER_SIZE = 6        # 页指针
# 每个内部节点能存的键数:
fan_out = PAGE_SIZE // (KEY_SIZE + POINTER_SIZE)  # ≈ 1170
# 高度3的树能存:
capacity = 1170 * 1170 * 16  # ≈ 2000万行（叶节点每页约16条记录）
```

## 14.6 B 树变体总结

```
B 树家族：

B 树 (1970)
├── B+ 树 → 数据库索引（MySQL InnoDB, PostgreSQL）
├── B* 树 → 更高利用率（节点至少 2/3 满）
└── B-link 树 → 并发访问优化
```

## 小结

- B 树解决磁盘 I/O 瓶颈：增大扇出，降低树高
- B+ 树：数据只在叶节点，叶节点链表化
- 数据库索引首选 B+ 树（范围查询+顺序访问）
- B 树高度 O(log_t n)，t 可达数百，h ≤ 4 对应数十亿记录

## 练习

1. 计算：t=200 时，高度为 3 的 B+ 树最多存储多少条记录
2. 实现 B 树的删除操作（合并与借键）
3. 分析为什么哈希索引比 B+ 树更快但不支持范围查询

---

**上一章：** [红黑树](02-red-black.md) | **下一章（Part 5）：** [线段树](../part5-advanced/01-segment-tree.md)
