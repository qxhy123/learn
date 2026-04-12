# 第2章：二叉树的基本术语与性质

## 2.1 核心术语

理解以下术语是学习二叉树的基础：

```
              A          ← 根节点 (Root)
            /   \
           B     C       ← A 的子节点
          / \     \
         D   E     F     ← 叶节点: D, E, F
```

| 术语 | 定义 | 示例（上图） |
|------|------|-------------|
| **根节点 Root** | 没有父节点的唯一节点 | A |
| **子节点 Child** | 某节点的直接下级 | B、C 是 A 的子节点 |
| **父节点 Parent** | 某节点的直接上级 | A 是 B、C 的父节点 |
| **叶节点 Leaf** | 没有子节点的节点 | D、E、F |
| **内部节点 Internal** | 至少有一个子节点的非根节点 | B、C |
| **兄弟节点 Sibling** | 有相同父节点的节点 | B 与 C 互为兄弟 |
| **祖先 Ancestor** | 从根到该节点路径上的所有节点 | A、B 是 D 的祖先 |
| **后代 Descendant** | 该节点子树中的所有节点 | B、D、E 是 A 的后代 |

## 2.2 深度、高度与层

这三个概念容易混淆，务必分清：

```
层(Level):
  Level 0:        A
  Level 1:      B   C
  Level 2:    D  E    F
```

### 节点的深度 (Depth)

**节点深度** = 从根节点到该节点的边数

```python
def node_depth(root, target, depth=0):
    """计算目标节点的深度"""
    if root is None:
        return -1  # 未找到
    if root.val == target:
        return depth
    
    left = node_depth(root.left, target, depth + 1)
    if left != -1:
        return left
    return node_depth(root.right, target, depth + 1)

# 根节点深度 = 0
# B、C 的深度 = 1
# D、E、F 的深度 = 2
```

### 节点的高度 (Height)

**节点高度** = 从该节点到其最深叶节点的边数

```python
def tree_height(root):
    """计算树（或子树）的高度"""
    if root is None:
        return -1  # 空树高度定义为 -1（也有定义为 0 的，注意区分）
    
    left_height = tree_height(root.left)
    right_height = tree_height(root.right)
    
    return 1 + max(left_height, right_height)

# 叶节点高度 = 0
# B 的高度 = 1（B→D 或 B→E，一条边）
# A（根）的高度 = 2（A→B→D）
```

> **常见混淆**：有些资料将空树高度定义为 0，叶节点高度为 1。本教程采用**空树=-1，叶节点=0**的约定，与大多数竞赛和面试题一致。

### 树的层 (Level)

**节点的层** = 深度 + 1（从 1 开始计数）

## 2.3 重要数学性质

### 性质一：节点数与层的关系

第 i 层（从 0 开始）**最多**有 $2^i$ 个节点：

```
Level 0: 最多 1  = 2^0 个节点
Level 1: 最多 2  = 2^1 个节点
Level 2: 最多 4  = 2^2 个节点
Level k: 最多 2^k 个节点
```

### 性质二：高度为 h 的树的节点数

- **最少**：h+1 个节点（退化为链表）
- **最多**：$2^{h+1} - 1$ 个节点（完全二叉树）

```python
# 验证：高度为 2 的满二叉树节点数
# 2^(2+1) - 1 = 7
#       1
#      / \
#     2   3
#    / \ / \
#   4  5 6  7
```

### 性质三：叶节点与双子节点的关系

设树中叶节点数为 $n_0$，有两个子节点的节点数为 $n_2$，则：

$$n_0 = n_2 + 1$$

```python
def count_nodes_by_type(root):
    """统计各类型节点数"""
    if root is None:
        return 0, 0, 0  # n0, n1, n2
    
    if root.left is None and root.right is None:
        return 1, 0, 0  # 叶节点
    
    l0, l1, l2 = count_nodes_by_type(root.left)
    r0, r1, r2 = count_nodes_by_type(root.right)
    
    n0 = l0 + r0
    n1 = l1 + r1
    n2 = l2 + r2
    
    if root.left and root.right:
        n2 += 1  # 当前节点有两个子节点
    elif root.left or root.right:
        n1 += 1  # 当前节点只有一个子节点
    
    return n0, n1, n2

# 验证：n0 == n2 + 1
```

### 性质四：n 个节点的完全二叉树高度

$$h = \lfloor \log_2 n \rfloor$$

这是二叉搜索树和堆操作 O(log n) 的理论基础。

## 2.4 路径与子树

### 路径 (Path)

路径是从某节点到其后代节点经过的**边的序列**（也可定义为节点序列）。

```python
def find_path(root, target):
    """找到从根到目标节点的路径"""
    if root is None:
        return None
    if root.val == target:
        return [root.val]
    
    # 向左子树搜索
    left_path = find_path(root.left, target)
    if left_path is not None:
        return [root.val] + left_path
    
    # 向右子树搜索
    right_path = find_path(root.right, target)
    if right_path is not None:
        return [root.val] + right_path
    
    return None
```

### 子树 (Subtree)

以节点 v 为根的子树包含 v 及其所有后代节点。

```python
def count_nodes(root):
    """计算子树节点总数"""
    if root is None:
        return 0
    return 1 + count_nodes(root.left) + count_nodes(root.right)

def is_subtree(main_root, sub_root):
    """判断 sub_root 是否是 main_root 的子树"""
    if sub_root is None:
        return True
    if main_root is None:
        return False
    if is_same_tree(main_root, sub_root):
        return True
    return is_subtree(main_root.left, sub_root) or \
           is_subtree(main_root.right, sub_root)

def is_same_tree(p, q):
    """判断两棵树是否完全相同"""
    if p is None and q is None:
        return True
    if p is None or q is None:
        return False
    return p.val == q.val and \
           is_same_tree(p.left, q.left) and \
           is_same_tree(p.right, q.right)
```

## 2.5 树的直径

**直径**是树中任意两节点间最长路径的长度（边数）。

```python
def diameter_of_binary_tree(root):
    """
    计算二叉树的直径
    关键洞察：最长路径不一定经过根节点
    """
    max_diameter = [0]
    
    def height(node):
        if node is None:
            return -1
        
        left_h = height(node.left)
        right_h = height(node.right)
        
        # 经过当前节点的路径长度
        current_diameter = left_h + right_h + 2
        max_diameter[0] = max(max_diameter[0], current_diameter)
        
        return 1 + max(left_h, right_h)
    
    height(root)
    return max_diameter[0]

# 示例：
#       1
#      / \
#     2   3
#    / \
#   4   5
# 直径 = 3（路径 4-2-1-3 或 5-2-1-3）
```

## 2.6 平衡性概念

**平衡二叉树**：任意节点的左右子树高度差不超过 1。

```python
def is_balanced(root):
    """
    判断是否为高度平衡的二叉树
    返回 -1 表示不平衡，否则返回树高
    """
    def check(node):
        if node is None:
            return 0  # 空树高度为 0（这里用不同约定）
        
        left_h = check(node.left)
        if left_h == -1:
            return -1
        
        right_h = check(node.right)
        if right_h == -1:
            return -1
        
        if abs(left_h - right_h) > 1:
            return -1  # 不平衡
        
        return 1 + max(left_h, right_h)
    
    return check(root) != -1
```

## 小结

| 概念 | 关键点 |
|------|--------|
| 深度 | 从根到节点的边数，根深度=0 |
| 高度 | 从节点到最深叶的边数，叶高度=0 |
| $n_0 = n_2 + 1$ | 叶节点数=双子节点数+1 |
| 直径 | 不一定经过根节点 |
| 平衡 | 任意节点左右子树高度差≤1 |

## 练习

1. 编写函数计算树中所有节点的深度之和
2. 验证性质 $n_0 = n_2 + 1$ 在你构建的树上成立
3. 实现一个函数，返回从根到所有叶节点的所有路径

---

**上一章：** [什么是二叉树](01-introduction.md) | **下一章：** [二叉树的分类](03-types.md)
