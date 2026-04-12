# 第3章：二叉树的分类

## 3.1 按结构分类

### 满二叉树 (Full Binary Tree)

每个节点**要么是叶节点，要么有两个子节点**（没有只有一个子节点的情况）。

```
满二叉树（✓）:          非满二叉树（✗）:
        1                      1
       / \                    / \
      2   3                  2   3
     / \                    /
    4   5                  4
```

```python
def is_full_binary_tree(root):
    """判断是否为满二叉树"""
    if root is None:
        return True
    # 叶节点：合法
    if root.left is None and root.right is None:
        return True
    # 只有一个子节点：非满
    if root.left is None or root.right is None:
        return False
    # 两个子节点都有：递归检查
    return is_full_binary_tree(root.left) and \
           is_full_binary_tree(root.right)
```

### 完全二叉树 (Complete Binary Tree)

除最后一层外，每层节点数达到最大值；最后一层节点**从左向右连续填充**。

```
完全二叉树（✓）:         非完全二叉树（✗）:
        1                      1
       / \                    / \
      2   3                  2   3
     / \ /                    \   \
    4  5 6                     5   6
```

```python
from collections import deque

def is_complete_binary_tree(root):
    """
    判断是否为完全二叉树
    思路：BFS 层序遍历，一旦遇到空节点，后续不应有非空节点
    """
    if root is None:
        return True
    
    queue = deque([root])
    reached_end = False  # 是否已遇到空节点
    
    while queue:
        node = queue.popleft()
        
        if node is None:
            reached_end = True
        else:
            if reached_end:
                return False  # 空节点后出现非空节点
            queue.append(node.left)
            queue.append(node.right)
    
    return True
```

**重要性质**：完全二叉树可以用数组高效存储（见第4章）。

### 完美二叉树 (Perfect Binary Tree)

每层节点数都达到最大值，即高度为 h 的完美二叉树有 $2^{h+1}-1$ 个节点。

```
完美二叉树（高度=2）:
        1
       / \
      2   3
     / \ / \
    4  5 6  7
```

```python
def is_perfect_binary_tree(root, height=None, depth=0):
    """判断是否为完美二叉树"""
    if root is None:
        return True
    
    # 首次调用，计算树高
    if height is None:
        h = root
        height = 0
        while h.left:
            height += 1
            h = h.left
    
    # 叶节点深度应等于树高
    if root.left is None and root.right is None:
        return depth == height
    
    if root.left is None or root.right is None:
        return False
    
    return is_perfect_binary_tree(root.left, height, depth + 1) and \
           is_perfect_binary_tree(root.right, height, depth + 1)
```

### 退化树 (Degenerate Tree) / 斜树

每个节点只有一个子节点，退化为链表。这是二叉搜索树性能最差的情况。

```
左斜树：       右斜树：
    1               1
   /                 \
  2                   2
 /                     \
3                       3
```

## 3.2 按用途分类

### 二叉搜索树 (BST - Binary Search Tree)

满足：左子树所有值 < 根值 < 右子树所有值（递归成立）。

```
合法 BST:          非法 BST（3 在 5 右边，但 3 < 5）:
      5                    5
     / \                  / \
    3   7                3   7
   / \ / \                  /
  2  4 6  8                3
```

### 平衡二叉搜索树 (Balanced BST)

在 BST 基础上保证树高为 O(log n)：
- AVL 树（严格平衡）
- 红黑树（近似平衡，用于 C++ STL map/set）

### 堆 (Heap)

完全二叉树 + 堆序性质：
- **最大堆**：父节点值 ≥ 子节点值
- **最小堆**：父节点值 ≤ 子节点值

```
最大堆:
        9
       / \
      7   8
     / \ /
    4  5 6
```

### 线段树 (Segment Tree)

用于区间查询和更新，每个节点存储区间的聚合信息（见 Part5）。

### 字典树 (Trie)

多叉树（特殊情况下是二叉树），用于字符串的前缀匹配（见 Part5）。

## 3.3 关系图

```
二叉树
├── 满二叉树（每节点0或2个子节点）
│   └── 完美二叉树（所有叶在同一层）
├── 完全二叉树（最后层从左填充）
│   └── 完美二叉树（特殊情况）
├── 退化树（链表）
└── 按用途
    ├── 二叉搜索树 (BST)
    │   ├── AVL 树
    │   └── 红黑树
    ├── 堆
    ├── 线段树
    └── Trie
```

## 3.4 各类型性质对比

| 类型 | 高度范围 | 节点数（高度h） | 典型用途 |
|------|---------|----------------|---------|
| 完美二叉树 | 确定 | $2^{h+1}-1$ | 理论分析 |
| 完全二叉树 | O(log n) | $[2^h, 2^{h+1}-1]$ | 堆、数组存储 |
| BST（平衡） | O(log n) | 任意 | 有序字典 |
| BST（退化） | O(n) | 任意 | （最差情况） |
| 堆 | O(log n) | 任意 | 优先队列 |

## 3.5 判断完全二叉树的节点数（O(log²n) 优化）

利用完全二叉树的性质，可以比 O(n) 更快地计算节点数：

```python
def count_nodes_complete(root):
    """
    利用完全二叉树性质，O(log²n) 计算节点数
    
    思路：
    - 如果左子树高度 == 右子树高度，则左子树是完美二叉树
    - 否则右子树是完美二叉树（高度比左子树小1）
    """
    if root is None:
        return 0
    
    def get_height(node, go_left):
        h = 0
        while node:
            h += 1
            node = node.left if go_left else node.right
        return h
    
    left_h = get_height(root, True)   # 最左路径高度
    right_h = get_height(root, False)  # 最右路径高度
    
    if left_h == right_h:
        # 完美二叉树
        return (1 << left_h) - 1  # 2^h - 1
    else:
        # 递归计算
        return 1 + count_nodes_complete(root.left) + \
                   count_nodes_complete(root.right)
```

## 小结

| 类型 | 核心条件 |
|------|---------|
| 满二叉树 | 每节点 0 或 2 个子节点 |
| 完全二叉树 | 最后层从左连续 |
| 完美二叉树 | 所有叶节点同层 |
| BST | 左 < 根 < 右（递归） |

## 练习

1. 判断一棵树是否同时是满二叉树和完全二叉树
2. 给定节点数 n，计算完全二叉树的最小高度
3. 实现一个函数，判断给定数组（层序）是否表示一个合法的最大堆

---

**上一章：** [基本术语](02-terminology.md) | **下一章：** [存储与表示方法](04-representation.md)
