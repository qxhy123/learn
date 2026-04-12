# 第1章：什么是二叉树

## 1.1 从现实世界出发

在解决问题时，我们经常需要组织具有**层级关系**的数据：

- 公司的组织架构：CEO → 部门总监 → 经理 → 员工
- 文件系统：根目录 → 子目录 → 文件
- 决策过程：每个问题有两个选择（是/否）

当每个节点**最多有两个子节点**时，这种层级结构就是**二叉树**。

```
        CEO
       /   \
     CTO   CFO
    /   \     \
  Dev1  Dev2  Finance
```

## 1.2 形式化定义

**二叉树**是满足以下条件的数据结构：

1. 由**节点（Node）**组成
2. 有一个特殊的**根节点（Root）**（或为空树）
3. 每个节点**最多**有两个子节点，分别称为**左子节点**和**右子节点**
4. 每个子节点本身也是一棵二叉树（递归定义）

> 关键词：**最多两个子节点**。这与"多叉树"（任意数量子节点）区分开来。

## 1.3 节点的构成

一个二叉树节点包含三部分：

```
    ┌─────────────┐
    │    value    │  ← 存储的数据
    ├──────┬──────┤
    │ left │right │  ← 指向左/右子树的指针
    └──────┴──────┘
```

### Python 实现

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left    # 左子节点（TreeNode 或 None）
        self.right = right  # 右子节点（TreeNode 或 None）
```

### C++ 实现

```cpp
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};
```

## 1.4 构建第一棵二叉树

```
        1
       / \
      2   3
     / \
    4   5
```

```python
# 方法一：手动链接节点
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

# 方法二：从列表构建（层序方式，None 表示空节点）
from collections import deque

def build_tree(values):
    """从层序列表构建二叉树，None 表示空节点"""
    if not values or values[0] is None:
        return None
    
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    
    while queue and i < len(values):
        node = queue.popleft()
        
        # 处理左子节点
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        
        # 处理右子节点
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    
    return root

# 使用示例
root = build_tree([1, 2, 3, 4, 5])
# 构建出的树：
#       1
#      / \
#     2   3
#    / \
#   4   5
```

## 1.5 为什么二叉树如此重要

| 特性 | 说明 |
|------|------|
| **高效查找** | 二叉搜索树中查找 O(log n) |
| **自然递归** | 树的结构天然适合递归处理 |
| **表达能力强** | 可以表示各种层级关系 |
| **算法基础** | 堆、Trie、线段树等均基于二叉树 |

## 1.6 空树与单节点树

两个特殊情况需要注意：

```python
# 空树
empty_tree = None

# 单节点树（只有根节点）
single_node = TreeNode(42)
# single_node.left == None
# single_node.right == None
```

在算法实现中，**处理空树（None）是递归的终止条件**，必须首先判断。

## 1.7 打印树结构（调试利器）

```python
def print_tree(root, level=0, prefix="Root: "):
    """可视化打印二叉树结构"""
    if root is None:
        return
    print(" " * (level * 4) + prefix + str(root.val))
    if root.left or root.right:
        if root.left:
            print_tree(root.left, level + 1, "L--- ")
        else:
            print(" " * ((level + 1) * 4) + "L--- None")
        if root.right:
            print_tree(root.right, level + 1, "R--- ")
        else:
            print(" " * ((level + 1) * 4) + "R--- None")

# 测试
root = build_tree([1, 2, 3, 4, 5])
print_tree(root)
# 输出：
# Root: 1
#     L--- 2
#         L--- 4
#         R--- 5
#     R--- 3
```

## 小结

- 二叉树是每个节点最多有两个子节点的树形结构
- 节点由值、左指针、右指针组成
- 空树（None）是合法的二叉树
- 递归是处理二叉树最自然的方式

## 练习

1. 手动创建一棵 7 个节点的完全二叉树
2. 实现 `build_tree` 函数的反向操作：将二叉树转回层序列表
3. 计算一棵树中节点的总数（提示：用递归）

---

**下一章：** [二叉树的基本术语与性质](02-terminology.md)
