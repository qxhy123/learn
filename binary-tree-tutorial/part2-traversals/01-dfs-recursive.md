# 第5章：深度优先遍历（递归实现）

## 5.1 三种 DFS 遍历

深度优先遍历（DFS）按照**根节点的访问顺序**分三种：

```
        1
       / \
      2   3
     / \
    4   5
```

| 遍历方式 | 顺序 | 访问序列 | 记忆口诀 |
|---------|------|---------|---------|
| 前序 Preorder | 根→左→右 | 1,2,4,5,3 | **根**先行 |
| 中序 Inorder | 左→根→右 | 4,2,5,1,3 | 根**中**间 |
| 后序 Postorder | 左→右→根 | 4,5,2,3,1 | 根**后**到 |

## 5.2 递归实现（三合一模板）

递归实现极其简洁，只需调整访问根节点的时机：

```python
def preorder(root):
    """前序遍历：根 → 左 → 右"""
    if root is None:
        return []
    result = [root.val]                    # 先访问根
    result.extend(preorder(root.left))
    result.extend(preorder(root.right))
    return result

def inorder(root):
    """中序遍历：左 → 根 → 右"""
    if root is None:
        return []
    result = []
    result.extend(inorder(root.left))
    result.append(root.val)                # 中间访问根
    result.extend(inorder(root.right))
    return result

def postorder(root):
    """后序遍历：左 → 右 → 根"""
    if root is None:
        return []
    result = []
    result.extend(postorder(root.left))
    result.extend(postorder(root.right))
    result.append(root.val)                # 最后访问根
    return result
```

### 通用回调模板

```python
def dfs(root, visit):
    """通用 DFS 框架，visit 是对节点执行操作的函数"""
    if root is None:
        return
    # 前序位置：visit(root)
    dfs(root.left, visit)
    # 中序位置：visit(root)
    dfs(root.right, visit)
    # 后序位置：visit(root)
```

## 5.3 前中后序的直觉理解

用一个形象的比喻：把树的轮廓用线标出，绕树走一圈：

```
每个节点被经过3次：
1. 第一次经过（来时）→ 前序
2. 第二次经过（从左子树返回）→ 中序
3. 第三次经过（从右子树返回）→ 后序
```

## 5.4 实际应用场景

### 前序遍历的应用

**1. 复制二叉树**

```python
def copy_tree(root):
    """深拷贝二叉树"""
    if root is None:
        return None
    # 前序：先创建根，再递归复制子树
    new_node = TreeNode(root.val)
    new_node.left = copy_tree(root.left)
    new_node.right = copy_tree(root.right)
    return new_node
```

**2. 打印目录结构**

```python
def print_directory(root, prefix="", is_last=True):
    """前序遍历打印树形目录"""
    if root is None:
        return
    connector = "└── " if is_last else "├── "
    print(prefix + connector + str(root.val))
    
    children = []
    if root.left:
        children.append((root.left, False))
    if root.right:
        children.append((root.right, True))
    
    for i, (child, last) in enumerate(children):
        extension = "    " if is_last else "│   "
        print_directory(child, prefix + extension, last)
```

### 中序遍历的应用

**BST 中序遍历 = 有序序列（这是 BST 最重要的性质之一）**

```python
def bst_to_sorted_list(root):
    """中序遍历 BST 得到有序列表"""
    result = []
    
    def inorder(node):
        if node is None:
            return
        inorder(node.left)
        result.append(node.val)
        inorder(node.right)
    
    inorder(root)
    return result

def kth_smallest_in_bst(root, k):
    """BST 中第 k 小的元素（中序遍历的第 k 个）"""
    count = [0]
    result = [None]
    
    def inorder(node):
        if node is None or result[0] is not None:
            return
        inorder(node.left)
        count[0] += 1
        if count[0] == k:
            result[0] = node.val
            return
        inorder(node.right)
    
    inorder(root)
    return result[0]
```

### 后序遍历的应用

**后序：子树结果先计算，再用于父节点 → 自底向上**

```python
def calculate_subtree_sum(root):
    """
    计算每个子树的节点值之和，并赋值给该节点
    后序：先计算子树的和，再更新当前节点
    """
    if root is None:
        return 0
    left_sum = calculate_subtree_sum(root.left)
    right_sum = calculate_subtree_sum(root.right)
    total = root.val + left_sum + right_sum
    root.val = total  # 用子树总和替换原值
    return total

def delete_tree(root):
    """
    释放二叉树内存
    后序：先删子节点，再删自己（C++ 场景）
    """
    if root is None:
        return
    delete_tree(root.left)
    delete_tree(root.right)
    # del root（Python 中由 GC 处理）
    root.left = None
    root.right = None
```

## 5.5 经典递归问题

### 问题1：树的最大深度

```python
def max_depth(root):
    if root is None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

### 问题2：路径总和

```python
def has_path_sum(root, target_sum):
    """是否存在根到叶的路径，路径和等于 target_sum"""
    if root is None:
        return False
    # 到达叶节点
    if root.left is None and root.right is None:
        return root.val == target_sum
    # 递归检查左右子树，目标减去当前值
    remaining = target_sum - root.val
    return has_path_sum(root.left, remaining) or \
           has_path_sum(root.right, remaining)
```

### 问题3：翻转二叉树

```python
def invert_tree(root):
    """翻转二叉树（镜像）"""
    if root is None:
        return None
    # 后序：先翻转子树，再交换
    root.left = invert_tree(root.left)
    root.right = invert_tree(root.right)
    root.left, root.right = root.right, root.left
    return root
```

### 问题4：最大路径和（LeetCode 124，Hard）

```python
def max_path_sum(root):
    """
    二叉树中任意路径（不必经过根）的最大节点值之和
    
    关键：每个节点可以作为路径的"拐点"
    - 向上贡献：max(0, val + max(left_gain, right_gain))
    - 本地路径：val + left_gain + right_gain
    """
    max_sum = [float('-inf')]
    
    def gain(node):
        if node is None:
            return 0
        
        left_gain = max(0, gain(node.left))   # 负增益舍弃
        right_gain = max(0, gain(node.right))
        
        # 经过当前节点的最大路径
        price = node.val + left_gain + right_gain
        max_sum[0] = max(max_sum[0], price)
        
        # 向父节点贡献：只能选一侧
        return node.val + max(left_gain, right_gain)
    
    gain(root)
    return max_sum[0]
```

## 5.6 递归的时间与空间复杂度

对于高度为 h、节点数为 n 的二叉树：

| 复杂度 | 说明 |
|-------|------|
| **时间** O(n) | 每个节点恰好被访问一次 |
| **空间** O(h) | 递归栈深度 = 树高 |
| 平衡树空间 | O(log n) |
| 退化树空间 | O(n)（可能栈溢出！） |

> 对于极深的树（n > 10^4），递归可能导致**栈溢出**，应使用迭代实现（见下一章）。

## 小结

- 三种 DFS 遍历只差根节点的访问时机
- 前序适合"自顶向下"，后序适合"自底向上"
- BST 中序遍历 = 有序序列（极重要！）
- 递归空间复杂度 O(h)，极深树需改用迭代

## 练习

1. 实现"返回所有根到叶路径"的函数
2. 用后序遍历计算树中节点个数
3. 不用额外空间，判断两棵树是否对称（镜像）

---

**上一章（Part 1）：** [存储与表示](../part1-foundations/04-representation.md) | **下一章：** [深度优先遍历（迭代）](02-dfs-iterative.md)
