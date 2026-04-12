# 第9章：二叉搜索树（BST）概念与性质

## 9.1 BST 的定义

**二叉搜索树（Binary Search Tree）**满足以下性质（递归成立）：

> 对于任意节点 v：
> - v 的**左子树**中所有节点的值 **< v.val**
> - v 的**右子树**中所有节点的值 **> v.val**（严格不等，无重复值）

```
合法 BST:             非法 BST:
      8                    8
     / \                  / \
    3   10               3   10
   / \    \             / \    \
  1   6    14          1   6    14
     / \                  / \
    4   7                2   7  ← 2 < 3，但在 3 的右子树中！
```

## 9.2 核心性质：中序遍历有序

BST 的**中序遍历**（左→根→右）产生严格递增序列。这是 BST 最重要的性质：

```python
def inorder(root):
    if root is None:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

# BST: inorder([8,3,10,1,6,None,14]) = [1, 3, 4, 6, 7, 8, 10, 14]
# 严格递增 ✓
```

**推论**：
- BST 中序遍历第 k 个元素 = 第 k 小元素
- 验证 BST 等价于验证中序序列严格递增

## 9.3 验证 BST

### 方法一：中序遍历（直觉简单）

```python
def is_valid_bst_inorder(root):
    """验证 BST：中序遍历应严格递增"""
    prev = [float('-inf')]
    
    def inorder(node):
        if node is None:
            return True
        if not inorder(node.left):
            return False
        if node.val <= prev[0]:
            return False
        prev[0] = node.val
        return inorder(node.right)
    
    return inorder(root)
```

### 方法二：范围约束（推荐）

每个节点必须在 (min, max) 范围内：

```python
def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    """
    验证 BST：传递合法范围约束
    时间 O(n)，空间 O(h)
    """
    if root is None:
        return True
    
    # 当前节点必须在范围内
    if root.val <= min_val or root.val >= max_val:
        return False
    
    # 左子树：上界变为当前值
    # 右子树：下界变为当前值
    return is_valid_bst(root.left, min_val, root.val) and \
           is_valid_bst(root.right, root.val, max_val)

# 常见错误：只检查直接父子关系
# 这是错的！
def is_valid_bst_WRONG(root):
    if root is None:
        return True
    if root.left and root.left.val >= root.val:
        return False
    if root.right and root.right.val <= root.val:
        return False
    # 错误！没有检查更深层节点的范围约束
    return is_valid_bst_WRONG(root.left) and is_valid_bst_WRONG(root.right)
```

## 9.4 BST 的查找性质

在 BST 中查找值 x 时，每次比较后**搜索空间减半**（平衡树）：

```python
def search_bst(root, val):
    """BST 查找：O(h) 时间"""
    if root is None:
        return None
    if root.val == val:
        return root
    elif val < root.val:
        return search_bst(root.left, val)   # 向左
    else:
        return search_bst(root.right, val)  # 向右

# 迭代版（更高效，无递归开销）
def search_bst_iterative(root, val):
    curr = root
    while curr:
        if curr.val == val:
            return curr
        elif val < curr.val:
            curr = curr.left
        else:
            curr = curr.right
    return None
```

**时间复杂度**：
- 平衡 BST：O(log n)
- 退化 BST（链）：O(n)

## 9.5 前驱与后继

BST 中节点 v 的**中序前驱**（inorder predecessor）= 小于 v 的最大值
BST 中节点 v 的**中序后继**（inorder successor）= 大于 v 的最小值

```python
def find_predecessor(root, val):
    """
    找到 BST 中值 val 的中序前驱
    思路：
    - 若 val 有左子树，前驱 = 左子树最大值
    - 否则，前驱 = 最近的"向右拐"的祖先
    """
    predecessor = None
    curr = root
    
    while curr:
        if curr.val < val:
            predecessor = curr  # 候选前驱（比 val 小）
            curr = curr.right   # 继续找更大的
        else:
            curr = curr.left
    
    return predecessor

def find_successor(root, val):
    """
    找到 BST 中值 val 的中序后继
    """
    successor = None
    curr = root
    
    while curr:
        if curr.val > val:
            successor = curr  # 候选后继（比 val 大）
            curr = curr.left  # 继续找更小的
        else:
            curr = curr.right
    
    return successor

def find_min(root):
    """BST 最小值：一直向左"""
    while root.left:
        root = root.left
    return root

def find_max(root):
    """BST 最大值：一直向右"""
    while root.right:
        root = root.right
    return root
```

## 9.6 BST 的重要统计量

### 第 k 小元素

```python
def kth_smallest(root, k):
    """
    BST 第 k 小元素（LeetCode 230）
    中序遍历的第 k 个节点
    
    进阶：若频繁调用，在节点中存储左子树大小，O(h) 定位
    """
    stack = []
    curr = root
    count = 0
    
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        
        curr = stack.pop()
        count += 1
        
        if count == k:
            return curr.val
        
        curr = curr.right
    
    return -1
```

### 两数之和（BST 版本）

```python
def find_target_in_bst(root, k):
    """
    BST 中是否存在两节点之和为 k（LeetCode 653）
    利用 BST 的有序性：中序得到有序数组，双指针
    """
    nums = []
    
    def inorder(node):
        if node:
            inorder(node.left)
            nums.append(node.val)
            inorder(node.right)
    
    inorder(root)
    
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == k:
            return True
        elif s < k:
            left += 1
        else:
            right -= 1
    
    return False
```

## 9.7 平均情况分析

对于随机插入 n 个不同键的 BST：

| 操作 | 平均情况 | 最坏情况 |
|------|---------|---------|
| 查找 | O(log n) | O(n) |
| 插入 | O(log n) | O(n) |
| 删除 | O(log n) | O(n) |
| 中序遍历 | O(n) | O(n) |

> 最坏情况（O(n)）发生在插入有序序列时，树退化为链表。这是引入平衡树的动机。

## 小结

- BST：左子树 < 根 < 右子树（递归成立）
- 中序遍历 = 有序序列（最重要性质）
- 验证 BST 要传递范围约束，不能只看直接父子
- 平衡 BST 操作 O(log n)，退化时 O(n)

## 练习

1. 将有序数组转为高度最小的 BST
2. 找出 BST 中两个被错误交换的节点并恢复
3. 统计 BST 中满足 [L, R] 范围的节点值之和

---

**上一章（Part 2）：** [Morris 遍历](../part2-traversals/04-morris.md) | **下一章：** [BST 增删查改](02-bst-operations.md)
